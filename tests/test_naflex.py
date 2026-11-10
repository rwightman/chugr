import os

import pytest
import torch

from chugr.naflex import LengthBucketer, NaFlexBatchScheduler, NaFlexBatcher


def test_scheduler_deterministic_and_budget():
    kw = dict(seq_lens=(256, 512, 1024), max_tokens_per_batch=8192, num_samples=1000, per_row_text_tokens=64,
              batch_divisor=4, seed=3, world_size=2)
    a, b = NaFlexBatchScheduler(**kw), NaFlexBatchScheduler(**kw)
    assert a.interval_schedule(5, num_workers=3) == b.interval_schedule(5, num_workers=3)  # rank identical
    assert a.interval_schedule(5, num_workers=3) != a.interval_schedule(6, num_workers=3)  # reshuffled per interval
    for seq_len, bs in a.interval_schedule(0):
        assert bs % 4 == 0 or bs < 4
        assert bs * (seq_len + 64) <= 8192
    assert a.num_batches_for_workers(3) % 3 == 0
    assert a.num_samples_for_workers(1) >= 1000  # per rank ceil(1000 / 2), both ranks


def _fake_processor(seq_len, fail_every=0):
    count = {'n': 0}

    def fn(sample):
        count['n'] += 1
        if fail_every and count['n'] % fail_every == 0:
            raise RuntimeError('bad sample')
        n = min(seq_len, sample['n'])
        return {
            'image_input': {
                'patches': torch.ones(n, 4),
                'patch_coord': torch.zeros(n, 2, dtype=torch.long),
                'patch_valid': torch.ones(n, dtype=torch.bool),
            },
            'text_input': torch.tensor([sample['i']]),
        }
    return fn


def test_batcher_pads_to_bucket_and_replaces_failures():
    sched = NaFlexBatchScheduler(seq_lens=(16, 32), max_tokens_per_batch=128, num_samples=64, batch_divisor=1)
    procs = {16: _fake_processor(16, fail_every=5), 32: _fake_processor(32, fail_every=5)}
    batcher = NaFlexBatcher(sched, procs, handler=lambda ex: True)
    src = ({'i': i, 'n': 10 + i % 20} for i in range(10_000))
    batches = list(batcher(src))
    assert [(b['image_input']['patches'].shape[1], b['image_input']['patches'].shape[0]) for b in batches] == \
        sched.interval_schedule(0)
    for b in batches:
        assert b['image_input']['patch_valid'].shape == b['image_input']['patches'].shape[:2]


def test_length_bucketer_reorders_only():
    samples = [{'json': {'pages': [{'text': ['x' * (i % 37)]}]}, 'i': i} for i in range(300)]
    out = list(LengthBucketer(pool=100, chunk=10)(iter(samples)))
    assert sorted(s['i'] for s in out) == list(range(300))


IDL = '/home/wiggs/work/pp/data/idl/idl-train-0000{0..3}.tar'


@pytest.mark.skipif(not os.path.exists('/home/wiggs/work/pp/data/idl/idl-train-00000.tar'), reason='local IDL shards')
@pytest.mark.parametrize('resampled', [True, False])
def test_naflex_doc_read_loader(resampled):
    import functools
    import chugr
    from chugr.common import DataCfg, ImageInputCfg

    def image_factory(seq_len):
        cfg = ImageInputCfg(size=None, mode='L', transform_type='doc_naflex', patch_size=16, max_seq_len=seq_len)
        return chugr.create_image_preprocessor(cfg, is_training=True)

    task_cfg = chugr.DataTaskDocReadCfg(
        image_process_fn=image_factory(512),
        image_process_factory=image_factory,
        text_process_fn=chugr.create_text_preprocessor('facebook/bart-base', max_length=256),
        error_handler='log_and_continue',
    )
    data_cfg = DataCfg(source=IDL, num_samples=64, num_workers=2, resampled=resampled,
                       naflex_seq_lens=[256, 512], naflex_max_tokens=4096, naflex_batch_divisor=1, length_bucketing=True)
    lb = chugr.create_loader(data_cfg, task_cfg, is_training=True)
    lb.set_interval(0)
    batches = list(lb)
    assert len(batches) == lb.num_batches
    for b in batches:
        p = b['image_input']['patches']
        assert p.shape[1] in (256, 512) and p.shape[0] * p.shape[1] <= 4096
        assert b['text_input'].shape[0] == p.shape[0]


def test_repeating_shard_list():
    from chugr.wds.shardlists import ShuffledShardList
    it = iter(ShuffledShardList([f's{i}' for i in range(5)], seed=1, interval=0, repeat=True))
    urls = [next(it)['url'] for _ in range(15)]
    assert sorted(urls[:5]) == sorted(urls[5:10]) == sorted(urls[10:]) == [f's{i}' for i in range(5)]
    assert urls[:5] != urls[5:10]  # each pass reshuffled


class _ListDataset:
    """ map-style dataset of raw samples, optionally failing to load some indices """

    def __init__(self, n, bad=()):
        self.n, self.bad = n, set(bad)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if idx in self.bad:
            raise RuntimeError('bad sample')
        return {'i': idx, 'n': 10 + idx % 20}


class _MapProcessor:
    """ picklable (dataloader workers) fake processor """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, sample):
        return _fake_processor(self.seq_len)(sample)


def test_shuffled_interval_indices_continue_across_intervals():
    from chugr.common import shuffled_interval_indices
    from chugr.hfds.loader import IntervalSampler
    n, per = 100, 30
    stream = [i for interval in range(7) for i in shuffled_interval_indices(n, per, interval, seed=5)]
    # each pass over the dataset is a full permutation, short intervals don't skip samples
    assert sorted(stream[:100]) == sorted(stream[100:200]) == list(range(100))
    assert stream[:100] != stream[100:200]
    # longer than the dataset, and the sampler draws the same rank shares
    assert sorted(shuffled_interval_indices(n, 250, 0, seed=5)[:200]) == sorted(list(range(100)) * 2)
    sampler = IntervalSampler(range(n), num_samples=per, rank=1, world_size=2, seed=5, interval=3)
    assert list(sampler) == shuffled_interval_indices(n, per, 3, seed=5)[1::2]


def test_map_dataset_schedule_and_rank_split():
    from chugr.common import shuffled_interval_indices
    from chugr.naflex import NaFlexMapDataset
    n, world = 200, 2
    sched = NaFlexBatchScheduler(
        seq_lens=(16, 32), max_tokens_per_batch=128, num_samples=64, batch_divisor=1, seed=7, world_size=world)
    procs = {16: _fake_processor(16), 32: _fake_processor(32)}
    seen = []
    for interval in range(3):
        per_rank = []
        for rank in range(world):
            ds = NaFlexMapDataset(
                _ListDataset(n), sched, procs, interval=interval, rank=rank, world_size=world, seed=7)
            batches = list(ds)
            # rank identical (seq_len, batch_size) schedule
            assert [(b['image_input']['patches'].shape[1], b['image_input']['patches'].shape[0]) for b in batches] == \
                sched.interval_schedule(interval)
            per_rank.append([int(i) for b in batches for i in b['text_input'].flatten()])
        total = sum(len(r) for r in per_rank)
        expected = shuffled_interval_indices(n, total, interval, seed=7)
        assert per_rank == [expected[r::world] for r in range(world)]
        assert not set(per_rank[0]) & set(per_rank[1])
        seen += expected
    assert len(set(seen)) == len(seen) <= n  # no repeats until the dataset is exhausted


def test_map_dataset_replaces_failures():
    from chugr.naflex import NaFlexMapDataset
    bad = set(range(0, 100, 3))
    sched = NaFlexBatchScheduler(seq_lens=(16, 32), max_tokens_per_batch=128, num_samples=64, batch_divisor=1)
    procs = {16: _fake_processor(16), 32: _fake_processor(32)}
    batches = list(NaFlexMapDataset(_ListDataset(100, bad=bad), sched, procs, handler=lambda ex: True))
    assert [(b['image_input']['patches'].shape[1], b['image_input']['patches'].shape[0]) for b in batches] == \
        sched.interval_schedule(0)  # all batches full
    assert not bad & {int(i) for b in batches for i in b['text_input'].flatten()}

    def reraise(ex):
        raise ex
    with pytest.raises(RuntimeError):
        list(NaFlexMapDataset(_ListDataset(100, bad=bad), sched, procs, handler=reraise))


def test_map_dataset_workers_follow_schedule():
    from torch.utils.data import DataLoader
    from chugr.naflex import NaFlexMapDataset
    sched = NaFlexBatchScheduler(seq_lens=(16, 32), max_tokens_per_batch=128, num_samples=70, batch_divisor=1, seed=1)
    procs = {16: _MapProcessor(16), 32: _MapProcessor(32)}
    ds = NaFlexMapDataset(_ListDataset(500), sched, procs, interval=2, seed=1)
    batches = list(DataLoader(ds, batch_size=None, num_workers=3))
    assert len(batches) == sched.num_batches_for_workers(3)
    # workers interleave so loader batch i is schedule entry i
    assert [(b['image_input']['patches'].shape[1], b['image_input']['patches'].shape[0]) for b in batches] == \
        sched.interval_schedule(2, num_workers=3)
    ids = [int(i) for b in batches for i in b['text_input'].flatten()]
    assert len(set(ids)) == len(ids) == sched.num_samples_for_workers(3)


def test_safe_dataset_wraps():
    from chugr.hfds import SafeDataset
    ds = SafeDataset(_ListDataset(10, bad={8, 9}))
    assert ds[9]['i'] == 0 and ds[8]['i'] == 0 and ds[3]['i'] == 3


class _FakeStream:
    """ stand-in for a streaming HF dataset (finite pass, set_epoch) """

    def __init__(self, n):
        self.n, self.epoch = n, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter([{'i': i, 'n': 10 + i % 20, 'epoch': self.epoch} for i in range(self.n)])


def test_wrapped_iterable_repeat_quota_and_stages():
    from chugr.hfds.wrappers import WrappedIterableDataset
    ds = WrappedIterableDataset(_FakeStream(5), interval_count=3, num_samples=12, repeat=True)
    out = list(ds)
    assert [s['i'] for s in out] == [0, 1, 2, 3, 4] * 2 + [0, 1]  # wraps to fill the quota
    assert {s['epoch'] for s in out} == {3}
    assert len(list(WrappedIterableDataset(_FakeStream(5)))) == 5  # single pass by default
    with pytest.raises(RuntimeError):
        list(WrappedIterableDataset(_FakeStream(0), num_samples=4, repeat=True))
    # NaFlex batching of the repeating stream follows the schedule
    sched = NaFlexBatchScheduler(seq_lens=(16, 32), max_tokens_per_batch=128, num_samples=64, batch_divisor=1)
    batcher = NaFlexBatcher(sched, {16: _fake_processor(16), 32: _fake_processor(32)})
    batches = list(WrappedIterableDataset(_FakeStream(7), repeat=True, stages=[batcher]))
    assert [(b['image_input']['patches'].shape[1], b['image_input']['patches'].shape[0]) for b in batches] == \
        sched.interval_schedule(0)
