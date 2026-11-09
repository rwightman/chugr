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
