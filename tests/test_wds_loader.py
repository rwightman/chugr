""" Rank / worker splitting of the webdataset loader (simulated ranks, local IDL shards). """
import os

import pytest
import webdataset as wds

from chugr.common import DistributedCfg, rank_seed
from chugr.wds.loader import create_loader_wds

IDL_DIR = '/home/wiggs/work/pp/data/idl'
TRAIN = IDL_DIR + '/idl-train-000{00..07}.tar'
EVAL_ONE = IDL_DIR + '/idl-train-00000.tar'
EVAL_FOUR = IDL_DIR + '/idl-train-0000{0..3}.tar'
pytestmark = pytest.mark.skipif(not os.path.exists(IDL_DIR + '/idl-train-00007.tar'), reason='local IDL shards')


def _identity(sample):
    return {'identity': sample['__url__'].split('/')[-1] + '::' + sample['__key__']}


def _train(rank, world=2, seed=42, workers=2, resampled=True, n=16):
    bundle = create_loader_wds(
        TRAIN, task_pipeline=[wds.map(_identity)], is_training=True, num_samples=n * world, batch_size=2,
        num_workers=workers, persistent_workers=False, resampled=resampled, seed=seed,
        distributed=DistributedCfg(world_size=world, global_rank=rank, local_rank=rank))
    return [k for batch in bundle.loader for k in batch['identity']]


def _eval(source, rank, world, workers, n=None, bs=4):
    bundle = create_loader_wds(
        source, task_pipeline=[wds.map(_identity)], is_training=False, num_samples=n, batch_size=bs,
        num_workers=workers, persistent_workers=False,
        distributed=DistributedCfg(world_size=world, global_rank=rank, local_rank=rank))
    return [k for batch in bundle.loader for k in batch['identity']]


def test_rank_seed():
    assert rank_seed(42, 0) == 42
    seeds = {rank_seed(s, r) for s in range(40, 50) for r in range(64)}
    assert len(seeds) == 10 * 64  # no (seed, rank) collisions for nearby seeds


@pytest.mark.parametrize('workers', [0, 2])
def test_resampled_streams_differ_across_ranks_and_seeds(workers):
    a, b = _train(0, workers=workers), _train(1, workers=workers)
    assert len(set(a) & set(b)) < len(a) // 2
    assert _train(1, seed=42, workers=workers) != _train(0, seed=43, workers=workers)


def test_train_shards_split_by_explicit_rank(monkeypatch):
    # no RANK / WORLD_SIZE env (e.g. SLURM launch), the split must come from DistributedCfg
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    shards = [{k.split('::')[0] for k in _train(r, resampled=False)} for r in range(2)]
    assert shards[0] and shards[1] and not shards[0] & shards[1]


@pytest.mark.parametrize('world,workers', [(1, 0), (2, 0), (3, 2), (2, 3)])
def test_eval_sample_cap_exact(world, workers):
    ref = _eval(EVAL_ONE, 0, 1, 0, n=50)
    assert len(ref) == len(set(ref)) == 50
    keys = [k for r in range(world) for k in _eval(EVAL_ONE, r, world, workers, n=50)]
    assert sorted(keys) == sorted(ref)  # same eval set, each sample once, for any world size / worker count


@pytest.mark.parametrize('source,workers', [(EVAL_FOUR, 2), (EVAL_ONE, 2)])
def test_eval_full_split(source, workers):
    # all samples, split by shard (4 shards >= 2 ranks * 2 workers) or by sample (one shard)
    ref = _eval(source, 0, 1, 0)
    per_rank = [_eval(source, r, 2, workers) for r in range(2)]
    assert sorted(per_rank[0] + per_rank[1]) == sorted(ref) and len(set(ref)) == len(ref)
    if source == EVAL_FOUR:
        assert not {k.split('::')[0] for k in per_rank[0]} & {k.split('::')[0] for k in per_rank[1]}
