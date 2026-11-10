import random

import numpy as np

from .types import SharedCount


_RANK_SEED_STRIDE = 1000003


def rank_seed(seed: int, rank: int = 0) -> int:
    """ Seed for per rank randomness (process RNGs, dataloader worker seeds).

    Ranks are spaced by a large stride (vs seed + rank) so runs w/ nearby seeds don't share random streams
    (seed 42 rank 1 == seed 43 rank 0), rank 0 keeps `seed` so single process runs are unchanged.
    """
    return seed + _RANK_SEED_STRIDE * rank


def seed_worker(worker_id):
    import torch
    worker_seed = torch.initial_seed()
    random.seed(worker_seed)
    np.random.seed(worker_seed % 2**32)


def get_pytorch_worker_seed(increment=0, initial_seed=None):
    """get dataloader worker seed from pytorch
    """
    from torch.utils.data import get_worker_info

    increment_value = increment.get_value() if isinstance(increment, SharedCount) else increment
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        num_workers = worker_info.num_workers
        if increment_value:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment_value * max(1, num_workers)
    else:
        # a fallback when no dataloader workers are present (num_workers=0)
        import torch

        if initial_seed is None:
            initial_seed = torch.initial_seed()

        # generate seed from initial via torch.Generator so it matches DL worker seeds
        seed = torch.empty((), dtype=torch.int64).random_(
            generator=torch.Generator().manual_seed(initial_seed)).item()

        if increment_value:
            seed += increment_value

    return seed


def shuffled_interval_indices(dataset_len: int, num_indices: int, interval: int = 0, seed: int = 0):
    """ Shuffled dataset indices for one interval of a map-style dataset (all ranks, split w/ [rank::world_size]).

    Each pass over the dataset is a fresh permutation (seed + pass), intervals consume the permutations
    sequentially (interval i starts where i - 1 ended), so intervals shorter than the dataset still cycle through
    every sample before repeating. Identical across ranks / workers for the same args.
    """
    import torch
    assert dataset_len > 0, 'dataset is empty'
    epoch, offset = divmod(interval * num_indices, dataset_len)
    indices = []
    while len(indices) < num_indices:
        g = torch.Generator()
        g.manual_seed(seed + epoch)
        perm = torch.randperm(dataset_len, generator=g).tolist()
        indices.extend(perm[offset:])
        offset = 0
        epoch += 1
    return indices[:num_indices]
