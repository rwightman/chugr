import math
import warnings
from copy import deepcopy
from typing import Callable, List, Optional

import torch
from torch.utils.data import DataLoader, Sampler

from chugr.common import (
    LoaderBundle, DistributedCfg, SharedCount, rank_seed, seed_worker, shuffled_interval_indices)

from .collate import HfCollate
from .wrappers import SafeDataset, WrappedIterableDataset

_SAMPLE_SHUFFLE_SIZE = 2000


class IntervalSampler(Sampler):
    """ Shuffled, rank sharded sampler w/ an optional per-interval sample limit (for intervals < 1 epoch).

    Each interval draws a fresh permutation (seed + interval), samples are consumed sequentially across
    intervals so that short intervals still cycle through the full dataset.
    """

    def __init__(
            self,
            dataset,
            num_samples: Optional[int] = None,
            rank: int = 0,
            world_size: int = 1,
            seed: int = 0,
            interval: int = 0,
    ):
        self.dataset_len = len(dataset)
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.interval = interval
        total = min(num_samples, self.dataset_len) if num_samples else self.dataset_len
        self.num_samples_per_rank = total // world_size

    def set_epoch(self, interval: int):
        self.interval = interval

    def __len__(self):
        return self.num_samples_per_rank

    def __iter__(self):
        per_interval = self.num_samples_per_rank * self.world_size
        indices = shuffled_interval_indices(self.dataset_len, per_interval, self.interval, self.seed)
        return iter(indices[self.rank::self.world_size])


class EvalSampler(Sampler):
    """ Rank strided eval sampler, every sample exactly once across ranks w/o padding.

    Unlike DistributedSampler (drop_last=False), samples are not repeated to even out the per-rank counts (ranks
    differ by at most one sample). Repeats would be counted twice in the gathered metrics and depend on the world
    size. Eval has no per batch collectives, so uneven per-rank batch counts are fine.
    """

    def __init__(self, dataset, rank: int = 0, world_size: int = 1):
        self.indices = range(rank, len(dataset), world_size)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)


def _disable_decode(ds):
    import datasets

    to_set = []
    for k, v in ds.features.items():
        if isinstance(v, datasets.Image):
            d = deepcopy(v)
            d.decode = False
            to_set.append((k, d))
        elif isinstance(v, datasets.Audio):
            d = deepcopy(v)
            d.decode = False
            to_set.append((k, d))
    for k, d in to_set:
        ds = ds.cast_column(k, d)
    return ds


def create_loader_hf(
        source: str,
        split: str,
        task_pipeline: Optional[List[Callable]] = None,
        data_dir: Optional[str] = None,
        num_samples: Optional[int] = None,
        streaming: bool = False,
        is_training: bool = False,
        batch_size: Optional[int] = 1,
        resampled: bool = False,
        multi_interval: bool = True,
        num_batches_round: str = 'ceil',
        num_workers: int = 4,
        persistent_workers: bool = True,
        start_interval: int = 0,
        seed: int = 0,
        collate_fn: Optional[Callable] = None,
        sample_shuffle_size: int = _SAMPLE_SHUFFLE_SIZE,
        distributed: DistributedCfg = DistributedCfg(),
        disable_decode: bool = False,
        naflex_factory: Optional[Callable] = None,
        handler: Optional[Callable] = None,
):
    """

    Args:
        source:
        split:
        task_pipeline:
        data_dir:
        num_samples:
        streaming:
        is_training:
        batch_size:
        resampled:
        multi_interval:
        num_batches_round:
        num_workers:
        persistent_workers:
        start_interval:
        seed:
        collate_fn:
        sample_shuffle_size:
        distributed:
        disable_decode:
        naflex_factory: num_samples (per interval) -> (NaFlexBatchScheduler, per seq_len processors, image key),
            train w/ schedule driven (variable seq_len & batch size) NaFlex batching, replaces the task pipeline
            and fixed size batching
        handler: Exception handler for NaFlex batching (failed samples are replaced)

    Returns:

    """
    from datasets import VerificationMode, load_dataset
    batched = batch_size is not None and batch_size >= 0
    naflex = is_training and naflex_factory is not None

    if naflex:
        pass  # the NaFlex dataset wrapper processes and collates
    elif collate_fn is not None:
        assert task_pipeline is None, 'task_pipeline should not be set if custom collation function is used.'
    elif batched or task_pipeline is not None:
        # collation fn applies task pipeline
        assert task_pipeline is not None, 'task_pipeline is needed'
        collate_fn = HfCollate(
            task_pipeline,
            apply_collate=batched,
        )

    if streaming:
        from datasets.distributed import split_dataset_by_node

        dataset = load_dataset(
            source,
            data_dir=data_dir,
            streaming=True,
        )

        if split not in dataset:
            assert False, f'Split {split} not in dataset ({dataset.keys()})'
        dataset = dataset[split]
        if disable_decode:
            dataset = _disable_decode(dataset)

        if not is_training and num_samples:
            # eval, cap the stream before the rank split, the count is exact and the eval set doesn't change w/ world
            # size. NOTE w/ dataloader workers, HF splits the cap across the workers' shards, so which samples
            # depends on num_workers (the first num_samples of the stream w/o workers)
            dataset = dataset.take(num_samples)

        # FIXME num_samples calc, get a reliable estimate from dataset in streaming mode
        if num_samples is None:
            info = dataset.info
            if info.splits is not None and split in info.splits:
                num_samples = info.splits[split].num_examples

        if is_training and multi_interval:
            assert num_samples, (
                "num_samples must be available in dataset metadata or manually provided for multi-interval training")

        if is_training:
            dataset = dataset.shuffle(seed, buffer_size=sample_shuffle_size)

        # FIXME split_dataset_by_node has some concerns as currently implemented
        dataset = split_dataset_by_node(dataset, distributed.global_rank, distributed.world_size)
        interval_count = SharedCount(start_interval)

        if is_training:
            # HF assigns the data shards (files) of this rank to the dataloader workers, a worker w/o a shard
            # yields nothing, which a fixed number of batches per worker can't tolerate
            num_shards = getattr(dataset, 'num_shards', None) or getattr(dataset, 'n_shards', None)
            if num_shards and num_workers > num_shards:
                warnings.warn(
                    f'Reducing num_workers from {num_workers} to {num_shards}, the number of data shards '
                    f'of the streaming dataset (per rank).')
                num_workers = num_shards
        num_workers_nonzero = max(num_workers, 1)
        # worker seeds (augmentation) differ across ranks, the dataset shuffle uses the shared seed
        dl_generator = torch.Generator()
        dl_generator.manual_seed(rank_seed(seed, distributed.global_rank))
        loader_kwargs = dict(
            sampler=None,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            generator=dl_generator,
            worker_init_fn=seed_worker,
        )

        mean_batch_size = None
        if naflex:
            # schedule driven batching, every worker yields its share of the rank identical schedule from its
            # (repeating) share of the stream
            from chugr.naflex import NaFlexBatcher

            assert num_samples, 'NaFlex batching requires num_samples (per interval).'
            scheduler, processors, image_key = naflex_factory(num_samples)
            batcher = NaFlexBatcher(
                scheduler, processors, image_key=image_key, interval=interval_count, handler=handler)
            dataset = WrappedIterableDataset(dataset, interval_count=interval_count, repeat=True, stages=[batcher])
            base_loader = DataLoader(dataset=dataset, batch_size=None, **loader_kwargs)
            num_batches = scheduler.num_batches_for_workers(num_workers_nonzero)
            num_samples = scheduler.num_samples_for_workers(num_workers_nonzero)
            mean_batch_size = scheduler.mean_batch_size_for_workers(num_workers_nonzero)
        else:
            num_samples_per_worker = None
            num_batches = 0
            if is_training and batched and batch_size and num_samples:
                # the same number of full batches on every rank, enforced by a fixed number of samples per
                # worker w/ the stream wrapping around (as the wds loader, see the notes there)
                round_fn = math.floor if num_batches_round == 'floor' else math.ceil
                global_batch_size = batch_size * distributed.world_size
                num_batches = round_fn(num_samples / global_batch_size)
                num_batches_per_worker = max(round_fn(num_batches / num_workers_nonzero), 1)
                num_batches = num_batches_per_worker * num_workers_nonzero
                num_samples = num_batches * global_batch_size
                num_samples_per_worker = num_batches_per_worker * batch_size
            elif num_samples:
                # eval (or unbatched), an estimate, split_dataset_by_node gives each rank ~num_samples / world_size
                num_batches = math.ceil(num_samples / distributed.world_size / (batch_size or 1))
            dataset = WrappedIterableDataset(
                dataset,
                interval_count=interval_count,
                num_samples=num_samples_per_worker,
                repeat=num_samples_per_worker is not None,
            )
            # HF datasets treats batch_size differently than torch defaults, in torch batch_size = None
            # disables batching, in HF it returns the full dataset. We restore torch behaviour here.
            base_loader = DataLoader(
                dataset=dataset,
                collate_fn=collate_fn,
                drop_last=batched and is_training,
                batch_size=batch_size,
                **loader_kwargs,
            )

        loader = LoaderBundle(
            loader=base_loader,
            num_batches=num_batches,
            num_samples=num_samples,
            shared_interval=interval_count,
            mean_batch_size=mean_batch_size,
        )

    else:
        # split may include slicing, e.g. 'train[:1000]' or 'validation[:10%]'
        dataset = load_dataset(
            source,
            data_dir=data_dir,
            split=split,
            verification_mode=VerificationMode.BASIC_CHECKS,
        )
        if not is_training and num_samples and num_samples < len(dataset):
            dataset = dataset.select(range(num_samples))
        if disable_decode:
            dataset = _disable_decode(dataset)
        if naflex:
            return _create_loader_hf_naflex(
                dataset,
                naflex_factory=naflex_factory,
                num_samples=num_samples,
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                start_interval=start_interval,
                seed=seed,
                distributed=distributed,
                handler=handler,
            )

        dataset = SafeDataset(dataset)

        if is_training:
            # shuffled (per interval), sharded by rank, optionally limited to num_samples per interval
            sampler = IntervalSampler(
                dataset,
                num_samples=num_samples,
                rank=distributed.global_rank,
                world_size=distributed.world_size,
                seed=seed,
                interval=start_interval,
            )
        elif distributed.world_size > 1:
            sampler = EvalSampler(dataset, rank=distributed.global_rank, world_size=distributed.world_size)
        else:
            sampler = None

        base_loader = DataLoader(
            dataset=dataset,
            collate_fn=collate_fn,
            sampler=sampler,
            drop_last=is_training,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
        )

        loader = LoaderBundle(
            loader=base_loader,
            num_batches=len(base_loader),
            num_samples=len(sampler) * distributed.world_size if is_training else len(dataset),
            sampler=sampler,
        )

    return loader


def _create_loader_hf_naflex(
        dataset,
        naflex_factory: Callable,
        num_samples: Optional[int],
        num_workers: int,
        persistent_workers: bool,
        start_interval: int,
        seed: int,
        distributed: DistributedCfg,
        handler: Optional[Callable],
):
    """ Train loader w/ schedule driven NaFlex batching over a map-style dataset. """
    from chugr.naflex import NaFlexMapDataset

    # samples per interval (all ranks), the schedule rounds up to full batches per rank / worker
    num_samples = min(num_samples, len(dataset)) if num_samples else len(dataset)
    scheduler, processors, image_key = naflex_factory(num_samples)
    interval_count = SharedCount(start_interval)
    # NOTE not wrapped in SafeDataset, the NaFlex wrapper replaces samples that fail to load or process
    naflex_dataset = NaFlexMapDataset(
        dataset,
        scheduler=scheduler,
        processors=processors,
        image_key=image_key,
        interval=interval_count,
        rank=distributed.global_rank,
        world_size=distributed.world_size,
        seed=seed,
        handler=handler,
    )
    # worker seeds (augmentation, replacement samples) differ across ranks, indices / schedule use the shared seed
    dl_generator = torch.Generator()
    dl_generator.manual_seed(rank_seed(seed, distributed.global_rank))
    base_loader = DataLoader(
        dataset=naflex_dataset,
        batch_size=None,  # batching done by the dataset wrapper
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
        generator=dl_generator,
        worker_init_fn=seed_worker,
    )
    num_workers_nonzero = max(num_workers, 1)
    return LoaderBundle(
        loader=base_loader,
        num_batches=scheduler.num_batches_for_workers(num_workers_nonzero),
        num_samples=scheduler.num_samples_for_workers(num_workers_nonzero),
        shared_interval=interval_count,
        mean_batch_size=scheduler.mean_batch_size_for_workers(num_workers_nonzero),
    )
