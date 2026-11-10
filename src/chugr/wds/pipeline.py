from functools import partial
from itertools import islice
from typing import Callable, List, Optional, Sequence, Union

import torch.utils.data
import webdataset
import webdataset as wds

from chugr.common import ShardSpec, collate
from .filters import detshuffle_v2
from .shardlists import ResampledShardsV2, ShuffledShardList
from .tariterators import tarfile_to_samples_nothrow


def split_samples_by_node_and_worker(src, rank: int = 0, world_size: int = 1):
    """ Split a sample stream that is identical on all ranks / workers, rank r keeps every world_size-th sample
    (independent of the worker count), its workers split those. """
    info = torch.utils.data.get_worker_info()
    num_workers = info.num_workers if info is not None else 1
    worker_id = info.id if info is not None else 0
    yield from islice(islice(src, rank, None, world_size), worker_id, None, num_workers)


def build_data_pipeline(
        shards: ShardSpec,
        task_pipeline: List[Callable],
        is_training: bool = False,
        batch_size: int = 0,
        resampled: bool = False,
        multi_interval: bool = False,
        seed: int = 0,
        shared_interval_count: int = -1,
        num_batches_per_worker: int = 0,
        sample_shuffle_initial: int = 1,
        sample_shuffle_size: int = 1,
        collate_fn: Optional[Callable] = None,
        batched_task_pipeline: Optional[List[Callable]] = None,
        handler=wds.reraise_exception,
        max_samples: Optional[int] = None,
        repeat_shards: bool = False,
        worker_seed: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
):
    """

    Args:
        task_pipeline: task-specific pipeline for processing samples
        shards: ShardSpec w/ shard url list and optional sizes and weights
        is_training: Train mode. Enables shuffling of samples and forces consistent batch #s across workers.
        batch_size: Batch for sample collation. Collation / batching disabled if batch_size == 0.
        resampled: Enable resampling with replacement of shards.
        multi_interval:
        seed:
        shared_interval_count:
        num_batches_per_worker:
        sample_shuffle_initial:
        sample_shuffle_size:
        collate_fn:
        batched_task_pipeline: An optional task-specific pipeline for processing batched samples
        handler: Exception handler
        max_samples: Eval, limit the eval set to the first `max_samples` samples of the shard list. Exact for any
            number of ranks / workers, every worker reads those samples and keeps its share, so leave it unset
            to evaluate all samples (split by shard where possible).
        repeat_shards: Train w/o resampling, cycle through the shard list (reshuffled each pass)
        worker_seed: Seed for per worker randomness (shard resampling, sample shuffle) when the seed can't be
            recovered from DataLoader workers (num_workers=0), must differ across ranks. Defaults to `seed`.
        rank: Rank of this process, shards (train, eval w/ enough shards) or samples (eval) are split across
            ranks. Passed explicitly, not derived from RANK / WORLD_SIZE env vars or torch.distributed state
            (neither is reliably visible in dataloader workers for all launchers / start methods).
        world_size: Number of ranks
        num_workers: Number of dataloader workers (eval split)

    Returns:

    """
    worker_seed = seed if worker_seed is None else worker_seed
    if not isinstance(task_pipeline, (list, tuple)):
        task_pipeline = [task_pipeline]
    assert len(task_pipeline)

    if resampled:
        datapipe = [ResampledShardsV2(
            shards.urls,
            weights=shards.weights,
            deterministic=True,
            interval=shared_interval_count,
            seed=worker_seed,
        )]
    else:
        assert shards.weights is None, \
            "upsampling_factors is only supported when sampling with replacement (resampled=False)."
        if is_training:
            datapipe = [ShuffledShardList(
                shards.urls,
                seed=seed,
                interval=shared_interval_count,
                repeat=repeat_shards,
            )]
        else:
            datapipe = [wds.SimpleShardList(
                shards.urls,
            )]

    # at this point we have an iterator over all the shards
    if is_training:
        if not resampled:
            if world_size > 1:
                # as wds.split_by_node, w/ the explicit rank
                datapipe.append(wds.slice(rank, None, world_size))
            datapipe.append(wds.split_by_worker)
        # at this point, we have an iterator over the shards assigned to each worker at each node
        datapipe.extend([
            tarfile_to_samples_nothrow(handler=handler),
            detshuffle_v2(
                bufsize=sample_shuffle_size,
                initial=sample_shuffle_initial,
                seed=worker_seed,
                interval=shared_interval_count,
                unique_worker=True,
            )
            # wds.shuffle(
            #     bufsize=sample_shuffle_size,
            #     initial=sample_shuffle_initial,
            # ),
        ])
    else:
        if max_samples:
            # capped eval set, the first max_samples samples of the shard list for any world size / worker count.
            # Every worker reads (but doesn't decode / process) that head of the stream and keeps its share.
            datapipe.extend([
                wds.tarfile_to_samples(handler=handler),
                wds.slice(max_samples),
                partial(split_samples_by_node_and_worker, rank=rank, world_size=world_size),
            ])
        elif len(shards.urls) >= world_size * max(num_workers, 1):
            # enough shards for every worker of every rank, split shards so each is read once
            if world_size > 1:
                datapipe.append(wds.slice(rank, None, world_size))
            datapipe.extend([
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=handler),
            ])
        else:
            # few (or one) eval shards, split samples across ranks, every rank reads the samples of its workers'
            # shards but only decodes / processes its share
            datapipe.extend([
                wds.split_by_worker,
                wds.tarfile_to_samples(handler=handler),
            ])
            if world_size > 1:
                datapipe.append(wds.slice(rank, None, world_size))

    # task specific decode and map pipline (per-sample)
    datapipe.extend(task_pipeline)

    # collation, tensor output disabled with batch_size == 0 or None
    if batch_size:
        # NOTE torch default_collate handles dicts, wds default_collate does not
        collate_fn = collate_fn or collate
        datapipe.extend([
            wds.batched(
                batch_size,
                partial=not is_training,
                collation_fn=collate_fn,
            )
        ])

        # task-specific batched pipeline (per-batch)
        if batched_task_pipeline:
            datapipe.extend(batched_task_pipeline)

    datapipe = wds.DataPipeline(*datapipe)

    if is_training and num_batches_per_worker > 0:
        if multi_interval:
            datapipe = datapipe.with_epoch(num_batches_per_worker)  # each worker is iterating over this
        else:
            datapipe = datapipe.repeat(nbatches=num_batches_per_worker)

    return datapipe
