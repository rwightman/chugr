""" NaFlex (variable resolution patch sequence) batching for document pipelines.

Adapted from the open_clip NaFlex streaming data pipeline:

* NaFlexBatchScheduler: a deterministic (seq_len, batch_size) schedule, identical across ranks (seeded), sampled
  per batch from a set of patch sequence length buckets, w/ batch size from a token budget (patches + an optional
  per-row text token cost). Shuffled per interval, padded and stride-split across dataloader workers.
* NaFlexBatcher: webdataset stage that pulls samples, runs the task processor for the batch's seq_len bucket
  (decode / render + image transform + text processing), skips & replaces samples that fail to process so every
  batch is full (equal step shapes across ranks), and collates patches padded to the bucket seq_len.
* LengthBucketer: reorders samples so similar (text) lengths batch together, reduces decoder padding.

NOTE schedule driven batching needs an effectively infinite sample stream so every rank / worker can always fill
its schedule, the loader uses resampled shards or (w/o replacement) a repeating, per-pass reshuffled shard list.
"""
import json
import logging
import math
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from chugr.common import collate, collate_naflex

_logger = logging.getLogger(__name__)

_MAX_CONSECUTIVE_FAILURES = int(os.environ.get('CHUGR_MAX_CONSECUTIVE_FAILURES', '1000'))


def calculate_batch_size(tokens_per_batch: int, row_tokens: int, divisor: int = 1, max_size: Optional[int] = None):
    batch_size = tokens_per_batch // max(row_tokens, 1)
    if divisor > 1:
        batch_size = (batch_size // divisor) * divisor
    batch_size = max(batch_size, 1)
    if max_size is not None:
        batch_size = min(batch_size, max_size)
    return batch_size


def _interval_value(interval) -> int:
    if hasattr(interval, 'get_value'):
        return int(interval.get_value())
    return int(interval or 0)


class NaFlexBatchScheduler:
    """ Deterministic, rank identical NaFlex (seq_len, batch_size) schedule for one interval (epoch). """

    def __init__(
            self,
            seq_lens: Sequence[int],
            max_tokens_per_batch: int,
            num_samples: Optional[int] = None,
            num_tokens: Optional[int] = None,
            seq_len_probs: Optional[Sequence[float]] = None,
            per_row_text_tokens: int = 0,
            batch_divisor: int = 8,
            seed: int = 42,
            shuffle: bool = True,
            world_size: int = 1,
    ):
        if (num_samples is None) == (num_tokens is None):
            raise ValueError('Specify exactly one of `num_samples` or `num_tokens` for NaFlex batching.')
        if seq_len_probs is not None:
            if len(seq_len_probs) != len(seq_lens):
                raise ValueError('`seq_len_probs` must match `seq_lens` length.')
            pairs = sorted({int(s): float(p) for s, p in zip(seq_lens, seq_len_probs)}.items())
            self.seq_lens = [s for s, _ in pairs]
            total = sum(p for _, p in pairs)
            assert total > 0
            self.seq_len_probs = [p / total for _, p in pairs]
        else:
            self.seq_lens = sorted(set(int(s) for s in seq_lens))
            self.seq_len_probs = None
        assert self.seq_lens and all(s > 0 for s in self.seq_lens)
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.per_row_text_tokens = int(per_row_text_tokens or 0)
        self.batch_divisor = int(batch_divisor)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.world_size = max(1, int(world_size))

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        schedule = []
        if num_samples is not None:
            remaining = math.ceil(int(num_samples) / self.world_size)
            while remaining > 0:
                seq_len = self._next_seq_len(generator)
                bs = calculate_batch_size(
                    self.max_tokens_per_batch, seq_len + self.per_row_text_tokens, self.batch_divisor, remaining)
                schedule.append((seq_len, bs))
                remaining -= bs
        else:
            remaining = math.ceil(int(num_tokens) / self.world_size)
            while remaining > 0:
                seq_len = self._next_seq_len(generator)
                bs = calculate_batch_size(
                    min(self.max_tokens_per_batch, remaining), seq_len + self.per_row_text_tokens, self.batch_divisor)
                schedule.append((seq_len, bs))
                remaining -= bs * seq_len
        self._schedule = schedule

    def _next_seq_len(self, generator: torch.Generator) -> int:
        if self.seq_len_probs is None:
            idx = int(torch.randint(0, len(self.seq_lens), (1,), generator=generator).item())
        else:
            idx = int(torch.multinomial(torch.tensor(self.seq_len_probs), 1, generator=generator).item())
        return self.seq_lens[idx]

    @staticmethod
    def pad_schedule_for_workers(schedule: List[Tuple[int, int]], num_workers: int) -> List[Tuple[int, int]]:
        if num_workers <= 1 or not schedule:
            return schedule
        target = math.ceil(len(schedule) / num_workers) * num_workers
        pad = target - len(schedule)
        if pad > 0:
            schedule = schedule + (schedule * math.ceil(pad / len(schedule)))[:pad]
        return schedule

    def interval_schedule(self, interval: int, num_workers: int = 1) -> List[Tuple[int, int]]:
        schedule = self.pad_schedule_for_workers(list(self._schedule), max(1, num_workers))
        if self.shuffle:
            random.Random(self.seed + interval).shuffle(schedule)
        return schedule

    def worker_schedule(self, interval: int) -> List[Tuple[int, int]]:
        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        worker_id = info.id if info is not None else 0
        return self.interval_schedule(interval, num_workers)[worker_id::num_workers]

    def num_batches_for_workers(self, num_workers: int) -> int:
        """ Batches per rank per interval. """
        return len(self.pad_schedule_for_workers(list(self._schedule), max(1, num_workers)))

    def num_samples_for_workers(self, num_workers: int) -> int:
        """ Samples per interval (all ranks). """
        sched = self.pad_schedule_for_workers(list(self._schedule), max(1, num_workers))
        return sum(bs for _, bs in sched) * self.world_size

    def mean_batch_size_for_workers(self, num_workers: int) -> float:
        sched = self.pad_schedule_for_workers(list(self._schedule), max(1, num_workers))
        return sum(bs for _, bs in sched) / len(sched)


class NaFlexBatcher:
    """ webdataset stage, schedule driven NaFlex batching w/ per seq_len bucket task processors. """

    def __init__(
            self,
            scheduler: NaFlexBatchScheduler,
            processors: Dict[int, Callable[[Dict[str, Any]], Any]],
            image_key: str = 'image_input',
            interval=0,
            handler: Optional[Callable[[Exception], bool]] = None,
            drop_last: bool = True,
    ):
        missing = set(scheduler.seq_lens) - set(processors.keys())
        assert not missing, f'Missing processors for seq_lens {missing}'
        self.scheduler = scheduler
        self.processors = processors
        self.image_key = image_key
        self.interval = interval
        self.handler = handler
        self.drop_last = drop_last

    def _collate(self, samples: List[Dict[str, Any]], seq_len: int) -> Dict[str, Any]:
        images = [s.pop(self.image_key) for s in samples]
        out = collate(samples)
        out[self.image_key] = collate_naflex(images, pad_to=seq_len)
        return out

    def __call__(self, src: Iterable[Dict[str, Any]]):
        interval = _interval_value(self.interval)
        samples_iter = iter(src)
        for seq_len, batch_size in self.scheduler.worker_schedule(interval):
            processor = self.processors[seq_len]
            batch = []
            failures = 0
            while len(batch) < batch_size:
                try:
                    sample = next(samples_iter)
                except StopIteration:
                    break
                try:
                    processed = processor(sample)
                except Exception as ex:
                    # skip & replace, a falsy handler return (e.g. reraise) stops
                    if self.handler is not None and not self.handler(ex):
                        raise
                    failures += 1
                    if failures > _MAX_CONSECUTIVE_FAILURES:
                        raise
                    continue
                failures = 0
                batch.append(processed)
            if len(batch) == batch_size or (batch and not self.drop_last):
                yield self._collate(batch, seq_len)


def annotation_text_length(sample: Dict[str, Any], keys: Sequence[str] = ('json',)) -> int:
    """ Approximate text length (characters) of a raw document sample's annotation (for length bucketing). """
    for k in keys:
        if k not in sample:
            continue
        anno = sample[k]
        if isinstance(anno, bytes):
            anno = json.loads(anno)
            sample[k] = anno  # keep decoded
        pages = anno.get('pages', anno) if isinstance(anno, dict) else anno
        total = 0
        try:
            for page in pages:
                page = page.get('lines', page)
                total += sum(len(t) for t in page.get('text', []))
        except (AttributeError, TypeError):
            return 0
        n = max(len(pages), 1)
        return total // n  # mean per page (pages are sampled)
    return 0


class LengthBucketer:
    """ webdataset stage, reorder samples in pools sorted by length (chunks shuffled), reorder only. """

    def __init__(
            self,
            length_fn: Callable[[Dict[str, Any]], int] = annotation_text_length,
            pool: int = 1024,
            chunk: int = 64,
            seed: int = 42,
            interval=0,
    ):
        self.length_fn = length_fn
        self.pool = max(1, int(pool))
        self.chunk = max(1, int(chunk))
        self.seed = int(seed)
        self.interval = interval

    def _flush(self, buffer, rng):
        buffer.sort(key=self.length_fn)
        chunks = [buffer[i:i + self.chunk] for i in range(0, len(buffer), self.chunk)]
        rng.shuffle(chunks)
        for c in chunks:
            yield from c

    def __call__(self, src):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rng = random.Random(self.seed + _interval_value(self.interval) * 131 + worker_id)
        buffer = []
        for sample in src:
            buffer.append(sample)
            if len(buffer) >= self.pool:
                yield from self._flush(buffer, rng)
                buffer = []
        if buffer:
            yield from self._flush(buffer, rng)
