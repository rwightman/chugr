import logging
from itertools import islice
from typing import Callable, List, Optional

from torch.utils.data import Dataset, IterableDataset

from chugr.common import SharedCount

_logger = logging.getLogger(__name__)

class SafeDataset(Dataset):
    """
    This is a Dataset wrapped by a try/except in the __getitem__ in case
    the hfds datasets used have errors/corrupt data. A sample that fails to load is replaced by
    the next loadable one (wrapping around at the end of the dataset), w/ a warning.
    """

    def __init__(self, original_dataset, max_retry=10):
        self.ds = original_dataset
        self.max_retry = max_retry

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        err = None
        num = len(self.ds)
        for try_idx in range(min(self.max_retry, num)):
            try_at = (idx + try_idx) % num  # wrap, idx + try_idx can run past the end
            try:
                item = self.ds[try_at]
            except Exception as e:
                err = e
                continue
            if try_idx:
                _logger.warning(f'Sample {idx} failed to load ({err!r}), replaced by sample {try_at}.')
            return item
        raise err if err is not None else IndexError(idx)



class WrappedIterableDataset(IterableDataset):
    """ Wrap a streaming (iterable) HF dataset for interval based training loops.

    * The interval count (shared w/ dataloader workers) sets the dataset epoch (shard / buffer shuffle seed), set
      from the main process only (via `set_interval_count`). The HF epoch is a shared memory tensor updated w/ a
      non-atomic in-place add, workers setting it concurrently race (an overshot / differing epoch per worker breaks
      the worker shard partition, shards are skipped or read twice).
    * `num_samples` w/ `repeat=True` makes every dataloader worker yield exactly `num_samples` samples per
      interval, wrapping around its share of the stream if it runs out (a few samples repeat), so every rank sees
      the same number of batches (as the wds loader).
    * `stages` are applied to the sample stream (generator fn taking an iterable, e.g. NaFlexBatcher which then
      yields batches following its schedule).
    """

    def __init__(
            self,
            original_dataset,
            interval_count=None,
            max_retry=10,
            num_samples: Optional[int] = None,
            repeat: bool = False,
            stages: Optional[List[Callable]] = None,
    ):
        self.ds = original_dataset
        self.max_retry = max_retry
        self.interval_count = interval_count
        self.ds.set_epoch(self._get_interval_count())
        self.num_samples = num_samples
        self.repeat = repeat
        self.stages = list(stages or [])

    def _get_interval_count(self):
        if isinstance(self.interval_count, SharedCount):
            return self.interval_count.get_value()
        return self.interval_count or 0

    def set_interval_count(self, interval_count):
        """ Main process only, workers (persistent or not) see the shared count and HF epoch. """
        if isinstance(self.interval_count, SharedCount):
            self.interval_count.set_value(interval_count)
        else:
            self.interval_count = interval_count
        self.ds.set_epoch(interval_count)

    def _samples(self):
        # NOTE no set_epoch here, see the class docstring
        if not self.repeat:
            yield from self.ds
            return
        while True:
            # NOTE wraps w/ the same epoch (shuffle seed), a per rank / worker reshuffle could break the node split
            count = 0
            for sample in self.ds:
                count += 1
                yield sample
            if not count:
                raise RuntimeError(
                    'No samples for this dataloader worker, a repeating streaming dataset needs at least one data '
                    'shard (file) per worker of every rank.')

    def __iter__(self):
        src = self._samples()
        if self.num_samples:
            src = islice(src, self.num_samples)
        for stage in self.stages:
            src = stage(src)
        yield from src
