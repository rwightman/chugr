import random
import sys

from torch.utils.data import IterableDataset

from chugr.common import SharedCount, get_pytorch_worker_seed
from .helpers import expand_urls


class ShuffledShardList(IterableDataset):
    """An iterable dataset yielding a list of urls that is deterministically shuffled based on epoch."""

    def __init__(
            self,
            urls,
            seed=0,
            interval=-1,
            num_sub_intervals=None,
            repeat: bool = False,
    ):
        """Iterate through the list of shards.

        Args:
            repeat: cycle through the shards indefinitely (each pass reshuffled w/ a seed shared by all
                ranks / workers so node / worker splitting stays consistent), for schedule driven loaders that
                need an infinite sample stream w/o sampling shards with replacement.
        """
        super().__init__()
        self.repeat = repeat
        self.urls = expand_urls(urls)
        assert len(self.urls) and isinstance(self.urls[0], str)
        self.seed = seed
        self.interval = interval
        self.num_sub_intervals = num_sub_intervals  # FIXME experimental feature

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        """Return an iterator over the shards."""
        urls = self.urls.copy()

        # Set epoch
        if isinstance(self.interval, SharedCount):
            interval = self.interval.get_value()
        else:
            # NOTE: this is interval tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.interval += 1
            interval = self.interval

        pass_idx = 0
        while True:
            pass_urls = list(urls)
            if self.seed is not None:
                # Shuffle with the same seed across all nodes/workers in each interval or super interval
                if self.num_sub_intervals is None:
                    seed = self.seed + interval
                else:
                    # Keep shuffling consistent across the super epochs
                    seed = self.seed + (interval // self.num_sub_intervals)
                random.Random(seed * 1000003 + pass_idx if pass_idx else seed).shuffle(pass_urls)

            # Restrict to shards in the sub epoch if needed
            if self.num_sub_intervals is not None:
                pass_urls = pass_urls[interval % self.num_sub_intervals::self.num_sub_intervals]

            # Yield shards
            for url in pass_urls:
                yield dict(url=url)

            if not self.repeat:
                break
            pass_idx += 1


class ResampledShardsV2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
            self,
            urls,
            weights=None,
            nshards=sys.maxsize,
            worker_seed_fn=None,
            deterministic=False,
            interval=-1,
            seed=None,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        if weights is not None:
            self.urls, self.weights = expand_urls(urls, weights)
            assert len(self.urls) == len(self.weights), \
                f"Number of urls {len(self.urls)} and weights {len(self.weights)} should match."
        else:
            self.urls = expand_urls(urls)
            self.weights = None
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed_fn = worker_seed_fn
        self.deterministic = deterministic
        self.interval = interval
        self.seed = seed  # only used when seed cannot be recovered from DL workers

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.interval, SharedCount):
            interval = self.interval.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.interval += 1
            interval = self.interval

        if self.deterministic:
            # reset seed w/ interval if deterministic
            if self.worker_seed_fn is None:
                # pytorch worker seed should be deterministic (per-worker)
                # It is init by process seed, rank, & worker id
                seed = get_pytorch_worker_seed(interval, initial_seed=self.seed)
            else:
                seed = self.worker_seed_fn() + interval
            self.rng.seed(seed)

        for _ in range(self.nshards):
            if self.weights is None:
                yield dict(url=self.rng.choice(self.urls))
            else:
                yield dict(url=self.rng.choices(self.urls, weights=self.weights, k=1)[0])
