from torch.utils.data import DataLoader

from chugr.common import LoaderBundle, SharedCount
from chugr.hfds.loader import EvalSampler
from chugr.hfds.wrappers import WrappedIterableDataset


class _EpochDataset:
    """ Stand-in for a streaming HF dataset, records set_epoch calls. """

    def __init__(self, n=5):
        self.n = n
        self.epoch = 0
        self.calls = []

    def set_epoch(self, epoch):
        self.calls.append(epoch)
        self.epoch = epoch

    def __iter__(self):
        return iter(range(self.n))


def test_eval_sampler_exact_split():
    for n in range(8):
        for world in range(1, 6):
            per_rank = [list(EvalSampler(range(n), rank=r, world_size=world)) for r in range(world)]
            assert sorted(i for p in per_rank for i in p) == list(range(n))  # each sample once, no padding
            assert max(map(len, per_rank)) - min(map(len, per_rank)) <= 1


def test_wrapped_iterable_epoch_set_by_main_only():
    # the HF epoch is shared w/ the dataloader workers, setting it from the workers races
    ds = _EpochDataset()
    count = SharedCount(3)
    wrapped = WrappedIterableDataset(ds, interval_count=count, num_samples=7, repeat=True)
    assert ds.calls == [3]
    bundle = LoaderBundle(loader=DataLoader(wrapped, batch_size=None), shared_interval=count)
    bundle.set_interval(4)
    assert ds.calls == [3, 4] and count.get_value() == 4
    assert list(bundle) == [0, 1, 2, 3, 4, 0, 1]
    assert ds.calls == [3, 4]  # iteration (in workers) leaves the epoch alone
