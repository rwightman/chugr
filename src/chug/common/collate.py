import collections
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.data._utils.collate import default_collate_fn_map, default_collate_err_msg_format

NAFLEX_KEYS = ('patches', 'patch_coord', 'patch_valid')


def is_naflex_sample(elem) -> bool:
    return isinstance(elem, collections.abc.Mapping) and all(k in elem for k in NAFLEX_KEYS)


def collate_naflex(batch: List[Dict[str, torch.Tensor]], pad_multiple: int = 8) -> Dict[str, torch.Tensor]:
    """ Pad variable length patch sequences (NaFlex) to the longest in the batch.

    Args:
        batch: list of dicts w/ 'patches' [N, D] (or [N, Ph, Pw, C]), 'patch_coord' [N, 2], 'patch_valid' [N]
        pad_multiple: round padded length up to a multiple of this (kernel friendly shapes)
    """
    max_len = max(b['patches'].shape[0] for b in batch)
    if pad_multiple > 1:
        max_len = ((max_len + pad_multiple - 1) // pad_multiple) * pad_multiple
    batch_size = len(batch)
    patch_shape = batch[0]['patches'].shape[1:]
    patches = torch.zeros((batch_size, max_len, *patch_shape), dtype=batch[0]['patches'].dtype)
    patch_coord = torch.zeros((batch_size, max_len, 2), dtype=torch.int64)
    patch_valid = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i, b in enumerate(batch):
        n = b['patches'].shape[0]
        patches[i, :n] = b['patches']
        patch_coord[i, :n] = b['patch_coord']
        patch_valid[i, :n] = b['patch_valid']
    return {'patches': patches, 'patch_coord': patch_coord, 'patch_valid': patch_valid}


def collate(batch):
    r"""
        A customized collate function that handles collection type of element within each batch.

        This collate function has been tweaked to provide different functionality when handling
        dictionary samples. Certain keys are excluded or not tensorized.

        Args:
            batch: a single batch to be collated
    """
    elem = batch[0]
    elem_type = type(elem)

    if elem_type in default_collate_fn_map:
        return default_collate_fn_map[elem_type](batch)

    for collate_type in default_collate_fn_map:
        if isinstance(elem, collate_type):
            return default_collate_fn_map[collate_type](batch)

    if is_naflex_sample(elem):
        return collate_naflex(batch)

    if isinstance(elem, collections.abc.Mapping):
        try:
            out = {}
            for key in elem:
                if key.startswith('__'):
                    # skip keys starting with '__', e.g. '__key__',
                    continue
                elif key.startswith('_'):
                    # do not recurse or tensorize values for keys starting with '_', e.g. '_parse'
                    out[key] = [d[key] for d in batch]
                else:
                    out[key] =  collate([d[key] for d in batch])
            out = elem_type(out)
            return out
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [collate(samples) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))