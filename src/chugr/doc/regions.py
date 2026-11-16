""" Token aligned text regions and patch level text labels from OCR line annotations.

* Line regions are 4 point polys (TL, TR, BR, BL in reading order) from the annotation 'poly' (preferred) or
  'bbox' (x, y, w, h, an axis aligned poly) in normalized page coordinates.
* Token regions slice a line poly along its reading direction by the (decoded) character span of each token, one
  region per token, index aligned w/ the per line token ids.
* Patch labels: the token whose region contains a patch center (text patches), or a blank id for patches that
  don't touch any line region (blank patches), sampled to a fixed number of labels per page.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class PatchLabelCfg:
    num_labels: int = 512  # patch labels per page (fixed, padded w/ ignore_id)
    blank_frac: float = 0.2  # max fraction of the labels that are blank (no text) patches
    blank_id: Optional[int] = None  # label for blank patches, defaults to the tokenizer pad id, < 0 disables
    ignore_id: int = -100
    region_source: str = 'poly'  # 'poly' (falls back to 'bbox' if absent) or 'bbox'


def line_polys(anno_page: Dict, source: str = 'poly') -> Optional[np.ndarray]:
    """ Normalized line polys [L, 4, 2] (x, y) from a page annotation (lines level), None if not available. """
    texts = anno_page.get('text') or []
    polys = anno_page.get('poly') if source == 'poly' else None
    if polys and len(polys) == len(texts) and all(p and len(p) == 4 for p in polys):
        return np.array([[[c['X'], c['Y']] for c in p] for p in polys], dtype=np.float32).reshape(-1, 4, 2)
    boxes = anno_page.get('bbox')
    if boxes and len(boxes) == len(texts):
        b = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        x0, y0, x1, y1 = b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        return np.stack([
            np.stack([x0, y0], -1), np.stack([x1, y0], -1), np.stack([x1, y1], -1), np.stack([x0, y1], -1),
        ], 1)
    return None


class LineTokenizer:
    """ Per line token ids + per token character fractions along the line, w/ a token length cache. """

    def __init__(self, tokenizer: Callable):
        self.tokenizer = tokenizer
        self._len_cache: Dict[int, int] = {}

    def _token_len(self, t: int) -> int:
        n = self._len_cache.get(t)
        if n is None:
            n = len(self.tokenizer.decode([t]))
            self._len_cache[t] = n
        return n

    def __call__(self, lines: Sequence[str]) -> Tuple[List[int], List[int], np.ndarray]:
        """ Returns (token ids, line index per token, [T, 2] start / end fraction of the line per token). """
        ids, line_idx, spans = [], [], []
        for li, line in enumerate(lines):
            line_ids = self.tokenizer(line, add_special_tokens=False).input_ids
            if not line_ids:
                continue
            lens = np.array([self._token_len(t) for t in line_ids], dtype=np.float32)
            ends = np.cumsum(lens)
            total = max(ends[-1], 1.)
            spans.append(np.stack([(ends - lens) / total, ends / total], -1))
            ids += line_ids
            line_idx += [li] * len(line_ids)
        spans = np.concatenate(spans) if spans else np.zeros((0, 2), dtype=np.float32)
        return ids, line_idx, spans


def token_polys(polys: np.ndarray, line_idx: Sequence[int], spans: np.ndarray) -> np.ndarray:
    """ Slice line polys [L, 4, 2] into token polys [T, 4, 2] by start / end fractions along the top & bottom edges. """
    p = polys[np.asarray(line_idx, dtype=np.int64)]
    f0, f1 = spans[:, 0, None], spans[:, 1, None]
    tl, tr, br, bl = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    top, bot = tr - tl, br - bl
    return np.stack([tl + f0 * top, tl + f1 * top, bl + f1 * bot, bl + f0 * bot], 1)


def _in_quads(pts: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """ pts [T, C, 2] inside (convex, either winding) quads [T, 4, 2] -> [T, C] bool. """
    e = np.roll(quads, -1, axis=1) - quads  # [T, 4, 2]
    d = pts[:, :, None, :] - quads[:, None, :, :]  # [T, C, 4, 2]
    cross = e[:, None, :, 0] * d[..., 1] - e[:, None, :, 1] * d[..., 0]  # [T, C, 4]
    return (cross >= 0).all(-1) | (cross <= 0).all(-1)


def patch_labels(
        polys: np.ndarray,
        token_ids: Sequence[int],
        line_polys: np.ndarray,
        patch_coord: torch.Tensor,
        patch_size: int,
        cfg: PatchLabelCfg,
        blank_id: int,
        rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Fixed size patch labels for a patchified page.

    Args:
        polys: token polys [T, 4, 2] in pixels of the patchified image
        token_ids: [T] token id per poly
        line_polys: line polys [L, 4, 2] in pixels (blank patches don't touch any line)
        patch_coord: [N, 2] (y, x) patch grid coords, patch (y, x) covers pixels [y * P, (y + 1) * P) x ...
        patch_size: P
        cfg: label config
        blank_id: blank patch label (< 0 for no blank labels)
        rng: numpy generator for sampling

    Returns:
        patch_label_index [K] (index into the patch sequence, 0 for unused), patch_label [K] (ignore_id unused)
    """
    K = cfg.num_labels
    coord = patch_coord.numpy() if isinstance(patch_coord, torch.Tensor) else np.asarray(patch_coord)
    gh, gw = int(coord[:, 0].max()) + 1, int(coord[:, 1].max()) + 1
    index_map = np.full((gh, gw), -1, dtype=np.int64)
    index_map[coord[:, 0], coord[:, 1]] = np.arange(len(coord))
    label_map = np.full((gh, gw), -1, dtype=np.int64)
    touched = np.zeros((gh, gw), dtype=bool)
    ps = float(patch_size)

    if len(polys):
        # candidate patch centers in each token poly's bounding box, (c + 0.5) * P in [x0, x1]
        lo = np.ceil(polys.min(1) / ps - 0.5).astype(np.int64)  # [T, 2] (x, y)
        hi = np.floor(polys.max(1) / ps - 0.5).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array([gw - 1, gh - 1]))
        ext = np.maximum(hi - lo + 1, 0)
        keep = (ext > 0).all(1)
        if keep.any():
            mx, my = int(ext[keep, 0].max()), int(ext[keep, 1].max())
            ox, oy = np.meshgrid(np.arange(mx), np.arange(my))
            ox, oy = ox.reshape(-1), oy.reshape(-1)
            cx = lo[keep, 0, None] + ox[None]  # [T', C]
            cy = lo[keep, 1, None] + oy[None]
            in_box = (cx <= hi[keep, 0, None]) & (cy <= hi[keep, 1, None])
            pts = np.stack([(cx + 0.5) * ps, (cy + 0.5) * ps], -1)
            inside = _in_quads(pts, polys[keep]) & in_box
            ids = np.asarray(token_ids, dtype=np.int64)[keep]
            t_idx, c_idx = np.nonzero(inside)
            label_map[cy[t_idx, c_idx], cx[t_idx, c_idx]] = ids[t_idx]  # overlaps, last token wins

    if len(line_polys):
        # patches touching a line's bounding box are never blank
        x0y0 = np.floor(line_polys.min(1) / ps).astype(np.int64)
        x1y1 = np.floor(line_polys.max(1) / ps).astype(np.int64)
        x0y0 = np.clip(x0y0, 0, [gw - 1, gh - 1])
        x1y1 = np.clip(x1y1, 0, [gw - 1, gh - 1])
        for (x0, y0), (x1, y1) in zip(x0y0, x1y1):
            touched[y0:y1 + 1, x0:x1 + 1] = True

    valid = index_map >= 0
    text = np.nonzero(valid & (label_map >= 0))
    num_blank = int(round(K * cfg.blank_frac)) if blank_id >= 0 else 0
    num_text = min(len(text[0]), K - num_blank)
    sel = rng.choice(len(text[0]), num_text, replace=False) if num_text else np.zeros(0, dtype=np.int64)
    out_index = [index_map[text[0][sel], text[1][sel]]]
    out_label = [label_map[text[0][sel], text[1][sel]]]
    if num_blank:
        blank = np.nonzero(valid & ~touched & (label_map < 0))
        num_blank = min(len(blank[0]), num_blank)
        sel = rng.choice(len(blank[0]), num_blank, replace=False) if num_blank else np.zeros(0, dtype=np.int64)
        out_index.append(index_map[blank[0][sel], blank[1][sel]])
        out_label.append(np.full(num_blank, blank_id, dtype=np.int64))

    index = torch.zeros(K, dtype=torch.long)
    label = torch.full((K,), cfg.ignore_id, dtype=torch.long)
    out_index, out_label = np.concatenate(out_index), np.concatenate(out_label)
    index[:len(out_index)] = torch.from_numpy(out_index)
    label[:len(out_label)] = torch.from_numpy(out_label)
    return index, label


def empty_patch_labels(cfg: PatchLabelCfg) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(cfg.num_labels, dtype=torch.long), torch.full((cfg.num_labels,), cfg.ignore_id, dtype=torch.long)
