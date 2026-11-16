""" Synthetic text rendering for reading warmup.

Renders (real) document text lines w/ varied fonts, sizes, and layouts. The training target is exactly the rendered
text, so rendered pages need no layout annotations (bboxes), only the text.
"""
import glob
import io
import logging
import os
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

_logger = logging.getLogger(__name__)

_FONT_DIRS = ('/usr/share/fonts', '/usr/local/share/fonts', '~/.fonts', '~/.local/share/fonts')

# Latin text fonts (excludes symbol, dingbat, script, math, CJK fallback fonts)
_DEFAULT_FONT_RE = (
    r'^(DejaVu(Sans|Serif)(Condensed|Mono)?(-(Bold|Oblique|Italic|BoldOblique|BoldItalic))?'
    r'|Liberation(Sans|Serif|Mono|SansNarrow)-.*'
    r'|Free(Sans|Serif|Mono).*'
    r'|(C059|P052|NimbusSans|NimbusSansNarrow|NimbusRoman|NimbusMonoPS|URWBookman|URWGothic)-.*'
    r')\.(ttf|otf)$'
)

# printable latin-1 (IDL OCR text is ~all latin-1), other chars are dropped from rendered text + target
_CHARSET = set(chr(c) for c in range(0x20, 0x7f)) | set(chr(c) for c in range(0xa1, 0x100))


@dataclass
class SynthTextCfg:
    font_paths: Optional[List[str]] = None  # explicit font files, default: discover system fonts (font_pattern)
    font_pattern: str = _DEFAULT_FONT_RE  # regex on font file basename for discovery
    lines_range: Tuple[int, int] = (1, 48)  # rendered lines per sample (after wrapping), log-uniform
    max_chars: int = 3000  # cap on rendered characters (keep the target within the text length budget)
    font_size_range: Tuple[int, int] = (16, 44)  # px
    width_range: Tuple[int, int] = (500, 1600)  # px, text block width
    line_spacing_range: Tuple[float, float] = (1.0, 1.6)  # x font size
    margin_range: Tuple[float, float] = (0.02, 0.12)  # fraction of text block width
    page_prob: float = 0.3  # place the text block on a letter aspect page (vs a tight crop around the text)
    paragraph_gap_prob: float = 0.1  # per line chance of an extra blank gap before it
    center_prob: float = 0.08  # per line chance of centered (vs left aligned) line
    blur_prob: float = 0.2
    noise_prob: float = 0.2
    jpeg_prob: float = 0.2


def discover_fonts(pattern: str = _DEFAULT_FONT_RE, dirs: Sequence[str] = _FONT_DIRS) -> List[str]:
    regex = re.compile(pattern)
    found = {}
    for d in dirs:
        for path in glob.glob(os.path.join(os.path.expanduser(d), '**', '*.*'), recursive=True):
            name = os.path.basename(path)
            if regex.match(name) and name not in found:
                found[name] = path
    return sorted(found.values())


def _check_font(path: str) -> bool:
    """ Font loads and (if fontTools is available) covers the charset. """
    try:
        ImageFont.truetype(path, 20)
    except Exception:
        return False
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return True
    try:
        cmap = TTFont(path, lazy=True).getBestCmap() or {}
    except Exception:
        return False
    return all(ord(c) in cmap for c in _CHARSET if c != ' ')


@lru_cache(maxsize=512)
def _load_font(path: str, size: int):
    return ImageFont.truetype(path, size)


def _wrap_line(text: str, font, width: int) -> List[str]:
    """ Greedy word wrap to `width` px, words longer than the width are split by character. """
    if font.getlength(text) <= width:
        return [text]
    out = []
    current = ''
    for word in text.split(' '):
        candidate = word if not current else current + ' ' + word
        if font.getlength(candidate) <= width:
            current = candidate
            continue
        if current:
            out.append(current)
        while font.getlength(word) > width and len(word) > 1:
            n = len(word)
            while n > 1 and font.getlength(word[:n]) > width:
                n -= 1
            out.append(word[:n])
            word = word[n:]
        current = word
    if current:
        out.append(current)
    return out


def clean_text_lines(lines: Sequence[str]) -> List[str]:
    out = []
    for line in lines:
        line = ''.join(c for c in line.replace('\t', ' ') if c in _CHARSET)
        line = ' '.join(line.split())
        if line:
            out.append(line)
    return out


class SynthTextRenderer:
    """ Render text lines to a grayscale document image, returns (image, rendered lines). """

    def __init__(self, cfg: Optional[SynthTextCfg] = None):
        self.cfg = cfg = cfg or SynthTextCfg()
        paths = cfg.font_paths or discover_fonts(cfg.font_pattern)
        self.fonts = [p for p in paths if _check_font(p)]
        assert self.fonts, f'No usable fonts found (pattern {cfg.font_pattern}).'
        _logger.info(f'SynthTextRenderer using {len(self.fonts)} fonts.')

    def __call__(self, lines: Sequence[str], rng: Optional[random.Random] = None, return_boxes: bool = False):
        """ Render a span of `lines`, returns (image, rendered lines) or w/ return_boxes (image, rendered lines, line
        ink boxes [L, 4] (x0, y0, x1, y1) in image pixels). """
        cfg = self.cfg
        rng = rng or random
        lines = clean_text_lines(lines)
        if not lines:
            raise RuntimeError('No renderable text.')

        font_path = rng.choice(self.fonts)
        font_size = rng.randint(*cfg.font_size_range)
        font = _load_font(font_path, font_size)
        block_w = rng.randint(*cfg.width_range)
        lo, hi = cfg.lines_range
        num_lines = int(round(lo * (hi / lo) ** rng.random())) if hi > lo else lo

        # contiguous span of source lines, wrapped to the block width
        start = rng.randrange(len(lines))
        rendered = []
        line_center = []
        line_gap = []
        num_chars = 0
        for src in lines[start:]:
            center = rng.random() < cfg.center_prob
            gap = rng.random() < cfg.paragraph_gap_prob
            for i, wrapped in enumerate(_wrap_line(src, font, block_w)):
                if len(rendered) >= num_lines or (rendered and num_chars + len(wrapped) > cfg.max_chars):
                    break
                rendered.append(wrapped)
                line_center.append(center)
                line_gap.append(gap and i == 0 and len(rendered) > 1)
                num_chars += len(wrapped) + 1
            else:
                continue
            break

        line_h = int(font_size * rng.uniform(*cfg.line_spacing_range))
        text_w = int(max(font.getlength(l) for l in rendered))
        text_h = line_h * (len(rendered) + sum(line_gap))
        if not any(line_center):
            block_w = max(text_w, min(block_w, text_w + font_size * 4))  # tight-ish block for left aligned text
        margin_x = int(block_w * rng.uniform(*cfg.margin_range)) + 2
        margin_y = int(block_w * rng.uniform(*cfg.margin_range)) + 2
        canvas_w = block_w + 2 * margin_x
        canvas_h = text_h + 2 * margin_y
        offset_x, offset_y = margin_x, margin_y
        if rng.random() < cfg.page_prob:
            # letter-ish page, text block placed at a random position
            page_w = max(canvas_w, int(canvas_w * rng.uniform(1.0, 1.4)))
            page_h = max(canvas_h, int(page_w * rng.uniform(1.25, 1.45)))
            offset_x += rng.randint(0, page_w - canvas_w)
            offset_y += rng.randint(0, page_h - canvas_h)
            canvas_w, canvas_h = page_w, page_h

        bg = rng.randint(215, 255)
        ink = rng.randint(0, min(80, bg - 120))
        img = Image.new('L', (canvas_w, canvas_h), color=bg)
        draw = ImageDraw.Draw(img)
        y = offset_y
        boxes = []
        for text, center, gap in zip(rendered, line_center, line_gap):
            if gap:
                y += line_h
            x = offset_x
            if center:
                x += max(0, (block_w - int(font.getlength(text))) // 2)
            draw.text((x, y), text, font=font, fill=ink)
            if return_boxes:
                boxes.append(draw.textbbox((x, y), text, font=font))
            y += line_h

        if rng.random() < cfg.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.0)))
        if rng.random() < cfg.noise_prob:
            img = _add_noise(img, rng.uniform(3, 12), rng)
        if rng.random() < cfg.jpeg_prob:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=rng.randint(30, 90))
            img = Image.open(io.BytesIO(buf.getvalue())).convert('L')
        if return_boxes:
            return img, rendered, boxes
        return img, rendered


def _add_noise(img: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    import numpy as np
    arr = np.asarray(img, dtype=np.float32)
    arr = arr + np.random.default_rng(rng.randrange(2 ** 31)).normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode='L')
