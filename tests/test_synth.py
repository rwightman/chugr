import json
import random

import pytest
from PIL import Image

from chugr import FeatureInfo, ImageFeatureInfo
from chugr.doc import DocReadProcessor, SynthTextCfg, SynthTextRenderer
from chugr.doc.synth_text import _wrap_line, _load_font, clean_text_lines

LINES = [
    'BROWN & WILLIAMSON TOBACCO CORPORATION',
    'After evaluations and rejections of tar reduction samples 49B, 50B, and 51B we have reviewed the data',
    'Price sensitive € 12.50 ™',
    '',
    'averyveryveryverylongwordthatmustbesplitacrosslinesbecauseitisfartoolongforanywidth',
]


def test_clean_text_lines():
    assert clean_text_lines(['  a\tb  ', '', '€™', 'café → x']) == ['a b', 'café x']


def test_wrap_line_respects_width():
    renderer = SynthTextRenderer()
    font = _load_font(renderer.fonts[0], 24)
    wrapped = _wrap_line(LINES[1] + ' ' + LINES[4], font, 300)
    assert len(wrapped) > 1 and all(font.getlength(w) <= 300 for w in wrapped)
    assert ''.join(wrapped).replace(' ', '') == (LINES[1] + LINES[4]).replace(' ', '')


def test_renderer_output():
    renderer = SynthTextRenderer(SynthTextCfg(lines_range=(1, 8)))
    rng = random.Random(0)
    for _ in range(20):
        img, lines = renderer(LINES, rng=rng)
        assert isinstance(img, Image.Image) and img.mode == 'L'
        assert 1 <= len(lines) <= 8
        assert all(l and l == l.strip() for l in lines)
        assert all(c not in ''.join(lines) for c in '€™')


@pytest.mark.parametrize('squeeze', [True, False])
def test_doc_read_processor_synth(squeeze):
    proc = DocReadProcessor(
        image_process_fn=lambda img: img.size,
        text_process_fn=lambda text: {'text_input': 'IN:' + text, 'text_target': 'TGT:' + text},
        image_input_feat=ImageFeatureInfo('image_input', input_key='pdf;tif', image_mode='L'),
        text_input_feat=FeatureInfo('text_input', input_key='pages'),
        text_target_feat=FeatureInfo('text_target', input_key=None),
        squeeze_pages=squeeze,
        synth_prob=1.0,
    )
    sample = {'json': json.dumps({'pages': [{'text': []}, {'text': LINES}]}).encode(), '__key__': 'k', '__url__': 'u'}
    out = proc(sample)  # no pdf / tif present, synth path never decodes the page image
    parse = out['_parse']
    assert parse['synthetic'] and parse['page_indices'] == [1]
    text = '\n'.join(parse['page_text'][0])
    get = (lambda v: v) if squeeze else (lambda v: v[0])
    assert get(out['text_input']) == 'IN:' + text and get(out['text_target']) == 'TGT:' + text
    assert isinstance(get(out['image_input']), tuple)
