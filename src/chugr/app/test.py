""" Iterate a document loader for throughput benchmarking & sample inspection.

Example:
    python -m chugr.app.test --data.source "idl-train-0000{0..3}.tar" --data.batch-size 8 --data.num-samples 1000 \\
        --input.transform-type doc_naflex --input.patch-size 16 --input.max-seq-len 4096 --task doc_read
"""
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

import chugr
from chugr.common import DataArg, DataCfg, ImageInputCfg, Serializable, parse_args

_logger = logging.getLogger('chugr.test')


@dataclass
class TestArgs(Serializable):
    data: DataArg = field(default_factory=lambda: DataArg(source=''))
    input: ImageInputCfg = field(default_factory=lambda: ImageInputCfg(transform_type='doc_better', size=(1024, 768)))
    task: str = 'doc_read'  # one of 'doc_read', 'doc_vqa'
    tokenizer: Optional[str] = None  # HF tokenizer name, text left as strings if not set
    text_max_length: int = 1024
    is_training: bool = True
    num_batches: int = 50
    dump_dir: Optional[str] = None  # write first batch images (un-normalized) here


def _dump_image(x, input_cfg: ImageInputCfg, path: str):
    from torchvision.utils import save_image
    if isinstance(x, dict):
        # reassemble naflex patches into an image
        valid = x['patch_valid']
        patches = x['patches'][valid]
        coords = x['patch_coord'][valid]
        p = input_cfg.patch_size
        c = patches.shape[-1] // (p * p)
        h, w = (coords.max(0).values + 1).tolist()
        img = torch.zeros(c, h * p, w * p)
        for patch, (py, px) in zip(patches, coords.tolist()):
            img[:, py * p:(py + 1) * p, px * p:(px + 1) * p] = patch.reshape(p, p, c).permute(2, 0, 1)
        x = img
    mean = torch.tensor(input_cfg.mean).view(-1, 1, 1)
    std = torch.tensor(input_cfg.std).view(-1, 1, 1)
    save_image((x * std + mean).clamp(0, 1), path)


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    args: TestArgs = parse_args(TestArgs, argv=argv, description='chugr loader test')
    print(args.dumps_yaml())

    img_fn = chugr.create_image_preprocessor(args.input, is_training=args.is_training)
    txt_fn = None
    if args.tokenizer:
        txt_fn = chugr.create_text_preprocessor(args.tokenizer, max_length=args.text_max_length)
    if args.task == 'doc_read':
        task_cfg = chugr.DataTaskDocReadCfg(
            image_process_fn=img_fn,
            text_process_fn=txt_fn,
            text_target_feat=chugr.FeatureInfo('text_target', input_key=None) if txt_fn else None,
            error_handler='dump_and_reraise',
        )
    elif args.task == 'doc_vqa':
        task_cfg = chugr.DataTaskDocVqaCfg(
            image_process_fn=img_fn,
            text_process_fn=txt_fn,
            text_target_feat=chugr.FeatureInfo('text_target', input_key=None) if txt_fn else None,
        )
    else:
        assert False, f'Unknown task {args.task}'

    loader = chugr.create_loader(DataCfg.from_arg(args.data), task_cfg, is_training=args.is_training)
    start = time.time()
    num_samples = 0
    for i, batch in enumerate(loader):
        image = batch['image_input']
        bs = (image['patches'] if isinstance(image, dict) else image).shape[0]
        num_samples += bs
        if i == 0:
            if isinstance(image, dict):
                _logger.info(f"patches {tuple(image['patches'].shape)}, valid {image['patch_valid'].sum(1).tolist()}")
            else:
                _logger.info(f'image {tuple(image.shape)}')
            if args.dump_dir:
                os.makedirs(args.dump_dir, exist_ok=True)
                for j in range(bs):
                    x = {k: v[j] for k, v in image.items()} if isinstance(image, dict) else image[j]
                    _dump_image(x, args.input, os.path.join(args.dump_dir, f'sample_{j}.png'))
                    text = batch['text_input'][j]
                    with open(os.path.join(args.dump_dir, f'sample_{j}.txt'), 'w') as f:
                        f.write(text if isinstance(text, str) else str(text))
        if i + 1 >= args.num_batches:
            break
    elapsed = time.time() - start
    _logger.info(f'{num_samples} samples in {elapsed:.1f}s, {num_samples / elapsed:.1f} samples/s')


if __name__ == '__main__':
    sys.exit(main())
