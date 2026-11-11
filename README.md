# Chugging Data (`chugr`)

Data pipelines for training multi-modal models on documents and image + text pairs, built on
[`webdataset`](https://github.com/webdataset/webdataset) and Hugging Face [`datasets`](https://github.com/huggingface/datasets).

* `webdataset` tar shards for scalable (streamed, sharded) pretraining, Hugging Face `datasets` (map-style or streaming)
  for exploration, validation and fine-tuning, behind one config driven `create_loader()`
* On the fly PDF decoding and rendering, so large collections of native `.pdf` files can be streamed w/o pre-rendering
  pages to `.png` / `.tiff`. The PDF decoder works at the webdataset level and can be plugged into other pipelines
* Document image transforms (incl. an `albumentations` port of the `nougat` augmentations) and NaFlex (native aspect,
  variable resolution) patch sequence transforms
* NaFlex batching: per batch sequence lengths sized to a token budget, w/ a rank-identical schedule. This padded,
  per sample scheme is the focus for variable aspect / size images, not NaViT style multi-image sequence packing
* Task pipelines for document reading (OCR targets), document VQA and image-text, w/ tokenization and prompt handling
* Interval based training: fixed sample counts per interval, the same number of batches on every rank, shard lists that
  repeat (reshuffled per pass) or are resampled, distributed eval w/o padded or repeated samples

`chugr` is the continuation of the original `chug` library (huggingface/chug), renamed and revived. It provides the
data pipelines for [`pixr`](https://github.com/rwightman/pixr).

## Installation

```bash
pip install --pre chugr
# optional extras: albumentations based transforms (doc_nougat), HF tokenizers by name + synthetic text font checks
pip install --pre "chugr[alb,text]"
```

PDF rendering uses [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) by default.
[fitz / PyMuPDF](https://github.com/pymupdf/PyMuPDF) can be used instead if your use case is okay with its AGPL-3.0
license, it must be installed separately and enabled w/ `CHUGR_USE_AGPL_PYMUPDF=1`.

## Status

Alpha. Major features work and have been used for document reading pretraining and DocVQA fine-tuning, expect breaking
changes. `pip install --pre chugr` installs the current dev version.

### Changes since `chug`
* `simple_parsing` dependency replaced by a minimal `Serializable` mixin and nested dataclass CLI parser (`chugr.common.parse_args`)
* NaFlex (native aspect, variable resolution) document transforms (`doc_naflex`) producing patch sequences + coords, padded by `chugr.common.collate`
* Fixes for `doc_basic` / `doc_better` train transforms, albumentations 2.x support for `doc_nougat`
* HF datasets loader: shuffled & sharded `IntervalSampler` for training (w/ intervals shorter than an epoch), split slicing (`train[:1000]`)
* `python -m chugr.app.test` to benchmark loaders and dump (reassembled) samples
* NaFlex batching (`chugr.naflex`, adapted from OpenCLIP's streaming NaFlex loader): a rank-identical schedule of
  `(seq_len, batch_size)` per batch drawn from `naflex_seq_lens` / `naflex_seq_len_probs` and sized to a token budget
  (`naflex_max_tokens`, w/ optional per row `naflex_text_tokens` cost), samples are resized to each batch's patch
  budget. `length_bucketing` pool sorts samples by target text length. `mean_batch_size` is exposed on the loader for
  loss scaling.
* Webdataset shard lists repeat (reshuffled per pass) instead of requiring `resampled`, so fixed sample count
  intervals and NaFlex batching work w/o sampling shards w/ replacement
* NaFlex batching for HF datasets. Map-style (`format hfds`, `NaFlexMapDataset` adapted from OpenCLIP's
  `NaFlexMapDatasetWrapper`): the same schedule over rank sharded shuffled indices that continue through the dataset
  across intervals, failed samples are replaced. Streaming (`hfids`): `NaFlexBatcher` over the repeating stream.
* Streaming HF datasets yield a fixed number of train batches per worker (the stream wraps around), so every rank
  sees the same number of batches per interval, as the webdataset loader. `num_workers` is limited to the number of
  data shards (files) per rank. The dataset epoch (shard / shuffle seed) is set from the main process only.
* Ranks are split w/ the explicit `DistributedCfg` rank / world size (no reliance on `RANK` / `WORLD_SIZE` env vars
  in dataloader workers), per rank seeds are spaced by a large stride (`chugr.common.rank_seed`) so runs w/ nearby
  seeds don't share streams.
* Eval splits are exact: map-style HF datasets are split across ranks w/o padding or repeated samples (uneven per rank
  counts), uncapped streaming eval reads every sample once.
* Webdataset eval: `num_samples` below the dataset size caps the eval set to exactly the first `num_samples` samples
  for any number of ranks / workers (every worker reads that head of the stream, keep it small). W/o a cap all
  samples are used, split by shard if there are at least `ranks * workers` shards, otherwise by sample.
* Text preprocessing w/ a `prompt_end_token`: an over-long prompt (question) is shortened so the prompt end token
  and target (answer) still fit in `max_length`.

## Roadmap

### Nearish
* Cleanup and refinement, codebase will change
* Documentation & unit-tests
* Support reading of info .json/.yaml files for automatic shard info resolution for webdatasets (like timm)

### Mediumish
* Option to output bbox annotations for lines (or word + word output) for tasks that leverage layout
* Unified preprocessor functions for combined image + text tokenization (img+text token interleaving, etc.)

### Longish 
* Increase range of task pipelines for other tasks, modelling needs
* Support additional modalities & targets (video, audio, detection/dense pixel targets, image/video/audio targets)
* Explore alternatives to .tar shards (array_record, arrow, etc)

## Design

### Submodule Hierarchy

The library has been designed so that functions, classes at different levels can be used independently.

If one wants to build a loader & pipeline with JSON/YAML serializable configs, use the top-level `chugr.create_loader()` in `chugr/loader.py`. Depending on dataset sources, one can easily switch this between webdataset, HF datasets (in the future, other sources).

Bypassing the highest level, one can also call `build_pipeline_*` methods in `task_pipeline` and then call `create_loader_wds` with a full array of args for `wds` only use cases.

If one doesn't want to use `chugr` loaders and pipelines at all, `image`, `text`, and `wds` (especially decoder) functionality may be useful in other projects.

#### Library modules (highest to lowest level)

The dependencies of modules within the library are intended to follow the hierarchy below. e.g. doc depends on wds, but wds should never depend on doc.

```
app
|
loader (chugr/loader.py)
|
task_pipeline
|
doc
|
wds, hfds, image, text
|
common
```

### Submodules

#### `common`

Configs, structures (dataclasses) for general use across the library

#### `wds`

Webdataset (`wds` for short) specific code. Extensions and alterations of webdataset functionality to fit covered use case and improve robustness.

All data pipelines in `chugr` currently leverage `wds` pipelines, even when not using `wds` datasets. 

Document oriented decoding (pdf decoder) is present in `chugr/wds/decode.py`, it can be used with any webdataset pipeline as a decoder. e.g. `wds.decode(chugr.wds.DecodeDoc('pill'), 'pill')`

#### `hfds`

Hugging Face `datasets` support. A minimal wrapper that allows `datasets` to be used with chugr processing pipelines. 

The processing pipelines remain webdataset based when using `datasets`, they are invoked by a custom collate class.

#### `image`

Image processing, `torchvision` and `albumentations` based transform building code. A mix of generic image (imagenet, simclr) transforms and document specific transforms, including an implementation of `albumentations` based `nougat` transforms.

#### `text`

Text processing, tokenization code.

#### `doc`

Document processing code. Currently focused on processors that apply image/pdf decoders and process document OCR or VQA annotations.

#### `task_pipeline`

Task specific pipelines, where dataset formats meet modelling needs. 

Inputs to task pipelines are sample dictionaries based on the dataset form, they are decoded and then processed into outputs that match model input requirements.

Task specific pipelines that handle the data <--> model input interface are inserted into an encompassing data pipeline which handles shard lists, shuffle, wrapping, distributed worker, splitting, batching, etc.

#### `naflex`

NaFlex (variable sequence length) batching helpers: `NaFlexBatchScheduler` (rank-identical seq len / batch size
schedule from a token budget), `NaFlexBatcher` (per seq len processors + batching of a sample stream),
`NaFlexMapDataset` (the same over a map-style dataset), `LengthBucketer`.

#### `chugr.loader`

This lone top-level file includes the main factory methods for creating loaders w/ associated pipelines from config dataclasses.

#### `app`

Most applications using `chugr` will exist outside of the lib in training libraries, etc. Some builtin utility / exploration apps will be included here.

## Datasets

Datasets that work well with this library can be found on the Hugging Face Hub under the `pixparse` organization (https://huggingface.co/pixparse).

We'll add links to other noteworthy datasets that can be used as we become aware of them.


## Usage / Examples

### Document Reading, Training w/ IDL
```python
import chugr
img_cfg = chugr.ImageInputCfg(size=(1024, 768), transform_type='doc_better')
img_fn = chugr.create_image_preprocessor(input_cfg=img_cfg, is_training=True)
txt_fn = chugr.create_text_preprocessor(
    'naver-clova-ix/donut-base',
    prompt_end_token='<s_idl>',
    task_start_token='<s_idl>',  # NOTE needs to be added to tokenizer
)

task_cfg = chugr.DataTaskDocReadCfg(
    image_process_fn=img_fn,
    text_process_fn=txt_fn,
    page_sampling='random',
    error_handler='dump_and_reraise',
)
data_cfg = chugr.DataCfg(
    source='pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/idl-wds/resolve/main/idl-train-0{0000..2999}.tar',
    batch_size=8,
    num_samples=3144726,
    format='wds',
)
lb = chugr.create_loader(
    data_cfg,
    task_cfg,
    is_training=True,
)
ii = iter(lb)
sample = next(ii)
```

### Document Reading, Exploring IDL
```python
import chugr
task_cfg = chugr.DataTaskDocReadCfg(page_sampling='all')
data_cfg = chugr.DataCfg(
    source='pixparse/idl-wds',
    split='train',
    batch_size=None,
    format='hfids',
    num_workers=0,    
)
lb = chugr.create_loader(
    data_cfg,
    task_cfg,
)
ii = iter(lb)
sample = next(ii)
```

### Document Reading, Training with PDFA

```python
import chugr
img_cfg = chugr.ImageInputCfg(size=(1024, 768), transform_type='doc_nougat')
img_fn = chugr.create_image_preprocessor(input_cfg=img_cfg, is_training=True)
txt_fn = chugr.create_text_preprocessor(
    'naver-clova-ix/donut-base',
    prompt_end_token='<s_pdfa>',
    task_start_token='<s_pdfa>',  # NOTE needs to be added to tokenizer
)

task_cfg = chugr.DataTaskDocReadCfg(
    image_process_fn=img_fn,
    text_process_fn=txt_fn,
    page_sampling='random',
)
data_cfg = chugr.DataCfg(
    source='pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/pdfa-english-train/resolve/main/pdfa-eng-train-{000000..005000}.tar',
    batch_size=8,
    num_samples=1000000,  # FIXME replace with actual
    format='wds',   
)
lb = chugr.create_loader(
    data_cfg,
    task_cfg,
    is_training=True,
)
ii = iter(lb)
sample = next(ii)
```

### Document Reading, Exploring PDFA

```python
import chugr

task_cfg = chugr.DataTaskDocReadCfg(
    page_sampling='all',
)
data_cfg = chugr.DataCfg(
    source='pixparse/pdfa-eng-wds',
    split='train',
    batch_size=None,
    format='hfids',
    num_workers=0,
)
lb = chugr.create_loader(
    data_cfg,
    task_cfg,
)
ii = iter(lb)
sample = next(ii)
```


### NaFlex (variable resolution) document input

```python
import chugr
from PIL import Image

# pages keep their aspect ratio, resized to fit a patch budget (max_seq_len patches of patch_size x patch_size)
img_cfg = chugr.ImageInputCfg(transform_type='doc_naflex', patch_size=16, max_seq_len=1024)
img_fn = chugr.create_image_preprocessor(input_cfg=img_cfg, is_training=True)
out = img_fn(Image.new('L', (850, 1100), 255))
# {'patches': [N, 256], 'patch_coord': [N, 2] (y, x), 'patch_valid': [N] bool}, N <= max_seq_len
```

For batches w/ a sequence length (patch budget) drawn per batch and the batch size set by a token budget, set
`naflex_seq_lens`, `naflex_seq_len_probs` and `naflex_max_tokens` in `DataCfg` (webdataset, map-style or streaming
HF datasets).

### Image + Text

#### Training

```python
import chugr
import transformers
from functools import partial
img_cfg = chugr.ImageInputCfg(size=(512, 512), transform_type='image_timm')
img_fn = chugr.create_image_preprocessor(input_cfg=img_cfg, is_training=True)
tokenizer = transformers.AutoTokenizer.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
txt_fn = partial(chugr.tokenize, max_length=1000, tokenizer=tokenizer)
task_cfg = chugr.DataTaskImageTextCfg(
    image_process_fn=img_fn,
    text_process_fn=txt_fn,
)
data_cfg = chugr.DataCfg(
    source='pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/cc12m-wds/resolve/main/cc12m-train-{0000..2175}.tar',
    batch_size=8,
    num_samples=10968539,
    format='wds',   
)
lb = chugr.create_loader(
    data_cfg,
    task_cfg,
    is_training=True,
)
ii = iter(lb)
sample = next(ii)
```

### Document VQA

#### Training, Fine-tuning
```python
import chugr
img_cfg = chugr.ImageInputCfg(size=(1024, 768), transform_type='doc_basic')
img_fn = chugr.create_image_preprocessor(img_cfg, is_training=True)
txt_fn = chugr.create_text_preprocessor(
    'naver-clova-ix/donut-base-finetuned-docvqa',
    prompt_end_token='<s_answer>',
    task_start_token='<s_docvqa>',
)

task_cfg = chugr.DataTaskDocVqaCfg(
    image_process_fn=img_fn,
    text_process_fn=txt_fn,
)
data_cfg = chugr.DataCfg(
    source='pipe:curl -s -f -L https://huggingface.co/datasets/pixparse/docvqa-wds/resolve/main/docvqa-train-{000..383}.tar',
    batch_size=8,
    format='wds',
    num_samples=39463,
)
lb = chugr.create_loader(
    data_cfg,
    task_cfg,
    is_training=True,
)
ii = iter(lb)
sample = next(ii)
```

#### Exploration

```python
import chugr
task_cfg = chugr.DataTaskDocVqaCfg(
    question_prefix='Question: ',
    question_suffix='',
    answer_prefix='Answer: ',
    answer_suffix=''
)
data_cfg = chugr.DataCfg(
    source='pixparse/docvqa-single-page-questions',
    split='validation',
    batch_size=None,
    format='hfids',
    num_workers=0,
)
lb = chugr.create_loader(
    data_cfg,
    task_cfg
)
ii = iter(lb)
sample = next(ii)
```

## Acknowledgement

`chugr` evolved from the `webdataset` datapipeline used successfully in the [OpenCLIP](https://github.com/mlfoundations/open_clip) project. Thanks to all the contributors in that project. Future work will likely involve closing the loop and leveraging `chugr` in OpenCLIP for increased capability.

The image/document augmentations in `chugr` rely on a number of external influences. Our document oriented `doc_better` torchvision augmentations are influenced by `nougat`, and the `doc_nougat` is a direct adaptation of the [`albumentations`](https://albumentations.ai/) + `cv2` document pipeline in [`nougat`](https://github.com/facebookresearch/nougat). Several image augmentations leverage existing work in the `timm` library.

Also, big thanks to the maintainers of [`webdataset`](https://github.com/webdataset/webdataset) and Hugging Face [`datasets`](https://github.com/huggingface/datasets).

## License

Apache 2.0, see [LICENSE](LICENSE). The optional PyMuPDF backend is AGPL-3.0 licensed (see above).
