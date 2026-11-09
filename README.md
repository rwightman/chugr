# Chugging Data (`chugr`)

Continuation of the original `chug` library (huggingface/chug), renamed and revived.

A library to help w/ efficient training for multi-modal data. Initially focused on image & document + text tasks.

`chugr` currently leverages `webdataset` and Hugging Face `datasets`.

`webdataset` tar files and dataset pipelines are preferred for scalable pretraining. 

Hugging Face `datasets` are supported and work great for exploration, validation, and fine-tune use cases.

`chugr` provides on the fly PDF decoding and rendering via either pypdfium2 (https://github.com/pypdfium2-team/pypdfium2) as a default, or fitz/pymupdf (https://github.com/pymupdf/PyMuPDF) if your use case is okay with their AGPL-3.0 license. `fitz` support must be manually enabled. The pdf handling is implemented at the webdataset level, so you can plug it in to other webdataset pipelines. This enables large scale sharded streaming of native .pdf files without needing to pre-render to .png/.tiff, etc.

## Status

This library is still a WIP, consider this an alpha release (pre announcement). Major features should be working, the library has been tested with several PDF datasets we will shortly make public. However, do expect there will still be breaking changes, lots of improvements, etc.

`pip install --pre chugr` will install the current dev version.

### TODOs

### Nearish
* Cleanup and refinement, codebase will change
* Documentation & unit-tests
* Support reading of info .json/.yaml files for automatic shard info resolution for webdatasets (like timm)

### Recent (revive branch)
* `simple_parsing` dependency replaced by a minimal `Serializable` mixin and nested dataclass CLI parser (`chugr.common.parse_args`)
* NaFlex (native aspect, variable resolution) document transforms (`doc_naflex`) producing patch sequences + coords, padded by `chugr.common.collate`
* Fixes for `doc_basic` / `doc_better` train transforms, albumentations 2.x support for `doc_nougat`
* HF datasets loader: shuffled & sharded `IntervalSampler` for training (w/ intervals shorter than an epoch), split slicing (`train[:1000]`)
* `python -m chugr.app.test` to benchmark loaders and dump (reassembled) samples

### Mediumish
* Option to output bbox annotations for lines (or word + word output) for tasks that leverage layout
* Unified preprocessor functions for combined image + text tokenization (img+text token interleaving, etc.)
* Image token (patch) packing ala NaViT. Online bin packing based algorithms integrated with image preprocessing and pipeline.

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

#### `chugr.loader`

This lone top-level file includes the main factory methods for creating loaders w/ associated pipelines from config dataclasses.

#### `app`

Most applications using `chugr` will exist outside of the lib in training libraries, etc. Some builtin utility / exploration apps will be included here.

## Concepts

WIP

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


### Image + Text

### Training

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
from chugr.task_pipeline import create_task_pipeline
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
from chugr.task_pipeline import create_task_pipeline
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

`chugr` evolve from the `webdataset` datapipeline used successfully in the [OpenCLIP](https://github.com/mlfoundations/open_clip) project. Thanks to all the contributors in that project. Future work will likely involve closing the loop and leveraging `chugr` in OpenCLIP for increased capability.

The image/document augmentations in `chugr` rely on a number of external influences. Our document oriented `doc_better` torchvision augmentations are influenced by `nougat`, and the `doc_nougat` is a direct adaptation of the [`albumentations`](https://albumentations.ai/) + `cv2` document pipeline in [`nougat`](https://github.com/facebookresearch/nougat). Several image augmentations leverage existing work in the `timm` library.

Also, big thanks to the maintainers of [`webdataset`](https://github.com/webdataset/webdataset) and Hugging Face [`datasets`](https://github.com/huggingface/datasets).
