import logging
from functools import partial
from typing import Callable, List, Optional, Union

import torch

_logger = logging.getLogger(__name__)
_num_prompts_truncated = 0


def prompt_end_pos(tokens: torch.Tensor, prompt_end_token_id: int, empty_default: int = 0) -> int:
    end_pos = torch.nonzero(tokens == prompt_end_token_id)
    return end_pos[-1].item() if end_pos.numel() > 0 else empty_default


def text_input_to_target(
        text_input_ids: torch.Tensor,
        tokenizer: Optional[Callable] = None,
        prompt_end_token: Optional[Union[str, int]] = None,
        ignore_id: int = -100,
        pad_token_id: Optional[int] = None,
):
    target = text_input_ids.clone()

    if pad_token_id is None:
        assert tokenizer is not None, 'tokenizer must be specified if pad_token_id is not.'
        pad_token_id = tokenizer.pad_token_id

    # model doesn't need to predict pad token
    target[target == pad_token_id] = ignore_id

    if prompt_end_token is not None:
        if isinstance(prompt_end_token, str):
            assert tokenizer is not None, 'tokenizer must be specified if prompt_end_token_id required.'
            prompt_end_token_id = tokenizer.convert_tokens_to_ids(prompt_end_token)
        else:
            prompt_end_token_id = prompt_end_token  # already an int

        # model doesn't need to predict prompt (for VQA)
        end_pos = prompt_end_pos(target, prompt_end_token_id, empty_default=-1)
        if end_pos < 0:
            # prompt end truncated away (prompt longer than max_length), nothing left to predict
            target[:] = ignore_id
        else:
            target[:end_pos + 1] = ignore_id

    return target


def tokenize(
        text: str,
        tokenizer: Callable,
        max_length: int,
        ids_only: bool = True,
):
    output = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    if ids_only:
        return output.input_ids[0]
    return output


def truncate_prompt(
        token_ids: List[int],
        max_length: int,
        prompt_end_token_id: int,
        special_token_ids=(),
) -> List[int]:
    """ Shorten the prompt of an over-long `prompt + target` token sequence so the prompt end token and the target
    still fit in `max_length`.

    Tokens are removed from the end of the prompt body, the prompt's leading special tokens (task start, question
    prefix) and trailing special tokens (question suffix, prompt end) are kept. Returns the ids unchanged if they
    fit or there is no prompt end token, and may still return more than `max_length` tokens if the target alone
    is too long (the caller truncates the tail).
    """
    over = len(token_ids) - max_length
    ends = [i for i, t in enumerate(token_ids) if t == prompt_end_token_id]
    if over <= 0 or not ends:
        return token_ids
    special = set(special_token_ids) | {prompt_end_token_id}
    body_end = ends[-1]
    while body_end > 0 and token_ids[body_end - 1] in special:
        body_end -= 1
    body_start = 0
    while body_start < body_end and token_ids[body_start] in special:
        body_start += 1
    cut = min(over, body_end - body_start)
    if cut <= 0:
        return token_ids
    return token_ids[:body_end - cut] + token_ids[body_end:]


def special_token_ids(tokenizer) -> set:
    """ Special & added token ids, for HF tokenizers and wrappers w/ a subset of the interface. """
    ids = set(getattr(tokenizer, 'all_special_ids', None) or ())
    get_added_vocab = getattr(tokenizer, 'get_added_vocab', None)
    if get_added_vocab is not None:
        ids |= set(get_added_vocab().values())
    return ids


def _tokenize_keep_prompt_end(
        text: str,
        text_input_ids: torch.Tensor,
        tokenizer: Callable,
        max_length: int,
        prompt_end_token: Union[str, int],
) -> torch.Tensor:
    """ Re-tokenize a (possibly) truncated `text_input_ids` w/ the prompt shortened instead of the tail cut off. """
    global _num_prompts_truncated
    pad_token_id = tokenizer.pad_token_id
    if bool((text_input_ids == pad_token_id).any()):
        return text_input_ids  # padded, nothing was truncated
    prompt_end_token_id = prompt_end_token
    if isinstance(prompt_end_token, str):
        prompt_end_token_id = tokenizer.convert_tokens_to_ids(prompt_end_token)
    special = special_token_ids(tokenizer)
    ends = torch.nonzero(text_input_ids == prompt_end_token_id)
    if len(ends) and all(int(t) in special for t in text_input_ids[:int(ends[0])]):
        # the prompt end survived truncation w/ only special tokens before it (e.g. a task start token as the
        # prompt end), no prompt body to shorten, skip re-tokenizing
        return text_input_ids
    full_ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(full_ids) <= max_length:
        return text_input_ids
    ids = truncate_prompt(full_ids, max_length, prompt_end_token_id, special)
    if len(ids) == len(full_ids):
        return text_input_ids  # no prompt (body) to shorten
    _num_prompts_truncated += 1
    n = _num_prompts_truncated
    if n == 10 ** (len(str(n)) - 1):  # log the 1st, 10th, 100th, ... (per process / dataloader worker)
        _logger.warning(
            f'Prompt shortened by {len(full_ids) - len(ids)} tokens to fit the target in max_length={max_length} '
            f'({n} prompts shortened so far in this process).')
    ids = ids[:max_length]  # target alone too long, the tail is still cut
    num_pad = max_length - len(ids)
    if num_pad:
        pad = [pad_token_id] * num_pad
        ids = pad + ids if getattr(tokenizer, 'padding_side', 'right') == 'left' else ids + pad
    return torch.tensor(ids, dtype=text_input_ids.dtype)


def prepare_text_input(
        text_input,
        tokenizer: Callable,
        max_length: int,
        task_start_token: Optional[str] = None,
        prompt_end_token: Optional[str] = None,
        add_eos_token: bool = True,
        ignore_id: int = -100,
        include_target: bool = True,
        return_dict: bool = True,
        input_key: str = "text_input",
        target_key: str = "text_target",
):
    """
    Simple data preprocessing for raw-text data.
    """
    if task_start_token:
        text_input = task_start_token + text_input

    if add_eos_token:
        text_input += tokenizer.eos_token

    text_input_ids = tokenize(text_input, tokenizer=tokenizer, max_length=max_length)
    if prompt_end_token is not None:
        # an over-long prompt (question) is shortened so the prompt end token & target (answer) aren't truncated away
        text_input_ids = _tokenize_keep_prompt_end(text_input, text_input_ids, tokenizer, max_length, prompt_end_token)

    if include_target:
        text_target_ids = text_input_to_target(text_input_ids, tokenizer, prompt_end_token, ignore_id)
        if return_dict:
            return {input_key: text_input_ids, target_key: text_target_ids}
        else:
            return text_input_ids, text_target_ids
    else:
        # FIXME calculate prompt end pos for validation use (target not needed)
        if return_dict:
            return {input_key: text_input_ids}
        else:
            return text_input_ids


def create_text_preprocessor(
        tokenizer: Union[str, Callable],
        max_length: int = 1024,
        task_start_token: Optional[str] = None,
        prompt_end_token: Optional[str] = None,
        ignore_id: int = -100,
        include_target: bool = True,
        return_dict: bool = True,
        input_key: str = "text_input",
        target_key: str = "text_target",
):
    if isinstance(tokenizer, str):
        import transformers
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)

    # FIXME just binding prepare_text_input fn for now
    #  This is currently tailored to the Donut style image enc + text decoder use case.
    #  More functionality / variety required.

    preprocess_fn = partial(
        prepare_text_input,
        tokenizer=tokenizer,
        max_length=max_length,
        task_start_token=task_start_token,
        prompt_end_token=prompt_end_token,
        ignore_id=ignore_id,
        include_target=include_target,
        return_dict=return_dict,
        input_key=input_key,
        target_key=target_key,
    )
    return preprocess_fn