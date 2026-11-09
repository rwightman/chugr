""" Nested dataclass <-> command line arguments

A small replacement for the subset of `simple_parsing` used by chug & pixparse apps.

* nested dataclass fields map to dotted args, e.g. `--task.opt.learning-rate 1e-4`
  (dashes and underscores are both accepted)
* `subgroups` select the dataclass type of a field by name, e.g. `--task cruller_pretrain`
* a `--config` yaml/json file provides defaults that command line args override
"""
import argparse
import dataclasses
import sys
import typing
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar

from .serialize import _is_union, _unwrap_optional, from_dict, resolve_field_types, _load_file

T = TypeVar('T')

_NONE_STRINGS = ('none', 'null')


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    lv = v.lower()
    if lv in ('true', 't', 'yes', 'y', '1'):
        return True
    if lv in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}.')


def _scalar_parser(tp):
    def _parse(v: str):
        if v.lower() in _NONE_STRINGS:
            return None
        if tp is bool:
            return str2bool(v)
        if tp is int:
            return int(v)
        if tp is float:
            return float(v)
        if _is_union(tp):
            # e.g. Union[int, str] -> try members in order
            for a in typing.get_args(tp):
                if a is type(None):
                    continue
                try:
                    return _scalar_parser(a)(v)
                except (ValueError, argparse.ArgumentTypeError):
                    continue
            return v
        return v
    return _parse


def _arg_spec(tp):
    """ Return (nargs, item parser) for a field type. """
    tp = _unwrap_optional(tp)
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return '+', _scalar_parser(args[0])
        item_tp = args[0] if args else str
        return (len(args) if args else '+'), _scalar_parser(item_tp)
    if origin is list:
        return '+', _scalar_parser(args[0] if args else str)
    if _is_union(tp):
        # union containing a sequence type, accept one or more values
        for a in args:
            if typing.get_origin(a) in (list, tuple):
                return '+', _scalar_parser(typing.get_args(a)[0] if typing.get_args(a) else str)
    return None, _scalar_parser(tp)


def _flatten_fields(cls, prefix: str = '') -> Dict[str, Any]:
    """ dotted name -> field type for all leaf fields of a (nested) dataclass type """
    out = {}
    field_types = resolve_field_types(cls)
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        tp = field_types[f.name]
        name = f'{prefix}{f.name}'
        inner = _unwrap_optional(tp)
        if dataclasses.is_dataclass(inner):
            out.update(_flatten_fields(inner, prefix=name + '.'))
        else:
            out[name] = tp
    return out


def _set_nested(d: Dict[str, Any], dotted: str, value):
    keys = dotted.split('.')
    for k in keys[:-1]:
        if d.get(k) is None:
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def _norm(name: str) -> str:
    return name.replace('-', '_')


def parse_args(
        cls: Type[T],
        argv: Optional[Sequence[str]] = None,
        subgroups: Optional[Dict[str, Dict[str, type]]] = None,
        subgroup_defaults: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
) -> T:
    """ Parse command line args into dataclass `cls`.

    Args:
        cls: The top level config dataclass.
        argv: Arguments, defaults to sys.argv[1:].
        subgroups: Mapping of dotted field name -> {choice name: dataclass type}. The choice is selected
            with `--<field> <name>` and the selected type's fields become `--<field>.<subfield>` args.
        subgroup_defaults: Default choice name per subgroup field.
        description: Parser description.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    subgroups = subgroups or {}
    subgroup_defaults = subgroup_defaults or {}

    # stage 1: config file + subgroup selection
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    for sg_name, choices in subgroups.items():
        pre.add_argument(
            f'--{sg_name.replace("_", "-")}', f'--{sg_name}', dest=sg_name, type=str,
            default=None, choices=list(choices.keys()))
    pre_args, _ = pre.parse_known_args(argv)

    file_cfg = _load_file(pre_args.config) if pre_args.config else {}

    selected = {}
    for sg_name, choices in subgroups.items():
        choice = getattr(pre_args, sg_name)
        if choice is None:
            # allow config file to select via a `_name` key in the subgroup dict
            sg_file = file_cfg.get(sg_name)
            if isinstance(sg_file, dict) and '_name' in sg_file:
                choice = sg_file.pop('_name')
            else:
                choice = subgroup_defaults.get(sg_name)
        assert choice is not None, f'A value for --{sg_name} is required (one of {list(choices.keys())}).'
        selected[sg_name] = choice

    # stage 2: build full parser over all leaf fields
    leaf_types = {}
    field_types = resolve_field_types(cls)
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.name in subgroups:
            sub_cls = subgroups[f.name][selected[f.name]]
            leaf_types.update(_flatten_fields(sub_cls, prefix=f.name + '.'))
            continue
        tp = field_types[f.name]
        inner = _unwrap_optional(tp)
        if dataclasses.is_dataclass(inner):
            leaf_types.update(_flatten_fields(inner, prefix=f.name + '.'))
        else:
            leaf_types[f.name] = tp

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--config', type=str, default=None, help='yaml or json config file')
    for sg_name, choices in subgroups.items():
        parser.add_argument(
            f'--{sg_name.replace("_", "-")}', f'--{sg_name}', dest=f'__sg_{sg_name}', type=str,
            default=None, choices=list(choices.keys()))

    for dotted, tp in leaf_types.items():
        nargs, item_parser = _arg_spec(tp)
        flags = {f'--{dotted}', f'--{dotted.replace("_", "-")}'}
        kwargs = dict(dest=dotted, type=item_parser, default=argparse.SUPPRESS)
        if nargs is not None:
            kwargs['nargs'] = nargs
        parser.add_argument(*sorted(flags), **kwargs)

    args = parser.parse_args(argv)

    # merge: file config < command line
    merged: Dict[str, Any] = dict(file_cfg)
    for dotted, value in vars(args).items():
        if dotted == 'config' or dotted.startswith('__sg_'):
            continue
        _set_nested(merged, dotted, value)

    # build nested dataclasses
    kwargs = {}
    for f in dataclasses.fields(cls):
        if not f.init or f.name not in merged:
            if f.name in subgroups:
                kwargs[f.name] = subgroups[f.name][selected[f.name]]()
            continue
        value = merged[f.name]
        if f.name in subgroups:
            sub_cls = subgroups[f.name][selected[f.name]]
            kwargs[f.name] = _build_with_defaults(sub_cls, value or {})
            continue
        tp = _unwrap_optional(field_types[f.name])
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            kwargs[f.name] = _build_with_defaults(tp, value)
        else:
            from .serialize import from_value
            kwargs[f.name] = from_value(field_types[f.name], value)
    return cls(**kwargs)


def _build_with_defaults(cls, overrides: Dict[str, Any]):
    """ Build dataclass `cls` from partial (nested) overrides, keeping field defaults for the rest. """
    from .serialize import from_value
    field_types = resolve_field_types(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if not f.init or f.name not in overrides:
            continue
        value = overrides[f.name]
        tp = _unwrap_optional(field_types[f.name])
        if dataclasses.is_dataclass(tp) and isinstance(value, dict):
            kwargs[f.name] = _build_with_defaults(tp, value)
        else:
            kwargs[f.name] = from_value(field_types[f.name], value)
    unknown = set(overrides.keys()) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise TypeError(f'Unknown fields {sorted(unknown)} for {cls.__name__}.')
    return cls(**kwargs)
