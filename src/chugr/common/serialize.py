""" Minimal dataclass (de)serialization

A small replacement for the subset of `simple_parsing.helpers.Serializable` used by chugr & pixparse.
Supports nested dataclasses, Optional / Union / Tuple / List / Dict fields, and json or yaml files.
"""
import dataclasses
import json
import types
import typing
from pathlib import Path
from typing import Any, Dict, Type, TypeVar, Union

T = TypeVar('T')

_NONE_TYPE = type(None)


def _is_union(tp) -> bool:
    origin = typing.get_origin(tp)
    return origin is Union or origin is types.UnionType


def _unwrap_optional(tp):
    """ Optional[X] -> X, leave other unions untouched """
    if _is_union(tp):
        args = [a for a in typing.get_args(tp) if a is not _NONE_TYPE]
        if len(args) == 1:
            return args[0]
    return tp


def resolve_field_types(cls) -> Dict[str, Any]:
    try:
        return typing.get_type_hints(cls)
    except Exception:
        return {f.name: f.type for f in dataclasses.fields(cls)}


def to_dict(obj) -> Any:
    """ Recursively convert dataclasses (and containers of) to plain python types. """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in dataclasses.fields(obj):
            if not f.init and not f.metadata.get('serialize', False):
                continue
            out[f.name] = to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def from_value(tp, value):
    """ Convert a plain python value to the annotated type `tp`, recursing into dataclasses. """
    if value is None:
        return None
    tp = _unwrap_optional(tp)

    if dataclasses.is_dataclass(tp):
        if isinstance(value, tp):
            return value
        assert isinstance(value, dict), f'Expected dict for {tp}, got {type(value)}'
        return from_dict(tp, value)

    if _is_union(tp):
        # try dataclass members first when given a dict, otherwise pass through unchanged
        if isinstance(value, dict):
            for a in typing.get_args(tp):
                if dataclasses.is_dataclass(a):
                    try:
                        return from_dict(a, value)
                    except (TypeError, AssertionError):
                        continue
        if isinstance(value, list):
            for a in typing.get_args(tp):
                if typing.get_origin(a) in (list, tuple):
                    return from_value(a, value)
        return value

    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            value = (value,)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_value(args[0], v) for v in value)
        if args and len(args) == len(value):
            return tuple(from_value(a, v) for a, v in zip(args, value))
        return tuple(value)
    if origin is list:
        item_tp = args[0] if args else Any
        return [from_value(item_tp, v) for v in value]
    if origin is dict:
        val_tp = args[1] if len(args) == 2 else Any
        return {k: from_value(val_tp, v) for k, v in value.items()}
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    field_types = resolve_field_types(cls)
    init_fields = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = set(data.keys()) - init_fields
    if unknown:
        raise TypeError(f'Unknown fields {sorted(unknown)} for {cls.__name__}.')
    kwargs = {k: from_value(field_types[k], v) for k, v in data.items()}
    return cls(**kwargs)


def _load_file(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix in ('.yaml', '.yml'):
        import yaml
        return yaml.safe_load(text) or {}
    return json.loads(text)


class Serializable:
    """ Mixin for dataclasses w/ dict, json, and yaml (de)serialization. """

    def to_dict(self) -> Dict[str, Any]:
        return to_dict(self)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return from_dict(cls, data)

    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        return from_dict(cls, _load_file(path))

    def save(self, path: Union[str, Path]):
        path = Path(path)
        if path.suffix in ('.yaml', '.yml'):
            path.write_text(self.dumps_yaml())
        else:
            path.write_text(self.dumps_json())

    def dumps_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def dumps_yaml(self) -> str:
        import yaml
        return yaml.safe_dump(self.to_dict(), sort_keys=False)
