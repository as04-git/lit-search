"""Strict readers for deserializing handle payloads.

Every reader in this module **refuses** rather than defaults.  This is the
opposite of the convention used by
:meth:`lit_search.core.search_result.SearchResult.from_dict`, which reads its
fields with ``metadata.get("query", "")`` and therefore turns a missing or
renamed field into an empty-but-plausible value.  For a store whose whole
purpose is to make an incomplete result impossible to mistake for a complete
one, a silently defaulted field is the same failure in miniature: the object
loads, looks well formed, and no longer means what it says.

The rule these helpers enforce:

* a required key that is absent is an error, never a default,
* a key whose value has the wrong type is an error, never a coercion,
* an unrecognised key is an error, because it is evidence that the payload was
  written by a schema this reader does not understand.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

from lit_search.exceptions import HandleSchemaError

T = TypeVar("T")


def require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Return *value* as a mapping or refuse.

    Parameters
    ----------
    value : Any
        Value to check.
    context : str
        Dotted path of the value, used in the error message.

    Returns
    -------
    Mapping[str, Any]
        The value, narrowed to a mapping.

    Raises
    ------
    HandleSchemaError
        If *value* is not a mapping with string keys.
    """
    if not isinstance(value, Mapping):
        raise HandleSchemaError(f"{context}: expected an object, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise HandleSchemaError(f"{context}: expected string keys, got {type(key).__name__}")
    return value


def reject_unknown_keys(payload: Mapping[str, Any], allowed: Iterable[str], context: str) -> None:
    """Refuse a payload carrying keys this schema version does not define.

    Ignoring an unknown key is how schema drift becomes silent: a reader built
    for version 1 would quietly discard a version 2 field that changes the
    meaning of the record.  Refusing forces every field addition through an
    explicit schema version bump.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Payload to inspect.
    allowed : Iterable[str]
        Keys defined by this schema version.
    context : str
        Dotted path of the payload, used in the error message.

    Returns
    -------
    None

    Raises
    ------
    HandleSchemaError
        If *payload* contains any key outside *allowed*.
    """
    unknown = sorted(set(payload) - set(allowed))
    if unknown:
        raise HandleSchemaError(
            f"{context}: unrecognised key(s) {unknown}; "
            "this payload was written by a schema this reader does not understand"
        )


def require(payload: Mapping[str, Any], key: str, expected: type[T], context: str) -> T:
    """Read a required key of a required type, or refuse.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Payload to read from.
    key : str
        Key that must be present.
    expected : type[T]
        Type the value must already have.  No coercion is attempted.
    context : str
        Dotted path of the payload, used in the error message.

    Returns
    -------
    T
        The value.

    Raises
    ------
    HandleSchemaError
        If the key is absent or the value has the wrong type.
    """
    if key not in payload:
        raise HandleSchemaError(f"{context}.{key}: required key is missing")
    value = payload[key]
    # bool is a subclass of int; keep the two distinguishable.
    if expected is int and isinstance(value, bool):
        raise HandleSchemaError(f"{context}.{key}: expected int, got bool")
    if not isinstance(value, expected):
        raise HandleSchemaError(
            f"{context}.{key}: expected {expected.__name__}, got {type(value).__name__}"
        )
    return value


def require_optional(
    payload: Mapping[str, Any], key: str, expected: type[T], context: str
) -> T | None:
    """Read a key that must be present but whose value may be ``null``.

    "Present and null" and "absent" are different facts and are kept
    distinguishable: the first says the writer had nothing to record, the
    second says the writer did not know about the field at all.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Payload to read from.
    key : str
        Key that must be present.
    expected : type[T]
        Type the value must have when it is not ``None``.
    context : str
        Dotted path of the payload, used in the error message.

    Returns
    -------
    T | None
        The value, or ``None``.

    Raises
    ------
    HandleSchemaError
        If the key is absent or the value is neither ``None`` nor *expected*.
    """
    if key not in payload:
        raise HandleSchemaError(f"{context}.{key}: required key is missing")
    if payload[key] is None:
        return None
    return require(payload, key, expected, context)


def require_str_list(payload: Mapping[str, Any], key: str, context: str) -> list[str]:
    """Read a required list whose every element must be a string.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Payload to read from.
    key : str
        Key that must be present.
    context : str
        Dotted path of the payload, used in the error message.

    Returns
    -------
    list[str]
        The list, unchanged.

    Raises
    ------
    HandleSchemaError
        If the key is absent, not a list, or holds a non-string element.
    """
    raw = require(payload, key, list, context)
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise HandleSchemaError(
                f"{context}.{key}[{index}]: expected a string, got {type(item).__name__}"
            )
    return list(raw)


def require_float_list(payload: Mapping[str, Any], key: str, context: str) -> list[float]:
    """Read a required list whose every element must be a real number.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Payload to read from.
    key : str
        Key that must be present.
    context : str
        Dotted path of the payload, used in the error message.

    Returns
    -------
    list[float]
        The list, with integers widened to floats.

    Raises
    ------
    HandleSchemaError
        If the key is absent, not a list, or holds a non-numeric element.
    """
    raw = require(payload, key, list, context)
    values: list[float] = []
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise HandleSchemaError(
                f"{context}.{key}[{index}]: expected a number, got {type(item).__name__}"
            )
        if not math.isfinite(item):
            # NaN compares false against everything, so a NaN score silently
            # scrambles any sort it takes part in.  Refuse it at the boundary.
            raise HandleSchemaError(f"{context}.{key}[{index}]: expected a finite number")
        values.append(float(item))
    return values


def require_enum(payload: Mapping[str, Any], key: str, enum_type: Any, context: str) -> Any:
    """Read a required key whose value must name a member of a closed enum.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Payload to read from.
    key : str
        Key that must be present.
    enum_type : Any
        A ``str``-valued :class:`enum.Enum` subclass.
    context : str
        Dotted path of the payload, used in the error message.

    Returns
    -------
    Any
        The enum member.

    Raises
    ------
    HandleSchemaError
        If the key is absent or its value is not a member of *enum_type*.
    """
    raw = require(payload, key, str, context)
    try:
        return enum_type(raw)
    except ValueError as exc:
        allowed = sorted(member.value for member in enum_type)
        raise HandleSchemaError(f"{context}.{key}: {raw!r} is not one of {allowed}") from exc
