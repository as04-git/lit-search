"""Tests for the strict readers that make schema refusal work.

These readers exist because the library's own ``from_dict`` methods read with
``.get(key, default)``.  That turns a missing field into an empty-but-plausible
value, which is the same class of failure as a database that never ran being
recorded as one that found nothing.
"""

from __future__ import annotations

import pytest

from lit_search.exceptions import HandleSchemaError
from lit_search.store.lineage import OperationName
from lit_search.store.strict import (
    reject_unknown_keys,
    require,
    require_enum,
    require_float_list,
    require_mapping,
    require_optional,
)


class TestRequireMapping:
    """Only an object with string keys is a payload."""

    def test_accepts_a_mapping(self) -> None:
        """A plain dict passes through."""
        assert require_mapping({"a": 1}, "ctx") == {"a": 1}

    @pytest.mark.parametrize("value", [[], "text", 3, None])
    def test_refuses_non_mappings(self, value) -> None:
        """Anything that is not an object is refused."""
        with pytest.raises(HandleSchemaError, match="expected an object"):
            require_mapping(value, "ctx")

    def test_refuses_non_string_keys(self) -> None:
        """A JSON object always has string keys; anything else is suspect."""
        with pytest.raises(HandleSchemaError, match="expected string keys"):
            require_mapping({1: "a"}, "ctx")


class TestRequire:
    """A required key is read at its declared type, or refused."""

    def test_reads_a_present_value(self) -> None:
        """The happy path returns the value unchanged."""
        assert require({"a": "x"}, "a", str, "ctx") == "x"

    def test_missing_key_is_refused(self) -> None:
        """A missing key never becomes a default."""
        with pytest.raises(HandleSchemaError, match=r"ctx\.a: required key is missing"):
            require({}, "a", str, "ctx")

    def test_wrong_type_is_not_coerced(self) -> None:
        """A value of the wrong type is an error, not a conversion."""
        with pytest.raises(HandleSchemaError, match="expected str, got int"):
            require({"a": 1}, "a", str, "ctx")

    def test_bool_is_not_accepted_as_int(self) -> None:
        """``bool`` subclasses ``int``; the two stay distinguishable."""
        with pytest.raises(HandleSchemaError, match="expected int, got bool"):
            require({"a": True}, "a", int, "ctx")


class TestRequireOptional:
    """Present-and-null is a different fact from absent."""

    def test_null_value_is_accepted(self) -> None:
        """A writer with nothing to record writes null."""
        assert require_optional({"a": None}, "a", str, "ctx") is None

    def test_present_value_is_read(self) -> None:
        """A non-null value is read at its declared type."""
        assert require_optional({"a": "x"}, "a", str, "ctx") == "x"

    def test_absent_key_is_still_refused(self) -> None:
        """Nullable does not mean optional."""
        with pytest.raises(HandleSchemaError, match="required key is missing"):
            require_optional({}, "a", str, "ctx")


class TestRejectUnknownKeys:
    """Ignoring an unknown key is how schema drift becomes silent."""

    def test_allows_a_known_subset(self) -> None:
        """A payload using only declared keys passes."""
        reject_unknown_keys({"a": 1}, ("a", "b"), "ctx")

    def test_refuses_and_names_the_offending_keys(self) -> None:
        """The message says exactly which keys were not understood."""
        with pytest.raises(HandleSchemaError, match=r"\['c', 'd'\]"):
            reject_unknown_keys({"a": 1, "c": 2, "d": 3}, ("a", "b"), "ctx")


class TestRequireFloatList:
    """Scores must be finite numbers."""

    def test_widens_integers(self) -> None:
        """JSON has one number type; integers are valid scores."""
        assert require_float_list({"s": [1, 0.5]}, "s", "ctx") == [1.0, 0.5]

    def test_refuses_booleans(self) -> None:
        """``True`` is not a score."""
        with pytest.raises(HandleSchemaError, match="expected a number, got bool"):
            require_float_list({"s": [True]}, "s", "ctx")

    def test_refuses_strings(self) -> None:
        """A stringified score is refused rather than parsed."""
        with pytest.raises(HandleSchemaError, match="expected a number, got str"):
            require_float_list({"s": ["0.5"]}, "s", "ctx")

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_refuses_non_finite_values(self, value) -> None:
        """A NaN score silently scrambles any sort it takes part in."""
        with pytest.raises(HandleSchemaError, match="finite"):
            require_float_list({"s": [value]}, "s", "ctx")


class TestRequireEnum:
    """Closed enums stay closed."""

    def test_reads_a_member(self) -> None:
        """A declared value maps to its enum member."""
        assert require_enum({"n": "union"}, "n", OperationName, "ctx") is OperationName.UNION

    def test_refuses_an_unknown_value_and_lists_the_allowed_ones(self) -> None:
        """The message tells the reader what would have been accepted."""
        with pytest.raises(HandleSchemaError, match="'vibes' is not one of"):
            require_enum({"n": "vibes"}, "n", OperationName, "ctx")
