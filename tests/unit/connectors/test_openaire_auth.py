"""Focused tests for OpenAIRE's one-hour access-token refresh flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from lit_search.connectors.openaire import OpenAIREConnector
from lit_search.connectors.openaire_auth import (
    OPENAIRE_REFRESH_URL,
    OpenAIREAuthenticationError,
    OpenAIRETokenManager,
)
from lit_search.utils.logging_config import sanitize_message


def _token_response(
    access_token: str,
    *,
    refresh_token: str = "month-token",
    expires_in: object = "3600",
) -> MagicMock:
    """Return a successful response shaped like OpenAIRE's auth payload."""
    response = MagicMock()
    response.json.return_value = {
        "access_token": access_token,
        "token_type": "Bearer",
        "refresh_token": refresh_token,
        "expires_in": expires_in,
        "scope": "openid",
    }
    return response


def test_refresh_token_is_exchanged_and_bearer_is_attached() -> None:
    """A month-lived refresh token makes the fast authenticated path usable."""
    session = MagicMock()
    session.get.return_value = _token_response("hour-token")
    connector = OpenAIREConnector(refresh_token="month-token")

    with patch.object(connector, "_get_session", return_value=session):
        headers = connector._prepare_headers({})

    assert headers["Authorization"] == "Bearer hour-token"
    session.get.assert_called_once_with(
        OPENAIRE_REFRESH_URL,
        params={"refreshToken": "month-token"},
        timeout=30.0,
    )
    assert connector.min_request_interval == 0.5
    assert connector.request_quota.limit == 7200


def test_access_token_is_cached_until_its_proactive_refresh_deadline() -> None:
    """Repeated API pages do not exchange a token on every request."""
    now = [100.0]
    session = MagicMock()
    session.get.side_effect = [_token_response("first"), _token_response("second")]
    manager = OpenAIRETokenManager(refresh_token="month-token", clock=lambda: now[0])

    assert manager.get_access_token(session) == "first"
    now[0] = 3_639.0
    assert manager.get_access_token(session) == "first"
    now[0] = 3_641.0
    assert manager.get_access_token(session) == "second"
    assert session.get.call_count == 2


def test_returned_refresh_token_is_adopted_for_the_next_exchange() -> None:
    """A server-side refresh-token rotation does not break the live connector."""
    now = [0.0]
    session = MagicMock()
    session.get.side_effect = [
        _token_response("first", refresh_token="rotated", expires_in=10),
        _token_response("second", refresh_token="rotated-again", expires_in=10),
    ]
    manager = OpenAIRETokenManager(refresh_token="original", clock=lambda: now[0])

    assert manager.get_access_token(session) == "first"
    assert manager.refresh_token == "rotated"
    now[0] = 9.1
    assert manager.get_access_token(session) == "second"
    assert session.get.call_args_list[1].kwargs["params"] == {"refreshToken": "rotated"}
    assert manager.refresh_token == "rotated-again"


def test_original_refresh_token_remains_the_stable_quota_identity() -> None:
    """Rotated bearer material cannot split one account's shared hourly bucket."""
    first_session = MagicMock()
    first_session.get.return_value = _token_response(
        "first-hour-token", refresh_token="first-rotated-token"
    )
    second_session = MagicMock()
    second_session.get.return_value = _token_response(
        "second-hour-token", refresh_token="second-rotated-token"
    )
    first = OpenAIREConnector(refresh_token="shared-month-token")
    second = OpenAIREConnector(refresh_token="shared-month-token")

    with patch.object(first, "_get_session", return_value=first_session):
        first._prepare_headers({})
    with patch.object(second, "_get_session", return_value=second_session):
        second._prepare_headers({})

    assert first._access_token != second._access_token
    assert first._refresh_token != second._refresh_token
    assert first.rate_limit_scope == second.rate_limit_scope
    assert first.rate_limit_scope.fingerprint != "anon"
    assert "shared-month-token" not in first.rate_limit_scope.key


def test_unknown_age_access_token_is_refreshed_when_refresh_token_exists() -> None:
    """The connector does not guess that an independently acquired token is fresh."""
    session = MagicMock()
    session.get.return_value = _token_response("replacement")
    manager = OpenAIRETokenManager(
        access_token="possibly-expired",
        refresh_token="month-token",
    )

    assert manager.get_access_token(session) == "replacement"
    session.get.assert_called_once()


def test_known_fresh_access_token_avoids_an_unnecessary_exchange() -> None:
    """Callers that know remaining lifetime can use the supplied token first."""
    session = MagicMock()
    manager = OpenAIRETokenManager(
        access_token="known-fresh",
        refresh_token="month-token",
        access_token_expires_in=600,
    )

    assert manager.get_access_token(session) == "known-fresh"
    session.get.assert_not_called()


def test_failed_refresh_is_loud_and_never_downgrades_to_anonymous() -> None:
    """Invalid auth stops header preparation; no uncredentialed retry follows."""
    session = MagicMock()
    response = MagicMock(status_code=401)
    failure = requests.HTTPError(
        "401 for https://example.invalid?refreshToken=month-token",
        response=response,
    )
    session.get.side_effect = failure
    connector = OpenAIREConnector(refresh_token="month-token")

    with (
        patch.object(connector, "_get_session", return_value=session),
        pytest.raises(OpenAIREAuthenticationError) as caught,
    ):
        connector._prepare_headers({})

    assert "month-token" not in str(caught.value)
    assert "HTTP 401" in str(caught.value)
    assert session.get.call_count == 1
    assert connector.min_request_interval == 0.5


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"access_token": "secret-from-body", "token_type": "NotBearer", "expires_in": 3600},
        {"access_token": "secret-from-body", "token_type": "Bearer", "expires_in": "never"},
    ],
)
def test_malformed_refresh_response_fails_without_echoing_credentials(payload: object) -> None:
    """Bad auth schemas cannot leak either supplied or returned token material."""
    session = MagicMock()
    response = MagicMock()
    response.json.return_value = payload
    session.get.return_value = response
    manager = OpenAIRETokenManager(refresh_token="month-token")

    with pytest.raises(OpenAIREAuthenticationError) as caught:
        manager.get_access_token(session)

    rendered = str(caught.value)
    assert "month-token" not in rendered
    assert "secret-from-body" not in rendered


def test_refresh_token_query_parameter_is_globally_redacted() -> None:
    """Any future log of the auth URL is sanitized by the shared filter."""
    rendered = sanitize_message(
        f"GET {OPENAIRE_REFRESH_URL}?refreshToken=month-token&harmless=value"
    )

    assert "month-token" not in rendered
    assert "refreshToken=%2A%2A%2A" in rendered
    assert "harmless=value" in rendered


def _http_auth_error(status: int) -> requests.HTTPError:
    """Build an HTTP authentication rejection without credential material."""
    response = MagicMock(status_code=status)
    return requests.HTTPError(f"HTTP {status}", response=response)


def test_rejected_graph_token_is_force_refreshed_and_retried_once() -> None:
    """A still-cached but rejected token is replaced before one retry."""
    connector = OpenAIREConnector(
        access_token="rejected-hour-token",
        refresh_token="month-token",
        access_token_expires_in=600,
    )
    session = MagicMock()
    session.get.return_value = _token_response("replacement-hour-token")
    success = MagicMock(spec=requests.Response)

    with (
        patch.object(connector, "_get_session", return_value=session),
        patch.object(
            connector,
            "_request_with_retry",
            side_effect=[_http_auth_error(401), success],
        ) as request,
    ):
        assert connector._get("https://api.openaire.eu/graph/v3/research-products") is success

    assert request.call_count == 2
    assert request.call_args_list[0].kwargs["headers"]["Authorization"] == (
        "Bearer rejected-hour-token"
    )
    assert request.call_args_list[1].kwargs["headers"]["Authorization"] == (
        "Bearer replacement-hour-token"
    )
    session.get.assert_called_once_with(
        OPENAIRE_REFRESH_URL,
        params={"refreshToken": "month-token"},
        timeout=30.0,
    )


def test_second_graph_rejection_fails_after_exactly_one_retry() -> None:
    """A replacement token rejection is loud and never starts a loop."""
    connector = OpenAIREConnector(
        access_token="rejected-hour-token",
        refresh_token="month-token",
        access_token_expires_in=600,
    )
    session = MagicMock()
    session.get.return_value = _token_response("replacement-hour-token")

    with (
        patch.object(connector, "_get_session", return_value=session),
        patch.object(
            connector,
            "_request_with_retry",
            side_effect=[_http_auth_error(403), _http_auth_error(401)],
        ) as request,
        pytest.raises(requests.HTTPError, match="HTTP 401"),
    ):
        connector._get("https://api.openaire.eu/graph/v3/research-products")

    assert request.call_count == 2
    assert session.get.call_count == 1


def test_static_access_token_rejection_is_not_retried() -> None:
    """Without a refresh token, an authentication failure remains loud."""
    connector = OpenAIREConnector(access_token="static-hour-token")

    with (
        patch.object(
            connector,
            "_request_with_retry",
            side_effect=_http_auth_error(401),
        ) as request,
        pytest.raises(requests.HTTPError, match="HTTP 401"),
    ):
        connector._get("https://api.openaire.eu/graph/v3/research-products")

    request.assert_called_once()
