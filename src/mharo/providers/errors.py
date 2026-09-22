"""Provider error hierarchy — router inhe parse karke fallback/rotation karta hai.
Sab real runtime errors, koi stub nahi."""

from __future__ import annotations


class ProviderError(Exception):
    """Provider layer ke sab errors ka base."""

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model


class RateLimitError(ProviderError):
    """HTTP 429 — router isse dekhta hai: key rotate / provider fallback / backoff."""

    def __init__(self, message: str, *, provider: str = "", model: str = "") -> None:
        super().__init__(message, provider=provider, model=model)


class AuthError(ProviderError):
    """HTTP 401/403 — key invalid ya revoke. Router agla key try karta hai."""


class TimeoutError2(ProviderError):
    """Request timeout — retry with backoff."""


class ValidationError2(ProviderError):
    """400 — payload galat hai. Retry nahi karna."""


class BadGatewayError(ProviderError):
    """502/503/504 — provider down. Fallback provider."""


class BaseUrlError(ProviderError):
    """base_url resolve/connect nahi hua (offline/local down)."""


__all__ = [
    "ProviderError",
    "RateLimitError",
    "AuthError",
    "TimeoutError2",
    "ValidationError2",
    "BadGatewayError",
    "BaseUrlError",
]
