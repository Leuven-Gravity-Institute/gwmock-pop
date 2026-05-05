"""Typed exceptions for external population loading."""

from __future__ import annotations


class PopulationError(Exception):
    """Base exception for population-loading failures."""


class PopulationFetchError(PopulationError):
    """Raised when a remote population catalogue cannot be fetched."""


class PopulationValidationError(PopulationError, ValueError):
    """Raised when a population catalogue fails schema or data validation."""


__all__ = ["PopulationError", "PopulationFetchError", "PopulationValidationError"]
