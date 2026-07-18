"""Frozen open-vocabulary detector adapters for the OVD-Diagnose benchmark.

Each adapter wraps a pretrained OVD model behind a uniform interface so the
diagnostic runner can prompt any model with any vocabulary. See base.OVDAdapter.
"""
from .base import OVDAdapter, ADAPTERS, build_adapter  # noqa: F401
