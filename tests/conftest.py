"""Classify tests so offline and external-service runs remain separate."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items):
    """Mark every ordinary test as unit unless it explicitly says integration."""

    unit_marker = pytest.mark.unit
    for item in items:
        if "integration" not in item.keywords:
            item.add_marker(unit_marker)
