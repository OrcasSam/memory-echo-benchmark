"""Synthetic stand-ins for optional upstream integrations in unit tests."""

import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def mock_com(monkeypatch):
    """Exercise our CoM adapter without bundling the upstream CoM checkout."""

    src = ModuleType("src")
    src.__path__ = []
    com = ModuleType("src.com")
    com.__path__ = []
    filter_module = ModuleType("src.com.filter")
    models_module = ModuleType("src.com.models")

    def com_filter(nodes, _query_vector, **_kwargs):
        return SimpleNamespace(
            main_chain=sorted(
                nodes, key=lambda node: node.similarity_to_query, reverse=True
            )
        )

    filter_module.com_filter = com_filter
    models_module.TurnNode = SimpleNamespace
    for module in (src, com, filter_module, models_module):
        monkeypatch.setitem(sys.modules, module.__name__, module)
