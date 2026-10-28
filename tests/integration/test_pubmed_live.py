"""Opt-in live PubMed checks.

Run with ``$env:RUN_PUBMED_INTEGRATION = "1"``. The default integration run is
skipped so ordinary offline test commands never depend on NCBI or the network.
"""

from __future__ import annotations

import os

import pytest

from app.pubmed_client_v2 import PubMedClientV2


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_PUBMED_INTEGRATION") != "1",
    reason="设置 RUN_PUBMED_INTEGRATION=1 才运行联网 PubMed 测试",
)
def test_live_pubmed_search_returns_a_normalized_document():
    documents = PubMedClientV2().search("lung cancer immunotherapy", limit=1)

    assert documents
    assert documents[0].id.startswith("pubmed:")
    assert documents[0].title
    assert documents[0].url
