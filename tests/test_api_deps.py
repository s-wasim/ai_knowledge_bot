from unittest.mock import patch

from app.api import deps
from app.retrieval.hybrid import HybridRetriever


class TestGetRetrieverAndMode:
    def setup_method(self):
        deps.reset_singletons()

    def teardown_method(self):
        deps.reset_singletons()

    @patch("app.retrieval.factory.is_embedding_available", return_value=True)
    def test_returns_hybrid_retriever_when_embeddings_available(self, _):
        retriever, mode = deps.get_retriever_and_mode()
        assert isinstance(retriever, HybridRetriever)
        assert mode == "hybrid"

    @patch("app.retrieval.factory.is_embedding_available", return_value=False)
    def test_reports_degraded_without_the_embedding_model(self, _):
        """Search still works on full-text and symbols; the point is that the
        downgrade is named rather than silent."""
        retriever, mode = deps.get_retriever_and_mode()
        assert isinstance(retriever, HybridRetriever)
        assert mode == "degraded"

    @patch("app.retrieval.factory.is_embedding_available", return_value=False)
    def test_caches_singleton_across_calls(self, _):
        retriever_1, _mode = deps.get_retriever_and_mode()
        retriever_2, _mode2 = deps.get_retriever_and_mode()
        assert retriever_1 is retriever_2


class TestGetGraph:
    def setup_method(self):
        deps.reset_singletons()

    def test_returns_compiled_graph(self):
        graph = deps.get_graph()
        assert hasattr(graph, "invoke")
        assert hasattr(graph, "stream")

    def test_caches_singleton_across_calls(self):
        graph_1 = deps.get_graph()
        graph_2 = deps.get_graph()
        assert graph_1 is graph_2
