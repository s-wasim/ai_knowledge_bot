from app.api import deps


class TestGetRetrieverAndMode:
    def setup_method(self):
        deps.reset_singletons()

    def test_returns_fts_retriever_when_no_voyage_key(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        retriever, mode = deps.get_retriever_and_mode()
        assert mode == "fts"
        assert retriever is not None

    def test_caches_singleton_across_calls(self, monkeypatch):
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
        retriever_1, _ = deps.get_retriever_and_mode()
        retriever_2, _ = deps.get_retriever_and_mode()
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
