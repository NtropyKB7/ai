"""Test-session isolation for external recommendation dependencies.

This file is loaded by pytest before test-module collection.  The application
creates its Chroma client and embedding adapter at import time, so both must be
replaced before importing ``app.main`` or ``recommendation_service``.
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import chromadb
import pytest
import langchain_huggingface
from pydantic import PydanticDeprecatedSince20


class _IsolatedEmbeddingAdapter:
    """Non-networking stand-in used only during automated tests."""

    def __init__(self, *args, **kwargs):
        self.model_name = kwargs.get("model_name")

    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0]


_ORIGINAL_HUGGINGFACE_EMBEDDINGS = langchain_huggingface.HuggingFaceEmbeddings
langchain_huggingface.HuggingFaceEmbeddings = _IsolatedEmbeddingAdapter


class _IsolatedCollection:
    def upsert(self, **kwargs):
        return None

    def query(self, **kwargs):
        return {"metadatas": [[]]}


class _IsolatedChromaClient:
    """In-memory-shaped fake that never opens the repository's .chroma data."""

    def __init__(self, *args, **kwargs):
        self.collection = _IsolatedCollection()

    def heartbeat(self):
        return 1

    def get_or_create_collection(self, name):
        return self.collection


_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR")
_SESSION_CHROMA_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="ntropy-ai-pytest-chroma-",
    ignore_cleanup_errors=True,
)
os.environ["CHROMA_DB_DIR"] = _SESSION_CHROMA_DIRECTORY.name


def _is_loopback(address) -> bool:
    if isinstance(address, tuple) and address:
        return address[0] in {"127.0.0.1", "::1", "localhost"}
    # Local pipe/socket addresses are not external network destinations.
    return not isinstance(address, tuple)


def _guard_socket_connect(sock, address):
    if _is_loopback(address):
        return _ORIGINAL_SOCKET_CONNECT(sock, address)
    raise AssertionError(f"Automated tests must not access external address: {address!r}")


def _guard_create_connection(address, *args, **kwargs):
    if _is_loopback(address):
        return _ORIGINAL_CREATE_CONNECTION(address, *args, **kwargs)
    raise AssertionError(f"Automated tests must not access external address: {address!r}")


# Apply safety guards before pytest imports application test modules.
socket.socket.connect = _guard_socket_connect
socket.create_connection = _guard_create_connection


def pytest_sessionfinish(session, exitstatus):
    socket.socket.connect = _ORIGINAL_SOCKET_CONNECT
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    langchain_huggingface.HuggingFaceEmbeddings = _ORIGINAL_HUGGINGFACE_EMBEDDINGS
    if _ORIGINAL_CHROMA_DB_DIR is None:
        os.environ.pop("CHROMA_DB_DIR", None)
    else:
        os.environ["CHROMA_DB_DIR"] = _ORIGINAL_CHROMA_DB_DIR
    _SESSION_CHROMA_DIRECTORY.cleanup()


@pytest.fixture(scope="module")
def isolated_recommendation_runtime():
    """Import an isolated application module graph for evaluation tests only."""
    isolation = tempfile.TemporaryDirectory(prefix="ntropy-ai-tests-")
    monkeypatch = pytest.MonkeyPatch()
    original_cwd = Path.cwd()
    original_app_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "app" or name.startswith("app.")
    }

    try:
        # Remove only the import cache entries. Existing tests retain their
        # direct references to the original production modules.
        for module_name in original_app_modules:
            sys.modules.pop(module_name, None)

        # Settings reads .env and the Chroma path relative to cwd. The isolated
        # graph is imported only after moving to an empty temporary directory.
        os.chdir(isolation.name)
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
        monkeypatch.setattr(
            langchain_huggingface,
            "HuggingFaceEmbeddings",
            _IsolatedEmbeddingAdapter,
        )
        monkeypatch.setattr(chromadb, "PersistentClient", _IsolatedChromaClient)
        monkeypatch.setattr(socket.socket, "connect", _guard_socket_connect)
        monkeypatch.setattr(socket, "create_connection", _guard_create_connection)

        with warnings.catch_warnings():
            # In a full-suite run production config has already emitted this
            # warning. Avoid reporting the same known warning a second time for
            # the isolated module graph; focused evaluation still reports it.
            if "app.core.config" in original_app_modules:
                warnings.filterwarnings(
                    "ignore",
                    category=PydanticDeprecatedSince20,
                )

            router_module = importlib.import_module("app.api.router")
            main_module = importlib.import_module("app.main")
            chroma_module = importlib.import_module("app.rag.chroma_client")
            product_module = importlib.import_module("app.schemas.product")
            recommendation_module = importlib.import_module(
                "app.services.recommendation_service"
            )

        yield SimpleNamespace(
            app=main_module.app,
            router_module=router_module,
            chroma_manager=chroma_module.chroma_manager,
            FinancialProductSchema=product_module.FinancialProductSchema,
            ProductRecommendationRequest=product_module.ProductRecommendationRequest,
            ProductRecommendationResponse=product_module.ProductRecommendationResponse,
            RecommendationService=recommendation_module.RecommendationService,
            temp_directory=Path(isolation.name),
        )
    finally:
        isolated_module_names = [
            name
            for name in sys.modules
            if name == "app" or name.startswith("app.")
        ]
        for module_name in isolated_module_names:
            sys.modules.pop(module_name, None)
        sys.modules.update(original_app_modules)

        os.chdir(original_cwd)
        monkeypatch.undo()
        isolation.cleanup()
