from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compose_mounts_configurable_host_path_at_fixed_container_path():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "source: ${CHROMA_HOST_DIR:-./.local/chroma-data}" in compose
    assert "target: /data/chroma" in compose
    assert "CHROMA_DB_DIR: /data/chroma" in compose
    assert (
        "FINANCIAL_PRODUCT_COLLECTION: "
        "${FINANCIAL_PRODUCT_COLLECTION:-financial_products}"
    ) in compose


def test_local_chroma_data_is_excluded_from_git_and_docker_build_context():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert ".local/" in gitignore.splitlines()
    assert ".local/" in dockerignore.splitlines()
    assert ".chroma/" in gitignore.splitlines()
    assert ".chroma/" in dockerignore.splitlines()


def test_environment_example_contains_names_without_credentials():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    values = {}
    for line in example.splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key] = value

    assert values == {
        "OPENAI_API_KEY": "",
        "OPENAI_CLASSIFICATION_MODEL": "gpt-5-nano",
        "OPENAI_GENERATION_MODEL": "gpt-4o-mini",
        "FINLIFE_API_KEY": "",
        "CHROMA_HOST_DIR": "./.local/chroma-data",
        "CHROMA_DB_DIR": "/data/chroma",
        "FINANCIAL_PRODUCT_COLLECTION": "financial_products",
    }


def test_direct_local_application_keeps_repository_relative_default():
    config_source = (ROOT / "app/core/config.py").read_text(encoding="utf-8")

    assert 'CHROMA_DB_DIR: str = "./.chroma"' in config_source
    assert 'FINANCIAL_PRODUCT_COLLECTION: str = "financial_products"' in config_source
