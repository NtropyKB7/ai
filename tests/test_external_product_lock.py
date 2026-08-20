from pathlib import Path

import pytest

from app.commands.sync_external_products import single_process_lock


def test_single_process_lock_rejects_concurrent_holder(tmp_path: Path):
    lock_path = tmp_path / "sync.lock"
    with single_process_lock(lock_path):
        with pytest.raises(RuntimeError, match="already running"):
            with single_process_lock(lock_path):
                pass


def test_lock_is_released_after_context(tmp_path: Path):
    lock_path = tmp_path / "sync.lock"
    with single_process_lock(lock_path):
        pass
    with single_process_lock(lock_path):
        pass
