from typing import Any

import pytest

from miles.utils import object_store_config


class TestParseSize:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (12345, 12345),
            ("12345", 12345),
            ("1kb", 1024),
            ("2k", 2 * 1024),
            ("64mb", 64 * 1024**2),
            ("3m", 3 * 1024**2),
            ("2gb", 2 * 1024**3),
            ("1g", 1024**3),
            ("1.5gb", int(1.5 * 1024**3)),
            ("  2GB ", 2 * 1024**3),
        ],
    )
    def test_parses_ints_and_unit_suffixes(self, value: Any, expected: int):
        """_parse_size handles ints, plain digit strings, and kb/mb/gb suffixes case-insensitively."""
        assert object_store_config._parse_size(value) == expected

    def test_rejects_garbage(self):
        """_parse_size raises ValueError on a non-numeric string without a known unit."""
        with pytest.raises(ValueError):
            object_store_config._parse_size("lots")


class TestMooncakeStoreConfig:
    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "local_hostname": "10.0.0.1",
            "master_server_address": "10.0.0.2:50051",
            "protocol": "tcp",
            "global_segment_size": "2gb",
            "local_buffer_size": "1gb",
        }

    def test_contributing_process_parses_segment_size(self):
        """A contributing process gets the configured global_segment_size parsed to bytes."""
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=True)
        assert config["global_segment_size"] == 2 * 1024**3
        assert config["local_buffer_size"] == 1024**3
        assert config["master_server_addr"] == "10.0.0.2:50051"
        assert config["protocol"] == "tcp"
        assert config["local_hostname"] == "10.0.0.1"

    def test_non_contributing_process_gets_zero_segment(self):
        """A non-contributing process passes global_segment_size=0 (pure client semantics)."""
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=False)
        assert config["global_segment_size"] == 0

    def test_env_fallbacks(self, monkeypatch: pytest.MonkeyPatch):
        """Unset kwargs fall back to MOONCAKE_* environment variables."""
        monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "10.1.1.1")
        monkeypatch.setenv("MOONCAKE_MASTER", "10.1.1.2:50051")
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "tcp")
        monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "64mb")
        config = object_store_config.compute_mooncake_store_config({}, contribute_segment=True)
        assert config["local_hostname"] == "10.1.1.1"
        assert config["master_server_addr"] == "10.1.1.2:50051"
        assert config["protocol"] == "tcp"
        assert config["global_segment_size"] == 64 * 1024**2

    def test_kwargs_take_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch):
        """Explicit init kwargs win over MOONCAKE_* environment variables."""
        monkeypatch.setenv("MOONCAKE_MASTER", "10.9.9.9:50051")
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=True)
        assert config["master_server_addr"] == "10.0.0.2:50051"

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        """With no kwargs and no env, protocol/metadata/segment sizes use built-in defaults."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setattr(object_store_config, "_local_hostname", lambda: "127.0.0.1")
        config = object_store_config.compute_mooncake_store_config({}, contribute_segment=True)
        assert config["protocol"] == "rdma"
        assert config["metadata_server"] == "P2PHANDSHAKE"
        assert config["global_segment_size"] == 8 * 1024**3
        assert config["local_buffer_size"] == 32 * 1024**3
        assert config["master_server_addr"] == ""


def _clear_mooncake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "MOONCAKE_LOCAL_HOSTNAME",
        "MOONCAKE_TE_META_DATA_SERVER",
        "MOONCAKE_LOCAL_BUFFER_SIZE",
        "MOONCAKE_PROTOCOL",
        "MOONCAKE_DEVICE",
        "MOONCAKE_MASTER",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
