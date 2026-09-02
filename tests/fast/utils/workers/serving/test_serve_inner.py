from types import SimpleNamespace
from typing import Any

import pytest

from miles.utils.workers.serving import serve_inner
from miles.utils.workers.serving.serve_inner import parse_own_args

SPECS_PATH = "tests.fast.utils.workers.e2e.e2e_worker.compute_specs"
POOL_ID = "e2e-pool"


class TestParseOwnArgs:
    def test_the_spec_table_and_the_pool_it_serves_are_read(self) -> None:
        """These two are the whole of what the pod needs to find the one spec it is a worker of."""
        args = parse_own_args(["--specs", SPECS_PATH, "--pool-id", POOL_ID])

        assert (args.specs, args.pool_id) == (SPECS_PATH, POOL_ID)

    def test_an_omitted_pool_id_is_a_usage_error(self) -> None:
        """A process that does not know which pool it serves would pick a spec at random."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--specs", SPECS_PATH])

        assert exc_info.value.code == 2

    def test_an_omitted_spec_table_is_a_usage_error(self) -> None:
        """Without the run's spec table there is nothing to match the pool id against."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--pool-id", POOL_ID])

        assert exc_info.value.code == 2

    def test_unknown_inner_option_is_a_usage_error(self) -> None:
        """The inner entrypoint rejects an option it does not define instead of ignoring it."""
        with pytest.raises(SystemExit) as exc_info:
            parse_own_args(["--specs", SPECS_PATH, "--pool-id", POOL_ID, "--unknown-option", "1"])

        assert exc_info.value.code == 2


def _served(monkeypatch: pytest.MonkeyPatch, *, has_dualstack_ipv6: bool) -> dict[str, Any]:
    served: dict[str, Any] = {}
    monkeypatch.setattr(serve_inner.sys, "argv", ["serve_inner", "--specs", SPECS_PATH, "--pool-id", POOL_ID])
    monkeypatch.setattr(serve_inner.socket, "has_dualstack_ipv6", lambda: has_dualstack_ipv6)
    monkeypatch.setattr(serve_inner, "compute_serve_worker_spec", lambda **kwargs: SimpleNamespace(worker_class="w"))
    monkeypatch.setattr(serve_inner, "create_worker", lambda spec, **kwargs: object())
    monkeypatch.setattr(serve_inner, "create_rpc_app", lambda worker: "app")
    monkeypatch.setattr(serve_inner, "read_worker_in_pod_index", lambda environ: 0)
    monkeypatch.setattr(
        serve_inner,
        "_rpc_port_of",
        lambda spec: SimpleNamespace(effective_static_port=lambda worker_in_pod_index: 8123),
    )
    monkeypatch.setattr(serve_inner.uvicorn, "run", lambda app, host, port: served.update(host=host, port=port))

    serve_inner.main()
    return served


class TestTheAddressAWorkerIsServedOn:
    def test_binds_the_dual_stack_wildcard_where_the_platform_offers_one(self, monkeypatch):
        """The cell view publishes the pod ip, and on an ipv6-only cluster that is an ipv6 address."""
        assert _served(monkeypatch, has_dualstack_ipv6=True)["host"] == serve_inner.IPV6_WILDCARD_HOST

    def test_binds_the_ipv4_wildcard_where_ipv6_is_unavailable(self, monkeypatch):
        """Asking for the dual-stack wildcard where there is no ipv6 stack leaves the worker unserved."""
        assert _served(monkeypatch, has_dualstack_ipv6=False)["host"] == serve_inner.IPV4_WILDCARD_HOST

    def test_the_dual_stack_wildcard_is_the_unspecified_ipv6_address(self):
        """Only the unspecified address accepts the ipv4-mapped connections an ipv4 client makes."""
        assert (serve_inner.IPV6_WILDCARD_HOST, serve_inner.IPV4_WILDCARD_HOST) == ("::", "0.0.0.0")

    def test_serves_the_rpc_port_the_spec_declares_whichever_wildcard_it_binds(self, monkeypatch):
        """The address a client dials is the published pod ip and this port, so the port may not move."""
        assert _served(monkeypatch, has_dualstack_ipv6=True)["port"] == 8123
        assert _served(monkeypatch, has_dualstack_ipv6=False)["port"] == 8123
