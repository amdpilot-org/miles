# Reduced MoE P2P investigation

This directory contains a bounded reproduction for radixark/miles#2856. It uses
two assigned AMD Instinct MI350X GPUs, a synthetic 4-expert Qwen3-MoE config,
and no checkpoint download.

## Reproduction

```bash
cd /job/miles
git switch amdpilot/j-381b642ecebc
reports/j-381b642ecebc/run.sh
```

The runner starts only its own `torchrun` subprocesses. It gives process-group
construction a 30-second timeout and bounds each complete check with `timeout`.
The private AITER cache is kept under `/job/.cache/j-381b642ecebc`, outside the
Git worktree.

## Checks

1. `same-node` constructs real Gloo and NCCL process groups, constructs the
   production Miles CPU replica through `UpdateWeightP2P._create_cpu_replica`,
   transfers all reduced-model parameters from GPU 0 to GPU 1 with Mooncake's
   HIP transport, and verifies SHA-256 equality of every reconstructed tensor.
2. `issue-boundary` constructs the same real groups, then asks the production
   replica path to use `ServerArgs(nnodes=2)`. Both ranks must fail in
   `MultimemAllGatherer` with the reported unregistered `MagicMock` process-group
   error.

## Recorded run

The run completed on 2026-09-10 UTC with two assigned `gfx950` AMD Instinct
MI350X devices. `rocm-smi` reported `card0` GUID `17079` and `card1` GUID
`11995`; both are MI350X devices. The fixture requires and uses both GPUs.

| Check | Result |
|---|---|
| Same-node production path | Pass on both ranks |
| Real process groups | Gloo CPU and NCCL GPU groups constructed and sanity-reduced |
| Reduced CPU replica | `Qwen3MoeForCausalLM`, 12 parameters, 5,248 elements |
| Mooncake HIP transfer | One contiguous 10,496-byte GPU 0 to GPU 1 write |
| Reconstructed equality | All 12 parameter slices matched source SHA-256 |
| Declared `nnodes=2` boundary | Both ranks reproduced the `MagicMock` error |

The tested Miles revision was `e5125a97e1fd383f005f4de258a5985026e09425`.
The imported SGLang source was `/sgl-workspace/sglang/python/sglang` at Git
commit `a8e5c632fe40555f720d4f2c69771ea8cf24f3c4`; its worktree had 16
untracked generated HIP source files, so the commit identifies the source tree
but does not imply a clean release artifact. Torch was
`2.9.1+rocm7.2.0.lw.git7e1940d4` with HIP `7.2.26015-fc0010cf6a`.

The fixture prints the actual imported paths. In the recorded run they were:

- Miles: `/job/miles/miles/__init__.py`
- SGLang: `/sgl-workspace/sglang/python/sglang/__init__.py`
- Megatron Bridge: `/opt/venv/lib/python3.10/site-packages/megatron/bridge/__init__.py`
- Torch Python: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch native: `/opt/venv/lib/python3.10/site-packages/torch/_C.cpython-310-x86_64-linux-gnu.so`
- Mooncake native: `/opt/venv/lib/python3.10/site-packages/mooncake/engine.cpython-310-x86_64-linux-gnu.so`
- AITER native: `/sgl-workspace/aiter/aiter/jit/module_aiter_core.so`

The installed Mooncake wheel has no importlib metadata or `__version__`; the
native module path above is therefore the recorded identity. Mooncake topology
discovery found no RDMA HCAs, so this run used its intra-node HIP transport.

## Scope

The passing `same-node` check is a production-path reproduction, not a
production-topology reproduction. It proves that the reduced model, real
two-GPU process groups, Mooncake HIP transfer, and reconstructed weight
equality work together when `nnodes=1`.

The `issue-boundary` check is only a reduced boundary reproduction. Both ranks
run on one MI350X node, so declaring `nnodes=2` does not exercise real
multi-node rank placement or RDMA. It does reproduce the exact condition that
causes `ParallelismContext` to hand `MultimemAllGatherer` a mocked CPU group.

Although the fixture constructs real Gloo and NCCL groups for coordination and
transfer, `_create_cpu_replica` itself still enters `ParallelismContext` and
uses its mock groups. With `nnodes=1`, SGLang short-circuits the topology probe
before reading `cpu_group`. This check therefore does not prove SGLang model
construction under real runtime parallel state; it proves the Miles production
replica method, real two-GPU coordination groups, Mooncake HIP transfer, and
reconstructed weight equality for the reduced same-node case.

Neither check downloads or runs Qwen3-235B-A22B, and neither check proves that
the original multi-node 235B P2P update succeeds after avoiding the boundary.
No production fix is proposed here.

The first attempts with 12 separate tiny Mooncake segments showed unreliable
reconstruction on this HIP build. The committed fixture deliberately uses one
contiguous shared buffer, matching the production P2P shared-buffer design, and
then reconstructs all parameter slices for equality. This is a fixture choice,
not evidence that Mooncake batch transfer of many tiny separate segments is
correct on this stack.
