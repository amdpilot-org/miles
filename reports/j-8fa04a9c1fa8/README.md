# Miles two-GPU weight-update investigation

## Status

Draft investigation in progress. This report will be updated with the fixture, exact commits, numerical checks, CUDA-event timings, memory trends, and any negative result.

## Scope

- Exercise the actual Miles FSDP training hook and distributed weight-update hook.
- Use one assigned AMD Instinct MI350X for the actor/trainer and one for the SGLang rollout engine.
- Use a locally initialized, tiny Qwen2 architecture; do not interpret it as evidence for larger models or MI355X performance equivalence.
- Run 256 real optimizer steps across 64 successive rollout/weight-update cycles.
- After every update, check the served weight version, engine checksum comparison, and fixed-token output against a fresh-loaded control.
- Record CUDA-event phase timings and CUDA memory trends.

## Environment observed

- Runtime image: `amdpilotv2/miles-job:gbt350-d957-20260909`
- Python: `/opt/venv/bin/python` (3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- GPUs: 2 × AMD Instinct MI350X (`gfx950`)
- Miles base commit under investigation: `e5125a97e1fd383f005f4de258a5985026e09425`

## Reproduction outline

The final fixture and exact command will be committed under this directory. The run will use both assigned GPUs, unique rendezvous state, and bounded process-group timeouts. It will stop immediately after the 64th update and its checks.
