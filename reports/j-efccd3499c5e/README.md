# Issue 1289 GPU validation

This fixture uses a local synthetic `Qwen3_5Config` with one target layer and one MTP layer. It runs real Megatron-Core forward/backward work under two-rank DDP on the assigned AMD Instinct MI350X GPUs, checks Megatron bridge mappings and Miles name-conversion hooks, exports through Megatron bridge conversion objects, perturbs parameters, and restores them from the exported HF state.

Run it with:

```bash
port=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 \
  --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:$port \
  --rdzv-id=j-efccd3499c5e-$port \
  reports/j-efccd3499c5e/validate_mtp_bridge.py \
  --cycles 32 --output reports/j-efccd3499c5e/gpu-results.json
```

The fixture uses a bounded 120-second process-group timeout, a unique dynamically selected rendezvous port, and no model downloads.

## Measured result

The full run completed 32 cycles on two AMD Instinct MI350X (`gfx950`) GPUs:

- 11 target parameters and 12 MTP parameters all had Megatron bridge mappings.
- All 23 parameters also had Miles name mappings.
- The eight inner MTP parameters were absent from the Megatron bridge under the legacy `transformer_layer` name, as expected, and all eight were mapped by the Miles dual-name hook.
- Maximum loss, gradient, and parameter rank differences were `0.0`.
- The minimum parameter update was `6.742775440216064e-07`; no cycle had a zero-update parameter.
- Maximum export/restore difference was `0.0`.
- GPU phase totals were approximately `2717.79 ms` forward, `762.24 ms` backward, `30.71 ms` optimizer, `64.25 ms` export, and `32.65 ms` restore.

The complete numerical record is in `gpu-results.json`.

## Limitation

The reduced Qwen3.5 configuration exposes a shape incompatibility in the installed Miles `mbridge` QKV tensor-merge hook. The fixture therefore uses the actual Megatron bridge conversion objects for tensor export/restore and uses the Miles hooks for name mapping and dual-name validation. This result does not claim validation of the original large-model topology.
