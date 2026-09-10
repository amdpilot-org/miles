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
