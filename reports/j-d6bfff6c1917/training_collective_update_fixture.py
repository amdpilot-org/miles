#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().sum().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendezvous", default="tcp://127.0.0.1:31114")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--output", default="/job/artifacts/issue-920/training_collective_update.json")
    args = parser.parse_args()

    torch.manual_seed(920)
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method=args.rendezvous,
        world_size=1,
        rank=0,
        timeout=timedelta(seconds=args.timeout_seconds),
    )
    try:
        device = torch.device("cuda", 0)
        model = torch.nn.Linear(1024, 1024, device=device, dtype=torch.float32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        data = torch.randn(128, 1024, device=device, dtype=torch.float32)
        target = torch.randn(128, 1024, device=device, dtype=torch.float32)
        initial_loss = None
        cycles = []
        for cycle in range(args.cycles):
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            gradient = model.weight.grad
            if gradient is None:
                raise RuntimeError("missing gradient")
            dist.barrier()
            dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
            grad_norm = float(gradient.detach().float().norm().item())
            optimizer.step()
            torch.cuda.synchronize()
            loss_value = float(loss.detach().item())
            if initial_loss is None:
                initial_loss = loss_value
            cycles.append(
                {
                    "cycle": cycle,
                    "loss": loss_value,
                    "gradient_norm": grad_norm,
                    "weight_checksum": checksum(model.weight),
                    "gradient_finite": bool(torch.isfinite(gradient).all().item()),
                }
            )
        final_loss = cycles[-1]["loss"]
        result = {
            "backend": dist.get_backend(),
            "world_size": dist.get_world_size(),
            "rendezvous": args.rendezvous,
            "timeout_seconds": args.timeout_seconds,
            "device": torch.cuda.get_device_name(0),
            "dtype": "float32",
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_decreased": final_loss < initial_loss,
            "all_gradients_finite": all(row["gradient_finite"] for row in cycles),
            "cycles": cycles,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        if not result["loss_decreased"] or not result["all_gradients_finite"]:
            raise RuntimeError("training fixture numerical check failed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
