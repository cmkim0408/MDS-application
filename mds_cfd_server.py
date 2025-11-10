#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MDS CFD-lite Production Runner (Pseudo-time diffusion solver)
-------------------------------------------------------------
- Runs the stabilized diffusion–reaction iteration on larger 3D grids.
- Supports GPU/CPU execution, checkpointing, and resume from saved state.
- Built on top of the v2 pseudo-time solver from the smoke test.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from mds_cfd_smoketest import (
    _build_padding_indices,
    enforce_boundary_conditions,
    laplacian_3d,
)


def load_initial_state(path: Path, device: torch.device, dtype: torch.dtype):
    state = torch.load(path, map_location=device)
    C_Fe = state["C_Fe"].to(device=device, dtype=dtype).contiguous()
    C_H = state["C_H"].to(device=device, dtype=dtype).contiguous()
    meta = state.get("meta", {})
    return C_Fe, C_H, meta


def save_checkpoint(path: Path, C_Fe, C_H, meta):
    payload = {
        "C_Fe": C_Fe.detach().cpu(),
        "C_H": C_H.detach().cpu(),
        "meta": meta,
    }
    torch.save(payload, path)


def run_solver(args):
    torch.set_grad_enabled(False)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float64 if args.precision == "float64" else torch.float32
    print(f"[ServerRun] Device: {device} (requested: {args.device or 'auto'})")
    print(f"[ServerRun] Precision: {dtype}")

    nx, ny, nz = args.nx, args.ny, args.nz
    dx = float(args.dx)
    D_Fe, D_H = args.diffusivity_fe, args.diffusivity_h
    k_surf = args.k_surface
    C_sat = args.c_sat
    bulk_pH = args.bulk_pH
    bulk_H = 10.0 ** (-bulk_pH)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.initial_state:
        init_path = Path(args.initial_state)
        print(f"[ServerRun] Loading initial state: {init_path}")
        C_Fe, C_H, meta = load_initial_state(init_path, device, dtype)
        if C_Fe.shape != (nx, ny, nz):
            raise ValueError(
                f"Initial state grid {tuple(C_Fe.shape)} != requested {(nx, ny, nz)}"
            )
        print(f"[ServerRun] Initial state meta: {json.dumps(meta, indent=2)}")
    else:
        C_Fe = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
        C_H = torch.full((nx, ny, nz), bulk_H, device=device, dtype=dtype)

    mask_surface = torch.zeros_like(C_Fe)
    mask_surface[:, :, 0] = 1.0

    kernel = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=dtype)
    kernel[0, 0, 1, 1, 1] = -6.0
    kernel[0, 0, 0, 1, 1] = kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = kernel[0, 0, 1, 1, 2] = 1.0

    pad_idx = _build_padding_indices((nx, ny, nz), device)

    max_diffusivity = max(D_Fe, D_H)
    stability_dt = (dx * dx) / (2.0 * max_diffusivity)
    if args.pseudo_dt is None:
        pseudo_dt = args.dt_factor * stability_dt
    else:
        pseudo_dt = min(float(args.pseudo_dt), stability_dt)
    alpha = float(args.alpha)

    print(f"[ServerRun] Grid: {nx} x {ny} x {nz} | dx={dx} m")
    print(f"[ServerRun] Delta_t(limit)={stability_dt:.3e} s | Delta_t(use)={pseudo_dt:.3e} s | alpha={alpha}")
    print(f"[ServerRun] Convergence tol = {args.tol:.2e}")
    print(f"[ServerRun] Max steps = {args.steps}")
    print(f"[ServerRun] Log interval = every {args.log_interval} steps")
    print(f"[ServerRun] Checkpoint interval = every {args.checkpoint_interval} steps")

    report_every = max(1, args.log_interval)
    ckpt_every = max(1, args.checkpoint_interval)

    start_step = 0
    if args.initial_state:
        start_step = int(meta.get("step", 0))
        print(f"[ServerRun] Resuming from step {start_step}")

    t0 = time.time()
    max_rate = float("inf")

    for step in range(start_step + 1, args.steps + 1):
        enforce_boundary_conditions(C_Fe, C_H, bulk_H)

        lap_Fe = laplacian_3d(C_Fe, dx, kernel, pad_idx, dirichlet_top=0.0)
        lap_H = laplacian_3d(C_H, dx, kernel, pad_idx, dirichlet_top=bulk_H)

        R = k_surf * mask_surface * torch.clamp(1.0 - C_Fe / C_sat, min=0.0)

        delta_Fe = D_Fe * lap_Fe + R
        delta_H = D_H * lap_H - 2.0 * R

        C_Fe_new = C_Fe + alpha * pseudo_dt * delta_Fe
        C_H_new = C_H + alpha * pseudo_dt * delta_H

        enforce_boundary_conditions(C_Fe_new, C_H_new, bulk_H)
        C_Fe_new.clamp_(min=0.0, max=C_sat)
        C_H_new.clamp_(min=1e-12, max=1e-5)

        diff_Fe = torch.abs(C_Fe_new - C_Fe) / pseudo_dt
        diff_H = torch.abs(C_H_new - C_H) / pseudo_dt
        max_rate = torch.max(diff_Fe).item()
        max_rate = max(max_rate, torch.max(diff_H).item())

        C_Fe, C_H = C_Fe_new, C_H_new

        if step % report_every == 0 or step == start_step + 1 or max_rate < args.tol:
            pH_tmp = -torch.log10(C_H)
            delta_pH_tmp = pH_tmp[:, :, -1].mean() - pH_tmp[:, :, 0].mean()
            print(
                f"[ServerRun] iter {step:7d} | |DeltaC/Delta t|_inf = {max_rate:.3e} | Delta pH = {delta_pH_tmp.item():.3f}"
            )

        if step % ckpt_every == 0 or max_rate < args.tol:
            meta = {
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "dx": dx,
                "D_Fe": D_Fe,
                "D_H": D_H,
                "k_surf": k_surf,
                "C_sat": C_sat,
                "bulk_pH": bulk_pH,
                "device": str(device),
                "precision": str(dtype),
                "step": step,
                "delta_t": pseudo_dt,
                "alpha": alpha,
                "tol": args.tol,
                "max_rate": max_rate,
                "timestamp": time.time(),
            }
            ckpt_name = f"server_state_step{step:06d}.pt"
            save_checkpoint(output_dir / ckpt_name, C_Fe, C_H, meta)
            print(f"[ServerRun] Saved checkpoint: {ckpt_name}")

        if max_rate < args.tol:
            print(f"[ServerRun] Converged at iter {step} (|DeltaC/Delta t|_inf={max_rate:.3e} < tol).")
            break
    else:
        print(f"[ServerRun] Reached max iterations without meeting tol (|DeltaC/Delta t|_inf={max_rate:.3e}).")

    elapsed = time.time() - t0
    pH = -torch.log10(C_H)
    pH_bottom = pH[:, :, 0].mean().item()
    pH_top = pH[:, :, -1].mean().item()
    delta_pH = pH_top - pH_bottom

    print(f"[ServerRun] Delta pH (top - bottom) ~ {delta_pH:.3f}")
    print(f"[ServerRun] [Fe2+] min..max = {C_Fe.min().item():.3e} .. {C_Fe.max().item():.3e}")
    print(f"[ServerRun] Runtime = {elapsed/60.0:.2f} min")

    if args.save_final:
        final_meta = {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "D_Fe": D_Fe,
            "D_H": D_H,
            "k_surf": k_surf,
            "C_sat": C_sat,
            "bulk_pH": bulk_pH,
            "device": str(device),
            "precision": str(dtype),
            "step": step,
            "delta_t": pseudo_dt,
            "alpha": alpha,
            "tol": args.tol,
            "max_rate": max_rate,
            "elapsed_s": elapsed,
        }
        final_path = output_dir / "server_state_final.pt"
        save_checkpoint(final_path, C_Fe, C_H, final_meta)
        print(f"[ServerRun] Saved final state: {final_path.name}")


def parse_args():
    parser = argparse.ArgumentParser(description="MDS CFD-lite production runner.")
    parser.add_argument("--nx", type=int, default=200, help="grid size along x")
    parser.add_argument("--ny", type=int, default=200, help="grid size along y")
    parser.add_argument("--nz", type=int, default=50, help="grid size along z")
    parser.add_argument("--dx", type=float, default=1e-4, help="grid spacing (m)")
    parser.add_argument("--steps", type=int, default=20000, help="maximum pseudo-time iterations")
    parser.add_argument("--alpha", type=float, default=0.25, help="relaxation factor")
    parser.add_argument("--pseudo-dt", type=float, default=None, help="explicit pseudo-time step (s)")
    parser.add_argument("--dt-factor", type=float, default=0.8, help="fraction of stability dt when pseudo-dt is None")
    parser.add_argument("--tol", type=float, default=5e-7, help="convergence tolerance on |DeltaC/Delta t|_inf")
    parser.add_argument("--log-interval", type=int, default=50, help="logging interval (iterations)")
    parser.add_argument("--checkpoint-interval", type=int, default=500, help="checkpoint interval (iterations)")
    parser.add_argument("--output-dir", type=str, default="server_outputs", help="directory for checkpoints/results")
    parser.add_argument("--initial-state", type=str, default=None, help="optional checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="device override, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument("--precision", type=str, default="float32", choices=["float32", "float64"], help="floating precision")
    parser.add_argument("--diffusivity-fe", type=float, default=7e-10, help="Fe2+ diffusivity (m^2/s)")
    parser.add_argument("--diffusivity-h", type=float, default=9e-9, help="H+ diffusivity (m^2/s)")
    parser.add_argument("--k-surface", type=float, default=2e-6, help="surface reaction rate constant (s^-1)")
    parser.add_argument("--c-sat", type=float, default=1e-2, help="Fe2+ saturation concentration (M)")
    parser.add_argument("--bulk-pH", type=float, default=7.0, help="bulk pH at top boundary")
    parser.add_argument("--save-final", action="store_true", help="save final state checkpoint at the end")
    parser.add_argument("--no-save-final", dest="save_final", action="store_false", help="disable final state save")
    parser.set_defaults(save_final=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_solver(args)


if __name__ == "__main__":
    main()


