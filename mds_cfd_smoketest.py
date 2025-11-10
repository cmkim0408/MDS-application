#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MDS CFD-lite Smoke Test (Fe2+ + H+ diffusion–reaction)
------------------------------------------------------
Purpose:
  - Tiny 3D problem to sanity-check your environment (CPU or GPU).
  - Runs in seconds on a laptop; no heavy plotting by default.
  - Saves a small checkpoint "smoketest_state.pt" for inspection.

What it does (simplified physics for a smoke test):
  - 3D diffusion–reaction for Fe2+ and H+ with a surface source at z=0.
  - Explicit pseudo-time diffusion iteration (Jacobi/Gauss–Seidel style) with relaxation.
  - Dirichlet top boundary, Neumann lateral boundaries, and surface flux source.
  - Prints device, convergence metric |DeltaC/Delta t|_inf, and Delta pH (top - bottom).

Later (on server):
  - Increase grid size / steps; same code path works on GPU.
"""
import argparse
import time
import torch
import torch.nn.functional as F


def _build_padding_indices(shape, device):
    """Pre-compute index tensors used for Neumann padding via index_select."""
    nx, ny, nz = shape
    idx_x = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.arange(nx, device=device, dtype=torch.long),
            torch.full((1,), nx - 1, dtype=torch.long, device=device),
        )
    )
    idx_y = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.arange(ny, device=device, dtype=torch.long),
            torch.full((1,), ny - 1, dtype=torch.long, device=device),
        )
    )
    idx_z = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            torch.arange(nz, device=device, dtype=torch.long),
            torch.full((1,), nz - 1, dtype=torch.long, device=device),
        )
    )
    return idx_x, idx_y, idx_z


def laplacian_3d(z: torch.Tensor, dx: float, kernel: torch.Tensor, pad_idx, dirichlet_top: float) -> torch.Tensor:
    """
    3D Laplacian with explicit Neumann (x,y,bottom) and Dirichlet (top) handling.
    z: (nx, ny, nz)
    returns: Laplacian(z) with the same shape
    """
    idx_x, idx_y, idx_z = pad_idx
    z_neumann = z.index_select(0, idx_x).index_select(1, idx_y).index_select(2, idx_z)
    z_neumann[:, :, -1] = dirichlet_top  # overwrite the top ghost cell with fixed Dirichlet value
    z4d = z_neumann.unsqueeze(0).unsqueeze(0)  # (1,1,nx+2,ny+2,nz+2)
    out = F.conv3d(z4d, kernel) / (dx * dx)
    return out[0, 0]


def enforce_boundary_conditions(C_Fe: torch.Tensor, C_H: torch.Tensor, bulk_H: float) -> None:
    """Apply Dirichlet at top plane and Neumann (zero-gradient) on lateral faces."""
    # Dirichlet at top plane
    C_Fe[:, :, -1] = 0.0
    C_H[:, :, -1] = bulk_H

    # Neumann (zero normal gradient) on lateral faces by explicit index copy
    C_Fe[0, :, :] = C_Fe[1, :, :]
    C_Fe[-1, :, :] = C_Fe[-2, :, :]
    C_Fe[:, 0, :] = C_Fe[:, 1, :]
    C_Fe[:, -1, :] = C_Fe[:, -2, :]

    C_H[0, :, :] = C_H[1, :, :]
    C_H[-1, :, :] = C_H[-2, :, :]
    C_H[:, 0, :] = C_H[:, 1, :]
    C_H[:, -1, :] = C_H[:, -2, :]

def main():
    parser = argparse.ArgumentParser(description="MDS CFD-lite smoke test (tiny 3D diffusion–reaction).")
    parser.add_argument("--nx", type=int, default=20, help="grid size x (tiny default, increased for stability)")
    parser.add_argument("--ny", type=int, default=20, help="grid size y (tiny default, increased for stability)")
    parser.add_argument("--nz", type=int, default=5,  help="grid size z (tiny default, increased for stability)")
    parser.add_argument("--dx", type=float, default=1e-4, help="grid spacing (m)")
    parser.add_argument("--steps", type=int, default=2000, help="최대 반복 횟수 (pseudo-time iteration)")
    parser.add_argument("--alpha", type=float, default=0.3, help="relaxation factor α (권장 0.2~0.5)")
    parser.add_argument("--pseudo-dt", type=float, default=None, help="의사-시간 스텝 Delta t (미지정 시 안정 조건 기반 자동 설정)")
    parser.add_argument("--tol", type=float, default=1e-6, help="수렴 판정 |DeltaC/Delta t| 최대값 임계값")
    parser.add_argument("--report-every", type=int, default=100, help="로그 출력 주기 (반복 수 기준)")
    parser.add_argument("--plot", action="store_true", help="quicklook plot (optional)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SmokeTest] Device: {device}")
    torch.manual_seed(args.seed)

    # Grid & constants (kept small for laptop test)
    nx, ny, nz = args.nx, args.ny, args.nz
    dx = float(args.dx)
    D_Fe, D_H = 7e-10, 9e-9       # m^2/s
    k_surf    = 2e-6              # s^-1 (effective near-surface rate)
    C_sat     = 1e-2              # M  (cap Fe2+ to avoid divergence)
    bulk_pH   = 7.0
    bulk_H    = 10.0 ** (-bulk_pH)

    # State variables
    torch.set_grad_enabled(False)

    C_Fe = torch.zeros((nx, ny, nz), device=device, dtype=torch.float32)
    C_H  = torch.full((nx, ny, nz), bulk_H, device=device, dtype=torch.float32)

    # ZVI surface at z=0 plane
    mask_surface = torch.zeros_like(C_Fe)
    mask_surface[:, :, 0] = 1.0  # bottom plane as ZVI surface

    # 3D Laplacian kernel
    kernel = torch.zeros((1, 1, 3, 3, 3), device=device, dtype=torch.float32)
    kernel[0, 0, 1, 1, 1] = -6.0
    kernel[0, 0, 0, 1, 1] = kernel[0, 0, 2, 1, 1] = 1.0
    kernel[0, 0, 1, 0, 1] = kernel[0, 0, 1, 2, 1] = 1.0
    kernel[0, 0, 1, 1, 0] = kernel[0, 0, 1, 1, 2] = 1.0

    pad_idx = _build_padding_indices((nx, ny, nz), device)

    max_diffusivity = max(D_Fe, D_H)
    stability_dt = (dx * dx) / (2.0 * max_diffusivity)
    if args.pseudo_dt is None:
        pseudo_dt = 0.9 * stability_dt
    else:
        pseudo_dt = min(float(args.pseudo_dt), stability_dt)
    alpha = float(args.alpha)

    print(f"[SmokeTest] Grid: {nx} x {ny} x {nz} | dx={dx} m | steps<={args.steps}")
    print(f"[SmokeTest] Delta_t(limit)={stability_dt:.3e} s | Delta_t(use)={pseudo_dt:.3e} s | alpha={alpha}")
    print(f"[SmokeTest] Convergence tol (|DeltaC/Delta t|_inf) = {args.tol:.1e}")

    t0 = time.time()
    report_every = max(1, args.report_every)
    max_rate = float("inf")

    for step in range(1, args.steps + 1):
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
        C_H_new.clamp_(min=1e-9, max=1e-6)

        diff_Fe = torch.abs(C_Fe_new - C_Fe) / pseudo_dt
        diff_H = torch.abs(C_H_new - C_H) / pseudo_dt
        max_rate = torch.max(diff_Fe).item()
        max_rate = max(max_rate, torch.max(diff_H).item())

        C_Fe, C_H = C_Fe_new, C_H_new

        if step % report_every == 0 or step == 1 or max_rate < args.tol:
            pH_tmp = -torch.log10(C_H)
            delta_pH_tmp = pH_tmp[:, :, -1].mean() - pH_tmp[:, :, 0].mean()
            print(f"[SmokeTest] iter {step:5d} | |DeltaC/Delta t|_inf = {max_rate:.3e} | Delta pH = {delta_pH_tmp.item():.3f}")

        if max_rate < args.tol:
            print(f"[SmokeTest] Converged at iter {step} (|DeltaC/Delta t|_inf={max_rate:.3e} < tol).")
            break
    else:
        print(f"[SmokeTest] Reached max iterations without meeting tol (|DeltaC/Delta t|_inf={max_rate:.3e}).")

    elapsed = time.time() - t0

    # Quick stats
    with torch.no_grad():
        pH = -torch.log10(C_H)
        pH_bottom = pH[:, :, 0].mean().item()
        pH_top    = pH[:, :, -1].mean().item()
        delta_pH  = pH_top - pH_bottom
        print(f"[SmokeTest] Delta pH (top - bottom) ~ {delta_pH:.3f}")
        print(f"[SmokeTest] [Fe2+] min..max = {C_Fe.min().item():.2e} .. {C_Fe.max().item():.2e}")
        print(f"[SmokeTest] Runtime = {elapsed:.2f} s")

    # Save tiny checkpoint for verification
    torch.save({
        "C_Fe": C_Fe.detach().cpu(),
        "C_H":  C_H.detach().cpu(),
        "meta": {
            "nx": nx, "ny": ny, "nz": nz, "dx": dx,
            "D_Fe": D_Fe, "D_H": D_H, "k_surf": k_surf, "C_sat": C_sat,
            "bulk_pH": bulk_pH, "device": str(device)
        }
    }, "smoketest_state.pt")
    print("[SmokeTest] Saved: smoketest_state.pt")

    # Optional quicklook plot (single image, no fancy styling)
    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np
        mid_z = nz // 2
        pH_np  = (-torch.log10(C_H)).detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        
        # Mid-plane pH
        im1 = axes[0].imshow(pH_np[:, :, mid_z], cmap='viridis', origin='lower')
        axes[0].set_title(f"pH (mid-plane, z={mid_z})")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0])
        
        # Side view (x-z plane at y=ny//2)
        pH_xz = pH_np[:, ny//2, :]
        im2 = axes[1].imshow(pH_xz, cmap='viridis', origin='lower', aspect='auto')
        axes[1].set_title(f"pH (x-z plane, y={ny//2})")
        axes[1].set_xlabel("z (bottom=0, top={})".format(nz-1))
        axes[1].set_ylabel("x")
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig("smoketest_quicklook.png", dpi=150)
        print("[SmokeTest] Saved: smoketest_quicklook.png")
        plt.close()

if __name__ == "__main__":
    main()
