# plot_results_refined.py
"""
Refined visualization for CFD-MDS stable simulation results
Generates publication-quality figure (pH and Fe2+ distribution)
Author: C. Kim
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 1️⃣ 경로 설정 =====
PATH = "results_stable/server_state_final.pt"
SAVE_DIR = os.path.dirname(PATH)
SAVE_NAME = "Fig_S5_Fe2_pH_refined.png"

# ===== 2️⃣ 데이터 로드 =====
state = torch.load(PATH, map_location="cpu")
C_Fe = state["C_Fe"].numpy()
C_H = state["C_H"].numpy()
meta = state.get("meta", {"dx": 1e-4})

# ===== 3️⃣ 계산 =====
pH = -np.log10(C_H)
z_profile = pH.mean(axis=(0, 1))
z_axis = np.arange(len(z_profile)) * meta["dx"] * 1e3  # mm 단위

mid_z = pH.shape[2] // 2
pH_slice = pH[:, :, mid_z]
Fe_slice = C_Fe[:, :, mid_z]

# ===== 4️⃣ 시각화 설정 =====
plt.rcParams.update({
    "font.family": "Helvetica",
    "font.size": 12,
    "axes.linewidth": 1,
    "axes.edgecolor": "black",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "xtick.direction": "in",
    "ytick.direction": "in",
})

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# --- pH 단면 ---
vmin_pH, vmax_pH = pH_slice.min(), pH_slice.max()
im0 = axes[0].imshow(pH_slice, cmap="viridis", origin="lower",
                     vmin=vmin_pH, vmax=vmax_pH)
axes[0].set_title("pH (mid-plane)")
axes[0].set_xlabel("x (index)")
axes[0].set_ylabel("y (index)")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="pH")

# --- Fe2+ 로그 단면 ---
Fe_log = np.log10(Fe_slice + 1e-12)
vmax_Fe = np.nanmax(Fe_log)
vmin_Fe = vmax_Fe - 2.0  # 상위 2-log 범위 강조
im1 = axes[1].imshow(Fe_log, cmap="plasma", origin="lower",
                     vmin=vmin_Fe, vmax=vmax_Fe)
axes[1].set_title("log$_{10}$[Fe$^{2+}$] (mid-plane, M)")
axes[1].set_xlabel("x (index)")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="log$_{10}$(M)")

# --- pH depth profile ---
axes[2].plot(z_axis, z_profile, color="navy", lw=2)
axes[2].invert_xaxis()
axes[2].set_xlabel("Depth from ZVI surface (mm)")
axes[2].set_ylabel("Mean pH")
axes[2].set_title("Mean pH vs Depth")
axes[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
save_path = os.path.join(SAVE_DIR, SAVE_NAME)
plt.savefig(save_path, dpi=400)
plt.show()

print(f"[Saved] {save_path}")
print(f"[pH range] {vmin_pH:.3f}–{vmax_pH:.3f}")
print(f"[Fe2+] log range] {vmin_Fe:.2f}–{vmax_Fe:.2f}")
