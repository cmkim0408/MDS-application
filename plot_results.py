import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# ----- 1️⃣ 경로 설정 -----
path = "results_stable/server_state_final.pt"
save_dir = os.path.dirname(path)

# ----- 2️⃣ 데이터 로드 -----
state = torch.load(path, map_location="cpu")
C_Fe = state["C_Fe"].numpy()
C_H = state["C_H"].numpy()
meta = state["meta"]

# ----- 3️⃣ 계산 -----
pH = -np.log10(C_H)
z_profile = pH.mean(axis=(0,1))           # z방향 평균 pH
z_axis = np.arange(len(z_profile)) * meta["dx"] * 1e3  # mm 단위

mid_z = pH.shape[2] // 2
pH_slice = pH[:,:,mid_z]
Fe_slice = C_Fe[:,:,mid_z]

# ----- 4️⃣ 그림 -----
plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(1, 3, figsize=(14,4))

# pH 단면
im0 = axes[0].imshow(pH_slice, cmap="viridis", origin="lower")
axes[0].set_title("pH (mid-plane)")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# Fe2+ 단면
im1 = axes[1].imshow(Fe_slice, cmap="plasma", origin="lower")
axes[1].set_title("[Fe$^{2+}$] (mid-plane, M)")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# pH depth profile
axes[2].plot(z_axis, z_profile, color="blue", lw=2)
axes[2].invert_xaxis()
axes[2].set_xlabel("Depth from ZVI surface (mm)")
axes[2].set_ylabel("pH")
axes[2].set_title("Mean pH vs Depth")
axes[2].grid(True)

plt.tight_layout()
save_path = os.path.join(save_dir, "Fig_S5_Fe2_pH.png")
plt.savefig(save_path, dpi=300)
print(f"[Saved] {save_path}")
