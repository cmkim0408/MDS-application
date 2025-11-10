
# MDS CFD-lite Smoke Test (Tiny 3D Diffusion–Reaction)

This is a **minimal** pseudo-time diffusion–reaction solver to validate your local environment before running the full GPU simulation on the server.

## What it does
- Tiny 3D steady-state **diffusion–reaction** system for **Fe²⁺** and **H⁺**.
- Localized Fe²⁺ generation at a "ZVI surface" (bottom plane) and bulk boundary at the top plane.
- Prints loss trend and **Delta pH (top - bottom)** as a quick sanity check.
- Saves a small checkpoint: `smoketest_state.pt`.
- Optional quicklook plot: `smoketest_quicklook.png` (single image).

## How to run (laptop)

### 기본 실행 (권장)
```bash
pip install torch matplotlib
python mds_cfd_smoketest.py --steps 1500 --plot
```

기본 설정은 pseudo-time diffusion iteration으로 안정성이 개선된 버전입니다:
- 격자 크기: 20×20×5
- 완화 계수: α = 0.3
- 의사-시간 스텝: Delta t = 0.9 × dx² / (2·max(D))
- 수렴 기준: |DeltaC/Delta t|_inf < 1e-6

### 안정성 조정

만약 수렴이 더디거나 진동이 관찰된다면, 다음 파라미터를 조정하세요.

1. **완화 계수 α 줄이기 (0.2~0.5 권장)**:
```bash
python mds_cfd_smoketest.py --alpha 0.25 --plot
```

2. **의사-시간 스텝을 더 보수적으로 설정**:
```bash
python mds_cfd_smoketest.py --pseudo-dt 2e-4 --plot
```

3. **수렴 허용치를 완화 또는 강화**:
```bash
python mds_cfd_smoketest.py --tol 5e-6  # 더 엄격한 수렴
```

### Expected console output (example)
[SmokeTest] Device: cpu
[SmokeTest] Grid: 20 x 20 x 5 | dx=0.0001 m | steps≤1500
[SmokeTest] Delta_t(limit)=5.56e-01 s | Delta_t(use)=5.00e-01 s | alpha=0.3
[SmokeTest] Convergence tol (|DeltaC/Delta t|_inf) = 1.0e-06
[SmokeTest] iter     1 | |DeltaC/Delta t|_inf = 2.13e-04 | Delta pH = 0.000
[SmokeTest] iter   100 | |DeltaC/Delta t|_inf = 1.12e-05 | Delta pH = 0.041
[SmokeTest] iter   300 | |DeltaC/Delta t|_inf = 4.87e-06 | Delta pH = 0.086
[SmokeTest] Converged at iter 342 (|DeltaC/Delta t|_inf=9.91e-07 < tol).
[SmokeTest] Delta pH(top - bottom) ~ 0.234
[SmokeTest] [Fe2+] min..max = 0.00e+00 .. 1.23e-04
[SmokeTest] Runtime = 3.45 s
[SmokeTest] Saved: smoketest_state.pt
[SmokeTest] Saved: smoketest_quicklook.png   # only if --plot is used
```

### 정상 동작 확인 포인트
- ✅ `Device: cpu` (또는 `cuda`) 표시
- ✅ |DeltaC/Delta t|_inf (최대 변화율)이 감소하며 tol 이하로 떨어짐
- ✅ Delta pH(top - bottom)가 양수 (아래쪽이 약간 더 산성 → pH 낮음)
- ✅ `smoketest_state.pt` 파일 생성
- ✅ `--plot` 사용 시 `smoketest_quicklook.png` 생성 (2개의 subplot: mid-plane과 side view)

### 정상 출력 이미지의 특징
정상적으로 동작하면 이미지가 다음과 같이 보입니다:
- pH 값이 하단(0 plane) 근처에서 낮고, 위로 갈수록 완만히 증가
- 색상은 위쪽이 더 노란/밝고, 아래쪽이 어두운 gradation
- **Checkerboard 패턴이 없어야 함** (패턴이 있다면 진동 상태)

> **Note:** Exact numbers will vary, but you should see |DeltaC/Delta t|_inf 감소와 양의 Delta pH(top - bottom) (bottom slightly 더 산성).

## Move to server later
You can reuse the same script and just change parameters:
```bash
python mds_cfd_smoketest.py --nx 200 --ny 200 --nz 50 --steps 20000 --alpha 0.25 --pseudo-dt 2e-4 --tol 5e-7 --plot
```
- On GPU, it will print `Device: cuda` automatically if CUDA is available.
- Increase grid/steps as needed for production-quality figures.
- Server에서는 더 큰 격자와 많은 스텝을 사용하므로 alpha, Delta t를 조금 더 보수적으로 잡는 것이 안전합니다.

## Files
- `mds_cfd_smoketest.py`: the tiny smoke test script.
- `mds_cfd_server.py`: production-scale runner with checkpointing and resume support.
- `smoketest_state.pt`: saved checkpoint containing tensors and metadata.
- `smoketest_quicklook.png`: optional quicklook (only if `--plot` is used).

---

## Server 런너 (`mds_cfd_server.py`)

### 기본 실행 예시
```bash
python mds_cfd_server.py \
  --nx 200 --ny 200 --nz 50 \
  --steps 20000 \
  --alpha 0.25 \
  --tol 5e-7 \
  --output-dir runs/exp001 \
  --save-final
```

### 주요 기능
- GPU/CPU 자동 선택 (`--device`로 강제 지정 가능)
- 의사-시간 스텝 자동 산정(`--dt-factor`) 또는 수동 지정(`--pseudo-dt`)
- 주기적 체크포인트 저장(`--checkpoint-interval`)
- 저장된 상태(`.pt`)로부터 재시작(`--initial-state`)
- `--precision float64`로 고정밀 계산 가능
- `server_state_stepXXXXXX.pt`와 `server_state_final.pt`에 텐서 및 메타데이터 저장

### 체크포인트 재개
```bash
python mds_cfd_server.py \
  --initial-state runs/exp001/server_state_step010000.pt \
  --steps 30000 \
  --output-dir runs/exp001_resume
```

### 권장 팁
- 서버에서는 `alpha`와 `Delta t`를 조금 보수적으로 설정하면 진동 없이 수렴
- CUDA 환경에서 `--device cuda:0`과 `--precision float32` 조합이 가장 빠름
- 로그(`--log-interval`)와 체크포인트 주기를 workload에 맞춰 조절하세요.
