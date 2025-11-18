# Governing ODE Discovery — PINN + Sparse Regression

**Project**: discover the damped-oscillator ODE and parameters (ζ, ω) from time-series data.

---

## Objective

Generate synthetic trajectories for a damped oscillator, train a PINN to obtain smooth derivatives, apply sparse regression to discover the governing ODE and recover the parameters ζ and ω, and report accuracy metrics.

---

## Files produced by the pipeline

* `dataset/data.csv` — training data (columns: `traj_id,t,x`).
* `dataset/metadata.json` — ground-truth parameters (`zeta`, `omega`) and per-trajectory initial conditions (`A`, `B`).
* `dataset/output.json` — discovered ODE, coefficients, estimated parameters, and error metrics.

---

## Requirements

* Python 3.8+
* Packages (use `requirements.txt`):

  ```text
  numpy
  pandas
  scikit-learn
  scipy
  matplotlib
  torch==2.4.0+cpu
  ```

> **Important:** install the CPU PyTorch wheel. If pip attempts to download CUDA packages, cancel and run:
>
> ```bash
> pip install torch==2.4.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
> ```

---

## How to install

### Linux / WSL / macOS

```bash
pip install -r requirements.txt
# If torch CUDA wheel is attempted, use the CPU wheel command above instead.
```

### Windows (CMD / PowerShell)

```powershell
pip install -r requirements.txt
# If pip pulls CUDA, run the explicit CPU wheel command above.
```

---

## How to run

From project root run (Linux/macOS/Windows):

```bash
python discover_oscillator.py
```

The single script performs:

1. dataset generation (5 trajectories × 500 samples)
2. PINN training (t → x_pred)
3. autograd for ẋ and ẍ
4. sparse regression (LassoCV) to find `x_ddot = a*x + b*x_dot`
5. conversion to `omega` and `zeta`
6. ODE simulation and error metrics
7. writes `dataset/{data.csv,metadata.json,output.json}`

---

## Expected `output.json` structure

```json
{
  "discovered_ode": "x_ddot = a*x + b*x_dot",
  "coefficients": {"a": -39.48, "b": -1.256, "intercept": 0.0},
  "estimated_parameters": {"omega": 6.283, "zeta": 0.10},
  "true_parameters": {"omega_true": 6.283, "zeta_true": 0.10},
  "errors": {"rms": 0.024, "nrmse": 0.0038, "omega_rel_error": 0.001, "zeta_rel_error": 0.012}
}
```

---

## Evaluation metrics

* **RMS**: root mean square error between simulated trajectory from discovered ODE and ground-truth analytic trajectory.
* **NRMSE**: RMS divided by signal range (`x_max - x_min`).
* **Parameter relative errors** for `omega` and `zeta`.

---

## Notes and recommendations

* The PINN must be trained to obtain smooth, differentiable `x(t)` for stable autograd derivatives. Removing the PINN forces finite-difference derivatives which are noisy.
* Use 3–5 trajectories and 400–800 samples per trajectory in experiments. The shipped script uses 5×500 by default.
* If `a_est >= 0` after regression, the model recovery failed; try more data, reduce noise, or tune normalization/regularization.
* The script is single-file and produces all outputs in `dataset/`.

---

## Directory structure

```
project/
├─ discover_oscillator.py   # full pipeline
├─ requirements.txt
└─ dataset/
   ├─ data.csv
   ├─ metadata.json
   └─ output.json
```

---

## Extending the project

* Add measurement noise and test robustness.
* Replace Lasso with STRidge or SINDy packages for different sparsity strategies.
* Add command-line flags for number of trajectories, points, noise level, and training epochs.

---

## Author

Team: Paach Chawani

---
