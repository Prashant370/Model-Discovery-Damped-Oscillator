# Author: Team - paach chawani 

"""
Full pipeline:
- dataset generation (5 traj x 500 pts)
- save data.csv, metadata.json
- PINN training (t -> x_pred)
- derivatives via autograd
- sparse regression (LassoCV) to discover x_ddot = a*x + b*x_dot
- convert to omega, zeta
- simulate discovered ODE for each traj and compute RMS, NRMSE
- save output.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LassoCV
from scipy.integrate import solve_ivp
import math

# -------------------------
# 1) dataset generation
# -------------------------
out_dir = Path("dataset")
out_dir.mkdir(exist_ok=True)

# true system (hidden from model)
zeta_true = 0.1
omega_true = 2 * np.pi
omega_d_true = omega_true * np.sqrt(max(0.0, 1 - zeta_true ** 2))

# initial conditions for 5 trajectories (A,B)
inits = {
    1: {"A": 25.0, "B": 64.0},
    2: {"A": 12.0, "B": -5.0},
    3: {"A": -20.0, "B": 30.0},
    4: {"A": 5.0, "B": 15.0},
    5: {"A": -10.0, "B": -25.0},
}

n_traj = len(inits)
n_points = 500
t0, t1 = 0.0, 10.0
t_grid = np.linspace(t0, t1, n_points)

rows = []
noise_std = 0.0  # set >0.0 to add measurement noise
for tid, p in inits.items():
    A, B = p["A"], p["B"]
    x = np.exp(-zeta_true * omega_true * t_grid) * (A * np.cos(omega_d_true * t_grid) + B * np.sin(omega_d_true * t_grid))
    if noise_std > 0:
        x = x + np.random.normal(0, noise_std, size=x.shape)
    for ti, xi in zip(t_grid, x):
        rows.append({"traj_id": int(tid), "t": float(ti), "x": float(xi)})

df = pd.DataFrame(rows)
data_csv = out_dir / "data.csv"
df.to_csv(data_csv, index=False)

meta = {"system": {"zeta": float(zeta_true), "omega": float(omega_true)}, "trajectories": inits}
meta_json = out_dir / "metadata.json"
with open(meta_json, "w") as f:
    json.dump(meta, f, indent=2)

print(f"Saved dataset: {data_csv} and metadata: {meta_json}")

# -------------------------
# 2) PINN: simple MLP fit
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# prepare training data (all trajectories combined)
T = df["t"].values.astype(np.float32)[:, None]
X = df["x"].values.astype(np.float32)[:, None]

t_tensor = torch.from_numpy(T).to(device)
x_tensor = torch.from_numpy(X).to(device)

# define MLP
class MLP(nn.Module):
    def __init__(self, layers=[1, 64, 64, 64, 1], act=nn.Tanh):
        super().__init__()
        seq = []
        for i in range(len(layers)-1):
            seq.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                seq.append(act())
        self.net = nn.Sequential(*seq)
    def forward(self, t):
        return self.net(t)

model = MLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# training (fit x(t) only)
n_epochs = 3000
batch_size = 1024
dataset = torch.utils.data.TensorDataset(t_tensor, x_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

print("Training PINN to fit x(t)...")
for ep in range(n_epochs):
    running = 0.0
    for bt, bx in loader:
        opt.zero_grad()
        pred = model(bt)
        loss = loss_fn(pred, bx)
        loss.backward()
        opt.step()
        running += loss.item() * bt.shape[0]
    if (ep+1) % 500 == 0 or ep == 0:
        print(f"epoch {ep+1}/{n_epochs} loss {running/len(dataset):.4e}")
print("PINN training finished.")

# -------------------------
# 3) evaluate PINN and compute derivatives
# -------------------------
# We'll evaluate on a sufficiently dense grid combining all original t points per trajectory.
# Use unique t_grid (same used in generation) to compute derivative fields per trajectory.
t_eval = np.unique(df["t"].values).astype(np.float32)
t_eval_tensor = torch.from_numpy(t_eval.reshape(-1,1)).to(device)
t_eval_tensor.requires_grad_(True)

with torch.set_grad_enabled(True):
    x_pred = model(t_eval_tensor)                         # (N,1)
    dx_dt = torch.autograd.grad(x_pred, t_eval_tensor, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t_eval_tensor, grad_outputs=torch.ones_like(dx_dt), create_graph=True)[0]

x_pred_np = x_pred.detach().cpu().numpy().ravel()
dx_dt_np = dx_dt.detach().cpu().numpy().ravel()
d2x_dt2_np = d2x_dt2.detach().cpu().numpy().ravel()

# map these back to full dataset rows (matching t values)
# For regression it's fine to use the dense unique t_eval points
# Build Theta = [x, x_dot] and target y = x_ddot
Theta = np.vstack([x_pred_np, dx_dt_np]).T
y = d2x_dt2_np

# optionally filter near-zero amplitude times to avoid ill-conditioning
amp = np.abs(x_pred_np)
keep_mask = amp > 1e-6  # discard points where x ~ 0 (optional)
if keep_mask.sum() < 50:
    keep_mask = np.ones_like(amp, dtype=bool)

Theta_use = Theta[keep_mask]
y_use = y[keep_mask]

# normalize Theta columns for numerical stability
col_means = Theta_use.mean(axis=0)
col_scales = Theta_use.std(axis=0)
col_scales[col_scales == 0] = 1.0
Theta_norm = (Theta_use - col_means) / col_scales

# -------------------------
# 4) sparse regression (LassoCV)
# -------------------------
print("Running LassoCV sparse regression...")
lasso = LassoCV(cv=5, n_alphas=50, max_iter=20000).fit(Theta_norm, y_use)
coef_norm = lasso.coef_  # coefficients for normalized Theta
intercept = lasso.intercept_

# un-normalize coefficients: y = (Theta - mean)/scale * coef_norm + intercept
# So original coef = coef_norm/scale, and addition: intercept_adjust = intercept - sum(mean/scale * coef_norm)
coef = coef_norm / col_scales
intercept_adjust = intercept - np.sum(col_means * coef)

a_est = float(coef[0])   # coefficient on x
b_est = float(coef[1])   # coefficient on x_dot

print(f"discovered coefficients: a = {a_est:.6g}, b = {b_est:.6g}, intercept = {float(intercept_adjust):.6g}")

# -------------------------
# 5) convert to zeta, omega
# -------------------------
# a = -omega^2, b = -2 zeta omega
if a_est >= 0:
    omega_est = float('nan')
    zeta_est = float('nan')
    print("Warning: a_est >= 0, cannot compute omega = sqrt(-a).")
else:
    omega_est = math.sqrt(-a_est)
    zeta_est = -b_est / (2.0 * omega_est)

print(f"estimated omega = {omega_est:.6g}, estimated zeta = {zeta_est:.6g}")

# -------------------------
# 6) simulate discovered ODE and compute errors
# -------------------------
# ODE form: x_ddot = a_est * x + b_est * x_dot
def rhs(t, y):
    # y = [x, x_dot]
    return [y[1], a_est * y[0] + b_est * y[1]]

# For comparison we simulate each trajectory on the original t_grid and compute errors.
x_true_all = []
x_est_all = []
t_all = []

for tid, p in inits.items():
    A, B = p["A"], p["B"]
    # initial conditions from analytic solution at t=0:
    x0 = A
    dx0 = B * omega_d_true  # careful: if using representation x = e^{-zeta w t}(A cos + B sin) then at t=0:
    # derivative at t=0 is: dx/dt|0 = -zeta*omega*A + omega_d*( -A*0? ) Wait simpler: compute from analytic formula explicitly:
    # compute analytic derivative at t=0 for consistency
    def analytic_x_and_dx_at0(A, B):
        # x(t) = e^{-zeta w t} (A cos(wd t) + B sin(wd t))
        # dx/dt = e^{-z w t} [ -z w (A cos + B sin) + (-A wd sin + B wd cos) ]
        x0 = A
        dx0 = -zeta_true * omega_true * A + omega_d_true * B
        return x0, dx0
    x0, dx0 = analytic_x_and_dx_at0(A, B)

    # simulate discovered ODE on the same time grid
    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), [x0, dx0], t_eval=t_grid, rtol=1e-8, atol=1e-10, method='RK45')
    x_est = sol.y[0]
    # compute true (noise-free) analytic x(t) for this A,B
    x_true = np.exp(-zeta_true * omega_true * t_grid) * (A * np.cos(omega_d_true * t_grid) + B * np.sin(omega_d_true * t_grid))
    x_true_all.append(x_true)
    x_est_all.append(x_est)
    t_all.append(t_grid)

# flatten
x_true_all = np.concatenate(x_true_all)
x_est_all = np.concatenate(x_est_all)
t_all = np.concatenate(t_all)

# compute RMS and NRMSE
diff = x_est_all - x_true_all
rms = float(np.sqrt(np.mean(diff**2)))
x_range = float(x_true_all.max() - x_true_all.min())
nrmse = float(rms / x_range) if x_range != 0 else float('nan')

# parameter relative errors
omega_err_rel = abs(omega_est - omega_true) / abs(omega_true) if not math.isnan(omega_est) else float('nan')
zeta_err_rel = abs(zeta_est - zeta_true) / abs(zeta_true) if not math.isnan(zeta_est) else float('nan')

# -------------------------
# 7) save output.json
# -------------------------
out = {
    "discovered_ode": "x_ddot = a*x + b*x_dot",
    "coefficients": {"a": a_est, "b": b_est, "intercept": float(intercept_adjust)},
    "estimated_parameters": {"omega": omega_est, "zeta": zeta_est},
    "true_parameters": {"omega_true": float(omega_true), "zeta_true": float(zeta_true)},
    "errors": {
        "rms": rms,
        "nrmse": nrmse,
        "omega_rel_error": omega_err_rel,
        "zeta_rel_error": zeta_err_rel
    }
}

output_json = out_dir / "output.json"
with open(output_json, "w") as f:
    json.dump(out, f, indent=2)

print(f"Saved results to {output_json}")
print("Summary:")
print(f"  a = {a_est:.6g}, b = {b_est:.6g}")
print(f"  omega_est = {omega_est:.6g} (true {omega_true:.6g}), zeta_est = {zeta_est:.6g} (true {zeta_true:.6g})")
print(f"  RMS = {rms:.6g}, NRMSE = {nrmse:.6g}")
