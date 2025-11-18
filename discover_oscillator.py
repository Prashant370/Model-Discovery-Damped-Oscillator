# pinned_per_traj_with_output.py
# Train one PINN per trajectory (float64), collect derivatives, then ElasticNet/Lasso -> weighted OLS debias.
# Saves results to dataset/output.json (same structure as original pipeline).

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.linear_model import LassoCV, LinearRegression, ElasticNetCV
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# Settings
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)   # double precision
DTYPE = torch.float64

out_dir = Path("dataset")
out_dir.mkdir(exist_ok=True)

# true system
zeta_true = 0.1
omega_true = 2 * math.pi
omega_d_true = omega_true * math.sqrt(max(0.0, 1 - zeta_true**2))

# trajectories
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
dt = t_grid[1] - t_grid[0]

# generate dataset (no noise)
rows = []
for tid, p in inits.items():
    A, B = p["A"], p["B"]
    x = np.exp(-zeta_true * omega_true * t_grid) * (A * np.cos(omega_d_true * t_grid) + B * np.sin(omega_d_true * t_grid))
    for ti, xi in zip(t_grid, x):
        rows.append({"traj_id": int(tid), "t": float(ti), "x": float(xi), "A": float(A), "B": float(B)})
df = pd.DataFrame(rows)
df.to_csv(out_dir / "data.csv", index=False)
with open(out_dir / "metadata.json", "w") as f:
    json.dump({"system":{"zeta":zeta_true,"omega":omega_true}, "trajectories": inits}, f, indent=2)

# -------------------------
# Per-trajectory model definition
# -------------------------
class SmallMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, t):
        return self.net(t)

def train_model_on_traj(t_np, x_np, epochs=8000, lr=1e-3, batch_size=128, use_dx_supervision=False, dx_weight=1.0):
    model = SmallMLP(hidden=128).to(device).to(dtype=DTYPE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(epochs*0.5), int(epochs*0.8)], gamma=0.1)
    loss_fn = nn.MSELoss()
    if use_dx_supervision:
        dx_fd = np.zeros_like(x_np).ravel()
        dx_fd[1:-1] = (x_np[2:,0] - x_np[:-2,0]) / (2*dt)
        dx_fd[0] = (x_np[1,0] - x_np[0,0]) / dt
        dx_fd[-1] = (x_np[-1,0] - x_np[-2,0]) / dt
        dx_fd = dx_fd.reshape(-1,1).astype(np.float64)
        dx_fd_t = torch.from_numpy(dx_fd).to(device).to(dtype=DTYPE)
    t_t = torch.from_numpy(t_np.astype(np.float64)).to(device).to(dtype=DTYPE)
    x_t = torch.from_numpy(x_np.astype(np.float64)).to(device).to(dtype=DTYPE)
    n = t_np.shape[0]
    indices = np.arange(n)
    for ep in range(epochs):
        perm = indices
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i+batch_size]
            bt = t_t[batch_idx].clone().detach().requires_grad_(True)
            bx = x_t[batch_idx]
            opt.zero_grad()
            pred = model(bt)
            loss_x = loss_fn(pred, bx)
            if use_dx_supervision:
                grads = torch.autograd.grad(pred, bt, grad_outputs=torch.ones_like(pred), create_graph=True, retain_graph=True)[0]
                dx_pred = grads[:,0:1]
                loss_dx = loss_fn(dx_pred, dx_fd_t[batch_idx])
                loss = loss_x + dx_weight * loss_dx
            else:
                loss = loss_x
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu().numpy()) * bt.shape[0]
        scheduler.step()
        if (ep+1) % 1000 == 0 or ep == 0:
            print(f"traj ep {ep+1}/{epochs} avg loss {epoch_loss/n:.3e}")
    return model

# -------------------------
# Train models per trajectory and collect derivatives
# -------------------------
x_pred_all = []
dx_dt_all = []
d2x_dt2_all = []

for tid, p in inits.items():
    A, B = p["A"], p["B"]
    t_np = t_grid.reshape(-1,1).astype(np.float64)
    x_np = (np.exp(-zeta_true * omega_true * t_grid) * (A * np.cos(omega_d_true * t_grid) + B * np.sin(omega_d_true * t_grid))).reshape(-1,1).astype(np.float64)

    print(f"\nTraining trajectory {tid} model...")
    model = train_model_on_traj(t_np, x_np, epochs=8000, lr=1e-3, batch_size=128, use_dx_supervision=False)

    t_t = torch.from_numpy(t_np).to(device).to(dtype=DTYPE).requires_grad_(True)
    with torch.set_grad_enabled(True):
        x_pred_t = model(t_t)
        grads = torch.autograd.grad(x_pred_t, t_t, grad_outputs=torch.ones_like(x_pred_t), create_graph=True)[0]
        dx_dt_t = grads[:,0:1]
        d2_all = torch.autograd.grad(dx_dt_t, t_t, grad_outputs=torch.ones_like(dx_dt_t), create_graph=True)[0]
        d2x_dt2_t = d2_all[:,0:1]
    x_pred_all.append(x_pred_t.detach().cpu().numpy().ravel())
    dx_dt_all.append(dx_dt_t.detach().cpu().numpy().ravel())
    d2x_dt2_all.append(d2x_dt2_t.detach().cpu().numpy().ravel())

x_pred_np = np.concatenate(x_pred_all)
dx_dt_np = np.concatenate(dx_dt_all)
d2x_dt2_np = np.concatenate(d2x_dt2_all)

# -------------------------
# Build regression library and filter low amplitude points
# -------------------------
Theta = np.vstack([x_pred_np, dx_dt_np]).T
y = d2x_dt2_np

amp = np.abs(x_pred_np)
keep_mask = amp > 1e-6
if keep_mask.sum() < 200:
    keep_mask = amp > 1e-9
Theta_use = Theta[keep_mask]
y_use = y[keep_mask]

col_means = Theta_use.mean(axis=0)
col_scales = Theta_use.std(axis=0)
col_scales[col_scales==0] = 1.0
Theta_norm = (Theta_use - col_means) / col_scales

# -------------------------
# Sparse regression: try ElasticNetCV first (less shrinkage), fallback to LassoCV
# -------------------------
print("\nRunning ElasticNetCV for selection (less shrinkage than Lasso)...")
try:
    enet = ElasticNetCV(l1_ratio=[0.1,0.5,0.9], cv=5, n_alphas=60, max_iter=20000).fit(Theta_norm, y_use)
    coef_norm = enet.coef_
    intercept_norm = enet.intercept_
    method_used = "ElasticNetCV"
except Exception as e:
    print("ElasticNet failed, falling back to LassoCV:", e)
    lasso = LassoCV(cv=5, n_alphas=80, max_iter=20000).fit(Theta_norm, y_use)
    coef_norm = lasso.coef_
    intercept_norm = lasso.intercept_
    method_used = "LassoCV"

coef_unscaled = coef_norm / col_scales
intercept_unscaled = float(intercept_norm - np.sum(col_means * coef_unscaled))

sel_mask = np.abs(coef_norm) > 1e-8
print(f"Selection mask (features used): {sel_mask} (feature order [x, xdot])")

# -------------------------
# Debias with weighted OLS (weight by |x| amplitude)
# -------------------------
weights = np.abs(Theta_use[:,0]) + 1e-6
if sel_mask.sum() == 0:
    sel_mask = np.array([True, True])

lr = LinearRegression()
lr.fit(Theta_use[:, sel_mask], y_use, sample_weight=weights)
coef_debiased = np.zeros(Theta_use.shape[1])
coef_debiased[sel_mask] = lr.coef_
intercept_debiased = float(lr.intercept_)

a_est = float(coef_debiased[0])
b_est = float(coef_debiased[1])
print(f"\nMethod: {method_used} selection + weighted OLS debias")
print(f"Debiased coefficients: a = {a_est:.6g}, b = {b_est:.6g}, intercept = {intercept_debiased:.6g}")

# -------------------------
# Convert to omega, zeta
# -------------------------
if a_est >= 0:
    omega_est = float('nan'); zeta_est = float('nan')
    print("a_est >= 0, cannot compute omega")
else:
    omega_est = math.sqrt(-a_est)
    zeta_est = -b_est / (2.0 * omega_est)

print(f"\nEstimated omega = {omega_est:.10g} (true {omega_true:.10g})")
print(f"Estimated zeta  = {zeta_est:.10g} (true {zeta_true:.10g})")

# -------------------------
# Simulate discovered ODE and compute RMS / nRMSE
# -------------------------
def rhs(t, y):
    return [y[1], a_est*y[0] + b_est*y[1]]

x_true_all = []
x_est_all = []
for tid,p in inits.items():
    A,B = p["A"], p["B"]
    x_true = np.exp(-zeta_true * omega_true * t_grid) * (A*np.cos(omega_d_true*t_grid) + B*np.sin(omega_d_true*t_grid))
    x0 = A
    dx0 = -zeta_true * omega_true * A + omega_d_true * B
    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), [x0, dx0], t_eval=t_grid, rtol=1e-9, atol=1e-12)
    x_est = sol.y[0]
    x_true_all.append(x_true)
    x_est_all.append(x_est)
x_true_all = np.concatenate(x_true_all); x_est_all = np.concatenate(x_est_all)
diff = x_est_all - x_true_all
rms = float(np.sqrt(np.mean(diff**2)))
x_range = float(x_true_all.max() - x_true_all.min())
nrmse = rms / x_range
omega_err_rel = abs(omega_est - omega_true) / abs(omega_true)
zeta_err_rel = abs(zeta_est - zeta_true) / abs(zeta_true)

print(f"\nRMS = {rms:.6g}, NRMSE = {nrmse:.6g}")
print(f"Relative errors -> omega: {omega_err_rel:.6g}, zeta: {zeta_err_rel:.6g}")

# -------------------------
# Save results to output.json (same structure as original)
# -------------------------
out = {
    "discovered_ode": "x_ddot = a*x + b*x_dot",
    "coefficients": {"a": a_est, "b": b_est, "intercept": intercept_debiased},
    "estimated_parameters": {"omega": omega_est, "zeta": zeta_est},
    "true_parameters": {"omega_true": float(omega_true), "zeta_true": float(zeta_true)},
    "errors": {
        "rms": rms,
        "nrmse": nrmse,
        "omega_rel_error": omega_err_rel,
        "zeta_rel_error": zeta_err_rel
    },
    "method": method_used
}

output_json = out_dir / "output.json"
with open(output_json, "w") as f:
    json.dump(out, f, indent=2)

print(f"Saved results to {output_json}")
