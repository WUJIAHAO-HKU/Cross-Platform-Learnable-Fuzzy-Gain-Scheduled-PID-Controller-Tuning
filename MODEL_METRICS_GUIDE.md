# Model Metrics Guide

This document specifies the online (in-model) and offline (post-processing) metrics for the hierarchical hybrid control project.

## 1. Signal Naming (Timeseries Recommended)
| Name | Dim | Description |
|------|-----|-------------|
| q | 7x1 | Joint positions |
| q_ref | 7x1 | Joint reference positions |
| u | 7x1 | Applied joint torques (controller output) |
| Joint_Error | 7x1 | q_ref - q |
| err_sq | 7x1 | Joint-wise squared error |
| e2_sum | 1 | Sum(err_sq) |
| int_e2 | 1 | Integral of e2_sum (∫ Σ e_i^2 dt) |
| mse | 1 | int_e2 / (t * 7 + ε) |
| rmse_q | 1 | sqrt(mse) |
| err_norm | 1 | sqrt(e2_sum) (instantaneous L2 norm) |
| u2_sum | 1 | Σ u_i^2 |
| energy_u | 1 | ∫ Σ u_i^2 dt |

## 2. Online Block Graph (Core Metrics)
1. Subtract(q_ref, q) → Joint_Error
2. Element-Wise Product (Joint_Error .* Joint_Error) → err_sq
3. Sum (err_sq, all elements) → e2_sum
4. Integrator(e2_sum) → int_e2
5. Clock → t
6. Gain(t, 7) → t7
7. Sum(t7, ε=1e-9 Const) → denom
8. Divide(int_e2, denom) → mse
9. Math Function sqrt(mse) → rmse_q
10. Element-Wise Product(u .* u) → u2
11. Sum(u2) → u2_sum
12. Integrator(u2_sum) → energy_u
13. Math Function sqrt(e2_sum) → err_norm

## 3. Offline Metrics (compute_run_statistics.m)
| Metric | Field | Description |
|--------|-------|-------------|
| Final RMSE | rmse_final | rmse_q(end) or recomputed |
| Final energy | energy_final | energy_u(end) |
| Joint RMSE vector | joint_rmse_vector | sqrt(mean(e_i^2)) per joint |
| Overshoot | overshoot_per_joint | (max(q_j) - ref_final)/ref_final |
| Settling time | settling_time | Max time all joints enter band |
| Steady-state error | steady_state_error | q_j(end)-ref_final |
| Error quantiles | error_quantiles | Flattened abs error distribution |
| Max velocity | max_velocity | Estimated via finite diff |
| Mean | mean_abs_torque | Mean absolute torque |
| Energy density | control_energy_density | energy_final / total_time |

## 4. Lyapunov / Safety (Planned Extensions)
Add signals:
- lyapunov_V = e^T Q e + e_dot^T R e_dot
- dV_dt (approx via finite diff or analytical if modelled)
- cbf_h (barrier) & cbf_margin

## 5. Reward Shaping (Future RL Integration)
Candidate decomposition (weight later):
- r_tracking = -rmse_q
- r_energy = -λ * u2_sum (scaled)
- r_smooth = -μ * ||Δu||^2 (need Delay + Subtract)
- r_safety = -κ * max(0, -cbf_margin)

## 6. Implementation Tips
- Use timeseries output format for consistent post-processing.
- Add Bus Creator to group metrics for RL access.
- Add small ε in denominators to avoid divide-by-zero at t≈0.
- Keep Integrators resettable if you introduce episodic training.

## 7. Validation Checklist
- [ ] rmse_q decreases and stabilizes
- [ ] energy_u monotonic increasing only
- [ ] err_norm near zero in steady state
- [ ] Overshoot within design bounds (<15%)
- [ ] Settling time consistent across minor parameter tweaks

## 8. Reproducibility
When capturing baselines: store
- Controller gains (Kp, Kd)
- Trajectory spec (type, amplitude, duration)
- Random seed
- Model revision hash (manual tag)

## 9. Future Extensions
- Add joint limit proximity metric: min(q_upper - q, q - q_lower)
- Add torque saturation ratio: mean(|u| > limit * 0.95)
- Add frequency content (FFT of error) for oscillation detection

---
This guide supports consistent metric generation and interpretation for the hierarchical hybrid control study.
