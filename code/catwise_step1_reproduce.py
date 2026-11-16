#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Sample, mask and dipole of Secrest et al. (2021) before any template."""
import numpy as np
import healpy as hp
import catwise as cw

cat = cw.load_catalogue()
m = cw.mask()
N = cw.count_map(cat)
m &= N > 0  # pixels emptied by the catalogue-level masks
np.save(cw.OUT_CW / "mask.npy", m)
print(f"{len(cat['ra'])} quasars, {m.sum()} pixels unmasked (f_sky = {m.mean():.3f}), N in mask {N[m].sum():.0f}")

Dk, x, alpha = cw.kinematic_amplitude(cat)
print(f"kinematic expectation: x = {x:.2f}, alpha = {alpha:.2f}, D_kin = {Dk:.4f}  (Secrest et al. 2021: x ~ 1.7, alpha = 1.26, D_kin = 0.0072)")

d = cw.secrest_dipole(N, m)
print("Secrest estimator  :", cw.describe(d), f" ratio to kinematic {np.linalg.norm(d)/Dk:.2f}")
print("   (published       : D = 0.0155 toward (238.2, +28.8), 27.8 deg from CMB)")

beta, cov, names, od = cw.poisson_fit(N, m, T={"abs_elat": np.abs(cw.PIX_ELAT)})
dv = np.array([beta["dx"], beta["dy"], beta["dz"]])
print("Poisson + |elat|   :", cw.describe(dv, cov[1:4, 1:4]), f" overdispersion {od:.2f}")
