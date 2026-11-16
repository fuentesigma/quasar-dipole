#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Ellis-Baldwin prediction for CatWISE: slope x of N(>S) near W1 = 16.4, and the Doppler shift of each
source with its own spectral index, over the whole sky and inside the mask.
"""
import numpy as np
import healpy as hp
import catwise as cw

BETA = cw.V_CMB / cw.C_KMS
CUT = 16.4
c = cw.load_catalogue(w1_max=16.5)
m = np.load(cw.OUT_CW / "mask.npy")
inmask = m[hp.ang2pix(cw.NSIDE, c["ra"], c["dec"], lonlat=True)]
alpha = np.abs(c["alpha_W1"])  # S ~ nu^-alpha in the Ellis-Baldwin convention

print("1. slope x of log N(>S) at the cut")
for lab, s in (("all sky", np.ones(len(alpha), bool)), ("in mask", inmask)):
    w1 = c["w1"][s]
    for dm in (0.02, 0.05, 0.1, 0.2, 0.3):
        x = np.log10((w1 < CUT).sum() / (w1 < CUT - dm).sum()) / (0.4 * dm)
        print(f"   {lab}, window {dm:.2f} mag: x = {x:.2f}")
    near = (w1 > CUT - 0.1) & (w1 < CUT)
    print(f"   {lab}: mean alpha all = {alpha[s & (c['w1'] < CUT)].mean():.3f}, "
          f"near the cut = {alpha[s][near].mean():.3f}")


def boosted(s):
    w1, a = c["w1"][s], alpha[s]
    n = lambda e: np.sum(w1 - 2.5 * (1 + a) * np.log10(1 + e) < CUT)
    return 2 * BETA + (n(BETA) - n(-BETA)) / (2 * n(0.0))


print("2-3. per-source boosting")
for lab, s in (("all sky", np.ones(len(alpha), bool)), ("in mask", inmask)):
    D = boosted(s)
    print(f"   {lab}: D_kin = {D:.5f}  (equivalent x(1+alpha) = {D / BETA - 2:.3f})")
x, a = cw.kinematic_amplitude(cw.load_catalogue())[1:]
print(f"used so far: x = {x:.2f}, alpha = {a:.2f}, D_kin = {(2 + x * (1 + a)) * BETA:.5f}")
