#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Shells in the redshift corrected for a trial velocity along the CMB (appendix B). Scans the velocity on
mocks with 370 km/s and on the data.
"""
import healpy as hp
import numpy as np

from mocks import NSIDE_ARF, cells, theory_fields
from quaia import C_KMS, CMB, CMB_V, load, ols
from quaia_step2_jackknife import mock_y
from results import footprint

rng = np.random.default_rng(52)
BETAS = np.arange(-3000, 3001, 100.0)
WIDTHS = (0.05, 0.01, 0.002)
N_MOCK = 12


def chi2_curve(y, c, W):
    out = []
    for b in BETAS:
        yd = y + b / C_KMS * c
        s = np.floor(yd / W).astype(np.int64)
        s -= s.min()
        cnt = np.maximum(np.bincount(s), 1)
        mu = np.bincount(s, weights=yd) / cnt
        out.append(np.sum((yd - mu[s]) ** 2))
    out = np.array(out)
    return out - out.min()


def run():
    d = load()
    keep = footprint(d)
    c = d["n"][keep] @ CMB
    T = d["T"][keep]
    v_kin = CMB_V / C_KMS * CMB

    def clean(y):
        coef, _ = ols(T, y)
        return y - T @ coef

    cell = cells(d["T_raw"])
    order_cell = np.argsort(cell, kind="stable")
    pix = hp.ang2pix(NSIDE_ARF, d["ra"], d["dec"], lonlat=True)
    mask = hp.ud_grade(hp.read_map("data/selection_function_NSIDE64_G20.5.fits"), NSIDE_ARF) > 0.5
    zmed = np.median(d["z"][mask[pix]])
    fields = theory_fields(3 * NSIDE_ARF - 1)

    print("Step 5c: shells in de-Dopplered ln(1+z), trial velocity along the CMB (km/s); |b|>30 S>0.5, templates removed")
    yd = clean(d["y"][keep])
    for W in WIDTHS:
        mins = []
        for i in range(N_MOCK):
            ym = clean(mock_y(d, cell, order_cell, pix, fields, zmed, v_kin)[keep])
            cm = chi2_curve(ym, c, W)
            mins.append(BETAS[np.argmin(cm)])
        cd = chi2_curve(yd, c, W)
        print(f"  W = {W:.3f}: mocks (input 370) minimum at {np.mean(mins):+6.0f} +- {np.std(mins):5.0f} "
              f"(range {min(mins):+.0f} to {max(mins):+.0f}); data minimum at {BETAS[np.argmin(cd)]:+6.0f}; "
              f"data chi2 rise from the minimum to +-1000 km/s: {np.interp(BETAS[np.argmin(cd)] + 1000, BETAS, cd):.3g}, "
              f"{np.interp(BETAS[np.argmin(cd)] - 1000, BETAS, cd):.3g} (in units of sum of squares; per-source variance "
              f"{np.var(yd):.3g})", flush=True)


if __name__ == "__main__":
    run()
