#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Effect of the flux limit G < 20.5 on the dipole of the redshifts (section 3.1).

Adds a known motion to the real catalogue, which shifts ln(1+z) and G, and compares the dipoles with and
without it. Also gives the spectral index alpha used for the counts.
"""
import sys

import numpy as np

from mocks import cells, dipole_cells
from quaia import C_KMS, CMB, dipole, load
from results import footprint

G_LIM = 20.45
VELOCITIES = (370.0, 800.0)
# Gaia DR3 AB minus Vega offsets (Riello et al. 2021) and pivot wavelengths in nm
AB_BP, AB_RP, LAM_BP, LAM_RP = 0.0155, 0.3561, 511.0, 777.0


def alpha_bp_rp(d_bp, d_rp):
    """Spectral index alpha (F_nu ~ nu^-alpha) from the BP and RP magnitudes."""
    col_ab = (d_bp + AB_BP) - (d_rp + AB_RP)
    return col_ab / (2.5 * np.log10(LAM_RP / LAM_BP))


def estimators(d, y, keep, cell):
    n, T = d["n"][keep], d["T"][keep]
    yk = y[keep]
    return {"raw": dipole(n, yk)[0], "templates": dipole(n, yk, T)[0], "cells": dipole_cells(n, yk, cell[keep])}


def run():
    d = load()
    from astropy.table import Table
    t = Table.read("data/quaia_G20.5.fits")
    alpha = alpha_bp_rp(np.asarray(t["phot_bp_mean_mag"], float), np.asarray(t["phot_rp_mean_mag"], float))
    ok = np.isfinite(alpha)
    alpha_med = np.median(alpha[ok])
    alpha = np.where(ok, alpha, alpha_med)
    cell = cells(d["T_raw"])
    base = footprint(d)
    cos = d["n"] @ CMB

    print(f"Step 1: kinematic response of the redshift dipole, |b|>30 S>0.5, cut G < {G_LIM}")
    print(f"  alpha (BP-RP): median {alpha_med:.2f}, 16-84% {np.percentile(alpha[ok], 16):.2f} to "
          f"{np.percentile(alpha[ok], 84):.2f}")
    G = d["G"]
    edge = base & (G > G_LIM - 0.05) & (G < G_LIM)
    allk = base & (G < G_LIM)
    print(f"  quasars: {allk.sum()}; mean ln(1+z) {d['y'][allk].mean():.4f} overall, {d['y'][edge].mean():.4f} "
          f"in the last 0.05 mag before the cut")
    dlnN_dG = np.log(np.sum(base & (G > G_LIM - 0.1) & (G < G_LIM)) / np.sum(base & (G > G_LIM - 0.2) & (G < G_LIM - 0.1))) / 0.1
    print(f"  count slope at the cut: dlnN/dG = {dlnN_dG:.2f} per mag, x = {dlnN_dG / (0.4 * np.log(10)):.2f}")

    ref = estimators(d, d["y"], allk, cell)
    for vk in VELOCITIES:
        beta = vk / C_KMS
        for label, a in (("redshift shift only", None), ("shift + flux limit, alpha per source", alpha),
                         (f"shift + flux limit, alpha = {alpha_med:.2f} for all", np.full_like(alpha, alpha_med))):
            y = d["y"] - beta * cos
            if a is None:
                keep = allk
            else:
                g_obs = G - 2.5 / np.log(10) * (1 + a) * beta * cos
                keep = base & (g_obs < G_LIM)
            got = estimators(d, y, keep, cell)
            parts = []
            for k in ("raw", "templates", "cells"):
                dv = (got[k] - ref[k]) * C_KMS
                parts.append(f"{k} {dv @ CMB:6.1f} (off-axis {np.linalg.norm(dv - (dv @ CMB) * CMB):4.1f})")
            print(f"  v = {vk:3.0f} km/s, {label:<40} recovered along CMB, km/s: " + "; ".join(parts))
    print("  Recovered = (dipole with motion) - (dipole without), same sky; pure Doppler shift should return v.")


if __name__ == "__main__":
    run()
