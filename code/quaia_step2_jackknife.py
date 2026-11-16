#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Jackknife error of the corrected dipole of the redshifts, calibrated on the mocks, and the comparison
of 370 and 800 km/s (sections 3.3 and 5).
"""
import healpy as hp
import numpy as np

from mocks import NSIDE_ARF, cells, dipole_cells, theory_fields
from quaia import C_KMS, CMB, CMB_V, describe, dipole, load
from results import footprint

rng = np.random.default_rng(12)
NMOCK_JK = 40
SAMPLES = {"|b|>30 S>0.5": dict(), "|b|>40 S>0.5": dict(bcut=40.0)}
V_ALT = 800.0


def fit(d, y, keep, cell, how):
    if how == "templates":
        return dipole(d["n"][keep], y[keep], d["T"][keep])[0]
    return dipole_cells(d["n"][keep], y[keep], cell[keep])


def _sums(X, y, key, K):
    """Sums of X^T X and X^T y for each group."""
    p = X.shape[1]
    XtX = np.empty((K, p, p))
    for i in range(p):
        for j in range(i, p):
            XtX[:, i, j] = XtX[:, j, i] = np.bincount(key, weights=X[:, i] * X[:, j], minlength=K)
    Xty = np.column_stack([np.bincount(key, weights=X[:, i] * y, minlength=K) for i in range(p)])
    return XtX, Xty


def loo(d, y, keep, cell, how, reg):
    """Dipoles with one region left out, from the sums above (same fits as fit(), but faster)."""
    counts = np.bincount(reg[keep], minlength=reg.max() + 1)
    ids = np.flatnonzero(counts >= 2000)
    R = reg.max() + 1
    n, yk, rk = d["n"][keep], y[keep], reg[keep]
    if how == "templates":
        X = np.column_stack([np.ones(len(yk)), n, d["T"][keep]])
        XtX, Xty = _sums(X, yk, rk, R)
        A, b = XtX.sum(0), Xty.sum(0)
        v = np.array([-np.linalg.solve(A - XtX[r], b - Xty[r])[1:4] for r in ids])
    else:
        ck = cell[keep]
        C = ck.max() + 1
        X = np.column_stack([np.ones(len(yk)), n, yk])  # count, sum n, sum y, and all products
        S, _ = _sums(X, yk, ck * R + rk, C * R)
        S = S.reshape(C, R, 5, 5)
        tot = S.sum(1)

        def within(Sc):
            N = np.maximum(Sc[:, 0, 0], 1e-300)[:, None, None]
            M = Sc[:, 1:, 1:] - Sc[:, 1:, :1] * Sc[:, :1, 1:] / N
            M = M[Sc[:, 0, 0] > 0].sum(0)
            return -np.linalg.solve(M[:3, :3], M[:3, 3])

        v = np.array([within(tot - S[:, r]) for r in ids])
    return v, ids


def jackknife(d, y, keep, cell, how, reg):
    v, _ = loo(d, y, keep, cell, how, reg)
    m = len(v)
    return (m - 1) / m * ((v - v.mean(0)).T @ (v - v.mean(0))), m


def mock_y(d, cell, order_cell, pix, fields, zmed, v_kin):
    N = len(d["z"])
    z0 = np.empty(N)
    z0[order_cell] = d["z"][np.lexsort((rng.random(N), cell))]
    alms = hp.synalm(fields, lmax=len(fields[0]) - 1, new=True)
    f = [hp.alm2map(a, NSIDE_ARF) for a in alms]
    dz = np.where(z0 >= zmed, f[1][pix], f[0][pix])
    return np.log1p(z0 + dz) - d["n"] @ v_kin


def run():
    d = load()
    cell = cells(d["T_raw"])
    order_cell = np.argsort(cell, kind="stable")
    pix = hp.ang2pix(NSIDE_ARF, d["ra"], d["dec"], lonlat=True)
    mask = hp.ud_grade(hp.read_map("data/selection_function_NSIDE64_G20.5.fits"), NSIDE_ARF) > 0.5
    zmed = np.median(d["z"][mask[pix]])
    fields = theory_fields(3 * NSIDE_ARF - 1)
    v_kin = CMB_V / C_KMS * CMB
    regions = {ns: hp.ang2pix(ns, d["ra"], d["dec"], lonlat=True) for ns in (2, 4)}

    mocks = [mock_y(d, cell, order_cell, pix, fields, zmed, v_kin) for _ in range(NMOCK_JK)]
    old = np.load("out/redshift_mocks.npz")
    names = list(old["dip_names"])

    print("Step 2: jackknife error on the corrected redshift dipole (km/s); mock covariance from out/redshift_mocks.npz")
    for sname, kw in SAMPLES.items():
        keep = footprint(d, **kw)
        for how in ("templates", "cells"):
            mk = old["dip_cells"][:, names.index(f"{sname}: {how}")]
            bias = mk.mean(0) - v_kin
            C_mock = np.cov(mk.T)
            v = fit(d, d["y"], keep, cell, how) - bias
            print(f"\n  {sname}, {how}: corrected dipole {describe(v)}")
            print(f"    mock error per component {np.sqrt(np.diag(C_mock)).mean() * C_KMS:4.0f}, "
                  f"along CMB {np.sqrt(CMB @ C_mock @ CMB) * C_KMS:4.0f}")
            covs = {"mocks": C_mock}
            for ns, reg in regions.items():
                C_jk, nreg = jackknife(d, d["y"], keep, cell, how, reg)
                jk_m = [jackknife(d, y, keep, cell, how, reg)[0] for y in mocks]
                cal = np.mean([np.sqrt(np.diag(c)).mean() for c in jk_m]) / np.sqrt(np.diag(C_mock)).mean()
                C_cal = C_jk / cal ** 2
                covs[f"jackknife {ns}"] = C_cal
                print(f"    jackknife NSIDE {ns} ({nreg} regions): raw {np.sqrt(np.diag(C_jk)).mean() * C_KMS:4.0f} per component; "
                      f"on mocks it gives {cal:.2f} x the true scatter -> calibrated "
                      f"{np.sqrt(np.diag(C_cal)).mean() * C_KMS:4.0f} = "
                      f"{np.sqrt(np.diag(C_cal)).mean() / np.sqrt(np.diag(C_mock)).mean():.2f} x mocks; "
                      f"along CMB {np.sqrt(CMB @ C_cal @ CMB) * C_KMS:4.0f}")
            for cname, C in covs.items():
                Ci = np.linalg.inv(C)
                a, s = v @ CMB * C_KMS, np.sqrt(CMB @ C @ CMB) * C_KMS
                chi = {vv: (v - vv / C_KMS * CMB) @ Ci @ (v - vv / C_KMS * CMB) for vv in (0.0, CMB_V, V_ALT)}
                print(f"    {cname:<12} along CMB {a:+5.0f} +- {s:3.0f}: {(CMB_V - a) / s:+.1f} sd from 370, "
                      f"{(V_ALT - a) / s:+.1f} sd from {V_ALT:.0f}; 3D chi2 (3 dof) at 0 / 370 / {V_ALT:.0f} km/s: "
                      f"{chi[0.0]:5.1f} / {chi[CMB_V]:5.1f} / {chi[V_ALT]:5.1f}; "
                      f"Delta chi2 (800 - 370) {chi[V_ALT] - chi[CMB_V]:+5.1f}")


if __name__ == "__main__":
    run()
