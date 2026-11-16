#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Checks of the joint test (section 6): joint mocks with our motion only, all 64 subsets of the templates,
and the Ellis-Baldwin factor that the counts would need.
"""
import itertools
import sys

import healpy as hp
import numpy as np

from mocks import NSIDE_ARF, cells, theory_fields
from quaia import C_KMS, CMB, CMB_V, dipole, load, pixel_galactic, template_maps
from quaia_step1_kinematic import alpha_bp_rp
from quaia_step2_jackknife import loo, mock_y
from quaia_step3_joint import R_RESP, design, poisson_fit
from results import footprint
from sky import NSIDE, NPIX, TEMPLATES, unit

rng = np.random.default_rng(31)
NS_JK = 4
LMAX = 3 * NSIDE - 1


def setup():
    d = load()
    from astropy.table import Table
    tab = Table.read("data/quaia_G20.5.fits")
    alpha_med = np.nanmedian(alpha_bp_rp(np.asarray(tab["phot_bp_mean_mag"], float),
                                         np.asarray(tab["phot_rp_mean_mag"], float)))
    pl, pb = pixel_galactic()
    S = hp.read_map("data/selection_function_NSIDE64_G20.5.fits")
    mpix = (np.abs(pb) > 30) & (S > 0.5)
    keep = footprint(d)
    G = d["G"][keep]
    x = np.sum(G > 20.45) / 0.05 / len(G) * 2.5 / np.log(10)
    th, ph = hp.pix2ang(NSIDE, np.arange(NPIX))
    return dict(d=d, A=2 + x * (1 + alpha_med), x=x, alpha=alpha_med, pvec=unit(pl, pb), S=S, mpix=mpix,
                keep=keep, Tpix=template_maps(),
                N=np.bincount(hp.ang2pix(NSIDE, d["ra"], d["dec"], lonlat=True), minlength=NPIX).astype(float),
                reg_s=hp.ang2pix(NS_JK, d["ra"], d["dec"], lonlat=True), reg_p=hp.ang2pix(NS_JK, th, ph))


def counts_fit(s, N, cols):
    m = s["mpix"]
    X = design(s["pvec"][m], s["Tpix"][m][:, cols], len(cols) > 0)
    return poisson_fit(N[m], X, np.zeros(m.sum()))[0]


def counts_loo(s, N, ids, cols):
    out = []
    for r in ids:
        k = s["mpix"] & (s["reg_p"] != r)
        X = design(s["pvec"][k], s["Tpix"][k][:, cols], len(cols) > 0)
        out.append(poisson_fit(N[k], X, np.zeros(k.sum()))[0][1:4])
    return np.array(out)


def joint(s, y, N, bias_z, r):
    """Velocities from the redshifts and the counts, their jackknife covariance and chi2 (NSIDE 4)."""
    d, keep = s["d"], s["keep"]
    allc = list(range(len(TEMPLATES)))
    vz = (dipole(d["n"][keep], y[keep], d["T"][keep])[0] - bias_z) / r
    vN = counts_fit(s, N, allc)[1:4] / s["A"]
    vz_l, ids = loo(d, y, keep, None, "templates", s["reg_s"])
    vN_l = counts_loo(s, N, ids, allc) / s["A"]
    J = np.column_stack([vz_l / r, vN_l])
    m = len(J)
    C = (m - 1) / m * ((J - J.mean(0)).T @ (J - J.mean(0)))
    Cd = C[3:, 3:] + C[:3, :3] - C[:3, 3:] - C[3:, :3]
    diff = vN - vz
    return vz, vN, diff @ np.linalg.solve(Cd, diff), Cd


def measured_cl(s, beta):
    m = s["mpix"]
    X = design(s["pvec"][m], s["Tpix"][m], True)
    mu = np.exp(X @ beta)
    delta = np.zeros(NPIX)
    delta[m] = s["N"][m] / mu - 1
    nbar = s["N"][m].mean()
    cl = hp.anafast(delta, lmax=LMAX) / m.mean() - 4 * np.pi / NPIX / nbar
    cl = np.clip(cl, 0, None)
    cl[0] = 0.0
    cl[1] = cl[2]
    return cl, mu, X


def run(n_mock):
    s = setup()
    d = s["d"]
    v_kin = CMB_V / C_KMS * CMB
    old = np.load("out/redshift_mocks.npz")
    bias_z = old["dip_cells"][:, list(old["dip_names"]).index("|b|>30 S>0.5: templates")].mean(0) - v_kin
    allc = list(range(len(TEMPLATES)))
    beta = counts_fit(s, s["N"], allc)

    print(f"Step 3b, |b|>30 S>0.5, counts and redshifts with all six templates; 2 + x(1+alpha) = {s['A']:.2f}")

    # Ellis-Baldwin factor needed for 370 km/s
    for lab, cols, off in (("templates", allc, False), ("S offset", [], True), ("S offset + templates", allc, True)):
        m = s["mpix"]
        X = design(s["pvec"][m], s["Tpix"][m][:, cols], len(cols) > 0)
        b = poisson_fit(s["N"][m], X, np.log(s["S"][m]) if off else np.zeros(m.sum()))[0]
        print(f"  [3] counts, {lab:<21}: D along CMB {b[1:4] @ CMB:.4f}; pure motion at 370 km/s needs "
              f"2 + x(1+alpha) = {b[1:4] @ CMB / (CMB_V / C_KMS):.1f} (measured {s['A']:.2f}; "
              f"{2 + 0.74 * (1 + s['alpha']):.2f} with the differential slope)")

    # all 64 subsets of the templates
    vz_full, vN_full, chi_full, Cd = joint(s, d["y"], s["N"], bias_z, R_RESP)
    print(f"  data: common-velocity chi2 (3 dof) = {chi_full:.1f}; counts - redshift along CMB "
          f"{(vN_full - vz_full) @ CMB * C_KMS:+.0f} +- {np.sqrt(CMB @ Cd @ CMB) * C_KMS:.0f} km/s")
    rows = []
    keep = s["keep"]
    for k in range(len(TEMPLATES) + 1):
        for cols in itertools.combinations(allc, k):
            cols = list(cols)
            T = d["T"][keep][:, cols] if cols else None
            vz = dipole(d["n"][keep], d["y"][keep], T)[0] / R_RESP
            vN = counts_fit(s, s["N"], cols)[1:4] / s["A"]
            diff = vN - vz
            rows.append((cols, diff @ CMB * C_KMS, diff @ np.linalg.solve(Cd, diff), vN @ CMB * C_KMS, vz @ CMB * C_KMS))
    chis = np.array([r[2] for r in rows])
    dc = np.array([r[1] for r in rows])
    print(f"  [2] 64 template subsets (same subset for counts and redshifts; covariance of the full model; "
          f"no mock bias removed): chi2 {chis.min():.1f} to {chis.max():.1f}, median {np.median(chis):.1f}; "
          f"fraction with p < 0.01 (chi2 > 11.3): {np.mean(chis > 11.34):.2f}; "
          f"counts - redshift along CMB {dc.min():+.0f} to {dc.max():+.0f} km/s")
    order = np.argsort(chis)
    for i in list(order[:3]) + list(order[-2:]):
        cols, c, ch, vn, vzz = rows[i]
        print(f"      {'+'.join(TEMPLATES[j] for j in cols) or 'none':<40} chi2 {ch:6.1f}; along CMB counts {vn:+5.0f}, "
              f"redshift {vzz:+5.0f}, difference {c:+5.0f}")
    sys.stdout.flush()

    # joint mocks with our motion only
    cl, mu_fit, X = measured_cl(s, beta)
    b_null = beta.copy()
    b_null[1:4] = s["A"] * v_kin
    mu_null = np.exp(X @ b_null)
    cell = cells(d["T_raw"])
    order_cell = np.argsort(cell, kind="stable")
    pixA = hp.ang2pix(NSIDE_ARF, d["ra"], d["dec"], lonlat=True)
    maskA = hp.ud_grade(s["S"], NSIDE_ARF) > 0.5
    zmed = np.median(d["z"][maskA[pixA]])
    fields = theory_fields(3 * NSIDE_ARF - 1)
    m = s["mpix"]
    res = []
    for i in range(n_mock):
        g = hp.synfast(cl, NSIDE, lmax=LMAX)
        ln = np.exp(g - g.var() / 2)
        Nm = np.zeros(NPIX)
        Nm[m] = rng.poisson(mu_null * ln[m])
        ym = mock_y(d, cell, order_cell, pixA, fields, zmed, v_kin)
        vz, vN, chi, _ = joint(s, ym, Nm, bias_z, 1.0)
        res.append(np.concatenate([vz, vN, [chi]]))
        if (i + 1) % 25 == 0:
            r_ = np.array(res)
            print(f"    ... {i + 1} mocks; chi2 >= data in {np.mean(r_[:, 6] >= chi_full):.3f}", flush=True)
            np.save("out/quaia_step3b_mocks.npy", r_)
    r_ = np.array(res)
    np.save("out/quaia_step3b_mocks.npy", r_)
    vz_m, vN_m, chi_m = r_[:, :3], r_[:, 3:6], r_[:, 6]
    diff_m = vN_m - vz_m
    Cm = np.cov(diff_m.T)
    dd = vN_full - vz_full
    chi_mockcov = dd @ np.linalg.solve(Cm, dd)
    chi_mocks_mockcov = np.einsum("ij,ij->i", diff_m - diff_m.mean(0), np.linalg.solve(Cm, (diff_m - diff_m.mean(0)).T).T)
    print(f"  [1] {n_mock} joint null mocks (370 km/s only):")
    print(f"      count velocity along CMB: mean {vN_m.mean(0) @ CMB * C_KMS:+.0f}, sd {np.std(vN_m @ CMB) * C_KMS:.0f}; "
          f"redshift: mean {vz_m.mean(0) @ CMB * C_KMS:+.0f}, sd {np.std(vz_m @ CMB) * C_KMS:.0f} (input 370)")
    print(f"      jackknife chi2 on mocks: median {np.median(chi_m):.1f} (chi2 with 3 dof: 2.4), 99th percentile "
          f"{np.percentile(chi_m, 99):.1f}; data {chi_full:.1f} -> p = {np.mean(chi_m >= chi_full):.4f} "
          f"(upper limit {1 / n_mock:.4f} if zero)")
    print(f"      with the mock covariance of (counts - redshift): data chi2 {chi_mockcov:.1f} (3 dof); "
          f"mocks >= data: {np.mean(chi_mocks_mockcov >= chi_mockcov):.4f}; along CMB "
          f"{dd @ CMB * C_KMS:+.0f} +- {np.sqrt(CMB @ Cm @ CMB) * C_KMS:.0f} km/s")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
