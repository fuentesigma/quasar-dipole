#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Half skies and jackknife for the CatWISE dipole, against mocks with our motion, the templates, LCDM
clustering and shot noise only.
"""
import numpy as np
import healpy as hp
import catwise as cw

rng = np.random.default_rng(11)
NMOCK, NMOCK_JK = 300, 40
LMAX = 3 * cw.NSIDE - 1

cat = cw.load_catalogue()
m = np.load(cw.OUT_CW / "mask.npy")
N = cw.count_map(cat)
Dk = cw.kinematic_amplitude(cat)[0]
T0 = cw.templates()
T0["abs_elat"] = np.abs(cw.PIX_ELAT)
SETS = {"Quaia templates": {k: t for k, t in T0.items() if t[m].std() > 0}}
SETS["+ WISE depth"] = dict(SETS["Quaia templates"], **dict(np.load(cw.OUT_CW / "depth_maps.npz")))

HALVES = {"Galactic N/S": (cw.PIX_B > 0, cw.PIX_B < 0),
          "ecliptic N/S": (cw.PIX_ELAT > 0, cw.PIX_ELAT < 0)}
th, ph = hp.pix2ang(cw.NSIDE, np.arange(cw.NPIX))
REGIONS = {ns: hp.ang2pix(ns, th, ph) for ns in (2, 4)}

# LCDM clustering spectrum, extended to ell = 1 (C_1 = C_2) and beyond ell = 60 with a power law
th_cl = np.load(cw.OUT_CW / "cl_theory_desi_b1.3.npy")
cl = np.zeros(LMAX + 1)
cl[2:61] = th_cl[2:61]
cl[1] = cl[2]
ell = np.arange(LMAX + 1)
cl[61:] = cl[60] * (ell[61:] / 60.0) ** -1.0


def fit(Nm, sel, T):
    b, c, _, _ = cw.poisson_fit(Nm, sel, T=T)
    return np.array([b["dx"], b["dy"], b["dz"]]), c[1:4, 1:4]


def halves(Nm, T):
    return {h: [fit(Nm, m & s, T)[0] for s in pair] for h, pair in HALVES.items()}


def jackknife(Nm, T, ns):
    reg = REGIONS[ns]
    ids = [r for r in np.unique(reg[m]) if (m & (reg == r)).sum() >= 10]
    d = np.array([fit(Nm, m & (reg != r), T)[0] for r in ids])
    n = len(d)
    return (n - 1) / n * ((d - d.mean(0)).T @ (d - d.mean(0)))


def lognormal():
    g = hp.synfast(cl, cw.NSIDE, lmax=LMAX, pixwin=True)
    return np.exp(g - g.var() / 2) - 1


for name, T in SETS.items():
    beta, _, _, _ = cw.poisson_fit(N, m, T=T)
    Z = {k: (T[k] - T[k][m].mean()) / T[k][m].std() for k in T}
    syst = sum(beta[k] * Z[k] for k in T)
    nbar = np.exp(beta["b0"])

    d_data, cov_poisson = fit(N, m, T)
    h_data = halves(N, T)
    jk_data = {ns: jackknife(N, T, ns) for ns in REGIONS}

    D_m, h_m, jk_m = [], {h: [] for h in HALVES}, {ns: [] for ns in REGIONS}
    for i in range(NMOCK):
        lam = nbar * np.exp(cw.PIX_VEC @ (Dk * cw.CMB) + syst) * (1 + lognormal())
        Nm = rng.poisson(np.clip(lam, 0, None)).astype(float)
        D_m.append(fit(Nm, m, T)[0])
        for h, (a, b) in halves(Nm, T).items():
            h_m[h].append(a - b)
        if i < NMOCK_JK:
            for ns in REGIONS:
                jk_m[ns].append(jackknife(Nm, T, ns))
    D_m = np.array(D_m)
    cov_null = np.cov(D_m.T)

    print(f"\n=== {name}")
    D = np.linalg.norm(d_data)
    print(f"data: {cw.describe(d_data)}   D/D_kin = {D / Dk:.2f}")
    print(f"null mocks: |D| = {np.linalg.norm(D_m, axis=1).mean():.4f} +- {np.linalg.norm(D_m, axis=1).std():.4f}, "
          f"p(|D| >= data) = {np.mean(np.linalg.norm(D_m, axis=1) >= D):.3f}")
    print(f"per-component scatter: Poisson x overdispersion {np.sqrt(np.diag(cov_poisson)).mean():.4f}, "
          f"null mocks {np.sqrt(np.diag(cov_null)).mean():.4f}")

    print("half skies (difference between the two halves):")
    for h, (a, b) in h_data.items():
        diff = a - b
        dm = np.array(h_m[h])
        Cd = np.cov(dm.T)
        chi2 = diff @ np.linalg.solve(Cd, diff)
        chi2_m = np.einsum("ij,ij->i", dm, np.linalg.solve(Cd, dm.T).T)
        print(f"  {h:13s}: halves D = {np.linalg.norm(a) / Dk:.2f} and {np.linalg.norm(b) / Dk:.2f} x D_kin, "
              f"{np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))):3.0f} deg apart; "
              f"|diff| = {np.linalg.norm(diff):.4f} vs mocks {np.linalg.norm(dm, axis=1).mean():.4f}; "
              f"chi2 = {chi2:.1f} (3 dof), p = {np.mean(chi2_m >= chi2):.3f}")

    excess = d_data - Dk * cw.CMB
    for ns in REGIONS:
        Cjk = jk_data[ns]
        cal = np.mean([np.sqrt(np.diag(c)).mean() for c in jk_m[ns]]) / np.sqrt(np.diag(cov_null)).mean()
        Cjk_cal = Cjk / cal ** 2
        chi2_jk = excess @ np.linalg.solve(Cjk_cal, excess)
        print(f"jackknife NSIDE {ns}: data scatter per component {np.sqrt(np.diag(Cjk)).mean():.4f}; "
              f"on null mocks the jackknife gives {cal:.2f} x the true scatter -> calibrated "
              f"{np.sqrt(np.diag(Cjk_cal)).mean():.4f} = {np.sqrt(np.diag(Cjk_cal)).mean() / np.sqrt(np.diag(cov_null)).mean():.2f} x null")
        print(f"   excess over kinematic: chi2 = {chi2_jk:.1f} (3 dof) with the calibrated jackknife covariance")
    chi2_null = excess @ np.linalg.solve(cov_null, excess)
    print(f"excess over kinematic with the null covariance: chi2 = {chi2_null:.1f} (3 dof)")
