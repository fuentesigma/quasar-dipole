#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Joint test of the dipole of the counts and the dipole of the redshifts (section 6).

Poisson fit of the counts with the templates, the selection function or both, Ellis-Baldwin conversion to a
velocity, and the jackknife covariance of the two dipoles together.
"""
import healpy as hp
import numpy as np

from quaia import C_KMS, CMB, CMB_V, describe, dipole, load, pixel_galactic, template_maps
from quaia_step1_kinematic import alpha_bp_rp
from quaia_step2_jackknife import loo
from results import BCUT, SMIN, footprint
from sky import NSIDE, NPIX, TEMPLATES, lb, unit

R_RESP = 0.909  # quaia_step2b_response.py
V_ALT = 800.0


def poisson_fit(N, X, offset, iters=60):
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(N.mean() / np.exp(offset).mean())
    for _ in range(iters):
        mu = np.exp(X @ beta + offset)
        step = np.linalg.solve((X.T * mu) @ X, X.T @ (N - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(X @ beta + offset)
    phi = np.sum((N - mu) ** 2 / mu) / (len(N) - X.shape[1])
    return beta, np.linalg.inv((X.T * mu) @ X) * max(phi, 1.0), phi


def design(nvec, T, use_t):
    cols = [np.ones(len(nvec)), *nvec.T]
    if use_t:
        cols += [(t - t.mean()) / t.std() for t in T.T]
    return np.column_stack(cols)


def run():
    d = load()
    from astropy.table import Table
    tab = Table.read("data/quaia_G20.5.fits")
    alpha = alpha_bp_rp(np.asarray(tab["phot_bp_mean_mag"], float), np.asarray(tab["phot_rp_mean_mag"], float))
    alpha_med = np.nanmedian(alpha)

    pl, pb = pixel_galactic()
    pvec = unit(pl, pb)
    S = hp.read_map("data/selection_function_NSIDE64_G20.5.fits")
    Tpix = template_maps()
    spix = hp.ang2pix(NSIDE, d["ra"], d["dec"], lonlat=True)
    N = np.bincount(spix, minlength=NPIX).astype(float)
    th, ph = hp.pix2ang(NSIDE, np.arange(NPIX))

    v_kin = CMB_V / C_KMS * CMB
    old = np.load("out/redshift_mocks.npz")
    names = list(old["dip_names"])

    rows = []
    print("Step 3: joint count + redshift dipole, Quaia G < 20.5")
    for bcut in (BCUT, 40.0):
        sname = f"|b|>{bcut:.0f} S>{SMIN}"
        mpix = (np.abs(pb) > bcut) & (S > SMIN)
        keep = footprint(d, bcut=bcut)
        G = d["G"][keep]
        n_edge = np.sum(G > 20.45) / 0.05
        x = n_edge / len(G) * 2.5 / np.log(10)
        A = 2 + x * (1 + alpha_med)
        print(f"\n=== {sname}: {mpix.sum()} pixels, {keep.sum()} quasars; x = {x:.2f}, alpha = {alpha_med:.2f}, "
              f"2 + x(1+alpha) = {A:.2f}; expected count dipole for 370 km/s D = {A * CMB_V / C_KMS:.4f}")

        bias_z = old["dip_cells"][:, names.index(f"{sname}: templates")].mean(0) - v_kin
        vz = (dipole(d["n"][keep], d["y"][keep], d["T"][keep])[0] - bias_z) / R_RESP
        print(f"  redshift dipole (templates, mock bias removed, / r = {R_RESP}): {describe(vz)}; "
              f"along CMB {vz @ CMB * C_KMS:+.0f}")

        for model, use_t, use_s in (("templates", True, False), ("S offset", False, True),
                                    ("S offset + templates", True, True)):
            off = np.log(S[mpix]) if use_s else np.zeros(mpix.sum())
            X = design(pvec[mpix], Tpix[mpix], use_t)
            beta, cov, phi = poisson_fit(N[mpix], X, off)
            Dv, Dcov = beta[1:4], cov[1:4, 1:4]
            vN = Dv / A
            print(f"\n  counts, {model}: D = {np.linalg.norm(Dv):.4f} = {np.linalg.norm(Dv) / (A * CMB_V / C_KMS):.2f} x expected,"
                  f" {np.degrees(np.arccos(Dv @ CMB / np.linalg.norm(Dv))):.0f} deg from CMB; overdispersion {phi:.2f}")
            print(f"    as a velocity: {describe(vN, Dcov / A ** 2)}; along CMB {vN @ CMB * C_KMS:+.0f}")

            for ns in (2, 4):
                reg_s = hp.ang2pix(ns, d["ra"], d["dec"], lonlat=True)
                reg_p = hp.ang2pix(ns, th, ph)
                vz_loo, ids = loo(d, d["y"], keep, None, "templates", reg_s)
                vz_loo = vz_loo / R_RESP
                vN_loo = []
                for r in ids:
                    k = mpix & (reg_p != r)
                    o = np.log(S[k]) if use_s else np.zeros(k.sum())
                    vN_loo.append(poisson_fit(N[k], design(pvec[k], Tpix[k], use_t), o)[0][1:4] / A)
                J = np.column_stack([vz_loo, np.array(vN_loo)])
                m = len(J)
                C = (m - 1) / m * ((J - J.mean(0)).T @ (J - J.mean(0)))
                Cz, CN = C[:3, :3], C[3:, 3:]
                rho = (CMB @ C[:3, 3:] @ CMB) / np.sqrt((CMB @ Cz @ CMB) * (CMB @ CN @ CMB))
                vec = np.concatenate([vz, vN])
                Ci = np.linalg.inv(C)
                H = np.vstack([np.eye(3), np.eye(3)])
                Cv = np.linalg.inv(H.T @ Ci @ H)
                vbest = Cv @ H.T @ Ci @ vec
                chi2 = lambda v: (vec - H @ v) @ Ci @ (vec - H @ v)
                diff = vN - vz
                Cd = CN + Cz - C[:3, 3:] - C[3:, :3]
                print(f"    jackknife NSIDE {ns} ({m} regions): along CMB, redshift +-{np.sqrt(CMB @ Cz @ CMB) * C_KMS:.0f}, "
                      f"counts +-{np.sqrt(CMB @ CN @ CMB) * C_KMS:.0f}, correlation {rho:+.2f}")
                print(f"      counts - redshift: {np.linalg.norm(diff) * C_KMS:.0f} km/s, along CMB "
                      f"{diff @ CMB * C_KMS:+.0f} +- {np.sqrt(CMB @ Cd @ CMB) * C_KMS:.0f}; "
                      f"chi2 for one common velocity {diff @ np.linalg.solve(Cd, diff):.1f} (3 dof)")
                rows.append(dict(bcut=bcut, model=model, ns=ns, vN=vN @ CMB * C_KMS, vz=vz @ CMB * C_KMS,
                                 eN=np.sqrt(CMB @ CN @ CMB) * C_KMS, ez=np.sqrt(CMB @ Cz @ CMB) * C_KMS,
                                 diff=diff @ CMB * C_KMS, ed=np.sqrt(CMB @ Cd @ CMB) * C_KMS,
                                 chi2=diff @ np.linalg.solve(Cd, diff), D=np.linalg.norm(Dv), A=A))
                print(f"      common velocity: {describe(vbest, Cv)}; chi2 (6 dof) at best / 370 / {V_ALT:.0f} along CMB: "
                      f"{chi2(vbest):.1f} / {chi2(v_kin):.1f} / {chi2(V_ALT / C_KMS * CMB):.1f}")

    np.save("out/quaia_step3_joint.npy", rows, allow_pickle=True)


if __name__ == "__main__":
    run()
