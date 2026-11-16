#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Residual systematic of the dipole of the redshifts and the requirements for a kinematic test (section 7).

v_sys = -c [D[f] (mu_c - mu_g) + f D[mu_c] + (1 - f) D[mu_g]] / r, fitted on the DESI matches.
"""
import numpy as np

from quaia import C_KMS, CMB, desi_match, dipole, load
from results import footprint

R = 0.909                # quaia_step2b_response.py
SIG_MOCK_CMB = 147.0     # mock error along the CMB, |b|>30, templates (quaia_step2_jackknife.py)
SIG_JK_CMB = (207.0, 188.0)     # calibrated jackknife along the CMB, NSIDE 2 and 4 (quaia_step2_jackknife.py)
SIG_MOCK_COMP, SIG_JK_COMP = 180.0, (209.0, 205.0)   # per component
CL_FLOOR = (101.0, 126.0)       # corrected map at l = 2 and 3 as a velocity per component (results.py)
CLUSTER = 36.3                  # clustering per component (arf_theory.py)
GOALS = {"800 vs 370 km/s at 5 sd": 430.0 / 5, "800 vs 370 km/s at 3 sd": 430.0 / 3,
         "detect 370 km/s at 5 sd": 370.0 / 5, "detect 370 km/s at 3 sd": 370.0 / 3}


def run():
    d = load()
    D = desi_match(d)
    m = np.isfinite(D["z"]) & (D["z"] > 0)
    delta = d["y"][m] - np.log1p(D["z"][m])
    cat = np.abs(delta) > 0.1
    f, mu_c, mu_g = cat.mean(), delta[cat].mean(), delta[~cat].mean()
    print("Step 4: systematic floor of the Quaia redshift dipole")
    print(f"  DESI matches {m.sum()}: catastrophic fraction f = {f:.3f}; mean error, catastrophic mu_c = {mu_c:+.3f}, "
          f"good mu_g = {mu_g:+.4f}; mu_c - mu_g = {mu_c - mu_g:+.3f}")

    # error model fitted on the templates, applied to the whole region
    T = d["T"]
    X = np.column_stack([np.ones(m.sum()), T[m]])
    e = np.linalg.lstsq(X, delta, rcond=None)[0]
    ef = np.linalg.lstsq(X, cat.astype(float), rcond=None)[0]
    eg = np.linalg.lstsq(X[~cat], delta[~cat], rcond=None)[0]
    ec = np.linalg.lstsq(X[cat], delta[cat], rcond=None)[0]
    keep = footprint(d)
    n = d["n"][keep]
    raw = dipole(n, d["y"][keep])[0]
    eps_pred = T[keep] @ e[1:]
    f_pred = T[keep] @ ef[1:]
    g_pred = T[keep] @ eg[1:]
    c_pred = T[keep] @ ec[1:]
    D_eps = -dipole(n, eps_pred)[0]          # dipole() returns minus the coefficient
    D_f, D_g, D_c = -dipole(n, f_pred)[0], -dipole(n, g_pred)[0], -dipole(n, c_pred)[0]
    print(f"  predicted from the DESI error model, |b|>30 S>0.5, before any correction (km/s along CMB / amplitude):")
    v_eps = -C_KMS * D_eps / R
    v_fc = -C_KMS * D_f * (mu_c - mu_g) / R
    v_g = -C_KMS * (1 - f) * D_g / R
    v_c = -C_KMS * f * D_c / R
    print(f"    all errors            {v_eps @ CMB:+6.0f} / {np.linalg.norm(v_eps):5.0f}")
    print(f"    catastrophic rate     {v_fc @ CMB:+6.0f} / {np.linalg.norm(v_fc):5.0f}   (dipole of f: {np.linalg.norm(D_f):.4f})")
    print(f"    direction of failures {v_c @ CMB:+6.0f} / {np.linalg.norm(v_c):5.0f}   (dipole of mu_c: {np.linalg.norm(D_c):.4f})")
    print(f"    good redshifts        {v_g @ CMB:+6.0f} / {np.linalg.norm(v_g):5.0f}")
    print(f"    sum of the three      {(v_fc + v_c + v_g) @ CMB:+6.0f} / {np.linalg.norm(v_fc + v_c + v_g):5.0f}")
    print(f"    raw data dipole       {raw @ CMB * C_KMS:+6.0f} / {np.linalg.norm(raw) * C_KMS:5.0f}   "
          f"(errors explain {np.linalg.norm(v_eps) / np.linalg.norm(raw * C_KMS):.2f} of the amplitude; "
          f"the rest is the real redshift mix changing with the selection)")

    # residual after the correction
    jk = [np.sqrt(max(s ** 2 - SIG_MOCK_CMB ** 2, 0)) for s in SIG_JK_CMB]
    jkc = [np.sqrt(max(s ** 2 - SIG_MOCK_COMP ** 2, 0)) for s in SIG_JK_COMP]
    clf = [np.sqrt(max(c ** 2 - CLUSTER ** 2, 0)) for c in CL_FLOOR]
    floor = float(np.mean(jk + jkc + clf))
    print(f"  floor after correction, km/s: jackknife excess along CMB {jk[0]:.0f}, {jk[1]:.0f}; per component "
          f"{jkc[0]:.0f}, {jkc[1]:.0f}; corrected-map power at l = 2, 3 minus clustering {clf[0]:.0f}, {clf[1]:.0f}"
          f" -> floor about {floor:.0f} km/s")
    print(f"  as a residual dipole of the mean redshift error: |D[eps]| = {floor * R / C_KMS:.5f} in ln(1+z); "
          f"if all from the failures: |D[mu_c]| = {floor * R / C_KMS / f:.4f} (before correction {np.linalg.norm(D_c):.4f})")

    # requirements
    Nq = keep.sum()
    sig_y = d["y"][keep].std()
    k_sky = SIG_MOCK_CMB / (C_KMS * sig_y * np.sqrt(3 / Nq))
    print(f"\n  statistics: N = {Nq}, spread of ln(1+z) sigma_y = {sig_y:.3f}; c sigma_y sqrt(3/N) = "
          f"{C_KMS * sig_y * np.sqrt(3 / Nq):.0f} km/s; mock error {SIG_MOCK_CMB:.0f} -> sky and clustering factor k = {k_sky:.2f}")
    print("  requirements (total error along the CMB = sqrt(stat^2 + sys^2)):")
    for goal, tot in GOALS.items():
        line = f"    {goal:<26} total < {tot:4.0f} km/s:"
        for mult in (1, 3, 10):
            stat = SIG_MOCK_CMB / np.sqrt(mult)
            sysmax = np.sqrt(tot ** 2 - stat ** 2) if tot > stat else np.nan
            line += f"  N = {mult:2d} x Quaia: sys < {sysmax:4.0f}" if np.isfinite(sysmax) else f"  N = {mult:2d} x Quaia: impossible"
        for sysv in (floor, floor / 2):
            if tot > sysv:
                need = (SIG_MOCK_CMB / np.sqrt(tot ** 2 - sysv ** 2)) ** 2
                line += f";  sys {sysv:3.0f}: N >= {need:4.1f} x Quaia"
            else:
                line += f";  sys {sysv:3.0f}: impossible"
        print(line)
    for sysv in (50.0, 25.0):
        print(f"  sys < {sysv:.0f} km/s needs |D[eps]| < {sysv * R / C_KMS:.1e} in ln(1+z); with Quaia's failure rate "
              f"f = {f:.2f}, |D[mu_c]| < {sysv * R / C_KMS / f:.1e} (Quaia DR3 before correction {np.linalg.norm(D_c):.1e},"
              f" so the correction must remove {1 - sysv * R / C_KMS / f / np.linalg.norm(D_c):.3f} of it)")
    print(f"  best possible with the present floor ({floor:.0f} km/s), any N: 800 vs 370 at {430 / floor:.1f} sd, "
          f"370 detected at {370 / floor:.1f} sd")


if __name__ == "__main__":
    run()
