#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Dipole of the redshifts in shells of observed ln(1+z) of width W (appendix B): response to an added
motion, scatter on the mocks and change with W.
"""
import healpy as hp
import numpy as np

from mocks import NSIDE_ARF, cells, theory_fields
from quaia import C_KMS, CMB, CMB_V, load
from quaia_step2_jackknife import mock_y
from results import footprint

rng = np.random.default_rng(51)
WIDTHS = (np.inf, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002)
N_MOCK = 60


def shell_dipole(n, y, T, W):
    if np.isinf(W):
        s = np.zeros(len(y), int)
    else:
        s = np.floor(y / W).astype(int)
        s -= s.min()
    cnt = np.bincount(s)
    good = cnt[s] >= 20
    s, n, y, T = s[good], n[good], y[good], T[good]
    cnt = np.maximum(np.bincount(s), 1)

    def dm(x):
        return x - (np.bincount(s, weights=x, minlength=len(cnt)) / cnt)[s]

    X = np.column_stack([dm(c) for c in np.column_stack([n, T]).T])
    coef = np.linalg.lstsq(X, dm(y), rcond=None)[0]
    return -coef[:3]


def all_widths(n, y, T):
    return np.array([shell_dipole(n, y, T, W) for W in WIDTHS])


def run():
    d = load()
    keep = footprint(d)
    n, T = d["n"][keep], d["T"][keep]
    v_kin = CMB_V / C_KMS * CMB

    # response
    base = all_widths(n, d["y"][keep], T)
    inj = all_widths(n, d["y"][keep] - n @ v_kin, T)
    R = (inj - base) @ CMB / (CMB_V / C_KMS)
    inj2 = all_widths(n, d["y"][keep] - 2 * n @ v_kin, T)
    R2 = (inj2 - base) @ CMB / (2 * CMB_V / C_KMS)

    # mocks
    cell = cells(d["T_raw"])
    order_cell = np.argsort(cell, kind="stable")
    pix = hp.ang2pix(NSIDE_ARF, d["ra"], d["dec"], lonlat=True)
    mask = hp.ud_grade(hp.read_map("data/selection_function_NSIDE64_G20.5.fits"), NSIDE_ARF) > 0.5
    zmed = np.median(d["z"][mask[pix]])
    fields = theory_fields(3 * NSIDE_ARF - 1)
    mk = []
    for i in range(N_MOCK):
        ym = mock_y(d, cell, order_cell, pix, fields, zmed, v_kin)
        mk.append(all_widths(n, ym[keep], T))
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1} mocks", flush=True)
    mk = np.array(mk)                                   # (mock, width, 3)
    cal_m = (mk @ CMB) / R * C_KMS                      # along CMB, calibrated
    cal_d = (base @ CMB) / R * C_KMS
    bias = cal_m.mean(0) - CMB_V

    print("Step 5b: redshift dipole along the CMB in shells of observed ln(1+z), |b|>30 S>0.5, templates (km/s)")
    print(f"  {'width':>7} {'shells':>6} {'response R':>10} {'(800 km/s)':>10} {'mock mean':>9} {'mock sd':>7} "
          f"{'data':>6} {'data - bias':>11} {'sd from 370':>11}")
    yk = d["y"][keep]
    for j, W in enumerate(WIDTHS):
        ns = 1 if np.isinf(W) else int(np.ptp(yk) / W) + 1
        sd = cal_m[:, j].std()
        corr = cal_d[j] - bias[j]
        print(f"  {('all' if np.isinf(W) else f'{W:.3f}'):>7} {ns:6d} {R[j]:10.3f} {R2[j]:10.3f} {cal_m[:, j].mean():9.0f} "
              f"{sd:7.0f} {cal_d[j]:6.0f} {corr:11.0f} {(CMB_V * 0.909 - corr) / sd:11.1f}")

    # differences from the single shell
    diff_m = cal_m[:, 1:] - cal_m[:, :1]
    diff_d = cal_d[1:] - cal_d[0]
    use = [j for j in range(diff_m.shape[1]) if diff_m[:, j].std() > 0]
    C = np.cov(diff_m[:, use].T)
    dd = diff_d[use] - diff_m[:, use].mean(0)
    chi2 = dd @ np.linalg.solve(C, dd)
    chi2_m = [((x - diff_m[:, use].mean(0)) @ np.linalg.solve(C, x - diff_m[:, use].mean(0))) for x in diff_m[:, use]]
    print(f"  changes with width relative to one shell, data vs mocks: chi2 = {chi2:.1f} for {len(use)} widths; "
          f"mocks: median {np.median(chi2_m):.1f}, fraction >= data {np.mean(np.array(chi2_m) >= chi2):.3f} "
          f"(mock covariance from {N_MOCK} mocks, in-sample: indicative only)")
    for j in use:
        z = (diff_d[j] - diff_m[:, j].mean()) / diff_m[:, j].std()
        print(f"    width {WIDTHS[j + 1]:.3f}: data change {diff_d[j]:+6.0f}, mocks {diff_m[:, j].mean():+5.0f} +- "
              f"{diff_m[:, j].std():4.0f} -> {z:+.1f} sd")
    np.savez("out/quaia_step5b_shells.npz", widths=np.array(WIDTHS), R=R, R2=R2, mocks=mk, data=base)


if __name__ == "__main__":
    run()
