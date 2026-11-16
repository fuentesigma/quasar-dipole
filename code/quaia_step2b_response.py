#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Response r of the Quaia redshifts to a shift of the true redshift, from the DESI matches (section 3.1),
and the corrected dipole in subsamples (section 5).
"""
import healpy as hp
import numpy as np

from quaia import C_KMS, CMB, describe, desi_match, dipole, load
from results import footprint

rng = np.random.default_rng(21)
BIN = 0.02
NBOOT = 200


def response(yD, yQ, patch, lo=0.25, hi=1.55):
    edges = np.arange(lo, hi + BIN / 2, BIN)
    k = (yD > lo) & (yD < hi)
    idx = np.digitize(yD[k], edges) - 1
    nb = len(edges) - 1
    cnt = np.bincount(idx, minlength=nb)
    mean = np.bincount(idx, weights=yQ[k], minlength=nb) / np.maximum(cnt, 1)
    good = cnt >= 50
    centres = 0.5 * (edges[1:] + edges[:-1])
    slope = np.gradient(mean[good], centres[good])
    w = cnt[good]
    return np.sum(slope * w) / w.sum(), centres[good], slope


def boot(yD, yQ, patch):
    ids = np.unique(patch)
    out = []
    for _ in range(NBOOT):
        pick = rng.choice(ids, len(ids))
        wts = np.bincount(np.searchsorted(ids, pick), minlength=len(ids))[np.searchsorted(ids, patch)]
        rep = np.repeat(np.arange(len(yD)), wts)
        out.append(response(yD[rep], yQ[rep], patch[rep])[0])
    return np.std(out)


def run():
    d = load()
    D = desi_match(d)
    m = np.isfinite(D["z"]) & (D["z"] > 0)
    yD, yQ = np.log1p(D["z"][m]), d["y"][m]
    patch = hp.ang2pix(8, d["ra"][m], d["dec"][m], lonlat=True)
    print(f"Step 2b (1): response of the Quaia redshift to the true (DESI) redshift; {m.sum()} matches")
    slope_ols = np.polyfit(yD, yQ, 1)[0]
    print(f"  straight-line slope of y_Q on y_D, all matches: {slope_ols:.3f}")
    r, c, s = response(yD, yQ, patch)
    print(f"  local response r (binned mean, averaged over matches): {r:.3f} +- {boot(yD, yQ, patch):.3f}")
    for lo, hi in ((0.25, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.55)):
        k = (c >= lo) & (c < hi)
        print(f"    y_D {lo:.2f}-{hi:.2f} (z {np.expm1(lo):.1f}-{np.expm1(hi):.1f}): mean local slope {s[k].mean():.3f}")
    good = np.abs(yQ - yD) < 0.1
    rg, _, _ = response(yD[good], yQ[good], patch[good])
    print(f"  without catastrophic (|dy| > 0.1, {1 - good.mean():.3f} of matches): r = {rg:.3f}")
    print(f"  Expected kinematic signal along the CMB with this response: {370 * r:.0f} km/s "
          f"(range {370 * rg:.0f} if only non-catastrophic sources responded fully)")
    for lab, k in (("G < 20", d["G"][m] < 20), ("G >= 20", d["G"][m] >= 20)):
        print(f"    {lab:<8} r = {response(yD[k], yQ[k], patch[k])[0]:.3f}")

    print("\nStep 2b (2): corrected dipole (templates) in subsamples, |b|>30 S>0.5, km/s")
    keep = footprint(d)
    frac = d["zerr"] / (1 + d["z"])
    med = np.median(frac[keep])
    zmed = np.median(d["z"][keep])
    subs = {"all": keep, "G < 20": keep & (d["G"] < 20), "G >= 20": keep & (d["G"] >= 20),
            "precise sigma_z": keep & (frac < med), "imprecise sigma_z": keep & (frac >= med),
            f"z < {zmed:.2f}": keep & (d["z"] < zmed), f"z >= {zmed:.2f}": keep & (d["z"] >= zmed),
            "0.8 < z < 2.2": keep & (d["z"] > 0.8) & (d["z"] < 2.2),
            "north (b > 0)": keep & (d["b"] > 0), "south (b < 0)": keep & (d["b"] < 0)}
    for lab, k in subs.items():
        v, cov, _ = dipole(d["n"][k], d["y"][k], d["T"][k])
        print(f"  {lab:<20} N {k.sum():7d}  {describe(v, cov)}; along CMB {v @ CMB * C_KMS:+5.0f}")
    print("  (raw dipoles, no mock-bias subtraction; errors are least squares only)")


if __name__ == "__main__":
    run()
