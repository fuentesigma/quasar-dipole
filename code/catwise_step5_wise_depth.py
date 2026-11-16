#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Maps of the depth of WISE built from the catalogue, and their effect on the dipole and the residual power.

log_w1cov, log_w2cov   mean number of W1 and W2 exposures
w1e_faint, w2e_faint   median photometric error for 16.0 < W1 < 16.4
w12e_faint             median W1-W2 error at the faint end
frac_near_cut          fraction with 0.8 <= W1-W2 < 0.9
"""
import numpy as np
import healpy as hp
import catwise as cw

LMAX = 3 * cw.NSIDE - 1
cat = cw.load_catalogue()
m = np.load(cw.OUT_CW / "mask.npy")
N = cw.count_map(cat)
Dk = cw.kinematic_amplitude(cat)[0]
pix = hp.ang2pix(cw.NSIDE, cat["ra"], cat["dec"], lonlat=True)


def pix_median(vals, sel):
    out = np.full(cw.NPIX, np.nan)
    order = np.argsort(pix[sel])
    p, v = pix[sel][order], vals[sel][order]
    bounds = np.r_[0, np.flatnonzero(np.diff(p)) + 1, len(p)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        out[p[a]] = np.median(v[a:b])
    return out


def smooth(x, fwhm_deg=3.0):
    """Smooth a masked map. Outside the mask it is set to the mean, then divided by the smoothed mask."""
    good = m & np.isfinite(x)
    y = np.where(good, x, 0.0)
    w = good.astype(float)
    ys = hp.smoothing(y, fwhm=np.radians(fwhm_deg))
    ws = hp.smoothing(w, fwhm=np.radians(fwhm_deg))
    return np.where(ws > 0.05, ys / np.maximum(ws, 1e-6), np.nanmean(x[good]))


faint = (cat["w1"] > 16.0) & (cat["w1"] < 16.4)
depth = {
    "log_w1cov": smooth(np.log10(cw.mean_map(cat, "w1cov"))),
    "log_w2cov": smooth(np.log10(cw.mean_map(cat, "w2cov"))),
    "w1e_faint": smooth(pix_median(cat["w1e"], faint)),
    "w2e_faint": smooth(pix_median(cat["w2e"], faint)),
    "w12e_faint": smooth(pix_median(cat["w12e"], faint)),
    "frac_near_cut": smooth(np.bincount(pix, weights=(cat["w12"] < 0.9), minlength=cw.NPIX) / np.maximum(N, 1)),
}
np.savez(cw.OUT_CW / "depth_maps.npz", **depth)

T = cw.templates(); T["abs_elat"] = np.abs(cw.PIX_ELAT)
T = {k: t for k, t in T.items() if t[m].std() > 0}
Q = dict(T)                     # Quaia templates + |elat|
QD = dict(T, **depth)           # + WISE depth maps

# correlation of the depth maps with each other and with the CMB direction
print("-- depth maps: correlation with |elat| and their own dipole direction --")
for k, v in depth.items():
    r = np.corrcoef(v[m], T["abs_elat"][m])[0, 1]
    t = np.full(cw.NPIX, hp.UNSEEN); t[m] = (v[m] - v[m].mean()) / v[m].std()
    _, dip = hp.fit_dipole(t)
    print(f"  {k:14s} corr with |elat| {r:+.2f}; own dipole {np.degrees(np.arccos(dip @ cw.CMB / np.linalg.norm(dip))):5.1f} deg from CMB")


def report(Tfit, label):
    b, c, names, od = cw.poisson_fit(N, m, T=Tfit)
    d = np.array([b["dx"], b["dy"], b["dz"]])
    sig = "  ".join(f"{k}:{b[k] / np.sqrt(c[names.index(k), names.index(k)]):+5.1f}s" for k in depth if k in Tfit)
    print(f"{label:34s} {cw.describe(d)}  D/D_kin = {np.linalg.norm(d) / Dk:4.2f}  over-disp {od:.2f}  {sig}")
    return b, d


def residual_power(Tfit, b, d):
    Z = {k: (Tfit[k] - Tfit[k][m].mean()) / Tfit[k][m].std() for k in Tfit}
    mu = np.exp(b["b0"] + cw.PIX_VEC @ d + sum(b[k] * Z[k] for k in Tfit))
    dl = np.zeros(cw.NPIX); dl[m] = N[m] / mu[m] - 1
    nbar = N[m].sum() / (m.sum() * hp.nside2pixarea(cw.NSIDE))
    return (hp.anafast(dl, lmax=LMAX) / m.mean() - 1 / nbar) / hp.pixwin(cw.NSIDE, lmax=LMAX) ** 2


print("\n-- dipole fits --")
bq, dq = report(Q, "Quaia templates + |elat|")
for k in depth:
    report(dict(Q, **{k: depth[k]}), f"  + {k}")
bqd, dqd = report(QD, "Quaia templates + all depth maps")

th = np.load(cw.OUT_CW / "cl_theory.npy")
ct = np.zeros(LMAX + 1); ct[:len(th)] = th
ell = np.arange(LMAX + 1); ct[len(th):] = th[-1] * (ell[len(th):] / (len(th) - 1)) ** -1.0
pq, pqd = residual_power(Q, bq, dq), residual_power(QD, bqd, dqd)
print("\n-- residual large-scale power relative to LCDM --")
for lo, hi in [(2, 5), (6, 15), (16, 40), (41, 100)]:
    s = slice(lo, hi + 1)
    print(f"  ell {lo:3d}-{hi:3d}: Quaia templates {pq[s].mean() / ct[s].mean():5.1f}x   + depth maps {pqd[s].mean() / ct[s].mean():5.1f}x")
np.save(cw.OUT_CW / "residual_power_depth.npy", np.array([pq, pqd]))
