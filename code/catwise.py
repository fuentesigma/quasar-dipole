#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""CatWISE quasars of Secrest et al. (2021): catalogue, mask, counts and dipole fits (appendix C).
Vectors are equatorial here, Galactic in quaia.py.
"""
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u

from sky import (C_KMS, CMB_B, CMB_L, CMB_V as V_CMB, DATA, LOG_TEMPLATES, METHOD as ROOT, NPIX, NSIDE, OUT,
                 TEMPLATES, template_dict)

OUT_CW = OUT / "catwise"

theta, phi = hp.pix2ang(NSIDE, np.arange(NPIX))
PIX_VEC = np.array(hp.pix2vec(NSIDE, np.arange(NPIX))).T            # equatorial unit vectors
_pc = SkyCoord(ra=np.degrees(phi) * u.deg, dec=(90 - np.degrees(theta)) * u.deg)
PIX_L, PIX_B = _pc.galactic.l.deg, _pc.galactic.b.deg
PIX_ELAT = _pc.barycentricmeanecliptic.lat.deg


def gal_unit(l, b):
    c = SkyCoord(l=l * u.deg, b=b * u.deg, frame="galactic").icrs
    return c.cartesian.xyz.value


def to_lb(v):
    c = SkyCoord(x=v[0], y=v[1], z=v[2], representation_type="cartesian", frame="icrs")
    g = c.galactic
    return g.l.deg % 360, g.b.deg


CMB = gal_unit(CMB_L, CMB_B)


def load_catalogue(w1_max=16.4, w1_min=9.0):
    d = fits.open(DATA / "catwise" / "catwise_agns_masked_final_w1lt16p5_alpha.fits")[1].data
    s = (d["w1"] < w1_max) & (d["w1"] > w1_min)
    return {k: np.asarray(d[k][s], float) for k in
            ("ra", "dec", "l", "b", "elat", "w1", "w2", "w12", "w1e", "w2e", "w12e", "w1cov", "w2cov", "ebv", "alpha_W1")}


def _ellipse(lon, lat, lon0, lat0, a, e, pa):
    """Pixels inside an ellipse on the sphere given by its foci, as in Secrest et al."""
    c0 = SkyCoord(l=lon0 * u.deg, b=lat0 * u.deg, frame="galactic")
    f1 = c0.directional_offset_by((90 - pa) * u.deg, a * e * u.deg)
    f2 = c0.directional_offset_by((270 - pa) * u.deg, a * e * u.deg)
    p = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame="galactic")
    return p.separation(f1).deg + p.separation(f2).deg <= 2 * a


def mask(galcut=30.0, symmetric=True):
    """Mask at NSIDE 64, Galactic cut and the point-source masks of Secrest et al., made symmetric."""
    m = np.abs(PIX_B) > galcut
    ps = fits.open(DATA / "catwise" / "MASKS_exclude_master_final.fits")[1].data
    gl = SkyCoord(ra=ps["ra"] * u.deg, dec=ps["dec"] * u.deg).galactic
    lon, lat = gl.l.deg, gl.b.deg
    circ = (ps["pa"] <= 2) & (ps["radius"] < 30)
    for L, B, r in zip(lon[circ], lat[circ], ps["radius"][circ]):
        for sgn in ((1, -1) if symmetric else (1,)):
            v = sgn * gal_unit(L, B)
            m[hp.query_disc(NSIDE, v, np.radians(r))] = False
    ell = ps["pa"] > 2
    for L, B, r, ba, pa in zip(lon[ell], lat[ell], ps["radius"][ell], ps["ba"][ell], ps["pa"][ell]):
        m[_ellipse(PIX_L, PIX_B, L, B, r, ba, pa)] = False
        if symmetric:
            m[_ellipse(PIX_L, PIX_B, L + 180.0, -B, r, ba, -pa)] = False
    return m


def count_map(cat, sel=None):
    s = np.ones(len(cat["ra"]), bool) if sel is None else sel
    return np.bincount(hp.ang2pix(NSIDE, cat["ra"][s], cat["dec"][s], lonlat=True), minlength=NPIX).astype(float)


def mean_map(cat, key):
    pix = hp.ang2pix(NSIDE, cat["ra"], cat["dec"], lonlat=True)
    n = np.bincount(pix, minlength=NPIX)
    s = np.bincount(pix, weights=cat[key], minlength=NPIX)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def templates(extra=None):
    T = template_dict()
    if extra:
        T.update(extra)
    return T


def secrest_dipole(N, m):
    """Estimator of Secrest et al., linear correction in |ecliptic latitude| then healpy.fit_dipole."""
    x = np.abs(PIX_ELAT[m])
    k, c0 = np.polyfit(x, N[m], 1)
    Nc = np.full(NPIX, hp.UNSEEN)
    Nc[m] = N[m] / (1 + k / c0 * x)
    mono, dip = hp.fit_dipole(Nc)
    return dip / mono


def poisson_fit(N, m, T=None, dipole=True, iters=50):
    """Poisson fit <N_p> = exp(b0 + d.n_p + sum_k c_k T_kp), coefficients and covariance."""
    cols = [np.ones(m.sum())]
    if dipole:
        cols += list(PIX_VEC[m].T)
    names = ["b0"] + (["dx", "dy", "dz"] if dipole else [])
    if T:
        for k, t in T.items():
            t = t[m]
            cols.append((t - t.mean()) / t.std())
            names.append(k)
    X = np.column_stack(cols)
    y = N[m]
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.mean())
    for _ in range(iters):
        mu = np.exp(X @ beta)
        H = (X.T * mu) @ X
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(X @ beta)
    cov = np.linalg.inv((X.T * mu) @ X)
    phi_od = np.sum((y - mu) ** 2 / mu) / (len(y) - X.shape[1])  # over-dispersion from clustering
    return dict(zip(names, beta)), cov * max(phi_od, 1.0), names, phi_od


def describe(d, cov=None):
    D = np.linalg.norm(d)
    l, b = to_lb(d)
    ang = np.degrees(np.arccos(d @ CMB / D))
    e = "" if cov is None else f" +- {np.sqrt(np.trace(cov) / 3):.4f}"
    return f"D = {D:.4f}{e} toward (l,b) = ({l:5.1f}, {b:+5.1f}), {ang:5.1f} deg from CMB"


def kinematic_amplitude(cat):
    """Ellis & Baldwin, D = [2 + x(1+alpha)] beta, with x the slope of log N(>S) at the limit."""
    w1 = cat["w1"]
    lim = 16.4
    n1, n2 = (w1 < lim - 0.05).sum(), (w1 < lim).sum()
    x = np.log10(n2 / n1) / (0.4 * 0.05)  # d log N / d log S = -(1/0.4) d log N / d m, sign below
    alpha = abs(np.nanmean(cat["alpha_W1"]))  # stored as S ~ nu^{+alpha}
    return (2 + x * (1 + alpha)) * V_CMB / C_KMS, x, alpha
