#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""LCDM clustering of CatWISE with the redshift distribution from DESI, weighted for the incompleteness
of DESI in W1, against the residual power of the data.
"""
import numpy as np
import healpy as hp
import pandas as pd
import camb
from camb.sources import SplinedSourceWindow
from astropy.coordinates import SkyCoord
import astropy.units as u
import catwise as cw

cat = cw.load_catalogue()
m = np.load(cw.OUT_CW / "mask.npy")
pix = hp.ang2pix(cw.NSIDE, cat["ra"], cat["dec"], lonlat=True)
D = pd.read_csv(cw.DATA / "desi_catwise" / "desi_catwise_like.csv"); D = D[D.zwarn == 0]
cc = SkyCoord(ra=cat["ra"] * u.deg, dec=cat["dec"] * u.deg)
idx, sep, _ = cc.match_to_catalog_sky(SkyCoord(ra=D.ra.values * u.deg, dec=D.dec.values * u.deg))
dpix = np.bincount(hp.ang2pix(cw.NSIDE, D.ra.values, D.dec.values, lonlat=True), minlength=cw.NPIX)
foot = (m & (dpix >= 20))[pix]
hit = (sep.arcsec < 1.5) & m[pix]
M = D.iloc[idx[hit]]
w1 = cat["w1"][hit]
# weight each spectrum by 1 / (fraction of CatWISE sources with spectra in its W1 bin)
bins = np.arange(9, 16.45, 0.1)
nall = np.histogram(cat["w1"][foot], bins)[0]
nhit = np.histogram(w1, bins)[0]
comp = np.where(nall > 0, nhit / np.maximum(nall, 1), 1)
w = 1 / np.maximum(comp[np.clip(np.digitize(w1, bins) - 1, 0, len(comp) - 1)], 0.05)
keep = M.spectype.values != "STAR"
z, w = M.z.values[keep], w[keep]
print(f"weighted: z<0.3 {np.average(z < 0.3, weights=w):.1%}, z<0.8 {np.average(z < 0.8, weights=w):.1%}, median "
      f"{np.median(np.repeat(z, np.round(w * 3).astype(int))):.2f}")


def theory(zs, ws, bias_low=1.3, lmax=60):
    h, e = np.histogram(zs, bins=96, range=(0.0, 4.8), weights=ws, density=True)
    zc = 0.5 * (e[1:] + e[:-1])
    b = np.maximum(0.278 * ((1 + zc) ** 2 - 6.565) + 2.393, bias_low)
    p = camb.CAMBparams()
    p.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120)
    p.InitPower.set_params(As=2.1e-9, ns=0.965)
    p.set_for_lmax(lmax, lens_potential_accuracy=0)
    p.SourceWindows = [SplinedSourceWindow(z=zc + 1e-3, W=h, bias_z=b)]
    p.SourceTerms.limber_windows = False
    p.SourceTerms.counts_redshift = True
    p.SourceTerms.counts_lensing = False
    return camb.get_results(p).get_source_cls_dict(raw_cl=True)["W1xW1"][: lmax + 1]


LMAX = 3 * cw.NSIDE - 1
res = np.load(cw.OUT_CW / "residual_power_depth.npy")[1]  # data after Quaia templates + depth maps
old = np.load(cw.OUT_CW / "cl_theory.npy")
for bl in (1.0, 1.3, 1.6):
    new = theory(z, w, bl)
    np.save(cw.OUT_CW / f"cl_theory_desi_b{bl}.npy", new)
    rr = [res[a:b + 1].mean() / new[a:b + 1].mean() for a, b in [(2, 5), (6, 15), (16, 40), (41, 60)]]
    print(f"low-z bias floor {bl}: prediction / old prediction at ell 2-5: {new[2:6].mean() / old[2:6].mean():.1f}x;  "
          f"data / prediction at ell 2-5, 6-15, 16-40, 41-60: " + ", ".join(f"{r:.1f}x" for r in rr)
          + f";  rms clustering dipole {np.sqrt(9 * new[2] / (4 * np.pi)):.4f}")
