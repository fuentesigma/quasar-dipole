#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Redshifts and classes of CatWISE-like sources from DESI DR1 spectra (data/desi_catwise)."""
import numpy as np
import healpy as hp
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import catwise as cw

cat = cw.load_catalogue()
m = np.load(cw.OUT_CW / "mask.npy")
pix = hp.ang2pix(cw.NSIDE, cat["ra"], cat["dec"], lonlat=True)
inmask = m[pix]
D = pd.read_csv(cw.DATA / "desi_catwise" / "desi_catwise_like.csv")
D = D[D.zwarn == 0]
cd = SkyCoord(ra=D.ra.values * u.deg, dec=D.dec.values * u.deg)
cc = SkyCoord(ra=cat["ra"] * u.deg, dec=cat["dec"] * u.deg)
idx, sep, _ = cc.match_to_catalog_sky(cd)
hit = (sep.arcsec < 1.5) & inmask
# DESI footprint: NSIDE-64 pixels where DESI measured many CatWISE-like objects
dpix = np.bincount(hp.ang2pix(cw.NSIDE, D.ra.values, D.dec.values, lonlat=True), minlength=cw.NPIX)
foot = m & (dpix >= 20)
infoot = foot[pix]
print(f"CatWISE in mask: {inmask.sum()}; inside DESI footprint: {infoot.sum()}; with a good DESI spectrum: {hit.sum()} "
      f"({hit.sum() / infoot.sum():.0%} of those in the footprint)")

M = D.iloc[idx[hit]]
w1 = cat["w1"][hit]
print("\nclasses:", (M.spectype.value_counts(normalize=True) * 100).round(1).to_dict())
for lo, hi in [(9, 15.5), (15.5, 16.0), (16.0, 16.4)]:
    s = (w1 >= lo) & (w1 < hi)
    fr = M.spectype[s].value_counts(normalize=True)
    comp = s.sum() / (infoot & (cat["w1"] >= lo) & (cat["w1"] < hi)).sum()
    print(f"  W1 {lo}-{hi}: spectra for {comp:.0%}; STAR {fr.get('STAR', 0):.1%}, GALAXY {fr.get('GALAXY', 0):.1%}, QSO {fr.get('QSO', 0):.1%}")

z = M.z.values[M.spectype.values != "STAR"]
print(f"\nredshifts (non-stars): median {np.median(z):.2f}; z<0.3: {np.mean(z < 0.3):.1%}; z<0.8: {np.mean(z < 0.8):.1%}; z>2: {np.mean(z > 2):.1%}")
np.save(cw.OUT_CW / "desi_z_catwise.npy", z)

# contaminant fraction against the templates
T = cw.templates(); T["abs_elat"] = np.abs(cw.PIX_ELAT); T.update(dict(np.load(cw.OUT_CW / "depth_maps.npz")))
mp = hp.ang2pix(cw.NSIDE, cat["ra"][hit], cat["dec"][hit], lonlat=True)
star = (M.spectype.values == "STAR")
lowz = (M.spectype.values != "STAR") & (M.z.values < 0.3)
print("\nper-object correlation of being a star / a z<0.3 object with the templates (t-statistic):")
for k in ["dust", "stars", "unwise", "unwisescan", "abs_elat", "w12e_faint", "log_w1cov"]:
    t = T[k][mp]; t = (t - t.mean()) / t.std()
    out = []
    for flag in (star, lowz):
        r = np.corrcoef(t, flag)[0, 1]
        out.append(r * np.sqrt(len(t)))
    print(f"  {k:11s} star {out[0]:+6.1f}   low-z {out[1]:+6.1f}")
