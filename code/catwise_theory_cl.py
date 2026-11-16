#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""LCDM angular power spectrum of CatWISE-like quasars with CAMB, without the Limber approximation."""
import numpy as np
import camb
from camb.sources import SplinedSourceWindow
from astropy.io import fits
import catwise as cw

z = fits.open(cw.DATA / "quaia_G20.5.fits")[1].data["redshift_quaia"]
z = z[(z > 0.05) & (z < 4.5)]


def cl(bias_scale=1.0, zshift=0.0, lmax=60):
    h, e = np.histogram(z + zshift, bins=80, range=(0.02, 4.8), density=True)
    zc = 0.5 * (e[1:] + e[:-1])
    b = bias_scale * (0.278 * ((1 + zc) ** 2 - 6.565) + 2.393)  # Laurent et al. 2017
    p = camb.CAMBparams()
    p.set_cosmology(H0=67.4, ombh2=0.0224, omch2=0.120)
    p.InitPower.set_params(As=2.1e-9, ns=0.965)
    p.set_for_lmax(lmax, lens_potential_accuracy=0)
    p.SourceWindows = [SplinedSourceWindow(z=zc, W=h, bias_z=np.clip(b, 0.5, None))]
    p.SourceTerms.limber_windows = False
    p.SourceTerms.counts_redshift = True
    p.SourceTerms.counts_lensing = False
    r = camb.get_results(p)
    c = r.get_source_cls_dict(raw_cl=True)["W1xW1"]
    return c[: lmax + 1]


for bs, dz, lab in [(1.0, 0.0, "fiducial (Quaia dN/dz, Laurent bias)"), (1.3, 0.0, "bias x1.3"),
                    (1.0, -0.3, "dN/dz shifted by -0.3 in z")]:
    c = cl(bs, dz)
    ell = np.arange(len(c))
    c1 = c[2] * (c[2] / c[3])  # mild extrapolation to ell = 1
    print(f"{lab:38s} C_2 = {c[2]:.2e}  C_10 = {c[10]:.2e}  C_1 ~ {c1:.2e}  "
          f"rms clustering dipole = {np.sqrt(9 * c1 / (4 * np.pi)):.4f}")
np.save(cw.OUT_CW / "cl_theory.npy", cl())
