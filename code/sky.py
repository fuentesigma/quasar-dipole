#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Constants, HEALPix grid and selection templates shared by quaia.py and catwise.py."""
from pathlib import Path

import healpy as hp
import numpy as np

METHOD = Path(__file__).resolve().parents[1]
DATA, OUT = METHOD / "data", METHOD / "out"

C_KMS = 299792.458
CMB_V, CMB_L, CMB_B = 369.82, 264.021, 48.253  # Planck 2018
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
TEMPLATES = ["dust", "stars", "m10", "unwise", "unwisescan", "mcs"]
LOG_TEMPLATES = ("stars", "unwise", "unwisescan")
TEMPLATE_DIR = DATA / "templates" / "selection_function_template_maps"


def unit(l_deg, b_deg):
    """Unit vectors from longitude and latitude in degrees."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], axis=-1)


def lb(v):
    """Longitude and latitude in degrees from unit vectors."""
    v = np.asarray(v)
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))


def template_dict():
    """Selection templates per pixel at NSIDE 64 (celestial), stars, unwise and unwisescan in log10."""
    maps = {}
    for k in TEMPLATES:
        x = np.load(TEMPLATE_DIR / f"map_{k}_NSIDE64.npy").astype(float)
        maps[k] = np.log10(np.maximum(x, 1.0)) if k in LOG_TEMPLATES else x
    return maps
