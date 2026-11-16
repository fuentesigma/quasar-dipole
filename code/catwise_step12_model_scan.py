#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

"""Every subset of seven groups of templates for CatWISE, with |ecliptic latitude| always in: dust, stars,
m10, unwise, unwisescan, the WISE depth maps and |b|. Amplitude, direction and AIC of each model.
"""
import itertools
import numpy as np
import catwise as cw

cat = cw.load_catalogue()
m = np.load(cw.OUT_CW / "mask.npy")
N = cw.count_map(cat)
Dk = cw.kinematic_amplitude(cat)[0]
T0 = cw.templates()
depth = dict(np.load(cw.OUT_CW / "depth_maps.npz"))
GROUPS = {"dust": ["dust"], "stars": ["stars"], "m10": ["m10"], "unwise": ["unwise"],
          "unwisescan": ["unwisescan"], "depth": list(depth), "absb": ["abs_b"]}
ALL = dict(T0, **depth, abs_b=np.abs(cw.PIX_B), abs_elat=np.abs(cw.PIX_ELAT))

y = N[m]
rows = []
for r in range(len(GROUPS) + 1):
    for combo in itertools.combinations(GROUPS, r):
        keys = ["abs_elat"] + [k for g in combo for k in GROUPS[g]]
        b, cov, names, od = cw.poisson_fit(N, m, T={k: ALL[k] for k in keys})
        d = np.array([b["dx"], b["dy"], b["dz"]])
        X = [np.ones(m.sum())] + list(cw.PIX_VEC[m].T) + [(ALL[k][m] - ALL[k][m].mean()) / ALL[k][m].std() for k in keys]
        mu = np.exp(np.column_stack(X) @ np.array([b[n] for n in names]))
        loglike = np.sum(y * np.log(mu) - mu)
        rows.append(("+".join(combo) or "elat only", len(names), loglike, od, np.linalg.norm(d) / Dk,
                     np.degrees(np.arccos(d @ cw.CMB / np.linalg.norm(d)))))

od_ref = max(r[3] for r in rows if r[0] == "elat only")
aic = np.array([-2 * r[2] / od_ref + 2 * r[1] for r in rows])
order = np.argsort(aic)
ratio = np.array([r[4] for r in rows]); ang = np.array([r[5] for r in rows])
print(f"{len(rows)} models. D/D_kin: median {np.median(ratio):.2f}, range {ratio.min():.2f}-{ratio.max():.2f}; "
      f"angle to CMB: median {np.median(ang):.0f}, range {ang.min():.0f}-{ang.max():.0f} deg")
print("\nbest 15 by AIC (over-dispersion-corrected):")
for i in order[:15]:
    r = rows[i]
    print(f"  dAIC {aic[i] - aic[order[0]]:7.1f}  D/Dk {r[4]:.2f}  {r[5]:4.0f} deg  {r[0]}")
for lab, sel in (("with |b|", ["absb" in r[0] for r in rows]), ("without |b|", ["absb" not in r[0] for r in rows]),
                 ("with stars", ["stars" in r[0] for r in rows]), ("without stars", ["stars" not in r[0] for r in rows])):
    s = np.array(sel)
    print(f"{lab:13s}: D/Dk median {np.median(ratio[s]):.2f} (range {ratio[s].min():.2f}-{ratio[s].max():.2f}), "
          f"angle median {np.median(ang[s]):.0f} deg")
good = aic - aic[order[0]] < 10
print(f"models within dAIC < 10 of the best: {good.sum()}, D/Dk {ratio[good].min():.2f}-{ratio[good].max():.2f}, "
      f"angle {ang[good].min():.0f}-{ang[good].max():.0f} deg")
np.savez(cw.OUT_CW / "model_scan.npz", names=[r[0] for r in rows], aic=aic, ratio=ratio, angle=ang)
