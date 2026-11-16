#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar

import sys
from pathlib import Path

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import plotstyle
from quaia import C_KMS, GC, desi_match, dipole, load, ols
from results import perm_dipole

OUT = Path("figs")
COL, FULL = 3.46, 7.09
BLUE, MAGENTA, TEAL = "#1f5fa8", "#b0306a", "#2a9d8f"
TINT, TINT_2 = "#e4ebf5", "#9db4d6"
INK, INK_2, MUTED, BAND = plotstyle.INK, plotstyle.INK_2, plotstyle.MUTED, TINT
V_MOTION = 370.0


def style():
    plotstyle.use()
    mpl.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.6,
                         "lines.markersize": 4.5, "figure.dpi": 150, "pdf.fonttype": 42})


SKY_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "quaia", [BLUE, "#8fafd4", "#f6f6f4", "#d895b3", MAGENTA])
SKY_RANGE = (-0.07, 0.0, 0.025)
SKY_EDGE, SKY_EDGE_LW = "white", 3.0


def galactic_mean_map(d, values, nside=64, fwhm_deg=12.0):
    pix = hp.ang2pix(nside, d["l"], d["b"], lonlat=True)
    npix = hp.nside2npix(nside)
    s = np.bincount(pix, weights=values, minlength=npix)
    n = np.bincount(pix, minlength=npix).astype(float)
    fw = np.radians(fwhm_deg)
    s_sm, n_sm = hp.smoothing(s, fwhm=fw), hp.smoothing(n, fwhm=fw)
    m = s_sm / np.maximum(n_sm, 1e-3 * n_sm.max())
    cover = hp.smoothing((n > 0).astype(float), fwhm=fw)
    return m, cover


def mollweide(ax, m, cover, norm, cmap):
    lon = np.radians(np.linspace(-180, 180, 1441))
    lat = np.radians(np.linspace(-90, 90, 721))
    LON, LAT = np.meshgrid(lon, lat)
    ll, bb = np.degrees(-LON) % 360, np.degrees(LAT)
    val = hp.get_interp_val(m, ll, bb, lonlat=True)
    cov = hp.get_interp_val(cover, ll, bb, lonlat=True)
    val = np.ma.masked_where(cov < 0.5, val)
    im = ax.pcolormesh(LON, LAT, val, cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.contour(LON, LAT, cov, levels=[0.5], colors=[SKY_EDGE], linewidths=SKY_EDGE_LW)
    ax.spines["geo"].set(edgecolor=SKY_EDGE, linewidth=SKY_EDGE_LW)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(color="white", lw=0.4, alpha=0.5)
    ax.set_facecolor("white")
    return im


def mark(ax, l, b, marker, **kw):
    x = -np.radians(((l + 180) % 360) - 180)
    ax.plot(x, np.radians(b), marker, **kw)


def panel_dust(ax, d, compact=False):
    D = desi_match(d)
    m = np.isfinite(D["z"])
    ebv = d["T_raw"][m, 0]
    edges = np.quantile(ebv, np.linspace(0, 1, 11))
    idx = np.clip(np.digitize(ebv, edges) - 1, 0, 9)
    xc = np.array([np.median(ebv[idx == i]) for i in range(10)])
    _, patch = np.unique(hp.ang2pix(8, d["ra"][m], d["dec"][m], lonlat=True), return_inverse=True)
    P = patch.max() + 1
    rng = np.random.default_rng(13)
    boot_w = np.array([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(500)], float)

    def boot_err(y):
        key = patch * 10 + idx
        s = np.bincount(key, weights=y, minlength=10 * P).reshape(P, 10)
        n = np.bincount(key, minlength=10 * P).reshape(P, 10)
        return ((boot_w @ s) / np.maximum(boot_w @ n, 1)).std(axis=0)
    x0, x1 = xc[0], 0.27
    ax.axhspan(-V_MOTION / C_KMS * 1e3, V_MOTION / C_KMS * 1e3, color=BAND, lw=0, zorder=0)
    slope = {}
    for y, color, name in ((d["y"][m], BLUE, "Quaia"), (np.log1p(D["z"][m]), INK, "DESI")):
        coef, _ = ols(ebv[:, None], y)
        a0 = y.mean() - coef[0] * ebv.mean()
        ref = a0 + coef[0] * x0
        slope[name] = coef[0]
        mean = np.array([y[idx == i].mean() for i in range(10)]) - ref
        err = boot_err(y)
        xs = np.array([x0, x1])
        ax.plot(xs, coef[0] * (xs - x0) * 1e3, color=color, lw=1.2, zorder=2)
        ax.errorbar(xc, mean * 1e3, err * 1e3, fmt="o", color=color, mec="white", mew=0.6, ms=4.5,
                    elinewidth=0.8, zorder=3)
    yq, yd = slope["Quaia"] * (x1 - x0) * 1e3, slope["DESI"] * (x1 - x0) * 1e3
    xa = x1 + 0.008
    for lo, hi, text in ((yd, 0.0, "selection"), (yq, yd, "errors")):
        ax.annotate("", (xa, lo), (xa, hi), arrowprops=dict(arrowstyle="<->", color=INK_2, lw=0.8,
                                                             shrinkA=0, shrinkB=0))
        ax.text(xa + 0.006, 0.5 * (lo + hi), text, fontsize=6.5, color=INK_2, va="center")
    ax.text(0.232, slope["DESI"] * (0.232 - x0) * 1e3 + 1.3, "DESI", color=INK, fontsize=7, ha="center")
    ax.text(0.205, slope["Quaia"] * (0.205 - x0) * 1e3 - 3.4, "Quaia", color=BLUE, fontsize=7, ha="center")
    ax.text(0.088, 0.62, "shift from our motion", fontsize=6, color=INK_2, va="center")
    ax.set_xlim(0.0, 0.33); ax.set_ylim(-17, 4)
    ax.set_xlabel(r"dust extinction $A_V$  [mag]")
    ax.set_ylabel(r"$\langle\ln(1+z)\rangle$, relative  [$10^{-3}$]")


def panel_dipole(ax, d):
    rows = [("whole sky", np.ones(len(d["y"]), bool)), (r"$|b|>30^\circ$", np.abs(d["b"]) > 30),
            (r"$|b|<30^\circ$", np.abs(d["b"]) < 30), (r"$|\delta|>30^\circ$", np.abs(d["dec"]) > 30),
            (r"$|\delta|<30^\circ$", np.abs(d["dec"]) < 30), (r"$G<20.0$", d["G"] < 20.0),
            (r"$20.0<G<20.5$", d["G"] >= 20.0)]
    o = np.load("out/redshift_mocks.npz")
    names = list(o["dip_names"])
    j = names.index("|b|>30 S>0.5: templates")
    mocks = o["dip_cells"][:, j]
    corr = o["real_dip"][j] - (mocks.mean(0) - o["v_kin"])
    corr_gc, corr_err = corr @ GC * C_KMS, np.sqrt(GC @ np.cov(mocks.T) @ GC) * C_KMS
    ax.axvspan(-V_MOTION, V_MOTION, color=BAND, lw=0, zorder=0)
    ax.axvline(0, color=plotstyle.AXIS, lw=0.8, zorder=1)
    for i, (name, k) in enumerate(rows):
        n, y, T = d["n"][k], d["y"][k], d["T"][k]
        v_raw, cov = perm_dipole(d, k)
        _, _, c = dipole(n, y, T)
        v_map = dipole(n, T @ c)[0]
        yi = len(rows) - i
        ax.errorbar(v_raw @ GC * C_KMS, yi + 0.13, xerr=np.sqrt(GC @ cov @ GC) * C_KMS, fmt="o", color=BLUE,
                    mec="white", mew=0.6, elinewidth=0.8, zorder=3)
        ax.plot(v_map @ GC * C_KMS, yi - 0.13, "o", mfc="white", mec=MAGENTA, mew=1.4, zorder=3)
    ax.errorbar(corr_gc, -0.4, xerr=corr_err, fmt="D", color=TEAL, mec="white", mew=0.6, elinewidth=0.8, ms=4.5)
    ax.set_yticks(list(range(len(rows), 0, -1)) + [-0.4])
    ax.set_yticklabels([r[0] for r in rows] + [r"corrected, $|b|>30^\circ$"])
    ax.get_yticklabels()[-1].set_color(INK)
    ax.axhline(0.35, color=plotstyle.GRID, lw=0.8)
    ax.set_xlabel("velocity of the dipole towards the Galactic Centre  [km/s]")
    ax.set_xlim(-900, 6800)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-1.0, len(rows) + 0.6)
    ax.text(-250, 0.35 + 0.5 * (len(rows) + 0.25), "our motion", rotation=90, fontsize=6.5, color=INK_2,
            ha="center", va="center")


FNL_BINS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 20), (20, 30), (30, 45), (45, 61)]
LMIN_ARF, LMIN_COUNTS = 10, 15


AXIS_C = plotstyle.AXIS


def tag(ax, letter, x=-0.02, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


def fig_overview(d):
    """Figure 1 (overview.pdf)."""
    _, _, c = dipole(d["n"], d["y"], d["T"])
    pred = d["T"] @ c
    data, cover = galactic_mean_map(d, d["y"] - d["y"].mean())
    model, _ = galactic_mean_map(d, pred - pred.mean())
    lo, mid, hi = SKY_RANGE
    norm = mpl.colors.TwoSlopeNorm(vmin=lo, vcenter=mid, vmax=hi)
    fig = plt.figure(figsize=(FULL, 5.3))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.32], hspace=0.12)
    top = outer[0].subgridspec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.04)
    for i, (m, letter) in enumerate(((data, "a"), (model, "b"))):
        ax = fig.add_subplot(top[0, i], projection="mollweide")
        im = mollweide(ax, m, cover, norm, SKY_CMAP)
        mark(ax, 0, 0, "+", color="white", ms=7, mew=1.2)
        tag(ax, letter, x=0.0, y=1.0)
    cax = fig.add_subplot(top[0, 2])
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(r"$\Delta\langle\ln(1+z)\rangle$", labelpad=2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.set_ticks([-0.06, -0.04, -0.02, 0.0, 0.02])
    bot = outer[1].subgridspec(1, 2, width_ratios=[1, 1.08], wspace=0.42)
    cx = fig.add_subplot(bot[0, 0])
    panel_dust(cx, d)
    tag(cx, "c", x=-0.2)
    dx = fig.add_subplot(bot[0, 1])
    panel_dipole(dx, d)
    dx.set_xlabel("velocity towards the Galactic Centre  [km/s]")
    dx.text(1370, 7.42, "measured", fontsize=6.5, color=BLUE, ha="center", va="bottom")
    dx.text(2400, 6.62, "templates", fontsize=6.5, color=MAGENTA, ha="left", va="top")
    tag(dx, "d", x=-0.5)
    fig.savefig(OUT / "overview.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(OUT / "overview.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_joint():
    """Figure 2 (joint.pdf)."""
    from scipy import stats
    rows = [r for r in np.load("out/quaia_step3_joint.npy", allow_pickle=True) if r["ns"] == 4]
    mocks = np.load("out/quaia_step3b_mocks.npy")[:, 6]
    names = {"templates": "templates", "S offset": "selection function", "S offset + templates": "both"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(FULL, 2.7), gridspec_kw=dict(width_ratios=[1.55, 1]))

    ax.axvline(V_MOTION, color=INK_2, lw=0.9, zorder=1)
    ax.axvline(800.0, color=INK_2, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.axvline(0.0, color=AXIS_C, lw=0.8, zorder=1)
    ylab = []
    for i, r in enumerate(rows):
        yi = len(rows) - i + (0.5 if r["bcut"] == 30 else 0.0)
        ax.errorbar(r["vN"], yi + 0.14, xerr=r["eN"], fmt="s", color=MAGENTA, mec="white", mew=0.6,
                    elinewidth=0.8, ms=4.2, zorder=3)
        ax.errorbar(r["vz"], yi - 0.14, xerr=r["ez"], fmt="o", color=BLUE, mec="white", mew=0.6,
                    elinewidth=0.8, ms=4.5, zorder=3)
        ylab.append((yi, names[r["model"]]))
    ax.set_yticks([y for y, _ in ylab])
    ax.set_yticklabels([t for _, t in ylab])
    for bcut, y0 in ((30, len(rows) + 0.5 - 1), (40, 2.0)):
        ax.text(-1180, y0, rf"$|b|>{bcut}^\circ$", fontsize=7, color=INK, ha="left", va="center", rotation=90)
    ax.axhline(3.75, color=plotstyle.GRID, lw=0.8)
    top = len(rows) + 1.15
    ax.text(V_MOTION - 25, top, "our motion", fontsize=6.5, color=INK_2, ha="right", va="bottom")
    ax.text(825.0, top, r"800 km/s", fontsize=6.5, color=INK_2, ha="left", va="bottom")
    r0 = rows[0]
    ax.text(r0["vN"], len(rows) + 0.5 + 0.42, "counts", fontsize=7, color=MAGENTA, ha="center", va="bottom")
    ax.text(r0["vz"], len(rows) + 0.5 + 0.30, "redshifts", fontsize=7, color=BLUE, ha="center", va="bottom")
    ax.set_xlim(-1250, 1800)
    ax.set_ylim(0.4, len(rows) + 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("velocity along the CMB dipole  [km/s]")
    ax.set_title("a", loc="left", x=-0.07, y=1.0, pad=4, fontsize=10, fontweight="bold")

    data_chi2 = rows[0]["chi2"]
    bins = np.linspace(0, 26, 53)
    bx.hist(mocks, bins=bins, color=TINT_2, lw=0, zorder=2)
    xs = np.linspace(0.05, 26, 300)
    bx.plot(xs, stats.chi2.pdf(xs, 3) * len(mocks) * (bins[1] - bins[0]), color=INK_2, lw=1.0, ls=(0, (3, 2)),
            zorder=3)
    bx.axvline(data_chi2, color=MAGENTA, lw=1.4, zorder=4)
    bx.text(data_chi2 - 0.4, bx.get_ylim()[1] * 0.92, "data", fontsize=7, color=INK, ha="right", va="top")
    bx.text(7.5, bx.get_ylim()[1] * 0.55, f"{len(mocks)} mock skies\nwith our motion only", fontsize=6.5,
            color=INK_2, ha="left", va="center")
    bx.set_xlabel(r"$\chi^2$ for a single velocity")
    bx.set_ylabel("mock skies")
    bx.set_xlim(0, 26)
    bx.grid(axis="x", visible=False)
    bx.set_title("b", loc="left", x=-0.07, y=1.0, pad=4, fontsize=10, fontweight="bold")
    fig.savefig(OUT / "joint.pdf", bbox_inches="tight")
    fig.savefig(OUT / "joint.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    style()
    which = sys.argv[1:] or ["overview", "joint"]
    if "overview" in which:
        fig_overview(load())
    if "joint" in which:
        fig_joint()
