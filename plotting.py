import argparse
import io
import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import mne

from params import VOL_SPACING, SUBJECT_TO, CONTRAST, DATES, fnames

WINDOWS = {
    "T1": ("Early", 0.132, 0.320),
    "T2": ("Late", 0.324, 0.512),
}


def pas_label(resps):
    return "Unaware (PAS1)" if sorted(int(r) for r in resps) == [1] else "Aware (PAS2-3-4)"


def load_grand_diff(src_info, diff_name):
    diffs = []
    for subject in DATES:
        path = fnames.sub_stcs(subject=subject, src_info=src_info, method="LCMV_EVOKED_CONTRAST")
        with open(path, "rb") as f:
            diffs.append(pickle.load(f)[diff_name])
    grand = diffs[0].copy()
    grand.data = np.mean([stc.data for stc in diffs], axis=0)
    return grand, len(diffs)


def is_time_axis(ax):
    legend = ax.get_legend()
    if legend is not None and legend.get_title().get_text() == "Vertex":
        return True
    return ax.get_xlabel().lower().startswith("time")


def default_outdir():
    parts = Path(fnames.group_stcs_report).resolve().parts
    if "derivatives" not in parts:
        raise SystemExit("No 'derivatives' folder in the report path, pass --outdir instead.")
    return Path(*parts[:parts.index("derivatives")]) / "figures"


def shared_clim(window_stcs, thresh_pct):
    vals = np.abs(np.concatenate([stc.data[:, 0] for stc in window_stcs.values()]))
    thr, top = np.percentile(vals, [thresh_pct, 99.95])
    return dict(kind="value", pos_lims=[float(thr), float((thr + top) / 2), float(top)])


def render_window(stc, src, clim, dpi):
    idx = int(np.argmax(np.abs(stc.data[:, 0])))

    fig = stc.plot(src=src, subject=SUBJECT_TO, subjects_dir=fnames.subjects_dir,
                   mode="stat_map", clim=clim, show=False)

    for ax in fig.axes:
        if is_time_axis(ax):
            ax.set_visible(False)
        else:
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
    for txt in fig.findobj(mpl.text.Text):
        txt.set_color("white")
    fig.patch.set_facecolor("black")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf), stc.vertices[0][idx], stc.data[idx, 0]


def main(outdir, dpi, thresh_pct):
    src_info = f"volume-spacing-{VOL_SPACING}mm"
    cond_a, cond_b = CONTRAST
    diff_name = f"{cond_a}-{cond_b}"
    title = f"{pas_label(CONTRAST[cond_a])} − {pas_label(CONTRAST[cond_b])}"

    grand, n_subjects = load_grand_diff(src_info, diff_name)
    src = mne.read_source_spaces(fnames.src_volume(subject=SUBJECT_TO, vol_spacing=VOL_SPACING))

    window_stcs = {key: grand.copy().crop(t0, t1).mean() for key, (_, t0, t1) in WINDOWS.items()}
    clim = shared_clim(window_stcs, thresh_pct)
    print(f"Shared colour limits (|value|): {clim['pos_lims']}")
    panels = {key: render_window(stc, src, clim, dpi) for key, stc in window_stcs.items()}

    fig = plt.figure(figsize=(10, 8), facecolor="black")
    fig.suptitle(f"{title}  | Beamformer LCMV evoked, grand average (n={n_subjects})",
                 color="white", fontsize=14, weight="bold")
    gs = fig.add_gridspec(len(WINDOWS), 1, hspace=0.3, top=0.88, bottom=0.11, left=0.02, right=0.98)
    lab_a, lab_b = (pas_label(CONTRAST[c]).split()[0] for c in (cond_a, cond_b))
    fig.text(0.02, 0.005,
             f"Values: Beamformer LCMV source strength for {lab_a} minus {lab_b} trials, averaged over the window and "
             f"{n_subjects} participants (fsaverage).\n"
             "The beamformer output is noise-normalised, so values are relative to the "
             "estimated noise at each location\n"
             f"Red = stronger for {lab_a}, blue = stronger for {lab_b} "
             f"|values| < {clim['pos_lims'][0]:.3f} ({thresh_pct:g}th percentile) not shown.",
             color="0.75", fontsize=8, va="bottom", linespacing=1.5)

    for row, (key, (name, t0, t1)) in enumerate(WINDOWS.items()):
        img, vertex, value = panels[key]
        ax = fig.add_subplot(gs[row])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{key}: {name} window ({t0 * 1000:.0f}–{t1 * 1000:.0f} ms)",
                     loc="left", color="white", fontsize=12, weight="bold", pad=18)
        ax.text(0, 1.02, f"Peak vertex {vertex}  |  value {value:.3g}",
                transform=ax.transAxes, color="0.75", fontsize=9, va="bottom")

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"evoked_contrast_{diff_name}_T1_T2.png"
    fig.savefig(out_path, dpi=dpi, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot grand-average Beamformer LCMV evoked contrast for T1 and T2.")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="Default: 'figures' folder next to 'derivatives'.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--thresh-pct", type=float, default=96,
                        help="Hide |values| below this percentile (MNE default is 96). Display only.")
    args = parser.parse_args()
    main(args.outdir or default_outdir(), args.dpi, args.thresh_pct)