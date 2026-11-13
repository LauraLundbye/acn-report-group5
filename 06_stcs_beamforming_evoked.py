import pickle
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import mne
from mne.transforms import apply_trans
from scipy import stats

from params import (VOL_SPACING, L_FREQ, H_FREQ, FILT_METHOD, SUBJECT_TO, TIME_WINDOWS,
                    CONTRAST, EVENT_ID, DATES, fnames)
from utils.logger import setup_report, save_report


# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
FIG_DIR = Path(fnames.ucloud_memberfiles) / "figures"

SUBJECTS = sorted(DATES)  # DATES maps subject ID -> recording date

# Spatio-temporal cluster-based permutation test of the per-subject difference, one test per window
CLUSTER_FORMING_P = 0.05            
ALPHA = 0.05 / len(TIME_WINDOWS)

# FreeSurfer volume atlas of SUBJECT_TO used to name cluster locations.
ATLAS = "aparc+aseg"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def same_vertices(stc_a, stc_b):
    return all(np.array_equal(a, b) for a, b in zip(stc_a.vertices, stc_b.vertices))


def grand_average(stc_list):
    """Mean over subjects. Refuses to average stcs defined on different vertices."""
    ref = stc_list[0]
    for stc in stc_list[1:]:
        if not same_vertices(ref, stc):
            raise RuntimeError("Source estimates have different vertices; check the morphs.")
    ga = ref.copy()
    ga.data = np.mean([stc.data for stc in stc_list], axis=0)
    return ga


def subject_evoked_stcs(subject, src_info):
    """Evoked LCMV stcs per condition (morphed) and the difference."""
    epochs = mne.read_epochs(
        fnames.sub_filtered_ica_epochs(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD))

    stim_codes = [EVENT_ID["stimulus/0"], EVENT_ID["stimulus/1"]]
    stim_epochs = epochs[np.isin(epochs.events[:, 2], stim_codes)].pick("mag")  # magnetometers only

    cond_epochs = {cond: stim_epochs[f"subjective_response in {resps}"] for cond, resps in CONTRAST.items()}
    n_before = {cond: len(ep) for cond, ep in cond_epochs.items()}
    mne.epochs.equalize_epoch_counts(list(cond_epochs.values()), method="mintime")  # in place
    n_each = len(next(iter(cond_epochs.values())))
    eq_epochs = mne.concatenate_epochs(list(cond_epochs.values()))

    # Common filter for both conditions, from the pooled equalised trials
    rank = mne.compute_rank(eq_epochs, tol=1e-6, tol_kind="relative")
    noise_cov = mne.compute_covariance(eq_epochs, tmin=None, tmax=0, rank=rank)
    data_cov = mne.compute_covariance(eq_epochs, tmin=0, tmax=None, rank=rank)  # full post-stimulus, as in the single-trial script

    fwd = mne.read_forward_solution(fnames.forward_model(subject=subject, src_info=src_info))
    filters = mne.beamformer.make_lcmv(
        eq_epochs.info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        pick_ori=None,                          # MNE default: orientations pooled -> non-negative output
        rank=rank,
    )

    morph = mne.read_source_morph(fnames.morph(subject=subject, src_info=src_info, subject_to=SUBJECT_TO))

    stcs = {}
    for cond, ep in cond_epochs.items():
        stc = mne.beamformer.apply_lcmv(ep.average(), filters)
        stcs[cond] = morph.apply(stc)

    cond_a, cond_b = CONTRAST
    stcs[f"{cond_a}-{cond_b}"] = stcs[cond_a] - stcs[cond_b]
    return stcs, n_before, n_each


def cluster_test(diff_stcs, adjacency, t0, t1):
    """Spatio-temporal one-sample cluster permutation test of the per-subject differences against 0,
    within one window. Clusters can extend over neighbouring voxels and neighbouring time points.
    diff_stcs: one full-length difference stc per subject."""
    win_stcs = [stc.copy().crop(t0, t1) for stc in diff_stcs]
    X = np.array([stc.data.T for stc in win_stcs])   # (n_subjects, n_times, n_voxels)
    if adjacency.shape[0] != X.shape[2]:
        raise RuntimeError("Adjacency and source estimates have different numbers of voxels.")
    times = win_stcs[0].times

    t_threshold = stats.t.ppf(1 - CLUSTER_FORMING_P / 2, df=len(X) - 1)
    t_obs, clusters, cluster_p, _ = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        threshold=t_threshold,
        n_permutations="all",   # exact test: 2**n_subjects sign flips
        tail=0,                 # two-sided: PAS1 > PAS234 and PAS1 < PAS234
        adjacency=adjacency,    # spatial adjacency; neighbouring time points are adjacent too (max_step=1)
    )
    # t_obs: (n_times, n_voxels); each cluster: (time indices, voxel indices) of its points

    results, sig_mask = [], np.zeros(t_obs.shape, dtype=bool)
    for (t_idx, v_idx), p in zip(clusters, cluster_p):
        mask = np.zeros(t_obs.shape, dtype=bool)
        mask[t_idx, v_idx] = True
        peak = np.argmax(np.abs(t_obs[t_idx, v_idx]))
        results.append(dict(n_voxels=len(np.unique(v_idx)), p=float(p),
                            sign="PAS1 > PAS234" if t_obs[mask].mean() > 0 else "PAS1 < PAS234",
                            voxels=np.unique(v_idx), peak_voxel=int(v_idx[peak]),
                            t_start=float(times[t_idx.min()]), t_end=float(times[t_idx.max()]),
                            t_peak=float(times[t_idx[peak]])))
        if p < ALPHA:
            sig_mask |= mask
    results.sort(key=lambda r: r["p"])
    return t_obs, t_threshold, results, sig_mask, times


def plot_timecourses(grand_full, cond_a, cond_b):
    """Sanity check: mean over voxels per condition. Responses should appear after stimulus onset."""
    times = grand_full[cond_a].times * 1000
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for cond in (cond_a, cond_b):
        ax.plot(times, grand_full[cond].data.mean(axis=0), label=f"{cond}: mean over voxels")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    for t0, t1 in TIME_WINDOWS.values():
        ax.axvspan(t0 * 1000, t1 * 1000, color="grey", alpha=0.2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("LCMV output (a.u.)")
    ax.set_title("Per-condition grand average")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def voxel_labels(src_to, voxel_idx, atlas_img, id_to_name):
    """Atlas region name at each source-space voxel (voxel_idx indexes the in-use voxels of src_to).
    Source positions are in FreeSurfer surface RAS (metres) -> mm -> atlas voxel via inverse vox2ras_tkr."""
    rr_mm = src_to[0]["rr"][src_to[0]["vertno"][voxel_idx]] * 1000
    ijk = np.round(apply_trans(np.linalg.inv(atlas_img.header.get_vox2ras_tkr()), rr_mm)).astype(int)
    ids = np.asarray(atlas_img.dataobj)[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return [id_to_name.get(int(i), f"id {int(i)}") for i in ids]


def describe_cluster(cluster, src_to, atlas_img, id_to_name, n_regions=3):
    """Peak voxel region + the most common regions among the cluster's voxels (each voxel counted once)."""
    names = voxel_labels(src_to, cluster["voxels"], atlas_img, id_to_name)
    peak_name = voxel_labels(src_to, np.array([cluster["peak_voxel"]]), atlas_img, id_to_name)[0]
    regions, counts = np.unique(names, return_counts=True)
    order = np.argsort(counts)[::-1][:n_regions]
    return peak_name, ", ".join(f"{regions[i]} ({counts[i]})" for i in order)


def plot_volume(stc, src_to, title, clim="auto"):
    """Orthogonal slices on SUBJECT_TO's T1. clim="auto" is the MNE default."""
    fig = stc.plot(
        src=src_to,
        subject=SUBJECT_TO,
        subjects_dir=fnames.subjects_dir,
        mode="stat_map",
        clim=clim,
        show=False,
    )
    fig.suptitle(title)
    return fig


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    src_info = f"volume-spacing-{VOL_SPACING}mm"
    cond_a, cond_b = CONTRAST
    diff_name = f"{cond_a}-{cond_b}"
    conds = [cond_a, cond_b, diff_name]

    src_to = mne.read_source_spaces(fnames.src_volume(subject=SUBJECT_TO, vol_spacing=VOL_SPACING))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------- per subject --------------------
    trial_counts = {}
    full = {c: [] for c in conds}                                  # full time courses
    win = {c: {w: [] for w in TIME_WINDOWS} for c in conds}         # window means

    for subject in SUBJECTS:
        stcs, n_before, n_each = subject_evoked_stcs(subject, src_info)
        trial_counts[subject] = (n_before, n_each)

        # The morphed stcs must live on SUBJECT_TO's source space, otherwise
        # stc.plot(src=src_to) puts values at the wrong voxels.
        src_vertno = [s["vertno"] for s in src_to]
        if not all(np.array_equal(a, b) for a, b in zip(stcs[cond_a].vertices, src_vertno)):
            raise RuntimeError(f"{subject}: morphed stc vertices do not match the {SUBJECT_TO} source space. "
                               f"Recompute the morph with src_to=<{SUBJECT_TO} volume source space>.")

        with open(fnames.sub_stcs(subject=subject, src_info=src_info, method="LCMV_EVOKED_CONTRAST"), "wb") as f:
            pickle.dump(stcs, f)

        for c in conds:
            full[c].append(stcs[c])
            for w, (t0, t1) in TIME_WINDOWS.items():
                win[c][w].append(stcs[c].copy().crop(t0, t1).mean())

    print("\n=== Trials per condition (before -> after equalising) ===")
    for s, (n_before, n_each) in trial_counts.items():
        print(f"{s}: {n_before} -> {n_each} each")

    # -------------------- grand averages --------------------
    grand_full = {c: grand_average(full[c]) for c in conds}
    grand_win = {c: {w: grand_average(win[c][w]) for w in TIME_WINDOWS} for c in conds}

    # Sanity check: difference of grand averages == grand average of differences
    for w in TIME_WINDOWS:
        check = grand_win[cond_a][w].data - grand_win[cond_b][w].data
        assert np.allclose(check, grand_win[diff_name][w].data)

    # -------------------- cluster permutation test per window --------------------
    adjacency = mne.spatial_src_adjacency(src_to)
    atlas_img = nib.load(Path(fnames.subjects_dir) / SUBJECT_TO / "mri" / f"{ATLAS}.mgz")
    id_to_name = {v: k for k, v in mne.read_freesurfer_lut()[0].items()}

    stat_results = {}
    for w in TIME_WINDOWS:
        t0, t1 = TIME_WINDOWS[w]
        stat_results[w] = cluster_test(full[diff_name], adjacency, t0, t1)
        t_obs, t_thr, results, sig_mask, times = stat_results[w]
        print(f"\n{w}: t threshold {t_thr:.2f}, {len(results)} clusters, "
              f"{sum(r['p'] < ALPHA for r in results)} with p < {ALPHA:.3f}")
        for i, r in enumerate(results):
            r["peak_region"], r["regions"] = describe_cluster(r, src_to, atlas_img, id_to_name)
            print(f"  cluster {i + 1}: {r['n_voxels']} voxels, {r['t_start'] * 1000:.0f}-{r['t_end'] * 1000:.0f} ms, "
                  f"p = {r['p']:.4f}, {r['sign']}, peak: {r['peak_region']} at {r['t_peak'] * 1000:.0f} ms; "
                  f"regions: {r['regions']}")

    stats_path = Path(fnames.group_stcs_report).parent / f"cluster_test_{diff_name}.pkl"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "wb") as f:
        pickle.dump({"alpha": ALPHA, "cluster_forming_p": CLUSTER_FORMING_P, "subjects": SUBJECTS,
                     "vertices": grand_win[diff_name][next(iter(TIME_WINDOWS))].vertices,
                     "results": {w: dict(t_obs=r[0], t_threshold=r[1], clusters=r[2], times=r[4])
                                 for w, r in stat_results.items()}}, f)

    # -------------------- report + figures --------------------
    report_path = Path(fnames.group_stcs_report)
    report = setup_report(report_path, title="Group Source Time Courses Report")
    section = f"Evoked LCMV {diff_name}"

    fig = plot_timecourses(grand_full, cond_a, cond_b)
    fig.savefig(FIG_DIR / f"lcmv_evoked_timecourses_{SUBJECT_TO}.png", dpi=200)
    report.add_figure(fig, title="Grand average time courses",
                      caption="Grey: analysis windows. Sanity check: responses should appear after stimulus onset.",
                      section=section, replace=True)
    plt.close(fig)

    for w, (t0, t1) in TIME_WINDOWS.items():
        win_label = f"{w} ({int(t0 * 1000)}-{int(t1 * 1000)} ms)"

        for c in (cond_a, cond_b):
            title = f"Grand average {c} | {win_label}"
            fig = plot_volume(grand_win[c][w], src_to, title)
            fig.savefig(FIG_DIR / f"lcmv_evoked_{c}_{w}_{SUBJECT_TO}.png", dpi=200)
            report.add_figure(fig, title=title,
                              caption=f"Mean over {len(SUBJECTS)} subjects, window-averaged. "
                                      f"Slices at the voxel with the largest absolute value. "
                                      f"MNE automatic colour scale, specific to this map.",
                              section=section, replace=True)
            plt.close(fig)

        # contrast: t-map restricted to significant clusters
        t_obs, t_thr, results, sig_mask, times = stat_results[w]
        rows = "".join(f"<tr><td>{i + 1}</td><td>{r['n_voxels']}</td>"
                       f"<td>{r['t_start'] * 1000:.0f}-{r['t_end'] * 1000:.0f}</td><td>{r['p']:.4f}</td>"
                       f"<td>{r['sign']}</td><td>{r['peak_region']} ({r['t_peak'] * 1000:.0f} ms)</td>"
                       f"<td>{r['regions']}</td></tr>"
                       for i, r in enumerate(results[:10]))
        report.add_html(
            f"<p>Spatio-temporal one-sample cluster permutation test of the per-subject {diff_name} difference "
            f"against 0 within the window; clusters can extend over neighbouring voxels and time points "
            f"(n = {len(SUBJECTS)}, exact test, two-sided). Cluster-forming threshold |t| > {t_thr:.2f} "
            f"(p < {CLUSTER_FORMING_P}, df = {len(SUBJECTS) - 1}); clusters significant at p < {ALPHA:.3f} "
            f"(0.05 Bonferroni-corrected for {len(TIME_WINDOWS)} windows).</p>"
            f"<table border='1' cellpadding='4'><tr><th>Cluster</th><th>Voxels</th><th>Time (ms)</th><th>p</th>"
            f"<th>Direction</th><th>Peak region ({ATLAS}, time)</th><th>Most common regions (voxels)</th></tr>"
            f"{rows}</table>" + ("" if results else "<p>No clusters formed.</p>"),
            title=f"Cluster test {diff_name} | {win_label}", section=section, replace=True)

        if sig_mask.any():
            stc_t = grand_full[diff_name].copy().crop(*TIME_WINDOWS[w])     # window time points
            stc_t.data = np.where(sig_mask, t_obs, 0.0).T                  # (n_voxels, n_times)
            t_max = max(float(np.abs(t_obs[sig_mask]).max()), t_thr * 1.01)
            clim = dict(kind="value", pos_lims=[t_thr, (t_thr + t_max) / 2, t_max])
            title = f"t-map, significant clusters | {diff_name} | {win_label}"
            fig = plot_volume(stc_t, src_to, title, clim=clim)
            fig.savefig(FIG_DIR / f"lcmv_evoked_{diff_name}_{w}_clusters_{SUBJECT_TO}.png", dpi=200)
            report.add_figure(fig, title=title,
                              caption=f"t values of the voxels and time points in clusters with p < {ALPHA:.3f}; "
                                      f"everything else set to 0. Slices at the largest |t|. "
                                      f"Red: PAS1 > PAS234, blue: PAS1 < PAS234. A significant cluster does not "
                                      f"mean every voxel or time point in it differs.",
                              section=section, replace=True)
            plt.close(fig)

        # per-subject contrast maps (descriptive only, MNE automatic colour scale)
        for s, stc in zip(SUBJECTS, win[diff_name][w]):
            title = f"{s} | {diff_name} | {w}"
            fig = plot_volume(stc, src_to, title)
            report.add_figure(fig, title=title, section=f"{section} | per subject", replace=True)
            plt.close(fig)

    save_report(report, report_path)