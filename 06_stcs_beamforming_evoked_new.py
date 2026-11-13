import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne

from params import (VOL_SPACING, L_FREQ, H_FREQ, FILT_METHOD, SUBJECT_TO, TIME_WINDOWS,
                    CONTRAST, EVENT_ID, DATES, fnames)
from utils.logger import setup_report, save_report

mpl.use("Agg")

def evoked_contrast_stcs(subject, src_info):
    epochs_path = fnames.sub_filtered_ica_epochs(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
    fwd_path = fnames.forward_model(subject=subject, src_info=src_info)
    morph_path = fnames.morph(subject=subject, src_info=src_info, subject_to=SUBJECT_TO)

    # stimulus epochs only
    epochs = mne.read_epochs(epochs_path)

    stim_codes = [EVENT_ID["stimulus/0"], EVENT_ID["stimulus/1"]]
    stim_epochs = epochs[np.isin(epochs.events[:, 2], stim_codes)].pick("mag")

    # equalise trial counts
    cond_epochs = {cond: stim_epochs[f"subjective_response in {resps}"] for cond, resps in CONTRAST.items()}
    mne.epochs.equalize_epoch_counts(list(cond_epochs.values()), method="mintime")
    eq_epochs = mne.concatenate_epochs(list(cond_epochs.values()))

    # LCMV filter from the pooled, equalised trials
    rank = mne.compute_rank(eq_epochs, tol=1e-6, tol_kind="relative")

    noise_cov = mne.compute_covariance(eq_epochs, tmin=None, tmax=0, rank=rank)
    data_cov = mne.compute_covariance(eq_epochs, tmin=0, tmax=None, rank=rank)

    fwd = mne.read_forward_solution(fwd_path)

    spatial_filter = mne.beamformer.make_lcmv(
        eq_epochs.info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        rank=rank)

    # evoked per condition to fsaverage
    morph = mne.read_source_morph(morph_path)
    stcs = {cond: morph.apply(mne.beamformer.apply_lcmv(ep.average(), spatial_filter))
            for cond, ep in cond_epochs.items()}

    n_each = len(next(iter(cond_epochs.values())))
    return stcs, n_each


if __name__ == "__main__":
    src_info = f"volume-spacing-{VOL_SPACING}mm"
    cond_a, cond_b = CONTRAST
    diff_name = f"{cond_a}-{cond_b}"

    # ------------------------------------------------------------------
    # Per subject: compute, save, keep the difference
    # ------------------------------------------------------------------
    n_trials = {}
    diffs = []

    for subject in DATES:
        stcs, n_trials[subject] = evoked_contrast_stcs(subject, src_info)
        stcs[diff_name] = stcs[cond_a] - stcs[cond_b]

        out_path = fnames.sub_stcs(subject=subject, src_info=src_info, method="LCMV_EVOKED_CONTRAST")
        with open(out_path, "wb") as f:
            pickle.dump(stcs, f)

        diffs.append(stcs[diff_name])

    # ------------------------------------------------------------------
    # Grand average of the difference at each time point
    # ------------------------------------------------------------------
    grand = diffs[0].copy()
    grand.data = np.mean([stc.data for stc in diffs], axis=0)

    # ------------------------------------------------------------------
    # Group report
    # ------------------------------------------------------------------
    report_path = Path(fnames.group_stcs_report)  # setup_report needs a Path
    report = setup_report(report_path, title="Group Source Time Courses Report")
    section = f"Evoked beamformer {diff_name}"

    report.add_html(f"<p>Subjects (n={len(n_trials)}). Trials per condition per subject: {n_trials}</p>",
                    title="Evoked beamformer | subjects", section=section, replace=True)

    src_to = mne.read_source_spaces(fnames.src_volume(subject=SUBJECT_TO, vol_spacing=VOL_SPACING))

    # window-averaged maps
    for win, (t0, t1) in TIME_WINDOWS.items():
        fig = grand.copy().crop(t0, t1).mean().plot(
            src=src_to,
            subject=SUBJECT_TO,
            subjects_dir=fnames.subjects_dir,
            mode="stat_map",
            show=False,
        )
        report.add_figure(
            fig,
            title=f"Grand average {diff_name} | {win} ({int(t0 * 1000)}-{int(t1 * 1000)} ms)",
            caption=f"Mean over {len(n_trials)} subjects of the window-averaged LCMV difference on {SUBJECT_TO}.",
            section=section,
            replace=True,
        )
        plt.close(fig)

    # one colour scale for all time points (display only): hide |values| below the 95th percentile
    #clim = dict(kind="value", pos_lims=[float(v) for v in np.percentile(np.abs(grand.data), [95, 99, 100])])

    figs, captions = [], []
    for t in grand.times:
        fig = grand.plot(
            src=src_to,
            subject=SUBJECT_TO,
            subjects_dir=fnames.subjects_dir,
            mode="stat_map",
            #clim=clim,
            initial_time=t,   # slices go to the largest |difference| at this time point
            show=False,
        )
        figs.append(fig)
        captions.append(f"{t * 1000:.0f} ms. Mean over {len(n_trials)} subjects on {SUBJECT_TO} ")

    report.add_figure(figs, title=f"Grand average {diff_name}",
                      caption=captions, section=section, replace=True)
    for fig in figs:
        plt.close(fig)

    save_report(report, report_path)