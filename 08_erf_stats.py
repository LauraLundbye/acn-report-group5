import mne
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pathlib import Path
from params import (
    fnames, L_FREQ, H_FREQ, FILT_METHOD, DATES, 
    EVOKED_CONDS, 
    VAN_EVOKED_TMIN, VAN_EVOKED_TMAX,   
    VAN_EVOKED_SENSORS, P3A_EVOKED_SENSORS, P3A_EVOKED_TMIN, P3A_EVOKED_TMAX   
)
from utils.logger import setup_report, save_report


def stats(window_means, conditions):
    if len(conditions) != 2:
        raise ValueError("A paired t-test requires exactly two conditions.")

    data1 = np.asarray(window_means[conditions[0]], dtype=float)
    data2 = np.asarray(window_means[conditions[1]], dtype=float)

    if len(data1) != len(data2):
        raise ValueError("Conditions must contain the same number of participants.")

    if not (np.isfinite(data1).all() and np.isfinite(data2).all()):
        raise ValueError("Data contain NaN or infinite values.")

    t_stat, p_value = ttest_rel(data1, data2)

    print(f"Paired t-test: {conditions[0]} vs {conditions[1]}")
    print(f"t({len(data1) - 1}) = {t_stat:.3f}, p = {p_value:.5f}")
    print(f"Mean difference = {(data1 - data2).mean():.6g}")

    return t_stat, p_value

def plot_results(evokeds, picks, region, highlighted_windows, savepath=None):

    colours = ["steelblue", "goldenrod"]
    conditions = list(evokeds.keys())
    condition_labels = dict(zip(conditions, ("Unaware", "Aware")))
    fig, ax = plt.subplots(figsize=(8,5), dpi=300)

    max_abs_amplitude = 0.0
    for scale_picks in (VAN_EVOKED_SENSORS, P3A_EVOKED_SENSORS):
        for cond_evokeds in evokeds.values():
            data = np.stack([
                evoked.get_data(picks=scale_picks)
                for evoked in cond_evokeds
            ])
            participant_evokeds = data.mean(axis=1)
            grandaverage_evoked = participant_evokeds.mean(axis=0)
            if len(participant_evokeds) > 1:
                sem = participant_evokeds.std(axis=0, ddof=1) / np.sqrt(
                    len(participant_evokeds)
                )
            else:
                sem = np.zeros_like(grandaverage_evoked)
            max_abs_amplitude = max(
                max_abs_amplitude,
                np.max(np.abs(grandaverage_evoked) + sem),
            )
    y_limit = max_abs_amplitude * 1.05

    window_colours = ["#d9e2ec", "#f4d7cf"]
    for (window_label, tmin, tmax), colour in zip(highlighted_windows, window_colours):
        ax.axvspan(tmin, tmax, color=colour, alpha=0.65, label=window_label, zorder=0)

    # Plot grand-average ERFs
    for i, (cond, cond_evokeds) in enumerate(evokeds.items()):
        times = cond_evokeds[0].times

        data = np.stack([
            evoked.get_data(picks=picks)
            for evoked in cond_evokeds
        ])
        # Shape: participants x sensors x time
        participant_evokeds = data.mean(axis=1)
        grandaverage_evoked = participant_evokeds.mean(axis=0)
        if len(participant_evokeds) > 1:
            sem = participant_evokeds.std(axis=0, ddof=1) / np.sqrt(
                len(participant_evokeds)
            )
        else:
            sem = np.zeros_like(grandaverage_evoked)

        ax.fill_between(
            times,
            grandaverage_evoked - sem,
            grandaverage_evoked + sem,
            color=colours[i % len(colours)],
            alpha=0.2,
            linewidth=0,
        )
        ax.plot(
            times,
            grandaverage_evoked,
            label=condition_labels[cond],
            color=colours[i % len(colours)]
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set(
        xlabel="Time (s)",
        ylabel="Amplitude",
        title=f"Grand-average ERF ({region})"
    )
    ax.set_ylim(-y_limit, y_limit)
    ax.legend()

    # Create a small inset in the lower-right corner
    ax_inset = inset_axes(
        ax,
        width="20%",
        height="20%",
        loc="lower right",
        borderpad=1,
    )

    info = evokeds[conditions[0]][0].info.copy()
    info["bads"] = []  # Don't mark bad channels in red

    # Plot all magnetometers
    mne.viz.plot_sensors(
        info,
        kind="topomap",
        ch_type="mag",
        axes=ax_inset,
        show_names=False,
        pointsize=0.2,
        show=False,
    )

    # Overlay selected sensors
    selected_info = mne.pick_info(
        info,
        mne.pick_channels(info["ch_names"], include=picks),
    )

    mne.viz.plot_sensors(
        selected_info,
        kind="topomap",
        ch_type="mag",
        axes=ax_inset,
        show_names=False,
        pointsize=10,
        show=False,
    )

    ax_inset.set_title("Sensors", fontsize=8)
        
    # Aesthetics: remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()

    return fig




if __name__ == "__main__":

    FIG_DIR = Path(fnames.ucloud_memberfiles) / "figures"
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # Open a report
    # ------------------------------------------------------------------
    report_path = Path(fnames.erf_report)
    report = setup_report(report_path, title=f"Group ERF Report")

    evokeds = {cond: [] for cond in EVOKED_CONDS}
    VAN_occ_window_means = {cond: [] for cond in EVOKED_CONDS}
    VAN_fro_window_means = {cond: [] for cond in EVOKED_CONDS}

    P3A_occ_window_means = {cond: [] for cond in EVOKED_CONDS}
    P3A_fro_window_means = {cond: [] for cond in EVOKED_CONDS}

    for subject in DATES.keys():
        for cond in EVOKED_CONDS:
            evoked_path = fnames.sub_filtered_ica_evoked(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD, cond=cond.replace("/", "-"))
            evoked = mne.read_evokeds(evoked_path)
            evokeds[cond].extend(evoked)

            VAN_occ_data = evoked[0].get_data(picks=VAN_EVOKED_SENSORS, tmin=VAN_EVOKED_TMIN, tmax=VAN_EVOKED_TMAX)
            VAN_fro_data = evoked[0].get_data(picks=P3A_EVOKED_SENSORS, tmin=VAN_EVOKED_TMIN, tmax=VAN_EVOKED_TMAX)

            P3A_occ_data = evoked[0].get_data(picks=VAN_EVOKED_SENSORS, tmin=P3A_EVOKED_TMIN, tmax=P3A_EVOKED_TMAX)
            P3A_fro_data = evoked[0].get_data(picks=P3A_EVOKED_SENSORS, tmin=P3A_EVOKED_TMIN, tmax=P3A_EVOKED_TMAX)

            # Mean across sensors and time points
            VAN_occ_mean_amplitude = VAN_occ_data.mean()
            VAN_fro_mean_amplitude = VAN_fro_data.mean()

            P3A_occ_mean_amplitude = P3A_occ_data.mean()
            P3A_fro_mean_amplitude = P3A_fro_data.mean()


            VAN_occ_window_means[cond].append(VAN_occ_mean_amplitude)
            VAN_fro_window_means[cond].append(VAN_fro_mean_amplitude)

            P3A_occ_window_means[cond].append(P3A_occ_mean_amplitude)
            P3A_fro_window_means[cond].append(P3A_fro_mean_amplitude)

    VAN_occ_group_means = {cond: np.mean(VAN_occ_window_means[cond]) for cond in EVOKED_CONDS}
    VAN_fro_group_means = {cond: np.mean(VAN_fro_window_means[cond]) for cond in EVOKED_CONDS}
    P3A_occ_group_means = {cond: np.mean(P3A_occ_window_means[cond]) for cond in EVOKED_CONDS}
    P3A_fro_group_means = {cond: np.mean(P3A_fro_window_means[cond]) for cond in EVOKED_CONDS}

    VAN_occ_t, VAN_occ_p = stats(VAN_occ_window_means, EVOKED_CONDS)
    VAN_fro_t, VAN_fro_p = stats(VAN_fro_window_means, EVOKED_CONDS)

    VAN_occ_stats = pd.DataFrame({
    "Time Window": "VAN",
    "Region": "Occipital",
    "t": [VAN_occ_t],
    "p": [VAN_occ_p]
    })

    VAN_fro_stats = pd.DataFrame({
    "Time Window": "VAN",
    "Region": "Frontal",
    "t": [VAN_fro_t],
    "p": [VAN_fro_p]
    })
    
    P3A_occ_t, P3A_occ_p = stats(P3A_occ_window_means, EVOKED_CONDS)
    P3A_fro_t, P3A_fro_p = stats(P3A_fro_window_means, EVOKED_CONDS)

    P3A_occ_stats = pd.DataFrame({
    "Time Window": "P3A",
    "Region": "Occipital",
    "t": [P3A_occ_t],
    "p": [P3A_occ_p]
    })

    P3A_fro_stats = pd.DataFrame({
    "Time Window": "P3A",
    "Region": "Frontal",
    "t": [P3A_fro_t],
    "p": [P3A_fro_p]
    })

    section = "Group-Level ERF Plots"
    highlighted_windows = [
        ("Early window", VAN_EVOKED_TMIN, VAN_EVOKED_TMAX),
        ("Late window", P3A_EVOKED_TMIN, P3A_EVOKED_TMAX),
    ]
    fig_occipital = plot_results(
        evokeds, VAN_EVOKED_SENSORS, "Occipital", highlighted_windows
    )
    fig_occipital.savefig(FIG_DIR / "ERF_occipital.png", dpi=200)
    report.add_figure(fig_occipital, title="Group-Level ERF, Occipital Region",
                      caption="Grand-average occipital ERFs with the early and late time windows highlighted.",
                      section=section, replace=True)
    plt.close(fig_occipital)

    fig_frontal = plot_results(
        evokeds, P3A_EVOKED_SENSORS, "Frontal", highlighted_windows
    )
    fig_frontal.savefig(FIG_DIR / "ERF_frontal.png", dpi=200)
    report.add_figure(fig_frontal, title="Group-Level ERF, Frontal Region",
                      caption="Grand-average frontal ERFs with the early and late time windows highlighted.",
                      section=section, replace=True)
    plt.close(fig_frontal)

    sectiontopo = "Topographic Maps"
    ga = {cond: mne.grand_average(evokeds[cond]) for cond in EVOKED_CONDS}
    evoked_contr = mne.combine_evoked([ga[EVOKED_CONDS[0]], ga[EVOKED_CONDS[1]]], weights=[1, -1],)
    times = [0.096, 0.18, 0.276, 0.3, 0.348, 0.436, 0.576]
    fig_topo = evoked_contr.plot_topomap(times,ch_type="mag")
    report.add_figure(fig_topo, title="Topographic Maps of Grand Average Difference Between Conditions",
                      caption="",
                      section=sectiontopo, replace=True)
    plt.close(fig_topo)

    fig_topo_unaw = ga[EVOKED_CONDS[0]].plot_topomap(times, ch_type="mag")
    report.add_figure(fig_topo_unaw, title="Topographic Maps of Unaware Condition",
                      caption="",
                      section=sectiontopo, replace=True)
    plt.close(fig_topo_unaw)

    fig_topo_aw = ga[EVOKED_CONDS[1]].plot_topomap(times, ch_type="mag")
    report.add_figure(fig_topo_aw, title="Topographic Maps of Aware Condition",
                      caption="",
                      section=sectiontopo, replace=True)
    plt.close(fig_topo_aw)

    time_windows = [(VAN_EVOKED_TMIN, VAN_EVOKED_TMAX), (P3A_EVOKED_TMIN, P3A_EVOKED_TMAX)]
    window_colours = ["#d9e2ec", "#f4d7cf"]

    for cond in EVOKED_CONDS:
        fig_ga = ga[cond].plot(picks="mag", gfp=True, show=False)
        for ax in fig_ga.axes:
            for (tmin, tmax), colour in zip(time_windows, window_colours):
                ax.axvspan(tmin, tmax, color=colour, alpha=0.5, zorder=0)
        fig_ga.suptitle(f"Grand average: {cond}")
        report.add_figure(
            fig_ga,
            title=f"Grand Average Waveforms ({cond})",
            caption="",
            section=section,
            replace=True,
    )
    plt.close(fig_ga)

    report.add_html(VAN_occ_stats.to_html(index = False), title = "Early Occipital T-Test Results", replace = "true")
    report.add_html(VAN_fro_stats.to_html(index = False), title = "Early Frontal T-Test Results", replace = "true")

    report.add_html(P3A_occ_stats.to_html(index = False), title = "Late Occipital T-Test Results", replace = "true")
    report.add_html(P3A_fro_stats.to_html(index = False), title = "Late Frontal T-Test Results", replace = "true")

    save_report(report, report_path)