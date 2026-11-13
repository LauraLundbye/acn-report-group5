import numpy as np

# stolen from : https://neurowaves.science/6-meg-pipeline-gallery/notebooks/mne/mne_kit_photodiode_pipeline
def rising_edges(x, sf, frac=0.9, refractory_s=0.5):
    lo, hi = np.percentile(x, [1, 99])
    above = x > lo + frac*(hi-lo)
    cand = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    keep = [cand[0]]
    for i in cand[1:]:
        if (i-keep[-1])/sf > refractory_s: keep.append(i)
    return np.asarray(keep, int)


def falling_edges(x, sf, frac=0.9, refractory_s=0.012):
    """
    Shoutout to study group 1 for making this function!

    Detecting the true end of a our stimulus while ignoring high-frequency projector flickers.
    frac=0.9 sets off from 90% brightness.
    refractory_s=0.03 (30ms) ensures it bridges 8ms hardware flickers but stops before the mask.
    """
    
    # Calculating the stable baseline (lo) and the true peak (hi), ignoring extreme glitches
    lo, hi = np.percentile(x, [1, 99])

    # I then create a boolean mask where True means the signal is currently below 
    # the threshold (dark)
    below = x < lo + frac * (hi - lo)

    # I finf every single flicker of projector checking where the current sample is dark 
    # (below[1:]) BUT the previous sample was bright (~below[:-1])
    cand = np.flatnonzero(below[1:] & ~below[:-1]) + 1

    # Initialize the empty list to store my true trigger endings
    keep = []
    
    # Looping through every single candidate EXCEPT the very last one since I am
    # comparing with next candidate
    # I use so I can compare the current dip to the NEXT dip in the list
    for i in range(len(cand) - 1):
        
        # Calculating the time gap (in seconds) between the current dip and the next dip
        gap_in_seconds = (cand[i+1] - cand[i]) / sf
        
        # If the gap is LARGER than the refractory period (here chosen so that it may be
        # larger than screen flickers, which happen each 8 ms circa, and smaller than stimulus 
        # presentation, 33 ms), 
        # the screen stayed dark afterward, so the current dip was the true end of the stimulus.
        if gap_in_seconds > refractory_s:
            keep.append(cand[i])
            
    # Edge Case: The absolute last dip in the entire recording must be a true ending
    # because there are no more flickers after it I manually save it.
    if len(cand) > 0:
        keep.append(cand[-1])
        
    # Converting the clean list of true endings into a standard NumPy integer array
    return np.asarray(keep, int)



def adjust_timing_of_events_photodiode(raw, events, event_id, photodiode_ch_name = "MISC002", report=None, stim_trigs = [1, 3]):
    """

    """
    
    photodiode_data = raw.get_data(picks=photodiode_ch_name).squeeze()

    photodiode_trigs = rising_edges(photodiode_data, raw.info["sfreq"])
    
    photodiode_trigs = photodiode_trigs + raw.first_samp

    # find the trigger values for the events with photodiode (the gabor patches)
    events_stim = np.array([event for event in events if event[-1] in stim_trigs])

    # compare event timing from photodiode and triggers
    diff = []

    for event in events_stim:
        samp = event[0]


        # find the nearest matching photodiode event (photodiode should be before the trigger???)
        difference = np.abs(photodiode_trigs - samp)
        idx = difference.argmin()
        diff.append(photodiode_trigs[idx]-samp)
        
    # count each of the occurences 
    unique_diff, counts = np.unique(diff, return_counts=True)

    # find the mode
    mode = unique_diff[np.argmax(counts)]
    if report:

        html = f"""
        <h3>Photodiode timing differences</h3>

        <p>
            <strong>Mode (most common difference):</strong>
            {mode} samples
        </p>

        <table style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr style="background-color: #f2f2f2;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">
                        Difference (samples)
                    </th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;">
                        Count
                    </th>
                </tr>
            </thead>
            <tbody>
        """

        for difference, count in zip(unique_diff, counts):

            # Highlight the mode
            if difference == mode:
                row_style = "background-color: #d9ead3; font-weight: bold;"
            else:
                row_style = ""

            html += f"""
                <tr style="{row_style}">
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">
                        {difference}
                    </td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">
                        {count}
                    </td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        report.add_html(
            html,
            title="Photodiode timing differences",
            replace=True
        )


    adjusted_events = events.copy()
    adjusted_events[:, 0] = adjusted_events[:, 0] + mode

    if report:
        
        plot_duration = 5  # seconds shown in each plot
        n_plots = 10

        recording_duration = raw.n_times / raw.info["sfreq"]

        start_times = np.linspace(
            0,
            recording_duration - plot_duration,
            n_plots
        )

        figures = []

        for start in start_times:

            fig = raw.plot(
                picks=photodiode_ch_name,
                events=adjusted_events,
                start=start,
                duration=5,
                event_id=event_id,
                show=False,
                show_scalebars=False,
                event_color="darkorange",
            )

            figures.append(fig)

        report.add_figure(
            figures,
            title="Triggers after adjusting for delay determined using the photodiode",
            replace=True,
        )

    return adjusted_events


import numpy as np


def estimate_stimulus_durations(
    raw,
    events,
    stim_trigs=(1, 3),
    photodiode_ch_name="MISC002",
    frac=0.9,
    refractory_s=0.012,
):
    """
    Estimate stimulus durations from photodiode onset and offset edges.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object containing the photodiode channel.

    events : ndarray, shape (n_events, 3)
        MNE events array. Events with IDs in `stim_trigs` are treated
        as stimulus onsets.

    stim_trigs : tuple of int
        Event IDs corresponding to stimulus onsets.

    photodiode_ch_name : str
        Name of the photodiode channel.

    frac : float
        Threshold fraction used for detecting photodiode edges.

    refractory_s : float
        Refractory period used for detecting falling edges.

    Returns
    -------
    stimulus_lengths : ndarray
        Stimulus durations in seconds.

    stim_begin : ndarray
        Stimulus onset sample numbers.

    stim_end : ndarray
        Matched stimulus offset sample numbers.
    """

    sfreq = raw.info["sfreq"]

    # --------------------------------------------------------------
    # Get photodiode signal
    # --------------------------------------------------------------
    photodiode_data = raw.get_data(
        picks=photodiode_ch_name
    ).squeeze()

    # --------------------------------------------------------------
    # Detect photodiode onset and offset edges
    # --------------------------------------------------------------
    photodiode_trigs_begin = rising_edges(
        photodiode_data,
        sfreq,
        frac=frac,
    )

    photodiode_trigs_end = falling_edges(
        photodiode_data,
        sfreq,
        frac=frac,
        refractory_s=refractory_s,
    )

    # Convert from indices relative to raw data to MNE sample numbers
    photodiode_trigs_begin += raw.first_samp
    photodiode_trigs_end += raw.first_samp

    # --------------------------------------------------------------
    # Get stimulus events
    # --------------------------------------------------------------
    events_stim = events[
        np.isin(events[:, 2], stim_trigs)
    ]

    stim_begin = events_stim[:, 0]

    # --------------------------------------------------------------
    # Match each stimulus onset to the first photodiode falling
    # edge occurring after that onset
    # --------------------------------------------------------------
    stim_end = []
    stimulus_lengths = []

    for sample in stim_begin:

        valid_endings = photodiode_trigs_end[
            photodiode_trigs_end > sample
        ]

        if len(valid_endings) == 0:
            # No falling edge after this stimulus
            stim_end.append(np.nan)
            stimulus_lengths.append(np.nan)
            continue

        associated_ending = valid_endings[0]

        stim_end.append(associated_ending)

        duration = (associated_ending - sample) / sfreq
        stimulus_lengths.append(duration)

    return (
        np.asarray(stimulus_lengths, dtype=float),
        np.asarray(stim_begin),
        np.asarray(stim_end, dtype=float),
    )


def add_stimulus_duration_to_report(
    report,
    stimulus_lengths,
    title="Stimulus durations",
):
    """
    Add a summary of photodiode-derived stimulus durations to an MNE Report.
    """

    valid_lengths = stimulus_lengths[
        np.isfinite(stimulus_lengths)
    ]

    if len(valid_lengths) == 0:
        html = """
        <p><strong>No valid stimulus durations were found.</strong></p>
        """

        report.add_html(
            html,
            title=title,
            replace=True,
        )
        return

    unique_lengths, counts = np.unique(
        valid_lengths,
        return_counts=True
    )

    mean_duration = np.mean(valid_lengths)
    median_duration = np.median(valid_lengths)
    min_duration = np.min(valid_lengths)
    max_duration = np.max(valid_lengths)

    html = f"""
    <h3>Photodiode-derived stimulus durations</h3>

    <p>
        <strong>Number of stimuli:</strong> {len(valid_lengths)}
    </p>

    <p>
        <strong>Mean:</strong> {mean_duration * 1000:.2f} ms<br>
        <strong>Median:</strong> {median_duration * 1000:.2f} ms<br>
        <strong>Minimum:</strong> {min_duration * 1000:.2f} ms<br>
        <strong>Maximum:</strong> {max_duration * 1000:.2f} ms
    </p>

    <h4>Duration distribution</h4>

    <table style="border-collapse: collapse; width: 100%;">
        <thead>
            <tr style="background-color: #f2f2f2;">
                <th style="border: 1px solid #ddd; padding: 8px;">
                    Duration (ms)
                </th>
                <th style="border: 1px solid #ddd; padding: 8px;">
                    Count
                </th>
            </tr>
        </thead>
        <tbody>
    """

    for duration, count in zip(unique_lengths, counts):

        html += f"""
            <tr>
                <td style="border: 1px solid #ddd; padding: 8px;
                           text-align: center;">
                    {duration * 1000:.2f}
                </td>
                <td style="border: 1px solid #ddd; padding: 8px;
                           text-align: center;">
                    {count}
                </td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """

    report.add_html(
        html,
        title=title,
        replace=True,
    )