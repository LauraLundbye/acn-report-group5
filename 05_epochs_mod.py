
import mne
import pandas as pd

from params import fnames, EPOCHS_TMIN, EPOCHS_TMAX, BASELINE, DOWNSAMPLE_RATE, EVENT_ID, MIN_DURATION, L_FREQ, H_FREQ, FILT_METHOD, PHOTODIODE_TMAX, EPOCHS_REJECT
from utils.argparser import setup_argparser
from utils.logger import setup_report, save_report
from utils.update_event_id import update_event_id
from utils.photodiode import adjust_timing_of_events_photodiode, estimate_stimulus_durations, add_stimulus_duration_to_report




if __name__ == "__main__":
    parser = setup_argparser(description="Epoch the data", subject=True)
    args = parser.parse_args()
    subject = args.subject

    # ------------------------------------------------------------------
    # Open preprocessing report
    # ------------------------------------------------------------------
    report_path = fnames.preproc_report(subject=subject)
    report = setup_report(report_path, title=f"Preprocessing Report | {subject} ")

    # ------------------------------------------------------------------
    # Load the ICA preprocessed data
    # ------------------------------------------------------------------

    filtered_ica_path = fnames.sub_filtered_ica(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)

    filtered_ica_data = mne.io.read_raw_fif(filtered_ica_path)

    events = mne.find_events(filtered_ica_data, stim_channel="STI101", min_duration=MIN_DURATION)

    event_id = update_event_id(events, EVENT_ID)

    # ------------------------------------------------------------------
    # Adjusting events based on photodiode information
    # ------------------------------------------------------------------
    pd_tmax = PHOTODIODE_TMAX.get(subject)

    if pd_tmax is None:
        pd_raw = filtered_ica_data
        events_pd = events
    else:
        # photodiode fell off -> estimate the (constant) delay only from the part where it was attached
        pd_raw = filtered_ica_data.copy().crop(tmax=pd_tmax)
        events_pd = events[events[:, 0] <= pd_raw.last_samp]

    events_pd_adjusted = adjust_timing_of_events_photodiode(pd_raw, events_pd, event_id, report=report, stim_trigs=[1, 3])

    # apply the same shift to all events
    shift = events_pd_adjusted[0, 0] - events_pd[0, 0]
    events_adjusted = events.copy()
    events_adjusted[:, 0] += shift

    # Sanity check: stimulus durations from photodiode
    lengths, stim_begin, stim_end = estimate_stimulus_durations(pd_raw, events_adjusted, stim_trigs=(1, 3))
    add_stimulus_duration_to_report(report, lengths)

    epochs = mne.Epochs(
        filtered_ica_data,
        events = events_adjusted, 
        event_id = event_id,
        tmin = EPOCHS_TMIN, 
        tmax = EPOCHS_TMAX, 
        baseline = BASELINE,
        reject = None,
        preload = True
    )

    # ------------------------------------------------------------------
    # Downsample epochs
    # ------------------------------------------------------------------

    epochs.resample(DOWNSAMPLE_RATE)

    report.add_events(events, title="Events", event_id = event_id, sfreq=filtered_ica_data.info["sfreq"]) 
    #report.add_epochs(epochs, title = "Downsampled Epochs")

    # ------------------------------------------------------------------
    # Add behavioural data to epochs
    # ------------------------------------------------------------------


    behav_path = fnames.behav_sub(subject=subject)
    behav_data = pd.read_csv(behav_path)

    behav_data_expanded = behav_data.loc[behav_data.index.repeat(4)].reset_index(drop=True)

    behav_data_expanded["epoch_type"] = [
        epoch_type
        for _ in range(len(behav_data))
        for epoch_type in ["target", "mask", "response", "PAS_rating"]
        ]
    
    epochs.metadata = behav_data_expanded
    epochs.drop_bad(reject=EPOCHS_REJECT)
    report.add_figure(epochs.plot_drop_log(show=False), title="Drop log", replace=True)


    report.add_epochs(epochs, title = "Epochs with Behavioural Data")
 
    # ------------------------------------------------------------------
    # Save epochs. Remember to include relevant parameters in file name
    # ------------------------------------------------------------------

    epoch_path = fnames.sub_filtered_ica_epochs(subject = subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
    epochs.save(epoch_path, overwrite = True)


    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    save_report(report, report_path)