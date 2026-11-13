"""
This script illustrates how to prepare evoked-response data for statistical analysis.

For each participant, it loads the preprocessed epochs, applies the rejection criteria defined in EVOKED_REJECT, and computes an average evoked response for each condition specified in EVOKED_CONDS. The resulting evoked responses are saved for use in a separate group-level analysis script (08_erf_stats.py).

For decoding analyses, you can follow a similar approach: run the decoding separately for each participant and save the resulting classification accuracies (or other metrics of interest). These participant-level results can then be used for group-level statistical testing.
"""
import mne, numpy as np
from utils.argparser import setup_argparser
from params import (
    fnames, L_FREQ, H_FREQ, FILT_METHOD,
    EVOKED_CONDS, EVENT_ID, CONTRAST
)


if __name__ == "__main__":
    parser = setup_argparser(description="Compute evoked responses for participants to prepare for stats", subject=True)
    args = parser.parse_args()
    subject = args.subject

    epochs_filtered_ica_path = fnames.sub_filtered_ica_epochs(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
    epochs = mne.read_epochs(epochs_filtered_ica_path)

    stim_codes = [EVENT_ID["stimulus/0"], EVENT_ID["stimulus/1"]]   # 1 and 3
    stim_epochs = epochs[np.isin(epochs.events[:, 2], stim_codes)].copy()

    cond_epochs = {cond: stim_epochs[f"subjective_response in {resps}"] for cond, resps in CONTRAST.items()}
    mne.epochs.equalize_epoch_counts(list(cond_epochs.values()), method="mintime")
    eq_epochs = mne.concatenate_epochs(list(cond_epochs.values()))

    for cond, epochs_cond in cond_epochs.items():
        epochs_cond = cond_epochs[cond]
        evoked_cond = epochs_cond.average()
        outpath_cond = fnames.sub_filtered_ica_evoked(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD, cond=cond.replace("/", "-"))
        evoked_cond.save(outpath_cond, overwrite=True)
        