import pickle
import numpy as np
import pandas as pd
import mne

from params import (VOL_SPACING, L_FREQ, H_FREQ, FILT_METHOD, CONTRAST, EVENT_ID, fnames, SUBJECT_TO)
from utils.argparser import setup_argparser


def get_trial_metadata(subject):
    """Rebuild per-trial metadata in the same order as the trials in the stcs pickle.

    Repeats the trial selection from 06_stcs_beamforming.py: stimulus trials,
    split by CONTRAST in dict order, concatenated. No MEG data is loaded.
    """
    epochs_path = fnames.sub_filtered_ica_epochs(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
    epochs = mne.read_epochs(epochs_path, preload=False)
    stim_codes = [EVENT_ID["stimulus/0"], EVENT_ID["stimulus/1"]]   # 1 and 3
    stim = epochs[np.isin(epochs.events[:, 2], stim_codes)]

    parts = []
    for cond, resps in CONTRAST.items():
        ep = stim[f"subjective_response in {resps}"]
        meta = ep.metadata.copy()
        meta["event_code"] = ep.events[:, 2]   # stimulus/0 (1) vs stimulus/1 (3)
        meta["label"] = cond
        parts.append(meta)

    return pd.concat(parts, ignore_index=True)


def load_stcs(subject, method="beamforming-noeq"):
    """Load morphed single-trial stcs and labels saved by 06_stcs_beamforming.py."""
    src_info = f"volume-spacing-{VOL_SPACING}mm"
    stcs_path = fnames.sub_stcs_morphed(subject=subject, src_info=src_info,
                                        method=method, subject_to=SUBJECT_TO)

    with open(stcs_path, "rb") as f:
        data = pickle.load(f)

    return data["stcs"], np.asarray(data["labels"])


def load_subject(subject, method="beamforming-noeq"):
    """Return X (n_trials, n_voxels, n_times), metadata, times and vertices for one subject.

    Checks that the rebuilt metadata lines up trial-by-trial with the saved stcs.
    """
    stcs, labels = load_stcs(subject, method=method)
    meta = get_trial_metadata(subject)

    # alignment checks: fails if the epochs file was regenerated after the stcs were made
    assert len(meta) == len(stcs), f"{subject}: {len(meta)} metadata rows vs {len(stcs)} stcs"
    assert (meta["label"].values == labels).all(), f"{subject}: label order does not match"

    X = np.stack([stc.data for stc in stcs])   # (n_trials, n_voxels, n_times)
    times = stcs[0].times
    vertices = stcs[0].vertices

    return X, meta, times, vertices


if __name__ == "__main__":
    parser = setup_argparser(description="Load single-trial beamformer stcs with per-trial metadata", subject=True)
    args = parser.parse_args()
    subject = args.subject

    X, meta, times, vertices = load_subject(subject, method="beamforming-noeq")

    print(f"Subject {subject}")
    print(f"  X shape (trials, voxels, times): {X.shape}, dtype {X.dtype}")
    print(f"  time range: {times[0]:.3f} to {times[-1]:.3f} s")
    print(f"  trials per label:\n{meta['label'].value_counts().to_string()}")
    print(f"  metadata columns: {list(meta.columns)}")
