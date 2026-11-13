# Load in packages and setup
import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import pickle as pkl
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# importing from our local utils
import sys
sys.path.append("../scripts/")

# import from params
from params import (CONTRAST, DATES, fnames, SUBJECT_TO, L_FREQ, H_FREQ, FILT_METHOD, EVENT_ID)

# Set up clf and other variables
ncv=10
scaler = StandardScaler()
kfold = StratifiedKFold(ncv, shuffle=True)
clf = LogisticRegression(
    class_weight = "balanced", 
    random_state=99,
    C=1.0, # consider grid search for parameters
    max_iter=1000
)

# Set up RegressOutContrast class
class RegressOutContrast(BaseEstimator, TransformerMixin):

    def __init__(self, n_sensors):
        self.n_sensors = n_sensors

    def fit(self, X, y=None):

        # Separate MEG and contrast
        meg = X[:, :self.n_sensors]
        contrast = X[:, self.n_sensors]

        # Design matrix
        self.X_design_ = np.column_stack([
            np.ones(len(contrast)),
            contrast
        ])

        # Fit regression using TRAINING data only
        self.beta_ = np.linalg.lstsq(
            self.X_design_,
            meg,
            rcond=None
        )[0]

        return self

    def transform(self, X):

        # Separate MEG and contrast
        meg = X[:, :self.n_sensors]
        contrast = X[:, self.n_sensors]

        # Design matrix for these trials
        X_design = np.column_stack([
            np.ones(len(contrast)),
            contrast
        ])

        # Predicted MEG
        meg_hat = X_design @ self.beta_

        # Remove contrast-related activity
        meg_residuals = meg - meg_hat

        return meg_residuals

# Def func to add back in metadata to the beamforming-noeq... .pkl file
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


def plot_subject_means(subject_data, times, title, save_path=None):
    subject_means = np.array(subject_data).mean(axis=1)   # (n_subjects, n_times)
    grand_mean = subject_means.mean(axis=0)
    sem = subject_means.std(axis=0, ddof=1) / np.sqrt(len(subject_means))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    for subj_mean in subject_means:
        ax.plot(times, subj_mean * 100, alpha=0.15, linewidth=1, color="darkgreen")

    ax.plot(times, grand_mean * 100, linewidth=2, color="darkgreen",
            label="Decoding accuracy\naveraged across subjects")
    
    # If wanting to add between-subject spread around main-line
    ax.fill_between(
        times, 
        (grand_mean - sem)*100, 
        (grand_mean + sem)*100, 
        color="darkgreen", 
        alpha=0.2, 
        linewidth=0)

    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_xlabel("Time (s)")
    ax.axvline(0, color="grey", linestyle="--", label="Stimulus presentation")
    ax.axhline(50, color="grey", linestyle=":", linewidth=1, label="Chance level")

    for spine_label in ["top", "right"]:
        ax.spines[spine_label].set_visible(False)

    ax.legend(frameon=False)
    ax.set_title(title)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    plt.show()

if __name__ == "__main__":

    FIG_DIR = Path(fnames.ucloud_memberfiles) / "figures"
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    subject_data_src = []
    times = None

    for subject in DATES:
        stcs_path = fnames.sub_stcs_noeq(subject=subject, src_info="volume-spacing-10mm", method="beamforming-noeq", subject_to=SUBJECT_TO)
        with open(stcs_path, "rb") as f:
            d = pkl.load(f)
        
        stcs = d["stcs"]
        class_labels = np.asarray(d["labels"])
        assert len(stcs) == len(class_labels), f"{subject}: stcs/labels mismatch"

        
        meta = get_trial_metadata(subject)
        assert len(meta) == len(stcs), f"{subject}: {len(meta)} metadata rows vs {len(stcs)} stcs"
        assert (meta["label"].values == class_labels).all(), f"{subject}: label order does not match"

        contrast = meta["target_contrast"].to_numpy()
        assert not np.isnan(contrast).any(), f"{subject}: NaN contrast values"

        y = (class_labels != "PAS1").astype(int)

        X_src = np.stack([s.data for s in stcs]).astype(np.float32)
        n_vertices = X_src.shape[1]

        if times is None:
            times = stcs[0].times
        else:
            assert np.allclose(times, stcs[0].times), f"{subject}: different time axis"
    
        pipeline_source = Pipeline([
            ("regress_out_contrast", RegressOutContrast(n_sensors=n_vertices)),
            ("scaler", scaler),
            ("clf", clf),
        ])

        scores = np.empty((ncv, X_src.shape[-1]))
        for t in tqdm(range(X_src.shape[-1]), desc=subject):
            X_with_contrast = np.column_stack([X_src[:, :, t], contrast])
            scores[:, t] = cross_val_score(
                pipeline_source, X_with_contrast, y, cv=kfold,
                scoring="balanced_accuracy", n_jobs=-1,
            )

        subject_data_src.append(scores)
        del d, stcs, X_src


    plot_subject_means(
        subject_data_src,
        times,
        title="Source space awareness decoding accuracy — contrast effect regressed out",
        save_path=FIG_DIR / "source_decoding_all_vertices.png",
    )