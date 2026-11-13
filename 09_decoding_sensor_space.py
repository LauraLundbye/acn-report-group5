# Load in packages and setup
import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# importing from our local utils
import sys
sys.path.append("../scripts/")

# import from params
from params import (L_FREQ, H_FREQ, FILT_METHOD, CONTRAST, DATES, fnames)

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


# Set up clf and other variables
ncv=10
scaler = StandardScaler()
kfold = StratifiedKFold(ncv, shuffle=True)
#TODO: Resample classes -undersampling scikitlearn
clf = LogisticRegression(
    class_weight = "balanced", 
    random_state=99,
    C=1.0, # consider grid search for parameters
    max_iter=1000
)


if __name__ == "__main__":

    # Initiate Empty list for data-storage
    subject_data = []

    #Path settings
    FIG_DIR = Path(fnames.ucloud_memberfiles) / "figures"
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    # Start loop over each subject
    for subject in DATES:

        #Load and prepare data
        epochs_path = fnames.sub_filtered_ica_epochs(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
        epochs = mne.read_epochs(epochs_path)
        epochs = epochs.drop_channels(epochs.info["bads"])
        epochs.pick("mag") # for speed only use magnetometers in this notebook

        MEG_data = epochs.get_data()
        contrast = epochs.metadata["target_contrast"].to_numpy()

        # for the decoder we combine pas-scores 1 together, and 2 + 3 + 4 together
        epochs.metadata["subjective_combined"] = np.where(
        epochs.metadata["subjective_response"].isin([1]),
        0,
        1
        )

        pas_score = epochs.metadata["subjective_combined"]
        n_sensors = MEG_data.shape[1]

        # Decoding across timepoints (from regress_out... notebook)
        scores = np.empty((ncv, MEG_data.shape[-1]))

        pipeline_regress_out_no_print = Pipeline([
            ("regress_out_contrast", RegressOutContrast(n_sensors=n_sensors)),
            ("scaler", scaler),
            ("clf", clf)
        ])

        for t in tqdm(range(MEG_data.shape[-1])): # loop over time points
            X_t = MEG_data[:, :, t]
            
            X_with_contrast = np.column_stack([
                X_t,
                contrast
            ])
            scores_t = cross_val_score(
                pipeline_regress_out_no_print, 
                X_with_contrast, 
                pas_score, 
                cv=kfold
                )
                
            scores[:, t] = scores_t
        
        subject_data.append(scores)

    # Plotting results
    all_scores = np.array(subject_data)
    subject_means = all_scores.mean(axis=1)
    grand_mean = subject_means.mean(axis=0)

    ## If wanting to add between-subject spread around main-line
    # sem = standard error of mean (or standard error of measurement)
    sem = subject_means.std(axis=0, ddof=1) / np.sqrt(len(subject_means))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

    # Plot each subject's average across CV folds
    for subj_mean in subject_means:
        ax.plot(
            epochs.times,
            subj_mean * 100,
            alpha=0.15,        # increased from notebook, due to fewer lines displayed
            linewidth=1,
            color="darkgreen"
        )

    # Plot the average across subjects
    ax.plot(
        epochs.times,
        grand_mean * 100,
        linewidth=2,
        color="darkgreen",
        label="Decoding accuracy\naveraged across subjects"
    )

    # If wanting to add between-subject spread around main-line
    ax.fill_between(
        epochs.times, 
        (grand_mean - sem)*100, 
        (grand_mean + sem)*100, 
        color="darkgreen", 
        alpha=0.2, 
        linewidth=0)

    # Aesthetics
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Time (s)")
    ax.axvline(0, color="grey", linestyle="--", label="Stimulus presentation")
    ax.axhline(50, color="grey", linestyle=":", linewidth=1, label="Chance level")

    # remove top and right spines
    for spine_label in ["top", "right"]:
        ax.spines[spine_label].set_visible(False)

    # add legend
    ax.legend(frameon=False)
    ax.set_title("Sensor space awareness decoding accuracy — contrast effect regressed out")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "timepoint_decoding_accuracy.png", dpi=200)