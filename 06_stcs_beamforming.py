
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mne

from params import (VOL_SPACING, L_FREQ, H_FREQ, FILT_METHOD, TIME_WINDOWS, CONTRAST, EVENT_ID, fnames, SUBJECT_TO)
from utils.argparser import setup_argparser
from utils.logger import setup_report, save_report

mpl.use("Agg")

if __name__ == "__main__":
    parser = setup_argparser(description="Single-trial beamforming lmcv source time courses for the contrast", subject=True)
    args = parser.parse_args()
    subject = args.subject

    # comment: Beamforming assumes that all sources are not correlated.
    # If two sources are very close to each other, they may cancel each other out.

    # only the volume forward model is used for beamforming.
    
    src_info = f"volume-spacing-{VOL_SPACING}mm"

    epochs_path = fnames.sub_filtered_ica_epochs(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
    fwd_path = fnames.forward_model(subject=subject, src_info=src_info)
    morph_path = fnames.morph(subject=subject, src_info=src_info, subject_to=SUBJECT_TO)

    # ------------------------------------------------------------------
    # Load and equalise trials
    # ------------------------------------------------------------------
    epochs = mne.read_epochs(epochs_path)
    stim_codes = [EVENT_ID["stimulus/0"], EVENT_ID["stimulus/1"]]   # 1 and 3
    stim_epochs = epochs[np.isin(epochs.events[:, 2], stim_codes)].copy().pick("mag") #pick magnetometers only!!!!!

    cond_epochs = {cond: stim_epochs[f"subjective_response in {resps}"] for cond, resps in CONTRAST.items()}
    
    n_before = {cond: len(ep) for cond, ep in cond_epochs.items()}
    #mne.epochs.equalize_epoch_counts(list(cond_epochs.values()), method="mintime") #timing differences between each event list will be minimized
    #n_each = len(next(iter(cond_epochs.values())))

    eq_epochs = mne.concatenate_epochs(list(cond_epochs.values()))
    labels = np.concatenate([[cond] * len(ep) for cond, ep in cond_epochs.items()])

    # MEMORY: free intermediate copies of the epochs
    del epochs, stim_epochs, cond_epochs

    rank = mne.compute_rank(eq_epochs, tol=1e-6, tol_kind="relative") # compute rank 

    # ------------------------------------------------------------------
    # Open a report
    # ------------------------------------------------------------------
    report_path = fnames.stcs_report_method(subject=subject, method="beamforming-noeq")
    report = setup_report(report_path, title=f"Source Time Courses Report | {subject} ")


    # ------------------------------------------------------------------
    # Covariance matrices
    # ------------------------------------------------------------------
    noise_cov = mne.compute_covariance(eq_epochs, tmin=None, tmax=0, rank=rank)
    data_cov = mne.compute_covariance(eq_epochs, tmin=0, tmax=None, rank=rank)
    
    report.add_covariance(noise_cov, info=eq_epochs.info, title="Beamformer Noise covariance", replace=True)
    report.add_covariance(data_cov, info=eq_epochs.info, title="Beamformer Data covariance", replace=True)

   
    
    # ------------------------------------------------------------------
    # Beamformer (LCMV) spatial filter
    # ------------------------------------------------------------------
    
    fwd = mne.read_forward_solution(fwd_path)

    filters = mne.beamformer.make_lcmv(
        eq_epochs.info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        rank=rank)

    # MEMORY: forward model is no longer needed once the filter exists
    del fwd

    # ------------------------------------------------------------------
    # One source time course per trial, saved with its condition label
    # ------------------------------------------------------------------
    stcs = mne.beamformer.apply_lcmv_epochs(eq_epochs, filters, return_generator=True)

    

    #out_path = fnames.sub_stcs(subject=subject, src_info=src_info, method="beamforming")

    #with open(out_path, "wb") as f:
       # pickle.dump({"stcs": stcs, "labels": labels}, f)

    #also save morphed files
    morph = mne.read_source_morph(morph_path)
    stcs_morphed = [morph.apply(stc) for stc in stcs]

    out_path_morphed = fnames.sub_stcs_morphed(subject=subject, src_info=src_info,
                                               method="beamforming-noeq", subject_to=SUBJECT_TO)

    with open(out_path_morphed, "wb") as f:
        pickle.dump({"stcs": stcs_morphed, "labels": labels}, f)


    # ------------------------------------------------------------------
    # Save Report
    # ------------------------------------------------------------------
    save_report(report, report_path) 