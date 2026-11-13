import mne
from pathlib import Path
import pickle

from params import SRC_SPACING, VOL_SPACING, L_FREQ, H_FREQ, FILT_METHOD, SUBJECT_TO, MINIMUM_NORM_METHOD, fnames
from utils.argparser import setup_argparser
from utils.logger import setup_report, save_report


if __name__ == "__main__":
    parser = setup_argparser(description="Compute source time courses from epochs", subject=True)
    args = parser.parse_args()
    subject = args.subject

    epochs_path = fnames.sub_filtered_ica_epochs(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
        
    # ------------------------------------------------------------------
    # Load epochs, and choose a subset of them (all targets for example as you may run into memory problems otherwise)
    # ------------------------------------------------------------------
    epochs = mne.read_epochs(epochs_path)
    stim_epochs = epochs["stimulus"]

    # ------------------------------------------------------------------
    # Open a report
    # ------------------------------------------------------------------
   
    report_path = fnames.stcs_report(subject=subject)
    report = setup_report(report_path, title=f"Source Time Courses Report | {subject} ")


    # ------------------------------------------------------------------
    # Compute covariance between sensors for whitening and add to report
    # ------------------------------------------------------------------    


    noise_cov = mne.compute_covariance(inst=stim_epochs, tmin=None, tmax=0)

    report.add_covariance(noise_cov, info=stim_epochs.info, title="Covariance between sensors before stimulus onset", replace=True)


    
    paths_src = [
        fnames.src_surface(subject=subject, surf_spacing=SRC_SPACING), 
        fnames.src_volume(subject=subject, vol_spacing=VOL_SPACING), 
        fnames.src_combined(subject=subject, surf_spacing=SRC_SPACING, vol_spacing=VOL_SPACING)
    ]



    for src_path in paths_src:
        src_info = src_path.name.replace("-src.fif", "")
        src_info = src_info.replace(f"{subject}_", "")


        fwd_path = fnames.forward_model(subject=subject, src_info=src_info)
        morph_path = fnames.morph(subject=subject, src_info=src_info, subject_to=SUBJECT_TO)

        # ------------------------------------------------------------------
        # Inverse
        # ------------------------------------------------------------------    
        fwd = mne.read_forward_solution(fwd_path)

        inverse_operator = mne.minimum_norm.make_inverse_operator(stim_epochs.info, fwd, noise_cov)

        # ------------------------------------------------------------------
        # Get source time courses
        # ------------------------------------------------------------------

        stcs = mne.minimum_norm.apply_inverse_epochs(
            stim_epochs, inverse_operator,
            method=MINIMUM_NORM_METHOD,
            lambda2=1.0 / 9.0,
            return_generator=True
        )

        #morph to fsaverage to we can analyse at group level
        
        morph = mne.read_source_morph(morph_path)
        stcs_morphed = [morph.apply(stc) for stc in stcs]

        # ------------------------------------------------------------------
        # Save morphed source time courses
        # ------------------------------------------------------------------ 
        stcs_path = fnames.sub_stcs(subject=subject, src_info=src_info, method=MINIMUM_NORM_METHOD)

        with open(stcs_path, "wb") as f:
            pickle.dump(stcs_morphed, f)


    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------ 
    save_report(report, report_path)