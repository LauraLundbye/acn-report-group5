from pathlib import Path
from utils.fnames import FileNames


# ------------------------------------------------------------------
# PREPROCESSING PARAMETERS
# ------------------------------------------------------------------

## list of bad channels for each subject
BAD_CHANNELS = {
    "0188": ["MEG0541", "MEG0922"],
    "0189": ["MEG2132", "MEG2142"],
    "0190": ["MEG0922"],
    "0191": [],
    "0192": [],
    "0193": ["MEG1933"],
    "0194": [],
    "0195": ["MEG0722"]
}


DATES = {
    "0188": "20260908_000000",
    "0189": "20260908_000000",
    "0190": "20260909_000000",
    "0191": "20260909_000000",
    "0192": "20260909_000000",
    "0193": "20260910_000000",
    "0194": "20260910_000000",
    "0195": "20260910_000000"
}

## Filtering
L_FREQ = 0.1
H_FREQ = 40
ICA_L_FREQ = 1
FILT_METHOD = "fir"

## Epoching
EPOCHS_TMIN = -0.2
EPOCHS_TMAX = 0.7
BASELINE = (None, 0) # from tmin to time of stimulation
DOWNSAMPLE_RATE = 250
MIN_DURATION = 0.002
EPOCHS_REJECT = dict(mag=4e-12, grad=4000e-13)


EVENT_ID = {
    'stimulus/0': 1,
    'stimulus/1': 3,
    'mask': 4,
    'response/stimulus/0': 6,
    'response/stimulus/1' :  8,
    'response/PAS/1' : 10,
    'response/PAS/2': 12,
    'response/PAS/3': 14,
    'response/PAS/4': 16,
    'response/auto': 32,
    'response/PAS/auto': 64}

## photodiode fell off during the recording for subject 0188.
PHOTODIODE_TMAX = {
    "0188": 769.0,
}

# ICA
N_ICA_COMPONENTS = 0.99
ICA_METHOD = "fastica"
RANDOM_STATE = 42
ICA_DECIM = 10

## EVOKEDS
EVOKED_CONDS = ["subjective_response == 1", "subjective_response in [2, 3, 4]"]
CONTRAST = {"PAS1": [1], "PAS234": [2, 3, 4]}

#EVOKED_REJECT
VAN_EVOKED_TMIN = 0.132
VAN_EVOKED_TMAX = 0.320
P3A_EVOKED_TMIN = 0.324
P3A_EVOKED_TMAX = 0.512
VAN_EVOKED_SENSORS = ["MEG1641", "MEG1721","MEG1711","MEG1741","MEG1731","MEG1941","MEG1911","MEG1921","MEG1931","MEG2141","MEG2041","MEG2111","MEG2121","MEG2031","MEG2341","MEG2331","MEG2131","MEG2311","MEG2321","MEG2511","MEG2541","MEG2431","MEG2521","MEG2531"]
P3A_EVOKED_SENSORS = ["MEG0121","MEG0341","MEG0311","MEG0321","MEG0511","MEG0541", "MEG0331","MEG0521","MEG0531","MEG0611","MEG0641","MEG0811","MEG0821","MEG1011","MEG0621","MEG0911","MEG0941","MEG1021","MEG1031","MEG0921","MEG0931","MEG1241","MEG1211","MEG1231","MEG1221","MEG1411"]

SUBJECTIVE_RESPONSES = [1, 2, 3, 4]   # PAS ratings in epochs.metadata["subjective_response"]
TIME_WINDOWS = {"T1": (VAN_EVOKED_TMIN, VAN_EVOKED_TMAX),"T2": (P3A_EVOKED_TMIN, P3A_EVOKED_TMAX),}

# ------------------------------------------------------------------
# SOURCE MODELLING
# ------------------------------------------------------------------
SRC_SPACING = "oct6"
VOL_SPACING = 10
SUBJECT_TO = "fsaverage"
MINIMUM_NORM_METHOD ="MNE"


# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
# determining the user to find a nice place to save derivatives 
p = Path(__file__)

"""
uCloud_user = next(
    part for part in p.parts
    if "#" in part and part.split("#")[1].isdigit()
)
"""
# TODO fix
uCloud_user = "group5"

fnames = FileNames()
fnames.add('root', '/work')
fnames.add('raw', '{root}/raw')


fnames.add('subjects_dir', '{root}/freesurfer')
fnames.add('ucloud_memberfiles', '{root}'+f'/{uCloud_user}')
fnames.add('derivatives', '{ucloud_memberfiles}/derivatives')
fnames.add('meg_derivatives', '{derivatives}/MEG')

fnames.add('behav', '{root}/behav')
fnames.add('behav_sub', '{behav}/{subject}_experiment_data.csv')

fnames.add('sub_meg_derivatives', '{meg_derivatives}/{subject}')
fnames.add('sub_raw', '{raw}/{subject}/{date}/workshop_2026_raw.fif')

# subject level derivatives
fnames.add('sub_filtered', '{sub_meg_derivatives}/{subject}_filtered_{l_freq}-{h_freq}_{method}_meg.fif')
fnames.add('sub_filtered_ica', '{sub_meg_derivatives}/{subject}_filtered_{l_freq}-{h_freq}_{method}_ICA_meg.fif')
fnames.add('sub_ica', '{sub_meg_derivatives}/{subject}_ica.fif')
fnames.add('sub_filtered_ica_epochs', '{sub_meg_derivatives}/{subject}_filtered_{l_freq}-{h_freq}_{method}_ICA-epo.fif')
fnames.add('sub_stcs', '{sub_meg_derivatives}/{subject}_{src_info}_{method}_stcs.pkl')
fnames.add('sub_stcs_morphed', '{sub_meg_derivatives}/{subject}_{src_info}_{method}_{subject_to}_stcs.pkl')
fnames.add('sub_filtered_ica_evoked', '{sub_meg_derivatives}/{subject}_filtered_{l_freq}-{h_freq}_{method}_ICA-{cond}-ave.fif')
fnames.add('sub_stcs_noeq', '{sub_meg_derivatives}/{subject}_{src_info}_{method}_{subject_to}_stcs.pkl')

# reports
fnames.add('reports', '{derivatives}/reports')
fnames.add('anatomy_report', '{reports}/{subject}/{subject}_anatomy_report.h5')
fnames.add('preproc_report', '{reports}/{subject}/{subject}_preproc_report.h5')
fnames.add('stcs_report', '{reports}/{subject}/{subject}_stcs_report.h5')
fnames.add('group_stcs_report', '{reports}/group/group_stcs_report.h5')
fnames.add('erf_report', '{reports}/group/group_erf_report.h5')
fnames.add('stcs_report_method', '{reports}/{subject}/{subject}_{method}_stcs_report.h5')

# SOURCE SPACE 
fnames.add('fwd', '{root}/fwd')
fnames.add('fwd_sub', '{fwd}/{subject}')

# trans
fnames.add('head_mri_t', '{subjects_dir}/trans/{subject}_head_mri-trans.fif')


# bem
fnames.add('bem_model', '{fwd_sub}/{subject}_bem.fif')
fnames.add('bem_sol', '{fwd_sub}/{subject}_bem_solution.fif')

# source spaces

fnames.add('src_surface', '{fwd_sub}/{subject}_surface-spacing-{surf_spacing}-src.fif')
fnames.add('src_volume', '{fwd_sub}/{subject}_volume-spacing-{vol_spacing}mm-src.fif')
fnames.add('src_combined', '{fwd_sub}/{subject}_combined-surface-spacing-{surf_spacing}-volume-spacing-{vol_spacing}mm-src.fif')


# forward model 
fnames.add('forward_model', '{fwd_sub}/{subject}_{src_info}-fwd.fif')

# morph
fnames.add('morph', '{fwd_sub}/{subject}_{src_info}_to_{subject_to}-morph.h5')