import mne

from params import fnames, L_FREQ, H_FREQ, ICA_L_FREQ, FILT_METHOD, BAD_CHANNELS, DATES
from utils.argparser import setup_argparser
from utils.logger import setup_report, save_report




if __name__ == "__main__":
    parser = setup_argparser(description="Apply filter to raw data", subject=True)
    args = parser.parse_args()
    subject = args.subject


    # ------------------------------------------------------------------
    # Get subject-specific information
    # ------------------------------------------------------------------
    bad_chs_sub = BAD_CHANNELS.get(subject, [])
    date = DATES.get(subject, None)
    
    if bad_chs_sub == []:
        print(f"No bad channels defined for {subject}")

    if date is None:
        raise ValueError(f"No recording date defined for {subject}. Make sure to add it to the DATES dictionary in the params.py file")

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    raw_path = fnames.sub_raw(subject=subject, date=date)
    raw = mne.io.read_raw_fif(raw_path, preload=True)
    
    raw.info["bads"] = bad_chs_sub

    # ------------------------------------------------------------------
    # Create preprocessing report
    # ------------------------------------------------------------------

    report_path = fnames.preproc_report(subject=subject)
    report = setup_report(report_path, title=f"Preprocessing Report | {subject} ")

    # add raw data to the report with out any preprocessing
    report.add_raw(
        raw, 
        title="Raw data", 
        psd=True,  # PSD useful for identifying noisy/bad channels
        butterfly=20, # show 20 segments 
        replace=True
    )
    
    # ------------------------------------------------------------------
    # Apply standard bandpass filter
    # ------------------------------------------------------------------
    raw_filtered = raw.copy()
    raw_filtered.filter(L_FREQ, H_FREQ, method=FILT_METHOD)

    # save the filtered data
    filtered_path = fnames.sub_filtered(subject=subject, l_freq=L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    raw_filtered.save(filtered_path, overwrite=True)

    report.add_raw(
        raw_filtered, 
        title=f"Filtered data ({L_FREQ}-{H_FREQ} Hz)", 
        psd=True, 
        butterfly=20, 
        replace=True
    )
    
    del raw_filtered

    # ------------------------------------------------------------------
    # Filter for ICA (higher high-pass cutoff, to remove slow drifts that can interfere with ICA
    # ------------------------------------------------------------------
    raw_ica = raw.copy()
    del raw
    raw_ica.filter(ICA_L_FREQ, H_FREQ, method=FILT_METHOD)

    # save the filtered data
    ica_path = fnames.sub_filtered_ica(subject=subject, l_freq=ICA_L_FREQ, h_freq=H_FREQ, method=FILT_METHOD)
    ica_path.parent.mkdir(parents=True, exist_ok=True)
    raw_ica.save(ica_path, overwrite=True)

    report.add_raw(
        raw_ica, 
        title=f"Filtered ICA data ({ICA_L_FREQ}-{H_FREQ} Hz)", 
        psd=True, 
        butterfly=20, 
        replace=True
    )
    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    save_report(report, report_path)