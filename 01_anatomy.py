import mne
from pathlib import Path


from params import SRC_SPACING, VOL_SPACING, SUBJECT_TO, fnames
from utils.argparser import setup_argparser
from utils.logger import setup_report, save_report

# HINT FOR params.py
#SRC_SPACING = "oct6"
#VOL_SPACING = 10 #mm
#SUBJECT_TO = "fsaverage"


if __name__ == "__main__":
    parser = setup_argparser(description="Set up BEM and source spaces for a given subject", subject=True)
    args = parser.parse_args()
    subject = args.subject

    # ------------------------------------------------------------------
    # Create anatomy report
    # ------------------------------------------------------------------
    report_path = fnames.anatomy_report(subject=subject)
    report = setup_report(report_path, title=f"Anatomy Report | {subject}")
    
    # ------------------------------------------------------------------
    # Setup boundary element model (BEM)
    # ------------------------------------------------------------------
    bem_file = fnames.bem_model(subject=subject)
    bem_sol_file = fnames.bem_sol(subject=subject)
    
    try:
        model = mne.make_bem_model(subject, subjects_dir=fnames.subjects_dir)
    except(RuntimeError):
        model = mne.make_bem_model(subject, subjects_dir=fnames.subjects_dir, conductivity=(0.3,))
    
    bem_file.parent.mkdir(parents=True, exist_ok=True)

    mne.write_bem_surfaces(bem_file, model, overwrite=True)

    bem_sol = mne.make_bem_solution(model, verbose=None)
    mne.write_bem_solution(bem_sol_file, bem_sol, overwrite=True, verbose=None)

    report.add_bem(subject=subject, title="BEM surfaces", subjects_dir=fnames.subjects_dir, replace=True)

    # ------------------------------------------------------------------
    # Set up source space
    # ------------------------------------------------------------------

    # HINT for params.py
    # fnames.add('src_surface', '{fwd_sub}/{subject}_surface-spacing-{surf_spacing}-src.fif')
    # fnames.add('src_volume', '{fwd_sub}/{subject}_volume-spacing-{vol_spacing}mm-src.fif')
    # fnames.add('src_combined', '{fwd_sub}/{subject}_combined-surface-spacing-{surf_spacing}-volume-spacing-{vol_spacing}mm-src.fif')
    
    # surface
    src_surf_path = fnames.src_surface(subject=subject, surf_spacing=SRC_SPACING)
    src_surf = mne.setup_source_space(subject, spacing=SRC_SPACING, subjects_dir=fnames.subjects_dir)
    mne.write_source_spaces(src_surf_path, src_surf, overwrite=True)

    # volume source space
    src_vol_path = fnames.src_volume(subject=subject, vol_spacing=VOL_SPACING)
    src_vol = mne.setup_volume_source_space(subject, subjects_dir=fnames.subjects_dir, pos=VOL_SPACING, bem=bem_sol)
    mne.write_source_spaces(src_vol_path, src_vol, overwrite=True)

    # combined surface and volume source space 
    src_combined_path = fnames.src_combined(subject=subject, surf_spacing=SRC_SPACING, vol_spacing=VOL_SPACING)
    src_combined = src_surf + src_vol
    mne.write_source_spaces(src_combined_path, src_combined, overwrite=True)
    

    # ------------------------------------------------------------------
    # Compute morphs to fsaverage and save
    # ------------------------------------------------------------------
    if not subject == SUBJECT_TO:
    
        paths_subject_to = [
            fnames.src_surface(subject=SUBJECT_TO, surf_spacing=SRC_SPACING), 
            fnames.src_volume(subject=SUBJECT_TO, vol_spacing=VOL_SPACING), 
            fnames.src_combined(subject=SUBJECT_TO, surf_spacing=SRC_SPACING, vol_spacing=VOL_SPACING)
        ]

        paths_subject_from = [
            src_surf_path,
            src_vol_path,
            src_combined_path
        ]


        for src_to_path, src_from_path in zip(paths_subject_to, paths_subject_from):
            src_to = mne.read_source_spaces(src_to_path)
            src_from = mne.read_source_spaces(src_from_path)
            
            morph = mne.compute_source_morph(
                src_from,
                subject_from=subject,
                subject_to=SUBJECT_TO,
                src_to=src_to,
                subjects_dir=fnames.subjects_dir,
                verbose=False,
            )

            src_info = src_from_path.name.replace("-src.fif", "")
            src_info = src_info.replace(f"{subject}_", "")

            morph_path = fnames.morph(subject=subject, src_info=src_info, subject_to=SUBJECT_TO)
            morph.save(morph_path, overwrite=True)

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    save_report(report, report_path)
