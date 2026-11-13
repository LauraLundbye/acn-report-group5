
from pathlib import Path
import mne

def setup_report(report_path: Path, title: str) -> mne.Report:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = mne.open_report(report_path, title=title)

    return report

def save_report(report: mne.Report, report_path: Path):
    report.save(report_path, overwrite=True)
    # also change .h5 extension to .html for easier viewing outside of MNE environment
    html_report_path = report_path.with_suffix('.html')
    print(f"Saving HTML version of report to {html_report_path}")
    report.save(html_report_path, overwrite=True)