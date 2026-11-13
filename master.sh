#!/usr/bin/env bash
# Usage:
#   bash master.sh          # run all subjects
#   bash master.sh 0188     # run one subject


subjects=("0188" "0189" "0190" "0191" "0192" "0193" "0194" "0195") #

run_group_stats=true

# If a subject is provided, override the list
if [ "$#" -eq 1 ]; then 
subjects=("$1")
run_group_stats=false
elif [ "$#" -gt 1 ]; then
echo "Usage: $0 [subject]"
exit 1
fi

# Path to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


for subject in "${subjects[@]}"; do
    echo "========================================"
    echo "Running pipeline for subject: ${subject}"
    echo "========================================"

    ## Core preprocessing
    #python "${SCRIPT_DIR}/01_anatomy.py" --subject "${subject}"
    #python "${SCRIPT_DIR}/02_forward.py" --subject "${subject}"
    #python "${SCRIPT_DIR}/03_filter.py"  --subject "${subject}"
    #python "${SCRIPT_DIR}/04_ica.py"     --subject "${subject}"
    #python "${SCRIPT_DIR}/05_epochs.py"  --subject "${subject}"
    #python "${SCRIPT_DIR}/06_stcs_beamforming.py"  --subject "${subject}"
    #python "${SCRIPT_DIR}/07_evoked.py"  --subject "${subject}"
done

if [ "$run_group_stats" = true ]; then
echo "========================================"
echo "Running group-level analysis"
echo "========================================"

#python "${SCRIPT_DIR}/08_erf_stats.py"
python "${SCRIPT_DIR}/09_decoding_sensor_space.py"
python "${SCRIPT_DIR}/09_decoding_source_space.py"
fi