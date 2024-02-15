#!/bin/bash
base_path=$(dirname "$0")            # relative
base_path=$(cd "$MY_PATH" && pwd)    # absolutized and normalized
if [[ -z "$base_path" ]] ; then  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
echo "$base_path"

datasets=("SUPPORT" "SEER" "METABRIC" "MIMIC")
echo "=============================================================================================="
echo "Starting datasets tuning"
echo "=============================================================================================="
for dataset in ${datasets[@]}; do
    echo "Starting dataset run for MLP on <$dataset>"
    python $base_path/../src/tuning/tune_sota_models.py --dataset $dataset
    echo "Tuning MLP <$dataset> done"
    echo -e "\n\n\n\n\n"
    echo "=============================================================================================="
    echo -e "\n\n\n\n\n"
done
echo "Finished executing datasets"