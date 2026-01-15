# Script that will go through /SCRATCH/biomlbench/runs/<whatever_folder>/<run_name> and /SCRATCH/agentomics-ml/outputs/<run_name> for each available run_name
# (the whatever folder needs to be exactly 1 folder, not any amount of folders)
# then run 
# python ./src/eval/arch_summary.py --run-name <run_name> --base-path <the_path_Where_the_run_name_is> --output-path archsumms/<run_name>.txt
#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p archsumms

# Check /SCRATCH/biomlbench/runs/<whatever_folder>/<run_name>
if [ -d "/SCRATCH/biomlbench/runs" ]; then
    for folder in /SCRATCH/biomlbench/runs/*/; do
        if [ -d "$folder" ]; then
            for dataset_folder in "$folder"*/; do
                if [ -d "$dataset_folder" ]; then
                    for run_folder in "$dataset_folder"*/; do
                        if [ -d "$run_folder/code/run_files" ]; then
                            # Process all runs in the run_files directory
                            for run_name_dir in "$run_folder/code/run_files"/*/; do
                                if [ -d "$run_name_dir" ]; then
                                    run_basename=$(basename "$run_name_dir")
                                    iters_path="$run_name_dir"
                                    echo "Processing biomlbench run: $run_basename from $run_folder"
                                    python ./src/eval/arch_summary.py --run-name "$run_basename" --iters-path "$iters_path" --output-path "archsumms/${run_basename}.txt"
                                fi
                            done
                        fi
                    done
                fi
            done
        fi
    done
fi

# Check /SCRATCH/agentomics-ml/outputs/<run_name>
if [ -d "/SCRATCH/agentomics-ml/outputs" ]; then
    for run_name in /SCRATCH/agentomics-ml/outputs/*/; do
        if [ -d "$run_name" ]; then
            run_basename=$(basename "$run_name")
            iters_path="$run_name/run_files"
            echo "Processing agentomics-ml run: $run_basename"
            python ./src/eval/arch_summary.py --run-name "$run_basename" --iters-path "$iters_path" --output-path "archsumms/${run_basename}.txt"
        fi
    done
fi

echo "Architecture summaries complete!"

# Generate CSV from text files
echo "Generating architecture summaries CSV..."
python ./src/eval/arch_to_csv.py --archsumms-dir archsumms --output-csv archsumms/architecture_summaries.csv

