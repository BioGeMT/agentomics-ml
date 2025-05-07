#!/bin/bash

# Check if run_name is provided as an argument
if [ $# -eq 0 ]; then
    echo "Please provide a work dir path as an argument"
    exit 1
fi

# Create data.yaml file
cat > "MetaGPT/metagpt/ext/sela/data.yaml" << EOF
datasets_dir: "/repository/datasets" # path to the datasets directory
work_dir: $work_dir # path to the workspace directory
role_dir: storage/SELA # relative path to the role directory
EOF

echo "Augmented data.yaml"