
#to be overwritten to metagpt
# -*- coding: utf-8 -*-
import argparse
import asyncio
import json
import os
from pathlib import Path
import traceback # Added for error printing
import sys # Ensure sys is imported

import openml
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Assuming these imports work after utils.py is also patched
try:
    from metagpt.ext.sela.insights.solution_designer import SolutionDesigner
    # Import DATA_CONFIG from the *patched* utils module
    from metagpt.ext.sela.utils import DATA_CONFIG, SPECIAL_INSTRUCTIONS, DI_INSTRUCTION, TASK_PROMPT # Import necessary constants from utils
except ImportError as e:
     print(f"Dataset.py Error: Could not import SELA modules: {e}")
     SolutionDesigner = None # Define placeholders if import fails
     DATA_CONFIG = {} # Define placeholder
     # Define constants locally if import fails (less ideal)
     SPECIAL_INSTRUCTIONS = {}
     DI_INSTRUCTION = "## Attention\n..." # Add full text if needed
     TASK_PROMPT = "# User requirement\n..." # Add full text if needed

BASE_USER_REQUIREMENT = """
This is a {datasetname} dataset. Your goal is to predict the target column `{target_col}`.
Perform data analysis, data preprocessing, feature engineering, and modeling to predict the target.
Report {metric} on the eval data. Do not plot or make any visualizations.
"""

# Constants from original dataset.py (can be removed if imported from utils)
# USE_AG = ... TEXT_MODALITY = ... IMAGE_MODALITY = ... STACKING = ...

SEED = 100; TRAIN_TEST_SPLIT = 0.8; TRAIN_DEV_SPLIT = 0.75
OPENML_DATASET_IDS = [ 41021, 42727, 41980, 42225, 531, 41143, 31, 42733, 41162, 1067, 40498, 40982, 12, 40984, 4538 ]
CUSTOM_DATASETS = [ ("04_titanic", "Survived"), ("05_house-prices-advanced-regression-techniques", "SalePrice"),
    ("06_santander-customer-transaction-prediction", "target"), ("07_icr-identify-age-related-conditions", "Class"), ]
DSAGENT_DATASETS = [("concrete-strength", "Strength"), ("smoker-status", "smoking"), ("software-defects", "defects")]


# Functions from original dataset.py
def get_split_dataset_path(dataset_name, config):
    """Provides paths where split files are expected to be SAVED (work_dir)."""
    work_dir = config.get("work_dir"); assert work_dir, "work_dir missing"
    path = Path(work_dir)
    split_datasets = {k: path / f"split_{k.replace('_wo_target','').replace('_target','')}{'_wo_target' if '_wo_target' in k else ''}{'_target' if '_target' in k else ''}.csv"
                      for k in ["train", "dev", "dev_wo_target", "dev_target", "test", "test_wo_target", "test_target"]}
    return {k: str(v) for k, v in split_datasets.items()}

def get_user_requirement(task_name, config):
    datasets = config.get("datasets", {}); assert task_name in datasets, f"Dataset {task_name} not found in config"
    dataset = datasets[task_name]; user_requirement = dataset.get("user_requirement"); assert user_requirement, f"user_requirement missing for {task_name}"
    return user_requirement

def save_datasets_dict_to_yaml(datasets_dict, name="datasets.yaml"):
    # Save relative to this script's location (in SELA extension dir)
    save_path = Path(__file__).parent / name; print(f"Saving datasets dictionary to: {save_path}")
    try:
        with open(save_path, "w", encoding='utf-8') as file_handle: # Use different variable name
             yaml.dump(datasets_dict, file_handle, default_flow_style=False)
    except Exception as e: print(f"Error saving datasets YAML {save_path}: {e}")

def create_dataset_dict(dataset):
    name = getattr(dataset, 'name', 'unknown'); target_col = getattr(dataset, 'target_col', None); assert target_col, f"Target column not set for {name}"
    print(f"Creating dataset dict for: {name}"); metric = "unknown"; user_req = "Predict target"
    try: metric = dataset.get_metric()
    except Exception as e: print(f"Warning: Error getting metric for {name}: {e}")
    try: user_req = dataset.create_base_requirement()
    except Exception as e: print(f"Warning: Error creating base requirement for {name}: {e}")
    return { "dataset": name, "user_requirement": user_req, "metric": metric, "target_col": target_col }

def generate_di_instruction(output_dir, special_instruction):
    sp_instr_prompt = SPECIAL_INSTRUCTIONS.get(special_instruction, ""); return DI_INSTRUCTION.format(output_dir=str(output_dir), special_instruction=sp_instr_prompt)

# ---> generate_task_requirement MOVED BACK HERE <---
def generate_task_requirement(task_name, data_config, is_di=True, special_instruction=None):
    """Generates the task prompt, pointing to splits in the work_dir."""
    user_requirement = get_user_requirement(task_name, data_config) # Assumes this works

    work_dir = data_config.get("work_dir")
    datasets_dir = data_config.get("datasets_dir")
    dataset_folder = data_config.get("datasets", {}).get(task_name, {}).get("dataset")

    assert work_dir, "'work_dir' missing from data_config"
    assert datasets_dir, "'datasets_dir' missing from data_config"
    assert dataset_folder, f"'dataset' key missing for task {task_name}"

    # Paths for splits are now relative to work_dir
    work_dir_path = Path(work_dir)
    train_path = str(work_dir_path / "split_train.csv")
    dev_path = str(work_dir_path / "split_dev.csv")
    test_path = str(work_dir_path / "split_test_wo_target.csv") # Path for test set without labels

    # Path to original dataset description/info file
    data_info_path = Path(datasets_dir) / dataset_folder / "dataset_info.json" # Check for info json first
    if not data_info_path.exists():
         # Fallback to description.md if info.json not found
         data_info_path = Path(datasets_dir) / dataset_folder / "dataset_description.md" # Or description.md? Check file listing
    data_info_path_str = str(data_info_path) if data_info_path.exists() else "Not Found"

    # Output dir for results (usually within work_dir)
    output_dir = work_dir_path # Assume results go directly in work_dir

    additional_instruction = generate_di_instruction(output_dir, special_instruction) if is_di else ""

    # TASK_PROMPT uses the modified paths now
    final_user_requirement = TASK_PROMPT.format(
        user_requirement=user_requirement,
        additional_instruction=additional_instruction,
        train_path=train_path, # Points to work_dir
        dev_path=dev_path,     # Points to work_dir
        test_path=test_path,   # Points to work_dir
        data_info_path=data_info_path_str # Points to original dataset dir
    )
    print("--- Generated Task Requirement (using work_dir splits) ---")
    print(final_user_requirement)
    print("--------------------------------------------------------")
    return final_user_requirement
# ---> END generate_task_requirement <---


# ExpDataset class definition with get_raw_dataset and save_dataset modified
class ExpDataset:
    description: str = None; metadata: dict = None; target_col: str = None; name: str = None
    dataset_dir: str = DATA_CONFIG.get("datasets_dir", "/repository/Agentomics-ML/datasets") # Default if needed

    def __init__(self, name, dataset_dir=None, **kwargs):
        self.name = name; self.dataset_dir = Path(dataset_dir if dataset_dir is not None else self.dataset_dir)
        if not self.dataset_dir: raise ValueError("Dataset directory is not set")
        self.target_col = kwargs.get("target_col", None); self.force_update = kwargs.get("force_update", False)
        print(f"Initializing ExpDataset for '{self.name}' in dir '{self.dataset_dir}'")
        ds_path = self.dataset_path; assert ds_path.is_dir(), f"Dataset directory does not exist: {ds_path}"
        self.save_dataset(target_col=self.target_col)

    @property
    def dataset_path(self) -> Path: return self.dataset_dir / self.name

    def check_dataset_exists(self):
        """Checks if split files exist in the run's work_dir."""
        work_dir = DATA_CONFIG.get("work_dir")
        if not work_dir: print("Warning: work_dir not in DATA_CONFIG for check_dataset_exists."); return False
        save_path = Path(work_dir)
        fnames = [ f"split_{s}.csv" for s in ["train", "dev", "test"]] + [ f"split_{s}_{t}.csv" for s in ["dev", "test"] for t in ["wo_target", "target"] ]
        all_exist = all((save_path / fname).exists() for fname in fnames)
        if all_exist: print(f"Found existing split files in {save_path}")
        return all_exist

    def check_datasetinfo_exists(self):
        """Checks if dataset info exists in the run's work_dir."""
        work_dir = DATA_CONFIG.get("work_dir")
        if not work_dir: return False
        info_path = Path(work_dir) / f"{self.name}_dataset_info.json"
        return info_path.exists()

    def get_raw_dataset(self):
        # Use the main dataset directory instead of 'raw/' subdir
        dataset_root_dir = self.dataset_path
        print(f"get_raw_dataset [Patched]: Looking for data in {dataset_root_dir}")
        train_df = None; test_df = None
        train_csv_path = dataset_root_dir / "train.csv"; test_csv_path = dataset_root_dir / "test.csv"
        print(f"get_raw_dataset [Patched]: Checking for {train_csv_path}")
        assert train_csv_path.exists(), f"Dataset `train.csv` not found in {dataset_root_dir}"
        try: train_df = pd.read_csv(train_csv_path); print(f"get_raw_dataset [Patched]: Loaded train.csv (shape: {train_df.shape})")
        except Exception as e: print(f"Error loading {train_csv_path}: {e}"); raise
        print(f"get_raw_dataset [Patched]: Checking for {test_csv_path}")
        if test_csv_path.exists():
            try: test_df = pd.read_csv(test_csv_path); print(f"get_raw_dataset [Patched]: Loaded test.csv (shape: {test_df.shape})")
            except Exception as e: print(f"Error loading {test_csv_path}: {e}"); test_df = None
        else: print(f"get_raw_dataset [Patched]: Optional test.csv not found at {test_csv_path}.")
        return train_df, test_df

    def get_dataset_info(self):
        try: raw_df, _ = self.get_raw_dataset()
        except FileNotFoundError: print("Error: Cannot get dataset info, raw train.csv not found."); return None
        num_classes = 0
        if self.target_col:
            if self.target_col not in raw_df.columns: print(f"Warning: Target column '{self.target_col}' not found in raw data columns.")
            else: num_classes = raw_df[self.target_col].nunique()
        metadata = { "NumberOfClasses": num_classes, "NumberOfFeatures": raw_df.shape[1], "NumberOfInstances": raw_df.shape[0],
                     "NumberOfInstancesWithMissingValues": int(raw_df.isnull().any(axis=1).sum()), "NumberOfMissingValues": int(raw_df.isnull().sum().sum()),
                     "NumberOfNumericFeatures": raw_df.select_dtypes(include="number").shape[1],
                     "NumberOfSymbolicFeatures": raw_df.select_dtypes(include=["object", "category", "string"]).shape[1] }
        dataset_info = { "name": self.name, "description": getattr(self,'description',''), "target_col": self.target_col,
                         "metadata": metadata, "df_head": self.get_df_head(raw_df) }
        self.metadata = metadata
        return dataset_info

    def get_df_head(self, raw_df): return raw_df.head().to_string(index=False)

    def get_metric(self):
        if not hasattr(self, 'metadata') or self.metadata is None: self.get_dataset_info()
        if not hasattr(self, 'metadata') or self.metadata is None: return "unknown"
        num_classes = self.metadata["NumberOfClasses"]; metric = "unknown"
        if num_classes == 2: metric = "f1_binary"
        elif 2 < num_classes <= 200: metric = "f1_weighted"
        elif num_classes > 200 or num_classes == 0: metric = "rmse"
        print(f"Determined metric: {metric} (num_classes={num_classes})")
        return metric

    def create_base_requirement(self):
        metric = self.get_metric(); assert self.target_col, "Target column must be set"; req = BASE_USER_REQUIREMENT.format(datasetname=self.name, target_col=self.target_col, metric=metric)
        return req

    def save_dataset(self, target_col):
        if not self.force_update and self.check_dataset_exists(): print(f"Split datasets for '{self.name}' found in work_dir. Skipping generation."); self.check_and_save_dataset_info(); return
        print(f"Generating/Saving Dataset splits for '{self.name}' (force={self.force_update})..."); df, test_df = self.get_raw_dataset(); assert target_col and target_col in df.columns, f"Target '{target_col}' invalid/missing."
        SEED = 100; TRAIN_TEST_SPLIT_RATIO = 0.8; TRAIN_DEV_SPLIT_RATIO = 0.75; stratify_col = df[target_col] if df[target_col].nunique() < len(df)//2 else None
        if test_df is None:
            print("Splitting train->train/dev/test..."); train_dev_df, test_df = train_test_split(df, test_size=1-TRAIN_TEST_SPLIT_RATIO, random_state=SEED, stratify=stratify_col); stratify_col_td = train_dev_df[target_col] if train_dev_df[target_col].nunique()<len(train_dev_df)//2 else None; train_df, dev_df = train_test_split(train_dev_df, test_size=1-TRAIN_DEV_SPLIT_RATIO, random_state=SEED, stratify=stratify_col_td); test_has_target=True
        else:
            print("Using provided test.csv; splitting train->train/dev..."); test_has_target = target_col in test_df.columns; train_df, dev_df = train_test_split(df, test_size=1 - TRAIN_DEV_SPLIT_RATIO, random_state=SEED, stratify=stratify_col);
            if not test_has_target: print(f"Warning: Provided test.csv missing target '{target_col}'.")
        print(f"Split sizes: Train={len(train_df)}, Dev={len(dev_df)}, Test={len(test_df)}");
        self.save_split_datasets(train_df, "train"); self.save_split_datasets(dev_df, "dev", target_col); self.save_split_datasets(test_df, "test", target_col if test_has_target else None)
        self.check_and_save_dataset_info() # Saves info to work_dir

    def check_and_save_dataset_info(self):
         """Checks if dataset info exists in work_dir and saves if not."""
         work_dir = DATA_CONFIG.get("work_dir")
         if not work_dir: print("Warning: work_dir not found. Cannot check/save dataset_info.json."); return
         info_path = Path(work_dir) / f"{self.name}_dataset_info.json" # Save in work_dir
         if not info_path.exists() or self.force_update:
             print(f"Generating/Saving dataset info for '{self.name}' to {info_path}")
             dataset_info = self.get_dataset_info() # Generate info
             self.save_datasetinfo(dataset_info, info_path) # Pass target path
         else: print(f"Dataset info for {self.name} already exists at {info_path}")

    def save_datasetinfo(self, dataset_info, save_path: Path): # Modified signature
        """Saves dataset info to the specified path (now expected to be in work_dir)."""
        if dataset_info is None: print("Warning: Dataset info is None, cannot save."); return
        print(f"Saving dataset info to: {save_path}")
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
            # ---> CORRECTED: Use 'f' as file handle variable <---
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(dataset_info, f, indent=4, ensure_ascii=False)
            # ---> END CORRECTION <---
        except Exception as e:
            print(f"Error saving dataset_info.json to {save_path}: {e}")

    def save_split_datasets(self, df, split_name, target_col=None):
        """Saves split files into the run's work_dir."""
        work_dir = DATA_CONFIG.get("work_dir"); assert work_dir, "work_dir missing"; path = Path(work_dir); path.mkdir(parents=True, exist_ok=True)
        split_file_path = path / f"split_{split_name}.csv"; print(f"Saving {split_file_path}..."); df.to_csv(split_file_path, index=False)
        if target_col and target_col in df.columns:
            df_wo_target = df.drop(columns=[target_col]); split_wo_target_file_path = path / f"split_{split_name}_wo_target.csv"; print(f"Saving {split_wo_target_file_path}..."); df_wo_target.to_csv(split_wo_target_file_path, index=False)
            df_target = df[[target_col]].copy(); df_target.rename(columns={target_col: "target"}, inplace=True); split_target_file_path = path / f"split_{split_name}_target.csv"; print(f"Saving {split_target_file_path}..."); df_target.to_csv(split_target_file_path, index=False)
        elif target_col: print(f"Warning: Target '{target_col}' missing in split '{split_name}'.")


class OpenMLExpDataset(ExpDataset): # Kept original - may need adjustments if used
    def __init__(self, name, dataset_dir, dataset_id, **kwargs):
        self.dataset_id = dataset_id; print(f"Fetching OpenML dataset ID: {self.dataset_id}")
        self.dataset = openml.datasets.get_dataset( self.dataset_id, download_data=False, download_qualities=False, download_features_meta_data=True )
        self.name = self.dataset.name or f"openml_{dataset_id}"; self.target_col = self.dataset.default_target_attribute; assert self.target_col, f"No default target for OpenML {dataset_id}"
        # OpenML still uses base class __init__ which calls save_dataset, which now saves splits to work_dir
        super().__init__(self.name, dataset_dir, target_col=self.target_col, **kwargs)

    def get_raw_dataset(self): # Keep original OpenML get_raw_dataset - it creates raw/ subdir
        print(f"OpenML get_raw_dataset for {self.name} (ID: {self.dataset_id})")
        dataset = self.dataset; dataset_df, *_ = dataset.get_data(dataset_format="dataframe", target=self.target_col)
        raw_dir = self.dataset_path / "raw"; raw_dir.mkdir(parents=True, exist_ok=True) # Uses raw subdir
        raw_train_path = raw_dir / "train.csv"; print(f"Saving downloaded OpenML data to {raw_train_path}")
        dataset_df.to_csv(raw_train_path, index=False)
        return dataset_df, None # Return the dataframe and None for test_df

    def get_dataset_info(self): # Keep original OpenML get_dataset_info
        raw_train_path = self.dataset_path / "raw" / "train.csv"; print(f"OpenML get_dataset_info: loading raw data from {raw_train_path}")
        if not raw_train_path.exists(): print("Attempting download..."); self.get_raw_dataset(); assert raw_train_path.exists(), "Download failed"
        try: raw_df = pd.read_csv(raw_train_path)
        except Exception as e: print(f"Error reading raw OpenML train file {raw_train_path}: {e}"); return None
        num_classes = 0;
        if self.target_col and self.target_col in raw_df.columns: num_classes = raw_df[self.target_col].nunique()
        metadata = { "NumberOfClasses": num_classes, "NumberOfFeatures": raw_df.shape[1], "NumberOfInstances": raw_df.shape[0],
                     "NumberOfInstancesWithMissingValues": int(raw_df.isnull().any(axis=1).sum()), "NumberOfMissingValues": int(raw_df.isnull().sum().sum()),
                     "NumberOfNumericFeatures": raw_df.select_dtypes(include="number").shape[1],
                     "NumberOfSymbolicFeatures": raw_df.select_dtypes(include=["object", "category", "string"]).shape[1] }
        dataset_info = { "name": self.name, "description": self.dataset.description, "target_col": self.target_col,
                         "metadata": metadata, "df_head": self.get_df_head(raw_df) }
        try: dataset_info["metadata"].update(self.dataset.qualities) # Add OpenML qualities
        except Exception as e: print(f"Warning: could not add OpenML qualities: {e}")
        self.metadata = dataset_info["metadata"]; # Store metadata
        return dataset_info


async def process_dataset(dataset, solution_designer: SolutionDesigner, save_analysis_pool, datasets_dict):
    # Keep original async process_dataset
    print(f"Processing dataset {dataset.name} asynchronously...")
    if save_analysis_pool and solution_designer:
        dataset_info = dataset.get_dataset_info()
        if dataset_info: await solution_designer.generate_solutions(dataset_info, dataset.name)
        else: print(f"Skipping solution generation for {dataset.name} (missing info).")


def parse_cli_args(): # Keep original parse_cli_args
    parser = argparse.ArgumentParser(description="SELA Dataset Processor Command Line")
    parser.add_argument("--force_update", action="store_true", help="Force update datasets (regenerate splits)")
    parser.add_argument("--save_analysis_pool", action="store_true", default=False, help="Save analysis pool (off by default)")
    parser.add_argument("--no_save_analysis_pool", dest="save_analysis_pool", action="store_false", help="Do not save analysis pool")
    parser.add_argument("--dataset", type=str, help="Name of the specific dataset to process.")
    parser.add_argument("--target_col", type=str, help="(Optional) Target column name override.")
    parser.add_argument("--openml_id", type=int, help="(Optional) OpenML dataset ID to process.")
    return parser.parse_args()


if __name__ == "__main__": # Keep original __main__ but with syntax fixes
    print("--- Running dataset.py as main script ---")
    args = parse_cli_args(); datasets_dir = DATA_CONFIG.get("datasets_dir"); assert datasets_dir, "'datasets_dir' missing"; print(f"Using datasets_dir: {datasets_dir}")
    force_update = args.force_update; save_analysis_pool = args.save_analysis_pool; processed_something = False
    solution_designer = SolutionDesigner() if save_analysis_pool and SolutionDesigner is not None else None

    if args.openml_id:
        print(f"Processing OpenML dataset ID: {args.openml_id}")
        # Corrected try...except block
        try:
            openml_dataset = OpenMLExpDataset("", datasets_dir, args.openml_id, force_update=force_update)
            processed_something = True
            print(f"Completed OpenML: {openml_dataset.name}")
        except Exception as e:
            print(f"Error processing OpenML {args.openml_id}: {e}"); traceback.print_exc()
    elif args.dataset:
        print(f"Processing custom dataset: {args.dataset}"); target_col = args.target_col; all_known = dict(CUSTOM_DATASETS + DSAGENT_DATASETS); target_col = target_col or all_known.get(args.dataset); assert target_col, f"Target col for '{args.dataset}' not provided/found."
        # Corrected try...except block
        try:
            custom_dataset = ExpDataset(args.dataset, datasets_dir, target_col=target_col, force_update=force_update)
            processed_something = True
            print(f"Completed custom: {args.dataset}")
        except Exception as e:
            print(f"Error processing custom dataset {args.dataset}: {e}"); traceback.print_exc()
    else:
        print("No specific dataset (--dataset) or OpenML ID (--openml_id) provided. Exiting."); sys.exit(1) # Use sys.exit

    # Optional: Add call to process_dataset if analysis pool is needed for single run
    # if processed_something and save_analysis_pool and solution_designer:
    #     dataset_obj = openml_dataset if args.openml_id else custom_dataset
    #     if dataset_obj:
    #         asyncio.run(process_dataset(dataset_obj, solution_designer, True, {}))

    if processed_something: print("--- dataset.py finished ---")
    else: print("--- dataset.py failed to process any dataset ---"); sys.exit(1) # Use sys.exit

# --- End of Overwritten dataset.py ---
