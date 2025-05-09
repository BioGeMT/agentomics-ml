#to be overwritten to metagpt
# -*- coding: utf-8 -*-
import os
import re
from datetime import datetime
from pathlib import Path
import traceback
import sys

import nbformat
import yaml
from loguru import logger as _logger
from nbclient import NotebookClient
from nbformat.notebooknode import NotebookNode

try:
    from metagpt.roles.role import Role
except ImportError:
    print("SELA Utils Warning: Could not import Role at initial load time.")
    Role = object

# --- Corrected Config Loading ---
# Define absolute paths to the config files within the cloned MetaGPT fork
# This ensures utils.py (wherever it's imported from) finds these specific files.
SELA_EXTENSION_CONFIG_DIR = Path("/tmp/MetaGPT_fork_sela/metagpt/ext/sela/")
DEFAULT_DATA_YAML_PATH = SELA_EXTENSION_CONFIG_DIR / "data.yaml"
DEFAULT_DATASETS_YAML_PATH = SELA_EXTENSION_CONFIG_DIR / "datasets.yaml"

def load_data_config(config_path: Path = None):
    """Loads YAML config. If config_path is None, uses predefined default."""
    if config_path is None:
        # This case is used by solution_designer.py calling load_data_config()
        # AND by the global DATA_CONFIG load below.
        config_path = DEFAULT_DATA_YAML_PATH
        print(f"SELA Utils: load_data_config called with no args, using default: {config_path}")
    elif not isinstance(config_path, Path):
         try:
             config_path = Path(config_path)
         except TypeError:
             print(f"SELA Utils Error: Invalid type passed for config_path: {type(config_path)}"); return {}

    print(f"SELA Utils: Loading config: {config_path}");
    if not config_path.exists():
        print(f"SELA Utils Warn: Config not found: {config_path}"); return {}
    try:
        with open(config_path, "r", encoding='utf-8') as stream:
            data_config = yaml.safe_load(stream)
        if data_config is None:
            print(f"SELA Utils Warn: Config loaded None: {config_path}"); return {}
        if not isinstance(data_config, dict):
            print(f"SELA Utils Warn: Config not dict: {config_path}"); return {}
        print(f"SELA Utils: Loaded config from {config_path}"); return data_config
    except Exception as exc:
        print(f"SELA Utils Error: Loading/Parsing {config_path}: {exc}"); return {}

# These global loads will now use the absolute paths defined above
DATASET_CONFIG = load_data_config(DEFAULT_DATASETS_YAML_PATH);
DATA_CONFIG = load_data_config(DEFAULT_DATA_YAML_PATH)
DATA_CONFIG["datasets"] = DATASET_CONFIG.get("datasets", {});
print(f"SELA Utils: Final DATA_CONFIG keys after loading defaults: {list(DATA_CONFIG.keys())}")
# --- End of Corrected Config Loading ---

# --- Corrected Logger Initialization ---
def get_mcts_logger():
    """Initializes and returns the MCTS logger."""
    logfile_level = "DEBUG"; name: str = None; current_date = datetime.now(); formatted_date = current_date.strftime("%Y%m%d"); log_name = formatted_date
    work_dir = DATA_CONFIG.get("work_dir"); role_dir = DATA_CONFIG.get("role_dir"); log_file_path = None

    if work_dir and role_dir:
        try:
            log_path_base = Path(work_dir) / role_dir
            log_path_base.mkdir(parents=True, exist_ok=True)
            log_file_path = log_path_base / f"{log_name}.txt"
            print(f"SELA Utils: Determined log file path: {log_file_path}")
        except Exception as e:
            print(f"SELA Utils Error: Log path construction: {e}"); log_file_path = None

    if log_file_path is None:
        log_file_path = Path("/tmp") / f"sela_fallback_{log_name}.log"
        print(f"SELA Utils Warn: Using fallback log path: {log_file_path}")
        try:
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"SELA Utils Error: Failed create fallback log dir: {e}"); log_file_path=None

    try: _logger.remove()
    except ValueError: pass

    _logger.level("MCTS", color="<green>", no=25)
    if log_file_path and log_file_path.parent.exists():
        try:
            _logger.add(log_file_path, level=logfile_level, rotation="10 MB", retention="1 day")
            print(f"SELA Utils: Added log handler: {log_file_path}")
        except Exception as e:
            print(f"SELA Utils Error: Failed add log handler {log_file_path}: {e}")
    else:
        print(f"SELA Utils Warning: Could not add file handler for MCTS logger.")

    _logger.propagate = False; return _logger
mcts_logger = get_mcts_logger()
# --- End of Corrected Logger Initialization ---

# --- Original Functions from utils.py (Keep all including clean_json_from_rsp) ---
# generate_task_requirement is now in dataset_fixed.py
# from metagpt.ext.sela.data.dataset import TASK_PROMPT, SPECIAL_INSTRUCTIONS, DI_INSTRUCTION # These are needed by generate_task_requirement

def get_exp_pool_path(task_name, data_config, pool_name="analysis_pool"):
    datasets_dir=data_config.get("datasets_dir"); assert datasets_dir, "'datasets_dir' not found"; datasets=data_config.get("datasets",{}); assert task_name in datasets; dataset=datasets[task_name]; dataset_name=dataset.get("dataset"); assert dataset_name; data_path=os.path.join(datasets_dir, dataset_name); exp_pool_path=os.path.join(data_path, f"{pool_name}.json"); return exp_pool_path if os.path.exists(exp_pool_path) else None

def change_plan(role, plan):
    if Role is None or not isinstance(role, Role): print("SELA Utils Error: Invalid 'role'."); return True
    print(f"SELA Utils: Attempting change plan: {plan}")
    if not hasattr(role, 'planner') or not hasattr(role.planner, 'plan') or not hasattr(role.planner.plan, 'tasks'): return True
    tasks = role.planner.plan.tasks; finished = True; task_index_to_update = -1
    for i, task in enumerate(tasks):
        if not getattr(task, 'is_finished', False): finished = False; task_index_to_update = i; break
    if not finished and task_index_to_update != -1: setattr(tasks[task_index_to_update], 'plan', plan)
    elif finished: print("SELA Utils: All tasks finished.")
    return finished

def is_cell_to_delete(cell: NotebookNode) -> bool:
    if cell.get("cell_type") != "code": return False
    if "outputs" in cell:
        for output in cell["outputs"]:
            if output and output.get("output_type") == "error": print(f"SELA Utils: Deleting cell error: {output.get('ename')}"); return True
    return False

def process_cells(nb: NotebookNode) -> NotebookNode:
    if not nb or "cells" not in nb: return nb; new_cells = []; exec_count = 1; cells_deleted_count = 0
    for cell in nb["cells"]:
        if not is_cell_to_delete(cell):
            if cell.get("cell_type") == "code": cell["execution_count"] = exec_count; exec_count += 1
            new_cells.append(cell)
        else: cells_deleted_count += 1
    if cells_deleted_count > 0: print(f"SELA Utils: Removed {cells_deleted_count} error cell(s).")
    nb["cells"] = new_cells; return nb

def save_notebook(role: Role, save_dir: str = "", name: str = "", save_to_depth=False):
    if Role is None or not isinstance(role, Role): print("SELA Utils Error: Invalid role."); return
    save_path = Path(save_dir); save_path.mkdir(parents=True, exist_ok=True); assert hasattr(role, 'execute_code') and hasattr(role.execute_code, 'nb'), "Role missing notebook"
    try:
        processed_nb = process_cells(role.execute_code.nb)
        file_path = save_path / f"{name}.ipynb"
        print(f"SELA Utils: Saving notebook: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
             nbformat.write(processed_nb, f)
    except Exception as e:
        print(f"SELA Utils Error saving notebook: {e}"); traceback.print_exc()

    if save_to_depth:
        clean_file_path = save_path / f"{name}_clean.ipynb"
        try:
            assert hasattr(role, 'planner') and hasattr(role.planner, 'plan'); tasks = role.planner.plan.tasks; codes = [getattr(t, 'code', '') for t in tasks]; codes = [c for c in codes if c]; clean_nb = nbformat.v4.new_notebook()
            for code in codes: clean_nb.cells.append(nbformat.v4.new_code_cell(code))
            print(f"SELA Utils: Saving clean notebook: {clean_file_path}");
            with open(clean_file_path, 'w', encoding='utf-8') as f: nbformat.write(clean_nb, f)
        except Exception as e: print(f"SELA Utils Error saving clean notebook: {e}"); traceback.print_exc()

async def load_execute_notebook(role):
    if Role is None or not isinstance(role, Role): print("SELA Utils Error: Invalid role."); return None
    assert hasattr(role, 'planner') and hasattr(role.planner, 'plan'); tasks = role.planner.plan.tasks; codes = [getattr(t, 'code', '') for t in tasks]; codes = [c for c in codes if c]; assert hasattr(role, 'execute_code'); executor = role.execute_code; executor.nb = nbformat.v4.new_notebook(); role_timeout = getattr(role, 'role_timeout', 600); executor.nb_client = NotebookClient(executor.nb, timeout=role_timeout); print(f"SELA Utils: Executing {len(codes)} blocks..."); final_success = True
    for i, code in enumerate(codes):
        block_num = i + 1; print(f" Executing Block {block_num}/{len(codes)}...")
        try:
            outputs, success = await executor.run(code)
            print(f"  Success: {success}")
            if not success:
                final_success = False
                print(f"  Block {block_num} FAILED.")
                if executor.nb.cells and executor.nb.cells[-1].outputs:
                    for output in executor.nb.cells[-1].outputs:
                        if output.output_type == 'error': print(f"    Error: {output.ename}: {output.evalue}")
                break
        except Exception as e:
            print(f"  Exception during execution of block {block_num}: {e}"); traceback.print_exc(); final_success = False; break
    print(f"SELA Utils: Finished execution. Success: {final_success}"); return executor

def clean_json_from_rsp(text):
    if not isinstance(text, str): return ""
    pattern = r"```json(.*?)```"; matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        json_str = "".join(matches).strip()
        if (json_str.startswith('{') and json_str.endswith('}')) or (json_str.startswith('[') and json_str.endswith(']')): print("SELA Utils: Extracted JSON from ```json block."); return json_str
        else: print("SELA Utils Warn: Found ```json block but content invalid JSON."); return ""
    else: text_stripped = text.strip(); return text_stripped if (text_stripped.startswith('{') and text_stripped.endswith('}')) or (text_stripped.startswith('[') and text_stripped.endswith(']')) else ""
# --- End of Original Functions ---
