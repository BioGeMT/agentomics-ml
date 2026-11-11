"""Prompt fragments for the temporal generation task (Virtual Embryo Challenge, Task 1)."""
from agentomics.datasets.conditional_generation_contract import (
    CONTROLS_FILE_NAME,
    N_CELLS_COLUMN_NAME,
    PERTURBATIONS_FILE_NAME,
)
from agentomics.datasets.data_contract import GENERATION_LABELS_FILE_NAME, INPUT_DIR_NAME
from agentomics.datasets.temporal_generation_contract import MAX_LOG_EXPRESSION, STAGE_COLUMN_NAME

MINI_TRAIN_CELLS_PER_STAGE = 1000

SAMPLE_IDENTITY = "target stages"


def describe_split_folder() -> str:
    """Completes the sentence 'Each split folder ...'."""
    return (
        f"contains an input/ folder with {CONTROLS_FILE_NAME} (the observed cells of the earlier developmental stages: "
        f"log1p-normalised expression, obs '{STAGE_COLUMN_NAME}' e.g. 'E8.5'; the source cell-type annotation may be in "
        f"obs 'celltype') and {PERTURBATIONS_FILE_NAME} (the target stages to generate cells for: {STAGE_COLUMN_NAME}, "
        f"{N_CELLS_COLUMN_NAME}) and {GENERATION_LABELS_FILE_NAME} (the measured cells of every target stage: "
        f"log1p-normalised expression, obs '{STAGE_COLUMN_NAME}'). Despite the file names, there are no controls or "
        "perturbations: the condition is the developmental stage."
    )


def task_description() -> str:
    return f"""    - Task: given the observed cells of earlier mouse embryonic stages, generate the cells of a later, unobserved stage (temporal forecasting of the single-cell expression distribution).
    - The final model is given every observed stage and must forecast later stages that no split covers, further in the future than any transition you can validate on. Expect the inference input to contain more observed stages than your training input, and a target stage further away. The model must use the stage values (embryonic days) of its input, not hard-coded stage names.
    - Predictions are sets of cells scored as populations against the measured cells of the target stage: differential expression and change direction against the latest observed stage, plus the cell-state distribution (MMD in PCA space) and gene-gene co-variation. A model must therefore generate a realistic population with progenitors, differentiating and terminal cell types in plausible proportions, not an average cell; repeating one profile scores below simply copying the latest observed stage.
    - Later stages contain many cell types absent from earlier stages (consecutive stages can share only a few annotated types), so a model that only reweights or shifts the existing cell types is limited.
    - Expression is log1p-normalised (not raw counts), over the whole transcriptome. Predictions must stay on that scale: finite, non-negative, at most {MAX_LOG_EXPRESSION}. Keep matrices sparse where possible; the dense matrix of one stage is several GB.
    - External public data and published methods may be used, but must be disclosed in your outputs.
"""


def split_files_instructions() -> str:
    return (
        f"Every split holds {GENERATION_LABELS_FILE_NAME}. For each split, write {INPUT_DIR_NAME}/{PERTURBATIONS_FILE_NAME} "
        f"with that split's target stages and their {N_CELLS_COLUMN_NAME}, {GENERATION_LABELS_FILE_NAME} with the measured "
        f"cells of those target stages, and {INPUT_DIR_NAME}/{CONTROLS_FILE_NAME} with the observed cells the model is given. "
        "Every target stage must be later than every observed stage of its split. A stage that is a target in one split "
        f"can be observed input in another (e.g. train: E8.5 -> E9.5; validation: E8.5 + E9.5 -> E10.5). Keep the "
        f"'{STAGE_COLUMN_NAME}' obs column and the full gene axis in its original order, and do not rescale the expression."
    )


def validation_instructions() -> str:
    return (
        "Split by target stage, never by cell. The model is judged on forecasting a later stage from all earlier ones, "
        "so the validation split should forecast the latest available stage from the stages before it."
    )


def mini_train_size_instructions() -> str:
    return (
        f"the target stages of the training split, each with at most {MINI_TRAIN_CELLS_PER_STAGE} randomly sampled measured "
        f"cells in {GENERATION_LABELS_FILE_NAME} and {N_CELLS_COLUMN_NAME} set to that number, and at most "
        f"{MINI_TRAIN_CELLS_PER_STAGE} observed cells per stage in {INPUT_DIR_NAME}/{CONTROLS_FILE_NAME}"
    )


def coverage_requirement() -> str:
    return (
        f"The inference script must generate cells for every target stage listed in {INPUT_DIR_NAME}/{PERTURBATIONS_FILE_NAME}. "
        "Don't skip any stages."
    )


def generated_cells_description() -> str:
    return f"""An h5ad file (AnnData) of generated cells:
        - For every row of {INPUT_DIR_NAME}/{PERTURBATIONS_FILE_NAME}, exactly '{N_CELLS_COLUMN_NAME}' generated cells (one row each), with obs column '{STAGE_COLUMN_NAME}' set to that row's stage.
        - Genes (var_names): those of {INPUT_DIR_NAME}/{CONTROLS_FILE_NAME}, in the same order (the whole transcriptome).
        - X: log1p-normalised expression like the inputs: finite, non-negative, at most {MAX_LOG_EXPRESSION}. Store X as a scipy sparse matrix (CSR). Predictions must not contain NaN or infinity.
        - Sample each cell from the predicted distribution of the target stage, so the cells vary like real cells do. An average cell, or one profile repeated, breaks the distribution and co-variation metrics."""
