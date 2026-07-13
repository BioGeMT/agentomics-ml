"""
Dataset validator for UX-102.

Provides read-only validation of Agentomics datasets with structured, actionable
reports. Never mutates source data or creates persistent prepared artifacts.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from datasets import data_contract


class Severity(Enum):
    """Check result severity levels."""
    PASS = "PASS"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"


class OverallStatus(Enum):
    """Overall validation status."""
    VALID = "VALID"
    VALID_WITH_WARNINGS = "VALID WITH WARNINGS"
    INVALID = "INVALID"


@dataclass
class DatasetCheck:
    """Represents a single validation check result."""
    code: str  # Stable machine-readable identifier
    severity: Severity
    scope: str  # dataset | train | validation | test
    summary: str  # One-line result
    remediation: Optional[str] = None  # Optional next action
    path: Optional[str] = None  # Optional relevant path
    details: Optional[str] = None  # Optional sanitized details

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "code": self.code,
            "severity": self.severity.value,
            "scope": self.scope,
            "summary": self.summary,
        }
        if self.remediation:
            result["remediation"] = self.remediation
        if self.path:
            result["path"] = self.path
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class DatasetSummary:
    """Summary statistics about the dataset."""
    train_rows: Optional[int] = None
    validation_rows: Optional[int] = None
    test_rows: Optional[int] = None
    classification_classes: Optional[Dict[str, int]] = None  # class name -> count per split
    regression_range: Optional[Dict[str, tuple]] = None  # split -> (min, max)


@dataclass
class DatasetValidationReport:
    """Complete validation report for a dataset."""
    schema_version: int
    dataset_name: str
    public_path: str
    detected_layout: Optional[str] = None  # "flat_csv" or "folder"
    resolved_task_type: Optional[str] = None
    overall_status: OverallStatus = OverallStatus.INVALID
    summaries: DatasetSummary = field(default_factory=DatasetSummary)
    checks: List[DatasetCheck] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "public_path": self.public_path,
            "detected_layout": self.detected_layout,
            "resolved_task_type": self.resolved_task_type,
            "overall_status": self.overall_status.value,
            "summaries": {},
            "checks": [check.to_dict() for check in self.checks]
        }

        # Add summaries if present
        if self.summaries.train_rows is not None:
            result["summaries"]["train_rows"] = self.summaries.train_rows
        if self.summaries.validation_rows is not None:
            result["summaries"]["validation_rows"] = self.summaries.validation_rows
        if self.summaries.test_rows is not None:
            result["summaries"]["test_rows"] = self.summaries.test_rows
        if self.summaries.classification_classes:
            result["summaries"]["classification_classes"] = self.summaries.classification_classes
        if self.summaries.regression_range:
            result["summaries"]["regression_range"] = {
                split: {"min": r[0], "max": r[1]}
                for split, r in self.summaries.regression_range.items()
            }

        return result


class DatasetValidator:
    """Validates Agentomics datasets without mutation."""

    def __init__(
        self,
        dataset_name: str,
        datasets_root: Path,
        test_datasets_root: Optional[Path] = None,
        task_type: Optional[str] = None,
        verbose: bool = False
    ):
        self.dataset_name = dataset_name
        self.datasets_root = Path(datasets_root)
        self.test_datasets_root = Path(test_datasets_root) if test_datasets_root else None
        self.requested_task_type = task_type
        self.verbose = verbose
        self.checks: List[DatasetCheck] = []
        self.report = DatasetValidationReport(
            schema_version=1,
            dataset_name=dataset_name,
            public_path=str(self.datasets_root / dataset_name)
        )

    def add_check(self, check: DatasetCheck):
        """Add a check to the report."""
        self.checks.append(check)
        self.report.checks.append(check)

    def validate(self) -> DatasetValidationReport:
        """Run all validation checks and return a report."""
        # Check 1: Dataset name safety
        if not self._check_dataset_name_safe():
            # Fatal - don't continue
            self._finalize_report()
            return self.report

        # Check 2: Dataset exists
        dataset_path = self.datasets_root / self.dataset_name
        if not self._check_dataset_exists(dataset_path):
            self._finalize_report()
            return self.report

        # Check 3: Detect layout
        layout = self._detect_layout(dataset_path)
        if layout is None:
            self._finalize_report()
            return self.report

        self.report.detected_layout = layout

        # Check 4: Validate based on layout
        if layout == "flat_csv":
            self._validate_flat_csv_dataset(dataset_path)
        else:  # folder
            self._validate_folder_dataset(dataset_path)

        # Check 5: Hidden test if present
        if self.test_datasets_root:
            test_dataset_path = self.test_datasets_root / self.dataset_name / data_contract.TEST_SPLIT
            if test_dataset_path.exists():
                self._validate_hidden_test(test_dataset_path, dataset_path)
            else:
                self.add_check(DatasetCheck(
                    code="HIDDEN_TEST_MISSING",
                    severity=Severity.INFO,
                    scope="dataset",
                    summary="No hidden test set found",
                    details=f"No test data in {self.test_datasets_root / self.dataset_name}"
                ))

        # Finalize
        self._finalize_report()
        return self.report

    def _check_dataset_name_safe(self) -> bool:
        """Check for path traversal and absolute paths."""
        if ".." in self.dataset_name or self.dataset_name.startswith("/"):
            self.add_check(DatasetCheck(
                code="DATASET_NAME_UNSAFE",
                severity=Severity.ERROR,
                scope="dataset",
                summary=f"Dataset name contains path traversal or absolute path: {self.dataset_name}",
                remediation="Use a simple directory name without '..' or '/'."
            ))
            return False

        self.add_check(DatasetCheck(
            code="DATASET_NAME_SAFE",
            severity=Severity.PASS,
            scope="dataset",
            summary="Dataset name is safe"
        ))
        return True

    def _check_dataset_exists(self, dataset_path: Path) -> bool:
        """Check if dataset directory exists."""
        if not dataset_path.exists():
            self.add_check(DatasetCheck(
                code="DATASET_NOT_FOUND",
                severity=Severity.ERROR,
                scope="dataset",
                summary=f"Dataset directory not found: {dataset_path}",
                remediation=f"Create the dataset directory or check the name. Available datasets: {self._list_available_datasets()}"
            ))
            return False

        if not dataset_path.is_dir():
            self.add_check(DatasetCheck(
                code="DATASET_NOT_DIRECTORY",
                severity=Severity.ERROR,
                scope="dataset",
                summary=f"Dataset path is not a directory: {dataset_path}",
                remediation="Dataset path must be a directory."
            ))
            return False

        self.add_check(DatasetCheck(
            code="DATASET_EXISTS",
            severity=Severity.PASS,
            scope="dataset",
            summary=f"Dataset directory found",
            path=str(dataset_path)
        ))
        return True

    def _list_available_datasets(self) -> str:
        """List available datasets in the datasets root."""
        try:
            datasets = [d.name for d in self.datasets_root.iterdir() if d.is_dir()]
            if datasets:
                return ", ".join(sorted(datasets[:10]))
            else:
                return "none found"
        except Exception:
            return "unable to list"

    def _detect_layout(self, dataset_path: Path) -> Optional[str]:
        """Detect dataset layout: flat_csv or folder."""
        # Check for flat CSV layout
        has_train_csv = (dataset_path / "train.csv").exists()
        has_train_folder = (dataset_path / data_contract.TRAIN_SPLIT).exists()

        if has_train_csv and has_train_folder:
            self.add_check(DatasetCheck(
                code="LAYOUT_AMBIGUOUS",
                severity=Severity.ERROR,
                scope="dataset",
                summary="Dataset has both train.csv and train/ folder - layout is ambiguous",
                remediation="Remove either train.csv (for folder layout) or train/ (for flat CSV layout)."
            ))
            return None
        elif has_train_csv:
            self.add_check(DatasetCheck(
                code="LAYOUT_DETECTED",
                severity=Severity.PASS,
                scope="dataset",
                summary="Detected layout: flat CSV"
            ))
            return "flat_csv"
        elif has_train_folder:
            self.add_check(DatasetCheck(
                code="LAYOUT_DETECTED",
                severity=Severity.PASS,
                scope="dataset",
                summary="Detected layout: folder"
            ))
            return "folder"
        else:
            self.add_check(DatasetCheck(
                code="LAYOUT_UNKNOWN",
                severity=Severity.ERROR,
                scope="dataset",
                summary="No train.csv or train/ folder found",
                remediation="Create either train.csv (flat CSV) or train/ folder (folder layout). See docs/user-guide/datasets.md"
            ))
            return None

    def _validate_folder_dataset(self, dataset_path: Path):
        """Validate folder-based dataset."""
        # Check public dataset structure
        try:
            data_contract.validate_public_dataset_entries(dataset_path)
            self.add_check(DatasetCheck(
                code="PUBLIC_STRUCTURE_VALID",
                severity=Severity.PASS,
                scope="dataset",
                summary="Public dataset structure is valid"
            ))
        except ValueError as e:
            self.add_check(DatasetCheck(
                code="PUBLIC_STRUCTURE_INVALID",
                severity=Severity.ERROR,
                scope="dataset",
                summary=str(e),
                remediation="Remove unsupported top-level entries or move test data to test_datasets/"
            ))
            return  # Fatal

        # Check for symlinks
        if self._check_for_symlinks(dataset_path):
            return  # Fatal

        # Validate train split (required)
        train_path = dataset_path / data_contract.TRAIN_SPLIT
        if not train_path.exists():
            self.add_check(DatasetCheck(
                code="TRAIN_MISSING",
                severity=Severity.ERROR,
                scope="train",
                summary="Required train/ split not found",
                remediation=f"Create {train_path} with input/ and labels.csv"
            ))
            return

        # Validate train split structure
        train_labels, train_input_structure = self._validate_split_structure(
            train_path, "train", is_train=True
        )
        if train_labels is None or train_input_structure is None:
            return  # Errors already added

        # Validate validation split if present
        validation_path = dataset_path / data_contract.VALIDATION_SPLIT
        validation_labels = None
        if validation_path.exists():
            validation_labels, _ = self._validate_split_structure(
                validation_path, "validation",
                expected_input_structure=train_input_structure
            )
        else:
            self.add_check(DatasetCheck(
                code="VALIDATION_MISSING",
                severity=Severity.WARN,
                scope="dataset",
                summary="No validation/ split found",
                details="Agentomics will create a validation split from training data"
            ))

        # Check train/validation ID overlap (BLOCKER)
        if validation_labels is not None and train_labels is not None:
            self._check_id_overlap(train_labels, validation_labels)

        # Task type and label validation
        self._validate_task_type_and_labels(dataset_path, train_labels, validation_labels, None)

        # Informational checks
        self._add_dataset_description_check(dataset_path)

    def _validate_flat_csv_dataset(self, dataset_path: Path):
        """Validate flat CSV dataset."""
        self.add_check(DatasetCheck(
            code="FLAT_CSV_DETECTED",
            severity=Severity.INFO,
            scope="dataset",
            summary="Flat CSV layout - will be converted to folder layout during preparation"
        ))

        # Try to parse train.csv
        try:
            train_csv = dataset_path / "train.csv"
            df = pd.read_csv(train_csv, nrows=1)  # Just check header

            # Check if metadata has label_column
            metadata_path = dataset_path / data_contract.METADATA_FILE_NAME
            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    label_column = metadata.get("label_column")
                    if label_column:
                        if label_column in df.columns:
                            self.add_check(DatasetCheck(
                                code="FLAT_CSV_VALID",
                                severity=Severity.PASS,
                                scope="train",
                                summary=f"Flat CSV appears valid with label column: {label_column}"
                            ))
                        else:
                            self.add_check(DatasetCheck(
                                code="FLAT_CSV_LABEL_COLUMN_MISSING",
                                severity=Severity.ERROR,
                                scope="train",
                                summary=f"label_column '{label_column}' not found in train.csv",
                                remediation=f"Available columns: {list(df.columns)}"
                            ))
                    else:
                        self.add_check(DatasetCheck(
                            code="FLAT_CSV_NO_LABEL_COLUMN",
                            severity=Severity.WARN,
                            scope="train",
                            summary="No label_column in metadata.json",
                            remediation="Add label_column to metadata.json or run interactively"
                        ))
                except Exception:
                    pass
            else:
                self.add_check(DatasetCheck(
                    code="FLAT_CSV_NO_METADATA",
                    severity=Severity.WARN,
                    scope="train",
                    summary="No metadata.json found for flat CSV dataset",
                    remediation="Add metadata.json with task_type and label_column"
                ))

        except Exception as e:
            self.add_check(DatasetCheck(
                code="FLAT_CSV_PARSE_ERROR",
                severity=Severity.ERROR,
                scope="train",
                summary=f"Failed to parse train.csv: {str(e)}",
                remediation="Ensure train.csv is a valid CSV file"
            ))

    def _validate_split_structure(
        self,
        split_path: Path,
        split_name: str,
        expected_input_structure: Optional[List[str]] = None,
        is_train: bool = False
    ) -> tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """Validate a single split's structure and return labels + input structure."""

        # Check split folder entries
        try:
            data_contract.validate_split_entries(split_path, split_name)
            self.add_check(DatasetCheck(
                code=f"{split_name.upper()}_STRUCTURE_VALID",
                severity=Severity.PASS,
                scope=split_name,
                summary=f"{split_name}/ structure is valid"
            ))
        except ValueError as e:
            self.add_check(DatasetCheck(
                code=f"{split_name.upper()}_STRUCTURE_INVALID",
                severity=Severity.ERROR,
                scope=split_name,
                summary=str(e),
                remediation=f"Ensure {split_name}/ contains only input/ and labels.csv"
            ))
            return None, None

        # Validate input/ exists and has content
        input_path = split_path / data_contract.INPUT_DIR_NAME
        if not input_path.exists() or not list(input_path.iterdir()):
            self.add_check(DatasetCheck(
                code=f"{split_name.upper()}_INPUT_EMPTY",
                severity=Severity.ERROR,
                scope=split_name,
                summary=f"{split_name}/input/ is empty or missing",
                remediation=f"Add data files to {split_name}/input/"
            ))
            return None, None

        # Record or validate input structure
        current_structure = data_contract.record_input_dir_structure(input_path)
        if is_train:
            self.add_check(DatasetCheck(
                code="TRAIN_INPUT_STRUCTURE_RECORDED",
                severity=Severity.PASS,
                scope="train",
                summary=f"train/input/ structure recorded: {len(current_structure)} top-level entries"
            ))
            input_structure = current_structure
        else:
            # Validate against train structure
            if current_structure != expected_input_structure:
                self.add_check(DatasetCheck(
                    code=f"{split_name.upper()}_INPUT_MISMATCH",
                    severity=Severity.ERROR,
                    scope=split_name,
                    summary=f"{split_name}/input/ structure doesn't match train/input/",
                    remediation="Ensure all splits have identical top-level input/ structure",
                    details=f"Expected: {expected_input_structure}, Got: {current_structure}"
                ))
                return None, expected_input_structure
            else:
                self.add_check(DatasetCheck(
                    code=f"{split_name.upper()}_INPUT_MATCHES",
                    severity=Severity.PASS,
                    scope=split_name,
                    summary=f"{split_name}/input/ structure matches train/input/"
                ))
            input_structure = expected_input_structure

        # Validate labels.csv
        labels_path = split_path / data_contract.LABELS_FILE_NAME
        try:
            # Read labels WITHOUT requiring numeric values (let task type determine that)
            labels = data_contract.validate_and_read_labels(
                labels_path,
                data_contract.LABEL_COLUMN_NAME,
                require_numeric_values=False
            )
            self.add_check(DatasetCheck(
                code=f"{split_name.upper()}_LABELS_VALID",
                severity=Severity.PASS,
                scope=split_name,
                summary=f"{split_name}/labels.csv is valid ({len(labels)} rows)"
            ))
            return labels, input_structure
        except ValueError as e:
            self.add_check(DatasetCheck(
                code=f"{split_name.upper()}_LABELS_INVALID",
                severity=Severity.ERROR,
                scope=split_name,
                summary=str(e),
                remediation=f"Fix {split_name}/labels.csv format. Must have exactly 'id,label' columns with unique non-empty IDs."
            ))
            return None, input_structure

    def _check_id_overlap(self, train_labels: pd.DataFrame, validation_labels: pd.DataFrame):
        """Check for train/validation ID overlap - BLOCKER."""
        train_ids = set(train_labels[data_contract.ID_COLUMN_NAME].str.strip())
        val_ids = set(validation_labels[data_contract.ID_COLUMN_NAME].str.strip())

        overlap = train_ids & val_ids
        if overlap:
            overlap_count = len(overlap)
            overlap_sample = sorted(list(overlap))[:10]

            self.add_check(DatasetCheck(
                code="DATASET_ID_OVERLAP",
                severity=Severity.ERROR,
                scope="dataset",
                summary=f"{overlap_count} IDs appear in both train and validation",
                remediation="Ensure train and validation IDs are completely disjoint. Split by independent unit (patient, family, batch) to prevent data leakage.",
                details=f"Sample overlapping IDs: {overlap_sample}" if self.verbose else None
            ))
        else:
            self.add_check(DatasetCheck(
                code="DATASET_ID_DISJOINT",
                severity=Severity.PASS,
                scope="dataset",
                summary="Train and validation IDs are disjoint"
            ))

    def _validate_task_type_and_labels(
        self,
        dataset_path: Path,
        train_labels: Optional[pd.DataFrame],
        validation_labels: Optional[pd.DataFrame],
        test_labels: Optional[pd.DataFrame]
    ):
        """Validate task type and label consistency."""
        if train_labels is None:
            return

        # Try to load metadata
        metadata_path = dataset_path / data_contract.METADATA_FILE_NAME
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                self.add_check(DatasetCheck(
                    code="METADATA_VALID",
                    severity=Severity.PASS,
                    scope="dataset",
                    summary="metadata.json is valid JSON"
                ))
            except json.JSONDecodeError as e:
                self.add_check(DatasetCheck(
                    code="METADATA_INVALID",
                    severity=Severity.ERROR,
                    scope="dataset",
                    summary=f"metadata.json is invalid JSON: {e}",
                    remediation="Fix JSON syntax in metadata.json"
                ))
                return
        else:
            if not self.requested_task_type:
                self.add_check(DatasetCheck(
                    code="TASK_TYPE_UNRESOLVED",
                    severity=Severity.WARN,
                    scope="dataset",
                    summary="No metadata.json and no --task-type specified",
                    remediation="Add metadata.json with task_type or use --task-type flag"
                ))
                return

        # Resolve task type
        task_type = self.requested_task_type or metadata.get("task_type")
        if not task_type:
            self.add_check(DatasetCheck(
                code="TASK_TYPE_MISSING",
                severity=Severity.WARN,
                scope="dataset",
                summary="Task type not specified",
                remediation="Add task_type to metadata.json or use --task-type flag"
            ))
            return

        self.report.resolved_task_type = task_type

        # Validate based on task type
        if task_type == "regression":
            self._validate_regression_labels(train_labels, validation_labels, test_labels)
        elif task_type == "classification":
            self._validate_classification_labels(train_labels, validation_labels, test_labels, metadata)
        else:
            self.add_check(DatasetCheck(
                code="TASK_TYPE_INVALID",
                severity=Severity.ERROR,
                scope="dataset",
                summary=f"Invalid task type: {task_type}",
                remediation="task_type must be 'classification' or 'regression'"
            ))

    def _validate_regression_labels(
        self,
        train_labels: pd.DataFrame,
        validation_labels: Optional[pd.DataFrame],
        test_labels: Optional[pd.DataFrame]
    ):
        """Validate regression labels are numeric and finite."""
        for split_name, labels in [("train", train_labels), ("validation", validation_labels), ("test", test_labels)]:
            if labels is None:
                continue

            values = pd.to_numeric(labels[data_contract.LABEL_COLUMN_NAME], errors="coerce")

            # Check non-numeric
            non_numeric_mask = values.isna()
            if non_numeric_mask.any():
                non_numeric_sample = labels.loc[non_numeric_mask, data_contract.LABEL_COLUMN_NAME].head(5).tolist()
                self.add_check(DatasetCheck(
                    code=f"{split_name.upper()}_LABELS_NON_NUMERIC",
                    severity=Severity.ERROR,
                    scope=split_name,
                    summary=f"{split_name} has {non_numeric_mask.sum()} non-numeric label values",
                    remediation=f"Regression requires numeric labels. Sample: {non_numeric_sample}"
                ))
                continue

            # Check non-finite
            non_finite_mask = values.isin([float("inf"), float("-inf")])
            if non_finite_mask.any():
                self.add_check(DatasetCheck(
                    code=f"{split_name.upper()}_LABELS_NON_FINITE",
                    severity=Severity.ERROR,
                    scope=split_name,
                    summary=f"{split_name} has {non_finite_mask.sum()} non-finite label values (inf/-inf)",
                    remediation="Replace infinite values with finite numbers"
                ))
                continue

            # Store summary
            if split_name == "train":
                self.report.summaries.train_rows = len(labels)
            elif split_name == "validation":
                self.report.summaries.validation_rows = len(labels)
            elif split_name == "test":
                self.report.summaries.test_rows = len(labels)

            self.add_check(DatasetCheck(
                code=f"{split_name.upper()}_REGRESSION_VALID",
                severity=Severity.PASS,
                scope=split_name,
                summary=f"{split_name} has {len(labels)} regression labels, range: [{values.min():.2f}, {values.max():.2f}]"
            ))

    def _validate_classification_labels(
        self,
        train_labels: pd.DataFrame,
        validation_labels: Optional[pd.DataFrame],
        test_labels: Optional[pd.DataFrame],
        metadata: dict
    ):
        """Validate classification labels."""
        # Get unique classes from train
        train_classes = set(train_labels[data_contract.LABEL_COLUMN_NAME].str.strip().unique())

        if len(train_classes) < 2:
            self.add_check(DatasetCheck(
                code="CLASSIFICATION_TOO_FEW_CLASSES",
                severity=Severity.ERROR,
                scope="train",
                summary=f"Training data has only {len(train_classes)} unique class(es)",
                remediation="Classification requires at least 2 classes"
            ))
            return

        self.report.summaries.train_rows = len(train_labels)

        # Check validation classes are known
        if validation_labels is not None:
            val_classes = set(validation_labels[data_contract.LABEL_COLUMN_NAME].str.strip().unique())
            unknown_val_classes = val_classes - train_classes
            if unknown_val_classes:
                self.add_check(DatasetCheck(
                    code="VALIDATION_UNKNOWN_CLASSES",
                    severity=Severity.WARN,
                    scope="validation",
                    summary=f"Validation has {len(unknown_val_classes)} class(es) not in training: {sorted(unknown_val_classes)}",
                    details="These samples cannot be properly evaluated"
                ))
            self.report.summaries.validation_rows = len(validation_labels)

        # Check test classes are known
        if test_labels is not None:
            test_classes = set(test_labels[data_contract.LABEL_COLUMN_NAME].str.strip().unique())
            unknown_test_classes = test_classes - train_classes
            if unknown_test_classes:
                self.add_check(DatasetCheck(
                    code="TEST_UNKNOWN_CLASSES",
                    severity=Severity.ERROR,
                    scope="test",
                    summary=f"Test has {len(unknown_test_classes)} class(es) not in training: {sorted(unknown_test_classes)}",
                    remediation="Test classes must be present in training data"
                ))
            self.report.summaries.test_rows = len(test_labels)

        # Check binary classification metadata
        if len(train_classes) == 2:
            pos_class = metadata.get("positive_class")
            neg_class = metadata.get("negative_class")

            if pos_class and neg_class:
                if pos_class not in train_classes or neg_class not in train_classes:
                    self.add_check(DatasetCheck(
                        code="BINARY_CLASS_MISMATCH",
                        severity=Severity.ERROR,
                        scope="dataset",
                        summary=f"positive_class/negative_class in metadata don't match training classes",
                        remediation=f"Training classes: {sorted(train_classes)}, Metadata: positive='{pos_class}', negative='{neg_class}'"
                    ))
            elif pos_class or neg_class:
                self.add_check(DatasetCheck(
                    code="BINARY_CLASS_INCOMPLETE",
                    severity=Severity.ERROR,
                    scope="dataset",
                    summary="Both positive_class and negative_class must be specified together",
                    remediation="Either specify both or remove both from metadata.json"
                ))

        self.add_check(DatasetCheck(
            code="CLASSIFICATION_VALID",
            severity=Severity.PASS,
            scope="train",
            summary=f"Training has {len(train_labels)} samples across {len(train_classes)} classes: {sorted(train_classes)}"
        ))

    def _validate_hidden_test(self, test_path: Path, public_dataset_path: Path):
        """Validate hidden test split structure."""
        # Validate test structure
        train_input_structure = data_contract.record_input_dir_structure(
            public_dataset_path / data_contract.TRAIN_SPLIT / data_contract.INPUT_DIR_NAME
        )

        test_labels, _ = self._validate_split_structure(
            test_path, "test",
            expected_input_structure=train_input_structure
        )

        if test_labels is not None:
            self.add_check(DatasetCheck(
                code="HIDDEN_TEST_VALID",
                severity=Severity.PASS,
                scope="test",
                summary=f"Hidden test set is valid ({len(test_labels)} samples)"
            ))

    def _check_for_symlinks(self, dataset_path: Path) -> bool:
        """Check for symlinks in dataset (security policy)."""
        symlinks = []
        for item in dataset_path.rglob("*"):
            if item.is_symlink():
                symlinks.append(str(item.relative_to(dataset_path)))
                if len(symlinks) >= 10:
                    break

        if symlinks:
            self.add_check(DatasetCheck(
                code="SYMLINKS_DETECTED",
                severity=Severity.ERROR,
                scope="dataset",
                summary=f"Dataset contains {len(symlinks)} symlink(s)",
                remediation="Replace symlinks with real files. Symlinks are rejected for security.",
                details=f"Sample symlinks: {symlinks[:5]}"
            ))
            return True

        return False

    def _add_dataset_description_check(self, dataset_path: Path):
        """Check for dataset_description.md."""
        desc_path = dataset_path / data_contract.DATASET_DESCRIPTION_FILE_NAME
        if not desc_path.exists():
            self.add_check(DatasetCheck(
                code="DATASET_DESCRIPTION_MISSING",
                severity=Severity.WARN,
                scope="dataset",
                summary="No dataset_description.md found",
                details="A description helps the agent understand your data. See docs/user-guide/dataset-best-practices.md"
            ))
        else:
            self.add_check(DatasetCheck(
                code="DATASET_DESCRIPTION_PRESENT",
                severity=Severity.PASS,
                scope="dataset",
                summary="dataset_description.md found"
            ))

        # Add group leakage warning
        self.add_check(DatasetCheck(
            code="GROUP_LEAKAGE_NOT_CHECKED",
            severity=Severity.UNKNOWN,
            scope="dataset",
            summary="Group/batch/family leakage is not automatically checked",
            details="If samples are related (same patient, family, batch), ensure they're in the same split. See docs/user-guide/dataset-best-practices.md"
        ))

    def _finalize_report(self):
        """Finalize the validation report with overall status."""
        has_error = any(c.severity == Severity.ERROR for c in self.checks)
        has_warn = any(c.severity == Severity.WARN for c in self.checks)

        if has_error:
            self.report.overall_status = OverallStatus.INVALID
        elif has_warn:
            self.report.overall_status = OverallStatus.VALID_WITH_WARNINGS
        else:
            self.report.overall_status = OverallStatus.VALID


def render_human_output(report: DatasetValidationReport) -> str:
    """Render validation report in human-readable format."""
    lines = []
    lines.append(f"Dataset check: {report.dataset_name}")
    if report.detected_layout:
        lines.append(f"Layout: {report.detected_layout}")
    if report.resolved_task_type:
        lines.append(f"Task: {report.resolved_task_type}")
    lines.append("")

    for check in report.checks:
        # Severity label with padding
        severity_label = f"{check.severity.value:7}"
        scope_label = f"[{check.scope}]" if check.scope != "dataset" else ""

        lines.append(f"{severity_label} {scope_label} {check.summary}".strip())

        if check.remediation:
            for remediation_line in check.remediation.split('\n'):
                lines.append(f"         {remediation_line}")

        if check.details:
            lines.append(f"         Details: {check.details}")

    lines.append("")

    # Count by severity
    error_count = sum(1 for c in report.checks if c.severity == Severity.ERROR)
    warn_count = sum(1 for c in report.checks if c.severity == Severity.WARN)
    unknown_count = sum(1 for c in report.checks if c.severity == Severity.UNKNOWN)

    result_text = f"Result: {report.overall_status.value}"
    issues = []
    if error_count > 0:
        issues.append(f"{error_count} error{'s' if error_count != 1 else ''}")
    if warn_count > 0:
        issues.append(f"{warn_count} warning{'s' if warn_count != 1 else ''}")
    if unknown_count > 0:
        issues.append(f"{unknown_count} unknown")

    if issues:
        result_text += f" ({', '.join(issues)})"

    lines.append(result_text)
    lines.append("No agent or provider call was started.")

    return "\n".join(lines)


def render_json_output(report: DatasetValidationReport) -> str:
    """Render validation report as JSON."""
    return json.dumps(report.to_dict(), indent=2)


def validate_dataset(
    dataset_name: str,
    datasets_root: Path,
    test_datasets_root: Optional[Path] = None,
    task_type: Optional[str] = None,
    verbose: bool = False,
    json_output: bool = False
) -> tuple[str, int]:
    """
    Validate a dataset and return (output_string, exit_code).

    Exit codes:
        0: valid (including warnings/unknown)
        1: invalid (has errors)
    """
    validator = DatasetValidator(
        dataset_name=dataset_name,
        datasets_root=datasets_root,
        test_datasets_root=test_datasets_root,
        task_type=task_type,
        verbose=verbose
    )

    report = validator.validate()

    if json_output:
        output = render_json_output(report)
    else:
        output = render_human_output(report)

    exit_code = 0 if report.overall_status != OverallStatus.INVALID else 1

    return output, exit_code
