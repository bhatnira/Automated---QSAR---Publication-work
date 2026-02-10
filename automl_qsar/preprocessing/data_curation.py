"""
QSAR Data Curation Agent
Statistically-Driven Log Transformation and Data Quality Control

This module provides automated data curation for QSAR datasets including:
- SMILES validation and canonicalization
- Duplicate removal
- Activity value extraction and cleaning
- Task inference (regression/classification)
- Statistical-driven log transformation decision
- Censored data handling
"""

import re
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from scipy.stats import shapiro, skew
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# Utility Functions
# ============================================================

def parse_smiles(smiles: str) -> Optional[object]:
    """
    Parse a SMILES string into an RDKit molecule object.
    
    Args:
        smiles: SMILES string
        
    Returns:
        RDKit Mol object or None if parsing fails
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES parsing")
    
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        return mol
    except:
        return None


def canonicalize_smiles(mol) -> Optional[str]:
    """
    Convert RDKit molecule to canonical SMILES.
    
    Args:
        mol: RDKit Mol object
        
    Returns:
        Canonical SMILES string or None
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for SMILES canonicalization")
    
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


def extract_numeric_activity(val: Any) -> Tuple[Optional[float], Optional[str]]:
    """
    Extract numeric value and censoring info from activity data.
    
    Handles formats like:
    - "100" -> (100.0, None)
    - ">100" -> (100.0, "right")
    - "<10" -> (10.0, "left")
    - "~50" -> (50.0, None)
    
    Args:
        val: Activity value (string or numeric)
        
    Returns:
        Tuple of (numeric_value, censoring_type)
        censoring_type is None, "left" (<), or "right" (>)
    """
    if pd.isna(val):
        return None, None

    s = str(val).strip()
    censor = None

    # Check for censoring indicators
    if s.startswith(">") or s.startswith("≥"):
        censor = "right"
        s = s[1:]
    elif s.startswith("<") or s.startswith("≤"):
        censor = "left"
        s = s[1:]
    elif s.startswith("~") or s.startswith("≈"):
        s = s[1:]

    # Remove non-numeric characters except decimal point and scientific notation
    s = re.sub(r"[^\d\.eE\-\+]", "", s)

    try:
        v = float(s)
        if v <= 0:
            return None, censor
        return v, censor
    except:
        return None, censor


def convert_to_nM(value: float, unit: str) -> Optional[float]:
    """
    Convert activity value to nanomolar (nM).
    
    Args:
        value: Activity value
        unit: Unit string (nM, uM, µM, mM, M, pM)
        
    Returns:
        Value in nM or None if unit not recognized
    """
    if value is None:
        return None
        
    unit = str(unit).lower().strip()
    
    conversions = {
        'nm': 1.0,
        'um': 1e3,
        'µm': 1e3,
        'μm': 1e3,
        'mm': 1e6,
        'm': 1e9,
        'pm': 1e-3,
        'fm': 1e-6,
    }
    
    factor = conversions.get(unit)
    if factor is None:
        return None
    
    return value * factor


def calculate_molecular_weight(mol) -> Optional[float]:
    """Calculate molecular weight from RDKit mol."""
    if not RDKIT_AVAILABLE or mol is None:
        return None
    try:
        return rdMolDescriptors.CalcExactMolWt(mol)
    except:
        return None


# ============================================================
# Main QSAR Data Curation Agent
# ============================================================

class QSARDataCurationAgent:
    """
    Automated QSAR Data Curation Pipeline
    
    Performs:
    1. SMILES validation and canonicalization
    2. Duplicate molecule removal
    3. Activity value extraction and cleaning
    4. Task type inference (regression/classification)
    5. Statistical-driven log transformation
    6. Censored data handling
    7. Report generation
    
    Args:
        df: Input DataFrame with SMILES and activity data
        smiles_col: Name of column containing SMILES strings
        activity_col: Name of column containing activity values
        unit: Unit of activity values (default: "nM")
        verbose: Print progress messages (default: True)
        
    Example:
        >>> df = pd.read_csv("dataset.csv")
        >>> agent = QSARDataCurationAgent(
        ...     df=df,
        ...     smiles_col="smiles",
        ...     activity_col="IC50",
        ...     unit="nM"
        ... )
        >>> clean_df = agent.run()
        >>> agent.generate_report("qsar_report.txt")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        activity_col: str,
        unit: str = "nM",
        verbose: bool = True
    ):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for QSARDataCurationAgent")
        
        self.raw_df = df.copy()
        self.smiles_col = smiles_col
        self.activity_col = activity_col
        self.unit = unit
        self.verbose = verbose

        self.report = []
        self.cleaned_df = None
        self.task = None
        self.statistics = {}
        
        # Validate input columns exist
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in DataFrame")
        if activity_col not in df.columns:
            raise ValueError(f"Activity column '{activity_col}' not found in DataFrame")

    def log(self, text: str):
        """Add message to report and optionally print."""
        self.report.append(text)
        if self.verbose:
            print(f"  {text}")

    def validate_smiles(self):
        """Validate SMILES strings and identify invalid molecules."""
        mols = []
        valid = []

        for s in self.raw_df[self.smiles_col]:
            mol = parse_smiles(s)
            if mol is None:
                mols.append(None)
                valid.append(False)
            else:
                mols.append(mol)
                valid.append(True)

        self.raw_df["_Mol"] = mols
        self.raw_df["_Valid_SMILES"] = valid

        n_invalid = valid.count(False)
        n_valid = valid.count(True)
        
        self.statistics['n_invalid_smiles'] = n_invalid
        self.statistics['n_valid_smiles'] = n_valid
        
        self.log(f"SMILES validation: {n_valid} valid, {n_invalid} invalid")

    def standardize_smiles(self):
        """Canonicalize SMILES and remove duplicates."""
        # Filter to valid SMILES only
        self.raw_df = self.raw_df[self.raw_df["_Valid_SMILES"]].copy()

        # Canonicalize
        canon = []
        for mol in self.raw_df["_Mol"]:
            canon.append(canonicalize_smiles(mol))

        self.raw_df["Canonical_SMILES"] = canon
        
        # Calculate molecular weights
        mw = []
        for mol in self.raw_df["_Mol"]:
            mw.append(calculate_molecular_weight(mol))
        self.raw_df["_MW"] = mw
        
        # Remove duplicates
        before = len(self.raw_df)
        self.raw_df.drop_duplicates("Canonical_SMILES", inplace=True)
        after = len(self.raw_df)
        
        n_duplicates = before - after
        self.statistics['n_duplicates'] = n_duplicates
        
        self.log(f"Duplicate molecules removed: {n_duplicates}")

    def clean_activity(self):
        """Extract and clean activity values."""
        values = []
        censored = []

        for v in self.raw_df[self.activity_col]:
            val, cen = extract_numeric_activity(v)
            values.append(val)
            censored.append(cen)

        self.raw_df["_Activity_raw"] = values
        self.raw_df["Censored"] = censored

        # Count censored values
        n_left_censored = censored.count("left")
        n_right_censored = censored.count("right")
        
        self.statistics['n_left_censored'] = n_left_censored
        self.statistics['n_right_censored'] = n_right_censored
        
        if n_left_censored > 0 or n_right_censored > 0:
            self.log(f"Censored values: {n_left_censored} left (<), {n_right_censored} right (>)")

        # Remove invalid activity values
        before = len(self.raw_df)
        self.raw_df = self.raw_df[self.raw_df["_Activity_raw"].notna()]
        after = len(self.raw_df)
        
        n_invalid_activity = before - after
        self.statistics['n_invalid_activity'] = n_invalid_activity
        
        self.log(f"Invalid activity values removed: {n_invalid_activity}")

        # Unit conversion to nM
        self.raw_df["Activity_nM"] = self.raw_df["_Activity_raw"].apply(
            lambda x: convert_to_nM(x, self.unit)
        )
        
        # Remove failed conversions
        before = len(self.raw_df)
        self.raw_df = self.raw_df[self.raw_df["Activity_nM"].notna()]
        after = len(self.raw_df)
        
        if before - after > 0:
            self.log(f"Failed unit conversions: {before - after}")

    def detect_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers in activity values.
        
        Args:
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean mask of outliers
        """
        y = self.raw_df["Activity_nM"].values
        
        if method == 'iqr':
            Q1, Q3 = np.percentile(y, [25, 75])
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers = (y < lower) | (y > upper)
        elif method == 'zscore':
            z = np.abs((y - np.mean(y)) / np.std(y))
            outliers = z > threshold
        else:
            outliers = np.zeros(len(y), dtype=bool)
        
        n_outliers = outliers.sum()
        self.statistics['n_outliers'] = int(n_outliers)
        
        if n_outliers > 0:
            self.log(f"Outliers detected ({method}): {n_outliers}")
        
        return outliers

    def infer_task(self):
        """Infer whether this is a regression or classification task."""
        y = self.raw_df["Activity_nM"]
        unique = np.unique(y.dropna())

        if len(unique) == 1:
            task = "INVALID_SINGLE_CLASS"
        elif y.dtype.kind in "fi" and len(unique) > 10:
            task = "REGRESSION"
        elif len(unique) == 2:
            task = "BINARY_CLASSIFICATION"
        elif len(unique) <= 10:
            task = "MULTICLASS_CLASSIFICATION"
        else:
            task = "REGRESSION"

        self.task = task
        self.statistics['task_type'] = task
        self.log(f"Inferred task type: {task}")

    def statistical_log_decision(self) -> bool:
        """
        Decide whether to apply log transform based on:
        - Skewness
        - Normality (Shapiro-Wilk test)
        
        Returns:
            True if log transform should be applied
        """
        if not SCIPY_AVAILABLE:
            self.log("SciPy not available - applying log transform by default for IC50/Ki data")
            return True
            
        y = self.raw_df["Activity_nM"].values
        
        # Calculate skewness
        skewness = skew(y)
        
        # Shapiro-Wilk test (limited to 5000 samples)
        sample_size = min(len(y), 5000)
        if len(y) >= 3:
            shapiro_stat, shapiro_p = shapiro(y[:sample_size])
        else:
            shapiro_p = 1.0
            
        self.statistics['skewness_raw'] = float(skewness)
        self.statistics['shapiro_p_raw'] = float(shapiro_p)

        self.log(f"Skewness (raw): {skewness:.3f}")
        self.log(f"Shapiro-Wilk p-value (raw): {shapiro_p:.3e}")

        apply_log = False

        # Decision criteria
        if abs(skewness) > 1.0:
            apply_log = True
            self.log("  → High skewness detected")
        if shapiro_p < 0.05:
            apply_log = True
            self.log("  → Non-normal distribution detected")
            
        # For typical IC50/Ki data, log transform is usually appropriate
        y_range = np.max(y) / np.min(y) if np.min(y) > 0 else np.inf
        if y_range > 100:
            apply_log = True
            self.log(f"  → Large dynamic range detected ({y_range:.1f}x)")

        return apply_log

    def apply_transform(self, apply_log: bool = True):
        """Apply transformation to activity values."""
        if apply_log:
            # Convert to pIC50 (or pKi, pEC50, etc.)
            self.raw_df["Target"] = -np.log10(self.raw_df["Activity_nM"] * 1e-9)
            self.statistics['transform'] = 'pIC50'
            self.log("✓ Log transform applied: pIC50 = -log10(IC50_M)")
            
            # Check transformed distribution
            if SCIPY_AVAILABLE:
                y_transformed = self.raw_df["Target"].values
                skewness_t = skew(y_transformed)
                self.statistics['skewness_transformed'] = float(skewness_t)
                self.log(f"  Skewness (transformed): {skewness_t:.3f}")
        else:
            self.raw_df["Target"] = self.raw_df["Activity_nM"]
            self.statistics['transform'] = 'none'
            self.log("Log transform NOT applied - using raw values")

    def calculate_activity_stats(self):
        """Calculate and log activity statistics."""
        y = self.raw_df["Target"].values
        
        stats = {
            'n_samples': len(y),
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y)),
            'median': float(np.median(y)),
        }
        
        self.statistics['target_stats'] = stats
        
        self.log(f"Target statistics:")
        self.log(f"  N: {stats['n_samples']}")
        self.log(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
        self.log(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")

    def finalize(self):
        """Prepare final cleaned DataFrame."""
        # Select and rename columns
        output_cols = ["Canonical_SMILES", "Target", "Censored"]
        
        # Add molecular weight if calculated
        if "_MW" in self.raw_df.columns:
            output_cols.append("_MW")
            self.raw_df = self.raw_df.rename(columns={"_MW": "MW"})
            output_cols[-1] = "MW"
        
        self.cleaned_df = self.raw_df[output_cols].reset_index(drop=True)
        
        # Rename SMILES column
        self.cleaned_df = self.cleaned_df.rename(columns={"Canonical_SMILES": "SMILES"})

        self.log(f"✓ Final dataset size: {len(self.cleaned_df)}")

    def generate_report(self, path: str = "qsar_data_report.txt") -> str:
        """
        Generate curation report.
        
        Args:
            path: Output file path
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("QSAR DATA CURATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {timestamp}\n\n")
            
            f.write("-" * 40 + "\n")
            f.write("CURATION LOG\n")
            f.write("-" * 40 + "\n")
            for line in self.report:
                f.write(line + "\n")
            
            f.write("\n" + "-" * 40 + "\n")
            f.write("STATISTICS\n")
            f.write("-" * 40 + "\n")
            for key, value in self.statistics.items():
                if isinstance(value, dict):
                    f.write(f"\n{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        return path

    def generate_markdown_report(self, path: str = "qsar_data_report.md") -> str:
        """Generate curation report in Markdown format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# QSAR Data Curation Report

**Generated:** {timestamp}

## Summary

| Metric | Value |
|--------|-------|
| Initial samples | {self.statistics.get('n_valid_smiles', 0) + self.statistics.get('n_invalid_smiles', 0)} |
| Invalid SMILES | {self.statistics.get('n_invalid_smiles', 0)} |
| Duplicates removed | {self.statistics.get('n_duplicates', 0)} |
| Invalid activities | {self.statistics.get('n_invalid_activity', 0)} |
| Final samples | {len(self.cleaned_df) if self.cleaned_df is not None else 0} |
| Task type | {self.statistics.get('task_type', 'Unknown')} |
| Transform applied | {self.statistics.get('transform', 'Unknown')} |

## Curation Log

"""
        for line in self.report:
            report += f"- {line}\n"
        
        if 'target_stats' in self.statistics:
            stats = self.statistics['target_stats']
            report += f"""
## Target Statistics

| Statistic | Value |
|-----------|-------|
| N | {stats['n_samples']} |
| Mean | {stats['mean']:.4f} |
| Std | {stats['std']:.4f} |
| Min | {stats['min']:.4f} |
| Max | {stats['max']:.4f} |
| Median | {stats['median']:.4f} |
"""
        
        with open(path, "w") as f:
            f.write(report)
        
        return path

    def run(self, remove_outliers: bool = False) -> pd.DataFrame:
        """
        Run the complete data curation pipeline.
        
        Args:
            remove_outliers: Whether to remove detected outliers
            
        Returns:
            Cleaned DataFrame with columns: SMILES, Target, Censored
        """
        if self.verbose:
            print("\n" + "=" * 50)
            print("🔬 QSAR Data Curation Agent")
            print("=" * 50)
        
        self.log(f"Starting QSAR data curation pipeline")
        self.log(f"Initial rows: {len(self.raw_df)}")
        self.log(f"SMILES column: {self.smiles_col}")
        self.log(f"Activity column: {self.activity_col}")
        self.log(f"Unit: {self.unit}")

        # Step 1: Validate SMILES
        self.validate_smiles()
        
        # Step 2: Standardize and deduplicate
        self.standardize_smiles()
        
        # Step 3: Clean activity values
        self.clean_activity()
        
        # Step 4: Detect outliers (optional removal)
        if remove_outliers:
            outlier_mask = self.detect_outliers()
            before = len(self.raw_df)
            self.raw_df = self.raw_df[~outlier_mask]
            self.log(f"Outliers removed: {before - len(self.raw_df)}")
        
        # Step 5: Infer task type
        self.infer_task()
        
        # Step 6: Decide on log transformation (for regression)
        if self.task == "REGRESSION":
            apply_log = self.statistical_log_decision()
            self.apply_transform(apply_log)
        else:
            self.raw_df["Target"] = self.raw_df["Activity_nM"]
            self.statistics['transform'] = 'none'
            self.log("Classification task - no log transform applied")
        
        # Step 7: Calculate final statistics
        self.calculate_activity_stats()
        
        # Step 8: Finalize
        self.finalize()
        
        self.log("✓ Pipeline completed successfully")
        
        if self.verbose:
            print("=" * 50 + "\n")

        return self.cleaned_df

    def get_statistics(self) -> Dict[str, Any]:
        """Return curation statistics."""
        return self.statistics.copy()


# ============================================================
# Convenience function
# ============================================================

def curate_qsar_data(
    df: pd.DataFrame,
    smiles_col: str,
    activity_col: str,
    unit: str = "nM",
    remove_outliers: bool = False,
    report_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to curate QSAR data in one call.
    
    Args:
        df: Input DataFrame
        smiles_col: Name of SMILES column
        activity_col: Name of activity column
        unit: Activity unit (nM, uM, etc.)
        remove_outliers: Whether to remove outliers
        report_path: Path to save report (optional)
        verbose: Print progress
        
    Returns:
        Tuple of (cleaned_df, statistics)
        
    Example:
        >>> clean_df, stats = curate_qsar_data(
        ...     df, "SMILES", "IC50_nM", unit="nM"
        ... )
    """
    agent = QSARDataCurationAgent(
        df=df,
        smiles_col=smiles_col,
        activity_col=activity_col,
        unit=unit,
        verbose=verbose
    )
    
    cleaned_df = agent.run(remove_outliers=remove_outliers)
    
    if report_path:
        if report_path.endswith('.md'):
            agent.generate_markdown_report(report_path)
        else:
            agent.generate_report(report_path)
    
    return cleaned_df, agent.get_statistics()
