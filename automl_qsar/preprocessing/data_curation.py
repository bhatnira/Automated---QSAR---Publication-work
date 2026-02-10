"""
QSAR Data Curation Agent - Multi-Task Support
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from scipy.stats import shapiro, skew
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.preprocessing import LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TaskType(Enum):
    REGRESSION_SINGLE = "regression_single"
    REGRESSION_MULTI = "regression_multi"
    CLASSIFICATION_BINARY = "classification_binary"
    CLASSIFICATION_MULTICLASS = "classification_multiclass"
    CLASSIFICATION_MULTILABEL = "classification_multilabel"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class TargetConfig:
    name: str
    task_type: TaskType = TaskType.UNKNOWN
    unit: str = "nM"
    transform: str = "none"
    n_classes: int = 0
    classes: List[str] = field(default_factory=list)
    label_encoder: Any = None
    statistics: Dict[str, Any] = field(default_factory=dict)


def infer_task_type(series: pd.Series) -> TaskType:
    valid = series.dropna()
    if len(valid) == 0:
        return TaskType.UNKNOWN
    
    if valid.dtype == 'object':
        sample = valid.iloc[:100]
        has_letters = sample.apply(lambda x: any(c.isalpha() for c in str(x)))
        if has_letters.sum() > len(sample) * 0.5:
            n_unique = valid.nunique()
            return TaskType.CLASSIFICATION_BINARY if n_unique == 2 else TaskType.CLASSIFICATION_MULTICLASS
    
    try:
        numeric = pd.to_numeric(valid, errors='coerce')
        valid_numeric = numeric.dropna()
        
        if len(valid_numeric) == 0:
            n_unique = valid.nunique()
            return TaskType.CLASSIFICATION_BINARY if n_unique == 2 else TaskType.CLASSIFICATION_MULTICLASS
        
        n_unique = valid_numeric.nunique()
        all_integers = np.allclose(valid_numeric.values, valid_numeric.values.astype(int))
        is_label_like = all_integers and n_unique <= 10 and valid_numeric.min() >= 0 and valid_numeric.max() < 20
        
        if is_label_like and n_unique == 2:
            return TaskType.CLASSIFICATION_BINARY
        elif is_label_like and n_unique <= 10:
            return TaskType.CLASSIFICATION_MULTICLASS
        return TaskType.REGRESSION_SINGLE
    except:
        return TaskType.UNKNOWN


class QSARDataCurationAgent:
    def __init__(self, df, smiles_col='SMILES', target_cols=None, units=None, verbose=True, 
                 skewness_threshold=1.0, shapiro_p_threshold=0.05):
        self.df = df.copy()
        self.smiles_col = smiles_col
        self.verbose = verbose
        self.skewness_threshold = skewness_threshold
        self.shapiro_p_threshold = shapiro_p_threshold
        
        if target_cols is None:
            self.target_cols = [c for c in df.columns if c != smiles_col]
        elif isinstance(target_cols, str):
            self.target_cols = [target_cols]
        else:
            self.target_cols = list(target_cols)
        
        if units is None:
            self.units = {col: None for col in self.target_cols}
        elif isinstance(units, str):
            self.units = {col: units for col in self.target_cols}
        else:
            self.units = units
        
        self.target_configs = {}
        self.label_encoders = {}
        self.cleaned_df = None
        self.overall_task = TaskType.UNKNOWN
        self._stats = {}
    
    def _log(self, msg):
        if self.verbose:
            print(f"[DataCuration] {msg}")
    
    def _validate_smiles(self, smiles):
        if not smiles or pd.isna(smiles):
            return False
        if not HAS_RDKIT:
            return isinstance(smiles, str) and len(smiles) > 0
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            return mol is not None
        except:
            return False
    
    def _should_log_transform(self, values):
        stats = {'skewness': 0, 'shapiro_p': 1.0}
        if HAS_SCIPY and len(values) >= 3:
            stats['skewness'] = float(skew(values))
            sample = values[:5000] if len(values) > 5000 else values
            try:
                _, stats['shapiro_p'] = shapiro(sample)
            except:
                pass
        should_log = (abs(stats['skewness']) > self.skewness_threshold and 
                      stats['shapiro_p'] < self.shapiro_p_threshold and np.all(values > 0))
        return should_log, stats
    
    def _analyze_targets(self):
        reg_count = 0
        class_count = 0
        
        for col in self.target_cols:
            task_type = infer_task_type(self.df[col])
            config = TargetConfig(name=col, task_type=task_type, unit=self.units.get(col, "nM"))
            values = self.df[col].dropna()
            
            if task_type == TaskType.REGRESSION_SINGLE:
                reg_count += 1
                numeric = pd.to_numeric(values, errors='coerce').dropna()
                if len(numeric) > 0:
                    should_log, stats = self._should_log_transform(numeric.values)
                    config.statistics = stats
                    config.transform = 'pIC50' if should_log else 'none'
            else:
                class_count += 1
                config.n_classes = values.nunique()
                config.classes = sorted([str(v) for v in values.unique()])
                config.transform = 'label_encode'
                config.statistics = {'class_counts': dict(values.value_counts())}
            
            self.target_configs[col] = config
            self._log(f"Target '{col}': {task_type.value}, transform={config.transform}")
        
        if len(self.target_cols) == 1:
            self.overall_task = self.target_configs[self.target_cols[0]].task_type
        elif reg_count > 0 and class_count > 0:
            self.overall_task = TaskType.MIXED
        elif reg_count > 1:
            self.overall_task = TaskType.REGRESSION_MULTI
        elif class_count > 1:
            self.overall_task = TaskType.CLASSIFICATION_MULTILABEL
        
        self._stats['overall_task'] = self.overall_task.value
    
    def _validate_molecules(self, df):
        initial = len(df)
        mask = df[self.smiles_col].apply(self._validate_smiles)
        valid_df = df[mask].copy()
        removed = initial - len(valid_df)
        if removed > 0:
            self._log(f"Removed {removed} invalid SMILES ({len(valid_df)} remaining)")
        self._stats['invalid_smiles'] = removed
        return valid_df
    
    def _handle_duplicates(self, df):
        initial = len(df)
        agg_funcs = {}
        
        for col in self.target_cols:
            config = self.target_configs.get(col)
            if config and config.task_type == TaskType.REGRESSION_SINGLE:
                agg_funcs[col] = 'mean'
            else:
                agg_funcs[col] = 'first'
        
        other_cols = [c for c in df.columns if c not in self.target_cols and c != self.smiles_col]
        for col in other_cols:
            agg_funcs[col] = 'first'
        
        if agg_funcs:
            df = df.groupby(self.smiles_col, as_index=False).agg(agg_funcs)
        else:
            df = df.drop_duplicates(subset=self.smiles_col)
        
        removed = initial - len(df)
        if removed > 0:
            self._log(f"Merged {removed} duplicates ({len(df)} remaining)")
        self._stats['duplicates'] = removed
        return df
    
    def _handle_missing(self, df):
        initial = len(df)
        df = df.dropna(subset=self.target_cols)
        removed = initial - len(df)
        if removed > 0:
            self._log(f"Removed {removed} rows with missing targets ({len(df)} remaining)")
        self._stats['missing'] = removed
        return df
    
    def _transform_targets(self, df):
        df = df.copy()
        for col in self.target_cols:
            config = self.target_configs.get(col)
            if not config:
                continue
            
            if config.task_type == TaskType.REGRESSION_SINGLE:
                if config.transform == 'pIC50':
                    values = pd.to_numeric(df[col], errors='coerce')
                    unit = self.units.get(col, 'nM')
                    if unit == 'nM':
                        molar = values * 1e-9
                    elif unit == 'uM':
                        molar = values * 1e-6
                    elif unit == 'pM':
                        molar = values * 1e-12
                    else:
                        molar = values
                    molar = np.clip(molar, 1e-12, None)
                    df[col] = -np.log10(molar)
                    self._log(f"Transformed '{col}' to pIC50")
                    trans = df[col].dropna()
                    config.statistics.update({
                        'mean': float(trans.mean()),
                        'std': float(trans.std()),
                        'min': float(trans.min()),
                        'max': float(trans.max())
                    })
            else:
                if HAS_SKLEARN:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                    config.label_encoder = le
                    self._log(f"Label encoded '{col}': {list(le.classes_)}")
        return df
    
    def run(self):
        self._log("Starting data curation...")
        self._log(f"Input: {len(self.df)} compounds, {len(self.target_cols)} target(s)")
        self._analyze_targets()
        df = self._validate_molecules(self.df)
        df = self._handle_duplicates(df)
        df = self._handle_missing(df)
        df = self._transform_targets(df)
        self.cleaned_df = df
        self._log(f"Curation complete: {len(df)} compounds")
        return df
    
    def get_statistics(self):
        return self._stats.copy()
    
    def get_target_configs(self):
        return self.target_configs.copy()
    
    def get_label_encoders(self):
        return self.label_encoders.copy()


def curate_qsar_data(df, smiles_col='SMILES', target_cols=None, units=None, verbose=True, **kwargs):
    agent = QSARDataCurationAgent(df=df, smiles_col=smiles_col, target_cols=target_cols, 
                                   units=units, verbose=verbose, **kwargs)
    cleaned = agent.run()
    return cleaned, agent.get_statistics()
