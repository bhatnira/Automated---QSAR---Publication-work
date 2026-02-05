# AutoML QSAR Modeling Software

A comprehensive automated machine learning pipeline for Quantitative Structure-Activity Relationship (QSAR) modeling.

## Architecture

The software follows a modular pipeline architecture:

1. **Featurization Module** - Molecular descriptor generation (RDKit, ECFP, Pharmacophore, Embeddings, Graph)
2. **Preprocessing Module** - Data scaling, dimensionality reduction, feature selection
3. **Model Selection Module** - Linear, Tree-based, Kernel, Neural Networks, GNN models
4. **Hyperparameter Optimization** - Bayesian, Genetic, Grid Search
5. **Evaluation Module** - Nested CV, LOSO, Q², RMSE, MAE metrics
6. **Ensemble Module** - Top-K pipeline consensus and uncertainty estimation
7. **Prediction & Interpretability** - Model predictions and feature importance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from automl_qsar import AutoMLQSAR

# Initialize the AutoML QSAR pipeline
automl = AutoMLQSAR()

# Load data (SMILES and activity values)
automl.load_data('molecules.csv', smiles_col='SMILES', target_col='Activity')

# Run the automated pipeline
results = automl.fit()

# Make predictions
predictions = automl.predict(new_smiles)

# Get model interpretation
interpretation = automl.interpret()
```

## Project Structure

```
automl_qsar/
├── featurization/      # Molecular featurization methods
├── preprocessing/      # Data preprocessing utilities
├── models/            # ML model implementations
├── optimization/      # Hyperparameter optimization
├── evaluation/        # Model evaluation metrics
├── ensemble/          # Ensemble methods
├── interpretation/    # Model interpretability
└── pipeline/          # Main pipeline orchestration
```

## Requirements

- Python >= 3.8
- RDKit
- scikit-learn
- PyTorch / PyTorch Geometric (for GNN)
- optuna (Bayesian optimization)
- DEAP (Genetic algorithms)
- pandas, numpy, matplotlib, seaborn

## License

MIT License
