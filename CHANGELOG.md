# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- ChemBERTa and MolBERT embedding integration
- Multi-objective optimization
- Interactive visualization dashboard
- REST API for model serving
- Docker deployment
- Additional GNN architectures (GAT, GraphSAGE)
- Quantum chemical descriptors
- Active learning module

## [0.1.0] - 2026-02-05

### Added
- Initial release of AutoML QSAR framework
- Complete featurization module with 7 methods:
  - RDKit molecular descriptors
  - ECFP/Morgan fingerprints
  - MACCS keys
  - Atom Pair fingerprints
  - Pharmacophore features
  - Molecular embeddings (placeholder)
  - Graph representations for GNN
- Preprocessing module with:
  - Three scaling methods (Standard, MinMax, Robust)
  - Feature selection (Variance, Correlation, Boruta)
  - Dimensionality reduction (PCA, UMAP)
- Model selection module with 12+ models:
  - Linear models (Linear, Ridge, Lasso, ElasticNet)
  - Tree-based (RandomForest, GradientBoosting, XGBoost)
  - Kernel methods (SVR, KernelRidge)
  - Neural networks (Feedforward, Deep NN, GNN)
- Hyperparameter optimization:
  - Bayesian optimization (Optuna)
  - Genetic algorithms
  - Grid search
- Evaluation module:
  - Multiple metrics (RMSE, MAE, R², Q², CCC)
  - K-Fold, LOSO, and Nested CV
- Ensemble methods:
  - Voting ensemble
  - Stacking ensemble
  - Top-K ensemble with uncertainty estimation
- Interpretation module:
  - Feature importance (tree, linear, permutation)
  - SHAP integration
  - LIME integration
- Main AutoMLQSAR pipeline with:
  - End-to-end automation
  - Model persistence
  - Prediction with uncertainty
  - Comprehensive API
- Documentation:
  - README with quick start
  - Comprehensive DOCUMENTATION.md
  - QUICK_REFERENCE.md
  - PROJECT_SUMMARY.md
  - CONTRIBUTING.md
- Examples and tutorials
- Unit tests framework
- Configuration file support (YAML)
- MIT License

### Features
- Automated molecular featurization
- Multi-model training and evaluation
- Hyperparameter optimization
- Robust cross-validation
- Ensemble learning
- Uncertainty quantification
- Model interpretability
- Save/load functionality
- Batch prediction support

### Technical Details
- Python 3.8+ compatible
- Modular architecture
- Error handling throughout
- Type hints
- Comprehensive docstrings
- PEP 8 compliant code

### Dependencies
- Core: numpy, pandas, scikit-learn, scipy, RDKit
- Optional: PyTorch, PyTorch Geometric, XGBoost, Optuna, SHAP, LIME

## [0.0.1] - 2026-02-04

### Added
- Initial project structure
- Basic module scaffolding

---

## Version History Summary

- **0.1.0** (2026-02-05): First production-ready release with complete pipeline
- **0.0.1** (2026-02-04): Initial project setup

## Future Versions

### v0.2.0 (Planned)
- Advanced embedding models (ChemBERTa, MolBERT)
- Enhanced GNN architectures
- Interactive visualization dashboard
- Improved documentation with video tutorials

### v0.3.0 (Planned)
- REST API for model serving
- Docker containerization
- Cloud deployment scripts
- Multi-objective optimization

### v1.0.0 (Planned)
- Stable API
- Production deployment tools
- Comprehensive test coverage (>90%)
- Performance benchmarks
- Published paper integration

---

**Note**: This is an active project. Features and APIs may change before v1.0.0.
