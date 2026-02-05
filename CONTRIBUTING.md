# Contributing to AutoML QSAR

Thank you for your interest in contributing to AutoML QSAR!

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, package versions)

### Suggesting Enhancements

We welcome suggestions for new features:
- Describe the feature and its benefits
- Provide examples of usage
- Explain implementation ideas if you have them

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Update documentation
7. Commit your changes (`git commit -am 'Add new feature'`)
8. Push to the branch (`git push origin feature/your-feature`)
9. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions/classes
- Keep functions focused and modular

### Testing

- Write unit tests for new features
- Ensure all existing tests pass
- Aim for >80% code coverage

### Documentation

- Update README.md if adding new features
- Add docstrings with examples
- Update DOCUMENTATION.md for major changes
- Add examples in `examples/` directory

## Development Setup

```bash
# Clone repository
git clone https://github.com/bhatnira/Automated---QSAR---Publication-work.git
cd Automated---QSAR---Publication-work

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Check code style
black automl_qsar/
flake8 automl_qsar/
```

## Areas for Contribution

We especially welcome contributions in:

1. **New Featurizers**
   - Additional molecular descriptors
   - New fingerprint types
   - Pre-trained embeddings (ChemBERTa, MolBERT)

2. **Model Implementations**
   - Additional ML algorithms
   - Improved GNN architectures
   - Transfer learning methods

3. **Optimization Algorithms**
   - Advanced HPO methods
   - Multi-objective optimization
   - AutoML meta-learning

4. **Interpretability**
   - New interpretation methods
   - Visualization improvements
   - Domain-specific explanations

5. **Documentation**
   - Tutorials and examples
   - Use case studies
   - API documentation

6. **Performance**
   - Speed optimizations
   - Memory efficiency
   - Parallel processing

## Questions?

Feel free to open an issue for any questions or discussions!

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
