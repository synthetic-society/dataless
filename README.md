# dataless

[![Tests](https://github.com/synthetic-society/demo-scaling-identification/actions/workflows/tests.yml/badge.svg)](https://github.com/synthetic-society/demo-scaling-identification/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/synthetic-society/dataless/branch/main/graph/badge.svg)](https://codecov.io/gh/synthetic-society/dataless)

A Python package for modeling and forecasting the effectiveness of identification techniques at scale. It provides tools to predict how the accuracy of identification methods changes as the population size increases. The research behind this package is detailed in the paper: [A scaling law to model the effectiveness of identification techniques](https://www.nature.com/articles/s41467-024-55296-6), published in 2025 in *Nature Communications*.

## Overview

This package helps analyze three types of identification methods:
- **Exact matching**: Identifying individuals using exact matches of attributes (e.g., demographics)
- **Sparse matching**: Identification using sparse data points (e.g., location history)
- **Robust matching**: Machine learning-based identification handling noisy or approximate data

Key terminology:
- **κ (kappa)**: The fraction of people accurately identified in a population
- **Gallery size**: The number of individuals against which identification is attempted
- **k-anonymity**: A privacy measure ensuring each combination of attributes appears at least k times

## Features

- **Empirical Analysis**: Fast numpy code to analyze identification accuracy across different gallery sizes
- **Scaling Prediction**: Two-parameter Bayesian model to forecast identification correctness (κ), uniqueness, and % of k-anonymity violations at larger scales
- **Extrapolation**: Methods to extrapolate small-scale experimental results to real-world scenarios

## Installation

### From PyPI

```bash
pip install pydataless
```

### From source (development)

This project uses [uv](https://docs.astral.sh/uv/) for package management:

```bash
git clone https://github.com/synthetic-society/dataless.git
cd dataless
uv sync
```

### Requirements

- Python ≥ 3.10
- numpy ≥ 2.0.0
- pandas ≥ 2.2.2
- scipy ≥ 1.14.0
- matplotlib ≥ 3.9.1

## Getting Started

### Quick Example

The main use case is predicting how identification accuracy degrades as population size increases:

```python
from dataless import PYPExtrapolation
import numpy as np

# Step 1: Create training data with observed accuracy at small scales
# n = population size, correctness = identification accuracy (fraction correctly identified)
n = [10, 50, 100, 500]
correctness = [0.99, 0.97, 0.95, 0.90]

# Step 2: Fit the model
model = PYPExtrapolation(n, correctness=correctness)

# Step 3: Predict accuracy at larger scales
large_populations = np.array([1_000, 10_000, 100_000, 1_000_000])
predictions = model.predict(large_populations)
print(predictions)
# Example output: [0.87505023 0.79165748 0.71371539 0.64308667]

# Step 4: Get a summary of the fitted model
print(model.summary())
```

You can also train from uniqueness scores:

```python
n = [10, 50, 100, 500]
uniqueness = [0.95, 0.90, 0.85, 0.80] 

model = PYPExtrapolation(n, uniqueness=uniqueness)

large_populations = np.array([1_000, 10_000, 100_000, 1_000_000])
predictions = model.predict(large_populations)

print(predictions)
# Example output: [0.77117386 0.68920739 0.61585755 0.55030553]
``` 


### Understanding the Output

- **correctness** values range from 0 to 1, where 1 means perfect identification
- The model predicts how the correctness decreases as population size grows
- This helps assess whether an identification method will remain effective at scale

### Available Models

| Model | Description | Best for |
|-------|-------------|----------|
| `PYPExtrapolation` | Pitman-Yor Process | Most scenarios |
| `FLExtrapolation` | Entropy-based baseline | Baseline |
| `ExpDecayExtrapolation` | Exponential decay | Baseline |
| `PolynomialExtrapolation` | Polynomial fit | Baseline |


## Development

### Running Tests
```bash
uv sync --extras test
uv run pytest tests/ --cov=dataless --cov-report=xml --cov-report=term
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests to ensure they pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Reporting Issues

Please report bugs and request features using the issue tracker. When reporting bugs:
- Describe what you expected to happen
- Describe what actually happened
- Include code samples and error messages if relevant
- Include version information (Python, dataless, key dependencies)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.


## Acknowledgements

If you use this package in your research, please cite:

```
@article{rocher2025scaling,
  title={A scaling law to model the effectiveness of identification techniques},
  author={Rocher, Luc and Hendrickx, Julien M and Montjoye, Yves-Alexandre de},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={347},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

```