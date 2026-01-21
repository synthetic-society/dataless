# dataless

[![Tests](https://github.com/synthetic-society/demo-scaling-identification/actions/workflows/tests.yml/badge.svg)](https://github.com/synthetic-society/demo-scaling-identification/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/synthetic-society/dataless/branch/main/graph/badge.svg)](https://codecov.io/gh/synthetic-society/dataless)

A Python package for modeling and forecasting the effectiveness of identification techniques at scale. It provides tools to predict how the accuracy of identification methods changes as the population size increases.

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
import pandas as pd
import numpy as np

# Step 1: Create training data with observed accuracy at small scales
# n = population size, κ = identification accuracy (fraction correctly identified)
data = pd.DataFrame({
    "n": [10, 50, 100, 500],
    "κ": [0.99, 0.97, 0.95, 0.90]
})

# Step 2: Fit the model
model = PYPExtrapolation(data)

# Step 3: Predict accuracy at larger scales
large_populations = np.array([1_000, 10_000, 100_000, 1_000_000])
predictions = model.predict(large_populations)
print(predictions)
# Example output: [0.86, 0.78, 0.71, 0.65]

# Step 4: Get a summary of the fitted model
print(model.summary())
```

### Understanding the Output

- **κ (kappa)** values range from 0 to 1, where 1 means perfect identification
- The model predicts how κ decreases as population size grows
- This helps assess whether an identification method will remain effective at scale

### Available Models

| Model | Description | Best for |
|-------|-------------|----------|
| `PYPExtrapolation` | Pitman-Yor Process (recommended) | Most scenarios |
| `FLExtrapolation` | Entropy-based baseline | Baseline |
| `ExpDecayExtrapolation` | Exponential decay | Baseline |
| `PolynomialExtrapolation` | Polynomial fit | Baseline |

## Usage

### Basic Example
```python
from dataless import PYPExtrapolation
import pandas as pd
import numpy as np

# Create sample data: identification accuracy at different gallery sizes
d = pd.DataFrame({'n': [1, 10, 100], 'κ': [1, 0.99, 0.95]})

# Train model and predict accuracy at larger scales
model = PYPExtrapolation(d)
model.predict(np.array([1, 10, 100, 1000, 10000]))
# array([1.        , 0.99000117, 0.95000214, 0.88420427, 0.81462242])
```

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
