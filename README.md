# dataless

[![Tests](https://github.com/synthetic-society/demo-scaling-identification/actions/workflows/tests.yml/badge.svg)](https://github.com/synthetic-society/demo-scaling-identification/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/synthetic-society/demo-scaling-identification/branch/main/graph/badge.svg)](https://codecov.io/gh/synthetic-society/demo-scaling-identification)

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

This project uses [pixi](https://pixi.sh/latest/) for package management to ensure reproducible environments:

```bash
pixi install
```

Requirements:
- Python ≥ 3.11
- numpy ≥ 2.0.0
- pandas ≥ 2.2.2
- scipy ≥ 1.14.0

## Usage

### Basic Example
```python
from dataless.extrapolate import PYPExtrapolation
import pandas as pd
import numpy as np

# Create sample data: identification accuracy at different gallery sizes
d = pd.DataFrame({'n': [1, 10, 100], 'κ': [1, 0.99, 0.95]})

# Train model and predict accuracy at larger scales
model = PYPExtrapolation(d)
model.train()
model.test(np.array([1, 10, 100, 1000, 10000]))
# array([1.        , 0.99000117, 0.95000214, 0.88420427, 0.81462242])
```

## Development

### Running Tests
```bash
pixi run test
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
