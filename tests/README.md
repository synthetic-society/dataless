# Tests Documentation

This directory contains unit tests for the dataless package.

## Test Structure

- `test_empirical.py`: Tests for empirical analysis utilities
- `test_model.py`: Tests for statistical models and core functions
- `test_extrapolate.py`: Tests for extrapolation models
- `conftest.py`: Shared fixtures and test configuration
- `pytest.ini`: Pytest configuration settings

## Running Tests

### Basic Usage

Run all tests:
```bash
pytest
```

Run tests with detailed output:
```bash
pytest -v
```

### Specific Test Selection

Run tests from a specific file:
```bash
pytest tests/test_model.py
```

Run a specific test class:
```bash
pytest tests/test_model.py::TestPYP
```

Run a specific test:
```bash
pytest tests/test_model.py::TestPYP::test_init_with_d_alpha
```

### Test Categories

Skip slow tests:
```bash
pytest -m "not slow"
```

Run only slow tests:
```bash
pytest -m "slow"
```

### Coverage Reports

Run tests with coverage:
```bash
pytest --cov=dataless
```

Generate HTML coverage report:
```bash
pytest --cov=dataless --cov-report=html
```

## Test Organization

### Fixtures

Common test fixtures are defined in `conftest.py`:
- `sample_sizes`: Standard array of sample sizes
- `training_data`: Standard training DataFrame
- `sample_frequencies`: Sample frequency data
- `sample_multiplicities`: Sample multiplicity data

### Test Categories

1. **Empirical Tests** (`test_empirical.py`)
   - Frequency calculations
   - Entropy computations
   - DataFrame operations
   - Uniqueness/correctness metrics

2. **Model Tests** (`test_model.py`)
   - PYP core functions
   - Utility functions
   - PYP and FLModel classes
   - K-anonymity calculations

3. **Extrapolation Tests** (`test_extrapolate.py`)
   - Abstract base class implementation
   - Concrete model implementations
   - Training functionality
   - Prediction behavior

## Best Practices

1. **Test Independence**
   - Each test should be independent and not rely on state from other tests
   - Use fixtures for common setup
   - Clean up any resources after tests

2. **Test Coverage**
   - Aim for comprehensive coverage of functionality
   - Include edge cases and error conditions
   - Test both valid and invalid inputs

3. **Test Organization**
   - Group related tests in classes
   - Use descriptive test names
   - Include docstrings explaining test purpose

4. **Assertions**
   - Use appropriate assertion methods
   - Include meaningful error messages
   - Test both positive and negative cases

## Adding New Tests

When adding new tests:

1. Choose the appropriate test file based on functionality
2. Follow the existing naming conventions
3. Add necessary fixtures to `conftest.py` if needed
4. Include docstrings explaining test purpose
5. Ensure tests are independent
6. Add appropriate assertions
7. Update this README if adding new test categories

## Common Testing Patterns

1. **Setup-Execute-Assert Pattern**
   ```python
   def test_something():
       # Setup
       model = PYP(d=0.5, α=1.0)
       
       # Execute
       result = model.correctness(100)
       
       # Assert
       assert 0 <= result <= 1
   ```

2. **Parameterized Testing**
   ```python
   @pytest.mark.parametrize("n,k", [
       (10, 2),
       (100, 5),
       (1000, 10)
   ])
   def test_parameterized(n, k):
       # Test implementation
   ```

3. **Exception Testing**
   ```python
   def test_invalid_input():
       with pytest.raises(ValueError):
           # Code that should raise ValueError
   ```

## Troubleshooting

1. **Test Discovery Issues**
   - Ensure test files start with "test_"
   - Ensure test classes start with "Test"
   - Ensure test functions start with "test_"

2. **Fixture Issues**
   - Check fixture scope
   - Verify fixture dependencies
   - Check fixture availability in conftest.py

3. **Common Errors**
   - ImportError: Check import paths and virtual environment
   - AssertionError: Check test logic and expected values
   - TypeError: Check parameter types and function signatures
