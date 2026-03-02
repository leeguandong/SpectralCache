# Contributing to SpectralCache

Thank you for your interest in contributing to SpectralCache! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/SpectralCache.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests and formatting checks
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SpectralCache.git
cd SpectralCache

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

## Code Style

We use `black` for code formatting:

```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest benchmark/test_spectralcache_unit.py

# Run with coverage
pytest --cov=. --cov-report=html
```

## Adding New Features

### 1. Benchmark Scripts

If adding a new benchmark:
- Place it in `benchmark/`
- Follow the naming convention: `{feature}_benchmark.py`
- Include command-line arguments for flexibility
- Save results to a structured output directory
- Add documentation in the script docstring

### 2. Documentation

- Update `README.md` if adding user-facing features
- Update `docs/IMPLEMENTATION.md` for technical details
- Add inline comments for complex logic
- Include docstrings for all functions and classes

### 3. Paper Figures

If generating new figures for the paper:
- Save high-resolution PDFs to `paper/`
- Include the generation script in `benchmark/`
- Document the exact commands to reproduce the figure

## Pull Request Guidelines

### PR Title Format

Use conventional commit format:
- `feat: Add new feature`
- `fix: Fix bug in component`
- `docs: Update documentation`
- `test: Add tests for feature`
- `refactor: Refactor code`
- `perf: Improve performance`

### PR Description

Include:
1. **What**: Brief description of changes
2. **Why**: Motivation for the changes
3. **How**: Technical approach (if non-trivial)
4. **Testing**: How you tested the changes
5. **Results**: Performance impact (if applicable)

Example:
```markdown
## What
Add support for PixArt-Sigma model

## Why
Extend SpectralCache to support more DiT architectures

## How
- Implement PixArt-specific adapter in cache/diffusers_adapters/
- Add PixArt benchmark script
- Update documentation

## Testing
- Tested on PixArt-Sigma-XL-2-1024-MS
- Verified 1.8× speedup with LPIPS 0.18

## Results
| Model | Speedup | LPIPS |
|-------|---------|-------|
| PixArt-Sigma | 1.82× | 0.183 |
```

## Reporting Issues

When reporting bugs, include:
1. **Environment**: OS, Python version, PyTorch version, GPU
2. **Steps to reproduce**: Minimal code example
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Error messages**: Full traceback if applicable

## Code Review Process

1. Maintainers will review your PR within 1-2 weeks
2. Address review comments by pushing new commits
3. Once approved, maintainers will merge your PR
4. Your contribution will be acknowledged in the release notes

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

- Open an issue for questions about contributing
- Tag maintainers in your PR for faster review
- Join our discussions for general questions

Thank you for contributing to SpectralCache! 🎉
