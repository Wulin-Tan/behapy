# Contributing to behapy

Thank you for your interest in contributing to `behapy`! We welcome contributions in the form of bug reports, feature requests, documentation improvements, and code changes.

## How to Contribute

1.  **Report Bugs**: Use the GitHub Issue Tracker to report bugs. Please provide a clear description and a minimal reproducible example if possible.
2.  **Suggest Features**: Open an issue to discuss new features or improvements.
3.  **Submit Pull Requests**:
    *   Fork the repository.
    *   Create a new branch for your changes.
    *   Make sure your code follows the project's style (Black, isort).
    *   Add tests for any new functionality.
    *   Ensure all tests pass.
    *   Submit a PR with a clear description of your changes.

## Development Setup

We recommend using a virtual environment for development:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Code Style

We use `black` for code formatting and `isort` for import sorting. You can run them manually or use pre-commit hooks.

```bash
black behapy tests
isort behapy tests
```

## Testing

Run tests using `pytest`:

```bash
pytest
```

## Contact

If you have any questions or need further assistance, please feel free to reach out:
- Email: wulin.tan9527@gmail.com
- GitHub Issues: `https://github.com/Wulin-Tan/behapy/issues`
