# Contributing to SignApp

Thank you for your interest in contributing to SignApp! We welcome contributions from everyone.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/SignApp.git`
3. Create a virtual environment: `python -m venv .venv`
4. Install development dependencies: `pip install -r requirements.txt -e ".[dev]"`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Run tests: `pytest tests/`
4. Run linting: `ruff check src/ tests/` and `black src/ tests/`
5. Commit your changes: `git commit -am "Add your message"`
6. Push to your fork: `git push origin feature/your-feature`
7. Create a Pull Request

## Code Style

We follow PEP 8 with:
- Line length: 100 characters
- Use type hints where possible
- Add docstrings to functions and classes

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Example: "Add disfluency removal API endpoint"

## Testing

- Write tests for new features in `tests/`
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Reporting Issues

When reporting issues, please include:
- Python version
- OS and system info
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Questions?

Feel free to open an issue with your question or contact the maintainers.

Thank you for contributing!
