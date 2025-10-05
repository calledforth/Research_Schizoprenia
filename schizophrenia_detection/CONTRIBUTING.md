# Contributing to Schizophrenia Detection Project

We welcome contributions! This document outlines how to contribute to the schizophrenia detection project. It covers development setup, coding standards, pull request process, and notebook style guidelines.

## Development Setup

### Local Development

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/yourusername/schizophrenia_detection.git
cd schizophrenia_detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

5. Install pre-commit hooks:
```bash
pre-commit install
```

### Google Colab Development

For notebook-based development:

1. Open the notebook in Colab
2. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. Navigate to the project directory:
```python
%cd /content/drive/MyDrive/schizophrenia_detection
```

4. Install dependencies:
```python
!pip install -r requirements.txt
```

## Coding Standards

### Code Style

We follow PEP 8 style guidelines with some additional conventions:

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 88 characters (Black formatter default)
- Use type hints where possible
- Write descriptive docstrings for all functions and classes
- Use meaningful variable and function names

### Code Formatting

We use the following tools for code formatting:

- **Black**: For code formatting
- **isort**: For import sorting
- **flake8**: For linting

Run these tools before committing:
```bash
black .
isort .
flake8 .
```

### Documentation

- All public functions and classes must have docstrings
- Use Google-style docstrings
- Include parameter types and return types
- Add examples where appropriate

Example:
```python
def preprocess_fmri_data(fmri_path: str, output_path: str) -> nib.Nifti1Image:
    """
    Preprocess fMRI data using the standard pipeline.
    
    Args:
        fmri_path: Path to the input fMRI file
        output_path: Path to save the preprocessed output
        
    Returns:
        Preprocessed fMRI image
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        ProcessingError: If preprocessing fails
        
    Example:
        >>> preprocessed = preprocess_fmri_data("raw.nii.gz", "preprocessed.nii.gz")
        >>> isinstance(preprocessed, nib.Nifti1Image)
        True
    """
    pass
```

## Testing

### Running Tests

Run the test suite:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest --cov=schizophrenia_detection tests/
```

### Writing Tests

- Write unit tests for all new functions
- Test edge cases and error conditions
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)

Example:
```python
def test_fmri_preprocessing_with_valid_input():
    # Arrange
    input_path = "test_data/fmri.nii.gz"
    output_path = "test_data/preprocessed.nii.gz"
    
    # Act
    result = preprocess_fmri_data(input_path, output_path)
    
    # Assert
    assert isinstance(result, nib.Nifti1Image)
    assert os.path.exists(output_path)
```

## Pull Request Process

### Before Submitting

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "feat: add your feature description"
```

3. Push to your fork:
```bash
git push origin feature/your-feature-name
```

### Pull Request Guidelines

1. Use a descriptive title and description
2. Link to relevant issues
3. Include screenshots for UI changes
4. Add tests for new functionality
5. Update documentation as needed
6. Ensure all CI checks pass

### Commit Message Format

We follow the Conventional Commits specification:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for code style changes
- `refactor:` for code refactoring
- `test:` for adding or updating tests
- `chore:` for maintenance tasks

Examples:
```
feat: add support for MEG preprocessing
fix: resolve memory leak in data loader
docs: update installation instructions
```

## Notebook Style Guidelines

### General Structure

1. **Title and Description**: Clear title and brief description at the top
2. **Setup**: Import statements and configuration in the first cells
3. **Data Loading**: Load and validate data
4. **Processing**: Main analysis or processing steps
5. **Visualization**: Results visualization
6. **Conclusion**: Summary and next steps

### Code Cells

- Keep cells focused on a single task
- Add comments explaining complex operations
- Use meaningful variable names
- Include print statements for debugging (remove in final version)

### Markdown Cells

- Use markdown cells to explain each section
- Include mathematical formulas where appropriate
- Add references to relevant papers or documentation
- Use headers to structure the notebook

### Notebook Linting

We use `nbqa` to run linting tools on notebooks:

```bash
nbqa black notebooks/
nbqa isort notebooks/
nbqa flake8 notebooks/
```

### Example Notebook Structure

```markdown
# fMRI Preprocessing Pipeline

This notebook demonstrates the complete fMRI preprocessing pipeline for schizophrenia detection.

## Setup

Import required libraries and configure paths.
```

```python
# Import libraries
import numpy as np
import nibabel as nib
from schizophrenia_detection.data_processing import fmri_preprocessing

# Configure paths
data_path = "data/fmri/"
output_path = "data/preprocessed/"
```

```markdown
## Data Loading

Load the raw fMRI data and validate the format.
```

```python
# Load data
fmri_file = "sub-01_func.nii.gz"
fmri_img = nib.load(os.path.join(data_path, fmri_file))

# Validate
print(f"Image shape: {fmri_img.shape}")
print(f"Data type: {fmri_img.get_fdata().dtype}")
```

## Code Review Process

### What We Look For

- **Functionality**: Does the code work as intended?
- **Style**: Is the code well-formatted and readable?
- **Documentation**: Are functions properly documented?
- **Tests**: Are there adequate tests?
- **Performance**: Is the code efficient?

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Functions have proper docstrings
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or clearly documented)
- [ ] Performance is acceptable

## Getting Help

If you need help with contributing:

1. Check the [documentation](docs/INDEX.md)
2. Search existing issues
3. Create a new issue with the "question" label
4. Join our discussions (link to be added)

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the code of conduct

Thank you for contributing to the schizophrenia detection project!