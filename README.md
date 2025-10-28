# Scanned Document Preprocessing Project

This project is designed to collect, analyze, and preprocess scanned document samples from various scanner devices. The goal is to create a structured dataset suitable for model training and analysis.

## Project Structure

- **data/**
  - **raw/**: Contains the raw scanned document samples collected from different scanner devices.
  - **annotations/**: Contains `labels.csv`, which stores the labeled dataset with columns such as `scanner_model` and `file_name`.
  - **processed/**: Holds the processed images after preprocessing steps.

- **notebooks/**
  - **01_exploration.ipynb**: A Jupyter notebook for exploratory data analysis, including visualizing image properties and initial dataset exploration.

- **src/**
  - **__init__.py**: Marks the directory as a Python package.
  - **collect_samples.py**: Handles the collection of scanned document samples from various scanner models and saves them in the raw data directory.
  - **analyze_images.py**: Analyzes basic image properties such as resolution, format, and color channels of the scanned documents.
  - **preprocess.py**: Performs image preprocessing tasks, including resizing images, converting to grayscale, denoising, and normalizing the images.
  - **dataset.py**: Structures the dataset for model training, organizing the processed images and their corresponding labels.
  - **utils.py**: Contains utility functions that can be used across different scripts, such as image loading and saving functions.

- **scripts/**
  - **run_preprocess.py**: Serves as the entry point to run the preprocessing steps on the collected images.

- **tests/**
  - **test_preprocess.py**: Contains unit tests for the preprocessing functions to ensure they work as expected.

- **requirements.txt**: Lists the dependencies required for the project.

- **pyproject.toml**: Used for project configuration and dependency management.

- **.gitignore**: Specifies files and directories to be ignored by version control.

- **README.md**: Documentation for the project, including setup instructions and usage guidelines.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd scanned-docs-preprocessing
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Collect scanned document samples using the `collect_samples.py` script.

4. Analyze the collected images with `analyze_images.py`.

5. Preprocess the images using the `run_preprocess.py` script.

6. Explore the dataset using the Jupyter notebook `01_exploration.ipynb`.

## Usage Guidelines

- Ensure that the scanned documents are stored in the `data/raw` directory.
- The `labels.csv` file in `data/annotations` should be updated with the appropriate labels for each scanned document.
- Processed images will be saved in the `data/processed` directory after running the preprocessing script.

For any issues or contributions, please refer to the project's issue tracker or contact the maintainers.