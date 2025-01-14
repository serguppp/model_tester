# Model Tester

This repository contains the `main.py` script, which is used for testing machine learning models.

## Requirements

- Python 3.12
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/model_tester.git
    ```
2. Navigate to the project directory:
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

usage: main.py [-h] [--file_path FILE_PATH] [--print_results] [--save_results] [--print_iteration_results] [--samples SAMPLES] [--linear_regresion_iterations LINEAR_REGRESION_ITERATIONS]

options:
  -h, --help            show help message and exit
  --file_path FILE_PATH
                        Path to the data file
  --print_results       Print results to the console
  --save_results        Save results to a file
  --print_iteration_results
                        Print results of each iteration
  --samples SAMPLES     Number of samples to generate
  --linear_regresion_iterations LINEAR_REGRESION_ITERATIONS
                        Number of iterations for linear regression

## main.py

The `main.py` script is responsible for:

- Loading the machine learning model
- Running tests on the model
- Outputting the test results

