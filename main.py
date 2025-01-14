import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model tester")
    parser.add_argument(
        "--file_path",
        type=pathlib.Path,
        help="Path to the data file",
    )
    parser.add_argument(
        "--print_results",
        action="store_true",
        help="Print results to the console",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save results to a file",
    )
    parser.add_argument(
        "--print_iteration_results",
        action="store_true",
        help="Print results of each iteration",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--linear_regresion_iterations",
        type=int,
        default=500,
        help="Number of iterations for linear regression",
    )
    return parser


def get_decision_and_descriptive_data(
    data: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Splits the input DataFrame into decision (target) and descriptive (feature) data.
    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame where the first column is assumed to be the target variable (y)
        and the remaining columns are the feature variables (X).
    Returns:
    --------
    tuple[pd.Series, pd.DataFrame]
        A tuple containing:
        - pd.Series: The target variable (y) extracted from the first column of the input DataFrame.
        - pd.DataFrame: The feature variables (X) extracted from the remaining columns of the input DataFrame.
    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'target': [1, 0, 1],
    ...     'feature1': [10, 20, 30],
    ...     'feature2': [0.1, 0.2, 0.3]
    ... })
    >>> x, y = get_decision_and_descriptive_data(data)
    >>> print(y)
    0    1
    1    0
    2    1
    Name: target, dtype: int64
    >>> print(x)
       feature1  feature2
    0        10       0.1
    1        20       0.2
    2        30       0.3
    """

    y = data.iloc[:, 0]
    x = data.iloc[:, 1:]
    return x, y


def code_text_attributes(
    x: pd.DataFrame, y: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Encodes categorical text attributes in the given DataFrames using Label Encoding.
    Parameters:
    x (pd.DataFrame): The input DataFrame containing features. Columns with dtype 'object' will be label encoded.
    y (pd.DataFrame): The input DataFrame containing target labels. The entire DataFrame will be label encoded.
    Returns:
    tuple[pd.DataFrame, np.ndarray]: A tuple containing:
        - pd.DataFrame: The transformed DataFrame with encoded features.
        - np.ndarray: The transformed array with encoded target labels.
    Example:
    >>> import pandas as pd
    >>> from sklearn.preprocessing import LabelEncoder
    >>> x = pd.DataFrame({'feature1': ['A', 'B', 'A'], 'feature2': [1, 2, 3]})
    >>> y = pd.DataFrame({'target': ['yes', 'no', 'yes']})
    >>> x_encoded, y_encoded = code_text_attributes(x, y)
    >>> print(x_encoded)
       feature1  feature2
    0         0         1
    1         1         2
    2         0         3
    >>> print(y_encoded)
    [1 0 1]
    """
    encoder = LabelEncoder()
    x = x.apply(
        lambda col: encoder.fit_transform(col) if col.dtypes == "object" else col
    )
    y = encoder.fit_transform(y)
    return x, y


def normalize_data(x: pd.DataFrame) -> np.ndarray:
    """
    Normalize the features in the dataset.

    This function scales the features in the input DataFrame to have zero mean and unit variance.

    Parameters:
    x (pd.DataFrame): The input DataFrame containing the features to be normalized.

    Returns:
    np.ndarray: The normalized features as a NumPy array.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(x)


def generate_synthetic_data(
    x: pd.DataFrame, y: pd.DataFrame, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for classification tasks.
    This function generates synthetic data based on the input features and labels
    using the `make_classification` function from scikit-learn. The synthetic data
    will have the same number of features as the input data and a specified number
    of samples.
    Parameters:
    x (pd.DataFrame): The input features dataframe. The number of features in the
                      synthetic data will match the number of columns in this dataframe.
    y (pd.DataFrame): The input labels dataframe. The number of classes in the synthetic
                      data will match the number of unique values in this dataframe.
    samples (int): The number of synthetic samples to generate.
    Returns:
    tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                                   - x_synthetic: The generated synthetic features.
                                   - y_synthetic: The generated synthetic labels.
    Example:
    >>> x = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    >>> y = pd.DataFrame([0, 1, 0])
    >>> x_synthetic, y_synthetic = generate_synthetic_data(x, y, 100)
    >>> print(x_synthetic.shape)
    (100, 2)
    >>> print(y_synthetic.shape)
    (100,)
    """

    x_synthetic, y_synthetic = make_classification(
        n_samples=samples,
        n_features=x.shape[1],
        n_informative=5,
        n_classes=len(np.unique(y)),
        random_state=42,
    )
    return x_synthetic, y_synthetic


def main() -> None:
    """
    This function performs the following steps:
    1. Parses command-line arguments.
    2. Reads the input data file.
    3. Preprocesses the data by encoding categorical features and normalizing numerical features.
    4. Generates synthetic data based on the preprocessed data.
    5. Splits the synthetic data into training and testing sets.
    6. Trains multiple machine learning models on the training set.
    7. Evaluates the models on the testing set using accuracy, precision, recall, F1-score, and ROC-AUC.
    8. Prints and/or saves the results based on the command-line arguments.

    Command-line arguments:
    --file_path: Path to the data file (required).
    --print_results: Print results to the console (optional).
    --save_results: Save results to a file (optional).
    --print_iteration_results: Print results of each iteration (optional).
    --samples: Number of samples to generate for synthetic data (default: 500).
    --linear_regresion_iterations: Number of iterations for linear regression (default: 500).
    """
    parser = setup_parser()
    args = parser.parse_args()
    data = pd.read_csv(args.file_path)

    x, y = get_decision_and_descriptive_data(data)
    x, y = code_text_attributes(x, y)
    x_scaled = normalize_data(x)
    X_synthetic, y_synthetic = generate_synthetic_data(x_scaled, y, args.samples)

    x_train, x_test, y_train, y_test = train_test_split(
        X_synthetic, y_synthetic, test_size=0.2, random_state=42
    )

    results = []
    models = {
        "Logistic Regression": LogisticRegression(max_iter=args.samples),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )

        roc_auc = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)
            try:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
            except ValueError:
                print(f"ROC-AUC is not available for {name}")

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Precision": report["macro avg"]["precision"],
                "Recall": report["macro avg"]["recall"],
                "F1-score": report["macro avg"]["f1-score"],
                "ROC-AUC": roc_auc,
            }
        )
        if args.print_iteration_results:
            print(f"Model: {name}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {report['macro avg']['precision']:.4f}")
            print(f"Recall: {report['macro avg']['recall']:.4f}")
            print(f"F1-score : {report['macro avg']['f1-score']:.4f}")
            if isinstance(roc_auc, int):
                print(f"ROC-AUC: {roc_auc:.4f}")
            else:
                print("ROC-AUC: None")

    results_df = pd.DataFrame(results)
    if args.print_results:
        print(results_df)
    if args.save_results:
        results_df.to_csv(f"{args.file_path.name}_results.csv", index=False)


if __name__ == "__main__":
    main()
