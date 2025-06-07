import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_dataset(input_path: str, output_dir: str):
    """
    Load dataset, remove duplicates, split into train/test sets,
    separate features and targets, save CSVs, and return outputs.
    """
    df = pd.read_csv(input_path)
    df_clean = df.drop_duplicates()

    # Define features and targets
    feature_cols = [
        'Relative_Compactness', 'Surface_Area', 'Wall_Area', 'Roof_Area',
        'Overall_Height', 'Orientation', 'Glazing_Area', 'Glazing_Area_Distribution'
    ]
    target_cols = ['Heating_Load', 'Cooling_Load']

    # Split features and targets
    X = df_clean[feature_cols]
    y = df_clean[target_cols]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save full cleaned dataset
    cleaned_path = os.path.join(output_dir, "dataset_clean.csv")
    df_clean.to_csv(cleaned_path, index=False)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, test_df, df_clean, cleaned_path, train_path, test_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "energy_efficiency_data.csv")
    output_dir = os.path.join(base_dir, "outputs")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file://" + os.path.abspath("../mlruns"))
    mlflow.set_experiment("Energy_Efficiency_Preprocessing")

    with mlflow.start_run(run_name="Preprocessing"):
        train_df, test_df, df_clean, cleaned_path, train_path, test_path = preprocess_dataset(
            input_file, output_dir
        )

        # Log params
        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_param("n_features", 8)
        mlflow.log_param("n_targets", 2)

        # Log metrics
        mlflow.log_metric("rows_cleaned", df_clean.shape[0])
        mlflow.log_metric("train_rows", train_df.shape[0])
        mlflow.log_metric("test_rows", test_df.shape[0])

        # Log artifacts
        mlflow.log_artifact(cleaned_path)
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)
