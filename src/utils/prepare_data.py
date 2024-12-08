import os
import zipfile
import pandas as pd


def extract_dataset(zip_path: str, extract_to: str) -> str:
    """
    Extracts the Enron Spam Dataset from a zip file and returns the CSV file path.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory where the data should be extracted.

    Returns:
        str: Path to the extracted CSV file.

    Raises:
        FileNotFoundError: If the zip file does not exist.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset not found: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Find the extracted CSV file
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.endswith('.csv'):
                return os.path.join(root, file)

    raise FileNotFoundError("No CSV file found in the extracted dataset.")


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Loads emails and their attributes from the extracted CSV file.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the dataset by normalizing the email messages.

    Args:
        df (pd.DataFrame): DataFrame containing raw email data.

    Returns:
        pd.DataFrame: DataFrame with normalized email content.
    """
    df['Message'] = df['Message'].str.lower()
    df['Message'] = df['Message'].str.replace(r'<[^>]+>', '', regex=True)  # Remove HTML tags
    df['Message'] = df['Message'].str.replace(r'\s+', ' ', regex=True)    # Normalize spaces
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the dataset into training, validation, and testing sets.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        test_size (float): Proportion of data to allocate to testing and validation.
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: DataFrames for train, validation, and test sets.
    """
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train, test_size=test_size, random_state=random_state)
    return train, val, test


def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, output_dir: str) -> None:
    """
    Saves the train, validation, and test sets to CSV files.

    Args:
        train (pd.DataFrame): Training set.
        val (pd.DataFrame): Validation set.
        test (pd.DataFrame): Test set.
        output_dir (str): Directory where the CSV files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Data splits saved to {output_dir}")