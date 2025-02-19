    # preprocessing_func
import pandas as pd

def preprocessing_func(file_path):
    """
    Preprocess financial data from an Excel or CSV file.
    - Loads the data
    - Drops columns with more than 80% missing values
    - Fills missing numerical values with the mean
    - Fills missing categorical values with the mode
    - Removes duplicate rows
    - Converts data into a structured format
    """
    # Load the file based on extension
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide an Excel (.xlsx) or CSV (.csv) file.")

    # Drop columns with more than 80% missing values
    df.dropna(thresh=0.2 * len(df), axis=1, inplace=True)

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Numerical: Fill with mean
    
    for col in df.select_dtypes(include=['object']).columns:  # Categorical: Fill with mode
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Convert to dictionary format for LLM processing
    finance_data = df.to_dict(orient='records')
    
    return finance_data