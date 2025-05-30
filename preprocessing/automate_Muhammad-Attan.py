import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

def load_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' tidak ditemukan. Lokasi saat ini: {os.getcwd()}")
    return pd.read_csv(file_path)

def handle_missing_values(df):
    return df.fillna(df.mean(numeric_only=True))

def handle_duplicate_data(df):
    return df.drop_duplicates()

def outlier_handling_all_columns(df):
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df

def bin_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal weight'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

def encode_categorical_features(df):
    df = df.copy()
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
    return df

def scale_features(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return df
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def preprocess_data(file_path, target_column):
    df = load_data(file_path)
    df = handle_missing_values(df)
    df = handle_duplicate_data(df)
    df = outlier_handling_all_columns(df)

    # Binning BMI → BMI_Category (integer encoded)
    if 'BMI' in df.columns:
        df['BMI_Category'] = df['BMI'].apply(bin_bmi)
        

    # Pisahkan fitur dan target
    X = df.drop(columns=[target_column,'BMI'])
    y = df[target_column]

    # Scaling fitur numerik
    X = scale_features(X)

    # Encode fitur kategorik 
    X = encode_categorical_features(X)


    # Gabungkan kembali
    processed_data = pd.concat([X, y], axis=1)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return processed_data, X_train, X_test, y_train, y_test

def save_processed_data(df, output_path):
    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to: {output_path}")

if __name__ == "__main__":
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../diabetes_dataset_raw.csv'))
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'diabetes_dataset_processing.csv'))
    target_column = 'Outcome'

    processed_data, X_train, X_test, y_train, y_test = preprocess_data(file_path, target_column)
    save_processed_data(processed_data, output_path)
