"""
=========================================================
Spotify Genre Segmentation
Data Preprocessing Module
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
Project: Spotify Songs Genre Segmentation
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
import joblib


class DataPreprocessor:
    """
    Data preprocessing pipeline for Spotify dataset.
    """

    def __init__(self, filepath):

        self.filepath = filepath
        self.df = None
        self.scaler = StandardScaler()

        # Features used for ML/DL
        self.features = [
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
            "duration_ms",
            "time_signature"
        ]

    # ----------------------------------------------------

    def load_dataset(self):

        """Load CSV Dataset"""

        print("=" * 60)
        print("Loading Dataset")
        print("=" * 60)

        self.df = pd.read_csv(self.filepath)

        print("Dataset Loaded Successfully")
        print("Shape :", self.df.shape)

        return self.df

    # ----------------------------------------------------

    def dataset_information(self):

        print("\nDataset Information\n")

        print(self.df.info())

    # ----------------------------------------------------

    def statistical_summary(self):

        print("\nStatistical Summary\n")

        print(self.df.describe())

    # ----------------------------------------------------

    def missing_values(self):

        print("\nMissing Values\n")

        print(self.df.isnull().sum())

    # ----------------------------------------------------

    def remove_missing_values(self):

        before = self.df.shape[0]

        self.df.dropna(inplace=True)

        after = self.df.shape[0]

        print("\nMissing Values Removed")

        print("Rows Removed :", before - after)

    # ----------------------------------------------------

    def duplicate_values(self):

        duplicates = self.df.duplicated().sum()

        print("\nDuplicate Rows :", duplicates)

    # ----------------------------------------------------

    def remove_duplicates(self):

        before = self.df.shape[0]

        self.df.drop_duplicates(inplace=True)

        after = self.df.shape[0]

        print("\nDuplicate Rows Removed")

        print("Rows Removed :", before - after)

    # ----------------------------------------------------

    def unique_values(self):

        print("\nUnique Values\n")

        for column in self.df.columns:

            print(f"{column:25} : {self.df[column].nunique()}")

    # ----------------------------------------------------

    def select_features(self):

        print("\nSelected Features")

        print(self.features)

        X = self.df[self.features]

        return X

    # ----------------------------------------------------

    def feature_scaling(self):

        print("\nScaling Features...")

        X = self.select_features()

        X_scaled = self.scaler.fit_transform(X)

        print("Scaling Completed")

        return X_scaled

    # ----------------------------------------------------

    def save_scaler(self,
                    filename="models/spotify_scaler.pkl"):

        joblib.dump(self.scaler, filename)

        print(f"\nScaler Saved : {filename}")

    # ----------------------------------------------------

    def save_processed_dataset(
            self,
            filename="data/processed/spotify_clustered_songs.csv"
    ):

        self.df.to_csv(filename, index=False)

        print(f"\nProcessed Dataset Saved : {filename}")

    # ----------------------------------------------------

    def preprocessing_pipeline(self):

        """
        Complete preprocessing pipeline
        """

        self.load_dataset()

        self.dataset_information()

        self.statistical_summary()

        self.missing_values()

        self.remove_missing_values()

        self.duplicate_values()

        self.remove_duplicates()

        self.unique_values()

        X_scaled = self.feature_scaling()

        self.save_scaler()

        return self.df, X_scaled


# =========================================================
# Standalone Execution
# =========================================================

if __name__ == "__main__":

    DATASET = "data/raw/spotify_songs.csv"

    processor = DataPreprocessor(DATASET)

    df, X_scaled = processor.preprocessing_pipeline()

    print("\nPreprocessing Completed Successfully")

    print("Scaled Data Shape :", X_scaled.shape)
