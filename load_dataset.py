"""
=========================================================
Spotify Genre Segmentation
Dataset Loader Module
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
=========================================================
"""

from src.import_libraries import *


class DatasetLoader:
    """
    Dataset Loading and Basic Analysis
    """

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.df = None

    # =====================================================
    # Load Dataset
    # =====================================================

    def load_dataset(self):

        try:

            self.df = pd.read_csv(self.dataset_path)

            print("=" * 60)
            print("DATASET LOADED SUCCESSFULLY")
            print("=" * 60)

            print(f"Dataset Path : {self.dataset_path}")
            print(f"Rows         : {self.df.shape[0]}")
            print(f"Columns      : {self.df.shape[1]}")

            return self.df

        except Exception as e:

            print(f"Error Loading Dataset : {e}")

            return None

    # =====================================================
    # Dataset Information
    # =====================================================

    def dataset_info(self):

        if self.df is None:
            print("Load dataset first.")
            return

        print("\n" + "=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)

        self.df.info()

    # =====================================================
    # Dataset Shape
    # =====================================================

    def dataset_shape(self):

        if self.df is None:
            return None

        print("\nDataset Shape :", self.df.shape)

        return self.df.shape

    # =====================================================
    # Column Names
    # =====================================================

    def column_names(self):

        if self.df is None:
            return None

        print("\nColumns\n")

        for column in self.df.columns:

            print(column)

        return list(self.df.columns)

    # =====================================================
    # Data Types
    # =====================================================

    def data_types(self):

        if self.df is None:
            return None

        print("\nData Types\n")

        print(self.df.dtypes)

        return self.df.dtypes

    # =====================================================
    # First Records
    # =====================================================

    def head(self, rows=5):

        if self.df is None:
            return None

        print("\nFirst Records\n")

        print(self.df.head(rows))

        return self.df.head(rows)

    # =====================================================
    # Last Records
    # =====================================================

    def tail(self, rows=5):

        if self.df is None:
            return None

        print("\nLast Records\n")

        print(self.df.tail(rows))

        return self.df.tail(rows)

    # =====================================================
    # Missing Values
    # =====================================================

    def missing_values(self):

        if self.df is None:
            return None

        missing = self.df.isnull().sum()

        print("\nMissing Values\n")

        print(missing)

        return missing

    # =====================================================
    # Duplicate Values
    # =====================================================

    def duplicate_values(self):

        if self.df is None:
            return None

        duplicates = self.df.duplicated().sum()

        print("\nDuplicate Rows :", duplicates)

        return duplicates

    # =====================================================
    # Statistical Summary
    # =====================================================

    def statistical_summary(self):

        if self.df is None:
            return None

        summary = self.df.describe(include="all")

        print("\nStatistical Summary\n")

        print(summary)

        return summary

    # =====================================================
    # Numerical Features
    # =====================================================

    def numerical_columns(self):

        if self.df is None:
            return None

        columns = self.df.select_dtypes(
            include=np.number
        ).columns.tolist()

        print("\nNumerical Columns\n")

        print(columns)

        return columns

    # =====================================================
    # Categorical Features
    # =====================================================

    def categorical_columns(self):

        if self.df is None:
            return None

        columns = self.df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        print("\nCategorical Columns\n")

        print(columns)

        return columns

    # =====================================================
    # Unique Values
    # =====================================================

    def unique_values(self):

        if self.df is None:
            return None

        print("\nUnique Values\n")

        for column in self.df.columns:

            print(f"{column} : {self.df[column].nunique()}")

    # =====================================================
    # Memory Usage
    # =====================================================

    def memory_usage(self):

        if self.df is None:
            return None

        memory = self.df.memory_usage(
            deep=True
        ).sum() / 1024**2

        print(f"\nMemory Usage : {memory:.2f} MB")

        return memory

    # =====================================================
    # Save Dataset
    # =====================================================

    def save_dataset(
        self,
        output_path="data/processed/loaded_dataset.csv"
    ):

        if self.df is None:
            return

        self.df.to_csv(
            output_path,
            index=False
        )

        print(f"\nDataset Saved : {output_path}")

    # =====================================================
    # Complete Dataset Report
    # =====================================================

    def dataset_report(self):

        self.dataset_shape()

        self.column_names()

        self.data_types()

        self.missing_values()

        self.duplicate_values()

        self.memory_usage()

        self.unique_values()

        self.statistical_summary()

    # =====================================================
    # Complete Loading Pipeline
    # =====================================================

    def loading_pipeline(self):

        self.load_dataset()

        self.dataset_report()

        return self.df


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    DATASET = "data/raw/spotify_songs.csv"

    loader = DatasetLoader(DATASET)

    dataframe = loader.loading_pipeline()

    print("\nDataset Loading Completed Successfully.")
