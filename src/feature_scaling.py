"""
=========================================================
Spotify Genre Segmentation
Feature Scaling Module
=========================================================
"""

from src.import_libraries import *
from src.config import *


class FeatureScaling:

    def __init__(self, features):

        self.features = features

        self.scaler = StandardScaler()

        self.scaled_features = None

    # =====================================================
    # Standard Scaling
    # =====================================================

    def standard_scaling(self):

        self.scaled_features = self.scaler.fit_transform(
            self.features
        )

        print("Standard Scaling Completed.")

        return self.scaled_features

    # =====================================================
    # Min-Max Scaling
    # =====================================================

    def minmax_scaling(self):

        scaler = MinMaxScaler()

        self.scaled_features = scaler.fit_transform(
            self.features
        )

        print("Min-Max Scaling Completed.")

        return self.scaled_features

    # =====================================================
    # Convert to DataFrame
    # =====================================================

    def scaled_dataframe(self):

        if self.scaled_features is None:

            self.standard_scaling()

        return pd.DataFrame(

            self.scaled_features,

            columns=self.features.columns

        )

    # =====================================================
    # Save Scaler
    # =====================================================

    def save_scaler(self):

        joblib.dump(
            self.scaler,
            SCALER_FILE
        )

        print(f"Scaler Saved : {SCALER_FILE}")

    # =====================================================
    # Load Scaler
    # =====================================================

    def load_scaler(self):

        self.scaler = joblib.load(
            SCALER_FILE
        )

        print("Scaler Loaded.")

    # =====================================================
    # Transform New Data
    # =====================================================

    def transform(self, data):

        return self.scaler.transform(data)

    # =====================================================
    # Save Scaled Dataset
    # =====================================================

    def save_scaled_dataset(self):

        df = self.scaled_dataframe()

        path = os.path.join(
            PROCESSED_DATA_DIR,
            "scaled_features.csv"
        )

        df.to_csv(
            path,
            index=False
        )

        print(f"Saved : {path}")

    # =====================================================
    # Scaling Pipeline
    # =====================================================

    def scaling_pipeline(self):

        self.standard_scaling()

        self.save_scaler()

        self.save_scaled_dataset()

        print("\nFeature Scaling Completed.")

        return self.scaled_dataframe()


if __name__ == "__main__":

    from src.load_dataset import DatasetLoader
    from src.feature_selection import FeatureSelection

    loader = DatasetLoader(DATASET_PATH)

    df = loader.loading_pipeline()

    selector = FeatureSelection(df)

    features = selector.feature_selection_pipeline()

    scaler = FeatureScaling(features)

    scaler.scaling_pipeline()
