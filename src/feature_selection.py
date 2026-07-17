"""
=========================================================
Spotify Genre Segmentation
Feature Selection Module
=========================================================
"""

from src.import_libraries import *
from src.config import *


class FeatureSelection:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

        self.features = None

    # =====================================================
    # Default Audio Features
    # =====================================================

    def select_audio_features(self):

        self.features = self.df[FEATURE_COLUMNS]

        print("=" * 60)
        print("SELECTED AUDIO FEATURES")
        print("=" * 60)

        print(self.features.columns.tolist())

        return self.features

    # =====================================================
    # Custom Feature Selection
    # =====================================================

    def select_custom_features(self, columns):

        self.features = self.df[columns]

        print("Custom Features Selected")

        print(columns)

        return self.features

    # =====================================================
    # Numerical Features
    # =====================================================

    def select_numerical_features(self):

        self.features = self.df.select_dtypes(
            include=np.number
        )

        print(self.features.columns.tolist())

        return self.features

    # =====================================================
    # Correlation Matrix
    # =====================================================

    def correlation_matrix(self):

        if self.features is None:

            self.select_audio_features()

        corr = self.features.corr()

        plt.figure(figsize=(10,8))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            annot=False
        )

        plt.title("Feature Correlation")

        plt.savefig(
            os.path.join(
                IMAGE_DIR,
                "feature_correlation.png"
            )
        )

        plt.show()

        return corr

    # =====================================================
    # Feature Variance
    # =====================================================

    def feature_variance(self):

        if self.features is None:

            self.select_audio_features()

        variance = self.features.var()

        print(variance)

        return variance

    # =====================================================
    # Save Selected Features
    # =====================================================

    def save_features(self):

        if self.features is None:

            return

        path = os.path.join(
            PROCESSED_DATA_DIR,
            "selected_features.csv"
        )

        self.features.to_csv(
            path,
            index=False
        )

        print(f"Saved : {path}")

    # =====================================================
    # Pipeline
    # =====================================================

    def feature_selection_pipeline(self):

        self.select_audio_features()

        self.correlation_matrix()

        self.feature_variance()

        self.save_features()

        print("\nFeature Selection Completed.")

        return self.features


if __name__ == "__main__":

    from src.load_dataset import DatasetLoader

    loader = DatasetLoader(DATASET_PATH)

    df = loader.loading_pipeline()

    selector = FeatureSelection(df)

    selector.feature_selection_pipeline()
