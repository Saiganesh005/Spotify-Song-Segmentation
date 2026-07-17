"""
=========================================================
Spotify Genre Segmentation
Exploratory Data Analysis (EDA)
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
=========================================================
"""

from src.import_libraries import *
from src.config import *


class ExploratoryDataAnalysis:

    def __init__(self, dataframe):

        self.df = dataframe

        os.makedirs(IMAGE_DIR, exist_ok=True)

    # =====================================================
    # Save Plot
    # =====================================================

    def save_plot(self, filename):

        plt.tight_layout()

        plt.savefig(
            os.path.join(IMAGE_DIR, filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()

        print(f"Saved : {filename}")

    # =====================================================
    # Dataset Overview
    # =====================================================

    def dataset_overview(self):

        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)

        print(self.df.info())

        print("\nShape :", self.df.shape)

        print("\nColumns")

        print(self.df.columns.tolist())

    # =====================================================
    # Statistical Summary
    # =====================================================

    def statistical_summary(self):

        summary = self.df.describe(include="all")

        print(summary)

        summary.to_csv(
            os.path.join(
                REPORT_DIR,
                "statistical_summary.csv"
            )
        )

    # =====================================================
    # Missing Values
    # =====================================================

    def missing_values(self):

        missing = self.df.isnull().sum()

        print(missing)

        plt.figure(figsize=(12,5))

        missing.plot(kind="bar")

        plt.title("Missing Values")

        self.save_plot(
            "01_missing_values.png"
        )

    # =====================================================
    # Duplicate Values
    # =====================================================

    def duplicate_values(self):

        duplicates = self.df.duplicated().sum()

        print(f"Duplicate Rows : {duplicates}")

    # =====================================================
    # Correlation Heatmap
    # =====================================================

    def correlation_heatmap(self):

        plt.figure(figsize=(12,10))

        sns.heatmap(

            self.df.select_dtypes(
                include=np.number
            ).corr(),

            cmap="coolwarm",

            annot=False

        )

        plt.title("Correlation Heatmap")

        self.save_plot(
            "02_correlation_heatmap.png"
        )

    # =====================================================
    # Feature Distribution
    # =====================================================

    def feature_distribution(self):

        numerical = self.df.select_dtypes(
            include=np.number
        ).columns

        for feature in numerical:

            plt.figure(figsize=(8,4))

            sns.histplot(
                self.df[feature],
                kde=True
            )

            plt.title(feature)

            self.save_plot(
                f"distribution_{feature}.png"
            )

    # =====================================================
    # Boxplots
    # =====================================================

    def boxplots(self):

        numerical = self.df.select_dtypes(
            include=np.number
        ).columns

        for feature in numerical:

            plt.figure(figsize=(8,4))

            sns.boxplot(
                x=self.df[feature]
            )

            plt.title(feature)

            self.save_plot(
                f"boxplot_{feature}.png"
            )

    # =====================================================
    # Genre Distribution
    # =====================================================

    def genre_distribution(self):

        if "playlist_genre" not in self.df.columns:
            return

        plt.figure(figsize=(10,5))

        sns.countplot(

            data=self.df,

            x="playlist_genre",

            order=self.df[
                "playlist_genre"
            ].value_counts().index

        )

        plt.xticks(rotation=45)

        plt.title("Genre Distribution")

        self.save_plot(
            "03_genre_distribution.png"
        )

    # =====================================================
    # Subgenre Distribution
    # =====================================================

    def subgenre_distribution(self):

        if "playlist_subgenre" not in self.df.columns:
            return

        plt.figure(figsize=(14,6))

        sns.countplot(

            data=self.df,

            x="playlist_subgenre",

            order=self.df[
                "playlist_subgenre"
            ].value_counts().index

        )

        plt.xticks(rotation=90)

        plt.title("Subgenre Distribution")

        self.save_plot(
            "04_subgenre_distribution.png"
        )

    # =====================================================
    # Popular Songs
    # =====================================================

    def popularity_distribution(self):

        if "track_popularity" not in self.df.columns:
            return

        plt.figure(figsize=(8,5))

        sns.histplot(
            self.df["track_popularity"],
            kde=True
        )

        plt.title("Track Popularity")

        self.save_plot(
            "05_track_popularity.png"
        )

    # =====================================================
    # Pair Plot
    # =====================================================

    def pairplot(self):

        numerical = self.df.select_dtypes(
            include=np.number
        )

        sns.pairplot(
            numerical.sample(
                min(500, len(numerical)),
                random_state=RANDOM_STATE
            )
        )

        plt.savefig(
            os.path.join(
                IMAGE_DIR,
                "06_pairplot.png"
            )
        )

        plt.show()

    # =====================================================
    # Feature Relationships
    # =====================================================

    def scatter_plots(self):

        pairs = [

            ("danceability", "energy"),

            ("energy", "valence"),

            ("tempo", "danceability"),

            ("loudness", "energy")

        ]

        for x, y in pairs:

            if x in self.df.columns and y in self.df.columns:

                plt.figure(figsize=(7,5))

                sns.scatterplot(

                    data=self.df,

                    x=x,

                    y=y,

                    alpha=0.5

                )

                plt.title(f"{x} vs {y}")

                self.save_plot(
                    f"{x}_vs_{y}.png"
                )

    # =====================================================
    # Dataset Report
    # =====================================================

    def generate_report(self):

        report = {

            "Rows": self.df.shape[0],

            "Columns": self.df.shape[1],

            "Missing Values":
                self.df.isnull().sum().sum(),

            "Duplicate Rows":
                self.df.duplicated().sum(),

            "Numerical Features":
                len(
                    self.df.select_dtypes(
                        include=np.number
                    ).columns
                ),

            "Categorical Features":
                len(
                    self.df.select_dtypes(
                        exclude=np.number
                    ).columns
                )

        }

        report = pd.DataFrame(
            report,
            index=[0]
        )

        report.to_csv(

            os.path.join(
                REPORT_DIR,
                "eda_report.csv"
            ),

            index=False

        )

        print(report)

    # =====================================================
    # Complete EDA Pipeline
    # =====================================================

    def eda_pipeline(self):

        self.dataset_overview()

        self.statistical_summary()

        self.missing_values()

        self.duplicate_values()

        self.correlation_heatmap()

        self.feature_distribution()

        self.boxplots()

        self.genre_distribution()

        self.subgenre_distribution()

        self.popularity_distribution()

        self.scatter_plots()

        self.pairplot()

        self.generate_report()

        print("\nEDA Completed Successfully.")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    from src.load_dataset import DatasetLoader

    loader = DatasetLoader(DATASET_PATH)

    dataframe = loader.loading_pipeline()

    eda = ExploratoryDataAnalysis(dataframe)

    eda.eda_pipeline()
