"""
=========================================================
Spotify Genre Segmentation
Visualization Module (Part 1)
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
Project : Spotify Genre Segmentation
=========================================================
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --------------------------------------------------------
# Create Output Directory
# --------------------------------------------------------

os.makedirs("outputs/images", exist_ok=True)

# --------------------------------------------------------
# Plot Style
# --------------------------------------------------------

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")


# ========================================================
# Visualization Class
# ========================================================

class DataVisualization:

    def __init__(self, dataframe):

        self.df = dataframe

    # ====================================================
    # Save Plot
    # ====================================================

    def save_plot(self, filename):

        plt.tight_layout()

        plt.savefig(
            f"outputs/images/{filename}",
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved : outputs/images/{filename}")

        plt.show()

    # ====================================================
    # Dataset Information
    # ====================================================

    def dataset_information(self):

        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)

        print("Rows :", self.df.shape[0])
        print("Columns :", self.df.shape[1])

        print("\nColumn Names\n")

        print(self.df.columns.tolist())

        print("\nData Types\n")

        print(self.df.dtypes)

    # ====================================================
    # Missing Values
    # ====================================================

    def missing_values(self):

        missing = self.df.isnull().sum()

        plt.figure(figsize=(10,5))

        missing.plot(
            kind="bar"
        )

        plt.title("Missing Values")

        plt.ylabel("Count")

        self.save_plot(
            "01_missing_values.png"
        )

        return missing

    # ====================================================
    # Playlist Genre Distribution
    # ====================================================

    def playlist_genre_distribution(self):

        if "playlist_genre" not in self.df.columns:
            return

        plt.figure(figsize=(10,6))

        sns.countplot(
            data=self.df,
            x="playlist_genre",
            order=self.df["playlist_genre"]
            .value_counts()
            .index
        )

        plt.xticks(rotation=45)

        plt.title("Playlist Genre Distribution")

        self.save_plot(
            "02_playlist_genre_distribution.png"
        )

    # ====================================================
    # Playlist Subgenre Distribution
    # ====================================================

    def playlist_subgenre_distribution(self):

        if "playlist_subgenre" not in self.df.columns:
            return

        plt.figure(figsize=(14,6))

        sns.countplot(
            data=self.df,
            x="playlist_subgenre",
            order=self.df["playlist_subgenre"]
            .value_counts()
            .index
        )

        plt.xticks(rotation=90)

        plt.title("Playlist Subgenre Distribution")

        self.save_plot(
            "03_playlist_subgenre_distribution.png"
        )

    # ====================================================
    # Track Popularity
    # ====================================================

    def track_popularity(self):

        if "track_popularity" not in self.df.columns:
            return

        plt.figure(figsize=(8,5))

        plt.hist(
            self.df["track_popularity"],
            bins=30
        )

        plt.xlabel("Popularity")

        plt.ylabel("Frequency")

        plt.title("Track Popularity Distribution")

        self.save_plot(
            "04_track_popularity.png"
        )

    # ====================================================
    # Danceability Histogram
    # ====================================================

    def danceability(self):

        plt.figure(figsize=(8,5))

        plt.hist(
            self.df["danceability"],
            bins=30
        )

        plt.xlabel("Danceability")

        plt.ylabel("Frequency")

        plt.title("Danceability Distribution")

        self.save_plot(
            "05_danceability.png"
        )

    # ====================================================
    # Energy Histogram
    # ====================================================

    def energy(self):

        plt.figure(figsize=(8,5))

        plt.hist(
            self.df["energy"],
            bins=30
        )

        plt.xlabel("Energy")

        plt.ylabel("Frequency")

        plt.title("Energy Distribution")

        self.save_plot(
            "06_energy.png"
        )

    # ====================================================
    # Loudness Histogram
    # ====================================================

    def loudness(self):

        plt.figure(figsize=(8,5))

        plt.hist(
            self.df["loudness"],
            bins=30
        )

        plt.xlabel("Loudness")

        plt.ylabel("Frequency")

        plt.title("Loudness Distribution")

        self.save_plot(
            "07_loudness.png"
        )

    # ====================================================
    # Tempo Histogram
    # ====================================================

    def tempo(self):

        plt.figure(figsize=(8,5))

        plt.hist(
            self.df["tempo"],
            bins=30
        )

        plt.xlabel("Tempo")

        plt.ylabel("Frequency")

        plt.title("Tempo Distribution")

        self.save_plot(
            "08_tempo.png"
        )

    # ====================================================
    # Valence Histogram
    # ====================================================

    def valence(self):

        plt.figure(figsize=(8,5))

        plt.hist(
            self.df["valence"],
            bins=30
        )

        plt.xlabel("Valence")

        plt.ylabel("Frequency")

        plt.title("Valence Distribution")

        self.save_plot(
            "09_valence.png"
        )

    # ====================================================
    # Run Basic Visualizations
    # ====================================================

    def run_basic_visualizations(self):

        self.dataset_information()

        self.missing_values()

        self.playlist_genre_distribution()

        self.playlist_subgenre_distribution()

        self.track_popularity()

        self.danceability()

        self.energy()

        self.loudness()

        self.tempo()

        self.valence()

        print("\nBasic Visualizations Completed.")
# =====================================================
# Correlation Heatmap
# =====================================================

def correlation_heatmap(self):

    numeric_df = self.df.select_dtypes(include=np.number)

    plt.figure(figsize=(14,10))

    sns.heatmap(

        numeric_df.corr(),

        annot=True,

        cmap="coolwarm",

        fmt=".2f",

        linewidths=0.5

    )

    plt.title("Correlation Heatmap")

    self.save_plot(
        "10_correlation_heatmap.png"
    )


# =====================================================
# Boxplots
# =====================================================

def boxplots(self):

    numeric_df = self.df.select_dtypes(include=np.number)

    plt.figure(figsize=(16,8))

    sns.boxplot(
        data=numeric_df
    )

    plt.xticks(rotation=90)

    plt.title("Feature Boxplots")

    self.save_plot(
        "11_boxplots.png"
    )


# =====================================================
# Pairplot
# =====================================================

def pairplot(self):

    features = [

        "danceability",

        "energy",

        "valence",

        "tempo"

    ]

    available = [

        col

        for col in features

        if col in self.df.columns

    ]

    if len(available) < 2:

        return

    sns.pairplot(

        self.df[available],

        diag_kind="hist"

    )

    plt.savefig(

        "outputs/images/12_pairplot.png",

        dpi=300,

        bbox_inches="tight"

    )

    plt.show()

    print("Saved : outputs/images/12_pairplot.png")


# =====================================================
# Feature Distribution
# =====================================================

def feature_distribution(self):

    numeric_columns = [

        "danceability",

        "energy",

        "speechiness",

        "acousticness",

        "instrumentalness",

        "liveness",

        "valence"

    ]

    available = [

        col

        for col in numeric_columns

        if col in self.df.columns

    ]

    for feature in available:

        plt.figure(figsize=(8,5))

        sns.histplot(

            self.df[feature],

            kde=True,

            bins=30

        )

        plt.title(f"{feature} Distribution")

        self.save_plot(

            f"distribution_{feature}.png"

        )


# =====================================================
# Violin Plots
# =====================================================

def violin_plots(self):

    if "playlist_genre" not in self.df.columns:

        return

    features = [

        "danceability",

        "energy",

        "valence"

    ]

    for feature in features:

        if feature not in self.df.columns:

            continue

        plt.figure(figsize=(10,6))

        sns.violinplot(

            data=self.df,

            x="playlist_genre",

            y=feature

        )

        plt.xticks(rotation=45)

        plt.title(f"{feature} by Playlist Genre")

        self.save_plot(

            f"violin_{feature}.png"

        )


# =====================================================
# Scatter Plots
# =====================================================

def scatter_plots(self):

    scatter_pairs = [

        ("danceability", "energy"),

        ("energy", "valence"),

        ("tempo", "danceability"),

        ("loudness", "energy")

    ]

    for x, y in scatter_pairs:

        if x not in self.df.columns:

            continue

        if y not in self.df.columns:

            continue

        plt.figure(figsize=(8,6))

        sns.scatterplot(

            data=self.df,

            x=x,

            y=y,

            alpha=0.6

        )

        plt.title(f"{x} vs {y}")

        self.save_plot(

            f"{x}_vs_{y}.png"

        )


# =====================================================
# KDE Distribution
# =====================================================

def kde_plots(self):

    features = [

        "danceability",

        "energy",

        "valence",

        "tempo"

    ]

    for feature in features:

        if feature not in self.df.columns:

            continue

        plt.figure(figsize=(8,5))

        sns.kdeplot(

            self.df[feature],

            fill=True

        )

        plt.title(f"{feature} KDE Distribution")

        self.save_plot(

            f"kde_{feature}.png"

        )


# =====================================================
# Numerical Summary
# =====================================================

def numerical_summary(self):

    numeric = self.df.select_dtypes(include=np.number)

    print("=" * 60)

    print("NUMERICAL SUMMARY")

    print("=" * 60)

    print(

        numeric.describe()

    )


# =====================================================
# Run Statistical Visualizations
# =====================================================

def run_statistical_visualizations(self):

    self.correlation_heatmap()

    self.boxplots()

    self.pairplot()

    self.feature_distribution()

    self.violin_plots()

    self.scatter_plots()

    self.kde_plots()

    self.numerical_summary()

    print("\nStatistical Visualizations Completed.")
# =====================================================
# Required Imports
# =====================================================

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples


# =====================================================
# Elbow Method
# =====================================================

def elbow_method(
        self,
        X,
        max_clusters=10):

    inertia = []

    for k in range(1, max_clusters + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        model.fit(X)

        inertia.append(model.inertia_)

    plt.figure(figsize=(8,5))

    plt.plot(

        range(1, max_clusters + 1),

        inertia,

        marker="o",

        linewidth=2

    )

    plt.xlabel("Number of Clusters")

    plt.ylabel("Inertia")

    plt.title("Elbow Method")

    self.save_plot(
        "13_elbow_method.png"
    )


# =====================================================
# KMeans Cluster Distribution
# =====================================================

def cluster_distribution(self):

    if "Cluster" not in self.df.columns:
        return

    plt.figure(figsize=(8,5))

    sns.countplot(

        data=self.df,

        x="Cluster",

        order=sorted(
            self.df["Cluster"].unique()
        )

    )

    plt.title("Cluster Distribution")

    self.save_plot(
        "14_cluster_distribution.png"
    )


# =====================================================
# PCA Cluster Visualization
# =====================================================

def pca_clusters(
        self,
        X):

    if "Cluster" not in self.df.columns:
        return

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(X)

    plt.figure(figsize=(10,8))

    plt.scatter(

        reduced[:,0],

        reduced[:,1],

        c=self.df["Cluster"],

        cmap="viridis",

        s=20

    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("K-Means PCA Clusters")

    plt.colorbar(label="Cluster")

    self.save_plot(
        "15_pca_clusters.png"
    )


# =====================================================
# Genre vs Cluster
# =====================================================

def genre_vs_cluster(self):

    if "playlist_genre" not in self.df.columns:
        return

    if "Cluster" not in self.df.columns:
        return

    plt.figure(figsize=(12,6))

    sns.countplot(

        data=self.df,

        x="playlist_genre",

        hue="Cluster"

    )

    plt.xticks(rotation=45)

    plt.title("Genre vs Cluster")

    self.save_plot(
        "16_genre_vs_cluster.png"
    )


# =====================================================
# Playlist vs Cluster
# =====================================================

def playlist_vs_cluster(self):

    if "playlist_name" not in self.df.columns:
        return

    if "Cluster" not in self.df.columns:
        return

    top_playlist = self.df[
        "playlist_name"
    ].value_counts().head(15).index

    subset = self.df[
        self.df["playlist_name"].isin(
            top_playlist
        )
    ]

    plt.figure(figsize=(15,6))

    sns.countplot(

        data=subset,

        x="playlist_name",

        hue="Cluster"

    )

    plt.xticks(rotation=90)

    plt.title("Playlist vs Cluster")

    self.save_plot(
        "17_playlist_vs_cluster.png"
    )


# =====================================================
# Cluster Centers
# =====================================================

def cluster_centers(
        self,
        X):

    if "Cluster" not in self.df.columns:
        return

    model = KMeans(

        n_clusters=len(
            self.df["Cluster"].unique()
        ),

        random_state=42,

        n_init=20

    )

    model.fit(X)

    centers = pd.DataFrame(
        model.cluster_centers_
    )

    plt.figure(figsize=(14,6))

    sns.heatmap(

        centers,

        cmap="coolwarm",

        annot=True,

        fmt=".2f"

    )

    plt.title("Cluster Centers")

    self.save_plot(
        "18_cluster_centers.png"
    )


# =====================================================
# Silhouette Analysis
# =====================================================

def silhouette_analysis(
        self,
        X):

    if "Cluster" not in self.df.columns:
        return

    values = silhouette_samples(

        X,

        self.df["Cluster"]

    )

    plt.figure(figsize=(8,5))

    plt.hist(

        values,

        bins=25

    )

    plt.xlabel("Silhouette Score")

    plt.ylabel("Frequency")

    plt.title("Silhouette Distribution")

    self.save_plot(
        "19_silhouette_analysis.png"
    )


# =====================================================
# Cluster Popularity
# =====================================================

def cluster_popularity(self):

    if "Cluster" not in self.df.columns:
        return

    if "track_popularity" not in self.df.columns:
        return

    plt.figure(figsize=(8,5))

    sns.boxplot(

        data=self.df,

        x="Cluster",

        y="track_popularity"

    )

    plt.title("Track Popularity by Cluster")

    self.save_plot(
        "20_cluster_popularity.png"
    )


# =====================================================
# Cluster Summary
# =====================================================

def cluster_summary(self):

    if "Cluster" not in self.df.columns:
        return

    summary = self.df.groupby(
        "Cluster"
    ).mean(
        numeric_only=True
    )

    print("=" * 60)

    print("CLUSTER SUMMARY")

    print("=" * 60)

    print(summary)

    summary.to_csv(
        "outputs/ml_cluster_summary.csv"
    )


# =====================================================
# Run Machine Learning Visualizations
# =====================================================

def run_ml_visualizations(
        self,
        X):

    self.elbow_method(X)

    self.cluster_distribution()

    self.pca_clusters(X)

    self.genre_vs_cluster()

    self.playlist_vs_cluster()

    self.cluster_centers(X)

    self.silhouette_analysis(X)

    self.cluster_popularity()

    self.cluster_summary()

    print("\nMachine Learning Visualizations Completed.")
# =====================================================
# Deep Learning Training Loss
# =====================================================

def training_loss(self, history):

    if history is None:
        return

    plt.figure(figsize=(8,5))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
        linewidth=2
    )

    if "val_loss" in history.history:

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss",
            linewidth=2
        )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title("Deep Learning Training Loss")

    plt.legend()

    self.save_plot(
        "21_dl_training_loss.png"
    )


# =====================================================
# Latent Space PCA
# =====================================================

def latent_pca(
        self,
        encoded_features,
        labels):

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(
        encoded_features
    )

    plt.figure(figsize=(10,8))

    plt.scatter(

        reduced[:,0],

        reduced[:,1],

        c=labels,

        cmap="plasma",

        s=20

    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("Latent Feature Space")

    plt.colorbar(label="Cluster")

    self.save_plot(
        "22_latent_pca.png"
    )


# =====================================================
# Reconstruction Error
# =====================================================

def reconstruction_error(
        self,
        reconstruction_errors):

    plt.figure(figsize=(8,5))

    plt.hist(

        reconstruction_errors,

        bins=35

    )

    plt.xlabel("Reconstruction Error")

    plt.ylabel("Frequency")

    plt.title("Reconstruction Error Distribution")

    self.save_plot(
        "23_reconstruction_error.png"
    )


# =====================================================
# Deep Cluster Distribution
# =====================================================

def deep_cluster_distribution(self):

    if "DL_Cluster" not in self.df.columns:
        return

    plt.figure(figsize=(8,5))

    sns.countplot(

        data=self.df,

        x="DL_Cluster",

        order=sorted(
            self.df["DL_Cluster"].unique()
        )

    )

    plt.title("Deep Learning Cluster Distribution")

    self.save_plot(
        "24_dl_cluster_distribution.png"
    )


# =====================================================
# Genre vs Deep Cluster
# =====================================================

def deep_cluster_vs_genre(self):

    if "DL_Cluster" not in self.df.columns:
        return

    if "playlist_genre" not in self.df.columns:
        return

    plt.figure(figsize=(12,6))

    sns.countplot(

        data=self.df,

        x="playlist_genre",

        hue="DL_Cluster"

    )

    plt.xticks(rotation=45)

    plt.title("Genre vs Deep Cluster")

    self.save_plot(
        "25_dl_genre_vs_cluster.png"
    )


# =====================================================
# Playlist vs Deep Cluster
# =====================================================

def deep_cluster_vs_playlist(self):

    if "DL_Cluster" not in self.df.columns:
        return

    if "playlist_name" not in self.df.columns:
        return

    top = self.df[
        "playlist_name"
    ].value_counts().head(15).index

    subset = self.df[
        self.df["playlist_name"].isin(top)
    ]

    plt.figure(figsize=(15,6))

    sns.countplot(

        data=subset,

        x="playlist_name",

        hue="DL_Cluster"

    )

    plt.xticks(rotation=90)

    plt.title("Playlist vs Deep Cluster")

    self.save_plot(
        "26_dl_playlist_vs_cluster.png"
    )


# =====================================================
# Deep Cluster Summary
# =====================================================

def deep_cluster_summary(self):

    if "DL_Cluster" not in self.df.columns:
        return

    summary = self.df.groupby(

        "DL_Cluster"

    ).mean(

        numeric_only=True

    )

    print("=" * 60)

    print("DEEP LEARNING CLUSTER SUMMARY")

    print("=" * 60)

    print(summary)

    summary.to_csv(

        "outputs/deep_learning_cluster_summary.csv"

    )


# =====================================================
# Run Deep Learning Visualizations
# =====================================================

def run_dl_visualizations(
        self,
        history,
        encoded_features,
        labels,
        reconstruction_errors):

    self.training_loss(history)

    self.latent_pca(
        encoded_features,
        labels
    )

    self.reconstruction_error(
        reconstruction_errors
    )

    self.deep_cluster_distribution()

    self.deep_cluster_vs_genre()

    self.deep_cluster_vs_playlist()

    self.deep_cluster_summary()

    print("\nDeep Learning Visualizations Completed.")


# =====================================================
# Run All Visualizations
# =====================================================

def run_all_visualizations(
        self,
        X_scaled=None,
        history=None,
        encoded_features=None,
        labels=None,
        reconstruction_errors=None):

    print("=" * 60)
    print("RUNNING ALL VISUALIZATIONS")
    print("=" * 60)

    # -------------------------
    # Basic EDA
    # -------------------------

    self.run_basic_visualizations()

    # -------------------------
    # Statistical Analysis
    # -------------------------

    self.run_statistical_visualizations()

    # -------------------------
    # Machine Learning
    # -------------------------

    if X_scaled is not None:

        self.run_ml_visualizations(
            X_scaled
        )

    # -------------------------
    # Deep Learning
    # -------------------------

    if (

        history is not None

        and

        encoded_features is not None

        and

        labels is not None

        and

        reconstruction_errors is not None

    ):

        self.run_dl_visualizations(

            history,

            encoded_features,

            labels,

            reconstruction_errors

        )

    print("\nAll Visualizations Generated Successfully.")
if __name__ == "__main__":

    from data_preprocessing import DataPreprocessor

    DATASET = "data/raw/spotify_songs.csv"

    processor = DataPreprocessor(DATASET)

    dataframe, X_scaled = processor.preprocessing_pipeline()

    visualizer = DataVisualization(dataframe)

    visualizer.run_basic_visualizations()

    visualizer.run_statistical_visualizations()

    print("\nVisualization Module Executed Successfully.")
