"""
=========================================================
Spotify Genre Segmentation
Visualization Module (Part 1)
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
Project : Spotify Songs Genre Segmentation using
          Machine Learning & Deep Learning
=========================================================
"""

import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --------------------------------------------------------
# Plot Style
# --------------------------------------------------------

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

# --------------------------------------------------------
# Create Output Folder
# --------------------------------------------------------

os.makedirs("outputs/images", exist_ok=True)


class DataVisualization:

    def __init__(self, dataframe):

        self.df = dataframe

    # =====================================================
    # Save Figure
    # =====================================================

    def save_plot(self, filename):

        plt.tight_layout()

        plt.savefig(
            f"outputs/images/{filename}",
            dpi=300,
            bbox_inches="tight"
        )

        print(f"{filename} Saved Successfully")

        plt.show()

    # =====================================================
    # Dataset Overview
    # =====================================================

    def dataset_overview(self):

        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)

        print("\nShape")

        print(self.df.shape)

        print("\nColumns")

        print(self.df.columns.tolist())

        print("\nData Types")

        print(self.df.dtypes)

        print("\nMissing Values")

        print(self.df.isnull().sum())

        print("\nDuplicate Rows")

        print(self.df.duplicated().sum())

    # =====================================================
    # Playlist Genre Distribution
    # =====================================================

    def playlist_genre_distribution(self):

        plt.figure(figsize=(10,6))

        order = self.df["playlist_genre"].value_counts().index

        sns.countplot(
            data=self.df,
            x="playlist_genre",
            order=order,
            palette="viridis"
        )

        plt.title("Playlist Genre Distribution")

        plt.xlabel("Playlist Genre")

        plt.ylabel("Number of Songs")

        plt.xticks(rotation=30)

        self.save_plot("02_playlist_genre.png")

    # =====================================================
    # Playlist Subgenre Distribution
    # =====================================================

    def playlist_subgenre_distribution(self):

        plt.figure(figsize=(14,6))

        order = self.df["playlist_subgenre"].value_counts().index

        sns.countplot(
            data=self.df,
            x="playlist_subgenre",
            order=order,
            palette="magma"
        )

        plt.title("Playlist Subgenre Distribution")

        plt.xlabel("Playlist Subgenre")

        plt.ylabel("Number of Songs")

        plt.xticks(rotation=90)

        self.save_plot("03_playlist_subgenre.png")

    # =====================================================
    # Track Popularity
    # =====================================================

    def popularity_distribution(self):

        plt.figure(figsize=(9,5))

        sns.histplot(
            self.df["track_popularity"],
            bins=30,
            kde=True,
            color="royalblue"
        )

        plt.title("Track Popularity Distribution")

        plt.xlabel("Popularity")

        plt.ylabel("Frequency")

        self.save_plot("04_track_popularity.png")

    # =====================================================
    # Danceability
    # =====================================================

    def danceability_distribution(self):

        plt.figure(figsize=(9,5))

        sns.histplot(
            self.df["danceability"],
            bins=30,
            kde=True,
            color="green"
        )

        plt.title("Danceability Distribution")

        plt.xlabel("Danceability")

        plt.ylabel("Frequency")

        self.save_plot("05_danceability.png")

    # =====================================================
    # Energy
    # =====================================================

    def energy_distribution(self):

        plt.figure(figsize=(9,5))

        sns.histplot(
            self.df["energy"],
            bins=30,
            kde=True,
            color="orange"
        )

        plt.title("Energy Distribution")

        plt.xlabel("Energy")

        plt.ylabel("Frequency")

        self.save_plot("06_energy.png")

    # =====================================================
    # Tempo
    # =====================================================

    def tempo_distribution(self):

        plt.figure(figsize=(9,5))

        sns.histplot(
            self.df["tempo"],
            bins=30,
            kde=True,
            color="purple"
        )

        plt.title("Tempo Distribution")

        plt.xlabel("Tempo")

        plt.ylabel("Frequency")

        self.save_plot("07_tempo.png")

    # =====================================================
    # Loudness
    # =====================================================

    def loudness_distribution(self):

        plt.figure(figsize=(9,5))

        sns.histplot(
            self.df["loudness"],
            bins=30,
            kde=True,
            color="red"
        )

        plt.title("Loudness Distribution")

        plt.xlabel("Loudness")

        plt.ylabel("Frequency")

        self.save_plot("08_loudness.png")

    # =====================================================
    # Numerical Boxplots
    # =====================================================

    def boxplot_visualization(self):

        features = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo"
        ]

        plt.figure(figsize=(14,8))

        sns.boxplot(
            data=self.df[features]
        )

        plt.xticks(rotation=40)

        plt.title("Box Plot of Numerical Features")

        self.save_plot("09_boxplot.png")

    # =====================================================
    # Pair Plot
    # =====================================================

    def pair_plot(self):

        features = [
            "danceability",
            "energy",
            "valence",
            "tempo",
            "track_popularity"
        ]

        pair = sns.pairplot(
            self.df[features],
            corner=True,
            diag_kind="hist"
        )

        pair.fig.suptitle(
            "Pair Plot of Important Features",
            y=1.02
        )

        pair.savefig(
            "outputs/images/10_pairplot.png",
            dpi=300
        )

        plt.show()

        print("10_pairplot.png Saved Successfully")

    # =====================================================
    # Run All Part 1 Visualizations
    # =====================================================

    def run_part1(self):

        self.dataset_overview()

        self.playlist_genre_distribution()

        self.playlist_subgenre_distribution()

        self.popularity_distribution()

        self.danceability_distribution()

        self.energy_distribution()

        self.tempo_distribution()

        self.loudness_distribution()

        self.boxplot_visualization()

        self.pair_plot()
# =====================================================
# Correlation Heatmap
# =====================================================

def correlation_heatmap(self):

    numerical_df = self.df.select_dtypes(include="number")

    plt.figure(figsize=(14,10))

    sns.heatmap(
        numerical_df.corr(),
        cmap="coolwarm",
        annot=False,
        linewidths=0.5
    )

    plt.title("Correlation Heatmap")

    self.save_plot("11_correlation_heatmap.png")


# =====================================================
# Elbow Method
# =====================================================

def elbow_method(self, X_scaled):

    from sklearn.cluster import KMeans

    inertia = []

    K = range(1, 11)

    for k in K:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        model.fit(X_scaled)

        inertia.append(model.inertia_)

    plt.figure(figsize=(8,5))

    plt.plot(
        K,
        inertia,
        marker="o",
        linewidth=2
    )

    plt.xlabel("Number of Clusters (K)")

    plt.ylabel("Inertia")

    plt.title("Elbow Method")

    self.save_plot("12_elbow_method.png")


# =====================================================
# Cluster Distribution
# =====================================================

def cluster_distribution(self):

    if "Cluster" not in self.df.columns:

        print("Cluster column not found.")

        return

    plt.figure(figsize=(8,5))

    sns.countplot(
        data=self.df,
        x="Cluster",
        palette="viridis"
    )

    plt.title("Cluster Distribution")

    plt.xlabel("Cluster")

    plt.ylabel("Number of Songs")

    self.save_plot("13_cluster_distribution.png")


# =====================================================
# PCA Visualization
# =====================================================

def pca_visualization(self, X_scaled):

    from sklearn.decomposition import PCA

    if "Cluster" not in self.df.columns:

        print("Cluster column not found.")

        return

    pca = PCA(n_components=2)

    components = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10,8))

    plt.scatter(
        components[:,0],
        components[:,1],
        c=self.df["Cluster"],
        cmap="viridis",
        s=20
    )

    plt.xlabel("Principal Component 1")

    plt.ylabel("Principal Component 2")

    plt.title("PCA Cluster Visualization")

    plt.colorbar(label="Cluster")

    self.save_plot("14_pca_clusters.png")


# =====================================================
# Cluster by Playlist Genre
# =====================================================

def cluster_by_genre(self):

    if "Cluster" not in self.df.columns:

        return

    plt.figure(figsize=(12,6))

    sns.countplot(
        data=self.df,
        x="playlist_genre",
        hue="Cluster"
    )

    plt.xticks(rotation=30)

    plt.title("Clusters by Playlist Genre")

    self.save_plot("15_cluster_by_genre.png")


# =====================================================
# Cluster by Playlist Name
# =====================================================

def cluster_by_playlist(self):

    if "Cluster" not in self.df.columns:

        return

    top_playlist = self.df["playlist_name"].value_counts().head(15).index

    playlist_df = self.df[
        self.df["playlist_name"].isin(top_playlist)
    ]

    plt.figure(figsize=(14,8))

    sns.countplot(
        data=playlist_df,
        y="playlist_name",
        hue="Cluster"
    )

    plt.title("Clusters by Playlist")

    self.save_plot("16_cluster_by_playlist.png")


# =====================================================
# Deep Learning Cluster Distribution
# =====================================================

def deep_cluster_distribution(self):

    if "DL_Cluster" not in self.df.columns:

        return

    plt.figure(figsize=(8,5))

    sns.countplot(
        data=self.df,
        x="DL_Cluster",
        palette="magma"
    )

    plt.title("Deep Learning Cluster Distribution")

    self.save_plot("19_dl_cluster_distribution.png")


# =====================================================
# Deep Learning Cluster by Genre
# =====================================================

def deep_cluster_by_genre(self):

    if "DL_Cluster" not in self.df.columns:

        return

    plt.figure(figsize=(12,6))

    sns.countplot(
        data=self.df,
        x="playlist_genre",
        hue="DL_Cluster"
    )

    plt.xticks(rotation=30)

    plt.title("Deep Learning Cluster by Genre")

    self.save_plot("20_dl_cluster_by_genre.png")


# =====================================================
# Deep Learning Cluster by Playlist
# =====================================================

def deep_cluster_by_playlist(self):

    if "DL_Cluster" not in self.df.columns:

        return

    top_playlist = self.df["playlist_name"].value_counts().head(15).index

    playlist_df = self.df[
        self.df["playlist_name"].isin(top_playlist)
    ]

    plt.figure(figsize=(14,8))

    sns.countplot(
        data=playlist_df,
        y="playlist_name",
        hue="DL_Cluster"
    )

    plt.title("Deep Learning Cluster by Playlist")

    self.save_plot("21_dl_cluster_by_playlist.png")


# =====================================================
# Run All Part 2
# =====================================================

def run_part2(self, X_scaled):

    self.correlation_heatmap()

    self.elbow_method(X_scaled)

    self.cluster_distribution()

    self.pca_visualization(X_scaled)

    self.cluster_by_genre()

    self.cluster_by_playlist()

    self.deep_cluster_distribution()

    self.deep_cluster_by_genre()

    self.deep_cluster_by_playlist()

# =========================================================
# Standalone Execution
# =========================================================

if __name__ == "__main__":

    df = pd.read_csv("data/raw/spotify_songs.csv")

    visualizer = DataVisualization(df)

    visualizer.run_part1()
    visualization.run_part2()
  
