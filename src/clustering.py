"""
=========================================================
Spotify Genre Segmentation
Clustering Module
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
=========================================================
"""

from src.import_libraries import *
from src.config import *


class Clustering:

    def __init__(self, features):

        self.features = features

        self.model = None

        self.labels = None

    # =====================================================
    # Elbow Method
    # =====================================================

    def elbow_method(self, max_clusters=10):

        inertia = []

        cluster_range = range(1, max_clusters + 1)

        for k in cluster_range:

            model = KMeans(
                n_clusters=k,
                random_state=RANDOM_STATE,
                n_init=N_INIT,
                max_iter=MAX_ITER
            )

            model.fit(self.features)

            inertia.append(model.inertia_)

        plt.figure(figsize=(8,5))

        plt.plot(cluster_range, inertia, marker="o")

        plt.xlabel("Number of Clusters")

        plt.ylabel("Inertia")

        plt.title("Elbow Method")

        plt.grid(True)

        plt.savefig(
            os.path.join(
                IMAGE_DIR,
                "elbow_method.png"
            )
        )

        plt.show()

        return inertia

    # =====================================================
    # Train KMeans
    # =====================================================

    def train(self):

        self.model = KMeans(

            n_clusters=N_CLUSTERS,

            random_state=RANDOM_STATE,

            n_init=N_INIT,

            max_iter=MAX_ITER

        )

        self.labels = self.model.fit_predict(
            self.features
        )

        print("=" * 60)

        print("KMEANS TRAINING COMPLETED")

        print("=" * 60)

        return self.labels

    # =====================================================
    # Predict
    # =====================================================

    def predict(self, data):

        return self.model.predict(data)

    # =====================================================
    # Cluster Centers
    # =====================================================

    def cluster_centers(self):

        centers = pd.DataFrame(

            self.model.cluster_centers_,

            columns=FEATURE_COLUMNS

        )

        print(centers)

        return centers

    # =====================================================
    # Cluster Sizes
    # =====================================================

    def cluster_sizes(self):

        sizes = pd.Series(

            self.labels

        ).value_counts().sort_index()

        print(sizes)

        return sizes

    # =====================================================
    # Add Cluster Column
    # =====================================================

    def add_clusters(self, dataframe):

        dataframe = dataframe.copy()

        dataframe["Cluster"] = self.labels

        return dataframe

    # =====================================================
    # Save Model
    # =====================================================

    def save_model(self):

        joblib.dump(

            self.model,

            KMEANS_MODEL_FILE

        )

        print(f"Saved : {KMEANS_MODEL_FILE}")

    # =====================================================
    # Load Model
    # =====================================================

    def load_model(self):

        self.model = joblib.load(

            KMEANS_MODEL_FILE

        )

        print("Model Loaded Successfully.")

    # =====================================================
    # Save Clustered Dataset
    # =====================================================

    def save_clustered_dataset(self, dataframe):

        dataframe = self.add_clusters(dataframe)

        dataframe.to_csv(

            CLUSTER_DATASET,

            index=False

        )

        print(f"Saved : {CLUSTER_DATASET}")

    # =====================================================
    # Summary
    # =====================================================

    def summary(self):

        print()

        print("=" * 60)

        print("CLUSTERING SUMMARY")

        print("=" * 60)

        print(f"Number of Clusters : {N_CLUSTERS}")

        print(f"Samples            : {len(self.labels)}")

        print()

        self.cluster_sizes()

    # =====================================================
    # Complete Pipeline
    # =====================================================

    def clustering_pipeline(self, dataframe):

        self.elbow_method()

        self.train()

        self.cluster_centers()

        self.cluster_sizes()

        self.save_model()

        self.save_clustered_dataset(dataframe)

        self.summary()

        return self.labels


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    from src.load_dataset import DatasetLoader
    from src.feature_selection import FeatureSelection
    from src.feature_scaling import FeatureScaling

    loader = DatasetLoader(DATASET_PATH)

    df = loader.loading_pipeline()

    selector = FeatureSelection(df)

    features = selector.feature_selection_pipeline()

    scaler = FeatureScaling(features)

    scaled_features = scaler.scaling_pipeline()

    clustering = Clustering(scaled_features)

    clustering.clustering_pipeline(df)
