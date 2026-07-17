"""
=========================================================
Spotify Genre Segmentation
Deployment Module
=========================================================

Author : Robbi Sai Ganesh Devi Prasad
=========================================================
"""

from src.import_libraries import *
from src.config import *


class SpotifyDeployment:

    def __init__(self):

        self.scaler = None
        self.model = None

        self.load_models()

    # =====================================================
    # Load Models
    # =====================================================

    def load_models(self):

        try:

            self.scaler = joblib.load(SCALER_FILE)

            self.model = joblib.load(KMEANS_MODEL_FILE)

            print("=" * 60)
            print("MODELS LOADED SUCCESSFULLY")
            print("=" * 60)

        except Exception as e:

            print(f"Error Loading Models : {e}")

    # =====================================================
    # Preprocess Input
    # =====================================================

    def preprocess(self, input_data):

        if isinstance(input_data, list):

            input_data = np.array(input_data).reshape(1, -1)

        elif isinstance(input_data, pd.DataFrame):

            input_data = input_data[FEATURE_COLUMNS]

        scaled = self.scaler.transform(input_data)

        return scaled

    # =====================================================
    # Predict Cluster
    # =====================================================

    def predict_cluster(self, input_data):

        scaled = self.preprocess(input_data)

        prediction = self.model.predict(scaled)

        return int(prediction[0])

    # =====================================================
    # Predict Multiple Songs
    # =====================================================

    def batch_prediction(self, dataframe):

        scaled = self.preprocess(dataframe)

        dataframe = dataframe.copy()

        dataframe["Cluster"] = self.model.predict(scaled)

        return dataframe

    # =====================================================
    # Cluster Description
    # =====================================================

    def cluster_description(self, cluster):

        descriptions = {

            0: "High Energy Dance Songs",

            1: "Relaxing Acoustic Songs",

            2: "Pop & Commercial Songs",

            3: "Instrumental Music",

            4: "Upbeat Workout Songs",

            5: "Slow Emotional Songs"

        }

        return descriptions.get(

            cluster,

            "Unknown Cluster"

        )

    # =====================================================
    # Prediction Result
    # =====================================================

    def prediction_result(self, input_data):

        cluster = self.predict_cluster(input_data)

        return {

            "Cluster": cluster,

            "Description":

                self.cluster_description(cluster)

        }

    # =====================================================
    # Predict From CSV
    # =====================================================

    def predict_csv(self, csv_path):

        df = pd.read_csv(csv_path)

        result = self.batch_prediction(df)

        output = os.path.join(

            PREDICTION_DIR,

            "predictions.csv"

        )

        result.to_csv(

            output,

            index=False

        )

        print(f"Saved : {output}")

        return result

    # =====================================================
    # Model Information
    # =====================================================

    def model_information(self):

        print("=" * 60)

        print("DEPLOYMENT INFORMATION")

        print("=" * 60)

        print("Scaler :", SCALER_FILE)

        print("Model  :", KMEANS_MODEL_FILE)

        print("Features")

        for feature in FEATURE_COLUMNS:

            print("-", feature)

    # =====================================================
    # Deployment Pipeline
    # =====================================================

    def deployment_pipeline(self):

        self.model_information()

        print("\nDeployment Ready.")

        return self


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    deployment = SpotifyDeployment()

    deployment.deployment_pipeline()

    sample_song = [

        0.82,      # danceability
        0.78,      # energy
        5,         # key
        -5.2,      # loudness
        1,         # mode
        0.05,      # speechiness
        0.10,      # acousticness
        0.00,      # instrumentalness
        0.12,      # liveness
        0.74,      # valence
        128.5,     # tempo
        210000,    # duration_ms
        4          # time_signature

    ]

    result = deployment.prediction_result(sample_song)

    print(result)
