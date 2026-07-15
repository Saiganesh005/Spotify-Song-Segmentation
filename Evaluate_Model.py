from sklearn.metrics import silhouette_score

sil_score = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("Silhouette Score :", sil_score)

from sklearn.metrics import silhouette_score

sil_score = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("Silhouette Score :", sil_score)

from sklearn.metrics import davies_bouldin_score

davies_score = davies_bouldin_score(
    X_scaled,
    df["Cluster"]
)

print("Davies-Bouldin Score :", davies_score)

import joblib

joblib.dump(kmeans, "spotify_kmeans_model.pkl")

joblib.dump(scaler, "spotify_scaler.pkl")

print("Model and Scaler saved successfully.")
df.to_csv(
    "spotify_clustered_songs.csv",
    index=False
)

print("Clustered dataset saved successfully.")
