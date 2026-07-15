# ============================================================
#  PCA Visualization
# ============================================================

from sklearn.decomposition import PCA

# Reduce dimensions from multiple features to 2 components
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,8))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Cluster"],
    cmap="viridis",
    s=20
)

plt.title("Spotify Song Clusters using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar(label="Cluster")

plt.show()

# ============================================================
#Cluster Distribution by Playlist Genre
# ============================================================

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x="playlist_genre",
    hue="Cluster"
)

plt.xticks(rotation=45)

plt.title("Clusters by Playlist Genre")

plt.show()
# ============================================================
# Cluster Distribution by Playlist Name
# ============================================================

top_playlist = df["playlist_name"].value_counts().head(15).index

playlist_df = df[df["playlist_name"].isin(top_playlist)]

plt.figure(figsize=(14,8))

sns.countplot(
    data=playlist_df,
    y="playlist_name",
    hue="Cluster"
)

plt.title("Top Playlist Names by Cluster")

plt.show()
