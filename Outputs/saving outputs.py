import os

# Create output folder
os.makedirs("Outputs/images", exist_ok=True)

plt.figure(figsize=(10,8))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=df["Cluster"],
    cmap="viridis"
)

plt.title("Spotify Song Clusters using PCA")

plt.xlabel("Principal Component 1")

plt.ylabel("Principal Component 2")

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "outputs/images/12_pca_clusters.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Cluster",
    palette="viridis"
)

plt.title("Cluster Distribution")

plt.tight_layout()

plt.savefig(
    "outputs/images/13_cluster_distribution.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x="playlist_genre",
    hue="Cluster"
)

plt.xticks(rotation=45)

plt.title("Clusters by Playlist Genre")

plt.tight_layout()

plt.savefig(
    "outputs/images/14_cluster_by_genre.png",
    dpi=300
)

plt.show()
top = df["playlist_name"].value_counts().head(15).index

temp = df[df["playlist_name"].isin(top)]

plt.figure(figsize=(14,8))

sns.countplot(
    data=temp,
    y="playlist_name",
    hue="Cluster"
)

plt.title("Clusters by Playlist Name")

plt.tight_layout()

plt.savefig(
    "outputs/images/15_cluster_by_playlist.png",
    dpi=300
)

plt.show()

