#Playlist Genre Distribution
plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    x="playlist_genre",
    order=df['playlist_genre'].value_counts().index
)

plt.xticks(rotation=45)
plt.xlabel("Playlist Genre")
plt.ylabel("Number of Songs")
plt.title("Distribution of Playlist Genres")
plt.show()

genre_counts = df['playlist_genre'].value_counts()

print("Most Frequent Genre :", genre_counts.idxmax(), "->", genre_counts.max())
print("Least Frequent Genre :", genre_counts.idxmin(), "->", genre_counts.min())
plt.tight_layout()

plt.savefig(
    "outputs/images/01_playlist_genre_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
#Playlist Subgenre Distribution

plt.figure(figsize=(14,7))

sns.countplot(
    data=df,
    x="playlist_subgenre",
    order=df['playlist_subgenre'].value_counts().index
)

plt.xticks(rotation=45)
plt.xlabel("Playlist Subgenre")
plt.ylabel("Number of Songs")
plt.title("Distribution of Playlist Subgenres")


subgenre_counts = df['playlist_subgenre'].value_counts()

print("Most Frequent Subgenre :", subgenre_counts.idxmax(), "->", subgenre_counts.max())
print("Least Frequent Subgenre :", subgenre_counts.idxmin(), "->", subgenre_counts.min())
#Track Popularity Distribution

plt.tight_layout()

plt.savefig(
    "outputs/images/02_playlist_subgenre_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
#Track Popularity
plt.figure(figsize=(8,5))

sns.histplot(
    df["track_popularity"],
    bins=30,
    kde=True
)

plt.xlabel("Track Popularity")
plt.ylabel("Frequency")
plt.title("Distribution of Track Popularity")


plt.savefig(
    "outputs/images/03_track_popularity.png",
    dpi=300
)

plt.show()
#Danceability Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df["danceability"],
    bins=30,
    kde=True
)

plt.title("Danceability Distribution")

plt.tight_layout()

plt.savefig(
    "outputs/images/04_danceability.png",
    dpi=300
)

plt.show()
#Energy Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df["energy"],
    bins=30,
    kde=True
)

plt.title("Energy Distribution")

plt.tight_layout()

plt.savefig(
    "outputs/images/05_energy.png",
    dpi=300
)

plt.show()
#Tempo Distribution

plt.figure(figsize=(8,5))

sns.histplot(
    df["tempo"],
    bins=30,
    kde=True
)

plt.title("Tempo Distribution")
plt.tight_layout()

plt.savefig(
    "outputs/images/06_tempo.png",
    dpi=300
)

plt.show()
#Loudness Distribution
plt.figure(figsize=(8,5))

sns.histplot(
    df["loudness"],
    bins=30,
    kde=True
)

plt.title("Loudness Distribution")

plt.tight_layout()

plt.savefig(
    "outputs/images/07_loudness.png",
    dpi=300
)

plt.show()
#Box Plot of Audio Features

features = [
    'danceability',
    'energy',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence'
]

plt.figure(figsize=(14,6))

sns.boxplot(data=df[features])

plt.xticks(rotation=45)

plt.title("Box Plot of Audio Features")


plt.tight_layout()

plt.savefig(
    "outputs/images/08_boxplot.png",
    dpi=300
)

plt.show()
#Pair Plot

pair_features = [
    'danceability',
    'energy',
    'valence',
    'acousticness'
]

sns.pairplot(df[pair_features])
pair.fig.suptitle(
    "Pair Plot of Audio Features",
    y=1.02
)

pair.savefig(
    "outputs/images/09_pairplot.png",
    dpi=300
)
