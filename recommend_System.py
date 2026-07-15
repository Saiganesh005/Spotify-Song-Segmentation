
# ============================================================
# Song Recommendation System
# ============================================================

def recommend_song(song_name):

    if song_name not in df["track_name"].values:
        return "Song not found."

    cluster = df.loc[
        df["track_name"] == song_name,
        "Cluster"
    ].iloc[0]

    recommendations = df[
        df["Cluster"] == cluster
    ][[
        "track_name",
        "track_artist",
        "playlist_genre"
    ]].head(10)

    return recommendations
  recommend_song("Closer")
