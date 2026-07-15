# ============================================================
#Feature Selection
# ============================================================

features = [
    'danceability',
    'energy',
    'key',
    'loudness',
    'mode',
    'speechiness',
    'acousticness',
    'instrumentalness',
    'liveness',
    'valence',
    'tempo',
    'duration_ms',
    'time_signature'
]

X = df[features]

print("Selected Features")

X.head()
# Number of rows and selected features
print("Feature Matrix Shape :", X.shape)
# Missing values in selected features
print(X.isnull().sum())
