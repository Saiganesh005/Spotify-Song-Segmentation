# ============================================================
#Correlation Heatmap
# ============================================================

numeric_df=df.select_dtypes(include=['int64','float64'])

print(numeric_df.columns)

corr_mtx=numeric_df.corr()
print(corr_mtx)

plt.figure(figsize=(14,10))
sns.heatmap(corr_mtx,annot=True,cmap='coolwarm',fmt='.2f',linewidth=0.5)
plt.title("Correlation Matrix of Splotify Audio Features")
plt.show()

corr_mtx["track_popularity"].sort_values(ascending=False)
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
