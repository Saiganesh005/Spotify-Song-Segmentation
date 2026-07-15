# ============================================================
#Standardize Features
# ============================================================
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print("Scaled Feature Shape: ",X_scaled.shape)
# ============================================================
#  Elbow Method
# ============================================================

wcss = []

for i in range(2,11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=20
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))

plt.plot(range(2,11), wcss, marker='o')

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()
# ============================================================
#  Silhouette Score
# ============================================================
for i in range(2,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    print(f"Clusters = {i}  -->  Silhouette Score = {score:.4f}")
