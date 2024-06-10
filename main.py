import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist


# 1
data = pd.read_csv('dataset3_l5.csv', delimiter=';')

# 2
print(f'2. к-кість записів: {data.shape[0]}')
print(f'cols: {data.columns.tolist()}')

# 3, 4
col_to_del = [col for col in data.columns if 'Concrete compressive strength' in col]

if col_to_del:
    data = data.drop(columns=col_to_del)
else:
    print("no data found")

data.to_csv('dataset3_l5_upd.csv', index=False)

# print(data.head())
print(f'upd cols: {data.columns.tolist()}')

# 5
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].str.replace(',', '.').astype(float)


X = data[['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)', 'Fly Ash (component 3)(kg in a m^3 mixture)', 'Water  (component 4)(kg in a m^3 mixture)', 'Superplasticizer (component 5)(kg in a m^3 mixture)', 'Coarse Aggregate  (component 6)(kg in a m^3 mixture)', 'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)']].values


# Elbow Method
def elbow_method(X):
    distortions = []
    K = range(1, 20)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


elbow_method(X)


# Silhouette Method
def silhouette_method(X):
    silhouette_scores = []
    K = range(2, 20)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('The Silhouette Method showing the optimal k')
    plt.show()


silhouette_method(X)


# Prediction Strength Method (Simplified)
def prediction_strength(X, max_clusters=10):
    def calc_prediction_strength(X, k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        cluster_labels = np.unique(labels)

        pred_strengths = []
        for label in cluster_labels:
            cluster_points = X[labels == label]
            if len(cluster_points) > 1:
                new_kmeans = KMeans(n_clusters=2)
                new_kmeans.fit(cluster_points)
                pred_labels = new_kmeans.labels_
                if len(np.unique(pred_labels)) == 2:
                    dist_within_0 = pairwise_distances(cluster_points[pred_labels == 0])
                    dist_within_1 = pairwise_distances(cluster_points[pred_labels == 1])
                    dist_between = pairwise_distances(cluster_points[pred_labels == 0],
                                                      cluster_points[pred_labels == 1])
                    pred_strength = (np.mean(dist_between > dist_within_0.max()) + np.mean(
                        dist_between > dist_within_1.max())) / 2
                    pred_strengths.append(pred_strength)

        return np.mean(pred_strengths) if pred_strengths else 0

    strengths = []
    K = range(2, max_clusters)
    for k in K:
        strengths.append(calc_prediction_strength(X, k))

    plt.figure(figsize=(8, 4))
    plt.plot(K, strengths, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Prediction Strength')
    plt.title('Prediction Strength Method showing the optimal k')
    plt.show()


prediction_strength(X)

# Визначаємо оптимальну кількість кластерів за результатами методів
optimal_k = 4  # Це значення змінюється залежно від результатів

# Кластеризація з використанням KMeans
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10)
kmeans.fit(X)
print("Координати центрів кластерів KMeans:\n", kmeans.cluster_centers_)

# Кластеризація з використанням AgglomerativeClustering
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_clustering.fit_predict(X)

# Обчислення центрів кластерів для AgglomerativeClustering
agg_centers = np.array([X[agg_labels == i].mean(axis=0) for i in range(optimal_k)])
print("Координати центрів кластерів AgglomerativeClustering:\n", agg_centers)