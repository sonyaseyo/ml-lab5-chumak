import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


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

    
def print_cluster_centers(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    print(f'Координати центрів для {n_clusters} кластерів:\n{centers}\n')
    return centers

distorsions = []

for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    distorsions.append(kmeans.inertia_)

fig = plt.figure(figsize=(15, 5))
plt.plot(range(2, 20), distorsions)
plt.grid(True)
plt.title('elbow curve')
plt.savefig('elbow_curve.png')



