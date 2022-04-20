import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



data = pd.read_csv('cluster/total.csv')

X = data.drop(columns = ['sum', 'label','film_id', 'rate'], inplace = False)
X = MinMaxScaler().fit_transform(X)
cov_matrix = pd.DataFrame(np.cov(X.T))


pd.DataFrame(X)

# K-means Elbow method

error = []
cls = [2, 3, 4, 5, 6, 7, 8]
for k in cls:
    model = KMeans(n_clusters = 5)
    model.fit(X)
    error.append(model.inertia_)


plt.plot(cls, error, 'b--o', linewidth = 2.5)
plt.title('Elbow method')
plt.xlabel('# Clusters')
plt.ylabel('Error')



model = KMeans(n_clusters = 3)
model.fit(X)


print(np.bincount(model.labels_))
print(model.cluster_centers_)



labels = model.labels_
y0 = np.where(labels == 0)
y1 = np.where(labels == 1)
y2 = np.where(labels == 2)


cluster1 = data.iloc[y0]
cluster2 = data.iloc[y1]
cluster3 = data.iloc[y2]
cluster1.to_csv('cluster1111.csv')
cluster2.to_csv('cluster2222.csv')
cluster3.to_csv('cluster3333.csv')



rate1 = cluster1['rate']
rate1.value_counts()


rate2 = cluster2['rate']
rate2.value_counts()


rate3 = cluster3['rate']
rate3.value_counts()




# Save the cluster object

import pickle
pickle.dump(model, open('user_cluster.p', "wb" ))

data['label'].unique()