from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = datasets.load_iris()
X = df.data
y = df.target

kmeans_model = KMeans(n_clusters = 3, random_state=1).fit(X)
y_kmeans = kmeans_model.fit_predict(X)

import matplotlib.pyplot as plt

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')


plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], s = 100, c = 'red', label = 'Centroid')

plt.legend()

from sklearn.metrics import silhouette_score
silhouette_score(X, y_kmeans, metric = 'euclidean')

from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(y, y_kmeans)

from sklearn.metrics.cluster import normalized_mutual_info_score
normalized_mutual_info_score(y, y_kmeans)


'''END OF THE FIRST CODE===================='''

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = datasets.load_iris()
y = df.target
yframe=pd.DataFrame(y)
dfr = pd.DataFrame(data=df.data,
                  columns=df.feature_names)
dfr["target"] = yframe
dfr

kmeans_model =KMeans(n_clusters=3)
kmeans_model.fit(dfr[['sepal length (cm)','target']])
dfr['kmeans_3']=kmeans_model.labels_
dfr

plt.scatter(x=dfr['sepal length (cm)'],y=dfr['target'],c = dfr['kmeans_3'])

#Intrinsic Method
from sklearn.metrics import silhouette_score
silhouette_score(X, y_kmeans, metric = 'euclidean')

#Adjusted Rand Index
from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(y, y_kmeans)

#Mutual Information
from sklearn.metrics.cluster import normalized_mutual_info_score
normalized_mutual_info_score (y, y_kmeans)