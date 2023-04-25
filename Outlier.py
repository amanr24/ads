# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv('Iris.csv')
data.head()

new_data=data.drop(['Id'],axis=1)
new_data.boxplot()

# create arrays
X = new_data.drop('Species',axis=1).values

# instantiate model
nbrs = NearestNeighbors(n_neighbors = 3)

# fit model
nbrs.fit(X)

# distances and indexes of k-neaighbors from model outputs
distances, indexes = nbrs.kneighbors(X)
# plot mean of k-distances of each observation

plt.plot(distances.mean(axis =1))

# visually determine cutoff values > 0.15
outlier_index = np.where(distances.mean(axis = 1) > 0.3)
outlier_index

# filter outlier values
outlier_values = new_data.iloc[outlier_index]
outlier_values

# data wrangling
import pandas as pd
# visualization
import matplotlib.pyplot as plt
# algorithm
from sklearn.cluster import DBSCAN

# input data
df = data[["SepalLengthCm", "SepalWidthCm"]]
# specify & fit model
model = DBSCAN(eps = 0.4, min_samples = 10).fit(df)

# visualize outputs
colors = model.labels_
plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"], c = colors)

# outliers dataframe
outliers = data[model.labels_ == -1]
print(outliers)

"""END OF FIRST CODE================"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pycaret.datasets import get_data
df = get_data('anomaly')
df.head()

df.describe()

df.columns

sns.boxplot(x="variable", y="value", data=pd.melt(df))
plt.show()

sns.pairplot(df)

from pycaret.anomaly import *
setup = setup(df, session_id = 123)

models()

lof = create_model('lof')
plot_model(lof)

knn = create_model('knn')
plot_model(knn)

knn_predictions = predict_model(knn, data = df)

knn_predictions

from sklearn import datasets
df = datasets.load_iris()
X = df.data
y = df.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)

fitted = knn.fit(X_train, y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(pred, y_test))

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv('/content/sample_data/boston.csv')
df.head()

df.shape

df['TAX'].describe()

sns.boxplot(df['TAX'])

sns.distplot(df['TAX'])

"""## Winsorization"""

upper_limit = df["TAX"].quantile(0.99)
lower_limit = df['TAX'].quantile(0.01)

df['TAX'] = np.where(df['TAX'] > upper_limit, upper_limit,
                         np.where(df['TAX'] < lower_limit, lower_limit, df['TAX']))

df['TAX'].describe()

sns.distplot(df['TAX'])

sns.boxplot(df['TAX'])