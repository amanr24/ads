import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/content/sample_data/Mobile-Company-Telco-Customer-Churn.csv', delimiter = ',')
pd.pandas.set_option('display.max_columns', None)

df.head(5)

df = df.drop(['customerID'], axis=1)

df.head(5)

df.isnull().sum()

df.dtypes

from sklearn.preprocessing import LabelEncoder
lbe = LabelEncoder()

cat_cols = [col for col in df.columns if df[col].dtype=="O"]
print(cat_cols)
for col in cat_cols:
  if len(df[col].unique()) < 10:
    df[col]= lbe.fit_transform(df[col])
  else:
    df = df.drop(col, axis = 1)

df

df.dtypes

X = df.drop('Churn', axis = 1)
y = df['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_arr = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn)
print(fp)
print(fn)
print(tp)

df_cm = pd.DataFrame(cm_arr, index = [i for i in "01"],
                  columns = [i for i in "01"])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True)

"""## Accuracy"""

accuracy = (tp + tn) / (tp + tn + fp + fn)
accuracy

"""## Error Rate"""

error_rate = (fp + fn) / (tp + tn + fp + fn)
error_rate

"""## Precision"""

precision = (tp) / (tp + fp)
precision

"""## Sensitivity"""

sensitivity = (tp) / (tp + fn)
sensitivity

"""## Specificity"""

specificity = (tn) / (tp + fp)
specificity

"""## ROC"""

import math
ROC = math.sqrt(sensitivity ** 2 + specificity ** 2) / math.sqrt(2)
ROC

"""## F1 Score"""

f1_score = (2 * precision * sensitivity) / (precision + sensitivity)
f1_score

"""## Geometric Mean"""

geometric_mean = math.sqrt(sensitivity * specificity)
geometric_mean

"""## False Positive Rate"""

fpr = 1 - specificity
fpr

"""## False Negative Rate"""

fnr = 1 - sensitivity
fnr

"""## ROC Curve"""

import matplotlib.pyplot as plt
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=0)
print(fpr)
auc = metrics.roc_auc_score(y_test, y_pred)

plt.plot(fpr,tpr, label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

"""# Logistic Regression"""

df = pd.read_csv('/content/sample_data/Salary_Data.csv')

# df.head(5)
print(df.shape)
X = df['YearsExperience']
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
X_train= X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred

y_mean = sum(y) / len(y)
x_mean = sum(X) / len(X)

"""## Karl Pearson's Coefficient"""

x2 = 0
for val in X:
  x2 += (val - x_mean) * (val - x_mean)

y2 = 0
for val in y:
  y2 += (val - y_mean) * (val - y_mean)

num = 0
for i, j in zip(X, y):
  num += (i - x_mean) * (j - y_mean)

r = num / (math.sqrt(x2 * y2))
print(r)

"""## Coefficient of Determination"""

from sklearn.metrics import r2_score, mean_squared_error
R_sq = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(mse)

"""##Mean Squared Error"""

mse = 0

for i , j in zip(y_test, y_pred):
  mse += (i - j) * (i - j)

print(mse / len(y_test))

"""## Root mean squared error"""

root_mean_sq_error = math.sqrt(mse)
print(root_mean_sq_error)

"""## Root mean squared relative error

"""

rmser = 0

for i,j in zip(y_test, y_pred):
  rmser += ((i - j) / i) ** 2

rmser = math.sqrt(rmser / len(y_test))
print(rmser)

"""## Mean Absolute error"""

mae = 0
for i,j in zip(y_test, y_pred):
  mae += abs(i - j)

mae = mae / len(y_test)
print(mae)

"""## Mean Absolute Percentage Error"""

mape = 0
for i,j in zip(y_test, y_pred):
  mape = abs((i - j) / i)

mape = (mape / len(y_test)) * 100
print(mape)

"""classification===================="""

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf_tree = DecisionTreeClassifier();
clf_tree.fit(X_train, y_train);

y_pred = clf_tree.predict(X_test)
print(y_pred)

"""Evaluation Metrics"""

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("True Negatives {}".format(tn))
print("False Negatives {}".format(fn))
print("True Positives {}".format(tp))
print("False Positives {}".format(fp))

acc = (tn + tp) / (tn + tp + fn + fp)
print("Accuracy {}".format(acc))

error_rate = (fn + fp) / (tn + tp + fn + fp)
print("Error Rate {}".format(error_rate))

precision = tp / (tp + fp)
print("Precision {}".format(precision))

sns = tp / (tp + fn)
spc = tn / (tn + fp)
print("Sensitivity {}".format(sns))
print("Specificity {}".format(spc))

import math

roc = math.sqrt((sns * sns) + (spc * spc)) / math.sqrt(2)
print("ROC {}".format(roc))

GM = math.sqrt(sns * spc)
print("Geometric Mean {}".format(GM))

f1 = (2 * sns * precision) / (precision + sns)
print("f1 score {}".format(f1))

fpr = 1 - spc
fnr = 1 - sns
power = 1 - fnr
print("False positive Rate {}".format(fpr))
print("false negative Rate {}".format(fnr))
print("Power {}".format(power))

"""Plot ROC Curve"""

from sklearn.metrics import roc_curve, roc_auc_score

false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_pred)
print('roc_auc_score for DecisionTree: ', roc_auc_score(y_test, y_pred))

import matplotlib.pyplot as plt

plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls='--')
plt.plot([0, 0], [0, 1], c='.7')
plt.plot([1, 1], c='.7')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

"""Regression"""

from google.colab import files
import io
import pandas as pd
import numpy as np

uploaded = files.upload()
df = pd.read_excel(io.BytesIO(uploaded['regdata.xlsx']))

df.head()

import seaborn as sns

df2 = df[['Price', 'Dem']]
# rho=df2['Price'].corr(df2['Demand'])
df2['naturalLogPrice'] = np.log(df2['Price'])
df2['naturalLogDemand'] = np.log(df2['Dem'])

sns.regplot(x="naturalLogPrice", y="naturalLogDemand", data=df2, fit_reg=True)

X = df2[['naturalLogPrice']]
y = df2['naturalLogDemand']

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
print(y_pred)

"""Evaluation Metrics"""

from scipy.stats import pearsonr

list1 = df2['naturalLogPrice']
list2 = df2['naturalLogDemand']

corr, _ = pearsonr(list1, list2)
print('Pearsons correlation: %.3f' % corr)

a = np.sum((y - y_pred) ** 2)
n = np.size(y)

mse = a / n
print("Mean Squared Error", mse)

rmse = math.sqrt(mse)
print("Root Mean Squared Error ", rmse)

q = np.sum((y - y_pred) ** 2)
my = np.sum(y) / n
mx = np.sum(X) / n
p = np.sum((y - my) ** 2)

R2 = 1 - (q / p)
print("Coefficient of Determination ", R2)

b = np.sum(((y - y_pred) / y) ** 2)
rmsre = math.sqrt(b / n)
print("Root Mean Squared Relative Error ", rmsre)

a = np.sum(abs(y - y_pred))
n = np.size(y)

mae = a / n
print("Mean Absolute Error ", mae)

b = np.sum(abs((y - y_pred) / y))
mape = (100 * b) / n
print("Mean absolute Percentage Error", mape)