from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from numpy import where
import pandas as pd

df = pd.read_csv('Churn_Modelling.csv')
df.head()

import seaborn as sns

data = df[['CreditScore', 'Age', 'Exited',]]
print(data.head(10))
sns.scatterplot(data = data, x ='CreditScore', y = 'Age', hue = 'Exited')

from sklearn.preprocessing import LabelEncoder
for col in df.columns:
  if df[col].dtype == 'O':
    label_encode = LabelEncoder()
    df[col] = label_encode.fit_transform(df[col])
df

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#Splitting the data with stratification
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(df.drop('Exited',axis=1), df['Exited'], test_size = 0.2, random_state = 101)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred))

smote = SMOTE(sampling_strategy='auto',k_neighbors=5,random_state = 101)
X_oversample, y_oversample = smote.fit_resample(X_train, y_train)

clf.fit(X_oversample,y_oversample)
y_predo=clf.predict(X_test)
print(classification_report(y_test, y_predo))

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

print(classification_report(y_test, classifier.predict(X_test)))

classifier.fit(X_oversample, y_oversample)
print(classification_report(y_test, classifier.predict(X_test)))

smote = SMOTE(random_state = 101)
X, y = smote.fit_resample(df[['CreditScore', 'Age']], df['Exited'])
#Creating a new Oversampling Data Frame
df_oversampler = pd.DataFrame(X, columns = ['CreditScore', 'Age'])
df_oversampler['Exited']=y
print(df_oversampler.head())

sns.countplot(data=df_oversampler,x='Exited')

from collections import Counter
X=df[['CreditScore', 'Age']]
y=df['Exited']

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)

sns.scatterplot(data = df_oversampler, x ='CreditScore', y = 'Age', hue = 'Exited')

'''END OF FIRST CODE======================='''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/sample_data/Mobile-Company-Telco-Customer-Churn.csv')
df.head(5)

df = df.drop(['customerID', 'TotalCharges'], axis = 1)

df.info()

from sklearn.preprocessing import LabelEncoder

# df['StreamingMovies'].dtype
for col in df.columns:
  if df[col].dtype == 'O':
    label_encode = LabelEncoder()
    df[col] = label_encode.fit_transform(df[col])
df

df['Churn'].value_counts()

sns.countplot(x ='Churn', data = df)

X = df.drop('Churn', axis = 1)
y = df['Churn']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

pred = tree.predict(X_test)

# print(y_test)
y_test.value_counts()

# print(pred)
left_customers = X_test.loc[y_test == 1]
print(len(left_customers))
plt.hist(left_customers['MonthlyCharges'])
plt.show()

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=84)
X_res, y_res = sm.fit_resample(X_train, y_train)

tree.fit(X_res, y_res)

pred_res = tree.predict(X_test)

print(accuracy_score(y_test, pred_res))
print(classification_report(y_test, pred_res))