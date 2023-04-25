# https://openmv.net/info/travel-times

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/sample_data/travel-times.csv')
df.head(10)
df = df[df['FuelEconomy'] != '-']

df.head(10)
df.drop(['Date', 'StartTime'], axis =1)
df.isnull().mean()
# gives percentage of null values in particular column
#deletion of rows with missing data
remove_na_rows = df.dropna(axis = 0)
print(remove_na_rows.shape)
print(df.shape)

from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
X = np.array(df['Distance']).reshape(-1, 1)
y = np.array(df['TotalTime']).reshape(-1, 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print(regr.score(X_test, y_test))
y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')

plt.show()
df['AvgSpeed'].plot.density(color='green')
plt.title('Density plot for Speeding')
plt.show()

"------------------"

import pandas as pd
import numpy as np
df =pd.read_csv("/content/loan_data_set.csv")
df

na_variables = [ var for var in df.columns if df[var].isnull().mean() > 0 ]
#for finding null values in cols
na_variables

# mean imputation
df1 = df
df1
missing_col = ["LoanAmount"]

for i in missing_col:
  df1.loc[df1.loc[:,i].isnull(),i]=df1.loc[:,i].mean()

df1

# median imputation
df2=df
for i in missing_col:
  df2.loc[df2.loc[:,i].isnull(),i]=df2.loc[:,i].median()

df2

# Mode imputation

df4 = df
df4
missing_col = ["LoanAmount"]

for i in missing_col:
  df4.loc[df4.loc[:,i].isnull(),i]=df4.loc[:,i].mode()

df4

#categorical to numerical

from sklearn.preprocessing import OrdinalEncoder

data=df
oe =OrdinalEncoder()
result = oe.fit_transform(data)
print(result)

#random sample
df5=df
df5['LoanAmount'].dropna().sample(df5['LoanAmount'].isnull().sum(),random_state=0)
df5

# frequent category imputation
df6=df
m= df6["Gender"].mode()
m=m.tolist()

frq_imp = df6["Gender"].fillna(m[0])
frq_imp.unique()

#regression imputation
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
df1=df[["CoapplicantIncome","LoanAmount"]]


# col=df1["LoanAmount"].dropna()
# df1.head()
testdf = df1[df1['LoanAmount'].isnull()==True]
testdf
traindf = df1[df1['LoanAmount'].isnull()==False]
traindf


lr.fit(traindf['LoanAmount'],traindf['CoapplicantIncome'])
# testdf.drop("LoanAmount",axis=1,inplace=True)
# testdf
pred = lr.predict(testdf)
# testdf['LoanAmount']= pred