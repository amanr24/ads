import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/content/drive/My Drive/ADS/supermarket_sales - Sheet1.csv')
df.head()
#Mean SD LQ UQ max min
df.describe()
#Count of null values
df.info()
#Median
df.median()

#Mode
print(df['Product line'].mode())
print(df['City'].mode())
print(df['Payment'].mode())
print(df['Customer type'].mode())
print(df['Gender'].mode())
#Scatter plot
plt.scatter(df['Tax 5%'], df['Unit price'], c ="blue")
plt.scatter(df['gross income'], df['Unit price'], c ="blue")
plt.scatter(df['Quantity'], df['Total'], c ="blue")

#Box plot
x2=df['Tax 5%']
x4=df['gross income']
x5=df['Rating']
data = pd.DataFrame({ "Tax 5%": x2,"gross income": x4,"Rating": x5})


# Plot the dataframe
ax = data[[ 'Tax 5%','gross income','Rating']].plot(kind='box', title='boxplot')

plt.boxplot(df['Total'])
#Trimmed mean
from scipy import stats
stats.trim_mean(df['Total'], 0.1)
#Summation

df['total'].sum()

#Frequency
count = df['Product line'].value_counts()
print(count)

#Variance
df.var()
#Correlation matrix
df.corr()
#Standard error of mean
df.sem()
#sum of squares
sos=0
for val in df['Total']:
  sos=val*val+sos
print(sos)

#Skewness
df.skew()

#kurtosis
sr = pd.Series(df['Total'])
print(sr.kurtosis())

g=sns.distplot(df['Total'])

import plotly.express as px

fig = px.scatter(df, x="x", y="y",
                 size='z',
                 hover_data=['z'])

fig.show()

#
sns.histplot(data=df, x='y', kde=True)