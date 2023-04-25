!pip install pyglet

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from IPython import display

df = pd.read_csv("/content/sample_data/tesla.csv")

df.shape

plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'], color='red')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize = 18)
plt.show()

sns.histplot(df["High"], kde=True, color="m")

sns.displot(df["High"], kde=True, color="m")

sns.jointplot(x = "Open", y = "Close",
              kind = "scatter", data = df)

plt.show()

sns.histplot(data=df, x="Open")

fig = go.Figure(data=go.Ohlc(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']))
fig.show()

import matplotlib.pyplot as plt
plt.scatter(df['Open'], df['Close'], c ="blue")

sns.pairplot(df)
plt.show()

#column chart plot between 2 attributes
df1 = df.head(10)
df1.plot.bar()
plt.bar(df1['High'], df1['Low'])
plt.xlabel("High")
plt.ylabel("Low")
plt.show()

import plotly.express as px
df1=df.head(15)
fig = px.line(df1, x="Date", y="Volume")
fig.show()

#Density Chart
df['Adj Close'].plot.density(color='red')
plt.title('Density plot for Adj Close')
plt.show()

'''End of first code============================'''

import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/My Drive/ADS/supermarket_sales - Sheet1.csv')

df.head()

# Scatter plot
import matplotlib.pyplot as plt

plt.scatter(df['Tax 5%'], df['Unit price'], c="blue")

# BoxPlot
x2 = df['Tax 5%']
x4 = df['gross income']
x5 = df['Rating']
data = pd.DataFrame({"Tax 5%": x2, "gross income": x4, "Rating": x5})

# Plot the dataframe
ax = data[['Tax 5%', 'gross income', 'Rating']].plot(kind='box', title='boxplot')

# Distribution Chart / Distplot
import seaborn as sns

g = sns.distplot(df['Total'])

# JoinPlot
sns.jointplot(x='Total', y='Tax 5%', data=df)

# Pairplot
sns.pairplot(df)
# to show
plt.show()

# Histogram
df['Rating'].hist()
plt.show()

lst = df['Product line'].unique()
print(lst)
# t=[0,0,0,0,0,0]
# for i in df:
#   print(i)
#   if i[5] in lst:
#     if i[5]=='Health and beauty':
#       t[0]=t[0]+i[9]

# print(t)

t = [23, 17, 35, 29, 12, 41]
plt.pie(t, labels=lst, autopct='% 1.1f %%', shadow=True)
plt.show()

# Density Chart
df['Rating'].plot.density(color='green')
plt.title('Density plot for Speeding')
plt.show()

# scatter Matrix
pd.plotting.scatter_matrix(df)

# rugplot
import seaborn as sns
import matplotlib.pyplot as plt

data = df
data.head(5)
plt.figure(figsize=(15, 5))
sns.rugplot(data=data, x="Total")
plt.show()

# column chart
# plot between 2 attributes
df1 = df.head(10)
df1.plot.bar()
plt.bar(df1['Gender'], df1['Total'])
plt.xlabel("Gender")
plt.ylabel("Total")
plt.show()

import plotly.express as px

df1 = df.head(15)
fig = px.line(df1, x="Date", y="Total", color='City')
fig.show()

# Bubble Chart
import plotly.express as px

fig = px.scatter(df1, x="Total", y="Tax 5%", size="Quantity", color="City", hover_name="Product line", log_x=True,
                 size_max=60)
fig.show()

# Parallel
import plotly.express as px

df1 = df.sample(n=100)
fig = px.parallel_coordinates(df1, color="Total",
                              dimensions=['Quantity', 'Unit price', 'Rating', ],
                              color_continuous_scale=px.colors.diverging.Tealrose,
                              color_continuous_midpoint=2)
fig.show()

# Creating Andrews curves
df1 = df[['Quantity', 'Total', 'Rating']]
df1 = df1.sample(n=50)
x = pd.plotting.andrews_curves(df1, 'Rating')

# plotting the Curve
x.plot()

# Display
plt.show()

import plotly.express as px

# df = px.data.medals_wide(indexed=True)
fig = px.imshow(df1)
fig.show()

import plotly.express as px

fig = px.line(df1, x='Date', y="Total")
fig.show()
