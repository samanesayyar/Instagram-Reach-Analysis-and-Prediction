# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:15:46 2024

@author: Samane
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.renderers.default='browser'

data = pd.read_csv("archive/Instagram data.csv", encoding = 'latin1')
# print(data.head())
# print(data.isnull().sum())
# print(data.info())

##distribution of impressions I have received from home
# plt.figure(figsize=(10, 8))
# plt.style.use('fivethirtyeight')
# plt.title("Distribution of Impressions From Home")
# sns.distplot(data['From Home'])
# plt.show()

##wordCloud of the caption column
text="".join(i for i in data['Caption'])
stopwords=set(STOPWORDS)
wordcloud=WordCloud(stopwords=stopwords, background_color="black").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
### the relationship between the number of likes and the number of impressions
# figure=px.scatter(data_frame=data,x="Impressions",y="Likes", size="Likes", trendline="ols",title = "Relationship Between Likes and Impressions")
# figure.show()
### Relationship Between Post Saves and Total Impressions
# figure = px.scatter(data_frame = data, x="Impressions",
#                     y="Saves", size="Saves", trendline="ols", 
#                     title = "Relationship Between Post Saves and Total Impressions")
# figure.show()

###correlation of all the columns with the Impressions column:
# correlation = data.corr(numeric_only=True)
# print(correlation["Impressions"].sort_values(ascending=False))

##relationship between the total profile visits and the number of followers
# figure = px.scatter(data_frame = data, x="Profile Visits",
#                     y="Follows", size="Follows", trendline="ols", 
#                     title = "Relationship Between Profile Visits and Followers Gained")
# figure.show()
###Instagram Reach Prediction Model
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)
# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
print(model.predict(features))
