#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data=pd.read_csv('GBvideos.csv')
data.head()
categories = pd.read_json('GB_category_id.json')


# In[2]:


data.head(5)


# In[3]:


data.isna().any().any()


# In[4]:


data.isna().sum().sum()


# In[5]:


data.loc[:, data.isnull().any()].columns


# In[6]:


data= data.fillna("")


# In[7]:


data.isna().sum().sum()


# In[8]:


data.describe()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[10]:


d=sns.boxplot(x=data['views'])


# In[11]:


f=sns.boxplot(x=data['views'],showfliers=False)


# In[12]:


d=sns.boxplot(x=data['likes'])


# In[13]:


f=sns.boxplot(x=data['likes'],showfliers=False)


# In[14]:


d=sns.boxplot(x=data['dislikes'])


# In[15]:


f=sns.boxplot(x=data['dislikes'],showfliers=False)


# In[16]:


d=sns.boxplot(x=data['comment_count'])


# In[17]:


f=sns.boxplot(x=data['comment_count'],showfliers=False)


# In[18]:


s=['views','likes','dislikes','comment_count']
for i in s:
    Q1 = data[i].quantile(0.25)
    Q3 = data[i].quantile(0.75)
    IQR = Q3 - Q1     
    filter=(data[i] > Q1 - 1.5 * IQR) & (data[i] < Q3 + 1.5 *IQR)
    data=data.loc[filter]


# In[19]:


del data['thumbnail_link']


# In[20]:


t=['comments_disabled','ratings_disabled','video_error_or_removed']
for i in t:
    data[i].replace(True,1,inplace=True)


# In[21]:


t=['comments_disabled','ratings_disabled','video_error_or_removed']
for i in t:
    data[i] = data[i].astype(int) 


# In[22]:


data.reset_index(inplace=True)
data['trending']=0
for i in range(1,25676):
    if(i%8==0):
        data['trending'][i]=0
    else:
        data['trending'][i]=1


# In[23]:


categories = {int(category['id']): category['snippet']['title'] for category in categories['items']}


# In[24]:


data=data[data['category_id']!=29]


# In[25]:


data.category_id = data.category_id.astype('category')


# In[26]:


data['trending_date'] = pd.to_datetime(data['trending_date'], format='%y.%d.%m').dt.date


# In[27]:


publish_time = pd.to_datetime(data.publish_time, format='%Y-%m-%dT%H:%M:%S.%fZ')
data['publish_date'] = publish_time.dt.date


# In[28]:


data['days_to_trending'] = (data.trending_date -data.publish_date).dt.days


# In[29]:


data.set_index(['trending_date','video_id'],inplace=True)


# In[30]:


data['dislike_percentage'] =data['dislikes'] / (data['dislikes'] + data['likes'])


# In[31]:


data['publish_time'] = pd.to_datetime(data['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
data["ldratio"] =data["likes"] / data["dislikes"]
data["perc_comment"] = data["comment_count"] /data["views"]
data["perc_reaction"] = (data["likes"] +data["dislikes"]) /data["views"]


# In[32]:


data["hour"]=data['publish_time'].dt.hour
by_hour =data.groupby("hour").mean()

plt.plot(by_hour.index.values, by_hour["views"])
plt.scatter(by_hour.index.values, by_hour["views"])
plt.xlabel("Hour of Day")
plt.ylabel("Average Number of Views")
plt.title("Average Amount of Views on Trending Videos by the Hour")
plt.show()


# In[33]:


def distribution_cont(data, var):
    plt.hist(data[data["dislikes"] != 0][var])
    plt.xlabel(f"{var}")
    plt.ylabel("Count")
    plt.title(f"Distribution of Trending Video {var}")
    plt.show()
for i in ["views", "likes", "dislikes", "comment_count", "ldratio", "perc_reaction", "perc_comment"]:
    distribution_cont(data, i)


# In[34]:


sns.set(font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
sns_ax = sns.countplot([categories[i] for i in data.category_id])
_, labels = plt.xticks()
_ = sns_ax.set_xticklabels(labels, rotation=60)


# In[41]:


table = pd.pivot_table(data, index=data.index.labels[0])
table.index =data.index.levels[0]
_ = table[['likes','dislikes','comment_count']].plot()
_ = table[['views']].plot()
_ = table[['comments_disabled','ratings_disabled']].plot()


# In[36]:


corrolation_list = ['views', 'likes', 'dislikes', 'comment_count']
hm_data = data[corrolation_list].corr() 
display(hm_data)


# In[37]:


import numpy as np
contvars = data[["views", "likes", "dislikes", "comment_count", "ldratio", "perc_comment", "perc_reaction"]]
corr = contvars.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# In[39]:


by_channel =data.groupby(["channel_title"]).size().sort_values(ascending = False).head(20)
sns.barplot(by_channel.values, by_channel.index.values, palette = "rocket")
plt.title("Top 20 Most Frequent Trending Youtube Channels")
plt.xlabel("Video Count")
plt.show()


# In[40]:


top_channels2 = data.groupby("channel_title").size().sort_values(ascending = False)
top_channels2 = list(top_channels2[top_channels2.values >= 20].index.values)
only_top2 =data
for i in list(data["channel_title"].unique()):
    if i not in top_channels2:
        only_top2 = only_top2[only_top2["channel_title"] != i]

by_views = only_top2.groupby(["channel_title"]).mean().sort_values(by = "views", ascending = False).head(20)
sns.barplot(by_views["views"], by_views.index.values, palette = "rocket")
plt.title("Top 20 Most Viewed Trending Youtube Channels")
plt.xlabel("Average Views")
plt.show()


# In[41]:


top_channels =data.groupby("channel_title").size().sort_values(ascending = False)
top_channels = list(top_channels[top_channels.values >= 20].index.values)
only_top = data
for i in list(data["channel_title"].unique()):
    if i not in top_channels:
        only_top = only_top[only_top["channel_title"] != i]

like_channel = only_top[only_top["dislikes"] != 0].groupby(["channel_title"]).mean().sort_values(by = "ldratio", ascending = False).head(20)
sns.barplot(like_channel["ldratio"], like_channel.index.values, palette = "rocket")
plt.title("Top 20 Most Liked Trending Youtube Channels")
plt.xlabel("Average Like to Dislike Ratio")
plt.show()


# In[42]:


def over_time(youtube, var):    
    averages = data[data["dislikes"] != 0].groupby("trending_date").mean()
    plt.plot(averages.index.values, averages[var])
    plt.xticks(rotation = 90)
    plt.xlabel("Date")
    plt.ylabel(f"Average {var}")
    plt.title(f"Average {var} Over Time (11/14/17 - 6/14/18)")
    plt.show()
over_time(data, "ldratio")


# In[43]:


over_time(data, "perc_reaction")


# In[44]:


data.reset_index(inplace=True)


# In[45]:


data.drop(['index'], axis=1)


# In[46]:


from sklearn.model_selection import train_test_split
predictors =data.drop(['dislike_percentage', 'ldratio','title','trending_date', 'video_id','channel_title','category_id','publish_time','tags','description','trending','publish_date','days_to_trending' ], axis=1)
target =data['trending']
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.7, random_state = 0)


# In[63]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[48]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[49]:


from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[50]:


# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[51]:


from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[52]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[53]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[54]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian,acc_linear_svc, acc_decisiontree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




