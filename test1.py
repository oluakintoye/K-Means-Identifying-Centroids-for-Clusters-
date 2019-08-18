import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report

df=pd.read_csv("College.csv",index_col=0)

print(df.head())
print(df.info())
print(df.describe())

sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad', data=df, hue='Private',
           palette='coolwarm', height=6, aspect=1, fit_reg=False)
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)

kmeans.fit(df.drop('Private', axis=1))

centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['Outstate'], df['F.Undergrad'], s =50, c='b')


plt.show()


sns.set_style('whitegrid')
sns.lmplot('S.F.Ratio','Grad.Rate', data=df, hue='Private',
           palette='coolwarm', height=6, aspect=1, fit_reg=False)
plt.show()

def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0

df['Cluster'] = df['Private'].apply(converter)
df.head()

print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# plt.scatter(df['S.F.Ratio'],df['Grad.Rate'], s =50, c='b')
# plt.scatter(1.03631389e+04, 6.75925926e+01, s=200, c='g', marker='s')
# plt.scatter(1.81323468e+03,6.51195815e+01, s=200, c='r', marker='s')
