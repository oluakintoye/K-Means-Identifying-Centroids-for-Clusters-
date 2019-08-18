from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from datacleaner import autoclean
import random
import numpy as np
import pandas as pd





def main():
    data = pd.read_csv("College.csv",index_col=0)

    model = KMeans(n_clusters=2)

    model = model.fit(scale(data.drop('Private', axis=1)))

    centroids = model.cluster_centers_
    print(centroids)
    print(centroids[0][5])
    print(centroids[1][5])
    print(centroids[0][7])
    print(centroids[1][7])
    print(model.labels_)

    plt.figure(figsize=(8,6))
    plt.scatter(data['Outstate'], data['F.Undergrad'], c = model.labels_.astype(np.float))
    plt.scatter(centroids[0][7], centroids[1][7], s=200, c='g', marker='s')
    plt.scatter(centroids[0][5], centroids[1][5], s=200, c='r', marker='s')
    plt.show()


if __name__ == "__main__":
    main()

