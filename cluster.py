from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from datacleaner import autoclean
import random
import numpy as np
import pandas as pd

# Creating fake income/age cluster for N people in K Clusters
def createClusteredData(N, k):
    random.seed(10)
    pointsPerCluster = float(N)/k
    X = []
    for i in range(k):
        incomeCentroid = random.uniform(20000.0, 200000.0)
        ageCentroid = random.uniform(21.0,70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
    X = np.array(X)
    return X
#
# PATH = "KDDTest+.csv"
#
# data = pd.read_csv(PATH, header=0, na_values="?")
#
# data = autoclean(data)

def main():
    data = createClusteredData(100, 5)

    model = KMeans(n_clusters=5)

    model = model.fit(scale(data))

    model.cluster_centers_

    print(model.labels_)

    plt.figure(figsize=(8,6))
    plt.scatter(data[:,0], data[:,1], c = model.labels_.astype(np.float))

    plt.show()


if __name__ == "__main__":
    main()

