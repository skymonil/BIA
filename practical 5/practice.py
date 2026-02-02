import numpy as np, pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load Ds
data = pd.read_csv('clustering-data.csv')

# Keep reqd columns
X = data[['ApplicantIncome', 'LoanAmount']]

kmeans = KMeans(n_clusters=8,random_state=0)
clusters = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
colors =['blue', 'cyan', 'green']

for i in range:
    plt.scatter(
        X[clusters == i]['ApplicantIncome'],
        X[clusters == i]['LoanAmount']
    )