import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("clustering-data.csv")

X = data[['ApplicantIncome', 'LoanAmount']].values

k = 3

# Random initial centroids
centroids = X[np.random.choice(len(X), k, replace=False)]

while True:
    # Step 1: Calculate distances
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))

    # Step 2: Assign clusters
    clusters = np.argmin(distances, axis=0)

    # Step 3: New centroids
    new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

    # Stop if centroids don’t change
    if np.allclose(centroids, new_centroids):
        break

    centroids = new_centroids

# Plot clusters
colors = ['blue', 'green', 'cyan']

for i in range(k):
    plt.scatter(
        X[clusters == i][:,0],
        X[clusters == i][:,1],
        color=colors[i]
    )

plt.scatter(centroids[:,0], centroids[:,1], color='red', s=150)
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.title("K-Means Clustering")
plt.show()
