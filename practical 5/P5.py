import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("clustering-data.csv")

# keep reqd cols
X = data[['ApplicantIncome', 'LoanAmount']]



# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X)

# Get centroids
centroids = kmeans.cluster_centers_

# Plot clusters
colors = ['blue', 'green', 'cyan']

for i in range(3):
    plt.scatter(
        X[clusters == i]['ApplicantIncome'],
        X[clusters == i]['LoanAmount'],
        color=colors[i],
    )

plt.scatter(centroids[:,0], centroids[:,1], color='red', s=150)
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.title("K-Means Clustering (sklearn)")
plt.show()
