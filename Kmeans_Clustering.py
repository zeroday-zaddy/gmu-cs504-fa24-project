# libraries and dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

file_path = 'C:/Users/singh/Documents/CS504/Project/Sprint 3/Dataset/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_20241111.csv'
data = pd.read_csv(file_path)


# Selecting relevant features for clustering
# These columns were chosen as they provide demographic and case status information
selected_columns = [
    'age_group', 'sex', 'race', 
    'hosp_yn', 'icu_yn', 'death_yn'
]
subset_data = data[selected_columns].copy()
print(f"Sample data before processing:\n{subset_data.head()}")


# Handling missing values by replacing them with 'Unknown'
# making sure that clustering algorithms can work without interruptions
subset_data.fillna('Unknown', inplace=True)

# Encoding categorical features into numeric format using OneHotEncoder
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(subset_data).toarray()
print(f"Encoded data shape: {encoded_data.shape}")
print(f"Feature categories: {encoder.categories_}")

# Scaling the encoded data using StandardScaler
# Scaling ensures that all features contribute equally to the clustering process
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data)
print(f"Data scaled. Sample scaled data:\n{scaled_data[:5]}")


# Applying KMeans clustering to find groups in the data
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
print(f"KMeans clustering applied with {n_clusters} clusters.")
print(f"Cluster centers:\n{kmeans.cluster_centers_}")
print(f"Cluster assignments for first 10 samples: {clusters[:10]}")

# Adding cluster labels to the subset data for reference
subset_data['Cluster'] = clusters
print(f"Sample data with clusters:\n{subset_data[['age_group', 'sex', 'race', 'Cluster']].head()}")

# Using PCA (Principal Component Analysis) to reduce data to 2 dimensions for 2D plot visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Plotting the clusters using the first two PCA components with explicit labels and cluster annotations
plt.figure(figsize=(12, 8))
colors = ['blue', 'orange', 'green']
for cluster in np.unique(clusters):
    plt.scatter(
        pca_data[clusters == cluster, 0],  # First PCA component
        pca_data[clusters == cluster, 1],  # Second PCA component
        color=colors[cluster],
        label=(
            f"Cluster {cluster}:\n"
            f"{'Mild Cases' if cluster == 0 else 'Moderate Cases' if cluster == 1 else 'Severe Cases'}"
        ),
        alpha=0.6
    )

# Adding explicit labels for axes
plt.title('Clustering of COVID-19 Case Severity Based on Demographics and Outcomes', fontsize=14)
plt.xlabel('PCA Component 1: Demographics (Age, Sex, Race)', fontsize=12)
plt.ylabel('PCA Component 2: Case Outcomes (Hospitalization, ICU, Death)', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True)
plt.show()
