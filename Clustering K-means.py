import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Loaded the dataset
data = pd.read_csv('C:/Users/singh/Documents/CS504/Project/Sprint 3/Dataset/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_20241111.csv')
print(f"Initial dataset shape: {data.shape}")

# Selected relevant features for clustering
# These columns were chosen as they provide demographic and case status information
selected_columns = [
    'age_group', 'sex', 'race', 'ethnicity', 
    'hosp_yn', 'icu_yn', 'death_yn', 
    'underlying_conditions_yn'
]
subset_data = data[selected_columns].copy()
print(f"Subset data shape (selected columns only): {subset_data.shape}")
print(f"Sample data before processing:\n{subset_data.head()}")

# Handling missing values by replacing them with 'Unknown'
# making sure that clustering algorithms can work without interruptions
subset_data.fillna('Unknown', inplace=True)
print("Missing values handled by filling with 'Unknown'.")

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

# Adding cluster labels to the original data for reference
data['Cluster'] = clusters
print("Cluster labels added to the original dataset.")
print(f"Sample data with clusters:\n{data[['age_group', 'sex', 'race', 'Cluster']].head()}")

# Using PCA (Principal Component Analysis) to reduce data to 2 dimensions for visualization
# This helps visualizing the clusters in a simple 2D plot
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
print(f"PCA transformation applied. Explained variance ratio: {pca.explained_variance_ratio_}")

# Plotting the clusters using the first two PCA components
plt.figure(figsize=(8, 6))
for cluster in np.unique(clusters):
    plt.scatter(
        pca_data[clusters == cluster, 0],  # First PCA component
        pca_data[clusters == cluster, 1],  # Second PCA component
        label=f'Cluster {cluster}'
    )
plt.title('Clustering of COVID-19 Cases')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

print("Clustering visualization completed.")
