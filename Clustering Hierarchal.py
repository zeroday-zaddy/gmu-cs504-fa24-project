import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
data = pd.read_csv('/data/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_20241111.csv')
print(f"Initial dataset shape: {data.shape}")

# Selecting relevant features for clustering
selected_columns = [
    'age_group', 'sex', 'race', 'ethnicity', 
    'hosp_yn', 'icu_yn', 'death_yn', 
    'underlying_conditions_yn'
]
subset_data = data[selected_columns].copy()
print(f"Subset data shape (selected columns only): {subset_data.shape}")

# missing values by replacing them with 'Unknown'
subset_data.fillna('Unknown', inplace=True)
print("Missing values handled by filling with 'Unknown'.")

# Encoding categorical features into numeric format using OneHotEncoder
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(subset_data).toarray()
print(f"Encoded data shape: {encoded_data.shape}")

# Scaling the encoded data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data)
print("Data scaled.")

# Selecting a subset of the data to reduce memory usage
subset_size = 5000  # to see the samples for clustering
subset_indices = np.random.choice(scaled_data.shape[0], subset_size, replace=False)
scaled_data_subset = scaled_data[subset_indices]
print(f"Subset of size {subset_size} selected for clustering.")

# Performing hierarchical clustering on the subset
hierarchical_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = hierarchical_clustering.fit_predict(scaled_data_subset)
print(f"Hierarchical clustering applied with 3 clusters on a subset of {subset_size} samples.")

# Generating and plotting the dendrogram for the subset
linkage_matrix = linkage(scaled_data_subset, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=10)
plt.title('Dendrogram for Hierarchical Clustering (Subset)')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
plt.show()

# Visualizing clusters using PCA for 2D representation
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data_subset)
plt.figure(figsize=(8, 6))
for cluster in np.unique(clusters):
    plt.scatter(
        pca_data[clusters == cluster, 0], 
        pca_data[clusters == cluster, 1], 
        label=f'Cluster {cluster}'
    )
plt.title('Hierarchical Clustering Visualization (Subset)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

print("Hierarchical clustering completed and visualized.")
