#libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:/Users/singh/Documents/CS504/Project/Sprint 3/Dataset/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_20241111.csv'
data = pd.read_csv(file_path)


# Select numerical columns for clustering
numerical_columns = ['case_positive_specimen_interval', 'case_onset_interval']
data_selected = data[numerical_columns]

# Handle missing values by dropping rows with NaN in selected columns
data_cleaned = data_selected.dropna()

# Reduce sample size for memory-efficient computation 
sampled_data = data_cleaned.sample(n=1000, random_state=42)

# Standardize the sampled data
scaler = StandardScaler()
data_scaled_sampled = scaler.fit_transform(sampled_data)

# Perform hierarchical clustering
linkage_matrix_sampled = linkage(data_scaled_sampled, method='ward')

# Plot the dendrogram for the sampled data
plt.figure(figsize=(10, 7))
plt.title("Hierarchical Clustering Dendrogram (Reduced Sample)")
dendrogram(linkage_matrix_sampled, truncate_mode='lastp', p=12, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
plt.xlabel("Cluster Size")
plt.ylabel("Distance")
plt.show()

# Create clusters using a distance threshold or desired number of clusters
max_distance = 5  # Example for distance
clusters = fcluster(linkage_matrix_sampled, max_distance, criterion='distance')

# Add the cluster labels to the original sampled data
sampled_data['Cluster'] = clusters

# Display the first few rows with the cluster labels
print(sampled_data.head())

# Visualize the clusters in a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(sampled_data['case_positive_specimen_interval'], 
            sampled_data['case_onset_interval'], 
            c=sampled_data['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.title("Cluster Visualization")
plt.xlabel("Case Positive Specimen Interval")
plt.ylabel("Case Onset Interval")
plt.colorbar(label="Cluster")
plt.show()
