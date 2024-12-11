import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load the dataset (replace with your dataset if running locally)
file_path = 'C:/Users/singh/Documents/CS504/Project/Sprint 3/Dataset/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_20241111.csv'
data = pd.read_csv(file_path)

# Data preprocessing: Aggregating by state
state_data = data.groupby('res_state').agg({
    'age_group': lambda x: x.value_counts(normalize=True).to_dict(),
    'sex': lambda x: x.value_counts(normalize=True).to_dict(),
    'race': lambda x: x.value_counts(normalize=True).to_dict(),
    'hosp_yn': lambda x: x.value_counts(normalize=True).to_dict(),
    'icu_yn': lambda x: x.value_counts(normalize=True).to_dict(),
    'underlying_conditions_yn': lambda x: x.value_counts(normalize=True).to_dict()
}).reset_index()

# Transform dictionaries into separate columns for analysis
state_features = pd.json_normalize(state_data['age_group']).add_prefix('age_')
sex_features = pd.json_normalize(state_data['sex']).add_prefix('sex_')
race_features = pd.json_normalize(state_data['race']).add_prefix('race_')
hosp_features = pd.json_normalize(state_data['hosp_yn']).add_prefix('hosp_')
icu_features = pd.json_normalize(state_data['icu_yn']).add_prefix('icu_')
conditions_features = pd.json_normalize(state_data['underlying_conditions_yn']).add_prefix('conditions_')

# Combine all features into a single DataFrame
features = pd.concat([state_features, sex_features, race_features, hosp_features, icu_features, conditions_features], axis=1)
features.fillna(0, inplace=True)  # Filling missing values with 0 (no cases for that category)

# Hierarchical clustering using Ward's method
linkage_matrix = linkage(features, method='ward')

# Dendrogram visualization
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=state_data['res_state'].values, leaf_rotation=90, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram (States)')
plt.xlabel('States')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()
