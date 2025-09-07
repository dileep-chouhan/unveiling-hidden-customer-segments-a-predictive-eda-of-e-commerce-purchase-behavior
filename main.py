import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_customers = 300
data = {
    'CustomerID': range(1, num_customers + 1),
    'Age': np.random.randint(18, 65, num_customers),
    'Income': np.random.randint(20000, 150000, num_customers),
    'SpendingScore': np.random.randint(1, 101, num_customers),
    'ProductA': np.random.randint(0, 10, num_customers), #Purchases of Product A
    'ProductB': np.random.randint(0, 10, num_customers), #Purchases of Product B
    'ProductC': np.random.randint(0, 10, num_customers)  #Purchases of Product C
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preprocessing ---
# (In a real-world scenario, this would involve handling missing values, outliers, etc.)
# For this example, the data is already clean.
# --- 3. Exploratory Data Analysis (EDA) ---
# Calculate descriptive statistics
print("Descriptive Statistics:")
print(df.describe())
# --- 4. Customer Segmentation using Hierarchical Clustering ---
# Select features for clustering (excluding CustomerID)
X = df[['Age', 'Income', 'SpendingScore', 'ProductA', 'ProductB', 'ProductC']]
# Perform hierarchical clustering
linked = linkage(X, 'ward')
# Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Customer ID')
plt.ylabel('Distance')
plt.tight_layout()
# Save dendrogram plot
dendrogram_filename = 'dendrogram.png'
plt.savefig(dendrogram_filename)
print(f"Dendrogram saved to {dendrogram_filename}")
# --- 5. Visualization of Customer Segments (Example: Scatter Plot) ---
#  (This section could be expanded to include other visualizations based on the identified segments)
plt.figure(figsize=(8,6))
plt.scatter(df['Income'], df['SpendingScore'], c=linked[:,2], cmap='viridis') #Color points by distance in dendrogram
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation based on Income and Spending Score')
plt.colorbar(label='Cluster Distance')
plt.tight_layout()
# Save scatter plot
scatter_plot_filename = 'customer_segments.png'
plt.savefig(scatter_plot_filename)
print(f"Scatter plot saved to {scatter_plot_filename}")
print("EDA and Customer Segmentation Complete.")