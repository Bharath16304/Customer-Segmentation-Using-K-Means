import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Load Data
df = pd.read_csv('customer_data.csv')  # columns: age, income, frequency, spending

# Preprocessing
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['age', 'income', 'frequency', 'spending']])

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

# 3D Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['income'], df['spending'], df['frequency'], c=df['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
ax.set_xlabel('Income')
ax.set_ylabel('Spending')
ax.set_zlabel('Frequency')
plt.show()
