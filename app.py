import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings 
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
customers = pd.read_csv('Mall_Customers.csv')

# Streamlit App
st.title("Customer Segmentation Project")
st.write("This is a project showcasing customer segmentation using KMeans clustering.")

# Description
st.write("""
## Description
This project analyzes customer data from a mall and performs customer segmentation using KMeans clustering.
The project involves data visualization, exploratory analysis, and clustering analysis.
""")

# Display the plots and analysis
# Distribution Plot
st.subheader("Distribution Plot")
sns.distplot(customers['Annual Income (k$)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Density')
plt.title('Distribution of Annual Income')
st.pyplot()

    # Histograms
st.subheader("Histograms")
columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    sns.set_style("darkgrid")
    plt.figure()
    sns.histplot(customers[i], kde=True, shrink=True, bins=20, color='blue', label='Histogram')
    plt.xlabel(i)
    plt.ylabel('Density')
    plt.title(f'Distribution of {i}')
    plt.legend()
    st.pyplot(plt)

    # KDE Plots
st.subheader("KDE Plots")
for i in columns:
    sns.set_style("darkgrid")
    plt.figure()
    sns.kdeplot(data=customers, x=i, shade=True, hue='Gender')
    plt.xlabel(i)
    plt.ylabel('Density')
    plt.title(f'Distribution of {i}')
    st.pyplot(plt)

    # Pairplot
st.subheader("Pairplot")
plot = sns.pairplot(data=customers, hue='Gender', markers=['o', 's'])
st.pyplot(plot)

    # Heatmap
st.subheader("Correlation Heatmap")
plt.figure()
sns.heatmap(customers.corr(numeric_only=True), annot=True, linewidth=.5, cmap='coolwarm')
st.pyplot(plt)

# Clustering Analysis
st.subheader("Clustering Analysis")
st.write("""
The following sections demonstrate customer segmentation using KMeans clustering.
""")

# Income Clustering
st.subheader("Income Clustering")
ks = range(1, 11)
inertias = []
samples = customers[['Annual Income (k$)']]

for k in ks:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(samples)
    inertias.append(model.inertia_)

plt.figure()
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
st.pyplot(plt)

knn = KMeans(n_clusters=3, random_state=42)
knn.fit(customers[['Annual Income (k$)']])
customers['income_cluster'] = knn.labels_
st.write(f'Inertia: {knn.inertia_}')
st.write(f'Clusters:\n{customers["income_cluster"].value_counts()}')

# Clusters Summary
st.write("Clusters Summary:")
st.write(customers.groupby('income_cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())

# Spending and Income Clustering
st.subheader("Spending and Income Clustering")
ks2 = range(1, 11)
inertias2 = []
samples2 = customers[['Annual Income (k$)', 'Spending Score (1-100)']]
for k in ks2:
    model2 = KMeans(n_clusters=k, random_state=42)
    model2.fit(samples2)
    inertias2.append(model2.inertia_)

plt.figure()
plt.plot(ks2, inertias2, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks2)
st.pyplot(plt)

knn2 = KMeans(n_clusters=5, random_state=42)
knn2.fit(customers[['Annual Income (k$)', 'Spending Score (1-100)']])
customers['spending_income_clust'] = knn2.labels_
st.write(f'Inertia = {knn2.inertia_}')

centers = pd.DataFrame(knn2.cluster_centers_, columns=['x', 'center'])
plt.figure(figsize=(10,8))
plt.scatter(data=centers, x= 'x', y='center', s=60, c='red', marker='o')
sns.scatterplot(data=customers,x='Annual Income (k$)', y='Spending Score (1-100)', hue='spending_income_clust', palette='deep', )
# Add KDE plots around each cluster center
for index, row in centers.iterrows():
    sns.kdeplot(data=customers[customers['spending_income_clust'] == index],
                x='Annual Income (k$)', y='Spending Score (1-100)', shade=True, alpha=0.15, cmap='Reds')

plt.legend()
st.pyplot(plt)
