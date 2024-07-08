import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('marketing_campaign.csv')

# Title and Introduction
st.title("Customer Personality Analysis")
st.write("""
    This application uses clustering and the Apriori algorithm to analyze customer personality data.
""")

# Display Data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Data Preprocessing
st.subheader('Data Preprocessing')
selected_columns = st.multiselect('Select columns for clustering', data.columns.tolist(), default=['Income', 'Age', 'MntWines', 'MntFruits'])

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[selected_columns])

# Clustering
st.subheader('Clustering')
num_clusters = st.slider('Select number of clusters', 2, 10, 3)
kmeans = KMeans(n_clusters=num_clusters)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualize Clusters
st.subheader('Cluster Visualization')
fig, ax = plt.subplots()
sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]], hue=data['Cluster'], palette='viridis', ax=ax)
st.pyplot(fig)

# Apriori Algorithm
st.subheader('Market Basket Analysis with Apriori Algorithm')
min_support = st.slider('Select minimum support', 0.01, 0.5, 0.1)
min_threshold = st.slider('Select minimum threshold for association rules', 0.1, 1.0, 0.5)

basket = (data.groupby(['ID'])[['MntWines', 'MntFruits']]
          .sum().applymap(lambda x: 1 if x > 0 else 0))

frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)

st.write("Frequent Itemsets")
st.write(frequent_itemsets)

st.write("Association Rules")
st.write(rules)

# Footer
st.markdown("---")
st.write("© 2024 [Your Name]. All rights reserved.")
