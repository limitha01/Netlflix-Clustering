from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

# Use 'Agg' backend for non-GUI plotting
matplotlib.use('Agg')

app = Flask(__name__)

# Load the dataset
try:
    df = pd.read_csv('Netflix_Dataset.csv')
    print("Dataset loaded successfully")
except FileNotFoundError:
    print("Error: Dataset file not found.")
    exit()

# Preprocessing
def preprocess_data(df):
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    # Label Encoding for categorical columns
    for column in ['rating', 'listed_in', 'type']:
        if column in df.columns:
            df[column] = label_encoder.fit_transform(df[column].astype(str))

    # One-Hot Encoding for 'type'
    if 'type' in df.columns:
        encoded_data = onehot_encoder.fit_transform(df[['type']])
        df = pd.concat([df, pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(['type']))], axis=1)

    df.fillna(0, inplace=True)
    return df

df = preprocess_data(df)
numeric_data = df.select_dtypes(include=[np.number])

# Load KMeans Model
try:
    kmeans_model = joblib.load('netflix_kmeans_model.pkl')
    print("K-Means model loaded successfully")
except FileNotFoundError:
    print("Error: K-Means model not found.")
    exit()

# Perform PCA for visualization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
pca = PCA(n_components=2)
data_2d = pca.fit_transform(scaled_data)

# Apply KMeans and Hierarchical Clustering
kmeans_labels = kmeans_model.predict(scaled_data)
hc_model = AgglomerativeClustering(n_clusters=3)
hc_labels = hc_model.fit_predict(scaled_data)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'feature{i}']) for i in range(1, 4)]
        scaled_input = scaler.transform([features])
        prediction = kmeans_model.predict(scaled_input)[0]
        result = f'Cluster {prediction} (K-Means)'
    except Exception as e:
        result = f"Error: {e}"
    return render_template('result.html', prediction=result)

@app.route('/visualize')
def visualize():
    try:
        img_folder = 'static/plots'
        os.makedirs(img_folder, exist_ok=True)

        # Plot K-Means Clusters
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=kmeans_labels, palette='viridis')
        plt.title('K-Means Clustering Visualization')
        kmeans_plot_path = os.path.join(img_folder, 'kmeans_plot.png')
        plt.savefig(kmeans_plot_path)
        plt.close()

        # Plot Hierarchical Clustering
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=hc_labels, palette='plasma')
        plt.title('Hierarchical Clustering Visualization')
        hc_plot_path = os.path.join(img_folder, 'hc_plot.png')
        plt.savefig(hc_plot_path)
        plt.close()

        return render_template('visualize.html', kmeans_image=kmeans_plot_path, hc_image=hc_plot_path)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
