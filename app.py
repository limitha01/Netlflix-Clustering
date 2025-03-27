from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from kmodes.kprototypes import KPrototypes
import os

app = Flask(__name__)

# Load Dataset
try:
    df = pd.read_csv('Netflix_Dataset.csv')
except FileNotFoundError:
    print("Error: Netflix_Dataset.csv not found.")
    exit()

# Load Model
try:
    model = joblib.load('netflix_kmeans_model.pkl')
except FileNotFoundError:
    print("Error: Model file not found.")
    exit()

# Preprocess Columns
label_encoder = LabelEncoder()
df['Rating_Label'] = label_encoder.fit_transform(df['Rating'])
df['ListedIn_Label'] = label_encoder.fit_transform(df['Listed In'])
df['Dates'] = pd.to_datetime(df['Dates'], errors='coerce')
df['Days_Since'] = (pd.Timestamp.now() - df['Dates']).dt.days

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    column_name = request.form.get('column')
    try:
        if column_name == 'Type':
            one_hot = pd.get_dummies(df['Type'], prefix='Type')
            data = one_hot.values
        elif column_name == 'Dates':
            data = df[['Days_Since']].values
        elif column_name == 'Rating':
            data = df[['Rating_Label']].values
        elif column_name == 'Listed In':
            data = df[['ListedIn_Label']].values
        else:
            return "Invalid Column Selected"

        # Apply K-Prototypes
        kproto = KPrototypes(n_clusters=4, init='Cao', n_init=5)
        clusters = kproto.fit_predict(data, categorical=[0])
        df['Cluster'] = clusters

        return render_template('result.html', column=column_name, clusters=clusters, model_name='K-Prototypes')
    except Exception as e:
        return str(e)

@app.route('/visualize', methods=['POST'])
def visualize():
    column_name = request.form.get('column')
    try:
        if column_name in ['Rating', 'Listed In']:
            data = df[[f'{column_name}_Label']].values
        else:
            data = df[['Days_Since']].values

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)

        plt.figure(figsize=(8, 6))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=df['Cluster'], cmap='viridis')
        plt.title(f'{column_name} Cluster Visualization using PCA')
        plt.colorbar(label='Cluster')

        # Save Image
        if not os.path.exists('static'):
            os.makedirs('static')
        plt.savefig('static/cluster_plot.png')
        plt.close()

        return render_template('visualize.html', image_path='static/cluster_plot.png', model_name='K-Prototypes')
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
