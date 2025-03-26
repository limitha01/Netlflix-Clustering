from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import matplotlib
from sklearn.preprocessing import StandardScaler

# Use 'Agg' backend for non-GUI plotting
matplotlib.use('Agg')

# Initialise Flask app
app = Flask(__name__)

# Load trained model
try:
    model = joblib.load('netflix_kmeans_model.pkl')
    model_name = 'K-Means (netflix_kmeans_model.pkl)'
    print("K-Means model loaded successfully")
except FileNotFoundError:
    print("Error: Model file not found. Ensure 'netflix_kmeans_model.pkl' exists.")
    exit()

# Load dataset for visualization
try:
    df = pd.read_csv('Netflix_Dataset.csv')
    print("Dataset loaded successfully")

    numeric_data = df.select_dtypes(include=[np.number])

    if numeric_data.shape[1] < 3:
        raise ValueError("Error: Dataset must have at least 3 numeric columns for clustering.")

    # Perform Standard Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data.iloc[:, :3])

    # Perform PCA for visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(scaled_data)

    # Predict using the trained model
    labels = model.predict(scaled_data)

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'feature{i}']) for i in range(1, 4)]

        # Check for the correct number of features
        if len(features) != 3:
            raise ValueError("Please provide exactly 3 features.")

        # Scale input data to match training data
        scaled_input = scaler.transform([features])

        prediction = model.predict(scaled_input)[0]
        result = f'Cluster {prediction} (Model: {model_name})'
    except Exception as e:
        result = f"Error: {e}"
    return render_template('result.html', prediction=result)

@app.route('/visualize')
def visualize():
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', edgecolors='k')
        plt.title('Cluster Visualization using PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster')

        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)

        # Save the image
        image_path = os.path.join('static', 'cluster_plot.png')
        plt.savefig(image_path)
        plt.close()

        return render_template('visualize.html', image_path=image_path, model_name=model_name)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
