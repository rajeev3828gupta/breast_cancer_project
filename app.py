from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# loading model
model = pickle.load(open('models/model.pkl', 'rb'))

# Load and fit scaler on training data
df = pd.read_csv('notebook and dataset/breast cancer.csv')
# Drop only diagnosis and the unnamed column, keep id as first feature
df = df.drop(['diagnosis'], axis=1, errors='ignore')
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)
    
scaler = StandardScaler()
scaler.fit(df)

# flask app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = request.form['feature']
        features = features.split(',')
        features = [f.strip() for f in features]
        np_features = np.asarray(features, dtype=np.float32)

        if len(np_features) != 31:
            message = [f'Error: Expected 31 features, got {len(np_features)}']
            return render_template('index.html', message=message, error=True)

        # Scale features before prediction
        scaled_features = scaler.transform(np_features.reshape(1, -1))
        
        # prediction
        pred = model.predict(scaled_features)
        message = ['Cancrouse' if pred[0] == 1 else 'Not Cancrouse']
        return render_template('index.html', message=message, error=False)
    except Exception as e:
        message = [f'Error: {str(e)}']
        return render_template('index.html', message=message, error=True)


if __name__ == '__main__':
    app.run(debug=True)

