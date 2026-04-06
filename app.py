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
        patient_id = request.form['patient_id'].strip()
        
        # Convert ID to int
        try:
            patient_id_int = int(patient_id)
        except ValueError:
            message = [f'Error: Patient ID must be a valid number']
            return render_template('index.html', message=message, error=True)
        
        # Load full dataset with id column
        df_full = pd.read_csv('notebook and dataset/breast cancer.csv')
        
        # Find the row with matching ID
        patient_row = df_full[df_full['id'] == patient_id_int]
        
        if patient_row.empty:
            message = [f'Error: Patient ID {patient_id} not found in database']
            return render_template('index.html', message=message, error=True)
        
        # Extract ID and features
        id_val = patient_row['id'].values[0]
        features = patient_row.drop(['id', 'diagnosis'], axis=1, errors='ignore')
        if 'Unnamed: 32' in features.columns:
            features = features.drop('Unnamed: 32', axis=1)
        features = features.values.flatten()
        
        # Include ID as first feature (to match scaler training)
        all_features = np.concatenate([[id_val], features])
        
        if len(all_features) != 31:
            message = [f'Error: Expected 31 features, got {len(all_features)}']
            return render_template('index.html', message=message, error=True)
        
        np_features = np.asarray(all_features, dtype=np.float32)
        
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

