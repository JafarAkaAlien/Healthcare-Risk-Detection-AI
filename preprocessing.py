import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import StandardScaler

# Load dataset

data = pd.read_csv('test_dataset.csv')
features = data[['Age', 'Blood Pressure']]
target = data['Heart Attack Risk']

x1 = features['Blood Pressure']
x2 = x1
x0 = features['Age']
for it in range(len(x1)):
    numerator, denominator = x1[it].split('/')

    # Convert to integers
    numerator = int(numerator)
    denominator = int(denominator)
    x1[it] = numerator/denominator
    # Output
    #print(x1[it])

# Normalize features
features['Blood Pressure'] = x1
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)