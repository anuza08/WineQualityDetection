import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset to a Pandas DataFrame
wine_dataset = pd.read_csv(r"C:/Users/ANUJA/Downloads/winequality-red.csv")

# Define a function to map the quality scores to the new labels
def map_quality(value):
    if value == 5:
        return 'bad'
    elif value == 6:
        return 'moderate'
    elif value == 4:
        return 'worst'
    elif value == 7:
        return 'best quality'
    else:
        return 'other'

# Apply the function to the 'quality' column
wine_dataset['quality_label'] = wine_dataset['quality'].apply(map_quality)

# Filter out rows with 'other' quality to focus on the specified labels
wine_dataset = wine_dataset[wine_dataset['quality_label'] != 'other']

# Separate the data and labels
X = wine_dataset.drop(['quality', 'quality_label'], axis=1)
Y = wine_dataset['quality_label']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Save the model to disk
filename = 'wine_quality_model.pkl'
pickle.dump(model, open(filename, 'wb'))

# Load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

# Now you can use this loaded_model for making predictions
# For example:
input_data = [[7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5]]
prediction = loaded_model.predict(input_data)

# Print the prediction result
print(f'Predicted wine quality: {prediction[0]}')
