import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
from pathlib import Path

# Load dataset
file_path = Path(__file__).parent / 'datasets' / 'nutrition_table.csv'
data = pd.read_csv(file_path)

# Preprocessing
scaler = MinMaxScaler()
label_encoder = LabelEncoder()

data[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']] = scaler.fit_transform(
    data[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]
)

data['Food Group'] = label_encoder.fit_transform(data['Food Group'])

# Training RandomForest
X = data[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]
y = data['Food Group']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# BMI Calculation and Meal Recommendation
def calculate_bmi(weight, height):
    return weight / (height ** 2)

def classify_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'

def recommend_meals(bmi_category, user_features):
    filtered_data = data.copy()

    # Filter data based on BMI needs
    if bmi_category == 'Underweight':
        filtered_data = filtered_data[filtered_data['Calories'] > 0.7]  # High calorie foods
    elif bmi_category == 'Overweight' or bmi_category == 'Obese':
        filtered_data = filtered_data[filtered_data['Calories'] < 0.3]  # Low calorie foods
    
    if filtered_data.empty:
        return "No suitable foods found."

    # Predict based on user input
    prediction = model.predict(user_features)
    predicted_food_group = label_encoder.inverse_transform(prediction)
    print(predicted_food_group)
    
    # Get example food from predicted group
    example_food = filtered_data[filtered_data['Food Group'] == prediction[0]].sample(n=1)

    return example_food[['name', 'Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]

# User Input
weight = 75  # kg
height = 1.75  # meters
bmi = calculate_bmi(weight, height)
bmi_category = classify_bmi(bmi)

# User feature input
user_features = np.array([[0.4, 0.1, 0.2, 0.3]])  # Example scaled input

# Generate Recommendation
recommendation = recommend_meals(bmi_category, user_features)
print(f"Based on your BMI ({bmi:.2f}, {bmi_category}), we recommend:\n", recommendation)
