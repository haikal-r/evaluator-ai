import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Data Dummy untuk Demonstrasi
data = {
    "Workout Name": ["Push Ups", "Jump Rope", "Burpees", "Squats", "Plank"],
    "Experience Level": ["Beginner", "Intermediate", "Advanced", "Beginner", "Advanced"],
    "Equipment Needed": ["None", "Rope", "None", "None", "Mat"],
    "Calories Burned per 30 min": [100, 300, 250, 150, 200],
    "Image": ["pushups.jpg", "jumprope.jpg", "burpees.jpg", "squats.jpg", "plank.jpg"],
}

# Load dataset into a DataFrame
df = pd.DataFrame(data)

# Encode categorical columns
le_experience = LabelEncoder()
df["Experience Level Encoded"] = le_experience.fit_transform(df["Experience Level"])

le_equipment = LabelEncoder()
df["Equipment Needed Encoded"] = le_equipment.fit_transform(df["Equipment Needed"])

# Define Features (X) and Target (y)
X = df[["Experience Level Encoded", "Equipment Needed Encoded", "Calories Burned per 30 min"]]
y = df["Workout Name"]

# Encode Target (Workout Name)
le_workout = LabelEncoder()
y_encoded = le_workout.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model and encoders
model_dir = "workout_model"
os.makedirs(model_dir, exist_ok=True)  # Buat folder jika belum ada

# Simpan model dan encoder ke dalam folder workout_model
model_path = os.path.join(model_dir, "workout_model.pkl")
le_workout_path = os.path.join(model_dir, "label_encoder_workout.pkl")

joblib.dump(model, model_path)
joblib.dump(le_workout, le_workout_path)
print("Model and encoders saved successfully.")

# Predict Workout Name and Image for new data
def predict_workout(new_input):
    """
    Predict workout name and return associated image.
    """
    # Predict the workout
    pred = model.predict(new_input)
    predicted_workout_idx = int(round(pred[0]))  
    predicted_workout_name = le_workout.inverse_transform([predicted_workout_idx])[0]
    
    # Retrieve image associated with the workout name
    workout_row = df[df["Workout Name"] == predicted_workout_name].iloc[0]
    return {
        "Workout Name": workout_row["Workout Name"],
        "Image": workout_row["Image"],
    }

# Input baru
new_data = pd.DataFrame({
    "Experience Level Encoded": [le_experience.transform(["Beginner"])[0]],
    "Equipment Needed Encoded": [le_equipment.transform(["None"])[0]],
    "Calories Burned per 30 min": [120]
})

result = predict_workout(new_data)
print("Predicted Workout and Image:", result)
