import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from pathlib import Path 

# Sample Dataset
# data = {
#     "Workout Name": ["Push-Up", "Jumping Jacks", "Squats", "Plank", "Burpees"],
#     "Experience Level": ["beginner", "beginner", "intermediate", "beginner", "advanced"],
#     "Equipment Needed": ["no_equipment", "no_equipment", "no_equipment", "no_equipment", "no_equipment"],
#     "Calories Burned per 30min": [200, 250, 300, 100, 400],
#     "Images": [
#         "push_up_image.jpg",
#         "jumping_jacks_image.jpg",
#         "squats_image.jpg",
#         "plank_image.jpg",
#         "burpees_image.jpg",
#     ],
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

file_path = Path(__file__).parent / 'datasets' / 'workout_data.csv'
df = pd.read_csv(file_path)

# Encode Categorical Data
le_experience = LabelEncoder()
le_equipment = LabelEncoder()

df["Experience Level"] = le_experience.fit_transform(df["Experience Level"])
df["Equipment Needed"] = le_equipment.fit_transform(df["Equipment Needed"])

# Features and Target
X = df[["Experience Level","Equipment Needed","Calories Burned per 30 min"]]
y = df["Workout Name"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RandomForest Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# User Input
user_input = {
    "experience_level": "Beginner",
    "equipment_needed": "No",  # Ignored in model but can be used for additional rules
    "training_duration": "20_minute",  # Could map to calories (e.g., proportion of 30 minutes)
    # "accessibility": "no_equipment",
}

# Preprocess User Input
input_data = [
    le_experience.transform([user_input["experience_level"]])[0],
    le_equipment.transform([user_input["equipment_needed"]])[0],
    200,  # Assume 20 minutes burns 2/3 of 30-min workout calories
]

# print('input data =>', input_data)

# Prediction
predicted_workout = rf.predict([input_data])[0]

# Find Image
workout_image = df.loc[df["Workout Name"] == predicted_workout, "Image"].values[0]

# Output Result
print(f"Recommended Workout: {predicted_workout}")
print(f"Image: {workout_image}")

# Calculate Accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
