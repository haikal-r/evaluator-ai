import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path

class WorkoutRecommender:
    def __init__(self, df):
        """
        Initialize WorkoutRecommender with model directory and placeholders for components
        
        Args:
            model_dir (str): Directory to save/load models and encoders
        """
        self.model_dir = 'workout_model'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Placeholders for key components
        self.model = None
        self.le_experience = None
        self.le_equipment = None
        self.le_workout = None
        self.df = df
    
    def prepare_training_data(self):
        """
        Prepare training data by encoding categorical features
        
        Args:
            data (dict): Dictionary containing workout data
        
        Returns:
            tuple: Prepared features (X) and target (y)
        """
        # Encode categorical columns
        self.le_experience = LabelEncoder()
        self.df["Experience Level Encoded"] = self.le_experience.fit_transform(self.df["Experience Level"])
        
        self.le_equipment = LabelEncoder()
        self.df["Equipment Needed Encoded"] = self.le_equipment.fit_transform(self.df["Equipment Needed"])
        
        # Encode target (Workout Name)
        self.le_workout = LabelEncoder()
        y_encoded = self.le_workout.fit_transform(self.df["Workout Name"])
        
        # Define Features (X)
        X = self.df[["Experience Level Encoded", "Equipment Needed Encoded", "Calories Burned per 30 min"]]
        
        return X, y_encoded
    
    def train_model(self, n_estimators=100, test_size=0.2, random_state=42):
        """
        Train the RandomForestRegressor model
        
        Args:
            data (dict): Dictionary containing workout data
            n_estimators (int): Number of trees in the forest
            test_size (float): Proportion of data to use for testing
            random_state (int): Seed for reproducibility
        
        Returns:
            dict: Model performance metrics
        """
        # Prepare data
        X, y = self.prepare_training_data()
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        # Save model and encoders
        self.save_model()
        
        return {
            "mean_squared_error": mse,
            "n_estimators": n_estimators
        }
    
    def save_model(self):
        """
        Save trained model and label encoders to files
        """
        # Ensure model components are trained
        if not all([self.model, self.le_workout, self.le_experience, self.le_equipment]):
            raise ValueError("Model must be trained before saving")
        
        # Save paths
        model_path = os.path.join(self.model_dir, "workout_model.pkl")
        le_workout_path = os.path.join(self.model_dir, "label_encoder_workout.pkl")
        le_experience_path = os.path.join(self.model_dir, "label_encoder_experience.pkl")
        le_equipment_path = os.path.join(self.model_dir, "label_encoder_equipment.pkl")
        
        # Save components
        joblib.dump(self.model, model_path)
        joblib.dump(self.le_workout, le_workout_path)
        joblib.dump(self.le_experience, le_experience_path)
        joblib.dump(self.le_equipment, le_equipment_path)
    
    def load_model(self):
        """
        Load previously saved model and encoders
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Construct file paths
            model_path = os.path.join(self.model_dir, "workout_model.pkl")
            le_workout_path = os.path.join(self.model_dir, "label_encoder_workout.pkl")
            le_experience_path = os.path.join(self.model_dir, "label_encoder_experience.pkl")
            le_equipment_path = os.path.join(self.model_dir, "label_encoder_equipment.pkl")
            
            # Load components
            self.model = joblib.load(model_path)
            self.le_workout = joblib.load(le_workout_path)
            self.le_experience = joblib.load(le_experience_path)
            self.le_equipment = joblib.load(le_equipment_path)
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_workout(self, input_data):
        """
        Predict workout based on input features
        
        Args:
            input_data (dict): Input features for prediction
        
        Returns:
            dict: Predicted workout name and image
        """
        # Ensure model is loaded
        if not self.model:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        # Prepare input data
        input_df = pd.DataFrame({
            "Experience Level Encoded": [self.le_experience.transform([input_data['experience_level']])[0]],
            "Equipment Needed Encoded": [self.le_equipment.transform([input_data['equipment']])[0]],
            "Calories Burned per 30 min": [input_data['calories']]
        })
        
        # Predict the top 2 workouts
        preds = []
        for _ in range(2):
            pred = self.model.predict(input_df)
            noise = np.random.normal(0, 0.5)  # Tambahkan sedikit noise
            pred_with_noise = pred[0] + noise
            preds.append(int(round(pred_with_noise)))
        
        # Pastikan workouts berbeda
        preds = list(set(preds))
        
        # Jika masih kurang dari 2, ambil workouts terdekat
        while len(preds) < 2:
            preds.append(preds[0] + 1)
        
        workout_results = []
        
        for predicted_workout_idx in preds[:2]:
            predicted_workout_name = self.le_workout.inverse_transform([predicted_workout_idx])[0]
            
            predicted_row = self.df[self.df['Workout Name'] == predicted_workout_name]
            image_path = predicted_row['Image'].values[0]
            
            workout_results.append({
                "workout_name": predicted_workout_name,
                "image": image_path
            })
        
        return {
            "workouts": workout_results
        }
        
# Sample workout data
    
file_path = Path(__file__).parent / 'datasets' / 'workout_data.csv'
df = pd.read_csv(file_path)

# Create recommender
recommender = WorkoutRecommender(df)

# Train the model
# print("Training Model:")
training_results = recommender.train_model()
# print(f"Model Training Results: {training_results}")

# Predict a workout
# print("\nPredicting Workout:")
# new_workout_input = {
#     "experience_level": "Beginner",
#     "equipment": "Yes",
#     "calories": 120
# }

# result = recommender.predict_workout(new_workout_input)
# print("Predicted Workout:", result)