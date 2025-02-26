import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class NutritionWorkoutRecommender:
    def __init__(self, dataset_path):
        # Membaca dataset
        self.df = pd.read_csv(dataset_path)
        
        # Encoding kolom kategorikal
        self.label_encoders = {}
        categorical_features = ["activity_level", "training_duration", "accessibility", "workout_plan", "food_name"]
        for feature in categorical_features:
            le = LabelEncoder()
            self.df[feature] = le.fit_transform(self.df[feature])
            self.label_encoders[feature] = le
        
        # Mempersiapkan fitur dan label
        X = self.df[["BMI", "activity_level", "training_duration", "accessibility"]]
        y_nutrition = self.df[["Calories", "Protein", "Fat", "Carbohydrate"]]
        y_workout = self.df["workout_plan"]
        y_food = self.df["food_name"]
        
        # Membagi data
        X_train_n, X_test_n, y_nutrition_train, y_nutrition_test = train_test_split(X, y_nutrition, test_size=0.2, random_state=42)
        X_train_w, X_test_w, y_workout_train, y_workout_test = train_test_split(X, y_workout, test_size=0.2, random_state=42)
        X_train_f, X_test_f, y_food_train, y_food_test = train_test_split(X, y_food, test_size=0.2, random_state=42)
        
        # Melatih model
        self.nutrition_model = RandomForestRegressor(random_state=42)
        self.nutrition_model.fit(X_train_n, y_nutrition_train)
        
        self.workout_model = RandomForestClassifier(random_state=42)
        self.workout_model.fit(X_train_w, y_workout_train)
        
        self.food_model = RandomForestClassifier(random_state=42)
        self.food_model.fit(X_train_f, y_food_train)
        
        # Menyimpan dataset asli untuk referensi nama
        self.original_df = pd.read_csv(dataset_path)
    
    def recommend_nutrition_and_workout(self, bmi, activity_level, training_duration, accessibility):
        """
        Memberikan rekomendasi nutrisi dan workout berdasarkan input pengguna
        
        Parameters:
        - bmi: Indeks Massa Tubuh pengguna
        - activity_level: Level aktivitas (akan di-encode)
        - training_duration: Durasi latihan (akan di-encode)
        - accessibility: Aksesibilitas (akan di-encode)
        
        Returns:
        Dictionary berisi rekomendasi makanan, nutrisi, dan workout
        """
        # Encode input kategorikal
        activity_level_encoded = self.label_encoders['activity_level'].transform([activity_level])[0]
        training_duration_encoded = self.label_encoders['training_duration'].transform([training_duration])[0]
        accessibility_encoded = self.label_encoders['accessibility'].transform([accessibility])[0]
        
        # Persiapkan input untuk prediksi
        input_data = np.array([[bmi, activity_level_encoded, training_duration_encoded, accessibility_encoded]])
        
        # Prediksi nutrisi
        nutrition_pred = self.nutrition_model.predict(input_data)[0]
        
        # Prediksi workout
        workout_pred = self.workout_model.predict(input_data)[0]
        workout_name = self.label_encoders['workout_plan'].inverse_transform([workout_pred])[0]
        
        # Prediksi makanan
        food_pred = self.food_model.predict(input_data)[0]
        food_name = self.label_encoders['food_name'].inverse_transform([food_pred])[0]
        
        # Temukan baris makanan yang sesuai untuk detail nutrisi
        food_details = self.original_df[self.original_df['food_name'] == food_name].iloc[0]
        
        return {
            'food_name': food_name,
            'nutrition': {
                'calories': round(nutrition_pred[0], 2),
                'protein': round(nutrition_pred[1], 2),
                'fat': round(nutrition_pred[2], 2),
                'carbohydrate': round(nutrition_pred[3], 2)
            },
            'workout_plan': workout_name,
            'additional_food_info': {
                'serving_size': food_details['serving_size'],
                'food_category': food_details['food_category']
            }
        }
    
    def evaluate_models(self):
        """
        Mengevaluasi performa model
        """
        from sklearn.metrics import mean_squared_error, accuracy_score
        
        # Evaluasi model nutrisi
        nutrition_pred = self.nutrition_model.predict(X_test_n)
        nutrition_mse = mean_squared_error(y_nutrition_test, nutrition_pred)
        
        # Evaluasi model workout
        workout_pred = self.workout_model.predict(X_test_w)
        workout_accuracy = accuracy_score(y_workout_test, workout_pred)
        
        return {
            'nutrition_model_mse': nutrition_mse,
            'workout_model_accuracy': workout_accuracy
        }

# Contoh penggunaan
if __name__ == "__main__":
    recommender = NutritionWorkoutRecommender('nutrition_workout_dataset.csv')
    
    # Contoh rekomendasi untuk seseorang
    recommendation = recommender.recommend_nutrition_and_workout(
        bmi=25.5, 
        activity_level='moderate', 
        training_duration='30-60 menit', 
        accessibility='gym'
    )
    
    print("Rekomendasi Makanan:")
    print(f"Nama Makanan: {recommendation['food_name']}")
    print("\nNilai Nutrisi:")
    for nutrient, value in recommendation['nutrition'].items():
        print(f"{nutrient.capitalize()}: {value}")
    
    print(f"\nRencana Workout: {recommendation['workout_plan']}")
    print(f"Kategori Makanan: {recommendation['additional_food_info']['food_category']}")
    print(f"Ukuran Porsi: {recommendation['additional_food_info']['serving_size']}")
    
    # Evaluasi model
    model_performance = recommender.evaluate_models()
    print("\nPerforma Model:")
    print(f"MSE Model Nutrisi: {model_performance['nutrition_model_mse']}")
    print(f"Akurasi Model Workout: {model_performance['workout_model_accuracy']}")