import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

class NutritionWorkoutRecommender:
    def __init__(self, nutrition_dataset_path, workout_dataset_path):
        # Membaca dataset
        self.nutrition_df = pd.read_csv(nutrition_dataset_path)
        self.workout_df = pd.read_csv(workout_dataset_path)
        
        # Encoding kolom kategorikal untuk dataset nutrisi
        self.nutrition_label_encoders = {}
        nutrition_categorical_features = ["Food Group"]
        for feature in nutrition_categorical_features:
            le = LabelEncoder()
            self.nutrition_df[feature] = le.fit_transform(self.nutrition_df[feature])
            self.nutrition_label_encoders[feature] = le
        
        # Encoding kolom kategorikal untuk dataset workout
        self.workout_label_encoders = {}
        workout_categorical_features = ["activity_level", "equipment_needed"]
        for feature in workout_categorical_features:
            le = LabelEncoder()
            self.workout_df[feature] = le.fit_transform(self.workout_df[feature])
            self.workout_label_encoders[feature] = le
        
        # Mempersiapkan fitur dan label untuk nutrisi
        X_nutrition = self.nutrition_df[["Calories", "Protein (g)", "Fat (g)", "Carbohydrate (g)"]]
        y_food = self.nutrition_df["name"]
        
        # Mempersiapkan fitur dan label untuk workout
        X_workout = self.workout_df[["activity_level"]]
        y_workout_plan = self.workout_df["workout_name"]
        
        # Membagi data
        X_train_food, self.X_test_food, y_food_train, self.y_food_test = train_test_split(X_nutrition, y_food, test_size=0.2, random_state=42)
        X_train_workout, self.X_test_workout, y_workout_train, self.y_workout_test = train_test_split(X_workout, y_workout_plan, test_size=0.2, random_state=42)
        
        # Melatih model
        self.food_model = RandomForestClassifier(random_state=42)
        self.food_model.fit(X_train_food, y_food_train)
        
        self.workout_model = RandomForestClassifier(random_state=42)
        self.workout_model.fit(X_train_workout, y_workout_train)
        
        # Menyimpan dataset asli untuk referensi
        self.original_nutrition_df = pd.read_csv(nutrition_dataset_path)
        self.original_workout_df = pd.read_csv(workout_dataset_path)
    
    def recommend_nutrition_and_workout(self, calories, protein, fat, carbohydrate, difficulty_level, duration, intensity):
        """
        Memberikan rekomendasi nutrisi dan workout berdasarkan input pengguna
        
        Parameters:
        Untuk nutrisi:
        - calories: Total kalori
        - protein: Jumlah protein
        - fat: Jumlah lemak
        - carbohydrate: Jumlah karbohidrat
        
        Untuk workout:
        - difficulty_level: Level kesulitan (akan di-encode)
        - duration: Durasi latihan (akan di-encode)
        - intensity: Intensitas latihan (akan di-encode)
        
        Returns:
        Dictionary berisi rekomendasi makanan dan workout
        """
        # Persiapkan input untuk prediksi makanan
        nutrition_input = np.array([[calories, protein, fat, carbohydrate]])
        
        # Prediksi makanan
        food_pred = self.food_model.predict(nutrition_input)[0]
        food_name = self.nutrition_label_encoders['food_category'].inverse_transform([food_pred])[0]
        
        # Persiapkan input untuk prediksi workout
        workout_input = np.array([[difficulty_level, duration, intensity]])
        
        # Prediksi workout
        workout_pred = self.workout_model.predict(workout_input)[0]
        workout_name = workout_pred
        
        # Temukan detail makanan
        food_details = self.original_nutrition_df[self.original_nutrition_df['food_name'] == food_name].iloc[0]
        
        # Temukan detail workout
        workout_details = self.original_workout_df[self.original_workout_df['workout_name'] == workout_name].iloc[0]
        
        return {
            'food_recommendation': {
                'food_name': food_name,
                'food_category': food_details['food_category'],
                'serving_size': food_details['serving_size'],
                'nutritional_details': {
                    'calories': food_details['calories'],
                    'protein': food_details['protein'],
                    'fat': food_details['fat'],
                    'carbohydrate': food_details['carbohydrate']
                }
            },
            'workout_recommendation': {
                'workout_name': workout_name,
                'difficulty_level': workout_details['difficulty_level'],
                'duration': workout_details['duration'],
                'intensity': workout_details['intensity'],
                'equipment_needed': workout_details['equipment_needed']
            }
        }
    
    def evaluate_models(self):
        """
        Mengevaluasi performa model
        """
        from sklearn.metrics import accuracy_score
        
        # Evaluasi model makanan
        food_pred = self.food_model.predict(self.X_test_food)
        food_accuracy = accuracy_score(self.y_food_test, food_pred)
        
        # Evaluasi model workout
        workout_pred = self.workout_model.predict(self.X_test_workout)
        workout_accuracy = accuracy_score(self.y_workout_test, workout_pred)
        
        return {
            'food_model_accuracy': food_accuracy,
            'workout_model_accuracy': workout_accuracy
        }

# Contoh penggunaan
if __name__ == "__main__":
    dataset_food = Path(__file__).parent / 'datasets' / 'nutrition_table.csv'
    dataset_workout = Path(__file__).parent / 'datasets' / 'workout_data.csv'
    print(dataset_workout)
    recommender = NutritionWorkoutRecommender(
        nutrition_dataset_path = dataset_food, 
        workout_dataset_path = dataset_workout
    )
    
    # Contoh rekomendasi 
    recommendation = recommender.recommend_nutrition_and_workout(
        calories=500, 
        protein=30, 
        fat=15, 
        carbohydrate=50,
        difficulty_level=2,  # Sesuaikan dengan encoding di dataset Anda
        duration=3,          # Sesuaikan dengan encoding di dataset Anda
        intensity=2          # Sesuaikan dengan encoding di dataset Anda
    )
    
    print("Rekomendasi Makanan:")
    print(f"Nama Makanan: {recommendation['food_recommendation']['food_name']}")
    print(f"Kategori Makanan: {recommendation['food_recommendation']['food_category']}")
    print(f"Ukuran Porsi: {recommendation['food_recommendation']['serving_size']}")
    
    print("\nDetail Nutrisi:")
    for nutrient, value in recommendation['food_recommendation']['nutritional_details'].items():
        print(f"{nutrient.capitalize()}: {value}")
    
    print("\nRekomendasi Workout:")
    print(f"Nama Workout: {recommendation['workout_recommendation']['workout_name']}")
    print(f"Level Kesulitan: {recommendation['workout_recommendation']['difficulty_level']}")
    print(f"Durasi: {recommendation['workout_recommendation']['duration']}")
    print(f"Intensitas: {recommendation['workout_recommendation']['intensity']}")
    print(f"Peralatan yang Dibutuhkan: {recommendation['workout_recommendation']['equipment_needed']}")
    
    # Evaluasi model
    model_performance = recommender.evaluate_models()
    print("\nPerforma Model:")
    print(f"Akurasi Model Makanan: {model_performance['food_model_accuracy']}")
    print(f"Akurasi Model Workout: {model_performance['workout_model_accuracy']}")