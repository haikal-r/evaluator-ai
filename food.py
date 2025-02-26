from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import joblib
import numpy as np
import os
import sys


class NutritionModel:
    def __init__(self, df):
        self.df = df
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        
        # Fitur yang akan digunakan
        self.df['Calories'] = self.df['Calories'].astype('float64')
        self.features_to_scale = ['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']
        self.df['meal_type'] = df['Food Group'].map({
            'Dairy and Egg Products': 'breakfast',
            'Vegetables': 'breakfast',  
            'Baked Foods': 'lunch',
            'Vegetables': 'lunch',  
            'Fish': 'lunch',
            'Meats': 'lunch',
            'Fruits': 'dinner',
            'Vegetables': 'dinner',  
        })
    
    def train_model(self):
        # Pisahkan fitur dan target
        X = self.df[self.features_to_scale]
        y = self.df['name']  # Menggunakan nama sebagai target
        
        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Encode target (name) menjadi numerik
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Melatih model
        self.model.fit(X_scaled, y_encoded)
    
    def calculate_meal_distribution(self, user_input):
        """
        Distribusi kebutuhan gizi berdasarkan input user ke dalam kategori meal.
        """
        distribution = {
            "breakfast": {key: value * 0.3 for key, value in user_input.items()},
            "lunch": {key: value * 0.4 for key, value in user_input.items()},
            "dinner": {key: value * 0.3 for key, value in user_input.items()},
        }
        return distribution

    def find_best_match(self, target_nutrients, meal_type, restriction=None):
        """
        Temukan makanan yang paling mendekati kebutuhan gizi untuk kategori meal tertentu.
        """
        # Filter dataset berdasarkan meal_type
        # meal_df = self.df[self.df['meal_type'] == meal_type]
        
        # # Exclude makanan sesuai dengan restriction
        # if restriction == 'vegetarian':
        #     meal_df = meal_df[~meal_df['Food Group'].isin(['Meats', 'Fish'])]

        # # Hitung jarak Euclidean untuk setiap makanan
        # distances = meal_df.apply(
        #     lambda row: np.sqrt(
        #         sum(
        #             (row[feature] - target_nutrients[feature]) ** 2 
        #             for feature in self.features_to_scale
        #         )
        #     ),
        #     axis=1
        # )
        
        # # Pilih makanan dengan jarak terpendek
        # best_match_idx = distances.nsmallest(2).index
        # best_match = meal_df.loc[best_match_idx]
        # return {
        #     "name": best_match['name'],
        #     "image": best_match['Images'],  # Tambahkan gambar
        #     "nutrition": {feature: best_match[feature] for feature in self.features_to_scale}
        # }
        
        
        def calculate_distance(row, target):
            return np.sqrt(
                sum(
                    (row[feature] - target[feature]) ** 2 
                    for feature in self.features_to_scale
                )
            )
        
        
        # Bagi target kalori menjadi tiga bagian untuk breakfast, lunch, dan dinner
        meals = {}
        
        sub_meal_df = self.df[self.df['meal_type'] == meal_type]

        if restriction == 'vegetarian':
            sub_meal_df = sub_meal_df[~sub_meal_df['Food Group'].isin(['Meats', 'Fish'])]

        # Hitung jarak Euclidean untuk setiap makanan
        distances = sub_meal_df.apply(lambda row: calculate_distance(row, {**target_nutrients, 'calories': target_nutrients['Calories']}), axis=1)

         # Pilih 2 makanan dengan jarak terpendek
        best_matches_idx = distances.nsmallest(2).index
        best_matches = sub_meal_df.loc[best_matches_idx]

        meals[meal_type] = [
            {
                "name": row['name'],
                "image": row['Images'],
                "serving_per_gram": row['Serving Weight 1 (g)'],
                "serving_description": row['Serving Description 1 (g)'],
                "nutrition": {feature: row[feature] for feature in self.features_to_scale}
            }
            for _, row in best_matches.iterrows()
        ]

        return meals

    def generate_recommendation(self, user_input, restriction=None):
        """
        Rekomendasi makanan berdasarkan input user.
        """
        # Hitung distribusi kebutuhan gizi
        meal_distribution = self.calculate_meal_distribution(user_input)
        
        recommendations = [
            {
                "type": meal_type,
                **self.find_best_match(target_nutrients, meal_type, restriction)
            }
            for meal_type, target_nutrients in meal_distribution.items()
        ]
        
        return recommendations


# Path dataset
file_path = Path(__file__).parent / 'datasets' / 'food_data.csv'
df = pd.read_csv(file_path)


# # Input user
user_input = {
    'Calories': 1500,
    'Fat (g)': 20,
    'Protein (g)': 20,
    'Carbohydrate (g)': 30,
}
restriction = 'vegetarian'  # Tambahkan restriction


# Train and export model
nutrition_model = NutritionModel(df)
nutrition_model.train_model()

# recommendation = nutrition_model.generate_recommendation(user_input, restriction)
# print("Recommendation:", recommendation)

# export model
model_dir = Path("food_model")
model_path = model_dir / "nutrition_model.pkl"

os.makedirs(model_dir, exist_ok=True)

# Simpan model
joblib.dump(nutrition_model, model_path)
# print(f"Model saved to {model_path}")

# Gunakan model yang disimpan
# model_path = Path("nutrition_model.pkl")
# loaded_model = joblib.load(model_path)
# recommendation = loaded_model.generate_recommendation(user_input, restriction)
# print("Recommendation:", recommendation)
