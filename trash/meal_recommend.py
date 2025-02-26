import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import scipy.spatial.distance as distance
from pathlib import Path
from sklearn.metrics.pairwise import euclidean_distances


class MealRecommenderProduction:
    def __init__(self, model_path=None):
        """
        Inisialisasi recommender untuk production
        
        Parameters:
        - model_path: Path ke model .pkl yang sudah dilatih
        """
        self.scaler = None
        self.model = None
        self.df = None
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_and_train(self, df, output_path='meal_recommender_model.pkl'):
        """
        Persiapan, pelatihan, dan penyimpanan model
        
        Parameters:
        - df: DataFrame berisi informasi makanan
        - output_path: Path untuk menyimpan model
        """
        # Preprocessing data
        self.df = df[['name', 'Food Group', 'Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]
        self.df['Meal Type'] = df['Food Group'].map({
            'Dairy and Egg Products': 'Breakfast',
            'Meats': 'Lunch',
            'Fruits': 'Dinner'
        })

        # Sekarang dropna() untuk memastikan tidak ada baris yang memiliki nilai NaN
        self.df = self.df.dropna(subset=['Food Group', 'Meal Type'])
        
        # Inisialisasi dan fitting scaler
        self.scaler = MinMaxScaler()
        features_to_scale = ['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']
        
        # Scale fitur
        scaled_features = self.scaler.fit_transform(self.df[features_to_scale])
        X = pd.DataFrame(scaled_features, columns=features_to_scale)
        
        # Gunakan rata-rata sebagai default goals
        user_nutritional_goals = X.mean().values
        
        # Hitung jarak dari goals
        y = [distance.euclidean(row, user_nutritional_goals) for _, row in X.iterrows()]
        # y = self.df['name']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Latih model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Simpan model lengkap dengan scaler dan dataframe
        with open(output_path, 'wb') as file:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'df': self.df
            }, file)
        
        print(f"Model tersimpan di {output_path}")
        
    def load_model(self, model_path):
        """
        Memuat model yang sudah disimpan
        
        Parameters:
        - model_path: Path ke file model .pkl
        """
        with open(model_path, 'rb') as file:
            saved_data = pickle.load(file)
        
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.df = saved_data['df']
    
    def recommend_meals(self, user_features):
        """
        Merekomendasikan makanan dari model yang sudah dilatih
        
        Parameters:
        - user_features: Kebutuhan nutrisi pengguna
        
        Returns:
        - Dictionary rekomendasi makanan
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model belum dimuat. Gunakan load_model() terlebih dahulu.")
        
        # Scale user features
        scaled_user_features = self.scaler.transform(user_features)
        
        recommendations = {}
        meal_types = ['Breakfast', 'Lunch', 'Dinner']
        
        for meal_type in meal_types:
            # Filter makanan sesuai tipe meal
            meal_df = self.df[self.df['Meal Type'] == meal_type]
            
            if not meal_df.empty:
                # Ambil fitur yang sudah di-scale
                features_to_scale = ['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']
                X_meal = self.scaler.transform(meal_df[features_to_scale])

                scores = self.model.predict(X_meal)
                best_meal_index = np.argmin(np.abs(scores))
                print("SCORES =>", scores)
                print("USER =>", scaled_user_features)
                # print("HASIL =>", np.argmin(np.abs(scores)))
                # recommended_meal = meal_df.iloc[best_meal_index]

                # distances = np.linalg.norm(X_meal - scaled_user_features, axis=1)
            
                # # Pilih makanan dengan jarak terdekat
                # best_meal_index = np.argmin(distances)
                print('best =>', best_meal_index)
                recommended_meal = meal_df.iloc[best_meal_index]
                print(meal_df)

           
                # Simpan rekomendasi dengan detail nutrisi
                recommendations[meal_type] = {
                    'name': recommended_meal['name'],
                    'Calories': recommended_meal['Calories'],
                    'Fat (g)': recommended_meal['Fat (g)'],
                    'Protein (g)': recommended_meal['Protein (g)'],
                    'Carbohydrate (g)': recommended_meal['Carbohydrate (g)']
                }
        
        return recommendations

# Contoh penggunaan untuk training
def train_and_save_model():
    # Buat DataFrame contoh (ganti dengan data sebenarnya)
    file_path = Path(__file__).parent / 'datasets' / 'nutrition_table.csv'
    df = pd.read_csv(file_path)
    
    # Inisialisasi dan latih model
    recommender = MealRecommenderProduction()
    recommender.prepare_and_train(df, output_path='meal_recommender_model.pkl')

# Contoh penggunaan untuk inference/production
def use_saved_model():
    # Muat model yang sudah disimpan
    recommender = MealRecommenderProduction('meal_recommender_model.pkl')
    
    # Contoh fitur pengguna (sudah di-scale)
    user_features = np.array([[0.9, 0.9, 0.102, 0.2]])  # [Calories, Fat, Protein, Carbs]


    
    # Dapatkan rekomendasi
    recommendations = recommender.recommend_meals(user_features)
    
    # Tampilkan rekomendasi
    for meal_type, food in recommendations.items():
        print(f"{meal_type} Recommendation: {food}")

# Jalankan training
if __name__ == "__main__":
    # Pilih salah satu:
    # train_and_save_model()
    # atau
    use_saved_model()