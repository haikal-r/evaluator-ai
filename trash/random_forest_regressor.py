import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Load dataset
file_path = Path(__file__).parent / 'datasets' / 'nutrition_table.csv'
df = pd.read_csv(file_path)

# Preprocessing
df = df[['name', 'Food Group', 'Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']].dropna()

# Normalisasi data menggunakan MinMaxScaler
scaler = MinMaxScaler()
df[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']] = scaler.fit_transform(
    df[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]
)

# Pisahkan data untuk setiap kategori makanan (Breakfast, Lunch, Dinner)
df_breakfast = df[df['Food Group'] == 'Dairy and Egg Products']
df_lunch = df[df['Food Group'] == 'Meats']
df_dinner = df[df['Food Group'] == 'Fruits']

# Fungsi untuk melatih model RandomForest untuk setiap kategori makanan
def train_model_for_category(category_df):
    X = category_df[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]
    y = category_df.index  # Gunakan indeks untuk mengacu ke nama makanan
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# Melatih model untuk setiap kategori
model_breakfast = train_model_for_category(df_breakfast)
model_lunch = train_model_for_category(df_lunch)
model_dinner = train_model_for_category(df_dinner)

# User's BMI-based needs (contoh kebutuhan kalori, lemak, protein, karbohidrat)
user_features = np.array([[0.4, 0.1, 0.2, 0.3]])  # Nilai scaled dari kebutuhan user

# Fungsi untuk memberikan rekomendasi makanan untuk kategori tertentu
def recommend_food(model, category_df, user_features):
    predicted_index = int(model.predict(user_features)[0])
    recommended_food = category_df.iloc[predicted_index]['name']
    return recommended_food

# Rekomendasi makanan untuk setiap kategori
recommended_breakfast = recommend_food(model_breakfast, df_breakfast, user_features)
recommended_lunch = recommend_food(model_lunch, df_lunch, user_features)
recommended_dinner = recommend_food(model_dinner, df_dinner, user_features)

# Output rekomendasi
print("Rekomendasi makanan untuk Breakfast:", recommended_breakfast)
print("Rekomendasi makanan untuk Lunch:", recommended_lunch)
print("Rekomendasi makanan untuk Dinner:", recommended_dinner)