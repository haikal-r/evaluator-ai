import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# Load dataset
file_path = Path(__file__).parent / 'datasets' / 'nutrition_table.csv'
df = pd.read_csv(file_path)

# Preprocessing
df = df[['name', 'Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']].dropna()
scaler = MinMaxScaler()
df[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']] = scaler.fit_transform(
    df[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]
)

# Training RandomForest Regressor
X = df[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]
y = df.index  # Gunakan indeks untuk mengacu ke nama makanan
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# User's BMI-based needs (contoh kebutuhan kalori, lemak, protein, karbohidrat)
user_features = np.array([[0.4, 0.1, 0.2, 0.3]])  # Nilai scaled dari kebutuhan user

# Predict makanan terbaik
predicted_index = int(model.predict(user_features)[0])
recommended_food = df.iloc[predicted_index]['name']

print("Rekomendasi makanan berdasarkan kebutuhan Anda:", recommended_food)
