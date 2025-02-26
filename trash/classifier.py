import pandas as pd
from pathlib import Path

# Load dataset
file_path = Path(__file__).parent / 'datasets' / 'nutrition_table.csv'
df = pd.read_csv(file_path)

# Menampilkan beberapa baris pertama dataset
print(df.head())

# Memeriksa apakah ada nilai yang hilang
print(df.isnull().sum())
# Menentukan fitur dan target
X = df[['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']]
y = df['name']
from sklearn.model_selection import train_test_split

# Membagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Membuat model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Melatih model
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

# Membuat prediksi pada data pengujian
y_pred = model.predict(X_test)

# Menghitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f'Akurasi model: {accuracy}')
# Fungsi untuk prediksi
def predict_food(user_input):
    prediction = model.predict([[
        user_input['Calories'],
        user_input['Fat (g)'],
        user_input['Protein (g)'],
        user_input['Carbohydrate (g)']
    ]])
    return prediction[0]

# Contoh input pengguna
user_input = {
    'Calories': 150,
    'Fat (g)': 2.0,
    'Protein (g)': 5.0,
    'Carbohydrate (g)': 20.0
}

# Prediksi makanan
food_result = predict_food(user_input)
print(f'Makanan yang cocok: {food_result}')
