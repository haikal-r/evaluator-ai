import pandas as pd

# Dataset makanan
food_data = pd.DataFrame({
    "Food Name": ["Scrambled Eggs", "Grilled Chicken", "Avocado Salad", "Oatmeal", "Fruit Salad", "Grilled Fish"],
    "Calories": [200, 300, 250, 150, 100, 400],
    "Protein (g)": [14, 30, 3, 5, 1, 25],
    "Fat (g)": [15, 5, 20, 3, 0.5, 15],
    "Carbohydrates (g)": [1, 0, 10, 27, 25, 2]
})

# Input pengguna
user_input = {
    "weight": 50,
    "height": 175,
    "bmi": 16.33,
    "activity_level": "not_active",
    "training_duration": 20,  # dalam menit
}

# Menghitung TDEE (contoh sederhana)
def calculate_tdee(weight, height, age, gender, activity_level):
    bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * 19)  # Asumsi umur = 19 tahun
    activity_multiplier = {
        "not_active": 1.2,
        "moderately_active": 1.55,
        "very_active": 1.9
    }
    return bmr * activity_multiplier.get(activity_level, 1.2)

# Hitung kebutuhan kalori harian
tdee = calculate_tdee(user_input["weight"], user_input["height"], 19, "male", user_input["activity_level"])

# Distribusi kalori per waktu makan
calories_distribution = {
    "breakfast": tdee * 0.25,
    "lunch": tdee * 0.40,
    "dinner": tdee * 0.35
}

# Fungsi untuk memilih makanan berdasarkan kebutuhan kalori
def recommend_food(data, calorie_limit):
    return data[data["Calories"] <= calorie_limit].sort_values(by="Calories", ascending=False).head(1)

# Rekomendasi makanan
recommendations = {}
for meal, calorie_limit in calories_distribution.items():
    recommendations[meal] = recommend_food(food_data, calorie_limit)

# Tampilkan hasil
for meal, food in recommendations.items():
    print(f"Rekomendasi untuk {meal.capitalize()}:")
    print(food)
    print()
