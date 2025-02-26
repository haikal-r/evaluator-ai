import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from pathlib import Path
from joblib import dump


# Load dataset
file_path = Path(__file__).parent / 'datasets' / 'nutrition_table.csv'
df = pd.read_csv(file_path)

# Preprocessing
df = df[['name', 'Food Group', 'Calories', 'Protein (g)', 'Fat (g)', 'Carbohydrate (g)', 'Serving Weight 1 (g)']]
df['Meal Type'] = df['Food Group'].map({
    'Dairy and Egg Products': 'Breakfast',
    'Meats': 'Lunch',
    'Fruits': 'Dinner'
})
df = df.dropna(subset=['Meal Type'])

scaler = MinMaxScaler()
df[['Calories', 'Protein (g)', 'Fat (g)', 'Carbohydrate (g)']] = scaler.fit_transform(
    df[['Calories', 'Protein (g)', 'Fat (g)', 'Carbohydrate (g)']]
)

label_encoder = LabelEncoder()
# dump(label_encoder, 'label_encoder.joblib')
df['Meal Type'] = label_encoder.fit_transform(df['Meal Type'])

print('data frame', df)

# Train Random Forest
X = df[['Calories', 'Protein (g)', 'Fat (g)', 'Carbohydrate (g)']]
y = df['name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Functions
def classify_fitness_goal(weight, height):
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'

def calculate_tdee(weight, height, age, gender, activity_level):
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height * 100 - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height * 100 - 5 * age - 161

    activity_factors = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'active': 1.725,
        'very active': 1.9
    }
    tdee = bmr * activity_factors[activity_level.lower()]
    return tdee

def generate_meal_plan(tdee, restrictions):
    meal_plan = {'Breakfast': [], 'Lunch': [], 'Dinner': []}

    for meal_type in ['Breakfast', 'Lunch', 'Dinner']:
        meal_type_code = label_encoder.transform([meal_type])[0]
        print('meal_type_code', meal_type_code)
        filtered_df = df[df['Meal Type'] == meal_type_code]

        if restrictions == 'Vegetarian':
            filtered_df = filtered_df[~filtered_df['Food Group'].isin(['Meats', 'Fish'])]

        if not filtered_df.empty:
            sampled_meal = filtered_df.sample(n=1)
            for _, row in sampled_meal.iterrows():
                protein_calories = row['Protein (g)'] 
                fat_calories = row['Fat (g)'] 
                carb_calories = row['Carbohydrate (g)'] 
                total_calories = row['Calories']   

                meal_plan[meal_type].append({
                    'name': row['name'],
                    'Calories Total': total_calories,
                    'Protein Calories': protein_calories,
                    'Fat Calories': fat_calories,
                    'Carb Calories': carb_calories,
                    'Serving Size (g)': row['Serving Weight 1 (g)'] 
                })

    return meal_plan

# Example Input
user_inputs = {
    'weight': 75,  # in kg
    'height': 1.75,  # in meters
    'age': 30,
    'gender': 'male',
    'activity_level': 'moderate',
    'dietary_restrictions': ''
}

# Process
fitness_goal = classify_fitness_goal(user_inputs['weight'], user_inputs['height'])
tdee = calculate_tdee(user_inputs['weight'], user_inputs['height'], user_inputs['age'], user_inputs['gender'], user_inputs['activity_level'])
meal_plan = generate_meal_plan(tdee, user_inputs['dietary_restrictions'])

# Display Output
# print(f"Fitness Goal: {fitness_goal}")
# print(f"TDEE: {tdee:.2f} kcal/day\n")

# print("Meal Plan:")
# for meal, items in meal_plan.items():
#     print(f"{meal}:")
#     for item in items:
#         print(f"  - {item['name']}")
#         print(f"    Calories Total: {item['Calories Total']:.2f} calories")
#         print(f"    Protein: {item['Protein Calories']:.2f} calories")
#         print(f"    Fats: {item['Fat Calories']:.2f} calories")
#         print(f"    Carbs: {item['Carb Calories']:.2f} calories")
#         print(f"    Serving Size: {item['Serving Size (g)']:.2f} g\n")

# # Calculate Accuracy
y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
print('predict', y_pred)
# print(f"Model Accuracy: {accuracy:.2f}")

# joblib.dump(model, 'model.pkl')