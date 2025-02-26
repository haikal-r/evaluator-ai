# import pandas as pd
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import label_binarize
# from sklearn.preprocessing import MinMaxScaler

# def food_recommendation():
#     file_path = Path(__file__).parent / 'nutrition_table.csv'

#     try:
#         df = pd.read_csv(file_path, encoding='utf-8')  
#         print(df.head()) 
#         print(df.columns)

       
#         nutrisi_cols = ['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']
#         scaler = MinMaxScaler()
#         df[nutrisi_cols] = scaler.fit_transform(df[nutrisi_cols])

#         # Bobot untuk fitur
#         bobot_nutrisi = {'Calories': 0.3, 'Protein (g)': 0.25, 'Fat (g)': 0.2, 'Carbohydrate (g)': 0.15}
#         bobot_preference = {'Vegan': 0.3, 'Vegetarian': 0.2, 'Gluten-free': 0.1, 'No Preference': 0.0}
#         bobot_availability = {'Yes': 0.1, 'No': 0.0}

#         # Fungsi untuk menghitung prioritas
#         def hitung_prioritas(row):
#             nutrisi_score = sum(row[col] * weight for col, weight in bobot_nutrisi.items())
#             preference_score = bobot_preference.get(row['Food Group'], 0)
#             availability_score = bobot_availability.get(row['Calories'], 0)

#             # Total skor prioritas
#             total_skor = nutrisi_score + preference_score + availability_score

#             # Tentukan level prioritas
#             if total_skor <= 0.4:
#                 return 'Rendah'
#             elif total_skor <= 0.7:
#                 return 'Sedang'
#             else:
#                 return 'Tinggi'

#         # Tambahkan kolom Prioritas
#         df['Prioritas'] = df.apply(hitung_prioritas, axis=1)

#         print(df[['name', 'Prioritas']].head())

#         uploaded_file_path = Path(__file__).parent / 'nutrition_table.csv'
#         data = pd.read_csv(uploaded_file_path)

#         # Example criteria weights (adjust these based on the AHP pairwise comparison process)
#         criteria_weights = {
#             "Calories": 0.3,       # Energy intake is key
#             "Protein (g)": 0.25,   # Muscle maintenance and growth
#             "Fat (g)": 0.15,       # Essential fatty acids
#             "Carbohydrate (g)": 0.2,  # Energy balance
#             "Fiber (g)": 0.1       # Digestive health
#         }

#         # Normalize each column based on the criteria
#         def normalize_column(column):
#             return (column - column.min()) / (column.max() - column.min())

#         for criterion in criteria_weights.keys():
#             if criterion in data.columns:
#                 data[criterion + "_normalized"] = normalize_column(data[criterion])

#         # Compute a weighted score for each meal
#         data['Score'] = 0
#         for criterion, weight in criteria_weights.items():
#             if criterion + "_normalized" in data.columns:
#                 data['Score'] += data[criterion + "_normalized"] * weight

#         # Classify meals into BMI categories
#         def classify_bmi(bmi):
#             if bmi < 18.5:
#                 return "Underweight"
#             elif 18.5 <= bmi < 24.9:
#                 return "Normal weight"
#             elif 25 <= bmi < 29.9:
#                 return "Overweight"
#             else:
#                 return "Obese"

#         # Example: Generate meals for each BMI category
#         user_bmi = 23  # Replace with user BMI
#         bmi_category = classify_bmi(user_bmi)

#         # Filter meals based on BMI-specific criteria
#         if bmi_category == "Underweight":
#             meal_criteria = (data["Calories"] > 300) & (data["Protein (g)"] > 10)  
#         elif bmi_category == "Normal weight":
#             meal_criteria = (data["Calories"].between(200, 500)) & (data["Fiber (g)"] > 3)
#         elif bmi_category == "Overweight":
#             meal_criteria = (data["Calories"] < 400) & (data["Fat (g)"] < 15)
#         else:  # Obese
#             meal_criteria = (data["Calories"] < 300) & (data["Carbohydrate (g)"] < 50)

#         # Select meals matching criteria
#         filtered_meals = data[meal_criteria].sort_values(by="Score", ascending=False)
#         print("FILTER", filtered_meals.head())

#         # Generate a meal plan for breakfast, lunch, and dinner
#         breakfast = filtered_meals.iloc[0:3][["name", "Calories", "Protein (g)", "Score"]]  # Top 3 for breakfast
#         lunch = filtered_meals.iloc[3:6][["name", "Calories", "Protein (g)", "Score"]]     # Next 3 for lunch
#         dinner = filtered_meals.iloc[6:9][["name", "Calories", "Protein (g)", "Score"]]    # Next 3 for dinner

#         # Print meal plans
#         print("Breakfast Meals:")
#         print(breakfast)
#         print("\nLunch Meals:")
#         print(lunch)
#         print("\nDinner Meals:")
#         print(dinner)

#     except FileNotFoundError:
#         print(f"Error: File not found at {file_path}")

#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         return {"error": str(e)}

#     except pd.errors.EmptyDataError:
#         print("Error: The file is empty or invalid.")
#         return {"error": "The file is empty or invalid."}

#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         return {"error": "An unexpected error occurred."}