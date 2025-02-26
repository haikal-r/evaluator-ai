import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from pathlib import Path

# Load dataset


def recommend_meals(user_features):
    """
    Recommend breakfast, lunch, and dinner based on user nutritional needs.
    
    Parameters:
    - df: DataFrame containing food items and their nutritional information
    - user_features: Scaled nutritional requirements [Calories, Fat, Protein, Carbohydrates]
    
    Returns:
    - Dictionary of recommended meals
    """
    # Preprocessing
    file_path = Path(__file__).parent / 'datasets' / 'nutrition_table.csv'
    df = pd.read_csv(file_path)
    df = df[['name', 'Food Group', 'Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']].dropna()
    df['Meal Type'] = df['Food Group'].map({
        'Dairy and Egg Products': 'Breakfast',
        'Meats': 'Lunch',
        'Fruits': 'Dinner'
    })
    df = df.dropna(subset=['Meal Type'])
    
    # Prepare scaler and scaling features
    scaler = MinMaxScaler()
    features_to_scale = ['Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    # Prepare features and target
    X = df[features_to_scale]
    y = df.index
    
    # Train RandomForest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Recommend meals for each meal type
    recommendations = {}
    meal_types = ['Breakfast', 'Lunch', 'Dinner']
    
    for meal_type in meal_types:
        # Filter dataframe for specific meal type
        meal_df = df[df['Meal Type'] == meal_type]
        
        # If meal type exists in the dataset
        if not meal_df.empty:
            # Prepare features for the specific meal type
            X_meal = meal_df[features_to_scale]
            
            # Predict best meal
            predicted_index = int(model.predict(user_features)[0])
            
            # Ensure the predicted index is within the filtered meal type dataframe
            predicted_index = predicted_index % len(meal_df)
            recommended_food = meal_df.iloc[predicted_index]['name']
            
            recommendations[meal_type] = recommended_food
    
    return recommendations

# Example usage
# Assume df is your DataFrame with columns: 
# ['name', 'Meal Type', 'Calories', 'Fat (g)', 'Protein (g)', 'Carbohydrate (g)']

# Example user features (scaled values representing nutritional needs)
# [Calories, Fat, Protein, Carbohydrates]
user_features = np.array([[0.4, 0.1, 0.2, 0.3]])

# Get recommendations

meal_recommendations = recommend_meals(user_features)

# Print recommendations
for meal_type, food in meal_recommendations.items():
    print(f"{meal_type} Recommendation: {food}")