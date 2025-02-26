from flask import Flask, request, jsonify
from food import NutritionModel
import joblib
from pathlib import Path
from werkzeug.exceptions import BadRequest
from workout import WorkoutRecommender
import pandas as pd
from helper.calculate_tdee import calculate_tdee
import sys

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status" : "success",
        "message" : "Hello world"
    })

@app.route('/food', methods=['POST'])
def predict_food():
    
    user_input = request.get_json()
    required_keys = ['age', 'weight', 'height', 'activity_level', 'dietary_preferences']

    for key in required_keys:
        if key not in user_input or user_input[key] is None or user_input[key] == '':
            return jsonify({"error": f"Missing or invalid value for '{key}'"}), 400

    data = {
        'age': user_input['age'],
        'weight': user_input['weight'],
        'height': user_input['height'],
        'activity_level': user_input['activity_level']
    }
    
    
    nutrition_needed = calculate_tdee(data)
    print(nutrition_needed)
    user_restriction = user_input['dietary_preferences']
    
    file_path = Path(__file__).parent / 'food_model' / 'nutrition_model.pkl'
    loaded_model = joblib.load(file_path)

    recommendation = loaded_model.generate_recommendation(nutrition_needed, user_restriction)
    return jsonify({'data': recommendation})

@app.route('/workout', methods=['POST'])
def predict_workout():
    try:        
        file_path = Path(__file__).parent / 'datasets' / 'workout_data.csv'
        df = pd.read_csv(file_path)
        recommender = WorkoutRecommender(df)
       

        # Get input data from request
        # input_data = request.get_json()
        input_data = {
            "experience_level": "Beginner",
            "equipment": "No",
            "calories": 200
        }

        if not input_data:
            raise BadRequest("No input data provided")

        result = recommender.predict_workout(input_data)
        return jsonify({"data": result}), 200
    
    except BadRequest as e:
        return jsonify({
            "error": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
        }), 500


if __name__ == "__main__":
    app.run(debug=True)