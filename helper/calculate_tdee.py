import sys

def calculate_tdee(user_input):
    """
    Menghitung TDEE dan kebutuhan nutrisi harian berdasarkan data user.

    Args:
        height (float): Tinggi badan dalam cm.
        weight (float): Berat badan dalam kg.
        bmi (float): Body Mass Index.
        activity_level (str): Tingkat aktivitas, salah satu dari ['sedentary', 'light', 'moderate', 'active', 'very active'].

    Returns:
        dict: Kebutuhan kalori, protein, karbohidrat, dan lemak per hari.
    """
    # Menghitung BMR menggunakan rumus Harris-Benedict (untuk laki-laki dan perempuan berbeda, asumsi di sini untuk laki-laki)
    age = user_input['age']
    weight = user_input['weight']
    height = user_input['height']
    activity_level = 'moderate'
    
    bmr = 10 * weight + 6.25 * height - 5 * age + 5 
    print('bmr', bmr) 

    # Faktor aktivitas
    activity_factors = {
        'sedentary': 1.2,          # Tidak aktif (banyak duduk)
        'light': 1.375,           # Aktivitas ringan
        'moderate': 1.55,         # Aktivitas sedang
        'active': 1.725,          # Sangat aktif
        'very_active': 1.9        # Sangat aktif (atlet)
    }

    if activity_level not in activity_factors:
        raise ValueError("Activity level tidak valid. Pilih dari: sedentary, light, moderate, active, very active.")

    # Menghitung TDEE
    tdee = bmr * activity_factors[activity_level]

    # Kebutuhan makronutrisi (asumsi)
    protein = weight * 2  # 2 gram protein per kg berat badan
    fat = weight * 1      # 1 gram lemak per kg berat badan
    carbo = (tdee - (protein * 4 + fat * 9)) / 4  # Sisa kalori untuk karbohidrat

    return {
        'Calories': round(bmr, 2),
        'Protein (g)': round(protein, 2),  # gram
        'Carbohydrate (g)': round(carbo, 2),  # gram
        'Fat (g)': round(fat, 2)  # gram
    }