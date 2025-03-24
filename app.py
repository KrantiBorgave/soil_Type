from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from model import predict_soil

app = Flask(__name__)
CORS(app)  # Important for cross-origin requests

# pH range function
def get_pH(soil_type):
    pH = {
        'alike': '5.5 to 7.5',
        'clay': '6.0 to 7.5',
        'dry rocky': '5.5 to 7.0',
        'grassy': '6.0 to 7.5',
        'gravel': '6.0 to 7.5',
        'humus': '5.5 to 7.0',
        'loam': '6.0 to 7.0',
        'not': 'undefined',
        'sandy': '5.5 to 7.5',
        'silty': '6.0 to 7.5',
        'yellow': '5.0 to 6.5'
    }
    return pH.get(soil_type.lower(), 'pH range not available')

# pH range function
def get_crop(soil_type):
    crop = {
        'alike': 'Wide variety of crops depending on composition (e.g., wheat, corn, tomatoes)',
        'clay': 'Rice, potatoes, cabbage, beans, carrots',
        'dry rocky': 'Very few crops; only drought-resistant plants like cacti',
        'grassy': 'Wheat, barley, oats, corn, lettuce, spinach',
        'gravel': 'Drought-resistant crops like grapes, olives, root vegetables',
        'humus': 'Leafy vegetables, tomatoes, fruits, root crops, herbs',
        'loam': 'Wheat, corn, carrots, beans, herbs',
        'not': 'No crops can grow here unless soil is added',
        'sandy': 'Carrots, potatoes, onions, strawberries',
        'silty': 'Lettuce, tomatoes, peas, cabbage',
        'yellow': 'Rice, millet, legumes'
    }
    return crop.get(soil_type.lower(), 'crop information not available')

def get_potass(soil_type):
    potass = {
        'alike': 'undefined',
        'clay': '0.5 to 3',
        'dry rocky': '0.05 to 0.3',
        'grassy': '0.3 to 1.5',
        'gravel': '0.05 to 0.3',
        'humus': '1 to 3',
        'loam': '0.5 to 2.5',
        'not': 'undefined',
        'sandy': '0.1 to 0.5',
        'silty': '0.1 to 0.8',
        'yellow': '0.2 to 1'
    }
    return potass.get(soil_type.lower(), 'potassium range not available')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read image
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_array = np.array(img)
        
        # Get prediction
        soil_type = predict_soil(img_array)

        # Get pH range
        pH = get_pH(soil_type)

        crop = get_crop(soil_type)

        potass = get_potass(soil_type)

        return jsonify({
            'soil_type': soil_type,
            'pH': pH,
            'crop': crop,
            'potass' : potass
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
