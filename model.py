import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("final_model.h5")

# Define soil classes
SOIL_CLASSES = [
    'alike',
    'clay',
    'dry rocky',
    'grassy',
    'gravel',
    'humus',
    'loam',
    'not',
    'sandy',
    'silty',
    'yellow',
]

def predict_soil(image_array):
    """Predicts soil type from an image array"""
    img = tf.image.resize(image_array, [256, 256])  # Normalize
    img = img / 255.0  # Resize for model
    img = tf.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    return SOIL_CLASSES[prediction.argmax()]
