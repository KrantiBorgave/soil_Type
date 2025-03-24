// App.js
import React, { useState, useEffect } from 'react';
import { View, Text, Button, Image, ActivityIndicator } from 'react-native';
import * as tf from '@tensorflow/tfjs';
import { bundleResourceIO } from '@tensorflow/tfjs-react-native';
import * as ImagePicker from 'expo-image-picker';

// Load your TFLite model (place soil_model.tflite in assets/)
const modelJson = require('./assets/model.json');
const modelWeights = require('./assets/weights.bin');

export default function App() {
  const [model, setModel] = useState(null);
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState('');
  const [phRange, setPhRange] = useState('');
  const [loading, setLoading] = useState(false);

  // Load TensorFlow and model
  useEffect(() => {
    (async () => {
      await tf.ready();
      const loadedModel = await tf.loadGraphModel(bundleResourceIO(modelJson, modelWeights));
      setModel(loadedModel);
    })();
  }, []);

  // Mapping predicted class index to soil type
  const soilTypes = [ 'alike', 'clay', 'dry rocky', 'grassy', 'gravel', 'humus', 'loam', 'not', 'sandy', 'silty', 'yellow',];

  // Function to get pH range
  const getPhRange = (soilType) => {
    const phRanges = {
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
        'yellow': '5.0 to 6.5',
    };
    return phRanges[soilType] || 'pH range not available';
  };

  const getcrop = (soilType) => {
    const crop = {
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
    };
    return crop[soilType] || 'crop information not available';
  };

  const getpotass = (soilType) => {
    const potassRanges = {
        'alike': 'undefined',
        'clay': '0.5% to 3%',
        'dry rocky': '0.05% to 0.3%',
        'grassy': '0.3% to 1.5%',
        'gravel': '0.05% to 0.3%',
        'humus': '1% to 3%',
        'loam': '0.5% to 2.5%',
        'not': 'undefined',
        'sandy': '0.1% to 0.5%',
        'silty': '0.1% to 0.8%',
        'yellow': '0.2% to 1%',
    };
    return potassRanges[soilType] || 'potassium range not available';
  };

  // Pick image from gallery
  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      predictSoilType(result.assets[0].uri);
    }
  };

  // Predict soil type
  const predictSoilType = async (uri) => {
    if (!model) return;
    setLoading(true);
    try {
      // Preprocess image
      const imageTensor = await tf.reshape(
        tf.browser.fromPixels(await loadImage(uri)).resizeNearestNeighbor([224, 224]),
        [1, 224, 224, 3]
      );

      // Predict
      const output = model.predict(imageTensor);
      const predictedClass = tf.argMax(output.dataSync());
      setPrediction(`Soil Type: ${predictedClass}`);

      // Set prediction and pH range
      setPrediction(`Soil Type: ${soilType}`);
      setPhRange(`pH Range: ${getPhRange(soilType)}`);
      setcrop(`crop: ${getcrop(soilType)}`);
      setpotass(`potass: ${getpotass(soilTypes)}`);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={{ flex: 1, padding: 20 }}>
      <Button title="Upload Soil Image" onPress={pickImage} />
      {image && <Image source={{ uri: image }} style={{ width: 200, height: 200 }} />}
      {loading && <ActivityIndicator size="large" />}
      {prediction && <Text style={{ fontSize: 20 }}>{prediction}</Text>}
      {phRange && <Text style={{ fontSize: 18, marginTop: 10 }}>{phRange}</Text>}
    </View>
  );
}

// Helper to load image
async function loadImage(uri) {
  const response = await fetch(uri);
  const blob = await response.blob();
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}