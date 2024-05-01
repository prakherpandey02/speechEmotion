import streamlit as st
import librosa
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input

# Define custom objects for loading the model
custom_objects = {
    'tf': tf,
    'StandardScaler': StandardScaler,  # Include any custom classes used in the model
    'Input': Input  # Ensure Input layer compatibility
    # Add any other custom objects used in the model here
}

# Load the trained model with custom objects
model = load_model('ser.h5', custom_objects=custom_objects, compile=False)

# Load and preprocess new audio data
def process_audio(file_path):
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(mfccs_processed.reshape(1, -1))
    scaled_features = np.expand_dims(scaled_features, axis=2)
    return scaled_features

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Streamlit UI
st.title('Speech Emotion Recognition')

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    processed_data = process_audio(uploaded_file)
    prediction = model.predict(processed_data)
    emotion_label = emotion_labels[np.argmax(prediction)]
    st.write(f'Predicted Emotion: {emotion_label}')
