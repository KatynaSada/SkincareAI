import os
import random
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Function to display images and get labels
def label_images(images, labels):
    labeled_data = {}
    st.write("Please label the following images:")
    for i, img in enumerate(images):
        st.image(img, caption=f"Image {i+1}")
        label = st.radio(f"Label for Image {i+1}:", ('Acne', 'No Acne'), key=f"label_{i}")
        labeled_data[i] = 1 if label == 'Acne' else 0
    return labeled_data

# Load images from the directory
def load_images_from_directory(directory, num_samples=5):
    data_gen = ImageDataGenerator(rescale=1./255.)
    generator = data_gen.flow_from_directory(
        directory, 
        target_size=(150, 150), 
        batch_size=num_samples, 
        class_mode='categorical'
    )
    images, labels = next(generator)
    return images, labels, generator.class_indices

# Train the model
def train_model(X, y, learning_rate, epochs):
    mobilenet = MobileNet(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
    
    for layer in mobilenet.layers:
        layer.trainable = False
    
    model = Sequential([
        mobilenet,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy'])
    
    history = model.fit(X, y, epochs=epochs, batch_size=8, validation_split=0.2)
    return model, history

# Evaluate and plot the results
def evaluate_model(model, X_val, y_val, history):
    predicted_classes = np.argmax(model.predict(X_val), axis=1)
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_val, predicted_classes, average='weighted')

    accuracy = accuracy_score(y_val, predicted_classes)

    st.write('Precision:', precision)
    st.write('Recall:', recall)
    st.write('F1-Score:', f1_score)
    st.write('Accuracy:', accuracy)

    cm = confusion_matrix(y_val, predicted_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Acne', 'Acne'], yticklabels=['No Acne', 'Acne'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'], loc='upper left')
    st.pyplot(plt)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'], loc='upper left')
    st.pyplot(plt)

# Streamlit app setup
st.title("Interactive Model Training for Acne Detection")

# Define directories
training_dir = '/Users/katyna/.cache/kagglehub/datasets/xtvgie/face-datasets/versions/1/Acne/Train'

# Load and label images
images, labels, class_indices = load_images_from_directory(training_dir, num_samples=10)
if "labeled_data" not in st.session_state:
    st.session_state.labeled_data = label_images(images, labels)

# Prepare data for training
labeled_data = st.session_state.labeled_data
X = np.array([img_to_array(img) for img in images])
y = np.array([labeled_data[i] for i in range(len(images))])

# Train the model
if st.button("Train Model"):
    learning_rate = st.slider("Select Learning Rate", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001)
    epochs = st.slider("Select Number of Epochs", min_value=1, max_value=50, value=10, step=1)
    
    model, history = train_model(X, y, learning_rate, epochs)
    X_val, y_val = X, y  # For demonstration purposes, using the same data for validation
    evaluate_model(model, X_val, y_val, history)
