# Image Recognition and Recipe Generation Application

This application leverages machine learning for image recognition and recipe generation based on the identified ingredients. Built with Flask for the web framework and TensorFlow for model training, this project combines image processing and natural language processing techniques to deliver a seamless user experience.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
  - [Model Architecture](#model-architecture)
- [Directory Structure](#directory-structure)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

## Features

- Upload images of various ingredients to get predictions.
- Generate recipes based on recognized ingredients.
- Translate recipe instructions into multiple languages.
- Display relevant YouTube videos for each recipe.

## Technologies Used

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras
- **NLP**: Hugging Face Transformers, Google Translate API
- **Frontend**: HTML, Tailwind CSS
- **Data Processing**: NumPy, Pandas

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/repository.git
   cd repository
2. Create a virtual environment and activate it:
   ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the required packages:
   ```bash
   pip install -r requirements.txt

5. Set your YouTube API key in the main.py file:
   ```bash
   YOUTUBE_API_KEY = 'insert your key here'


**USAGE**
```bash
python main.py
```
Open your browser and go to http://127.0.0.1:5000/.

- Upload an image of an ingredient and view the predicted class.

- Click on "Generate Recipe" to create a recipe based on the predicted ingredients.

- This project implements a machine learning model for ingredient classification using MobileNetV2 and provides a Flask application for image uploads and recipe generation based on the classified ingredients.

## Model Training

The model training process is detailed in the `model.ipynb` file. Below are the steps involved in building and training the model:

### 1. Data Preparation
- Images are organized in a directory structure with subdirectories for each ingredient class.
- The dataset includes training, validation, and test sets.

### 2. Image Processing
- Images are preprocessed using TensorFlow's `ImageDataGenerator`, which performs data augmentation to improve model generalization.

### 3. Model Architecture
- The application utilizes **MobileNetV2** as the backbone for feature extraction.
- MobileNetV2 is a lightweight CNN architecture designed for mobile and edge devices, focusing on efficiency and speed.
- The model architecture consists of:
  - Two dense layers with ReLU activation.
  - An output layer with softmax activation for multi-class classification.

### 4. Training
- The model is trained using categorical cross-entropy loss and the Adam optimizer.
- Early stopping is employed to prevent overfitting.

### 5. Model Saving
- After training, the model is saved as `model.h5` for later use in the Flask application.

### Model Architecture Code
```python
import tensorflow as tf

pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
pretrained_model.trainable = False

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(36, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

```
**Directory**
/repository
│
├── /uploads                # Folder for uploaded images
├── /templates              # HTML templates for Flask
├── /static                 # Static files (CSS, JS, images)
│
├── main.py                 # Main Flask application
├── model.ipynb             # Jupyter notebook for training the model
├── model.h5                # Saved Keras model
├── requirements.txt        # Python packages required
└── README.md               # Project documentation


## API Endpoints

- **GET /**: Render the home page.
- **POST /upload**: Handle image upload and return prediction.
- **POST /recipe**: Generate a recipe based on ingredients.
- **POST /set_ingredients**: Set ingredients for recipe generation.
- **GET /about**: Render the about page.
