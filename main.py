from flask import Flask, render_template, request, redirect, url_for, abort,make_response,request, jsonify
import os
import requests
import numpy as np
from googletrans import Translator
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import torch
from transformers import FlaxAutoModelForSeq2SeqLM
from transformers import AutoTokenizer

YOUTUBE_API_KEY = 'insert your key here'
app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('model.h5')

ingredients=[]
ingstr=""
# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

# Function to make predictions
def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction




@app.route('/')
def home():
    try:
         global ingredients
         global ingstr
         ingstr=""
         ingredients = []
         return render_template('index.html')
    except Exception as e:
        # Log the exception for debugging
        print(e)
        # Return an error response
        abort(500)

# Route to handle file upload and display prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    # Get the uploaded file
    try:

        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            
            # Save the uploaded file to the 'uploads' folder
            file_path = os.path.join('uploads', uploaded_file.filename)
            uploaded_file.save(file_path)
            # Make prediction
            prediction = predict(file_path)
            # Get the predicted class
            predicted_class = np.argmax(prediction)
            labels=["apple","banana","beetroot","bell pepper","cabbage","capsicum","carrot","cauliflower","chilli pepper","corn","cucumber","eggplant","garlic","ginger","grapes","jalepeno","kiwi","lemon","lettuce","mango","onion","orange","paprika","pear","peas","pineapple","pomegranate","potato","raddish","soy beans","spinach","sweetcorn","sweetpotato","tomato","turnip","watermelon"]
            # Return the predicted class to display on the webpage
            global ingredients
            global ingstr
            ingstr+=labels[predicted_class]+","
            print("-----------> ingstr:\n",ingstr)
            ingredients.clear()
            ingredients.append(ingstr)
            return render_template('predict.html', res=ingredients)

        
        else:
            return 'No file uploaded'
    except Exception as e:
            print("invalid image detected!")

@app.route('/recipe',methods=['POST','GET'])
def gen_recipe():
    global ingredients
    print("00000000000000000000000ing from gen func:000000000000000000  ",ingredients)
    MODEL_NAME_OR_PATH = "flax-community/t5-recipe-generation"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)

    prefix = "items: "
    generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "num_beams": 5,
    "length_penalty": 1.5,
}


    special_tokens = tokenizer.all_special_tokens
    tokens_map = {
        "<sep>": "--",
        "<section>": "\n"
    }
    def skip_special_tokens(text, special_tokens):
        for token in special_tokens:
            text = text.replace(token, "")

        return text

    def target_postprocessing(texts, special_tokens):
        if not isinstance(texts, list):
            texts = [texts]
        
        new_texts = []
        for text in texts:
            text = skip_special_tokens(text, special_tokens)

            for k, v in tokens_map.items():
                text = text.replace(k, v)

            new_texts.append(text)

        return new_texts

    def generation_function(texts):
        _inputs = texts if isinstance(texts, list) else [texts]
        inputs = [prefix + inp for inp in _inputs]
        inputs = tokenizer(
            inputs, 
            max_length=256, 
            padding="max_length", 
            truncation=True, 
            return_tensors="jax"
        )

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        output_ids = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            **generation_kwargs
        )
        generated = output_ids.sequences
        generated_recipe = target_postprocessing(
            tokenizer.batch_decode(generated, skip_special_tokens=False),
            special_tokens
        )
        return generated_recipe

#     items = [
#    "4 tablespoons vegetable oil, divided,  4 small potatoes, peeled and halved ,2 large onions, finely chopped,2 cloves garlic, minced,1 tablespoon minced fresh ginger root,2 medium tomatoes, peeled and chopped,1 teaspoon salt,1 teaspoon ground cumin, ½ teaspoon chili powder,½ teaspoon ground black pepper, ½ teaspoon ground turmeric,2 tablespoons plain yogurt,2 tablespoons chopped fresh mint leaves ,½ teaspoon ground cardamom ,1 (2 inch) piece cinnamon stick ,3 pounds boneless, skinless chicken pieces cut into chunks,1 pound basmati rice,2 ½ tablespoons vegetable oil,1 large onion, diced,5 pods cardamom,3 whole cloves,1 (1 inch) piece cinnamon stick,½ teaspoon ground ginger,1 pinch powdered saffron ,4 cups chicken stock,1 ½ teaspoons salt"
# ]
    generated = generation_function(ingredients)
    # recipe_text = "\n".join(generated)
    mystr=""
    for text in generated:
        sections = text.split("\n")
        for section in sections:
            section = section.strip()
            if section.startswith("title:"):
                section = section.replace("title:", "")
                headline = "TITLE"
            elif section.startswith("ingredients:"):
                section = section.replace("ingredients:", "")
                headline = "INGREDIENTS"
            elif section.startswith("directions:"):
                section = section.replace("directions:", "")
                headline = "DIRECTIONS"
            
            if headline == "TITLE":
                print(f"[{headline}]: {section.strip().capitalize()}")
                mystr+=f"[{headline}]: {section.strip().capitalize()}"
            else:
                section_info = [f"  - {i+1}: {info.strip().capitalize()}" for i, info in enumerate(section.split("--"))]
                print(f"[{headline}]:")
                mystr+=f"[{headline}]:"
                print("\n".join(section_info))
                mystr+="\n".join(section_info)

        print(mystr)
    translator = Translator()
    text_to_translate = mystr
    target_language = 'hi'
    translated_text_hindi = translator.translate(text_to_translate, dest=target_language)
    translated_text_tel = translator.translate(text_to_translate, dest='te')
    # print(translated_text_hindi.text)
    # ingredients = request.args.get('ingredients')
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400

    query=ingredients
    
    youtube_api_url = f'https://www.googleapis.com/youtube/v3/search?part=snippet&maxResults=10&q={query}&key={YOUTUBE_API_KEY}'
    
    response = requests.get(youtube_api_url)
    
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch data from YouTube API'}), 500
    
    data = response.json()
    video_links = [
        f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'
        for item in data.get('items', [])
        if item['id']['kind'] == 'youtube#video'
    ]
    # return render_template('recipe.html',links=video_links)
    return render_template('recipe.html',recipe=mystr,tr_h=translated_text_hindi.text,tr_tel=translated_text_tel.text,links=video_links)
    
@app.route('/set_ingredients', methods=['POST'])
def set_ingredients():
    global ingredients
    ingredients.clear()
    ingredients.append(request.form['ingredients'])
    print(f"Received ingredients: {ingredients}")  # Debug print
    return make_response('', 204)
        
@app.route('/about')
def about():
    return render_template('about.html')



if __name__ == '__main__':
    app.run(debug=True)
