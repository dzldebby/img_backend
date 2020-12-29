import aiohttp
import asyncio
import uvicorn
import os
import fastai
from fastai.vision import *
from PIL import Image
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import requests
import joblib
from requests_testadapter import Resp
from flask import Flask, render_template, request, jsonify
from flask_restful import Api, Resource, reqparse
import requests

from flask import Flask, render_template

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
path = Path(__file__).parent
classes = ["airplane", "ambulance", "animal", "artist", "aurora", "baby", "beach", "bear", "bedroom", "bicycle", "bird", "boats", "book", "bridge", "building", "bus", "cars", "castle", "cat", "city", "clouds", "college", "column", "concert", "couple", "crops", "dance", "dawn", "deer", "desert", "dessert", "doctor", "dog", "dolphins", "field", "fire", "floor", "food", "golf", "graffiti", "grandfather", "grandmother", "grass", "hair", "hand", "horse", "hospital", "house", "human", "insect", "kid", "library", "lights", "man", "moon", "mountain", "music", "nature", "neon", "nurse", "ocean", "painting", "palm", "party", "person", "phone", "plant", "rain", "rainforest", "restaurant", "river", "robot", "rocks", "roses", "shirt", "shop", "sign", "sky", "skyscraper", "snow", "soccer", "sports", "stadium", "staircase", "stars", "storm", "street", "sun", "sunrise", "temple", "tree", "truck", "vegetable", "water", "waves", "weed", "windows", "woman", "wood"]
learn = load_learner(path)

print("model loaded")


def sorted_prob(classes,probs):
  pairs = []
  for i,prob in enumerate(probs):
    pairs.append([prob.item(),i])
  pairs.sort(key = lambda o: o[0], reverse=True)
  return pairs

def getname(arr):
    result = []
    probability = []
    for i in range(1,2):
        element = arr[i]
        result.append(classes[element[1]])
        probability.append((element[0] * 100))
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/after', methods=['GET', 'POST'])
def after():
    file = request.files['file']
    file.save('static/file.jpg')
    return jsonify({'result': 'success'})


@app.route('/randoms', methods=['GET'])
def randoms():
    response = requests.get('https://source.unsplash.com/random/500x500')
    imgraw = BytesIO(response.content)
    img = open_image(imgraw)

    prediction = learn.predict(img)[2]

    bests = sorted_prob(classes, prediction)
    result = getname(bests)

    return jsonify({'result': str(result), 'url': response.url})


@app.route('/local', methods=['GET'])
def local():
    img = open('static/file.jpg', 'rb').read()
    imgraw = BytesIO(img)
    img = open_image((imgraw))
    prediction = learn.predict(img)[2]

    bests = sorted_prob(classes, prediction)
    result = getname(bests)

    return jsonify({'result': str(result)})



if __name__ == "__main__":
    app.run(debug=True)
