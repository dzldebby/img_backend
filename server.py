import asyncio
import aiohttp
from fastai.vision import *
from io import BytesIO
from flask import Flask, render_template, request, jsonify
import requests

export_file_url = 'https://drive.google.com/uc?export=download&id=1gEn5Q4P8SxSpqQAZ-cpcWEsZLk7V3_ZK'
export_file_name = 'Ultimate-100Labels.pkl'


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
path = Path(__file__).parent
classes = ["airplane", "ambulance", "animal", "artist", "aurora", "baby", "beach", "bear", "bedroom", "bicycle", "bird", "boats", "book", "bridge", "building", "bus", "cars", "castle", "cat", "city", "clouds", "college", "column", "concert", "couple", "crops", "dance", "dawn", "deer", "desert", "dessert", "doctor", "dog", "dolphins", "field", "fire", "floor", "food", "golf", "graffiti", "grandfather", "grandmother", "grass", "hair", "hand", "horse", "hospital", "house", "human", "insect", "kid", "library", "lights", "man", "moon", "mountain", "music", "nature", "neon", "nurse", "ocean", "painting", "palm", "party", "person", "phone", "plant", "rain", "rainforest", "restaurant", "river", "robot", "rocks", "roses", "shirt", "shop", "sign", "sky", "skyscraper", "snow", "soccer", "sports", "stadium", "staircase", "stars", "storm", "street", "sun", "sunrise", "temple", "tree", "truck", "vegetable", "water", "waves", "weed", "windows", "woman", "wood"]


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

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
