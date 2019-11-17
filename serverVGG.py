import glob
import os, logging
import sys
import time
import re

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import socketio
import base64
from aiohttp import web
from io import BytesIO

#Lib's for model
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

found = False

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
print("Loaded Model")

# extract a single face from a given photograph
def extract_face(filename, is_extracted, required_size=(224, 224)):
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    if(not is_extracted):
        results = detector.detect_faces(pixels)
        if(results == []):
            return results
        else:
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            face = pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
    else:
        image = Image.fromarray(pixels)
    
    image = image.resize(required_size)
    face_array = asarray(image)
    
    return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames, is_extracted=False):
    faces = [extract_face(f, is_extracted) for f in filenames]
    if(faces == [[]]):
        return faces
    else:
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        yhat = model.predict(samples)
        return yhat

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, i, thresh=0.5):
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        global found
        found = True
        dir_name = filenames[i]
        dir_name = re.search(r'\\(\D+)\\', dir_name)
        dir_name = ' '.join(re.findall('[A-Z][^A-Z]*', dir_name.group(1)))
        print("Face de {} encontrada".format(dir_name))
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        return {"nome": dir_name, "minimun_distance": '%.3f' % (score), "result": "Found"}
    
def image_to_decode(face_image_encoded):
    decoded_data = base64.b64decode(face_image_encoded)
    filename = 'image_to_crop.jpg'
    with open(filename, 'wb') as f:
        f.write(decoded_data)

def recognize_face(face_image_encoded, input_embeddings):

    image_to_decode(face_image_encoded)
    target_file = glob.glob("image_to_crop.jpg")
    embeddings_target = get_embeddings(target_file, False)
    
    global found
    found = False
    image_result = {"nome": "Nenhuma face detectada.", "minimun_distance": str(1), "result": "Not Found"}

    if(embeddings_target != [[]]):
        for i, embedding in enumerate(input_embeddings):
            if(not found):
                for target in embeddings_target:
                    image_result = is_match(target, embedding, i)
            else:
                break

        if(not found):
            image_result = {"nome": "Nenhuma pessoa identificada.", "minimun_distance": str(1), "result": "Not Found"}

    return image_result


filenames = [img for img in glob.glob('images/*/*.jpg', recursive=True)]
input_embeddings = get_embeddings(filenames, True)
print("Load base embeddings")

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)


async def index(request):
    """Serve the client-side application."""
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.on('connect', namespace='/')
def connect(sid, environ):
	print("connect ", sid)


@sio.on('chat message', namespace='/')
async def message(sid, data):
    #print("message ", data)
   
    dictionary = dict(data)

    base64 = dictionary.get("img")

    result = recognize_face(base64, input_embeddings)

    print(result)

    await sio.emit('reply', result, namespace='/')


app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
	web.run_app(app)
