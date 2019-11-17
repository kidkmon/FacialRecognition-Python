import glob
import os
import sys
import time

import socketio
import base64
from aiohttp import web
from io import BytesIO

#Lib's for model
import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model
from keras.optimizers import SGD
from PIL import Image


path = "./model/"

loaded_model = load_model(path+'model_complete.h5', custom_objects={'tf': tf})
print("Loaded model from disk")


opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy',
                     optimizer=opt, metrics=['accuracy'])


class FaceCropper(object):
    CASCADE_PATH = "haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            img, 1.1, 3, minSize=(100, 100))    

        if (show_result):
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        facecnt = len(faces)
        height, width = img.shape[:2]

        if(facecnt == 0):
            try:
                os.remove("face_cropped.png")
                return 0
            except FileNotFoundError:
                return 0
        else:
            for (x, y, w, h) in faces:
                r = max(w, h) / 2
                centerx = x + w / 2
                centery = y + h / 2
                nx = int(centerx - r)
                ny = int(centery - r)
                nr = int(r * 2)

                faceimg = img[ny:ny+nr, nx:nx+nr]
                cv2.imwrite("face_cropped.png", faceimg)
                return 1


detecter = FaceCropper()


def image_to_embedding(face_image_encoded, model, encoded=True):
    if(encoded):
        decoded_data = base64.b64decode(face_image_encoded)
        filename = 'image_to_crop.png'
        with open(filename, 'wb') as f:
            f.write(decoded_data)

        face = detecter.generate(filename, False)
        
        if(face == 0):
            return 0
        else:
            time.sleep(2)
            img_decoded = Image.open('face_cropped.png')
            image = img_decoded.resize((96, 96))
            image = np.asarray(image)
    else:
        img_decoded = face_image_encoded
        image = cv2.resize(img_decoded, (96, 96))

    img = image[..., ::-1]
    img = np.around(np.divide(img, float(np.max(img))), decimals=12)
    # img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    # x_train = x_train.reshape(1, *x_train.shape)

    embedding = model.predict_on_batch(x_train)
    return embedding


def create_input_image_embeddings():
    input_embeddings = {}

    for file in glob.glob("images/*"):
        person_name = os.path.splitext(os.path.basename(file))[0]
        image_file = cv2.imread(file, 1)
        input_embeddings[person_name] = image_to_embedding(
            image_file, loaded_model, False)

    return input_embeddings


def recognize_face(face_image_encoded, input_embeddings, model):

    embedding = image_to_embedding(face_image_encoded, model)

    if(type(embedding) == int):
        return {"nome": "Face nao reconhecida", "minimun_distance": str(1), "result": "Not Found"}

    minimum_distance = 200
    name = None

    for (input_name, input_embedding) in input_embeddings.items():
        euclidean_distance = np.linalg.norm(embedding-input_embedding)

        print('Euclidean distance from %s is %s' %
              (input_name, euclidean_distance))

        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name

    if minimum_distance < 0.68:
        return {"nome": str(name), "minimun_distance": str(minimum_distance), "result": "Found"}
    else:
        return {"nome": "Tente novamente", "minimun_distance": str(minimum_distance), "result": "Not Found"}


input_embeddings = create_input_image_embeddings()
sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

h5 = False


async def index(request):
    """Serve the client-side application."""
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')


@sio.on('connect', namespace='/')
def connect(sid, environ):
	print("connect ", sid)


@sio.on('chat message', namespace='/')
async def message(sid, data):
    print("message ", data)
    print("valor do h5:: " + str(h5))

    dictionary = dict(data)

    base64 = dictionary.get("img")

    result = recognize_face(base64, input_embeddings, loaded_model)

    print(result)

    await sio.emit('reply', result, namespace='/')


app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
	web.run_app(app)
