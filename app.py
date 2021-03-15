"""
Name: Dev Guha

I tried deploying the api on heroku free tier, but the project size was turning out to be greated then 500mb
which is not allowed on heroku free tier. So here is the code.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
import json
import pandas as pd
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import base64
import numpy as np
import io
import tensorflow as tf
from PIL import Image
tf.compat.v1.disable_eager_execution()
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
req_labels = "beagle, chihuahua, doberman, french_bulldog, golden_retriever, malamute, pug, saint_bernard, scottish_deerhound, tibetan_mastiff"
req_labels = req_labels.split(", ")
model = tf.keras.models.load_model("dogs_clf_v5_30.h5")
global graph
graph = tf.compat.v1.get_default_graph()

@app.route('/api', methods=['POST'])
def get_dog_breed():
   size = (224,224)
   model = tf.keras.models.load_model("dogs_clf_v5_30.h5")
   graph = tf.compat.v1.get_default_graph()
   data = request.get_json()
   b64 = data["image"]
   dog_img = base64.b64decode(b64)
   dog_img = Image.open(io.BytesIO(dog_img))
   dog_img = dog_img.resize(size)
   dog_img = np.array(dog_img)
   dog_img = np.array([dog_img])
   with graph.as_default():
      res = model.predict(dog_img)
   res = list(res[0])
   for i, ele in enumerate(res):
      if ele == max(res):
         ret = {
                     "breed": req_labels[i],
                     "score": str(ele)
                  }                          
   ret = json.dumps(ret)
   return jsonify(ret)

if __name__=='__main__':
    app.run(port=int(os.environ.get('PORT', 5000)), debug=True)
