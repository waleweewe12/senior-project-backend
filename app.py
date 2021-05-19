import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import requests
import time

app = Flask(__name__)
CORS(app)

#config firebase firestore  
cred = credentials.Certificate('./venomous-snake-firebase-adminsdk-dtu7f-e80b320dfa.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

#get model_data from firestore
model_data = {}
for i in ['body', 'head', 'mid', 'tail']:
    model_data[i] = db.collection(u'model').document(i).get().to_dict()
#load model from model_information
models = {}
for i in ['body', 'head', 'mid', 'tail']:
    path_to_downloaded_file = tf.keras.utils.get_file(model_data[i]['fileName'], model_data[i]['url'])
    models[i] = tf.keras.models.load_model(path_to_downloaded_file)

snakes = [
    'งูเห่า',
    'งูจงอาง',
    'งูสามเหลี่ยม',
    'งูทับสมิงคลา',
    'งูแมวเซา',
    'งูกัปปะ',
    'งูเขียวหางไหม้ท้องเหลือง',
    'งูเขียวหางไหม้ตาโต',
    'งูเขียวหางไหม้ภูเก็ต',
    'งูเขียวหางไหม้ลายเสือ',
    'งูต้องไฟ',
    'งูปล้องทอง',
    'งูปล้องหวายหัวดำ',
    'งูสามเหลี่ยมหัวแดงหางแดง'
]

def get_latest_model():
    """
        Note: โดยปกติแล้ว model, ค่า threshold, ค่า weight นั้นควรจะต้องทำการ load ใหม่ทุกครั้งที่ทำการ predict เพราะค่าพวกนี้
        มีการอัปเดตตลอดเวลา แต่เนื่องจากตัว model ใช้เวลา load ค่อนข้างนาน ดังนั้นจึงทำการโหลด model_data มาเทียบก่อนว่า timestamp ของ model
        มีการเปลี่ยนแปลงหรือไม่ หาก timestamp มีการเปลี่ยนแปลงจึงค่อยโหลด model ใหม่
    """
    #get model_data from firestore for check
    model_check = {}
    for i in ['body', 'head', 'mid', 'tail']:
        model_check[i] = db.collection(u'model').document(i).get().to_dict()
    #load model from firebase storage
    for i in ['body', 'head', 'mid', 'tail']:
        if(model_data[i]['timestamp'] != model_check[i]['timestamp']):
            path_to_downloaded_file = tf.keras.utils.get_file(model_check[i]['fileName'], model_check[i]['url'])
            models[i] = tf.keras.models.load_model(path_to_downloaded_file)

def get_latest_threshold():
    #load threshold from firestore
    threshold = {}
    for i in ['body', 'head', 'mid', 'tail']:
        threshold[i] = db.collection(u'threshold').document(i).get().to_dict()
    threshold_list = []
    for snake in snakes:
        ts = []
        for key in threshold:
            ts.append(float(threshold[key][snake]))
        threshold_list.append(ts)
    return threshold_list

def get_latest_weight():
    #load weight from firestore
    weight = {}
    for i in ['head', 'mid', 'tail']:
        weight[i] = db.collection(u'weight').document(i).get().to_dict()
    weight_list = []
    for snake in snakes:
        w = []
        for key in weight:
            w.append(weight[key][snake])
        weight_list.append(w)
    return weight_list

def predicted(data):
    # prepare data
    images = {}
    for key in data:
        url = data[key]
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        image = image.resize((299, 299))
        image = np.array(image)
        images[key] = np.array([image])
    #prediction
    get_latest_model()
    result_set = {}
    time_stamp = int(time.time() * 1000)
    for key in data:
        X = tf.keras.applications.inception_v3.preprocess_input(images[key], data_format=None)
        prob = models[key].predict(X)
        temp_dict = {}
        for i in range(len(prob[0])):
            temp_dict[str(i)] = str(round(prob[0][i], 4))
        result_set[key] = temp_dict
        #save result to firestore
        db.collection(u'result').document(f'{time_stamp}').set({
            'dateTime':time_stamp,
            'imageUrl':data[key],
            'predicted':result_set[key],
            'section':key,
            'status':'ยังไม่อัพโหลด'
        }) 
    #return response_data
    response_data = {}
    response_data['predicted'] = result_set
    response_data['threshold'] = get_latest_threshold()
    response_data['weight'] = get_latest_weight()
    return response_data

@app.route('/', methods=['GET'])
def hello():
    return "Hello World"

@app.route('/upload', methods=['POST'])
def test():
    data = request.json
    result = predicted(data)
    return result
   
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)