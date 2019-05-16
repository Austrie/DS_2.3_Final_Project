import math
import argparse
from flask_restplus import Api, Resource, fields
from flask import Flask, jsonify, request, make_response, abort, render_template, redirect, url_for
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from flask_restplus import Api, Resource, fields, marshal_with
from flask import Flask, request, jsonify
import numpy as np
from werkzeug.datastructures import FileStorage
from PIL import Image
from keras.models import model_from_json
import tensorflow as tf
from keras.models import load_model
import time
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from keras.wrappers.scikit_learn import KerasRegressor
import pandas
import csv

cred = credentials.Certificate('ups-warehouse-app-firebase-adminsdk-97yy8-2a2c26ad1b.json')
firebase_admin.initialize_app(cred)

db = firestore.client()
# print('About to print users')
# users_ref = db.collection(u'ds23_flask')
# docs = users_ref.get()
# # print("Number of items", str(len(docs)))
# for doc in docs:
#     print(u'{} => {}'.format(doc.id, doc.to_dict()))

application = Flask(__name__)
api = Api(application, version='1.0',
          title='International Airline Passengers Prediction',
          description='Predicting number of passengers taking international flights per month')
ns = api.namespace('Make_School', description='Methods')

timestamp_model = api.model(
    "Timestamp",
    {
        'month': fields.Integer("Month (e.g. January is '01', December is '12', etc.)"),
        'year': fields.Integer("Year (e.g. 1984,  2018, etc.)")
    }
)
graph = tf.get_default_graph()
model = None

# single_parser = api.parser()
# single_parser.add_argument('file', location='files',
#                            type=FileStorage, required=True)
#
# model = load_model('my_model.h5')
# graph = tf.get_default_graph()
#
# with open('model_architecture.json', 'r') as f:
#     new_model_1 = model_from_json(f.read())
# new_model_1.load_weights('model_weights.h5')


# def cylinder_volume(radius, height):
#     vol = (math.pi) * (radius ** 2)  * height
#     return vol
#
#
# def summation(a, b):
#     return a+b

# @ns.route('/addition')
# class Addition(Resource):
#     @api.doc(parser=single_parser, description='Enter two integers')
#     def get(self):
#         args = single_parser.parse_args()
#         n1 = args.n
#         m1 = args.m
#         r = summation(n1, m1)
#         return {'add': r}

@application.route('/')
def hello_world():
    doc_ref = db.collection(u'ds23_flask').document(u'last_time_accessed')
    doc_ref.set({
        u'time': time.time(),
    })
    # print('About to print users')
    # users_ref = db.collection(u'users')
    # docs = users_ref.get()
    # for doc in docs:
    #     print(u'{} => {}'.format(doc.id, doc.to_dict()))
    # return "hi"
    # r = args.radius # request.args.get('r', type=int)
    # h = args.height # request.args.get('h', type=int)
    # return jsonify({cylinder_volume(r, h)})

@ns.route('/prediction')
class MLPPrediction(Resource):
    """Enter month and date for passenger prediction"""
    @api.expect(timestamp_model)
    def post(self):
        doc_ref = db.collection(u'ds23_flask').document(u'last_time_accessed')
        doc_ref.set({
            u'time': time.time(),
        })
        global model
        if model is None:
            with open('KerasRegressor.pkl', 'rb') as input:
               model = pickle.load(input)
        print('Payload:', api.payload)
        timestamp = {'Month': api.payload['month'], 'Year': api.payload['year']}
        if not timestamp['Month'] or timestamp['Month'] < 1 or timestamp['Month'] > 12:
            statement = {"ERROR": 'Improper month format. Remember "month" has to be between' \
                + ' 1 and 12 (inclusive)'}
            print(statement)
            return statement
        if not timestamp['Year'] or timestamp['Year'] < 1949 or timestamp['Year'] > 9999:
            statement = {"ERROR": 'Improper year format. Remember "year" has to be between' \
                + ' 1949 and 9999 (inclusive)'}
            print(statement)
            return statement
        timestamp['Month'] = [timestamp['Month']]
        timestamp['Year'] = [timestamp['Year']]
        print('Timestamp:', timestamp)
        df = pandas.DataFrame.from_dict(timestamp)
        print("DF:", df)
        prediction = model.predict(df)
        print(prediction)
        # f = request.files['data_file']
        # if not f:
        #     return "No file"
        # stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        # csv_input = csv.reader(stream)
        # writer = csv.writer(open("./airline-passengers.csv", 'w'))
        # for row in csv_input:
        #     writer.writerow(row)
        # X, Y = load_data
        # prediction = predict(X)
        # data = [row for row in csv_input]
        # doc_ref = db.collection(u'ds23_flask').document(str(time.time()))
        # doc_ref.set({
        #     u'time': time.time(),
        #     u'prediction': str(r),
        #     u'csv': data,
        #     u'prediction': prediction,
        #     u'real': Y
        # })
        doc_ref = db.collection(u'ds23_flask').document(str(time.time()))
        doc_ref.set({
            u'time': time.time(),
            u'data': timestamp,
            u'prediction': prediction.tolist(),
        })
        return {'prediction': str(prediction)}

# @ns.route('/old-prediction')
# class CNNPrediction(Resource):
#     """Uploads your data to the CNN"""
#     @api.doc(parser=single_parser, description='Upload an mnist image')
#     def post(self):
#         # doc_ref = db.collection(u'ds23_flask').document(u'last_time_accessed')
#         # doc_ref.set({
#         #     u'time': time.time(),
#         # })
#         # model = None
#         # with open('KerasRegressor.pkl', 'rb') as input:
#         #    model = pickle.load(input)
#         # f = request.files['data_file']
#         # if not f:
#         #     return "No file"
#         # stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
#         # csv_input = csv.reader(stream)
#         # writer = csv.writer(open("./airline-passengers.csv", 'w'))
#         # for row in csv_input:
#         #     writer.writerow(row)
#         # X, Y = load_data
#         # prediction = predict(X)
#         # data = [row for row in csv_input]
#         # doc_ref = db.collection(u'ds23_flask').document(str(time.time()))
#         # doc_ref.set({
#         #     u'time': time.time(),
#         #     u'prediction': str(r),
#         #     u'csv': data,
#         #     u'prediction': prediction,
#         #     u'real': Y
#         # })
#         prediction = "HI"
#         return {'prediction': str(prediction)}

def baseline_model():
  # create model
  model = Sequential()
  model.add(Dense(2, input_dim=2, kernel_initializer='normal', activation='relu'))
  model.add(Dense(1, kernel_initializer='normal'))
  # Compile model
  model.compile(loss='mean_squared_error', optimizer='adam')
  return model


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Calculate volume of a Cylinder')
    # parser.add_argument('-r', '-radius', type=int, required=True)
    # parser.add_argument('-H', '-height', type=int, required=True)
    # args = parser.parse_args()
    application.run(host='0.0.0.0', port=9000)
