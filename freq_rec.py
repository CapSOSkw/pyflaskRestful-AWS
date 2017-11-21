from flask import Flask,request
from flask_restful import reqparse, abort, Api, Resource
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from pprint import pprint
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.cluster import KMeans
import json
from collections import OrderedDict
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import geocoder
import random
from time import sleep
from sklearn.externals import joblib
from datetime import datetime
import warnings
from sqlalchemy import create_engine
import pymysql

#Ignore warnings
warnings.filterwarnings('ignore')

#connect to mysql
db = pymysql.connect(host="",port=3306,user="", passwd="",db="bigdata")
cnx = create_engine('mysql+pymysql://USER:PASSWORD@HOST:3306/bigdata',echo=False)
cursor = db.cursor()

app = Flask(__name__)
api = Api(app)

########## All arguments ##########
parser = reqparse.RequestParser()
parser.add_argument('baseId')
parser.add_argument('agencyId')
parser.add_argument("lastName")
parser.add_argument("firstName")
parser.add_argument('pickupDate')
parser.add_argument('pickupTime')
parser.add_argument('pickupAddress')
parser.add_argument('dropOffAddress')
parser.add_argument('pickupCity')
parser.add_argument('dropOffCity')
parser.add_argument('dropOffZipCode')
parser.add_argument('pickupPoint')
parser.add_argument('dropOffPoint')

my_dict = OrderedDict()
res_dict = OrderedDict()
res_dict2 = OrderedDict()

########## Get customer id number from text data based on DB dataset ##########
def get_customer_id(cust_fullname):
    try:
        SQL = "select DISTINCT customer_id from oneyear WHERE cust_fullname=" + "'" + cust_fullname + "'"
        cursor.execute(SQL)
        customer_id = cursor.fetchone()[0]
    except:
        customer_id = -1
    return customer_id

########### Get text data from fleet based on DB dataset ###########
def get_textdata(fleet_text):            #'B88'
    SQL = "select DISTINCT driver_FN, driver_LN, driver_phone from oneyear where fleet=" + "'" + fleet_text + "'"
    cursor.execute(SQL)
    temp = cursor.fetchone()
    driver_FN = temp[0]
    driver_LN = temp[1]
    driver_phone = temp[2]
    return driver_FN, driver_LN, driver_phone

########## Get driver fleet and the number of trips from DB ##########
########## The list returned might contain NULL, it should be removed later on ###########
def get_driver_and_number(user_id):
    SQL = "select D1,D2,D3,D4,D5,D6,D7,D8,D9,D10 from freqReco WHERE USER_ID=" + str(user_id)
    cursor.execute(SQL)
    temp = cursor.fetchone()
    return list(temp)


class my_api(Resource):
    def post(self):
        temp = []
        args = parser.parse_args()
        fleet_list = []
        cache = []
        dict_temp = {}
        
        ######## Restore the post data into python dictionary ##########
        my_dict['user info'] = {'base': args['baseId'], 'company': args['agencyId'].upper(),
                                'cust_FN': args["firstName"].upper(), 'cust_LN': args['lastName'].upper(),
                                'date': args['pickupDate'], 'time': args['pickupTime'],
                                'pickup location': args['pickupAddress'].upper(),
                                'pickup city': args['pickupCity'].upper(),
                                'dropoff location': args['dropOffAddress'].upper(),
                                'dropoff city': args['dropOffCity'].upper(), 'pickup point': args['pickupPoint'],
                                'dropoff point': args['dropOffPoint']}
        
        cust_fullname = my_dict['user info']['cust_FN'] + " " + my_dict['user info']['cust_LN']
        cust_fullname_id = get_customer_id(cust_fullname)
        raw_driver_number = get_driver_and_number(cust_fullname_id)
        
        ########## Remove NULL value ##########
        No_None_driver_number = list(filter(None.__ne__, raw_driver_number))

        ########## Create a dictionary like {"driver fleet": { "Number": "# of trips"} } ###########
        for i in No_None_driver_number:
            raw_x = i.replace("['","").replace("',","").replace("'","").replace("]","").split(" ")
            cache.append(raw_x)
            fleet_list.append(raw_x[0])
            dict_temp[str(raw_x[0])] = {'Number':int(raw_x[1])}

        
        ########## Update dictionary like {"driver fleet": { "Number": "# of trips", "firstName": "****"}, ... }
        for i in fleet_list:
            result_text_data = get_textdata(i)
            dict_temp[str(i)].update({'firstName':result_text_data[0], 'lastName': result_text_data[1], 'phone':result_text_data[2]})
            print(dict_temp[str(i)])

        ########## Return -1 if the customer name fails to match ############
        if cust_fullname_id == -1:
            res_dict2['data'] = []
            res_dict2['result'] = "NO_DATA"
            return res_dict2

        else:
            j = 1
            for i in fleet_list:
                temp.append({'firstName':dict_temp[str(i)]['firstName'], 'lastName':dict_temp[str(i)]['lastName'],
                         'phone': dict_temp[str(i)]['phone'], 'fleetNum':str(i), 'priority':str(j),'NumberOfTrips':dict_temp[str(i)]['Number']})
                j+=1
            res_dict['data'] = temp
            res_dict['result'] = "SUCCESS"

        return res_dict

api.add_resource(my_api, '/')

if __name__=='__main__':
    app.run(debug=False, host='127.0.0.1', port='8888')    # Open localhost with port 8888
