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
import operator

#Ignore warnings
warnings.filterwarnings('ignore')


#connect to mysql
db = pymysql.connect("localhost", "root", "0702","mytest")
cnx = create_engine('mysql+pymysql://root:0702@localhost/mytest',echo=False)
cursor = db.cursor()


def get_fleet(id):
    SQL = "select fleet from oneyear where fleet_id=" + str(id)
    cursor.execute(SQL)
    temp = cursor.fetchone()[0]
    return temp

def get_username(id):
    SQL = "select cust_fullname from oneyear WHERE customer_id=" + str(id)
    cursor.execute(SQL)
    temp = cursor.fetchone()[0]
    return temp

def get_customer_id(cust_fullname):
    try:
        SQL = "select DISTINCT customer_id from oneyear WHERE cust_fullname=" + "'" + cust_fullname + "'"
        cursor.execute(SQL)
        customer_id = cursor.fetchone()[0]
    except:
        customer_id = -1
    print(customer_id)
    return customer_id

driver_data = []
user_data = []

for i in pd.read_sql("select fleet_id from oneyear", cnx).values:
    driver_data.append(int(i))

for i in pd.read_sql("select customer_id from oneyear", cnx).values:
    user_data.append(int(i))

set_user_data = set(user_data)

dict = OrderedDict()

for user_id in list(set_user_data):
    temp_dict = OrderedDict()
    cache = []
    SQL = "select fleet_id from oneyear WHERE customer_id=" + str(user_id)
    cursor.execute(SQL)
    temp_fleet_id = cursor.fetchall()
    for item in temp_fleet_id:
        cache.append(item[0])
        
    set_cache = set(cache)

    for i in list(set_cache):
        temp_dict[str(get_fleet(i))] = cache.count(int(i))
        
    temp_dict = sorted(temp_dict.items(), key=operator.itemgetter(1), reverse=True) # Sort dictionary by values
    dict[str(get_username(user_id))] = temp_dict
    

df = pd.DataFrame.from_dict(dict, orient='index')
df['USER'] = df['USER'].apply(lambda x: re.sub("\s\s+"," ",x).strip())
df['USER_ID'] = df['USER'].apply(lambda x:get_customer_id(x))
df.to_csv('*****.csv')
