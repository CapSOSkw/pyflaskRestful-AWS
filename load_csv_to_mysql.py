import pymysql
import pandas as pd
from sqlalchemy import create_engine
from pandas.io import sql
from collections import defaultdict

db = pymysql.connect(host="",port=3306,user="", passwd="",db="bigdata")

#Insert pandas dataframe into mysql
cnx = create_engine('mysql+pymysql://HOST:PASSWORD@HOST:3306/bigdata')

data = pd.read_csv('*****.csv',encoding='ISO-8859-1')

data.to_sql(name='MYSQL-TABLE-NAME', con=cnx, if_exists='replace', index=False,)
