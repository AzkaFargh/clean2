# app/views.py

from flask import Flask, request, jsonify
from app.models import Image
import mysql.connector
from app.prediction import predict_svm
app = Flask(__name__)


db_config = {
    'user': 'root',
    'password': 'admin',
    'host': 'localhost',
    'database': 'db_melon',
    'port': '5000'  # Pastikan port disesuaikan dengan port MySQL yang digunakan
}

def connect_to_database():
    try:
        conn = mysql.connector.connect(**db_config)
        print("Connected to MySQL database")
        return conn
    except mysql.connector.Error as err:
        print("Error connecting to MySQL:", err)
        return None

