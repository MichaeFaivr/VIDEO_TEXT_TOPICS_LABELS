from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

USERS_DATABASE_NAME = 'users15_sqlalchemy.db' # better to read from config file

# Initialize the database with SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///{}'.format(USERS_DATABASE_NAME)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
