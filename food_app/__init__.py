from flask import Flask
from food_app.config import Config

app = Flask(__name__)
app.config.from_object(Config)

from food_app.app import app
