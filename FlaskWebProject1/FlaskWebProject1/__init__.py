"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)
import os
os.environ['PYTHONUTF8'] = '1'

import FlaskWebProject1.views
