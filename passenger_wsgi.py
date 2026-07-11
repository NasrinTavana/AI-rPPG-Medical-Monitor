import sys
import os

# مسیر فعلی 
sys.path.insert(0, os.path.dirname(__file__))

from server import app as application