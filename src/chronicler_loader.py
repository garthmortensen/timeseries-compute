# because chronicler is in a parent directory, i have to jump through a few hoops to import it
# i carved these hoops out into this helper function so i can just call it in my scripts

import os
from chronicler import Chronicler
# import sys

# # handles relative logger dir
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def init_chronicler():
    current_script_path = os.path.abspath(__file__)  # "/myproject/run.py"
    return Chronicler(current_script_path)
