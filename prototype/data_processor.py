# handle relative directory imports for chronicler
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import chronicler
import logging as l
from chronicler import Chronicler

# pass current script name
current_script_path = os.path.abspath(__file__)  # "/myproject/run.py"
chronicler = Chronicler(current_script_path)
