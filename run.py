from chronicler import Chronicler
import os
import logging as l

# pass current script name
current_script_path = os.path.abspath(__file__)  # "/myproject/run.py"
chronicler = Chronicler(current_script_path)


ascii_banner = """\n
        hello there
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n
"""
l.info(ascii_banner)

