from flask import Flask
from chronicler import Chronicler
import os
import logging as l

# pass current script name
current_script_path = os.path.abspath(__file__)  # "/myproject/run.py"
chronicler = Chronicler(current_script_path)

app = Flask(__name__)

@app.route("/")
def hello_world():
    printed_string = "hello world"
    l.info(printed_string)
    return printed_string

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
