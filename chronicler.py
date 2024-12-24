import logging
import time
import os
import sys


class Chronicler:
    """
    A logging utility class to initialize and manage logging config.

    Writes to both `stdout` (for  aws cloudwatch) and a timestamped log file in `./logs`.

    Attributes:
        log_file (str): The path to the log file being written.
    
    Methods:
        __init__(script_path): Initializes the logging  config.
    """
    def __init__(self, script_path):
        # for log filename and print filepath
        script_name = os.path.splitext(os.path.basename(script_path))[0]

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = f"./logs/{timestamp}_{script_name}.log"

        # ensure exists
        os.makedirs("./logs", exist_ok=True)

        handlers = [
            # aws cloudwatch monitors stdout and stderr and can auto pick them up
            logging.StreamHandler(sys.stdout),  
            logging.FileHandler(filename=self.log_file, mode="w"),  # write to file
        ]

        # 20240202_154449 INFO practice.py:56 log_meta| pid:    10629
        log_format = (
            "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(funcName)s| %(message)s"
        )
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt="%Y%m%d_%H%M%S",
            handlers=handlers,
        )

        logging.info(f"Logging initialized for {script_path}")
