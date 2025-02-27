#!/usr/bin/env python3
# chronicler.py

import logging
import time
import os
import sys
import subprocess
from typing import List, Dict, Union


class GitInfo:
    """
    A class to retrieve Git branch, commit hash, and repo state.
    """

    def __init__(self, repo_path: str = "./") -> None:
        """
        Initializes the GitInfo class.

        Args:
            repo_path (str, optional): Path to the Git repository. Defaults to "./".
        """
        self.repo_path = repo_path
        self.branch = None
        self.commit_hash = None
        self.is_clean = None
        self.update_git_info()

    def run_git_command(self, command: List[str]) -> str:
        """
        Runs a Git command in the repository.

        Args:
            command (List[str]): A list of command arguments.

        Returns:
            str: Output from the Git command or an error message if the command fails.
        """
        try:
            return subprocess.check_output(command, cwd=self.repo_path).strip().decode()
        except subprocess.CalledProcessError:
            return "Not a repo"
        except FileNotFoundError:
            return "Not a repo"

    def update_git_info(self) -> None:
        """
        Updates the Git branch, commit hash, and repository state.
        """
        self.branch = self.run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        self.commit_hash = self.run_git_command(["git", "rev-parse", "--short", "HEAD"])
        status_output = self.run_git_command(["git", "status", "--porcelain"])
        
        if status_output == "Not a repo":
            self.is_clean = "Not a repo"
        elif status_output:
            self.is_clean = False
        else:
            self.is_clean = True

    def get_info(self) -> Dict[str, Union[str, bool, None]]:
        """
        Returns Git information as a dictionary.

        Returns:
            Dict[str, Union[str, bool, None]]: Git information including branch, commit hash, and cleanliness status.
        """
        return {
            "branch": self.branch,
            "commit_hash": self.commit_hash,
            "is_clean": self.is_clean,
        }


class Chronicler:
    """
    A logging utility class to initialize and manage logging configuration.
    
    Writes logs to both `stdout` (for AWS CloudWatch) and a timestamped log file in `./logs`.
    """

    def __init__(self, script_path: str) -> None:
        """
        Initializes the Chronicler class and sets up logging for the script.

        Args:
            script_path (str): Path of the script for which logging is being initialized.
        """
        script_name = os.path.splitext(os.path.basename(script_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.log_file = f"./logs/{timestamp}_{script_name}.log"
        os.makedirs("./logs", exist_ok=True)

        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(filename=self.log_file, mode="w"),  # write to file
        ]

        # 20240202_154449 INFO practice.py:56 log_meta| pid:    10629
        log_format = "%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(funcName)s| %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt="%Y%m%d_%H%M%S",
            handlers=handlers,
        )

        ascii_banner = """\n
        >  ┓       • ┓     <
        > ┏┣┓┏┓┏┓┏┓┓┏┃┏┓┏┓ <
        > ┗┛┗┛ ┗┛┛┗┗┗┗┗ ┛  <
        """
        logging.info(ascii_banner)
        logging.info(f"Logging initialized for {script_path}")

        git_info = GitInfo(repo_path="./")
        git_meta = git_info.get_info()
        logging.info(f"git branch: {git_meta['branch']}")
        logging.info(f"git rev-parse HEAD: {git_meta['commit_hash']}")
        logging.info(
            f"git status --porcelain: {'is_clean' if git_meta['is_clean'] else 'is_dirty'}"
        )


def init_chronicler() -> Chronicler:
    """
    Initializes and returns an instance of the Chronicler class.

    Returns:
        Chronicler: An instance of the Chronicler class.
    """
    current_script_path = os.path.abspath(__file__)  # "/myproject/run.py"
    return Chronicler(current_script_path)
