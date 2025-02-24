# tests/test_chronicler.py

import os
from src.chronicler import init_chronicler, GitInfo

def test_init_chronicler():
    # Initialize Chronicler
    chronicler = init_chronicler()
    # Check that logs folder is created
    assert os.path.exists("./logs"), "Logs folder does not exist"
    # Check that a log file is assigned
    assert chronicler.log_file.endswith(".log"), "No *.log found."

def test_git_info():
    git_info = GitInfo()
    git_meta = git_info.get_info()
    # Basic checks on returned info dictionary
    assert "branch" in git_meta
    assert "commit_hash" in git_meta
    assert ("is_clean" or "is_dirty") in git_meta
