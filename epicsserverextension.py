
from subprocess import Popen


def load_jupyter_server_extension(nbapp):
    """Start python server"""
    Popen(["python", "surrogate_model_server.py", "--no-monitor"])