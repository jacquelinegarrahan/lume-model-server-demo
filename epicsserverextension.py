from subprocess import Popen


def load_jupyter_server_extension(nbapp):
    Popen(["python", "surrogate_model_server.py"])