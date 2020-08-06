from subprocess import Popen
import os

def load_jupyter_server_extension(nbapp):
    """serve the bokeh-app directory with bokeh server"""
    env = os.environ.copy()
    env["EPICS_CA_ADDR_LIST"]="0.0.0.0"
    env["EPICS_PVA_ADDR_LIST"]="0.0.0.0"
    Popen(["bokeh", "serve", "surrogate_model_client.py", "--allow-websocket-origin=*"], env=env)