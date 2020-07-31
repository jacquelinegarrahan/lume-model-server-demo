from lume_epics.epics_server import Server
from surrogate_model import Model
from lume_model.utils import load_variables
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-monitor', help='Run server without monitor.', action='store_false', dest="monitor", default=True)
args = parser.parse_args()
monitor = args.monitor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

variable_file = "files/surrogate_model_variables.pickle"
input_variables, output_variables = load_variables(variable_file)

prefix = "test"
model_file = "files/SG_CNN_FULL_072820_SurrogateModel.h5"

model_kwargs= {"model_file": model_file, "input_variables": input_variables, "output_variables": output_variables}
server = Server(Model, input_variables, output_variables, prefix, model_kwargs=model_kwargs, protocols=["pva"])

server.start(monitor=monitor) # monitor = False does not loop in main thread