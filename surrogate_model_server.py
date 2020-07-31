from lume_epics.epics_server import Server
from surrogate_model import Model
from lume_model.utils import load_variables
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

variable_file = "files/surrogate_model_variables.pickle"
input_variables, output_variables = load_variables(variable_file)

prefix = "test"
model_file = "files/SG_CNN_FULL_072820_SurrogateModel.h5"

model_kwargs= {"model_file": model_file, "input_variables": input_variables, "output_variables": output_variables}
server = Server(Model, input_variables, output_variables, prefix, model_kwargs=model_kwargs)

server.start() # monitor = False does not loop in main thread