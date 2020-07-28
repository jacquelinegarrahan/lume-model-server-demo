import random
import h5py
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

from lume_model.models import SurrogateModel
from lume_model.utils import load_variables
from lume_model.variables import (
    ScalarInputVariable,
    ScalarOutputVariable,
    ImageInputVariable,
    ImageOutputVariable,
)

 

class BaseModel:
    def __init__(self, *, model_file, input_variables, output_variables):

        # Save init
        self.model_file = model_file
        self.input_variables = input_variables
        self.output_variables = output_variables

        # Load attributes from file
        with h5py.File(self.model_file, "r") as h5:
            self.attrs = dict(h5.attrs)
            print("Loaded Attributes successfully")

        # Load model etc.
        self.json_string = self.attrs["JSON"]

        print("Loaded Architecture successfully")

        # load model in thread safe manner
        self.thread_graph = tf.Graph()
        with self.thread_graph.as_default():
            self.model = tf.keras.models.model_from_json(
                self.json_string.decode("utf-8")
            )
            self.model.load_weights("files/CNN_060420_SurrogateModel.h5")

        print("Loaded Weights successfully")

    # SUBCLASSING THE SURROGATE MODEL SUBCLASS ENFORCES THAT THIS IS DEFINED
    def evaluate(self, input_variables):

        # ALL BELOW -> ONLINE SURROGATE
        settings = {}

        input_variables = {input_variable.name: input_variable for input_variable in input_variables}

        for variable_name in self.input_ordering:
            if variable_name in input_variables:
                if input_variables[variable_name].value is not None:
                    settings[variable_name] = input_variables[variable_name].value

                else:
                    settings[variable_name] = input_variables[variable_name].default

            else:
                settings[variable_name] = self.input_variables[variable_name].default

        # MUST IMPLEMENT A SETTINGS FORMAT FROM DICT -> MODEL INPUT
        formatted_input = self.format_input(settings)
        # __________________________________________

        # call prediction in threadsafe manner
        with self.thread_graph.as_default():
            model_output = self.model.predict(formatted_input)

        # MUST IMPLEMENT AN OUTPUT -> DICT METHOD
        output = self.parse_output(model_output)

        # PREPARE OUTPUTS WILL FORMAT RETURN VARIABLES (DICT-> VARIABLES)
        return self.prepare_outputs(output)


    # CAN BE IN BASE
    def random_evaluate(self):
        random_input = copy.deepcopy(self.input_variables)
        for variable in self.input_ordering:
            if self.input_variables[variable].variable_type == "scalar":
                random_input[variable].value = np.random.uniform(
                    self.input_variables[variable].range[0],
                    self.input_variables[variable].range[1],
                )

            else:
                random_input[variable].value = self.input_variables[variable].default

        pred = self.evaluate(random_input)

        return pred


    # CAN BE IN BASE
    def prepare_outputs(self, predicted_output):
        """
        Prepares the model outputs to be served so no additional
        manipulation happens in the OnlineSurrogateModel class

        Parameters
        ----------
        model_outputs: dict
            Dictionary of output variables to np.ndarrays of outputs

        Returns
        -------
        dict
            Dictionary of output variables to respective scalars
            (reduced dimensionality of the numpy arrays)

        Note
        ----
        This could also be accomplished by reshaping/sampling the arrays
        in scaling.
        """
        for variable in self.output_variables.values():
            if variable.variable_type == "scalar":
                self.output_variables[variable.name].value = predicted_output[
                    variable.name
                ]

            elif variable.variable_type == "image":
                self.output_variables[variable.name].value = predicted_output[
                    variable.name
                ].reshape(variable.shape)

                # update limits
                if self.output_variables[variable.name].x_min_variable:
                    self.output_variables[variable.name].x_min = predicted_output[
                        self.output_variables[variable.name].x_min_variable
                    ]

                if self.output_variables[variable.name].x_max_variable:
                    self.output_variables[variable.name].x_max = predicted_output[
                        self.output_variables[variable.name].x_max_variable
                    ]

                if self.output_variables[variable.name].y_min_variable:
                    self.output_variables[variable.name].y_min = predicted_output[
                        self.output_variables[variable.name].y_min_variable
                    ]

                if self.output_variables[variable.name].y_max_variable:
                    self.output_variables[variable.name].y_max = predicted_output[
                        self.output_variables[variable.name].y_max_variable
                    ]

        return list(self.output_variables.values())



class ScaledModel(BaseModel):
    """
    Example Usage:
    Load model and use a dictionary of inputs to evaluate the NN.
    """

    def __init__(self, *, model_file=None, input_variables=None, output_variables=None):
        super().__init__(model_file=model_file, input_variables=input_variables, output_variables=output_variables)

        # CHECK FOR ATTRIBUTES

        # Collect attributes
        self.input_ordering = self.attrs["input_ordering"]
        self.output_ordering = self.attrs["output_ordering"]

        self.input_scales = {
            self.input_ordering[i]: self.attrs["input_scales"][i]
            for i in range(len(self.input_ordering))
        }
        self.output_scales = {
            self.output_ordering[i]: self.attrs["output_scales"][i]
            for i in range(len(self.output_ordering))
        }
        self.input_offsets = {
            self.input_ordering[i]: self.attrs["input_offsets"][i]
            for i in range(len(self.input_ordering))
        }
        self.output_offsets = {
            self.output_ordering[i]: self.attrs["output_offsets"][i]
            for i in range(len(self.output_ordering))
        }

        ## Set basic values needed for input and output scaling
        self.model_value_max = self.attrs["upper"]
        self.model_value_min = self.attrs["lower"]

    def scale_inputs(self, input_values):
        data_scaled = {}

        for i, input_variable in enumerate(self.input_ordering):

            if self.input_variables[input_variable].variable_type == "scalar":
                data_scaled[input_variable] = self.model_value_min + (
                    (input_values[input_variable] - self.input_offsets[input_variable])
                    * (self.model_value_max - self.model_value_min)
                    / self.input_scales[input_variable]
                )

            elif self.input_variables[input_variable].variable_type == "image":
                data_scaled[input_variable] = (self.model_value_max - self.model_value_min) * (
                    (input_values[input_variable] / self.input_scales[input_variable])
                    - self.input_offsets[input_variable]
                )

                data_scaled[input_variable] = data_scaled[input_variable].reshape(
                    self.input_variables[input_variable].shape
                )

        return data_scaled

    def unscale_outputs(self, output_values):
        unscaled_outputs = {}
        for output_variable in output_values:

            # Scale scalar variable
            if self.output_variables[output_variable].variable_type == "scalar":
                unscaled_outputs[output_variable] = (
                    output_values[output_variable]
                    * self.output_scales[output_variable]
                    / (self.model_value_max - self.model_value_min)
                    + self.output_offsets[output_variable]
                )

            # Scale image variable
            elif self.output_variables[output_variable].variable_type == "image":
                unscaled_image = (
                    (output_values[output_variable] / (self.model_value_max - self.model_value_min))
                    + self.output_offsets[output_variable]
                ) * self.output_scales[output_variable]

                # Reshape image
                unscaled_outputs[output_variable] = unscaled_image.reshape(
                    self.output_variables[output_variable].shape
                )

        return unscaled_outputs


class MyModel(ScaledModel):
    def format_input(self, input_dictionary):
        # scale inputs
        input_dictionary = self.scale_inputs(input_dictionary)

        image = input_dictionary["input_image"].reshape(1, 50, 50, 1)
        scalar_inputs = np.array([
            input_dictionary["laser_radius"],
            input_dictionary["maxb(2)"],
            input_dictionary["phi(1)"],
            input_dictionary["total_charge:value"],
            input_dictionary["in_xmin"],
            input_dictionary["in_xmax"],
            input_dictionary["in_ymin"],
            input_dictionary["in_ymax"]
            ]).reshape((1,8))

        model_input = [image, scalar_inputs]
        return  model_input


    def parse_output(self, model_output):
        parsed_output = {}
        parsed_output["x:y"] = model_output[0]
        parsed_output.update(dict(zip(self.output_ordering[1:], model_output[1][0].T)))

        # unscale
        parsed_output = self.unscale_outputs(parsed_output)
        return parsed_output


