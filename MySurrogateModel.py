import random
import h5py
import copy
import numpy as np
import tensorflow as tf
from tensorflow import keras

from lume_model.models import SurrogateModel
from lume_model.variables import (
    ScalarInputVariable,
    ScalarOutputVariable,
    ImageInputVariable,
    ImageOutputVariable,
)

# TEMPORARY FIX FOR SAME NAME INPUT/OUTPUT VARS
REDUNDANT_INPUT_OUTPUT = ["xmin", "xmax", "ymin", "ymax"]


# Some input/output variables have the same name and must be unique.
# Below are utility functions to fix this:


def apply_temporary_ordering_patch(ordering, prefix):
    # TEMPORARY PATCH FOR INPUT/OUTPUT REDUNDANT VARS
    rebuilt_order = copy.copy(ordering)
    for i, val in enumerate(ordering):
        if val in REDUNDANT_INPUT_OUTPUT:
            rebuilt_order[i] = f"{prefix}_{val}"

    return rebuilt_order


class MySurrogateModel(SurrogateModel):
    """
    Example Usage:
    Load model and use a dictionary of inputs to evaluate the NN.
    """


    # DEFINE INPUT + OUTPUT VARIABLES
    # SUBCLASSING THE lume-model SURROGATE MODEL CLASS ENFORCES THAT THESE ARE DEFINED

    input_variables = {
        "laser_radius": ScalarInputVariable(
            name="laser_radius",
            default=3.47986980e-01,
            units="mm",
            range=[1.000000e-01, 5.000000e-01],
        ),
        "maxb(2)": ScalarInputVariable(
            name="maxb(2)",
            default=4.02751972e-02,
            units="T",
            range=[0.000000e00, 1.000000e-01],
        ),
        "phi(1)": ScalarInputVariable(
            name="phi(1)",
            default=-7.99101687e00,
            units="degrees",
            range=[-1.000000e01, 1.000000e01],
        ),
        "total_charge:value": ScalarInputVariable(
            name="total_charge:value",
            default=0.0,
            units="m",
            range=[0.0, 300],
        ),
        "in_xmin": ScalarInputVariable(
            name="in_xmin",
            default=-3.47874295e-04,
            units="m",
            range=[-4.216e-04, 3.977e-04],
        ),
        "in_ymin": ScalarInputVariable(
            name="in_ymin",
            default=-3.47874295e-04,
            units="m",
            range=[-4.216e-04, 3.977e-04],
        ),
        "in_xmax": ScalarInputVariable(
            name="in_xmax",
            default=-3.47874295e-04,
            units="m",
            range=[-1.117627e-01, 1.120053e-01],
        ),
        "in_ymax": ScalarInputVariable(
            name="in_ymax",
            default=-3.47874295e-04,
            units="m",
            range=[-1.117627e-01, 1.120053e-01],
        ),
        "input_image": ImageInputVariable(
            name="input_image",
            default=np.zeros((50, 50)),
            axis_labels=["x", "y"],
            range=[0, 10],
            x_min = 0,
            y_min = 0,
            x_max = 0,
            y_max = 0
        ),
    }

    output_variables = {
        "end_core_emit_95percent_x": ScalarOutputVariable(
            name="end_core_emit_95percent_x", default=0.0, units="mm-rad"
        ),
        "end_core_emit_95percent_y": ScalarOutputVariable(
            name="end_core_emit_95percent_y", default=0.0, units="mm-rad"
        ),
        "end_core_emit_95percent_z": ScalarOutputVariable(
            name="end_core_emit_95percent_z", default=0.0, units="mm-rad"
        ),
        "end_mean_kinetic_energy": ScalarOutputVariable(
            name="end_mean_kinetic_energy", default=0.0, units="eV"
        ),
        "end_mean_x": ScalarOutputVariable(name="end_mean_x", default=0.0, units="mm"),
        "end_mean_y": ScalarOutputVariable(name="end_mean_y", default=0.0, units="mm"),
        "end_n_particle_loss": ScalarOutputVariable(
            name="end_n_particle_loss", default=0.0, units="number"
        ),
        "end_norm_emit_x": ScalarOutputVariable(
            name="end_norm_emit_x", default=0.0, units="mm-mrad",
        ),
        "end_norm_emit_y": ScalarOutputVariable(
            name="end_norm_emit_y", default=0.0, units="mm-mrad",
        ),
        "end_norm_emit_z": ScalarOutputVariable(
            name="end_norm_emit_z", default=0.0, units="mm-mrad",
        ),
        "end_sigma_x": ScalarOutputVariable(name="end_sigma_x", default=0.0, units="mm"),
        "end_sigma_xp": ScalarOutputVariable(
            name="end_sigma_xp", default=0.0, units="mrad"
        ),
        "end_sigma_y": ScalarOutputVariable(name="end_sigma_y", default=0.0, units="mm"),
        "end_sigma_yp": ScalarOutputVariable(
            name="end_sigma_yp", default=0.0, units="mrad"
        ),
        "end_sigma_z": ScalarOutputVariable(name="end_sigma_z", default=0.0, units="mm"),
        "end_total_charge": ScalarOutputVariable(
            name="end_total_charge", default=0.0, units="C",
        ),
        "out_xmin": ScalarOutputVariable(name="out_xmin", default=0.0, units="m"), # UNUSED
        "out_xmax": ScalarOutputVariable(name="out_xmax", default=0.0, units="m"), # UNUSED
        "out_ymin": ScalarOutputVariable(name="out_ymin", default=0.0, units="m"), # UNUSED
        "out_ymax": ScalarOutputVariable(name="out_ymax", default=0.0, units="m"), # UNUSED
        "x:y": ImageOutputVariable(
            name="x:y", shape=(50, 50), units=["mm", "mm"], axis_labels=["x", "y"]
        ),
    }

    def __init__(self, model_file=None, stock_image_input=None):
        # Save init
        self.model_file = model_file
        self.stock_image_input = stock_image_input
        # Run control
        self.configure()

    def __str__(self):
        if self.type == "scalar":
            s = f"""The inputs are: {', '.join(self.input_names)} and the outputs: {', '.join(self.output_names)}"""
        elif self.type == "image":
            s = f"""The inputs are: {', '.join(self.input_names)} and the output: {', '.join(self.output_names)}"""
        elif self.type == "both":
            s = f"""The inputs are: {', '.join(self.input_names)} and the output: {', '.join(self.output_names)}. Requires image input and output as well"""
        return s

    def configure(self):

        ## Open the File
        with h5py.File(self.model_file, "r") as h5:
            attrs = dict(h5.attrs)
            print("Loaded Attributes successfully")
        self.__dict__.update(attrs)
        self.json_string = self.JSON
        print("Loaded Architecture successfully")

        # load model in thread safe manner
        self.thread_graph = tf.Graph()
        with self.thread_graph.as_default():
            self.model = tf.keras.models.model_from_json(
                self.json_string.decode("utf-8")
            )
            self.model.load_weights(self.model_file)

        # TEMPORARY PATCH FOR INPUT/OUTPUT REDUNDANT VARS
        self.input_ordering = apply_temporary_ordering_patch(self.input_ordering, "in")
        self.output_ordering = apply_temporary_ordering_patch(
            self.output_ordering, "out"
        )

        print("Loaded Weights successfully")
        ## Set basic values needed for input and output scaling
        self.model_value_max = 1  # attrs['upper']
        self.model_value_min = 0  # attrs['lower']

        if self.type == "image":
            self.image_scale = self.output_scales[-1]
            self.image_offset = self.output_offsets[-1]
            self.output_scales = self.output_scales[:-1]
            self.output_offsets = self.output_offsets[:-1]
        elif self.type == "both":
            self.image_scale = self.output_scales[-1]
            self.image_offset = self.output_offsets[-1]
            self.output_scales = self.output_scales[:-1]
            self.output_offsets = self.output_offsets[:-1]
            self.scalar_variables = len(self.input_ordering)
            self.scalar_outputs = len(self.output_ordering)

    def scale_inputs(self, input_values):
        data_scaled = self.model_value_min + (
            (input_values - self.input_offsets[0 : self.scalar_variables])
            * (self.model_value_max - self.model_value_min)
            / self.input_scales[0 : self.scalar_variables]
        )
        return data_scaled

    def scale_outputs(self, output_values):
        data_scaled = self.model_value_min + (
            (output_values - self.output_offsets)
            * (self.model_value_max - self.model_value_min)
            / self.output_scales
        )
        return data_scaled

    def scale_image(self, image_values):
        data_scaled = 2 * ((image_values / self.image_scale) - self.image_offset)
        return data_scaled

    def unscale_image(self, image_values):
        data_unscaled = ((image_values / 2) + self.image_offset) * self.image_scale
        return data_unscaled

    def unscale_inputs(self, input_values):
        data_unscaled = (
            (input_values - self.model_value_min)
            * (self.input_scales[0 : self.scalar_variables])
            / (self.model_value_max - self.model_value_min)
        ) + self.input_offsets[0 : self.scalar_variables]
        return data_unscaled

    def unscale_outputs(self, output_values):
        data_unscaled = (
            (output_values - self.model_value_min)
            * (self.output_scales)
            / (self.model_value_max - self.model_value_min)
        ) + self.output_offsets
        return data_unscaled

    def evaluate_scalar(self, settings):
        if self.type == "image":
            print(
                "To evaluate an image NN, please use the method .evaluateImage(settings)."
            )
            output = 0
        else:
            vec = np.array([[settings[key] for key in self.input_ordering]])
            inputs_scaled = self.scale_inputs(vec)
            model_output = self.model.predict(inputs_scaled)
            model_output = self.unscale_outputs(predicted_outputs)
            output = dict(zip(self.output_ordering, model_output.T))

        return output

    # SUBCLASSING THE SURROGATE MODEL SUBCLASS ENFORCES THAT THIS IS DEFINED
    def evaluate(self, input_variables):
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

        if not "images" in settings:
            settings["image"] = self.stock_image_input

        vec = np.array([settings[key] for key in self.input_ordering])
        image = np.array([settings["image"]])

        inputs_scalar_scaled = self.scale_inputs([vec])
        inputs_image_scaled = self.scale_image(image)

        # call prediction in threadsafe manner
        with self.thread_graph.as_default():
            predicted_output = self.model.predict(
                [inputs_image_scaled, inputs_scalar_scaled]
            )

        predicted_image_scaled = np.array(predicted_output[0])
        predicted_scalars_scaled = predicted_output[1]
        predicted_scalars_unscaled = self.unscale_outputs(predicted_scalars_scaled)

        predicted_extents = predicted_scalars_unscaled[
            :, int(self.scalar_outputs - self.ndim[0]) :
        ]
        predicted_image_unscaled = self.unscale_image(
            predicted_image_scaled.reshape(
                predicted_image_scaled.shape[0], int(self.bins[0] * self.bins[1])
            )
        )

        predicted_output = dict(zip(self.output_ordering, predicted_scalars_unscaled.T))

        predicted_output["extents"] = predicted_extents
        predicted_output["x:y"] = predicted_image_unscaled.reshape(
            (int(self.bins[0]), int(self.bins[1]))
        )

        self.prior_settings = settings

        # PREPARE OUTPUTS WILL FORMAT RETURN VARIABLES
        return self.prepare_outputs(predicted_output)

    def evaluate_image(self, settings, position_scale=10e6):
        vec = np.array([[settings[key] for key in self.input_ordering]])

        inputs_scaled = self.scale_inputs(vec)
        predicted_outputs = self.model.predict(inputs_scaled)
        predicted_outputs_limits = self.unscale_outputs(
            predicted_outputs[:, : self.ndim[0]]
        )
        predicted_outputs_image = self.unscale_image(
            predicted_outputs[:, int(scalar_outputs - 4) :]
        )

        output = predicted_outputs_image.reshape((int(self.bins[0]), int(self.bins[1])))
        return output, extent

    def use_stock_input_image(self):
        data = np.load(self.stock_image_input)
        return data

    def generate_random_input(self):
        if self.type == "both":
            values = np.zeros(len(self.input_ordering))
            for i in range(len(self.input_ordering)):
                values[i] = random.uniform(
                    self.input_ranges[i][0], self.input_ranges[i][1]
                )
            individual = dict(zip(self.input_ordering, values.T))
            individual["image"] = self.use_stock_input_image()
        else:
            values = np.zeros(len(self.input_ordering))
            for i in range(len(self.input_ordering)):
                values[i] = random.uniform(
                    self.input_ranges[i][0], self.input_ranges[i][1]
                )
            individual = dict(zip(self.input_ordering, values.T))

        return individual

    def random_evaluate(self):
        individual = self.generate_random_input()
        if self.type == "scalar":
            random_eval_output = self.evaluate(individual)
            print("Output Generated")
        elif self.type == "image":
            random_eval_output, extent = self.evaluate_image(individual)
            print("Output Generated")
        else:
            random_eval_output = self.evaluate(individual)
            print("Output Generated")
        return random_eval_output

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
                ][0]

        # update image
        extents = list(predicted_output["extents"][0])


        self.output_variables["x:y"].value = predicted_output["x:y"]
        self.output_variables["x:y"].x_min = extents[0]
        self.output_variables["x:y"].x_max = extents[1]
        self.output_variables["x:y"].y_min = extents[2]
        self.output_variables["x:y"].y_max = extents[3]

        return list(self.output_variables.values())
