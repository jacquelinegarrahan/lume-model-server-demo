
from bokeh.io import output_notebook, show
from bokeh.layouts import column, row
from bokeh import palettes

from bokeh.io import curdoc
#from bokeh.resources import INLINE#
from lume_model.utils import load_variables
from lume_epics.client.widgets.plots import ImagePlot
from lume_epics.client.widgets.controls import build_sliders
import os.environ["EPICS_CA_ADDR_LIST"]="0.0.0.0"

from lume_epics.client.controller import Controller

prefix = "test" # Prefix used by our server
variable_filename =  "files/surrogate_model_variables_2.pickle"
input_variables, output_variables = load_variables(variable_filename)

controller = Controller("ca")

inputs = [
          input_variables["distgen:r_dist:sigma_xy:value"], 
          input_variables["distgen:t_dist:length:value"],
#          input_variables["distgen:total_charge:value"],
          input_variables["SOL1:solenoid_field_scale"],
          input_variables["CQ01:b1_gradient"],
          input_variables["SQ01:b1_gradient"],
          input_variables["L0A_phase:dtheta0_deg"],
    #      input_variables["L0A_scale:voltage"],
          input_variables["end_mean_z"]
        ]

sliders = build_sliders(inputs, controller, prefix)

# create image plot
output_variables = [output_variables["x:y"]]
image_plot = ImagePlot(output_variables, controller, prefix)


# build the plot using specific bokeh palette
pal = palettes.viridis(256)
image_plot.build_plot(palette=pal)

# render
curdoc().title = "Demo App"
curdoc().add_root(
            row(
                column(sliders, width=350), column(image_plot.plot)
                ) 
    )

curdoc().add_periodic_callback(image_plot.update, 250)