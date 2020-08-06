import os

os.environ["EPICS_CA_ADDR_LIST"]="0.0.0.0"

import epics


val = epics.caget("test:distgen:r_dist:sigma_xy:value")

if val:
    print("VALUE FOUND")
    print(val)

else:
    print("Value not found.")