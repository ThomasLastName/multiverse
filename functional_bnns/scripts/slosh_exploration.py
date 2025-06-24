#
# ~~~ Get the data
import numpy as np
from bnns.data.slosh_70_15_15 import out_np, coords_np, inputs_np

#
# ~~~ Verify that the first feature is "Sea level rise in the year 2100"
sea_level_rise = inputs_np[:, 0]
print("")
print("First feaure: sea level rise:")
print(
    f"    In our dataset, the lower value is {sea_level_rise.min()}, whereas the paper states a lower value of -20"
)
print(
    f"    In our dataset, the upper value is {sea_level_rise.max()}, whereas the paper states an upper value of 350"
)

#
# ~~~ Verify that the second feature is "Heading of the eye of the storm when it made landfall"
headings = inputs_np[:, 1]
headings = np.where(
    headings < 60, headings + 360, headings
)  # ~~~ to make the headings contiguous, we added 360 to all headings less than 60
print("")
print("Second feaure: heading upon landfall:")
print(
    f"    In our dataset, the lower value is {headings.min()}, whereas the paper states a lower value of 204.0349"
)
print(
    f"    In our dataset, the upper value is {headings.max()}, whereas the paper states an upper value of 384.02444"
)

#
# ~~~ Verify that the third feature is "Velocity of the eye of the storm when it made landfall"
velocity = inputs_np[:, 2]
print("")
print("Third feaure: velocity upon landfall:")
print(
    f"    In our dataset, the lower value is {velocity.min()}, whereas the paper states a lower value of 0"
)
print(
    f"    In our dataset, the upper value is {velocity.max()}, whereas the paper states an upper value of 40"
)

#
# ~~~ Verify that the fourth feature is "Minimum air pressure of the storm when it made landfall"
min_air_p = inputs_np[:, 3]
print("")
print("Fourth feaure: minimum air pressure upon landfall:")
print(
    f"    In our dataset, the lower value is {min_air_p.min()}, whereas the paper states a lower value of 930"
)
print(
    f"    In our dataset, the upper value is {min_air_p.max()}, whereas the paper states an upper value of 980"
)

#
# ~~~ Verify that the fifth feature is "Latitude of the storm when it made landfall"
latitude = inputs_np[:, 4]
print("")
print("Fifth feaure: latitude upon landfall:")
print(
    f"    In our dataset, the lower value is {latitude.min()}, whereas the paper states a lower value of 38.32527"
)
print(
    f"    In our dataset, the upper value is {latitude.max()}, whereas the paper states an upper value of 39.26811"
)

print("")
