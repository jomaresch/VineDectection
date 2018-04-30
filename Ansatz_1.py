import dexUtils as dex
import time

import dexUtilsDraw as dexDraw

# Open the results from the Faster R-CNN and load them into a Python list
IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180304_132027.dat')
IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180402_182434.dat')
# IMAGE_LIST = dex.open_box_file('VID_20180304_131842.dat')
# IMAGE_LIST = dex.open_box_file('VID_20180304_125923.dat')
# IMAGE_LIST = dex.open_box_file('VID_20180304_132027.dat')
IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180304_131842.dat')
# IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180304_125923.dat')

# Define the folder where original images are located
IMAGE_FOLDER = "Frames_VID_20180304_132027"
# IMAGE_FOLDER = "Frames_VID_20180304_131842"
# IMAGE_FOLDER = "Frames_VID_20180304_125923"


# Sort the list by grouping boxes which are located on the same frame
IMAGE_LIST = dex.sort_and_group_list(IMAGE_LIST)

# Delete all boxes with a confidence lower than 0.4
IMAGE_LIST = dex.drop_boxes_with_low_confidence(IMAGE_LIST, 0.4)

# If two boxes mark the same wine, only one should be used
IMAGE_LIST = dex.clean_list_from_overlapping_boxes(IMAGE_LIST)

# Delete all boxes which are not in the center of the image --> not the row we are looking for
IMAGE_LIST = dex.delete_not_centred_boxes(IMAGE_LIST)

start = time.time()
# initialise item counter an delete_box_list
item_counter = dex.count_wine_with_vertical_line(IMAGE_LIST, 0.5)
end = time.time()
print(end-start)

print(item_counter)