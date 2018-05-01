import dexUtils as dex
import dexUtilsDraw as dexDraw
import time

# Open the results from the Faster R-CNN and load them into a Python list
# IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180304_132027.dat')
# IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180402_182434.dat')
# IMAGE_LIST = dex.open_box_file('VID_20180304_131842.dat')
# IMAGE_LIST = dex.open_box_file('VID_20180304_125923.dat')
# IMAGE_LIST = dex.open_box_file('VID_20180304_132027.dat')
# IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180304_131842.dat')
# IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180304_125923.dat')
IMAGE_LIST = dex.open_box_file('FHD_JPG_VID_20180304_132119.dat')

# Define the folder where original images are located
# IMAGE_FOLDER = "Frames_VID_20180304_132027"
IMAGE_FOLDER = "Frames_VID_20180402_182434"
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

# initialise item counter an delete_box_list
item_counter = 0
vine_counter = 0

delete_boxes_list = []

start_time = time.time()
# for each image find the successor for each image
for image in IMAGE_LIST:

    # you can't calculate a successor of the last image because it's the last image
    if (IMAGE_LIST.index(image) == (len(IMAGE_LIST) - 1)):
        break

    # if its not the last, check each box
    for box in image:

        # if the box is already in a chain and is labeled you don't need check
        if (box[9] is not None):
            continue

        # calculate all successors of this box and save them in a list --> chain
        chain = dex.get_successor_chain(IMAGE_LIST, box)

        # if the chain contains only one box, you can't predict an successor --> this
        # box should be deleted
        if (len(chain) <= 1):
            delete_boxes_list.append(box)
            continue

        if(box[3] == 'wine'):
            vine_counter += 1
        # ++++PREDICTION PART++++
        # we can't predict till the end, so we if the last 5 boxes in the chain are predicted
        while (not dex.are_last_chain_items_predicted(chain)):

            # get the mean movement/step of the camera, so we can predict where the next item should be
            step = dex.mean_x_movement_in_chain(chain)

            # get the last element of the chain
            last_element = chain[-1]

            # predict ONE chain element with the "step" value and the last chain element
            new_liste = dex.append_list_with_predicted_successor_box(IMAGE_LIST, last_element, step)

            # append_list_with_predicted_successor_box() returns 'None' if the predicted box
            # is outside the image or it tries to predict the successor of the last frame
            # e.g. --> image is 1920x1080 and the box should be at 1960 pixel
            if new_liste is None:
                break
            # if not, overwrite the old list with the new one
            else:
                IMAGE_LIST = new_liste

            # calculate the new chain with the predicted box added to the list
            chain = dex.get_successor_chain(IMAGE_LIST, box)

        # we delete the last 3 predicted elements, because the could be to inaccurate
        # liste = dex.deleteLastPredicted(chain, liste)

        # a chain represent a unique element which is displayed on different frames
        # to mark boxes with the same element all boxes in the chain get the same item_id
        chain = dex.get_successor_chain(IMAGE_LIST, box)
        IMAGE_LIST = dex.set_chain_id(chain, IMAGE_LIST, item_counter)
        # increment the counter
        item_counter += 1

# in the end all boxes with no successor get deleted
IMAGE_LIST = dex.remove_elements_from_list(IMAGE_LIST, delete_boxes_list)

end_time = time.time()
print("Counted vine: " +  str(vine_counter))
print("Time to count Items: " + str(end_time-start_time))

item_counter = dex.count_vine_with_vertical_line(IMAGE_LIST, 0.5)

print(item_counter)

# now each box has a item_id, so we can calculate the distance between
# boxes and map them to positions
position_list = dex.calculate_distance_and_map_to_positions(IMAGE_LIST)

# because each item(wine) is on multiple frames, each item has multiple positions
# so we calculate the mean for each item position
position_list = dex.get_mean_positions(position_list)

# the position_list only contains the position, so we add attributes: class and item_id
position_list = dex.add_attributes_to_position_list(position_list, IMAGE_LIST)

# sort the list ascending by position
position_list.sort(key=lambda x: x[0])

# filter elements with the class 'wine'
wine_position_list = dex.filter_position_list(position_list, "wine")

q_25, q_75, dis_mean = dex.get_quartiles(wine_position_list)

lower_threshold, upper_threshold = dex.get_threshold(q_25, q_75)

# identify wines where the distance is higher than usual --> outliers
outliers = dex.identify_outliers(wine_position_list, dis_mean*1.5)
print(outliers)

# build trace for element position
trace_1 = dexDraw.get_plot_trace_for_positions(position_list)

# append this trace with the outliers
trace_list = dexDraw.get_plot_traces_for_outlier(position_list, outliers, trace_1)

# display the plot
dexDraw.plot_traces(trace_list)

# display all images and save them
dexDraw.draw_boxes_and_save_in_4_threads(IMAGE_LIST, IMAGE_FOLDER)
