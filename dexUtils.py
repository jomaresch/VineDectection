from PIL import Image, ImageDraw
import numpy as np
import pickle
import sys
import numpy
import copy
import PIL
import PIL.ImageFont as ImageFont
from threading import Thread
import itertools


# VU.draw_box(image_list[1]).save('1.png')

# def draw_box(box_list):
#     color_1 = (0, 255, 0)
#     color_2 = (255, 0, 0)
#     color_3 = (0, 0, 255)
#
#     image_path = box_list[0][0]
#     image = Image.open('images/'+image_path)
#     draw = ImageDraw.Draw(image)
#     color = color_3
#     for box in box_list:
#         if box[3] == 'bar':
#             color = color_1++
#         elif box[3] == 'wine':
#             color = color_2
#         else:
#             color = color_3
#         if(box[8]):
#             color = color_3
#
#         if box[9] is None:
#             text = 'None'
#         else:
#             text = str(box[9])
#
#         text = box[3]+ ': ' + text
#         draw.line((box[4], box[6], box[5], box[6]), fill=color, width=6)
#         draw.line((box[4], box[7], box[5], box[7]), fill=color, width=6)
#         draw.line((box[4], box[6], box[4], box[7]), fill=color, width=6)
#         draw.line((box[5], box[6], box[5], box[7]), fill=color, width=6)
#         draw.text((box[4], box[6]), text, fill='black', font= ImageFont.truetype('arial.ttf', 32))
#     return image
#
# def drawConnections(b1,b2):
#     image_path  = b1[0]
#     image = Image.open('imagesnew/' + image_path)
#     draw = ImageDraw.Draw(image)
#     draw.line((b1[4], b1[6], b2[4], b2[6]), fill=(0,0,0), width=6)
#     draw.line((b1[4], b1[7], b2[4], b2[7]), fill=(0,0,0), width=6)
#     draw.line((b1[5], b1[6], b2[5], b2[6]), fill=(0,0,0), width=6)
#     draw.line((b1[5], b1[7], b2[5], b2[7]), fill=(0,0,0), width=6)
#     return image
#
#
# def draw_box2(image,box_list):
#     color_1 = (0, 255, 0)
#     color_2 = (255, 0, 0)
#     color_3 = (0, 0, 255)
#
#     draw = ImageDraw.Draw(image)
#     color = color_3
#     for box in box_list:
#         if box[3] == 'bar':
#             color = color_1
#         elif box[3] == 'wine':
#             color = color_2
#         draw.line((box[4], box[6], box[5], box[6]), fill=color, width=6)
#         draw.line((box[4], box[7], box[5], box[7]), fill=color, width=6)
#         draw.line((box[4], box[6], box[4], box[7]), fill=color, width=6)
#         draw.line((box[5], box[6], box[5], box[7]), fill=color, width=6)
#     return image

def get_overlap(bb1, bb2):
    assert bb1[4] < bb1[5]
    assert bb1[6] < bb1[7]
    assert bb2[4] < bb2[5]
    assert bb2[6] < bb2[7]

    x_left = max(bb1[4], bb2[4])
    y_top = max(bb1[6], bb2[6])
    x_right = min(bb1[5], bb2[5])
    y_bottom = min(bb1[7], bb2[7])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[5] - bb1[4]) * (bb1[7] - bb1[6])
    bb2_area = (bb2[5] - bb2[4]) * (bb2[7] - bb2[6])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0

    return iou


def drop_boxes_with_low_confidence(list, confidence):
    newlist = []
    for image in list:
        # x[10] == confidence of the box
        newlist.append([x for x in image if x[10] >= confidence])
    return newlist


def open_box_file(name):
    with open("imageData/" + name, 'rb') as fp:
        image_list = pickle.load(fp)
    return image_list


def saveBoxFile(list, name):
    with open("imageData/" + name, 'wb') as fp:
        pickle.dump(list, fp)


def meanDistance(box_1, box_2):
    mySum = 0
    for x in range(4, 8):
        mySum = mySum + abs(box_1[x] - box_2[x])
    return mySum / 4


def mean_x_distance(box_1, box_2):
    m1 = box_1[4] + box_1[5]
    m1 = round(m1 / 2)
    m2 = box_2[4] + box_2[5]
    m2 = round(m2 / 2)

    return abs(m1 - m2)


def mean_x_distance_without_abs(box_1, box_2):
    m1 = box_1[4] + box_1[5]
    m1 = round(m1 / 2)
    m2 = box_2[4] + box_2[5]
    m2 = round(m2 / 2)

    return m2 - m1


def meanXDistanceBox(box_1, box_2):
    m1 = box_1.minX + box_1.maxX
    m1 = round(m1 / 2)
    m2 = box_2.minX + box_2.maxX
    m2 = round(m2 / 2)

    return m1 - m2


def meanYDistance(box_1, box_2):
    m1 = box_1[6] + box_1[7]
    m1 = round(m1 / 2)
    m2 = box_2[6] + box_2[7]
    m2 = round(m2 / 2)

    return abs(m1 - m2)


def statusBar(i, max):
    sys.stdout.write('\r')
    sys.stdout.write(str(round(i * 100 / max, 2)) + ' %')
    sys.stdout.flush()


def get_box_area(box):
    x = abs(box[5] - box[4])
    y = abs(box[7] - box[6])
    return x * y


def clean_list_from_overlapping_boxes(list):
    for index_1, image in enumerate(list):
        for index_2, box in enumerate(image):
            for index_3, box2 in enumerate(image):
                overlap_percentage = get_overlap(box, box2)
                if (0.2 < overlap_percentage < 1.0 and box[3] == box2[3]):
                    # pick the lager box
                    if (get_box_area(box) > get_box_area(box2)):
                        del (list[index_1][index_3])
                    else:
                        del (list[index_1][index_2])
                    break
    return list


def BoxToList(box):
    return [box.image, box.widht, box.height, box.label, box.minX, box.maxX, box.minY, Box.maxY]


def ListToBox(list):
    return Box(list[0], list[1], list[2], list[3], list[4], list[5], list[6], list[7])


def getSuccessor(list, element):
    # list with elemtent and predecessor
    for pair in list:
        if (element == pair[1]):
            return pair[0]


def getAllSuccessors(list, startElemment):
    element = startElemment
    listWithSameElements = []
    listWithSameElements.append(startElemment)
    while (getSuccessor(list, element) is not None):
        listWithSameElements.append(getSuccessor(list, element))
        element = getSuccessor(list, element)
    return listWithSameElements


def getMeanStep(list, image_name):
    templist = []
    dis = []
    for element in list:
        if (element[0][0] == image_name):
            templist.append(element)
    for element in templist:
        dis.append(mean_x_distance(element[0], element[1]))

    if (len(dis) == 0):
        return 0.0
    else:
        return round(numpy.mean(dis))


def getMatchCount(list, image_name):
    count = 0
    for element in list:
        if (element[0][0] == image_name):
            count = count + 1
    return count


def getImageAttributes(matchlist, boxlist):
    attributeList = []

    for frame in boxlist:
        frameName = frame[0][0]
        boxCount = len(frame)
        attributeList.append([frameName,
                              getMeanStep(matchlist, frameName),
                              boxCount,
                              getMatchCount(matchlist, frameName)])

    return attributeList


def predictMissingBoxes(matchlist, boxlist):
    imageAttributes = getImageAttributes(matchlist, boxlist)
    newBoxList = copy.deepcopy(boxlist)
    for index, image in enumerate(boxlist):
        if (index == 0):
            continue
        for box in image:
            if (hasPredecessor(box, matchlist)):
                continue
            else:
                step = getMeanStep(matchlist, box[0])
                predecessor = boxlist[index - 1][0]
                predictedBox = predictPredecessor(box, step, predecessor)
                if (predictedBox is None):
                    continue
                else:
                    for index2, frame in enumerate(newBoxList):
                        if (frame[0][0] == predictedBox[0]):
                            newBoxList[index2].append(predictedBox)
    return newBoxList


def createMatchList(list):
    newList = []
    for index_image, image in enumerate(list):
        if (index_image == 0):
            continue
        for index_box_now, box_now in enumerate(list[index_image]):
            distances = []
            saved_inds = []
            for index_box_prev, box_prev in enumerate(list[index_image - 1]):
                if (box_now[3] == box_prev[3]):
                    distances.append(mean_x_distance(box_now, box_prev))
                    saved_inds.append(index_box_prev)
            if (len(distances) == 0):
                continue
            min_dis = min(distances)
            if (min_dis < 150):
                newList.append([box_now, list[index_image - 1][saved_inds[distances.index(min_dis)]]])
    return newList


def hasPredecessor(box, matchlist):
    for match in matchlist:
        if (match[0] == box):
            return True
    return False


def predictPredecessor(box, step, predecessor):
    if (box[4] - step < 200 or box[5] - step > box[1] - 200):
        return None
    else:
        print("predicted" + predecessor[0])
        return [predecessor[0], predecessor[1], predecessor[2], box[3], box[4] - step, box[5] - step, box[6], box[7],
                True]


def printList(list):
    print()
    for i in list:
        print(i)
    print()


def resizeImage(image_name, width, height):
    img = Image.open('old/' + image_name)
    img = img.resize((width, height), PIL.Image.LANCZOS)
    img.save('new/' + image_name)


def sort_and_group_list(boxes):
    image_list = []
    image = []

    image.append(boxes[0])
    for box in boxes:
        # decode 1,2,3 to class names
        # box[3] == class name
        if box[3] == 1 or box[3] == 3:
            box[3] = 'wine'
        elif box[3] == 2:
            box[3] = 'bar'

        # box[0] == name of the image
        if (box[0] == image[0][0]):
            image.append(box)
        else:
            image_list.append(image)
            image = []
            image.append(box)

    image_list.append(image)
    del image_list[0][0]
    return image_list


def delete_not_centred_boxes(list):
    # and if the box is too close to the borders
    # [4]  == x_min
    # [5]  == x_max
    # [6]  == y_min
    # [7]  == y_max
    cleaned_list = []

    for image in list:
        for box in image:
            if (box[6] < (box[2] / 2) < box[7]) and (box[4] > (box[1] * 0.04)) and (box[5] < (box[1] * 0.96)):
                cleaned_list.append(box)

    return sort_and_group_list(cleaned_list)


def get_successor_element(element, list, index):

    # last element
    if (index + 1 == len(list)):
        return None
    else:
        next_image = list[index + 1]
    overlaps = []
    distances = []
    boxes = []
    for box in next_image:
        # if boxes have the same class and box has no ItemID
        if (box[3] == element[3] and box[9] is None):
            # add the distance to the list
            distances.append(mean_x_distance(box, element))
            # add overlaps to the list
            overlaps.append(get_overlap(box, element))
            boxes.append(box)

    if overlaps:
        if (max(overlaps) > 0.0):
            # pick the box with the biggest overlap
            return boxes[(overlaps.index(max(overlaps)))]

    if distances:
        min_dis = min(distances)
        if (min_dis < 150):
            # else pick the box with the lowest distance only if lower than 150px
            return boxes[(distances.index(min_dis))]

    return None


def mean_x_movement_in_chain(chain):
    distances = []
    for index, box in enumerate(chain):
        # last element
        if (index + 1 == len(chain)):
            continue
        else:
            distances.append(mean_x_distance(box, chain[index + 1]))

    # return mean of distances with numpy
    return int(round(np.mean(distances)))


def get_image_index(box, list):
    for image in list:
        if box in image:
            return list.index(image)


def append_list_with_predicted_successor_box(list, previous_Element, step):
    index = get_image_index(previous_Element, list)
    # if the predicted would be last --> break
    if (index + 1 == len(list)):
        return None
    # else use the values of the predecessor
    v0 = list[index + 1][0][0]
    v1 = previous_Element[1]
    v2 = previous_Element[2]
    v3 = previous_Element[3]
    # adjust the x values with the step
    v4 = previous_Element[4] - step
    v5 = previous_Element[5] - step
    v6 = previous_Element[6]
    v7 = previous_Element[7]
    if (v4 <= 0):
        return None

    item = [v0, v1, v2, v3, v4, v5, v6, v7, True, None]
    list[index + 1].append(item)
    return list


def appendListWithPredictedPredecessorValue(list, nextElement, step):
    index = get_image_index(nextElement, list)
    if (index == 0):
        return None
    v0 = list[index - 1][0][0]
    v1 = nextElement[1]
    v2 = nextElement[2]
    v3 = nextElement[3]
    v4 = nextElement[4] + step
    v5 = nextElement[5] + step
    v6 = nextElement[6]
    v7 = nextElement[7]
    v9 = nextElement[9]
    if (v5 > v1):
        if (v5 > v1 + 40):
            return None
        else:
            v5 = v1
    else:
        list[index - 1].append([v0, v1, v2, v3, v4, v5, v6, v7, True, v9])
        return list


def addValue_8_and_9_ToBox(list):
    list2 = []
    for image in list:
        image_new = []
        for box in image:
            image_new.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], False, None])
        list2.append(image_new)

    return list2


def get_successor_chain(list, box):
    index = get_image_index(box, list)
    successor_chain = []
    predecessor = box
    for index_box, image in enumerate(list):
        if index_box < index:
            continue
        successor_element = get_successor_element(predecessor, list, index_box)
        # if there is no successor_element
        if successor_element is None:
            break
        # else append successor_element to the chain
        successor_chain.append(successor_element)
        predecessor = successor_element
    # insert the initial box to the chain
    successor_chain.insert(0, box)
    return successor_chain


def are_last_chain_items_predicted(chain):
    # too short list
    if (len(chain) < 5):
        return False

    if (chain[-1][8] and chain[-2][8] and chain[-3][8] and chain[-4][8] and chain[-5][8]):
        return True

    return False


def deleteLastPredicted(chain, list):
    while (chain[-1][8]):
        imageIndex = get_image_index(chain[-1], list)
        list[imageIndex].remove(chain[-1])
        chain.remove(chain[-1])
    return list


def set_chain_id(chain, list, id):
    for element in chain:
        image_index = get_image_index(element, list)
        box_index = list[image_index].index(element)
        list[image_index][box_index][9] = id
    return list


def drop_box(list, box):
    index = get_image_index(box, list)
    list[index].remove(box)
    if (len(list[index]) == 0):
        del list[index]
    return list


def remove_elements_from_list(list, elements):
    for box in elements:
        list = drop_box(list, box)
    return list


def isBoxInArea(box):
    threshold = int(box[1] * 0.5)
    return box[4] < threshold < box[5], box[3]


def getBoxLowerArea(image):
    threshold = int(image[0][1] * 0.5)
    bars = 0
    wines = 0
    for box in image:
        if (box[5] < threshold):
            if (box[3] == "wine"):
                wines += 1
            if (box[3] == "bar"):
                bars += 1
    return wines, bars


def getBoxUpperArea(image):
    threshold = int(image[0][1] * 0.5)
    bars = 0
    wines = 0
    for box in image:
        if (box[4] > threshold):
            if (box[3] == "wine"):
                wines += 1
            if (box[3] == "bar"):
                bars += 1
    return wines, bars


def countItems(list):
    wineCount = 0
    prevWineCount = 0
    barCount = 0
    prevBarCount = 0
    finalWineCount = 0
    finalBarCount = 0
    for index, image in enumerate(list):
        if index == 0:
            finalWineCount += getBoxLowerArea(image)[0]
            finalBarCount += getBoxLowerArea(image)[1]
        if index == len(list) - 1:
            finalWineCount += getBoxUpperArea(image)[0]
            finalBarCount += getBoxUpperArea(image)[1]
        for box in image:
            v1, v2 = isBoxInArea(box)
            if v1:
                if v2 == "wine":
                    wineCount += 1
                if v2 == "bar":
                    barCount += 1
        if (wineCount - prevWineCount >= 0):
            finalWineCount += wineCount - prevWineCount
        if (barCount - prevBarCount >= 0):
            finalBarCount += barCount - prevBarCount

        prevWineCount = wineCount
        prevBarCount = barCount
        wineCount = 0
        barCount = 0

    return finalWineCount, finalBarCount


def countInDisList(list):
    finalWineCount = 0
    finalBarCount = 0
    for item in list:
        if (item[1] == "wine"):
            finalWineCount += 1
        if (item[1] == "bar"):
            finalBarCount += 1
    return finalWineCount, finalBarCount

def getChainByID(list, ID):
    found = False

    for image in list:
        for box in image:
            if (box[9] == ID):
                firstElement = box
                found = True
        if found:
            break
    chain = get_successor_chain(list, firstElement)
    if (len(chain) == 1):
        return None
    else:
        return chain


def getAllDistancesInList(list):
    distances = []
    for image in list:
        combinations = itertools.combinations(image, 2)
        if combinations is None:
            continue
        else:
            for c in combinations:
                distances.append([c[0][9], c[1][9], mean_x_distance(c[0], c[1])])

    return distances


def getSmalerDisList(distances, counter):
    sortedList = []
    s = set()

    for d in distances:
        s.add((d[0], d[1]))

    for ss in s:
        h = []
        for d in distances:
            if (d[0] == ss[0] and d[1] == ss[1]) or (d[0] == ss[1] and d[1] == ss[0]):
                h.append(d[2])
        sortedList.append([ss[0], ss[1], int(numpy.mean(h))])

    return sortedList


def buildPositionList(list, counter):
    pos = []
    for i in range(counter + 1):
        pos.append(0)
    for i in range(counter + 1):
        if i == 0:
            pos[0] = 0
        else:
            index = getPositionListIndex(list, i - 1, i)
            if (index is not None):
                pos[i] = pos[i - 1] + list[index][2]
                del list[index]
    return pos


def getPositionListIndex(list, v1, v2):
    for i in list:
        if (i[0] == v1 and i[1] == v2) or (i[1] == v1 and i[0] == v2):
            return list.index(i)

    return None


def get_mean_x_position(box):
    return (round((box[4] + box[5]) / 2))


def merge_close_items(pos_list, lower_threshold):
    pos_list.sort(key=lambda x: x[0])
    wines = filter_position_list(pos_list, "wine")
    close_items = []
    for index, wine in enumerate(wines):
        if (index < (len(wines)-1)):
            next_wine = wines[index+1]
            distance = abs(wine[0] - next_wine[0])
            if(distance < lower_threshold):
                close_items.append([wine, next_wine])

    for item in close_items:
        new_position = (item[0][0] + item[1][0]) / 2
        pos_list.append([new_position, item[0][1], item[0][2]])
        pos_list.remove(item[0])
        pos_list.remove(item[1])
    return pos_list


def get_object_with_min_x_position(image):
    result = image[0]
    for box in image:
        min_pos = get_mean_x_position(result)
        this_pos = get_mean_x_position(box)
        if (this_pos < min_pos):
            result = box
    return result


def get_object_with_min_x_position_and_value(dis, image1):
    copy_of_image1 = copy.copy(image1)
    obj = get_object_with_min_x_position(copy_of_image1)
    while not dis[obj[9]]:
        copy_of_image1.remove(obj)
        obj = get_object_with_min_x_position(copy_of_image1)

    return obj


def calculate_distance_and_map_to_positions(list):
    distance_list = []
    for i in range(5000):
        distance_list.append([])

    distance_list[get_object_with_min_x_position(list[0])[9]].append(0)

    for image in list:
        # get the first element which has already a value in the distance_list
        first = get_object_with_min_x_position_and_value(distance_list, image)
        first_position = sum(distance_list[first[9]]) / len(distance_list[first[9]])
        for box in image:
            if (first == box):
                continue
            dis_to_box = mean_x_distance_without_abs(first, box)
            distance_list[box[9]].append(first_position + dis_to_box)

    # drop empty parts of the list
    distance_list = [x for x in distance_list if x]

    return distance_list


def get_mean_positions(distance_list):
    final_list = []
    for i in distance_list:
        if (len(i) == 0):
            continue
        m = sum(i) / len(i)
        final_list.append(round(m))
    return final_list


def get_class_name_by_id(id, list):
    for image in list:
        for box in image:
            if (box[9] == id):
                return box[3]
    return


def add_attributes_to_position_list(distance_list, list):
    return_list = []
    print(distance_list)
    for i, dis in enumerate(distance_list):
        return_list.append([dis, get_class_name_by_id(i, list), i])


    return_list = merge_close_items(return_list, lower(return_list))

    return return_list


def get_quartiles(list1):
    list2 = []
    print(list1)
    for index, item, in enumerate(list1):
        # if not the last image
        if (index < (len(list1) - 1)):
            list2.append([item[2],list1[index + 1][2],list1[index + 1][0] - item[0]])
    print(list2)
    return np.percentile(np.array([x[2] for x in list2]), 25), np.percentile(np.array([x[2] for x in list2]), 75)


def lower(list):
    return get_threshold(get_quartiles(list)[0],get_quartiles(list)[1])[0]


def filter_position_list(pos_list, my_filter):
    return list(filter(lambda x: x[1] == my_filter, pos_list))


def get_threshold(q_25, q_75):
    dif = q_75 - q_25
    return q_25 - (1.5 * dif), q_75 + (1.5 * dif)


def identify_outliers(wine_position_list, upper_threshold):
    outliers =[]
    for index, wine in enumerate(wine_position_list):
        if (index < (len(wine_position_list)-1)):
            next_wine = wine_position_list[index+1]
            distance = abs(wine[0] - next_wine[0])
            if(distance > upper_threshold):
                outliers.append([wine, next_wine])

    return outliers


#       0   filename
#       1   width
#       2   height
#       3   class
#       4   xmin
#       5   xmax
#       6   ymin
#       7   ymax
#       8   isBoxPredicted?
#       9   ItemID
#       10  confidence


class Box:
    def __init__(self, img, wid, hei, c, x1, x2, y1, y2):
        self.minX = x1
        self.minY = y1
        self.maxX = x2
        self.maxY = y2
        self.label = c
        self.image = img
        self.height = hei
        self.widht = wid
