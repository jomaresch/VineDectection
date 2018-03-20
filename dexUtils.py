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

#VU.draw_box(image_list[1]).save('1.png')

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
#             color = color_1
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


def openBoxFile(name):
    with open("imageData/"+name, 'rb') as fp:
        imagelist = pickle.load(fp)
    return imagelist


def saveBoxFile(list,name):
    with open("imageData/" + name, 'wb') as fp:
        pickle.dump(list, fp)


def meanDistance(box_1,box_2):
	mySum = 0
	for x in range(4,8):
		mySum = mySum + abs(box_1[x] - box_2[x])
	return mySum/4

def meanXDistance(box_1, box_2):
    m1 = box_1[4]+box_1[5]
    m1 = round(m1/2)
    m2 = box_2[4] + box_2[5]
    m2 = round(m2 / 2)

    return abs(m1-m2)

def meanXDistanceWithoutAbs(box_1, box_2):
    m1 = box_1[4]+box_1[5]
    m1 = round(m1/2)
    m2 = box_2[4] + box_2[5]
    m2 = round(m2 / 2)

    return m2-m1

def meanXDistanceBox(box_1, box_2):
    m1 = box_1.minX+box_1.maxX
    m1 = round(m1/2)
    m2 = box_2.minX+box_2.maxX
    m2 = round(m2 / 2)

    return m1-m2

def meanYDistance(box_1, box_2):
    m1 = box_1[6] + box_1[7]
    m1 = round(m1 / 2)
    m2 = box_2[6] + box_2[7]
    m2 = round(m2 / 2)

    return abs(m1-m2)

def statusBar(i,max):
    sys.stdout.write('\r')
    sys.stdout.write(str(round(i*100/max,2)) +' %')
    sys.stdout.flush()

def getBoxArea(box):
    x = abs(box[5]-box[4])
    y = abs(box[7]-box[6])
    return x*y

def cleanListFromOverlappingBoxes(list):
    liste = list
    for index_1, image in enumerate(liste):
        for index_2, box in enumerate(image):
            for index_3, box2 in enumerate(image):
                overlapPerc = get_overlap(box, box2)
                if (0.0 < overlapPerc < 1.0 and box[3] == box2[3]):
                    if (getBoxArea(box) > getBoxArea(box2)):
                        del (liste[index_1][index_3])
                    else:
                        del (liste[index_1][index_2])
                    break
    return liste

def BoxToList(box):
    return [box.image,box.widht,box.height,box.label,box.minX,box.maxX,box.minY,Box.maxY]

def ListToBox(list):
    return Box(list[0],list[1],list[2],list[3],list[4],list[5],list[6],list[7])

def getSuccessor(list,element):
    #list with elemtent and predecessor
    for pair in list:
        if (element == pair[1]):
            return pair[0]

def getAllSuccessors(list,startElemment):
    element = startElemment
    listWithSameElements =[]
    listWithSameElements.append(startElemment)
    while (getSuccessor(list, element) is not None):
        listWithSameElements.append(getSuccessor(list, element))
        element = getSuccessor(list, element)
    return listWithSameElements

def getMeanStep(list, image_name):
    templist =[]
    dis = []
    for element in list:
        if(element[0][0] == image_name):
            templist.append(element)
    for element in templist:
        dis.append(meanXDistance(element[0], element[1]))

    if(len(dis) == 0):
        return 0.0
    else:
        return round(numpy.mean(dis))

def getMatchCount(list,image_name):
    count = 0
    for element in list:
        if (element[0][0] == image_name):
            count = count + 1
    return count

def getImageAttributes (matchlist, boxlist):
    attributeList =[]

    for frame in boxlist:
        frameName = frame[0][0]
        boxCount = len(frame)
        attributeList.append([frameName,
                              getMeanStep(matchlist,frameName),
                              boxCount,
                              getMatchCount(matchlist,frameName)])

    return attributeList

def predictMissingBoxes(matchlist, boxlist):
    imageAttributes = getImageAttributes(matchlist, boxlist)
    newBoxList = copy.deepcopy(boxlist)
    for index,image in enumerate(boxlist):
        if (index == 0):
            continue
        for box in image:
            if(hasPredecessor(box,matchlist)):
                continue
            else:
                step = getMeanStep(matchlist,box[0])
                predecessor = boxlist[index-1][0]
                predictedBox = predictPredecessor(box,step,predecessor)
                if(predictedBox is None):
                    continue
                else:
                    for index2, frame in enumerate(newBoxList):
                        if (frame[0][0] == predictedBox[0]):
                            newBoxList[index2].append(predictedBox)
    return newBoxList

def createMatchList(list):
    newList =[]
    for index_image, image in enumerate(list):
        if (index_image == 0):
            continue
        for index_box_now, box_now in enumerate(list[index_image]):
            distances = []
            saved_inds = []
            for index_box_prev, box_prev in enumerate(list[index_image - 1]):
                if (box_now[3] == box_prev[3]):
                    distances.append(meanXDistance(box_now, box_prev))
                    saved_inds.append(index_box_prev)
            if (len(distances) == 0):
                continue
            min_dis = min(distances)
            if (min_dis < 150):
                newList.append([box_now, list[index_image - 1][saved_inds[distances.index(min_dis)]]])
    return newList


def hasPredecessor( box, matchlist):
    for match in matchlist:
        if(match[0] == box):
            return True
    return False

def predictPredecessor(box,step, predecessor):
    if(box[4] - step < 200 or box[5] - step > box[1]-200):
        return None
    else:
        print("predicted" + predecessor[0])
        return [predecessor[0], predecessor[1], predecessor[2], box[3],  box[4] - step , box[5] - step , box[6] , box[7], True]

def printList(list):
    print()
    for i in list:
        print(i)
    print()


def resizeImage(image_name, width, height):
    img = Image.open('old/' + image_name)
    img = img.resize((width,height), PIL.Image.LANCZOS)
    img.save('new/'+ image_name)

def predictOutputToList(boxes):

    imagelist = []
    image = []

    image.append(boxes[0])

    for box in boxes:
        if box[3] == 1:
            box[3] = 'wine'
        elif box[3] == 2:
            box[3] = 'bar'
        elif box[3] == 3:
            box[3] = 'wineProtected'

        if (box[0] == image[0][0]):
            image.append(box)
        else:
            imagelist.append(image)
            image = []
            image.append(box)
    imagelist.append(image)
    del imagelist[0][0]
    return imagelist

def deleteNotCentredBoxes(list):
    cleanedList =[]
    for image in list:
        for box in image:
            if(box[6] < (box[2]/2)< box[7]) and (box[4] > (box[1] * 0.04)) and (box[5] < (box[1] * 0.96)):
                cleanedList.append(box)

    return predictOutputToList(cleanedList)

def getSuccessorElement(element, list, index):
    if(index+1 == len(list)):
        return None
    else:
        nextImage = list[index+1]
    overlaps =[]
    distances =[]
    boxes = []
    for box in nextImage:
        if(box[3] == element[3]):
            distances.append(meanXDistance(box, element))
            overlaps.append(get_overlap(box,element))
            boxes.append(box)

    if overlaps:
        if (max(overlaps) > 0.0):
            return boxes[(overlaps.index(max(overlaps)))]

    if distances:
        min_dis = min(distances)
        if (min_dis < 150):
            return boxes[(distances.index(min_dis))]

    return None

def meanXDistanceList(list):
    dis = []
    for index, box in enumerate(list):
        if(index + 1 == len(list)):
            continue
        else:
            dis.append(meanXDistance(box, list[index+1]))
    return int(round(np.mean(dis)))

def getImageIndex(box, list):
    for image in list:
        if box in image:
            return list.index(image)

def appendListWithPredictedSuccessorValue(list, prevElement, step):
    index  = getImageIndex(prevElement, list)
    if (index + 1 == len(list)):
        return None
    v0 = list[index+1][0][0]
    v1 = prevElement[1]
    v2 = prevElement[2]
    v3 = prevElement[3]
    v4 = prevElement[4] - step
    v5 = prevElement[5] - step
    v6 = prevElement[6]
    v7 = prevElement[7]
    if (v4 <= 0):
        return None

    item = [v0,v1,v2,v3,v4,v5,v6,v7,True,None]
    list[index+1].append(item)
    return list

def appendListWithPredictedPredecessorValue(list, nextElement, step):
    index  = getImageIndex(nextElement, list)
    if (index == 0):
        return None
    v0 = list[index-1][0][0]
    v1 = nextElement[1]
    v2 = nextElement[2]
    v3 = nextElement[3]
    v4 = nextElement[4] + step
    v5 = nextElement[5] + step
    v6 = nextElement[6]
    v7 = nextElement[7]
    v9 = nextElement[9]
    if(v5 > v1):
        if (v5 > v1+40):
            return None
        else:
            v5 = v1
    else:
        list[index-1].append([v0,v1,v2,v3,v4,v5,v6,v7,True,v9])
        return list

def addValue_8_and_9_ToBox(list):
    list2 = []
    for image in list:
        image_new = []
        for box in image:
            image_new.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], False, None])
        list2.append(image_new)

    return list2

def getSuccessorChain(list, box):
    index = getImageIndex(box, list)
    successorChain = []
    predecessor = box
    for i, image in enumerate(list):
        if i <index:
            continue
        t = getSuccessorElement(predecessor, list, i)
        if t is None:
            break
        successorChain.append(t)
        predecessor = t
    successorChain.insert(0,box)
    return successorChain

def areLastChainItemsPredicted(chain):
    if(len(chain) < 6):
        return False
    if(chain[-1][8] and chain[-2][8] and chain[-3][8] and chain[-4][8] and chain[-5][8] and chain[-6][8]):
        return True
    return False

def deleteLastPredicted(chain, list):
    while(chain[-1][8]):
        imageIndex = getImageIndex(chain[-1], list)
        list[imageIndex].remove(chain[-1])
        chain.remove(chain[-1])
    return list

def setChainID(chain, list, id):
    for element in chain:
        imageIndex = getImageIndex(element,list)
        boxIndex = list[imageIndex].index(element)
        list[imageIndex][boxIndex][9] = id
    return list

def dropBox(list, box):
    index = getImageIndex(box,list)
    list[index].remove(box)
    if(len(list[index]) == 0):
        del list[index]
    return list

def removeElementsFromList(list, elements):
    for box in elements:
        list = dropBox(list, box)
    return list

# def drawBoxesAndSave(list):
#     for image in list:
#         print(image)
#         draw_box(image).save('imagesnew/' + image[0][0])
#
# def drawBoxesAndSaveIn4Threads(list):
#     length = len(list)
#     half = int(length/2)
#     quarter = int(half/2)
#     print("Thread 1 started")
#     t1 = Thread(target=drawBoxesAndSave, args=(list[:quarter],))
#     print("Thread 2 started")
#     t2 = Thread(target=drawBoxesAndSave, args=(list[quarter:half],))
#     print("Thread 3 started")
#     t3 = Thread(target=drawBoxesAndSave, args=(list[half+1:half+quarter],))
#     print("Thread 4 started")
#     t4 = Thread(target=drawBoxesAndSave, args=(list[half+quarter+1:],))
#     t1.start()
#     t2.start()
#     t3.start()
#     t4.start()
#     t1.join()
#     t2.join()
#     t3.join()
#     t4.join()

def getChainByID(list, ID):
    found = False

    for image in list:
        for box in image:
            if(box[9] == ID):
                firstElement = box
                found = True
        if found:
            break
    chain = getSuccessorChain(list, firstElement)
    if (len(chain) == 1):
        return None
    else:
        return chain

def getAllDistancesInList(list):
    distances = []
    for image in list:
        combinations = itertools.combinations(image,2)
        if combinations is None:
            continue
        else:
            for c in combinations:
                distances.append([c[0][9],c[1][9], meanXDistance(c[0],c[1])])

    return distances

def getSmalerDisList(distances, counter):
    sortedList = []
    s = set()

    for d in distances:
        s.add((d[0] ,d[1]))

    for ss in s:
        h = []
        for d in distances:
            if(d[0] == ss[0] and d[1] == ss[1]) or (d[0] == ss[1] and d[1] == ss[0]):
                h.append(d[2])
        sortedList.append([ss[0],ss[1],int(numpy.mean(h))])

    return sortedList

def buildPositionList(list,counter):
    pos = []
    for i in range(counter + 1):
        pos.append(0)
    for i in range(counter+1):
        if i == 0:
            pos[0] = 0
        else:
            index = getPositionListIndex(list,i-1,i)
            if(index is not None):
                pos[i] = pos[i-1] + list[index][2]
                del list[index]
    printList(list)
    return pos

def getPositionListIndex(list, v1, v2):
    for i in list:
        if(i[0] == v1 and i[1] == v2) or (i[1] == v1 and i[0] == v2):
            return list.index(i)

    return None

def getMeanXPos(box):
    return (round((box[4] + box[5]) / 2))

def getObjectWithMinXPosition(image):
    result = image[0]
    for box in image:
        minPos = getMeanXPos(result)
        thisPos = getMeanXPos(box)
        if(thisPos < minPos):
            result = box
    return result

def getObjectWithMinXPositionAndAValue(dis,image1):
    lulul = copy.copy(image1)
    obj = getObjectWithMinXPosition(lulul)
    while not dis[obj[9]]:
        lulul.remove(obj)
        obj = getObjectWithMinXPosition(lulul)

    return obj



def getDisList(list):
    dislist =[]
    for i in range(30):
        dislist.append([])

    dislist[getObjectWithMinXPosition(list[0])[9]].append(0)

    for image in list:
        # if(image[0][0] == "VID_20180304_131842_Frame_000000336.png"):
        #     return dislist
        first = getObjectWithMinXPositionAndAValue(dislist, image)
        firstPos = sum(dislist[first[9]]) / len(dislist[first[9]])
        for box in image:
            if (first == box):
                continue
            dis_to_box = meanXDistanceWithoutAbs(first,box)
            dislist[box[9]].append(firstPos+dis_to_box)

    return dislist

def getMeanOfDisList(dislist):
    finallist = []
    for i in dislist:
        if (len(i) == 0):
            continue
        m = sum(i) / len(i)
        finallist.append(round(m))
    return finallist

def getLabelById(id, list):
    for image in list:
        for box in image:
            if(box[9] == id):
                return box[3]
    return

def addAttributesToDisList(dislist, list):
    returnList = []
    for i, dis in enumerate(dislist):
        returnList.append([dis, getLabelById(i,list), i])
    return returnList
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


class Box:
    def __init__(self,img,wid,hei,c,x1,x2,y1,y2):
        self.minX = x1
        self.minY = y1
        self.maxX = x2
        self.maxY = y2
        self.label = c
        self.image = img
        self.height = hei
        self.widht = wid
