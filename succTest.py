import dexUtils as dex
import dexUtilsDraw as dexDraw
import time
import plotly as py
from plotly.graph_objs import *

# open data
liste = dex.openBoxFile('VID_20180304_131842.dat')

# list of boxes to list of images with list of boxes
liste = dex.predictOutputToList(liste)

# clean from overlapping
liste = dex.cleanListFromOverlappingBoxes(liste)

# add false to all because they are not predicted, and None because hey are not part of a chain
liste = dex.addValue_8_and_9_ToBox(liste)

# delet all Boxes which are not in the center of the image
liste = dex.deleteNotCentredBoxes(liste)

# item counter
counter = 0
delete_boxes_list = []

# for every image
for image in liste:
    # you can't calculate a successor for the last image because it's the last
    if (liste.index(image) == (len(liste) - 1)):
        break
    # if its not the last, check each box
    for box in image:

        # if the box is already in a chain and is labeled you don't need check
        if (box[9] is not None):
            continue
        # calculate all successors of this box and save them in a list
        chain = dex.getSuccessorChain(liste, box)

        # if the chain has no direkt successor, you can't predict an successor
        if (len(chain) == 1):
            delete_boxes_list.append(box)
            continue

        # ++++PREDICTION PART++++
        # we can't predict till the end, so we stop after the last X boxes in the chain are predicted
        while (not dex.areLastChainItemsPredicted(chain)):

            # get the mean movement of the cam, so we can predict where the next item should be
            step = dex.meanXDistanceList(chain)

            #get the last element of the chain
            lastElement = chain[-1]

            # predict ONE chain element with the "step" value and the last chain element
            newListe = dex.appendListWithPredictedSuccessorValue(liste, lastElement, step)

            # appendListWithPredictedValue() returns 'None' if the predicted box is out of the image
            # or it trys to predict the successor of the last frame
            if newListe is None:
                break
            else:
                liste = newListe

            # calculate the new chain with the predicted box
            chain = dex.getSuccessorChain(liste, box)

        # we delete the last 3 predicted elements, because the could be to inaccurate
        #liste = dex.deleteLastPredicted(chain, liste)

        # give each unique item a ID
        chain = dex.getSuccessorChain(liste, box)
        liste = dex.setChainID(chain, liste, counter)
        counter += 1

liste = dex.removeElementsFromList(liste, delete_boxes_list)

disl = dex.getDisList(liste)

disl = dex.getMeanOfDisList(disl)

disl = dex.addAttributesToDisList(disl,liste)

disl.sort(key=lambda x: x[0])

print(disl)

# plot all points
dexDraw.plot(disl)

# display all images and save them
dexDraw.drawBoxesAndSaveIn4Threads(liste)