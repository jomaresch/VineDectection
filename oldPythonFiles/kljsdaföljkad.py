import dexUtils as dex
import copy
liste = dex.openBoxFile('LG_G6_20180210_144208_saved_boxes.dat')

liste = dex.predictOutputToList(liste)

liste = dex.cleanListFromOverlappingBoxes(liste)
liste2 = []
for image in liste:
    image_new =[]
    for box in image:
        image_new.append([box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7],False])
    liste2.append(image_new)

liste = liste2


# print(liste[0])
#
# liste2 =[]
# liste2.append([liste[0][0][0],liste[0][0][1],liste[0][0][2],dex.Box(liste[0][0][4],liste[0][0][6],liste[0][0][5],liste[0][0][7],liste[0][0][3])])
#
# print(liste2)
# box = liste2[0][3]
# print(box.maxX)

# for count,i in enumerate(liste):
#     if(count == 0):
#         continue
#     dex.statusBar(count, len(liste))
#     dex.draw_box2(dex.draw_box(i),liste[count-1]).save('imagesnew/' + i[0][0])

# boxlist = []
#
# for image in liste:
#     templist =[]
#     for box in image:
#         templist.append(dex.Box(box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7]))
#     boxlist.append(templist)



# print()
# for b in liste[76]:
#     print(b)
#
# print()
# for b in liste[75]:
#     for b2 in liste[76]:
#         matcher = dex.meanXDisance(b,b2)
#         print(b,"--",matcher,"--",b2)
#
# newList =[]

# for index, image in enumerate(boxlist):
#     if (index == 0):
#         continue
#     for index_1, box_now in enumerate(boxlist[index]):
#         distances = []
#         ind = []
#         for index_2, box_prev in enumerate(boxlist[index - 1]):
#             if(box_now.label == box_prev.label):
#                 distances.append(dex.meanXDisanceBox(box_now, box_prev))
#                 ind.append(index_2)
#         if (len(distances) == 0):
#             continue
#         min_distance = min(distances)
#         if(min_distance < 70):
#             newList.append([box_now,boxlist[index - 1][ind[distances.index(min_distance)]]])


# for index_image, image in enumerate(liste):
#     if (index_image == 0):
#         continue
#     for index_box_now, box_now in enumerate(liste[index_image]):
#         distances =[]
#         saved_inds = []
#         for index_box_prev,box_prev in enumerate(liste[index_image-1]):
#             if(box_now[3] == box_prev[3]):
#                 distances.append(dex.meanXDisance(box_now, box_prev))
#                 saved_inds.append(index_box_prev)
#         if(len(distances) == 0):
#             continue
#         min_dis = min(distances)
#         if(min_dis < 110):
#             newList.append([box_now, liste[index_image - 1][saved_inds[distances.index(min_dis)]]])

# box_matches =[]
# for match in newList:
#     box_matches.append([dex.ListToBox(match[0]),dex.ListToBox(match[1]),dex.meanXDisance(match[0], match[1])])

# for i in box_matches:
#     print(i[1])

# for index, i in enumerate(newList):
#     print(index)
#     dex.drawConnections(i[0],i[1]).save('imagesnew/'+i[0][0])

# print(newList[0])
# element  = newList[0][0]
#
#
# print(dex.getMeanStep(newList,newList[40][0][0]))
#
# print(newList[40][0][0])
#
# dex.printList(dex.getImageAttributes(newList,liste))

# for image in liste:
#     dex.draw_box(image).save('imagesnew/' + image[0][0])

# matches = dex.createMatchList(liste)
# liste = dex.predictMissingBoxes(matches,liste)
# matches = dex.createMatchList(liste)
# liste = dex.predictMissingBoxes(matches,liste)
# matches = dex.createMatchList(liste)
# liste = dex.predictMissingBoxes(matches,liste)
# matches = dex.createMatchList(liste)
# liste = dex.predictMissingBoxes(matches,liste)

# liste = dex.cleanListFromOverlappingBoxes(liste)

liste = dex.deleteNotCentredBoxes(liste)
matches = dex.createMatchList(liste)
# for image in liste:
#     dex.draw_box(image).save('imagesnew/' + image[0][0])


# t = dex.getAllSuccessors(newList,element)
# dex.printList(t)
# for index, i in enumerate(newList):
#     print()
m = dex.getAllSuccessors(matches,liste[0][0])
dex.printList(m)

print()

XX = []

for i , image in enumerate(liste):
    if i == 0:
        succz = liste[0][0]

    t = dex.getSuccessorElement(succz ,liste[i+1])
    if not t:
        break

    XX.append(t)
    print(t)
    succz = t

print(dex.meanXDistanceList(XX))

print(XX[-1])

print(dex.getImageIndex(XX[-1], liste))
#print(dex.getSuccessorElement(liste[0][0], liste[1]))

