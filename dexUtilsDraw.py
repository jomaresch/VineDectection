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
import plotly as py
from plotly.graph_objs import *

def draw_box(box_list):
    color_bar = (244, 100, 100)
    color_wine = (104, 249, 255)
    color_predicted = (57, 83, 186)

    image_path = box_list[0][0]
    image = Image.open('images/'+image_path)
    draw = ImageDraw.Draw(image)
    for box in box_list:
        if box[3] == 'bar':
            color = color_bar
        elif box[3] == 'wine':
            color = color_wine
        else:
            color = color_predicted

        if(box[8]):
            color = color_predicted

        if box[9] is None:
            text = 'None'
        else:
            text = str(box[9])

        text = box[3]+ ': ' + text
        draw.line((box[4], box[6], box[5], box[6]), fill=color, width=6)
        draw.line((box[4], box[7], box[5], box[7]), fill=color, width=6)
        draw.line((box[4], box[6], box[4], box[7]), fill=color, width=6)
        draw.line((box[5], box[6], box[5], box[7]), fill=color, width=6)
        draw.text((box[4], box[6]), text, fill='black', font= ImageFont.truetype('arial.ttf', 32))
    return image

def drawConnections(b1,b2):
    image_path  = b1[0]
    image = Image.open('imagesnew/' + image_path)
    draw = ImageDraw.Draw(image)
    draw.line((b1[4], b1[6], b2[4], b2[6]), fill=(0,0,0), width=6)
    draw.line((b1[4], b1[7], b2[4], b2[7]), fill=(0,0,0), width=6)
    draw.line((b1[5], b1[6], b2[5], b2[6]), fill=(0,0,0), width=6)
    draw.line((b1[5], b1[7], b2[5], b2[7]), fill=(0,0,0), width=6)
    return image

def draw_box2(image,box_list):
    color_1 = (0, 255, 0)
    color_2 = (255, 0, 0)
    color_3 = (0, 0, 255)

    draw = ImageDraw.Draw(image)
    color = color_3
    for box in box_list:
        if box[3] == 'bar':
            color = color_1
        elif box[3] == 'wine':
            color = color_2
        draw.line((box[4], box[6], box[5], box[6]), fill=color, width=6)
        draw.line((box[4], box[7], box[5], box[7]), fill=color, width=6)
        draw.line((box[4], box[6], box[4], box[7]), fill=color, width=6)
        draw.line((box[5], box[6], box[5], box[7]), fill=color, width=6)
    return image

def drawBoxesAndSave(list):
    for image in list:
        print(image)
        draw_box(image).save('imagesnew/' + image[0][0])

def drawBoxesAndSaveIn4Threads(list):
    length = len(list)
    half = int(length/2)
    quarter = int(half/2)
    print("Thread 1 started")
    t1 = Thread(target=drawBoxesAndSave, args=(list[:quarter],))
    print("Thread 2 started")
    t2 = Thread(target=drawBoxesAndSave, args=(list[quarter:half],))
    print("Thread 3 started")
    t3 = Thread(target=drawBoxesAndSave, args=(list[half+1:half+quarter],))
    print("Thread 4 started")
    t4 = Thread(target=drawBoxesAndSave, args=(list[half+quarter+1:],))
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()

def plot(points):
    Xn = []
    Yn = []
    Zn = []
    colors = []
    names = []

    points.sort(key=lambda x: x[0])
    for i, point in enumerate(points):
        Xn.append(0)
        Yn.append(point[0])
        Zn.append(0)
        if (point[1] == "wine"):
            colors.append(0)
        if (point[1] == "bar"):
            colors.append(1)
        names.append(str(point[2]) + "---" + str(point[0]) + " " + point[1])

    colorScale = [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]

    trace2 = Scatter3d(
        x=Xn,
        y=Yn,
        z=Zn,
        mode='markers',
        name='actors',
        marker=Marker(
            symbol='dot',
            size=6,
            color=colors,
            colorscale=colorScale,
            line=Line(color='rgb(50,50,50)', width=0.5)),
        text=names,
        hoverinfo='text')

    data = Data([trace2])

    py.offline.plot(Figure(data=data))