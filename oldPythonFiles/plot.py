# import matplotlib.pyplot as plt
#
#
#
import plotly as py
# from plotly.graph_objs import *
#import plotly.plotly as py
from plotly.graph_objs import *
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# from matplotlib.lines import Line2D
# import numpy as np

# t = np.arange(0.0, 1.0, 0.1)
# s = np.sin(2*np.pi*t)
# linestyles = ['_', '-', '--', ':']
# markers = []
# for m in Line2D.markers:
#     try:
#         if len(m) == 1 and m != ' ':
#             markers.append(m)
#     except TypeError:
#         pass
#
# styles = markers + [
#     r'$\lambda$',
#     r'$\bowtie$',
#     r'$\circlearrowleft$',
#     r'$\clubsuit$',
#     r'$\checkmark$']
#
# colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
#
# plt.figure(figsize=(8,8))
#
# axisNum = 0
# for row in range(6):
#     for col in range(5):
#         axisNum += 1
#         ax = plt.subplot(6, 5, axisNum)
#         color = colors[axisNum % len(colors)]
#         if axisNum < len(linestyles):
#             plt.plot(t, s, linestyles[axisNum], color=color, markersize=10)
#         else:
#             style = styles[(axisNum - len(linestyles)) % len(styles)]
#             plt.plot(t, s, linestyle='None', marker=style, color=color, markersize=10)
#         ax.set_yticklabels([])
#         ax.set_xticklabels([])
#
# plt.show()

# lines = plt.plot([1,2,3,4], [1,1,1,1], 'ro')
# plt.axis([0, 6, 0, 20])
# plt.setp(lines, linestyle='-')
# plt.show()
points = [[-290, 'wine'], [-35, 'bar'], [144, 'wine'], [551, 'wine'], [596, 'wine'], [979, 'wine'],
          [1452, 'wine'], [1485, 'wine'], [1875, 'wine'], [2310, 'bar'], [2412, 'wine'], [2831, 'wine'],
          [3285, 'wine'], [3773, 'wine'], [4229, 'wine'], [4296, 'wine'], [4530, 'bar'], [4662, 'bar'],
          [4784, 'wine'], [5165, 'wine'], [5656, 'wine'], [6045, 'wine'], [6137, 'wine'], [6414, 'bar'],
          [6544, 'bar'], [6657, 'wine'], [7201, 'wine']]
Xn =[]
Yn =[]
Zn =[]
colors = []
names = []
for i, point in enumerate(points):
    Xn.append(0)
    Yn.append(point[0])
    Zn.append(0)
    if (point[1] == "wine"):
        colors.append(0)
    if (point[1] == "bar"):
        colors.append(1)
    names.append(str(point[0])+ " "+ point [1])



cScale =  [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]

# Xe = [1,2,None,2,3,None]
# Ye = [1,2,None,2,3,None]
# Ze = [1,2,None,2,3,None]
# trace1=Scatter3d(x=Xe,
#                y=Ye,
#                z=Ze,
#                mode='lines',
#                line=Line(color='rgb(125,0,125)', width=5),
#             text = 'nice boy',
#                hoverinfo='text'
#                )
trace2=Scatter3d(x=Xn,
               y=Yn,
               z=Zn,
               mode='markers',
               name='actors',
               marker=Marker(symbol='dot',
                             size=6,
                             color=colors,
                             colorscale=cScale,
                             line=Line(color='rgb(50,50,50)', width=0.5)
                             ),
               text=names,
               hoverinfo='text'
               )

data=Data([trace2])
fig=Figure(data=data)

py.offline.plot(fig)