import networkx as nx
import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = cv2.imread("C:/Users/VENKATESH/Desktop/occupancy_map.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/content/drive/MyDrive/Mobile_Robotics/occupancy_map.png")
imgg = mpimg.imread("C:/Users/VENKATESH/Desktop/occupancy_map.png")
occupancy_grid = (np.asarray(img) > 0).astype(int)
k = 2
def Check_neighby(v):
    global occupancy_grid
    x = int(v[0])
    y = int(v[1])
    flag = 0
    for i in range(0, 2):
        for j in range(0, 2):
            if 0 <= x + i <= 680 and 0 <= y + j <= 623:
                if occupancy_grid[x + i][y + j] == 1:
                    pass
                else:
                    flag = -1
            if 0 <= x - i <= 680 and 0 <= y - j <= 623:
                if occupancy_grid[x - i][y - j] == 1:
                    pass
                else:
                    flag = -1
    if 0 <= x + 1 <= 680 and 0 <= y - 1 <= 623:
        if occupancy_grid[x + 1][y - 1] == 1:
            pass
        else:
            flag = -1
    if 0 <= x - 1 <= 680 and 0 <= y + 1 <= 623:
        if occupancy_grid[x - 1][y + 1] == 1:
            pass
        else:
            flag = -1
    if flag == -1:
        return False
    else:
        return True

def str_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2 - y1) > abs(x2 - x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2 - y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    if rev:
        points.reverse()
    return points

def straight(v1, v2):
    set_points = str_line(v1[0], v1[1], v2[0], v2[1])

    for points in set_points:
        if occupancy_grid[points[0], points[1]] == 1 and Check_neighby((points[0], points[1])) == True:
            continue
        else:
            return False
    return True

def d(v1, v2):
    x1 = v1[0]
    y1 = v1[1]
    x2 = v2[0]
    y2 = v2[1]
    distance = math.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
    return distance

def speciald(a, b):
    v1 = G.nodes[a]["pos"]
    v2 = G.nodes[b]["pos"]
    x1 = v1[0]
    y1 = v1[1]
    x2 = v2[0]
    y2 = v2[1]
    distance = math.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
    return distance

def findvertex(M):
    global l
    x = random.randint(0, 679)
    y = random.randint(0, 622)
    if 0 <= x < 680 and 0 <= y < 623:
        v = (x, y)
        if M[x][y] == 1:
            return v
        else:
            return findvertex(M)
    else:
        return findvertex(M)

def CP(M, N, dmax):
    global G
    for i in range(0, N + 1):
        vnew = findvertex(M)
        addvertex(G, vnew, dmax)
    l = nx.dfs_successors(G, source=0)
    k = 0
    for i in l:
        if i == 0:
            k += 1
        elif i == 1:
            k += 1
    if k != 2:
        return CP(M, 500, dmax)  # adding 500 more samples so that s and g connect
    else:
        return G

def addvertex(G, vnew, dmax):
    global k
    G.add_node(k, pos=vnew)
    if len(G.nodes()) > 1:
        for node in G.nodes():
            v = G.nodes[node]["pos"]
            if v != vnew and d(v, vnew) <= dmax:
                if straight(v, vnew) == True:
                    G.add_edge(node, k)
                    G[node][k]['weight'] = d(v, vnew)
    k = k + 1
    return

s = (635, 140)
g = (350, 400)
G = nx.Graph()
G.add_node(0, pos=s)
G.add_node(1, pos=g)

Gfinal = CP(occupancy_grid, 2500, 75)
pos = nx.get_node_attributes(Gfinal, 'pos')
Astar = nx.astar_path(Gfinal, 0, 1, heuristic=speciald, weight='weight')
length = nx.astar_path_length(Gfinal, 0, 1, heuristic=speciald, weight="weight")
print(Astar)
print(length)
imgplot = plt.imshow(np.transpose(imgg))
nx.draw_networkx(Gfinal, pos, node_size=1, with_labels=False, width=0.5)
plt.savefig("PRM_2500_plus.png",dpi=1200)
plt.show()

# plotting on the Fig.3b

Q = [(s[1], s[0])]
for points in Astar:
    v = G.nodes[points]["pos"]
    x = v[1]
    y = v[0]
    new = (x, y)
    Q.append(new)

    cv2.line(img2, Q[0], Q[1], (255, 0, 0), 1)
    Q.pop(0)

cv2.imshow(img2)