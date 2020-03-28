import pygraphviz as pgv
import networkx as nx

import math

def scale_graph(G, alpha):

    H = G.copy()

    for currVStr in nx.nodes(H):

        currV = H.node[currVStr]

        x = float(currV['pos'].split(",")[0])
        y = float(currV['pos'].split(",")[1])

        x = x * alpha
        y = y * alpha

        currV['pos'] = str(x)+","+str(y)

    return H

def normalize_edge_length(G):

    shortest_edge = float("inf")

    vertices_positions = nx.get_node_attributes(G, "pos")

    for e in nx.edges(G):

        (u,v) = e

        x_v = float(vertices_positions[v].split(",")[0])
        y_v = float(vertices_positions[v].split(",")[1])

        x_u = float(vertices_positions[u].split(",")[0])
        y_u = float(vertices_positions[u].split(",")[1])

        curr_length = math.sqrt((x_v-x_u)**2+(y_v-y_u)**2)

        shortest_edge = min(shortest_edge, curr_length)

    alpha = 1/shortest_edge

    G = scale_graph(G, alpha)

    return G

def vertex_degree(G):

    degree = sorted(dict(nx.degree(G)).values())[-1]

    return degree


def computearea(G):

    G = normalize_edge_length(G)
    (width, height) = boundingBox(G)

    area = width*height

    return area



def boundingBox(G):

    all_pos = nx.get_node_attributes(G, "pos").values()

    coo_x = sorted([float(p.split(",")[0]) for p in all_pos])
    coo_y = sorted([float(p.split(",")[1]) for p in all_pos])

    min_x = float(coo_x[0])
    max_x = float(coo_x[-1])

    min_y = float(coo_y[0])
    max_y = float(coo_y[-1])

    width = abs(max_x - min_x)
    height = abs(max_y - min_y)

    return (width, height)


def aspectRatio(G):

    bb = boundingBox(G)

    aspectRatio_value = bb[0]/bb[1]

    return aspectRatio_value


def diameter(G):

    return nx.diameter(G)



#ERROR Functions

def areaerror(G, referenceArea=10):
    area = computearea(G)
    return abs(referenceArea-area)


def aspectRatioerror(G, referenceAR=1):
    aspectRatio_value = aspectRatio(G)
    return abs(referenceAR-aspectRatio_value)


