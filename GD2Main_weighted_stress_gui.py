import math
import sys
import time
import os
import random

import numpy as np
from scipy.optimize import minimize

#NetworkX
import networkx as nx
from networkx.drawing.nx_agraph import write_dot
from networkx.drawing.nx_agraph import read_dot as nx_read_dot

#Metrics
import ksymmetry
import crossings
import stress
import neighbors_preservation
import uniformity_edge_length
import areafunctions


def scale_graph(G, alpha):

    H = G.copy()

    for currVStr in nx.nodes(H):

        currV = H.nodes[currVStr]

        x = float(currV['pos'].split(",")[0])
        y = float(currV['pos'].split(",")[1])

        x = x * alpha
        y = y * alpha

        currV['pos'] = str(x)+","+str(y)

    return H


def writeSPXPositiontoNetworkXGraph(G, X):
    '''
    Convert matrix X to NetworkX graph structure
    '''
    positions = dict()
    sorted_v = sorted(nx.nodes(G))
    for i in range(0, len(sorted_v)):
        v = sorted_v[i]
        x = X[i,:][0]
        y = X[i,:][1]
        v_pos = str(x)+","+str(y)
        positions[v] = v_pos
    nx.set_node_attributes(G, positions, 'pos')
    return G


def netoworkxPositionsToMatrix(G):
    '''
        Convert NetwokX pos to Matrix
    '''

    n = nx.number_of_nodes(G)
    X_curr = np.random.rand(n,2)*100 - 50
    vertices_positions = nx.get_node_attributes(G, "pos")
    nodes_list_sorted = sorted(nx.nodes(G))

    for i in range(0, len(nodes_list_sorted)):
        curr_n_id = nodes_list_sorted[i]
        x = float(vertices_positions[curr_n_id].split(",")[0])
        y = float(vertices_positions[curr_n_id].split(",")[1])
        tmp = np.zeros((2))
        tmp[0] = x
        tmp[1] = y
        X_curr[i] = tmp

    return X_curr


def computeGraphDistances(G):
    '''
        Computes all pairs shortest paths on the given graph.
    '''

    G_undirected = nx.Graph(G)
    distances = nx.floyd_warshall(G_undirected)
    return distances

def printMetrics(G):
    '''
        Set inital values before optimization
    '''

    global initial_st
    global all_pairs_sp
    global initial_cr
    global initial_ar
    global initial_asp
    # Do some preliminary stuff

    # Stress will be normalized considering the first value as max
    # To speed up ST precompute all pairs shortest paths
    initial_st = 1
    if compute_st:
        initial_st = stress.stress(G, all_sp=all_pairs_sp)
        print("ST:", initial_st, end=" - ")
        if all_pairs_sp is None:
            all_pairs_sp = nx.shortest_path(G)

    # To speed up NP precompute all pairs shortest paths
    if compute_np:
        if all_pairs_sp is None:
            all_pairs_sp = nx.shortest_path(G)
        initial_np = neighbors_preservation.compute_neig_preservation(G, all_sp=all_pairs_sp)
        print("NP:", initial_np, end=" - ")

    initial_sym = 0
    if compute_sym:
        initial_sym = ksymmetry.get_symmetric_score(G)
        print("Sym:", abs(initial_sym), end=" - ")

    initial_cr = 1
    if compute_cr:
        initial_cr = len(crossings.count_crossings(G))
        print("CR:", initial_cr, end=" - ")

    initial_ue = 0
    if compute_ue:
        initial_ue = uniformity_edge_length.uniformity_edge_length(G)
        print("UE:", initial_ue, end=" - ")

    initial_ar = 1
    if compute_ar:
        initial_ar = areafunctions.areaerror(G)
        print("AR:", initial_ar, end=" - ")

    initial_asp = 1
    if compute_asp:
        initial_asp = areafunctions.aspectRatioerror(G)
        print("ASP:", initial_asp, end=" - ")

    print("")

    return


draw_counter = 0
def metrics_evaluator(X, print_val=False):

    '''
        Evaluates the metrics of the given layout and weights them
    '''
    global G
    global all_pairs_sp
    # Add some additional global variables
    global OUTPUT_FOLDER
    global graph_name
    global cnvs, cnvs_size, cnvs_padding, draw_counter

    n = nx.number_of_nodes(G)

    #Reshape the 1D array to a n*2 matrix
    X = X.reshape((n,2))
    return_val = 0.0

    G = writeSPXPositiontoNetworkXGraph(G, X)

    ue = 0
    ue_count = 0
    if compute_ue:
        ue = uniformity_edge_length.uniformity_edge_length(G)
        ue_count = ue
        # if log%100==0:
            # print("UE:", ue, end=" - ")
        ue *= abs(compute_ue)

    st = 0
    st_count=0
    if compute_st:
        st = stress.stress(G, all_sp=all_pairs_sp)
        st_count = st
        # if log%100==0:
            # print("ST:", st, end=" - ")
        st *= abs(compute_st)/initial_st

    sym = 0
    sym_count = 0
    if compute_sym:
        G = scale_graph(G, 1000)
        sym = ksymmetry.get_symmetric_score(G)
        G = scale_graph(G, 1/1000)
        sym_count = sym
        # if log%100==0:
            # print("Sym:", abs(sym), end=" - ")
        sym = 1-sym
        sym *= abs(compute_sym)

    np = 0
    np_count = 0
    if compute_np:
        np = neighbors_preservation.compute_neig_preservation(G, all_sp=all_pairs_sp)
        np_count = np
        np = 1-np
        np *= abs(compute_np)

    cr = 0
    cr_count = 0
    if compute_cr:
        cr = len(crossings.count_crossings(G))
        cr_count = cr
        if not initial_cr==0: cr *= abs(compute_cr)/initial_cr
        else: cr = 0

    ar = 0
    ar_count = 0
    if compute_ar:
        ar = areafunctions.areaerror(G)
        ar_count = ar
        ar = abs(ar-1)
        ar *= abs(compute_ar)/initial_ar

    # Aspect ratio
    asp = 0
    asp_count = 0
    if compute_asp:
        asp = areafunctions.aspectRatioerror(G)
        asp_count = asp
        asp = abs(asp-1)
        asp *= abs(compute_asp)/initial_asp

    return_val = ue+st+sym+np+cr+ar+asp

    if print_val:
        print("score: ", return_val)

    if mode=="GUI":
      if draw_counter%100==0:
          min_x, min_y, max_x, max_y = 0, 0, 0, 0
          for currVStr in nx.nodes(G):
              currV = G.nodes[currVStr]
              x = float(currV['pos'].split(",")[0])
              y = float(currV['pos'].split(",")[1])
              min_x = min(min_x,x)
              max_x = max(max_x,x)
              min_y = min(min_y, y)
              max_y = max(max_y, y)
              currV['pos'] = str(x)+","+str(y)

          cnvs.delete("all")
          scl = (cnvs_size-cnvs_padding)/(max(max_y-min_y, max_x-min_x))
          tx = cnvs_padding/2
          ty = cnvs_padding/2
          pos_dict = nx.get_node_attributes(G, 'pos')
          for edge in nx.edges(G):
              (s,t) = edge
              x_source = float(pos_dict[s].split(",")[0])
              x_target = float(pos_dict[t].split(",")[0])
              y_source = float(pos_dict[s].split(",")[1])
              y_target = float(pos_dict[t].split(",")[1])
              cnvs.create_line((x_source-min_x)*scl+tx, (y_source-min_y)*scl+ty, (x_target-min_x)*scl+tx, (y_target-min_y)*scl+ty)
              print((x_source-min_x)*scl, (x_target-min_x)*scl, (y_source-min_y)*scl, (y_target-min_y)*scl)
              cnvs.update()
      draw_counter += 1

    return return_val

import torch
def torch_to_numpy(X_torch):
  n = len(X_torch)
  X = np.random.rand(n,2)
  for i in range(n):
     X[i][0], X[i][1] = X_torch[i][0], X_torch[i][1]
  return X

def numpy_to_torch(X):
  n = len(X)
  X_torch = torch.rand(n, 2, requires_grad = True)
  for i in range(n):
     X_torch[i][0], X_torch[i][1] = X[i][0], X[i][1]
  return X_torch

def minimize_with_torch(func, X, lr=.01, prec=.001, max_iter=1000):
  step = 1
  i = 0
  while i<max_iter:
    X_numpy = torch_to_numpy(X)
    s = func(X_numpy)
    #print(s)
    s.backward()
    with torch.no_grad():
      X = X - lr*X.grad
    X.requires_grad = True
    i += 1
  #print('i:', i)
  #print('X:', X)
def optimize(G):
    X = netoworkxPositionsToMatrix(G)
    n = nx.number_of_nodes(G)
    # Use gradient descent to optimize the metrics_evaluator function
    # keep the X as a flattened 1D array and reshape it inside the
    # metrics_evaluator function as a 2D array/matrix
    X = X.flatten()
    res = minimize(metrics_evaluator, X, method='L-BFGS-B')
    X = res.x.reshape((n,2))
    #******************TORCH*************
    #X_torch = numpy_to_torch(X)
    #minimize_with_torch(metrics_evaluator, X_torch)
    #X = torch_to_numpy(X_torch)
    #************************************
    return X


# main
# Input
if len(sys.argv)<4:
 print('usage:python3 GD2Main.py input_folder_path output_folder_path mode(GUI/console)')
 quit()

#GRAPH_PATH = sys.argv[1]
INPUT_FOLDER = sys.argv[1]
OUTPUT_FOLDER = sys.argv[2] # Output folder
G = None
graph_name = ""
mode = sys.argv[3]

def select_graph(graph_file_name):
  global INPUT_FOLDER, G, graph_name, all_pairs_sp
  GRAPH_PATH = INPUT_FOLDER+graph_file_name
  input_file_name = os.path.basename(GRAPH_PATH)
  graph_name = input_file_name.split(".")[0]
  print(graph_name)

  # Reading the graphs
  G = nx_read_dot(GRAPH_PATH) #this should be the default structure
  #if not nx.is_connected(G):
  #    print('The graph is disconnected')
  #    quit()

  # convert ids to integers
  G = nx.convert_node_labels_to_integers(G)

  # Set zero coordinates for all vertices
  for i in nx.nodes(G):
    x = random.uniform(0, 1)
    y = random.uniform(0, 1)
    #if i==0: x, y = 0, 0
    #if i==1: x, y = 1, 1
    #if i==2: x, y = 0, 1
    #if i==3: x, y = 1, 0
    curr_pos = str(x)+","+str(y)
    nx.set_node_attributes(G, {i:curr_pos}, "pos")

  G = scale_graph(G, 100)
  write_dot(G, OUTPUT_FOLDER + graph_name + '_initial.dot')
  G = scale_graph(G, 1/100)
  all_pairs_sp = None

# Metrics weights
compute_ue=0 #Uniformity Edge lengths
compute_st=0 # Stress
compute_sym=0 # Symmetry
compute_np=0 # Neighbor Preservation
compute_cr=0 #Crossings
compute_ar=0 #Area
compute_asp=0 #Aspect ratio

def init_metrics_weight():
  global compute_ue, compute_st, compute_sym, compute_np, compute_cr, compute_ar, compute_asp
  compute_ue=0 #Uniformity Edge lengths
  compute_sym=0 # Symmetry
  compute_np=0 # Neighbor Preservation
  compute_cr=0 #Crossings
  compute_ar=0 #Area
  compute_asp=0 #Aspect ratio

#weight_param = int(sys.argv[6])
weight_param = 0

compute_st = 1
#compute_st = int(sys.argv[7])


# Metric specific global variables
initial_st = 1
initial_cr = 1
initial_ar = 1
initial_asp = 1
all_pairs_sp = None

def run_GD():
  global G, OUTPUT_FOLDER, graph_name
  curr_G = G.copy()
  print("Initial metrics")
  printMetrics(curr_G)
  final_position_matrix = optimize(G)
  curr_G = writeSPXPositiontoNetworkXGraph(curr_G, final_position_matrix)
  write_dot(curr_G, OUTPUT_FOLDER + graph_name + '_final.dot')
  curr_G = G.copy()
  write_dot(curr_G, OUTPUT_FOLDER + graph_name + '_final.dot')
  print("Final Metrics")
  printMetrics(curr_G)
  metrics_evaluator(final_position_matrix, print_val=True)

if mode=="console":
  input_file_name = sys.argv[4]
  select_graph(input_file_name)
  compute_st = int(sys.argv[5])
  weight_param = int(sys.argv[6])
  metric = sys.argv[7]
  if metric=="0":
    compute_ue = weight_param
  elif metric=="1":
    compute_st = weight_param
  elif metric=="2":
    compute_sym = weight_param
  elif metric=="3":
    compute_np = weight_param
  elif metric=="4":
    compute_cr = weight_param
  elif metric=="5":
    compute_ar = weight_param
  elif metric=="6":
    compute_asp = weight_param
  run_GD()

#*************GUI**********************

if mode=="GUI":
  import tkinter
  from tkinter import StringVar
  from tkinter import OptionMenu
  from tkinter import DoubleVar
  from tkinter import Scale
  from tkinter import HORIZONTAL
  from tkinter import Canvas
  from tkinter import Label

  master = tkinter.Tk()

  master.title("GD")
  row_counter = 0

  graph_class_label = Label(master, text="Graph:")
  graph_class_label.grid(row=row_counter, column=0)

  def graph_class_menu(val):
    if val=="Path":
      select_graph("path.dot")
    elif val=="Cycle":
      select_graph("cycle.dot")
    elif val=="Tree":
      select_graph("tree.dot")

  variable = StringVar(master)
  variable.set("None")
  w = OptionMenu(master, variable, "Path", "Cycle", "Tree", command = graph_class_menu)
  w.grid(row=row_counter, column=1)
  row_counter += 1

  stress_weight_label = Label(master, text="Stress weight:")
  stress_weight_label.grid(row=row_counter, column = 0)

  def scale_changed(val):
    global compute_st
    compute_st = int(val)

  var = DoubleVar()
  scale = Scale( master, variable = var, from_=0, to=10, orient=HORIZONTAL, command=scale_changed)
  scale.grid(row=row_counter, column=1)
  row_counter += 1

  metric_weight_label = Label(master, text="Metric weight:")
  metric_weight_label.grid(row=row_counter, column = 0)

  def metric_weight_scale_changed(val):
    global weight_param
    weight_param = int(val)

  metric_weight_var = DoubleVar()
  metric_weight_scale = Scale(master, variable=metric_weight_var, from_=0, to=10, orient=HORIZONTAL, command=metric_weight_scale_changed)
  metric_weight_scale.grid(row=row_counter, column=1)
  row_counter += 1


  metric_label = Label(master, text="Metric:")
  metric_label.grid(row=row_counter, column= 0)

  def metric_menu_changed(val):
    #print(val)
    init_metrics_weight()
    global compute_ue, compute_st, compute_sym, compute_np, compute_cr, compute_ar, compute_asp, weight_param
    if val=="Edge uniformity":
      compute_ue = weight_param
    elif val=="Stress":
      compute_st = weight_param
    elif val=="Symmetry":
      compute_sym = weight_param
    elif val=="Neighborhood preservation":
      compute_np = weight_param
    elif val=="Crossing":
      compute_cr = weight_param
    elif val=="Area":
      compute_ar = weight_param
    elif val=="Aspect ratio":
      compute_asp = weight_param

  var_metric = StringVar(master)
  var_metric.set("None")
  metric_menu = OptionMenu(master, var_metric, "Edge uniformity", "Stress", "Symmetry", "Neighborhood preservation", "Crossing", "Area", "Aspect ratio", command = metric_menu_changed)
  metric_menu.grid(row=row_counter, column = 1)
  row_counter += 1

  def run_button():
    run_GD()


  B = tkinter.Button(master, text ="Run", command = run_button)
  B.grid(row=row_counter, column=1)
  row_counter += 1

  cnvs_size = 400
  cnvs_padding = 10
  cnvs = Canvas(master, width=cnvs_size, height=cnvs_size)
  cnvs.grid(row=row_counter,column=0)
  row_counter += 1


  master.mainloop()

