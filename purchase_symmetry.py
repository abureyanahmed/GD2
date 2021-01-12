import torch
import networkx as nx
import matplotlib.pyplot as plt
#limits = plt.axis("off")  # turn of axis
#import torch_geometric.data
import crossings
import math

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 

        return ang_deg

def get_dis(x1, y1, x2, y2):
  return torch.sqrt((x1-x2)**2 + (y1-y2)**2)

'''
G = nx.dodecahedral_graph()
nx.draw(G)
plt.show()
nx.draw(G, pos=nx.spring_layout(G))  # use spring layout
plt.show()
nx.draw(G, pos=nx.random_layout(G))  # use spring layout
plt.show()
'''

G = nx.dodecahedral_graph()
print(G.edges())
#data = from_networkx(G)
n = len(G.nodes())
#adj_mat = torch.empty(n, n, dtype=torch.float)
#print(adj_mat)
#pos=nx.spring_layout(G)
pos=nx.random_layout(G)
print(len(pos.keys()), n, pos[0])
nx.draw(G, pos=pos)
plt.show()

positions = dict()
for i in range(0, n):
        x = pos[i][0]
        y = pos[i][1]
        v_pos = str(x)+","+str(y)
        positions[i] = v_pos
nx.set_node_attributes(G, positions, 'pos')
cr_arr = crossings.count_crossings(G)
print(cr_arr)
for e1, e2, p, _ in cr_arr:
  u, v = e1
  if G.has_edge(u, v):
    G.remove_edge(u, v)
  G.add_edge(u, n)
  G.add_edge(v, n)
  u, v = e2
  if G.has_edge(u, v):
    G.remove_edge(u, v)
  G.add_edge(u, n)
  G.add_edge(v, n)
  x, y = p
  pos[n] = [x, y]
  n = n+1
pos_tensor = torch.empty(n, 2, dtype=torch.float)
for i in range(0, n):
  pos_tensor[i][0] = torch.tensor(pos[i][0])
  pos_tensor[i][1] = torch.tensor(pos[i][1])
print(pos_tensor)
ls = torch.tensor(0)
thrld = torch.tensor(5.0)
for i in range(n):
  for j in range(i+1, n):
    #print(pos[i], pos[j])
    d = get_dis(pos_tensor[i][0], pos_tensor[i][1], pos_tensor[j][0], pos_tensor[j][1])
    angle = ang([pos[i], pos[j]], [[0, 0], [10, 0]])
    angle = torch.tensor(-math.radians(angle))
    #print(angle)
    for k in range(n):
      pos_tensor[k][0] = pos_tensor[k][0] - pos_tensor[i][0]
      pos_tensor[k][1] = pos_tensor[k][1] - pos_tensor[i][1]
      pos_tensor[k][0] = torch.cos(angle) * pos_tensor[k][0] - torch.sin(angle) * pos_tensor[k][1]
      pos_tensor[k][1] = torch.sin(angle) * pos_tensor[k][0] + torch.cos(angle) * pos_tensor[k][1]
      pos_tensor[k][0] = pos_tensor[k][0] - d/2
    for e in G.edges():
      u, v = e
      x1, y1, x2, y2 = pos_tensor[u][0], pos_tensor[u][1], pos_tensor[v][0], pos_tensor[v][1]
      x1 = -x1
      x2 = -x2
      min_d = torch.tensor(10000000.00)
      for e2 in G.edges():
        u, v = e2
        x3, y3, x4, y4 = pos_tensor[u][0], pos_tensor[u][1], pos_tensor[v][0], pos_tensor[v][1]
        if get_dis(x1, y1, x3, y3)>get_dis(x1, y1, x4, y4):
          t = x3
          x3 = x4
          x4 = t
          t = y3
          y3 = y4
          y4 = t
        cur_d = get_dis(x1, y1, x3, y3) + get_dis(x2, y2, x4, y4)
        if cur_d<min_d:
          min_d = cur_d
      ls = ls + torch.min(min_d, thrld)
    print("loss", ls)

