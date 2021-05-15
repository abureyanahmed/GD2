import torch
from torch import nn, optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

#G = nx.cycle_graph(4)
G = nx.cycle_graph(10)
#G = nx.complete_graph(10)
#G = nx.complete_graph(5)
#G = nx.balanced_tree(2, 4)
#print(G.edges())
pos = torch.rand(G.number_of_nodes(), 2)
print(pos)
pos_nx = dict()
for i in G.nodes():
 pos_nx[i] = [pos[i][0], pos[i][1]]
nx.draw(G, pos=pos_nx, with_labels=True)
plt.show()
pos.requires_grad = True
optimizer = optim.Adam([pos], lr=0.0005)

for iter in range(200):
#for iter in range(10):
 #loss = torch.tensor([0.0 for j in range(G.number_of_nodes())])
 loss = torch.tensor(0.0)
 loss.requires_grad = True
 for i in range(G.number_of_nodes()):
 #for i in [0,G.number_of_nodes()-1]:
  ngbrs = list(G.neighbors(i))
  n = len(ngbrs)
  if n<=1:
   continue
  v = torch.index_select(pos, 0, torch.tensor(ngbrs)).sub(pos[i])
  xcoord = v[:,0]
  ycoord = v[:,1]
  norm = torch.norm(v, dim=1, keepdim=True)
  angle = torch.atan(ycoord.div(xcoord.add(norm.flatten()))).mul(2).add(math.pi)
  #angle = torch.atan(ycoord.div(xcoord))
  angle_sorted, ind = torch.sort(angle)
  angle_diff = angle_sorted[:-1].sub(angle_sorted[1:])
  last = torch.tensor(math.pi*2).add(angle_sorted[-1].sub(angle_sorted[0]))
  angle_diff = torch.cat((angle_diff, last.unsqueeze(0)))
  sensitivity = 1.0
  energy = torch.exp(angle_diff.mul(-1).mul(sensitivity)).sum()
  loss = loss.add(energy)

 print("loss", loss)

 optimizer.zero_grad()
 loss = loss.sum()
 loss.backward()
 optimizer.step()

 '''
 print(pos)
 for i in G.nodes():
  pos_nx[i] = [pos[i][0], pos[i][1]]
 nx.draw(G, pos=pos_nx, with_labels=True)
 plt.show()
 '''

print(pos)
for i in G.nodes():
 pos_nx[i] = [pos[i][0], pos[i][1]]
nx.draw(G, pos=pos_nx, with_labels=True)
plt.show()



