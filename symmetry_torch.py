import torch
from torch import nn, optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

def reflection(p, cntr, rot_mat, ant_rot_mat, axis='X'):
 rfx = torch.tensor([-1.0, 1.0])
 rfy = torch.tensor([1.0, -1.0])
 if axis=='X':
  return ((p-cntr).matmul(rot_mat)*rfx.matmul(ant_rot_mat))+cntr
 else:
  return ((p-cntr).matmul(rot_mat)*rfy.matmul(ant_rot_mat))+cntr

niter = 20000

def getLR(i):
    base_lr = 0.5
    return np.exp(-i*2/niter) * base_lr

def euclidean_dis(x, y):
 z = x - y
 return z.pow(2).sum(axis=1)

'''
tensor([[0.2567, 0.7063],
        [0.3638, 0.0835],
        [0.5345, 0.7840],
        [0.4646, 0.2164]])
tensor([[ 22.4828,   6.9211],
        [  8.4237,  10.7121],
        [-21.6916,  -5.4308],
        [ -7.5953, -10.4122]], requires_grad=True)
'''

#G = nx.cycle_graph(4)
G = nx.cycle_graph(10)
#G = nx.complete_graph(10)
#G = nx.balanced_tree(2, 4)
#print(G.edges())
pos = torch.rand(G.number_of_nodes(), 2)
#pos = torch.tensor([[0.1, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
'''
pos = torch.tensor([[0.2567, 0.7063],
        [0.3638, 0.0835],
        [0.5345, 0.7840],
        [0.4646, 0.2164]])
'''
print(pos)
pos_nx = dict()
for i in G.nodes():
 pos_nx[i] = [pos[i][0], pos[i][1]]
nx.draw(G, pos=pos_nx, with_labels=True)
plt.show()
pos.requires_grad = True
#optimizer = optim.SGD([pos], lr=0.5, momentum=0.9, nesterov=True)
#optimizer = optim.SGD([pos], lr=0.01, momentum=0.9, nesterov=True)
#optimizer = optim.Adam([pos], lr=0.0001)
optimizer = optim.Adam([pos], lr=0.001)

#cntr = pos.sum(axis=0).div(G.number_of_nodes())

#for i in range(500):
for i in range(50):
#for i in range(400):
 '''
 pos_r = torch.transpose(pos, 0, -1).repeat(pos.shape[0], 1)
 pos_m = pos.repeat(1, pos.shape[0]).sub(pos_r).div(2)
 print(pos)
 print(pos_m)
 '''
 loss = torch.tensor(0.0)
 loss.requires_grad = True
 for j in range(G.number_of_nodes()):
  for k in range(G.number_of_nodes()):
   if j==k:
    continue
   cntr = pos[j].add(pos[k]).div(2)
   ang_vec = pos[j].sub(pos[k])
   ang = torch.atan(ang_vec[1].div(ang_vec[0]))
   ang = ang.item()
   rot_mat = torch.tensor([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
   ant_rot_mat = torch.tensor([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
   #cntr = pos.sum(axis=0).div(G.number_of_nodes())
   #if i%5==0:
   # cntr = pos.sum(axis=0).div(G.number_of_nodes())
   #print(cntr)
   edges = torch.stack([torch.cat([pos[u], pos[v]]) for u, v in G.edges()])
   #print(edges)
   m = len(list(G.edges()))
   #cntr = cntr.repeat(m*m, 1)
   e0 = edges.repeat(1, m).view(-1,edges.shape[1])
   e1 = edges.repeat(m, 1)
   p1, p2, p3, p4 = e0[:,:2], e0[:, 2:], e1[:,:2], e1[:,2:]
   #print("p3", p3)
   rfp3 = reflection(p3, cntr, rot_mat, ant_rot_mat)
   #print("rfp3", rfp3)
   rfp4 = reflection(p4, cntr, rot_mat, ant_rot_mat)
   dif1 = euclidean_dis(p1, rfp3) + euclidean_dis(p2, rfp4)
   #dif1 = euclidean_dis(p1, p3) + euclidean_dis(p2, p4)
   #print("dif1", dif1)
   dif2 = euclidean_dis(p1, rfp4) + euclidean_dis(p2, rfp3)
   #dif2 = euclidean_dis(p1, p4) + euclidean_dis(p2, p3)
   dif = torch.min(dif1, dif2)
   #print(dif)
   #loss = torch.clamp(dif, 0, 10.0).sum()
   #loss = loss.add(torch.clamp(dif, 0, 10.0).sum())
   #loss = torch.clamp(dif, 0, 1.0).sum()
   # The following tries to allign all edges
   #loss = dif.sum()
   #dif = torch.clamp(dif, 0, 0.2).sum()
   tol_x = pos[:,0].max().sub(pos[:,0].min())
   tol_y = pos[:,1].max().sub(pos[:,1].min())
   tol = torch.min(tol_x, tol_y)
   dif = torch.clamp(dif, 0, tol.div(10).item()).sum()
   loss = loss.add(dif.sum())
   #loss = dif1.sum()
   #loss = torch.clamp(dif, .05, 10.0).sum()

 print("loss", loss)

 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
 #scheduler = optim.lr_scheduler.LambdaLR(optimizer, getLR)
 #scheduler.step()

 '''
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



