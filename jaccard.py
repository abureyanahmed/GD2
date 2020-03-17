
def M_c(y_opt, y_tilde):
 r = [0 for i in range(len(y_opt))]
 for i in range(len(y_opt)):
  if y_opt[i]!=y_tilde[i]:
   r[i] = 1
 return r

def delta(y_opt, y_tilde):
 r = M_c(y_opt, y_tilde)
 r_union = [0 for i in range(len(y_opt))]
 for i in range(len(y_opt)):
  if y_opt[i]==1:
   r_union[i] = 1
  elif r[i]==1:
   r_union[i] = 1
 if sum(r_union)==0: return 1
 else: return sum(r)/sum(r_union)

def lovasz(m, delta, y_opt, y_tilde):
 p = len(m)
 print("m:", m)
 myy = zip(m, y_opt, y_tilde)
 print("myy:", myy)
 res = sorted(myy, key = lambda myy:myy[0])
 print("res:", res)
 res = reversed(res)
 print("reversed:", res)
 m, y_opt, y_tilde = zip(*res)
 print("m, y_opt, y_tilde =", m, y_opt, y_tilde)
 s = 0
 for i in range(p):
  s += m[i]*(delta(y_opt[:i+1], y_tilde[:i+1])-delta(y_opt[:i], y_tilde[:i]))
 return s

y_opt = [-1, -1, 1, 1, 1]
y_tilde = [1, -1, -1, 1, 1]
r = M_c(y_opt, y_tilde)
print("r:", r)
jac = delta(y_opt, y_tilde)
print("jac:", jac)
m = [.1, .5, .4, .2, .7]
s = lovasz(m, delta, y_opt, y_tilde)
print("lovasz:", s)
