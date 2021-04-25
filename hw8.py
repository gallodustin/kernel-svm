#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
from PIL import Image
import statistics


# In[39]:


three = np.loadtxt("three.txt")
eight = np.loadtxt("eight.txt")


# In[40]:


#q2.1 part 1
firstThree = three[0,:]
firstThree = firstThree.reshape((16,16))
firstThree = firstThree.transpose()

img = Image.fromarray(firstThree)
img.show(firstThree)


# In[41]:


#q2.1 part 2
firstEight = eight[0,:]
firstEight = firstEight.reshape((16,16))
firstEight = firstEight.transpose()

img = Image.fromarray(firstEight)
img.show(firstEight)


# In[44]:


#q2.2
both = np.concatenate((three, eight), axis=0)
y = []
for i in range(256):
    y.append(statistics.mean(both[:,i]))
ylong = np.array(y)
y = ylong.reshape((16,16))
y = y.transpose()
img = Image.fromarray(y)
img.show(y)


# In[51]:


#q2.3
centered = np.subtract(both,ylong)
s = np.matmul(centered.transpose(),centered)
s = s/399
print(s[0:5,0:5])


# In[67]:


#q2.4
import scipy.linalg
w,v = scipy.linalg.eigh(s, subset_by_index=[254,255])
print(w)
v1 = v[:,0]
v1 /= np.max(np.abs(v1),axis=0)
v1 *= (255/v1.max())
v1 = v1.reshape((16,16))
v1 = v1.transpose()
img = Image.fromarray(v1)
img.show(v1)
v2 = v[:,1]
v2 /= np.max(np.abs(v2),axis=0)
v2 *= (255/v2.max())
v2 = v2.reshape((16,16))
v2 = v2.transpose()
img = Image.fromarray(v2)
img.show(v2)


# In[76]:


#q2.5
import matplotlib.pyplot as plt
xv = np.matmul(centered,v)
print(xv[0])
print(xv[200])
plt.plot(xv[0:199,0], xv[0:199,1], 'rd', label="threes")
plt.plot(xv[200:399,0], xv[200:399,1], 'bs', label="eights")
plt.legend()


# In[79]:


#q3.1
# q = [A move, A stay, B move, B stay]
q = [0,0,0,0]
# A=1, B=2, move=1, stay=0
# start at state A=1
state = 1
for trial in range(200):
    if state == 1:
        move = q[0]
        stay = q[1]
        if move >= stay:
            state = 2
            q[0] = 0.5*q[0] + 0.5*(0 + 0.8*max(q[2],q[3]))
        else:
            q[1] = 0.5*q[1] + 0.5*(1 + 0.8*max(q[0],q[1]))
    else:
        move = q[2]
        stay = q[3]
        if move >= stay:
            state = 1
            q[2] = 0.5*q[2] + 0.5*(0 + 0.8*max(q[0],q[1]))
        else:
            q[3] = 0.5*q[3] + 0.5*(1 + 0.8*max(q[2],q[3]))
    print(q)


# In[85]:


#q3.2
import random
# q = [A move, A stay, B move, B stay]
q = [0,0,0,0]
# A=1, B=2, move=1, stay=0
# start at state A=1
state = 1
for trial in range(200):        
    if state == 1:
        move = q[0]
        stay = q[1]
        if random.uniform(0,1) > 0.5:
            if random.uniform(0,1) > 0.5:
                state = 2
                q[0] = 0.5*q[0] + 0.5*(0 + 0.8*max(q[2],q[3]))                
            else:
                q[1] = 0.5*q[1] + 0.5*(1 + 0.8*max(q[0],q[1]))
        elif move >= stay:
            state = 2
            q[0] = 0.5*q[0] + 0.5*(0 + 0.8*max(q[2],q[3]))
        else:
            q[1] = 0.5*q[1] + 0.5*(1 + 0.8*max(q[0],q[1]))
    else:
        move = q[2]
        stay = q[3]
        if random.uniform(0,1) > 0.5:
            if random.uniform(0,1) > 0.5:
                state = 1
                q[2] = 0.5*q[2] + 0.5*(0 + 0.8*max(q[0],q[1]))
            else:
                q[3] = 0.5*q[3] + 0.5*(1 + 0.8*max(q[2],q[3]))
        elif move >= stay:
            state = 1
            q[2] = 0.5*q[2] + 0.5*(0 + 0.8*max(q[0],q[1]))
        else:
            q[3] = 0.5*q[3] + 0.5*(1 + 0.8*max(q[2],q[3]))
    print(q)


# In[ ]:




