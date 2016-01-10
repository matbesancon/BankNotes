# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 02:04:19 2015

@author: MATHIEU
"""

#####################################################"    

## Visualize GLM in one dimension

fl = np.polyfit(data1["d"],data1["class"],deg=1)
plt.plot(data1["d"],data1["class"],'+',label="data")
plt.plot(np.arange(-2.3,2.3,0.05),np.arange(-2.3,2.3,0.05)*fl[1]+fl[0],'r',label="model")
plt.grid()
plt.xlabel("d")
plt.ylabel("outcome")
plt.title("Linear regression on a binary outcome")
plt.legend(loc="bottom right")
plt.savefig("figures/linear_binary.png")
plt.show()

plt.plot(data1["d"],data1["class"],'+',label="data",markersize=3.6)
plt.plot(np.arange(-2.5,2.5,0.05),1-1/(np.exp(np.arange(-2.5,2.5,0.05)*(5)+.5)+1),label="alpha = 0.5, beta = 5",linewidth=2)
plt.plot(np.arange(-2.5,2.5,0.05),1-1/(np.exp(np.arange(-2.5,2.5,0.05)*(2)+.5)+1),label="alpha = 0.5, beta = 2",linewidth=2)
plt.plot(np.arange(-2.5,2.5,0.05),1-1/(np.exp(np.arange(-2.5,2.5,0.05)*(5)+3)+1),label="alpha = 3, beta = 30",linewidth=2)
plt.grid()
plt.xlabel("d")
plt.ylabel("outcome")
plt.title("Logistic regression")
# plt.plot([-2.99,2.99],[0.5,0.5],'g-')
plt.plot([-.13,-.13],[0,1],'m--',label='decision boundary')
plt.legend(loc="center right",fontsize="medium")
plt.plot([-2,2],[-.01,1.005],'w+')
plt.savefig("figures/logistic_coeff.png")
plt.show()

plt.plot(data1["d"],data1["class"],'+',label="data",markersize=3.6)
plt.plot(np.arange(-2.5,2.5,0.05),1-1/(np.exp(np.arange(-2.5,2.5,0.05)*(5)+.5)+1),'r',label="model",linewidth=2)
plt.grid()
plt.xlabel("d")
plt.ylabel("outcome")
plt.title("Logistic regression")
# plt.plot([-2.99,2.99],[0.5,0.5],'g-')
plt.plot([-.13,-.13],[0,1],'m--',label='decision boundary')
plt.legend(loc="center right",fontsize="medium")
plt.plot([-2,2],[-.01,1.005],'w+')
plt.savefig("figures/logistic_binary.png")
plt.show()


#####################################################"

## Visualize GLMs in two dimensions

# variables k_resid, vari
# learning the weights
w = learn_weights(data1.iloc[:,(0,1,3)])

# building the mesh
xmesh, ymesh = np.meshgrid(np.arange(data1["vari"].min()-.5,data1["vari"].max()+.5,.01),\
    np.arange(data1["k_resid"].min()-.5,data1["k_resid"].max()+.5,.01))

pmap = pd.DataFrame(np.c_[np.ones((len(xmesh.ravel()),)),xmesh.ravel(),ymesh.ravel()])
p = np.array([])
for line in pmap.values:
    p = np.append(p,(prob_log(line,w)))
    
p = p.reshape(xmesh.shape)

plt.contourf(xmesh, ymesh, np.power(p,8), cmap= 'RdBu',alpha=.5)
plt.plot(data1[data1["class"]==1]["vari"],data1[data1["class"]==1]["k_resid"],'+',label='Class 0')
plt.plot(data1[data1["class"]==0]["vari"],data1[data1["class"]==0]["k_resid"],'r+',label='Class 1')
plt.legend(loc="upper right")
plt.title('2-dimension logistic regression result')
plt.xlabel('vari')
plt.ylabel('k_resid')
plt.grid()
plt.savefig("figures/2dimension.png")
plt.show()