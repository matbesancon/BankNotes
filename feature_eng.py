# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 18:23:27 2015

@author: MATHIEU
"""

ggplot.ggplot(ggplot.aes(x='skew',y='kurtosis'),data=data0)+ \
    ggplot.geom_point() + ggplot.geom_smooth(method='lm')
plt.show()

a, b = stats.linregress(data0["skew"],data0["kurtosis"])[:2]
plt.plot(data0["skew"],data0["kurtosis"],'+')
plt.plot( np.array([-15,15]) ,b+a*np.array([-15,15]),'r')
plt.title('Simple linear regression')
plt.xlabel('Skewness')
plt.ylabel('Kurtosis')
plt.savefig("figures/linear_reg.png")
plt.show()

a, b, c = np.polyfit(data0["skew"],data0["kurtosis"],deg=2)
plt.plot(data0["skew"],data0["kurtosis"],'+')
plt.plot(np.arange(-15,15,.5) ,c+b*np.arange(-15,15,.5)+a*np.arange(-15,15,.5)*np.arange(-15,15,.5),'r')
plt.title('2nd degree polynomial regression')
plt.xlabel('Skewness')
plt.ylabel('Kurtosis')
plt.show()

d0 = data0[data0["class"]==0]
d1 = data0[data0["class"]==1]

p0 = plt.scatter(d0['skew'],c+b*d0["skew"] +a*d0["skew"]*d0["skew"]-d0["kurtosis"],c='b',marker='+',label="0")
p1 = plt.scatter(d1['skew'],c+b*d1["skew"] +a*d1["skew"]*d1["skew"]-d1["kurtosis"],c='r',marker='+',label="1")
plt.title('Explanatory variable vs Regression residuals')
plt.xlabel('Skewness')
plt.ylabel('Residuals')
plt.legend(["0","1"])
plt.show()

plt.plot(d0["skew"],d0["entropy"],'+',label="Class 0")
plt.plot(d1["skew"],d1["entropy"],'r+',label="Class 1")
plt.xlabel("Skewness")
plt.ylabel("Entropy")
plt.grid()
plt.legend()
plt.show()

ft = np.polyfit(data0["skew"],data0["entropy"],deg=2)
plt.plot(d0["skew"],d0["entropy"],'+',label="Class 0")
plt.plot(d1["skew"],d1["entropy"],'r+',label="Class 1")
plt.plot(np.arange(-15,14.5,.5),
         ft[0]*np.arange(-15,14.5,.5)*np.arange(-15,14.5,.5)+ft[1]*np.arange(-15,14.5,.5)+ft[2],'-',linewidth=2 ,label="Fitted polynom")
plt.xlabel("Skewness")
plt.ylabel("Entropy")
plt.grid()
plt.legend(loc="bottom center")
plt.show()

# Class dependant polynomial regression
f0 = np.polyfit(d0["skew"],d0["entropy"],deg=2)
x = np.arange(-15,14,.5)
f1 = np.polyfit(d1["skew"],d1["entropy"],deg=2)

plt.plot(x,f0[0]*x*x+f0[1]*x+f0[2],'-',label="Fitted 0")
plt.plot(d0["skew"],d0["entropy"],'+',alpha=.7,label="Class 0")

plt.plot(x,f1[0]*x*x+f1[1]*x+f1[2],'-',label="Fitted 1")
plt.plot(d1["skew"],d1["entropy"],'m+',alpha=.7,label="Class 1")

plt.title("Class dependant fit")
plt.xlabel("Skewness")
plt.ylabel("Entropy")
plt.grid()
plt.legend(loc='bottom center')
plt.savefig("class_depend.png")
plt.show()

# residuals of the class-dependent model
plt.plot(d0["skew"],f0[0]*d0["skew"]*d0["skew"]+f0[1]*d0["skew"]+f0[2]-d0["entropy"],'b+',label="Class 0")
plt.plot(d1["skew"],f1[0]*d1["skew"]*d1["skew"]+f1[1]*d1["skew"]+f1[2]-d1["entropy"],'r+',label="Class 1")
plt.legend()
plt.grid()
plt.xlabel("Skewness")
plt.ylabel("Residuals")
plt.savefig("res_class_dep.png")
plt.show()


# Computing the d variable 
d = (data0["entropy"]-f0[0]*data0["skew"]*data0["skew"]-f0[1]*data0["skew"]-f0[2])**2-\
    (data0["entropy"]-f1[0]*data0["skew"]*data0["skew"]-f1[1]*data0["skew"]-f1[2])**2

d0["d"] = d[data0["class"]==0]
d1["d"] = d[data0["class"]==1]

plt.grid()
plt.plot(d0["skew"],d0["d"],'b+',label="Class 0")
plt.plot(d1["skew"],d1["d"],'r+',label="Class 1")
plt.legend()
plt.title("d vs skewness for each class")
plt.xlabel("Skewness")
plt.ylabel("d")
plt.show()

# Fitting a polynomial model between the skew and kurtosis
a, b, c = np.polyfit(data0["skew"],data0["kurtosis"],deg=2)

# copying the data
data1 = data0.copy()

# computing the residual of the regression against kurtosis
data1.columns = ['vari', 'skew', 'k_resid', 'entropy', 'class']
data1["k_resid"] = data0["kurtosis"] - (a*(data0["skew"])**2 + b*data0["skew"] + c)

# computing the feature from the entropy regression
data1.columns  = ['vari', 'skew', 'k_resid', 'd', 'class']

# Class-dependent model
f0 = np.polyfit(d0["skew"],d0["entropy"],deg=2)
f1 = np.polyfit(d1["skew"],d1["entropy"],deg=2)

data1["d"] = abs(data0["entropy"]-f0[0]*data0["skew"]*data0["skew"]-f0[1]*data0["skew"]-f0[2])-\
    abs(data0["entropy"]-f1[0]*data0["skew"]*data0["skew"]-f1[1]*data0["skew"]-f1[2])

data1 = data1.drop("skew",1)
# data normalization
data1.iloc[:,:3] = data1.iloc[:,:3]/np.sqrt(np.var(data1.iloc[:,:3]))

col = list('r' if i==1 else 'b' for i in data1["class"])    
pd.tools.plotting.scatter_matrix(data1.ix[:,:3],figsize=(6,3),
                                 color=col,diagonal='kde')
plt.show()



