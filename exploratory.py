# -*- coding: utf-8 -*-

"""
Created on Sat Dec 26 13:45:58 2015

@title: exploratory.py
@description: Visualization of the banknote dataset
@author: Mathieu Besancon
"""

# exploratory
print(sum(data0['class'])/len(data0['class']))

plt.plot(data0["vari"])
plt.plot(data0["skew"])
plt.plot(data0["kurtosis"])
plt.plot(data0["entropy"])


data0.groupby('class').vari.plot()
data0.groupby('class').skew.plot()
data0.groupby('class').kurtosis.plot()
data0.groupby('class').entropy.plot()
 
# pd.tools.plotting.parallel_coordinates(data0, 'class')
 

for v in data0.columns[:4]:
    ggplot.ggsave(ggplot.ggplot(ggplot.aes(x=v, color='class'),\
                    data=data0)+ ggplot.geom_density()+ \
                    ggplot.geom_point(ggplot.aes(y=0),alpha=0.2)+ \
                    ggplot.ggtitle('KDE '+v)+
                    ggplot.ylab("KDE"), 'KDE_'+v+'.png',width=18,height=12)

col = list('r' if i==1 else 'b' for i in data0["class"])    
pd.tools.plotting.scatter_matrix(data0.ix[:,:4],figsize=(6,3),
                                 color=col,diagonal='kde')

d0 = data0[data0["class"]==0]
d1 = data0[data0["class"]==1]

for v in data0.columns[:4]:
    plt.figure(figsize=(9,4)) 
    ax1 =  plt.subplot(121)
    stats.probplot(d0[v],dist='norm',plot=plt)
    plt.title("Normal QQ-plot "+v + " - Class 0")
    plt.grid()
    ax2 = plt.subplot(122)
    stats.probplot(d1[v],dist='norm',plot=plt)
    plt.grid()    
    plt.title("Normal QQ-plot "+v + " - Class 1")
    plt.savefig("qqplot_"+v+".png")
    plt.show()
    
plt.pcolor(data0.corr())
plt.colorbar()
plt.yticks(np.arange(0.5,5.5),range(0,5))
plt.xticks(np.arange(0.5,5.5),range(0,5))
plt.show()

# Boxplot
data0.groupby("class").boxplot(figsize=(9,5))
plt.savefig("boxplot.png")

plt.bar(np.arange(4),np.var(data0.iloc[:,:4]),width=.5)
plt.xticks(np.arange(4)+.25,['Variance','Skewness','Kurtosis','Entropy'])
plt.grid()
plt.title('Variance of the different variables')
plt.ylabel('Variance')


