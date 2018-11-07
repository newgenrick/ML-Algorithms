import numpy as np 
import matplotlib.pyplot as plt

X = []
Y = []
for line in open("data_1d.csv"):
	x,y = line.split(",")
	X.append(float(x))
	Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

denominator = X.mean()*X.sum() - X.dot(X)
a = (Y.sum()*X.mean()-X.dot(Y))/denominator
b = (X.dot(Y)*X.mean()-Y.mean()*X.dot(X))/denominator

Yhat = a*X+b
plt.scatter(X,Y)
plt.plot(X,Yhat)
plt.show()

SSres = np.square(Y-Yhat).sum()
SStot = np.square(Y-Y.mean()).sum()
r2 = 1-(SSres/SStot)
print("R-squared = "+str(r2))
