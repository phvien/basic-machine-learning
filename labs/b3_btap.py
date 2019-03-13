# B1500058
# Pham Hoang Vien
# Sang 3
from sklearn.model_selection import train_test_split
import pandas as pd
import math

#1)
# Phan chia tap du lieu. 
df_housing=pd.read_csv("./text_csv/Housing_2019.csv", index_col=0)
df_data = df_housing.ix[:,(1,2,4,10)]
df_target = df_housing.ix[:,(0)]
#print df_target

# Huan luyen mo hinh
from sklearn import linear_model
X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2, random_state=100)
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

# Su dung chi so MSE & RMSE danh gia mo hinh
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print "MSE:",mse

rmse = math.sqrt(mse)
print "RMSE:",rmse

#2) Viet ham LR2
import numpy as np
import matplotlib.pyplot as plt

# Tap du lieu
X=np.array([1,2,4])
Y=np.array([2,3,6])

def LR2(X, Y, alpha, times, theta0, theta1):
	m = len(X)
	theta00=theta0
	theta11=theta1
	tmp_tt0 = 0
	tmp_tt1 = 0
	for i in range(0, times):
		print "Number of time:",i
		for j in range(0,m):
			# theta0
			h=theta0 + theta11*X[j]
			sigma_tt0 = (Y[j]-h)*1
			tmp_tt0 = tmp_tt0 + sigma_tt0
			#theta0=theta0 + alpha*?*(Y[j]-h)*1
			print "Phan tu:",j, "y=",Y[j],"h=",h, "sigma_tt0 m=",j, ":", tmp_tt0
			#theta1
			h=theta00 + theta1*X[j]
			sigma_tt1 = (Y[j]-h)*X[j]
			tmp_tt1 = tmp_tt1 + sigma_tt1
			#theta1 = theta1 + alpha*?*(Y[j]-h)*X[j]
			print "Phan tu:",j, "y=",Y[j], "sigma_tt1 m=",j,tmp_tt1
			#theta00=theta0
			#theta11=theta1
		theta00 = theta0 + alpha*tmp_tt0
		theta11 = theta1 + alpha*tmp_tt1
		print "theta0=",theta00
		print "theta1=",theta11
	return [theta00, theta11]
theta=LR2(X, Y, 0.2, 2, 0, 1)
print theta

# Du doan gia tri y cho 3 phan tu sau: x=0, x=3, x=5
#theta = LR2(X,Y,0.2,2,0,1)
XX = [0,3,5]
for i in range(0,3):
	YY = theta[0]+theta[1]*XX[i]
	print "*",theta[0],"-", theta[1]
	print "**",round(YY, 3)
