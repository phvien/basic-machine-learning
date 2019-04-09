import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeRegressor


wineRed = pd.read_csv("text_csv/winequality-red.csv", sep=";")
data = wineRed.ix[:,:11]
target = wineRed["quality"]
thucte = 0
dubao = 0
sum_as = 0
from sklearn.model_selection import KFold
kf = KFold(n_splits=50, shuffle=True)
for train_index, test_index in kf.split(data):
	# In gia tri chi so cua tap huan luyen va tap ktra
	# print "Train:", train_index, "Test:",test_index
	# Tao bien X_train va X_test de luu thuoc tinh cua tap train va test
	X_train, X_test = data.ix[train_index], data.ix[test_index]
	y_train, y_test = target.ix[train_index], target.ix[test_index]
	# In thuoc tinh cua du lieu ktra
	#print "X_test:", X_test
	model = GaussianNB()
	model.fit(X_train, y_train)
	#print model
	thucte = y_test
	dubao = model.predict(X_test)
	# Cau 4)
	cnf_matrix_gnb = confusion_matrix(thucte, dubao)
	print "Danh gia: ",cnf_matrix_gnb
	# Cau 5)
	accuracy = accuracy_score(thucte, dubao)*100
	print "Do chinhs xac:",accuracy
	sum_as += accuracy
	regressor = DecisionTreeRegressor(random_state=0)
	regressor.fit(X_train,y_train)
	y_pred = regressor.predict(X_test)
	print "Accurary DicisionTree is " , accuracy_score(y_test,y_pred)*100
print "Do chinh xac tb 50 lan lap:",sum_as/50
#print "Dem X_train:",len(X_train)
#print "Dem X_test:",len(X_test)
print "------------------------------------------------------------"
#print "Thuc te:",thucte
#print "-------------------------------------------------------------"
#print "Du bao:",dubao
#print "-------------------------------------------------------------"
