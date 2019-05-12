import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix

# Doc tap du lieu
dt = pd.read_csv("./text_csv/default_of_credit_card_clients.csv", delimiter=";")
data = dt.ix[1:,1:24]
target = dt.ix[1:,24]

#print data
#print target

# Phan chia tap du lieu theo phuong thuc hold-out
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

# Xay dung mo hinh huan luyen Multi-layer Perceptron Classifier
mlp = MLPClassifier(activation="logistic", solver="sgd", learning_rate_init=0.3, hidden_layer_sizes=(3,1), random_state=1000)
mlp.fit(X_train, y_train)

# Du doan nhan tap du lieu kiem tra X_test
y_pred = mlp.predict(X_test) # Du doan nhan tren tap du lieu kiem tra
print "Accuracy of MLPClassifier is:",accuracy_score(y_test, y_pred)*100

"""
# Xem trong so giua input_layer va hidden_layer_1
print "----------------------------------------------"
print "W[I;H(1)]:"
print mlp.coefs_[0]
 
# Xem trong so giua hidden_layer thu 1 va hidden_layer_2
print "W[H(1);H(2)]:"
print mlp.coefs_[1]

# Xem gia tri trong so cua w0 va w1
print "----------------------------------------------"
print "w0 =",mlp.coefs_[0][0][0]
print "w1 =",mlp.coefs_[0][1][0]

# Xem vector trong so cua 
print "----------------------------------------------"
print "Vector W[x0:H00]:",mlp.coefs_[0][:,

# Xem danh sach vector bias cua hidden_layer_1
print "----------------------------------------------"
print "DS bias hidden_layer_1:"
print mlp.intercepts_[0]

# Tuong tu la DS bias cua hidden_layer_2
print "DS bias hidden_layer_2:"
print mlp.intercepts_[1]

#------------------------------------------------------------------------------------------------------------
#  Tinh do chinh xac voi giai thuat cay quyet dinh:
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print "Accuracy of DecistionTreeClassifier is:",accuracy_score(y_test, y_pred)*100
cmatrix = confusion_matrix(y_test, y_pred, labels=[0,1])
"""

"""
# Do chinh xac tong the cua MLPClassifier sau 10 lan lap:
for i in range(0,10):
    # Phan chia theo nghi thuc hold-out
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=10*i)

    # Xay dung mo hinh MLPClassifier:
    mlp = MLPClassifier(activation="logistic", solver="sgd", learning_rate_init=0.2, hidden_layer_sizes=(3,1), random_state=5)
    mlp.fit(X_train, y_train)
    
    # Gia tri du bao
    y_pred = mlp.predict(X_test)
    
    # Do chinh xac tung phan lop cua tung vao lap   
    from sklearn.metrics import confusion_matrix
    c_matrix = confusion_matrix(y_test, y_pred)
    print "Do chinh xac cua an lap", i,":"
    print c_matrix
    # Do chinh xac tong the cua lan lap thu i
    print "Accuracy", i,":", accuracy_score(y_test, y_pred)*100
    
# So sanh voi giai thuat cay quyet dinh sau 10 lan lap:
for i in range(0,10):
    # Phan chia theo nghi thuc hold-out
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=10*i)

    # Xay dung mo hinh cay quyet dinh:
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth = 3, min_samples_leaf = 5)
    clf.fit(X_train, y_train)
    # Gia tri du bao
    y_pred = clf.predict(X_test)
    
    # Do chinh xac tung phan lop cua tung vao lap   
    from sklearn.metrics import confusion_matrix
    cmatrix = confusion_matrix(y_test, y_pred)
    print "Do chinh xac cua an lap(DecisionTreeClassifier)", i,":"
    print ">>>",cmatrix

    # Do chinh xac tong the cua lan lap thu i
    print "Accuracy", i,":", accuracy_score(y_test, y_pred)*100    

"""
