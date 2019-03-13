# Pham Hoang Vien
# B1500058
# Sang thu 3

import pandas as pd
import numpy as np

# Cau 1: Doc du lieu tu tap tin
# 1a)
data_wine_white = pd.read_csv("./text_csv/winequality-white.csv", delimiter=";")
print "1.a. Import done!"

# 1b)
print "1.b. Number of elements:",len(data_wine_white) #4898 elements
get_quality_column=data_wine_white["quality"]
number_of_labels = np.unique(get_quality_column)
print "1.b. Number of labels:",len(number_of_labels) #7 labels
print number_of_labels

# 1c) Phan chia tap du lieu huan luyen / kiem tra
from sklearn.model_selection import train_test_split
df_array = np.array(data_wine_white)
df_data = df_array[:,:11]
df_target = df_array[:,11]
X_train, X_test, y_train, y_test = train_test_split(df_data, df_target, test_size=0.2, random_state=5)
print'X_test:',len(X_test) # X_test = 980
print'y_test:',len(y_test) # y_test = 980
print '-------------------------------------'
# 1d) xay dung mo hinh cay quyet dinh
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
y_pred = clf_gini.predict(X_test)

# 1e) Tinh do chinh xac tong the va tung phan
from sklearn.metrics import accuracy_score
print'Accuracy is:', accuracy_score(y_test, y_pred)*100
print '-------------------------------------'
# 1f)
from sklearn.metrics import confusion_matrix
conf = confusion_matrix(y_test, y_pred)
#print conf

# Cau 2: Xay dung mo hinh dua vao chi so do loi thong tin
df2 = pd.read_csv('./text_csv/male-female.csv')
df2_array = np.array(df2)
x_data = df2_array[:,1:4]
y_target = df2_array[:,4]
print x_data
print y_target
print '-------------------------------------'
X_train, X_test, y_train, y_test = train_test_split(x_data, y_target)
print X_train
print y_train
print '-------------------------------------'
clf_2 = DecisionTreeClassifier(criterion='entropy', random_state=5, max_depth=3, min_samples_leaf=5)
mymodel = clf_2.fit(X_train, y_train)
# Du lieu test_model co gia tri theo de bai
# Chieu cao: 133
# Chieu dai mai toc: 37
# Giong noi: 1
test_model = np.array([[190, 16, 0]])
print test_model
print 'Gioi tinh 1-nu || 2-nam:',mymodel.predict(test_model)
# Ket qua du doan cho ra la 1 - nu.
