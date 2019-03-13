# Pham Hoang Vien
# MSSV: B1500058

import numpy as np

#Cau 1:
A = np.array([2,5,3,6,7,8])
print A

#Cau 2:
B = np.array([range(1,201)])
print B

#Cau 3:
C = np.linspace(start=0, stop=1000, num=500+1, dtype=int)
print C

#Cau 4:
A5 = A + 5
print A5

#Cau 5:
B3 = B * 3
print B3

#Cau 6:
A.sort()
A_sort = A
print 'In mang A da sap xep'
print A_sort

#Cau 7:
Dict = {
	'Name': 'Pham Hoang Vien',
	'Age': '20',
	'Course': 'Nguyen Ly May Hoc'
}
print Dict

#Cau 8:
Dict['Course'] = 'Tri Tue Nhan Tao'
print Dict

#Cau 9:
name = raw_input('Moi ban nhap ten: ')
print ('Hello, ' + name)

#Cau 10:
a = input('a = ')
b = input('b = ')
c = input('c = ')
if a == 0:
	print 'Nhap a > 0!'
	a = input('a = ')	
	delta = (b*b) - (4*a*c)
elif delta < 0:
	print 'Phuong trinh vo nghiem'
elif delta == 0:
	x = -b/2*a
	print ('Phuong trinh co nghiem kep' + x)
else:
	x1 = (b*np.sprt(delta))/(2*a)	
	x2 = (-b*np.sqrt(delta))/(2*a)
	print 'Phuong trinh co nghiem x1: ' + x1
	print 'Phuong trinh co nghiem x2: ' + x2

#Cau 11:
aa = input('x = ')
bb = input('y = ')
cc = input('z = ')
maxnumber = max(aa, bb, cc)		
print 'So lon nhat trong 3 so nhap vao la ' + maxnumber

#Cau 12:
X = np.array([1,2,3,4,5,6,7,8,9])
X = X.reshape((3,3))
print X

#Cau 13:
Y = [11,22,33,44,55,66,77,88,99,111,222,333]
Y = Y.reshape((3,4))
print Y

#Cau 14:
Z = np.dot(X,Y)
print Z

#Cau 15:
import matplotlib.pylot as plt
x = np.linspace(-10, 10, 10000)
y = np.sin(x)
plt.scatter(x,y)
plt.show()

#Cau 16:
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace (-5,5,10000)
y = x*x*x - 2*(x*x) + x +5
plt.scatter(x,y)
plt.show()
