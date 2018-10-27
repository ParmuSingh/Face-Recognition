# from PIL import Image
from cv2 import imread
import numpy as np
from pickle import dump

data = []

# lbl = [papa, mummy, prince]


# FACE 1
for i in range(330):
	try:
		file = open("E:/Datasets!!/family_pics/papa/papa"+str(i)+".jpg", "rb")
		# img = Image.open(file)
		# img = np.asarray(img)
		img = imread("E:/Datasets!!/family_pics/papa/papa"+str(i)+".jpg")
		img = img/255.0
		print("face imported.")
		data.append([img, [1, 0, 0]])
	except:
		print("image not a face.")

# FACE 2
for i in range(313):
	try:
		file = open("E:/Datasets!!/family_pics/mummy/mummy"+str(i)+".jpg", "rb")
		# img = Image.open(file)
		# img = np.asarray(img)
		img = imread("E:/Datasets!!/family_pics/mummy/mummy"+str(i)+".jpg")
		img = img/255.0
		print("face imported.")
		data.append([img, [0, 1, 0]])
	except:
		print("image not a face.")

# FACE 3
for i in range(290):
	try:
		file = open("E:/Datasets!!/family_pics/prince/prince"+str(i)+".jpg", "rb")
		# img = Image.open(file)
		# img = np.asarray(img)
		img = imread("E:/Datasets!!/family_pics/prince/prince"+str(i)+".jpg")
		img = img/255.0
		print("face imported.")
		data.append([img, [0, 0, 1]])
	except:
		print("image not a face.")

# print(data, end='\n\n\n\n')
# print(data[0])

dump(data, open("data.pkl", "wb"))