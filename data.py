# from PIL import Image
from cv2 import imread
import numpy as np
from pickle import dump

data = []

# lbl = [dad, mom, me]

###############################################################################
# THIS IS TO BE USED AFTER YOU HAVE EXTRACTED FACES FROM preprocess_script.py #
###############################################################################

# EXAMPLE OF WHAT IS TO BE REPLACED BY "use path to your data": "E:/Datasets!!/family_pics/dad/dad"

# FACE 1
for i in range(330):
	try:
		img = imread("use path to your data"+str(i)+".jpg")
		img = img/255.0 # normalization
		print("face imported.")
		data.append([img, [1, 0, 0]]) # one-hot encoding : [probability of dad, probability of mom, probability of me]
	except:
		print("image not a face.")

# FACE 2
for i in range(313):
	try:
		img = imread("use path to your data"+str(i)+".jpg")
		img = img/255.0
		print("face imported.")
		data.append([img, [0, 1, 0]])
	except:
		print("image not a face.")

# FACE 3
for i in range(290):
	try:
		img = imread("use path to your data"+str(i)+".jpg")
		img = img/255.0
		print("face imported.")
		data.append([img, [0, 0, 1]])
	except:
		print("image not a face.")

# print(data, end='\n\n\n\n')
# print(data[0])

dump(data, open("data.pkl", "wb"))
