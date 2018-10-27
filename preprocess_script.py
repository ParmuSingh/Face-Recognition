import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('E:\workspace_py\OpenCV Cascades\haarcascades\haarcascades\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('E:\workspace_py\OpenCV Cascades\haarcascades\haarcascades\haarcascade_eye.xml')

video = cv2.VideoCapture("./prince.mp4")

i = 0
while True:
	#Capture frame-by-frame
	ret, frame = video.read()

	if not ret:
		break
	
	#Our operations on the frame come here
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	# face_cascade.load('haarcascade_frontalface_default.xml')
	for (x,y,w,h) in faces:
		# img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi = frame[y:y+h, x:x+w]
		img = cv2.resize(roi, (256, 256))
		#cv2.imshow('face', img)
		cv2.imwrite('prince' + str(i) + '.jpg', img)
	i += 1
	print(i)

#When everything done, release the capture
video.release()
