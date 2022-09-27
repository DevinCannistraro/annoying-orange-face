import cv2
import dlib
from imutils import face_utils, resize
import numpy as np

#https://www.youtube.com/watch?v=V2gmgkSqyi8
#https://medium.com/@MeerAjaz/the-function-is-not-implemented-6804e9b38b06

def createBox(img,points,scale=5):
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,points,(255,255,255))
    cv2.imshow('mask',mask)

orange_img = cv2.imread('franc.png')
orange_img = cv2.resize(orange_img, dsize=(512, 512))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    faces = detector(img)

    result = orange_img.copy()
    compositImg = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if len(faces) > 0:
        face = faces[0]
        x1,y1 = face.left(), face.top()
        x2,y2 = face.right(),face.bottom()
        compositImg = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        landmarks = predictor(imgGray,face)

        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            cv2.circle(compositImg,(x,y),5,(50,50,255),cv2.FILLED)
            cv2.putText(compositImg,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255))
    if len(myPoints) > 0:
        myPoints = np.array(myPoints)
        imgLeftEye = createBox(compositImg,myPoints[48:60])

    cv2.imshow('result', compositImg)

    you were at 20:28 in the linked youtube video. currently trying to get a much cleaner mask / outline for
        tracing the mouth and eyes. we need to test if this is a viable proceedure to do
        this seems somewhat promising.


    # cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
