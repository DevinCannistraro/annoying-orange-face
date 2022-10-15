import cv2
import dlib
from imutils import face_utils, resize
import numpy as np
import skimage
import os
import PIL
from PIL import Image

#https://www.youtube.com/watch?v=V2gmgkSqyi8
#https://medium.com/@MeerAjaz/the-function-is-not-implemented-6804e9b38b06

def createBox(img,points,scale=5):
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask,points,(255,255,255))
    bbox = cv2.boundingRect(points)
    x,y,w,h = bbox

    masked = cv2.bitwise_and(img, mask)
    imgCrop = masked[y:y+h,x:x+w]
    #imgCrop = resize(imgCrop, width=200)

    # make transparent

    alpha = np.sum(imgCrop, axis=-1) > 0

    # Convert True/False to 0/255 and change type to "uint8" to match "na"
    alpha = np.uint8(alpha * 255)

    # Stack new alpha layer with existing image to go from BGR to BGRA, i.e. 3 channels to 4 channels
    res = np.dstack((imgCrop, alpha))
    imgCrop = res

    return imgCrop

def anti_alias_resize_path(im,scaler,name):
    print("test")
    width, height = im.size
    im = im.resize((width * scaler, height * scaler), resample=Image.Resampling.LANCZOS)
    im.save("TEMP_IMS//" + name + ".png")


def do_PIL_processing(mouth_path,left_eye_path,right_eye_path):
    mouth_im = Image.open(mouth_path)
    left_eye_im = Image.open(left_eye_path)
    right_eye_im = Image.open(right_eye_path)
    anti_alias_resize_path(mouth_im,20,"mouth_resize")
    anti_alias_resize_path(left_eye_im, 20, "left_resize")
    anti_alias_resize_path(right_eye_im, 20, "right_resize")

def add_image_at_position(base,overlay,position):
    x_offset,y_offset = position
    base[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(5)

if not os.path.exists("TEMP_IMS"):
    os.mkdir("TEMP_IMS")

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    faces = detector(img)

    compositImg = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if len(faces) > 0:
        face = faces[0]
        x1,y1 = face.left(), face.top()
        x2,y2 = face.right(),face.bottom()
        #compositImg = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        landmarks = predictor(imgGray,face)

        myPoints = []
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
            #cv2.circle(compositImg,(x,y),5,(50,50,255),cv2.FILLED)
            #cv2.putText(compositImg,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,0,255))
        if len(myPoints) > 0:
            myPoints = np.array(myPoints)
            imgMouth = createBox(img,np.int32([myPoints[48:60]]))
            imgLeftEye = createBox(img, np.int32([myPoints[36:42]]))
            imgRightEye = createBox(img, np.int32([myPoints[42:48]]))

            cv2_intermediate_mouth_path = "TEMP_IMS//intermediate_mouth.png"
            cv2_intermediate_left_eye_path = "TEMP_IMS//intermediate_left_eye.png"
            cv2_intermediate_right_eye_path = "TEMP_IMS//intermediate_right_eye.png"
            cv2.imwrite(cv2_intermediate_mouth_path, imgMouth)
            cv2.imwrite(cv2_intermediate_left_eye_path, imgLeftEye)
            cv2.imwrite(cv2_intermediate_right_eye_path, imgRightEye)

            do_PIL_processing(cv2_intermediate_mouth_path,cv2_intermediate_left_eye_path,cv2_intermediate_right_eye_path)

    cv2.imshow('result', compositImg)

    #you were at 20:28 in the linked youtube video. currently trying to get a much cleaner mask / outline for
    #    tracing the mouth and eyes. we need to test if this is a viable proceedure to do
    #    this seems somewhat promising.


    # cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
