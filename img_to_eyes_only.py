import cv2
import dlib
from imutils import face_utils, resize
import numpy as np
import skimage
import os
import PIL
from PIL import Image
from scipy import interpolate
from matplotlib import pyplot as plt

#https://www.youtube.com/watch?v=V2gmgkSqyi8
#https://medium.com/@MeerAjaz/the-function-is-not-implemented-6804e9b38b06

def createBox(img,points):
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

def resize_image_CV2(im,scalar):
    width = int(im.shape[1] * scalar)
    height = int(im.shape[0] * scalar)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized

def resize_points(points,scalar):
    new_points = points * scalar
    return new_points

def anti_alias_resize_path_PIL(im,scaler,name):
    width, height = im.size
    im = im.resize((width * scaler, height * scaler), resample=Image.Resampling.LANCZOS)
    return im #im.save("TEMP_IMS//" + name + ".png")

def do_PIL_processing(mouth_path,left_eye_path,right_eye_path):
    mouth_im = Image.open(mouth_path)
    left_eye_im = Image.open(left_eye_path)
    right_eye_im = Image.open(right_eye_path)
    mouth_im = anti_alias_resize_path_PIL(mouth_im,2,"mouth_resize")
    left_eye_im = anti_alias_resize_path_PIL(left_eye_im, 2, "left_resize")
    right_eye_im = anti_alias_resize_path_PIL(right_eye_im, 2, "right_resize")

    blank_image = Image.new("RGBA",(1920,1080),(0,0,0,0))

    blank_image.paste(mouth_im,(300,100),mouth_im)
    blank_image.paste(left_eye_im, (1000, 400), left_eye_im)
    blank_image.paste(right_eye_im, (1500, 700), right_eye_im)

    #Image.Image.paste(blank_image,mouth_im,(300,100))
    #Image.Image.paste(blank_image, left_eye_im, (1000, 400))
    #Image.Image.paste(blank_image, left_eye_im, (1500, 700))

    blank_image.save("TEMP_IMS//blank" + ".png")




def add_image_at_position(base,overlay,position):
    x_offset,y_offset = position
    base[y_offset:y_offset + overlay.shape[0], x_offset:x_offset + overlay.shape[1]] = overlay

def get_interpolated_points(np_points,desired_num_points):
    x_vals = []
    y_vals = []
    for point in np_points:
        x_vals.append(point[0])
        y_vals.append(point[1])
    x = np.array(x_vals)
    y = np.array(y_vals)

    # append the starting x,y coordinates
    x = np.r_[x, x[0]]
    y = np.r_[y, y[0]]

    # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
    # is needed in order to force the spline fit to pass through all the input points.
    tck, u = interpolate.splprep([x, y], s=0, per=True)

    # evaluate the spline fits for 1000 evenly spaced distance values
    xi, yi = interpolate.splev(np.linspace(0, 1, desired_num_points), tck)

    # plot the result
    #fig, ax = plt.subplots(1, 1)
    #ax.plot(x, y, 'or')
    #ax.plot(xi, yi, '-b')
    #print(len(xi))
    #plt.show()

    new_points = []
    for index in range(len(xi)):
        new_points.append([xi[index],yi[index]])

    return np.array(new_points)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

if not os.path.exists("TEMP_IMS"):
    os.mkdir("TEMP_IMS")

img = cv2.imread("test_im.jpg")

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
        INITIAL_SCALE_FACTOR = 10 # must be int used to scale up to get smoother edges after slicing
        DOWNSAMPLE_SCALE_FACTOR = .2 # scaled down after to increase edge smoothness
        # scale up points so we can downscale after and have higher res
        myPoints = resize_points(myPoints,INITIAL_SCALE_FACTOR)
        img = resize_image_CV2(img,INITIAL_SCALE_FACTOR)

        mouth_points = np.int32([get_interpolated_points(myPoints[48:60],200)])
        imgMouth = createBox(img,mouth_points)
        left_eye_points = np.int32([get_interpolated_points(myPoints[36:42], 200)])
        imgLeftEye = createBox(img, left_eye_points)
        right_eye_points = np.int32([get_interpolated_points(myPoints[42:48], 200)])
        imgRightEye = createBox(img, right_eye_points)

        #downscale
        #imgMouth = resize_image_CV2(imgMouth, DOWNSAMPLE_SCALE_FACTOR)
        #imgLeftEye = resize_image_CV2(imgLeftEye, DOWNSAMPLE_SCALE_FACTOR)
        #imgRightEye = resize_image_CV2(imgRightEye, DOWNSAMPLE_SCALE_FACTOR)

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

# issues
# make transparent off mask so eyeball doesn't go transparent
#the key is to upscale before you mask but after you detect.
#to do this just multiply the detected points by the scale factor
#upscale it a large amount then downscale after masking to get clean edges

#after doing the above see if you can get the alpha to work off only making it
#transparent where the mask is black. this will ideally make it not make the pupil
#transparent