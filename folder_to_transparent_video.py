import os
import subprocess
import cv2
import shutil
import numpy as np
import skimage.exposure

#ffmpeg -i test_ims_for_video/%04d.png -r 30  -pix_fmt yuva420p video.webm
#ffmpeg -i test_ims_for_video/%04d.png -r 30 -vcodec png z.mov


TEMP_FOLDER = "temp"
INPUT_FILE = "formatted_face_green.mp4"
FRAME_QUALITY = 3


def makeTransparent(path,outputPath):
    src = cv2.imread(path)

    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,15,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)

    cv2.imwrite(outputPath,dst)


def makeTransparentChroma(inputFolder,outputFolder):
    # this would be great if it made it transparent https://stackoverflow.com/questions/51719472/remove-green-background-screen-from-image-using-opencv-python?noredirect=1&lq=1
    # load image
    img = cv2.imread(inputFolder)

    # convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # extract A channel
    A = lab[:, :, 1]

    # threshold A channel
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # blur threshold image
    blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=5, sigmaY=5, borderType=cv2.BORDER_DEFAULT)

    # stretch so that 255 -> 255 and 127.5 -> 0
    mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255)).astype(np.uint8)

    # add mask to image as alpha channel
    result = img.copy()
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    cv2.imwrite(outputFolder,result)


def makeFolderTransparent(inputFolder,outputFolder):
    if not os.path.exists(outputFolder):
        os.mkdir(outputFolder)
    else:
        shutil.rmtree(outputFolder)
        os.mkdir(outputFolder)

    for index,file in enumerate(os.listdir(inputFolder)):
        outputFile = outputFolder + "/" + str(index).zfill(6) + ".png"
        makeTransparentChroma(inputFolder+"/"+file,outputFile)


if not os.path.exists(TEMP_FOLDER):
    os.mkdir(TEMP_FOLDER)
else:
    shutil.rmtree(TEMP_FOLDER)
    os.mkdir(TEMP_FOLDER)

print("Making frames")
command = "ffmpeg -i "+INPUT_FILE+" -qscale:v "+str(FRAME_QUALITY)+" "+TEMP_FOLDER+"/frame%06d.jpg -hide_banner" + " -loglevel error"
subprocess.call(command, shell=True)

makeFolderTransparent(TEMP_FOLDER,"temp_output")

command = "ffmpeg -i temp_output/%06d.png -r 30 -vcodec png z4.mov"
subprocess.call(command)

shutil.rmtree(TEMP_FOLDER)
shutil.rmtree("temp_output")

#command = "ffmpeg -i formatted_face.mp4 -vf colorkey=0x000000:0.1:0.1 -pix_fmt yuva420p output.webm"
#subprocess.call(command)
#command = "ffmpeg -i output.webm -vcodec png z3.mov"
#subprocess.call(command)
