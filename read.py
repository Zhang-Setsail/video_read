from time import sleep
import numpy as np
from scipy.signal import convolve2d
import cv2
import datetime
import os
import time

def bayer(img):
    img_r = np.zeros(img.shape)
    img_g = np.zeros(img.shape)
    img_b = np.zeros(img.shape)
    img_ir =  np.zeros(img.shape)
    
    img_r[2::4,0::4] = img[2::4,0::4]
    img_r[0::4,2::4] = img[0::4,2::4]
    
    img_g[1::2,0::2] = img[1::2,0::2]
    img_g[0::2,1::2] = img[0::2,1::2]
    
    img_b[0::4,0::4] = img[0::4,0::4]
    img_b[2::4,2::4] = img[2::4,2::4]
    
    img_ir[1::2,1::2] = img[1::2,1::2]
    
    return img_r, img_g, img_b, img_ir

# Demosaicing
def bilinear(img):    
    r, g, b, ir = bayer(img)
    # green interpolation
    k_g = 1/4 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    convg = convolve2d(g, k_g, 'same')
    g = g + convg
    
    # red interpolation
    k_r = 1/2 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    convr1 = convolve2d(r, k_r, 'same')
    k_r = 1/3 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    convr2 = convolve2d(r + convr1, k_r, 'same')
    k_r = 1/4 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    convr3 = convolve2d(convr2, k_r, 'same')
    r = convr2 + convr3 #+ convr2 + convr3
    
    # blue interpolation
    k_b = 1/2 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    convb1 = convolve2d(b, k_b, 'same')
    k_b = 1/3 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    convb2 = convolve2d(b + convb1, k_b, 'same')
    k_b = 1/4 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    convb3 = convolve2d(convb2, k_b, 'same')
    b =  convb2 + convb3  #b + convb1 +
    
    # ir interpolation
    k_ir = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    convir1 =convolve2d(ir, k_ir, 'same')
    k_ir = 1/4 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    convir2 =convolve2d(ir+convir1, k_ir, 'same')
    ir = ir + convir1 + convir2
    
    return r, g, b, ir

def data_process(raw_image):
    data = raw_image.tobytes()
    data = np.frombuffer(data, np.uint16)

    img = data.reshape(1944, 2592) / 4
    img = img.astype(np.uint8)
    return img


if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    # # print(cv2.CAP_DSHOW)
    # # print(cv2.CAP_PROP_EXPOSURE)
    # # print("CAP_PROP_AUTO_EXPOSURE")
    # # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    # # print(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    # # cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    # # print("CAP_PROP_EXPOSURE")
    # # print(cap.get(cv2.CAP_PROP_EXPOSURE))
    # # print(cv2.CAP_PROP_EXPOSURE)
    # # cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    # # print(cap.get(cv2.CAP_PROP_CONVERT_RGB))

    path = "./" + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir(path)

    # while True:
    for i in range(120):
        # check, raw_image = cap.read()
        filename = path + "/" + datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".npy"
        npy_filename = filename + ".npy"
        png_filename = filename + ".png"
        print(npy_filename)
        print(png_filename)
        time.sleep(2)

    # # print(cap.isOpened())
    # print(raw_image)
    # print(raw_image.shape)
    # print(raw_image.dtype)
    # print(np.max(raw_image))

        # data = raw_image.tobytes()
        # data = np.frombuffer(data, np.uint16)

        # # print(data)
        # # print(data.shape)
        # # print(data.dtype)
        # # print(np.max(data))

        # img = data.reshape(1944, 2592)
        # # print(img)
        # # print(img.shape)
        # print(np.max(img))
        # print(np.min(img))
        # np.save(npy_filename, img)

        # img = img / 4

        # r, g, b, ir = bilinear(img)

        # imga = np.asarray(np.stack((r,g,b), axis=-1), dtype="uint8" )

        # # cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        # # cv2.resizeWindow('video', 800, 600)
        # # cv2.imshow('video', imga)
        # print(np.average(ir))

        # cv2.imwrite(png_filename, imga)