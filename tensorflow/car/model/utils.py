import random
import pickle
import math
from io import BytesIO
import os
import glob
import socket

from PIL import Image
import numpy as np

import envoy


def scale(img, size=128):
    '''
    accepts: PIL image, size of square sides
    returns: PIL image scaled so sides lenght = size 
    '''
    size = (size, size)
    img.thumbnail(size, Image.ANTIALIAS)
    return img

def img_to_binary(img):
    '''
    accepts: PIL image
    returns: binary stream (used to save to database)
    '''
    file = BytesIO()
    img.save(file, format='jpeg')
    return file.getvalue()

def arr_to_binary(arr):
    '''
    accepts: numpy array with shape (Hight, Width, Channels)
    returns: binary stream (used to save to database)
    '''
    img = arr_to_img(arr)
    return img_to_binary(img)

def arr_to_img(arr):
    '''
    accepts: numpy array with shape (Hight, Width, Channels)
    returns: binary stream (used to save to database)
    '''
    arr = np.uint8(arr)
    img = Image.fromarray(arr)
    return img

def binary_to_img(binary):
    '''
    accepts: binary file object from BytesIO
    returns: PIL image
    '''
    img = BytesIO(binary)
    return Image.open(img)

def create_video(img_dir_path, out_video_path):
    # Setup path to the images with telemetry.
    full_path = os.path.join(img_dir_path, 'frame_*.png')

    # Run ffmpeg.
    command = ("""ffmpeg
               -framerate 30/1
               -pattern_type glob -i '%s'
               -c:v libx264
               -r 15
               -pix_fmt yuv420p
               -y
               %s""" % (full_path, out_video_path))
    response = envoy.run(command)