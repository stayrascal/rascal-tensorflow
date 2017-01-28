import os
import time
import numpy as np
from PIL import Image

import pickle

import model

class Session():
    '''
    Class to store images and vehicle data to the local file system and later retrieve them as arrays or generators.
    '''

    def __init__(self, path):
        print('Loading Session')
        self.session_dir = path
        self.frame_count = 0

    def put(self, img, angle=None, throttle=None, milliseconds=None):
        '''
        Save image with encoded angle, throttle and time data in the filename
        '''
        self.frame_count += 1

        file_name = str("%s/" % self.session_dir +
                        "frame_" + str(self.frame_count).zfill(5) + 
                        "_ttl_" + str(throttle) + 
                        "_agl_" + str(angle) + 
                        "_mil_" + str(milliseconds) +
                        '.jpg')
        img.save(file_name, 'jpeg')

    def get(self, file_path):
        '''
        Retrieve an image and the data saved with it.
        '''

        with Image.open(file_path) as img:
            img_arr = np.array(img)

        file = file_path.split('/')[-1]
        file = file.split('.')[0]
        file = file.split('_')

        throttle = int(file[3])
        angle = int(file[5])
        milliseconds = int(file[7])

        data = {'throttle':throttle, 'angle':angle, 'milliseconds': milliseconds}

        return img_arr, data
    
    def img_paths(self):
        '''
        Return a list of file paths for the images in the session.
        '''
        files = os.listdir(self.session_dir)
        files = [file for file in files if file[-3:] == 'jpg']
        files.sort()
        file_paths = [os.path.join(self.session_dir, file) for file in files]
        return file_paths

    def img_count(self):
        return len(self.img_paths())

    def load_dataset(self, img_paths=None):
        '''
        Return image arrays and data arrays.

            X - array of n samples if immage arrays representing.
            Y - array with the shape (samples, 1) containing the angle lable for each image

            Where n is the number of recorded images.
        '''

        gen = self.load_generator(img_paths=img_paths)

        X = [] # images
        Y = [] #velocity (angle, speed)
        for _ in range(self.img_count()):
            x, y = next(gen)
            X.append(x)
            Y.append(y)

        X = np.array(x) # iamge array [[image1]], [[image2]]
        Y = np.array(Y) # array [[angle1, speed1], [angle2, speed2] ...]

        return X, Y

    def load_generator(self, img_paths=None, add_dim=False):
        '''
        Return a generator that will loops through image arrays and data labels.

        add_dim: add dimension to returned arrays as needed by keras
        '''
        if img_paths == None:
            img_paths = self.img_paths()
        
        while True:
            for file in img_paths:
                img_arr, data = self.get(file)

                # return only angle for now
                data_arr = np.array(data['angle'])

                if add_dim == True:
                    data_arr = data_arr.reshape((1,) + data_arr.shape)
                    img_arr = img_arr.reshape((1,) + img_arr.shape)
                yield img_arr, data

        def vaiant_generator(self, img_paths, variant_funcs):
            def orient(arr, flip=False):
                if flip == False:
                    return arr
                else:
                    return np.fliplr(arr)

            print('images before variants %s' % len(img_paths))

            while True:
                for flip in [True, False]:
                    for v in variant_funcs:
                        for img_path in img_paths:
                            img = Image.open(img_path)
                            img = np.array(img)
                            img = v['func'](img, **v['args'])
                            img = orient(img, flip=flip)
                            angle, speed = self.parse_file_name(img_path)
                            if flip == True:
                                angle = -angle # reverse stering angle
                            y = np.array([angle, speed])
                            x = np.expand_dims(x, axis=0)
                            y = y.reshape(1, 2)
                            yield x, y
        
    class SessionHandler():
        '''
        convienience class to create and load sessions.
        '''

        def __init__(self, session_path):
            self.session_path = os.path.expanduser(session_path)

        def new(self, name=None):
            '''
            Create a new session
            '''

            path = self.make_session_dir(self.session_path, session_name=name)
            session = Session(path)
            return session

        def load(self, name):
            '''
            Load a session given it's name.
            '''
            path = os.path.join(self.session_path, name)
            session = Session(path)
            return session

        def last(self):
            '''
            Return the last created session.
            '''
            dirs = [ name for name in os.listdir(self.session_path) if os.path.isdir(os.path.join(self.session_path, name))]
            dirs.sort()
            path = os.path.join(self.session_path, dirs[-1])
            session = Session(path)
            return session

        def make_session_dir(self, base_path, session_name=None):
            '''
            Make a new dir with a given name. If name doesn't exist sue the current date/time.
            '''

            base_path = os.path.expanduser(base_path)
            if session_name is None:
                session_dir_name = time.strftime('%Y_%m_%d__%I_%M_%S_%p')
            else:
                session_dir_name = session_name

            print('Creating a new session directory: %s' % session_dir_name)

            session_full_path = os.path.join(base_path, session_dir_name)
            if not os.path.exists(session_full_path):
                os.makedirs(session_full_path)
            return session_full_path

def pickle_sessions(session_folder, session_names, file_path):
    '''
    Combine, pickle and safe session data to a file.

    'session_folder' where the session folders reside
    'session_names' the names of the folders of the sessions to Combine
    'file_path' name of the pickled ile that will be saved
    '''

    sh = model.sessions.SessionHandler(session_folder)

    X = []
    Y = []

    for name in session_names:
        s = sh.load(name)
        x, y = s.load_dataset()
        X.append(x)
        Y.append(y)

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    with open(file_path, 'wb') as file:
        pkl = pickle.dump((X,Y), file)