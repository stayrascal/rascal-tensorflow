"""
Script to start server to drive your car.
"""

import model

import keras

# Load a trained keras model and use it in the KerasAngle pilot
# model_file = ''

model_file = '/home/'
model = keras.models.load_model(model_file)
pilot = model.pilots.KerasAngle(model=model, throttle=20)

# Specify where sessions data should be saved
sh = model.sessions.SessionHandler(sessions_path='~/data/sessions')
session = sh.new()

# Start server
w = model.remotes.RemoteServer(session, pilot, port=8886)
w.start()

# in a browser go to localhost:8887 to drive your car