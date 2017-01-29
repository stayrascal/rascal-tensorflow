"""
Script to start server to drive your car.
"""

import model

# Setup how server will save files and which pilot to use
pilot = model.pilots.BasePilot()

session_path = '~/rascal_data/sessions'
sh = model.sessions.SessionHandler(session_path=session_path)
session = sh.new()

# start server
w = model.remotes.RemoteServer(session, pilot, port=8886)
w.start()

# in a browser go to localhost:8887 to drive your car