"""
Script to start controlling your car remotely via on Raspberry Pi that 
constantly requests directions from a remote server. See serve_no_pilot.py
to start a server on your laptop. 

Usage:
    drive.py [--remote=<name>] 


Options:
  --remote=<name>   recording session name
"""

import os
from docopt import docopt
import model


# Get args
args = docopt(__doc__)

if __name__ == '__main__':
    remote_url = args['--remote']

    # Set up your PWM values for your steering and throttle actuator here.
    mythrottle = model.actuators.PWMThrottleActuator(channel=0,
                                                  min_pulse=280,
                                                  max_pulse=490,
                                                  zero_pulse=350)

    mysteering = model.actuators.PWMSteeringActuator(channel=1,
                                                  left_pulse=300,
                                                  right_pulse=400)

    #asych img capture from picamera
    mycamera = model.sensors.PiVideoStream()
    
    #Get all autopilot signals from remote host
    mypilot = model.remotes.RemoteClient(remote_url, vehicle_id='mycar')

    #Create your car your car
    car = model.vehicles.BaseVehicle(camera=mycamera,
                                  steering_actuator=mysteering,
                                  throttle_actuator=mythrottle,
                                  pilot=mypilot)

    
    #Start the drive loop
    car.start()