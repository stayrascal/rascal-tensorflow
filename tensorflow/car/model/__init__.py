import os
uname = os.uname()

if uname[1] != 'raspberrypi':
    from . import(utils, model, datasets, remotes, sensors, actuators,
                vehicles, pilots, templates)

else: 
    print('Detected running on rasberrypi. Only importing select modules.')
    from . import actuators, remotes, sensors, vehicles