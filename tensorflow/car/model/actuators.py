import time
import sys


def map_range(x, X_min, X_max, Y_min, Y_max):
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    XY_ratio = X_range / Y_range
    y = ((x - X_min) / XY_ratio + Y_min) // 1
    return int(y)


class BaseSteeringActuator():
    ''' Placeholder until real logic is implemented '''

    def update(self, angle):
        print('BaseSteeringActuator.update: angle=%s' % angle)


class BaseThrottleActuator():
    ''' Pleaceholder until real logic is implemented '''

    def update(self, throttle):
        print('BaseThrottleActuator.update: throttle=%s' % throttle)


class Adafruit_PCA9685_Actuator():

    def __init__(self, channel, frequency=60):
        import Adafruit_PCA9685
        # Initialise the PCA9685 using the default address (0x40).
        self.pwm = Adafruit_PCA9685.PCA9685()

        # Set frequency to 60hz, good for servos.
        self.pwm.set_pwm_freq(frequency)
        self.channel = channel


class PWMteeringActuator(Adafruit_PCA9685_Actuator):

    #max angle  whees can turn
    LEFT_ANGLE = -45
    RIGHT_ANGLE= 45

    def __init__(self, channel=1, frequency=60, left_pulse=290, right_pulse=490):
        super().__init__(channel, frequency)
        self.left_pulse = left_pulse
        self.right_pulse = right_pulse

    def update(self, angle):
        # map absolute angle to angle that vehicle can implement.
        pulse = map_range(angle, self.LEFT_ANGLE, self.RIGHT_ANGLE,
                                self.left_pulse, self.right_pulse)

        self.pwm.set_pwm(self.channel, 0, pulse)

class PWMThrottleActuator(Adafruit_PCA9685_Actuator):

    MIN_THROTITLE = -100
    MAX_THROTITLE = 100

    def __init__(self, channel=0, frequency=60, max_pulse=300, min_pulse=490, zero_pulse=350):
        super().__init__(channel, frequency)
        self.max_pulse = max_pulse
        self.min_pulse = min_pulse
        self.zero_pulse = zero_pulse
        self.calibrate()

    def calibrate(self):
        # Calibrate ESC
        print('center: %s' % self.zero_pulse)
        # Set Max Throttle
        self.pwm.set_pwm(self.channel, 0, self.zero_pulse)
        time.sleep(1)

    def update(self, throttle):
        print('throttle update: %s' % throttle)
        if throttle > 0:
            pulse = map_range(throttle, 0, self.MAX_THROTITLE, 
            self.zero_pulse, self.max_pulse)
        else:
            pulse = map_range(throttle, self.MIN_THROTITLE, 0,
            self.MIN_THROTITLE, self.zero_pulse)
        print('pulse: %s' % pulse)
        sys.stdout.flush()
        self.pwm.set_pwm(self.channel, 0, pulse)
        return '123'