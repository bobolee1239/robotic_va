#!/usr/bin/env python3

from beamforming.uca import UCA, pixel_ring
import logging
import threading
import time
import numpy as np

import sounddevice as sd

import RPi.GPIO as GPIO

pwm1  = 18
pwm2  = 13
# setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(pwm1, GPIO.OUT)
GPIO.setup(pwm2, GPIO.OUT)

p1 = GPIO.PWM(pwm1, 1000)  # BCM 18, Freq 1000Hz
p2 = GPIO.PWM(pwm2, 1000)  # BCM 18, Freq 1000Hz

p1.start(0.5)
p2.start(0.5)

def sslHandler(firer, direction, polar_angle):
    """
    callback function to handler ssl event
    """
    pixel_ring.set_direction(direction)
    print('In callback: src @ {:.2f}, @{:.2f}, delays = {}'.format(direction,
            polar_angle, np.array(delays)*self.fs))
    # range of direction: -180 ~ 180
    ## if direction > 180:
    ##     direction -= 360

    ## assert(direction < 180 and direction > 180, "direction range wrong")

    ## out1 = direction / 180 * 0.5 + 0.5
    ## out2 = -direction / 180 * 0.5 + 0.5

    ## p1.ChangeDutyCycle(out1 * 100)
    ## p2.ChangeDutyCycle(out2 * 100)

    if direction > 180:
        out1 = (360 - direction) / 180
        out2 = 0
    else:
        out2 = direction / 180
        out1 = 0

    p1.ChangeDutyCycle(out1 * 50)
    p2.ChangeDutyCycle(out2 * 50)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # setup UCA
    q = threading.Event()
    uca = UCA(fs=16000, nframes=2000, radius=0.032, num_mics=6, \
                quit_event=q, name='respeaker-7')

    uca.on('ssl_done', sslHandler)
    enhanced = None

    while not q.is_set():
        try:
            if uca.wakeup('hello amber'):
                print('Wake up')
                chunks = uca.listen(duration=9, timeout=3)
                enhanced = uca.beamforming(chunks)
            # stop motor after beamforming
            p1.ChangeDutyCycle(0)
            p2.ChangeDutyCycle(0)
        except KeyboardInterrupt:
            print('Quit')
            q.set()
            break
    uca.close()

p1.stop()
p2.stop()
GPIO.cleanup()
