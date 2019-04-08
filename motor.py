#!/usr/bin/env python3

from beamforming.uca import UCA
import logging
import threading 
import time
import numpy as np 

import sounddevice as sd

def sslHandler(firer, direction):
    """
    callback function to handler ssl event
    """

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
        except KeyboardInterrupt:
            print('Quit')
            q.set()
            break

    uca.close()

