#
## ******************************
##  AUTHOR: Brian Lee 
##  DATE  : 12th Feb, 2019
##
## ******************************
##  FILE        : uca.py 
##  DESCRIPTION : 
##                  
##
#

import collections
import threading 
import numpy as np 
import pyaudio

try:
    # python2 supporting
    import Queue
except:
    # python3
    import queue as Queue




class UCA(object):
    """
    UCA (Uniform Circular Array)
    
    Design Based on Respeaker 7 mics array architecture
    """
    SOUND_SPEED = 343.2
    def __init__(self, fs=16000, nframes=2000, radius=0.032, num_mics=6, quit_event=None):
        self.radius     = radius 
        self.fs         = fs
        self.nframes    = nframes 

        self.pyaudio_instance = pyaudio.Pyaudio()
         
