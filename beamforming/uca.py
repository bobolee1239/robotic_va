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
import os

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
    def __init__(self, fs=16000, nframes=2000, radius=0.032, num_mics=6, quit_event=None, name='respeaker-7'):
        self.radius     = radius 
        self.fs         = fs
        self.nframes    = nframes 

        self.pyaudio_instance = pyaudio.Pyaudio()
        
        self.device_idx = None
        for i in range(self.pyaudio_instance.get_device_count()):
            dev  = self.pyaudio_instance.get_device_info_by_index(i)
            name = dev['name'].encode('utf-8')
            print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
            if name.lower().find(b'respeaker') >= 0 and dev['maxInputChannels'] >= num_mics:
                print('Use {}'.format(name))
                self.device_idx = i
                break

        if not self.device_idx:
            raise ValueError('Wrong #channels of mic array!')

        self.stream = self.pyaudio_instance.open(
            input       = True,
            start       = False,
            format      = pyaudio.paInt16,
            channels    = (num_mics + 2 if name == 'respeaker-7' else num_mic),
            rate        = self.fs,
            frames_per_buffers  = int(self.nframes),
            stream_callback     = self._callback, 
            input_device_index  = self.device_idx
        )

        self.quit_event = quit_event if quit_event else threading.Event()

        self.listen_queue = Queue.Queue()
        self.detect_queue = Queue.Queue()

        self.active = False

        self.listen_history = collections.deque(maxlen=16)
        self.detect_history = collections.deque(maxlen=64)

        self.listen_countdown = [0, 0]

        self.decoder = UCA.create_decoder()
        self.decoder.start_utt()

    def _callback(self, in_data, frame_count, time_info, status):
        """
        pyaudio callback function
        """
    @staticmethod
    def create_decoder():
        from pocketsphinx.pocketsphinx import Decoder

        path              = os.path.dirname(os.path.realpath(__file__))
        pocketsphinx_dir = os.getenv('POCKETSPHINX_DATA', 
                                os.path.join(path, 'assets', 'pocketsphinx-data')) 
        hmm = os.getenv('POCKETSPHINX_HMM', os.path.join(pocketsphinx_dir, 'hmm'))
        dic = os.getenv('POCKETSPHINX_DIC', os.path.join(pocketsphinx_dir, 'dictionary.txt'))
        kws = os.getenv('POCKETSPHINX_KWS', os.path.join(pocketsphinx_dir, 'keywords.txt')) 

        config = Decoder.default_config()
        config.set_string('-hmm', hmm)
        config.set_string('-dict', dic)
        config.set_string('-kws', kws)
        # config.set_int('-samprate', fs) # uncomment for fs != 16k. use config.set_float() on ubuntu
        config.set_int('-nfft', 512)
        config.set_float('-vad_threshold', 2.7)
        config.set_string('-logfn', os.devnull)
, os.devnull
