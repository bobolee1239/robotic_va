#!/usr/bin/env python3

from beamforming.uca import UCA
import logging
import threading 
import time
import boto3
import numpy as np 
from scipy import signal
import socket

import sounddevice as sd


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # setup socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # bind the socket to the port
    server_addr = ('140.114.57.81', 7777)
    print('Connecting to {0[0]}:{0[1]} ...'.format(server_addr))

    sock.connect(server_addr)
    
    # setup UCA
    q = threading.Event()
    uca = UCA(fs=16000, nframes=2000, radius=0.032, num_mics=6, \
                quit_event=q, name='respeaker-7')

    enhanced = None

    lex_client = boto3.client('lex-runtime', region_name="us-west-2")
    isFailed = False
    while not q.is_set():
        try:
            if uca.wakeup('hello amber'):
                print('Wake up')
                chunks = uca.listen(duration=1, timeout=1)
                enhanced = uca.beamforming(chunks)
                
                sig_len = enhanced.shape[0]
                if sig_len > 16000:
                    print('\t*sig too long')
                    enhanced = enhanced[0:16000]
                elif sig_len < 16000:
                    print('\t*sig too short')
                    enhanced = np.concatenate((enhanced, 
                                np.zeros(16000 - sig_len, dtype='int16')), axis=0)

                print('Streaming to server ...')
                sock.sendall(enhanced.tostring())

                print('Received: ', repr(sock.recv(2)))
                
        except KeyboardInterrupt:
            print('Quit')
            q.set()
            break
    uca.close()

