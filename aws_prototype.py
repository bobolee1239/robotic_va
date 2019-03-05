#!/usr/bin/env python3

from beamforming.uca import UCA
import logging
import threading 
import time
import boto3
import numpy as np 
from scipy import signal

import sounddevice as sd


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    q = threading.Event()
    uca = UCA(fs=16000, nframes=2000, radius=0.032, num_mics=6, \
                quit_event=q, name='respeaker-7')

    enhanced = None

    lex_client = boto3.client('lex-runtime', region_name="us-west-2")
    isFailed = False
    while not q.is_set():
        try:
            if uca.wakeup('amber'):
                print('Wake up')
                time.sleep(1.0)
                chunks = uca.listen()
                enhanced = uca.beamforming(chunks)

                response = lex_client.post_content(
                    botName = "musicBot", 
                    botAlias = "songRequestor",
                    userId = "brian", 
                    sessionAttributes = {
                    
                    }, 
                    requestAttributes = {
                    }, 
                    contentType = "audio/lpcm; sample-rate=8000; sample-size-bits=16; channel-count=1; is-big-endian=false", 
                    accept = 'audio/pcm', 
                    inputStream = (signal.decimate(enhanced/2**15, 2)* 2**15).astype('<i2').tostring()
                )

                if response["dialogState"] == 'ReadyForFulfillment': break
                elif response["dialogState"] == "Failed": isFailed = True

                content = np.fromstring(response["audioStream"].read(), dtype="<i2")

                # Print Message
                sd.play(content, 16000)
                print('\n-------------------')
                print(response["message"])
                
                if isFailed: break
        except KeyboardInterrupt:
            print('Quit')
            q.set()
            break
    uca.close()

    if not isFailed:
        print("\n///// Request Information ///// ")
        for keys in response["slots"].keys():
            print("  * " + keys + ": " + response["slots"][keys])
        print("\n\nConversation END!")


