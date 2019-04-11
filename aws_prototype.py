#!/usr/bin/env python3

from beamforming.uca import UCA, pixel_ring
import logging
import threading
import time
import boto3
import numpy as np
from scipy import signal

import sounddevice as sd

def sslHandler(firer, direction, polar_angle):
    pixel_ring.set_direction(direction)
    print('In callback: src @ {:.2f}, @{:.2f}, delays = {}'.format(direction,
            polar_angle))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    q = threading.Event()
    uca = UCA(fs=16000, nframes=2000, radius=0.032, num_mics=6, \
                quit_event=q, name='respeaker-7')
    uca.on('ssl_done', sslHandler)

    enhanced = None

    lex_client = boto3.client('lex-runtime', region_name="us-west-2")
    isFailed = False
    while not q.is_set():
        try:
            if uca.wakeup('hello amber'):
                print('Wake up')
                chunks = uca.listen()
                enhanced = uca.beamforming(chunks)

                response = lex_client.post_content(
                    botName = "musicBot",
                    botAlias = "songRequestor",
                    userId = "bobolee",
                    sessionAttributes = {

                    },
                    requestAttributes = {
                    },
                    contentType = "audio/lpcm; sample-rate=8000; sample-size-bits=16; channel-count=1; is-big-endian=false",
                    accept = 'audio/pcm',
                    inputStream = (signal.decimate(enhanced/2**15, 2)* 2**15).astype('<i2').tostring()
                )

                if response["dialogState"] == 'ReadyForFulfillment': break
                if response["dialogState"] == 'Fulfilled': break
                elif response["dialogState"] == "Failed": isFailed = True

                content = np.fromstring(response["audioStream"].read(), dtype="<i2")

                # Print request
                sd.play(enhanced / 2**14, 16000)
                time.sleep(3.0)

                # Print Message
                sd.play(content / np.max(content), 16000)
                print('\n-------------------')
                print(response["message"])

                if isFailed: break
        except KeyboardInterrupt:
            print('Quit')
            q.set()
            break
    uca.close()

    if not isFailed:
        content = np.fromstring(response["audioStream"].read(), dtype="<i2")
        sd.play(content / np.max(content), 16000)
        print('\n-------------------')
        print(response["message"])

        print("\n///// Request Information ///// ")
        for keys in response["slots"].keys():
            print("  * " + keys + ": " + response["slots"][keys])
        print("\n\nConversation END!")
