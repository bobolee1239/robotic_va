#!/usr/bin/env python3
#
##  Copyright (c) 2019 Tsung-Han Brian Lee
##  ---------------------------------------------------------------------------
##  * AUTHOR     : Tsung-Han Brian Lee
##  * DESCRIPTION:
##       1. It's a robotic voice assistant project which integrated
##          sound source localization(SSL), sound source separation
##          (SSS), cloud-based chatbot(AWS-LEX) and virtual sound
##          field control.
##       2. It's under MIT License.
##  * REFERENCE  :
##       1. https://github.com/respeaker
##  ---------------------------------------------------------------------------
##  Permission is hereby granted, free of charge, to any person obtaining a
##  copy of this software and associated documentation files (the "Software"),
##  to deal in the Software without restriction, including without limitation
##  the rights to use, copy, modify, merge, publish, distribute, sublicense,
##  and/or sell copies of the Software, and to permit persons to whom the
##  Software is furnished to do so, subject to the following conditions:
##
##  The above copyright notice and this permission notice shall be included in
##  all copies or substantial portions of the Software.

##  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
##  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
##  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
##  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
##  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
##  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
##  IN THE SOFTWARE.
##  ---------------------------------------------------------------------------
#
DEBUG = True

from beamforming.uca import UCA, pixel_ring  # audio I/O & beamforming algo
import logging                               # logger to display information
import threading                             # thread
import time                                  # for sleep ...
import boto3                                 # to send AJAX request to AWS LEX
import numpy as np                           # science computation
from scipy import signal                     # resamping signal
import socket                                # socket i/o
import sys                                   # parse command line argument

if DEBUG:
    import sounddevice as sd                 # for debuggin sake

# logger to display information with different levels
logger = logging.getLogger('ROBOTIC_VA')
##
#   A dictionary to recorde the sound source location
#   and correpsonding frequency.
##
loc_history = {}

##
#   Callback Handler Fcn which will be fired after each SSL job is done!
#   ----------------------------------------------------------------------
#     GOALS:
#       1. light up the led pixel ring
#       2. control the vehicle to face to the user
#     ARGUEMNTS:
#       1. firer : object which fired event, that is, a uca class instance
#       2. direction [double] : horizontal angle of the user w.r.t vehicle
#       2. polar_angle [double] : verticle angle of the user w.r.t vehicle
#   -----------------------------------------------------------------------
##
def sslHandler(firer, direction, polar_angle):
    # user interaction : LED & console
    pixel_ring.set_direction(direction)

    # transform range from 0 ~ 360 -> -180 ~ 180
    if direction > 180:
         direction -= 360

    logger.info('[UCA Callback] src @ {:.2f}, @{:.2f}'.format(direction,
                 polar_angle))

    key = 5*(int(direction) // 5)
    loc_history[int(key)] = loc_history.get(int(key), 0) + 1

##
#   Playing audio back with specific sampling rate
#   --------------------------------------------------------------------------
#     ARGUMENTS:
#       1. in_data [numpy.ndarray] : digital signal points
#       2. fs      [doulbe]        : sampling rate, unit: Hz
#   --------------------------------------------------------------------------
##
def playAudio(in_data, fs, effect=None):
    if DEBUG:
        sd.play(in_data, fs)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 ./roboticVA <host_ip_addr> <host_port_num>')
        exit(-1)
    server_ip_addr  = sys.argv[1]
    server_port_num = int(sys.argv[2])

    # setup logger level
    logging.basicConfig(level=logging.INFO)

    ##
    #   Init Uniform Circular Array (UCA) for audio Input
    ##
    q = threading.Event()
    uca = UCA(fs=16000, nframes=2000, radius=0.032, num_mics=6, \
                quit_event=q, name='respeaker-7')

    uca.on('ssl_done', sslHandler)          # register callback handler
    enhanced = None                         # To store enhanced signal

    ##
    #   Init LEX-RUNTIME SERVICE to send AJAX request to AWS-LEX
    ##
    lex_client = boto3.client('lex-runtime', region_name="us-west-2")

    ##
    #   Socket io
    ##
    # init socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # server_addr
    server_addr = (server_ip_addr, server_port_num)

    logger.info('Binding to {0[0]}:{0[1]}'.format(server_addr))
    sock.bind(server_addr)

    # listening for incoming connection
    sock.listen(1)
    # waiting for connectino
    logger.info('Waiting for connection ...')
    connection, client_addr = sock.accept()
    logger.info('Connection from {}'.format(client_addr[0]))

    ##
    #       Robotic VA Routine
    ##
    isFailed = False
    while not q.is_set():
        try:
            #  wake up VA with a keyword
            if uca.wakeup('hello amber'):
                logger.info('Wake up')
                chunks = uca.listen()
                enhanced = uca.beamforming(chunks)

                logger.info('sending ajax request to AWS-LEX')
                # sending AJAX request to AWS-LEX
                response = lex_client.post_content(
                    botName = "RoboticVA",
                    botAlias = "roboticVA",
                    userId = "teaLab",
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

                if DEBUG:
                    logger.info('  * Playing back response ...')
                content = np.fromstring(response["audioStream"].read(), dtype="<i2")

                # Play enhanced speech back
                # if DEBUG:
                    # logger.info('Playing enhanced speech ...')
                    # playAudio(enhanced / 2**14, 16000)
                    # time.sleep(3.0)

                ##
                #   Playing response back to user
                ##
                playAudio(content / np.max(content), 16000)
                logger.info('\n-------------------')
                logger.info(' [RESPONSE]: ' + response["message"])

                if isFailed: break
        except KeyboardInterrupt:
            logger.info('Quit')
            q.set()
            break
        except Exception as e:
            logger.warn(e)
    uca.close()

    if not isFailed:
        content = np.fromstring(response["audioStream"].read(), dtype="<i2")
        playAudio(content / np.max(content), 16000)
        logger.info('\n-------------------')
        logger.info(response["message"])

        print("\n///// Request Information ///// ")
        for keys in response["slots"].keys():
            print("  * " + keys + ": " + response["slots"][keys])
        print("\n\n////////// Conversation END! //////////")

    ##
    #   Rotate the Robotic VA
    #   -------------------------------
    #   1. determine the angle
    #   2. publish command
    #   3. sleep awhile
    #   2. publish command
    ##
    logger.info('[ROBOTIC VA] loc_history len: {}'.format(len(loc_history)))
    logger.info('[ROBOTIC VA]\n loc_history: {}'.format(loc_history))

    goal = max(loc_history, key=lambda k: loc_history[k])
    logger.info('[ROBOTIC VA] goal: {}'.format(goal))

    rotation_time      = 3

    linear  = 0.0
    angular = float(goal) * 0.01745329251 / rotation_time
    command = np.array([linear, angular], dtype='float32')

    #  send to client
    connection.sendall(command.tostring())

    time.sleep(rotation_time + 1)

    linear  = 0.0
    angular = 0.0
    command = np.array([linear, angular], dtype='float32')

    # send to client again
    connection.sendall(command.tostring())

    sock.close()
    # empty the history
    loc_history = {}

    # hang the process
    while True:
        pass
