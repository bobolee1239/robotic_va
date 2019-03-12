#!/usr/bin/env python3 

import os
import numpy as np
from librosa import stft
from librosa.feature import melspectrogram
from scipy.io.wavfile import read
from models.cnn import *
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error
import socket

# mfsc parameters
n_fft = 1024
hop_length = 256
power = 2
n_mels = 64
n_mfcc = 20
width = 3
sampling_rate = 16000

# model parameters
num_of_class = 10
input_size = [64, 63, 1] # [time, frequency, channel]
class_list = ['down', 'up', 'stop', 'off', 'left', 'yes', 'no', 'right', 'on', 'go']

# model
M = vgg(
    size=input_size,
    n_lbl=num_of_class,
    lr=1e-3
)
M.restore('./params/vgg_speech_command/')

def mfsc(x, sr):
    # zero mean
    x_mean = np.mean(x)
    x_var  = np.var(x - x_mean)
    x  = (x - x_mean) / np.sqrt(x_var + 1e-6)
    
    # spectrum
    X = np.abs(stft(y=x, n_fft=n_fft, hop_length=hop_length)) ** power
    S = np.log10(1 + 10 * (melspectrogram(S=X, sr=sr, n_mels=n_mels)))
    
    return np.reshape(S, [1, 64, 63, 1])

if __name__ == '__main__':
    
    # init socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # server_addr
    server_addr = ('140.114.57.81', 7777)

    print('Binding to {0[0]}:{0[1]}'.format(server_addr))
    sock.bind(server_addr)

    # listening for incoming connection
    sock.listen(1)
    # waiting for connectino
    print('Waiting for connection ...')
    connection, client_addr = sock.accept()

    try:
        print('Connection from ', client_addr)
        while True:
            # keep receiving data
            receives = b''
            while True:
                data = connection.recv(1024)
                if data:
                    receives += data
                else:
                    print('no more data!')
                    break
            
            # load audio
            # sr, x = read('./down_example.wav')        # must be one sec
            x = np.fromstring(receives, dtype='<i2')
            assert(x.shape[0] == 160000)
            x /= 32768.0                                # normalization
            feature = mfsc(x, sampling_rate)            # feature extraction

            # inference
            predict = M.predict(x=feature)[0] # output is one hot vector
            predict = class_list[np.argmax(predict)]
            print(predict)
                
    finally:
        # clean up connection
        connection.close()
