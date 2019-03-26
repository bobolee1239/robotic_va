
#!/usr/bin/env python3
#
## Copyright 2019 Tsung-Han Brian Lee, Shincheng Huang
## *****************************************************
##  AUTHOR: Tsung-Han Brian Lee,
##          Shincheng Huang
##  DATE  : 12th Feb, 2019
##
## *****************************************************
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
import logging
from gcc_phat import gcc_phat
import math
from pixel_ring import pixel_ring
import sounddevice as sd

try:
    # python2 supporting
    import Queue
except:
    # python3
    import queue as Queue

from respeaker.vad import vad

logger              = logging.getLogger('uca')
collecting_audio    = os.getenv('COLLECTING_AUDIO', 'no')


class UCA(object):
    listening_mask   = (1<<0)
    detecting_mask   = (1<<1)
    validation = []
    SOUND_SPEED = 343.2
    """
    UCA (Uniform Circular Array)

    Design Based on Respeaker 7 mics array architecture
    """
    def __init__(self, fs=16000, nframes=2000, radius=0.032, num_mics=6, quit_event=None, name='respeaker-7'):
        self.radius     = radius
        self.fs         = fs
        self.nframes    = nframes
        self.nchannels  = (num_mics + 2 if name == 'respeaker-7' else num_mic)
        self.num_mics   = num_mics
        self.max_delay  = radius * 2 / UCA.SOUND_SPEED
        self.delays     = None

        self.pyaudio_instance = pyaudio.PyAudio()

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
            channels    = self.nchannels,
            rate        = self.fs,
            frames_per_buffer   = int(self.nframes),
            stream_callback     = self._callback,
            input_device_index  = self.device_idx
        )

        self.quit_event = quit_event if quit_event else threading.Event()

        # multi-channels input
        self.listen_queue = Queue.Queue()
        # mono-channel input
        self.detect_queue = Queue.Queue()

        self.active = False
        self.status = 0

        self.listen_history = collections.deque(maxlen=16)
        self.detect_history = collections.deque(maxlen=64)

        # index 0 for listening duration count,
        # index 1 for timeout duration count.
        self.listen_countdown = [0, 0]

        self.decoder = UCA.create_decoder()
        self.decoder.start_utt()

        # build tdoa matrix
        mic_theta           = np.arange(0, 2*np.pi, 2*np.pi/6)
        self.tdoa_matrix    = np.array([np.cos(mic_theta[1:]), np.sin(mic_theta[1:])]).T

        self.tdoa_matrix -= np.ones((len(mic_theta)-1, 2)) \
                             * np.array([np.cos(mic_theta[0]), np.sin(mic_theta[0])])
#################### wired #######################################
        self.tdoa_measures  = np.ones(((len(mic_theta)-1, ))) \
                                * UCA.SOUND_SPEED / (self.radius)       # diameter => radius
#################### wired #######################################
        df = self.fs / self.nframes / 2                # frequency domain resolution
        frq = np.arange(0, (self.fs/2) , df)        # freqeuncy axis
        k = 2*np.pi * frq / UCA.SOUND_SPEED         # wave number
        # create Rxx placeholder
        self.Rxx = np.zeros((6,6,len(frq))
        # UCA geometry
        mic_theta = np.arange(np.pi/6, 2*np.pi, np.pi/3)
        GD_matrix = np.ones((3,6))*np.array([np.cos(mic_theta), np.sin(mic_theta), np.zeros(6)] ) # 3 x 6
	    # 6 microphones
        for i in range(1, 7):
            for j in range(1,7):
                # modify sum(..., 2) -> sum(...)
                s_Rxx = np.abs(sum(GD_matrix[:,i-1]-GD_matrix[:,j-1]))/np.pi * k
                self.Rxx[i-1][j-1][:] = np.sinc(s_Rxx)

    def wakeup(self, keyword=None):
        self.decoder.end_utt()
        self.decoder.start_utt()

        # flag up detecting
        self.status |= UCA.detecting_mask

        # clear detecting queue
        self.detect_history.clear()
        self.detect_queue.queue.clear()

        self.stream.start_stream()
        result = None
        logger.info('Start detecting ...')

        while not self.quit_event.is_set():
            if self.detect_queue.qsize() > 4:
                logger.info('Too many delays, {0} in queue'.format(self.detect_queue.qsize()))
            elif self.detect_queue.empty():
                continue

            data = self.detect_queue.get()

            self.detect_history.append(data)

            self.decoder.process_raw(data, False, False)

            hypothesis = self.decoder.hyp()
            if hypothesis:


                if collecting_audio != 'no':
                    logger.debug(collecting_audio)
                    # save detect_history as wave ?
                # clear history
                self.detect_history.clear()
                if keyword:
                    if hypothesis.hypstr.find(keyword) >= 0:
                        result = hypothesis.hypstr
                        break
                    else:
                        self.decoder.end_utt()
                        self.decoder.start_utt()
                        self.detect_history.clear()
                else:
                    result = hypothesis.hypstr
                    break

        # flag down detecting
        self.status &= ~UCA.detecting_mask
        self.stop()
        return result

        def DOA(self, buf):
            best_guess = None
        MIC_GROUP_N = self.num_mics - 1
        MIC_GROUP = [[1+i, 1] for i in range(1, MIC_GROUP_N+1)]

        tau = [0] * MIC_GROUP_N

        # estimate each group of delay
        for i, v in enumerate(MIC_GROUP):
            tau[i] = gcc_phat(buf[v[0]::8], buf[v[1]::8], fs=self.fs, max_tau=self.max_delay, interp=10)

        # save delays for separation
        # self.delays = [0] + tau

        # least square solution of (cos, sin)
        sol = np.linalg.pinv(self.tdoa_matrix).dot( \
              (self.tdoa_measures * np.array(tau)).reshape(MIC_GROUP_N, 1))

############################################
        phi_in_rad = min( sol[1] / math.sin(math.atan2(sol[1],sol[0]) ), 1 )
        phi = 90 - np.rad2deg( math.asin(phi_in_rad) ) # phi in degree
        # found out theta
        # another 180.0 for positive value, 30.0 for respeaker architecture
        return ([(math.atan2(sol[1], sol[0])/np.pi*180.0 + 210.0) % 360
                , phi], [0] + tau)
#############################################

    def listen(self, duration=9, timeout=3):
        vad.reset()

        # setting countdown value
        self.listen_countdown[0] = (duration*self.fs + self.nframes -1) / self.nframes
        self.listen_countdown[1] = (timeout*self.fs + self.nframes - 1) / self.nframes

        self.listen_queue.queue.clear()
        self.status |= UCA.listening_mask
        self.start()

        logger.info('Start Listening')

        def _listen():
            """
            Generator for input signals
            """
            try:
                data = self.listen_queue.get(timeout=timeout)
                while data and not self.quit_event.is_set():
                    yield data
                    data = self.listen_queue.get(timeout=timeout)
            except Queue.Empty:
                pass
            self.stop()

        return _listen()

    def start(self):
        if self.stream.is_stopped():
            self.stream.start_stream()

    def stop(self):
        if not self.status and self.stream.is_active():
            self.stream.stop_stream()

    def close(self):
        self.quit()
        self.stream.close()
        pixel_ring.off()

    def quit(self):
        self.status = 0     # flag down everything
        self.quit_event.set()
        self.listen_queue.put('') # put logitical false into queue

    def beamforming(self, chunks):
        delays = [0.0] * (self.num_mics-1)

        enhanced_speech = []
        nCount = 0
        for chunk in chunks:
            nCount += 1
            if nCount < 2: continue
            # decode from binary stream
            raw_sigs = np.fromstring(chunk, dtype='int16')

            # casting int16 to double floating number
            raw_sigs = raw_sigs / (2**15)

            # tdoa & doa estimation based on planar wavefront
            #direction, delays = self.DOA(raw_sigs)
            ## TODO... expand gcc_phat and DOA function in following
            #DOA
            best_guess = None
            MIC_GROUP_N = self.num_mics - 1
            MIC_GROUP = [[1+i, 1] for i in range(1, MIC_GROUP_N+1)]
            tau = [0] * MIC_GROUP_N

            # rfft
            tf_sigs = np.zeros(( 6,len(raw_sigs[1::8]) ))
            for i in range(1,7): # 1~6
                tf_sigs[i-1] = raw_sigs[i::8]
            len_of_sig= len(tf_sigs[0])
            tf_sigs = np.fft.rfft(tf_sigs, 2*len_of_sig)

            # estimate each group of delay
            # gcc_phat
            interp = 10
            max_tau = self.max_delay
            n = None
            for i, v in enumerate(MIC_GROUP):
            #    print(v[1])
                SIG = tf_sigs[v[0]-1]
                RESIG = tf_sigs[0]
                n = len_of_sig*2
                R = SIG * np.conj(RESIG)
                cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
                max_shift = int(interp * n / 2)
                if max_tau:
                    max_shift = np.minimum(int(interp * self.fs * max_tau), max_shift)
                cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
                # find max cross correlation index
                shift = np.argmax(np.abs(cc)) - max_shift
                tau[i] = shift / float(interp * self.fs)
                #tau[i] = gcc_phat(raw_sigs[v[0]::8], buf[v[1]::8], fs=self.fs, max_tau=self.max_delay, interp=10)

                # least square solution of (cos, sin)
            sol = np.linalg.pinv(self.tdoa_matrix).dot( \
              (self.tdoa_measures * np.array(tau)).reshape(MIC_GROUP_N, 1))
            phi_in_rad = min( sol[1] / math.sin(math.atan2(sol[1],sol[0]) ), 1 )
            phi = 90 - np.rad2deg( math.asin(phi_in_rad) ) # phi in degree
            direction = [(math.atan2(sol[1], sol[0])/np.pi*180.0 + 210.0) % 360, phi]

        # setting led && logger info
            pixel_ring.set_direction(direction[0])
            logger.debug('@ {:.2f} @ {:.2f}, delays = {}'.format(direction[0], direction[1], np.array(tau)*self.fs))
            # direction[0] is horizintal angle ; direction[1] is elevation angle

            ## TODO...
            ## *************  apply MVDR beamformer  ****************
            # frq
            kappa = np.array([np.cos(np.deg2rad(direction[0]))*np.sin(np.deg2rad(90-direction[1])), \
                              np.sin(np.deg2rad(direction[0]))*np.sin(np.deg2rad(90-direction[1])),  \
                              np.cos(np.deg2rad(90-direction[1]))] )
            df = self.fs / n                # frequency domain resolution
            frq = np.arange(0, (self.fs/2) , df)        # freqeuncy axis
            k = 2*np.pi * frq / UCA.SOUND_SPEED         # wave number

            # UCA geometry
            mic_theta = np.arange(np.pi/6, 2*np.pi, np.pi/3)
            GD_matrix = np.ones((3,6))*np.array([np.cos(mic_theta), np.sin(mic_theta), np.zeros(6)] ) # 3 x 6
            # scan frq
            source_half = np.zeros((len(frq),))
            for i in range(1,len(frq)+1):
                # apply frequency mask
                if i*df < 500: continue
                elif (i * df > 5000):
                    source_half[i-1] = tf_sigs[0, i-1] * 0.2

                A = np.exp(1j*np.dot(kappa.T,GD_matrix)*k[i-1]) # 1x6
                w = np.dot(np.linalg.inv(self.Rxx[:,:,i-1] + 0.01*np.eye(6)), A)
                W = w/(np.dot(np.conj(A.T), w))
                #W = np.dot(np.linalg.inv(np.linalg.inv(np.dot(np.conj(A.T), w))), w)
                source_half[i-1] = np.dot(W.T.conj() , tf_sigs[:,i-1])

            enhanced_speech.append(np.fft.irfft(source_half, n=n))

            ## # *************  apply DAS beamformer  ****************
            ## int_delays = (np.array(delays)*self.fs).astype('int16')
            ## int_delays -= int(np.min(int_delays))
            ## max_delays = np.max(int_delays);

            ## toAdd = np.zeros((raw_sigs.size//8 + max_delays,
            ##                   self.num_mics), dtype='int16')
            ## # manupilate integer delays
            ## for i in range(self.num_mics):
            ##     # 1. Padding zero in the front and back
            ##     # 2. shift 2 bits (devide by 4)
            ##     toAdd[:, i] = np.concatenate((np.zeros(int_delays[i], dtype='int16'),
            ##                                    raw_sigs[i+1::8] >> 2,
            ##                                    np.zeros(max_delays-int_delays[i], dtype='int16')),
            ##                                   axis=0)
            ##
            ## # add them together
            ## enhanced_speech.append(np.sum(toAdd, axis=1, dtype='int16'))
            ## # *************************************************

        return np.concatenate(enhanced_speech, axis=0)

    def _callback(self, in_data, frame_count, time_info, status):
        """
        Pyaudio callback function
        """
        # decode bytes stream
        mulChans = np.fromstring(in_data, dtype='int16')
        mono     = mulChans[7::8].tostring()

        if self.status & UCA.detecting_mask:
            # signed 16 bits little endian
            self.detect_queue.put(mono)

        if self.status & UCA.listening_mask:

            active = vad.is_speech(mono)

            if active:
                if not self.active:
                    for d in self.listen_history:
                        self.listen_queue.put(d)
                        self.listen_countdown[0] -= 1 # count down timeout
                    self.listen_history.clear()

                self.listen_queue.put(in_data)
                self.listen_countdown[1] -= 1         # count down listening time
            else:
                if self.active:
                    self.listen_queue.put(in_data)
                else:
                    self.listen_history.append(in_data)
                self.listen_countdown[1] -= 1         # coutn down listening time

            if self.listen_countdown[0] <= 0 or self.listen_countdown[1] <= 0:
                self.listen_queue.put('')
                self.status &= ~self.listening_mask
                logger.info('Stop listening')

            self.active = active

        return None, pyaudio.paContinue




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

        return Decoder(config)




def task(quit_event):
    import time

    uca = UCA(fs=16000, nframes=2000, radius=0.032, num_mics=6, \
                quit_event=quit_event, name='respeaker-7')

    while not quit_event.is_set():
        if uca.wakeup('bagel'):
            print('Wake up')
            chunks = uca.listen()
            enhanced = uca.beamforming(chunks)

            sd.play(enhanced, 16000)
            time.sleep(10.0)

    uca.close()

def main():
    import time

    logging.basicConfig(level=logging.DEBUG)

    q = threading.Event()
    t = threading.Thread(target=task, args=(q, ))
    t.start()
    while True:
        try:
            time.sleep(1.0)
        except KeyboardInterrupt:
            print('Quit')
            q.set()
            break
    # wait for the thread
    t.join()

if __name__ == '__main__':
    main()
