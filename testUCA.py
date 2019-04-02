import numpy as np
from scipy import signal

import sounddevice as sd


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # setup UCA
    q = threading.Event()
    uca = UCA(fs=16000, nframes=2000, radius=0.032, num_mics=6, \
                quit_event=q, name='respeaker-7')

    enhanced = None

    isFailed = False
    while not q.is_set():
        try:
            if uca.wakeup('hello amber'):
                print('Wake up')
                chunks = uca.listen(duration=1, timeout=1)
                enhanced = uca.beamforming(chunks)
        except KeyboardInterrupt:
            print('Quit')
            q.set()
            break
    uca.close()
                                                                                                
