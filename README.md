# ROS ROBOTIC VA PROJECT

__AUTHOR__ : Tsung-Han Brian Lee <br />
__LICENSE__ : MIT

---

## Get Started

Environment
  * Tested Operating System : `Ubuntu 16` | `Raspian Strench`
  * Python Dependencies: `pyaudio`, `pyusb`, `webrtcvad`, and `pocketsphinx`

>  Make sure you have up-to-date version of pip, setuptools and wheel before install python dependencies

```bash
$ python3 -m pip install --upgrade pip setuptools wheel
```

Device Firmware (can be found in github/respeaker)
  * Respeaker 8 mics DFU (Device Firmware Upgrade)
  * Drivers for Raspberry Pi

To access USB device without root permission, you can add a udev `.rules` file to `/etc/udev/rules.d`
```bash
$ echo 'SUBSYSTEM=="usb", MODE="0666"' | sudo tee -a /etc/udev/rules.d/60-usb.rules
$ sudo udevadm control -R  # then re-plug the usb device
```

---

## Build Customized Keywords

Configure `dictionary.txt` and `keyword.txt` in `script/beamforming/assets/pocketsphinx-data` directory
<ul>
	<li>specific explanation
    <a href="https://github.com/respeaker/get_started_with_respeaker/issues/68">CLICK HERE</a>
  </li>
	<li>For more keywords dictionary
		<a href="https://raw.githubusercontent.com/respeaker/pocketsphinx-data/master/dictionary.txt">CLICK HERE</a>
  </li>
</ul>

---

## Plugin your own Source Localizatoin or Source Separation Algorithm

simply modify method `beamforming` of `class UCA` in `beamforming/uca.py`
```python
# FILE: scripts/beamforming/uca.py

class UCA(object):
  ...
  def beamforming(self, chunks):
    ...
    for chunk in chunks:
      ##
      #   IMPLEMENT YOUR ALGORITHM FOLLOWING, GOOD LUCK!
      ##
    ...
  ...
```

---

## REFERENCE

__Respeaker Microphone Array__
  1. https://github.com/respeaker
