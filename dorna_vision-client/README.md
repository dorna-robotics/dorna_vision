# dorna_vision_client

Python client for the [dorna_vision](https://github.com/dorna-robotics/dorna_vision) server.

## Install

```bash
git clone https://github.com/dorna-robotics/dorna_vision.git
cd dorna_vision/dorna_vision-client
pip install -e .
```

To update later, run `git pull` and the installed package picks up the changes.

## Usage

```python
from dorna_vision_client import VisionClient

vc = VisionClient()
vc.connect()   # defaults: host="127.0.0.1", port=8765

devs = vc.camera_list()
serial_number = devs[0]["serial_number"]

vc.camera_add(serial_number=serial_number, mode="bgrd", stream={"width": 848, "height": 480, "fps": 15})
vc.detection_add(
    name="aruco1",
    camera_serial_number=serial_number,
    detection={"cmd": "aruco", "marker_length": 20, "dictionary": "DICT_4X4_50"},
)

valid = vc.detection_run(name="aruco1")
print(valid)

jpeg, meta = vc.detection_get_image(name="aruco1", which="img")
with open("snap.jpg", "wb") as f:
    f.write(jpeg)

vc.camera_remove(serial_number)
vc.close()
```
