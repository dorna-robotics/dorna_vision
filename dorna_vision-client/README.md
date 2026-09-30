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

# Save the last run's image on THIS computer — full resolution, JPEG quality 100
vc.detection("aruco1").save_img("captures/snap.jpg", type="img")

vc.camera_remove(serial_number)
vc.close()
```

## Images

After a run, a detection's images come back over the API, so they can be
kept on the computer calling it rather than on the vision unit.

| type | image |
|---|---|
| `img` | the annotated frame |
| `img_roi` | the unannotated ROI crop — the one to keep for a training dataset |
| `img_thr` | the threshold mask (contour / polygon detections) |
| `color_img`, `depth_img`, `ir_img` | the raw camera frames |

```python
det = vc.detection("aruco1")
det.run()

# Written here, full resolution, JPEG quality 100 by default
det.save_img("captures/a.jpg", type="img_roi")
det.save_img("captures/a.jpg", type="img_roi", quality=90)

# A folder (trailing "/" or an existing directory) gets a timestamped name,
# roi_<timestamp>.jpg for img_roi — call it in a loop to build a dataset
det.save_img("captures/", type="img_roi")

# Or the bytes themselves
jpeg, meta = det.get_img(type="img", quality=85)
```

`save_img` creates folders as needed and returns the path it wrote. The bytes
are JPEG whatever the file extension. This is the calling computer's disk; the
detection's own `display={"save_img_roi": ...}` option writes on the vision
unit instead.
