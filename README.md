## Installation
Notice that the program has been tested only on Python 3.7+.

### 1- Install the camera module
Follow the instruction given (here)[https://github.com/dorna-robotics/camera?tab=readme-ov-file#installation] to fully install the camera module first.
### 2- Build the library
Use `git clone` to download the `dorna_vision` repository, or simply download the [zip file](https://github.com/dorna-robotics/dorna_vision/archive/master.zip), and unzip the file.  
```bash
git clone https://github.com/dorna-robotics/dorna_vision.git
```
Next, go to the downloaded directory, where the `requirements.txt` file is located, and run:
```bash
# install requirements
pip install -r requirements.txt
```
Finally
```bash
pip install . --upgrade --force-reinstall
```

## Pattern detection app
Pattern detection app is a graphical development environment inside Jupyter Notebook, that helps you build the pattern detection system and tune the parameters. Once the application is developed, you can directly call the detection modules inside your code without running the detection app. 
Open a Jupyter Notebook and create a new `Python 3` file. Next, add the following three lines of code to your Jupyter Notebook to start the  pattern detection app:
```python
%matplotlib widget
from dorna_vision import App
App().detect_pattern()
```
There are multiple tabs available to design and tune your detection parameters
> 🚨 **Notice:** Make sure that the camera is connected to your computer when running the app.  

### Source
Select the source of your image. You can connect an `Intel RealSense` camera and capture the image or directly load and an image.  
Use `Save image as` section to save the captured image.  
Click on `Terminate the app` once your testing is over and you no longer need to run the application.

### Visual adjustment
- Region of interest: Select specific region of the image where you want to run the detection. Use mouse to create a polygon on the image as the region of interest.
- Intensity: Change the image contrast and brightness.
- Color mask: Remove certain colors. This is useful specially in color based detection applications.

### Pattern detection
This tab consists of several pattern detection algorithms that you can apply to your image:
- Ellipse detection: Multiple parameters are available to fine tune your ellipse detection method. Use `Axes range` and `Axes ratio` to filter the detected ellipses based on their major and minor axes. For example, if you are detecting circle, then `Axes ratio` parameter is close to `1`.
- Polygon detection: Multiple parameters are available to fine tune your polygon detection method. Use `Side`, `Area` and `Perimeter` to further filter the detected polygons. For example when detecting triangles the `Side` parameter should be `3`.
- Contour detection: Contour detection is a technique used in image processing to identify the boundaries or outlines of objects within an image. Multiple parameters are available to fine tune your contour detection method.
- Aruco detection: Multiple parameters are available to fine tune your Aruco detection method.

### 6D pose
The 6D pose of an object in a three-dimensional space refers to its position and orientation with respect to the camera coordinate (frame) system. It is defined by six parameters:
- Position: The position of the object is still defined by the 3D coordinates `[x, y, z]` of its center in space.
- Orientation (modified axis-angle representation): The orientation is defined by a vector `r = [a, b, c]`, which means that you have to rotate an angle `theta = norm(r)` (radian) around the axis described by unit vector `r/theta`.  
To run this method, specify 3 points with respects to the detected object oriented bounding box. The camera use these 3 points to find a plane passing these 3 points in 3D space, and return the center of the plane, and its orientation with respect to the camera.

> 🚨 **Notice:** Make sure that all the 3 points selected on the object have valid `[x, y, z]`, otherwise the 6D pose is not accurate.

> 🚨 **Notice:** This method only works when the image source comes from the 3D camera.

> 🚨 **Notice:** This method does not work with Aruco detection method, as the Aruco already provides the object 6D pose.

### Result
#### Return Value
List of all the items detected in the image. Each element in the list is a dictionary, consists of the following keys:
- `"id": int`: ID of the item detected, starting from `0`.
- `"timestamp": float`: Representing the timestamp that the camera captured the image.
- `"obb": [(pxl_x, pxl_y), (w,h), rot]`: Representing the oriented bounding box around the detected object. `(pxl_x, pxl_y)` is the center pixel of the `obb`, `(w,h)` is the width and height of the bounding box (when it is not rotated), and `rot` is the angle of the rotation.
- `"pose": [valid, rvec, tvec]`: Representing the 6D pose of the object detected. `valid` is 1 of the 3 points selected in the 6D pose section form a valid plane, otherwise 0. `rvec`, represents the orientation vector, and `tvec` represents the position vector

For example, this is a sample return value
```python
[{"id": 8, "timestamp": 1716844763.0551214, "obb": [[78, 586], [0, 0], 0], "pose": [1, [0.2939886259984075, 0.002876835873020688, -0.19520854119772166], [-13.236917033395125, -46.815401495552216, 95.23421774762359]]}]
```
#### Configuration
Is a dictionary, representing all the parameters associated to this specific detection instance. Use this configuration later to run the detection directly in your code without running the detection app GUI.
Here is a sample detection configuration
```python
{"poi_value": [], "color_enb": 0, "color_h": [60, 120], "color_s": [85, 170], "color_v": [85, 170], "color_inv": 0, "roi_enb": 0, "roi_value": [], "roi_inv": 0, "intensity_enb": 0, "intensity_alpha": 2.0, "intensity_beta": 50, "method_value": 4, "m_elp_pf_mode": 0, "m_elp_nfa_validation": 1, "m_elp_min_path_length": 50, "m_elp_min_line_length": 10, "m_elp_sigma": 1, "m_elp_gradient_threshold_value": 20, "m_elp_axes": [20, 100], "m_elp_ratio": [0.0, 1.0], "m_circle_inv": 1, "m_circle_type": 0, "m_circle_thr": 127, "m_circle_blur": 3, "m_circle_mean_sub": 0, "m_circle_radius": [1, 30], "m_poly_inv": 1, "m_poly_type": 0, "m_poly_thr": 127, "m_poly_blur": 3, "m_poly_mean_sub": 0, "m_poly_value": 3, "m_poly_area": [100, 100000], "m_poly_perimeter": [10, 100000], "m_cnt_inv": 1, "m_cnt_type": 0, "m_cnt_thr": 127, "m_cnt_blur": 3, "m_cnt_mean_sub": 0, "m_cnt_area": [100, 100000], "m_cnt_perimeter": [10, 100000], "m_aruco_dictionary": "DICT_6X6_250", "m_aruco_marker_length": 10, "m_aruco_refine": "CORNER_REFINE_NONE", "m_aruco_subpix": 0}
```
## Pattern detection API
Once configuration parameter is obtained (using Pattern detection app), you can directly call the detection app in your code.
Here is code example, to directly call the detection app in your code.
```python
from camera import Camera
from dorna_vision import Detect

# create the camera object and connect to it
camera = Camera()
camera.connect()

# create the pattern detection object by passing the camera object
d = Detect(camera)

# use the config dictionary obtained from the pattern detection app
config = {"poi_value": [], "color_enb": 0, "color_h": [60, 120], "color_s": [85, 170], "color_v": [85, 170], "color_inv": 0, "roi_enb": 0, "roi_value": [], "roi_inv": 0, "intensity_enb": 0, "intensity_alpha": 2.0, "intensity_beta": 50, "method_value": 4, "m_elp_pf_mode": 0, "m_elp_nfa_validation": 1, "m_elp_min_path_length": 50, "m_elp_min_line_length": 10, "m_elp_sigma": 1, "m_elp_gradient_threshold_value": 20, "m_elp_axes": [20, 100], "m_elp_ratio": [0.0, 1.0], "m_circle_inv": 1, "m_circle_type": 0, "m_circle_thr": 127, "m_circle_blur": 3, "m_circle_mean_sub": 0, "m_circle_radius": [1, 30], "m_poly_inv": 1, "m_poly_type": 0, "m_poly_thr": 127, "m_poly_blur": 3, "m_poly_mean_sub": 0, "m_poly_value": 3, "m_poly_area": [100, 100000], "m_poly_perimeter": [10, 100000], "m_cnt_inv": 1, "m_cnt_type": 0, "m_cnt_thr": 127, "m_cnt_blur": 3, "m_cnt_mean_sub": 0, "m_cnt_area": [100, 100000], "m_cnt_perimeter": [10, 100000], "m_aruco_dictionary": "DICT_6X6_250", "m_aruco_marker_length": 10, "m_aruco_refine": "CORNER_REFINE_NONE", "m_aruco_subpix": 0}

# run pattern detection
for i in range(10):
    retval = d.pattern(config)

# always close the camera and detection instances once you no longer need them
camera.close()
d.close()
```
`pattern(self, config=None, save_path=None) -> retval`


