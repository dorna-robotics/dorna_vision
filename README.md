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

### Step 1: Open a Jupyter Notebook

Start by running a Jupyter Notebook instance and create a new `Python 3` file.

### Step 2: Run the vision application

Add the following three lines of code to your Jupyter Notebook to start the  pattern detection app:
```python
%matplotlib widget
from dorna_vision import App
App().detect_pattern()
```
There are multiple tabs available to design and tune your detection parameters
> 🚨 **Notice:** Make sure that the camera is connected to your computer when running the app.  

#### Source
Select the source of your image. You can connect an `Intel RealSense` camera and capture the image or directly load and an image.  
Use `Save image as` section to save the captured image.  
Click on `Terminate the app` once your testing is over and you no longer need to run the application.

#### Visual adjustment
- Region of interest: Select specific region of the image where you want to run the detection. Use mouse to create a polygon on the image as the region of interest.
- Intensity: Change the image contrast and brightness.
- Color mask: Remove certain colors. This is useful specially in color based detection applications.

#### Pattern detection
This tab consists of several pattern detection algorithms that you can apply to your image:
- Ellipse detection: Multiple parameters are available to fine tune your ellipse detection method. Use `Axes range` and `Axes ratio` to filter the detected ellipses based on their major and minor axes. For example, if you are detecting circle, then `Axes ratio` parameter is close to `1`.
- Polygon detection: Multiple parameters are available to fine tune your polygon detection method. Use `Side`, `Area` and `Perimeter` to further filter the detected polygons. For example when detecting triangles the `Side` parameter should be `3`.
- Contour detection: Contour detection is a technique used in image processing to identify the boundaries or outlines of objects within an image. Multiple parameters are available to fine tune your contour detection method.
- Aruco detection: Multiple parameters are available to fine tune your Aruco detection method.

#### 6D pose
The 6D pose of an object in a three-dimensional space refers to its position and orientation with respect to the camera coordinate (frame) system. It is defined by six parameters:
- Position: The position of the object is still defined by the 3D coordinates `(x, y, z)` of its center in space.
- Orientation (Axis-Angle Representation)
The orientation is defined by:
  - A rotation axis, represented by a unit vector \(\mathbf{u} = (u_x, u_y, u_z)\).
  - An angle of rotation, \(\theta\), which specifies how much the object is rotated around the axis \(\mathbf{u}\).

In this section select 3 points with respect to the Oriented bounding box around the detected object.  
> 🚨 **Notice:** This method only works when the image source comes from the 3D camera.

> 🚨 **Notice:** This method does not work with Aruco detection method, as the Aruco already provides the object 6D pose.



The 6D pose of an object can also be represented using the axis-angle representation for orientation. This method involves specifying a rotation axis and an angle of rotation around that axis. Here’s how it works:

#### Position
The position of the object is still defined by the 3D coordinates (x, y, z) of its center in space.

#### Orientation (Axis-Angle Representation)
The orientation is defined by:
- A rotation axis, represented by a unit vector \(\mathbf{u} = (u_x, u_y, u_z)\).
- An angle of rotation, \(\theta\), which specifies how much the object is rotated around the axis \(\mathbf{u}\).




