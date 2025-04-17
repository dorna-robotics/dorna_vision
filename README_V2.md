## Table of Contents

- [Introduction ](#introduction)
- [Items Included ](#items-included)
- [Installation Instructions ](#installation-instructions)
- [Configuring IP Address ](#configuring-ip-address)
- [Detection App ](#detection-app)
- [Initialization ](#initialization)
  - [Camera mounting ](#camera-mounting)
  - [Frame ](#frame)
  - [AI Models ](#ai-models)
- [Image ](#image)
  - [Source ](#source)
  - [Orientation ](#orientation)
  - [Region of Interest ](#region-of-interest)
  - [Intensity ](#intensity)
  - [Color Mask ](#color-mask)
- [Detection Algorithms  ](#detection-algorithms )
  - [Ellipse ](#ellipse)
  - [Polygon ](#polygon)
  - [Contour ](#contour)
  - [Aruco ](#aruco)
  - [OCR ](#ocr)
  - [Bar Code ](#bar-code)
  - [Object Detection ](#object-detection)
  - [Image Classification ](#image-classification)
- [Settings ](#settings)
  - [2D Limit ](#2d-limit)
  - [3D Limit ](#3d-limit)
  - [6D Pose ](#6d-pose)
  - [Output Format ](#output-format)
- [Results ](#results)
  - [API Call ](#api-call)
  - [Return ](#return)
- [Helper Functions ](#helper-functions)
  - [Pixel to XYZ  ](#pixel-to-xyz)
- [Upgrade the Software ](#upgrade-the-software)
- [Troubleshooting and Resolving Common Issues ](#troubleshooting-and-resolving-common-issues)


# Introduction
In this guide, we introduce the Dorna Vision Kit and cover the fundamentals of using vision in automation projects.

# Items Included
<p align="center">
  <img src="docs/images/vision_kit_items.jpg" width="500">
</p>

1. 3D camera and mounting plate 
2. Vision kit computer
3. Ethernet cable
4. Power supply
5. Cable guide brackets
6. USB cable

# Installation Instructions

1. **Attach the Camera**  
   Mount the camera and its plate onto the robot. Align the two holes on the mounting plate with the holes on the robot’s 5th axis body and secure them. Make sure the camera plate is straight and not tilted.
   <p align="center">
   <img src="docs/images/mount_j5.jpg" width="500">
   </p>

2. **Route the USB Cable**  
   Plug the USB cable into the robot's USB port. Ensure the screws on the USB connector are securely fastened to prevent loose connections.

3. **Connect the USB Cable to the Vision Kit**  
   Pass the USB cable to the vision kit and insert it into one of the blue USB 3.0 ports (blue USB ports).
   <p align="center">
   <img src="docs/images/cable_3.jpg" width="500">
   </p>

4. **Organize the USB Cable**  
   Use the cable guides to route the USB cable neatly. Ensure the cable does not obstruct the robot’s movement or become tangled.
   <p align="center">
   <img src="docs/images/cable_1.jpg" width="500">
   </p>
   <p align="center">
   <img src="docs/images/cable_2.jpg" width="500">
   </p>
5. **(Optional) Set Up the Ethernet Splitter**  
   - To minimize the number of Ethernet cables running between devices, use the Ethernet splitter.  
   - Place the robot controller, vision kit, and splitter close together for easy connection.  
   - Connect the robot controller and vision kit to the splitter using the provided short Ethernet cables. Plug these cables into the two Ethernet ports on the splitter (the side with two ports).  
   - Power the splitter by connecting its USB cable to one of the gray USB 2.0 ports on the vision kit.

6. **Connect the Vision Kit to the Ethernet network**  
   Connect the Vision Kit to your Ethernet network using the Ethernet cable to connect it to your computer or router. If you have set up an Ethernet splitter, connect an Ethernet cable from the single-port side of the splitter to your computer or router.


7. **Power the Vision Kit**  
   Connect the USB-C power supply cable to the vision kit computer and plug the other end into a wall outlet.

> 🚨 **Notice:** Ensure all connections, including the camera and cables, are properly set up before turning on the vision kit.

# Connect to the Vision Kit

The vision kit is equipped with a single-board computer running a 64-bit Debian-based Linux distribution to execute your vision applications. The default hostname, username, and password for the vision kit are as follows:

```bash
# hostname
vision

# username
dorna

# password
dorna
```

## Configuring IP Address

If the vision kit is connected to a router with a DHCP server, the IP address will be automatically assigned. If it is connected directly to your computer via the Ethernet port, you will need to manually configure the IP address.

1. **SSH into the Vision Kit**  
   Open a terminal and use the following command to SSH into the vision kit (log in with the password `dorna`):

    ```bash
    ssh dorna@vision.local
    ```

2. **Open the Network Configuration Tool**  
   Type the following command to open the network configuration interface:

    ```bash
    sudo nmtui
    ```

3. **Edit the Connection**  
   - Select `Edit a connection`.
   - Choose the relevant connection type (e.g., `Wired connection 1`).

4. **Configure the IPv4 Settings**  
   - Set `IPv4 CONFIGURATION` to `Manual`.
   - Enter the following details:
     - `Address`: Assign a static IP address to the vision kit (e.g., `192.168.1.100`).
     - `Gateway`: Enter the gateway address (typically your router's IP address, e.g., `192.168.1.1`).
     - `DNS`: Use a DNS server address, such as `8.8.8.8` (Google's public DNS).

5. **Save the Configuration**  
   - Click `OK` to save your settings.
   - Press <Back> and then <OK> to exit the configuration tool.

6. **Reboot the Vision Kit**  
   Type `sudo reboot` to reboot the vision kit and apply the changes.

> 🚨 **Note:** If the vision kit is connected to a router with a DHCP server, the IP address will be assigned automatically by the router.

## Accessing Dorna Lab
To access the Dorna Lab software, enter the vision kit's IP address in your web browser by typing 
```bash
http://vision_kit_ip_address
```  
To connect to a robot from the Dorna Lab session, follow these steps:
1. Click on `Settings` in the top navbar.
2. Go to `General Info`.
3. Under `Robot System IP`, enter the robot's IP address.
4. Click `Set`.

# Detection App

The vision kit comes with a built-in **detection app** software which lets you to visually build your vision applications directly from a web browser. The detection app  generates the necessary API calls for you. You can then use these generated calls within your code to perform object detection, simplifying the process and integrating detection capabilities seamlessly into your applications.

To access the detection app, navigate to:
```bash
http://vision_kit_ip_address:8888/doc/workspaces/auto-g/tree/Downloads/vision/example/detection_app.ipynb
```

Alternatively, access Dorna Lab via the Vision Kit, open a new Python3 kernel in Jupyter Notebook and run the following code:
```python
%matplotlib widget
from dorna_vision import Detection_app
x = Detection_app()
```
</p>
<p align="center">
<img src="docs/images/app.jpg" width="500">
</p>

## Initialization
The first tab in the detection app is called Initialization.

</p>
<p align="center">
<img src="docs/images/initialization.jpg" width="500">
</p>

### Camera mounting
Here we first choose the type of camera `Mounting setup`:
1. **Eye-in-hand**: The configuration where the camera is installed on the robot is called the **Eye-in-hand** configuration. This allows the robot to have a dynamic viewpoint, providing real-time visual feedback from the perspective of the tool or the gripper.
2. **Eye-to-hand**: Another configuration, where the camera is installed at a fixed location and not on the robot, is called **Eye-to-Hand**.

The default configuration for Dorna vision kit is the eye-in-hand configuration.
For the eye-in-hand configuration to work properly, the vision processor needs to know the position of the robot when it takes the image. For that the robot and vision processor need to communicate with each other. We enter the ip address of the robot here so the vision processor can communicate with the robot controller.  
To convert the coordinates from the camera’s image frame of reference to the robot’s one, the Vision system uses a set of vectors that enables this transformation. These vectors are set by default in the Vision processor and should not be changed in order to prevent possible calculation errors.
</p>
<p align="center">
<img src="docs/images/camera_configuration.jpg" width="500">
</p>

> 🚨 **Note:** The eye-in-hand configuration only works for the Dorna TA model.

### Frame
Frame is the coordinate system that when specified, all the reported positions of the detected objects are reported with respect to it. The frame can be specified with 6D pose values of `x, y, z, a, b, c` where `x, y, z` are the translation vector and `a, b, c` represent the orientation of the frame in the reference frame.
The definition of this frame is slightly different in the two cases Eye-in-hand and Eye-to-hand.

#### Eye in hand:
In this case, the camera is mounted on the robot and as such it has no fixed position in space, so we specify the frame with respect to the robot's base.
</p>
<p align="center">
<img src="docs/images/frames.jpg" width="500">
</p>

The default frame with values `[0,0,0,0,0,0]` means that all reported values of the vision system are with respect to the robot base which is the desired case in most applications. 
</p>
<p align="center">
<img src="docs/images/frame.jpg" width="500">
</p>

#### Eye to hand: 
In this case, since the camera itself is a fixed frame in space, we should specify the frame's `x, y, z, a, b, c` values with respect to the camera itself. Here, if we do not specify a frame, all reported values from the vision system will be with respect to the camera's frame which is located in the middle of its two lenses. 

In our scenario, we use Eye In hand configuration and we leave the frame as the default `[0,0,0,0,0,0]`, which is the robot’s base.

### AI Models
AI Models allows us to use AI to identify features, patterns, or objects of interest in images. This can be useful when you need to identify complex things within an image, such as scene context, specific objects, or text, and use that information to make decisions. To use this feature, you must select Object detection or Image classification from the Detection method drop-down menu and introduce in Model path the location path for the AI Model you will use.
</p>
<p align="center">
<img src="docs/images/AI_models.jpg" width="500">
</p>

If we do not want to use AI Models, you can also leave the `Model path` in blank and use classic Computer Vision algorithms instead.
After finishing setting this section according to our needs, we must click on `Initialize Parameters` to initialize the Vision system. If everything was done correctly, you should see an image captured by the camera.

</p>
<p align="center">
<img src="docs/images/initial_image.jpg" width="500">
</p>

> 🚨 **Note:** After the Vision system has been successfully initialized, the Camera Mounting, Frame, and AI Models parameters will no longer be modifiable, and a new initialization will be needed to change them. All the parameters that will be discussed next will remain modifiable after the camera has been initialized.

## Image
The next is the Image tab. Here, the input source can be chosen, and you can use some simple image filters to modify the input image, preparing it for the detection algorithm. 

### Source 
You can either use your stereo camera as the input image source, or you can use an image file when testing your detection algorithm. If `File` is selected, the exact address for the file must be introduced in `File path`.
</p>
<p align="center">
<img src="docs/images/source.jpg" width="500">
</p>

### Orientation
This allows you to rotate the image clockwise in increments of 90° using the `Clockwise rotation` drop-down menu.

<div align="center">

<table>
  <tr>
    <td><img src="docs/images/vial_no_rot.jpg" width="250"></td>
    <td><img src="docs/images/vial_rot_180.jpg" width="250"></td>
  </tr>
  <tr>
    <td align="center">Original image</td>
    <td align="center">Image rotated 180°</td>
  </tr>
</table>

</div>

> 🚨 **Note:** Due to the plotting utility used, the image may appear stretched in the display depending on the orientation, but the image is processed by the Vision system maintaining its original aspect ratio.

### Region of Interest 
Next, we can specify the region of interest. This ROI will restrict the area where the vision system looks for the objects of interest. This can be useful when objects similar to the ones we want to detect are expected to be found in regions that should not be considered, such as a background or an area where objects have already been classified. You can specify this region using the polygon tool available in this section. The polygon can have as many points as needed, and they can be dragged to update the polygon’s shape or position. We have 3 check-boxes available for the ROI:

- `Apply ROI`: must be checked for the region of interest to be effective.
- `Invert Region`: If checked, the Vision system will consider all pixels outside of the polygon and disregard the ones inside instead.
- `Crop`: If checked, the detection algorithm will only receive the pixels contained within a bounding box tangent to our ROI’s polygon. This is especially useful when using AI models, as the pixels within the ROI will be much more relevant within the cropped area than within the original image, which will enhance the results.
- `Offset`: Enables an additional region outside of the ROI polygon to be considered to be considered by the Vision system. This can be useful if we want to use a point or line as the center of our ROI.

</p>
<p align="center">
<img src="docs/images/ROI.jpg" width="500">
</p>

> 🚨 **Note:** After the ROI polygon is closed, a list with the pixel coordinates of all the polygon’s corners will be displayed in the ROI box.

### Intensity
By adjusting the image’s contrast and brightness we can enhance its quality for the detection algorithm. This is useful to adapt the camera to the lighting conditions of the environment where it will be used. After the desired values of `Contrast` and `Brightness` have been selected using the sliding bars, check the `Apply the intensity` checkbox to apply the changes. 

<div align="center">

<table>
  <tr>
    <td><img src="docs/images/jag_org.jpg" width="250"></td>
    <td><img src="docs/images/jag_cont_bright.jpg" width="250"></td>
  </tr>
  <tr>
    <td align="center">Original image</td>
    <td align="center">Image with Contrast 0.84 and Brightness 63</td>
  </tr>
</table>

</div>


> 🚨 **Note:** `Contrast` and `Brightness` settings that work well under specific lighting conditions may not work as well under other ones. To ensure performance consistency, it's recommended to ensure that the lighting conditions are the same every time the Vision system is used.

### Color Mask 
Color masking is a feature similar to the `Region of interest`. Both include or exclude a group of pixels from the image, so the detection algorithm only runs on a specific part of the image. Using the color mask, we include/exclude pixels based on their “HSV” (Hue, Saturation, and Value) values. 
For example, if there are blue and red boxes on a conveyor belt and we only want to detect the red boxes, we can exclude the HSV values corresponding to the blue colors so that the blue pixels are disregarded, and consequently the blue boxes are not detected. This method has four parameters we can control:
- `Hue, Saturation, Value`: These slide bars control the HSV values for the color mask.
- `Invert color mask`: This inverts the color mask so that the pixels with colors within the selected HSV values are excluded from your image and the pixels with colors outside of the selected HSV values are included.
After the color mask parameters have been set, you must check `Apply color mask` to apply the changes and make the color masking effective. 

</p>
<p align="center">
<img src="docs/images/color_mask.jpg" width="500">
</p>

A simple way of knowing what HSV values to use for a particular color is to use the `Color picker`Color picker to select the main color you want to mask. Clicking in the color display will open a color palette where we can see the current color selected and change it by typing the desired RGB (red, green, blue) values for the desired color or by sliding the color slide bar. We can also use the color dropper option by clicking in the dropper. This will allow us to pick any color from the image to use it for the color mask. Selecting a new color will automatically update the HSV value ranges by centering them at the selected color and adding a +-20 range for H, S, and V. The `Color picker` will also display at all times the HSV values of the current color selected and it’s corresponding HEX color code. 

</p>
<p align="center">
<img src="docs/images/color_mask_palette.jpg" width="500">
</p>

## Detection Algorithms 
The next tab is `Detection`. First, we should select what kind of detection method we want to use from the `Detection method` drop-down menu, and then we need to tune the detection parameters to make it detect properly. The patterns included in the app are sufficient for many real-world applications. For more demanding applications or when higher reliability is required, we may need to use AI detection.

### Ellipse 
The first detection method is the ellipse. It has a simple geometrical structure with only 2 parameters describing its shape, but at the same time, it can be mapped to a wide range of closed curved shapes. This detection method has 6 parameters we must set:
- `Auto detection`: Enables the vision system to use incomplete edges as part of candidate ellipses. If not selected, the system will only detect ellipses that are fully continuous.
- `False alarm validation`: Enables the Vision system to statistically assess whether an edge is real or a false positive. Enabling this will make the ellipse detection more robust but could also lead to false negatives.
- `Min path length`: The minimum length for a path to be considered a valid ellipse.
- `Min line length`: The minimum length of a single line segment for it to be considered as a valid part of an ellipse.
- `Blur`: Used to increase the accuracy of the ellipse detection by reducing noise in the image through blurring.
- `Gradient`: Used to control how rapidly pixel values have to change for them to be recognized as edges. Lowering this value will make it easier to identify edges, but could also lead to false positives that could be generate false ellipses. Increasing the value will make the detection more robust, but it will also increase the chance of edges not being recognized and consequently ellipse not being detected.

</p>
<p align="center">
<img src="docs/images/ellipse.jpg" width="500">
</p>

### Polygon 
Polygon detection can be used to detect polygons with different numbers of sides. The actual image that is fed to the polygon detection algorithm is a black and white image that you can see on the interface’s right-hand side. This detection method has six parameters that can be controlled:
- `Inverse`: Inverts the black and white image.
- `Type`: Sets the thresholding method for the segmentation. Can be Otsu(default), Binary or Gaussian.
- `Threshold value`: Sets the minimum value a pixel must reach to be considered above the threshold.
- `Smoothing blur`: Sets how much the image will be blurred to reduce noise.
- `Mean subtract`: Used for adaptive thresholding such that for each pixel the threshold will be the mean of the pixels in its vicinity minus the Mean subtract value.
- `Sides`: The number of sides a detected polygon must have for it to be reported by the system.

</p>
<p align="center">
<img src="docs/images/polygon.jpg" width="500">
</p>

### Contour 
Contour detection algorithms can identify a wide variety of complex shapes, making them ideal for objects too intricate to fit into simple shape categories like ellipses or polygons. However, their flexibility also means they may occasionally detect unintended patterns, leading to false positives. The controllable parameters in this case are similar to the `Polygon` detection. 

</p>
<p align="center">
<img src="docs/images/contour.jpg" width="500">
</p>

### Aruco 
Aruco markers are widely used in vision detection systems for their ability to provide robust, efficient, and accurate localization and identification. Each marker has a distinct ID, allowing for individual recognition and tracking.

</p>
<p align="center">
<img src="docs/images/arucos.jpg" width="500">
</p>

We can set the type of Aruco marker and the dimensions of it here, and then `Aruco detection` will accurately identify the marker's position along with its orientation in the 3D environment using only the 2D image data. There are four parameters that must be set for Aruco detection:
- `Dictionary`: Establishes which Aruco dictionary of the 25 standard ones will be used for the detection.
- `Marker length`: Establishes the real length of the sides of the Aruco markers. 
- `Refinement`: Establishes which refinement strategy, if any at all, will be used to improve the detection results.
- `Subpixel`: Enables subpixel refinement. This can potentially make the detection significantly more precise, at the cost of reducing the detection speed.

</p>
<p align="center">
<img src="docs/images/aruco_detection.jpg" width="500">
</p>

> 🚨 **Note:** It’s important to measure the real length of the marker’s sides after printing them with a precise measuring instrument such as a caliper. Providing an inaccurate marker length can cause significant errors in pose estimation.

### OCR
Optical Character Recognition uses an AI model to detect text characters or numbers. This is useful to detect text in labels, displays, or objects. The AI model is set by default within the application and cannot be changed or modified by the user. 
Confidence: This parameter uses a sliding bar to set the minimum confidence level for the OCR detection model to report a string of detected text.

The `Confidence` parameter uses a sliding bar to set the minimum confidence level for the OCR detection model to report a string of detected text.

</p>
<p align="center">
<img src="docs/images/OCR.jpg" width="500">
</p>

### Bar Code
Bar code detection allows for quick and reliable identification of bar code tags from  commonly used 1D and 2D bar code standards. To configure this, you must select from the `Format` drop-down menu the standard corresponding to bar code tag you want to identify. You can also select `Any` for the system to report all the tags corresponding to any of the standards. Selecting a specific standard could be useful if tags from different bar code standards such as UPC-A and QR are expected to appear in the same image and only one of them should be detected. 

</p>
<p align="center">
<img src="docs/images/bar_code_detection.jpg" width="500">
</p>

### Object Detection
This detection method will enable the system to use the AI model it was provided with during the initialization process to detect objects of interest in the image. You can control the minimum confidence level for an object detection to be reported with the `Confidence` slide bar. If the model was trained to detect objects of multiple classes and it is desired to only detect a limited set of them, you can write the names of all the desired classes separated by commas in the `Detection classes` box.

</p>
<p align="center">
<img src="docs/images/object_detection.jpg" width="500">
</p>

It's sometimes advantageous to use a ROI with the crop feature enabled to enhance the `Object detection` performance. By using a cropped ROI that contains the objects that you want to detect, those objects will have a higher relevance in the image that AI model will receive, potentially improving the detection results.

<div align="center">

<table>
  <tr>
    <td><img src="docs/images/object_detection_no_crop.jpg" width="250"></td>
    <td><img src="docs/images/object_detection_crop.jpg" width="250"></td>
  </tr>
  <tr>
    <td align="center">Original object detection</td>
    <td align="center">Object detection with cropped ROI</td>
  </tr>
</table>

</div>

### Image Classification
This will enable the system to use the AI model it was provided with during the initialization process to classify the image. Similarly, as with `Object detection`, you can control the minimum confidence level for an image class to be reported with the `Confidence` slide bar.

## Settings 
Now that the source and the detection algorithm are all selected, we introduce some useful general adjustments to the detection algorithm. 

### 2D Limit 
The 2D limit setting allows you to set the range of object sizes detected by the detection algorithm based on the properties of their respective bounding box. This setting has two parameters we must set:
- `Aspect ratio`: Defines the minimum and maximum ratio between the smaller axis size to the larger axis size. 
- `Area`: The minimum and maximum number of pixels in the detected bounding box. 

</p>
<p align="center">
<img src="docs/images/2D.jpg" width="500">
</p>

After the `Aspect ratio` and `Area` parameters have been set, you must check `Apply 2D constraints` to make them effective.

### 3D Limit 
The 3D Limit Settings section allows you to define a rectangular prism in the coordinate frame (as defined in the `Initialization` tab) that will be used to filter out detected objects outside of it. This setting has four parameters we must set:
- `X, Y, Z`: The minimum and maximum accepted values for the `x, y, z` dimensions. The detection algorithm will only report detected objects whose center `x, y, z` coordinates are within the prism. 
- `Invert the range`: Check to make the Vision system report the detected objects whose center are outside the defined prism instead of the ones whose center is inside of it.

</p>
<p align="center">
<img src="docs/images/3D.jpg" width="500">
</p>

After the 3D Limit parameters have been set, you must check `Apply 3D constraints` to make them effective.

> 🚨 **Note:** It’s important to note that only the object’s center coordinates are used for the 3D limit filter. It’s possible to have an object not fully confined within the prism that will still be reported if its center is within the prism. 

### 6D Pose 
All detection algorithms that we discussed in the previous tab, output the 2D pixel coordinates of the detected objects (except for the Aruco detection). However, since our camera provides depth information for each pixel, it is possible to translate the output results of these detection algorithms from the 2D image space to the 3D space. 
The Dorna detection algorithm uses three key points for each instance of the detected pattern to find its pose in the 3D space. We should choose these three points so that they reside on the surface of our detected objects. 

</p>
<p align="center">
<img src="docs/images/6D_example.jpg" width="500">
</p>

Note that the larger and shorter axes of the ellipse will get rotated and rescaled for the bounding box to match with the bounding box of the detected pattern. You should specify 3 points on this ellipse that will be mapped on your object’s surface. 

</p>
<p align="center">
<img src="docs/images/6D.jpg" width="500">
</p>

While selecting the key points on object, keep an eye on the output result below, you’ll be able to observe the yellow dots appearing on the detected pattern, this helps you find the proper place for setting the sample points. 

</p>
<p align="center">
<img src="docs/images/6D_detection.jpg" width="500">
</p>

After identifying the 2D object in the image, the detection algorithm uses the 3D coordinates of the 3 specified points to create a triangle in 3D space. Then, the object’s position and rotation in 3D space are determined based on the position and orientation of this triangle. 

### Output Format 
In this section, you can format the output results of the detection algorithm. 
- `Max detections per run`: Limits the number of detections results the algorithm returns, the default value for this parameter is 1. 
- `Shuffle return data`: Makes the algorithm randomly shuffle the list of the detected instances each time it gets executed. 
- `Save the annotated image`: Saves the annotated image. This may be helpful for debugging the detection algorithm. However, in most cases, it should be disabled as storing image files will occupy your system's memory. This option could also be used for collecting data for the machine learning tasks which you'll learn about later. 
- `Save the annotated ROI image`: Save the ROI image.

</p>
<p align="center">
<img src="docs/images/output.jpg" width="500">
</p>

## Results 
The Application Builder is a tool for setting up the detection algorithm in a way that suits your tasks. Our final goal would be to extract detection results from the algorithm and use it for decision making in robotic applications. In the result tab, you can find what is needed for this task. 

</p>
<p align="center">
<img src="docs/images/results.jpg" width="500">
</p>

### API Call 
The API call is the Python code that you need to run in your program to initialize the camera and detection algorithm and then to execute the detection algorithm. After initializing the robot and detection objects, the code creates the `prm` variable which includes all the parameters needed to call your detection algorithm based on the configurations selected in the application.
After this, the detection algorithm with input `prm` is called. The detection algorithm will return all instances of the detected objects in the `retval` variable. 

> 🚨 **Note:** The API calls can be executed as many times as needed within a program. As long as the required Camera Mounting, Frame and AI Models parameters don’t change, only one initialization is required.

### Return

The `retval` fields are as follows: 

- `timestamp`: If the source was selected as `Stereo camera`, the timestamp will be that of the moment when the camera took the image. If the source is selected as `File`, the timestamp will be that of the moment when the Vision system opened the file. The timestamp is expressed in seconds. 
- `cls`: The class key. It describes what is the type of the detected object. For example, its value could be "cnt" for contour, "poly" for polygon, and so on. In the case of the "OCR" detection, the value of this key will be the exact detected text. In the case of `Object detection` and `Image classification`, the class will be the one detected by the AI model.
- `conf`: This is the statistical confidence value, describing how sure the algorithm is of the pattern it has detected. This value has no use when using the classic detection methods, but it will be very important when using ML detection methods. 
- `center` and `corners`: Correspond to the geometrical position of the detected object’s bounding box corners and its center. The values are in pixel units, and the reference of the coordinate is the top-left corner of the input image. 
- `xyz`: Is the position in the 3D space, corresponding to the center pixel of the detected pattern. 
- `tvec` and `rvec`: Are the translational (xyz) and rotational (abc) part of the 6D pose of the detected object, calculated based on the 6D pose algorithm that we explained earlier. 
- `color`: The BGR (blue, green, red) values of the color of the detected object’s bounding box.


## Helper Functions 

### Pixel to XYZ 
This function can be used to extract the 3D position of any of the pixels on the screen. It is a handy tool to check whether the camera and the initialization parameters have been set up correctly or not.


# Upgrade the Software

To upgrade the vision kit software, SSH into the vision kit and run the following command:

```bash
sudo mkdir -p /home/dorna/Downloads && sudo rm -rf /home/dorna/Downloads/upgrade && sudo mkdir /home/dorna/Downloads/upgrade && sudo git clone -b vision https://github.com/dorna-robotics/upgrade.git /home/dorna/Downloads/upgrade && cd /home/dorna/Downloads/upgrade && sudo sh setup.sh dorna_ta
```
> 🚨 **Note:** Before trying to upgrade, make sure the Vision controller is connected to the internet. This can be done by connecting it directly to your router, SSH into it, and running the command ifconfig. If the controller is connected to the internet, running this command will display the controller’s IP address.

# Troubleshooting and Resolving Common Issues

## 1. Running Multiple Detection Sessions
You cannot run multiple sessions of the detection app at the same time. If you are done with a detection session and need to start a new one, you should kill the existing kernel in Jupyter Notebook. To do this:
- Go to the **Jupyter Notebook** interface.
- In the **"Running"** tab, you will see a list of active notebooks.
- Find the notebook running the detection app.
- Click the **"Shutdown"** button next to the notebook to kill the kernel and stop the session.

This will free up resources for a new session.

## 2. Camera Not Responsive
If the camera is not responsive, follow these steps:
- Disconnect and reconnect the USB cable from the vision kit side.
- After reconnecting, kill the Jupyter session running the program by following the steps above.
- Re-run the program by restarting the Jupyter notebook.

This should resolve any camera connectivity issues and allow the detection app to function properly again.
