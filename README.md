## Installation
Notice that the program has been tested only on Python 3.7+.

### Download
First, use `git clone` to download the repository:  
```bash
git clone https://github.com/dorna-robotics/dorna_vision.git
```
Or simply download the [zip file](https://github.com/dorna-robotics/dorna_vision/archive/master.zip), and uncompress the file.  

### Install
Next, go to the downloaded directory, where the `setup.py` file is located, and run:
```bash
python setup.py install --force
```

## App
```python
%matplotlib widget
from vision import App
App().detect_pattern()
```


# Dorna Vision Tutorial

This tutorial guides you through setting up and running the Dorna Vision application using a Jupyter Notebook. The provided code can be executed as is, and you can customize its parameters to suit your specific application.

## Step 1: Open a Jupyter Notebook

Start by opening a Jupyter Notebook instance and create a new file.

## Step 2: Run the Dorna Vision Application

Add the following three lines of code to your Jupyter Notebook to start the Dorna Vision application:

```python
%matplotlib widget
from dorna_vision import App
App.run()
```
## Step 3: 
Visual adjustment: 
    Region of interest: Use this function to select a region in your image, that you are interested in to run all your vision algorithm. A region selector is provided, by clicking on the imgae and adding new points, you can end adding new points to your region by clicking on the first point created, remove the selected image, by pressing Esc, once the points are added, you can drag each to modify the region. Appl the ROI checkbox is there to enable and disable the ROI. Invert the Invert the roi is to set the ROI region.
    
    dorna_vision
        detection alforithm
        Camera module
        App - run
        Detect("path to the config")
        