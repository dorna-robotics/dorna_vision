## Installation
Notice that the program has been tested only on Python 3.7+.

### Download
First, use `git clone` to download the repository:  
```bash
git clone https://github.com/dorna-robotics/dorna_vision.git
```
Or simply download the [zip file](https://github.com/dorna-robotics/dorna_vision/archive/master.zip), and unzip the file.  

### Install
Next, go to the downloaded directory, where the `setup.py` file is located, and run:
```bash
python setup.py install --force
```

## Vision application
Dorna Vision application is a development environment inside Jupyter Notebook, that helps you build the vision system and tune the parameters. Once the application is developed, you can directly call the vision modules inside your code without running the Vision App. 

### Step 1: Open a Jupyter Notebook

Start by running a Jupyter Notebook instance and create a new file.

### Step 2: Run the vision application

Add the following three lines of code to your Jupyter Notebook to start the  Vision application:
```python
%matplotlib widget
from vision import App
App().detect_pattern()
```
