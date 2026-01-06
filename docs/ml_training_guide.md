# 🧠 ML Training Guide (Detection, Classification & Keypoints)

This guide walks you step-by-step through:

- 📷 Creating and labeling datasets in **Roboflow**
- 📊 Training **Object Detection** models
- 🎓 Training **Classification** models
- 🧑‍🔧 Training **Keypoint Detection** models

You can jump directly to any section:

- [✨ Getting Started: Accounts](#getting-started)
- [📷 Data Collection & Roboflow Setup](#data-collection-roboflow)
- [📊 Object Detection Training ](#object-detection-training)
- [🎓 Classification Training](#classification-training)
- [🧑‍🔧 Keypoint Detection Workflow ](#keypoint-detection-training)

---

<a id="getting-started"></a>
<details>
<summary><strong>✨ Getting Started: Accounts</strong></summary>

To follow this guide you'll need two free accounts. Create them now.

#### A) Google Colab
1. Visit <https://colab.research.google.com>.
2. Sign in with your Google account or create one (**Create account**).

#### B) Roboflow
1. Go to <https://roboflow.com> and click **Sign Up** (free tier is fine).

</details>

---

<a id="data-collection-roboflow"></a>
<details>
<summary><strong>📷 Data Collection & Roboflow Setup</strong></summary>

This section applies to **all three** task types: **classification**, **object detection**, and **keypoint detection**.

---

### 1. Decide What You Want to Detect or Classify

Examples:

- **Object Detection:**
  - Apple / pear
  - Cats, dogs, cars
  - Vials / caps

- **Classification:**
  - Empty vs filled container
  - Good vs bad product
  - Small / medium / large items

- **Keypoint Detection:**
  - Tip and base of a tube
  - Corners of a box

Try to define **clear, non-overlapping categories** so labeling is easy and consistent.

---

### 2. Collect Your Images

You can use:

- Phone camera
- Digital camera
- Robot camera
- Existing images

**General tips:**

1. **Match real conditions**  
   Use similar lighting, distance, and angles to what you expect during real use (inference).

2. **Use diverse scenarios**  
   - Different backgrounds  
   - Different lighting (bright, dim, shadows)  
   - Slight variations in angle and position  

3. **Include edge cases**  
   - Partially visible objects  
   - Slight occlusions  
   - Different orientations  

4. **Balanced dataset**  
   Try to collect similar numbers of images per class (when possible).

You can see an example image set here:  
👉 **Sample folder:** <https://drive.google.com/drive/folders/1U3uedqbndjVraDxkVg8IPtNmZ3qrNuqY?usp=sharing>

---

### 3. Create a Project in Roboflow

1. Go to <https://roboflow.com> and **sign in**.
2. Click **New Project**.

   ![New Project](https://i.imgur.com/94aYGQf.png)

3. Choose the **project type** based on what you want:

   - **Classification** – each image belongs to one class (e.g., *good* or *bad*)
   - **Object Detection** – you draw **boxes** around objects
   - **Keypoint Detection** – you mark **points** (e.g., joints, tips)

You can create separate projects for different tasks (e.g., one for detection, another for classification).

---

### 4. Upload Your Images

1. Go to the **Upload Data** tab in your Roboflow project.

   ![Upload Data](https://i.imgur.com/kLLrlkl.png)

2. Drag and drop your images or select them from your computer.
3. Roboflow will upload and prepare them for annotation.

> Next: Labeling instructions are inside each training dropdown.

</details>

---

<a id="object-detection-training"></a>
<details>
<summary><strong>📊 Object Detection Training </strong></summary>

Train an **object detection** model to find and locate objects in an image.

---

### A) Label Your Images (Detection)

- Go to the **Annotation** tab.
- Use the **Bounding Box Tool** or **Polygon / Smart Polygon Tool** for detection (Note: Both methods are equivalent, as polygons will be converted to boxes automatically.).
- Draw a box (or polygon) tightly around **each** object that belongs to your classes.
- Be consistent:
  - Use the **same class name** for the same object type every time.
  - Include partially visible objects if you would also want the model to detect them at inference time.
  - Avoid loose boxes and avoid cutting off parts of the object.

You will see a screen similar to:

![Generate Dataset](https://i.imgur.com/TeoZQOl.png)


---

### B) Generate a Version of Your Dataset (Detection)

1. Go to the **Generate** tab.
2. Choose your **Train / Validation / Test** split (common: **70% / 20% / 10%**).

![Train/Test Split](https://i.imgur.com/2TpnsPI.png)

3. Apply **Preprocessing** (recommended):  
   - ✅ **Auto-Orient**  
   - ✅ **Resize** to a fixed resolution (e.g., `416×416`)

![Preprocessing](https://i.imgur.com/NTZJohn.png)

4. Click **Generate** to create a **dataset version**.

---

### C) Export the Dataset (Detection)

1. Go to the **Versions** tab.
2. Choose the dataset version you created.
3. Click **Export Dataset** → choose **Pascal VOC** 

![Export Dataset (VOC)](https://i.imgur.com/fhYPjwT.png)

4. Click **Show download code**, then **Continue**. You will paste that code into Colab.

---

### D) Train in Colab

#### 1) Open the Notebook
👉 **Training Notebook:**  
https://colab.research.google.com/drive/1sCEudInd4qJ-fH-ODTRq6NGmZRTV0rO4?usp=sharing

Open it in Colab → go to File → Save a copy in Drive → then click Open Notebook when the popup appears.

#### 2) Turn on GPU
- **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**

#### 3) Insert Your Roboflow Download Code
Paste the Roboflow code (from **Export Dataset → Show download code**) in the first cell. Example:

```python
!pip install roboflow
import cv2
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY_HERE")
project = rf.workspace("YOUR_WORKSPACE_NAME").project("YOUR_PROJECT_NAME")
version = project.version(VERSION_NUMBER)  # e.g. 1, 2, 3...
dataset = version.download("voc")
```


#### 4) Choose YOLOX (Optional)
Note: If you’d like to train a different variant of YOLOX, you can do so by changing the model_type variable at the top of the second cell of the notebook (for example, "nano" or "small").
If you’re not sure which version to use, you can safely leave it as the default setting.:
```python
model_type = "tiny"  # or "medium", "large", etc.
```

#### 5) Run All Cells
- **Runtime → Run all** (Ctrl + F9) 

#### 6) Save Your Model
- After training completes, the Colab notebook will automatically generate and download a .pkl file containing your trained model. You can find it in your browser’s downloads.

</details>

---

<a id="classification-training"></a>
<details>
<summary><strong>🎓 Classification Training</strong></summary>

Train a **classification** model to decide which category an image belongs to (e.g., `empty` vs `grain`).

---

### A) Label Your Images (Classification)

- Open the **Annotation Editor**.

  ![Annotation Editor](https://i.imgur.com/HPAHPLN.png)

- Assign **one class per image** (e.g., `empty`, `grain`, `good`, `bad`).
- You **do not** draw boxes—choose the single label that best describes the entire image.
- Keep classes mutually exclusive and consistently applied.

---

### B) Generate a Version of Your Dataset (Classification)

1. Go to the **Generate** tab.
2. Create a **new version**.
3. Choose a **Train / Validation / Test** split, e.g., **75% / 20% / 5%**.

![Train/Test Split](https://i.imgur.com/2TpnsPI.png)

4. Under **Preprocessing**, select:
   - ✅ **Auto-Orient**
   - ✅ **Resize and stretch to 224×224**

![Preprocessing](https://i.imgur.com/NTZJohn.png)

5. Click **Generate** to create the dataset version.

---

### C) Export the Dataset (Classification)

1. Go to the **Versions** tab.
2. Select your dataset version.
3. Click **Export Dataset** → choose **Folder Structure**.

![Export Dataset (Folder)](https://i.imgur.com/ZfgIwzZ.png)

4. Click **Show download code**, then **Continue**. You will paste that code into Colab.

---

### D) Train in Colab (Classification)

#### 1) Open the Notebook
1) **Open the Notebook**  
👉 **Classification Training Notebook:**  
https://colab.research.google.com/drive/1MufwJDQHpLsh9VsZdn7ZfDxWydJyH6U2?usp=sharing  

Open it in Colab → File → Save a copy in Drive → then click Open Notebook when the popup appears.


#### 2) Turn on GPU (when you start training)
- **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save**

#### 3) Insert Your Roboflow Download Code
Paste the snippet from **Export Dataset → Show download code**. Example:

```python
!pip install roboflow
import cv2
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY_HERE")
project = rf.workspace("YOUR_WORKSPACE_NAME").project("YOUR_PROJECT_NAME")
version = project.version(VERSION_NUMBER)
dataset = version.download("folder")
```


#### 4) Configure Augmentations (Optional)
> **Note:** Augmentations are **optional** and can be configured in the **first cell** of the notebook.  
> If you’re not familiar with them, it’s best to **leave the default settings unchanged**.  
>  
> Example:
> ```python
> # ----- Augmentation Toggle -----
> aug = {
>     "aug_on": True,           # Set to False to disable all augmentations
>     "vertical_flip": True,    # Enable or disable vertical flips
>     "horizontal_flip": True,  # Enable or disable horizontal flips
>     "rotation": 1,            # Degrees of rotation; set to 0 to disable rotation
> }
> ```

#### 5) Run All Cells
- **Runtime → Run all** (Ctrl + F9)

#### 6) Save Your Model
- After training completes, the Colab notebook will automatically generate and download a .pkl file containing your trained model. You can find it in your browser’s downloads.
</details>

---

<a id="keypoint-detection-training"></a>
<details>
<summary><strong>🧑‍🔧 Keypoint Detection Workflow</strong></summary>
