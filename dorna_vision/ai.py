import os
import pickle
import ncnn
from ncnn.utils.objects import Detect_Object
import numpy as np
import importlib
from openvino import Core
from types import SimpleNamespace
import cv2
import math
from dorna_vision.visual import *
from dorna_vision.util import *

def letterbox_image(image, target_size=(640, 640), pad_color=(128, 128, 128)):
    """
    Resize image to fit target_size while maintaining aspect ratio.
    Pads the image with the specified pad_color using cv2.copyMakeBorder and returns the new image,
    scale factor, and padding values.
    """
    ih, iw = image.shape[:2]
    tw, th = target_size  # target width and height
    scale = min(tw / iw, th / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w = (tw - nw) // 2
    pad_h = (th - nh) // 2

    # Calculate exact border sizes (handles odd differences gracefully)
    top = pad_h
    bottom = th - nh - pad_h
    left = pad_w
    right = tw - nw - pad_w

    # Add borders using OpenCV's highly optimized copyMakeBorder
    new_image = cv2.copyMakeBorder(image_resized, top, bottom, left, right,
                                   borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return new_image, scale, pad_w, pad_h


def hex_to_bgr(hex_color):
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB tuple
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Reverse tuple to convert to BGR
    return rgb[::-1]

class KP(object):
    def __init__(self, path=None, device_name="CPU", **kwargs): 
        # load the bin and param
        with open(path, 'rb') as file:
            data = pickle.load(file)
        
        self.detection = {
            "od": OD(path=None, device_name=device_name, data={"bin":data["bin"], "xml":data["xml"], "cls":data["cls"], "colors":data["colors"], "meta":data["meta"]}),
            "kp": {k: OD(path=None, device_name=device_name, data=data["kp"][k]) for k in data["kp"] if data["kp"][k]},
        }

    

    def od(self, img, conf=0.5, cls=[], **kwargs):
        return self.detection["od"](img, conf=conf, cls=cls, **kwargs)


    def kp(self, img, label, bb, offset=20,conf=0.5, cls=[], **kwargs):
        retval = []

        roi = ROI(img, corners=bb, crop=True, offset=offset) 
        
        # check if label exists, if not, return empty list
        if label not in self.detection["kp"]:
            return []

        # list of valid keypoints
        valid_cls = list(self.detection["kp"][label].cls)
        if cls:
            valid_cls = [c for c in cls if c in valid_cls]

        # detection
        retval = self.detection["kp"][label](roi.img, conf=conf, cls=valid_cls)

        # format
        for r in retval:
            r.center = roi.pxl_to_orig([r.rect.x+r.rect.w/2, r.rect.y+r.rect.h/2])

        # return
        return retval

    def __del__(self):
        # del od
        self.detection["od"].__del__()

        # del kp
        for k in self.detection["kp"]:
            self.detection["kp"][k].__del__()
        

class OD(object):
    def __init__(self, path=None, device_name="CPU", **kwargs):
        # load the bin and param
        if path is not None:
            with open(path, 'rb') as file:
                data = pickle.load(file)
        else:
            data = kwargs["data"]
             
        # classes
        self.cls = data["cls"]

        # colors
        self.colors = {k:hex_to_bgr(data["colors"][k]) for k in data["colors"]}

        # Load model
        self.core = Core()
        self.model = self.core.read_model(model=bytes(data['xml'], 'utf-8'), weights=bytes(data['bin']))
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device_name)
        self.input_shape = self.compiled_model.input(0).shape  # Expecting shape [N, C, H, W]

        
    def __del__(self):
        try:
            del self.model
            del self.compiled_model
            self.core = None  # Release Core instance
        except:
            pass

    # Postprocessing: adjust outputs to produce correct bounding boxes.
    def _postprocess(self, outputs, img_size, p6=False):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
        hsizes = [img_size[0] // stride for stride in strides]
        wsizes = [img_size[1] // stride for stride in strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))
        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
        return outputs

    # Non-Maximum Suppression (NMS)
    def _nms(self, boxes, scores, nms_thr):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= nms_thr)[0]
            order = order[inds + 1]
        return keep


    def _multiclass_nms(self, boxes, scores, nms_thr, score_thr):
        cls_inds = scores.argmax(1)
        cls_scores = scores[np.arange(len(cls_inds)), cls_inds]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            return None
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        valid_cls_inds = cls_inds[valid_score_mask]
        keep = self._nms(valid_boxes, valid_scores, nms_thr)
        if keep:
            dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1)
        return dets



    # [[obj], ...]
    def __call__(self, img, conf=0.5, cls=[], max_det=None, nms_thr=0.3, **kwargs):
        # objects
        objects = []

        # Preprocess image: Instead of preserving the aspect ratio with padding,
        # we directly resize every image to the fixed input size (e.g. 416x416).
        #orig_h, orig_w = img.shape[:2]
        fixed_size = (self.input_shape[2], self.input_shape[3])  # (height, width) e.g. (416, 416)

        """
        # Resize image directly to fixed size.
        resized_img = cv2.resize(img, (fixed_size[1], fixed_size[0]), interpolation=cv2.INTER_LINEAR)

        # Compute separate scale factors for width and height.
        scale_w = fixed_size[1] / orig_w
        scale_h = fixed_size[0] / orig_h
        """

        resized_img, scale, pad_w, pad_h = letterbox_image(
            img, target_size=(fixed_size[1], fixed_size[0]), pad_color=(128, 128, 128)
        )


        # Prepare image for inference (convert to CHW and add batch dimension).
        preprocessed_img = resized_img.transpose((2, 0, 1))
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0).astype(np.float32)

        # Run inference.
        output = self.compiled_model([preprocessed_img])[self.compiled_model.output(0)]
        
        # Postprocess results.
        predictions = self._postprocess(output, fixed_size)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4, None] * predictions[:, 5:]

        # Convert center-format boxes to xyxy format.
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        
        # Scale bounding boxes back to original image dimensions.
        """     
        boxes_xyxy[:, [0, 2]] /= scale_w
        boxes_xyxy[:, [1, 3]] /= scale_h
        """
        boxes_xyxy[:, [0, 2]] -= pad_w
        boxes_xyxy[:, [1, 3]] -= pad_h
        boxes_xyxy /= scale

        # nms
        dets = self._multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=conf)

        # Visualize and show results.
        if dets is None:
            return objects
        
        final_boxes = dets[:, :4].tolist()
        final_scores = dets[:, 4].tolist()
        final_cls_inds = dets[:, 5].tolist()
        for i in range(len(final_boxes)):
            # conf
            score = final_scores[i]
            if score < conf:
                continue
            
            # class
            label = self.cls[int(final_cls_inds[i])]
            if cls and label not in cls:
                continue

            # obj
            obj = SimpleNamespace(
                prob=float(score),
                cls=label,
                rect=SimpleNamespace(
                    x=final_boxes[i][0],
                    y=final_boxes[i][1],
                    w=final_boxes[i][2] - final_boxes[i][0],
                    h=final_boxes[i][3] - final_boxes[i][1],
                ),
                color=self.colors[label],
            )
            objects.append(obj)

        # Sort the list by 'prob' in descending order
        objects = sorted(objects, key=lambda x: x.prob, reverse=True)

        # top elements
        if max_det:            
            # Select the top `max_det` elements
            objects = objects[:max_det]

        return objects
    

class CLS(object):
    def __init__(self, path, target_size=(224, 224), device_name="CPU", **kwargs):
        # load the bin and param
        with open(path, 'rb') as file:
            data = pickle.load(file)
             
        # classes
        self.cls = data["cls"]
        self.colors = {k:hex_to_bgr(data["colors"][k]) for k in data["colors"]}
        
        # Initialize OpenVINO Core
        self.core = Core()
        self.model = self.core.read_model(model=bytes(data['xml'], 'utf-8'), weights=bytes(data['bin']))
        self.compiled_model = self.core.compile_model(model=self.model, device_name=device_name)

        self.target_size = target_size

        # Retrieve the model's input and output information
        self.input_tensor = self.compiled_model.input(0)
        self.output_tensor = self.compiled_model.output(0)

        # Mean and scale values as per OpenVINO's page (BGR channel order)
        self.mean = np.array([123.675, 116.28, 103.53])
        self.scale = np.array([58.395, 57.12, 57.375])


    def __del__(self):
        try:
            del self.model
            del self.compiled_model
            self.core = None  # Release Core instance
        except:
            pass


    def __call__(self, img, conf=0.5, **kwargs):
        retval = []

        # Resize the image to the expected input dimensions (e.g., 224x224)
        #img_resized = cv2.resize(img, self.target_size)
        img_resized, _, _, _ = letterbox_image(img, self.target_size)


        # Normalize the image by subtracting the mean and dividing by scale for each channel
        img_normalized = img_resized.astype(np.float32)
        for i in range(3):  # For each channel (BGR)
            img_normalized[:, :, i] = (img_normalized[:, :, i] - self.mean[i]) / self.scale[i]

        # Convert the image from HWC to CHW format
        img_chw = img_normalized.transpose(2, 0, 1)

        # Add a batch dimension to create shape (1, C, H, W)
        input_data = np.expand_dims(img_chw, axis=0)

        # Run inference using the compiled model
        results = self.compiled_model([input_data])

        # Process and display the results
        output_data = results[self.output_tensor]

        # Apply the softmax function to convert logits to probabilities
        exp_scores = np.exp(output_data)
        softmax_probs = exp_scores / np.sum(exp_scores)

        # Get the predicted class
        predicted_class = np.argmax(softmax_probs)

        # Get the probability (confidence score) of the predicted class
        probability = softmax_probs[0][predicted_class]

        if probability > conf:
            retval = [
                [self.cls[predicted_class], float(probability), self.colors[self.cls[predicted_class]]]
            ]

        return retval


class OCR:
    def __init__(self, device="CPU"):
        """
        Initialization:
          - Load the dictionary from dict_path.
          - Initialize OpenVINO core.
          - Load and compile the detection model using the provided XML and BIN files.
          - Load and compile the recognition model.
        """
        spec = importlib.util.find_spec("dorna_vision")
        if spec and spec.origin:
            model_folder = os.path.dirname(spec.origin)  # Store the path

        det_model_path = os.path.join(model_folder,"model", "ocr", "horizontal-text-detection-0001.xml")
        det_weights_path = os.path.join(model_folder,"model", "ocr", "horizontal-text-detection-0001.bin")
        rec_model_path = os.path.join(model_folder,"model", "ocr", "inference.pdmodel")
        dict_path = os.path.join(model_folder,"model", "ocr", "ppocr_keys_v1.txt")
        
        # Load dictionary for CTC decoding
        if not os.path.exists(dict_path):
            raise FileNotFoundError(f"Dictionary file {dict_path} not found.")
        with open(dict_path, "r", encoding="utf-8") as f:
            keys = [line.strip() for line in f if line.strip() != ""]
        self.char_list = [""] + keys  # Reserve index 0 as blank

        # Initialize OpenVINO core
        self.core = Core()

        # Load and compile detection model
        if not os.path.exists(det_model_path) or not os.path.exists(det_weights_path):
            raise FileNotFoundError("Detection model files not found.")
        self.det_model = self.core.read_model(model=det_model_path, weights=det_weights_path)
        
        # Expected input shape [1, 3, 640, 640]
        self.det_model.reshape({self.det_model.input(0): [1, 3, 640, 640]})
        self.det_compiled = self.core.compile_model(self.det_model, device)
        self.det_input = self.det_compiled.input(0)
        self.det_output = self.det_compiled.output(0)

        # Load and compile recognition model
        if not os.path.exists(rec_model_path):
            raise FileNotFoundError("Recognition model file not found.")
        self.rec_model = self.core.read_model(model=rec_model_path)
        
        # Expected input shape: [1, 3, 48, -1] (height fixed, dynamic width)
        self.rec_model.reshape({self.rec_model.input(0): [1, 3, 48, -1]})
        self.rec_compiled = self.core.compile_model(self.rec_model, device)
        self.rec_input = self.rec_compiled.input(0)
        self.rec_output = self.rec_compiled.output(0)


    def __del__(self):
        try:
            del self.det_model
            del self.det_compiled
            del self.rec_model
            del self.rec_compiled
            self.core = None  # Release Core instance
        except Exception as ex:
            pass

    def preprocess_crop(self, crop, target_height=48, target_width=320):
        """
        Preprocess a cropped text region for recognition:
        - Convert BGR to RGB.
        - Resize to target height while preserving aspect ratio.
        - Normalize pixels to [-1, 1] and pad to target width.
        Returns the preprocessed image with an added batch dimension.
        """
        #crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        h, w, _ = crop.shape
        ratio = w / h
        new_w = int(math.ceil(target_height * ratio))
        new_w = min(new_w, target_width)
        resized = cv2.resize(crop, (new_w, target_height))
        resized = resized.astype("float32") / 255.0
        resized = (resized - 0.5) / 0.5
        resized = resized.transpose(2, 0, 1)
        padded = np.zeros((3, target_height, target_width), dtype="float32")
        padded[:, :, :new_w] = resized
        return np.expand_dims(padded, 0)


    def ctc_decode(self, logits, char_list):
        """
        Simple CTC decoding:
        - Take argmax along classes for each timestep.
        - Collapse consecutive duplicates and ignore blank token (index 0).
        Returns the decoded string.
        """
        indices = np.argmax(logits, axis=2)[0]
        decoded = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != 0:
                decoded.append(char_list[idx])
            prev = idx
        return "".join(decoded)


    def ocr(self, img, conf=0.5, detection_enabled=True, **kwargs):
        """
        Run the OCR pipeline:
          - If detection_enabled is True, run text detection on the full image, map detected regions back to original coordinates,
            and run recognition on each valid detected region.
          - If detection_enabled is False, assume img is a pre-cropped text region and run recognition directly.
        Parameters:
          img: Image matrix (read using cv2, not a file path)
          detection_enabled: Flag to enable text detection (True) or recognition-only (False)
          conf: Confidence threshold for filtering detection results
        Returns:
          A list of dictionaries for each detected region with keys: 'box', 'text', and 'conf'
        """
        results = []
        if detection_enabled:
            # Preprocess the image for detection using letterbox resize
            det_img, scale, pad_w, pad_h = letterbox_image(img, (640, 640))
            #det_input_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB).astype("float32")
            det_input_img = det_img.astype("float32")
            det_input_img = det_input_img.transpose(2, 0, 1)
            det_input_img = np.expand_dims(det_input_img, 0)

            det_result = self.det_compiled([det_input_img])[self.det_output]
            det_result = np.array(det_result)

            if det_result.ndim == 3:
                det_result = np.squeeze(det_result, axis=0)

            for b in det_result:
                x1, y1, x2, y2, score = b
                if score < conf:
                    continue
                # Ensure proper box coordinates
                x_min_letter = min(x1, x2)
                x_max_letter = max(x1, x2)
                y_min_letter = min(y1, y2)
                y_max_letter = max(y1, y2)
                # Map coordinates from letterboxed image back to original image
                x_min = int((x_min_letter - pad_w) / scale)
                x_max = int((x_max_letter - pad_w) / scale)
                y_min = int((y_min_letter - pad_h) / scale)
                y_max = int((y_max_letter - pad_h) / scale)
                if x_min >= x_max or y_min >= y_max:
                    continue
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(img.shape[1], x_max)
                y_max = min(img.shape[0], y_max)
                crop = img[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    continue

                rec_input_crop = self.preprocess_crop(crop)
                rec_result = self.rec_compiled([rec_input_crop])[self.rec_output]
                text = self.ctc_decode(rec_result, self.char_list)
                results.append([
                    [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]],
                     [text, float(score)],
                ])

                
        else:
            # Recognition-only mode: assume img is a cropped text region.
            rec_input_img = self.preprocess_crop(img)
            rec_result = self.rec_compiled([rec_input_img])[self.rec_output]

            text = self.ctc_decode(rec_result, self.char_list)
            h, w = img.shape[:2]
            results.append([
                [[0, 0], [0, h], [w, h], [w, 0]],
                [text, 1],
            ])
        return results


class CLS_NCNN(object):
    def __init__(self, path, target_size=224, num_threads=2, use_gpu=False, **kwargs):
        self.path = path
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        # load the bin and param
        with open(self.path, 'rb') as file:
            data = pickle.load(file)

        # Save them temporarily as .param and .bin files for model loading
        with open('temp.param', 'w', encoding='utf-8') as f:
            f.write(data['param'])

        with open('temp.bin', 'wb') as f:
            f.write(data['bin'])

        # load the model    
        self.net.load_param("temp.param")
        self.net.load_model("temp.bin")
        self.cls = data["cls"]
        self.model = data["meta"]["model"]

        # remove the temporary files
        os.remove('temp.param')
        os.remove('temp.bin')

    
    def __del__(self):
        self.net = None

    # [[obj], ...]
    def __call__(self, img, conf=0.5, **kwargs):
        retval = []
        if self.model.startswith("shufflenet_v2"):
            # img
            img_h = img.shape[0]
            img_w = img.shape[1]
            mat_in = ncnn.Mat.from_pixels_resize(
                img,
                ncnn.Mat.PixelType.PIXEL_BGR,
                img_w,
                img_h,
                self.target_size,
                self.target_size,
            )
            mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

            # extractor
            ex = self.net.create_extractor()
            ex.input("in0", mat_in)
            _, mat_out = ex.extract("out0")
        
            # manually call softmax on the fc output
            # convert result into probability
            softmax = ncnn.create_layer("Softmax")

            pd = ncnn.ParamDict()
            softmax.load_param(pd)

            softmax.forward_inplace(mat_out, self.net.opt)
            mat_out = mat_out.reshape(mat_out.w * mat_out.h * mat_out.c)

            cls_scores = np.array(mat_out).astype(float).tolist()
            max_cls, max_score = max(zip(self.cls, cls_scores), key=lambda x: x[1])

            # confidence
            if max_score >= conf:        
                retval = [[max_cls, max_score]]

        return retval


class OD_NCNN(object):
    def __init__(self, path, target_size=416, num_threads=2, use_gpu=False, **kwargs):
        self.path = path
        self.target_size = target_size
        self.num_threads = num_threads
        self.use_gpu = use_gpu

        self.mean_vals = []
        self.norm_vals = [1 / 255.0, 1 / 255.0, 1 / 255.0]

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = self.use_gpu
        self.net.opt.num_threads = self.num_threads

        # load the bin and param
        with open(self.path, 'rb') as file:
            data = pickle.load(file)

        # Save them temporarily as .param and .bin files for model loading
        with open('temp.param', 'w', encoding='utf-8') as f:
            f.write(data['param'])

        with open('temp.bin', 'wb') as f:
            f.write(data['bin'])

        # load the model    
        self.net.load_param("temp.param")
        self.net.load_model("temp.bin")
        self.cls = data["cls"]
        self.model = data["meta"]["model"]

        # remove the temporary files
        os.remove('temp.param')
        os.remove('temp.bin')

    
    def __del__(self):
        self.net = None


    # [[obj], ...]
    def __call__(self, img, conf=0.5, cls=[], max_det=None, **kwargs):
        objects = []
        if self.model.startswith("yolov4"):
            img_h = img.shape[0]
            img_w = img.shape[1]
            mat_in = ncnn.Mat.from_pixels_resize(
                img,
                ncnn.Mat.PixelType.PIXEL_BGR2RGB,
                img_w,
                img_h,
                self.target_size,
                self.target_size,
            )
            mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

            ex = self.net.create_extractor()
            ex.input("data", mat_in)

            _, mat_out = ex.extract("output")
            

            # method 1, use ncnn.Mat.row to get the result, no memory copy
            for i in range(mat_out.h):
                values = mat_out.row(i)

                obj = Detect_Object()
                obj.prob = values[1]

                # conf
                if obj.prob < conf:
                    continue

                # cls
                obj.cls = self.cls[int(values[0]-1)]
                if cls and obj.cls not in cls:
                    continue
                
                obj.rect.x = values[2] * img_w
                obj.rect.y = values[3] * img_h
                obj.rect.w = values[4] * img_w - obj.rect.x
                obj.rect.h = values[5] * img_h - obj.rect.y

                objects.append(obj)

            # Sort the list by 'prob' in descending order
            objects = sorted(objects, key=lambda x: x.prob, reverse=True)

            # top elements
            if max_det:            
                # Select the top `max_det` elements
                objects = objects[:max_det]
        
        return objects


"""class OCR_NCNN(PaddleOCR):
    def __init__(self, lang='en', use_angle_cls=True, **kwargs):
        spec = importlib.util.find_spec("dorna_vision")
        if spec and spec.origin:
            model_folder = os.path.dirname(spec.origin)  # Store the path
        self.net = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, precision='fp16',
            det_model_dir= os.path.join(model_folder,"model", "ocr", "en_PP-OCRv3_det_infer"),
            rec_model_dir= os.path.join(model_folder,"model", "ocr", "en_PP-OCRv4_rec_infer"),
            rec_char_dict_path= os.path.join(model_folder,"model", "ocr", "en_dict.txt"),
            vis_font_path= os.path.join(model_folder,"model", "ocr", "simfang.ttf"),
            e2e_char_dict_path=os.path.join(model_folder,"model", "ocr", "ic15_dict.txt"),
            cls_model_dir=os.path.join(model_folder,"model", "ocr", "ch_ppocr_mobile_v2.0_cls_infer"),
            show_log=False)
        
    
    def ocr(self, img, conf=0.5, cls=True, **kwargs):
        retval = []
        _retval = self.net.ocr(img, cls=cls)[0]
        if _retval is not None:
            for r in _retval:
                if r[1][1] < conf:
                    continue
                retval.append(r)
        return retval
    

    def __del__(self):
        self.net = None"""
