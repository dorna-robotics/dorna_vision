import os
import pickle
import ncnn
from ncnn.utils.objects import Detect_Object
from paddleocr import PaddleOCR
import numpy as np
import importlib
from openvino.runtime import Core
from types import SimpleNamespace
import cv2

def hex_to_bgr(hex_color):
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    # Convert to RGB tuple
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Reverse tuple to convert to BGR
    return rgb[::-1]

class OD(object):
    def __init__(self, path, device_name="CPU", **kwargs):
        # load the bin and param
        with open(path, 'rb') as file:
            data = pickle.load(file)
             
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
        del self.model
        del self.compiled_model
        self.core = None  # Release Core instance


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
        orig_h, orig_w = img.shape[:2]
        fixed_size = (self.input_shape[2], self.input_shape[3])  # (height, width) e.g. (416, 416)


        # Resize image directly to fixed size.
        resized_img = cv2.resize(img, (fixed_size[1], fixed_size[0]), interpolation=cv2.INTER_LINEAR)

        # Compute separate scale factors for width and height.
        scale_w = fixed_size[1] / orig_w
        scale_h = fixed_size[0] / orig_h

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
        boxes_xyxy[:, [0, 2]] /= scale_w
        boxes_xyxy[:, [1, 3]] /= scale_h

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
                prob=score,
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
        del self.model
        del self.compiled_model
        self.core = None  # Release Core instance


    def __call__(self, img, conf=0.5, **kwargs):
        retval = []

        # Resize the image to the expected input dimensions (e.g., 224x224)
        img_resized = cv2.resize(img, self.target_size)


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
                [self.cls[predicted_class], probability, self.colors[self.cls[predicted_class]]]
            ]

        return retval




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


class OCR(PaddleOCR):
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
        self.net = None