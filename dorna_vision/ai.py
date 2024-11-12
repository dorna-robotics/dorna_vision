import os
import pickle
import ncnn
from ncnn.utils.objects import Detect_Object
from paddleocr import PaddleOCR
import numpy as np

class CLS(object):
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

class OD(object):
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
        self.net = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, precision='fp16',
            det_model_dir="model\ocr\en_PP-OCRv3_det_infer",
            rec_model_dir="model\ocr\en_PP-OCRv4_rec_infer",
            rec_char_dict_path="model\ocr\en_dict.txt",
            vis_font_path="model\ocr\simfang.ttf",
            e2e_char_dict_path="model\ocr\ic15_dict.txt",
            cls_model_dir="model\ocr\ch_ppocr_mobile_v2.0_cls_infer",
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