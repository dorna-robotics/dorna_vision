import pickle
import ncnn
from ncnn.utils.objects import Detect_Object
from paddleocr import PaddleOCR

class OD(object):
    def __init__(self, path, target_size=416, num_threads=2, use_gpu=False):
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
        self.net.load_param(data["param"])
        self.net.load_model(data["bin"])
        self.class_names = data["classes"]

    
    def __del__(self):
        self.net = None

    # [[obj], ...]
    def __call__(self, img, conf=0.5, max_det=None, cls=[]):
        img_h = img.shape[0]
        img_w = img.shape[1]

        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            img.shape[1],
            img.shape[0],
            self.target_size,
            self.target_size,
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        ex = self.net.create_extractor()
        ex.input("data", mat_in)

        _, mat_out = ex.extract("output")

        objects = []

        # method 1, use ncnn.Mat.row to get the result, no memory copy
        for i in range(mat_out.h):
            values = mat_out.row(i)

            obj = Detect_Object()
            obj.prob = values[1]

            # conf
            if obj.prob < conf:
                continue
            else:
                obj.conf = conf

            # cls
            obj.cls = values[0]-1
            if cls and obj.cls not in cls:
                continue
            
            obj.rect.x = values[2] * img_w
            obj.rect.y = values[3] * img_h
            obj.rect.w = values[4] * img_w - obj.rect.x
            obj.rect.h = values[5] * img_h - obj.rect.y

            objects.append(obj)

        # top elements
        if max_det:
            # Sort the list by 'prob' in descending order
            sorted_objects = sorted(objects, key=lambda x: x.prob, reverse=True)
            
            # Select the top `max_det` elements
            objects = sorted_objects[:max_det]

        
        return objects


class OCR(PaddleOCR):
    def __init__(self, lang='en', use_angle_cls=True):
        self.net = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls)
    
    def ocr(self, img, conf=0.5, cls=True):
        return self.net.ocr(img, drop_score=conf, cls=cls)
    
    def __del__(self):
        self.net = None