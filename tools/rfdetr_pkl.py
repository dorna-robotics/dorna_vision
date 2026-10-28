# The RF-DETR → Jetson pickle, shared verbatim by train_object_detection,
# train_segmentation and train_keypoint (and their checks).
import onnx
from rfdetr.assets.coco_classes import COCO_CLASSES

COCO_CLASS_NAMES = list(COCO_CLASSES.values())


def slot_names(model, num_slots):
    """The model's logit slots named as RF-DETR's predict() names them —
    a pretrained model's sparse COCO ids, a keypoint model's background
    slot, else slot i = class_names[i]. None marks a slot it never reports."""
    names = list(model.class_names)
    keypoints_per_class = list(getattr(model.model.args, "num_keypoints_per_class", None) or [])
    slots = [None] * num_slots
    if num_slots > len(names) and names == COCO_CLASS_NAMES:
        for i, coco_id in enumerate(COCO_CLASSES):
            if i < len(names):
                slots[coco_id] = names[i]
    elif keypoints_per_class and keypoints_per_class[0] == 0:
        foreground = [slot for slot, k in enumerate(keypoints_per_class) if k > 0]
        for i, slot in enumerate(foreground[:len(names)]):
            slots[slot] = names[i]
    else:
        for i, name in enumerate(names[:num_slots]):
            slots[i] = name
    return slots


def rfdetr_model_dict(model, onnx_path, model_type, variant, colors, keypoint_names=None, skeleton=None):
    """The pickle the Jetson's OD / SEG / KP load (dorna_vision/ai.py)."""
    with open(onnx_path, "rb") as f:
        onnx_data = f.read()
    graph = onnx.load_from_string(onnx_data)
    num_slots = next(o for o in graph.graph.output if o.name == "labels").type.tensor_type.shape.dim[2].dim_value
    model_dict = {
        "onnx": onnx_data,
        "cls": slot_names(model, num_slots),
        "colors": colors,
        "meta": {
            "type": model_type,
            "arch": "rf-detr",
            "variant": variant,
            "image_size": int(model.model_config.resolution),
            "precision": "fp16",   # the Jetson's TensorRT engine; RF-DETR's activations fit FP16
            # the engine takes the camera's uint8 BGR image; RF-DETR's graph wants RGB, ImageNet mean/std
            "input": {"format": "uint8-bgr", "channels": "rgb",
                      "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
    }
    if model_type == "kp":
        model_dict["meta"]["num_keypoints_per_class"] = list(model.model.args.num_keypoints_per_class)
        model_dict["meta"]["trace_alpha"] = float(model.model.postprocess.trace_alpha)
        model_dict["keypoint_names"] = keypoint_names
        model_dict["skeleton"] = skeleton
    return model_dict
