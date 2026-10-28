"""Classifier head-to-head on one dataset (Roboflow folder export):
DINOv3 backbones frozen + a linear head, vs MobileNetV4-Small fine-tuned
with train_classification.ipynb's recipe. Same letterbox (the notebook's)
for all. Prints one result line per model; JSON summary at the end."""
import copy, json, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
from timm.utils import ModelEmaV2
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = sys.argv[1]
torch.manual_seed(0); np.random.seed(0)
dev = torch.device("cuda")
IMAGE_SIZE = (224, 224)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class LetterboxTransform:  # train_classification.ipynb, verbatim
    def __init__(self, target_size=(224, 224), color=128):
        self.target_size = target_size
        self.color = color

    def __call__(self, image):
        arr = np.array(image)
        ih, iw = arr.shape[:2]
        tw, th = self.target_size
        scale = min(tw / iw, th / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = np.full((th, tw, 3), self.color, dtype=np.uint8)
        pw = (tw - nw) // 2
        ph = (th - nh) // 2
        out[ph:ph + nh, pw:pw + nw] = resized
        return Image.fromarray(out)


val_tx = transforms.Compose([LetterboxTransform(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
split = {s: datasets.ImageFolder(os.path.join(ROOT, s), transform=val_tx) for s in ("train", "valid", "test")}
classes = split["train"].classes
print("classes", classes, {s: len(d) for s, d in split.items()}, flush=True)
results = []


# ── DINOv3: frozen backbone + linear head ────────────────────────────
@torch.no_grad()
def features(model, ds):
    out, ys = [], []
    for x, y in DataLoader(ds, batch_size=64, shuffle=False, num_workers=2):
        out.append(model(x.to(dev)).float().cpu()); ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


for name in ["vit_small_patch16_dinov3.lvd1689m", "vit_small_plus_patch16_dinov3.lvd1689m",
             "vit_base_patch16_dinov3.lvd1689m", "convnext_tiny.dinov3_lvd1689m",
             "convnext_small.dinov3_lvd1689m", "convnext_base.dinov3_lvd1689m"]:
    t0 = time.time()
    model = timm.create_model(name, pretrained=True, num_classes=0).to(dev).eval()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    f = {s: features(model, d) for s, d in split.items()}
    head = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000))
    head.fit(*f["train"])
    r = {"model": name, "params_M": round(params, 1), "method": "frozen + linear (C by 5-fold CV on train)",
         "valid_acc": round(100 * head.score(*f["valid"]), 2), "test_acc": round(100 * head.score(*f["test"]), 2),
         "train_acc": round(100 * head.score(*f["train"]), 2), "seconds": round(time.time() - t0, 1)}
    results.append(r); print(json.dumps(r), flush=True)
    del model; torch.cuda.empty_cache()


# ── MobileNetV4-Small: train_classification.ipynb's recipe ───────────
EPOCHS, BATCH_SIZE, LR, LABEL_SMOOTHING = 50, 64, 0.003, 0.1
WARMUP_EPOCHS = min(5, max(1, EPOCHS // 10))
train_tx = transforms.Compose([
    LetterboxTransform(IMAGE_SIZE),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), shear=0, fill=128),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD), transforms.RandomErasing(p=0.1)])
train_ds = datasets.ImageFolder(os.path.join(ROOT, "train"), transform=train_tx)
labels_all = [l for _, l in train_ds.samples]
counts = np.bincount(labels_all, minlength=len(classes))
sampler = WeightedRandomSampler((1.0 / counts[labels_all]).tolist(), num_samples=len(labels_all), replacement=True)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(split["valid"], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(split["test"], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
total_steps = max(1, len(train_loader) * EPOCHS)
EMA_DECAY = max(0.95, min(0.9999, 1.0 - 1.0 / max(50, total_steps // 5)))


def acc(m, loader):
    m.eval(); c = t = 0
    with torch.no_grad():
        for x, y in loader:
            p = m(x.to(dev)).argmax(1).cpu(); c += (p == y).sum().item(); t += len(y)
    return 100 * c / max(t, 1)


t0 = time.time()
name = "mobilenetv4_conv_small"
model = timm.create_model(name, pretrained=True, num_classes=len(classes)).to(dev)
nn.init.zeros_(model.get_classifier().weight); nn.init.zeros_(model.get_classifier().bias)
ema = ModelEmaV2(model, decay=EMA_DECAY, device=dev)
crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
sched = SequentialLR(opt, [LinearLR(opt, start_factor=0.01, total_iters=WARMUP_EPOCHS),
                           CosineAnnealingLR(opt, T_max=max(1, EPOCHS - WARMUP_EPOCHS))], milestones=[WARMUP_EPOCHS])
best_val, best_state, last_val = -1.0, None, None
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        opt.zero_grad(); loss = crit(model(x.to(dev)), y.to(dev)); loss.backward(); opt.step(); ema.update(model)
    sched.step()
    raw, em = acc(model, val_loader), acc(ema.module, val_loader)
    chosen, state = (em, ema.module.state_dict()) if em >= raw else (raw, model.state_dict())
    last_val = chosen
    if chosen > best_val:
        best_val, best_state = chosen, copy.deepcopy(state)
model.load_state_dict(best_state)
r = {"model": name, "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
     "method": "fine-tuned, notebook recipe (best epoch by valid)",
     "valid_acc": round(best_val, 2), "valid_acc_last_epoch": round(last_val, 2),
     "test_acc": round(acc(model, test_loader), 2), "seconds": round(time.time() - t0, 1)}
results.append(r); print(json.dumps(r), flush=True)
json.dump(results, open(os.path.join(os.path.dirname(ROOT.rstrip("/")), "cls_bench.json"), "w"), indent=1)
print("done", flush=True)
