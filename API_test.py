import argparse
import os
import platform
import sys
from pathlib import Path
import time
import math
import torch
from flask import Flask, request, jsonify
import shutil

# Adding YOLOv5 root directory to the system path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Import YOLOv5 dependencies
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

app = Flask(__name__)

opt = { 
    "weights" : "weights/yolov9-e.pt",
    "imgsz" : (640, 640),
    "conf_thres": 0.25,
    "iou_thres" : 0.45,
    "classes": None,
    "device": 'cpu',
    "half": False,
    "vid_stride": 1, #video frame-rate stride
    "augment" : False,
    "visualize" : False,
    "agnostic_nms" : False,
    "max_det" : 1000,
    "name": 'exp',
    "exist_ok": False,
    "save_txt": False,
}

def classNames():
    cocoClassNames = ['cardboard', 'clothes', 'food', 'glass', 'metal', 'paper', 'plastic', 'plastic bag', 'trash', 'wood', 'shoe']
    return cocoClassNames

def colorLabels(classid):
    if classid == 0:
        color = (85, 45, 255)
    elif classid == 1: 
        color = (222, 82, 175)
    elif classid == 2:
        color = (0, 204, 255)
    elif classid == 3:
        color = (0, 149, 255)
    else:
        color = (200, 100, 0)
    return tuple(color)

@smart_inference_mode()
def objectDetection(source):
    frameCount = 0
    ptime = 0
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path('runs/detect') / opt["name"], exist_ok=opt["exist_ok"])  # increment run
    (save_dir / 'labels' if opt["save_txt"] else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(opt["device"])
    model = DetectMultiBackend(opt["weights"], device=device, dnn=False, data=ROOT / 'data/coco.yaml', fp16=opt["half"])

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(opt["imgsz"], s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=opt["vid_stride"])
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, stride=stride, auto=pt, vid_stride=opt["vid_stride"])
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        frameCount += 1
        print("Frame No :", frameCount)
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = opt["visualize"]
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=opt["augment"], visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, opt["conf_thres"], opt["iou_thres"], opt["classes"], opt["agnostic_nms"], max_det=opt["max_det"])
        totalDetections = 0
        frameRate = 0

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1   
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    totalDetections += int(n)

                # Write results 
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    print("x1, y1, x2, y2", x1, y1, x2, y2)
                    cat = int(cls)
                    if cat >= len(classNames()):  # Ensure cat index is within bounds
                        print(f"Warning: Class index {cat} is out of range for classNames")
                        continue
                    color = colorLabels(cat)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), color, 3)
                    className = classNames()
                    name = className[cat]
                    conf = math.ceil((conf * 100)) / 100
                    label = f'{name} {conf}'
                    textSize = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
                    c2 = x1 + textSize[0], y1 - textSize[1] - 3
                    cv2.rectangle(im0, (x1, y1), c2, color, -1)
                    cv2.putText(im0, label, (x1, y1 - 2), cv2.FONT_HERSHEY_PLAIN, 0.5, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                ctime = time.time()
                frameRate = 1 / (ctime - ptime)
                ptime = ctime

                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # video or stream
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        yield im0, frameRate, im0.shape, totalDetections

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_location = os.path.join('temp', file.filename)
    if not os.path.exists('temp'):
        os.makedirs('temp')

    file.save(file_location)

    results = []
    for im0, frameRate, dimensions, totalDetections in objectDetection(file_location):
        result = {
            'frame_rate': frameRate,
            'dimensions': list(dimensions),
            'total_detections': totalDetections,
            'image_path': f'results/{file.filename}'  # Path where the resulting image/video is saved
        }
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
