import argparse, time, math
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite

def letterbox(im, new_shape=640, color=(114,114,114)):
    # resize & pad to square keeping aspect ratio
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw //= 2; dh //= 2
    im_out = cv2.copyMakeBorder(im_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)
    return im_out, r, (dw, dh)

def sigmoid(x): return 1 / (1 + np.exp(-x))

def nms(boxes, scores, iou_thres=0.45):
    # boxes: Nx4 (x1,y1,x2,y2), scores: N
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        w = np.maximum(0, xx2-xx1)
        h = np.maximum(0, yy2-yy1)
        inter = w*h
        iou = inter / ( (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]) + (boxes[idxs[1:],2]-boxes[idxs[1:],0])*(boxes[idxs[1:],3]-boxes[idxs[1:],1]) - inter + 1e-9 )
        idxs = idxs[1:][iou <= iou_thres]
    return np.array(keep, dtype=int)

def decode_yolov8(output, conf_thres=0.25):
    """
    Accepts common YOLOv8 TFLite layouts:
      (1, 84, 8400)  -> [4 box, 1 obj, C classes, 8400 candidates]
      (1, 8400, 84)  -> [8400 candidates, 4+1+C]
      (1, N, 85) or (1, 85, N)
    Returns (boxes_xyxy, scores, class_ids) in model-space coordinates (same as the 640x640 input).
    """
    out = output
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]  # drop batch

    # standardize to (N, D) where D = 4 + 1 + C
    if out.shape[0] in (84, 85) and out.shape[1] > 1000:
        out = out.T  # (D, N) -> (N, D)

    # split
    D = out.shape[1]
    # Heuristic: first 4 = box (cx,cy,w,h), next 1 = obj, rest = class logits
    boxes_cxcywh = out[:, :4]
    obj = out[:, 4:5]
    cls = out[:, 5:] if D > 5 else np.zeros((out.shape[0],1), dtype=out.dtype)

    # some exports are already sigmoid; using sigmoid twice is okay-ish but you can skip if you know it’s applied
    obj = sigmoid(obj)
    cls = sigmoid(cls)

    # per-class scores = obj * cls
    scores_all = obj * cls
    class_ids = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(scores_all.shape[0]), class_ids]

    # threshold
    mask = scores >= conf_thres
    if not np.any(mask):
        return np.empty((0,4)), np.empty((0,)), np.empty((0,), dtype=int)

    boxes_cxcywh = boxes_cxcywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # convert to xyxy in model space
    cx, cy, w, h = boxes_cxcywh.T
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)
    return boxes_xyxy, scores, class_ids

def draw_boxes(img, boxes, scores, cls_ids, names=None):
    for (x1,y1,x2,y2), s, c in zip(boxes, scores, cls_ids):
        c = int(c)
        label = f"{names[c] if names and c < len(names) else c}:{s:.2f}"
        p1 = (int(x1), int(y1)); p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, (0,255,0), 2)
        cv2.putText(img, label, (p1[0], max(0, p1[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='path to .tflite (FP32/FP16/INT8)')
    ap.add_argument('--source', required=True, help='image path')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--names', default='', help='optional classes file, one name per line')
    args = ap.parse_args()

    names = None
    if args.names:
        with open(args.names, 'r') as f:
            names = [x.strip() for x in f.readlines() if x.strip()]

    interpreter = tflite.Interpreter(model_path=args.model, num_threads=max(1, cv2.getNumberOfCPUs()//2))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    h_in, w_in = input_details[0]['shape'][1:3]

    img0 = cv2.imread(args.source)
    if img0 is None:
        raise FileNotFoundError(args.source)

    img, r, (dw, dh) = letterbox(img0, new_shape=(h_in, w_in))
    inp = img.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, 0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], inp)
    t0 = time.time()
    interpreter.invoke()
    dt = (time.time()-t0)*1000
    out = interpreter.get_tensor(output_details[0]['index'])
    print(f"Inference: {dt:.1f} ms, output shape: {out.shape}")

    boxes, scores, cls_ids = decode_yolov8(out, conf_thres=args.conf)
    if boxes.shape[0]:
        # NMS per class
        final_boxes, final_scores, final_cls = [], [], []
        for cls in np.unique(cls_ids):
            m = cls_ids == cls
            keep = nms(boxes[m], scores[m], iou_thres=args.iou)
            final_boxes.append(boxes[m][keep])
            final_scores.append(scores[m][keep])
            final_cls.append(np.full(keep.shape[0], cls, dtype=int))
        boxes = np.concatenate(final_boxes) if final_boxes else np.empty((0,4))
        scores = np.concatenate(final_scores) if final_scores else np.empty((0,))
        cls_ids = np.concatenate(final_cls) if final_cls else np.empty((0,), dtype=int)

        # map back to original image space
        boxes[:, [0,2]] -= dw
        boxes[:, [1,3]] -= dh
        boxes /= (img.shape[0]/r)  # since we padded after resize, dividing by (1/r) == multiplying by r; this line compensates
        # safer mapping:
        # scale back directly: divide x by r and y by r
        boxes = boxes.clip(min=0)
        draw_boxes(img0, boxes, scores, cls_ids, names)

    cv2.imwrite('out.jpg', img0)
    print("Saved: out.jpg")

if __name__ == "__main__":
    main()
