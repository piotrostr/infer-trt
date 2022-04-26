import pycuda.autoinit
import cv2
import numpy as np
import tensorrt as trt

from src.trt_model import TensorRTModel


class YOLO(TensorRTModel):

    INPUT_WIDTH = 416
    INPUT_HEIGHT = 416
    CONF_THRESH = 0.8
    NMS_THRESH = 0.5

    def __init__(self, logger):
        super().__init__(logger)
        self.img_shape = None

    def get_engine(self):
        with trt.Runtime(self.TRT_LOGGER) as runtime:
            with open("models/yolo.engine", "rb") as f:
                return runtime.deserialize_cuda_engine(f.read())

    def preprocess(self, img: np.ndarray):
        self.img_shape = img.shape
        img = cv2.resize(img, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)) / 255.0
        img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return img

    def postprocess(self, out):
        out[0] = out[0].reshape(1, -1, 1, 4)
        out[1] = out[1].reshape(1, -1, 4)
        box_array, confs = out
        if type(box_array).__name__ != "ndarray":
            box_array = box_array.cpu().detach().numpy()
            confs = confs.cpu().detach().numpy()
        num_classes = confs.shape[2]
        box_array = box_array[:, :, 0]
        max_conf = np.max(confs, axis=2)
        max_id = np.argmax(confs, axis=2)
        boxes_batch = []
        for i in range(box_array.shape[0]):
            argwhere = max_conf[i] > self.CONF_THRESH
            l_box_array = box_array[i, argwhere, :]
            l_max_conf = max_conf[i, argwhere]
            l_max_id = max_id[i, argwhere]
            bboxes = []
            for j in range(num_classes):
                cls_argwhere = l_max_id == j
                ll_box_array = l_box_array[cls_argwhere, :]
                ll_max_conf = l_max_conf[cls_argwhere]
                ll_max_id = l_max_id[cls_argwhere]
                keep = self.nms_cpu(ll_box_array, ll_max_conf, self.NMS_THRESH)
                if keep.size > 0:
                    ll_box_array = ll_box_array[keep, :]
                    ll_max_conf = ll_max_conf[keep]
                    ll_max_id = ll_max_id[keep]
                    for k in range(ll_box_array.shape[0]):
                        bboxes.append(
                            [
                                ll_box_array[k, 0],
                                ll_box_array[k, 1],
                                ll_box_array[k, 2],
                                ll_box_array[k, 3],
                                ll_max_conf[k],
                                ll_max_conf[k],
                                ll_max_id[k],
                            ]
                        )
            boxes_batch.append(bboxes)
        (boxes,) = boxes_batch
        h, w, *_ = self.img_shape
        labels, confidences, bboxes = [], [], []
        for box in boxes:
            x1, y1, x2, y2 = (
                int(box[0] * w),
                int(box[1] * h),
                int(box[2] * w),
                int(box[3] * h),
            )
            labels.append(box[6])
            confidences.append(box[5])
            bboxes.append([x1, y1, x2, y2])
        return labels, confidences, bboxes

    @staticmethod
    def nms_cpu(boxes, confs, nms_thresh=0.5, min_mode=False):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = confs.argsort()[::-1]
        keep = []
        while order.size > 0:
            idx_self = order[0]
            idx_other = order[1:]
            keep.append(idx_self)
            xx1 = np.maximum(x1[idx_self], x1[idx_other])
            yy1 = np.maximum(y1[idx_self], y1[idx_other])
            xx2 = np.minimum(x2[idx_self], x2[idx_other])
            yy2 = np.minimum(y2[idx_self], y2[idx_other])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            if min_mode:
                over = inter / np.minimum(areas[order[0]], areas[order[1:]])
            else:
                over = inter / (areas[order[0]] + areas[order[1:]] - inter)
            inds = np.where(over <= nms_thresh)[0]
            order = order[inds + 1]
        return np.array(keep)

