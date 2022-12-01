# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import cv2
import numpy as np
import onnxruntime as ort
from ocr.config import *
try:
    from openvino.runtime import Core
except Exception as e:
    print("The current platform does not support this library !")
    pass

def multiclass_nms(bboxs, num_classes, match_threshold=0.6, match_metric='ios'):
    '''
    多目标nms，用于切图拼图合并
    '''
    final_boxes = []
    for c in range(num_classes):
        idxs = bboxs[:, 0] == c
        if np.count_nonzero(idxs) == 0: continue
        r = nms(bboxs[idxs, 1:], match_threshold, match_metric)
        final_boxes.append(np.concatenate([np.full((r.shape[0], 1), c), r], 1))
    return final_boxes


def nms(dets, match_threshold=0.6, match_metric='iou'):
    """ Apply NMS to avoid detecting too many overlapping bounding boxes.
        Args:
            dets: shape [N, 5], [score, x1, y1, x2, y2]
            match_metric: 'iou' or 'ios'
            match_threshold: overlap thresh for match metric.
    """
    if dets.shape[0] == 0:
        return dets[[], :]
    scores = dets[:, 0]
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int)

    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            if match_metric == 'iou':
                union = iarea + areas[j] - inter
                match_value = inter / union
            elif match_metric == 'ios':
                smaller = min(iarea, areas[j])
                match_value = inter / smaller
            else:
                raise ValueError()
            if match_value >= match_threshold:
                suppressed[j] = 1
    keep = np.where(suppressed == 0)[0]
    dets = dets[keep, :]
    return dets

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]      # 取剩余的检测框
        rest_boxes = boxes[indexes, :]

        # 根据边界过滤掉一些框，减少IOU的判断时间
        r_thresh = 1/2
        br = rest_boxes[...,0] > (current_box[...,0] + current_box[...,2])*(1-r_thresh) # 右侧框
        bl = rest_boxes[...,2] < (current_box[...,0] + current_box[...,2])*r_thresh     # 左侧框
        bt = rest_boxes[...,1] > (current_box[...,1] + current_box[...,3])*(1-r_thresh) # 下侧框
        bb = rest_boxes[...,3] < (current_box[...,1] + current_box[...,3])*r_thresh     # 上侧框
        mask = ( br | bl | bt | bb )
        good_indexes = indexes[mask]    # 已经符合条件的index
        rest_indexes = indexes[(1 - mask).astype(np.bool)]
        rest_boxes = boxes[rest_indexes]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        # indexes = rest_indexes[iou <= iou_threshold]
        indexes = np.concatenate([good_indexes, rest_indexes[iou <= iou_threshold]])

    return box_scores[picked, :]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.
    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.
    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


class PicoDetNMS(object):
    """
    Args:
        input_shape (int): network input image size
        scale_factor (float): scale factor of ori image
    """

    def __init__(self,
                 score_threshold=0.2,
                 nms_threshold=0.7,
                 nms_top_k=1000,
                 keep_top_k=300):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k

    def __call__(self, decode_boxes, select_scores):
        batch_size = 1
        out_boxes_list = []
        for batch_id in range(batch_size):
            # nms
            bboxes = np.concatenate(decode_boxes, axis=0)
            confidences = np.concatenate(select_scores, axis=0)
            picked_box_probs = []
            picked_labels = []
            for class_index in range(0, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > self.score_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = bboxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1)
                box_probs = hard_nms(
                    box_probs,
                    iou_threshold=self.nms_threshold,
                    top_k=self.keep_top_k,
                    candidate_size= 1000 )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])

            if len(picked_box_probs) == 0:
                out_boxes_list.append(np.empty((0, 4)))

            else:
                picked_box_probs = np.concatenate(picked_box_probs)
                # clas score box
                out_boxes_list.append(
                    np.concatenate(
                        [
                            np.expand_dims(
                                np.array(picked_labels),
                                axis=-1), np.expand_dims(
                                    picked_box_probs[:, 4], axis=-1),
                            picked_box_probs[:, :4]
                        ],
                        axis=1))

        out_boxes_list = np.concatenate(out_boxes_list, axis=0)
        return out_boxes_list


def merge_boxes(boxes):
    '''
    合并检测框box
    args:
        result_boxes: 检测网络输出的已经用置信度过滤的检测框信息
    return:
        result_boxes: 过滤后的检测框
    '''
    finall_boxes = []
    h_delta = 30
    boxes_t_ = boxes[:,3]
    boxes_b_ = boxes[:,5]
    boxes_h_ = boxes_b_ - boxes_t_ + 1

    def merge(box):
        class_id = box[0]
        l,r = box[2],box[4]
        t,b = box[3],box[5]
        h = b-t + 1
        w = r-l + 1        
        idxs_samel = np.where((abs(t - boxes_t_) < h_delta) & (abs(h - boxes_h_) < h_delta)) 
        # 筛选出高度在差不多位置的
        rest_boxes = boxes[idxs_samel]
        boxes_l = rest_boxes[:,2]
        boxes_r = rest_boxes[:,4]
        boxes_t = rest_boxes[:,3]
        boxes_b = rest_boxes[:,5]
        boxes_h = boxes_b - boxes_t + 1
        boxes_w = boxes_r - boxes_l + 1                               

        # 计算IOU或IOS
        xmins = np.stack((np.tile(l,len(rest_boxes)),boxes_l)).max(0)
        ymins = np.stack((np.tile(t,len(rest_boxes)),boxes_t)).max(0)
        xmaxs = np.stack((np.tile(r,len(rest_boxes)),boxes_r)).min(0)
        ymaxs = np.stack((np.tile(b,len(rest_boxes)),boxes_b)).min(0)

        ws = xmaxs-xmins+1
        hs = ymaxs-ymins+1  
        ws[ws < 0] = 0
        hs[hs < 0] = 0  
        inter_areas = ws * hs
        box_area = np.tile(w*h,len(rest_boxes))
        boxes_areas = boxes_h * boxes_w
        ious = inter_areas / (box_area + boxes_areas - inter_areas)
        idxs = set(np.where((ious > 0) & (ious < 1))[0])

        new_boxes = []
        if len(idxs) == 0:                                  # 没有同行左或右相邻的框
            new_box = tuple(list(box))
        else:
            tmp_boxes = np.concatenate([box[np.newaxis],rest_boxes[np.array(list(idxs))]])
            xmin = tmp_boxes[:,2].min()
            ymin = tmp_boxes[:,3].min()
            xmax = tmp_boxes[:,4].max()
            ymax = tmp_boxes[:,5].max() 
            conf = tmp_boxes[:,1].max()                             # 置信度取最大的
            class_id = tmp_boxes[:,0][np.argmax(tmp_boxes[:,1])]    # 类比取置信度最大的那个框的类别
            new_box = (class_id,conf,xmin,ymin,xmax,ymax)
        return new_box
    tr = time.time()
    # pool = ThreadPool(processes = 10)
    # new_boxes = pool.map(merge, boxes)
    # pool.close()
    # pool.join()
    new_boxes = list(map(merge,boxes))
    print("pool cost:",time.time()-tr)
    new_boxes = np.array([list(box) for box in set(new_boxes)])
    return new_boxes

def merge_boxes2(ori_boxes):
    '''
    合并检测框
    args:
        ori_boxes: 检测框结果
    return:
        new_boxes: 合并后的检测框
    '''
    new_boxes = []
    delta = 5
    h_delta = 10
    wh_r_th = 1.5
    merged_boxes_array = None
    merged_boxes = []
    mask = np.ones(len(ori_boxes)).astype(np.bool)
    for i in range(len(ori_boxes)):
        mask[i] = False
        box = ori_boxes[i]
        l,r,t,b = box[2],box[4],box[3],box[5]
        w,h = r-l+1, b-t+1
        c = box[0]

        # 加上新合并完的框，也参与判断
        boxes = ori_boxes[mask]
        if merged_boxes:
            boxes = np.concatenate((boxes,merged_boxes_array))
        boxes_c = boxes[:,0]
        boxes_l = boxes[:,2]
        boxes_r = boxes[:,4]
        boxes_t = boxes[:,3]
        boxes_b = boxes[:,5]
        boxes_h = boxes_b - boxes_t + 1
        boxes_w = boxes_r - boxes_l + 1
        
        idxs_r = set(np.where((boxes_l < r) & (boxes_l > l))[0])                                                        # 右侧相邻框
        idxs_l = set(np.where((boxes_r > l) & (boxes_r < r))[0])                                                        # 左侧相邻框
        idxs_samel = set(np.where(((abs(b - boxes_b) < h_delta) | (abs(t - boxes_t) < h_delta)) 
                        & (abs(h - boxes_h) < h_delta))[0])                                                             # 同行框
        idxs_cover = set(np.where((l>=boxes_l) & (r <= boxes_r))[0])                                                    # 覆盖框
        # idxs_text = set(np.where( boxes_whr > wh_r_th)[0])                                                            # 宽高比大于阈值，认为是文字行
        idxs_samec = set(np.where(c==boxes_c)[0])                                                                          # 类别相同

        idxs = (idxs_l | idxs_r | idxs_cover)  & idxs_samel & idxs_samec
        if len(idxs) == 0:                                  # 没有同行左或右相邻的框
            new_boxes.append(tuple(list(box))) 
        else:
            tmp_boxes = np.concatenate((boxes[np.array(list(idxs))],box[np.newaxis]))
            for idx in idxs:
                if tuple(boxes[idx,:]) in new_boxes:     # 如果在合并后的框集合中
                    new_boxes.remove(tuple(boxes[idx]))
            xmin = tmp_boxes[:,2].min()
            ymin = tmp_boxes[:,3].min()
            xmax = tmp_boxes[:,4].max()
            ymax = tmp_boxes[:,5].max()
            conf = tmp_boxes[:,1].max()
            class_id = tmp_boxes[:,0][np.argmax(conf)]
            new_boxes.append((class_id,conf,xmin,ymin,xmax,ymax))
            merged_boxes.append((class_id,conf,xmin,ymin,xmax,ymax))
            merged_boxes_array = np.array([list(box) for box in set(merged_boxes)])
    new_boxes = np.array([list(box) for box in set(new_boxes)])
    return new_boxes

class PicoDet():
    def __init__(self,
                 model_pb_path,
                 label_path,
                 prob_threshold=0.4,
                 nms_threshold=0.3):
        # 读取类别文件，获取类别列表
        self.classes = list(
            map(lambda x: x.strip(), open(label_path, 'r').readlines()))
        self.num_classes = len(self.classes)
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        # self.nms = multiclass_nms()
        self.nms = PicoDetNMS(score_threshold=self.prob_threshold,
                            nms_threshold=self.nms_threshold,
                            nms_top_k=1000,
                            keep_top_k=300)

        # 均值、标准差，用于归一化
        self.mean = np.array(
            [103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(
            [57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)

        # 初始化onnx推理
        self.net = self.onnx_init(model_pb_path)
        # 初始化openvino推理
        self.net_vino = self.openvino_init(model_pb_path)        
        # 根据网络结构，获取输入名称和尺寸
        inputs_name = [a.name for a in self.net.get_inputs()]
        inputs_shape = {
            k: v.shape
            for k, v in zip(inputs_name, self.net.get_inputs())
        }
        self.input_shape = inputs_shape['image'][2:]


    def onnx_init(self,model_path):
        '''
        onnx模型初始化
        '''
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            net = ort.InferenceSession(model_path, so)
        except Exception as ex:
            print(ex)
            net = None
        return net

    def openvino_init(self,model_path):
        '''
        openvino模型初始化
        '''
        try:
            ie = Core()
            net = ie.read_model(model_path)
            input_layer = net.input(0)
            input_shape = input_layer.partial_shape
            # 输入batch改为1
            input_shape[0] = 1
            net.reshape({input_layer: input_shape})
            compiled_model = ie.compile_model(net, 'CPU')
        except Exception as ex:
            print(ex)
            compiled_model = None
        return compiled_model

    def _normalize(self, img):
        '''
        图像归一化
        '''
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean / 255.0) / (self.std / 255.0)
        return img

    def resize_image(self, srcimg, keep_ratio=False):
        '''
        图像缩放

        Args:
            srcimg 原始输入图片
        Returns:
            keep_ratio 是否保持原图宽高比
        '''
        top, left, newh, neww = 0, 0, self.input_shape[0], self.input_shape[1]
        origin_shape = srcimg.shape[:2]
        im_scale_y = newh / float(origin_shape[0])
        im_scale_x = neww / float(origin_shape[1])
        img_shape = np.array([
            [float(self.input_shape[0]), float(self.input_shape[1])]
        ]).astype('float32')
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')
        img = cv2.resize(srcimg,  tuple(self.input_shape), interpolation=2)

        return img, img_shape, scale_factor

    def preprocess(self,srcimg):
        '''
        数据预处理
        '''

        # 缩放到推理尺寸
        img, im_shape, scale_factor = self.resize_image(srcimg)
        # 维度转置+添加维度
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        inputs_dict = {
            'im_shape': im_shape,
            'image': blob,
            'scale_factor': scale_factor
        }
        inputs_name = [a.name for a in self.net.get_inputs()]
        net_inputs = {k: inputs_dict[k] for k in inputs_name}
        return net_inputs
    
    def det_onnx(self, srcimg):
        '''
        目标检测模型推理接口

        Args:
            srcimg 原始数据
        Returns:
            result_list 检测结果列表
        '''
        net_inputs = self.preprocess(srcimg)

        result_list = []
        try:
            t1 = time.time()
            outs = self.net.run(None, net_inputs)
            t2 = time.time()
            print("infer cost:",t2-t1)
            outs = np.array(outs[0])
            # 过滤检测结果：置信度大于阈值，索引大于-1
            expect_boxes = (outs[:, 1] > self.prob_threshold) & (outs[:, 0] > -1)
            result_boxes = outs[expect_boxes, :]
            t_m = time.time()
            result_boxes = merge_boxes2(result_boxes)
            print("merge cost:",time.time()-t_m)
            
            for i in range(result_boxes.shape[0]):
                class_id, conf = int(result_boxes[i, 0]), result_boxes[i, 1]
                class_name = self.classes[class_id]
                xmin, ymin, xmax, ymax = int(result_boxes[i, 2]), int(result_boxes[
                    i, 3]), int(result_boxes[i, 4]), int(result_boxes[i, 5])
                width = (xmax-xmin)
                height =(ymax-ymin)
                if(width * 1.0 / height >= 1.5 and conf>=0.4 and width>2 and height>2):
                    result = {"classid":class_id,
                            "classname":class_name,
                            "confidence":conf,
                            "box":[xmin,ymin,xmax,ymax]}
                    result_list.append(result)
        except Exception as e:
            print(e)

        return result_list

    def vino_preprocess(self,img):
        n,c, H,W = list(self.net_vino.inputs[0].shape)
        im_scale_y = H / float(img.shape[0])
        im_scale_x = W / float(img.shape[1])
        scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')
        img = cv2.resize(img, (W, H), interpolation=2)
        # img = img[:,:,::-1]
        # img = self._normalize(img)
        # # 转回RGB
        # img = img[:,:,::-1]
        # 维度转置+添加维度
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32)
        return blob,scale_factor

    def det_vino(self,srcimg):
        '''
        目标检测模型推理接口(基于openvino库推理)
        '''
        result_list = []
        img,scale_factor = self.vino_preprocess(srcimg)
        try:
            t1 = time.time()
            t = 1
            for i in range(t):
                output = self.net_vino.infer_new_request({0: img,1:scale_factor}) # 这里要加上缩放因子
            t2 = time.time()
            print("infer cost:",(t2-t1)/t)
            outs = list(output.values())

            # nms
            # num_outs = int(len(outs) / 2)
            # decode_boxes = []
            # select_scores = []
            # for out_idx in range(num_outs):
            #     decode_boxes.append(outs[out_idx])
            #     select_scores.append(outs[out_idx + num_outs])
            # outs = self.nms(decode_boxes, select_scores)
            # t3 = time.time()
            # print("nms cost:",t3-t2)
            
            # 过滤检测结果：置信度大于阈值，索引大于-1
            outs = np.array(outs[0])
            expect_boxes = (outs[:, 1] > self.prob_threshold) & (outs[:, 0] > -1)
            result_boxes = outs[expect_boxes, :]
            
            # 合并检测框
            tm = time.time()
            result_boxes = merge_boxes2(result_boxes)
            print("merge cost:",time.time()-tm)
            
            for i in range(result_boxes.shape[0]):
                class_id, conf = int(result_boxes[i, 0]), result_boxes[i, 1]
                class_name = self.classes[class_id]
                xmin, ymin, xmax, ymax = int(result_boxes[i, 2]), int(result_boxes[
                    i, 3]), int(result_boxes[i, 4]), int(result_boxes[i, 5])
                width = (xmax-xmin)
                height =(ymax-ymin)
                if(width * 1.0 / height >= 1.5 and conf>=0.4 and width>2 and height>2):
                    result = {"classid":class_id,
                            "classname":class_name,
                            "confidence":conf,
                            "box":[xmin,ymin,xmax,ymax]}
                    result_list.append(result)
        except Exception as e:
            print(e)
        return result_list

    def infer(self,img):
        # self.net_vino = None
        if self.net_vino is None:
            print("infer by onnx ...")
            det_results = self.det_onnx(img)
        else:
            print("infer by openvino ...")
            det_results = self.det_vino(img)
            #det_results = self.det_onnx(img)
        return det_results
