import time
import cv2
import numpy as np
import logging

from ocr.crnn import pp_pred
import ocr.config as config
from ocr.det.picodet import PicoDet

#log = logging.getLogger()
#log.setLevel("DEBUG")

'''
金蝶财务软件文字和图标识别
实现思路：
    OCR得到所有文字的坐标（去除置信度很低的）
    根据文字位置mask，对剩余的非文字部分，做sobel,轮廓检测，记录位置
'''

# 初始化模型
# OCR识别
ocr_predict = pp_pred.PPrecPredictor(config.model_path,
                                 config.infer_h,
                                 config.batch,
                                 config.keys_txt_path,
                                 config.in_names,
                                 config.out_names)

det_net = PicoDet(model_pb_path = config.det_model_path,
                label_path = config.label_path,
                prob_threshold = config.confThreshold,
                nms_threshold =config.nmsThreshold)


def edge_detect(img,thresh = 10):
    '''
    边缘检测
    输入：
        img:        灰度图像
        thresh:     二值化阈值
    返回:
        edges:      二值化后的边缘图
    '''
    # 因为是从右到左做减法，因此有可能得到负值，如果输出为uint8类型，则会只保留灰度差为正的那部分，所以就只有右边缘会保留下来
    
    # grad_X = cv2.Sobel(img,cv2.CV_64F,1,0,3) # cv2.CV_64F
    # grad_Y = cv2.Sobel(img,cv2.CV_64F,0,1,3)

    # Robert算子边缘检测
    kernelx = np.array([[-1,0],[0,1]], dtype=int)
    kernely = np.array([[0,-1],[1,0]], dtype=int)
    grad_X = cv2.filter2D(img, cv2.CV_16S, kernelx)
    grad_Y = cv2.filter2D(img, cv2.CV_16S, kernely)

    grad_X = cv2.convertScaleAbs(grad_X)      
    grad_Y = cv2.convertScaleAbs(grad_Y)
    
    # #求梯度图像
    grad = cv2.addWeighted(grad_X,0.5,grad_Y,0.5,0)
    edges = cv2.threshold(grad,thresh,255,cv2.THRESH_BINARY)[1]/255

    # edges = cv2.Canny(img,10,200)

    return edges.astype(np.uint8)

def remove_single_line(mask,line_idx,direction = 0,len_th = 50):
    '''
    去除某一行或列中的长线段
    '''
    h,w = mask.shape
    idxs = []
    if direction == 0:      # 竖线
        pos = np.where(mask[:,line_idx] > 0)[0]
    else:
        pos = np.where(mask[line_idx,:] > 0)[0]
    if len(pos) == 0:
        return mask
    sub = pos[1:] - pos[0:len(pos)-1]
    seg = np.where(sub > 1)[0]
    seg_len = len(seg)
    starts =  [0] + list(seg + 1)
    ends = list(seg) + [len(pos)-1]
    idxs = [[pos[starts[idx]],pos[ends[idx]]] for idx in range(len(starts))]
    idxs = sum(idxs,[])

    for i in range(0,len(idxs),2):
        seg_len = idxs[i+1]-idxs[i]
        if  seg_len > len_th:
            if direction == 0 :
                mask[idxs[i]:idxs[i+1],line_idx] = 0
                # 判断左右两列是否也是线段
                if idxs[i] > 0 and idxs[i+1] < h-1 and mask[idxs[i]:idxs[i+1],line_idx-1].sum() == seg_len and mask[idxs[i]-1,line_idx-1] == 0 and mask[idxs[i+1]+1,line_idx-1] == 0:
                    mask[idxs[i]:idxs[i+1],line_idx-1] = 0
                if idxs[i] > 0 and idxs[i+1] < h-1 and mask[idxs[i]:idxs[i+1],line_idx+1].sum() == seg_len and mask[idxs[i]-1,line_idx+1] == 0 and mask[idxs[i+1]+1,line_idx+1] == 0:
                    mask[idxs[i]:idxs[i+1],line_idx+1] = 0
            else:
                mask[line_idx,idxs[i]:idxs[i+1]] = 0
                # 判断上下两行是否也是线段
                if idxs[i] > 0 and idxs[i+1] < w-1 and mask[line_idx-1,idxs[i]:idxs[i+1]].sum() == seg_len and mask[line_idx-1, idxs[i]-1] == 0 and mask[line_idx-1, idxs[i+1]+1] == 0:
                    mask[line_idx-1, idxs[i]:idxs[i+1]] = 0
                if idxs[i] > 0 and idxs[i+1] < w-1 and mask[line_idx+1,idxs[i]:idxs[i+1]].sum() == seg_len and mask[line_idx+1, idxs[i]-1] == 0 and mask[line_idx+1, idxs[i+1]+1] == 0:
                    mask[line_idx+1, idxs[i]:idxs[i+1]] = 0
    return mask

def remove_line(mask):
    '''
    删除横竖线,使用numpy数组判断
    '''
    h,w = mask.shape
    # 去除竖线
    [remove_single_line(mask,w_,0) for w_ in range(w)]
    # 去除横线
    [remove_single_line(mask,h_,1) for h_ in range(h)]
    return mask

def remove_line2(img,edges):
    '''
    删除横竖线,使用opencv的霍夫直线检测
    '''
    h,w = edges.shape
    draw_img = img.copy()
    # minLineLength = 20
    # maxLineGap = 5
    # lines = cv2.HoughLinesP(edges, 0.5, np.pi / 2, 10 ,minLineLength,maxLineGap)      # 概率直线检测
    # min_line_len = 20
    # straight_lines = []
    
    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     # 过滤斜线，长度太短的线
    #     line_len = 0
    #     if x1==x2 and abs(y1-y2) > min_line_len:
    #         edges[min(y1,y2):max(y1,y2),x1] = 0 
    #         line_len = abs(y1-y2)
    #     elif y1 == y2 and abs(x1-x2) > min_line_len:
    #         edges[y1,min(x1,x2):max(x1,x2)] = 0 
    #         line_len = abs(x1-x2)
    #     if x1 !=x2 and y1!=y2 or line_len < min_line_len:
    #         continue  
    #     if line_len > 0:
    #         straight_lines.append(line[0])
    #     cv2.line(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    edges[:,np.sum(edges,axis=0) == h] = 0
    edges[np.sum(edges,axis=1) == w,:] = 0

    lines  =  cv2.HoughLines(edges,1,np.pi/2,50)
    L = 1500
    min_len = 50
    for line in  lines:
        rho,theta = line[0]
        # 强制转成横和竖线
        a  =  0 if theta > 0 else 1
        b  =  1 if theta > 0 else 0
        x0  =  a*rho
        y0  =  b*rho
        x1  = min(w-1, max(0,int(x0  +  L*(-b))))
        y1  = min(h-1, max(0,int(y0  +  L*(a))))
        x2  = min(w-1, max(0,int(x0  -  L*(-b))))
        y2  = min(h-1, max(0,int(y0  -  L*(a))))
        # logging.debug( (x1, y1), (x2, y2))
        cv2.line(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        if theta == 0: # 竖直
            edges = remove_single_line(edges,x1,0,min_len)
        else:
            edges = remove_single_line(edges,y1,1,min_len)

    return edges

def dilate(edge,direc = 0,delta=1):
    '''
    上下各膨胀delta个像素
    '''
    idxs = np.where(edge > 0)
    h,w = edge.shape
    ys = idxs[0]
    xs = idxs[1] 
    if direc == 0:                       # 上下
        ys_up = ys -delta
        ys_down = ys + delta 
        ys_up[ys_up < 0] = 0
        ys_down[ys_down >= h] = h-1
        edge[(ys_up,xs)] = 1
        edge[(ys_down,xs)] = 1
    else:                                # 左右
        xs_left = xs -delta
        xs_right = xs + delta 
        xs_left[xs_left < 0] = 0
        xs_right[xs_right >= w] = w-1
        edge[(ys,xs_left)] = 1
        edge[(ys,xs_right)] = 1
    return edge

def remove_edge_line(mask,ed=5):
    '''
    去除图像边缘线
    '''
    h,w = mask.shape
    ed = 5
    mask[:ed,:] = 0
    mask[h-ed:,:] = 0
    mask[:,:ed] = 0
    mask[:,w-ed:] = 0
    return mask

def remove_connect_line(mask):
    h,w = mask.shape
    mask_ = mask.copy()
    r = 6
    for h_ in range(h):
        cond1 = h_ > 0 and h_ < h-1 
        if mask_[h_,:].sum() > w - r:
            mask[h_,:] = 0
    for w_ in range(w):
        if mask_[:,w_].sum() > h - r:
            mask[:,w_] = 0
    return mask

def remove_connectRegion(mask_):
    '''
    删除不符合条件的连通域
    '''
    t1 = time.time()
    # 连通域查找
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_, connectivity=8)
    h,w = mask_.shape
    num = 0
    for i in range(1,num_labels):
        # 连通域外接矩形左上角坐标和宽高
        label_x = stats[i][0]
        label_y = stats[i][1]
        label_width = stats[i][2]
        label_height = stats[i][3]
        wh_r = label_width/label_height
        area =  label_width*label_height
        cond1 = area > 30*30 or label_height > h/2 or label_width > w/2 or label_height <=3 and label_width >= 10 or label_width <=3 and label_height >=10
        if not cond1:
            continue
        
        # 连通域外接区域
        label_mask = (labels[label_y:label_y + label_height, label_x:label_x + label_width]  == i).astype(np.uint8)
        # 面积比阈值
        area_rth = 0.3
        # 面积比
        area_r = label_mask.sum() / area
        # 删除框线
        cond2 = area_r < area_rth and (((wh_r > 1.5 or wh_r < 1/1.5) and label_height > 10) or (label_height > 100 and label_width > 100) )
        cond3 = wh_r > 50 and label_height < 10 or wh_r < 0.2 and label_width < 10
        cond4 = area_r > 0.9 and (wh_r > 10 or wh_r < 0.2)
        if cond2 or cond3 or cond4:
            # 将该轮廓置零 
            mask_[label_y:label_y + label_height, label_x:label_x + label_width][labels[label_y:label_y + label_height, label_x:label_x + label_width]==i] = 0        
            num += 1
    # print(time.time()-t1)
    return mask_


def merge_boxes(boxes):
    '''
    合并检测框
    '''
    boxes = np.array(boxes)
    new_boxes = []
    delta = 5
    h_delta = 5
    wh_r_th = 1.5

    boxes_l = boxes[:,0]
    boxes_r = boxes[:,0] + boxes[:,2]
    boxes_t = boxes[:,1]
    boxes_h = boxes[:,3]
    boxes_whr = boxes[:,2]/boxes[:,3]

    for i in range(len(boxes)):
        box = boxes[i]
        l,r = box[0],box[0]+box[2]
        
        idxs_r = set(np.where((boxes_l - r < delta) & (boxes_l > l))[0])                                                        # 右侧相邻框
        idxs_l = set(np.where((l - boxes_r < delta ) & (boxes_r < r))[0])                                                       # 左侧相邻框
        idxs_samel = set(np.where((abs(box[1] - boxes_t) < h_delta) & (abs(box[3] - boxes_h) < h_delta))[0])                    # 同行框
        idxs_text = set(np.where( boxes_whr > wh_r_th)[0])                                                                      # 宽高比大于阈值，认为是文字行

        idxs = (idxs_l | idxs_r) & idxs_samel & idxs_text
        if len(idxs) == 0:                                  # 没有同行左或右相邻的框
            new_boxes.append(tuple(list(box))) 
        else:
            tmp_boxes = boxes[np.array(list(idxs)+[i])]
            x = tmp_boxes[:,0].min()
            y = tmp_boxes[:,1].min()
            bb_w = (tmp_boxes[:,0]+ tmp_boxes[:,2]).max() - x
            bb_h = (tmp_boxes[:,1]+ tmp_boxes[:,3]).max() - y
            new_boxes.append((x,y,bb_w,bb_h))
    new_boxes = list(set(new_boxes))
    return new_boxes


def get_item_boxs(img,r = 1,ksize = 3,close = True,mergebox = False):
    '''
    获取元素框，包括文字和图标
    输入：
        img:    输入图像
        r:      图像缩放比例
        ksize:  闭运算核大小
        close:  是否进行闭运算
    输出:
        boxes:  检测出的所有box

    '''
    t1 = time.time()
    ori_h,ori_w = img.shape[:2]
    draw_img2 = img.copy()
    if r != 1:
        img = cv2.resize(img,(int(ori_w*r),int(ori_h*r)),cv2.INTER_LANCZOS4)
    # 灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 边缘检测
    edges = edge_detect(gray,10)
    t2 = time.time()
    # logging.debug(f"edge_detect cost: {t2-t1}")
   
    # 连通域检测和去除
    edges = remove_connectRegion(edges)
    t3 = time.time()
    # logging.debug(f"remove_connectRegion cost: {t3-t2}")
    
    # 闭运算连接相邻文字区域，减少块数
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(ksize,ksize))
    if close:
        edges = cv2.morphologyEx(edges,op=cv2.MORPH_CLOSE,kernel=kernel)
    t4 = time.time()
    # logging.debug(f"morph close cost: {t4-t3}")
    
    # 查找剩余轮廓
    contours,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t5 = time.time()
    # logging.debug(f"findContours cost: {t5-t4}")

    boxes = []
    # 过滤box
    for contour in contours:
        (x, y, bb_w, bb_h) = cv2.boundingRect(contour)
        hwr_th = 2      # 高宽比阈值
        whr_th = 10     # 宽高比阈值
        area_th = 4*4   # 面积阈值
        if bb_h > 50 or bb_h < 2 or bb_w < 2 or bb_h/bb_w > hwr_th or bb_w * bb_h < area_th or bb_h < 5 and bb_w / bb_h > whr_th:
            continue
        # 映射回原始尺寸
        box = (int(x/r), int(y/r), int(bb_w/r), int(bb_h/r))
        boxes.append(box)
    
    t6 = time.time()
    # logging.debug(f"filter boxes cost: {t6-t5}")

    # 拼接相邻的box
    if mergebox:
        old_boxes = boxes
        for i in range(10):
            boxes = merge_boxes(old_boxes)
            if len(boxes) == len(old_boxes):
                break
            old_boxes = boxes
        t7 = time.time()
        # logging.debug(f"merge boxes cost: {t7-t6}")

    return boxes

def xyxy2xywh(box):
    xmin,ymin,xmax,ymax = box
    x = int(float(xmin))
    y = int(float(ymin))
    w = int(float(xmax-xmin))+1
    h = int(float(ymax-ymin))+1
    box = x,y,w,h
    return box


def ocr_rec(img,slice = False, use_mp = config.use_mp, process_num = config.process_num):
    '''
    页面元素识别
    输入：
        img:            待识别页面图像
        slice:          是否使用自动切图拼图的方式检测,对很小的目标有比较好的效果(模型已合并，设置无效)
        use_mp:         使用多线程
        process_num:    线程数
    返回：
        结果字典: {"texts":[ ((x,y,w,h),'文字内容',conf),
                            ......,],
                   "icos":[(x,y,w,h),
                            ......,]}
    '''
    
    # 获取所有的元素位置（文本+图标）
    t1 = time.time()

    det_results = det_net.infer(img)
    logging.debug(f"det cost: {time.time()-t1}")
    texts = []
    icos = []
    for item in det_results:
        cls_name = item["classname"]
        box = xyxy2xywh(item["box"])
        if cls_name == "text":
            texts.append((box,img))
        else:
            icos.append(box)

    results = []
    # # 调用OCR识别
    t2 = time.time()
    ocr_results = ocr_predict(texts,use_mp, process_num)
    logging.debug(f"OCR num {len(texts)} cost: {time.time()-t2}")

    # # 对结果进行分类,区分文字和图标
    results = {"texts":ocr_results,
               "icos":icos}
    return ocr_results

def py_nms(dets, thresh):
  """Pure Python NMS baseline."""
  #x1、y1、x2、y2、以及score赋值
  # （x1、y1）（x2、y2）为box的左上和右下角标
  x1 = dets[:, 0]
  y1 = dets[:, 1]
  x2 = dets[:, 2]
  y2 = dets[:, 3]
  scores = dets[:, 4]
  #每一个候选框的面积
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  #order是按照score降序排序的
  order = scores.argsort()[::-1]
  # print("order:",order)

  keep = []
  while order.size > 0:
      i = order[0]
      keep.append(i)
      #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
      xx1 = np.maximum(x1[i], x1[order[1:]])
      yy1 = np.maximum(y1[i], y1[order[1:]])
      xx2 = np.minimum(x2[i], x2[order[1:]])
      yy2 = np.minimum(y2[i], y2[order[1:]])
      #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
      w = np.maximum(0.0, xx2 - xx1 + 1)
      h = np.maximum(0.0, yy2 - yy1 + 1)
      inter = w * h
      #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
      ovr = inter / (areas[i] + areas[order[1:]] - inter)
      #找到重叠度不高于阈值的矩形框索引
      inds = np.where(ovr <= thresh)[0]
      # print("inds:",inds)
      #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
      order = order[inds + 1]
  return keep

def get_template(color_image,color_img_temp,template_threshold=0.95):
    '''
    color_image:匹配的原始图（大图）
    color_img_temp:匹配用的模板图（小图）
    template_threshold:模板匹配的置信度（0.1-0.99之间）
    '''
    img_gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
    template_img = cv2.cvtColor(color_img_temp,cv2.COLOR_BGR2GRAY)
    h, w = template_img.shape[:2]
    res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
    
    loc = np.where(res >= template_threshold)#大于模板阈值的目标坐标
    score = res[res >= template_threshold]#大于模板阈值的目标置信度
    #将模板数据坐标进行处理成左上角、右下角的格式
    xmin = np.array(loc[1])
    ymin = np.array(loc[0])
    xmax = xmin+w
    ymax = ymin+h
    xmin = xmin.reshape(-1,1)#变成n行1列维度
    xmax = xmax.reshape(-1,1)#变成n行1列维度
    ymax = ymax.reshape(-1,1)#变成n行1列维度
    ymin = ymin.reshape(-1,1)#变成n行1列维度
    score = score.reshape(-1,1)#变成n行1列维度
    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack(data_hlist)#将xmin、ymin、xmax、yamx、scores按照列进行拼接
    thresh = 0.3#NMS里面的IOU交互比阈值

    keep_dets = py_nms(data_hstack, thresh)
    
    dets = data_hstack[keep_dets]#最终的nms获得的矩形框
    arr = np.array(dets)

    arr = arr[np.argsort(arr[:,1])]
    return arr


    # def get_template(color_image,color_img_temp,conf = 0.7):
    #     '''
    #     img_gray:待检测的灰度图片格式
    #     template_img:模板小图，也是灰度化了
    #     template_threshold:模板匹配的置信度
    #     '''
    #     image = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
    #     img_temp = cv2.cvtColor(color_img_temp,cv2.COLOR_BGR2GRAY)
    #     height,width= img_temp.shape
    #     # cv2.imwrite("temp_qq.jpg",img_temp)
    #     results = cv2.matchTemplate(image, img_temp, 5)

    #     location = []
    #     score = []
    #     for y in range(len(results)):
    #         for x in range(len(results[y])):
    #             if results[y][x] > conf:
    #                 # cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), 2)
    #                 loc = [(x, y), (x + width, y + height)]
    #                 location.append(loc)
    #                 score.append(results[y][x])
        

    #     return location