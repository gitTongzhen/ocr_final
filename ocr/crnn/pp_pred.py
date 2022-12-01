import onnxruntime as ort
import numpy as np
import time
import traceback
from multiprocessing.dummy import Pool as ThreadPool
import cv2

from ocr.crnn.util import get_rotate_crop_image

class PPrecPredictor:
    '''
    百度的识别模型
    '''
    def __init__(self, model_path ,target_h ,batch_num,keys_txt_path="",in_names="in",out_names=["out"]):
        self.keys_txt_path = keys_txt_path
        self.in_names = in_names
        self.out_names = out_names        
        self.target_h = target_h
        self.batch_num = batch_num
        self.sess = ort.InferenceSession(model_path)
 
        self.character_str = []
        character_dict_path = keys_txt_path
        use_space_char = True
        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def preprocess(self,im,max_wh_ratio = 3):
        target_h = self.target_h
        scale = im.shape[0] * 1.0 / target_h
        w = im.shape[1] / scale
        w = int(w)
        img = cv2.resize(im,(w, target_h),interpolation=cv2.INTER_AREA) # ,interpolation=cv2.INTER_AREA 在这里效果最好

        if self.batch_num > 1:
            # 最大
            max_resized_w = int(max_wh_ratio * target_h)
            padding_im = np.zeros((target_h,max_resized_w,3), dtype=np.float32)
            padding_im[:,0:w,:] = img
            img = padding_im

        img -= 127.5
        img /= 127.5

        image = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)
        return transformed_image

    def npbox2box(self,npbox):
        '''
        numpy格式转列表格式
        '''
        x = npbox[0][0]
        y = npbox[0][1]
        w = npbox[2][0] - x + 1
        h = npbox[2][1] - y + 1
        return x,y,w,h

    def check_edge(self,x,y,bb_w,bb_h,h,w):
        '''
        边界检查
        '''
        x = max(0,x)
        y = max(0,y)
        bb_w = w-bb_w-x if x + bb_w > w else bb_w
        bb_h = h-bb_h-y if y + bb_h > h else bb_h
        return x,y,bb_w,bb_h

    def expand(self,x,y,bb_w,bb_h,h,w,delta=1):
        '''
        外扩delta个像素
        '''
        x -= delta
        y -= delta 
        bb_w += 2*delta
        bb_h += 2*delta
        x,y,bb_w,bb_h = self.check_edge(x,y,bb_w,bb_h,h,w)
        return x,y,bb_w,bb_h

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        th = 0.15
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token
            # 不使用置信度阈值过滤
            # char_list = [
            #     self.character[text_id]
            #     for idx,text_id in enumerate(text_index[batch_idx][selection])
            # ]
            # 使用置信度阈值过滤
            char_list = [
                self.character[text_id]
                for idx,text_id in enumerate(text_index[batch_idx][selection])
                if text_prob[batch_idx][selection][idx] > th
            ]

            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

    def predict_rbg(self, im):
        """
        预测
        """
        t1 = time.time()
        preds = self.sess.run(self.out_names, {self.in_names: im.astype(np.float32)}) # 12ms
        # print("run cost: ",time.time()-t1)
        preds = preds[0]
        # preds = softmax(preds)
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)

        return text
    
    def pp_predict(self,data):
        '''
        多线程预测原子函数
        '''
        im,box = data
        x,y,bb_w,bb_h = box 
        # 外扩delta个像素
        # x,y,bb_w,bb_h = self.expand(x,y,bb_w,bb_h,im.shape[0],im.shape[1],2)
        box = np.array([[x,y],[x+bb_w,y],[x+bb_w,y+bb_h],[x,y+bb_h]])
        try:
            # 裁剪
            im = get_rotate_crop_image(im, box.astype(np.float32))
            if im.shape[0] > 2 and im.shape[1] > 2:
                partImg = self.preprocess(im.astype(np.float32))
                result = self.predict_rbg(partImg)  ##识别的文本
            else:
                result = [("",0)]
        except Exception as e:
            print(traceback.format_exc())
            result = [("",0)]
        simPred,prob = result[0]
        return self.npbox2box(box),simPred,prob

    def __call__(self,texts_list,use_mp = False, process_num = 1):
        
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img
        """

        results = []
        if not use_mp:
            # 不用多线程
            for box,im in texts_list:
                if  not (box[2] > 3 and box[3] > 3 and box[2] / box[3] < 50):
                    continue
                x,y,bb_w,bb_h = box 
                box = np.array([[x,y],[x+bb_w,y],[x+bb_w,y+bb_h],[x,y+bb_h]])
                # 裁剪
                # partImg_array = get_rotate_crop_image(im, box.astype(np.float32))
                partImg = self.preprocess(im.astype(np.float32))
                try:
                    result = self.predict_rbg(partImg)  ##识别的文本
                except Exception as e:
                    print(traceback.format_exc())
                    continue
                simPred,prob = result[0]
                results.append([self.npbox2box(box),simPred])
        else:
            # 多线程方法二（更快）
            datas = [(im,box) for box,im in texts_list if box[2] > 3 and box[3] > 3 and box[2] / box[3] < 50]
            pool = ThreadPool(processes = process_num)
            results = pool.map(self.pp_predict, datas)
            pool.close()
            pool.join()
            used_boxes = set([box[1] for box in datas])         # 做了识别的框
            text_boxes = set([box[0] for box in texts_list])    # 所有框
            rest_boxes = list(text_boxes - used_boxes)          # 剩余未做识别的框
            results.extend([(self.npbox2box(np.array([[x,y],[x+bb_w,y],[x+bb_w,y+bb_h],[x,y+bb_h]])),"",0) for x,y,bb_w,bb_h in rest_boxes])

        return results
