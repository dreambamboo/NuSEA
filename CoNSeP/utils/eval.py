# -*- coding=utf-8 -*-
'''
Pixel Level Evaluation
'''

import numpy as np
import sklearn.metrics as metrics
from PIL import Image
import cv2
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
import torch
from tqdm import *
import pickle
from scipy.ndimage import binary_erosion
'''
Object Level Evaluation
'''
from scipy.ndimage.measurements import center_of_mass
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff as hausdorff
from scipy.spatial import distance_matrix

class EvaluationPixel(object):
    def __init__(self, is_save):
        self.is_save = is_save  # save results or not
    def __call__(self, predict, truth, save_path):
        # predict truth : 1000*1000*10
        assert predict.shape[:-1] == truth.shape[:-1], "evaluation shape mismatch"
        dice1 = self._dice1_(predict, truth)
        iou1 = self._IoU_(predict, truth)
        acc1 = self._acc_(predict, truth)
        recall = self._recall_(predict, truth)
        precision = self._precision_(predict, truth)

        if self.is_save:
            fp = open(save_path+'_pred.dat',"wb")
            ft = open(save_path+'_truth.dat',"wb")
            pickle.dump(predict,fp)
            pickle.dump(truth,ft)
            fp.close()
            ft.close()
            h, w, c = predict.shape
            save_img = np.zeros((h, w, 3))
            predict[predict > 0] = 255
            truth[truth > 0] = 255
            save_img[:, :, 0] = predict[:, :, 0]
            save_img[:, :, 1] = truth[:, :, 0]
            Image.fromarray(save_img.astype(np.uint8)).convert("RGB").save(save_path+'.png')

        return {"dice":dice1,
                "iou":iou1,
                "acc":acc1,
                "recall":recall,
                "precision":precision}

    def _dice1_(self, predict, truth):
        pre_dice = predict[:, :, 0].copy()  # To calculate the dice coefficient, only the semantic segmentation results are needed.
        truth_dice = truth[:, :, 0].copy()
        pre_dice[pre_dice > 0] = 1
        pre_dice[pre_dice < 1] = 0  # Binarization
        truth_dice[truth_dice > 0] = 1
        truth_dice[truth_dice < 1] = 0
        pre_dice = self._fill_(pre_dice)  # Fill the holes in the connected components.
        up = (pre_dice * truth_dice).sum()
        down = (pre_dice.sum() + truth_dice.sum())
        dice = 2 * up / down
        return dice

    def _IoU_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy() 
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou))  
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou) 
        iou_pred = jaccard_score(truth_iou.astype(np.float32).reshape(-1),
                                            pre_iou.astype(np.float32).reshape(-1))
        return iou_pred  

    def _F1_score_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()  
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou)) 
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou)  
        TP = truth_iou * pre_iou  
        FP = pre_iou - TP  
        FN = truth_iou - TP  
        precision = TP.sum() / (TP.sum() + FP.sum())
        recall = TP.sum() / (TP.sum() + FN.sum())
        F1_score = (2 * precision * recall) / (precision + recall)
        return F1_score

    def _acc_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy() 
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou)) 
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou)  
        return accuracy_score(truth_iou.flatten(), pre_iou.flatten())

    def _recall_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy()  
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou)) 
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou) 
        return recall_score(truth_iou.flatten(), pre_iou.flatten())

    def _precision_(self, predict, truth):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy() 
            truth = truth.detach().cpu().numpy()
        pre_iou = predict[:, :, 0].copy()  
        truth_iou = truth[:, :, 0].copy()
        pre_iou = np.where(pre_iou > 0, np.ones_like(pre_iou), np.zeros_like(pre_iou))  
        truth_iou = np.where(truth_iou > 0, np.ones_like(truth_iou), np.zeros_like(truth_iou))
        pre_iou = self._fill_(pre_iou)  
        return precision_score(truth_iou.flatten(), pre_iou.flatten())

    def _fill_(self, predict):
        predict *= 255
        h, w = predict.shape
        pre3 = np.zeros((h, w, 3))
        contours, t = cv2.findContours(predict.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.drawContours(pre3, contours, i, (255, 0, 0), -1)
        return pre3[:, :, 0] / 255


class EvaluationInstance(object):
    def __init__(self, is_save):
        self.is_save = is_save  
    def __call__(self, predict, truth, save_path):
        if isinstance(predict, torch.Tensor):
            predict = predict.detach().cpu().numpy().astype(np.uint16)  # 如果是torch先变成numpy
            truth = truth.detach().cpu().numpy().astype(np.uint16)
        else:
            predict = predict.astype(np.uint16)
            truth = truth.astype(np.uint16)
        self.predict = predict
        self.truth = truth

        true_id_list = list(np.unique(truth))  
        pred_id_list = list(np.unique(predict))
        assert max(true_id_list) == len(true_id_list) - 1
        assert max(pred_id_list) == len(pred_id_list) - 1 
       
        true_masks = [
            None,
        ]
        print("List All Masks!")
        for t in tqdm(true_id_list[1:]):
            t_mask = np.array(truth == t, np.uint8)  # binary mask
            t_mask = np.max(t_mask, axis=2)  # h w
            true_masks.append(t_mask.astype(np.uint8))

        pred_masks = [
            None,
        ]
        for p in tqdm(pred_id_list[1:]):
            p_mask = np.array(predict == p, np.uint8)  # binary mask
            p_mask = np.max(p_mask, axis=2)  # h w
            p_mask = self._fill_(p_mask)  
            pred_masks.append(p_mask.astype(np.uint8))

        self.true_id_list = true_id_list
        self.pred_id_list = pred_id_list
        self.true_masks = true_masks
        self.pred_masks = pred_masks

        assert self.predict.shape[:-1] == self.truth.shape[:-1], "evaluation shape mismatch"

        if self.debug:
            [dq, sq, pq] = [0,0,0]
            aji_score = 0
            Dice_obj, IoU_obj, Hausdorff = 0,0,0
        else:
            [dq, sq, pq], _ = self._PQ_DQ_SQ_()
            aji_score = self._AJI_()
            Dice_obj, IoU_obj, Hausdorff = self._dice_iou_hausdorff_obj_()
        
        asd, mmsd = self._ASD_MMSD_()

        if self.is_save:
            predict = self.predict.copy()
            truth = self.truth.copy()
            h, w, c = predict.shape
            save_img = np.zeros((h, w, 3))
            predict[predict > 0] = 255
            truth[truth > 0] = 255
            save_img[:, :, 0] = predict[:, :, 0]
            save_img[:, :, 1] = truth[:, :, 0]
            Image.fromarray(save_img.astype(np.uint8)).convert("RGB").save(save_path)

        return {"DQ-F1":dq,
                "SQ":sq,
                "PQ":pq,
                "AJI":aji_score,
                "dice_obj":Dice_obj,
                "iou_obj":IoU_obj,
                "hausdorff_obj":Hausdorff,
                "Average_Symmetric_Surface_Distance": asd, 
                "Maximum_mean_surface_distance": mmsd
                }

    def _PQ_DQ_SQ_(self, match_iou=0.5):
        """`match_iou` is the IoU threshold level to determine the pairing between
        GT instances `p` and prediction instances `g`. `p` and `g` is a pair
        if IoU > `match_iou`. However, pair of `p` and `g` must be unique
        (1 prediction instance to 1 GT instance mapping).

        If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
        in bipartite graphs) is caculated to find the maximal amount of unique pairing.

        If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
        the number of pairs is also maximal.

        Fast computation requires instance IDs are in contiguous orderding
        i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
        and `by_size` flag has no effect on the result.

        Returns:
            [dq, sq, pq]: measurement statistic

            [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                          pairing information to perform measurement

        """
        print("Is calculating PQ_DQ_SQ ...")

        assert match_iou >= 0.0, "Cant' be negative"
        pred = self.predict.copy()
        gt = self.truth.copy()
        true_id_list = self.true_id_list.copy()
        pred_id_list = self.pred_id_list.copy()
        true_masks = self.true_masks.copy()
        pred_masks = self.pred_masks.copy()

        # prefill with value
        pairwise_iou = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )

        # caching pairwise iou

        for true_id in tqdm(true_id_list[1:]):  # 0-th is background  for each gt
            t_mask = true_masks[true_id]
            pred_true_overlap = pred[t_mask > 0, :]  
            pred_true_overlap_id = np.unique(pred_true_overlap).astype(np.uint16)
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:  
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id]
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                iou = inter / (total - inter)
                pairwise_iou[true_id - 1, pred_id - 1] = iou
        #
        if match_iou >= 0.5:  
            paired_iou = pairwise_iou[pairwise_iou > match_iou]  
            pairwise_iou[pairwise_iou <= match_iou] = 0.0

            keep_id_list = np.argmax(pairwise_iou, axis=1)
            tmp_iou = np.zeros_like(pairwise_iou)
            tmp_iou[np.arange(tmp_iou.shape[0]), keep_id_list] = pairwise_iou[np.arange(tmp_iou.shape[0]), keep_id_list]
            pairwise_iou = tmp_iou

            paired_true, paired_pred = np.nonzero(pairwise_iou)  
            paired_iou = pairwise_iou[paired_true, paired_pred] 
            paired_true += 1  
            paired_pred += 1 
        else:  # * Exhaustive maximal unique pairing  
            #### Munkres pairing with scipy library
            # the algorithm return (row indices, matched column indices)
            # if there is multiple same cost in a row, index of first occurence
            # is return, thus the unique pairing is ensure
            # inverse pair to get high IoU as minimum
            paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
            ### extract the paired cost and remove invalid pair
            paired_iou = pairwise_iou[paired_true, paired_pred]

            # now select those above threshold level
            # paired with iou = 0.0 i.e no intersection => FP or FN
            paired_true = list(paired_true[paired_iou > match_iou] + 1)
            paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
            paired_iou = paired_iou[paired_iou > match_iou]

        # get the actual FP and FN
        unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]  # FN
        unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]  # FP
        # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

        #
        tp = len(paired_true)
        fp = len(unpaired_pred)
        fn = len(unpaired_true)
        # get the F1-score i.e DQ
        dq = tp / (tp + 0.5 * fp + 0.5 * fn)
        # get the SQ, no paired has 0 iou so not impact
        sq = paired_iou.sum() / (tp + 1.0e-6)

        return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]

    def _AJI_(self):
        print('Is calculating AJI ...')
        pred = self.predict.copy()
        gt = self.truth.copy()
        true_id_list = self.true_id_list.copy()
        pred_id_list = self.pred_id_list.copy()
        true_masks = self.true_masks.copy()
        pred_masks = self.pred_masks.copy()

        # prefill with value
        pairwise_inter = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )  
        pairwise_union = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )

        # caching pairwise

        for true_id in tqdm(true_id_list[1:]):  # 0-th is background
            t_mask = true_masks[true_id]  
            pred_true_overlap = pred[t_mask > 0, :]  
            pred_true_overlap_id = np.unique(pred_true_overlap).astype(np.uint16)  
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:  #
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask = pred_masks[pred_id] 
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                pairwise_inter[true_id - 1, pred_id - 1] = inter
                pairwise_union[true_id - 1, pred_id - 1] = total - inter

        pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
        #### Munkres pairing to find maximal unique pairing 
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]
        # now select all those paired with iou != 0.0 i.e have intersection
        paired_true = paired_true[paired_iou > 0.0]
        paired_pred = paired_pred[paired_iou > 0.0]
        paired_inter = pairwise_inter[paired_true, paired_pred]
        paired_union = pairwise_union[paired_true, paired_pred]
        paired_true = list(paired_true + 1)  # index to instance ID
        paired_pred = list(paired_pred + 1)
        overall_inter = paired_inter.sum()
        overall_union = paired_union.sum()
        # add all unpaired GT and Prediction into the union
        unpaired_true = np.array(
            [idx for idx in true_id_list[1:] if idx not in paired_true]
        )
        unpaired_pred = np.array(
            [idx for idx in pred_id_list[1:] if idx not in paired_pred]
        )
        for true_id in unpaired_true:
            
            overall_union += true_masks[true_id].sum()
        for pred_id in unpaired_pred:
            overall_union += pred_masks[pred_id].sum()
        #
        aji_score = overall_inter / overall_union
        return aji_score

    def _ASD_MMSD_(self, asd_flag=True):  
        # Average_Symmetric_Surface_Distance 和 Maximum mean surface distance

        print('Is calculating ASD MMSD ...')

        pred = self.predict.copy()  # pred gt 均为1000*1000*10
        gt = self.truth.copy()
        gt_id_list = self.true_id_list.copy()  
        pred_id_list = self.pred_id_list.copy()
        gt_masks = self.true_masks.copy()  # none, then mask
        pred_masks = self.pred_masks.copy()

        # pred_labeled = label(pred, connectivity=2)
        Ns = len(pred_id_list) - 1  
        # gt_labeled = label(gt, connectivity=2)
        Ng = len(gt_id_list) - 1

        # --- compute asd mmsd--- #
        pred_objs_area = np.sum(pred > 0)  # total area of objects in image 
        gt_objs_area = np.sum(gt > 0)  # total area of objects in groundtruth gt

        # compute how well groundtruth object overlaps its segmented object
        # dice_g = 0.0
        # iou_g = 0.0
        # hausdorff_g = 0.0
        asd_g = 0.0
        mmsd_g = 0.0

        for i in tqdm(range(1, Ng + 1)):  
            gt_i = gt_masks[i]  # hw 0-1
            overlap_parts = pred[gt_i > 0, :]  

            # get intersection objects numbers in image
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]  
            gamma_i = float(np.sum(gt_i)) / gt_objs_area  

            # show_figures((pred_labeled, gt_i, overlap_parts))

            if obj_no.size == 0:  # no intersection object
                # dice_i = 0
                # iou_i = 0

                # find nearest segmented object in hausdorff distance
                if asd_flag:  
                    min_asd = 1e3  
                    min_mmsd = 1e3

                    # find overlap object in a window [-50, 50]
                    pred_cand_indices = self.find_candidates(gt_i, pred)  

                    for j in pred_cand_indices:  
                        pred_j = pred_masks[j]  
                        seg_ind = np.argwhere(pred_j)  
                        gt_ind = np.argwhere(gt_i)
                        seg_bound = self.find_edge_coordinates(pred_j)
                        gt_bound = self.find_edge_coordinates(gt_i)

                        # haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])  
                        asd_dist1_tmp = np.min(distance_matrix(seg_bound, gt_bound), axis=1)  
                        asd_dist2_tmp = np.min(distance_matrix(gt_bound, seg_bound), axis=1)
                        asd_dist_tmp = np.mean(np.concatenate([asd_dist1_tmp, asd_dist2_tmp]))

                        # mmsd
                        mmsd_tmp = np.max([np.mean(asd_dist1_tmp), np.mean(asd_dist2_tmp)])

                        if asd_dist_tmp < min_asd:
                            min_asd = asd_dist_tmp

                        if mmsd_tmp < min_mmsd:
                            min_mmsd = mmsd_tmp

                    asd_i = min_asd  
                    mmsd_i = min_mmsd
            else:
                # find max overlap object
                obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
                seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number 
                pred_i = pred_masks[seg_obj]  # segmented object

                overlap_area = np.max(obj_areas)  # overlap area  
                # dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
                # iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)
                if np.sum(pred_i) + np.sum(gt_i) - overlap_area == 0:
                    print(np.sum(pred_i), np.sum(gt_i), overlap_area)

                # compute asd distance
                if asd_flag:
                    seg_ind = np.argwhere(pred_i)
                    gt_ind = np.argwhere(gt_i)  # 

                    seg_bound = self.find_edge_coordinates(pred_i)
                    gt_bound = self.find_edge_coordinates(gt_i)


                    # haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
                    asd_dist1 = np.min(distance_matrix(seg_bound, gt_bound), axis=1)  
                    asd_dist2 = np.min(distance_matrix(gt_bound, seg_bound), axis=1)
                    asd_i = np.mean(np.concatenate([asd_dist1, asd_dist2]))

                    # 下面计算mmsd
                    mmsd_i = np.max([np.mean(asd_dist1), np.mean(asd_dist2)])

            if asd_flag:
                asd_g += gamma_i * asd_i
                mmsd_g += gamma_i * mmsd_i

        # compute how well segmented object overlaps its groundtruth object
        # dice_s = 0.0
        # iou_s = 0.0
        asd_s = 0.0
        mmsd_s = 0.0
        for j in tqdm(range(1, Ns + 1)):  
            pred_j = pred_masks[j]
            overlap_parts = gt[pred_j > 0, :]

            # get intersection objects number in gt
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]

            # show_figures((pred_j, gt_labeled, overlap_parts))

            sigma_j = float(np.sum(pred_j)) / pred_objs_area  
            # no intersection object
            if obj_no.size == 0:
                # dice_j = 0
                # iou_j = 0

                # find nearest groundtruth object in hausdorff distance
                if asd_flag:
                    min_asd = 1e3
                    min_mmsd = 1e3

                    # find overlap object in a window [-50, 50]
                    gt_cand_indices = self.find_candidates(pred_j, gt)

                    for i in gt_cand_indices:
                        gt_i = gt_masks[i]
                        seg_ind = np.argwhere(pred_j)
                        gt_ind = np.argwhere(gt_i)
                        seg_bound = self.find_edge_coordinates(pred_j)
                        gt_bound = self.find_edge_coordinates(gt_i)

                        # haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
                        asd_dist1_tmp = np.min(distance_matrix(seg_bound, gt_bound), axis=1)  
                        asd_dist2_tmp = np.min(distance_matrix(gt_bound, seg_bound), axis=1)
                        asd_dist_tmp = np.mean(np.concatenate([asd_dist1_tmp, asd_dist2_tmp]))

                        # mmsd
                        mmsd_tmp = np.max([np.mean(asd_dist1_tmp), np.mean(asd_dist2_tmp)])


                        if asd_dist_tmp < min_asd:
                            min_asd = asd_dist_tmp

                        if mmsd_tmp < min_mmsd:
                            min_mmsd = mmsd_tmp

                    asd_j = min_asd
                    mmsd_j = min_mmsd
            else:
                # find max overlap gt
                gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
                gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
                gt_j = gt_masks[gt_obj]  # groundtruth object

                # overlap_area = np.max(gt_areas)  # overlap area

                # dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
                # iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

                # compute hausdorff distance
                if asd_flag:
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_j)
                    seg_bound = self.find_edge_coordinates(pred_j)
                    gt_bound = self.find_edge_coordinates(gt_j)

                    # haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])
                    asd_dist1 = np.min(distance_matrix(seg_bound, gt_bound), axis=1) 
                    asd_dist2 = np.min(distance_matrix(gt_bound, seg_bound), axis=1)
                    asd_j = np.mean(np.concatenate([asd_dist1, asd_dist2]))

                    mmsd_j = np.max([np.mean(asd_dist1), np.mean(asd_dist2)])

            # dice_s += sigma_j * dice_j
            # iou_s += sigma_j * iou_j
            if asd_flag:
                asd_s += sigma_j * asd_j
                mmsd_s += sigma_j * mmsd_j
        # 返回dice、iou、hd
        return (asd_g + asd_s) / 2, (mmsd_g + mmsd_s) / 2

    def _dice_iou_hausdorff_obj_(self, hausdorff_flag=True):
        """ Compute the object-level metrics between predicted and
            groundtruth: dice, iou, hausdorff """
        print('Is calculating dice, iou, hausdorff ...')

        pred = self.predict.copy()
        gt = self.truth.copy()
        gt_id_list = self.true_id_list.copy()
        pred_id_list = self.pred_id_list.copy()
        gt_masks = self.true_masks.copy()
        pred_masks = self.pred_masks.copy()


        # pred_labeled = label(pred, connectivity=2)
        Ns = len(pred_id_list) - 1  
        # gt_labeled = label(gt, connectivity=2)
        Ng = len(gt_id_list) - 1

        # --- compute dice, iou, hausdorff --- #
        pred_objs_area = np.sum(pred > 0)  # total area of objects in image
        gt_objs_area = np.sum(gt > 0)  # total area of objects in groundtruth gt

        # compute how well groundtruth object overlaps its segmented object
        dice_g = 0.0
        iou_g = 0.0
        hausdorff_g = 0.0

        for i in tqdm(range(1, Ng + 1)):  
            gt_i = gt_masks[i]  # hw 0-1
            overlap_parts = pred[gt_i > 0, :]  

            # get intersection objects numbers in image
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0] 
            gamma_i = float(np.sum(gt_i)) / gt_objs_area  

            # show_figures((pred_labeled, gt_i, overlap_parts))

            if obj_no.size == 0:  #
                dice_i = 0
                iou_i = 0

                # find nearest segmented object in hausdorff distance
                if hausdorff_flag:  
                    min_haus = 1e3

                    # find overlap object in a window [-50, 50]
                    pred_cand_indices = self.find_candidates(gt_i, pred)  

                    for j in pred_cand_indices:  
                        pred_j = pred_masks[j]
                        seg_ind = np.argwhere(pred_j)  
                        gt_ind = np.argwhere(gt_i)
                        haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])  

                        if haus_tmp < min_haus:
                            min_haus = haus_tmp
                    haus_i = min_haus  
            else:
                # find max overlap object
                obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
                seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number  
                pred_i = pred_masks[seg_obj]  # segmented object

                overlap_area = np.max(obj_areas)  # overlap area  
                dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
                iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)
                if np.sum(pred_i) + np.sum(gt_i) - overlap_area == 0:
                    print(np.sum(pred_i), np.sum(gt_i), overlap_area)

                # compute hausdorff distance
                if hausdorff_flag:
                    seg_ind = np.argwhere(pred_i)
                    gt_ind = np.argwhere(gt_i)
                    haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

            dice_g += gamma_i * dice_i  
            iou_g += gamma_i * iou_i
            if hausdorff_flag:
                hausdorff_g += gamma_i * haus_i

        # compute how well segmented object overlaps its groundtruth object
        dice_s = 0.0
        iou_s = 0.0
        hausdorff_s = 0.0
        for j in tqdm(range(1, Ns + 1)):  
            pred_j = pred_masks[j]
            overlap_parts = gt[pred_j > 0, :]

            # get intersection objects number in gt
            obj_no = np.unique(overlap_parts)
            obj_no = obj_no[obj_no != 0]

            # show_figures((pred_j, gt_labeled, overlap_parts))

            sigma_j = float(np.sum(pred_j)) / pred_objs_area  
            # no intersection object
            if obj_no.size == 0:
                dice_j = 0
                iou_j = 0

                # find nearest groundtruth object in hausdorff distance
                if hausdorff_flag:
                    min_haus = 1e3

                    # find overlap object in a window [-50, 50]
                    gt_cand_indices = self.find_candidates(pred_j, gt)

                    for i in gt_cand_indices:
                        gt_i = gt_masks[i]
                        seg_ind = np.argwhere(pred_j)
                        gt_ind = np.argwhere(gt_i)
                        haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                        if haus_tmp < min_haus:
                            min_haus = haus_tmp
                    haus_j = min_haus
            else:
                # find max overlap gt
                gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
                gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
                gt_j = gt_masks[gt_obj]  # groundtruth object

                overlap_area = np.max(gt_areas)  # overlap area

                dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
                iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

                # compute hausdorff distance
                if hausdorff_flag:
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_j)
                    haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

            dice_s += sigma_j * dice_j
            iou_s += sigma_j * iou_j
            if hausdorff_flag:
                hausdorff_s += sigma_j * haus_j
        # 返回dice、iou、hd距离指标
        return (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2

    def find_candidates(self, obj_i, objects_labeled, radius=50):
        """
        find object indices in objects_labeled in a window centered at obj_i
        when computing object-level hausdorff distance
        """
        if radius > 400:
            return np.array([])

        h, w, _ = objects_labeled.shape  # h w 10
        x, y = center_of_mass(obj_i)
        x, y = int(x), int(y)
        r1 = x - radius if x - radius >= 0 else 0
        r2 = x + radius if x + radius <= h else h
        c1 = y - radius if y - radius >= 0 else 0
        c2 = y + radius if y + radius < w else w
        indices = np.unique(objects_labeled[r1:r2, c1:c2, :])  
        indices = indices[indices != 0]  

        if indices.size == 0:  
            indices = self.find_candidates(obj_i, objects_labeled, 2 * radius)

        return indices

    def _fill_(self, predict):
        predict *= 255
        h, w = predict.shape
        pre3 = np.zeros((h, w, 3))
        contours, t = cv2.findContours(predict.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for i in range(0, len(contours)):
            cv2.drawContours(pre3, contours, i, (255, 0, 0), -1)
        return pre3[:, :, 0] / 255

    def find_edge_coordinates(self, cell_mask):
        eroded_mask = binary_erosion(cell_mask)
        edge_mask = cell_mask - eroded_mask
        edge_coordinates = np.argwhere(edge_mask == 1)
        return edge_coordinates

