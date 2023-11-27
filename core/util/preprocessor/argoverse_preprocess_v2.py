# About: script to processing argoverse forecasting dataset
# Author: Jianbang LIU @ RPAI, CUHK
# Date: 2021.07.16

import os
import argparse
from os.path import join as pjoin
import copy
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import sparse

import warnings

# import torch
from torch.utils.data import Dataset, DataLoader

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline

from core.util.preprocessor.base import Preprocessor
from core.util.cubic_spline import Spline2D

warnings.filterwarnings("ignore")

RESCALE_LENGTH = 1.0    # the rescale length th turn the lane vector into equal distance pieces


class ArgoversePreprocessor(Preprocessor):
    def __init__(self,
                 root_dir,
                 split="train",
                 algo="tnt",
                 obs_horizon=20,
                 obs_range=100,
                 pred_horizon=30,
                 normalized=True,
                 save_dir=None):
        super(ArgoversePreprocessor, self).__init__(root_dir, algo, obs_horizon, obs_range, pred_horizon)

        self.LANE_WIDTH = {'MIA': 3.84, 'PIT': 3.97}
        self.COLOR_DICT = {"AGENT": "#d33e4c", "OTHERS": "#d3e8ef", "AV": "#007672"}

        self.split = split
        self.normalized = normalized

        self.am = ArgoverseMap()
        self.loader = ArgoverseForecastingLoader(pjoin(self.root_dir, self.split+"_obs" if split == "test" else split))

        self.save_dir = save_dir

    def __getitem__(self, idx):
        f_path = self.loader.seq_list[idx]
        seq = self.loader.get(f_path)
        path, seq_f_name_ext = os.path.split(f_path)
        seq_f_name, ext = os.path.splitext(seq_f_name_ext)

        df = copy.deepcopy(seq.seq_df)
        return self.process_and_save(df, seq_id=seq_f_name, dir_=self.save_dir)

    def process(self, dataframe: pd.DataFrame,  seq_id, map_feat=True):
        data = self.read_argo_data(dataframe)   # 读取轨迹数据
        data = self.get_obj_feats(data)        # 获取轨迹特征

        data['graph'] = self.get_lane_graph(data)   # 获取车道图
        data['seq_id'] = seq_id
        # visualization for debug purpose
        # self.visualize_data(data)
        return pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def read_argo_data(df: pd.DataFrame):
        '''
        读取轨迹数据
        '''
        city = df["CITY_NAME"].values[0]

        """TIMESTAMP, TRACK_ID, OBJECT_TYPE, X, Y, CITY_NAME"""
        agt_ts = np.sort(np.unique(df['TIMESTAMP'].values))     # the time stamps of the agent
        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i     # the mapping from the time stamp to the index

        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)
        '''
           X  Y    ->    array([[1, 4],
        0  1  4                 [2, 5],
        1  2  5                 [3, 6]])
        2  3  6
        '''
        

        steps = [mapping[x] for x in df['TIMESTAMP'].values]    # the time stamp of each trajectory point
        steps = np.asarray(steps, np.int64)

        objs = df.groupby(['TRACK_ID', 'OBJECT_TYPE']).groups   # 返回一个字典，key是 ('TRACK_ID', 'OBJECT_TYPE') 元组，value是索引
        keys = list(objs.keys())    # 获取所有的 ('TRACK_ID', 'OBJECT_TYPE') 组合，然后将其转换为一个列表
        obj_type = [x[1] for x in keys]     # x[1] 表示每个元组的第二个元素，即 'OBJECT_TYPE'

        agt_idx = obj_type.index('AGENT')   # 获取 'AGENT' 的索引
        idcs = objs[keys[agt_idx]]        # 获取与 'AGENT' 对应的索引列表。这些索引指向在原始 DataFrame df 中的行，这些行的 'OBJECT_TYPE' 是 'AGENT'

        agt_traj = trajs[idcs]          # 获取 'AGENT' 对应的轨迹
        agt_step = steps[idcs]        # 获取 'AGENT' 对应的时间戳

        del keys[agt_idx]
        ctx_trajs, ctx_steps = [], []   # ctx_trajs 是一个列表，列表中的每个元素是一个轨迹，轨迹是一个二维数组，每一行是一个点的坐标。ctx 的意思是 context
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['city'] = city
        data['trajs'] = [agt_traj] + ctx_trajs  # 这里 'AGENT' 代表自车
        data['steps'] = [agt_step] + ctx_steps  # 'steps' 是时间戳
        return data

    def get_obj_feats(self, data):
        '''
        读取轨迹特征
        '''
        # get the origin and compute the oritentation of the target agent   中心点
        orig = data['trajs'][0][self.obs_horizon-1].copy().astype(np.float32)

        # comput the rotation matrix 旋转矩阵
        if self.normalized:
            pre, conf = self.am.get_lane_direction(data['trajs'][0][self.obs_horizon-1], data['city'])
            if conf <= 0.1:
                pre = (orig - data['trajs'][0][self.obs_horizon-4]) / 2.0
            theta = - np.arctan2(pre[1], pre[0]) + np.pi / 2
            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]], np.float32)
        else:
            # if not normalized, do not rotate.
            theta = None
            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]], np.float32)

        # get the target candidates and candidate gt，gt 是 ground truth 的意思
        agt_traj_obs = data['trajs'][0][0: self.obs_horizon].copy().astype(np.float32)
        agt_traj_fut = data['trajs'][0][self.obs_horizon:self.obs_horizon+self.pred_horizon].copy().astype(np.float32)
        ctr_line_candts = self.am.get_candidate_centerlines_for_traj(agt_traj_obs, data['city'], viz=False) # 获取候选中心线

        # rotate the center lines and find the reference center line
        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T     # 旋转自车轨迹
        for i, _ in enumerate(ctr_line_candts):
            ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T     # 旋转中心线

        tar_candts = self.lane_candidate_sampling(ctr_line_candts, [0, 0], viz=False)   # 获取候选线

        if self.split == "test":
            tar_candts_gt, tar_offse_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(ctr_line_candts, agt_traj_fut)   # 获取参考线
            tar_candts_gt, tar_offse_gt = self.get_candidate_gt(tar_candts, agt_traj_fut[-1])   # 获取候选线的 ground truth

        # self.plot_target_candidates(ctr_line_candts, agt_traj_obs, agt_traj_fut, tar_candts)
        # if not np.all(offse_gt < self.LANE_WIDTH[data['city']]):
        #     self.plot_target_candidates(ctr_line_candts, agt_traj_obs, agt_traj_fut, tar_candts)

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []     # ctrs 是 centerlines 的意思
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        for traj, step in zip(data['trajs'], data['steps']):    # 遍历所有的轨迹
            if self.obs_horizon-1 not in step:  # 如果不包含预测点，就跳过
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T  # traj_nd 是 normalized 的意思

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), np.float32)
            has_pred = np.zeros(self.pred_horizon, bool)
            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
                # future_mask 数组标记了那些 step 值在 [self.obs_horizon, self.obs_horizon + self.pred_horizon) 范围内的元素
            post_step = step[future_mask] - self.obs_horizon    # post_step 是 step 值在 [0, self.pred_horizon) 范围内的元素
            post_traj = traj_nd[future_mask]    # post_traj 是 traj_nd 中 step 值在 [self.obs_horizon, self.obs_horizon + self.pred_horizon) 范围内的元素
            gt_pred[post_step] = post_traj   # 将 post_traj 的值赋给 gt_pred
            has_pred[post_step] = True     # 将 post_step 对应的 has_pred 置为 True

            # collect the observation
            obs_mask = step < self.obs_horizon  # obs_mask 数组标记了那些 step 值小于 self.obs_horizon 的元素
            step_obs = step[obs_mask]   # step_obs 是 step 值小于 self.obs_horizon 的元素
            traj_obs = traj_nd[obs_mask]    
            idcs = step_obs.argsort()   # 对 step_obs 进行排序，返回排序后的索引
            step_obs = step_obs[idcs]   # 按照 step_obs 的值进行排序
            traj_obs = traj_obs[idcs]   # 按照 step_obs 的值进行排序

            for i in range(len(step_obs)):  # 找到观察轨迹开始的位置，然后从那个位置开始截取 step_obs 和 traj_obs
                if step_obs[i] == self.obs_horizon - len(step_obs) + i:
                    break
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:]

            if len(step_obs) <= 1:  # 如果观察轨迹的长度小于等于 1，就跳过
                continue

            feat = np.zeros((self.obs_horizon, 3), np.float32)
            has_obs = np.zeros(self.obs_horizon, bool)

            feat[step_obs, :2] = traj_obs   # 将 feat 的前两列（即 0 和 1），索引为 step_obs 的位置的值设置为 traj_obs
            feat[step_obs, 2] = 1.0 # 将 feat 的第三列（即 2），索引为 step_obs 的位置的值设置为 1.0。这表示在观测期间，这些steps是被观察到的
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                # 如果 feat 的最后一行的前两列的值超出了范围，就跳过
                continue

            feats.append(feat)                  # displacement vectors
            has_obss.append(has_obs)            # whether the step is observed
            gt_preds.append(gt_pred)            # the ground truth of future prediction
            has_preds.append(has_pred)          # whether the step is predicted

        # if len(feats) < 1:
        #     raise Exception()

        feats = np.asarray(feats, np.float32)
        has_obss = np.asarray(has_obss, bool)
        gt_preds = np.asarray(gt_preds, np.float32)
        has_preds = np.asarray(has_preds, bool)

        # plot the splines
        # self.plot_reference_centerlines(ctr_line_candts, splines, feats[0], gt_preds[0], ref_idx)

        # # target candidate filtering
        # tar_candts = np.matmul(rot, (tar_candts - orig.reshape(-1, 2)).T).T
        # inlier = np.logical_and(np.fabs(tar_candts[:, 0]) <= self.obs_range, np.fabs(tar_candts[:, 1]) <= self.obs_range)
        # if not np.any(candts_gt[inlier]):
        #     raise Exception("The gt of target candidate exceeds the observation range!")

        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats                   # displacement vectors
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds           # whether the step is predicted
        data['gt_preds'] = gt_preds             # the ground truth of future prediction
        data['tar_candts'] = tar_candts         # the target candidate centerlines
        data['gt_candts'] = tar_candts_gt       # the ground truth of target candidate centerlines
        data['gt_tar_offset'] = tar_offse_gt    # the ground truth of target offset

        data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
        data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
        return data

    def get_lane_graph(self, data):
        """
        Get a rectangle area defined by pred_range.
        获取一个由 pred_range (horizon?) 定义的矩形区域，车道线
        """
        x_min, x_max, y_min, y_max = -self.obs_range, self.obs_range, -self.obs_range, self.obs_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
        lane_ids = self.am.get_lane_ids_in_xy_bbox(data['orig'][0], data['orig'][1], data['city'], radius * 1.5)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        for lane_id in lane_ids:
            lane = self.am.city_lane_centerlines_dict[data['city']][lane_id]
            lane = copy.deepcopy(lane)

            centerline = np.matmul(data['rot'], (lane.centerline - data['orig'].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]
            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = self.am.get_lane_segment_polygon(lane_id, data['city'])
                polygon = copy.deepcopy(polygon)
                lane.centerline = centerline
                lane.polygon = np.matmul(data['rot'], (polygon[:, :2] - data['orig'].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        ctrs, feats, turn, control, intersect = [], [], [], [], []
        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline  # 车道中心线？
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))   # 每两个点的中点  有什么用？
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))    # 每两个点之间的差值，即每两个点的位移向量

            x = np.zeros((num_segs, 2), np.float32)
            if lane.turn_direction == 'LEFT':
                x[:, 0] = 1
            elif lane.turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)

            control.append(lane.has_traffic_control * np.ones(num_segs, np.float32))
            intersect.append(lane.is_intersection * np.ones(num_segs, np.float32))

        lane_idcs = []
        count = 0
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)
        num_nodes = count
        lane_idcs = np.concatenate(lane_idcs, 0)    # 将 lane_idcs 列表中的所有元素拼接成一个一维数组

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['lane_idcs'] = lane_idcs

        return graph

    def visualize_data(self, data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_ctrs = data['graph']['ctrs']
        lines_feats = data['graph']['feats']
        lane_idcs = data['graph']['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        trajs = data['feats'][:, :, :2]
        has_obss = data['has_obss']
        preds = data['gt_preds']
        has_preds = data['has_preds']
        for i, [traj, has_obs, pred, has_pred] in enumerate(zip(trajs, has_obss, preds, has_preds)):
            self.plot_traj(traj[has_obs], pred[has_pred], i)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    @staticmethod
    def get_ref_centerline(cline_list, pred_gt):
        '''
        从给定的车道中心线列表 cline_list 中选择一个参考中心线。
        选择的依据是预测轨迹的最后一个点（pred_gt[-1, :2]）到每条中心线的最近距离。
        函数返回选择的参考中心线及其在列表中的索引
        '''
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx

    def plot_reference_centerlines(self, cline_list, splines, obs, pred, ref_line_idx):
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        for centerline_coords in cline_list:
            visualize_centerline(centerline_coords)

        for i, spline in enumerate(splines):
            xy = np.stack([spline.x_fine, spline.y_fine], axis=1)
            if i == ref_line_idx:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="r", alpha=0.7, linewidth=1, zorder=10)
            else:
                plt.plot(xy[:, 0], xy[:, 1], "--", color="b", alpha=0.5, linewidth=1, zorder=10)

        self.plot_traj(obs, pred)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.show()
        # plt.show(block=False)
        # plt.pause(0.5)

    def plot_traj(self, obs, pred, traj_id=None):
        assert len(obs) != 0, "ERROR: The input trajectory is empty!"
        traj_na = "t{}".format(traj_id) if traj_id else "traj"
        obj_type = "AGENT" if traj_id == 0 else "OTHERS"

        plt.plot(obs[:, 0], obs[:, 1], color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)
        plt.plot(pred[:, 0], pred[:, 1], "d-", color=self.COLOR_DICT[obj_type], alpha=1, linewidth=1, zorder=15)

        plt.text(obs[0, 0], obs[0, 1], "{}_s".format(traj_na))

        if len(pred) == 0:
            plt.text(obs[-1, 0], obs[-1, 1], "{}_e".format(traj_na))
        else:
            plt.text(pred[-1, 0], pred[-1, 1], "{}_e".format(traj_na))


def ref_copy(data):
    if isinstance(data, list):
        return [ref_copy(x) for x in data]
    if isinstance(data, dict):
        d = dict()
        for key in data:
            d[key] = ref_copy(data[key])
        return d
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--root", type=str, default="../dataset")
    parser.add_argument("-d", "--dest", type=str, default="../dataset")
    parser.add_argument("-s", "--small", action='store_true', default=False)
    args = parser.parse_args()

    # args.root = "/home/jb/projects/Code/trajectory-prediction/TNT-Trajectory-Predition/dataset"
    raw_dir = os.path.join(args.root, "raw_data")
    interm_dir = os.path.join(args.dest, "interm_data" if not args.small else "interm_data_small")

    for split in ["train", "val", "test"]:
        # construct the preprocessor and dataloader
        argoverse_processor = ArgoversePreprocessor(root_dir=raw_dir, split=split, save_dir=interm_dir)
        loader = DataLoader(argoverse_processor,
                            batch_size=1 if sys.gettrace() else 16,     # 1 batch in debug mode
                            num_workers=0 if sys.gettrace() else 3,    # use only 0 worker in debug mode
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

        for i, data in enumerate(tqdm(loader)):
            if args.small:
                if split == "train" and i >= 200:
                    break
                elif split == "val" and i >= 50:
                    break
                elif split == "test" and i >= 50:
                    break
