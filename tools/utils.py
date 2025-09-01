
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import numpy as np
import torch
from torch import nn
def list_current_dir_files(path):
    files = []
    for entry in os.listdir(path):  # '.' 表示当前工作目录
        if entry.endswith('.txt'):
            files.append(entry)
    return files
def get_all_files_absolute_paths(directory):
    absolute_paths = []
    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 使用os.path.join将根目录和文件名合并，然后使用os.path.abspath获取绝对路径
            if file.endswith('.txt'):
                absolute_paths.append(os.path.abspath(os.path.join(root, file)))
    return absolute_paths



class euclidean_angle_Loss(nn.Module):
    def __init__(self, angle_weight=1.0, distance_weight=1.0):
        super(euclidean_angle_Loss, self).__init__()
        self.angle_weight = angle_weight
        self.distance_weight = distance_weight

    def forward(self, v1, v2):

        dot_product = torch.sum(v1 * v2, dim=1)
        norm_v1 = torch.norm(v1, p=2, dim=1)
        norm_v2 = torch.norm(v2, p=2, dim=1)

        eps = 1e-8  # 小的常数防止分母为0
        cos_theta = dot_product / (norm_v1 * norm_v2 + eps)

        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))


        euclidean_distance = torch.norm(v1 - v2, p=2, dim=1)
        print('theta', theta, 'euclidean_distance', euclidean_distance)

        # 组合损失
        total_loss = self.angle_weight * theta + self.distance_weight * euclidean_distance

        return total_loss
class euclidean_cosine_Loss(nn.Module):
    def __init__(self, euclidean_weight=1.0, cosine_weight=1.0, reduction='mean'):
        super(euclidean_cosine_Loss, self).__init__()
        self.euclidean_weight = euclidean_weight  # 欧氏距离权重
        self.cosine_weight = cosine_weight  # 余弦距离权重
        self.reduction = reduction  # 可以是 'mean' 或 'sum'

    def forward(self, vec1, vec2):
        # 计算欧氏距离
        euclidean_distance = torch.norm(vec1 - vec2, p=2, dim=1)  # 对每个样本计算欧氏距离

        # 计算余弦相似度，然后转换为余弦距离
        cos_sim = nn.functional.cosine_similarity(vec1, vec2, dim=1)
        cosine_distance = 1 - cos_sim  # 余弦距离

        # 结合两种距离
        combined_loss = (self.euclidean_weight * euclidean_distance) + \
                        (self.cosine_weight * cosine_distance)

        # 根据 reduction 参数选择如何聚合损失
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:
            raise ValueError("reduction must be one of ('mean', 'sum')")

class CustomLoss(nn.Module):
    def __init__(self) :
        super().__init__()
    def euclidean_distance(self,outputs, targets):
        distance_abs = outputs - targets
        distance_loss = torch.norm(distance_abs, p=2, dim=1).mean()
        return distance_loss
    def cosine_distance_loss(self,input1, target1):
        """    - loss: 余弦距离损失值。    """
        # 计算每条轨迹的向量表示
        vectors_input1 = input1[1:] - input1[:-1]
        vectors_target1 = target1[1:] - target1[:-1]
        cosine_similarities = torch.nn.functional.cosine_similarity(vectors_input1, vectors_target1, dim=1)  # 计算余弦相似度
        average_cosine_similarity = cosine_similarities.mean()
        cosine_loss = 1 - average_cosine_similarity
        return cosine_loss

    def forward(self,outputs,targets):
        distance_loss= self.euclidean_distance(outputs, targets)
        cosine_loss=self.cosine_distance_loss(outputs, targets)
        loss=distance_loss+cosine_loss
        return loss


class Custom_cosine_length_Loss(nn.Module):
    def __init__(self,alpha=0.2, beta=0.8):
        super(Custom_cosine_length_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets):
        p = predictions
        t = targets

        euclidean_distance = torch.abs(torch.sqrt(p[0]**2+p[1]**2)-torch.sqrt(t[0]**2+t[1]**2))
        cosine_distance = 1 - nn.functional.cosine_similarity(p, t, dim=0)  # 要验证余弦距离计算是否正确

        totoal_loss = self.alpha * euclidean_distance + self.beta * cosine_distance

        return totoal_loss




def list_current_dir_files(path):
    files = []
    for entry in os.listdir(path):  # '.' 表示当前工作目录
        if entry.endswith('.txt'):
            files.append(entry)
    return files


def get_score(sub_file_path):
    # sub_file_path='C:\\Users\\Eruka\\Desktop\\202409OceanProject/0909_CNN_Roms/result/train_CNN_Both_ex10/result50/'
    all_file = list_current_dir_files(sub_file_path)
    score_12h = 0
    score_24h = 0
    score_mean = 0
    total_num = 0
    for file_name in all_file:
        GPS_data = np.loadtxt(sub_file_path + file_name, delimiter=' ', skiprows=0, dtype=float,
                              usecols=(2, 3, 4, 5))  # 真值/预测值
        score_12h += np.sqrt(np.sum((GPS_data[12, :2] - GPS_data[12, 2:4]) ** 2))
        score_24h += np.sqrt(np.sum((GPS_data[24, :2] - GPS_data[24, 2:4]) ** 2))
        score_mean += np.mean(
            np.sqrt(((GPS_data[:, :2] - GPS_data[:, 2:4]) ** 2)[0] + ((GPS_data[:, :2] - GPS_data[:, 2:4]) ** 2)[1]))
        total_num += 1
    return {'score_12h': score_12h / total_num, 'score_24h': score_24h / total_num,
            'score_mean': score_mean / total_num, 'total_num': total_num,'sub_file_path':sub_file_path}


def best_score(file_path):
    all_file = os.listdir(file_path)

    data_best = None
    score_12h_best = 9999
    score_24h_best = 9999
    score_mean_best = 9999
    score_mean_best_epoch = None
    for one_dir in all_file:
        sub_file_path = file_path + one_dir + '/'
        try:
            data = get_score(sub_file_path)
        except:
            continue
        if (data['score_mean'] < score_mean_best):
            score_mean_best = data['score_mean']
            data_best = data

    print(data_best)
    return data_best

def drawing_GPS_track(drawing_path):
    all_file = get_all_files_absolute_paths(drawing_path)

    all_file.sort(reverse=True)
    for file_path in all_file:
        print('\r drawing:', file_path, end='')
        try:
            marge_array1 = np.loadtxt(file_path, delimiter=' ', dtype='str')
            # print(marge_array1)
            merged_array = marge_array1[:, 2:6]
            merged_array = merged_array.astype(float)
            # print(merged_array)
            fig = plt.figure(figsize=(30, 30))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
            lonabs = merged_array[:, [0, 2]].max() - merged_array[:, [0, 2]].min()
            latabs = merged_array[:, [1, 3]].max() - merged_array[:, [1, 3]].min()
            # 设置地图范围

            ax.set_extent(
                [(merged_array[:, [0, 2]].min() - lonabs * 0.1), (merged_array[:, [0, 2]].max() + lonabs * 0.1),
                 (merged_array[:, [1, 3]].min() - latabs * 0.1), (merged_array[:, [1, 3]].max() + latabs * 0.1)])

            ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)  # 添加国界线
            ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)  # 添加海岸线
            ax.add_feature(cfeature.LAND, facecolor='coral')  # 添加陆地颜色
            ax.add_feature(cfeature.OCEAN, facecolor='aqua')  # 添加海洋颜色

            # 转换并绘制GPS轨迹
            # print('转换并绘制GPS轨迹1')
            ax.plot(merged_array[:, 2], merged_array[:, 3], transform=ccrs.Geodetic(), marker='*', linestyle='-',
                    color='blue', linewidth=1, label='Predict ')
            # print('转换并绘制GPS轨迹2')
            ax.plot(merged_array[:, 0], merged_array[:, 1], transform=ccrs.Geodetic(), marker='+', linestyle='-',
                    color='red', linewidth=1, label='Real')
            # print('转换并绘制GPS轨迹3')
            plt.legend(fontsize=(32))  # 图标大小为latabs*100，通常设置16,32
            # print('转换并绘制GPS轨迹4')
            plt.savefig(file_path[:-4] + '.png')
            # plt.show()
            # break
        except:
            continue
if __name__ == '__main__':
    best_score('result/train_CNN_Both_ex10/')
    # loss_fn=Custom_cosine_length_Loss()
    #
    # pos1=torch.tensor([[2,2]],dtype=float)
    # pos2=torch.tensor([[-1,-1]],dtype=float)
    # loss=loss_fn(pos1,pos2)
    # print(loss)