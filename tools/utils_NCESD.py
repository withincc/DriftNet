import os
import torch
import numpy as np
'''
Datetime:2025
Author:Yongxiang Chen
Description：a NCESD score tool
'''
def list_current_dir_files(path,file_type='.txt'):
    return [entry for entry in os.listdir(path) if entry.endswith(file_type)]
#基于球面余弦定理的大圆距离计算方法
def per_step_Spherical_distance(real_GPS, pred_GPS):
    true_lon, true_lat = real_GPS[:, 0], real_GPS[:, 1]  # 真实观测轨迹
    pred_lon, pred_lat = pred_GPS[:, 0], pred_GPS[:, 1]  # 预测轨迹
    R = 6371.004
    epsilon = 1e-16
    # 将角度转换为弧度
    real_lat_rad = true_lon * (torch.pi / 180.0)
    real_lon_rad = true_lat * (torch.pi / 180.0)
    pred_lat_rad = pred_lon * (torch.pi / 180.0)
    pred_lon_rad = pred_lat * (torch.pi / 180.0)

    # 计算球面余弦定理中的cos C
    cos_c = (torch.sin(real_lat_rad) * torch.sin(pred_lat_rad) +
             torch.cos(real_lat_rad) * torch.cos(pred_lat_rad) *
             torch.cos(real_lon_rad - pred_lon_rad+epsilon))
    # 限制cos_c的范围在[-1, 1]，避免数值误差导致的问题
    cos_c = torch.clamp(cos_c, -1.0000, 1.0000)#很有必要
    # 1.0000000000000002 1.0
    # 同样是1.0，有些会显示 1.0 ，有些则会在后面接一个小尾巴，这是由于Python的精度导致的

    # 计算弧长对应的角（弧度）
    arc_length = torch.acos(cos_c)

    Spherical_distance = R * arc_length

    return Spherical_distance
def one_track_NCSD_score(real_GPS, pred_GPS,his_length):
    cumulative_separation =per_step_Spherical_distance(real_GPS[0:his_length], pred_GPS[0:his_length]).sum()
    real_GPS_last,real_GPS_next=real_GPS[:his_length-1],real_GPS[1:his_length]
    trajectory_length=per_step_Spherical_distance(real_GPS_last,real_GPS_next).sum()
    normalized_separation = cumulative_separation / trajectory_length
    return normalized_separation

def calculate_cosine_distance(input1, target1):

    vectors_input1 = input1[1:] - input1[:-1]
    vectors_target1 = target1[1:] - target1[:-1]

    cosine_similarities = torch.nn.functional.cosine_similarity(vectors_input1, vectors_target1, dim=1)  # 计算余弦相似度
    average_cosine_similarity = cosine_similarities.mean()

    cosine_loss = 1 - average_cosine_similarity

    return cosine_loss
def one_file_score(file_path, his_length):
    # 加载数据
    GPS_data = np.loadtxt(file_path, delimiter=None, skiprows=0, dtype=float,
                          usecols=(2, 3, 4, 5))  # 真值/预测值

    GPS_data = torch.tensor(GPS_data[:his_length, :], dtype=torch.float64)
    GPS_data[0, 0:2] = GPS_data[0, 2:4]
    real_GPS = GPS_data[:his_length, 0:2]
    pred_GPS = GPS_data[:his_length, 2:4]

    # 地球平均半径，单位为千米
    PSSD_score=per_step_Spherical_distance(real_GPS, pred_GPS)

    ACD73 = calculate_cosine_distance(real_GPS[0:73], pred_GPS[0:73])
    ACD49 = calculate_cosine_distance(real_GPS[0:49], pred_GPS[0:49])
    ACD25 = calculate_cosine_distance(real_GPS[0:25], pred_GPS[0:25])
    NCSD_score73=one_track_NCSD_score(real_GPS=real_GPS, pred_GPS=pred_GPS,his_length=73)

    NCSD_score49=one_track_NCSD_score(real_GPS=real_GPS, pred_GPS=pred_GPS,his_length=49)
    NCSD_score25=one_track_NCSD_score(real_GPS=real_GPS, pred_GPS=pred_GPS,his_length=25)
    NCSD_score=torch.stack([ACD73,NCSD_score73,ACD49,NCSD_score49,ACD25,NCSD_score25])
    epoch_score=torch.cat([NCSD_score,PSSD_score],dim=0)

    return epoch_score

def one_epoch_result_score(result_path,his_length=73):#传入result路径，则输出该result的平均得分
    one_epoch_result_file_list = list_current_dir_files(result_path)
    epoch_score=one_file_score(os.path.join(result_path,one_epoch_result_file_list[0]),his_length)

    for i in range(1,len(one_epoch_result_file_list)):
        file_path=os.path.join(result_path,one_epoch_result_file_list[i])
        try:
            epoch_score+=one_file_score(file_path,his_length)
        except:continue
    return len(one_epoch_result_file_list),epoch_score/len(one_epoch_result_file_list)


def proj_best_score(proj_path):
    proj_path_result_list = os.listdir(proj_path)
    num, best_epoch_score = one_epoch_result_score(os.path.join(proj_path ,proj_path_result_list[0]), his_length=73)
    best_result_epoch_path=proj_path_result_list[0]
    for i in range(1,len(proj_path_result_list)):
        one_epoch_result_dir=os.path.join(proj_path , proj_path_result_list[i])
        if len(list_current_dir_files(one_epoch_result_dir))==0:
            continue
        num, result_epoch_score = one_epoch_result_score(one_epoch_result_dir)
        if result_epoch_score[1] < best_epoch_score[1]:
            best_epoch_score = result_epoch_score
            best_result_epoch_path = proj_path_result_list[i]
    return best_result_epoch_path, num, np.array(best_epoch_score, dtype=float)

def get_titles(his_length=None):
    BASE_COLUMNS = ('Proj', 'epoch', 'num')
    FINAL_COLUMNS = ('ACD72','NCSD72','ACD48','NCSD48','ACD24','NCSD24')
    HOUR_COLUMNS = [f'score_{h}h' for h in range(his_length)]
    COLUMN_NAMES = BASE_COLUMNS +FINAL_COLUMNS+ tuple(HOUR_COLUMNS)
    all_score = [list(COLUMN_NAMES)]
    return all_score
def all_proj_score(result_path,his_length=73):#传入所有proj路径，则输出该路径下所有proj的平均得分
    all_proj_result_list = next(os.walk(result_path))[1]
    print(all_proj_result_list)
    all_score = get_titles(his_length=his_length)
    for i in range(len(all_proj_result_list)):
        print('\r', 'precess:', i+1, '/', len(all_proj_result_list), end='')
        proj_result_name = all_proj_result_list[i]
        best_result_epoch_path, num, best_epoch_score = proj_best_score(os.path.join(result_path , proj_result_name))
        best_epoch_score=np.round(best_epoch_score,4)
        one_score = np.concatenate(([proj_result_name, best_result_epoch_path, str(num)], best_epoch_score), axis=0)
        all_score.append(one_score)


    # np.savetxt(result_path+self.task_name+'result_score1031.txt',all_score,delimiter='\t',fmt='%s')
    # print(all_score)
    np.savetxt(result_path + 'utils_NCESD0813.csv', all_score, delimiter=',', fmt='%s')
    print('结果已保存到:',result_path + 'utils_NCESD0813.csv')
    return all_score

if __name__ =='__main__':
    pass

    # all_score=all_proj_score('4-3-计算数据密集海域得分/result_SCORE/',his_length=73)
    # all_score=all_proj_score('4-2-计算数据稀疏海域得分/result_SCORE/',his_length=73)
    all_score=all_proj_score('result/',his_length=73)


