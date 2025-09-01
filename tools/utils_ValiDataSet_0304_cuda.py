import os
import logging
import time

import numpy as np
import torch
from torch import nn
from datetime import datetime, timedelta
from scipy.interpolate import griddata
from functools import lru_cache
import xarray as xr

def get_all_files_absolute_paths(directory):
    return [os.path.abspath(os.path.join(root, file)) for root, _, files in os.walk(directory) for file in files]
def list_current_dir_files(path):
    return [entry for entry in os.listdir(path) if entry.endswith('.txt')]

def lon_min_max_normalize(lon):
    return 2* ((lon-116)/(123-116)) - 1
def lat_min_max_normalize(lat):
    return 2* ((lat-22)/(29-22)) - 1
def datetime_min_max_normalize(t):
    return 2 * (t / 366.5) - 1

class CreateData(nn.Module):

    def __init__(self, Current_Data_path=None, att_path='../roms3_cur.nc.att', ex=10,device='cuda',vali_directory = None):
        super(CreateData, self).__init__()
        self.Current_Data_path = Current_Data_path
        self.device=device
        self.ex = ex
        import xarray as xr
        datt = xr.open_dataset(att_path)  # roms台湾海峡模型文件
        lon_rho_data = datt.lon_rho.data
        lat_rho_data = datt.lat_rho.data

        self.lon_rho_data_numpy = lon_rho_data
        self.lat_rho_data_numpy = lat_rho_data

        self.lon_rho_data_tensor = torch.tensor(lon_rho_data).to(self.device)
        self.lat_rho_data_tensor = torch.tensor(lat_rho_data).to(self.device)

        self.lon_rho_normalize_tensor = torch.tensor(lon_min_max_normalize(lon_rho_data)).to(self.device)
        self.lat_rho_normalize_tensor = torch.tensor(lat_min_max_normalize(lat_rho_data)).to(self.device)

        self.att_mask = torch.tensor(datt.mask_rho.data).to(self.device)
        roms_data = xr.open_dataset(Current_Data_path )

        roms_data_tensor,GPSData_valifile_data=self.roms_data_tensor_to_cuda(roms_data,vali_directory,device=self.device)
        self.roms_data_tensor=roms_data_tensor
        self.GPSData_valifile_data=GPSData_valifile_data
        # self.roms_data = xr.open_dataset(Current_Data_path + 'combined_data_2019_2021.nc')
        # self.roms_data2 = xr.open_dataset(Current_Data_path + 'combinted_small.nc')


    def roms_data_tensor_to_cuda(self,roms_data,vali_directory,device='cuda'):

        def get_GPS_info(one_GPS_data):
            GPSTime = one_GPS_data[0] + one_GPS_data[1]
            try:
                time_last = datetime.strptime(GPSTime, '%Y-%m-%d%H:%M:%S')
            except:
                time_last = datetime.strptime(GPSTime, '%Y-%m-%d%H:%M')

            return time_last

        save_pth = {}
        files = list_current_dir_files(vali_directory)
        GPSData_valifile_data={}
        for i, file in enumerate(files):
            GPS_Data = np.loadtxt(os.path.join(vali_directory, file), delimiter=' ', skiprows=0, dtype=str)
            GPSData_valifile_data[file]=GPS_Data
            for i in range(len(GPS_Data)):
                current_Time = get_GPS_info(GPS_Data[i])
                uo = roms_data.u_eastward.sel(ocean_time=current_Time).data
                vo = roms_data.v_northward.sel(ocean_time=current_Time).data
                uo = torch.nan_to_num(torch.tensor(uo))
                vo = torch.nan_to_num(torch.tensor(vo))

                save_pth[current_Time] = {
                    'uo': uo.to(device),
                    'vo': vo.to(device)
                }
        return save_pth,GPSData_valifile_data

    @lru_cache(maxsize=128)
    def get_GridPosi(self, this_lon, this_lat):
        this_lon = float(this_lon)
        this_lat = float(this_lat)

        tensor_abs = (self.lon_rho_data_tensor - this_lon)**2 + (self.lat_rho_data_tensor - this_lat)**2

        flat_index = torch.argmin(tensor_abs)
        row, col = torch.unravel_index(flat_index, (400,400))
        return row.item(), col.item(),tensor_abs



    def get_interpolated_value(self, current_lon, current_lat,row, col,uo_first, uo_next, vo_first, vo_next):
        posi = np.array([current_lon, current_lat])

        lon_rho_data=self.lon_rho_data_numpy[row - self.ex:row + self.ex+1, col - self.ex:col + self.ex+1].flatten()
        lat_rho_data = self.lat_rho_data_numpy[row - self.ex:row + self.ex+1, col - self.ex:col + self.ex+1].flatten()

        uo_last_point = griddata((lon_rho_data,lat_rho_data),uo_first.flatten(), posi, method='linear')
        vo_last_point = griddata((lon_rho_data,lat_rho_data),vo_first.flatten(), posi, method='linear')
        uo_next_point = griddata((lon_rho_data,lat_rho_data),uo_next.flatten(), posi, method='linear')
        vo_next_point = griddata((lon_rho_data,lat_rho_data),vo_next.flatten(), posi, method='linear')

        return uo_last_point, vo_last_point, uo_next_point, vo_next_point

    def get_GPS_info(self, one_GPS_data):
        GPSTime = one_GPS_data[0] + one_GPS_data[1]
        current_lon, current_lat = float(one_GPS_data[2]), float(one_GPS_data[3])
        try:
            time_last = datetime.strptime(GPSTime, '%Y-%m-%d%H:%M:%S')
        except:
            time_last = datetime.strptime(GPSTime, '%Y-%m-%d%H:%M')
        time_next = time_last + timedelta(hours=1)

        return current_lon, current_lat, time_last, time_next

    def getOceanData_2h(self,first_Time, next_Time, row, col):
        uo_first_Time=self.roms_data_tensor[first_Time]['uo'][row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
        uo_next_Time = self.roms_data_tensor[next_Time]['uo'][row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
        vo_first_Time = self.roms_data_tensor[first_Time]['vo'][row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
        vo_next_Time = self.roms_data_tensor[next_Time]['vo'][row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
        return uo_first_Time,uo_next_Time,vo_first_Time,vo_next_Time
    # uo_first, uo_next, vo_first, vo_next
    def forward(self, one_GPS_data):
        current_lon, current_lat, time_last, time_next = self.get_GPS_info(one_GPS_data)
        # 时间维度嵌入
        day_of_year = datetime_min_max_normalize(time_last.timetuple().tm_yday + (time_last.time().hour) / 24)
        day_of_year_full = torch.full((self.ex*2+1, self.ex*2+1), day_of_year,device=self.device)
        # 局部位置编码
        row, col, tensor_abs_ex = self.get_GridPosi(one_GPS_data[2], one_GPS_data[3])  # 获取到输入的经坐标数据
        uo_first, uo_next, vo_first, vo_next = self.getOceanData_2h(time_last, time_next,row, col)

        OceanData = torch.stack((
            day_of_year_full,

            tensor_abs_ex[row - self.ex:row + self.ex+1, col - self.ex:col + self.ex+1],
            self.lon_rho_normalize_tensor[row - self.ex:row + self.ex+1, col - self.ex:col + self.ex+1],
            self.lat_rho_normalize_tensor[row - self.ex:row + self.ex+1, col - self.ex:col + self.ex+1],
            uo_first,
            vo_first,
            uo_next,

            vo_next,
            self.att_mask[row - self.ex:row + self.ex+1, col - self.ex:col + self.ex+1]
        ), dim=0)

        uo_last_point, vo_last_point, uo_next_point, vo_next_point = self.get_interpolated_value(current_lon, current_lat, row, col,uo_first.cpu().numpy(), uo_next.cpu().numpy(), vo_first.cpu().numpy(), vo_next.cpu().numpy())
        DriftData = [day_of_year,0, lon_min_max_normalize(current_lon),lat_min_max_normalize(current_lat),uo_last_point.item(), vo_last_point.item(), uo_next_point.item(), vo_next_point.item(),1]

        DriftData = torch.tensor(DriftData).to(self.device)

        return DriftData, OceanData


@torch.no_grad()
def evaluate(model,Vali_DataSet,loss_function,task_name,device,epoch):
    if os.path.exists(f"./result/{task_name}/{epoch:04d}") is False:
        os.makedirs(f"./result/{task_name}/{epoch:04d}")
    # vali_files = list_current_dir_files(vali_directory,file_type='.txt')
    vali_files=Vali_DataSet.GPSData_valifile_data
    model.eval()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    vali_number=0
    for file_n,vali_file in enumerate(vali_files):

        vali_GPS_data=vali_files[vali_file]
        arr1 = np.zeros((len(vali_GPS_data), 2))
        arr1[0][0] = float(vali_GPS_data[0][2])
        arr1[0][1] = float(vali_GPS_data[0][3])
        GPSTime = vali_GPS_data[0][0] + ' ' + vali_GPS_data[0][1]
        first_Time = datetime.strptime(GPSTime, '%Y-%m-%d %H:%M:%S')

        for i in range(0,len(vali_GPS_data)-1):
            next_Time = first_Time + timedelta(hours=i)
            date1 = next_Time.strftime('%Y-%m-%d')
            time1 = next_Time.strftime('%H:%M:%S')
            one_gps_data=[date1,time1,arr1[i][0],arr1[i][1]]
            try:
                DriftData,OceanData=Vali_DataSet(one_gps_data)
                DriftData,OceanData=DriftData.unsqueeze(0).float(), OceanData.unsqueeze(0).float()
                target_data = [[(float(vali_GPS_data[i + 1][2])-float(vali_GPS_data[i][2]) ),
                            (float(vali_GPS_data[i + 1][3])-float(vali_GPS_data[i][3]) )]]

                outputs = model(DriftData.to(device), OceanData.to(device))
            except Exception as e:
                print(f'{vali_file}-{i} meet a error:',e)
                break
            predict_pos = outputs.cpu().detach().numpy()
            predict_pos=np.nan_to_num(predict_pos)
            arr1[i + 1][0] = predict_pos[0][0] + arr1[i][0]
            arr1[i + 1][1] = predict_pos[0][1] + arr1[i][1]
            loss = loss_function(outputs, torch.tensor(target_data).to(device))
            accu_loss += loss.detach()
            print('\r',f' [evaluate_epoch {epoch}] {file_n}/60 loss: {loss.item():.6f}',end='')
        else:
            vali_number += len(vali_GPS_data)
            marge_array1 = np.concatenate((vali_GPS_data, arr1), axis=1)
            np.savetxt(f"./result/{task_name}/{epoch:04d}/" + vali_file, marge_array1, delimiter=' ', fmt='%s')

    return accu_loss.item() / (vali_number + 1)
# task_name, model, vali_directory, device=device, start_epoch=0,end_epoch=4, stride_epoch=2
def run_evaluate(task_name,model,Vali_DataSet,device='cuda',start_epoch=449,end_epoch=999,stride_epoch=2):
    if os.path.exists(f'result/{task_name}/') is False:
        os.makedirs(f'result/{task_name}/')
    if os.path.exists(f'checkpoint/{task_name}/') is False:
        os.makedirs(f'checkpoint/{task_name}/')
    e_s_t = time.time()
    def get_all_files_absolute_paths(directory):
        return [os.path.abspath(os.path.join(root, file)) for root, _, files in os.walk(directory) for file in files]

    checkpoint_evaluate_path=f'checkpoint/{task_name}/checkpoint-result.pth'
    if os.path.exists(checkpoint_evaluate_path):
        checkpoint_epoch=torch.load(checkpoint_evaluate_path)['checkpoint_epoch']
    else:checkpoint_epoch=start_epoch
    checkpoint_list = get_all_files_absolute_paths('weights/' + task_name)
    checkpoint_list.sort()
    model=model.to(device)

    # Current_Data_path = '../', att_path = '../roms3_cur.nc.att', vali_directory = None, ex = 10, device = 'cuda'
    for i, path_checkpoint in enumerate(checkpoint_list[::stride_epoch]):
        checkpoint = torch.load(path_checkpoint)
        state_dict = checkpoint['net']
        the_epoch=checkpoint['epoch']
        if the_epoch <= checkpoint_epoch:
            continue
        print(model.load_state_dict(state_dict, strict=True))
        val_loss = evaluate(model=model,
                        Vali_DataSet=Vali_DataSet,
                        loss_function=torch.nn.L1Loss().to(device),
                        task_name= task_name,
                        device=device, epoch=the_epoch)
        check_info={'checkpoint_epoch':the_epoch}
        torch.save(check_info,checkpoint_evaluate_path)
        print(f'{task_name},epoch:{the_epoch},              val_loss:{val_loss}')
    print(f'{task_name} val_time', (time.time() - e_s_t) / 60, '分钟')


if __name__ == '__main__':
    # task_name = os.path.basename(__file__)[:-3]
    device = 'cuda'
    import os
    task_name = 'X0304_Run_MultiHead4'
    vali_directory = '/home/fjhyj/cyx/X0212/GPSData_vali_72h_n60/'

    from X0417_RunNoiseBase import DriftFormer_0220_MultiHead


    model = DriftFormer_0220_MultiHead(hidden_dim=48, depth=24, num_heads=4).float()
    run_evaluate(task_name, model, vali_directory, device=device, start_epoch=0,end_epoch=4, stride_epoch=2)