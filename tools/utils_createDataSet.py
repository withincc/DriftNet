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
'''
Datetime:2025
Author:Yongxiang Chen
'''
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

    def __init__(self, Current_Data_path=None, att_path='../roms3_cur.nc.att', ex=10,device='cuda'):
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
        self.roms_data = xr.open_dataset(Current_Data_path)


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

    # [row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
    def getOceanData_2h(self,first_Time, next_Time, row, col):
        uo_first_Time = self.roms_data.u_eastward.sel(ocean_time=first_Time).data[row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
        uo_next_Time = self.roms_data.u_eastward.sel(ocean_time=next_Time).data[row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
        vo_first_Time = self.roms_data.v_northward.sel(ocean_time=first_Time).data[row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
        vo_next_Time = self.roms_data.v_northward.sel(ocean_time=next_Time).data[row - self.ex: row + self.ex + 1, col - self.ex: col + self.ex + 1]
        uo_first_Time = torch.nan_to_num(torch.tensor(uo_first_Time)).to(self.device)
        uo_next_Time = torch.nan_to_num(torch.tensor(uo_next_Time)).to(self.device)
        vo_first_Time = torch.nan_to_num(torch.tensor(vo_first_Time)).to(self.device)
        vo_next_Time = torch.nan_to_num(torch.tensor(vo_next_Time)).to(self.device)

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
        BuoyData = [day_of_year,0, lon_min_max_normalize(current_lon),lat_min_max_normalize(current_lat),uo_last_point.item(), vo_last_point.item(), uo_next_point.item(), vo_next_point.item(),1]

        BuoyData = torch.tensor(BuoyData).to(self.device)

        return BuoyData, OceanData

