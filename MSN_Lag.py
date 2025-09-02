import os
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
from tqdm import tqdm


# 常量定义
SCALE_MULTIPLIER = 100
FORMAT_DATE = '%Y-%m-%d'
FORMAT_TIME_LIST = '%Y%m%d'
NC_FILE_EXTENSION = '.nc'


def get_all_files_absolute_paths(directory):
    absolute_paths = []
    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 使用os.path.join将根目录和文件名合并，然后使用os.path.abspath获取绝对路径
            if file.endswith('.txt'):
                absolute_paths.append(os.path.abspath(os.path.join(root, file)))
    return absolute_paths


# 获取文件列表
file_dir1 = get_all_files_absolute_paths(
    'C:/Users/Eruka/Desktop/202409OceanProject/X0212/GPS_Data/GPSData_vali_72h_n60/')
file_dir2 = get_all_files_absolute_paths('C:/Users/Eruka/Desktop/202409OceanProject/GPS_Data_Split/Both_GPS/Both_GPS/')
file_list = file_dir1

# 初始化时间列表
time_list = []

# 遍历每个文件，生成时间范围
for file_path in file_list:
    GPS_file = np.loadtxt(file_path, delimiter=' ', skiprows=0, dtype=str)
    start_time = datetime.strptime(GPS_file[0, 0], '%Y-%m-%d') - timedelta(days=2)
    end_time = datetime.strptime(GPS_file[-1, 0], '%Y-%m-%d') + timedelta(days=5)

    while start_time <= end_time:
        time_list.append(start_time)  # 添加原始 datetime 对象
        start_time += timedelta(days=1)

# 去重并排序
time_list = sorted(list(set(time_list)))

# 将时间格式化为 YYYYMMDD 格式
formatted_time_list = [time.strftime('%Y%m%d') for time in time_list]
source_nc_dir = 'D:/drift/drift/nc/roms3_npzd_res/his/'

nc_list = os.listdir(source_nc_dir)


def standardize(data):
    mean = np.nanmean(data)
    std = np.nanstd(data)
    return (data - mean) / std, mean, std


# 反标准化函数
def inverse_standardize(data, mean, std):
    return data * std + mean


def add_gaussian_noise_to_Lagrange(noise_scale):
    np.random.seed(2025)
    target_nv_dir = f'D:/drift/drift/nc/roms3_npzd_res/Noise04014_{int(noise_scale*10):02d}/'

    os.makedirs(target_nv_dir, exist_ok=True)
    pbar=tqdm(formatted_time_list)
    for file_name_time in pbar:
        file_name = 'roms3_npzd_his.nc.' + file_name_time
        try:
            ds = xr.open_dataset(os.path.join(source_nc_dir, file_name))
            time_data= ds.ocean_time.data
            output_file = os.path.join(target_nv_dir, f'roms3_npzd_his.nc.{file_name_time}')
            for i in range(len(time_data)):
                if 'u_eastward' in ds or 'v_northward' in ds:
                    u_eastward_data=ds['u_eastward'].isel(ocean_time=i).data
                    v_northward_data=ds['v_northward'].isel(ocean_time=i).data

                    u_noise=np.random.uniform(0, noise_scale, u_eastward_data.shape)
                    v_noise=np.random.uniform(0, noise_scale, v_northward_data.shape)

                    u_eastward_noisy=u_eastward_data+(u_eastward_data*noise_scale)+(u_eastward_data*u_noise)
                    v_northward_noisy=v_northward_data+(v_northward_data*noise_scale)+(v_northward_data*v_noise)
                    ds['u_eastward'].loc[{'ocean_time': time_data[i]}] = u_eastward_noisy.astype('float32')
                    ds['v_northward'].loc[{'ocean_time': time_data[i]}] = v_northward_noisy.astype('float32')

                if 'u' in ds or 'v' in ds:
                    # 对 u_eastward 和 v_northward 进行标准化
                    u_data = ds['u'].isel(ocean_time=i).data
                    v_data = ds['v'].isel(ocean_time=i).data
                    u_noise = np.random.uniform(0, noise_scale, u_data.shape)
                    v_noise = np.random.uniform(0, noise_scale, v_data.shape)

                    u_data_noisy = u_data + (u_data * noise_scale) + (u_data * u_noise)
                    v_data_noisy = v_data + (v_data * noise_scale) + (v_data * v_noise)


                    ds['u'].loc[{'ocean_time': time_data[i]}] = u_data_noisy.astype('float32')
                    ds['v'].loc[{'ocean_time': time_data[i]}] = v_data_noisy.astype('float32')


            # 保存为新的 NC 文件
            ds.to_netcdf(output_file)
            pbar.set_description(f"已将处理后的数据保存到 {output_file}")
        except Exception as e:
            pbar.set_description(f" {file_name[:-11]}error: {e}")

            continue




if __name__ == '__main__':
    add_gaussian_noise_to_Lagrange(noise_scale=0.05)
    # add_gaussian_noise_to_Lagrange(noise_scale=0.1)
    # add_gaussian_noise_to_Lagrange(noise_scale=0.2)
    # add_gaussian_noise_to_Lagrange(noise_scale=0.3)
    # add_gaussian_noise_to_Lagrange(noise_scale=0.4)
    # add_gaussian_noise_to_Lagrange(noise_scale=0.5)

    # file0=xr.open_dataset('D:/drift/drift/nc/roms3_npzd_res/his/roms3_npzd_his.nc.20190617')
    # print(file0.u)
    # file1 = xr.open_dataset('D:/drift/drift/nc/roms3_npzd_res/Zscore0406_noise01/roms3_npzd_his.nc.20190617')
    # file2=xr.open_dataset('D:/drift/drift/nc/roms3_npzd_res/Zscore0406_noise02/roms3_npzd_his.nc.20190617')
    # file3=xr.open_dataset('D:/drift/drift/nc/roms3_npzd_res/Zscore0406_noise03/roms3_npzd_his.nc.20190617')
    #
    # print(file0.sel(ocean_time='2019-06-17T01:00:00').v_northward.isel(s_rho=0).data[200,200])
    # print(file1.sel(ocean_time='2019-06-17T01:00:00').v_northward.isel(s_rho=0).data[200,200])
    # print(file2.sel(ocean_time='2019-06-17T01:00:00').v_northward.isel(s_rho=0).data[200,200])
    # print(file3.sel(ocean_time='2019-06-17T01:00:00').v_northward.isel(s_rho=0).data[200,200])
