import os
import numpy as np
import xarray as xr
from tqdm import tqdm


def add_noise_to_grid(noise_scale):
    np.random.seed(2025)
    ds = xr.open_dataset( 'G:/opendrift000.nc')
    time_list=ds.time.data
    output_file = f'G:/opendrift{int(noise_scale*100):03d}.nc'
    for i in tqdm(range(len(time_list))):
        u_data = ds['u'].isel(time=i).data
        v_data = ds['v'].isel(time=i).data
        u_noise = np.random.uniform(0, noise_scale, u_data.shape)
        v_noise = np.random.uniform(0, noise_scale, v_data.shape)
        u_data_noisy = u_data + (u_data * noise_scale) + (u_data * u_noise)
        v_data_noisy = v_data + (v_data * noise_scale) + (v_data * v_noise)
        ds['u'].loc[{'time': time_list[i]}] = u_data_noisy
        ds['v'].loc[{'time': time_list[i]}] = v_data_noisy
    ds.to_netcdf(output_file)
if __name__ == '__main__':
    add_noise_to_grid(noise_scale=0.05)
    add_noise_to_grid(noise_scale=0.1)
    add_noise_to_grid(noise_scale=0.2)
    add_noise_to_grid(noise_scale=0.3)
