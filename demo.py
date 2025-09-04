import random
import numpy as np
from datetime import datetime,timedelta
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
'''
Datetime:2025
Author:Yongxiang Chen
'''
from DriftNet_Project.tools import utils
from DriftNet_Project.tools.utils_createDataSet import CreateData
from DriftNet import driftnet
seed=2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def drawing_one_track(task_name,his_arr):

    his_arr = his_arr[:, 2:4]
    his_arr = his_arr.astype(float)
    lonabs = his_arr[:, 0].max() - his_arr[:, 0].min()
    latabs = his_arr[:, 1].max() - his_arr[:, 1].min()
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())

    ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('10m'), facecolor='aliceblue')
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax.set_extent(
        [(his_arr[:, 0].min() - lonabs * 0.1), (his_arr[:, 0].max() + lonabs * 0.1),
         (his_arr[:, 1].min() - latabs * 0.1), (his_arr[:, 1].max() + latabs * 0.1)])


    ax.plot(his_arr[:, 0], his_arr[:, 1], transform=ccrs.Geodetic(), marker='*', linestyle='-',
            markevery=20,color='darkred', linewidth=2, label='Predict ')
    ax.plot(his_arr[-1, 0], his_arr[-1, 1], 'go', markersize=5, transform=ccrs.PlateCarree())
    ax.text(his_arr[-1, 0], his_arr[-1, 1], f'End',
            fontsize=20, color='red', transform=ccrs.PlateCarree())
    # ax.plot(his_arr[:, 0], his_arr[:, 1], transform=ccrs.Geodetic(), marker='+', linestyle='-',
    #         color='red', linewidth=1, label='Real')
    plt.legend(fontsize=(32))
    plt.savefig(f'result/{task_name}.png',
                    dpi = int(300),
                    bbox_inches = 'tight',
                     )
    plt.show()

@torch.no_grad()
def Generator_track(task_name,get_data,start_date,start_time,start_lon,start_lat,his_length:int,model,device):
    model.eval()
    result_his = []
    next_date,next_time,next_lon,next_lat=start_date,start_time,float(start_lon),float(start_lat)

    next_DateTime = datetime.strptime(start_date + ' ' + start_time, '%Y-%m-%d %H:%M:%S')
    next_date = next_DateTime.strftime('%Y-%m-%d')
    next_time = next_DateTime.strftime('%H:%M:%S')
    one_gps_data = [next_date,next_time,next_lon, next_lat]
    result_his.append(one_gps_data)

    for i in range(0,his_length-1):

        BuoyData,OceanData=get_data(one_gps_data)
        BuoyData,OceanData=BuoyData.unsqueeze(0).float(), OceanData.unsqueeze(0).float()
        outputs = model(BuoyData.to(device), OceanData.to(device))
        predict_pos=np.nan_to_num(outputs.cpu().detach().numpy())
        next_lon = next_lon+predict_pos[0][0]
        next_lat = next_lat+predict_pos[0][1]

        next_DateTime = next_DateTime + timedelta(hours=1)
        next_date = next_DateTime.strftime('%Y-%m-%d')
        next_time = next_DateTime.strftime('%H:%M:%S')
        print(one_gps_data)
        one_gps_data = [next_date,next_time,  next_lon, next_lat]
        result_his.append(one_gps_data)

    else:
        result_his=np.array(result_his)
        np.savetxt(f"./result/{task_name}.txt" , result_his, delimiter=' ', fmt='%s')
        drawing_one_track(task_name,result_his)
    return result_his

if __name__ == '__main__':

    # device="cuda" if torch.cuda.is_available() else "cpu"
    device="cpu"
    weight_path="weight/X0415_RunMultiHeadBase_MH8/checkpoint-train.pth"
    get_data = CreateData(Current_Data_path="Data/romsData.nc",
                          att_path='Data/attData.nc')
    model = driftnet(hidden_dim=48, depth=24, num_heads=8).float()
    checkpoint_path=torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint_path['net'], strict=True)


    Generator_track(task_name="his001",
                    get_data=get_data,
                    start_date="2019-06-21",
                    start_time="04:00:00",
                    start_lon="120.229318",
                    start_lat="26.493334",
                    his_length=72,
                    model=model,
                    device=device)

    Generator_track(task_name="his002",
                    get_data=get_data,
                    start_date="2020-05-18",
                    start_time="09:00:00",
                    start_lon="120.449748",
                    start_lat="26.817918",
                    his_length=72,
                    model=model,
                    device=device)

    Generator_track(task_name="his003",
                    get_data=get_data,
                    start_date="2020-08-22",
                    start_time="08:00:00",
                    start_lon="119.046671",
                    start_lat="24.879371",
                    his_length=72,
                    model=model,
                    device=device)