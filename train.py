import os
import sys
import time
import argparse
import math
import numpy as np
from datetime import datetime,timedelta
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import random
from upload_github.tools import utils
from upload_github.tools.utils_ValiDataSet_0304_cuda import CreateData
from DriftNet import driftnet
seed=2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
def list_current_dir_files(path,file_type='.txt'):
    return [entry for entry in os.listdir(path) if entry.endswith(file_type)]
@torch.no_grad()
def evaluate(model, get_data,loss_function,vali_directory,task_name, device, epoch):
    if os.path.exists(f"./result/{task_name}/{epoch:04d}") is False:
        os.makedirs(f"./result/{task_name}/{epoch:04d}")
    vali_files = list_current_dir_files(vali_directory,file_type='.txt')
    model.eval()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    vali_number=0
    for file_n,vali_file in enumerate(vali_files):
        file_path=  os.path.join(vali_directory, vali_file)
        vali_GPS_data=np.loadtxt(file_path,delimiter=' ',skiprows=0,dtype=str)
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
                BuoyData,OceanData=get_data(one_gps_data)
                BuoyData,OceanData=BuoyData.unsqueeze(0).float(), OceanData.unsqueeze(0).float()
                target_data = [[(float(vali_GPS_data[i + 1][2])-float(vali_GPS_data[i][2]) ),
                            (float(vali_GPS_data[i + 1][3])-float(vali_GPS_data[i][3]) )]]
                outputs = model(BuoyData.to(device), OceanData.to(device))
            except Exception as e:
                print(f'{vali_file}-{i} meet a error:',e)
                break
            predict_pos = outputs.cpu().detach().numpy()
            predict_pos=np.nan_to_num(predict_pos)
            arr1[i + 1][0] = predict_pos[0][0] + arr1[i][0]
            arr1[i + 1][1] = predict_pos[0][1] + arr1[i][1]
            loss = loss_function(outputs, torch.tensor(target_data).to(device))
            accu_loss += loss.detach()
            # print('\r',f' [evaluate_epoch {epoch}] {file_n}/60 loss: {loss.item():.6f}',end='')
        else:
            vali_number += len(vali_GPS_data)
            marge_array1 = np.concatenate((vali_GPS_data, arr1), axis=1)
            np.savetxt(f"./result/{task_name}/{epoch:04d}/" + vali_file, marge_array1, delimiter=' ', fmt='%s')
    return accu_loss.item() / (vali_number + 1)

def run_evaluate(task_name,model,get_data,vali_directory,device='cuda',start_epoch=449,stride_epoch=2):
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
    for i, path_checkpoint in enumerate(checkpoint_list[::stride_epoch]):
        checkpoint = torch.load(path_checkpoint)
        state_dict = checkpoint['net']
        the_epoch=checkpoint['epoch']
        if the_epoch <= checkpoint_epoch:
            continue
        print(model.load_state_dict(state_dict, strict=True))
        val_loss = evaluate(model=model,
                        get_data=get_data,
                        loss_function=torch.nn.L1Loss().to(device),
                        vali_directory=vali_directory,
                        task_name= task_name,
                        device=device, epoch=the_epoch)
        check_info={'checkpoint_epoch':the_epoch}
        torch.save(check_info,checkpoint_evaluate_path)
        print(f'{task_name},epoch:{the_epoch},val_loss:{val_loss}')
    print(f'{task_name} val_time', (time.time() - e_s_t) / 60, '分钟')
def train_one_epoch(model, optimizer, loss_function, data_loader,device, epoch):
    model.train()
    mean_loss = torch.zeros(1).to (device)
    for step,data in enumerate(data_loader):
        BuoyData,OceanData,DeltaPosiData = data
        outputs = model(BuoyData.to(device),OceanData.to(device))
        loss = loss_function(outputs, DeltaPosiData.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return mean_loss.item()
def test_one_epoch(model,  loss_function, data_loader,device, epoch):
    model.eval()
    mean_loss = torch.zeros(1).to (device)
    for step,data in enumerate(data_loader):
        BuoyData,OceanData,DeltaPosiData = data
        outputs = model(BuoyData.to(device),OceanData.to(device))
        loss = loss_function(outputs, DeltaPosiData.to(device))
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
    return mean_loss.item()
def main_fun(task_name, model,train_Dataset_path,vali_Dataset_path,epochs,batch_size,device='cuda'):
    print(f'{task_name} start training')
    from upload_github.tools.utils_MyDataSet_LoadCuda_0303 import MyDataSet_0303 as my_dataset
    train_dataset = my_dataset(DataPath=train_Dataset_path, device=device)
    vali_dataset=my_dataset(DataPath=vali_Dataset_path, device=device)
    weights_path = f"./weights/{task_name}"

    os.makedirs(weights_path,exist_ok=True)
    os.makedirs(f"./runs/{task_name}",exist_ok=True)
    os.makedirs(f"./checkpoint/{task_name}",exist_ok=True)
    checkpoint_path=f"./checkpoint/{task_name}/checkpoint-train.pth"
    tb_writer = SummaryWriter(f'./runs/{task_name}')
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    # batch_size=batch_size,
                                                    # pin_memory=True,
                                                    # num_workers=nw,
                                                    collate_fn=train_dataset.collate_fn)
    vali_dataset_loader = torch.utils.data.DataLoader(vali_dataset,
                                                    # batch_size=batch_size,
                                                    # pin_memory=True,
                                                    # num_workers=nw,
                                                    collate_fn=vali_dataset.collate_fn)
    print('train_data_loader',train_data_loader.__len__())
    print('vali_dataset_loader',vali_dataset_loader.__len__())
    model=model.to(device)
    loss_function= utils.euclidean_cosine_Loss(5, 5).to(device)
    pg = [p for p in model.parameters() if p.requires_grad]
    lr, lrf = 0.0001, 0.01
    optimizer = optim.Adam(pg, lr=lr,  weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)



    #从最近的断点加载
    checkpoint={}
    if os.path.exists(checkpoint_path):
        print('load_checkpoint:')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(model.load_state_dict(checkpoint['net'], strict=True))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])  # 加载lr_scheduler
        start_epoch = checkpoint['epoch']
        train_loss_epoch = checkpoint['train_loss_epoch']
        train_loss_min = checkpoint['train_loss_min']
        train_loss_model=checkpoint['train_loss_model']
        vali_loss_epoch = checkpoint['vali_loss_epoch']
        vali_loss_min = checkpoint['vali_loss_min']
        vali_loss_model = checkpoint['vali_loss_min']


    else:
        checkpoint['task_name'] = task_name
        start_epoch = 0
        vali_loss_epoch = 0
        train_loss_epoch = 0
        vali_loss_min = float('inf')
        train_loss_min = float('inf')
        train_loss_model=None
        vali_loss_model=None
    for epoch in range(start_epoch+1,epochs):
        train_loss = train_one_epoch(model=model,
                                        optimizer=optimizer,
                                        loss_function=loss_function,
                                        data_loader=train_data_loader,
                                        device=device, epoch=epoch)
        vali_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     loss_function=loss_function,
                                     data_loader=vali_dataset_loader,
                                     device=device, epoch=epoch)
        scheduler.step()
        print("[epoch {}] train_loss: {}".format(epoch, round(train_loss, 6)))
        if train_loss<train_loss_min:
            train_loss_min=train_loss
            train_loss_epoch=epoch
            train_loss_model=model.state_dict()
        if vali_loss<vali_loss_min:
            vali_loss_min=vali_loss
            vali_loss_epoch=epoch
            vali_loss_model = model.state_dict()

        tb_writer.add_scalar("train_loss", train_loss, epoch)
        tb_writer.add_scalar("vali_loss", vali_loss, epoch)
        tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        checkpoint = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,

            'vali_loss_min':vali_loss_min,
            'vali_loss_epoch': vali_loss_epoch,
            'vali_loss_model': vali_loss_model,
            'train_loss_min':train_loss_min,
            'train_loss_epoch': train_loss_epoch,
            'train_loss_model':train_loss_model,
        }
        torch.save(checkpoint, f"./weights/{task_name}/{epoch:04d}.pth")
        torch.save(checkpoint, checkpoint_path)
        print(f'保存第{epoch}轮模型-{task_name}')

    return checkpoint
def run_min_loss_evaluate(checkpoint,model,Current_Data_path,GPS_file_directory,device,vali_directory,task_name):

    Vali_DataSet = CreateData(Current_Data_path=Current_Data_path,
                              att_path='/home/fjhyj/cyx/roms3_cur.nc.att',
                              ex=10, device=device, vali_directory=GPS_file_directory)
    model.load_state_dict(checkpoint['vali_loss_model'], strict=True)
    evaluate(model=model, get_data=Vali_DataSet,
             loss_function=torch.nn.L1Loss().to(device),
             vali_directory=vali_directory, task_name=task_name, device=device, epoch=checkpoint['vali_loss_epoch'])
    model.load_state_dict(checkpoint['train_loss_model'], strict=True)
    evaluate(model=model, get_data=Vali_DataSet,
             loss_function=torch.nn.L1Loss().to(device),
             vali_directory=vali_directory, task_name=task_name, device=device, epoch=checkpoint['train_loss_epoch'])


def run_noise_exeriment(opt_run_model):
    NoiseSigma=opt_run_model.NoiseSigma
    MultiHead=opt_run_model.MultiHead
    device=opt_run_model.device
    task_name = f'MultiHead{MultiHead}_Noise{NoiseSigma}'
    model = driftnet(hidden_dim=48, depth=24, num_heads=MultiHead).float()
    Current_Data_path = f'./Data/simpl0309_noise_{NoiseSigma}.nc'
    train_Dataset_path = f'./Data/gaussian_{NoiseSigma}/real_train_data/'
    vali_Dataset_path = f'./Data/gaussian_{NoiseSigma}/real_vali_data/'
    GPS_file_directory = f'./Data/GPS_vali/'

    checkpoint = main_fun(task_name,model,train_Dataset_path,vali_Dataset_path,epochs=500,batch_size=32,device=device)
    run_min_loss_evaluate(checkpoint,model ,Current_Data_path, GPS_file_directory, device, GPS_file_directory, task_name)
if __name__ == '__main__':

    device = 'cuda'
    parser_run_model = argparse.ArgumentParser()
    parser_run_model.add_argument('--NoiseSigma', type=str,default=0,help='NoiseSigma')
    parser_run_model.add_argument('--MultiHead', type=int,default=1,help='MultiHead')
    parser_run_model.add_argument('--device', type=str,default='cuda', help='device')
    opt_run_model = parser_run_model.parse_args()
    run_noise_exeriment(opt_run_model)

# python X0330_RunBase.py --NoiseSigma=20 --MultiHead=6 --device=cuda:1