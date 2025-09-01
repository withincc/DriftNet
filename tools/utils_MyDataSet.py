import os
from pathlib import Path
import utils
import torch
from torch.utils.data import Dataset
def list_current_dir_pt(path):
    files = []
    for entry in os.listdir(path):  # '.' 表示当前工作目录
        if entry.endswith('.pt'):
            files.append(entry)
    return files
class MyDataSet(Dataset):

    def __init__(self, DataPath: str):
        self.DriftDataPath: str = DataPath + 'Drift_data/'
        self.OceanDataPath: str=DataPath+'Ocean_data/'
        self.DeltaPosiDataPath: str=DataPath+'DeltaPosi_data/'

        self.file_name_list=list_current_dir_pt(self.DriftDataPath)

        c= self.get_all_files_absolute_paths(self.DriftDataPath)
        o= self.get_all_files_absolute_paths(self.OceanDataPath)
        d=self.get_all_files_absolute_paths(self.DeltaPosiDataPath)
        if not (len(c)>0 and len(o)>0 and len(d)>0):
            print('数据集为空')
            raise Exception('数据集为空')


    def __len__(self):
        return len(self.file_name_list)
    def __getitem__(self, item):
        DriftData = torch.load(self.DriftDataPath+self.file_name_list[item], weights_only=True).float()
        OceanData = torch.load(self.OceanDataPath+self.file_name_list[item],weights_only=True).float()
        DeltaPosiData = torch.load(self.DeltaPosiDataPath+self.file_name_list[item],weights_only=True).float()

        return DriftData,OceanData, DeltaPosiData
    @staticmethod
    def collate_fn(batch):
        DriftData,OceanData, DeltaPosiData = tuple(zip(*batch))

        DriftData = torch.cat(DriftData, dim=0)
        OceanData = torch.cat(OceanData, dim=0)
        DeltaPosiData = torch.cat(DeltaPosiData, dim=0)

        return DriftData,OceanData,DeltaPosiData
    def get_all_files_absolute_paths(self,directory):
        absolute_paths = []
        # 遍历目录
        for root, dirs, files in os.walk(directory):
            for file in files:
                # 使用os.path.join将根目录和文件名合并，然后使用os.path.abspath获取绝对路径
                if file.endswith('.pt'):
                    absolute_paths.append(os.path.abspath(os.path.join(root, file)))
        return absolute_paths

if __name__ == '__main__':
    task_name = current_path = Path.cwd().name
    print('task_name：',task_name)
    train_data_set=MyDataSet('Data/test_data/')
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=1,
                                               shuffle=True,
                                               collate_fn=train_data_set.collate_fn)
    print(train_loader.__len__())
    for i, (CurData,OceanData, DeltaPosiData) in enumerate(train_loader):
        print('CurData:',CurData.shape)
        print('OceanData:',OceanData.shape)
        print('DeltaPosiData:',DeltaPosiData.shape)
        # print('OceanData',OceanData[0,:,0,0])
        break



