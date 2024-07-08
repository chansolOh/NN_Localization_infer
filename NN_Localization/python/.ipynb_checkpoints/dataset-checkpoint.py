import torch.utils.data as Data
import os
import numpy as np
import PIL

class Seg_Cnn_Dataset(Data.Dataset):
    def __init__(self, path, dir_range,transform =None, augmentation=None):
        super().__init__()
        dir_list = os.listdir(path)[dir_range[0]:dir_range[1]]
        self.map_img_path_list = []
        self.sensor_img_path_list = []
        self.label_path_list = []
        for dir_name in dir_list:
            map_img_path    = os.path.join(path, dir_name,"map_img")
            sensor_img_path = os.path.join(path, dir_name,"sensor_img")
            label_path      = os.path.join(path, dir_name,"label")

            self.map_img_path_list    += [os.path.join(map_img_path,i) for i in sorted(os.listdir(map_img_path)) if i.split('.')[-1] in ['jpg','png']]
            self.sensor_img_path_list += [os.path.join(sensor_img_path,i) for i in sorted(os.listdir(sensor_img_path)) if i.split('.')[-1] in ['jpg','png']]
            self.label_path_list      += [os.path.join(label_path,i) for i in sorted(os.listdir(label_path))if i.split('.')[-1] in ['txt']]

        self.transform = transform
        self.augmentation = augmentation
        
        self.map_size=320

        self.pos_min =-20
        self.pos_max = 20
        self.angle_min = -180
        self.angle_max = 180
        self.angle_class_num = 72
        self.angle_interval = int((self.angle_max-self.angle_min)/self.angle_class_num)
        self.rotated_pos_min = 0#utils.rotate_label(np.array([self.pos_min,self.pos_min]),45/180*np.pi)[0]
        
    def __getitem__(self, index):
        map_img     = np.array(PIL.Image.open(self.map_img_path_list[index]))
        crop_img    = np.array(PIL.Image.open(self.sensor_img_path_list[index]))
        label       = np.loadtxt(self.label_path_list[index], delimiter=",")

        loc,rot = label[:2],label[-1]

        if self.augmentation!=None:
            deg = np.random.rand(1)*360 - 180
            map_img,crop_img, loc, rot =  self.augmentation(map_img,crop_img,loc,rot,deg[0])
            
        loc = loc - self.rotated_pos_min
        rot = np.eye(self.angle_class_num)[int((rot/np.pi*180-self.angle_min+self.angle_interval/2   )// self.angle_interval )-1]


        if self.transform!=None:
            map_img = self.transform(map_img)
            crop_img = self.transform(crop_img)

        return {'map_img':map_img, 
                'crop_img':crop_img, 
                'location':loc,
                'rotation':rot}

    def __len__(self):
        return len(self.map_img_path_list)
