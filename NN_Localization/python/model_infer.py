import torch
import torchvision.transforms as transforms
import numpy as np
import models
import utils
from PIL import Image

model_name = ["seq_c","seq","static"][0]
model_dict = {
    "seq_c":{
        "class_num":36,
        "weight" : "../weights/GSC_C_invert_obs_loc10_rot90_76ep.pt",
        "deg_min":-90,
        "deg_interval":5,
    },
    "seq":{
        "class_num":36,
        "weight" : "../weights/GSC_invert_obs_loc10_rot90_116ep.pt",
        "deg_min":-90,
        "deg_interval":5,
    },
    "static":{
        "class_num":72,
        "weight" : "../weights/GSC_no_invert_newmap_loc20_89ep.pt",
        "deg_min":-180,
        "deg_interval":5,
    },
}

### model structure select
if model_name =="seq_c":
    model = models.NN_Localization_Conf(model_dict[model_name]["class_num"]).cuda()
else:
    model = models.NN_Localization(model_dict[model_name]["class_num"]).cuda()

#### model weight load
model.load_state_dict(torch.load(model_dict[model_name]["weight"])["model_state_dict"] )
model.eval()

### transform numpy to torch tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    ])

### img load 
map_img_arr     = np.array(Image.open("./test_data/map/00000.png").convert("RGB"))
sensor_img_arr  = np.array(Image.open("./test_data/sensor/00000.png").convert("RGB"))
label_arr       = np.loadtxt("./test_data/label/00000.txt", delimiter=",")

### channel check & select 1ch
if map_img_arr.ndim == 3    : map_img_arr = map_img_arr[...,0]
if sensor_img_arr.ndim == 3 : sensor_img_arr = sensor_img_arr[...,0]

map_img_input      = transform(map_img_arr)[None,...].cuda()
sensor_img_input   = transform(sensor_img_arr)[None,...].cuda()


### prediction
if model_name== "seq_c":
    pred_pos, pred_rot, pos_conf, rot_conf = model(map_img_input,sensor_img_input)
    pred_pos, pred_rot, pos_conf, rot_conf = \
    pred_pos.detach().cpu().numpy(), pred_rot.detach().cpu().numpy(), pos_conf.detach().cpu().numpy(), rot_conf.detach().cpu().numpy()
else:
    pred_pos, pred_rot = model(map_img_input,sensor_img_input)
    pred_pos, pred_rot = pred_pos.detach().cpu().numpy(), pred_rot.detach().cpu().numpy()
rot_deg = np.argmax(pred_rot, axis=1)*model_dict[model_name]["deg_interval"] \
    + model_dict[model_name]["deg_min"] +model_dict[model_name]["deg_interval"]/2

### ICP
pred_point = utils.restore(sensor_img_arr,pred_pos[0], rot_deg[0]/180*np.pi)
x,y,angle, points = utils.ICP_tune_loc(map_img_arr, pred_point[:2],pred_pos[0])
ICP_pred_point = (points.T[:-1] + pred_pos.T).round().astype(np.int16)

x,y = pred_pos[0]+[x,y]
angle += rot_deg[0]

### output : x, y, angle
### ICP_pred_point = prediction sensor point(ICP) 





######################## output check
import matplotlib.pyplot as plt
print("####label####")
print("x,y : ",label_arr[:2])
print("angle : ",label_arr[2]/np.pi*180)
print("####prediction###")
print("x,y : ",x,y)
print("angle : ",angle)
plt.imshow(map_img_arr)
plt.scatter(ICP_pred_point[0],ICP_pred_point[1],c="r", s=1)
plt.scatter(pred_point[0],pred_point[1],c="b", s=1, alpha=0.7)
plt.show()






