import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import open3d as o3d

def rotate_label(location,deg): # inverse clock direction(image)

    mat = np.array([[np.cos(-deg),-np.sin(-deg)],
                    [np.sin(-deg),np.cos(-deg)]])
    loc = mat.dot(location[:,None])
    return loc.T[0]
def rotate(map_img,sensor_img,location,rotation,deg):
    loc = rotate_label(location,deg/180*np.pi)
    w,h=map_img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1) # inverse clock direction
    rotated_map_img = cv2.warpAffine(map_img, M, (w, h), borderMode = cv2.BORDER_REFLECT)
    rotated_sensor_img = cv2.warpAffine(sensor_img, M, (w, h), borderMode = cv2.BORDER_CONSTANT, borderValue=(0))
    return rotated_map_img, rotated_sensor_img, loc, rotation

def rotate_only_map(map_img,sensor_img, location,rotation,deg):
    loc = rotate_label(location,deg/180*np.pi)
    w,h=map_img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1) # inverse clock direction
    rotated_map_img = cv2.warpAffine(map_img, M, (w, h), borderMode = cv2.BORDER_REFLECT)
    rot = rotation - deg/180*np.pi
    if rot< -np.pi: rot+= np.pi*2
    elif rot > np.pi: rot-= np.pi*2
    return rotated_map_img, sensor_img, loc, rot

def rotation_tuner(sensor_img,location,rot,Range=[-90,90]):

    w,h=sensor_img.shape
    rot_deg = rot/np.pi*180
    deg_min,deg_max = Range

    if   rot_deg < deg_min:
        over_deg = rot_deg-deg_min
        deg = (np.random.rand(1)*(deg_max-deg_min)-over_deg).round().astype(np.int32)[0]
        M = cv2.getRotationMatrix2D((w/2, h/2), -deg, 1)
        rotated_sensor_img = cv2.warpAffine(sensor_img, M, (w, h), borderMode = cv2.BORDER_REFLECT)

        loc = rotate_label(location,-deg/180*np.pi)

        return rotated_sensor_img, loc,rot+deg/180*np.pi

    elif rot_deg > deg_max:
        over_deg = rot_deg-deg_max
        deg = (np.random.rand(1)*(deg_max-deg_min)+over_deg).round().astype(np.int32)[0]
        M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1)
        rotated_sensor_img = cv2.warpAffine(sensor_img, M, (w, h), borderMode = cv2.BORDER_REFLECT)

        loc = rotate_label(location,deg/180*np.pi)
        
        return rotated_sensor_img, loc,rot-deg/180*np.pi
    
    else :return sensor_img, location,rot


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class Hook:
    def __init__(self):
        self.Input = None
        self.Output = None
        self.module_name = None
    def forward_hook(self, module, input, output):
        self.Input = input
        self.Output = output
        self.module_name = module

def restore(target_img, loc,rot):
    pos = np.flip(np.argwhere(np.array(target_img)>=0.1),axis=1)
    x,y = loc
    angle = rot
    target = pos-160
    target = np.hstack((target, np.ones((target.shape[0],1))))
    rot_mat = np.array( [[ np.cos(angle), -np.sin(angle), 0  ],
                         [ np.sin(angle),  np.cos(angle), 0  ],
                         [0,0,1]])
    trans_mat = np.array([[1,0,x+160],
                          [0,1,y+160],
                          [0,0,1]])
    return trans_mat.dot(rot_mat).dot(target.T)


def cls_to_func(cls):
    func_list = np.array([np.arcsin, np.arccos,np.arcsin, np.arccos, np.arcsin])
    return func_list[np.argmax(cls,axis=1)]


def ICP(map_img, pred_point): # pred point = x,y   , map_img shape = (320,320)
    if map_img.max()<=1:
        map_point = map_img.numpy()*255
    map_point = np.argwhere(map_img<110).T

    pred_point = np.vstack((pred_point-160, np.zeros(pred_point.shape[-1]))).T 
    map_point = np.vstack((np.flip(map_point,axis=0)-160, np.zeros(map_point.shape[-1]))).T 

    pred_pcd = o3d.geometry.PointCloud()
    map_pcd = o3d.geometry.PointCloud()

    pred_pcd.points = o3d.utility.Vector3dVector(pred_point)
    map_pcd.points = o3d.utility.Vector3dVector(map_point)

    icp_result = o3d.pipelines.registration.registration_icp(
        pred_pcd, map_pcd, 10,np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    TF = icp_result.transformation

    x,y = TF[0,-1],TF[1,-1]
    rot_mat = TF[:3,:3].copy()
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=True)
    return x,y,euler[-1], np.asarray(pred_pcd.transform(TF).points)+160

def ICP_tune_loc(map_img, pred_point, loc): # pred point = x,y   , map_img shape = (320,320)
    if map_img.max()<=1:
        map_point = map_img.numpy()*255
    map_point = np.argwhere(map_img<110).T

    pred_point = np.vstack((pred_point-160 -loc[:,None], np.zeros(pred_point.shape[-1]))).T 
    map_point = np.vstack((np.flip(map_point,axis=0)-160 - loc[:,None], np.zeros(map_point.shape[-1]))).T 

    pred_pcd = o3d.geometry.PointCloud()
    map_pcd = o3d.geometry.PointCloud()

    pred_pcd.points = o3d.utility.Vector3dVector(pred_point)
    map_pcd.points = o3d.utility.Vector3dVector(map_point)

    icp_result = o3d.pipelines.registration.registration_icp(
        pred_pcd, map_pcd, 10,np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    )

    TF = icp_result.transformation

    x,y = TF[0,-1],TF[1,-1]
    rot_mat = TF[:3,:3].copy()
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=True)
    return x,y,euler[-1], np.asarray(pred_pcd.transform(TF).points)+160