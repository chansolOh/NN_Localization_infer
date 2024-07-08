import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import sys
sys.path.append("./python")
import models
import utils
import cv2
import time




LASER_MAX_DIST = 6.0
MAP_RESOLUTION = 0.05 

USE_SQUARE_OBSTACLE  =   1
USE_CIRCLE_OBSTACLE  =   1
MAX_NUM_OBSTACLE     =   6
OBSTACLE_SIZE_MIN    =   2
OBSTACLE_SIZE_MAX    =   12

class Display():
    def __init__(self, win):


        self.map_scale = None
        self.map_max = 720
        self.map_dir_path = "../maps"
        self.crop_img_size = 320
        self.crop_img_scale = 1.5
        self.lidar_angle = 270
        self.lidar_dist = LASER_MAX_DIST/MAP_RESOLUTION
        self.sensor_view_size = 240
        self.input_map_view_size = 240
        self.pred_view_size = int(self.crop_img_size*self.crop_img_scale)

        self.model_type = ["seq","seq_c","stat"][1]

        if self.model_type in ["seq","seq_c"]:
            self.model_angle_min = -90
            self.model_angle_max = 90
            self.model_class_num = 36
        else:
            self.model_angle_min = -180
            self.model_angle_max = 180
            self.model_class_num = 72
        self.model_angle_interval = int((self.model_angle_max - self.model_angle_min)/self.model_class_num)
        self.model_rotated_pos_min = 0
        self.laser_boldering = True
        self.laser_gaussian = True
        
        self.obstacle_num = 40
        self.obs_move_scale = 20
        self.obs_scale = 4



        self.win = win
        self.can = tk.Canvas(self.win, width = 640,height=640, bg="white")
        self.can.place(x=200,y=50)
        self.can.bind("<B1-Motion>", self.drag)
        self.can.bind("<Button-1>", self.drag)
        
        self.win.bind("<KeyPress>",self.key_press)



        self.can_crop = tk.Canvas(self.win, width = self.crop_img_size*self.crop_img_scale,
                                  height = self.crop_img_size*self.crop_img_scale,
                                  bg = "white")
        self.can_crop.place(x=1400, y=50)
        self.can_pred = tk.Canvas(self.win, width = self.crop_img_size*self.crop_img_scale,
                                  height = self.crop_img_size*self.crop_img_scale,
                                  bg = "white")
        self.can_pred.place(x=1400, y= 550)

        self.can_sensor = tk.Canvas(self.win, width = self.sensor_view_size,
                                  height = self.sensor_view_size,
                                  bg = "white")
        self.can_sensor.place(x=1150, y= 550)

        self.can_input_map = tk.Canvas(self.win, width = self.input_map_view_size,
                                  height = self.input_map_view_size,
                                  bg = "white")
        self.can_input_map.place(x=1150, y= 800)
        
        self.err_string = tk.StringVar()
        self.label_err = tk.Label(self.win, textvariable=self.err_string, font=("Arial", 10), background="white")
        self.label_err.place(x=1000,y=450)

        self.fig, self.ax = plt.subplots(figsize = (6,2))
        self.line = self.ax.bar([],[])
        
        self.can_plt = FigureCanvasTkAgg(self.fig,master = self.win)
        self.can_plt.draw()
        self.can_plt.get_tk_widget().place(x=530,y=800)

        self.fig_conf, self.ax_conf = plt.subplots(figsize = (2,5))
        self.line_conf = self.ax_conf.bar([],[])
        
        self.can_plt_conf = FigureCanvasTkAgg(self.fig_conf,master = self.win)
        self.can_plt_conf.draw()
        self.can_plt_conf.get_tk_widget().place(x=1200,y=30)

        # 버튼 생성
        self.button_map_load = tk.Button(self.win, text="Load Image", command=self.load_image)
        self.button_map_load.place(x = 20,y=50)

        self.button_loop = tk.Button(self.win, text="Loop start", command=self.Loop_button)
        self.button_loop.place(x = 20,y=100)

        # self.button_predict = tk.Button(self.win, text="Predict", command=self.predict)
        # self.button_predict.place(x = 1000,y=900)

        self.map_img_can=None
        self.crop_img_can=None
        self.sensor_img_can = None
        self.input_map_img_can=None
        self.input_map_img = None
        self.pred_img_can = None

        self.model_input_map = None
        self.model_input_sensor = None

        self.class_rot = None
   

        self.robot_pos = np.array([100,100])
        self.robot_pos_old = np.array([100,100])
        self.robot_angle = 0
        self.robot_angle_old = 0
        self.robot_turn_old = 0
        self.robot_scale = 10
        self.robot_head_length = 20
        self.robot_turn_scale = 10
        self.robot_forward_back_scale = 4
        
        self.Loop_flag = False
        self.d_xy = None
        self.eo_xy = 0
        self.eo_deg = 0
        self.dt = 0.03
        self.task_num=1
        self.task_buf_num=0
        self.task_buf = np.array([[300,200],
                         [300,250],
                         [300,300],
                         [300,500],
                         [300,700],])

        self.create_robot()
        self.model_init()

    def class_hist_display(self):
        self.ax.cla()
        x = np.arange(self.model_angle_min,self.model_angle_max,int((self.model_angle_max-self.model_angle_min)/self.model_class_num))
        self.ax.bar(x,self.class_rot)
        self.can_plt.draw()

        self.ax_conf.cla()
        self.ax_conf.set_ylim([0,1])
        x = ["loc_c", "rot_c"]
        self.ax_conf.bar(x,np.hstack((self.loc_conf[0],self.rot_conf[0])),color =["red","blue"] )
        self.can_plt_conf.draw()

    def model_init(self):
        self.config_dict={
            "stat":{
                "augmentation_range":360,
                "class_num":72,
                #"weight" : torch.load("./weights/GAT_seg_cnn_sgd_chained1_loc20_64ep.pt")["model_state_dict"],
                "weight" : torch.load("./weights/GSC_no_invert_newmap_loc20_89ep.pt")["model_state_dict"],
                
            },
            "seq":{
                "augmentation_range":180,
                "class_num":36,
                # "weight" : torch.load("./weights/GAT_seg_cnn_sgd_chained1_loc20_rot90_65ep.pt")["model_state_dict"],
                # "weight" : torch.load("./weights/GSC_invert_newmap_loc20_rot90_52ep.pt")["model_state_dict"],
                # "weight" : torch.load("./weights/GSC_no_invert_newmap_loc20_rot90_53ep_old.pt")["model_state_dict"],
                 "weight" : torch.load("./weights/GSC_invert_obs_loc10_rot90_116ep.pt")["model_state_dict"],
                
            },
            "seq_c":{
                "augmentation_range":180,
                "class_num":36,
                # "weight" : torch.load("./weights/GSC_C_invert_newmap_loc20_rot90_99ep.pt")["model_state_dict"],
                # "weight" : torch.load("./weights/GSC_C_invert_newmap_loc20_rot90_loss_invert_82ep.pt")["model_state_dict"],
                # "weight" : torch.load("./weights/GSC_C_invert_obs_loc10_rot90_76ep.pt")["model_state_dict"],
                "weight" : torch.load("./weights/GSC_C_invert_obs_loc10_rot90_final_100ep.pt")["model_state_dict"],

            }

        }

        if self.model_type=="seq":
            self.model = models.NN_Localization(self.config_dict[self.model_type]["class_num"]).cuda()
            self.model.load_state_dict(self.config_dict[self.model_type]["weight"])
        elif self.model_type == "seq_c":
            self.model = models.NN_Localization_Conf(self.config_dict[self.model_type]["class_num"]).cuda()
            self.model.load_state_dict(self.config_dict[self.model_type]["weight"])
        elif self.model_type == "stat":
            self.model = models.NN_Localization(self.config_dict[self.model_type]["class_num"]).cuda()
            self.model.load_state_dict(self.config_dict[self.model_type]["weight"])
            
        self.model.eval()
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))])
        self.transform_gaussian = transforms.Compose(
            [transforms.ToTensor(),
             transforms.GaussianBlur(kernel_size=5,sigma=1),
            transforms.Normalize((0.5), (0.5))])
        
        
    def predict(self):
        if self.model_input_map!=None:
            st = time.time()
            model_input_map_arr = np.array(self.model_input_map)
            if model_input_map_arr.ndim ==3 : model_input_map_arr=model_input_map_arr[...,0]
            model_input_sensor_arr = np.array(self.model_input_sensor)

            if self.laser_gaussian:
                input_sensor = self.transform_gaussian(model_input_sensor_arr)[None,...]
            else:
                input_sensor = self.transform(model_input_sensor_arr)[None,...]

            input_map = self.transform(model_input_map_arr)[None,...]
            
            self.loc_conf,self.rot_conf = np.array([0]),np.array([0])
            if self.model_type == "seq_c":
                out_loc,out_rot,self.loc_conf,self.rot_conf = self.model(input_map.cuda(),input_sensor.cuda())
                self.loc_conf, self.rot_conf = self.loc_conf.detach().cpu().numpy(), self.rot_conf.detach().cpu().numpy()
            else:
                out_loc,out_rot = self.model(input_map.cuda(),input_sensor.cuda())
            out_loc,out_rot = out_loc.detach().cpu().numpy(), out_rot.detach().cpu().numpy()
            
            self.class_rot = out_rot[0]
            rot_deg=np.argmax(out_rot, axis=1)*self.model_angle_interval + self.model_angle_min+self.model_angle_interval/2
 

            pred_point = utils.restore(model_input_sensor_arr,out_loc[0]+self.model_rotated_pos_min, rot_deg[0]/180*np.pi)

            # x,y,angle, points = utils.ICP(model_input_map_arr,pred_point[:-1])
            x,y,angle, points = utils.ICP_tune_loc(model_input_map_arr, pred_point[:2],out_loc[0])
            ICP_pred_point = (points.T[:-1] + out_loc.T).round().astype(np.int16)
            
            if self.model_type in ["seq","seq_c"]:
                robot_angle_err = (self.robot_angle -self.robot_angle_old)
                if robot_angle_err>180:
                    robot_angle_err -= 360
                elif robot_angle_err<-180:
                    robot_angle_err += 360
                rot_err = np.abs(rot_deg[0]+angle - robot_angle_err).round(4)
                loc_point = self.trans_point(self.robot_pos/self.map_scale-self.robot_pos_old/self.map_scale,[0,0],-self.robot_angle)
                loc_err =  np.sqrt(np.sum((out_loc[0] + [x,y]  - loc_point)**2)).round(4)
                self.err_string.set(f"rot_err : {rot_err }\n loc_err : {loc_err}\n \n"+\
                                    f"rot_conf : {self.rot_conf[0].round(4)}\n loc_conf : {self.loc_conf[0].round(4)}")
            else:
                rot_err = np.abs(rot_deg[0]+angle -self.robot_angle).round(4)
                if rot_err >=360: rot_err-=360
                elif rot_err <=-360: rot_err +=360
                loc_err =  np.sqrt(np.sum((out_loc[0] + [x,y]  - (self.robot_pos/self.map_scale - self.robot_pos_old/self.map_scale))**2)).round(4)
                self.err_string.set(f"rot_err : {rot_err }\n loc_err : {loc_err}")

            # print("icp rot :",rot_deg[0] + angle)
            # print("icp_loc :",out_loc[0]/2 + [x,y])
            # print(out_loc)
            # print(x,y)
            # print("label_rot:", self.robot_angle-self.robot_angle_old)
            # print("label_loc:", self.robot_pos)
            pred_point = pred_point[:-1].round().astype(np.int16)

            if self.pred_img_can is not None:
                self.can_pred.delete(self.pred_img_can)
            input_map_arr = np.tile(model_input_map_arr[...,None],(1,1,3))

            input_map_arr[pred_point[1],pred_point[0],0] = 150
            input_map_arr[pred_point[1],pred_point[0],1] = 220
            input_map_arr[pred_point[1],pred_point[0],2] = 200

            input_map_arr[ICP_pred_point[1],ICP_pred_point[0],0] = 255
            input_map_arr[ICP_pred_point[1],ICP_pred_point[0],1] = 0
            input_map_arr[ICP_pred_point[1],ICP_pred_point[0],2] = 0

    
            self.pred_img_tk = ImageTk.PhotoImage(Image.fromarray(input_map_arr).resize((self.pred_view_size, self.pred_view_size)))
            self.pred_img_can = self.can_pred.create_image(0, 0, anchor=tk.NW, image=self.pred_img_tk)
            self.can_pred.tag_raise(self.robot_body_can_pred)
            self.can_pred.tag_raise(self.robot_head_can_pred)
            #print("pred_time :", time.time()-st)
            self.class_hist_display()

    def trans_point(self,target_pt,center_pos,rot=0,pos=[0,0]):

        th = rot/180*np.pi
        if isinstance(target_pt,list):
            target_pt = np.array(target_pt)
        if target_pt.ndim ==1:
            target_pt = target_pt[None,:]

        target_pt = np.hstack((target_pt,np.ones((target_pt.shape[0],1))))

        centering_mat = np.array([[1,0,-center_pos[0]],
                                  [0,1,-center_pos[1]],
                                  [0,0,1]])
        rot_mat = np.array([[np.cos(th), -np.sin(th),0],
                            [np.sin(th),  np.cos(th),0],
                            [0,0,1]])
        restore_mat = np.array([[1,0,pos[0]],
                                [0,1,pos[1]],
                                [0,0,1]])

        return restore_mat.dot(rot_mat).dot(centering_mat).dot(target_pt.T)[:-1].T
    
    def create_robot(self,color='blue'):
        self.robot_body_can = self.create_circle(self.can,self.robot_pos,self.robot_scale, fill=color, color=color)
        self.robot_head_can = self.can.create_line(self.robot_pos[0],self.robot_pos[1], self.robot_pos[0]+self.robot_scale*2, self.robot_pos[1],width = 2,fill=color)
        self.robot_body_can_crop = self.create_circle(self.can_crop,[self.crop_img_size*self.crop_img_scale/2,self.crop_img_size*self.crop_img_scale/2],
                                                 self.robot_scale*self.crop_img_scale, fill=color,color=color)
        self.robot_head_can_crop = self.can_crop.create_line(self.crop_img_size*self.crop_img_scale/2,
                                                        self.crop_img_size*self.crop_img_scale/2,
                                                        self.crop_img_size*self.crop_img_scale/2+self.robot_head_length,
                                                        self.crop_img_size*self.crop_img_scale/2,
                                                   width = 4,fill=color)
        
        self.robot_body_can_pred = self.create_circle(self.can_pred,[self.crop_img_size*self.crop_img_scale/2,self.crop_img_size*self.crop_img_scale/2],
                                                 self.robot_scale*self.crop_img_scale, fill=color,color=color)
        self.robot_head_can_pred = self.can_pred.create_line(self.crop_img_size*self.crop_img_scale/2,
                                                        self.crop_img_size*self.crop_img_scale/2,
                                                        self.crop_img_size*self.crop_img_scale/2+self.robot_head_length,
                                                        self.crop_img_size*self.crop_img_scale/2,
                                                   width = 4,fill=color)

        self.robot_lidar_range = self.create_circle(self.can_crop,[self.crop_img_size*self.crop_img_scale/2,self.crop_img_size*self.crop_img_scale/2],
                           scale=self.lidar_dist*2*self.crop_img_scale,  width=1, color="#2288cf")
        self.set_robot_pose(rotation=self.robot_angle)


    def create_circle(self, can,pos,scale, width=2,color='red', fill=None):
        return can.create_oval(pos[0]-scale*0.5, pos[1]-scale*0.5, pos[0]+scale*0.5, pos[1]+scale*0.5,
                             width=width, fill = fill, outline=color)
    def set_robot_pose(self,position=None, rotation=0):
        if not isinstance(position,np.ndarray) and position==None : position = self.robot_pos
        self.robot_pos = position
        self.can.coords(self.robot_body_can,[position[0]-self.robot_scale*0.5 ,
                                            position[1]-self.robot_scale*0.5 ,
                                            position[0]+self.robot_scale*0.5 ,
                                            position[1]+self.robot_scale*0.5 , ])
        
        head_point = np.array(self.can.coords(self.robot_head_can)).reshape((2,2))
        new_head_point = self.trans_point(head_point,head_point[0], rotation,
                                           position)
        self.can.coords(self.robot_head_can,new_head_point.flatten().tolist())

        head_point_crop = np.array(self.can_crop.coords(self.robot_head_can_crop)).reshape((2,2))
        new_head_point_crop = self.trans_point(head_point_crop,head_point_crop[0], rotation,
                                           np.array([self.crop_img_size*self.crop_img_scale/2,self.crop_img_size*self.crop_img_scale/2]))
        self.can_crop.coords(self.robot_head_can_crop,new_head_point_crop.flatten().tolist())


        if self.model_type in ["seq","seq_c"]:
            if rotation != self.robot_turn_old:
                head_point_crop = np.array(self.can_pred.coords(self.robot_head_can_pred)).reshape((2,2))
                new_head_point_crop = self.trans_point(head_point_crop,head_point_crop[0], rotation-self.robot_turn_old,
                                            np.array([self.crop_img_size*self.crop_img_scale/2,self.crop_img_size*self.crop_img_scale/2]))
                self.can_pred.coords(self.robot_head_can_pred,new_head_point_crop.flatten().tolist())
                self.robot_turn_old = rotation

            
        else:

            self.can_pred.coords(self.robot_head_can_pred,new_head_point_crop.flatten().tolist())


        self.can.tag_raise(self.robot_body_can)
        self.can.tag_raise(self.robot_head_can)

    def gen_obs(self,init):
        if init:
            img_arr = np.array(self.model_map_img_org)
            self.obs_all_idx = np.argwhere(img_arr[...,0]>250)
            random_idx = np.random.choice(len(self.obs_all_idx),self.obstacle_num,replace=False)
            self.obs_pos = self.obs_all_idx[random_idx]
            x,y = np.meshgrid(np.arange(-self.obs_scale,self.obs_scale), np.arange(-self.obs_scale,self.obs_scale))
            x,y = x.flatten(), y.flatten()
            self.obs_mesh = np.vstack((y,x))

        else:
            img_arr = np.array(self.model_map_img_org)
            self.obs_pos = (self.obs_pos+(np.random.rand(self.obs_pos.shape[0],self.obs_pos.shape[1])-0.5)*np.random.randint(self.obs_move_scale) ).T.round().astype(np.int16) #shape = (2, point_n)
            self.obs_new_pos = np.argwhere(img_arr[self.obs_pos[0],self.obs_pos[1],0]>250)

            self.gen_pos = self.obs_all_idx[np.random.choice(len(self.obs_all_idx), len(self.obs_pos.T) - len(self.obs_new_pos), replace=False)]
            self.obs_pos = np.vstack((self.obs_pos.T[self.obs_new_pos[:,0]], self.gen_pos ))

        circle_idx = (self.obs_pos[...,None] + self.obs_mesh[None,...]).transpose(2,0,1).reshape(-1,2)
        img_arr[circle_idx.T[0],circle_idx.T[1],0] = 0
        img_arr[circle_idx.T[0],circle_idx.T[1],1] = 0
        img_arr[circle_idx.T[0],circle_idx.T[1],2] = 0

        if self.d_xy is not None:
            idx = self.d_xy[::-1][:,None] + self.obs_mesh
            img_arr[idx[0],idx[1],0] = 150
            img_arr[idx[0],idx[1],1] = 200
            img_arr[idx[0],idx[1],2] = 30

        self.map_img_org = Image.fromarray(img_arr)
        if not init :
            if self.map_img_can is not None:
                self.can.delete(self.map_img_can)
            self.map_img_resize = self.map_img_org.resize((self.map_w,self.map_h))
            self.map_img_tk = ImageTk.PhotoImage(self.map_img_resize)
            self.map_img_can = self.can.create_image(0, 0, anchor=tk.NW, image=self.map_img_tk)

  

            

    def load_image(self):
        # 파일 선택 다이얼로그 열기
        file_path = filedialog.askopenfilename(initialdir =self.map_dir_path)
        if file_path:
            # 이미지를 열고 크기를 조정
            self.map_img_org = Image.open(file_path).convert("RGB")
            self.model_map_img_org = self.map_img_org
            self.gen_obs(init=True)

            self.map_scale = self.map_max/max(self.map_img_org.size)
            if np.argmax(self.map_img_org.size)==0:
                self.map_w,self.map_h = self.map_max, int(self.map_img_org.size[1]*self.map_scale)
            else:
                self.map_w,self.map_h = int(self.map_img_org.size[0]*self.map_scale), self.map_max
            self.map_img_resize = self.map_img_org.resize((self.map_w,self.map_h))
            self.can.config(width = self.map_w, height=self.map_h)

            if self.map_img_can is not None:
                self.can.delete(self.map_img_can)

            self.map_img_tk = ImageTk.PhotoImage(self.map_img_resize)
            self.map_img_can = self.can.create_image(0, 0, anchor=tk.NW, image=self.map_img_tk)
            self.road_point = np.argwhere(np.array(self.map_img_org)[...,0] >=250)
            # 라벨에 이미지 표시

    def drag(self, e):
        self.robot_pos_old = self.robot_pos
        self.robot_angle_old = self.robot_angle
        self.robot_pos = np.array([e.x,e.y])
        self.gen_obs(init=False)
        self.set_robot_pose(position=self.robot_pos)
        self.get_global_map_img()
        self.zoom_in_img_update()
        self.get_laser_data()
        self.predict()

    def get_global_map_img(self):
        if self.input_map_img !=None:
            if self.input_map_img_can !=None:
                self.can_input_map.delete(self.input_map_img_can)
            if self.model_type in ["seq","seq_c"]:
                self.model_input_map = self.input_map_img#.resize((self.crop_img_size,self.crop_img_size))
                w,h = self.model_input_map.size
                M = cv2.getRotationMatrix2D((w/2,h/2), self.robot_angle_old, 1)
                self.model_input_map = cv2.warpAffine(np.array(self.model_input_map), M, (w, h), borderMode = cv2.BORDER_REFLECT)
                self.model_input_map = Image.fromarray(self.model_input_map)

                self.input_map_img_tk = ImageTk.PhotoImage(self.model_input_map.resize((self.input_map_view_size,self.input_map_view_size)))
                self.input_map_img_can = self.can_input_map.create_image(0,0,anchor=tk.NW, image = self.input_map_img_tk)
            else:

                self.input_map_img_tk = ImageTk.PhotoImage(self.input_map_img.resize((self.input_map_view_size,self.input_map_view_size)))
                self.input_map_img_can = self.can_input_map.create_image(0,0,anchor=tk.NW, image = self.input_map_img_tk)
                self.model_input_map = self.input_map_img.resize((self.crop_img_size,self.crop_img_size))




    def zoom_in_img_update(self):
        if self.map_img_can is not None:
            x,y = self.robot_pos[0]/self.map_scale, self.robot_pos[1]/self.map_scale
            self.crop_img = self.map_img_org.crop((x-self.crop_img_size/2,
                                            y-self.crop_img_size/2, 
                                            x+self.crop_img_size/2,
                                            y+self.crop_img_size/2))#.resize((int(self.crop_img_size*self.crop_img_scale),
                                                                            #int(self.crop_img_size*self.crop_img_scale)))
            self.model_crop_img = self.model_map_img_org.crop((x-self.crop_img_size/2,
                                            y-self.crop_img_size/2, 
                                            x+self.crop_img_size/2,
                                            y+self.crop_img_size/2))
            self.input_map_img = self.model_crop_img

    def get_laser_data(self):
        if self.map_img_can is not None:
            img_arr = np.array(self.crop_img)
            if img_arr.ndim ==2:
                img_arr = np.tile(img_arr[...,None],(1,1,3))
 

            gray_img_arr = img_arr[...,0]

            wall_pos = np.argwhere(gray_img_arr<110).T
            crop_center = np.array([self.crop_img_size/2,
                           self.crop_img_size/2])

            laser_dist = np.sqrt(np.sum((wall_pos-crop_center[:,None])**2,axis=0))
            result_idx = np.where(laser_dist<(self.lidar_dist))[0]
            new_wall_pos = np.array([wall_pos[0][result_idx],
                                     wall_pos[1][result_idx]])
            laser_dist = np.sqrt(np.sum((new_wall_pos-crop_center[:,None])**2,axis=0))
            deg = np.arctan2(new_wall_pos[0]-crop_center[0],
                            new_wall_pos[1]-crop_center[1])

            result_pos_right = []
            tmp_deg = self.robot_angle-int(self.lidar_angle/2)
            tmp_deg_old = tmp_deg-1
            while tmp_deg < self.robot_angle + int(self.lidar_angle/2):
                i = tmp_deg
                if i<-180 : i+=360; tmp_deg_old+=360
                elif i>180 : i-=360; tmp_deg_old-=360
                same_deg_pos = np.argwhere((deg/np.pi*180<i)&(deg/np.pi*180>tmp_deg_old) )
                tmp_deg_old = tmp_deg
                if len(same_deg_pos)!=0:
                    min_dist = np.min(laser_dist[same_deg_pos])
                    min_dist_pos = np.argmin(laser_dist[same_deg_pos])
                    result_pos_right.append(same_deg_pos[min_dist_pos])
                    tmp_deg+=np.arctan2(1,min_dist)/np.pi*180
                else:
                    tmp_deg+=1
            if len(result_pos_right)==0:return
                
            result_pos_left=[]
            tmp_deg = self.robot_angle+int(self.lidar_angle/2)
            tmp_deg_old = tmp_deg+1
            while tmp_deg > self.robot_angle - int(self.lidar_angle/2):
                i = tmp_deg
                if i<-180 : i+=360; tmp_deg_old+=360
                elif i>180 : i-=360; tmp_deg_old-=360
                same_deg_pos = np.argwhere((deg/np.pi*180>i)&(deg/np.pi*180<tmp_deg_old) )
                tmp_deg_old = tmp_deg
                if len(same_deg_pos)!=0:
                    min_dist = np.min(laser_dist[same_deg_pos])
                    min_dist_pos = np.argmin(laser_dist[same_deg_pos])
                    result_pos_left.append(same_deg_pos[min_dist_pos])
                    tmp_deg-=np.arctan2(1,min_dist)/np.pi*180
                else:
                    tmp_deg-=1
                
            if len(result_pos_left)==0:return
            # import pdb;pdb.set_trace()
            result_idx = np.unique(np.intersect1d( np.array(result_pos_right).T[0], np.array(result_pos_left).T[0]   ))

            # rr = np.array(result_pos_right)[...,None]  ## i,2,1
            # rl = np.array(result_pos_left).T[None,...]   ## 1,2,k
            # confusion_idx = np.all(np.tile(rr,(1,1,rl.shape[-1])) == np.tile(rl,(rr.shape[-1],1,1)), axis=1)  ## i,k
            # rr_idx = np.any(confusion_idx,axis=1)
            # result_idx = rr[rr_idx].T

            # result_idx = np.array(result_pos).T[0]
            # result_idx = np.unique(result_idx)


            new_wall_pos = np.array([new_wall_pos[0][result_idx],
                                     new_wall_pos[1][result_idx]])

            if self.laser_boldering:
                new_wall_pos = np.concatenate([[new_wall_pos[0],new_wall_pos[1]],
                                         [new_wall_pos[0]+1,new_wall_pos[1]],
                                         [new_wall_pos[0],new_wall_pos[1]+1],
                                         [new_wall_pos[0]+1,new_wall_pos[1]+1],
                                         [new_wall_pos[0]-1,new_wall_pos[1]],
                                         [new_wall_pos[0],new_wall_pos[1]-1],
                                         [new_wall_pos[0]-1,new_wall_pos[1]-1],
                                         [new_wall_pos[0]-1,new_wall_pos[1]+1],
                                         [new_wall_pos[0]+1,new_wall_pos[1]-1]], axis=1)


            new_wall_pos= np.unique(new_wall_pos,axis=1)
            new_wall_pos = np.clip(new_wall_pos,0,int(self.crop_img_size))
            img_arr[new_wall_pos[0],new_wall_pos[1],0]=255



            

            self.crop_img_tk = ImageTk.PhotoImage(Image.fromarray(img_arr).resize((int(self.crop_img_size*self.crop_img_scale),
                                                                                   int(self.crop_img_size*self.crop_img_scale))))
            if self.crop_img_can is not None:
                self.can_crop.delete(self.crop_img_can)
            self.crop_img_can = self.can_crop.create_image(0,0,anchor=tk.NW, image = self.crop_img_tk)
            self.can_crop.tag_raise(self.robot_body_can_crop)
            self.can_crop.tag_raise(self.robot_head_can_crop)

            self.can_crop.tag_raise(self.robot_lidar_range)

            self.sensor_img = np.zeros(gray_img_arr.shape)
            rotated_point =self.trans_point(new_wall_pos.T,center_pos=np.array(self.sensor_img.shape)/2, rot=self.robot_angle,pos=np.array(self.sensor_img.shape)/2).T.astype(np.int32)
            self.sensor_img[rotated_point[0],rotated_point[1]] = 255
            if self.sensor_img_can is not None:
                self.can_sensor.delete(self.sensor_img_can)
            self.sensor_img_tk = ImageTk.PhotoImage(Image.fromarray(self.sensor_img).resize((self.sensor_view_size,self.sensor_view_size)))
            self.sensor_img_can = self.can_sensor.create_image(0,0,anchor=tk.NW, image = self.sensor_img_tk)

            self.model_input_sensor = Image.fromarray(self.sensor_img.round().astype(np.uint8)).resize((self.crop_img_size,self.crop_img_size))
            
                
            
 
    

    def key_press(self,key):
        if self.Loop_flag:
            if key.keysym == "Left":self.turn_left()
            if key.keysym == "Right":self.turn_right()
            if key.keysym == "Up":self.forward()
            if key.keysym == "Down":self.backward()

        else:
            self.gen_obs(init=False)
            if key.keysym == "Left":self.turn_left()
            if key.keysym == "Right":self.turn_right()
            if key.keysym == "Up":self.forward()
            if key.keysym == "Down":self.backward()
            self.get_global_map_img()
            self.zoom_in_img_update()
            self.get_laser_data()
            self.predict()

        

    def turn_left(self):
        self.robot_pos_old = self.robot_pos
        self.robot_angle_old = self.robot_angle
        self.robot_angle -=self.robot_turn_scale
        if self.robot_angle<=-180: self.robot_angle +=360 
        self.set_robot_pose(rotation=-self.robot_turn_scale)

        
    def turn_right(self):
        self.robot_pos_old = self.robot_pos
        self.robot_angle_old = self.robot_angle
        self.robot_angle +=self.robot_turn_scale
        if self.robot_angle>=180: self.robot_angle -=360 
        self.set_robot_pose(rotation= self.robot_turn_scale)

        
    def forward(self):
        self.robot_pos_old = self.robot_pos
        self.robot_angle_old = self.robot_angle
        next_pos = self.robot_pos+np.array([self.robot_forward_back_scale,0])
        new_head_point=self.trans_point(next_pos,self.robot_pos,self.robot_angle,self.robot_pos)
        self.set_robot_pose(position=new_head_point[0])

        
    def backward(self):
        self.robot_pos_old = self.robot_pos
        self.robot_angle_old = self.robot_angle
        next_pos = self.robot_pos-np.array([self.robot_forward_back_scale,0])
        new_head_point=self.trans_point(next_pos,self.robot_pos,self.robot_angle,self.robot_pos)
        self.set_robot_pose(position=new_head_point[0])

    def Loop_button(self):
        self.Loop_flag = not self.Loop_flag
        self.Loop()
    def Loop(self):
        if self.Loop_flag:
            self.gen_obs(init=False)
            self.auto_pilot()
            self.get_global_map_img()
            self.zoom_in_img_update()
            self.get_laser_data()
            self.predict()
            self.win.after(30, self.Loop)

    def auto_pilot(self):
        self.robot_pos_old = self.robot_pos
        self.robot_angle_old = self.robot_angle

        if self.task_num==1:
            random_idx = np.random.choice(len(self.road_point),1)
            self.d_xy = self.road_point[random_idx][0][::-1]
            # self.d_xy = self.task_buf[self.task_buf_num]
            # self.task_buf_num+=1
            tmp_err = self.d_xy-self.robot_pos/self.map_scale

            self.d_deg = np.arctan2(tmp_err[1],tmp_err[0])/np.pi*180
            self.task_num+=1


        if self.task_num==2:
            e_deg = self.d_deg - self.robot_angle

            ed_deg = (e_deg - self.eo_deg)*self.dt
            pd_deg_out = 0.13*e_deg + 0.05*(ed_deg)
            self.eo_deg = e_deg

            self.robot_angle += pd_deg_out
            if self.robot_angle>=180: self.robot_angle -=360 
            self.set_robot_pose(rotation= pd_deg_out)
            
            if np.abs(e_deg)<=2:
                print("deg cor")
                self.task_num+=1

        elif self.task_num==3:
            e_xy = self.d_xy-self.robot_pos /self.map_scale
            
            ed_xy = (e_xy - self.eo_xy)*self.dt
            pd_xy_out = 0.15*e_xy + 0.1*(ed_xy)
            pd_xy_out = np.clip(pd_xy_out,-8,8)
            self.eo_xy = e_xy

            next_pos = self.robot_pos + pd_xy_out*self.map_scale
            self.set_robot_pose(position=next_pos)
            
            if np.sqrt(np.sum((e_xy)**2))<=5:
                print("pos cor")
                self.task_num=1



def on_closing():
    print("Performing cleanup tasks...")
    # 여기서 정리 작업을 수행합니다.
    print("Exiting the program.")
    sys.exit()


# 메인 윈도우 생성
window = tk.Tk()
window.protocol("WM_DELETE_WINDOW", on_closing)
window.geometry('1920x1080')
window.title("Image Loader")

display = Display(win= window)

window.mainloop()