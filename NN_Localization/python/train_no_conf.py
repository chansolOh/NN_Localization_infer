import models
import utils
import dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import wandb
import torchvision.transforms as transforms
from linear_warmup_cosine_annealing_warm_restarts_weight_decay import ChainedScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
os.system("ulimit -n 10000")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices star
os.environ["CUDA_VISIBLE_DEVICES"]= "0" 



sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "rot1_loc1_acc"},
    "parameters": {
        "batch_size": {"value":32},
        "epochs": {"value": 140},
        "optimizer": {"value": "sgd"},
        "scheduler": {"value":"chained1"},
        "dataset":{"value":"obs_loc10_rot90"},
        "augmentation":{"value":"invert"},
        
    },
}
sweep_id = wandb.sweep(sweep = sweep_config, project = "NN_localization_sweep3")

def visualizer(dataset,model, train_data):
    for data in dataset:
        map_img, crop_img, loc,rot = data["map_img"], data["crop_img"], data["location"], data["rotation"]
        out_loc, out_rot = model(map_img.cuda(), crop_img.cuda())
        out_loc,out_rot = out_loc.detach().cpu().numpy(), out_rot.detach().cpu().numpy()
        fig, ax = plt.subplots(2,len(out_loc),figsize=(25,5))
        rot_deg=np.argmax(out_rot, axis=1)*train_data.angle_interval
        gt_rot_deg = np.argmax(rot,axis=1)*train_data.angle_interval
        for num in range(len(out_loc)):
            pred_point = utils.restore(crop_img[num][0],out_loc[num]+train_data.rotated_pos_min, (rot_deg[num]+train_data.angle_min+(train_data.angle_interval/2))/180*np.pi) 
            ax[0][num].imshow(map_img[num][0])
            ax[0][num].scatter(pred_point[0],pred_point[1],s=1,c='r')

            gt_point = utils.restore(crop_img[num][0],loc[num]+train_data.rotated_pos_min, (gt_rot_deg[num]+train_data.angle_min+(train_data.angle_interval/2))/180*np.pi)
            ax[1][num].imshow(map_img[num][0])
            ax[1][num].scatter(gt_point[0],gt_point[1],s=1,c='r')

        break
    return fig

def compute_loss(data,model,loc_crit,rot_crit):
    map_img, crop_img, loc,rot = data["map_img"], data["crop_img"], data["location"], data["rotation"]
    loc, rot = loc.float().cuda(), rot.float().cuda()
    out_loc, out_rot = model(map_img.cuda(), crop_img.cuda())

    loc_loss = loc_crit(out_loc,loc) 
    rot_loss = rot_crit(out_rot,rot)
    loss =  loc_loss + rot_loss
    return loss, loc_loss,rot_loss

def main():

    wandb.init()

    batch_size = wandb.config.batch_size
    epoch=wandb.config.epochs

    it = 0
    log_per_it = int(10000/batch_size)

    dataset_dict={
        "loc20":{
            "data_dir_path":"../data_10000_loc20",
            "augmentation_range":360,
            "class_num":72,
            
        },
        "loc20_rot90":{
            "data_dir_path":"../data_10000_loc20_rot90",
            "augmentation_range":180,
            "class_num":36,
        },
        "newmap_loc20_rot90":{
            "data_dir_path":"../data_10000_newmap_loc20_rot90",
            "augmentation_range":180,
            "class_num":36,
        },
        "newmap_loc20":{
            "data_dir_path":"../data_10000_newmap_loc20",
            "augmentation_range":360,
            "class_num":72,
        },
         "obs_loc10_rot90":{
            "data_dir_path":"../data_10000_obs_loc10_rot90",
            "augmentation_range":180,
            "class_num":36,
        }
    }

 

    model = models.NN_Localization(rot_out_num=dataset_dict[wandb.config.dataset]["class_num"])
    #model.load_state_dict(torch.load("../weights/GAT_seg_cnn_sgd_chained1_loc20_rot90_65ep.pt")["model_state_dict"])
    model = model.cuda()

    optimizer_dict = {"adam":torch.optim.Adam(model.parameters(), lr=1e-3),
                      "sgd":torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)}
    optimizer = optimizer_dict[wandb.config.optimizer]


    scheduler_dict = {"step_lr":torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9),
                      "chained1":ChainedScheduler(optimizer, T_0 = 20, T_mul = 2,eta_min = 0.0,gamma = 0.7,max_lr = 0.001,warmup_steps= 3 ,),
                      "chained2":ChainedScheduler(optimizer, T_0 = 10, T_mul = 2,eta_min = 0.0,gamma = 0.4,max_lr = 0.001,warmup_steps= 3 ,),
                      }

    scheduler = scheduler_dict[wandb.config.scheduler]


    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2)


    # hook = utils.Hook()
    # model.merge_seq.register_forward_hook(hook.forward_hook)

    loc_criterion = torch.nn.L1Loss()
    rot_criterion = torch.nn.CrossEntropyLoss()

    transform_dict = {
        "invert":transforms.Compose([
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=5, sigma= (0.1,2)),
                transforms.RandomInvert(),
                transforms.Normalize((0.5), (0.5))
                ]),
        "no_invert":transforms.Compose([
                transforms.ToTensor(),
                transforms.GaussianBlur(kernel_size=5, sigma= (0.1,2)),
                transforms.Normalize((0.5), (0.5))
                ])
    }
    transform_train = transform_dict[wandb.config.augmentation]
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])
        
        
    train_data = dataset.Seg_Cnn_Dataset(dataset_dict[wandb.config.dataset],dir_range=[0,9],transform = transform_train, augmentation=utils.rotate_only_map)
    test_data  = dataset.Seg_Cnn_Dataset(dataset_dict[wandb.config.dataset],dir_range=[9,11],transform = transform1)
    valid_data = dataset.Seg_Cnn_Dataset(dataset_dict[wandb.config.dataset],dir_range=[11,13],transform = transform1)


    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size=batch_size, num_workers=10,shuffle = True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(dataset = test_data,  batch_size=10, num_workers=10,shuffle = False, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset = valid_data, batch_size=10, num_workers=10,shuffle = False, drop_last=True)

    loss_batch_size = len(train_loader)
    

    model_name = f"GSC_{wandb.config.augmentation}_{wandb.config.dataset}"
    wandb.run.name = model_name
    wandb.run.save()
    # torch.save(model,f"../model_pth/sweep1_{model_name}.pth")
    wandb.watch(
        model,
        log = "all",
        log_freq=loss_batch_size,
        idx= None,
        log_graph= (False)
    )

    valid_loss_buffer = []
    valid_loss_min = 100
    for ep in range(epoch):
        train_loss = 0.0
        loc_loss_total = 0.0
        rot_loss_total = 0.0
        loss_it = 0
        model.train()
        train_tqdm = tqdm(train_loader)
        for i, data in enumerate(train_tqdm):
            it +=1
            loss_it +=1
            optimizer.zero_grad()
            loss, loc_loss, rot_loss = compute_loss(data,model,loc_criterion,rot_criterion)
            loss.backward() 
            optimizer.step()

            train_loss += loss.item()
            loc_loss_total += loc_loss.item()
            rot_loss_total += rot_loss.item()
            train_tqdm.set_description('loss: {:09f}'.format(loss.item()))
            if it%log_per_it==0:
                wandb.log({
                    "train_loss":train_loss/loss_it,
                    "loc_loss":  loc_loss_total/loss_it,
                    "rot_loss":  rot_loss_total/loss_it,
                    # "merge_seq1_input" :wandb.Image(hook.Input[0][0][0],caption=f'it:{it}' )    
                    })
                
        model.eval()
        fig = visualizer(test_loader, model, train_data)

        wandb.log({"infer_test_seg_cnn":fig})
        plt.close()

        scheduler.step()

            
        valid_loss = 0
        model.eval()
        valid_tqdm = enumerate(tqdm(valid_loader))    
        for i ,data in valid_tqdm:
            loss,_,_ = compute_loss(data,model,loc_criterion,rot_criterion)
            valid_loss += loss.item()
        wandb.log({
                "lr":scheduler.get_lr()[0],
                "val_loss":valid_loss/ loss_batch_size })
        
        valid_loss_buffer.append(valid_loss / loss_batch_size)
        print('ep:{} ### train_loss:{:08f}  ###  val_loss:{:08f}'.format(ep + 1, train_loss / loss_batch_size, valid_loss / loss_batch_size))


        ### acc test
        rot_err_th1=[]
        rot_err_th2=[]
        loc_err_th1=[]
        loc_err_th2=[]

        rot_err_threshold1 = 1 ## +-1(-10 ~ 10)
        rot_err_threshold2 = 2
        loc_err_threshold1 = 2 ## +-1, 1pix = 5cm
        loc_err_threshold2 = 4
        for i ,data in enumerate(tqdm(test_loader)):
            map_img, crop_img, loc,rot = data["map_img"], data["crop_img"], data["location"], data["rotation"]
            out_loc,out_rot = model(map_img.cuda(), crop_img.cuda())
            out_loc,out_rot = out_loc.detach().cpu().numpy(), out_rot.detach().cpu().numpy()
            loc_err_th1 += [np.sqrt(np.sum((out_loc-loc.numpy())**2, axis=1))<loc_err_threshold1]
            loc_err_th2 += [np.sqrt(np.sum((out_loc-loc.numpy())**2, axis=1))<loc_err_threshold2]
            rot_err_th1 += [np.abs(np.argmax(out_rot,axis=1) - np.argmax(rot.numpy(),axis=1))<rot_err_threshold1]
            rot_err_th2 += [np.abs(np.argmax(out_rot,axis=1) - np.argmax(rot.numpy(),axis=1))<rot_err_threshold2]

        wandb.log({
                f"rot_acc +-{rot_err_threshold1*train_data.angle_interval}": np.mean(rot_err_th1),
                f"rot_acc +-{rot_err_threshold2*train_data.angle_interval}": np.mean(rot_err_th2),
                f"loc_acc +-{loc_err_threshold1}": np.mean(loc_err_th1),
                f"loc_acc +-{loc_err_threshold2}": np.mean(loc_err_th2),
                f"rot1_loc1_acc" : np.logical_and(loc_err_th1,rot_err_th1).mean(),
                f"rot2_loc2_acc" : np.logical_and(loc_err_th2,rot_err_th2).mean(),
                })
        test_loc=[]
        test_rot=[]
        for data in (test_loader):
            loc,rot = data["location"], data["rotation"]
            test_loc+=[loc]
            test_rot+=[rot]
        test_loc = np.concatenate(test_loc)
        test_rot = np.concatenate(test_rot)
        loc_cor2_idx = np.concatenate(loc_err_th2)
        rot_cor2_idx = np.concatenate(rot_err_th2)
        cor_locs = (test_loc[loc_cor2_idx]+train_data.rotated_pos_min).astype(np.int32)
        cor_rots = np.argmax(test_rot[rot_cor2_idx],axis=1)*5-180
        loc_hist = np.histogram(cor_locs,bins=range(int(train_data.pos_min-train_data.rotated_pos_min),int(train_data.pos_max+train_data.rotated_pos_min)+1,1))
        rot_hist = np.histogram(cor_rots,bins=range(-180,185,5))
        wandb.log({
            "loc_correct_hist":wandb.Histogram(np_histogram=loc_hist),
            "rot_correct_hist":wandb.Histogram(np_histogram=rot_hist),
        })
        
        
        if valid_loss_min > valid_loss_buffer[-1]:
            valid_loss_min = valid_loss_buffer[-1]
            torch.save({"model_state_dict":model.state_dict(),
                        # "description":model_description,
                        },
                    f"../weights/{model_name}_{ep}ep.pt")

wandb.agent(sweep_id, function = main)
