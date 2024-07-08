import torch
import utils
from copy import deepcopy
from typing import List, Tuple
from einops import rearrange

class Feature_Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.map_seq = torch.nn.Sequential(
            torch.nn.Conv2d(1,8,3,1,1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(True),
        
            torch.nn.Conv2d(8,16,3,1,1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(True),
            torch.nn.MaxPool2d(4,4),
        
            torch.nn.Conv2d(16,16,3,1,1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(True),
            
            torch.nn.Conv2d(16,32,3,1,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(True),
            torch.nn.MaxPool2d(4,4),

            torch.nn.Conv2d(32,64,3,1,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(True),

            torch.nn.Conv2d(64,128,3,1,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(True),

            
            )
        
        self.sensor_seq = deepcopy(self.map_seq)

        

        self.map_seq.apply(utils.weight_init_xavier_uniform)
        self.sensor_seq.apply(utils.weight_init_xavier_uniform)


    def forward(self,map_img, sensor_img):
        map_feature = self.map_seq(map_img)
        sensor_feature = self.sensor_seq(sensor_img)

        return map_feature, sensor_feature


class AttentionalGNN(torch.nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1
    
class AttentionalPropagation(torch.nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        torch.nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))
    
class MultiHeadedAttention(torch.nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = torch.nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))
    
def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    #### query (1,64,4,384)
    #### key   (1,64,4,384)
    #### value (1,64,4,384)
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    #### scores (1,4,384,384)
    prob = torch.nn.functional.softmax(scores, dim=-1)
    #### prob (1,4,384,384)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob  ## einsum (1,64,4,384)

class KeypointEncoder(torch.nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])
        torch.nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        return self.encoder(kpts)
    

def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]

def MLP(channels: List[int], do_bn: bool = True) -> torch.nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            torch.nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(torch.nn.BatchNorm1d(channels[i]))
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)



class NN_Localization(torch.nn.Module):
    default_config = {
        'descriptor_dim': 128,
        'keypoint_encoder': [32, 64, 128,128],
        'GNN_layers': ['self', 'cross'] * 9,
        'feature_out_size':[20,20]
    }
    def __init__(self, rot_out_num):
        super().__init__()
        self.config = self.default_config
        self.cnn = Feature_Extractor()
        self.kenc = KeypointEncoder(self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = AttentionalGNN(feature_dim=self.config['descriptor_dim'],layer_names=self.config['GNN_layers'])
        self.final_proj = torch.nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        self.pos_y, self.pos_x = torch.meshgrid(torch.arange(self.config['feature_out_size'][0]),
                                                torch.arange(self.config['feature_out_size'][1]))
        self.pos = torch.cat([self.pos_y.flatten()[None,...], self.pos_x.flatten()[None,...]])[None,...].float().cuda()
        
        self.reduction_cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(128,128,3,1,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(True),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(128,128,3,1,1)
        )
        self.reduction_cnn2 = deepcopy(self.reduction_cnn1)

        self.fc_layer = torch.nn.Linear(100*100, 512)
        self.fc_bn = torch.nn.BatchNorm1d(512)

        self.loc_fc1 = torch.nn.Linear(512,256)
        self.loc_fc1_bn = torch.nn.BatchNorm1d(256)
        self.loc_fc2 = torch.nn.Linear(256,2)

        self.rot_fc1 = torch.nn.Linear(512,256)
        self.rot_fc1_bn = torch.nn.BatchNorm1d(256)
        self.rot_fc2 = torch.nn.Linear(256,rot_out_num)

        self.reduction_cnn1.apply(utils.weight_init_xavier_uniform)
        self.reduction_cnn2.apply(utils.weight_init_xavier_uniform)
        torch.nn.init.xavier_uniform_(self.fc_layer.weight)
        torch.nn.init.xavier_uniform_(self.loc_fc1.weight)
        torch.nn.init.xavier_uniform_(self.loc_fc2.weight)
        torch.nn.init.xavier_uniform_(self.rot_fc1.weight)
        torch.nn.init.xavier_uniform_(self.rot_fc2.weight)

        torch.nn.init.ones_(self.fc_bn.weight)
        torch.nn.init.zeros_(self.fc_bn.bias)

        torch.nn.init.ones_(self.loc_fc1_bn.weight)
        torch.nn.init.zeros_(self.loc_fc1_bn.bias)

        torch.nn.init.ones_(self.rot_fc1_bn.weight)
        torch.nn.init.zeros_(self.rot_fc1_bn.bias)


    def forward(self,map_img, sensor_img):
        map_f, sensor_f = self.cnn(map_img,sensor_img) # b, ch(64), 20,20 
        map_f       = rearrange(map_f,      'b c w h -> b c (w h)')
        sensor_f    = rearrange(sensor_f,   'b c w h -> b c (w h)')
        # import pdb;pdb.set_trace()
        map_desc    = map_f + self.kenc(self.pos)
        sensor_desc = sensor_f + self.kenc(self.pos)

        map_desc, sensor_desc = self.gnn(map_desc,sensor_desc)
        #map_mdesc, sensor_mdesc = self.final_proj(map_desc), self.final_proj(sensor_desc)
        map_desc        = rearrange(map_desc,      'b c (w h) -> b c w h', w=self.config['feature_out_size'][0],h=self.config['feature_out_size'][1])
        sensor_desc     = rearrange(sensor_desc,      'b c (w h) -> b c w h', w=self.config['feature_out_size'][0],h=self.config['feature_out_size'][1])
        map_mdesc, sensor_mdesc = self.reduction_cnn1(map_desc), self.reduction_cnn2(sensor_desc)
        map_mdesc       = rearrange(map_mdesc,'b c w h-> b c (w h)')
        sensor_mdesc    = rearrange(sensor_mdesc,'b c w h-> b c (w h)')
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', map_mdesc, sensor_mdesc)
        scores_flat = rearrange(scores,'b w h -> b (w h)')

        fc_out = torch.nn.functional.relu(self.fc_bn(self.fc_layer(scores_flat)))
        loc_fc = torch.nn.functional.relu(self.loc_fc1_bn(self.loc_fc1(fc_out)))
        loc_fc = self.loc_fc2(loc_fc)
        rot_fc = torch.nn.functional.relu(self.rot_fc1_bn(self.rot_fc1(fc_out)))
        rot_fc = self.rot_fc2(rot_fc)

        return loc_fc,rot_fc
    

class NN_Localization_Conf(torch.nn.Module):
    default_config = {
        'descriptor_dim': 128,
        'keypoint_encoder': [32, 64, 128,128],
        'GNN_layers': ['self', 'cross'] * 9,
        'feature_out_size':[20,20]
    }
    def __init__(self, rot_out_num):
        super().__init__()
        self.config = self.default_config
        self.cnn = Feature_Extractor()
        self.kenc = KeypointEncoder(self.config['descriptor_dim'], self.config['keypoint_encoder'])
        self.gnn = AttentionalGNN(feature_dim=self.config['descriptor_dim'],layer_names=self.config['GNN_layers'])
        self.final_proj = torch.nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        self.pos_y, self.pos_x = torch.meshgrid(torch.arange(self.config['feature_out_size'][0]),
                                                torch.arange(self.config['feature_out_size'][1]))
        self.pos = torch.cat([self.pos_y.flatten()[None,...], self.pos_x.flatten()[None,...]])[None,...].float().cuda()
        
        self.reduction_cnn1 = torch.nn.Sequential(
            torch.nn.Conv2d(128,128,3,1,1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(True),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(128,128,3,1,1)
        )
        self.reduction_cnn2 = deepcopy(self.reduction_cnn1)

        self.fc_layer = torch.nn.Linear(100*100, 512)
        self.fc_bn = torch.nn.BatchNorm1d(512)

        self.loc_fc1 = torch.nn.Linear(512,256)
        self.loc_fc1_bn = torch.nn.BatchNorm1d(256)
        self.loc_fc2 = torch.nn.Linear(256,2)

        self.rot_fc1 = torch.nn.Linear(512,256)
        self.rot_fc1_bn = torch.nn.BatchNorm1d(256)
        self.rot_fc2 = torch.nn.Linear(256,rot_out_num)

        self.loc_conf_fc1 = torch.nn.Linear(512,128)
        self.loc_conf_fc1_bn = torch.nn.BatchNorm1d(128)
        self.loc_conf_fc2 = torch.nn.Linear(128,1)

        self.rot_conf_fc1 = torch.nn.Linear(512,128)
        self.rot_conf_fc1_bn = torch.nn.BatchNorm1d(128)
        self.rot_conf_fc2 = torch.nn.Linear(128,1)



        self.reduction_cnn1.apply(utils.weight_init_xavier_uniform)
        self.reduction_cnn2.apply(utils.weight_init_xavier_uniform)
        torch.nn.init.xavier_uniform_(self.fc_layer.weight)
        torch.nn.init.xavier_uniform_(self.loc_fc1.weight)
        torch.nn.init.xavier_uniform_(self.loc_fc2.weight)
        torch.nn.init.xavier_uniform_(self.rot_fc1.weight)
        torch.nn.init.xavier_uniform_(self.rot_fc2.weight)
        torch.nn.init.xavier_uniform_(self.loc_conf_fc1.weight)
        torch.nn.init.xavier_uniform_(self.loc_conf_fc2.weight)
        torch.nn.init.xavier_uniform_(self.rot_conf_fc1.weight)
        torch.nn.init.xavier_uniform_(self.rot_conf_fc2.weight)

        torch.nn.init.ones_(self.fc_bn.weight)
        torch.nn.init.zeros_(self.fc_bn.bias)

        torch.nn.init.ones_(self.loc_fc1_bn.weight)
        torch.nn.init.zeros_(self.loc_fc1_bn.bias)

        torch.nn.init.ones_(self.rot_fc1_bn.weight)
        torch.nn.init.zeros_(self.rot_fc1_bn.bias)

        torch.nn.init.ones_(self.loc_conf_fc1_bn.weight)
        torch.nn.init.zeros_(self.loc_conf_fc1_bn.bias)

        torch.nn.init.ones_(self.rot_conf_fc1_bn.weight)
        torch.nn.init.zeros_(self.rot_conf_fc1_bn.bias)


    def forward(self,map_img, sensor_img):
        map_f, sensor_f = self.cnn(map_img,sensor_img) # b, ch(64), 20,20 
        map_f       = rearrange(map_f,      'b c w h -> b c (w h)')
        sensor_f    = rearrange(sensor_f,   'b c w h -> b c (w h)')
        # import pdb;pdb.set_trace()
        map_desc    = map_f + self.kenc(self.pos)
        sensor_desc = sensor_f + self.kenc(self.pos)

        map_desc, sensor_desc = self.gnn(map_desc,sensor_desc)
        #map_mdesc, sensor_mdesc = self.final_proj(map_desc), self.final_proj(sensor_desc)
        map_desc        = rearrange(map_desc,      'b c (w h) -> b c w h', w=self.config['feature_out_size'][0],h=self.config['feature_out_size'][1])
        sensor_desc     = rearrange(sensor_desc,      'b c (w h) -> b c w h', w=self.config['feature_out_size'][0],h=self.config['feature_out_size'][1])
        map_mdesc, sensor_mdesc = self.reduction_cnn1(map_desc), self.reduction_cnn2(sensor_desc)
        map_mdesc       = rearrange(map_mdesc,'b c w h-> b c (w h)')
        sensor_mdesc    = rearrange(sensor_mdesc,'b c w h-> b c (w h)')
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', map_mdesc, sensor_mdesc)
        scores_flat = rearrange(scores,'b w h -> b (w h)')

        fc_out = torch.nn.functional.relu(self.fc_bn(self.fc_layer(scores_flat)))
        loc_fc = torch.nn.functional.relu(self.loc_fc1_bn(self.loc_fc1(fc_out)))
        loc_fc = self.loc_fc2(loc_fc)
        rot_fc = torch.nn.functional.relu(self.rot_fc1_bn(self.rot_fc1(fc_out)))
        rot_fc = self.rot_fc2(rot_fc)
        loc_conf_fc = torch.nn.functional.relu(self.loc_conf_fc1_bn(self.loc_conf_fc1(fc_out)))
        loc_conf_fc = torch.nn.functional.sigmoid(self.loc_conf_fc2(loc_conf_fc))
        rot_conf_fc = torch.nn.functional.relu(self.rot_conf_fc1_bn(self.rot_conf_fc1(fc_out)))
        rot_conf_fc = torch.nn.functional.sigmoid(self.rot_conf_fc2(rot_conf_fc))

        return loc_fc, rot_fc, loc_conf_fc,rot_conf_fc