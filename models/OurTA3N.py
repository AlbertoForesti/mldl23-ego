import models
import torch.nn as nn
from models.I3D import I3D
from models.I3D import InceptionI3d
from torch.nn.init import normal_, constant_
from models import TRNmodule

class BaselineTA3N(nn.Module):

    VALID_ENDPOINTS = (
        'Backbone',
        'Spatial module',
        'Temporal module',
        'Gy',
        'Logits',
        'Predictions',
    )

    def __init__(self, in_features_dim, num_classes, model_config):
        
        self.end_points = {}
        self.train_segments = model_config.train_segments
        self.val_segments = model_config.val_segments
        self.frame_aggregation=model_config.frame_aggregation
        self._final_endpoint = model_config.final_endpoint
        super(BaselineTA3N, self).__init__()
        
        """
        this is a way to get the number of features at input
        it is the number of features in input before the logits endpoint in I3D
        """
       
        end_point = 'Spatial module' # just a fully connected layer
        fc_spatial_module = self.FullyConnectedLayer(in_features_dim=in_features_dim, out_features_dim=in_features_dim)
        std = 0.001
        constant_(fc_spatial_module.bias, 0)
        normal_(fc_spatial_module.weight, 0, std)
		
        self.end_points[end_point] = fc_spatial_module # spatial module is just a fully connected layer
        
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Temporal module'
        self.end_points[end_point] = self.TemporalModule(in_features_dim, self.train_segments)
        in_features_dim = self.end_points[end_point].out_features_dim
        if self._final_endpoint == end_point:
            return

        end_point = 'Gy'
        fc_gy = self.FullyConnectedLayer(in_features_dim=in_features_dim, out_features_dim=in_features_dim)
        constant_(fc_gy.bias, 0)
        normal_(fc_gy.weight, 0, std)

        self.end_points[end_point] = fc_gy
        if not self._final_endpoint == end_point:
            
            self.fc_classifier_video = nn.Linear(in_features_dim, num_classes)
            std = 0.001
            normal_(self.fc_classifier_video.weight, 0, std)
            constant_(self.fc_classifier_video.bias, 0)
            
        self.build()
        
        

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])


    def forward(self, x, is_train=True):
        num_segments = self.train_segments if is_train else self.val_segments
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                if end_point == 'Temporal module':
                    x = self._modules[end_point](x, num_segments)  # use _modules to work with dataparallel    
                else:
                    x = self._modules[end_point](x)  # use _modules to work with dataparallel
        pass

    class FullyConnectedLayer(nn.Module):
        def __init__(self, in_features_dim, out_features_dim, dropout=0.8):
            super(BaselineTA3N.FullyConnectedLayer, self).__init__()
            self.in_features_dim = in_features_dim
            self.out_features_dim = out_features_dim
            
            """Here I am doing what is done in the official code, 
            in the first fc layer the output dimension is the minimum between the input feature dimension and 1024"""
            self.relu = nn.ReLU(inplace=True) # Again using the architecture of the official code
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(self.in_features_dim, self.out_features_dim)
            self.bias = self.fc.bias
            self.weight = self.fc.weight
        
        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x


    class TemporalModule(nn.Module):
        def __init__(self, in_features_dim, train_segments, temporal_pooling = 'TemPooling') -> None:
            super(BaselineTA3N.TemporalModule, self).__init__()
            self.pooling_type = temporal_pooling
            self.in_features_dim = in_features_dim
            self.train_segments = train_segments
            if temporal_pooling == 'TemPooling':
                self.out_features_dim = self.in_features_dim
                pass
            elif temporal_pooling == 'TemRelation':
                self.num_bottleneck = 512
                self.trn = TRNmodule.RelationModule(in_features_dim, self.num_bottleneck, self.train_segments)
                self.out_features_dim = self.num_bottleneck
                pass
            else:
                raise NotImplementedError
        
        def forward(self, x, num_segments):
            if self.pooling_type == 'TemRelation':
                x = x.view((-1, num_segments) + x.size()[-1:])
                return self.trn(x)
            elif self.pooling_type =="TempPooling":
                x = x.view((-1, 1, num_segments) + x.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
                

                x = nn.AvgPool2d([num_segments, 1])(x)  # e.g. 16 x 1 x 1 x 512
                x= x.squeeze(1).squeeze(1)  # e.g. 16 x 512
                return x
               
    class FeatureExtractorModule(nn.Module):

        VALID_BACKBONES = {
            'i3d': I3D
        }

        def __init__(self, num_class, modality, model_config, **kwargs):
            super(BaselineTA3N.FeatureExtractorModule, self).__init__()
            self.backbone = I3D(num_class, modality, model_config, **kwargs)
            self.feat_dim = self.backbone.feat_dim
        
        def forward(self, x):
            logits, features = self.backbone(x)
            features = features['feat']
            return features.view(-1, features.size()[-1]) 

  
