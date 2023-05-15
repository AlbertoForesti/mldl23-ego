import models
import torch
import torch.nn as nn
from torch.autograd import Function

from torch.nn.init import normal_, constant_
from models import TRNmodule
from collections import OrderedDict

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
        self.model_config = model_config
        super(BaselineTA3N, self).__init__()        
        """
        this is a way to get the number of features at input
        it is the number of features in input before the logits endpoint in I3D
        """
       
        end_point = 'Spatial module' # just a fully connected layer
        fc_spatial_module = self.FullyConnectedLayer(in_features_dim=in_features_dim, out_features_dim=in_features_dim, dropout=model_config.dropout)
        std = 0.001
        constant_(fc_spatial_module.bias, 0)
        normal_(fc_spatial_module.weight, 0, std)
		
        self.end_points[end_point] = fc_spatial_module # spatial module is just a fully connected layer
        
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Gsd'
        if end_point in self.model_config.blocks:
            self.end_points[end_point] = self.DomainClassifier(in_features_dim, model_config.beta[0])
            if self._final_endpoint == end_point:
                return
        
        end_point = 'Temporal module'
        self.end_points[end_point] = self.TemporalModule(in_features_dim, self.train_segments, temporal_pooling=model_config.frame_aggregation)
        in_features_dim = self.end_points[end_point].out_features_dim
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Gtd'
        if end_point in self.model_config.blocks:
            self.end_points[end_point] = self.DomainClassifier(in_features_dim, model_config.beta[1])
            if self._final_endpoint == end_point:
                return
        
        end_point = 'Gy'
        fc_gy = self.FullyConnectedLayer(in_features_dim=in_features_dim, out_features_dim=in_features_dim, dropout=model_config.dropout)
        constant_(fc_gy.bias, 0)
        normal_(fc_gy.weight, 0, std)

        self.end_points[end_point] = fc_gy

        end_point='Logits'

        self.fc_classifier_video = nn.Linear(in_features_dim, num_classes)
        std = 0.001
        normal_(self.fc_classifier_video.weight, 0, std)
        constant_(self.fc_classifier_video.bias, 0)

        self.build()
        
        

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])


    def forward(self, source, target=None, is_train=True):
        num_segments = self.train_segments if is_train else self.val_segments
        
        source = self._modules['Spatial module'](source)

        if is_train:
            target = self._modules['Spatial module'](target)
        
        raise UserWarning(f'shape of source is {source.shape}')
        
        if 'Gsd' in self.end_points and is_train:
            predictions_gsd_source = self._modules['Gsd'](source) # to concat
            predictions_gsd_target = self._modules['Gsd'](target)
        else:
            predictions_gsd_source = None
            predictions_gsd_target = None

        source = self._modules['Temporal module'](source, num_segments)

        if is_train:
            target = self._modules['Temporal module'](target, num_segments)
        
        if 'Gtd' in self.end_points and is_train:
            predictions_gtd_source = self._modules['Gtd'](source) # to concat
            predictions_gtd_target = self._modules['Gtd'](target)
        else:
            predictions_gtd_source = None
            predictions_gtd_target = None
            
        source = self._modules['Gy'](source)

        logits = self.fc_classifier_video(source)
        
        return logits, {"pred_gsd_source": predictions_gsd_source,"pred_gsd_target": predictions_gsd_target, \
                        "pred_gtd_source": predictions_gtd_source,"pred_gtd_target": predictions_gtd_target}
    
   
    class SpatialModule(nn.Module):
        def __init__(self, n_fcl, in_features_dim, out_features_dim, dropout=0.5):
            
            super(BaselineTA3N.SpatialModule, self).__init__()
            
            fc_layers = []
            fc_layers.append(BaselineTA3N.FullyConnectedLayer(in_features_dim, out_features_dim, dropout))

            for i in range(n_fcl-1):
                fc_layers.append(BaselineTA3N.FullyConnectedLayer(out_features_dim, out_features_dim, dropout))
            
            self.fc_layers = nn.Sequential(fc_layers)
            
            self.bias = self.fc_layers.bias
            self.weight = self.fc_layers.weight

            std = 0.001
            normal_(self.weight, 0, std)
            constant_(self.bias, 0)
        
        def forward(self, x):
            return self.fc_layers(x)
            

    class FullyConnectedLayer(nn.Module):
        def __init__(self, in_features_dim, out_features_dim, dropout=0.5):
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
                self.trn = TRNmodule.RelationModuleMultiScale(in_features_dim, self.num_bottleneck, self.train_segments)
                self.out_features_dim = self.num_bottleneck
                pass
            else:
                raise NotImplementedError
        
        def forward(self, x, num_segments):
            if self.pooling_type == 'TemRelation':
                x = x.view((-1, num_segments) + x.size()[-1:])
                x = self.trn(x)
                return torch.sum(x, 1)
            elif self.pooling_type =="TemPooling":
                x = x.view((-1, 1, num_segments) + x.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
                if x is None:
                    raise UserWarning('Reshape view no good')

                x = nn.AvgPool2d([num_segments, 1])(x)  # e.g. 16 x 1 x 1 x 512

                if x is None:
                    raise UserWarning('avgpool2d no good')

                x= x.squeeze(1).squeeze(1)  # e.g. 16 x 512
                
                if x is None:
                    raise UserWarning('Reshape squeeze no good')
                return x
            else:
                raise NotImplementedError
               
    

    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x, beta):
            ctx.beta = beta
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.neg() * ctx.beta
            return grad_input, None
    
    class DomainClassifier(nn.Module):

        def __init__(self, in_features_dim, beta):

            std = 0.001

            super(BaselineTA3N.DomainClassifier, self).__init__()
            self.in_features_dim = in_features_dim
            self.domain_classifier = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(self.in_features_dim, self.in_features_dim)),
                ('relu1', nn.ReLU(inplace=True)),
                ('linear2', nn.Linear(self.in_features_dim, 2))
            ]))
            self.beta = beta

            self.bias = nn.ParameterList([self.domain_classifier[0].bias, self.domain_classifier[2].bias])
            
            self.weight = nn.ParameterList([self.domain_classifier[0].weight, self.domain_classifier[2].weight])

            for bias in self.bias:
                constant_(bias, 0)
            
            for weight in self.weight:
                normal_(weight, 0, std)
                    
        def forward(self, x):
            BaselineTA3N.GradReverse.apply(x,self.beta)
            x = self.domain_classifier(x)
            return x