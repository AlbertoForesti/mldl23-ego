import models
import torch
import torch.nn as nn
import itertools
from torch.autograd import Function
from scipy.special import factorial, comb
from numpy.random import randint
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
		
        self.end_points[end_point] = fc_spatial_module # spatial module is just a fully connected layer
        
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Gsd'
        if end_point in self.model_config.blocks:
            self.end_points[end_point] = self.DomainClassifier(in_features_dim, model_config.beta0)
            if self._final_endpoint == end_point:
                return

        if 'copnet' in self.model_config.cop_type:
            out_features_dim_copnet = 2 if 'simple' in self.model_config.cop_type else factorial(self.model_config.cop_samples, exact=True)
            self.permute_type = 'simple' if 'simple' in self.model_config.cop_type else 'complex'
            self.end_points['copnet'] = self.COPNet(in_features_dim, out_features_dim_copnet)
        
        end_point = 'Temporal module'
        self.end_points[end_point] = self.TemporalModule(in_features_dim, self.train_segments, temporal_pooling=model_config.frame_aggregation, model_config=self.model_config)
        in_features_dim = self.end_points[end_point].out_features_dim
        if self._final_endpoint == end_point:
            return

        if 'trn' in self.model_config.cop_type:
            out_features_dim_copnet = 2 if 'simple' in self.model_config.cop_type else factorial(self.model_config.cop_samples, exact=True)
            self.permute_type = 'simple' if 'simple' in self.model_config.cop_type else 'complex'
            if 'unified' in self.model_config.cop_type:
                end_point_name = 'copnet_trn_unified'
            else:
                end_point_name = 'copnet_trn_separate'
            self.end_points[end_point_name] = self.FullyConnectedLayer(in_features_dim, out_features_dim_copnet)
        
        if 'Grd' in self.model_config.blocks and 'Temporal module' in self.end_points and self.model_config.frame_aggregation == 'TemRelation':
            for i in range(self.train_segments-1):
                self.end_points[f'Grd_{i}'] = self.DomainClassifier(self.end_points['Temporal module'].num_bottleneck,model_config.beta2)
        
        end_point = 'Gtd'
        if end_point in self.model_config.blocks:
            self.end_points[end_point] = self.DomainClassifier(in_features_dim, model_config.beta1)
            if self._final_endpoint == end_point:
                return
        
        end_point = 'Gy'
        fc_gy = self.FullyConnectedLayer(in_features_dim=in_features_dim, out_features_dim=in_features_dim, dropout=model_config.dropout)

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

    def get_trans_attn(self, pred_domain):
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 - entropy
        return weights

    def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
        weights_attn = self.get_trans_attn(pred_domain)

        weights_attn = weights_attn.view(-1, num_segments-1, 1).repeat(1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 4 x 256)
        # raise UserWarning(f'Dimensions are weights_attn={weights_attn.shape}, feat_fc={feat_fc.shape}')
        feat_fc_attn = (weights_attn+1) * feat_fc

        return feat_fc_attn, weights_attn[:,:,0]

    def forward(self, source, target=None, is_train=True):
        num_segments = self.train_segments if is_train else self.val_segments
        
        source = self._modules['Spatial module'](source)

        if is_train:
            target = self._modules['Spatial module'](target)
        
        if 'Gsd' in self.end_points and is_train:
            predictions_gsd_source = self._modules['Gsd'](source.view((-1,1024))) # to concat
            predictions_gsd_target = self._modules['Gsd'](target.view((-1,1024))) # aa
        else:
            predictions_gsd_source = None
            predictions_gsd_target = None
        
        predictions_grd_target = None
        feats_trn_target = feats_trn_source = None
        predictions_cop_source = predictions_cop_target = None
        labels_predictions_cop_source = labels_predictions_cop_target = None

        if 'copnet' in self.end_points and is_train:
            permuted_source, labels_predictions_cop_source = self._permute(source, self.permute_type)
            permuted_target, labels_predictions_cop_target = self._permute(target, self.permute_type)
            predictions_cop_source = self._modules['copnet'](permuted_source)
            predictions_cop_target = self._modules['copnet'](permuted_target)
            """raise UserWarning(f'Predictions {labels_predictions_cop_source}\
                              \nPermuted {source}\
                              \nOriginal {permuted_source}\
                              \nPredictions {predictions_cop_source}')"""
        
        if 'copnet_trn_unified' in self.end_points:
            source, labels_predictions_cop_source = self._permute(source, self.permute_type, sample_clips=self.model_config.cop_samples)
            if is_train:
                target, labels_predictions_cop_target = self._permute(target, self.permute_type, sample_clips=self.model_config.cop_samples)
        
        if 'copnet_trn_separate' in self.end_points and is_train:
            permuted_source, labels_predictions_cop_source = self._permute(source, self.permute_type, sample_clips=self.model_config.cop_samples)
            permuted_source, _ = self._modules['Temporal module'](source, num_segments)
            permuted_source = torch.sum(permuted_source, 1)
            predictions_cop_source = self._modules['copnet_trn_separate'](permuted_source)

            if is_train:
                permuted_target, labels_predictions_cop_target = self._permute(target, self.permute_type, sample_clips=self.model_config.cop_samples)
                permuted_target, _ = self._modules['Temporal module'](target, num_segments)
                permuted_target = torch.sum(permuted_target, 1)
                predictions_cop_target = self._modules['copnet_trn_separate'](permuted_target)

        source, feats_trn_source = self._modules['Temporal module'](source, num_segments, is_train=is_train)
        if is_train:
            target, feats_trn_target = self._modules['Temporal module'](target, num_segments)
        else:
            target=None

        if 'Grd' in self.model_config.blocks and self.model_config.frame_aggregation == 'TemRelation':
            predictions_grd_source = {}
            for i, feats_trn_source_single_scale in enumerate(feats_trn_source.values()):
                predictions_grd_source[f'Grd_{i}'] = self._modules[f'Grd_{i}'](feats_trn_source_single_scale)

            if is_train:
                predictions_grd_target = {}
                for i, feats_trn_target_single_scale in enumerate(feats_trn_target.values()):
                    predictions_grd_target[f'Grd_{i}'] = self._modules[f'Grd_{i}'](feats_trn_target_single_scale)
            else:
                predictions_grd_target = None
        else:
            predictions_grd_source = None
            predictions_grd_target = None
        
        if self.model_config.attention:

            tensors = ()
            for pred in predictions_grd_source.values():
                tensors = tensors + (pred.view(-1,1,2),)

            pred_fc_domain_relation_video_source = torch.cat(tensors,1).view(-1,2)
            source, _ = self.get_attn_feat_relation(source, pred_fc_domain_relation_video_source, num_segments)

            if is_train:
                tensors = ()
                for pred in predictions_grd_target.values():
                    tensors = tensors + (pred.view(-1,1,2),)

                pred_fc_domain_relation_video_target = torch.cat(tensors,1).view(-1,2)
                target, _ = self.get_attn_feat_relation(target, pred_fc_domain_relation_video_target, num_segments)

        if self.model_config.frame_aggregation == 'TemRelation':
            source = torch.sum(source, 1)
            if is_train:
                target = torch.sum(target, 1)
        
        if 'copnet_trn_unified' in self.end_points:
            predictions_cop_source = self._modules['copnet_trn_unified'](source)
            if is_train:
                predictions_cop_target = self._modules['copnet_trn_unified'](target)

        if 'Gtd' in self.end_points and is_train:
            predictions_gtd_source = self._modules['Gtd'](source)
            predictions_gtd_target = self._modules['Gtd'](target)
        else:
            predictions_gtd_source = None
            predictions_gtd_target = None
        
        feats_gy_source = self._modules['Gy'](source)

        if is_train:
            feats_gy_target = self._modules['Gy'](target)
        else:
            feats_gy_target = None
        
        logits = self.fc_classifier_video(feats_gy_source)
        predictions_clf_source = logits

        # raise UserWarning(f'source = {source.shape}, gy source = {feats_gy_source.shape}, logits = {logits.shape}')
        
        if is_train:
            predictions_clf_target = self.fc_classifier_video(target)
        else:
            predictions_clf_target = None
        
        return logits, {"pred_gsd_source": predictions_gsd_source,"pred_gsd_target": predictions_gsd_target, \
                        "pred_gtd_source": predictions_gtd_source,"pred_gtd_target": predictions_gtd_target, \
                        "pred_grd_source": predictions_grd_source,"pred_grd_target": predictions_grd_target, \
                        "pred_cop_source": predictions_cop_source,"pred_cop_target": predictions_cop_target, \
                        "pred_clf_source": predictions_clf_source,"pred_clf_target": predictions_clf_target, \
                        "feats_gy_source": feats_gy_source,"feats_gy_target": feats_gy_target, \
                        "label_cop_source": labels_predictions_cop_source,"label_cop_target": labels_predictions_cop_target}

    def _permute(self, x, permute_type, sample_clips=3):
        shape = x.shape
        batch_size = x.shape[0]
        permutations = list(itertools.permutations([i for i in range(sample_clips)], r=sample_clips))

        tmp = list(itertools.combinations(range(x.shape[1]), sample_clips))
        x_permuted = x[:,tmp[randint(0, len(tmp))],:] # sample x clips out of five
            

        if permute_type == 'simple':
            shift_mask = randint(0,2,batch_size)
            permutation_vector = torch.Tensor([permutations[randint(0,len(permutations))] if i == 0 else [j for j in range(sample_clips)] for i in shift_mask]).long().to(self.device)
        else:
            shift_mask = randint(0,len(permutations),batch_size)
            permutation_vector = torch.Tensor([permutations[i] for i in shift_mask]).long().to(self.device)
        
        permutation_vector = permutation_vector.unsqueeze(2)
        repeat_shape = torch.Size([1,1,shape[2]])
        permutation_vector = permutation_vector.repeat(repeat_shape)
        x_permuted = torch.gather(x_permuted,1,permutation_vector)
        labels = torch.Tensor(shift_mask).long().to(self.device)
        return x_permuted, labels

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
    
    class COPNet(nn.Module):
        def __init__(self, in_features_dim, out_features_dim, must_aggregate_clips=False, dropout=0.2):
            super(BaselineTA3N.COPNet, self).__init__()
            self.bn = nn.BatchNorm1d(in_features_dim)
            self.in_features_dim = in_features_dim
            self.fc_pairwise_relations = BaselineTA3N.FullyConnectedLayer(in_features_dim=2*in_features_dim, out_features_dim=in_features_dim, dropout=dropout)
            self.num_classes = factorial(3, exact=True) # all possible permutations
            self.n_relations = comb(3, 2, exact=True)
            self.permutations = list(itertools.permutations([i for i in range(3)], r=3))
            self.fc_video = BaselineTA3N.FullyConnectedLayer(in_features_dim=self.n_relations*in_features_dim, out_features_dim=out_features_dim)
            
        def forward(self, x):
            shape = x.shape
            x = x.view(-1, shape[-1])
            x = self.bn(x)
            x = x.view(shape)
            row_indices = list(range(x.shape[1]))
            combinations = list(itertools.combinations(row_indices, 2))
            first_iteration = True
            for combination in combinations:
                tensors = ()
                for index in combination:
                    tensors = tensors + (x[:, index, :],)
                relation_feats_concatenated = torch.cat(tensors, 1)
                relation_feats_fc = self.fc_pairwise_relations(relation_feats_concatenated)
                relation_feats_fc = self.bn(relation_feats_fc)
                if first_iteration:
                    relation_feats_fc_concatenated = relation_feats_fc
                    first_iteration = False
                else:
                    relation_feats_fc_concatenated = torch.cat((relation_feats_fc_concatenated, relation_feats_fc), 1)
            order_preds_all = self.fc_video(relation_feats_fc_concatenated)
            return order_preds_all

    class FullyConnectedLayer(nn.Module):
        def __init__(self, in_features_dim, out_features_dim, dropout=0.5):
            super(BaselineTA3N.FullyConnectedLayer, self).__init__()
            self.in_features_dim = in_features_dim
            self.out_features_dim = out_features_dim
            
            """Here I am doing what is done in the official code, 
            in the first fc layer the output dimension is the minimum between the input feature dimension and 1024"""
            self.relu = nn.ReLU() # Again using the architecture of the official code
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(self.in_features_dim, self.out_features_dim)
            self.bias = self.fc.bias
            self.weight = self.fc.weight
            std = 0.001
            normal_(self.fc.weight, 0, std)
            constant_(self.fc.bias, 0)
        
        def forward(self, x):
            x = self.fc(x)
            x = self.relu(x)
            x = self.dropout(x)
            return x


    class TemporalModule(nn.Module):
        def __init__(self, in_features_dim, train_segments, temporal_pooling = 'TemPooling', model_config=None, cop_trn=False) -> None:
            super(BaselineTA3N.TemporalModule, self).__init__()
            self.pooling_type = temporal_pooling
            self.in_features_dim = in_features_dim
            self.train_segments = train_segments
            self.model_config = model_config
            self.cop_trn = cop_trn
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if temporal_pooling == 'TemPooling':
                self.out_features_dim = self.in_features_dim
            elif temporal_pooling == 'TemRelation':
                self.num_bottleneck = 512
                self.trn = TRNmodule.RelationModuleMultiScale(in_features_dim, self.num_bottleneck, self.train_segments)
                self.out_features_dim = self.num_bottleneck
            else:
                raise NotImplementedError
        
        def tempooling(self, x, num_segments):
            x = x.view((-1, 1, num_segments) + x.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)
            if x is None:
                raise UserWarning('Reshape view no good')

            x = nn.AvgPool2d([num_segments, 1])(x)  # e.g. 16 x 1 x 1 x 512

            if x is None:
                raise UserWarning('avgpool2d no good')

            x = x.squeeze(1).squeeze(1)  # e.g. 16 x 512
            
            if x is None:
                raise UserWarning('Reshape squeeze no good')
            return x, None
        
        def forward(self, x, num_segments, is_train=True):
            if self.pooling_type == 'TemRelation':
                x = x.view((-1, num_segments) + x.size()[-1:])
                x, feats = self.trn(x)
                return x, feats
                
            elif self.pooling_type == "TemPooling":
                return self.tempooling(x, num_segments)

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
            x = BaselineTA3N.GradReverse.apply(x,self.beta)
            x = self.domain_classifier(x)
            return x
