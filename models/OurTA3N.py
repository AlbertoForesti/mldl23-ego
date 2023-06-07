import models
import torch
import torch.nn as nn
import itertools
from torch.autograd import Function
from scipy.special import factorial, comb
from random import randint
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
		
        self.end_points[end_point] = fc_spatial_module # spatial module is just a fully connected layer
        
        if self._final_endpoint == end_point:
            return
        
        end_point = 'Gsd'
        if end_point in self.model_config.blocks:
            self.end_points[end_point] = self.DomainClassifier(in_features_dim, model_config.beta0)
            if self._final_endpoint == end_point:
                return
        
        end_point = 'Temporal module'
        self.end_points[end_point] = self.TemporalModule(in_features_dim, self.train_segments, temporal_pooling=model_config.frame_aggregation, model_config=self.model_config)
        in_features_dim = self.end_points[end_point].out_features_dim
        if self._final_endpoint == end_point:
            return
        
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
            predictions_gsd_target = self._modules['Gsd'](target.view((-1,1024)))
        else:
            predictions_gsd_source = None
            predictions_gsd_target = None
        

        predictions_grd_target = None
        feats_trn_target = feats_trn_source = None
        predictions_cop_source = predictions_cop_target = None
        labels_predictions_cop_source = labels_predictions_cop_target = None
        attn_weights_cop = None

        if self.model_config.frame_aggregation != 'COP':
            source, feats_trn_source = self._modules['Temporal module'](source, num_segments, is_train=is_train)
            if is_train:
                target, feats_trn_target = self._modules['Temporal module'](target, num_segments)
        else:
            source, predictions_cop_source, labels_predictions_cop_source, attn_weights_cop = self._modules['Temporal module'](source, num_segments, is_train=is_train)
            if is_train:
                target, predictions_cop_target, labels_predictions_cop_target, _ = self._modules['Temporal module'](target, num_segments, is_train=is_train)
        if not is_train:
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

        if 'Gtd' in self.end_points and is_train:
            predictions_gtd_source = self._modules['Gtd'](source)
            predictions_gtd_target = self._modules['Gtd'](target)
        else:
            predictions_gtd_source = None
            predictions_gtd_target = None
        
        feats_gy_source = self._modules['Gy'](source)

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
                        "label_cop_source": labels_predictions_cop_source,"label_cop_target": labels_predictions_cop_target, \
                        "attn_weights_cop": attn_weights_cop}

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
        def __init__(self, in_features_dim, n_clips, dropout=0.5, attention=False):
            super(BaselineTA3N.COPNet, self).__init__()
            self.bn = nn.BatchNorm1d(in_features_dim)
            self.iter = 0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.in_features_dim = in_features_dim
            self.n_clips = n_clips
            self.fc_pairwise_relations = BaselineTA3N.FullyConnectedLayer(in_features_dim=2*in_features_dim, out_features_dim=in_features_dim, dropout=dropout)
            self.num_classes = factorial(3, exact=True) # all possible permutations
            self.n_relations = comb(3, 2, exact=True)
            self.permutations = list(itertools.permutations([i for i in range(3)], r=3))
            self.fc_video = BaselineTA3N.FullyConnectedLayer(in_features_dim=self.n_relations*in_features_dim, out_features_dim=6)
            self.attention = attention
        
        def forward(self, x, num_segments):
            self.iter += 1
            shape = x.shape
            weighted_input = torch.empty((0,)+ shape[1:]).to(self.device)
            order_preds_all = torch.empty((0,len(self.permutations))).to(self.device)
            labels = torch.empty((0,len(self.permutations))).to(self.device)
            permutation = self.permutations[randint(0,len(self.permutations)-1)]
            
            if self.iter == 7500:
                params = []
                for param in self.parameters():
                    print(param.grad)
                    if param.grad is None:
                        raise UserWarning(f'None params')
                    params.append(param.grad.abs().mean().item())
                raise UserWarning(f'params grad {params}')

            tmp = list(itertools.combinations(range(x.shape[1]), 3))
            x = x.view(-1, shape[-1])
            x = self.bn(x)
            x = x.view(shape)
            x = x[:,tmp[randint(0, len(tmp)-1)],:]
            permuted_video = x[:, permutation, :]
            row_indices = list(range(permuted_video.shape[1]))
            combinations = list(itertools.combinations(row_indices, 2))

            

            first_iteration = True
            for combination in combinations:
                tensors = ()
                for index in combination:
                    tensors = tensors + (permuted_video[:, index, :],)
                relation_feats_concatenated = torch.cat(tensors, 1)
                relation_feats_fc = self.fc_pairwise_relations(relation_feats_concatenated)
                relation_feats_fc = self.bn(relation_feats_fc)
                if first_iteration:
                    relation_feats_fc_concatenated = relation_feats_fc
                    first_iteration = False
                else:
                    relation_feats_fc_concatenated = torch.cat((relation_feats_fc_concatenated, relation_feats_fc), 1)
            order_preds_all = self.fc_video(relation_feats_fc_concatenated)
            dist = []
            for p in self.permutations:
                if p == permutation:
                    dist.append(1)
                else:
                    dist.append(0)
            
            dist = torch.Tensor(dist).to(self.device)
            labels = dist.repeat(x.shape[0],1)

            raise UserWarning(f'{labels}')

            attn_weights = self.get_attn(order_preds_all, permutation)

            
            if self.attention:
                
                weighted_input = (attn_weights+1).t().unsqueeze(2).repeat(1,1,x.shape[-1]) * x
            
            if self.attention:
                return order_preds_all, labels, weighted_input, attn_weights.var(dim=1)
            else:
                return order_preds_all, labels, x, None

        def get_attn(self, order_preds, permutation):
            softmax = nn.Softmax(dim=1)
            if self.iter == 7500:
                raise UserWarning(f'order preds {order_preds}')
            probs = softmax(order_preds) #32 x 120

            if self.iter == 7500:
                raise UserWarning(f'probs {probs}')
            
            weights = torch.empty((0,probs.shape[0])).to(self.device) # 5 x 32
            for new_order, original_order in enumerate(permutation): # iterates 5 times (number of clips in video)
                correct_pred_indices = self.get_correct_pred_indices(original_order, new_order)
                # raise UserWarning(f'permutation: {permutation}, probs shape: {probs.shape}\n correct pred indeces: {correct_pred_indices}')
                probs_weird = torch.sum(probs[:,correct_pred_indices], dim=1).unsqueeze(0)
                # raise UserWarning(f'weights={weights.shape}, probs={probs.shape}, probs_weird = {probs_weird.shape}')
                weights = torch.cat((weights, probs_weird))
            return weights
        
        def get_correct_pred_indices(self, original_order, new_order):
            indices = []
            for i, permutation in enumerate(self.permutations):
                if permutation[new_order] == original_order:
                    indices.append(i)
            return indices

    class FullyConnectedLayer(nn.Module):
        def __init__(self, in_features_dim, out_features_dim, dropout=0.5):
            super(BaselineTA3N.FullyConnectedLayer, self).__init__()
            self.in_features_dim = in_features_dim
            self.out_features_dim = out_features_dim
            
            """Here I am doing what is done in the official code, 
            in the first fc layer the output dimension is the minimum between the input feature dimension and 1024"""
            self.relu = nn.LeakyReLU() # Again using the architecture of the official code
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
        def __init__(self, in_features_dim, train_segments, temporal_pooling = 'TemPooling', model_config=None) -> None:
            super(BaselineTA3N.TemporalModule, self).__init__()
            self.pooling_type = temporal_pooling
            self.in_features_dim = in_features_dim
            self.train_segments = train_segments
            self.model_config = model_config
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if temporal_pooling == 'TemPooling' or temporal_pooling == 'COP':
                self.out_features_dim = self.in_features_dim
                if temporal_pooling == 'COP':
                    self.cop = BaselineTA3N.COPNet(in_features_dim, model_config.train_segments, dropout=model_config.dropout, attention=model_config.attention_cop)
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

            elif self.pooling_type == "COP":
                order_preds, labels, weighted_input, attn_weights = self.cop(x, self.train_segments)
                if self.model_config.attention_cop:
                    x, _ = self.tempooling(weighted_input, num_segments)
                    return  x, order_preds, labels, attn_weights
                else:
                    x, _ = self.tempooling(x, num_segments)
                    return x, order_preds, labels, attn_weights
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