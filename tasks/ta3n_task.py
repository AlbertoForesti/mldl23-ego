from abc import ABC
import torch
from utils import utils
from functools import reduce
import wandb
import tasks
from utils.logger import logger

from typing import Dict, Tuple


class ActionRecognition(tasks.Task, ABC):
    """Action recognition model."""
    
    def __init__(self, name: str, task_models: Dict[str, torch.nn.Module], batch_size: int, 
                 total_batch: int, models_dir: str, num_classes: int,
                 num_clips: int, model_args: Dict[str, float], args, **kwargs) -> None:
        """Create an instance of the action recognition model.

        Parameters
        ----------
        name : str
            name of the task e.g. action_classifier, domain_classifier...
        task_models : Dict[str, torch.nn.Module]
            torch models, one for each different modality adopted by the task
        batch_size : int
            actual batch size in the forward
        total_batch : int
            batch size simulated via gradient accumulation
        models_dir : str
            directory where the models are stored when saved
        num_classes : int
            number of labels in the classification task
        num_clips : int
            number of clips
        model_args : Dict[str, float]
            model-specific arguments
        """
        super().__init__(name, task_models, batch_size, total_batch, models_dir, args, **kwargs)
        self.model_args = model_args

        # self.accuracy and self.loss track the evolution of the accuracy and the training loss
        self.accuracy = utils.Accuracy(topk=(1, 5), classes=num_classes)
        
        self.classification_loss  = utils.AverageMeter()
        self.gsd_loss = utils.AverageMeter()
        self.gtd_loss = utils.AverageMeter()
        self.grd_loss = utils.AverageMeter()
        self.lae_loss = utils.AverageMeter()
        self.cop_loss = utils.AverageMeter()

        self.attn_cop_weights = [utils.AverageMeter() for i in range(5)]

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_clips = num_clips

        # Use the cross entropy loss as the default criterion for the classification task
        self.criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                   reduce=None, reduction='none')
        
        # Initializeq the model parameters and the optimizer
        optim_params = {}
        self.optimizer = dict()
        for m in self.modalities:
            optim_params[m] = filter(lambda parameter: parameter.requires_grad, self.task_models[m].parameters())
            self.optimizer[m] = torch.optim.SGD(optim_params[m], model_args[m].lr,
                                                weight_decay=model_args[m].weight_decay,
                                                momentum=model_args[m].sgd_momentum)

    def forward(self, data_source: Dict[str, torch.Tensor], data_target: Dict[str, torch.Tensor]=None, is_train=True,**kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Forward step of the task

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            a dictionary that stores the input data for each modality 

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            output logits and features
        """

        logits = {}
        features = {}
        for i_m, m in enumerate(self.modalities):
            if is_train:
                logits[m], feat = self.task_models[m](data_source[m], data_target[m], is_train=is_train,**kwargs)
            else:
                logits[m], feat = self.task_models[m](data_source[m], None, is_train=is_train,**kwargs)
            if i_m == 0:
                for k in feat.keys():
                    features[k] = {}
            for k in feat.keys():
                features[k][m] = feat[k]

        return logits, features
    
    def compute_loss(self, logits: torch.Tensor, class_label: torch.Tensor, features: Dict[str, torch.Tensor]):
        classification_loss = self.criterion(logits, class_label) #cross entropy loss

        if 'Gsd' in self.model_args['RGB'].blocks:
            pred_gsd_source = features['pred_gsd_source']
            domain_label_source=torch.zeros(pred_gsd_source.shape[0], dtype=torch.int64)    
            
            pred_gsd_target = features['pred_gsd_target']
            domain_label_target=torch.ones(pred_gsd_target.shape[0], dtype=torch.int64)

            domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
            pred_gsd_all=torch.cat((pred_gsd_source, pred_gsd_target),0)

            gsd_loss = self.criterion(pred_gsd_all, domain_label_all)
            self.gsd_loss.update(torch.mean(gsd_loss) / (self.total_batch / self.batch_size), self.batch_size) # this shouldn't be a cross-entropy loss tbh, look at paper
        
        if  self.model_args['RGB'].frame_aggregation == 'COP':
            pred_cop_source = features['pred_cop_source']
            pred_cop_target = features['pred_cop_target']

            label_cop_source = features['label_cop_source']
            label_cop_target = features['label_cop_target']
            
            pred_cop_all = torch.cat((pred_cop_source, pred_cop_target),0)
            label_cop_all = torch.cat((label_cop_source, label_cop_target),0)

            cop_loss = self.criterion(pred_cop_all, label_cop_all)
            self.cop_loss.update(torch.mean(cop_loss) / (self.total_batch / self.batch_size), self.batch_size)

            if self.model_args['RGB'].attention_cop == 'Yes':
                for i in range(5):
                    self.attn_cop_weights[i].update(features['attn_weights_cop'][i])

        if 'Gtd' in self.model_args['RGB'].blocks:
            pred_gtd_source = features['pred_gtd_source']
            domain_label_source=torch.zeros(pred_gtd_source.shape[0], dtype=torch.int64)
        
            pred_gtd_target = features['pred_gtd_target']
            domain_label_target=torch.ones(pred_gtd_target.shape[0], dtype=torch.int64)
            
            domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
            pred_gtd_all=torch.cat((pred_gtd_source,pred_gtd_target),0)

            gtd_loss = self.criterion(pred_gtd_all, domain_label_all)
            self.gtd_loss.update(torch.mean(gtd_loss) / (self.total_batch / self.batch_size), self.batch_size)
            
            if 'ta3n' in self.model_args['RGB'].blocks:
                pred_clf_all = torch.cat((logits, features['pred_clf_target']))
                
                """entropy_gtd = torch.special.entr(pred_gtd_all).sum(dim=1)
                entropy_clf = torch.special.entr(pred_clf_all).sum(dim=1)
                lae_loss = entropy_clf + torch.mul(entropy_clf, entropy_gtd)
                self.lae_loss.update(torch.mean(lae_loss)/(self.total_batch / self.batch_size), self.batch_size)"""

                lae_loss = self.attentive_entropy(pred_clf_all, pred_gtd_all)
                self.lae_loss.update(lae_loss/(self.total_batch / self.batch_size), self.batch_size)
        
        if 'Grd' in self.model_args['RGB'].blocks and self.model_args['RGB'].frame_aggregation == 'TemRelation':
            grd_loss = []
            for pred_grd_source_single_scale, pred_grd_target_single_scale in zip(features['pred_grd_source'].values(), features['pred_grd_target'].values()):
                domain_label_source = torch.zeros(pred_grd_source_single_scale.shape[0], dtype=torch.int64)
                domain_label_target = torch.ones(pred_grd_target_single_scale.shape[0], dtype=torch.int64)

                domain_label_all=torch.cat((domain_label_source, domain_label_target),0).to(self.device)
                pred_grd_all_single_scale = torch.cat((pred_grd_source_single_scale, pred_grd_target_single_scale))

                grd_loss_single_scale = self.criterion(pred_grd_all_single_scale, domain_label_all)
                grd_loss.append(grd_loss_single_scale)
            grd_loss = sum(grd_loss)/(len(grd_loss))
            self.grd_loss.update(torch.mean(grd_loss) / (self.total_batch / self.batch_size), self.batch_size)

        
        
        # self.loss.update((torch.mean(classification_loss) - torch.mean(lambda_s*gsd_loss + lambda_t*gtd_loss) )/ (self.total_batch / self.batch_size), self.batch_size)
        # we need different losses to backpropagate to different parts of the network
        
        self.classification_loss.update(torch.mean(classification_loss) / (self.total_batch / self.batch_size), self.batch_size)
    

    def attentive_entropy(self, pred, pred_domain):
        softmax = torch.nn.Softmax(dim=1)
        logsoftmax = torch.nn.LogSoftmax(dim=1)

        # attention weight
        entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
        weights = 1 + entropy

        # attentive entropy
        loss = torch.mean(weights * torch.sum(-softmax(pred) * logsoftmax(pred), 1))
        
        return loss
    
    def compute_accuracy(self, logits: Dict[str, torch.Tensor], label: torch.Tensor):
        """Fuse the logits from different modalities and compute the classification accuracy.

        Parameters
        ----------
        logits : Dict[str, torch.Tensor]
            logits of the different modalities
        label : torch.Tensor
            ground truth
        """
        # fused_logits = reduce(lambda x, y: x + y, logits.values())

        self.accuracy.update(logits, label)

    def wandb_log(self):
        """Log the current loss and top1/top5 accuracies to wandb."""
        logs = {
            'loss verb': self.classification_loss.val,
            'top1-accuracy': self.accuracy.avg[1],
            'top5-accuracy': self.accuracy.avg[5]
        }
        a = self.model_args['RGB'].attention_cop
        b = a == 'Yes'
        raise UserWarning(f'self.model_args[\'RGB\'].attention_cop is {a} and {b}')
        if self.model_args['RGB'].attention_cop == 'Yes':
            
            for i in range(5):
                logs[f'attn_cop_{i}']=self.attn_cop_weights[i].avg

        # Log the learning rate, separately for each modality.
        for m in self.modalities:
            logs[f'lr_{m}'] = self.optimizer[m].param_groups[-1]['lr']
        wandb.log(logs)

    def reduce_learning_rate(self):
        """Perform a learning rate step."""
        for m in self.modalities:
            prev_lr = self.optimizer[m].param_groups[-1]["lr"]
            new_lr = self.optimizer[m].param_groups[-1]["lr"] / 10
            self.optimizer[m].param_groups[-1]["lr"] = new_lr

            logger.info(f"Reducing learning rate modality {m}: {prev_lr} --> {new_lr}")

    def reset_loss(self):
        """Reset the classification loss.

        This method must be called after each optimization step.
        """
        if 'Gsd' in self.model_args['RGB'].blocks:
            self.gsd_loss.reset()
        
        if 'Gtd' in self.model_args['RGB'].blocks:
            self.gtd_loss.reset()
            if 'ta3n' in self.model_args['RGB'].blocks:
                self.lae_loss.reset()
        
        if 'Grd' in self.model_args['RGB'].blocks and self.model_args['RGB'].frame_aggregation == 'TemRelation':
            self.grd_loss.reset()
        
        if  self.model_args['RGB'].frame_aggregation == 'COP':
            self.cop_loss.reset()

        self.classification_loss.reset()

    def reset_acc(self):
        """Reset the classification accuracy."""
        self.accuracy.reset()

    def step(self):
        """Perform an optimization step.

        This method performs an optimization step and resets both the loss
        and the accuracy.
        """
        super().step()
        self.reset_loss()
        self.reset_acc()

    def backward(self, retain_graph: bool = False):
        """Compute the gradients for the current value of the classification loss.

        Set retain_graph to true if you need to backpropagate multiple times over
        the same computational graph.

        Parameters
        ----------
        retain_graph : bool, optional
            whether the computational graph should be retained, by default False
        """
        # self.loss.val.backward(retain_graph=retain_graph)

        loss = 0

        if 'Gsd' in self.model_args['RGB'].blocks:
            loss += self.gsd_loss.val
        
        if 'Gtd' in self.model_args['RGB'].blocks:
            loss += self.gtd_loss.val
            if 'ta3n' in self.model_args['RGB'].blocks:
                loss += self.model_args['RGB'].gamma * self.lae_loss.val
        
        if 'Grd' in self.model_args['RGB'].blocks and self.model_args['RGB'].frame_aggregation == 'TemRelation':
            loss += self.grd_loss.val

        if  self.model_args['RGB'].frame_aggregation == 'COP':
            loss += self.model_args['RGB'].delta*self.cop_loss.val

        loss += self.classification_loss.val

        loss.backward(retain_graph=retain_graph)
