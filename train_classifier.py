from datetime import datetime
from statistics import mean
from utils.logger import logger
import torch.nn.parallel
import torch.optim
from torch.utils.data import Subset
import torch
from utils.loaders import EpicKitchensDataset
from utils.args import args
from utils.utils import pformat_dict
import utils
import numpy as np
import os
import models as model_list
import tasks
import wandb

# global variables among training functions
training_iterations = 0
modalities = None
np.random.seed(13696641)
torch.manual_seed(13696641)


def init_operations():
    """
    parse all the arguments, generate the logger, check gpus to be used and wandb
    """
    logger.info("Running with parameters: " + pformat_dict(args, indent=1))

    # this is needed for multi-GPUs systems where you just want to use a predefined set of GPUs
    if args.gpus is not None:
        logger.debug('Using only these GPUs: {}'.format(args.gpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)

    # wanbd logging configuration
    if args.wandb_name is not None:
        wandb.login(key='5fb520f5bf470ccfb910f234718b0bebff47d47d')
        config = {
            'lr': args.models['RGB'].lr,
            'dropout': args.models['RGB'].dropout, 
            'clip_aggregation': args.models['RGB'].frame_aggregation, 
            'blocks': args.models['RGB'].blocks,
            'attention_grd': args.models['RGB'].attention,
            'shift': args.dataset.shift}
        if 'Gsd' in args.models['RGB'].blocks:
            config['beta0'] = args.models['RGB'].beta0
        if 'Gtd' in args.models['RGB'].blocks:
            config['beta1'] = args.models['RGB'].beta1
        if 'Grd' in args.models['RGB'].blocks:
            config['beta2'] = args.models['RGB'].beta2
        if 'Grd' in args.models['RGB'] and args.models['RGB'].attention:
            config['gamma'] = args.models['RGB'].gamma
        if args.models['RGB'].frame_aggregation == 'COP':
            config['delta'] = args.models['RGB'].delta

        wandb.init(group=args.wandb_name, dir=args.wandb_dir, config=config)
        wandb.run.name = args.name + "_" + args.dataset.shift.split("-")[0] + "_" + args.dataset.shift.split("-")[-1]


def main():
    global training_iterations, modalities
    init_operations()

    modalities = args.modality

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    num_classes, valid_labels, source_domain, target_domain = utils.utils.get_domains_and_labels(args)
    # device where everything is run
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # these dictionaries are for more multi-modal training/testing, each key is a modality used
    models = {}
    logger.info("Instantiating models per modality")
    for m in modalities:
        logger.info('{} Net\tModality: {}'.format(args.models[m].model, m))
        # notice that here, the first parameter passed is the input dimension
        # In our case it represents the feature dimensionality which is equivalent to 1024 for I3D
        
        models[m] = getattr(model_list, args.models[m].model)(args.in_feat_dim, num_classes, args.models[m], **args.models[m].kwargs)

    # the models are wrapped into the ActionRecognition task which manages all the training steps
    action_classifier = tasks.ActionRecognition("action-classifier", models, args.batch_size,
                                                args.total_batch, args.models_dir, num_classes,
                                                args.train.num_clips, args.models, args=args)
    action_classifier.load_on_gpu(device)

    if args.action == "train":
        # resume_from argument is adopted in case of restoring from a checkpoint
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        # define number of iterations I'll do with the actual batch: we do not reason with epochs but with iterations
        # i.e. number of batches passed
        # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation technique is adopted
        training_iterations = args.train.num_iter * (args.total_batch // args.batch_size)
        # all dataloaders are generated here
        dataset_src = EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True)
        dataset_trg = EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True)
        train_loader_source = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        train_loader_target = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'val', args.dataset, None, None, None,
                                                                     None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)
        if args.debug:
            subset_size = 100  # Specify the desired size of the subset

            # Get the original dataset from the dataloader
            original_dataset_train_source = train_loader_source.dataset
            original_dataset_train_target = train_loader_target.dataset

            # Create a Subset object with the desired subset indices
            subset_indices = range(subset_size)  # Adjust the indices as per your requirements
            
            subset_train_source = Subset(original_dataset_train_source, subset_indices)
            subset_train_target = Subset(original_dataset_train_target, subset_indices)
            

            train_loader_source = torch.utils.data.DataLoader(subset_train_source,batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
            train_loader_target = torch.utils.data.DataLoader(subset_train_target,batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)

        train(action_classifier, train_loader_source, train_loader_target, val_loader, device, num_classes)
        dataloader_src = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[0], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=len(dataset_src), shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        dataloader_trg = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                       'train', args.dataset, None, None, None,
                                                                       None, load_feat=True),
                                                   batch_size=len(dataset_trg), shuffle=True,
                                                   num_workers=args.dataset.workers, pin_memory=True, drop_last=True)
        source_data, source_label = next(iter(dataloader_src))
        source_data = source_data['RGB'].to(device)
        target_data, target_label = next(iter(dataloader_trg))
        target_data = target_data['RGB'].to(device)
        data_source= {'RGB': source_data}
        data_target= {'RGB': target_data}
        logits, tmp = action_classifier.forward(data_source, data_target)
        features = {}
        for k, v in tmp.items():
            # features[k] = torch.mean(v.values())
            features[k] = v['RGB']
        feats_gy_source = features['feats_gy_source']
        feats_gy_target = features['feats_gy_target']
        torch.save(feats_gy_source, f"feats_source_all_{args.dataset.shift}.pt")
        torch.save(feats_gy_target, f"feats_target_all_{args.dataset.shift}.pt")


    elif args.action == "validate":
        if args.resume_from is not None:
            action_classifier.load_last_model(args.resume_from)
        val_loader = torch.utils.data.DataLoader(EpicKitchensDataset(args.dataset.shift.split("-")[-1], modalities,
                                                                     'val', args.dataset, None, None, None,
                                                                     None, load_feat=True),
                                                 batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.dataset.workers, pin_memory=True, drop_last=False)

        validate(action_classifier, val_loader, device, action_classifier.current_iter, num_classes)
    wandb.finish()


def train(action_classifier, train_loader_source, train_loader_target, val_loader, device, num_classes):
    """
    function to train the model on the test set
    action_classifier: Task containing the model to be trained
    train_loader: dataloader containing the training data
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    num_classes: int, number of classes in the classification problem
    """
    global training_iterations, modalities

    data_loader_source = iter(train_loader_source)
    data_loader_target = iter(train_loader_source)
    action_classifier.train(True)
    action_classifier.zero_grad()
    iteration = action_classifier.current_iter * (args.total_batch // args.batch_size)

    # the batch size should be total_batch but batch accumulation is done with batch size = batch_size.
    # real_iter is the number of iterations if the batch size was really total_batch
    for i in range(iteration, training_iterations):
        # iteration w.r.t. the paper (w.r.t the bs to simulate).... i is the iteration with the actual bs( < tot_bs)
        real_iter = (i + 1) / (args.total_batch // args.batch_size)
        if real_iter == args.train.lr_steps:
            # learning rate decay at iteration = lr_steps
            action_classifier.reduce_learning_rate()
        # gradient_accumulation_step is a bool used to understand if we accumulated at least total_batch
        # samples' gradient
        gradient_accumulation_step = real_iter.is_integer()

        """
        Retrieve the data from the loaders
        """
        start_t = datetime.now()
        # the following code is necessary as we do not reason in epochs so as soon as the dataloader is finished we need
        # to redefine the iterator
        try:
            source_data, source_label = next(data_loader_source)
            
        except StopIteration:
            data_loader_source = iter(train_loader_source)
            source_data, source_label = next(data_loader_source)
        
        try:
            target_data, target_label = next(data_loader_target)
            
        except StopIteration:
            data_loader_target= iter(train_loader_target)
            target_data, target_label = next(data_loader_target)
        end_t = datetime.now()

        ''' Action recognition'''
        source_label = source_label.to(device)
        target_label=target_label.to(device)
        data_source= {}
        data_target= {}

        
        # in case of multi-clip training one clip per time is processed
        for m in modalities:
            data_source[m] = source_data[m].to(device)
            data_target[m] = target_data[m].to(device)


        if data_source is None or data_target is None :
            raise UserWarning('train_classifier: Cannot be None type')
        logits, tmp = action_classifier.forward(data_source, data_target)

        features = {}
        for k, v in tmp.items():
            # features[k] = torch.mean(v.values())
            features[k] = v['RGB']
        
        logits = logits['RGB']

        action_classifier.compute_loss(logits, source_label, features)
        action_classifier.backward(retain_graph=False)
        action_classifier.compute_accuracy(logits, source_label)
        action_classifier.wandb_log()

        # update weights and zero gradients if total_batch samples are passed
        if gradient_accumulation_step:
            logger.info("[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                        (real_iter, args.train.num_iter, action_classifier.classification_loss.val, action_classifier.classification_loss.avg,
                         action_classifier.accuracy.val[1], action_classifier.accuracy.avg[1]))

            action_classifier.check_grad()
            action_classifier.step()
            action_classifier.zero_grad()

        # every eval_freq "real iteration" (iterations on total_batch) the validation is done, notice we validate and
        # save the last 9 models
        if gradient_accumulation_step and real_iter % args.train.eval_freq == 0:
            val_metrics = validate(action_classifier, val_loader, device, int(real_iter), num_classes)

            if val_metrics['top1'] <= action_classifier.best_iter_score:
                logger.info("New best accuracy {:.2f}%"
                            .format(action_classifier.best_iter_score))
            else:
                logger.info("New best accuracy {:.2f}%".format(val_metrics['top1']))
                action_classifier.best_iter = real_iter
                action_classifier.best_iter_score = val_metrics['top1']
                wandb.run.summary["best_accuracy"] = val_metrics['top1']
            wandb.log(val_metrics)

            action_classifier.save_model(real_iter, val_metrics['top1'], prefix=None)
            action_classifier.train(True)
    feats_gy_source = features['feats_gy_source']
    feats_gy_target = features['feats_gy_target']
    torch.save(feats_gy_source, f"feats_source_{args.dataset.shift}.pt")
    torch.save(feats_gy_target, f"feats_target_{args.dataset.shift}.pt")
    #src_tsne = TSNE(n_components=2).fit_transform(feats_gy_source)
    #trg_tsne = TSNE(n_components=2).fit_transform(feats_gy_target)
    #plt.scatter(src_tsne[:,0], src_tsne[:,1], c='red')
    #plt.scatter(trg_tsne[:,0],trg_tsne[:,1], c='blue')
    #plt.savefig(f"tsne_{args.dataset.shift}.eps", format="eps")


def validate(model, val_loader, device, it, num_classes):
    """
    function to validate the model on the test set
    model: Task containing the model to be tested
    val_loader: dataloader containing the validation data
    device: device on which you want to test
    it: int, iteration among the training num_iter at which the model is tested
    num_classes: int, number of classes in the classification problem
    """
    global modalities

    model.reset_acc()
    model.train(False)
    logits = {}

    # Iterate over the models
    with torch.no_grad():
        for i_val, (data, label) in enumerate(val_loader):
            label = label.to(device)

            for m in modalities:
                batch = data[m].shape[0]
                logits[m] = torch.zeros((batch, num_classes)).to(device)

            
            for m in modalities:
                data[m] = data[m].to(device)

            output, _ = model(data, is_train=False)
            for m in modalities:
                logits[m] = output[m]

            model.compute_accuracy(logits['RGB'], label)

            if (i_val + 1) % (len(val_loader) // 5) == 0:
                logger.info("[{}/{}] top1= {:.3f}% top5 = {:.3f}%".format(i_val + 1, len(val_loader),
                                                                          model.accuracy.avg[1], model.accuracy.avg[5]))

        class_accuracies = [(x / y) * 100 for x, y in zip(model.accuracy.correct, model.accuracy.total)]
        logger.info('Final accuracy: top1 = %.2f%%\ttop5 = %.2f%%' % (model.accuracy.avg[1],
                                                                      model.accuracy.avg[5]))
        for i_class, class_acc in enumerate(class_accuracies):
            logger.info('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                         int(model.accuracy.correct[i_class]),
                                                         int(model.accuracy.total[i_class]),
                                                         class_acc))

    logger.info('Accuracy by averaging class accuracies (same weight for each class): {}%'
                .format(np.array(class_accuracies).mean(axis=0)))
    test_results = {'top1': model.accuracy.avg[1], 'top5': model.accuracy.avg[5],
                    'class_accuracies': np.array(class_accuracies)}

    with open(os.path.join(args.log_dir, f'val_precision_{args.dataset.shift.split("-")[0]}-'
                                         f'{args.dataset.shift.split("-")[-1]}.txt'), 'a+') as f:
        f.write("[%d/%d]\tAcc@top1: %.2f%%\n" % (it, args.train.num_iter, test_results['top1']))

    return test_results


if __name__ == '__main__':
    main()
