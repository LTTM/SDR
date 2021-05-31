import utils
import argparser

from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
from shutil import copy
import time

import torch
torch.backends.cudnn.benchmark = True

from utils.run_utils import *
from metrics import StreamSegMetrics

from segmentation_module import make_model, make_model_v2

from train import Trainer
import tasks


def main(opts):

    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}_{opts.name}/"

    device, rank, world_size, logger = define_distrib_training(opts, logdir_full)

    logger.print(f"Device: {device}")
    logger.print(f"Rank: {rank}, world size: {world_size}")

    # Set up random seed
    setup_random_seeds(opts)

    # Set up dataloader
    train_dst, val_dst, test_dst, n_classes = get_dataset(opts, rank=rank)
    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size,
                                   sampler=DistributedSampler(train_dst, num_replicas=world_size, rank=rank),
                                   num_workers=opts.num_workers, drop_last=True)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                 sampler=DistributedSampler(val_dst, num_replicas=world_size, rank=rank),
                                 num_workers=opts.num_workers)
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size if opts.crop_val else 1,
                                  sampler=DistributedSampler(test_dst, num_replicas=world_size, rank=rank),
                                  num_workers=opts.num_workers)
    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)},"
                f" Test set: {len(test_dst)}, n_classes {n_classes}")
    logger.info(f"Total batch size is {opts.batch_size * world_size}")

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    step_checkpoint = None
    if opts.net_pytorch:
        model = make_model_v2(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    else:
        model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))

    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")

    if opts.step == 0:  # if step 0, we don't need to instance the model_old
        model_old = None
    else:  # instance model_old
        if opts.net_pytorch:
            model_old = make_model_v2(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))
        else:
            model_old = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step - 1))

    if opts.fix_bn:
        model.fix_bn()

    logger.debug(model)

    # xxx Set up optimizer
    params = []
    if not opts.freeze:
        params.append({"params": filter(lambda p: p.requires_grad, model.body.parameters()),
                       'weight_decay': opts.weight_decay})

    params.append({"params": filter(lambda p: p.requires_grad, model.head.parameters()),
                   'weight_decay': opts.weight_decay})

    params.append({"params": filter(lambda p: p.requires_grad, model.cls.parameters()),
                   'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)

    if model_old is not None:
        if opts.where_to_sim == 'GPU_server':
            from apex.parallel import DistributedDataParallel
            from apex import amp
            [model, model_old], optimizer = amp.initialize([model.to(device), model_old.to(device)], optimizer,
                                                     opt_level=opts.opt_level)
            model_old = DistributedDataParallel(model_old)
        else:  # on MacOS and on Windows apex not supported
            model = model.to(device)
            model_old = model_old.to(device)
    else:
        if opts.where_to_sim == 'GPU_server':
            from apex.parallel import DistributedDataParallel
            from apex import amp
            model, optimizer = amp.initialize(model.to(device), optimizer, opt_level=opts.opt_level)
            # Put the model on GPU
            model = DistributedDataParallel(model, delay_allreduce=True)
        else:  # on MacOS and on Windows apex not supported
            model = model.to(device)

    # Load old model from old weights if step > 0!
    if opts.step > 0:
        # get model path
        if not opts.test:
            path = f"{logdir_full}/{task_name}_{opts.name}_{opts.step - 1}.pth"
        else:
            path = opts.step_ckpt

        if opts.step_ckpt is not None and opts.step == 1:
            path_FT = opts.step_ckpt
        else:
            path_FT = path.replace(opts.name, 'FT')

        if (not os.path.exists(path)):  # and opts.name != 'EWC' and opts.name != 'MiB' and opts.name != 'PI' and opts.name != 'RW':
            if opts.task == '19-1' or opts.task == '15-5' or opts.task == '10-10':
                logger.info(f"[!] WARNING: Checkpoint of old model is {path_FT} and it is NOT copied into {path}")
                path = path_FT
            else:
                try:
                    copy(path_FT, path)
                    logger.info(f"[!] WARNING: Checkpoint of old model is {path_FT} and IT IS copied into {path}")
                except:
                    path_FT = f"logs/{opts.task}/{opts.task}_FT/{opts.task}-{opts.dataset}_FT//{opts.task}-{opts.dataset}_FT_0.pth"
                    copy(path_FT, path)
                    logger.info(f"[!] WARNINGG: Checkpoint of old model is {path_FT} and IT IS copied into {path}")

        # generate model from path
        if os.path.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            if opts.net_pytorch:
                net_dict = model.state_dict()
                pretrained_dict = {k.replace('module.', ''): v for k, v in step_checkpoint['model_state'].items() if
                                   (k.replace('module.', '') in net_dict)}
                net_dict.update(pretrained_dict)
                model.load_state_dict(net_dict, strict=False)
                del net_dict
            else:
                model.load_state_dict(step_checkpoint['model_state'],
                                      strict=False)  # False because of incr. classifiers

            if opts.init_balanced:
                # implement the balanced initialization (new cls has weight of background and bias = bias_bkg - log(N+1)
                model.init_new_classifier(device)
            # Load state dict from the model state dict, that contains the old model parameters
            if opts.net_pytorch:
                net_dict_old = model_old.state_dict()
                pretrained_dict = {k.replace('module.', ''): v for k, v in step_checkpoint['model_state'].items() if
                                  (k.replace('module.', '') in net_dict_old)}  # and (
                # v.shape == net_dict[k.replace('module.', '')].shape)
                net_dict_old.update(pretrained_dict)
                model_old.load_state_dict(net_dict_old, strict=True)
                del net_dict_old
            else:
                model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters
            logger.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
        elif opts.debug:
            logger.info(f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?")
            exit()
        else:
            raise FileNotFoundError(path)
        # put the old model into distributed memory and freeze it
        for par in model_old.parameters():
            par.requires_grad = False
        model_old.eval()

    # Set up Trainer
    trainer_state = None
    # if not first step, then instance trainer from step_checkpoint
    if opts.step > 0 and step_checkpoint is not None:
        if 'trainer_state' in step_checkpoint:
            trainer_state = step_checkpoint['trainer_state']

    # instance trainer (model must have already the previous step weights)
    trainer = Trainer(model, model_old, device=device, opts=opts, trainer_state=trainer_state,
                      classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step), logdir=logdir_full)

    # Handle checkpoint for current model (model old will always be as previous step or None)
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location="cpu")

        if opts.net_pytorch:
            net_dict = model.state_dict()
            pretrained_dict = {k.replace('module.',''): v for k, v in checkpoint['model_state'].items() if
                               (k.replace('module.','') in net_dict) and (v.shape == net_dict[k.replace('module.','')].shape)}
            net_dict.update(pretrained_dict)
            model.load_state_dict(net_dict)
        else:
            model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint['best_score']
        logger.info("[!] Model restored from %s" % opts.ckpt)
        # if we want to resume training, resume trainer from checkpoint
        if 'trainer_state' in checkpoint:
            trainer.load_state_dict(checkpoint['trainer_state'])
        del checkpoint
    else:
        if opts.step == 0:
            logger.info("[!] Train from scratch")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    if rank == 0 and opts.sample_num > 0:
        if (not opts.where_to_sim == 'GPU_server') or opts.net_pytorch:
            sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=True)  # sample idxs for visualization
        else:
            sample_ids = np.random.choice(len(val_loader), opts.sample_num, replace=False)  # sample idxs for visualization
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    TRAIN = not opts.test
    val_metrics = StreamSegMetrics(n_classes, opts)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0,100, (1,1)))

    # load prototypes if needed
    logger.info(f"Prototypes initialization to zero vectors")
    prototypes = torch.zeros([sum(tasks.get_per_task_classes(opts.dataset, opts.task, opts.step)),
                              opts.feat_dim])
    prototypes.requires_grad = False
    count_features = torch.zeros([sum(tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))],
                                 dtype=torch.long)
    count_features.requires_grad = False
    if opts.step > 0 and (opts.loss_de_prototypes > 0 or opts.lfc_sep_clust > 0 or opts.loss_fc):
        logger.info(f"Prototypes loaded from previous checkpoint")

        prototypes_old = step_checkpoint["prototypes"]
        count_features_old = step_checkpoint["count_features"]

        prototypes[0:prototypes_old.shape[0],:] = prototypes_old
        count_features[0:count_features_old.shape[0]] = count_features_old
        del step_checkpoint

    logger.info(f"Current prototypes are {prototypes}")
    logger.info(f"Current count_features is {count_features}")
    prototypes = prototypes.to(device)
    count_features = count_features.to(device)

    # train/val here
    while cur_epoch < opts.epochs and TRAIN:
        # =====  Train  =====
        model.train()

        epoch_loss, prototypes, count_features = trainer.train(cur_epoch=cur_epoch, optim=optimizer, world_size=world_size,
                                   train_loader=train_loader, scheduler=scheduler, logger=logger,
                                   print_int=opts.print_interval, prototypes=prototypes, count_features=count_features)

        logger.info(f"End of Epoch {cur_epoch+1}/{opts.epochs}, Average Loss={epoch_loss[0]+epoch_loss[1]},"
                    f" Class Loss={epoch_loss[0]}, Reg Loss={epoch_loss[1]}")

        # =====  Log metrics on Tensorboard =====
        logger.add_scalar("E-Loss", epoch_loss[0]+epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-reg", epoch_loss[1], cur_epoch)
        logger.add_scalar("E-Loss-cls", epoch_loss[0], cur_epoch)

        # =====  Validation  =====
        if (cur_epoch + 1) % opts.val_interval == 0:
            logger.info("validate on val set...")
            model.eval()
            val_loss, val_score, ret_samples = trainer.validate(loader=val_loader, metrics=val_metrics, world_size=world_size,
                                                                ret_samples_ids=sample_ids, logger=logger)
            logger.print("Done validation on Val set")
            logger.info(f"End of Validation {cur_epoch+1}/{opts.epochs}, Validation Loss={val_loss[0]+val_loss[1]},"
                        f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
            logger.info(val_metrics.to_str(val_score))

            # =====  Save Best Model  =====
            if rank == 0:  # save best model at the last iteration
                score = val_score['Mean IoU']
                # best model to build incremental steps
                save_ckpt(f"{logdir_full}/{task_name}_{opts.name}_{opts.step}.pth",
                          model, trainer, optimizer, scheduler, cur_epoch, score, prototypes, count_features)
                logger.info("[!] Checkpoint saved.")

            # =====  Log metrics on Tensorboard =====
            # visualize validation score and samples
            logger.add_scalar("V-Loss", val_loss[0]+val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-reg", val_loss[1], cur_epoch)
            logger.add_scalar("V-Loss-cls", val_loss[0], cur_epoch)
            logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
            logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
            logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
            logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
            # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)

            # keep the metric to print them at the end of training
            results["V-IoU"] = val_score['Class IoU']
            results["V-Acc"] = val_score['Class Acc']

            for k, (img, target, lbl) in enumerate(ret_samples):
                img = (denorm(img) * 255).astype(np.uint8)
                target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
                lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

                concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

        cur_epoch += 1

    # =====  Save Best Model at the end of training =====
    if rank == 0 and TRAIN:  # save best model at the last iteration
        # best model to build incremental steps
        save_ckpt(f"{logdir_full}/{task_name}_{opts.name}_{opts.step}.pth",
                  model, trainer, optimizer, scheduler, cur_epoch, best_score, prototypes, count_features)
        logger.info("[!] Checkpoint saved.")

    if not (opts.where_to_sim == 'GPU_windows' or opts.where_to_sim == 'CPU_windows'):
        torch.distributed.barrier()

    # xxx From here starts the test code
    logger.info("*** Test the model on all seen classes...")

    # load best model
    if TRAIN:
        if opts.net_pytorch:
            model = make_model_v2(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
        else:
            model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))

        # Put the model on GPU
        if opts.where_to_sim == 'GPU_server':
            DistributedDataParallel(model.cuda(device))
        else:  # on MacOS and on Windows apex not supported
            model = model.to(device)

        ckpt = f"{logdir_full}/{task_name}_{opts.name}_{opts.step}.pth"
        checkpoint = torch.load(ckpt, map_location="cpu")

        if opts.net_pytorch:
            net_dict = model.state_dict()
            pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state'].items() if
                               (k.replace('module.', '') in net_dict) and (
                                           v.shape == net_dict[k.replace('module.', '')].shape)}
            net_dict.update(pretrained_dict)
            model.load_state_dict(net_dict)
        else:
            model.load_state_dict(checkpoint["model_state"])  # , strict=True)

        logger.info(f"*** Model restored from {ckpt}")
        del checkpoint
        trainer = Trainer(model, None, device=device, opts=opts, logdir=logdir_full)

    model.eval()

    if opts.test:
        val_loss, val_score, _ = trainer.validate(loader=test_loader, metrics=val_metrics, logger=logger,
                                                  world_size=world_size, vis_dir=logdir_full, label2color=label2color, denorm=denorm)
    else:
        val_loss, val_score, _ = trainer.validate(loader=test_loader, metrics=val_metrics, logger=logger, world_size=world_size)

    logger.print("Done test")
    logger.info(f"*** End of Test, Total Loss={val_loss[0]+val_loss[1]},"
                f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    logger.add_results(results)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)

    logger.close()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparser.get_argparser()

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}_{opts.name}/"

    os.makedirs(f"{logdir_full}", exist_ok=True)

    main(opts)
    print('TOTAL TIME: ', time.time() - start_time)
