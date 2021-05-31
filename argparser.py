import argparse
import tasks


def modify_command_options(opts):
    if opts.dataset == 'voc':
        opts.num_classes = 21
    if opts.dataset == 'ade':
        opts.num_classes = 150

    if not opts.visualize:
        opts.sample_num = 0

    if opts.where_to_sim == 'GPU_server':
        opts.net_pytorch = False

    if opts.method is not None:
        if opts.method == 'FT':
            pass
        if opts.method == 'LWF':
            opts.loss_kd = 100
        if opts.method == 'CIL':
            opts.loss_CIL == 1
        if opts.method == 'LWF-MC':
            opts.icarl = True
            opts.icarl_importance = 10
        if opts.method == 'ILT':
            opts.loss_kd = 100
            opts.loss_de = 100
        if opts.method == 'EWC':
            opts.regularizer = "ewc"
            opts.reg_importance = 1000
        if opts.method == 'RW':
            opts.regularizer = "rw"
            opts.reg_importance = 1000
        if opts.method == 'PI':
            opts.regularizer = "pi"
            opts.reg_importance = 1000
        if opts.method == 'MiB':
            opts.loss_kd = 10
            opts.unce = True
            opts.unkd = True
            opts.init_balanced = True
        if opts.method == 'SDR':
            # Note: for the best results these hyperparameters may need to be changed.
            # Typical ranges are:
            # loss_kd : 1 - 100
            # loss_de_prototypes : 1e-3 - 1e-1
            # lfc (same value is used for both attractive and repulsive) : 1e-3 - 1e-2
            # lfs : 1e-5 - 1e-3
            # A kick-start could be to use loss_kd 10, loss_de_prototypes 1e-2, lfc 1e-3 and lfs 1e-4

            opts.loss_kd = 100
            opts.unce = True
            opts.unkd = True
            opts.loss_featspars = 1e-3
            opts.lfs_normalization = 'max_maskedforclass'
            opts.lfs_shrinkingfn = 'exponential'
            opts.lfs_loss_fn_touse = 'ratio'
            opts.loss_de_prototypes = 0.01
            opts.loss_de_prototypes_sumafter = True
            opts.lfc_sep_clust = 1e-3
            opts.loss_fc = 1e-3

    opts.no_overlap = not opts.overlap
    opts.no_cross_val = not opts.cross_val

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    # NB: on CPU not feasible because of inplace_ABN functions.
    # on GPU_windows need to remove apex since not supported
    # on GPU_server code as it has been downloaded
    parser.add_argument('--where_to_sim', type=str, choices=['GPU_windows', 'GPU_server', 'CPU', 'CPU_windows'], default='GPU_server')
    parser.add_argument("--net_pytorch", action='store_false', default=True,
                        help='whether to use default resnet from pytorch or to use the network as in MiB (default: True)')
    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help='number of workers (default: 1)')

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'ade'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None), set by method modify_command_options()")

    # Method Options
    # BE CAREFUL USING THIS, THEY WILL OVERRIDE ALL THE OTHER PARAMETERS.
    # This argument serves to use default parameters for the methods defined in function: modify_command_options()
    parser.add_argument("--method", type=str, default=None,
                        choices=['FT', 'LWF', 'LWF-MC', 'ILT', 'EWC', 'RW', 'PI', 'MiB', 'CIL', 'SDR'],
                        help="The method you want to use. BE CAREFUL USING THIS, IT MAY OVERRIDE OTHER PARAMETERS.")

    # Train Options
    parser.add_argument("--epochs", type=int, default=30,
                        help="epoch number (default: 30)")
    parser.add_argument("--fix_bn", action='store_true', default=False,
                        help='fix batch normalization during training (default: False)')

    parser.add_argument("--batch_size", type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 513)")

    parser.add_argument("--lr", type=float, default=0.007,
                        help="learning rate (default: 0.007)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step'], help="lr schedule policy (default: poly)")
    parser.add_argument("--lr_decay_step", type=int, default=5000,
                        help="decay step for stepLR (default: 5000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")
    parser.add_argument("--bce", default=False, action='store_true',
                        help="Whether to use BCE or not (default: no)")

    # whether to consider clustering on feature spaces as loss
    parser.add_argument("--loss_fc", type=float, default=0.,  # Features Clustering
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable features clustering loss")
    parser.add_argument("--lfc_L2normalized", action='store_true', default=False,
                        help="enable features clustering loss L2 normalized (default False)")
    parser.add_argument("--lfc_nobgr", action='store_true', default=False,
                        help="enable features clustering loss without background (default False)")
    parser.add_argument("--lfc_orth_sep", action='store_true', default=False,
                        help="Orthogonal separation loss applied on the current prototypes only")
    parser.add_argument("--lfc_orth_maxonly", action='store_true', default=False,
                        help="Orthogonal separation loss, only the maximum value is considered")
    parser.add_argument("--lfc_sep_clust", type=float, default=0.,  # Separation of Clusters
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable separation between clusters loss")
    parser.add_argument("--lfc_sep_clust_ison_proto", action='store_true', default=False,
                        help="enable separation clustering loss on prototypes (default False)")
    # whether to consider Soft Nearest Neighbor Loss (SNNL) as loss at features space
    parser.add_argument("--loss_SNNL", type=float, default=0.,  # SNNL
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable SNNL at feature level")
    parser.add_argument("--loss_featspars", type=float, default=0.,  # features sparsification
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable features sparsification loss")
    parser.add_argument("--lfs_normalization", type=str, default='max_foreachfeature',
                        choices=['L1', 'L2', 'max_foreachfeature', 'max_maskedforclass', 'max_overall', 'softmax'],
                        help="The method you want to use to normalize lfs")
    parser.add_argument("--lfs_shrinkingfn", type=str, default='squared',
                        choices=['squared', 'power3', 'exponential'],
                        help="The method you want to use to shrink the lfs")
    parser.add_argument("--lfs_loss_fn_touse", type=str, default='ratio',
                        choices=['ratio', 'max_minus_ratio', 'lasso', 'entropy'],
                        help="The loss function you want to use for the lfs")
    parser.add_argument("--loss_bgruncertainty", type=float, default=0.,
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable background uncertainty loss")
    parser.add_argument("--lbu_inverse", action='store_true', default=False,
                        help="enable inverse on lbu loss")
    parser.add_argument("--lbu_mean", action='store_true', default=False,
                        help="enable lbu_mean on lbu loss")
    parser.add_argument("--loss_CIL", type=float, default=0.,
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable loss of CIL paper")
    parser.add_argument("--feat_dim", type=float, default=2048,
                        help="Dimensionality of the features space (default: 2048 as in Resnet-101)")



    # Validation Options
    parser.add_argument("--val_on_trainset", action='store_true', default=False,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--cross_val", action='store_true', default=False,
                        help="If validate on training or on validation (default: Train)")
    parser.add_argument("--crop_val", action='store_false', default=True,
                        help='do crop for validation (default: True)')

    # Logging Options
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=0,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument("--debug",  action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--visualize",  action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=15,
                        help="epoch interval for eval (default: 15)")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="epoch interval for saving model (default: 1)")

    # Model Options
    parser.add_argument("--backbone", type=str, default='resnet101',
                        choices=['resnet50', 'resnet101'], help='backbone for the body (def: resnet50)')
    parser.add_argument("--output_stride", type=int, default=16,
                        choices=[8, 16], help='stride for the backbone (def: 16)')
    parser.add_argument("--no_pretrained", action='store_true', default=False,
                        help='Wheather to use pretrained or not (def: True)')
    parser.add_argument("--norm_act", type=str, default="iabn_sync",
                        choices=['iabn_sync', 'iabn', 'abn', 'std'], help='Which BN to use (def: abn_sync')
    parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"], default="mean",
                        help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
    parser.add_argument("--pooling", type=int, default=32,
                        help='pooling in ASPP for the validation phase (def: 32)')

    # Test and Checkpoint options
    parser.add_argument("--test",  action='store_true', default=False,
                        help="Whether to train or test only (def: train and test)")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")

    # Parameters for Knowledge Distillation of ILTSS (https://arxiv.org/abs/1907.13372)
    parser.add_argument("--freeze", action='store_true', default=False,
                        help="Use this to freeze the feature extractor in incremental steps")
    parser.add_argument("--loss_de", type=float, default=0.,  # Distillation on Encoder
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable distillation on Encoder (L2)")
    parser.add_argument("--loss_de_maskedold", default=False, action='store_true',
                        help="If enabled, loss_de is masked to consider only old classes features (default: False)")
    parser.add_argument("--loss_de_prototypes", type=float, default=0.,  # Distillation on Encoder
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable loss_de with prototypes (idea 1b)")
    parser.add_argument("--loss_de_prototypes_sumafter", action='store_true', default=False,
                        help="Whether to sum after of average during loss_DE")
    parser.add_argument("--loss_de_cosine", action='store_true', default=False,
                        help="Use cosine similarity ad distillation function on the encoded features")
    parser.add_argument("--loss_kd", type=float, default=0.,  # Distillation on Output
                        help="Set this hyperparameter to a value greater than "
                             "0 to enable Knowledge Distillation (Soft-CrossEntropy)")

    # Parameters for EWC, RW, and SI (from Riemannian Walks https://arxiv.org/abs/1801.10112)
    parser.add_argument("--regularizer", default=None, type=str, choices=['ewc', 'rw', 'pi'],
                        help="regularizer you want to use. Default is None")
    parser.add_argument("--reg_importance", type=float, default=1.,
                        help="set this par to a value greater than 0 to enable regularization")
    parser.add_argument("--reg_alpha", type=float, default=0.9,
                        help="Hyperparameter for RW and EWC that controls the update of Fisher Matrix")
    parser.add_argument("--reg_no_normalize", action='store_true', default=False,
                        help="If EWC, RW, PI must be normalized or not")
    parser.add_argument("--reg_iterations", type=int, default=10,
                        help="If RW, the number of iterations after each the update of the score is done")

    # Arguments for ICaRL (from https://arxiv.org/abs/1611.07725)
    parser.add_argument("--icarl", default=False, action='store_true',
                        help="If enable ICaRL or not (def is not)")
    parser.add_argument("--icarl_importance",  type=float, default=1.,
                        help="the regularization importance in ICaRL (def is 1.)")
    parser.add_argument("--icarl_disjoint", action='store_true', default=False,
                        help="Which version of icarl is to use (def: combined)")
    parser.add_argument("--icarl_bkg", action='store_true', default=False,
                        help="If use background from GT (def: No)")

    # METHODS
    parser.add_argument("--init_balanced", default=False, action='store_true',
                        help="Enable Background-based initialization for new classes")
    parser.add_argument("--unkd", default=False, action='store_true',
                        help="Enable Unbiased Knowledge Distillation instead of Knowledge Distillation")
    parser.add_argument("--alpha", default=1., type=float,
                        help="The parameter to hard-ify the soft-labels. Def is 1.")
    parser.add_argument("--unce", default=False, action='store_true',
                        help="Enable Unbiased Cross Entropy instead of CrossEntropy")

    # Incremental parameters
    parser.add_argument("--task", type=str, default="19-1", choices=tasks.get_task_list(),
                        help="Task to be executed (default: 19-1)")
    parser.add_argument("--step", type=int, default=0,
                        help="The incremental step in execution (default: 0)")
    # Consider the dataset as done in
    # http://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Michieli_Incremental_Learning_Techniques_for_Semantic_Segmentation_ICCVW_2019_paper.pdf
    # and https://arxiv.org/pdf/1911.03462.pdf : same as disjoint scenario (default) but with label of old classes in
    # new images, if present.
    parser.add_argument("--no_mask", action='store_true', default=False,
                        help="Use this to not mask the old classes in new training set, i.e. use labels of old classes"
                             " in new training set (if present)")
    parser.add_argument("--overlap", action='store_true', default=False,
                        help="Use this to not use the new classes in the old training set")
    parser.add_argument("--step_ckpt", default=None, type=str,
                        help="path to trained model at previous step. Leave it None if you want to use def path")
    parser.add_argument('--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O0')


    return parser
