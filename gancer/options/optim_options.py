from .base_options import BaseOptions


class OptimOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
                '--objective',
                type=str,
                default='elementwise',
                help='which objective function to optimize over [ideal_l2 | ideal_l1 |elementwise]')
        self.parser.add_argument(
                '--lambda_dual',
                type=float,
                default=0.02,
                help='initial value for dual regularizer.')
        self.parser.add_argument(
                '--dual_decay',
                type=float,
                default=1.001,
                help='decay rate for dual regularizer')
        self.parser.add_argument(
                '--dual_decay_niter',
                type=int,
                default=4,
                help='freqency of decaying dual regularizer.')
        self.parser.add_argument(
                '--niter_decay',
                type=int,
                default=100,
                help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument(
                '--nsamples',
                type=int,
                default=10,
                help='number of samples to generate per update')
        self.parser.add_argument(
                '--display_freq',
                type=int,
                default=200,
                help='frequency of showing training results on screen')
        self.parser.add_argument(
                '--display_single_pane_ncols',
                type=int,
                default=0,
                help=
                'if positive, display all images in a single visdom web panel with certain number of images per row.'
                )
        self.parser.add_argument(
                '--update_html_freq',
                type=int,
                default=1000,
                help='frequency of saving training results to html')
        self.parser.add_argument(
                '--print_freq',
                type=int,
                default=200,
                help='frequency of showing training results on console')
        self.parser.add_argument(
                '--save_latest_freq',
                type=int,
                default=5000,
                help='frequency of saving the latest results')
        self.parser.add_argument(
                '--save_epoch_freq',
                type=int,
                default=5,
                help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument(
                '--ntest',
                type=int,
                default=float("inf"),
                help='# of test examples.')
        self.parser.add_argument(
                '--no_lsgan',
                action='store_true',
                help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument(
                '--lr_policy',
                type=str,
                default='lambda',
                help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument(
                '--results_dir',
                type=str,
                default='./results/',
                help='saves results here.')
        self.parser.add_argument(
                '--aspect_ratio',
                type=float,
                default=1.0,
                help='aspect ratio of result images')
        self.parser.add_argument(
                '--phase', type=str, default='test', help='train, test, val, etc.')
        self.parser.add_argument(
                '--which_epoch',
                type=str,
                default='latest',
                help=
                'which epoch to load? set to latest to use latest cached model.')
        self.isTrain = False
        self.parser.add_argument(
                '--ipm_niter',
                type=int,
                default=50,
                help='total number of ipm iterations.')
        self.parser.add_argument(
                '--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument(
                '--pool_size',
                type=int,
                default=50,
                help=
                'the size of image buffer that stores previously generated images')
        self.parser.add_argument(
                '--lr',
                type=float,
                default=0.0001,
                help='initial learning rate for adam')
        self.parser.add_argument(
                '--lr_decay_iters',
                type=int,
                default=50,
                help='multiply by a gamma every lr_decay_iters iterations')
    
        self.isTrain = False
        self.isOptim = True 
