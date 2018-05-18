import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pudb


class Vox2VoxModel(BaseModel):
    def name(self):
        return 'Vox2VoxModel'

    def initialize(self, opt):
        ''' Parses opts and initializes the relevant networks. '''
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.no_img = opt.no_img and (opt.output_nc == 1)

        # load and define networks according to opts
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain or self.isOptim:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain or self.isOptim:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # i can still use this object. no PIL involved
            self.fake_AB_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(
                    use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                    self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if self.isOptim:
            # define loss functions: optimality and feasibility
            self.multi_obj = False
            if opt.objective == 'ideal_l2':
                self.criterionOptim = torch.nn.MSELoss()
            elif opt.objective == 'ideal_l1':
                self.criterionOptim = torch.nn.L1Loss()
            elif opt.objective == 'elementwise':
                self.criterionOptim = lambda m, n: (m * n).mean()
            elif opt.objective == 'linear_l2':
                self.criterionOptim = lambda m, n: (m * n).mean()
            else:
                raise NotImplementedError('i do not recognize this objective [{}]'.format(opt.objective))
            self.criterionGAN = networks.GANLoss(
                    use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            if opt.objective == 'linear_l2':
                # self.criterionSmooth = lambda m, n: (m - n).norm()
                self.criterionSmooth = torch.nn.MSELoss()
                self.multi_obj = True
            else:
                self.criterionSmooth = lambda m: ((m + 1) / 2).norm()


            self.lambda_dual = self.opt.lambda_dual
            self.dual_decay = self.opt.dual_decay

            # freeze discriminator
            for param in self.netD.parameters():
                param.requires_grad = False

            # initialize optimizer
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                    self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))


        print('---------- Networks initialized ----------')
        networks.print_network(self.netG)
        if self.isTrain or self.isOptim:
            networks.print_network(self.netD)
        print('------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.is_feasible = input['is_feasible']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
        if self.isOptim:
            self.optimObj = input['obj']
            if len(self.gpu_ids) > 0:
                self.optimObj = self.optimObj.cuda(self.gpu_ids[0], async=True)

            if self.multi_obj:
                self.ideal = input['ideal']
                if len(self.gpu_ids) > 0:
                    self.ideal = self.ideal.cuda(self.gpu_ids[0], async=True)




    def generate_samples(self, nsamples):
        samples = []
        for ix in range(nsamples):
            self.forward()
            if self.no_img:
                fake_B = util.tensor2vid(self.fake_B.data, gray_to_rgb=False)
            else:
                fake_B = util.tensor2vid(self.fake_B.data)
            samples.append(fake_B)
        return OrderedDict(
                [('fake_{}'.format(ix), val) for ix, val in enumerate(samples)])

    def forward(self):
        # run this after setting inputs
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        # no backprop on gradients
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # first stop backprop to the generator by detaching fake_B
        # add real_A, fake_B to query. Concatenates on axis=1
        # TODO: Why do we detach?
        fake_AB = self.fake_AB_pool.query(
                torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        
        # added for augmenting training with infeasible points.
        if self.is_feasible.sum().item() == len(self.is_feasible):
            self.loss_D_real = self.criterionGAN(pred_real, True)
        elif self.is_feasible.sum().item() == 0:
            self.loss_D_real = self.criterionGAN(pred_real, False) * 5 
        elif self.is_feasible.sum().item() == len(self.is_feasible) * -1:
            self.loss_D_real = self.criterionGAN(pred_real, True)
        else:
            raise ValueError('something went wrong with training augment.')
        
        # combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # first G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)  # log trick?

        # second G(A) = B
        if self.is_feasible.sum().item() == len(self.is_feasible) :
            self.loss_G_L1 = self.criterionL1(self.fake_B,
                    self.real_B) * self.opt.lambda_A
        elif self.is_feasible.sum().item() == len(self.is_feasible) * -1 :
            self.loss_G_L1 = self.criterionL1(self.fake_B,
                    self.real_B) * self.opt.lambda_A
        elif self.is_feasible.sum().item() == 0:
            self.loss_G_L1 = self.criterionL1(self.fake_B,
                    self.real_B) * self.opt.lambda_A * 0.0
        else:
            raise ValueError('something went wrong with training augment.')
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

        self.loss_G.backward()

    def optimize_ipman(self):
        # pu.db
        noise_scale = 0.01 * torch.rand(1).tolist()[0] + 1.0
        self.real_A = Variable(self.input_A / noise_scale)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

        self.optimizer_G.zero_grad()
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        # TODO: I am using criterion GAN but might have to change it to a fixed
        # log function.
        # self.loss_feas = -1 * (pred_fake + 1e-6).log().mean()
        self.loss_feas = self.criterionGAN(pred_fake, True)

        self.loss_opt = self.criterionOptim(self.fake_B, self.optimObj) + 1 

        if self.multi_obj is True:
            self.loss_smooth = self.criterionSmooth(self.fake_B, self.ideal) 
        else:
            self.loss_smooth = self.criterionSmooth(self.fake_B) * 0.0001


        self.loss_G = self.loss_feas + self.loss_opt * self.lambda_dual + self.loss_smooth * self.lambda_dual
        self.loss_G.backward()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]), ('G_L1', self.loss_G_L1.data[0]), ('D_real', self.loss_D_real.data[0]), ('D_fake', self.loss_D_fake.data[0])])

    def get_current_ipman_errors(self):
        return OrderedDict([('G_Feas', self.loss_feas.data[0]), ('G_Opt', self.loss_opt.data[0]), ('G_smooth', self.loss_smooth.data[0])])

    def get_current_visuals(self):
        real_A = util.tensor2vid(self.real_A.data)
        if self.no_img:
            fake_B = util.tensor2vid(self.fake_B.data, gray_to_rgb=False)
            real_B = util.tensor2vid(self.real_B.data, gray_to_rgb=False)
        else:
            fake_B = util.tensor2vid(self.fake_B.data)
            real_B = util.tensor2vid(self.real_B.data)
        if self.isOptim:
            if self.no_img:
                obj = util.tensor2vid(self.optimObj.data, gray_to_rgb=False)
            else:
                obj = util.tensor2vid(self.optimObj.data)
                # obj = util.tensor2vid(self.ideal.data)
            return OrderedDict([('input', real_A), ('decision', fake_B), ('obj', obj), ('feas', real_B)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_lambda_dual(self, decay_weight=1):
        self.lambda_dual = self.lambda_dual * self.dual_decay * decay_weight
        print('lambda = %.7f' % self.lambda_dual)

