import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def rosenbrock(x, a):
    quad = torch.Tensor([[1.0, 0.0], [0.0, 0.0]])
    lin1 = torch.Tensor([[-2 * a, 0.0]])
    lin2 = torch.Tensor([[0.0, 1.0]])
    term1 = torch.mm(torch.mm(x, quad), x.t()) + \
        torch.mm(lin1, x.t()) + (a ** 2)
    term2 = -1 * torch.mm(torch.mm(x, quad), x.t()) + torch.mm(lin2, x.t())
    return (term1 + 100 * term2 * term2).mean()


def get_loss_func(mode='test'):
    if mode == 'test':
        return lambda n: (n * 2).mean()
    elif mode == 'linear':
        return lambda m, n: (m * n).mean()
    elif mode == 'test_quadratic':
        return lambda n: (torch.pow(n - 1, 2)).mean()
    elif mode == 'quadratic' or mode == 'bilinear':
        return lambda v, Q, r: (torch.mm(torch.mm(v, Q), v.t()) + torch.mm(r, v.t())).mean()
    elif mode == 'rosenbrock':
        return rosenbrock
    else:
        raise NotImplementedError('distribution not recognized')


def bimodal(n):
    ''' bimodal distribution with mu,sigma = (4,1.25) and (-4, 1.25)
    '''
    cat = np.random.randint(2, size=(1, n))
    norm1 = np.random.normal(4, 0.75, (1, n))
    norm2 = np.random.normal(-4, 0.75, (1, n))
    return torch.Tensor(norm1 * cat + norm2 * (1 - cat))


def ring_dist(n):
    ''' shaped like a ring. '''
    x1 = np.random.normal(0, 1.25, size=(1, n))
    y1 = np.random.normal(0, 1.25, size=(1, n))
    points = np.concatenate([x1, y1])
    normalized = np.array([point / np.linalg.norm(point)
                           for point in points.T]).T
    points = points + 10 * normalized

    return torch.Tensor(points)


def box_dist(n):
    ''' shaped like a box, i.e., 3 < |x|, |y| < 5. '''
    cat = np.random.randint(2, size=(1, n))
    x1 = np.random.uniform(low=0, high=15, size=(1, n))
    y1 = np.random.uniform(low=10, high=15, size=(1, n))
    xy1 = np.concatenate([x1, y1])

    x2 = np.random.uniform(low=10, high=15, size=(1, n))
    y2 = np.random.uniform(low=0, high=10, size=(1, n))
    xy2 = np.concatenate([x2, y2])

    return torch.Tensor(cat * xy1 + (1 - cat) * xy2)


def noisy_box_dist(n):
    ''' shaped like a box, i.e., 3 < |x|, |y| < 5. '''
    cat = np.random.randint(2, size=(1, n))
    x1 = np.random.uniform(low=0, high=15, size=(1, n))
    y1 = np.random.uniform(low=10, high=15, size=(1, n))
    xy1 = np.concatenate([x1, y1])

    x2 = np.random.uniform(low=10, high=15, size=(1, n))
    y2 = np.random.uniform(low=0, high=10, size=(1, n))
    xy2 = np.concatenate([x2, y2])

    noise = np.random.normal(0, 0.4, size=(2, n))
    return torch.Tensor(cat * xy1 + (1 - cat) * xy2 + noise)


def adv_noisy_box_dist(n):
    x1 = np.random.uniform(low=-30, high=30, size=(1, n))
    y1 = np.random.uniform(low=-30, high=30, size=(1, n))
    xy1 = np.concatenate([x1, y1])

    xy2 = np.zeros((2, n))
    cat = 0 if np.abs(np.linalg.norm(xy1) - 10) <= 2 else 1
    return torch.Tensor(cat * xy1 + (1 - cat) * xy2)


def get_distribution_sampler(mode='normal'):
    ''' where we define the feasible set '''
    if mode == 'normal':
        return lambda n: torch.Tensor(np.random.normal(5, 1, (1, n)))
    elif mode == 'bimodal':
        return bimodal
    elif mode == 'box_dist':
        return box_dist
    elif mode == 'noisy_box_dist':
        return noisy_box_dist
    elif mode == 'ring_dist':
        return ring_dist
    else:
        raise NotImplementedError('distribution not recognized')


def get_generator_input_sampler(mode='normal'):
    # Uniform distribution (not Gaussian?)
                    # d_fake_error.backward()
    if mode == 'normal':
        return lambda m, n: torch.randn(m, n)
    elif mode == 'bimodal':
        return lambda m, n: torch.randn(m, n)
    elif mode == 'box_dist':
        return lambda m, n: torch.randn(m, n)
        # return lambda m, n: torch.randn(m, n, 2)
    elif mode == 'noisy_box_dist':
        return lambda m, n: torch.randn(m, n)
    elif mode == 'ring_dist':
        return lambda m, n: torch.randn(m, n)
    else:
        raise NotImplementedError('sampler not recognized')


def get_adversarial_distribution(mode='normal'):
    if mode == 'noisy_box_dist':
        return adv_noisy_box_dist
    else:
        raise NotImplementedError('adversary not recognized')


#
# Models                  #
#

class Generator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        model = [
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        model = [
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class GANModel():

    def __init__(self, data, model, load_model=False):
        self.id = model['id']
        self.data = data
        self.model = model
        self.preprocess = data['preprocess_func']
        self.d_sampler = get_distribution_sampler(data['mode'])
        self.g_sampler = get_generator_input_sampler(data['mode'])
        self.epoch = 0

        if model['stronger_d'] is True:
            self.adversarial_d_sampler = get_adversarial_distribution(
                data['mode'])

        if load_model is True:
            save_filename = 'ganG_{}_id_{}.pth'.format(
                data['mode'], model['id'])
            save_path = os.path.join(model['save_dir'], save_filename)
            self.G = torch.load(save_path)

            save_filename = 'ganD_{}_id_{}.pth'.format(
                data['mode'], model['id'])
            save_path = os.path.join(model['save_dir'], save_filename)
            self.D = torch.load(save_path)

        else:
            self.G = Generator(
                input_size=model['g_input_size'],
                    # d_fake_error.backward()
                hidden_size=model['g_hidden_size'],
                output_size=model['g_output_size'])
            self.D = Discriminator(
                input_size=data['d_input_func'](model['d_input_size']),
                hidden_size=model['d_hidden_size'],
                output_size=model['d_output_size'])

        self.criterion = nn.BCELoss()
        self.d_optimizer = optim.Adam(
            self.D.parameters(),
            lr=model['d_learning_rate'],
            betas=model['optim_betas'])
        self.g_optimizer = optim.Adam(
            self.G.parameters(),
            lr=model['g_learning_rate'],
            betas=model['optim_betas'])
        self.optimizer_loss = get_loss_func(model['loss'])

    def generate_optimal_sol(self, n_samples=1):
        opt_input = Variable(self.g_sampler(
            n_samples, self.model['g_input_size']))
        opt_sol = self.G(opt_input).detach().numpy()
        return opt_sol


def extract(v):
    return v.data.storage().tolist()


def stats(v):
    return np.round(np.array([np.mean(v), np.std(v)]), 3)


def decorate_with_diffs(x, exponent):
    mean = torch.mean(x.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(x.size()), mean.tolist()[0][0])
    diffs = torch.pow(x - Variable(mean_broadcast), exponent)
    return torch.cat([x, diffs], 1)


def freeze_grads(mdl):
    for param in mdl.parameters():
        param.requires_grad = False
    return mdl


def apply_preprocess(data):
    if data['preprocess'] is True:
        data['preprocess_func'] = lambda x: decorate_with_diffs(x, 2.0)
        data['d_input_func'] = lambda x: x * 2
    else:
        data['preprocess_func'] = lambda x: x
        data['d_input_func'] = lambda x: x
    return data
