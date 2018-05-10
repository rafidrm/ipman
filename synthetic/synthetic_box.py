import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import shapiro
import pudb
from pprint import pprint


from options import load_params
import plot_utils as plot


def get_loss_func(mode='test'):
    if mode == 'test':
        return lambda n: (n * 2).mean()
    elif mode == 'linear':
        return lambda m, n: (m * n).mean()
    elif mode == 'test_quadratic':
        return lambda n: (torch.pow(n - 1, 2)).mean()
    elif mode == 'quadratic':
        return lambda v, Q, r: (torch.mm(torch.mm(v, Q), v.t()) + torch.mm(r, v.t())).mean()
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


def generate_hist(mdl, n_samples=1000, show_real=False, save_fig=False):
    if save_fig:
        save_filename = 'id_{}_epoch_{}.png'.format(
            mdl.model['id'], mdl.epoch)
        save_path = os.path.join(mdl.model['plots_dir'], save_filename)

    gen_input = Variable(mdl.g_sampler(n_samples, mdl.model['g_input_size']))
    g_output = mdl.G(gen_input).detach().numpy()
    # gen_input = gen_input.numpy()
    if save_fig:
        fig, ax = plot.plot_hist(g_output, alpha=0.7, save_fig=save_path)
    else:
        fig, ax = plot.plot_hist(g_output, alpha=0.7)

    if show_real:
        gen_dist = mdl.d_sampler(n_samples).numpy()[0]
        if save_fig:
            plot.plot_hist(gen_dist, fig, ax, color='#ed7d31',
                           alpha=0.7, save_fig=save_path)
        else:
            plot.plot_hist(gen_dist, fig, ax, color='#ed7d31', alpha=0.7)


def generate_2d_samples(mdl, n_samples=1000, show_real=True, save_fig=False):
    ''' plot 2-D samples'''
    if save_fig:
        save_filename = 'id_{}_loss_{}_epoch_{}.png'.format(
            mdl.model['id'], mdl.model['loss'], mdl.epoch)
        save_path = os.path.join(mdl.model['plots_dir'], save_filename)
        print('saving to: {}'.format(save_path))

    gen_input = Variable(mdl.g_sampler(n_samples, mdl.model['g_input_size']))
    g_output = mdl.G(gen_input).detach().numpy().T

    kwargs = {'alpha': 0.5, 'numticks': 5}
    if save_fig:
        kwargs['save_fig'] = save_path
    else:
        pass

    fig, ax = plot.plot_2d_samples(g_output, **kwargs)

    if show_real:
        gen_dist = mdl.d_sampler(n_samples).numpy()
        kwargs['color'] = '#ed7d31'
        kwargs['alpha'] = 0.1
        kwargs['zorder'] = -10
        kwargs['mode'] = mdl.data['mode']
        plot.plot_2d_samples(gen_dist, fig, ax, **kwargs)
        # fig2, ax2 = plot.plot_hist(gen_dist, color='#ed7d31', alpha=0.7)
        # if save_fig:
        #     plot.plot_2d_samples(
        # gen_dist, fig, ax, color='#ed7d31', alpha=0.3, save_fig=save_path)
        # else:
        # plot.plot_2d_samples(gen_dist, fig, ax, color='#ed7d31', alpha=0.3)


def optimizer(mdl):
    # first freeze gradients of the discriminator
    mdl.D = freeze_grads(mdl.D)
    model = mdl.model
    dual = model['dual_init']

    if model['loss'] == 'linear':
        linear_loss = torch.Tensor([-1, 0])
    elif model['loss'] == 'quadratic':
        quad_loss = torch.Tensor([[0, 1], [1, 0]])
        lin_loss = torch.Tensor([[-8, -8]])
        # quad_loss = torch.Tensor([[1, 0], [0, 1]])
        # lin_loss = torch.Tensor([[-10, -22]])
    else:
        raise NotImplementedError('dont use this loss.')

    for epoch in range(model['o_num_epochs']):
        for _ in range(model['g_steps']):
            mdl.G.zero_grad()

            g_error = 0
            for __ in range(model['minibatch_size']):
                gen_input = Variable(
                    mdl.g_sampler(1, model['g_input_size']))
                g_fake_data = mdl.G(gen_input)
                g_fake_decision = mdl.D(mdl.preprocess(g_fake_data))
                g_feas_error = -1 * (g_fake_decision + 1e-20).log()
                # g_feas_error = mdl.criterion(
                #    g_fake_decision, Variable(torch.ones(1)))

                # if linear uncomment this:
                # g_opt_error = mdl.optimizer_loss(g_fake_data, linear_loss)
                # if quadratic uncomment this:
                g_opt_error = mdl.optimizer_loss(
                    g_fake_data, quad_loss, lin_loss)

                g_error += g_feas_error + dual * g_opt_error
            g_error.backward()

            mdl.g_optimizer.step()

        if epoch % model['print_interval'] == 0:
            mdl.epoch = epoch
            dual = dual * model['dual_decay']
            opt = mdl.generate_optimal_sol()
            # pu.db
            print('epoch: {}, feas: {} + dual: {} *  opt: {}'.format(
                epoch,
                round(extract(g_feas_error)[0], 3),
                dual,
                round(extract(g_opt_error)[0], 3)))
            print('         Optimal sol: {}'.format(opt[0]))

        if epoch % model['plot_interval'] == 0:
            generate_2d_samples(mdl, show_real=True, save_fig=True)

            if epoch > 14000:
                model['dual_decay'] = 1.005

    opt_sol = mdl.generate_optimal_sol(10)
    print(opt_sol)


def debug_discriminator(mdl):
    ''' Some test code to let me see how well discriminator performs. '''
    mdl.D = freeze_grads(mdl.D)
    model = mdl.model
    d_real_data = Variable(mdl.d_sampler(model['d_input_size']))
    d_real_decision = mdl.D(mdl.preprocess(d_real_data))
    d_real_error = mdl.criterion(
        d_real_decision, Variable(torch.ones(1)))

    res = [(d_real_data[0, 0], d_real_error)]
    for i in range(20):
        d_test = d_real_data + i
        d_err_decision = mdl.D(mdl.preprocess(d_test))
        d_err_error = mdl.criterion(
            d_err_decision, Variable(torch.zeros(1)))
        res.append((d_test[0, 0], d_err_error))
    print(res)


def box_experiment(data_params={}, model_params={}, stage='predict'):
    data, model = load_params(data_params, model_params)
    data = apply_preprocess(data)
    if stage == 'predict':
        mdl = GANModel(data, model)

        convergence_counter = 0
        for epoch in range(model['p_num_epochs']):
            for _ in range(model['d_steps']):
                mdl.D.zero_grad()
                # pu.db
                d_error = 0
                for __ in range(model['minibatch_size']):
                    d_real_data = Variable(mdl.d_sampler(1))
                    # d_real_data = Variable(
                    #     mdl.d_sampler(model['minibatch_size']))
                    d_real_decision = mdl.D(mdl.preprocess(d_real_data).t())
                    d_real_error = mdl.criterion(
                        d_real_decision, Variable(torch.ones(1)))
                    # d_real_error.backward()  # compute gradients but don't
                    # update yet

                    if np.random.rand() > 0.3:
                        d_gen_input = mdl.g_sampler(1, model['g_input_size'])
                        # d_gen_input = Variable(
                        #     mdl.g_sampler(model['minibatch_size'], model['g_input_size']))
                        # detach to avoid training G
                        d_fake_data = mdl.G(d_gen_input).detach()
                        d_fake_decision = mdl.D(mdl.preprocess(d_fake_data))
                    else:
                        d_gen_input = Variable(mdl.adversarial_d_sampler(1))
                        d_fake_decision = mdl.D(
                            mdl.preprocess(d_gen_input).t())

                    d_fake_error = mdl.criterion(
                        d_fake_decision, Variable(torch.zeros(1)))
                    # d_fake_error.backward()
                    d_error += d_real_error + d_fake_error
                d_error.backward()

                mdl.d_optimizer.step()

            for _ in range(model['g_steps']):
                mdl.G.zero_grad()
                # pu.db

                g_total_error = 0
                for __ in range(model['minibatch_size']):
                    gen_input = Variable(
                        mdl.g_sampler(1, model['g_input_size']))
                    # gen_input = Variable(
                    # mdl.g_sampler(model['minibatch_size'],
                    # model['g_input_size']))
                    g_fake_data = mdl.G(gen_input)
                    g_fake_decision = mdl.D(mdl.preprocess(g_fake_data))
                    g_error = mdl.criterion(
                        g_fake_decision, Variable(torch.ones(1)))
                    # g_error.backward()
                    g_total_error += g_error
                g_total_error.backward()

                mdl.g_optimizer.step()

            if epoch % model['print_interval'] == 0:
                print('epoch: {}, D: {}/{}, G: {}'.format(
                    epoch,
                    extract(d_real_error)[0],
                    extract(d_fake_error)[0],
                    extract(g_error)[0]))
                real_dist = stats(extract(d_real_data))
                fake_dist = stats(extract(d_fake_data))
                # print('       Real: {}, Fake: {}'.format(real_dist,
                # fake_dist))

        # final discriminator update
        for _ in range(50):
            mdl.D.zero_grad()
            # pu.db
            d_error = 0
            for __ in range(model['minibatch_size']):
                d_real_data = Variable(mdl.d_sampler(1))
                # d_real_data = Variable(
                #     mdl.d_sampler(model['minibatch_size']))
                d_real_decision = mdl.D(mdl.preprocess(d_real_data).t())
                d_real_error = mdl.criterion(
                    d_real_decision, Variable(torch.ones(1)))

                if np.random.rand() > 0.3:
                    d_gen_input = mdl.g_sampler(1, model['g_input_size'])
                    # detach to avoid training G
                    d_fake_data = mdl.G(d_gen_input).detach()
                    d_fake_decision = mdl.D(mdl.preprocess(d_fake_data))

                else:
                    d_gen_input = Variable(mdl.adversarial_d_sampler(1))
                    d_fake_decision = mdl.D(
                        mdl.preprocess(d_gen_input).t())
                d_fake_decision = mdl.D(mdl.preprocess(d_fake_data))
                d_fake_error = mdl.criterion(
                    d_fake_decision, Variable(torch.zeros(1)))
                # d_fake_error.backward()
                d_error += d_real_error + d_fake_error
            d_error.backward()

            mdl.d_optimizer.step()

        save_filename = 'ganG_{}_id_{}.pth'.format(
            data['mode'], model['id'])
        save_path = os.path.join(model['save_dir'], save_filename)
        torch.save(mdl.G, save_path)

        save_filename = 'ganD_{}_id_{}.pth'.format(
            data['mode'], model['id'])
        save_path = os.path.join(model['save_dir'], save_filename)
        torch.save(mdl.D, save_path)
        generate_2d_samples(mdl, show_real=True)
        plot.show()

    elif stage == 'optimize':
        mdl = GANModel(data, model, load_model=True)
        # debug_discriminator(mdl)
        mdl.epoch = 'pre'
        generate_2d_samples(mdl, show_real=True, save_fig=True)
        # plot.show()

        optimizer(mdl)
        generate_2d_samples(mdl, show_real=True)
        # plot.show()

    elif stage == 'plot':
        mdl = GANModel(data, model, load_model=True)
        generate_2d_samples(mdl, show_real=True)
        plot.show()

    else:
        raise NotImplementedError('dont recognize stage {}'.format(stage))


if __name__ == "__main__":
    # noisy_box1: generator too good for discriminator
    # noisy_box2: looks a lot better
    data_params = {'preprocess': False, 'mode': 'noisy_box_dist'}
    model_params = {'p_num_epochs': 10000, 'tol': 0.01,
                    'g_input_size': 2,
                    'g_output_size': 2,
                    'd_input_size': 2,
                    'minibatch_size': 50,
                    'loss': 'quadratic',
                    'o_num_epochs': 16100,
                    'dual_decay': 1.01,  # 1.005,
                    'dual_init': 0.05,
                    'plot_interval': 1000,
                    'stronger_d': True,
                    'id': 'noisy_box2',
                    'notes': 'minimizes 2xy - 8x - 8y'}

    print('*********************\n data parameters:\n*********************')
    pprint(data_params)
    print('*********************\n model parameters:\n*********************')
    pprint(model_params)
    # model_params['d_input_size'] = model_params['minibatch_size']
    box_experiment(data_params=data_params,
                   model_params=model_params, stage='optimize')

    # IF this doesnt work add dropout.
