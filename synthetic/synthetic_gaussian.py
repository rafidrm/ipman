import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.stats import shapiro
import pudb

from options import load_params
import plot_utils as plot


def get_loss_func(mode='test'):
    if mode == 'test':
        return lambda n: (n * 2).mean()
    if mode == 'test_quadratic':
        return lambda n: (torch.pow(n, 2)).mean()
    else:
        raise NotImplementedError('distribution not recognized')


def bimodal(n):
    ''' bimodal distribution with mu,sigma = (4,1.25) and (-4, 1.25)
    '''
    cat = np.random.randint(2, size=(1, n))
    norm1 = np.random.normal(4, 0.75, (1, n))
    norm2 = np.random.normal(-4, 0.75, (1, n))
    return torch.Tensor(norm1 * cat + norm2 * (1 - cat))


def get_distribution_sampler(mode='normal'):
    ''' where we define the feasible set '''
    if mode == 'normal':
        return lambda n: torch.Tensor(np.random.normal(5, 1, (1, n)))
    elif mode == 'bimodal':
        return bimodal
    elif mode == 'box':
        return lambda n: torch.Tensor(2 * np.random.rand(1, n) + 5 + np.random.normal(0, 0.4, (1, n)))
    else:
        raise NotImplementedError('distribution not recognized')


def get_generator_input_sampler(mode='normal'):
    # Uniform distribution (not Gaussian?)
    if mode == 'normal':
        return lambda m, n: torch.randn(m, n)
    elif mode == 'bimodal':
        return lambda m, n: torch.randn(m, n)
    elif mode == 'box':
        return lambda m, n: torch.rand(m, n)
    else:
        raise NotImplementedError('distribution not recognized')


###########################
# Models                  #
###########################
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        model = [
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(0.2),
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
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(0.2),
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


def optimizer(mdl):
    # first freeze gradients of the discriminator
    mdl.D = freeze_grads(mdl.D)
    model = mdl.model
    dual = model['dual_init']
    for epoch in range(model['o_num_epochs']):
        mdl.epoch = epoch
        for _ in range(model['g_steps']):
            mdl.G.zero_grad()
            gen_input = Variable(
                mdl.g_sampler(model['minibatch_size'], model['g_input_size']))
            g_fake_data = mdl.G(gen_input)
            g_fake_decision = mdl.D(mdl.preprocess(g_fake_data.t()))
            g_feas_error = mdl.criterion(
                g_fake_decision, Variable(torch.ones(1)))
            g_opt_error = mdl.optimizer_loss(g_fake_data)
            pu.db
            g_error = g_opt_error + dual * g_feas_error
            g_error.backward()

            mdl.g_optimizer.step()

        if epoch % model['print_interval'] == 0:
            dual = dual / model['dual_decay']
            opt = mdl.generate_optimal_sol()
            print('epoch: {}, feas: {} opt: {} dual: {}'.format(
                epoch,
                round(extract(g_feas_error)[0], 3),
                round(extract(g_opt_error)[0], 3),
                round(dual, 3)))
            print('         Optimal sol: {}'.format(opt[0][0]))

        if epoch % model['plot_interval'] == 0:
            generate_hist(mdl, save_fig=True)

    opt_sol = mdl.generate_optimal_sol(10)
    print(opt_sol)


def stopping_criteria(mdl):
    d_real_data = Variable(mdl.d_sampler(mdl.model['d_input_size']))
    d_test = d_real_data + 3
    d_err_decision = mdl.D(mdl.preprocess(d_test))
    d_err_error = mdl.criterion(
        d_err_decision, Variable(torch.ones(1)))
    err_above = np.round(d_err_error.tolist(), 6)

    d_test = d_real_data - 3
    d_err_decision = mdl.D(mdl.preprocess(d_test))
    d_err_error = mdl.criterion(
        d_err_decision, Variable(torch.ones(1)))
    err_below = np.round(d_err_error.tolist(), 6)
    print('       Above: {}, Below: {}'.format(err_above, err_below))
    if err_above > 10 ** (-2):
        return False
    if err_below > 10 ** (-2):
        return False
    return True


def debug_discriminator(mdl):
    ''' Some test code to let me see how well discriminator performs. '''
    # mdl.D = freeze_grads(mdl.D)
    model = mdl.model
    d_real_data = Variable(mdl.d_sampler(model['d_input_size']))
    d_real_decision = mdl.D(mdl.preprocess(d_real_data))
    d_real_error = mdl.criterion(
        d_real_decision, Variable(torch.ones(1)))

    res = [(d_real_data[0, 0], d_real_error)]
    for i in range(4):
        d_test = d_real_data + i * 2
        d_err_decision = mdl.D(mdl.preprocess(d_test))
        d_err_error = mdl.criterion(
            d_err_decision, Variable(torch.zeros(1)))
        res.append((round(d_test[0, 0].tolist(), 6),
                    round(d_err_error.tolist(), 6)))
    print(res)


def gaussian_experiment(data_params={}, model_params={}, stage='predict'):
    data, model = load_params(data_params, model_params)
    data = apply_preprocess(data)
    if stage == 'predict':
        mdl = GANModel(data, model)

        convergence_counter = 0
        for epoch in range(model['p_num_epochs']):
            for _ in range(model['d_steps']):
                mdl.D.zero_grad()

                d_real_data = Variable(mdl.d_sampler(model['d_input_size']))
                d_real_decision = mdl.D(mdl.preprocess(d_real_data))
                d_real_error = mdl.criterion(
                    d_real_decision, Variable(torch.ones(1)))
                d_real_error.backward()  # compute gradients but don't update yet

                d_gen_input = Variable(
                    mdl.g_sampler(model['minibatch_size'], model['g_input_size']))
                # detach to avoid training G
                d_fake_data = mdl.G(d_gen_input).detach()
                d_fake_decision = mdl.D(mdl.preprocess(d_fake_data.t()))
                d_fake_error = mdl.criterion(
                    d_fake_decision, Variable(torch.zeros(1)))
                d_fake_error.backward()

                mdl.d_optimizer.step()

            for _ in range(model['g_steps']):
                mdl.G.zero_grad()

                gen_input = Variable(
                    mdl.g_sampler(model['minibatch_size'], model['g_input_size']))
                g_fake_data = mdl.G(gen_input)
                g_fake_decision = mdl.D(mdl.preprocess(g_fake_data.t()))
                g_error = mdl.criterion(
                    g_fake_decision, Variable(torch.ones(1)))
                g_error.backward()

                mdl.g_optimizer.step()

            if epoch % model['print_interval'] == 0:
                print('epoch: {}, D: {}/{}, G: {}'.format(
                    epoch,
                    extract(d_real_error)[0],
                    extract(d_fake_error)[0],
                    extract(g_error)[0]))
                real_dist = stats(extract(d_real_data))
                fake_dist = stats(extract(d_fake_data))
                # print('       Real: {}, Fake: {}'.format(real_dist, fake_dist))

                if model['stop'] == True:
                    good_d = stopping_criteria(mdl)
                    good_g = (extract(g_error)[0] <= 1.1)
                    if good_d and good_g:
                        print('Stopping here.')
                        break

        # one last time train the discriminator a lot
        for _ in range(model['d_steps'] * 2):
            mdl.D.zero_grad()

            d_real_data = Variable(mdl.d_sampler(model['d_input_size']))
            d_real_decision = mdl.D(mdl.preprocess(d_real_data))
            d_real_error = mdl.criterion(
                d_real_decision, Variable(torch.ones(1)))
            d_real_error.backward()  # compute gradients but don't update yet

            d_gen_input = Variable(
                mdl.g_sampler(model['minibatch_size'], model['g_input_size']))
            # detach to avoid training G
            d_fake_data = mdl.G(d_gen_input).detach()
            d_fake_decision = mdl.D(mdl.preprocess(d_fake_data.t()))
            d_fake_error = mdl.criterion(
                d_fake_decision, Variable(torch.zeros(1)))
            d_fake_error.backward()

            mdl.d_optimizer.step()

        save_filename = 'ganG_{}_id_{}.pth'.format(
            data['mode'], model['id'])
        save_path = os.path.join(model['save_dir'], save_filename)
        torch.save(mdl.G, save_path)

        save_filename = 'ganD_{}_id_{}.pth'.format(
            data['mode'], model['id'])
        save_path = os.path.join(model['save_dir'], save_filename)
        torch.save(mdl.D, save_path)
        generate_hist(mdl, show_real=True)
        plot.show()

    elif stage == 'optimize':
        mdl = GANModel(data, model, load_model=True)
        mdl.epoch = 'pre'
        pu.db
        generate_hist(mdl, show_real=True, save_fig=True)
        optimizer(mdl)
        # generate_hist(mdl, show_real=False)
        # plot.show()


if __name__ == "__main__":
    # Gauss1: retrain with 10x D and its pretty good
    # 1dBox1: need to tweak dual decay rate. It doesn't punish enough.
    # 1dBox2: i tweaked the generator input to be box
    # 1dBox3: with stopping criteria
    data_params = {'preprocess': True, 'mode': 'box'}
    model_params = {'p_num_epochs': 30000, 'tol': 0.01,
                    'loss': 'test_quadratic',
                    'd_steps': 10,
                    'o_num_epochs': 8000,
                    'dual_decay': 1.2,
                    'dual_init': 1000.0,
                    'id': '1dBox1',
                    'stop': True}
    gaussian_experiment(data_params=data_params,
                        model_params=model_params, stage='optimize')
