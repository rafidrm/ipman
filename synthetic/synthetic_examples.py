import argparse
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


from options import load_params, load_opt_params
import plot_utils as plot
from gan_utils import *


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


def generate_epsilon_hist(mdl, epsilons, save_fig=False):
    if save_fig:
        save_filename = 'id_{}_error_hist.png'.format(
            mdl.model['id'], mdl.epoch)
        save_path = os.path.join(mdl.model['plots_dir'], save_filename)
    if save_fig:
        fig, ax = plot.plot_hist(epsilons, alpha=0.7, save_fig=save_path)
    else:
        fig, ax = plot.plot_hist(epsilons, alpha=0.7)


def generate_2d_samples(mdl, n_samples=1000, samples=None, show_real=True, save_fig=False):
    ''' plot 2-D samples'''
    if save_fig:
        save_filename = 'id_{}_loss_{}_epoch_{}.png'.format(
            mdl.model['id'], mdl.model['loss'], mdl.epoch)
        save_path = os.path.join(mdl.model['plots_dir'], save_filename)
        print('saving to: {}'.format(save_path))

    if samples is None:
        gen_input = Variable(
            mdl.g_sampler(n_samples, mdl.model['g_input_size']))
        g_output = mdl.G(gen_input).detach().numpy().T
    else:
        g_output = samples

    kwargs = {'alpha': 0.3, 'numticks': 5}
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


def optimizer(mdl, compare='None'):
    # first freeze gradients of the discriminator
    mdl.D = freeze_grads(mdl.D)
    model = mdl.model
    dual = model['dual_init']
    set_dual = False

    if model['loss'] == 'linear':
        linear_loss = torch.Tensor([1, 0])
        optimizer_loss_func = lambda x: mdl.optimizer_loss(x, linear_loss)
        err_func = lambda x: (x * linear_loss).tolist()[0]
        decision_err_func = lambda x: np.abs(x[0] + 1)

    elif model['loss'] == 'bilinear':
        quad_loss = torch.Tensor([[0, 0.5], [0.5, 0]])
        lin_loss = torch.Tensor([[-4, -4]])
        optimizer_loss_func = lambda x: mdl.optimizer_loss(
            x, quad_loss, lin_loss)
        err_func = lambda x: (x[0] * x[1]) - 4 * (x[0] + x[1])
        decision_err_func = lambda x: min(
            np.linalg.norm(x - np.array([-1, 17])),
            np.linalg.norm(x - np.array([17, -1])))

    elif model['loss'] == 'quadratic':
        quad_loss = torch.Tensor([[1, 0], [0, 1]])
        lin_loss = torch.Tensor([[-10, -22]])
        optimizer_loss_func = lambda x: mdl.optimizer_loss(
            x, quad_loss, lin_loss)
        err_func = lambda x: (x[0] - 5) ** 2 + (x[1] - 11) ** 2
        decision_err_func = lambda x: np.linalg.norm(x - np.array([5, 11]))

    elif model['loss'] == 'rosenbrock':
        optimizer_loss_func = lambda x: rosenbrock(x, 3.5)
        err_func = lambda x: (3.5 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        decision_err_func = lambda x: np.linalg.norm(
            x - np.array([3.5, 12.25]))

    else:
        raise NotImplementedError('dont use this loss.')

    error_rate = []

    if compare == 'unconstrained':
        barrier_func = lambda x: torch.zeros(1)
    else:
        barrier_func = lambda x: -1 * (x + 1e-20).log()

    for epoch in range(model['o_num_epochs']):
        for _ in range(model['g_steps']):
            mdl.G.zero_grad()

            g_error = 0
            for __ in range(model['minibatch_size']):
                gen_input = Variable(
                    mdl.g_sampler(1, model['g_input_size']))
                g_fake_data = mdl.G(gen_input)
                g_fake_decision = mdl.D(mdl.preprocess(g_fake_data))
                g_feas_error = barrier_func(g_fake_decision)

                g_opt_error = optimizer_loss_func(g_fake_data)

                g_error += g_feas_error + dual * g_opt_error
            g_error.backward()

            mdl.g_optimizer.step()

        if epoch % model['print_interval'] == 0:
            mdl.epoch = epoch
            dual = dual * model['dual_decay']
            opt = mdl.generate_optimal_sol()
            print('epoch: {}, feas: {} + dual: {} *  opt: {}'.format(
                epoch,
                round(extract(g_feas_error)[0], 3),
                dual,
                round(extract(g_opt_error)[0], 3)))
            print('         Optimal sol: {}'.format(opt[0]))

        if epoch % model['plot_interval'] == 0:
            generate_2d_samples(mdl, show_real=True, save_fig=True)

            # if epoch > 14000:
            if epoch > 40000 and set_dual == False:
                model['dual_decay'] = 1.005
                set_dual = True

        if epoch % 2500 == 0:
            if compare == 'store' or compare == 'unconstrained':
                dist_sol = generate_samples_and_store(mdl, err_func)
                error_rate.append((epoch, dist_sol))

    opt_sol = list(mdl.generate_optimal_sol(1000))

    scores = []
    for sol in opt_sol:
        err = err_func(sol)
        decision_err = decision_err_func(sol)

        diff_err = np.abs(err - model['optval'])
        scores.append((diff_err, sol, decision_err))

    scores = sorted(scores, key=lambda x: x[0])
    scores = scores[0:round(1000 * model['cvar_alpha'])]
    all_errs = np.array([score[0] for score in scores]).T
    sols = np.array([score[1] for score in scores]).T
    decision_err = np.array([score[2] for score in scores]).T
    generate_2d_samples(mdl, samples=sols, show_real=True, save_fig=True)

    # calculate the relative loss
    mean_ds_errs = np.mean(decision_err)

    print('\n\nerror stats: [min | max | mean | decision]:')
    print('[ {}\t{}\t{}\t{} ]\n\n'.format(
        np.min(all_errs), np.mean(all_errs), np.max(all_errs), mean_ds_errs))

    if compare == 'store' or compare == 'unconstrained':
        return error_rate


def generate_samples_and_store(mdl, err_func):
    opt_sol = list(mdl.generate_optimal_sol(1000))
    scores = []

    for sol in opt_sol:
        err = err_func(sol)
        err = np.abs(err - mdl.model['optval'])
        scores.append((err, sol))
    scores = sorted(scores, key=lambda x: x[0])
    scores = scores[0:round(1000 * mdl.model['cvar_alpha'])]
    sols = np.array([score[1] for score in scores]).T
    return sols


def generate_samples_and_evaluate(mdl, err_func):
    opt_sol = list(mdl.generate_optimal_sol(1000))
    total_errors = []

    for sol in opt_sol:
        err = err_func(sol)
        err = np.abs(err - mdl.model['optval'])
        total_errors.append(err)

    mean_err = np.mean(total_errors)
    return mean_err


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
                d_error = 0
                for __ in range(model['minibatch_size']):
                    d_real_data = Variable(mdl.d_sampler(1))
                    d_real_decision = mdl.D(mdl.preprocess(d_real_data).t())
                    d_real_error = mdl.criterion(
                        d_real_decision, Variable(torch.ones(1)))
                    # update yet

                    if np.random.rand() > 0.3:
                        d_gen_input = mdl.g_sampler(1, model['g_input_size'])
                        # detach to avoid training G
                        d_fake_data = mdl.G(d_gen_input).detach()
                        d_fake_decision = mdl.D(mdl.preprocess(d_fake_data))
                    else:
                        d_gen_input = Variable(mdl.adversarial_d_sampler(1))
                        d_fake_decision = mdl.D(
                            mdl.preprocess(d_gen_input).t())

                    d_fake_error = mdl.criterion(
                        d_fake_decision, Variable(torch.zeros(1)))
                    d_error += d_real_error + d_fake_error
                d_error.backward()

                mdl.d_optimizer.step()

            for _ in range(model['g_steps']):
                mdl.G.zero_grad()

                g_total_error = 0
                for __ in range(model['minibatch_size']):
                    gen_input = Variable(
                        mdl.g_sampler(1, model['g_input_size']))
                    g_fake_data = mdl.G(gen_input)
                    g_fake_decision = mdl.D(mdl.preprocess(g_fake_data))
                    g_error = mdl.criterion(
                        g_fake_decision, Variable(torch.ones(1)))
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

        # final discriminator update
        for _ in range(50):
            mdl.D.zero_grad()
            d_error = 0
            for __ in range(model['minibatch_size']):
                d_real_data = Variable(mdl.d_sampler(1))
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

    elif stage == 'compare':
        mdl = GANModel(data, model, load_model=True)
        mdl.epoch = 'pre'
        res_real = optimizer(mdl, compare='store')

        mdl = GANModel(data, model, load_model=True)
        mdl.epoch = 'pre'
        res_fake = optimizer(mdl, compare='unconstrained')

        for real, fake in zip(res_real, res_fake):
            save_filename = 'comparison_id_{}_loss_{}_epoch_{}.png'.format(
                mdl.model['id'], mdl.model['loss'], real[0])
            save_path = os.path.join(mdl.model['plots_dir'], save_filename)

            kwargs = {'alpha': 0.3, 'numticks': 5, 'save_fig': save_path}

            fig, ax = plot.plot_2d_samples(real[1], **kwargs)

            kwargs['color'] = '#be3392'
            kwargs['zorder'] = -5
            fig, ax = plot.plot_2d_samples(fake[1], fig, ax, **kwargs)

            gen_dist = mdl.d_sampler(1000).numpy()
            kwargs['color'] = '#ed7d31'
            kwargs['alpha'] = 0.1
            kwargs['zorder'] = -10
            kwargs['mode'] = mdl.data['mode']
            plot.plot_2d_samples(gen_dist, fig, ax, **kwargs)

    else:
        raise NotImplementedError('dont recognize stage {}'.format(stage))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='generate synthetic examples.')
    parser.add_argument('--stage', type=str, default='predict',
                        help='stage of the synthetic example generator: [predict | plot | optimize | compare]')
    parser.add_argument('--opt', type=str, default='linear',
                        help='which optimization problem: [linear | bilinear | quadratic | rosenbrock ]')
    args = parser.parse_args()
    model_params = load_opt_params(args.opt)

    # linear_hard_minx_model_params = {'p_num_epochs': 10000, 'tol': 0.01,
    #                                  'g_input_size': 2,
    #                                  'g_output_size': 2,
    #                                  'd_input_size': 2,
    #                                  'minibatch_size': 50,
    #                                  'loss': 'linear',
    #                                  'o_num_epochs': 70100,
    #                                  'dual_decay': 1.01,
    #                                  'dual_init': 0.05,
    #                                  'plot_interval': 5000,
    #                                  'stronger_d': True,
    #                                  'id': 'noisy_box2',
    #                                  'optval': 8,
    #                                  'cvar_alpha': 0.90,
    #                                  'notes': 'minimizing hard linear with clipping'}

    print('*********************\n data parameters:\n*********************')
    pprint(data_params)
    print('*********************\n model parameters:\n*********************')
    pprint(model_params)
    # model_params['d_input_size'] = model_params['minibatch_size']
    box_experiment(data_params={}, model_params=model_params, stage='optimize')
