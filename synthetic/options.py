

DATA_PARAMS = {
    'mode': 'noisy_box_dist',
    'preprocess': False,
}


MODEL_PARAMS = {
    'g_input_size': 1,  # dimension of noise
    'g_hidden_size': 50,  # generator complexity
    'g_output_size': 1,  # size of generator output
    'd_input_size': 100,  # minibatch size
    'd_hidden_size': 50,  # discriminator complexity
    'd_output_size': 1,  # single dimension for real or fake
    'minibatch_size': 100,
    'd_learning_rate': 2e-4,
    'g_learning_rate': 2e-4,
    'optim_betas': (0.9, 0.999),
    'p_num_epochs': 30000,
    'print_interval': 200,
    'd_steps': 10,
    'g_steps': 1,
    'convergence': 5,
    'tol': 0.005,
    'loss': 'test_quadratic',
    'o_num_epochs': 8000,
    'dual_decay': 1.008,
    'dual_init': 100000.0,
    'save_dir': 'checkpoints',
    'plots_dir': 'plots',
    'plot_interval': 1000,
    'stop': False,
    'stronger_d': False,
    'cvar_alpha': 0.95,
}


def load_params(data_par={}, model_par={}):
    data = DATA_PARAMS.copy()
    model = MODEL_PARAMS.copy()

    data.update(data_par)
    model.update(model_par)

    return data, model


def load_opt_params(opt):
    if opt == 'linear':
        params = {'p_num_epochs': 10000, 'tol': 0.01,
                  'g_input_size': 2,
                  'g_output_size': 2,
                  'd_input_size': 2,
                  'minibatch_size': 50,
                  'loss': 'linear',
                  'o_num_epochs': 2500,
                  'dual_decay': 1.01,
                  'dual_init': 0.05,
                  'plot_interval': 400,
                  'stronger_d': True,
                  'id': 'noisy_box2',
                  'optval': -1,
                  'cvar_alpha': 0.95,
                  'notes': 'minimizing linear with clipping'}

    elif opt == 'quadratic':
        params = {'p_num_epochs': 10000, 'tol': 0.01,
                  'g_input_size': 2,
                  'g_output_size': 2,
                  'd_input_size': 2,
                  'minibatch_size': 50,
                  'loss': 'quadratic',
                  'o_num_epochs': 10100,
                  'dual_decay': 1.01,
                  'dual_init': 0.05,
                  'plot_interval': 2000,
                  'stronger_d': True,
                  'id': 'noisy_box2',
                  'optval': 0,
                  'cvar_alpha': 0.90,
                  'notes': 'minimizing quadratic with clipping'}

    elif opt == 'bilinear':
        params = {'p_num_epochs': 10000, 'tol': 0.01,
                  'g_input_size': 2,
                  'g_output_size': 2,
                  'd_input_size': 2,
                  'minibatch_size': 50,
                  'loss': 'bilinear',
                  'o_num_epochs': 12100,
                  'dual_decay': 1.01,
                  'dual_init': 0.05,
                  'plot_interval': 1000,
                  'stronger_d': True,
                  'id': 'noisy_box2',
                  'optval': -81,
                  'cvar_alpha': 0.9,
                  'notes': 'minimizing bilinear with clipping'}

    elif opt == 'rosenbrock':
        params = {'p_num_epochs': 10000, 'tol': 0.01,
                  'g_input_size': 2,
                  'g_output_size': 2,
                  'd_input_size': 2,
                  'minibatch_size': 50,
                  'loss': 'rosenbrock',
                  'o_num_epochs': 89100,
                  'dual_decay': 1.2,
                  'dual_init': 0.0001,
                  'plot_interval': 5000,
                  'print_interval': 100,
                  'stronger_d': True,
                  'id': 'noisy_box2',
                  'optval': 0,
                  'cvar_alpha': 0.9,
                  'notes': 'minimizing quadratic with clipping'}

    else:
        raise NotImplementedError('did not recognize opt [{}]'.format(opt))

    return params
