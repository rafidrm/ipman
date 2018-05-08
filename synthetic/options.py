

DATA_PARAMS = {
    'mode': 'normal',
    'preprocess': True,
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
}


def load_params(data_par={}, model_par={}):
    data = DATA_PARAMS.copy()
    model = MODEL_PARAMS.copy()

    data.update(data_par)
    model.update(model_par)

    return data, model
