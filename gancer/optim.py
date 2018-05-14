import os
import time
from options.optim_options import OptimOptions 
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import pudb

opt = OptimOptions().parse()
# opt.nThreads = 1  # test code only supports 1 thread
# opt.batchSize = 1  # test code only supports 1 batch
# opt.serial_batches = True
opt.no_flip = True
opt.niter = opt.ipm_niter
opt.niter_decay = opt.dual_decay_niter
opt.epoch_count = 1
opt.augment_train = False


data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
# print('#training images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(
    opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = {}, Phase = {}, Epoch = {}'.format(
    opt.name, opt.phase, opt.which_epoch))

total_steps = 0



for epoch in range(1, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_ipman()
        
        if total_steps % opt.display_freq == 0:
            visuals = model.get_current_visuals()
            # del visuals['real_B']
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(visuals, epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_ipman_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t,
                                            t_data)
            if opt.display_id > 0:
                visualizer.plot_current_errors(
                    epoch,
                    float(epoch_iter) / dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch {}, total_steps {})'.format(
                epoch, total_steps))
            model.save('optim_latest')

        iter_data_time = time.time()
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch {}, iters {}'.format(
            epoch, total_steps))
        model.save('optim_latest')
        model.save('optim_{}'.format(epoch))

    print('End of epoch {} / {} \t Time taken {} sec'.format(
        epoch, opt.niter + opt.niter_decay,
        time.time() - epoch_start_time))
    
    if epoch % opt.dual_decay_niter == 0:
        samples = model.generate_samples(opt.nsamples)
        for label, im in samples.items():
            visualizer.save_to_mat(webpage, epoch, label, im)
        model.update_lambda_dual()
    model.update_learning_rate()

