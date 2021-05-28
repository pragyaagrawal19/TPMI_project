import os
import json
import argparse
import numpy as np
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms, utils
from torch.autograd import Variable
from copy import deepcopy

from models.vae.parallelly_reparameterized_vae import ParallellyReparameterizedVAE
from models.vae.sequentially_reparameterized_vae import SequentiallyReparameterizedVAE
from models.student_teacher import StudentTeacher
from helpers.layers import EarlyStopping, init_weights
from datasets.loader import get_split_data_loaders, get_loader
from optimizers.adamnormgrad import AdamNormGrad
from helpers.grapher import Grapher
from helpers.fid import train_fid_model, FID
from helpers.metrics import calculate_consistency, calculate_fid, estimate_fisher, calculate_fid_from_generated_images_gpu
from helpers.utils import float_type, ones_like, \
    append_to_csv, num_samples_in_loader, check_or_create_dir, \
    dummy_context, number_of_parameters

parser = argparse.ArgumentParser(description='LifeLong VAE Pytorch')
model_path = '/data/pragya/VAE/train_model'
output_dir='/data/pragya/VAE/mnist_output_dir'
mnist_checkpoint_dir='/data/pragya/VAE/mnist_checkpoint2'
checkpoint_dir='/data/pragya/VAE/checkpoint'
save_path='/data/pragya/VAE/mnist_generated/'
batch_path='/data/pragya/VAE/mnist_batch_path2/'
# Task parameters
parser.add_argument('--uid', type=str, default="",
                    help="add a custom task-specific unique id; appended to name (default: None)")
parser.add_argument('--task', type=str, default="mnist",
                    help="""task to work on (can specify multiple) [mnist / cifar10 /
                    fashion / svhn_centered / svhn / clutter / permuted] (default: mnist)""")
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='minimum number of epochs to train (default: 10)')
parser.add_argument('--num_sample', type=int, default=10, metavar='N',
                    help='Num of samples for few shot learning (default: 10)')
parser.add_argument('--continuous-size', type=int, default=32, metavar='L',
                    help='latent size of continuous variable when using mixture or gaussian (default: 32)')
parser.add_argument('--discrete-size', type=int, default=1,
                    help='initial dim of discrete variable when using mixture or gumbel (default: 1)')
parser.add_argument('--download', type=int, default=1,
                    help='download dataset from s3 (default: 1)')
parser.add_argument('--data-dir', type=str, default='./.datasets', metavar='DD',
                    help='directory which contains input data')
parser.add_argument('--output-dir', type=str, default='/data/pragya/VAE/mnist_output_dir', metavar='OD',
                    help='directory which contains csv results')
parser.add_argument('--model-dir', type=str, default='/data/pragya/VAE/train_model', metavar='MD',
                    help='directory which contains trained models')
parser.add_argument('--fid-model-dir', type=str, default='/data/pragya/VAE/train_model',
                    help='directory which contains trained FID models')
parser.add_argument('--calculate-fid-with', type=str, default=None,
                    help='enables FID calc & uses model conv/inceptionv3  (default: None)')
parser.add_argument('--disable-augmentation', action='store_true',
                    help='disables student-teacher data augmentation')

# train / eval or resume modes
parser.add_argument('--resume-training-with', type=int, default=None,
                    help='tries to load the model from model_dir and resume training [use int] (default: None)')
parser.add_argument('--eval-with', type=int, default=None,
                    help='tries to load the model from model_dir and evaluate the test dataset [use int] (default: None)')
parser.add_argument('--eval-with-loader', type=int, default=None,
                    help='if there are many loaders use ONLY this loader [use int] (default: None)')

# Model parameters
parser.add_argument('--filter-depth', type=int, default=32,
                    help='number of initial conv filter maps (default: 32)')
parser.add_argument('--reparam-type', type=str, default='isotropic_gaussian',
                    help='isotropic_gaussian, discrete or mixture [default: isotropic_gaussian]')
parser.add_argument('--layer-type', type=str, default='conv',
                    help='dense or conv (default: conv)')
parser.add_argument('--nll-type', type=str, default='bernoulli',
                    help='bernoulli or gaussian (default: bernoulli)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--vae-type', type=str, default='parallel',
                    help='vae type [sequential or parallel] (default: parallel)')
parser.add_argument('--normalization', type=str, default='groupnorm',
                    help='normalization type: batchnorm/groupnorm/instancenorm/none (default: groupnorm)')
parser.add_argument('--activation', type=str, default='elu',
                    help='activation function (default: elu)')
parser.add_argument('--disable-sequential', action='store_true',
                    help='enables standard batch training')
parser.add_argument('--shuffle-minibatches', action='store_true',
                    help='shuffles the student\'s minibatch (default: False)')
parser.add_argument('--use-relational-encoder', action='store_true',
                    help='uses a relational network as the encoder projection layer')
parser.add_argument('--use-pixel-cnn-decoder', action='store_true',
                    help='uses a pixel CNN decoder (default: False)')
parser.add_argument('--disable-gated-conv', action='store_true',
                    help='disables gated convolutional structure (default: False)')
parser.add_argument('--disable-student-teacher', action='store_true',
                    help='uses a standard VAE without Student-Teacher architecture')

# Optimization related
parser.add_argument('--optimizer', type=str, default="adamnorm",
                    help="specify optimizer (default: rmsprop)")
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--early-stop', action='store_true',
                    help='enable early stopping (default: False)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')

# Regularizer related
parser.add_argument('--disable-regularizers', action='store_true',
                    help='disables mutual info and consistency regularizers')
parser.add_argument('--few_sort', action='store_true',
                    help='for few sort learning')
parser.add_argument('--monte-carlo-infogain', action='store_true',
                    help='use the MC version of mutual information gain / false is analytic (default: False)')
parser.add_argument('--continuous-mut-info', type=float, default=0.0,
                    help='-continuous_mut_info * I(z_c; x) is applied (opposite dir of disc)(default: 0.0)')
parser.add_argument('--discrete-mut-info', type=float, default=0.0,
                    help='+discrete_mut_info * I(z_d; x) is applied (default: 0.0)')
parser.add_argument('--kl-reg', type=float, default=1.0,
                    help='hyperparameter to scale KL term in ELBO')
parser.add_argument('--generative-scale-var', type=float, default=1.0,
                    help='scale variance of prior in order to capture outliers')
parser.add_argument('--consistency-gamma', type=float, default=1.0,
                    help='consistency_gamma * KL(Q_student | Q_teacher) (default: 1.0)')
parser.add_argument('--likelihood-gamma', type=float, default=0.0,
                    help='log-likelihood regularizer between teacher and student, 0 is disabled (default: 0.0)')
parser.add_argument('--mut-clamp-strategy', type=str, default="clamp",
                    help='clamp mut info by norm / clamp / none (default: clamp)')
parser.add_argument('--mut-clamp-value', type=float, default=100.0,
                    help='max / min clamp value if above strategy is clamp (default: 100.0)')
parser.add_argument('--ewc-gamma', type=float, default=0,
                    help='any value greater than 0 enables EWC with this hyper-parameter (default: 0)')

# Visdom parameters
parser.add_argument('--visdom-url', type=str, default="http://localhost",
                    help='visdom URL for graphs (default: http://localhost)')
parser.add_argument('--visdom-port', type=int, default="8097",
                    help='visdom port for graphs (default: 8097)')

# Device parameters
parser.add_argument('--seed', type=int, default=None,
                    help='seed for numpy and pytorch (default: None)')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpus available (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#device= torch.device("cuda:0")
device=torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(2)
print(torch.cuda.is_available())
# handle randomness / non-randomness
if args.seed is not None:
    print("setting seed %d" % args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed_all(args.seed)


def build_optimizer(model):
    optim_map = {
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "adamnorm": AdamNormGrad,
        "adadelta": optim.Adadelta,
        "sgd": optim.SGD,
        "lbfgs": optim.LBFGS
    }
    # filt = filter(lambda p: p.requires_grad, model.parameters())
    # return optim_map[args.optimizer.lower().strip()](filt, lr=args.lr)
    return optim_map[args.optimizer.lower().strip()](
        model.parameters(), lr=args.lr
    )
def register_plots(loss, grapher, epoch, prefix='train'):
    for k, v in loss.items():
        if isinstance(v, map):
            register_plots(loss[k], grapher, epoch, prefix=prefix)

        if 'mean' in k or 'scalar' in k:
            key_name = k.split('_')[0]
            value = v.item() if not isinstance(v, (float, np.float32, np.float64)) else v
            grapher.register_single({'%s_%s' % (prefix, key_name): [[epoch], [value]]},
                                    plot_type='line')


def register_images(images, names, grapher, prefix="train"):
    ''' helper to register a list of images '''
    if isinstance(images, list):
        assert len(images) == len(names)
        for im, name in zip(images, names):
            register_images(im, name, grapher, prefix=prefix)
    else:
        images = torch.min(images.detach(), ones_like(images))
        grapher.register_single({'{}_{}'.format(prefix, names): images},
                                plot_type='imgs')


def _add_loss_map(loss_tm1, loss_t):
    if not loss_tm1: # base case: empty dict
        resultant = {'count': 1}
        for k, v in loss_t.items():
            if 'mean' in k or 'scalar' in k:
                resultant[k] = v.detach()

        return resultant

    resultant = {}
    for (k, v) in loss_t.items():
        if 'mean' in k or 'scalar' in k:
            resultant[k] = loss_tm1[k] + v.detach()

    # increment total count
    resultant['count'] = loss_tm1['count'] + 1
    return resultant


def _mean_map(loss_map):
    for k in loss_map.keys():
        loss_map[k] /= loss_map['count']

    return loss_map

def train(epoch, model, fisher, optimizer, loader, grapher, prefix='train'):
    ''' train loop helper '''
    return execute_graph(epoch=epoch, model=model, fisher=fisher,
                         data_loader=loader, grapher=grapher,
                          optimizer=optimizer, prefix='train')


def test(epoch, model, fisher, loader, grapher, prefix='test'):
    test_loss= execute_graph(epoch, model=model, fisher=fisher, data_loader=loader, grapher=grapher, optimizer=None, prefix='test')

    return test_loss


def execute_graph(epoch, model, fisher, data_loader, grapher, optimizer=None, prefix='test'):
    ''' execute the graph; when 'train' is in the name the model runs the optimizer '''
    model.eval() if not 'train' in prefix else model.train()
    assert optimizer is not None if 'train' in prefix else optimizer is None
    loss_map, params, num_samples = {}, {}, 0
    print("training started")
    i=0
    for data, _ in data_loader:
        print("running the VAE and extract loss for data_ "+str(i))
        data = Variable(data).cuda() if args.cuda else Variable(data)
        
        if 'train' in prefix:
            # zero gradients on optimizer
            # before forward pass
            optimizer.zero_grad()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            
            output_map = model(data)
            loss_t = model.loss_function(output_map, fisher)
        i=i+1
        if 'train' in prefix:
            # compute bp and optimize
            
            loss_t['loss_mean'].backward()
            loss_t['grad_norm_mean'] = torch.norm( # add norm of vectorized grads to plot
                nn.utils.parameters_to_vector(model.parameters())
            )
            optimizer.step()

        with torch.no_grad() if 'train' not in prefix else dummy_context():
            loss_map = _add_loss_map(loss_map, loss_t)
            num_samples += data.size(0)
    
    loss_map = _mean_map(loss_map) # reduce the map to get actual means
    print('{}[Epoch {}][{} samples]: Average loss: {:.4f}\tELBO: {:.4f}\tKLD: {:.4f}\tNLL: {:.4f}\tMut: {:.4f}'.format(
        prefix, epoch, num_samples,
        loss_map['loss_mean'].item(),
        loss_map['elbo_mean'].item(),
        loss_map['kld_mean'].item(),
        loss_map['nll_mean'].item(),
        loss_map['mut_info_mean'].item()))

    # gather scalar values of reparameterizers (if they exist)
    reparam_scalars = model.student.get_reparameterizer_scalars()

    # plot the test accuracy, loss and images
    if grapher: # only if grapher is not None
        register_plots({**loss_map, **reparam_scalars}, grapher, epoch=epoch, prefix=prefix)
        images = [output_map['augmented']['data'], output_map['student']['x_reconstr']]
        img_names = ['original_imgs', 'vae_reconstructions']
        register_images(images, img_names, grapher, prefix=prefix)
        grapher.show()
    if 'train' not in prefix:
        print("saving the original and generated batches of images for epoch "+str(epoch))
        images = [output_map['augmented']['data'], output_map['student']['x_reconstr']]
        img_names = ['original_imgs', 'vae_reconstructions']
        os.makedirs(batch_path, mode=0o744, exist_ok=True)
        epoch_path = os.path.join(batch_path, str(args.num_sample))
        os.makedirs(epoch_path, mode=0o744, exist_ok=True)
        for i in range(0,len(images)):
            utils.save_image(images[i], os.path.join(epoch_path, str(img_names[i])+str(epoch)+".png"))
    # return this for early stopping
    loss_val = {'loss_mean': loss_map['loss_mean'].detach().item(),
                'elbo_mean': loss_map['elbo_mean'].detach().item()}
    loss_map.clear()
    params.clear()
    return loss_val




def generate(student_teacher, grapher, epoch, name='teacher'):
    model = {
        'teacher': student_teacher.teacher,
        'student': student_teacher.student
    }

    if model[name] is not None: # handle base case
        model[name].eval()
        # random generation
        gen = student_teacher.generate_synthetic_samples(model[name],
                                                         args.batch_size)
        gen = torch.min(gen, ones_like(gen))
        grapher.register_single({'generated_%s'%name: gen}, plot_type='imgs')
        
        # sequential generation for discrete and mixture reparameterizations
        if args.reparam_type == 'mixture' or args.reparam_type == 'discrete':
            gen = student_teacher.generate_synthetic_sequential_samples(model[name]).detach()
            gen = torch.min(gen, ones_like(gen))
            grapher.register_single({'sequential_generated_%s'%name: gen}, plot_type='imgs')
def get_myloader(args):
    loaders=get_loader(args)
    if args.disable_sequential: # vanilla batch training
        loaders = get_loader(args)
        loaders = [loaders] if not isinstance(loaders, list) else loaders
    else: # classes split
        loaders = get_split_data_loaders(args, num_classes=10)
    print("train = ", num_samples_in_loader(loaders[0].train_loader),
    " | test = ", num_samples_in_loader(loaders[0].test_loader))
    return loaders
def get_model(loader):
    args.img_shp =  loader[0].img_shp,
    if args.vae_type == 'sequential':
        # Sequential : P(y|x) --> P(z|y, x) --> P(x|z)
        # Keep a separate VAE spawn here in case we want
        # to parameterize the sequence of reparameterizers
        vae = SequentiallyReparameterizedVAE(loader[0].img_shp,
                                             kwargs=vars(args))
    elif args.vae_type == 'parallel':
        # Ours: [P(y|x), P(z|x)] --> P(x | z)
        vae = ParallellyReparameterizedVAE(loader[0].img_shp,
                                           kwargs=vars(args))
    else:
        raise Exception("unknown VAE type requested")

    # build the combiner which takes in the VAE as a parameter
    # and projects the latent representation to the output space
    student_teacher = StudentTeacher(vae, kwargs=vars(args))
    grapher = Grapher(env=student_teacher.get_name(),
                      server=args.visdom_url,
                      port=args.visdom_port)
    return student_teacher, grapher
def get_model_and_loader():
    ''' helper to return the model and the loader '''
    if args.disable_sequential: # vanilla batch training
        loaders = get_loader(args)
        loaders = [loaders] if not isinstance(loaders, list) else loaders
    else: # classes split
        loaders = get_split_data_loaders(args, num_classes=10)

    for l in loaders:
        print("train = ", num_samples_in_loader(l.train_loader),
              " | test = ", num_samples_in_loader(l.test_loader))

    # append the image shape to the config & build the VAE
    args.img_shp =  loaders[0].img_shp,
    if args.vae_type == 'sequential':
        # Sequential : P(y|x) --> P(z|y, x) --> P(x|z)
        # Keep a separate VAE spawn here in case we want
        # to parameterize the sequence of reparameterizers
        vae = SequentiallyReparameterizedVAE(loaders[0].img_shp,
                                             kwargs=vars(args))
    elif args.vae_type == 'parallel':
        # Ours: [P(y|x), P(z|x)] --> P(x | z)
        vae = ParallellyReparameterizedVAE(loaders[0].img_shp,
                                           kwargs=vars(args))
    else:
        raise Exception("unknown VAE type requested")

    # build the combiner which takes in the VAE as a parameter
    # and projects the latent representation to the output space
    student_teacher = StudentTeacher(vae, kwargs=vars(args))
    #student_teacher = init_weights(student_teacher)

    # build the grapher object
    grapher = Grapher(env=student_teacher.get_name(),
                      server=args.visdom_url,
                      port=args.visdom_port)

    return [student_teacher, loaders, grapher]


def lazy_generate_modules(model, img_shp):
    ''' Super hax, but needed for building lazy modules '''
    model.eval()
    data = float_type(args.cuda)(args.batch_size, *img_shp).normal_()
    model(Variable(data))


def test_and_generate(epoch, model, fisher, loader, grapher):
    test_loss = test(epoch=epoch, model=model,
                     fisher=fisher, loader=loader.test_loader,
                     grapher=grapher, prefix='test')
    generate(model, grapher, 'student') # generate student samples
    generate(model, grapher, 'teacher') # generate teacher samples
    return test_loss


def eval_model(data_loaders, model, fid_model, args):
    ''' simple helper to evaluate the model over all the loaders'''
    for loader in data_loaders:
        test_loss = test(epoch=-1, model=model, fisher=None,
                         loader=loader.test_loader, grapher=None, prefix='test')

        # evaluate and save away one-time metrics
        check_or_create_dir(os.path.join(args.output_dir))
        append_to_csv([test_loss['elbo_mean']], os.path.join(args.output_dir, "{}_test_elbo.csv".format(args.uid)))
        append_to_csv(calculate_consistency(model, loader, args.reparam_type, args.vae_type, args.cuda),
                      os.path.join(args.output_dir, "{}_consistency.csv".format(args.uid)))
        with open(os.path.join(args.output_dir, "{}_conf.json".format(args.uid)), 'w') as f:
            json.dump(model.student.config, f)

def train_loop(data_loaders, model, fid_model, grapher, k, args):
    print("length of data loader",len(data_loaders))
    ''' simple helper to run the entire train loop; not needed for eval modes'''
    optimizer = build_optimizer(model.student)     # collect our optimizer
    print("there are {} params with {} elems in the st-model and {} params in the student with {} elems".format(
        len(list(model.parameters())), number_of_parameters(model),
        len(list(model.student.parameters())), number_of_parameters(model.student))
    )
   
    # main training loop
    fisher = None
    print("data_loader_length", len(data_loaders))
    for j, loader in enumerate(data_loaders):
        num_epochs = args.epochs # TODO: randomize epochs by something like: + np.random.randint(0, 13)
        print("training current distribution for {} epochs".format(num_epochs))
        early = EarlyStopping(model, max_steps=50, burn_in_interval=None) if args.early_stop else None
                              #burn_in_interval=int(num_epochs*0.2)) if args.early_stop else None

        test_loss = None
        print("creating output directory")
        check_or_create_dir(os.path.join(args.output_dir))
        for epoch in range(1, num_epochs + 1):
            print("running for epoch {} ".format(epoch))
            
            train(epoch, model, fisher, optimizer, loader.train_loader, grapher)
            test_loss = test(epoch, model, fisher, loader.test_loader, grapher)
            if args.early_stop and early(test_loss['loss_mean']):
                early.restore() # restore and test+generate again
                test_loss = test_and_generate(epoch, model, fisher, loader, grapher)
                
                break
            print("calculated test_loss {}".format(epoch))
            generate(model, grapher, 'student') # generate student samples
            generate(model, grapher, 'teacher') # generate teacher samples
            check_or_create_dir(os.path.join(args.output_dir))
            
                
            if args.calculate_fid_with is not None:
                    num_fid_samples = args.num_sample
                    fid_score=calculate_fid(fid_model=fid_model,
                                        model=model,
                                        loader=loader, grapher=None,
                                        num_samples=num_fid_samples,
                                        cuda=args.cuda)
        
                    print("fid_score"+str(fid_score)+ "for_epoch"+str(epoch))
                    num_of_sample=args.num_sample
                    path_fid = os.path.join(args.output_dir,"sample_"+str(num_of_sample))
                    check_or_create_dir(os.path.join(path_fid))
                    append_to_csv([fid_score],
                          os.path.join(path_fid, "mnist_fid3.csv"))
                    #append_to_csv([fid_score],
                          #os.path.join(path_fid, "mnist_fid.csv"))
        # evaluate and save away one-time metrics, these include:
        #    1. test elbo
        #    2. FID
        #    3. consistency
        #    4. num synth + num true samples
        #    5. dump config to visdom
        
            #fid_loss_map = _add_loss_map(fid_loss_map, fid_score)
            
        ####################################################################################################
        append_to_csv([test_loss['elbo_mean']], os.path.join(args.output_dir, "{}_test_elbo.csv".format(args.uid)))
        append_to_csv([test_loss['elbo_mean']], os.path.join(args.output_dir, "{}_test_elbo.csv".format(args.uid)))
        num_synth_samples = np.ceil(epoch * args.batch_size * model.ratio)
        num_true_samples = np.ceil(epoch * (args.batch_size - (args.batch_size * model.ratio)))
        append_to_csv([num_synth_samples],os.path.join(args.output_dir, "{}_numsynth.csv".format(args.uid)))
        append_to_csv([num_true_samples], os.path.join(args.output_dir, "{}_numtrue.csv".format(args.uid)))
        append_to_csv([epoch], os.path.join(args.output_dir, "{}_epochs.csv".format(args.uid)))
        grapher.vis.text(num_synth_samples, opts=dict(title="num_synthetic_samples"))
        grapher.vis.text(num_true_samples, opts=dict(title="num_true_samples"))
        grapher.vis.text(pprint.PrettyPrinter(indent=4).pformat(model.student.config),
                         opts=dict(title="config"))

        # calc the consistency using the **PREVIOUS** loader
        if j > 0:
            append_to_csv(calculate_consistency(model, data_loaders[j - 1], args.reparam_type, args.vae_type, args.cuda),
                          os.path.join(args.output_dir, "{}_consistency.csv".format(args.uid)))


    
        print("making checkpoint directory")
        os.makedirs(mnist_checkpoint_dir, mode=0o744, exist_ok=True)
        checkpoint={"epoch":num_epochs*(k-1), "model_state":model.state_dict(),"fid_model_state":fid_model.state_dict(),  "fid":fid_score,"optim_state":optimizer.state_dict()}
        model.to(device)
        sample=args.num_sample
        #checkpoint  = Variable(checkpoint).cuda() if args.cuda else Variable(checkpoint)
        path_check=os.path.join(mnist_checkpoint_dir, "checkpoint_new_sample_{}.pth".format(sample))
        fid_model.to(device)
        grapher.save()
        torch.save(checkpoint,path_check)
        grapher.save() # save the remote visdom graphs
        if j != len(data_loaders) - 1:
            print("j", j)
            if args.ewc_gamma > 0:
                # calculate the fisher from the previous data loader
                print("computing fisher info matrix....")
                fisher_tmp = estimate_fisher(model.student, # this is pre-fork
                                             loader, args.batch_size,
                                             cuda=args.cuda)
                if fisher is not None:
                    assert len(fisher) == len(fisher_tmp), "#fisher params != #new fisher params"
                    for (kf, vf), (kft, vft) in zip(fisher.items(), fisher_tmp.items()):
                        fisher[kf] += fisher_tmp[kft]
                else:
                    fisher = fisher_tmp

            # spawn a new student & rebuild grapher; we also pass
            # the new model's parameters through a new optimizer.
            if not args.disable_student_teacher:
                model.fork()
                lazy_generate_modules(model, data_loaders[0].img_shp)
                optimizer = build_optimizer(model.student)
                print("there are {} params with {} elems in the st-model and {} params in the student with {} elems".format(
                    len(list(model.parameters())), number_of_parameters(model),
                    len(list(model.student.parameters())), number_of_parameters(model.student))
                )

            else:
                # increment anyway for vanilla models
                # so that we can have a separate visdom env
                model.current_model += 1

            grapher = Grapher(env=model.get_name(),
                              server=args.visdom_url,
                              port=args.visdom_port)

            

def _set_model_indices(model, grapher, idx, args):
    def _init_vae(img_shp, config):
        if args.vae_type == 'sequential':
            # Sequential : P(y|x) --> P(z|y, x) --> P(x|z)
            # Keep a separate VAE spawn here in case we want
            # to parameterize the sequence of reparameterizers
            vae = SequentiallyReparameterizedVAE(img_shp,
                                                 **{'kwargs': config})
        elif args.vae_type == 'parallel':
            # Ours: [P(y|x), P(z|x)] --> P(x | z)
            vae = ParallellyReparameterizedVAE(img_shp,
                                               **{'kwargs': config})
        else:
            raise Exception("unknown VAE type requested")

        return vae

    if idx > 0:         # create some clean models to later load in params
        model.current_model = idx
        if not args.disable_augmentation:
            model.ratio = idx / (idx + 1.0)
            num_teacher_samples = int(args.batch_size * model.ratio)
            num_student_samples = max(args.batch_size - num_teacher_samples, 1)
            print("#teacher_samples: ", num_teacher_samples,
                  " | #student_samples: ", num_student_samples)

            # copy args and reinit clean models for student and teacher
            config_base = vars(args)
            config_teacher = deepcopy(config_base)
            config_student = deepcopy(config_base)
            config_teacher['discrete_size'] += idx - 1
            config_student['discrete_size'] += idx
            model.student = _init_vae(model.student.input_shape, config_student)
            if not args.disable_student_teacher:
                model.teacher = _init_vae(model.student.input_shape, config_teacher)

        # re-init grapher
        grapher = Grapher(env=model.get_name(),
                          server=args.visdom_url,
                          port=args.visdom_port)

    return model, grapher


def run1(args):
    iteration=args.epochs
    torch.autograd.set_detect_anomaly(True)
    f=11
    print("loading data_loader for mnist dataset")
    data_loaders=get_myloader(args)
    print("retriving loaded model")
    path_check=os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(f-1))
    loaded_checkpoint=torch.load(path_check)
    vae = ParallellyReparameterizedVAE(data_loaders[0].img_shp,
                                           kwargs=vars(args))
    model1= StudentTeacher(vae, kwargs=vars(args))
    model1.load_state_dict(loaded_checkpoint["model_state"], torch.load(path_check, map_location=device))
    model1.to(device)
    print("loading fid_model")
    fid_batch_size = args.batch_size
    fid_type='conv'
    fid_model1= FID(data_loaders[0].img_shp, data_loaders[0].output_size, batch_size=fid_batch_size, fid_type=fid_type, kwargs=vars(args))
    fid_model1.load_state_dict(loaded_checkpoint["fid_model_state"], torch.load(path_check, map_location=device))
    fid_model1.to(device)
    i=1
    if args.few_sort:

                    print("starting few sort learning for mnist dataset")
                    # load the mnist loader it will act as a student 
                    
                    model, grapher=get_model(data_loaders) 
                    model.fork_student_teacher(model1)#loading the previous model into teacher 
                    lazy_generate_modules(model, data_loaders[0].img_shp)
                    #optimizer = build_optimizer(model.student) #build optimizer for student
                    print("start training for mnist with new grapher and fid_model")
                    #fid_batch_size = args.batch_size 
                    #fid_model= FID(data_loaders[0].img_shp, data_loaders[0].output_size, batch_size=fid_batch_size, fid_type=fid_type, kwargs=vars(args))
                    train_loop(data_loaders, model, fid_model1, grapher, i, args)
                    #train_loop(data_loaders, model, fid_model, grapher, i, args)
            
            
'''
            else:
                print("continuing mnist training")
            
                path_check=os.path.join(mnist_checkpoint_dir, "checkpoint_{}.pth".format(i-1))
                loaded_checkpoint1=torch.load(path_check)
                print("retriving loaded model")
                vae = ParallellyReparameterizedVAE(data_loaders[0].img_shp,
                                           kwargs=vars(args))
                print("model_loading")
                model2= StudentTeacher(vae, kwargs=vars(args))
                model2.load_state_dict(loaded_checkpoint1["model_state"], torch.load(path_check, map_location=device))
                model2.to(device)
                print("loading fid_model")
                fid_batch_size = args.batch_size
                fid_model2= FID(data_loaders[0].img_shp, data_loaders[0].output_size, batch_size=fid_batch_size,
                    fid_type=fid_type, kwargs=vars(args))
                fid_model2.load_state_dict(loaded_checkpoint1["fid_model_state"], torch.load(path_check, map_location=device))
                fid_model2.to(device)
                optimizer2 = build_optimizer(model2.student) 
                print("training for iteration:"+str(i))
                VAE_train_loop(data_loaders, grapher, model2, fid_model2, i,optimizer2, args)
            #train_loop(data_loaders, model1, fid_model1, grapher,i, args)
            #print("loading grapher")
                
            #VAE_train_loop(data_loaders, grapher, model2, fid_model2, i,optimizer2, args)
'''        
                    
if __name__ == "__main__":
    run1(args)
