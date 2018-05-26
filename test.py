import argparse
import logging
import tensorboardX
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from model.model import VAE_mnist
from data_loader import MnistDataLoader
from trainer import Trainer



logging.basicConfig(level=logging.INFO, format='')

parser = argparse.ArgumentParser(description='PyTorch Template')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--verbosity', default=2, type=int,
                    help='verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)')
parser.add_argument('--save-dir', default='saved', type=str,
                    help='directory of saved model (default: saved)')
parser.add_argument('--save-freq', default=1, type=int,
                    help='training checkpoint frequency (default: 1)')
parser.add_argument('--data-dir', default='datasets', type=str,
                    help='directory of training/testing data (default: datasets)')
parser.add_argument('--validation-split', default=0.1, type=float,
                    help='ratio of split validation data, [0.0, 1.0) (default: 0.1)')
parser.add_argument('--validation-fold', default=0, type=int,
                    help='select part of data to be used as validation set (default: 0)')
parser.add_argument('--no-cuda', action="store_true",
                    help='use CPU instead of GPU')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

hidden_size = 40
model = VAE_mnist(hidden_size)

if args.resume is '':
    model_path = 'saved/VAE_mnist/model_best.pth.tar'
else:
    model_path = args.resume

checkpoint = torch.load(model_path)
# load state dict from checkpoint
state_dict = checkpoint['state_dict']

# load params from state dict
model.load_state_dict(state_dict)
model.eval()

model.summary()

trsfm = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = datasets.MNIST('../data', train=False, download=True, transform=trsfm)

avg_mu = torch.zeros((10, hidden_size), dtype=torch.float32)
avg_logvar = torch.zeros((10, hidden_size), dtype=torch.float32)
label_count = torch.zeros((10, 1), dtype=torch.float32)


out_path = 'saved/generated_images'

with torch.no_grad():
    for index in range(20):
        # index = 5
        image, target = dataset[index]
        
        image.to(device)
        image = image.unsqueeze(0)
        output, mu, logvar = model(image)
        # print(mu)
        # print(logvar)
        # rand_out = model.decoder(torch.randn_like(mu))
        # print(rand_out - output)
        
        output = output.view(28, 28)
        # print(output.shape)
        avg_mu[target, :] += mu[0, :]
        avg_logvar[target, :] += logvar[0, :]
        label_count[target] += 1
        save_image(output, out_path + "/img{}.png".format(index))
        # plt.imshow(output)
        # plt.savefig(out_path + "img{}.png".format(index))
        # print(mu.shape)
        # break

avg_mu /= label_count
avg_logvar /= label_count

# print(avg_mu)
    # print(model.encoder.output())

