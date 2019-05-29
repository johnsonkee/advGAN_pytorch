import torch
import torchvision.datasets
import torchvision.transforms as transforms
import time
import numpy as np

from torch.utils.data import DataLoader
from advGAN import AdvGAN_Attack
from models import MNIST_target_net
from torch.utils.data import TensorDataset


use_cuda=True
image_nc=1
epochs = 60
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "./MNIST_target_model.pth"
targeted_model = MNIST_target_net().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

mifgsm_b_adv = np.load('mifgsm_b_adv.npy')
my_label = mnist_dataset.train_labels.clone()
mydataset = TensorDataset(torch.tensor(mifgsm_b_adv),my_label)
my_dataloader = DataLoader(mydataset,batch_size=batch_size, shuffle=False, num_workers=1)

start_time = time.time()

advGAN = AdvGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

advGAN.train(dataloader, my_dataloader, epochs)

end_time = time.time()
print("advGAN training time is {}".format(end_time - start_time))
