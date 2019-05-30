import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
import numpy as np
import time
from torch.utils.data import TensorDataset

use_cuda=True
image_nc=1
batch_size = 128

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
pretrained_model = "./MNIST_target_model.pth"
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_20.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.2, 0.2)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('MNIST training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(mnist_dataset)))

# test adversarial examples in MNIST testing dataset
mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0

start_time = time.time()
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.2, 0.2)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    tmp = adv_img.detach().cpu().numpy()
    np.save("norm2_modelA_originGAN.npy",tmp)

    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

end_time = time.time()
print("generating adversary time: {}".format(end_time-start_time))
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))


# my adv examples
mifgsm_b_adv = np.load('mifgsm_b_adv.npy')
my_label = mnist_dataset.train_labels.clone()
mydataset = TensorDataset(torch.tensor(mifgsm_b_adv),my_label)
my_dataloader = DataLoader(mydataset,batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
for i, data in enumerate(my_dataloader, 0):
    adv_img, test_label = data
    adv_img, test_label = adv_img.to(device), test_label.to(device)
    tmp = adv_img.detach().cpu().numpy()
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

end_time = time.time()
print("generating adversary time: {}".format(end_time-start_time))

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in my adv: %f\n'%(num_correct.item()/len(mnist_dataset_test)))
