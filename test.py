from retinanet import dataloader
from retinanet.dataloader import CSVDataset, collater, AspectRatioBasedSampler, Resizer, Normalizer
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

path_to_train = "dataset/train.csv"
path_to_class_list = "dataset/classes.csv"


dataset = CSVDataset(train_file=path_to_train, class_list=path_to_class_list)

train_set, val_set = random_split(dataset=dataset, lengths=[1000, 120])

print("Number of training images: {}".format(len(train_set)))
print("Number of validation images: {}".format(len(val_set)))
print()

sampler = AspectRatioBasedSampler(data_source=train_set, batch_size=1, drop_last=False)
train_loader = DataLoader(dataset=train_set)

it = iter(train_loader)
first = next(it)

print(first)