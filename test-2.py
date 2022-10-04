import torch
from utils.visualisers import float_to_uint8_image_tensor, show_uint8_image_tensor
from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader

from torch.utils.data import Subset, random_split

import torchvision.transforms.functional as F

from network.datamodules import FullTableDatasetModule

table_datamodule = FullTableDatasetModule(
    path='dataset/',
    batch_size=2,
    train_eval_split=0.8
)

train_dataloader = table_datamodule.train_dataloader()

images, targets = next(iter(train_dataloader))

for t in targets:
    print(t)






# def collate_fn(batch):
#     return tuple(zip(*batch))

# t = Compose([
#         PILToTensor(),
#         ConvertImageDtype(torch.float)
# ])

# full_dataset = CSVDataset(path="dataset/", tranforms=t)

# dataset = Subset(dataset=full_dataset, indices=range(20))

# train_set, eval_set = random_split(dataset=dataset, lengths=[16, 4])

# train_loader = DataLoader(dataset=train_set, batch_size=2, collate_fn=collate_fn)

# images, targets = next(iter(train_loader))

# # print()
# # print(image)
# # print(targets)

# print(type(images))
# print()
# print()
# targets = [{k: v for k, v in t.items()} for t in targets]

# print(type(targets))
# print()
# for t in targets:
#     print(t)



# import pandas as pd

# df = pd.DataFrame({'State':['Texas', 'Texas', 'Florida', 'Florida'], 
#                    'a':[1,3,5,7], 'b':[2,4,6,8], 'c':['table','table','table','table']})

# def listify(x):

#     df = x[['a', 'b']]
#     print()
#     print()
#     print(df.values.tolist())
#     print()
#     print()

#     return list([x['a'], x['b']])

# print(df)
# print()

# result_df = df.groupby('State').apply(lambda x: x[['a', 'b', 'c']].values.tolist())

# print(type(result_df.at['Florida']))
# print()