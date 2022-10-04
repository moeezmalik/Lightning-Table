

def collate_fn(batch):
    return tuple(zip(*batch))

# Also add IoU calculation and other metrics here