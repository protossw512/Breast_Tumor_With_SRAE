from util import VGG_transforms, VGG_dataloader, bio_datasets
import numpy as np
import matplotlib.pyplot as plt
import torchvision

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated

if __name__ == '__main__':
    data_dir = "/home/wangxiny/Bio/Model_0412_1"
    data_transforms = VGG_transforms()
    image_datasets = bio_datasets(data_dir, data_transforms)
    dataloaders = VGG_dataloader(image_datasets)
    class_names = image_datasets['train'].classes
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
