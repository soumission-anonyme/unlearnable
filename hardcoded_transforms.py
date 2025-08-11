import torch
import torchvision

def transforms(t):
    return globals().get(t)

cifar10_T = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

cifar10_TN = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

cifar10_CFTN = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])


cifar10_CFTNE = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

cifar10_IRFTN = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

cifar10_CFT = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
])


cifar10_CFJGTUNE = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(                 # Randomly change brightness, contrast, saturation
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    ),
    torchvision.transforms.RandomApply([                # Randomly apply Gaussian Blur
        torchvision.transforms.GaussianBlur(kernel_size=3)
    ], p=0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x + 0.1 * torch.randn_like(x)),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
])

tiny_CFJTN = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(64),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
])

tiny_CFJT = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(64),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    torchvision.transforms.ToTensor(),
])

tiny_FT = torchvision.transforms.Compose([
    # torchvision.transforms.RandomResizedCrop(64),
    torchvision.transforms.RandomHorizontalFlip(),
    # torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    torchvision.transforms.ToTensor(),
])

tiny_T = torchvision.transforms.Compose([
    # torchvision.transforms.Resize(64),
    # torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
])

tiny_TN = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
])

if __name__ == "__main__":
    print(globals())