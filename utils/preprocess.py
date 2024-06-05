from torchvision import transforms

# Transform to be applied to each image
resize_dims = (256, 256)
n_mean_std = [0.5, 0.5, 0.5]
data_transform = transforms.Compose([
    transforms.Resize(resize_dims, antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(n_mean_std, n_mean_std),
])
