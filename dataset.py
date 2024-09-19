path_base = './data'
dir_train   = 'train'
dir_test    = 'test'
dir_conform = 'normal'
dir_defect  = 'patient'

path_train = os.path.join(path_base,dir_train)
path_test  = os.path.join(path_base,dir_test)

path_train_conform  = os.path.join(path_train,dir_conform)
path_train_defect   = os.path.join(path_train,dir_defect)
path_test_conform   = os.path.join(path_test,dir_conform)
path_test_defect    = os.path.join(path_test,dir_defect)




# Transformations pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomAffine(degrees=(0, 180), translate=(0, 0.01), scale=(0.9, 1), fill=255),
    transforms.RandomPerspective(distortion_scale=0.2, fill=255),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Train set
trainset = datasets.ImageFolder(path_train, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
print(trainset)


# Test set
testset = datasets.ImageFolder(path_test, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
print("\n",testset)
