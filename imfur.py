'''
!pip
install
torchsummary
!pip
install
torchgeometry
'''
from torchsummary import summary
from torchgeometry.losses import one_hot
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import imageio
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode, CenterCrop
from collections import OrderedDict
from PIL import ImageFilter
import random
import wandb

#!nvidia - smi - L

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# Number of class in the data set (3: neoplastic, non neoplastic, background)
num_classes = 3

# Number of epoch
epochs = 30

# Hyperparameters for training
learning_rate = 2e-04
batch_size = 4
display_step = 50

# Model path
checkpoint_path = '/kaggle/working/unet_model.pth'
pretrained_path = "/kaggle/input/pretrain/unet_model.pth"
# Initialize lists to keep track of loss and accuracy
loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []

transform = Compose([CenterCrop(1024),
                     Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
                     PILToTensor()])


class UNetDataClass(Dataset):
    def __init__(self, images_path, masks_path, transform, blur_radius=2, rotation_angle=180):
        super(UNetDataClass, self).__init__()

        images_list = os.listdir(images_path)
        masks_list = os.listdir(masks_path)

        images_list = [images_path + image_name for image_name in images_list]
        masks_list = [masks_path + mask_name for mask_name in masks_list]

        self.images_list = images_list
        self.masks_list = masks_list
        self.transform = transform
        self.blur_radius = blur_radius
        self.rotation_angle = rotation_angle

    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]

        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)

        # Additional Gaussian blur
        data = data.filter(ImageFilter.GaussianBlur(self.blur_radius))

        # Additional Random Rotation
        if self.rotation_angle > 0:
            random_angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            data = data.rotate(random_angle)
            label = label.rotate(random_angle)

        # Normalize
        data = self.transform(data) / 255
        label = self.transform(label) / 255

        label = torch.where(label > 0.65, 1.0, 0.0)

        label[2, :, :] = 0.0001
        label = torch.argmax(label, 0).type(torch.int64)

        return data, label

    def __len__(self):
        return len(self.images_list)


images_path = "/kaggle/input/bkai-igh-neopolyp/train/train/"
masks_path = "/kaggle/input/bkai-igh-neopolyp/train_gt/train_gt/"

unet_dataset = UNetDataClass(images_path, masks_path, transform)

train_size = 0.8
valid_size = 0.2

train_set, valid_set = random_split(unet_dataset,
                                    [int(train_size * len(unet_dataset)),
                                     int(valid_size * len(unet_dataset))])

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)


class inception_block_same(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inception_block_same, self).__init__()
        # [inp]->[conv 5x5]->[out]
        self.conv5_1 = nn.Conv2d(in_channels, out_channels // 16, kernel_size=1, stride=1, padding='same')
        self.conv5 = nn.Conv2d(out_channels // 16, out_channels // 4, kernel_size=5, stride=1, padding='same')
        # [inp]->[conv 3x3]->[out]
        self.conv3_1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding='same')
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding='same')
        # [inp]->([conv 3x1]+[conv 1x3])->[conv 3x3]->[out]
        self.conv33_1 = nn.Conv2d(in_channels, out_channels // 8, kernel_size=1, stride=1, padding='same')
        self.conv33_h = nn.Conv2d(out_channels // 8, out_channels // 4, kernel_size=(3, 1), stride=1, padding='same')
        self.conv33_v = nn.Conv2d(out_channels // 8, out_channels // 4, kernel_size=(1, 3), stride=1, padding='same')
        self.conv33_c = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=(3, 3), stride=1, padding='same')
        # [inp]->[maxpool]->[out]
        self.pool_1 = nn.Conv2d(in_channels, out_channels // 8, kernel_size=1, stride=1, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
        # [inp]->[out]
        self.conv1_1 = nn.Conv2d(in_channels, out_channels // 8, kernel_size=1, stride=1, padding='same')
        self.ReLU = nn.ReLU()

    def forward(self, x):
        # [inp]->[conv 5x5]->[out]
        l1 = self.conv5_1(x)
        l1 = self.conv5(l1)
        l1 = self.ReLU(l1)
        # [inp]->[conv 3x3]->[out]
        l2 = self.conv3_1(x)
        l2 = self.conv3(l2)
        l2 = self.ReLU(l2)
        # [inp]->([conv 3x1]+[conv 1x3])->[conv 3x3]->[out]
        l3 = self.conv33_1(x)

        l31 = self.conv33_h(l3)
        l31 = self.ReLU(l31)
        l32 = self.conv33_v(l3)
        l32 = self.ReLU(l32)

        l3 = torch.cat([l31, l32], axis=1)
        l3 = self.conv33_c(l3)
        l3 = self.ReLU(l3)
        # [inp]->[maxpool]->[out]
        l4 = self.pool_1(x)
        l4 = self.ReLU(l4)
        l4 = self.pool(l4)
        # [inp]->[out]
        l5 = self.conv1_1(x)
        l5 = self.ReLU(l5)
        out = torch.cat([l1, l2, l3, l4, l5], axis=1)
        return out


class inception_block_down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inception_block_down, self).__init__()
        # [inp]->[conv 5x5]->[out]
        self.conv5_1 = nn.Conv2d(in_channels, out_channels // 16, kernel_size=1, stride=1, padding='same')
        self.conv5 = nn.Conv2d(out_channels // 16, out_channels // 4, kernel_size=5, stride=2, padding=2)
        # [inp]->[conv 3x3]->[out]
        self.conv3_1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding='same')
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=2, padding=1)
        # [inp]->([conv 3x1]+[conv 1x3])->[conv 3x3]->[out]
        self.conv33_1 = nn.Conv2d(in_channels, out_channels // 8, kernel_size=1, stride=1, padding='same')
        self.conv33_h = nn.Conv2d(out_channels // 8, out_channels // 4, kernel_size=(3, 1), stride=1, padding='same')
        self.conv33_v = nn.Conv2d(out_channels // 8, out_channels // 4, kernel_size=(1, 3), stride=1, padding='same')
        self.conv33_c = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=(3, 3), stride=2, padding=1)
        # [inp]->[maxpool]->[out]
        self.pool_1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding='same')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.ReLU = nn.ReLU()

    def forward(self, x):
        # [inp]->[conv 5x5]->[out]
        l1 = self.conv5_1(x)
        l1 = self.conv5(l1)
        l1 = self.ReLU(l1)
        # [inp]->[conv 3x3]->[out]
        l2 = self.conv3_1(x)
        l2 = self.conv3(l2)
        l2 = self.ReLU(l2)
        # [inp]->([conv 3x1]+[conv 1x3])->[conv 3x3]->[out]
        l3 = self.conv33_1(x)

        l31 = self.conv33_h(l3)
        l31 = self.ReLU(l31)
        l32 = self.conv33_v(l3)
        l32 = self.ReLU(l32)

        l3 = torch.cat([l31, l32], axis=1)
        l3 = self.conv33_c(l3)
        l3 = self.ReLU(l3)
        # [inp]->[maxpool]->[out]
        l4 = self.pool_1(x)
        l4 = self.ReLU(l4)
        l4, indices = self.pool(l4)

        out = torch.cat([l1, l2, l3, l4], axis=1)
        return out, indices


class inception_block_up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(inception_block_up, self).__init__()
        # [inp]->[transposeconv 5x5]->[out]
        self.conv5_1 = nn.Conv2d(in_channels, out_channels // 16, kernel_size=(1, 1), stride=1, padding='same')
        self.conv5 = nn.ConvTranspose2d(out_channels // 16, out_channels // 4, kernel_size=(5, 5), stride=2, padding=2,
                                        output_padding=1)
        # [inp]->[transposeconv 2x2]->[out]
        self.conv2_1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 1), stride=1, padding='same')
        self.conv2 = nn.ConvTranspose2d(out_channels // 4, out_channels // 2, kernel_size=(2, 2), stride=2)
        # [inp]->[unmaxpool 2x2]->[out]
        self.pool_1 = nn.Conv2d(in_channels, skip_channels // 4, kernel_size=(1, 1), stride=1, padding='same')
        self.pool = nn.MaxUnpool2d(kernel_size=(2, 2), stride=2)
        self.pool_a = nn.Conv2d(skip_channels // 4, out_channels // 4, kernel_size=1, stride=1, padding='same')

        self.ReLU = nn.ReLU()

    def forward(self, x, indices):
        # [inp]->[transposeconv 5x5]->[out]
        l1 = self.conv5_1(x)
        l1 = self.conv5(l1)
        l1 = self.ReLU(l1)
        # [inp]->[transposeconv 2x2]->[out]
        l2 = self.conv2_1(x)
        l2 = self.conv2(l2)
        l2 = self.ReLU(l2)
        # [inp]->[unmaxpool 2x2]->[out]
        l3 = self.pool_1(x)
        l3 = self.ReLU(l3)
        l3 = self.pool(l3, indices)
        l3 = self.pool_a(l3)
        l3 = self.ReLU(l3)

        out = torch.cat([l1, l2, l3], axis=1)
        return out


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.same = inception_block_same(in_channels, out_channels)
        self.down = inception_block_down(out_channels, out_channels)

    def forward(self, x):
        skip = self.same(x)
        out, indices = self.down(skip)
        return out, skip, indices


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(decoder, self).__init__()
        self.up = inception_block_up(in_channels, out_channels, skip_channels)
        self.same = inception_block_same(out_channels + skip_channels, out_channels)

    def forward(self, x, skip, indices):
        x = self.up(x, indices)
        catted = torch.cat([x, skip], axis=1)
        out = self.same(catted)
        return out


class UnetModel(nn.Module):
    def __init__(self, n_class=3):
        super(UnetModel, self).__init__()
        # 5 encoder:
        self.enc1 = encoder(3, 32)
        self.enc2 = encoder(32, 64)
        self.enc3 = encoder(64, 128)
        self.enc4 = encoder(128, 256)
        self.enc5 = encoder(256, 512)
        # 2 same block
        self.same1 = inception_block_same(512, 512)
        self.same2 = inception_block_same(512, 512)
        # 5 decoder
        self.dec5 = decoder(512, 256, 512)
        self.dec4 = decoder(256, 128, 256)
        self.dec3 = decoder(128, 64, 128)
        self.dec2 = decoder(64, 32, 64)
        self.dec1 = decoder(32, 16, 32)
        # conv 1x1
        self.out = nn.Conv2d(16, n_class, kernel_size=1, padding='same')

    def forward(self, image):
        o1, s1, i1 = self.enc1(image)
        o2, s2, i2 = self.enc2(o1)
        o3, s3, i3 = self.enc3(o2)
        o4, s4, i4 = self.enc4(o3)
        o5, s5, i5 = self.enc5(o4)

        b1 = self.same1(o5)
        b2 = self.same2(b1)

        d5 = self.dec5(b2, s5, i5)
        d4 = self.dec4(d5, s4, i4)
        d3 = self.dec3(d4, s3, i3)
        d2 = self.dec2(d3, s2, i2)
        d1 = self.dec1(d2, s1, i1)

        ret = self.out(d1)

        return ret


class CEDiceLoss(nn.Module):
    def __init__(self, weights) -> None:
        super(CEDiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.weights: torch.Tensor = weights

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(input.shape))
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}".format(
                    input.device, target.device))
        if not self.weights.shape[1] == input.shape[1]:
            raise ValueError("The number of weights must equal the number of classes")
        if not torch.sum(self.weights).item() == 1:
            raise ValueError("The sum of all weights must equal 1")

        # cross entropy loss
        celoss = nn.CrossEntropyLoss(self.weights)(input, target)

        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target, num_classes=input.shape[1],
                                 device=input.device, dtype=input.dtype)

        # compute the actual dice score
        dims = (2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        cardinality = torch.sum(input_soft + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)

        dice_score = torch.sum(dice_score * self.weights, dim=1)

        return torch.mean(1. - dice_score) + celoss


def weights_init(model):
    if isinstance(model, nn.Linear):
        # Xavier Distribution
        torch.nn.init.xavier_uniform_(model.weight)


def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


# Train function for each epoch
def train(train_dataloader, valid_dataloader, learing_rate_scheduler, epoch, display_step):
    print(f"Start epoch #{epoch + 1}, learning rate for this epoch: {learing_rate_scheduler.get_last_lr()}")
    start_time = time.time()
    train_loss_epoch = 0
    test_loss_epoch = 0
    last_loss = 999999999
    model.train()
    for i, (data, targets) in enumerate(train_dataloader):

        # Load data into GPU
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(data)

        # Backpropagation, compute gradients
        loss = loss_function(outputs, targets.long())
        loss.backward()

        # Apply gradients
        optimizer.step()

        # Save loss
        train_loss_epoch += loss.item()
        if (i + 1) % display_step == 0:
            #             accuracy = float(test(test_loader))
            print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {:.4f}'.format(
                epoch + 1, (i + 1) * len(data), len(train_dataloader.dataset),
                100 * (i + 1) * len(data) / len(train_dataloader.dataset),
                loss.item()))

    print(f"Done epoch #{epoch + 1}, time for this epoch: {time.time() - start_time}s")
    train_loss_epoch /= (i + 1)

    # Evaluate the validation set
    model.eval()
    with torch.no_grad():
        for data, target in valid_dataloader:
            data, target = data.to(device), target.to(device)
            test_output = model(data)
            test_loss = loss_function(test_output, target)
            test_loss_epoch += test_loss.item()

    test_loss_epoch /= (i + 1)

    return train_loss_epoch, test_loss_epoch


# Test function
def test(dataloader):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            test_loss += targets.size(0)
            correct += torch.sum(pred == targets).item()
    return 100.0 * correct / test_loss


model = UnetModel()
# model.apply(weights_init)
# model = nn.DataParallel(model)
checkpoint = torch.load(pretrained_path)

new_state_dict = OrderedDict()
for k, v in checkpoint['model'].items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model = nn.DataParallel(model)
model.to(device)

weights = torch.Tensor([[0.4, 0.55, 0.05]]).cuda()
loss_function = CEDiceLoss(weights)

# Define the optimizer (Adam optimizer)
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
# optimizer.load_state_dict(checkpoint['optimizer'])

# Learning rate scheduler
learing_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)

save_model(model, optimizer, checkpoint_path)
#skip training
'''
wandb.login(
    # set the wandb project where this run will be logged
    #     project= "PolypSegment",
    key="censored",
)
wandb.init(
    project="bkao-neopolyp"
)
# Training loop
train_loss_array = []
test_loss_array = []
last_loss = 9999999999999
for epoch in range(epochs):
    train_loss_epoch = 0
    test_loss_epoch = 0
    (train_loss_epoch, test_loss_epoch) = train(train_dataloader,
                                                valid_dataloader,
                                                learing_rate_scheduler, epoch, display_step)

    if test_loss_epoch < last_loss:
        save_model(model, optimizer, checkpoint_path)
        last_loss = test_loss_epoch

    learing_rate_scheduler.step()
    train_loss_array.append(train_loss_epoch)
    test_loss_array.append(test_loss_epoch)
    wandb.log({"Train loss": train_loss_epoch, "Valid loss": test_loss_epoch})
#     train_accuracy.append(test(train_loader))
#     valid_accuracy.append(test(test_loader))
#     print("Epoch {}: loss: {:.4f}, train accuracy: {:.4f}, valid accuracy:{:.4f}".format(epoch + 1,
#                                         train_loss_array[-1], train_accuracy[-1], valid_accuracy[-1]))
'''

torch.cuda.empty_cache()

# load_model(model, checkpoint)

plt.rcParams['figure.dpi'] = 90
plt.rcParams['figure.figsize'] = (6, 4)
epochs_array = range(epochs)

# Plot Training and Test loss
plt.plot(epochs_array, train_loss_array, 'g', label='Training loss')
# plt.plot(epochs_array, test_loss_array, 'b', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

for i, (data, label) in enumerate(train_dataloader):
    img = data
    mask = label
    break

fig, arr = plt.subplots(4, 3, figsize=(16, 12))
arr[0][0].set_title('Image')
arr[0][1].set_title('Segmentation')
arr[0][2].set_title('Predict')

model.eval()
with torch.no_grad():
    predict = model(img)

for i in range(4):
    arr[i][0].imshow(img[i].permute(1, 2, 0));

    arr[i][1].imshow(F.one_hot(mask[i]).float())

    arr[i][2].imshow(F.one_hot(torch.argmax(predict[i], 0).cpu()).float())


class UNetTestDataClass(Dataset):
    def __init__(self, images_path, transform):
        super(UNetTestDataClass, self).__init__()

        images_list = os.listdir(images_path)
        images_list = [images_path + i for i in images_list]

        self.images_list = images_list
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.images_list[index]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data) / 255
        return data, img_path, h, w

    def __len__(self):
        return len(self.images_list)


path = '/kaggle/input/bkai-igh-neopolyp/test/test/'
unet_test_dataset = UNetTestDataClass(path, transform)
test_dataloader = DataLoader(unet_test_dataset, batch_size=8, shuffle=True)

for i, (data, path, h, w) in enumerate(test_dataloader):
    img = data
    break

fig, arr = plt.subplots(5, 2, figsize=(16, 12))
arr[0][0].set_title('Image');
arr[0][1].set_title('Predict');

model.eval()
with torch.no_grad():
    predict = model(img)

for i in range(5):
    arr[i][0].imshow(img[i].permute(1, 2, 0));
    arr[i][1].imshow(F.one_hot(torch.argmax(predict[i], 0).cpu()).float())

model.eval()
if not os.path.isdir("/kaggle/working/predicted_masks"):
    os.mkdir("/kaggle/working/predicted_masks")
for _, (img, path, H, W) in enumerate(test_dataloader):
    a = path
    b = img
    h = H
    w = W

    with torch.no_grad():
        predicted_mask = model(b)
    for i in range(len(a)):
        image_id = a[i].split('/')[-1].split('.')[0]
        filename = image_id + ".png"

        mask2img = (ToPILImage()(F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))

        mask2img = Resize((1024, 1024), interpolation=InterpolationMode.NEAREST)(mask2img)
        mask2img = CenterCrop((h[i].item(), w[i].item()))(mask2img)

        mask2img.save(os.path.join("/kaggle/working/predicted_masks/", filename))


def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)


def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)


def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:, :, ::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:, :, channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


MASK_DIR_PATH = '/kaggle/working/predicted_masks'  # change this to the path to your output mask folder
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']
df.to_csv(r'output.csv', index=False)