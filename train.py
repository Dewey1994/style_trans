import argparse
import os
import sys
import random
from PIL import Image
import numpy as np
import torch
import glob
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from models import TransformerNet, VGG16
from utils import *


def train():
    parser = argparse.ArgumentParser(description='parser for style transfer')
    parser.add_argument('--dataset_path', type=str, default=r'C:\Users\Dewey\data\celeba', help='path to training dataset')
    parser.add_argument('--style_image', type=str, default='mosaic.jpg', help='path to style img')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--image_size',type=int, default=256, help='training image size')
    parser.add_argument('--style_img_size', type=int, default=256, help='style image size')
    parser.add_argument("--lambda_content", type=float, default=1e5, help="Weight for content loss")
    parser.add_argument("--lambda_style", type=float, default=1e10, help="Weight for style loss")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument("--checkpoint_model", type=str, help="Optional path to checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=2000, help="Batches between saving model")
    parser.add_argument("--sample_interval", type=int, default=1000, help="Batches between saving image samples")
    parser.add_argument('--sample_format',type=str, default='jpg', help='sample image format')
    args = parser.parse_args()

    style_name = args.style_image.split('/')[-1].split('.')[0]
    os.makedirs(f'images/outputs/{style_name}-training', exist_ok=True)  # f-string格式化字符串
    os.makedirs('checkpoints', exist_ok=True)

    def save_sample(batch):
        transformer.eval()
        with torch.no_grad():
            output = transformer(image_samples.to(device))
            img_grid = denormalize(torch.cat((image_samples.cpu(), output.cpu()), 2))
            save_image(img_grid, f"images/outputs/{style_name}-training/{batch}.jpg", nrow=4)
            transformer.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = datasets.ImageFolder(args.dataset_path, train_transform(args.image_size))
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    transformer = TransformerNet().to(device)
    vgg = VGG16(requires_grad=False).to(device)

    if args.checkpoint_model:
        transformer.load_state_dict(torch.load(args.checkpoint_model))

    optimizer = Adam(transformer.parameters(), lr=args.lr)
    l2_loss = nn.MSELoss().to(device)

    # load style image
    style = style_transform(args.style_img_size)(Image.open(args.style_image))
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)

    # style_image features
    style_features = vgg(style)
    gram_style = [gram(x) for x in style_features]

    # visualization the image
    image_samples = []
    for path in random.sample(glob.glob(f'{args.dataset_path}/*/*.{args.sample_format}'), 8):
        image_samples += [style_transform(args.image_size)(Image.open(path))]
    image_samples = torch.stack(image_samples)
    c_loss = 0
    s_loss = 0
    t_loss = 0

    for epoch in range(args.epochs):
        for i, (img,_) in enumerate(dataloader):

            optimizer.zero_grad()

            image_original = img.to(device)
            image_transformed = transformer(image_original)

            origin_features = vgg(image_original)
            transformed_features = vgg(image_transformed)

            content_loss = args.lambda_content * l2_loss(transformed_features.relu_2_2, origin_features.relu_2_2)

            style_loss = 0
            for ii, jj in zip(transformed_features, gram_style):
                gram_t_features = gram(ii)
                style_loss += l2_loss(gram_t_features, jj) # buyiyang
            style_loss *= args.lambda_style

            loss = content_loss + style_loss
            loss.backward()
            optimizer.step()

            c_loss += content_loss.item()
            s_loss += style_loss.item()
            t_loss += loss.item()
            print('[Epoch %d/%d] [Batch %d/%d] [Content: %.2f (%.2f) Style: %.2f (%.2f) Total: %.2f (%.2f)]' % (
                epoch + 1,
                args.epochs,
                i,
                len(train_dataset),
                content_loss.item(),
                np.mean(c_loss),
                style_loss.item(),
                np.mean(s_loss),
                loss.item(),
                np.mean(t_loss),
            ))

            batches_done = epoch * len(dataloader) + i + 1
            if batches_done % args.sample_interval == 0:
                save_sample(batches_done)

            if args.checkpoint_interval > 0 and batches_done % args.checkpoint_interval == 0:
                style_name = os.path.basename(args.style_image).split(".")[0]
                torch.save(transformer.state_dict(), f"checkpoints/{style_name}_{batches_done}.pth")


if __name__ == '__main__':
    train()




