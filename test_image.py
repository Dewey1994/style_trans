from models import TransformerNet
from utils import *
import torch
from torch.autograd import Variable
import argparse
import os
import tqdm
from torchvision.utils import save_image
from PIL import Image

def test():
    image_path = './image_folder'
    ckpt = './path_to_ckpt'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = test_transform()

    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(ckpt))
    transformer.eval()
    image = torch.Tensor(transform(Image.open(image_path))).to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        style_trans_img = denormalize(transformer(image)).cpu()

    fn = image_path.split("/")[-1]
    save_image(style_trans_img, f"images/outputs/stylized-{fn}")


if __name__ == '__main__':
    test()
