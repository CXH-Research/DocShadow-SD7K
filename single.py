import warnings

import time
from accelerate import Accelerator
from torchvision.utils import save_image
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np
from config import Config
from models import *
from utils import *
import os

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)


def single_test(path_img):
    accelerator = Accelerator()

    inp = Image.open(path_img).convert('RGB')
    
    inp = np.array(inp)

    inp = F.to_tensor(inp)

    # Model & Metrics
    model = Model()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model)

    model.eval()

    if not os.path.exists("result"):
        os.makedirs("result")

    res = model(inp)

    save_image(res, os.path.join(os.getcwd(), "result", os.path.basename(path_img)))


if __name__ == '__main__':
    path_img = 'enter the path of image here'
    single_test(path_img)