import warnings

import time
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, mean_squared_error, structural_similarity_index_measure
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_validation_data
from models import *
from utils import *

warnings.filterwarnings('ignore')

opt = Config('config.yml')

seed_everything(opt.OPTIM.SEED)


def infer():
    accelerator = Accelerator()

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR

    val_dataset = get_validation_data(val_dir, {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H, 'ori': True})
    testloader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

    # Model & Metrics
    model = Model()

    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)

    model.eval()

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_rmse = 0

    if not os.path.exists("result"):
        os.makedirs("result")

    total_time = 0

    for idx, test_data in enumerate(tqdm(testloader)):
        # get the inputs; data is a list of [targets, inputs, filename]
        inp = test_data[0].contiguous()
        tar = test_data[1]

        start_time = time.time()
        with torch.no_grad():
            res = model(inp)
        end_time = time.time()

        total_time += end_time - start_time

        save_image(res, os.path.join(os.getcwd(), "result", test_data[2][0] + '_pred.png'))
        save_image(tar, os.path.join(os.getcwd(), "result", test_data[2][0] + '_gt.png'))

        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1)
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1)
        stat_rmse += mean_squared_error(torch.mul(res, 255), torch.mul(tar, 255), squared=False)

    total_time /= len(testloader)
    stat_psnr /= size
    stat_ssim /= size
    stat_rmse /= size

    print("PSNR: {}, SSIM: {}, RMSE: {}, TIM: {}".format(stat_psnr, stat_ssim, stat_rmse, total_time))


if __name__ == '__main__':
    infer()
