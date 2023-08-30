import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab
from torchvision.models.vgg import vgg16, VGG16_Weights


class Perceptual(torch.nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT).features[:16]
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            module.to(x.device)
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        for pred_im_feature, gt_feature in zip(pred_im_features, gt_features):
            loss.append(F.mse_loss(pred_im_feature, gt_feature))
        return sum(loss) / len(loss)


# class FusionLoss(nn.Module):
#
#     def __init__(self, ssim_window_size=5, alpha=0.5):
#
#         """Initialisation of the DeepLPF loss function
#         :param ssim_window_size: size of averaging window for SSIM
#         :param alpha: interpolation paramater for L1 and SSIM parts of the loss
#         :returns: N/A
#         :rtype: N/A
#         """
#
#         super(FusionLoss, self).__init__()
#         self.alpha = alpha
#         self.ssim_window_size = ssim_window_size
#
#     def create_window(self, window_size, num_channel):
#         """Window creation function for SSIM metric. Gaussian weights are applied to the window.
#         Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
#
#         :param window_size: size of the window to compute statistics
#         :param num_channel: number of channels
#         :returns: Tensor of shape Cx1xWindow_sizexWindow_size
#         :rtype: Tensor
#         """
#
#         _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
#         _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#         window = Variable(_2D_window.expand(num_channel, 1, window_size, window_size).contiguous())
#
#         return window
#
#     def gaussian(self, window_size, sigma):
#         """
#         Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
#         :param window_size: size of the SSIM sampling window e.g. 11
#         :param sigma: Gaussian variance
#         :returns: 1xWindow_size Tensor of Gaussian weights
#         :rtype: Tensor
#         """
#         gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#         return gauss / gauss.sum()
#
#     def compute_ssim(self, img1, img2):
#         """Computes the structural similarity index between two images. This function is differentiable.
#         Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
#         :param img1: image Tensor BxCxHxW
#         :param img2: image Tensor BxCxHxW
#         :returns: mean SSIM
#         :rtype: float
#
#         """
#         (_, num_channel, _, _) = img1.size()
#         window = self.create_window(self.ssim_window_size, num_channel)
#         device = img1.device
#
#         if img1.is_cuda:
#             window = window.to(device)
#             window = window.type_as(img1)
#
#         mu1 = F.conv2d(
#             img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
#         mu2 = F.conv2d(
#             img2, window, padding=self.ssim_window_size // 2, groups=num_channel)
#
#         mu1_sq = mu1.pow(2)
#         mu2_sq = mu2.pow(2)
#         mu1_mu2 = mu1 * mu2
#
#         sigma1_sq = F.conv2d(
#             img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
#         sigma2_sq = F.conv2d(
#             img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
#         sigma12 = F.conv2d(
#             img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2
#
#         C1 = 0.01 ** 2
#         C2 = 0.03 ** 2
#
#         ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
#         ssim_map2 = ((mu1_sq.to(device) + mu2_sq.to(device) + C1) *
#                      (sigma1_sq.to(device) + sigma2_sq.to(device) + C2))
#         ssim_map = ssim_map1.to(device) / ssim_map2.to(device)
#         v1 = 2.0 * sigma12.to(device) + C2
#         v2 = sigma1_sq.to(device) + sigma2_sq.to(device) + C2
#         cs = torch.mean(v1 / v2)
#
#         return ssim_map.mean(), cs
#
#     def compute_msssim(self, img1, img2):
#         """Computes the multi scale structural similarity index between two images. This function is differentiable.
#         Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
#
#         :param img1: image Tensor BxCxHxW
#         :param img2: image Tensor BxCxHxW
#         :returns: mean SSIM
#         :rtype: float
#
#         """
#         if img1.shape[2] != img2.shape[2]:
#             img1 = img1.transpose(2, 3)
#
#         if img1.shape != img2.shape:
#             raise RuntimeError('Input images must have the same shape (%s vs. %s).',
#                                img1.shape, img2.shape)
#         if img1.ndim != 4:
#             raise RuntimeError('Input images must have four dimensions, not %d',
#                                img1.ndim)
#
#         device = img1.device
#         weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
#         levels = weights.size()[0]
#         ssims = []
#         mcs = []
#         for _ in range(levels):
#             ssim, cs = self.compute_ssim(img1, img2)
#
#             # Relu normalize (not compliant with original definition)
#             ssims.append(ssim)
#             mcs.append(cs)
#
#             img1 = F.avg_pool2d(img1, (2, 2))
#             img2 = F.avg_pool2d(img2, (2, 2))
#
#         ssims = torch.stack(ssims)
#         mcs = torch.stack(mcs)
#
#         # Simple normalize (not compliant with original definition)
#         # TODO: remove support for normalize == True (kept for backward support)
#         ssims = (ssims + 1) / 2
#         mcs = (mcs + 1) / 2
#
#         pow1 = mcs ** weights
#         pow2 = ssims ** weights
#
#         output = torch.prod(pow1[:-1] * pow2[-1])
#         return output
#
#     def forward(self, predicted_img_batch, target_img_batch, gradient_regulariser=None):
#         """Forward function for the CURL loss
#
#         :param predicted_img_batch:
#         :param gradient_regulariser:
#         :param target_img_batch: Tensor of shape BxCxWxH
#         :returns: value of loss function
#         :rtype: float
#
#         """
#         device = predicted_img_batch.device
#         num_images = target_img_batch.shape[0]
#         target_img_batch = target_img_batch
#
#         ssim_loss_value = torch.zeros(1, 1).to(device)
#         l1_loss_value = torch.zeros(1, 1).to(device)
#         cosine_rgb_loss_value = torch.zeros(1, 1).to(device)
#         hsv_loss_value = torch.zeros(1, 1).to(device)
#         rgb_loss_value = torch.zeros(1, 1).to(device)
#
#         predicted_img_lab = torch.clamp(ImageProcessing.rgb_to_lab(predicted_img_batch, device=device), 0, 1)
#         target_img_lab = torch.clamp(ImageProcessing.rgb_to_lab(target_img_batch, device=device), 0, 1)
#
#         # target_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(target_img_batch, device=device), 0, 1)
#         # predicted_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(predicted_img_batch, device=device), 0, 1)
#         target_img_hsv = ImageProcessing.rgb_to_hsv_new(target_img_batch, device=device)
#         predicted_img_hsv = ImageProcessing.rgb_to_hsv_new(predicted_img_batch, device=device)
#
#         predicted_img_hue = (predicted_img_hsv[:, 0, :, :] * 2 * math.pi)
#         predicted_img_val = predicted_img_hsv[:, 2, :, :]
#         predicted_img_sat = predicted_img_hsv[:, 1, :, :]
#         target_img_hue = (target_img_hsv[:, 0, :, :] * 2 * math.pi)
#         target_img_val = target_img_hsv[:, 2, :, :]
#         target_img_sat = target_img_hsv[:, 1, :, :]
#
#         target_img_L_ssim = target_img_lab[:, 0:1, :, :]
#         predicted_img_L_ssim = predicted_img_lab[:, 0:1, :, :]
#
#         ssim_value = self.compute_msssim(predicted_img_L_ssim, target_img_L_ssim)
#
#         ssim_loss_value += (1.0 - ssim_value)
#
#         predicted_img_1 = predicted_img_val * predicted_img_sat * torch.cos(predicted_img_hue)
#         predicted_img_2 = predicted_img_val * predicted_img_sat * torch.sin(predicted_img_hue)
#
#         target_img_1 = target_img_val * target_img_sat * torch.cos(target_img_hue)
#         target_img_2 = target_img_val * target_img_sat * torch.sin(target_img_hue)
#
#         predicted_img_hsv = torch.stack((predicted_img_1, predicted_img_2, predicted_img_val), 3)
#         target_img_hsv = torch.stack((target_img_1, target_img_2, target_img_val), 3)
#
#         l1_loss_value += F.smooth_l1_loss(predicted_img_lab, target_img_lab)
#         rgb_loss_value += F.smooth_l1_loss(predicted_img_batch, target_img_batch)
#         hsv_loss_value += F.smooth_l1_loss(predicted_img_hsv, target_img_hsv)
#
#         cosine_rgb_loss_value += (1 - torch.mean(
#             torch.nn.functional.cosine_similarity(predicted_img_batch, target_img_batch, dim=1)))
#
#         ssim_loss_value = ssim_loss_value / num_images
#         cosine_rgb_loss_value = cosine_rgb_loss_value / num_images
#
#         # curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
#         #              hsv_loss_value + 10*ssim_loss_value + torch.mean(1e-6*gradient_regulariser))/6
#
#         if not torch.is_tensor(gradient_regulariser):
#             curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
#                          hsv_loss_value + 10 * ssim_loss_value) / 6
#             return curl_loss
#         else:
#             curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
#                          hsv_loss_value + 10 * ssim_loss_value + torch.mean(1e-6 * gradient_regulariser)) / 6
#             # return rgb_loss_value,cosine_rgb_loss_value,l1_loss_value,hsv_loss_value,10*ssim_loss_value,
#             # torch.mean(1e-6*gradient_regulariser)
#
#         return curl_loss


# class ImageProcessing(object):
#
#     @staticmethod
#     def rgb_to_lab(img, device):
#         img = img.to(device)
#         img = img.permute(0, 3, 2, 1)
#         shape = img.shape
#         img = img.contiguous()
#         img = img.view(img.shape[0], -1, 3)
#         img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
#                                                                        min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(
#             0.04045).float()
#
#         rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
#             [0.412453, 0.212671, 0.019334],  # R
#             [0.357580, 0.715160, 0.119193],  # G
#             [0.180423, 0.072169,
#              0.950227],  # B
#         ]), requires_grad=False).to(device)
#
#         img = torch.matmul(img, rgb_to_xyz)
#         img = torch.mul(img, Variable(torch.FloatTensor(
#             [1 / 0.950456, 1.0, 1 / 1.088754]), requires_grad=False).to(device))
#
#         epsilon = 6 / 29
#
#         img = ((img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float()) + \
#               (torch.clamp(img, min=0.0001) **
#                (1.0 / 3.0) * img.gt(epsilon ** 3).float())
#
#         fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0, 500.0, 0.0],  # fx
#                                                     # fy
#                                                     [116.0, -500.0, 200.0],
#                                                     # fz
#                                                     [0.0, 0.0, -200.0],
#                                                     ]), requires_grad=False).to(device)
#
#         img = torch.matmul(img, fxfyfz_to_lab) + Variable(
#             torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).to(device)
#
#         img = img.view(shape)
#         img = img.permute(0, 3, 2, 1)
#
#         img[:, 0, :, :] = img[:, 0, :, :] / 100
#         img[:, 1, :, :] = (img[:, 1, :, :] / 110 + 1) / 2
#         img[:, 2, :, :] = (img[:, 2, :, :] / 110 + 1) / 2
#
#         img[(img != img).detach()] = 0
#
#         img = img.contiguous()
#         return img.to(device)
#
#     @staticmethod
#     def lab_to_rgb(img, device):
#         img = img.permute(0, 3, 2, 1)
#         shape = img.shape
#         img = img.contiguous()
#         img = img.view(img.shape[0], -1, 3)
#         img_copy = img.clone()
#
#         img_copy[:, :, 0] = img[:, :, 0] * 100
#         img_copy[:, :, 1] = ((img[:, :, 1] * 2) - 1) * 110
#         img_copy[:, :, 2] = ((img[:, :, 2] * 2) - 1) * 110
#
#         img = img_copy.clone().to(device)
#         del img_copy
#
#         lab_to_fxfyfz = Variable(torch.FloatTensor([  # X Y Z
#             [1 / 116.0, 1 / 116.0, 1 / 116.0],  # R
#             [1 / 500.0, 0, 0],  # G
#             [0, 0, -1 / 200.0],  # B
#         ]), requires_grad=False).to(device)
#
#         img = torch.matmul(img + Variable(torch.FloatTensor([16.0, 0.0, 0.0])).to(device), lab_to_fxfyfz)
#
#         epsilon = 6.0 / 29.0
#
#         img = (((3.0 * epsilon ** 2 * (img - 4.0 / 29.0)) * img.le(epsilon).float()) +
#                ((torch.clamp(img, min=0.0001) ** 3.0) * img.gt(epsilon).float()))
#
#         # denormalize for D65 white point
#         img = torch.mul(img, Variable(
#             torch.FloatTensor([0.950456, 1.0, 1.088754])).to(device))
#
#         xyz_to_rgb = Variable(torch.FloatTensor([  # X Y Z
#             [3.2404542, -0.9692660, 0.0556434],  # R
#             [-1.5371385, 1.8760108, -0.2040259],  # G
#             [-0.4985314, 0.0415560, 1.0572252],  # B
#         ]), requires_grad=False).to(device)
#
#         img = torch.matmul(img, xyz_to_rgb)
#
#         img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
#                                                                         min=0.0001) ** (
#                                                                     1 / 2.4) * 1.055) - 0.055) * img.gt(
#             0.0031308).float()
#
#         img = img.view(shape)
#         img = img.permute(0, 3, 2, 1)
#
#         img = img.contiguous()
#         img[(img != img).detach()] = 0
#         return img
#
#     @staticmethod
#     def swapimdims_3HW_HW3(img):
#         """Move the image channels to the first dimension of the numpy
#         multi-dimensional array
#
#         :param img: numpy nd array representing the image
#         :returns: numpy nd array with permuted axes
#         :rtype: numpy nd array
#
#         """
#         if img.ndim == 3:
#             return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
#         elif img.ndim == 4:
#             return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)
#
#     @staticmethod
#     def swapimdims_HW3_3HW(img):
#         """Move the image channels to the last dimensiion of the numpy
#         multi-dimensional array
#
#         :param img: numpy nd array representing the image
#         :returns: numpy nd array with permuted axes
#         :rtype: numpy nd array
#
#         """
#         if img.ndim == 3:
#             return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
#         elif img.ndim == 4:
#             return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)
#
#     @staticmethod
#     def load_image(img_filepath, normaliser, mode, t_size):
#         """Loads an image from file as a numpy multi-dimensional array
#
#         :param img_filepath: filepath to the image
#         :returns: image as a multi-dimensional numpy array
#         :rtype: multi-dimensional numpy array
#
#         """
#         myimage = np.array(Image.open(img_filepath))
#
#         if mode == 'True':
#             w_input = t_size[0]
#             h_input = t_size[1]
#             myimage = np.array(cv2.resize(myimage, (w_input, h_input)))
#
#         img = ImageProcessing.normalise_image(myimage, normaliser)  # NB: imread normalises to 0-1
#         return img
#
#     @staticmethod
#     def normalise_image(img, normaliser):
#         """Normalises image data to be a float between 0 and 1
#
#         :param img: Image as a numpy multi-dimensional image array
#         :returns: Normalised image as a numpy multi-dimensional image array
#         :rtype: Numpy array
#
#         """
#         img = img.astype('float32') / normaliser
#         return img
#
#     @staticmethod
#     def compute_mse(original, result):
#         """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.
#
#         :param original: input RGB image as a numpy array
#         :param result: target RGB image as a numpy array
#         :returns: the mean squared error between the input and target images
#         :rtype: float
#
#         """
#         return ((original - result) ** 2).mean()
#
#     @staticmethod
#     def compute_psnr(image_batchA, image_batchB, max_intensity):
#         """Computes the PSNR for a batch of input and output images
#
#         :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
#         :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
#         :param max_intensity: maximum intensity possible in the image (e.g. 255)
#         :returns: average PSNR for the batch of images
#         :rtype: float
#
#         """
#         num_images = image_batchA.shape[0]
#         psnr_val = 0.0
#
#         for i in range(0, num_images):
#             imageA = image_batchA[i, 0:3, :, :]
#             imageB = image_batchB[i, 0:3, :, :]
#             imageB = np.maximum(0, np.minimum(imageB, max_intensity))
#             psnr_val += 10 * \
#                         np.log10(max_intensity ** 2 /
#                                  ImageProcessing.compute_mse(imageA, imageB))
#
#         return psnr_val / num_images
#
#     @staticmethod
#     def compute_ssim(image_batchA, image_batchB):
#         """Computes the SSIM for a batch of input and output images
#
#         :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
#         :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
#         :param max_intensity: maximum intensity possible in the image (e.g. 255)
#         :returns: average PSNR for the batch of images
#         :rtype: float
#
#         """
#         num_images = image_batchA.shape[0]
#         ssim_val = 0.0
#         for i in range(0, num_images):
#             imageA = ImageProcessing.swapimdims_3HW_HW3(
#                 image_batchA[i, 0:3, :, :])
#             imageB = ImageProcessing.swapimdims_3HW_HW3(
#                 image_batchB[i, 0:3, :, :])
#             ssim_val += ssim(imageA, imageB, data_range=imageA.max() - imageA.min(), multichannel=True,
#                              gaussian_weights=True, win_size=11)
#
#         return ssim_val / num_images
#
#     @staticmethod
#     def hsv_to_rgb(img):
#         """Converts a HSV image to RGB
#         PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
#         Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
#
#         :param img: HSV image
#         :returns: RGB image
#         :rtype: Tensor
#
#         """
#         img = torch.clamp(img, 0, 1)
#         img = img.permute(0, 3, 2, 1)
#
#         m1 = 0
#         m2 = (img[:, :, :, 2] * (1 - img[:, :, :, 1]) - img[:, :, :, 2]) / 60
#         m3 = 0
#         m4 = -1 * m2
#         m5 = 0
#
#         r = img[:, :, :, 2] + torch.clamp(img[:, :, :, 0] * 360 - 0, 0, 60) * m1 + torch.clamp(
#             img[:, :, :, 0] * 360 - 60, 0,
#             60) * m2 + torch.clamp(
#             img[:, :, :, 0] * 360 - 120, 0, 120) * m3 + torch.clamp(img[:, :, :, 0] * 360 - 240, 0,
#                                                                     60) * m4 + torch.clamp(
#             img[:, :, :, 0] * 360 - 300, 0, 60) * m5
#
#         m1 = (img[:, :, :, 2] - img[:, :, :, 2] * (1 - img[:, :, :, 1])) / 60
#         m2 = 0
#         m3 = -1 * m1
#         m4 = 0
#
#         g = img[:, :, :, 2] * (1 - img[:, :, :, 1]) + torch.clamp(img[:, :, :, 0] * 360 - 0, 0, 60) * m1 + torch.clamp(
#             img[:, :, :, 0] * 360 - 60,
#             0, 120) * m2 + torch.clamp(img[:, :, :, 0] * 360 - 180, 0, 60) * m3 + torch.clamp(
#             img[:, :, :, 0] * 360 - 240, 0,
#             120) * m4
#
#         m1 = 0
#         m2 = (img[:, :, :, 2] - img[:, :, :, 2] * (1 - img[:, :, :, 1])) / 60
#         m3 = 0
#         m4 = -1 * m2
#
#         b = img[:, :, :, 2] * (1 - img[:, :, :, 1]) + torch.clamp(img[:, :, :, 0] * 360 - 0, 0, 120) * m1 + torch.clamp(
#             img[:, :, :, 0] * 360 -
#             120, 0, 60) * m2 + torch.clamp(img[:, :, :, 0] * 360 - 180, 0, 120) * m3 + torch.clamp(
#             img[:, :, :, 0] * 360 - 300, 0, 60) * m4
#
#         img = torch.stack((r, g, b), 3)
#         img[(img != img).detach()] = 0
#
#         img = img.permute(0, 3, 2, 1)
#         img = img.contiguous()
#         img = torch.clamp(img, 0, 1)
#
#         return img
#
#     @staticmethod
#     def hsv_to_rgb_new(img, device):
#         assert (img.shape[1] == 3)
#
#         h, s, v = img[:, 0] * 360, img[:, 1], img[:, 2]
#         h_ = (h - torch.floor(h / 360) * 360) / 60
#         c = s * v
#         x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))
#
#         zero = torch.zeros_like(c)
#         y = torch.stack((
#             torch.stack((c, x, zero), dim=1),
#             torch.stack((x, c, zero), dim=1),
#             torch.stack((zero, c, x), dim=1),
#             torch.stack((zero, x, c), dim=1),
#             torch.stack((x, zero, c), dim=1),
#             torch.stack((c, zero, x), dim=1),
#         ), dim=0)
#
#         index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
#         rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0)
#         return rgb.to(device)
#
#     @staticmethod
#     def rgb_to_hsv(img, device):
#         img = img.to(device)
#         img = torch.clamp(img, 1e-9, 1)
#
#         img = img.permute(0, 3, 2, 1)
#         shape = img.shape
#
#         img = img.contiguous()
#         img = img.view(img.shape[0], -1, 3)
#
#         mx = torch.max(img, 2)[0]
#         mn = torch.min(img, 2)[0]
#
#         ones = Variable(torch.FloatTensor(
#             torch.ones((img.shape[0], img.shape[1])))).to(device)
#         zero = Variable(torch.FloatTensor(torch.zeros(shape[0:3]))).to(device)
#
#         img = img.view(shape)
#
#         ones1 = ones[:, 0:math.floor((ones.shape[1] / 2))]
#         ones2 = ones[:, math.floor(ones.shape[1] / 2):(ones.shape[1])]
#
#         mx1 = mx[:, 0:math.floor((ones.shape[1] / 2))]
#         mx2 = mx[:, math.floor(ones.shape[1] / 2):(ones.shape[1])]
#         mn1 = mn[:, 0:math.floor((ones.shape[1] / 2))]
#         mn2 = mn[:, math.floor(ones.shape[1] / 2):(ones.shape[1])]
#
#         df1 = torch.add(mx1, torch.mul(ones1 * -1, mn1))
#         df2 = torch.add(mx2, torch.mul(ones2 * -1, mn2))
#
#         df = torch.cat((df1, df2), 0)
#         del df1, df2
#         df = df.view(shape[0:3]) + 1e-10
#         mx = mx.view(shape[0:3])
#
#         img = img.to(device)
#         df = df.to(device)
#         mx = mx.to(device)
#
#         g = img[:, :, :, 1].clone().to(device)
#         b = img[:, :, :, 2].clone().to(device)
#         r = img[:, :, :, 0].clone().to(device)
#
#         img_copy = img.clone()
#
#         img_copy[:, :, :, 0] = (((g - b) / df) * r.eq(mx).float() + (2.0 + (b - r) / df)
#                                 * g.eq(mx).float() + (4.0 + (r - g) / df) * b.eq(mx).float())
#         img_copy[:, :, :, 0] = img_copy[:, :, :, 0] * 60.0
#
#         zero = zero.to(device)
#         img_copy2 = img_copy.clone()
#
#         img_copy2[:, :, :, 0] = img_copy[:, :, :, 0].lt(zero).float(
#         ) * (img_copy[:, :, :, 0] + 360) + img_copy[:, :, :, 0].ge(zero).float() * (img_copy[:, :, :, 0])
#
#         img_copy2[:, :, :, 0] = img_copy2[:, :, :, 0] / 360
#
#         del img, r, g, b
#
#         img_copy2[:, :, :, 1] = mx.ne(zero).float() * (df / mx) + \
#                                 mx.eq(zero).float() * zero
#         img_copy2[:, :, :, 2] = mx
#
#         img_copy2[(img_copy2 != img_copy2).detach()] = 0
#
#         img = img_copy2.clone()
#
#         img = img.permute(0, 3, 2, 1)
#         img = torch.clamp(img, 1e-9, 1)
#
#         return img
#
#     @staticmethod
#     def rgb_to_hsv_new(img, device, epsilon=1e-10):
#         assert (img.shape[1] == 3)
#
#         r, g, b = img[:, 0], img[:, 1], img[:, 2]
#         max_rgb, argmax_rgb = img.max(1)
#         min_rgb, argmin_rgb = img.min(1)
#
#         max_min = max_rgb - min_rgb + epsilon
#
#         h1 = 60.0 * (g - r) / max_min + 60.0
#         h2 = 60.0 * (b - g) / max_min + 180.0
#         h3 = 60.0 * (r - b) / max_min + 300.0
#
#         h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0) / 360
#         s = max_min / (max_rgb + epsilon)
#         v = max_rgb
#
#         return torch.stack((h / 360, s, v), dim=1).to(device)
#
#     @staticmethod
#     def apply_curve(img, C, slope_sqr_diff, channel_in, channel_out, device):
#         """Applies a peicewise linear curve defined by a set of knot points to
#         an image channel
#
#         :param img: image to be adjusted
#         :param C: predicted knot points of curve
#         :returns: adjusted image
#         :rtype: Tensor
#
#         """
#         # img = img.unsqueeze(0)
#         # C = C.unsqueeze(0)
#         # slope_sqr_diff = slope_sqr_diff.unsqueeze(0)
#
#         curve_steps = C.shape[1] - 1
#
#         '''
#         Compute the slope of the line segments
#         '''
#         slope = C[:, 1:] - C[:, :-1]
#         slope_sqr_diff = slope_sqr_diff + torch.sum((slope[:, 1:] - slope[:, :-1]) ** 2, 1)[:, None]
#
#         r = img[:, None, :, :, channel_in].repeat(1, slope.shape[1] - 1, 1, 1) * curve_steps
#
#         s = torch.arange(slope.shape[1] - 1)[None, :, None, None].repeat(img.shape[0], 1, img.shape[1], img.shape[2])
#
#         r = r.to(device)
#         s = s.to(device)
#         r = r - s
#
#         sl = slope[:, :-1, None, None].repeat(1, 1, img.shape[1], img.shape[2]).to(device)
#         scl = torch.mul(sl, r)
#
#         sum_scl = torch.sum(scl, 1) + C[:, 0:1, None].repeat(1, img.shape[1], img.shape[2]).to(device)
#         img_copy = img.clone()
#
#         img_copy[:, :, :, channel_out] = img[:, :, :, channel_out] * sum_scl
#
#         img_copy = torch.clamp(img_copy, 0, 1)
#         return img_copy, slope_sqr_diff
#
#     @staticmethod
#     def adjust_hsv(img, S, device):
#         img = img.permute(0, 3, 2, 1)
#         shape = img.shape
#         img = img.contiguous()
#
#         S1 = torch.exp(S[:, 0:int(S.shape[1] / 4)])
#         S2 = torch.exp(S[:, (int(S.shape[1] / 4)):(int(S.shape[1] / 4) * 2)])
#         S3 = torch.exp(S[:, (int(S.shape[1] / 4) * 2):(int(S.shape[1] / 4) * 3)])
#         S4 = torch.exp(S[:, (int(S.shape[1] / 4) * 3):(int(S.shape[1] / 4) * 4)])
#
#         slope_sqr_diff = Variable(torch.zeros(img.shape[0], 1) * 0.0).to(device)
#
#         '''
#         Adjust Hue channel based on Hue using the predicted curve
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img, S1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)
#
#         '''
#         Adjust Saturation channel based on Hue using the predicted curve
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1, device=device)
#
#         '''
#         Adjust Saturation channel based on Saturation using the predicted curve
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1, device=device)
#
#         '''
#         Adjust Value channel based on Value using the predicted curve
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2, device=device)
#
#         img = img_copy.clone()
#         del img_copy
#
#         img[(img != img).detach()] = 0
#
#         img = img.permute(0, 3, 2, 1)
#         img = img.contiguous()
#         return img, slope_sqr_diff
#
#     @staticmethod
#     def adjust_rgb(img, R, device):
#         img = img.permute(0, 3, 2, 1)
#         shape = img.shape
#         img = img.contiguous()
#
#         '''
#         Extract the parameters of the three curves
#         '''
#         R1 = torch.exp(R[:, 0:int(R.shape[1] / 3)])
#         R2 = torch.exp(R[:, (int(R.shape[1] / 3)):(int(R.shape[1] / 3) * 2)])
#         R3 = torch.exp(R[:, (int(R.shape[1] / 3) * 2):(int(R.shape[1] / 3) * 3)])
#
#         '''
#         Apply the curve to the R channel
#         '''
#         slope_sqr_diff = Variable(torch.zeros(img.shape[0], 1) * 0.0).to(device)
#
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img, R1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)
#
#         '''
#         Apply the curve to the G channel
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1, device=device)
#
#         '''
#         Apply the curve to the B channel
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img_copy, R3, slope_sqr_diff, channel_in=2, channel_out=2, device=device)
#
#         img = img_copy.clone()
#         del img_copy
#
#         img[(img != img).detach()] = 0
#
#         img = img.permute(0, 3, 2, 1)
#         img = img.contiguous()
#         return img, slope_sqr_diff
#
#     @staticmethod
#     def adjust_lab(img, L, device):
#         img = img.permute(0, 3, 2, 1)
#
#         shape = img.shape
#         img = img.contiguous()
#
#         '''
#         Extract predicted parameters for each L,a,b curve
#         '''
#         L1 = torch.exp(L[:, 0:int(L.shape[1] / 3)])
#         L2 = torch.exp(L[:, (int(L.shape[1] / 3)):(int(L.shape[1] / 3) * 2)])
#         L3 = torch.exp(L[:, (int(L.shape[1] / 3) * 2):(int(L.shape[1] / 3) * 3)])
#
#         slope_sqr_diff = Variable(torch.zeros(img.shape[0], 1) * 0.0).to(device)
#
#         '''
#         Apply the curve to the L channel
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img, L1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)
#
#         '''
#         Now do the same for the a channel
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1, device=device)
#
#         '''
#         Now do the same for the b channel
#         '''
#         img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
#             img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2, device=device)
#
#         img = img_copy.clone()
#         del img_copy
#
#         img[(img != img).detach()] = 0
#
#         img = img.permute(0, 3, 2, 1)
#         img = img.contiguous()
#
#         return img, slope_sqr_diff

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, inp, tar):
        lab_inp, lab_tar = rgb_to_lab(inp), rgb_to_lab(tar)
        l1, a1, b1 = torch.moveaxis(lab_inp, 1, 0)[:3]
        l2, a2, b2 = torch.moveaxis(lab_tar, 1, 0)[:3]
        return torch.mean(torch.sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2))


if __name__ == '__main__':
    tensor1 = torch.randn(1, 3, 360, 540)
    tensor2 = torch.randn(1, 3, 360, 540)
    loss = ColorLoss()
    l = loss(tensor1, tensor2)
    im_orig = tensor1.squeeze(0).permute(1, 2, 0).numpy()
    im_edit = tensor2.squeeze(0).permute(1, 2, 0).numpy()
    from skimage import color
    lab_orig = color.rgb2lab(im_orig)
    lab_edit = color.rgb2lab(im_edit)
    de_diff = color.deltaE_cie76(lab_orig, lab_edit)
    print(np.mean(de_diff))
    print(l)
