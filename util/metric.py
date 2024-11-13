"""
Metric used in test_quantitive.py

"""
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import lpips
import torch
import math
from torch.nn import functional as torchF
from util.util import PARA_NOR

class calculate_all_metrics():
    def __init__(self):
        self.loss_lpips = lpips.LPIPS(net='alex').cuda()
    def calculate_lpips(self, X, Y):
        """
        Calculate lpips,
        Variables X, Y is a PyTorch Tensor/Variable with shape Nx3xHxW
        (N patches of size HxW, RGB images scaled in [-1,+1]).
        This returns d, a length N Tensor/Variable.
        """
        L = self.loss_lpips.forward(X*2-1, Y*2-1)  # change [0,+1] to [-1,+1]
        L = L.detach().mean()
        return L
    def run(self, visuals, metric_labels):
        all_results = {}
        for key in metric_labels:
            if key in ['Reflectance', 'Shading_ori', 'Shading_new', 'Relighted']:
                results = self.image_metric(visuals, key + '_gt', key + '_predict')
            elif key == 'Reconstruct':
                results = self.image_metric(visuals, 'Image_input', 'Reconstruct')
            elif key == 'Input_and_relighted_gt':
                results = self.image_metric(visuals, 'Image_input', 'Relighted_gt')
            elif key == 'light_position_color':
                pan_tilt_color_gt = inverse_normalize_pan_tilt_color(visuals['light_position_color_original'])
                pan_tilt_color_predict = inverse_normalize_pan_tilt_color(visuals['light_position_color_predict'])
                light_position_angle_error, pan_error, tilt_error = pan_tilt2angle(pan_tilt_color_predict[:, :2],
                                                                                   pan_tilt_color_gt[:, :2])
                light_color_angle_error = angular_distance(pan_tilt_color_predict[:, 2:5], pan_tilt_color_gt[:, 2:5])
                results = [light_position_angle_error, pan_error, tilt_error, light_color_angle_error]
            else:
                raise Exception("key error")
            all_results[key] = results
        return all_results

    def image_metric(self, tensors, gt_label, predict_label):
        # X: (N,3,H,W) a batch of non-negative RGB images (0~1)
        # Y: (N,3,H,W)
        ssim_value = ssim(tensors[gt_label], tensors[predict_label], data_range=1.0, size_average=True) # return (N,)
        mse_value = torchF.mse_loss(tensors[gt_label], tensors[predict_label])
        psnr_value = 10 * torch.log10(1.0 / mse_value)
        lpips_value = self.calculate_lpips(tensors[gt_label], tensors[predict_label])
        mps_value = (1 - lpips_value + ssim_value) / 2.0
        return [mse_value, ssim_value, psnr_value, lpips_value, mps_value]


def inverse_normalize_pan_tilt_color(data):
    """
    This function transforms the normalized light condition into the real value.
    The normalized data is [cos(pan)/2+0.5, sin(pan)/2+0.5, cos(tilt)/2+0.5, sin(tilt)/2+0.5, R/255, G/255, B/255]
    :param data: normalized light condition
    :return: pan(deg), tilt(deg), rgb(0~255)
    """
    # This function transforms the normalized light condition into the real value.
    # The normalized data is [cos(pan)/2+0.5, sin(pan)/2+0.5, cos(tilt), sin(tilt), R/255, G/255, B/255]
    data_angle = data[:, :4]
    data_angle[:, :2] = (data_angle[:, :2] - PARA_NOR['pan_b']) / PARA_NOR['pan_a']
    data_angle[:, 2:] = (data_angle[:, 2:] - PARA_NOR['tilt_b']) / PARA_NOR['tilt_a']
    data_rgb = data[:, 4:] * 255.0
    pan = torch.atan2(data_angle[:, 1], data_angle[:, 0]) * 180 / math.pi
    pan = pan.unsqueeze(1)
    tilt = torch.atan2(data_angle[:, 3], data_angle[:, 2]) * 180 / math.pi
    tilt = tilt.unsqueeze(1)
    pan_tilt_color = torch.cat((pan, tilt, data_rgb), dim=1)

    return pan_tilt_color


def pan_tilt2angle(pan_tilt_pred, pan_tilt_target):
    """
    :param pan_tilt_pred: prediction
    :param pan_tilt_target: target
    :return:
    """
    # inputs and targets should be pan and tilt.
    def pan_tilt_to_vector(pan, tilt):
        pan = pan / 180 * math.pi
        tilt = tilt / 180 * math.pi
        vector = torch.zeros(pan.size()[0], 3).cuda()
        vector[:, 0] = torch.mul(torch.sin(tilt), torch.cos(pan))
        vector[:, 1] = torch.mul(torch.sin(tilt), torch.sin(pan))
        vector[:, 2] = torch.cos(tilt)
        return vector

    vector_pred = pan_tilt_to_vector(pan_tilt_pred[:, 0], pan_tilt_pred[:, 1])
    vector_target = pan_tilt_to_vector(pan_tilt_target[:, 0], pan_tilt_target[:, 1])

    result = torch.sum(torch.mul(vector_pred, vector_target), dim=1)
    # pytorch acos occurs nan error.
    eps = 1e-6
    result = torch.clamp(result, min=-1 + eps, max=1 - eps)
    result = torch.mean(torch.acos(result))
    angle_result = result * 180 / math.pi
    # also return error of pan and tilt.
    pan_error = distance_angle(pan_tilt_pred[:, 0], pan_tilt_target[:, 0])
    tilt_error = distance_angle(pan_tilt_pred[:, 1], pan_tilt_target[:, 1])
    return angle_result, pan_error, tilt_error


def distance_angle(predict, target):
    """
    :param predict: angle in deg
    :param target: angle in deg
    :return: the angle between predict and target in deg
    """
    predict_rad = predict / (180 / math.pi)
    target_rad = target / (180 / math.pi)
    angle_cos = torch.cos(predict_rad) * torch.cos(target_rad) + torch.sin(predict_rad) * torch.sin(target_rad)
    # pytorch acos occurs nan error.
    eps = 1e-6
    angle_cos = torch.clamp(angle_cos, min=-1 + eps, max=1 - eps)
    result_rad = torch.mean(torch.acos(angle_cos))
    result_deg = result_rad * 180 / math.pi
    return result_deg


def angular_distance(color_predict, color_gt):
    result = torch.cosine_similarity(color_predict, color_gt, dim=1)
    #torch.sum(torch.mul(color_predict, color_gt), dim=1)
    # pytorch acos occurs nan error.
    eps = 1e-6
    result = torch.clamp(result, min=-1 + eps, max=1 - eps)
    result = torch.mean(torch.acos(result))
    angle_result = result * 180 / math.pi
    return angle_result

