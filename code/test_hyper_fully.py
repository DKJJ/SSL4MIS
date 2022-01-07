import argparse
import os
import shutil
from PIL import Image

import h5py
# import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from PIL import Image

import cv2

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, net, test_save_path, FLAGS, patch_size=[256, 256]):
    image = np.load(FLAGS.root_path + "/data/" + case).astype('float32')
    image2 = image[10:40:10, :, :].copy()
    # image = image - np.mean(image, axis=(1, 2)).reshape((60, 1, 1))
    # image = image / np.maximum(np.std(image, axis=(1, 2)) / 255, 0.0001).reshape((60, 1, 1))
    image -= np.mean(image, axis=(1, 2), keepdims=True)
    image /= np.clip(np.std(image, axis=(1, 2), keepdims=True), 1e-6, 1e6)
    for band in range(image.shape[0]):
        image[band, :, :] = cv2.medianBlur(image[band, :, :], ksize=3)
    label = np.array(Image.open(FLAGS.root_path + "/label/" + case.replace('.npy', '_mask.png'))) / 255
    # h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    # image = h5f['image'][:]
    # label = h5f['label'][:]
    prediction = np.zeros_like(label)
    x, y = image.shape[1], image.shape[2]
    slice = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=0)
    input = torch.from_numpy(slice).unsqueeze(
        0).float().cuda()
    net.eval()
    with torch.no_grad():
        if FLAGS.model == "unet_urpc" or FLAGS.model == "unet_ds":
            out_main, _, _, _ = net(input)
        else:
            out_main = net(input)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        prediction = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    # second_metric = calculate_metric_percase(prediction == 2, label == 2)
    # third_metric = calculate_metric_percase(prediction == 3, label == 3)

    image2 = np.transpose(image2, (1, 2, 0)).astype(np.float64)
    image2 -= np.mean(image2, axis=(0, 1), keepdims=True)
    image2 /= np.std(image2, axis=(0, 1), keepdims=True)
    image2 *= 16
    image2 += 128
    img_itk = Image.fromarray(image2.astype(np.uint8))
    prd_itk = Image.fromarray(prediction.astype(np.uint8)*255)
    lab_itk = Image.fromarray(label.astype(np.uint8)*255)
    width, height = img_itk.size
    cat_itk = Image.new(img_itk.mode, (width * 3, height))
    cat_itk.paste(img_itk)
    cat_itk.paste(prd_itk, box=(width, 0))
    cat_itk.paste(lab_itk, box=(width*2, 0))
    cat_itk.save(test_save_path + case + ".png")
    # sitk.WriteImage(cat_itk, test_save_path + case + ".png")
    return first_metric#, second_metric, third_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/split_0_test.txt', 'r') as f:
        image_list = f.readlines()
        image_list = [item.replace('\n', '')
                            for item in image_list]
    # image_list = sorted([item.replace('\n', '').split(".")[0]
    #                      for item in image_list])
    snapshot_path = "../model/{}_{}_labeled/{}".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=60,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    save_mode_path = os.path.join(
        snapshot_path, 'iter_4000.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric = test_single_volume(
            case, net, test_save_path, FLAGS)
        # first_metric, second_metric, third_metric = test_single_volume(
        #     case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        # second_total += np.asarray(second_metric)
        # third_total += np.asarray(third_metric)
    avg_metric = [first_total / len(image_list)]
    # avg_metric = [first_total / len(image_list), second_total /
    #               len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    # print((metric[0]+metric[1]+metric[2])/3)
