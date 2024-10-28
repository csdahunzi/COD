import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.eufnet_final import Net
from utils.tdataloader import multi_test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=416, help='testing size')
parser.add_argument('--pth_path', type=str, default='checkpoints/dnmodel/BGNet-24.pth')

for _data_name in ['CAMO']:
    # data_path = '../Dataset/COD-TestDataset/{}/'.format(_data_name)
    data_path = '../Dataset/RGB+IR_Testset/{}/'.format(_data_name)
    save_path = './results/dntest/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Net()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path+'edge/', exist_ok=True)
    image_root = '{}/Imgs_rgb/'.format(data_path)
    infr_root = '{}/Imgs_ir/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = multi_test_dataset(image_root, image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, infr, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        infr = infr.cuda()

        # _, _, res, e = model(image)
        res = model(image,infr)
        res = res[0]
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imwrite(save_path+name, (res*255).astype(np.uint8))
        # e = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=True)
        # e = e.data.cpu().numpy().squeeze()
        # e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        # imageio.imwrite(save_path+'edge/'+name, (e*255).astype(np.uint8))
