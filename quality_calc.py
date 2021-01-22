import argparse
from PerceptualSimilarity import models
from PerceptualSimilarity.util import util
import cv2
from Niqe.niqe import calculate_niqe
from Ssim.ssim import calculate_ssim
import os
# from Brisque.brisque import calculate_brisque

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-q', '--quality', type=str, default='NR',
                    help='choose from 1)FR 2)NR')
parser.add_argument('-p0','--path0', type=str, default='./imgs/test/')
parser.add_argument('-p1','--path1', type=str, default='./imgs/test_jpeg/')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

if opt.quality == 'FR':
    ## Initializing the model
    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu,version=opt.version)

    img_path0 = opt.path0
    img_path1 = opt.path1

    names0 = os.listdir(img_path0)
    names0 = sorted(names0)
    names1 = os.listdir(img_path1)
    names1 = sorted(names1)
    print(names1, names0)
    nums = names0.__len__() - 1

    dist02 = 0
    dist01 = 0
    for i in range(nums):
        # Load images
        image_path = img_path0 + names0[i+1]
        img0 = util.load_image(image_path) 
        image_path = img_path1 + names1[i]
        img1 = util.load_image(image_path)

        dist02 += calculate_ssim(img0, img1, 255.0)

        img0 = util.im2tensor(img0)# RGB image from [-1,1]
        img1 = util.im2tensor(img1)

        if(opt.use_gpu):
            img0 = img0.cuda()
            img1 = img1.cuda()

        # Compute distance
        dist01 += model.forward(img0,img1)
    
    dist02 = dist02/nums
    dist01 = dist01/nums
    print('Distance: %.3f'%dist01)
    print('Ssim: %.3f'%dist02)

elif opt.quality == 'NR':
    img_path = opt.path0
    niqe_result = 0
    names = os.listdir(img_path)
    names = sorted(names)
    nums = names.__len__() - 1
    for image_name in names[1:]:
        image_path = img_path + image_name
        img = cv2.imread(image_path)
        niqe_result += calculate_niqe(img, 0, input_order='HWC', convert_to='y')
        print(niqe_result)
    # brisque_result = calculate_brisque(img)
    niqe_result = niqe_result/nums
    print('niqe: %.3f'%niqe_result)
    # print('brisque: %.3f'%brisque_result)

else:
    print("wrong input type!")
