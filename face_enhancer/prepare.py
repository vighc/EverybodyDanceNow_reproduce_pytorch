import os
from pathlib import Path
import cv2
from tqdm import tqdm

face_sync_dir = Path('/OpenPose_Pose_transfer/data/face/ ')
face_sync_dir.mkdir(exist_ok=True)
test_sync_dir = Path('/OpenPose_Pose_transfer/data/face/test_sync/ ')
test_sync_dir.mkdir(exist_ok=True)
test_real_dir = Path('/OpenPose_Pose_transfer/data/face/test_real/ ')
test_real_dir.mkdir(exist_ok=True)
test_img = Path('/OpenPose_Pose_transfer/data/target/test_img/ ')
test_img.mkdir(exist_ok=True)
test_label = Path('/OpenPose_Pose_transfer/data/target/test_label/ ')
test_label.mkdir(exist_ok=True)

train_dir = '/OpenPose_Pose_transfer/data/target/train/train_img/'
label_dir = '/OpenPose_Pose_transfer/data/target/train/train_label/'

print('Prepare test_real....')
for img_idx in tqdm(range(len(os.listdir(train_dir)))):
    img = cv2.imread(train_dir+'{:05}.png'.format(img_idx))
    label = cv2.imread(label_dir+'{:05}.png'.format(img_idx))
    cv2.imwrite(str(test_real_dir)+'{:05}.png'.format(img_idx),img)
    cv2.imwrite(str(test_img)+'{:05}.png'.format(img_idx),img)
    cv2.imwrite(str(test_label)+'{:05}.png'.format(img_idx),label)

print('Prepare test_sync....')
import os
import torch
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import sys
pix2pixhd_dir = Path('/OpenPose_Pose_transfer/src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

from src.pix2pixHD.data.data_loader import CreateDataLoader
from src.pix2pixHD.models.models import create_model
import src.pix2pixHD.util.util as util
from src.pix2pixHD.util.visualizer import Visualizer
from src.pix2pixHD.util import html
import src.config.test_opt as opt
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
opt.checkpoints_dir = '/OpenPose_Pose_transfer/checkpoints/'
opt.dataroot='/OpenPose_Pose_transfer/data/target/'
opt.name='target'
opt.nThreads=0
opt.results_dir='/face_enhancer/prepare/'

iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

model = create_model(opt)

for data in tqdm(dataset):
    minibatch = 1
    generated = model.inference(data['label'], data['inst'])

    visuals = OrderedDict([('synthesized_image', util.tensor2im(generated.data[0]))])
    img_path = data['path']
    visualizer.save_images(webpage, visuals, img_path)
webpage.save()
torch.cuda.empty_cache()

print('Copy the synthesized images...')
synthesized_image_dir = './prepare/target/test_latest/images/'
for img_idx in tqdm(range(len(os.listdir(synthesized_image_dir)))):
    img = cv2.imread(synthesized_image_dir+' {:05}_synthesized_image.jpg'.format(img_idx))
    cv2.imwrite(str(test_sync_dir) + '{:05}.png'.format(img_idx), img)


