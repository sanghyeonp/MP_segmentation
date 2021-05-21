import sys
sys.path.append("/home/sangp/mp_research/segmentation_models")
from segmentation_models_pytorch.utils.metrics import Accuracy, IoU, Recall, Fscore, Precision
import torch, os, csv, code
import torch.nn as nn
import segmentation_models_pytorch as smp
from PIL import Image
from preprocess import Microplastic_data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as torch_transforms
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import crop
from torchvision.utils import save_image
import albumentations as A
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# torch 0. is black

def adjustSize(pil_img, crop_size=256):
	"""
	return : PIL image
	"""
	width, height = pil_img.size

	if width % crop_size == 0 and height % crop_size == 0:
		return pil_img

	added_width = 0
	added_height = 0
	if width % crop_size != 0:
		added_width = crop_size - (width % crop_size)
		width = width + added_width
	if height % crop_size != 0:
		added_height = crop_size - (height % crop_size)
		height = height + added_height

	new_img = Image.new(pil_img.mode, (width, height), (0, 0, 0))
	new_img.paste(pil_img)
	return new_img, added_width, added_height


def predictByTTA(model, device, img_path, state_dict_path, transformation, mode):
	"""
	Produce a predictive mask corresponding to the given fl image with TTA implemented
	mode: 'mean', 'majority'
	"""
	model.load_state_dict(torch.load(state_dict_path))
	model.to(device)
	model.eval()

	normalize = torch_transforms.Normalize(mean=[0.1034, 0.0308, 0.0346],
										 std=[0.0932, 0.0273, 0.0302])
	transform = torch_transforms.Compose([torch_transforms.ToTensor(),
											 	normalize])

	fl = Image.open(img_path).convert('RGB')	# PIL Image
	adjusted_fl_pil_image, added_width, added_height = adjustSize(fl, crop_size=256)
	width, height = adjusted_fl_pil_image.size

	pred = None

	for y in range(0, height, 256):
		pred_row = None
		for x in range(0, width, 256):
			fl_crop = np.asarray(crop(adjusted_fl_pil_image, y, x, 256, 256))
			pred_crops = [((model(transform(fl_crop).unsqueeze(0).to(device, dtype=torch.float32))) > 0.5).float().squeeze(0)]

			for albu_type in transformation:
				aug_transform = A.Compose([ albu_type ])
				fl_crop_transform = transform(Image.fromarray(aug_transform(image=fl_crop)['image'])).unsqueeze(0).to(device, dtype=torch.float32)

				if mode == 'majority':
					pred_crops.append((model(fl_crop_transform) > 0.5).float().squeeze(0))
				elif mode == 'mean':
					pred_crops.append(model(fl_crop_transform).squeeze(0))

			if mode=='majority':
				pred_crop = ((sum(pred_crops) / (len(transformation) + 1)) >= 0.5).float()
			elif mode == 'mean':
				pred_crop = ((sum(pred_crops) / (len(transformation) + 1)) > 0.5).float()
				
			del pred_crops
			
			if x + 256 == width and y + 256 == height:		# right-bottom corner
				pred_crop = pred_crop[:, :256 - added_height, :256 - added_width]
			elif x + 256 == width:		# right sides
				pred_crop = pred_crop[:, :, :256 - added_width]
					
			elif y + 256 == height:		# bottom sides
				pred_crop = pred_crop[:, :256 - added_height, :]

			if pred_row is None:
				pred_row = pred_crop
			else:	
				pred_row = torch.cat((pred_row, pred_crop), dim=2)	# (aka: C * H * W)]

		if pred is None:
			pred = pred_row
		else:
			pred = torch.cat((pred, pred_row), dim=1)

	return pred.cpu()	# Model will make prediction for finding MP as 1.


def predict(model, device, img_path, state_dict_path):
	# Torch image : [channel, height, width]
	# np to pil : [width, height, channel] -> PIL Image
	"""
	1 fl image for 1 cv parameters
	Produce a predictive mask corresponding to the given fl image

	ReturnL: predicted image as tensor
	"""
	# Initialize the model with saved parameters
	model.load_state_dict(torch.load(state_dict_path))
	model.to(device)
	model.eval()

	normalize = torch_transforms.Normalize(mean=[0.1034, 0.0308, 0.0346],
										 std=[0.0932, 0.0273, 0.0302])
	transform = torch_transforms.Compose([torch_transforms.ToTensor(),
											 	normalize])


	fl = Image.open(img_path).convert('RGB')	# PIL Image

	adjusted_fl_pil_image, added_width, added_height = adjustSize(fl, crop_size=256)
	width, height = adjusted_fl_pil_image.size

	pred = None

	for y in range(0, height, 256):
		pred_row = None
		for x in range(0, width, 256):
			fl_crop = crop(adjusted_fl_pil_image, y, x, 256, 256)

			pred_crop = ((model(transform(fl_crop).unsqueeze(0).to(device, dtype=torch.float32))) > 0.5).float().squeeze(0)
			
			if x + 256 == width and y + 256 == height:		# right-bottom corner
				pred_crop = pred_crop[:, :256 - added_height, :256 - added_width]
			elif x + 256 == width:		# right sides
				pred_crop = pred_crop[:, :, :256 - added_width]
					
			elif y + 256 == height:		# bottom sides
				pred_crop = pred_crop[:, :256 - added_height, :]

			if pred_row is None:
				pred_row = pred_crop
			else:	
				pred_row = torch.cat((pred_row, pred_crop), dim=2)	# (aka: C * H * W)]

		if pred is None:
			pred = pred_row
		else:
			pred = torch.cat((pred, pred_row), dim=1)

	return pred.cpu() # Model will make prediction for finding MP as 1.


def evaluation(model, device, fl_path, gt_path, state_dict_path, metrics, transformation=None, mode=None):
	"""
	Evaluate one fl image
	"""
	if mode == 'mean':
		save2 = os.path.join(os.getcwd(), 'testset_evaluation_mean')
		if not os.path.exists(save2):
			os.mkdir(save2)
		save2 = os.path.join(os.getcwd(), 'testset_evaluation_mean', 'cv_' + state_dict_path.split(sep='/')[-1].split(sep='_')[1] + '_evaluation')
		if not os.path.exists(save2):
			os.mkdir(save2)
	elif mode == 'majority':
		save2 = os.path.join(os.getcwd(), 'testset_evaluation_majority')
		if not os.path.exists(save2):
			os.mkdir(save2)
		save2 = os.path.join(os.getcwd(), 'testset_evaluation_majority', 'cv_' + state_dict_path.split(sep='/')[-1].split(sep='_')[1] + '_evaluation')
		if not os.path.exists(save2):
			os.mkdir(save2)
	else:
		save2 = os.path.join(os.getcwd(), 'testset_evaluation')
		if not os.path.exists(save2):
			os.mkdir(save2)
		save2 = os.path.join(os.getcwd(), 'testset_evaluation', 'cv_' + state_dict_path.split(sep='/')[-1].split(sep='_')[1] + '_evaluation')
		if not os.path.exists(save2):
			os.mkdir(save2)

	if transformation is None and mode is None:
		pred_mask = predict(model=model, device=device, img_path=fl_path, state_dict_path=state_dict_path)
	else:
		pred_mask = predictByTTA(model=model, device=device, img_path=fl_path, state_dict_path=state_dict_path, transformation=transformation, mode=mode)
	
	save_image((~(pred_mask.bool())).float(), os.path.join(save2, 'cv_' + state_dict_path.split(sep='/')[-1].split(sep='_')[1] + '_' + fl_path.split(sep='/')[-1].split(sep='_')[0] + '_pred_mask.png'))
	true_mask = torch_transforms.ToTensor()(Image.open(gt_path).convert('L'))
	
	# Check how mask is noted
	# ALWAYS!!! When using confusion matrix, check if 0. is background and 1. is MP-representing pixel.
	"""
	# confusion_matrix(true, pred)
					TRUE
				0			1
	------------------------------
		 0	-	TN 			FN
	PRED 	-	
		 1	-	FP 			TP

	"""
	tn, fp, fn, tp = confusion_matrix((~(true_mask.bool())).float().flatten().numpy(), pred_mask.flatten().numpy(), labels=[0, 1]).ravel()

	performance_scores = []
	for i, metric in enumerate(metrics):
		performance_score = metric(pred_mask, (~true_mask.bool()).float()).item()
		performance_scores.append(performance_score)

	return state_dict_path.split(sep='/')[-1].split(sep='_')[1], fl_path.split(sep='/')[-1].split(sep='_')[0], performance_scores, [tp, fn, fp, tn]
	

def testset_evaluation(model, device, testset_path, state_dict_path, metrics, transformation=None, mode=None):
	"""
	Testset evaluation for one cv
	"""

	fl_imgs_path = sorted([os.path.join(testset_path, f) for f in os.listdir(testset_path) if os.path.isfile(os.path.join(testset_path, f))])
	true_masks_path = sorted([os.path.join(testset_path, 'labels', f) for f in os.listdir(os.path.join(testset_path, 'labels'))])
	running_performances = []	# Formated as ['k_fold', 'assigned_number', 'recall', 'precision', 'fscore', 'accuracy', 'iou']
	running_conf = []	# Formated as ['k_fold', 'assigned_number', 'tn', 'fp', 'fn', 'tp']

	for p, fl_img_path in enumerate(tqdm(fl_imgs_path)):
		k_fold, img_num, performance_scores, conf = evaluation(	model=model, device=device, fl_path=fl_img_path, 
																gt_path=true_masks_path[p], state_dict_path=state_dict_path, 
																metrics=metrics, transformation=transformation, mode=mode
																)

		running_performances.append([k_fold, img_num] + performance_scores)
		running_conf.append([k_fold, img_num] + conf)
		del k_fold; del img_num; del performance_scores; del conf

	return running_performances, running_conf


def eval4allcv(model, device, testset_path, state_dir_path, metrics, transformation=None, mode=None):
	# Initialize csv files
	if mode == 'mean':
		write2_performance = os.path.join(os.getcwd(), 'evaluation_performances_mean.csv')
		perf_csv = open(write2_performance, 'wt', newline="")
		perf_csv_writer = csv.writer(perf_csv)
		perf_csv_writer.writerow(['k_fold', 'assigned_number', 'accuracy', 'recall', 'precision', 'fscore', 'iou'])

		write2_conf = os.path.join(os.getcwd(), 'evaluation_conf_mean.csv')
		conf_csv = open(write2_conf, 'wt', newline="")
		conf_csv_writer = csv.writer(conf_csv)
		conf_csv_writer.writerow(['k_fold', 'assigned_number', 'tp', 'fn', 'fp', 'tn'])
	elif mode == 'majority':
		write2_performance = os.path.join(os.getcwd(), 'evaluation_performances_majority.csv')
		perf_csv = open(write2_performance, 'wt', newline="")
		perf_csv_writer = csv.writer(perf_csv)
		perf_csv_writer.writerow(['k_fold', 'assigned_number', 'accuracy', 'recall', 'precision', 'fscore', 'iou'])

		write2_conf = os.path.join(os.getcwd(), 'evaluation_conf_majority.csv')
		conf_csv = open(write2_conf, 'wt', newline="")
		conf_csv_writer = csv.writer(conf_csv)
		conf_csv_writer.writerow(['k_fold', 'assigned_number', 'tp', 'fn', 'fp', 'tn'])
	else:
		write2_performance = os.path.join(os.getcwd(), 'evaluation_performances.csv')
		perf_csv = open(write2_performance, 'wt', newline="")
		perf_csv_writer = csv.writer(perf_csv)
		perf_csv_writer.writerow(['k_fold', 'assigned_number', 'accuracy', 'recall', 'precision', 'fscore', 'iou'])

		write2_conf = os.path.join(os.getcwd(), 'evaluation_conf.csv')
		conf_csv = open(write2_conf, 'wt', newline="")
		conf_csv_writer = csv.writer(conf_csv)
		conf_csv_writer.writerow(['k_fold', 'assigned_number', 'tp', 'fn', 'fp', 'tn'])	

	state_dirs_path = sorted([os.path.join(state_dir_path, d) for d in os.listdir(state_dir_path)])
	for state_dict_path in tqdm(state_dirs_path):
		running_performances, running_conf = testset_evaluation(model=model, device=device, 
																testset_path=testset_path, state_dict_path=state_dict_path, 
																metrics=metrics, transformation=transformation, mode=mode
																)
		sum_performance = [0, 0, 0, 0, 0]
		sum_conf = [0, 0, 0, 0]
		for i, p in enumerate(running_performances):
			perf_csv_writer.writerow(p)
			conf_csv_writer.writerow(running_conf[i])
			for pos, score in enumerate(p[2:]):
				sum_performance[pos] += score
			for pos, count in enumerate(running_conf[i][2:]):
				sum_conf[pos] += count
			
		perf_csv_writer.writerow([running_performances[0][0], 'Mean'] + list(np.array(sum_performance) / len(running_performances)))
		conf_csv_writer.writerow([running_conf[0][0], 'Mean'] + list(np.array(sum_conf) / len(running_conf)))
		del running_performances; del running_conf

	perf_csv.close()
	conf_csv.close()


if __name__ == '__main__':
	#======================================================= INPUT ========================================================#

	testset_path = "/home/sangp/mp_research/new_mp_dataset/dataset_1"
	
	model = smp.Unet(encoder_name="resnet101", activation='sigmoid', in_channels=3, classes=1)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	state_dir_path = os.path.join(os.getcwd(), 'best_models')
	
	metrics = [	Accuracy(balanced=True, threshold=None).to(device), Recall(threshold=None).to(device), Precision(threshold=None).to(device), Fscore(threshold=None).to(device), 
				 IoU(threshold=None).to(device)]

	transformation = [	A.RandomBrightness(limit=[-0.2, 0.2], always_apply=True), 
						A.RandomContrast(limit=[0.2, 0.6], always_apply=True), 
						A.HueSaturationValue(hue_shift_limit=[-10, -10], sat_shift_limit=[50, 50], val_shift_limit=[10, 50], always_apply=True)
					]

	#=======================================================================================================================#
	eval4allcv(model=model, device=device, testset_path=testset_path, state_dir_path=state_dir_path, metrics=metrics, transformation=None, mode=None)
	eval4allcv(model=model, device=device, testset_path=testset_path, state_dir_path=state_dir_path, metrics=metrics, transformation=transformation, mode='mean')
	eval4allcv(model=model, device=device, testset_path=testset_path, state_dir_path=state_dir_path, metrics=metrics, transformation=transformation, mode='majority')

# code.interact(local=dict(globals(), **locals()))