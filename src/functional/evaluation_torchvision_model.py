import torch
import os
import numpy as np
from PIL import Image
from torchvision.transforms import transforms as torch_transforms
from torchvision.transforms.functional import crop
from torchvision.utils import save_image
import albumentations as A
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def adjustImg(img, patch_size=256):
	"""
	Adjust the width and height of fluorescence image by adding black region if necessary.
	This allows generation of 256x256 patches where all the regions of fluorescence image is included.
	"""
	w, h = img.size 	# w: width, h: h

	if w % patch_size == 0 and h % patch_size == 0: 	# If fluorescence image does not need modification, as its width and height are divisible by patch size.
		return img, 0, 0

	added_w, added_h = 0, 0
	if w % patch_size != 0:
		added_w = patch_size - (w % patch_size)
		w += added_w
	if h % patch_size != 0:
		added_h = patch_size - (h % patch_size)
		h += added_h

	new_img = Image.new(img.mode, (w, h), (0, 0, 0))
	new_img.paste(img)
	return new_img, added_w, added_h


def predict(model, device, fl_path, TTA, patch_size=256):
	"""
	Make prediction mask for given fluorescence image.
	"""

	normalize = torch_transforms.Normalize(mean=[0.1034, 0.0308, 0.0346],
										 std=[0.0932, 0.0273, 0.0302])
	transform = torch_transforms.Compose([torch_transforms.ToTensor(),
											 	normalize])

	fl = Image.open(fl_path).convert('RGB')

	adj_fl, added_w, added_h = adjustImg(fl, patch_size=patch_size)
	w, h = adj_fl.size

	pred = None

	for y in range(0, h, 256):
		pred_row = None
		for x in range(0, w, 256):
			fl_crop = crop(adj_fl, y, x, 256, 256)

			if TTA is False:
				pred_crop = ((model(transform(fl_crop).unsqueeze(0).to(device, dtype=torch.float32)))['out'] > 0.5).float().squeeze(0)
			else:
				pred_crops = [((model(transform(fl_crop).unsqueeze(0).to(device, dtype=torch.float32)))['out'] > 0.5).float().squeeze(0)]

				for albu_type in transformation:
					aug_transform = A.Compose([ albu_type ])
					fl_crop_transform = transform(Image.fromarray(aug_transform(image=fl_crop)['image'])).unsqueeze(0).to(device, dtype=torch.float32)
					pred_crops.append(model(fl_crop_transform)['out'].squeeze(0))
				pred_crop = ((sum(pred_crops) / (len(transformation) + 1)) > 0.5).float()

			if x + 256 == w and y + 256 == h:		# right-bottom corner
				pred_crop = pred_crop[:, :256 - added_h, :256 - added_w]
			elif x + 256 == w:		# right sides
				pred_crop = pred_crop[:, :, :256 - added_w]
					
			elif y + 256 == h:		# bottom sides
				pred_crop = pred_crop[:, :256 - added_h, :]

			if pred_row is None:
				pred_row = pred_crop
			else:	
				pred_row = torch.cat((pred_row, pred_crop), dim=2)

		if pred is None:
			pred = pred_row
		else:
			pred = torch.cat((pred, pred_row), dim=1)

	return pred.cpu()


def testset_evaluation(model, device, testset_path, weight, metrics, save2, write2, TTA):

	model.load_state_dict(torch.load(weight))
	model.to(device)
	model.eval()

	fl_img_names = sorted([fl for fl in os.listdir(testset_path) if fl != 'labels'])

	running_performances = np.array([0 for _ in range(len(metrics))], dtype='float64')
	running_confusion = np.array([0 for _ in range(4)], dtype='float64') 
	for fl_name in tqdm(fl_img_names, desc="Test set evaluation", leave=False):
		pred_mask = predict(model=model, device=device, fl_path=os.path.join(testset_path, fl_name), TTA=TTA)
		save_image((~(pred_mask.bool())).float(), os.path.join(save2, fl_name))

		gt_mask = torch_transforms.ToTensor()(Image.open(os.path.join(testset_path, 'labels', fl_name)).convert('L'))
		tn, fp, fn, tp = confusion_matrix((~(gt_mask.bool())).float().flatten().numpy(), pred_mask.flatten().numpy(), labels=[0, 1]).ravel()
		running_confusion += np.array([tp, fp, fn, tn])

		performance_scores = []
		for i, metric in enumerate(metrics):
			performance_score = metric(pred_mask, (~gt_mask.bool()).float()).item()
			performance_scores.append(performance_score)
		running_performances += np.array(performance_scores)
		write2.writerow([fl_name.split(sep='.')[0]] + performance_scores + [''] + [tp, fp, fn, tn])
	write2.writerow(["Mean"] + list(running_performances/len(fl_img_names)) + [''] + list(running_confusion/len(fl_img_names)))
	
	return list(running_performances/len(fl_img_names)), list(running_confusion/len(fl_img_names))
