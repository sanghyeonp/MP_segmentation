import numpy as np
from PIL import Image
import code, os
from tqdm import tqdm
# FP: red
# FN: green (124, 252, 0)


def generateQualitativeImg(gt_path, pred_path, save2):
	gt = np.transpose(np.asarray(Image.open(gt_path).convert('RGB')), (2, 0, 1))
	pred = np.transpose(np.asarray(Image.open(pred_path).convert('RGB')), (2, 0, 1))
	qual_gt = np.copy(gt)
	for red_ch, gt_ch in enumerate(gt):
		for row_idx, gt_row in enumerate(gt_ch):
			pred_row = pred[red_ch][row_idx]
			dif = np.where(gt_row!=pred_row)[0]
			for val_idx in dif:
				gt_val = gt_row[val_idx]
				pred_val = pred_row[val_idx]
				if gt_val == 0 and pred_val == 255:	# GT is MP but Predicted as Background (FN)
					qual_gt[red_ch][row_idx][val_idx]=124
					qual_gt[1][row_idx][val_idx]=252
					qual_gt[2][row_idx][val_idx]=0
				elif gt_val == 255 and pred_val == 0:	# GT is background but predicted as MP (FP)
					qual_gt[red_ch][row_idx][val_idx]=255
					qual_gt[1][row_idx][val_idx]=0
					qual_gt[2][row_idx][val_idx]=0
		break

	qual_gt = np.transpose(qual_gt, (1, 2, 0))
	Image.fromarray(qual_gt).save(os.path.join(save2, gt_path.split(sep='/')[-1]))


save2 = os.path.join(os.getcwd(), 'qualitative')
if not os.path.exists(save2):
	os.mkdir(save2)

gt_img_paths = sorted([os.path.join("/home/sangp/mp_research/new_mp_dataset/dataset_1/labels", gt) for gt in os.listdir("/home/sangp/mp_research/new_mp_dataset/dataset_1/labels")])

pred_dirs = [p for p in os.listdir(os.getcwd()) if 'testset_evaluation' in p]
for pred_dir in tqdm(pred_dirs):
	name = pred_dir[19:]
	save22 = os.path.join(save2, name)
	if not os.path.exists(save22):
		os.mkdir(save22)
	pred_img_paths = sorted([os.path.join(os.getcwd(), pred_dir, p) for p in os.listdir(os.path.join(os.getcwd(), pred_dir))])
	for i, gt_img_path in enumerate(tqdm(gt_img_paths)):
		pred_img_path = pred_img_paths[i]
		generateQualitativeImg(gt_path=gt_img_path, pred_path=pred_img_path, save2=save22)
