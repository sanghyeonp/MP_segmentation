import numpy as np
from PIL import Image
import code, os
from tqdm import tqdm
from torchvision.transforms.functional import crop
# FP: red
# FN: green (124, 252, 0)

# np.asarray(crop(adjusted_fl_pil_image, y, x, 256, 256))

def cropping(img_path, save2):
	crop(Image.open(img_path).convert('RGB'), 1236, 690, 500, 500).save(os.path.join(save2, img_path.split(sep='/')[-1]))
	x = crop(Image.open(img_path).convert('RGB'), 1236, 690, 500, 500)
	crop(x, 88, 105, 38, 38).save(os.path.join(save2, img_path.split(sep='/')[-1].split(sep='.')[0] + "_part_1.png"))
	crop(x, 297, 72, 90, 90).save(os.path.join(save2, img_path.split(sep='/')[-1].split(sep='.')[0] + "_part_2.png"))
	crop(x, 181, 407, 78, 58).save(os.path.join(save2, img_path.split(sep='/')[-1].split(sep='.')[0] + "_part_3.png"))

save2 = os.path.join(os.getcwd(), 'qualitative_patch')
if not os.path.exists(save2):
	os.mkdir(save2)

# For number 94

pred_dirs = [p for p in os.listdir(os.path.join(os.getcwd(), 'qualitative'))]
for pred_dir in tqdm(pred_dirs):
	save22 = os.path.join(save2, pred_dir)
	if not os.path.exists(save22):
		os.mkdir(save22)
	
	cv_dirs = [os.path.join(os.getcwd(), 'qualitative', pred_dir, cv) for cv in os.listdir(os.path.join(os.getcwd(), 'qualitative', pred_dir))]
	for cv_dir in cv_dirs:
		save222 = os.path.join(save22, cv_dir.split(sep='/')[-1])
		if not os.path.exists(save222):
			os.mkdir(save222)

		pred_img_path = [os.path.join(cv_dir, p) for p in os.listdir(cv_dir) if '94' in p][0]
		cropping(img_path=pred_img_path, save2=save222)


# code.interact(local=dict(globals(), **locals()))
