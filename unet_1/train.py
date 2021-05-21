import sys
sys.path.append("/home/sangp/mp_research/segmentation_models")
import torch, os, code, csv
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess import Microplastic_data
from fit import evaluate, train_model
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import Accuracy, IoU, Recall, Fscore, Precision
from loss import DiceLoss, DiceBCELoss
from segmentation_models_pytorch.utils.base import Activation
import albumentations as A




#======================================================= INPUT ========================================================#
data_path = "/home/sangp/mp_research/new_mp_patches_dataset"

cuda_number = 0
batch_size = 10
total_epoch = 20
threshold = 0.5
testset_number = 1

# criterion = DiceLoss()
criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(9))

device = torch.device('cuda:{}'.format(cuda_number) if torch.cuda.is_available() else 'cpu')
metrics = [	Accuracy(balanced=True, threshold=None).to(device), Recall(threshold=None).to(device), 
			Precision(threshold=None).to(device), Fscore(threshold=None).to(device), IoU(threshold=None).to(device)]

# transformation = A.Compose([A.Flip(p=0.5),
# 							A.RandomBrightness(limit=[-0.2, 0.2], always_apply=False, p=0.5), 
# 							A.RandomContrast(limit=[0.2, 0.6], always_apply=False, p=0.5), 
# 							A.HueSaturationValue(hue_shift_limit=[-10, -10], sat_shift_limit=[50, 50], val_shift_limit=[10, 50], always_apply=False, p=0.5)
# 							], p=0.6)

transformation = None
#=======================================================================================================================#

activate_by = Activation(activation='sigmoid')

datasets = [Microplastic_data(os.path.join(data_path, dataset_name), transform=None) for dataset_name in sorted(os.listdir(data_path)) if int(dataset_name.split(sep='_')[-1]) != testset_number]

criterion.to(device)
activate_by.to(device)

crossValidation_result = []

training_csv = open(os.path.join(os.getcwd(), "training_process.csv"), 'wt', newline="")
training_csv_writer = csv.writer(training_csv)
training_csv_writer.writerow(['cv', 'epoch', 'training_loss', 'validation_loss', 'accuracy', 'recall', 'precision', 'fscore', 'iou'])

for dataset_idx, dataset in enumerate(tqdm(datasets, desc='Cross validation')):
	#======================================================= INPUT ========================================================#
	model_unet = smp.Unet(encoder_name="resnet101", in_channels=3, classes=1, encoder_weights="imagenet")
	# optimizer = torch.optim.Adam(model_unet.parameters(), lr=0.001)
	optimizer = torch.optim.SGD(model_unet.parameters(), momentum=0.9, lr=0.001)
	#=======================================================================================================================#
	model_unet.to(device)

	val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
	train_sets = datasets[:dataset_idx] + datasets[dataset_idx + 1:]
	train_loaders = [DataLoader(train_set, batch_size=batch_size, shuffle=True) for train_set in train_sets]
	
	# Find the performance before training the model
	validation_loss, val_performances = evaluate(	model=model_unet, device=device, total_epoch=total_epoch, current_epoch=0, val_loader=val_loader, 
													activation=activate_by, criterion=criterion, metrics=metrics, threshold=threshold
													)
	training_csv_writer.writerow([dataset_idx + 1, 0, 'NA', validation_loss] + val_performances)
	
	# Training and validation starts here.
	best_epoch, evaluations = train_model(	model=model_unet, device=device, total_epoch=total_epoch, 
											train_loaders=train_loaders, val_loader=val_loader, cv_number=dataset_idx + 1, 
											criterion=criterion, optimizer=optimizer, metrics=metrics, activation=activate_by, threshold=threshold, 
											writer=training_csv_writer
											)
	crossValidation_result.append((best_epoch, evaluations))


training_csv.close()

# printing cross-validation result
for testset_idx, result in enumerate(crossValidation_result):
	print("For cross-validation [{}], best model saved at epoch [{}] by validation loss :".format(testset_idx + 1, result[0]))
	print(result[1])
print("======================================Training Completed=================================")

# code.interact(local=dict(globals(), **locals()))