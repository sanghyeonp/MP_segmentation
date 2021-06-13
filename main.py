# loss.py from https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# unet arch. and evaluation metrics from https://github.com/qubvel/segmentation_models.pytorch
# Accuracy in segmentation_models.segmentation_models_pytorch.utils.metrics.py was modified to calculate balanced accuracy
import code
import os
import csv
from datetime import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d
from torch.utils.data import DataLoader
import src.segmentation_models.segmentation_models_pytorch as smp
from torchvision.models.segmentation import fcn_resnet101, deeplabv3_resnet101
from src.pytorch_nested_unet.archs import NestedUNet
from src.functional.loss import DiceLoss, DiceBCELoss
from src.segmentation_models.segmentation_models_pytorch.utils.metrics import Accuracy, IoU, Recall, Fscore, Precision
from src.segmentation_models.segmentation_models_pytorch.utils.base import Activation
from src.functional.preprocess import Microplastic_data
from src.functional.fit_unet import evaluate as evaluate_unet
from src.functional.fit_unet import train_model as train_unet
from src.functional.fit_torchvision_model import evaluate as evaluate_torchvision
from src.functional.fit_torchvision_model import train_model as train_torchvision
from src.functional.evaluation_unet import testset_evaluation as testset_evaluation_unet
from src.functional.evaluation_torchvision_model import testset_evaluation as testset_evaluation_torchvision

def main(model, train=True, weights=None, test=True, optimizer='adam', momentum=0.9, lr=0.001, epoch=20, batch_size=10, criterion='dice', TTA=False):
	"""
	model_name : choose model from either 'unet', 'fcn', 'deeplabv3', 'nested_unet'
			these models will be pre-trained models
			'unet' will have ResNet101 as encoder backbone
			'fcn' and 'deeplabv3' will have their classification layer
	
	criterion : choose loss function from either 'bce', 'dice', 'dicebce'

	optimizer : choose optimization method from either 'adam', 'sgd'
	"""
	
	torch.manual_seed(0)
	model_name, loss_name, optim_name = model, criterion, optimizer

	if train:
		# Create saving location
		"""
		train log: saves losses and validation scores for every epochs
		saved model: saves the model with lowest validation loss
		"""
		time = datetime.now()
		t = time.strftime("%Y%m%d_%H%M%S")
		print("Initiating... Model [{}] Loss function [{}] Optimizer [{}]".format(model_name, loss_name, optim_name))

		if not os.path.exists(os.path.join(os.getcwd(), 'result')):
			os.mkdir(os.path.join(os.getcwd(), 'result'))
		if not os.path.exists(os.path.join(os.getcwd(), 'result', 'train_log')):
			os.mkdir(os.path.join(os.getcwd(), 'result', 'train_log'))
		if not os.path.exists(os.path.join(os.getcwd(), 'result', 'train_log', model_name)):
			os.mkdir(os.path.join(os.getcwd(), 'result', 'train_log', model_name))
		if not os.path.exists(os.path.join(os.getcwd(), 'result', 'saved_model')):
			os.mkdir(os.path.join(os.getcwd(), 'result', 'saved_model'))
		if not os.path.exists(os.path.join(os.getcwd(), 'result', 'saved_model', model_name)):
			os.mkdir(os.path.join(os.getcwd(), 'result', 'saved_model', model_name))
		
		TTA_is = False
		if TTA is not False:
			TTA_is = True
		
		filename = '{}_model[{}]_loss[{}]_optim[{}]_epoch[{}]_TTA[{}]'.format(t, model_name, criterion, optimizer, epoch, TTA_is)
		print("File name [{}]".format(filename))
		with open(os.path.join(os.getcwd(), 'result', 'train_log', model_name, filename+'.csv'), 'wt', newline='') as f1:
			f1_writer = csv.writer(f1)
			f1_writer.writerow(['Epoch', 'Train loss', 'Val loss', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU'])

			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			evaluation_metrics = [	Accuracy(balanced=True, threshold=None).to(device), Recall(threshold=None).to(device), 
									Precision(threshold=None).to(device), Fscore(threshold=None).to(device), IoU(threshold=None).to(device)]

			# Initialize model
			if model_name == 'unet':		# Initialize U-Net model
				model = smp.Unet(encoder_name="resnet101", in_channels=3, classes=1, encoder_weights="imagenet")
				evaluate, train_model = evaluate_unet, train_unet
			elif model_name == 'fcn':	# Initialize FCN model
				model = fcn_resnet101(pretrained=True)
				# The last convolutional layer is modified to produce a binary output.
				model.classifier._modules['4'] = Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
				model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
				evaluate, train_model = evaluate_torchvision, train_torchvision
			elif model_name == 'deeplabv3':	# Initialize Deeplabv3 model
				model = deeplabv3_resnet101(pretrained=True)
				# The last convolutional layer is modified to produce a binary output.
				model.classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
				model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
				evaluate, train_model = evaluate_torchvision, train_torchvision
			elif model_name == 'nested_unet':
				model = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False)
				evaluate, train_model = evaluate_unet, train_unet

			# Initialize optimizer
			if optimizer == 'adam':
				optimizer = torch.optim.Adam(model.parameters(), lr=lr)
			elif optimizer == 'sgd':
				optimizer = torch.optim.SGD(model.parameters(), momentum=momentum, lr=lr)

			# Initialize criterion
			if criterion == 'bce':
				criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(9))
			elif criterion == 'dice':
				criterion = DiceLoss()
			elif criterion == 'dicebce':
				criterion = DiceBCELoss()

			activation = Activation(activation='sigmoid') # U-Net architecture is not initialized with activation at the end

			model.to(device)
			criterion.to(device)
			activation.to(device)

			train_set = Microplastic_data(os.path.join(os.getcwd(), 'dataset', 'train'))
			train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
			val_set = Microplastic_data(os.path.join(os.getcwd(), 'dataset', 'validation'))
			val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

			print("Time: {}".format(time))
			print("Start training...\n")
			val_loss, val_performances = evaluate(	model=model, device=device, total_epoch=epoch, epoch=0, val_loader=val_loader, 
													activation=activation, criterion=criterion, metrics=evaluation_metrics
													)
			f1_writer.writerow([0, 'NA', val_loss] +  val_performances)

			best_epoch, evaluations = train_model(	model=model, device=device, total_epoch=epoch, 
													train_loader=train_loader, val_loader=val_loader, 
													criterion=criterion, optimizer=optimizer, metrics=evaluation_metrics, 
													activation=activation,	writer=f1_writer, filename=filename,
													save2=os.path.join(os.getcwd(), 'result', 'saved_model', model_name)
													)
			print("\nFinished training...")
			print("Time: {}\n".format(datetime.now()))
			print("Training result for [{}]:".format(filename))
			tqdm.write("Model saved at best epoch [{}]\nValidation loss [{:.4f}]\nAccuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\
						\nF1-score [{:.4f}]\nIoU [{:.4f}]\n".format(best_epoch, evaluations[0], evaluations[1], evaluations[2],
																	evaluations[3], evaluations[4], evaluations[5]
																	)
						)
		
		if test:	# Train 및 test 다 True 일 때는, training 할때 save된 parameter로 test 하기
			print("Evaluating...")
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred')):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'pred'))
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred', model_name)):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'pred', model_name))
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred', model_name, filename)):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'pred', model_name, filename))
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'testset_evaluation')):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'testset_evaluation'))
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'testset_evaluation', model_name)):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'testset_evaluation', model_name))

			if model_name == 'unet' or model_name == 'nested_unet':
				testset_evaluation = testset_evaluation_unet
			elif model_name == 'fcn' or model_name == 'deeplabv3':
				testset_evaluation = testset_evaluation_torchvision

			with open(os.path.join(os.getcwd(), 'result', 'testset_evaluation', model_name, filename+'.csv'), 'wt', newline='') as f2:
				f2_writer =csv.writer(f2)
				f2_writer.writerow(['Image number', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU', '', 'TP', 'FP', 'FN', 'TN'])
				performances, confusion = testset_evaluation(	model=model, device=device, testset_path=os.path.join(os.getcwd(), 'dataset', 'test'), 
																weight=os.path.join(os.getcwd(), 'result', 'saved_model', model_name, filename+'.pth'), metrics=evaluation_metrics, 
																save2=os.path.join(os.getcwd(), 'result', 'pred', model_name, filename), write2=f2_writer, TTA=TTA
																)
			print("Finished evaluation...\n")
			print("Evaluation result for [{}]:".format(filename))
			print("Accuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\nF1-score [{:.4f}]\nIoU [{:.4f}]\
					\nTP [{}] | FP [{}] | FN [{}] | TN [{}]\n".format(performances[0], performances[1], performances[2], performances[3], performances[4],
																	confusion[0], confusion[1], confusion[2], confusion[3]))

	else:	# If training is not being done. Load saved model.
		if test:
			if weights is None:
				raise AttributeError("Need to provide pre-trained weight if performing only testing.")
			
			t = datetime.now().strftime("%Y%m%d_%H%M%S")
			device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
			evaluation_metrics = [	Accuracy(balanced=True, threshold=None).to(device), Recall(threshold=None).to(device), 
									Precision(threshold=None).to(device), Fscore(threshold=None).to(device), IoU(threshold=None).to(device)]

			TTA_is = False
			if TTA is not False:
				TTA_is = True

			filename = '{}_model[{}]_pretrained[{}]_TTA[{}]'.format(t, model_name, weights.split(sep='/')[-1], TTA_is)

			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred')):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'pred'))
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred', model_name)):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'pred', model_name))
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'pred', model_name, filename)):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'pred', model_name, filename))
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'testset_evaluation')):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'testset_evaluation'))
			if not os.path.exists(os.path.join(os.getcwd(), 'result', 'testset_evaluation', model_name)):
				os.mkdir(os.path.join(os.getcwd(), 'result', 'testset_evaluation', model_name))

			if model_name == 'unet' or model_name == 'nested_unet':
				if model_name == 'unet':
					model = smp.Unet(encoder_name="resnet101", in_channels=3, classes=1, encoder_weights=None)
				elif model_name == 'nested_unet':
					model = NestedUNet(num_classes=1, input_channels=3, deep_supervision=False)
				testset_evaluation = testset_evaluation_unet
			elif model_name == 'fcn' or model_name == 'deeplabv3':
				if model_name == 'fcn':
					model = fcn_resnet101(pretrained=True)
					# The last convolutional layer is modified to produce a binary output.
					model.classifier._modules['4'] = Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
					model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
				elif model_name == 'deeplabv3':
					model = deeplabv3_resnet101(pretrained=True)
					# The last convolutional layer is modified to produce a binary output.
					model.classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
					model.aux_classifier._modules['4'] = Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
				testset_evaluation = testset_evaluation_torchvision

			print("Evaluating...")
			with open(os.path.join(os.getcwd(), 'result', 'testset_evaluation', model_name, filename+'.csv'), 'wt', newline='') as f2:
				f2_writer =csv.writer(f2)
				f2_writer.writerow(['Image number', 'Accuracy', 'Recall', 'Precision', 'F1-score', 'IoU', '', 'TP', 'FP', 'FN', 'TN'])
				performances, confusion = testset_evaluation(	model=model, device=device, testset_path=os.path.join(os.getcwd(), 'dataset', 'test'), 
																weight=os.path.join(weights), metrics=evaluation_metrics, 
																save2=os.path.join(os.getcwd(), 'result', 'pred', model_name, filename), write2=f2_writer, TTA=TTA
																)
			print("Finished evaluation...\n")
			print("Evaluation result for [{}]".format(filename))
			print("Accuracy [{:.4f}]\nRecall [{:.4f}]\nPrecision [{:.4f}]\nF1-score [{:.4f}]\nIoU [{:.4f}]\
					\nTP [{}] | FP [{}] | FN [{}] | TN [{}]\n".format(performances[0], performances[1], performances[2], performances[3], performances[4],
																	confusion[0], confusion[1], confusion[2], confusion[3]))


	# with open(os.path.join(os.getcwd(), 'train_log', ))


if __name__ == '__main__':

	# main(model='unet', train=True, weights=None, test=True, optimizer='sgd', momentum=0.9, lr=0.001, epoch=15, batch_size=10, criterion='dice', TTA=False)
	main(model='unet', train=False, weights="/home/sangp/bachelor_thesis/git_repo/microplastics/result/saved_model/unet/20210613_192714_model[unet]_loss[dice]_optim[sgd]_epoch[15]_TTA[False].pth", 
		test=True, optimizer='adam', momentum=0.9, lr=0.001, epoch=20, batch_size=10, criterion='dice', TTA=False)
	# main(model='fcn', epoch=5)
	# main(model='deeplabv3', epoch=5)
	# main(model='nested_unet', epoch=5)
	# code.interact(local=dict(globals(), **locals()))
