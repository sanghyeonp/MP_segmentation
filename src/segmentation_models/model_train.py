# https://github.com/qubvel/segmentation_models.pytorch

"""
Loop over loader to obtain, 
x = (train image(s) tensor, mask image(s) tensor)
x[0].shape -> (batch sie, channels, size, size)fl and mask image
"""

import torch, os
import sys
sys.path.append("/home/sangp/mp_research")
# sys.path.append("/home/sangp/mp_research/segmentation/SegLoss/losses_pytorch")	# https://github.com/JunMa11/SegLoss
from preprocess import Microplastic_data
# from dice_loss import IoULoss, SoftDiceLoss
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics, losses
from torch.utils.data import DataLoader
from datetime import datetime
import code

# code.interact(local=dict(globals(), **locals()))


def train(	model, 
			train_loaders, 
			device,
			optimizer,
			criterion,
			current_epoch, 
			total_epoch,
			log):

	model.train()
	running_lose = 0.0
	
	total_train_loader = 0
	for train_loader in train_loaders:
		total_train_loader += len(train_loader)

	current_batch = 0
	total_imgs = 0
	for loader_n, train_loader in enumerate(train_loaders):
		for batch_idx, (imgs, true_masks) in enumerate(train_loader):
			current_batch += 1

			if device != 'cpu':
				imgs, true_masks = imgs.to(device, dtype=torch.float32), true_masks.to(device, dtype=torch.long)
			
			total_imgs += imgs.size(0)

			optimizer.zero_grad()
			pred_mask = model(imgs)
			
			loss = criterion(pred_mask, true_masks)
			# code.interact(local=dict(globals(), **locals()))
			running_lose += loss.item()

			loss.backward()
			optimizer.step()

			if current_batch % log == 0:
				print("Epoch [{}/{}], Train set [{}/{}], Iteration [{}/{}] [{}/{}], Loss [{}]".format(current_epoch, total_epoch, 
																										loader_n + 1, len(train_loaders),
																										batch_idx + 1, len(train_loader), 
																										current_batch, total_train_loader,
																										running_lose/total_imgs))
	return running_lose/total_train_loader


def evaluate(model, 
			test_loader, 
			device,
			criterion,
			min_validation,
			current_epoch, 
			total_epoch,
			log):
	model.eval()

	total_imgs = 0
	if isinstance(criterion, tuple):
		running_lose = [0.0 for _ in range(len(criterion))]
	running_lose = 0.0

	for batch_n, (imgs, true_masks) in enumerate(test_loader):
		if device != 'cpu':
			imgs, true_masks = imgs.to(device, dtype=torch.float32), true_masks.to(device, dtype=torch.long)
		
		total_imgs += imgs.size(0)

		pred_masks = model(imgs)
		if isinstance(criterion, tuple):
			for i in range(len(running_lose)):
				running_lose[i] += criterion[i](pred_masks, true_masks).item()
		else:
			loss = criterion(pred_masks, true_masks)
			running_lose += loss.item()

		if (batch_n + 1) % log == 0:
			if isinstance(criterion, tuple):
				print("Epoch [{}/{}], Iteration [{}/{}], Loss [{}]".format(current_epoch + 1, total_epoch, 
																			(batch_n + 1), len(test_loader),
																			[loss / total_imgs for loss in running_lose]))
			else:
				print("Epoch [{}/{}], Iteration [{}/{}], Loss [{}]".format(current_epoch + 1, total_epoch, 
																			(batch_n + 1), len(test_loader),
																			loss / total_imgs))
	if not isinstance(criterion, tuple):
		return loss / len(test_loader)
	return [loss / total_imgs for loss in running_lose]


if __name__ == "__main__":
	try:
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		dataset_path = "/home/sangp/mp_research/mp_dataset/"
		
		total_k = 3	# k-fold cross validation
		batch_size = 10
		total_epoch = 3
		log = 500
		min_validation = 100000

		datasets = sorted(os.listdir(dataset_path))
		start_time = datetime.now()
		print("Started training -> {}".format(start_time))
		train_losses = []
		valid_losses = []
		for test_n in range(total_k):
			model = smp.Unet("resnet101", in_channels=3, classes=1, encoder_weights="imagenet")
			if device != 'cpu':
				model.to(device)

			optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.0001)
			criterion = losses.DiceLoss().to(device)

			test_set = Microplastic_data(path=os.path.join(dataset_path, datasets[test_n]), transform=None)
			test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
			
			train_loaders = []

			for train_n in range(total_k):
				if not train_n == test_n:
					train_set = Microplastic_data(path=os.path.join(dataset_path, datasets[train_n]), transform=None)
					train_loaders.append(DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True))

			x = []
			y = []
			for epoch in range(total_epoch):
				start_epoch = datetime.now()
				print("Training...{}\nTest set -> dataset_{}\n".format(start_epoch, test_n + 1))
				train_loss = train(model=model, train_loaders=train_loaders, device=device, optimizer=optimizer, criterion=criterion, current_epoch=epoch + 1, total_epoch=total_epoch, log=log)
				x.append(train_loss)
				print("Test [Dataset {}] Epoch [{}/{}], Train loss [{}], Time taken [{}]".format(test_n + 1, epoch + 1, total_epoch, train_loss, datetime.now()-start_epoch))
				test_criterion = metrics.IoU().to(device)
				valid_loss = evaluate(model=model, test_loader=test_loader, device=device, criterion=(criterion, test_criterion), min_validation=min_validation, current_epoch=epoch, total_epoch=total_epoch, log=log)
				y.append(valid_loss)
				print("Test [Dataset {}] Epoch [{}/{}], IoU [{}], Time taken [{}]".format(test_n + 1, epoch + 1, total_epoch, valid_loss, datetime.now()-start_epoch))
			train_losses.append(x)
			valid_losses.append(y)

		print("Time taken: {}".format(datetime.now() - start_time))
		print("Train losses: {}".format(train_losses))
		print("Validation losses: {}".format(valid_losses))

	except KeyboardInterrupt:
		code.interact(local=dict(globals(), **locals()))
