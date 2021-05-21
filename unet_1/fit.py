import torch, os, code, csv
from torchvision.utils import save_image
from torchvision.transforms import transforms
from tqdm import tqdm


def train(model, device, cv_number, total_epoch, current_epoch, train_loader, criterion, optimizer, threshold=0.5):
	if os.path.exists(os.path.join(os.getcwd(), "train_loss_per_iter.csv")):
		open_mode = 'a'
	else:
		open_mode = 'wt'
	with open(os.path.join(os.getcwd(), "train_loss_per_iter.csv"), open_mode, newline='') as f:
		f_writer = csv.writer(f)
		if open_mode == 'wt':
			f_writer.writerow(['cv', 'epoch', 'iteration', 'train loss'])

		running_loss = 0
		n_imgs = 0

		for batch_idx, (fl_imgs, true_masks) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
			n_imgs += fl_imgs.size(0)
			fl_imgs, true_masks = fl_imgs.to(device, dtype=torch.float32), true_masks.to(device, dtype=torch.float32)

			optimizer.zero_grad()

			pred_masks = model(fl_imgs)
			batch_loss = 0
			for i, true_mask in enumerate(true_masks):
				loss = criterion(pred_masks[i], (~true_mask.bool()).float())
				running_loss += loss.item()
				batch_loss += loss

			loss = batch_loss / fl_imgs.size(0)
			f_writer.writerow([cv_number, current_epoch, batch_idx + 1, loss.item()])
			loss.backward()
			optimizer.step()

			if (batch_idx + 1) % 1000 == 0:
				tqdm.write("T: Epoch [{}/{}] Iteration [{}/{}] Train Loss [{}]".format(current_epoch, total_epoch,
																						batch_idx + 1, len(train_loader),
																						running_loss / n_imgs))
	return running_loss, n_imgs


def evaluate(model, device, total_epoch, current_epoch, val_loader, activation, criterion, metrics, threshold=0.5):
	running_performance = [0 for _ in range(len(metrics))]
	running_loss = 0
	n_imgs = 0

	for batch_idx, (fl_imgs, true_masks) in enumerate(tqdm(val_loader,
													desc="Evaluation [{}/{}]".format(current_epoch, total_epoch),
													leave=True)):
		n_imgs += fl_imgs.size(0)
		fl_imgs, true_masks = fl_imgs.to(device, dtype=torch.float32), true_masks.to(device, dtype=torch.float32)

		pred_masks = model(fl_imgs)

		for i, true_mask in enumerate(true_masks):
			loss = criterion(pred_masks[i], (~true_mask.bool()).float())
			running_loss += loss.item()

			for p, metric in enumerate(metrics):
				running_performance[p] += metric((activation(pred_masks[i]) > threshold).float(), (~true_mask.bool()).float()).item()

	return running_loss / n_imgs, [m / n_imgs for m in running_performance]


def train_model(model, device, total_epoch, train_loaders, val_loader, cv_number, criterion, optimizer, metrics, activation, threshold, writer):

	currentPath = os.path.abspath(os.getcwd())
	if not os.path.exists(os.path.join(currentPath, 'best_models')):
		os.mkdir(os.path.join(currentPath, 'best_models'))
	savePath = os.path.join(currentPath, 'best_models')

	train_losses, validation_losses, accuracies, recalls, precisions, fscores, ious = [], [], [], [], [], [], []

	min_validation_loss = 1e8
	best_epoch = 0

	for epoch in tqdm(range(total_epoch), desc="Epoch", leave=True):
		model.train()
		running_train_loss = 0.0
		total_imgs = 0		# total number of images in all train sets

		for dataset_idx, train_loader in enumerate(tqdm(train_loaders, desc="Training datasets", leave=True)):
			train_loss, n_imgs = train(	model=model, device=device, cv_number=cv_number, total_epoch=total_epoch, current_epoch=epoch + 1, train_loader=train_loader, 
										criterion=criterion, optimizer=optimizer 
										)
			running_train_loss += train_loss
			total_imgs += n_imgs
		train_losses.append(running_train_loss / total_imgs)
		tqdm.write("T Finished: Epoch [{}/{}] Train Loss [{}]".format(epoch + 1, total_epoch, train_losses[-1]))

		model.eval()
		validation_loss, performances = evaluate(	model=model, device=device, total_epoch=total_epoch, current_epoch=epoch + 1, val_loader=val_loader, 
													activation=activation, criterion=criterion, metrics=metrics, threshold=threshold
												)

		if validation_loss < min_validation_loss:
			torch.save(model.state_dict(), os.path.join(savePath, 'cv_{}_best_model.pth'.format(cv_number)))
			min_validation_loss = validation_loss
			best_epoch = epoch


		validation_losses.append(validation_loss)
		accuracies.append(performances[0])
		recalls.append(performances[1])
		precisions.append(performances[2])
		fscores.append(performances[3])
		ious.append(performances[4])

		tqdm.write("E: Epoch [{}/{}] Validation loss [{}] Accuracy [{}] Recall [{}]\nPrecision [{}] F1-score [{}] IoU [{}]".format(epoch + 1, total_epoch,
																																	validation_loss,
																																	accuracies[-1],
																																	recalls[-1],
																																	precisions[-1],
																																	fscores[-1],
																																	ious[-1]))
		writer.writerow([cv_number, epoch + 1, train_losses[-1], validation_loss] + performances)
	tqdm.write("Test set {} results:\n".format(cv_number))
	tqdm.write("Train losses: {}\nValidation losses: {}\nAccuracy: {}\nRecall: {}\nPrecision: {}\nF-score: {}\nIoU: {}\n".format(train_losses, validation_losses, accuracies,
																																recalls, precisions, fscores, ious)
																																)
	return best_epoch + 1, [validation_losses[best_epoch], accuracies[best_epoch], recalls[best_epoch], precisions[best_epoch], fscores[best_epoch], ious[best_epoch]]


# code.interact(local=dict(globals(), **locals()))