import os
import torch
from tqdm import tqdm


def train(model, device, total_epoch, epoch, train_loader, criterion, optimizer, threshold=0.5):

	running_loss = 0
	n_imgs = 0

	for batch_idx, (fl_imgs, true_masks) in enumerate(tqdm(train_loader, desc="Training epoch [{}/{}]".format(epoch, total_epoch), leave=False)):
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
		loss.backward()
		optimizer.step()

	return running_loss / n_imgs


def evaluate(model, device, total_epoch, epoch, val_loader, activation, criterion, metrics, threshold=0.5):
	performances = [0 for _ in range(len(metrics))]
	running_loss = 0
	n_imgs = 0

	for batch_idx, (fl_imgs, true_masks) in enumerate(tqdm(val_loader,
													desc="Evaluation [{}/{}]".format(epoch, total_epoch),
													leave=False)):
		n_imgs += fl_imgs.size(0)
		fl_imgs, true_masks = fl_imgs.to(device, dtype=torch.float32), true_masks.to(device, dtype=torch.float32)

		pred_masks = model(fl_imgs)

		for i, true_mask in enumerate(true_masks):
			loss = criterion(pred_masks[i], (~true_mask.bool()).float())
			running_loss += loss.item()

			for p, metric in enumerate(metrics):
				performances[p] += metric((activation(pred_masks[i]) > threshold).float(), (~true_mask.bool()).float()).item()

	if epoch == 0:
		tqdm.write("Epoch [{}/{}] Val loss [{:.4f}] Accuracy [{:.4f}] Recall [{:.4f}] Precision [{:.4f}] F1-score [{:.4f}] IoU [{:.4f}]".format(epoch, total_epoch,
																																					running_loss / n_imgs,
																																					performances[0],
																																					performances[1],
																																					performances[2],
																																					performances[3],
																																					performances[4]))

	return running_loss / n_imgs, [m / n_imgs for m in performances]


def train_model(model, device, total_epoch, train_loader, val_loader, criterion, optimizer, metrics, activation, writer, filename, save2, threshold=0.5):
	info = {'train_losses':[], 'val_losses':[], 'accuracies':[], 'recalls':[], 'precisions':[], 'fscores':[], 'ious':[]}
	
	min_val_loss = 1e8
	best_epoch = 0

	for epoch in tqdm(range(total_epoch), desc="Epoch", leave=False):
		epoch += 1
		# Training the model
		model.train()
		train_loss = train(	model=model, device=device, total_epoch=total_epoch, epoch=epoch, 
							train_loader=train_loader, criterion=criterion, optimizer=optimizer, threshold=threshold)
		info['train_losses'].append(train_loss)

		# Evaluating the model
		model.eval()
		val_loss, performances = evaluate(	model=model, device=device, total_epoch=total_epoch, epoch=epoch, val_loader=val_loader, 
											activation=activation, criterion=criterion, metrics=metrics, threshold=threshold
											)

		if val_loss < min_val_loss:
			torch.save(model.state_dict(), os.path.join(save2, filename+'.pth'))
			min_val_loss = val_loss
			best_epoch = epoch

		info['val_losses'].append(val_loss)
		info['accuracies'].append(performances[0])
		info['recalls'].append(performances[1])
		info['precisions'].append(performances[2])
		info['fscores'].append(performances[3])
		info['ious'].append(performances[4])

		tqdm.write("Epoch [{}/{}] Train loss [{:.4f}] Val loss [{:.4f}] Accuracy [{:.4f}] Recall [{:.4f}] Precision [{:.4f}] F1-score [{:.4f}] IoU [{:.4f}]".format(epoch, total_epoch,
																																					train_loss,
																																					val_loss,
																																					performances[0],
																																					performances[1],
																																					performances[2],
																																					performances[3],
																																					performances[4]))


		
		writer.writerow([epoch, info['train_losses'][-1], val_loss] + performances)

	return best_epoch, [info['val_losses'][best_epoch - 1], info['accuracies'][best_epoch - 1], info['recalls'][best_epoch - 1], info['precisions'][best_epoch - 1], info['fscores'][best_epoch - 1], info['ious'][best_epoch - 1]]
