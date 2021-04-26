import argparse
import cv2
import os
import math
import shutil
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter

# This function is taken from https://github.com/WillBrennan/BlurDetection2
def estimate_blur(image, threshold=20):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur_map = cv2.Laplacian(image, cv2.CV_64F)
	score = np.var(blur_map)
	
	return blur_map, score, bool(score<threshold)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
class FacesDataset(Dataset):
	def __init__(self, image_locations, transforms=None):
		# image_locations is a list of path to each image ['path/to/image1.jpg', 'path/to/image2.jpg', ...]
		self.image_locations 	= image_locations
		self.transforms 		= transforms

	def __len__(self):
		return len(self.image_locations)

	def __getitem__(self, idx):
		
		if torch.is_tensor(idx):
			idx = idx.tolist()

		image = Image.open(self.image_locations[idx])
		label = torch.round(torch.rand(1)).long()
		# 1->blurred, 0->original
		if label==1:
			radius = torch.randint(low=3, high=7, size=(1,))
			random = torch.round(torch.rand(1)).long()
			if random==1:
				image_blurred = image.filter(ImageFilter.BoxBlur(radius=radius))
			else:
				image_blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
			
			image = self.transforms(image_blurred)
			#image_blurred = self.transforms(image_blurred)
			#random = torch.randint(low=0, high=2, size=image.shape).bool()
			#image[random] = image_blurred[random]
		else:
			image = self.transforms(image)
		
		
		return image, label

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.cnn = nn.Sequential(
					nn.Conv2d(3, 64, kernel_size=(3,3), padding=(1,1), stride=1),
					nn.ReLU(inplace=True),
					nn.MaxPool2d(kernel_size=3, stride=3),
					nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1), stride=1),
					nn.ReLU(inplace=True),
					nn.MaxPool2d(kernel_size=3, stride=3),
					nn.Conv2d(128, 256, kernel_size=(3,3), padding=(1,1), stride=1),
					nn.ReLU(inplace=True),
					nn.MaxPool2d(kernel_size=7, stride=7),
					)
		self.fc = nn.Sequential(
					nn.Linear(256*4*4, 1024),
					nn.ReLU(inplace=True),
					nn.Dropout(0.5),
					nn.Linear(1024, 2)
					)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
		    
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			if isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, x):
		out = self.cnn(x)
		out = out.view(x.shape[0], -1)
		out = self.fc(out)
		return out
		
def create_path_list(path, folders):
	files_list = []
	for folder in folders:
		p = path+folder+'/'
		files = os.listdir(p)
		files = [p+f for f in files]
		files_list.extend(files)
	return files_list

def train(model, epoch, train_loader, optimizer):
	model.train()
	losses = AverageMeter()
	acc = AverageMeter() 	
	for batch_idx, (data, label) in enumerate(train_loader):
		#pdb.set_trace()
		optimizer.zero_grad()
		data, label = data.to(device), label.to(device)
		label = label.squeeze(1)
		output = model(data)
		loss = F.cross_entropy(output, label)
		loss.backward()
		optimizer.step()
		pred = output.max(1, keepdim=True)[1]
		correct = pred.eq(label.data.view_as(pred)).cpu().sum()
		losses.update(loss.item(), data.shape[0])
		acc.update(correct.item()/data.shape[0], data.shape[0])
	print(f'Epoch: {epoch}')
	print(f'\t train-> loss: {losses.avg:.4f}, acc: {acc.avg:.4f}')

def test(model, epoch, test_loader, best_accuracy):
	model.eval()
	losses = AverageMeter()
	acc = AverageMeter()
	with torch.no_grad():
		for batch_idx, (data, label) in enumerate(test_loader):
			data, label = data.to(device), label.to(device)
			output = model(data)
			label = label.squeeze(1)
			loss = F.cross_entropy(output, label)
			pred = output.max(1, keepdim=True)[1]
			correct = pred.eq(label.data.view_as(pred)).cpu().sum()
			losses.update(loss.item(), data.shape[0])
			acc.update(correct.item()/data.shape[0], data.shape[0])
			
		if acc.avg>best_accuracy:
			best_accuracy = acc.avg
			state = {
					'state_dict': model.state_dict(),
					'accuracy': best_accuracy
			}
			torch.save(state, 'model.pt')
		print(f'\t test-> loss: {losses.avg:.4f}, acc: {acc.avg:.4f}, best: {best_accuracy:.4f}')
	return best_accuracy

def CNN():
	# Train on 3000 images from FFHQ dataset
	train_ids 	= ['00000', '01000', '02000']
	# Test the model on 1000 images from FFHQ dataset
	test_ids 	= ['03000']
	face_data_path = '/home/nano01/a/rathi2/Faces_Dataset/'
	train_locations = create_path_list(path=face_data_path, folders=train_ids)
	test_locations 	= create_path_list(path=face_data_path, folders=test_ids)
	transform = transforms.Compose([
					transforms.Resize((256, 256)),
					transforms.ToTensor()
					])

	train_dataset 	= FacesDataset(train_locations, transforms=transform)
	test_dataset 	= FacesDataset(test_locations, transforms=transform)
	train_loader 	= DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32)
	test_loader 	= DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=32) 
	epochs = 15
	model = Model()
	model.to(device)
	print(model)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.2, verbose=False)
	best_accuracy = 0.0
	for epoch in range(epochs):
		train(model, epoch, train_loader, optimizer)
		best_accuracy = test(model, epoch, test_loader, best_accuracy)
		scheduler.step()

def test_kanye_images_cnn(model, image):
	transform = transforms.Compose([
					transforms.Resize((256,256)),
					transforms.ToTensor()
					])
	with torch.no_grad():
		image = transform(image)
		image = image.unsqueeze(0)
		image = image.to(device)
		output = model(image)
		pred = output.max(1, keepdim=True)[1]
		# 1->blurred, 0->not blurred
		if pred==1:
			return True
		else:
			return False

def test_kanye_images_lap(image):
	blur_map, score, blurred = estimate_blur(image)
	return score, blurred

if __name__ == '__main__':
	
	torch.random.manual_seed(1234)
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	
	# Train the deep learning model
	CNN()
	# Load the trained model to perform evaluation
	model = Model()
	model.to(device)
	state = torch.load('model.pt', map_location='cpu')
	missing_keys, unexpected_keys = model.load_state_dict(state['state_dict'], strict=False)
	accuracy = state['accuracy']
	print(f'\n Missing keys : {missing_keys}, Unexpected Keys: {unexpected_keys}, Accuracy: {accuracy}')
	# Load test images
	kanye_image_files = os.listdir('./kw')
	folder_path = './kw/'
	# Create folders to store different result images
	cnn_blurred_images = './cnn_blurred_images/'
	cnn_clean_images = './cnn_clean_images/'
	lap_blurred_images = './lap_blurred_images/'
	lap_clean_images = './lap_clean_images/'
	
	# Clear the contents from previous runs
	try:
		shutil.rmtree(cnn_blurred_images)
		shutil.rmtree(cnn_clean_images)
		shutil.rmtree(lap_blurred_images)
		shutil.rmtree(lap_clean_images)
	except OSError:
		pass
	
	try:
		os.mkdir(cnn_blurred_images)
		os.mkdir(cnn_clean_images)
		os.mkdir(lap_blurred_images)
		os.mkdir(lap_clean_images)
	except OSError:
		pass
	
	scores = []
	for image_name in kanye_image_files:
		image = Image.open(folder_path+image_name)
		# Evaluate on deep learning model
		cnn_blurred = test_kanye_images_cnn(model, image)
		# Evaluate on variance of Laplacian
		image_bgr = cv2.imread(folder_path+image_name)
		score, lap_blurred = test_kanye_images_lap(image_bgr)
		# Store the scores to examine the distribution and decide the threshold manually
		scores.append(score)
		# Store the results in different folders
		plt.imshow(image)
		if cnn_blurred:
			plt.savefig(cnn_blurred_images+image_name)
		else:
			plt.savefig(cnn_clean_images+image_name)
		
		plt.text(10,50, s=str(int(score))+', '+str(lap_blurred), color='red', size='large')
		if lap_blurred:
			plt.savefig(lap_blurred_images+image_name)
		else:
			plt.savefig(lap_clean_images+image_name)
		plt.clf()
		
		if cnn_blurred!=lap_blurred:	
			print(f'{image_name} -> CNN: {cnn_blurred}, Lap: {lap_blurred}')
	
	# Analyze the results
	cnn_blurred = os.listdir(cnn_blurred_images)
	lap_blurred = os.listdir(lap_blurred_images)

	cnn_blurred = set(cnn_blurred)
	lap_blurred = set(lap_blurred)

	common = cnn_blurred.intersection(lap_blurred)
	cnn_unique = cnn_blurred - lap_blurred
	lap_unique = lap_blurred - cnn_blurred

	print(f'# Blurred images identified by CNN : {len(cnn_blurred)}')
	print(f'# Blurred images identified by Laplacian : {len(lap_blurred)}')
	print(f'# Common images between CNN and Laplacian : {len(common)}')
		
