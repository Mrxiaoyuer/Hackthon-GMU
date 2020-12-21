import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



class CustomDataset(Dataset):
	"""Custom trajectories dataset."""

	def __init__(self, root_dir, transform=None):

		datapath = root_dir[0]
		labelpath = root_dir[1]

		self.transform = transform
		self.data = torch.utils.data.TensorDataset(torch.from_numpy(np.load(datapath)).unsqueeze(1).float())
		
		modes = np.load(labelpath)

		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(modes)

		Y = integer_encoded
		
		self.labels = torch.utils.data.TensorDataset(torch.from_numpy(Y).long())


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):

		image = self.data[idx]
		label = self.labels[idx]

		if self.transform:
			image = self.transform(image)

		return image, label



root_dir = ["./data/images.npy", "./data/labels.npy"]

dataset = CustomDataset(root_dir)

torch.manual_seed(0)


trainset, testset = torch.utils.data.random_split(dataset, [2000, 785])
trainset_1 = [trainset]


trainset1, trainset2, testset = torch.utils.data.random_split(dataset, [1000, 1000, 785])
trainset_2 = [trainset1, trainset2]


trainset1, trainset2, trainset3, trainset4, testset = torch.utils.data.random_split(dataset, [500, 500, 500, 500, 785])
trainset_4 = [trainset1, trainset2, trainset3, trainset4]


trainset1, trainset2, trainset3, trainset4, trainset5, trainset6, trainset7, trainset8, testset = torch.utils.data.random_split(dataset, [250, 250, 250, 250, 250, 250, 250, 250, 785])
trainset_8 = [trainset1, trainset2, trainset3, trainset4, trainset5, trainset6, trainset7, trainset8]

# print(len(trainset), len(testset))