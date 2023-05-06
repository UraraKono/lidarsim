import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torchvision
import open3d as o3d

from model import UNet

X_DIM = 500
Y_DIM = 500
Z_DIM = 3
NUM_EPOCHS = 100
TEST_SPLIT = 0.15
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
PIN_MEMORY = False if DEVICE == "cpu" else True
MODEL_PATH = "./ckpts/"
PLOT_PATH = "./plots/"

class GridDataset(Dataset):
    def __init__(self, path):
        self.logpath = path
    def __len__(self):
        # Return the number of total samples contained in the dataset
        return len(os.listdir(os.path.join(self.logpath, 'x')))
    def __getitem__(self, idx):
        # print('Index', idx)
        # Grab from the current index
        pose_old = np.fromfile(self.logpath + '/pose/pose_old_{}.bin'.format(idx), dtype=np.float32)[:3]
        pose_new = np.fromfile(self.logpath + '/pose/pose_new_{}.bin'.format(idx), dtype=np.float32)[:3]
        # Append
        pose = torch.cat([torch.tensor(pose_old), torch.tensor(pose_new)])
        # Read input, convert to matrix, permute dimension to (Z_DIM, X_DIM, Y_DIM)
        x = torch.from_numpy(np.fromfile(self.logpath + '/x/x_{}.bin'.format(idx), dtype=np.float32).reshape(X_DIM, Y_DIM, Z_DIM).transpose(2, 0, 1))
        y = torch.from_numpy(np.fromfile(self.logpath + '/y/y_{}.bin'.format(idx), dtype=np.float32).reshape(X_DIM, Y_DIM, Z_DIM).transpose(2, 0, 1))

        # Return a tuple of (x, pose, y)
        return (x, pose, y)
    
def predict(model, testLoader, criterion):
    totalTestLoss = 0
    # Switch off autograd for evaluation
    with torch.no_grad():
        model.eval()

        # Loop over the test set
        for (i, (x, pose, y)) in enumerate(testLoader):
            # Send the input to the device
            (x, pose, y) = (x.to(DEVICE), pose.to(DEVICE), y.to(DEVICE))

            # Make the predictions and calculate the testing loss
            pred = model(x, pose)
            loss = criterion(pred, y)
            totalTestLoss += loss.item()

    return totalTestLoss

if __name__ == '__main__':
    
    print(os.cpu_count()) # 16 - W; 10 - M
    print(DEVICE)
    
    # Load the log files from 'logs' directory
    LOGS_PATH = './logs/train'
    LOGS_TEST_PATH = './logs/test'
    # Define the path to the x and y logs, get all files in the directory
    X_PATH = sorted(os.listdir(os.path.join(LOGS_PATH, 'x')))
    Y_PATH = sorted(os.listdir(os.path.join(LOGS_PATH, 'y')))
    # print(X_PATH, Y_PATH)

    # Partition the data into training and testing splits using 85% of
    # the data for training and the remaining 15% for testing
    # split = train_test_split(X_PATH, Y_PATH,
    #     test_size=TEST_SPLIT, random_state=42)

    # # Unpack the data split
    # (trainX, testX) = split[:2]
    # (trainY, testY) = split[2:]

    # Shuffle the data
    trainX = np.random.permutation(X_PATH)
    trainY = np.random.permutation(Y_PATH)
    testX = np.random.permutation(sorted(os.listdir(os.path.join(LOGS_TEST_PATH, 'x'))))
    testY = np.random.permutation(sorted(os.listdir(os.path.join(LOGS_TEST_PATH, 'y'))))
    # print(len(trainX), len(trainY), len(testX), len(testY))

    # Create the train and test datasets
    trainDS = GridDataset(LOGS_PATH)
    testDS = GridDataset(LOGS_TEST_PATH)
    # Create the training and test data loaders
    testLoader = DataLoader(trainDS, shuffle=False,
        batch_size=1, pin_memory=PIN_MEMORY,
        num_workers=10)
    
    criterion = nn.MSELoss() # BCEWithLogitsLoss()
    # print(unet(x).shape)
    
     # Load the model from disk (doesn't need to initialize the model)
    # Load the model from disk (need to initialize the model)
    unet = UNet(retain_dim=True).to(DEVICE)
    unet.load_state_dict(torch.load(MODEL_PATH + 'final.pth', map_location=DEVICE))
    
    with torch.no_grad():
        unet.eval()
        for (i, (x, pose, y)) in enumerate(testLoader):
            # Send the input to the device
            (x, pose, y) = (x.to(DEVICE), pose.to(DEVICE), y.to(DEVICE))

            # Make the predictions and calculate the testing loss
            pred = unet(x, pose).cpu().numpy().squeeze().transpose(1,2,0)
            # print(pred)
            # loss = criterion(pred, y)
    
            # Visualize point cloud overlayed on o3d voxel grid
            pcd_in = o3d.geometry.PointCloud()
            pcd = o3d.geometry.PointCloud()
            # for i in range(grid.shape[0]):
            #      # o3d is xzy
            #     pcd.points.append(np.array((grid[i][0] * 0.2 + 0.1, grid[i][2] * 0.2 + 0.1, grid[i][1] * 0.2 + 0.1)))
            x = x.cpu().numpy().squeeze().transpose(1,2,0)
            
            print(np.max(pred))

            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    pcd.points.append(np.array((i * 0.2 + 0.1, j * 0.2 + 0.1, pred[i,j,0] * 0.2 + 0.1)))
                    pcd.points.append(np.array((i * 0.2 + 0.1, j * 0.2 + 0.1, pred[i,j,1] * 0.2 + 0.1)))
                    pcd.points.append(np.array((i * 0.2 + 0.1, j * 0.2 + 0.1, pred[i,j,2] * 0.2 + 0.1)))
                    pcd_in.points.append(np.array((i * 0.2 + 0.1, j * 0.2 + 0.1, x[i,j,0] * 0.2 + 0.1)))
                    pcd_in.points.append(np.array((i * 0.2 + 0.1, j * 0.2 + 0.1, x[i,j,1] * 0.2 + 0.1)))
                    pcd_in.points.append(np.array((i * 0.2 + 0.1, j * 0.2 + 0.1, x[i,j,2] * 0.2 + 0.1)))
            o3d.visualization.draw_geometries([pcd])