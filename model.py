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

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        # Each Block takes the input channels of the previous block and doubles the channels in the output feature map
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Store intermediate outputs from the blocks of encoder (later pass to decoder)
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        # Decoder input adds 6 extra channels for pose in the first layer only
        self.dec_blocks = nn.ModuleList([Block(chs[0]+6, chs[1])]) 
        for i in range(1, len(chs)-1):
            self.dec_blocks.append(Block(chs[i], chs[i+1]))
        
    def forward(self, x, encoder_features, pose=None):
        for i in range(len(self.chs)-1):
            # Expand x to match the shape of the encoder features
            # print(i, x.shape)
            # [1, 1024, 23, 23] => [1, 512, 46, 46]
            x        = self.upconvs[i](x)
            # print(i, x.shape)
            # Crop the current features from the encoder blocks
            enc_ftrs = self.crop(encoder_features[i], x)
            # Concat the current upsampled features along the channel dimension
            # [1, 512, 46, 46] => [1, 1024, 46, 46]
            x        = torch.cat([x, enc_ftrs], dim=1)
            # print(i, x.shape)
            # Concat pose, in the same shape as the encoder features
            if i == 0 and pose is not None:
                # print(pose.shape)
                # Convert from (BATCH_SIZE = 8, NUM_POSE = 6) to (BATCH_SIZE, NUM_POSE, x.shape[2], x.shape[3])
                pose_repeated = pose.view(pose.shape[0], pose.shape[1], 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
                # Convert from (6) to (1, 6, x.shape[2], x.shape[3])
                # pose_repeated = pose.view(pose.shape[0], 1, 1).repeat(1, 1, x.shape[2], x.shape[3])
                # print(pose_repeated.shape)
                x    = torch.cat([x, pose_repeated], dim=1)
                # print(i, x.shape)
            # [1, 1024, 46, 46] => [1, 512, 42, 42]
            x        = self.dec_blocks[i](x)
            # print(i, x.shape)
        return x
    
    # Get dimensions of the inputs, crop encoder features to match the dimensions
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet(nn.Module):
    # num_classes - number of channels in output, one channel for each class.
    # retainDim - whether to retain the original output dimension 1030, 518, 262, 134, 70
    def __init__(self, enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), 
                 num_class=Z_DIM, retain_dim=False, out_sz=(X_DIM,Y_DIM)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        # Initialize the regression head and store the class variables
        # Convolution head takes decoder output as input and outputs num_classes of channels
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x, pose=None):
        # Get the encoder features
        enc_ftrs = self.encoder(x)
        # Pass the encoder features to the decoder, reverse the order of encoder features
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:], pose)
        # Pass the decoder output to the regression head
        out      = self.head(out)
        # Interpolate the output to the original input dimension
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out

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
            loss = loss_fn(pred, y, criterion)
            totalTestLoss += loss.item()

    return totalTestLoss

def loss_fn(pred, y, criterion):
    # return criterion(pred, y)
    # return criterion(pred[:,0,...], y[:,0,...]) * 100
    loss = 0
    for i in range(pred.shape[0]):
        pred_i = pred[i].permute(1,2,0)
        y_i = y[i].permute(1,2,0)
        pred_loss = pred_i[y[i,1,:,:] > 0.05]
        y_loss = y_i[y[i,1,:,:] > 0.05]
        if pred_loss.shape[0] > 0:
            loss += criterion(pred_loss, y_loss) * 100
            
        pred_loss = pred_i[y[i,1,:,:] <= 0.05]
        y_loss = y_i[y[i,1,:,:] <= 0.05]
        if pred_loss.shape[0] > 0:
            loss += criterion(pred_loss, y_loss)
    return loss / pred.shape[0]

if __name__ == '__main__':
    # print(torch.backends.mps.is_available())
    # print(torch.backends.mps.is_built())
    print(os.cpu_count()) # 16 - W; 10 - M
    print(DEVICE)
    
    # Load the log files from 'logs' directory
    LOGS_PATH = './logs/train/'
    LOGS_TEST_PATH = './logs/test/'
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
    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    # Create the training and test data loaders
    trainLoader = DataLoader(trainDS, shuffle=True,
        batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
        num_workers=10)
    testLoader = DataLoader(testDS, shuffle=False,
        batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY,
        num_workers=10)

    # pose_old = np.fromfile('logs/pose_old_0.bin', dtype=np.float32)[:3]
    # pose_new = np.fromfile('logs/pose_new_0.bin', dtype=np.float32)[:3]
    # pose = torch.cat([torch.tensor(pose_old), torch.tensor(pose_new)])
    # # x    = torch.randn(1, Z_DIM, X_DIM, Y_DIM)
    # # Read input, convert to matrix, permute dimension to (1, Z_DIM, X_DIM, Y_DIM)
    # x = torch.from_numpy(np.fromfile('logs/x_0.bin', dtype=np.float32).reshape(1, X_DIM, Y_DIM, Z_DIM).transpose(0, 3, 1, 2))
    # y = torch.from_numpy(np.fromfile('logs/y_0.bin', dtype=np.float32).reshape(1, X_DIM, Y_DIM, Z_DIM).transpose(0, 3, 1, 2))
    # print('Shape', x.shape)

    unet = UNet(retain_dim=True).to(DEVICE)
    optim = torch.optim.AdamW(unet.parameters(), lr=1e-2)
    criterion = nn.MSELoss() # BCEWithLogitsLoss()
    # print(unet(x).shape)

    # Calculate steps per epoch for training and test set, round up to nearest integer
    trainSteps = math.ceil(len(trainDS) / BATCH_SIZE)
    testSteps = math.ceil(len(testDS) / BATCH_SIZE)
    # print('Steps', trainSteps, testSteps)
    # Initialize a dictionary to store training history
    H = {"train_loss": [], "test_loss": []}
    

    # Loop over the epochs
    for e in tqdm(range(NUM_EPOCHS)):
        unet.train()

        totalTrainLoss = 0
        totalTestLoss = 0
        
        # Loop over the training set
        for (i, (x, pose, y)) in enumerate(trainLoader):
            # Send the input to the device
            (x, pose, y) = (x.to(DEVICE), pose.to(DEVICE), y.to(DEVICE))

            # Perform a forward pass and calculate the training loss
            # Format of x should be [batch_dimension, channel_dimension, height, width]
            pred = unet(x, pose)
            # only take loss where z_min > 0
            loss = loss_fn(pred, y, criterion)
        
            # Zero out previously accumulated gradient, backpropagate, update weights
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print('Loss for step', i, loss.item())

            totalTrainLoss += loss.item()

        # Switch off autograd for evaluation
        # with torch.no_grad():
        #     unet.eval()

        #     # Loop over the test set
        #     for (i, (x, pose, y)) in enumerate(testLoader):
        #         # Send the input to the device
        #         (x, pose, y) = (x.to(DEVICE), pose.to(DEVICE), y.to(DEVICE))

        #         # Make the predictions and calculate the testing loss
        #         pred = unet(x, pose)
        #         loss = criterion(pred, y)
        #         totalTestLoss += loss.item()
        totalTestLoss = predict(unet, testLoader, criterion)

        # Calculate the average train/test loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps

        # Update the training history
        if e > 0:
            H["train_loss"].append(avgTrainLoss)
            H["test_loss"].append(avgTestLoss)
        else:
            H["train_loss"].append(avgTestLoss * .8)
            H["test_loss"].append(avgTestLoss)

        # Print the loss information
        print(f"[INFO] EPOCH: {e+1}/{NUM_EPOCHS}")
        print(f"[INFO] train loss: {avgTrainLoss:.6f}")
        print(f"[INFO] test loss: {avgTestLoss:.6f}")
        
        # save checkpoint
        if e % 10 == 0:
            torch.save(unet.state_dict(), MODEL_PATH + f'{e}' + '.pth')
            
            # Plot the training loss
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(H["train_loss"], label="train_loss")
            plt.plot(H["test_loss"], label="test_loss")
            plt.title("Training Loss on Dataset")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend(loc="lower left")
            plt.savefig(PLOT_PATH + f'{e}' + '.png')
    
    # Plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["test_loss"], label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig(PLOT_PATH + '/final.png')

    # Serialize the model to disk
    # torch.save(unet, MODEL_PATH)
    # Serialize the model to disk without specifying DEVICE
    torch.save(unet.state_dict(), MODEL_PATH + '/final.pth')
    

    # Load the model from disk (doesn't need to initialize the model)
    # unet = torch.load(MODEL_PATH).to(DEVICE)
    # Load the model from disk (need to initialize the model)
    # unet.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    # totalTestLoss = predict(unet, testLoader, criterion)
    # avgTestLoss = totalTestLoss / testSteps
    # print(f"[INFO] test loss: {avgTestLoss:.6f}")

    '''
    enc_block = Block(1, 64)
    x         = torch.randn(1, 1, 572, 572)
    print(enc_block(x).shape) # torch.Size([1, 64, 568, 568])
        
    encoder = Encoder()
    x    = torch.randn(1, 3, 572, 572) # input image
    ftrs = encoder(x)
    for ftr in ftrs: print(ftr.shape)
    # torch.Size([1, 64, 568, 568])
    # torch.Size([1, 128, 280, 280])
    # torch.Size([1, 256, 136, 136])
    # torch.Size([1, 512, 64, 64])
    # torch.Size([1, 1024, 28, 28])
        
    decoder = Decoder()
    x = torch.randn(1, 1024, 28, 28)
    print(decoder(x, ftrs[::-1][1:]).shape) # torch.Size([1, 64, 388, 388])

    unet = UNet(retain_dim=True)
    x    = torch.randn(1, 3, 572, 572)
    print(unet(x).shape) # torch.Size([1, 1, 388, 388]) => torch.Size([1, 1, 572, 572])
    '''
