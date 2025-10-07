import torch
import torch.nn as nn

import os
import sys
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
print(script_directory)
import time 

import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

## Plot in real-time
class LiveCurve:
    def __init__(self, title="Learning Curve", ylabel="Loss"):
        plt.ion()  # interactive mode
        self.fig, self.ax = plt.subplots()
        (self.train_line,) = self.ax.plot([], [], label="Train")
        (self.dev_line,)   = self.ax.plot([], [], label="Dev")
        self.train_hist, self.dev_hist = [], []
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel(ylabel)
        self.ax.set_title(title)
        self.ax.grid(True)
        self.ax.legend()
        self.fig.show()          # non-blocking
        self.fig.canvas.draw()

    def update(self, train_loss, dev_loss):
        self.train_hist.append(train_loss)
        self.dev_hist.append(dev_loss)
        xs = range(1, len(self.train_hist) + 1)
        self.train_line.set_data(xs, self.train_hist)
        self.dev_line.set_data(xs, self.dev_hist)
        self.ax.relim(); self.ax.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)  # allow GUI to refresh

class TIMITDataset(torch.utils.data.Dataset):
    def __init__(self, train_data_path, label_data_path=None, validation_ratio=0.2, mode='train', isPostprocess=True, stride=5):
        self.mode = mode
        # 1. Read data from data_path
        data = np.load(train_data_path)
        print(f"Data type: {data.dtype}, shape: {data.shape}")
        if label_data_path:
            raw_labels = np.load(label_data_path)
            # print(f"Labels type: {raw_labels.dtype}, shape: {raw_labels.shape}" if raw_labels is not None else "No labels provided")
            labels = raw_labels.astype(np.int64)

        # 1.1 Postprocess data (e.g., framing)
        if isPostprocess:
            data = self.data_postprocess(data, stride)

        print(f"(Processed) data shape: {data.shape}")
        # 2. Preprocess data (e.g., normalization, padding)
        if mode != 'test':
            # 2.1 Split into training and validation sets
            percentage = int(data.shape[0] * (1 - validation_ratio))
            if mode == 'train':
                self.data = data[:percentage]
                self.labels = labels[:percentage]
            elif mode == 'validation':
                self.data = data[percentage:]
                self.labels = labels[percentage:]

            self.data = torch.tensor(self.data, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.int64)
        else:
            self.data = torch.tensor(data, dtype=torch.float32)

        self.dim = self.data.shape[1]

        print(f'Finished reading the {mode} set of TIMIT ({len(self.data)} samples found, each dim = {self.data.shape[1]})')

    # def Sample(self, data, stride):
    #     print(f"Start Sample()")
    #     data = data.reshape(-1, 11, 39)
    #     data = data[:,5,:]
    #     base = data.copy()
    #     for step in range(1, stride+1, 1):
    #         Rshift = np.roll(base,step,axis=0)
    #         data = np.concatenate((Rshift,data), axis=1)
    #     print(f"finish Rshift()")
    #     for step in range(-1, -1-1*stride, -1):
    #         Lshift = np.roll(base,step,axis=0)
    #         data = np.concatenate((data,Lshift), axis=1)
    #     print(f"finish Lshift()")
    #     data = np.reshape(data, (-1,2*stride+1,39))
    #     return data
    
    def data_postprocess(self, data, stride):
        """
        Convert flattened MFCC windows (N, (2*stride+1)*39) into (N, 2*stride+1, 39).

        Parameters
        ----------
        data : np.ndarray
            Shape (N, (2*stride+1)*39), e.g. (N, 429) for stride=5.
        stride : int
            Context radius. Window width = 2*stride+1.

        Returns
        -------
        np.ndarray
            Shape (N, 2*stride+1, 39). The center frame is [:, stride, :].
        """
        data = data.reshape(-1, 11, 39)
        print(f"\tPostprocessed data shape: {data.shape}")
        
        data = data[:,5,:]
        print(f"\tdata shape: {data.shape}")
        padded = np.pad(data, ((stride, stride), (0,0)), mode='constant', constant_values=0)
        windows = []
        for i in range(stride, stride + data.shape[0]):
            # take a slice [i-stride : i+stride+1]
            win = padded[i - stride : i + stride + 1]
            windows.append(win)
        data = np.array(windows)
        return data.reshape(data.shape[0], -1)
        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single data point (input, target)
        if self.mode in ['train', 'validation']:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]
        
class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), # 1
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 2048), # 2
            nn.BatchNorm1d(2048),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(2048, 3072), # 3
            nn.BatchNorm1d(3072),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(3072, 4096), # 4
            nn.BatchNorm1d(4096),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(4096, 3072), # 5
            nn.BatchNorm1d(3072),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(3072, 2048), # 6
            nn.BatchNorm1d(2048),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(2048, 1024), # 7
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 512), # 8
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(512, 128), # 9
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return x

def prep_dataLoader(data_path, label_path=None, batch_size=32, validation_ratio=0.2, mode='train', isPostprocess=True, stride=5):
    dataset = TIMITDataset(data_path, label_path, validation_ratio, mode, isPostprocess, stride)
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=(mode=='train'))
    return data_loader

def train_loop(dataloader, model, loss_fn, optimizer):
    train_acc = 0.0
    train_loss = 0.0
    model.train()
    print(f"Training on {len(dataloader.dataset)} samples...")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    for X, y in dataloader:
        # print(f"\tBatch X shape: {X.shape}, y shape: {y.shape}")
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        _, top_pred = torch.max(pred, 1)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += (top_pred.cpu() == y.cpu()).sum().item()
        train_loss += loss.item() * X.size(0)

    return train_acc, train_loss

def val_loop(dataloader, model, loss_fn):
    val_acc = 0.0
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = model(X)
            loss = loss_fn(pred, y)
            _, val_pred = torch.max(pred, 1)

            val_acc += (val_pred.cpu() == y.cpu()).sum().item()
            val_loss += loss.item() * X.size(0)

    return val_acc, val_loss

if __name__ == "__main__":
    # Hyperparameters ###################################
    Epochs = 50
    Batch_size = 1024
    Val_ratio = 0.05
    Learning_rate = 0.00001
    stride = 15
    isPostProcessing = True
    #####################################################

    live = LiveCurve(title="TIMIT DNN Learning Curve", ylabel="MSE")

    history = {"train_loss": [], "dev_loss": []}

    train_set = prep_dataLoader(data_path=script_directory + r'/Data/train_11.npy', label_path=script_directory + r'/Data/train_label_11.npy', batch_size=Batch_size, validation_ratio=Val_ratio, mode='train', isPostprocess=isPostProcessing, stride=stride)
    val_set = prep_dataLoader(data_path=script_directory + r'/Data/train_11.npy', label_path=script_directory + r'/Data/train_label_11.npy', batch_size=Batch_size, validation_ratio=Val_ratio, mode='validation', isPostprocess=isPostProcessing, stride=stride)
    test_set = prep_dataLoader(data_path=script_directory + r'/Data/test_11.npy', label_path=None, batch_size=Batch_size, validation_ratio=Val_ratio, mode='test', isPostprocess=False, stride=5)

    model = DNN(train_set.dataset.dim, 39).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=Learning_rate, betas=(0.95, 0.99), weight_decay=0.01)

    start_time = time.perf_counter() 
    for t in range(Epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_acc, train_loss = train_loop(train_set, model, loss_fn, optimizer)
        val_acc, val_loss = val_loop(val_set, model, loss_fn)
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(val_loss)
        live.update(train_loss, val_loss)
        print(f"Train Loss: {train_loss/len(train_set.dataset):.4f}, Train Acc: {train_acc/len(train_set.dataset):.4f}")
        print(f"Val Loss: {val_loss/len(val_set.dataset):.4f}, Val Acc: {val_acc/len(val_set.dataset):.4f}")    
        # time.sleep(0.8)  # to ensure the plot updates properly
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")

    plt.ioff()
    plt.show()
    # print(model)