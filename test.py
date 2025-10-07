import numpy as np

def sample(data, stride):
    # data shape: (N, 39) where N = #frames
    # Pad along time axis (axis=0)
    padded = np.pad(data, ((stride, stride), (0,0)), mode='constant', constant_values=0)
    
    # Collect windows with stride context
    windows = []
    for i in range(stride, stride + data.shape[0]):
        # take a slice [i-stride : i+stride+1]
        win = padded[i - stride : i + stride + 1]
        windows.append(win)
    
    return np.array(windows)  # shape: (N, 2*stride+1, 39)


mfcc = np.array([
    [0,0,0,0],  # frame0
    [1,1,1,1],  # frame1
    [2,2,2,2],  # frame2
    [3,3,3,3],  # frame3
    [4,4,4,4]   # frame4
])

out = sample(mfcc, stride=2)
print("Output shape:", out.shape)
print(out)



