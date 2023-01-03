import torch
import torchvision
import glob
import cv2
import numpy as np

# states = torch.load("./states/imgs.pth")
# # states = states.cpu().numpy()
# print(states)

frameSize = (320,210)

out = cv2.VideoWriter('./states/output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

for filename in glob.glob('./states/*.png'):
    img = cv2.imread(filename)
    out.write(img)

out.release()
