import cv2
import time
import imutils
import time

# image = cv2.imread('../test images/blur 2.jpg')
# resized_image = imutils.resize(image, width=400)
# # print(image[0])
# # print()
# # print(resized_image[0])

# model = cv2.dnn_superres.DnnSuperResImpl_create()

# start = time.time()
# model.readModel('ESPCN_x2.pb')
# model.setModel('espcn',2)
# enhanced_espcn = model.upsample(resized_image)
# end = time.time()
# print('Time spent in enhancing image with ESPCN is: ', end-start)
# cv2.imshow('ESPCN',enhanced_espcn)
# cv2.imshow('Original',resized_image)


# start = time.time()
# model.readModel('FSRCNN_x2.pb')
# model.setModel('fsrcnn',2)
# enhanced_fsrcnn = model.upsample(resized_image)
# end = time.time()
# print('Time spent in enhancing image with FSRCNN is: ', end-start)
# cv2.imshow('FSRCNN',enhanced_fsrcnn)

# cv2.imshow( "Original", resized_image)

##################### ENHANCING VIDEO QUALITY #################################
# vid = cv2.VideoCapture('../test images/video 1.mp4')

# model = cv2.dnn_superres.DnnSuperResImpl_create()
# model.readModel('ESPCN_x2.pb')
# model.setModel('espcn',2)

# while True:
#     ret, frame = vid.read()
    
#     resized_frame = imutils.resize(frame, width=400)
#     enhanced_espcn = model.upsample(resized_frame)
    
#     cv2.imshow('Original', resized_frame)
#     cv2.imshow('Enhanced', enhanced_espcn)
    
#     key = cv2.waitKey(10) & 0xFF
    
#     if key == ord("q"):
#         break

# cv2.waitKey(0)
# cv2.destroyAllWindows()

########################## ESRGAN IMPLEMENTATION #############################################

import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
