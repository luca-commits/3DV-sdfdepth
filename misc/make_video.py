import cv2 
import numpy as np
import os

rgb1_path = "/home/casimir/ETH/kitti/nerf_visuals/nerf_no_gt/theirs"
rgb2_path = "/home/casimir/ETH/kitti/nerf_visuals/nerf_no_gt/ours"

rgb1_dirs = [os.path.join(rgb1_path, file) for file in sorted(os.listdir(rgb1_path))]
rgb2_dirs = [os.path.join(rgb2_path, file) for file in sorted(os.listdir(rgb2_path))]

frame_shape = cv2.imread(rgb1_dirs[0]).shape
new_shape = (frame_shape[1], frame_shape[0]*2)

# breakpoint()


output = cv2.VideoWriter('stacked.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, new_shape)
# output = cv2.VideoWriter('renders/videos/stacked.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps_1, frame_size)
for rgb1_dir, rgb2_dir in zip(rgb1_dirs, rgb2_dirs):
    # vid_capture.read() methods returns a tuple, first element is a bool 
    # and the second is frame

    img1 = cv2.imread(rgb1_dir)
    img2 = cv2.imread(rgb2_dir)
    

    # combined = np.vstack((frame_1, frame_2))
    combined = np.vstack((img1, img2))

    output.write(combined);

    cv2.imshow('Frame',combined)
    # 20 is in milliseconds, try to increase the value, say 50 and observe
    key = cv2.waitKey(20)
    
    if key == ord('q'):
        break
	
# Release the objects
output.release()