import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from time import time


# paths
your_directory = os.path.dirname(os.path.abspath(__file__))
rate_path = your_directory + r'\rate.png'
badapple_path = your_directory + r'\badapple.mp4'
gradient_path = your_directory + r'\gradient.png'
save_path = your_directory + r'\frames_non_gradient'


# setup
rate = cv2.imread(rate_path)
gradient_original = cv2.imread(gradient_path)

figures, axis = plt.subplots(1, 2)

axis[0].imshow(rate)
axis[1].imshow(gradient_original)

plt.show()

# main func without gradient
def frame2exch(frame: np.ndarray, rate_rect: tuple = ((875, 515), (1155, 725)), rate_clr: tuple = (129, 201, 149), alpha_bad: float = 0.7, alpha_gradient: float = 0.5):
    frame = cv2.resize(frame, (rate_rect[1][0]-rate_rect[0][0], rate_rect[1][1]-rate_rect[0][1]))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(gray, 30, 200) 
    contours, hierarchy = cv2.findContours(edged,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    rate_copy = rate.copy()
    x, y = rate_rect[0]
    for contour in contours:
        contour = contour.reshape(-1, 2)
        for index in np.arange(contour.shape[0]-1):
            cv2.line(rate_copy, (x+contour[index][0], y+contour[index][1]), (x+contour[index+1][0], y+contour[index+1][1]), rate_clr, 3)
    rate_copy = cv2.addWeighted(rate_copy, alpha_bad, rate, 1 - alpha_bad, 0)

    return rate_copy

#main func with gradient
def frame2exch(frame: np.ndarray, rate_rect: tuple = ((875, 515), (1155, 725)), rate_clr: tuple = (129, 201, 149), alpha_bad: float = 0.7, alpha_gradient: float = 0.5):
    frame = cv2.resize(frame, (rate_rect[1][0]-rate_rect[0][0], rate_rect[1][1]-rate_rect[0][1]))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    edged = cv2.Canny(gray, 30, 200) 
    contours, hierarchy = cv2.findContours(edged,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    rate_copy = rate.copy()
    x, y = rate_rect[0]
    for contour in contours:
        contour = contour.reshape(-1, 2)
        for index in np.arange(contour.shape[0]-1):
            cv2.line(rate_copy, (x+contour[index][0], y+contour[index][1]), (x+contour[index+1][0], y+contour[index+1][1]), rate_clr, 3)
    rate_copy = cv2.addWeighted(rate_copy, alpha_bad, rate, 1 - alpha_bad, 0)

    x, y, w, h = 800, 0, 800, gradient_original.shape[1]-500
    gradient = cv2.resize(cv2.rotate(gradient_original[y:y+h, x:x+w], cv2.ROTATE_90_CLOCKWISE), (rate_rect[1][0]-rate_rect[0][0], rate_rect[1][1]-rate_rect[0][1]))

    mask = np.zeros(frame.shape[:-1], np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    height, width = frame.shape[:-1]
    mask1 = np.zeros((height+2, width+2), np.uint8)
    cv2.floodFill(mask, mask1, (0,0), 255)
    mask = np.where(frame[:,:,2] == 0, 255, 0)

    croped = rate_copy[rate_rect[0][1]:rate_rect[1][1], rate_rect[0][0]:rate_rect[1][0]]
    croped_gradient = np.where(mask.reshape(*mask.shape, 1) == [0], gradient, croped)
    croped_gradient = cv2.addWeighted(croped_gradient, alpha_gradient, croped, 1 - alpha_gradient, 0)

    rate_copy[rate_rect[0][1]:rate_rect[1][1], rate_rect[0][0]:rate_rect[1][0]] = croped_gradient

    return rate_copy

# start compiling

badapple = cv2.VideoCapture(badapple_path)
cnt_frames = int(badapple.get(cv2.CAP_PROP_FRAME_COUNT))

last_frame = len(os.listdir(save_path))
print(f'Last frame: {last_frame}')

start = time()
count = 0
while True: 
    success, frame = badapple.read()

    if not success:
        break

    if last_frame > count:
        count += 1
        continue

    cv2.imwrite(f'{save_path}\\{str(count).zfill(4)}.png', frame2exch(frame))

    count += 1
    if not count % 10:
        print(f"{count}/{cnt_frames} is gone in {(time()-start):.2f} seconds")
        start = time()