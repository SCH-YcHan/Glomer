import win32gui, win32ui, win32con
import numpy as np
import cv2 as cv
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from time import time
from mmseg.apis import init_model, inference_model, show_result_pyplot

config_path = '../../work_dirs/pidnet-s_2xb6-120k_256x256-glomer/pidnet-s_2xb6-120k_256x256-glomer.py'
checkpoint_path = '../../work_dirs/pidnet-s_2xb6-120k_256x256-glomer/iter_300000.pth'

model = init_model(config_path, checkpoint_path, device='cuda:0')

class WindowCapture:

    # properties
    w = 0
    h = 0
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name):
        # find the handle for the window we want to capture
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]

        # account for the window border and titlebar and cut them off
        border_pixels = 8
        titlebar_pixels = 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)

        return img

wincap = WindowCapture("Image Viewer 256")

palette = model.dataset_meta["palette"]

loop_time = time()

count = 0
Mean_FPS = 0

while(True):
    screenshot = wincap.get_screenshot()
    pred_screenshot = inference_model(model, screenshot)
    seg_map = pred_screenshot.pred_sem_seg.data[0]
    seg_map = seg_map.cpu().numpy()
    seg_colored = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(palette):
        seg_colored[seg_map == i] = color
    seg_screenshot = cv.addWeighted(screenshot, 0.7, seg_colored, 0.3, 0)
    cv.imshow("Segmentation_256", seg_screenshot)
    
    FPS = 1 / (time() - loop_time)
    Mean_FPS += FPS
    count += 1

    if count % 50 == 0:
        print('Mean FPS {}'.format(Mean_FPS/50))
        Mean_FPS = 0

    loop_time = time()

    # press 'q' with the output window focused to exit.
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break