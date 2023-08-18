import window
import cv2
import time
import win32gui
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from mmseg.apis import init_model, inference_model, show_result_pyplot

config_path = '../../work_dirs/pidnet-s_2xb6-120k_1024x1024-glomer/pidnet-s_2xb6-120k_1024x1024-glomer.py'
checkpoint_path = '../../work_dirs/pidnet-s_2xb6-120k_1024x1024-glomer/iter_120000.pth'

model = init_model(config_path, checkpoint_path, device='cuda:0')

ESC_KEY=27
FRAME_RATE = 60
SLEEP_TIME = 1/FRAME_RATE

capture = window.WindowCapture("Motic Digital Slide Assistant",FRAME_RATE)

while True:
    start=time.time()
    frame = capture.screenshot()
    seg_frame = inference_model(model, frame)
    vis_frame = show_result_pyplot(model, frame, seg_frame, show=False)
    cv2.imshow("Segmentation", vis_frame)
    delta= time.time()-start
    if delta <SLEEP_TIME:
        time.sleep(SLEEP_TIME-delta)
    key= cv2.waitKey(1) & 0xFF
    if key== ESC_KEY:
        break