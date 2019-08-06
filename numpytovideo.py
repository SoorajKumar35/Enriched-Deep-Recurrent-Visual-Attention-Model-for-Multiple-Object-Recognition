import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
    frames: (N, H, W, D) 
"""
def record(frames, name, framerate, captions=None):

    (N, H, W, D) = frames.shape
    bottom_buffer = 30
    if (type(captions)) != type(None):
        assert N == len(captions)
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(name, fourcc, framerate, (H+bottom_buffer, W))
    else:
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(name, fourcc, framerate, (H, W))
        
        
        
    for i in range(N):
        window = frames[i]
        if  (type(captions) != type(None)):
            frame = np.zeros((H+bottom_buffer, W, D)).astype(np.uint8)
            frame[:100] = window
            cv2.putText(frame, text=captions[i], org = (H,0), fontFace=3, fontScale=15, color=(255,255,255), thickness=5)
        else:
            frame = window    
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        #print(frame.shape)
        out.write(frame)
        #plt.imshow(frame)
        #plt.show()
    cap.release()
    out.release()


def main():
    pass

if __name__ == "__main__":
    main()
