# https://docs.opencv.org/4.x/dd/de7/group__videoio.html

import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []                                    # Initialize list of frames
    while True:
        ret, frame = cap.read()                    # read in the next frame, assign to 'frame' and return a flag (ret). ret is true if there is a next frame and false if the video has ended
        if not ret:                                # video has ended, break out of while loop
            break
        frames.append(frame)                       # video has not ended, append frame to list of frames
    return frames                                  # return list of frames

def save_video(output_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')                       # output video type
    out = cv2.VideoWriter(output_video_path, fourcc, 59.94, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))    # 59.94 - fps; shape[1] - width of frame; shape[0] - height        
    for frame in output_video_frames:                              # loop over each frame
        out.write(frame)                                           # write frame to video writer
    out.release()