from ultralytics import YOLO                    # object detection
import supervision as sv                        # object detection and tracking
import pickle                                   # de/serialization
import os                                       # Misc operating system interfaces
import cv2                                      # Shapes and text
import numpy as np                              # Support for large, multi-dimensional arrays and matrices, plus high-level math functions
import pandas as pd                             # Data analysis and manipulation
import sys                                      # System specific parameters and functions
sys.path.append('../')
from utils import get_centre_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_centre_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]           # get track id of ball (1) or empty dictionary and bbox or empty list. This list is used for interpolations
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])  # Translating ball positions to a pandas dataframe

        df_ball_positions = df_ball_positions.interpolate()                             # Interpolate missing values
        df_ball_positions = df_ball_positions.bfill()                                   # Backfill, to eliminate edge case of first frame not being detected

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]    # return ball positions to original format (list of dictionaries of lists, first line of this function)

        return ball_positions
    
    def detect_frames(self,frames):
        batch_size = 20                                                                    # to safeguard against memory issues
        detections = []
        for i in range(0,len(frames),batch_size):                                          # incremented by batch_size each time
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)         # predict over all frames in a batch (with 0.1 confidence threshold)
            detections += detections_batch                                                 # append detections in batch to list of detections
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):             # get object tracks from detections
        
        # read tracker data from stub_path, so that the tracker doesn't need to be re-run each time
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:                            # with statement is used for exception handling and cleaner code - https://www.geeksforgeeks.org/with-statement-in-python/#
                tracks = pickle.load(f)                                 # deserializes data into object format
            return tracks
        
        detections = self.detect_frames(frames)

        tracks = {                                                      # dictionary of lists for tracking objects
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):             # switch class names from key:value to value:key
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)         # convert to Supervision detection format

            # convert goalkeeper object to player (detection was switching between the two, 
            # probably because of the small size of the dataset in the yolov5 model used in the training ipynb)
            for object_ind , class_id in enumerate(detection_supervision.class_id):    
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # print(detection_supervision)

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)         # track objects

            tracks["players"].append({})                                                               # appends to dictionary
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:                                    # loop over frames detected by tracker
                # extract params, indices are obtained from object in supervision format in print() below
                bbox = frame_detection[0].tolist()                                           
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['player']:                                 # assign bboxes to players and append to tracklist
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}            # assign bboxes to referees and append to tracklist

            for frame_detection in detection_supervision:                          # for the ball
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}                   # track_id = 1 since there's only 1 ball

            # print(detection_with_tracks)

        if stub_path is not None:                      # write out tracker results so that it doesn't need to be re-run every time
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)                  # serializes object output

        return tracks                                                         # dictionary of list of dictionaries
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])                                      # y2 - bottom of bbox
        x_center, _ = get_centre_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),               # major, minor
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect) ),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    
    def draw_triangle(self,frame,bbox,color):
        y = int(bbox[1])
        x, _ = get_centre_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20]
        ])

        cv2.drawContours(frame,[triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame,[triangle_points],0,(0,0,0),2)

        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (800,0), (1500,120), (255,255,255), -1)
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)           # Draw rectangle as overlay

        team_ball_control_until_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = team_ball_control_until_frame[team_ball_control_until_frame==1].shape[0]    
        team_2_num_frames = team_ball_control_until_frame[team_ball_control_until_frame==2].shape[0]    
        team_1 = team_1_num_frames / ( team_1_num_frames + team_2_num_frames )                          # num of times Team 1 has ball
        team_2 = team_2_num_frames / ( team_1_num_frames + team_2_num_frames )                          # num of times Team 2 has ball

        cv2.putText(frame, f"Possession",(950,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"City: {team_1*100:.2f}%",(810,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"United: {team_2*100:.2f}%",(1030,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames=[]
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()                                # so that the original frames are preserved

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():       # draw players
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                # if track_id == 17:
                #     frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball',False):                                      # draw red triangle on player that has the ball
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))

            for _, referee in referee_dict.items():            # draw referees
                frame = self.draw_ellipse(frame, referee["bbox"], (0,255,255))
            
            for track_id, ball in ball_dict.items():           # draw ball
                frame = self.draw_triangle(frame, ball["bbox"], (0,255,0))

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames



