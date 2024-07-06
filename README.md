# footballAnalysis
In this project the players, referees and football are detected and tracked in an input video, using the You Only Look Once (YOLO) computer vision model. A dataset from Roboflow was used to train the model. Based on the color of their shirts, players were assigned to teams using Kmeans for pixel segmentation and clustering. This information was used to measure a team's ball possession for the duration of the video. Optical flow was used to to measure camera movement between frames, enabling us to measure a player's movement. Perspective transformation was then applied, to convert the unit of a player's movement to metres, rather than pixels. Finally, a player's speed and the distance covered are calculated. 

![image](https://github.com/ksarkara/footballAnalysis/assets/113844617/1f52c590-e0a7-4b4f-9f13-5047623ad377)

# Requirements:
- Python 3.x
- Pickle
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
- scikit-learn
  
The reference Youtube video (link below) shows installation steps, if needed

# Problems that I ran into and solutions:
_Google Colab issues:_
- App needs to be added to Google Drive before you can use it
- For !yolo task=detect mode=train model=yolov5l.pt data={dataset.location}/data.yaml epochs=100 imgsz=640
  - Runtime options needed to be changed to use the GPU, otherwise the yolo command was not found
  - yolov5l.pt needed to be used (still running) as x or xu failed with an error on the first epoch


_Ellipses weren't being drawn:_
- Check annotation code misalignment, especially when looping over frames


_NumPy error allocating 20+ MiB for array in draw_camera_movement():overlay = frame.copy():_
- run vscode as admin; other solutions are here: https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type


_Camera movement was always showing as zero:_
- Added "max_distance = distance" line in get_camera_movement()

# Future Work:
- Interpolate positions for a player (potential approaches are modifying dataset to indicate individual players or, less favorably, remembering multiple track_ids for the same player)
- Tighten ball and player tracking (potential approach is training using a model with more params - x or xu)

# Reference:
This project is heavily based on abdullahtarek's work, found on:
- Github: https://github.com/abdullahtarek/football_analysis
- and Youtube: https://www.youtube.com/watch?v=neBZ6huolkg
