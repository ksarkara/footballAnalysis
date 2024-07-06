# footballAnalysis
The players, referees and football in an input video were detected and tracked, using the You Only Look Once (YOLO) computer vision model. A dataset from Roboflow was used to train the model. Based on the color of their shirts, players were assigned to teams via pixel segmentation and clustering using KMeans. This information was used to measure a team's ball possession for the duration of the video. Optical flow was used to measure camera movement between frames and subsequently, estimate a player's movement. Perspective transformation was then applied, to convert the unit of a player's estimated movement from pixels to metres. Finally, a player's speed and  distance covered are calculated, within the defined perspective.

![image](https://github.com/ksarkara/footballAnalysis/assets/113844617/1f52c590-e0a7-4b4f-9f13-5047623ad377)

## Requirements:
- Python 3.x
- Pickle
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
- scikit-learn
  
The Youtube video in the [Reference](https://github.com/ksarkara/footballAnalysis/blob/main/README.md#reference) section has installation steps, if needed.

## [Input Video](https://drive.google.com/file/d/1g_3Udg9BxMUkl_4EPLTd345vn8zzPpI4/view?usp=sharing)

## [Output Video](https://drive.google.com/file/d/1pLKxrznguuYUiTf1AmH3J61x5pBZGDqn/view?usp=sharing)

## [Training Model](https://drive.google.com/file/d/14R-DaMKO4PjsI04aDBjP6lnUzvDAixad/view?usp=sharing)

## Problems that I ran into and solutions:
### Google Colab issues:
- App needs to be added to Google Drive before you can use it
- For !yolo task=detect mode=train model=yolov5l.pt data={dataset.location}/data.yaml epochs=100 imgsz=640
  - Runtime options needed to be changed to use the GPU, otherwise the yolo command was not found
  - yolov5l.pt needed to be used (still running) as x or xu failed with an error on the first epoch

### Ellipses weren't being drawn:
- Check annotation code misalignment, especially when looping over frames

### NumPy error allocating 20+ MiB for array in draw_camera_movement():overlay = frame.copy():
- run vscode as admin; other solutions are [here](https://stackoverflow.com/questions/57507832/unable-to-allocate-array-with-shape-and-data-type)

### Camera movement was always showing as zero:
- Added "max_distance = distance" line in get_camera_movement()

## Future Work:
- Interpolate positions for a player (potential approaches are modifying dataset to indicate individual players or, less favorably, remembering multiple track_ids for the same player)
- Tighten ball and player tracking (potential approach is training using a model with more params - x or xu)

## Reference:
This project is heavily based on [abdullahtarek's](https://github.com/abdullahtarek) work, found on [Github](https://github.com/abdullahtarek/football_analysis) and [Youtube](https://www.youtube.com/watch?v=neBZ6huolkg)
