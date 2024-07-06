# footballAnalysis
This project is heavily based on abdullahtarek's work, found on:
- Github: https://github.com/abdullahtarek/football_analysis
- and Youtube: https://www.youtube.com/watch?v=neBZ6huolkg

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
The Youtube video shows installation steps, if needed

# Problems that I ran into
Google Colab:
- App needs to be added to Google Drive before you can use it
- For !yolo task=detect mode=train model=yolov5l.pt data={dataset.location}/data.yaml epochs=100 imgsz=640
  - Runtime options needed to be changed to use the GPU, otherwise the yolo command was not found
  - yolov5l.pt needed to be used (still running) as x or xu failed with an error on the first epoch
