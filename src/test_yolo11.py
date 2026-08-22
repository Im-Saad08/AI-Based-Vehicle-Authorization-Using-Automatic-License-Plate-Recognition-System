#from ultralytics import YOLO

# 1. Load the pre-trained YOLOv11 model
#model = YOLO("yolo11n.pt")

# 2. Path to your input video file 
# (Place your video in your project folder or use an absolute path)
#video_source = "img/input/video.mp4" 

# 3. Process the video
# 'save=True' automatically creates an output video with tracking boxes
#results = model(source=video_source, save=True, conf=0.25)

#print("Video processing complete! Check the 'runs/detect/predict-5' folder for the output.")


#from ultralytics import YOLO

#model = YOLO("models/trained/yolo11n_best.pt")

#results = model.predict(
#    source="img/input/video.mp4", #insert input path (image or video)
#    conf=0.5,
#    save=True
#)

from ultralytics import YOLO

model = YOLO("models/trained/yolo11n_best.pt")

print("MODEL CLASSES:")
print(model.names)

print("\nMODEL:")
print(model)
