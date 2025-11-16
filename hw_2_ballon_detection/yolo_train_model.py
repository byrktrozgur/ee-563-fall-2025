from ultralytics import YOLO

def main():
    # Load a COCO-pretrained YOLOv8n segmentation model
    model = YOLO("yolov8n.pt")

    # Display model information (optional)
    model.info()

    # Train the model on your balloon dataset
    results = model.train(
        data="balloon_dataset\\balloon.yml", 
        epochs= 100,
        imgsz=640,
        workers=0,           
        device=0 # for CPU --> device = cpu
    )

if __name__ == "__main__":
    main()
        