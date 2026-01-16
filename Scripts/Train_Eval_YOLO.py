from ultralytics import YOLO
import json
import os


# List your datasets and corresponding data.yaml files
tasks = [
    {
        "name": "Tree-Top-View",
        "data": "Data/Tree-Top-View/data.yaml",
        "project": "YOLO_TreeCanopy",
        "epochs": 10
    },
    {
        "name": "Wildfire",
        "data": "Data/wildfire/data.yaml",
        "project": "YOLO_Wildfire",
        "epochs": 10
    }
]

#Root folder to save metrics
output_folder =  "YOLO Model"

#Ensure folder exists
os.makedirs(output_folder, exist_ok=True)

# Train and evaluate each model
for task in tasks:
    print(f"\n Training model for: {task['name']}")
    model = YOLO("yolov8n.pt")

    model.train(
        data=task["data"],
        epochs=task["epochs"],
        imgsz=640,
        project=task["project"],
        name="exp",
        verbose=True,
        resume=False
    )

    print(f"Evaluating model for: {task['name']}")
    metrics = model.val(data=task["data"], split="test")  # Uses test set
    


output_path = os.path.join(output_folder, f"{task['name'].replace(' ', '_')}_metrics.json")
with open(output_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("Training completed and metrics saved.")