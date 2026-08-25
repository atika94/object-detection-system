from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo26n.pt")


def detect_image(image_path):
    """Detect objects in an image."""
    results = model.predict(
        source=image_path,
        conf=0.4,
        save=True
    )

    print("Image detection completed.")
    return results


def detect_webcam():
    """Run real-time object detection using webcam."""
    model.predict(
        source=0,
        conf=0.4,
        show=True
    )


if __name__ == "__main__":

    print("Object Detection System")
    print("-----------------------")
    print("1. Image Detection")
    print("2. Webcam Detection")

    choice = input("Choose an option: ")

    if choice == "1":
        image_path = input("Enter image path: ")
        detect_image(image_path)

    elif choice == "2":
        detect_webcam()

    else:
        print("Invalid choice.")