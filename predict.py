from ultralytics import YOLO


# Load pretrained model
model = YOLO("yolo26n.pt")


def detect_image(image_path):
    """Detect objects in an image."""

    model.predict(
        source=image_path,
        conf=0.4,
        save=True
    )

    print("Image detection completed.")


def detect_video(video_path):
    """Detect objects in a video."""

    model.predict(
        source=video_path,
        conf=0.4,
        save=True
    )

    print("Video detection completed.")


def detect_webcam():
    """Run real-time detection using webcam."""

    model.predict(
        source=0,
        conf=0.4,
        show=True
    )


if __name__ == "__main__":

    print("\nObject Detection System")
    print("-----------------------")
    print("1. Image Detection")
    print("2. Video Detection")
    print("3. Webcam Detection")

    choice = input("\nChoose an option: ")

    if choice == "1":

        image_path = input("Enter image path: ")
        detect_image(image_path)

    elif choice == "2":

        video_path = input("Enter video path: ")
        detect_video(video_path)

    elif choice == "3":

        detect_webcam()

    else:

        print("Invalid choice.")