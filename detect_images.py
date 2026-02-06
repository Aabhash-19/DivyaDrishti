import os
import cv2
from ultralytics import YOLO

class ImageDetector:
    def __init__(self, model_path='runs/detect/train/weights/best.pt'):
        try:
            self.model = YOLO(model_path)
            print("✅ Custom model loaded!")
        except:
            self.model = YOLO('yolov8n.pt')
            print("⚠️ Using pre-trained model")
    
    def detect_images(self, image_dir, output_dir='results'):
        """Detect objects in all images in directory"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        image_path = "/Users/irray/Desktop/Projects/DivyaDrishti/Datasets/Object Detection-Images/test_images"
        
        for img_file in os.listdir(image_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_path, img_file)
                
                
                results = self.model(img_path)
                
                
                output_path = os.path.join(output_dir, f"detected_{img_file}")
                results[0].save(filename=output_path)
                print(f"✅ Processed: {img_file}")

if __name__ == '__main__':
    detector = ImageDetector()
    test_dir = "/Users/irray/Desktop/Projects/DivyaDrishti/Datasets/Object Detection-Images/test_images"
    detector.detect_images(test_dir)