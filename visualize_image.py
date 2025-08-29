import os
import cv2

class YoloVisualize:
    MODE_TRAIN = 0
    MODE_VAL = 1
    def __init__(self, path_dataset):
        self.path_dataset = path_dataset

        # Chuyen cac class thanh mot dictionary, trong do key: value la id va ten cua class
        classes_path = os.path.join(path_dataset, "classes.txt")
        with open(classes_path, "r") as f:
            self.classes = f.read().splitlines()
        self.classes = {i: c for i, c in enumerate(self.classes)}

        self.set_mode(YoloVisualize.MODE_TRAIN)
    # luu cac anh va cac label
    def set_mode(self, mode=MODE_TRAIN):
        if mode == self.MODE_TRAIN:
            self.path_images = os.path.join(self.path_dataset, "generate-data/Output_1/train", "images")
            self.path_labels = os.path.join(self.path_dataset, "generate-data/Output_1/train", "labels")
        else:
            self.path_images = os.path.join(self.path_dataset, "generate-data/Output_1/val", "images")
            self.path_labels = os.path.join(self.path_dataset, "generate-data/Output_1/val", "labels")
        self.num_images = len(os.listdir(self.path_images))
        num_labels = len(os.listdir(self.path_labels))
        self.images_name = sorted(os.listdir(self.path_images))
        self.labels_name = sorted(os.listdir(self.path_labels))
        assert self.num_images == num_labels
        self.frame_index = 0
    
    # duyet qua tung frame
    def next_frame(self):
        self.frame_index+=1
        if self.frame_index >= self.num_images:
            self.frame_index = 0
    def prev_frame(self):
        self.frame_index-=1
        if self.frame_index < 0:
            self.frame_index = self.num_images - 1
    #visualize anh
    def seek_frame(self, idx):
        image_file = os.path.join(self.path_images, self.images_name[idx])
        label_file = os.path.join(self.path_labels, self.labels_name[idx])
        with open(label_file, "r") as f:
            lines = f.read().splitlines()
        image = cv2.imread(image_file)
        for line in lines:
            class_index, x, y, w,  h = map(float, line.split())
            cx = int(x * image.shape[1])
            cy = int(y * image.shape[0])
            w = int(w * image.shape[1])
            h = int(h * image.shape[0])
            x = cx - w // 2
            y = cy - h // 2
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, self.classes[int(class_index)], (x, y - 10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image
    def run(self):
        while True:
            frame = self.seek_frame(self.frame_index)
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow(f"Yolo Visualizer {self.path_dataset}", frame)
            key = cv2.waitKey(0)
            if key == ord('q') or key == 27 or key == -1:
                break
            elif key == ord('d'):
                self.next_frame()
            elif key == ord('a'):
                self.prev_frame()
            elif key == ord('t'):
                self.set_mode(YoloVisualize.MODE_TRAIN)
            elif key == ord('v'):
                self.set_mode(YoloVisualize.MODE_VAL)
            cv2.destroyAllWindows()




    

if __name__ == "__main__":
    file_path = ""
    vis = YoloVisualize(file_path)
    vis.run()