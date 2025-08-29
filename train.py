import os
EPOCHS = 100
NUM_DATASET = 2


from ultralytics import YOLO
import os
import sys
import argparse
import yaml




def get_args():
    parser = argparse.ArgumentParser(description="Train Yolo")
    parser.add_argument('--epochs', '-e', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--datasets', '-d', type=int, default=NUM_DATASET, help='Number of Dataset')
    args = parser.parse_args()
    return args

def make_data_yaml():
    dict = {
        'train': list(),
        'val': list(),
        'test': 'TestImages',
        'nc': 1,
        'names': ['Soup']
    }
    for i in range(args.datasets):
        dict['train'].append(f'generate-data/Output_{i + 1}/train/images')
        dict['val'].append(f'generate-data/Output_{i + 1}/val/images')
    return dict

if __name__ == "__main__":
    args = get_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    with open("yolo_params.yaml", "w") as f:
        yaml.dump(make_data_yaml(), f)
    

    model = YOLO("yolov8n-seg.pt")
    print(this_dir)
    results = model.train(
        data=r"yolo_params.yaml",
        epochs=args.epochs,
        device=0,
        single_cls=True,
        mosaic=0.4,
        optimizer='AdamW',
        lr0=0.0001,
        lrf=0.0001,
        momentum=0.9,
        name='basic',
        task='segment'  
    )