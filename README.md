# ship-detection

## Steps

1. Create a directory structure for training, testing and validation as following

		-train
			-images
			-labels
		-val
			-images
			-labels
		-test
			-images
			-labels

2. Download the dataset from Kaggle
	
	Dataset Link: https://www.kaggle.com/competitions/airbus-ship-detection/data

3. Now run the script **convert-rle-bbox.py** and convert Run length encoding into the Bounding boxes.
	
	While converting RLE to bbox we can also spliting our dataset into train, test and validation sets.

4. Now that we have our dataset ready in the format of yolo, its time to use YOLOv7 github repo and train our model.

5. first thing run requirements.txt file with weights and baises package

		pip install -r requirements.txt wandb

6. Now Before start training we need to add our dataset path in /data folder.

	I already created a **ship.yaml** file that contains path to train, val and test dataset, but we might have to change and add a absolute path.

		/data/ship.yaml

7. Now let's start training

		python train.py --batch-size 16 --data /data/ship.yaml --img 224 224 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml --epochs 20

	
	In the terminal you will see a weights and baises login setup, you can choose any option you want and then it will start the training.



## Checklist

- [ ] Create train, val, test folders with images and labels
- [ ] Download the dataset from kaggle
- [ ] Run **convert-rle-bbox.py** script
- [ ] Run requirements.txt file with wandb package
- [ ] Add .yaml file in /data folder
- [ ] start the training