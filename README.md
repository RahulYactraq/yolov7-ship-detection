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

3. Now run the script **convert-RLE-bbox.py** and convert Run length encoding into the Bounding boxes.
	
	While converting RLE to bbox we can also spliting our dataset into train, test and validation sets.

4. Now that we have our dataset ready in the format of yolo, its time to use YOLOv7 github repo and train our model.

5. 