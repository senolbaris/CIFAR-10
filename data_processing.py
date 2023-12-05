import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os


processed_data_dir = "Processed Data"

def create_train_data(train_data_directory, WIDTH, HEIGHT, random_state, val_size, CHANNELS=3):
	train_images = []
	train_category = []
	path = os.path.join(train_data_directory, "train")
	in_path = os.listdir(path)
	for folder in in_path:	
		folder_path = os.listdir(os.path.join(path, folder))
		num_of_classes = len(in_path)
		for image in folder_path:
			image_path = os.path.join(path, folder, image)
			img = cv2.imread(image_path)
			img = cv2.resize(img, (WIDTH, HEIGHT))
			img = img / 255.0
			train_images.append(img)
			train_category.append(folder)


	train_category = np.array(train_category)

	label_encoder = LabelEncoder()
	train_category = label_encoder.fit_transform(train_category)

	train_images = np.array(train_images)

	train_images = np.reshape(train_images, (-1, WIDTH, HEIGHT, CHANNELS))
	train_category = np.reshape(train_category, (-1, 1))

	train_category = to_categorical(train_category, num_of_classes)

	X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_category, random_state=random_state, test_size = val_size)


	np.save("X_train.npy", X_train)
	np.save("X_val.npy", X_val)
	np.save("Y_train.npy", Y_train)
	np.save("Y_val.npy", Y_val)

	print("Saving is done!")


def create_test_data(test_data_directory, WIDTH, HEIGHT, CHANNLES=3):
	test_images = []
	test_category = []
	test_path = os.path.join(test_data_directory, "Test")
	in_test = os.listdir(test_path)
	for folder in in_test:
		folder_path = os.path.join(test_path, folder)
		in_folder = os.listdir(folder_path)
		for images in in_folder:
			image_path = os.path.join(folder_path, images)
			image = cv2.imread(image_path)
			image = cv2.resize(image, (WIDTH, HEIGHT))
			image = image / 255.0
			test_images.append(image)
			test_category.append(folder)

	print("Reading is done!")

	test_images = np.array(test_images)
	test_category = np.array(test_category)

	test_images = np.reshape(test_images, (-1, WIDTH, HEIGHT, CHANNLES))
	test_category = np.reshape(test_category, (-1, 1))


	label_encoder = LabelEncoder()
	test_category = label_encoder.fit_transform(test_category)
	test_category = to_categorical(test_category, 4)

	
	np.save("X_test.npy", test_images)
	np.save("Y_test.npy", test_category)
	print("Saving is done!")

def load_test_data():
	X_test = np.load(processed_data_dir+"/X_test.npy")
	Y_test = np.load(processed_data_dir+"/Y_test.npy")

	return X_test, Y_test
	
create_train_data("cifar10", 64, 64, 1, 0.2)
create_test_data("cifar10", 64, 64)