import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import matplotlib.pyplot as plt

X_train = np.load("x_train.npy")
Y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")
X_test = np.load("X_test.npy")
Y_test = np.load("Y_test.npy")

print(X_test.shape, Y_test.shape)
print(X_test[0])
print(Y_test[0])
print(X_test[0].shape)


print(X_train.shape)
print(Y_train.shape)

def build_model(WEIGHT, HEIGHT, CHANNELS, drop_out_rate, num_of_classes):
	model = Sequential()

	model.add(Conv2D(input_shape=(WEIGHT, HEIGHT, CHANNELS),
								filters=32,
								kernel_size=(2, 2),
								padding="same",
								activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(rate=drop_out_rate))

	model.add(Conv2D(filters=64,
					kernel_size=(2, 2),
					padding="same",
					activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(rate=drop_out_rate))
	model.add(Conv2D(filters=128,
					kernel_size=(2, 2),
					padding="same",
					activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(rate=drop_out_rate))
	model.add(Conv2D(filters=128,
					kernel_size=(2, 2),
					padding="same",
					activation="relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(units=512,
					activation="relu"))
	#model.add(Dropout(rate=drop_out_rate))
	#model.add(Dropout(rate=drop_out_rate))

	model.add(Dense(units=num_of_classes,
					activation="softmax"))

	model.compile(optimizer="Adam",
				loss="categorical_crossentropy",
				metrics=["accuracy"])

	return model


def train_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size):
	history = model.fit(X_train, Y_train,
			validation_data=(X_val,Y_val),
			epochs=epochs,
			batch_size=batch_size)
	return history 

def visualize_history(history):
    history = history.history
    plt.figure(0)
    plt.plot(history['accuracy'], label='train accuracy')
    plt.plot(history['val_accuracy'], label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('train_accuracy')

    plt.figure(1)
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('train_loss')

def save_model(model):
	model_json = model.to_json()
	with open("model.json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model.h5")
	print("Saving is done!")


model = build_model(64, 64, 3, 0.25, 4)
model.summary()
history = train_model(model, X_train, Y_train, X_val, Y_val, 20, 64)
save_model(model)
visualize_history(history)



