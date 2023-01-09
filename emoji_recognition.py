
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

NUMBER_OF_EMOJIS = 1
EPOCH_NUMBER = 15

#Make red, green, blue values in range [0, 1]
train_gen = ImageDataGenerator(rescale=1.0 / 255)
validation_gen = ImageDataGenerator(rescale=1.0 / 255)

#Create train set
train = train_gen.flow_from_directory("emojis/",
    target_size=(64,64),
    batch_size=3,
    class_mode="sparse")

#Create validation set
valid = train_gen.flow_from_directory("validation/",
    target_size=(64,64),
    batch_size=3,
    class_mode="sparse")

#Make model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(64,64,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPool2D(2,2), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(13)
])

#Compile model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True),],
    metrics=["accuracy"]
)

#Train model
model_fit = model.fit(train, epochs=EPOCH_NUMBER, validation_data=valid)

#Load test image
img = tf.keras.utils.load_img("test.png", target_size=(64,64))
x = tf.keras.utils.img_to_array(img)
images = np.stack([x])

#Get similarity values
#Values are in range [-6144, 6144] because max value is 64x64x3 (image height x image width x (red + green + blue))
if(NUMBER_OF_EMOJIS == 1):
    val = model.predict(images)

    print("Classes and their indexes:\n", train.class_indices)
    print("\nValues for each class:\n",val)
    value = {i for i in train.class_indices if train.class_indices[i]==np.argmax(val, axis=1)}
    print(value)

    #Making graph
    colors = ["#57020a", "#041cb8", "#0a9acf", "#879693", "#bd6a06", "#c20c4e", "#fc0303", "#ef24f2", "#1e4a29", "#cfca88", "#8059ab", "#3cb58b", "#188026"]
    x_axis = ["Angry", "Pule", "Cry", "Fear", "Happy", "Kiss", "Heart", "Love", "Sad", "Smile", "Tongue", "Wink", "Wow"]
    y_axis = []
    for y in np.array(val[0]).astype(np.int64):
        y_axis.append((y + 6144)/122)
    
    plt.barh(x_axis, y_axis, color=colors)
    plt.xlabel("Similarity (%)")
    plt.title(str(EPOCH_NUMBER) + " epoha")
    plt.show()
    '''

    acc = model_fit.history['accuracy']
    val_acc = model_fit.history['val_accuracy']
    loss = model_fit.history['loss']
    val_loss = model_fit.history['val_loss']

    title = str(EPOCH_NUMBER)+ " epoha"
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'b', color='#1e2b10', label='Training')
    plt.plot(epochs, val_acc, 'b', color='#00c96b', label='Validation')
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.title("Accuracy, " + title)
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'b', color='#4a0113', label='Training')
    plt.plot(epochs, val_loss, 'b', color='blue', label='Validation')
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Loss, " + title)
    plt.legend()
    plt.show()
    '''
    
if(NUMBER_OF_EMOJIS>1):
    import cv2

    image = cv2.imread(r"testmulti.png")
    number_of_emojis = 4

    w=image.shape[1]
    images = []
    y1=0
    x1=0

    y2=image.shape[0]
    x2=0

    for i in range(0, image.shape[1], int(w/number_of_emojis)+1):
        x2=i+int(w/number_of_emojis)
        crop_image = cv2.imread(r"testmulti.png")[y1:y2, x1:x2]
        images.append(crop_image)
        x1=x2+1

    for im in images:
        a = cv2.resize(im, (64, 64), interpolation = cv2.INTER_AREA)
        x = tf.keras.utils.img_to_array(a)
        val = model.predict(np.stack([x]))

        print("Classes and their indexes:\n", train.class_indices)
        print("\nValues for each class:\n",val)
        value = {i for i in train.class_indices if train.class_indices[i]==np.argmax(val, axis=1)}
        print(value)
