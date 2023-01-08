
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

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
model_fit = model.fit(train, epochs=10, validation_data=valid)

#Load test image
img = tf.keras.utils.load_img("test.png", target_size=(64,64))
x = tf.keras.utils.img_to_array(img)
images = np.stack([x])

#Get similarity values
#Values are in range [-6144, 6144] because max value is 64x64x3 (image height x image width x (red + green + blue))
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
    #abs(np.min(np.array(val[0]).astype(np.int64)))
    y_axis.append((y + 6144)/100)

plt.barh(x_axis, y_axis, color=colors)
plt.xlabel("Similarity (%)")
plt.title("")
plt.show()

