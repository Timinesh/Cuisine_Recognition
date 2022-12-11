import os
Haryana_dir = os.path.join("C:/Users/timin/Downloads/data/keras/images/Haryana")

South_Indian_dir = os.path.join("C:/Users/timin/Downloads/data/keras/images/South_Indian")

Italian_dir = os.path.join("C:/Users/timin/Downloads/data\keras/images/Italian")

Japanese_dir = os.path.join("C:/Users/timin/Downloads/data/keras/images/Japanese")

train_Haryana_names = os.listdir(Haryana_dir)
print(train_Haryana_names[:4])

train_South_Indian_names = os.listdir(South_Indian_dir)
print(train_South_Indian_names[:4])

train_Italian_names = os.listdir(Italian_dir)
print(train_Italian_names[:4])

train_Japanese_names = os.listdir(Japanese_dir)
print(train_Japanese_names[:4])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_Haryana_pix = [os.path.join(Haryana_dir, fname) 
                for fname in train_Haryana_names[pic_index-8:pic_index]]
next_South_Indian_pix = [os.path.join(South_Indian_dir, fname) 
                for fname in train_South_Indian_names[pic_index-8:pic_index]]

print ("Haryana...")
print()
for i, img_path in enumerate(next_Haryana_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')
  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

print ("South Indian...")
print()
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
for i, img_path in enumerate(next_South_Indian_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')
  
  img = mpimg.imread(img_path)
  plt.imshow(img)
  
plt.show()

batch_size = 1

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        "C:/Users/timin/Downloads/data/keras/images", 
        target_size=(200, 200), 
        batch_size=batch_size,
        classes = ['Haryana','South_Indian','Italian','Japanese'],
        class_mode='categorical')

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])


total_sample=train_generator.n

n_epochs = 30
history = model.fit_generator(
        train_generator, 
        steps_per_epoch=int(total_sample/batch_size),  
        epochs=n_epochs,
        verbose=1)

plt.figure(figsize=(7,4))
plt.plot([i+1 for i in range(n_epochs)],history.history['acc'],'-o',c='k',lw=2,markersize=9)
plt.grid(True)
plt.title("Training accuracy with epochs\n",fontsize=18)
plt.xlabel("Training epochs",fontsize=15)
plt.ylabel("Training accuracy",fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()