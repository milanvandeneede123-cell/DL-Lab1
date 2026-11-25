import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_resnet_model():
    # 1. Paths to the data on the BSC
    data_dir = "./mame_data/data"
    csv_path = "./mame_data/MAMe_dataset.csv"

    print("Reading dataset CSV...")
    df = pd.read_csv(csv_path)

    train_df = df[df['Subset'] == 'train'].copy()
    val_df = df[df['Subset'] == 'val'].copy()

    batch_size = 128
    img_height, img_width = 256, 256

    print("Creating datasets with ImageDataGenerator.flow_from_dataframe...")
    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=data_dir,
        x_col="Image file",
        y_col="Medium",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=data_dir,
        x_col="Image file",
        y_col="Medium",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    # 2. Model (TO WRITE: functional API, ResNet with skip connections)
    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    for block in range(4):  # 4 blocks with increasing filter sizes
        for inter_block in range([1, 1, 1, 1][block]):  # residual blocks
            shortcut = x
            strides = (1, 1) 
            if inter_block == 0 and block != 0:  # downsample at the start of each block except the first
                strides = (2, 2)
                shortcut = tf.keras.layers.Conv2D(64*2**block, (1, 1), strides=(2, 2), padding='same')(x)
                shortcut = tf.keras.layers.BatchNormalization(momentum=0.9)(shortcut)
            x = tf.keras.layers.Conv2D(64*2**block, (3, 3), strides=strides, padding='same')(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Conv2D(64*2**block, (3, 3), strides=(1, 1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
            x = tf.keras.layers.Add()([x, shortcut])
            x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(29, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # 3. Compile the model (same settings as the baseline)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 4. Callbacks: only the CSV logger for now (unique model name, do not overwrite other runs)
    model_name = "34_resnet_model"
    callbacks = [tf.keras.callbacks.CSVLogger(f"{model_name}_history.csv", append=False)]

    print("Start training...")

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=callbacks
    )

    model.save(f"{model_name}.keras")
    print(f"Model and history saved as {model_name}.keras and {model_name}_history.csv")

if __name__ == "__main__":
    train_resnet_model()
