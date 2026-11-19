import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_regularized_model():
    # 1. Pad naar data op de BSC
    data_dir = "./mame_data/data"
    csv_path = "./mame_data/MAMe_dataset.csv"

    print("Inlezen van dataset CSV...")
    df = pd.read_csv(csv_path)

    train_df = df[df['Subset'] == 'train'].copy()
    val_df = df[df['Subset'] == 'val'].copy()

    batch_size = 128
    img_height, img_width = 256, 256

    print("Datasets aanmaken via ImageDataGenerator.flow_from_dataframe...")
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

    # 2. Data Augmentation (wordt enkel tijdens training toegepast, niet bij validatie)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Lichte L2 regularisatie
    reg = tf.keras.regularizers.l2(1e-4)

    # 3. Geregulariseerde Architectuur (zelfde basis als het overfitting model)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(256, 256, 3)),
        data_augmentation,

        # Block 1 (256x256 -> 128x128)
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 2 (128x128 -> 64x64)
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 3 (64x64 -> 32x32)
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        tf.keras.layers.SpatialDropout2D(0.1),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 4 (32x32 -> 8x8)
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=reg),
        tf.keras.layers.SpatialDropout2D(0.2),
        tf.keras.layers.MaxPooling2D((4, 4)),

        # Classifier met Dropout vóór de Dense-lagen
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(29, activation='softmax')
    ])

    # 4. Compileer het model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 5. Callbacks: CSVLogger + ReduceLROnPlateau + EarlyStopping
    model_name = "01_regularized_model"
    callbacks = [
        tf.keras.callbacks.CSVLogger(f"{model_name}_history.csv", append=False),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            verbose=1,
            min_delta=1e-3,
            restore_best_weights=True
        )
    ]

    print("Start training (regularized scenario)...")

    # Meer epochs omdat regularisatie het leren bewust vertraagt; EarlyStopping stopt op tijd
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=60,
        callbacks=callbacks
    )

    model.save(f"{model_name}.keras")
    print(f"Model en geschiedenis opgeslagen als {model_name}.keras en {model_name}_history.csv")

if __name__ == "__main__":
    train_regularized_model()