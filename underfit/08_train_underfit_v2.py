import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_underfit_model():
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

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(256, 256, 3)),

        # Block 1 (256x256 -> 128x128)
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 2 (128x128 -> 64x64)
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Classifier: 64x64x64 -> 64 gemiddelde features -> 29 klassen
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(29, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    model_name = "03_underfit_model"
    csv_logger = tf.keras.callbacks.CSVLogger(f"{model_name}_history.csv", append=False)

    print("Start training (underfitting scenario v3)...")

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[csv_logger]
    )

    model.save(f"{model_name}.keras")
    print(f"Model en geschiedenis opgeslagen als {model_name}.keras en {model_name}_history.csv")

if __name__ == "__main__":
    train_underfit_model()