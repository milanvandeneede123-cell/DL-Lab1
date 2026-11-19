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
        tf.keras.layers.RandomContrast(0.1),
    ], name="data_augmentation")

    # Iets sterkere L2 regularisatie om schommelingen en overfitting af te remmen
    reg_conv = tf.keras.regularizers.l2(3e-4)
    reg_dense = tf.keras.regularizers.l2(5e-4)

    # 3. Geregulariseerde Architectuur (compactere Dense-kop + gerichte L2 & Dropout)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(256, 256, 3)),
        data_augmentation,

        # Block 1 (256x256 -> 128x128)
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_regularizer=reg_conv),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 2 (128x128 -> 64x64)
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=reg_conv),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 3 (64x64 -> 32x32)
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=reg_conv),
        tf.keras.layers.SpatialDropout2D(0.1),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Block 4 (32x32 -> 8x8)
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=reg_conv),
        tf.keras.layers.SpatialDropout2D(0.15),
        tf.keras.layers.MaxPooling2D((4, 4)),

        # Classifier: kleinere Dense-laag (128 i.p.v. 256) om parameter-explosie en overfitting te beperken
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.45),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=reg_dense),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(29, activation='softmax')
    ])

    # 4. Compileer het model met iets lagere initiële learning rate voor gladdere convergentie
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, clipnorm=1.0),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 5. Callbacks: CSVLogger + ReduceLROnPlateau + EarlyStopping
    model_name = "02_regularized_model"
    callbacks = [
        tf.keras.callbacks.CSVLogger(f"{model_name}_history.csv", append=False),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=12,
            verbose=1,
            min_delta=1e-3,
            restore_best_weights=True
        )
    ]

    print("Start training (regularized scenario v2)...")

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=70,
        callbacks=callbacks
    )

    model.save(f"{model_name}.keras")
    print(f"Model en geschiedenis opgeslagen als {model_name}.keras en {model_name}_history.csv")

    # 6. Eerlijke evaluatie op normale inferentie (zonder augmentatie/dropout)
    print("\nEvaluatie op normale inferentie (zonder training-ruis):")
    train_eval_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=data_dir,
        x_col="Image file",
        y_col="Medium",
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )
    train_loss, train_acc = model.evaluate(train_eval_generator, verbose=0)
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"Inference Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Inference Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    train_regularized_model()