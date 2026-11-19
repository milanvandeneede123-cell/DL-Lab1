import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_regularized_v3():
    # 1. Pad naar data op de BSC
    data_dir = "./mame_data/data"
    csv_path = "./mame_data/MAMe_dataset.csv"

    print("Inlezen van dataset CSV...")
    df = pd.read_csv(csv_path)

    train_df = df[df['Subset'] == 'train'].copy()
    val_df = df[df['Subset'] == 'val'].copy()

    # Grotere batch_size (256 i.p.v. 128) voor nauwkeurigere gradiënten en een gladdere val-curve
    batch_size = 256
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

    # 2. Bewezen Data Augmentation uit v1 (zonder RandomContrast)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.1),
    ], name="data_augmentation")

    # Lichte L2 regularisatie uit v1
    reg = tf.keras.regularizers.l2(1e-4)

    # 3. Winnende Architectuur uit v1 (Dense(256) behouden)
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

        # Classifier
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

    # 5. Vloeiende Learning Rate Schedule (10 epochs constant, daarna exponentiële decay)
    def scheduler(epoch, lr):
        if epoch < 10:
            return 5e-4
        else:
            return float(5e-4 * tf.math.exp(0.07 * (10 - epoch)))

    model_name = "03_regularized_model"
    callbacks = [
        tf.keras.callbacks.CSVLogger(f"{model_name}_history.csv", append=False),
        tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=12,
            start_from_epoch=15,
            verbose=1,
            min_delta=1e-3,
            restore_best_weights=True
        )
    ]

    print("Start training (regularized scenario v3 - smooth convergence)...")

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=60,
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
    train_regularized_v3()