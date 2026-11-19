import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_underfit_model():
    # 1. Pas dit pad aan naar de werkelijke locatie van je afbeeldingen op de BSC
    data_dir = "./mame_data/data" 
    csv_path = "./mame_data/MAMe_dataset.csv"
    
    print("Inlezen van dataset CSV...")
    df = pd.read_csv(csv_path)
    
    # Splits op basis van de officiële 'Subset' kolom
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

    # 2. Stabiele Underfitting Architectuur (Zonder BatchNorm & zonder GlobalAveragePooling)
    # Slechts ~16.000 parameters: leert direct vanaf epoch 1, maar plafonneert rond 25-35%
    initial_width = 8

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(256, 256, 3)),
        
        # Block 1 (256x256 -> 64x64)
        tf.keras.layers.Conv2D(initial_width, (3, 3), padding='same', activation='elu'),
        tf.keras.layers.MaxPooling2D((4, 4)),
        
        # Block 2 (64x64 -> 16x16)
        tf.keras.layers.Conv2D(initial_width * 2, (3, 3), padding='same', activation='elu'),
        tf.keras.layers.MaxPooling2D((4, 4)),
        
        # Block 3 (16x16 -> 4x4)
        tf.keras.layers.Conv2D(initial_width * 4, (3, 3), padding='same', activation='elu'),
        tf.keras.layers.MaxPooling2D((4, 4)),
        
        # Classifier (4x4x32 = 512 features -> 29 klassen)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(29, activation='softmax') 
    ])

    # 3. Compileer het model met lr=5e-4 voor een stabiele, gladde leercurve
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # 4. Callback om resultaten op te slaan per epoch
    model_name = "05_underfit_model"
    csv_logger = tf.keras.callbacks.CSVLogger(f"{model_name}_history.csv", append=False)

    print("Start training (underfitting scenario)...")
    
    # 5. Train 30 epochs zodat het afvlakken (plateau) van de curve duidelijk zichtbaar is
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