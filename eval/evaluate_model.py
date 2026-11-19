import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

def evaluate_best_model():
    # 1. Paden op de BSC
    data_dir = "./mame_data/data"
    csv_path = "./mame_data/MAMe_dataset.csv"
    model_path = "01_regularized_model.keras"

    batch_size = 128
    img_height, img_width = 256, 256

    print(f"Laden van model: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print("Inlezen van dataset CSV...")
    df = pd.read_csv(csv_path)
    all_classes = sorted(df['Medium'].unique().tolist())

    train_df = df[df['Subset'] == 'train'].copy()
    val_df = df[df['Subset'] == 'val'].copy()
    test_df = df[df['Subset'] == 'test'].copy()

    datagen = ImageDataGenerator(rescale=1./255)

    def make_eval_generator(subset_df):
        return datagen.flow_from_dataframe(
            dataframe=subset_df,
            directory=data_dir,
            x_col="Image file",
            y_col="Medium",
            target_size=(img_height, img_width),
            batch_size=batch_size,
            classes=all_classes,
            class_mode="categorical",
            shuffle=False  # Cruciaal zodat y_true en y_pred exact oplijnen
        )

    train_gen = make_eval_generator(train_df)
    val_gen = make_eval_generator(val_df)
    test_gen = make_eval_generator(test_df)

    # 2. Eerlijke evaluatie in inferentiemodus (zonder augmentatie/dropout)
    print("\n--- Evaluatie op normale inferentie ---")
    train_loss, train_acc = model.evaluate(train_gen, verbose=1)
    val_loss, val_acc = model.evaluate(val_gen, verbose=1)
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)

    # 3. Voorspellingen op de test set voor gedetailleerde metrics
    print("\nVoorspellingen genereren op de test set...")
    test_gen.reset()
    y_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.array(test_gen.classes)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=all_classes, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    # 4. Overzicht printen en opslaan voor je verslag
    summary_text = (
        f"=== Evaluatieresultaten: {model_path} ===\n"
        f"Inference Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}\n"
        f"Inference Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}\n"
        f"Inference Test  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}\n"
        f"Test Balanced Accuracy (Macro Recall): {bal_acc:.4f}\n\n"
        f"=== Classification Report (Test Set) ===\n"
        f"{report}\n"
    )

    print("\n" + summary_text)

    with open("01_test_evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Confusion matrix opslaan als CSV (makkelijk om lokaal netjes te plotten)
    cm_df = pd.DataFrame(cm, index=all_classes, columns=all_classes)
    cm_df.to_csv("01_test_confusion_matrix.csv")
    print("Opgeslagen: 01_test_evaluation_report.txt en 01_test_confusion_matrix.csv")

if __name__ == "__main__":
    evaluate_best_model()