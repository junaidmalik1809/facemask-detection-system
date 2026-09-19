import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"
TEST_DIR  = BASE_DIR / "data" / "test"
MODEL_PATH = BASE_DIR / "models" / "mask_detector.keras"

# -------------------------------------------------
# Training parameters
# -------------------------------------------------
INIT_LR = 1e-4
EPOCHS = 8
BATCH_SIZE = 4
IMG_SIZE = (128, 128)

def build_model():
    """Build a lighter MobileNetV2 model for low memory systems."""
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(128, 128, 3)      # Important: match IMG_SIZE
    )

    base_model.trainable = False

    head = base_model.output
    head = AveragePooling2D(pool_size=(4, 4))(head)
    head = Flatten(name="flatten")(head)
    head = Dense(64, activation="relu")(head)      # smaller dense layer
    head = Dropout(0.4)(head)
    head = Dense(2, activation="softmax")(head)

    model = Model(inputs=base_model.input, outputs=head)
    return model
def main():
    print("[INFO] Loading images...")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    print("Class indices:", train_generator.class_indices)
    # Expected: {'with_mask': 0, 'without_mask': 1}

    print("[INFO] Building model...")
    model = build_model()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=INIT_LR),
        metrics=["accuracy"]
    )

    # Save the best model during training
    checkpoint = ModelCheckpoint(
        str(MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    print("[INFO] Training model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=test_generator,
        validation_steps=len(test_generator),
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    print(f"[INFO] Training finished. Best model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()