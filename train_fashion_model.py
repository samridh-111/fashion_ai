import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras import layers, Model

# -----------------------------
# 1. Basic config
# -----------------------------
DATA_DIR = "data/Clothes_Dataset"
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS_HEAD = 5       
EPOCHS_FINE_TUNE = 3 


train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,     
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,     
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=IMG_SIZE + (3,),
    pooling="avg",  
)

base_model.trainable = False 


inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image_input")

x = data_augmentation(inputs)
x = preprocess_input(x)              
x = base_model(x, training=False)
x = layers.Dense(256, activation="relu", name="embedding")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

fashion_model = Model(inputs, outputs, name="fashion_resnet_model")

fashion_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

fashion_model.summary()


print("\n🚀 Training classification head...")
history_head = fashion_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
)

print("\n🔧 Fine-tuning top layers of ResNet...")

base_model.trainable = True

# Freeze most layers, unfreeze last 30
for layer in base_model.layers[:-30]:
    layer.trainable = False

fashion_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_finetune = fashion_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE_TUNE,
)


os.makedirs("model", exist_ok=True)

full_model_path = "model/fashion_model_resnet50.keras"
fashion_model.save(full_model_path)
print(f"\n✅ Saved full fine-tuned model to {full_model_path}")

# Build embedding-only model 
embedding_layer = fashion_model.get_layer("embedding")
embedding_model = Model(
    inputs=fashion_model.input,
    outputs=embedding_layer.output,
    name="fashion_embedding_model",
)

embed_path = "model/fashion_embedding_resnet50.keras"
embedding_model.save(embed_path)
print(f"✅ Saved embedding model to {embed_path}")
