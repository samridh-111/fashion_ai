import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

DATA_DIR = "data/Clothes_Dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

#Train set
train_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,     
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

#val + test
temp_ds = image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,      
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

# Get number of classes
num_classes = len(train_ds.class_names)
print(f"Number of classes: {num_classes}")

temp_batches =tf.data.experimental.cardinality(temp_ds)
val_ds=temp_ds.take(temp_batches//2)
test_ds=temp_ds.skip(temp_batches//2)

AUTOTUNE=tf.data.AUTOTUNE

train_ds=train_ds.prefetch(AUTOTUNE)
val_ds=val_ds.prefetch(AUTOTUNE)
test_ds=test_ds.prefetch(AUTOTUNE)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

# Add classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

fashion_model = Model(inputs=base_model.input, outputs=predictions)

fashion_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Model created successfully!")
print(f"Total parameters: {fashion_model.count_params()}")

fashion_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

fashion_model.evaluate(test_ds)