import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout,concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Callback ,CSVLogger
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from data_generator import DataGenerator

# Prepare IDs and data generators
input_dir = './Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
train_and_test_ids = [f.name.split('_')[-1] for f in os.scandir(input_dir) if f.is_dir()]
train_and_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2,random_state=42)
train_ids, test_ids = train_test_split(train_and_test_ids, test_size=0.15, random_state=42)
train_generator = DataGenerator(train_ids, input_dir)
val_generator = DataGenerator(val_ids, input_dir)
test_generator = DataGenerator(test_ids, input_dir)

def conv_block(inputs, num_filters, kernel_initializer, dropout_rate=None):
    conv = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(inputs)
    conv = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv)
    if dropout_rate:
        conv = Dropout(dropout_rate)(conv)
    return conv

def up_conv_block(inputs, skip_connection, num_filters, kernel_initializer):
    up = UpSampling2D(size=(2, 2))(inputs)
    up = Conv2D(num_filters, 2, activation='relu', padding='same', kernel_initializer=kernel_initializer)(up)
    merge = concatenate([skip_connection, up], axis=3)
    conv = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(merge)
    conv = Conv2D(num_filters, 3, activation='relu', padding='same', kernel_initializer=kernel_initializer)(conv)
    return conv

def unet(ker_init, dropout):
    inputs = Input((128, 128, 4))
    # Downsampling path
    conv1 = conv_block(inputs, 32, ker_init)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 64, ker_init)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 128, ker_init)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 256, ker_init)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_block(pool4, 512, ker_init, dropout)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    # Bottleneck
    conv6 = conv_block(pool5, 1024, ker_init, dropout)
    # Upsampling path
    conv7 = up_conv_block(conv6, conv5, 512, ker_init)
    conv8 = up_conv_block(conv7, conv4, 256, ker_init)
    conv9 = up_conv_block(conv8, conv3, 128, ker_init)
    conv10 = up_conv_block(conv9, conv2, 64, ker_init)
    conv11 = up_conv_block(conv10, conv1, 32, ker_init)
    outputs = Conv2D(4, 1, activation='softmax')(conv11)
    return Model(inputs=inputs, outputs=outputs)


model = unet('he_normal', 0.2)

# Dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    class_num = 4
    total_loss = 0
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:, :, :, i])
        y_pred_f = K.flatten(y_pred[:, :, :, i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        total_loss += loss
    total_loss /= class_num
    return total_loss

# Define per class evaluation of dice coef
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(K.abs(y_true[:, :, :, 1] * y_pred[:, :, :, 1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 1])) + K.sum(K.square(y_pred[:, :, :, 1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(K.abs(y_true[:, :, :, 2] * y_pred[:, :, :, 2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 2])) + K.sum(K.square(y_pred[:, :, :, 2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    intersection = K.sum(K.abs(y_true[:, :, :, 3] * y_pred[:, :, :, 3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:, :, :, 3])) + K.sum(K.square(y_pred[:, :, :, 3])) + epsilon)

# Computing Precision
def precision(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Computing Sensitivity
def sensitivity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

# Computing Specificity
def specificity(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        'accuracy',
        tf.keras.metrics.MeanIoU(num_classes=4),
        dice_coef,
        precision,
        sensitivity,
        specificity,
        dice_coef_necrotic,
        dice_coef_edema,
        dice_coef_enhancing
    ]
)
# Callbacks
csv_logger = CSVLogger('training.log', separator=',', append=False)
callbacks = [
    EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001, verbose=1),
    ModelCheckpoint(filepath='model_{epoch:02d}-{val_loss:.6f}.weights.h5', verbose=1, save_best_only=True, save_weights_only=True),
    csv_logger
]
# Train the model
history =  model.fit(train_generator,
                    epochs=35,
                    steps_per_epoch=len(train_ids),
                    callbacks=callbacks,
                    validation_data = val_generator,
                    verbose=1
                    )
model.save("brain_tumor.h5")
# # Evaluate the model on the test set
test_metrics = model.evaluate(test_generator, verbose=1)
print(f'Test set metrics: {test_metrics}')

