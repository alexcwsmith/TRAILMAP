import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, Conv3DTranspose, concatenate, \
    Cropping3D, Input, SpatialDropout3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.metrics import Precision, Recall
import tensorflow_addons as tfa

input_dim = 64
output_dim = 64

#%%
def weighted_bce_loss(axon_w=1.5, bg_w=0.2, art_w=0.8, edge_w=0.05):
    eps = K.epsilon()
    def loss(y_true, y_pred):
        y_true_axon = y_true[...,0]  # [B,D,H,W]
        y_pred_axon = y_pred[...,0]  # [B,D,H,W]
    
        # manual per‐voxel BCE, no implicit reductions
        bce = -(
            y_true_axon * tf.math.log(y_pred_axon + eps)
          + (1 - y_true_axon) * tf.math.log(1 - y_pred_axon + eps)
        )  # shape [B,D,H,W]
        
        w = (y_true[:, :, :, : , 0]*axon_w +
             y_true[:, :, :, : , 1]*bg_w   +
             y_true[:, :, :, : , 2]*art_w  +
             y_true[:, :, :, : , 3]*edge_w)
        return tf.reduce_sum(bce * w) / (tf.reduce_sum(w) + eps)
    return loss


def weighted_binary_crossentropy(y_true, y_pred):
 #   loss = create_weighted_binary_crossentropy(1.5, 0.2, 0.8, 0.05)(y_true, y_pred)
    loss = weighted_bce_loss(1.5, 0.2, 0.8, 0.05)(y_true, y_pred)

    return loss


def adjusted_accuracy(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1, keepdims=True)

    mask = K.equal(weights, 1)

    axons_true = y_true[:, :, :, :, 0]
    axons_true = K.expand_dims(axons_true, -1)

    mask_true = tf.boolean_mask(axons_true, mask)
    mask_pred = tf.boolean_mask(y_pred, mask)

    return K.mean(K.equal(mask_true, K.round(mask_pred)))


def axon_precision(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    mask_true = tf.boolean_mask(y_true[:, :, :, :, 0], mask)
    mask_pred = tf.boolean_mask(y_pred[:, :, :, :, 0], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def axon_recall(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    mask_true = tf.boolean_mask(y_true[:, :, :, :, 0], mask)
    mask_pred = tf.boolean_mask(y_pred[:, :, :, :, 0], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(mask_true, 0, 1)))

    recall = true_positives / (actual_positives + K.epsilon())

    return recall

def axon_f1(y_true, y_pred, smooth=1e-6, thresh=0.5):
    # flatten
    y_t = tf.reshape(y_true[...,0], [-1])
    y_p = tf.reshape(tf.cast(y_pred[...,0] >= thresh, tf.float32), [-1])

    # intersection, sums
    inter = tf.reduce_sum(y_t * y_p)
    p_sum = tf.reduce_sum(y_t) + tf.reduce_sum(y_p)

    return (2. * inter + smooth) / (p_sum + smooth)

def artifact_precision(y_true, y_pred):
    weights = y_true[:, :, :, :, 2]

    mask = tf.equal(weights, 1)
    mask_true = tf.boolean_mask(y_true[:, :, :, :, 2], mask)
    mask_pred = tf.boolean_mask(1 - y_pred[:, :, :, :, 0], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def f1_score(y_true, y_pred):

    precision = axon_precision(y_true, y_pred)
    recall = axon_recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def edge_axon_precision(y_true, y_pred):
    weights = tf.reduce_sum(y_true, axis=-1)

    mask = tf.equal(weights, 1)

    mask_true = tf.boolean_mask(y_true[:, :, :, :, 0], mask)
    mask_pred = tf.boolean_mask(y_pred[:, :, :, :, 0], mask)
    mask_edge_true = tf.boolean_mask(y_true[:, :, :, :, 3], mask)

    true_positives = K.sum(K.round(K.clip(mask_true * mask_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(mask_pred, 0, 1)))

    edge_count = K.sum(K.round(K.clip(mask_edge_true * mask_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon() - edge_count)

    return precision


#%%
# 0. Mixed-precision policy (if you have an NVIDIA GPU)
mixed_precision.set_global_policy('mixed_float16')

def get_unet3d(input_dim, 
               base_filters=32, 
               learning_rate=1e-4, 
               weight_decay=1e-5, 
               drop_rate=0.1,
               clipnorm=.01,
               group_size=16):
    inp = Input((input_dim, input_dim, input_dim, 1))
  #  GN = lambda: tfa.layers.GroupNormalization(groups=group_size, axis=-1)

    # ENCODER
    #Level 1
    x = Conv3D(base_filters, 3, padding='same', activation='relu')(inp) # 32
    x = BatchNormalization()(x)
    x = SpatialDropout3D(drop_rate)(x)
    x = Conv3D(base_filters*2, 3, padding='same', activation='relu')(x) #64
    x = BatchNormalization()(x)
    skip1 = SpatialDropout3D(drop_rate)(x)
    
    #Level 2
    x = MaxPooling3D()(skip1)
    x = Conv3D(base_filters*2, 3, padding='same', activation='relu')(x) #64
    x = BatchNormalization()(x)
    x = SpatialDropout3D(drop_rate)(x)
    x = Conv3D(base_filters*4, 3, padding='same', activation='relu')(x) #128
    x = BatchNormalization()(x)
    skip2 = SpatialDropout3D(drop_rate)(x)

    #Level 3
    x = MaxPooling3D()(skip2)
    x = Conv3D(base_filters*4, 3, padding='same', activation='relu')(x) #128
    x = BatchNormalization()(x)
    x = SpatialDropout3D(drop_rate)(x)
    x = Conv3D(base_filters*8, 3, padding='same', activation='relu')(x) #256
    x = BatchNormalization()(x)
    skip3 = SpatialDropout3D(drop_rate)(x)

    #Level 4
    x = MaxPooling3D()(skip3)
    x = Conv3D(base_filters*8, 3, padding='same', activation='relu')(x)  #256
    x = BatchNormalization()(x)
    x = Conv3D(base_filters*16, 3, padding='same', activation='relu')(x) #512
    x = BatchNormalization()(x)
    x = SpatialDropout3D(drop_rate)(x)
    skip4 = SpatialDropout3D(drop_rate)(x)

    # DECODER helper (no cropping!)
    def up_block(x, skip, filters):
        x = Conv3DTranspose(filters*2, 2, strides=2, padding='same', activation='relu')(x)
        x = concatenate([x, skip])
        x = Conv3D(filters, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout3D(drop_rate)(x)
        x = Conv3D(filters, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        return x

    x = up_block(x, skip3, 256)
    x = up_block(x, skip2, 128)
    x = up_block(x, skip1, 64)

    outputs = Conv3D(1, (1,1,1), activation='sigmoid', dtype='float32')(x)
    model = Model(inp, outputs)

  #  axon_precision = Precision(name="axon_precision", thresholds=0.5)
   # axon_recall    = Recall(name="axon_recall",    thresholds=0.5)
   # f1 = tfa.metrics.F1Score(num_classes=1, threshold=0.5, average="micro")
   
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clipnorm=clipnorm
    )
    model.compile(
        optimizer=optimizer,
#        loss=weighted_binary_crossentropy,
        loss=weighted_bce_loss(1.5, 0.2, 0.8, 0.05),
        metrics=[axon_precision, axon_recall],
    )
    return model
