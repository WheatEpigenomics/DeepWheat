import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
import logging
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import matplotlib.pyplot as plt

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("predict_model.log"),
        logging.StreamHandler()
    ]
)

# Custom attention layer
class FeatureAttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FeatureAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = self.add_weight(
            shape=(1, input_shape[-1]),
            initializer='random_normal',
            trainable=True,
            name="attention_weights"
        )

    def call(self, inputs):
        attention = tf.nn.softmax(self.attention_weights, axis=-1)
        return inputs * attention

# Define the Root Mean Squared Error (RMSE) metric
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Load data
def load_training_parameters(params_path):
    try:
        with open(params_path, "r") as f:
            best_params = json.load(f)
        logging.info(f"Loaded best parameters from {params_path}")
        return best_params
    except Exception as e:
        logging.error(f"Error loading best parameters from {params_path}: {e}")
        raise

def load_model_weights(model, weights_path):
    try:
        model.load_weights(weights_path)
        logging.info(f"Loaded model weights from {weights_path}")
    except Exception as e:
        logging.error(f"Error loading model weights from {weights_path}: {e}")
        raise

def load_prediction_data(epigenetics_path, sequence_path):
    try:
        # load HDF5 data set 
        with h5py.File(sequence_path, 'r') as hf:
            sequence_input_data = hf['dataset_2'][:]
        with h5py.File(epigenetics_path, 'r') as hf:
            epigenetics_input_data = hf['dataset_1'][:]
        logging.info(f"Loaded epigenetics data from {epigenetics_path}, shape: {epigenetics_input_data.shape}")
        logging.info(f"Loaded sequence data from {sequence_path}, shape: {sequence_input_data.shape}")
    except Exception as e:
        logging.error(f"Error loading HDF5 data: {e}")
        raise

    try:
        # select region 
        ATAC_index = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))
        K27ac_index = np.concatenate((np.arange(11000, 14500), np.arange(17500, 18200)))
        K27_index = np.concatenate((np.arange(21000, 24500), np.arange(27500, 28200)))
        K36_index = np.concatenate((np.arange(31000, 34500), np.arange(37500, 38200)))
        K4_index = np.concatenate((np.arange(41000, 44500), np.arange(47500, 48200)))
        epi_index = np.concatenate((ATAC_index, K27ac_index, K27_index, K36_index, K4_index))
        seq_index = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))

        epigenetics_input_data = epigenetics_input_data[:, epi_index]
        sequence_input_data = sequence_input_data[:, seq_index, :]
        logging.info(f"Processed epigenetics data shape: {epigenetics_input_data.shape}")
        logging.info(f"Processed sequence data shape: {sequence_input_data.shape}")
    except Exception as e:
        logging.error(f"Error processing indices: {e}")
        raise

    try:
        # Normalize data
        sequence_input_data = (sequence_input_data - np.mean(sequence_input_data, axis=0)) / np.std(sequence_input_data, axis=0)
        epigenetics_input_data = (epigenetics_input_data - np.mean(epigenetics_input_data, axis=0)) / np.std(epigenetics_input_data, axis=0)
        logging.info("Data normalization completed.")
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        raise

    return epigenetics_input_data, sequence_input_data

# Define the residual block and the model building function
def identity_residual_block(input_layer, num_filters, kernel_size, L2, drop_rate, activation):
    shortcut = input_layer
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation=None,
                      kernel_regularizer=tf.keras.regularizers.l2(L2), padding="same")(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Dropout(drop_rate)(x)
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation=None,
                      kernel_regularizer=tf.keras.regularizers.l2(L2), padding="same")(x)
    x = layers.BatchNormalization()(x)

    if K.int_shape(input_layer)[-1] != num_filters:
        shortcut = layers.Conv2D(filters=num_filters, kernel_size=(1, 1), activation=None,
                                 kernel_regularizer=tf.keras.regularizers.l2(L2), padding="same")(input_layer)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation(activation)(x)
    x = layers.Dropout(drop_rate)(x)
    return x

def build_model_from_params(params, epigenetics_input_shape, sequence_input_shape):
    try:
        # epigenome branch
        epi_input = keras.Input(shape=(epigenetics_input_shape,), 
                                name="epi_input", dtype=tf.float32)
        epi_input_reshape = layers.Reshape(target_shape=(5, epigenetics_input_shape // 5, 1))(epi_input)
        epi_input_transpose = layers.Permute((2, 1, 3))(epi_input_reshape)
        epi_input_padding = layers.ZeroPadding2D(padding=((10, 9), (0, 0)))(epi_input_transpose)

        epi_conv = layers.Conv2D(filters=params['kernel_num1'], kernel_size=(params['kernel_size'], 5), activation=None,
                                 kernel_regularizer=tf.keras.regularizers.l2(params['L2']), padding="valid", use_bias=False)(epi_input_padding)
        epi_conv_bn = layers.BatchNormalization()(epi_conv)
        epi_conv_act = layers.Activation(params['activation'])(epi_conv_bn)
        epi_conv_dp = layers.Dropout(params['drop_rate'])(epi_conv_act)

        epi_input_padding_l2 = layers.ZeroPadding2D(padding=((5, 4), (0, 0)))(epi_conv_dp)
        epi_conv_l2 = layers.Conv2D(filters=params['kernel_num2'], kernel_size=(params['kernel_size'], 1), activation=None,
                                    kernel_regularizer=tf.keras.regularizers.l2(params['L2']), padding="valid")(epi_input_padding_l2)
        epi_conv_bn_l2 = layers.BatchNormalization()(epi_conv_l2)
        epi_conv_act_l2 = layers.Activation(params['activation'])(epi_conv_bn_l2)
        epi_conv_dp_l2 = layers.Dropout(params['drop_rate'])(epi_conv_act_l2)
        epi_pool_l2 = layers.AveragePooling2D(pool_size=(5, 1), strides=5, padding="valid")(epi_conv_dp_l2)

        epi_attention = FeatureAttentionLayer()(epi_pool_l2)

        # Sequence branch
        seq_input = keras.Input(shape=(sequence_input_shape[0], sequence_input_shape[1], 1),
                                name="seq_input", dtype=tf.float32)
        seq_input_padding = layers.ZeroPadding2D(padding=((10, 9), (0, 0)))(seq_input)

        seq_conv = layers.Conv2D(filters=params['kernel_num1'], kernel_size=(params['kernel_size'], sequence_input_shape[1]), activation=None,
                                 kernel_regularizer=tf.keras.regularizers.l2(params['L2']), padding="valid", use_bias=False)(seq_input_padding)
        seq_conv_bn = layers.BatchNormalization()(seq_conv)
        seq_conv_act = layers.Activation(params['activation'])(seq_conv_bn)
        seq_conv_dp = layers.Dropout(params['drop_rate'])(seq_conv_act)

        seq_input_padding_l2 = layers.ZeroPadding2D(padding=((5, 4), (0, 0)))(seq_conv_dp)
        seq_conv_l2 = layers.Conv2D(filters=params['kernel_num2'], kernel_size=(params['kernel_size'], 1), activation=None,
                                    kernel_regularizer=tf.keras.regularizers.l2(params['L2']), padding="valid")(seq_input_padding_l2)
        seq_conv_bn_l2 = layers.BatchNormalization()(seq_conv_l2)
        seq_conv_act_l2 = layers.Activation(params['activation'])(seq_conv_bn_l2)
        seq_conv_dp_l2 = layers.Dropout(params['drop_rate'])(seq_conv_act_l2)
        seq_pool_l2 = layers.AveragePooling2D(pool_size=(5, 1), strides=5, padding="valid")(seq_conv_dp_l2)

        seq_attention = FeatureAttentionLayer()(seq_pool_l2)

        # merge branch
        merged_input = layers.Concatenate(axis=2)([epi_attention, seq_attention])

        # Residual block
        x = merged_input
        for _ in range(params['num_res_blocks']):
            x = identity_residual_block(
                x,
                num_filters=params['kernel_num1'],
                kernel_size=(params['res_block_kernel_size'], params['res_block_kernel_size']),
                L2=params['L2'],
                drop_rate=params['drop_rate'],
                activation=params['activation']
            )

        merged_flat = layers.Flatten()(x)
        dense1 = layers.Dense(256, activation=None, kernel_regularizer=tf.keras.regularizers.l2(params['L2']))(merged_flat)
        dense1_bn = layers.BatchNormalization()(dense1)
        dense1_act = layers.Activation(params['activation'])(dense1_bn)
        dense1_dp = layers.Dropout(params['drop_rate'])(dense1_act)

        dense2 = layers.Dense(64, activation=None, kernel_regularizer=tf.keras.regularizers.l2(params['L2']))(dense1_dp)
        dense2_bn = layers.BatchNormalization()(dense2)
        dense2_act = layers.Activation(params['activation'])(dense2_bn)
        dense2_dp = layers.Dropout(params['drop_rate'])(dense2_act)

        dense3 = layers.Dense(16, activation=None, kernel_regularizer=tf.keras.regularizers.l2(params['L2']))(dense2_dp)
        dense3_bn = layers.BatchNormalization()(dense3)
        dense3_act = layers.Activation(params['activation'])(dense3_bn)

        output_1d = layers.Dense(1, activation=None, kernel_regularizer=tf.keras.regularizers.l2(params['L2']))(dense3_act)
        output = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(output_1d)

        model = keras.Model(inputs=[epi_input, seq_input], outputs=output)
        return model

# Predict
def main():
    try:
        # File path configuration
        best_params_path = "best_params.json"  # Best parameters file saved by the training 
        model_weights_path = "./model.best.h5"  # Best model weights saved by the training 
        need_predict_seq_path = "need_predict-seq.h5"
        need_predict_epi_path = "need_predict-epi.h5"

        # Check files
        required_files = [best_params_path, model_weights_path, need_predict_seq_path, need_predict_epi_path]
        for file_path in required_files:
            if not os.path.isfile(file_path):
                logging.error(f"Required file not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # Load best parameters
        best_params = load_training_parameters(best_params_path)

        # Get load branch
        try:
            with h5py.File(need_predict_epi_path, 'r') as hf:
                need_predict_epi = hf['dataset_1'][:]
            with h5py.File(need_predict_seq_path, 'r') as hf:
                need_predict_seq = hf['dataset_2'][:]
            
            # Determine the input shape based on the preprocessing during training
            ATAC_index = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))
            K27ac_index = np.concatenate((np.arange(11000, 14500), np.arange(17500, 18200)))
            K27_index = np.concatenate((np.arange(21000, 24500), np.arange(27500, 28200)))
            K36_index = np.concatenate((np.arange(31000, 34500), np.arange(37500, 38200)))
            K4_index = np.concatenate((np.arange(41000, 44500), np.arange(47500, 48200)))
            epi_index = np.concatenate((ATAC_index, K27ac_index, K27_index, K36_index, K4_index))
            seq_index = np.concatenate((np.arange(1000, 4500), np.arange(7500, 8200)))

            epigenetics_input_shape = len(epi_index)
            sequence_input_shape = (len(seq_index), need_predict_seq.shape[2])

            # Build model 
            model = build_model_from_params(best_params, epigenetics_input_shape, sequence_input_shape)

            # Compile the model
            optimizer_name = best_params['optimizer']
            learning_rate = best_params['learning_rate']
            if optimizer_name == 'Adam':
                optimizer = Adam(learning_rate=learning_rate)
            elif optimizer_name == 'RMSprop':
                optimizer = RMSprop(learning_rate=learning_rate)
            elif optimizer_name == 'SGD':
                optimizer = SGD(learning_rate=learning_rate)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")

            model.compile(
                loss='mse',
                optimizer=optimizer,
                metrics=['mse', 'mae', 'mape', root_mean_squared_error, tfa.metrics.RSquare(name="r_square")]
            )

            # Load the weights
            load_model_weights(model, model_weights_path)
        except Exception as e:
            logging.error(f"Error building and loading model: {e}")
            raise

        # Load prediction data
        try:
            epigenetics_input_data, sequence_input_data = load_prediction_data(need_predict_epi_path, need_predict_seq_path)
            logging.info("Prediction data loaded and preprocessed.")
        except Exception as e:
            logging.error(f"Error loading and preprocessing prediction data: {e}")
            raise

        # Predict
        try:
            predictions = model.predict([epigenetics_input_data, sequence_input_data])
            logging.info(f"Predictions completed, shape: {predictions.shape}")
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            raise

        # Save predictions
        try:
            df_predictions = pd.DataFrame({'Predicted_Expression': predictions.flatten()})
            df_predictions.to_csv("need_predict_results.csv", index=False)
            logging.info("Predictions saved to need_predict_results.csv")
        except Exception as e:
            logging.error(f"Error saving predictions: {e}")
            raise

    except Exception as e:
        logging.critical(f"Critical error in prediction process: {e}")
        raise

if __name__ == "__main__":
    main()
