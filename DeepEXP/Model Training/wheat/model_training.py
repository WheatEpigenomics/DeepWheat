import numpy as np
import pandas as pd
import h5py
import gc
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import tensorflow_addons as tfa
import optuna
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import os
import json
import logging
import tensorflow as tf

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("select_canshu.log"),
        logging.StreamHandler()
    ]
)

# Define the Root Mean Squared Error (RMSE) metric
def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Data loading and preprocessing
def load_data(epigenetics_path, sequence_path, label_path, index_path, random_order_path):
    try:
        # load gene expression data
        label_data = pd.read_csv(label_path, delimiter="\t").values
        logging.info(f"Labels loaded from {label_path}, shape: {label_data.shape}")
    except Exception as e:
        logging.error(f"Error loading label data from {label_path}: {e}")
        raise

    try:
        # load HDF5 data
        with h5py.File(sequence_path, 'r') as hf:
            sequence_input_data = hf['dataset_2'][:]
        with h5py.File(epigenetics_path, 'r') as hf:
            epigenetics_input_data = hf['dataset_1'][:]
        logging.info(f"Epigenetics data shape: {epigenetics_input_data.shape}")
        logging.info(f"Sequence data shape: {sequence_input_data.shape}")
    except Exception as e:
        logging.error(f"Error loading HDF5 data: {e}")
        raise

    try:
        # Select specific features from certain regions based on the requirements
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
        # split Test2
        with open(index_path, 'r') as file:
            G4700_indices = [int(line.strip()) for line in file]
        indices = np.array(G4700_indices)  

        test_2_epi = epigenetics_input_data[indices]
        test_2_seq = sequence_input_data[indices]
        test_2_label = label_data[indices]
        new_epi_data = np.delete(epigenetics_input_data, indices, axis=0)
        new_seq_data = np.delete(sequence_input_data, indices, axis=0)
        new_label_data = np.delete(label_data, indices, axis=0)
    except Exception as e:
        logging.error(f"Error splitting test2 data: {e}")
        raise

    try:
        # random data
        if os.path.isfile(random_order_path):
            random_order = pd.read_csv(random_order_path, delimiter="\t", header=None)[0].values.tolist()
            logging.info(f"Loaded random order from {random_order_path}")
        else:
            random_order = list(range(new_epi_data.shape[0]))
            np.random.shuffle(random_order)
            np.savetxt(random_order_path, random_order, fmt='%i')
            logging.info(f"Generated and saved random order to {random_order_path}")

        new_epi_data = new_epi_data[random_order]
        new_seq_data = new_seq_data[random_order]
        new_label_data = new_label_data[random_order]
        gc.collect()
    except Exception as e:
        logging.error(f"Error shuffling data: {e}")
        raise

    try:
        # Split the data into training, validation, and test sets
        train_data_epi = new_epi_data[:70000]
        train_data_seq = new_seq_data[:70000]
        train_data_label = new_label_data[:70000].squeeze()
        valid_data_epi = new_epi_data[70000:92000]
        valid_data_seq = new_seq_data[70000:92000]
        valid_data_label = new_label_data[70000:92000].squeeze()
        test_data_epi = new_epi_data[92000:]
        test_data_seq = new_seq_data[92000:]
        test_data_label = new_label_data[92000:].squeeze()
        test2_data_epi = test_2_epi
        test2_data_seq = test_2_seq
        test2_data_label = test_2_label.squeeze()
    except Exception as e:
        logging.error(f"Error splitting datasets: {e}")
        raise

    try:
        # Sample a subset of the training data for hyperparameter tuning
        train_size = train_data_epi.shape[0]
        sample_ratio = 0.2
        sample_size = int(train_size * sample_ratio)
        sample_indices = np.random.choice(train_size, sample_size, replace=False)
        sub_train_data_epi = train_data_epi[sample_indices]
        sub_train_data_seq = train_data_seq[sample_indices]
        sub_train_data_label = train_data_label[sample_indices].squeeze()

        logging.info("Sub-training data shapes:")
        logging.info(f"Epigenetics: {sub_train_data_epi.shape}")
        logging.info(f"Sequence: {sub_train_data_seq.shape}")
        logging.info(f"Labels: {sub_train_data_label.shape}")
    except Exception as e:
        logging.error(f"Error sampling sub-training data: {e}")
        raise

    return {
        'train': (train_data_epi, train_data_seq, train_data_label),
        'valid': (valid_data_epi, valid_data_seq, valid_data_label),
        'test': (test_data_epi, test_data_seq, test_data_label),
        'test2': (test2_data_epi, test2_data_seq, test2_data_label),
        'sub_train': (sub_train_data_epi, sub_train_data_seq, sub_train_data_label)
    }

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

def build_model(kernel_num1, kernel_num2, kernel_size, L2, drop_rate,
                num_res_blocks, res_block_kernel_size, optimizer, activation,
                epigenetics_input_shape, sequence_input_shape):

    try:
        # epigenomic branches
        epi_input = keras.Input(shape=(epigenetics_input_shape,),
                                name="epi_input", dtype=tf.float32)
        epi_input_reshape = layers.Reshape(target_shape=(5, epigenetics_input_shape // 5, 1))(epi_input)
        epi_input_transpose = layers.Permute((2, 1, 3))(epi_input_reshape)
        epi_input_padding = layers.ZeroPadding2D(padding=((10, 9), (0, 0)))(epi_input_transpose)

        epi_conv = layers.Conv2D(filters=kernel_num1, kernel_size=(kernel_size, 5), activation=None,
                                 kernel_regularizer=tf.keras.regularizers.l2(L2), padding="valid", use_bias=False)(epi_input_padding)
        epi_conv_bn = layers.BatchNormalization()(epi_conv)
        epi_conv_act = layers.Activation(activation)(epi_conv_bn)
        epi_conv_dp = layers.Dropout(drop_rate)(epi_conv_act)

        epi_input_padding_l2 = layers.ZeroPadding2D(padding=((5, 4), (0, 0)))(epi_conv_dp)
        epi_conv_l2 = layers.Conv2D(filters=kernel_num2, kernel_size=(kernel_size, 1), activation=None,
                                    kernel_regularizer=tf.keras.regularizers.l2(L2), padding="valid")(epi_input_padding_l2)
        epi_conv_bn_l2 = layers.BatchNormalization()(epi_conv_l2)
        epi_conv_act_l2 = layers.Activation(activation)(epi_conv_bn_l2)
        epi_conv_dp_l2 = layers.Dropout(drop_rate)(epi_conv_act_l2)
        epi_pool_l2 = layers.AveragePooling2D(pool_size=(5, 1), strides=5, padding="valid")(epi_conv_dp_l2)

       
        epi_output = epi_pool_l2

        # sequences branches
        seq_input = keras.Input(shape=(sequence_input_shape[0], sequence_input_shape[1], 1),
                                name="seq_input", dtype=tf.float32)
        seq_input_padding = layers.ZeroPadding2D(padding=((10, 9), (0, 0)))(seq_input)

        seq_conv = layers.Conv2D(filters=kernel_num1, kernel_size=(kernel_size, sequence_input_shape[1]), activation=None,
                                 kernel_regularizer=tf.keras.regularizers.l2(L2), padding="valid", use_bias=False)(seq_input_padding)
        seq_conv_bn = layers.BatchNormalization()(seq_conv)
        seq_conv_act = layers.Activation(activation)(seq_conv_bn)
        seq_conv_dp = layers.Dropout(drop_rate)(seq_conv_act)

        seq_input_padding_l2 = layers.ZeroPadding2D(padding=((5, 4), (0, 0)))(seq_conv_dp)
        seq_conv_l2 = layers.Conv2D(filters=kernel_num2, kernel_size=(kernel_size, 1), activation=None,
                                    kernel_regularizer=tf.keras.regularizers.l2(L2), padding="valid")(seq_input_padding_l2)
        seq_conv_bn_l2 = layers.BatchNormalization()(seq_conv_l2)
        seq_conv_act_l2 = layers.Activation(activation)(seq_conv_bn_l2)
        seq_conv_dp_l2 = layers.Dropout(drop_rate)(seq_conv_act_l2)
        seq_pool_l2 = layers.AveragePooling2D(pool_size=(5, 1), strides=5, padding="valid")(seq_conv_dp_l2)

        
        seq_output = seq_pool_l2

        # Merge branches
        merged_input = layers.Concatenate(axis=2)([epi_output, seq_output])

        # Residual blocks
        x = merged_input
        for _ in range(num_res_blocks):
            x = identity_residual_block(
                x,
                num_filters=kernel_num1,
                kernel_size=(res_block_kernel_size, res_block_kernel_size),
                L2=L2,
                drop_rate=drop_rate,
                activation=activation
            )

        merged_flat = layers.Flatten()(x)
        dense1 = layers.Dense(256, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2))(merged_flat)
        dense1_bn = layers.BatchNormalization()(dense1)
        dense1_act = layers.Activation(activation)(dense1_bn)
        dense1_dp = layers.Dropout(drop_rate)(dense1_act)

        dense2 = layers.Dense(64, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2))(dense1_dp)
        dense2_bn = layers.BatchNormalization()(dense2)
        dense2_act = layers.Activation(activation)(dense2_bn)
        dense2_dp = layers.Dropout(drop_rate)(dense2_act)

        dense3 = layers.Dense(16, activation=None, kernel_regularizer=tf.keras.regularizers.l2(L2))(dense2_dp)
        dense3_bn = layers.BatchNormalization()(dense3)
        dense3_act = layers.Activation(activation)(dense3_bn)

        # Output layer with 'relu' activation to ensure non-negative predictions
        output_1d = layers.Dense(1, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(L2))(dense3_act)
        output = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(output_1d)

        model = keras.Model(inputs=[epi_input, seq_input], outputs=output)
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=[
                'mse',
                'mae',
                'mape',
                root_mean_squared_error,
                tfa.metrics.RSquare(name="r_square")
            ]
        )
        return model
    except Exception as e:
        logging.error(f"Error building model: {e}")
        raise

# Optuna hyperparameter optimization
def objective(trial, sub_train_data):
    try:
        sub_train_epi, sub_train_seq, sub_train_label = sub_train_data

        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-2, log=True)
        kernel_num1 = trial.suggest_int('kernel_num1', 32, 128)
        kernel_num2 = trial.suggest_int('kernel_num2', 64, 256)
        kernel_size = trial.suggest_int('kernel_size', 10, 60)
        L2 = trial.suggest_float('L2', 1e-6, 1e-2, log=True)
        drop_rate = trial.suggest_float('drop_rate', 0.1, 0.8)
        num_res_blocks = trial.suggest_int('num_res_blocks', 2, 16)
        res_block_kernel_size = trial.suggest_int('res_block_kernel_size', 2, 11)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
        activation = trial.suggest_categorical('activation', ['relu', 'elu', 'selu', 'tanh'])

        if optimizer_name == 'Adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'SGD':
            optimizer = SGD(learning_rate=learning_rate)

        # Get the input shape
        epigenetics_input_shape = sub_train_epi.shape[1]
        sequence_input_shape = sub_train_seq.shape[1:]  # e.g. (len, channels)

        # Build the model
        model = build_model(
            kernel_num1, kernel_num2, kernel_size, L2, drop_rate,
            num_res_blocks, res_block_kernel_size, optimizer, activation,
            epigenetics_input_shape, sequence_input_shape
        )
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=[
                'mse',
                'mae',
                'mape',
                root_mean_squared_error,
                tfa.metrics.RSquare(name="r_square")
            ]
        )

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        val_r_square_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(sub_train_epi), 1):
            try:
                train_epi, val_epi = sub_train_epi[train_idx], sub_train_epi[val_idx]
                train_seq, val_seq = sub_train_seq[train_idx], sub_train_seq[val_idx]
                train_label, val_label = sub_train_label[train_idx], sub_train_label[val_idx]

                fold_model = build_model(
                    kernel_num1, kernel_num2, kernel_size, L2, drop_rate,
                    num_res_blocks, res_block_kernel_size, optimizer, activation,
                    epigenetics_input_shape, sequence_input_shape
                )
                fold_model.compile(
                    loss='mse',
                    optimizer=optimizer,
                    metrics=[
                        'mse',
                        'mae',
                        'mape',
                        root_mean_squared_error,
                        tfa.metrics.RSquare(name="r_square")
                    ]
                )

                fold_early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_r_square', patience=15, mode='max', restore_best_weights=True
                )

                history = fold_model.fit(
                    x=[train_epi, train_seq],
                    y=train_label,
                    validation_data=([val_epi, val_seq], val_label),
                    batch_size=64,
                    epochs=100,
                    verbose=0,
                    callbacks=[fold_early_stopping]
                )
                val_r_square_scores.append(max(history.history['val_r_square']))
                logging.info(f"Trial {trial.number}, Fold {fold}: val_r_square={max(history.history['val_r_square'])}")

                # release memory
                K.clear_session()
                gc.collect()
            except Exception as e:
                logging.error(f"Error in Trial {trial.number}, Fold {fold}: {e}")
                return -np.inf

        mean_val_r2 = np.mean(val_r_square_scores)
        logging.info(f"Trial {trial.number} completed with mean val R2: {mean_val_r2}")
        return mean_val_r2
    except Exception as e:
        logging.error(f"Error in objective function: {e}")
        return -np.inf

# Main training process
def main():
    try:
        # load file
        h5d_file_path = "./spike-best.model.h5"
        label_file_path = "../spike_gene_expression.tpm"
        random_order_path = "./random-select-canshu.txt"
        index_file_path = '../test2_index.txt'
        sequence_h5_path = '../spike_sequence.h5'
        epigenetics_h5_path = '../spike_epi.h5'

        # check data
        required_files = [label_file_path, sequence_h5_path, epigenetics_h5_path, index_file_path]
        for file_path in required_files:
            if not os.path.isfile(file_path):
                logging.error(f"Required file not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")

        # load data
        data = load_data(
            epigenetics_path=epigenetics_h5_path,
            sequence_path=sequence_h5_path,
            label_path=label_file_path,
            index_path=index_file_path,
            random_order_path=random_order_path
        )

        train_data_epi, train_data_seq, train_data_label = data['train']
        valid_data_epi, valid_data_seq, valid_data_label = data['valid']
        test_data_epi, test_data_seq, test_data_label = data['test']
        test2_data_epi, test2_data_seq, test2_data_label = data['test2']
        sub_train_data = data['sub_train']

        # define Optuna
        def optuna_objective(trial):
            return objective(trial, sub_train_data)

        study = optuna.create_study(direction="maximize")
        study.optimize(optuna_objective, n_trials=80)  

        if study.best_trial:
            best_params = study.best_params
            logging.info(f"Best parameters: {best_params}")
        else:
            logging.error("No trials were completed successfully.")
            raise RuntimeError("Optuna study failed to find any valid trials.")

        # Save the best parameters
        try:
            with open("best_params.json", "w") as f:
                json.dump(best_params, f)
            logging.info("Best parameters saved to best_params.json")
        except Exception as e:
            logging.error(f"Error saving best parameters to best_params.json: {e}")
            raise

        gc.collect()

        # Retrain the final model using the best hyperparameters
        try:
            if best_params['optimizer'] == 'Adam':
                optimizer = Adam(learning_rate=best_params['learning_rate'])
            elif best_params['optimizer'] == 'RMSprop':
                optimizer = RMSprop(learning_rate=best_params['learning_rate'])
            elif best_params['optimizer'] == 'SGD':
                optimizer = SGD(learning_rate=best_params['learning_rate'])

            epigenetics_input_shape = train_data_epi.shape[1]
            sequence_input_shape = train_data_seq.shape[1:]

            final_model = build_model(
                best_params['kernel_num1'], best_params['kernel_num2'],
                best_params['kernel_size'], best_params['L2'], best_params['drop_rate'],
                best_params['num_res_blocks'], best_params['res_block_kernel_size'], optimizer,
                best_params['activation'],
                epigenetics_input_shape, sequence_input_shape
            )
            final_model.compile(
                loss='mse',
                optimizer=optimizer,
                metrics=[
                    'mse',
                    'mae',
                    'mape',
                    root_mean_squared_error,
                    tfa.metrics.RSquare(name="r_square")
                ]
            )
        except Exception as e:
            logging.error(f"Error building final model: {e}")
            raise

        try:
            checkpoint = keras.callbacks.ModelCheckpoint(
                h5d_file_path, monitor='val_r_square', verbose=1, save_best_only=True, mode='max'
            )

            reduce_lr_on_plateau = keras.callbacks.ReduceLROnPlateau(
                monitor='val_r_square', factor=0.5, patience=5, min_lr=1e-7, verbose=1
            )

            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_r_square', patience=30, mode='max', restore_best_weights=True
            )
        except Exception as e:
            logging.error(f"Error setting up callbacks: {e}")
            raise

        # Training final model
        try:
            history = final_model.fit(
                x=[train_data_epi, train_data_seq],
                y=train_data_label,
                validation_data=([valid_data_epi, valid_data_seq], valid_data_label),
                batch_size=64,
                epochs=1000,
                callbacks=[checkpoint, reduce_lr_on_plateau, early_stopping],
                verbose=1
            )
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

        # Release memory and load the best weights
        try:
            K.clear_session()
            gc.collect()
            final_model.load_weights(h5d_file_path)
            logging.info(f"Loaded best model weights from {h5d_file_path}")
        except Exception as e:
            logging.error(f"Error loading model weights from {h5d_file_path}: {e}")
            raise

        # Test set evaluation
        try:
            expression_gene_prediction = final_model.predict([test_data_epi, test_data_seq])

            mse = tf.keras.metrics.mean_squared_error(test_data_label.flatten(), expression_gene_prediction.flatten()).numpy()
            mae = tf.keras.metrics.mean_absolute_error(test_data_label.flatten(), expression_gene_prediction.flatten()).numpy()
            mape = tf.keras.metrics.mean_absolute_percentage_error(test_data_label.flatten(), expression_gene_prediction.flatten()).numpy()
            rmse = np.sqrt(mse)
            r2 = r2_score(test_data_label.flatten(), expression_gene_prediction.flatten())
            pearson_corr = scipy.stats.pearsonr(test_data_label.flatten(), expression_gene_prediction.flatten())

            logging.info(f"Test Set MSE: {mse}")
            logging.info(f"Test Set MAE: {mae}")
            logging.info(f"Test Set MAPE: {mape}")
            logging.info(f"Test Set RMSE: {rmse}")
            logging.info(f"Test Set R² Score: {r2}")
            logging.info(f"Test Set Pearson Correlation: {pearson_corr}")
        except Exception as e:
            logging.error(f"Error during test set evaluation: {e}")
            raise

        # Test2 evaluation
        try:
            test2_gene_prediction = final_model.predict([test2_data_epi, test2_data_seq])

            mse_test2 = tf.keras.metrics.mean_squared_error(test2_data_label.flatten(), test2_gene_prediction.flatten()).numpy()
            mae_test2 = tf.keras.metrics.mean_absolute_error(test2_data_label.flatten(), test2_gene_prediction.flatten()).numpy()
            mape_test2 = tf.keras.metrics.mean_absolute_percentage_error(test2_data_label.flatten(), test2_gene_prediction.flatten()).numpy()
            rmse_test2 = np.sqrt(mse_test2)
            r2_test2 = r2_score(test2_data_label.flatten(), test2_gene_prediction.flatten())
            pearson_corr_test2 = scipy.stats.pearsonr(test2_data_label.flatten(), test2_gene_prediction.flatten())

            logging.info(f"Test Set 2 MSE: {mse_test2}")
            logging.info(f"Test Set 2 MAE: {mae_test2}")
            logging.info(f"Test Set 2 MAPE: {mape_test2}")
            logging.info(f"Test Set 2 RMSE: {rmse_test2}")
            logging.info(f"Test Set 2 R² Score: {r2_test2}")
            logging.info(f"Test Set 2 Pearson Correlation: {pearson_corr_test2}")
        except Exception as e:
            logging.error(f"Error during test set 2 evaluation: {e}")
            raise

        # Save the test set results
        try:
            df_test = pd.DataFrame({
                'True_Label': test_data_label.flatten(),
                'Predicted_Expression': expression_gene_prediction.flatten(),
                'MSE': mse,
                'MAE': mae,
                'MAPE': mape,
                'RMSE': rmse,
                'R2_Score': r2,
                'Pearson_Correlation': pearson_corr[0],
                'Pearson_p_value': pearson_corr[1]
            })
            df_test.to_csv("test_results-1.csv", index=False)
            logging.info("Test results saved to test_results-1.csv")

            df_test2 = pd.DataFrame({
                'True_Label_Test2': test2_data_label.flatten(),
                'Predicted_Expression_Test2': test2_gene_prediction.flatten(),
                'MSE_Test2': mse_test2,
                'MAE_Test2': mae_test2,
                'MAPE_Test2': mape_test2,
                'RMSE_Test2': rmse_test2,
                'R2_Score_Test2': r2_test2,
                'Pearson_Correlation_Test2': pearson_corr_test2[0],
                'Pearson_p_value_Test2': pearson_corr_test2[1]
            })
            df_test2.to_csv("test2_results-2.csv", index=False)
            logging.info("Test Set 2 results saved to test2_results-2.csv")
        except Exception as e:
            logging.error(f"Error saving test results: {e}")
            raise

        # Plot the training process
        try:
            def plot_training(history):
                output_dir = "./Fig-out-select-canshu"
                os.makedirs(output_dir, exist_ok=True)

                plt.figure(figsize=(18, 10))

                # Plot the loss function
                plt.subplot(2, 3, 1)
                plt.plot(history.history['loss'], label='Training Loss (MSE)')
                plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss (MSE)')
                plt.legend()

                # Plot the Mean Squared Error (MSE)
                plt.subplot(2, 3, 2)
                plt.plot(history.history['mse'], label='Training MSE')
                plt.plot(history.history['val_mse'], label='Validation MSE')
                plt.xlabel('Epoch')
                plt.ylabel('MSE')
                plt.title('Training and Validation MSE')
                plt.legend()

                # Plot the Mean Absolute Error (MAE)
                plt.subplot(2, 3, 3)
                plt.plot(history.history['mae'], label='Training MAE')
                plt.plot(history.history['val_mae'], label='Validation MAE')
                plt.xlabel('Epoch')
                plt.ylabel('MAE')
                plt.title('Training and Validation MAE')
                plt.legend()

                # Plot the Mean Absolute Percentage Error (MAPE)
                plt.subplot(2, 3, 4)
                plt.plot(history.history['mape'], label='Training MAPE')
                plt.plot(history.history['val_mape'], label='Validation MAPE')
                plt.xlabel('Epoch')
                plt.ylabel('MAPE')
                plt.title('Training and Validation MAPE')
                plt.legend()

                # Plot the Root Mean Squared Error (RMSE)
                plt.subplot(2, 3, 5)
                plt.plot(history.history['root_mean_squared_error'], label='Training RMSE')
                plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
                plt.xlabel('Epoch')
                plt.ylabel('RMSE')
                plt.title('Training and Validation RMSE')
                plt.legend()

                # Plot the coefficient of determination (R²)
                plt.subplot(2, 3, 6)
                plt.plot(history.history['r_square'], label='Training R²')
                plt.plot(history.history['val_r_square'], label='Validation R²')
                plt.xlabel('Epoch')
                plt.ylabel('R² Score')
                plt.title('Training and Validation R² Score')
                plt.legend()

                plt.tight_layout()
                pdf_path = os.path.join(output_dir, "training_history-random-1.pdf")
                plt.savefig(pdf_path)
                plt.close()
                logging.info(f"Training history saved to {pdf_path}")

            plot_training(history)
        except Exception as e:
            logging.error(f"Error plotting training history: {e}")
            raise

    except Exception as e:
        logging.critical(f"Critical error in main training loop: {e}")
        raise

if __name__ == "__main__":
    main()



