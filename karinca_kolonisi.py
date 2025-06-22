import tensorflow as tf
import numpy as np
import os
import cv2
import gc
import time
import matplotlib.pyplot as plt
from random import shuffle

# 📁 Klasör ayarları
base_dir = "/content/data_for_train/data_for_train"
image_width = 128
image_height = 128

# 🔧 ACO parametre alanı
param_space = {
    'filters': [128,256,512],
    'dropout': [0.2, 0.3, 0.4],
    'dense_units': [256, 512, 1024],
    'kernel_size': [(2,2),(3, 3), (5, 5)],
    'activation': ['relu'],
    'optimizer': ['adam'],
    'batch_size': [24,32,48]
}


alpha = 1.0
beta = 2.0
q0 = 0.7
num_ants = 5
num_iterations =10
evaporation_rate = 0.1

def initialize_table(param_space):
    return {key: {str(v): 1.0 for v in values} for key, values in param_space.items()}

pheromones = initialize_table(param_space)
heuristics = initialize_table(param_space)

def create_label(folder):
    return np.array([1, 0]) if folder == "Gray mold" else np.array([0, 1])

def dataset_load(folder_name):
    dataset = []
    path = os.path.join(base_dir, folder_name)
    for folder in os.listdir(path):
        for image in os.listdir(os.path.join(path, folder)):
            try:
                label = create_label(folder)
                path_image = os.path.join(path, folder, image)
                img_data = cv2.imdecode(np.fromfile(path_image, np.uint8), -1)
                if img_data is None:
                    continue
                img_data = cv2.resize(img_data, (image_width, image_height))
                img_data = img_data / 255.0
                if len(img_data.shape) == 3 and img_data.shape[2] == 3:
                    dataset.append([img_data, label])
            except:
                continue
    shuffle(dataset)
    return dataset

def select_parameters(pheromones, heuristics):
    selected_params = {}
    for key in pheromones:
        options = list(pheromones[key].keys())
        tau = np.array([pheromones[key][o] ** alpha for o in options])
        eta = np.array([heuristics[key][o] ** beta for o in options])
        probs = tau * eta
        probs /= probs.sum()
        idx = np.argmax(probs) if np.random.rand() < q0 else np.random.choice(len(options), p=probs)
        val = options[idx]
        if key == 'kernel_size': val = eval(val)
        elif key in ['filters', 'dense_units', 'batch_size']: val = int(val)
        elif key == 'dropout': val = float(val)
        selected_params[key] = val
    return selected_params

def update_pheromones(pheromones, best_params, best_score):
    for key in pheromones:
        for k in pheromones[key]:
            pheromones[key][k] *= (1 - evaporation_rate)
        pheromones[key][str(best_params[key])] += best_score

def evaluate_model(params, images_train, label_train, images_valid, label_valid, images_test, label_test):
    try:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(params['filters'], params['kernel_size'], strides=(4, 4), activation=params['activation'], input_shape=(image_width, image_height, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(3, strides=2),

            tf.keras.layers.Conv2D(params['filters'], (3, 3), padding='same', activation=params['activation']),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(3, strides=2),

            tf.keras.layers.Conv2D(params['filters'], (3, 3), padding='same', activation=params['activation']),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(params['filters'], (3, 3), padding='same', activation=params['activation']),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(params['filters'], (3, 3), padding='same', activation=params['activation']),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(3, strides=2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(params['dense_units'], activation=params['activation']),
            tf.keras.layers.Dropout(params['dropout']),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        model.compile(optimizer=params['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])

        start_time = time.time()

        history = model.fit(
            images_train[:100], label_train[:100],
            validation_data=(images_valid[:30], label_valid[:30]),
            epochs=70, batch_size=params['batch_size'], verbose=0
        )

        elapsed_time = time.time() - start_time
        val_loss, val_acc = model.evaluate(images_valid[:30], label_valid[:30], verbose=0)
        test_loss, test_acc = model.evaluate(images_test[:30], label_test[:30], verbose=0)

        tf.keras.backend.clear_session()
        gc.collect()
        return val_acc, val_loss, test_acc, test_loss, history, elapsed_time
    except:
        tf.keras.backend.clear_session()
        gc.collect()
        return 0.0, 0.0, 0.0, 0.0, None, 0


train = dataset_load('train')
valid = dataset_load('val')
test = dataset_load('test')

images_train = np.array([i[0] for i in train])
label_train = np.array([i[1] for i in train])
images_valid = np.array([i[0] for i in valid])
label_valid = np.array([i[1] for i in valid])
images_test = np.array([i[0] for i in test])
label_test = np.array([i[1] for i in test])


best_score = 0
best_params = None
best_history = None
best_time = 0
best_loss = 0
best_test_acc = 0
best_test_loss = 0

for it in range(num_iterations):
    print(f"\n🌀 Iteration {it+1}")
    for ant in range(num_ants):
        params = select_parameters(pheromones, heuristics)
        val_acc, val_loss, test_acc, test_loss, history, elapsed_time = evaluate_model(params, images_train, label_train, images_valid, label_valid, images_test, label_test)
        print(f"🐜 Ant {ant+1}: {params} → Val_Acc: {val_acc:.4f} | Val_Loss: {val_loss:.4f} | Test_Acc: {test_acc:.4f} | Test_Loss: {test_loss:.4f} | Time: {elapsed_time:.2f}s")
        if val_acc > best_score:
            best_score = val_acc
            best_params = params
            best_history = history
            best_time = elapsed_time
            best_loss = val_loss
            best_test_acc = test_acc
            best_test_loss = test_loss
    update_pheromones(pheromones, best_params, best_score)

# 📢 Sonuç
print("\n🎯 Best Params:", best_params)
print(f"✅ Best Validation Accuracy: {best_score:.4f}")
print(f"📉 Best Validation Loss: {best_loss:.6f}")
print(f"🧪 Best Test Accuracy: {best_test_acc:.4f}")
print(f"🧪 Best Test Loss: {best_test_loss:.6f}")
print(f"🕒 Training Time: {best_time:.2f} seconds")