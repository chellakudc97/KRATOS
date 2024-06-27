import os
import tensorflow as tf
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, InputLayer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from skimage import io, exposure
from skimage.transform import resize
from sklearn.decomposition import PCA


# Model creation with layers
def create_model(input_shape, num_classes):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', name='dense'),
        Dense(num_classes, activation='softmax', name='Classifier')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Training the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history

# Evaluation
def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    return test_loss, test_acc

# Extracting features for visualization
def extract_and_reshape_features(model, X):
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer('dense').output)
    features = feature_extractor.predict(X)
    num_samples = features.shape[0]
    flattened_features = features.reshape(num_samples, -1)
    return flattened_features

# Visualizing clusters with t-SNE
def visualize_clusters(features, labels, title='t-SNE Visualization of Clusters'):
    num_samples = features.shape[0]
    perplexity = min(30, num_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, learning_rate=300)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels, palette=sns.color_palette('hsv', len(np.unique(labels))))
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='best')
    plt.show()

# Saliency map for interpretability
def get_gradients(model, inputs, class_index):
    inputs = tf.convert_to_tensor(inputs)
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions = model(inputs)[:, class_index]
    gradients = tape.gradient(predictions, inputs)
    return gradients.numpy()

def compute_saliency_map(model, inputs, labels):
    saliency_maps = []
    for i, input_image in enumerate(inputs):
        class_index = labels[i]
        gradients = get_gradients(model, np.expand_dims(input_image, axis=0), class_index)
        saliency_maps.append(gradients[0])
    return np.array(saliency_maps)

def display_preprocessing_steps(original, normalized, gamma_adjusted, contrast_adjusted):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Original', 'Resized & Normalized', 'Gamma Adjusted', 'Contrast Adjusted']
    images = [original, normalized, gamma_adjusted, contrast_adjusted]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Loading data
def load_images_from_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.png'):
            img_path = os.path.join(folder, filename)
            try:
                img = io.imread(img_path, as_gray=True)
                img_resized = resize(img, (128, 128))
                if img_resized.ndim == 2:
                    img_resized = np.expand_dims(img_resized, axis=-1)
                    img_normalized = (img_resized - np.min(img_resized)) / (np.max(img_resized) - np.min(img_resized))
                    img_gamma = exposure.adjust_gamma(img_normalized, gamma=3)
                    img_contrast = exposure.equalize_adapthist(img_gamma, clip_limit=0.01)
                images.append(img_contrast)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        if len(images) == 1:
            display_preprocessing_steps(img, img_normalized, img_gamma, img_contrast)
    return np.array(images)

def load_labels(folder):
    labels = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.png'):
            label = extract_label_from_filename(filename)
            labels.append(label)
    return np.array(labels)

def preprocess_CT_data(base_path):
    classes = ['adenocarcinoma', 'normal']
    all_images = {'train': [], 'test': [], 'valid': []}
    all_labels = {'train': [], 'test': [], 'valid': []}

    for split in ['train', 'test', 'valid']:
        for label, class_name in enumerate(classes):
            class_folder = os.path.join(base_path, split, class_name)
            images = load_images_from_folder(class_folder)
            labels = [label] * len(images)
            all_images[split].extend(images)
            all_labels[split].extend(labels)

    for split in ['train', 'test', 'valid']:
        all_images[split] = np.array(all_images[split])
        all_labels[split] = np.array(all_labels[split])

    return all_images['train'], all_labels['train'], all_images['valid'], all_labels['valid'], all_images['test'], all_labels['test']

def extract_label_from_filename(filename):
    return filename.split('_')[0]

def main(data_type, file_path):
    if data_type == 'CT':
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_CT_data(file_path)
        num_classes = 2

    input_shape = X_train.shape[1:]
    model = create_model(input_shape, num_classes)

    train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    evaluate_model(model, X_test, y_test)

    features = extract_and_reshape_features(model, X_test)
    visualize_clusters(features, y_test, title=f't-SNE Visualization of {data_type} Clusters')

    saliency_maps = compute_saliency_map(model, X_test, y_test)
    flattened_saliency_maps = saliency_maps.reshape((saliency_maps.shape[0], -1))
    visualize_clusters(flattened_saliency_maps, y_test, title=f'Saliency Map Visualization of {data_type} Clusters')

    
data_type = 'CT'
file_path = '..\\archive\\Data'
main(data_type, file_path)