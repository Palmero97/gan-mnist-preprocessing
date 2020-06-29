# -*- coding: utf-8 -*-
"""Dataset transformer.

Apply equally distributed transformations to the instances of a given supervised
data set. In particular, transformations to the MNIST dataset are the ones
included.
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

from torchvision import datasets, transforms
from os.path import join
from transformers import IdTransformer, InvColorTransformer, \
    ProjectedPcaTransformer, NoisyProjectedPcaTransformer, ImgPcaTransformer


# Fix random seed
random.seed(1)

# Path setup
OUTPUT_FOLDER_PATH = "output_data_transformer"

A_TRAIN_FOLDER_PATH = join(OUTPUT_FOLDER_PATH, "trainA")
B_TRAIN_FOLDER_PATH = join(OUTPUT_FOLDER_PATH, "trainB")
A_TEST_FOLDER_PATH = join(OUTPUT_FOLDER_PATH, "testA")
B_TEST_FOLDER_PATH = join(OUTPUT_FOLDER_PATH, "testB")


def partition_and_transform(data, transformers):
    """Partitions a numpy dataset into n equally sized partitions.

    n is equal to the number of transformation functions.

    Args:
        data (numpy): numpy array or matrix to be transformed
        transformers (list): list of transformer objects

    Returns:
        (list) transformed partitions
        (list) original indices of the instances of each partition
    """

    # Create a list containing all data indices
    idx = list(range(data.shape[0]))

    # Shuffle those indices and create N partitions
    random.shuffle(idx)

    n = len(transformers)
    partitions_idx = [idx[i::n] for i in range(n)]

    partitions_transformed = []

    # Iterate over the different partitions and apply the transformations
    for partition_idx, transformer in zip(partitions_idx, transformers):
        partitions_transformed.append(
            transformer.transform(data[partition_idx]))

    return partitions_transformed, partitions_idx


def transform_data_and_label(data, labels, transformers, shuffle=True):
    """Shuffles and applies transformations evenly to the data instances.

    Args:
        data (numpy): numpy array or matrix to be transformed
        labels (numpy): numpy array with the corresponding label of each instance
        transformers (list): list of transformer objects
        shuffle (bool): if True instances are shuffled after applying the
            transformations and concatenated

    Returns:
        (numpy or list[numpy]) transformed data
        (numpy or list[numpy) reorganized labels
    """

    data_tr, data_partitions_idx = partition_and_transform(data,
                                                           transformers)

    # Reorder the original labels
    labels_tr = [labels[idx] for idx in data_partitions_idx]

    if shuffle:
        # Concat the partitions under the same numpy matrix
        data_tr = np.concatenate(data_tr)
        labels_tr = np.concatenate(labels_tr)

        # Shuffle both data so that the transformations are distributed
        idx = list(range(data_tr.shape[0]))
        random.shuffle(idx)
        data_tr = data_tr[idx]

        # Reorder again the labels
        labels_tr = labels_tr[idx]

    return data_tr, labels_tr


def load_mnist_into_numpy(normalize=False):
    """ Download mnist dataset into numpy

    Downloads the mnist dataset, both train and test, as tensors and transforms
    it into x and y numpy arrays. This arrays are returned.

    Returns:
        (numpy): x_train
        (numpy): y_train
        (numpy): x_test
        (numpy): y_test
    """

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])

    # Download and load the training  data
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True,
                              train=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=60000,
                                              shuffle=False)

    # Download and load the test data
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True,
                             train=False, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                             shuffle=False)

    if normalize:
        # Load all the tensors
        x_train, y_train = iter(trainloader).next()

        # Transform to numpy for easy handling and eliminate extra dim
        x_train, y_train = x_train.numpy().reshape((-1, 28, 28)),y_train.numpy()

        # Load all the tensors
        x_test, y_test = iter(testloader).next()

        # Transform to numpy for easy handling and eliminate extra dim
        x_test, y_test = x_test.numpy().reshape((-1, 28, 28)), y_test.numpy()

    else:
        x_train, y_train = trainset.data.numpy(), trainset.targets.numpy()

        x_test, y_test = testset.data.numpy(), testset.targets.numpy()

    return x_train, y_train, x_test, y_test


def plot_img_grid(x, y, title, nrows=4, ncols=4):
    """ Plot instances and labels in grid.

    Plots instances and its given labels are plot in a nrows*nrows grid.

    Args:
        x (numpy): instances
        y (numpy): labels
        title (str): title of the plot
        nrows (int): number of rows in the grid
        ncols (int): number of cols in the grid
    """

    # Plots several example in a grid nrows*ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(6, 6))
    i = 0

    for row in range(nrows):
        for col in range(ncols):
            img = x[i]
            ax[row][col].imshow(img, cmap="Greys")
            fig.show()
            ax[row][col].set_title("label: {}".format(y[i]))
            i += 1
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title, fontsize=14)
    plt.show()


def create_joint_transformed(x_train, y_train, x_test, y_test,
                             transformers,
                             x_train_path, y_train_path,
                             x_test_path, y_test_path,
                             save=True):
    """Apply transformations to the given dataset.

    Apply transformations to the given dataset in a proportional manner
    according to the number of those transformations. At the end if selected
    the transformed datasets are saved in npy format.

    Args:
        x_train (numpy): train instances to be transformed
        y_train (numpy): labels from the train instances
        x_test (numpy): test instances to be transformed
        y_test (numpy): labels from test instances
        transformers (list): list of transformer objects
        x_train_path (str): path where x_train is saved
        y_train_path (str): path where y_train is saved
        x_test_path (str): path where x_test is saved
        y_test_path (str): path where y_test is saved
        save (bool): indicates whether to save or not save the datasets

    Returns:
        (numpy): x_train transformed
        (numpy): y_train updated to match x_train
        (numpy) x_test transformed
        (numpy) y_test updated to match x_test
    """
    x_train_tr_neg, y_train_tr = transform_data_and_label(x_train, y_train,
                                                          transformers)

    x_test_tr_neg, y_test_tr = transform_data_and_label(x_test, y_test,
                                                        transformers)

    if save:
        # Save the transformed data
        if not os.path.exists(OUTPUT_FOLDER_PATH):
            os.makedirs(OUTPUT_FOLDER_PATH)

        np.save(x_train_path, x_train_tr_neg)
        np.save(y_train_path, y_train_tr)

        np.save(x_test_path, x_test_tr_neg)
        np.save(y_test_path, y_test_tr)

    return x_train_tr_neg, y_train_tr, x_test_tr_neg, y_test_tr


def create_joint_custom_mnist(transformers, transf_name="",
                              normalize=False, save=True):
    """Apply custom transformations to mnist dataset.

    Load MNIST dataset and apply custom transformations. At the end if selected
    the transformers datasets are saved in npy format.

    Args:
        transformers (list): list of transformer objects
        transf_name (str): name which appends to the generated files
        normalize (bool): indicates whether to normalize the MNIST dataset
        save (bool): indicates whether to save or not save the datasets

    Returns:
        (numpy): x_train transformed
        (numpy): y_train updated to match x_train
        (numpy) x_test transformed
        (numpy) y_test updated to match x_test
    """

    x_train, y_train, x_test, y_test = \
        load_mnist_into_numpy(normalize=normalize)

    # Check the shape of the training data
    print("Shape of the training data X: {}".format(x_train.shape))
    print("Shape of the training data y: {}".format(y_train.shape))

    # Check the shape of testing data
    print("Shape of the testing data X: {}".format(x_test.shape))
    print("Shape of the testing data y: {}".format(y_test.shape))

    x_train_tr, y_train_tr, x_test_tr, y_test_tr = \
        create_joint_transformed(x_train, y_train, x_test, y_test,
                                 transformers,
                                 join(OUTPUT_FOLDER_PATH,
                                      "mnist_x_train_tr_" + transf_name),
                                 join(OUTPUT_FOLDER_PATH,
                                      "mnist_y_train_tr_" + transf_name),
                                 join(OUTPUT_FOLDER_PATH,
                                      "mnist_x_test_tr_" + transf_name),
                                 join(OUTPUT_FOLDER_PATH,
                                      "mnist_y_test_tr_" + transf_name),
                                 save=save)

    plot_img_grid(x_train_tr, y_train_tr, "MNIST train set transformed")
    plot_img_grid(x_test_tr, y_test_tr, "MNIST test set transformed")

    return x_train_tr, y_train_tr, x_test_tr, y_test_tr


def create_joint_inv_color_mnist(save=True):
    """Apply an inverse color transformation to mnist dataset.

    Load MNIST dataset and apply one transformation. The transformation
    inverts the colors of the images. At the end if selected the transformed
    datasets are saved in npy format.

    Args:
        save (bool): indicates whether to save or not save the datasets

    Returns:
        (numpy): x_train transformed
        (numpy): y_train updated to match x_train
        (numpy) x_test transformed
        (numpy) y_test updated to match x_test
    """

    transformers = [
        IdTransformer(),
        InvColorTransformer(),
    ]

    return create_joint_custom_mnist(transformers,
                                     transf_name="inv_col",
                                     save=save)


def create_joint_proj_pca_mnist(n_components=0.3, save=True):
    """Apply a pca reduction transformation to mnist dataset.

    Load MNIST dataset and apply one transformation. The transformation
    projects the instances into an smaller feature space by using PCA
    decomposition, then from that reduced feature space they are projected back
    to the original domain.

    Args:
        n_components (int): number of components or amount of variance explained
                      for the PCA transformer
        save (bool): indicates whether to save or not save the datasets

    Returns:
        (numpy): x_train transformed
        (numpy): y_train updated to match x_train
        (numpy) x_test transformed
        (numpy) y_test updated to match x_test
    """

    transformers = [
        IdTransformer(),
        ProjectedPcaTransformer(n_components=n_components),
    ]

    return create_joint_custom_mnist(transformers,
                                     transf_name="proj_pca",
                                     normalize=True, save=save)


def create_joint_noisy_proj_pca_mnist(n_components=0.3, noise_factor=0.5,
                                      save=True):
    """Apply a noisy pca reduction transformation to mnist dataset.

    Load MNIST dataset and apply one transformation. The transformation
    projects the instances into an smaller feature space by using PCA
    decomposition. Then, gaussian noise is added to those projections.
    Finally, the projections in the reduced feature space are projected
    back to the original dimension.

    Args:
        n_components (int): number of components or amount of variance explained
                      for the PCA transformer
        noise_factor (int): amount of gaussian noise added
        save (bool): indicates whether to save or not save the datasets

    Returns:
        (numpy): x_train transformed
        (numpy): y_train updated to match x_train
        (numpy) x_test transformed
        (numpy) y_test updated to match x_test
    """

    transformers = [
        IdTransformer(),
        NoisyProjectedPcaTransformer(n_components=n_components,
                                     noise_factor=noise_factor),
    ]

    return create_joint_custom_mnist(transformers,
                                     transf_name="noisy_proj_pca",
                                     normalize=True, save=save)


def create_separated_transformed(data, labels, transformers,
                                 a_folder_path, b_folder_path,
                                 subset=False, subset_size=10000,
                                 tr_img_names=("a", "b"),
                                 paired=False):
    """Apply transformations to the given images and divide them in partitions.

    Transformations are applied to the dataset. The dataset is divided into N
    partitions which are the number of transformations. In the case of the
    paired option only two transformations can be applied. Finally, if wanted
    images are saved into different folders, one for each transformation.


    Args:
        data (numpy): instances
        labels (numpy): labels
        transformers (list): list of transformer objects
        a_folder_path (str): folder where the imgs of the 1st trsf. are saved
        b_folder_path (str): folder where the imgs of the 2nd trsf. are saved
        subset (bool): indicates if only a subset of subset_size instances is
                       used
        subset_size (int): size of the subset if subset equals True
        paired (bool): indicates if instances in both folders should be paired
        tr_img_names (tuple): the different names to assign if unpaired

    Returns:
        (list): contains numpy arrays of the data
                separated regarding the transformations
        (list): contains numpy arrays of the labels
                separated regarding the transformations

    """
    data_tr, labels_tr = transform_data_and_label(data, labels,
                                                  transformers,
                                                  shuffle=False)

    # Save images in two different folders. If we are considering a subset of
    # the instances, then change the name appropietly.
    if subset:
        a_folder_path, b_folder_path = \
            a_folder_path + "_subset", b_folder_path + "_subset"
    folders_path = [a_folder_path, b_folder_path]

    # Create output folders if they dont exist
    if not os.path.exists(a_folder_path):
        os.makedirs(a_folder_path)
    if not os.path.exists(b_folder_path):
        os.makedirs(b_folder_path)

    # If we are using all data change subset_size so that we consider
    # all data in both partitions
    if not subset:
        subset_size = data_tr[0].shape[0]

    if not paired:
        # Unpaired data

        # Iterate over the two data partitions and save their instances into
        # their respective folders
        for data_tr_part, labels_tr_part, folder_path, img_name \
                in zip(data_tr, labels_tr, folders_path, tr_img_names):

            for i, (x, y) in enumerate(zip(data_tr_part[:subset_size],
                                           labels_tr_part[:subset_size])):

                x = cv2.normalize(x, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                cv2.imwrite(join(folder_path,
                                 "{}_{}_{}.png".format(img_name, i, str(y))),
                            x)

    else:
        # Paired data

        unique_labels = np.unique(labels)

        # Calculate, given the subset_size, the amount of instances in each
        # of the classes
        label_subset_size = subset_size // unique_labels.shape[0]

        # Iterate over the two data partitions, match pairs belonging to the
        # same label and save their instances into their respective folders
        for label in unique_labels:

            # Get indices from both transf. which belong to the label
            idx_label_tr_a = np.nonzero(labels_tr[0] == label)
            idx_label_tr_b = np.nonzero(labels_tr[1] == label)

            # Recover the instances from those indices
            data_label_tr_a = data_tr[0][idx_label_tr_a][:label_subset_size]
            data_label_tr_b = data_tr[1][idx_label_tr_b][:label_subset_size]

            # Save all the instances as individual imgs
            for i, (xa, xb) in enumerate(zip(data_label_tr_a, data_label_tr_b)):
                # Transf. a
                xa = cv2.normalize(xa, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                cv2.imwrite(join(folders_path[0], "{}_{}.png".
                                 format(i, str(label))),
                            xa)

                # Transf. b
                xb = cv2.normalize(xb, None, 0, 255, norm_type=cv2.NORM_MINMAX)
                cv2.imwrite(join(folders_path[1], "{}_{}.png".
                                 format(i, str(label))),
                            xb)

    return data_tr, labels_tr


def create_separated_custom_mnist(transformers, normalize=False,
                                  subset=False, subset_size=1000,
                                  paired=False):
    """Apply custom transformation to mnist dataset.

    Load MNIST dataset and apply one transformation. At the end the transformed
    dataset is saved in different folders:
    -   trainA
    -   trainB
    -   testA
    -   testB

    Args:
        transformers (list): list of transformer objects
        normalize (bool): indicates whether to normalize the MNIST dataset
        subset (bool): indicates if only a subset of subset_size instances is
                       used
        subset_size (int): size of the subset if subset equals True
        paired (bool): indicates if instances in both folders should be paired

    Returns:
        (list): x_train containing the partitions transformed
        (list): y_train containing the partitions updated to match x_train
        (list) x_test containing the partititions transformed
        (list) y_test containing the partitions updated to match x_test
    """

    x_train, y_train, x_test, y_test = \
        load_mnist_into_numpy(normalize=normalize)

    # Check the shape of the training data
    print("Shape of the training data X: {}".format(x_train.shape))
    print("Shape of the training data y: {}".format(y_train.shape))

    # Check the shape of testing data
    print("Shape of the testing data X: {}".format(x_test.shape))
    print("Shape of the testing data y: {}".format(y_test.shape))

    x_train_tr, y_train_tr = \
        create_separated_transformed(data=x_train, labels=y_train,
                                     transformers=transformers,
                                     a_folder_path=A_TRAIN_FOLDER_PATH,
                                     b_folder_path=B_TRAIN_FOLDER_PATH,
                                     subset=subset, subset_size=subset_size,
                                     paired=paired)

    plot_img_grid(x_train_tr[0], y_train_tr[0], "MNIST train set transf. 1")
    plot_img_grid(x_train_tr[1], y_train_tr[1], "MNIST train set transf. 2")

    x_test_tr, y_test_tr = \
        create_separated_transformed(data=x_test, labels=y_test,
                                     transformers=transformers,
                                     a_folder_path=A_TEST_FOLDER_PATH,
                                     b_folder_path=B_TEST_FOLDER_PATH,
                                     subset=subset, subset_size=subset_size,
                                     paired=paired)

    return x_train_tr, y_train_tr, x_test_tr, y_test_tr


def create_separated_inv_color_mnist(subset=False, subset_size=1000,
                                     paired=False):
    """Apply inverse color transformation to mnist dataset.

    Load MNIST dataset and apply one transformation. The transformation
    inverts the colors of the images. At the end the transformed dataset
    is saved in different folders:
    -   trainA
    -   trainB
    -   testA
    -   testB

    Args:
        subset (bool): indicates if only a subset of subset_size instances is
                       used
        subset_size (int): size of the subset if subset equals True
        paired (bool): indicates if instances in both folders should be paired

    Returns:
        (list): x_train containing the partitions transformed
        (list): y_train containing the partitions updated to match x_train
        (list) x_test containing the partititions transformed
        (list) y_test containing the partitions updated to match x_test
    """
    transformers = [
        IdTransformer(),
        InvColorTransformer(),
    ]

    return create_separated_custom_mnist(transformers, subset=subset,
                                         subset_size=subset_size, paired=paired)


def create_separated_pca_reduced_mnist(height=8,
                                       subset=False, subset_size=1000,
                                       paired=False):
    """Apply a pca reduction transformation to mnist dataset.

    Load MNIST dataset and apply one transformation. The transformation
    projects the instances into an smaller feature space by using PCA
    decomposition, images are created from that projection. Thus, the size
    of the transformed images is smaller than the original ones.
    At the end the transformed dataset is saved in different folders:
    -   trainA
    -   trainB
    -   testA
    -   testB

    Args:
        n_components: number of components or amount of variance explained
              for the PCA transformer
        subset (bool): indicates if only a subset of subset_size instances is
                       used
        subset_size (int): size of the subset if subset equals True
        paired (bool): indicates if instances in both folders should be paired

    Returns:
        (list): x_train containing the partitions transformed
        (list): y_train containing the partitions updated to match x_train
        (list) x_test containing the partititions transformed
        (list) y_test containing the partitions updated to match x_test
    """
    transformers = [
        IdTransformer(),
        ImgPcaTransformer(h=8),
    ]

    return create_separated_custom_mnist(transformers, normalize=True,
                                         subset=subset, subset_size=subset_size,
                                         paired=paired)


def create_separated_proj_pca_mnist(n_components=0.3,
                                    subset=False, subset_size=5000,
                                    paired=False):
    """Apply a pca reduction transformation to mnist dataset.

    Load MNIST dataset and apply one transformation. The transformation
    projects the instances into an smaller feature space by using PCA
    decomposition, then from that reduced feature space they are projected back
    to the original domain.
    At the end the transformed dataset is saved in different folders:
    -   trainA
    -   trainB
    -   testA
    -   testB

    Args:
        n_components (int): number of components or amount of variance explained
              for the PCA transformer
        subset (bool): indicates if only a subset of subset_size instances is
                       used
        subset_size (int): size of the subset if subset equals True
        paired (bool): indicates if instances in both folders should be paired

    Returns:
        (list): x_train containing the partitions transformed
        (list): y_train containing the partitions updated to match x_train
        (list) x_test containing the partitions transformed
        (list) y_test containing the partitions updated to match x_test
    """
    transformers = [
        IdTransformer(),
        ProjectedPcaTransformer(n_components=n_components),
    ]

    return create_separated_custom_mnist(transformers, normalize=False,
                                         subset=subset, subset_size=subset_size,
                                         paired=paired)


def create_separated_noisy_proj_pca_mnist(n_components=0.3, noise_factor=0.5,
                                          subset=False, subset_size=5000,
                                          paired=False):
    """Apply a noisy pca reduction transformation to mnist dataset.

    Load MNIST dataset and apply one transformation. The transformation
    projects the instances into an smaller feature space by using PCA
    decomposition. Then, gaussian noise is added to those projections.
    At the end the transformed dataset is saved in different folders:
    -   trainA
    -   trainB
    -   testA
    -   testB

    Args:
        n_components (int): number of components or amount of variance explained
              for the PCA transformer
        noise_factor: number that multiplies the values of the gaussian noise.
        subset (bool): indicates if only a subset of subset_size instances is
                       used
        subset_size (int): size of the subset if subset equals True
        paired (bool): indicates if instances in both folders should be paired

    Returns:
        (list): x_train containing the partitions transformed
        (list): y_train containing the partitions updated to match x_train
        (list) x_test containing the partitions transformed
        (list) y_test containing the partitions updated to match x_test
    """
    transformers = [
        IdTransformer(),
        NoisyProjectedPcaTransformer(n_components=n_components,
                                     noise_factor=noise_factor),
    ]

    return create_separated_custom_mnist(transformers, normalize=True,
                                         subset=subset, subset_size=subset_size,
                                         paired=paired)

