import torch
# from models.SmartSequential import SmartSequential
# import models.CNN_MNIST
from src import init
from src import hyperparams
from src import data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_input_img(X_in):
    x = X_in[0, 0].detach().numpy()
    plt.imshow(x, cmap="gray")
    plt.pause(1e-8)


def plot_hidden_layers(C, rows, cols):
    c = C.detach().numpy()
    fig, axs = plt.subplots(rows, cols)
    for i in range(rows * cols):
        axs[i // cols, i % cols].imshow(c[0, i])
    plt.pause(1e-8)

def get_vertical_indices(num_images, max_channels):
    start_idx = (max_channels - num_images) // 2
    end_idx = start_idx + num_images
    return start_idx, end_idx

def plot_conv_layers(X_in, conv_outputs, X_idx=0, max_channels=8):
    num_layers = len(conv_outputs) + 1  # Including input image
    fig, axes = plt.subplots(nrows=max_channels, ncols=num_layers, figsize=(num_layers * 2, max_channels * 2))

    # Plot input image
    input_images = X_in.shape[1]
    start_idx, end_idx = get_vertical_indices(input_images, max_channels)
    for i in range(start_idx, end_idx):
        axes[i, 0].imshow(X_in[X_idx, i - start_idx].cpu().detach().numpy(), cmap='gray')
        axes[i, 0].axis('off')
        # Enlarge the input image
        # axes[i, 0].set_box_aspect(1)  # Adjust the aspect ratio
    axes[start_idx, 0].set_title('Input Image', fontsize=12)

    # Plot conv layers
    for layer_idx, layer_output in enumerate(conv_outputs):
        num_images = layer_output.shape[1]
        start_idx, end_idx = get_vertical_indices(num_images, max_channels)
        for i in range(start_idx, end_idx):
            axes[i, layer_idx + 1].imshow(layer_output[X_idx, i - start_idx].cpu().detach().numpy())
            axes[i, layer_idx + 1].axis('off')
            # Enlarge the first layer
            # if layer_idx != 0:
            #     axes[i, layer_idx + 1].set_box_aspect(2)
        axes[start_idx, layer_idx + 1].set_title(f'Layer {layer_idx + 1}', fontsize=12)

    # Turn off unused axes
    for i in range(max_channels):
        for j in range(num_layers):
            if not axes[i, j].images:
                axes[i, j].axis('off')

    # Adjust layout to reduce horizontal space and maintain proportions
    plt.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.05, wspace=0.02, hspace=0.1)
    plt.show()


def plot_latent_space(E, y):
    # plot the digits in the 3d latent space, colour coded by the digit
    e = E.detach().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        is_i = y == i
        ax.scatter(e[is_i, 0], e[is_i, 1], e[is_i, 2], label=str(i))
    plt.legend()
    plt.pause(1e-8)


def main(config):
    DH, MH, TH = hyperparams.read(config)

    model = init.get_model(MH)
    model.eval()

    layers = list(model.architecture)

    conv1 = torch.nn.Sequential(*layers[:3])
    conv2 = torch.nn.Sequential(*layers[3:6])
    conv3 = torch.nn.Sequential(*layers[6:9])
    encoder = torch.nn.Sequential(*layers[9:17])
    decoder = torch.nn.Sequential(*layers[17:])
    
    print(conv1)
    print(conv2)
    print(conv3)
    print(encoder)
    print(decoder)

    X, X_train, X_test, y, y_train, y_test = \
        data.get_data(dataset=DH["DATASET"], 
                      reshape=DH["RESHAPE"], \
                      test_size=TH["TEST_SIZE"])

    X_in = X_test.unsqueeze(1)
    C1 = conv1(X_in)
    C2 = conv2(C1)
    C3 = conv3(C2)
    E = encoder(C3)
    Y = decoder(E)

    conv_layers = [C1, C2, C3]
    plot_conv_layers(X_in, conv_layers, X_idx=1)
    # plot_latent_space(E, y_test)
    plt.show()

main("configs/config_surgery.json")