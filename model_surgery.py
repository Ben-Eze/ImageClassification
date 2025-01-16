import torch
# from models.SmartSequential import SmartSequential
# import models.CNN_MNIST
from src import init
from src import hyperparams
from src import data

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# def plot_input_img(X_in):
#     x = X_in[0, 0].detach().numpy()
#     plt.imshow(x, cmap="gray")
#     plt.pause(1e-8)


# def plot_hidden_layers(C, rows, cols):
#     c = C.detach().numpy()
#     fig, axs = plt.subplots(rows, cols)
#     for i in range(rows * cols):
#         axs[i // cols, i % cols].imshow(c[0, i])
#     plt.pause(1e-8)

def plot_conv_layers(X_in, conv_outputs, max_channels=8):
    num_layers = len(conv_outputs) + 1  # Including input image
    fig, axes = plt.subplots(nrows=max_channels, ncols=num_layers, figsize=(num_layers * 3, max_channels * 3))

    # Plot input image
    for i in range(max_channels):
        if i < X_in.shape[1]:
            axes[i, 0].imshow(X_in[0, i].cpu().detach().numpy(), cmap='gray')
        axes[i, 0].axis('off')
    axes[0, 0].set_title('Input Image')

    # Plot conv layers
    for layer_idx, layer_output in enumerate(conv_outputs):
        for i in range(max_channels):
            if i < layer_output.shape[1]:
                axes[i, layer_idx + 1].imshow(layer_output[0, i].cpu().detach().numpy())
            axes[i, layer_idx + 1].axis('off')
        axes[0, layer_idx + 1].set_title(f'Conv Layer {layer_idx + 1}')

    # Adjust layout to reduce white space
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)
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

    # plot_input_img(X_in)
    # plot_hidden_layers(C1, rows=2, cols=2)
    # plot_hidden_layers(C2, rows=2, cols=4)
    # plot_hidden_layers(C3, rows=2, cols=4)
    conv_layers = [C1, C2, C3]
    plot_conv_layers(X_in, conv_layers)
    plot_latent_space(E, y_test)
    plt.show()

main("configs/config_surgery.json")