import torch
import models.MLP_MNIST
import models.CNN_MNIST
from models.SmartSequential import SmartSequential
import matplotlib.pyplot as plt

def bound(x):
    # goes from infinite range centered about 0.5 to [0, 16]
    return 16 * torch.nn.functional.sigmoid(x-0.5)

def main(model_path):
    # MODEL
    state_dict = torch.load(model_path)
    # get the Class
    ModelClass = SmartSequential.module_dict[state_dict["MODEL_CLASS"]]    
    model = ModelClass(state_dict["CONFIG"])            # set the architecture
    model.load_state_dict(state_dict)                   # load the params
    model.eval()

    # IMAGE
    unbounded_pixels = torch.nn.Parameter(
        torch.rand((1, 8, 8), dtype=torch.float32, requires_grad=True)
    )

    # OPTIMISER
    optimiser = torch.optim.AdamW(
        params=[unbounded_pixels],
        lr = 0.02
    )

    # LOSS FUNC
    loss_func = lambda logits, target: \
        torch.nn.functional.cross_entropy(0.1*logits, target)
    
    digit = 6

    optimum_confidence = torch.full((1, 10, ), 0, dtype=torch.float32)
    optimum_confidence[0, digit] = 1

    N_epochs = 10001

    for epoch_i in range(N_epochs):
        pixels = bound(unbounded_pixels)
        logits = model(pixels)
        loss = loss_func(logits, optimum_confidence)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


        if not (epoch_i % 50):
            print(
                f"Epoch no. : {epoch_i}\n"
                "Confidences:"
            )

            print(*[f"{i} : {logits.softmax(1)[0, i].item():.2f}" for i in range(10)], sep=", ")
            # print(*[f"{i} : {logits[0, i].item():.2f}" for i in range(10)], sep=", ")

            pixels_np = bound(unbounded_pixels)[0].detach().numpy()
            plt.imshow(pixels_np, "gray")
            plt.pause(1e-8)
            
    print(model(unbounded_pixels).softmax(dim=1))

    print("Training complete")

    plt.show()



if __name__ == "__main__":
    # main('models/state_dicts/SmallCNN9944.pt')
    # main('models/state_dicts/SmallCNN9556.pt')
    main('models/state_dicts/BigCNN9972.pt')
    # main('models/state_dicts/test_MLP,loss=0.5219,acc=0.9361.pt')
    # main('models/state_dicts/SmallCNN,loss=0.0272,acc=0.9917.pt')