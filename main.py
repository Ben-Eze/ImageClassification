import torch
import models.MLP_MNIST     # required if using model even if shaded out!
import models.CNN_MNIST     # required if using model even if shaded out!
from models.SmartSequential import SmartSequential
from src import data
from src import training
from src import hyperparams
from src import save_load
from src import init


def main(config):
    # HYPERPARAMETERS:
    #   DH - data, MH - model, TH - training
    DH, MH, TH = hyperparams.read(config)

    # PYTORCH 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed=TH["SEED"])

    # DATA
    X, X_train, X_test, y, y_train, y_test = \
        data.get_data(dataset=DH["DATASET"], 
                      reshape=DH["RESHAPE"], \
                      test_size=TH["TEST_SIZE"])
    
    # MODEL, OPTIMISER, SCHEDULER, LOSS
    model = init.get_model(MH, DEVICE)
    optimiser = init.get_optimiser(model, TH)
    scheduler, step_scheduler = init.get_scheduler(optimiser, TH)
    loss_function = init.get_loss_function(TH)

    # TRAIN
    model, curr_performance, training_complete = training.training_loop(
        model, optimiser, loss_function, scheduler, step_scheduler,
        X_train, y_train, X_test, y_test, 
        TH["N_EPOCHS"], TH["EVAL_INTERVAL"]
    )

    # SAVE
    save_load.save(model, 
                file_path=MH["SAVE_FILE"], 
                loss=curr_performance["loss_test"], 
                acc=curr_performance["accuracy_test"], 
                )


if __name__ == "__main__":
    # main("configs/config1.json")
    main("configs/MLP.json")
    # main("configs/config_SmallCNN.json")
    # main("configs/config_SmallCNN_overfit.json")
    # main("configs/config_autoencoder.json")
    # main(config="configs/MNIST_CNN_fromscratch.json")
    # main(config="configs/MNIST_overfit.json")
    # main(config="configs/MNIST_CNN_fromload.json")