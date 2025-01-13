import torch
import models.MLP_MNIST     # required if using model even if shaded out!
import models.CNN_MNIST     # required if using model even if shaded out!
from models.SmartSequential import SmartSequential
from src import data
from src import training
from src import hyperparams
from src import save_load


def main(config):
    # Load device hyperparams (DH), ...
    #      model hyperparams (MH) and ...
    #      training hyperparams (TH)
    DH, MH, TH = hyperparams.read(config)

    # Pytorch initialisation
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed=TH["SEED"])

    # Load data
    X, X_train, X_test, y, y_train, y_test = \
        data.get_data(dataset=DH["DATASET"], 
                      reshape=DH["RESHAPE"], \
                      test_size=TH["TEST_SIZE"])
    
    # Load/initialise model
    if "LOAD" in MH and MH["LOAD"] is not None:
        state_dict = torch.load(MH["LOAD"])
        # get the Class
        ModelClass = SmartSequential.module_dict[state_dict["MODEL_CLASS"]]    
        model = ModelClass(state_dict["CONFIG"])
        model.load_state_dict(state_dict)
    else:
        ModelClass = SmartSequential.module_dict[MH["MODEL_CLASS"]]
        model = ModelClass(CONFIG=MH["CONFIG"])

    # Optimiser
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=TH["LR"]
    )

    # TODO: put this in the config .json file
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer=optimiser, 
    #     mode="min", 
    #     patience=TH["SCHEDULER_PATIENCE"], 
    #     factor=TH["SCHEDULER_FACTOR"]
    # )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimiser, 
        gamma=0.9)

    # Loss function
    loss_function = torch.nn.CrossEntropyLoss()

    # Train
    model, curr_performance, training_complete = training.training_loop(
        model, optimiser, loss_function, scheduler,
        X_train, y_train, X_test, y_test, 
        TH["N_EPOCHS"], TH["EVAL_INTERVAL"]
    )

    print(f"Training Complete: {training_complete}\n")

    save_load.save(model, 
                file_path=MH["SAVE_FILE"], 
                loss=curr_performance["loss_test"], 
                acc=curr_performance["accuracy_test"], 
                )


if __name__ == "__main__":
    # main("configs/config1.json")
    # main("configs/config2.json")
    # main("configs/config_SmallCNN.json")
    main(config="configs/MNIST_CNN_fromscratch.json")
    # main(config="configs/MNIST_CNN_fromload.json")