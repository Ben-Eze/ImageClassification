import torch
from models.MLP_MNIST import MLP_MNIST
from src import data
from src import training
from src import hyperparams
from src import save_load

# load model hyperparams (MH) and training hyperparams (TH)
MH, TH = hyperparams.read("configs/config0.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed=TH["SEED"])

X, X_train, X_test, y, y_train, y_test = data.get_MNIST(TH["TEST_SIZE"])

model = MLP_MNIST(MH["INPUT_BREADTH"], 
                  MH["OUTPUT_BREADTH"],
                  MH["HIDDEN_DEPTH"], 
                  MH["HIDDEN_BREADTH"],
                  MH["BIAS"], MH["DROPOUT_P"])

optimiser = torch.optim.AdamW(
    model.parameters(), lr=TH["LR"]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimiser, 
    mode="min", 
    patience=TH["SCHEDULER_PATIENCE"], 
    factor=TH["SCHEDULER_FACTOR"]
)

loss_function = torch.nn.CrossEntropyLoss()

model, curr_performance, training_complete = training.training_loop(
    model, optimiser, loss_function, scheduler,
    X_train, y_train, X_test, y_test, 
    TH["N_EPOCHS"], TH["EVAL_INTERVAL"]
)

print(f"Training Complete: {training_complete}\n")

save_load.save(model, MH["SAVE_FILE"], training_complete)