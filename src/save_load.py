import os
from pathlib import Path
import torch


def save(model, file_path, loss=None, acc=None, training_complete=None):
    path = Path(file_path)
    dir = path.parent

    # This reduces the chance of overwriting a model accidentally
    # For more important training on bigger models, would want to ensure unique 
    # file names to eliminate the risk of overwriting
    if training_complete is not None and not training_complete:
        fname = path.name
        file_path = f"{dir}/INTERRUPTED-{fname}"

    os.makedirs(dir, exist_ok=True)
    if file_path.endswith(".pt"):
        file_path = file_path[:-3]
    
    if loss is not None:
        file_path += f",loss={loss:.4f}"

    if acc is not None:
        file_path += f",acc={acc:.4f}"
        file_path += ".pt"
    
    torch.save(model.state_dict(), file_path)
    
    print(f"Model ({get_num_params(model)}-parameter {type(model)}) " 
          f"saved successfully to {file_path}")


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)