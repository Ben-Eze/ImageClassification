import os
from pathlib import Path
import torch


def save(model, file_path, training_complete=True):
    path = Path(file_path)
    dir = path.parent

    # This reduces the chance of overwriting a model accidentally
    # For more important training on bigger models, would want to ensure unique 
    # file names to eliminate the risk of overwriting
    if not training_complete:
        fname = path.name
        file_path = f"{dir}/INTERRUPTED-{fname}"

    os.makedirs(dir, exist_ok=True)

    torch.save(model.state_dict(), file_path)
    
    print(f"Model ({get_num_params(model)}-parameter {type(model)}) " 
          "saved successfully")


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)