import torch

import matplotlib.pyplot as plt
import numpy as np

from larynx.models.self_supervised.miscellaneous import get_dataloaders, get_model

def save_image(gt, input, result, name):
    ax = plt.subplot(1, 3, 1)
    ax.set_title("Ground Truth")
    ax.get_figure().set_size_inches(8, 3)
    plt.imshow(gt, cmap='gray')

    ax = plt.subplot(1, 3, 2)
    ax.set_title("Input")
    plt.imshow(input, cmap='gray')

    ax = plt.subplot(1, 3, 3)
    ax.set_title("Output")
    plt.imshow(result, cmap='gray')

    pth = "reports/figures/temp" + f'/figure_{name}.png'
    plt.tight_layout()
    plt.savefig(pth)

model, device = get_model()
load_model = torch.load('models/self_supervised_learning/ViT_best_model_23_06_2023.pt')
model.load_state_dict(load_model["state_dict"])


print(f'The model loaded.. Best epoch of {load_model["epoch"]}')
print(f'Trained for {load_model["max_epochs"]}.')

###########################################

train_loader, val_loader, _ = get_dataloaders(DEBUG_MODE=True)

print("Entering Validation")
total_val_loss = 0
val_step = 0
model.eval()
for val_batch in val_loader:
    val_step += 1
    inputs, gt_input = (
        val_batch["image"].to(device),
        val_batch["gt_image"].to(device),
    )
    outputs, outputs_v2 = model(inputs)
    print(outputs.shape)
    print(inputs.shape)
    
    outputs = outputs.cpu().detach().numpy()
    for i in range(len(inputs)):
        gt = val_batch["gt_image"][i, 0, :, :]
        input = val_batch["image"][i, 0, :, :]
        result = outputs[i, 0, :, :]

        np.clip(result, 0, 1, out=result)
        save_image(gt, input, result, f'{i}_{val_step}')
        
    exit()


print("Done")