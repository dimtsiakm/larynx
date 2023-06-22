import torch
import os
import matplotlib.pyplot as plt

from larynx.models.self_supervised.miscellaneous import get_dataloaders, get_model

def save_image(gt, input, result, name):
    plt.subplot(1, 3, 1)
    plt.imshow(gt, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(input, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(result, cmap='gray')

    pth = "reports/figures/ViT/comparison_results" + f'/figure_comparison_{name}.png'
    plt.savefig(pth)

model, device = get_model()
load_model = torch.load('models/ViT/best_model_500_epochs.pt')
model.load_state_dict(load_model["state_dict"])


print(f'The model loaded.. Best epoch of {load_model["epoch"]}')
###########################################

train_loader, val_loader, _ = get_dataloaders()

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
    # print("Input shape: {}".format(inputs.shape))
    outputs, outputs_v2 = model(inputs)
    
    print(outputs.shape)
    print(inputs.shape)
    # exit()

    outputs = outputs.cpu().detach().numpy()
    for i in range(len(inputs)):
        save_image(val_batch["gt_image"][i, 0, :, :], val_batch["image"][i, 0, :, :], outputs[i, 0, :, :], f'{i}_{val_step}')
        if i > 4:
            break


print("Done")