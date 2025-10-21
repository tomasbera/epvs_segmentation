import torch
from monai.losses import DiceCELoss
from monai.networks.layers import Norm
from monai.networks.nets import UNet

import helpers
from preprocess import run_preprocessing
from train_model import train_model


def run_main(idun=False):
    model_dir = "./result_model"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if idun:
        base_data_path = "/cluster/projects/vc/data/mic/closed/MRI_PVS/opennero/braindata"
        mask_base_path = "/cluster/projects/vc/data/mic/closed/MRI_PVS/opennero/binary_epvs_groundtruth/mask"
        data_input = run_preprocessing(base_data_path=base_data_path, mask_base_path=mask_base_path)

    else:
        data_input = run_preprocessing()

    helpers.show_transfer_data(data_input)
    pixel_values = helpers.calculate_pixels(data_input[0])


    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    #loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, weight=helpers.calc_w(pixel_values[0, 0], pixel_values[0, 1]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

    train_model(model, data_input, loss_function, optimizer, 600, model_dir)

if __name__ == "__main__":
    run_main(idun=True)