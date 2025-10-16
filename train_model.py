import os
import numpy as np
import torch

from helpers import dice_val

def to_float(x):
    print("loss/epoch metric: ", x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()  # GPU tensor → float
    return float(x)  # already float


def train_model(model, data, loss, optim, max_epochs, model_dir, test_interval = 2, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_val = []
    save_metrics_train = []
    save_metrics_val = []
    train_loader, test_loader = data

    for epoch in range(max_epochs):
        print("-"*5)
        print(f"Epoch {epoch+1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0

        for batch in train_loader:

            train_step += 1
            volume = batch["image"]
            label = batch["label"]

            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()
            outputs = model(volume)

            train_loss = loss(outputs, label)

            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()

            print(f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                  f"Train_loss: {train_loss.item():.4f}")

            train_metric = dice_val(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(to_float(train_epoch_loss))
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')
        save_metrics_train.append(to_float(epoch_metric_train))
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metrics_train)

        if (epoch + 1) % test_interval == 0:
            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_val = 0
                test_step = 0

                for batch in test_loader:
                    test_step += 1

                    test_volume = batch["image"]
                    test_label = batch["label"]
                    test_label = test_label != 0
                    test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                    test_out = model(test_volume)

                    test_loss = loss(test_out, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_val(test_out, test_label)
                    epoch_metric_val += test_metric

                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                print("Type of test_epoch_loss:", type(test_epoch_loss))
                save_loss_val.append(to_float(test_epoch_loss))
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_val)

                epoch_metric_val /= test_step
                print(f'test_dice_epoch: {epoch_metric_val:.4f}')
                print("Type of epoch_metric_val:", type(epoch_metric_val))
                print(f'test_dice_epoch: {epoch_metric_val:.4f}')
                save_metrics_val.append(to_float(epoch_metric_val))
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metrics_val)

                if epoch_metric_val > best_metric:
                    best_metric = epoch_metric_val
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))

                print(
                    f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

