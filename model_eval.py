import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


class EcalModel:
    def __init__(self, model, include_background=True):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.include_background = include_background

        # Dice metric
        self.metric = DiceMetric(include_background=include_background, reduction="mean_batch")

        # Helper for thresholding sigmoid outputs
        self.post_pred = AsDiscrete(threshold=0.5)
        self.post_label = AsDiscrete()  # for ground truth

    def test_model(self, test_loader, verbose=False):
        self.model.eval()
        dice_scores = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                inputs = batch["img"].to(self.device)
                labels = batch["mask"].to(self.device)

                outputs = self.model(inputs)
                outputs = self.post_pred(outputs)  # threshold
                labels = self.post_label(labels)

                score = self.metric(y_pred=outputs, y=labels)
                dice_scores.append(score.item())

                if verbose:
                    print(f"Batch {i+1}, Dice score: {score.item():.4f}")

        avg_score = float(torch.tensor(dice_scores).mean())
        print(f"Average Dice score: {avg_score:.4f}")
        return avg_score
