import torch 
from tqdm import tqdm
from utils.bbox_utils import convert_cells_to_bboxes
from utils.nms import nms
from utils.checkpoint import save_checkpoint

class Trainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, anchors, device, config):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.anchors = torch.tensor(anchors).to(device)
        self.device = device
        self.config = config

    def train_epoch(self):
        self.model.train()
        loop = tqdm(self.train_loader, leave=True)
        losses = []

        for batch_idx, (x, y) in enumerate(loop):
            x = x.to(self.device)
            y0, y1, y2 = (
                y[0].to(self.device),
                y[1].to(self.device),
                y[2].to(self.device)
            )

            outputs = self.model(x)
            loss = (
                self.loss_fn(outputs[0], y0, self.anchors[0]) +
                self.loss_fn(outputs[1], y1, self.anchors[1]) +
                self.loss_fn(outputs[2], y2, self.anchors[2])
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        mean_loss = sum(losses) / len(losses)
        
        return mean_loss

    def fit(self):
        for epoch in range(self.config.EPOCHS):
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}")
            loss = self.train_epoch()
            print(f"Loss: {loss}")

            if self.config.SAVE_MODEL:
                save_checkpoint(self.model, self.optimizer, self.config.CHECKPOINT_FILE)