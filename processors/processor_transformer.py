import torch
from torch import amp
from.processor_base import ProcessorBase

class ProcessorTransformer(ProcessorBase):

    def train(self):
         super(ProcessorTransformer, self).train()
         self.scaler = amp.GradScaler(self.device)
         print("Training Transformer")
         print(self.device)

         for epoch in range(self.epochs):
                self.train_step()
                self.validation_step()

    
    def train_step(self):
        super(ProcessorTransformer, self).train_step()
        print("Training step")
        for n_iter, (img, pid, target_cam, target_view) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            self.optimizer_center.zero_grad()
            img = img.to(self.device)
            target = pid.to(self.device)
            target_cam = target_cam.to(self.device)
            target_view = target_view.to(self.device)
            with amp.autocast(self.device):
                score, feat = self.model(img, target, cam_label=target_cam, view_label=target_view)
                # score, feat = self.model(img, target, cam_label=target_cam, view_label=target_view )
                loss = self.loss_fn(score, feat, target, target_cam)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    
    def validation_step(self):
         super(ProcessorTransformer, self).validation_step()

         with torch.no_grad():
             pass

    


        
        