from pytorch_lightning import Callback
import time


class TimerCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            self.start_time = time.time()

    def on_train_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            total_time = time.time() - self.start_time
            print(f"Total training time: {total_time:.2f} seconds")
            trainer.logger.log_metrics({"total_training_time": total_time})


class ModelSizeCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        trainer.logger.log_metrics({"total_params": total_params, "trainable_params": trainable_params})