import torch

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

import copy
import datetime

import numpy as np

from pathlib import Path

from src.utils import get_grad_norm, get_parameter_norm


VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class DefaultEngine(Engine):

    def __init__(
        self,
        func,
        model,
        crit,
        optimizer,
        config,
    ):
        ## Ignite Engine does not have objects in below lines.
        ## Thus, we assign class variables to access these object, during the procedure.
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config

        ## Ignite Engine only needs function to run.
        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None

        self.device = next(model.parameters()).device


    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        ## Attaching would be repaeted for serveral metrics.
        ## Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine, 
                metric_name,
            )

        training_metric_names = ["loss", "acc", "|param|", "|g_param|"]

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        ## If the verbosity is set, progress bar would be shown for mini-batch iterations.
        ## Without ignite, you can use tqdm to implement progress bar.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None)
            pbar.attach(train_engine, training_metric_names)

        ## If the verbosity is set, statistics would be shown after each epoch.
        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                print("Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.2e} acc={:.3f}".format(
                    engine.state.epoch,
                    engine.state.metrics["|param|"],
                    engine.state.metrics["|g_param|"],
                    engine.state.metrics["loss"],
                    engine.state.metrics["acc"],
                ))

        validation_metric_names = ["loss", "acc"]
        
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        ## Do same things for validation engine.
        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print("Validation - loss={:.4e} acc={:.3f} best_loss={:.4e}".format(
                    engine.state.metrics["loss"],
                    engine.state.metrics["acc"],
                    engine.best_loss,
                ))


    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics["loss"])
        if loss <= engine.best_loss: ## If current epoch returns lower validation loss,
            engine.best_loss = loss  ## Update lowest validation loss.
            engine.best_model = copy.deepcopy(engine.model.state_dict()) # Update best model weights.


    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                "model": engine.best_model,
                "config": config,
                **kwargs,
            }, 
            config.model_fpath,
        )


class EngineForRestrictedImageNet(DefaultEngine):

    def __init__(
        self, 
        func, 
        model, 
        crit, 
        optimizer, 
        scheduler, 
        config,
    ):
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()

        super().__init__(func, model, crit, optimizer, config)


    @staticmethod
    def train(engine, mini_batch):
        ## You have to reset the gradients of all model parameters
        ## before to take another step in gradient descent.
        engine.model.train() ## because we assign model as class variable, we can easily access to it
        engine.optimizer.zero_grad()

        ## Unpack with lazy loading.
        input   = mini_batch["inp"].to(engine.device)
        labels  = mini_batch["tar"].to(engine.device)
        ## |input|  = (batch_size, 3, height, width)
        ## |labels| = (batch_size)

        with torch.cuda.amp.autocast():
            ## Take feed-forward
            output = engine.model(input)
            ## |output| = (batch_size, n !< 9)

            ## Hance ResNet50 does not using softmax layer,
            ## we need to apply cross entropy loss, not nll loss.
            ## See: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L204
            loss = torch.nn.functional.cross_entropy(output, labels)

            ## Calculate accuracy.
            ## See: https://docs.microsoft.com/ko-kr/windows/ai/windows-ml/tutorials/pytorch-analysis-train-model
            _, predicted = torch.max(output, 1)
            acc = (predicted == labels).sum()

        ## If we are using gpu, not cpu,
        if engine.config.gpu_id >= 0:
            engine.scaler.scale(loss).backward()
        else:
            loss.backward()

        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        ## In order to avoid gradient exploding, we apply gradient clipping.
        torch.nn.utils.clip_grad_norm_(
            engine.model.parameters(),
            engine.config.max_grad_norm,
        )
        ## Take a step of gradient descent.
        if engine.config.gpu_id >= 0:
            ## Use caler instead of engine.optimizer.step() if using GPU.
            engine.scaler.step(engine.optimizer)
        else:
            engine.optimizer.step()

        scale = engine.scaler.get_scale()
        engine.scaler.update()

        ## No update scheduler when errors occured in mixed precision policy.
        if scale == engine.scaler.get_scale() and engine.scheduler != None:
            engine.scheduler.step()

        loss = loss.cpu().detach().numpy()
        acc = acc.cpu().detach().numpy()

        return {
            "loss": loss,
            "acc": acc,
            "|param|": p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            "|g_param|": g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }


    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            ## Unpack with lazy loading.
            input   = mini_batch["inp"].to(engine.device)
            labels  = mini_batch["tar"].to(engine.device)
            ## |input|  = (batch_size, 3, height, width)
            ## |labels| = (batch_size)

            with torch.cuda.amp.autocast():
                ## Take feed-forward
                output = engine.model(input)
                ## |output| = (batch_size, n !< 9)

                ## Hance ResNet50 does not using softmax layer,
                ## we need to apply cross entropy loss, not nll loss.
                ## See: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L204
                loss = torch.nn.functional.cross_entropy(output, labels)

                ## Calculate accuracy.
                ## See: https://docs.microsoft.com/ko-kr/windows/ai/windows-ml/tutorials/pytorch-analysis-train-model
                _, predicted = torch.max(output, 1)
                acc = (predicted == labels)

        loss = loss.cpu().detach().numpy()
        acc = np.mean(acc.cpu().detach().numpy())

        return {
            "loss": loss,
            "acc": acc,
        }


    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    
    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics["loss"])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    
    @staticmethod
    def save_model(engine, train_engine, model_dir, config):
        avg_train_loss = train_engine.state.metrics["loss"]
        avg_valid_loss = engine.state.metrics["loss"]

        avg_train_acc = train_engine.state.metrics["acc"]
        avg_valid_acc = engine.state.metrics["acc"]

        model_fname = Path(".".join([
            # config.model_fpath,                                     ## user-entered hyper-params
            "%02d" % train_engine.state.epoch,                      ## current epoch
            "%.2f-%.3f" % (avg_train_loss, avg_train_acc),          ## train assets
            "%.2f-%.3f" % (avg_valid_loss, avg_valid_acc),          ## valid assets
            "pth",                                                  ## extension
        ]))

        ## Unlike other tasks, we need to save current model, not best model.
        torch.save({
            "resnet50": engine.model.state_dict(),
            "config": config,
        }, model_dir / model_fname)


class RestrictedImageNetTrainer():

    def __init__(
        self, 
        config,
    ):
        self.config = config
        
        ## Set a filename for model of last epoch.
        ## We need to put every information to filename, as much as possible.
        self.model_dir = Path(config.ckpt, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model_dir.mkdir(parents=True, exist_ok=True)


    def train(
        self,
        model, 
        crit, 
        optimizer, 
        scheduler,
        train_loader, 
        valid_loader,
    ):
        train_engine = EngineForRestrictedImageNet(
            EngineForRestrictedImageNet.train,
            model, 
            crit, 
            optimizer, 
            scheduler, 
            self.config,
        )
        validation_engine = EngineForRestrictedImageNet(
            EngineForRestrictedImageNet.validate,
            model, 
            crit, 
            optimizer=None, ## no need to throw optimizer
            scheduler=None, ## no need to throw scheduler
            config=self.config,
        )

        EngineForRestrictedImageNet.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose,
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,          ## event
            run_validation,                  ## function
            validation_engine, valid_loader, ## arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,                 ## event
            EngineForRestrictedImageNet.check_best, ## function
        )
        ## Save models for each epochs.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            EngineForRestrictedImageNet.save_model,
            train_engine,
            self.model_dir,
            self.config,
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        return model
