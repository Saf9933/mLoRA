import logging
from typing import Callable, Dict, Optional
import torch

import mlora.profiler
from mlora.config import MLoRAConfig, TaskConfig
from mlora.model.args import MLoRAData
from mlora.model.llm import LLMModel
from mlora.model.tokenizer import Tokenizer

from .dispatcher import DISPATCHER_CLASS, Dispatcher
from .task import Task

class Executor:
    model_: LLMModel
    tokenizer_: Tokenizer
    dispatcher_: Dispatcher
    loss_dict: Dict[str, float]  # Dictionary to store loss values per adapter

    def __init__(self, model: LLMModel, tokenizer: Tokenizer, config: MLoRAConfig) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer
        self.loss_dict = {}  # Initialize the loss dictionary

        dispatcher_name = config.dispatcher_.name_
        assert dispatcher_name in DISPATCHER_CLASS
        self.dispatcher_ = DISPATCHER_CLASS[dispatcher_name](config.dispatcher_)

        hook_func = {
    "init": self.__task_init_hook,
    "running": self.__task_to_running_hook,
    "ready": self.__task_to_ready_hook,
    "done": self.__task_to_done_hook,
    "terminate": self.__task_to_terminate_hook,
}


        for hook, cb in hook_func.items():
            self.dispatcher_.register_hook(hook, cb)

    def register_hook(self, name: str, cb: Callable):
        self.dispatcher_.register_hook(name, cb)

    def __task_init_hook(self, task: Task):
        logging.info(
            f"Init {task.task_type()} : {task.task_name()} "
            + f"task with adapters: {task.adapter_name()}"
        )
        task.prepare(self.model_.linears_info(), self.tokenizer_)

    def execute(self) -> None:
        mm_collect_step = 0
        while not self.dispatcher_.is_done():
            data: MLoRAData | None = self.dispatcher_.data()
            assert data is not None

            torch.cuda.reset_peak_memory_stats(device=self.model_.device_)

            batch_size = data.batch_size()
            token_len = data.token_len()
            output = self.model_.forward(data.model_data())
            labels = torch.tensor(data.batch_tokens_, dtype=torch.long)
            total_loss = None

            for config in data.data_config_:
                loss = config.loss_fn_(output, labels, torch.tensor(data.batch_mask_))
                if loss:
                    total_loss = loss if total_loss is None else total_loss + loss

            if total_loss is not None:
                total_loss.backward()
                task_name = data.task_name_
                self.loss_dict[task_name] = total_loss.item()

            self.dispatcher_.step()
            mm_collect_step += 1
            