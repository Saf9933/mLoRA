import logging
from typing import Callable, Dict, Optional
import torch
import optuna
import json
import os

import mlora.profiler
from mlora.config import MLoRAConfig, TaskConfig
from mlora.model.args import MLoRAData
from mlora.model.llm import LLMModel
from mlora.model.tokenizer import Tokenizer
from mlora.executor.dispatcher import DISPATCHER_CLASS, Dispatcher
from mlora.executor.task import Task

# Set up structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")


class Executor:
    model_: LLMModel
    tokenizer_: Tokenizer
    dispatcher_: Dispatcher
    loss_dict: Dict[str, float]  # Dictionary to store loss values per adapter

    def __init__(
            self,
            model: LLMModel,
            tokenizer: Tokenizer,
            config: MLoRAConfig) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer
        self.loss_dict = {}  # Initialize the loss dictionary

        dispatcher_name = config.dispatcher_.name_
        if dispatcher_name not in DISPATCHER_CLASS:
            raise ValueError(
                f"Dispatcher '{dispatcher_name}' not found in DISPATCHER_CLASS.")
        self.dispatcher_ = DISPATCHER_CLASS[dispatcher_name](
            config.dispatcher_)

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
            f"Initializing task '{
                task.task_name()}' of type '{
                task.task_type()}' with adapters: {
                task.adapter_name()}")
        task.prepare(self.model_.linears_info(), self.tokenizer_)

    def execute(self) -> None:
        mm_collect_step = 0
        while not self.dispatcher_.is_done():
            data: Optional[MLoRAData] = self.dispatcher_.data()
            assert data is not None

            torch.cuda.reset_peak_memory_stats(device=self.model_.device_)
            try:
                batch_size = data.batch_size()
                token_len = data.token_len()
                output = self.model_.forward(data.model_data())
                labels = torch.tensor(data.batch_tokens_, dtype=torch.long)
                total_loss = None

                for config in data.data_config_:
                    loss = config.loss_fn_(
                        output, labels, torch.tensor(
                            data.batch_mask_))
                    if loss:
                        total_loss = loss if total_loss is None else total_loss + loss

                if total_loss is not None:
                    total_loss.backward()
                    task_name = data.task_name_
                    self.loss_dict[task_name] = total_loss.item()

                logging.info(
                    f"Step {mm_collect_step}: Loss for task '{task_name}': {
                        total_loss.item()}")

            except Exception as e:
                logging.error(
                    f"Error during execution step {mm_collect_step}: {e}")
                raise

            self.dispatcher_.step()
            mm_collect_step += 1


def load_lora_config(config_path: str, model_path: str) -> MLoRAConfig:
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(config_path, "r") as f:
        adapter_config = json.load(f)

    logging.info("Successfully loaded LoRA configuration.")

    config = MLoRAConfig("demo/lora/lora_case_1.yaml")
    config.adapters = {
        "lora_sft_0": {
            "type": "lora", "name": "lora_sft_0", "path": model_path, "r": adapter_config.get(
                "r", 16), "alpha": adapter_config.get(
                "alpha", 1.0), "dropout": adapter_config.get(
                    "dropout", 0.1), "target_modules": adapter_config.get(
                        "target_modules", [
                            "module1", "module2"]), "optimizer": adapter_config.get(
                                "optimizer", "adamw"), "lr": adapter_config.get(
                                    "learning_rate", 1e-3)}}
    logging.info("LoRA configuration loaded with parameters: " +
                 str(config.adapters["lora_sft_0"]))
    return config


def objective(trial):
    rank = trial.suggest_int("rank", 4, 64)
    alpha = trial.suggest_float("alpha", 0.1, 10.0)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    logging.info(
        f"Running trial with parameters: rank={rank}, alpha={alpha}, lr={lr}, dropout={dropout}")

    try:
        model = LLMModel.from_pretrained(
            "adapters/lora_sft_0/adapter_model.bin",
            device="cuda",
            precision="float32")
    except AttributeError as e:
        logging.warning("Fallback to direct instantiation of LLMModel.")
        model = LLMModel("adapters/lora_sft_0/adapter_model.bin")

    try:
        tokenizer = Tokenizer("mlora/model/tokenizer/tokenizer.py")
    except FileNotFoundError as e:
        logging.error("Tokenizer file not found.")
        raise e

    config = load_lora_config(
        "adapters/lora_sft_0/adapter_config.json",
        "adapters/lora_sft_0/adapter_model.bin")

    executor = Executor(model, tokenizer, config)
    try:
        executor.execute()
    except Exception as e:
        logging.error(f"Execution error: {e}")
        raise

    loss = executor.loss_dict.get("test_0", float("inf"))
    logging.info(f"Trial completed with loss: {loss}")
    print(f"Trial loss: {loss}")
    return loss


if __name__ == "__main__":
    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial), n_trials=20)

        logging.info(f"Best hyperparameters found: {study.best_params}")
        print("Best hyperparameters found:", study.best_params)

        for trial in study.trials:
            logging.info(
                f"Trial {
                    trial.number}: Params: {
                    trial.params}, Loss: {
                    trial.value}")
            print(
                f"Trial {
                    trial.number}: Params: {
                    trial.params}, Loss: {
                    trial.value}")

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")
        raise
