import torch


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_dir: str = r'checkpoints'
    output_dir: str = 'output'
    logdir: str = r'logdir'

    class DatasetConfig:
        dataset_dir: str = r'D:\dataset\face\FER+'
        image_shape = (48, 48)
        # votes_count = 10
        do_augment_data = True

        shuffle = True
        flip = True
        scale_factor = 0.25
        rotation_factor = 30

    class ModelConfig:
        in_channels: int = 3
        n_classes: int = 10      # number of output classes (neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, unknown, NF[Not face])
        n_groups_gm: int = 16
        start_hidden_channels: int = 64,
        n_hidden_expansion: int = 3

    class TrainConfig:
        number_epochs: int = 100
        resume_training: bool = False

        # ========== CUDA training parameters ===========
        is_cudnn_enabled: bool = True
        is_cudnn_benchmark_enabled: bool = True
        is_cudnn_deterministic: bool = False
        gpus: tuple = (0,)
        batch_size: int = 32
        shuffle: bool = True
        workers: int = 16
        pin_memory: bool = True

        # ========== LR scheduler parameters ===========
        lr_scheduler_name: str = 'cosine'       # [multistep/steplr/cosine]
        min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
        lr: float = 6e-4  # max learning rate
        warmup_iters: int = 2000  # how many steps to warm up for
        max_iters: int = 600000  # should be ~= max_iters per Chinchilla

        # ========== Optimizer parameters ===========
        optimizer_name: str = 'adamw'    # [sgd/adam/ramsprop/adamw]
        weight_decay: float = 1e-1
        beta1: float = 0.9
        beta2: float = 0.95
        momentum: float = 0.0
        use_nesterov: bool = False
        alpha: float = 0.99
        centered: float = False

    class ValidConfig:
        batch_size: int = 16
        shuffle: bool = False
        workers: int = 16
        pin_memory: bool = True

    class TestConfig:
        checkpoint_name: str = r'checkpoints/datesnet_model_cont.pth'
