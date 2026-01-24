import torch
from ..layers.modules import AdversarialDiscriminator


import torch
import importlib
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    LinearLR, SequentialLR, LambdaLR
)

def select_optimizer(optimizer, model, lr, weight_decay = 0.01):
    """
    Selects and initializes an optimizer based on config.
    Supported: 'adam', 'adamw', 'adafactor', 'lion', 'prodigy', 
               '8bit_adamw', '8bit_lion'
    """
    opt_name = optimizer.lower()

    # 1. Standard PyTorch Optimizers
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    elif opt_name == "adafactor":
        # Adafactor is built into newer PyTorch or available via Transformers
        try:
            from torch.optim import Adafactor
            return Adafactor(
                model.parameters(), 
                lr=lr, 
                relative_step=False, 
                scale_parameter=False, 
                warmup_init=False
            )
        except ImportError:
            raise ImportError("Adafactor not found. Ensure you have PyTorch 2.x or 'transformers' installed.")

    # 2. BitsAndBytes (8-bit) Optimizers
    elif "8bit" in opt_name:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise ImportError("Please install bitsandbytes: pip install bitsandbytes")
        import bitsandbytes as bnb
        
        if opt_name == "8bit_adamw":
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == "8bit_lion":
            return bnb.optim.Lion8bit(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Lion (requires 'timm' or 'bitsandbytes' - using bnb version for speed)
    elif opt_name == "lion":
        if importlib.util.find_spec("bitsandbytes") is not None:
            import bitsandbytes as bnb
            return bnb.optim.Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ImportError("Lion optimizer requires 'bitsandbytes'. Run: pip install bitsandbytes")

    # 4. Prodigy (Auto-LR)
    elif opt_name == "prodigy":
        if importlib.util.find_spec("prodigyopt") is None:
            raise ImportError("Please install Prodigy: pip install prodigyopt")
        from prodigyopt import Prodigy
        
        # Note: Prodigy typically requires lr=1.0 to perform auto-tuning
        if lr != 1.0:
            print(f"--- Warning: Prodigy works best with lr=1.0, but you provided {lr} ---")
            
        return Prodigy(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay, 
            safeguard_warmup=True
        )

    else:
        raise ValueError(f"Optimizer {opt_name} is not supported.")
    



def select_scheduler(scheduler, optimizer, warmup_steps = 1, total_steps=1e9, t0 = 1000, gamma=0.95, step_size = 1000):
    # Check if we should skip scheduling (Prodigy)
    if "prodigy" in str(type(optimizer)).lower():
        print("--- Prodigy detected: Returning Identity Scheduler (No-op) ---")
        return LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    sched_name = scheduler
    remaining_steps = total_steps - warmup_steps

    # 1. Warmup

    warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)

    # 2. Main Scheduler
    if sched_name == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=remaining_steps)
    elif sched_name == "cosine_warmstart":
        main_sched = CosineAnnealingWarmRestarts(optimizer, T_0=t0)
    elif sched_name == "step":
        main_sched = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == "linear":
        main_sched = LinearLR(optimizer, start_factor=1.0, end_factor=0.001, total_iters=remaining_steps)
    else:
        return StepLR(optimizer, step_size=step_size, gamma=0.1)

    return SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup_steps])

# --- Usage Example ---
# class Config:
#     optimizer = "8bit_adamw"
#     lr = 1e-4
#     weight_decay = 0.05
#
# opt = select_optimizer(Config(), model)


def create_optimizer_dict(model, device, config, dataloader, num_batch_types = -1):
    scaler = torch.amp.GradScaler(enabled=config.amp)
    DAB_separate_optim = True if config.dab_weight >0 else False

    # This maybe should be part of training code 
    if config.ADV and num_batch_types > 1:
        discriminator = AdversarialDiscriminator(
            d_model=config.layer_size, # embsize
            n_cls=num_batch_types,
        ).to(device)
        print(discriminator)
    else:
        discriminator = None

    optimizer = select_optimizer(optimizer=config.optimizer, 
                                 model=model, 
                                 lr=config.lr, 
                                 weight_decay = config.get('adam_weight_decay', 0.01))
    
    scheduler = select_scheduler(scheduler=config.scheduler, 
                                 optimizer=optimizer, 
                                 warmup_steps=config.get('warmup_steps', 5*config.epochs*len(dataloader)), 
                                 total_steps=config.epochs*len(dataloader), 
                                 t0 = config.epochs*len(dataloader)//5,
                                 gamma = config.schedule_ratio,
                                 step_size=len(dataloader))
    

    if DAB_separate_optim:
        optimizer_dab = torch.optim.AdamW(model.parameters(), lr=config.lr)
        scheduler_dab = torch.optim.lr_scheduler.StepLR(
            optimizer_dab, config.schedule_interval, gamma=config.schedule_ratio
        )
    else:
        optimizer_dab = None
        scheduler_dab = None

    if config.ADV:
        optimizer_E = torch.optim.AdamW(model.parameters(), lr=config.lr_ADV)
        scheduler_E = torch.optim.lr_scheduler.StepLR(
            optimizer_E, config.schedule_interval, gamma=config.schedule_ratio
        )
        optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=config.lr_ADV)
        scheduler_D = torch.optim.lr_scheduler.StepLR(
            optimizer_D, config.schedule_interval, gamma=config.schedule_ratio
        )
    else:
        optimizer_E = None
        scheduler_E = None
        optimizer_D = None
        scheduler_D = None

    optimizer_dict={
        "scaler": scaler,
        "discriminator": discriminator,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "optimizer_dab": optimizer_dab,
        "scheduler_dab": scheduler_dab,
        "optimizer_E": optimizer_E,
        "scheduler_E": scheduler_E,
        "optimizer_D": optimizer_D,
        "scheduler_D": scheduler_D,
        'DAB_separate_optim': DAB_separate_optim
    }
    return optimizer_dict
