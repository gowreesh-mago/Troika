import argparse

parser = argparse.ArgumentParser()


# model config
parser.add_argument("--model_name", help="model name", type=str)
parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
parser.add_argument("--dataset", help="name of the dataset", type=str, default='mit-states')
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--epoch_start", help="start epoch", default=0, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=48, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=16, type=int)
parser.add_argument("--num_workers", help="number of workers", default=4, type=int)
parser.add_argument("--context_length", help="sets the context length of the clip model", default=8, type=int)
parser.add_argument("--attr_dropout", help="add dropout to attributes", type=float, default=0.3)
parser.add_argument("--yml_path", help="yml path", type=str)
parser.add_argument("--clip_arch", help="clip path", type=str)
parser.add_argument("--dataset_path", help="dataset path", type=str)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n", default=5, type=int, help="saves the model every n epochs")
parser.add_argument("--save_final_model", help="indicate if you want to save the model state dict()", action="store_true")
parser.add_argument("--load_model", default=None, help="load the trained model")
parser.add_argument("--test_only", action="store_true", help="only run testing, no training")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=int)
parser.add_argument("--same_prim_sample", help="if sample same prim samples", action="store_true")

parser.add_argument("--open_world", help="evaluate on open world setup", default=False)
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)
parser.add_argument("--text_encoder_batch_size", help="batch size of the text encoder", default=16, type=int)
parser.add_argument('--threshold', type=float, default=None, help="optional threshold")
parser.add_argument('--threshold_trials', type=int, default=50, help="how many threshold values to try")

parser.add_argument("--adapter_dim", help="middle dimension of Adapter", type=int, default=64)
parser.add_argument("--init_lamda", help="lamda initialization value", type=float, default=0.1)
parser.add_argument("--cmt_layers", help="Number of layers in cross-attention", type=int, default=2)

# Optimizer and scheduler
parser.add_argument("--optimizer", help="optimizer type", type=str, default="AdamW", choices=["Adam", "SGD", "AdamW"])
parser.add_argument("--scheduler", help="learning rate scheduler", type=str, default="cosine_w_warmup", choices=["StepLR", "linear_w_warmup", "cosine_w_warmup"])
parser.add_argument("--warmup_proportion", help="warmup proportion for linear/cosine scheduler", type=float, default=0.01)
parser.add_argument("--step_size", help="step size for StepLR", type=int, default=5)
parser.add_argument("--gamma", help="gamma for StepLR", type=float, default=0.5)

# Action-Adverb model specific parameters
parser.add_argument("--features_dir", help="directory with pre-extracted I3D features", type=str)
parser.add_argument("--glove_path", help="path to GloVe embeddings (e.g., glove.6B.300d.txt)", type=str)
parser.add_argument("--hidden_dim", help="hidden dimension for model", type=int, default=768)
parser.add_argument("--num_heads", help="number of attention heads", type=int, default=8)
parser.add_argument("--temperature", help="temperature for similarity scaling", type=float, default=50.0)
parser.add_argument("--freeze_glove", help="freeze GloVe embeddings", action="store_true")
parser.add_argument("--cmt_dropout", help="dropout in cross-modal traction layers", type=float, default=0.1)
parser.add_argument("--dropout", help="dropout rate", type=float, default=0.3)

# Loss weights (same as Troika's pair/attr/obj weights)
parser.add_argument("--pair_loss_weight", help="weight for pair classification loss", type=float, default=1.0)
parser.add_argument("--action_loss_weight", help="weight for action classification loss", type=float, default=1.0)
parser.add_argument("--adverb_loss_weight", help="weight for adverb classification loss", type=float, default=1.0)

# Validation metric
parser.add_argument("--val_metric", help="validation metric for model selection", type=str, default="best_loss")

# Wandb logging
parser.add_argument("--use_wandb", action="store_true", help="use Weights & Biases for logging")
parser.add_argument("--wandb_project", type=str, default="action-adverb-classification", help="wandb project name")
parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity/team name")
parser.add_argument("--wandb_run_name", type=str, default=None, help="wandb run name")
