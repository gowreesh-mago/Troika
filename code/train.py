#
import argparse
import os
import pickle
import pprint

import numpy as np
import torch
import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from model.model_factory import get_model
from parameters import parser

# from test import *
import test as test
from dataset import CompositionDataset, ActionAdverbDataset
from utils import *

# Wandb for logging (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")

def train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    model.train()
    best_metric = 0
    best_loss = 1e5
    best_epoch = 0
    final_model_state = None

    val_results = []

    scheduler = get_scheduler(optimizer, config, len(train_dataloader))

    # Get device
    device = next(model.parameters()).device

    # Prepare training indices based on model type
    if config.model_name == 'action_adverb_model':
        # For ActionAdverbModel: create tensors of all action/adverb indices
        all_action_indices = torch.arange(len(train_dataset.actions)).to(device)
        all_adverb_indices = torch.arange(len(train_dataset.adverbs)).to(device)
        train_pairs = None  # Not used for ActionAdverbModel
    else:
        # For Troika: create train pairs tensor
        attr2idx = train_dataset.attr2idx
        obj2idx = train_dataset.obj2idx
        train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                    for attr, obj in train_dataset.train_pairs]).to(device)
        all_action_indices = None
        all_adverb_indices = None

    train_losses = []
    global_step = 0

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_action_correct = 0
        epoch_adverb_correct = 0
        epoch_pair_correct = 0
        epoch_total = 0

        for bid, batch in enumerate(train_dataloader):

            # Forward pass depends on model type
            if config.model_name == 'action_adverb_model':
                predict = model(batch, all_action_indices, all_adverb_indices)
            else:
                predict = model(batch, train_pairs)

            loss = model.loss_calu(predict, batch)

            # Compute training accuracies
            if config.model_name == 'action_adverb_model':
                pair_logits, action_logits, adverb_logits = predict

                action_pred = action_logits.argmax(dim=1)
                adverb_pred = adverb_logits.argmax(dim=1)
                pair_pred = pair_logits.argmax(dim=1)

                action_gt = batch['action_idx'].to(device)
                adverb_gt = batch['adverb_idx'].to(device)
                pair_gt = batch['pair_idx'].to(device)

                epoch_action_correct += (action_pred == action_gt).sum().item()
                epoch_adverb_correct += (adverb_pred == adverb_gt).sum().item()
                epoch_pair_correct += (pair_pred == pair_gt).sum().item()
                epoch_total += len(action_gt)

            # normalize loss to account for batch accumulation
            loss = loss / config.gradient_accumulation_steps

            # backward pass
            loss.backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            scheduler = step_scheduler(scheduler, config, bid, len(train_dataloader))

            epoch_train_losses.append(loss.item())

            # Log to wandb every step
            if config.use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    'train/loss_step': loss.item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch': i,
                    'global_step': global_step
                }
                wandb.log(log_dict, step=global_step)

            global_step += 1

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

        progress_bar.close()

        # Compute epoch metrics
        epoch_loss = np.mean(epoch_train_losses)
        progress_bar.write(f"epoch {i+1} train loss {epoch_loss}")
        train_losses.append(epoch_loss)

        # Compute accuracies
        if config.model_name == 'action_adverb_model' and epoch_total > 0:
            train_action_acc = epoch_action_correct / epoch_total
            train_adverb_acc = epoch_adverb_correct / epoch_total
            train_pair_acc = epoch_pair_correct / epoch_total

            print(f"  Train Action Acc: {train_action_acc:.4f}")
            print(f"  Train Adverb Acc: {train_adverb_acc:.4f}")
            print(f"  Train Pair Acc: {train_pair_acc:.4f}")

            # Log to wandb
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'train/loss_epoch': epoch_loss,
                    'train/action_acc': train_action_acc,
                    'train/adverb_acc': train_adverb_acc,
                    'train/pair_acc': train_pair_acc,
                    'epoch': i
                }, step=global_step)

        if (i + 1) % config.save_every_n == 0:
            torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))

        print("Evaluating val dataset:")
        val_result = evaluate(model, val_dataset, config)
        val_results.append(val_result)

        # Log validation metrics to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            val_log_dict = {f'val/{k}': v for k, v in val_result.items()}
            val_log_dict['epoch'] = i
            wandb.log(val_log_dict, step=global_step)

        if config.val_metric == 'best_loss' and val_result.get('best_loss', val_result.get('loss', 1e5)) < best_loss:
            best_loss = val_result.get('best_loss', val_result.get('loss', 1e5))
            best_epoch = i
            print(f"  New best model at epoch {i+1}!")
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))
        if config.val_metric != 'best_loss' and val_result.get(config.val_metric, 0) > best_metric:
            best_metric = val_result[config.val_metric]
            best_epoch = i
            print(f"  New best model at epoch {i+1}!")
            torch.save(model.state_dict(), os.path.join(
                config.save_path, "val_best.pt"))

        final_model_state = model.state_dict()
        if i + 1 == config.epochs:
            print("--- Evaluating test dataset on Closed World ---")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "val_best.pt"
            )))
            evaluate(model, test_dataset, config)

    if config.save_final_model:
        torch.save(final_model_state, os.path.join(config.save_path, f'final_model.pt'))


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
            model, dataset, config)
    test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        if key in test_stats:
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
            test_saved_results[key] = round(test_stats[key], 4)

    # Add top-k accuracies for action-adverb model
    if config.model_name == 'action_adverb_model':
        top1_pair, top5_pair = compute_topk_accuracy(all_logits, all_pair_gt, k_values=[1, 5])
        test_saved_results['top1_pair_acc'] = round(top1_pair, 4)
        test_saved_results['top5_pair_acc'] = round(top5_pair, 4)
        result += f"top1  {round(top1_pair, 4)}| top5  {round(top5_pair, 4)}| "

        # Action and adverb accuracies
        action_acc = test_stats.get('attr_acc', 0)
        adverb_acc = test_stats.get('obj_acc', 0)
        test_saved_results['action_acc'] = round(action_acc, 4)
        test_saved_results['adverb_acc'] = round(adverb_acc, 4)

    print(result)
    test_saved_results['loss'] = loss_avg
    return test_saved_results

def compute_topk_accuracy(logits, targets, k_values=[1, 5]):
    """
    Compute top-k accuracy for given k values.

    Args:
        logits: (N, num_classes) tensor of logits
        targets: (N,) tensor of ground truth class indices
        k_values: list of k values to compute accuracy for

    Returns:
        tuple of accuracy values for each k
    """
    with torch.no_grad():
        maxk = max(k_values)
        batch_size = targets.size(0)

        # Get top-k predictions
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        accuracies = []
        for k in k_values:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc_k = correct_k.mul_(100.0 / batch_size).item()
            accuracies.append(acc_k)

        return tuple(accuracies)



if __name__ == "__main__":
    config = parser.parse_args()
    if config.yml_path:
        load_args(config.yml_path, config)
    print(config)
    # set the seed value
    set_seed(config.seed)

    # Initialize wandb if requested
    if config.use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb logging requested but wandb is not installed!")
            print("Install with: pip install wandb")
            print("Continuing without wandb logging...")
            config.use_wandb = False
        else:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=vars(config),
                reinit=True
            )
            print(f"Wandb initialized: {wandb.run.name}")

    # Set device (CUDA, MPS, or CPU)
    # Note: Using CPU for stability (MPS has some buffer allocation issues)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA')
    else:
        device = torch.device('cpu')
        print('Using CPU')

    dataset_path = config.dataset_path

    # Load appropriate dataset based on model type
    if config.model_name == 'action_adverb_model':
        print("Loading ActionAdverbDataset...")
        train_dataset = ActionAdverbDataset(
            data_dir=dataset_path,
            features_dir=config.features_dir,
            split='train'
        )
        val_dataset = ActionAdverbDataset(
            data_dir=dataset_path,
            features_dir=config.features_dir,
            split='val'
        )
        test_dataset = ActionAdverbDataset(
            data_dir=dataset_path,
            features_dir=config.features_dir,
            split='test'
        )

        # Extract vocabularies and dimensions for ActionAdverbModel
        action_vocab = train_dataset.actions
        adverb_vocab = train_dataset.adverbs
        flow_dim = train_dataset.flow_dim
        rgb_dim = train_dataset.rgb_dim

        model = get_model(
            config,
            action_vocab=action_vocab,
            adverb_vocab=adverb_vocab,
            flow_dim=flow_dim,
            rgb_dim=rgb_dim
        ).to(device)

    else:
        print("Loading CompositionDataset for Troika...")
        train_dataset = CompositionDataset(dataset_path,
                                           phase='train',
                                           split='compositional-split-natural',
                                           same_prim_sample=config.same_prim_sample)

        val_dataset = CompositionDataset(dataset_path,
                                         phase='val',
                                         split='compositional-split-natural')

        test_dataset = CompositionDataset(dataset_path,
                                           phase='test',
                                           split='compositional-split-natural')

        allattrs = train_dataset.attrs
        allobj = train_dataset.objs
        classes = [cla.replace(".", " ").lower() for cla in allobj]
        attributes = [attr.replace(".", " ").lower() for attr in allattrs]
        offset = len(attributes)

        model = get_model(config, attributes=attributes, classes=classes, offset=offset).to(device)

    # Test-only mode: load model and evaluate without training
    if config.test_only:
        if config.load_model is None:
            raise ValueError("--test_only requires --load_model to specify a checkpoint")

        print(f"Loading model from {config.load_model}")
        model.load_state_dict(torch.load(config.load_model, map_location=device))

        print("Evaluating on test dataset:")
        test_result = evaluate(model, test_dataset, config)

        print("\nTest Results:")
        for key, value in test_result.items():
            print(f"  {key}: {value}")

        # Log test results to wandb
        if config.use_wandb and WANDB_AVAILABLE:
            test_log_dict = {f'test/{k}': v for k, v in test_result.items()}
            wandb.log(test_log_dict)

        print("Testing complete!")
    else:
        # Normal training mode
        optimizer = get_optimizer(model, config)

        os.makedirs(config.save_path, exist_ok=True)

        train_model(model, optimizer, config, train_dataset, val_dataset, test_dataset)

        with open(os.path.join(config.save_path, "config.pkl"), "wb") as fp:
            pickle.dump(config, fp)
        write_json(os.path.join(config.save_path, "config.json"), vars(config))
        print("done!")

    # Finish wandb run
    if config.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
