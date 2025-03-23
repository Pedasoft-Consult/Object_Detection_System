import os
import yaml
import torch
import argparse
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging

# Import local modules
from src.data.data_loader import create_data_loaders
from src.models.yolo import create_model
from src.models.loss import YOLOLoss
from src.utils.metrics import calculate_map
from src.utils.visualization import visualize_batch


def setup_logging(config):
    """
    Set up logging configuration

    Args:
        config (dict): Main configuration

    Returns:
        logger: Configured logger
    """
    log_dir = os.path.join(config['logs_dir'], 'training')
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    return logger


def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch, logger, writer=None):
    """
    Train the model for one epoch

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to use (cuda/cpu)
        epoch: Current epoch
        logger: Logger
        writer: TensorBoard writer

    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Move data to device
        images = batch['images'].to(device)
        targets = []
        for boxes, labels in zip(batch['boxes'], batch['labels']):
            boxes = torch.tensor(boxes, device=device)
            labels = torch.tensor(labels, device=device)
            targets.append({'boxes': boxes, 'labels': labels})

        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)

        # Calculate loss
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log batch loss
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

        # Log to TensorBoard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * num_batches + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)

            # Log learning rate
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', lr, global_step)

            # Visualize predictions every 100 batches
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    # Get predictions for visualization
                    vis_image = visualize_batch(images[0].cpu(), predictions[0].cpu(), targets[0])
                    writer.add_image('train/predictions', vis_image, global_step)

    avg_loss = total_loss / num_batches
    logger.info(f'Epoch {epoch + 1}: Average train loss = {avg_loss:.4f}')

    return avg_loss


def validate(model, val_loader, loss_fn, device, epoch, logger, writer=None):
    """
    Validate the model

    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to use (cuda/cpu)
        epoch: Current epoch
        logger: Logger
        writer: TensorBoard writer

    Returns:
        tuple: (Average loss, mAP)
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Validation {epoch + 1}', leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch['images'].to(device)
            targets = []
            for boxes, labels in zip(batch['boxes'], batch['labels']):
                boxes = torch.tensor(boxes, device=device)
                labels = torch.tensor(labels, device=device)
                targets.append({'boxes': boxes, 'labels': labels})

            # Forward pass
            predictions = model(images)

            # Calculate loss
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

            # Store predictions and targets for mAP calculation
            all_predictions.extend(predictions)
            all_targets.extend(targets)

            # Visualize predictions
            if writer and batch_idx == 0:
                vis_image = visualize_batch(images[0].cpu(), predictions[0].cpu(), targets[0])
                writer.add_image('val/predictions', vis_image, epoch)

    # Calculate average loss
    avg_loss = total_loss / num_batches

    # Calculate mAP
    mAP = calculate_map(all_predictions, all_targets)

    logger.info(f'Epoch {epoch + 1}: Validation loss = {avg_loss:.4f}, mAP = {mAP:.4f}')

    if writer:
        writer.add_scalar('val/loss', avg_loss, epoch)
        writer.add_scalar('val/mAP', mAP, epoch)

    return avg_loss, mAP


def save_checkpoint(model, optimizer, epoch, loss, mAP, checkpoint_dir, is_best=False):
    """
    Save model checkpoint

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Validation loss
        mAP: Mean Average Precision
        checkpoint_dir: Directory to save checkpoint
        is_best: Whether this is the best model so far
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': loss,
        'mAP': mAP
    }

    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)

    # Save best model if required
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)


def main(config_path):
    """
    Main training function

    Args:
        config_path (str): Path to the main configuration file
    """
    # Load configurations
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(os.path.join(os.path.dirname(config_path), 'model_config.yaml'), 'r') as f:
        model_config = yaml.safe_load(f)

    with open(os.path.join(os.path.dirname(config_path), 'data_config.yaml'), 'r') as f:
        data_config = yaml.safe_load(f)

    # Set up logging
    logger = setup_logging(config)
    logger.info("Starting training...")

    # Set random seeds for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])

    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(data_config)
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")

    # Create model
    logger.info("Creating model...")
    model = create_model(model_config, num_classes=len(data_config['dataset']['classes']))
    model = model.to(device)

    # Create loss function
    loss_fn = YOLOLoss(
        model_config['model']['num_classes'],
        device=device,
        box_weight=model_config['model']['loss']['box_weight'],
        obj_weight=model_config['model']['loss']['obj_weight'],
        cls_weight=model_config['model']['loss']['cls_weight'],
        iou_type=model_config['model']['loss']['iou_type']
    )

    # Create optimizer
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=0.9,
            weight_decay=config['training']['weight_decay']
        )

    # Create learning rate scheduler
    if config['training']['lr_scheduler'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif config['training']['lr_scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:  # plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=config['training']['early_stopping_patience'] // 2,
            verbose=True
        )

    # Set up TensorBoard
    if config['logging']['tensorboard']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_log_dir = os.path.join(config['logs_dir'], 'tensorboard', f'run_{timestamp}')
        writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logs will be saved to {tb_log_dir}")
    else:
        writer = None

    # Training loop
    logger.info("Starting training loop...")
    best_map = 0.0
    early_stopping_counter = 0
    checkpoint_dir = os.path.join(config['models_dir'], 'checkpoints')

    for epoch in range(config['training']['num_epochs']):
        start_time = time.time()

        # Train one epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, logger, writer
        )

        # Validate
        val_loss, mAP = validate(
            model, val_loader, loss_fn, device, epoch, logger, writer
        )

        # Update learning rate
        if config['training']['lr_scheduler'] == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Check if this is the best model
        is_best = mAP > best_map
        if is_best:
            best_map = mAP
            early_stopping_counter = 0
            logger.info(f"New best model with mAP: {best_map:.4f}")
        else:
            early_stopping_counter += 1

        # Save checkpoint
        if (epoch + 1) % config['logging']['save_checkpoint_freq'] == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_loss, mAP, checkpoint_dir, is_best
            )

        # Early stopping
        if early_stopping_counter >= config['training']['early_stopping_patience']:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Log epoch time
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

    # Save final model
    final_model_dir = os.path.join(config['models_dir'], 'final')
    os.makedirs(final_model_dir, exist_ok=True)

    # Load best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model with mAP: {checkpoint['mAP']:.4f}")

    # Save final model in PyTorch format
    torch.save(model.state_dict(), os.path.join(final_model_dir, 'model.pt'))

    # Export model to ONNX if enabled
    if model_config['optimization']['export']['format'] == 'onnx':
        dummy_input = torch.randn(1, 3,
                                  data_config['preprocessing']['resize']['height'],
                                  data_config['preprocessing']['resize']['width']).to(device)
        onnx_path = os.path.join(final_model_dir, 'model.onnx')

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=model_config['optimization']['export']['opset_version'],
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )

        logger.info(f"Exported model to ONNX format: {onnx_path}")

    logger.info("Training completed successfully!")

    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO object detection model")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    main(args.config)