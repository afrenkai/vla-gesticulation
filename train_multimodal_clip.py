import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

from multimodal_gesture_clip import (
    MultimodalGestureCLIP,
    MultimodalGestureConfig,
    GENEAMultimodalDataset,
)


class MultimodalCLIPTrainer:
    """Trainer for CLIP-style multimodal gesture model"""

    def __init__(
        self,
        model: MultimodalGestureCLIP,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: MultimodalGestureConfig,
        lr: float = 1e-4,
        device: str = 'cuda',
        log_dir: str = 'runs',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer - only train projection heads and decoder
        trainable_params = []
        for name, param in model.named_parameters():
            if 'wav2vec2' not in name and 'bert' not in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-6
        )

        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0

        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences"""
        # Find max audio length
        max_audio_len = max(item['audio'].shape[0] for item in batch)

        # Pad audio
        audio_batch = []
        for item in batch:
            audio = item['audio']
            if audio.shape[0] < max_audio_len:
                padding = max_audio_len - audio.shape[0]
                audio = F.pad(audio, (0, padding))
            audio_batch.append(audio)

        audio_batch = torch.stack(audio_batch)

        # Tokenize text
        texts = [item['text'] for item in batch]
        text_inputs = self.model.text_encoder.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors='pt'
        )

        # Pad motion sequences
        motion_batch = []
        max_motion_len = max(
            item['motion'].shape[0] for item in batch if item['motion'] is not None
        )

        for item in batch:
            motion = item['motion']
            if motion is not None:
                if motion.shape[0] < max_motion_len:
                    padding = max_motion_len - motion.shape[0]
                    motion = F.pad(motion, (0, 0, 0, padding))
                elif motion.shape[0] > max_motion_len:
                    motion = motion[:max_motion_len]
                motion_batch.append(motion)

        if motion_batch:
            motion_batch = torch.stack(motion_batch)
        else:
            motion_batch = None

        return {
            'audio': audio_batch,
            'text_input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'motion': motion_batch,
            'filenames': [item['filename'] for item in batch],
        }

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_audio_text_loss = 0.0
        total_audio_motion_loss = 0.0
        total_text_motion_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio = batch['audio'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            motion = batch['motion'].to(self.device) if batch['motion'] is not None else None

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                audio=audio,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                motion=motion,
            )

            loss = outputs['total_loss']

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            if 'audio_text_loss' in outputs:
                total_audio_text_loss += outputs['audio_text_loss'].item()
            if 'audio_motion_loss' in outputs:
                total_audio_motion_loss += outputs['audio_motion_loss'].item()
            if 'text_motion_loss' in outputs:
                total_text_motion_loss += outputs['text_motion_loss'].item()

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr'],
            })

            # TensorBoard logging
            if batch_idx % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                if 'audio_text_loss' in outputs:
                    self.writer.add_scalar('train/audio_text_loss', outputs['audio_text_loss'].item(), self.global_step)
                if 'audio_motion_loss' in outputs:
                    self.writer.add_scalar('train/audio_motion_loss', outputs['audio_motion_loss'].item(), self.global_step)

            self.global_step += 1

        # Epoch statistics
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_at_loss = total_audio_text_loss / num_batches
        avg_am_loss = total_audio_motion_loss / num_batches
        avg_tm_loss = total_text_motion_loss / num_batches

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Audio-Text Loss: {avg_at_loss:.4f}")
        print(f"  Avg Audio-Motion Loss: {avg_am_loss:.4f}")
        print(f"  Avg Text-Motion Loss: {avg_tm_loss:.4f}")

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_audio_text_loss = 0.0

        pbar = tqdm(self.val_loader, desc=f'Validation {epoch}')
        for batch in pbar:
            # Move to device
            audio = batch['audio'].to(self.device)
            text_input_ids = batch['text_input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            motion = batch['motion'].to(self.device) if batch['motion'] is not None else None

            # Forward pass
            outputs = self.model(
                audio=audio,
                text_input_ids=text_input_ids,
                attention_mask=attention_mask,
                motion=motion,
            )

            loss = outputs['total_loss']
            total_loss += loss.item()
            if 'audio_text_loss' in outputs:
                total_audio_text_loss += outputs['audio_text_loss'].item()

            pbar.set_postfix({'val_loss': loss.item()})

        # Validation statistics
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_at_loss = total_audio_text_loss / num_batches

        print(f"\nValidation Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Audio-Text Loss: {avg_at_loss:.4f}")

        # TensorBoard logging
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/audio_text_loss', avg_at_loss, epoch)

        return avg_loss

    def train(self, num_epochs: int, save_dir: str = 'checkpoints'):
        """Full training loop"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate(epoch)

            # Learning rate scheduling
            self.scheduler.step()

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': self.config,
            }

            # Save latest
            torch.save(checkpoint, save_dir / 'latest.pth')

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, save_dir / 'best.pth')
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

            # Save periodic checkpoint
            if epoch % 10 == 0:
                torch.save(checkpoint, save_dir / f'checkpoint_epoch_{epoch}.pth')

        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train Multimodal Gesture CLIP')
    parser.add_argument('--data_dir', type=str, default='genea_dataset',
                       help='Path to GENEA dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='Directory for tensorboard logs')

    args = parser.parse_args()

    # Setup logging directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.log_dir = f'runs/gesture_clip_{timestamp}'

    print("="*60)
    print("Multimodal Gesture CLIP Training")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print(f"Log directory: {args.log_dir}")
    print("="*60)

    # Create config
    config = MultimodalGestureConfig(
        batch_size=args.batch_size,
    )

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = GENEAMultimodalDataset(
        data_dir=args.data_dir,
        split='val',  # Using val as train since trn is corrupted
        config=config,
    )

    # Split validation dataset for train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print("\nInitializing model...")
    model = MultimodalGestureCLIP(config)

    # Create trainer
    trainer = MultimodalCLIPTrainer(
        model=model,
        train_loader=None,  # Will be set in train method
        val_loader=None,
        config=config,
        lr=args.lr,
        device=args.device,
        log_dir=args.log_dir,
    )

    # Create data loaders with custom collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=trainer.collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=trainer.collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
    )

    trainer.train_loader = train_loader
    trainer.val_loader = val_loader

    # Train
    print("\nStarting training...")
    trainer.train(num_epochs=args.num_epochs, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
