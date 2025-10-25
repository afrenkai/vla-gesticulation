import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import Wav2Vec2Model, BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader


@dataclass
class MultimodalGestureConfig:
    # Audio config
    audio_sample_rate: int = 16000
    audio_dim: int = 768  # Wav2Vec2 output

    # Text config
    text_dim: int = 768  # BERT output
    max_text_length: int = 128

    # Motion config
    motion_fps: int = 30
    bvh_joints: int = 83
    bvh_channels_per_joint: int = 6  # position + rotation
    motion_dim: int = 512  # Projected motion features

    # SO-100 upper body config
    so100_upper_joints: int = 12  # Estimate for SO-100 upper body

    # Embedding dimensions
    embedding_dim: int = 512  # Shared embedding space (CLIP-style)

    # Emotion config
    num_emotions: int = 7  # neutral, happy, sad, angry, surprised, fear, disgust

    # Training config
    temperature: float = 0.07  # Temperature for contrastive loss (CLIP default)
    batch_size: int = 32


class BVHParser:
    """Parse BVH motion capture files"""

    def __init__(self):
        self.joint_names = []
        self.joint_hierarchy = []
        self.frame_time = 0.0
        self.num_frames = 0

    def parse(self, bvh_path: str) -> np.ndarray:
        """
        Parse BVH file and extract motion data

        Returns:
            motion_data: (num_frames, num_channels) array
        """
        with open(bvh_path, 'r') as f:
            lines = f.readlines()

        # Find MOTION section
        motion_idx = None
        for i, line in enumerate(lines):
            if line.strip() == 'MOTION':
                motion_idx = i
                break

        if motion_idx is None:
            raise ValueError("No MOTION section found in BVH file")

        # Parse metadata
        self.num_frames = int(lines[motion_idx + 1].split(':')[1].strip())
        self.frame_time = float(lines[motion_idx + 2].split(':')[1].strip())

        # Parse motion data
        motion_start = motion_idx + 3
        motion_data = []
        for line in lines[motion_start:motion_start + self.num_frames]:
            values = [float(x) for x in line.strip().split()]
            motion_data.append(values)

        return np.array(motion_data, dtype=np.float32)

    def extract_upper_body_joints(self, motion_data: np.ndarray) -> np.ndarray:
        """
        Extract upper body joints relevant for SO-100

        From BVH hierarchy, we want:
        - Spine joints (b_spine0-3)
        - Neck and head (b_neck0, b_head)
        - Arms (left/right shoulder, elbow, wrist)

        Args:
            motion_data: (num_frames, 498) - 83 joints * 6 channels

        Returns:
            upper_body_motion: (num_frames, num_upper_joints * 6)
        """
        # Indices for upper body joints (approximate based on typical BVH structure)
        # body_world (6) + b_root (6) + b_spine0-3 (24) + b_neck0 (6) + b_head (6) = 48 channels
        # + left arm: shoulder(6) + elbow(6) + wrist(6) = 18
        # + right arm: shoulder(6) + elbow(6) + wrist(6) = 18
        # Total: ~84 channels for upper body

        # For simplicity, take first 84 channels (body root + spine + neck + arms)
        upper_body_channels = min(84, motion_data.shape[1])
        return motion_data[:, :upper_body_channels]


class TSVParser:
    """Parse TSV transcription files with timestamps"""

    @staticmethod
    def parse(tsv_path: str) -> List[Tuple[float, float, str]]:
        """
        Parse TSV file with time-aligned transcriptions

        Returns:
            List of (start_time, end_time, word) tuples
        """
        with open(tsv_path, 'r') as f:
            lines = f.readlines()

        transcription = []
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                start_time = float(parts[0])
                end_time = float(parts[1])
                word = parts[2]
                transcription.append((start_time, end_time, word))

        return transcription

    @staticmethod
    def get_full_text(transcription: List[Tuple[float, float, str]]) -> str:
        """Get full text from transcription"""
        return ' '.join([word for _, _, word in transcription])


class GENEAMultimodalDataset(Dataset):
    """
    Multimodal dataset for GENEA Challenge data
    Loads audio (wav), text (tsv), and motion (bvh) data
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'val',  # 'val' or 'tst'
        config: MultimodalGestureConfig = None,
        audio_resampler = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config or MultimodalGestureConfig()

        # Setup resampler
        if audio_resampler is None:
            self.audio_resampler = torchaudio.transforms.Resample(
                orig_freq=44100,
                new_freq=self.config.audio_sample_rate
            )
        else:
            self.audio_resampler = audio_resampler

        # Load metadata
        metadata_path = self.data_dir / split / f'{split}_metadata.csv'
        self.metadata = pd.read_csv(metadata_path, header=None,
                                   names=['filename', 'finger_incl', 'speaker_id'])

        # Initialize parsers
        self.bvh_parser = BVHParser()

        print(f"Loaded {len(self.metadata)} samples from {split} split")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        filename = row['filename']
        speaker_id = row['speaker_id']

        # Load audio
        wav_path = self.data_dir / self.split / 'wav' / f'{filename}.wav'
        audio, sr = torchaudio.load(str(wav_path))

        # Resample audio to 16kHz
        if sr != self.config.audio_sample_rate:
            audio = self.audio_resampler(audio)
        audio = audio.squeeze(0)  # Remove channel dimension

        # Load transcription
        tsv_path = self.data_dir / self.split / 'tsv' / f'{filename}.tsv'
        transcription = TSVParser.parse(str(tsv_path))
        text = TSVParser.get_full_text(transcription)

        # Load motion (if available)
        bvh_path = self.data_dir / self.split / 'bvh' / f'{filename}.bvh'
        motion = None
        if bvh_path.exists():
            motion_data = self.bvh_parser.parse(str(bvh_path))
            motion = self.bvh_parser.extract_upper_body_joints(motion_data)
            motion = torch.from_numpy(motion)

        return {
            'audio': audio,
            'text': text,
            'motion': motion,
            'transcription': transcription,
            'speaker_id': speaker_id,
            'filename': filename,
        }


class AudioEncoder(nn.Module):
    """Encode audio to embedding space using Wav2Vec2"""

    def __init__(self, config: MultimodalGestureConfig):
        super().__init__()
        self.config = config

        # Wav2Vec2 for audio encoding
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # Freeze Wav2Vec2 weights
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Project to embedding space
        self.projection = nn.Sequential(
            nn.Linear(config.audio_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: (batch, time) waveform at 16kHz

        Returns:
            embeddings: (batch, embedding_dim)
        """
        # Extract features with Wav2Vec2
        with torch.no_grad():
            outputs = self.wav2vec2(audio)
            features = outputs.last_hidden_state  # (batch, seq_len, 768)

        # Temporal pooling
        features = features.transpose(1, 2)  # (batch, 768, seq_len)
        pooled = self.temporal_pool(features).squeeze(-1)  # (batch, 768)

        # Project to embedding space
        embeddings = self.projection(pooled)

        # L2 normalize (CLIP-style)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class TextEncoder(nn.Module):
    """Encode text to embedding space using BERT"""

    def __init__(self, config: MultimodalGestureConfig):
        super().__init__()
        self.config = config

        # BERT for text encoding
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Freeze BERT weights
        for param in self.bert.parameters():
            param.requires_grad = False

        # Project to embedding space
        self.projection = nn.Sequential(
            nn.Linear(config.text_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def forward(self, text_input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            embeddings: (batch, embedding_dim)
        """
        # Extract features with BERT
        with torch.no_grad():
            outputs = self.bert(input_ids=text_input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token, (batch, 768)

        # Project to embedding space
        embeddings = self.projection(features)

        # L2 normalize (CLIP-style)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class MotionEncoder(nn.Module):
    """Encode motion sequence to embedding space"""

    def __init__(self, config: MultimodalGestureConfig):
        super().__init__()
        self.config = config

        # Temporal CNN for motion encoding
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(84, 256, kernel_size=5, padding=2),  # Upper body channels
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(512, config.motion_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(config.motion_dim),
            nn.ReLU(),
        )

        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # Project to embedding space
        self.projection = nn.Sequential(
            nn.Linear(config.motion_dim, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion: (batch, time, channels) upper body motion

        Returns:
            embeddings: (batch, embedding_dim)
        """
        # Transpose for Conv1d
        motion = motion.transpose(1, 2)  # (batch, channels, time)

        # Convolutional encoding
        features = self.conv_blocks(motion)

        # Temporal pooling
        pooled = self.temporal_pool(features).squeeze(-1)  # (batch, motion_dim)

        # Project to embedding space
        embeddings = self.projection(pooled)

        # L2 normalize (CLIP-style)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings


class EmotionClassifier(nn.Module):
    """Classify emotion from multimodal embeddings"""

    def __init__(self, config: MultimodalGestureConfig):
        super().__init__()
        self.config = config

        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.embedding_dim // 2, config.num_emotions),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embedding_dim)

        Returns:
            logits: (batch, num_emotions)
        """
        return self.classifier(embeddings)


class MotionDecoder(nn.Module):
    """Decode from multimodal embeddings to SO-100 gestures"""

    def __init__(self, config: MultimodalGestureConfig, output_seq_len: int = 60):
        super().__init__()
        self.config = config
        self.output_seq_len = output_seq_len

        # Expand embedding to sequence
        self.embedding_expander = nn.Linear(
            config.embedding_dim,
            config.motion_dim * output_seq_len
        )

        # Temporal transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.motion_dim,
            nhead=8,
            dim_feedforward=config.motion_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # Output projection to SO-100 joints
        self.output_projection = nn.Sequential(
            nn.Linear(config.motion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.so100_upper_joints * 6),  # 6 DOF per joint
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        emotion_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embedding_dim) multimodal embeddings
            emotion_logits: (batch, num_emotions) optional emotion conditioning

        Returns:
            gestures: (batch, output_seq_len, so100_upper_joints * 6)
        """
        batch_size = embeddings.shape[0]

        # Condition on emotion if provided
        if emotion_logits is not None:
            emotion_weights = F.softmax(emotion_logits, dim=-1)
            # Simple emotion conditioning by concatenation
            embeddings = embeddings + 0.1 * emotion_weights @ torch.randn(
                self.config.num_emotions,
                self.config.embedding_dim,
                device=embeddings.device
            )

        # Expand embedding to sequence
        expanded = self.embedding_expander(embeddings)  # (batch, motion_dim * seq_len)
        expanded = expanded.view(batch_size, self.output_seq_len, self.config.motion_dim)

        # Create learnable query tokens
        query = nn.Parameter(torch.randn(
            self.output_seq_len,
            self.config.motion_dim
        )).to(embeddings.device).unsqueeze(0).expand(batch_size, -1, -1)

        # Transformer decoding
        decoded = self.transformer_decoder(query, expanded)  # (batch, seq_len, motion_dim)

        # Project to gestures
        gestures = self.output_projection(decoded)  # (batch, seq_len, so100_joints * 6)

        return gestures


class MultimodalGestureCLIP(nn.Module):
    """
    CLIP-style multimodal model for emotion-aware gesticulation

    Learns a shared embedding space for:
    - Audio (speech)
    - Text (transcription)
    - Motion (gestures)

    Can classify emotions and generate SO-100 gestures
    """

    def __init__(self, config: MultimodalGestureConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.motion_encoder = MotionEncoder(config)

        # Emotion classifier
        self.emotion_classifier = EmotionClassifier(config)

        # Motion decoder for gesture generation
        self.motion_decoder = MotionDecoder(config)

        # Learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / config.temperature))

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder(audio)

    def encode_text(self, text_input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_input_ids, attention_mask)

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return self.motion_encoder(motion)

    def contrastive_loss(
        self,
        embeddings_1: torch.Tensor,
        embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        """
        CLIP-style contrastive loss

        Args:
            embeddings_1: (batch, embedding_dim)
            embeddings_2: (batch, embedding_dim)

        Returns:
            loss: scalar
        """
        # Cosine similarity
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * embeddings_1 @ embeddings_2.t()

        # Symmetric cross-entropy loss
        labels = torch.arange(len(embeddings_1), device=embeddings_1.device)
        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.t(), labels)

        return (loss_1 + loss_2) / 2

    def forward(
        self,
        audio: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        motion: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with contrastive learning

        Args:
            audio: (batch, time) waveforms
            text_input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            motion: (batch, time, channels)

        Returns:
            Dictionary with embeddings and losses
        """
        outputs = {}

        # Encode modalities
        if audio is not None:
            audio_emb = self.encode_audio(audio)
            outputs['audio_embeddings'] = audio_emb

        if text_input_ids is not None and attention_mask is not None:
            text_emb = self.encode_text(text_input_ids, attention_mask)
            outputs['text_embeddings'] = text_emb

        if motion is not None:
            motion_emb = self.encode_motion(motion)
            outputs['motion_embeddings'] = motion_emb

        # Compute contrastive losses
        total_loss = 0.0

        if 'audio_embeddings' in outputs and 'text_embeddings' in outputs:
            audio_text_loss = self.contrastive_loss(audio_emb, text_emb)
            outputs['audio_text_loss'] = audio_text_loss
            total_loss += audio_text_loss

        if 'audio_embeddings' in outputs and 'motion_embeddings' in outputs:
            audio_motion_loss = self.contrastive_loss(audio_emb, motion_emb)
            outputs['audio_motion_loss'] = audio_motion_loss
            total_loss += audio_motion_loss

        if 'text_embeddings' in outputs and 'motion_embeddings' in outputs:
            text_motion_loss = self.contrastive_loss(text_emb, motion_emb)
            outputs['text_motion_loss'] = text_motion_loss
            total_loss += text_motion_loss

        outputs['total_loss'] = total_loss

        return outputs

    def generate_gesture(
        self,
        audio: Optional[torch.Tensor] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        predict_emotion: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate SO-100 gestures from audio and/or text

        Args:
            audio: (batch, time) waveforms
            text_input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            predict_emotion: whether to predict and use emotion

        Returns:
            Dictionary with generated gestures and emotion predictions
        """
        # Encode input
        if audio is not None:
            embeddings = self.encode_audio(audio)
        elif text_input_ids is not None:
            embeddings = self.encode_text(text_input_ids, attention_mask)
        else:
            raise ValueError("Must provide either audio or text")

        # Predict emotion
        emotion_logits = None
        if predict_emotion:
            emotion_logits = self.emotion_classifier(embeddings)

        # Generate gestures
        gestures = self.motion_decoder(embeddings, emotion_logits)

        return {
            'gestures': gestures,
            'emotion_logits': emotion_logits,
            'embeddings': embeddings,
        }


if __name__ == "__main__":
    # Test the implementation
    config = MultimodalGestureConfig()

    # Create dataset
    dataset = GENEAMultimodalDataset(
        data_dir='genea_dataset',
        split='val',
        config=config
    )

    print(f"\nDataset size: {len(dataset)}")

    # Load sample
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  Audio shape: {sample['audio'].shape}")
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Motion shape: {sample['motion'].shape if sample['motion'] is not None else 'None'}")
    print(f"  Speaker ID: {sample['speaker_id']}")

    # Create model
    model = MultimodalGestureCLIP(config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_audio = torch.randn(2, 16000 * 3)  # 3 seconds
    dummy_motion = torch.randn(2, 90, 84)  # 3 seconds at 30fps

    # Tokenize dummy text
    tokenizer = model.text_encoder.tokenizer
    dummy_texts = ["Hello, how are you?", "I am doing great!"]
    text_inputs = tokenizer(
        dummy_texts,
        padding=True,
        truncation=True,
        max_length=config.max_text_length,
        return_tensors='pt'
    )

    # Forward pass for training
    outputs = model(
        audio=dummy_audio,
        text_input_ids=text_inputs['input_ids'],
        attention_mask=text_inputs['attention_mask'],
        motion=dummy_motion
    )

    print(f"\nTraining outputs:")
    print(f"  Total loss: {outputs['total_loss'].item():.4f}")
    print(f"  Audio-text loss: {outputs['audio_text_loss'].item():.4f}")
    print(f"  Audio-motion loss: {outputs['audio_motion_loss'].item():.4f}")

    # Test gesture generation
    with torch.no_grad():
        generated = model.generate_gesture(audio=dummy_audio[:1])
        print(f"\nGenerated gestures shape: {generated['gestures'].shape}")
        print(f"Emotion logits: {generated['emotion_logits']}")

        # Get emotion prediction
        emotion_pred = torch.argmax(generated['emotion_logits'], dim=-1)
        emotion_names = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust']
        print(f"Predicted emotion: {emotion_names[emotion_pred.item()]}")
