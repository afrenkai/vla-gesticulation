import torch
import torch.nn as nn
import torchaudio
from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.robots.configs import LeKiwiRobotConfig
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
import librosa
from transformers import Wav2Vec2Model


@dataclass
class LeKiwiAudioGestureConfig:
    audio_feature_dim: int = 768
    sample_rate: int = 16000
    max_audio_length: int = 16000 * 5  # 5 seconds
    
    arm_dof: int = 6  # SO-ARM101 has 6 degrees of freedom
    gesture_dim: int = 6  # Only control arm, not wheels
    
    hidden_dim: int = 512
    num_layers: int = 4
    dropout: float = 0.1
    temporal_context_window: int = 30  # frames
    audio_encoder = "Wav2Vec2Bert"
    robot_type: str = "lekiwi"
    robot_port: str = "/dev/ttyACM0"  # Adjust for your setup


class AudioEncoder(nn.Module):
    
    def __init__(self, config: LeKiwiAudioGestureConfig):
        super().__init__()
        self.config = config
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.audio_encoder)
        
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            
        self.projection = nn.Linear(config.audio_feature_dim, config.hidden_dim)
        
    def forward(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_waveform: (batch, time) audio at 16kHz
        Returns:
            audio_features: (batch, seq_len, hidden_dim)
        """
        outputs = self.wav2vec2(audio_waveform)
        audio_features = outputs.last_hidden_state  # (batch, seq_len, 768)
        
        audio_features = self.projection(audio_features)
        
        return audio_features


class GestureDecoder(nn.Module):
    def __init__(self, config: LeKiwiAudioGestureConfig):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.action_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.gesture_dim)
        )
        
        self.prosody_encoder = nn.Sequential(
            nn.Linear(3, 64),  # pitch, energy, rate
            nn.ReLU(),
            nn.Linear(64, config.hidden_dim)
        )
        
        self.gesture_styles = nn.Parameter(torch.randn(5, config.hidden_dim))  # emphasis, point, wave, rest, describe
        
    def forward(self, audio_features: torch.Tensor, 
                prosody_features: Optional[torch.Tensor] = None,
                style_idx: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            audio_features: (batch, seq_len, hidden_dim)
            prosody_features: (batch, seq_len, 3) optional prosody info
            style_idx: optional gesture style (0=emphasis, 1=point, 2=wave, 3=rest, 4=describe)
        Returns:
            gestures: (batch, seq_len, gesture_dim)
        """
        if prosody_features is not None:
            prosody_encoded = self.prosody_encoder(prosody_features)
            audio_features = audio_features + prosody_encoded
        
        if style_idx is not None:
            style_embed = self.gesture_styles[style_idx].unsqueeze(0).unsqueeze(0)
            audio_features = audio_features + style_embed
        
        features = self.transformer(audio_features)
        
        gestures = self.action_head(features)
        
        return gestures


class LeKiwiAudioGestureVLA(nn.Module, Policy):
    """
    VLA model for audio-to-gesticulation on LeKiwi mobile manipulator
    Maps speech audio to SO-ARM101 gestures while keeping base stationary
    
    The model generates natural co-speech gestures:
    - Emphasis gestures for stressed words
    - Rhythmic movements following speech prosody
    - Pointing and referential gestures
    - Rest position during silence
    """
    
    def __init__(self, config: LeKiwiAudioGestureConfig):
        super().__init__()
        self.config = config
        
        self.audio_encoder = AudioEncoder(config)
        self.gesture_decoder = GestureDecoder(config)
        
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=44100, 
            new_freq=config.sample_rate
        )
        
        self.joint_limits = torch.tensor([
            [-2.6, 2.6],   # shoulder_pan
            [-1.5, 1.5],   # shoulder_lift
            [-2.8, 2.8],   # elbow
            [-1.5, 1.5],   # wrist_flex
            [-3.0, 3.0],   # wrist_roll
            [0.0, 1.0]     # gripper
        ])
        
        self.rest_position = torch.tensor([0.0, -0.5, 1.2, -0.7, 0.0, 0.5])
        
        self.speech_threshold = 0.02  # RMS energy threshold
        
    def clamp_to_limits(self, actions: torch.Tensor) -> torch.Tensor:
        limits = self.joint_limits.to(actions.device)
        clamped = torch.clamp(actions, min=limits[:, 0], max=limits[:, 1])
        return clamped
    
    def detect_speech(self, audio: np.ndarray) -> bool:
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > self.speech_threshold
    
    def extract_prosody(self, audio: np.ndarray, sr: int = 16000) -> torch.Tensor:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
        
        energy = np.sqrt(np.mean(audio ** 2))
        
        zcr = np.mean(librosa.zero_crossings(audio))
        
        return torch.tensor([pitch / 500.0, energy * 10.0, zcr * 100.0], dtype=torch.float32)
    
    def preprocess_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        if sr != self.config.sample_rate:
            audio = self.resampler(audio)
        
        audio = audio / (torch.abs(audio).max() + 1e-8)
        
        if audio.shape[-1] > self.config.max_audio_length:
            audio = audio[..., :self.config.max_audio_length]
        else:
            padding = self.config.max_audio_length - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        
        return audio
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            batch: Dictionary containing:
                - 'audio': (batch, time) audio waveforms
                - 'action': (batch, seq_len, 6) target arm positions
                - 'sample_rate': audio sample rate
                - 'gesture_style': optional style index
        
        Returns:
            Dictionary with 'action' predictions and loss
        """
        audio = batch['audio']
        
        if 'sample_rate' in batch:
            sr = batch['sample_rate']
            audio = torch.stack([self.preprocess_audio(a, sr) for a in audio])
        
        audio_features = self.audio_encoder(audio)
        
        prosody = None
        if 'extract_prosody' in batch and batch['extract_prosody']:
            prosody_list = []
            for audio_sample in audio.cpu().numpy():
                p = self.extract_prosody(audio_sample, self.config.sample_rate)
                prosody_list.append(p)
            prosody = torch.stack(prosody_list).unsqueeze(1).repeat(1, audio_features.shape[1], 1)
            prosody = prosody.to(audio_features.device)
        
        style_idx = batch.get('gesture_style', None)
        
        gestures = self.gesture_decoder(audio_features, prosody, style_idx)
        
        gestures = self.clamp_to_limits(gestures)
        
        output = {'action': gestures}
        
        if 'action' in batch:
            target_gestures = batch['action']
            min_len = min(gestures.shape[1], target_gestures.shape[1])
            
            position_loss = nn.functional.mse_loss(gestures[:, :min_len], target_gestures[:, :min_len])
            
            velocity = gestures[:, 1:] - gestures[:, :-1]
            smoothness_loss = torch.mean(velocity ** 2)
            
            loss = position_loss + 0.1 * smoothness_loss
            output['loss'] = loss
        
        return output
    
    def select_action(self, observation: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Real-time action selection for deployment on LeKiwi
        
        Args:
            observation: Dictionary with 'audio' key containing recent audio buffer
        
        Returns:
            action: (6,) SO-ARM101 joint positions
        """
        self.eval()
        with torch.no_grad():
            audio = observation['audio']
            
            if isinstance(audio, torch.Tensor):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio
            
            is_speaking = self.detect_speech(audio_np)
            
            if not is_speaking:
                return self.rest_position.to(audio.device if isinstance(audio, torch.Tensor) else 'cpu')
            
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio).float()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            batch = {'audio': audio, 'sample_rate': self.config.sample_rate}
            output = self.forward(batch)
            
            action = output['action'][0, -1]
            
        return action


class LeKiwiAudioGestureTrainer:
    def __init__(self, model: LeKiwiAudioGestureVLA, lr: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(batch)
        loss = output['loss']
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {'loss': loss.item(), 'lr': self.scheduler.get_last_lr()[0]}


class LeKiwiGestureController:
    def __init__(self, model: LeKiwiAudioGestureVLA, robot_config: Optional[dict] = None):
        self.model = model
        self.model.eval()
        
        self.audio_buffer = []
        self.buffer_size = model.config.sample_rate * 2  # 2 second rolling buffer
        
        if robot_config:
            self.robot = make_robot(robot_config)
        else:
            self.robot = None
            
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        self.audio_buffer.extend(audio_chunk)
        if len(self.audio_buffer) > self.buffer_size:
            self.audio_buffer = self.audio_buffer[-self.buffer_size:]
    
    def get_current_gesture(self) -> np.ndarray:
        if len(self.audio_buffer) < 1000:  # Need minimum audio
            return self.model.rest_position.numpy()
        
        audio_tensor = torch.tensor(self.audio_buffer, dtype=torch.float32)
        observation = {'audio': audio_tensor}
        action = self.model.select_action(observation)
        
        return action.cpu().numpy()
    
    def execute_gesture(self, action: np.ndarray):
        if self.robot:
            full_action = np.concatenate([action, np.zeros(3)])  # zero matrix for wheels
            self.robot.send_action(full_action)


if __name__ == "__main__":
    config = LeKiwiAudioGestureConfig(
        arm_dof=6,
        gesture_dim=6,
        hidden_dim=512,
        num_layers=4,
        robot_port="/dev/ttyACM0"
    )
    
    model = LeKiwiAudioGestureVLA(config)
    
    print("LeKiwi Audio-to-Gesture VLA Model")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dummy_audio = torch.randn(2, 16000 * 3)  
    dummy_actions = torch.randn(2, 149, 6)  
    
    batch = {
        'audio': dummy_audio,
        'action': dummy_actions,
        'sample_rate': 16000,
        'extract_prosody': True
    }
    
    output = model(batch)
    print(f"\nTraining:")
    print(f"Output action shape: {output['action'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    
    observation = {'audio': torch.randn(16000)}
    action = model.select_action(observation)
    print(f"\nInference:")
    print(f"Action (SO-ARM101 joints): {action.numpy()}")
    
    controller = LeKiwiGestureController(model)
    
    for i in range(5):
        audio_chunk = np.random.randn(1600)  # 0.1 second chunks
        controller.add_audio_chunk(audio_chunk)
        gesture = controller.get_current_gesture()
        print(f"Step {i}: Gesture {gesture[:3]}...")  # Print first 3 joints

