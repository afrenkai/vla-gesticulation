import torch
import torchaudio
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Dict
import time

from multimodal_gesture_clip import (
    MultimodalGestureCLIP,
    MultimodalGestureConfig,
)


class SO100GestureController:
    """
    Real-time gesture controller for SO-100 robot using trained CLIP model

    Maps emotion-aware speech to upper body gestures in real-time
    """

    def __init__(
        self,
        model: MultimodalGestureCLIP,
        config: MultimodalGestureConfig,
        device: str = 'cuda',
        robot_config: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

        # Audio processing
        self.audio_resampler = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=config.audio_sample_rate
        )

        # Robot configuration for SO-100
        self.robot = None
        if robot_config:
            try:
                from lerobot.common.robot_devices.robots.factory import make_robot
                self.robot = make_robot(robot_config)
                print("✓ Connected to SO-100 robot")
            except Exception as e:
                print(f"Warning: Could not connect to robot: {e}")
                print("Running in simulation mode")

        # Joint limits for SO-100 upper body (12 joints)
        self.joint_limits = self._get_so100_joint_limits()

        # Rest position
        self.rest_position = np.array([
            0.0, 0.0, -0.5, 1.2, -0.7, 0.0,  # Left arm: shoulder_pan, shoulder_lift, elbow, wrist_flex, wrist_roll, gripper
            0.0, 0.0, -0.5, 1.2, -0.7, 0.0,  # Right arm
        ])

        # Gesture buffer for smoothing
        self.gesture_buffer = []
        self.buffer_size = 5

        # Emotion labels
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust']

    def _get_so100_joint_limits(self) -> np.ndarray:
        """Get joint limits for SO-100 upper body"""
        # Approximate limits for SO-100 arms (adjust based on actual robot specs)
        limits = np.array([
            [-2.6, 2.6],   # Left shoulder pan
            [-1.5, 1.5],   # Left shoulder lift
            [-2.8, 2.8],   # Left elbow
            [-1.5, 1.5],   # Left wrist flex
            [-3.0, 3.0],   # Left wrist roll
            [0.0, 1.0],    # Left gripper
            [-2.6, 2.6],   # Right shoulder pan
            [-1.5, 1.5],   # Right shoulder lift
            [-2.8, 2.8],   # Right elbow
            [-1.5, 1.5],   # Right wrist flex
            [-3.0, 3.0],   # Right wrist roll
            [0.0, 1.0],    # Right gripper
        ])
        return limits

    def clamp_to_limits(self, actions: np.ndarray) -> np.ndarray:
        """Clamp actions to joint limits"""
        clamped = np.clip(actions, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return clamped

    def smooth_gestures(self, gesture: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing to gestures"""
        self.gesture_buffer.append(gesture)
        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)

        # Moving average
        smoothed = np.mean(self.gesture_buffer, axis=0)
        return smoothed

    @torch.no_grad()
    def generate_gesture_from_audio(
        self,
        audio_path: str,
        predict_emotion: bool = True,
        smooth: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate gestures from audio file

        Args:
            audio_path: Path to audio file
            predict_emotion: Whether to predict emotion
            smooth: Whether to apply temporal smoothing

        Returns:
            Dictionary with:
                - gestures: (seq_len, 12) SO-100 joint positions
                - emotion: predicted emotion label
                - emotion_probs: emotion probabilities
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.config.audio_sample_rate:
            audio = self.audio_resampler(audio)

        # Prepare for model
        audio = audio.squeeze(0)  # Remove channel dimension
        audio = audio.unsqueeze(0).to(self.device)  # Add batch dimension

        # Generate gestures
        outputs = self.model.generate_gesture(
            audio=audio,
            predict_emotion=predict_emotion,
        )

        # Extract results
        gestures = outputs['gestures'][0].cpu().numpy()  # (seq_len, so100_joints * 6)

        # Reshape to (seq_len, 12) - taking only positions (first 2 DOF per joint)
        # For simplicity, we take pairs of consecutive values as joint positions
        num_frames = gestures.shape[0]
        so100_gestures = gestures[:, :12]  # First 12 values

        # Clamp to joint limits
        so100_gestures = np.array([self.clamp_to_limits(g) for g in so100_gestures])

        # Get emotion
        emotion_idx = None
        emotion_label = None
        emotion_probs = None

        if outputs['emotion_logits'] is not None:
            emotion_probs = torch.softmax(outputs['emotion_logits'][0], dim=-1).cpu().numpy()
            emotion_idx = np.argmax(emotion_probs)
            emotion_label = self.emotion_labels[emotion_idx]

        return {
            'gestures': so100_gestures,
            'emotion': emotion_label,
            'emotion_probs': emotion_probs,
            'emotion_idx': emotion_idx,
        }

    @torch.no_grad()
    def generate_gesture_from_text(
        self,
        text: str,
        predict_emotion: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate gestures from text

        Args:
            text: Input text
            predict_emotion: Whether to predict emotion

        Returns:
            Dictionary with gestures and emotion
        """
        # Tokenize text
        tokenizer = self.model.text_encoder.tokenizer
        text_inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors='pt'
        )

        text_input_ids = text_inputs['input_ids'].to(self.device)
        attention_mask = text_inputs['attention_mask'].to(self.device)

        # Generate gestures
        outputs = self.model.generate_gesture(
            text_input_ids=text_input_ids,
            attention_mask=attention_mask,
            predict_emotion=predict_emotion,
        )

        # Extract results
        gestures = outputs['gestures'][0].cpu().numpy()
        so100_gestures = gestures[:, :12]
        so100_gestures = np.array([self.clamp_to_limits(g) for g in so100_gestures])

        # Get emotion
        emotion_label = None
        emotion_probs = None

        if outputs['emotion_logits'] is not None:
            emotion_probs = torch.softmax(outputs['emotion_logits'][0], dim=-1).cpu().numpy()
            emotion_idx = np.argmax(emotion_probs)
            emotion_label = self.emotion_labels[emotion_idx]

        return {
            'gestures': so100_gestures,
            'emotion': emotion_label,
            'emotion_probs': emotion_probs,
        }

    def execute_gestures(
        self,
        gestures: np.ndarray,
        fps: int = 30,
        loop: bool = False,
    ):
        """
        Execute gestures on SO-100 robot

        Args:
            gestures: (seq_len, 12) joint positions
            fps: Playback frame rate
            loop: Whether to loop gestures
        """
        if self.robot is None:
            print("No robot connected, simulating execution...")
            for i, gesture in enumerate(gestures):
                print(f"Frame {i}: {gesture[:6]}")  # Print first 6 joints
                time.sleep(1.0 / fps)
            return

        # Execute on real robot
        frame_time = 1.0 / fps

        while True:
            for gesture in gestures:
                # Add zeros for base (SO-100 mobile base has 3 DOF)
                full_action = np.concatenate([gesture, np.zeros(3)])

                # Send to robot
                self.robot.send_action(full_action)

                # Wait for next frame
                time.sleep(frame_time)

            if not loop:
                break

        # Return to rest position
        rest_full = np.concatenate([self.rest_position, np.zeros(3)])
        self.robot.send_action(rest_full)

    def visualize_gestures(self, gestures: np.ndarray, save_path: Optional[str] = None):
        """Visualize gesture sequence (simple plot)"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle('SO-100 Gesture Sequence', fontsize=16)

            for joint_idx in range(12):
                ax = axes[joint_idx // 4, joint_idx % 4]
                joint_values = gestures[:, joint_idx]

                ax.plot(joint_values, linewidth=2)
                ax.set_title(f'Joint {joint_idx}')
                ax.set_xlabel('Frame')
                ax.set_ylabel('Position (rad)')
                ax.grid(True, alpha=0.3)

                # Show limits
                ax.axhline(y=self.joint_limits[joint_idx, 0], color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=self.joint_limits[joint_idx, 1], color='r', linestyle='--', alpha=0.5)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Saved visualization to {save_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            print("matplotlib not installed, skipping visualization")


def load_checkpoint(checkpoint_path: str, device: str = 'cuda') -> MultimodalGestureCLIP:
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = MultimodalGestureCLIP(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.4f}")

    return model, config


def main():
    parser = argparse.ArgumentParser(description='SO-100 Gesture Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, default=None,
                       help='Path to audio file')
    parser.add_argument('--text', type=str, default=None,
                       help='Input text')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize gestures')
    parser.add_argument('--execute', action='store_true',
                       help='Execute on real SO-100 robot')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Save visualization to path')
    parser.add_argument('--robot_port', type=str, default='/dev/ttyACM0',
                       help='Robot serial port')

    args = parser.parse_args()

    if args.audio is None and args.text is None:
        parser.error("Must provide either --audio or --text")

    print("="*60)
    print("SO-100 Gesture Generation")
    print("="*60)

    # Load model
    print("\nLoading model...")
    model, config = load_checkpoint(args.checkpoint, args.device)

    # Setup robot config if executing
    robot_config = None
    if args.execute:
        robot_config = {
            'robot_type': 'so100',
            'port': args.robot_port,
        }

    # Create controller
    controller = SO100GestureController(
        model=model,
        config=config,
        device=args.device,
        robot_config=robot_config,
    )

    # Generate gestures
    print("\nGenerating gestures...")
    if args.audio:
        print(f"  Input: Audio from {args.audio}")
        results = controller.generate_gesture_from_audio(args.audio)
    else:
        print(f"  Input: Text '{args.text}'")
        results = controller.generate_gesture_from_text(args.text)

    gestures = results['gestures']
    emotion = results['emotion']
    emotion_probs = results['emotion_probs']

    print(f"\nResults:")
    print(f"  Generated {len(gestures)} frames")
    print(f"  Duration: {len(gestures) / 30:.2f} seconds @ 30fps")
    print(f"  Predicted emotion: {emotion}")
    if emotion_probs is not None:
        print(f"  Emotion probabilities:")
        for label, prob in zip(controller.emotion_labels, emotion_probs):
            print(f"    {label}: {prob:.3f}")

    # Visualize
    if args.visualize or args.save_path:
        print("\nVisualizing gestures...")
        controller.visualize_gestures(gestures, save_path=args.save_path)

    # Execute on robot
    if args.execute:
        print("\nExecuting on SO-100 robot...")
        response = input("Ready to execute? (y/n): ")
        if response.lower() == 'y':
            controller.execute_gestures(gestures, fps=30, loop=False)
            print("✓ Execution complete")
        else:
            print("Execution cancelled")

    print("\nDone!")


if __name__ == "__main__":
    main()
