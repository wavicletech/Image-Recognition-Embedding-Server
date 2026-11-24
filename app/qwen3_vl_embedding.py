"""Qwen3-VL-Embedding loader.

Adapted from QwenLM/Qwen3-VL-Embedding src/models/qwen3_vl_embedding.py.
FlashAttention is not used: RTX 2080 is Turing (sm_75) and only has fp16 tensor cores.
"""

import os
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
# Match Qwen/Qwen3-VL-Embedding-2B preprocessor_config.json. Any larger image is
# resized down to this, keeping aspect ratio, which is what Simplismart does.
MIN_PIXELS = 4096
MAX_PIXELS = 1310720
FPS = 1.0
MAX_FRAMES = 64
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS

MODEL_FORWARD_KEYS = {
    "input_ids",
    "attention_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "cache_position",
    "logits_to_keep",
    "mm_token_type_ids",
}


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.visual

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLForEmbeddingOutput]:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def sample_frames(frames, max_segments):
    duration = len(frames)
    if duration <= max_segments:
        return frames
    frame_id_list = np.linspace(0, duration - 1, max_segments, dtype=int).tolist()
    return [frames[frame_idx] for frame_idx in frame_id_list]


def is_image_path(path: str) -> bool:
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"}
    if path.startswith(("http://", "https://")):
        clean_path = urlparse(path).path
    else:
        clean_path = path
    _, ext = os.path.splitext(clean_path.lower())
    return ext in image_extensions


def is_video_input(video) -> bool:
    if isinstance(video, str):
        return True
    if isinstance(video, list) and video:
        first_elem = video[0]
        if isinstance(first_elem, Image.Image):
            return True
        if isinstance(first_elem, str):
            return is_image_path(first_elem)
    return False


def _move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class Qwen3VLEmbedder:
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        total_pixels: int = MAX_TOTAL_PIXELS,
        fps: float = FPS,
        max_frames: int = MAX_FRAMES,
        default_instruction: str = "Represent the user's input.",
        **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.fps = fps
        self.max_frames = max_frames
        self.default_instruction = default_instruction

        kwargs.setdefault("dtype", torch.float16)
        kwargs.setdefault("attn_implementation", "sdpa")
        kwargs.setdefault("low_cpu_mem_usage", True)

        self.model = Qwen3VLForEmbedding.from_pretrained(
            model_name_or_path, trust_remote_code=True, **kwargs
        ).to(device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_name_or_path, padding_side="right"
        )
        self.model.eval()

    @torch.no_grad()
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        model_inputs = {key: value for key, value in inputs.items() if key in MODEL_FORWARD_KEYS}
        outputs = self.model(**model_inputs)
        return {
            "last_hidden_state": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask"),
        }

    def _truncate_tokens(self, token_ids: List[int], max_length: int) -> List[int]:
        if len(token_ids) <= max_length:
            return token_ids
        special_token_ids = set(self.processor.tokenizer.all_special_ids)
        num_special = sum(1 for token_idx in token_ids if token_idx in special_token_ids)
        num_non_special_to_keep = max_length - num_special
        final_token_ids = []
        non_special_kept_count = 0
        for token_idx in token_ids:
            if token_idx in special_token_ids:
                final_token_ids.append(token_idx)
            elif non_special_kept_count < num_non_special_to_keep:
                final_token_ids.append(token_idx)
                non_special_kept_count += 1
        return final_token_ids

    def format_model_input(
        self,
        text=None,
        image=None,
        video=None,
        instruction=None,
        fps=None,
        max_frames=None,
    ) -> List[Dict]:
        if instruction:
            instruction = instruction.strip()
            if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
                instruction = instruction + "."

        content = []
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction or self.default_instruction}]},
            {"role": "user", "content": content},
        ]

        texts = [] if text is None else ([text] if isinstance(text, str) else text)
        if image is None:
            images = []
        elif not isinstance(image, list):
            images = [image]
        else:
            images = image
        if video is None:
            videos = []
        elif is_video_input(video):
            videos = [video]
        else:
            videos = video

        if not texts and not images and not videos:
            content.append({"type": "text", "text": "NULL"})
            return conversation

        for vid in videos:
            video_kwargs = {"total_pixels": self.total_pixels}
            if isinstance(vid, list):
                video_content = vid
                if self.max_frames is not None:
                    video_content = sample_frames(video_content, self.max_frames)
                video_content = [
                    ("file://" + ele if isinstance(ele, str) else ele) for ele in video_content
                ]
            elif isinstance(vid, str):
                video_content = vid if vid.startswith(("http://", "https://")) else "file://" + vid
                video_kwargs = {"fps": fps or self.fps, "max_frames": max_frames or self.max_frames}
            else:
                raise TypeError(f"Unrecognized video type: {type(vid)}")
            if video_content:
                content.append({"type": "video", "video": video_content, **video_kwargs})

        for img in images:
            if isinstance(img, Image.Image):
                image_content = img
            elif isinstance(img, str):
                image_content = img if img.startswith(("http://", "https://", "file://")) else "file://" + img
            else:
                raise TypeError(f"Unrecognized image type: {type(img)}")
            content.append(
                {
                    "type": "image",
                    "image": image_content,
                    "min_pixels": self.min_pixels,
                    "max_pixels": self.max_pixels,
                }
            )

        for txt in texts:
            content.append({"type": "text", "text": txt})
        return conversation

    def _preprocess_inputs(self, conversations: List[List[Dict]], add_generation_prompt: bool = True) -> Dict[str, Any]:
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        images, video_inputs, video_kwargs = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos = list(videos)
            video_metadata = list(video_metadata)
        else:
            videos, video_metadata = None, None

        return self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=False,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )

    @staticmethod
    def input_token_count(processed_inputs: Dict[str, Any]) -> int:
        mask = processed_inputs.get("attention_mask")
        if torch.is_tensor(mask):
            return int(mask.sum().item())
        ids = processed_inputs.get("input_ids")
        if torch.is_tensor(ids):
            return int(ids.shape[-1])
        return 0

    @staticmethod
    def _pooling_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_tensor = attention_mask.flip(dims=[1])
        last_one_positions = flipped_tensor.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def _embed_processed(self, processed_inputs: Dict[str, Any], normalize: bool = True) -> torch.Tensor:
        processed_inputs = _move_to_device(processed_inputs, self.model.device)
        outputs = self.forward(processed_inputs)
        embeddings = self._pooling_last(outputs["last_hidden_state"], outputs["attention_mask"])
        if normalize:
            embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
        return embeddings

    def process(self, inputs: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        conversations = [
            self.format_model_input(
                text=ele.get("text"),
                image=ele.get("image"),
                video=ele.get("video"),
                instruction=ele.get("instruction"),
                fps=ele.get("fps"),
                max_frames=ele.get("max_frames"),
            )
            for ele in inputs
        ]
        processed_inputs = self._preprocess_inputs(conversations, add_generation_prompt=True)
        return self._embed_processed(processed_inputs, normalize=normalize)

    def embed_conversations(
        self,
        conversations: List[List[Dict]],
        add_generation_prompt: bool = True,
        normalize: bool = True,
    ) -> torch.Tensor:
        processed_inputs = self._preprocess_inputs(
            conversations, add_generation_prompt=add_generation_prompt
        )
        return self._embed_processed(processed_inputs, normalize=normalize)
