"""Single-model VLN agent aligned with the action-only 7B training dataset.

The model receives the global instruction and up to eight uniformly sampled
frames from the complete observation history.  It is never given a
sub-instruction and is expected to emit only a comma-separated action plan.
"""

import base64
import re
from collections import OrderedDict
from io import BytesIO
from typing import List, Sequence

import numpy as np
from habitat.core.agent import Agent
from PIL import Image

from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine


SYSTEM_PROMPT = "You are an intelligent navigation robot."
IMAGE_SIZE = (640, 480)
MAX_HISTORY_IMAGES = 8
ACTION_MAPPING = {
    "move forward 25 cm": 1,
    "turn left 15 degrees": 2,
    "turn right 15 degrees": 3,
    "stop": 0,
}


def global_uniform_indices(num_frames: int, max_images: int = MAX_HISTORY_IMAGES) -> List[int]:
    """Match the dataset's global uniform sampling exactly.

    Candidates are every frame from the episode start through the current
    observation.  Python's ``round`` is intentionally used because the dataset
    builder used the same expression.
    """
    count = min(int(num_frames), int(max_images))
    if count <= 0:
        return []
    if count == 1:
        return [0]
    return [round(index * (num_frames - 1) / (count - 1)) for index in range(count)]


def observation_history_description(num_images: int) -> str:
    """Return the exact action-dataset image description, including 1-image grammar."""
    if num_images == 1:
        return "Above is 1 image. It is the Observation history."
    return (
        f"Above are {num_images} images. They form the Observation history and are ordered from earlier "
        "to more recent views."
    )


def build_action_prompt(global_instruction: str, num_images: int) -> str:
    """Build the action-only prompt used by global-history M2 records."""
    if not 1 <= num_images <= MAX_HISTORY_IMAGES:
        raise ValueError(f"Expected 1-{MAX_HISTORY_IMAGES} images, got {num_images}.")
    return (
        f"{'<image>' * num_images}\n"
        f"{observation_history_description(num_images)}\n"
        f"Global Instruction: {global_instruction}\n"
        "Task: Provide the next 3 actions to execute. Available actions: "
        "1) Move forward 25 cm, 2) Turn left 15 degrees, 3) Turn right 15 degrees, 4) stop. "
        "Output stop only when the entire task is completed."
    )


def parse_action_string(action_text: str) -> List[int]:
    """Parse a model action plan into Habitat action ids (stop is 0)."""
    actions = []
    for part in re.split(r"[,，\n]", str(action_text).lower()):
        part = part.strip()
        if not part:
            continue
        for action_name, action_id in ACTION_MAPPING.items():
            if action_name in part:
                actions.append(action_id)
                break
        else:
            match = re.search(r"\b([1-4])\b", part)
            if match:
                action_id = int(match.group(1))
                actions.append(0 if action_id == 4 else action_id)
    return actions


class SingleActionAgent(Agent):
    """Habitat ``Agent`` that performs action-only, global-history inference."""

    def __init__(
        self,
        model_path: str = "",
        model_type: str = "qwen2_5_vl",
        device_map="auto",
        torch_dtype: str = "bfloat16",
        action_horizon: int = 3,
        max_tokens: int = 32,
        temperature: float = 0.0,
        image_size=IMAGE_SIZE,
        jpeg_quality: int = 85,
        server_url: str = None,
        server_model: str = "single_action",
        http_timeout_s: float = 120.0,
        image_data_url_cache_size: int = 512,
    ):
        if not model_path and not server_url:
            raise ValueError("Specify model_path for local inference or server_url for OpenAI-compatible inference.")

        self.image_size = tuple(image_size)
        self.action_horizon = max(1, int(action_horizon))
        self.jpeg_quality = max(1, min(int(jpeg_quality), 100))
        self.server_url = server_url.rstrip("/") if server_url else None
        self.server_model = server_model
        self.request_config = RequestConfig(max_tokens=max_tokens, temperature=temperature)
        self.http_client = None
        self.engine = None

        if self.server_url:
            import httpx

            self.http_client = httpx.Client(timeout=httpx.Timeout(float(http_timeout_s)))
        else:
            self.engine = TransformersEngine(
                model=model_path,
                model_type=model_type,
                torch_dtype=torch_dtype,
                device_map={"": 0} if device_map == "auto" else device_map,
            )

        self._image_data_url_cache = OrderedDict()
        self.image_data_url_cache_size = max(1, int(image_data_url_cache_size))
        self.reset()

    def close(self):
        if self.http_client is not None:
            self.http_client.close()
            self.http_client = None

    def reset(self):
        self.rgb_history: List[Image.Image] = []
        self.pending_action_list: List[int] = []
        self.last_action_str = ""
        self.last_prompt = ""
        self.last_frame_indices: List[int] = []
        self.just_predicted = False
        self._image_data_url_cache.clear()

    def _normalize_frame(self, rgb: np.ndarray) -> Image.Image:
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported RGB observation shape: {rgb.shape}")
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        return Image.fromarray(rgb).convert("RGB").resize(self.image_size)

    def _jpeg_round_trip(self, images: Sequence[Image.Image]) -> List[Image.Image]:
        """Match the local rollout image path (JPEG quality 85) used by the agent stack."""
        normalized = []
        for image in images:
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=self.jpeg_quality)
            with Image.open(BytesIO(buffer.getvalue())) as decoded:
                normalized.append(decoded.convert("RGB"))
        return normalized

    def select_global_history_images(self) -> List[Image.Image]:
        self.last_frame_indices = global_uniform_indices(len(self.rgb_history))
        return [self.rgb_history[index] for index in self.last_frame_indices]

    def _image_data_url(self, image: Image.Image, frame_index: int) -> str:
        cached = self._image_data_url_cache.get(frame_index)
        if cached is not None:
            self._image_data_url_cache.move_to_end(frame_index)
            return cached
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality)
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"
        self._image_data_url_cache[frame_index] = data_url
        self._image_data_url_cache.move_to_end(frame_index)
        while len(self._image_data_url_cache) > self.image_data_url_cache_size:
            self._image_data_url_cache.popitem(last=False)
        return data_url

    def _predict_local(self, prompt: str, images: Sequence[Image.Image]) -> str:
        request = InferRequest(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            images=self._jpeg_round_trip(images),
        )
        response = self.engine.infer([request], self.request_config)[0]
        return response.choices[0].message.content.strip()

    def _predict_server(self, prompt: str, images: Sequence[Image.Image]) -> str:
        content = [
            {"type": "image_url", "image_url": {"url": self._image_data_url(image, frame_index)}}
            for image, frame_index in zip(images, self.last_frame_indices)
        ]
        content.append({"type": "text", "text": re.sub(r"<image>", "", prompt).strip()})
        response = self.http_client.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "model": self.server_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "max_tokens": self.request_config.max_tokens,
                "temperature": self.request_config.temperature,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _global_instruction(observations) -> str:
        instruction = observations.get("instruction", "")
        if isinstance(instruction, dict):
            return str(instruction.get("text", ""))
        return str(instruction)

    def act(self, observations, info):
        self.rgb_history.append(self._normalize_frame(observations["rgb"]))
        self.just_predicted = False

        if not self.pending_action_list:
            images = self.select_global_history_images()
            self.last_prompt = build_action_prompt(self._global_instruction(observations), len(images))
            self.last_action_str = (
                self._predict_server(self.last_prompt, images)
                if self.server_url
                else self._predict_local(self.last_prompt, images)
            )
            self.pending_action_list = parse_action_string(self.last_action_str)[:self.action_horizon]
            if not self.pending_action_list:
                self.pending_action_list = [0]
            self.just_predicted = True

        return {"action": self.pending_action_list.pop(0)}
