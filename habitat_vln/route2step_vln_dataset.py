"""Route2Step-compatible R2R/RxR dataset loader for Habitat-Lab v0.2.4."""

import json
import os
from typing import Optional

from habitat.core.registry import registry
from habitat.datasets.utils import VocabDict
from habitat.datasets.vln.r2r_vln_dataset import VLNDatasetV1
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"


@registry.register_dataset(name="Route2StepVLNDataset-v1")
class Route2StepVLNDatasetV1(VLNDatasetV1):
    """Load standard R2R files and RxR guide files with one schema."""

    def from_json(self, json_str: str, scenes_dir: Optional[str] = None) -> None:
        payload = json.loads(json_str)
        vocab = payload.get("instruction_vocab") or {}
        self.instruction_vocab = VocabDict(word_list=vocab.get("word_list", []))

        for raw_episode in payload.get("episodes", []):
            episode_data = dict(raw_episode)
            raw_instruction = episode_data.pop("instruction", {})
            if isinstance(raw_instruction, InstructionData):
                instruction = raw_instruction
            elif isinstance(raw_instruction, dict):
                instruction = InstructionData(
                    instruction_text=raw_instruction.get("instruction_text", ""),
                    instruction_tokens=raw_instruction.get("instruction_tokens"),
                )
            else:
                instruction = InstructionData(instruction_text=str(raw_instruction))

            # Some RxR-derived files omit this R2R-specific identifier.
            episode_data.setdefault("trajectory_id", 0)
            episode = VLNEpisode(instruction=instruction, **episode_data)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = [NavigationGoal(**goal) if isinstance(goal, dict) else goal for goal in episode.goals]
            self.episodes.append(episode)
