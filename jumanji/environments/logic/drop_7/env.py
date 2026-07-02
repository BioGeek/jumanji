# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cached_property
from typing import Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.animation as animation
from numpy.typing import NDArray

from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.logic.drop_7 import utils
from jumanji.environments.logic.drop_7.types import Observation, State
from jumanji.environments.logic.drop_7.viewer import Drop7Viewer
from jumanji.types import TimeStep, restart, termination, transition
from jumanji.viewer import Viewer


class Drop7(Environment[State, specs.DiscreteArray, Observation]):
    """A JAX implementation of Drop7.

    The board has seven rows and seven columns. The agent chooses a column in
    which to drop the current disk. Visible disks clear when their value equals
    the size of their contiguous horizontal or vertical run. Hidden gray disks
    store an underlying value and are revealed by adjacent clears.

    - observation: `Observation`
        - board: int32 array of shape (7, 7). Empty cells are 0, visible disks
            are 1 through 7, and hidden gray disks are -1.
        - current_disk: int32 scalar in [1, 7], the disk to drop next.
        - action_mask: bool array of shape (7,), indicating columns that are not full.
        - step_count: int32 scalar, number of elapsed steps.

    - action: int32 scalar in [0, 6], the column in which to drop the disk.

    - reward: float32 scalar, the number of visible disks cleared by the move.

    - episode termination: selecting a full column, filling every column, or
        reaching the time limit.
    """

    def __init__(
        self,
        time_limit: int = 1000,
        num_initial_hidden_rows: int = 0,
        viewer: Optional[Viewer[State]] = None,
    ) -> None:
        """Instantiate a Drop7 environment.

        Args:
            time_limit: maximum number of environment steps before termination.
            num_initial_hidden_rows: number of bottom rows filled with hidden gray
                disks on reset. Must be in [0, 6].
            viewer: `Viewer` used for rendering. Defaults to `Drop7Viewer`.
        """
        if not 0 <= num_initial_hidden_rows < utils.BOARD_SIZE:
            raise ValueError(
                "`num_initial_hidden_rows` must be in [0, 6], "
                f"got {num_initial_hidden_rows}."
            )
        self.time_limit = time_limit
        self.num_initial_hidden_rows = num_initial_hidden_rows
        super().__init__()
        self._viewer = viewer or Drop7Viewer()

    def __repr__(self) -> str:
        return "\n".join(
            [
                "Drop7 environment:",
                f" - time_limit: {self.time_limit}",
                f" - num_initial_hidden_rows: {self.num_initial_hidden_rows}",
            ]
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        """Reset the environment."""
        key, board_key, disk_key = jax.random.split(key, 3)
        board, blocks = self._generate_initial_board(board_key)
        current_disk = utils.sample_disk(disk_key)
        action_mask = utils.get_action_mask(board)
        state = State(
            board=board,
            blocks=blocks,
            current_disk=current_disk,
            action_mask=action_mask,
            score=jnp.array(0, jnp.float32),
            step_count=jnp.array(0, jnp.int32),
            key=key,
        )
        timestep = restart(observation=self._state_to_observation(state))
        return state, timestep

    def step(
        self, state: State, action: chex.Array
    ) -> Tuple[State, TimeStep[Observation]]:
        """Run one timestep of Drop7 dynamics."""
        key, disk_key = jax.random.split(state.key)
        is_valid = state.action_mask[action]
        next_disk = utils.sample_disk(disk_key)

        def valid_step() -> Tuple[chex.Array, chex.Array, chex.Array]:
            board = utils.drop_disk(state.board, action, state.current_disk)
            board, blocks, reward = utils.resolve_board(board, state.blocks)
            return board, blocks, reward

        board, blocks, reward = jax.lax.cond(
            is_valid,
            valid_step,
            lambda: (
                state.board,
                state.blocks,
                jnp.array(0, jnp.float32),
            ),
        )
        action_mask = utils.get_action_mask(board)
        step_count = state.step_count + 1
        next_state = State(
            board=board,
            blocks=blocks,
            current_disk=jnp.where(is_valid, next_disk, state.current_disk),
            action_mask=action_mask,
            score=state.score + reward,
            step_count=step_count,
            key=key,
        )
        observation = self._state_to_observation(next_state)
        done = (~is_valid) | (~jnp.any(action_mask)) | (step_count >= self.time_limit)
        timestep = jax.lax.cond(
            done,
            termination,
            transition,
            reward,
            observation,
        )
        return next_state, timestep

    @cached_property
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the Drop7 observation."""
        return specs.Spec(
            Observation,
            "ObservationSpec",
            board=specs.BoundedArray(
                shape=(utils.BOARD_SIZE, utils.BOARD_SIZE),
                dtype=jnp.int32,
                minimum=utils.HIDDEN_DISK,
                maximum=utils.NUM_DISK_VALUES,
                name="board",
            ),
            current_disk=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=1,
                maximum=utils.NUM_DISK_VALUES,
                name="current_disk",
            ),
            action_mask=specs.BoundedArray(
                shape=(utils.BOARD_SIZE,),
                dtype=bool,
                minimum=False,
                maximum=True,
                name="action_mask",
            ),
            step_count=specs.BoundedArray(
                shape=(),
                dtype=jnp.int32,
                minimum=0,
                maximum=self.time_limit,
                name="step_count",
            ),
        )

    @cached_property
    def action_spec(self) -> specs.DiscreteArray:
        """Returns the column action spec."""
        return specs.DiscreteArray(utils.BOARD_SIZE, name="action", dtype=jnp.int32)

    def _generate_initial_board(
        self, key: chex.PRNGKey
    ) -> Tuple[chex.Array, chex.Array]:
        disk_values = jax.random.randint(
            key,
            shape=(utils.BOARD_SIZE, utils.BOARD_SIZE),
            minval=1,
            maxval=utils.NUM_DISK_VALUES + 1,
            dtype=jnp.int32,
        )
        hidden_rows = jnp.arange(utils.BOARD_SIZE) < self.num_initial_hidden_rows
        hidden_mask = jnp.broadcast_to(hidden_rows[:, None], disk_values.shape)
        board = jax.nn.one_hot(disk_values - 1, utils.NUM_DISK_VALUES, dtype=jnp.int32)
        board = jnp.where(hidden_mask[..., None], board, jnp.zeros_like(board))
        blocks = hidden_mask.astype(jnp.int32)
        return board, blocks

    def _state_to_observation(self, state: State) -> Observation:
        return Observation(
            board=utils.visible_board(state.board, state.blocks),
            current_disk=state.current_disk,
            action_mask=state.action_mask,
            step_count=state.step_count,
        )

    def render(self, state: State) -> Optional[NDArray]:
        """Render the current board state."""
        return self._viewer.render(state)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> animation.FuncAnimation:
        """Animate a sequence of Drop7 states."""
        return self._viewer.animate(
            states=states, interval=interval, save_path=save_path
        )

    def close(self) -> None:
        """Perform viewer cleanup."""
        self._viewer.close()
