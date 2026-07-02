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

from typing import Optional

import chex
import jax
import jax.numpy as jnp
import pytest

from jumanji.environments.logic.drop_7 import utils
from jumanji.environments.logic.drop_7.env import Drop7
from jumanji.environments.logic.drop_7.types import State
from jumanji.testing.env_not_smoke import (
    check_env_does_not_smoke,
    check_env_specs_does_not_smoke,
)
from jumanji.testing.pytrees import assert_is_jax_array_tree
from jumanji.types import TimeStep


@pytest.fixture
def drop_7() -> Drop7:
    """Instantiates a default Drop7 environment."""
    return Drop7()


def make_board(values: chex.Array) -> chex.Array:
    """Create a one-hot Drop7 board from visible disk values."""
    return (
        jax.nn.one_hot(
            jnp.maximum(values, 1) - 1,
            utils.NUM_DISK_VALUES,
            dtype=jnp.int32,
        )
        * (values > 0)[..., None]
    )


def make_state(
    board: chex.Array,
    blocks: Optional[chex.Array] = None,
    current_disk: int = 1,
) -> State:
    if blocks is None:
        blocks = jnp.zeros((utils.BOARD_SIZE, utils.BOARD_SIZE), jnp.int32)
    action_mask = utils.get_action_mask(board)
    return State(
        board=board,
        blocks=blocks,
        current_disk=jnp.array(current_disk, jnp.int32),
        action_mask=action_mask,
        score=jnp.array(0, jnp.float32),
        step_count=jnp.array(0, jnp.int32),
        key=jax.random.PRNGKey(0),
    )


def test_drop_7__reset_jit(drop_7: Drop7) -> None:
    """Confirm that reset is jittable and only compiled once."""
    chex.clear_trace_counter()
    reset_fn = jax.jit(chex.assert_max_traces(drop_7.reset, n=1))
    key = jax.random.PRNGKey(0)
    state, timestep = reset_fn(key)

    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)
    assert_is_jax_array_tree(state)

    state, timestep = reset_fn(key)
    assert isinstance(timestep, TimeStep)
    assert isinstance(state, State)


def test_drop_7__step_jit(drop_7: Drop7) -> None:
    """Confirm that step is jittable and only compiled once."""
    state, _ = drop_7.reset(jax.random.PRNGKey(0))
    action = jnp.array(0, jnp.int32)

    chex.clear_trace_counter()
    step_fn = jax.jit(chex.assert_max_traces(drop_7.step, n=1))
    new_state, next_timestep = step_fn(state, action)

    assert isinstance(next_timestep, TimeStep)
    assert_is_jax_array_tree(new_state)
    assert jnp.sum(utils.occupied_cells(new_state.board)) == 1

    new_state, next_timestep = step_fn(new_state, action)
    assert isinstance(next_timestep, TimeStep)


def test_drop_7__resolve_row_clear() -> None:
    """Three adjacent 3-disks clear as in the notebook's reduce function."""
    values = jnp.zeros((utils.BOARD_SIZE, utils.BOARD_SIZE), jnp.int32)
    values = values.at[0, :3].set(3)
    board, blocks, reward = utils.resolve_board(
        make_board(values),
        jnp.zeros_like(values),
    )

    assert jnp.sum(utils.occupied_cells(board)) == 0
    assert jnp.sum(blocks) == 0
    assert reward == 3.0


def test_drop_7__resolve_column_clear_after_gravity() -> None:
    """Two 2-disks fall to the bottom and then clear vertically."""
    values = jnp.zeros((utils.BOARD_SIZE, utils.BOARD_SIZE), jnp.int32)
    values = values.at[2, 0].set(2)
    values = values.at[4, 0].set(2)
    board, _, reward = utils.resolve_board(
        make_board(values),
        jnp.zeros_like(values),
    )

    assert jnp.sum(utils.occupied_cells(board)) == 0
    assert reward == 2.0


def test_drop_7__resolve_unblocks_adjacent_hidden_disk() -> None:
    """A clear cracks an adjacent hidden disk and reveals its underlying value."""
    values = jnp.zeros((utils.BOARD_SIZE, utils.BOARD_SIZE), jnp.int32)
    values = values.at[0, 0].set(1)
    values = values.at[0, 1].set(4)
    blocks = jnp.zeros_like(values).at[0, 1].set(1)

    board, blocks, reward = utils.resolve_board(make_board(values), blocks)
    observation = utils.visible_board(board, blocks)

    assert reward == 1.0
    assert blocks[0, 1] == 0
    assert observation[0, 1] == 4


def test_drop_7__step_invalid_action_terminates(drop_7: Drop7) -> None:
    """Selecting a full column terminates and leaves the board unchanged."""
    values = jnp.zeros((utils.BOARD_SIZE, utils.BOARD_SIZE), jnp.int32)
    values = values.at[:, 0].set(7)
    state = make_state(make_board(values))

    new_state, timestep = jax.jit(drop_7.step)(state, jnp.array(0, jnp.int32))

    assert timestep.last()
    assert timestep.reward == 0.0
    assert jnp.array_equal(new_state.board, state.board)
    assert new_state.step_count == state.step_count + 1


def test_drop_7__observation_hides_blocked_disks(drop_7: Drop7) -> None:
    """Hidden disks are represented as -1 in observations."""
    values = jnp.zeros((utils.BOARD_SIZE, utils.BOARD_SIZE), jnp.int32)
    values = values.at[0, 0].set(5)
    blocks = jnp.zeros_like(values).at[0, 0].set(1)
    state = make_state(make_board(values), blocks=blocks)

    observation = drop_7._state_to_observation(state)

    assert observation.board[0, 0] == utils.HIDDEN_DISK
    assert observation.current_disk == state.current_disk


def test_drop_7__does_not_smoke(drop_7: Drop7) -> None:
    """Test that we can run an episode without errors."""
    check_env_does_not_smoke(drop_7)


def test_drop_7__specs_does_not_smoke(drop_7: Drop7) -> None:
    """Test that we can access specs without errors."""
    check_env_specs_does_not_smoke(drop_7)
