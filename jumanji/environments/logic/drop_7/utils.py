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

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

BOARD_SIZE = 7
NUM_DISK_VALUES = 7
HIDDEN_DISK = -1
EMPTY = 0


def sample_disk(key: chex.PRNGKey) -> chex.Array:
    """Sample a Drop7 disk value in [1, 7]."""
    return jax.random.randint(
        key,
        shape=(),
        minval=1,
        maxval=NUM_DISK_VALUES + 1,
        dtype=jnp.int32,
    )


def disk_to_one_hot(disk: chex.Array) -> chex.Array:
    """Convert a disk value in [1, 7] to a one-hot vector."""
    return jax.nn.one_hot(disk - 1, NUM_DISK_VALUES, dtype=jnp.int32)


def board_values(board: chex.Array) -> chex.Array:
    """Return the underlying disk values from a one-hot encoded board."""
    values = jnp.arange(1, NUM_DISK_VALUES + 1, dtype=jnp.int32)
    return jnp.sum(board * values, axis=-1)


def occupied_cells(board: chex.Array) -> chex.Array:
    """Return a boolean mask of non-empty board cells."""
    return jnp.any(board > 0, axis=-1)


def visible_board(board: chex.Array, blocks: chex.Array) -> chex.Array:
    """Build the observed board with hidden disks masked as -1."""
    values = board_values(board)
    occupied = values > EMPTY
    return jnp.where((blocks > 0) & occupied, HIDDEN_DISK, values)


def get_action_mask(board: chex.Array) -> chex.Array:
    """Return which columns have room for a new disk."""
    return ~occupied_cells(board)[BOARD_SIZE - 1]


def drop_disk(board: chex.Array, action: chex.Array, disk: chex.Array) -> chex.Array:
    """Drop a disk into a column.

    This assumes the action is valid. Rows are indexed from bottom to top, so
    the disk lands at the first empty row in the selected column.
    """
    col_occupied = occupied_cells(board)[:, action]
    row = jnp.sum(col_occupied.astype(jnp.int32))
    return board.at[row, action].set(disk_to_one_hot(disk))


def apply_gravity(
    board: chex.Array, blocks: chex.Array
) -> Tuple[chex.Array, chex.Array]:
    """Collapse each column toward row 0 while preserving disk order."""

    def collapse_column(
        board_col: chex.Array, blocks_col: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        occupied = jnp.any(board_col > 0, axis=-1)
        new_rows = jnp.cumsum(occupied.astype(jnp.int32)) - 1
        safe_rows = jnp.where(occupied, new_rows, 0)
        moving_board = board_col * occupied[:, None]
        moving_blocks = blocks_col * occupied.astype(blocks_col.dtype)
        collapsed_board = jnp.zeros_like(board_col).at[safe_rows].add(moving_board)
        collapsed_blocks = jnp.zeros_like(blocks_col).at[safe_rows].add(moving_blocks)
        return collapsed_board, collapsed_blocks

    return jax.vmap(collapse_column, in_axes=(1, 1), out_axes=(1, 1))(board, blocks)


def _consecutive_counts(occupied: chex.Array) -> chex.Array:
    """Count consecutive occupied cells ending at each position in a line."""

    def scan_fn(
        count: chex.Array, is_occupied: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        next_count = (count + 1) * is_occupied.astype(jnp.int32)
        return next_count, next_count

    _, counts = jax.lax.scan(scan_fn, jnp.array(0, jnp.int32), occupied)
    return counts


def _line_clears(values: chex.Array, blocks: chex.Array) -> chex.Array:
    occupied = values > EMPTY
    left_counts = _consecutive_counts(occupied)
    right_counts = jnp.flip(_consecutive_counts(jnp.flip(occupied)))
    run_lengths = left_counts + right_counts - 1
    return occupied & (blocks == 0) & (values == run_lengths)


def find_clears(board: chex.Array, blocks: chex.Array) -> chex.Array:
    """Find visible disks whose value equals their row or column run length."""
    values = board_values(board)
    row_clears = jax.vmap(_line_clears)(values, blocks)
    col_clears = jax.vmap(_line_clears)(values.T, blocks.T).T
    return row_clears | col_clears


def unblock(blocks: chex.Array, clears: chex.Array) -> chex.Array:
    """Crack hidden disks adjacent to cleared visible disks."""
    zeros_row = jnp.zeros((1, BOARD_SIZE), dtype=jnp.int32)
    zeros_col = jnp.zeros((BOARD_SIZE, 1), dtype=jnp.int32)
    clear_int = clears.astype(jnp.int32)
    adjacent_clears = (
        jnp.concatenate([clear_int[1:, :], zeros_row], axis=0)
        + jnp.concatenate([zeros_row, clear_int[:-1, :]], axis=0)
        + jnp.concatenate([clear_int[:, 1:], zeros_col], axis=1)
        + jnp.concatenate([zeros_col, clear_int[:, :-1]], axis=1)
    )
    return jnp.maximum(0, blocks - adjacent_clears)


def resolve_board(
    board: chex.Array, blocks: chex.Array
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Resolve gravity, clears, and unblocking until the board is stable."""
    board, blocks = apply_gravity(board, blocks)
    clears = find_clears(board, blocks)
    initial_score = jnp.array(0, jnp.float32)

    def should_continue(
        carry: Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> chex.Array:
        _, _, _, pending_clears = carry
        return jnp.any(pending_clears)

    def clear_once(
        carry: Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
        board, blocks, score, pending_clears = carry
        cleared = pending_clears[..., None]
        board = board * (~cleared)
        blocks = unblock(blocks, pending_clears)
        score = score + jnp.sum(pending_clears).astype(jnp.float32)
        board, blocks = apply_gravity(board, blocks)
        next_clears = find_clears(board, blocks)
        return board, blocks, score, next_clears

    board, blocks, score, _ = jax.lax.while_loop(
        should_continue,
        clear_once,
        (board, blocks, initial_score, clears),
    )
    return board, blocks, score
