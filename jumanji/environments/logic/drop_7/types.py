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

from typing import TYPE_CHECKING, NamedTuple

import chex
from typing_extensions import TypeAlias

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

Board: TypeAlias = chex.Array


@dataclass
class State:
    """State of the Drop7 game.

    board: one-hot encoded disk values of shape (7, 7, 7). The last axis stores
        values 1 through 7. Hidden gray disks keep their underlying value here.
    blocks: gray cover counts of shape (7, 7). A value of 0 means the disk is visible.
    current_disk: value of the disk the agent must drop next, in [1, 7].
    action_mask: indicates which columns can accept the current disk.
    score: cumulative number of visible disks cleared.
    step_count: number of environment steps elapsed since reset.
    key: random key used for sampling future disks.
    """

    board: Board  # (7, 7, 7)
    blocks: Board  # (7, 7)
    current_disk: chex.Numeric  # ()
    action_mask: chex.Array  # (7,)
    score: chex.Numeric  # ()
    step_count: chex.Numeric  # ()
    key: chex.PRNGKey  # (2,)


class Observation(NamedTuple):
    """Observation of the Drop7 game.

    board: visible board of shape (7, 7). Empty cells are 0, visible disks are
        1 through 7, and hidden gray disks are -1.
    current_disk: value of the disk the agent must drop next, in [1, 7].
    action_mask: indicates which columns can accept the current disk.
    step_count: number of environment steps elapsed since reset.
    """

    board: Board  # (7, 7)
    current_disk: chex.Numeric  # ()
    action_mask: chex.Array  # (7,)
    step_count: chex.Numeric  # ()
