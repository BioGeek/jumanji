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

from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import matplotlib.animation
import matplotlib.pyplot as plt

import jumanji.environments
from jumanji.environments.logic.drop_7 import utils
from jumanji.environments.logic.drop_7.types import State
from jumanji.viewer import Viewer


class Drop7Viewer(Viewer):
    COLORS = {
        0: "#111827",
        1: "#ef4444",
        2: "#3b82f6",
        3: "#22c55e",
        4: "#f97316",
        5: "#a855f7",
        6: "#0f766e",
        7: "#eab308",
        "hidden": "#9ca3af",
        "hidden_text": "#374151",
        "text": "#f9fafb",
        "edge": "#e5e7eb",
        "bg": "#f8fafc",
    }

    def __init__(
        self,
        name: str = "Drop7",
        board_size: int = utils.BOARD_SIZE,
    ) -> None:
        """Viewer for the Drop 7 environment.

        Args:
            name: the window name to be used when initialising the window.
            board_size: size of the board.
        """
        self._name = name
        self._board_size = board_size

        # The animation must be stored in a variable that lives as long as the
        # animation should run. Otherwise, the animation will get garbage-collected.
        self._animation: Optional[matplotlib.animation.Animation] = None

    def render(self, state: State) -> None:
        """Renders the current state of the game board.

        Args:
            state: is the current game state to be rendered.
        """
        self._clear_display()
        fig, ax = self.get_fig_ax()
        fig.suptitle(
            f"Drop7    Score: {int(state.score)}    Disk: {int(state.current_disk)}",
            size=20,
        )
        self.draw_board(ax, state)
        self._display_human(fig)

    def animate(
        self,
        states: Sequence[State],
        interval: int = 200,
        save_path: Optional[str] = None,
    ) -> matplotlib.animation.FuncAnimation:
        """Creates an animated gif of the Drop7 game board based on the sequence of game states.

        Args:
            states: is a list of `State` objects representing the sequence of game states.
            interval: the delay between frames in milliseconds, default to 200.
            save_path: the path where the animation file should be saved. If it is None, the plot
                will not be saved.

        Returns:
            Animation object that can be saved as a GIF, MP4, or rendered with HTML.
        """
        fig, ax = self.get_fig_ax()
        fig.suptitle("Drop7    Score: 0", size=20)
        plt.tight_layout()

        def make_frame(state_index: int) -> None:
            state = states[state_index]
            self.draw_board(ax, state)
            fig.suptitle(
                f"Drop7    Score: {int(state.score)}    Disk: {int(state.current_disk)}",
                size=20,
            )

        self._animation = matplotlib.animation.FuncAnimation(
            fig,
            make_frame,
            frames=len(states),
            interval=interval,
        )

        # Save the animation as a gif.
        if save_path:
            self._animation.save(save_path)

        return self._animation

    def get_fig_ax(self) -> Tuple[plt.Figure, plt.Axes]:
        """This function returns a `Matplotlib` figure and axes object for displaying
        the Drop 7 game board.

        Returns:
            A tuple containing the figure and axes objects.
        """
        exists = plt.fignum_exists(self._name)
        if exists:
            fig = plt.figure(self._name)
            ax = fig.get_axes()[0]
        else:
            fig = plt.figure(
                self._name,
                figsize=(6.0, 6.0),
                facecolor=self.COLORS["bg"],
            )
            plt.tight_layout()
            if not plt.isinteractive():
                fig.show()
            ax = fig.add_subplot()
        return fig, ax

    def render_tile(self, tile_value: int, ax: plt.Axes, row: int, col: int) -> None:
        """Render a single Drop7 tile."""
        if tile_value == utils.HIDDEN_DISK:
            color = self.COLORS["hidden"]
            text = "?"
            text_color = self.COLORS["hidden_text"]
        else:
            color = self.COLORS[int(tile_value)]
            text = "" if tile_value == utils.EMPTY else str(tile_value)
            text_color = self.COLORS["text"]

        rect = plt.Rectangle([col - 0.5, row - 0.5], 1, 1, color=color)
        ax.add_patch(rect)
        if text:
            ax.text(
                col,
                row,
                text,
                color=text_color,
                ha="center",
                va="center",
                size=26,
                weight="bold",
            )

    def draw_board(self, ax: plt.Axes, state: State) -> None:
        """Draw the game board with the current state."""
        ax.clear()
        board = utils.visible_board(state.board, state.blocks)

        for row in range(self._board_size):
            draw_row = self._board_size - row - 1
            for col in range(self._board_size):
                self.render_tile(
                    tile_value=int(board[row, col]),
                    ax=ax,
                    row=draw_row,
                    col=col,
                )

        ax.imshow(jnp.flip(board, axis=0), alpha=0)
        ax.set_xticks(jnp.arange(-0.5, self._board_size - 1, 1))
        ax.set_yticks(jnp.arange(-0.5, self._board_size - 1, 1))
        ax.tick_params(
            top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
        )
        ax.grid(color=self.COLORS["edge"], linestyle="-", linewidth=3)

    def close(self) -> None:
        plt.close(self._name)

    def _display_human(self, fig: plt.Figure) -> None:
        if plt.isinteractive():
            fig.canvas.draw()
            if jumanji.environments.is_colab():
                plt.show(self._name)
        else:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    def _clear_display(self) -> None:
        if jumanji.environments.is_colab():
            import IPython.display

            IPython.display.clear_output(True)
