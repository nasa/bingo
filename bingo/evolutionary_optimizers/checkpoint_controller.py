"""
This module is a helper class for controlling times of checkpoints in a 
time-controlled evolutionary optimization.
"""

from datetime import datetime
import numpy as np


class CheckpointController:
    """Controller for times between checkpoints

    Parameters
    ----------
    start_time : datetime
        starting time of the optimization
    max_time : float
        maximum allowed time of optimization in seconds
    convergence_check_frequency : int
        number of evolutionary generations between standard checkpoints
    safety_factor : float, optional
        favctor by which to reuce maximum time to help ensure finishing on time,
        by default 0.97
    """

    def __init__(
        self, start_time, max_time, convergence_check_frequency, safety_factor=0.98
    ):
        self._start_time = start_time
        self._max_time = max_time
        self._check_freq = convergence_check_frequency
        self._last_check_time = start_time
        self._safety_factor = safety_factor

        self._weights = np.array([4, 2, 1])
        self._max_record = len(self._weights)
        self._gen_speeds = []

    def record_check(self, num_generations):
        """Records that a checkpoint has just been reached

        Parameters
        ----------
        num_generations : int
            the number of generations that were evolved since the last checkpoint
        """
        now = datetime.now()
        self._gen_speeds.append(
            (now - self._last_check_time).total_seconds() / num_generations
        )
        if len(self._gen_speeds) > self._max_record:
            self._gen_speeds = self._gen_speeds[-self._max_record :]
        self._last_check_time = now

    def estimate_remaining_checkpoints(self):
        """Estimate number of checkpoints that can be performed before time limit

        Returns
        -------
        float
            estimated number of checkpoints that can be performed, can be
            fractional
        """
        if self._max_time is None or len(self._gen_speeds) == 0:
            return None
        gen_speed = np.sum(
            self._gen_speeds * self._weights[: len(self._gen_speeds)]
        ) / np.sum(self._weights[: len(self._gen_speeds)])
        remaining_time = (
            self._max_time * self._safety_factor
            - (datetime.now() - self._start_time).total_seconds()
        )
        return remaining_time / gen_speed / self._check_freq

    def get_gens_to_evolve(self):
        """Calculate the number of generations for next checkpoint

        Returns the default value unless a fractional-sized checkpoint is
        needed to stay under the time limit

        Returns
        -------
        int
            number of generations to evolve before next checkpoint
        """
        if self._max_time is None:
            return self._check_freq

        est_remaining_checks = self.estimate_remaining_checkpoints()

        if est_remaining_checks is None or est_remaining_checks > 1:
            return self._check_freq

        return max(1, int(est_remaining_checks * self._check_freq))
