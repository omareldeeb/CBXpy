from concurrent.futures import ThreadPoolExecutor, as_completed, wait, Future
from typing import Callable, List, Optional

import numpy as np

from .cbo import CBO


class DistributedCBO:
    def __init__(
        self,
        num_agent_batches: int,
        synchronization_interval: Optional[int] = 50,
        synchronization_method: str = 'mean',
        synchronization_criterion: str = 'interval',
        early_stopping_criterion: Callable[[CBO], bool] = None,
        verbose: bool = True,
        **kwargs
    ) -> None:
        if early_stopping_criterion is None:
            # Stop early if all particles are done
            early_stopping_criterion = lambda dynamics: all([dyn.terminate() for dyn in dynamics])

        self.dynamics = [CBO(batch_args=None, M=1, **kwargs) for _ in range(num_agent_batches)]
        self.synchronization_interval = synchronization_interval
        self.early_stopping_criterion = early_stopping_criterion
        self.verbose = verbose

        self._num_steps = 0
        self._num_synchronizations = 0

        self._sync_methods = {
            'mean': self._synchronize_mean,
            'running_mean': self._synchronize_running_mean,
            # TODO: weighted mean, ...
        }
        assert synchronization_method in self._sync_methods, f"Invalid synchronization method: {synchronization_method}"
        self._synchronize = self._sync_methods[synchronization_method]

        self._sync_criterions = {
            'interval': self._interval_synchronization,
        }
        assert synchronization_criterion in self._sync_criterions, f"Invalid synchronization criterion: {synchronization_criterion}"
        self._synchronization_criterion = self._sync_criterions[synchronization_criterion]


    def optimize(self, num_steps: int) -> None:
        all_futures = []
        current_futures = []
        with ThreadPoolExecutor(max_workers=len(self.dynamics)) as executor:
            for _ in range(num_steps):
                if self.early_stopping_criterion(self.dynamics):
                    if self.verbose:
                        print("DistCBO: Early stopping criterion met.")
                    break
                futures = [executor.submit(self._optimize_instance, dynamic) for dynamic in self.dynamics]
                current_futures = futures
                all_futures.extend(futures)
                self._num_steps += 1

                if self._synchronization_criterion():
                    if self.verbose:
                        print(f"DistCBO: Synchronizing at step {self._num_steps}")

                    self._synchronize(current_futures, all_futures)
                    self._num_synchronizations += 1
                    all_futures = []

        best_particle = self.best_particle()
        best_energy = self.best_energy()
        if self.verbose:
            print(f"DistCBO: Optimization finished after {self._num_steps} steps. Synchronized {self._num_synchronizations} times.")
            print(f"DistCBO: Best particle: {best_particle}, best energy: {best_energy}")
            print(f"DistCBO: Number of function evaluations: {[dyn.num_f_eval for dyn in self.dynamics]}")

        return best_particle 


    def best_particle(self) -> np.ndarray:
        best_particles = np.array([dynamic.best_particle for dynamic in self.dynamics])
        best_energy = np.array([dynamic.best_energy for dynamic in self.dynamics])

        return best_particles[np.argmin(best_energy)]
    

    def best_energy(self) -> float:
        best_energy = np.array([dynamic.best_energy for dynamic in self.dynamics])

        return np.min(best_energy)
    

    def _optimize_instance(self, dynamic: CBO) -> CBO:
        dynamic.step()

        return dynamic


    def _synchronize_mean(self, current_futures: List[Future], all_futures: List[Future]) -> None:
        # Waits for all of them to complete
        wait(all_futures)

        best_particles = np.array([dyn.best_particle for dyn in self.dynamics])
        consensus_point = np.mean(best_particles, axis=0)

        # Update all dynamics with the global consensus point
        for dynamic in self.dynamics:
            dynamic.consensus = consensus_point[None, :]
            dynamic.drift = dynamic.x - dynamic.consensus
            dynamic.x = dynamic.x - dynamic.correction(dynamic.lamda * dynamic.dt * dynamic.drift) + dynamic.sigma * dynamic.noise()


    def _synchronize_running_mean(self, current_futures: List[Future], all_futures: List[Future]) -> None:
        raise NotImplementedError("Running mean synchronization not implemented yet.")
    

    def _interval_synchronization(self, **kwargs) -> bool:
        assert self.synchronization_interval is not None and self.synchronization_interval > 0, "Invalid synchronization interval"

        return self._num_steps % self.synchronization_interval == 0
