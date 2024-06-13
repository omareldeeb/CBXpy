from concurrent.futures import ThreadPoolExecutor, as_completed, wait, Future
from typing import Callable, List

import numpy as np

from .cbo import CBO


# Constants
SYNCHRONIZATION_INTERVAL: int = 50
VERBOSE: bool = True
SYNCHRONIZATION_METHOD: str = 'mean'
# Stop early if all particles are done
EARLY_STOPPING_CRITERION: Callable[[CBO], bool] = lambda dynamics: all([dyn.terminate() for dyn in dynamics])


class DistributedCBO:
    def __init__(
        self,
        num_agent_batches: int,
        synchronization_interval: int = SYNCHRONIZATION_INTERVAL,
        synchronization_method: str = SYNCHRONIZATION_METHOD,
        early_stopping_criterion: Callable[[CBO], bool] = EARLY_STOPPING_CRITERION,
        verbose: bool = VERBOSE,
        **kwargs
    ) -> None:
        self.dynamics = [CBO(batch_args=None, M=1, **kwargs) for _ in range(num_agent_batches)]
        self.synchronization_interval = synchronization_interval
        self.early_stopping_criterion = early_stopping_criterion
        self.verbose = verbose

        self._num_steps = 0
        self._num_synchronizations = 0

        self._sync_methods = {
            'mean': self._synchronize_mean,
            # TODO: weighted mean, ...
        }

        assert synchronization_method in self._sync_methods, f"Invalid synchronization method: {synchronization_method}"
        self.synchronization_method = self._sync_methods[synchronization_method]


    def _synchronize_mean(self) -> None:
        best_particles = np.array([dynamic.best_particle for dynamic in self.dynamics])

        consensus_point = np.mean(best_particles, axis=0)

        for dynamic in self.dynamics:
            dynamic.consensus = consensus_point[None, :]
            dynamic.drift = dynamic.x - dynamic.consensus
            dynamic.x = dynamic.x - dynamic.correction(dynamic.lamda * dynamic.dt * dynamic.drift) + dynamic.sigma * dynamic.noise()

    
    def _synchronize(self) -> None:
        self.synchronization_method()

    
    def _optimize_instance(self, dynamic: CBO) -> CBO:
        dynamic.step()

        return dynamic


    def optimize(self, num_steps: int) -> None:
        with ThreadPoolExecutor(max_workers=len(self.dynamics)) as executor:
            for _ in range(num_steps):
                if self.early_stopping_criterion(self.dynamics):
                    if self.verbose:
                        print("DistCBO: Early stopping criterion met.")
                    break
                futures = [executor.submit(self._optimize_instance, dynamic) for dynamic in self.dynamics]
                self._num_steps += 1

                if self._num_steps % self.synchronization_interval == 0:
                    if self.verbose:
                        print(f"DistCBO: Synchronizing at step {self._num_steps}")
                    
                    # Waits for all of them to complete
                    completed_futures = list(as_completed(futures))
                    
                    for future in completed_futures:
                        dyn = future.result()   # Nothing to do with result for now

                    self._synchronize()
                    self._num_synchronizations += 1

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


            