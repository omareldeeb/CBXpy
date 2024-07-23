from concurrent.futures import ThreadPoolExecutor, as_completed, wait, Future
from typing import Callable, List, Optional, Dict
import threading

import numpy as np

from cbx.scheduler import scheduler

from .cbo import CBO

USE_ASYNC_COMMUNICAION = True

class DistributedCBO:
    def __init__(
        self,
        num_agent_batches: int,
        synchronization_interval: Optional[int] = 50,
        synchronization_method: str = 'mean',
        synchronization_alpha: float = 1.0,
        synchronization_criterion: str = 'interval',
        early_stopping_criterion: Callable[[CBO], bool] = None,
        verbose: bool = True,
        **kwargs
    ) -> None:
        if early_stopping_criterion is None:
            # Stop early if all particles are done
            early_stopping_criterion = lambda dynamics: all([dyn.terminate() for dyn in dynamics])

        self.dynamics = [CBO(batch_args=None, M=1, verbosity=0, **kwargs) for _ in range(num_agent_batches)]
        self.synchronization_interval = synchronization_interval
        self.synchronization_alpha = synchronization_alpha
        self.early_stopping_criterion = early_stopping_criterion
        self.verbose = verbose

        self._num_steps = 0
        self._num_synchronizations = 0
        self._num_communications = 0

        self._sync_methods = {
            'mean': self._synchronize_mean,
            'running_mean': self._synchronize_running_mean,
            'weighted_mean': self._synchronize_weighted_mean,
        }
        assert synchronization_method in self._sync_methods, f"Invalid synchronization method: {synchronization_method}"
        self._synchronize = self._sync_methods[synchronization_method]

        self._sync_criterions = {
            'interval': self._interval_synchronization,
        }
        assert synchronization_criterion in self._sync_criterions, f"Invalid synchronization criterion: {synchronization_criterion}"
        self._synchronization_criterion = self._sync_criterions[synchronization_criterion]

        self.global_mutex = threading.Lock()
        self.energies: Dict[CBO, float] = {dynamic: None for dynamic in self.dynamics}
        self.consensus_points: Dict[CBO, np.ndarray] = {dynamic: None for dynamic in self.dynamics}


    def optimize(self, num_steps: int, sched: scheduler = None) -> None:
        all_futures = []
        current_futures = []
        with ThreadPoolExecutor(max_workers=len(self.dynamics)) as executor:
            for _ in range(num_steps):
                if self.early_stopping_criterion(self.dynamics):
                    break
                current_futures = [executor.submit(self._optimize_instance, dynamic, sched) for dynamic in self.dynamics]
                all_futures.extend(current_futures)
                self._num_steps += 1

                if not USE_ASYNC_COMMUNICAION and self._synchronization_criterion():
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
    

    def _optimize_instance(self, dynamic: CBO, sched: scheduler = None) -> CBO:
        if sched is None:
            sched = scheduler([])
        elif sched == 'default':
            sched = dynamic.default_sched()

        dynamic.step()
        sched.update(dynamic)

        if USE_ASYNC_COMMUNICAION and dynamic.it % self.synchronization_interval == 0:
            consensus, energy = dynamic.consensus, dynamic.energy

            all_consensus_points = None
            all_energies = None
            with self.global_mutex:
                self._num_communications += 1
                # Push the current state to the consensus point
                self.consensus_points[dynamic] = consensus
                self.energies[dynamic] = energy

                valid_dynamics = [dyn for dyn in self.dynamics if self.energies[dyn] is not None]
                d = self.dynamics[0].x.shape[-1]

                # Pull the consensus point to the current state
                all_consensus_points = np.array([self.consensus_points[dynamic] for dynamic in valid_dynamics]).reshape(-1, d)
                all_energies = np.array([self.energies[dynamic] for dynamic in valid_dynamics]).reshape(-1)

            weights = np.exp(-self.synchronization_alpha * all_energies)
            consensus_point = np.average(all_consensus_points, weights=weights, axis=0)

            # Update current particle's state
            dynamic.consensus = consensus_point[None, :]
            dynamic.drift = dynamic.x - dynamic.consensus
            dynamic.x = dynamic.x - dynamic.correction(dynamic.lamda * dynamic.dt * dynamic.drift) + dynamic.sigma * dynamic.noise()



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
    

    def _synchronize_weighted_mean(self, current_futures: List[Future], all_futures: List[Future]) -> None:
        # Wait for all futures to complete
        wait(all_futures)

        d = self.dynamics[0].x.shape[-1]
        all_particles = np.array([dyn.x for dyn in self.dynamics]).reshape(-1, d)
        all_energies = np.array([dyn.energy for dyn in self.dynamics]).reshape(-1)

        weights = np.exp(-self.synchronization_alpha * all_energies)
        if np.sum(weights) == 0:
            weights = np.ones_like(weights) / len(weights)

        consensus_point = np.average(all_particles, weights=weights, axis=0)

        # Update all dynamics with the global consensus point
        for dynamic in self.dynamics:
            dynamic.consensus = consensus_point[None, :]
            dynamic.drift = dynamic.x - dynamic.consensus
            dynamic.x = dynamic.x - dynamic.correction(dynamic.lamda * dynamic.dt * dynamic.drift) + dynamic.sigma * dynamic.noise()

    

    def _interval_synchronization(self, **kwargs) -> bool:
        assert self.synchronization_interval is not None and self.synchronization_interval > 0, "Invalid synchronization interval"

        return self._num_steps % self.synchronization_interval == 0
