"""Closed-form warmup/cosine schedule with explicit extension semantics."""
import math


class ContinuousCosineLR:
    """Epoch counters denote COMPLETED epochs, with no recursive LR drift.

    Extension remaps completed/target to the requested horizon, so an old floor
    can rise to the new cosine value. Optimizer moments are never reset. This
    is horizon adaptation, not a claim of bit-identical LR to the old horizon.
    """
    def __init__(self, optimizer, T_max, eta_min=1e-6, warmup_epochs=0):
        self.optimizer = optimizer
        self.T_max = max(1, int(T_max))
        self.eta_min = float(eta_min)
        self.warmup_epochs = int(warmup_epochs)
        self.base_lrs = [g.get('initial_lr', g['lr']) for g in optimizer.param_groups]
        self.last_epoch = 0
        self.restart_epoch = None
        self._apply()

    def _apply(self):
        lrs = []
        for base in self.base_lrs:
            if self.restart_epoch is not None:
                phase = (self.last_epoch-self.restart_epoch)/max(1, self.T_max-self.restart_epoch)
            else:
                phase = self.last_epoch/self.T_max
            lr = self.eta_min + .5*(base-self.eta_min)*(1+math.cos(math.pi*min(1,max(0,phase))))
            if self.restart_epoch is None and self.warmup_epochs and self.last_epoch < self.warmup_epochs:
                lr *= .01+.99*self.last_epoch/self.warmup_epochs
            lrs.append(lr)
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group['lr'] = lr
        self._last_lr = lrs

    def step(self):
        self.last_epoch += 1
        self._apply()

    def get_last_lr(self):
        return list(self._last_lr)

    def state_dict(self):
        return dict(T_max=self.T_max, eta_min=self.eta_min, base_lrs=self.base_lrs,
                    last_epoch=self.last_epoch, warmup_epochs=self.warmup_epochs,
                    restart_epoch=self.restart_epoch)

    def load_state_dict(self, state):
        # Read legacy CosineAnnealingLR and SequentialLR checkpoints. Target
        # horizon and warmup are from THIS invocation, never from old state.
        old = state.get('_schedulers', [state])[-1]
        self.base_lrs = list(old.get('base_lrs', self.base_lrs))
        self.last_epoch = int(state.get('last_epoch', 0))
        self.restart_epoch = state.get('restart_epoch')

    def resume_at(self, completed, warm_restart=False):
        self.last_epoch = int(completed)
        if warm_restart:
            self.restart_epoch = int(completed)
        self._apply()
