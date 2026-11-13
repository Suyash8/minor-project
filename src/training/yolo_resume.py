"""Ultralytics trainer extension with preserved resumable last checkpoint."""
from pathlib import Path
import json
import os
import shutil
import math


def extension_trainer(base, target_epochs):
    """Adapt before scheduler construction; Ultralytics otherwise ignores epochs."""
    class ExtendedOBBTrainer(base):
        def check_resume(self, overrides):
            super().check_resume(overrides)
            if self.resume:
                self.args.epochs = int(target_epochs)
                previous_cos = self.args.cos_lr
                self.args.cos_lr = True
                print(f"[resume] horizon adaptation uses cosine at new total={target_epochs}; LR may rise from the old floor (previous cos_lr={previous_cos})")
                # A resumed global epoch must not replay batch warmup.
                self.args.warmup_epochs = 0.

        def resume_training(self, ckpt):
            super().resume_training(ckpt)
            if ckpt is not None and self.resume:
                completed = ckpt.get('epoch', -1)+1
                if ckpt.get('optimizer') is None:
                    print("[resume][WARNING] fresh optimizer warmup over first resumed epoch; old moments unavailable")
                    self._fresh_resume_batch = 0
                    def warm_fresh(trainer):
                        trainer._fresh_resume_batch += 1
                        fraction = min(1., trainer._fresh_resume_batch/max(1,len(trainer.train_loader)))
                        for group in trainer.optimizer.param_groups:
                            group['lr'] = group.get('initial_lr', trainer.args.lr0)*trainer.lf(trainer.epoch)*(.01+.99*fraction)
                    self.add_callback('on_train_batch_start', warm_fresh)
                for group in self.optimizer.param_groups:
                    base_lr = group.get('initial_lr', self.args.lr0)
                    group['lr'] = base_lr*self.lf(completed)
                self.scheduler._last_lr = [g['lr'] for g in self.optimizer.param_groups]

        def final_eval(self):
            # Native final_eval strips optimizer and sets epoch=-1 on last.pt.
            # Preserve full state for future horizon extension and preemption.
            full = self.last.with_name('last.resumable.pt')
            if self.last.exists():
                shutil.copy2(self.last, full)
            try:
                return super().final_eval()
            finally:
                if full.exists():
                    os.replace(full, self.last)
    return ExtendedOBBTrainer


def checkpoint_progress(ckpt, run_dir):
    """Completed count, including old stripped Ultralytics files (epoch=-1)."""
    epoch = int(ckpt.get('epoch', -1))
    if epoch >= 0:
        return epoch+1
    results = Path(run_dir)/'results.csv'
    if results.exists():
        import csv
        rows = list(csv.DictReader(results.open()))
        if rows:
            return max(int(float({k.strip():v for k,v in row.items()}.get('epoch', 0))) for row in rows)
    train_args = ckpt.get('train_args') or {}
    return int(train_args.get('epochs', 0))
