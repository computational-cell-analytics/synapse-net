import importlib
import os
import unittest
from shutil import rmtree
from unittest import mock

import numpy as np
import torch


class TestOverwriteGuard(unittest.TestCase):
    """The guard that stops a training run from replacing the checkpoints of a previous one."""

    tmp_folder = "./tmp_training_safeguards"
    name = "a-previous-run"

    def setUp(self):
        self.checkpoint_folder = os.path.join(self.tmp_folder, "checkpoints", self.name)
        os.makedirs(self.checkpoint_folder, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _write_checkpoint(self, name="latest", iteration=10, with_scaler=True):
        save_dict = {"iteration": iteration, "epoch": 1, "best_epoch": 1, "best_metric": 0.5,
                     "current_metric": 0.5, "model_state": {}, "optimizer_state": {}, "train_time": 1.0}
        if with_scaler:
            save_dict["scaler_state"] = {}
        torch.save(save_dict, os.path.join(self.checkpoint_folder, f"{name}.pt"))

    def test_refuses_to_overwrite_a_previous_run(self):
        from synapse_net.training.supervised_training import _check_overwrite
        self._write_checkpoint()
        with self.assertRaisesRegex(ValueError, "already has checkpoints"):
            _check_overwrite(self.tmp_folder, self.name, overwrite=False, resume=False)

    def test_overwrite_and_resume_bypass_the_guard(self):
        from synapse_net.training.supervised_training import _check_overwrite
        self._write_checkpoint()
        _check_overwrite(self.tmp_folder, self.name, overwrite=True, resume=False)
        _check_overwrite(self.tmp_folder, self.name, overwrite=False, resume=True)

    def test_guard_allows_a_new_run(self):
        from synapse_net.training.supervised_training import _check_overwrite
        # No checkpoint yet, and no save_root at all.
        _check_overwrite(self.tmp_folder, self.name, overwrite=False, resume=False)
        _check_overwrite(None, self.name, overwrite=False, resume=False)


class TestResume(unittest.TestCase):
    """Resuming has to continue the previous run, not start a new one from its weights."""

    tmp_folder = "./tmp_training_resume"
    name = "a-previous-run"

    def setUp(self):
        self.checkpoint_folder = os.path.join(self.tmp_folder, "checkpoints", self.name)
        os.makedirs(self.checkpoint_folder, exist_ok=True)

    def tearDown(self):
        try:
            rmtree(self.tmp_folder)
        except OSError:
            pass

    def _write_checkpoint(self, name="latest", iteration=10, with_scaler=True):
        save_dict = {"iteration": iteration, "epoch": 1, "best_epoch": 1, "best_metric": 0.5,
                     "current_metric": 0.5, "model_state": {}, "optimizer_state": {}, "train_time": 1.0}
        if with_scaler:
            save_dict["scaler_state"] = {}
        torch.save(save_dict, os.path.join(self.checkpoint_folder, f"{name}.pt"))

    def test_resume_returns_a_stem_and_the_remaining_iterations(self):
        from synapse_net.training.supervised_training import _prepare_resume
        self._write_checkpoint(iteration=10)
        load_from_checkpoint, remaining = _prepare_resume(self.tmp_folder, self.name, 100, mixed_precision=True)

        # torch-em appends '.pt', so the stem must NOT carry the extension, and must resolve.
        self.assertFalse(load_from_checkpoint.endswith(".pt"))
        self.assertTrue(os.path.exists(f"{load_from_checkpoint}.pt"))
        self.assertIsInstance(load_from_checkpoint, str)
        # 'n_iterations' is the total, so only the missing ones are requested.
        self.assertEqual(remaining, 90)

    def test_resume_backs_up_the_previous_best(self):
        from synapse_net.training.supervised_training import _prepare_resume
        self._write_checkpoint()
        self._write_checkpoint(name="best")
        _prepare_resume(self.tmp_folder, self.name, 100, mixed_precision=True)
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_folder, "best-before-resume.pt")))

    def test_resume_without_a_previous_run(self):
        from synapse_net.training.supervised_training import _prepare_resume
        with self.assertRaisesRegex(ValueError, "does not exist"):
            _prepare_resume(self.tmp_folder, self.name, 100, mixed_precision=True)

    def test_resume_of_a_finished_run(self):
        from synapse_net.training.supervised_training import _prepare_resume
        self._write_checkpoint(iteration=100)
        with self.assertRaisesRegex(ValueError, "already at iteration"):
            _prepare_resume(self.tmp_folder, self.name, 100, mixed_precision=True)

    def test_resume_with_mismatched_mixed_precision(self):
        from synapse_net.training.supervised_training import _prepare_resume
        # torch-em reads 'scaler_state' unconditionally when the new trainer has a scaler.
        self._write_checkpoint(with_scaler=False)
        with self.assertRaisesRegex(ValueError, "without mixed precision"):
            _prepare_resume(self.tmp_folder, self.name, 100, mixed_precision=True)
        # ... but continuing it without mixed precision is fine.
        _, remaining = _prepare_resume(self.tmp_folder, self.name, 100, mixed_precision=False)
        self.assertEqual(remaining, 90)


class TestSeeding(unittest.TestCase):
    """'--seed' has to cover the training itself, not just the train / val split."""

    def test_set_seed_makes_weight_init_reproducible(self):
        from synapse_net.training.supervised_training import _set_seed, get_3d_model

        _set_seed(42)
        first = [p.detach().clone() for p in get_3d_model(out_channels=2, initial_features=4).parameters()]
        _set_seed(42)
        same = [p.detach().clone() for p in get_3d_model(out_channels=2, initial_features=4).parameters()]
        _set_seed(7)
        other = [p.detach().clone() for p in get_3d_model(out_channels=2, initial_features=4).parameters()]

        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, same)))
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(first, other)))

    def test_set_seed_covers_random_and_numpy(self):
        import random
        from synapse_net.training.supervised_training import _set_seed

        _set_seed(42)
        values = (random.random(), np.random.rand())
        _set_seed(42)
        self.assertEqual(values, (random.random(), np.random.rand()))

    def test_deterministic_disables_cudnn_benchmark(self):
        from synapse_net.training.supervised_training import _set_seed

        benchmark = torch.backends.cudnn.benchmark
        deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            torch.backends.cudnn.benchmark = True
            _set_seed(42, deterministic=False)
            self.assertTrue(torch.backends.cudnn.benchmark)  # untouched by default
            self.assertFalse(torch.are_deterministic_algorithms_enabled())

            _set_seed(42, deterministic=True)
            self.assertFalse(torch.backends.cudnn.benchmark)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
        finally:
            # This is global state, so it must not leak into the other tests.
            torch.backends.cudnn.benchmark = benchmark
            torch.use_deterministic_algorithms(deterministic, warn_only=True)


class TestTrainingArgumentsArePassedOn(unittest.TestCase):
    """The new arguments have to reach 'supervised_training' from every task module."""

    def test_mitochondria_passes_the_new_arguments(self):
        from synapse_net.training.mitochondria import mitochondria_training

        with mock.patch("synapse_net.training.mitochondria.supervised_training") as training:
            mitochondria_training(name="m", train_paths=["a.h5"], val_paths=["b.h5"],
                                  resume=True, overwrite=True, seed=7, deterministic=True)
        kwargs = training.call_args.kwargs
        self.assertIs(kwargs["resume"], True)
        self.assertIs(kwargs["overwrite"], True)
        self.assertEqual(kwargs["seed"], 7)
        self.assertIs(kwargs["deterministic"], True)

    def test_cristae_passes_the_new_arguments(self):
        from synapse_net.training.cristae import cristae_training

        with mock.patch("synapse_net.training.cristae.supervised_training") as training:
            cristae_training(name="c", train_paths=["a.h5"], val_paths=["b.h5"],
                             resume=True, overwrite=True, seed=7, deterministic=True)
        kwargs = training.call_args.kwargs
        self.assertIs(kwargs["resume"], True)
        self.assertIs(kwargs["overwrite"], True)
        self.assertEqual(kwargs["seed"], 7)
        self.assertIs(kwargs["deterministic"], True)

    def test_vol_em_mitochondria_passes_the_new_arguments(self):
        from synapse_net.training.mitochondria_vol_em import vol_em_mitochondria_training

        with mock.patch("synapse_net.training.mitochondria_vol_em.supervised_training") as training:
            vol_em_mitochondria_training(name="v", train_paths=["a.h5"], val_paths=["b.h5"],
                                         resume=True, overwrite=True, seed=7, deterministic=True)
        kwargs = training.call_args.kwargs
        self.assertIs(kwargs["resume"], True)
        self.assertIs(kwargs["overwrite"], True)
        self.assertEqual(kwargs["seed"], 7)
        self.assertIs(kwargs["deterministic"], True)

    def test_defaults_are_unchanged(self):
        # The published recipes must not become seeded or overwriting by default.
        from synapse_net.training.mitochondria import mitochondria_training

        with mock.patch("synapse_net.training.mitochondria.supervised_training") as training:
            mitochondria_training(name="m", train_paths=["a.h5"], val_paths=["b.h5"])
        kwargs = training.call_args.kwargs
        self.assertIs(kwargs["resume"], False)
        self.assertIs(kwargs["overwrite"], False)
        self.assertIsNone(kwargs["seed"])


class TestScaleFactorsReachTheModel(unittest.TestCase):
    """'supervised_training' used to build the model without the scale factors it was given."""

    def _build_model(self, **kwargs):
        # 'synapse_net.training' re-exports the function 'supervised_training' under the name of its
        # own module, which shadows it, so the module has to be looked up explicitly.
        st = importlib.import_module("synapse_net.training.supervised_training")

        with mock.patch.object(st, "get_supervised_loader"), \
             mock.patch.object(st, "get_3d_model") as get_model, \
             mock.patch.object(st.torch_em, "default_segmentation_trainer"):
            st.supervised_training(
                name="m", train_paths=("a.h5",), val_paths=("b.h5",), label_key="labels",
                patch_shape=(32, 512, 512), **kwargs,
            )
        return get_model.call_args.kwargs

    def test_supervised_training_passes_scale_factors_to_the_model(self):
        scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]
        kwargs = self._build_model(scale_factors=scale_factors, norm=None)
        self.assertEqual(kwargs["scale_factors"], scale_factors)
        self.assertIsNone(kwargs["norm"])

    def test_supervised_training_keeps_the_default_scale_factors(self):
        # Not passing them must not change the model that the other recipes build.
        self.assertNotIn("scale_factors", self._build_model())

    def test_two_anisotropic_levels_build_a_usable_model(self):
        from synapse_net.training.supervised_training import get_3d_model

        model = get_3d_model(out_channels=2, initial_features=4, norm=None,
                             scale_factors=[[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]])
        self.assertEqual(sum(1 for module in model.modules() if "Norm" in type(module).__name__), 0)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.rand(1, 1, 8, 64, 64)).shape), (1, 2, 8, 64, 64))


if __name__ == "__main__":
    unittest.main()
