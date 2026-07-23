import os
import tempfile
import unittest
from unittest import mock

import numpy as np


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("XDG_CONFIG_HOME", os.path.join(tempfile.gettempdir(), "synapse-net-test-config"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "synapse-net-test-cache"))

from qtpy.QtWidgets import QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox

from synapse_net.inference import get_segmentation_function as public_get_segmentation_function
from synapse_net.inference import active_zone as active_zone_module
from synapse_net.inference import compartments as compartments_module
from synapse_net.inference import cristae as cristae_module
from synapse_net.inference import inference as inference_module
from synapse_net.inference.active_zone import segment_active_zone
from synapse_net.inference.compartments import segment_compartments
from synapse_net.inference.cristae import segment_cristae
from synapse_net.inference.inference import (
    _segment_ribbon_AZ,
    get_segmentation_function,
)
from synapse_net.inference.mitochondria import segment_mitochondria
from synapse_net.inference.postprocessing.ribbon import segment_ribbon
from synapse_net.inference.vesicles import VESICLE_SEGMENTATION_MODES, segment_vesicles
from synapse_net.tools import segmentation_widget


class _Event:
    def connect(self, callback):
        pass


class _LayerEvents:
    def __init__(self):
        self.inserted = _Event()
        self.removed = _Event()


class _Layers(list):
    def __init__(self):
        super().__init__()
        self.events = _LayerEvents()


class _Viewer:
    def __init__(self):
        self.layers = _Layers()
        self.added_labels = []

    def add_labels(self, data, **kwargs):
        self.added_labels.append((data, kwargs))


class TestSegmentationFunctionResolver(unittest.TestCase):
    def test_model_families(self):
        expected_functions = {
            (
                "vesicles_2d",
                "vesicles_3d",
                "vesicles_cryo",
                "vesicles_2d_maus",
                "vesicles_3d_endbulb",
                "vesicles_3d_innerear",
            ): segment_vesicles,
            ("mitochondria", "mitochondria2"): segment_mitochondria,
            ("active_zone",): segment_active_zone,
            ("compartments",): segment_compartments,
            ("ribbon",): _segment_ribbon_AZ,
            ("cristae", "cristae2", "cristae3", "cristae4"): segment_cristae,
        }
        for model_types, expected_function in expected_functions.items():
            for model_type in model_types:
                with self.subTest(model_type=model_type):
                    self.assertIs(get_segmentation_function(model_type), expected_function)
                    self.assertIs(public_get_segmentation_function(model_type), expected_function)

    def test_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "Unknown model type"):
            get_segmentation_function("unknown")


class TestSegmentationWidget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.viewer = _Viewer()
        self.current_viewer_patcher = mock.patch.object(
            segmentation_widget.napari, "current_viewer", return_value=self.viewer
        )
        self.current_viewer_patcher.start()
        self.widget = segmentation_widget.SegmentationWidget()

    def tearDown(self):
        self.widget.close()
        self.widget.deleteLater()
        self.app.processEvents()
        self.current_viewer_patcher.stop()

    def _select_model(self, model_type):
        index = self.widget.model_selector.findText(model_type)
        self.assertNotEqual(index, -1)
        self.widget.model_selector.setCurrentIndex(index)
        self.app.processEvents()

    def test_dynamic_postprocessing_parameters(self):
        test_cases = (
            ("vesicles_2d", {
                "min_size": (QSpinBox, 500, 1),
                "mode": (QComboBox, "distance-watershed", None),
                "threshold": (QDoubleSpinBox, 0.5, 0.01),
            }),
            ("mitochondria", {
                "min_size": (QSpinBox, 50_000, 1),
                "seed_distance": (QSpinBox, 6, 1),
            }),
            ("active_zone", {
                "min_size": (QSpinBox, 500, 1),
                "foreground_threshold": (QDoubleSpinBox, 0.5, 0.01),
            }),
            ("compartments", {
                "boundary_threshold": (QDoubleSpinBox, 0.4, 0.01),
                "n_slices_exclude": (QSpinBox, 0, 1),
                "min_z_extent": (QSpinBox, 10, 1),
            }),
            ("ribbon", {
                "threshold": (QDoubleSpinBox, 0.5, 0.01),
                "n_slices_exclude": (QSpinBox, 20, 1),
                "min_membrane_size": (QSpinBox, 50_000, 1),
                "n_ribbons": (QSpinBox, 1, 1),
            }),
            ("cristae", {
                "min_size": (QSpinBox, 2_000, 1),
                "foreground_threshold": (QDoubleSpinBox, 0.5, 0.01),
                "erosion_distance_nm": (QDoubleSpinBox, 0.0, 0.1),
            }),
        )
        for model_type, expected_parameters in test_cases:
            with self.subTest(model_type=model_type):
                self._select_model(model_type)
                self.assertEqual(set(self.widget.postprocessing_parameter_widgets), set(expected_parameters))
                for name, (widget_type, default, step) in expected_parameters.items():
                    parameter = self.widget.postprocessing_parameter_widgets[name]
                    self.assertIsInstance(parameter, widget_type)
                    if isinstance(parameter, QCheckBox):
                        self.assertEqual(parameter.isChecked(), default)
                    elif isinstance(parameter, QComboBox):
                        self.assertEqual(parameter.currentText(), default)
                    else:
                        self.assertAlmostEqual(parameter.value(), default)
                        self.assertAlmostEqual(parameter.singleStep(), step)

                if model_type == "vesicles_2d":
                    mode = self.widget.postprocessing_parameter_widgets["mode"]
                    options = tuple(mode.itemText(index) for index in range(mode.count()))
                    self.assertEqual(options, VESICLE_SEGMENTATION_MODES)

    def test_selection_resets_values_and_placeholder_clears_controls(self):
        self._select_model("vesicles_2d")
        self.widget.postprocessing_parameter_widgets["min_size"].setValue(515)

        self._select_model("compartments")
        self._select_model("vesicles_2d")
        self.assertEqual(self.widget.postprocessing_parameter_widgets["min_size"].value(), 500)

        self._select_model("- choose -")
        self.assertEqual(self.widget.postprocessing_parameter_widgets, {})
        self.assertEqual(self.widget.postprocessing_settings_layout.count(), 0)

    def test_choice_postprocessing_parameter_is_collected(self):
        self._select_model("vesicles_2d")
        self.widget.postprocessing_parameter_widgets["mode"].setCurrentText("label")
        self.widget.postprocessing_parameter_widgets["threshold"].setValue(0.6)
        kwargs = self.widget._get_postprocessing_kwargs()
        self.assertEqual(kwargs["mode"], "label")
        self.assertAlmostEqual(kwargs["threshold"], 0.6)

    def test_vesicle_postprocessing_parameters_are_forwarded(self):
        self._select_model("vesicles_2d")
        self.widget.postprocessing_parameter_widgets["min_size"].setValue(25)
        self.widget.postprocessing_parameter_widgets["mode"].setCurrentText("label")
        self.widget.postprocessing_parameter_widgets["threshold"].setValue(0.6)
        image = np.zeros((8, 8), dtype="float32")
        model = object()

        def get_layer_data(selector_name, return_metadata=False):
            return {} if return_metadata else image

        with (
            mock.patch.object(self.widget, "_get_layer_selector_data", side_effect=get_layer_data),
            mock.patch.object(self.widget, "_handle_resolution", return_value=None),
            mock.patch.object(segmentation_widget, "get_device", return_value="cpu"),
            mock.patch.object(segmentation_widget, "get_model", return_value=model),
            mock.patch.object(segmentation_widget, "_get_current_tiling", return_value={}),
            mock.patch.object(segmentation_widget, "run_segmentation", return_value=np.zeros_like(image)) as run,
            mock.patch.object(segmentation_widget, "show_info"),
        ):
            self.widget.on_predict()

        self.assertEqual(run.call_args.kwargs["min_size"], 25)
        self.assertEqual(run.call_args.kwargs["mode"], "label")
        self.assertAlmostEqual(run.call_args.kwargs["threshold"], 0.6)

    def test_postprocessing_parameter_is_forwarded(self):
        self._select_model("active_zone")
        self.widget.postprocessing_parameter_widgets["min_size"].setValue(515)
        self.widget.postprocessing_parameter_widgets["foreground_threshold"].setValue(0.75)
        image = np.zeros((4, 8, 8), dtype="float32")
        model = object()

        def get_layer_data(selector_name, return_metadata=False):
            return {} if return_metadata else image

        with (
            mock.patch.object(self.widget, "_get_layer_selector_data", side_effect=get_layer_data),
            mock.patch.object(self.widget, "_handle_resolution", return_value=None),
            mock.patch.object(segmentation_widget, "get_device", return_value="cpu"),
            mock.patch.object(segmentation_widget, "get_model", return_value=model),
            mock.patch.object(segmentation_widget, "_get_current_tiling", return_value={}),
            mock.patch.object(segmentation_widget, "run_segmentation", return_value=np.zeros_like(image)) as run,
            mock.patch.object(segmentation_widget, "show_info"),
        ):
            self.widget.on_predict()

        self.assertEqual(run.call_args.kwargs["min_size"], 515)
        self.assertEqual(run.call_args.kwargs["foreground_threshold"], 0.75)
        self.assertEqual(run.call_args.kwargs["model_type"], "active_zone")
        self.assertIs(run.call_args.kwargs["model"], model)

    def test_run_segmentation_shows_busy_state(self):
        self._select_model("active_zone")
        image = np.zeros((4, 8, 8), dtype="float32")
        captured = {}

        def get_layer_data(selector_name, return_metadata=False):
            return {} if return_metadata else image

        def fake_run(*args, **kwargs):
            # Capture the button state while the blocking segmentation is "running".
            captured["enabled"] = self.widget.predict_button.isEnabled()
            captured["text"] = self.widget.predict_button.text()
            return np.zeros_like(image)

        with (
            mock.patch.object(self.widget, "_get_layer_selector_data", side_effect=get_layer_data),
            mock.patch.object(self.widget, "_handle_resolution", return_value=None),
            mock.patch.object(segmentation_widget, "get_device", return_value="cpu"),
            mock.patch.object(segmentation_widget, "get_model", return_value=object()),
            mock.patch.object(segmentation_widget, "_get_current_tiling", return_value={}),
            mock.patch.object(segmentation_widget, "run_segmentation", side_effect=fake_run),
            mock.patch.object(segmentation_widget, "show_info"),
        ):
            self.widget.on_predict()

        # During the blocking call the button was disabled and relabeled to the busy text.
        self.assertFalse(captured["enabled"])
        self.assertEqual(captured["text"], "Computing…")
        # After completion the button is restored.
        self.assertTrue(self.widget.predict_button.isEnabled())
        self.assertEqual(self.widget.predict_button.text(), "Run Segmentation")

    def test_cristae_parameters_have_tooltips(self):
        self._select_model("cristae")
        spec = segmentation_widget._POSTPROCESSING_PARAMETER_SPECS[segmentation_widget.segment_cristae]
        for name in ("min_size", "foreground_threshold", "erosion_distance_nm"):
            with self.subTest(parameter=name):
                expected = spec[name]["tooltip"]
                self.assertTrue(expected)  # a non-empty description is configured
                self.assertEqual(self.widget.postprocessing_parameter_widgets[name].toolTip(), expected)


class TestSelectedPostprocessingParameters(unittest.TestCase):
    def test_active_zone_forwards_foreground_threshold(self):
        image = np.zeros((3, 4, 5), dtype="float32")
        predictions = np.zeros((1,) + image.shape, dtype="float32")
        expected = np.zeros_like(image, dtype="uint32")
        with (
            mock.patch.object(active_zone_module, "get_prediction", return_value=predictions),
            mock.patch.object(active_zone_module, "_run_segmentation", return_value=expected) as run,
        ):
            result = active_zone_module.segment_active_zone(
                image, model=object(), verbose=False, foreground_threshold=0.75
            )

        self.assertIs(result, expected)
        self.assertEqual(run.call_args.kwargs["foreground_threshold"], 0.75)

    def test_compartment_parameters_are_forwarded(self):
        image = np.zeros((3, 4, 5), dtype="float32")
        prediction = np.zeros_like(image)
        expected = np.zeros_like(image, dtype="uint32")
        with (
            mock.patch.object(compartments_module, "get_prediction", return_value=prediction),
            mock.patch.object(compartments_module, "_segment_compartments_3d", return_value=expected) as segment,
        ):
            result = compartments_module.segment_compartments(
                image,
                model=object(),
                verbose=False,
                boundary_threshold=0.65,
                n_slices_exclude=2,
                min_z_extent=7,
            )

        self.assertIs(result, expected)
        self.assertEqual(segment.call_args.kwargs["boundary_threshold"], 0.65)
        self.assertEqual(segment.call_args.kwargs["n_slices_exclude"], 2)
        self.assertEqual(segment.call_args.kwargs["min_z_extent"], 7)

    def test_compartment_min_z_extent_is_forwarded_by_name(self):
        prediction = np.zeros((2, 4, 5), dtype="float32")
        with (
            mock.patch.object(
                compartments_module, "_segment_compartments_2d", return_value=np.zeros((4, 5), dtype="uint32")
            ) as segment_2d,
            mock.patch.object(
                compartments_module, "_merge_segmentation_3d", return_value=np.zeros_like(prediction)
            ) as merge,
            mock.patch.object(compartments_module, "_postprocess_seg_3d", side_effect=lambda seg: seg),
        ):
            compartments_module._segment_compartments_3d(
                prediction, boundary_threshold=0.65, min_z_extent=7
            )

        self.assertEqual(merge.call_args.kwargs["min_z_extent"], 7)
        for call in segment_2d.call_args_list:
            self.assertEqual(call.kwargs["boundary_threshold"], 0.65)

    def test_ribbon_parameters_are_forwarded(self):
        image = np.zeros((3, 4, 5), dtype="float32")
        vesicles = np.zeros_like(image, dtype="uint32")
        predictions = {
            "ribbon": np.zeros_like(image),
            "PD": np.zeros_like(image),
            "membrane": np.zeros_like(image),
        }
        expected = {"ribbon": np.zeros_like(image, dtype="uint8")}
        with (
            mock.patch.object(
                inference_module, "segment_ribbon_synapse_structures", return_value=predictions
            ),
            mock.patch.object(inference_module, "_ribbon_AZ_postprocessing", return_value=expected) as postprocess,
        ):
            result = inference_module._segment_ribbon_AZ(
                image,
                model=object(),
                tiling=None,
                scale=None,
                verbose=False,
                extra_segmentation=vesicles,
                resolution=(1.0, 1.0, 1.0),
                n_slices_exclude=2,
                n_ribbons=3,
                min_membrane_size=17,
            )

        self.assertIs(result, expected)
        args = postprocess.call_args.args
        self.assertIs(args[0], predictions)
        self.assertIs(args[1], vesicles)
        self.assertEqual(args[2:], (2, 3, (1.0, 1.0, 1.0), 17))

    def test_ribbon_zero_slice_exclusion_uses_full_volume(self):
        prediction = np.zeros((3, 4, 5), dtype=bool)
        prediction[1, 2, 3] = True
        vesicles = np.zeros_like(prediction, dtype="uint32")

        with mock.patch("builtins.print"):
            result = segment_ribbon(prediction, vesicles, n_slices_exclude=0, n_ribbons=1)

        np.testing.assert_array_equal(result > 0, prediction)

    def test_cristae_parameters_are_forwarded(self):
        image = np.zeros((3, 4, 5), dtype="float32")
        mitochondria = np.ones_like(image, dtype="uint32")
        predictions = np.zeros((2,) + image.shape, dtype="float32")
        expected = np.zeros_like(image, dtype="uint32")
        with (
            mock.patch.object(cristae_module, "get_prediction", return_value=predictions),
            mock.patch.object(cristae_module, "_run_segmentation", return_value=expected) as run,
        ):
            result = cristae_module.segment_cristae(
                image,
                voxel_size=2.0,
                model=object(),
                verbose=False,
                extra_segmentation=mitochondria,
                foreground_threshold=0.75,
                erosion_distance_nm=8.0,
            )

        self.assertIs(result, expected)
        self.assertEqual(run.call_args.kwargs["foreground_threshold"], 0.75)
        self.assertEqual(run.call_args.kwargs["erode_voxels"], 4)


if __name__ == "__main__":
    unittest.main()
