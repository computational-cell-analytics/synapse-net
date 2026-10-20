import os
import unittest


# We just test that the plugins can be imported.
class TestPlugin(unittest.TestCase):
    def test_classification_widget(self):
        from synapse_net.tools.classification_widget import ClassificationWidget

    def test_distance_measure_widget(self):
        from synapse_net.tools.distance_measure_widget import DistanceMeasureWidget

    def test_morphology_widget(self):
        from synapse_net.tools.morphology_widget import MorphologyWidget

    def test_refine_classification_widget(self):
        from synapse_net.tools.refine_classification_widget import RefineClassificationWidget

    def test_segmentation_widget(self):
        from synapse_net.tools.segmentation_widget import SegmentationWidget

    def test_vesicle_pool_widget(self):
        from synapse_net.tools.vesicle_pool_widget import VesiclePoolWidget


if __name__ == "__main__":
    unittest.main()
