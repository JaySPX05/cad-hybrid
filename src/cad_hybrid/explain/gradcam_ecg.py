"""Grad-CAM heatmaps for the ECG branch.

Use these to check the model looks at the traces (ST segments, Q waves), not at paper
margins, grid patterns or printed labels. Attention on non-trace areas means the model
has learned a shortcut.
"""

from __future__ import annotations


def heatmap(model, input_tensor, target_class: int, target_layer=None):
    """Return a Grad-CAM heatmap (H x W, values 0-1) for one image tensor.

    ``target_layer`` defaults to the last convolutional block; confirm the right layer
    for the chosen backbone.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    if target_layer is None:
        raise NotImplementedError("Set target_layer for the chosen backbone (e.g. model.conv_head)")
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        return cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0]
