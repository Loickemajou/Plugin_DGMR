import pytest
import numpy as np
import tensorflow as tf
from dgmr_module_plugin import forecast


def test_forecast_incorrect_shape():
    # Test Case: Ensure that the function raises an exception for incorrect input shape

    # Create a dummy input tensor of incorrect shape (3, 256, 256, 1)
    input_frames = tf.convert_to_tensor(
        np.random.rand(3, 256, 256, 1), dtype=tf.float32
    )

    # Check if an exception is raised
    with pytest.raises(ValueError):
        forecast(input_frames)


def test_forecast_shape():
    # Test Case: Check the output shape of the forecast function

    # Create a dummy input tensor of shape (4, 256, 256, 1)
    input_frames = tf.convert_to_tensor(
        np.random.rand(4, 256, 256, 1), dtype=tf.float32
    )

    # Call the forecast function
    output = forecast(input_frames, num_samples=1, include_input_frames_in_result=False)

    # Check the output shape
    expected_shape = (1, 18, 256, 256, 1)  # 1 sample, 18 predicted frames
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"


def test_forecast_includes_input_frames():
    # Test Case: Check if the output includes input frames when specified

    # Create a dummy input tensor of shape (4, 256, 256, 1)
    input_frames = tf.convert_to_tensor(
        np.random.rand(4, 256, 256, 1), dtype=tf.float32
    )

    # Call the forecast function with include_input_frames_in_result=True
    output = forecast(input_frames, num_samples=1, include_input_frames_in_result=True)

    # Check the output shape
    expected_shape = (
        1,
        22,
        256,
        256,
        1,
    )  # 1 sample, 22 total frames (4 input + 18 output)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"


# To run the tests, use the command: pytest -v test.py
