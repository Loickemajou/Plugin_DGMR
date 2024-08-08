# -*- coding: utf-8 -*-
"""
pysteps.nowcasts.dgmr
======================

This is a deep learning model for performing nowcasting on radar images.

The DGMR model is a state-of-the-art generative model designed for weather nowcasting. 
It leverages a combination of convolutional neural networks (CNNs) and generative 
adversarial networks (GANs) to produce high-resolution, realistic rainfall forecasts.

DGMR model weight is not made public and need to be downloaded with Google cloud shell terminal, with the link "gsutil -m cp -r \
  "gs://dm-nowcasting-example-data/tfhub_snapshots" \ ."
https://console.cloud.google.com/storage/browser/dm-nowcasting-example-data/tfhub_snapshots;tab=objects?cloudshell=true&project=friendly-retina-382415&prefix&forceOnObjectsSortingFiltering=false

For more details, please see Ravuri, S., Lenc, K., Willson, M. et al. 
Skilful precipitation nowcasting using deep generative models of radar. Nature 597, 672–677 (2021).


.. autosummary::
    :toctree: ../generated/

    forecast

"""


# Import the needed libraries
import os
import tensorflow as tf
import tensorflow_hub as hub






  
def _load_model(input_height, input_width,TFHUB_BASE_PATH):
    """

     Parameters
     ----------
     input_height: int
      The height of the frames supported by the model
     input_width: int
      The width of the frames supported by the model

     TFHUB_BASE_PATH: string
         Contains the path where the model is saved
         should be in the format {path to the folder}/tfhub_snapshots


     Returns
     -------
    The loaded model
    """
    print("--> Loading model...")

    hub_module = hub.load(
        os.path.join(TFHUB_BASE_PATH, f"{input_height}x{input_width}")
    )
    # Note this has loaded a legacy TF1 model for running under TF2 eager mode.
    # This means we need to access the module via the "signatures" attribute. See
    # https://github.com/tensorflow/hub/blob/master/docs/migration_tf2.md#using-lower-level-apis
    # for more information.
    return hub_module.signatures["default"]


def forecast( input_frames,path, num_samples=1, include_input_frames_in_result=False, **kwargs
):
    module=_load_model(256,256,path)
    print("---> Model Loaded, Making prediction")
    """
    
      Does the prediction on the input frames.


      Parameters
      ----------
      input frames: tensor (T_out,H,W,C) where T=4, H=256,W=256,C=1

      num_samples: The number of different samples to draw.

      include_input_frames_in_result: If True, will return a total of 22 frames
      along the time axis, the 4 input frames followed by 18 predicted frames.
      Otherwise will only return the 18 predicted frames.
      path: string
      Contains the path where the model is saved
      should be in the format {path to the folder}/tfhub_snapshots
      

      Hints
      -----
      It should be noted that the inut frame of the DGMR model should be well 
      prcocessed for a good functionnning of the model. That is, 
      -The  model takes four frames at a times for prediction (concatenate the four frames into 
      ont frame of size 4)
      -- Crop the images to have a size of 256 by 256.



      Returns
      -------
      A tensor of shape (num_samples,T_out,H,W,C), where T_out is either 18 or 22
      as described above.

    Make predictions from a TF-Hub snapshot of the 'Generative Method' model.




    """
    
    NUM_INPUT_FRAMES = 4
    #Checks whether the input frame has the correct shape supported by the model
    if input_frames.shape==(4,256,256,1):
        input_frames = tf.math.maximum(input_frames, 0.0)
        # Add a batch dimension and tile along it to create a copy of the input for
        # each sample:
        input_frames = tf.expand_dims(input_frames, 0)
        input_frames = tf.tile(input_frames, multiples=[num_samples, 1, 1, 1, 1])

        # Sample the latent vector z for each sample:
        _, input_signature = module.structured_input_signature
        z_size = input_signature["z"].shape[1]
        z_samples = tf.random.normal(shape=(num_samples, z_size))

        inputs = {
            "z": z_samples,
            "labels$onehot": tf.ones(shape=(num_samples, 1)),
            "labels$cond_frames": input_frames,
        }
        samples = module(**inputs)["default"]
        if not include_input_frames_in_result:
            # The module returns the input frames alongside its sampled predictions, we
            # slice out just the predictions:
            samples = samples[:, NUM_INPUT_FRAMES:, ...]

        # Take positive values of rainfall only.
        samples = tf.math.maximum(samples, 0.0)
    else:
        raise Exception('Incorrect shape  for DGMR. DGMR shapes need to be preprocessed as follows\n'+
                        '(4,256,256,1) where 4=batch size: four frames\n'+
                                       '256 by 256 the size of the cropped images\n'+
                                        '1: Indicating only one channel in this case (precipitation)')
        
    return samples