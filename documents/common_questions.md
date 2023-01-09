## FAQ

- [FAQ](#faq)
  - [How to process noisy label problem in CT dataset?](#how-to-process-noisy-label-problem-in-ct-dataset)
 
 ## How to process noisy label problem in CT dataset?
In some datasets, the annotation only includes organs and does not include tumors. In these cases, the tumors are improperly labeled as organs. To address this issue, we treat tumors as organs as well (for example, the liver tumor is also part of liver) and use the one vs. all manner for prediction. Each channel only predicts whether the voxels belong to the corresponding organ or tumor (voxels of liver tumors will be predicted as liver, as well as liver tumors). Only the labeled ground truth are used to back-propagate and update the network. This manner avoids the noisy label (false negative ground truth) problem for tumor segmentation.