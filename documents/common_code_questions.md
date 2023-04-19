## FAQ

- [FAQ](#faq)
- [How to process noisy label problem in CT dataset?](#how-to-process-noisy-label-problem-in-ct-dataset)
- [How to train new dataset with new organ?](#how-to-train-new-dataset-with-new-organ)
- [How to generate specific prediction for different dataset?](#how-to-generate-specific-prediction-for-different-dataset)
- [How to inference pseudo label without postlabel?](#how-to-inference-pseudo-label-without-postlabel)
- [How to customize CLIP embedding for your own dataset?](#how-to-customize-clip-embedding-for-your-own-dataset)

 
 ## How to process noisy label problem in CT dataset?
In some datasets, the annotation only includes organs and does not include tumors. In these cases, the tumors are improperly labeled as organs. To address this issue, we treat tumors as organs as well (for example, the liver tumor is also part of liver) and use the one vs. all manner for prediction. Each channel only predicts whether the voxels belong to the corresponding organ or tumor (voxels of liver tumors will be predicted as liver, as well as liver tumors). Only the labeled ground truth are used to back-propagate and update the network. This manner avoids the noisy label (false negative ground truth) problem for tumor segmentation.


 ## How to train new dataset with new organ?
1. Set the following index for new organ. (e.g. 33 for vermiform appendix)  
2. Check if there are any organs that are not divided into left and right in the dataset. (e.g. kidney, lung, etc.) The `RL_Splitd` in `label_transfer.py` is used to processed this case.  
3. Set up a new transfer list for new dataset in TEMPLATE (line 58 in label_transfer.py). (If a new dataset with Intestine labeled as 1 and vermiform appendix labeled as 2, we set the transfer list as [19, 33])  
4. Run the program `label_transfer.py` to get new post-processing labels.  
5. Rename the directory for new dataset with number prefix, e.g. 13. (This is used to identify which dataset the data comes from)
6. Generate txt file with postfix `xxx_train.txt`, `xxx_val.txt` and `xxx_test.txt` for new dataset and add the name to `--dataset_list` argument in `train.py` `validation.py` `test.py`.  

**The current Template** is shown below

|  Index   | Organ  |
|  ----  | ----  |
| 1  | Spleen |
| 2  | Right Kidney |
| 3  | Left Kidney |
| 4  | Gall Bladder |
| 5  | Esophagus |
| 6  | Liver |
| 7  | Stomach |
| 8  | Aorta |
| 9  | Postcava |
| 10  | Portal Vein and Splenic Vein |
| 11  | Pancreas |
| 12  | Right Adrenal Gland |
| 13  | Left Adrenal Gland |
| 14  | Duodenum |
| 15  | Hepatic Vessel |
| 16  | Right Lung |
| 17  | Left Lung |
| 18  | Colon |
| 19  | Intestine |
| 20  | Rectum |
| 21  | Bladder |
| 22  | Prostate |
| 23  | Head of Femur Left |
| 24  | Head of Femur Right |
| 25  | Celiac Truck |
| 26  | Kidney Tumor |
| 27  | Liver Tumor |
| 28  | Pancreas Tumor |
| 29  | Hepatic Vessel Tumor |
| 30  | Lung Tumor |
| 31  | Colon Tumor |
| 32  | Kidney Cyst |

 ## How to generate specific prediction for different dataset?
1. Ensure that the dataset is properly loaded and preprocessed so that it can be used by the model.  
2. Use `MERGE_MAPPING_v1` in `utils/utils.py` to control the label mapping from universal model to specific dataset.  
3. In the tuple, the first item is the index of the label in the universal model's template, and the second item is the index of the label in the specific dataset. For example, if the label for "liver" is indexed as 1 and the label for "liver tumor" is indexed as 2 in the LiTS dataset, the mapping tuple would be `(6,1), (27,2)`.  
4. Finally, you may want to post-process the final results as needed, as per your specific requirements.

 ## How to inference pseudo label without postlabel?
We add `pred_pseudo.py` file, where you can save the generated pseudo label without post label. `python -W ignore test.py --resume MODEL.pth --data_root_path YOUR_DATA_DIR`

 ## How to customize CLIP embedding for your own dataset?
The clip embedding is generated with text encode in [CLIP](https://github.com/openai/CLIP). We offer example in `pretrained_weights/clip_embedding.py` and you should revise `ORGAN_NAME` according to your dataset. 

