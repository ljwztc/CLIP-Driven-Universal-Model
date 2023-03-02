## FAQ

- [FAQ](#faq)
  - [What is the text prompt when the model is used for tumor detection?](#What-is-the-text-prompt-when-the-model-is-used-for-tumor-detection)
  - [What is dataset-specific modelsd?](What-is-dataset-specific-modelsd)
  - [Can we see finetuning a universal model as generally more promising]()
 
 
 ## What is the text prompt when the model is used for tumor detection?
We used “A computerized tomography of a [CLS]” as the text prompt for both organs and tumors (Table 1). The resulting CLIP embedding is only used for representing a specific class (like widely used one-hot embedding), so the embedding is independent to the input image (or whether there is tumor/organ in this image). The basic mechanism is similar to [DoDNet](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_DoDNet_Learning_To_Segment_Multi-Organ_and_Tumors_From_Multiple_Partially_CVPR_2021_paper.html) or [MaskFormer](https://arxiv.org/abs/2107.06278).

 ## What is dataset-specific modelsd?
We only trained one Universal Model (on a combination of 14 datasets) to produce the results in Tables 2-5. In contrast, all the baseline methods are dataset-specific. These baseline models were trained on each specific dataset and were among the top in the competitions. We did not re-train these baseline models because their performances were publicly available on these benchmarks.

 ## Can we see finetuning a universal model as generally more promising?
Both using Swin UNETR, fully-supervised pre-training on 3,400 CT scans works significantly better than self-supervised pre-training on 5,000 CT scans. Seems like fully-supervised pre-training is preferable because the supervision density is much higher than self-supervision and the gap between the proxy and target tasks is smaller. Self-supervised learning has to design some proxy tasks such as Jigsaw, Rotation, Reconstruction, Coloration. The features learned from these tasks have limited relationships with the target segmentation tasks. In contrast, Universal Models learn features by segmentation directly. It is expected to have stronger features (at least for segmentation tasks) than other self-supervised pre-trained models. On the other hand, self-supervised learning has the potential to be scaled up by learning from a much larger number of unlabeled CT scans (e.g., 1 million). It is not easy for Universal Model to be scaled up by this amount of labeled CT scans. But this is my own opinion—need more investigation.