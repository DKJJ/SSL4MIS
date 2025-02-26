# Semi-supervised-learning-for-medical-image-segmentation.

* Recently, semi-supervised image segmentation has become a hot topic in medical image computing, unfortunately, there are only a few open-source codes and datasets, since the privacy policy and others. For easy evaluation and fair comparison, we are trying to build a semi-supervised medical image segmentation benchmark to boost the semi-supervised learning research in the medical image computing community. **If you are interested, you can push your implementations or ideas to this repo or contact us at any time**.  
* This repo has re-implemented these semi-supervised methods (with some modifications for semi-supervised medical image segmentation, more details please refer to these original works): (1) [Mean Teacher](https://papers.nips.cc/paper/6719-mean-teachers-are-better-role-models-weight-averaged-consistency-targets-improve-semi-supervised-deep-learning-results.pdf); (2) [Entropy Minimization](https://openaccess.thecvf.com/content_CVPR_2019/papers/Vu_ADVENT_Adversarial_Entropy_Minimization_for_Domain_Adaptation_in_Semantic_Segmentation_CVPR_2019_paper.pdf); (3) [Deep Adversarial Networks](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_47); (4) [Uncertainty Aware Mean Teacher](https://arxiv.org/pdf/1907.07034.pdf); (5) [Interpolation Consistency Training](https://arxiv.org/pdf/1903.03825.pdf); (6) [Uncertainty Rectified Pyramid Consistency](https://arxiv.org/pdf/2012.07042.pdf); (7) [Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226); (8) [Cross Consistency Training](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ouali_Semi-Supervised_Semantic_Segmentation_With_Cross-Consistency_Training_CVPR_2020_paper.pdf); (9) [Deep Co-Training](https://openaccess.thecvf.com/content_ECCV_2018/papers/Siyuan_Qiao_Deep_Co-Training_for_ECCV_2018_paper.pdf); (10) [Cross Teaching between CNN and Transformer](https://arxiv.org/pdf/2112.04894.pdf). In addition, several backbones networks (both 2D and 3D) are also supported in this repo, such as **UNet, nnUNet, VNet, AttentionUNet, ENet, Swin-UNet, etc**.

* This project was originally developed for our previous works [URPC](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_30) ([MICCAI2021](https://miccai2021.org/en/) early accept). Now and future, we are still working on extending it to be more user-friendly and support more approaches to further boost and ease this topic research. **If you use this codebase in your research, please cite the following works**:

		@InProceedings{luo2021urpc,
		author={Luo, Xiangde and Liao, Wenjun and Chen, Jieneng and Song, Tao and Chen, Yinan and Zhang, Shichuan and Chen, Nianyong and Wang, Guotai and Zhang, Shaoting},
		title={Efficient Semi-supervised Gross Target Volume of Nasopharyngeal Carcinoma Segmentation via Uncertainty Rectified Pyramid Consistency},
		booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021},
		year={2021},
		pages={318--329}}
		 
		@InProceedings{luo2021dtc,
		title={Semi-supervised Medical Image Segmentation through Dual-task Consistency},
		author={Luo, Xiangde and Chen, Jieneng and Song, Tao and  Wang, Guotai},
		journal={AAAI Conference on Artificial Intelligence},
		year={2021},
		pages={8801-8809}}
		
		@article{luo2021ctbct,
  		title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
  		author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
  		journal={arXiv preprint arXiv:2112.04894},
  		year={2021}}
		
		@misc{ssl4mis2020,
		title={{SSL4MIS}},
		author={Luo, Xiangde},
		howpublished={\url{https://github.com/HiLab-git/SSL4MIS}},
		year={2020}}
		
## Literature reviews of semi-supervised learning approach for medical image segmentation (**SSL4MIS**).
|Date|The First and Last Authors|Title|Code|Reference|
|---|---|---|---|---|
|2021-12|X. Xu and P. Yan|Shadow-consistent Semi-supervised Learning for Prostate Ultrasound Segmentation|[Code](https://github.com/DIAL-RPI/SCO-SSL)|[TMI2021](https://ieeexplore.ieee.org/document/9667363)|
|2021-12|L. Hu and Y. Wang|Semi-supervised NPC segmentation with uncertainty and attention guided consistency|None|[KBS2021](https://www.sciencedirect.com/science/article/abs/pii/S0950705121011205)|
|2021-12|J. Peng and M. Pedersoli|Self-Paced Contrastive Learning for Semi-supervised Medical Image Segmentation with Meta-labels|[Code](https://github.com/jizongFox/Self-paced-Contrastive-Learning)|[NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/8b5c8441a8ff8e151b191c53c1842a38-Paper.pdf)|
|2021-12|Y. Xie and Y. Xia|Intra- and Inter-pair Consistency for Semi-supervised Gland Segmentation|None|[TIP2021](https://ieeexplore.ieee.org/document/9662661)|
|2021-12|K. Chaitanya and E. Konukoglu|Local contrastive loss with pseudo-label based self-training for semi-supervised medical image segmentation|[Code](https://github.com/krishnabits001/pseudo_label_contrastive_training)|[Arxiv](https://arxiv.org/pdf/2112.09645.pdf)|
|2021-12|X. Luo and S. Zhang|Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer|[Code](https://github.com/HiLab-git/SSL4MIS)|[Arxiv](https://arxiv.org/pdf/2112.04894.pdf)|
|2021-12|Y. Zhang and J. Zhang|Uncertainty-Guided Mutual Consistency Learning for Semi-Supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2112.02508.pdf)|
|2021-12|J. Wang and Q. Zhou|Separated Contrastive Learning for Organ-at-Risk and Gross-Tumor-Volume Segmentation with Limited Annotat|[Code](https://github.com/jcwang123/Separate_CL)|[AAAI2022](https://arxiv.org/pdf/2112.02743.pdf)|
|2021-12|J. Chen and Y. Lu|MT-TransUNet: Mediating Multi-Task Tokens in Transformers for Skin Lesion Segmentation and Classification|[Code](https://github.com/JingyeChen/MT-TransUNet)|[Arxiv](https://arxiv.org/pdf/2112.01767.pdf)|
|2021-12|C. Seibold and R. Stiefelhagen|Reference-guided Pseudo-Label Generation for Medical Semantic Segmentation|None|[AAAI2022](https://arxiv.org/abs/2112.00735)|
|2021-11|X. Zheng and C. Sham|Uncertainty-Aware Deep Co-training for Semi-supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2111.11629v1.pdf)|
|2021-11|J. Peng and M. Pedersoli|Diversified Multi-prototype Representation for Semi-supervised Segmentation|[Code](https://github.com/jizongFox/MI-based-Regularized-Semi-supervised-Segmentation)|[Arxiv](https://arxiv.org/pdf/2111.08651.pdf)|
|2021-10|J. Hou and J. Deng|Semi-Supervised Semantic Segmentation of Vessel Images using Leaking Perturbations|None|[WACV2021](https://arxiv.org/pdf/2110.11998.pdf)|
|2021-10|M. Xu and J. Jacob|MisMatch: Learning to Change Predictive Confidences with Attention for Consistency-Based, Semi-Supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2110.12179.pdf)|
|2021-10|H. Wu and J. Qin|Collaborative and Adversarial Learning of Focused and Dispersive Representations for Semi-supervised Polyp Segmentation|None|[ICCV2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Collaborative_and_Adversarial_Learning_of_Focused_and_Dispersive_Representations_for_ICCV_2021_paper.pdf)|
|2021-10|Y. Shi and Y. Gao|Inconsistency-aware Uncertainty Estimation for Semi-supervised Medical Image Segmentation|[Code](https://github.com/koncle/CoraNet)|[TMI2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9558816)|
|2021-09|Z. Xu and R. Tong|All-Around Real Label Supervision: Cyclic Prototype Consistency Learning for Semi-supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2109.13930.pdf)|
|2021-09|G. Wang and S. Zhang|Semi-Supervised Segmentation of Radiation-Induced Pulmonary Fibrosis from Lung CT Scans with Multi-Scale Guided Dense Attention|[Code](https://github.com/HiLab-git/PF-Net)|[TMI2021](https://arxiv.org/pdf/2109.14172.pdf)|
|2021-09|K. Wang and Y. Wang|Tripled-Uncertainty Guided Mean Teacher Model for Semi-supervised Medical Image Segmentation|[Code](https://github.com/DeepMedLab/Tri-U-MT)|[MICCAI2021](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_42)|
|2021-09|H. Huang and R. Tong|3D Graph-S2Net: Shape-Aware Self-ensembling Network for Semi-supervised Segmentation with Bilateral Graph Convolution|None|[MICCAI2021](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_39)|
|2021-09|L. Zhu and B. Ooi|Semi-Supervised Unpaired Multi-Modal Learning for Label-Efficient Medical Image Segmentation|[Code](https://github.com/nusdbsystem/SSUMML)|[MICCAI2021](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_37)|
|2021-09|R. Zhang and G. Li|Self-supervised Correction Learning for Semi-supervised Biomedical Image Segmentation|[Code](https://github.com/ReaFly/SemiMedSeg)|[MICCAI2021](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_13)|
|2021-09|D. Kiyasseh and A. Chen|Segmentation of Left Atrial MR Images via Self-supervised Semi-supervised Meta-learning|None|[MICCAI2021](https://link.springer.com/chapter/10.1007/978-3-030-87196-3_2)|
|2021-09|Y. Wu and J. Cai|Enforcing Mutual Consistency of Hard Regions for Semi-supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2109.09960.pdf)|
|2021-09|X. Zeng and Y. Wang|Reciprocal Learning for Semi-supervised Segmentation|[Code](https://github.com/XYZach/RLSSS)|[MICCAI2021](https://dilincv.github.io/papers/reciprocal_miccai2021.pdf)|
|2021-09|G. Zhang and S. Jiang|Automatic segmentation of organs at risk and tumors in CT images of lung cancer from partially labelled datasets with a semi-supervised conditional nnU-Net|None|[CMPB2021](https://doi.org/10.1016/j.cmpb.2021.106419)|
|2021-09|J. Chen and G. Yang|Adaptive Hierarchical Dual Consistency for Semi-Supervised Left Atrium Segmentation on Cross-Domain Data|[Code](https://github.com/Heye-SYSU/AHDC)|[TMI2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9540830)|
|2021-09|X. Hu and Y. Shi|Semi-supervised Contrastive Learning for Label-efficient Medical Image Segmentation|[Code](https://github.com/xhu248/semi_cotrast_seg)|[MICCAI2021](https://arxiv.org/pdf/2109.07407.pdf)|
|2021-09|G. Chen and J. Shi|MTANS: Multi-Scale Mean Teacher Combined Adversarial Network with Shape-Aware Embedding for Semi-Supervised Brain Lesion Segmentation|[Code](https://github.com/wzcgx/MTANS)|[NeuroImage2021](https://www.sciencedirect.com/science/article/pii/S1053811921008417)|
|2021-08|H. Peiris and M. Harandi|Duo-SegNet: Adversarial Dual-Views for Semi-Supervised Medical Image Segmentation|[Code](https://github.com/himashi92/Duo-SegNet)|[MICCAI2021](https://arxiv.org/pdf/2108.11154.pdf)|
|2021-08|J. Sun and Y. Kong|Semi-Supervised Medical Image Semantic Segmentation with Multi-scale Graph Cut Loss|None|[ICIP2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9506098)|
|2021-08|X. Shen and J. Lu|PoissonSeg: Semi-Supervised Few-Shot Medical Image Segmentation via Poisson Learning|None|[ArXiv](https://arxiv.org/pdf/2108.11694.pdf)|
|2021-08|C. You and J. Duncan|SimCVD: Simple Contrastive Voxel-Wise Representation Distillation for Semi-Supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2108.06227.pdf)|
|2021-08|C. Li and P. Heng|Self-Ensembling Co-Training Framework for Semi-supervised COVID-19 CT Segmentation|None|[JBHI2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9511146)|
|2021-08|H. Yang and P. H. N. With|Medical Instrument Segmentation in 3D US by Hybrid Constrained Semi-Supervised Learning|None|[JBHI2021](https://arxiv.org/pdf/2107.14476.pdf)|
|2021-07|Q. Xu and X. Wang|Semi-supervised Medical Image Segmentation with Confidence Calibration|None|[IJCNN](https://ieeexplore.ieee.org/document/9534435)|
|2021-07|W. Ding and H. Hawash|RCTE: A Reliable and Consistent Temporal-ensembling Framework for Semi-supervised Segmentation of COVID-19 Lesions|None|[Information Fusion2021](https://www.sciencedirect.com/science/article/pii/S0020025521007490)|
|2021-06|X. Liu and S. A. Tsaftaris|Semi-supervised Meta-learning with Disentanglement for Domain-generalised Medical Image Segmentation|[Code](https://github.com/vios-s/DGNet)|[MICCAI2021](https://arxiv.org/pdf/2106.13292.pdf)|
|2021-06|P. Pandey and Mausam|Contrastive Semi-Supervised Learning for 2D Medical Image Segmentation|None|[MICCAI2021](https://arxiv.org/pdf/2106.06801v1.pdf)|
|2021-06|C. Li and Y. Yu|Hierarchical Deep Network with Uncertainty-aware Semi-supervised Learning for Vessel Segmentation|None|[Arxiv](https://arxiv.org/pdf/2105.14732.pdf)|
|2021-05|J. Xiang and S. Zhang|Self-Ensembling Contrastive Learning for Semi-Supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2105.12924.pdf)|
|2021-05|S. Li and C. Guan|Hierarchical Consistency Regularized Mean Teacher for Semi-supervised 3D Left Atrium Segmentation|None|[Arxiv](https://arxiv.org/pdf/2105.10369.pdf)|
|2021-05|C. You and J. Duncan|Momentum Contrastive Voxel-wise Representation Learning for Semi-supervised Volumetric Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2105.07059.pdf)|
|2021-05|Z. Xie and J. Yang|Semi-Supervised Skin Lesion Segmentation with Learning Model Confidence|None|[ICASSP2021](https://ieeexplore.ieee.org/document/9414297)|
|2021-04|S. Reiß and R. Stiefelhagen|Every Annotation Counts: Multi-label Deep Supervision for Medical Image Segmentation|None|[CVPR2021](https://arxiv.org/pdf/2104.13243.pdf)|
|2021-04|S. Chatterjee and A. Nurnberger|DS6, Deformation-aware Semi-supervised Learning: Application to Small Vessel Segmentation with Noisy Training Data|[Code](https://github.com/soumickmj/DS6)|[MIDL](https://openreview.net/pdf?id=2t0_AxD1otB)|
|2021-04|A. Meyer and M. Rak|Uncertainty-Aware Temporal Self-Learning (UATS): Semi-Supervised Learning for Segmentation of Prostate Zones and Beyond|[Code](https://github.com/suhitaghosh10/UATS)|[Arxiv](https://arxiv.org/pdf/2104.03840.pdf)|
|2021-04|Y. Li and P. Heng|Dual-Consistency Semi-Supervised Learning with Uncertainty Quantification for COVID-19 Lesion Segmentation from CT Images|None|[MICCAI2021](https://arxiv.org/pdf/2104.03225.pdf)|
|2021-03|Y. Zhang and C. Zhang|Dual-Task Mutual Learning for Semi-Supervised Medical Image Segmentation|[Code](https://github.com/YichiZhang98/DTML)|[PRCV2021](https://arxiv.org/pdf/2103.04708.pdf)|
|2021-03|J. Peng and C. Desrosiers|Boosting Semi-supervised Image Segmentation with Global and Local Mutual Information Regularization|[Code](https://github.com/jizongFox/MI-based-Regularized-Semi-supervised-Segmentation)|[MELBA](https://arxiv.org/pdf/2103.04813.pdf)|
|2021-03|Y. Wu and L. Zhang|Semi-supervised Left Atrium Segmentation with Mutual Consistency Training|None|[MICCAI2021](https://arxiv.org/pdf/2103.02911)|
|2021-02|J. Peng and Y. Wang|Medical Image Segmentation with Limited Supervision: A Review of Deep Network Models|None|[Arxiv](https://arxiv.org/pdf/2103.00429.pdf)|
|2021-02|J. Dolz and I. B. Ayed|Teach me to segment with mixed supervision: Confident students become masters|[Code](https://github.com/josedolz/MSL-student-becomes-master)|[IPMI2021](https://arxiv.org/pdf/2012.08051.pdf)|
|2021-02|C. Cabrera and K. McGuinness|Semi-supervised Segmentation of Cardiac MRI using Image Registration|None|[Under review for MIDL2021](https://openreview.net/pdf?id=ZMBea7SLdi)|
|2021-02|Y. Wang and A. Yuille|Learning Inductive Attention Guidance for Partially Supervised Pancreatic Ductal Adenocarcinoma Prediction|None|[TMI2021](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9357342)|
|2021-02|R. Alizadehsaniand U R. Acharya|Uncertainty-Aware Semi-supervised Method using Large Unlabelled and Limited Labeled COVID-19 Data|None|[Arxiv](https://arxiv.org/ftp/arxiv/papers/2102/2102.06388.pdf)|
|2021-02|D. Yang and D. Xu|Federated Semi-Supervised Learning for COVID Region Segmentation in Chest CT using Multi-National Data from China, Italy, Japan|None|[MedIA2021](https://www.sciencedirect.com/science/article/pii/S1361841521000384)|
|2020-01|E. Takaya and S. Kurihara|Sequential Semi-supervised Segmentation for Serial Electron Microscopy Image with Small Number of Labels|[Code](https://github.com/eichitakaya/Sequential-Semi-supervised-Segmentation)|[Journal of Neuroscience Methods](https://www.sciencedirect.com/science/article/pii/S0165027021000017)|
|2021-01|Y. Zhang and Z. He|Semi-supervised Cardiac Image Segmentation via Label Propagation and Style Transfer|None|[Arxiv](https://arxiv.org/pdf/2012.14785.pdf)|
|2020-12|H. Wang and D. Chen|Unlabeled Data Guided Semi-supervised Histopathology Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2012.09373.pdf)|
|2020-12|X. Luo and S. Zhang|Efficient Semi-supervised  Gross Target Volume of Nasopharyngeal Carcinoma Segmentation via Uncertainty Rectified Pyramid Consistency|[Code](https://github.com/HiLab-git/SSL4MIS)|[MICCAI2021](https://arxiv.org/pdf/2012.07042.pdf)|
|2020-12|M. Abdel‐Basset and M. Ryan|FSS-2019-nCov: A Deep Learning Architecture for Semi-supervised Few-Shot Segmentation of COVID-19 Infection|None|[Knowledge-Based Systems2020](https://www.sciencedirect.com/science/article/pii/S0950705120307760)|
|2020-11|N. Horlava and N. Scherf|A comparative study of semi- and self-supervised semantic segmentation of biomedical microscopy data|None|[Arxiv](https://arxiv.org/pdf/2011.08076.pdf)|
|2020-11|P. Wang and C. Desrosiers|Self-paced and self-consistent co-training for semi-supervised image segmentation|None|[MedIA2021](https://arxiv.org/pdf/2011.00325.pdf)|
|2020-10|Y. Sun and L. Wang|Semi-supervised Transfer Learning for Infant Cerebellum Tissue Segmentation|None|[MLMI2020](http://liwang.web.unc.edu/files/2020/10/Sun2020_Chapter_Semi-supervisedTransferLearnin.pdf)|
|2020-10|L. Chen and D. Merhof|Semi-supervised Instance Segmentation with a Learned Shape Prior|[Code](https://github.com/looooongChen/shape_prior_seg)|[LABELS2020](https://link.springer.com/chapter/10.1007/978-3-030-61166-8_10)|
|2020-10|S. Shailja and B.S. Manjunath|Semi supervised segmentation and graph-based tracking of 3D nuclei in time-lapse microscopy|[Code](https://github.com/s-shailja/ucsb_ctc)|[Arxiv](https://arxiv.org/pdf/2010.13343.pdf)|
|2020-10|L. Sun and Y. Yu|A Teacher-Student Framework for Semi-supervised Medical Image Segmentation From Mixed Supervision|None|[Arxiv](https://arxiv.org/pdf/2010.12219.pdf)|
|2020-10|J. Ma and X. Yang|Active Contour Regularized Semi-supervised Learning for COVID-19 CT Infection Segmentation with Limited Annotations|[Code](https://zenodo.org/record/4246238#.X6PSyogzZFE)|[Physics in Medicine & Biology2020](https://iopscience.iop.org/article/10.1088/1361-6560/abc04e/pdf)|
|2020-10|W. Hang and J. Qin|Local and Global Structure-Aware Entropy Regularized Mean Teacher Model for 3D Left Atrium Segmentation|[Code](https://github.com/3DMRIs/LG-ER-MT)|[MICCAI2020](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_55)|
|2020-10|K. Tan and J. Duncan|A Semi-supervised Joint Network for Simultaneous Left Ventricular Motion Tracking and Segmentation in 4D Echocardiography|None|[MICCAI2020](https://link.springer.com/chapter/10.1007/978-3-030-59725-2_45)|
|2020-10|Y. Wang and Z. He|Double-Uncertainty Weighted Method for Semi-supervised Learning|None|[MICCAI2020](https://link.springer.com/chapter/10.1007%2F978-3-030-59710-8_53)|
|2020-10|K. Fang and W. Li|DMNet: Difference Minimization Network for Semi-supervised Segmentation in Medical Images|None|[MICCAI2020](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_52)|
|2020-10|X. Cao and L. Cheng|Uncertainty Aware Temporal-Ensembling Model for Semi-supervised ABUS Mass Segmentation|None|[TMI2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9214845)|
|2020-09|Z. Zhang and W. Zhang|Semi-supervised Semantic Segmentation of Organs at Risk on 3D Pelvic CT Images|None|[Arxiv](https://arxiv.org/ftp/arxiv/papers/2009/2009.09571.pdf)|
|2020-09|J. Wang and G. Xie|Semi-supervised Active Learning for Instance Segmentation via Scoring Predictions|None|[BMVC2020](http://scholar.google.com/scholar_url?url=https://www.bmvc2020-conference.com/assets/papers/0031.pdf&hl=zh-CN&sa=X&d=4465129548770333798&ei=u85pX6XsJNKsmwG4zr6YCw&scisig=AAGBfm1GGUNfq7zId6WBRyppRRjnPSpLsQ&nossl=1&oi=scholaralrt&html=&cited-by=)|
|2020-09|X. Luo and S. Zhang|Semi-supervised Medical Image Segmentation through Dual-task Consistency|[Code](https://github.com/Luoxd1996/DTC)|[AAAI2021](https://arxiv.org/pdf/2009.04448.pdf)|
|2020-08|X. Huo and Q. Tian|ATSO: Asynchronous Teacher-Student Optimization for Semi-Supervised Medical Image Segmentation|None|[Arxiv](https://arxiv.org/pdf/2006.13461.pdf)|
|2020-08|Y. Xie and Y. Xia|Pairwise Relation Learning for Semi-supervised Gland Segmentation|None|[MICCAI2020](https://arxiv.org/pdf/2008.02699.pdf)|
|2020-07|K. Chaitanya and E. Konukoglu|Semi-supervised Task-driven Data Augmentation for Medical Image Segmentation|[Code](https://github.com/krishnabits001/task_driven_data_augmentation)|[Arxiv](https://arxiv.org/pdf/2007.05363.pdf)|
|2020-07|S. Li and X. He|Shape-aware Semi-supervised 3D Semantic Segmentation for Medical Images|[Code](https://github.com/kleinzcy/SASSnet)|[MICCAI2020](https://arxiv.org/pdf/2007.10732.pdf)|
|2020-07|Y. Li and Y. Zheng |Self-Loop Uncertainty: A Novel Pseudo-Label for Semi-Supervised Medical Image Segmentation|None|[MICCAI2020](https://arxiv.org/abs/2007.09854)|
|2020-07|Z. Zhao and P. Heng|Learning Motion Flows for Semi-supervised Instrument Segmentation from Robotic Surgical Video|[Code](https://github.com/zxzhaoeric/Semi-InstruSeg/)|[MICCAI2020](https://arxiv.org/abs/2007.02501)|
|2020-07|Y. Zhou and P. Heng|Deep Semi-supervised Knowledge Distillation for Overlapping Cervical Cell Instance Segmentation|[Code](https://github.com/SIAAAAAA/MMT-PSM)|[MICCAI2020](https://arxiv.org/pdf/2007.10787.pdf)|
|2020-07|A. Tehrani and H. Rivaz|Semi-Supervised Training of Optical Flow Convolutional Neural Networks in Ultrasound Elastography|None|[MICCAI2020](https://arxiv.org/pdf/2007.01421.pdf)|
|2020-07|Y. He and S. Li|Dense biased networks with deep priori anatomy and hard region adaptation: Semi-supervised learning for fine renal artery segmentation|None|[MedIA2020](https://www.sciencedirect.com/science/article/pii/S1361841520300864)|
|2020-07|J. Peng and C. Desrosiers|Mutual information deep regularization for semi-supervised segmentation|[Code](https://github.com/jizongFox/deep-clustering-toolbox)|[MIDL2020](https://openreview.net/pdf?id=iunvffXgPm)|
|2020-07|Y. Xia and H. Roth|Uncertainty-aware multi-view co-training for semi-supervised medical image segmentation and domain adaptation|None|[WACV2020](https://arxiv.org/abs/1811.12506),[MedIA2020](https://www.sciencedirect.com/science/article/pii/S1361841520301304)|
|2020-07|X. Li and P. Heng|Transformation-Consistent Self-Ensembling Model for Semisupervised Medical Image Segmentation|[Code](https://github.com/xmengli999/TCSM)|[TNNLS2020](https://ieeexplore.ieee.org/document/9104928)|
|2020-06|F. Garcıa and S. Ourselin|Simulation of Brain Resection for Cavity Segmentation Using Self-Supervised and Semi-Supervised Learning|None|[MICCAI2020](https://arxiv.org/pdf/2006.15693.pdf)|
|2020-06|H. Yang and P. With|Deep Q-Network-Driven Catheter Segmentation in 3D US by Hybrid Constrained Semi-Supervised Learning and Dual-UNet|None|[MICCAI2020](https://arxiv.org/pdf/2006.14702.pdf)|
|2020-05|G. Fotedar and X. Ding|Extreme Consistency: Overcoming Annotation Scarcity and Domain Shifts|None|[MICCAI2020](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_68)|
|2020-04|C. Liu and C. Ye|Semi-Supervised Brain Lesion Segmentation Using Training Images with and Without Lesions|None|[ISBI2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9098565)|
|2020-04|R. Li and D. Auer|A Generic Ensemble Based Deep Convolutional Neural Network for Semi-Supervised Medical Image Segmentation|[Code](https://github.com/ruizhe-l/semi-segmentation)|[ISBI2020](https://arxiv.org/pdf/2004.07995.pdf)|
|2020-04|K. Ta and J. Duncan|A Semi-Supervised Joint Learning Approach to Left Ventricular Segmentation and Motion Tracking in Echocardiography|None|[ISBI2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9098664)|
|2020-04|Q. Chang and D. Metaxas|Soft-Label Guided Semi-Supervised Learning for Bi-Ventricle Segmentation in Cardiac Cine MRI|None|[ISBI2020](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9098546)|
|2020-04|D. Fan and L. Shao|Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images|[Code](https://github.com/DengPingFan/Inf-Net)|[TMI2020](https://ieeexplore.ieee.org/document/9098956)|
|2019-10|L. Yu and P. Heng|Uncertainty-aware self-ensembling model for semi-supervised 3D left atrium segmentation|[Code](https://github.com/yulequan/UA-MT)|[MICCAI2019](https://arxiv.org/pdf/1907.07034.pdf)|
|2019-10|G. Bortsova and M. Bruijne|Semi-Supervised Medical Image Segmentation via Learning Consistency under Transformations|None|[MICCAI2019](https://arxiv.org/pdf/1911.01218.pdf)|
|2019-10|Y. He and S. Li|DPA-DenseBiasNet: Semi-supervised 3D Fine Renal Artery Segmentation with Dense Biased Network and Deep Priori Anatomy|None|[MICCAI2019](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_16)|
|2019-10|H. Zheng and X. Han|Semi-supervised Segmentation of Liver Using Adversarial Learning with Deep Atlas Prior|None|[MICCAI2019](https://link.springer.com/chapter/10.1007/978-3-030-32226-7_17)|
|2019-10|P. Ganayea and H. Cattin|Removing Segmentation Inconsistencies with Semi-Supervised Non-Adjacency Constraint|[Code](https://github.com/trypag/NonAdjLoss)|[MedIA2019](https://www.sciencedirect.com/science/article/abs/pii/S1361841519300866)|
|2019-10|Y. Zhao and C. Liu|Multi-view Semi-supervised 3D Whole Brain Segmentation with a Self-ensemble Network|None|[MICCAI2019](https://link.springer.com/chapter/10.1007/978-3-030-32248-9_29)|
|2019-10|H. Kervade and I. Ayed|Curriculum semi-supervised segmentation|None|[MICCAI2019](https://arxiv.org/pdf/1904.05236.pdf)|
|2019-10|S. Chen and M. Bruijne|Multi-task Attention-based Semi-supervised Learning for Medical Image Segmentation|None|[MICCAI2019](https://arxiv.org/pdf/1907.12303.pdf)|
|2019-10|Z. Xu and M. Niethammer|DeepAtlas: Joint Semi-Supervised Learning of Image Registration and Segmentation|None|[MICCAI2019](https://arxiv.org/pdf/1904.08465.pdf)|
|2019-10|S. Sedai and R. Garnavi|Uncertainty Guided Semi-supervised Segmentation of Retinal Layers in OCT Images|None|[MICCAI2019](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_32)|
|2019-10|G. Pombo and P. Nachev|Bayesian Volumetric Autoregressive Generative Models for Better Semisupervised Learning|[Code](https://github.com/guilherme-pombo/3DPixelCNN)|[MICCAI2019](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_47)|
|2019-06|W. Cui and C. Ye|Semi-Supervised Brain Lesion Segmentation with an Adapted Mean Teacher Model|None|[IPMI2019](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_43)|
|2019-06|K. Chaitanya and E. Konukoglu|Semi-supervised and Task-Driven Data Augmentation|[Code](https://github.com/krishnabits001/task_driven_data_augmentation)|[IPMI2019](http://link-springer-com-443.webvpn.fjmu.edu.cn/chapter/10.1007%2F978-3-030-20351-1_3)|
|2019-04|M. Jafari and P. Abolmaesumi|Semi-Supervised Learning For Cardiac Left Ventricle Segmentation Using Conditional Deep Generative Models as Prior|None|[ISBI2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8759292)|
|2019-03|Z. Zhao and Z. Zeng|Semi-Supervised Self-Taught Deep Learning for Finger Bones Segmentation|None|[BHI](https://ieeexplore.ieee.org/document/8834460)|
|2019-03|J. Peng and C. Desrosiers|Deep co-training for semi-supervised image segmentation|[Code](https://github.com/jizongFox/deep-clustering-toolbox)|[PR2020](https://www.sciencedirect.com/science/article/pii/S0031320320300741/pdfft?md5=ecbfff8052e991b23c1796f97588d7e5&pid=1-s2.0-S0031320320300741-main.pdf)|
|2019-01|Y. Zhou and A. Yuille|Semi-Supervised 3D Abdominal Multi-Organ Segmentation via Deep Multi-Planar Co-Training|None|[WACV2019](http://www.robots.ox.ac.uk/~tvg/publications/2019/dmpct_wacv.pdf)|
|2018-10|P. Ganaye and H. Cattin|Semi-supervised Learning for Segmentation Under Semantic Constraint|[Code](https://github.com/trypag/NonAdjLoss)|[MICCAI2018](https://link.springer.com/chapter/10.1007/978-3-030-00931-1_68)|
|2018-10|A. Chartsias and S. Tsaftari|Factorised spatial representation learning: application in semi-supervised myocardial segmentation|None|[MICCAI2018](https://arxiv.org/pdf/1803.07031.pdf)|
|2018-09|X. Li and P. Heng|Semi-supervised Skin Lesion Segmentation via Transformation Consistent Self-ensembling Model|[Code](https://github.com/xmengli999/TCSM)|[BMVC2018](https://arxiv.org/pdf/1808.03887.pdf)|
|2018-04|Z. Feng and D. Shen|Semi-supervised learning for pelvic MR image segmentation based on multi-task residual fully convolutional networks|None|[ISBI2018](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8363713)|
|2017-09|L. Gu and S. Aiso|Semi-supervised Learning for Biomedical Image Segmentation via Forest Oriented Super Pixels(Voxels)|None|[MICCAI2017](https://link.springer.com/chapter/10.1007/978-3-319-66182-7_80)|
|2017-09|S. Sedai and R. Garnavi|Semi-supervised Segmentation of Optic Cup in Retinal Fundus Images Using Variational Autoencoder|None|[MICCAI2017](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_9)|
|2017-09|W. Bai and D. Rueckert|Semi-supervised Learning for Network-Based Cardiac MR Image Segmentation|None|[MICCAI2017](https://link.springer.com/chapter/10.1007/978-3-319-66185-8_29)|

## Code for semi-supervised medical image segmentation.
Some implementations of semi-supervised learning methods can be found in this [Link](https://github.com/Luoxd1996/SSL4MIS/tree/master/code).

## Conclusion
* This repository provides daily-update literature reviews, algorithms' implementation, and some examples of using PyTorch for semi-supervised medical image segmentation. The project is under development. Currently, it supports 2D and 3D semi-supervised image segmentation and includes five widely-used algorithms' implementations.
		
* In the next two or three months, we will provide more algorithms' implementations, examples, and pre-trained models.

## Questions and Suggestions
* If you have any questions or suggestions about this project, please contact me through email: `luoxd1996@gmail.com` or QQ Group (Chinese):`906808850`. 
