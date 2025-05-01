# Path Planning on Multi-level Point Cloud with a Weighted Traversability Graph 

**Title (ZH)**: 多级点云权重通过性图路径规划 

**Authors**: Yujie Tang, Quan Li, Hao Geng, Yangmin Xie, Hang Shi, Yusheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21622)  

**Abstract**: This article proposes a new path planning method for addressing multi-level terrain situations. The proposed method includes innovations in three aspects: 1) the pre-processing of point cloud maps with a multi-level skip-list structure and data-slimming algorithm for well-organized and simplified map formalization and management, 2) the direct acquisition of local traversability indexes through vehicle and point cloud interaction analysis, which saves work in surface fitting, and 3) the assignment of traversability indexes on a multi-level connectivity graph to generate a weighted traversability graph for generally search-based path planning. The A* algorithm is modified to utilize the traversability graph to generate a short and safe path. The effectiveness and reliability of the proposed method are verified through indoor and outdoor experiments conducted in various environments, including multi-floor buildings, woodland, and rugged mountainous regions. The results demonstrate that the proposed method can properly address 3D path planning problems for ground vehicles in a wide range of situations. 

**Abstract (ZH)**: 本文提出了一种应对多级地形情况的新型路径规划方法。该方法在三个方面进行了创新：1) 使用多级跳跃列表结构和数据瘦身算法对点云地图进行预处理，以实现地图的有序化和简化管理，2) 通过车辆与点云的交互分析直接获取局部可通行性指标，省去了表面拟合的工作，3) 在多级连接图上分配可通行性指标以生成加权可通行性图，用于基于搜索的路径规划。对A*算法进行了修改，使其能够利用可通行性图生成短且安全的路径。通过在多种环境，包括多层建筑物、林地和崎岖的山区等地进行室内外实验，验证了所提出方法的有效性和可靠性。该研究成果表明，所提出的方法能够妥善解决地面车辆在多种情况下的三维路径规划问题。 

---
# Real Time Semantic Segmentation of High Resolution Automotive LiDAR Scans 

**Title (ZH)**: 实时高分辨率汽车LiDAR扫描语义分割 

**Authors**: Hannes Reichert, Benjamin Serfling, Elijah Schüssler, Kerim Turacan, Konrad Doll, Bernhard Sick  

**Link**: [PDF](https://arxiv.org/pdf/2504.21602)  

**Abstract**: In recent studies, numerous previous works emphasize the importance of semantic segmentation of LiDAR data as a critical component to the development of driver-assistance systems and autonomous vehicles. However, many state-of-the-art methods are tested on outdated, lower-resolution LiDAR sensors and struggle with real-time constraints. This study introduces a novel semantic segmentation framework tailored for modern high-resolution LiDAR sensors that addresses both accuracy and real-time processing demands. We propose a novel LiDAR dataset collected by a cutting-edge automotive 128 layer LiDAR in urban traffic scenes. Furthermore, we propose a semantic segmentation method utilizing surface normals as strong input features. Our approach is bridging the gap between cutting-edge research and practical automotive applications. Additionaly, we provide a Robot Operating System (ROS2) implementation that we operate on our research vehicle. Our dataset and code are publicly available: this https URL. 

**Abstract (ZH)**: 近年来，许多先前研究强调了LiDAR数据语义分割在驾驶员辅助系统和自动驾驶车辆开发中的重要性，是其关键组成部分。然而，许多最先进的方法是在过时且分辨率较低的LiDAR传感器上进行测试，并且难以满足实时约束。本研究提出了一种针对现代高分辨率LiDAR传感器的新型语义分割框架，旨在同时满足准确性和实时处理需求。我们提出了一种新型LiDAR数据集，该数据集由先进的汽车128层LiDAR在城市交通场景中收集。此外，我们提出了一种利用表面法线作为强大输入特征的语义分割方法。我们的方法填补了尖端研究与实际汽车应用之间的差距。另外，我们提供了在研究车辆上运行的基于Robot Operating System (ROS2)的实现。我们的数据集和代码已公开提供：[此链接]。 

---
# REHEARSE-3D: A Multi-modal Emulated Rain Dataset for 3D Point Cloud De-raining 

**Title (ZH)**: REHEARSE-3D: 一种多模态模拟雨滴点云去雨数据集 

**Authors**: Abu Mohammed Raisuddin, Jesper Holmblad, Hamed Haghighi, Yuri Poledna, Maikol Funk Drechsler, Valentina Donzella, Eren Erdal Aksoy  

**Link**: [PDF](https://arxiv.org/pdf/2504.21699)  

**Abstract**: Sensor degradation poses a significant challenge in autonomous driving. During heavy rainfall, the interference from raindrops can adversely affect the quality of LiDAR point clouds, resulting in, for instance, inaccurate point measurements. This, in turn, can potentially lead to safety concerns if autonomous driving systems are not weather-aware, i.e., if they are unable to discern such changes. In this study, we release a new, large-scale, multi-modal emulated rain dataset, REHEARSE-3D, to promote research advancements in 3D point cloud de-raining. Distinct from the most relevant competitors, our dataset is unique in several respects. First, it is the largest point-wise annotated dataset, and second, it is the only one with high-resolution LiDAR data (LiDAR-256) enriched with 4D Radar point clouds logged in both daytime and nighttime conditions in a controlled weather environment. Furthermore, REHEARSE-3D involves rain-characteristic information, which is of significant value not only for sensor noise modeling but also for analyzing the impact of weather at a point level. Leveraging REHEARSE-3D, we benchmark raindrop detection and removal in fused LiDAR and 4D Radar point clouds. Our comprehensive study further evaluates the performance of various statistical and deep-learning models. Upon publication, the dataset and benchmark models will be made publicly available at: this https URL. 

**Abstract (ZH)**: 传感器退化对自主驾驶构成显著挑战。在暴雨中，雨滴的干扰会负面影响LiDAR点云质量，例如导致不准确的点测量。如果自主驾驶系统不具备天气 aware 性能，即无法识别此类变化，这可能会导致安全隐患。在这项研究中，我们发布了一个新的大规模多模式模拟降雨数据集REHEARSE-3D，以促进3D点云除雨研究的进步。与最相关竞争对手相比，我们的数据集在多个方面具有独特性。首先，它是最大的逐点标注数据集；其次，它是唯一一个包含在受控天气环境中记录的高分辨率LiDAR数据（LiDAR-256）和4D雷达点云的数据集，无论白天还是黑夜。此外，REHEARSE-3D包括了降雨特性信息，这对于传感器噪声建模和点级天气影响分析都具有重要意义。利用REHEARSE-3D，我们对融合LiDAR和4D雷达点云的雨滴检测与去除进行了基准测试。我们进一步的全面研究还评估了各种统计和深度学习模型的性能。发布后，数据集和基准模型将公开发布于：this https URL。 

---
# GauSS-MI: Gaussian Splatting Shannon Mutual Information for Active 3D Reconstruction 

**Title (ZH)**: GauSS-MI: 高斯点云 Shannon 互信息active 3D重建 

**Authors**: Yuhan Xie, Yixi Cai, Yinqiang Zhang, Lei Yang, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21067)  

**Abstract**: This research tackles the challenge of real-time active view selection and uncertainty quantification on visual quality for active 3D reconstruction. Visual quality is a critical aspect of 3D reconstruction. Recent advancements such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have notably enhanced the image rendering quality of reconstruction models. Nonetheless, the efficient and effective acquisition of input images for reconstruction-specifically, the selection of the most informative viewpoint-remains an open challenge, which is crucial for active reconstruction. Existing studies have primarily focused on evaluating geometric completeness and exploring unobserved or unknown regions, without direct evaluation of the visual uncertainty within the reconstruction model. To address this gap, this paper introduces a probabilistic model that quantifies visual uncertainty for each Gaussian. Leveraging Shannon Mutual Information, we formulate a criterion, Gaussian Splatting Shannon Mutual Information (GauSS-MI), for real-time assessment of visual mutual information from novel viewpoints, facilitating the selection of next best view. GauSS-MI is implemented within an active reconstruction system integrated with a view and motion planner. Extensive experiments across various simulated and real-world scenes showcase the superior visual quality and reconstruction efficiency performance of the proposed system. 

**Abstract (ZH)**: 本研究解决了实时主动视角选择和视觉质量不确定性量化在主动3D重建中的挑战。视觉质量是3D重建的关键 aspects。最近的进展如神经辐射场（NeRF）和三维高斯散点图（3DGS）显著地提高了重建模型的图像渲染质量。然而，为重建高效和有效地获取输入图像，特别是在选择最有信息量的视角方面的挑战仍然未解决，这是主动重建的关键。现有研究主要集中在评估几何完整性并探索未观察或未知区域，而没有直接评估重建模型内的视觉不确定性。为弥补这一差距，本文提出一种概率模型来量化每个高斯的视觉不确定性。基于香农互信息，我们提出了高斯散点图香农互信息（GauSS-MI）准则，用于实时评估新视角下的视觉互信息，辅助选择下一个最佳视角。GauSS-MI 在结合视图和运动规划器的主动重建系统中实现。广泛实验在各种模拟和真实场景中展示了所提系统的卓越视觉质量和重建效率性能。 

---
# Early Exit and Multi Stage Knowledge Distillation in VLMs for Video Summarization 

**Title (ZH)**: Early Exit和多阶段知识蒸馏在视频摘要的VLMs中的应用 

**Authors**: Anas Anwarul Haq Khan, Utkarsh Verma, Prateek Chanda, Ganesh Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21831)  

**Abstract**: We introduce DEEVISum (Distilled Early Exit Vision language model for Summarization), a lightweight, efficient, and scalable vision language model designed for segment wise video summarization. Leveraging multi modal prompts that combine textual and audio derived signals, DEEVISum incorporates Multi Stage Knowledge Distillation (MSKD) and Early Exit (EE) to strike a balance between performance and efficiency. MSKD offers a 1.33% absolute F1 improvement over baseline distillation (0.5%), while EE reduces inference time by approximately 21% with a 1.3 point drop in F1. Evaluated on the TVSum dataset, our best model PaLI Gemma2 3B + MSKD achieves an F1 score of 61.1, competing the performance of significantly larger models, all while maintaining a lower computational footprint. We publicly release our code and processed dataset to support further research. 

**Abstract (ZH)**: 我们介绍了DEEVISum（一种用于视频摘要的轻量级、高效且可扩展的多模态知识蒸馏和早期退出视觉语言模型） 

---
# Solving Copyright Infringement on Short Video Platforms: Novel Datasets and an Audio Restoration Deep Learning Pipeline 

**Title (ZH)**: 解决短视频平台上的版权侵权问题：新型数据集及音频恢复深度学习pipeline 

**Authors**: Minwoo Oh, Minsu Park, Eunil Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.21772)  

**Abstract**: Short video platforms like YouTube Shorts and TikTok face significant copyright compliance challenges, as infringers frequently embed arbitrary background music (BGM) to obscure original soundtracks (OST) and evade content originality detection. To tackle this issue, we propose a novel pipeline that integrates Music Source Separation (MSS) and cross-modal video-music retrieval (CMVMR). Our approach effectively separates arbitrary BGM from the original OST, enabling the restoration of authentic video audio tracks. To support this work, we introduce two domain-specific datasets: OASD-20K for audio separation and OSVAR-160 for pipeline evaluation. OASD-20K contains 20,000 audio clips featuring mixed BGM and OST pairs, while OSVAR160 is a unique benchmark dataset comprising 1,121 video and mixed-audio pairs, specifically designed for short video restoration tasks. Experimental results demonstrate that our pipeline not only removes arbitrary BGM with high accuracy but also restores OSTs, ensuring content integrity. This approach provides an ethical and scalable solution to copyright challenges in user-generated content on short video platforms. 

**Abstract (ZH)**: 短视频平台如YouTube Shorts和TikTok面临显著的版权合规挑战，侵权者经常嵌入任意背景音乐(BGM)以遮盖原创音轨(OST)并规避内容原创性检测。为解决这一问题，我们提出了一种将音乐源分离(MSS)和跨模态视频-音乐检索(CMVMR)结合的新 Pipeline。该方法有效分离任意BGM与原始OST，使视频音频轨道的恢复成为可能。为了支持这项工作，我们引入了两个领域特定的数据集：OASD-20K用于音频分离和OSVAR-160用于Pipeline评估。OASD-20K包含20,000个混合BGM和OST的音频片段，而OSVAR-160是一个专门设计用于短视频恢复任务的独特基准数据集，包含1,121个视频和混合音频对。实验结果表明，我们的Pipeline不仅以高精度移除任意BGM，还恢复了OST，确保了内容完整性。该方法为解决短视频平台用户生成内容中的版权挑战提供了伦理和可扩展的解决方案。 

---
# Vision Transformers in Precision Agriculture: A Comprehensive Survey 

**Title (ZH)**: 精准农业中视觉变换器的应用：一项全面综述 

**Authors**: Saber Mehdipour, Seyed Abolghasem Mirroshandel, Seyed Amirhossein Tabatabaei  

**Link**: [PDF](https://arxiv.org/pdf/2504.21706)  

**Abstract**: Detecting plant diseases is a crucial aspect of modern agriculture - it plays a key role in maintaining crop health and increasing overall yield. Traditional approaches, though still valuable, often rely on manual inspection or conventional machine learning techniques, both of which face limitations in scalability and accuracy. Recently, Vision Transformers (ViTs) have emerged as a promising alternative, offering benefits such as improved handling of long-range dependencies and better scalability for visual tasks. This survey explores the application of ViTs in precision agriculture, covering tasks from classification to detection and segmentation. We begin by introducing the foundational architecture of ViTs and discuss their transition from Natural Language Processing (NLP) to computer vision. The discussion includes the concept of inductive bias in traditional models like Convolutional Neural Networks (CNNs), and how ViTs mitigate these biases. We provide a comprehensive review of recent literature, focusing on key methodologies, datasets, and performance metrics. The survey also includes a comparative analysis of CNNs and ViTs, with a look at hybrid models and performance enhancements. Technical challenges - such as data requirements, computational demands, and model interpretability - are addressed alongside potential solutions. Finally, we outline potential research directions and technological advancements that could further support the integration of ViTs in real-world agricultural settings. Our goal with this study is to offer practitioners and researchers a deeper understanding of how ViTs are poised to transform smart and precision agriculture. 

**Abstract (ZH)**: Vision Transformers在精准农业中的应用：从分类到检测和分割 

---
# Enhancing Self-Supervised Fine-Grained Video Object Tracking with Dynamic Memory Prediction 

**Title (ZH)**: 基于动态记忆预测的自监督细粒度视频目标跟踪增强方法 

**Authors**: Zihan Zhou, Changrui Dai, Aibo Song, Xiaolin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.21692)  

**Abstract**: Successful video analysis relies on accurate recognition of pixels across frames, and frame reconstruction methods based on video correspondence learning are popular due to their efficiency. Existing frame reconstruction methods, while efficient, neglect the value of direct involvement of multiple reference frames for reconstruction and decision-making aspects, especially in complex situations such as occlusion or fast movement. In this paper, we introduce a Dynamic Memory Prediction (DMP) framework that innovatively utilizes multiple reference frames to concisely and directly enhance frame reconstruction. Its core component is a Reference Frame Memory Engine that dynamically selects frames based on object pixel features to improve tracking accuracy. In addition, a Bidirectional Target Prediction Network is built to utilize multiple reference frames to improve the robustness of the model. Through experiments, our algorithm outperforms the state-of-the-art self-supervised techniques on two fine-grained video object tracking tasks: object segmentation and keypoint tracking. 

**Abstract (ZH)**: 基于多参考帧动态记忆预测的帧重建方法在复杂场景下的视频分析 

---
# ClassWise-CRF: Category-Specific Fusion for Enhanced Semantic Segmentation of Remote Sensing Imagery 

**Title (ZH)**: 类别智融合-CRF：面向遥感影像语义分割的类别特定融合 

**Authors**: Qinfeng Zhu, Yunxi Jiang, Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2504.21491)  

**Abstract**: We propose a result-level category-specific fusion architecture called ClassWise-CRF. This architecture employs a two-stage process: first, it selects expert networks that perform well in specific categories from a pool of candidate networks using a greedy algorithm; second, it integrates the segmentation predictions of these selected networks by adaptively weighting their contributions based on their segmentation performance in each category. Inspired by Conditional Random Field (CRF), the ClassWise-CRF architecture treats the segmentation predictions from multiple networks as confidence vector fields. It leverages segmentation metrics (such as Intersection over Union) from the validation set as priors and employs an exponential weighting strategy to fuse the category-specific confidence scores predicted by each network. This fusion method dynamically adjusts the weights of each network for different categories, achieving category-specific optimization. Building on this, the architecture further optimizes the fused results using unary and pairwise potentials in CRF to ensure spatial consistency and boundary accuracy. To validate the effectiveness of ClassWise-CRF, we conducted experiments on two remote sensing datasets, LoveDA and Vaihingen, using eight classic and advanced semantic segmentation networks. The results show that the ClassWise-CRF architecture significantly improves segmentation performance: on the LoveDA dataset, the mean Intersection over Union (mIoU) metric increased by 1.00% on the validation set and by 0.68% on the test set; on the Vaihingen dataset, the mIoU improved by 0.87% on the validation set and by 0.91% on the test set. These results fully demonstrate the effectiveness and generality of the ClassWise-CRF architecture in semantic segmentation of remote sensing images. The full code is available at this https URL. 

**Abstract (ZH)**: 一种类别级融合架构：ClassWise-CRF 

---
# Revisiting Diffusion Autoencoder Training for Image Reconstruction Quality 

**Title (ZH)**: 重新审视扩散自编码器训练以提高图像重建质量 

**Authors**: Pramook Khungurn, Sukit Seripanitkarn, Phonphrm Thawatdamrongkit, Supasorn Suwajanakorn  

**Link**: [PDF](https://arxiv.org/pdf/2504.21368)  

**Abstract**: Diffusion autoencoders (DAEs) are typically formulated as a noise prediction model and trained with a linear-$\beta$ noise schedule that spends much of its sampling steps at high noise levels. Because high noise levels are associated with recovering large-scale image structures and low noise levels with recovering details, this configuration can result in low-quality and blurry images. However, it should be possible to improve details while spending fewer steps recovering structures because the latent code should already contain structural information. Based on this insight, we propose a new DAE training method that improves the quality of reconstructed images. We divide training into two phases. In the first phase, the DAE is trained as a vanilla autoencoder by always setting the noise level to the highest, forcing the encoder and decoder to populate the latent code with structural information. In the second phase, we incorporate a noise schedule that spends more time in the low-noise region, allowing the DAE to learn how to perfect the details. Our method results in images that have accurate high-level structures and low-level details while still preserving useful properties of the latent codes. 

**Abstract (ZH)**: 扩散自编码器（DAEs）通常被公式化为一个噪声预测模型，并使用线性-$\beta$噪声调度进行训练，其中大部分采样步骤处于高噪声水平。由于高噪声水平与恢复大规模图像结构相关，而低噪声水平与恢复细节相关，这种配置可能导致低质量和模糊的图像。然而，应该有可能在花费更少步骤恢复结构的同时改进细节，因为潜在代码中应包含结构信息。基于这一洞察，我们提出了一种新的DAE训练方法，以提高重建图像的质量。我们将训练分为两个阶段。在第一阶段，DAE作为普通的自编码器进行训练，始终将噪声水平设置为最高，迫使编码器和解码器向潜在代码填充结构信息。在第二阶段，我们引入一种噪声调度，使其在低噪声区域花费更多时间，允许DAE学习如何完美细节。我们的方法生成的图像具有准确的高层结构和低层细节，同时保留潜在代码的有用属性。 

---
# Vision-Language Model-Based Semantic-Guided Imaging Biomarker for Early Lung Cancer Detection 

**Title (ZH)**: 基于视觉-语言模型的语义引导影像生物标志物在早期肺癌检测中的应用 

**Authors**: Luoting Zhuang, Seyed Mohammad Hossein Tabatabaei, Ramin Salehi-Rad, Linh M. Tran, Denise R. Aberle, Ashley E. Prosper, William Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21344)  

**Abstract**: Objective: A number of machine learning models have utilized semantic features, deep features, or both to assess lung nodule malignancy. However, their reliance on manual annotation during inference, limited interpretability, and sensitivity to imaging variations hinder their application in real-world clinical settings. Thus, this research aims to integrate semantic features derived from radiologists' assessments of nodules, allowing the model to learn clinically relevant, robust, and explainable features for predicting lung cancer. Methods: We obtained 938 low-dose CT scans from the National Lung Screening Trial with 1,246 nodules and semantic features. The Lung Image Database Consortium dataset contains 1,018 CT scans, with 2,625 lesions annotated for nodule characteristics. Three external datasets were obtained from UCLA Health, the LUNGx Challenge, and the Duke Lung Cancer Screening. We finetuned a pretrained Contrastive Language-Image Pretraining model with a parameter-efficient fine-tuning approach to align imaging and semantic features and predict the one-year lung cancer diagnosis. Results: We evaluated the performance of the one-year diagnosis of lung cancer with AUROC and AUPRC and compared it to three state-of-the-art models. Our model demonstrated an AUROC of 0.90 and AUPRC of 0.78, outperforming baseline state-of-the-art models on external datasets. Using CLIP, we also obtained predictions on semantic features, such as nodule margin (AUROC: 0.81), nodule consistency (0.81), and pleural attachment (0.84), that can be used to explain model predictions. Conclusion: Our approach accurately classifies lung nodules as benign or malignant, providing explainable outputs, aiding clinicians in comprehending the underlying meaning of model predictions. This approach also prevents the model from learning shortcuts and generalizes across clinical settings. 

**Abstract (ZH)**: 目标：许多机器学习模型利用语义特征、深度特征或两者来评估肺结节的恶性程度。然而，这些模型在推理过程中依赖于手动注释、解释能力有限以及对成像变异的敏感性限制了其在临床场景中的应用。因此，本研究旨在结合放射科医生对结节的评估产生的语义特征，使模型能够学习与临床相关、稳健且可解释的特征，以预测肺癌。方法：我们获得了来自国家肺癌筛查试验的938例低剂量CT扫描，包含1,246个结节及其语义特征。Lung Image Database Consortium数据集包含1,018例CT扫描，其中2,625个病灶标注了结节特征。我们还获得了来自UCLA Health、LUNGx挑战赛和杜克肺癌筛查的三个外部数据集。我们使用参数高效微调方法对预训练的对比语言-图像预训练模型进行微调，以对齐影像和语义特征并预测一年后的肺癌诊断结果。结果：我们使用AUROC和AUPRC评估了一年肺癌诊断的性能，并将其与三种最新模型进行了比较。我们的模型的AUROC为0.90，AUPRC为0.78，在外部数据集上优于基线最新模型。使用CLIP，我们还获得了结节边缘（AUROC: 0.81）、结节一致性（0.81）和胸膜固定（0.84）等语义特征的预测，这些特征可用于解释模型预测。结论：我们的方法准确地将肺结节分类为良性或恶性，提供了可解释的输出，帮助临床医生理解模型预测的含义。此外，这种方法还能防止模型学习捷径，并在不同临床场景中泛化。 

---
# Geolocating Earth Imagery from ISS: Integrating Machine Learning with Astronaut Photography for Enhanced Geographic Mapping 

**Title (ZH)**: 基于国际空间站的地球成像地理定位：结合宇航员摄影与机器学习的地理制图增强方法 

**Authors**: Vedika Srivastava, Hemant Kumar Singh, Jaisal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2504.21194)  

**Abstract**: This paper presents a novel approach to geolocating images captured from the International Space Station (ISS) using advanced machine learning algorithms. Despite having precise ISS coordinates, the specific Earth locations depicted in astronaut-taken photographs often remain unidentified. Our research addresses this gap by employing three distinct image processing pipelines: a Neural Network based approach, a SIFT based method, and GPT-4 model. Each pipeline is tailored to process high-resolution ISS imagery, identifying both natural and man-made geographical features. Through extensive evaluation on a diverse dataset of over 140 ISS images, our methods demonstrate significant promise in automated geolocation with varied levels of success. The NN approach showed a high success rate in accurately matching geographical features, while the SIFT pipeline excelled in processing zoomed-in images. GPT-4 model provided enriched geographical descriptions alongside location predictions. This research contributes to the fields of remote sensing and Earth observation by enhancing the accuracy and efficiency of geolocating space-based imagery, thereby aiding environmental monitoring and global mapping efforts. 

**Abstract (ZH)**: 本文提出了一种使用高级机器学习算法将国际空间站（ISS）拍摄的图像进行地理定位的新方法。尽管拥有精确的ISS坐标，宇航员拍摄的图片中展示的具体地球位置往往仍未被识别。我们的研究通过采用三种不同的图像处理管道填补了这一空白：基于神经网络的方法、基于SIFT的方法以及GPT-4模型。每种管道都针对高分辨率的ISS图像进行了定制，以识别自然和人造的地理特征。通过对包含超过140张ISS图像的多样数据集进行广泛的评估，我们的方法显示出在自动地理定位方面有显著的前景，且具有不同的成功率。基于神经网络的方法在准确匹配地理特征方面表现出较高的成功率，而基于SIFT的管道在处理放大图像方面表现出色。GPT-4模型则提供了地理位置预测及丰富的地理描述。本研究为遥感和地球观测领域作出了贡献，通过提高基于空间的图像的地理定位的准确性和效率，从而辅助环境监测和全球制图工作。 

---
# Light Weight CNN for classification of Brain Tumors from MRI Images 

**Title (ZH)**: 轻量级CNN在 MRI 图像肿瘤分类中的应用 

**Authors**: Natnael Alemayehu  

**Link**: [PDF](https://arxiv.org/pdf/2504.21188)  

**Abstract**: This study presents a convolutional neural network (CNN)-based approach for the multi-class classification of brain tumors using magnetic resonance imaging (MRI) scans. We utilize a publicly available dataset containing MRI images categorized into four classes: glioma, meningioma, pituitary tumor, and no tumor. Our primary objective is to build a light weight deep learning model that can automatically classify brain tumor types with high accuracy. To achieve this goal, we incorporate image preprocessing steps, including normalization, data augmentation, and a cropping technique designed to reduce background noise and emphasize relevant regions. The CNN architecture is optimized through hyperparameter tuning using Keras Tuner, enabling systematic exploration of network parameters. To ensure reliable evaluation, we apply 5-fold cross-validation, where each hyperparameter configuration is evaluated across multiple data splits to mitigate overfitting. Experimental results demonstrate that the proposed model achieves a classification accuracy of 98.78%, indicating its potential as a diagnostic aid in clinical settings. The proposed method offers a low-complexity yet effective solution for assisting in early brain tumor diagnosis. 

**Abstract (ZH)**: 基于卷积神经网络的磁共振成像多类脑肿瘤分类方法 

---
# Dance Style Recognition Using Laban Movement Analysis 

**Title (ZH)**: 基于劳班运动分析的舞蹈风格识别 

**Authors**: Muhammad Turab, Philippe Colantoni, Damien Muselet, Alain Tremeau  

**Link**: [PDF](https://arxiv.org/pdf/2504.21166)  

**Abstract**: The growing interest in automated movement analysis has presented new challenges in recognition of complex human activities including dance. This study focuses on dance style recognition using features extracted using Laban Movement Analysis. Previous studies for dance style recognition often focus on cross-frame movement analysis, which limits the ability to capture temporal context and dynamic transitions between movements. This gap highlights the need for a method that can add temporal context to LMA features. For this, we introduce a novel pipeline which combines 3D pose estimation, 3D human mesh reconstruction, and floor aware body modeling to effectively extract LMA features. To address the temporal limitation, we propose a sliding window approach that captures movement evolution across time in features. These features are then used to train various machine learning methods for classification, and their explainability explainable AI methods to evaluate the contribution of each feature to classification performance. Our proposed method achieves a highest classification accuracy of 99.18\% which shows that the addition of temporal context significantly improves dance style recognition performance. 

**Abstract (ZH)**: 基于劳班运动分析的舞蹈風格識別：結合時序上下文的新型管道研究 

---
# Transcending Dimensions using Generative AI: Real-Time 3D Model Generation in Augmented Reality 

**Title (ZH)**: 超越维度：使用生成式AI进行实时增强现实中的3D模型生成 

**Authors**: Majid Behravan, Maryam Haghani, Denis Gracanin  

**Link**: [PDF](https://arxiv.org/pdf/2504.21033)  

**Abstract**: Traditional 3D modeling requires technical expertise, specialized software, and time-intensive processes, making it inaccessible for many users. Our research aims to lower these barriers by combining generative AI and augmented reality (AR) into a cohesive system that allows users to easily generate, manipulate, and interact with 3D models in real time, directly within AR environments. Utilizing cutting-edge AI models like Shap-E, we address the complex challenges of transforming 2D images into 3D representations in AR environments. Key challenges such as object isolation, handling intricate backgrounds, and achieving seamless user interaction are tackled through advanced object detection methods, such as Mask R-CNN. Evaluation results from 35 participants reveal an overall System Usability Scale (SUS) score of 69.64, with participants who engaged with AR/VR technologies more frequently rating the system significantly higher, at 80.71. This research is particularly relevant for applications in gaming, education, and AR-based e-commerce, offering intuitive, model creation for users without specialized skills. 

**Abstract (ZH)**: 传统3D建模需要专门的技术知识、专业的软件和耗时的过程，使其对许多用户来说难以触及。我们的研究旨在通过将生成性AI与增强现实(AR)结合，形成一个统一系统，让用户能够轻松地在实时时直接在AR环境中生成、操作和互动3D模型，从而降低这些障碍。利用如Shap-E等前沿AI模型，我们解决了在AR环境中将2D图像转换为3D表示的复杂挑战。通过高级对象检测方法（如Mask R-CNN）来应对诸如物体隔离、处理复杂背景和实现无缝用户交互等关键挑战。从35名参与者的表现来看，系统的总体SUS评分为69.64，而更频繁接触AR/VR技术的参与者对系统的评价更高，达到了80.71。这项研究特别适用于游戏、教育和基于AR的电子商务应用，为用户提供直观的3D模型创建功能，无需专门技能。 

---
