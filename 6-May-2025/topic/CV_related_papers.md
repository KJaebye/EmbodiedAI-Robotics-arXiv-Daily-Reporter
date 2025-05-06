# Point Cloud Recombination: Systematic Real Data Augmentation Using Robotic Targets for LiDAR Perception Validation 

**Title (ZH)**: 点云重组：用于LiDAR感知验证的机器人目标驱动的系统化现实数据增强 

**Authors**: Hubert Padusinski, Christian Steinhauser, Christian Scherl, Julian Gaal, Jacob Langner  

**Link**: [PDF](https://arxiv.org/pdf/2505.02476)  

**Abstract**: The validation of LiDAR-based perception of intelligent mobile systems operating in open-world applications remains a challenge due to the variability of real environmental conditions. Virtual simulations allow the generation of arbitrary scenes under controlled conditions but lack physical sensor characteristics, such as intensity responses or material-dependent effects. In contrast, real-world data offers true sensor realism but provides less control over influencing factors, hindering sufficient validation. Existing approaches address this problem with augmentation of real-world point cloud data by transferring objects between scenes. However, these methods do not consider validation and remain limited in controllability because they rely on empirical data. We solve these limitations by proposing Point Cloud Recombination, which systematically augments captured point cloud scenes by integrating point clouds acquired from physical target objects measured in controlled laboratory environments. Thus enabling the creation of vast amounts and varieties of repeatable, physically accurate test scenes with respect to phenomena-aware occlusions with registered 3D meshes. Using the Ouster OS1-128 Rev7 sensor, we demonstrate the augmentation of real-world urban and rural scenes with humanoid targets featuring varied clothing and poses, for repeatable positioning. We show that the recombined scenes closely match real sensor outputs, enabling targeted testing, scalable failure analysis, and improved system safety. By providing controlled yet sensor-realistic data, our method enables trustworthy conclusions about the limitations of specific sensors in compound with their algorithms, e.g., object detection. 

**Abstract (ZH)**: 基于LiDAR的智能移动系统在开放世界应用中的感知验证因实际环境条件的多变性而面临挑战。虚拟模拟可以在受控条件下生成任意场景，但缺乏物理传感器特性，如强度响应或材料依赖效果。相比之下，真实世界数据提供了真实的传感器现实性，但在控制影响因素方面受到限制，妨碍了充分的验证。现有方法通过在场景之间转移对象来增强现实点云数据，但这些方法不考虑验证，且在可控性方面仍有限制，因为它们依赖于经验数据。我们通过提出点云重新组合解决了这些限制，该方法系统地通过将受控实验室环境中物理目标对象测量获取的点云整合到捕获的点云场景中来增强捕获的点云场景。因此，能够创建大量和多样化的可重复、物理准确的测试场景，考虑到现象感知的遮挡，并与注册的3D网格对齐。使用Ouster OS1-128 Rev7传感器，我们演示了通过引入人形目标（具有不同的服装和姿势）来增强真实世界的城市和农村场景，以实现可重复定位。结果显示，重新组合的场景与实际传感器输出高度一致，从而实现有针对性的测试、可扩展的故障分析和改进的系统安全性。通过提供受控但传感器现实的数据，我们的方法能够就特定传感器与其算法组合的限制性得出可信结论，例如物体检测。 

---
# Estimating Commonsense Scene Composition on Belief Scene Graphs 

**Title (ZH)**: 在信念场景图上估计常识场景组成 

**Authors**: Mario A.V. Saucedo, Vignesh Kottayam Viswanathan, Christoforos Kanellakis, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.02405)  

**Abstract**: This work establishes the concept of commonsense scene composition, with a focus on extending Belief Scene Graphs by estimating the spatial distribution of unseen objects. Specifically, the commonsense scene composition capability refers to the understanding of the spatial relationships among related objects in the scene, which in this article is modeled as a joint probability distribution for all possible locations of the semantic object class. The proposed framework includes two variants of a Correlation Information (CECI) model for learning probability distributions: (i) a baseline approach based on a Graph Convolutional Network, and (ii) a neuro-symbolic extension that integrates a spatial ontology based on Large Language Models (LLMs). Furthermore, this article provides a detailed description of the dataset generation process for such tasks. Finally, the framework has been validated through multiple runs on simulated data, as well as in a real-world indoor environment, demonstrating its ability to spatially interpret scenes across different room types. 

**Abstract (ZH)**: 此研究建立了常识场景组成的概念，并focus于通过估计未见物体的空间分布来扩展信念场景图。具体而言，常识场景组成能力指的是对场景中相关物体的空间关系的理解，本文将其建模为所有可能的语义对象类位置的联合概率分布。提出的框架包括用于学习概率分布的Correlation Information (CECI)模型的两种变体：（i）基于图卷积网络的基线方法，以及（ii）结合大语言模型（LLMs）的空间本体的神经符号扩展。此外，本文详细描述了此类任务的数据集生成过程。最后，该框架通过在模拟数据以及真实室内环境中的多次运行得到验证，展示了其在不同房间类型中对场景进行空间解释的能力。 

---
# Enhancing Lidar Point Cloud Sampling via Colorization and Super-Resolution of Lidar Imagery 

**Title (ZH)**: 通过激光雷达图像着色和超分辨率增强激光雷达点云采样 

**Authors**: Sier Ha, Honghao Du, Xianjia Yu, Tomi Westerlund  

**Link**: [PDF](https://arxiv.org/pdf/2505.02049)  

**Abstract**: Recent advancements in lidar technology have led to improved point cloud resolution as well as the generation of 360 degrees, low-resolution images by encoding depth, reflectivity, or near-infrared light within each pixel. These images enable the application of deep learning (DL) approaches, originally developed for RGB images from cameras to lidar-only systems, eliminating other efforts, such as lidar-camera calibration. Compared with conventional RGB images, lidar imagery demonstrates greater robustness in adverse environmental conditions, such as low light and foggy weather. Moreover, the imaging capability addresses the challenges in environments where the geometric information in point clouds may be degraded, such as long corridors, and dense point clouds may be misleading, potentially leading to drift errors.
Therefore, this paper proposes a novel framework that leverages DL-based colorization and super-resolution techniques on lidar imagery to extract reliable samples from lidar point clouds for odometry estimation. The enhanced lidar images, enriched with additional information, facilitate improved keypoint detection, which is subsequently employed for more effective point cloud downsampling. The proposed method enhances point cloud registration accuracy and mitigates mismatches arising from insufficient geometric information or misleading extra points. Experimental results indicate that our approach surpasses previous methods, achieving lower translation and rotation errors while using fewer points. 

**Abstract (ZH)**: 最近lidar技术的进步提高了点云分辨率，并可通过在每个像素中编码深度、反射率或近红外光生成360度低分辨率图像。这些图像使得能够将原本针对相机RGB图像开发的深度学习方法应用于纯lidar系统，从而避免了其他努力，如lidar-相机校准。与传统的RGB图像相比，lidar图像在低光和雾天等恶劣环境条件下表现出更大的鲁棒性。此外，成像能力解决了点云几何信息在长走廊等环境可能降级以及密集点云可能导致错误引导的问题，从而可能引起漂移误差。
因此，本文提出了一种新的框架，利用基于DL的颜色化和超分辨率技术处理lidar图像，从lidar点云中提取可靠的样本用于里程计估计。增强的lidar图像富含额外信息，促进了更有效的关键点检测，进而用于点云下采样。所提出的方法增强了点云配准精度，并减轻了由于几何信息不足或误导性额外点而导致的匹配错误。实验结果表明，我们的方法超过先前的方法，在使用较少点的情况下实现了更低的平移和旋转误差。 

---
# Waymo Driverless Car Data Analysis and Driving Modeling using CNN and LSTM 

**Title (ZH)**: 使用CNN和LSTM进行Waymo无人驾驶汽车数据分析与驾驶建模 

**Authors**: Aashish Kumar Misraa, Naman Jain, Saurav Singh Dhakad  

**Link**: [PDF](https://arxiv.org/pdf/2505.01446)  

**Abstract**: Self driving cars has been the biggest innovation in the automotive industry, but to achieve human level accuracy or near human level accuracy is the biggest challenge that research scientists are facing today. Unlike humans autonomous vehicles do not work on instincts rather they make a decision based on the training data that has been fed to them using machine learning models using which they can make decisions in different conditions they face in the real world. With the advancements in machine learning especially deep learning the self driving car research skyrocketed. In this project we have presented multiple ways to predict acceleration of the autonomous vehicle using Waymo's open dataset. Our main approach was to using CNN to mimic human action and LSTM to treat this as a time series problem. 

**Abstract (ZH)**: 自动驾驶汽车一直是汽车行业的最大创新，但要实现或接近人类级别的准确性是研究人员今天面临的最大挑战。与人类不同，自动驾驶车辆不是基于本能作出决策，而是根据机器学习模型提供的训练数据，在面对现实世界中的各种情况时作出决策。随着机器学习尤其是深度学习的进步，自动驾驶汽车的研究取得了迅速发展。在本项目中，我们使用Waymo的开放数据集展示了多种预测自动驾驶车辆加速度的方法。我们的主要方法是使用CNN来模拟人类行为，并使用LSTM将此问题视为时间序列问题。 

---
# Corr2Distrib: Making Ambiguous Correspondences an Ally to Predict Reliable 6D Pose Distributions 

**Title (ZH)**: Corr2Distrib: 将模糊对应关系转化为预测可靠6D姿态分布的助力 

**Authors**: Asma Brazi, Boris Meden, Fabrice Mayran de Chamisso, Steve Bourgeois, Vincent Lepetit  

**Link**: [PDF](https://arxiv.org/pdf/2505.02501)  

**Abstract**: We introduce Corr2Distrib, the first correspondence-based method which estimates a 6D camera pose distribution from an RGB image, explaining the observations. Indeed, symmetries and occlusions introduce visual ambiguities, leading to multiple valid poses. While a few recent methods tackle this problem, they do not rely on local correspondences which, according to the BOP Challenge, are currently the most effective way to estimate a single 6DoF pose solution. Using correspondences to estimate a pose distribution is not straightforward, since ambiguous correspondences induced by visual ambiguities drastically decrease the performance of PnP. With Corr2Distrib, we turn these ambiguities into an advantage to recover all valid poses. Corr2Distrib first learns a symmetry-aware representation for each 3D point on the object's surface, characterized by a descriptor and a local frame. This representation enables the generation of 3DoF rotation hypotheses from single 2D-3D correspondences. Next, we refine these hypotheses into a 6DoF pose distribution using PnP and pose scoring. Our experimental evaluations on complex non-synthetic scenes show that Corr2Distrib outperforms state-of-the-art solutions for both pose distribution estimation and single pose estimation from an RGB image, demonstrating the potential of correspondences-based approaches. 

**Abstract (ZH)**: Corr2Distrib：基于对应关系的6D相机位姿分布估计方法 

---
# Timing Is Everything: Finding the Optimal Fusion Points in Multimodal Medical Imaging 

**Title (ZH)**: Timing Is Everything: 寻找多模态医学成像中的最优融合点 

**Authors**: Valerio Guarrasi, Klara Mogensen, Sara Tassinari, Sara Qvarlander, Paolo Soda  

**Link**: [PDF](https://arxiv.org/pdf/2505.02467)  

**Abstract**: Multimodal deep learning harnesses diverse imaging modalities, such as MRI sequences, to enhance diagnostic accuracy in medical imaging. A key challenge is determining the optimal timing for integrating these modalities-specifically, identifying the network layers where fusion modules should be inserted. Current approaches often rely on manual tuning or exhaustive search, which are computationally expensive without any guarantee of converging to optimal results. We propose a sequential forward search algorithm that incrementally activates and evaluates candidate fusion modules at different layers of a multimodal network. At each step, the algorithm retrains from previously learned weights and compares validation loss to identify the best-performing configuration. This process systematically reduces the search space, enabling efficient identification of the optimal fusion timing without exhaustively testing all possible module placements. The approach is validated on two multimodal MRI datasets, each addressing different classification tasks. Our algorithm consistently identified configurations that outperformed unimodal baselines, late fusion, and a brute-force ensemble of all potential fusion placements. These architectures demonstrated superior accuracy, F-score, and specificity while maintaining competitive or improved AUC values. Furthermore, the sequential nature of the search significantly reduced computational overhead, making the optimization process more practical. By systematically determining the optimal timing to fuse imaging modalities, our method advances multimodal deep learning for medical imaging. It provides an efficient and robust framework for fusion optimization, paving the way for improved clinical decision-making and more adaptable, scalable architectures in medical AI applications. 

**Abstract (ZH)**: 多模态深度学习结合多种成像模态，如MRI序列，以提高医学影像诊断的准确性。一个关键挑战是在网络中确定融合模块的最佳插入层。当前方法通常依赖手动调优或 exhaustive 搜索，这在没有保证收敛到最优解的情况下计算成本高昂。我们提出了一种逐步前向搜索算法，该算法逐步激活并评估不同层的候选融合模块。在每一步中，算法从已学习的权重重新训练，并比较验证损失以识别性能最佳的配置。该过程系统地减少了搜索空间，从而可以在不全面测试所有可能模块放置的情况下有效地确定最优融合时间。该方法在两个多模态MRI数据集上进行了验证，每个数据集解决了不同的分类任务。我们的算法一致地识别出优于单模态基准、后期融合以及所有潜在融合放置的暴力组合的配置。这些架构在保持或提高AUC值的同时，展示了更优的准确性、F分数和特异性。此外，搜索的顺序性质显著减少了计算开销，使优化过程更实用。通过系统地确定融合影像模态的最佳时间，我们的方法推进了医学影像中的多模态深度学习。它提供了一种高效且稳健的融合优化框架，为改进临床决策并促进在医学AI应用中更适应性强、可扩展的架构铺平了道路。 

---
# Diagnostic Uncertainty in Pneumonia Detection using CNN MobileNetV2 and CNN from Scratch 

**Title (ZH)**: 使用CNN MobileNetV2和从零开始的CNN在肺炎检测中的诊断不确定性 

**Authors**: Kennard Norbert Sudiardjo, Islam Nur Alam, Wilson Wijaya, Lili Ayu Wulandhari  

**Link**: [PDF](https://arxiv.org/pdf/2505.02396)  

**Abstract**: Pneumonia Diagnosis, though it is crucial for an effective treatment, it can be hampered by uncertainty. This uncertainty starts to arise due to some factors like atypical presentations, limitations of diagnostic tools such as chest X-rays, and the presence of co-existing respiratory conditions. This research proposes one of the supervised learning methods, CNN. Using MobileNetV2 as the pre-trained one with ResNet101V2 architecture and using Keras API as the built from scratch model, for identifying lung diseases especially pneumonia. The datasets used in this research were obtained from the website through Kaggle. The result shows that by implementing CNN MobileNetV2 and CNN from scratch the result is promising. While validating data, MobileNetV2 performs with stability and minimal overfitting, while the training accuracy increased to 84.87% later it slightly decreased to 78.95%, with increasing validation loss from 0.499 to 0.6345. Nonetheless, MobileNetV2 is more stable. Although it takes more time to train each epoch. Meanwhile, after the 10th epoch, the Scratch model displayed more instability and overfitting despite having higher validation accuracy, training accuracy decreased significantly to 78.12% and the validation loss increased from 0.5698 to 1.1809. With these results, ResNet101V2 offers stability, and the Scratch model offers high accuracy. 

**Abstract (ZH)**: 肺炎诊断：尽管对于有效治疗至关重要，但可能会受到不确定性的影响 

---
# SuperEdit: Rectifying and Facilitating Supervision for Instruction-Based Image Editing 

**Title (ZH)**: SuperEdit: 修正并促进基于指令的图像编辑监督 

**Authors**: Ming Li, Xin Gu, Fan Chen, Xiaoying Xing, Longyin Wen, Chen Chen, Sijie Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.02370)  

**Abstract**: Due to the challenges of manually collecting accurate editing data, existing datasets are typically constructed using various automated methods, leading to noisy supervision signals caused by the mismatch between editing instructions and original-edited image pairs. Recent efforts attempt to improve editing models through generating higher-quality edited images, pre-training on recognition tasks, or introducing vision-language models (VLMs) but fail to resolve this fundamental issue. In this paper, we offer a novel solution by constructing more effective editing instructions for given image pairs. This includes rectifying the editing instructions to better align with the original-edited image pairs and using contrastive editing instructions to further enhance their effectiveness. Specifically, we find that editing models exhibit specific generation attributes at different inference steps, independent of the text. Based on these prior attributes, we define a unified guide for VLMs to rectify editing instructions. However, there are some challenging editing scenarios that cannot be resolved solely with rectified instructions. To this end, we further construct contrastive supervision signals with positive and negative instructions and introduce them into the model training using triplet loss, thereby further facilitating supervision effectiveness. Our method does not require the VLM modules or pre-training tasks used in previous work, offering a more direct and efficient way to provide better supervision signals, and providing a novel, simple, and effective solution for instruction-based image editing. Results on multiple benchmarks demonstrate that our method significantly outperforms existing approaches. Compared with previous SOTA SmartEdit, we achieve 9.19% improvements on the Real-Edit benchmark with 30x less training data and 13x smaller model size. 

**Abstract (ZH)**: 基于新型编辑指令的有效监督信号构建方法 

---
# Enhancing AI Face Realism: Cost-Efficient Quality Improvement in Distilled Diffusion Models with a Fully Synthetic Dataset 

**Title (ZH)**: 增强AI人脸真实性：使用完全合成数据集在蒸馏扩散模型中的成本效益质量提升 

**Authors**: Jakub Wąsala, Bartłomiej Wrzalski, Kornelia Noculak, Yuliia Tarasenko, Oliwer Krupa, Jan Kocoń, Grzegorz Chodak  

**Link**: [PDF](https://arxiv.org/pdf/2505.02255)  

**Abstract**: This study presents a novel approach to enhance the cost-to-quality ratio of image generation with diffusion models. We hypothesize that differences between distilled (e.g. FLUX.1-schnell) and baseline (e.g. FLUX.1-dev) models are consistent and, therefore, learnable within a specialized domain, like portrait generation. We generate a synthetic paired dataset and train a fast image-to-image translation head. Using two sets of low- and high-quality synthetic images, our model is trained to refine the output of a distilled generator (e.g., FLUX.1-schnell) to a level comparable to a baseline model like FLUX.1-dev, which is more computationally intensive. Our results show that the pipeline, which combines a distilled version of a large generative model with our enhancement layer, delivers similar photorealistic portraits to the baseline version with up to an 82% decrease in computational cost compared to FLUX.1-dev. This study demonstrates the potential for improving the efficiency of AI solutions involving large-scale image generation. 

**Abstract (ZH)**: 本研究提出了一种新的方法，以提高使用扩散模型生成图像的成本与质量比。我们假设萃取（例如，FLUX.1-schnell）和基线（例如，FLUX.1-dev）模型之间的差异是稳定可学的，因此可以在专门领域（如肖像生成）中学习。我们生成了一个合成配对数据集，并训练了一个快速图像到图像转换头部。使用低质量和高质量的合成图像两组，我们的模型被训练成将一个萃取生成器（例如，FLUX.1-schnell）的输出优化到与基线模型（如FLUX.1-dev）相当的水平，而FLUX.1-dev更具计算强度。结果显示，结合一个大型生成模型的萃取版本与我们增强层的管道，与基线版本相比，在计算成本上最多可减少82%，生成类似的逼真肖像。本研究展示了在大规模图像生成中提高人工智能解决方案效率的潜力。 

---
# DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization 

**Title (ZH)**: DualReal: 自适应联合训练在视频个性化中实现无损身份-运动融合 

**Authors**: Wenchuan Wang, Mengqi Huang, Yijing Tu, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.02192)  

**Abstract**: Customized text-to-video generation with pre-trained large-scale models has recently garnered significant attention through focusing on identity and motion consistency. Existing works typically follow the isolated customized paradigm, where the subject identity or motion dynamics are customized exclusively. However, this paradigm completely ignores the intrinsic mutual constraints and synergistic interdependencies between identity and motion, resulting in identity-motion conflicts throughout the generation process that systematically degrades. To address this, we introduce DualReal, a novel framework that, employs adaptive joint training to collaboratively construct interdependencies between dimensions. Specifically, DualReal is composed of two units: (1) Dual-aware Adaptation dynamically selects a training phase (i.e., identity or motion), learns the current information guided by the frozen dimension prior, and employs a regularization strategy to avoid knowledge leakage; (2) StageBlender Controller leverages the denoising stages and Diffusion Transformer depths to guide different dimensions with adaptive granularity, avoiding conflicts at various stages and ultimately achieving lossless fusion of identity and motion patterns. We constructed a more comprehensive benchmark than existing methods. The experimental results show that DualReal improves CLIP-I and DINO-I metrics by 21.7% and 31.8% on average, and achieves top performance on nearly all motion quality metrics. 

**Abstract (ZH)**: 定制化文本到视频生成通过预先训练的大规模模型聚焦身份和运动一致性，近年来引起了显著关注。为解决现有孤立定制 paradigm完全忽视身份和运动之间的内在相互约束和协同依赖性，从而导致生成过程中身份-运动冲突，我们引入了 DualReal，一个新颖的框架，采用自适应联合训练协作构建不同维度之间的依赖关系。DualReal 包含两个模块：（1）Dual-aware Adaptation 动态选择训练阶段（即身份或运动），根据冻结维度的先验学习当前信息，并采用正则化策略避免知识泄漏；（2）StageBlender Controller 利用去噪阶段和扩散变换器的深度，以自适应粒度指导不同维度，避免各阶段的冲突，最终实现身份和运动模式的无损融合。我们构建了一个比现有方法更全面的基准。实验结果表明，DualReal 在平均上分别将 CLIP-I 和 DINO-I 指标提高了 21.7% 和 31.8%，并在几乎所有运动质量指标上达到顶尖性能。 

---
# Benchmarking Feature Upsampling Methods for Vision Foundation Models using Interactive Segmentation 

**Title (ZH)**: 基于互动分割 benchmarking 视觉基础模型的特征上采样方法 

**Authors**: Volodymyr Havrylov, Haiwen Huang, Dan Zhang, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2505.02075)  

**Abstract**: Vision Foundation Models (VFMs) are large-scale, pre-trained models that serve as general-purpose backbones for various computer vision tasks. As VFMs' popularity grows, there is an increasing interest in understanding their effectiveness for dense prediction tasks. However, VFMs typically produce low-resolution features, limiting their direct applicability in this context. One way to tackle this limitation is by employing a task-agnostic feature upsampling module that refines VFM features resolution. To assess the effectiveness of this approach, we investigate Interactive Segmentation (IS) as a novel benchmark for evaluating feature upsampling methods on VFMs. Due to its inherent multimodal input, consisting of an image and a set of user-defined clicks, as well as its dense mask output, IS creates a challenging environment that demands comprehensive visual scene understanding. Our benchmarking experiments show that selecting appropriate upsampling strategies significantly improves VFM features quality. The code is released at this https URL 

**Abstract (ZH)**: Vision Foundation Models (VFMs)作为各类计算机视觉任务的一般性骨干，是大规模的预训练模型。随着VFMs的流行，人们越来越关注它们在密集预测任务中的有效性。然而，VFMs通常生成低分辨率特征，限制了它们在此情境下的直接应用。通过采用任务无关的特征上采样模块来提高VFMs特征分辨率，可以解决这一限制。为了评估这种方法的有效性，我们探讨了交互式分割(IS)作为评估VFMs特征上采样方法的新基准。由于其固有的多模态输入，包括图像和用户定义的点击集，以及其密集的掩膜输出，IS创建了一个对全面的视觉场景理解有严格要求的挑战性环境。我们的基准实验表明，选择适当的上采样策略显著提高了VFMs特征的质量。代码发布于此<a href="https://thishttpsurl.com">https://thishttpsurl.com</a>。 

---
# Segment Any RGB-Thermal Model with Language-aided Distillation 

**Title (ZH)**: 基于语言辅助蒸馏的RGB-热图分割模型 

**Authors**: Dong Xing, Xianxun Zhu, Wei Zhou, Qika Lin, Hang Yang, Yuqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01950)  

**Abstract**: The recent Segment Anything Model (SAM) demonstrates strong instance segmentation performance across various downstream tasks. However, SAM is trained solely on RGB data, limiting its direct applicability to RGB-thermal (RGB-T) semantic segmentation. Given that RGB-T provides a robust solution for scene understanding in adverse weather and lighting conditions, such as low light and overexposure, we propose a novel framework, SARTM, which customizes the powerful SAM for RGB-T semantic segmentation. Our key idea is to unleash the potential of SAM while introduce semantic understanding modules for RGB-T data pairs. Specifically, our framework first involves fine tuning the original SAM by adding extra LoRA layers, aiming at preserving SAM's strong generalization and segmentation capabilities for downstream tasks. Secondly, we introduce language information as guidance for training our SARTM. To address cross-modal inconsistencies, we introduce a Cross-Modal Knowledge Distillation(CMKD) module that effectively achieves modality adaptation while maintaining its generalization capabilities. This semantic module enables the minimization of modality gaps and alleviates semantic ambiguity, facilitating the combination of any modality under any visual conditions. Furthermore, we enhance the segmentation performance by adjusting the segmentation head of SAM and incorporating an auxiliary semantic segmentation head, which integrates multi-scale features for effective fusion. Extensive experiments are conducted across three multi-modal RGBT semantic segmentation benchmarks: MFNET, PST900, and FMB. Both quantitative and qualitative results consistently demonstrate that the proposed SARTM significantly outperforms state-of-the-art approaches across a variety of conditions. 

**Abstract (ZH)**: Segment Anything Model 基ETHOD (SARTM) 用于 RGB-热成像 (RGB-T) 语义分割 

---
# Accelerating Volumetric Medical Image Annotation via Short-Long Memory SAM 2 

**Title (ZH)**: 基于短长时记忆SAM加速体模医学图像标注 

**Authors**: Yuwen Chen, Zafer Yildiz, Qihang Li, Yaqian Chen, Haoyu Dong, Hanxue Gu, Nicholas Konz, Maciej A. Mazurowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.01854)  

**Abstract**: Manual annotation of volumetric medical images, such as magnetic resonance imaging (MRI) and computed tomography (CT), is a labor-intensive and time-consuming process. Recent advancements in foundation models for video object segmentation, such as Segment Anything Model 2 (SAM 2), offer a potential opportunity to significantly speed up the annotation process by manually annotating one or a few slices and then propagating target masks across the entire volume. However, the performance of SAM 2 in this context varies. Our experiments show that relying on a single memory bank and attention module is prone to error propagation, particularly at boundary regions where the target is present in the previous slice but absent in the current one. To address this problem, we propose Short-Long Memory SAM 2 (SLM-SAM 2), a novel architecture that integrates distinct short-term and long-term memory banks with separate attention modules to improve segmentation accuracy. We evaluate SLM-SAM 2 on three public datasets covering organs, bones, and muscles across MRI and CT modalities. We show that the proposed method markedly outperforms the default SAM 2, achieving average Dice Similarity Coefficient improvement of 0.14 and 0.11 in the scenarios when 5 volumes and 1 volume are available for the initial adaptation, respectively. SLM-SAM 2 also exhibits stronger resistance to over-propagation, making a notable step toward more accurate automated annotation of medical images for segmentation model development. 

**Abstract (ZH)**: 手动标注体素医学图像（如磁共振成像（MRI）和计算机断层扫描（CT））是一个劳动密集型和耗时的过程。基础模型在视频对象分割方面的最新进展，如Segment Anything Model 2（SAM 2），为通过手动标注一两个切片然后在整个人体体积中传播目标蒙版来显著加速标注过程提供了潜在机会。然而，SAM 2在这一方面的表现不尽相同。我们的实验表明，依赖单一的记忆库和注意力模块容易在目标在前一切片存在而在当前切片不存在的边界区域发生错误传播。为了解决这个问题，我们提出了一种新的架构Short-Long Memory SAM 2（SLM-SAM 2），它结合了独立的短期和长期记忆库以及各自的注意力模块，以提高分割准确性。我们在三个涵盖MRI和CT模态下器官、骨骼和肌肉的公开数据集上评估了SLM-SAM 2。结果显示，所提出的方法显著优于默认的SAM 2，在可用初步适应的5个体积和1个体积的情况下，平均Dice相似度系数分别提升了0.14和0.11。SLM-SAM 2还表现出更强的反过度传播能力，为医学图像的自动化标注和分割模型开发提供了更准确的步骤。 

---
# TEMPURA: Temporal Event Masked Prediction and Understanding for Reasoning in Action 

**Title (ZH)**: TEMPURA: 时间事件掩蔽预测与理解在动作推理中的应用 

**Authors**: Jen-Hao Cheng, Vivian Wang, Huayu Wang, Huapeng Zhou, Yi-Hao Peng, Hou-I Liu, Hsiang-Wei Huang, Kuang-Ming Chen, Cheng-Yen Yang, Wenhao Chai, Yi-Ling Chen, Vibhav Vineet, Qin Cai, Jenq-Neng Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01583)  

**Abstract**: Understanding causal event relationships and achieving fine-grained temporal grounding in videos remain challenging for vision-language models. Existing methods either compress video tokens to reduce temporal resolution, or treat videos as unsegmented streams, which obscures fine-grained event boundaries and limits the modeling of causal dependencies. We propose TEMPURA (Temporal Event Masked Prediction and Understanding for Reasoning in Action), a two-stage training framework that enhances video temporal understanding. TEMPURA first applies masked event prediction reasoning to reconstruct missing events and generate step-by-step causal explanations from dense event annotations, drawing inspiration from effective infilling techniques. TEMPURA then learns to perform video segmentation and dense captioning to decompose videos into non-overlapping events with detailed, timestamp-aligned descriptions. We train TEMPURA on VER, a large-scale dataset curated by us that comprises 1M training instances and 500K videos with temporally aligned event descriptions and structured reasoning steps. Experiments on temporal grounding and highlight detection benchmarks demonstrate that TEMPURA outperforms strong baseline models, confirming that integrating causal reasoning with fine-grained temporal segmentation leads to improved video understanding. 

**Abstract (ZH)**: 基于因果事件关系的理解及视频中细粒度时空定位的研究仍然挑战重重。为了解决这一问题，我们提出了TEMPURA（Temporal Event Masked Prediction and Understanding for Reasoning in Action）框架，这是一种两阶段训练框架，旨在增强视频的时间理解能力。TEMPURA首先通过掩蔽事件预测推理来重构缺失事件并从密集的事件注释中生成逐步因果解释，灵感来自于有效的填充技术。然后，它学习进行视频分割和密集字幕生成，以将视频分解为非重叠事件，并提供详细的时间戳对齐描述。我们使用我们整理的VER大规模数据集对TEMPURA进行训练，该数据集包含100万训练实例和50万段具有时间对齐事件描述和结构化推理步骤的视频。在时间定位和高光检测基准测试上的实验表明，TEMPURA优于强大的基线模型，证明将因果推理与细粒度的时间分割结合使用可以提高视频理解能力。 

---
