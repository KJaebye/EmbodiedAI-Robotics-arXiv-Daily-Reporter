# Research on visual simultaneous localization and mapping technology based on near infrared light 

**Title (ZH)**: 基于近红外光的视觉同时定位与建图技术研究 

**Authors**: Rui Ma, Mengfang Liu, Boliang Li, Xinghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02584)  

**Abstract**: In view of the problems that visual simultaneous localization and mapping (VSLAM) are susceptible to environmental light interference and luminosity inconsistency, the visual simultaneous localization and mapping technology based on near infrared perception (NIR-VSLAM) is proposed. In order to avoid ambient light interference, the near infrared light is innovatively selected as the light source. The luminosity parameter estimation of error energy function, halo factor and exposure time and the light source irradiance correction method are proposed in this paper, which greatly improves the positioning accuracy of Direct Sparse Odometry (DSO). The feasibility of the proposed method in four large scenes is verified, which provides the reference for visual positioning in automatic driving and mobile robot. 

**Abstract (ZH)**: 基于近红外感知的视觉 simultaneous localization and mapping 技术（NIR-VSLAM） 

---
# RGBSQGrasp: Inferring Local Superquadric Primitives from Single RGB Image for Graspability-Aware Bin Picking 

**Title (ZH)**: RGBSQ抓取：从单张RGB图像中推断局部超二次原始几何体以实现考虑抓取性的 bin 选择 

**Authors**: Yifeng Xu, Fan Zhu, Ye Li, Sebastian Ren, Xiaonan Huang, Yuhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02387)  

**Abstract**: Bin picking is a challenging robotic task due to occlusions and physical constraints that limit visual information for object recognition and grasping. Existing approaches often rely on known CAD models or prior object geometries, restricting generalization to novel or unknown objects. Other methods directly regress grasp poses from RGB-D data without object priors, but the inherent noise in depth sensing and the lack of object understanding make grasp synthesis and evaluation more difficult. Superquadrics (SQ) offer a compact, interpretable shape representation that captures the physical and graspability understanding of objects. However, recovering them from limited viewpoints is challenging, as existing methods rely on multiple perspectives for near-complete point cloud reconstruction, limiting their effectiveness in bin-picking. To address these challenges, we propose \textbf{RGBSQGrasp}, a grasping framework that leverages superquadric shape primitives and foundation metric depth estimation models to infer grasp poses from a monocular RGB camera -- eliminating the need for depth sensors. Our framework integrates a universal, cross-platform dataset generation pipeline, a foundation model-based object point cloud estimation module, a global-local superquadric fitting network, and an SQ-guided grasp pose sampling module. By integrating these components, RGBSQGrasp reliably infers grasp poses through geometric reasoning, enhancing grasp stability and adaptability to unseen objects. Real-world robotic experiments demonstrate a 92\% grasp success rate, highlighting the effectiveness of RGBSQGrasp in packed bin-picking environments. 

**Abstract (ZH)**: RGBSQGrasp：基于超二次曲面的单目视觉抓取框架 

---
# Controllable Motion Generation via Diffusion Modal Coupling 

**Title (ZH)**: 可控运动生成 via 推荐模态耦合 

**Authors**: Luobin Wang, Hongzhan Yu, Chenning Yu, Sicun Gao, Henrik Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02353)  

**Abstract**: Diffusion models have recently gained significant attention in robotics due to their ability to generate multi-modal distributions of system states and behaviors. However, a key challenge remains: ensuring precise control over the generated outcomes without compromising realism. This is crucial for applications such as motion planning or trajectory forecasting, where adherence to physical constraints and task-specific objectives is essential. We propose a novel framework that enhances controllability in diffusion models by leveraging multi-modal prior distributions and enforcing strong modal coupling. This allows us to initiate the denoising process directly from distinct prior modes that correspond to different possible system behaviors, ensuring sampling to align with the training distribution. We evaluate our approach on motion prediction using the Waymo dataset and multi-task control in Maze2D environments. Experimental results show that our framework outperforms both guidance-based techniques and conditioned models with unimodal priors, achieving superior fidelity, diversity, and controllability, even in the absence of explicit conditioning. Overall, our approach provides a more reliable and scalable solution for controllable motion generation in robotics. 

**Abstract (ZH)**: 基于扩散模型的鲁棒可控运动生成方法 

---
# Diffusion-Based mmWave Radar Point Cloud Enhancement Driven by Range Images 

**Title (ZH)**: 基于扩散范围图像驱动的毫米波雷达点云增强 

**Authors**: Ruixin Wu, Zihan Li, Jin Wang, Xiangyu Xu, Huan Yu, Zhi Zheng, Kaixiang Huang, Guodong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02300)  

**Abstract**: Millimeter-wave (mmWave) radar has attracted significant attention in robotics and autonomous driving. However, despite the perception stability in harsh environments, the point cloud generated by mmWave radar is relatively sparse while containing significant noise, which limits its further development. Traditional mmWave radar enhancement approaches often struggle to leverage the effectiveness of diffusion models in super-resolution, largely due to the unnatural range-azimuth heatmap (RAH) or bird's eye view (BEV) representation. To overcome this limitation, we propose a novel method that pioneers the application of fusing range images with image diffusion models, achieving accurate and dense mmWave radar point clouds that are similar to LiDAR. Benefitting from the projection that aligns with human observation, the range image representation of mmWave radar is close to natural images, allowing the knowledge from pre-trained image diffusion models to be effectively transferred, significantly improving the overall performance. Extensive evaluations on both public datasets and self-constructed datasets demonstrate that our approach provides substantial improvements, establishing a new state-of-the-art performance in generating truly three-dimensional LiDAR-like point clouds via mmWave radar. 

**Abstract (ZH)**: 毫米波雷达（mmWave雷达）在机器人和自动驾驶领域引起了广泛关注。然而，尽管在恶劣环境中具有感知稳定性，mmWave雷达生成的点云相对稀疏且含有大量噪声，这限制了其进一步发展。传统的mmWave雷达增强方法往往难以充分发挥扩散模型在超分辨率上的有效性，主要原因是不自然的距离-方位热图（RAH）或鸟瞰图（BEV）表示。为克服这一限制，我们提出了一种新颖的方法，首次将距离图像与图像扩散模型融合，实现了类似于LiDAR的精确且密集的mmWave雷达点云。得益于与人类观测相一致的投影，mmWave雷达的距离图像表示接近自然图像，使得预训练的图像扩散模型知识能够有效地迁移，显著提高整体性能。在公共数据集和自构建数据集上的广泛评估表明，我们的方法提供了实质性的改进，建立了通过mmWave雷达生成真正三维LiDAR-like点云的新最先进性能。 

---
# Class-Aware PillarMix: Can Mixed Sample Data Augmentation Enhance 3D Object Detection with Radar Point Clouds? 

**Title (ZH)**: 基于类别的柱状混合增强：混合样本数据增强能否提升雷达点云的3D目标检测？ 

**Authors**: Miao Zhang, Sherif Abdulatif, Benedikt Loesch, Marco Altmann, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02687)  

**Abstract**: Due to the significant effort required for data collection and annotation in 3D perception tasks, mixed sample data augmentation (MSDA) has been widely studied to generate diverse training samples by mixing existing data. Recently, many MSDA techniques have been developed for point clouds, but they mainly target LiDAR data, leaving their application to radar point clouds largely unexplored. In this paper, we examine the feasibility of applying existing MSDA methods to radar point clouds and identify several challenges in adapting these techniques. These obstacles stem from the radar's irregular angular distribution, deviations from a single-sensor polar layout in multi-radar setups, and point sparsity. To address these issues, we propose Class-Aware PillarMix (CAPMix), a novel MSDA approach that applies MixUp at the pillar level in 3D point clouds, guided by class labels. Unlike methods that rely a single mix ratio to the entire sample, CAPMix assigns an independent ratio to each pillar, boosting sample diversity. To account for the density of different classes, we use class-specific distributions: for dense objects (e.g., large vehicles), we skew ratios to favor points from another sample, while for sparse objects (e.g., pedestrians), we sample more points from the original. This class-aware mixing retains critical details and enriches each sample with new information, ultimately generating more diverse training data. Experimental results demonstrate that our method not only significantly boosts performance but also outperforms existing MSDA approaches across two datasets (Bosch Street and K-Radar). We believe that this straightforward yet effective approach will spark further investigation into MSDA techniques for radar data. 

**Abstract (ZH)**: 基于雷达点云的类意识柱混合（CAPMix）：一种用于3D感知任务的混合样本数据增强方法 

---
# Unveiling the Potential of Segment Anything Model 2 for RGB-Thermal Semantic Segmentation with Language Guidance 

**Title (ZH)**: 揭示段 anything 模型2在光照语义分割中的潜力及语言指导作用 

**Authors**: Jiayi Zhao, Fei Teng, Kai Luo, Guoqiang Zhao, Zhiyong Li, Xu Zheng, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02581)  

**Abstract**: The perception capability of robotic systems relies on the richness of the dataset. Although Segment Anything Model 2 (SAM2), trained on large datasets, demonstrates strong perception potential in perception tasks, its inherent training paradigm prevents it from being suitable for RGB-T tasks. To address these challenges, we propose SHIFNet, a novel SAM2-driven Hybrid Interaction Paradigm that unlocks the potential of SAM2 with linguistic guidance for efficient RGB-Thermal perception. Our framework consists of two key components: (1) Semantic-Aware Cross-modal Fusion (SACF) module that dynamically balances modality contributions through text-guided affinity learning, overcoming SAM2's inherent RGB bias; (2) Heterogeneous Prompting Decoder (HPD) that enhances global semantic information through a semantic enhancement module and then combined with category embeddings to amplify cross-modal semantic consistency. With 32.27M trainable parameters, SHIFNet achieves state-of-the-art segmentation performance on public benchmarks, reaching 89.8% on PST900 and 67.8% on FMB, respectively. The framework facilitates the adaptation of pre-trained large models to RGB-T segmentation tasks, effectively mitigating the high costs associated with data collection while endowing robotic systems with comprehensive perception capabilities. The source code will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于SAM2的新型混合交互框架SHIFNet及其在RGB-T感知任务中的应用 

---
# TS-CGNet: Temporal-Spatial Fusion Meets Centerline-Guided Diffusion for BEV Mapping 

**Title (ZH)**: TS-CGNet：时空融合结合中心线引导扩散的BEV映射 

**Authors**: Xinying Hong, Siyu Li, Kang Zeng, Hao Shi, Bomin Peng, Kailun Yang, Zhiyong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02578)  

**Abstract**: Bird's Eye View (BEV) perception technology is crucial for autonomous driving, as it generates top-down 2D maps for environment perception, navigation, and decision-making. Nevertheless, the majority of current BEV map generation studies focusing on visual map generation lack depth-aware reasoning capabilities. They exhibit limited efficacy in managing occlusions and handling complex environments, with a notable decline in perceptual performance under adverse weather conditions or low-light scenarios. Therefore, this paper proposes TS-CGNet, which leverages Temporal-Spatial fusion with Centerline-Guided diffusion. This visual framework, grounded in prior knowledge, is designed for integration into any existing network for building BEV maps. Specifically, this framework is decoupled into three parts: Local mapping system involves the initial generation of semantic maps using purely visual information; The Temporal-Spatial Aligner Module (TSAM) integrates historical information into mapping generation by applying transformation matrices; The Centerline-Guided Diffusion Model (CGDM) is a prediction module based on the diffusion model. CGDM incorporates centerline information through spatial-attention mechanisms to enhance semantic segmentation reconstruction. We construct BEV semantic segmentation maps by our methods on the public nuScenes and the robustness benchmarks under various corruptions. Our method improves 1.90%, 1.73%, and 2.87% for perceived ranges of 60x30m, 120x60m, and 240x60m in the task of BEV HD mapping. TS-CGNet attains an improvement of 1.92% for perceived ranges of 100x100m in the task of BEV semantic mapping. Moreover, TS-CGNet achieves an average improvement of 2.92% in detection accuracy under varying weather conditions and sensor interferences in the perception range of 240x60m. The source code will be publicly available at this https URL. 

**Abstract (ZH)**: BEV感知技术对自主驾驶至关重要：TS-CGNet-temporal-spatial融合与中心线引导扩散方法在BEV地图构建中的应用 

---
# Label-Efficient LiDAR Panoptic Segmentation 

**Title (ZH)**: 标签高效激光雷达全景分割 

**Authors**: Ahmet Selim Çanakçı, Niclas Vödisch, Kürsat Petek, Wolfram Burgard, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2503.02372)  

**Abstract**: A main bottleneck of learning-based robotic scene understanding methods is the heavy reliance on extensive annotated training data, which often limits their generalization ability. In LiDAR panoptic segmentation, this challenge becomes even more pronounced due to the need to simultaneously address both semantic and instance segmentation from complex, high-dimensional point cloud data. In this work, we address the challenge of LiDAR panoptic segmentation with very few labeled samples by leveraging recent advances in label-efficient vision panoptic segmentation. To this end, we propose a novel method, Limited-Label LiDAR Panoptic Segmentation (L3PS), which requires only a minimal amount of labeled data. Our approach first utilizes a label-efficient 2D network to generate panoptic pseudo-labels from a small set of annotated images, which are subsequently projected onto point clouds. We then introduce a novel 3D refinement module that capitalizes on the geometric properties of point clouds. By incorporating clustering techniques, sequential scan accumulation, and ground point separation, this module significantly enhances the accuracy of the pseudo-labels, improving segmentation quality by up to +10.6 PQ and +7.9 mIoU. We demonstrate that these refined pseudo-labels can be used to effectively train off-the-shelf LiDAR segmentation networks. Through extensive experiments, we show that L3PS not only outperforms existing methods but also substantially reduces the annotation burden. We release the code of our work at this https URL. 

**Abstract (ZH)**: 基于学习的机器人场景理解方法的主要瓶颈是对大量标注训练数据的高依赖性，这往往限制了其泛化能力。在激光雷达全景分割中，这一挑战更为突出，因为需要同时从复杂的高维度点云数据中解决语义和实例分割问题。在本文中，我们通过利用最近在标签高效视觉全景分割方面的进展，解决了在少量标注样本情况下激光雷达全景分割的挑战。为此，我们提出了一种新型方法——有限标签激光雷达全景分割（L3PS），该方法仅需少量标注数据。我们的方法首先利用一种标签高效的二维网络从少量标注图像中生成全景伪标签，并将其投影到点云上。然后，我们引入了一种新颖的3D细化模块，该模块充分利用了点云的几何特性。通过结合聚类技术、顺序扫描积累和地面点分离，该模块显著提高了伪标签的准确性，分割质量最高可提高10.6 PQ和7.9 mIoU。我们展示了这些细化后的伪标签可以有效训练现成的激光雷达分割网络。通过大量实验，我们证明L3PS不仅优于现有方法，还能显著减少标注负担。我们已在此网址发布了我们的代码：this https URL。 

---
# Data Augmentation for NeRFs in the Low Data Limit 

**Title (ZH)**: 低数据限制下用于NeRF的数据增强方法 

**Authors**: Ayush Gaggar, Todd D. Murphey  

**Link**: [PDF](https://arxiv.org/pdf/2503.02092)  

**Abstract**: Current methods based on Neural Radiance Fields fail in the low data limit, particularly when training on incomplete scene data. Prior works augment training data only in next-best-view applications, which lead to hallucinations and model collapse with sparse data. In contrast, we propose adding a set of views during training by rejection sampling from a posterior uncertainty distribution, generated by combining a volumetric uncertainty estimator with spatial coverage. We validate our results on partially observed scenes; on average, our method performs 39.9% better with 87.5% less variability across established scene reconstruction benchmarks, as compared to state of the art baselines. We further demonstrate that augmenting the training set by sampling from any distribution leads to better, more consistent scene reconstruction in sparse environments. This work is foundational for robotic tasks where augmenting a dataset with informative data is critical in resource-constrained, a priori unknown environments. Videos and source code are available at this https URL. 

**Abstract (ZH)**: 基于神经辐射场的方法在数据稀少的情况下失效，特别是在使用不完整场景数据进行训练时。先前工作仅在下一步最佳视图应用中增强训练数据，导致在稀疏数据情况下出现幻觉和模型崩溃。相比之下，我们提议在训练过程中通过从结合体素不确定性估计器与空间覆盖生成的后验不确定性分布中进行拒绝抽样来添加一组视图。我们在部分观测场景上验证了该方法；与现有的基线方法相比，我们的方法在多个场景重建基准测试中平均提高39.9%，且变异性降低87.5%。此外，我们还证明，从任何分布中抽样增强训练集可以在稀疏环境中获得更好的、更一致的场景重建效果。该项工作为在资源受限和先验未知环境中补充具有信息性数据的数据集提供了基础。有关视频和源代码在此处获取。 

---
# A Causal Framework for Aligning Image Quality Metrics and Deep Neural Network Robustness 

**Title (ZH)**: 因果框架下图像质量度量与深度神经网络鲁棒性的对齐 

**Authors**: Nathan Drenkow, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2503.02797)  

**Abstract**: Image quality plays an important role in the performance of deep neural networks (DNNs) and DNNs have been widely shown to exhibit sensitivity to changes in imaging conditions. Large-scale datasets often contain images under a wide range of conditions prompting a need to quantify and understand their underlying quality distribution in order to better characterize DNN performance and robustness. Aligning the sensitivities of image quality metrics and DNNs ensures that estimates of quality can act as proxies for image/dataset difficulty independent of the task models trained/evaluated on the data. Conventional image quality assessment (IQA) seeks to measure and align quality relative to human perceptual judgments, but here we seek a quality measure that is not only sensitive to imaging conditions but also well-aligned with DNN sensitivities. We first ask whether conventional IQA metrics are also informative of DNN performance. In order to answer this question, we reframe IQA from a causal perspective and examine conditions under which quality metrics are predictive of DNN performance. We show theoretically and empirically that current IQA metrics are weak predictors of DNN performance in the context of classification. We then use our causal framework to provide an alternative formulation and a new image quality metric that is more strongly correlated with DNN performance and can act as a prior on performance without training new task models. Our approach provides a means to directly estimate the quality distribution of large-scale image datasets towards characterizing the relationship between dataset composition and DNN performance. 

**Abstract (ZH)**: 图像质量在深度神经网络（DNN）性能中扮演重要角色，并且DNNs已被广泛证明对成像条件的变化表现出敏感性。为了更好地表征DNN性能和鲁棒性，需要量化和理解大规模数据集在不同条件下的内在质量分布。使图像质量度量的敏感性与DNNs相一致，可以确保质量估计能够作为独立于任务模型训练/评估的图像/数据难度的代理指标。传统的图像质量评估（IQA）旨在测量和调整质量与人类感知判断的相对关系，但在这里我们寻求一种不仅对成像条件敏感，而且与DNNs敏感性高度一致的质量度量。我们首先询问传统的IQA度量是否也对DNN性能有信息价值。为了回答这个问题，我们从因果角度重新定义IQA，并研究在哪些条件下质量度量能够预测DNN性能。我们理论和实验证明，当前的IQA度量在分类背景下对DNN性能的预测能力较弱。然后，我们使用我们的因果框架提出一种替代的方法和一个新的图像质量度量，该度量与DNN性能的相关性更强，并且可以在不训练新的任务模型的情况下作为性能的先验。我们的方法提供了一种直接估计大规模图像数据集的质量分布的方法，旨在表征数据集组成与DNN性能之间的关系。 

---
# UAR-NVC: A Unified AutoRegressive Framework for Memory-Efficient Neural Video Compression 

**Title (ZH)**: 统一自回归框架：一种内存高效神经视频压缩方法 

**Authors**: Jia Wang, Xinfeng Zhang, Gai Zhang, Jun Zhu, Lv Tang, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02733)  

**Abstract**: Implicit Neural Representations (INRs) have demonstrated significant potential in video compression by representing videos as neural networks. However, as the number of frames increases, the memory consumption for training and inference increases substantially, posing challenges in resource-constrained scenarios. Inspired by the success of traditional video compression frameworks, which process video frame by frame and can efficiently compress long videos, we adopt this modeling strategy for INRs to decrease memory consumption, while aiming to unify the frameworks from the perspective of timeline-based autoregressive modeling. In this work, we present a novel understanding of INR models from an autoregressive (AR) perspective and introduce a Unified AutoRegressive Framework for memory-efficient Neural Video Compression (UAR-NVC). UAR-NVC integrates timeline-based and INR-based neural video compression under a unified autoregressive paradigm. It partitions videos into several clips and processes each clip using a different INR model instance, leveraging the advantages of both compression frameworks while allowing seamless adaptation to either in form. To further reduce temporal redundancy between clips, we design two modules to optimize the initialization, training, and compression of these model parameters. UAR-NVC supports adjustable latencies by varying the clip length. Extensive experimental results demonstrate that UAR-NVC, with its flexible video clip setting, can adapt to resource-constrained environments and significantly improve performance compared to different baseline models. 

**Abstract (ZH)**: 基于自回归建模的统一记忆高效神经视频压缩框架（UAR-NVC） 

---
# Memory Efficient Continual Learning for Edge-Based Visual Anomaly Detection 

**Title (ZH)**: 基于边缘的视觉异常检测的高效持续学习方法 

**Authors**: Manuel Barusco, Lorenzo D'Antoni, Davide Dalle Pezze, Francesco Borsatti, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2503.02691)  

**Abstract**: Visual Anomaly Detection (VAD) is a critical task in computer vision with numerous real-world applications. However, deploying these models on edge devices presents significant challenges, such as constrained computational and memory resources. Additionally, dynamic data distributions in real-world settings necessitate continuous model adaptation, further complicating deployment under limited resources. To address these challenges, we present a novel investigation into the problem of Continual Learning for Visual Anomaly Detection (CLAD) on edge devices. We evaluate the STFPM approach, given its low memory footprint on edge devices, which demonstrates good performance when combined with the Replay approach. Furthermore, we propose to study the behavior of a recently proposed approach, PaSTe, specifically designed for the edge but not yet explored in the Continual Learning context. Our results show that PaSTe is not only a lighter version of STPFM, but it also achieves superior anomaly detection performance, improving the f1 pixel performance by 10% with the Replay technique. In particular, the structure of PaSTe allows us to test it using a series of Compressed Replay techniques, reducing memory overhead by a maximum of 91.5% compared to the traditional Replay for STFPM. Our study proves the feasibility of deploying VAD models that adapt and learn incrementally on CLAD scenarios on resource-constrained edge devices. 

**Abstract (ZH)**: Visual异常检测持续学习（CLAD）在边缘设备上的研究 

---
# State of play and future directions in industrial computer vision AI standards 

**Title (ZH)**: 工业计算机视觉AI标准的发展现状与未来方向 

**Authors**: Artemis Stefanidou, Panagiotis Radoglou-Grammatikis, Vasileios Argyriou, Panagiotis Sarigiannidis, Iraklis Varlamis, Georgios Th. Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.02675)  

**Abstract**: The recent tremendous advancements in the areas of Artificial Intelligence (AI) and Deep Learning (DL) have also resulted into corresponding remarkable progress in the field of Computer Vision (CV), showcasing robust technological solutions in a wide range of application sectors of high industrial interest (e.g., healthcare, autonomous driving, automation, etc.). Despite the outstanding performance of CV systems in specific domains, their development and exploitation at industrial-scale necessitates, among other, the addressing of requirements related to the reliability, transparency, trustworthiness, security, safety, and robustness of the developed AI models. The latter raises the imperative need for the development of efficient, comprehensive and widely-adopted industrial standards. In this context, this study investigates the current state of play regarding the development of industrial computer vision AI standards, emphasizing on critical aspects, like model interpretability, data quality, and regulatory compliance. In particular, a systematic analysis of launched and currently developing CV standards, proposed by the main international standardization bodies (e.g. ISO/IEC, IEEE, DIN, etc.) is performed. The latter is complemented by a comprehensive discussion on the current challenges and future directions observed in this regularization endeavor. 

**Abstract (ZH)**: 最近在人工智能（AI）和深度学习（DL）领域的显著进展也推动了计算机视觉（CV）领域取得了相应的重大进展，展示了在多个具有高度工业兴趣的应用领域（例如医疗保健、自动驾驶、自动化等）中 robust 的技术解决方案。尽管CV系统在其特定领域表现出色，但其在工业规模上的开发和利用需要解决与 AI 模型的可靠性和透明度、可信性、安全性和鲁棒性等相关要求，这迫切需要开发高效、全面且广泛采用的工业标准。在此背景下，本研究调查了目前工业计算机视觉 AI 标准的发展状况，重点强调了模型可解释性、数据质量和监管合规等关键方面。特别地，对主要国际标准化机构（如 ISO/IEC、IEEE、DIN 等）推出的和正在开发的 CV 标准进行了系统的分析。该分析补充了对该标准化努力中当前挑战和未来方向的全面讨论。 

---
# A dataset-free approach for self-supervised learning of 3D reflectional symmetries 

**Title (ZH)**: 无数据集自监督学习三维反射对称性的方法 

**Authors**: Issac Aguirre, Ivan Sipiran, Gabriel Montañana  

**Link**: [PDF](https://arxiv.org/pdf/2503.02660)  

**Abstract**: In this paper, we explore a self-supervised model that learns to detect the symmetry of a single object without requiring a dataset-relying solely on the input object itself. We hypothesize that the symmetry of an object can be determined by its intrinsic features, eliminating the need for large datasets during training. Additionally, we design a self-supervised learning strategy that removes the necessity of ground truth labels. These two key elements make our approach both effective and efficient, addressing the prohibitive costs associated with constructing large, labeled datasets for this task. The novelty of our method lies in computing features for each point on the object based on the idea that symmetric points should exhibit similar visual appearances. To achieve this, we leverage features extracted from a foundational image model to compute a visual descriptor for the points. This approach equips the point cloud with visual features that facilitate the optimization of our self-supervised model. Experimental results demonstrate that our method surpasses the state-of-the-art models trained on large datasets. Furthermore, our model is more efficient, effective, and operates with minimal computational and data resources. 

**Abstract (ZH)**: 本文探索了一种自监督模型，该模型能够在无需依赖数据集的情况下，仅通过输入的对象本身来学习检测单个对象的对称性。我们假设对象的对称性可以通过其固有特征来确定，从而在训练过程中无需大量数据集。此外，我们设计了一种自监督学习策略，消除了对ground truth标签的依赖。这两个关键元素使得我们的方法既有效又高效，能够解决构建大型标注数据集的高昂成本问题。我们的方法的创新之处在于，基于对称点应具有相似视觉特征的想法，为对象上的每个点计算特征。通过利用基础图像模型抽取的特征来计算点的视觉描述符，该方法为点云增添了视觉特征，促进了我们自监督模型的优化。实验结果表明，我们的方法在大型数据集上训练的模型上表现出更优越的效果。此外，我们的模型更高效、更有效，并且所需计算和数据资源较少。 

---
# RectifiedHR: Enable Efficient High-Resolution Image Generation via Energy Rectification 

**Title (ZH)**: RectifiedHR: 通过能量校正实现高效高分辨率图像生成 

**Authors**: Zhen Yang, Guibao Shen, Liang Hou, Mushui Liu, Luozhou Wang, Xin Tao, Pengfei Wan, Di Zhang, Ying-Cong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02537)  

**Abstract**: Diffusion models have achieved remarkable advances in various image generation tasks. However, their performance notably declines when generating images at resolutions higher than those used during the training period. Despite the existence of numerous methods for producing high-resolution images, they either suffer from inefficiency or are hindered by complex operations. In this paper, we propose RectifiedHR, an efficient and straightforward solution for training-free high-resolution image generation. Specifically, we introduce the noise refresh strategy, which theoretically only requires a few lines of code to unlock the model's high-resolution generation ability and improve efficiency. Additionally, we first observe the phenomenon of energy decay that may cause image blurriness during the high-resolution image generation process. To address this issue, we propose an Energy Rectification strategy, where modifying the hyperparameters of the classifier-free guidance effectively improves the generation performance. Our method is entirely training-free and boasts a simple implementation logic. Through extensive comparisons with numerous baseline methods, our RectifiedHR demonstrates superior effectiveness and efficiency. 

**Abstract (ZH)**: 基于噪声刷新和能量校正的训练免费高分辨率图像生成方法 

---
# BioD2C: A Dual-level Semantic Consistency Constraint Framework for Biomedical VQA 

**Title (ZH)**: BioD2C：一种生物医学VQA的双层语义一致性约束框架 

**Authors**: Zhengyang Ji, Shang Gao, Li Liu, Yifan Jia, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2503.02476)  

**Abstract**: Biomedical visual question answering (VQA) has been widely studied and has demonstrated significant application value and potential in fields such as assistive medical diagnosis. Despite their success, current biomedical VQA models perform multimodal information interaction only at the model level within large language models (LLMs), leading to suboptimal multimodal semantic alignment when dealing with complex tasks. To address this issue, we propose BioD2C: a novel Dual-level Semantic Consistency Constraint Framework for Biomedical VQA, which achieves dual-level semantic interaction alignment at both the model and feature levels, enabling the model to adaptively learn visual features based on the question. Specifically, we firstly integrate textual features into visual features via an image-text fusion mechanism as feature-level semantic interaction, obtaining visual features conditioned on the given text; and then introduce a text-queue-based cross-modal soft semantic loss function to further align the image semantics with the question semantics. Specifically, in this work, we establish a new dataset, BioVGQ, to address inherent biases in prior datasets by filtering manually-altered images and aligning question-answer pairs with multimodal context, and train our model on this dataset. Extensive experimental results demonstrate that BioD2C achieves state-of-the-art (SOTA) performance across multiple downstream datasets, showcasing its robustness, generalizability, and potential to advance biomedical VQA research. 

**Abstract (ZH)**: 医学生物视觉问答（VQA）已经在辅助医疗诊断等领域展示了显著的应用价值和潜力，并广泛研究。尽管取得了成功，当前的医学生物VQA模型仅在大规模语言模型（LLMs）的模型层面进行多模态信息交互，导致在处理复杂任务时的多模态语义对齐不足。为了解决这一问题，我们提出了BioD2C：一种新型的双层语义一致性约束框架，该框架在模型和特征层面实现了双层语义交互对齐，使模型能够根据问题自适应地学习视觉特征。具体而言，我们首先通过图文融合机制将文本特征整合到视觉特征中，实现特征层面的语义交互，获得基于给定文本的视觉特征；然后引入基于文本队列的跨模态软语义损失函数，进一步将图像语义与问题语义对齐。在本文中，我们建立了新的数据集BioVGQ，通过筛选手动修改的图像并使问题-答案对与多模态上下文对齐，解决了先前数据集中的固有偏差，并在该数据集上训练我们的模型。广泛的实验结果表明，BioD2C 在多个下游数据集中取得了最先进的（SOTA）性能，展示了其稳健性、通用性和推动医学生物VQA研究的潜力。 

---
# Exploring Model Quantization in GenAI-based Image Inpainting and Detection of Arable Plants 

**Title (ZH)**: 基于GenAI的图像修复与可耕地植物检测中模型量化探索 

**Authors**: Sourav Modak, Ahmet Oğuz Saltık, Anthony Stein  

**Link**: [PDF](https://arxiv.org/pdf/2503.02420)  

**Abstract**: Deep learning-based weed control systems often suffer from limited training data diversity and constrained on-board computation, impacting their real-world performance. To overcome these challenges, we propose a framework that leverages Stable Diffusion-based inpainting to augment training data progressively in 10% increments -- up to an additional 200%, thus enhancing both the volume and diversity of samples. Our approach is evaluated on two state-of-the-art object detection models, YOLO11(l) and RT-DETR(l), using the mAP50 metric to assess detection performance. We explore quantization strategies (FP16 and INT8) for both the generative inpainting and detection models to strike a balance between inference speed and accuracy. Deployment of the downstream models on the Jetson Orin Nano demonstrates the practical viability of our framework in resource-constrained environments, ultimately improving detection accuracy and computational efficiency in intelligent weed management systems. 

**Abstract (ZH)**: 基于深度学习的杂草控制系统常常受到有限的训练数据多样性以及车载计算能力受限的影响，影响其实用性能。为克服这些挑战，我们提出了一种框架，利用Stable Diffusion-based inpainting逐步增加训练数据——每次增加10%，最高可达200%的额外数据，从而增加样本的数量和多样性。我们的方法在YOLO11(l)和RT-DETR(l)两种先进的目标检测模型上进行了评估，使用mAP50指标来评估检测性能。我们探索了生成性 inpainting 模型和检测模型的量化策略（FP16 和 INT8），以在推理速度和准确性之间寻求平衡。在Jetson Orin Nano上的下游模型部署证明了该框架在资源受限环境中的实用可行性，最终提高了智能杂草管理系统中的检测准确性和计算效率。 

---
# BdSLW401: Transformer-Based Word-Level Bangla Sign Language Recognition Using Relative Quantization Encoding (RQE) 

**Title (ZH)**: BdSLW401：基于相对量化编码（RQE）的变压器驱动的孟加拉手语单词级识别 

**Authors**: Husne Ara Rubaiyeat, Njayou Youssouf, Md Kamrul Hasan, Hasan Mahmud  

**Link**: [PDF](https://arxiv.org/pdf/2503.02360)  

**Abstract**: Sign language recognition (SLR) for low-resource languages like Bangla suffers from signer variability, viewpoint variations, and limited annotated datasets. In this paper, we present BdSLW401, a large-scale, multi-view, word-level Bangla Sign Language (BdSL) dataset with 401 signs and 102,176 video samples from 18 signers in front and lateral views. To improve transformer-based SLR, we introduce Relative Quantization Encoding (RQE), a structured embedding approach anchoring landmarks to physiological reference points and quantize motion trajectories. RQE improves attention allocation by decreasing spatial variability, resulting in 44.3% WER reduction in WLASL100, 21.0% in SignBD-200, and significant gains in BdSLW60 and SignBD-90. However, fixed quantization becomes insufficient on large-scale datasets (e.g., WLASL2000), indicating the need for adaptive encoding strategies. Further, RQE-SF, an extended variant that stabilizes shoulder landmarks, achieves improvements in pose consistency at the cost of small trade-offs in lateral view recognition. The attention graphs prove that RQE improves model interpretability by focusing on the major articulatory features (fingers, wrists) and the more distinctive frames instead of global pose changes. Introducing BdSLW401 and demonstrating the effectiveness of RQE-enhanced structured embeddings, this work advances transformer-based SLR for low-resource languages and sets a benchmark for future research in this area. 

**Abstract (ZH)**: 低资源语言孟加拉手语识别中相对量化编码的大规模多视角单词级孟加拉手语数据集(BdSLW401)及其应用 

---
# Are Large Vision Language Models Good Game Players? 

**Title (ZH)**: 大型视觉语言模型是好的游戏选手吗？ 

**Authors**: Xinyu Wang, Bohan Zhuang, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02358)  

**Abstract**: Large Vision Language Models (LVLMs) have demonstrated remarkable abilities in understanding and reasoning about both visual and textual information. However, existing evaluation methods for LVLMs, primarily based on benchmarks like Visual Question Answering and image captioning, often fail to capture the full scope of LVLMs' capabilities. These benchmarks are limited by issues such as inadequate assessment of detailed visual perception, data contamination, and a lack of focus on multi-turn reasoning. To address these challenges, we propose \method{}, a game-based evaluation framework designed to provide a comprehensive assessment of LVLMs' cognitive and reasoning skills in structured environments. \method{} uses a set of games to evaluate LVLMs on four core tasks: Perceiving, Question Answering, Rule Following, and End-to-End Playing, with each target task designed to assess specific abilities, including visual perception, reasoning, decision-making, etc. Based on this framework, we conduct extensive experiments that explore the limitations of current LVLMs, such as handling long structured outputs and perceiving detailed and dense elements. Code and data are publicly available at this https URL. 

**Abstract (ZH)**: 大型视觉语言模型（LVLMs）在理解和推理视觉及文本信息方面展现了出色的 ability。然而，现有 LVLMs 的评价方法主要依托视觉问答和图像字幕等基准测试，往往无法全面捕捉 LVLMs 的能力。这些基准测试受限于详细视觉感知评估不足、数据污染以及多轮推理关注不充分等问题。为应对这些挑战，我们提出 \method{}，一种基于游戏的设计评价框架，旨在为 LVLMs 在结构化环境中的认知和推理能力提供全面评估。\method{} 使用一系列游戏来评估 LVLMs 在四个核心任务上的表现：感知、问答、规则遵循和端到端游戏，每个目标任务都旨在评估特定的能力，包括视觉感知、推理、决策等。基于此框架，我们进行了广泛的实验，探讨当前 LVLMs 的局限性，如处理长结构化输出和感知详细密集元素等问题。相关代码和数据可以在该 URL 公开访问：[这个 https URL]。 

---
# CQ CNN: A Hybrid Classical Quantum Convolutional Neural Network for Alzheimer's Disease Detection Using Diffusion Generated and U Net Segmented 3D MRI 

**Title (ZH)**: CQ CNN：一种用于阿尔茨海默病检测的混合经典量子卷积神经网络，基于扩散生成和U-net分割的3D MRI 

**Authors**: Mominul Islam, Mohammad Junayed Hasan, M.R.C. Mahdy  

**Link**: [PDF](https://arxiv.org/pdf/2503.02345)  

**Abstract**: The detection of Alzheimer disease (AD) from clinical MRI data is an active area of research in medical imaging. Recent advances in quantum computing, particularly the integration of parameterized quantum circuits (PQCs) with classical machine learning architectures, offer new opportunities to develop models that may outperform traditional methods. However, quantum machine learning (QML) remains in its early stages and requires further experimental analysis to better understand its behavior and limitations. In this paper, we propose an end to end hybrid classical quantum convolutional neural network (CQ CNN) for AD detection using clinically formatted 3D MRI data. Our approach involves developing a framework to make 3D MRI data usable for machine learning, designing and training a brain tissue segmentation model (Skull Net), and training a diffusion model to generate synthetic images for the minority class. Our converged models exhibit potential quantum advantages, achieving higher accuracy in fewer epochs than classical models. The proposed beta8 3 qubit model achieves an accuracy of 97.50%, surpassing state of the art (SOTA) models while requiring significantly fewer computational resources. In particular, the architecture employs only 13K parameters (0.48 MB), reducing the parameter count by more than 99.99% compared to current SOTA models. Furthermore, the diffusion-generated data used to train our quantum models, in conjunction with real samples, preserve clinical structural standards, representing a notable first in the field of QML. We conclude that CQCNN architecture like models, with further improvements in gradient optimization techniques, could become a viable option and even a potential alternative to classical models for AD detection, especially in data limited and resource constrained clinical settings. 

**Abstract (ZH)**: 基于临床MRI数据的阿尔茨海默病检测中的端到端混合经典量子卷积神经网络研究 

---
# One Patient's Annotation is Another One's Initialization: Towards Zero-Shot Surgical Video Segmentation with Cross-Patient Initialization 

**Title (ZH)**: 一位患者的标注是另一位患者的初始化：面向跨患者初始化的零样本手术视频分割 

**Authors**: Seyed Amir Mousavi, Utku Ozbulak, Francesca Tozzi, Nikdokht Rashidian, Wouter Willaert, Joris Vankerschaver, Wesley De Neve  

**Link**: [PDF](https://arxiv.org/pdf/2503.02228)  

**Abstract**: Video object segmentation is an emerging technology that is well-suited for real-time surgical video segmentation, offering valuable clinical assistance in the operating room by ensuring consistent frame tracking. However, its adoption is limited by the need for manual intervention to select the tracked object, making it impractical in surgical settings. In this work, we tackle this challenge with an innovative solution: using previously annotated frames from other patients as the tracking frames. We find that this unconventional approach can match or even surpass the performance of using patients' own tracking frames, enabling more autonomous and efficient AI-assisted surgical workflows. Furthermore, we analyze the benefits and limitations of this approach, highlighting its potential to enhance segmentation accuracy while reducing the need for manual input. Our findings provide insights into key factors influencing performance, offering a foundation for future research on optimizing cross-patient frame selection for real-time surgical video analysis. 

**Abstract (ZH)**: 基于其他患者标注帧的实时手术视频对象分割：一种创新的自动解决方案及其影响分析 

---
# Adaptive Camera Sensor for Vision Models 

**Title (ZH)**: 自适应摄像头传感器以供视觉模型使用 

**Authors**: Eunsu Baek, Sunghwan Han, Taesik Gong, Hyung-Sin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.02170)  

**Abstract**: Domain shift remains a persistent challenge in deep-learning-based computer vision, often requiring extensive model modifications or large labeled datasets to address. Inspired by human visual perception, which adjusts input quality through corrective lenses rather than over-training the brain, we propose Lens, a novel camera sensor control method that enhances model performance by capturing high-quality images from the model's perspective rather than relying on traditional human-centric sensor control. Lens is lightweight and adapts sensor parameters to specific models and scenes in real-time. At its core, Lens utilizes VisiT, a training-free, model-specific quality indicator that evaluates individual unlabeled samples at test time using confidence scores without additional adaptation costs. To validate Lens, we introduce ImageNet-ES Diverse, a new benchmark dataset capturing natural perturbations from varying sensor and lighting conditions. Extensive experiments on both ImageNet-ES and our new ImageNet-ES Diverse show that Lens significantly improves model accuracy across various baseline schemes for sensor control and model modification while maintaining low latency in image captures. Lens effectively compensates for large model size differences and integrates synergistically with model improvement techniques. Our code and dataset are available at this http URL. 

**Abstract (ZH)**: 基于深度学习的计算机视觉领域中，域适应仍然是一个持续性的挑战，往往需要对模型进行大量修改或依赖大量标记数据来解决。受人类视觉感知的启发，人类通过矫正镜头调整输入质量而非过度训练大脑，我们提出了一种新的摄像传感器控制方法Lens，该方法通过从模型视角捕获高质量图像来提升模型性能，而非依赖传统的人本中心传感器控制。Lens轻量级且能够实现实时自适应调整传感器参数，以适应特定模型和场景。核心上，Lens利用了VisiT，这是一种无需额外适应成本且针对特定模型的质量指标，在测试时使用置信分数评估未标记样本。为了验证Lens的有效性，我们引入了ImageNet-ES Diverse新基准数据集，该数据集捕捉了不同传感器和光照条件下自然的扰动。在ImageNet-ES和新引入的ImageNet-ES Diverse两个基准上的广泛实验表明，Lens显著提升了传感器控制和模型修改的各种基线方案的模型准确性，同时保持了低延迟的图像捕获。Lens能够有效补偿大型模型尺寸差异，并与模型改进技术协同工作。我们的代码和数据集可访问于此网址。 

---
# FASTer: Focal Token Acquiring-and-Scaling Transformer for Long-term 3D Object Detection 

**Title (ZH)**: FASTer: 用于长期三维物体检测的焦点 token 获取与缩放变换器 

**Authors**: Chenxu Dang, Zaipeng Duan, Pei An, Xinmin Zhang, Xuzhong Hu, Jie Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.01899)  

**Abstract**: Recent top-performing temporal 3D detectors based on Lidars have increasingly adopted region-based paradigms. They first generate coarse proposals, followed by encoding and fusing regional features. However, indiscriminate sampling and fusion often overlook the varying contributions of individual points and lead to exponentially increased complexity as the number of input frames grows. Moreover, arbitrary result-level concatenation limits the global information extraction. In this paper, we propose a Focal Token Acquring-and-Scaling Transformer (FASTer), which dynamically selects focal tokens and condenses token sequences in an adaptive and lightweight manner. Emphasizing the contribution of individual tokens, we propose a simple but effective Adaptive Scaling mechanism to capture geometric contexts while sifting out focal points. Adaptively storing and processing only focal points in historical frames dramatically reduces the overall complexity. Furthermore, a novel Grouped Hierarchical Fusion strategy is proposed, progressively performing sequence scaling and Intra-Group Fusion operations to facilitate the exchange of global spatial and temporal information. Experiments on the Waymo Open Dataset demonstrate that our FASTer significantly outperforms other state-of-the-art detectors in both performance and efficiency while also exhibiting improved flexibility and robustness. The code is available at this https URL. 

**Abstract (ZH)**: 基于激光雷达的 Recent 顶级时序 3D 检测器越来越多地采用区域为基础的范式。它们首先生成粗略的提议，然后编码和融合区域特征。然而，非区分性的采样和融合往往忽视了单个点的不同贡献，并随着输入帧数量的增长导致复杂性呈指数级增加。此外，任意的结果级拼接限制了全局信息的提取。在本文中，我们提出了一种焦点标记获取和缩放变换器（FASTer），它以动态选择焦点标记并在适应性和轻量化的方式中浓缩标记序列。强调单个标记的贡献，我们提出了一种简单而有效的自适应缩放机制，以捕获几何上下文并筛选出焦点点。仅在历史帧中适当地存储和处理焦点点显著降低了整体复杂性。此外，我们还提出了一种新的分组层次融合策略，逐步执行序列缩放和组内融合操作，以促进全局空间和时间信息的交换。在 Waymo 开放数据集上的实验表明，在性能和效率方面，我们的 FASTer 显著优于其他最先进的检测器，同时还表现出更好的灵活性和鲁棒性。代码可在以下链接获取：this https URL。 

---
# Efficient Diffusion as Low Light Enhancer 

**Title (ZH)**: 高效扩散作为低光增强器 

**Authors**: Guanzhou Lan, Qianli Ma, Yuqi Yang, Zhigang Wang, Dong Wang, Xuelong Li, Bin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2410.12346)  

**Abstract**: The computational burden of the iterative sampling process remains a major challenge in diffusion-based Low-Light Image Enhancement (LLIE). Current acceleration methods, whether training-based or training-free, often lead to significant performance degradation, highlighting the trade-off between performance and efficiency. In this paper, we identify two primary factors contributing to performance degradation: fitting errors and the inference gap. Our key insight is that fitting errors can be mitigated by linearly extrapolating the incorrect score functions, while the inference gap can be reduced by shifting the Gaussian flow to a reflectance-aware residual space. Based on the above insights, we design Reflectance-Aware Trajectory Refinement (RATR) module, a simple yet effective module to refine the teacher trajectory using the reflectance component of images. Following this, we introduce \textbf{Re}flectance-aware \textbf{D}iffusion with \textbf{Di}stilled \textbf{T}rajectory (\textbf{ReDDiT}), an efficient and flexible distillation framework tailored for LLIE. Our framework achieves comparable performance to previous diffusion-based methods with redundant steps in just 2 steps while establishing new state-of-the-art (SOTA) results with 8 or 4 steps. Comprehensive experimental evaluations on 10 benchmark datasets validate the effectiveness of our method, consistently outperforming existing SOTA methods. 

**Abstract (ZH)**: 反射-aware 扩散与提炼轨迹蒸馏 (ReDDiT) 用于低光图像增强 

---
