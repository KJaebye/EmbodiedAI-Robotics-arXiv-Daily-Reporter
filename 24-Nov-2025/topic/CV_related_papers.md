# Planning with Sketch-Guided Verification for Physics-Aware Video Generation 

**Title (ZH)**: 基于草图引导验证的物理意识视频生成规划 

**Authors**: Yidong Huang, Zun Wang, Han Lin, Dong-Ki Kim, Shayegan Omidshafiei, Jaehong Yoon, Yue Zhang, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2511.17450)  

**Abstract**: Recent video generation approaches increasingly rely on planning intermediate control signals such as object trajectories to improve temporal coherence and motion fidelity. However, these methods mostly employ single-shot plans that are typically limited to simple motions, or iterative refinement which requires multiple calls to the video generator, incuring high computational cost. To overcome these limitations, we propose SketchVerify, a training-free, sketch-verification-based planning framework that improves motion planning quality with more dynamically coherent trajectories (i.e., physically plausible and instruction-consistent motions) prior to full video generation by introducing a test-time sampling and verification loop. Given a prompt and a reference image, our method predicts multiple candidate motion plans and ranks them using a vision-language verifier that jointly evaluates semantic alignment with the instruction and physical plausibility. To efficiently score candidate motion plans, we render each trajectory as a lightweight video sketch by compositing objects over a static background, which bypasses the need for expensive, repeated diffusion-based synthesis while achieving comparable performance. We iteratively refine the motion plan until a satisfactory one is identified, which is then passed to the trajectory-conditioned generator for final synthesis. Experiments on WorldModelBench and PhyWorldBench demonstrate that our method significantly improves motion quality, physical realism, and long-term consistency compared to competitive baselines while being substantially more efficient. Our ablation study further shows that scaling up the number of trajectory candidates consistently enhances overall performance. 

**Abstract (ZH)**: Recent Video Generation Approaches Based on SketchVerification: A Training-Free Framework for Improved Motion Planning 

---
# Sparse Mixture-of-Experts for Multi-Channel Imaging: Are All Channel Interactions Required? 

**Title (ZH)**: 多通道成像中的稀疏专家混合模型：所有通道交互都是必要的吗？ 

**Authors**: Sukwon Yun, Heming Yao, Burkhard Hoeckendorf, David Richmond, Aviv Regev, Russell Littman  

**Link**: [PDF](https://arxiv.org/pdf/2511.17400)  

**Abstract**: Vision Transformers ($\text{ViTs}$) have become the backbone of vision foundation models, yet their optimization for multi-channel domains - such as cell painting or satellite imagery - remains underexplored. A key challenge in these domains is capturing interactions between channels, as each channel carries different information. While existing works have shown efficacy by treating each channel independently during tokenization, this approach naturally introduces a major computational bottleneck in the attention block - channel-wise comparisons leads to a quadratic growth in attention, resulting in excessive $\text{FLOPs}$ and high training cost. In this work, we shift focus from efficacy to the overlooked efficiency challenge in cross-channel attention and ask: "Is it necessary to model all channel interactions?". Inspired by the philosophy of Sparse Mixture-of-Experts ($\text{MoE}$), we propose MoE-ViT, a Mixture-of-Experts architecture for multi-channel images in $\text{ViTs}$, which treats each channel as an expert and employs a lightweight router to select only the most relevant experts per patch for attention. Proof-of-concept experiments on real-world datasets - JUMP-CP and So2Sat - demonstrate that $\text{MoE-ViT}$ achieves substantial efficiency gains without sacrificing, and in some cases enhancing, performance, making it a practical and attractive backbone for multi-channel imaging. 

**Abstract (ZH)**: Vision Transformer的跨通道注意力高效挑战：Sparse Mixture-of-Experts架构探索 

---
# Quantum Masked Autoencoders for Vision Learning 

**Title (ZH)**: 量子遮蔽自编码器在视觉学习中的应用 

**Authors**: Emma Andrews, Prabhat Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2511.17372)  

**Abstract**: Classical autoencoders are widely used to learn features of input data. To improve the feature learning, classical masked autoencoders extend classical autoencoders to learn the features of the original input sample in the presence of masked-out data. While quantum autoencoders exist, there is no design and implementation of quantum masked autoencoders that can leverage the benefits of quantum computing and quantum autoencoders. In this paper, we propose quantum masked autoencoders (QMAEs) that can effectively learn missing features of a data sample within quantum states instead of classical embeddings. We showcase that our QMAE architecture can learn the masked features of an image and can reconstruct the masked input image with improved visual fidelity in MNIST images. Experimental evaluation highlights that QMAE can significantly outperform (12.86% on average) in classification accuracy compared to state-of-the-art quantum autoencoders in the presence of masks. 

**Abstract (ZH)**: 量子掩码自编码器（QMAEs）：一种在量子态中学习缺失特征的新方法 

---
# MuM: Multi-View Masked Image Modeling for 3D Vision 

**Title (ZH)**: MuM: 多视图掩蔽图像建模以实现三维视觉 

**Authors**: David Nordström, Johan Edstedt, Fredrik Kahl, Georg Bökman  

**Link**: [PDF](https://arxiv.org/pdf/2511.17309)  

**Abstract**: Self-supervised learning on images seeks to extract meaningful visual representations from unlabeled data. When scaled to large datasets, this paradigm has achieved state-of-the-art performance and the resulting trained models such as DINOv3 have seen widespread adoption. However, most prior efforts are optimized for semantic understanding rather than geometric reasoning. One important exception is Cross-View Completion, CroCo, which is a form of masked autoencoding (MAE) tailored for 3D understanding. In this work, we continue on the path proposed by CroCo and focus on learning features tailored for 3D vision. In a nutshell, we extend MAE to arbitrarily many views of the same scene. By uniformly masking all views and employing a lightweight decoder with inter-frame attention, our approach is inherently simpler and more scalable than CroCo. We evaluate the resulting model, MuM, extensively on downstream tasks including feedforward reconstruction, dense image matching and relative pose estimation, finding that it outperforms the state-of-the-art visual encoders DINOv3 and CroCo v2. 

**Abstract (ZH)**: 自监督学习通过无标签数据提取有意义的视觉表示。当扩展到大型数据集时，这一范式已达到了最先进的性能，训练出的模型如DINOv3得到了广泛应用。然而，大多数早期努力主要针对语义理解而非几何推理。一个重要的例外是Cross-View Completion (CroCo)，它是为三维理解定制的掩蔽自编码（MAE）形式。在本文中，我们遵循CroCo提出的方法，专注于学习针对三维视觉的特征。简言之，我们将MAE扩展到同一场景的任意多个视图。通过均匀掩蔽所有视图并使用轻量级解码器和帧间注意力机制，我们的方法比CroCo更为简单和易于扩展。我们在包括前向重建、密集图像匹配和相对姿态估计在内的下游任务上广泛评估了由此产生的模型MuM，发现它在性能上优于最先进的视觉编码器DINOv3和CroCo v2。 

---
# Leveraging CVAE for Joint Configuration Estimation of Multifingered Grippers from Point Cloud Data 

**Title (ZH)**: 基于条件变分自编码器的多指灵巧手点云数据联合配置估计 

**Authors**: Julien Merand, Boris Meden, Mathieu Grossard  

**Link**: [PDF](https://arxiv.org/pdf/2511.17276)  

**Abstract**: This paper presents an efficient approach for determining the joint configuration of a multifingered gripper solely from the point cloud data of its poly-articulated chain, as generated by visual sensors, simulations or even generative neural networks. Well-known inverse kinematics (IK) techniques can provide mathematically exact solutions (when they exist) for joint configuration determination based solely on the fingertip pose, but often require post-hoc decision-making by considering the positions of all intermediate phalanges in the gripper's fingers, or rely on algorithms to numerically approximate solutions for more complex kinematics. In contrast, our method leverages machine learning to implicitly overcome these challenges. This is achieved through a Conditional Variational Auto-Encoder (CVAE), which takes point cloud data of key structural elements as input and reconstructs the corresponding joint configurations. We validate our approach on the MultiDex grasping dataset using the Allegro Hand, operating within 0.05 milliseconds and achieving accuracy comparable to state-of-the-art methods. This highlights the effectiveness of our pipeline for joint configuration estimation within the broader context of AI-driven techniques for grasp planning. 

**Abstract (ZH)**: 本文提出了一种高效方法，仅从视觉传感器、仿真或生成神经网络生成的手指多关节链点云数据中确定多指 gripper 的联合配置。众所周知的逆运动学（IK）技术可以在指尖姿态的基础上提供数学上精确的解决方案（当存在时），但通常需要考虑 gripper 手指中所有中间指节的位置进行后续决策，或者依赖算法对更复杂的运动学进行数值近似求解。相比之下，我们的方法利用机器学习隐式克服了这些挑战。这是通过条件变分自编码器（CVAE）实现的，该编码器以关键结构元素的点云数据作为输入，并重构相应的关节配置。我们使用 MultiDex 抓取数据集和 Allegro 手进行验证，在 0.05 毫秒内运行，并且精度与现有最佳方法相当。这突显了我们的流水线在更广泛的人工智能驱动的抓取规划技术中的有效性。 

---
# Range-Edit: Semantic Mask Guided Outdoor LiDAR Scene Editing 

**Title (ZH)**: 范围编辑：语义掩码引导的室外LiDAR场景编辑 

**Authors**: Suchetan G. Uppur, Hemant Kumar, Vaibhav Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.17269)  

**Abstract**: Training autonomous driving and navigation systems requires large and diverse point cloud datasets that capture complex edge case scenarios from various dynamic urban settings. Acquiring such diverse scenarios from real-world point cloud data, especially for critical edge cases, is challenging, which restricts system generalization and robustness. Current methods rely on simulating point cloud data within handcrafted 3D virtual environments, which is time-consuming, computationally expensive, and often fails to fully capture the complexity of real-world scenes. To address some of these issues, this research proposes a novel approach that addresses the problem discussed by editing real-world LiDAR scans using semantic mask-based guidance to generate novel synthetic LiDAR point clouds. We incorporate range image projection and semantic mask conditioning to achieve diffusion-based generation. Point clouds are transformed to 2D range view images, which are used as an intermediate representation to enable semantic editing using convex hull-based semantic masks. These masks guide the generation process by providing information on the dimensions, orientations, and locations of objects in the real environment, ensuring geometric consistency and realism. This approach demonstrates high-quality LiDAR point cloud generation, capable of producing complex edge cases and dynamic scenes, as validated on the KITTI-360 dataset. This offers a cost-effective and scalable solution for generating diverse LiDAR data, a step toward improving the robustness of autonomous driving systems. 

**Abstract (ZH)**: 基于语义掩码指导的现实LiDAR扫描编辑以生成新型合成LiDAR点云 

---
# A lightweight detector for real-time detection of remote sensing images 

**Title (ZH)**: 轻量级检测器用于实时遥感图像检测 

**Authors**: Qianyi Wang, Guoqiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2511.17147)  

**Abstract**: Remote sensing imagery is widely used across various fields, yet real-time detection remains challenging due to the prevalence of small objects and the need to balance accuracy with efficiency. To address this, we propose DMG-YOLO, a lightweight real-time detector tailored for small object detection in remote sensing images. Specifically, we design a Dual-branch Feature Extraction (DFE) module in the backbone, which partitions feature maps into two parallel branches: one extracts local features via depthwise separable convolutions, and the other captures global context using a vision transformer with a gating mechanism. Additionally, a Multi-scale Feature Fusion (MFF) module with dilated convolutions enhances multi-scale integration while preserving fine details. In the neck, we introduce the Global and Local Aggregate Feature Pyramid Network (GLAFPN) to further boost small object detection through global-local feature fusion. Extensive experiments on the VisDrone2019 and NWPU VHR-10 datasets show that DMG-YOLO achieves competitive performance in terms of mAP, model size, and other key metrics. 

**Abstract (ZH)**: 远程 sensing 图像在各类领域中有广泛的应用，但由于小目标的普遍存在和对准确性和效率之间的平衡要求，实时检测仍然具有挑战性。为了解决这一问题，我们提出 DMG-YOLO，这是一种针对遥感图像中小目标检测的轻量级实时检测器。具体来说，我们在主干中设计了一个双支路特征提取（DFE）模块，将特征图分割成两个并行分支：一个通过深度可分离卷积提取局部特征，另一个使用带有门控机制的视觉变换器捕获全局上下文。此外，通过膨胀卷积增强的多尺度特征融合（MFF）模块提升了多尺度信息的整合同时保留了细节。在颈部，我们引入了全局和局部聚合特征金字塔网络（GLAFPN），进一步通过全局-局部特征融合提升小目标检测性能。在 VisDrone2019 和 NWPU VHR-10 数据集上的广泛实验表明，DMG-YOLO 在 mAP、模型大小等关键指标上取得了竞争力的性能。 

---
# Spanning Tree Autoregressive Visual Generation 

**Title (ZH)**: 树状结构自回归视觉生成 

**Authors**: Sangkyu Lee, Changho Lee, Janghoon Han, Hosung Song, Tackgeun You, Hwasup Lim, Stanley Jungkyu Choi, Honglak Lee, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.17089)  

**Abstract**: We present Spanning Tree Autoregressive (STAR) modeling, which can incorporate prior knowledge of images, such as center bias and locality, to maintain sampling performance while also providing sufficiently flexible sequence orders to accommodate image editing at inference. Approaches that expose randomly permuted sequence orders to conventional autoregressive (AR) models in visual generation for bidirectional context either suffer from a decline in performance or compromise the flexibility in sequence order choice at inference. Instead, STAR utilizes traversal orders of uniform spanning trees sampled in a lattice defined by the positions of image patches. Traversal orders are obtained through breadth-first search, allowing us to efficiently construct a spanning tree whose traversal order ensures that the connected partial observation of the image appears as a prefix in the sequence through rejection sampling. Through the tailored yet structured randomized strategy compared to random permutation, STAR preserves the capability of postfix completion while maintaining sampling performance without any significant changes to the model architecture widely adopted in the language AR modeling. 

**Abstract (ZH)**: 基于图遍历的 Spanning Tree Autoregressive (STAR) 模型 

---
# RacketVision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis 

**Title (ZH)**: racketVision: 统一球拍和球分析的多项 racket 运动基准 

**Authors**: Linfeng Dong, Yuchen Yang, Hao Wu, Wei Wang, Yuenan HouZhihang Zhong, Xiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.17045)  

**Abstract**: We introduce RacketVision, a novel dataset and benchmark for advancing computer vision in sports analytics, covering table tennis, tennis, and badminton. The dataset is the first to provide large-scale, fine-grained annotations for racket pose alongside traditional ball positions, enabling research into complex human-object interactions. It is designed to tackle three interconnected tasks: fine-grained ball tracking, articulated racket pose estimation, and predictive ball trajectory forecasting. Our evaluation of established baselines reveals a critical insight for multi-modal fusion: while naively concatenating racket pose features degrades performance, a CrossAttention mechanism is essential to unlock their value, leading to trajectory prediction results that surpass strong unimodal baselines. RacketVision provides a versatile resource and a strong starting point for future research in dynamic object tracking, conditional motion forecasting, and multimodal analysis in sports. Project page at this https URL 

**Abstract (ZH)**: 我们介绍RacketVision：一项涵盖乒乓球、网球和羽毛球的新型数据集与基准，推动体育分析中的计算机视觉发展。该数据集首次提供了包含 racket 姿态的精细标注和传统球位信息，旨在研究复杂的人物物体交互。它旨在解决三项相互关联的任务：精细粒度的球跟踪、冗余关节 racket 姿态估计以及预测球轨迹。我们的基线评估揭示了一个关键见解：尽管简单地拼接 racket 姿态特征会降低性能，但交叉注意力机制是解锁其价值的关键，从而获得超越强大单模基线的轨迹预测结果。RacketVision 提供了一种多功能的资源，并为未来动态物体跟踪、条件运动预测和体育中的多模态分析研究奠定了坚实的基础。项目页面请访问此链接。 

---
# MedImageInsight for Thoracic Cavity Health Classification from Chest X-rays 

**Title (ZH)**: MedImageInsightchestX射线胸腔健康分类 

**Authors**: Rama Krishna Boya, Mohan Kireeti Magalanadu, Azaruddin Palavalli, Rupa Ganesh Tekuri, Amrit Pattanayak, Prasanthi Enuga, Vignesh Esakki Muthu, Vivek Aditya Boya  

**Link**: [PDF](https://arxiv.org/pdf/2511.17043)  

**Abstract**: Chest radiography remains one of the most widely used imaging modalities for thoracic diagnosis, yet increasing imaging volumes and radiologist workload continue to challenge timely interpretation. In this work, we investigate the use of MedImageInsight, a medical imaging foundational model, for automated binary classification of chest X-rays into Normal and Abnormal categories. Two approaches were evaluated: (1) fine-tuning MedImageInsight for end-to-end classification, and (2) employing the model as a feature extractor for a transfer learning pipeline using traditional machine learning classifiers. Experiments were conducted using a combination of the ChestX-ray14 dataset and real-world clinical data sourced from partner hospitals. The fine-tuned classifier achieved the highest performance, with an ROC-AUC of 0.888 and superior calibration compared to the transfer learning models, demonstrating performance comparable to established architectures such as CheXNet. These results highlight the effectiveness of foundational medical imaging models in reducing task-specific training requirements while maintaining diagnostic reliability. The system is designed for integration into web-based and hospital PACS workflows to support triage and reduce radiologist burden. Future work will extend the model to multi-label pathology classification to provide preliminary diagnostic interpretation in clinical environments. 

**Abstract (ZH)**: 胸部X光检查仍然是胸腔诊断中最常用的成像技术之一，然而不断增加的成像量和放射科医师的工作负担继续对及时解读构成挑战。本研究探讨了使用MedImageInsight这一医学影像基础模型，对胸部X光进行自动二分类，将其分为正常和异常两类。评估了两种方法：（1）端到端分类的MedImageInsight微调，（2）将模型作为特征提取器用于传统的机器学习分类器迁移学习管道。实验使用ChestX-ray14数据集和来自合作医院的真实临床数据进行。微调分类器取得了最佳性能，AUC-ROC为0.888，并且在校准方面优于迁移学习模型，展示了与CheXNet等现有架构相当的性能。这些结果突显了医学影像基础模型在减少特定任务训练需求的同时保持诊断可靠性的有效性。该系统设计用于集成到基于Web和医院PACS的工作流程中，以支持初步诊断和减轻放射科医师的负担。未来工作将扩展模型到多标签病理分类，以在临床环境中提供初步诊断解释。 

---
# Parameter-Free Neural Lens Blur Rendering for High-Fidelity Composites 

**Title (ZH)**: 无参数神经透镜模糊渲染高保真合成 

**Authors**: Lingyan Ruan, Bin Chen, Taehyun Rhee  

**Link**: [PDF](https://arxiv.org/pdf/2511.17014)  

**Abstract**: Consistent and natural camera lens blur is important for seamlessly blending 3D virtual objects into photographed real-scenes. Since lens blur typically varies with scene depth, the placement of virtual objects and their corresponding blur levels significantly affect the visual fidelity of mixed reality compositions. Existing pipelines often rely on camera parameters (e.g., focal length, focus distance, aperture size) and scene depth to compute the circle of confusion (CoC) for realistic lens blur rendering. However, such information is often unavailable to ordinary users, limiting the accessibility and generalizability of these methods. In this work, we propose a novel compositing approach that directly estimates the CoC map from RGB images, bypassing the need for scene depth or camera metadata. The CoC values for virtual objects are inferred through a linear relationship between its signed CoC map and depth, and realistic lens blur is rendered using a neural reblurring network. Our method provides flexible and practical solution for real-world applications. Experimental results demonstrate that our method achieves high-fidelity compositing with realistic defocus effects, outperforming state-of-the-art techniques in both qualitative and quantitative evaluations. 

**Abstract (ZH)**: 一致且自然的相机镜头模糊对于无缝融合3D虚拟对象到拍摄的真实场景中至关重要。现有的流水线通常依赖相机参数（如焦距、对焦距离、光圈大小）和场景深度来计算圈形失焦（CoC）以实现逼真的镜头模糊渲染。然而，这些信息对于普通用户来说往往不可用，限制了这些方法的可访问性和普适性。在本文中，我们提出了一种新颖的合成方法，直接从RGB图像中估算CoC图，从而避免了场景深度或相机元数据的需求。虚拟对象的CoC值通过其带符号的CoC图与深度之间的线性关系来推断，并使用神经重模糊网络生成逼真的镜头模糊。我们的方法为实际应用提供了灵活且实用的解决方案。实验结果表明，我们的方法在定性和定量评估中均实现了高保真合成，并且具有逼真的焦外效果，优于现有最先进的技术。 

---
# OmniGround: A Comprehensive Spatio-Temporal Grounding Benchmark for Real-World Complex Scenarios 

**Title (ZH)**: 全方位地面：现实世界复杂场景的综合性时空定位基准 

**Authors**: Hong Gao, Jingyu Wu, Xiangkai Xu, Kangni Xie, Yunchen Zhang, Bin Zhong, Xurui Gao, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16937)  

**Abstract**: Spatio-Temporal Video Grounding (STVG) aims to localize target objects in videos based on natural language descriptions. Despite recent advances in Multimodal Large Language Models, a significant gap remains between current models and real-world demands involving diverse objects and complex queries. We attribute this to limited benchmark scope, causing models to exhibit category bias, oversimplified reasoning, and poor linguistic robustness. To address these limitations, we introduce OmniGround, a comprehensive benchmark with 3,475 videos spanning 81 categories and complex real-world queries. We propose the Forward-Backward-Refinement annotation pipeline that combines multi-directional tracking with intelligent error correction for high-quality labels. We further introduce DeepSTG, a systematic evaluation framework quantifying dataset quality across four complementary dimensions beyond superficial statistics. Evaluations reveal performance average drop of 10.4% on complex real-world scenes, particularly with small/occluded objects and intricate spatial relations. Motivated by these, we propose PG-TAF, a training-free two-stage framework decomposing STVG into high-level temporal grounding and fine-grained spatio-temporal propagation. Experiments demonstrate PG-TAF achieves 25.6% and 35.6% improvements in m\_tIoU and m\_vIoU on OmniGround with consistent gains across four benchmarks. 

**Abstract (ZH)**: 跨模态时空视频定位（STVG）旨在基于自然语言描述在视频中定位目标物体。尽管近年来多模态大型语言模型取得了进展，但当前模型与涉及多种物体和复杂查询的现实世界需求之间仍然存在显著差距。我们将其归因于基准范围有限，导致模型表现出类别偏差、简化推理和语言鲁棒性差的问题。为了解决这些限制，我们引入了OmniGround，这是一个全面的基准，包含3,475个视频和81个类别，以及复杂的现实世界查询。我们提出了结合多向追踪与智能错误修正的注释流水线，以生成高质量标签。我们进一步提出了DeepSTG，这是一种系统性的评估框架，可以从四个互补维度量化数据集质量，超越表面统计指标。评估结果显示，在复杂现实世界场景中，尤其是在小目标和被遮挡对象以及复杂的空间关系中，m\_tIoU和m\_vIoU的平均性能下降了10.4%。受此启发，我们提出了PG-TAF，这是一种无需训练的两阶段框架，将STVG分解为高层时间定位和精细时空传播。实验表明，在OmniGround和其他四个基准上，PG-TAF分别在m\_tIoU和m\_vIoU上取得了25.6%和35.6%的改进，且在所有四个基准上均表现出一致的增益。 

---
# Mesh RAG: Retrieval Augmentation for Autoregressive Mesh Generation 

**Title (ZH)**: 网格RAG：检索增强的自回归网格生成 

**Authors**: Xiatao Sun, Chen Liang, Qian Wang, Daniel Rakita  

**Link**: [PDF](https://arxiv.org/pdf/2511.16807)  

**Abstract**: 3D meshes are a critical building block for applications ranging from industrial design and gaming to simulation and robotics. Traditionally, meshes are crafted manually by artists, a process that is time-intensive and difficult to scale. To automate and accelerate this asset creation, autoregressive models have emerged as a powerful paradigm for artistic mesh generation. However, current methods to enhance quality typically rely on larger models or longer sequences that result in longer generation time, and their inherent sequential nature imposes a severe quality-speed trade-off. This sequential dependency also significantly complicates incremental editing. To overcome these limitations, we propose Mesh RAG, a novel, training-free, plug-and-play framework for autoregressive mesh generation models. Inspired by RAG for language models, our approach augments the generation process by leveraging point cloud segmentation, spatial transformation, and point cloud registration to retrieve, generate, and integrate mesh components. This retrieval-based approach decouples generation from its strict sequential dependency, facilitating efficient and parallelizable inference. We demonstrate the wide applicability of Mesh RAG across various foundational autoregressive mesh generation models, showing it significantly enhances mesh quality, accelerates generation speed compared to sequential part prediction, and enables incremental editing, all without model retraining. 

**Abstract (ZH)**: 3D网格是从小型设计到模拟和机器人技术广泛应用中的关键构建块。传统上，网格由艺术家手动创建，这是一个耗时且难以扩展的过程。为了自动化和加速这一资产创建过程，自回归模型已成为一种强大的艺术网格生成范式。然而，当前提高质量的方法通常依赖于更大的模型或更长的序列，从而导致生成时间延长，并且其固有的顺序性质导致了严重的质量和速度权衡。这种顺序依赖性还显著地复杂化了增量编辑。为此，我们提出Mesh RAG，这是一种全新的、无需训练、即插即用的自回归网格生成框架。受RAG语言模型的启发，我们的方法通过利用点云分割、空间变换和点云注册来增强生成过程，以检索、生成和集成网格组件。这种基于检索的方法解耦了生成过程与严格的顺序依赖性，便于高效和并行的推断。我们展示了Mesh RAG在各种基础自回归网格生成模型中的广泛应用，证明它显著提高了网格质量、比顺序部分预测加快了生成速度，并且能够进行增量编辑，而无需重新训练模型。 

---
# SAM 3: Segment Anything with Concepts 

**Title (ZH)**: SAM 3: 基于概念的分割anything 

**Authors**: Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, Christoph Feichtenhofer  

**Link**: [PDF](https://arxiv.org/pdf/2511.16719)  

**Abstract**: We present Segment Anything Model (SAM) 3, a unified model that detects, segments, and tracks objects in images and videos based on concept prompts, which we define as either short noun phrases (e.g., "yellow school bus"), image exemplars, or a combination of both. Promptable Concept Segmentation (PCS) takes such prompts and returns segmentation masks and unique identities for all matching object instances. To advance PCS, we build a scalable data engine that produces a high-quality dataset with 4M unique concept labels, including hard negatives, across images and videos. Our model consists of an image-level detector and a memory-based video tracker that share a single backbone. Recognition and localization are decoupled with a presence head, which boosts detection accuracy. SAM 3 doubles the accuracy of existing systems in both image and video PCS, and improves previous SAM capabilities on visual segmentation tasks. We open source SAM 3 along with our new Segment Anything with Concepts (SA-Co) benchmark for promptable concept segmentation. 

**Abstract (ZH)**: 我们提出了一种统一模型Segment Anything Model (SAM) 3，该模型基于概念提示检测、分割和跟踪图像和视频中的对象，这些提示可以是简短的名词短语（例如，“黄色校车”）、图像示例，或两者的结合。可提示的概念分割（PCS）接收这些提示并返回所有匹配对象实例的分割掩码和唯一身份。为了推进PCS，我们构建了一个可扩展的数据引擎，产生一个高质量的数据集，包含400万唯一概念标签，涵盖图像和视频。我们的模型由图像级检测器和基于记忆的视频跟踪器组成，共享单一骨干网络。识别和定位通过存在性头部解耦，从而提升检测准确性。在图像和视频PCS中，SAM 3将现有系统的准确性翻倍，并在视觉分割任务上改进了先前的SAM能力。我们开源了SAM 3以及我们的新可提示概念分割基准Segment Anything with Concepts (SA-Co)。 

---
# A Machine Learning-Driven Solution for Denoising Inertial Confinement Fusion Images 

**Title (ZH)**: 基于机器学习的惯性约束融合图像去噪解决方案 

**Authors**: Asya Y. Akkus, Bradley T. Wolfe, Pinghan Chu, Chengkun Huang, Chris S. Campbell, Mariana Alvarado Alvarez, Petr Volegov, David Fittinghoff, Robert Reinovsky, Zhehui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.16717)  

**Abstract**: Neutron imaging is important in optimizing analysis of inertial confinement fusion (ICF) events such as those at the National Ignition Facility (NIF) and improving current and future ICF platforms. However, images of neutron sources are often degraded by various types of noise. Most commonly, Gaussian and Poisson noise often coexist within one image, obscuring fine details and blurring edges. These noise types often overlap, making them difficult to distinguish and remove using conventional filtering and thresholding methods. As a result, noise removal techniques that preserve image fidelity are important for analyzing and interpreting images of a neutron source. Current solutions include a combination of filtering and thresholding methodologies. In the past, machine learning approaches were rarely implemented due to a lack of ground truth neutron imaging data for ICF processes. However, recent advances in synthetic data production, particularly in the fusion imaging field, have opened opportunities to investigate new denoising procedures using both supervised and unsupervised machine learning methods. In this study, we implement an unsupervised autoencoder with a Cohen-Daubechies- Feauveau (CDF 97) wavelet transform in the latent space for mixed Gaussian-Poisson denoising. The network successfully denoises neutron imaging data. Additionally, it demonstrates lower reconstruction error and superior edge preservation metrics when benchmarked with data generated by a forward model and compared to non-ML-based filtering mechanisms such as Block-matching and 3D filtering (BM3D). This approach presents a promising advancement in neutron image noise reduction and three-dimensional reconstruction analysis of ICF experiments. 

**Abstract (ZH)**: 基于Cohen-Daubechies-Feauveau小波的无监督自编码器在混合高斯-泊松噪声去除中的应用：惯性约束聚变实验中中子成像数据降噪与三维重建分析 

---
# Ellipsoid-Based Decision Boundaries for Open Intent Classification 

**Title (ZH)**: 基于椭球决策边界的一种开放意图分类方法 

**Authors**: Yuetian Zou, Hanlei Zhang, Hua Xu, Songze Li, Long Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2511.16685)  

**Abstract**: Textual open intent classification is crucial for real-world dialogue systems, enabling robust detection of unknown user intents without prior knowledge and contributing to the robustness of the system. While adaptive decision boundary methods have shown great potential by eliminating manual threshold tuning, existing approaches assume isotropic distributions of known classes, restricting boundaries to balls and overlooking distributional variance along different directions. To address this limitation, we propose EliDecide, a novel method that learns ellipsoid decision boundaries with varying scales along different feature directions. First, we employ supervised contrastive learning to obtain a discriminative feature space for known samples. Second, we apply learnable matrices to parameterize ellipsoids as the boundaries of each known class, offering greater flexibility than spherical boundaries defined solely by centers and radii. Third, we optimize the boundaries via a novelly designed dual loss function that balances empirical and open-space risks: expanding boundaries to cover known samples while contracting them against synthesized pseudo-open samples. Our method achieves state-of-the-art performance on multiple text intent benchmarks and further on a question classification dataset. The flexibility of the ellipsoids demonstrates superior open intent detection capability and strong potential for generalization to more text classification tasks in diverse complex open-world scenarios. 

**Abstract (ZH)**: 文本开放意图分类对于现实中的对话系统至关重要，能够无需先验知识 robust 地检测未知用户意图，从而增强系统的 robust 性。虽然自适应决策边界方法通过消除手动阈值调整显示出巨大潜力，但现有方法假设已知类别的等方差分布，限制边界为球体，忽略了不同方向上的分布方差。为解决这一局限，我们提出 EliDecide，一种新颖的方法，学习沿不同特征方向具有不同尺度的椭球决策边界。首先，我们采用监督对比学习获取已知样本的 discriminative 特征空间。其次，我们应用可学习矩阵参数化每已知类别的椭球边界，提供比仅由中心和半径定义的球体边界更大的灵活性。第三，我们通过一个新颖设计的双重损失函数优化边界，该函数平衡经验风险和开放空间风险：扩展边界以覆盖已知样本，同时相对于合成的伪开放样本收缩边界。我们的方法在多个文本意图基准和一个问题分类数据集上达到 state-of-the-art 性能。椭球的灵活性展示了优越的开放意图检测能力和在更多复杂开放场景下的广泛泛化潜力。 

---
