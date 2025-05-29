# UP-SLAM: Adaptively Structured Gaussian SLAM with Uncertainty Prediction in Dynamic Environments 

**Title (ZH)**: UP-SLAM：动态环境中基于不确定性预测的自适应结构化高斯SLAM 

**Authors**: Wancai Zheng, Linlin Ou, Jiajie He, Libo Zhou, Xinyi Yu, Yan Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.22335)  

**Abstract**: Recent 3D Gaussian Splatting (3DGS) techniques for Visual Simultaneous Localization and Mapping (SLAM) have significantly progressed in tracking and high-fidelity mapping. However, their sequential optimization framework and sensitivity to dynamic objects limit real-time performance and robustness in real-world scenarios. We present UP-SLAM, a real-time RGB-D SLAM system for dynamic environments that decouples tracking and mapping through a parallelized framework. A probabilistic octree is employed to manage Gaussian primitives adaptively, enabling efficient initialization and pruning without hand-crafted thresholds. To robustly filter dynamic regions during tracking, we propose a training-free uncertainty estimator that fuses multi-modal residuals to estimate per-pixel motion uncertainty, achieving open-set dynamic object handling without reliance on semantic labels. Furthermore, a temporal encoder is designed to enhance rendering quality. Concurrently, low-dimensional features are efficiently transformed via a shallow multilayer perceptron to construct DINO features, which are then employed to enrich the Gaussian field and improve the robustness of uncertainty prediction. Extensive experiments on multiple challenging datasets suggest that UP-SLAM outperforms state-of-the-art methods in both localization accuracy (by 59.8%) and rendering quality (by 4.57 dB PSNR), while maintaining real-time performance and producing reusable, artifact-free static maps in dynamic this http URL project: this https URL 

**Abstract (ZH)**: 实时RGB-D SLAM系统UP-SLAM：通过并行框架实现动态环境下的解耦追踪与建图 

---
# Towards Human-Like Trajectory Prediction for Autonomous Driving: A Behavior-Centric Approach 

**Title (ZH)**: 面向自主驾驶的人类like轨迹预测：一种行为导向的方法 

**Authors**: Haicheng Liao, Zhenning Li, Guohui Zhang, Keqiang Li, Chengzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21565)  

**Abstract**: Predicting the trajectories of vehicles is crucial for the development of autonomous driving (AD) systems, particularly in complex and dynamic traffic environments. In this study, we introduce HiT (Human-like Trajectory Prediction), a novel model designed to enhance trajectory prediction by incorporating behavior-aware modules and dynamic centrality measures. Unlike traditional methods that primarily rely on static graph structures, HiT leverages a dynamic framework that accounts for both direct and indirect interactions among traffic participants. This allows the model to capture the subtle yet significant influences of surrounding vehicles, enabling more accurate and human-like predictions. To evaluate HiT's performance, we conducted extensive experiments using diverse and challenging real-world datasets, including NGSIM, HighD, RounD, ApolloScape, and MoCAD++. The results demonstrate that HiT consistently outperforms other top models across multiple metrics, particularly excelling in scenarios involving aggressive driving behaviors. This research presents a significant step forward in trajectory prediction, offering a more reliable and interpretable approach for enhancing the safety and efficiency of fully autonomous driving systems. 

**Abstract (ZH)**: Human-like 轨迹预测对于自动驾驶系统的发展至关重要，特别是在复杂的动态交通环境中。在本研究中，我们介绍了HiT（Human-like Trajectory Prediction）模型，该模型通过引入行为感知模块和动态中心性度量来增强轨迹预测。与主要依赖静态图结构的传统方法不同，HiT 利用了一个动态框架，能够考虑交通参与者之间的直接和间接交互。这使得模型能够捕捉周围车辆的细微但重要的影响，从而实现更准确和具有人类特征的预测。为了评估HiT的性能，我们使用了多样且具有挑战性的现实世界数据集，包括NGSIM、HighD、RounD、ApolloScape和MoCAD++。结果表明，HiT 在多个指标上都优于其他顶级模型，特别是在涉及侵略性驾驶行为的场景中表现尤为出色。本研究在轨迹预测方面迈出了重要一步，提供了增强完全自动驾驶系统安全性和效率的更可靠和可解释的方法。 

---
# Zero-Shot 3D Visual Grounding from Vision-Language Models 

**Title (ZH)**: 零样本3D视觉定位从视觉-语言模型 

**Authors**: Rong Li, Shijie Li, Lingdong Kong, Xulei Yang, Junwei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22429)  

**Abstract**: 3D Visual Grounding (3DVG) seeks to locate target objects in 3D scenes using natural language descriptions, enabling downstream applications such as augmented reality and robotics. Existing approaches typically rely on labeled 3D data and predefined categories, limiting scalability to open-world settings. We present SeeGround, a zero-shot 3DVG framework that leverages 2D Vision-Language Models (VLMs) to bypass the need for 3D-specific training. To bridge the modality gap, we introduce a hybrid input format that pairs query-aligned rendered views with spatially enriched textual descriptions. Our framework incorporates two core components: a Perspective Adaptation Module that dynamically selects optimal viewpoints based on the query, and a Fusion Alignment Module that integrates visual and spatial signals to enhance localization precision. Extensive evaluations on ScanRefer and Nr3D confirm that SeeGround achieves substantial improvements over existing zero-shot baselines -- outperforming them by 7.7% and 7.1%, respectively -- and even rivals fully supervised alternatives, demonstrating strong generalization under challenging conditions. 

**Abstract (ZH)**: 3D视觉定位（3DVG）旨在使用自然语言描述在3D场景中定位目标物体，从而支持增强现实和机器人等下游应用。现有方法通常依赖于标注的3D数据和预定义的类别，限制了其在开放世界环境中的可扩展性。我们提出了一种零样本3DVG框架SeeGround，该框架利用2D视觉语言模型（VLMs）绕过了专门的3D训练需求。为了弥合模态差距，我们引入了一种混合输入格式，将查询对齐的渲染视图与空间丰富的文本描述配对。该框架包含两个核心组件：透视适配模块，根据查询动态选择最佳视点；融合对齐模块，将视觉和空间信号整合以提高定位精度。在ScanRefer和Nr3D上的广泛评估表明，SeeGround在零样本基准之上取得了显著改进，分别优于它们7.7%和7.1%，甚至媲美全监督替代方案，展示了其在挑战性条件下的强大泛化能力。 

---
# Visual Loop Closure Detection Through Deep Graph Consensus 

**Title (ZH)**: 通过深度图共识进行视觉环回闭合检测 

**Authors**: Martin Büchner, Liza Dahiya, Simon Dorer, Vipul Ramtekkar, Kenji Nishimiya, Daniele Cattaneo, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.21754)  

**Abstract**: Visual loop closure detection traditionally relies on place recognition methods to retrieve candidate loops that are validated using computationally expensive RANSAC-based geometric verification. As false positive loop closures significantly degrade downstream pose graph estimates, verifying a large number of candidates in online simultaneous localization and mapping scenarios is constrained by limited time and compute resources. While most deep loop closure detection approaches only operate on pairs of keyframes, we relax this constraint by considering neighborhoods of multiple keyframes when detecting loops. In this work, we introduce LoopGNN, a graph neural network architecture that estimates loop closure consensus by leveraging cliques of visually similar keyframes retrieved through place recognition. By propagating deep feature encodings among nodes of the clique, our method yields high-precision estimates while maintaining high recall. Extensive experimental evaluations on the TartanDrive 2.0 and NCLT datasets demonstrate that LoopGNN outperforms traditional baselines. Additionally, an ablation study across various keypoint extractors demonstrates that our method is robust, regardless of the type of deep feature encodings used, and exhibits higher computational efficiency compared to classical geometric verification baselines. We release our code, supplementary material, and keyframe data at this https URL. 

**Abstract (ZH)**: 视觉循回闭合检测传统上依赖于位置识别方法来检索候选循回闭合，并使用计算成本高的RANSAC基几何验证进行验证。由于误判的循回闭合显著降低后续姿态图估计精度，在在线 simultaneous localization and mapping 情景中验证大量候选闭合受到有限时间和计算资源的约束。虽然大多数深度循回闭合检测方法仅操作键帧对，我们通过在检测循回闭合时考虑多个键帧的邻域来放宽这一约束。在本文中，我们引入了 LoopGNN，这是一种图神经网络架构，通过利用通过位置识别检索到的视觉相似键帧的团来估计循回闭合共识。通过在团的节点之间传播深度特征编码，我们的方法能够提供高精度估计并保持高召回率。在 TartanDrive 2.0 和 NCLT 数据集上的广泛实验评估表明，LoopGNN 超过了传统基线。此外，我们在各种关键点提取器上的消融研究表明，无论使用何种深度特征编码，我们的方法都是稳健的，并且在计算效率上优于经典几何验证基线。我们在此 https://链接提供了我们的代码、补充材料和键帧数据。 

---
# Understanding the learned look-ahead behavior of chess neural networks 

**Title (ZH)**: 理解象棋神经网络学习的前瞻行为 

**Authors**: Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2505.21552)  

**Abstract**: We investigate the look-ahead capabilities of chess-playing neural networks, specifically focusing on the Leela Chess Zero policy network. We build on the work of Jenner et al. (2024) by analyzing the model's ability to consider future moves and alternative sequences beyond the immediate next move. Our findings reveal that the network's look-ahead behavior is highly context-dependent, varying significantly based on the specific chess position. We demonstrate that the model can process information about board states up to seven moves ahead, utilizing similar internal mechanisms across different future time steps. Additionally, we provide evidence that the network considers multiple possible move sequences rather than focusing on a single line of play. These results offer new insights into the emergence of sophisticated look-ahead capabilities in neural networks trained on strategic tasks, contributing to our understanding of AI reasoning in complex domains. Our work also showcases the effectiveness of interpretability techniques in uncovering cognitive-like processes in artificial intelligence systems. 

**Abstract (ZH)**: 我们研究棋弈神经网络的展望能力，特别聚焦于Leela Chess Zero策略网络。我们在Jenner等人（2024）的基础上，分析了模型超越立即下一步棋，考虑未来移动和替代序列的能力。我们的研究发现，网络的展望行为高度依赖于具体棋局的上下文，表现差异显著。我们证明该模型能够处理最多七步棋前的棋盘状态信息，并且在不同的未来时间步骤中使用类似的内部机制。此外，我们提供了证据表明，该网络在考虑多种可能的移动序列，而不仅仅是关注单一的行棋线路。这些结果为神经网络在战略任务训练后如何产生复杂的展望能力提供了新的见解，有助于我们对复杂领域中AI推理的理解。我们的工作还展示了可解释性技术在揭示人工智能系统中类似认知的过程方面的有效性。 

---
# PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion 

**Title (ZH)**: PRISM: 逐步细化与插入用于稀疏运动的视频数据集凝练 

**Authors**: Jaehyun Choi, Jiwan Hur, Gyojin Han, Jaemyung Yu, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.22564)  

**Abstract**: Video dataset condensation has emerged as a critical technique for addressing the computational challenges associated with large-scale video data processing in deep learning applications. While significant progress has been made in image dataset condensation, the video domain presents unique challenges due to the complex interplay between spatial content and temporal dynamics. This paper introduces PRISM, Progressive Refinement and Insertion for Sparse Motion, for video dataset condensation, a novel approach that fundamentally reconsiders how video data should be condensed. Unlike the previous method that separates static content from dynamic motion, our method preserves the essential interdependence between these elements. Our approach progressively refines and inserts frames to fully accommodate the motion in an action while achieving better performance but less storage, considering the relation of gradients for each frame. Extensive experiments across standard video action recognition benchmarks demonstrate that PRISM outperforms existing disentangled approaches while maintaining compact representations suitable for resource-constrained environments. 

**Abstract (ZH)**: 视频数据集凝练已成为应对深度学习应用中大规模视频数据处理所带来计算挑战的关键技术。尽管在图像数据集凝练方面取得了显著进展，但由于空间内容与时间动态之间的复杂交互作用，视频领域提出了独特的挑战。本文介绍了PRISM：逐步精炼与稀疏运动插入方法，这是一种全新的视频数据集凝练方法，从根本上重新考虑了视频数据应该如何凝练。与之前的方法将静态内容与动态运动分离不同，我们的方法保留了这些元素之间的基本相互依赖关系。本文的方法逐步精炼并插入帧以全面适应动作中的运动，同时实现更好的性能并减少存储需求，并考虑了每帧梯度的关系。广泛的实验表明，PRISM在标准视频动作识别基准上优于现有的解耦方法，同时保持适用于资源受限环境的紧凑表示。 

---
# Scaling-up Perceptual Video Quality Assessment 

**Title (ZH)**: 提升感知视频质量评估 

**Authors**: Ziheng Jia, Zicheng Zhang, Zeyu Zhang, Yingji Liang, Xiaorong Zhu, Chunyi Li, Jinliang Han, Haoning Wu, Bin Wang, Haoran Zhang, Guanyu Zhu, Qiyong Zhao, Xiaohong Liu, Guangtao Zhai, Xiongkuo Min  

**Link**: [PDF](https://arxiv.org/pdf/2505.22543)  

**Abstract**: The data scaling law has been shown to significantly enhance the performance of large multi-modal models (LMMs) across various downstream tasks. However, in the domain of perceptual video quality assessment (VQA), the potential of scaling law remains unprecedented due to the scarcity of labeled resources and the insufficient scale of datasets. To address this, we propose \textbf{OmniVQA}, an efficient framework designed to efficiently build high-quality, human-in-the-loop VQA multi-modal instruction databases (MIDBs). We then scale up to create \textbf{OmniVQA-Chat-400K}, the largest MIDB in the VQA field concurrently. Our focus is on the technical and aesthetic quality dimensions, with abundant in-context instruction data to provide fine-grained VQA knowledge. Additionally, we have built the \textbf{OmniVQA-MOS-20K} dataset to enhance the model's quantitative quality rating capabilities. We then introduce a \textbf{complementary} training strategy that effectively leverages the knowledge from datasets for quality understanding and quality rating tasks. Furthermore, we propose the \textbf{OmniVQA-FG (fine-grain)-Benchmark} to evaluate the fine-grained performance of the models. Our results demonstrate that our models achieve state-of-the-art performance in both quality understanding and rating tasks. 

**Abstract (ZH)**: 数据扩展规律已被证明能显著增强大规模多模态模型在各种下游任务中的性能。然而，在感知视频质量评估（VQA）领域，数据扩展规律的潜力尚未被充分发掘，主要原因在于标注资源稀缺和数据集规模不足。为解决这一问题，我们提出了**OmniVQA**，一种高效框架，旨在高效构建高质量、人机结合的VQA多模态指令数据集（MIDB）。随后，我们扩展规模创建了**OmniVQA-Chat-400K**，这是VQA领域目前最大的MIDB。我们重点关注技术和美学质量维度，并提供了丰富的上下文内指令数据，以提供详细的VQA知识。此外，我们建立了**OmniVQA-MOS-20K**数据集，以增强模型的定量质量评分能力。我们还提出了一种**互补**的训练策略，有效利用数据集知识进行质量理解和质量评分任务。此外，我们提出了**OmniVQA-FG（细粒度）基准**来评估模型的细粒度性能。我们的结果显示，我们的模型在质量理解和评分任务中均实现了最先进的性能。 

---
# NFR: Neural Feature-Guided Non-Rigid Shape Registration 

**Title (ZH)**: 神经特征引导的非刚性形状配准 

**Authors**: Puhua Jiang, Zhangquan Chen, Mingze Sun, Ruqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22445)  

**Abstract**: In this paper, we propose a novel learning-based framework for 3D shape registration, which overcomes the challenges of significant non-rigid deformation and partiality undergoing among input shapes, and, remarkably, requires no correspondence annotation during training. Our key insight is to incorporate neural features learned by deep learning-based shape matching networks into an iterative, geometric shape registration pipeline. The advantage of our approach is two-fold -- On one hand, neural features provide more accurate and semantically meaningful correspondence estimation than spatial features (e.g., coordinates), which is critical in the presence of large non-rigid deformations; On the other hand, the correspondences are dynamically updated according to the intermediate registrations and filtered by consistency prior, which prominently robustify the overall pipeline. Empirical results show that, with as few as dozens of training shapes of limited variability, our pipeline achieves state-of-the-art results on several benchmarks of non-rigid point cloud matching and partial shape matching across varying settings, but also delivers high-quality correspondences between unseen challenging shape pairs that undergo both significant extrinsic and intrinsic deformations, in which case neither traditional registration methods nor intrinsic methods work. 

**Abstract (ZH)**: 基于学习的三维形状配准新型框架：克服显著非刚性变形和部分性问题，无需训练对应标注 

---
# Can NeRFs See without Cameras? 

**Title (ZH)**: NeRFs能在没有相机的情况下看见吗？ 

**Authors**: Chaitanya Amballa, Sattwik Basu, Yu-Lin Wei, Zhijian Yang, Mehmet Ergezer, Romit Roy Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.22441)  

**Abstract**: Neural Radiance Fields (NeRFs) have been remarkably successful at synthesizing novel views of 3D scenes by optimizing a volumetric scene function. This scene function models how optical rays bring color information from a 3D object to the camera pixels. Radio frequency (RF) or audio signals can also be viewed as a vehicle for delivering information about the environment to a sensor. However, unlike camera pixels, an RF/audio sensor receives a mixture of signals that contain many environmental reflections (also called "multipath"). Is it still possible to infer the environment using such multipath signals? We show that with redesign, NeRFs can be taught to learn from multipath signals, and thereby "see" the environment. As a grounding application, we aim to infer the indoor floorplan of a home from sparse WiFi measurements made at multiple locations inside the home. Although a difficult inverse problem, our implicitly learnt floorplans look promising, and enables forward applications, such as indoor signal prediction and basic ray tracing. 

**Abstract (ZH)**: 神经辐射场（NeRFs）通过优化体绘制场景函数以生成3D场景的新视图取得了显著成功。该场景函数描述了光学射线如何将3D物体的颜色信息传递到相机像素。类似地，射频（RF）或音频信号也可以视为将环境信息传递给传感器的一种载体。然而，与相机像素不同，RF/音频传感器接收到的信号是包含多种环境反射的混合信号（也称为“多径”）。是否可以利用这类多径信号来推断环境信息？我们证明，通过对神经辐射场进行重新设计，可以使它们学会从多径信号中学习，从而“看到”环境。作为扎根应用，我们旨在通过室内多个位置的稀疏WiFi测量来推断房屋的平面布局。尽管这是一个棘手的逆问题，但我们隐式学习得到的平面布局具有前景，并可应用于室内信号预测和基本光线追踪等前景应用。 

---
# VME: A Satellite Imagery Dataset and Benchmark for Detecting Vehicles in the Middle East and Beyond 

**Title (ZH)**: VME：用于检测中东及其他地区车辆的遥感图像数据集及其基准测试 

**Authors**: Noora Al-Emadi, Ingmar Weber, Yin Yang, Ferda Ofli  

**Link**: [PDF](https://arxiv.org/pdf/2505.22353)  

**Abstract**: Detecting vehicles in satellite images is crucial for traffic management, urban planning, and disaster response. However, current models struggle with real-world diversity, particularly across different regions. This challenge is amplified by geographic bias in existing datasets, which often focus on specific areas and overlook regions like the Middle East. To address this gap, we present the Vehicles in the Middle East (VME) dataset, designed explicitly for vehicle detection in high-resolution satellite images from Middle Eastern countries. Sourced from Maxar, the VME dataset spans 54 cities across 12 countries, comprising over 4,000 image tiles and more than 100,000 vehicles, annotated using both manual and semi-automated methods. Additionally, we introduce the largest benchmark dataset for Car Detection in Satellite Imagery (CDSI), combining images from multiple sources to enhance global car detection. Our experiments demonstrate that models trained on existing datasets perform poorly on Middle Eastern images, while the VME dataset significantly improves detection accuracy in this region. Moreover, state-of-the-art models trained on CDSI achieve substantial improvements in global car detection. 

**Abstract (ZH)**: 中东地区卫星图像中的车辆检测对于交通管理、城市规划和灾害响应至关重要。然而，当前模型在应对现实世界中的多样化挑战，特别是在不同地区之间，表现不佳。现有数据集中的地理偏差进一步加剧了这一挑战，这些数据集往往集中在特定区域，忽视了如中东这样的地区。为解决这一问题，我们提出了中东车辆（VME）数据集，该数据集专门用于中东国家高分辨率卫星图像中的车辆检测。数据集来源于Maxar，覆盖12个国家的54个城市，包含超过4,000个图像瓦片和超过100,000辆车辆的标注，采用手动和半自动化方法进行标注。此外，我们还引入了最大的卫星图像中汽车检测基准数据集（CDSI），结合了多个来源的数据，以增强全球汽车检测能力。实验结果显示，现有的数据集训练的模型在中东图像上的性能不佳，而VME数据集显著改善了该地区的检测精度。此外，基于CDSI训练的最新模型在全球汽车检测方面取得了显著改进。 

---
# Versatile Cardiovascular Signal Generation with a Unified Diffusion Transformer 

**Title (ZH)**: 统一扩散变换器实现多功能心血管信号生成 

**Authors**: Zehua Chen, Yuyang Miao, Liyuan Wang, Luyun Fan, Danilo P. Mandic, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22306)  

**Abstract**: Cardiovascular signals such as photoplethysmography (PPG), electrocardiography (ECG), and blood pressure (BP) are inherently correlated and complementary, together reflecting the health of cardiovascular system. However, their joint utilization in real-time monitoring is severely limited by diverse acquisition challenges from noisy wearable recordings to burdened invasive procedures. Here we propose UniCardio, a multi-modal diffusion transformer that reconstructs low-quality signals and synthesizes unrecorded signals in a unified generative framework. Its key innovations include a specialized model architecture to manage the signal modalities involved in generation tasks and a continual learning paradigm to incorporate varying modality combinations. By exploiting the complementary nature of cardiovascular signals, UniCardio clearly outperforms recent task-specific baselines in signal denoising, imputation, and translation. The generated signals match the performance of ground-truth signals in detecting abnormal health conditions and estimating vital signs, even in unseen domains, while ensuring interpretability for human experts. These advantages position UniCardio as a promising avenue for advancing AI-assisted healthcare. 

**Abstract (ZH)**: 多模态扩散变换器UniCardio在心血管信号重建与合成中的应用研究 

---
# Neural Restoration of Greening Defects in Historical Autochrome Photographs Based on Purely Synthetic Data 

**Title (ZH)**: 基于纯合成数据的历史自动彩摄影像绿色缺陷神经恢复 

**Authors**: Saptarshi Neil Sinha, P. Julius Kuehn, Johannes Koppe, Arjan Kuijper, Michael Weinmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.22291)  

**Abstract**: The preservation of early visual arts, particularly color photographs, is challenged by deterioration caused by aging and improper storage, leading to issues like blurring, scratches, color bleeding, and fading defects. In this paper, we present the first approach for the automatic removal of greening color defects in digitized autochrome photographs. Our main contributions include a method based on synthetic dataset generation and the use of generative AI with a carefully designed loss function for the restoration of visual arts. To address the lack of suitable training datasets for analyzing greening defects in damaged autochromes, we introduce a novel approach for accurately simulating such defects in synthetic data. We also propose a modified weighted loss function for the ChaIR method to account for color imbalances between defected and non-defected areas. While existing methods struggle with accurately reproducing original colors and may require significant manual effort, our method allows for efficient restoration with reduced time requirements. 

**Abstract (ZH)**: 早期视觉艺术的保存，尤其是彩色照片，面临着因老化和不当存储导致的退化问题，这些问题包括模糊、刮痕、颜色溢出和褪色等缺陷。本文提出了首个针对数字化Autochrome照片中绿色颜色缺陷的自动去除方法。我们的主要贡献包括基于合成数据集生成的方法以及使用生成AI并通过精心设计的损失函数进行视觉艺术的修复。为了解决缺乏用于分析损坏Autochrome中绿色缺陷的适当训练数据集的问题，我们提出了一种新颖的方法来在合成数据中准确模拟此类缺陷。我们还提出了对ChaIR方法的修改加权损失函数，以考虑到缺陷区域和非缺陷区域之间的颜色不平衡。与现有方法相比，我们的方法能够更高效地进行修复，减少所需的时间，同时能够更准确地再现原始颜色。 

---
# FaceEditTalker: Interactive Talking Head Generation with Facial Attribute Editing 

**Title (ZH)**: FaceEditTalker: 基于面部属性编辑的交互式头部讲话生成 

**Authors**: Guanwen Feng, Zhiyuan Ma, Yunan Li, Junwei Jing, Jiahao Yang, Qiguang Miao  

**Link**: [PDF](https://arxiv.org/pdf/2505.22141)  

**Abstract**: Recent advances in audio-driven talking head generation have achieved impressive results in lip synchronization and emotional expression. However, they largely overlook the crucial task of facial attribute editing. This capability is crucial for achieving deep personalization and expanding the range of practical applications, including user-tailored digital avatars, engaging online education content, and brand-specific digital customer service. In these key domains, the flexible adjustment of visual attributes-such as hairstyle, accessories, and subtle facial features is essential for aligning with user preferences, reflecting diverse brand identities, and adapting to varying contextual demands. In this paper, we present FaceEditTalker, a unified framework that enables controllable facial attribute manipulation while generating high-quality, audio-synchronized talking head videos. Our method consists of two key components: an image feature space editing module, which extracts semantic and detail features and allows flexible control over attributes like expression, hairstyle, and accessories; and an audio-driven video generation module, which fuses these edited features with audio-guided facial landmarks to drive a diffusion-based generator. This design ensures temporal coherence, visual fidelity, and identity preservation across frames. Extensive experiments on public datasets demonstrate that our method outperforms state-of-the-art approaches in lip-sync accuracy, video quality, and attribute controllability. Project page: this https URL 

**Abstract (ZH)**: Recent advances in audio-driven talking head generation have achieved impressive results in lip synchronization and emotional expression, but largely overlook the crucial task of facial attribute editing. This capability is crucial for achieving deep personalization and expanding the range of practical applications, including user-tailored digital avatars, engaging online education content, and brand-specific digital customer service. In these key domains, the flexible adjustment of visual attributes such as hairstyle, accessories, and subtle facial features is essential for aligning with user preferences, reflecting diverse brand identities, and adapting to varying contextual demands. In this paper, we present FaceEditTalker, a unified framework that enables controllable facial attribute manipulation while generating high-quality, audio-synchronized talking head videos. Our method consists of two key components: an image feature space editing module, which extracts semantic and detail features and allows flexible control over attributes like expression, hairstyle, and accessories; and an audio-driven video generation module, which fuses these edited features with audio-guided facial landmarks to drive a diffusion-based generator. This design ensures temporal coherence, visual fidelity, and identity preservation across frames. Extensive experiments on public datasets demonstrate that our method outperforms state-of-the-art approaches in lip-sync accuracy, video quality, and attribute controllability. 

---
# Real-Time Blind Defocus Deblurring for Earth Observation: The IMAGIN-e Mission Approach 

**Title (ZH)**: 地球观测中的实时盲焦blur去模糊：IMAGIN-e 任务方法 

**Authors**: Alejandro D. Mousist  

**Link**: [PDF](https://arxiv.org/pdf/2505.22128)  

**Abstract**: This work addresses mechanical defocus in Earth observation images from the IMAGIN-e mission aboard the ISS, proposing a blind deblurring approach adapted to space-based edge computing constraints. Leveraging Sentinel-2 data, our method estimates the defocus kernel and trains a restoration model within a GAN framework, effectively operating without reference images.
On Sentinel-2 images with synthetic degradation, SSIM improved by 72.47% and PSNR by 25.00%, confirming the model's ability to recover lost details when the original clean image is known. On IMAGIN-e, where no reference images exist, perceptual quality metrics indicate a substantial enhancement, with NIQE improving by 60.66% and BRISQUE by 48.38%, validating real-world onboard restoration. The approach is currently deployed aboard the IMAGIN-e mission, demonstrating its practical application in an operational space environment.
By efficiently handling high-resolution images under edge computing constraints, the method enables applications such as water body segmentation and contour detection while maintaining processing viability despite resource limitations. 

**Abstract (ZH)**: 本研究针对IMAGIN-e任务在ISS上获取的地球观测图像中的机械失焦问题，提出了一种适应基于空间的边缘计算约束的盲去模糊方法。利用Sentinel-2数据，我们的方法估计失焦核并在生成对抗网络（GAN）框架中训练复原模型，无需参考图像即可有效运行。在带有合成退化效果的Sentinel-2图像上，SSIM提高了72.47%，PSNR提高了25.00%，证实了该模型在已知原始清晰图像时恢复丢失细节的能力。在没有参考图像的IMAGIN-e任务中，感知质量指标显示有显著提升，NIQE提高了60.66%，BRISQUE提高了48.38%，验证了在实际空间环境中的在轨复原效果。该方法已在IMAGIN-e任务中部署，展示了其在资源受限的运行空间环境中的实际应用能力。通过在边缘计算约束下高效处理高分辨率图像，该方法在资源受限的情况下仍能保持分割水体和检测轮廓的应用性能。 

---
# EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance 

**Title (ZH)**: EPiC: 有效的视频相机控制学习与精确锚视频指导 

**Authors**: Zun Wang, Jaemin Cho, Jialu Li, Han Lin, Jaehong Yoon, Yue Zhang, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2505.21876)  

**Abstract**: Recent approaches on 3D camera control in video diffusion models (VDMs) often create anchor videos to guide diffusion models as a structured prior by rendering from estimated point clouds following annotated camera trajectories. However, errors inherent in point cloud estimation often lead to inaccurate anchor videos. Moreover, the requirement for extensive camera trajectory annotations further increases resource demands. To address these limitations, we introduce EPiC, an efficient and precise camera control learning framework that automatically constructs high-quality anchor videos without expensive camera trajectory annotations. Concretely, we create highly precise anchor videos for training by masking source videos based on first-frame visibility. This approach ensures high alignment, eliminates the need for camera trajectory annotations, and thus can be readily applied to any in-the-wild video to generate image-to-video (I2V) training pairs. Furthermore, we introduce Anchor-ControlNet, a lightweight conditioning module that integrates anchor video guidance in visible regions to pretrained VDMs, with less than 1% of backbone model parameters. By combining the proposed anchor video data and ControlNet module, EPiC achieves efficient training with substantially fewer parameters, training steps, and less data, without requiring modifications to the diffusion model backbone typically needed to mitigate rendering misalignments. Although being trained on masking-based anchor videos, our method generalizes robustly to anchor videos made with point clouds during inference, enabling precise 3D-informed camera control. EPiC achieves SOTA performance on RealEstate10K and MiraData for I2V camera control task, demonstrating precise and robust camera control ability both quantitatively and qualitatively. Notably, EPiC also exhibits strong zero-shot generalization to video-to-video scenarios. 

**Abstract (ZH)**: Recent Approaches on 3D Camera Control in Video Diffusion Models (VDMs) Often Create Anchor Videos to Guide Diffusion Models as a Structured Prior by Rendering from Estimated Point Clouds Following Annotated Camera Trajectories: Introducing EPiC, an Efficient and Precise Camera Control Learning Framework Without Expensive Camera Trajectory Annotations 

---
# Rethinking Gradient-based Adversarial Attacks on Point Cloud Classification 

**Title (ZH)**: 基于梯度的点云分类对抗攻击再思考 

**Authors**: Jun Chen, Xinke Li, Mingyue Xu, Tianrui Li, Chongshou Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21854)  

**Abstract**: Gradient-based adversarial attacks have become a dominant approach for evaluating the robustness of point cloud classification models. However, existing methods often rely on uniform update rules that fail to consider the heterogeneous nature of point clouds, resulting in excessive and perceptible perturbations. In this paper, we rethink the design of gradient-based attacks by analyzing the limitations of conventional gradient update mechanisms and propose two new strategies to improve both attack effectiveness and imperceptibility. First, we introduce WAAttack, a novel framework that incorporates weighted gradients and an adaptive step-size strategy to account for the non-uniform contribution of points during optimization. This approach enables more targeted and subtle perturbations by dynamically adjusting updates according to the local structure and sensitivity of each point. Second, we propose SubAttack, a complementary strategy that decomposes the point cloud into subsets and focuses perturbation efforts on structurally critical regions. Together, these methods represent a principled rethinking of gradient-based adversarial attacks for 3D point cloud classification. Extensive experiments demonstrate that our approach outperforms state-of-the-art baselines in generating highly imperceptible adversarial examples. Code will be released upon paper acceptance. 

**Abstract (ZH)**: 基于梯度的对抗攻击已成为评估点云分类模型鲁棒性的主导方法。然而，现有方法往往依赖于统一的更新规则，未能考虑点云的异质性，导致产生过多且可感知的扰动。本文重新思考基于梯度的攻击设计，分析传统梯度更新机制的局限性，并提出两种新策略以提高攻击的有效性和不可感知性。首先，我们引入WAAttack，这是一种新颖的框架，结合了加权梯度和自适应步长策略，以反映优化过程中各点的非均匀贡献。此方法通过根据每个点的局部结构和敏感性动态调整更新来实现更针对性和细微的扰动。其次，我们提出SubAttack，这是一种互补策略，将点云分解为子集，并将扰动努力集中在结构关键区域。这两种方法代表了对3D点云分类中基于梯度的对抗攻击的理论性重新思考。 extensive实验表明，我们的方法在生成高度不可感知的对抗示例方面优于现有最先进的基线。论文接受后将发布代码。 

---
# RePaViT: Scalable Vision Transformer Acceleration via Structural Reparameterization on Feedforward Network Layers 

**Title (ZH)**: RePaViT: 通过前向网络层结构重参数化实现可扩展的视觉变压器加速 

**Authors**: Xuwei Xu, Yang Li, Yudong Chen, Jiajun Liu, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21847)  

**Abstract**: We reveal that feedforward network (FFN) layers, rather than attention layers, are the primary contributors to Vision Transformer (ViT) inference latency, with their impact signifying as model size increases. This finding highlights a critical opportunity for optimizing the efficiency of large-scale ViTs by focusing on FFN layers. In this work, we propose a novel channel idle mechanism that facilitates post-training structural reparameterization for efficient FFN layers during testing. Specifically, a set of feature channels remains idle and bypasses the nonlinear activation function in each FFN layer, thereby forming a linear pathway that enables structural reparameterization during inference. This mechanism results in a family of ReParameterizable Vision Transformers (RePaViTs), which achieve remarkable latency reductions with acceptable sacrifices (sometimes gains) in accuracy across various ViTs. The benefits of our method scale consistently with model sizes, demonstrating greater speed improvements and progressively narrowing accuracy gaps or even higher accuracies on larger models. In particular, RePa-ViT-Large and RePa-ViT-Huge enjoy 66.8% and 68.7% speed-ups with +1.7% and +1.1% higher top-1 accuracies under the same training strategy, respectively. RePaViT is the first to employ structural reparameterization on FFN layers to expedite ViTs to our best knowledge, and we believe that it represents an auspicious direction for efficient ViTs. Source code is available at this https URL. 

**Abstract (ZH)**: 我们揭示了前馈网络（FFN）层而非注意力层是决定 Vision Transformer（ViT）推理延迟的主要因素，其影响随模型规模增加而愈加显著。这一发现突显了通过专注于优化 FFN 层来优化大规模 ViT 效率的关键机会。在本文中，我们提出了一种新型的通道闲置机制，该机制在测试期间促进了高效 FFN 层的结构重参数化。具体而言，在每个 FFN 层中有一组特征通道保持闲置并绕过非线性激活函数，从而形成一条线性路径，能够在推理期间进行结构重参数化。该机制导致了一类重参数化 Vision Transformer（RePaViTs），它们在不同 ViT 中实现了显著的延迟减少，同时在准确率方面有所折中（有时有所提高）。我们的方法的好处随着模型规模的增加而一致地增强，显示出更大的加速性能，并且在更大模型中逐渐缩小甚至超越了准确性差距。特别是，RePa-ViT-Large 和 RePa-ViT-Huge 分别在相同训练策略下获得了 66.8% 和 68.7% 的加速，同时 top-1 准确率分别提高了 1.7% 和 1.1%。据我们所知，RePaViT 是首次在 FFN 层上采用结构重参数化来加速 ViTs 的方法，我们认为这代表了高效 ViTs 的前景良好的方向。源代码可在以下网址获取。 

---
# FRAMES-VQA: Benchmarking Fine-Tuning Robustness across Multi-Modal Shifts in Visual Question Answering 

**Title (ZH)**: FRAMES-VQA：跨多模态变化的视觉问答 fine-tuning 稳定性基准评估 

**Authors**: Chengyue Huang, Brisa Maneechotesuwan, Shivang Chopra, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2505.21755)  

**Abstract**: Visual question answering (VQA) systems face significant challenges when adapting to real-world data shifts, especially in multi-modal contexts. While robust fine-tuning strategies are essential for maintaining performance across in-distribution (ID) and out-of-distribution (OOD) scenarios, current evaluation settings are primarily unimodal or particular to some types of OOD, offering limited insight into the complexities of multi-modal contexts. In this work, we propose a new benchmark FRAMES-VQA (Fine-Tuning Robustness across Multi-Modal Shifts in VQA) for evaluating robust fine-tuning for VQA tasks. We utilize ten existing VQA benchmarks, including VQAv2, IV-VQA, VQA-CP, OK-VQA and others, and categorize them into ID, near and far OOD datasets covering uni-modal, multi-modal and adversarial distribution shifts. We first conduct a comprehensive comparison of existing robust fine-tuning methods. We then quantify the distribution shifts by calculating the Mahalanobis distance using uni-modal and multi-modal embeddings extracted from various models. Further, we perform an extensive analysis to explore the interactions between uni- and multi-modal shifts as well as modality importance for ID and OOD samples. These analyses offer valuable guidance on developing more robust fine-tuning methods to handle multi-modal distribution shifts. The code is available at this https URL . 

**Abstract (ZH)**: 跨模态视觉问答(VQA)系统在适应实际数据变换时面临重大挑战，特别是在多模态背景下。虽然强大的微调策略对于在分布内(ID)和分布外(OOD)场景下维持性能至关重要，但当前的评估设置主要为单模态或特定类型的OOD，提供的多模态背景下的复杂性洞察有限。在本工作中，我们提出了一种新的基准FRAMES-VQA（跨模态变换下的VQA稳健微调），用于评估VQA任务的稳健微调。我们利用十个现有的VQA基准，包括VQAv2、IV-VQA、VQA-CP、OK-VQA等，并将它们分类为分布内、近分布外和远分布外数据集，覆盖单模态、多模态和对抗分布变换。我们首先进行现有稳健微调方法的全面比较。然后，通过计算来自不同模型的一模态和多模态嵌入的马哈拉诺比斯距离，定量评估分布变换。进一步地，我们进行了广泛的分析以探索一模态和多模态变换及其模态重要性之间的相互作用，特别是对于分布内和分布外样本。这些分析为开发处理多模态分布变换的更稳健微调方法提供了宝贵指导。代码可在以下链接获取。 

---
# Learning to See More: UAS-Guided Super-Resolution of Satellite Imagery for Precision Agriculture 

**Title (ZH)**: 学习看到更多：基于无人机引导的卫星图像超分辨率技术在精准农业中的应用 

**Authors**: Arif Masrur, Peder A. Olsen, Paul R. Adler, Carlan Jackson, Matthew W. Myers, Nathan Sedghi, Ray R. Weil  

**Link**: [PDF](https://arxiv.org/pdf/2505.21746)  

**Abstract**: Unmanned Aircraft Systems (UAS) and satellites are key data sources for precision agriculture, yet each presents trade-offs. Satellite data offer broad spatial, temporal, and spectral coverage but lack the resolution needed for many precision farming applications, while UAS provide high spatial detail but are limited by coverage and cost, especially for hyperspectral data. This study presents a novel framework that fuses satellite and UAS imagery using super-resolution methods. By integrating data across spatial, spectral, and temporal domains, we leverage the strengths of both platforms cost-effectively. We use estimation of cover crop biomass and nitrogen (N) as a case study to evaluate our approach. By spectrally extending UAS RGB data to the vegetation red edge and near-infrared regions, we generate high-resolution Sentinel-2 imagery and improve biomass and N estimation accuracy by 18% and 31%, respectively. Our results show that UAS data need only be collected from a subset of fields and time points. Farmers can then 1) enhance the spectral detail of UAS RGB imagery; 2) increase the spatial resolution by using satellite data; and 3) extend these enhancements spatially and across the growing season at the frequency of the satellite flights. Our SRCNN-based spectral extension model shows considerable promise for model transferability over other cropping systems in the Upper and Lower Chesapeake Bay regions. Additionally, it remains effective even when cloud-free satellite data are unavailable, relying solely on the UAS RGB input. The spatial extension model produces better biomass and N predictions than models built on raw UAS RGB images. Once trained with targeted UAS RGB data, the spatial extension model allows farmers to stop repeated UAS flights. While we introduce super-resolution advances, the core contribution is a lightweight and scalable system for affordable on-farm use. 

**Abstract (ZH)**: 无人航空系统（UAS）和卫星在精准农业中的关键数据来源及其权衡：一种使用超分辨率方法融合UAS和卫星成像的新框架 

---
# STA-Risk: A Deep Dive of Spatio-Temporal Asymmetries for Breast Cancer Risk Prediction 

**Title (ZH)**: STA-Risk: 胸部癌变风险时空不对称性的深入探讨 

**Authors**: Zhengbo Zhou, Dooman Arefan, Margarita Zuley, Jules Sumkin, Shandong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21699)  

**Abstract**: Predicting the risk of developing breast cancer is an important clinical tool to guide early intervention and tailoring personalized screening strategies. Early risk models have limited performance and recently machine learning-based analysis of mammogram images showed encouraging risk prediction effects. These models however are limited to the use of a single exam or tend to overlook nuanced breast tissue evolvement in spatial and temporal details of longitudinal imaging exams that are indicative of breast cancer risk. In this paper, we propose STA-Risk (Spatial and Temporal Asymmetry-based Risk Prediction), a novel Transformer-based model that captures fine-grained mammographic imaging evolution simultaneously from bilateral and longitudinal asymmetries for breast cancer risk prediction. STA-Risk is innovative by the side encoding and temporal encoding to learn spatial-temporal asymmetries, regulated by a customized asymmetry loss. We performed extensive experiments with two independent mammogram datasets and achieved superior performance than four representative SOTA models for 1- to 5-year future risk prediction. Source codes will be released upon publishing of the paper. 

**Abstract (ZH)**: 基于空间和时间不对称性的Transformer风险预测模型(STA-Risk)：乳腺癌风险预测 

---
# Any-to-Bokeh: One-Step Video Bokeh via Multi-Plane Image Guided Diffusion 

**Title (ZH)**: 任意到景深效果：基于多平面图像引导扩散的一步视频景深生成 

**Authors**: Yang Yang, Siming Zheng, Jinwei Chen, Boxi Wu, Xiaofei He, Deng Cai, Bo Li, Peng-Tao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21593)  

**Abstract**: Recent advances in diffusion based editing models have enabled realistic camera simulation and image-based bokeh, but video bokeh remains largely unexplored. Existing video editing models cannot explicitly control focus planes or adjust bokeh intensity, limiting their applicability for controllable optical effects. Moreover, naively extending image-based bokeh methods to video often results in temporal flickering and unsatisfactory edge blur transitions due to the lack of temporal modeling and generalization capability. To address these challenges, we propose a novel one-step video bokeh framework that converts arbitrary input videos into temporally coherent, depth-aware bokeh effects. Our method leverages a multi-plane image (MPI) representation constructed through a progressively widening depth sampling function, providing explicit geometric guidance for depth-dependent blur synthesis. By conditioning a single-step video diffusion model on MPI layers and utilizing the strong 3D priors from pre-trained models such as Stable Video Diffusion, our approach achieves realistic and consistent bokeh effects across diverse scenes. Additionally, we introduce a progressive training strategy to enhance temporal consistency, depth robustness, and detail preservation. Extensive experiments demonstrate that our method produces high-quality, controllable bokeh effects and achieves state-of-the-art performance on multiple evaluation benchmarks. 

**Abstract (ZH)**: Recent Advances in Diffusion-Based Editing Models Have Enabled Realistic Camera Simulation and Image-Based Bokeh, but Video Bokeh Remains largely Unexplored 

---
# A Novel Convolutional Neural Network-Based Framework for Complex Multiclass Brassica Seed Classification 

**Title (ZH)**: 基于卷积神经网络的复杂多类 Brassica 种子分类新型框架 

**Authors**: Elhoucine Elfatimia, Recep Eryigitb, Lahcen Elfatimi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21558)  

**Abstract**: Agricultural research has accelerated in recent years, yet farmers often lack the time and resources for on-farm research due to the demands of crop production and farm operations. Seed classification offers valuable insights into quality control, production efficiency, and impurity detection. Early identification of seed types is critical to reducing the cost and risk associated with field emergence, which can lead to yield losses or disruptions in downstream processes like harvesting. Seed sampling supports growers in monitoring and managing seed quality, improving precision in determining seed purity levels, guiding management adjustments, and enhancing yield estimations. This study proposes a novel convolutional neural network (CNN)-based framework for the efficient classification of ten common Brassica seed types. The approach addresses the inherent challenge of texture similarity in seed images using a custom-designed CNN architecture. The model's performance was evaluated against several pre-trained state-of-the-art architectures, with adjustments to layer configurations for optimized classification. Experimental results using our collected Brassica seed dataset demonstrate that the proposed model achieved a high accuracy rate of 93 percent. 

**Abstract (ZH)**: 近年来，农业研究加速发展，但由于作物生产和农场运营的需求，农民往往缺乏进行农场研究的时间和资源。种子分类为质量控制、生产效率和杂质检测提供了宝贵见解。早期识别种子类型对于降低田间出苗的成本和风险至关重要，这可以减少产量损失或下游过程如收获中的中断。种子采样支持种植者监控和管理种子质量，提高了种子纯度水平的精确度，指导管理调整，并提升产量估计。本研究提出了一种新型卷积神经网络（CNN）基于框架，用于高效分类十种常见的 Brassica 种子类型。该方法通过自定义设计的 CNN 架构解决了种子图像中固有的纹理相似性挑战。通过与多种预训练的先进架构评估模型性能，并对层配置进行调整以优化分类。使用我们收集的 Brassica 种子数据集的实验证明，所提模型的准确率为 93%。 

---
# Analytical Calculation of Weights Convolutional Neural Network 

**Title (ZH)**: 权重卷积神经网络的解析计算 

**Authors**: Polad Geidarov  

**Link**: [PDF](https://arxiv.org/pdf/2505.21557)  

**Abstract**: This paper presents an algorithm for analytically calculating the weights and thresholds of convolutional neural networks (CNNs) without using standard training procedures. The algorithm enables the determination of CNN parameters based on just 10 selected images from the MNIST dataset, each representing a digit from 0 to 9. As part of the method, the number of channels in CNN layers is also derived analytically. A software module was implemented in C++ Builder, and a series of experiments were conducted using the MNIST dataset. Results demonstrate that the analytically computed CNN can recognize over half of 1000 handwritten digit images without any training, achieving inference in fractions of a second. These findings suggest that CNNs can be constructed and applied directly for classification tasks without training, using purely analytical computation of weights. 

**Abstract (ZH)**: 本文提出了一种在不使用标准训练程序的情况下，通过解析方法计算卷积神经网络（CNN）权重和阈值的算法。该算法能够在仅使用MNIST数据集中10张代表0至9数字的手写图像的情况下确定CNN参数。方法还包括解析确定CNN层中的通道数。在C++ Builder中实现了一个软件模块，并使用MNIST数据集进行了多项实验。结果表明，解析计算得到的CNN可以在没有任何训练的情况下识别超过1000张手写数字图像中的半数，并在几分之一秒内完成推理。这些发现表明，可以通过纯解析计算权重来构建和直接应用于分类任务的CNN，而无需训练。 

---
# DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers 

**Title (ZH)**: DiffDecompose: Alpha-合成图像逐层分解的扩散变换器方法 

**Authors**: Zitong Wang, Hang Zhao, Qianyu Zhou, Xuequan Lu, Xiangtai Li, Yiren Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.21541)  

**Abstract**: Diffusion models have recently motivated great success in many generation tasks like object removal. Nevertheless, existing image decomposition methods struggle to disentangle semi-transparent or transparent layer occlusions due to mask prior dependencies, static object assumptions, and the lack of datasets. In this paper, we delve into a novel task: Layer-Wise Decomposition of Alpha-Composited Images, aiming to recover constituent layers from single overlapped images under the condition of semi-transparent/transparent alpha layer non-linear occlusion. To address challenges in layer ambiguity, generalization, and data scarcity, we first introduce AlphaBlend, the first large-scale and high-quality dataset for transparent and semi-transparent layer decomposition, supporting six real-world subtasks (e.g., translucent flare removal, semi-transparent cell decomposition, glassware decomposition). Building on this dataset, we present DiffDecompose, a diffusion Transformer-based framework that learns the posterior over possible layer decompositions conditioned on the input image, semantic prompts, and blending type. Rather than regressing alpha mattes directly, DiffDecompose performs In-Context Decomposition, enabling the model to predict one or multiple layers without per-layer supervision, and introduces Layer Position Encoding Cloning to maintain pixel-level correspondence across layers. Extensive experiments on the proposed AlphaBlend dataset and public LOGO dataset verify the effectiveness of DiffDecompose. The code and dataset will be available upon paper acceptance. Our code will be available at: this https URL. 

**Abstract (ZH)**: 基于Alpha混合图像的层级分解：一种扩散变换器框架 

---
# Equivariant Flow Matching for Point Cloud Assembly 

**Title (ZH)**: 点云装配的等变流动匹配 

**Authors**: Ziming Wang, Nan Xue, Rebecka Jörnsten  

**Link**: [PDF](https://arxiv.org/pdf/2505.21539)  

**Abstract**: The goal of point cloud assembly is to reconstruct a complete 3D shape by aligning multiple point cloud pieces. This work presents a novel equivariant solver for assembly tasks based on flow matching models. We first theoretically show that the key to learning equivariant distributions via flow matching is to learn related vector fields. Based on this result, we propose an assembly model, called equivariant diffusion assembly (Eda), which learns related vector fields conditioned on the input pieces. We further construct an equivariant path for Eda, which guarantees high data efficiency of the training process. Our numerical results show that Eda is highly competitive on practical datasets, and it can even handle the challenging situation where the input pieces are non-overlapped. 

**Abstract (ZH)**: 点云拼接的目标是通过对齐多个点云片段来重构完整的3D形状。本文提出了一种基于流匹配模型的新型等变求解器用于拼接任务。我们首先从理论上证明了通过流匹配学习等变分布的关键在于学习相关向量场。基于这一结果，我们提出了一种称为等变扩散拼接（Eda）的拼接模型，该模型在输入片段的条件下学习相关向量场。我们进一步为Eda构建了一种等变路径，以保证训练过程的高度数据效率。我们的数值结果表明，Eda在实际数据集上具有高度竞争力，并且即使在输入片段不重叠的情况下也能处理具有挑战性的场景。 

---
# Caption This, Reason That: VLMs Caught in the Middle 

**Title (ZH)**: 给这幅图配上 captions，解释那个原因：VLMs 处在中间位置。 

**Authors**: Zihan Weng, Lucas Gomez, Taylor Whittington Webb, Pouya Bashivan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21538)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable progress in visual understanding in recent years. Yet, they still lag behind human capabilities in specific visual tasks such as counting or relational reasoning. To understand the underlying limitations, we adopt methodologies from cognitive science, analyzing VLM performance along core cognitive axes: Perception, Attention, and Memory. Using a suite of tasks targeting these abilities, we evaluate state-of-the-art VLMs, including GPT-4o. Our analysis reveals distinct cognitive profiles: while advanced models approach ceiling performance on some tasks (e.g. category identification), a significant gap persists, particularly in tasks requiring spatial understanding or selective attention. Investigating the source of these failures and potential methods for improvement, we employ a vision-text decoupling analysis, finding that models struggling with direct visual reasoning show marked improvement when reasoning over their own generated text captions. These experiments reveal a strong need for improved VLM Chain-of-Thought (CoT) abilities, even in models that consistently exceed human performance. Furthermore, we demonstrate the potential of targeted fine-tuning on composite visual reasoning tasks and show that fine-tuning smaller VLMs substantially improves core cognitive abilities. While this improvement does not translate to large enhancements on challenging, out-of-distribution benchmarks, we show broadly that VLM performance on our datasets strongly correlates with performance on these other benchmarks. Our work provides a detailed analysis of VLM cognitive strengths and weaknesses and identifies key bottlenecks in simultaneous perception and reasoning while also providing an effective and simple solution. 

**Abstract (ZH)**: Vision-Language 模型在近年来展示了在视觉理解方面的显著进步。然而，在如计数或关系推理等特定视觉任务中，它们仍落后于人类能力。为了理解这些潜在限制，我们采用了来自认知科学的方法，从感知、注意和记忆这三大核心认知轴分析 Vision-Language 模型的表现。利用一系列针对这些能力的测试任务，我们评估了当前最先进的 Vision-Language 模型，包括 GPT-4o。我们的分析揭示了不同的认知特征：尽管高级模型在某些任务（如类别识别）上接近天花板水平，但在要求空间理解或选择性注意的任务上仍存在显著差距。我们调查了这些失败的根源以及潜在的改进方法，并采用视觉-文本解耦分析发现，处理直接视觉推理困难的模型在其生成的文本描述上进行推理时表现明显改善。这些实验揭示了即使在模型持续超越人类表现的情况下，也需要提高 Vision-Language 模型思维链（CoT）能力的强烈需求。此外，我们展示了针对复合视觉推理任务的微调潜力，并证明小型模型的微调显著增强了核心认知能力。虽然这种改进在挑战性的离分布基准测试上并未转化为显著提升，但我们表明，在我们的数据集上 Vision-Language 模型的表现与这些其他基准测试的性能之间有显著的相关性。我们的研究详细分析了 Vision-Language 模型的认知强项和弱点，并识别了同时感知和推理中的关键瓶颈，同时还提供了一个有效且简单的解决方案。 

---
# UniDB++: Fast Sampling of Unified Diffusion Bridge 

**Title (ZH)**: UniDB++: 统一扩散桥的快速采样 

**Authors**: Mokai Pan, Kaizhen Zhu, Yuexin Ma, Yanwei Fu, Jingyi Yu, Jingya Wang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21528)  

**Abstract**: Diffusion Bridges enable transitions between arbitrary distributions, with the Unified Diffusion Bridge (UniDB) framework achieving high-fidelity image generation via a Stochastic Optimal Control (SOC) formulation. However, UniDB's reliance on iterative Euler sampling methods results in slow, computationally expensive inference, while existing acceleration techniques for diffusion or diffusion bridge models fail to address its unique challenges: missing terminal mean constraints and SOC-specific penalty coefficients in its SDEs. We present UniDB++, a training-free sampling algorithm that significantly improves upon these limitations. The method's key advancement comes from deriving exact closed-form solutions for UniDB's reverse-time SDEs, effectively reducing the error accumulation inherent in Euler approximations and enabling high-quality generation with up to 20$\times$ fewer sampling steps. This method is further complemented by replacing conventional noise prediction with a more stable data prediction model, along with an SDE-Corrector mechanism that maintains perceptual quality for low-step regimes (5-10 steps). Additionally, we demonstrate that UniDB++ aligns with existing diffusion bridge acceleration methods by evaluating their update rules, and UniDB++ can recover DBIMs as special cases under some theoretical conditions. Experiments demonstrate UniDB++'s state-of-the-art performance in image restoration tasks, outperforming Euler-based methods in fidelity and speed while reducing inference time significantly. This work bridges the gap between theoretical generality and practical efficiency in SOC-driven diffusion bridge models. Our code is available at this https URL. 

**Abstract (ZH)**: 统一扩散桥梁增强版（UniDB++）：通过精确反向时间SDE解提高采样效率 

---
