# Prompt-to-Product: Generative Assembly via Bimanual Manipulation 

**Title (ZH)**: Prompt-to-Product: 生成组装通过双臂操作 

**Authors**: Ruixuan Liu, Philip Huang, Ava Pun, Kangle Deng, Shobhit Aggarwal, Kevin Tang, Michelle Liu, Deva Ramanan, Jun-Yan Zhu, Jiaoyang Li, Changliu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.21063)  

**Abstract**: Creating assembly products demands significant manual effort and expert knowledge in 1) designing the assembly and 2) constructing the product. This paper introduces Prompt-to-Product, an automated pipeline that generates real-world assembly products from natural language prompts. Specifically, we leverage LEGO bricks as the assembly platform and automate the process of creating brick assembly structures. Given the user design requirements, Prompt-to-Product generates physically buildable brick designs, and then leverages a bimanual robotic system to construct the real assembly products, bringing user imaginations into the real world. We conduct a comprehensive user study, and the results demonstrate that Prompt-to-Product significantly lowers the barrier and reduces manual effort in creating assembly products from imaginative ideas. 

**Abstract (ZH)**: 基于自然语言提示的自动装配产品生成pipeline降低了从想象概念创建装配产品的门槛并减少了手动 effort。 

---
# SPGrasp: Spatiotemporal Prompt-driven Grasp Synthesis in Dynamic Scenes 

**Title (ZH)**: SPGrasp: 动态场景中基于时空提示的抓取合成 

**Authors**: Yunpeng Mei, Hongjie Cao, Yinqiu Xia, Wei Xiao, Zhaohan Feng, Gang Wang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.20547)  

**Abstract**: Real-time interactive grasp synthesis for dynamic objects remains challenging as existing methods fail to achieve low-latency inference while maintaining promptability. To bridge this gap, we propose SPGrasp (spatiotemporal prompt-driven dynamic grasp synthesis), a novel framework extending segment anything model v2 (SAMv2) for video stream grasp estimation. Our core innovation integrates user prompts with spatiotemporal context, enabling real-time interaction with end-to-end latency as low as 59 ms while ensuring temporal consistency for dynamic objects. In benchmark evaluations, SPGrasp achieves instance-level grasp accuracies of 90.6% on OCID and 93.8% on Jacquard. On the challenging GraspNet-1Billion dataset under continuous tracking, SPGrasp achieves 92.0% accuracy with 73.1 ms per-frame latency, representing a 58.5% reduction compared to the prior state-of-the-art promptable method RoG-SAM while maintaining competitive accuracy. Real-world experiments involving 13 moving objects demonstrate a 94.8% success rate in interactive grasping scenarios. These results confirm SPGrasp effectively resolves the latency-interactivity trade-off in dynamic grasp synthesis. Code is available at this https URL. 

**Abstract (ZH)**: 基于时空提示的动态抓取实时交互合成（SPGrasp）：一种扩展segment anything model v2 (SAMv2)的新型框架 

---
# COMETH: Convex Optimization for Multiview Estimation and Tracking of Humans 

**Title (ZH)**: COMETH：多视图人类估计与跟踪的凸优化方法 

**Authors**: Enrico Martini, Ho Jin Choi, Nadia Figueroa, Nicola Bombieri  

**Link**: [PDF](https://arxiv.org/pdf/2508.20920)  

**Abstract**: In the era of Industry 5.0, monitoring human activity is essential for ensuring both ergonomic safety and overall well-being. While multi-camera centralized setups improve pose estimation accuracy, they often suffer from high computational costs and bandwidth requirements, limiting scalability and real-time applicability. Distributing processing across edge devices can reduce network bandwidth and computational load. On the other hand, the constrained resources of edge devices lead to accuracy degradation, and the distribution of computation leads to temporal and spatial inconsistencies. We address this challenge by proposing COMETH (Convex Optimization for Multiview Estimation and Tracking of Humans), a lightweight algorithm for real-time multi-view human pose fusion that relies on three concepts: it integrates kinematic and biomechanical constraints to increase the joint positioning accuracy; it employs convex optimization-based inverse kinematics for spatial fusion; and it implements a state observer to improve temporal consistency. We evaluate COMETH on both public and industrial datasets, where it outperforms state-of-the-art methods in localization, detection, and tracking accuracy. The proposed fusion pipeline enables accurate and scalable human motion tracking, making it well-suited for industrial and safety-critical applications. The code is publicly available at this https URL. 

**Abstract (ZH)**: 在 Industry 5.0 时代，监控人类活动对于确保人体工程学安全和整体福祉至关重要。虽然多相机集中设置可以提高姿态估计准确性，但往往会受到高计算成本和带宽需求的影响，限制了其 scalability 和实时适用性。将处理分散到边缘设备可以降低网络带宽和计算负载。另一方面，边缘设备资源受限会导致准确性下降，而计算的分散会导致时间和空间不一致性。我们通过提出 COMETH（Convex Optimization for Multiview Estimation and Tracking of Humans）来应对这一挑战，这是一种依赖于三个概念的轻量化实时多视角人类姿态融合算法：它结合运动学和生物力学约束以提高关节定位准确性；利用基于凸优化的逆运动学进行空间融合；并实施状态观察器以提高时间一致性。我们在公共和工业数据集上评估了 COMETH，在定位、检测和跟踪准确性方面均优于现有方法。所提出的融合流水线实现了准确且可扩展的人体运动追踪，使其适用于工业和安全关键应用。代码已在以下网址开源：this https URL。 

---
# To New Beginnings: A Survey of Unified Perception in Autonomous Vehicle Software 

**Title (ZH)**: 新的起点：自主车辆软件中统一感知的综述 

**Authors**: Loïc Stratil, Felix Fent, Esteban Rivera, Markus Lienkamp  

**Link**: [PDF](https://arxiv.org/pdf/2508.20892)  

**Abstract**: Autonomous vehicle perception typically relies on modular pipelines that decompose the task into detection, tracking, and prediction. While interpretable, these pipelines suffer from error accumulation and limited inter-task synergy. Unified perception has emerged as a promising paradigm that integrates these sub-tasks within a shared architecture, potentially improving robustness, contextual reasoning, and efficiency while retaining interpretable outputs. In this survey, we provide a comprehensive overview of unified perception, introducing a holistic and systemic taxonomy that categorizes methods along task integration, tracking formulation, and representation flow. We define three paradigms -Early, Late, and Full Unified Perception- and systematically review existing methods, their architectures, training strategies, datasets used, and open-source availability, while highlighting future research directions. This work establishes the first comprehensive framework for understanding and advancing unified perception, consolidates fragmented efforts, and guides future research toward more robust, generalizable, and interpretable perception. 

**Abstract (ZH)**: 自主驾驶车辆感知通常依赖于模块化的管道，将任务分解为检测、跟踪和预测。虽然具有可解释性，但这些管道易发生错误累积且各任务之间的协同作用有限。统一感知作为一种有前景的范式，将这些子任务整合到共享架构中，有可能提高鲁棒性、上下文推理能力和效率，同时保持可解释的输出。在本文综述中，我们提供了一个全面的统一感知综述，引入了一个整体和系统的分类框架，按照任务整合、跟踪公式化和表示流对方法进行分类。我们定义了三种范式——早期统一感知、晚期统一感知和全统一感知——并对现有方法、架构、训练策略、使用的数据集以及开源可用性进行了系统性回顾，同时指出了未来的研究方向。这项工作建立了第一个全面的统一感知理解与推进框架，整合了分散的努力，并为未来研究指明了更鲁棒、更通用和更可解释的感知方向。 

---
# SKGE-SWIN: End-To-End Autonomous Vehicle Waypoint Prediction and Navigation Using Skip Stage Swin Transformer 

**Title (ZH)**: SKGE-SWIN：基于跳跃阶段Swin变压器的端到端自主车辆路径点预测与导航 

**Authors**: Fachri Najm Noer Kartiman, Rasim, Yaya Wihardi, Nurul Hasanah, Oskar Natan, Bambang Wahono, Taufik Ibnu Salim  

**Link**: [PDF](https://arxiv.org/pdf/2508.20762)  

**Abstract**: Focusing on the development of an end-to-end autonomous vehicle model with pixel-to-pixel context awareness, this research proposes the SKGE-Swin architecture. This architecture utilizes the Swin Transformer with a skip-stage mechanism to broaden feature representation globally and at various network levels. This approach enables the model to extract information from distant pixels by leveraging the Swin Transformer's Shifted Window-based Multi-head Self-Attention (SW-MSA) mechanism and to retain critical information from the initial to the final stages of feature extraction, thereby enhancing its capability to comprehend complex patterns in the vehicle's surroundings. The model is evaluated on the CARLA platform using adversarial scenarios to simulate real-world conditions. Experimental results demonstrate that the SKGE-Swin architecture achieves a superior Driving Score compared to previous methods. Furthermore, an ablation study will be conducted to evaluate the contribution of each architectural component, including the influence of skip connections and the use of the Swin Transformer, in improving model performance. 

**Abstract (ZH)**: 基于端到端自主车辆模型的像素到像素上下文意识发展，本文提出SKGE-Swin架构。该架构利用具有跳层机制的Swin Transformer，以全局和多级网络方式扩展特征表示，通过Swin Transformer的Shifted Window基于多头自注意力（SW-MSA）机制从远处的像素中提取信息，并在特征提取的各个阶段保留关键信息，从而增强其理解和解析车辆周围复杂模式的能力。该模型在CARLA平台上使用对抗场景进行评估以模拟真实世界条件。实验结果表明，SKGE-Swin架构在驾驶得分上优于以往方法。此外，还将进行消融研究以评估每个架构组件的贡献，包括跳层连接和使用Swin Transformer对提高模型性能的影响。 

---
# Mixture of Contexts for Long Video Generation 

**Title (ZH)**: 长视频生成的上下文混合 

**Authors**: Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, Gordon Wetzstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.21058)  

**Abstract**: Long video generation is fundamentally a long context memory problem: models must retain and retrieve salient events across a long range without collapsing or drifting. However, scaling diffusion transformers to generate long-context videos is fundamentally limited by the quadratic cost of self-attention, which makes memory and computation intractable and difficult to optimize for long sequences. We recast long-context video generation as an internal information retrieval task and propose a simple, learnable sparse attention routing module, Mixture of Contexts (MoC), as an effective long-term memory retrieval engine. In MoC, each query dynamically selects a few informative chunks plus mandatory anchors (caption, local windows) to attend to, with causal routing that prevents loop closures. As we scale the data and gradually sparsify the routing, the model allocates compute to salient history, preserving identities, actions, and scenes over minutes of content. Efficiency follows as a byproduct of retrieval (near-linear scaling), which enables practical training and synthesis, and the emergence of memory and consistency at the scale of minutes. 

**Abstract (ZH)**: 长视频生成本质上是一个长上下文记忆问题：模型必须在长时间范围内保留和检索重要的事件而不发生崩溃或漂移。然而，将扩散变换器扩展以生成长上下文视频从根本上受限于自注意力的二次成本，这使得记忆和计算在长序列中不可行且难以优化。我们重新将长上下文视频生成视为内部信息检索任务，并提出一种简单的可学习稀疏注意力路由模块——上下文混合（MoC），作为有效的长期记忆检索引擎。在MoC中，每个查询动态选择几个有信息性的片段加上强制性的锚点（标题、局部窗口），并采用因果路由以防止循环闭合。随着我们扩展数据并逐渐稀疏路由，模型将计算资源分配给重要的历史记录，从而在数分钟的内容中保持身份、动作和场景的一致性。作为检索的副产品，效率得以提高（接近线性扩展），这使实际的训练和合成成为可能，并在数分钟的尺度上出现了记忆和一致性。 

---
# Surfel-based 3D Registration with Equivariant SE(3) Features 

**Title (ZH)**: 基于Surfel的3D注册_with_不变SE(3)特征 

**Authors**: Xueyang Kang, Hang Zhao, Kourosh Khoshelham, Patrick Vandewalle  

**Link**: [PDF](https://arxiv.org/pdf/2508.20789)  

**Abstract**: Point cloud registration is crucial for ensuring 3D alignment consistency of multiple local point clouds in 3D reconstruction for remote sensing or digital heritage. While various point cloud-based registration methods exist, both non-learning and learning-based, they ignore point orientations and point uncertainties, making the model susceptible to noisy input and aggressive rotations of the input point cloud like orthogonal transformation; thus, it necessitates extensive training point clouds with transformation augmentations. To address these issues, we propose a novel surfel-based pose learning regression approach. Our method can initialize surfels from Lidar point cloud using virtual perspective camera parameters, and learns explicit $\mathbf{SE(3)}$ equivariant features, including both position and rotation through $\mathbf{SE(3)}$ equivariant convolutional kernels to predict relative transformation between source and target scans. The model comprises an equivariant convolutional encoder, a cross-attention mechanism for similarity computation, a fully-connected decoder, and a non-linear Huber loss. Experimental results on indoor and outdoor datasets demonstrate our model superiority and robust performance on real point-cloud scans compared to state-of-the-art methods. 

**Abstract (ZH)**: 基于surfel的Pose学习回归方法在远程 sensing或数字遗产3D重建中多局部点云的3D对齐一致性保障中至关重要。虽然存在多种基于点云的注册方法，无论是非学习还是学习-Based，但它们忽略了点的姿态和点的不确定性，使得模型对噪声输入和输入点云的剧烈旋转（如正交变换）尤为敏感；因而需要采用具有变换增强的大量训练点云。为解决这些问题，我们提出了一种新颖的基于surfel的Pose学习回归方法。该方法可以使用虚拟视角摄像机参数从激光点云中初始化surfels，并学习显式的$\mathbf{SE(3)}$不变特征，包括通过$\mathbf{SE(3)}$不变卷积核预测源和目标扫描之间的相对变换，同时包含了位置和旋转信息。模型由一个$\mathbf{SE(3)}$不变卷积编码器、用于相似度计算的交叉注意机制、一个全连接解码器以及一个非线性Huber损失组成。在室内和室外数据集上的实验结果表明，该模型在实际点云扫描中优于现有方法，展现了优越性和鲁棒性。 

---
# Looking Beyond the Obvious: A Survey on Abstract Concept Recognition for Video Understanding 

**Title (ZH)**: 超越表面：视频理解中抽象概念识别综述 

**Authors**: Gowreesh Mago, Pascal Mettes, Stevan Rudinac  

**Link**: [PDF](https://arxiv.org/pdf/2508.20765)  

**Abstract**: The automatic understanding of video content is advancing rapidly. Empowered by deeper neural networks and large datasets, machines are increasingly capable of understanding what is concretely visible in video frames, whether it be objects, actions, events, or scenes. In comparison, humans retain a unique ability to also look beyond concrete entities and recognize abstract concepts like justice, freedom, and togetherness. Abstract concept recognition forms a crucial open challenge in video understanding, where reasoning on multiple semantic levels based on contextual information is key. In this paper, we argue that the recent advances in foundation models make for an ideal setting to address abstract concept understanding in videos. Automated understanding of high-level abstract concepts is imperative as it enables models to be more aligned with human reasoning and values. In this survey, we study different tasks and datasets used to understand abstract concepts in video content. We observe that, periodically and over a long period, researchers have attempted to solve these tasks, making the best use of the tools available at their disposal. We advocate that drawing on decades of community experience will help us shed light on this important open grand challenge and avoid ``re-inventing the wheel'' as we start revisiting it in the era of multi-modal foundation models. 

**Abstract (ZH)**: 视频内容中高级抽象概念的自动理解正在迅速发展 

---
# Occlusion Robustness of CLIP for Military Vehicle Classification 

**Title (ZH)**: CLIP在军用车辆分类中的遮挡鲁棒性 

**Authors**: Jan Erik van Woerden, Gertjan Burghouts, Lotte Nijskens, Alma M. Liezenga, Sabina van Rooij, Frank Ruis, Hugo J. Kuijf  

**Link**: [PDF](https://arxiv.org/pdf/2508.20760)  

**Abstract**: Vision-language models (VLMs) like CLIP enable zero-shot classification by aligning images and text in a shared embedding space, offering advantages for defense applications with scarce labeled data. However, CLIP's robustness in challenging military environments, with partial occlusion and degraded signal-to-noise ratio (SNR), remains underexplored. We investigate CLIP variants' robustness to occlusion using a custom dataset of 18 military vehicle classes and evaluate using Normalized Area Under the Curve (NAUC) across occlusion percentages. Four key insights emerge: (1) Transformer-based CLIP models consistently outperform CNNs, (2) fine-grained, dispersed occlusions degrade performance more than larger contiguous occlusions, (3) despite improved accuracy, performance of linear-probed models sharply drops at around 35% occlusion, (4) by finetuning the model's backbone, this performance drop occurs at more than 60% occlusion. These results underscore the importance of occlusion-specific augmentations during training and the need for further exploration into patch-level sensitivity and architectural resilience for real-world deployment of CLIP. 

**Abstract (ZH)**: Vision-语言模型（VLMs）如CLIP通过在共享嵌入空间中对齐图像和文本实现零-shot分类，在稀缺标注数据的防御应用中具有优势。然而，CLIP在具有部分遮挡和降级信噪比（SNR）的挑战性军事环境中的鲁棒性仍需进一步探索。我们使用包含18类军事车辆的自定义数据集研究CLIP变体在遮挡下的鲁棒性，并通过不同遮挡百分比下的归一化面积下的曲线下的面积（NAUC）进行评估。四个关键见解浮出水面：（1）基于Transformer的CLIP模型始终优于CNN，（2）细微分散的遮挡比大面积连续遮挡对性能的影响更严重，（3）尽管准确率有所提高，线性探查模型在大约35%遮挡时性能急剧下降，（4）通过微调模型的骨干网络，这种性能下降发生在超过60%遮挡时。这些结果强调了在训练期间使用遮挡特定增强措施的重要性，并指出了需要进一步探索区域级敏感性和架构稳健性以实现CLIP在实际部署中的应用。 

---
# SeqVLM: Proposal-Guided Multi-View Sequences Reasoning via VLM for Zero-Shot 3D Visual Grounding 

**Title (ZH)**: SeqVLM：基于提案引导的多视图序列推理用于零-shot 3D视觉锚定 

**Authors**: Jiawen Lin, Shiran Bian, Yihang Zhu, Wenbin Tan, Yachao Zhang, Yuan Xie, Yanyun Qu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20758)  

**Abstract**: 3D Visual Grounding (3DVG) aims to localize objects in 3D scenes using natural language descriptions. Although supervised methods achieve higher accuracy in constrained settings, zero-shot 3DVG holds greater promise for real-world applications since eliminating scene-specific training requirements. However, existing zero-shot methods face challenges of spatial-limited reasoning due to reliance on single-view localization, and contextual omissions or detail degradation. To address these issues, we propose SeqVLM, a novel zero-shot 3DVG framework that leverages multi-view real-world scene images with spatial information for target object reasoning. Specifically, SeqVLM first generates 3D instance proposals via a 3D semantic segmentation network and refines them through semantic filtering, retaining only semantic-relevant candidates. A proposal-guided multi-view projection strategy then projects these candidate proposals onto real scene image sequences, preserving spatial relationships and contextual details in the conversion process of 3D point cloud to images. Furthermore, to mitigate VLM computational overload, we implement a dynamic scheduling mechanism that iteratively processes sequances-query prompts, leveraging VLM's cross-modal reasoning capabilities to identify textually specified objects. Experiments on the ScanRefer and Nr3D benchmarks demonstrate state-of-the-art performance, achieving Acc@0.25 scores of 55.6% and 53.2%, surpassing previous zero-shot methods by 4.0% and 5.2%, respectively, which advance 3DVG toward greater generalization and real-world applicability. The code is available at this https URL. 

**Abstract (ZH)**: 三维视觉定位（3D视觉定位）旨在使用自然语言描述在三维场景中定位物体。虽然监督方法在受限环境中实现了更高的准确性，零样本3D视觉定位由于消除了特定场景的训练要求，在实际应用中具有更大的潜力。然而，现有的零样本方法在依赖单视图定位时面临空间推理受限的问题，并且可能出现语境遗漏或细节降解。为了解决这些问题，我们提出了SeqVLM，这是一个新颖的零样本3D视觉定位框架，该框架利用多视图真实场景图像中的空间信息进行目标物体推理。具体而言，SeqVLM首先通过3D语义分割网络生成3D实例提案，并通过语义过滤对其进行 refinement，仅保留与语义相关的候选提案。然后，采用提案导向的多视图投影策略将这些候选提案投影到真实场景图像序列上，在3D点云到图像的转换过程中保持空间关系和语境细节。此外，为了缓解VLM的计算负担，我们实现了动态调度机制，该机制迭代处理序列-查询提示，利用VLM的跨模态推理能力识别文本指定的对象。在ScanRefer和Nr3D基准上的实验展示了最先进的性能， Acc@0.25得分为55.6%和53.2%，分别超越了之前的最佳零样本方法4.0%和5.2%，从而推动了3D视觉定位向更广泛的通用性和实际应用方向发展。代码可在以下链接获得：this https URL。 

---
# ${C}^{3}$-GS: Learning Context-aware, Cross-dimension, Cross-scale Feature for Generalizable Gaussian Splatting 

**Title (ZH)**: ${C}^{3}$-GS: 学习上下文aware、跨维度、跨尺度特征以实现通用高斯散射 

**Authors**: Yuxi Hu, Jun Zhang, Kuangyi Chen, Zhe Zhang, Friedrich Fraundorfer  

**Link**: [PDF](https://arxiv.org/pdf/2508.20754)  

**Abstract**: Generalizable Gaussian Splatting aims to synthesize novel views for unseen scenes without per-scene optimization. In particular, recent advancements utilize feed-forward networks to predict per-pixel Gaussian parameters, enabling high-quality synthesis from sparse input views. However, existing approaches fall short in encoding discriminative, multi-view consistent features for Gaussian predictions, which struggle to construct accurate geometry with sparse views. To address this, we propose $\mathbf{C}^{3}$-GS, a framework that enhances feature learning by incorporating context-aware, cross-dimension, and cross-scale constraints. Our architecture integrates three lightweight modules into a unified rendering pipeline, improving feature fusion and enabling photorealistic synthesis without requiring additional supervision. Extensive experiments on benchmark datasets validate that $\mathbf{C}^{3}$-GS achieves state-of-the-art rendering quality and generalization ability. Code is available at: this https URL. 

**Abstract (ZH)**: $\mathbf{C}^{3}$-GS旨在通过结合上下文感知、跨维度和跨尺度约束来增强特征学习，以合成未见过场景的新型视图。我们的架构将三个轻量级模块整合到统一的渲染管道中，提高特征融合，无需额外监督即可实现照片级真实感合成。在基准数据集上的广泛实验验证了$\mathbf{C}^{3}$-GS 达到了最顶尖的渲染质量和泛化能力。代码可在以下链接获取：this https URL。 

---
# CaddieSet: A Golf Swing Dataset with Human Joint Features and Ball Information 

**Title (ZH)**: CaddieSet: 一名高尔夫挥杆数据集，包含人体关节特征和球信息 

**Authors**: Seunghyeon Jung, Seoyoung Hong, Jiwoo Jeong, Seungwon Jeong, Jaerim Choi, Hoki Kim, Woojin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.20491)  

**Abstract**: Recent advances in deep learning have led to more studies to enhance golfers' shot precision. However, these existing studies have not quantitatively established the relationship between swing posture and ball trajectory, limiting their ability to provide golfers with the necessary insights for swing improvement. In this paper, we propose a new dataset called CaddieSet, which includes joint information and various ball information from a single shot. CaddieSet extracts joint information from a single swing video by segmenting it into eight swing phases using a computer vision-based approach. Furthermore, based on expert golf domain knowledge, we define 15 key metrics that influence a golf swing, enabling the interpretation of swing outcomes through swing-related features. Through experiments, we demonstrated the feasibility of CaddieSet for predicting ball trajectories using various benchmarks. In particular, we focus on interpretable models among several benchmarks and verify that swing feedback using our joint features is quantitatively consistent with established domain knowledge. This work is expected to offer new insight into golf swing analysis for both academia and the sports industry. 

**Abstract (ZH)**: Recent Advances in Deep Learning Have Led to More Studies to Enhance Golfers' Shot Precision: CaddieSet Dataset for Quantitative Analysis of Swing and Ball Trajectory Relationship 

---
# Ultra-Low-Latency Spiking Neural Networks with Temporal-Dependent Integrate-and-Fire Neuron Model for Objects Detection 

**Title (ZH)**: 具有时间依赖性整合发放神经元模型的超低延迟脉冲神经网络及其在物体检测中的应用 

**Authors**: Chengjun Zhang, Yuhao Zhang, Jie Yang, Mohamad Sawan  

**Link**: [PDF](https://arxiv.org/pdf/2508.20392)  

**Abstract**: Spiking Neural Networks (SNNs), inspired by the brain, are characterized by minimal power consumption and swift inference capabilities on neuromorphic hardware, and have been widely applied to various visual perception tasks. Current ANN-SNN conversion methods have achieved excellent results in classification tasks with ultra-low time-steps, but their performance in visual detection tasks remains suboptimal. In this paper, we propose a delay-spike approach to mitigate the issue of residual membrane potential caused by heterogeneous spiking patterns. Furthermore, we propose a novel temporal-dependent Integrate-and-Fire (tdIF) neuron architecture for SNNs. This enables Integrate-and-fire (IF) neurons to dynamically adjust their accumulation and firing behaviors based on the temporal order of time-steps. Our method enables spikes to exhibit distinct temporal properties, rather than relying solely on frequency-based representations. Moreover, the tdIF neuron maintains energy consumption on par with traditional IF neuron. We demonstrate that our method achieves more precise feature representation with lower time-steps, enabling high performance and ultra-low latency in visual detection tasks. In this study, we conduct extensive evaluation of the tdIF method across two critical vision tasks: object detection and lane line detection. The results demonstrate that the proposed method surpasses current ANN-SNN conversion approaches, achieving state-of-the-art performance with ultra-low latency (within 5 time-steps). 

**Abstract (ZH)**: 基于延迟脉冲的神经形态Integrate-and-Fire神经元架构及其在视觉检测任务中的应用 

---
# MedNet-PVS: A MedNeXt-Based Deep Learning Model for Automated Segmentation of Perivascular Spaces 

**Title (ZH)**: MedNet-PVS: 一种基于MedNeX的深度学习模型，用于 PERIVASCULAR SPACES 的自动分割 

**Authors**: Zhen Xuen Brandon Low, Rory Zhang, Hang Min, William Pham, Lucy Vivash, Jasmine Moses, Miranda Lynch, Karina Dorfman, Cassandra Marotta, Shaun Koh, Jacob Bunyamin, Ella Rowsthorn, Alex Jarema, Himashi Peiris, Zhaolin Chen, Sandy R. Shultz, David K. Wright, Dexiao Kong, Sharon L. Naismith, Terence J. O'Brien, Ying Xia, Meng Law, Benjamin Sinclair  

**Link**: [PDF](https://arxiv.org/pdf/2508.20256)  

**Abstract**: Enlarged perivascular spaces (PVS) are increasingly recognized as biomarkers of cerebral small vessel disease, Alzheimer's disease, stroke, and aging-related neurodegeneration. However, manual segmentation of PVS is time-consuming and subject to moderate inter-rater reliability, while existing automated deep learning models have moderate performance and typically fail to generalize across diverse clinical and research MRI datasets. We adapted MedNeXt-L-k5, a Transformer-inspired 3D encoder-decoder convolutional network, for automated PVS segmentation. Two models were trained: one using a homogeneous dataset of 200 T2-weighted (T2w) MRI scans from the Human Connectome Project-Aging (HCP-Aging) dataset and another using 40 heterogeneous T1-weighted (T1w) MRI volumes from seven studies across six scanners. Model performance was evaluated using internal 5-fold cross validation (5FCV) and leave-one-site-out cross validation (LOSOCV). MedNeXt-L-k5 models trained on the T2w images of the HCP-Aging dataset achieved voxel-level Dice scores of 0.88+/-0.06 (white matter, WM), comparable to the reported inter-rater reliability of that dataset, and the highest yet reported in the literature. The same models trained on the T1w images of the HCP-Aging dataset achieved a substantially lower Dice score of 0.58+/-0.09 (WM). Under LOSOCV, the model had voxel-level Dice scores of 0.38+/-0.16 (WM) and 0.35+/-0.12 (BG), and cluster-level Dice scores of 0.61+/-0.19 (WM) and 0.62+/-0.21 (BG). MedNeXt-L-k5 provides an efficient solution for automated PVS segmentation across diverse T1w and T2w MRI datasets. MedNeXt-L-k5 did not outperform the nnU-Net, indicating that the attention-based mechanisms present in transformer-inspired models to provide global context are not required for high accuracy in PVS segmentation. 

**Abstract (ZH)**: 扩大血管周间隙(PVS)日益被认为是脑小血管疾病、阿尔茨海默病、中风和年龄相关神经退行性变的生物标志物。然而，手动分段PVS耗时且 intra-rater 可靠性中等，而现有自动化深度学习模型的性能一般，并且通常无法在多样化的临床和研究MRI数据集中泛化。我们改编了受Transformer启发的3D编码器-解码器卷积网络MedNeXt-L-k5，用于自动PVS分段。我们训练了两个模型：一个使用来自Human Connectome Project-Aging (HCP-Aging) 数据集的200张T2加权(T2w) MRI扫描图像，另一个使用来自六台扫描器的七个研究中40张异质的T1加权(T1w) MRI体积。模型性能通过内部5折交叉验证(5FCV)和留一站点在外交叉验证(LOSOCV)进行评估。使用HCP-Aging数据集中T2w图像训练的MedNeXt-L-k5模型在灰质(WM)上的体素级别Dice分数为0.88±0.06，与该数据集报告的intra-rater可靠性相当，且在文献中首次达到最高水平。使用HCP-Aging数据集中T1w图像训练的相同模型在灰质(WM)上的体素级别Dice分数显著降低至0.58±0.09。在 LOSOCV 下，模型在灰质(WM)上的体素级别Dice分数为0.38±0.16，在基底节(BG)上的体素级别Dice分数为0.35±0.12；而在基底节(BG)上的聚类级别Dice分数为0.61±0.19，在灰质(WM)上的聚类级别Dice分数为0.62±0.21。MedNeXt-L-k5为多样化的T1w和T2w MRI数据集提供了高效解决方案。MedNeXt-L-k5并未超越nnU-Net，表明变压器启发模型中的基于注意力的机制对于提供全局上下文并非实现PVS分段高准确性的必要条件。 

---
# Data-Efficient Point Cloud Semantic Segmentation Pipeline for Unimproved Roads 

**Title (ZH)**: 未经改善道路的数据高效点云语义分割管道 

**Authors**: Andrew Yarovoi, Christopher R. Valenta  

**Link**: [PDF](https://arxiv.org/pdf/2508.20135)  

**Abstract**: In this case study, we present a data-efficient point cloud segmentation pipeline and training framework for robust segmentation of unimproved roads and seven other classes. Our method employs a two-stage training framework: first, a projection-based convolutional neural network is pre-trained on a mixture of public urban datasets and a small, curated in-domain dataset; then, a lightweight prediction head is fine-tuned exclusively on in-domain data. Along the way, we explore the application of Point Prompt Training to batch normalization layers and the effects of Manifold Mixup as a regularizer within our pipeline. We also explore the effects of incorporating histogram-normalized ambients to further boost performance. Using only 50 labeled point clouds from our target domain, we show that our proposed training approach improves mean Intersection-over-Union from 33.5% to 51.8% and the overall accuracy from 85.5% to 90.8%, when compared to naive training on the in-domain data. Crucially, our results demonstrate that pre-training across multiple datasets is key to improving generalization and enabling robust segmentation under limited in-domain supervision. Overall, this study demonstrates a practical framework for robust 3D semantic segmentation in challenging, low-data scenarios. Our code is available at: this https URL. 

**Abstract (ZH)**: 在这种案例研究中，我们提出了一种高效的数据点云分割管道和训练框架，用于鲁棒地分割未经改善的道路和其他七个类别。我们的方法采用了两阶段的训练框架：首先，在公共城市数据集和一个小规模的领域特定数据集的混合数据上预训练基于投影的卷积神经网络；然后，仅在领域特定数据上微调轻量级预测头部。在这一过程中，我们探讨了点提示训练在批量归一化层中的应用以及Manifold Mixup作为正则化器的效果。我们还研究了结合直方图归一化环境光以提高性能的影响。仅使用目标领域中的50个标注点云，我们展示了我们提出的训练方法将平均交并比从33.5%提高到51.8%，并将总体准确率从85.5%提高到90.8%，这与仅在领域内部数据上进行朴素训练的效果相比。关键的是，我们的结果证明了跨多个数据集预训练对于提高泛化能力和在有限领域监督下实现鲁棒分割的重要性。总体而言，该项研究展示了在具有挑战性的低数据场景中实现鲁棒3D语义分割的实用框架。我们的代码可在：this https URL获取。 

---
