# RadarSFD: Single-Frame Diffusion with Pretrained Priors for Radar Point Clouds 

**Title (ZH)**: RadarSFD：单帧扩散模型预训练先验用于雷达点云 

**Authors**: Bin Zhao, Nakul Garg  

**Link**: [PDF](https://arxiv.org/pdf/2509.18068)  

**Abstract**: Millimeter-wave radar provides perception robust to fog, smoke, dust, and low light, making it attractive for size, weight, and power constrained robotic platforms. Current radar imaging methods, however, rely on synthetic aperture or multi-frame aggregation to improve resolution, which is impractical for small aerial, inspection, or wearable systems. We present RadarSFD, a conditional latent diffusion framework that reconstructs dense LiDAR-like point clouds from a single radar frame without motion or SAR. Our approach transfers geometric priors from a pretrained monocular depth estimator into the diffusion backbone, anchors them to radar inputs via channel-wise latent concatenation, and regularizes outputs with a dual-space objective combining latent and pixel-space losses. On the RadarHD benchmark, RadarSFD achieves 35 cm Chamfer Distance and 28 cm Modified Hausdorff Distance, improving over the single-frame RadarHD baseline (56 cm, 45 cm) and remaining competitive with multi-frame methods using 5-41 frames. Qualitative results show recovery of fine walls and narrow gaps, and experiments across new environments confirm strong generalization. Ablation studies highlight the importance of pretrained initialization, radar BEV conditioning, and the dual-space loss. Together, these results establish the first practical single-frame, no-SAR mmWave radar pipeline for dense point cloud perception in compact robotic systems. 

**Abstract (ZH)**: 毫米波雷达提供了对雾、烟、尘和低光照条件具有鲁棒性的感知能力，使其成为尺寸、重量和功率受限的机器人平台的理想选择。然而，当前的雷达成像方法依赖于合成孔径或多帧聚合来提高分辨率，这对于小型航拍、检测或可穿戴系统来说是不现实的。我们提出了RadarSFD，这是一种基于条件潜变量扩散的框架，可以从单帧雷达数据中重建类似于LiDAR的密集点云，而无需运动或合成孔径雷达（SAR）。我们的方法通过迁移单目深度估计器的几何先验到扩散骨干网络中，并通过通道级潜变量拼接将这些先验锚定到雷达输入，同时使用结合潜变量空间和像素空间损失的目标进行输出正则化。在RadarHD基准测试中，RadarSFD实现了35厘米的切比雪夫距离和28厘米的修正哈夫森距离，优于单帧RadarHD基线（56厘米，45厘米），并且与使用5到41帧的多帧方法保持竞争力。定性结果显示恢复了精细的墙和狭窄的间隙，并且在新环境中的实验证实了较强的泛化能力。消融研究突显了预训练初始化、雷达BEV条件化和双空间损失的重要性。这些结果确立了首个适用于紧凑型机器人系统的实用单帧、无需SAR的毫米波雷达成密集点云感知的管道。 

---
# SocialTraj: Two-Stage Socially-Aware Trajectory Prediction for Autonomous Driving via Conditional Diffusion Model 

**Title (ZH)**: SocialTraj：基于条件扩散模型的两阶段社会意识轨迹预测技术在自动驾驶中的应用 

**Authors**: Xiao Zhou, Zengqi Peng, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.17850)  

**Abstract**: Accurate trajectory prediction of surrounding vehicles (SVs) is crucial for autonomous driving systems to avoid misguided decisions and potential accidents. However, achieving reliable predictions in highly dynamic and complex traffic scenarios remains a significant challenge. One of the key impediments lies in the limited effectiveness of current approaches to capture the multi-modal behaviors of drivers, which leads to predicted trajectories that deviate from actual future motions. To address this issue, we propose SocialTraj, a novel trajectory prediction framework integrating social psychology principles through social value orientation (SVO). By utilizing Bayesian inverse reinforcement learning (IRL) to estimate the SVO of SVs, we obtain the critical social context to infer the future interaction trend. To ensure modal consistency in predicted behaviors, the estimated SVOs of SVs are embedded into a conditional denoising diffusion model that aligns generated trajectories with historical driving styles. Additionally, the planned future trajectory of the ego vehicle (EV) is explicitly incorporated to enhance interaction modeling. Extensive experiments on NGSIM and HighD datasets demonstrate that SocialTraj is capable of adapting to highly dynamic and interactive scenarios while generating socially compliant and behaviorally consistent trajectory predictions, outperforming existing baselines. Ablation studies demonstrate that dynamic SVO estimation and explicit ego-planning components notably improve prediction accuracy and substantially reduce inference time. 

**Abstract (ZH)**: 基于社会心理学原理的社会行为导向的 surrounding 车辆轨迹预测框架 

---
# FGGS-LiDAR: Ultra-Fast, GPU-Accelerated Simulation from General 3DGS Models to LiDAR 

**Title (ZH)**: FGGS-LiDAR: 超快速、GPU 加速从通用3DGS模型到LiDAR的仿真 

**Authors**: Junzhe Wu, Yufei Jia, Yiyi Yan, Zhixing Chen, Tiao Tan, Zifan Wang, Guangyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17390)  

**Abstract**: While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic rendering, its vast ecosystem of assets remains incompatible with high-performance LiDAR simulation, a critical tool for robotics and autonomous driving. We present \textbf{FGGS-LiDAR}, a framework that bridges this gap with a truly plug-and-play approach. Our method converts \textit{any} pretrained 3DGS model into a high-fidelity, watertight mesh without requiring LiDAR-specific supervision or architectural alterations. This conversion is achieved through a general pipeline of volumetric discretization and Truncated Signed Distance Field (TSDF) extraction. We pair this with a highly optimized, GPU-accelerated ray-casting module that simulates LiDAR returns at over 500 FPS. We validate our approach on indoor and outdoor scenes, demonstrating exceptional geometric fidelity; By enabling the direct reuse of 3DGS assets for geometrically accurate depth sensing, our framework extends their utility beyond visualization and unlocks new capabilities for scalable, multimodal simulation. Our open-source implementation is available at this https URL. 

**Abstract (ZH)**: FGGS-LiDAR：一种用于LiDAR模拟的即插即用框架 

---
# Automated Coral Spawn Monitoring for Reef Restoration: The Coral Spawn and Larvae Imaging Camera System (CSLICS) 

**Title (ZH)**: 自动珊瑚施肥监测系统：珊瑚施肥和幼虫成像 camera 系统 (CSLICS) 

**Authors**: Dorian Tsai, Christopher A. Brunner, Riki Lamont, F. Mikaela Nordborg, Andrea Severati, Java Terry, Karen Jackel, Matthew Dunbabin, Tobias Fischer, Scarlett Raine  

**Link**: [PDF](https://arxiv.org/pdf/2509.17299)  

**Abstract**: Coral aquaculture for reef restoration requires accurate and continuous spawn counting for resource distribution and larval health monitoring, but current methods are labor-intensive and represent a critical bottleneck in the coral production pipeline. We propose the Coral Spawn and Larvae Imaging Camera System (CSLICS), which uses low cost modular cameras and object detectors trained using human-in-the-loop labeling approaches for automated spawn counting in larval rearing tanks. This paper details the system engineering, dataset collection, and computer vision techniques to detect, classify and count coral spawn. Experimental results from mass spawning events demonstrate an F1 score of 82.4\% for surface spawn detection at different embryogenesis stages, 65.3\% F1 score for sub-surface spawn detection, and a saving of 5,720 hours of labor per spawning event compared to manual sampling methods at the same frequency. Comparison of manual counts with CSLICS monitoring during a mass coral spawning event on the Great Barrier Reef demonstrates CSLICS' accurate measurement of fertilization success and sub-surface spawn counts. These findings enhance the coral aquaculture process and enable upscaling of coral reef restoration efforts to address climate change threats facing ecosystems like the Great Barrier Reef. 

**Abstract (ZH)**: 珊瑚养殖用于珊瑚礁恢复需要准确连续的孢子计数以进行资源分配和仔珊瑚健康监测，但现有方法劳动密集且是珊瑚生产管道中的关键瓶颈。我们提出了珊瑚孢子和仔珊瑚成像相机系统（CSLICS），该系统使用低成本模块化相机和通过人力在循环标注方法训练的对象检测器进行自动孢子计数。本文详细介绍了该系统的工程设计、数据集收集以及计算机视觉技术用于检测、分类和计数珊瑚孢子的方法。大规模产孢事件的实验结果表明，对于不同胚胎发育阶段的表层孢子检测，CSLICS的F1分数为82.4%，对次表层孢子检测的F1分数为65.3%，与相同频率的手动采样方法相比，每次产孢事件可节省5,720小时的劳动时间。在大堡礁大规模珊瑚产孢事件期间，手动计数与CSLICS监测的比较显示CSLICS能够准确测量受精成功率和次表层孢子计数。这些发现提高了珊瑚养殖过程，使珊瑚礁恢复努力规模化成为可能，以应对如大堡礁生态系统所面临的一系列气候威胁。 

---
# Event-Based Visual Teach-and-Repeat via Fast Fourier-Domain Cross-Correlation 

**Title (ZH)**: 基于事件的视觉教示与重演：快速傅里叶域交叉相关方法 

**Authors**: Gokul B. Nair, Alejandro Fontan, Michael Milford, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.17287)  

**Abstract**: Visual teach-and-repeat navigation enables robots to autonomously traverse previously demonstrated paths by comparing current sensory input with recorded trajectories. However, conventional frame-based cameras fundamentally limit system responsiveness: their fixed frame rates (typically 30-60 Hz) create inherent latency between environmental changes and control responses. Here we present the first event-camera-based visual teach-and-repeat system. To achieve this, we develop a frequency-domain cross-correlation framework that transforms the event stream matching problem into computationally efficient Fourier space multiplications, capable of exceeding 300Hz processing rates, an order of magnitude faster than frame-based approaches. By exploiting the binary nature of event frames and applying image compression techniques, we further enhance the computational speed of the cross-correlation process without sacrificing localization accuracy. Extensive experiments using a Prophesee EVK4 HD event camera mounted on an AgileX Scout Mini robot demonstrate successful autonomous navigation across 4000+ meters of indoor and outdoor trajectories. Our system achieves ATEs below 24 cm while maintaining consistent high-frequency control updates. Our evaluations show that our approach achieves substantially higher update rates compared to conventional frame-based systems, underscoring the practical viability of event-based perception for real-time robotic navigation. 

**Abstract (ZH)**: 基于事件相机的视觉教示-重复导航使机器人能够通过将当前感测输入与记录轨迹进行比较，自主穿越 previously demonstrated 路径。然而，传统基于帧的相机从根本上限制了系统的响应性：它们固定的帧率（通常为 30-60 Hz）在环境变化和控制响应之间造成了固有的延迟。在这里，我们提出了第一个基于事件相机的视觉教示-重复系统。为此，我们开发了一种频域交叉相关框架，将事件流匹配问题转换为计算效率高的傅里叶空间乘法，能够超过 300 Hz 的处理速率，比基于帧的方法快一个数量级。通过利用事件帧的二进制性质并应用图像压缩技术，我们进一步加快了交叉相关过程的计算速度，同时不牺牲定位精度。使用安装在 AgileX Scout Mini 机器人上的 Prophesee EVK4 HD 事件相机进行的详实验表明，该系统在 4000 多米的室内外轨迹上实现了成功的自主导航。我们的系统实现了 ATEs 低于 24 cm，同时保持一致的高频率控制更新。我们的评估表明，与传统的基于帧的系统相比，我们的方法实现了显著更高的更新速率，突显了事件驱动感知在实时机器人导航中的实际可行性。 

---
# CoPlanner: An Interactive Motion Planner with Contingency-Aware Diffusion for Autonomous Driving 

**Title (ZH)**: CoPlanner: 一种具有应急意识扩散的交互式运动规划算法用于自主驾驶 

**Authors**: Ruiguo Zhong, Ruoyu Yao, Pei Liu, Xiaolong Chen, Rui Yang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.17080)  

**Abstract**: Accurate trajectory prediction and motion planning are crucial for autonomous driving systems to navigate safely in complex, interactive environments characterized by multimodal uncertainties. However, current generation-then-evaluation frameworks typically construct multiple plausible trajectory hypotheses but ultimately adopt a single most likely outcome, leading to overconfident decisions and a lack of fallback strategies that are vital for safety in rare but critical scenarios. Moreover, the usual decoupling of prediction and planning modules could result in socially inconsistent or unrealistic joint trajectories, especially in highly interactive traffic. To address these challenges, we propose a contingency-aware diffusion planner (CoPlanner), a unified framework that jointly models multi-agent interactive trajectory generation and contingency-aware motion planning. Specifically, the pivot-conditioned diffusion mechanism anchors trajectory sampling on a validated, shared short-term segment to preserve temporal consistency, while stochastically generating diverse long-horizon branches that capture multimodal motion evolutions. In parallel, we design a contingency-aware multi-scenario scoring strategy that evaluates candidate ego trajectories across multiple plausible long-horizon evolution scenarios, balancing safety, progress, and comfort. This integrated design preserves feasible fallback options and enhances robustness under uncertainty, leading to more realistic interaction-aware planning. Extensive closed-loop experiments on the nuPlan benchmark demonstrate that CoPlanner consistently surpasses state-of-the-art methods on both Val14 and Test14 datasets, achieving significant improvements in safety and comfort under both reactive and non-reactive settings. Code and model will be made publicly available upon acceptance. 

**Abstract (ZH)**: 基于 contingency 意识的扩散规划器（CoPlanner）：联合建模多代理交互轨迹生成与 contingency 意识运动规划 

---
# HOGraspFlow: Exploring Vision-based Generative Grasp Synthesis with Hand-Object Priors and Taxonomy Awareness 

**Title (ZH)**: HOGraspFlow: 基于手-物先验和分类意识的视觉生成性抓取合成探索 

**Authors**: Yitian Shi, Zicheng Guo, Rosa Wolf, Edgar Welte, Rania Rayyes  

**Link**: [PDF](https://arxiv.org/pdf/2509.16871)  

**Abstract**: We propose Hand-Object\emph{(HO)GraspFlow}, an affordance-centric approach that retargets a single RGB with hand-object interaction (HOI) into multi-modal executable parallel jaw grasps without explicit geometric priors on target objects. Building on foundation models for hand reconstruction and vision, we synthesize $SE(3)$ grasp poses with denoising flow matching (FM), conditioned on the following three complementary cues: RGB foundation features as visual semantics, HOI contact reconstruction, and taxonomy-aware prior on grasp types. Our approach demonstrates high fidelity in grasp synthesis without explicit HOI contact input or object geometry, while maintaining strong contact and taxonomy recognition. Another controlled comparison shows that \emph{HOGraspFlow} consistently outperforms diffusion-based variants (\emph{HOGraspDiff}), achieving high distributional fidelity and more stable optimization in $SE(3)$. We demonstrate a reliable, object-agnostic grasp synthesis from human demonstrations in real-world experiments, where an average success rate of over $83\%$ is achieved. 

**Abstract (ZH)**: Hand-Object GraspFlow 

---
# Improve bounding box in Carla Simulator 

**Title (ZH)**: 改进Carla模拟器中的边界框 

**Authors**: Mohamad Mofeed Chaar, Jamal Raiyn, Galia Weidl  

**Link**: [PDF](https://arxiv.org/pdf/2509.16773)  

**Abstract**: The CARLA simulator (Car Learning to Act) serves as a robust platform for testing algorithms and generating datasets in the field of Autonomous Driving (AD). It provides control over various environmental parameters, enabling thorough evaluation. Development bounding boxes are commonly utilized tools in deep learning and play a crucial role in AD applications. The predominant method for data generation in the CARLA Simulator involves identifying and delineating objects of interest, such as vehicles, using bounding boxes. The operation in CARLA entails capturing the coordinates of all objects on the map, which are subsequently aligned with the sensor's coordinate system at the ego vehicle and then enclosed within bounding boxes relative to the ego vehicle's perspective. However, this primary approach encounters challenges associated with object detection and bounding box annotation, such as ghost boxes. Although these procedures are generally effective at detecting vehicles and other objects within their direct line of sight, they may also produce false positives by identifying objects that are obscured by obstructions. We have enhanced the primary approach with the objective of filtering out unwanted boxes. Performance analysis indicates that the improved approach has achieved high accuracy. 

**Abstract (ZH)**: CARLA模拟器（Car Learning to Act）是自动驾驶（AD）领域测试算法和生成数据集的 robust 平台。它提供了对各种环境参数的控制，从而实现全面评估。开发边界框是深度学习中常用的工具，在自动驾驶应用中发挥着关键作用。CARLA模拟器中的主要数据生成方法是通过使用边界框来识别和界定感兴趣的物体，如车辆。在CARLA中，操作包括捕获地图上所有物体的坐标，随后将这些坐标与以自我车辆为中心的传感器坐标系统对齐，并相对于自我车辆的视角将这些物体封装在边界框内。然而，这种主要方法在物体检测和边界框注释方面遇到了挑战，例如幽灵框。尽管这些过程在检测直接视线内的车辆和其他物体方面通常是有效的，但也可能因为识别被遮挡的物体而产生假阳性。为了过滤出不必要的边界框，我们改进了主要方法。性能分析表明，改进的方法已经实现了高精度。 

---
# No Need for Real 3D: Fusing 2D Vision with Pseudo 3D Representations for Robotic Manipulation Learning 

**Title (ZH)**: 无需真实三维：将二维视觉与伪三维表示融合用于机器人操作学习 

**Authors**: Run Yu, Yangdi Liu, Wen-Da Wei, Chen Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.16532)  

**Abstract**: Recently,vision-based robotic manipulation has garnered significant attention and witnessed substantial advancements. 2D image-based and 3D point cloud-based policy learning represent two predominant paradigms in the field, with recent studies showing that the latter consistently outperforms the former in terms of both policy performance and generalization, thereby underscoring the value and significance of 3D information. However, 3D point cloud-based approaches face the significant challenge of high data acquisition costs, limiting their scalability and real-world deployment. To address this issue, we propose a novel framework NoReal3D: which introduces the 3DStructureFormer, a learnable 3D perception module capable of transforming monocular images into geometrically meaningful pseudo-point cloud features, effectively fused with the 2D encoder output features. Specially, the generated pseudo-point clouds retain geometric and topological structures so we design a pseudo-point cloud encoder to preserve these properties, making it well-suited for our framework. We also investigate the effectiveness of different feature fusion this http URL framework enhances the robot's understanding of 3D spatial structures while completely eliminating the substantial costs associated with 3D point cloud this http URL experiments across various tasks validate that our framework can achieve performance comparable to 3D point cloud-based methods, without the actual point cloud data. 

**Abstract (ZH)**: 基于Vision的机器人操纵最近取得了显著进展，两种主要范式是基于2D图像和基于3D点云的策略学习。研究表明，基于3D点云的方法在策略性能和泛化能力上优于基于2D图像的方法，强调了3D信息的价值和重要性。然而，基于3D点云的方法面临着高数据获取成本的重大挑战，限制了其可扩展性和实际部署。为解决这一问题，我们提出了一种新型框架NoReal3D：引入了一个可学习的3D感知模块3DStructureFormer，能够将单目图像转换为几何上有意义的伪点云特征，并与2D编码器输出特征有效融合。特别地，生成的伪点云保留了几何和拓扑结构，因此我们设计了一个伪点云编码器来保留这些特性，使之适合我们的框架。我们还研究了不同特征融合的有效性，该框架增强了机器人对3D空间结构的理解，同时完全消除了3D点云数据的巨大成本。来自各种任务的实验验证了我们的框架可以实现与基于3D点云的方法相当的性能，而无需实际的点云数据。 

---
# DINOv3-Diffusion Policy: Self-Supervised Large Visual Model for Visuomotor Diffusion Policy Learning 

**Title (ZH)**: DINOv3-扩散策略：自我监督的大规模视觉模型用于视觉运动扩散策略学习 

**Authors**: ThankGod Egbe, Peng Wang, Zhihao Guo, Zidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.17684)  

**Abstract**: This paper evaluates DINOv3, a recent large-scale self-supervised vision backbone, for visuomotor diffusion policy learning in robotic manipulation. We investigate whether a purely self-supervised encoder can match or surpass conventional supervised ImageNet-pretrained backbones (e.g., ResNet-18) under three regimes: training from scratch, frozen, and finetuned. Across four benchmark tasks (Push-T, Lift, Can, Square) using a unified FiLM-conditioned diffusion policy, we find that (i) finetuned DINOv3 matches or exceeds ResNet-18 on several tasks, (ii) frozen DINOv3 remains competitive, indicating strong transferable priors, and (iii) self-supervised features improve sample efficiency and robustness. These results support self-supervised large visual models as effective, generalizable perceptual front-ends for action diffusion policies, motivating further exploration of scalable label-free pretraining in robotic manipulation. Compared to using ResNet18 as a backbone, our approach with DINOv3 achieves up to a 10% absolute increase in test-time success rates on challenging tasks such as Can, and on-the-par performance in tasks like Lift, PushT, and Square. 

**Abstract (ZH)**: 本文评估了DINOv3，这是一种近期的大规模自监督视觉骨干模型，用于机器人操作中的visuomotor扩散策略学习。我们研究了在三种情况下纯自监督编码器的表现：从零开始训练、冻结和微调，与传统的监督ImageNet预训练骨干模型（如ResNet-18）进行比较。在统一的FiLM条件扩散策略下，我们针对四个基准任务（Push-T、Lift、Can、Square）发现：(i) 微调后的DINOv3在几个任务上与ResNet-18持平或超过ResNet-18；(ii) 冻结后的DINOv3仍具有竞争力，表明其具有较强的可迁移先验；(iii) 自监督特征提高了样本效率和鲁棒性。这些结果支持自监督的大规模视觉模型作为有效的、通用的感知前端，适用于动力扩散策略，在机器人操作中进一步探索无标签的预训练具有激励作用。与使用ResNet18作为骨干模型相比，我们的方法在如Can等具有挑战性的任务上的测试成功率提高了10%以上，在如Lift、PushT和Square等任务上表现相当。 

---
# VideoArtGS: Building Digital Twins of Articulated Objects from Monocular Video 

**Title (ZH)**: VideoArtGS: 基于单目视频构建articulated对象的数字孪生模型 

**Authors**: Yu Liu, Baoxiong Jia, Ruijie Lu, Chuyue Gan, Huayu Chen, Junfeng Ni, Song-Chun Zhu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17647)  

**Abstract**: Building digital twins of articulated objects from monocular video presents an essential challenge in computer vision, which requires simultaneous reconstruction of object geometry, part segmentation, and articulation parameters from limited viewpoint inputs. Monocular video offers an attractive input format due to its simplicity and scalability; however, it's challenging to disentangle the object geometry and part dynamics with visual supervision alone, as the joint movement of the camera and parts leads to ill-posed estimation. While motion priors from pre-trained tracking models can alleviate the issue, how to effectively integrate them for articulation learning remains largely unexplored. To address this problem, we introduce VideoArtGS, a novel approach that reconstructs high-fidelity digital twins of articulated objects from monocular video. We propose a motion prior guidance pipeline that analyzes 3D tracks, filters noise, and provides reliable initialization of articulation parameters. We also design a hybrid center-grid part assignment module for articulation-based deformation fields that captures accurate part motion. VideoArtGS demonstrates state-of-the-art performance in articulation and mesh reconstruction, reducing the reconstruction error by about two orders of magnitude compared to existing methods. VideoArtGS enables practical digital twin creation from monocular video, establishing a new benchmark for video-based articulated object reconstruction. Our work is made publicly available at: this https URL. 

**Abstract (ZH)**: 从单目视频构建 articulated 对象的数字双胞胎在计算机视觉中提出了一项基本挑战，这需要从有限视角输入中同时重建对象几何、部分分割和关节参数。单目视频因其简洁性和可扩展性提供了有吸引力的输入格式；然而，仅通过视觉监督来区分对象几何和部分动态仍然具有挑战性，因为相机和部分的联合运动导致了病态估计问题。尽管预训练的跟踪模型可以提供运动先验以缓解该问题，但如何有效将其整合到关节学习中仍亟待探索。为了解决这一问题，我们引入了 VideoArtGS，一种从单目视频重建 articulated 对象高保真数字双胞胎的新方法。我们提出了一种运动先验指导管道，用于分析 3D 轨迹、过滤噪声并提供可靠的关节参数初始化。我们还设计了一种混合中心-网格部分分配模块，用于基于关节的变形场，以捕获准确的部分运动。VideoArtGS 在关节重建和网格重建方面表现出最先进的性能，与现有方法相比，重建误差降低了两个数量级。VideoArtGS 实现了从单目视频创建实用数字双胞胎，并建立了基于视频的 articulated 对象重建的新基准。我们的工作已公开发布于：this https URL。 

---
# DepTR-MOT: Unveiling the Potential of Depth-Informed Trajectory Refinement for Multi-Object Tracking 

**Title (ZH)**: DepTR-MOT: 深度导向轨迹精炼在多目标跟踪中的潜力探索 

**Authors**: Buyin Deng, Lingxin Huang, Kai Luo, Fei Teng, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17323)  

**Abstract**: Visual Multi-Object Tracking (MOT) is a crucial component of robotic perception, yet existing Tracking-By-Detection (TBD) methods often rely on 2D cues, such as bounding boxes and motion modeling, which struggle under occlusions and close-proximity interactions. Trackers relying on these 2D cues are particularly unreliable in robotic environments, where dense targets and frequent occlusions are common. While depth information has the potential to alleviate these issues, most existing MOT datasets lack depth annotations, leading to its underexploited role in the domain. To unveil the potential of depth-informed trajectory refinement, we introduce DepTR-MOT, a DETR-based detector enhanced with instance-level depth information. Specifically, we propose two key innovations: (i) foundation model-based instance-level soft depth label supervision, which refines depth prediction, and (ii) the distillation of dense depth maps to maintain global depth consistency. These strategies enable DepTR-MOT to output instance-level depth during inference, without requiring foundation models and without additional computational cost. By incorporating depth cues, our method enhances the robustness of the TBD paradigm, effectively resolving occlusion and close-proximity challenges. Experiments on both the QuadTrack and DanceTrack datasets demonstrate the effectiveness of our approach, achieving HOTA scores of 27.59 and 44.47, respectively. In particular, results on QuadTrack, a robotic platform MOT dataset, highlight the advantages of our method in handling occlusion and close-proximity challenges in robotic tracking. The source code will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于深度信息的Visual多对象跟踪（DepTR-MOT） 

---
# CoBEVMoE: Heterogeneity-aware Feature Fusion with Dynamic Mixture-of-Experts for Collaborative Perception 

**Title (ZH)**: CoBEVMoE: 具有动态专家混合的异质特征融合协作感知 

**Authors**: Lingzhao Kong, Jiacheng Lin, Siyu Li, Kai Luo, Zhiyong Li, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17107)  

**Abstract**: Collaborative perception aims to extend sensing coverage and improve perception accuracy by sharing information among multiple agents. However, due to differences in viewpoints and spatial positions, agents often acquire heterogeneous observations. Existing intermediate fusion methods primarily focus on aligning similar features, often overlooking the perceptual diversity among agents. To address this limitation, we propose CoBEVMoE, a novel collaborative perception framework that operates in the Bird's Eye View (BEV) space and incorporates a Dynamic Mixture-of-Experts (DMoE) architecture. In DMoE, each expert is dynamically generated based on the input features of a specific agent, enabling it to extract distinctive and reliable cues while attending to shared semantics. This design allows the fusion process to explicitly model both feature similarity and heterogeneity across agents. Furthermore, we introduce a Dynamic Expert Metric Loss (DEML) to enhance inter-expert diversity and improve the discriminability of the fused representation. Extensive experiments on the OPV2V and DAIR-V2X-C datasets demonstrate that CoBEVMoE achieves state-of-the-art performance. Specifically, it improves the IoU for Camera-based BEV segmentation by +1.5% on OPV2V and the AP@50 for LiDAR-based 3D object detection by +3.0% on DAIR-V2X-C, verifying the effectiveness of expert-based heterogeneous feature modeling in multi-agent collaborative perception. The source code will be made publicly available at this https URL. 

**Abstract (ZH)**: 协作感知旨在通过多个代理共享信息来扩展感知覆盖范围和提高感知准确性。然而，由于视角和空间位置的差异，代理通常会获得异质性观测。现有的中间融合方法主要关注对齐相似特征，往往忽略了代理之间的感知多样性。为了解决这一局限，我们提出了一种新的协作感知框架CoBEVMoE，在Bird's Eye View (BEV)空间中运作，并结合了动态Mixture-of-Experts (DMoE)架构。在DMoE中，每个专家基于特定代理的输入特征动态生成，能够提取独特的可靠线索并关注共享语义。这种设计使得融合过程能够明确建模代理之间特征的相似性和异质性。此外，我们引入了动态专家度量损失(DEML)以增强专家之间的多样性并提高融合表示的可区分性。在OPV2V和DAIR-V2X-C数据集上的 extensive 实验表明，CoBEVMoE 达到了最先进的性能。具体而言，在OPV2V上，它将Camera-based BEV分割的IoU提高1.5%，在DAIR-V2X-C上，它将基于LiDAR的3D物体检测的AP@50提高3.0%，验证了基于专家的异质特征建模在多代理协作感知中的有效性。源代码将在此处公开。 

---
# SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving 

**Title (ZH)**: SQS：通过查询驱动的点积增强自主驾驶中的稀疏感知模型 

**Authors**: Haiming Zhang, Yiyao Zhu, Wending Zhou, Xu Yan, Yingjie Cai, Bingbing Liu, Shuguang Cui, Zhen Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.16588)  

**Abstract**: Sparse Perception Models (SPMs) adopt a query-driven paradigm that forgoes explicit dense BEV or volumetric construction, enabling highly efficient computation and accelerated inference. In this paper, we introduce SQS, a novel query-based splatting pre-training specifically designed to advance SPMs in autonomous driving. SQS introduces a plug-in module that predicts 3D Gaussian representations from sparse queries during pre-training, leveraging self-supervised splatting to learn fine-grained contextual features through the reconstruction of multi-view images and depth maps. During fine-tuning, the pre-trained Gaussian queries are seamlessly integrated into downstream networks via query interaction mechanisms that explicitly connect pre-trained queries with task-specific queries, effectively accommodating the diverse requirements of occupancy prediction and 3D object detection. Extensive experiments on autonomous driving benchmarks demonstrate that SQS delivers considerable performance gains across multiple query-based 3D perception tasks, notably in occupancy prediction and 3D object detection, outperforming prior state-of-the-art pre-training approaches by a significant margin (i.e., +1.3 mIoU on occupancy prediction and +1.0 NDS on 3D detection). 

**Abstract (ZH)**: SQS：基于查询的插件模块预训练方法用于自主驾驶的稀疏感知模型 

---
# ST-GS: Vision-Based 3D Semantic Occupancy Prediction with Spatial-Temporal Gaussian Splatting 

**Title (ZH)**: ST-GS: 基于视觉的时空高斯点云3D语义占用预测 

**Authors**: Xiaoyang Yan, Muleilan Pei, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.16552)  

**Abstract**: 3D occupancy prediction is critical for comprehensive scene understanding in vision-centric autonomous driving. Recent advances have explored utilizing 3D semantic Gaussians to model occupancy while reducing computational overhead, but they remain constrained by insufficient multi-view spatial interaction and limited multi-frame temporal consistency. To overcome these issues, in this paper, we propose a novel Spatial-Temporal Gaussian Splatting (ST-GS) framework to enhance both spatial and temporal modeling in existing Gaussian-based pipelines. Specifically, we develop a guidance-informed spatial aggregation strategy within a dual-mode attention mechanism to strengthen spatial interaction in Gaussian representations. Furthermore, we introduce a geometry-aware temporal fusion scheme that effectively leverages historical context to improve temporal continuity in scene completion. Extensive experiments on the large-scale nuScenes occupancy prediction benchmark showcase that our proposed approach not only achieves state-of-the-art performance but also delivers markedly better temporal consistency compared to existing Gaussian-based methods. 

**Abstract (ZH)**: 基于视觉的自动驾驶中全面场景理解的关键是3D占用率预测。尽管近期研究探索利用3D语义高斯分布来建模占用率以减少计算开销，但它们仍受到多视图空间交互不足和多帧时间一致性有限的限制。为克服这些限制，本文提出一种新型时空高斯点积（ST-GS）框架，以增强现有高斯基线框架中的空间和时间建模。具体而言，我们在双模式注意力机制内开发了一种指导信息驱动的空间聚合策略，以增强高斯表示的空间交互。此外，我们引入了一种几何感知的时间融合方案，有效利用历史上下文以提高场景补全的时间连续性。在大规模nuScenes占用率预测基准上的 extensive 实验展示了，我们提出的方法不仅达到最先进的性能，而且在时间一致性方面也显著优于现有的高斯基线方法。 

---
# StereoAdapter: Adapting Stereo Depth Estimation to Underwater Scenes 

**Title (ZH)**: StereoAdapter: 将立体深度估计适应于水下场景 

**Authors**: Zhengri Wu, Yiran Wang, Yu Wen, Zeyu Zhang, Biao Wu, Hao Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16415)  

**Abstract**: Underwater stereo depth estimation provides accurate 3D geometry for robotics tasks such as navigation, inspection, and mapping, offering metric depth from low-cost passive cameras while avoiding the scale ambiguity of monocular methods. However, existing approaches face two critical challenges: (i) parameter-efficiently adapting large vision foundation encoders to the underwater domain without extensive labeled data, and (ii) tightly fusing globally coherent but scale-ambiguous monocular priors with locally metric yet photometrically fragile stereo correspondences. To address these challenges, we propose StereoAdapter, a parameter-efficient self-supervised framework that integrates a LoRA-adapted monocular foundation encoder with a recurrent stereo refinement module. We further introduce dynamic LoRA adaptation for efficient rank selection and pre-training on the synthetic UW-StereoDepth-40K dataset to enhance robustness under diverse underwater conditions. Comprehensive evaluations on both simulated and real-world benchmarks show improvements of 6.11% on TartanAir and 5.12% on SQUID compared to state-of-the-art methods, while real-world deployment with the BlueROV2 robot further demonstrates the consistent robustness of our approach. Code: this https URL. Website: this https URL. 

**Abstract (ZH)**: 水下立体深度估计提供了用于导航、检查和建图等机器人任务的精确三维几何结构，能够在不昂贵的被动相机上提供度量深度，同时避免单目方法中的比例歧义。然而，现有方法面临两个关键挑战：(i) 在无需大量标注数据的情况下，高效地将大型视觉基础编码器适应到水下领域；(ii) 紧密融合全局一致但比例模糊的单目先验与局部度量但光度脆弱的立体对应关系。为解决这些挑战，我们提出了一种参数高效的自监督框架 StereoAdapter，该框架结合了一种 LoRA 调整的单目基础编码器和递归立体精炼模块。我们还引入了动态 LoRA 调整以高效选择秩，并在合成的 UW-StereoDepth-40K 数据集上预训练以增强在各种水下条件下的鲁棒性。在模拟和真实世界基准上的全面评估表明，相较于现有最佳方法，StereoAdapter 在 TartanAir 上提高了 6.11%，在 SQUID 上提高了 5.12%，而实际部署在 BlueROV2 机器人上进一步证明了我们方法的一致鲁棒性。代码：this https URL。网站：this https URL。 

---
# Prompt-Driven Agentic Video Editing System: Autonomous Comprehension of Long-Form, Story-Driven Media 

**Title (ZH)**: 基于提示的代理视频编辑系统：自主理解长形式叙事媒体 

**Authors**: Zihan Ding, Junlong Chen, Per Ola Kristensson, Junxiao Shen, Xinyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16811)  

**Abstract**: Creators struggle to edit long-form, narrative-rich videos not because of UI complexity, but due to the cognitive demands of searching, storyboarding, and sequencing hours of footage. Existing transcript- or embedding-based methods fall short for creative workflows, as models struggle to track characters, infer motivations, and connect dispersed events. We present a prompt-driven, modular editing system that helps creators restructure multi-hour content through free-form prompts rather than timelines. At its core is a semantic indexing pipeline that builds a global narrative via temporal segmentation, guided memory compression, and cross-granularity fusion, producing interpretable traces of plot, dialogue, emotion, and context. Users receive cinematic edits while optionally refining transparent intermediate outputs. Evaluated on 400+ videos with expert ratings, QA, and preference studies, our system scales prompt-driven editing, preserves narrative coherence, and balances automation with creator control. 

**Abstract (ZH)**: 创作者在编辑长格式、叙事丰富的视频时遇到困难并非因为UI复杂性，而是由于搜索、故事板制作和剪辑长达数小时的素材所造成的认知需求。现有的基于转录或嵌入的方法无法满足创意工作流程的需求，因为模型在追踪人物、推断动机以及连接分散的事件方面存在困难。我们提出了一种基于提示驱动的模块化编辑系统，通过自由格式的提示而非时间线帮助创作者重构多小时的内容。其核心是一个语义索引管道，通过时间分割、指导性记忆压缩和跨粒度融合构建全局叙事，生成可解释的情节、对话、情感和上下文的轨迹。用户在获得电影级别的编辑效果的同时，可以根据需要优化透明的中间输出。通过专家评分、QA和偏好研究评估了400多个视频，我们的系统实现了提示驱动编辑的扩展性，保持了叙事连贯性，并在自动化与创作者控制之间达到了平衡。 

---
# TS-P$^2$CL: Plug-and-Play Dual Contrastive Learning for Vision-Guided Medical Time Series Classification 

**Title (ZH)**: TS-P$^2$CL: 插头即用的双对比学习在视觉引导医学时序分类中的应用 

**Authors**: Qi'ao Xu, Pengfei Wang, Bo Zhong, Tianwen Qian, Xiaoling Wang, Ye Wang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17802)  

**Abstract**: Medical time series (MedTS) classification is pivotal for intelligent healthcare, yet its efficacy is severely limited by poor cross-subject generation due to the profound cross-individual heterogeneity. Despite advances in architectural innovations and transfer learning techniques, current methods remain constrained by modality-specific inductive biases that limit their ability to learn universally invariant representations. To overcome this, we propose TS-P$^2$CL, a novel plug-and-play framework that leverages the universal pattern recognition capabilities of pre-trained vision models. We introduce a vision-guided paradigm that transforms 1D physiological signals into 2D pseudo-images, establishing a bridge to the visual domain. This transformation enables implicit access to rich semantic priors learned from natural images. Within this unified space, we employ a dual-contrastive learning strategy: intra-modal consistency enforces temporal coherence, while cross-modal alignment aligns time-series dynamics with visual semantics, thereby mitigating individual-specific biases and learning robust, domain-invariant features. Extensive experiments on six MedTS datasets demonstrate that TS-P$^2$CL consistently outperforms fourteen methods in both subject-dependent and subject-independent settings. 

**Abstract (ZH)**: 医疗时间序列（MedTS）分类对于智能医疗至关重要，但由于个体间极大的异质性限制了跨个体生成能力，其效果受到了严重限制。尽管在架构创新和迁移学习技术方面取得了进步，当前方法仍受限于特定模态的归纳偏见，这限制了它们学习通用不变表示的能力。为克服这一限制，我们提出了TS-P$^2$CL，一种新颖的即插即用框架，该框架利用预训练视觉模型的通用模式识别能力。我们引入了一种视觉导向的范式，将1D生理信号转换为2D伪图像，建立了视觉域的桥梁。这种转换使得可以从自然图像中隐式访问丰富的语义先验。在这一统一空间内，我们采用了一种双对比学习策略：模内一致性确保时间上的连贯性，跨模对齐将时间序列动力学与视觉语义对齐，从而减轻个体特异性偏差并学习鲁棒的、跨域不变的特征。在六个MedTS数据集上的广泛实验表明，TS-P$^2$CL在主体依赖和主体无关设置中均一致地优于十四种方法。 

---
# Predicting Depth Maps from Single RGB Images and Addressing Missing Information in Depth Estimation 

**Title (ZH)**: 从单张RGB图像预测深度图并解决深度估计中的信息缺失问题 

**Authors**: Mohamad Mofeed Chaar, Jamal Raiyn, Galia Weidl  

**Link**: [PDF](https://arxiv.org/pdf/2509.17686)  

**Abstract**: Depth imaging is a crucial area in Autonomous Driving Systems (ADS), as it plays a key role in detecting and measuring objects in the vehicle's surroundings. However, a significant challenge in this domain arises from missing information in Depth images, where certain points are not measurable due to gaps or inconsistencies in pixel data. Our research addresses two key tasks to overcome this challenge. First, we developed an algorithm using a multi-layered training approach to generate Depth images from a single RGB image. Second, we addressed the issue of missing information in Depth images by applying our algorithm to rectify these gaps, resulting in Depth images with complete and accurate data. We further tested our algorithm on the Cityscapes dataset and successfully resolved the missing information in its Depth images, demonstrating the effectiveness of our approach in real-world urban environments. 

**Abstract (ZH)**: 深度成像在自动驾驶系统中的应用是自动驾驶系统的关键领域，它在检测和测量车辆周围物体方面发挥着重要作用。然而，该领域面临的显著挑战来自于深度图像中的缺失信息，某些点由于像素数据中的间隙或不一致性而无法测量。我们的研究针对这一挑战，提出了两个关键任务。首先，我们开发了一种多层训练方法的算法，以生成来自单个RGB图像的深度图像。其次，我们通过将该算法应用于纠正深度图像中的缺失信息，从而获得了完整且准确的数据。我们进一步在Cityscapes数据集上测试了该算法，并成功解决了其深度图像中的缺失信息，展示了该方法在真实城市环境中的有效性。 

---
# SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models 

**Title (ZH)**: SD-VLM: 空间度量与理解的深度编码视觉-语言模型 

**Authors**: Pingyi Chen, Yujing Lou, Shen Cao, Jinhui Guo, Lubin Fan, Yue Wu, Lin Yang, Lizhuang Ma, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.17664)  

**Abstract**: While vision language models (VLMs) excel in 2D semantic visual understanding, their ability to quantitatively reason about 3D spatial relationships remains under-explored, due to the deficiency of 2D images' spatial representation ability. In this paper, we analyze the problem hindering VLMs' spatial understanding abilities and propose SD-VLM, a novel framework that significantly enhances fundamental spatial perception abilities of VLMs through two key contributions: (1) propose Massive Spatial Measuring and Understanding (MSMU) dataset with precise spatial annotations, and (2) introduce a simple depth positional encoding method strengthening VLMs' spatial awareness. MSMU dataset covers massive quantitative spatial tasks with 700K QA pairs, 2.5M physical numerical annotations, and 10K chain-of-thought augmented samples. We have trained SD-VLM, a strong generalist VLM which shows superior quantitative spatial measuring and understanding capability. SD-VLM not only achieves state-of-the-art performance on our proposed MSMU-Bench, but also shows spatial generalization abilities on other spatial understanding benchmarks including Q-Spatial and SpatialRGPT-Bench. Extensive experiments demonstrate that SD-VLM outperforms GPT-4o and Intern-VL3-78B by 26.91% and 25.56% respectively on MSMU-Bench. Code and models are released at this https URL. 

**Abstract (ZH)**: 尽管视觉语言模型（VLMs）在二维语义视觉理解方面表现出色，但在定量推理三维空间关系方面的能力仍鲜有探索，这归因于二维图像在空间表示能力上的不足。在本文中，我们分析了阻碍VLMs空间理解能力的问题，并提出了一种新型框架SD-VLM，该框架通过两大关键贡献显著增强了VLMs的基本空间感知能力：（1）提出了一个包含精确空间注释的Massive Spatial Measuring and Understanding (MSMU)数据集；（2）引入了一种简单的深度位置编码方法，增强VLMs的空间意识。MSMU数据集涵盖了700K对QA、250万物理数值标注以及1万个链式思考增强样本的重大定量空间任务。我们训练了SD-VLM，这是一种强大的通用型VLM，展示了优越的定量空间测量和理解能力。SD-VLM不仅在我们提出的MSMU-Bench上达到了最先进的性能，还在其他空间理解基准测试如Q-Spatial和SpatialRGPT-Bench上展示了空间泛化能力。广泛实验表明，SD-VLM在MSMU-Bench上的性能分别比GPT-4o和Intern-VL3-78B高出26.91%和25.56%。代码和模型已发布在https://this-url。 

---
# A$^2$M$^2$-Net: Adaptively Aligned Multi-Scale Moment for Few-Shot Action Recognition 

**Title (ZH)**: A$^2$M$^2$-Net:自适应对齐多尺度矩方法在少量样本动作识别中的应用 

**Authors**: Zilin Gao, Qilong Wang, Bingbing Zhang, Qinghua Hu, Peihua Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.17638)  

**Abstract**: Thanks to capability to alleviate the cost of large-scale annotation, few-shot action recognition (FSAR) has attracted increased attention of researchers in recent years. Existing FSAR approaches typically neglect the role of individual motion pattern in comparison, and under-explore the feature statistics for video dynamics. Thereby, they struggle to handle the challenging temporal misalignment in video dynamics, particularly by using 2D backbones. To overcome these limitations, this work proposes an adaptively aligned multi-scale second-order moment network, namely A$^2$M$^2$-Net, to describe the latent video dynamics with a collection of powerful representation candidates and adaptively align them in an instance-guided manner. To this end, our A$^2$M$^2$-Net involves two core components, namely, adaptive alignment (A$^2$ module) for matching, and multi-scale second-order moment (M$^2$ block) for strong representation. Specifically, M$^2$ block develops a collection of semantic second-order descriptors at multiple spatio-temporal scales. Furthermore, A$^2$ module aims to adaptively select informative candidate descriptors while considering the individual motion pattern. By such means, our A$^2$M$^2$-Net is able to handle the challenging temporal misalignment problem by establishing an adaptive alignment protocol for strong representation. Notably, our proposed method generalizes well to various few-shot settings and diverse metrics. The experiments are conducted on five widely used FSAR benchmarks, and the results show our A$^2$M$^2$-Net achieves very competitive performance compared to state-of-the-arts, demonstrating its effectiveness and generalization. 

**Abstract (ZH)**: 基于自适应对齐的多尺度二阶矩网络：Few-Shot 动作识别中的自适应对齐视频动力学描述 

---
# Interpreting Attention Heads for Image-to-Text Information Flow in Large Vision-Language Models 

**Title (ZH)**: 图像到文本信息流中注意力头的解释 

**Authors**: Jinyeong Kim, Seil Kang, Jiwoo Park, Junhyeok Kim, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17588)  

**Abstract**: Large Vision-Language Models (LVLMs) answer visual questions by transferring information from images to text through a series of attention heads. While this image-to-text information flow is central to visual question answering, its underlying mechanism remains difficult to interpret due to the simultaneous operation of numerous attention heads. To address this challenge, we propose head attribution, a technique inspired by component attribution methods, to identify consistent patterns among attention heads that play a key role in information transfer. Using head attribution, we investigate how LVLMs rely on specific attention heads to identify and answer questions about the main object in an image. Our analysis reveals that a distinct subset of attention heads facilitates the image-to-text information flow. Remarkably, we find that the selection of these heads is governed by the semantic content of the input image rather than its visual appearance. We further examine the flow of information at the token level and discover that (1) text information first propagates to role-related tokens and the final token before receiving image information, and (2) image information is embedded in both object-related and background tokens. Our work provides evidence that image-to-text information flow follows a structured process, and that analysis at the attention-head level offers a promising direction toward understanding the mechanisms of LVLMs. 

**Abstract (ZH)**: 大型视觉语言模型（LVLMs）通过一系列注意力层将图像信息转移到文本中以回答视觉问题。虽然这种图像到文本的信息流是视觉问答的关键，但受到多个注意力层同时操作的影响，其内在机制仍难以解释。为了解决这一挑战，我们提出了一种基于组件归因方法的注意力头归因技术，以识别在信息转移中起关键作用的注意力头中的一致模式。利用注意力头归因，我们研究了LVLMs如何依赖特定的注意力头来识别和回答图像中主要对象的问题。我们的分析表明，一组独特的注意力头促进了图像到文本的信息流。令人惊讶的是，我们发现这些头的选择是由输入图像的语义内容而非其视觉外观所驱动的。我们进一步在token级别上研究了信息流，并发现：（1）文本信息首先传播到角色相关token和最终token，然后接收图像信息；（2）图像信息嵌入到与对象相关和背景相关的token中。我们的研究提供了图像到文本信息流遵循有序过程的证据，并表明在注意力头层面进行分析是有希望理解LVLM机制的方向。 

---
# MRN: Harnessing 2D Vision Foundation Models for Diagnosing Parkinson's Disease with Limited 3D MR Data 

**Title (ZH)**: MRN: 利用有限的3D MR数据和2D视觉基础模型诊断帕金森病 

**Authors**: Ding Shaodong, Liu Ziyang, Zhou Yijun, Liu Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.17566)  

**Abstract**: The automatic diagnosis of Parkinson's disease is in high clinical demand due to its prevalence and the importance of targeted treatment. Current clinical practice often relies on diagnostic biomarkers in QSM and NM-MRI images. However, the lack of large, high-quality datasets makes training diagnostic models from scratch prone to overfitting. Adapting pre-trained 3D medical models is also challenging, as the diversity of medical imaging leads to mismatches in voxel spacing and modality between pre-training and fine-tuning data. In this paper, we address these challenges by leveraging 2D vision foundation models (VFMs). Specifically, we crop multiple key ROIs from NM and QSM images, process each ROI through separate branches to compress the ROI into a token, and then combine these tokens into a unified patient representation for classification. Within each branch, we use 2D VFMs to encode axial slices of the 3D ROI volume and fuse them into the ROI token, guided by an auxiliary segmentation head that steers the feature extraction toward specific brain nuclei. Additionally, we introduce multi-ROI supervised contrastive learning, which improves diagnostic performance by pulling together representations of patients from the same class while pushing away those from different classes. Our approach achieved first place in the MICCAI 2025 PDCADxFoundation challenge, with an accuracy of 86.0% trained on a dataset of only 300 labeled QSM and NM-MRI scans, outperforming the second-place method by 5.5%.These results highlight the potential of 2D VFMs for clinical analysis of 3D MR images. 

**Abstract (ZH)**: 基于2D视觉基础模型的帕金森病自动诊断研究 

---
# An Empirical Study on the Robustness of YOLO Models for Underwater Object Detection 

**Title (ZH)**: YOLO模型在水下目标检测中鲁棒性的一项实证研究 

**Authors**: Edwine Nabahirwa, Wei Song, Minghua Zhang, Shufan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.17561)  

**Abstract**: Underwater object detection (UOD) remains a critical challenge in computer vision due to underwater distortions which degrade low-level features and compromise the reliability of even state-of-the-art detectors. While YOLO models have become the backbone of real-time object detection, little work has systematically examined their robustness under these uniquely challenging conditions. This raises a critical question: Are YOLO models genuinely robust when operating under the chaotic and unpredictable conditions of underwater environments? In this study, we present one of the first comprehensive evaluations of recent YOLO variants (YOLOv8-YOLOv12) across six simulated underwater environments. Using a unified dataset of 10,000 annotated images from DUO and Roboflow100, we not only benchmark model robustness but also analyze how distortions affect key low-level features such as texture, edges, and color. Our findings show that (1) YOLOv12 delivers the strongest overall performance but is highly vulnerable to noise, and (2) noise disrupts edge and texture features, explaining the poor detection performance in noisy images. Class imbalance is a persistent challenge in UOD. Experiments revealed that (3) image counts and instance frequency primarily drive detection performance, while object appearance exerts only a secondary influence. Finally, we evaluated lightweight training-aware strategies: noise-aware sample injection, which improves robustness in both noisy and real-world conditions, and fine-tuning with advanced enhancement, which boosts accuracy in enhanced domains but slightly lowers performance in original data, demonstrating strong potential for domain adaptation, respectively. Together, these insights provide practical guidance for building resilient and cost-efficient UOD systems. 

**Abstract (ZH)**: 水下目标检测中的YOLO模型鲁棒性研究：基于六个模拟水下环境的全面评估 

---
# Real-Time Fish Detection in Indonesian Marine Ecosystems Using Lightweight YOLOv10-nano Architecture 

**Title (ZH)**: 使用轻量级YOLOv10-nano架构在印度尼西亚marine生态系统中进行实时鱼类检测 

**Authors**: Jonathan Wuntu, Muhamad Dwisnanto Putro, Rendy Syahputra  

**Link**: [PDF](https://arxiv.org/pdf/2509.17406)  

**Abstract**: Indonesia's marine ecosystems, part of the globally recognized Coral Triangle, are among the richest in biodiversity, requiring efficient monitoring tools to support conservation. Traditional fish detection methods are time-consuming and demand expert knowledge, prompting the need for automated solutions. This study explores the implementation of YOLOv10-nano, a state-of-the-art deep learning model, for real-time marine fish detection in Indonesian waters, using test data from Bunaken National Marine Park. YOLOv10's architecture, featuring improvements like the CSPNet backbone, PAN for feature fusion, and Pyramid Spatial Attention Block, enables efficient and accurate object detection even in complex environments. The model was evaluated on the DeepFish and OpenImages V7-Fish datasets. Results show that YOLOv10-nano achieves a high detection accuracy with mAP50 of 0.966 and mAP50:95 of 0.606 while maintaining low computational demand (2.7M parameters, 8.4 GFLOPs). It also delivered an average inference speed of 29.29 FPS on the CPU, making it suitable for real-time deployment. Although OpenImages V7-Fish alone provided lower accuracy, it complemented DeepFish in enhancing model robustness. Overall, this study demonstrates YOLOv10-nano's potential for efficient, scalable marine fish monitoring and conservation applications in data-limited environments. 

**Abstract (ZH)**: 印度尼西亚的海洋生态系统，作为全球公认的珊瑚三角区的一部分，生物多样性极为丰富，需要高效的监测工具以支持保护工作。传统的鱼类检测方法耗时且需要专业知识，促使自动解决方案的需求。本研究探讨了将最先进的深度学习模型YOLOv10-nano应用于印度尼西亚水域内实时海洋鱼类检测，使用布纳肯国家海洋公园的测试数据。YOLOv10的架构包括CSPNet骨干、PAN特征融合以及金字塔空间注意力块等改进，使其即使在复杂环境中也能实现高效且准确的目标检测。该模型在DeepFish和OpenImages V7-Fish数据集上进行了评估。结果显示，YOLOv10-nano实现了较高的检测精度，mAP50为0.966，mAP50:95为0.606，同时保持较低的计算需求（2.7M参数，8.4 GFLOPs）。此外，它在CPU上的平均推理速度达到29.29 FPS，使其适合进行实时部署。虽然单独使用OpenImages V7-Fish数据集的精度较低，但它增强了DeepFish数据集的模型稳健性。总体而言，本研究展示了YOLOv10-nano在数据匮乏环境中进行高效、可扩展的海洋鱼类监测和保护应用的潜力。 

---
# Interpreting vision transformers via residual replacement model 

**Title (ZH)**: 通过残差替代模型解释视觉变换器 

**Authors**: Jinyeong Kim, Junhyeok Kim, Yumin Shim, Joohyeok Kim, Sunyoung Jung, Seong Jae Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17401)  

**Abstract**: How do vision transformers (ViTs) represent and process the world? This paper addresses this long-standing question through the first systematic analysis of 6.6K features across all layers, extracted via sparse autoencoders, and by introducing the residual replacement model, which replaces ViT computations with interpretable features in the residual stream. Our analysis reveals not only a feature evolution from low-level patterns to high-level semantics, but also how ViTs encode curves and spatial positions through specialized feature types. The residual replacement model scalably produces a faithful yet parsimonious circuit for human-scale interpretability by significantly simplifying the original computations. As a result, this framework enables intuitive understanding of ViT mechanisms. Finally, we demonstrate the utility of our framework in debiasing spurious correlations. 

**Abstract (ZH)**: 视觉变换器（ViTs）如何表示和处理世界？本文通过系统分析6600个跨所有层提取的特征，并引入残差替换模型，来解答这一长期存在的问题。我们的分析不仅揭示了从低级模式到高级语义的特征演化过程，还展示了ViTs通过专门的特征类型编码曲线和空间位置的方式。残差替换模型可扩展地生成一个简洁且忠实的人类可解释电路，显著简化了原始计算。由此，该框架使ViT机制的理解变得直观。最后，我们展示了该框架在消除无稽之比中的应用价值。 

---
# Pre-Trained CNN Architecture for Transformer-Based Image Caption Generation Model 

**Title (ZH)**: 基于预训练CNN架构的Transformer图像Caption生成模型 

**Authors**: Amanuel Tafese Dufera  

**Link**: [PDF](https://arxiv.org/pdf/2509.17365)  

**Abstract**: Automatic image captioning, a multifaceted task bridging computer vision and natural lan- guage processing, aims to generate descriptive textual content from visual input. While Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks have achieved significant advancements, they present limitations. The inherent sequential nature of RNNs leads to sluggish training and inference times. LSTMs further struggle with retaining information from earlier sequence elements when dealing with very long se- quences. This project presents a comprehensive guide to constructing and comprehending transformer models for image captioning. Transformers employ self-attention mechanisms, capturing both short- and long-range dependencies within the data. This facilitates efficient parallelization during both training and inference phases. We leverage the well-established Transformer architecture, recognized for its effectiveness in managing sequential data, and present a meticulous methodology. Utilizing the Flickr30k dataset, we conduct data pre- processing, construct a model architecture that integrates an EfficientNetB0 CNN for fea- ture extraction, and train the model with attention mechanisms incorporated. Our approach exemplifies the utilization of parallelization for efficient training and inference. You can find the project on GitHub. 

**Abstract (ZH)**: 自动图像配 captioning：一种连接计算机视觉和自然语言处理的多面任务，旨在从视觉输入生成描述性文本内容。尽管卷积神经网络（CNNs）和长短期记忆（LSTM）网络取得了显著进展，但仍存在局限性。RNNs 的固有序列性质导致训练和推理时间缓慢。LSTMs 在处理非常长的序列时，难以保留早期序列元素的信息。本项目提供了一种构建和理解用于图像配 captioning 的变压器模型的全面指南。变压器利用自注意力机制，在数据中捕获短程和长程依赖关系，这使得在训练和推理阶段实现高效并行化成为可能。我们利用了成熟的 Transformer 架构，该架构因其在处理序列数据方面的有效性而广受认可，并呈现了一种细致的方法。我们使用 Flickr30k 数据集进行数据预处理，构建了一个包含 EfficientNetB0 CNN 用于特征提取的模型架构，并通过集成注意力机制进行训练。我们的方法展示了并行化在高效训练和推理中的应用。该项目可以在 GitHub 上找到。 

---
# Guided and Unguided Conditional Diffusion Mechanisms for Structured and Semantically-Aware 3D Point Cloud Generation 

**Title (ZH)**: 指导与非指导的条件扩散机制用于结构化和语义感知的3D点云生成 

**Authors**: Gunner Stone, Sushmita Sarker, Alireza Tavakkoli  

**Link**: [PDF](https://arxiv.org/pdf/2509.17206)  

**Abstract**: Generating realistic 3D point clouds is a fundamental problem in computer vision with applications in remote sensing, robotics, and digital object modeling. Existing generative approaches primarily capture geometry, and when semantics are considered, they are typically imposed post hoc through external segmentation or clustering rather than integrated into the generative process itself. We propose a diffusion-based framework that embeds per-point semantic conditioning directly within generation. Each point is associated with a conditional variable corresponding to its semantic label, which guides the diffusion dynamics and enables the joint synthesis of geometry and semantics. This design produces point clouds that are both structurally coherent and segmentation-aware, with object parts explicitly represented during synthesis. Through a comparative analysis of guided and unguided diffusion processes, we demonstrate the significant impact of conditional variables on diffusion dynamics and generation quality. Extensive experiments validate the efficacy of our approach, producing detailed and accurate 3D point clouds tailored to specific parts and features. 

**Abstract (ZH)**: 基于扩散的点云生成框架：直接嵌入点语义条件以同时合成几何与语义 

---
# Echo-Path: Pathology-Conditioned Echo Video Generation 

**Title (ZH)**: ECHO-路径：基于病理条件的回声视频生成 

**Authors**: Kabir Hamzah Muhammad, Marawan Elbatel, Yi Qin, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.17190)  

**Abstract**: Cardiovascular diseases (CVDs) remain the leading cause of mortality globally, and echocardiography is critical for diagnosis of both common and congenital cardiac conditions. However, echocardiographic data for certain pathologies are scarce, hindering the development of robust automated diagnosis models. In this work, we propose Echo-Path, a novel generative framework to produce echocardiogram videos conditioned on specific cardiac pathologies. Echo-Path can synthesize realistic ultrasound video sequences that exhibit targeted abnormalities, focusing here on atrial septal defect (ASD) and pulmonary arterial hypertension (PAH). Our approach introduces a pathology-conditioning mechanism into a state-of-the-art echo video generator, allowing the model to learn and control disease-specific structural and motion patterns in the heart. Quantitative evaluation demonstrates that the synthetic videos achieve low distribution distances, indicating high visual fidelity. Clinically, the generated echoes exhibit plausible pathology markers. Furthermore, classifiers trained on our synthetic data generalize well to real data and, when used to augment real training sets, it improves downstream diagnosis of ASD and PAH by 7\% and 8\% respectively. Code, weights and dataset are available here this https URL 

**Abstract (ZH)**: 心血管疾病（CVDs）仍然是全球最主要的死亡原因，而超声心动图对于诊断常见和先天性心脏疾病至关重要。然而，某些病理状况的超声心动图数据稀缺，阻碍了稳健的自动化诊断模型的发展。在本文中，我们提出了一种名为Echo-Path的新颖生成框架，用于生成条件化的超声心动图视频，以特定心臟病理状况为条件。Echo-Path可以合成展现目标异常的现实超声视频序列，主要针对房间隔缺损（ASD）和肺动脉高压（PAH）。我们的方法将病理条件机制引入最先进的回声视频生成器中，使模型能够学习和控制心脏中特定疾病的结构和运动模式。定量评估表明，合成视频在分布距离上表现较低，表明具有高度视觉保真度。临床应用中，生成的回声图像表现出合理的病理标志。此外，基于我们合成数据训练的分类器在实际数据上有良好的泛化能力，并且当用于增强实际训练集时，分别将房间隔缺损和肺动脉高压的下游诊断准确率提高了7%和8%。代码、权重和数据集可在此链接获取：https://xxxxxx（原英文中的URL部分未提供具体链接，保持了原文的形式） 

---
# Ambiguous Medical Image Segmentation Using Diffusion Schrödinger Bridge 

**Title (ZH)**: 模糊医疗图像分割的扩散薛定谔桥方法 

**Authors**: Lalith Bharadwaj Baru, Kamalaker Dadi, Tapabrata Chakraborti, Raju S. Bapi  

**Link**: [PDF](https://arxiv.org/pdf/2509.17187)  

**Abstract**: Accurate segmentation of medical images is challenging due to unclear lesion boundaries and mask variability. We introduce \emph{Segmentation Schödinger Bridge (SSB)}, the first application of Schödinger Bridge for ambiguous medical image segmentation, modelling joint image-mask dynamics to enhance performance. SSB preserves structural integrity, delineates unclear boundaries without additional guidance, and maintains diversity using a novel loss function. We further propose the \emph{Diversity Divergence Index} ($D_{DDI}$) to quantify inter-rater variability, capturing both diversity and consensus. SSB achieves state-of-the-art performance on LIDC-IDRI, COCA, and RACER (in-house) datasets. 

**Abstract (ZH)**: 医疗图像准确分割具有挑战性，由于病变边界模糊和掩码变化性。我们提出了分割薛定谔桥（SSB），这是将薛定谔桥首次应用于含糊不清的医疗图像分割中，通过建模联合图像-掩码动力学以提高性能。SSB 保持结构完整性，无需额外指导即可界定模糊边界，并使用新型损失函数维持多样性。我们进一步提出了多样性偏差指数（$D_{DDI}$）以量化标注者间变异性，同时捕捉多样性和共识性。SSB 在 LIDC-IDRI、COCA 和 RACER（内部）数据集中达到了最先进的性能。 

---
# When Color-Space Decoupling Meets Diffusion for Adverse-Weather Image Restoration 

**Title (ZH)**: 当颜色空间解藕遇到扩散方法在恶劣天气图像恢复中的应用 

**Authors**: Wenxuan Fang, Jili Fan, Chao Wang, Xiantao Hu, Jiangwei Weng, Ying Tai, Jian Yang, Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.17024)  

**Abstract**: Adverse Weather Image Restoration (AWIR) is a highly challenging task due to the unpredictable and dynamic nature of weather-related degradations. Traditional task-specific methods often fail to generalize to unseen or complex degradation types, while recent prompt-learning approaches depend heavily on the degradation estimation capabilities of vision-language models, resulting in inconsistent restorations. In this paper, we propose \textbf{LCDiff}, a novel framework comprising two key components: \textit{Lumina-Chroma Decomposition Network} (LCDN) and \textit{Lumina-Guided Diffusion Model} (LGDM). LCDN processes degraded images in the YCbCr color space, separately handling degradation-related luminance and degradation-invariant chrominance components. This decomposition effectively mitigates weather-induced degradation while preserving color fidelity. To further enhance restoration quality, LGDM leverages degradation-related luminance information as a guiding condition, eliminating the need for explicit degradation prompts. Additionally, LGDM incorporates a \textit{Dynamic Time Step Loss} to optimize the denoising network, ensuring a balanced recovery of both low- and high-frequency features in the image. Finally, we present DriveWeather, a comprehensive all-weather driving dataset designed to enable robust evaluation. Extensive experiments demonstrate that our approach surpasses state-of-the-art methods, setting a new benchmark in AWIR. The dataset and code are available at: this https URL. 

**Abstract (ZH)**: 恶劣天气图像恢复 (Adverse Weather Image Restoration, AWIR) 是一项极具挑战性的任务，由于天气相关退化具有不可预测和动态的性质。传统的方法往往难以泛化到未见过或复杂的退化类型，而近期的提示学习方法则高度依赖视觉-语言模型的退化估计能力，导致恢复结果不一致。在本文中，我们提出了一种名为LCDiff的新型框架，包括两个关键组件：Lumina-Chroma 分解网络 (LCDN) 和 Lumina-引导扩散模型 (LGDM)。LCDN 在 YCbCr 颜色空间中处理退化图像，分别处理与退化相关的亮度和与退化无关的色度成分，有效减轻由天气引起的退化，同时保留色彩的保真度。为进一步提高恢复质量，LGDM 利用与退化相关的亮度信息作为引导条件，从而消除显式退化提示的需求。此外，LGDM 还引入了动态时间步长损失，以优化去噪网络，确保图像中低频和高频特征的平衡恢复。最后，我们提出了 DriveWeather，这是一个全面的全天候驾驶数据集，旨在进行稳健评估。大量实验结果表明，我们的方法超越了现有最先进的方法，建立了 AWIR 的新基准。数据集和代码可在以下链接获取：this https URL。 

---
# The 1st Solution for 7th LSVOS RVOS Track: SaSaSa2VA 

**Title (ZH)**: 7th LSVOS RVOS Track: SaSaSa2VA的第一解决方案 

**Authors**: Quanzhu Niu, Dengxian Gong, Shihao Chen, Tao Zhang, Yikang Zhou, Haobo Yuan, Lu Qi, Xiangtai Li, Shunping Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.16972)  

**Abstract**: Referring video object segmentation (RVOS) requires segmenting and tracking objects in videos conditioned on natural-language expressions, demanding fine-grained understanding of both appearance and motion. Building on Sa2VA, which couples a Multi-modal Large Language Model (MLLM) with the video segmentation model SAM2, we identify two key bottlenecks that limit segmentation performance: sparse frame sampling and reliance on a single [SEG] token for an entire video. We propose Segmentation Augmented and Selective Averaged Sa2VA SaSaSa2VA to address these issues. On the 7th LSVOS Challenge (RVOS track), SaSaSa2VA achieves a $J\&F$ of 67.45, ranking first and surpassing the runner-up by 2.80 points. This result and ablation studies demonstrate that efficient segmentation augmentation and test-time ensembling substantially enhance grounded MLLMs for RVOS. The code is released in Sa2VA repository: this https URL. 

**Abstract (ZH)**: 基于自然语言表达的视频对象分割（RVOS）要求在视频中根据自然语言表达对对象进行分割和跟踪，这需要对外观和运动进行精细的理解。我们在结合多模态大型语言模型（MLLM）和视频分割模型SAM2的Sa2VA基础上，识别出两个限制分割性能的关键瓶颈：稀疏的帧采样和整个视频依赖单一[SEG]标记。我们提出了一种分割增强和选择加权Sa2VA（SaSaSa2VA）方法来解决这些问题。在第7届LSVOS挑战赛（RVOS赛道）中，SaSaSa2VA取得了67.45的$J\&F$分数，排名第一，并且比亚军高出2.80分。这一结果和消融实验表明，高效的分割增强和测试时集成显著提升了基于MLLM的RVOS。代码发布在Sa2VA存储库中：this https URL。 

---
# PGSTalker: Real-Time Audio-Driven Talking Head Generation via 3D Gaussian Splatting with Pixel-Aware Density Control 

**Title (ZH)**: PGSTalker：基于3D高斯点绘制和像素感知密度控制的实时音频驱动头部生成 

**Authors**: Tianheng Zhu, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.16922)  

**Abstract**: Audio-driven talking head generation is crucial for applications in virtual reality, digital avatars, and film production. While NeRF-based methods enable high-fidelity reconstruction, they suffer from low rendering efficiency and suboptimal audio-visual synchronization. This work presents PGSTalker, a real-time audio-driven talking head synthesis framework based on 3D Gaussian Splatting (3DGS). To improve rendering performance, we propose a pixel-aware density control strategy that adaptively allocates point density, enhancing detail in dynamic facial regions while reducing redundancy elsewhere. Additionally, we introduce a lightweight Multimodal Gated Fusion Module to effectively fuse audio and spatial features, thereby improving the accuracy of Gaussian deformation prediction. Extensive experiments on public datasets demonstrate that PGSTalker outperforms existing NeRF- and 3DGS-based approaches in rendering quality, lip-sync precision, and inference speed. Our method exhibits strong generalization capabilities and practical potential for real-world deployment. 

**Abstract (ZH)**: 基于音频驱动的Head生成对于虚拟现实、数字 avatar 和电影制作具有重要意义。尽管基于NeRF的方法可以实现高保真重建，但它们存在渲染效率低和音视频同步不佳的问题。本文提出了基于3D高斯点渲染（3DGS）的实时音频驱动Head合成框架PGSTalker。为了提高渲染性能，我们提出了一种像素感知的密度控制策略，该策略能够自适应地分配点密度，在动态面部区域增强细节，同时在其他区域减少冗余。此外，我们引入了一种轻量级的多模态门控融合模块，以有效地融合音频和空间特征，从而提高高斯变形预测的准确性。在公共数据集上的广泛实验表明，PGSTalker在渲染质量、嘴唇同步精度和推理速度方面均优于现有基于NeRF和3DGS的方法。我们的方法展示了强大的泛化能力和在实际部署中的实用潜力。 

---
# PhysHDR: When Lighting Meets Materials and Scene Geometry in HDR Reconstruction 

**Title (ZH)**: PhysHDR：当照明、材料和场景几何在HDR重建中相遇时 

**Authors**: Hrishav Bakul Barua, Kalin Stefanov, Ganesh Krishnasamy, KokSheik Wong, Abhinav Dhall  

**Link**: [PDF](https://arxiv.org/pdf/2509.16869)  

**Abstract**: Low Dynamic Range (LDR) to High Dynamic Range (HDR) image translation is a fundamental task in many computational vision problems. Numerous data-driven methods have been proposed to address this problem; however, they lack explicit modeling of illumination, lighting, and scene geometry in images. This limits the quality of the reconstructed HDR images. Since lighting and shadows interact differently with different materials, (e.g., specular surfaces such as glass and metal, and lambertian or diffuse surfaces such as wood and stone), modeling material-specific properties (e.g., specular and diffuse reflectance) has the potential to improve the quality of HDR image reconstruction. This paper presents PhysHDR, a simple yet powerful latent diffusion-based generative model for HDR image reconstruction. The denoising process is conditioned on lighting and depth information and guided by a novel loss to incorporate material properties of surfaces in the scene. The experimental results establish the efficacy of PhysHDR in comparison to a number of recent state-of-the-art methods. 

**Abstract (ZH)**: 低动态范围（LDR）到高动态范围（HDR）图像转换是许多计算视觉问题中的一个基本任务。提出了一系列数据驱动的方法来解决这一问题，但这些方法缺乏对图像照明、光照和场景几何的显式建模，这限制了重建的HDR图像质量。由于照明和阴影与不同的材料（例如，镜面表面如玻璃和金属，以及漫射表面如木材和石头）相互作用的方式不同，建模材料特定属性（例如，镜面反射率和漫反射率）有可能提高HDR图像重建的质量。本文提出PhysHDR，这是一种基于隐式扩散的生成模型，用于HDR图像重建。去噪过程基于照明和深度信息，并由一种新颖的损失函数引导，以结合场景中表面的材料属性。实验结果表明，PhysHDR在与多种最新方法的比较中具有有效性。 

---
# Text-Scene: A Scene-to-Language Parsing Framework for 3D Scene Understanding 

**Title (ZH)**: 场景到文本：一种三维场景理解的场景到语言解析框架 

**Authors**: Haoyuan Li, Rui Liu, Hehe Fan, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16721)  

**Abstract**: Enabling agents to understand and interact with complex 3D scenes is a fundamental challenge for embodied artificial intelligence systems. While Multimodal Large Language Models (MLLMs) have achieved significant progress in 2D image understanding, extending such capabilities to 3D scenes remains difficult: 1) 3D environment involves richer concepts such as spatial relationships, affordances, physics, layout, and so on, 2) the absence of large-scale 3D vision-language datasets has posed a significant obstacle. In this paper, we introduce Text-Scene, a framework that automatically parses 3D scenes into textual descriptions for scene understanding. Given a 3D scene, our model identifies object attributes and spatial relationships, and then generates a coherent summary of the whole scene, bridging the gap between 3D observation and language without requiring human-in-the-loop intervention. By leveraging both geometric analysis and MLLMs, Text-Scene produces descriptions that are accurate, detailed, and human-interpretable, capturing object-level details and global-level context. Experimental results on benchmarks demonstrate that our textual parses can faithfully represent 3D scenes and benefit downstream tasks. To evaluate the reasoning capability of MLLMs, we present InPlan3D, a comprehensive benchmark for 3D task planning, consisting of 3174 long-term planning tasks across 636 indoor scenes. We emphasize clarity and accessibility in our approach, aiming to make 3D scene content understandable through language. Code and datasets will be released. 

**Abstract (ZH)**: 使智能体理解并交互复杂3D场景是体化人工智能系统的根本挑战。虽然多模态大型语言模型已经在2D图像理解方面取得了显著进展，但将其能力扩展到3D场景仍然困难重重：1) 3D环境涉及更丰富的概念，如空间关系、可用性、物理特性、布局等，2) 缺乏大规模3D视觉-语言数据集构成了重大障碍。在本文中，我们引入了Text-Scene框架，该框架能够自动将3D场景解析为文本描述以进行场景理解。给定一个3D场景，我们的模型识别人物属性和空间关系，然后生成整个场景的连贯总结，填补3D观测与语言之间的gap，而无需人工干预。通过结合几何分析和多模态大型语言模型，Text-Scene生成的描述准确、详细且易于人类理解，捕捉到物体级别的细节和全局上下文。基准测试结果表明，我们的文本解析能够忠实表示3D场景并利于下游任务。为了评估大型语言模型的推理能力，我们提出了InPlan3D，这是一个全面的3D任务规划基准，包含来自636个室内场景的3174个长期规划任务。我们强调方法的清晰性和可访问性，旨在通过语言使3D场景内容变得可理解。代码和数据集将公开发布。 

---
# ProtoVQA: An Adaptable Prototypical Framework for Explainable Fine-Grained Visual Question Answering 

**Title (ZH)**: ProtoVQA：一种可调节的原型框架，用于解释性细粒度视觉问答 

**Authors**: Xingjian Diao, Weiyi Wu, Keyi Kong, Peijun Qing, Xinwen Xu, Ming Cheng, Soroush Vosoughi, Jiang Gui  

**Link**: [PDF](https://arxiv.org/pdf/2509.16680)  

**Abstract**: Visual Question Answering (VQA) is increasingly used in diverse applications ranging from general visual reasoning to safety-critical domains such as medical imaging and autonomous systems, where models must provide not only accurate answers but also explanations that humans can easily understand and verify. Prototype-based modeling has shown promise for interpretability by grounding predictions in semantically meaningful regions for purely visual reasoning tasks, yet remains underexplored in the context of VQA. We present ProtoVQA, a unified prototypical framework that (i) learns question-aware prototypes that serve as reasoning anchors, connecting answers to discriminative image regions, (ii) applies spatially constrained matching to ensure that the selected evidence is coherent and semantically relevant, and (iii) supports both answering and grounding tasks through a shared prototype backbone. To assess explanation quality, we propose the Visual-Linguistic Alignment Score (VLAS), which measures how well the model's attended regions align with ground-truth evidence. Experiments on Visual7W show that ProtoVQA yields faithful, fine-grained explanations while maintaining competitive accuracy, advancing the development of transparent and trustworthy VQA systems. 

**Abstract (ZH)**: 基于原型的视觉问答（ProtoVQA）：可解释的视觉推理与地面truth对齐评价方法 

---
# Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity 

**Title (ZH)**: 晶格玻尔兹曼模型学习现实世界像素动态性 

**Authors**: Guangze Zheng, Shijie Lin, Haobo Zuo, Si Si, Ming-Shan Wang, Changhong Fu, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.16527)  

**Abstract**: This work proposes the Lattice Boltzmann Model (LBM) to learn real-world pixel dynamicity for visual tracking. LBM decomposes visual representations into dynamic pixel lattices and solves pixel motion states through collision-streaming processes. Specifically, the high-dimensional distribution of the target pixels is acquired through a multilayer predict-update network to estimate the pixel positions and visibility. The predict stage formulates lattice collisions among the spatial neighborhood of target pixels and develops lattice streaming within the temporal visual context. The update stage rectifies the pixel distributions with online visual representations. Compared with existing methods, LBM demonstrates practical applicability in an online and real-time manner, which can efficiently adapt to real-world visual tracking tasks. Comprehensive evaluations of real-world point tracking benchmarks such as TAP-Vid and RoboTAP validate LBM's efficiency. A general evaluation of large-scale open-world object tracking benchmarks such as TAO, BFT, and OVT-B further demonstrates LBM's real-world practicality. 

**Abstract (ZH)**: Lattice Boltzmann Model for Learning Real-World Pixel Dynamicity in Visual Tracking 

---
# Thermal Imaging-based Real-time Fall Detection using Motion Flow and Attention-enhanced Convolutional Recurrent Architecture 

**Title (ZH)**: 基于热成像的运动流和注意力增强卷积循环架构实时跌倒检测 

**Authors**: Christopher Silver, Thangarajah Akilan  

**Link**: [PDF](https://arxiv.org/pdf/2509.16479)  

**Abstract**: Falls among seniors are a major public health issue. Existing solutions using wearable sensors, ambient sensors, and RGB-based vision systems face challenges in reliability, user compliance, and practicality. Studies indicate that stakeholders, such as older adults and eldercare facilities, prefer non-wearable, passive, privacy-preserving, and real-time fall detection systems that require no user interaction. This study proposes an advanced thermal fall detection method using a Bidirectional Convolutional Long Short-Term Memory (BiConvLSTM) model, enhanced with spatial, temporal, feature, self, and general attention mechanisms. Through systematic experimentation across hundreds of model variations exploring the integration of attention mechanisms, recurrent modules, and motion flow, we identified top-performing architectures. Among them, BiConvLSTM achieved state-of-the-art performance with a ROC-AUC of $99.7\%$ on the TSF dataset and demonstrated robust results on TF-66, a newly emerged, diverse, and privacy-preserving benchmark. These results highlight the generalizability and practicality of the proposed model, setting new standards for thermal fall detection and paving the way toward deployable, high-performance solutions. 

**Abstract (ZH)**: 老年人跌倒是一个重要的公共卫生问题。现有的使用可穿戴传感器、环境传感器和基于RGB的视觉系统的解决方案在可靠性和用户遵守方面面临挑战。研究表明，相关方，如老年人和养老设施，更倾向于非穿戴的、被动的、隐私保护的和实时的跌倒检测系统，不需要用户互动。本研究提出了一种先进的热跌倒检测方法，采用了双向卷积长短期记忆（BiConvLSTM）模型，并结合了空间、时间、特征、自我和一般的注意力机制。通过系统性实验跨数百种模型变体探索注意力机制、递归模块和运动流的集成，我们确定了性能最佳的架构。其中，BiConvLSTM在TSF数据集上取得了最先进的性能，ROCAUC达到99.7%，并在新出现的多样化和隐私保护基准TF-66上展示了稳健的结果。这些结果突显了所提出模型的普遍适用性和实用性，为热跌倒检测设定了新标准，并为可部署的高性能解决方案铺平了道路。 

---
# From Canopy to Ground via ForestGen3D: Learning Cross-Domain Generation of 3D Forest Structure from Aerial-to-Terrestrial LiDAR 

**Title (ZH)**: 从机载到地面：通过ForestGen3D学习三维森林结构的跨域生成 

**Authors**: Juan Castorena, E. Louise Loudermilk, Scott Pokswinski, Rodman Linn  

**Link**: [PDF](https://arxiv.org/pdf/2509.16346)  

**Abstract**: The 3D structure of living and non-living components in ecosystems plays a critical role in determining ecological processes and feedbacks from both natural and human-driven disturbances. Anticipating the effects of wildfire, drought, disease, or atmospheric deposition depends on accurate characterization of 3D vegetation structure, yet widespread measurement remains prohibitively expensive and often infeasible. We introduce ForestGen3D, a novel generative modeling framework that synthesizes high-fidelity 3D forest structure using only aerial LiDAR (ALS) inputs. ForestGen3D is based on conditional denoising diffusion probabilistic models (DDPMs) trained on co-registered ALS/TLS (terrestrial LiDAR) data. The model learns to generate TLS-like 3D point clouds conditioned on sparse ALS observations, effectively reconstructing occluded sub-canopy detail at scale. To ensure ecological plausibility, we introduce a geometric containment prior based on the convex hull of ALS observations and provide theoretical and empirical guarantees that generated structures remain spatially consistent. We evaluate ForestGen3D at tree, plot, and landscape scales using real-world data from mixed conifer ecosystems, and show that it produces high-fidelity reconstructions that closely match TLS references in terms of geometric similarity and biophysical metrics, such as tree height, DBH, crown diameter and crown volume. Additionally, we demonstrate that the containment property can serve as a practical proxy for generation quality in settings where TLS ground truth is unavailable. Our results position ForestGen3D as a scalable tool for ecological modeling, wildfire simulation, and structural fuel characterization in ALS-only environments. 

**Abstract (ZH)**: 3D生态系统中生活和非生活组件的结构在决定生态过程和自然或人为干扰的反馈中发挥着关键作用。预测野火、干旱、疾病或大气沉降等影响依赖于对3D植被结构的准确表征，但广泛测量仍普遍非常昂贵且往往不可行。我们介绍了ForestGen3D，这是一种新型生成建模框架，仅使用机载LiDAR (ALS) 输入即可合成高保真3D森林结构。ForestGen3D 基于在共注册ALS/TLS (机载LiDAR) 数据上训练的条件去噪扩散概率模型 (DDPMs)。该模型学习在稀疏ALS观测条件下生成TLS样式的3D点云，有效地在较大范围内重建被遮挡的亚冠层细节。为了确保生态合理性，我们引入了基于ALS观测凸包的几何包含先验，并提供了生成结构在空间上保持一致的理论和实证保证。我们在包含混合冷杉生态系统的实地数据上评估了ForestGen3D，结果显示它产生了高保真的重建结果，在几何相似性和生物物理指标（如树高、胸径、冠幅和冠体积）方面与TLS参考结果高度匹配。此外，我们展示了包含属性在缺乏TLS地面真实值的情况下可以作为生成质量的实用代理。我们的研究将ForestGen3D 定位为LiDAR仅环境中的生态建模、野火模拟和结构燃料表征的可扩展工具。 

---
# Stabilizing Information Flow Entropy: Regularization for Safe and Interpretable Autonomous Driving Perception 

**Title (ZH)**: 稳定信息流熵：安全可解释自主驾驶感知的正则化 

**Authors**: Haobo Yang, Shiyan Zhang, Zhuoyi Yang, Jilong Guo, Jun Yang, Xinyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16277)  

**Abstract**: Deep perception networks in autonomous driving traditionally rely on data-intensive training regimes and post-hoc anomaly detection, often disregarding fundamental information-theoretic constraints governing stable information processing. We reconceptualize deep neural encoders as hierarchical communication chains that incrementally compress raw sensory inputs into task-relevant latent features. Within this framework, we establish two theoretically justified design principles for robust perception: (D1) smooth variation of mutual information between consecutive layers, and (D2) monotonic decay of latent entropy with network depth. Our analysis shows that, under realistic architectural assumptions, particularly blocks comprising repeated layers of similar capacity, enforcing smooth information flow (D1) naturally encourages entropy decay (D2), thus ensuring stable compression. Guided by these insights, we propose Eloss, a novel entropy-based regularizer designed as a lightweight, plug-and-play training objective. Rather than marginal accuracy improvements, this approach represents a conceptual shift: it unifies information-theoretic stability with standard perception tasks, enabling explicit, principled detection of anomalous sensor inputs through entropy deviations. Experimental validation on large-scale 3D object detection benchmarks (KITTI and nuScenes) demonstrates that incorporating Eloss consistently achieves competitive or improved accuracy while dramatically enhancing sensitivity to anomalies, amplifying distribution-shift signals by up to two orders of magnitude. This stable information-compression perspective not only improves interpretability but also establishes a solid theoretical foundation for safer, more robust autonomous driving perception systems. 

**Abstract (ZH)**: 深度学习在自主驾驶中的感知网络传统上依赖于数据密集型的训练范式和事后异常检测，经常忽视稳定信息处理所受的基本信息论约束。我们重新概念化深度神经编码器为分层通信链，逐级压缩原始传感器输入至任务相关潜在特征。在此框架下，我们确立了两种理论依据稳健感知的设计原则：（D1）连续层间互信息的平滑变化，以及（D2）潜在熵随网络深度的单调衰减。我们的分析表明，在现实架构假设下，尤其是由类似容量重复层构成的模块，强制平滑信息流（D1）自然促进熵衰减（D2），从而确保稳定压缩。基于这些洞察，我们提出了一种基于熵的新颖正则化器Eloss，设计为轻量级的即插即用训练目标。这种方法不仅仅代表对边际准确性的提升，而是概念上的转变：它统一了信息论稳定性和标准感知任务，通过熵偏差实现异常传感器输入的显式、原则性检测。大规模3D物体检测基准（KITTI和nuScenes）的实验验证表明，集成Eloss在保持甚至提升准确性的基础上，显著增强对异常的敏感性，熵变化信号放大两倍以上。这种稳定的信息压缩视角不仅提高了可解释性，还为更安全、更鲁棒的自主驾驶感知系统奠定了坚实的理论基础。 

---
# Imaging Modalities-Based Classification for Lung Cancer Detection 

**Title (ZH)**: 基于成像模态的肺癌检测分类 

**Authors**: Sajim Ahmed, Muhammad Zain Chaudhary, Muhammad Zohaib Chaudhary, Mahmoud Abbass, Ahmed Sherif, Mohammad Mahbubur Rahman Khan Mamun  

**Link**: [PDF](https://arxiv.org/pdf/2509.16254)  

**Abstract**: Lung cancer continues to be the predominant cause of cancer-related mortality globally. This review analyzes various approaches, including advanced image processing methods, focusing on their efficacy in interpreting CT scans, chest radiographs, and biological markers. Notably, we identify critical gaps in the previous surveys, including the need for robust models that can generalize across diverse populations and imaging modalities. This comprehensive synthesis aims to serve as a foundational resource for researchers and clinicians, guiding future efforts toward more accurate and efficient lung cancer detection. Key findings reveal that 3D CNN architectures integrated with CT scans achieve the most superior performances, yet challenges such as high false positives, dataset variability, and computational complexity persist across modalities. 

**Abstract (ZH)**: 肺癌继续是全球癌症相关死亡的主要原因。本文分析了各种方法，包括先进的图像处理技术，重点关注这些方法在解读CT扫描、胸部X光和生物标志物方面的有效性。值得注意的是，我们指出之前的研究存在一些关键空白，包括需要能够跨多种人群和成像模态泛化的 robust 模型。本综述旨在为研究人员和临床医生提供一个基础资源，指导未来更加准确和高效的肺癌检测努力。关键发现表明，结合CT扫描的3D CNN架构表现最优，但高假阳性率、数据集差异性和计算复杂性等挑战依然存在。 

---
# A study on Deep Convolutional Neural Networks, transfer learning, and Mnet model for Cervical Cancer Detection 

**Title (ZH)**: 深度卷积神经网络、迁移学习及Mnet模型在宫颈癌检测中的研究 

**Authors**: Saifuddin Sagor, Md Taimur Ahad, Faruk Ahmed, Rokonozzaman Ayon, Sanzida Parvin  

**Link**: [PDF](https://arxiv.org/pdf/2509.16250)  

**Abstract**: Early and accurate detection through Pap smear analysis is critical to improving patient outcomes and reducing mortality of Cervical cancer. State-of-the-art (SOTA) Convolutional Neural Networks (CNNs) require substantial computational resources, extended training time, and large datasets. In this study, a lightweight CNN model, S-Net (Simple Net), is developed specifically for cervical cancer detection and classification using Pap smear images to address these limitations. Alongside S-Net, six SOTA CNNs were evaluated using transfer learning, including multi-path (DenseNet201, ResNet152), depth-based (Serasnet152), width-based multi-connection (Xception), depth-wise separable convolutions (MobileNetV2), and spatial exploitation-based (VGG19). All models, including S-Net, achieved comparable accuracy, with S-Net reaching 99.99%. However, S-Net significantly outperforms the SOTA CNNs in terms of computational efficiency and inference time, making it a more practical choice for real-time and resource-constrained applications. A major limitation in CNN-based medical diagnosis remains the lack of transparency in the decision-making process. To address this, Explainable AI (XAI) techniques, such as SHAP, LIME, and Grad-CAM, were employed to visualize and interpret the key image regions influencing model predictions. The novelty of this study lies in the development of a highly accurate yet computationally lightweight model (S-Net) caPable of rapid inference while maintaining interpretability through XAI integration. Furthermore, this work analyzes the behavior of SOTA CNNs, investigates the effects of negative transfer learning on Pap smear images, and examines pixel intensity patterns in correctly and incorrectly classified samples. 

**Abstract (ZH)**: 早期和准确的巴氏涂片分析对于改善宫颈癌患者的治疗结果和降低 mortality 至关重要。针对现有的卷积神经网络 (CNN) 需要大量计算资源、长时间训练和大体量数据集的局限性，在本研究中，开发了一种专门用于宫颈癌检测和分类的轻量级 CNN 模型 S-Net（Simple Net），以克服这些限制。除了 S-Net 外，还评估了六种最新的 CNN 模型，包括多路径模型（DenseNet201、ResNet152）、基于深度的模型（Serasnet152）、基于宽度的多连接模型（Xception）、深度可分离卷积模型（MobileNetV2）和基于空间利用的模型（VGG19），所有模型，包括 S-Net，都实现了可比的准确性，其中 S-Net 达到了 99.99%。然而，S-Net 在计算效率和推理时间方面显著优于现有的 CNN 模型，使其成为实时和资源受限应用中更实用的选择。在基于 CNN 的医疗诊断中，一个主要的局限性在于决策过程缺乏透明度。为了应对这一挑战，在本研究中采用了可解释人工智能（XAI）技术，如 SHAP、LIME 和 Grad-CAM，以可视化并解释影响模型预测的关键图像区域。本研究的创新之处在于开发了一种高度准确且计算高效的模型（S-Net），能够在保持可解释性的同时实现快速推理，同时通过 XAI 整合增强解释性。此外，本研究分析了现有的 CNN 模型的行为，探究了负迁移学习对巴氏涂片图像的影响，并研究了正确和错误分类样本中的像素强度模式。 

---
