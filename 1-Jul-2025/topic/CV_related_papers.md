# Data-Driven Predictive Planning and Control for Aerial 3D Inspection with Back-face Elimination 

**Title (ZH)**: 基于数据驱动的预测性规划与控制：消除背面的无人机三维检测 

**Authors**: Savvas Papaioannou, Panayiotis Kolios, Christos G. Panayiotou, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23781)  

**Abstract**: Automated inspection with Unmanned Aerial Systems (UASs) is a transformative capability set to revolutionize various application domains. However, this task is inherently complex, as it demands the seamless integration of perception, planning, and control which existing approaches often treat separately. Moreover, it requires accurate long-horizon planning to predict action sequences, in contrast to many current techniques, which tend to be myopic. To overcome these limitations, we propose a 3D inspection approach that unifies perception, planning, and control within a single data-driven predictive control framework. Unlike traditional methods that rely on known UAS dynamic models, our approach requires only input-output data, making it easily applicable to off-the-shelf black-box UASs. Our method incorporates back-face elimination, a visibility determination technique from 3D computer graphics, directly into the control loop, thereby enabling the online generation of accurate, long-horizon 3D inspection trajectories. 

**Abstract (ZH)**: 无人机系统（UASs）自动检查是一种变革性的能力，有望革新各种应用领域。然而，这一任务本质复杂，因为它要求实现感知、规划和控制的无缝集成，而现有方法往往将这些方面分开处理。此外，它需要精确的长期规划以预测行动序列，而与许多当前技术相比，这些技术往往过于短视。为克服这些局限性，我们提出了一种3D检查方法，将感知、规划和控制统一在一个数据驱动的预测控制框架中。与依赖已知UAS动力学模型的传统方法不同，我们的方法只需要输入输出数据，使其易于应用于现成的黑盒子UAS。该方法将3D计算机图形中的背面消隐技术直接纳入控制环中，从而实现在线生成准确的长_horizon 3D检查轨迹。 

---
# Validation of AI-Based 3D Human Pose Estimation in a Cyber-Physical Environment 

**Title (ZH)**: 基于AI的3D人体姿态估计在赛博物理环境中的有效性验证 

**Authors**: Lisa Marie Otto, Michael Kaiser, Daniel Seebacher, Steffen Müller  

**Link**: [PDF](https://arxiv.org/pdf/2506.23739)  

**Abstract**: Ensuring safe and realistic interactions between automated driving systems and vulnerable road users (VRUs) in urban environments requires advanced testing methodologies. This paper presents a test environment that combines a Vehiclein-the-Loop (ViL) test bench with a motion laboratory, demonstrating the feasibility of cyber-physical (CP) testing of vehicle-pedestrian and vehicle-cyclist interactions. Building upon previous work focused on pedestrian localization, we further validate a human pose estimation (HPE) approach through a comparative analysis of real-world (RW) and virtual representations of VRUs. The study examines the perception of full-body motion using a commercial monocular camera-based 3Dskeletal detection AI. The virtual scene is generated in Unreal Engine 5, where VRUs are animated in real time and projected onto a screen to stimulate the camera. The proposed stimulation technique ensures the correct perspective, enabling realistic vehicle perception. To assess the accuracy and consistency of HPE across RW and CP domains, we analyze the reliability of detections as well as variations in movement trajectories and joint estimation stability. The validation includes dynamic test scenarios where human avatars, both walking and cycling, are monitored under controlled conditions. Our results show a strong alignment in HPE between RW and CP test conditions for stable motion patterns, while notable inaccuracies persist under dynamic movements and occlusions, particularly for complex cyclist postures. These findings contribute to refining CP testing approaches for evaluating next-generation AI-based vehicle perception and to enhancing interaction models of automated vehicles and VRUs in CP environments. 

**Abstract (ZH)**: 确保自动驾驶系统与城市环境中脆弱道路用户之间安全和现实的互动需要先进的测试方法。本文提出了一种结合Vehicle-in-the-Loop (ViL) 测试平台和运动实验室的测试环境，展示了在车辆-行人和车辆-自行车互动中进行计算物理（CP）测试的可行性。在之前行人定位工作的基础上，我们进一步通过现实世界（RW）和虚拟表示的对比分析验证了人体姿态估计（HPE）方法。研究使用商用单目相机基于3D骨骼检测的AI来检测全肢体运动感知。虚拟场景在Unreal Engine 5中生成，其中VRUs实时动画并投影到屏幕上以刺激相机。提出的刺激技术确保了正确的视角，使车辆感知更加真实。为了评估HPE在RW和CP域中的准确性和一致性，我们分析了检测的可靠性以及运动轨迹和关节估计的稳定性。验证包括动态测试场景，其中在受控条件下监视行走和骑行的人类avatar。结果表明，在稳定运动模式下，RW和CP测试条件下HPE存在良好的一致性，但在动态运动和遮挡下，特别是对于复杂的骑车姿势，存在明显的不准确性。这些发现为完善计算物理环境中基于下一代AI的车辆感知评估方法以及增强自动驾驶车辆与VRUs的交互模型做出了贡献。 

---
# GS-NBV: a Geometry-based, Semantics-aware Viewpoint Planning Algorithm for Avocado Harvesting under Occlusions 

**Title (ZH)**: GS-NBV: 基于几何、语义的遮挡环境下采椒视角规划算法 

**Authors**: Xiao'ao Song, Konstantinos Karydis  

**Link**: [PDF](https://arxiv.org/pdf/2506.23369)  

**Abstract**: Efficient identification of picking points is critical for automated fruit harvesting. Avocados present unique challenges owing to their irregular shape, weight, and less-structured growing environments, which require specific viewpoints for successful harvesting. We propose a geometry-based, semantics-aware viewpoint-planning algorithm to address these challenges. The planning process involves three key steps: viewpoint sampling, evaluation, and execution. Starting from a partially occluded view, the system first detects the fruit, then leverages geometric information to constrain the viewpoint search space to a 1D circle, and uniformly samples four points to balance the efficiency and exploration. A new picking score metric is introduced to evaluate the viewpoint suitability and guide the camera to the next-best view. We validate our method through simulation against two state-of-the-art algorithms. Results show a 100% success rate in two case studies with significant occlusions, demonstrating the efficiency and robustness of our approach. Our code is available at this https URL 

**Abstract (ZH)**: 基于几何的语义aware视角规划算法在自动采收中的高效识别点选择对于鳄梨的高效采摘至关重要。我们提出了一种几何基于的、具有语义意识的视角规划算法来解决这些挑战。规划过程涉及三个关键步骤：视角采样、评估和执行。从部分遮挡视图开始，系统首先检测水果，然后利用几何信息将视角搜索空间约束为1D圆，并均匀采样四个点以平衡效率和探索性。引入了一个新的采摘评分指标来评估视角适宜性并引导相机到下一个最佳视图。通过与两个最先进的算法的仿真实验验证了我们的方法，结果显示在两个涉及显著遮挡的案例研究中100%的成功率，证明了我们方法的高效性和鲁棒性。我们的代码已在此处提供：this https URL 

---
# InfGen: Scenario Generation as Next Token Group Prediction 

**Title (ZH)**: InfGen：场景生成作为下一个token组预测 

**Authors**: Zhenghao Peng, Yuxin Liu, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.23316)  

**Abstract**: Realistic and interactive traffic simulation is essential for training and evaluating autonomous driving systems. However, most existing data-driven simulation methods rely on static initialization or log-replay data, limiting their ability to model dynamic, long-horizon scenarios with evolving agent populations. We propose InfGen, a scenario generation framework that outputs agent states and trajectories in an autoregressive manner. InfGen represents the entire scene as a sequence of tokens, including traffic light signals, agent states, and motion vectors, and uses a transformer model to simulate traffic over time. This design enables InfGen to continuously insert new agents into traffic, supporting infinite scene generation. Experiments demonstrate that InfGen produces realistic, diverse, and adaptive traffic behaviors. Furthermore, reinforcement learning policies trained in InfGen-generated scenarios achieve superior robustness and generalization, validating its utility as a high-fidelity simulation environment for autonomous driving. More information is available at this https URL. 

**Abstract (ZH)**: 现实且互动的交通仿真对于培训和评估自动驾驶系统至关重要。然而，大多数现有的数据驱动仿真方法依赖于静态初始化或日志回放数据，限制了它们模拟动态、长期演变的场景和不断变化的代理群体的能力。我们提出了一种名为InfGen的场景生成框架，以自回归方式输出代理状态和轨迹。InfGen将整个场景表示为一系列标记，包括交通灯信号、代理状态和运动矢量，并使用变换器模型随时间进行交通仿真。这一设计使InfGen能够持续插入新的代理进入交通，支持无限场景生成。实验表明，InfGen生成的交通行为具有现实性、多样性和适应性。此外，在InfGen生成的场景中训练的强化学习策略表现出更高的鲁棒性和泛化能力，验证了其作为自动驾驶高保真仿真环境的实用性。更多信息请访问此网址。 

---
# Event-based Stereo Visual-Inertial Odometry with Voxel Map 

**Title (ZH)**: 基于事件的立体视觉-惯性里程计与体素地图 

**Authors**: Zhaoxing Zhang, Xiaoxiang Wang, Chengliang Zhang, Yangyang Guo, Zikang Yuan, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23078)  

**Abstract**: The event camera, renowned for its high dynamic range and exceptional temporal resolution, is recognized as an important sensor for visual odometry. However, the inherent noise in event streams complicates the selection of high-quality map points, which critically determine the precision of state estimation. To address this challenge, we propose Voxel-ESVIO, an event-based stereo visual-inertial odometry system that utilizes voxel map management, which efficiently filter out high-quality 3D points. Specifically, our methodology utilizes voxel-based point selection and voxel-aware point management to collectively optimize the selection and updating of map points on a per-voxel basis. These synergistic strategies enable the efficient retrieval of noise-resilient map points with the highest observation likelihood in current frames, thereby ensureing the state estimation accuracy. Extensive evaluations on three public benchmarks demonstrate that our Voxel-ESVIO outperforms state-of-the-art methods in both accuracy and computational efficiency. 

**Abstract (ZH)**: 基于体素的事件级立体视觉惯性定位系统 Voxel-ESVIO 

---
# Pixels-to-Graph: Real-time Integration of Building Information Models and Scene Graphs for Semantic-Geometric Human-Robot Understanding 

**Title (ZH)**: 像素到图：建筑信息模型与场景图的实时集成及其在语义几何人類机器人理解中的应用 

**Authors**: Antonello Longo, Chanyoung Chung, Matteo Palieri, Sung-Kyun Kim, Ali Agha, Cataldo Guaragnella, Shehryar Khattak  

**Link**: [PDF](https://arxiv.org/pdf/2506.22593)  

**Abstract**: Autonomous robots are increasingly playing key roles as support platforms for human operators in high-risk, dangerous applications. To accomplish challenging tasks, an efficient human-robot cooperation and understanding is required. While typically robotic planning leverages 3D geometric information, human operators are accustomed to a high-level compact representation of the environment, like top-down 2D maps representing the Building Information Model (BIM). 3D scene graphs have emerged as a powerful tool to bridge the gap between human readable 2D BIM and the robot 3D maps. In this work, we introduce Pixels-to-Graph (Pix2G), a novel lightweight method to generate structured scene graphs from image pixels and LiDAR maps in real-time for the autonomous exploration of unknown environments on resource-constrained robot platforms. To satisfy onboard compute constraints, the framework is designed to perform all operation on CPU only. The method output are a de-noised 2D top-down environment map and a structure-segmented 3D pointcloud which are seamlessly connected using a multi-layer graph abstracting information from object-level up to the building-level. The proposed method is quantitatively and qualitatively evaluated during real-world experiments performed using the NASA JPL NeBula-Spot legged robot to autonomously explore and map cluttered garage and urban office like environments in real-time. 

**Abstract (ZH)**: 自主机器人在高风险危险应用中 increasingly 担当关键支援角色，作为人类操作者的合作伙伴。为了完成复杂的任务，高效的人机协作和理解是必需的。虽然通常机器人规划依赖于 3D 几何信息，但人类操作者习惯于使用高层紧凑环境表示，例如代表建筑信息模型 (BIM) 的顶部向下 2D 地图。3D 场景图已 emerge 作为一种强大的工具，可以弥合人类可读的 2D BIM 地图和机器人 3D 地图之间的差距。在本工作中，我们引入了从图像像素和 LiDAR 地图实时生成结构化场景图的新型轻量级方法 Pix2G，用于资源受限机器人平台上的自主未知环境探索。为满足机载计算约束，框架设计为仅在 CPU 上执行所有操作。该方法的输出是一个去噪的 2D 顶部向下环境地图和一个结构分割的 3D 点云，它们通过多层图结构无缝连接，该多层图从对象级别到建筑物级别抽象信息。所提出的方法在使用 NASA JPL NeBula-Spot 六足机器人在现实世界实验中实现实时自主探索和测绘杂乱车库及类似城市办公室环境进行定量和定性评估。 

---
# DriveBLIP2: Attention-Guided Explanation Generation for Complex Driving Scenarios 

**Title (ZH)**: DriveBLIP2：面向复杂驾驶场景的注意力引导解释生成 

**Authors**: Shihong Ling, Yue Wan, Xiaowei Jia, Na Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.22494)  

**Abstract**: This paper introduces a new framework, DriveBLIP2, built upon the BLIP2-OPT architecture, to generate accurate and contextually relevant explanations for emerging driving scenarios. While existing vision-language models perform well in general tasks, they encounter difficulties in understanding complex, multi-object environments, particularly in real-time applications such as autonomous driving, where the rapid identification of key objects is crucial. To address this limitation, an Attention Map Generator is proposed to highlight significant objects relevant to driving decisions within critical video frames. By directing the model's focus to these key regions, the generated attention map helps produce clear and relevant explanations, enabling drivers to better understand the vehicle's decision-making process in critical situations. Evaluations on the DRAMA dataset reveal significant improvements in explanation quality, as indicated by higher BLEU, ROUGE, CIDEr, and SPICE scores compared to baseline models. These findings underscore the potential of targeted attention mechanisms in vision-language models for enhancing explainability in real-time autonomous driving. 

**Abstract (ZH)**: 基于BLIP2-OPT架构的DriveBLIP2框架：生成新兴驾驶场景的相关解释 

---
# Navigating with Annealing Guidance Scale in Diffusion Space 

**Title (ZH)**: 使用退火指导尺度在扩散空间中的导航 

**Authors**: Shai Yehezkel, Omer Dahary, Andrey Voynov, Daniel Cohen-Or  

**Link**: [PDF](https://arxiv.org/pdf/2506.24108)  

**Abstract**: Denoising diffusion models excel at generating high-quality images conditioned on text prompts, yet their effectiveness heavily relies on careful guidance during the sampling process. Classifier-Free Guidance (CFG) provides a widely used mechanism for steering generation by setting the guidance scale, which balances image quality and prompt alignment. However, the choice of the guidance scale has a critical impact on the convergence toward a visually appealing and prompt-adherent image. In this work, we propose an annealing guidance scheduler which dynamically adjusts the guidance scale over time based on the conditional noisy signal. By learning a scheduling policy, our method addresses the temperamental behavior of CFG. Empirical results demonstrate that our guidance scheduler significantly enhances image quality and alignment with the text prompt, advancing the performance of text-to-image generation. Notably, our novel scheduler requires no additional activations or memory consumption, and can seamlessly replace the common classifier-free guidance, offering an improved trade-off between prompt alignment and quality. 

**Abstract (ZH)**: 去噪扩散模型在基于文本提示生成高质量图像方面表现出色，但其效果很大程度上依赖于采样过程中的精细指引。无分类器指引（CFG）通过设置指引比例提供了一种广泛应用的生成调控机制，能够在图像质量和提示对齐之间取得平衡。然而，指引比例的选择对图像向视觉吸引且符合提示的图像收敛有重要影响。在本文中，我们提出了一种退火指引调度器，该调度器基于条件噪声信号动态调整指引比例。通过学习调度策略，我们的方法解决了CFG的不稳定行为。实验结果显示，我们的指引调度器显著提高了图像质量和与文本提示的一致性，提升了文本到图像生成的性能。值得注意的是，我们的新型调度器无需额外的激活或内存消耗，并能够无缝替代常见的无分类器指引，提供了提示对齐和质量之间的优化权衡。 

---
# Imagine for Me: Creative Conceptual Blending of Real Images and Text via Blended Attention 

**Title (ZH)**: Imagine for Me: 基于融合注意力的现实图像与文本的创造性概念融合 

**Authors**: Wonwoong Cho, Yanxia Zhang, Yan-Ying Chen, David I. Inouye  

**Link**: [PDF](https://arxiv.org/pdf/2506.24085)  

**Abstract**: Blending visual and textual concepts into a new visual concept is a unique and powerful trait of human beings that can fuel creativity. However, in practice, cross-modal conceptual blending for humans is prone to cognitive biases, like design fixation, which leads to local minima in the design space. In this paper, we propose a T2I diffusion adapter "IT-Blender" that can automate the blending process to enhance human creativity. Prior works related to cross-modal conceptual blending are limited in encoding a real image without loss of details or in disentangling the image and text inputs. To address these gaps, IT-Blender leverages pretrained diffusion models (SD and FLUX) to blend the latent representations of a clean reference image with those of the noisy generated image. Combined with our novel blended attention, IT-Blender encodes the real reference image without loss of details and blends the visual concept with the object specified by the text in a disentangled way. Our experiment results show that IT-Blender outperforms the baselines by a large margin in blending visual and textual concepts, shedding light on the new application of image generative models to augment human creativity. 

**Abstract (ZH)**: 将视觉和文本概念融合成新的视觉概念是人类独有的强大特质，可以激发创造力。然而，在实践中，人类的跨模态概念融合容易受到认知偏差的影响，如设计 fixation，这会导致设计空间中的局部极值。在本文中，我们提出了一种T2I扩散适配器“IT-Blender”，可以自动化融合过程以增强人类的创造力。关于跨模态概念融合的先前工作要么在不丢失细节的情况下编码真实图像受限，要么难以分离图像和文本输入。为了解决这些问题，IT-Blender利用预训练的扩散模型（SD和FLUX），将干净参考图像的潜在表示与噪声生成图像的潜在表示融合。结合我们新颖的融合注意力机制，IT-Blender可以在不丢失细节的情况下编码真实参考图像，并以分离的方式融合视觉概念和由文本指定的对象。我们的实验结果表明，IT-Blender在融合视觉和文本概念方面显著优于基线方法，为图像生成模型在增强人类创造力方面的应用提供了新的视角。 

---
# ADReFT: Adaptive Decision Repair for Safe Autonomous Driving via Reinforcement Fine-Tuning 

**Title (ZH)**: ADReFT: 自适应决策修复以通过强化调优实现安全自主驾驶 

**Authors**: Mingfei Cheng, Xiaofei Xie, Renzhi Wang, Yuan Zhou, Ming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23960)  

**Abstract**: Autonomous Driving Systems (ADSs) continue to face safety-critical risks due to the inherent limitations in their design and performance capabilities. Online repair plays a crucial role in mitigating such limitations, ensuring the runtime safety and reliability of ADSs. Existing online repair solutions enforce ADS compliance by transforming unacceptable trajectories into acceptable ones based on predefined specifications, such as rule-based constraints or training datasets. However, these approaches often lack generalizability, adaptability and tend to be overly conservative, resulting in ineffective repairs that not only fail to mitigate safety risks sufficiently but also degrade the overall driving experience. To address this issue, we propose Adaptive Decision Repair (ADReFT), a novel and effective repair method that identifies safety-critical states through offline learning from failed tests and generates appropriate mitigation actions to improve ADS safety. Specifically, ADReFT incorporates a transformer-based model with two joint heads, State Monitor and Decision Adapter, designed to capture complex driving environment interactions to evaluate state safety severity and generate adaptive repair actions. Given the absence of oracles for state safety identification, we first pretrain ADReFT using supervised learning with coarse annotations, i.e., labeling states preceding violations as positive samples and others as negative samples. It establishes ADReFT's foundational capability to mitigate safety-critical violations, though it may result in somewhat conservative mitigation strategies. Therefore, we subsequently finetune ADReFT using reinforcement learning to improve its initial capability and generate more precise and contextually appropriate repair decisions. Our evaluation results illustrate that ADReFT achieves better repair performance. 

**Abstract (ZH)**: 自主驾驶系统（ADSs）的安全关键风险依然存在，原因在于其设计和性能限制。在线修复在缓解这些限制方面发挥着关键作用，确保ADSs的运行时安全性和可靠性。现有的在线修复解决方案通过基于预定义规范（如规则约束或训练数据集）将不可接受的轨迹转换为可接受的轨迹，来强制执行ADS合规性。然而，这些方法往往缺乏通用性、适应性，并且倾向于采取过于保守的策略，导致修复效果不佳，不仅未能充分缓解安全风险，还降低了整体驾驶体验。为了解决这一问题，我们提出了自适应决策修复（ADReFT），这是一种新颖且有效的修复方法，通过离线学习失败测试识别安全关键状态，并生成适当的缓解措施以提高ADS安全性。具体而言，ADReFT 结合了一个基于变换器的模型和两个联合头：状态监控器和决策适配器，旨在捕捉复杂的驾驶环境交互，评估状态安全严重程度并生成适应性修复行动。鉴于缺乏状态安全性识别的或有知识，我们首先使用监督学习和粗略标注对ADReFT进行预训练，即标记违反行为之前的状态为正样本，其他状态为负样本。这为ADReFT奠定了基础能力以应对安全关键违规行为，但可能导致较为保守的缓解策略。因此，我们随后通过强化学习对ADReFT进行微调，以改进其初始能力并生成更精确和上下文相关的修复决策。我们的评估结果表明，ADReFT在修复性能上取得了更好的效果。 

---
# GroundingDINO-US-SAM: Text-Prompted Multi-Organ Segmentation in Ultrasound with LoRA-Tuned Vision-Language Models 

**Title (ZH)**: 基于GroundingDINO-US-SAM：文本提示多器官超声分割与LoRA调优的视觉-语言模型 

**Authors**: Hamza Rasaee, Taha Koleilat, Hassan Rivaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.23903)  

**Abstract**: Accurate and generalizable object segmentation in ultrasound imaging remains a significant challenge due to anatomical variability, diverse imaging protocols, and limited annotated data. In this study, we propose a prompt-driven vision-language model (VLM) that integrates Grounding DINO with SAM2 to enable object segmentation across multiple ultrasound organs. A total of 18 public ultrasound datasets, encompassing the breast, thyroid, liver, prostate, kidney, and paraspinal muscle, were utilized. These datasets were divided into 15 for fine-tuning and validation of Grounding DINO using Low Rank Adaptation (LoRA) to the ultrasound domain, and 3 were held out entirely for testing to evaluate performance in unseen distributions. Comprehensive experiments demonstrate that our approach outperforms state-of-the-art segmentation methods, including UniverSeg, MedSAM, MedCLIP-SAM, BiomedParse, and SAMUS on most seen datasets while maintaining strong performance on unseen datasets without additional fine-tuning. These results underscore the promise of VLMs in scalable and robust ultrasound image analysis, reducing dependence on large, organ-specific annotated datasets. We will publish our code on this http URL after acceptance. 

**Abstract (ZH)**: 超声成像中基于提示的视觉-语言模型在准确且具备泛化能力的物体分割方面仍面临重大挑战，这归因于解剖变异、多样的成像协议以及有限的标注数据。在本研究中，我们提出了一种基于提示的视觉-语言模型（VLM），将Grounding DINO与SAM2结合，以实现跨多种超声器官的物体分割。共使用了18个公开的超声数据集，涵盖乳腺、甲状腺、肝脏、前列腺、肾脏及旁脊肌。这些数据集中的15个用于在超声领域利用低秩适应（LoRA）微调Grounding DINO，并进行验证；剩余的3个数据集用于测试，以评估其在未见过的数据分布中的性能。全面的实验表明，本方法在大多数已见数据集上优于包括UniverSeg、MedSAM、MedCLIP-SAM、BiomedParse和SAMUS在内的最新分割方法，在未见过的数据集上保持着强大的性能，无需额外的微调。这些结果突显了VLM在可扩展且稳健的超声图像分析方面的潜力，减少了对大规模、器官特异性标注数据集的依赖。论文接受后，我们将发布我们的代码。 

---
# Deep Learning-Based Semantic Segmentation for Real-Time Kidney Imaging and Measurements with Augmented Reality-Assisted Ultrasound 

**Title (ZH)**: 基于深度学习的实时肾脏成像与测量的语义分割及增强现实辅助超声技术 

**Authors**: Gijs Luijten, Roberto Maria Scardigno, Lisle Faray de Paiva, Peter Hoyer, Jens Kleesiek, Domenico Buongiorno, Vitoantonio Bevilacqua, Jan Egger  

**Link**: [PDF](https://arxiv.org/pdf/2506.23721)  

**Abstract**: Ultrasound (US) is widely accessible and radiation-free but has a steep learning curve due to its dynamic nature and non-standard imaging planes. Additionally, the constant need to shift focus between the US screen and the patient poses a challenge. To address these issues, we integrate deep learning (DL)-based semantic segmentation for real-time (RT) automated kidney volumetric measurements, which are essential for clinical assessment but are traditionally time-consuming and prone to fatigue. This automation allows clinicians to concentrate on image interpretation rather than manual measurements. Complementing DL, augmented reality (AR) enhances the usability of US by projecting the display directly into the clinician's field of view, improving ergonomics and reducing the cognitive load associated with screen-to-patient transitions. Two AR-DL-assisted US pipelines on HoloLens-2 are proposed: one streams directly via the application programming interface for a wireless setup, while the other supports any US device with video output for broader accessibility. We evaluate RT feasibility and accuracy using the Open Kidney Dataset and open-source segmentation models (nnU-Net, Segmenter, YOLO with MedSAM and LiteMedSAM). Our open-source GitHub pipeline includes model implementations, measurement algorithms, and a Wi-Fi-based streaming solution, enhancing US training and diagnostics, especially in point-of-care settings. 

**Abstract (ZH)**: 基于深度学习的实时自动化肾脏容积测量：结合增强现实的超声成像技术 

---
# A Clinically-Grounded Two-Stage Framework for Renal CT Report Generation 

**Title (ZH)**: 基于临床的两阶段框架用于肾部CT报告生成 

**Authors**: Renjie Liang, Zhengkang Fan, Jinqian Pan, Chenkun Sun, Russell Terry, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.23584)  

**Abstract**: Generating radiology reports from CT scans remains a complex task due to the nuanced nature of medical imaging and the variability in clinical documentation. In this study, we propose a two-stage framework for generating renal radiology reports from 2D CT slices. First, we extract structured abnormality features using a multi-task learning model trained to identify lesion attributes such as location, size, enhancement, and attenuation. These extracted features are subsequently combined with the corresponding CT image and fed into a fine-tuned vision-language model to generate natural language report sentences aligned with clinical findings. We conduct experiments on a curated dataset of renal CT studies with manually annotated sentence-slice-feature triplets and evaluate performance using both classification metrics and natural language generation metrics. Our results demonstrate that the proposed model outperforms random baselines across all abnormality types, and the generated reports capture key clinical content with reasonable textual accuracy. This exploratory work highlights the feasibility of modular, feature-informed report generation for renal imaging. Future efforts will focus on extending this pipeline to 3D CT volumes and further improving clinical fidelity in multimodal medical AI systems. 

**Abstract (ZH)**: 从CT扫描生成肾部放射学报告仍然是一个复杂的任务，由于医学影像的细微性质和临床记录的多样性。在本研究中，我们提出了一种两阶段框架，用于从2D CT切片生成肾部放射学报告。首先，我们使用一个多任务学习模型提取结构化的异常特征，该模型被训练以识别病灶属性，如位置、大小、增强和衰减。提取的特征随后与相应的CT图像结合，并输入微调的视觉-语言模型，生成与临床发现对齐的自然语言报告句子。我们在一个包含手动标注的句子-切片-特征三元组的肾部CT研究数据集上进行了实验，并使用分类指标和自然语言生成指标评估性能。研究结果表明，所提出的模型在所有异常类型上都优于随机基线，生成的报告准确捕捉了关键的临床内容。这项探索性研究强调了模块化、基于特征报告生成在肾部影像中的可行性。未来的工作将侧重于将该流水线扩展到3D CT体积，并进一步提高多模态医疗AI系统的临床一致性。 

---
# Uncertainty-aware Diffusion and Reinforcement Learning for Joint Plane Localization and Anomaly Diagnosis in 3D Ultrasound 

**Title (ZH)**: 不确定性感知的扩散与强化学习在三维超声联合平面定位与异常诊断中的应用 

**Authors**: Yuhao Huang, Yueyue Xu, Haoran Dou, Jiaxiao Deng, Xin Yang, Hongyu Zheng, Dong Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.23538)  

**Abstract**: Congenital uterine anomalies (CUAs) can lead to infertility, miscarriage, preterm birth, and an increased risk of pregnancy complications. Compared to traditional 2D ultrasound (US), 3D US can reconstruct the coronal plane, providing a clear visualization of the uterine morphology for assessing CUAs accurately. In this paper, we propose an intelligent system for simultaneous automated plane localization and CUA diagnosis. Our highlights are: 1) we develop a denoising diffusion model with local (plane) and global (volume/text) guidance, using an adaptive weighting strategy to optimize attention allocation to different conditions; 2) we introduce a reinforcement learning-based framework with unsupervised rewards to extract the key slice summary from redundant sequences, fully integrating information across multiple planes to reduce learning difficulty; 3) we provide text-driven uncertainty modeling for coarse prediction, and leverage it to adjust the classification probability for overall performance improvement. Extensive experiments on a large 3D uterine US dataset show the efficacy of our method, in terms of plane localization and CUA diagnosis. Code is available at this https URL. 

**Abstract (ZH)**: 先天性子宫畸形（CUAs）可能导致不孕、流产、早产以及妊娠并发症风险增加。与传统的2D超声（US）相比，3D US可以重建冠状面，提供清晰的子宫形态可视化，有助于准确评估CUAs。本文提出了一种同时实现平面定位和CUA诊断的智能系统。我们的亮点包括：1）我们开发了一种局部（平面）和全局（体块/文本）指导下的去噪扩散模型，并采用自适应加权策略优化对不同条件的关注分配；2）我们引入了一种基于强化学习的框架，采用无监督奖励来提取冗余序列中的关键切片摘要，并跨多个平面整合信息以降低学习难度；3）我们提供了由文本驱动的不确定性建模进行粗略预测，并利用该模型调整分类概率以提高整体性能。在大量3D子宫超声数据集上的广泛实验表明，我们的方法在平面定位和CUA诊断方面有效。代码可在此处访问：这个链接。 

---
# Artificial Intelligence-assisted Pixel-level Lung (APL) Scoring for Fast and Accurate Quantification in Ultra-short Echo-time MRI 

**Title (ZH)**: 人工智能辅助像素级肺部(APL)评分：用于超短回波时间MRI的快速精准定量分析 

**Authors**: Bowen Xin, Rohan Hickey, Tamara Blake, Jin Jin, Claire E Wainwright, Thomas Benkert, Alto Stemmer, Peter Sly, David Coman, Jason Dowling  

**Link**: [PDF](https://arxiv.org/pdf/2506.23506)  

**Abstract**: Lung magnetic resonance imaging (MRI) with ultrashort echo-time (UTE) represents a recent breakthrough in lung structure imaging, providing image resolution and quality comparable to computed tomography (CT). Due to the absence of ionising radiation, MRI is often preferred over CT in paediatric diseases such as cystic fibrosis (CF), one of the most common genetic disorders in Caucasians. To assess structural lung damage in CF imaging, CT scoring systems provide valuable quantitative insights for disease diagnosis and progression. However, few quantitative scoring systems are available in structural lung MRI (e.g., UTE-MRI). To provide fast and accurate quantification in lung MRI, we investigated the feasibility of novel Artificial intelligence-assisted Pixel-level Lung (APL) scoring for CF. APL scoring consists of 5 stages, including 1) image loading, 2) AI lung segmentation, 3) lung-bounded slice sampling, 4) pixel-level annotation, and 5) quantification and reporting. The results shows that our APL scoring took 8.2 minutes per subject, which was more than twice as fast as the previous grid-level scoring. Additionally, our pixel-level scoring was statistically more accurate (p=0.021), while strongly correlating with grid-level scoring (R=0.973, p=5.85e-9). This tool has great potential to streamline the workflow of UTE lung MRI in clinical settings, and be extended to other structural lung MRI sequences (e.g., BLADE MRI), and for other lung diseases (e.g., bronchopulmonary dysplasia). 

**Abstract (ZH)**: 超短回波时间磁共振成像（UTE-MRI）在肺结构成像中的 recent 突破及其在囊性纤维化中的应用：基于像素级人工智能辅助肺部评分系统的可行性研究 

---
# Qwen-GUI-3B: A Lightweight Vision-Language Model for Cross-Resolution GUI Grounding 

**Title (ZH)**: Qwen-GUI-3B: 一种轻量级的多分辨率GUI语义定位视觉-语言模型 

**Authors**: ZongHan Hsieh, Tzer-Jen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.23491)  

**Abstract**: This paper introduces Qwen-GUI-3B, a lightweight Vision-Language Model (VLM) specifically designed for Graphical User Interface grounding tasks, achieving performance competitive with significantly larger models. Unlike large-scale VLMs (>7B parameters) that are computationally intensive and impractical for consumer-grade hardware, Qwen-GUI-3B delivers strong grounding accuracy while being fully trainable on a single GPU (RTX 4090). The model incorporates several key innovations: (i) combine cross-platform, multi-resolution dataset of 24K examples from diverse sources including mobile, desktop, and web GUI screenshots to effectively address data scarcity in high-resolution desktop environments; (ii) a two-stage fine-tuning strategy, where initial cross-platform training establishes robust GUI understanding, followed by specialized fine-tuning on high-resolution data to significantly enhance model adaptability; and (iii) data curation and redundancy reduction strategies, demonstrating that randomly sampling a smaller subset with reduced redundancy achieves performance comparable to larger datasets, emphasizing data diversity over sheer volume. Empirical evaluation on standard GUI grounding benchmarks-including ScreenSpot, ScreenSpot-v2, and the challenging ScreenSpot-Pro, highlights Qwen-GUI-3B's exceptional accuracy, achieving 84.9% on ScreenSpot and 86.4% on ScreenSpot-v2, surpassing prior models under 4B parameters. Ablation studies validate the critical role of balanced sampling and two-stage fine-tuning in enhancing robustness, particularly in high-resolution desktop scenarios. The Qwen-GUI-3B is available at: this https URL 

**Abstract (ZH)**: Qwen-GUI-3B：一种用于图形用户界面接地任务的轻量级视觉-语言模型 

---
# UltraTwin: Towards Cardiac Anatomical Twin Generation from Multi-view 2D Ultrasound 

**Title (ZH)**: UltraTwin: 从多视角二维超声图生成心脏解剖孪生图像的研究 

**Authors**: Junxuan Yu, Yaofei Duan, Yuhao Huang, Yu Wang, Rongbo Ling, Weihao Luo, Ang Zhang, Jingxian Xu, Qiongying Ni, Yongsong Zhou, Binghan Li, Haoran Dou, Liping Liu, Yanfen Chu, Feng Geng, Zhe Sheng, Zhifeng Ding, Dingxin Zhang, Rui Huang, Yuhang Zhang, Xiaowei Xu, Tao Tan, Dong Ni, Zhongshan Gou, Xin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23490)  

**Abstract**: Echocardiography is routine for cardiac examination. However, 2D ultrasound (US) struggles with accurate metric calculation and direct observation of 3D cardiac structures. Moreover, 3D US is limited by low resolution, small field of view and scarce availability in practice. Constructing the cardiac anatomical twin from 2D images is promising to provide precise treatment planning and clinical quantification. However, it remains challenging due to the rare paired data, complex structures, and US noises. In this study, we introduce a novel generative framework UltraTwin, to obtain cardiac anatomical twin from sparse multi-view 2D US. Our contribution is three-fold. First, pioneered the construction of a real-world and high-quality dataset containing strictly paired multi-view 2D US and CT, and pseudo-paired data. Second, we propose a coarse-to-fine scheme to achieve hierarchical reconstruction optimization. Last, we introduce an implicit autoencoder for topology-aware constraints. Extensive experiments show that UltraTwin reconstructs high-quality anatomical twins versus strong competitors. We believe it advances anatomical twin modeling for potential applications in personalized cardiac care. 

**Abstract (ZH)**: 基于稀疏多视角二维超声的心脏解剖孪生生成框架UltraTwin 

---
# Time-variant Image Inpainting via Interactive Distribution Transition Estimation 

**Title (ZH)**: 基于交互分布变换估计的时变图像 inpainting 

**Authors**: Yun Xing, Qing Guo, Xiaoguang Li, Yihao Huang, Xiaofeng Cao, Di Lin, Ivor Tsang, Lei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.23461)  

**Abstract**: In this work, we focus on a novel and practical task, i.e., Time-vAriant iMage inPainting (TAMP). The aim of TAMP is to restore a damaged target image by leveraging the complementary information from a reference image, where both images captured the same scene but with a significant time gap in between, i.e., time-variant images. Different from conventional reference-guided image inpainting, the reference image under TAMP setup presents significant content distinction to the target image and potentially also suffers from damages. Such an application frequently happens in our daily lives to restore a damaged image by referring to another reference image, where there is no guarantee of the reference image's source and quality. In particular, our study finds that even state-of-the-art (SOTA) reference-guided image inpainting methods fail to achieve plausible results due to the chaotic image complementation. To address such an ill-posed problem, we propose a novel Interactive Distribution Transition Estimation (InDiTE) module which interactively complements the time-variant images with adaptive semantics thus facilitate the restoration of damaged regions. To further boost the performance, we propose our TAMP solution, namely Interactive Distribution Transition Estimation-driven Diffusion (InDiTE-Diff), which integrates InDiTE with SOTA diffusion model and conducts latent cross-reference during sampling. Moreover, considering the lack of benchmarks for TAMP task, we newly assembled a dataset, i.e., TAMP-Street, based on existing image and mask datasets. We conduct experiments on the TAMP-Street datasets under two different time-variant image inpainting settings, which show our method consistently outperform SOTA reference-guided image inpainting methods for solving TAMP. 

**Abstract (ZH)**: 基于交互分布转换估计的时间变异图像修复（TAMP） 

---
# PixelBoost: Leveraging Brownian Motion for Realistic-Image Super-Resolution 

**Title (ZH)**: PixelBoost: 利用布朗运动实现逼真超分辨率图像 

**Authors**: Aradhana Mishra, Bumshik Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.23254)  

**Abstract**: Diffusion-model-based image super-resolution techniques often face a trade-off between realistic image generation and computational efficiency. This issue is exacerbated when inference times by decreasing sampling steps, resulting in less realistic and hazy images. To overcome this challenge, we introduce a novel diffusion model named PixelBoost that underscores the significance of embracing the stochastic nature of Brownian motion in advancing image super-resolution, resulting in a high degree of realism, particularly focusing on texture and edge definitions. By integrating controlled stochasticity into the training regimen, our proposed model avoids convergence to local optima, effectively capturing and reproducing the inherent uncertainty of image textures and patterns. Our proposed model demonstrates superior objective results in terms of learned perceptual image patch similarity (LPIPS), lightness order error (LOE), peak signal-to-noise ratio(PSNR), structural similarity index measure (SSIM), as well as visual quality. To determine the edge enhancement, we evaluated the gradient magnitude and pixel value, and our proposed model exhibited a better edge reconstruction capability. Additionally, our model demonstrates adaptive learning capabilities by effectively adjusting to Brownian noise patterns and introduces a sigmoidal noise sequencing method that simplifies training, resulting in faster inference speeds. 

**Abstract (ZH)**: 基于扩散模型的图像超分辨率技术往往在图像生成的真实性和计算效率之间面临权衡。减少采样步骤导致的推断时间缩短会加剧这一问题，使得生成的图像不够真实且模糊。为克服这一挑战，我们提出了一种新型扩散模型PixelBoost，强调了在推进图像超分辨率过程中拥抱布朗运动的随机性的意义，从而实现高度的真实感，尤其在纹理和边缘定义方面。通过将可控的随机性整合到训练方案中，我们的模型避免了局部最优的收敛，有效地捕捉和再现了图像纹理和模式的固有不确定性。与学习感知图像块相似度（LPIPS）、亮度顺序误差（LOE）、峰值信噪比（PSNR）、结构相似性指数测量（SSIM）以及视觉质量相关的客观结果表明，我们的模型表现更优。为了评估边缘增强，我们评估了梯度幅度和像素值，我们的模型展示了更好的边缘重建能力。此外，我们的模型具有自适应学习能力，能够有效地适应布朗噪声模式，并引入了一种指数噪声序列方法，简化了训练过程，从而提高了推断速度。 

---
# Aggregating Local Saliency Maps for Semi-Global Explainable Image Classification 

**Title (ZH)**: 基于局部显著性图的半全局可解释图像分类 

**Authors**: James Hinns, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2506.23247)  

**Abstract**: Deep learning dominates image classification tasks, yet understanding how models arrive at predictions remains a challenge. Much research focuses on local explanations of individual predictions, such as saliency maps, which visualise the influence of specific pixels on a model's prediction. However, reviewing many of these explanations to identify recurring patterns is infeasible, while global methods often oversimplify and miss important local behaviours. To address this, we propose Segment Attribution Tables (SATs), a method for summarising local saliency explanations into (semi-)global insights. SATs take image segments (such as "eyes" in Chihuahuas) and leverage saliency maps to quantify their influence. These segments highlight concepts the model relies on across instances and reveal spurious correlations, such as reliance on backgrounds or watermarks, even when out-of-distribution test performance sees little change. SATs can explain any classifier for which a form of saliency map can be produced, using segmentation maps that provide named segments. SATs bridge the gap between oversimplified global summaries and overly detailed local explanations, offering a practical tool for analysing and debugging image classifiers. 

**Abstract (ZH)**: 深度学习在图像分类任务中占据主导地位，但理解模型如何做出预测仍然是一个挑战。现有的许多研究集中在个体预测的局部解释上，例如显著性图，这类图可以可视化特定像素对模型预测的影响。然而，审核大量此类解释以识别重复模式是不现实的，而全局方法往往简化过于草率，未能捕捉到重要的局部行为。为了解决这一问题，我们提出了一种片段归因表（SATs）方法，该方法将局部显著性解释总结为（半）全局洞察。SATs 使用图像片段（例如，吉娃娃的“眼睛”）并利用显著性图来量化其影响。这些片段突显了模型依赖的概念，尤其是在不同实例中的依赖关系，并揭示了错误的相关性，如依赖背景或水印，即使在离分布测试性能变化不大时也是如此。SATs 可以解释任何可以生成某种形式显著性图的分类器，使用提供命名片段的分割图。SATs 介于过度简化的全局总结和过于详细的局部解释之间，为分析和调试图像分类器提供了实用工具。 

---
# VolumetricSMPL: A Neural Volumetric Body Model for Efficient Interactions, Contacts, and Collisions 

**Title (ZH)**: 基于体素的SMPL：一种高效交互、接触和碰撞的神经体素人体模型 

**Authors**: Marko Mihajlovic, Siwei Zhang, Gen Li, Kaifeng Zhao, Lea Müller, Siyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.23236)  

**Abstract**: Parametric human body models play a crucial role in computer graphics and vision, enabling applications ranging from human motion analysis to understanding human-environment interactions. Traditionally, these models use surface meshes, which pose challenges in efficiently handling interactions with other geometric entities, such as objects and scenes, typically represented as meshes or point clouds. To address this limitation, recent research has explored volumetric neural implicit body models. However, existing works are either insufficiently robust for complex human articulations or impose high computational and memory costs, limiting their widespread use. To this end, we introduce VolumetricSMPL, a neural volumetric body model that leverages Neural Blend Weights (NBW) to generate compact, yet efficient MLP decoders. Unlike prior approaches that rely on large MLPs, NBW dynamically blends a small set of learned weight matrices using predicted shape- and pose-dependent coefficients, significantly improving computational efficiency while preserving expressiveness. VolumetricSMPL outperforms prior volumetric occupancy model COAP with 10x faster inference, 6x lower GPU memory usage, enhanced accuracy, and a Signed Distance Function (SDF) for efficient and differentiable contact modeling. We demonstrate VolumetricSMPL's strengths across four challenging tasks: (1) reconstructing human-object interactions from in-the-wild images, (2) recovering human meshes in 3D scenes from egocentric views, (3) scene-constrained motion synthesis, and (4) resolving self-intersections. Our results highlight its broad applicability and significant performance and efficiency gains. 

**Abstract (ZH)**: 参数化人体模型在计算机图形学和视觉领域发挥着重要作用，能够支持从人体动作分析到理解人环境交互等广泛的应用。传统上，这些模型采用表面网格表示，这在处理与其它几何实体（如物体和场景）的交互时带来了挑战，这些实体通常以网格或点云形式表示。为解决这一限制，最近的研究探索了体积神经隐式人体模型。然而，现有工作要么无法可靠地处理复杂的人体动作，要么计算和内存成本高，限制了它们的广泛应用。为了解决这一问题，我们提出了VolumetricSMPL，这是一种利用神经混合权重（NBW）生成紧凑而高效的MLP解码器的体积神经体模特。与依赖大规模MLP的方法不同，NBW通过预测形状和姿态依赖系数动态混合一小组学习权重矩阵，极大地提高了计算效率同时保持了表达能力。VolumetricSMPL在推断速度、GPU内存使用、准确性和支持高效可微接触建模的签名距离函数（SDF）方面均优于之前的体素占用模型COAP，性能和效率均有显著提升。我们展示了VolumetricSMPL在四个具有挑战性的任务中的优势：（1）从野外图像重建人体-物体交互，（2）从第一人称视角恢复3D场景中的人体网格，（3）场景约束动作合成，（4）解决自相交问题。我们的结果突显了其广泛应用的潜力及其显著的性能和效率改进。 

---
# Score-based Diffusion Model for Unpaired Virtual Histology Staining 

**Title (ZH)**: 基于评分的扩散模型用于无配对虚拟组织学染色 

**Authors**: Anran Liu, Xiaofei Wang, Jing Cai, Chao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.23184)  

**Abstract**: Hematoxylin and eosin (H&E) staining visualizes histology but lacks specificity for diagnostic markers. Immunohistochemistry (IHC) staining provides protein-targeted staining but is restricted by tissue availability and antibody specificity. Virtual staining, i.e., computationally translating the H&E image to its IHC counterpart while preserving the tissue structure, is promising for efficient IHC generation. Existing virtual staining methods still face key challenges: 1) effective decomposition of staining style and tissue structure, 2) controllable staining process adaptable to diverse tissue and proteins, and 3) rigorous structural consistency modelling to handle the non-pixel-aligned nature of paired H&E and IHC images. This study proposes a mutual-information (MI)-guided score-based diffusion model for unpaired virtual staining. Specifically, we design 1) a global MI-guided energy function that disentangles the tissue structure and staining characteristics across modalities, 2) a novel timestep-customized reverse diffusion process for precise control of the staining intensity and structural reconstruction, and 3) a local MI-driven contrastive learning strategy to ensure the cellular level structural consistency between H&E-IHC images. Extensive experiments demonstrate the our superiority over state-of-the-art approaches, highlighting its biomedical potential. Codes will be open-sourced upon acceptance. 

**Abstract (ZH)**: 基于互信息引导的评分扩散模型的无配对虚拟染色 

---
# Deep Learning for Optical Misalignment Diagnostics in Multi-Lens Imaging Systems 

**Title (ZH)**: 多镜成像系统中光学错位诊断的深度学习方法 

**Authors**: Tomer Slor, Dean Oren, Shira Baneth, Tom Coen, Haim Suchowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.23173)  

**Abstract**: In the rapidly evolving field of optical engineering, precise alignment of multi-lens imaging systems is critical yet challenging, as even minor misalignments can significantly degrade performance. Traditional alignment methods rely on specialized equipment and are time-consuming processes, highlighting the need for automated and scalable solutions. We present two complementary deep learning-based inverse-design methods for diagnosing misalignments in multi-element lens systems using only optical measurements. First, we use ray-traced spot diagrams to predict five-degree-of-freedom (5-DOF) errors in a 6-lens photographic prime, achieving a mean absolute error of 0.031mm in lateral translation and 0.011$^\circ$ in tilt. We also introduce a physics-based simulation pipeline that utilizes grayscale synthetic camera images, enabling a deep learning model to estimate 4-DOF, decenter and tilt errors in both two- and six-lens multi-lens systems. These results show the potential to reshape manufacturing and quality control in precision imaging. 

**Abstract (ZH)**: 基于深度学习的逆设计方法实现多透镜成像系统光学测量中的误差诊断 

---
# MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation 

**Title (ZH)**: MEMFOF: 高效多帧光学流估计的高分辨率训练 

**Authors**: Vladislav Bargatin, Egor Chistov, Alexander Yakovenko, Dmitriy Vatolin  

**Link**: [PDF](https://arxiv.org/pdf/2506.23151)  

**Abstract**: Recent advances in optical flow estimation have prioritized accuracy at the cost of growing GPU memory consumption, particularly for high-resolution (FullHD) inputs. We introduce MEMFOF, a memory-efficient multi-frame optical flow method that identifies a favorable trade-off between multi-frame estimation and GPU memory usage. Notably, MEMFOF requires only 2.09 GB of GPU memory at runtime for 1080p inputs, and 28.5 GB during training, which uniquely positions our method to be trained at native 1080p without the need for cropping or downsampling. We systematically revisit design choices from RAFT-like architectures, integrating reduced correlation volumes and high-resolution training protocols alongside multi-frame estimation, to achieve state-of-the-art performance across multiple benchmarks while substantially reducing memory overhead. Our method outperforms more resource-intensive alternatives in both accuracy and runtime efficiency, validating its robustness for flow estimation at high resolutions. At the time of submission, our method ranks first on the Spring benchmark with a 1-pixel (1px) outlier rate of 3.289, leads Sintel (clean) with an endpoint error (EPE) of 0.963, and achieves the best Fl-all error on KITTI-2015 at 2.94%. The code is available at this https URL. 

**Abstract (ZH)**: Recent Advances in Memory-Efficient Multi-Frame Optical Flow Estimation: MEMFOF 

---
# VisionScores -- A system-segmented image score dataset for deep learning tasks 

**Title (ZH)**: VisionScores -- 一个基于系统分割的图像评分数据集，用于深度学习任务 

**Authors**: Alejandro Romero Amezcua, Mariano José Juan Rivera Meraz  

**Link**: [PDF](https://arxiv.org/pdf/2506.23030)  

**Abstract**: VisionScores presents a novel proposal being the first system-segmented image score dataset, aiming to offer structure-rich, high information-density images for machine and deep learning tasks. Delimited to two-handed piano pieces, it was built to consider not only certain graphic similarity but also composition patterns, as this creative process is highly instrument-dependent. It provides two scenarios in relation to composer and composition type. The first, formed by 14k samples, considers works from different authors but the same composition type, specifically, Sonatinas. The latter, consisting of 10.8K samples, presents the opposite case, various composition types from the same author, being the one selected Franz Liszt. All of the 24.8k samples are formatted as grayscale jpg images of $128 \times 512$ pixels. VisionScores supplies the users not only the formatted samples but the systems' order and pieces' metadata. Moreover, unsegmented full-page scores and the pre-formatted images are included for further analysis. 

**Abstract (ZH)**: VisionScores呈现一种新颖的提案，作为首个系统分割图像评分数据集，旨在为机器和深度学习任务提供结构丰富、信息密度高的图像。该数据集限定于两手动钢琴曲，不仅考虑了特定的图形相似性，还考虑了构成模式，因为这一创意过程高度依赖于乐器。该数据集提供了与作曲家和作品类型相关的两种情况。第一种由14,000个样本组成，考虑了不同作者但相同类型的作品，即奏鸣曲。第二种由10,800个样本组成，展示了相反的情况，即同一作者的各种作品类型，被选中的是 Franz Liszt。所有的24,800个样本均格式化为128×512像素的灰度jpg图像。VisionScores不仅为用户提供格式化的样本和系统的顺序，还提供乐谱元数据。此外，还包含未分割的完整页面乐谱和预格式化的图像以便进一步分析。 

---
# Decoupled Seg Tokens Make Stronger Reasoning Video Segmenter and Grounder 

**Title (ZH)**: 解耦的分割标记使视频段析器和Grounder更强的推理能力 

**Authors**: Dang Jisheng, Wu Xudong, Wang Bimei, Lv Ning, Chen Jiayu, Jingwen Zhao, Yichu liu, Jizhao Liu, Juncheng Li, Teng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22880)  

**Abstract**: Existing video segmenter and grounder approaches, exemplified by Sa2VA, directly fuse features within segmentation models. This often results in an undesirable entanglement of dynamic visual information and static semantics, thereby degrading segmentation accuracy. To systematically mitigate this issue, we propose DeSa2VA, a decoupling-enhanced prompting scheme integrating text pre-training and a linear decoupling module to address the information processing limitations inherent in SAM-2. Specifically, first, we devise a pre-training paradigm that converts textual ground-truth labels into point-level prompts while generating corresponding text masks. These masks are refined through a hybrid loss function to strengthen the model's semantic grounding capabilities. Next, we employ linear projection to disentangle hidden states that generated by a large language model into distinct textual and visual feature subspaces. Finally, a dynamic mask fusion strategy synergistically combines these decoupled features through triple supervision from predicted text/visual masks and ground-truth annotations. Extensive experiments demonstrate state-of-the-art performance across diverse tasks, including image segmentation, image question answering, video segmentation, and video question answering. Our codes are available at this https URL. 

**Abstract (ZH)**: 现有的视频分割和锚定方法，如Sa2VA，直接在分割模型中融合特征，这往往会带来动态视觉信息和静态语义的不必要纠缠，从而降低分割准确性。为了系统地缓解这一问题，我们提出DeSa2VA，一种解耦增强的提示方案，结合文本预训练和线性解耦模块以解决SAM-2固有的信息处理限制。具体而言，首先，我们设计了一种预训练范式，将文本地面真值标签转换为点级提示，同时生成相应的文本掩码。这些掩码通过混合损失函数进行精炼，以增强模型的语义锚定能力。其次，我们采用线性投影将大型语言模型生成的隐藏状态解耦为独立的文本和视觉特征子空间。最后，通过预测文本/视觉掩码和地面真值注释的三重监督，动态掩码融合策略协同结合这些解耦特征。广泛实验显示，我们在包括图像分割、图像问答、视频分割和视频问答等多样任务中取得了最先进的性能。我们的代码可在以下链接获取：this https URL。 

---
# STR-Match: Matching SpatioTemporal Relevance Score for Training-Free Video Editing 

**Title (ZH)**: STR-Match: 匹配时空相关性得分用于无监督视频编辑 

**Authors**: Junsung Lee, Junoh Kang, Bohyung Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.22868)  

**Abstract**: Previous text-guided video editing methods often suffer from temporal inconsistency, motion distortion, and-most notably-limited domain transformation. We attribute these limitations to insufficient modeling of spatiotemporal pixel relevance during the editing process. To address this, we propose STR-Match, a training-free video editing algorithm that produces visually appealing and spatiotemporally coherent videos through latent optimization guided by our novel STR score. The score captures spatiotemporal pixel relevance across adjacent frames by leveraging 2D spatial attention and 1D temporal modules in text-to-video (T2V) diffusion models, without the overhead of computationally expensive 3D attention mechanisms. Integrated into a latent optimization framework with a latent mask, STR-Match generates temporally consistent and visually faithful videos, maintaining strong performance even under significant domain transformations while preserving key visual attributes of the source. Extensive experiments demonstrate that STR-Match consistently outperforms existing methods in both visual quality and spatiotemporal consistency. 

**Abstract (ZH)**: 基于文本指导的视频编辑方法往往存在时间不一致性、运动失真以及尤为明显的域变换限制。我们归因于这些限制是由于编辑过程中对时空像素相关性的建模不足。为解决这一问题，我们提出了一种无需训练的视频编辑算法STR-Match，该算法通过我们的新型STR分数引导的潜在优化生成视觉上吸引人且时空一致的视频。该分数通过利用文本到视频（T2V）扩散模型中的2D空间注意力和1D时间模块捕获相邻帧间的时空像素相关性，而无需昂贵的3D注意力机制开销。整合到一个具有潜在遮罩的潜在优化框架中，STR-Match生成时间一致且视觉忠实的视频，在显著的域变换下仍能保持良好的性能，同时保留源视频的关键视觉属性。大量实验表明，STR-Match在视觉质量和时空一致性方面始终优于现有方法。 

---
# Region-Aware CAM: High-Resolution Weakly-Supervised Defect Segmentation via Salient Region Perception 

**Title (ZH)**: 基于区域aware的CAM：通过显著区域感知的高分辨率弱监督缺陷分割 

**Authors**: Hang-Cheng Dong, Lu Zou, Bingguo Liu, Dong Ye, Guodong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22866)  

**Abstract**: Surface defect detection plays a critical role in industrial quality inspection. Recent advances in artificial intelligence have significantly enhanced the automation level of detection processes. However, conventional semantic segmentation and object detection models heavily rely on large-scale annotated datasets, which conflicts with the practical requirements of defect detection tasks. This paper proposes a novel weakly supervised semantic segmentation framework comprising two key components: a region-aware class activation map (CAM) and pseudo-label training. To address the limitations of existing CAM methods, especially low-resolution thermal maps, and insufficient detail preservation, we introduce filtering-guided backpropagation (FGBP), which refines target regions by filtering gradient magnitudes to identify areas with higher relevance to defects. Building upon this, we further develop a region-aware weighted module to enhance spatial precision. Finally, pseudo-label segmentation is implemented to refine the model's performance iteratively. Comprehensive experiments on industrial defect datasets demonstrate the superiority of our method. The proposed framework effectively bridges the gap between weakly supervised learning and high-precision defect segmentation, offering a practical solution for resource-constrained industrial scenarios. 

**Abstract (ZH)**: 表面缺陷检测在工业质量检查中扮演着关键角色。近年来，人工智能的进展显著提高了检测过程的自动化水平。然而，传统语义分割和物体检测模型 heavily 依赖大规模注释数据集，这与缺陷检测任务的实际需求相矛盾。本文提出了一种新颖的弱监督语义分割框架，包含两个关键组件：区域感知类激活图（CAM）和伪标签训练。为了解决现有 CAM 方法的局限性，特别是低分辨率热图和细节保留不足的问题，我们引入了过滤引导反向传播（FGBP），通过过滤梯度幅度来细化目标区域，识别与缺陷更有相关性的区域。在此基础上，我们进一步开发了区域感知加权模块以增强空间精度。最后，实现了伪标签分割以迭代提升模型性能。在工业缺陷数据集上的全面实验表明了我们方法的优势。所提出的框架有效地弥合了弱监督学习与高精度缺陷分割之间的差距，提供了资源受限工业场景的一种实用解决方案。 

---
# Lightning the Night with Generative Artificial Intelligence 

**Title (ZH)**: 用生成式人工智能点亮夜空 

**Authors**: Tingting Zhou, Feng Zhang, Haoyang Fu, Baoxiang Pan, Renhe Zhang, Feng Lu, Zhixin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.22511)  

**Abstract**: The visible light reflectance data from geostationary satellites is crucial for meteorological observations and plays an important role in weather monitoring and forecasting. However, due to the lack of visible light at night, it is impossible to conduct continuous all-day weather observations using visible light reflectance data. This study pioneers the use of generative diffusion models to address this limitation. Based on the multi-band thermal infrared brightness temperature data from the Advanced Geostationary Radiation Imager (AGRI) onboard the Fengyun-4B (FY4B) geostationary satellite, we developed a high-precision visible light reflectance retrieval model, called Reflectance Diffusion (RefDiff), which enables 0.47~\mu\mathrm{m}, 0.65~\mu\mathrm{m}, and 0.825~\mu\mathrm{m} bands visible light reflectance retrieval at night. Compared to the classical models, RefDiff not only significantly improves accuracy through ensemble averaging but also provides uncertainty estimation. Specifically, the SSIM index of RefDiff can reach 0.90, with particularly significant improvements in areas with complex cloud structures and thick clouds. The model's nighttime retrieval capability was validated using VIIRS nighttime product, demonstrating comparable performance to its daytime counterpart. In summary, this research has made substantial progress in the ability to retrieve visible light reflectance at night, with the potential to expand the application of nighttime visible light data. 

**Abstract (ZH)**: 基于生成扩散模型的风云四号B星多通道红外亮度温度数据夜间可见光反射率反演研究 

---
# How Can Multimodal Remote Sensing Datasets Transform Classification via SpatialNet-ViT? 

**Title (ZH)**: Multimodal 遥感数据集是如何通过 SpatialNet-ViT 转变分类的？ 

**Authors**: Gautam Siddharth Kashyap, Manaswi Kulahara, Nipun Joshi, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2506.22501)  

**Abstract**: Remote sensing datasets offer significant promise for tackling key classification tasks such as land-use categorization, object presence detection, and rural/urban classification. However, many existing studies tend to focus on narrow tasks or datasets, which limits their ability to generalize across various remote sensing classification challenges. To overcome this, we propose a novel model, SpatialNet-ViT, leveraging the power of Vision Transformers (ViTs) and Multi-Task Learning (MTL). This integrated approach combines spatial awareness with contextual understanding, improving both classification accuracy and scalability. Additionally, techniques like data augmentation, transfer learning, and multi-task learning are employed to enhance model robustness and its ability to generalize across diverse datasets 

**Abstract (ZH)**: 遥感数据集在解决土地利用分类、物体存在检测和农村/城市分类等关键分类任务方面具有重要的潜力。然而，许多现有研究倾向于专注于狭窄的任务或数据集，这限制了它们在各种遥感分类挑战中的泛化能力。为克服这一局限，我们提出了一种新颖的模型SpatialNet-ViT，结合了视觉变换器（ViTs）和多任务学习（MTL）的力量。这种集成方法结合了空间意识和上下文理解，提高了分类准确性并增强了模型的扩展性。此外，还采用了数据增强、迁移学习和多任务学习等技术以增强模型的稳健性及其在多种数据集上的泛化能力。 

---
# Scalable Dynamic Origin-Destination Demand Estimation Enhanced by High-Resolution Satellite Imagery Data 

**Title (ZH)**: 高分辨率卫星影像数据增强的大规模动态起源-目的地需求估算 

**Authors**: Jiachao Liu, Pablo Guarda, Koichiro Niinuma, Sean Qian  

**Link**: [PDF](https://arxiv.org/pdf/2506.22499)  

**Abstract**: This study presents a novel integrated framework for dynamic origin-destination demand estimation (DODE) in multi-class mesoscopic network models, leveraging high-resolution satellite imagery together with conventional traffic data from local sensors. Unlike sparse local detectors, satellite imagery offers consistent, city-wide road and traffic information of both parking and moving vehicles, overcoming data availability limitations. To extract information from imagery data, we design a computer vision pipeline for class-specific vehicle detection and map matching, generating link-level traffic density observations by vehicle class. Building upon this information, we formulate a computational graph-based DODE model that calibrates dynamic network states by jointly matching observed traffic counts and travel times from local sensors with density measurements derived from satellite imagery. To assess the accuracy and scalability of the proposed framework, we conduct a series of numerical experiments using both synthetic and real-world data. The results of out-of-sample tests demonstrate that supplementing traditional data with satellite-derived density significantly improves estimation performance, especially for links without local sensors. Real-world experiments also confirm the framework's capability to handle large-scale networks, supporting its potential for practical deployment in cities of varying sizes. Sensitivity analysis further evaluates the impact of data quality related to satellite imagery data. 

**Abstract (ZH)**: 本研究提出了一种适用于多类介观网络模型的新型动态起终点需求估计（DODE）集成框架，该框架结合了高分辨率卫星影像与本地传感器的传统交通数据。与稀疏的本地检测器不同，卫星影像提供了涵盖停车和移动车辆的全市范围的道路和交通信息，克服了数据可用性限制。为了从影像数据中提取信息，我们设计了一种计算机视觉流水线，用于特定类别的车辆检测和轨迹匹配，生成按类别划分的链路级交通密度观测值。基于这些信息，我们提出了一个基于计算图的DODE模型，通过同时匹配来自本地传感器的观测交通流量计数和旅行时间与卫星影像导出的密度测量值，共同校准动态网络状态。为了评估所提框架的准确性和可扩展性，我们使用合成和实际数据进行了一系列数值实验。离样本测试结果表明，与传统数据结合卫星衍生的密度显著提高了估计性能，尤其是在没有本地传感器的链路上。实际应用场景还证实了该框架处理大规模网络的能力，并支持其在不同规模城市中的潜在实用部署。进一步的敏感性分析评估了与卫星影像数据相关的数据质量的影响。 

---
# ViFusionTST: Deep Fusion of Time-Series Image Representations from Load Signals for Early Bed-Exit Prediction 

**Title (ZH)**: ViFusionTST: 基于负荷信号的时间序列图像表示深度融合的早期离床预测 

**Authors**: Hao Liu, Yu Hu, Rakiba Rayhana, Ling Bai, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.22498)  

**Abstract**: Bed-related falls remain a leading source of injury in hospitals and long-term-care facilities, yet many commercial alarms trigger only after a patient has already left the bed. We show that early bed-exit intent can be predicted using only four low-cost load cells mounted under the bed legs. The resulting load signals are first converted into a compact set of complementary images: an RGB line plot that preserves raw waveforms and three texture maps - recurrence plot, Markov transition field, and Gramian angular field - that expose higher-order dynamics. We introduce ViFusionTST, a dual-stream Swin Transformer that processes the line plot and texture maps in parallel and fuses them through cross-attention to learn data-driven modality weights.
To provide a realistic benchmark, we collected six months of continuous data from 95 beds in a long-term-care facility. On this real-world dataset ViFusionTST reaches an accuracy of 0.885 and an F1 score of 0.794, surpassing recent 1D and 2D time-series baselines across F1, recall, accuracy, and AUPRC. The results demonstrate that image-based fusion of load-sensor signals for time series classification is a practical and effective solution for real-time, privacy-preserving fall prevention. 

**Abstract (ZH)**: 基于床铺的跌倒仍然是医院和长期照料设施中主要的伤害来源，然而许多商用警报器仅在患者已经离床后才触发。我们展示了仅使用四个低成本的压力传感器安装在床腿下即可预测患者的早期离床意图。由此产生的载荷信号首先被转换成一个紧凑的互补图像集：一个保持原始波形的RGB线图和三个纹理图——循环图、马尔科夫转换场和Gramian角场，它们揭示了更高阶的动力学。我们提出了一种双流Swin Transformer——ViFusionTST，该模型并行处理线图和纹理图，并通过交叉注意机制融合它们以学习数据驱动的模态权重。

为了提供一个实际的基准，我们从一个长期照料设施中收集了95张床连续六个月的数据。在该实际数据集上，ViFusionTST 的准确率达到0.885，F1分数达到0.794，超过了近期的1D和2D时间序列基准模型在F1、召回率、准确率和AUPRC方面的性能。结果表明，利用载荷传感器信号的时间序列分类的图像融合方法是一种实用且有效的实时、隐私保护跌倒预防解决方案。 

---
# A Complex UNet Approach for Non-Invasive Fetal ECG Extraction Using Single-Channel Dry Textile Electrodes 

**Title (ZH)**: 使用单通道干纺织电极的非侵入胎儿ECG提取的复杂UNet方法 

**Authors**: Iulia Orvas, Andrei Radu, Alessandra Galli, Ana Neacsu, Elisabetta Peri  

**Link**: [PDF](https://arxiv.org/pdf/2506.22457)  

**Abstract**: Continuous, non-invasive pregnancy monitoring is crucial for minimising potential complications. The fetal electrocardiogram (fECG) represents a promising tool for assessing fetal health beyond clinical environments. Home-based monitoring necessitates the use of a minimal number of comfortable and durable electrodes, such as dry textile electrodes. However, this setup presents many challenges, including increased noise and motion artefacts, which complicate the accurate extraction of fECG signals. To overcome these challenges, we introduce a pioneering method for extracting fECG from single-channel recordings obtained using dry textile electrodes using AI techniques. We created a new dataset by simulating abdominal recordings, including noise closely resembling real-world characteristics of in-vivo recordings through dry textile electrodes, alongside mECG and fECG. To ensure the reliability of the extracted fECG, we propose an innovative pipeline based on a complex-valued denoising network, Complex UNet. Unlike previous approaches that focused solely on signal magnitude, our method processes both real and imaginary components of the spectrogram, addressing phase information and preventing incongruous predictions. We evaluated our novel pipeline against traditional, well-established approaches, on both simulated and real data in terms of fECG extraction and R-peak detection. The results showcase that our suggested method achieves new state-of-the-art results, enabling an accurate extraction of fECG morphology across all evaluated settings. This method is the first to effectively extract fECG signals from single-channel recordings using dry textile electrodes, making a significant advancement towards a fully non-invasive and self-administered fECG extraction solution. 

**Abstract (ZH)**: 连续非侵入性妊娠监测对于最小化潜在并发症至关重要。胎儿心电图（fECG）代表了评估胎儿健康状况的一种有前景的工具，超越了临床环境。基于家庭的监测要求使用少量舒适且耐用的电极，如干纺织电极。然而，这种设置带来了许多挑战，包括增加的噪声和运动伪影，这使准确提取fECG信号变得复杂。为克服这些挑战，我们提出了一种采用AI技术从使用干纺织电极获得的单通道记录中提取fECG的开创性方法。我们通过模拟腹部记录创建了一个新的数据集，其中包括通过干纺织电极记录的真实世界特点近似的噪声，以及mECG和fECG。为了确保提取的fECG可靠性，我们提出了一种基于复值去噪网络Complex UNet的创新管道。与以前仅关注信号幅度的方法不同，我们的方法处理频谱的实部和虚部，解决了相位信息并防止不一致的预测。我们在模拟和真实数据上对我们的新颖管道与传统且成熟的方案进行了评估，以评估fECG提取和R峰检测。结果显示，我们建议的方法达到了新的技术水平，能够在所有评估设置中实现准确的fECG形态提取。该方法是首次有效从单通道记录中提取使用干纺织电极的fECG信号，为实现完全非侵入性和自我管理的fECG提取解决方案带来了重大进展。 

---
# Vision Transformers for Multi-Variable Climate Downscaling: Emulating Regional Climate Models with a Shared Encoder and Multi-Decoder Architecture 

**Title (ZH)**: 基于视觉变换器的多变量气候下scaling：共享编码器和多解码器架构模拟区域气候模型 

**Authors**: Fabio Merizzi, Harilaos Loukos  

**Link**: [PDF](https://arxiv.org/pdf/2506.22447)  

**Abstract**: Global Climate Models (GCMs) are critical for simulating large-scale climate dynamics, but their coarse spatial resolution limits their applicability in regional studies. Regional Climate Models (RCMs) refine this through dynamic downscaling, albeit at considerable computational cost and with limited flexibility. While deep learning has emerged as an efficient data-driven alternative, most existing studies have focused on single-variable models that downscale one variable at a time. This approach can lead to limited contextual awareness, redundant computation, and lack of cross-variable interaction. Our study addresses these limitations by proposing a multi-task, multi-variable Vision Transformer (ViT) architecture with a shared encoder and variable-specific decoders (1EMD). The proposed architecture jointly predicts three key climate variables: surface temperature (tas), wind speed (sfcWind), and 500 hPa geopotential height (zg500), directly from GCM-resolution inputs, emulating RCM-scale downscaling over Europe. We show that our multi-variable approach achieves positive cross-variable knowledge transfer and consistently outperforms single-variable baselines trained under identical conditions, while also improving computational efficiency. These results demonstrate the effectiveness of multi-variable modeling for high-resolution climate downscaling. 

**Abstract (ZH)**: 全球气候模型（GCMs）对于模拟大规模气候动态至关重要，但其粗略的空间分辨率限制了其在区域研究中的应用。区域气候模型（RCMs）通过动力降尺度来弥补这一不足，尽管存在较高的计算成本和灵活性有限的问题。尽管深度学习已经作为高效的数据驱动替代方案出现，但大多数现有研究集中在单一变量模型上，一次降尺度一个变量。这种做法可能导致上下文意识有限、冗余计算和跨变量交互不足。我们的研究通过提出一个共享编码器和变量特定解码器的多任务、多变量视力变换器（ViT）架构（1EMD）来解决这些限制。所提出架构直接从GCM分辨率输入中联合预测三个关键气候变量：地表温度（tas）、风速（sfcWind）和500 hPa位势高度（zg500），模拟欧洲尺度的降尺度。我们展示，我们的多变量方法实现了积极的跨变量知识迁移，并且在相同条件下训练的一变量基线模型中始终表现出更高的性能，同时提高计算效率。这些结果证明了多变量建模在高分辨率气候降尺度中的有效性。 

---
