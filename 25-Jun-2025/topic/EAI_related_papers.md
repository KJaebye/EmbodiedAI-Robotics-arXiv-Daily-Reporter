# ManiGaussian++: General Robotic Bimanual Manipulation with Hierarchical Gaussian World Model 

**Title (ZH)**: ManiGaussian++: 通用的基于分层高斯世界模型的双臂 manipulation 技术 

**Authors**: Tengbo Yu, Guanxing Lu, Zaijia Yang, Haoyuan Deng, Season Si Chen, Jiwen Lu, Wenbo Ding, Guoqiang Hu, Yansong Tang, Ziwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19842)  

**Abstract**: Multi-task robotic bimanual manipulation is becoming increasingly popular as it enables sophisticated tasks that require diverse dual-arm collaboration patterns. Compared to unimanual manipulation, bimanual tasks pose challenges to understanding the multi-body spatiotemporal dynamics. An existing method ManiGaussian pioneers encoding the spatiotemporal dynamics into the visual representation via Gaussian world model for single-arm settings, which ignores the interaction of multiple embodiments for dual-arm systems with significant performance drop. In this paper, we propose ManiGaussian++, an extension of ManiGaussian framework that improves multi-task bimanual manipulation by digesting multi-body scene dynamics through a hierarchical Gaussian world model. To be specific, we first generate task-oriented Gaussian Splatting from intermediate visual features, which aims to differentiate acting and stabilizing arms for multi-body spatiotemporal dynamics modeling. We then build a hierarchical Gaussian world model with the leader-follower architecture, where the multi-body spatiotemporal dynamics is mined for intermediate visual representation via future scene prediction. The leader predicts Gaussian Splatting deformation caused by motions of the stabilizing arm, through which the follower generates the physical consequences resulted from the movement of the acting arm. As a result, our method significantly outperforms the current state-of-the-art bimanual manipulation techniques by an improvement of 20.2% in 10 simulated tasks, and achieves 60% success rate on average in 9 challenging real-world tasks. Our code is available at this https URL. 

**Abstract (ZH)**: 基于多任务双臂 manipulation 的多体时空动态建模方法进展：ManiGaussian++ 

---
# Look to Locate: Vision-Based Multisensory Navigation with 3-D Digital Maps for GNSS-Challenged Environments 

**Title (ZH)**: 面向定位的基于视觉的多感官导航：针对GNSS受限制环境的三维数字地图导航 

**Authors**: Ola Elmaghraby, Eslam Mounier, Paulo Ricardo Marques de Araujo, Aboelmagd Noureldin  

**Link**: [PDF](https://arxiv.org/pdf/2506.19827)  

**Abstract**: In Global Navigation Satellite System (GNSS)-denied environments such as indoor parking structures or dense urban canyons, achieving accurate and robust vehicle positioning remains a significant challenge. This paper proposes a cost-effective, vision-based multi-sensor navigation system that integrates monocular depth estimation, semantic filtering, and visual map registration (VMR) with 3-D digital maps. Extensive testing in real-world indoor and outdoor driving scenarios demonstrates the effectiveness of the proposed system, achieving sub-meter accuracy of 92% indoors and more than 80% outdoors, with consistent horizontal positioning and heading average root mean-square errors of approximately 0.98 m and 1.25 °, respectively. Compared to the baselines examined, the proposed solution significantly reduced drift and improved robustness under various conditions, achieving positioning accuracy improvements of approximately 88% on average. This work highlights the potential of cost-effective monocular vision systems combined with 3D maps for scalable, GNSS-independent navigation in land vehicles. 

**Abstract (ZH)**: 在全球导航卫星系统（GNSS）受限环境中，如室内停车场或密集的城市峡谷，实现准确可靠的车辆定位仍然是一项重大挑战。本文提出了一种经济有效的基于视觉的多传感器导航系统，该系统结合了单目深度估计、语义过滤和视觉地图注册（VMR）以及3D数字地图。在实际的室内外驾驶场景中的广泛测试显示，该系统具有良好的效果，室内实现了92%以上的亚米级准确度，室外超过80%，水平定位和航向的一致均方根误差分别为约0.98米和1.25°。与基准方法相比，在各种条件下，提出的解决方案显著减少了漂移并提高了鲁棒性，平均定位精度提高了约88%。这项工作突显了低成本单目视觉系统与3D地图结合在地面车辆中实现可扩展且GNSS独立导航的潜在能力。 

---
# CronusVLA: Transferring Latent Motion Across Time for Multi-Frame Prediction in Manipulation 

**Title (ZH)**: CronusVLA：跨时间转移潜在运动以进行多帧预测的操作控制 

**Authors**: Hao Li, Shuai Yang, Yilun Chen, Yang Tian, Xiaoda Yang, Xinyi Chen, Hanqing Wang, Tai Wang, Feng Zhao, Dahua Lin, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19816)  

**Abstract**: Recent vision-language-action (VLA) models built on pretrained vision-language models (VLMs) have demonstrated strong generalization across manipulation tasks. However, they remain constrained by a single-frame observation paradigm and cannot fully benefit from the motion information offered by aggregated multi-frame historical observations, as the large vision-language backbone introduces substantial computational cost and inference latency. We propose CronusVLA, a unified framework that extends single-frame VLA models to the multi-frame paradigm through an efficient post-training stage. CronusVLA comprises three key components: (1) single-frame pretraining on large-scale embodied datasets with autoregressive action tokens prediction, which establishes an embodied vision-language foundation; (2) multi-frame encoding, adapting the prediction of vision-language backbones from discrete action tokens to motion features during post-training, and aggregating motion features from historical frames into a feature chunking; (3) cross-frame decoding, which maps the feature chunking to accurate actions via a shared decoder with cross-attention. By reducing redundant token computation and caching past motion features, CronusVLA achieves efficient inference. As an application of motion features, we further propose an action adaptation mechanism based on feature-action retrieval to improve model performance during finetuning. CronusVLA achieves state-of-the-art performance on SimplerEnv with 70.9% success rate, and 12.7% improvement over OpenVLA on LIBERO. Real-world Franka experiments also show the strong performance and robustness. 

**Abstract (ZH)**: Recent Vision-Language-Action (VLA) Models Built on Pretrained Vision-Language Models (VLMs) Have Demonstrated Strong Generalization Across Manipulation Tasks. However, They Remain Constrained by a Single-Frame Observation Paradigm and Cannot Fully Benefit from the Motion Information Offered by Aggregated Multi-Frame Historical Observations Due to the Substantial Computational Cost and Inference Latency Introduced by the Large Vision-Language Backbone. We Propose CronusVLA, a Unified Framework That Extends Single-Frame VLA Models to the Multi-Frame Paradigm Through an Efficient Post-Training Stage. 

---
# The Starlink Robot: A Platform and Dataset for Mobile Satellite Communication 

**Title (ZH)**: Starlink机器人平台及移动卫星通信数据集 

**Authors**: Boyi Liu, Qianyi Zhang, Qiang Yang, Jianhao Jiao, Jagmohan Chauhan, Dimitrios Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2506.19781)  

**Abstract**: The integration of satellite communication into mobile devices represents a paradigm shift in connectivity, yet the performance characteristics under motion and environmental occlusion remain poorly understood. We present the Starlink Robot, the first mobile robotic platform equipped with Starlink satellite internet, comprehensive sensor suite including upward-facing camera, LiDAR, and IMU, designed to systematically study satellite communication performance during movement. Our multi-modal dataset captures synchronized communication metrics, motion dynamics, sky visibility, and 3D environmental context across diverse scenarios including steady-state motion, variable speeds, and different occlusion conditions. This platform and dataset enable researchers to develop motion-aware communication protocols, predict connectivity disruptions, and optimize satellite communication for emerging mobile applications from smartphones to autonomous vehicles. The project is available at this https URL. 

**Abstract (ZH)**: Satellite Communication Integration into Mobile Devices: Performance Study Using the Starlink Robot Platform and Dataset 

---
# Fake or Real, Can Robots Tell? Evaluating Embodied Vision-Language Models on Real and 3D-Printed Objects 

**Title (ZH)**: 真假辨别，机器人能行吗？基于实体物体的视觉-语言模型评估 

**Authors**: Federico Tavella, Kathryn Mearns, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19579)  

**Abstract**: Robotic scene understanding increasingly relies on vision-language models (VLMs) to generate natural language descriptions of the environment. In this work, we present a comparative study of captioning strategies for tabletop scenes captured by a robotic arm equipped with an RGB camera. The robot collects images of objects from multiple viewpoints, and we evaluate several models that generate scene descriptions. We compare the performance of various captioning models, like BLIP and VLMs. Our experiments examine the trade-offs between single-view and multi-view captioning, and difference between recognising real-world and 3D printed objects. We quantitatively evaluate object identification accuracy, completeness, and naturalness of the generated captions. Results show that VLMs can be used in robotic settings where common objects need to be recognised, but fail to generalise to novel representations. Our findings provide practical insights into deploying foundation models for embodied agents in real-world settings. 

**Abstract (ZH)**: 基于视觉-语言模型的机器人场景理解：使用RGB相机捕捉的桌上场景配图策略比较研究 

---
# T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with Vision-Language Models 

**Title (ZH)**: T-Rex：基于视觉-语言模型的任务自适应空间表示提取在机器人操作中的应用 

**Authors**: Yiteng Chen, Wenbo Li, Shiyi Wang, Huiping Zhuang, Qingyao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19498)  

**Abstract**: Building a general robotic manipulation system capable of performing a wide variety of tasks in real-world settings is a challenging task. Vision-Language Models (VLMs) have demonstrated remarkable potential in robotic manipulation tasks, primarily due to the extensive world knowledge they gain from large-scale datasets. In this process, Spatial Representations (such as points representing object positions or vectors representing object orientations) act as a bridge between VLMs and real-world scene, effectively grounding the reasoning abilities of VLMs and applying them to specific task scenarios. However, existing VLM-based robotic approaches often adopt a fixed spatial representation extraction scheme for various tasks, resulting in insufficient representational capability or excessive extraction time. In this work, we introduce T-Rex, a Task-Adaptive Framework for Spatial Representation Extraction, which dynamically selects the most appropriate spatial representation extraction scheme for each entity based on specific task requirements. Our key insight is that task complexity determines the types and granularity of spatial representations, and Stronger representational capabilities are typically associated with Higher overall system operation costs. Through comprehensive experiments in real-world robotic environments, we show that our approach delivers significant advantages in spatial understanding, efficiency, and stability without additional training. 

**Abstract (ZH)**: 构建一种能够执行多种任务的一般机器人操作系统是在实际环境中具有挑战性的任务。基于视觉-语言模型（VLMs）的机器人操作任务表现出显著潜力，主要归因于它们从大规模数据集中获得的广泛世界知识。在这个过程中，空间表示（如表示物体位置的点或表示物体方向的向量）充当了VLMs与实际场景之间的桥梁，有效grounded VLMs的推理能力并应用于特定任务场景中。然而，现有的基于VLM的机器人方法通常采用固定的空间表示提取方案处理多种任务，导致空间表示能力不足或提取时间过长。在本文中，我们提出了T-Rex，一种基于任务的空间表示提取自适应框架，根据特定任务需求动态选择最合适的空间表示提取方案。我们的核心洞察是，任务复杂性决定了空间表示的类型和粒度，更强的空间表示能力通常伴随着更高的系统操作成本。通过在真实世界机器人环境中的全面实验，我们展示了该方法在空间理解、效率和稳定性方面具有显著优势，而无需额外训练。 

---
# Zero-Shot Parameter Learning of Robot Dynamics Using Bayesian Statistics and Prior Knowledge 

**Title (ZH)**: 基于贝叶斯统计和先验知识的机器人动力学零样本参数学习 

**Authors**: Carsten Reiners, Minh Trinh, Lukas Gründel, Sven Tauchmann, David Bitterolf, Oliver Petrovic, Christian Brecher  

**Link**: [PDF](https://arxiv.org/pdf/2506.19350)  

**Abstract**: Inertial parameter identification of industrial robots is an established process, but standard methods using Least Squares or Machine Learning do not consider prior information about the robot and require extensive measurements. Inspired by Bayesian statistics, this paper presents an identification method with improved generalization that incorporates prior knowledge and is able to learn with only a few or without additional measurements (Zero-Shot Learning). Furthermore, our method is able to correctly learn not only the inertial but also the mechanical and base parameters of the MABI Max 100 robot while ensuring physical feasibility and specifying the confidence intervals of the results. We also provide different types of priors for serial robots with 6 degrees of freedom, where datasheets or CAD models are not available. 

**Abstract (ZH)**: 基于贝叶斯统计的工业机器人惯性参数识别方法：融合先验知识的零样本学习 

---
# Robotic Perception with a Large Tactile-Vision-Language Model for Physical Property Inference 

**Title (ZH)**: 基于大型触觉-视觉-语言模型的机器人感知与物理属性推理 

**Authors**: Zexiang Guo, Hengxiang Chen, Xinheng Mai, Qiusang Qiu, Gan Ma, Zhanat Kappassov, Qiang Li, Nutan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19303)  

**Abstract**: Inferring physical properties can significantly enhance robotic manipulation by enabling robots to handle objects safely and efficiently through adaptive grasping strategies. Previous approaches have typically relied on either tactile or visual data, limiting their ability to fully capture properties. We introduce a novel cross-modal perception framework that integrates visual observations with tactile representations within a multimodal vision-language model. Our physical reasoning framework, which employs a hierarchical feature alignment mechanism and a refined prompting strategy, enables our model to make property-specific predictions that strongly correlate with ground-truth measurements. Evaluated on 35 diverse objects, our approach outperforms existing baselines and demonstrates strong zero-shot generalization. Keywords: tactile perception, visual-tactile fusion, physical property inference, multimodal integration, robot perception 

**Abstract (ZH)**: 物理属性推理可以显著增强机器人操作，通过适应性抓取策略使机器人能够安全高效地处理物体。以往的方法通常依赖于触觉或视觉数据，限制了其全面捕获属性的能力。我们提出了一种新的跨模态感知框架，将视觉观察与触觉表征结合在多模态视觉-语言模型中。我们的物理推理框架采用分层特征对齐机制和精细的提示策略，使我们的模型能够做出与真实测量高度相关的属性特定预测。在35种不同物体的评估中，我们的方法优于现有基线，并显示出强大的零样本通用性。关键词：触觉感知、视觉-触觉融合、物理属性推理、多模态集成、机器人感知。 

---
# Ontology Neural Network and ORTSF: A Framework for Topological Reasoning and Delay-Robust Control 

**Title (ZH)**: 本体神经网络与ORTSF：拓扑推理与延迟鲁棒控制的框架 

**Authors**: Jaehong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.19277)  

**Abstract**: The advancement of autonomous robotic systems has led to impressive capabilities in perception, localization, mapping, and control. Yet, a fundamental gap remains: existing frameworks excel at geometric reasoning and dynamic stability but fall short in representing and preserving relational semantics, contextual reasoning, and cognitive transparency essential for collaboration in dynamic, human-centric environments. This paper introduces a unified architecture comprising the Ontology Neural Network (ONN) and the Ontological Real-Time Semantic Fabric (ORTSF) to address this gap. The ONN formalizes relational semantic reasoning as a dynamic topological process. By embedding Forman-Ricci curvature, persistent homology, and semantic tensor structures within a unified loss formulation, ONN ensures that relational integrity and topological coherence are preserved as scenes evolve over time. The ORTSF transforms reasoning traces into actionable control commands while compensating for system delays. It integrates predictive and delay-aware operators that ensure phase margin preservation and continuity of control signals, even under significant latency conditions. Empirical studies demonstrate the ONN + ORTSF framework's ability to unify semantic cognition and robust control, providing a mathematically principled and practically viable solution for cognitive robotics. 

**Abstract (ZH)**: 自主robotic系统的发展已经在感知、定位、制图和控制方面取得了令人印象深刻的成果。然而，仍存在一个根本性的差距：现有的框架在几何推理和动态稳定性方面表现出色，但在表示和保留对动态、以人为本的环境中协作至关重要的关系语义、上下文推理和认知透明性方面却有所不足。本文介绍了一种统一架构，包括本体神经网络（ONN）和本体实时语义 Fabric（ORTSF），以填补这一空白。ONN 将关系语义推理形式化为动态拓扑过程。通过在统一的损失公式中嵌入 Forman-Ricci 曲率、持久同调和语义张量结构，ONN 确保随场景随时间演变，关系完整性和拓扑一致性得以保持。ORTSF 将推理轨迹转化为可执行的控制命令，同时补偿系统时延。它集成了预测性和时延感知操作符，确保相位裕量的保留和控制信号的连续性，即使在显著时延条件下也是如此。实证研究表明，ONN + ORTSF 框架能够统一语义认知和鲁棒控制，提供了一个数学原理上和实践上可行的认知机器人解决方案。 

---
# AnchorDP3: 3D Affordance Guided Sparse Diffusion Policy for Robotic Manipulation 

**Title (ZH)**: AnchorDP3: 基于3D功能引导的稀疏扩散策略 für 机器人操作 

**Authors**: Ziyan Zhao, Ke Fan, He-Yang Xu, Ning Qiao, Bo Peng, Wenlong Gao, Dongjiang Li, Hui Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19269)  

**Abstract**: We present AnchorDP3, a diffusion policy framework for dual-arm robotic manipulation that achieves state-of-the-art performance in highly randomized environments. AnchorDP3 integrates three key innovations: (1) Simulator-Supervised Semantic Segmentation, using rendered ground truth to explicitly segment task-critical objects within the point cloud, which provides strong affordance priors; (2) Task-Conditioned Feature Encoders, lightweight modules processing augmented point clouds per task, enabling efficient multi-task learning through a shared diffusion-based action expert; (3) Affordance-Anchored Keypose Diffusion with Full State Supervision, replacing dense trajectory prediction with sparse, geometrically meaningful action anchors, i.e., keyposes such as pre-grasp pose, grasp pose directly anchored to affordances, drastically simplifying the prediction space; the action expert is forced to predict both robot joint angles and end-effector poses simultaneously, which exploits geometric consistency to accelerate convergence and boost accuracy. Trained on large-scale, procedurally generated simulation data, AnchorDP3 achieves a 98.7% average success rate in the RoboTwin benchmark across diverse tasks under extreme randomization of objects, clutter, table height, lighting, and backgrounds. This framework, when integrated with the RoboTwin real-to-sim pipeline, has the potential to enable fully autonomous generation of deployable visuomotor policies from only scene and instruction, totally eliminating human demonstrations from learning manipulation skills. 

**Abstract (ZH)**: AnchorDP3：一种在随机化环境下实现双臂机器人 manipulation 状态领先性能的扩散策略框架 

---
# Scaffolding Dexterous Manipulation with Vision-Language Models 

**Title (ZH)**: 基于视觉语言模型的灵巧操作辅助 

**Authors**: Vincent de Bakker, Joey Hejna, Tyler Ga Wei Lum, Onur Celik, Aleksandar Taranovic, Denis Blessing, Gerhard Neumann, Jeannette Bohg, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2506.19212)  

**Abstract**: Dexterous robotic hands are essential for performing complex manipulation tasks, yet remain difficult to train due to the challenges of demonstration collection and high-dimensional control. While reinforcement learning (RL) can alleviate the data bottleneck by generating experience in simulation, it typically relies on carefully designed, task-specific reward functions, which hinder scalability and generalization. Thus, contemporary works in dexterous manipulation have often bootstrapped from reference trajectories. These trajectories specify target hand poses that guide the exploration of RL policies and object poses that enable dense, task-agnostic rewards. However, sourcing suitable trajectories - particularly for dexterous hands - remains a significant challenge. Yet, the precise details in explicit reference trajectories are often unnecessary, as RL ultimately refines the motion. Our key insight is that modern vision-language models (VLMs) already encode the commonsense spatial and semantic knowledge needed to specify tasks and guide exploration effectively. Given a task description (e.g., "open the cabinet") and a visual scene, our method uses an off-the-shelf VLM to first identify task-relevant keypoints (e.g., handles, buttons) and then synthesize 3D trajectories for hand motion and object motion. Subsequently, we train a low-level residual RL policy in simulation to track these coarse trajectories or "scaffolds" with high fidelity. Across a number of simulated tasks involving articulated objects and semantic understanding, we demonstrate that our method is able to learn robust dexterous manipulation policies. Moreover, we showcase that our method transfers to real-world robotic hands without any human demonstrations or handcrafted rewards. 

**Abstract (ZH)**: 灵巧机器人手对手部执行复杂操作任务至关重要，但由于演示数据收集的挑战和高维控制的难度，训练起来仍然很困难。虽然强化学习（RL）可以通过模拟生成经验数据来缓解数据瓶颈，但它通常依赖于精心设计的任务特定奖励函数，这阻碍了其可扩展性和泛化能力。因此，当前的灵巧操作研究往往基于参考轨迹进行。这些轨迹指定目标手部姿态以引导RL策略的探索，并指定物体姿态以提供密集的任务无关奖励。然而，获取合适的轨迹，特别是对于灵巧的手部来说，仍然是一个巨大挑战。尽管如此，明确的参考轨迹中的细节对于RL并不是必要的，因为最终它会优化运动。我们的核心洞察是现代视觉-语言模型（VLMs）已经编码了指定任务和有效引导探索所需的常识空间和语义知识。给出一个任务描述（例如，“打开柜门”）和一个视觉场景，我们的方法首先使用一个现成的VLM识别与任务相关的关键点（例如，把手、按钮），然后合成手部运动和物体运动的三维轨迹。随后，我们在模拟中训练一个低级别的残差RL策略以高保真度跟踪这些粗略的轨迹或“支架”。在涉及 articulated 物体和语义理解的多个模拟任务中，我们证明了我们的方法能够学习稳健的灵巧操作策略。此外，我们展示了我们的方法可以无缝转移到现实世界的机器人手中，无需任何人类演示或人工设计的奖励。 

---
# Preserving Sense of Agency: User Preferences for Robot Autonomy and User Control across Household Tasks 

**Title (ZH)**: 保持主动感: 用户在家庭任务中对机器人自主性和用户控制的偏好 

**Authors**: Claire Yang, Heer Patel, Max Kleiman-Weiner, Maya Cakmak  

**Link**: [PDF](https://arxiv.org/pdf/2506.19202)  

**Abstract**: Roboticists often design with the assumption that assistive robots should be fully autonomous. However, it remains unclear whether users prefer highly autonomous robots, as prior work in assistive robotics suggests otherwise. High robot autonomy can reduce the user's sense of agency, which represents feeling in control of one's environment. How much control do users, in fact, want over the actions of robots used for in-home assistance? We investigate how robot autonomy levels affect users' sense of agency and the autonomy level they prefer in contexts with varying risks. Our study asked participants to rate their sense of agency as robot users across four distinct autonomy levels and ranked their robot preferences with respect to various household tasks. Our findings revealed that participants' sense of agency was primarily influenced by two factors: (1) whether the robot acts autonomously, and (2) whether a third party is involved in the robot's programming or operation. Notably, an end-user programmed robot highly preserved users' sense of agency, even though it acts autonomously. However, in high-risk settings, e.g., preparing a snack for a child with allergies, they preferred robots that prioritized their control significantly more. Additional contextual factors, such as trust in a third party operator, also shaped their preferences. 

**Abstract (ZH)**: 机器人研究人员常常假设辅助机器人应该完全自主。然而，目前尚不清楚用户是否更偏好高度自主的机器人，因为之前关于辅助机器人研究的成果表明并非如此。高度自主的机器人可以减少用户的自主感，即感觉自己能够控制环境。实际上，用户希望对用于家庭辅助的机器人动作拥有多少控制？我们研究了不同自主水平的机器人如何影响用户的自主感，以及在不同风险情境下用户偏好何种自主水平。研究要求参与者在四种不同的自主水平下评估其作为机器人用户时的自主感，并根据不同家务任务对他们的机器人偏好进行排序。研究发现，用户的自主感主要受两个因素影响：（1）机器人是否自主行动；（2）第三方是否参与机器人编程或操作。值得注意的是，即使机器人自主行事，由最终用户编程的机器人也能最大程度地保留用户的自主感。但在高风险情境下，例如为有食物过敏的孩子准备零食时，参与者更倾向于机器人更多地优先考虑他们的控制权。此外，第三方操作者的可信度等因素也影响了他们的偏好。 

---
# Situated Haptic Interaction: Exploring the Role of Context in Affective Perception of Robotic Touch 

**Title (ZH)**: 情境化触觉交互：探索触觉感知中环境作用的研究 

**Authors**: Qiaoqiao Ren, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2506.19179)  

**Abstract**: Affective interaction is not merely about recognizing emotions; it is an embodied, situated process shaped by context and co-created through interaction. In affective computing, the role of haptic feedback within dynamic emotional exchanges remains underexplored. This study investigates how situational emotional cues influence the perception and interpretation of haptic signals given by a robot. In a controlled experiment, 32 participants watched video scenarios in which a robot experienced either positive actions (such as being kissed), negative actions (such as being slapped) or neutral actions. After each video, the robot conveyed its emotional response through haptic communication, delivered via a wearable vibration sleeve worn by the participant. Participants rated the robot's emotional state-its valence (positive or negative) and arousal (intensity)-based on the video, the haptic feedback, and the combination of the two. The study reveals a dynamic interplay between visual context and touch. Participants' interpretation of haptic feedback was strongly shaped by the emotional context of the video, with visual context often overriding the perceived valence of the haptic signal. Negative haptic cues amplified the perceived valence of the interaction, while positive cues softened it. Furthermore, haptics override the participants' perception of arousal of the video. Together, these results offer insights into how situated haptic feedback can enrich affective human-robot interaction, pointing toward more nuanced and embodied approaches to emotional communication with machines. 

**Abstract (ZH)**: 情感交互不仅仅是情绪识别；它是一个受情境影响的身体化、情境性过程，并且是通过互动共同创造的。在情感计算中，动态情绪交流中触觉反馈的作用仍亟待探索。本研究探讨情境性情感提示如何影响参与者对机器人触觉信号的感知和解释。在受控实验中，32名参与者观看了机器人经历正面行为（如接吻）、负面行为（如被打）或中性行为（如握手）的视频场景。之后，机器人通过穿戴在参与者身上的振动袖口传达其情绪反应。参与者根据视频、触觉反馈以及两者结合对机器人的情绪状态（正向或负向的效价以及强度）进行了评估。研究揭示了视觉情境和触觉之间的动态互动。参与者对触觉反馈的解释强烈受到视频中情绪情境的影响，视觉情境通常会抵消参与者感知到的触觉信号的效价。负面的触觉提示会放大交互的效价感知，而正面的提示则会软化它。此外，触觉会超越参与者对视频情绪强度的感知。综上所述，这些结果提供了关于情境性触觉反馈如何丰富情感人机交互的见解，指出了对机器情感交流进行更细致入微和身体化的处理方法。 

---
# CUPID: Curating Data your Robot Loves with Influence Functions 

**Title (ZH)**: CUPID: 通过影响函数精选机器人喜爱的数据 

**Authors**: Christopher Agia, Rohan Sinha, Jingyun Yang, Rika Antonova, Marco Pavone, Haruki Nishimura, Masha Itkina, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2506.19121)  

**Abstract**: In robot imitation learning, policy performance is tightly coupled with the quality and composition of the demonstration data. Yet, developing a precise understanding of how individual demonstrations contribute to downstream outcomes - such as closed-loop task success or failure - remains a persistent challenge. We propose CUPID, a robot data curation method based on a novel influence function-theoretic formulation for imitation learning policies. Given a set of evaluation rollouts, CUPID estimates the influence of each training demonstration on the policy's expected return. This enables ranking and selection of demonstrations according to their impact on the policy's closed-loop performance. We use CUPID to curate data by 1) filtering out training demonstrations that harm policy performance and 2) subselecting newly collected trajectories that will most improve the policy. Extensive simulated and hardware experiments show that our approach consistently identifies which data drives test-time performance. For example, training with less than 33% of curated data can yield state-of-the-art diffusion policies on the simulated RoboMimic benchmark, with similar gains observed in hardware. Furthermore, hardware experiments show that our method can identify robust strategies under distribution shift, isolate spurious correlations, and even enhance the post-training of generalist robot policies. Additional materials are made available at: this https URL. 

**Abstract (ZH)**: 在机器人模仿学习中，政策性能与示范数据的质量和组成紧密相关。然而，如何精确理解单个示范对下游结果（如闭环任务的成功与否）的贡献仍然是一个持续的挑战。我们提出了一种名为CUPID的机器人数据整理方法，基于一种新颖的影响函数理论形式化方法，用于模仿学习政策。给定一组评估滚动集，CUPID估计每个训练示范对政策期望回报的影响。这使得可以根据示范对政策闭环性能的影响对其进行排名和选择。我们使用CUPID通过1) 过滤掉损害政策性能的训练示范，以及2) 选择新收集的最有助于改进政策的轨迹来整理数据。广泛的模拟和硬件实验表明，我们的方法一致地识别出哪些数据驱动测试时的性能。例如，使用不到33%的整理数据进行训练可以在模拟的RoboMimic基准上获得最先进的扩散策略，硬件实验中也观察到类似的收益。此外，硬件实验表明，我们的方法可以识别出在分布转移下的稳健策略，分离出无关联的耦合，并且甚至可以增强通用机器人政策的后训练。更多材料可在以下链接获取：this https URL。 

---
# Multimodal Anomaly Detection with a Mixture-of-Experts 

**Title (ZH)**: 多模态专家混合异常检测 

**Authors**: Christoph Willibald, Daniel Sliwowski, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.19077)  

**Abstract**: With a growing number of robots being deployed across diverse applications, robust multimodal anomaly detection becomes increasingly important. In robotic manipulation, failures typically arise from (1) robot-driven anomalies due to an insufficient task model or hardware limitations, and (2) environment-driven anomalies caused by dynamic environmental changes or external interferences. Conventional anomaly detection methods focus either on the first by low-level statistical modeling of proprioceptive signals or the second by deep learning-based visual environment observation, each with different computational and training data requirements. To effectively capture anomalies from both sources, we propose a mixture-of-experts framework that integrates the complementary detection mechanisms with a visual-language model for environment monitoring and a Gaussian-mixture regression-based detector for tracking deviations in interaction forces and robot motions. We introduce a confidence-based fusion mechanism that dynamically selects the most reliable detector for each situation. We evaluate our approach on both household and industrial tasks using two robotic systems, demonstrating a 60% reduction in detection delay while improving frame-wise anomaly detection performance compared to individual detectors. 

**Abstract (ZH)**: 随着部署在多样化应用中的机器人数量增加，鲁棒多模态异常检测变得越来越重要。在机器人操作中，故障通常源自于（1）由不充分的任务模型或硬件限制引起的机器人驱动异常，以及（2）由动态环境变化或外部干扰引起的环境驱动异常。传统的异常检测方法要么通过低级别统计建模 proprioceptive 信号来关注前者，要么通过基于深度学习的视觉环境观察来关注后者，每种方法都有不同的计算和训练数据要求。为了有效捕捉来自两个源的异常，我们提出了一种专家混合框架，该框架将视觉语言模型用于环境监控和基于高斯混合回归的检测器用于跟踪作用力和机器人运动中的偏差相结合，并引入了一种基于置信度的融合机制，以动态选择每种情况下最可靠的检测器。我们在两种机器人系统上对家庭和工业任务进行了评估，结果显示与单一检测器相比，检测延迟降低了60%，同时帧级异常检测性能得到提高。 

---
# Unified Vision-Language-Action Model 

**Title (ZH)**: 统一的视觉-语言-动作模型 

**Authors**: Yuqi Wang, Xinghang Li, Wenxuan Wang, Junbo Zhang, Yingyan Li, Yuntao Chen, Xinlong Wang, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19850)  

**Abstract**: Vision-language-action models (VLAs) have garnered significant attention for their potential in advancing robotic manipulation. However, previous approaches predominantly rely on the general comprehension capabilities of vision-language models (VLMs) to generate action signals, often overlooking the rich temporal and causal structure embedded in visual observations. In this paper, we present UniVLA, a unified and native multimodal VLA model that autoregressively models vision, language, and action signals as discrete token sequences. This formulation enables flexible multimodal tasks learning, particularly from large-scale video data. By incorporating world modeling during post-training, UniVLA captures causal dynamics from videos, facilitating effective transfer to downstream policy learning--especially for long-horizon tasks. Our approach sets new state-of-the-art results across several widely used simulation benchmarks, including CALVIN, LIBERO, and Simplenv-Bridge, significantly surpassing previous methods. For example, UniVLA achieves 95.5% average success rate on LIBERO benchmark, surpassing pi0-FAST's 85.5%. We further demonstrate its broad applicability on real-world ALOHA manipulation and autonomous driving. 

**Abstract (ZH)**: Vision-language-action模型（VLAs）在推动机器人操作方面展现出巨大潜力，但以往的方法主要依赖于视觉语言模型（VLMs）的一般理解能力生成行动信号，往往忽视了视觉观察中丰富的时序和因果结构。本文介绍了UniVLA，一种统一且原生的多模态VLA模型，以自回归方式建模视觉、语言和行动信号的离散标记序列。该建模方法使得从大规模视频数据中学习灵活的多模态任务成为可能。通过在后训练期间引入世界建模，UniVLA从视频中捕捉因果动态，促进下游策略学习的有效转移，尤其是在长时序任务方面。我们的方法在包括CALVIN、LIBERO和Simplenv-Bridge等多个广泛使用的模拟基准测试中设置了新的最佳结果，显著超越了以往的方法。例如，UniVLA在LIBERO基准测试中的平均成功率达到95.5%，超过pi0-FAST的85.5%。我们进一步展示了其在真实世界ALOHA操作和自动驾驶中的广泛适用性。 

---
# Position: Intelligent Science Laboratory Requires the Integration of Cognitive and Embodied AI 

**Title (ZH)**: 位置：智能科学实验室需要认知智能与 embodied AI 的融合 

**Authors**: Sha Zhang, Suorong Yang, Tong Xie, Xiangyuan Xue, Zixuan Hu, Rui Li, Wenxi Qu, Zhenfei Yin, Tianfan Fu, Di Hu, Andres M Bran, Nian Ran, Bram Hoex, Wangmeng Zuo, Philippe Schwaller, Wanli Ouyang, Lei Bai, Yanyong Zhang, Lingyu Duan, Shixiang Tang, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.19613)  

**Abstract**: Scientific discovery has long been constrained by human limitations in expertise, physical capability, and sleep cycles. The recent rise of AI scientists and automated laboratories has accelerated both the cognitive and operational aspects of research. However, key limitations persist: AI systems are often confined to virtual environments, while automated laboratories lack the flexibility and autonomy to adaptively test new hypotheses in the physical world. Recent advances in embodied AI, such as generalist robot foundation models, diffusion-based action policies, fine-grained manipulation learning, and sim-to-real transfer, highlight the promise of integrating cognitive and embodied intelligence. This convergence opens the door to closed-loop systems that support iterative, autonomous experimentation and the possibility of serendipitous discovery. In this position paper, we propose the paradigm of Intelligent Science Laboratories (ISLs): a multi-layered, closed-loop framework that deeply integrates cognitive and embodied intelligence. ISLs unify foundation models for scientific reasoning, agent-based workflow orchestration, and embodied agents for robust physical experimentation. We argue that such systems are essential for overcoming the current limitations of scientific discovery and for realizing the full transformative potential of AI-driven science. 

**Abstract (ZH)**: 智能科学实验室：认知与体态智能深度融合的多层闭环框架 

---
# Bayesian Evolutionary Swarm Architecture: A Formal Epistemic System Grounded in Truth-Based Competition 

**Title (ZH)**: 基于真相竞争的贝叶斯进化蜂群架构：一种形式化的知识系统 

**Authors**: Craig Steven Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.19191)  

**Abstract**: We introduce a mathematically rigorous framework for an artificial intelligence system composed of probabilistic agents evolving through structured competition and belief revision. The architecture, grounded in Bayesian inference, measure theory, and population dynamics, defines agent fitness as a function of alignment with a fixed external oracle representing ground truth. Agents compete in a discrete-time environment, adjusting posterior beliefs through observed outcomes, with higher-rated agents reproducing and lower-rated agents undergoing extinction. Ratings are updated via pairwise truth-aligned utility comparisons, and belief updates preserve measurable consistency and stochastic convergence. We introduce hash-based cryptographic identity commitments to ensure traceability, alongside causal inference operators using do-calculus. Formal theorems on convergence, robustness, and evolutionary stability are provided. The system establishes truth as an evolutionary attractor, demonstrating that verifiable knowledge arises from adversarial epistemic pressure within a computable, self-regulating swarm. 

**Abstract (ZH)**: 我们提出了一种严格数学框架，用于由进化竞争和信念修正的概率代理组成的类人工智能系统。该架构基于贝叶斯推理、测度理论和种群动力学，定义代理适应度为与固定外部先验或代表真相的先验的一致性函数。代理在离散时间环境中竞争，通过观察结果调整后验信念，评级较高者复制，评级较低者灭绝。评级通过成对的与真相对齐的效用比较进行更新，信念更新保持可测的一致性和随机收敛。我们引入基于哈希的密码身份承诺以确保可追溯性，并使用do-因果推理运算符。提供了关于收敛性、鲁棒性和进化稳定性的形式定理。该系统将真理确立为进化吸引子，展示了在可计算的自我调节群中，可验证知识源于敌对的辩证压力。 

---
# Signal Use and Emergent Cooperation 

**Title (ZH)**: 信号使用与 Emergent 合作 

**Authors**: Michael Williams  

**Link**: [PDF](https://arxiv.org/pdf/2506.18920)  

**Abstract**: In this work, we investigate how autonomous agents, organized into tribes, learn to use communication signals to coordinate their activities and enhance their collective efficiency. Using the NEC-DAC (Neurally Encoded Culture - Distributed Autonomous Communicators) system, where each agent is equipped with its own neural network for decision-making, we demonstrate how these agents develop a shared behavioral system -- akin to a culture -- through learning and signalling. Our research focuses on the self-organization of culture within these tribes of agents and how varying communication strategies impact their fitness and cooperation. By analyzing different social structures, such as authority hierarchies, we show that the culture of cooperation significantly influences the tribe's performance. Furthermore, we explore how signals not only facilitate the emergence of culture but also enable its transmission across generations of agents. Additionally, we examine the benefits of coordinating behavior and signaling within individual agents' neural networks. 

**Abstract (ZH)**: 本研究探讨了组织成部落的自主代理如何通过通信信号学习协调其活动以提高集体效率。借助NEC-DAC（神经编码文化-分布式自主通信者）系统，其中每个代理拥有用于决策的神经网络，我们展示了这些代理如何通过学习和信号传递发展出一种类似文化的共享行为系统。我们的研究集中于这些代理部落内部文化的自我组织以及不同的通信策略如何影响它们的适应性和合作。通过对权威等级结构等不同社会结构的分析，我们表明合作文化显著影响了部落的表现。另外，我们探讨了通信信号不仅如何促进文化的兴起，还能使其在代理的各代之间进行传递。此外，我们还考察了在个体代理神经网络内部协调行为和信号传递带来的益处。 

---
# Persona Features Control Emergent Misalignment 

**Title (ZH)**: Personality Features Control Emergent Misalignment 

**Authors**: Miles Wang, Tom Dupré la Tour, Olivia Watkins, Alex Makelov, Ryan A. Chi, Samuel Miserendino, Johannes Heidecke, Tejal Patwardhan, Dan Mossing  

**Link**: [PDF](https://arxiv.org/pdf/2506.19823)  

**Abstract**: Understanding how language models generalize behaviors from their training to a broader deployment distribution is an important problem in AI safety. Betley et al. discovered that fine-tuning GPT-4o on intentionally insecure code causes "emergent misalignment," where models give stereotypically malicious responses to unrelated prompts. We extend this work, demonstrating emergent misalignment across diverse conditions, including reinforcement learning on reasoning models, fine-tuning on various synthetic datasets, and in models without safety training. To investigate the mechanisms behind this generalized misalignment, we apply a "model diffing" approach using sparse autoencoders to compare internal model representations before and after fine-tuning. This approach reveals several "misaligned persona" features in activation space, including a toxic persona feature which most strongly controls emergent misalignment and can be used to predict whether a model will exhibit such behavior. Additionally, we investigate mitigation strategies, discovering that fine-tuning an emergently misaligned model on just a few hundred benign samples efficiently restores alignment. 

**Abstract (ZH)**: 理解语言模型将训练中的行为泛化到更广泛的部署分布中的机制是AI安全中的一个重要问题。贝特利等人发现，对故意不安全的代码进行GPT-4o微调会导致“ emergent misalignment”，即模型对无关提示给出 stereotypically 恶意的回答。我们在此基础上进行了拓展研究，展示了在多种条件下出现的泛化 misalignment，包括对推理模型进行强化学习、对各种合成数据集进行微调以及在未接受安全训练的模型中。为了探究这一泛化 misalignment 的机制，我们使用稀疏自编码器应用“模型差异分析”方法比较微调前后模型的内部表示。这种方法揭示了激活空间中几种“ misaligned persona”的特征，包括一个毒性 persona 特征，它是 emergent misalignment 最强的控制因子，并可用于预测模型是否会表现出此类行为。此外，我们还研究了缓解策略，发现仅对少量良性样本进行微调可以高效地恢复模型的对齐。 

---
# A Survey of Multi-sensor Fusion Perception for Embodied AI: Background, Methods, Challenges and Prospects 

**Title (ZH)**: 多传感器融合感知综述：背景、方法、挑战与展望 

**Authors**: Shulan Ruan, Rongwei Wang, Xuchen Shen, Huijie Liu, Baihui Xiao, Jun Shi, Kun Zhang, Zhenya Huang, Yu Liu, Enhong Chen, You He  

**Link**: [PDF](https://arxiv.org/pdf/2506.19769)  

**Abstract**: Multi-sensor fusion perception (MSFP) is a key technology for embodied AI, which can serve a variety of downstream tasks (e.g., 3D object detection and semantic segmentation) and application scenarios (e.g., autonomous driving and swarm robotics). Recently, impressive achievements on AI-based MSFP methods have been reviewed in relevant surveys. However, we observe that the existing surveys have some limitations after a rigorous and detailed investigation. For one thing, most surveys are oriented to a single task or research field, such as 3D object detection or autonomous driving. Therefore, researchers in other related tasks often find it difficult to benefit directly. For another, most surveys only introduce MSFP from a single perspective of multi-modal fusion, while lacking consideration of the diversity of MSFP methods, such as multi-view fusion and time-series fusion. To this end, in this paper, we hope to organize MSFP research from a task-agnostic perspective, where methods are reported from various technical views. Specifically, we first introduce the background of MSFP. Next, we review multi-modal and multi-agent fusion methods. A step further, time-series fusion methods are analyzed. In the era of LLM, we also investigate multimodal LLM fusion methods. Finally, we discuss open challenges and future directions for MSFP. We hope this survey can help researchers understand the important progress in MSFP and provide possible insights for future research. 

**Abstract (ZH)**: 多传感器融合感知（MSFP）是具身AI的关键技术，可服务于多种下游任务（例如3D物体检测和语义分割）和应用场景（例如自动驾驶和 swarm 机器人）。 

---
# Surgery-R1: Advancing Surgical-VQLA with Reasoning Multimodal Large Language Model via Reinforcement Learning 

**Title (ZH)**: 手术-R1：通过强化学习驱动的推理多模态大型语言模型提升手术-VQLA 

**Authors**: Pengfei Hao, Shuaibo Li, Hongqiu Wang, Zhizhuo Kou, Junhang Zhang, Guang Yang, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19469)  

**Abstract**: In recent years, significant progress has been made in the field of surgical scene understanding, particularly in the task of Visual Question Localized-Answering in robotic surgery (Surgical-VQLA). However, existing Surgical-VQLA models lack deep reasoning capabilities and interpretability in surgical scenes, which limits their reliability and potential for development in clinical applications. To address this issue, inspired by the development of Reasoning Multimodal Large Language Models (MLLMs), we first build the Surgery-R1-54k dataset, including paired data for Visual-QA, Grounding-QA, and Chain-of-Thought (CoT). Then, we propose the first Reasoning MLLM for Surgical-VQLA (Surgery-R1). In our Surgery-R1, we design a two-stage fine-tuning mechanism to enable the basic MLLM with complex reasoning abilities by utilizing supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT). Furthermore, for an efficient and high-quality rule-based reward system in our RFT, we design a Multimodal Coherence reward mechanism to mitigate positional illusions that may arise in surgical scenarios. Experiment results demonstrate that Surgery-R1 outperforms other existing state-of-the-art (SOTA) models in the Surgical-VQLA task and widely-used MLLMs, while also validating its reasoning capabilities and the effectiveness of our approach. The code and dataset will be organized in this https URL. 

**Abstract (ZH)**: 近年来，手术场景理解领域取得了显著进展，特别是在机器人手术中的视觉问题定位回答任务（Surgical-VQLA）方面。然而，现有的Surgical-VQLA模型在手术场景中的深度推理能力和可解释性方面存在不足，这限制了其在临床应用中的可靠性和发展潜力。为了应对这一挑战，受Reasoning Multimodal Large Language Models (MLLMs) 发展的启发，我们首先构建了Surgery-R1-54k数据集，其中包括视觉-问答、语义-问答和推理链（CoT）配对数据。随后，我们提出了首个用于Surgical-VQLA的Reasoning MLLM（Surgery-R1）。在Surgery-R1中，我们设计了一种两阶段微调机制，通过监督微调（SFT）和强化微调（RFT）使基本的MLLM具备复杂的推理能力。此外，为了在RFT中构建一个高效且高质量的基于规则的奖励系统，我们设计了多模态一致性奖励机制以减轻手术场景中可能出现的位置错觉。实验结果表明，Surgery-R1在Surgical-VQLA任务和广泛使用的MLLMs中均优于其他现有先进（SOTA）模型，同时验证了其推理能力和方法的有效性。代码和数据集将在以下链接组织：[请提供链接]。 

---
# Mem4Nav: Boosting Vision-and-Language Navigation in Urban Environments with a Hierarchical Spatial-Cognition Long-Short Memory System 

**Title (ZH)**: Mem4Nav: 一种基于分层空间认知长短期记忆系统的城市环境ビジョン-and-语言导航增强方法 

**Authors**: Lixuan He, Haoyu Dong, Zhenxing Chen, Yangcheng Yu, Jie Feng, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19433)  

**Abstract**: Vision-and-Language Navigation (VLN) in large-scale urban environments requires embodied agents to ground linguistic instructions in complex scenes and recall relevant experiences over extended time horizons. Prior modular pipelines offer interpretability but lack unified memory, while end-to-end (M)LLM agents excel at fusing vision and language yet remain constrained by fixed context windows and implicit spatial reasoning. We introduce \textbf{Mem4Nav}, a hierarchical spatial-cognition long-short memory system that can augment any VLN backbone. Mem4Nav fuses a sparse octree for fine-grained voxel indexing with a semantic topology graph for high-level landmark connectivity, storing both in trainable memory tokens embedded via a reversible Transformer. Long-term memory (LTM) compresses and retains historical observations at both octree and graph nodes, while short-term memory (STM) caches recent multimodal entries in relative coordinates for real-time obstacle avoidance and local planning. At each step, STM retrieval sharply prunes dynamic context, and, when deeper history is needed, LTM tokens are decoded losslessly to reconstruct past embeddings. Evaluated on Touchdown and Map2Seq across three backbones (modular, state-of-the-art VLN with prompt-based LLM, and state-of-the-art VLN with strided-attention MLLM), Mem4Nav yields 7-13 pp gains in Task Completion, sufficient SPD reduction, and >10 pp nDTW improvement. Ablations confirm the indispensability of both the hierarchical map and dual memory modules. Our codes are open-sourced via this https URL. 

**Abstract (ZH)**: 大规模城市环境中基于视语导航（VLN）需要实体代理在复杂场景中 grounding 语言指令，并在长时间范围内回忆相关经验。先前的模块化管道提供了可解释性但缺乏统一记忆，而端到端 (M)LLM 代理在融合视觉和语言方面表现出色，但仍受固定上下文窗口和隐式空间推理的限制。我们引入了 \textbf{Mem4Nav}，这是一种分层空间认知长短期记忆系统，可以增强任何 VLN 主干。Mem4Nav 将稀疏八叉树与语义拓扑图融合，分别用于精细的体素索引和高层地标连接，将两者嵌入到通过可逆Transformer训练的记忆令牌中。长期记忆 (LTM) 压缩并保留八叉树和图节点的历史观察结果，而短期记忆 (STM) 缓存最近的多模态条目以相对坐标进行实时障碍物回避和局部规划。在每一步中，STM 检索锐化动态上下文，并在需要更深的历史记录时，LTM 令牌无损解码以重构过去嵌入。在 Touchdown 和 Map2Seq 上，Mem4Nav 在三种主干（模块化、基于提示的大规模视觉语言导航（VLN）和基于跨步注意力的大规模视觉语言模型（LLM）大规模视觉语言导航（VLN））上实现了 7-13 个百分点的任务完成度提高、足够的时间精度降低和超过 10 个百分点的 nDTW 改进。消融实验证实了分层地图和双记忆模块的不可或缺性。我们的代码可通过以下链接开源：this https URL。 

---
# EmoStage: A Framework for Accurate Empathetic Response Generation via Perspective-Taking and Phase Recognition 

**Title (ZH)**: EmoStage：一种基于换位思考和阶段识别的准确共情响应生成框架 

**Authors**: Zhiyang Qi, Keiko Takamizo, Mariko Ukiyo, Michimasa Inaba  

**Link**: [PDF](https://arxiv.org/pdf/2506.19279)  

**Abstract**: The rising demand for mental health care has fueled interest in AI-driven counseling systems. While large language models (LLMs) offer significant potential, current approaches face challenges, including limited understanding of clients' psychological states and counseling stages, reliance on high-quality training data, and privacy concerns associated with commercial deployment. To address these issues, we propose EmoStage, a framework that enhances empathetic response generation by leveraging the inference capabilities of open-source LLMs without additional training data. Our framework introduces perspective-taking to infer clients' psychological states and support needs, enabling the generation of emotionally resonant responses. In addition, phase recognition is incorporated to ensure alignment with the counseling process and to prevent contextually inappropriate or inopportune responses. Experiments conducted in both Japanese and Chinese counseling settings demonstrate that EmoStage improves the quality of responses generated by base models and performs competitively with data-driven methods. 

**Abstract (ZH)**: AI驱动的咨询系统兴起促进了对心理卫生护理需求的增加。虽然大规模语言模型（LLMs）具有巨大潜力，但当前的方法面临挑战，包括对客户心理状态和咨询阶段的理解有限、依赖高质量的训练数据以及商业部署相关的隐私顾虑。为了解决这些问题，我们提出了EmoStage框架，该框架通过利用开源LLMs的推理能力来增强同理心响应生成，无需额外的训练数据。该框架引入了换位思考来推断客户的心理状态和支持需求，从而实现情感共鸣的响应生成。此外，咨询阶段识别的引入确保了与咨询过程的对齐，并防止了上下文不适宜或不合时宜的响应。在日本和中文咨询环境中进行的实验表明，EmoStage提高了基础模型生成的响应质量，并且在数据驱动的方法中表现竞争。 

---
# Reinforcement Learning-Based Dynamic Grouping for Tubular Structure Tracking 

**Title (ZH)**: 基于强化学习的动态分组在管状结构追踪中应用 

**Authors**: Chong Di, Shuwang Zhou, Da Chen, Jean-Marie Mirebeau, Minglei Shu, Laurent D. Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18930)  

**Abstract**: The computation of minimal paths for the applications in tracking tubular structures such as blood vessels and roads is challenged by complex morphologies and environmental variations. Existing approaches can be roughly categorized into two research lines: the point-wise based models and the segment-wise based models. Although segment-wise approaches have obtained promising results in many scenarios, they often suffer from computational inefficiency and heavily rely on a prescribed prior to fit the target elongated shapes. We propose a novel framework that casts segment-wise tracking as a Markov Decision Process (MDP), enabling a reinforcement learning approach. Our method leverages Q-Learning to dynamically explore a graph of segments, computing edge weights on-demand and adaptively expanding the search space. This strategy avoids the high cost of a pre-computed graph and proves robust to incomplete initial information. Experimental reuslts on typical tubular structure datasets demonstrate that our method significantly outperforms state-of-the-art point-wise and segment-wise approaches. The proposed method effectively handles complex topologies and maintains global path coherence without depending on extensive prior structural knowledge. 

**Abstract (ZH)**: 针对跟踪如血管和道路等管状结构的最小路径计算，复杂的形态和环境变化构成了挑战。现有的方法大致可以分为两类：基于点的方法和基于片段的方法。虽然基于片段的方法在许多场景中取得了令人 promising 的结果，但它们往往受到计算效率低下的困扰，并且严重依赖于预设的先验知识来拟合目标延伸形状。我们提出了一种新型框架，将基于片段的跟踪问题视为马尔可夫决策过程（MDP），从而启用强化学习方法。我们的方法利用 Q-学习动态探索片段图，按需计算边权重并自适应扩展搜索空间。这种策略避开了预计算图的高成本，并且能够适应不完整初始信息。在典型管状结构数据集上的实验结果表明，我们的方法显著优于现有最佳的基于点和基于片段的方法。所提出的方法能够有效处理复杂的拓扑结构，并保持全局路径的连贯性，无需依赖广泛的先验结构知识。 

---
