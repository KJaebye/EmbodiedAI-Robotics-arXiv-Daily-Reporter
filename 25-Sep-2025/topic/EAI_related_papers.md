# VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation 

**Title (ZH)**: 视觉模拟：通过运动跟踪与生成实现视觉类人智能操作 

**Authors**: Shaofeng Yin, Yanjie Ze, Hong-Xing Yu, C. Karen Liu, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20322)  

**Abstract**: Humanoid loco-manipulation in unstructured environments demands tight integration of egocentric perception and whole-body control. However, existing approaches either depend on external motion capture systems or fail to generalize across diverse tasks. We introduce VisualMimic, a visual sim-to-real framework that unifies egocentric vision with hierarchical whole-body control for humanoid robots. VisualMimic combines a task-agnostic low-level keypoint tracker -- trained from human motion data via a teacher-student scheme -- with a task-specific high-level policy that generates keypoint commands from visual and proprioceptive input. To ensure stable training, we inject noise into the low-level policy and clip high-level actions using human motion statistics. VisualMimic enables zero-shot transfer of visuomotor policies trained in simulation to real humanoid robots, accomplishing a wide range of loco-manipulation tasks such as box lifting, pushing, football dribbling, and kicking. Beyond controlled laboratory settings, our policies also generalize robustly to outdoor environments. Videos are available at: this https URL . 

**Abstract (ZH)**: 人类态机器人在未结构化环境中的动操作需求緊密整合第一人称感知与全身控制。VisualMimic：一种统一第一人称视觉与分层全身控制的视觉仿真实现框架 

---
# mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies 

**Title (ZH)**: 思维导图：深度特征图中的空间记忆用于3D动作策略 

**Authors**: Remo Steiner, Alexander Millane, David Tingdahl, Clemens Volk, Vikram Ramasamy, Xinjie Yao, Peter Du, Soha Pouya, Shiwei Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.20297)  

**Abstract**: End-to-end learning of robot control policies, structured as neural networks, has emerged as a promising approach to robotic manipulation. To complete many common tasks, relevant objects are required to pass in and out of a robot's field of view. In these settings, spatial memory - the ability to remember the spatial composition of the scene - is an important competency. However, building such mechanisms into robot learning systems remains an open research problem. We introduce mindmap (Spatial Memory in Deep Feature Maps for 3D Action Policies), a 3D diffusion policy that generates robot trajectories based on a semantic 3D reconstruction of the environment. We show in simulation experiments that our approach is effective at solving tasks where state-of-the-art approaches without memory mechanisms struggle. We release our reconstruction system, training code, and evaluation tasks to spur research in this direction. 

**Abstract (ZH)**: 基于深度特征图的三维空间记忆在无人机控制策略中的应用：End-to-end学习无人机控制策略的神经网络结构已 emerge 作为一种有前景的方法来实现机器人的操作。在这种设置下，空间记忆——即记住场景的空间组成的能力——是一项重要技能。然而，将此类机制集成到机器人学习系统中仍是一个开放的研究问题。我们引入了一种基于语义三维重建的三维扩散策略 mindmap，该策略根据环境的语义三维重建生成无人机轨迹。我们在模拟实验中展示了我们的方法在记忆机制缺失的先进方法难以解决的任务中展现出有效性。我们发布了我们的重建系统、训练代码和评估任务，以推动这一方向的研究。 

---
# Parse-Augment-Distill: Learning Generalizable Bimanual Visuomotor Policies from Single Human Video 

**Title (ZH)**: Parse-Augment-Distill: 从单个人类视频学习可泛化的双臂视听运动策略 

**Authors**: Georgios Tziafas, Jiayun Zhang, Hamidreza Kasaei  

**Link**: [PDF](https://arxiv.org/pdf/2509.20286)  

**Abstract**: Learning visuomotor policies from expert demonstrations is an important frontier in modern robotics research, however, most popular methods require copious efforts for collecting teleoperation data and struggle to generalize out-ofdistribution. Scaling data collection has been explored through leveraging human videos, as well as demonstration augmentation techniques. The latter approach typically requires expensive simulation rollouts and trains policies with synthetic image data, therefore introducing a sim-to-real gap. In parallel, alternative state representations such as keypoints have shown great promise for category-level generalization. In this work, we bring these avenues together in a unified framework: PAD (Parse-AugmentDistill), for learning generalizable bimanual policies from a single human video. Our method relies on three steps: (a) parsing a human video demo into a robot-executable keypoint-action trajectory, (b) employing bimanual task-and-motion-planning to augment the demonstration at scale without simulators, and (c) distilling the augmented trajectories into a keypoint-conditioned policy. Empirically, we showcase that PAD outperforms state-ofthe-art bimanual demonstration augmentation works relying on image policies with simulation rollouts, both in terms of success rate and sample/cost efficiency. We deploy our framework in six diverse real-world bimanual tasks such as pouring drinks, cleaning trash and opening containers, producing one-shot policies that generalize in unseen spatial arrangements, object instances and background distractors. Supplementary material can be found in the project webpage this https URL. 

**Abstract (ZH)**: 从单个人工视频学习可泛化的双臂政策是现代机器人研究的一个重要前沿，然而，大多数流行的方法需要大量努力来收集遥操作数据，并且难以泛化到分布外。通过利用人类视频和演示增强技术扩展数据收集已经被探索。后者通常需要昂贵的模拟仿真并使用合成图像数据训练策略，因此引入了模拟仿真到现实应用之间的差距。与此同时，如关键点等替代状态表示显示出在类别级别泛化方面的巨大潜力。在本工作中，我们将这些途径整合到一个统一的框架中：PAD（解析-增强-精炼），用于从单个人工视频中学习可泛化的双臂政策。我们的方法依赖于三个步骤：(a) 将人工视频演示解析为机器人可执行的关键点-动作轨迹；(b) 使用双臂任务-运动规划在无模拟器的情况下大规模增强演示；(c) 将增强的轨迹精炼为关键点条件策略。在实验中，我们展示PAD在成功率和样本/成本效率方面均优于依赖图像策略和模拟仿真的双臂演示增强最新方法。我们在六种不同的现实世界双臂任务中部署了该框架，如倒饮料、清理垃圾和开容器，生成了在未见的空间布局、物体实例和背景干扰下泛化的单次演示策略。更多补充材料可以在项目网页上查看。 

---
# HL-IK: A Lightweight Implementation of Human-Like Inverse Kinematics in Humanoid Arms 

**Title (ZH)**: HL-IK: 人体like逆运动学的轻量级实现于类人臂中 

**Authors**: Bingjie Chen, Zihan Wang, Zhe Han, Guoping Pan, Yi Cheng, Houde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20263)  

**Abstract**: Traditional IK methods for redundant humanoid manipulators emphasize end-effector (EE) tracking, frequently producing configurations that are valid mechanically but not human-like. We present Human-Like Inverse Kinematics (HL-IK), a lightweight IK framework that preserves EE tracking while shaping whole-arm configurations to appear human-like, without full-body sensing at runtime. The key idea is a learned elbow prior: using large-scale human motion data retargeted to the robot, we train a FiLM-modulated spatio-temporal attention network (FiSTA) to predict the next-step elbow pose from the EE target and a short history of EE-elbow this http URL prediction is incorporated as a small residual alongside EE and smoothness terms in a standard Levenberg-Marquardt optimizer, making HL-IK a drop-in addition to numerical IK stacks. Over 183k simulation steps, HL-IK reduces arm-similarity position and direction error by 30.6% and 35.4% on average, and by 42.2% and 47.4% on the most challenging trajectories. Hardware teleoperation on a robot distinct from simulation further confirms the gains in anthropomorphism. HL-IK is simple to integrate, adaptable across platforms via our pipeline, and adds minimal computation, enabling human-like motions for humanoid robots. Project page: this https URL 

**Abstract (ZH)**: 类人逆运动学（HL-IK）：一种轻量级的保留末端执行器跟踪并形成长臂类人配置的逆运动学框架 

---
# AnchDrive: Bootstrapping Diffusion Policies with Hybrid Trajectory Anchors for End-to-End Driving 

**Title (ZH)**: AnchDrive: 使用混合轨迹锚点bootstrap端到端驾驶策略 

**Authors**: Jinhao Chai, Anqing Jiang, Hao Jiang, Shiyi Mu, Zichong Gu, Shugong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20253)  

**Abstract**: End-to-end multi-modal planning has become a transformative paradigm in autonomous driving, effectively addressing behavioral multi-modality and the generalization challenge in long-tail scenarios. We propose AnchDrive, a framework for end-to-end driving that effectively bootstraps a diffusion policy to mitigate the high computational cost of traditional generative models. Rather than denoising from pure noise, AnchDrive initializes its planner with a rich set of hybrid trajectory anchors. These anchors are derived from two complementary sources: a static vocabulary of general driving priors and a set of dynamic, context-aware trajectories. The dynamic trajectories are decoded in real-time by a Transformer that processes dense and sparse perceptual features. The diffusion model then learns to refine these anchors by predicting a distribution of trajectory offsets, enabling fine-grained refinement. This anchor-based bootstrapping design allows for efficient generation of diverse, high-quality trajectories. Experiments on the NAVSIM benchmark confirm that AnchDrive sets a new state-of-the-art and shows strong gen?eralizability 

**Abstract (ZH)**: 端到端多模態规划已成为自主驾驶领域的 transformative 帕累托，有效解决了长尾场景中的行为多模態性和泛化挑战。我们提出 AnchDrive，一种框架，能够有效启动扩散策略以缓解传统生成模型的高计算成本。与从纯噪声去噪不同，AnchDrive 使用丰富的一系列混合轨迹锚点初始化其规划器。这些锚点来源于两种互补的来源：一套静态的一般驾驶先验词汇和一组动态的、上下文感知的轨迹。动态轨迹由 Transformer 实时解码，处理密集和稀疏的感觉特征。扩散模型随后通过预测轨迹偏移的分布学习细化这些锚点，实现细粒度的改进。基于锚点的启动设计允许高效生成多样且高质量的轨迹。在 NAVSIM 基准上的实验表明，AnchDrive 达到了新的 SOTA，并展示了强大的泛化能力。 

---
# Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving 

**Title (ZH)**: 离散扩散在自主驾驶反射型 vision-language-action 模型中的应用 

**Authors**: Pengxiang Li, Yinan Zheng, Yue Wang, Huimin Wang, Hang Zhao, Jingjing Liu, Xianyuan Zhan, Kun Zhan, Xianpeng Lang  

**Link**: [PDF](https://arxiv.org/pdf/2509.20109)  

**Abstract**: End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems. 

**Abstract (ZH)**: 端到端（E2E）解决方案已成为自主驾驶系统的主要方法，Vision-Language-Action（VLA）模型作为一种新的范式，利用预训练的多模态知识从视觉语言模型（VLMs）中解释和交互复杂的现实环境。然而，这些方法仍受模仿学习限制的约束，在训练过程中难以内在编码物理规则。现有方法通常依赖于复杂的基于规则的后精修，或者采用在模拟中仍然基本受限的强化学习，或利用需要昂贵梯度计算的支持扩散指导。为了解决这些挑战，我们引入了ReflectDrive，这是一种新颖的学习框架，它通过离散扩散集成了一个反射机制以实现安全轨迹生成。我们首先离散化二维驾驶空间以构建动作词典，并通过微调使预训练的扩散语言模型能够用于规划任务。我们方法的核心是一个安全意识反射机制，它在不进行梯度计算的情况下进行迭代自我校正。该方法从基于目标的轨迹生成开始，以建模多模态驾驶行为。在此基础上，我们应用局部搜索方法来识别不安全的令牌并确定可行的解决方案，这些解决方案随后作为基于填充生成的安全锚点。在NAVSIM基准上评估，ReflectDrive在安全关键轨迹生成方面表现出显著优势，为自主驾驶系统提供了可扩展且可靠的方法。 

---
# Queryable 3D Scene Representation: A Multi-Modal Framework for Semantic Reasoning and Robotic Task Planning 

**Title (ZH)**: 可查询的3D场景表示：一种用于语义推理和机器人任务规划的多模态框架 

**Authors**: Xun Li, Rodrigo Santa Cruz, Mingze Xi, Hu Zhang, Madhawa Perera, Ziwei Wang, Ahalya Ravendran, Brandon J. Matthews, Feng Xu, Matt Adcock, Dadong Wang, Jiajun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20077)  

**Abstract**: To enable robots to comprehend high-level human instructions and perform complex tasks, a key challenge lies in achieving comprehensive scene understanding: interpreting and interacting with the 3D environment in a meaningful way. This requires a smart map that fuses accurate geometric structure with rich, human-understandable semantics. To address this, we introduce the 3D Queryable Scene Representation (3D QSR), a novel framework built on multimedia data that unifies three complementary 3D representations: (1) 3D-consistent novel view rendering and segmentation from panoptic reconstruction, (2) precise geometry from 3D point clouds, and (3) structured, scalable organization via 3D scene graphs. Built on an object-centric design, the framework integrates with large vision-language models to enable semantic queryability by linking multimodal object embeddings, and supporting object-level retrieval of geometric, visual, and semantic information. The retrieved data are then loaded into a robotic task planner for downstream execution. We evaluate our approach through simulated robotic task planning scenarios in Unity, guided by abstract language instructions and using the indoor public dataset Replica. Furthermore, we apply it in a digital duplicate of a real wet lab environment to test QSR-supported robotic task planning for emergency response. The results demonstrate the framework's ability to facilitate scene understanding and integrate spatial and semantic reasoning, effectively translating high-level human instructions into precise robotic task planning in complex 3D environments. 

**Abstract (ZH)**: 为了使机器人能够理解高层次的人类指令并执行复杂任务，关键挑战在于实现全面的场景理解：以有意义的方式解析和与三维环境互动。这需要一个智能地图，融合精确的几何结构和丰富的、人类可理解的语义。为此，我们提出了三维可查询场景表示（3D Queryable Scene Representation，3D QSR），这是一个基于多媒体数据的新型框架，统一了三个互补的三维表示：（1）来自全景重建的三维一致的新视角渲染和分割，（2）来自三维点云的精确几何结构，以及（3）通过三维场景图实现的结构化、可扩展的组织。基于对象中心的设计，该框架与大规模的多模态视觉语言模型集成，通过链接多模态对象嵌入实现代词查询性，并支持对象级别的几何、视觉和语义信息检索。检索的数据随后加载到机器人任务规划器中，以便下游执行。我们通过在Unity中模拟的机器人任务规划场景评估该方法，使用抽象语言指令并结合室内公共数据集Replica。此外，我们在一个真实的湿实验室环境的数字副本中应用该方法，测试QSR支持的机器人任务规划在紧急响应中的应用。结果表明，该框架能够促进场景理解并整合空间和语义推理，有效地将高层次的人类指令转化为复杂的三维环境中的精确机器人任务规划。 

---
# LLM Trainer: Automated Robotic Data Generating via Demonstration Augmentation using LLMs 

**Title (ZH)**: LLM训练器：通过LLM增强示范生成的自动化机器人数据生成 

**Authors**: Abraham George, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2509.20070)  

**Abstract**: We present LLM Trainer, a fully automated pipeline that leverages the world knowledge of Large Language Models (LLMs) to transform a small number of human demonstrations (as few as one) into a large robot dataset for imitation learning. Our approach decomposes demonstration generation into two steps: (1) offline demonstration annotation that extracts keyframes, salient objects, and pose-object relations; and (2) online keypose retargeting that adapts those keyframes to a new scene, given an initial observation. Using these modified keypoints, our system warps the original demonstration to generate a new trajectory, which is then executed, and the resulting demo, if successful, is saved. Because the annotation is reusable across scenes, we use Thompson sampling to optimize the annotation, significantly improving generation success rate. We evaluate our method on a range of tasks, and find that our data annotation method consistently outperforms expert-engineered baselines. We further show an ensemble policy that combines the optimized LLM feed-forward plan with a learned feedback imitation learning controller. Finally, we demonstrate hardware feasibility on a Franka Emika Panda robot. For additional materials and demonstration videos, please see the project website: this https URL 

**Abstract (ZH)**: LLM Trainer：利用大型语言模型知识自动生成机器人演示数据的流水线 

---
# MARG: MAstering Risky Gap Terrains for Legged Robots with Elevation Mapping 

**Title (ZH)**: MARG: 基于高程地图学习风险地型跨越技巧的腿足机器人技术 

**Authors**: Yinzhao Dong, Ji Ma, Liu Zhao, Wanyue Li, Peng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20036)  

**Abstract**: Deep Reinforcement Learning (DRL) controllers for quadrupedal locomotion have demonstrated impressive performance on challenging terrains, allowing robots to execute complex skills such as climbing, running, and jumping. However, existing blind locomotion controllers often struggle to ensure safety and efficient traversal through risky gap terrains, which are typically highly complex, requiring robots to perceive terrain information and select appropriate footholds during locomotion accurately. Meanwhile, existing perception-based controllers still present several practical limitations, including a complex multi-sensor deployment system and expensive computing resource requirements. This paper proposes a DRL controller named MAstering Risky Gap Terrains (MARG), which integrates terrain maps and proprioception to dynamically adjust the action and enhance the robot's stability in these tasks. During the training phase, our controller accelerates policy optimization by selectively incorporating privileged information (e.g., center of mass, friction coefficients) that are available in simulation but unmeasurable directly in real-world deployments due to sensor limitations. We also designed three foot-related rewards to encourage the robot to explore safe footholds. More importantly, a terrain map generation (TMG) model is proposed to reduce the drift existing in mapping and provide accurate terrain maps using only one LiDAR, providing a foundation for zero-shot transfer of the learned policy. The experimental results indicate that MARG maintains stability in various risky terrain tasks. 

**Abstract (ZH)**: 基于深强化学习的克服危险间隙地形的四足机器人控制器（MAstering Risky Gap Terrains, MARG） 

---
# Generalist Robot Manipulation beyond Action Labeled Data 

**Title (ZH)**: 超越动作标注数据的一般机器人 manipulation 

**Authors**: Alexander Spiridonov, Jan-Nico Zaech, Nikolay Nikolov, Luc Van Gool, Danda Pani Paudel  

**Link**: [PDF](https://arxiv.org/pdf/2509.19958)  

**Abstract**: Recent advances in generalist robot manipulation leverage pre-trained Vision-Language Models (VLMs) and large-scale robot demonstrations to tackle diverse tasks in a zero-shot manner. A key challenge remains: scaling high-quality, action-labeled robot demonstration data, which existing methods rely on for robustness and generalization. To address this, we propose a method that benefits from videos without action labels - featuring humans and/or robots in action - enhancing open-vocabulary performance and enabling data-efficient learning of new tasks. Our method extracts dense, dynamic 3D point clouds at the hand or gripper location and uses a proposed 3D dynamics predictor for self-supervision. This predictor is then tuned to an action predictor using a smaller labeled dataset for action alignment. We show that our method not only learns from unlabeled human and robot demonstrations - improving downstream generalist robot policies - but also enables robots to learn new tasks without action labels (i.e., out-of-action generalization) in both real-world and simulated settings. 

**Abstract (ZH)**: Recent Advances in Generalist Robot Manipulation via Action-Unlabeled Video Data and 3D Dynamics Prediction 

---
# GUIDE: A Diffusion-Based Autonomous Robot Exploration Framework Using Global Graph Inference 

**Title (ZH)**: GUIDE：基于全局图推理的扩散自主导航探索框架 

**Authors**: Zijun Che, Yinghong Zhang, Shengyi Liang, Boyu Zhou, Jun Ma, Jinni Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.19916)  

**Abstract**: Autonomous exploration in structured and complex indoor environments remains a challenging task, as existing methods often struggle to appropriately model unobserved space and plan globally efficient paths. To address these limitations, we propose GUIDE, a novel exploration framework that synergistically combines global graph inference with diffusion-based decision-making. We introduce a region-evaluation global graph representation that integrates both observed environmental data and predictions of unexplored areas, enhanced by a region-level evaluation mechanism to prioritize reliable structural inferences while discounting uncertain predictions. Building upon this enriched representation, a diffusion policy network generates stable, foresighted action sequences with significantly reduced denoising steps. Extensive simulations and real-world deployments demonstrate that GUIDE consistently outperforms state-of-the-art methods, achieving up to 18.3% faster coverage completion and a 34.9% reduction in redundant movements. 

**Abstract (ZH)**: 自主探索结构化和复杂室内环境仍然是一个具有挑战性的任务，现有方法往往难以适当建模未观察到的空间并规划全局高效路径。为解决这些问题，我们提出GUIDE，一种新颖的探索框架，结合了全局图推理与扩散决策机制。我们引入一种区域评估全局图表示，整合了已观察到的环境数据和未探索区域的预测，并通过区域级评估机制优先考虑可靠的结构性推理，同时忽略不确定的预测。基于这种丰富的表示，扩散策略网络生成了稳定、前瞻性的行动序列，显著减少了去噪步骤。广泛的技术模拟和实际部署表明，GUIDE 一致地优于现有最佳方法，实现了高达18.3%更快的覆盖率完成，并减少了34.9%的冗余移动。 

---
# D3Grasp: Diverse and Deformable Dexterous Grasping for General Objects 

**Title (ZH)**: D3Grasp: 多样化且可变形的通用物体灵巧抓取 

**Authors**: Keyu Wang, Bingcong Lu, Zhengxue Cheng, Hengdi Zhang, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.19892)  

**Abstract**: Achieving diverse and stable dexterous grasping for general and deformable objects remains a fundamental challenge in robotics, due to high-dimensional action spaces and uncertainty in perception. In this paper, we present D3Grasp, a multimodal perception-guided reinforcement learning framework designed to enable Diverse and Deformable Dexterous Grasping. We firstly introduce a unified multimodal representation that integrates visual and tactile perception to robustly grasp common objects with diverse properties. Second, we propose an asymmetric reinforcement learning architecture that exploits privileged information during training while preserving deployment realism, enhancing both generalization and sample efficiency. Third, we meticulously design a training strategy to synthesize contact-rich, penetration-free, and kinematically feasible grasps with enhanced adaptability to deformable and contact-sensitive objects. Extensive evaluations confirm that D3Grasp delivers highly robust performance across large-scale and diverse object categories, and substantially advances the state of the art in dexterous grasping for deformable and compliant objects, even under perceptual uncertainty and real-world disturbances. D3Grasp achieves an average success rate of 95.1% in real-world trials,outperforming prior methods on both rigid and deformable objects benchmarks. 

**Abstract (ZH)**: 实现对通用和可变形物体的多样化稳定灵巧抓取仍然是机器人领域的基本挑战，由于高维度的动作空间和感知的不确定性。本文提出D3Grasp，一种多模态感知引导的强化学习框架，旨在实现多样化和可变形灵巧抓取。 

---
# SAGE:State-Aware Guided End-to-End Policy for Multi-Stage Sequential Tasks via Hidden Markov Decision Process 

**Title (ZH)**: SAGE：具有状态感知引导的多阶段序列任务端到端策略隐马尔可夫决策过程 

**Authors**: BinXu Wu, TengFei Zhang, Chen Yang, JiaHao Wen, HaoCheng Li, JingTian Ma, Zhen Chen, JingYuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19853)  

**Abstract**: Multi-stage sequential (MSS) robotic manipulation tasks are prevalent and crucial in robotics. They often involve state ambiguity, where visually similar observations correspond to different actions. We present SAGE, a state-aware guided imitation learning framework that models tasks as a Hidden Markov Decision Process (HMDP) to explicitly capture latent task stages and resolve ambiguity. We instantiate the HMDP with a state transition network that infers hidden states, and a state-aware action policy that conditions on both observations and hidden states to produce actions, thereby enabling disambiguation across task stages. To reduce manual annotation effort, we propose a semi-automatic labeling pipeline combining active learning and soft label interpolation. In real-world experiments across multiple complex MSS tasks with state ambiguity, SAGE achieved 100% task success under the standard evaluation protocol, markedly surpassing the baselines. Ablation studies further show that such performance can be maintained with manual labeling for only about 13% of the states, indicating its strong effectiveness. 

**Abstract (ZH)**: 多阶段顺序（MSS）机器人操作任务在机器人领域普遍存在且至关重要。它们经常涉及状态不确定性，其中视觉上相似的观测对应不同的操作。我们提出了SAGE，一种状态感知引导的模仿学习框架，将任务建模为隐马尔可夫决策过程（HMDP）以明确捕捉隐含的任务阶段并解决不确定性。我们使用状态转换网络实例化HMDP以推断隐藏状态，并使用状态感知的动作策略基于观测和隐藏状态生成动作，从而在任务阶段之间实现去混淆。为了减少手动标注努力，我们提出了一种结合主动学习和软标签插值的半自动标注管道。在使用标准评估协议的多个具有状态不确定性的复杂MSS任务的真实世界实验中，SAGE实现了100%的任务成功，明显超过了基线。进一步的消融研究显示，这种性能可以通过仅为约13%的状态进行手动标注来保持，表明其强大的有效性。 

---
# Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments 

**Title (ZH)**: 我在哪里留下了我的眼镜？半静态现实环境中的开放词汇语义探索 

**Authors**: Benjamin Bogenberger, Oliver Harrison, Orrin Dahanaggamaarachchi, Lukas Brunke, Jingxing Qian, Siqi Zhou, Angela P. Schoellig  

**Link**: [PDF](https://arxiv.org/pdf/2509.19851)  

**Abstract**: Robots deployed in real-world environments, such as homes, must not only navigate safely but also understand their surroundings and adapt to environment changes. To perform tasks efficiently, they must build and maintain a semantic map that accurately reflects the current state of the environment. Existing research on semantic exploration largely focuses on static scenes without persistent object-level instance tracking. A consistent map is, however, crucial for real-world robotic applications where objects in the environment can be removed, reintroduced, or shifted over time. In this work, to close this gap, we propose an open-vocabulary, semantic exploration system for semi-static environments. Our system maintains a consistent map by building a probabilistic model of object instance stationarity, systematically tracking semi-static changes, and actively exploring areas that have not been visited for a prolonged period of time. In addition to active map maintenance, our approach leverages the map's semantic richness with LLM-based reasoning for open-vocabulary object-goal navigation. This enables the robot to search more efficiently by prioritizing contextually relevant areas. We evaluate our approach across multiple real-world semi-static environments. Our system detects 95% of map changes on average, improving efficiency by more than 29% as compared to random and patrol baselines. Overall, our approach achieves a mapping precision within 2% of a fully rebuilt map while requiring substantially less exploration and further completes object goal navigation tasks about 14% faster than the next-best tested strategy (coverage patrolling). A video of our work can be found at this http URL . 

**Abstract (ZH)**: 现实环境中（如家庭）部署的机器人不仅要安全导航，还要理解其环境并适应环境变化。为了高效执行任务，它们必须构建并维护一个准确反映当前环境状态的语义地图。现有基于语义探索的研究主要集中在静态场景，而忽略了持续的对象级实例跟踪。然而，在现实世界的机器人应用中，环境中的物体可能会被移除、重新引入或随着时间的推移而移动，因此一致的地图至关重要。在此工作中，为了弥补这一差距，我们提出了一种适用于半静态环境的开放词汇语义探索系统。该系统通过构建对象实例稳定性的概率模型、系统地跟踪半静态变化、并主动探索长时间未被访问的区域，来维护一致的地图。除了积极维护地图，我们的方法还利用基于LLM的推理来利用地图的语义丰富性进行开放词汇对象目标导航。这使得机器人可以通过优先考虑上下文相关区域来更有效地进行搜索。我们在多个真实世界的半静态环境中评估了我们的方法。我们的系统平均检测出95%的地图变化，与随机巡逻基线相比，效率提高了超过29%。总体而言，我们的方法在映射精度方面与完全重建的地图相差仅2%以内，同时探索需求大大减少，并且比测试的最佳策略（覆盖巡逻）快约14%完成对象目标导航任务。 

---
# DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations 

**Title (ZH)**: DynaFlow：嵌入动力学的流匹配方法，用于来自状态演示的一致运动生成 

**Authors**: Sowoo Lee, Dongyun Kang, Jaehyun Park, Hae-Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.19804)  

**Abstract**: This paper introduces DynaFlow, a novel framework that embeds a differentiable simulator directly into a flow matching model. By generating trajectories in the action space and mapping them to dynamically feasible state trajectories via the simulator, DynaFlow ensures all outputs are physically consistent by construction. This end-to-end differentiable architecture enables training on state-only demonstrations, allowing the model to simultaneously generate physically consistent state trajectories while inferring the underlying action sequences required to produce them. We demonstrate the effectiveness of our approach through quantitative evaluations and showcase its real-world applicability by deploying the generated actions onto a physical Go1 quadruped robot. The robot successfully reproduces diverse gait present in the dataset, executes long-horizon motions in open-loop control and translates infeasible kinematic demonstrations into dynamically executable, stylistic behaviors. These hardware experiments validate that DynaFlow produces deployable, highly effective motions on real-world hardware from state-only demonstrations, effectively bridging the gap between kinematic data and real-world execution. 

**Abstract (ZH)**: 本论文介绍了DynaFlow这一新颖框架，该框架直接将可微分模拟器嵌入到流匹配模型中。通过在动作空间中生成轨迹并借助模拟器将其映射为动态可行的状态轨迹，DynaFlow通过设计确保所有输出均具有物理一致性。这一端到端的可微分架构允许仅基于状态示例进行训练，使模型能够在生成物理一致的状态轨迹的同时推断出产生这些轨迹所需的动作序列。通过定量评估展示了该方法的有效性，并通过将生成的动作应用于物理Go1四足机器人展示了其实用性。机器人成功再现了数据集中多种步态，执行了开环控制下的长时序动作，并将不可能的运动学示例转换为动态可执行且具有风格化的行为。这些硬件实验验证了DynaFlow能够从仅状态示例中生成在实际硬件上可部署且高效的运动，从而有效弥合了运动学数据与实际执行之间的差距。 

---
# Formal Safety Verification and Refinement for Generative Motion Planners via Certified Local Stabilization 

**Title (ZH)**: 基于认证局部稳定性的生成运动规划器的形式安全验证与细化 

**Authors**: Devesh Nath, Haoran Yin, Glen Chou  

**Link**: [PDF](https://arxiv.org/pdf/2509.19688)  

**Abstract**: We present a method for formal safety verification of learning-based generative motion planners. Generative motion planners (GMPs) offer advantages over traditional planners, but verifying the safety and dynamic feasibility of their outputs is difficult since neural network verification (NNV) tools scale only to a few hundred neurons, while GMPs often contain millions. To preserve GMP expressiveness while enabling verification, our key insight is to imitate the GMP by stabilizing references sampled from the GMP with a small neural tracking controller and then applying NNV to the closed-loop dynamics. This yields reachable sets that rigorously certify closed-loop safety, while the controller enforces dynamic feasibility. Building on this, we construct a library of verified GMP references and deploy them online in a way that imitates the original GMP distribution whenever it is safe to do so, improving safety without retraining. We evaluate across diverse planners, including diffusion, flow matching, and vision-language models, improving safety in simulation (on ground robots and quadcopters) and on hardware (differential-drive robot). 

**Abstract (ZH)**: 基于学习的生成运动规划形式化安全验证方法 

---
# Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains 

**Title (ZH)**: 记忆增强潜力场理论：非凸域自适应控制的框架 

**Authors**: Dongzhe Zheng, Wenjie Mei  

**Link**: [PDF](https://arxiv.org/pdf/2509.19672)  

**Abstract**: Stochastic optimal control methods often struggle in complex non-convex landscapes, frequently becoming trapped in local optima due to their inability to learn from historical trajectory data. This paper introduces Memory-Augmented Potential Field Theory, a unified mathematical framework that integrates historical experience into stochastic optimal control. Our approach dynamically constructs memory-based potential fields that identify and encode key topological features of the state space, enabling controllers to automatically learn from past experiences and adapt their optimization strategy. We provide a theoretical analysis showing that memory-augmented potential fields possess non-convex escape properties, asymptotic convergence characteristics, and computational efficiency. We implement this theoretical framework in a Memory-Augmented Model Predictive Path Integral (MPPI) controller that demonstrates significantly improved performance in challenging non-convex environments. The framework represents a generalizable approach to experience-based learning within control systems (especially robotic dynamics), enhancing their ability to navigate complex state spaces without requiring specialized domain knowledge or extensive offline training. 

**Abstract (ZH)**: 增强记忆的潜在场理论：一种将历史经验融入随机最优控制的统一数学框架 

---
# RoboSSM: Scalable In-context Imitation Learning via State-Space Models 

**Title (ZH)**: RoboSSM：基于状态空间模型的可扩展上下文模仿学习 

**Authors**: Youngju Yoo, Jiaheng Hu, Yifeng Zhu, Bo Liu, Qiang Liu, Roberto Martín-Martín, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2509.19658)  

**Abstract**: In-context imitation learning (ICIL) enables robots to learn tasks from prompts consisting of just a handful of demonstrations. By eliminating the need for parameter updates at deployment time, this paradigm supports few-shot adaptation to novel tasks. However, recent ICIL methods rely on Transformers, which have computational limitations and tend to underperform when handling longer prompts than those seen during training. In this work, we introduce RoboSSM, a scalable recipe for in-context imitation learning based on state-space models (SSM). Specifically, RoboSSM replaces Transformers with Longhorn -- a state-of-the-art SSM that provides linear-time inference and strong extrapolation capabilities, making it well-suited for long-context prompts. We evaluate our approach on the LIBERO benchmark and compare it against strong Transformer-based ICIL baselines. Experiments show that RoboSSM extrapolates effectively to varying numbers of in-context demonstrations, yields high performance on unseen tasks, and remains robust in long-horizon scenarios. These results highlight the potential of SSMs as an efficient and scalable backbone for ICIL. Our code is available at this https URL. 

**Abstract (ZH)**: 基于状态空间模型的在上下文模仿学习（RoboSSM） 

---
# EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data 

**Title (ZH)**: EgoBridge: 基于第一人称人类数据的泛化imitation学习域适应 

**Authors**: Ryan Punamiya, Dhruv Patel, Patcharapong Aphiwetsa, Pranav Kuppili, Lawrence Y. Zhu, Simar Kareer, Judy Hoffman, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19626)  

**Abstract**: Egocentric human experience data presents a vast resource for scaling up end-to-end imitation learning for robotic manipulation. However, significant domain gaps in visual appearance, sensor modalities, and kinematics between human and robot impede knowledge transfer. This paper presents EgoBridge, a unified co-training framework that explicitly aligns the policy latent spaces between human and robot data using domain adaptation. Through a measure of discrepancy on the joint policy latent features and actions based on Optimal Transport (OT), we learn observation representations that not only align between the human and robot domain but also preserve the action-relevant information critical for policy learning. EgoBridge achieves a significant absolute policy success rate improvement by 44% over human-augmented cross-embodiment baselines in three real-world single-arm and bimanual manipulation tasks. EgoBridge also generalizes to new objects, scenes, and tasks seen only in human data, where baselines fail entirely. Videos and additional information can be found at this https URL 

**Abstract (ZH)**: 以自我为中心的人类体验数据为机器人操作的端到端 imitation learning 扩容提供了巨大的资源。然而，人类和机器人在视觉外观、传感器模态和运动学之间的显著领域差距阻碍了知识迁移。本文提出了 EgoBridge，一种统一的协同训练框架，通过领域适应显式对齐人类和机器人数据的策略潜在空间。基于最优传输（Optimal Transport，OT）在联合策略潜在特征和动作之间的不一致性度量中，我们学习了既能对齐人类和机器人领域又能保留对策略学习至关重要的动作相关信息的观测表示。在三个真实世界的单臂和双臂操作任务中，EgoBridge 的绝对策略成功率相较于基于人类数据增强的跨主体基线显著提高了 44%。EgoBridge 还可以泛化到仅在人类数据中出现的新物体、场景和任务，而基线则完全失败。更多视频和详细信息请参见此链接：[补充链接文字]。 

---
# Look as You Leap: Planning Simultaneous Motion and Perception for High-DOF Robots 

**Title (ZH)**: 凌空跃起时凝视：为高自由度机器人同时规划运动与感知 

**Authors**: Qingxi Meng, Emiliano Flores, Carlos Quintero-Peña, Peizhu Qian, Zachary Kingston, Shannan K. Hamlin, Vaibhav Unhelkar, Lydia E. Kavraki  

**Link**: [PDF](https://arxiv.org/pdf/2509.19610)  

**Abstract**: In this work, we address the problem of planning robot motions for a high-degree-of-freedom (DoF) robot that effectively achieves a given perception task while the robot and the perception target move in a dynamic environment. Achieving navigation and perception tasks simultaneously is challenging, as these objectives often impose conflicting requirements. Existing methods that compute motion under perception constraints fail to account for obstacles, are designed for low-DoF robots, or rely on simplified models of perception. Furthermore, in dynamic real-world environments, robots must replan and react quickly to changes and directly evaluating the quality of perception (e.g., object detection confidence) is often expensive or infeasible at runtime. This problem is especially important in human-centered environments such as homes and hospitals, where effective perception is essential for safe and reliable operation. To address these challenges, we propose a GPU-parallelized perception-score-guided probabilistic roadmap planner with a neural surrogate model (PS-PRM). The planner explicitly incorporates the estimated quality of a perception task into motion planning for high-DoF robots. Our method uses a learned model to approximate perception scores and leverages GPU parallelism to enable efficient online replanning in dynamic settings. We demonstrate that our planner, evaluated on high-DoF robots, outperforms baseline methods in both static and dynamic environments in both simulation and real-robot experiments. 

**Abstract (ZH)**: 基于感知评分指导的并行概率路障规划方法（用于高自由度机器人在动态环境下的运动规划） 

---
# Chasing Stability: Humanoid Running via Control Lyapunov Function Guided Reinforcement Learning 

**Title (ZH)**: 追逐稳定性：基于控制李雅普诺夫函数引导的强化学习 humanoid 运动控制 

**Authors**: Zachary Olkin, Kejun Li, William D. Compton, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2509.19573)  

**Abstract**: Achieving highly dynamic behaviors on humanoid robots, such as running, requires controllers that are both robust and precise, and hence difficult to design. Classical control methods offer valuable insight into how such systems can stabilize themselves, but synthesizing real-time controllers for nonlinear and hybrid dynamics remains challenging. Recently, reinforcement learning (RL) has gained popularity for locomotion control due to its ability to handle these complex dynamics. In this work, we embed ideas from nonlinear control theory, specifically control Lyapunov functions (CLFs), along with optimized dynamic reference trajectories into the reinforcement learning training process to shape the reward. This approach, CLF-RL, eliminates the need to handcraft and tune heuristic reward terms, while simultaneously encouraging certifiable stability and providing meaningful intermediate rewards to guide learning. By grounding policy learning in dynamically feasible trajectories, we expand the robot's dynamic capabilities and enable running that includes both flight and single support phases. The resulting policy operates reliably on a treadmill and in outdoor environments, demonstrating robustness to disturbances applied to the torso and feet. Moreover, it achieves accurate global reference tracking utilizing only on-board sensors, making a critical step toward integrating these dynamic motions into a full autonomy stack. 

**Abstract (ZH)**: 实现类人机器人等动态行为（如跑步）需要既 robust 又精确的控制器，因此设计起来很困难。经典控制方法为理解此类系统如何实现自身稳定提供了宝贵的见解，但合成用于非线性和混合动力学的实时控制器仍具有挑战性。近年来，强化学习（RL）因其能够处理这些复杂动力学而受到了越来越多的关注。在本工作中，我们将非线性控制理论中的想法，特别是控制李雅普诺夫函数（CLFs），以及优化的动态参考轨迹嵌入到强化学习训练过程中，以塑造奖励。这种CLF-RL方法消除了手动设计和调整启发式奖励项的需求，同时促进了可验证的稳定性，并为引导学习提供了有意义的中间奖励。通过将策略学习基于动态可行的轨迹，我们扩展了机器人的动态能力，并使机器人能够在包括飞行和单支撑相在内的跑步中表现出色。所获得的策略在跑步机和户外环境中可靠运行，表现出对作用于躯干和脚的干扰的鲁棒性。此外，它仅使用机载传感器实现了精确的整体参考轨迹跟踪，为将这些动态动作整合到完整的自主性堆栈中迈出了关键一步。 

---
# Agentic Scene Policies: Unifying Space, Semantics, and Affordances for Robot Action 

**Title (ZH)**: 代理场景策略：统一空间、语义和作用方式的机器人动作框架 

**Authors**: Sacha Morin, Kumaraditya Gupta, Mahtab Sandhu, Charlie Gauthier, Francesco Argenziano, Kirsty Ellis, Liam Paull  

**Link**: [PDF](https://arxiv.org/pdf/2509.19571)  

**Abstract**: Executing open-ended natural language queries is a core problem in robotics. While recent advances in imitation learning and vision-language-actions models (VLAs) have enabled promising end-to-end policies, these models struggle when faced with complex instructions and new scenes. An alternative is to design an explicit scene representation as a queryable interface between the robot and the world, using query results to guide downstream motion planning. In this work, we present Agentic Scene Policies (ASP), an agentic framework that leverages the advanced semantic, spatial, and affordance-based querying capabilities of modern scene representations to implement a capable language-conditioned robot policy. ASP can execute open-vocabulary queries in a zero-shot manner by explicitly reasoning about object affordances in the case of more complex skills. Through extensive experiments, we compare ASP with VLAs on tabletop manipulation problems and showcase how ASP can tackle room-level queries through affordance-guided navigation, and a scaled-up scene representation. (Project page: this https URL) 

**Abstract (ZH)**: 执行开放式的自然语言查询是机器人技术中的核心问题。虽然近期在模仿学习和视觉-语言-动作模型（VLAs）方面的进展已经使端到端策略变得颇具前景，但这些模型在面对复杂的指示和新场景时表现不佳。一种替代方案是设计一个明确的场景表示作为机器人与世界之间的查询接口，利用查询结果指导后续的动作规划。在这项工作中，我们提出了有能行动态场景策略（Agentic Scene Policies, ASP），这是一种利用现代场景表示的高级语义、空间和利用基于查询能力的有能行动态场景策略框架，以实现一个基于语言的机器人策略。ASP 可以在零样本情况下执行开放式词汇查询，通过明确推理对象的利用能力来应对更复杂的技能。通过大量实验，我们将 ASP 与 VLAs 在桌面上的操作问题上进行了对比，并展示了ASP如何通过利用基于利用导航和扩展的场景表示来应对房间级别的查询。 

---
# AnySafe: Adapting Latent Safety Filters at Runtime via Safety Constraint Parameterization in the Latent Space 

**Title (ZH)**: AnySafe: 通过潜在空间安全性约束参数化在运行时适应潜在安全性过滤器 

**Authors**: Sankalp Agrawal, Junwon Seo, Kensuke Nakamura, Ran Tian, Andrea Bajcsy  

**Link**: [PDF](https://arxiv.org/pdf/2509.19555)  

**Abstract**: Recent works have shown that foundational safe control methods, such as Hamilton-Jacobi (HJ) reachability analysis, can be applied in the latent space of world models. While this enables the synthesis of latent safety filters for hard-to-model vision-based tasks, they assume that the safety constraint is known a priori and remains fixed during deployment, limiting the safety filter's adaptability across scenarios. To address this, we propose constraint-parameterized latent safety filters that can adapt to user-specified safety constraints at runtime. Our key idea is to define safety constraints by conditioning on an encoding of an image that represents a constraint, using a latent-space similarity measure. The notion of similarity to failure is aligned in a principled way through conformal calibration, which controls how closely the system may approach the constraint representation. The parameterized safety filter is trained entirely within the world model's imagination, treating any image seen by the model as a potential test-time constraint, thereby enabling runtime adaptation to arbitrary safety constraints. In simulation and hardware experiments on vision-based control tasks with a Franka manipulator, we show that our method adapts at runtime by conditioning on the encoding of user-specified constraint images, without sacrificing performance. Video results can be found on this https URL 

**Abstract (ZH)**: 近期的研究表明，基础的安全控制方法，如哈密尔顿-雅可比（HJ）可达性分析，可以应用于世界模型的潜在空间中。虽然这种方法允许为基于视觉且难以建模的任务生成潜在空间的安全滤波器，但它假设安全约束在部署前已知且在部署过程中保持不变，从而限制了安全滤波器在不同场景中的适应性。为解决这一问题，我们提出了参数化的潜在空间安全滤波器，可以在运行时根据用户指定的安全约束进行适应。我们的核心思想是通过基于潜在空间的相似度度量来条件编码表示约束的图像，定义安全约束。通过对称校准将失败相似性的观念以一种原则性的方式进行对齐，从而控制系统接近约束表示的程度。参数化的安全滤波器完全在世界模型的想象中进行训练，将模型所见的任何图像视为潜在的测试时约束，从而允许在运行时对任意安全约束进行适应。在使用弗兰卡操作器进行的基于视觉的控制任务的仿真和硬件实验中，我们展示了该方法在运行时通过条件编码用户指定的约束图像进行适应，同时不牺牲性能。更多视频结果，请访问这个网址：这个 https URL。 

---
# OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation 

**Title (ZH)**: 全景VLA：一种用于机器人导航的跨模态视觉-语言-动作模型 

**Authors**: Noriaki Hirose, Catherine Glossop, Dhruv Shah, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2509.19480)  

**Abstract**: Humans can flexibly interpret and compose different goal specifications, such as language instructions, spatial coordinates, or visual references, when navigating to a destination. In contrast, most existing robotic navigation policies are trained on a single modality, limiting their adaptability to real-world scenarios where different forms of goal specification are natural and complementary. In this work, we present a training framework for robotic foundation models that enables omni-modal goal conditioning for vision-based navigation. Our approach leverages a high-capacity vision-language-action (VLA) backbone and trains with three primary goal modalities: 2D poses, egocentric images, and natural language, as well as their combinations, through a randomized modality fusion strategy. This design not only expands the pool of usable datasets but also encourages the policy to develop richer geometric, semantic, and visual representations. The resulting model, OmniVLA, achieves strong generalization to unseen environments, robustness to scarce modalities, and the ability to follow novel natural language instructions. We demonstrate that OmniVLA outperforms specialist baselines across modalities and offers a flexible foundation for fine-tuning to new modalities and tasks. We believe OmniVLA provides a step toward broadly generalizable and flexible navigation policies, and a scalable path for building omni-modal robotic foundation models. We present videos showcasing OmniVLA performance and will release its checkpoints and training code on our project page. 

**Abstract (ZH)**: 人类在导航到目的地时能够灵活地解释和组合不同类型的目標 specifications，如语言指令、空间坐标或视觉参考。相比之下，现有的大多数机器人导航策略仅针对单一模态进行训练，限制了它们在真实世界场景中的适应性，而不同的目标 specifications 自然且互补。在此项工作中，我们提出了一种训练框架，使基于视觉的导航能够适应多模态目标 conditioning。我们的方法利用高容量的视觉-语言-动作（VLA）骨干网络，并通过随机模态融合策略训练三种主要的目标模态：2D 姿态、第一人称图像和自然语言，以及它们的组合。该设计不仅扩展了可用数据集的范围，还鼓励策略发展更丰富的几何、语义和视觉表示。结果模型 OmniVLA 在未见过的环境中有较强的泛化能力、对稀少模态的鲁棒性，并且能够遵循新的自然语言指令。我们展示了 OmniVLA 在不同模态下优于专门基准模型，并提供了向新模态和任务微调的灵活性基础。我们相信 OmniVLA 为广泛泛化和灵活的导航策略提供了一步进展，并为构建多模态机器人基础模型指明了可扩展的道路。我们展示了 OmniVLA 的性能视频，并将在项目页面上发布其检查点和训练代码。 

---
# Self-evolved Imitation Learning in Simulated World 

**Title (ZH)**: 自我演化的模仿学习在模拟世界中 

**Authors**: Yifan Ye, Jun Cen, Jing Chen, Zhihe Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19460)  

**Abstract**: Imitation learning has been a trend recently, yet training a generalist agent across multiple tasks still requires large-scale expert demonstrations, which are costly and labor-intensive to collect. To address the challenge of limited supervision, we propose Self-Evolved Imitation Learning (SEIL), a framework that progressively improves a few-shot model through simulator interactions. The model first attempts tasksin the simulator, from which successful trajectories are collected as new demonstrations for iterative refinement. To enhance the diversity of these demonstrations, SEIL employs dual-level augmentation: (i) Model-level, using an Exponential Moving Average (EMA) model to collaborate with the primary model, and (ii) Environment-level, introducing slight variations in initial object positions. We further introduce a lightweight selector that filters complementary and informative trajectories from the generated pool to ensure demonstration quality. These curated samples enable the model to achieve competitive performance with far fewer training examples. Extensive experiments on the LIBERO benchmark show that SEIL achieves a new state-of-the-art performance in few-shot imitation learning scenarios. Code is available at this https URL. 

**Abstract (ZH)**: 自我演化的imitation学习（SEIL）：通过模拟器交互渐进优化Few-shot模型 

---
# ROPA: Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation 

**Title (ZH)**: ROPA：RGB-D 双手数据增强中的合成机器人姿态生成 

**Authors**: Jason Chen, I-Chun Arthur Liu, Gaurav Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2509.19454)  

**Abstract**: Training robust bimanual manipulation policies via imitation learning requires demonstration data with broad coverage over robot poses, contacts, and scene contexts. However, collecting diverse and precise real-world demonstrations is costly and time-consuming, which hinders scalability. Prior works have addressed this with data augmentation, typically for either eye-in-hand (wrist camera) setups with RGB inputs or for generating novel images without paired actions, leaving augmentation for eye-to-hand (third-person) RGB-D training with new action labels less explored. In this paper, we propose Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation (ROPA), an offline imitation learning data augmentation method that fine-tunes Stable Diffusion to synthesize third-person RGB and RGB-D observations of novel robot poses. Our approach simultaneously generates corresponding joint-space action labels while employing constrained optimization to enforce physical consistency through appropriate gripper-to-object contact constraints in bimanual scenarios. We evaluate our method on 5 simulated and 3 real-world tasks. Our results across 2625 simulation trials and 300 real-world trials demonstrate that ROPA outperforms baselines and ablations, showing its potential for scalable RGB and RGB-D data augmentation in eye-to-hand bimanual manipulation. Our project website is available at: this https URL. 

**Abstract (ZH)**: 通过模仿学习训练 robust 双手操作策略需要广泛覆盖机器人姿态、接触和场景上下文的示范数据。然而，收集多样且精确的实时示范代价高昂且耗时，这限制了其可扩展性。先前的研究通过数据增强来解决这一问题，通常集中在带有 RGB 输入的眼在手（腕部相机）设置上，或生成新的图像而没有配对的操作，而对于眼到手（第三人称）RGB-D 训练的数据增强，带有新操作标签的情况则探索较少。本文提出 Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation (ROPA)，这是一种离线模仿学习数据增强方法，通过调整 Stable Diffusion 来合成第三人称的 RGB 和 RGB-D 观测值的新机器人姿态。我们的方法同时生成相应的关节空间操作标签，并通过合适的双手操作接触约束应用约束优化来确保物理一致性。我们在 5 个模拟和 3 个真实世界任务上评估了该方法。我们的结果表明，在 2625 个模拟试验和 300 个真实世界试验中，ROPA 比基线和消融实验更优，展示了其在眼到手双手操作中实现可扩展的 RGB 和 RGB-D 数据增强的潜力。我们的项目网站可通过以下链接访问：this https URL。 

---
# Embodied AI: From LLMs to World Models 

**Title (ZH)**: 具身AI：从大规模语言模型到世界模型 

**Authors**: Tongtong Feng, Xin Wang, Yu-Gang Jiang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.20021)  

**Abstract**: Embodied Artificial Intelligence (AI) is an intelligent system paradigm for achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications and driving the evolution from cyberspace to physical systems. Recent breakthroughs in Large Language Models (LLMs) and World Models (WMs) have drawn significant attention for embodied AI. On the one hand, LLMs empower embodied AI via semantic reasoning and task decomposition, bringing high-level natural language instructions and low-level natural language actions into embodied cognition. On the other hand, WMs empower embodied AI by building internal representations and future predictions of the external world, facilitating physical law-compliant embodied interactions. As such, this paper comprehensively explores the literature in embodied AI from basics to advances, covering both LLM driven and WM driven works. In particular, we first present the history, key technologies, key components, and hardware systems of embodied AI, as well as discuss its development via looking from unimodal to multimodal angle. We then scrutinize the two burgeoning fields of embodied AI, i.e., embodied AI with LLMs/multimodal LLMs (MLLMs) and embodied AI with WMs, meticulously delineating their indispensable roles in end-to-end embodied cognition and physical laws-driven embodied interactions. Building upon the above advances, we further share our insights on the necessity of the joint MLLM-WM driven embodied AI architecture, shedding light on its profound significance in enabling complex tasks within physical worlds. In addition, we examine representative applications of embodied AI, demonstrating its wide applicability in real-world scenarios. Last but not least, we point out future research directions of embodied AI that deserve further investigation. 

**Abstract (ZH)**: 具身人工智能：实现人工通用智能的核心 paradigms 及其从虚拟空间到物理系统的演变：大型语言模型与世界模型的推动作用和未来发展展望 

---
# PersONAL: Towards a Comprehensive Benchmark for Personalized Embodied Agents 

**Title (ZH)**: PersONAL: 向全面个性化体态智能体基准迈进 

**Authors**: Filippo Ziliotto, Jelin Raphael Akkara, Alessandro Daniele, Lamberto Ballan, Luciano Serafini, Tommaso Campari  

**Link**: [PDF](https://arxiv.org/pdf/2509.19843)  

**Abstract**: Recent advances in Embodied AI have enabled agents to perform increasingly complex tasks and adapt to diverse environments. However, deploying such agents in realistic human-centered scenarios, such as domestic households, remains challenging, particularly due to the difficulty of modeling individual human preferences and behaviors. In this work, we introduce PersONAL (PERSonalized Object Navigation And Localization, a comprehensive benchmark designed to study personalization in Embodied AI. Agents must identify, retrieve, and navigate to objects associated with specific users, responding to natural-language queries such as "find Lily's backpack". PersONAL comprises over 2,000 high-quality episodes across 30+ photorealistic homes from the HM3D dataset. Each episode includes a natural-language scene description with explicit associations between objects and their owners, requiring agents to reason over user-specific semantics. The benchmark supports two evaluation modes: (1) active navigation in unseen environments, and (2) object grounding in previously mapped scenes. Experiments with state-of-the-art baselines reveal a substantial gap to human performance, highlighting the need for embodied agents capable of perceiving, reasoning, and memorizing over personalized information; paving the way towards real-world assistive robot. 

**Abstract (ZH)**: 近期，体现式AI的发展使代理能够执行越来越复杂的任务并适应多种环境。然而，在诸如家庭这样的现实的人本中心场景中部署这些代理仍然颇具挑战性，特别是由于难以建模个体的人类偏好和行为。在此项工作中，我们介绍了PersONAL（PERSonalized Object Navigation And Localization），一个全面的基准测试，旨在研究体现式AI中的个性化问题。代理必须识别、检索并导航至与特定用户相关联的对象，响应诸如“找到莉莉的背包”之类的自然语言查询。PersONAL包含来自HM3D数据集超过2,000个高质量的场景集，囊括30多个照片级真实感的家庭。每个场景集包括自然语言的场景描述，明确对象与其所有者之间的关联，要求代理进行用户特定语义的推理。基准测试支持两种评估模式：（1）在未见过的环境中进行主动导航，（2）在先前映射的场景中进行物体语义关联。使用当前最先进的基线进行的实验揭示了与人类表现之间存在显著差距，突显了能够感知、推理和记忆个性化信息的体现式代理的必要性；为走向实际辅助机器人铺平了道路。 

---
# Score the Steps, Not Just the Goal: VLM-Based Subgoal Evaluation for Robotic Manipulation 

**Title (ZH)**: 评分步骤，而不仅目标：基于VLM的机器人 manipulation子目标评估 

**Authors**: Ramy ElMallah, Krish Chhajer, Chi-Guhn Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19524)  

**Abstract**: Robot learning papers typically report a single binary success rate (SR), which obscures where a policy succeeds or fails along a multi-step manipulation task. We argue that subgoal-level reporting should become routine: for each trajectory, a vector of per-subgoal SRs that makes partial competence visible (e.g., grasp vs. pour). We propose a blueprint for StepEval, a cost-aware plug-in evaluation framework that utilizes vision-language models (VLMs) as automated judges of subgoal outcomes from recorded images or videos. Rather than proposing new benchmarks or APIs, our contribution is to outline design principles for a scalable, community-driven open-source project. In StepEval, the primary artifact for policy evaluation is the per-subgoal SR vector; however, other quantities (e.g., latency or cost estimates) are also considered for framework-optimization diagnostics to help the community tune evaluation efficiency and accuracy when ground-truth subgoal success labels are available. We discuss how such a framework can remain model-agnostic, support single- or multi-view inputs, and be lightweight enough to adopt across labs. The intended contribution is a shared direction: a minimal, extensible seed that invites open-source contributions, so that scoring the steps, not just the final goal, becomes a standard and reproducible practice. 

**Abstract (ZH)**: 机器人学习论文通常报告单个二元成功率（SR），这掩盖了沿多步操控任务中策略成功或失败的具体位置。我们argue应将子目标级别报告变为常规做法：对于每个轨迹，提供一个子目标级别SR向量，使部分能力可见（例如，抓取 vs 倾倒）。我们提出了一种StepEval的成本感知插件评估框架蓝图，该框架利用视觉-语言模型（VLMs）作为从记录的图像或视频中自动评估子目标结果的裁判。我们贡献在于概述了一个可扩展的、社区驱动的开源项目的德规范设计原则，而非提出新的基准或API。在StepEval中，策略评估的主要结果是子目标级别的SR向量；然而，其他量（例如，延迟或成本估算）也考虑用于框架优化诊断，以帮助社区在可用真实子目标成功标签时调优评估效率和准确性。我们讨论了此类框架如何保持模型无关性、支持单视角或多视角输入，并足够轻量以跨越实验室采用。我们旨在提供一个共享方向：一个最小化且可扩展的基础，邀请开源贡献，使得评分不仅限于最终目标，而是成为标准和可复现的做法。 

---
# Design Insights and Comparative Evaluation of a Hardware-Based Cooperative Perception Architecture for Lane Change Prediction 

**Title (ZH)**: 基于硬件的合作感知架构在变道预测中的设计洞察与比较评价 

**Authors**: Mohamed Manzour, Catherine M. Elias, Omar M. Shehata, Rubén Izquierdo, Miguel Ángel Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2509.20218)  

**Abstract**: Research on lane change prediction has gained attention in the last few years. Most existing works in this area have been conducted in simulation environments or with pre-recorded datasets, these works often rely on simplified assumptions about sensing, communication, and traffic behavior that do not always hold in practice. Real-world deployments of lane-change prediction systems are relatively rare, and when they are reported, the practical challenges, limitations, and lessons learned are often under-documented. This study explores cooperative lane-change prediction through a real hardware deployment in mixed traffic and shares the insights that emerged during implementation and testing. We highlight the practical challenges we faced, including bottlenecks, reliability issues, and operational constraints that shaped the behavior of the system. By documenting these experiences, the study provides guidance for others working on similar pipelines. 

**Abstract (ZH)**: 基于真实硬件在混合交通中探索协作变道预测及其实施经验 

---
# Steerable Adversarial Scenario Generation through Test-Time Preference Alignment 

**Title (ZH)**: 基于测试时偏好对齐的可引导对抗场景生成 

**Authors**: Tong Nie, Yuewen Mei, Yihong Tang, Junlin He, Jie Sun, Haotian Shi, Wei Ma, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.20102)  

**Abstract**: Adversarial scenario generation is a cost-effective approach for safety assessment of autonomous driving systems. However, existing methods are often constrained to a single, fixed trade-off between competing objectives such as adversariality and realism. This yields behavior-specific models that cannot be steered at inference time, lacking the efficiency and flexibility to generate tailored scenarios for diverse training and testing requirements. In view of this, we reframe the task of adversarial scenario generation as a multi-objective preference alignment problem and introduce a new framework named \textbf{S}teerable \textbf{A}dversarial scenario \textbf{GE}nerator (SAGE). SAGE enables fine-grained test-time control over the trade-off between adversariality and realism without any retraining. We first propose hierarchical group-based preference optimization, a data-efficient offline alignment method that learns to balance competing objectives by decoupling hard feasibility constraints from soft preferences. Instead of training a fixed model, SAGE fine-tunes two experts on opposing preferences and constructs a continuous spectrum of policies at inference time by linearly interpolating their weights. We provide theoretical justification for this framework through the lens of linear mode connectivity. Extensive experiments demonstrate that SAGE not only generates scenarios with a superior balance of adversariality and realism but also enables more effective closed-loop training of driving policies. Project page: this https URL. 

**Abstract (ZH)**: 对抗场景生成是自主驾驶系统安全性评估的一项成本效益高的方法。然而，现有方法往往局限于在抗对比性和现实性之间单一直接的权衡中。这会导致行为特定的模型，在推理时无法调整，缺乏生成适应多样化训练和测试需求的定制化场景的效率和灵活性。为了解决这一问题，我们将对抗场景生成任务重新定义为一个多目标偏好对齐问题，并提出了一种名为SAGE（Steerable Adversarial Scenario GEnerator）的新框架。SAGE允许在推理时对抗性和现实性之间的权衡进行细腻的控制，无需重新训练。我们首先提出了分层组别偏好优化方法，这是一种数据高效的离线对齐方法，通过解耦硬可行性约束与软偏好来学习平衡竞争目标。SAGE在推理时通过线性插值两个专家的权重来构建连续的策略谱，而不是训练固定模型。我们通过线性模式可连接性的视角提供了该框架的理论依据。大量实验表明，SAGE不仅生成具有优越对抗性和现实性平衡的场景，还能促进驾驶策略的更有效闭环训练。项目页面：this https URL。 

---
# Evaluation-Aware Reinforcement Learning 

**Title (ZH)**: 评估导向的强化学习 

**Authors**: Shripad Vilasrao Deshmukh, Will Schwarzer, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2509.19464)  

**Abstract**: Policy evaluation is often a prerequisite for deploying safety- and performance-critical systems. Existing evaluation approaches frequently suffer from high variance due to limited data and long-horizon tasks, or high bias due to unequal support or inaccurate environmental models. We posit that these challenges arise, in part, from the standard reinforcement learning (RL) paradigm of policy learning without explicit consideration of evaluation. As an alternative, we propose evaluation-aware reinforcement learning (EvA-RL), in which a policy is trained to maximize expected return while simultaneously minimizing expected evaluation error under a given value prediction scheme -- in other words, being "easy" to evaluate. We formalize a framework for EvA-RL and design an instantiation that enables accurate policy evaluation, conditioned on a small number of rollouts in an assessment environment that can be different than the deployment environment. However, our theoretical analysis and empirical results show that there is often a tradeoff between evaluation accuracy and policy performance when using a fixed value-prediction scheme within EvA-RL. To mitigate this tradeoff, we extend our approach to co-learn an assessment-conditioned state-value predictor alongside the policy. Empirical results across diverse discrete and continuous action domains demonstrate that EvA-RL can substantially reduce evaluation error while maintaining competitive returns. This work lays the foundation for a broad new class of RL methods that treat reliable evaluation as a first-class principle during training. 

**Abstract (ZH)**: 评价意识强化学习：在评价约束下最大化回报 

---
# Affective Computing and Emotional Data: Challenges and Implications in Privacy Regulations, The AI Act, and Ethics in Large Language Models 

**Title (ZH)**: 情感计算与情绪数据：在《AI法案》、隐私法规及大型语言模型伦理中的挑战与影响 

**Authors**: Nicola Fabiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.20153)  

**Abstract**: This paper examines the integration of emotional intelligence into artificial intelligence systems, with a focus on affective computing and the growing capabilities of Large Language Models (LLMs), such as ChatGPT and Claude, to recognize and respond to human emotions. Drawing on interdisciplinary research that combines computer science, psychology, and neuroscience, the study analyzes foundational neural architectures - CNNs for processing facial expressions and RNNs for sequential data, such as speech and text - that enable emotion recognition. It examines the transformation of human emotional experiences into structured emotional data, addressing the distinction between explicit emotional data collected with informed consent in research settings and implicit data gathered passively through everyday digital interactions. That raises critical concerns about lawful processing, AI transparency, and individual autonomy over emotional expressions in digital environments. The paper explores implications across various domains, including healthcare, education, and customer service, while addressing challenges of cultural variations in emotional expression and potential biases in emotion recognition systems across different demographic groups. From a regulatory perspective, the paper examines emotional data in the context of the GDPR and the EU AI Act frameworks, highlighting how emotional data may be considered sensitive personal data that requires robust safeguards, including purpose limitation, data minimization, and meaningful consent mechanisms. 

**Abstract (ZH)**: 本文研究情感 inteligence 与人工智能系统的集成，重点关注情感计算以及大型语言模型（如 ChatGPT 和 Claude）识别和响应人类情感的能力。通过结合计算机科学、心理学和神经科学的跨学科研究，该研究分析了用于处理面部表情的卷积神经网络（CNNs）和用于序列数据（如语音和文本）的递归神经网络（RNNs）等基础神经架构，以实现情感识别。文章探讨了将人类情感体验转化为结构化情感数据的过程，讨论了明示情感数据（在研究环境中通过知情同意收集）与潜在情感数据（通过日常数字互动被动收集）之间的区别。这引起了关于法律处理、AI 透明度和个体在数字环境中对情感表达的自主权的关切。文章探讨了情感智能在医疗保健、教育和客户服务等领域的影响，同时考虑不同文化背景下的情感表达差异以及不同地理人口群体中情感识别系统潜在偏见的挑战。从监管角度来看，文章在GDPR和欧盟AI法案的框架下探讨情感数据，强调情感数据可能被视为敏感个人数据，需要包括目的限制、数据最小化和有意义的同意机制在内的严格保护措施。 

---
# An effective control of large systems of active particles: An application to evacuation problem 

**Title (ZH)**: 大型活性粒子系统的有效控制：以 evacuation 问题为例 

**Authors**: Albina Klepach, Egor E. Nuzhin, Alexey A. Tsukanov, Nikolay V. Brilliantov  

**Link**: [PDF](https://arxiv.org/pdf/2509.19972)  

**Abstract**: Manipulation of large systems of active particles is a serious challenge across diverse domains, including crowd management, control of robotic swarms, and coordinated material transport. The development of advanced control strategies for complex scenarios is hindered, however, by the lack of scalability and robustness of the existing methods, in particular, due to the need of an individual control for each agent. One possible solution involves controlling a system through a leader or a group of leaders, which other agents tend to follow. Using such an approach we develop an effective control strategy for a leader, combining reinforcement learning (RL) with artificial forces acting on the system. To describe the guidance of active particles by a leader we introduce the generalized Vicsek model. This novel method is then applied to the problem of the effective evacuation by a robot-rescuer (leader) of large groups of people from hazardous places. We demonstrate, that while a straightforward application of RL yields suboptimal results, even for advanced architectures, our approach provides a robust and efficient evacuation strategy. The source code supporting this study is publicly available at: this https URL. 

**Abstract (ZH)**: 大规模活性粒子系统的操控在 crowd management、robotic swarms 控制和协调物质运输等领域是一个严重挑战。现有的方法由于缺乏可扩展性和鲁棒性，特别是在需要为每个代理单独控制时，限制了复杂场景下先进控制策略的发展。一种可能的解决方案是通过领导者或一组领导者操控系统，其他代理倾向于跟随领导者。利用这种方法，我们结合强化学习（RL）和作用于系统的虚拟力，开发出一种有效的领导者控制策略。为了描述领导者对活性粒子的引导，我们引入了广义 Vicsek 模型。然后，我们将该新型方法应用于机器人救援者（领导者）有效疏散大量人群远离危险场所的问题。我们证明，尽管直接应用 RL 能力有限，甚至对于先进的架构也是如此，我们的方法提供了更为稳健和高效的疏散策略。支持本研究的源代码可在以下网址获取：this https URL。 

---
# RDAR: Reward-Driven Agent Relevance Estimation for Autonomous Driving 

**Title (ZH)**: RDAR：基于奖励驱动的代理相关性估计在自动驾驶中的应用 

**Authors**: Carlo Bosio, Greg Woelki, Noureldin Hendy, Nicholas Roy, Byungsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.19789)  

**Abstract**: Human drivers focus only on a handful of agents at any one time. On the other hand, autonomous driving systems process complex scenes with numerous agents, regardless of whether they are pedestrians on a crosswalk or vehicles parked on the side of the road. While attention mechanisms offer an implicit way to reduce the input to the elements that affect decisions, existing attention mechanisms for capturing agent interactions are quadratic, and generally computationally expensive. We propose RDAR, a strategy to learn per-agent relevance -- how much each agent influences the behavior of the controlled vehicle -- by identifying which agents can be excluded from the input to a pre-trained behavior model. We formulate the masking procedure as a Markov Decision Process where the action consists of a binary mask indicating agent selection. We evaluate RDAR on a large-scale driving dataset, and demonstrate its ability to learn an accurate numerical measure of relevance by achieving comparable driving performance, in terms of overall progress, safety and performance, while processing significantly fewer agents compared to a state of the art behavior model. 

**Abstract (ZH)**: 人类驾驶员同时专注于少数几个代理对象，而自动驾驶系统需要处理大量代理对象，无论它们是人行横道上的行人还是路边停放的车辆。尽管注意力机制能够隐式地减少影响决策的输入元素，现有的用于捕捉代理交互的注意力机制通常是二次的，且计算成本高昂。我们提出了一种RDAR策略，通过确定可以被排除在预训练行为模型输入之外的代理对象，来学习每个代理对象的相关性——即每个代理对象对控制车辆行为的影响程度。我们将掩码过程形式化为马尔科夫决策过程，其中动作由指示代理选择的二元掩码组成。我们在大规模驾驶数据集上评估了RDAR，并展示了其能够在显著减少处理代理对象数量的情况下，实现与最先进的行为模型相当的驾驶性能，包括总体进度、安全性和表现。 

---
# DAWM: Diffusion Action World Models for Offline Reinforcement Learning via Action-Inferred Transitions 

**Title (ZH)**: DAWM：基于动作推断过渡的离线强化学习扩散动作世界模型 

**Authors**: Zongyue Li, Xiao Han, Yusong Li, Niklas Strauss, Matthias Schubert  

**Link**: [PDF](https://arxiv.org/pdf/2509.19538)  

**Abstract**: Diffusion-based world models have demonstrated strong capabilities in synthesizing realistic long-horizon trajectories for offline reinforcement learning (RL). However, many existing methods do not directly generate actions alongside states and rewards, limiting their compatibility with standard value-based offline RL algorithms that rely on one-step temporal difference (TD) learning. While prior work has explored joint modeling of states, rewards, and actions to address this issue, such formulations often lead to increased training complexity and reduced performance in practice. We propose \textbf{DAWM}, a diffusion-based world model that generates future state-reward trajectories conditioned on the current state, action, and return-to-go, paired with an inverse dynamics model (IDM) for efficient action inference. This modular design produces complete synthetic transitions suitable for one-step TD-based offline RL, enabling effective and computationally efficient training. Empirically, we show that conservative offline RL algorithms such as TD3BC and IQL benefit significantly from training on these augmented trajectories, consistently outperforming prior diffusion-based baselines across multiple tasks in the D4RL benchmark. 

**Abstract (ZH)**: 基于扩散的世界模型在离线强化学习（RL）中展示了强大的能力，能够生成具有高度现实感的长期轨迹。然而，许多现有方法没有直接生成动作和状态、奖励，限制了它们与依赖一-step 时差（TD）学习的标准值基离线RL算法的兼容性。虽然先前的工作探索了联合建模状态、奖励和动作以解决这一问题，但这样的建模往往会导致训练复杂度增加并在实际中表现不佳。我们提出了一种基于扩散的世界模型 \textbf{DAWM}，该模型在给定当前状态、动作和未来回报的情况下生成未来的状态-奖励轨迹，并配有一个逆动力学模型（IDM）以实现高效的动作推断。这种模块化设计可以生成适用于一-step TD 基础离线 RL 的完整合成转换，从而实现高效且计算成本低的训练。实验上，我们展示了保守的离线 RL 算法（如TD3BC和IQL）极大地受益于使用这些增强的轨迹进行训练，在 D4RL 基准上的多个任务中，这些算法在所有情况下都优于先前的基于扩散的方法。 

---
# Meow: End-to-End Outline Writing for Automatic Academic Survey 

**Title (ZH)**: Meow: 自动学术调研提纲生成 

**Authors**: Zhaoyu Ma, Yuan Shan, Jiahao Zhao, Nan Xu, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19370)  

**Abstract**: As academic paper publication numbers grow exponentially, conducting in-depth surveys with LLMs automatically has become an inevitable trend. Outline writing, which aims to systematically organize related works, is critical for automated survey generation. Yet existing automatic survey methods treat outline writing as mere workflow steps in the overall pipeline. Such template-based workflows produce outlines that lack in-depth understanding of the survey topic and fine-grained styles. To address these limitations, we propose Meow, the first metadata-driven outline writing framework that produces organized and faithful outlines efficiently. Specifically, we first formulate outline writing as an end-to-end task that generates hierarchical structured outlines from paper metadata. We then curate a high-quality dataset of surveys from arXiv, bioRxiv, and medRxiv, and establish systematic evaluation metrics for outline quality assessment. Finally, we employ a two-stage training approach combining supervised fine-tuning and reinforcement learning. Our 8B reasoning model demonstrates strong performance with high structural fidelity and stylistic coherence. 

**Abstract (ZH)**: 随着学术论文发表数量呈指数增长，自动使用大规模语言模型进行深入调查已成为一种不可避免的趋势。面向元数据的提纲写作框架Meow：高效生成组织化和忠实提纲 

---
