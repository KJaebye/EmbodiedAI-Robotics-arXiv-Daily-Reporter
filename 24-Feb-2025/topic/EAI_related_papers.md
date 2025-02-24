# BOSS: Benchmark for Observation Space Shift in Long-Horizon Task 

**Title (ZH)**: BOSS: 长时间 horizon 任务中观测空间变化的基准 

**Authors**: Yue Yang, Linfeng Zhao, Mingyu Ding, Gedas Bertasius, Daniel Szafir  

**Link**: [PDF](https://arxiv.org/pdf/2502.15679)  

**Abstract**: Robotics has long sought to develop visual-servoing robots capable of completing previously unseen long-horizon tasks. Hierarchical approaches offer a pathway for achieving this goal by executing skill combinations arranged by a task planner, with each visuomotor skill pre-trained using a specific imitation learning (IL) algorithm. However, even in simple long-horizon tasks like skill chaining, hierarchical approaches often struggle due to a problem we identify as Observation Space Shift (OSS), where the sequential execution of preceding skills causes shifts in the observation space, disrupting the performance of subsequent individually trained skill policies. To validate OSS and evaluate its impact on long-horizon tasks, we introduce BOSS (a Benchmark for Observation Space Shift). BOSS comprises three distinct challenges: "Single Predicate Shift", "Accumulated Predicate Shift", and "Skill Chaining", each designed to assess a different aspect of OSS's negative effect. We evaluated several recent popular IL algorithms on BOSS, including three Behavioral Cloning methods and the Visual Language Action model OpenVLA. Even on the simplest challenge, we observed average performance drops of 67%, 35%, 34%, and 54%, respectively, when comparing skill performance with and without OSS. Additionally, we investigate a potential solution to OSS that scales up the training data for each skill with a larger and more visually diverse set of demonstrations, with our results showing it is not sufficient to resolve OSS. The project page is: this https URL 

**Abstract (ZH)**: 机器人学致力于开发能够完成未见过的长期任务的视觉伺服机器人。分层方法通过由任务规划器安排技能组合来提供实现这一目标的途径，每种视觉运动技能都使用特定的imitation learning (IL) 算法进行预先训练。然而，即便是在如技能链接这样简单的长期任务中，分层方法也常常因我们识别出的一种问题——观察空间偏移（OSS）——而挣扎，这种问题会导致后续技能执行时观察空间出现偏移，破坏每个单独训练的技能策略的表现。为了验证OSS并评估其对长期任务的影响，我们引入了BOSS（观察空间偏移基准）。BOSS包括三个不同的挑战：“单一谓词偏移”、“累积谓词偏移”和“技能链接”，旨在评估OSS负面影响的不同方面。我们对BOSS评估了几个最近流行的IL算法，包括三种行为克隆方法和视觉语言动作模型OpenVLA。即使在最简单的挑战中，我们观察到，在有和没有OSS的情况下，技能表现的平均性能分别下降了67%、35%、34%和54%。此外，我们研究了OSS的一种可能解决方法，即通过使用更大且视觉多样更多的示范数据来放大每种技能的训练数据，结果表明这不足以解决OSS问题。项目页面：https://github.com/alibabaqwen/BOSS。 

---
# A Simulation Pipeline to Facilitate Real-World Robotic Reinforcement Learning Applications 

**Title (ZH)**: 一种促进现实机器人强化学习应用的仿真管道 

**Authors**: Jefferson Silveira, Joshua A. Marshall, Sidney N. Givigi Jr  

**Link**: [PDF](https://arxiv.org/pdf/2502.15649)  

**Abstract**: Reinforcement learning (RL) has gained traction for its success in solving complex tasks for robotic applications. However, its deployment on physical robots remains challenging due to safety risks and the comparatively high costs of training. To avoid these problems, RL agents are often trained on simulators, which introduces a new problem related to the gap between simulation and reality. This paper presents an RL pipeline designed to help reduce the reality gap and facilitate developing and deploying RL policies for real-world robotic systems. The pipeline organizes the RL training process into an initial step for system identification and three training stages: core simulation training, high-fidelity simulation, and real-world deployment, each adding levels of realism to reduce the sim-to-real gap. Each training stage takes an input policy, improves it, and either passes the improved policy to the next stage or loops it back for further improvement. This iterative process continues until the policy achieves the desired performance. The pipeline's effectiveness is shown through a case study with the Boston Dynamics Spot mobile robot used in a surveillance application. The case study presents the steps taken at each pipeline stage to obtain an RL agent to control the robot's position and orientation. 

**Abstract (ZH)**: 强化学习（RL）因其在解决机器人应用中的复杂任务方面的成功而受到关注。然而，其在物理机器人上的部署仍然面临着安全风险和相对较高的训练成本的挑战。为避免这些问题，RL代理通常在模拟器中训练，这引入了模拟与现实之间差距的新问题。本文介绍了一种RL管道，旨在减少这种现实差距，并促进在实地机器人系统中开发和部署RL策略。该管道将RL训练过程组织为初始步骤进行系统识别和三个训练阶段：核心仿真训练、高保真仿真和实地部署，每个阶段都增加了现实感以减少仿真到现实的差距。每个训练阶段都以输入策略为输入，对其进行改进，并将改进后的策略传递到下一个阶段或反馈回进行进一步改进。这一迭代过程将持续进行，直到策略达到所需的性能。通过使用波士顿动力公司Spot移动机器人在监视应用中的案例研究，展示了该管道的有效性。该案例研究介绍了每个管道阶段所采取的步骤，以获得控制机器人位置和方向的RL代理。 

---
# Reduced-Order Model Guided Contact-Implicit Model Predictive Control for Humanoid Locomotion 

**Title (ZH)**: 基于降阶模型引导的接触显式模型预测控制的人形步行控制 

**Authors**: Sergio A. Esteban, Vince Kurtz, Adrian B. Ghansah, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2502.15630)  

**Abstract**: Humanoid robots have great potential for real-world applications due to their ability to operate in environments built for humans, but their deployment is hindered by the challenge of controlling their underlying high-dimensional nonlinear hybrid dynamics. While reduced-order models like the Hybrid Linear Inverted Pendulum (HLIP) are simple and computationally efficient, they lose whole-body expressiveness. Meanwhile, recent advances in Contact-Implicit Model Predictive Control (CI-MPC) enable robots to plan through multiple hybrid contact modes, but remain vulnerable to local minima and require significant tuning. We propose a control framework that combines the strengths of HLIP and CI-MPC. The reduced-order model generates a nominal gait, while CI-MPC manages the whole-body dynamics and modifies the contact schedule as needed. We demonstrate the effectiveness of this approach in simulation with a novel 24 degree-of-freedom humanoid robot: Achilles. Our proposed framework achieves rough terrain walking, disturbance recovery, robustness under model and state uncertainty, and allows the robot to interact with obstacles in the environment, all while running online in real-time at 50 Hz. 

**Abstract (ZH)**: 人形机器人由于能够在为人类构建的环境中操作而具有广泛的应用潜力，但其部署受到控制其潜在高维非线性混合动力学挑战的阻碍。虽然简化模型如混合线性倒摆（HLIP）简单且计算效率高，但会失去全身表达性。同时，最近在接触隐式模型预测控制（CI-MPC）方面的进展使得机器人能够计划通过多种混合接触模式，但仍容易陷入局部极值，并需要大量调整。我们提出了一种结合HLIP和CI-MPC优点的控制框架。简化模型生成名义步态，而CI-MPC管理全身动力学并在必要时修改接触表调度。我们通过一个新的24自由度人形机器人Achilles在仿真中展示了该方法的有效性，实现崎岖地形行走、干扰恢复、在模型和状态不确定性下的鲁棒性，并允许机器人与环境中的障碍物互动，同时以50 Hz的在线实时速度运行。 

---
# Pick-and-place Manipulation Across Grippers Without Retraining: A Learning-optimization Diffusion Policy Approach 

**Title (ZH)**: 无需重新训练的手爪之间拾取放置操作：一种学习-优化扩散策略方法 

**Authors**: Xiangtong Yao, Yirui Zhou, Yuan Meng, Liangyu Dong, Lin Hong, Zitao Zhang, Zhenshan Bing, Kai Huang, Fuchun Sun, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2502.15613)  

**Abstract**: Current robotic pick-and-place policies typically require consistent gripper configurations across training and inference. This constraint imposes high retraining or fine-tuning costs, especially for imitation learning-based approaches, when adapting to new end-effectors. To mitigate this issue, we present a diffusion-based policy with a hybrid learning-optimization framework, enabling zero-shot adaptation to novel grippers without additional data collection for retraining policy. During training, the policy learns manipulation primitives from demonstrations collected using a base gripper. At inference, a diffusion-based optimization strategy dynamically enforces kinematic and safety constraints, ensuring that generated trajectories align with the physical properties of unseen grippers. This is achieved through a constrained denoising procedure that adapts trajectories to gripper-specific parameters (e.g., tool-center-point offsets, jaw widths) while preserving collision avoidance and task feasibility. We validate our method on a Franka Panda robot across six gripper configurations, including 3D-printed fingertips, flexible silicone gripper, and Robotiq 2F-85 gripper. Our approach achieves a 93.3% average task success rate across grippers (vs. 23.3-26.7% for diffusion policy baselines), supporting tool-center-point variations of 16-23.5 cm and jaw widths of 7.5-11.5 cm. The results demonstrate that constrained diffusion enables robust cross-gripper manipulation while maintaining the sample efficiency of imitation learning, eliminating the need for gripper-specific retraining. Video and code are available at this https URL. 

**Abstract (ZH)**: 基于扩散的零 shot 夹持器适应策略：混合学习-优化框架 

---
# Learning Long-Horizon Robot Manipulation Skills via Privileged Action 

**Title (ZH)**: 通过优先级动作学习长时_horizon机器人操作技能 

**Authors**: Xiaofeng Mao, Yucheng Xu, Zhaole Sun, Elle Miller, Daniel Layeghi, Michael Mistry  

**Link**: [PDF](https://arxiv.org/pdf/2502.15442)  

**Abstract**: Long-horizon contact-rich tasks are challenging to learn with reinforcement learning, due to ineffective exploration of high-dimensional state spaces with sparse rewards. The learning process often gets stuck in local optimum and demands task-specific reward fine-tuning for complex scenarios. In this work, we propose a structured framework that leverages privileged actions with curriculum learning, enabling the policy to efficiently acquire long-horizon skills without relying on extensive reward engineering or reference trajectories. Specifically, we use privileged actions in simulation with a general training procedure that would be infeasible to implement in real-world scenarios. These privileges include relaxed constraints and virtual forces that enhance interaction and exploration with objects. Our results successfully achieve complex multi-stage long-horizon tasks that naturally combine non-prehensile manipulation with grasping to lift objects from non-graspable poses. We demonstrate generality by maintaining a parsimonious reward structure and showing convergence to diverse and robust behaviors across various environments. Additionally, real-world experiments further confirm that the skills acquired using our approach are transferable to real-world environments, exhibiting robust and intricate performance. Our approach outperforms state-of-the-art methods in these tasks, converging to solutions where others fail. 

**Abstract (ZH)**: 长时域高接触任务使用强化学习学习具有稀疏奖励的高维状态空间探索不足，往往容易陷入局部最优，并需要针对复杂场景进行特定的任务奖励微调。本文提出了一种结构化框架，结合先验动作与 curriculum learning，使策略能够高效地获得长时域技能，无需依赖广泛的奖励工程或参考轨迹。具体而言，我们利用模拟中的先验动作和通用训练程序，这些程序在实际场景中难以实施。这些先验特权包括宽松的约束和虚拟力，以增强与物体的交互和探索。我们的实验结果成功实现了结合非抓取操作与抓取的复杂多阶段长时域任务，将物体从不可抓取的姿态提起。通过保持简洁的奖励结构并展示多种环境下的收敛和稳健行为，我们展示了通用性。此外，现实世界实验进一步证实，使用我们方法获得的技能在实际环境中的可迁移性，表现出稳健和复杂的性能。我们的方法在这些任务中优于现有最佳方法，能够收敛到其他方法失败的解决方案。 

---
# Exploring Embodied Multimodal Large Models: Development, Datasets, and Future Directions 

**Title (ZH)**: 探索具身多模态大型模型：发展、数据集及未来方向 

**Authors**: Shoubin Chen, Zehao Wu, Kai Zhang, Chunyu Li, Baiyang Zhang, Fei Ma, Fei Richard Yu, Qingquan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.15336)  

**Abstract**: Embodied multimodal large models (EMLMs) have gained significant attention in recent years due to their potential to bridge the gap between perception, cognition, and action in complex, real-world environments. This comprehensive review explores the development of such models, including Large Language Models (LLMs), Large Vision Models (LVMs), and other models, while also examining other emerging architectures. We discuss the evolution of EMLMs, with a focus on embodied perception, navigation, interaction, and simulation. Furthermore, the review provides a detailed analysis of the datasets used for training and evaluating these models, highlighting the importance of diverse, high-quality data for effective learning. The paper also identifies key challenges faced by EMLMs, including issues of scalability, generalization, and real-time decision-making. Finally, we outline future directions, emphasizing the integration of multimodal sensing, reasoning, and action to advance the development of increasingly autonomous systems. By providing an in-depth analysis of state-of-the-art methods and identifying critical gaps, this paper aims to inspire future advancements in EMLMs and their applications across diverse domains. 

**Abstract (ZH)**: 沉浸式多模态大型模型（EMLMs）因潜在地弥合感知、认知和行动之间的差距而在复杂的真实环境中受到了广泛关注。本文综述了此类模型的发展，包括大型语言模型（LLMs）、大型视觉模型（LVMs）及其他模型，同时也探讨了其他新兴架构。本文讨论了EMLMs的发展演变，重点关注沉浸式感知、导航、交互和模拟。此外，综述还详细分析了用于训练和评估这些模型的数据集，突出了多元化和高质量数据对于有效学习的重要性。该论文还指出了EMLMs面临的几个关键挑战，包括可扩展性、泛化能力和实时决策问题。最后，本文提出了未来方向，强调了将多模态感知、推理和行动的整合以推进更具自主性的系统的发展。通过深入分析最先进的方法并识别关键缺口，本文旨在启发未来在EMLMs及其跨多个领域应用方面的进展。 

---
# DDAT: Diffusion Policies Enforcing Dynamically Admissible Robot Trajectories 

**Title (ZH)**: DDAT: 扩散策略约束动态可接受的机器人轨迹 

**Authors**: Jean-Baptiste Bouvier, Kanghyun Ryu, Kartik Nagpal, Qiayuan Liao, Koushil Sreenath, Negar Mehr  

**Link**: [PDF](https://arxiv.org/pdf/2502.15043)  

**Abstract**: Diffusion models excel at creating images and videos thanks to their multimodal generative capabilities. These same capabilities have made diffusion models increasingly popular in robotics research, where they are used for generating robot motion. However, the stochastic nature of diffusion models is fundamentally at odds with the precise dynamical equations describing the feasible motion of robots. Hence, generating dynamically admissible robot trajectories is a challenge for diffusion models. To alleviate this issue, we introduce DDAT: Diffusion policies for Dynamically Admissible Trajectories to generate provably admissible trajectories of black-box robotic systems using diffusion models. A sequence of states is a dynamically admissible trajectory if each state of the sequence belongs to the reachable set of its predecessor by the robot's equations of motion. To generate such trajectories, our diffusion policies project their predictions onto a dynamically admissible manifold during both training and inference to align the objective of the denoiser neural network with the dynamical admissibility constraint. The auto-regressive nature of these projections along with the black-box nature of robot dynamics render these projections immensely challenging. We thus enforce admissibility by iteratively sampling a polytopic under-approximation of the reachable set of a state onto which we project its predicted successor, before iterating this process with the projected successor. By producing accurate trajectories, this projection eliminates the need for diffusion models to continually replan, enabling one-shot long-horizon trajectory planning. We demonstrate that our framework generates higher quality dynamically admissible robot trajectories through extensive simulations on a quadcopter and various MuJoCo environments, along with real-world experiments on a Unitree GO1 and GO2. 

**Abstract (ZH)**: Diffusion 模型通过其多模态生成能力在创建图像和视频方面表现出色。这些相同的生成能力使其在机器人研究中越来越受欢迎，其中扩散模型被用于生成机器人运动。然而，扩散模型的随机性质与描述机器人可行运动的精确动力学方程本质上存在冲突。因此，生成动态可接受的机器人轨迹是扩散模型的一个挑战。为了缓解这一问题，我们引入了 DDAT：扩散策略以生成黑盒机器人系统的可证明动态可接受轨迹。如果序列中的每个状态都由机器人动力学方程的可达集决定，则该状态序列是一个动态可接受轨迹。为了生成这样的轨迹，我们的扩散策略在训练和推断过程中将预测投影到动态可接受流形上，以使去噪神经网络的目标与动态可接受性约束一致。这些预测的自回归性质以及机器人动力学的黑盒性质使得这些投影极其具有挑战性。因此，我们通过迭代地将多面体下近似投影到一个状态的可达集上，并在其上面预测该状态的后继状态，然后迭代这个过程，以实现此过程中的投影后继状态，从而确保可接受性。通过生成准确的轨迹，这种投影消除了扩散模型不断重新规划的需要，从而实现一次性长时间轨迹规划。我们通过在四旋翼无人机和各种 MuJoCo 环境上的广泛仿真以及在 Unitree GO1 和 GO2 上的实地实验，证明了我们的框架能够生成更高质量的动态可接受机器人轨迹。 

---
# Bridging Text and Vision: A Multi-View Text-Vision Registration Approach for Cross-Modal Place Recognition 

**Title (ZH)**: 跨模态場景識別的多視點文本-視覺 Registration 方法 

**Authors**: Tianyi Shang, Zhenyu Li, Pengjie Xu, Jinwei Qiao, Gang Chen, Zihan Ruan, Weijun Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14195)  

**Abstract**: Mobile robots necessitate advanced natural language understanding capabilities to accurately identify locations and perform tasks such as package delivery. However, traditional visual place recognition (VPR) methods rely solely on single-view visual information and cannot interpret human language descriptions. To overcome this challenge, we bridge text and vision by proposing a multiview (360° views of the surroundings) text-vision registration approach called Text4VPR for place recognition task, which is the first method that exclusively utilizes textual descriptions to match a database of images. Text4VPR employs the frozen T5 language model to extract global textual embeddings. Additionally, it utilizes the Sinkhorn algorithm with temperature coefficient to assign local tokens to their respective clusters, thereby aggregating visual descriptors from images. During the training stage, Text4VPR emphasizes the alignment between individual text-image pairs for precise textual description. In the inference stage, Text4VPR uses the Cascaded Cross-Attention Cosine Alignment (CCCA) to address the internal mismatch between text and image groups. Subsequently, Text4VPR performs precisely place match based on the descriptions of text-image groups. On Street360Loc, the first text to image VPR dataset we created, Text4VPR builds a robust baseline, achieving a leading top-1 accuracy of 57% and a leading top-10 accuracy of 92% within a 5-meter radius on the test set, which indicates that localization from textual descriptions to images is not only feasible but also holds significant potential for further advancement, as shown in Figure 1. 

**Abstract (ZH)**: 基于多视角文本-视觉注册的自然语言理解在移动机器人位置识别中的应用：Text4VPR方法 

---
# WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents 

**Title (ZH)**: WorldCraft: 基于LLM代理的 PHOTO-真实感 3D 世界创建与定制 

**Authors**: Xinhang Liu, Chi-Keung Tang, Yu-Wing Tai  

**Link**: [PDF](https://arxiv.org/pdf/2502.15601)  

**Abstract**: Constructing photorealistic virtual worlds has applications across various fields, but it often requires the extensive labor of highly trained professionals to operate conventional 3D modeling software. To democratize this process, we introduce WorldCraft, a system where large language model (LLM) agents leverage procedural generation to create indoor and outdoor scenes populated with objects, allowing users to control individual object attributes and the scene layout using intuitive natural language commands. In our framework, a coordinator agent manages the overall process and works with two specialized LLM agents to complete the scene creation: ForgeIt, which integrates an ever-growing manual through auto-verification to enable precise customization of individual objects, and ArrangeIt, which formulates hierarchical optimization problems to achieve a layout that balances ergonomic and aesthetic considerations. Additionally, our pipeline incorporates a trajectory control agent, allowing users to animate the scene and operate the camera through natural language interactions. Our system is also compatible with off-the-shelf deep 3D generators to enrich scene assets. Through evaluations and comparisons with state-of-the-art methods, we demonstrate the versatility of WorldCraft, ranging from single-object customization to intricate, large-scale interior and exterior scene designs. This system empowers non-professionals to bring their creative visions to life. 

**Abstract (ZH)**: 基于 procedurally generated agents 的 WorldCraft：构建直观自然语言控制的 photorealistic 虚拟世界 

---
# Bridging vision language model (VLM) evaluation gaps with a framework for scalable and cost-effective benchmark generation 

**Title (ZH)**: 使用框架实现可扩展且成本有效的基准生成以弥合视觉语言模型评估差距 

**Authors**: Tim Rädsch, Leon Mayer, Simon Pavicic, A. Emre Kavur, Marcel Knopp, Barış Öztürk, Klaus Maier-Hein, Paul F. Jaeger, Fabian Isensee, Annika Reinke, Lena Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2502.15563)  

**Abstract**: Reliable evaluation of AI models is critical for scientific progress and practical application. While existing VLM benchmarks provide general insights into model capabilities, their heterogeneous designs and limited focus on a few imaging domains pose significant challenges for both cross-domain performance comparison and targeted domain-specific evaluation. To address this, we propose three key contributions: (1) a framework for the resource-efficient creation of domain-specific VLM benchmarks enabled by task augmentation for creating multiple diverse tasks from a single existing task, (2) the release of new VLM benchmarks for seven domains, created according to the same homogeneous protocol and including 162,946 thoroughly human-validated answers, and (3) an extensive benchmarking of 22 state-of-the-art VLMs on a total of 37,171 tasks, revealing performance variances across domains and tasks, thereby supporting the need for tailored VLM benchmarks. Adoption of our methodology will pave the way for the resource-efficient domain-specific selection of models and guide future research efforts toward addressing core open questions. 

**Abstract (ZH)**: 可靠的AI模型评估对于科学进步和实际应用至关重要。尽管现有的VLM基准提供了关于模型能力的一般洞察，但它们多样化的设计和对少数成像领域有限的关注，对跨域性能比较和特定领域评估提出了重大挑战。为此，我们提出了三个关键贡献：（1）一种通过任务增强创建单个现有任务衍生多种多样任务以实现资源高效创建领域特定VLM基准的框架；（2）推出针对七个领域的新VLM基准，根据相同的均匀协议创建，并包含162,946个全面的人工验证答案；（3）对22种最先进的VLM在总计37,171个任务上的广泛评估，揭示了不同领域和任务间的性能差异，从而支持定制化VLM基准的必要性。采用我们的方法将为资源高效选择模型铺平道路，并指导未来研究致力于解决核心开放问题。 

---
# Towards Physics-Guided Foundation Models 

**Title (ZH)**: 面向物理引导的基础模型 

**Authors**: Majid Farhadloo, Arun Sharma, Mingzhou Yang, Bharat Jayaprakash, William Northrop, Shashi Shekhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15013)  

**Abstract**: Traditional foundation models are pre-trained on broad datasets to reduce the training resources (e.g., time, energy, labeled samples) needed for fine-tuning a wide range of downstream tasks. However, traditional foundation models struggle with out-of-distribution prediction and can produce outputs that are unrealistic and physically infeasible. We propose the notation of physics-guided foundation models (PGFM), that is, foundation models integrated with broad or general domain (e.g., scientific) physical knowledge applicable to a wide range of downstream tasks. 

**Abstract (ZH)**: 基于物理引导的基础模型（PGFM）：整合广泛或通用领域物理知识的基础模型 

---
# Sce2DriveX: A Generalized MLLM Framework for Scene-to-Drive Learning 

**Title (ZH)**: Sce2DriveX：一种场景到驾驶学习的一般化多模态模型框架 

**Authors**: Rui Zhao, Qirui Yuan, Jinyu Li, Haofeng Hu, Yun Li, Chengyuan Zheng, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14917)  

**Abstract**: End-to-end autonomous driving, which directly maps raw sensor inputs to low-level vehicle controls, is an important part of Embodied AI. Despite successes in applying Multimodal Large Language Models (MLLMs) for high-level traffic scene semantic understanding, it remains challenging to effectively translate these conceptual semantics understandings into low-level motion control commands and achieve generalization and consensus in cross-scene driving. We introduce Sce2DriveX, a human-like driving chain-of-thought (CoT) reasoning MLLM framework. Sce2DriveX utilizes multimodal joint learning from local scene videos and global BEV maps to deeply understand long-range spatiotemporal relationships and road topology, enhancing its comprehensive perception and reasoning capabilities in 3D dynamic/static scenes and achieving driving generalization across scenes. Building on this, it reconstructs the implicit cognitive chain inherent in human driving, covering scene understanding, meta-action reasoning, behavior interpretation analysis, motion planning and control, thereby further bridging the gap between autonomous driving and human thought processes. To elevate model performance, we have developed the first extensive Visual Question Answering (VQA) driving instruction dataset tailored for 3D spatial understanding and long-axis task reasoning. Extensive experiments demonstrate that Sce2DriveX achieves state-of-the-art performance from scene understanding to end-to-end driving, as well as robust generalization on the CARLA Bench2Drive benchmark. 

**Abstract (ZH)**: 端到端自动驾驶：从传感器原始输入直接映射到低级车辆控制，是具身AI的重要组成部分。尽管在高层次交通场景语义理解中成功应用了多模态大型语言模型（MLLMs），但在有效地将这些概念性语义理解转换为低级运动控制命令，并在跨场景驾驶中实现泛化和共识方面仍面临挑战。我们引入了Sce2DriveX，这是一种类人的驾驶链式思考（CoT）推理MLLM框架。Sce2DriveX利用局部场景视频和全局BEV地图的多模态联合学习，深入理解远距离时空关系和道路拓扑，增强其在3D动态/静态场景中的综合感知和推理能力，实现跨场景驾驶的泛化。在此基础上，它重构了人类驾驶内在的隐式认知链条，涵盖场景理解、元动作推理、行为解释分析、运动规划和控制，从而进一步弥合自动驾驶与人类思维过程之间的差距。为了提升模型性能，我们开发了首个专门针对3D空间理解和长轴任务推理的视觉问答（VQA）驾驶指令数据集。广泛的实验表明，Sce2DriveX在从场景理解到端到端驾驶的性能上达到了最先进的水平，并在CARLA Bench2Drive基准测试中展现出鲁棒的泛化能力。 

---
# KOALA: Knowledge Conflict Augmentations for Robustness in Vision Language Models 

**Title (ZH)**: KOALA: 知识冲突增强for 视觉语言模型的鲁棒性 

**Authors**: Peter Carragher, Nikitha Rao, Abhinand Jha, R Raghav, Kathleen M. Carley  

**Link**: [PDF](https://arxiv.org/pdf/2502.14908)  

**Abstract**: The robustness of large language models (LLMs) against knowledge conflicts in unimodal question answering systems has been well studied. However, the effect of conflicts in information sources on vision language models (VLMs) in multimodal settings has not yet been explored. In this work, we propose \segsub, a framework that applies targeted perturbations to image sources to study and improve the robustness of VLMs against three different types of knowledge conflicts, namely parametric, source, and counterfactual conflicts. Contrary to prior findings that showed that LLMs are sensitive to parametric conflicts arising from textual perturbations, we find VLMs are largely robust to image perturbation. On the other hand, VLMs perform poorly on counterfactual examples (<30% accuracy) and fail to reason over source conflicts (<1% accuracy). We also find a link between hallucinations and image context, with GPT-4o prone to hallucination when presented with highly contextualized counterfactual examples. While challenges persist with source conflicts, finetuning models significantly improves reasoning over counterfactual samples. Our findings highlight the need for VLM training methodologies that enhance their reasoning capabilities, particularly in addressing complex knowledge conflicts between multimodal sources. 

**Abstract (ZH)**: 大语言模型（LLMs）在单模态问答系统中对抗知识冲突的稳健性已得到充分研究，但在多模态设置中信息源冲突对视觉语言模型（VLMs）的影响尚未被探索。在此工作中，我们提出了\segsub框架，通过对图像源应用目标化扰动来研究和改进VLMs在三种不同类型知识冲突（参数冲突、来源冲突和反事实冲突）下的稳健性。与先有发现不同，我们发现VLMs对来自图像扰动的参数冲突不敏感，表现出较大的稳健性。另一方面，VLMs在反事实示例上表现不佳（准确率<30%）并在来源冲突下几乎无法推理（准确率<1%）。我们还发现幻觉与图像上下文之间的关联，当GPT-4o面对高度上下文化的反事实示例时容易产生幻觉。虽然在来源冲突方面仍存在挑战，但模型微调显著提高了对反事实样本的推理能力。我们的发现强调了增强VLM推理能力的训练方法的需求，特别是在解决多模态来源之间的复杂知识冲突方面。 

---
