# TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models 

**Title (ZH)**: TA-VLA: 研究扭矩感知视觉-语言-行动模型的设计空间 

**Authors**: Zongzheng Zhang, Haobo Xu, Zhuo Yang, Chenghao Yue, Zehao Lin, Huan-ang Gao, Ziwei Wang, Hao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.07962)  

**Abstract**: Many robotic manipulation tasks require sensing and responding to force signals such as torque to assess whether the task has been successfully completed and to enable closed-loop control. However, current Vision-Language-Action (VLA) models lack the ability to integrate such subtle physical feedback. In this work, we explore Torque-aware VLA models, aiming to bridge this gap by systematically studying the design space for incorporating torque signals into existing VLA architectures. We identify and evaluate several strategies, leading to three key findings. First, introducing torque adapters into the decoder consistently outperforms inserting them into the this http URL, inspired by joint prediction and planning paradigms in autonomous driving, we propose predicting torque as an auxiliary output, which further improves performance. This strategy encourages the model to build a physically grounded internal representation of interaction dynamics. Extensive quantitative and qualitative experiments across contact-rich manipulation benchmarks validate our findings. 

**Abstract (ZH)**: 许多机器人操作任务需要感知和响应如扭矩等力信号，以评估任务是否成功完成，并实现闭环控制。然而，当前的视觉-语言-动作（VLA）模型缺乏整合此类微妙物理反馈的能力。在本文中，我们探索了扭矩感知的VLA模型，旨在通过系统研究将扭矩信号整合到现有VLA架构中的设计空间来弥合这一差距。我们识别并评估了几种策略，得到了三个关键发现。首先，将扭矩适配器引入解码器始终优于将其插入编码器或其他位置。受自主驾驶中的联合预测和规划范式的启发，我们提出了预测扭矩作为辅助输出的策略，这进一步提高了性能。这种策略促使模型构建一个与物理相互作用动力学相联系的内部表示。在接触丰富的操作基准上的广泛定量和定性实验中验证了我们的发现。 

---
# Graph-Fused Vision-Language-Action for Policy Reasoning in Multi-Arm Robotic Manipulation 

**Title (ZH)**: 基于图融合的视觉-语言-动作政策推理在多臂机器人操作中 

**Authors**: Shunlei Li, Longsen Gao, Jiuwen Cao, Yingbai Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07957)  

**Abstract**: Acquiring dexterous robotic skills from human video demonstrations remains a significant challenge, largely due to conventional reliance on low-level trajectory replication, which often fails to generalize across varying objects, spatial layouts, and manipulator configurations. To address this limitation, we introduce Graph-Fused Vision-Language-Action (GF-VLA), a unified framework that enables dual-arm robotic systems to perform task-level reasoning and execution directly from RGB-D human demonstrations. GF-VLA employs an information-theoretic approach to extract task-relevant cues, selectively highlighting critical hand-object and object-object interactions. These cues are structured into temporally ordered scene graphs, which are subsequently integrated with a language-conditioned transformer to produce hierarchical behavior trees and interpretable Cartesian motion primitives. To enhance efficiency in bimanual execution, we propose a cross-arm allocation strategy that autonomously determines gripper assignment without requiring explicit geometric modeling. We validate GF-VLA on four dual-arm block assembly benchmarks involving symbolic structure construction and spatial generalization. Empirical results demonstrate that the proposed representation achieves over 95% graph accuracy and 93% subtask segmentation, enabling the language-action planner to generate robust, interpretable task policies. When deployed on a dual-arm robot, these policies attain 94% grasp reliability, 89% placement accuracy, and 90% overall task success across stacking, letter-formation, and geometric reconfiguration tasks, evidencing strong generalization and robustness under diverse spatial and semantic variations. 

**Abstract (ZH)**: 从人类视频示范中获取灵巧的机器人技能仍然是一个重大挑战，主要原因是传统上依赖于低级轨迹复制，这往往无法在不同对象、空间布局和操作器配置之间进行泛化。为了解决这一局限性，我们引入了图融合视觉-语言-动作（GF-VLA）统一框架，该框架使双臂机器人系统能够直接从RGB-D人类示范中进行任务级推理和执行。GF-VLA采用信息论方法提取与任务相关的信息，选择性地突出关键的手-物和物-物交互。这些信息被结构化为时间有序的场景图，随后与语言条件下的变换器结合，生成分层的行为树和可解释的笛卡尔运动基元。为了提高双臂执行的效率，我们提出了一种跨臂分配策略，该策略能够自主确定夹持器分配，而不需显式几何建模。我们在四个涉及符号结构构建和空间泛化的双臂积木装配基准中验证了GF-VLA。实验结果表明，所提出的表示方法在图准确度和子任务分割上分别达到了95%和93%，使语言-动作规划者能够生成稳健且可解释的任务策略。当部署在双臂机器人上时，这些策略在堆叠、字母形成和几何重构任务中分别实现了94%的抓取可靠性、89%的放置精度和90%的整体任务成功率，证明了其在多样空间和语义变化下的强泛化能力和鲁棒性。 

---
# RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction 

**Title (ZH)**: RaC: 机器人学习在扩展恢复与修正能力下的长时_horizon_任务学习 

**Authors**: Zheyuan Hu, Robyn Wu, Naveen Enock, Jasmine Li, Riya Kadakia, Zackory Erickson, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.07953)  

**Abstract**: Modern paradigms for robot imitation train expressive policy architectures on large amounts of human demonstration data. Yet performance on contact-rich, deformable-object, and long-horizon tasks plateau far below perfect execution, even with thousands of expert demonstrations. This is due to the inefficiency of existing ``expert'' data collection procedures based on human teleoperation. To address this issue, we introduce RaC, a new phase of training on human-in-the-loop rollouts after imitation learning pre-training. In RaC, we fine-tune a robotic policy on human intervention trajectories that illustrate recovery and correction behaviors. Specifically, during a policy rollout, human operators intervene when failure appears imminent, first rewinding the robot back to a familiar, in-distribution state and then providing a corrective segment that completes the current sub-task. Training on this data composition expands the robotic skill repertoire to include retry and adaptation behaviors, which we show are crucial for boosting both efficiency and robustness on long-horizon tasks. Across three real-world bimanual control tasks: shirt hanging, airtight container lid sealing, takeout box packing, and a simulated assembly task, RaC outperforms the prior state-of-the-art using 10$\times$ less data collection time and samples. We also show that RaC enables test-time scaling: the performance of the trained RaC policy scales linearly in the number of recovery maneuvers it exhibits. Videos of the learned policy are available at this https URL. 

**Abstract (ZH)**: 基于人类在环的回放训练：提升交互丰富、可变形物体处理及长时 horizon 任务的机器人模仿学习性能 

---
# Temporal Counterfactual Explanations of Behaviour Tree Decisions 

**Title (ZH)**: 行为树决策的时间因果解释 

**Authors**: Tamlin Love, Antonio Andriella, Guillem Alenyà  

**Link**: [PDF](https://arxiv.org/pdf/2509.07674)  

**Abstract**: Explainability is a critical tool in helping stakeholders understand robots. In particular, the ability for robots to explain why they have made a particular decision or behaved in a certain way is useful in this regard. Behaviour trees are a popular framework for controlling the decision-making of robots and other software systems, and thus a natural question to ask is whether or not a system driven by a behaviour tree is capable of answering "why" questions. While explainability for behaviour trees has seen some prior attention, no existing methods are capable of generating causal, counterfactual explanations which detail the reasons for robot decisions and behaviour. Therefore, in this work, we introduce a novel approach which automatically generates counterfactual explanations in response to contrastive "why" questions. Our method achieves this by first automatically building a causal model from the structure of the behaviour tree as well as domain knowledge about the state and individual behaviour tree nodes. The resultant causal model is then queried and searched to find a set of diverse counterfactual explanations. We demonstrate that our approach is able to correctly explain the behaviour of a wide range of behaviour tree structures and states. By being able to answer a wide range of causal queries, our approach represents a step towards more transparent, understandable and ultimately trustworthy robotic systems. 

**Abstract (ZH)**: 可解释性是帮助利益相关者理解机器人的重要工具。特别是，机器人能够解释其为何作出特定决策或为何以某种方式行动的能力在这方面非常有用。行为树是一种流行的框架，用于控制机器人的决策和软件系统的决策，因此一个自然的问题是：由行为树驱动的系统是否能够回答“为什么”的问题。尽管已有研究表明行为树的可解释性，但现有的方法尚无法生成能详细说明机器人决策和行为原因的因果性反事实解释。因此，在本工作中，我们提出了一个新颖的方法来自动生成响应对比“为什么”问题的反事实解释。该方法首先从行为树的结构及其关于状态和个体行为树节点的领域知识中自动构建因果模型。然后查询和搜索该因果模型，以找到一组多样化的反事实解释。我们证明，我们的方法能够正确解释广泛类型的行为树结构和状态的行为。通过能够回答广泛的因果查询，我们的方法代表了向着更加透明、可理解且最终更可信赖的机器人系统迈出的一步。 

---
# Collaborative Exploration with a Marsupial Ground-Aerial Robot Team through Task-Driven Map Compression 

**Title (ZH)**: 基于任务驱动的地图压缩的袋鼠地面-空中机器人团队协同探索 

**Authors**: Angelos Zacharia, Mihir Dharmadhikari, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2509.07655)  

**Abstract**: Efficient exploration of unknown environments is crucial for autonomous robots, especially in confined and large-scale scenarios with limited communication. To address this challenge, we propose a collaborative exploration framework for a marsupial ground-aerial robot team that leverages the complementary capabilities of both platforms. The framework employs a graph-based path planning algorithm to guide exploration and deploy the aerial robot in areas where its expected gain significantly exceeds that of the ground robot, such as large open spaces or regions inaccessible to the ground platform, thereby maximizing coverage and efficiency. To facilitate large-scale spatial information sharing, we introduce a bandwidth-efficient, task-driven map compression strategy. This method enables each robot to reconstruct resolution-specific volumetric maps while preserving exploration-critical details, even at high compression rates. By selectively compressing and sharing key data, communication overhead is minimized, ensuring effective map integration for collaborative path planning. Simulation and real-world experiments validate the proposed approach, demonstrating its effectiveness in improving exploration efficiency while significantly reducing data transmission. 

**Abstract (ZH)**: 面向受限和大规模场景下自主机器人高效探索未知环境的协作探索框架 

---
# Decoding RobKiNet: Insights into Efficient Training of Robotic Kinematics Informed Neural Network 

**Title (ZH)**: 解码 RobKiNet：机床运动学启发神经网络高效训练的见解 

**Authors**: Yanlong Peng, Zhigang Wang, Ziwen He, Pengxu Chang, Chuangchuang Zhou, Yu Yan, Ming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.07646)  

**Abstract**: In robots task and motion planning (TAMP), it is crucial to sample within the robot's configuration space to meet task-level global constraints and enhance the efficiency of subsequent motion planning. Due to the complexity of joint configuration sampling under multi-level constraints, traditional methods often lack efficiency. This paper introduces the principle of RobKiNet, a kinematics-informed neural network, for end-to-end sampling within the Continuous Feasible Set (CFS) under multiple constraints in configuration space, establishing its Optimization Expectation Model. Comparisons with traditional sampling and learning-based approaches reveal that RobKiNet's kinematic knowledge infusion enhances training efficiency by ensuring stable and accurate gradient this http URL and quantitative analyses in a 2-DOF space validate its theoretical efficiency, while its application on a 9-DOF autonomous mobile manipulator robot(AMMR) demonstrates superior whole-body and decoupled control, excelling in battery disassembly tasks. RobKiNet outperforms deep reinforcement learning with a training speed 74.29 times faster and a sampling accuracy of up to 99.25%, achieving a 97.33% task completion rate in real-world scenarios. 

**Abstract (ZH)**: 基于运动学指导的神经网络在多约束连续可行集中的端到端采样及其在机器人任务与运动规划中的应用 

---
# Can SSD-Mamba2 Unlock Reinforcement Learning for End-to-End Motion Control? 

**Title (ZH)**: SSD-Mamba2能否解锁强化学习在端到端运动控制中的应用？ 

**Authors**: Gavin Tao, Yinuo Wang, Jinzhao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.07593)  

**Abstract**: End-to-end reinforcement learning for motion control promises unified perception-action policies that scale across embodiments and tasks, yet most deployed controllers are either blind (proprioception-only) or rely on fusion backbones with unfavorable compute-memory trade-offs. Recurrent controllers struggle with long-horizon credit assignment, and Transformer-based fusion incurs quadratic cost in token length, limiting temporal and spatial context. We present a vision-driven cross-modal RL framework built on SSD-Mamba2, a selective state-space backbone that applies state-space duality (SSD) to enable both recurrent and convolutional scanning with hardware-aware streaming and near-linear scaling. Proprioceptive states and exteroceptive observations (e.g., depth tokens) are encoded into compact tokens and fused by stacked SSD-Mamba2 layers. The selective state-space updates retain long-range dependencies with markedly lower latency and memory use than quadratic self-attention, enabling longer look-ahead, higher token resolution, and stable training under limited compute. Policies are trained end-to-end under curricula that randomize terrain and appearance and progressively increase scene complexity. A compact, state-centric reward balances task progress, energy efficiency, and safety. Across diverse motion-control scenarios, our approach consistently surpasses strong state-of-the-art baselines in return, safety (collisions and falls), and sample efficiency, while converging faster at the same compute budget. These results suggest that SSD-Mamba2 provides a practical fusion backbone for scalable, foresightful, and efficient end-to-end motion control. 

**Abstract (ZH)**: 基于SSD-Mamba2的选择性状态空间框架的端到端强化学习在运动控制中的应用：统一感知-行动策略，兼顾-bodied性和任务扩展性 

---
# Improving Machine Learning-Based Robot Self-Collision Checking with Input Positional Encoding 

**Title (ZH)**: 基于输入 positional encoding 提升机器学习驱动的机器人自碰撞检测 

**Authors**: Bartlomiej Kulecki, Dominik Belter  

**Link**: [PDF](https://arxiv.org/pdf/2509.07542)  

**Abstract**: This manuscript investigates the integration of positional encoding -- a technique widely used in computer graphics -- into the input vector of a binary classification model for self-collision detection. The results demonstrate the benefits of incorporating positional encoding, which enhances classification accuracy by enabling the model to better capture high-frequency variations, leading to a more detailed and precise representation of complex collision patterns. The manuscript shows that machine learning-based techniques, such as lightweight multilayer perceptrons (MLPs) operating in a low-dimensional feature space, offer a faster alternative for collision checking than traditional methods that rely on geometric approaches, such as triangle-to-triangle intersection tests and Bounding Volume Hierarchies (BVH) for mesh-based models. 

**Abstract (ZH)**: 这篇论文探讨了将广泛用于计算机图形学的 positional encoding 技术集成到二元分类模型的输入向量中，以提高自我碰撞检测的性能。结果表明，结合 positional encoding 能够提升分类精度，使模型更好地捕捉高频变化，从而更详细和精确地表示复杂的碰撞模式。论文显示，基于机器学习的技术，如在低维特征空间中工作的轻量级多层感知机（MLP），相比于依赖几何方法的传统技术（如三角形到三角形的相交测试和网格模型的包围体积层次结构 BVH），能提供一种更快的碰撞检测替代方案。 

---
# Text2Touch: Tactile In-Hand Manipulation with LLM-Designed Reward Functions 

**Title (ZH)**: Text2Touch: 基于LLM设计奖励函数的手中触觉操作 

**Authors**: Harrison Field, Max Yang, Yijiong Lin, Efi Psomopoulou, David Barton, Nathan F. Lepora  

**Link**: [PDF](https://arxiv.org/pdf/2509.07445)  

**Abstract**: Large language models (LLMs) are beginning to automate reward design for dexterous manipulation. However, no prior work has considered tactile sensing, which is known to be critical for human-like dexterity. We present Text2Touch, bringing LLM-crafted rewards to the challenging task of multi-axis in-hand object rotation with real-world vision based tactile sensing in palm-up and palm-down configurations. Our prompt engineering strategy scales to over 70 environment variables, and sim-to-real distillation enables successful policy transfer to a tactile-enabled fully actuated four-fingered dexterous robot hand. Text2Touch significantly outperforms a carefully tuned human-engineered baseline, demonstrating superior rotation speed and stability while relying on reward functions that are an order of magnitude shorter and simpler. These results illustrate how LLM-designed rewards can significantly reduce the time from concept to deployable dexterous tactile skills, supporting more rapid and scalable multimodal robot learning. Project website: this https URL 

**Abstract (ZH)**: 基于大型语言模型的Text2Touch：将语言模型设计的奖励应用于掌上多轴物体旋转的真实世界视觉触觉感知任务 

---
# Attention and Risk-Aware Decision Framework for Safe Autonomous Driving 

**Title (ZH)**: 基于注意力和风险意识的决策框架以实现安全自主驾驶 

**Authors**: Zhen Tian, Fujiang Yuan, Yangfan He, Qinghao Li, Changlin Chen, Huilin Chen, Tianxiang Xu, Jianyu Duan, Yanhong Peng, Zhihao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.07412)  

**Abstract**: Autonomous driving has attracted great interest due to its potential capability in full-unsupervised driving. Model-based and learning-based methods are widely used in autonomous driving. Model-based methods rely on pre-defined models of the environment and may struggle with unforeseen events. Proximal policy optimization (PPO), an advanced learning-based method, can adapt to the above limits by learning from interactions with the environment. However, existing PPO faces challenges with poor training results, and low training efficiency in long sequences. Moreover, the poor training results are equivalent to collisions in driving tasks. To solve these issues, this paper develops an improved PPO by introducing the risk-aware mechanism, a risk-attention decision network, a balanced reward function, and a safety-assisted mechanism. The risk-aware mechanism focuses on highlighting areas with potential collisions, facilitating safe-driving learning of the PPO. The balanced reward function adjusts rewards based on the number of surrounding vehicles, promoting efficient exploration of the control strategy during training. Additionally, the risk-attention network enhances the PPO to hold channel and spatial attention for the high-risk areas of input images. Moreover, the safety-assisted mechanism supervises and prevents the actions with risks of collisions during the lane keeping and lane changing. Simulation results on a physical engine demonstrate that the proposed algorithm outperforms benchmark algorithms in collision avoidance, achieving higher peak reward with less training time, and shorter driving time remaining on the risky areas among multiple testing traffic flow scenarios. 

**Abstract (ZH)**: 基于风险感知的改进PPO算法在自动驾驶中的应用 

---
# Quantum Machine Learning and Grover's Algorithm for Quantum Optimization of Robotic Manipulators 

**Title (ZH)**: 量子机器学习与Grover算法在机器人 manipulator 优化中的量子优化 

**Authors**: Hassen Nigatu, Shi Gaokun, Li Jituo, Wang Jin, Lu Guodong, Howard Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.07216)  

**Abstract**: Optimizing high-degree of freedom robotic manipulators requires searching complex, high-dimensional configuration spaces, a task that is computationally challenging for classical methods. This paper introduces a quantum native framework that integrates quantum machine learning with Grover's algorithm to solve kinematic optimization problems efficiently. A parameterized quantum circuit is trained to approximate the forward kinematics model, which then constructs an oracle to identify optimal configurations. Grover's algorithm leverages this oracle to provide a quadratic reduction in search complexity. Demonstrated on 1-DoF, 2-DoF, and dual-arm manipulator tasks, the method achieves significant speedups-up to 93x over classical optimizers like Nelder Mead as problem dimensionality increases. This work establishes a foundational, quantum-native framework for robot kinematic optimization, effectively bridging quantum computing and robotics problems. 

**Abstract (ZH)**: 基于量子机器学习和Grover算法的高自由度机器人 manipulator运动优化的量子原生框架 

---
# A Robot That Listens: Enhancing Self-Disclosure and Engagement Through Sentiment-based Backchannels and Active Listening 

**Title (ZH)**: 一个会倾听的机器人：基于情感的回应和主动倾听以增强自我披露和参与度 

**Authors**: Hieu Tran, Go-Eum Cha, Sooyeon Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2509.07873)  

**Abstract**: As social robots get more deeply integrated intoour everyday lives, they will be expected to engage in meaningful conversations and exhibit socio-emotionally intelligent listening behaviors when interacting with people. Active listening and backchanneling could be one way to enhance robots' communicative capabilities and enhance their effectiveness in eliciting deeper self-disclosure, providing a sense of empathy,and forming positive rapport and relationships with this http URL, we developed an LLM-powered social robot that can exhibit contextually appropriate sentiment-based backchannelingand active listening behaviors (active listening+backchanneling) and compared its efficacy in eliciting people's self-disclosurein comparison to robots that do not exhibit any of these listening behaviors (control) and a robot that only exhibitsbackchanneling behavior (backchanneling-only). Through ourexperimental study with sixty-five participants, we found theparticipants who conversed with the active listening robot per-ceived the interactions more positively, in which they exhibited the highest self-disclosures, and reported the strongest senseof being listened to. The results of our study suggest that the implementation of active listening behaviors in social robotshas the potential to improve human-robot communication andcould further contribute to the building of deeper human-robot relationships and rapport. 

**Abstract (ZH)**: 随着社会机器人越来越多地融入我们的日常生活，它们将被期望与人们互动时进行有意义的对话，并表现出社交情感智能的倾听行为。积极倾听和插话可能是增强机器人沟通能力、促进更深入自我披露、提供共情感并建立积极关系的一种方式。基于此，我们开发了一种具备情感驱动插话和积极倾听行为的大型语言模型驱动的社会机器人，并将其与不表现出任何这些倾听行为的控制机器人和仅表现出插话行为的机器人进行了比较，以评估其在促进人们自我披露方面的有效性。通过与六十五名参与者进行的实验研究，我们发现与积极倾听机器人对话的参与者对互动的感受更为积极，自我披露最多，且报告的共情感最强。我们的研究表明，在社会机器人中实施积极倾听行为有可能改善人机沟通，并进一步促进更深入的人机关系和建立积极的互动。 

---
# Bio-inspired decision making in swarms under biases from stubborn robots, corrupted communication, and independent discovery 

**Title (ZH)**: 受顽固机器人偏见、通信失真和独立发现影响的仿生群体决策 

**Authors**: Raina Zakir, Timoteo Carletti, Marco Dorigo, Andreagiovanni Reina  

**Link**: [PDF](https://arxiv.org/pdf/2509.07561)  

**Abstract**: Minimalistic robot swarms offer a scalable, robust, and cost-effective approach to performing complex tasks with the potential to transform applications in healthcare, disaster response, and environmental monitoring. However, coordinating such decentralised systems remains a fundamental challenge, particularly when robots are constrained in communication, computation, and memory. In our study, individual robots frequently make errors when sensing the environment, yet the swarm can rapidly and reliably reach consensus on the best among $n$ discrete options. We compare two canonical mechanisms of opinion dynamics -- direct-switch and cross-inhibition -- which are simple yet effective rules for collective information processing observed in biological systems across scales, from neural populations to insect colonies. We generalise the existing mean-field models by considering asocial biases influencing the opinion dynamics. While swarms using direct-switch reliably select the best option in absence of asocial dynamics, their performance deteriorates once such biases are introduced, often resulting in decision deadlocks. In contrast, bio-inspired cross-inhibition enables faster, more cohesive, accurate, robust, and scalable decisions across a wide range of biased conditions. Our findings provide theoretical and practical insights into the coordination of minimal swarms and offer insights that extend to a broad class of decentralised decision-making systems in biology and engineering. 

**Abstract (ZH)**: 最小化的机器人 swarm 提供了一种可扩展、稳健且成本效益高的方法，用于执行复杂任务，并有潜力在医疗保健、灾难响应和环境监测等领域进行转型。然而，协调这样的分散系统仍然是一个基本挑战，尤其是在机器人在通信、计算和内存方面受到约束的情况下。在我们的研究中，个体机器人在感知环境时经常出现错误，但 swarm 可以迅速且可靠地就 $n$ 个离散选项中最好的一个达成一致。我们比较了两种经典的意见动态机制——直接切换和交叉抑制，这些机制是生物系统中从神经群体到昆虫社群不同尺度上观察到的集体信息处理的简单而有效的规则。我们通过考虑影响意见动态的无社会偏见，推广了现有的均场模型。尽管在没有无社会动力的情况下，使用直接切换的 swarm 可以可靠地选择最佳选项，但在引入这种偏见后，其性能会下降，通常会导致决策僵局。相比之下，受生物启发的交叉抑制能够在广泛偏见条件下实现更快、更协调、更准确、更稳健和更可扩展的决策。我们的研究成果为最小化 swarm 的协调提供了理论和实践见解，并为生物学和工程领域的广泛分散决策系统提供了见解。 

---
# OmniAcc: Personalized Accessibility Assistant Using Generative AI 

**Title (ZH)**: OmniAcc: 使用生成式人工智能的个性化无障碍助手 

**Authors**: Siddhant Karki, Ethan Han, Nadim Mahmud, Suman Bhunia, John Femiani, Vaskar Raychoudhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.07220)  

**Abstract**: Individuals with ambulatory disabilities often encounter significant barriers when navigating urban environments due to the lack of accessible information and tools. This paper presents OmniAcc, an AI-powered interactive navigation system that utilizes GPT-4, satellite imagery, and OpenStreetMap data to identify, classify, and map wheelchair-accessible features such as ramps and crosswalks in the built environment. OmniAcc offers personalized route planning, real-time hands-free navigation, and instant query responses regarding physical accessibility. By using zero-shot learning and customized prompts, the system ensures precise detection of accessibility features, while supporting validation through structured workflows. This paper introduces OmniAcc and explores its potential to assist urban planners and mobility-aid users, demonstrated through a case study on crosswalk detection. With a crosswalk detection accuracy of 97.5%, OmniAcc highlights the transformative potential of AI in improving navigation and fostering more inclusive urban spaces. 

**Abstract (ZH)**: 基于GPT-4、卫星影像和OpenStreetMap数据的AI辅助无障碍导航系统OmniAcc：改善城市环境导航与包容性 

---
# Fine-Tuning Vision-Language Models for Visual Navigation Assistance 

**Title (ZH)**: 视觉语言模型的微调以实现视觉导航辅助 

**Authors**: Xiao Li, Bharat Gandhi, Ming Zhan, Mohit Nehra, Zhicheng Zhang, Yuchen Sun, Meijia Song, Naisheng Zhang, Xi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.07488)  

**Abstract**: We address vision-language-driven indoor navigation to assist visually impaired individuals in reaching a target location using images and natural language guidance. Traditional navigation systems are ineffective indoors due to the lack of precise location data. Our approach integrates vision and language models to generate step-by-step navigational instructions, enhancing accessibility and independence. We fine-tune the BLIP-2 model with Low Rank Adaptation (LoRA) on a manually annotated indoor navigation dataset. We propose an evaluation metric that refines the BERT F1 score by emphasizing directional and sequential variables, providing a more comprehensive measure of navigational performance. After applying LoRA, the model significantly improved in generating directional instructions, overcoming limitations in the original BLIP-2 model. 

**Abstract (ZH)**: 基于视觉语言的室内导航以辅助视觉障碍个体到达目标位置：融合视觉和语言模型生成逐步导航指令 

---
# DepthVision: Robust Vision-Language Understanding through GAN-Based LiDAR-to-RGB Synthesis 

**Title (ZH)**: DepthVision: 基于GAN的LiDAR到RGB合成的鲁棒多模态理解 

**Authors**: Sven Kirchner, Nils Purschke, Ross Greer, Alois C. Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.07463)  

**Abstract**: Ensuring reliable robot operation when visual input is degraded or insufficient remains a central challenge in robotics. This letter introduces DepthVision, a framework for multimodal scene understanding designed to address this problem. Unlike existing Vision-Language Models (VLMs), which use only camera-based visual input alongside language, DepthVision synthesizes RGB images from sparse LiDAR point clouds using a conditional generative adversarial network (GAN) with an integrated refiner network. These synthetic views are then combined with real RGB data using a Luminance-Aware Modality Adaptation (LAMA), which blends the two types of data dynamically based on ambient lighting conditions. This approach compensates for sensor degradation, such as darkness or motion blur, without requiring any fine-tuning of downstream vision-language models. We evaluate DepthVision on real and simulated datasets across various models and tasks, with particular attention to safety-critical tasks. The results demonstrate that our approach improves performance in low-light conditions, achieving substantial gains over RGB-only baselines while preserving compatibility with frozen VLMs. This work highlights the potential of LiDAR-guided RGB synthesis for achieving robust robot operation in real-world environments. 

**Abstract (ZH)**: 确保视觉输入降级或不足时机器人可靠运行仍然是机器人领域的核心挑战。本文介绍了DepthVision框架，该框架旨在解决这一问题。与现有的视觉-语言模型（VLMs）仅使用摄像头视觉输入结合语言不同，DepthVision通过条件生成对抗网络（GAN）结合嵌入式精炼网络，从稀疏LiDAR点云中合成RGB图像。这些合成视角随后与真实RGB数据结合使用Luminance-Aware Modality Adaptation（LAMA），根据环境光照条件动态融合两种类型的数据。这种方法可以在不需微调下游视觉-语言模型的情况下补偿传感器降级，例如黑暗或运动模糊。我们在各种模型和任务的真实和仿真数据集上评估了DepthVision，尤其关注安全性关键任务。实验结果表明，我们的方法在低光照条件下提高了性能，相对于仅使用RGB的基线实现了显著改进，同时保持与冻结的VLMs的兼容性。本文强调了LiDAR引导的RGB合成在实现实时环境中稳健机器人运行方面的潜在价值。 

---
# An efficient deep reinforcement learning environment for flexible job-shop scheduling 

**Title (ZH)**: 一种高效深度强化学习环境下的灵活作业 shop 调度方法 

**Authors**: Xinquan Wu, Xuefeng Yan, Mingqiang Wei, Donghai Guan  

**Link**: [PDF](https://arxiv.org/pdf/2509.07019)  

**Abstract**: The Flexible Job-shop Scheduling Problem (FJSP) is a classical combinatorial optimization problem that has a wide-range of applications in the real world. In order to generate fast and accurate scheduling solutions for FJSP, various deep reinforcement learning (DRL) scheduling methods have been developed. However, these methods are mainly focused on the design of DRL scheduling Agent, overlooking the modeling of DRL environment. This paper presents a simple chronological DRL environment for FJSP based on discrete event simulation and an end-to-end DRL scheduling model is proposed based on the proximal policy optimization (PPO). Furthermore, a short novel state representation of FJSP is proposed based on two state variables in the scheduling environment and a novel comprehensible reward function is designed based on the scheduling area of machines. Experimental results on public benchmark instances show that the performance of simple priority dispatching rules (PDR) is improved in our scheduling environment and our DRL scheduling model obtains competing performance compared with OR-Tools, meta-heuristic, DRL and PDR scheduling methods. 

**Abstract (ZH)**: 基于离散事件仿真的一种简单的柔性作业车间调度问题 chronological DRL 环境及端到端的基于 PPO 的 DRL 调度模型 

---
# A Knowledge-Guided Cross-Modal Feature Fusion Model for Local Traffic Demand Prediction 

**Title (ZH)**: 基于知识引导的跨模态特征融合模型用于局部交通需求预测 

**Authors**: Lingyu Zhang, Pengfei Xu, Guobin Wu, Jian Liang, Ruiyang Dong, Yunhai Wang, Xuan Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.06976)  

**Abstract**: Traffic demand prediction plays a critical role in intelligent transportation systems. Existing traffic prediction models primarily rely on temporal traffic data, with limited efforts incorporating human knowledge and experience for urban traffic demand forecasting. However, in real-world scenarios, traffic knowledge and experience derived from human daily life significantly influence precise traffic prediction. Such knowledge and experiences can guide the model in uncovering latent patterns within traffic data, thereby enhancing the accuracy and robustness of predictions. To this end, this paper proposes integrating structured temporal traffic data with textual data representing human knowledge and experience, resulting in a novel knowledge-guided cross-modal feature representation learning (KGCM) model for traffic demand prediction. Based on regional transportation characteristics, we construct a prior knowledge dataset using a large language model combined with manual authoring and revision, covering both regional and global knowledge and experiences. The KGCM model then learns multimodal data features through designed local and global adaptive graph networks, as well as a cross-modal feature fusion mechanism. A proposed reasoning-based dynamic update strategy enables dynamic optimization of the graph model's parameters, achieving optimal performance. Experiments on multiple traffic datasets demonstrate that our model accurately predicts future traffic demand and outperforms existing state-of-the-art (SOTA) models. 

**Abstract (ZH)**: 交通需求预测在智能交通运输系统中起着关键作用。现有的交通预测模型主要依赖于时间序列交通数据，较少尝试整合人类的知识和经验以进行城市交通需求预测。然而，在实际场景中，人类日常生活中的交通知识和经验显著影响精确的交通预测。这些知识和经验可以引导模型在交通数据中发现潜在模式，从而提高预测的准确性和鲁棒性。为此，本文提出将结构化的时间序列交通数据与代表人类知识和经验的文本数据相结合，提出了一种新型的知识引导的跨模态特征表示学习（KGCM）模型，用于交通需求预测。基于地区交通特征，我们使用大规模语言模型结合人工编写和修订构建了一个先验知识数据集，涵盖了地区性和全球性的知识和经验。KGCM模型通过设计的局部和全局自适应图网络以及跨模态特征融合机制学习多模态数据特征。提出的基于推理的动态更新策略能够动态优化图模型的参数，以实现最佳性能。在多个交通数据集上的实验表明，我们的模型能够准确预测未来交通需求，并优于现有最先进的（SOTA）模型。 

---
