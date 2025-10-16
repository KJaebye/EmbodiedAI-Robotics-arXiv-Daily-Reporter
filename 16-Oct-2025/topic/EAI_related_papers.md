# InternVLA-M1: A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy 

**Title (ZH)**: InternVLA-M1：一种面向通用机器人政策的空间引导型视觉-语言-行动框架 

**Authors**: Xinyi Chen, Yilun Chen, Yanwei Fu, Ning Gao, Jiaya Jia, Weiyang Jin, Hao Li, Yao Mu, Jiangmiao Pang, Yu Qiao, Yang Tian, Bin Wang, Bolun Wang, Fangjing Wang, Hanqing Wang, Tai Wang, Ziqin Wang, Xueyuan Wei, Chao Wu, Shuai Yang, Jinhui Ye, Junqiu Yu, Jia Zeng, Jingjing Zhang, Jinyu Zhang, Shi Zhang, Feng Zheng, Bowen Zhou, Yangkun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13778)  

**Abstract**: We introduce InternVLA-M1, a unified framework for spatial grounding and robot control that advances instruction-following robots toward scalable, general-purpose intelligence. Its core idea is spatially guided vision-language-action training, where spatial grounding serves as the critical link between instructions and robot actions. InternVLA-M1 employs a two-stage pipeline: (i) spatial grounding pre-training on over 2.3M spatial reasoning data to determine ``where to act'' by aligning instructions with visual, embodiment-agnostic positions, and (ii) spatially guided action post-training to decide ``how to act'' by generating embodiment-aware actions through plug-and-play spatial prompting. This spatially guided training recipe yields consistent gains: InternVLA-M1 outperforms its variant without spatial guidance by +14.6% on SimplerEnv Google Robot, +17% on WidowX, and +4.3% on LIBERO Franka, while demonstrating stronger spatial reasoning capability in box, point, and trace prediction. To further scale instruction following, we built a simulation engine to collect 244K generalizable pick-and-place episodes, enabling a 6.2% average improvement across 200 tasks and 3K+ objects. In real-world clustered pick-and-place, InternVLA-M1 improved by 7.3%, and with synthetic co-training, achieved +20.6% on unseen objects and novel configurations. Moreover, in long-horizon reasoning-intensive scenarios, it surpassed existing works by over 10%. These results highlight spatially guided training as a unifying principle for scalable and resilient generalist robots. Code and models are available at this https URL. 

**Abstract (ZH)**: 统一空间定位与机器人控制框架：InternVLA-M1及其在可扩展通用智能机器人指令遵循中的应用 

---
# LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models 

**Title (ZH)**: LIBERO-Plus:  vision-language-action模型的深入鲁棒性分析 

**Authors**: Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13626)  

**Abstract**: Visual-Language-Action (VLA) models report impressive success rates on robotic manipulation benchmarks, yet these results may mask fundamental weaknesses in robustness. We perform a systematic vulnerability analysis by introducing controlled perturbations across seven dimensions: objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures and sensor noise. We comprehensively analyzed multiple state-of-the-art models and revealed consistent brittleness beneath apparent competence. Our analysis exposes critical weaknesses: models exhibit extreme sensitivity to perturbation factors, including camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations. Surprisingly, models are largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely. Our findings challenge the assumption that high benchmark scores equate to true competency and highlight the need for evaluation practices that assess reliability under realistic variation. 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型在机器人操作基准测试中取得了令人印象深刻的成功率，但这些结果可能掩盖了鲁棒性方面的根本弱点。我们通过在七个维度上引入受控的扰动进行了系统的脆弱性分析：物体布局、相机视角、机器人初始状态、语言指令、光照条件、背景纹理和传感器噪声。我们全面分析了多个领先模型，并揭示了表面看似能力较强的实际脆弱性。我们的分析暴露了一些关键弱点：模型对扰动因素表现出极端的敏感性，包括相机视角和机器人初始状态，在适度扰动下性能从95%下降到低于30%。令人惊讶的是，模型对语言变化的敏感性很小，进一步的实验表明，模型倾向于完全忽略语言指令。我们的研究挑战了高基准分数等同于真正能力的观点，凸显了在实际变化条件下评估可靠性的必要性。 

---
# A Modular Object Detection System for Humanoid Robots Using YOLO 

**Title (ZH)**: 基于YOLO的人形机器人模块化目标检测系统 

**Authors**: Nicolas Pottier, Meng Cheng Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.13625)  

**Abstract**: Within the field of robotics, computer vision remains a significant barrier to progress, with many tasks hindered by inefficient vision systems. This research proposes a generalized vision module leveraging YOLOv9, a state-of-the-art framework optimized for computationally constrained environments like robots. The model is trained on a dataset tailored to the FIRA robotics Hurocup. A new vision module is implemented in ROS1 using a virtual environment to enable YOLO compatibility. Performance is evaluated using metrics such as frames per second (FPS) and Mean Average Precision (mAP). Performance is then compared to the existing geometric framework in static and dynamic contexts. The YOLO model achieved comparable precision at a higher computational cost then the geometric model, while providing improved robustness. 

**Abstract (ZH)**: 机器人领域中，计算机视觉依然是进步的重要障碍，许多任务受限于低效的视觉系统。本研究提出了一种基于YOLOv9的通用视觉模块，YOLOv9是为计算资源受限的环境（如机器人）优化的先进框架。该模型在专门为FIRA机器人Hurocup比赛设计的数据集上进行训练。通过在ROS1中实现一个新的视觉模块并在虚拟环境中启用YOLO兼容性。性能通过每秒帧数（FPS）和平均平均精确度（mAP）等指标进行评估。然后将YOLO模型的表现与现有的几何框架在静态和动态环境中进行比较。结果表明，YOLO模型在计算成本更高的情况下实现了可比较的精度，并提供了更好的鲁棒性。 

---
# Development of an Intuitive GUI for Non-Expert Teleoperation of Humanoid Robots 

**Title (ZH)**: 非专家操作类人机器人直观GUI的发展 

**Authors**: Austin Barret, Meng Cheng Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.13594)  

**Abstract**: The operation of humanoid robotics is an essential field of research with many practical and competitive applications. Many of these systems, however, do not invest heavily in developing a non-expert-centered graphical user interface (GUI) for operation. The focus of this research is to develop a scalable GUI that is tailored to be simple and intuitive so non-expert operators can control the robot through a FIRA-regulated obstacle course. Using common practices from user interface development (UI) and understanding concepts described in human-robot interaction (HRI) and other related concepts, we will develop a new interface with the goal of a non-expert teleoperation system. 

**Abstract (ZH)**: 人形机器人操作的图形用户界面设计研究：面向非专家的可扩展界面开发 

---
# Bridge the Gap: Enhancing Quadruped Locomotion with Vertical Ground Perturbations 

**Title (ZH)**: 桥接差距：通过垂直地面扰动增强四足运动能力 

**Authors**: Maximilian Stasica, Arne Bick, Nico Bohlinger, Omid Mohseni, Max Johannes Alois Fritzsche, Clemens Hübler, Jan Peters, André Seyfarth  

**Link**: [PDF](https://arxiv.org/pdf/2510.13488)  

**Abstract**: Legged robots, particularly quadrupeds, excel at navigating rough terrains, yet their performance under vertical ground perturbations, such as those from oscillating surfaces, remains underexplored. This study introduces a novel approach to enhance quadruped locomotion robustness by training the Unitree Go2 robot on an oscillating bridge - a 13.24-meter steel-and-concrete structure with a 2.0 Hz eigenfrequency designed to perturb locomotion. Using Reinforcement Learning (RL) with the Proximal Policy Optimization (PPO) algorithm in a MuJoCo simulation, we trained 15 distinct locomotion policies, combining five gaits (trot, pace, bound, free, default) with three training conditions: rigid bridge and two oscillating bridge setups with differing height regulation strategies (relative to bridge surface or ground). Domain randomization ensured zero-shot transfer to the real-world bridge. Our results demonstrate that policies trained on the oscillating bridge exhibit superior stability and adaptability compared to those trained on rigid surfaces. Our framework enables robust gait patterns even without prior bridge exposure. These findings highlight the potential of simulation-based RL to improve quadruped locomotion during dynamic ground perturbations, offering insights for designing robots capable of traversing vibrating environments. 

**Abstract (ZH)**: 腿式机器人，尤其是四足机器人，在穿越崎岖地形方面表现出色，但在应对垂直地面振荡等动态地面扰动方面的性能仍需进一步探索。本研究通过在一种2.0 Hz固有频率的振荡桥梁上训练Unitree Go2四足机器人，提出了一种增强四足机器人运动稳定性和适应性的新方法。该振荡桥梁由13.24米长的钢混结构组成，用于模拟运动的扰动。利用MuJoCo模拟中的强化学习（RL）和 proximal policy optimization (PPO) 算法，我们训练了15种不同的运动策略，结合了五种步态（ tönt, pace, bound, free, default）和三种训练条件：刚性桥梁以及两个不同高度调节策略的振荡桥梁设置（相对于桥梁表面或地面）。领域随机化确保了在实际桥梁上的零样本迁移。结果显示，振荡桥梁上训练的策略在稳定性与适应性方面优于刚性表面上训练的策略。本框架使四足机器人即使没有前桥暴露也能展现出稳健的步态模式。这些发现突显了基于模拟的RL在提高四足机器人在动态地面扰动环境下运动性能方面的潜力，为设计能够穿越振动环境的机器人提供了参考。 

---
# Adversarial Fine-tuning in Offline-to-Online Reinforcement Learning for Robust Robot Control 

**Title (ZH)**: 离线到在线强化学习中的对抗微调在鲁棒机器人控制中的应用 

**Authors**: Shingo Ayabe, Hiroshi Kera, Kazuhiko Kawamoto  

**Link**: [PDF](https://arxiv.org/pdf/2510.13358)  

**Abstract**: Offline reinforcement learning enables sample-efficient policy acquisition without risky online interaction, yet policies trained on static datasets remain brittle under action-space perturbations such as actuator faults. This study introduces an offline-to-online framework that trains policies on clean data and then performs adversarial fine-tuning, where perturbations are injected into executed actions to induce compensatory behavior and improve resilience. A performance-aware curriculum further adjusts the perturbation probability during training via an exponential-moving-average signal, balancing robustness and stability throughout the learning process. Experiments on continuous-control locomotion tasks demonstrate that the proposed method consistently improves robustness over offline-only baselines and converges faster than training from scratch. Matching the fine-tuning and evaluation conditions yields the strongest robustness to action-space perturbations, while the adaptive curriculum strategy mitigates the degradation of nominal performance observed with the linear curriculum strategy. Overall, the results show that adversarial fine-tuning enables adaptive and robust control under uncertain environments, bridging the gap between offline efficiency and online adaptability. 

**Abstract (ZH)**: 离线强化学习使得在无需风险在线交互的情况下获得样本高效策略成为可能，但基于静态数据集训练的策略在动作空间扰动如执行器故障的情况下依然脆弱。本研究提出了一种离线到在线框架，该框架在清洁数据上训练策略，然后进行对抗性微调，通过在执行的动作中注入扰动以诱导补偿行为并提高鲁棒性。一种基于指数移动平均信号的性能感知课程进一步在训练过程中调整扰动概率，平衡学习过程中的鲁棒性和稳定性。实验结果表明，所提出的方法在连续控制运动任务中比仅基于离线的方法更具鲁棒性，并且比从头开始训练更快地收敛。使微调和评估条件匹配能最大程度地提高对动作空间扰动的鲁棒性，而自适应课程策略则减轻了与线性课程策略相比观察到的标准性能退化。总体而言，结果表明对抗性微调能够在不确定环境中实现自适应和稳健的控制，弥合了离线效率和在线适应性之间的差距。 

---
# DAMM-LOAM: Degeneracy Aware Multi-Metric LiDAR Odometry and Mapping 

**Title (ZH)**: DAMM-LOAM: 退化感知多度量激光雷达里程计与 Mapping 

**Authors**: Nishant Chandna, Akshat Kaushal  

**Link**: [PDF](https://arxiv.org/pdf/2510.13287)  

**Abstract**: LiDAR Simultaneous Localization and Mapping (SLAM) systems are essential for enabling precise navigation and environmental reconstruction across various applications. Although current point-to-plane ICP algorithms perform effec- tively in structured, feature-rich environments, they struggle in scenarios with sparse features, repetitive geometric structures, and high-frequency motion. This leads to degeneracy in 6- DOF pose estimation. Most state-of-the-art algorithms address these challenges by incorporating additional sensing modalities, but LiDAR-only solutions continue to face limitations under such conditions. To address these issues, we propose a novel Degeneracy-Aware Multi-Metric LiDAR Odometry and Map- ping (DAMM-LOAM) module. Our system improves mapping accuracy through point cloud classification based on surface normals and neighborhood analysis. Points are classified into ground, walls, roof, edges, and non-planar points, enabling accurate correspondences. A Degeneracy-based weighted least squares-based ICP algorithm is then applied for accurate odom- etry estimation. Additionally, a Scan Context based back-end is implemented to support robust loop closures. DAMM-LOAM demonstrates significant improvements in odometry accuracy, especially in indoor environments such as long corridors 

**Abstract (ZH)**: 基于退化感知的多度量LiDAR simultaneous localization and mapping (DAMM-LOAM)模块 

---
# RoboHiMan: A Hierarchical Evaluation Paradigm for Compositional Generalization in Long-Horizon Manipulation 

**Title (ZH)**: RoboHiMan: 一种层次化的评估范式，用于长时效Manipulation中的组合作用泛化 

**Authors**: Yangtao Chen, Zixuan Chen, Nga Teng Chan, Junting Chen, Junhui Yin, Jieqi Shi, Yang Gao, Yong-Lu Li, Jing Huo  

**Link**: [PDF](https://arxiv.org/pdf/2510.13149)  

**Abstract**: Enabling robots to flexibly schedule and compose learned skills for novel long-horizon manipulation under diverse perturbations remains a core challenge. Early explorations with end-to-end VLA models show limited success, as these models struggle to generalize beyond the training distribution. Hierarchical approaches, where high-level planners generate subgoals for low-level policies, bring certain improvements but still suffer under complex perturbations, revealing limited capability in skill composition. However, existing benchmarks primarily emphasize task completion in long-horizon settings, offering little insight into compositional generalization, robustness, and the interplay between planning and execution. To systematically investigate these gaps, we propose RoboHiMan, a hierarchical evaluation paradigm for compositional generalization in long-horizon manipulation. RoboHiMan introduces HiMan-Bench, a benchmark of atomic and compositional tasks under diverse perturbations, supported by a multi-level training dataset for analyzing progressive data scaling, and proposes three evaluation paradigms (vanilla, decoupled, coupled) that probe the necessity of skill composition and reveal bottlenecks in hierarchical architectures. Experiments highlight clear capability gaps across representative models and architectures, pointing to directions for advancing models better suited to real-world long-horizon manipulation tasks. Videos and open-source code can be found on our project website: this https URL. 

**Abstract (ZH)**: 使机器人能够灵活调度和组合学习到的技能以应对多样化的干扰进行新颖的长期操作调用仍然是一个核心挑战。端到端的VLA模型早期探索显示有限的成功，因为这些模型难以在训练分布之外泛化。分层方法，其中高层规划器为低层策略生成子目标，在复杂干扰下表现出一定的提高，但仍然在技能组合方面显示出有限的能力。然而，现有的基准主要强调在长期设置下的任务完成，对组合泛化、鲁棒性和规划与执行之间的相互作用提供很少的见解。为了系统地研究这些差距，我们提出了RoboHiMan，这是一种分层评估范式，用于长期操作中的组合泛化。RoboHiMan引入了HiMan-Bench，这是一个在多样干扰下包含原子和组合任务的基准，并提供了一个多层次训练数据集以分析逐级数据扩展，并提出了三种评估范式（vanilla、解耦、耦合），以探究技能组合的必要性并揭示分层架构中的瓶颈。实验突显了代表性模型和架构在能力上的明显差距，指出了改进更适合真实世界长期操作任务的模型的方向。更多信息和开源代码可在我们的项目网站上找到：<这个链接>。 

---
# VLA-0: Building State-of-the-Art VLAs with Zero Modification 

**Title (ZH)**: VLA-0: 构建零修改状态下的一流VLAs 

**Authors**: Ankit Goyal, Hugo Hadfield, Xuning Yang, Valts Blukis, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2510.13054)  

**Abstract**: Vision-Language-Action models (VLAs) hold immense promise for enabling generalist robot manipulation. However, the best way to build them remains an open question. Current approaches often add complexity, such as modifying the existing vocabulary of a Vision-Language Model (VLM) with action tokens or introducing special action heads. Curiously, the simplest strategy of representing actions directly as text has remained largely unexplored. This work introduces VLA-0 to investigate this idea. We find that VLA-0 is not only effective; it is surprisingly powerful. With the right design, VLA-0 outperforms more involved models. On LIBERO, a popular benchmark for evaluating VLAs, VLA-0 outperforms all existing methods trained on the same robotic data, including $\pi_0.5$-KI, OpenVLA-OFT and SmolVLA. Furthermore, without large-scale robotics-specific training, it outperforms methods trained on large-scale robotic data, like $\pi_0.5$-KI, $\pi_0$, GR00T-N1 and MolmoAct. These findings also translate to the real world, where VLA-0 outperforms SmolVLA, a VLA model pre-trained on large-scale real data. This paper summarizes our unexpected findings and spells out the specific techniques required to unlock the high performance of this simple yet potent VLA design. Visual results, code, and trained models are provided here: this https URL. 

**Abstract (ZH)**: Vision-Language-Action模型（VLAs）在实现通用机器人操作方面展现出巨大的潜力。然而，如何构建它们仍然是一个开放的问题。当前的方法往往增加了复杂性，例如通过在视觉-语言模型（VLM）的现有词汇表中添加动作标记或将特殊动作头引入系统。有趣的是，直接将动作表示为文本的最简单策略仍未得到充分探索。本文引入了VLA-0来研究这一想法。我们发现，VLA-0不仅有效，而且出人意料地强大。通过合适的 designs，VLA-0 性能超过了更复杂的模型。在 LIBERO 这一流行的 VLAs 评估基准测试中，VLA-0 在使用相同机器人数据训练的所有现有方法中表现最佳，包括 $\pi_0.5$-KI、OpenVLA-OFT 和 SmolVLA。此外，即使没有大规模的特定于机器人训练，它也超过了在大规模机器人数据上训练的方法，如 $\pi_0.5$-KI、$\pi_0$、GR00T-N1 和 MolmoAct。这些发现也适用于现实世界，在那里 VLA-0 在 SmolVLA 上表现更好，SmolVLA 是在大规模真实数据上预训练的 VLA 模型。本文总结了我们意想不到的发现，并详细说明了解锁这种简单而强大的 VLA 设计高性能所需的具体技术。视觉结果、代码和训练模型在此处提供：this https URL。 

---
# Kinematic Kitbashing for Modeling Functional Articulated Objects 

**Title (ZH)**: 机械动力组件拼装法模拟功能性活动对象 

**Authors**: Minghao Guo, Victor Zordan, Sheldon Andrews, Wojciech Matusik, Maneesh Agrawala, Hsueh-Ti Derek Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13048)  

**Abstract**: We introduce Kinematic Kitbashing, an automatic framework that synthesizes functionality-aware articulated objects by reusing parts from existing models. Given a kinematic graph with a small collection of articulated parts, our optimizer jointly solves for the spatial placement of every part so that (i) attachments remain geometrically sound over the entire range of motion and (ii) the assembled object satisfies user-specified functional goals such as collision-free actuation, reachability, or trajectory following. At its core is a kinematics-aware attachment energy that aligns vector distance function features sampled across multiple articulation snapshots. We embed this attachment term within an annealed Riemannian Langevin dynamics sampler that treats functionality objectives as additional energies, enabling robust global exploration while accommodating non-differentiable functionality objectives and constraints. Our framework produces a wide spectrum of assembled articulated shapes, from trash-can wheels grafted onto car bodies to multi-segment lamps, gear-driven paddlers, and reconfigurable furniture, and delivers strong quantitative improvements over state-of-the-art baselines across geometric, kinematic, and functional metrics. By tightly coupling articulation-aware geometry matching with functionality-driven optimization, Kinematic Kitbashing bridges part-based shape modeling and functional assembly design, empowering rapid creation of interactive articulated assets. 

**Abstract (ZH)**: 基于运动的组件拼装自动框架：合成功能感知的articulated对象 

---
# Development of a Linear Guide-Rail Testbed for Physically Emulating ISAM Operations 

**Title (ZH)**: 基于ISAM操作物理仿真的一种线性导轨实验台开发 

**Authors**: Robert Muldrow, Channing Ludden, Christopher Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2510.13005)  

**Abstract**: In-Space Servicing, Assembly, and Manufacturing (ISAM) is a set of emerging operations that provides several benefits to improve the longevity, capacity, mo- bility, and expandability of existing and future space assets. Serial robotic ma- nipulators are particularly vital in accomplishing ISAM operations, however, the complex perturbation forces and motions associated with movement of a robotic arm on a free-flying satellite presents a complex controls problem requiring addi- tional study. While many dynamical models are developed, experimentally test- ing and validating these models is challenging given that the models operate in space, where satellites have six-degrees-of-freedom (6-DOF). This paper attempts to resolve those challenges by presenting the design and development of a new hardware-in-the-loop (HIL) experimental testbed utilized to emulate ISAM. This emulation will be accomplished by means of a 6-DOF UR3e robotic arm attached to a satellite bus. This satellite bus is mounted to a 1-DOF guide-rail system, en- abling the satellite bus and robotic arm to move freely in one linear direction. This experimental ISAM emulation system will explore and validate models for space motion, serial robot manipulation, and contact mechanics. 

**Abstract (ZH)**: 太空服务、装配与制造（ISAM）操作是一系列新兴运营活动，旨在提高现有和未来太空资产的寿命、容量、机动性和扩展性。串行机器人操作器在完成ISAM操作中至关重要，然而，自由飞卫星上机器人臂运动相关的复杂扰动力和运动带来了复杂的控制问题，需要额外的研究。虽然开发了诸多动态模型，但由于卫星在具有六自由度（6-DOF）的空间中操作，实验性地测试和验证这些模型极具挑战性。本文旨在通过展示一种新的硬件在环（HIL）实验测试平台来解决这些挑战，该平台用于模拟ISAM。这种模拟将通过连接到卫星总线的6-DOF UR3e机器人臂来实现。该卫星总线安装在一个单自由度导轨系统上，使卫星总线和机器人臂能够在单一线性方向上自由移动。该实验ISAM模拟系统将探索和验证空间运动、串行机器人操作以及接触力学的模型。 

---
# UNCAP: Uncertainty-Guided Planning Using Natural Language Communication for Cooperative Autonomous Vehicles 

**Title (ZH)**: UNCAP：基于自然语言通信的不确定性指导规划方法及其在协同自动驾驶车辆中的应用 

**Authors**: Neel P. Bhatt, Po-han Li, Kushagra Gupta, Rohan Siva, Daniel Milan, Alexander T. Hogue, Sandeep P. Chinchali, David Fridovich-Keil, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12992)  

**Abstract**: Safe large-scale coordination of multiple cooperative connected autonomous vehicles (CAVs) hinges on communication that is both efficient and interpretable. Existing approaches either rely on transmitting high-bandwidth raw sensor data streams or neglect perception and planning uncertainties inherent in shared data, resulting in systems that are neither scalable nor safe. To address these limitations, we propose Uncertainty-Guided Natural Language Cooperative Autonomous Planning (UNCAP), a vision-language model-based planning approach that enables CAVs to communicate via lightweight natural language messages while explicitly accounting for perception uncertainty in decision-making. UNCAP features a two-stage communication protocol: (i) an ego CAV first identifies the subset of vehicles most relevant for information exchange, and (ii) the selected CAVs then transmit messages that quantitatively express their perception uncertainty. By selectively fusing messages that maximize mutual information, this strategy allows the ego vehicle to integrate only the most relevant signals into its decision-making, improving both the scalability and reliability of cooperative planning. Experiments across diverse driving scenarios show a 63% reduction in communication bandwidth with a 31% increase in driving safety score, a 61% reduction in decision uncertainty, and a four-fold increase in collision distance margin during near-miss events. Project website: this https URL 

**Abstract (ZH)**: 安全的大规模协调多辆合作连接自主车辆依赖于高效且可解释的通信。现有的方法要么依赖于传输高带宽的原始传感器数据流，要么忽视共享数据中固有的感知和规划不确定性，导致系统既不具有可扩展性也不安全。为了解决这些限制，我们提出了一种基于视觉语言模型的合作自主规划方法——不确定性指导自然语言合作自主规划（UNCAP），该方法使自主车辆能够通过轻量级自然语言消息进行通信，并在决策过程中明确考虑感知不确定性。UNCAP 具有两阶段的通信协议：（i）一辆ego自主车辆首先识别最相关的车辆子集以进行信息交换，（ii）然后选择的自主车辆传输量化表示其感知不确定性的消息。通过选择性地融合最大化互信息的消息，该策略允许ego车辆仅整合最相关的信号进行决策，从而提高合作规划的可扩展性和可靠性。在多种驾驶场景下的实验表明，通信带宽减少了63%，驾驶安全性分数提高了31%，决策不确定性减少了61%，在接近碰撞事件中碰撞距离裕度提高了四倍。项目网站：这个 https URL。 

---
# Actron3D: Learning Actionable Neural Functions from Videos for Transferable Robotic Manipulation 

**Title (ZH)**: Actron3D：从视频学习可操作的神经函数以实现可转移的机器人 manipulation 

**Authors**: Anran Zhang, Hanzhi Chen, Yannick Burkhardt, Yao Zhong, Johannes Betz, Helen Oleynikova, Stefan Leutenegger  

**Link**: [PDF](https://arxiv.org/pdf/2510.12971)  

**Abstract**: We present Actron3D, a framework that enables robots to acquire transferable 6-DoF manipulation skills from just a few monocular, uncalibrated, RGB-only human videos. At its core lies the Neural Affordance Function, a compact object-centric representation that distills actionable cues from diverse uncalibrated videos-geometry, visual appearance, and affordance-into a lightweight neural network, forming a memory bank of manipulation skills. During deployment, we adopt a pipeline that retrieves relevant affordance functions and transfers precise 6-DoF manipulation policies via coarse-to-fine optimization, enabled by continuous queries to the multimodal features encoded in the neural functions. Experiments in both simulation and the real world demonstrate that Actron3D significantly outperforms prior methods, achieving a 14.9 percentage point improvement in average success rate across 13 tasks while requiring only 2-3 demonstration videos per task. 

**Abstract (ZH)**: Actron3D：一种从少量未校准单目RGB人类视频中获取可 Transfer 的6-DoF 手 manip 操作技能的框架 

---
# Gaussian Process Implicit Surfaces as Control Barrier Functions for Safe Robot Navigation 

**Title (ZH)**: 基于高斯过程隐表面的控制障碍函数在机器人安全导航中的应用 

**Authors**: Mouhyemen Khan, Tatsuya Ibuki, Abhijit Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.12919)  

**Abstract**: Level set methods underpin modern safety techniques such as control barrier functions (CBFs), while also serving as implicit surface representations for geometric shapes via distance fields. Inspired by these two paradigms, we propose a unified framework where the implicit surface itself acts as a CBF. We leverage Gaussian process (GP) implicit surface (GPIS) to represent the safety boundaries, using safety samples which are derived from sensor measurements to condition the GP. The GP posterior mean defines the implicit safety surface (safety belief), while the posterior variance provides a robust safety margin. Although GPs have favorable properties such as uncertainty estimation and analytical tractability, they scale cubically with data. To alleviate this issue, we develop a sparse solution called sparse Gaussian CBFs. To the best of our knowledge, GPIS have not been explicitly used to synthesize CBFs. We validate the approach on collision avoidance tasks in two settings: a simulated 7-DOF manipulator operating around the Stanford bunny, and a quadrotor navigating in 3D around a physical chair. In both cases, Gaussian CBFs (with and without sparsity) enable safe interaction and collision-free execution of trajectories that would otherwise intersect the objects. 

**Abstract (ZH)**: 基于拉格朗日方法的高斯过程控制障碍函数：统一框架及其应用 

---
# MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control 

**Title (ZH)**: MimicKit：运动模仿与控制的强化学习框架 

**Authors**: Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.13794)  

**Abstract**: MimicKit is an open-source framework for training motion controllers using motion imitation and reinforcement learning. The codebase provides implementations of commonly-used motion-imitation techniques and RL algorithms. This framework is intended to support research and applications in computer graphics and robotics by providing a unified training framework, along with standardized environment, agent, and data structures. The codebase is designed to be modular and easily configurable, enabling convenient modification and extension to new characters and tasks. The open-source codebase is available at: this https URL. 

**Abstract (ZH)**: MimicKit是基于运动模仿和强化学习训练运动控制器的开源框架 

---
# A New Perspective on Transformers in Online Reinforcement Learning for Continuous Control 

**Title (ZH)**: 在线连续控制中变换器的新视角 

**Authors**: Nikita Kachaev, Daniil Zelezetsky, Egor Cherepanov, Alexey K. Kovelev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2510.13367)  

**Abstract**: Despite their effectiveness and popularity in offline or model-based reinforcement learning (RL), transformers remain underexplored in online model-free RL due to their sensitivity to training setups and model design decisions such as how to structure the policy and value networks, share components, or handle temporal information. In this paper, we show that transformers can be strong baselines for continuous control in online model-free RL. We investigate key design questions: how to condition inputs, share components between actor and critic, and slice sequential data for training. Our experiments reveal stable architectural and training strategies enabling competitive performance across fully and partially observable tasks, and in both vector- and image-based settings. These findings offer practical guidance for applying transformers in online RL. 

**Abstract (ZH)**: 尽管变压器在离线或模型依赖的强化学习（RL）中表现出色且广受欢迎，但在在线模型自由的RL中仍受到探索不足，主要是由于其对训练设置和模型设计决策（如如何结构化策略和价值网络、共享组件或处理时间信息）的高度敏感性。本文展示了变压器可以作为在线模型自由RL中连续控制的有效基线。我们探讨了关键的设计问题：如何条件化输入、在演员和评论家之间共享组件以及在训练中切分序列数据。我们的实验揭示了稳定且有效的架构和训练策略，使其在完全可观测和部分可观测任务中以及向量和图像基设置中均能获得竞争力的性能。这些发现为在在线RL中应用变压器提供了实用指导。 

---
# DriveCritic: Towards Context-Aware, Human-Aligned Evaluation for Autonomous Driving with Vision-Language Models 

**Title (ZH)**: DriveCritic: 面向情境感知与人类价值观对齐的自动驾驶评估方法研究 

**Authors**: Jingyu Song, Zhenxin Li, Shiyi Lan, Xinglong Sun, Nadine Chang, Maying Shen, Joshua Chen, Katherine A. Skinner, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2510.13108)  

**Abstract**: Benchmarking autonomous driving planners to align with human judgment remains a critical challenge, as state-of-the-art metrics like the Extended Predictive Driver Model Score (EPDMS) lack context awareness in nuanced scenarios. To address this, we introduce DriveCritic, a novel framework featuring two key contributions: the DriveCritic dataset, a curated collection of challenging scenarios where context is critical for correct judgment and annotated with pairwise human preferences, and the DriveCritic model, a Vision-Language Model (VLM) based evaluator. Fine-tuned using a two-stage supervised and reinforcement learning pipeline, the DriveCritic model learns to adjudicate between trajectory pairs by integrating visual and symbolic context. Experiments show DriveCritic significantly outperforms existing metrics and baselines in matching human preferences and demonstrates strong context awareness. Overall, our work provides a more reliable, human-aligned foundation to evaluating autonomous driving systems. 

**Abstract (ZH)**: 基于DriveCritic框架评估自主驾驶规划者以与人类判断一致仍是一项关键挑战，现有先进的指标如扩展预测驾驶模型评分（EPDMS）在细腻场景中缺乏上下文意识。为解决这一问题，我们引入了DriveCritic，这是一个包含两项关键贡献的新框架：DriveCritic数据集，这是一个经过精心筛选的包含关键上下文场景的集合，并标注了两两的人类偏好；以及DriveCritic模型，这是一个基于视觉-语言模型（VLM）的评估器。通过两阶段监督学习和强化学习管道进行微调后，DriveCritic模型学习通过整合视觉和符号上下文来判断轨迹对。实验表明，DriveCritic在匹配人类偏好和展现强大的上下文意识方面显著优于现有指标和基线。总体而言，我们的工作为评估自主驾驶系统提供了一个更加可靠、与人类判断一致的基础。 

---
# SAJA: A State-Action Joint Attack Framework on Multi-Agent Deep Reinforcement Learning 

**Title (ZH)**: SAJA：多智能体深度强化学习中的状态-动作联合攻击框架 

**Authors**: Weiqi Guo, Guanjun Liu, Ziyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.13262)  

**Abstract**: Multi-Agent Deep Reinforcement Learning (MADRL) has shown potential for cooperative and competitive tasks such as autonomous driving and strategic gaming. However, models trained by MADRL are vulnerable to adversarial perturbations on states and actions. Therefore, it is essential to investigate the robustness of MADRL models from an attack perspective. Existing studies focus on either state-only attacks or action-only attacks, but do not consider how to effectively joint them. Simply combining state and action perturbations such as randomly perturbing states and actions does not exploit their potential synergistic effects. In this paper, we propose the State-Action Joint Attack (SAJA) framework that has a good synergistic effects. SAJA consists of two important phases: (1) In the state attack phase, a multi-step gradient ascent method utilizes both the actor network and the critic network to compute an adversarial state, and (2) in the action attack phase, based on the perturbed state, a second gradient ascent uses the critic network to craft the final adversarial action. Additionally, a heuristic regularizer measuring the distance between the perturbed actions and the original clean ones is added into the loss function to enhance the effectiveness of the critic's guidance. We evaluate SAJA in the Multi-Agent Particle Environment (MPE), demonstrating that (1) it outperforms and is more stealthy than state-only or action-only attacks, and (2) existing state or action defense methods cannot defend its attacks. 

**Abstract (ZH)**: 基于状态-动作联合攻击的多智能体深度强化学习robustness研究 

---
# Emotional Cognitive Modeling Framework with Desire-Driven Objective Optimization for LLM-empowered Agent in Social Simulation 

**Title (ZH)**: 基于欲望驱动目标优化的情感认知建模框架在社会模拟中的应用——赋能于大规模语言模型的代理 

**Authors**: Qun Ma, Xiao Xue, Xuwen Zhang, Zihan Zhao, Yuwei Guo, Ming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.13195)  

**Abstract**: The advent of large language models (LLMs) has enabled agents to represent virtual humans in societal simulations, facilitating diverse interactions within complex social systems. However, existing LLM-based agents exhibit severe limitations in affective cognition: They fail to simulate the bounded rationality essential for bridging virtual and real-world services; They lack empirically validated integration mechanisms embedding emotions within agent decision architectures. This paper constructs an emotional cognition framework incorporating desire generation and objective management, designed to achieve emotion alignment between LLM-based agents and humans, modeling the complete decision-making process of LLM-based agents, encompassing state evolution, desire generation, objective optimization, decision generation, and action execution. This study implements the proposed framework within our proprietary multi-agent interaction environment. Experimental results demonstrate that agents governed by our framework not only exhibit behaviors congruent with their emotional states but also, in comparative assessments against other agent types, demonstrate superior ecological validity and generate decision outcomes that significantly more closely approximate human behavioral patterns. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现使代理能够代表虚拟人类参与社会模拟，促进了复杂社会系统中多样化互动的可能性。然而，现有的基于LLM的代理在情感认知方面存在严重限制：它们无法模拟将虚拟与现实世界服务衔接所需的有限理性；缺乏将情绪嵌入代理决策架构中的经验证实整合机制。本文构建了一个包含欲望生成和目标管理的情感认知框架，旨在实现基于LLM的代理与人类之间的情感对齐，模型了基于LLM的代理完整的决策过程，涵盖状态演化、欲望生成、目标优化、决策生成和动作执行。本文在我们自主开发的多代理交互环境中实现了所提出框架。实验结果表明，受该框架指导的代理不仅表现出与其情感状态一致的行为，而且与其它代理类型相比，表现出更高的生态效度，并生成决策结果与人类行为模式更为接近。 

---
# SENTINEL: A Multi-Level Formal Framework for Safety Evaluation of LLM-based Embodied Agents 

**Title (ZH)**: 哨兵：基于大语言模型的体态智能体安全评估的多层次形式化框架 

**Authors**: Simon Sinong Zhan, Yao Liu, Philip Wang, Zinan Wang, Qineng Wang, Zhian Ruan, Xiangyu Shi, Xinyu Cao, Frank Yang, Kangrui Wang, Huajie Shao, Manling Li, Qi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.12985)  

**Abstract**: We present Sentinel, the first framework for formally evaluating the physical safety of Large Language Model(LLM-based) embodied agents across the semantic, plan, and trajectory levels. Unlike prior methods that rely on heuristic rules or subjective LLM judgments, Sentinel grounds practical safety requirements in formal temporal logic (TL) semantics that can precisely specify state invariants, temporal dependencies, and timing constraints. It then employs a multi-level verification pipeline where (i) at the semantic level, intuitive natural language safety requirements are formalized into TL formulas and the LLM agent's understanding of these requirements is probed for alignment with the TL formulas; (ii) at the plan level, high-level action plans and subgoals generated by the LLM agent are verified against the TL formulas to detect unsafe plans before execution; and (iii) at the trajectory level, multiple execution trajectories are merged into a computation tree and efficiently verified against physically-detailed TL specifications for a final safety check. We apply Sentinel in VirtualHome and ALFRED, and formally evaluate multiple LLM-based embodied agents against diverse safety requirements. Our experiments show that by grounding physical safety in temporal logic and applying verification methods across multiple levels, Sentinel provides a rigorous foundation for systematically evaluating LLM-based embodied agents in physical environments, exposing safety violations overlooked by previous methods and offering insights into their failure modes. 

**Abstract (ZH)**: Sentinel: 一种用于正式评估基于大型语言模型的具身代理物理安全性的一体化框架 

---
# DeepPlanner: Scaling Planning Capability for Deep Research Agents via Advantage Shaping 

**Title (ZH)**: DeepPlanner: 通过优势塑造扩展深入研究代理的规划能力 

**Authors**: Wei Fan, Wenlin Yao, Zheng Li, Feng Yao, Xin Liu, Liang Qiu, Qingyu Yin, Yangqiu Song, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2510.12979)  

**Abstract**: Large language models (LLMs) augmented with multi-step reasoning and action generation abilities have shown promise in leveraging external tools to tackle complex tasks that require long-horizon planning. However, existing approaches either rely on implicit planning in the reasoning stage or introduce explicit planners without systematically addressing how to optimize the planning stage. As evidence, we observe that under vanilla reinforcement learning (RL), planning tokens exhibit significantly higher entropy than other action tokens, revealing uncertain decision points that remain under-optimized. To address this, we propose DeepPlanner, an end-to-end RL framework that effectively enhances the planning capabilities of deep research agents. Our approach shapes token-level advantage with an entropy-based term to allocate larger updates to high entropy tokens, and selectively upweights sample-level advantages for planning-intensive rollouts. Extensive experiments across seven deep research benchmarks demonstrate that DeepPlanner improves planning quality and achieves state-of-the-art results under a substantially lower training budget. 

**Abstract (ZH)**: 增强多步推理和动作生成能力的大语言模型通过利用外部工具解决需要长期规划的复杂任务显示出潜力。然而，现有方法要么依赖于推理阶段的隐式规划，要么引入显式的规划器但没有系统地解决如何优化规划阶段的问题。为了解决这个问题，我们提出了DeepPlanner，一个端到端的 reinforcement learning 框架，有效提高了深度研究代理的规划能力。我们的方法通过基于熵的项塑造 token 级别的优势，为高熵 token 分配更大的更新，并有选择性地提升规划密集型 rollout 的样本级别优势。在七个深度研究基准上的广泛实验表明，DeepPlanner 提高了规划质量，并在显著降低的训练预算下达到了最先进的结果。 

---
# Provably Invincible Adversarial Attacks on Reinforcement Learning Systems: A Rate-Distortion Information-Theoretic Approach 

**Title (ZH)**: provably不可战胜的对抗攻击：强化学习系统的率--distortion信息论方法 

**Authors**: Ziqing Lu, Lifeng Lai, Weiyu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.13792)  

**Abstract**: Reinforcement learning (RL) for the Markov Decision Process (MDP) has emerged in many security-related applications, such as autonomous driving, financial decisions, and drone/robot algorithms. In order to improve the robustness/defense of RL systems against adversaries, studying various adversarial attacks on RL systems is very important. Most previous work considered deterministic adversarial attack strategies in MDP, which the recipient (victim) agent can defeat by reversing the deterministic attacks. In this paper, we propose a provably ``invincible'' or ``uncounterable'' type of adversarial attack on RL. The attackers apply a rate-distortion information-theoretic approach to randomly change agents' observations of the transition kernel (or other properties) so that the agent gains zero or very limited information about the ground-truth kernel (or other properties) during the training. We derive an information-theoretic lower bound on the recipient agent's reward regret and show the impact of rate-distortion attacks on state-of-the-art model-based and model-free algorithms. We also extend this notion of an information-theoretic approach to other types of adversarial attack, such as state observation attacks. 

**Abstract (ZH)**: 强化学习（RL）在马尔可夫决策过程（MDP）中的应用已 emergence 在许多安全相关领域，如自动驾驶、金融决策和无人机/机器人算法中。为了提高 RL 系统的健壮性/防御能力以对抗对手，研究各种对 RL 系统的对手攻击变得非常关键。大多数先前的工作考虑了在 MDP 中的确定性对手攻击策略，接受者（受害）代理可以通过逆转确定性攻击来击败这些攻击。在本文中，我们提出了一种可证明的“无敌”或“无法计数”的对手攻击类型。攻击者采用率失真信息论方法，随机改变代理对转移内核（或其他属性）的观察，使得代理在训练过程中几乎获得零或非常有限关于真实内核（或其他属性）的信息。我们推导出接受者代理的奖励后悔的信息论下界，并展示了率失真攻击对基于模型和无模型算法的影响。我们还将这一信息论方法的概念扩展到其他类型的对手攻击，如状态观察攻击。 

---
# AOAD-MAT: Transformer-based multi-agent deep reinforcement learning model considering agents' order of action decisions 

**Title (ZH)**: AOAD-MAT：基于变压器的多智能体深度强化学习模型，考虑动作决策顺序问题 

**Authors**: Shota Takayama, Katsuhide Fujita  

**Link**: [PDF](https://arxiv.org/pdf/2510.13343)  

**Abstract**: Multi-agent reinforcement learning focuses on training the behaviors of multiple learning agents that coexist in a shared environment. Recently, MARL models, such as the Multi-Agent Transformer (MAT) and ACtion dEpendent deep Q-learning (ACE), have significantly improved performance by leveraging sequential decision-making processes. Although these models can enhance performance, they do not explicitly consider the importance of the order in which agents make decisions. In this paper, we propose an Agent Order of Action Decisions-MAT (AOAD-MAT), a novel MAT model that considers the order in which agents make decisions. The proposed model explicitly incorporates the sequence of action decisions into the learning process, allowing the model to learn and predict the optimal order of agent actions. The AOAD-MAT model leverages a Transformer-based actor-critic architecture that dynamically adjusts the sequence of agent actions. To achieve this, we introduce a novel MARL architecture that cooperates with a subtask focused on predicting the next agent to act, integrated into a Proximal Policy Optimization based loss function to synergistically maximize the advantage of the sequential decision-making. The proposed method was validated through extensive experiments on the StarCraft Multi-Agent Challenge and Multi-Agent MuJoCo benchmarks. The experimental results show that the proposed AOAD-MAT model outperforms existing MAT and other baseline models, demonstrating the effectiveness of adjusting the AOAD order in MARL. 

**Abstract (ZH)**: 多智能体强化学习专注于训练多个共存于共享环境中的学习智能体的行为。近年来，借助序列决策过程，如多智能体变压器（MAT）和基于动作相关的深度Q学习（ACE）等MARL模型显著提升了性能。尽管这些模型能够增强性能，但它们并未明确考虑到智能体决策顺序的重要性。本文提出了一种考虑智能体决策顺序的多智能体强化学习模型——行动决策智能体顺序多智能体变压器（AOAD-MAT），这是一种新颖的MAT模型。该模型明确地将行动决策的顺序纳入学习过程中，从而使模型能够学习并预测最优的智能体行动顺序。AOAD-MAT模型利用基于Transformer的演员-评论家架构，动态调整智能体行动序列。为此，本文提出了一个与预测下一个行动智能体的子任务协同工作的新颖MARL架构，并将其整合进基于近端策略优化的损失函数中，以最大化序列决策的优势。通过在StarCraft多智能体挑战和多智能体MuJoCo基准上的广泛实验对所提出的方法进行了验证，实验结果表明，提出的AOAD-MAT模型优于现有的MAT模型和其他基线模型，证明了在MARL中调整AOAD顺序的有效性。 

---
# MotionBeat: Motion-Aligned Music Representation via Embodied Contrastive Learning and Bar-Equivariant Contact-Aware Encoding 

**Title (ZH)**: MotionBeat：通过身体对比学习和小节不变的接触感知编码的运动对齐音乐表示 

**Authors**: Xuanchen Wang, Heng Wang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2510.13244)  

**Abstract**: Music is both an auditory and an embodied phenomenon, closely linked to human motion and naturally expressed through dance. However, most existing audio representations neglect this embodied dimension, limiting their ability to capture rhythmic and structural cues that drive movement. We propose MotionBeat, a framework for motion-aligned music representation learning. MotionBeat is trained with two newly proposed objectives: the Embodied Contrastive Loss (ECL), an enhanced InfoNCE formulation with tempo-aware and beat-jitter negatives to achieve fine-grained rhythmic discrimination, and the Structural Rhythm Alignment Loss (SRAL), which ensures rhythm consistency by aligning music accents with corresponding motion events. Architecturally, MotionBeat introduces bar-equivariant phase rotations to capture cyclic rhythmic patterns and contact-guided attention to emphasize motion events synchronized with musical accents. Experiments show that MotionBeat outperforms state-of-the-art audio encoders in music-to-dance generation and transfers effectively to beat tracking, music tagging, genre and instrument classification, emotion recognition, and audio-visual retrieval. Our project demo page: this https URL. 

**Abstract (ZH)**: 音乐既是听觉的也是身体的现象，紧密关联人类运动，并自然地通过舞蹈表达。然而，现有的大多数音频表示忽视了这种身体维度，限制了它们捕捉驱动运动的节奏和结构线索的能力。我们提出了MotionBeat，一种用于运动对齐的音乐表示学习框架。MotionBeat通过两种新提出的优化目标进行训练：Body-aware Contrastive Loss（BCL），这是增强的InfoNCE公式，带有节拍感知和节拍抖动的负样本，以实现精细的节奏辨别；以及Structural Rhythm Alignment Loss（SRAL），通过使音乐重音与相应的运动事件对齐以确保节奏一致性。从架构上看，MotionBeat引入了小节等变相位旋转以捕捉周期性节奏模式，并使用基于接触的注意力以强调与音乐重音同步的运动事件。实验结果显示，MotionBeat在音乐到舞蹈生成中优于现有的音频编码器，并且能够有效转移至节拍跟踪、音乐标签化、流派和乐器分类、情绪识别以及音频视觉检索。我们的项目演示页面：this https URL。 

---
# StressTransfer: Stress-Aware Speech-to-Speech Translation with Emphasis Preservation 

**Title (ZH)**: 应力传递：注重强调的感知压力语音到语音翻译 

**Authors**: Xi Chen, Yuchen Song, Satoshi Nakamura  

**Link**: [PDF](https://arxiv.org/pdf/2510.13194)  

**Abstract**: We propose a stress-aware speech-to-speech translation (S2ST) system that preserves word-level emphasis by leveraging LLMs for cross-lingual emphasis conversion. Our method translates source-language stress into target-language tags that guide a controllable TTS model. To overcome data scarcity, we developed a pipeline to automatically generate aligned training data and introduce the "LLM-as-Judge" for evaluation. Experiments show our approach substantially outperforms baselines in preserving emphasis while maintaining comparable translation quality, speaker intent, and naturalness. Our work highlights the importance of prosody in translation and provides an effective, data-efficient solution for preserving paralinguistic cues in S2ST. 

**Abstract (ZH)**: 一种基于大语言模型的意识应力语音到语音翻译系统：通过跨语言应力转换保留词级强调 

---
# Agentic Discovery: Closing the Loop with Cooperative Agents 

**Title (ZH)**: 代理发现：与协同代理形成闭环 

**Authors**: J. Gregory Pauloski, Kyle Chard, Ian T. Foster  

**Link**: [PDF](https://arxiv.org/pdf/2510.13081)  

**Abstract**: As data-driven methods, artificial intelligence (AI), and automated workflows accelerate scientific tasks, we see the rate of discovery increasingly limited by human decision-making tasks such as setting objectives, generating hypotheses, and designing experiments. We postulate that cooperative agents are needed to augment the role of humans and enable autonomous discovery. Realizing such agents will require progress in both AI and infrastructure. 

**Abstract (ZH)**: 随着数据驱动方法、人工智能（AI）和自动化工作流程加速科学研究任务，我们发现发现率越来越受到人类决策任务，如设定目标、生成假设和设计实验的限制。我们假设需要合作代理来增强人类的作用，以实现自主发现。实现这样的代理将需要在人工智能和基础设施两个方面取得进展。 

---
# Epistemic-aware Vision-Language Foundation Model for Fetal Ultrasound Interpretation 

**Title (ZH)**: 知觉意识导向的视觉-语言基础模型在胎儿超声解释中的应用 

**Authors**: Xiao He, Huangxuan Zhao, Guojia Wan, Wei Zhou, Yanxing Liu, Juhua Liu, Yongchao Xu, Yong Luo, Dacheng Tao, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.12953)  

**Abstract**: Recent medical vision-language models have shown promise on tasks such as VQA, report generation, and anomaly detection. However, most are adapted to structured adult imaging and underperform in fetal ultrasound, which poses challenges of multi-view image reasoning, numerous diseases, and image diversity. To bridge this gap, we introduce FetalMind, a medical AI system tailored to fetal ultrasound for both report generation and diagnosis. Guided by clinical workflow, we propose Salient Epistemic Disentanglement (SED), which injects an expert-curated bipartite graph into the model to decouple view-disease associations and to steer preference selection along clinically faithful steps via reinforcement learning. This design mitigates variability across diseases and heterogeneity across views, reducing learning bottlenecks while aligning the model's inference with obstetric practice. To train FetalMind at scale, we curate FetalSigma-1M dataset, the first large-scale fetal ultrasound report corpus, comprising 20K reports from twelve medical centers, addressing the scarcity of domain data. Extensive experiments show that FetalMind outperforms open- and closed-source baselines across all gestational stages, achieving +14% average gains and +61.2% higher accuracy on critical conditions while remaining efficient, stable, and scalable. Project Page: this https URL. 

**Abstract (ZH)**: _recent医疗视觉语言模型在VQA、报告生成和异常检测等任务上展示了潜力。然而，大多数模型适应于结构化的成人影像，在胎儿超声波成像中表现不佳，这带来了多视角图像推理、多种疾病和图像多样性等方面的挑战。为解决这一问题，我们提出了FetalMind，一种针对胎儿超声波成像的医疗AI系统，用于报告生成和诊断。受临床工作流程的指导，我们提出了显着表征脱耦（SED）方法，该方法通过强化学习将专家编纂的二分图注入模型中，以解除视角-疾病关联，并沿着临床忠实的步骤引导偏好选择。该设计减少了疾病间的变异性以及视角间的异质性，降低了学习瓶颈，同时使模型的推断与产科实践保持一致。为了大规模训练FetalMind，我们构建了FetalSigma-1M数据集，这是首个大型胎儿超声波报告语料库，包含来自十二家医疗机构的20000份报告，解决了领域数据稀缺的问题。大规模实验表明，FetalMind在所有妊娠阶段的表现均优于开源和闭源基线，关键条件下准确率提高了61.2%，且保持高效、稳定和可扩展。_ 

---
