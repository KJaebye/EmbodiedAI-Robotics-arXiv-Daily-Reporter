# MoManipVLA: Transferring Vision-language-action Models for General Mobile Manipulation 

**Title (ZH)**: MoManipVLA: 将视觉-语言-动作模型应用于通用移动操作 

**Authors**: Zhenyu Wu, Yuheng Zhou, Xiuwei Xu, Ziwei Wang, Haibin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13446)  

**Abstract**: Mobile manipulation is the fundamental challenge for robotics to assist humans with diverse tasks and environments in everyday life. However, conventional mobile manipulation approaches often struggle to generalize across different tasks and environments because of the lack of large-scale training. In contrast, recent advances in vision-language-action (VLA) models have shown impressive generalization capabilities, but these foundation models are developed for fixed-base manipulation tasks. Therefore, we propose an efficient policy adaptation framework named MoManipVLA to transfer pre-trained VLA models of fix-base manipulation to mobile manipulation, so that high generalization ability across tasks and environments can be achieved in mobile manipulation policy. Specifically, we utilize pre-trained VLA models to generate waypoints of the end-effector with high generalization ability. We design motion planning objectives for the mobile base and the robot arm, which aim at maximizing the physical feasibility of the trajectory. Finally, we present an efficient bi-level objective optimization framework for trajectory generation, where the upper-level optimization predicts waypoints for base movement to enhance the manipulator policy space, and the lower-level optimization selects the optimal end-effector trajectory to complete the manipulation task. In this way, MoManipVLA can adjust the position of the robot base in a zero-shot manner, thus making the waypoints predicted from the fixed-base VLA models feasible. Extensive experimental results on OVMM and the real world demonstrate that MoManipVLA achieves a 4.2% higher success rate than the state-of-the-art mobile manipulation, and only requires 50 training cost for real world deployment due to the strong generalization ability in the pre-trained VLA models. 

**Abstract (ZH)**: 移动操作是机器人学在日常生活中协助人类完成多样化任务和环境挑战的基础难题。然而，传统的移动操作方法由于缺乏大规模训练数据，往往难以在不同任务和环境中泛化。相比之下，最近在视觉-语言-动作（VLA）模型方面取得的进展展示了卓越的泛化能力，但这些基础模型主要用于固定基座操作任务。因此，我们提出了一种高效的操作政策适应框架MoManipVLA，将预训练的固定基座操作的VLA模型转移到移动操作中，从而在移动操作策略中实现跨任务和环境的高泛化能力。具体而言，我们利用预训练的VLA模型生成具有高泛化能力的末端执行器航点。我们为移动基座和机器人臂设计了运动规划目标，旨在最大化轨迹的物理可行性。最后，我们提出了一种高效的双层目标优化框架用于轨迹生成，其中上层优化预测用于基座移动的航点以增强操作器策略空间，下层优化选择最优末端执行器轨迹以完成操作任务。通过这种方式，MoManipVLA可以在零样本情况下调整机器人基座的位置，从而使固定基座VLA模型预测的航点变得可行。广泛的实验结果在OVMM和实际环境中表明，MoManipVLA相较于最先进的移动操作实现了4.2%更高的成功率，并且由于预训练VLA模型的强大泛化能力，仅需50次训练成本即可实现实际部署。 

---
# Humanoid Policy ~ Human Policy 

**Title (ZH)**: 类人政策 ~ 人类政策 

**Authors**: Ri-Zhao Qiu, Shiqi Yang, Xuxin Cheng, Chaitanya Chawla, Jialong Li, Tairan He, Ge Yan, Lars Paulsen, Ge Yang, Sha Yi, Guanya Shi, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13441)  

**Abstract**: Training manipulation policies for humanoid robots with diverse data enhances their robustness and generalization across tasks and platforms. However, learning solely from robot demonstrations is labor-intensive, requiring expensive tele-operated data collection which is difficult to scale. This paper investigates a more scalable data source, egocentric human demonstrations, to serve as cross-embodiment training data for robot learning. We mitigate the embodiment gap between humanoids and humans from both the data and modeling perspectives. We collect an egocentric task-oriented dataset (PH2D) that is directly aligned with humanoid manipulation demonstrations. We then train a human-humanoid behavior policy, which we term Human Action Transformer (HAT). The state-action space of HAT is unified for both humans and humanoid robots and can be differentiably retargeted to robot actions. Co-trained with smaller-scale robot data, HAT directly models humanoid robots and humans as different embodiments without additional supervision. We show that human data improves both generalization and robustness of HAT with significantly better data collection efficiency. Code and data: this https URL 

**Abstract (ZH)**: 使用多样数据训练人形机器人操作策略可以增强其在不同任务和平台上的鲁棒性和泛化能力。然而，仅从机器人示范中学习劳动密集且难以大规模扩展，需要昂贵的远程操作数据收集。本文研究了一种更具扩展性的数据源——以人类为中心的示范，作为机器人学习的跨体态训练数据。我们从数据和建模两个方面减轻了人形机器人和人类之间的体态差距。我们收集了一个以任务为导向的以自我中心视角的数据集（PH2D），直接与人形机器人操作示范对齐。然后，我们训练了一个人类-人形机器人行为策略，称为Human Action Transformer（HAT）。HAT的状态-动作空间对人类和人形机器人统一，并可通过梯度重新瞄准到机器人的动作。HAT与较小规模的机器人数据共同训练，在无需额外监督的情况下直接建模人类和人形机器人作为不同的体态。实验证明，人类数据显著提高了HAT的泛化能力和鲁棒性，同时数据收集效率更高。代码和数据：这个 https URL。 

---
# FLEX: A Framework for Learning Robot-Agnostic Force-based Skills Involving Sustained Contact Object Manipulation 

**Title (ZH)**: FLEX：一种用于学习与物体接触相关的力基技能的机器人无关框架 

**Authors**: Shijie Fang, Wenchang Gao, Shivam Goel, Christopher Thierauf, Matthias Scheutz, Jivko Sinapov  

**Link**: [PDF](https://arxiv.org/pdf/2503.13418)  

**Abstract**: Learning to manipulate objects efficiently, particularly those involving sustained contact (e.g., pushing, sliding) and articulated parts (e.g., drawers, doors), presents significant challenges. Traditional methods, such as robot-centric reinforcement learning (RL), imitation learning, and hybrid techniques, require massive training and often struggle to generalize across different objects and robot platforms. We propose a novel framework for learning object-centric manipulation policies in force space, decoupling the robot from the object. By directly applying forces to selected regions of the object, our method simplifies the action space, reduces unnecessary exploration, and decreases simulation overhead. This approach, trained in simulation on a small set of representative objects, captures object dynamics -- such as joint configurations -- allowing policies to generalize effectively to new, unseen objects. Decoupling these policies from robot-specific dynamics enables direct transfer to different robotic platforms (e.g., Kinova, Panda, UR5) without retraining. Our evaluations demonstrate that the method significantly outperforms baselines, achieving over an order of magnitude improvement in training efficiency compared to other state-of-the-art methods. Additionally, operating in force space enhances policy transferability across diverse robot platforms and object types. We further showcase the applicability of our method in a real-world robotic setting. For supplementary materials and videos, please visit: this https URL 

**Abstract (ZH)**: 学会高效操作物体，特别是涉及持续接触（如推、滑动）和articulated部分（如抽屉、门）的物体，提出了重大挑战。传统的机器人中心强化学习（RL）、模仿学习和混合技术方法需要大量训练，并且在跨不同物体和机器人平台的泛化方面往往存在困难。我们提出了一种新的针对物体的操纵策略学习框架，在力空间中解耦机器人和物体。通过直接对物体的选定区域施加力，该方法简化了动作空间，减少了不必要的探索，并降低了仿真开销。这种方法在模拟中使用少量代表性物体进行训练，捕获物体动力学（如关节配置）的信息，使得策略能够有效地泛化到新未见过的物体上。解耦这些策略从机器人特定的动力学中分离出来，使它们可以直接转移到不同的机器人平台（如Kinova、Panda、UR5）上而无需重新训练。我们的评估表明，该方法显著优于基线方法，在培训效率上比其他最先进技术提高了十倍以上。此外，在力空间操作提升了策略在多种机器人平台和物体类型之间的迁移性。我们进一步展示了该方法在实际机器人环境中的适用性。更多补充材料和视频，请访问：this https URL 

---
# Artificial Spacetimes for Reactive Control of Resource-Limited Robots 

**Title (ZH)**: 资源受限机器人反应控制的人工时空 

**Authors**: William H. Reinhardt, Marc Z. Miskin  

**Link**: [PDF](https://arxiv.org/pdf/2503.13355)  

**Abstract**: Field-based reactive control provides a minimalist, decentralized route to guiding robots that lack onboard computation. Such schemes are well suited to resource-limited machines like microrobots, yet implementation artifacts, limited behaviors, and the frequent lack of formal guarantees blunt adoption. Here, we address these challenges with a new geometric approach called artificial spacetimes. We show that reactive robots navigating control fields obey the same dynamics as light rays in general relativity. This surprising connection allows us to adopt techniques from relativity and optics for constructing and analyzing control fields. When implemented, artificial spacetimes guide robots around structured environments, simultaneously avoiding boundaries and executing tasks like rallying or sorting, even when the field itself is static. We augment these capabilities with formal tools for analyzing what robots will do and provide experimental validation with silicon-based microrobots. Combined, this work provides a new framework for generating composed robot behaviors with minimal overhead. 

**Abstract (ZH)**: 基于场的反应式控制为缺乏机载计算能力的机器人提供了一种简约且去中心化的引导途径。这种方案适用于资源受限的机器人体积微小机器人，然而实现中的缺陷、行为限制以及缺乏正式保证阻碍了其应用。在这里，我们通过一种新的几何方法——人工时空，解决了这些挑战。我们展示了导航控制场的反应式机器人遵循广义相对论中光束的动力学。这一令人惊讶的联系使我们能够采用相对论和光学技术来构建和分析控制场。当实施时，人工时空能够引导机器人绕过结构化的环境，同时避开边界并执行诸如聚集或分类等任务，即使控制场本身是静态的。我们还通过正式工具分析了机器人的行为，并使用基于硅的微小机器人进行了实验验证。综合来看，这项工作提供了一种新的框架，用于生成具有最小开销的组合机器人行为。 

---
# Digital Beamforming Enhanced Radar Odometry 

**Title (ZH)**: 数字波束形成增强雷达里程计 

**Authors**: Jingqi Jiang, Shida Xu, Kaicheng Zhang, Jiyuan Wei, Jingyang Wang, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13252)  

**Abstract**: Radar has become an essential sensor for autonomous navigation, especially in challenging environments where camera and LiDAR sensors fail. 4D single-chip millimeter-wave radar systems, in particular, have drawn increasing attention thanks to their ability to provide spatial and Doppler information with low hardware cost and power consumption. However, most single-chip radar systems using traditional signal processing, such as Fast Fourier Transform, suffer from limited spatial resolution in radar detection, significantly limiting the performance of radar-based odometry and Simultaneous Localization and Mapping (SLAM) systems. In this paper, we develop a novel radar signal processing pipeline that integrates spatial domain beamforming techniques, and extend it to 3D Direction of Arrival estimation. Experiments using public datasets are conducted to evaluate and compare the performance of our proposed signal processing pipeline against traditional methodologies. These tests specifically focus on assessing structural precision across diverse scenes and measuring odometry accuracy in different radar odometry systems. This research demonstrates the feasibility of achieving more accurate radar odometry by simply replacing the standard FFT-based processing with the proposed pipeline. The codes are available at GitHub*. 

**Abstract (ZH)**: 雷达已成为自主导航不可或缺的传感器，尤其是在摄像机和LiDAR传感器失效的挑战性环境中。特别是，4D单芯片毫米波雷达系统因其实现低硬件成本和能耗下提供空间和多普勒信息的能力而日益受到关注。然而，大多数使用传统信号处理方法（如快速傅里叶变换）的单芯片雷达系统在雷达检测中受到有限的空间分辨率限制，显著限制了基于雷达的里程计和同步定位与建图（SLAM）系统的性能。本文开发了一种新颖的雷达信号处理管道，结合了空间域波束形成技术，并将其扩展到3D到达方向估计。利用公开数据集进行实验以评估和比较我们提出的信号处理管道与传统方法的性能。这些测试特别关注在多种场景下结构精度的评估和在不同雷达里程计系统中测量里程计精度。研究表明，仅通过将标准FFT处理替换为提出的管道即可实现更精确的雷达里程计。相关代码可在GitHub*上获取。 

---
# MindEye-OmniAssist: A Gaze-Driven LLM-Enhanced Assistive Robot System for Implicit Intention Recognition and Task Execution 

**Title (ZH)**: MindEye-OmniAssist：一种基于凝视的LLM增强辅助机器人系统，用于隐含意图识别和任务执行 

**Authors**: Zejia Zhang, Bo Yang, Xinxing Chen, Weizhuang Shi, Haoyuan Wang, Wei Luo, Jian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13250)  

**Abstract**: A promising effective human-robot interaction in assistive robotic systems is gaze-based control. However, current gaze-based assistive systems mainly help users with basic grasping actions, offering limited support. Moreover, the restricted intent recognition capability constrains the assistive system's ability to provide diverse assistance functions. In this paper, we propose an open implicit intention recognition framework powered by Large Language Model (LLM) and Vision Foundation Model (VFM), which can process gaze input and recognize user intents that are not confined to predefined or specific scenarios. Furthermore, we implement a gaze-driven LLM-enhanced assistive robot system (MindEye-OmniAssist) that recognizes user's intentions through gaze and assists in completing task. To achieve this, the system utilizes open vocabulary object detector, intention recognition network and LLM to infer their full intentions. By integrating eye movement feedback and LLM, it generates action sequences to assist the user in completing tasks. Real-world experiments have been conducted for assistive tasks, and the system achieved an overall success rate of 41/55 across various undefined tasks. Preliminary results show that the proposed method holds the potential to provide a more user-friendly human-computer interaction interface and significantly enhance the versatility and effectiveness of assistive systems by supporting more complex and diverse task. 

**Abstract (ZH)**: 一种由大规模语言模型和视觉基础模型驱动的开放隐含意图识别框架在辅助机器人系统中的凝视控制交互颇有前景。该框架能够处理凝视输入并识别不受预定义或特定场景限制的用户意图，并通过凝视驱动的大规模语言模型增强辅助机器人系统（MindEye-OmniAssist）来识别用户的意图并辅助完成任务。实验结果显示，在各类未定义任务中，该系统整体成功率为41/55。初步结果表明，所提出的方法具有为用户提供更友好的人机交互界面并显著增强辅助系统多样性和有效性、支持更多复杂和多样的任务的潜力。 

---
# Dense Policy: Bidirectional Autoregressive Learning of Actions 

**Title (ZH)**: 密集策略：双向自回归动作学习 

**Authors**: Yue Su, Xinyu Zhan, Hongjie Fang, Han Xue, Hao-Shu Fang, Yong-Lu Li, Cewu Lu, Lixin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13217)  

**Abstract**: Mainstream visuomotor policies predominantly rely on generative models for holistic action prediction, while current autoregressive policies, predicting the next token or chunk, have shown suboptimal results. This motivates a search for more effective learning methods to unleash the potential of autoregressive policies for robotic manipulation. This paper introduces a bidirectionally expanded learning approach, termed Dense Policy, to establish a new paradigm for autoregressive policies in action prediction. It employs a lightweight encoder-only architecture to iteratively unfold the action sequence from an initial single frame into the target sequence in a coarse-to-fine manner with logarithmic-time inference. Extensive experiments validate that our dense policy has superior autoregressive learning capabilities and can surpass existing holistic generative policies. Our policy, example data, and training code will be publicly available upon publication. Project page: https: //selenthis http URL. 

**Abstract (ZH)**: 主流的视运动策略主要依赖生成模型进行整体动作预测，而目前的自回归策略预测下一个标记或片段的效果欠佳。这促使我们寻找更有效的学习方法，以充分发挥自回归策略在机器人操作中的潜力。本文提出了一种双向扩展的学习方法，称为密集策略，以建立自回归策略在动作预测中的新范式。该方法采用轻量级的编码器-only架构，以自上而下的方式从初始单帧逐步细化展开动作序列，在对数时间内实现推理。大量实验验证了我们提出的密集策略在自回归学习能力上优于现有的整体生成策略。我们的策略、示例数据和训练代码将在发表后公开。项目页面: https://selenthis http URL。 

---
# HybridGen: VLM-Guided Hybrid Planning for Scalable Data Generation of Imitation Learning 

**Title (ZH)**: HybridGen: 基于VLM的混合规划方法实现可扩展的 imitation 学习数据生成 

**Authors**: Wensheng Wang, Ning Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.13171)  

**Abstract**: The acquisition of large-scale and diverse demonstration data are essential for improving robotic imitation learning generalization. However, generating such data for complex manipulations is challenging in real-world settings. We introduce HybridGen, an automated framework that integrates Vision-Language Model (VLM) and hybrid planning. HybridGen uses a two-stage pipeline: first, VLM to parse expert demonstrations, decomposing tasks into expert-dependent (object-centric pose transformations for precise control) and plannable segments (synthesizing diverse trajectories via path planning); second, pose transformations substantially expand the first-stage data. Crucially, HybridGen generates a large volume of training data without requiring specific data formats, making it broadly applicable to a wide range of imitation learning algorithms, a characteristic which we also demonstrate empirically across multiple algorithms. Evaluations across seven tasks and their variants demonstrate that agents trained with HybridGen achieve substantial performance and generalization gains, averaging a 5% improvement over state-of-the-art methods. Notably, in the most challenging task variants, HybridGen achieves significant improvement, reaching a 59.7% average success rate, significantly outperforming Mimicgen's 49.5%. These results demonstrating its effectiveness and practicality. 

**Abstract (ZH)**: 大规模多样示例数据的获取对于提高机器人模仿学习泛化能力至关重要。然而，在现实环境中生成复杂操作的此类数据具有挑战性。我们引入了HybridGen，这是一种结合了视觉语言模型(VLM)和混合规划的自动化框架。HybridGen采用两阶段流水线：首先使用VLM解析专家演示，将任务分解为专家依赖部分（对象为中心的姿态变换以实现精确控制）和可规划部分（通过路径规划合成多样化轨迹）；其次，姿态变换极大地扩展了第一阶段的数据。至关重要的是，HybridGen能够在不需要特定数据格式的情况下生成大量训练数据，使其广泛适用于多种模仿学习算法，我们也在多个算法上通过实验验证了这一点。跨七个任务及其变体的评估结果表明，使用HybridGen训练的智能体实现了显著的性能和泛化提升，平均提高了5%以上，超过了最新方法。特别地，在最具有挑战性的任务变体中，HybridGen实现了显著的改进，平均成功率达到了59.7%，远超Mimicgen的49.5%。这些结果证明了其有效性和实用性。 

---
# Rapid and Inexpensive Inertia Tensor Estimation from a Single Object Throw 

**Title (ZH)**: 单次物体投掷的快速和经济惯性张量估计 

**Authors**: Till M. Blaha, Mike M. Kuijper, Radu Pop, Ewoud J.J. Smeur  

**Link**: [PDF](https://arxiv.org/pdf/2503.13137)  

**Abstract**: The inertia tensor is an important parameter in many engineering fields, but measuring it can be cumbersome and involve multiple experiments or accurate and expensive equipment. We propose a method to measure the moment of inertia tensor of a rigid body from a single spinning throw, by attaching a small and inexpensive stand-alone measurement device consisting of a gyroscope, accelerometer and a reaction wheel. The method includes a compensation for the increase of moment of inertia due to adding the measurement device to the body, and additionally obtains the location of the centre of gravity of the body as an intermediate result. Experiments performed with known rigid bodies show that the mean accuracy is around 2\%. 

**Abstract (ZH)**: 一种基于单次旋转测量刚体惯性张量的方法 

---
# MIXPINN: Mixed-Material Simulations by Physics-Informed Neural Network 

**Title (ZH)**: MIXPINN: 由物理知情神经网络进行的混合材料仿真 

**Authors**: Xintian Yuan, Yunke Ao, Boqi Chen, Philipp Fuernstahl  

**Link**: [PDF](https://arxiv.org/pdf/2503.13123)  

**Abstract**: Simulating the complex interactions between soft tissues and rigid anatomy is critical for applications in surgical training, planning, and robotic-assisted interventions. Traditional Finite Element Method (FEM)-based simulations, while accurate, are computationally expensive and impractical for real-time scenarios. Learning-based approaches have shown promise in accelerating predictions but have fallen short in modeling soft-rigid interactions effectively. We introduce MIXPINN, a physics-informed Graph Neural Network (GNN) framework for mixed-material simulations, explicitly capturing soft-rigid interactions using graph-based augmentations. Our approach integrates Virtual Nodes (VNs) and Virtual Edges (VEs) to enhance rigid body constraint satisfaction while preserving computational efficiency. By leveraging a graph-based representation of biomechanical structures, MIXPINN learns high-fidelity deformations from FEM-generated data and achieves real-time inference with sub-millimeter accuracy. We validate our method in a realistic clinical scenario, demonstrating superior performance compared to baseline GNN models and traditional FEM methods. Our results show that MIXPINN reduces computational cost by an order of magnitude while maintaining high physical accuracy, making it a viable solution for real-time surgical simulation and robotic-assisted procedures. 

**Abstract (ZH)**: 混合材料仿真中基于图神经网络的物理约束网络（MIXPINN）：显式捕捉软硬组织交互 

---
# LIVEPOINT: Fully Decentralized, Safe, Deadlock-Free Multi-Robot Control in Cluttered Environments with High-Dimensional Inputs 

**Title (ZH)**: LIVEPOINT：在复杂环境中的高维输入下完全去中心化、安全、无死锁多机器人控制 

**Authors**: Jeffrey Chen, Rohan Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2503.13098)  

**Abstract**: Fully decentralized, safe, and deadlock-free multi-robot navigation in dynamic, cluttered environments is a critical challenge in robotics. Current methods require exact state measurements in order to enforce safety and liveness e.g. via control barrier functions (CBFs), which is challenging to achieve directly from onboard sensors like lidars and cameras. This work introduces LIVEPOINT, a decentralized control framework that synthesizes universal CBFs over point clouds to enable safe, deadlock-free real-time multi-robot navigation in dynamic, cluttered environments. Further, LIVEPOINT ensures minimally invasive deadlock avoidance behavior by dynamically adjusting agents' speeds based on a novel symmetric interaction metric. We validate our approach in simulation experiments across highly constrained multi-robot scenarios like doorways and intersections. Results demonstrate that LIVEPOINT achieves zero collisions or deadlocks and a 100% success rate in challenging settings compared to optimization-based baselines such as MPC and ORCA and neural methods such as MPNet, which fail in such environments. Despite prioritizing safety and liveness, LIVEPOINT is 35% smoother than baselines in the doorway environment, and maintains agility in constrained environments while still being safe and deadlock-free. 

**Abstract (ZH)**: 全分布式、安全且无死锁的多机器人在动态杂乱环境中的导航是机器人技术中的关键挑战。当前的方法要求精确的状态测量以通过控制边界函数（CBFs）等方式确保安全性和活跃性，这对从机载传感器如激光雷达和摄像头直接获取是具有挑战性的。本文引入了LIVEPOINT，一种分布式控制框架，通过点云合成通用CBFs以实现动态杂乱环境中多机器人安全且无死锁的实时导航。此外，LIVEPOINT通过基于新颖对称交互度量动态调整代理速度来确保最小侵入性的死锁避免行为。我们在涵盖门道和交叉口等高度限制多机器人场景的仿真实验中验证了该方法。结果显示，LIVEPOINT在具有挑战性的设置中实现了零碰撞或死锁，并且与基于优化的方法（如MPC和ORCA）以及神经方法（如MPNet）相比，成功率达到100%，即使在这些环境中这些方法也会失败。尽管LIVEPOINT优先考虑安全性和活跃性，在门道环境中其平滑度比基线方法高35%，同时在受限环境中保持敏捷性，并且仍然安全且无死锁。 

---
# Multi-Platform Teach-and-Repeat Navigation by Visual Place Recognition Based on Deep-Learned Local Features 

**Title (ZH)**: 基于深度学习局部特征的跨平台教姿学复现导航 

**Authors**: Václav Truhlařík, Tomáš Pivoňka, Michal Kasarda, Libor Přeučil  

**Link**: [PDF](https://arxiv.org/pdf/2503.13090)  

**Abstract**: Uniform and variable environments still remain a challenge for stable visual localization and mapping in mobile robot navigation. One of the possible approaches suitable for such environments is appearance-based teach-and-repeat navigation, relying on simplified localization and reactive robot motion control - all without a need for standard mapping. This work brings an innovative solution to such a system based on visual place recognition techniques. Here, the major contributions stand in the employment of a new visual place recognition technique, a novel horizontal shift computation approach, and a multi-platform system design for applications across various types of mobile robots. Secondly, a new public dataset for experimental testing of appearance-based navigation methods is introduced. Moreover, the work also provides real-world experimental testing and performance comparison of the introduced navigation system against other state-of-the-art methods. The results confirm that the new system outperforms existing methods in several testing scenarios, is capable of operation indoors and outdoors, and exhibits robustness to day and night scene variations. 

**Abstract (ZH)**: 均匀和变化环境仍是对移动机器人导航中稳定视觉定位与建图的挑战。一种适用于此类环境的方法是基于外观的教与重复导航，依靠简化定位和反应式机器人运动控制——无需标准建图。本研究提出了一种基于视觉地点识别技术的创新解决方案。主要贡献在于采用了一种新的视觉地点识别技术、一种新颖的水平位移计算方法以及一种跨不同类型移动机器人的多平台系统设计。此外，还引入了一个新的公开数据集，用于实验测试基于外观的导航方法。该研究还提供了引入的导航系统与现有先进方法的现实世界实验测试和性能对比。结果证实，新系统在多种测试场景中表现优于现有方法，能够在室内外环境中运行，并且具有对日夜场景变化的 robustness。 

---
# Free-form language-based robotic reasoning and grasping 

**Title (ZH)**: 自由形式语言驱动的机器人推理与抓取 

**Authors**: Runyu Jiao, Alice Fasoli, Francesco Giuliari, Matteo Bortolon, Sergio Povoli, Guofeng Mei, Yiming Wang, Fabio Poiesi  

**Link**: [PDF](https://arxiv.org/pdf/2503.13082)  

**Abstract**: Performing robotic grasping from a cluttered bin based on human instructions is a challenging task, as it requires understanding both the nuances of free-form language and the spatial relationships between objects. Vision-Language Models (VLMs) trained on web-scale data, such as GPT-4o, have demonstrated remarkable reasoning capabilities across both text and images. But can they truly be used for this task in a zero-shot setting? And what are their limitations? In this paper, we explore these research questions via the free-form language-based robotic grasping task, and propose a novel method, FreeGrasp, leveraging the pre-trained VLMs' world knowledge to reason about human instructions and object spatial arrangements. Our method detects all objects as keypoints and uses these keypoints to annotate marks on images, aiming to facilitate GPT-4o's zero-shot spatial reasoning. This allows our method to determine whether a requested object is directly graspable or if other objects must be grasped and removed first. Since no existing dataset is specifically designed for this task, we introduce a synthetic dataset FreeGraspData by extending the MetaGraspNetV2 dataset with human-annotated instructions and ground-truth grasping sequences. We conduct extensive analyses with both FreeGraspData and real-world validation with a gripper-equipped robotic arm, demonstrating state-of-the-art performance in grasp reasoning and execution. Project website: this https URL. 

**Abstract (ZH)**: 基于人类指令从杂乱料箱中进行机器人抓取的性能研究：vision-language模型的零样本应用及其局限性探究 

---
# Vision-based automatic fruit counting with UAV 

**Title (ZH)**: 基于视觉的无人机自动水果计数 

**Authors**: Hubert Szolc, Mateusz Wasala, Remigiusz Mietla, Kacper Iwicki, Tomasz Kryjak  

**Link**: [PDF](https://arxiv.org/pdf/2503.13080)  

**Abstract**: The use of unmanned aerial vehicles (UAVs) for smart agriculture is becoming increasingly popular. This is evidenced by recent scientific works, as well as the various competitions organised on this topic. Therefore, in this work we present a system for automatic fruit counting using UAVs. To detect them, our solution uses a vision algorithm that processes streams from an RGB camera and a depth sensor using classical image operations. Our system also allows the planning and execution of flight trajectories, taking into account the minimisation of flight time and distance covered. We tested the proposed solution in simulation and obtained an average score of 87.27/100 points from a total of 500 missions. We also submitted it to the UAV Competition organised as part of the ICUAS 2024 conference, where we achieved an average score of 84.83/100 points, placing 6th in a field of 23 teams and advancing to the finals. 

**Abstract (ZH)**: 无人机在智能农业中的果实自动计数系统研究 

---
# Mitigating Cross-Modal Distraction and Ensuring Geometric Feasibility via Affordance-Guided, Self-Consistent MLLMs for Food Preparation Task Planning 

**Title (ZH)**: 通过手段引导且自一致性多模態language model planning实现食品准备任务规划中的跨模态干扰缓解与几何可行性保证 

**Authors**: Yu-Hong Shen, Chuan-Yu Wu, Yi-Ru Yang, Yen-Ling Tai, Yi-Ting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.13055)  

**Abstract**: We study Multimodal Large Language Models (MLLMs) with in-context learning for food preparation task planning. In this context, we identify two key challenges: cross-modal distraction and geometric feasibility. Cross-modal distraction occurs when the inclusion of visual input degrades the reasoning performance of a MLLM. Geometric feasibility refers to the ability of MLLMs to ensure that the selected skills are physically executable in the environment. To address these issues, we adapt Chain of Thought (CoT) with Self-Consistency to mitigate reasoning loss from cross-modal distractions and use affordance predictor as skill preconditions to guide MLLM on geometric feasibility. We construct a dataset to evaluate the ability of MLLMs on quantity estimation, reachability analysis, relative positioning and collision avoidance. We conducted a detailed evaluation to identify issues among different baselines and analyze the reasons for improvement, providing insights into each approach. Our method reaches a success rate of 76.7% on the entire dataset, showing a substantial improvement over the CoT baseline at 36.7%. 

**Abstract (ZH)**: 我们研究了具有上下文学习的多模态大型语言模型（MLLMs）在食品准备任务规划中的应用。在这一背景下，我们识别出两个主要挑战：跨模态干扰和几何可行性。跨模态干扰是指视觉输入的引入降低了MLLM的推理性能。几何可行性是指MLLM确保所选择的技能在环境中实际可执行的能力。为了解决这些问题，我们采用了带有自我一致性（Self-Consistency）的推理链（Chain of Thought，CoT）以减轻来自跨模态干扰的推理损失，并使用效应器预测器作为技能前提条件来引导MLLM在几何可行性方面。我们构建了一个数据集来评估MLLM在数量估算、可达性分析、相对定位和碰撞避免方面的能力。我们进行了详细的评估，以识别不同基线中的问题，并分析改进的原因，从而为每种方法提供洞察。我们的方法在整个数据集上的成功率为76.7%，相比于CoT基线的36.7%，显示出显著的改进。 

---
# Robot Skin with Touch and Bend Sensing using Electrical Impedance Tomography 

**Title (ZH)**: 基于电气阻抗 tomography 的触觉与弯曲传感机器人皮肤 

**Authors**: Haofeng Chen, Bin Li, Bedrich Himmel, Xiaojie Wang, Matej Hoffmann  

**Link**: [PDF](https://arxiv.org/pdf/2503.13048)  

**Abstract**: Flexible electronic skins that simultaneously sense touch and bend are desired in several application areas, such as to cover articulated robot structures. This paper introduces a flexible tactile sensor based on Electrical Impedance Tomography (EIT), capable of simultaneously detecting and measuring contact forces and flexion of the sensor. The sensor integrates a magnetic hydrogel composite and utilizes EIT to reconstruct internal conductivity distributions. Real-time estimation is achieved through the one-step Gauss-Newton method, which dynamically updates reference voltages to accommodate sensor deformation. A convolutional neural network is employed to classify interactions, distinguishing between touch, bending, and idle states using pre-reconstructed images. Experimental results demonstrate an average touch localization error of 5.4 mm (SD 2.2 mm) and average bending angle estimation errors of 1.9$^\circ$ (SD 1.6$^\circ$). The proposed adaptive reference method effectively distinguishes between single- and multi-touch scenarios while compensating for deformation effects. This makes the sensor a promising solution for multimodal sensing in robotics and human-robot collaboration. 

**Abstract (ZH)**: 基于 Electrical Impedance Tomography 的柔性触觉传感器：同时检测接触力和弯曲的可拉伸电子皮肤 

---
# Large-area Tomographic Tactile Skin with Air Pressure Sensing for Improved Force Estimation 

**Title (ZH)**: 大面积基于气压感知的 tomographic 触感皮肤以提高力估计 

**Authors**: Haofeng Chen, Bedrich Himmel, Jiri Kubik, Matej Hoffmann, Hyosang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.13036)  

**Abstract**: This paper presents a dual-channel tactile skin that integrates Electrical Impedance Tomography (EIT) with air pressure sensing to achieve accurate multi-contact force detection. The EIT layer provides spatial contact information, while the air pressure sensor delivers precise total force measurement. Our framework combines these complementary modalities through: deep learning-based EIT image reconstruction, contact area segmentation, and force allocation based on relative conductivity intensities from EIT. The experiments demonstrated 15.1% average force estimation error in single-contact scenarios and 20.1% in multi-contact scenarios without extensive calibration data requirements. This approach effectively addresses the challenge of simultaneously localizing and quantifying multiple contact forces without requiring complex external calibration setups, paving the way for practical and scalable soft robotic skin applications. 

**Abstract (ZH)**: 这篇论文提出了一种双通道触觉皮肤，结合了电气阻抗成像（EIT）与气压传感，实现了精确的多接触力检测。EIT层提供空间接触信息，而气压传感器提供精确的总力测量。该框架通过基于深度学习的EIT图像重建、接触区域分割以及基于EIT相对电导率强度的力量分配，结合了这些互补的模态。实验结果显示，在单接触场景下平均力估算误差为15.1%，多接触场景下为20.1%，且无需大量校准数据。该方法有效地解决了同时定位和量化多个接触力的挑战，无需复杂外部校准设置，为实用且可扩展的软体机器人皮肤应用铺平了道路。 

---
# Sensorless Remote Center of Motion Misalignment Estimation 

**Title (ZH)**: 无传感器运动中心偏移估计 

**Authors**: Hao Yang, Lidia Al-Zogbi, Ahmet Yildiz, Nabil Simaan, Jie Ying Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13011)  

**Abstract**: Laparoscopic surgery constrains instrument motion around a fixed pivot point at the incision into a patient to minimize tissue trauma. Surgical robots achieve this through either hardware to software-based remote center of motion (RCM) constraints. However, accurate RCM alignment is difficult due to manual trocar placement, patient motion, and tissue deformation. Misalignment between the robot's RCM point and the patient incision site can cause unsafe forces at the incision site. This paper presents a sensorless force estimation-based framework for dynamically assessing and optimizing RCM misalignment in robotic surgery. Our experiments demonstrate that misalignment exceeding 20 mm can generate large enough forces to potentially damage tissue, emphasizing the need for precise RCM positioning. For misalignment $D\geq $ 20 mm, our optimization algorithm estimates the RCM offset with an absolute error within 5 mm. Accurate RCM misalignment estimation is a step toward automated RCM misalignment compensation, enhancing safety and reducing tissue damage in robotic-assisted laparoscopic surgery. 

**Abstract (ZH)**: 腹腔镜手术通过在患者切口处固定一个活动中心点来约束器械运动，以最小化组织创伤。手术机器人通过硬件到软件基于远程中心运动（RCM）约束来实现这一目标。然而，由于手动Trocar放置、患者运动和组织变形，准确的RCM对齐非常困难。机器人RCM点与患者切口位置之间的偏差可能会在切口处产生不安全的力量。本文提出了一种基于无传感器力估计的框架，用于动态评估和优化机器人手术中的RCM错位。实验结果表明，超过20毫米的错位可以产生足够的力量，可能损伤组织，突显了精确RCM定位的必要性。对于错位$D\geq$ 20毫米的情况，我们的优化算法估计RCM偏移的绝对误差在5毫米以内。准确的RCM错位估计是自动补偿RCM错位的第一步，有助于增强腹腔镜手术辅助下的安全性并减少组织损伤。 

---
# Robot Policy Transfer with Online Demonstrations: An Active Reinforcement Learning Approach 

**Title (ZH)**: 基于在线示范的机器人策略转移：一种主动强化学习方法 

**Authors**: Muhan Hou, Koen Hindriks, A.E. Eiben, Kim Baraka  

**Link**: [PDF](https://arxiv.org/pdf/2503.12993)  

**Abstract**: Transfer Learning (TL) is a powerful tool that enables robots to transfer learned policies across different environments, tasks, or embodiments. To further facilitate this process, efforts have been made to combine it with Learning from Demonstrations (LfD) for more flexible and efficient policy transfer. However, these approaches are almost exclusively limited to offline demonstrations collected before policy transfer starts, which may suffer from the intrinsic issue of covariance shift brought by LfD and harm the performance of policy transfer. Meanwhile, extensive work in the learning-from-scratch setting has shown that online demonstrations can effectively alleviate covariance shift and lead to better policy performance with improved sample efficiency. This work combines these insights to introduce online demonstrations into a policy transfer setting. We present Policy Transfer with Online Demonstrations, an active LfD algorithm for policy transfer that can optimize the timing and content of queries for online episodic expert demonstrations under a limited demonstration budget. We evaluate our method in eight robotic scenarios, involving policy transfer across diverse environment characteristics, task objectives, and robotic embodiments, with the aim to transfer a trained policy from a source task to a related but different target task. The results show that our method significantly outperforms all baselines in terms of average success rate and sample efficiency, compared to two canonical LfD methods with offline demonstrations and one active LfD method with online demonstrations. Additionally, we conduct preliminary sim-to-real tests of the transferred policy on three transfer scenarios in the real-world environment, demonstrating the policy effectiveness on a real robot manipulator. 

**Abstract (ZH)**: 基于在线示范的策略迁移 

---
# A Hierarchical Region-Based Approach for Efficient Multi-Robot Exploration 

**Title (ZH)**: 一种分层基于区域的方法，用于高效的多机器人探索 

**Authors**: Di Meng, Tianhao Zhao, Chaoyu Xue, Jun Wu, Qiuguo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12876)  

**Abstract**: Multi-robot autonomous exploration in an unknown environment is an important application in this http URL exploration methods only use information around frontier points or viewpoints, ignoring spatial information of unknown areas. Moreover, finding the exact optimal solution for multi-robot task allocation is NP-hard, resulting in significant computational time consumption. To address these issues, we present a hierarchical multi-robot exploration framework using a new modeling method called RegionGraph. The proposed approach makes two main contributions: 1) A new modeling method for unexplored areas that preserves their spatial information across the entire space in a weighted graph called RegionGraph. 2) A hierarchical multi-robot exploration framework that decomposes the global exploration task into smaller subtasks, reducing the frequency of global planning and enabling asynchronous exploration. The proposed method is validated through both simulation and real-world experiments, demonstrating a 20% improvement in efficiency compared to existing methods. 

**Abstract (ZH)**: 多机器人在未知环境中的自主探索是一种重要应用，在此网址中探讨了探索方法仅利用前沿点或视点附近的信息，忽视了未知区域的空间信息。此外，多机器人任务分配的精确最优解问题属于NP难问题，导致显著的计算时间消耗。为解决这些问题，我们提出了一种基于新型建模方法RegionGraph的分层多机器人探索框架。该方法的主要贡献包括：1) 一种新的建模方法，用于未探索区域，通过加权图RegionGraph在整个空间中保留其空间信息。2) 一种分层多机器人探索框架，将全局探索任务分解为较小的子任务，减少全局规划的频率，实现异步探索。所提出的方法通过仿真和实地实验得到了验证，相比于现有方法提高了20%的效率。 

---
# In vivo validation of Wireless Power Transfer System for Magnetically Controlled Robotic Capsule Endoscopy 

**Title (ZH)**: 磁控机器人胶囊内镜无线电源传输系统的在体验证 

**Authors**: Alessandro Catania, Michele Bertozzi, Nikita J. Greenidge, Benjamin Calme, Gabriele Bandini, Christian Sbrana, Roberto Cecchi, Alice Buffi, Sebastiano Strangio, Pietro Valdastri, Giuseppe Iannaccone  

**Link**: [PDF](https://arxiv.org/pdf/2503.12850)  

**Abstract**: This paper presents the in vivo validation of an inductive wireless power transfer (WPT) system integrated for the first time into a magnetically controlled robotic capsule endoscopy platform. The proposed system enables continuous power delivery to the capsule without the need for onboard batteries, thus extending operational time and reducing size constraints. The WPT system operates through a resonant inductive coupling mechanism, based on a transmitting coil mounted on the end effector of a robotic arm that also houses an external permanent magnet and a localization coil for precise capsule manipulation. To ensure robust and stable power transmission in the presence of coil misalignment and rotation, a 3D receiving coil is integrated within the capsule. Additionally, a closed-loop adaptive control system, based on load-shift keying (LSK) modulation, dynamically adjusts the transmitted power to optimize efficiency while maintaining compliance with specific absorption rate (SAR) safety limits. The system has been extensively characterized in laboratory settings and validated through in vivo experiments using a porcine model, demonstrating reliable power transfer and effective robotic navigation in realistic gastrointestinal conditions: the average received power was 110 mW at a distance of 9 cm between the coils, with variable capsule rotation angles. The results confirm the feasibility of the proposed WPT approach for autonomous, battery-free robotic capsule endoscopy, paving the way for enhanced diagnostic in gastrointestinal medicine. 

**Abstract (ZH)**: 本文介绍了将感应无线功率传输（WPT）系统首次集成到磁控机器人胶囊内镜平台中的体内外验证。该系统能够无需机载电池持续为胶囊供电，从而延长操作时间和减少尺寸限制。WPT系统通过谐振感应耦合机制工作，该机制包括安装在机器人手臂末端执行器上的一个线圈，该末端执行器还配备了一个外部永久磁铁和一个用于精确胶囊操作的定位线圈。为了在线圈对齐和旋转不准确的情况下保证可靠的功率传输，胶囊内部集成了一个三维接收线圈。此外，基于载荷移相键控（LSK）调制的闭环自适应控制系统能够动态调整传输功率以优化效率，同时保持与特定吸收率（SAR）安全限值的合规性。该系统在实验室中进行了广泛表征，并通过使用猪模型进行的体内实验进行了验证，结果显示在两线圈之间9 cm的距离下平均接收功率为110 mW，且胶囊旋转角度变化。实验结果证实了所提出的WPT方法在自主、无电池的机器人胶囊内镜领域的可行性，为胃肠医学诊断提供了增强的诊断工具。 

---
# MT-PCR: Leveraging Modality Transformation for Large-Scale Point Cloud Registration with Limited Overlap 

**Title (ZH)**: MT-PCR：利用模态转换进行大规模点云配准以利用有限重叠区域 

**Authors**: Yilong Wu, Yifan Duan, Yuxi Chen, Xinran Zhang, Yedong Shen, Jianmin Ji, Yanyong Zhang, Lu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12833)  

**Abstract**: Large-scale scene point cloud registration with limited overlap is a challenging task due to computational load and constrained data acquisition. To tackle these issues, we propose a point cloud registration method, MT-PCR, based on Modality Transformation. MT-PCR leverages a BEV capturing the maximal overlap information to improve the accuracy and utilizes images to provide complementary spatial features. Specifically, MT-PCR converts 3D point clouds to BEV images and eastimates correspondence by 2D image keypoints extraction and matching. Subsequently, the 2D correspondence estimates are then transformed back to 3D point clouds using inverse mapping. We have applied MT-PCR to Terrestrial Laser Scanning and Aerial Laser Scanning point cloud registration on the GrAco dataset, involving 8 low-overlap, square-kilometer scale registration scenarios. Experiments and comparisons with commonly used methods demonstrate that MT-PCR can achieve superior accuracy and robustness in large-scale scenes with limited overlap. 

**Abstract (ZH)**: 基于模态变换的大规模场景点云注册方法：MT-PCR 

---
# Energy-Aware Task Allocation for Teams of Multi-mode Robots 

**Title (ZH)**: 多模式机器人团队的能源意识任务分配 

**Authors**: Takumi Ito, Riku Funada, Mitsuji Sampei, Gennaro Notomista  

**Link**: [PDF](https://arxiv.org/pdf/2503.12787)  

**Abstract**: This work proposes a novel multi-robot task allocation framework for robots that can switch between multiple modes, e.g., flying, driving, or walking. We first provide a method to encode the multi-mode property of robots as a graph, where the mode of each robot is represented by a node. Next, we formulate a constrained optimization problem to decide both the task to be allocated to each robot as well as the mode in which the latter should execute the task. The robot modes are optimized based on the state of the robot and the environment, as well as the energy required to execute the allocated task. Moreover, the proposed framework is able to encompass kinematic and dynamic models of robots alike. Furthermore, we provide sufficient conditions for the convergence of task execution and allocation for both robot models. 

**Abstract (ZH)**: 本工作提出了一种新型多机器人任务分配框架，适用于能够在多种模式（如飞行、驾驶或行走）之间切换的机器人。我们首先提供了一种方法，将机器人的多模式特性编码为图，其中每个机器人的模式由一个节点表示。然后，我们提出了一个约束优化问题，以确定每个机器人分配的任务以及该机器人执行任务的模式。根据机器人的状态、环境以及执行分配任务所需的能量来优化机器人的模式。此外，所提出的框架能够涵盖相同类型的运动学和动力学模型。同时，我们提供了确保两种机器人模型的任务执行和分配收敛的充分条件。 

---
# DART: Dual-level Autonomous Robotic Topology for Efficient Exploration in Unknown Environments 

**Title (ZH)**: DART：双层级自主 robotic 独构在未知环境中的高效探索 

**Authors**: Qiming Wang, Yulong Gao, Yang Wang, Xiongwei Zhao, Yijiao Sun, Xiangyan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12782)  

**Abstract**: Conventional algorithms in autonomous exploration face challenges due to their inability to accurately and efficiently identify the spatial distribution of convex regions in the real-time map. These methods often prioritize navigation toward the nearest or information-rich frontiers -- the boundaries between known and unknown areas -- resulting in incomplete convex region exploration and requiring excessive backtracking to revisit these missed areas. To address these limitations, this paper introduces an innovative dual-level topological analysis approach. First, we introduce a Low-level Topological Graph (LTG), generated through uniform sampling of the original map data, which captures essential geometric and connectivity details. Next, the LTG is transformed into a High-level Topological Graph (HTG), representing the spatial layout and exploration completeness of convex regions, prioritizing the exploration of convex regions that are not fully explored and minimizing unnecessary backtracking. Finally, an novel Local Artificial Potential Field (LAPF) method is employed for motion control, replacing conventional path planning and boosting overall efficiency. Experimental results highlight the effectiveness of our approach. Simulation tests reveal that our framework significantly reduces exploration time and travel distance, outperforming existing methods in both speed and efficiency. Ablation studies confirm the critical role of each framework component. Real-world tests demonstrate the robustness of our method in environments with poor mapping quality, surpassing other approaches in adaptability to mapping inaccuracies and inaccessible areas. 

**Abstract (ZH)**: 传统自主探索算法由于无法准确高效地识别实时地图中凸区域的空间分布而面临挑战。这些方法通常优先导航至最近或信息丰富的边界——已知区域与未知区域的边界——导致凸区域探索不完整，并需要大量回溯以重访遗漏的区域。为解决这些限制，本文提出了一种创新的多级拓扑分析方法。首先，我们提出了一种低级拓扑图（LTG），通过均匀采样原始地图数据生成，捕捉关键的几何和连接性细节。然后，将LTG转化为高级拓扑图（HTG），表示凸区域的空间布局和探索完整性，优先探索未完全探索的凸区域，减少不必要的回溯。最后，采用新颖的局部人工势场（LAPF）方法进行运动控制，取代传统路径规划，提升整体效率。实验结果表明该方法的有效性。模拟测试显示，本框架显著缩短了探索时间和行驶距离，在速度和效率上均优于现有方法。消融研究确认了每个框架组件的关键作用。实地测试证明了在地图质量较差的环境中，本方法的鲁棒性，在应对地图不准确性和不可达区域方面优于其他方法。 

---
# Dynamic-Dark SLAM: RGB-Thermal Cooperative Robot Vision Strategy for Multi-Person Tracking in Both Well-Lit and Low-Light Scenes 

**Title (ZH)**: 动态暗视觉SLAM：在明暗场景下多目标跟踪的RGB-热成像协同机器人视觉策略 

**Authors**: Tatsuro Sakai, Kanji Tanaka, Jonathan Tay Yu Liang, Muhammad Adil Luqman, Daiki Iwata  

**Link**: [PDF](https://arxiv.org/pdf/2503.12768)  

**Abstract**: In robot vision, thermal cameras have significant potential for recognizing humans even in complete darkness. However, their application to multi-person tracking (MPT) has lagged due to data scarcity and difficulties in individual identification. In this study, we propose a cooperative MPT system that utilizes co-located RGB and thermal cameras, using pseudo-annotations (bounding boxes + person IDs) to train RGB and T trackers. Evaluation experiments demonstrate that the T tracker achieves remarkable performance in both bright and dark scenes. Furthermore, results suggest that a tracker-switching approach using a binary brightness classifier is more suitable than a tracker-fusion approach for information integration. This study marks a crucial first step toward ``Dynamic-Dark SLAM," enabling effective recognition, understanding, and reconstruction of individuals, occluding objects, and traversable areas in dynamic environments, both bright and dark. 

**Abstract (ZH)**: 在机器人视觉中，热成像摄像头在完全黑暗环境中识别人类具有巨大的潜力。然而，其在多人跟踪（MPT）中的应用因数据稀缺和个人识别困难而滞后。本研究提出了一种使用共驻RGB和热成像摄像头的合作MPT系统，通过伪标注（边界框+人员ID）训练RGB和T跟踪器。评估实验表明，T跟踪器在明亮和黑暗场景中均表现出色。此外，结果表明，使用二元亮度分类器的跟踪器切换方法比跟踪器融合方法更适合信息集成。本研究标志着朝着“动态黑暗SLAM”的重要第一步，使得在明暗交替的动态环境中有效识别、理解和重建个体、遮挡物和可通行区域成为可能。 

---
# Humanoids in Hospitals: A Technical Study of Humanoid Surrogates for Dexterous Medical Interventions 

**Title (ZH)**: 医院中的类人机器人：灵巧医疗操作代理人的技术研究 

**Authors**: Soofiyan Atar, Xiao Liang, Calvin Joyce, Florian Richter, Wood Ricardo, Charles Goldberg, Preetham Suresh, Michael Yip  

**Link**: [PDF](https://arxiv.org/pdf/2503.12725)  

**Abstract**: The increasing demand for healthcare workers, driven by aging populations and labor shortages, presents a significant challenge for hospitals. Humanoid robots have the potential to alleviate these pressures by leveraging their human-like dexterity and adaptability to assist in medical procedures. This work conducted an exploratory study on the feasibility of humanoid robots performing direct clinical tasks through teleoperation. A bimanual teleoperation system was developed for the Unitree G1 Humanoid Robot, integrating high-fidelity pose tracking, custom grasping configurations, and an impedance controller to safely and precisely manipulate medical tools. The system is evaluated in seven diverse medical procedures, including physical examinations, emergency interventions, and precision needle tasks. Our results demonstrate that humanoid robots can successfully replicate critical aspects of human medical assessments and interventions, with promising quantitative performance in ventilation and ultrasound-guided tasks. However, challenges remain, including limitations in force output for procedures requiring high strength and sensor sensitivity issues affecting clinical accuracy. This study highlights the potential and current limitations of humanoid robots in hospital settings and lays the groundwork for future research on robotic healthcare integration. 

**Abstract (ZH)**: 人stanbul老龄化人口和劳动力短缺驱动的医疗服务需求增加给医院带来了重大挑战。仿人机器人通过利用其类似人类的灵活性和适应性来协助医疗程序，有望缓解这些压力。本研究通过远程操作探索仿人机器人执行直接临床任务的可能性。为Unitree G1仿人机器人开发了一套双臂远程操作系统，集成了高保真姿态跟踪、自定义抓取配置和阻抗控制器，以安全、精确地操作医疗工具。该系统在七种不同的医疗程序中进行了评估，包括体格检查、紧急干预和精确针刺任务。研究结果表明，仿人机器人能够成功复制人类医疗服务的关键方面，在通气和超声引导任务中表现出有前景的定量性能。然而，仍存在一些挑战，包括在需要高力量的程序中力输出的限制以及传感器灵敏度问题对临床准确性的影响。本研究突显了仿人机器人在医院环境中的潜力和当前限制，并为未来的机器人医疗服务整合研究奠定了基础。 

---
# CDKFormer: Contextual Deviation Knowledge-Based Transformer for Long-Tail Trajectory Prediction 

**Title (ZH)**: CDKFormer：基于上下文偏差知识的 transformers 在长尾轨迹预测中的应用 

**Authors**: Yuansheng Lian, Ke Zhang, Meng Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.12695)  

**Abstract**: Predicting the future movements of surrounding vehicles is essential for ensuring the safe operation and efficient navigation of autonomous vehicles (AVs) in urban traffic environments. Existing vehicle trajectory prediction methods primarily focus on improving overall performance, yet they struggle to address long-tail scenarios effectively. This limitation often leads to poor predictions in rare cases, significantly increasing the risk of safety incidents. Taking Argoverse 2 motion forecasting dataset as an example, we first investigate the long-tail characteristics in trajectory samples from two perspectives, individual motion and group interaction, and deriving deviation features to distinguish abnormal from regular scenarios. On this basis, we propose CDKFormer, a Contextual Deviation Knowledge-based Transformer model for long-tail trajectory prediction. CDKFormer integrates an attention-based scene context fusion module to encode spatiotemporal interaction and road topology. An additional deviation feature fusion module is proposed to capture the dynamic deviations in the target vehicle status. We further introduce a dual query-based decoder, supported by a multi-stream decoder block, to sequentially decode heterogeneous scene deviation features and generate multimodal trajectory predictions. Extensive experiments demonstrate that CDKFormer achieves state-of-the-art performance, significantly enhancing prediction accuracy and robustness for long-tailed trajectories compared to existing methods, thus advancing the reliability of AVs in complex real-world environments. 

**Abstract (ZH)**: 基于上下文偏差知识的Transformer模型：面向长尾轨迹预测 

---
# KISS-SLAM: A Simple, Robust, and Accurate 3D LiDAR SLAM System With Enhanced Generalization Capabilities 

**Title (ZH)**: KISS-SLAM：一种具有增强泛化能力的简单、稳健且精确的3D LiDAR SLAM系统 

**Authors**: Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Ignacio Vizzo, Giorgio Grisetti, Cyrill Stachniss  

**Link**: [PDF](https://arxiv.org/pdf/2503.12660)  

**Abstract**: Robust and accurate localization and mapping of an environment using laser scanners, so-called LiDAR SLAM, is essential to many robotic applications. Early 3D LiDAR SLAM methods often exploited additional information from IMU or GNSS sensors to enhance localization accuracy and mitigate drift. Later, advanced systems further improved the estimation at the cost of a higher runtime and complexity. This paper explores the limits of what can be achieved with a LiDAR-only SLAM approach while following the "Keep It Small and Simple" (KISS) principle. By leveraging this minimalistic design principle, our system, KISS-SLAM, archives state-of-the-art performances in pose accuracy while requiring little to no parameter tuning for deployment across diverse environments, sensors, and motion profiles. We follow best practices in graph-based SLAM and build upon LiDAR odometry to compute the relative motion between scans and construct local maps of the environment. To correct drift, we match local maps and optimize the trajectory in a pose graph optimization step. The experimental results demonstrate that this design achieves competitive performance while reducing complexity and reliance on additional sensor modalities. By prioritizing simplicity, this work provides a new strong baseline for LiDAR-only SLAM and a high-performing starting point for future research. Further, our pipeline builds consistent maps that can be used directly for further downstream tasks like navigation. Our open-source system operates faster than the sensor frame rate in all presented datasets and is designed for real-world scenarios. 

**Abstract (ZH)**: 仅使用激光雷达进行鲁棒且精确的环境定位与建图：遵循“保持简单和简约”原则的LiDAR-only SLAM方法探索 

---
# VISO-Grasp: Vision-Language Informed Spatial Object-centric 6-DoF Active View Planning and Grasping in Clutter and Invisibility 

**Title (ZH)**: VISO-Grasp: 基于视觉-语言空间物体中心6自由度主动视图规划与抓取技术在杂乱环境和不可见性下的应用 

**Authors**: Yitian Shi, Di Wen, Guanqi Chen, Edgar Welte, Sheng Liu, Kunyu Peng, Rainer Stiefelhagen, Rania Rayyes  

**Link**: [PDF](https://arxiv.org/pdf/2503.12609)  

**Abstract**: We propose VISO-Grasp, a novel vision-language-informed system designed to systematically address visibility constraints for grasping in severely occluded environments. By leveraging Foundation Models (FMs) for spatial reasoning and active view planning, our framework constructs and updates an instance-centric representation of spatial relationships, enhancing grasp success under challenging occlusions. Furthermore, this representation facilitates active Next-Best-View (NBV) planning and optimizes sequential grasping strategies when direct grasping is infeasible. Additionally, we introduce a multi-view uncertainty-driven grasp fusion mechanism that refines grasp confidence and directional uncertainty in real-time, ensuring robust and stable grasp execution. Extensive real-world experiments demonstrate that VISO-Grasp achieves a success rate of $87.5\%$ in target-oriented grasping with the fewest grasp attempts outperforming baselines. To the best of our knowledge, VISO-Grasp is the first unified framework integrating FMs into target-aware active view planning and 6-DoF grasping in environments with severe occlusions and entire invisibility constraints. 

**Abstract (ZH)**: VISO-Grasp: 一种用于严重遮挡环境下目标导向抓取的新型视觉-语言启发系统 

---
# MUKCa: Accurate and Affordable Cobot Calibration Without External Measurement Devices 

**Title (ZH)**: MUKCa: 准确且经济的协作机器人标定方法无需外部测量设备 

**Authors**: Giovanni Franzese, Max Spahn, Jens Kober, Cosimo Della Santina  

**Link**: [PDF](https://arxiv.org/pdf/2503.12584)  

**Abstract**: To increase the reliability of collaborative robots in performing daily tasks, we require them to be accurate and not only repeatable. However, having a calibrated kinematics model is regrettably a luxury, as available calibration tools are usually more expensive than the robots themselves. With this work, we aim to contribute to the democratization of cobots calibration by providing an inexpensive yet highly effective alternative to existing tools. The proposed minimalist calibration routine relies on a 3D-printable tool as the only physical aid to the calibration process. This two-socket spherical-joint tool kinematically constrains the robot at the end effector while collecting the training set. An optimization routine updates the nominal model to ensure a consistent prediction for each socket and the undistorted mean distance between them. We validated the algorithm on three robotic platforms: Franka, Kuka, and Kinova Cobots. The calibrated models reduce the mean absolute error from the order of 10 mm to 0.2 mm for both Franka and Kuka robots. We provide two additional experimental campaigns with the Franka Robot to render the improvements more tangible. First, we implement Cartesian control with and without the calibrated model and use it to perform a standard peg-in-the-hole task with a tolerance of 0.4 mm between the peg and the hole. Second, we perform a repeated drawing task combining Cartesian control with learning from demonstration. Both tasks consistently failed when the model was not calibrated, while they consistently succeeded after calibration. 

**Abstract (ZH)**: 为了提高协作机器人在执行日常任务时的可靠性，我们需要它们不仅具备重复性还要求精确性。然而，拥有校准的运动学模型通常是奢侈的，因为可用的校准工具通常比机器人本身更昂贵。通过本项工作，我们旨在通过提供一种廉价而有效的替代工具来促进协作机器人校准的普及。所提出的简约校准程序仅依赖于一个3D打印工具作为校准过程中的唯一物理辅助。这个由两个插孔组成的球关节工具在末端执行器处 kinematically 限制机器人并在过程中收集训练集。优化程序更新名义模型以确保每个插孔及其未失真的平均距离的一致预测。我们在三台机器人平台上验证了该算法：Franka、Kuka 和 Kinova 协作机器人。校准后的模型将Franka和Kuka机器人的平均绝对误差从10毫米级降低到0.2毫米。我们提供了Franka机器人额外的两个实验方案以使改进更为具体。首先，我们实现了带校准模型和不带校准模型的笛卡尔控制，并使用它在一个 peg-in-the-hole 任务中执行标准操作，插销与孔之间的偏差为0.4毫米。其次，我们执行了结合笛卡尔控制与示例学习的重复绘制任务。当模型未校准时，两个任务均一致失败；而在校准后，两个任务均一致成功。 

---
# Focusing Robot Open-Ended Reinforcement Learning Through Users' Purposes 

**Title (ZH)**: 聚焦机器人通过用户目的的开放性强化学习 

**Authors**: Emilio Cartoni, Gianluca Cioccolini, Gianluca Baldassarre  

**Link**: [PDF](https://arxiv.org/pdf/2503.12579)  

**Abstract**: Open-Ended Learning (OEL) autonomous robots can acquire new skills and knowledge through direct interaction with their environment, relying on mechanisms such as intrinsic motivations and self-generated goals to guide learning processes. OEL robots are highly relevant for applications as they can autonomously leverage acquired knowledge to perform tasks beneficial to human users in unstructured environments, addressing challenges unforeseen at design time. However, OEL robots face a significant limitation: their openness may lead them to waste time learning information that is irrelevant to tasks desired by specific users. Here, we propose a solution called `Purpose-Directed Open-Ended Learning' (POEL), based on the novel concept of `purpose' introduced in previous work. A purpose specifies what users want the robot to achieve. The key insight of this work is that purpose can focus OEL on learning self-generated classes of tasks that, while unknown during autonomous learning (as typical in OEL), involve objects relevant to the purpose. This concept is operationalised in a novel robot architecture capable of receiving a human purpose through speech-to-text, analysing the scene to identify objects, and using a Large Language Model to reason about which objects are purpose-relevant. These objects are then used to bias OEL exploration towards their spatial proximity and to self-generate rewards that favour interactions with them. The solution is tested in a simulated scenario where a camera-arm-gripper robot interacts freely with purpose-related and distractor objects. For the first time, the results demonstrate the potential advantages of purpose-focused OEL over state-of-the-art OEL methods, enabling robots to handle unstructured environments while steering their learning toward knowledge acquisition relevant to users. 

**Abstract (ZH)**: 基于目的导向的开放 ended 学习（POEL） 

---
# Grasping Partially Occluded Objects Using Autoencoder-Based Point Cloud Inpainting 

**Title (ZH)**: 基于自编码器的点云修复用于抓取部分遮挡物体 

**Authors**: Alexander Koebler, Ralf Gross, Florian Buettner, Ingo Thon  

**Link**: [PDF](https://arxiv.org/pdf/2503.12549)  

**Abstract**: Flexible industrial production systems will play a central role in the future of manufacturing due to higher product individualization and customization. A key component in such systems is the robotic grasping of known or unknown objects in random positions. Real-world applications often come with challenges that might not be considered in grasping solutions tested in simulation or lab settings. Partial occlusion of the target object is the most prominent. Examples of occlusion can be supporting structures in the camera's field of view, sensor imprecision, or parts occluding each other due to the production process. In all these cases, the resulting lack of information leads to shortcomings in calculating grasping points. In this paper, we present an algorithm to reconstruct the missing information. Our inpainting solution facilitates the real-world utilization of robust object matching approaches for grasping point calculation. We demonstrate the benefit of our solution by enabling an existing grasping system embedded in a real-world industrial application to handle occlusions in the input. With our solution, we drastically decrease the number of objects discarded by the process. 

**Abstract (ZH)**: 灵活的工业生产系统将在未来制造中发挥核心作用，由于产品个体化和定制化的程度提高。此类系统的关键组成部分是在随机位置抓取已知或未知物体的机器人操作。现实世界的应用中常常会遇到在仿真或实验室条件下未考虑的挑战，其中最突出的是目标物体的部分遮挡。遮挡的例子包括摄像机视野中的支撑结构、传感器精度不足，或因生产过程导致的部件互相遮挡。在所有这些情况下，缺失信息导致抓取点计算的不足。本文提出了一种算法来重构缺失信息。我们的 inpaint 方法促进了稳健的对象匹配方法在抓取点计算中的实际应用。通过我们的解决方案，使已嵌入实际工业应用的现有抓取系统能够处理输入中的遮挡，显著减少了被过程废弃的物体数量。 

---
# Histogram Transporter: Learning Rotation-Equivariant Orientation Histograms for High-Precision Robotic Kitting 

**Title (ZH)**: Histogram 运输器：学习旋转不变的方向直方图以实现高精度机器人组装 

**Authors**: Jiadong Zhou, Yadan Zeng, Huixu Dong, I-Ming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.12541)  

**Abstract**: Robotic kitting is a critical task in industrial automation that requires the precise arrangement of objects into kits to support downstream production processes. However, when handling complex kitting tasks that involve fine-grained orientation alignment, existing approaches often suffer from limited accuracy and computational efficiency. To address these challenges, we propose Histogram Transporter, a novel kitting framework that learns high-precision pick-and-place actions from scratch using only a few demonstrations. First, our method extracts rotation-equivariant orientation histograms (EOHs) from visual observations using an efficient Fourier-based discretization strategy. These EOHs serve a dual purpose: improving picking efficiency by directly modeling action success probabilities over high-resolution orientations and enhancing placing accuracy by serving as local, discriminative feature descriptors for object-to-placement matching. Second, we introduce a subgroup alignment strategy in the place model that compresses the full spectrum of EOHs into a compact orientation representation, enabling efficient feature matching while preserving accuracy. Finally, we examine the proposed framework on the simulated Hand-Tool Kitting Dataset (HTKD), where it outperforms competitive baselines in both success rates and computational efficiency. Further experiments on five Raven-10 tasks exhibits the remarkable adaptability of our approach, with real-robot trials confirming its applicability for real-world deployment. 

**Abstract (ZH)**: 机器人快换是工业自动化中的一个关键任务，要求将物体精确排列成快换组件以支持下游生产过程。然而，在处理涉及精细方向对齐的复杂快换任务时，现有方法往往在准确性和计算效率上存在局限性。为应对这些挑战，我们提出了一种名为Histogram Transporter的新颖快换框架，该框架仅通过少数示范就能从零开始学习高精度的拾取和放置动作。首先，我们的方法通过有效的傅里叶基离散化策略从视觉观察中提取旋转等变方向直方图（EOHs），这些EOHs具有双重功能：通过直接建模高分辨率方向上的动作成功率来提高拾取效率，并作为对象到放置匹配的局部区分特征描述子以提高放置精度。其次，我们在放置模型中引入了一个子组对齐策略，将EOHs的整个光谱压缩为紧凑的方向表示，从而实现高效特征匹配同时保持准确性。最后，我们在模拟的手工具快换数据集中对该提出的框架进行了评估，在成功率和计算效率上均优于竞争性基线。在五个Raven-10任务上的进一步实验展示了我们方法的出色可适应性，实际机器人试验验证了其在实际部署中的适用性。 

---
# EmoBipedNav: Emotion-aware Social Navigation for Bipedal Robots with Deep Reinforcement Learning 

**Title (ZH)**: 情智双融步行导航：基于深度强化学习的双足机器人情感aware社会导航 

**Authors**: Wei Zhu, Abirath Raju, Abdulaziz Shamsah, Anqi Wu, Seth Hutchinson, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12538)  

**Abstract**: This study presents an emotion-aware navigation framework -- EmoBipedNav -- using deep reinforcement learning (DRL) for bipedal robots walking in socially interactive environments. The inherent locomotion constraints of bipedal robots challenge their safe maneuvering capabilities in dynamic environments. When combined with the intricacies of social environments, including pedestrian interactions and social cues, such as emotions, these challenges become even more pronounced. To address these coupled problems, we propose a two-stage pipeline that considers both bipedal locomotion constraints and complex social environments. Specifically, social navigation scenarios are represented using sequential LiDAR grid maps (LGMs), from which we extract latent features, including collision regions, emotion-related discomfort zones, social interactions, and the spatio-temporal dynamics of evolving environments. The extracted features are directly mapped to the actions of reduced-order models (ROMs) through a DRL architecture. Furthermore, the proposed framework incorporates full-order dynamics and locomotion constraints during training, effectively accounting for tracking errors and restrictions of the locomotion controller while planning the trajectory with ROMs. Comprehensive experiments demonstrate that our approach exceeds both model-based planners and DRL-based baselines. The hardware videos and open-source code are available at this https URL. 

**Abstract (ZH)**: 基于深度强化学习的情感感知双足机器人社会交互导航框架——EmoBipedNav 

---
# Being-0: A Humanoid Robotic Agent with Vision-Language Models and Modular Skills 

**Title (ZH)**: Being-0: 一种配备视觉语言模型和模块化技能的人形机器人代理 

**Authors**: Haoqi Yuan, Yu Bai, Yuhui Fu, Bohan Zhou, Yicheng Feng, Xinrun Xu, Yi Zhan, Börje F. Karlsson, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12533)  

**Abstract**: Building autonomous robotic agents capable of achieving human-level performance in real-world embodied tasks is an ultimate goal in humanoid robot research. Recent advances have made significant progress in high-level cognition with Foundation Models (FMs) and low-level skill development for humanoid robots. However, directly combining these components often results in poor robustness and efficiency due to compounding errors in long-horizon tasks and the varied latency of different modules. We introduce Being-0, a hierarchical agent framework that integrates an FM with a modular skill library. The FM handles high-level cognitive tasks such as instruction understanding, task planning, and reasoning, while the skill library provides stable locomotion and dexterous manipulation for low-level control. To bridge the gap between these levels, we propose a novel Connector module, powered by a lightweight vision-language model (VLM). The Connector enhances the FM's embodied capabilities by translating language-based plans into actionable skill commands and dynamically coordinating locomotion and manipulation to improve task success. With all components, except the FM, deployable on low-cost onboard computation devices, Being-0 achieves efficient, real-time performance on a full-sized humanoid robot equipped with dexterous hands and active vision. Extensive experiments in large indoor environments demonstrate Being-0's effectiveness in solving complex, long-horizon tasks that require challenging navigation and manipulation subtasks. For further details and videos, visit this https URL. 

**Abstract (ZH)**: 构建能够在现实世界物理任务中达到人类水平表现的自主机器人代理是类人机器人研究的最终目标。最近的进步在高层次认知方面取得了显著进展，采用了基础模型（FMs），并在低级技能开发方面取得了进展。然而，直接将这些组件结合起来常常由于长时间任务中的累积错误以及不同模块的变异延迟导致鲁棒性和效率不足。我们提出了Being-0，这是一种分层代理框架，将基础模型与模块化技能库相结合。基础模型处理高层次的认知任务，如指令理解、任务规划和推理，而技能库提供稳定的运动和灵巧操作，以支持低级控制。为了解决这些层次之间的差异，我们提出了一种新的连接器模块，由轻量级的视觉-语言模型（VLM）驱动。连接器通过将基于语言的计划转化为可执行的技能命令，并动态协调运动和操作来增强基础模型的物理能力，从而提高任务成功率。除了基础模型外，所有组件均部署在低成本的车载计算设备上，使Being-0能够在配备灵巧手和活动视觉的全尺寸类人机器人上实现高效的实时性能。在大型室内环境中的广泛实验表明，Being-0在解决需要复杂导航和操作子任务的长时间复杂任务方面非常有效。更多信息和视频参见此链接。 

---
# Closed-Loop Control and Disturbance Mitigation of an Underwater Multi-Segment Continuum Manipulator 

**Title (ZH)**: 水下多段连续体 manipulator 的闭环控制与干扰抑制 

**Authors**: Kyle L. Walker, Hsing-Yu Chen, Alix J. Partridge, Lucas Cruz da Silva, Adam A. Stokes, Francesco Giorgio-Serchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.12508)  

**Abstract**: The use of soft and compliant manipulators in marine environments represents a promising paradigm shift for subsea inspection, with devices better suited to tasks owing to their ability to safely conform to items during contact. However, limitations driven by material characteristics often restrict the reach of such devices, with the complexity of obtaining state estimations making control non-trivial. Here, a detailed analysis of a 1m long compliant manipulator prototype for subsea inspection tasks is presented, including its mechanical design, state estimation technique, closed-loop control strategies, and experimental performance evaluation in underwater conditions. Results indicate that both the configuration-space and task-space controllers implemented are capable of positioning the end effector to desired locations, with deviations of <5% of the manipulator length spatially and to within 5^{o} of the desired configuration angles. The manipulator was also tested when subjected to various disturbances, such as loads of up to 300g and random point disturbances, and was proven to be able to limit displacement and restore the desired configuration. This work is a significant step towards the implementation of compliant manipulators in real-world subsea environments, proving their potential as an alternative to classical rigid-link designs. 

**Abstract (ZH)**: 软性和顺应性 manipulator 在海洋环境中的应用为水下检测任务带来了潜在的范式转变，但由于材料特性限制了设备的灵活性，获得状态估计的复杂性使控制变得非 trivial。本文详细分析了一种长达 1 米的顺应性 manipulator 原型在水下检测任务中的应用，包括其机械设计、状态估计技术、闭环控制策略以及在水下条件下的实验性能评估。结果显示，实施的配置空间控制器和任务空间控制器都能将末端执行器定位到所需位置，空间偏差小于 manipulator 长度的 5%，角度偏差在所需配置角度的 5° 以内。此外，该 manipulator 还在承受各种干扰（包括最大 300 克的负载和随机点干扰）的情况下进行了测试，并被证明能够限制位移并恢复到所需配置。这项工作为顺应性 manipulator 在现实水下环境中的应用实施迈出了重要一步，证明了它们作为传统刚性连杆设计替代方案的潜力。 

---
# Modality-Composable Diffusion Policy via Inference-Time Distribution-level Composition 

**Title (ZH)**: 模态可组合扩散策略通过推理时分布级组合 

**Authors**: Jiahang Cao, Qiang Zhang, Hanzhong Guo, Jiaxu Wang, Hao Cheng, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12466)  

**Abstract**: Diffusion Policy (DP) has attracted significant attention as an effective method for policy representation due to its capacity to model multi-distribution dynamics. However, current DPs are often based on a single visual modality (e.g., RGB or point cloud), limiting their accuracy and generalization potential. Although training a generalized DP capable of handling heterogeneous multimodal data would enhance performance, it entails substantial computational and data-related costs. To address these challenges, we propose a novel policy composition method: by leveraging multiple pre-trained DPs based on individual visual modalities, we can combine their distributional scores to form a more expressive Modality-Composable Diffusion Policy (MCDP), without the need for additional training. Through extensive empirical experiments on the RoboTwin dataset, we demonstrate the potential of MCDP to improve both adaptability and performance. This exploration aims to provide valuable insights into the flexible composition of existing DPs, facilitating the development of generalizable cross-modality, cross-domain, and even cross-embodiment policies. Our code is open-sourced at this https URL. 

**Abstract (ZH)**: 模态可组合扩散策略（MCDP）: 一种有效的策略表示方法 

---
# Bio-Inspired Plastic Neural Networks for Zero-Shot Out-of-Distribution Generalization in Complex Animal-Inspired Robots 

**Title (ZH)**: 受生物启发的塑料神经网络在复杂生物启发机器人中实现零样本分布外泛化 

**Authors**: Binggwong Leung, Worasuchad Haomachai, Joachim Winther Pedersen, Sebastian Risi, Poramate Manoonpong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12406)  

**Abstract**: Artificial neural networks can be used to solve a variety of robotic tasks. However, they risk failing catastrophically when faced with out-of-distribution (OOD) situations. Several approaches have employed a type of synaptic plasticity known as Hebbian learning that can dynamically adjust weights based on local neural activities. Research has shown that synaptic plasticity can make policies more robust and help them adapt to unforeseen changes in the environment. However, networks augmented with Hebbian learning can lead to weight divergence, resulting in network instability. Furthermore, such Hebbian networks have not yet been applied to solve legged locomotion in complex real robots with many degrees of freedom. In this work, we improve the Hebbian network with a weight normalization mechanism for preventing weight divergence, analyze the principal components of the Hebbian's weights, and perform a thorough evaluation of network performance in locomotion control for real 18-DOF dung beetle-like and 16-DOF gecko-like robots. We find that the Hebbian-based plastic network can execute zero-shot sim-to-real adaptation locomotion and generalize to unseen conditions, such as uneven terrain and morphological damage. 

**Abstract (ZH)**: 人工神经网络可以用于解决多种机器人任务。然而，它们在面对分布外（OOD）情况时存在灾难性失败的风险。已有研究表明，利用一类称为Hebbian学习的突触可塑性可以在局部神经活动的基础上动态调整权重，从而提高策略的鲁棒性，使其能够适应环境中的不可预见变化。然而，带有Hebbian学习的网络可能会导致权重发散，进而导致网络不稳定。此外，目前Hebbian网络尚未被应用于解决具有许多自由度的复杂真实腿式移动机器人中的移动控制问题。在本研究中，我们通过引入权重规范化机制改进了Hebbian网络，分析了Hebbian权重的主要成分，并对Hebbian基可塑网络在真实18-DOF象粪甲虫样和16-DOF壁虎样机器人中的移动控制性能进行了全面评估。研究发现，基于Hebbian的学习网络可以实现零样本的仿真实到现实的移动适应性，并能泛化到未见过的条件，如不平地形和形态损伤。 

---
# TERL: Large-Scale Multi-Target Encirclement Using Transformer-Enhanced Reinforcement Learning 

**Title (ZH)**: TERL：基于Transformer增强强化学习的大规模多目标包围关键技术 

**Authors**: Heng Zhang, Guoxiang Zhao, Xiaoqiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.12395)  

**Abstract**: Pursuit-evasion (PE) problem is a critical challenge in multi-robot systems (MRS). While reinforcement learning (RL) has shown its promise in addressing PE tasks, research has primarily focused on single-target pursuit, with limited exploration of multi-target encirclement, particularly in large-scale settings. This paper proposes a Transformer-Enhanced Reinforcement Learning (TERL) framework for large-scale multi-target encirclement. By integrating a transformer-based policy network with target selection, TERL enables robots to adaptively prioritize targets and safely coordinate robots. Results show that TERL outperforms existing RL-based methods in terms of encirclement success rate and task completion time, while maintaining good performance in large-scale scenarios. Notably, TERL, trained on small-scale scenarios (15 pursuers, 4 targets), generalizes effectively to large-scale settings (80 pursuers, 20 targets) without retraining, achieving a 100% success rate. 

**Abstract (ZH)**: 大型多目标包围的Transformer增强强化学习框架 

---
# M2UD: A Multi-model, Multi-scenario, Uneven-terrain Dataset for Ground Robot with Localization and Mapping Evaluation 

**Title (ZH)**: M2UD：适用于地面机器人定位与建图评估的多模型、多场景、不规则地形数据集 

**Authors**: Yanpeng Jia, Shiyi Wang, Shiliang Shao, Yue Wang, Fu Zhang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12387)  

**Abstract**: Ground robots play a crucial role in inspection, exploration, rescue, and other applications. In recent years, advancements in LiDAR technology have made sensors more accurate, lightweight, and cost-effective. Therefore, researchers increasingly integrate sensors, for SLAM studies, providing robust technical support for ground robots and expanding their application domains. Public datasets are essential for advancing SLAM technology. However, existing datasets for ground robots are typically restricted to flat-terrain motion with 3 DOF and cover only a limited range of scenarios. Although handheld devices and UAV exhibit richer and more aggressive movements, their datasets are predominantly confined to small-scale environments due to endurance limitations. To fill these gap, we introduce M2UD, a multi-modal, multi-scenario, uneven-terrain SLAM dataset for ground robots. This dataset contains a diverse range of highly challenging environments, including cities, open fields, long corridors, and mixed scenarios. Additionally, it presents extreme weather conditions. The aggressive motion and degradation characteristics of this dataset not only pose challenges for testing and evaluating existing SLAM methods but also advance the development of more advanced SLAM algorithms. To benchmark SLAM algorithms, M2UD provides smoothed ground truth localization data obtained via RTK and introduces a novel localization evaluation metric that considers both accuracy and efficiency. Additionally, we utilize a high-precision laser scanner to acquire ground truth maps of two representative scenes, facilitating the development and evaluation of mapping algorithms. We select 12 localization sequences and 2 mapping sequences to evaluate several classical SLAM algorithms, verifying usability of the dataset. To enhance usability, the dataset is accompanied by a suite of development kits. 

**Abstract (ZH)**: 多模式多场景不规则地形地面机器人SLAM数据集M2UD 

---
# GameChat: Multi-LLM Dialogue for Safe, Agile, and Socially Optimal Multi-Agent Navigation in Constrained Environments 

**Title (ZH)**: GameChat: 多约束环境下多智能体导航的多语言模型对话安全、灵活和社会最优交互 

**Authors**: Vagul Mahadevan, Shangtong Zhang, Rohan Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2503.12333)  

**Abstract**: Safe, agile, and socially compliant multi-robot navigation in cluttered and constrained environments remains a critical challenge. This is especially difficult with self-interested agents in decentralized settings, where there is no central authority to resolve conflicts induced by spatial symmetry. We address this challenge by proposing a novel approach, GameChat, which facilitates safe, agile, and deadlock-free navigation for both cooperative and self-interested agents. Key to our approach is the use of natural language communication to resolve conflicts, enabling agents to prioritize more urgent tasks and break spatial symmetry in a socially optimal manner. Our algorithm ensures subgame perfect equilibrium, preventing agents from deviating from agreed-upon behaviors and supporting cooperation. Furthermore, we guarantee safety through control barrier functions and preserve agility by minimizing disruptions to agents' planned trajectories. We evaluate GameChat in simulated environments with doorways and intersections. The results show that even in the worst case, GameChat reduces the time for all agents to reach their goals by over 35% from a naive baseline and by over 20% from SMG-CBF in the intersection scenario, while doubling the rate of ensuring the agent with a higher priority task reaches the goal first, from 50% (equivalent to random chance) to a 100% perfect performance at maximizing social welfare. 

**Abstract (ZH)**: 安全、灵活且社交规范的多机器人在复杂受限环境下的导航：一种基于GameChat的新方法 

---
# Train Robots in a JIF: Joint Inverse and Forward Dynamics with Human and Robot Demonstrations 

**Title (ZH)**: 基于示现的联合逆向和正向动力学训练机器人：JIF方法 

**Authors**: Gagan Khandate, Boxuan Wang, Sarah Park, Weizhe Ni, Jaoquin Palacious, Kate Lampo, Philippe Wu, Rosh Ho, Eric Chang, Matei Ciocarlie  

**Link**: [PDF](https://arxiv.org/pdf/2503.12297)  

**Abstract**: Pre-training on large datasets of robot demonstrations is a powerful technique for learning diverse manipulation skills but is often limited by the high cost and complexity of collecting robot-centric data, especially for tasks requiring tactile feedback. This work addresses these challenges by introducing a novel method for pre-training with multi-modal human demonstrations. Our approach jointly learns inverse and forward dynamics to extract latent state representations, towards learning manipulation specific representations. This enables efficient fine-tuning with only a small number of robot demonstrations, significantly improving data efficiency. Furthermore, our method allows for the use of multi-modal data, such as combination of vision and touch for manipulation. By leveraging latent dynamics modeling and tactile sensing, this approach paves the way for scalable robot manipulation learning based on human demonstrations. 

**Abstract (ZH)**: 基于多模态人类演示的预训练方法在大规模数据集上的机器人操作技能学习 

---
# SharedAssembly: A Data Collection Approach via Shared Tele-Assembly 

**Title (ZH)**: SharedAssembly: 一种基于共享远程组装的数据收集方法 

**Authors**: Yansong Wu, Xiao Chen, Yu Chen, Hamid Sadeghian, Fan Wu, Zhenshan Bing, Sami Haddadin, Alexander König, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.12287)  

**Abstract**: Assembly is a fundamental skill for robots in both modern manufacturing and service robotics. Existing datasets aim to address the data bottleneck in training general-purpose robot models, falling short of capturing contact-rich assembly tasks. To bridge this gap, we introduce SharedAssembly, a novel bilateral teleoperation approach with shared autonomy for scalable assembly execution and data collection. User studies demonstrate that the proposed approach enhances both success rates and efficiency, achieving a 97.0% success rate across various sub-millimeter-level assembly tasks. Notably, novice and intermediate users achieve performance comparable to experts using baseline teleoperation methods, significantly enhancing large-scale data collection. 

**Abstract (ZH)**: 共享自治的双边远程操作方法：共享Assembly实现可扩展的装配执行与数据采集 

---
# Clarke Coordinates Are Generalized Improved State Parametrization for Continuum Robots 

**Title (ZH)**: Clarke 坐标是连续机器人的一种广义改进状态参数化表示。 

**Authors**: Reinhard M. Grassmann, Jessica Burgner-Kahrs  

**Link**: [PDF](https://arxiv.org/pdf/2503.12265)  

**Abstract**: In this letter, we demonstrate that previously proposed improved state parameterizations for soft and continuum robots are specific cases of Clarke coordinates. By explicitly deriving these improved parameterizations from a generalized Clarke transformation matrix, we unify various approaches into one comprehensive mathematical framework. This unified representation provides clarity regarding their relationships and generalizes them beyond existing constraints, including arbitrary joint numbers, joint distributions, and underlying modeling assumptions. This unification consolidates prior insights and establishes Clarke coordinates as a foundational tool, enabling systematic knowledge transfer across different subfields within soft and continuum robotics. 

**Abstract (ZH)**: 在这封信中，我们证明了之前提出的用于软体和连续体机器人的改进状态参数化是克拉克坐标的具体案例。通过从广义克拉克变换矩阵Explicitly推导这些改进的参数化，我们将各种方法统一到一个综合的数学框架中。这种统一的表示方式明确了它们之间的关系，超越了现有的限制，包括任意关节数量、关节分布和基础建模假设。这种统一整合了先前的见解，确立了克拉克坐标作为基础工具的作用，促进了软体和连续体机器人不同子领域的系统知识转移。 

---
# Nonparametric adaptive payload tracking for an offshore crane 

**Title (ZH)**: offshore起重机的非参数自适应载荷跟踪 

**Authors**: Torbjørn Smith, Olav Egeland  

**Link**: [PDF](https://arxiv.org/pdf/2503.12250)  

**Abstract**: A nonparametric adaptive crane control system is proposed where the crane payload tracks a desired trajectory with feedback from the payload position. The payload motion is controlled with the position of the crane tip using partial feedback linearization. This is made possible by introducing a novel model structure given in Cartesian coordinates. This Cartesian model structure makes it possible to implement a nonparametric adaptive controller which cancels disturbances by approximating the effects of unknown disturbance forces and structurally unknown dynamics in a reproducing kernel Hilbert space (RKHS). It is shown that the nonparametric adaptive controller leads to uniformly ultimately bounded errors in the presence of unknown forces and unmodeled dynamics. Moreover, it is shown that the Cartesian formulation has certain advantages in payload tracking control also in the non-adaptive case. The performance of the nonparametric adaptive controller is validated in simulation and experiments with good results. 

**Abstract (ZH)**: 一种基于反馈的非参自适应起重机控制系统，其中吊载跟踪期望轨迹并使用吊载位置反馈。通过部分反馈线性化，使用小车端点位置控制吊载运动。这通过引入一个新的基于笛卡尔坐标的新模型结构来实现。该笛卡尔模型结构使得能够在复制核希尔伯特空间（RKHS）中近似未知干扰力和结构未知动态的影响，从而消除干扰。证明了在存在未知力和未建模动态的情况下，非参自适应控制器导致均匀最终有界误差。此外，证明了在非自适应情况下，笛卡尔形式在吊载跟踪控制中具有某些优势。非参自适应控制器的性能在仿真和实验中得到了验证，结果良好。 

---
# GenOSIL: Generalized Optimal and Safe Robot Control using Parameter-Conditioned Imitation Learning 

**Title (ZH)**: GenOSIL：参数条件化imitation learning的通用最优与安全机器人控制 

**Authors**: Mumuksh Tayal, Manan Tayal, Ravi Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2503.12243)  

**Abstract**: Ensuring safe and generalizable control remains a fundamental challenge in robotics, particularly when deploying imitation learning in dynamic environments. Traditional behavior cloning (BC) struggles to generalize beyond its training distribution, as it lacks an understanding of the safety critical reasoning behind expert demonstrations. To address this limitation, we propose GenOSIL, a novel imitation learning framework that explicitly incorporates environment parameters into policy learning via a structured latent representation. Unlike conventional methods that treat the environment as a black box, GenOSIL employs a variational autoencoder (VAE) to encode measurable safety parameters such as obstacle position, velocity, and geometry into a latent space that captures intrinsic correlations between expert behavior and environmental constraints. This enables the policy to infer the rationale behind expert trajectories rather than merely replicating them. We validate our approach on two robotic platforms an autonomous ground vehicle and a Franka Emika Panda manipulator demonstrating superior safety and goal reaching performance compared to baseline methods. The simulation and hardware videos can be viewed on the project webpage: this https URL. 

**Abstract (ZH)**: 确保机器人在动态环境中安全且通用的控制仍然是一项基本挑战，特别是在部署模仿学习时。传统的行为克隆（BC）难以超越其训练分布进行泛化，因为它缺乏对专家示范背后的安全关键推理的理解。为了克服这一限制，我们提出了一种名为GenOSIL的新型模仿学习框架，该框架通过结构化的潜在表示显式地将环境参数纳入策略学习中。与将环境视为黑盒的传统方法不同，GenOSIL利用变分自编码器（VAE）将障碍物位置、速度和几何形状等可测量的安全参数编码到一个潜在空间中，该空间捕捉了专家行为与环境约束之间的内在关联。这使得策略能够推断专家轨迹背后的理由，而不仅仅是复制它们。我们在两个机器人平台上验证了这种方法——一个自主地面车辆和一个Franka Emika Panda 操作臂，结果显示其在安全性及目标达成性能方面优于基线方法。有关仿真和硬件视频，请参阅项目网页：this https URL。 

---
# D4orm: Multi-Robot Trajectories with Dynamics-aware Diffusion Denoised Deformations 

**Title (ZH)**: D4orm: 多机器人轨迹动态感知扩散去噪变形 

**Authors**: Yuhao Zhang, Keisuke Okumura, Heedo Woo, Ajay Shankar, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2503.12204)  

**Abstract**: This work presents an optimization method for generating kinodynamically feasible and collision-free multi-robot trajectories that exploits an incremental denoising scheme in diffusion models. Our key insight is that high-quality trajectories can be discovered merely by denoising noisy trajectories sampled from a distribution. This approach has no learning component, relying instead on only two ingredients: a dynamical model of the robots to obtain feasible trajectories via rollout, and a score function to guide denoising with Monte Carlo gradient approximation. The proposed framework iteratively optimizes the deformation from the previous round with this denoising process, allows \textit{anytime} refinement as time permits, supports different dynamics, and benefits from GPU acceleration. Our evaluations for differential-drive and holonomic teams with up to 16 robots in 2D and 3D worlds show its ability to discover high-quality solutions faster than other black-box optimization methods such as MPPI, approximately three times faster in a 3D holonomic case with 16 robots. As evidence for feasibility, we demonstrate zero-shot deployment of the planned trajectories on eight multirotors. 

**Abstract (ZH)**: 基于去噪方案的多机器人运动轨迹优化方法 

---
# Bench2FreeAD: A Benchmark for Vision-based End-to-end Navigation in Unstructured Robotic Environments 

**Title (ZH)**: Bench2FreeAD：基于视觉的面向未结构化机器人环境的端到端导航基准 

**Authors**: Yuhang Peng, Sidong Wang, Jihaoyu Yang, Shilong Li, Han Wang, Jiangtao Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.12180)  

**Abstract**: Most current end-to-end (E2E) autonomous driving algorithms are built on standard vehicles in structured transportation scenarios, lacking exploration of robot navigation for unstructured scenarios such as auxiliary roads, campus roads, and indoor settings. This paper investigates E2E robot navigation in unstructured road environments. First, we introduce two data collection pipelines - one for real-world robot data and another for synthetic data generated using the Isaac Sim simulator, which together produce an unstructured robotics navigation dataset -- FreeWorld Dataset. Second, we fine-tuned an efficient E2E autonomous driving model -- VAD -- using our datasets to validate the performance and adaptability of E2E autonomous driving models in these environments. Results demonstrate that fine-tuning through our datasets significantly enhances the navigation potential of E2E autonomous driving models in unstructured robotic environments. Thus, this paper presents the first dataset targeting E2E robot navigation tasks in unstructured scenarios, and provides a benchmark based on vision-based E2E autonomous driving algorithms to facilitate the development of E2E navigation technology for logistics and service robots. The project is available on Github. 

**Abstract (ZH)**: 当前大多数端到端（E2E）自动驾驶算法都是基于结构化交通场景的标准车辆构建的，缺乏对非结构化场景（如辅助道路、校园道路和室内环境）中机器人导航的探索。本文研究了非结构化道路环境中的端到端机器人导航。首先，我们介绍了两个数据采集管道——一个用于采集真实机器人数据，另一个利用Isaac Sim模拟器生成合成数据，共同构建了一个非结构化机器人导航数据集——FreeWorld数据集。其次，我们使用我们的数据集对一个高效的端到端自动驾驶模型——VAD——进行了微调，以验证端到端自动驾驶模型在这类环境中的性能和适应性。结果表明，通过我们的数据集微调显著增强了端到端自动驾驶模型在非结构化机器人环境中的导航潜力。因此，本文提出了首个针对非结构化场景中端到端机器人导航任务的数据集，并基于基于视觉的端到端自动驾驶算法提供了基准，以促进物流和服务机器人端到端导航技术的发展。该项目可在Github上获取。 

---
# DiffAD: A Unified Diffusion Modeling Approach for Autonomous Driving 

**Title (ZH)**: DiffAD: 自动驾驶统一扩散建模方法 

**Authors**: Tao Wang, Cong Zhang, Xingguang Qu, Kun Li, Weiwei Liu, Chang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12170)  

**Abstract**: End-to-end autonomous driving (E2E-AD) has rapidly emerged as a promising approach toward achieving full autonomy. However, existing E2E-AD systems typically adopt a traditional multi-task framework, addressing perception, prediction, and planning tasks through separate task-specific heads. Despite being trained in a fully differentiable manner, they still encounter issues with task coordination, and the system complexity remains high. In this work, we introduce DiffAD, a novel diffusion probabilistic model that redefines autonomous driving as a conditional image generation task. By rasterizing heterogeneous targets onto a unified bird's-eye view (BEV) and modeling their latent distribution, DiffAD unifies various driving objectives and jointly optimizes all driving tasks in a single framework, significantly reducing system complexity and harmonizing task coordination. The reverse process iteratively refines the generated BEV image, resulting in more robust and realistic driving behaviors. Closed-loop evaluations in Carla demonstrate the superiority of the proposed method, achieving a new state-of-the-art Success Rate and Driving Score. The code will be made publicly available. 

**Abstract (ZH)**: 端到端自主驾驶（E2E-AD）作为一种实现全自主驾驶的有前途的方法迅速 emergence。然而，现有的 E2E-AD 系统通常采用传统的多任务框架，分别通过特定任务的头部来解决感知、预测和规划任务。尽管是通过完全可微的方式进行训练，它们仍然面临任务协调问题，且系统复杂性仍然很高。在本工作中，我们提出了 DiffAD，一种新颖的扩散概率模型，重新定义自主驾驶为条件图像生成任务。通过将异构目标 rasterize 到统一的鸟瞰图（BEV）并建模其潜在分布，DiffAD 统一了各种驾驶目标，并在一个框架中联合优化所有驾驶任务，显著降低了系统复杂性并协调了任务间的协调。反转过程逐迭代细化生成的 BEV 图像，从而产生更稳健且真实的驾驶行为。在 Carla 中进行的闭环评估表明，所提出的方法具有优越性，实现了新的成功率（Success Rate）和驾驶评分（Driving Score）的最新成果。代码将公开发布。 

---
# ICCO: Learning an Instruction-conditioned Coordinator for Language-guided Task-aligned Multi-robot Control 

**Title (ZH)**: ICCO: 基于指令调节的多机器人任务对齐的语言引导控制协调器 

**Authors**: Yoshiki Yano, Kazuki Shibata, Maarten Kokshoorn, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2503.12122)  

**Abstract**: Recent advances in Large Language Models (LLMs) have permitted the development of language-guided multi-robot systems, which allow robots to execute tasks based on natural language instructions. However, achieving effective coordination in distributed multi-agent environments remains challenging due to (1) misalignment between instructions and task requirements and (2) inconsistency in robot behaviors when they independently interpret ambiguous instructions. To address these challenges, we propose Instruction-Conditioned Coordinator (ICCO), a Multi-Agent Reinforcement Learning (MARL) framework designed to enhance coordination in language-guided multi-robot systems. ICCO consists of a Coordinator agent and multiple Local Agents, where the Coordinator generates Task-Aligned and Consistent Instructions (TACI) by integrating language instructions with environmental states, ensuring task alignment and behavioral consistency. The Coordinator and Local Agents are jointly trained to optimize a reward function that balances task efficiency and instruction following. A Consistency Enhancement Term is added to the learning objective to maximize mutual information between instructions and robot behaviors, further improving coordination. Simulation and real-world experiments validate the effectiveness of ICCO in achieving language-guided task-aligned multi-robot control. The demonstration can be found at this https URL. 

**Abstract (ZH)**: 近期大规模语言模型的进展使语言引导的多机器人系统的发展成为可能，这些系统允许机器人根据自然语言指令执行任务。然而，在分布式多智能体环境中实现有效的协调仍然具有挑战性，原因包括（1）指令与任务需求之间的不一致，以及（2）当机器人独立解释含糊的指令时出现的行为不一致性。为了解决这些挑战，我们提出了一种多智能体强化学习框架——指令条件协调器（ICCO），旨在增强语言引导的多机器人系统中的协调性。ICCO包括一个协调器智能体和多个本地智能体，协调器通过整合语言指令与环境状态生成任务对齐且行为一致的指令（TACI），确保任务对齐和行为一致性。协调器和本地智能体共同训练以优化平衡任务效率和指令遵守的奖励函数。学习目标中加入了一致性增强项，以最大限度地提高指令与机器人行为之间的互信息，进一步提高协调性。模拟和实际实验验证了ICCO在实现语言引导的任务对齐多机器人控制方面的有效性。相关演示可访问此网址。 

---
# MUSE: A Real-Time Multi-Sensor State Estimator for Quadruped Robots 

**Title (ZH)**: MUSE：四足机器人实时多传感器状态估计器 

**Authors**: Ylenia Nisticò, João Carlos Virgolino Soares, Lorenzo Amatucci, Geoff Fink, Claudio Semini  

**Link**: [PDF](https://arxiv.org/pdf/2503.12101)  

**Abstract**: This paper introduces an innovative state estimator, MUSE (MUlti-sensor State Estimator), designed to enhance state estimation's accuracy and real-time performance in quadruped robot navigation. The proposed state estimator builds upon our previous work presented in [1]. It integrates data from a range of onboard sensors, including IMUs, encoders, cameras, and LiDARs, to deliver a comprehensive and reliable estimation of the robot's pose and motion, even in slippery scenarios. We tested MUSE on a Unitree Aliengo robot, successfully closing the locomotion control loop in difficult scenarios, including slippery and uneven terrain. Benchmarking against Pronto [2] and VILENS [3] showed 67.6% and 26.7% reductions in translational errors, respectively. Additionally, MUSE outperformed DLIO [4], a LiDAR-inertial odometry system in rotational errors and frequency, while the proprioceptive version of MUSE (P-MUSE) outperformed TSIF [5], with a 45.9% reduction in absolute trajectory error (ATE). 

**Abstract (ZH)**: 一种多传感器状态估计器MUSE在四足机器人导航中提升状态估计准确性和实时性能的研究 

---
# Maritime Mission Planning for Unmanned Surface Vessel using Large Language Model 

**Title (ZH)**: 基于大型语言模型的无人驾驶表面船舶航海任务规划 

**Authors**: Muhayy Ud Din, Waseem Akram, Ahsan B Bakht, Yihao Dong, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2503.12065)  

**Abstract**: Unmanned Surface Vessels (USVs) are essential for various maritime operations. USV mission planning approach offers autonomous solutions for monitoring, surveillance, and logistics. Existing approaches, which are based on static methods, struggle to adapt to dynamic environments, leading to suboptimal performance, higher costs, and increased risk of failure. This paper introduces a novel mission planning framework that uses Large Language Models (LLMs), such as GPT-4, to address these challenges. LLMs are proficient at understanding natural language commands, executing symbolic reasoning, and flexibly adjusting to changing situations. Our approach integrates LLMs into maritime mission planning to bridge the gap between high-level human instructions and executable plans, allowing real-time adaptation to environmental changes and unforeseen obstacles. In addition, feedback from low-level controllers is utilized to refine symbolic mission plans, ensuring robustness and adaptability. This framework improves the robustness and effectiveness of USV operations by integrating the power of symbolic planning with the reasoning abilities of LLMs. In addition, it simplifies the mission specification, allowing operators to focus on high-level objectives without requiring complex programming. The simulation results validate the proposed approach, demonstrating its ability to optimize mission execution while seamlessly adapting to dynamic maritime conditions. 

**Abstract (ZH)**: 无人水面船舶（USVs）在各种海运操作中至关重要。USV任务规划方法提供了自主解决监测、监视和物流问题的方案。现有的基于静态方法的方案难以适应动态环境，导致性能不佳、成本增加和故障风险增加。本文提出了一种新的任务规划框架，利用大型语言模型（LLMs），如GPT-4，来解决这些挑战。LLMs擅长理解自然语言指令、执行符号推理，并能灵活适应不断变化的情况。我们提出的方法将LLMs集成到海运任务规划中，以填补高层次人类指令与可执行计划之间的差距，实现对环境变化和未预见障碍的实时适应。此外，低级控制器的反馈被用来细化符号任务计划，确保其稳健性和适应性。该框架通过结合符号规划的力量和LLMs的推理能力，提高了USV操作的稳健性和有效性。此外，它简化了任务规范，使操作员能够专注于高层次目标，而无需复杂的编程。模拟结果验证了该方法的有效性，证明了其在动态海运条件下无缝优化任务执行的能力。 

---
# Generative Modeling of Adversarial Lane-Change Scenario 

**Title (ZH)**: 生成建模敌对变道场景 

**Authors**: Chuancheng Zhang, Zhenhao Wang, Jiangcheng Wang, Kun Su, Qiang Lv, Bin Jiang, Kunkun Hao, Wenyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12055)  

**Abstract**: Decision-making in long-tail scenarios is crucial to autonomous driving development, with realistic and challenging simulations playing a pivotal role in testing safety-critical situations. However, the current open-source datasets do not systematically include long-tail distributed scenario data, making acquiring such scenarios a formidable task. To address this problem, a data mining framework is proposed, which performs in-depth analysis on two widely-used datasets, NGSIM and INTERACTION, to pinpoint data with hazardous behavioral traits, aiming to bridge the gap in these overlooked scenarios. The approach utilizes Generative Adversarial Imitation Learning (GAIL) based on an enhanced Proximal Policy Optimization (PPO) model, integrated with the vehicle's environmental analysis, to iteratively refine and represent the newly generated vehicle trajectory. Innovatively, the solution optimizes the generation of adversarial scenario data from the perspectives of sensitivity and reasonable adversarial. It is demonstrated through experiments that, compared to the unfiltered data and baseline models, the approach exhibits more adversarial yet natural behavior regarding collision rate, acceleration, and lane changes, thereby validating its suitability for generating scenario data and providing constructive insights for the development of future scenarios and subsequent decision training. 

**Abstract (ZH)**: 长尾场景下的决策对自动驾驶发展至关重要，现实且具挑战性的模拟在测试安全关键情况中扮演着关键角色。然而，当前的开源数据集并未系统性地包含长尾分布场景数据，使得获取这些场景变得极具挑战。为解决这一问题，提出了一种数据挖掘框架，该框架对广泛使用的NGSIM和INTERACTION两个数据集进行深入分析，以识别具有危险行为特征的数据，旨在填补这些被忽视场景的空白。该方法利用基于增强的渐进策略优化（PPO）模型的生成对抗模仿学习（GAIL），结合车辆环境分析，以迭代方式细化和代表新生成的车辆轨迹。创新地，该解决方案从敏感性和合理的对抗性视角优化 adversarial 场景数据的生成。实验结果表明，与未经筛选的数据和基准模型相比，该方法在碰撞率、加速度和车道变更方面表现出更敌对但自然的行为，从而验证了其生成场景数据的适宜性和对未来场景开发以及后续决策训练的建设性洞察。 

---
# Hierarchical Reinforcement Learning for Safe Mapless Navigation with Congestion Estimation 

**Title (ZH)**: 层次强化学习在拥堵估计下的安全无地图导航 

**Authors**: Jianqi Gao, Xizheng Pang, Qi Liu, Yanjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.12036)  

**Abstract**: Reinforcement learning-based mapless navigation holds significant potential. However, it faces challenges in indoor environments with local minima area. This paper introduces a safe mapless navigation framework utilizing hierarchical reinforcement learning (HRL) to enhance navigation through such areas. The high-level policy creates a sub-goal to direct the navigation process. Notably, we have developed a sub-goal update mechanism that considers environment congestion, efficiently avoiding the entrapment of the robot in local minimum areas. The low-level motion planning policy, trained through safe reinforcement learning, outputs real-time control instructions based on acquired sub-goal. Specifically, to enhance the robot's environmental perception, we introduce a new obstacle encoding method that evaluates the impact of obstacles on the robot's motion planning. To validate the performance of our HRL-based navigation framework, we conduct simulations in office, home, and restaurant environments. The findings demonstrate that our HRL-based navigation framework excels in both static and dynamic scenarios. Finally, we implement the HRL-based navigation framework on a TurtleBot3 robot for physical validation experiments, which exhibits its strong generalization capabilities. 

**Abstract (ZH)**: 基于强化学习的无地图导航具有显著潜力。然而，在具有局部极小区域的室内环境中，它面临着挑战。本文介绍了一种利用层次强化学习（HRL）的安全无地图导航框架，以增强通过此类区域的导航能力。高层次策略创建子目标以引导导航过程。值得注意的是，我们开发了一种考虑环境拥堵的子目标更新机制，有效地避免了机器人陷入局部极小区域。低层次运动规划策略通过对安全强化学习的训练，根据获取的子目标实时输出控制指令。具体而言，为了增强机器人的环境感知，我们引入了一种新的障碍编码方法，评估障碍物对机器人运动规划的影响。为了验证基于HRL的导航框架的性能，我们在办公室、家庭和餐厅环境中进行了模拟。研究发现，基于HRL的导航框架在静态和动态场景中均表现出色。最后，我们在TurtleBot3机器人上实现了基于HRL的导航框架，进行了物理验证实验，展示了其强大的泛化能力。 

---
# Hydra-NeXt: Robust Closed-Loop Driving with Open-Loop Training 

**Title (ZH)**: Hydra-NeXt: 开环训练下的鲁棒闭环驾驶 

**Authors**: Zhenxin Li, Shihao Wang, Shiyi Lan, Zhiding Yu, Zuxuan Wu, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2503.12030)  

**Abstract**: End-to-end autonomous driving research currently faces a critical challenge in bridging the gap between open-loop training and closed-loop deployment. Current approaches are trained to predict trajectories in an open-loop environment, which struggle with quick reactions to other agents in closed-loop environments and risk generating kinematically infeasible plans due to the gap between open-loop training and closed-loop driving. In this paper, we introduce Hydra-NeXt, a novel multi-branch planning framework that unifies trajectory prediction, control prediction, and a trajectory refinement network in one model. Unlike current open-loop trajectory prediction models that only handle general-case planning, Hydra-NeXt further utilizes a control decoder to focus on short-term actions, which enables faster responses to dynamic situations and reactive agents. Moreover, we propose the Trajectory Refinement module to augment and refine the planning decisions by effectively adhering to kinematic constraints in closed-loop environments. This unified approach bridges the gap between open-loop training and closed-loop driving, demonstrating superior performance of 65.89 Driving Score (DS) and 48.20% Success Rate (SR) on the Bench2Drive dataset without relying on external experts for data collection. Hydra-NeXt surpasses the previous state-of-the-art by 22.98 DS and 17.49 SR, marking a significant advancement in autonomous driving. Code will be available at this https URL. 

**Abstract (ZH)**: 端到端自动驾驶研究目前面临一个关键挑战，即弥合开环训练与闭环部署之间的差距。当前的方法在开环环境中训练以预测轨迹，但在闭环环境中难以快速反应其他代理，并且由于开环训练与闭环驾驶之间的差距，存在生成动力学上不可行的计划的风险。在本文中，我们介绍了Hydra-NeXt，一种新颖的多分支规划框架，该框架将轨迹预测、控制预测和轨迹精炼网络统一在一个模型中。与仅处理一般案例规划的当前开环轨迹预测模型不同，Hydra-NeXt进一步利用控制解码器专注于短期动作，从而能够更快地响应动态情况和反应型代理。此外，我们提出了轨迹精炼模块，通过有效遵守闭环环境中动力学约束来增强和精炼规划决策。这种统一的方法弥合了开环训练与闭环驾驶之间的差距，在不依赖外部专家数据收集的情况下，Hydra-NeXt在Bench2Drive数据集上实现了65.89的驾驶得分（DS）和48.20%的成功率（SR），优于之前最先进的方法22.98 DS和17.49 SR，标志着自动驾驶领域的一个重要进步。代码将在以下链接获取：这个httpsURL。 

---
# Non-Normalized Solutions of Generalized Nash Equilibrium in Autonomous Racing 

**Title (ZH)**: 广义纳什均衡在自动驾驶赛车中的非正规解 

**Authors**: Mark Pustilnik, Francesco Borrelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.12002)  

**Abstract**: In dynamic games with shared constraints, Generalized Nash Equilibria (GNE) are often computed using the normalized solution concept, which assumes identical Lagrange multipliers for shared constraints across all players. While widely used, this approach excludes other potentially valuable GNE. This paper addresses the limitations of normalized solutions in racing scenarios through three key contributions. First, we highlight the shortcomings of normalized solutions with a simple racing example. Second, we propose a novel method based on the Mixed Complementarity Problem (MCP) formulation to compute non-normalized Generalized Nash Equilibria (GNE). Third, we demonstrate that our proposed method overcomes the limitations of normalized GNE solutions and enables richer multi-modal interactions in realistic racing scenarios. 

**Abstract (ZH)**: 在共享约束的动态博弈中，通用纳什均衡（GNE）通常使用正则化解的概念进行计算，该概念假定所有玩家对共享约束的拉格朗日乘子相同。虽然被广泛使用，但这种方法排除了其他潜在有价值的GNE。本文通过三个方面解决了正则化解在竞速场景中的局限性。首先，我们通过一个简单的竞速示例突显了正则化解的不足。其次，我们提出了一种基于混合互补问题（MCP）形式化的新方法来计算非正则化通用纳什均衡（GNE）。最后，我们证明了我们提出的方法克服了正则化GNE解的局限性，并能在现实竞速场景中实现更丰富的多模态互动。 

---
# Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation 

**Title (ZH)**: cloth manipulation用扩散动力学模型与生成状态估计方法 

**Authors**: Tongxuan Tian, Haoyang Li, Bo Ai, Xiaodi Yuan, Zhiao Huang, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.11999)  

**Abstract**: Manipulating deformable objects like cloth is challenging due to their complex dynamics, near-infinite degrees of freedom, and frequent self-occlusions, which complicate state estimation and dynamics modeling. Prior work has struggled with robust cloth state estimation, while dynamics models, primarily based on Graph Neural Networks (GNNs), are limited by their locality. Inspired by recent advances in generative models, we hypothesize that these expressive models can effectively capture intricate cloth configurations and deformation patterns from data. Building on this insight, we propose a diffusion-based generative approach for both perception and dynamics modeling. Specifically, we formulate state estimation as reconstructing the full cloth state from sparse RGB-D observations conditioned on a canonical cloth mesh and dynamics modeling as predicting future states given the current state and robot actions. Leveraging a transformer-based diffusion model, our method achieves high-fidelity state reconstruction while reducing long-horizon dynamics prediction errors by an order of magnitude compared to GNN-based approaches. Integrated with model-predictive control (MPC), our framework successfully executes cloth folding on a real robotic system, demonstrating the potential of generative models for manipulation tasks with partial observability and complex dynamics. 

**Abstract (ZH)**: 基于扩散生成模型的可变形物体感知与动力学建模 

---
# Sketch-to-Skill: Bootstrapping Robot Learning with Human Drawn Trajectory Sketches 

**Title (ZH)**: 从素描到技能：基于人类绘制轨迹素描的人工智能机器人学习启动方法 

**Authors**: Peihong Yu, Amisha Bhaskar, Anukriti Singh, Zahiruddin Mahammad, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2503.11918)  

**Abstract**: Training robotic manipulation policies traditionally requires numerous demonstrations and/or environmental rollouts. While recent Imitation Learning (IL) and Reinforcement Learning (RL) methods have reduced the number of required demonstrations, they still rely on expert knowledge to collect high-quality data, limiting scalability and accessibility. We propose Sketch-to-Skill, a novel framework that leverages human-drawn 2D sketch trajectories to bootstrap and guide RL for robotic manipulation. Our approach extends beyond previous sketch-based methods, which were primarily focused on imitation learning or policy conditioning, limited to specific trained tasks. Sketch-to-Skill employs a Sketch-to-3D Trajectory Generator that translates 2D sketches into 3D trajectories, which are then used to autonomously collect initial demonstrations. We utilize these sketch-generated demonstrations in two ways: to pre-train an initial policy through behavior cloning and to refine this policy through RL with guided exploration. Experimental results demonstrate that Sketch-to-Skill achieves ~96% of the performance of the baseline model that leverages teleoperated demonstration data, while exceeding the performance of a pure reinforcement learning policy by ~170%, only from sketch inputs. This makes robotic manipulation learning more accessible and potentially broadens its applications across various domains. 

**Abstract (ZH)**: 基于草图的技能训练：一种将手绘二维草图轨迹用于引导机器人操作的强化学习框架 

---
# Learning-based Estimation of Forward Kinematics for an Orthotic Parallel Robotic Mechanism 

**Title (ZH)**: 基于学习的正向运动学估计方法研究——用于矫形并联机器人机制 

**Authors**: Jingzong Zhou, Yuhan Zhu, Xiaobin Zhang, Sunil Agrawal, Konstantinos Karydis  

**Link**: [PDF](https://arxiv.org/pdf/2503.11855)  

**Abstract**: This paper introduces a 3D parallel robot with three identical five-degree-of-freedom chains connected to a circular brace end-effector, aimed to serve as an assistive device for patients with cervical spondylosis. The inverse kinematics of the system is solved analytically, whereas learning-based methods are deployed to solve the forward kinematics. The methods considered herein include a Koopman operator-based approach as well as a neural network-based approach. The task is to predict the position and orientation of end-effector trajectories. The dataset used to train these methods is based on the analytical solutions derived via inverse kinematics. The methods are tested both in simulation and via physical hardware experiments with the developed robot. Results validate the suitability of deploying learning-based methods for studying parallel mechanism forward kinematics that are generally hard to resolve analytically. 

**Abstract (ZH)**: 本文介绍了一种用于颈椎病患者辅助康复的具有三个 identical 五自由度链并连接到圆形支架末端执行器的3D并联机器人。系统的逆运动学通过解析方法解决，而基于学习的方法被用于解决正向运动学问题。所考虑的方法包括基于Koopman算子的方法以及基于神经网络的方法。任务是预测末端执行器轨迹的位置和姿态。用于训练这些方法的数据集基于通过逆运动学推导出的解析解。该方法在仿真和所开发的机器人硬件实验中进行了测试。结果验证了使用基于学习的方法研究一般难以解析的并联机构正向运动学问题的适用性。 

---
# Safe Multi-Robotic Arm Interaction via 3D Convex Shapes 

**Title (ZH)**: 基于3D凸形状的多机器人臂安全交互 

**Authors**: Ali Umut Kaypak, Shiqing Wei, Prashanth Krishnamurthy, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2503.11791)  

**Abstract**: Inter-robot collisions pose a significant safety risk when multiple robotic arms operate in close proximity. We present an online collision avoidance methodology leveraging 3D convex shape-based High-Order Control Barrier Functions (HOCBFs) to address this issue. While prior works focused on using Control Barrier Functions (CBFs) for human-robotic arm and single-arm collision avoidance, we explore the problem of collision avoidance between multiple robotic arms operating in a shared space. In our methodology, we utilize the proposed HOCBFs as centralized and decentralized safety filters. These safety filters are compatible with any nominal controller and ensure safety without significantly restricting the robots' workspace. A key challenge in implementing these filters is the computational overhead caused by the large number of safety constraints and the computation of a Hessian matrix per constraint. We address this challenge by employing numerical differentiation methods to approximate computationally intensive terms. The effectiveness of our method is demonstrated through extensive simulation studies and real-world experiments with Franka Research 3 robotic arms. 

**Abstract (ZH)**: 多机器人手臂在共享空间中的碰撞避免方法：基于3D凸形高阶控制屏障函数的在线碰撞避免技术 

---
# Controllable Latent Diffusion for Traffic Simulation 

**Title (ZH)**: 可控潜在扩散交通模拟 

**Authors**: Yizhuo Xiao, Mustafa Suphi Erden, Cheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11771)  

**Abstract**: The validation of autonomous driving systems benefits greatly from the ability to generate scenarios that are both realistic and precisely controllable. Conventional approaches, such as real-world test drives, are not only expensive but also lack the flexibility to capture targeted edge cases for thorough evaluation. To address these challenges, we propose a controllable latent diffusion that guides the training of diffusion models via reinforcement learning to automatically generate a diverse and controllable set of driving scenarios for virtual testing. Our approach removes the reliance on large-scale real-world data by generating complex scenarios whose properties can be finely tuned to challenge and assess autonomous vehicle systems. Experimental results show that our approach has the lowest collision rate of $0.098$ and lowest off-road rate of $0.096$, demonstrating superiority over existing baselines. The proposed approach significantly improves the realism, stability and controllability of the generated scenarios, enabling more nuanced safety evaluation of autonomous vehicles. 

**Abstract (ZH)**: 自主驾驶系统的验证极大地受益于能够生成既逼真又可精确控制的场景的能力。传统的方法，如实地测试驾驶，不仅成本高，而且缺乏灵活性，无法捕获有针对性的边界案例以进行彻底的评估。为了解决这些问题，我们提出了一种可控的潜在扩散方法，通过强化学习引导扩散模型的训练，以自动生成多样且可控的驾驶场景进行虚拟测试。该方法通过生成复杂场景，使其性质可以精细调整，从而挑战和评估自主车辆系统。实验结果表明，我们的方法具有最低的碰撞率（0.098）和最低的离路率（0.096），证明了其优于现有基线的方法。所提出的方法显著提高了生成场景的逼真性、稳定性和可控性，从而能够进行更为细致的自主车辆安全性评估。 

---
# A Smooth Analytical Formulation of Collision Detection and Rigid Body Dynamics With Contact 

**Title (ZH)**: 光滑的分析形式化表达碰撞检测与刚体动力学中的接触 

**Authors**: Onur Beker, Nico Gürtler, Ji Shi, A. René Geist, Amirreza Razmjoo, Georg Martius, Sylvain Calinon  

**Link**: [PDF](https://arxiv.org/pdf/2503.11736)  

**Abstract**: Generating intelligent robot behavior in contact-rich settings is a research problem where zeroth-order methods currently prevail. A major contributor to the success of such methods is their robustness in the face of non-smooth and discontinuous optimization landscapes that are characteristic of contact interactions, yet zeroth-order methods remain computationally inefficient. It is therefore desirable to develop methods for perception, planning and control in contact-rich settings that can achieve further efficiency by making use of first and second order information (i.e., gradients and Hessians). To facilitate this, we present a joint formulation of collision detection and contact modelling which, compared to existing differentiable simulation approaches, provides the following benefits: i) it results in forward and inverse dynamics that are entirely analytical (i.e. do not require solving optimization or root-finding problems with iterative methods) and smooth (i.e. twice differentiable), ii) it supports arbitrary collision geometries without needing a convex decomposition, and iii) its runtime is independent of the number of contacts. Through simulation experiments, we demonstrate the validity of the proposed formulation as a "physics for inference" that can facilitate future development of efficient methods to generate intelligent contact-rich behavior. 

**Abstract (ZH)**: 在接触丰富的环境中生成智能机器人行为是一个研究问题，当前主要依赖零阶方法。这些方法的成功很大程度上得益于其在非光滑和不连续优化景观面前的稳健性，这些景观是接触交互的特征，然而零阶方法仍然存在计算效率低下的问题。因此，有必要开发能够在接触丰富的环境中利用一阶和二阶信息（即梯度和海森矩阵）来实现更高效率的方法，用于感知、规划和控制。为此，我们提出了碰撞检测和接触建模的联合形式，与现有的可微模拟方法相比，具有以下优点：i) 涉及前向和逆向动力学完全为解析形式（不需要使用迭代方法求解优化或根查找问题），且平滑（即二阶可微），ii) 支持任意碰撞几何形状，无需进行凸分解，iii) 运行时间与接触点数量无关。通过模拟实验，我们证明了所提出的联合形式在作为“物理推断的基础”方面的有效性，这将有助于未来高效方法的发展，用于生成智能的接触丰富行为。 

---
# A Robust and Energy-Efficient Trajectory Planning Framework for High-Degree-of-Freedom Robots 

**Title (ZH)**: 高自由度机器人 robust 而且节能的轨迹规划框架 

**Authors**: Sajjad Hussain, Md Saad, Almas Baimagambetov, Khizer Saeed  

**Link**: [PDF](https://arxiv.org/pdf/2503.11716)  

**Abstract**: Energy efficiency and motion smoothness are essential in trajectory planning for high-degree-of-freedom robots to ensure optimal performance and reduce mechanical wear. This paper presents a novel framework integrating sinusoidal trajectory generation with velocity scaling to minimize energy consumption while maintaining motion accuracy and smoothness. The framework is evaluated using a physics-based simulation environment with metrics such as energy consumption, motion smoothness, and trajectory accuracy. Results indicate significant energy savings and smooth transitions, demonstrating the framework's effectiveness for precision-based applications. Future work includes real-time trajectory adjustments and enhanced energy models. 

**Abstract (ZH)**: 高自由度机器人轨迹规划中的能量效率和运动平滑性是确保最优性能和减少机械磨损的关键。本文提出了一种结合正弦轨迹生成与速度缩放的新型框架，以最小化能量消耗同时保持运动精度和平滑性。该框架使用基于物理的仿真环境进行评估，采用能量消耗、运动平滑性和轨迹精度等指标。结果显示显著的能量节省和平滑过渡，证明了该框架在基于精度的应用中的有效性。未来工作包括实时轨迹调整和增强的能量模型。 

---
# FloPE: Flower Pose Estimation for Precision Pollination 

**Title (ZH)**: Flower Pose Estimation for Precision Pollination 

**Authors**: Rashik Shrestha, Madhav Rijal, Trevor Smith, Yu Gu  

**Link**: [PDF](https://arxiv.org/pdf/2503.11692)  

**Abstract**: This study presents Flower Pose Estimation (FloPE), a real-time flower pose estimation framework for computationally constrained robotic pollination systems. Robotic pollination has been proposed to supplement natural pollination to ensure global food security due to the decreased population of natural pollinators. However, flower pose estimation for pollination is challenging due to natural variability, flower clusters, and high accuracy demands due to the flowers' fragility when pollinating. This method leverages 3D Gaussian Splatting to generate photorealistic synthetic datasets with precise pose annotations, enabling effective knowledge distillation from a high-capacity teacher model to a lightweight student model for efficient inference. The approach was evaluated on both single and multi-arm robotic platforms, achieving a mean pose estimation error of 0.6 cm and 19.14 degrees within a low computational cost. Our experiments validate the effectiveness of FloPE, achieving up to 78.75% pollination success rate and outperforming prior robotic pollination techniques. 

**Abstract (ZH)**: Flower Pose Estimation (FloPE)：受计算约束的机器人授粉系统中的实时花朵姿态估计框架 

---
# AugMapNet: Improving Spatial Latent Structure via BEV Grid Augmentation for Enhanced Vectorized Online HD Map Construction 

**Title (ZH)**: AugMapNet：通过BEV网格增强提高空间潜在结构构建增强向量在线高清地图构建 

**Authors**: Thomas Monninger, Md Zafar Anwar, Stanislaw Antol, Steffen Staab, Sihao Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.13430)  

**Abstract**: Autonomous driving requires an understanding of the infrastructure elements, such as lanes and crosswalks. To navigate safely, this understanding must be derived from sensor data in real-time and needs to be represented in vectorized form. Learned Bird's-Eye View (BEV) encoders are commonly used to combine a set of camera images from multiple views into one joint latent BEV grid. Traditionally, from this latent space, an intermediate raster map is predicted, providing dense spatial supervision but requiring post-processing into the desired vectorized form. More recent models directly derive infrastructure elements as polylines using vectorized map decoders, providing instance-level information. Our approach, Augmentation Map Network (AugMapNet), proposes latent BEV grid augmentation, a novel technique that significantly enhances the latent BEV representation. AugMapNet combines vector decoding and dense spatial supervision more effectively than existing architectures while remaining as straightforward to integrate and as generic as auxiliary supervision. Experiments on nuScenes and Argoverse2 datasets demonstrate significant improvements in vectorized map prediction performance up to 13.3% over the StreamMapNet baseline on 60m range and greater improvements on larger ranges. We confirm transferability by applying our method to another baseline and find similar improvements. A detailed analysis of the latent BEV grid confirms a more structured latent space of AugMapNet and shows the value of our novel concept beyond pure performance improvement. The code will be released soon. 

**Abstract (ZH)**: 自主驾驶需要理解基础设施元素，如车道和人行横道。为了安全导航，这种理解必须从多视角传感器数据中实时提取，并以矢量化形式表示。学习到的bird-eye视图（BEV）编码器常用于将多个视角的摄像头图像组合成一个联合的潜在BEV网格。传统的方法是从这个潜在空间预测一个中间的栅格化地图，提供密集的空间监督，但需要后处理成所需矢量化形式。更近的模型直接使用矢量化地图解码器提取基础设施元素为多边形，提供实例级信息。我们的方法，增强地图网络（AugMapNet），提出潜在BEV网格增强，这是一种新颖的技术，显著增强了潜在BEV表示。AugMapNet比现有架构更有效地结合了矢量解码和密集空间监督，同时保持了易集成和通用性与辅助监督相当。在nuScenes和Argoverse2数据集上的实验显示，与StreamMapNet基线相比，在60m范围内矢量化地图预测性能提高了13.3%，在更长距离上有更大的提高。我们通过将该方法应用于另一基线验证了其可迁移性，发现类似的改进。对潜在BEV网格的详细分析证实了AugMapNet具有更结构化的潜在空间，并展示了我们新颖概念的价值，超越了单纯的性能改进。代码即将发布。 

---
# 3D Hierarchical Panoptic Segmentation in Real Orchard Environments Across Different Sensors 

**Title (ZH)**: 跨不同传感器在真实果园环境中的3D分层全景分割 

**Authors**: Matteo Sodano, Federico Magistri, Elias Marks, Fares Hosn, Aibek Zurbayev, Rodrigo Marcuzzi, Meher V. R. Malladi, Jens Behley, Cyrill Stachniss  

**Link**: [PDF](https://arxiv.org/pdf/2503.13188)  

**Abstract**: Crop yield estimation is a relevant problem in agriculture, because an accurate crop yield estimate can support farmers' decisions on harvesting or precision intervention. Robots can help to automate this process. To do so, they need to be able to perceive the surrounding environment to identify target objects. In this paper, we introduce a novel approach to address the problem of hierarchical panoptic segmentation of apple orchards on 3D data from different sensors. Our approach is able to simultaneously provide semantic segmentation, instance segmentation of trunks and fruits, and instance segmentation of plants (a single trunk with its fruits). This allows us to identify relevant information such as individual plants, fruits, and trunks, and capture the relationship among them, such as precisely estimate the number of fruits associated to each tree in an orchard. Additionally, to efficiently evaluate our approach for hierarchical panoptic segmentation, we provide a dataset designed specifically for this task. Our dataset is recorded in Bonn in a real apple orchard with a variety of sensors, spanning from a terrestrial laser scanner to a RGB-D camera mounted on different robotic platforms. The experiments show that our approach surpasses state-of-the-art approaches in 3D panoptic segmentation in the agricultural domain, while also providing full hierarchical panoptic segmentation. Our dataset has been made publicly available at this https URL. We will provide the open-source implementation of our approach and public competiton for hierarchical panoptic segmentation on the hidden test sets upon paper acceptance. 

**Abstract (ZH)**: 基于多传感器3D数据的果园分层全景分割方法及其应用 

---
# Exploring 3D Activity Reasoning and Planning: From Implicit Human Intentions to Route-Aware Planning 

**Title (ZH)**: 探索3D活动推理与规划：从隐含人类意图到路径aware规划 

**Authors**: Xueying Jiang, Wenhao Li, Xiaoqin Zhang, Ling Shao, Shijian Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12974)  

**Abstract**: 3D activity reasoning and planning has attracted increasing attention in human-robot interaction and embodied AI thanks to the recent advance in multimodal learning. However, most existing works share two constraints: 1) heavy reliance on explicit instructions with little reasoning on implicit user intention; 2) negligence of inter-step route planning on robot moves. To bridge the gaps, we propose 3D activity reasoning and planning, a novel 3D task that reasons the intended activities from implicit instructions and decomposes them into steps with inter-step routes and planning under the guidance of fine-grained 3D object shapes and locations from scene segmentation. We tackle the new 3D task from two perspectives. First, we construct ReasonPlan3D, a large-scale benchmark that covers diverse 3D scenes with rich implicit instructions and detailed annotations for multi-step task planning, inter-step route planning, and fine-grained segmentation. Second, we design a novel framework that introduces progressive plan generation with contextual consistency across multiple steps, as well as a scene graph that is updated dynamically for capturing critical objects and their spatial relations. Extensive experiments demonstrate the effectiveness of our benchmark and framework in reasoning activities from implicit human instructions, producing accurate stepwise task plans, and seamlessly integrating route planning for multi-step moves. The dataset and code will be released. 

**Abstract (ZH)**: 三维活动推理解析与规划在多模态学习 recent 进展推动下的人机交互和具身AI中引起了广泛关注。然而，现有大部分工作存在两个限制：1) 过度依赖显式指令，而忽视用户的隐含意图推理；2) 忽视机器人移动过程中的跨步骤路径规划。为弥合这些差距，我们提出三维活动推理解析与规划，这是一种新颖的三维任务，能够从隐含指令中推理解析出意图活动，并将其分解为带有跨步骤路径和规划的步骤，同时借助场景分割提供的细粒度三维物体形状和位置指导。我们从两个视角来应对这一新的三维任务。首先，我们构建了ReasonPlan3D，这是一个大规模基准，涵盖了多种多样且包含丰富隐含指令的三维场景，并为多步骤任务规划、跨步骤路径规划和细粒度分割提供了详细的注释。其次，我们设计了一个新型框架，该框架引入了跨多步骤的上下文一致性渐进式计划生成，并设计了一个动态更新的场景图以捕捉关键物体及其空间关系。广泛的实验验证了我们在从隐含人类指令推理解析活动、产生准确的逐步任务规划以及无缝结合多步骤移动路径规划方面的有效性。数据集和代码将公开发布。 

---
# OptiPMB: Enhancing 3D Multi-Object Tracking with Optimized Poisson Multi-Bernoulli Filtering 

**Title (ZH)**: OptiPMB: 优化泊松多伯努利滤波以增强三维多目标跟踪 

**Authors**: Guanhua Ding, Yuxuan Xia, Runwei Guan, Qinchen Wu, Tao Huang, Weiping Ding, Jinping Sun, Guoqiang Mao  

**Link**: [PDF](https://arxiv.org/pdf/2503.12968)  

**Abstract**: Accurate 3D multi-object tracking (MOT) is crucial for autonomous driving, as it enables robust perception, navigation, and planning in complex environments. While deep learning-based solutions have demonstrated impressive 3D MOT performance, model-based approaches remain appealing for their simplicity, interpretability, and data efficiency. Conventional model-based trackers typically rely on random vector-based Bayesian filters within the tracking-by-detection (TBD) framework but face limitations due to heuristic data association and track management schemes. In contrast, random finite set (RFS)-based Bayesian filtering handles object birth, survival, and death in a theoretically sound manner, facilitating interpretability and parameter tuning. In this paper, we present OptiPMB, a novel RFS-based 3D MOT method that employs an optimized Poisson multi-Bernoulli (PMB) filter while incorporating several key innovative designs within the TBD framework. Specifically, we propose a measurement-driven hybrid adaptive birth model for improved track initialization, employ adaptive detection probability parameters to effectively maintain tracks for occluded objects, and optimize density pruning and track extraction modules to further enhance overall tracking performance. Extensive evaluations on nuScenes and KITTI datasets show that OptiPMB achieves superior tracking accuracy compared with state-of-the-art methods, thereby establishing a new benchmark for model-based 3D MOT and offering valuable insights for future research on RFS-based trackers in autonomous driving. 

**Abstract (ZH)**: 基于随机有限集的精确3D多目标跟踪方法 OptiPMB 

---
# Versatile Physics-based Character Control with Hybrid Latent Representation 

**Title (ZH)**: 基于混合潜在表示的多功能物理驱动角色控制 

**Authors**: Jinseok Bae, Jungdam Won, Donggeun Lim, Inwoo Hwang, Young Min Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.12814)  

**Abstract**: We present a versatile latent representation that enables physically simulated character to efficiently utilize motion priors. To build a powerful motion embedding that is shared across multiple tasks, the physics controller should employ rich latent space that is easily explored and capable of generating high-quality motion. We propose integrating continuous and discrete latent representations to build a versatile motion prior that can be adapted to a wide range of challenging control tasks. Specifically, we build a discrete latent model to capture distinctive posterior distribution without collapse, and simultaneously augment the sampled vector with the continuous residuals to generate high-quality, smooth motion without jittering. We further incorporate Residual Vector Quantization, which not only maximizes the capacity of the discrete motion prior, but also efficiently abstracts the action space during the task learning phase. We demonstrate that our agent can produce diverse yet smooth motions simply by traversing the learned motion prior through unconditional motion generation. Furthermore, our model robustly satisfies sparse goal conditions with highly expressive natural motions, including head-mounted device tracking and motion in-betweening at irregular intervals, which could not be achieved with existing latent representations. 

**Abstract (ZH)**: 我们提出了一种多功能的潜在表示，使物理模拟角色能够高效利用运动先验。为了建立一个强大的运动嵌入，该嵌入能够在多种任务中共享，物理控制器应当使用丰富且易于探索的潜在空间，能够生成高质量的运动。我们提出整合连续和离散的潜在表示，构建一个多功能的运动先验，适用于广泛挑战性的控制任务。具体而言，我们构建了一个离散潜变量模型来捕捉独特的后验分布且不会退化，并同时通过连续残差扩展采样向量，生成高质量且平滑的运动，而不会出现抖动现象。我们进一步引入了残差向量量化技术，不仅最大化了离散运动先验的能力，还在任务学习阶段高效地抽象了动作空间。我们证明，仅通过无条件运动生成遍历学习到的运动先验，我们的代理就能产生多样且平滑的运动。此外，我们的模型能够稳健地满足稀疏目标条件，包括具有高度表达力的自然运动，如头部显示器追踪和不规则间隔之间的运动插值，这是现有潜在表示无法实现的。 

---
# NuPlanQA: A Large-Scale Dataset and Benchmark for Multi-View Driving Scene Understanding in Multi-Modal Large Language Models 

**Title (ZH)**: NuPlanQA：多视图驾驶场景理解的大型数据集和基准测试集missive 

**Authors**: Sung-Yeon Park, Can Cui, Yunsheng Ma, Ahmadreza Moradipari, Rohit Gupta, Kyungtae Han, Ziran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12772)  

**Abstract**: Recent advances in multi-modal large language models (MLLMs) have demonstrated strong performance across various domains; however, their ability to comprehend driving scenes remains less proven. The complexity of driving scenarios, which includes multi-view information, poses significant challenges for existing MLLMs. In this paper, we introduce NuPlanQA-Eval, a multi-view, multi-modal evaluation benchmark for driving scene understanding. To further support generalization to multi-view driving scenarios, we also propose NuPlanQA-1M, a large-scale dataset comprising 1M real-world visual question-answering (VQA) pairs. For context-aware analysis of traffic scenes, we categorize our dataset into nine subtasks across three core skills: Road Environment Perception, Spatial Relations Recognition, and Ego-Centric Reasoning. Furthermore, we present BEV-LLM, integrating Bird's-Eye-View (BEV) features from multi-view images into MLLMs. Our evaluation results reveal key challenges that existing MLLMs face in driving scene-specific perception and spatial reasoning from ego-centric perspectives. In contrast, BEV-LLM demonstrates remarkable adaptability to this domain, outperforming other models in six of the nine subtasks. These findings highlight how BEV integration enhances multi-view MLLMs while also identifying key areas that require further refinement for effective adaptation to driving scenes. To facilitate further research, we publicly release NuPlanQA at this https URL. 

**Abstract (ZH)**: 近期多模态大规模语言模型在多个领域的表现得到了显著提升；然而，它们理解驾驶场景的能力仍需进一步验证。驾驶场景的复杂性，包括多视图信息，为现有的多模态大规模语言模型带来了重大挑战。本文介绍了NuPlanQA-Eval，一个用于驾驶场景理解的多视图多模态评估基准。为进一步支持多视图驾驶场景的泛化能力，我们还提出了包含100万真实世界视觉问答(VQA)配对数据集NuPlanQA-1M。为了进行交通场景的上下文感知分析，我们将数据集划分为包含九个子任务的三个核心技能：道路环境感知、空间关系识别和以自我为中心的推理。此外，我们介绍了BEV-LLM，将多视图图像中的鸟瞰图(BEV)特征集成到多模态大规模语言模型中。我们的评估结果显示，现有的多模态大规模语言模型在驾驶场景特定感知和以自我为中心的空间推理方面面临重大挑战。相比之下，BEV-LLM在九个子任务中的六个子任务中表现出显著的适应性，优于其他模型。这些发现突显了BEV集成如何增强多视图多模态语言模型，同时也指出了需要进一步改进的关键领域，以便更好地适应驾驶场景。为促进进一步研究，我们在<https://>公开发布了NuPlanQA。 

---
# MAP: Multi-user Personalization with Collaborative LLM-powered Agents 

**Title (ZH)**: MAP: 多用户个性化协作LLM代理 

**Authors**: Christine Lee, Jihye Choi, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2503.12757)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) and LLM-powered agents in multi-user settings underscores the need for reliable, usable methods to accommodate diverse preferences and resolve conflicting directives. Drawing on conflict resolution theory, we introduce a user-centered workflow for multi-user personalization comprising three stages: Reflection, Analysis, and Feedback. We then present MAP -- a \textbf{M}ulti-\textbf{A}gent system for multi-user \textbf{P}ersonalization -- to operationalize this workflow. By delegating subtasks to specialized agents, MAP (1) retrieves and reflects on relevant user information, while enhancing reliability through agent-to-agent interactions, (2) provides detailed analysis for improved transparency and usability, and (3) integrates user feedback to iteratively refine results. Our user study findings (n=12) highlight MAP's effectiveness and usability for conflict resolution while emphasizing the importance of user involvement in resolution verification and failure management. This work highlights the potential of multi-agent systems to implement user-centered, multi-user personalization workflows and concludes by offering insights for personalization in multi-user contexts. 

**Abstract (ZH)**: 大规模语言模型（LLMs）及其在多用户环境中的应用强调了需要可靠且用户友好的方法来满足多样化偏好并解决冲突指令。基于冲突解决理论，我们提出了一种以用户为中心的多用户个性化工作流，分为三个阶段：反思、分析和反馈。随后，我们介绍了一种针对多用户个性化需求的多代理系统MAP（Multi-Agent system for Multi-User Personalization），以实现该工作流。通过将子任务委托给专门的代理，MAP（1）检索并反思相关用户信息，通过代理间的交互提升可靠性，（2）提供详细的分析以增强透明度和易用性，（3）整合用户反馈以迭代优化结果。我们的用户研究结果（n=12）表明，MAP在冲突解决方面的有效性和易用性，同时强调了用户参与解决验证和故障管理的重要性。这项工作突显了多代理系统在实施用户为中心的多用户个性化工作流方面的潜力，并最终提出多用户环境下个性化方面的见解。 

---
# Agent-Based Simulation of UAV Battery Recharging for IoT Applications: Precision Agriculture, Disaster Recovery, and Dengue Vector Control 

**Title (ZH)**: 基于代理的无人机电池充电模拟研究：物联网应用中的精准农业、灾害恢复和登革热蚊媒控制 

**Authors**: Leonardo Grando, Juan Fernando Galindo Jaramillo, Jose Roberto Emiliano Leite, Edson Luiz Ursini  

**Link**: [PDF](https://arxiv.org/pdf/2503.12685)  

**Abstract**: The low battery autonomy of Unnamed Aerial Vehicles (UAVs or drones) can make smart farming (precision agriculture), disaster recovery, and the fighting against dengue vector applications difficult. This article considers two approaches, first enumerating the characteristics observed in these three IoT application types and then modeling an UAV's battery recharge coordination using the Agent-Based Simulation (ABS) approach. In this way, we propose that each drone inside the swarm does not communicate concerning this recharge coordination decision, reducing energy usage and permitting remote usage. A total of 6000 simulations were run to evaluate how two proposed policies, the BaseLine (BL) and ChargerThershold (CT) coordination recharging policy, behave in 30 situations regarding how each simulation sets conclude the simulation runs and how much time they work until recharging results. CT policy shows more reliable results in extreme system usage. This work conclusion presents the potential of these three IoT applications to achieve their perpetual service without communication between drones and ground stations. This work can be a baseline for future policies and simulation parameter enhancements. 

**Abstract (ZH)**: UAV电池续航有限性对智能农业、灾害恢复及防控登革热应用的挑战及其基于代理模型的电池再充电协调研究 

---
# Logic-RAG: Augmenting Large Multimodal Models with Visual-Spatial Knowledge for Road Scene Understanding 

**Title (ZH)**: 逻辑-RAG：增强大规模多模态模型的视觉空间知识以理解道路场景 

**Authors**: Imran Kabir, Md Alimoor Reza, Syed Billah  

**Link**: [PDF](https://arxiv.org/pdf/2503.12663)  

**Abstract**: Large multimodal models (LMMs) are increasingly integrated into autonomous driving systems for user interaction. However, their limitations in fine-grained spatial reasoning pose challenges for system interpretability and user trust. We introduce Logic-RAG, a novel Retrieval-Augmented Generation (RAG) framework that improves LMMs' spatial understanding in driving scenarios. Logic-RAG constructs a dynamic knowledge base (KB) about object-object relationships in first-order logic (FOL) using a perception module, a query-to-logic embedder, and a logical inference engine. We evaluated Logic-RAG on visual-spatial queries using both synthetic and real-world driving videos. When using popular LMMs (GPT-4V, Claude 3.5) as proxies for an autonomous driving system, these models achieved only 55% accuracy on synthetic driving scenes and under 75% on real-world driving scenes. Augmenting them with Logic-RAG increased their accuracies to over 80% and 90%, respectively. An ablation study showed that even without logical inference, the fact-based context constructed by Logic-RAG alone improved accuracy by 15%. Logic-RAG is extensible: it allows seamless replacement of individual components with improved versions and enables domain experts to compose new knowledge in both FOL and natural language. In sum, Logic-RAG addresses critical spatial reasoning deficiencies in LMMs for autonomous driving applications. Code and data are available at this https URL. 

**Abstract (ZH)**: 基于逻辑的检索增强生成（Logic-RAG）：提高自动驾驶场景中大模型的细粒度空间理解 

---
# Polytope Volume Monitoring Problem: Formulation and Solution via Parametric Linear Program Based Control Barrier Function 

**Title (ZH)**: 多面体体积监控问题：基于参数线性规划屏障函数的建模与求解 

**Authors**: Shizhen Wu, Jinyang Dong, Xu Fang, Ning Sun, Yongchun Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12546)  

**Abstract**: Motivated by the latest research on feasible space monitoring of multiple control barrier functions (CBFs) as well as polytopic collision avoidance, this paper studies the Polytope Volume Monitoring (PVM) problem, whose goal is to design a control law for inputs of nonlinear systems to prevent the volume of some state-dependent polytope from decreasing to zero. Recent studies have explored the idea of applying Chebyshev ball method in optimization theory to solve the case study of PVM; however, the underlying difficulties caused by nonsmoothness have not been addressed. This paper continues the study on this topic, where our main contribution is to establish the relationship between nonsmooth CBF and parametric optimization theory through directional derivatives for the first time, so as to solve PVM problems more conveniently. In detail, inspired by Chebyshev ball approach, a parametric linear program (PLP) based nonsmooth barrier function candidate is established for PVM, and then, sufficient conditions for it to be a nonsmooth CBF are proposed, based on which a quadratic program (QP) based safety filter with guaranteed feasibility is proposed to address PVM problems. Finally, a numerical simulation example is given to show the efficiency of the proposed safety filter. 

**Abstract (ZH)**: 基于最新关于多个控制 barrier 函数（CBFs）可行空间监控及多面体避碰的最新研究，本文研究了多面体容积监控（PVM）问题，其目标是设计非线性系统输入的控制律以防止某些状态相关的多面体体积减小至零。近期研究探索了将切比雪夫球方法应用于优化理论以解决PVM案例，但底层的非光滑性导致的困难尚未解决。本文继续探讨这一课题，我们的主要贡献是通过方向导数首次建立了非光滑CBF与参数优化理论之间的关系，以便更方便地解决PVM问题。具体而言，受到切比雪夫球方法的启发，我们为PVM建立了一个基于参数线性规划（PLP）的非光滑 barrier 函数候选，并提出了使其成为非光滑CBF的充分条件，基于此提出了一种基于二次规划（QP）的安全滤波器以确保可行性来解决PVM问题。最后，给出了一个数值仿真例子以展示所提出的安全滤波器的效率。 

---
# EgoEvGesture: Gesture Recognition Based on Egocentric Event Camera 

**Title (ZH)**: 基于第一人称事件相机的手势识别：EgoEvGesture 

**Authors**: Luming Wang, Hao Shi, Xiaoting Yin, Kailun Yang, Kaiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12419)  

**Abstract**: Egocentric gesture recognition is a pivotal technology for enhancing natural human-computer interaction, yet traditional RGB-based solutions suffer from motion blur and illumination variations in dynamic scenarios. While event cameras show distinct advantages in handling high dynamic range with ultra-low power consumption, existing RGB-based architectures face inherent limitations in processing asynchronous event streams due to their synchronous frame-based nature. Moreover, from an egocentric perspective, event cameras record data that include events generated by both head movements and hand gestures, thereby increasing the complexity of gesture recognition. To address this, we propose a novel network architecture specifically designed for event data processing, incorporating (1) a lightweight CNN with asymmetric depthwise convolutions to reduce parameters while preserving spatiotemporal features, (2) a plug-and-play state-space model as context block that decouples head movement noise from gesture dynamics, and (3) a parameter-free Bins-Temporal Shift Module (BSTM) that shifts features along bins and temporal dimensions to fuse sparse events efficiently. We further build the EgoEvGesture dataset, the first large-scale dataset for egocentric gesture recognition using event cameras. Experimental results demonstrate that our method achieves 62.7% accuracy in heterogeneous testing with only 7M parameters, 3.1% higher than state-of-the-art approaches. Notable misclassifications in freestyle motions stem from high inter-personal variability and unseen test patterns differing from training data. Moreover, our approach achieved a remarkable accuracy of 96.97% on DVS128 Gesture, demonstrating strong cross-dataset generalization capability. The dataset and models are made publicly available at this https URL. 

**Abstract (ZH)**: 自中心手势识别是提升自然人机交互的关键技术，但传统的基于RGB的方法在动态场景中受到运动模糊和光照变化的影响。虽然事件相机在处理高动态范围时表现出色且能耗极低，但现有的基于RGB的架构由于其基于同步帧的方式，在处理异步事件流时存在固有的局限性。此外，从自中心视角来看，事件相机记录的数据包括由头部运动和手部手势共同生成的事件，这增加了手势识别的复杂性。为了解决这一问题，我们提出了一种专为事件数据处理设计的新颖网络架构，包括（1）一种轻量级的CNN，采用不对称深度可分离卷积以减少参数同时保留时空特征，（2）一种可插拔的状态空间模型作为上下文块，可以解耦头部运动噪声与手势动力学，以及（3）一种无需参数的Bins-Time Shift模块（BSTM），沿 bins 和时间维度平移特征以高效融合稀疏事件。我们还构建了EgoEvGesture数据集，这是第一个用于事件相机下自中心手势识别的大规模数据集。实验结果表明，我们的方法在异构测试中达到了62.7%的准确率，比现有最佳方法高3.1%。自由式动作中的显著误分类主要是由于高个体间变异性以及不同训练数据的未知测试模式。此外，我们的方法在DVS128 Gesture数据集上达到了96.97%的准确率，显示出强大的跨数据集泛化能力。数据集和模型已公开发布。 

---
# LIAM: Multimodal Transformer for Language Instructions, Images, Actions and Semantic Maps 

**Title (ZH)**: LIAM: 多模态Transformer语言指令、图像、行动和语义地图 

**Authors**: Yihao Wang, Raphael Memmesheimer, Sven Behnke  

**Link**: [PDF](https://arxiv.org/pdf/2503.12230)  

**Abstract**: The availability of large language models and open-vocabulary object perception methods enables more flexibility for domestic service robots. The large variability of domestic tasks can be addressed without implementing each task individually by providing the robot with a task description along with appropriate environment information. In this work, we propose LIAM - an end-to-end model that predicts action transcripts based on language, image, action, and map inputs. Language and image inputs are encoded with a CLIP backbone, for which we designed two pre-training tasks to fine-tune its weights and pre-align the latent spaces. We evaluate our method on the ALFRED dataset, a simulator-generated benchmark for domestic tasks. Our results demonstrate the importance of pre-aligning embedding spaces from different modalities and the efficacy of incorporating semantic maps. 

**Abstract (ZH)**: 大规模语言模型和开放词汇物体感知方法的可用性为家庭服务机器人提供了更多的灵活性。通过向机器人提供任务描述和适当的环境信息，可以解决家庭任务的高变异性，无需单独实现每个任务。在这项工作中，我们提出了一种端到端模型LIAM，该模型基于语言、图像、动作和地图输入预测动作转录。语言和图像输入通过CLIP主干进行编码，我们为此设计了两个预训练任务以微调其权重并预对齐潜在空间。我们在ALFRED数据集上评估了我们的方法，该数据集是一个由模拟器生成的家用任务基准。我们的结果表明，不同模态嵌入空间的预对齐的重要性以及语义地图集成的有效性。 

---
# Formation Control of Multi-agent System with Local Interaction and Artificial Potential Field 

**Title (ZH)**: 多代理系统基于局部交互与人工势场的Formation控制 

**Authors**: Luoyin Zhao, Zheping Yan, Yuqing Wang, Raye Chen-Hua Yeow  

**Link**: [PDF](https://arxiv.org/pdf/2503.12199)  

**Abstract**: A novel local interaction control method (LICM) is proposed in this paper to realize the formation control of multi-agent system (MAS). A local interaction leader follower (LILF) structure is provided by coupling the advantages of information consensus and leader follower frame, the agents can obtain the state information of the leader by interacting with their neighbours, which will reduce the communication overhead of the system and the dependence on a single node of the topology. In addition, the artificial potential field (APF) method is introduced to achieve obstacle avoidance and collision avoidance between agents. Inspired by the stress response of animals, a stress response mechanism-artificial potential field (SRM-APF) is proposed, which will be triggered when the local minimum problem of APF occurs. Ultimately, the simulation experiments of three formation shapes, including triangular formation, square formation and hexagonal formation, validate the effectiveness of the proposed method. 

**Abstract (ZH)**: 一种新型局部交互控制方法（LICM）用于实现多 Robotics 系统（MAS）的编队控制。通过结合信息一致性与领导跟随框架的优点，提出了局部交互领导跟随（LILF）结构，代理可以通过与其邻居交互获得领导者的状态信息，从而减少系统的通信开销并降低对拓扑单节点的依赖。此外，引入人工势场（APF）方法以实现代理间的障碍物避障和碰撞避免。受动物应力反应的启发，提出了一种应力响应机制-人工势场（SRM-APF），当APF出现局部极小问题时，该机制将被触发。最终，对三角形编队、方形编队和六边形编队的仿真实验验证了所提出方法的有效性。 

---
# Value Gradients with Action Adaptive Search Trees in Continuous (PO)MDPs 

**Title (ZH)**: 基于动作自适应搜索树的连续(PO)MDPs中的价值梯度 

**Authors**: Idan Lev-Yehudi, Michael Novitsky, Moran Barenboim, Ron Benchetrit, Vadim Indelman  

**Link**: [PDF](https://arxiv.org/pdf/2503.12181)  

**Abstract**: Solving Partially Observable Markov Decision Processes (POMDPs) in continuous state, action and observation spaces is key for autonomous planning in many real-world mobility and robotics applications. Current approaches are mostly sample based, and cannot hope to reach near-optimal solutions in reasonable time. We propose two complementary theoretical contributions. First, we formulate a novel Multiple Importance Sampling (MIS) tree for value estimation, that allows to share value information between sibling action branches. The novel MIS tree supports action updates during search time, such as gradient-based updates. Second, we propose a novel methodology to compute value gradients with online sampling based on transition likelihoods. It is applicable to MDPs, and we extend it to POMDPs via particle beliefs with the application of the propagated belief trick. The gradient estimator is computed in practice using the MIS tree with efficient Monte Carlo sampling. These two parts are combined into a new planning algorithm Action Gradient Monte Carlo Tree Search (AGMCTS). We demonstrate in a simulated environment its applicability, advantages over continuous online POMDP solvers that rely solely on sampling, and we discuss further implications. 

**Abstract (ZH)**: 在连续状态、动作和观测空间中求解部分可观测马尔可夫决策过程（POMDPs）是许多实际移动性和机器人应用中自主规划的关键。当前的方法大多是基于采样的，难以在合理的时间内达到接近最优的解。我们提出了两个互补的理论贡献。首先，我们提出了一个新的多重重要性采样（MIS）树，用于价值估计，该树可以在搜索过程中共享价值信息，并支持基于梯度的动作更新。其次，我们提出了一种基于转换似然性的在线采样方法来计算价值梯度。该方法适用于MDPs，并通过粒子信念将其扩展到POMDPs，应用传播信念技巧。实际中，梯度估计器使用MIS树和高效的蒙特卡洛采样进行计算。这两部分被合并成一个新的规划算法：动作梯度蒙特卡洛树搜索（AGMCTS）。我们在一个模拟环境中展示了其应用性、相对于仅依赖采样的连续在线POMDP求解器的优势，并讨论了进一步的含义。 

---
# CHOrD: Generation of Collision-Free, House-Scale, and Organized Digital Twins for 3D Indoor Scenes with Controllable Floor Plans and Optimal Layouts 

**Title (ZH)**: CHOrD：生成无碰撞、家居规模且组织化的3D室内场景数字孪生，具有可控的平面图和最优布局 

**Authors**: Chong Su, Yingbin Fu, Zheyuan Hu, Jing Yang, Param Hanji, Shaojun Wang, Xuan Zhao, Cengiz Öztireli, Fangcheng Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11958)  

**Abstract**: We introduce CHOrD, a novel framework for scalable synthesis of 3D indoor scenes, designed to create house-scale, collision-free, and hierarchically structured indoor digital twins. In contrast to existing methods that directly synthesize the scene layout as a scene graph or object list, CHOrD incorporates a 2D image-based intermediate layout representation, enabling effective prevention of collision artifacts by successfully capturing them as out-of-distribution (OOD) scenarios during generation. Furthermore, unlike existing methods, CHOrD is capable of generating scene layouts that adhere to complex floor plans with multi-modal controls, enabling the creation of coherent, house-wide layouts robust to both geometric and semantic variations in room structures. Additionally, we propose a novel dataset with expanded coverage of household items and room configurations, as well as significantly improved data quality. CHOrD demonstrates state-of-the-art performance on both the 3D-FRONT and our proposed datasets, delivering photorealistic, spatially coherent indoor scene synthesis adaptable to arbitrary floor plan variations. 

**Abstract (ZH)**: CHOrD：一种用于可扩展合成室内3D场景的新颖框架，以创建房屋规模、无碰撞且层次结构化的室内数字孪生。 

---
# Learning Closed-Loop Parametric Nash Equilibria of Multi-Agent Collaborative Field Coverage 

**Title (ZH)**: 多代理协作场域覆盖的闭环参数纳什均衡学习 

**Authors**: Jushan Chen, Santiago Paternain  

**Link**: [PDF](https://arxiv.org/pdf/2503.11829)  

**Abstract**: Multi-agent reinforcement learning is a challenging and active field of research due to the inherent nonstationary property and coupling between agents. A popular approach to modeling the multi-agent interactions underlying the multi-agent RL problem is the Markov Game. There is a special type of Markov Game, termed Markov Potential Game, which allows us to reduce the Markov Game to a single-objective optimal control problem where the objective function is a potential function. In this work, we prove that a multi-agent collaborative field coverage problem, which is found in many engineering applications, can be formulated as a Markov Potential Game, and we can learn a parameterized closed-loop Nash Equilibrium by solving an equivalent single-objective optimal control problem. As a result, our algorithm is 10x faster during training compared to a game-theoretic baseline and converges faster during policy execution. 

**Abstract (ZH)**: 多智能体强化学习是一种由于固有的非平稳性和智能体间的耦合而具有挑战性和活跃的研究领域。一种用于建模多智能体强化学习问题下智能体间交互的方法是马尔科夫游戏。其中，一种特殊的马尔科夫游戏称为马尔科夫势能游戏，允许我们将马尔科夫游戏转换为单一目标的最优控制问题，其中目标函数为势能函数。在本文中，我们证明了一种存在于许多工程应用中的多智能体合作区域覆盖问题可以形式化为马尔科夫势能游戏，并通过求解等价的单一目标最优控制问题学习了一个参数化的闭环纳什均衡。因此，与博弈论基线相比，我们的算法在训练过程中快10倍，并且在策略执行过程中收敛更快。 

---
# Diffuse-CLoC: Guided Diffusion for Physics-based Character Look-ahead Control 

**Title (ZH)**: 基于物理的角色前瞻控制引导扩散：Diffuse-CLoC 

**Authors**: Xiaoyu Huang, Takara Truong, Yunbo Zhang, Fangzhou Yu, Jean Pierre Sleiman, Jessica Hodgins, Koushil Sreenath, Farbod Farshidian  

**Link**: [PDF](https://arxiv.org/pdf/2503.11801)  

**Abstract**: We present Diffuse-CLoC, a guided diffusion framework for physics-based look-ahead control that enables intuitive, steerable, and physically realistic motion generation. While existing kinematics motion generation with diffusion models offer intuitive steering capabilities with inference-time conditioning, they often fail to produce physically viable motions. In contrast, recent diffusion-based control policies have shown promise in generating physically realizable motion sequences, but the lack of kinematics prediction limits their steerability. Diffuse-CLoC addresses these challenges through a key insight: modeling the joint distribution of states and actions within a single diffusion model makes action generation steerable by conditioning it on the predicted states. This approach allows us to leverage established conditioning techniques from kinematic motion generation while producing physically realistic motions. As a result, we achieve planning capabilities without the need for a high-level planner. Our method handles a diverse set of unseen long-horizon downstream tasks through a single pre-trained model, including static and dynamic obstacle avoidance, motion in-betweening, and task-space control. Experimental results show that our method significantly outperforms the traditional hierarchical framework of high-level motion diffusion and low-level tracking. 

**Abstract (ZH)**: Diffuse-CLoC：一种用于物理导向前瞻控制的扩散框架，实现直观、可控且物理真实的运动生成 

---
# Industrial-Grade Sensor Simulation via Gaussian Splatting: A Modular Framework for Scalable Editing and Full-Stack Validation 

**Title (ZH)**: 工业级传感器模拟 via 高斯点绘：一种模块化框架，用于可扩展编辑和全流程验证 

**Authors**: Xianming Zeng, Sicong Du, Qifeng Chen, Lizhe Liu, Haoyu Shu, Jiaxuan Gao, Jiarun Liu, Jiulong Xu, Jianyun Xu, Mingxia Chen, Yiru Zhao, Peng Chen, Yapeng Xue, Chunming Zhao, Sheng Yang, Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11731)  

**Abstract**: Sensor simulation is pivotal for scalable validation of autonomous driving systems, yet existing Neural Radiance Fields (NeRF) based methods face applicability and efficiency challenges in industrial workflows. This paper introduces a Gaussian Splatting (GS) based system to address these challenges: We first break down sensor simulator components and analyze the possible advantages of GS over NeRF. Then in practice, we refactor three crucial components through GS, to leverage its explicit scene representation and real-time rendering: (1) choosing the 2D neural Gaussian representation for physics-compliant scene and sensor modeling, (2) proposing a scene editing pipeline to leverage Gaussian primitives library for data augmentation, and (3) coupling a controllable diffusion model for scene expansion and harmonization. We implement this framework on a proprietary autonomous driving dataset supporting cameras and LiDAR sensors. We demonstrate through ablation studies that our approach reduces frame-wise simulation latency, achieves better geometric and photometric consistency, and enables interpretable explicit scene editing and expansion. Furthermore, we showcase how integrating such a GS-based sensor simulator with traffic and dynamic simulators enables full-stack testing of end-to-end autonomy algorithms. Our work provides both algorithmic insights and practical validation, establishing GS as a cornerstone for industrial-grade sensor simulation. 

**Abstract (ZH)**: 基于高斯点云的传感器模拟在自主驾驶系统工业级验证中的应用与挑战克服：构建高效可解析场景编辑与扩展的传感器模拟系统 

---
# Low-pass sampling in Model Predictive Path Integral Control 

**Title (ZH)**: 低通采样在模型预测路径积分控制中的应用 

**Authors**: Piotr Kicki  

**Link**: [PDF](https://arxiv.org/pdf/2503.11717)  

**Abstract**: Model Predictive Path Integral (MPPI) control is a widely used sampling-based approach for real-time control, offering flexibility in handling arbitrary dynamics and cost functions. However, the original MPPI suffers from high-frequency noise in the sampled control trajectories, leading to actuator wear and inefficient exploration. In this work, we introduce Low-Pass Model Predictive Path Integral Control (LP-MPPI), which integrates low-pass filtering into the sampling process to eliminate detrimental high-frequency components and improve the effectiveness of the control trajectories exploration. Unlike prior approaches, LP-MPPI provides direct and interpretable control over the frequency spectrum of sampled trajectories, enhancing sampling efficiency and control smoothness. Through extensive evaluations in Gymnasium environments, simulated quadruped locomotion, and real-world F1TENTH autonomous racing, we demonstrate that LP-MPPI consistently outperforms state-of-the-art MPPI variants, achieving significant performance improvements while reducing control signal chattering. 

**Abstract (ZH)**: 低通滤波模型预测路径积分控制（LP-MPPI） 

---
# Exploring Causality for HRI: A Case Study on Robotic Mental Well-being Coaching 

**Title (ZH)**: 探索人机交互中的因果关系：一项关于机器人心理健康辅导的案例研究 

**Authors**: Micol Spitale, Srikar Babu, Serhan Cakmak, Jiaee Cheong, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2503.11684)  

**Abstract**: One of the primary goals of Human-Robot Interaction (HRI) research is to develop robots that can interpret human behavior and adapt their responses accordingly. Adaptive learning models, such as continual and reinforcement learning, play a crucial role in improving robots' ability to interact effectively in real-world settings. However, these models face significant challenges due to the limited availability of real-world data, particularly in sensitive domains like healthcare and well-being. This data scarcity can hinder a robot's ability to adapt to new situations. To address these challenges, causality provides a structured framework for understanding and modeling the underlying relationships between actions, events, and outcomes. By moving beyond mere pattern recognition, causality enables robots to make more explainable and generalizable decisions. This paper presents an exploratory causality-based analysis through a case study of an adaptive robotic coach delivering positive psychology exercises over four weeks in a workplace setting. The robotic coach autonomously adapts to multimodal human behaviors, such as facial valence and speech duration. By conducting both macro- and micro-level causal analyses, this study aims to gain deeper insights into how adaptability can enhance well-being during interactions. Ultimately, this research seeks to advance our understanding of how causality can help overcome challenges in HRI, particularly in real-world applications. 

**Abstract (ZH)**: 基于因果关系的交互分析：适应性机器人教练在工作场所积极心理学练习中的应用 

---
