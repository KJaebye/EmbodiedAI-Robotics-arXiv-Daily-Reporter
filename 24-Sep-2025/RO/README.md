# Residual Off-Policy RL for Finetuning Behavior Cloning Policies 

**Title (ZH)**: 残差离策RLbehavior克隆策略微调 

**Authors**: Lars Ankile, Zhenyu Jiang, Rocky Duan, Guanya Shi, Pieter Abbeel, Anusha Nagabandi  

**Link**: [PDF](https://arxiv.org/pdf/2509.19301)  

**Abstract**: Recent advances in behavior cloning (BC) have enabled impressive visuomotor control policies. However, these approaches are limited by the quality of human demonstrations, the manual effort required for data collection, and the diminishing returns from increasing offline data. In comparison, reinforcement learning (RL) trains an agent through autonomous interaction with the environment and has shown remarkable success in various domains. Still, training RL policies directly on real-world robots remains challenging due to sample inefficiency, safety concerns, and the difficulty of learning from sparse rewards for long-horizon tasks, especially for high-degree-of-freedom (DoF) systems. We present a recipe that combines the benefits of BC and RL through a residual learning framework. Our approach leverages BC policies as black-box bases and learns lightweight per-step residual corrections via sample-efficient off-policy RL. We demonstrate that our method requires only sparse binary reward signals and can effectively improve manipulation policies on high-degree-of-freedom (DoF) systems in both simulation and the real world. In particular, we demonstrate, to the best of our knowledge, the first successful real-world RL training on a humanoid robot with dexterous hands. Our results demonstrate state-of-the-art performance in various vision-based tasks, pointing towards a practical pathway for deploying RL in the real world. Project website: this https URL 

**Abstract (ZH)**: 最近行为克隆的进展使得可视化运动控制策略取得了显著成效。然而，这些方法受限于人类示范的质量、数据收集所需的 manual 努力，以及额外数据带来的边际收益递减。相比之下，强化学习通过自主与环境交互来训练代理，并且已经在多个领域展示了显著的成功。尽管如此，直接在真实世界机器人上训练强化学习策略仍然具有挑战性，原因包括样本效率低、安全问题以及从稀疏奖励中学习长时间任务的困难，尤其是在高自由度系统中。我们提出了一种结合行为克隆和强化学习优点的方法，采用残差学习框架。我们的方法利用行为克隆策略作为黑盒基础，并通过样本高效的行为聚类强化学习学习轻量级的逐步残差修正。我们证明，我们的方法仅需要稀疏的二元奖励信号，并且可以有效地提高高自由度系统的操作策略，无论是仿真还是真实世界中。特别地，我们首次展示了一种灵巧手的人形机器人在真实世界中使用强化学习训练成功的实例。我们的结果表明，在各种基于视觉的任务中达到了最先进的性能，展示了强化学习在真实世界部署的实际途径。项目网站: 这里是链接。 

---
# SOE: Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration 

**Title (ZH)**: SOE: 样本高效机器人策略自我改进通过在流形上探索 

**Authors**: Yang Jin, Jun Lv, Han Xue, Wendi Chen, Chuan Wen, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19292)  

**Abstract**: Intelligent agents progress by continually refining their capabilities through actively exploring environments. Yet robot policies often lack sufficient exploration capability due to action mode collapse. Existing methods that encourage exploration typically rely on random perturbations, which are unsafe and induce unstable, erratic behaviors, thereby limiting their effectiveness. We propose Self-Improvement via On-Manifold Exploration (SOE), a framework that enhances policy exploration and improvement in robotic manipulation. SOE learns a compact latent representation of task-relevant factors and constrains exploration to the manifold of valid actions, ensuring safety, diversity, and effectiveness. It can be seamlessly integrated with arbitrary policy models as a plug-in module, augmenting exploration without degrading the base policy performance. Moreover, the structured latent space enables human-guided exploration, further improving efficiency and controllability. Extensive experiments in both simulation and real-world tasks demonstrate that SOE consistently outperforms prior methods, achieving higher task success rates, smoother and safer exploration, and superior sample efficiency. These results establish on-manifold exploration as a principled approach to sample-efficient policy self-improvement. Project website: this https URL 

**Abstract (ZH)**: 智能代理通过不断 refinement 其能力并积极探索环境来进步。然而，由于行为模式崩溃，机器人策略往往缺乏足够的探索能力。现有的鼓励探索的方法通常依赖于随机扰动，这很不安全且会导致不稳定、不规则的行为，从而限制了其有效性。我们提出了 On-Manifold Exploration (SOE) 自我完善框架，该框架增强了机器人操作中的策略探索与改进。SOE 学习任务相关因素的紧凑潜在表示，并将探索限制在有效行动的流形上，确保了安全、多样性和有效性。它可以无缝集成到任意策略模型中作为插件模块，增强探索而不损害基策略性能。此外，结构化的潜在空间使人工指导的探索成为可能，进一步提高了效率和可控性。在仿真和真实任务中的广泛实验表明，SOE 一致地优于先前的方法，实现了更高的任务成功率、更平滑和更安全的探索以及更好的样本效率。这些结果确立了在流形上探索作为一种灵活性强的样本高效策略自我改进的方法。项目网站：这个 https URL。 

---
# Imitation-Guided Bimanual Planning for Stable Manipulation under Changing External Forces 

**Title (ZH)**: 模仿引导的双臂规划以应对外部力变化的稳定操作 

**Authors**: Kuanqi Cai, Chunfeng Wang, Zeqi Li, Haowen Yao, Weinan Chen, Luis Figueredo, Aude Billard, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2509.19261)  

**Abstract**: Robotic manipulation in dynamic environments often requires seamless transitions between different grasp types to maintain stability and efficiency. However, achieving smooth and adaptive grasp transitions remains a challenge, particularly when dealing with external forces and complex motion constraints. Existing grasp transition strategies often fail to account for varying external forces and do not optimize motion performance effectively. In this work, we propose an Imitation-Guided Bimanual Planning Framework that integrates efficient grasp transition strategies and motion performance optimization to enhance stability and dexterity in robotic manipulation. Our approach introduces Strategies for Sampling Stable Intersections in Grasp Manifolds for seamless transitions between uni-manual and bi-manual grasps, reducing computational costs and regrasping inefficiencies. Additionally, a Hierarchical Dual-Stage Motion Architecture combines an Imitation Learning-based Global Path Generator with a Quadratic Programming-driven Local Planner to ensure real-time motion feasibility, obstacle avoidance, and superior manipulability. The proposed method is evaluated through a series of force-intensive tasks, demonstrating significant improvements in grasp transition efficiency and motion performance. A video demonstrating our simulation results can be viewed at \href{this https URL}{\textcolor{blue}{this https URL}}. 

**Abstract (ZH)**: 动态环境下的机器人操作往往需要在不同抓取类型之间实现无缝过渡，以维持稳定性和效率。然而，实现平滑且适应性的抓取过渡仍是一项挑战，特别是在处理外部力和复杂运动约束时。现有的抓取过渡策略往往未能考虑到变化的外部力，也没有有效优化运动性能。在本工作中，我们提出了一种模仿引导的双臂规划框架，将有效的抓取过渡策略和运动性能优化相结合，以增强机器人操作中的稳定性和灵巧性。我们的方法引入了在抓取流形中采样稳定交点的策略，以实现单手抓取和双手抓取之间的无缝过渡，从而减少计算成本和重新抓取的无效性。此外，层次化的双阶段运动架构结合了基于模仿学习的全局路径生成器和二次规划驱动的局部规划器，以确保实时运动可行性、避开障碍物以及卓越的操作性能。提出的方​​法通过一系列力密集型任务进行评估，显示出在抓取过渡效率和运动性能方面的显著改进。我们的仿真结果演示视频可以在 \href{this https URL}{这个网址} 查看。 

---
# Proactive-reactive detection and mitigation of intermittent faults in robot swarms 

**Title (ZH)**: proactive-反应式检测与缓解机器人 swarm 中的间歇性故障 

**Authors**: Sinan Oğuz, Emanuele Garone, Marco Dorigo, Mary Katherine Heinrich  

**Link**: [PDF](https://arxiv.org/pdf/2509.19246)  

**Abstract**: Intermittent faults are transient errors that sporadically appear and disappear. Although intermittent faults pose substantial challenges to reliability and coordination, existing studies of fault tolerance in robot swarms focus instead on permanent faults. One reason for this is that intermittent faults are prohibitively difficult to detect in the fully self-organized ad-hoc networks typical of robot swarms, as their network topologies are transient and often unpredictable. However, in the recently introduced self-organizing nervous systems (SoNS) approach, robot swarms are able to self-organize persistent network structures for the first time, easing the problem of detecting intermittent faults. To address intermittent faults in robot swarms that have persistent networks, we propose a novel proactive-reactive strategy to detection and mitigation, based on self-organized backup layers and distributed consensus in a multiplex network. Proactively, the robots self-organize dynamic backup paths before faults occur, adapting to changes in the primary network topology and the robots' relative positions. Reactively, robots use one-shot likelihood ratio tests to compare information received along different paths in the multiplex network, enabling early fault detection. Upon detection, communication is temporarily rerouted in a self-organized way, until the detected fault resolves. We validate the approach in representative scenarios of faulty positional data occurring during formation control, demonstrating that intermittent faults are prevented from disrupting convergence to desired formations, with high fault detection accuracy and low rates of false positives. 

**Abstract (ZH)**: 间歇性故障是偶尔出现并消失的瞬态错误，尽管间歇性故障给可靠性和协调带来了重大挑战，现有机器人 swarm 故障容错研究主要集中在永久性故障上。其中一个原因是间歇性故障在典型由机器人 swarm 构成的完全自组织即兴网络中难以检测，因为这些网络拓扑是瞬态且往往不可预测的。然而，在最近引入的自组织神经系统（SoNS）方法中，机器人 swarm 首次能够自我组织持久的网络结构，从而减轻了检测间歇性故障的问题。为了应对具有持久网络结构的机器人 swarm 中的间歇性故障，我们提出了一种新颖的主动-被动检测与缓解策略，基于多层网络中的自组织备份层和分布式一致意见。主动地，机器人在故障发生前自组织动态备份路径，适应主网络拓扑和机器人相对位置的变化。被动地，机器人使用一次似然比检验来比较多层网络中不同路径接收到的信息，实现早期故障检测。检测到故障后，通信暂时以自组织方式重路由，直到故障被解决。我们在故障位置数据导致队形控制失效的代表性场景中验证了该方法，证明了间歇性故障不会破坏对期望队形的收敛，具有高故障检测准确性和低误报率。 

---
# MagiClaw: A Dual-Use, Vision-Based Soft Gripper for Bridging the Human Demonstration to Robotic Deployment Gap 

**Title (ZH)**: MagiClaw: 一种双用途、基于视觉的软 gripper，用于弥合人类示范与机器人部署的差距 

**Authors**: Tianyu Wu, Xudong Han, Haoran Sun, Zishang Zhang, Bangchao Huang, Chaoyang Song, Fang Wan  

**Link**: [PDF](https://arxiv.org/pdf/2509.19169)  

**Abstract**: The transfer of manipulation skills from human demonstration to robotic execution is often hindered by a "domain gap" in sensing and morphology. This paper introduces MagiClaw, a versatile two-finger end-effector designed to bridge this gap. MagiClaw functions interchangeably as both a handheld tool for intuitive data collection and a robotic end-effector for policy deployment, ensuring hardware consistency and reliability. Each finger incorporates a Soft Polyhedral Network (SPN) with an embedded camera, enabling vision-based estimation of 6-DoF forces and contact deformation. This proprioceptive data is fused with exteroceptive environmental sensing from an integrated iPhone, which provides 6D pose, RGB video, and LiDAR-based depth maps. Through a custom iOS application, MagiClaw streams synchronized, multi-modal data for real-time teleoperation, offline policy learning, and immersive control via mixed-reality interfaces. We demonstrate how this unified system architecture lowers the barrier to collecting high-fidelity, contact-rich datasets and accelerates the development of generalizable manipulation policies. Please refer to the iOS app at this https URL for further details. 

**Abstract (ZH)**: 从人类演示到机器人执行的操纵技能转移常常受到感应和形态“领域差距”的阻碍。本文介绍了一种多功能两指末端执行器MagiClaw，旨在弥合这一差距。MagiClaw可以作为手持工具进行直观的数据收集，也可以作为机器人末端执行器部署策略，确保硬件的一致性和可靠性。每个手指都配备了一个内置摄像头的Soft Polyhedral Network (SPN)，能够进行基于视觉的6-自由度力和接触变形的估计。这种本体感觉数据与集成的iPhone提供的6D姿态、RGB视频和基于LiDAR的深度图的外部感知数据融合。通过自定义iOS应用程序，MagiClaw可以同步流式传输多模态数据进行实时遥控、离线策略学习和通过混合现实界面进行沉浸式控制。本文展示了这种统一系统架构如何降低收集高保真、富有接触数据集的门槛，并加速可泛化的操纵策略的发展。有关iOS应用程序的更多信息，请参见这个链接。 

---
# A Multimodal Stochastic Planning Approach for Navigation and Multi-Robot Coordination 

**Title (ZH)**: 多模态随机规划方法在导航与多机器人协调中的应用 

**Authors**: Mark Gonzales, Ethan Oh, Joseph Moore  

**Link**: [PDF](https://arxiv.org/pdf/2509.19168)  

**Abstract**: In this paper, we present a receding-horizon, sampling-based planner capable of reasoning over multimodal policy distributions. By using the cross-entropy method to optimize a multimodal policy under a common cost function, our approach increases robustness against local minima and promotes effective exploration of the solution space. We show that our approach naturally extends to multi-robot collision-free planning, enables agents to share diverse candidate policies to avoid deadlocks, and allows teams to minimize a global objective without incurring the computational complexity of centralized optimization. Numerical simulations demonstrate that employing multiple modes significantly improves success rates in trap environments and in multi-robot collision avoidance. Hardware experiments further validate the approach's real-time feasibility and practical performance. 

**Abstract (ZH)**: 在本文中，我们提出了一种退火 horizon、基于采样的规划器，能够处理多模态策略分布。通过使用交叉熵方法在公共成本函数下优化多模态策略，我们的方法增加了对局部最小值的鲁棒性，并促进了解空间的有效探索。我们展示了我们的方法自然地扩展到多机器人无碰撞规划，使代理能够通过共享多样化的候选策略来避免死锁，并允许团队在不增加集中优化的计算复杂度的情况下最小化全局目标。数值仿真表明，采用多种模式显著提高了在陷阱环境和多机器人碰撞避免中的成功率。硬件实验进一步验证了该方法在实时性和实际性能上的可行性。 

---
# BiGraspFormer: End-to-End Bimanual Grasp Transformer 

**Title (ZH)**: 双臂抓取变换器：端到端双向抓取变换器 

**Authors**: Kangmin Kim, Seunghyeok Back, Geonhyup Lee, Sangbeom Lee, Sangjun Noh, Kyoobin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19142)  

**Abstract**: Bimanual grasping is essential for robots to handle large and complex objects. However, existing methods either focus solely on single-arm grasping or employ separate grasp generation and bimanual evaluation stages, leading to coordination problems including collision risks and unbalanced force distribution. To address these limitations, we propose BiGraspFormer, a unified end-to-end transformer framework that directly generates coordinated bimanual grasps from object point clouds. Our key idea is the Single-Guided Bimanual (SGB) strategy, which first generates diverse single grasp candidates using a transformer decoder, then leverages their learned features through specialized attention mechanisms to jointly predict bimanual poses and quality scores. This conditioning strategy reduces the complexity of the 12-DoF search space while ensuring coordinated bimanual manipulation. Comprehensive simulation experiments and real-world validation demonstrate that BiGraspFormer consistently outperforms existing methods while maintaining efficient inference speed (<0.05s), confirming the effectiveness of our framework. Code and supplementary materials are available at this https URL 

**Abstract (ZH)**: 双臂抓取对于处理大型和复杂物体的机器人而言是必不可少的。然而，现有方法要么仅关注单臂抓取，要么采用分阶段的抓取生成和双臂评估，导致协调问题，包括碰撞风险和力分布不均。为了解决这些局限性，我们提出了一种统一的端到端变压器框架BiGraspFormer，可以从对象点云直接生成协调的双臂抓取。我们的关键思想是Single-Guided Bimanual (SGB) 策略，该策略首先使用变压器解码器生成多种单臂抓取候选，然后通过专门的注意力机制利用这些候选的学习特征联合预测双臂姿态和质量分数。这种调节策略降低了12自由度搜索空间的复杂性，同时确保了协调的双臂操作。综合的仿真实验和现实世界的验证表明，BiGraspFormer在保持高效推理速度(<0.05s)的同时始终优于现有方法，证实了我们框架的有效性。相关代码和补充材料可在以下网址获取。 

---
# Spectral Signature Mapping from RGB Imagery for Terrain-Aware Navigation 

**Title (ZH)**: 基于RGB图像的光谱特征图映射及其地形感知导航 

**Authors**: Sarvesh Prajapati, Ananya Trivedi, Nathaniel Hanson, Bruce Maxwell, Taskin Padir  

**Link**: [PDF](https://arxiv.org/pdf/2509.19105)  

**Abstract**: Successful navigation in outdoor environments requires accurate prediction of the physical interactions between the robot and the terrain. To this end, several methods rely on geometric or semantic labels to classify traversable surfaces. However, such labels cannot distinguish visually similar surfaces that differ in material properties. Spectral sensors enable inference of material composition from surface reflectance measured across multiple wavelength bands. Although spectral sensing is gaining traction in robotics, widespread deployment remains constrained by the need for custom hardware integration, high sensor costs, and compute-intensive processing pipelines. In this paper, we present RGB Image to Spectral Signature Neural Network (RS-Net), a deep neural network designed to bridge the gap between the accessibility of RGB sensing and the rich material information provided by spectral data. RS-Net predicts spectral signatures from RGB patches, which we map to terrain labels and friction coefficients. The resulting terrain classifications are integrated into a sampling-based motion planner for a wheeled robot operating in outdoor environments. Likewise, the friction estimates are incorporated into a contact-force-based MPC for a quadruped robot navigating slippery surfaces. Thus, we introduce a framework that learns the task-relevant physical property once during training and thereafter relies solely on RGB sensing at test time. The code is available at this https URL. 

**Abstract (ZH)**: 基于RGB图像到光谱签名的神经网络（RS-Net）：桥梁感知与光谱数据丰富材料信息的 gap 

---
# FUNCanon: Learning Pose-Aware Action Primitives via Functional Object Canonicalization for Generalizable Robotic Manipulation 

**Title (ZH)**: FUNCanon: 基于功能性物体标准化的学习姿态感知动作 primitives 以实现通用化机器人操作 

**Authors**: Hongli Xu, Lei Zhang, Xiaoyue Hu, Boyang Zhong, Kaixin Bai, Zoltán-Csaba Márton, Zhenshan Bing, Zhaopeng Chen, Alois Christian Knoll, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19102)  

**Abstract**: General-purpose robotic skills from end-to-end demonstrations often leads to task-specific policies that fail to generalize beyond the training distribution. Therefore, we introduce FunCanon, a framework that converts long-horizon manipulation tasks into sequences of action chunks, each defined by an actor, verb, and object. These chunks focus policy learning on the actions themselves, rather than isolated tasks, enabling compositionality and reuse. To make policies pose-aware and category-general, we perform functional object canonicalization for functional alignment and automatic manipulation trajectory transfer, mapping objects into shared functional frames using affordance cues from large vision language models. An object centric and action centric diffusion policy FuncDiffuser trained on this aligned data naturally respects object affordances and poses, simplifying learning and improving generalization ability. Experiments on simulated and real-world benchmarks demonstrate category-level generalization, cross-task behavior reuse, and robust sim2real deployment, showing that functional canonicalization provides a strong inductive bias for scalable imitation learning in complex manipulation domains. Details of the demo and supplemental material are available on our project website this https URL. 

**Abstract (ZH)**: 通用机器人技能从端到端演示中获得往往会导致仅在训练分布内生效的任务特定策略。因此，我们引入了FunCanon框架，该框架将长时间 horizon 操作任务转化为由执行者、动词和物体定义的动作片段序列。这些片段将策略学习的重点放在动作本身上，而非孤立的任务，从而实现组合性和重用性。为了使策略具备姿态感知能力和类别普适性，我们执行功能对象规范化以实现功能对齐和自动操作轨迹转移，通过大型视觉语言模型的可用性线索将对象映射到共享的功能框架中。基于对齐数据训练的以物体为中心和以动作为中心的扩散策略FuncDiffuser自然地遵守对象可用性及其姿态，简化了学习过程并增强了泛化能力。在模拟和实际世界基准测试上的实验表明，自然展示了类别级泛化能力、跨任务行为重用以及鲁棒的仿真实际部署能力，说明功能规范化为在复杂操作领域可扩展的模仿学习提供了强大的归纳偏置。更多演示详情和补充材料请参见我们的项目网站：this https URL。 

---
# World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation 

**Title (ZH)**: World4RL: 扩展扩散模型在强化学习中用于机器人操作策略精炼 

**Authors**: Zhennan Jiang, Kai Liu, Yuxin Qin, Shuai Tian, Yupeng Zheng, Mingcai Zhou, Chao Yu, Haoran Li, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19080)  

**Abstract**: Robotic manipulation policies are commonly initialized through imitation learning, but their performance is limited by the scarcity and narrow coverage of expert data. Reinforcement learning can refine polices to alleviate this limitation, yet real-robot training is costly and unsafe, while training in simulators suffers from the sim-to-real gap. Recent advances in generative models have demonstrated remarkable capabilities in real-world simulation, with diffusion models in particular excelling at generation. This raises the question of how diffusion model-based world models can be combined to enhance pre-trained policies in robotic manipulation. In this work, we propose World4RL, a framework that employs diffusion-based world models as high-fidelity simulators to refine pre-trained policies entirely in imagined environments for robotic manipulation. Unlike prior works that primarily employ world models for planning, our framework enables direct end-to-end policy optimization. World4RL is designed around two principles: pre-training a diffusion world model that captures diverse dynamics on multi-task datasets and refining policies entirely within a frozen world model to avoid online real-world interactions. We further design a two-hot action encoding scheme tailored for robotic manipulation and adopt diffusion backbones to improve modeling fidelity. Extensive simulation and real-world experiments demonstrate that World4RL provides high-fidelity environment modeling and enables consistent policy refinement, yielding significantly higher success rates compared to imitation learning and other baselines. More visualization results are available at this https URL. 

**Abstract (ZH)**: 基于扩散模型的世界模型在机器人 manipulation 中增强预训练策略的研究 

---
# SlicerROS2: A Research and Development Module for Image-Guided Robotic Interventions 

**Title (ZH)**: SlicerROS2: 一种图像引导机器人干预研究与开发模块 

**Authors**: Laura Connolly, Aravind S. Kumar, Kapi Ketan Mehta, Lidia Al-Zogbi, Peter Kazanzides, Parvin Mousavi, Gabor Fichtinger, Axel Krieger, Junichi Tokuda, Russell H. Taylor, Simon Leonard, Anton Deguet  

**Link**: [PDF](https://arxiv.org/pdf/2509.19076)  

**Abstract**: Image-guided robotic interventions involve the use of medical imaging in tandem with robotics. SlicerROS2 is a software module that combines 3D Slicer and robot operating system (ROS) in pursuit of a standard integration approach for medical robotics research. The first release of SlicerROS2 demonstrated the feasibility of using the C++ API from 3D Slicer and ROS to load and visualize robots in real time. Since this initial release, we've rewritten and redesigned the module to offer greater modularity, access to low-level features, access to 3D Slicer's Python API, and better data transfer protocols. In this paper, we introduce this new design as well as four applications that leverage the core functionalities of SlicerROS2 in realistic image-guided robotics scenarios. 

**Abstract (ZH)**: 基于图像的机器人干预涉及将医学成像与机器人技术结合使用。SlicerROS2 是一个软件模块，结合了3D Slicer和机器人操作系统（ROS），旨在提供医学机器人研究的标准集成方法。SlicerROS2 的首次发布展示了使用3D Slicer和ROS的C++ API 实时加载和可视化机器人的可行性。在此初步发布的基础上，我们重新设计了该模块，使其更具模块性，提供低级功能访问，3D Slicer的Python API 访问，并改进了数据传输协议。本文介绍了这一新设计以及四种利用SlicerROS2核心功能的应用程序，适用于实际的基于图像的机器人场景。 

---
# ManipForce: Force-Guided Policy Learning with Frequency-Aware Representation for Contact-Rich Manipulation 

**Title (ZH)**: ManipForce：基于频率感知表示的力引导政策学习在接触丰富的操纵中 

**Authors**: Geonhyup Lee, Yeongjin Lee, Kangmin Kim, Seongju Lee, Sangjun Noh, Seunghyeok Back, Kyoobin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19047)  

**Abstract**: Contact-rich manipulation tasks such as precision assembly require precise control of interaction forces, yet existing imitation learning methods rely mainly on vision-only demonstrations. We propose ManipForce, a handheld system designed to capture high-frequency force-torque (F/T) and RGB data during natural human demonstrations for contact-rich manipulation. Building on these demonstrations, we introduce the Frequency-Aware Multimodal Transformer (FMT). FMT encodes asynchronous RGB and F/T signals using frequency- and modality-aware embeddings and fuses them via bi-directional cross-attention within a transformer diffusion policy. Through extensive experiments on six real-world contact-rich manipulation tasks - such as gear assembly, box flipping, and battery insertion - FMT trained on ManipForce demonstrations achieves robust performance with an average success rate of 83% across all tasks, substantially outperforming RGB-only baselines. Ablation and sampling-frequency analyses further confirm that incorporating high-frequency F/T data and cross-modal integration improves policy performance, especially in tasks demanding high precision and stable contact. 

**Abstract (ZH)**: 接触丰富的操作任务，如精确装配，要求精确控制相互作用力，现有模仿学习方法主要依赖于单目视觉演示。我们提出ManipForce，一种手持系统，用于在自然的人类演示过程中捕捉高频力- torque (F/T) 和 RGB 数据，以适用于接触丰富的操作。基于这些演示，我们引入了频率感知多模态Transformer (FMT)。FMT 使用频率和模态感知嵌入来编码异步 RGB 和 F/T 信号，并通过变压器扩散策略内的双向跨注意力将它们融合。通过在六项实际的接触丰富的操作任务（如齿轮装配、箱子翻转和电池插入）上的广泛实验，使用ManipForce演示训练的FMT实现了平均83%的成功率，在所有任务中都表现出色，显著优于仅基于RGB的基线。进一步的消融分析和采样频率分析证实，整合高频F/T数据和跨模态集成可以提高策略性能，特别是在要求高精度和稳定接触的任务中表现更为突出。 

---
# TacEva: A Performance Evaluation Framework For Vision-Based Tactile Sensors 

**Title (ZH)**: TacEva：基于视觉的触觉传感器性能评估框架 

**Authors**: Qingzheng Cong, Steven Oh, Wen Fan, Shan Luo, Kaspar Althoefer, Dandan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19037)  

**Abstract**: Vision-Based Tactile Sensors (VBTSs) are widely used in robotic tasks because of the high spatial resolution they offer and their relatively low manufacturing costs. However, variations in their sensing mechanisms, structural dimension, and other parameters lead to significant performance disparities between existing VBTSs. This makes it challenging to optimize them for specific tasks, as both the initial choice and subsequent fine-tuning are hindered by the lack of standardized metrics. To address this issue, TacEva is introduced as a comprehensive evaluation framework for the quantitative analysis of VBTS performance. The framework defines a set of performance metrics that capture key characteristics in typical application scenarios. For each metric, a structured experimental pipeline is designed to ensure consistent and repeatable quantification. The framework is applied to multiple VBTSs with distinct sensing mechanisms, and the results demonstrate its ability to provide a thorough evaluation of each design and quantitative indicators for each performance dimension. This enables researchers to pre-select the most appropriate VBTS on a task by task basis, while also offering performance-guided insights into the optimization of VBTS design. A list of existing VBTS evaluation methods and additional evaluations can be found on our website: this https URL 

**Abstract (ZH)**: 基于视觉的触觉传感器（VBTSs）的综合评估框架：针对典型应用场景的定量分析 

---
# Reduced-Order Model-Guided Reinforcement Learning for Demonstration-Free Humanoid Locomotion 

**Title (ZH)**: 基于降阶模型引导的强化学习人体运动控制 

**Authors**: Shuai Liu, Meng Cheng Lau  

**Link**: [PDF](https://arxiv.org/pdf/2509.19023)  

**Abstract**: We introduce Reduced-Order Model-Guided Reinforcement Learning (ROM-GRL), a two-stage reinforcement learning framework for humanoid walking that requires no motion capture data or elaborate reward shaping. In the first stage, a compact 4-DOF (four-degree-of-freedom) reduced-order model (ROM) is trained via Proximal Policy Optimization. This generates energy-efficient gait templates. In the second stage, those dynamically consistent trajectories guide a full-body policy trained with Soft Actor--Critic augmented by an adversarial discriminator, ensuring the student's five-dimensional gait feature distribution matches the ROM's demonstrations. Experiments at 1 meter-per-second and 4 meter-per-second show that ROM-GRL produces stable, symmetric gaits with substantially lower tracking error than a pure-reward baseline. By distilling lightweight ROM guidance into high-dimensional policies, ROM-GRL bridges the gap between reward-only and imitation-based locomotion methods, enabling versatile, naturalistic humanoid behaviors without any human demonstrations. 

**Abstract (ZH)**: Reduced-Order Model-Guided Reinforcement Learning for Humanoid Walking 

---
# Pure Vision Language Action (VLA) Models: A Comprehensive Survey 

**Title (ZH)**: 纯视觉语言动作（VLA）模型：全面综述 

**Authors**: Dapeng Zhang, Jin Sun, Chenghui Hu, Xiaoyan Wu, Zhenlong Yuan, Rui Zhou, Fei Shen, Qingguo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.19012)  

**Abstract**: The emergence of Vision Language Action (VLA) models marks a paradigm shift from traditional policy-based control to generalized robotics, reframing Vision Language Models (VLMs) from passive sequence generators into active agents for manipulation and decision-making in complex, dynamic environments. This survey delves into advanced VLA methods, aiming to provide a clear taxonomy and a systematic, comprehensive review of existing research. It presents a comprehensive analysis of VLA applications across different scenarios and classifies VLA approaches into several paradigms: autoregression-based, diffusion-based, reinforcement-based, hybrid, and specialized methods; while examining their motivations, core strategies, and implementations in detail. In addition, foundational datasets, benchmarks, and simulation platforms are introduced. Building on the current VLA landscape, the review further proposes perspectives on key challenges and future directions to advance research in VLA models and generalizable robotics. By synthesizing insights from over three hundred recent studies, this survey maps the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose VLA methods. 

**Abstract (ZH)**: 视觉语言动作（VLA）模型的出现标志着从传统基于政策的控制向通用机器人学的范式转变，重新定义了视觉语言模型（VLMs）从被动序列生成器为复杂动态环境中的操作和决策制定主动代理。本文综述了先进的VLA方法，旨在提供一个清晰的分类体系和对现有研究的系统全面综述。文章对不同场景下的VLA应用进行了全面分析，并将VLA方法分类为自回归、扩散、强化学习、混合和专门方法等几种范式，详细探讨了它们的动机、核心策略和实现方式。此外，文中还介绍了基础数据集、基准测试和仿真平台。在当前VLA景观的基础上，综述进一步提出了关键挑战和未来方向，以推动VLA模型和通用机器人学的研究。通过对三百多篇近期研究的综合洞察，本文勾勒了这一快速发展的领域轮廓，并指出了将塑造可扩展通用VLA方法发展的机会与挑战。 

---
# Category-Level Object Shape and Pose Estimation in Less Than a Millisecond 

**Title (ZH)**: 毫秒级类别级物体形状和姿态估计 

**Authors**: Lorenzo Shaikewitz, Tim Nguyen, Luca Carlone  

**Link**: [PDF](https://arxiv.org/pdf/2509.18979)  

**Abstract**: Object shape and pose estimation is a foundational robotics problem, supporting tasks from manipulation to scene understanding and navigation. We present a fast local solver for shape and pose estimation which requires only category-level object priors and admits an efficient certificate of global optimality. Given an RGB-D image of an object, we use a learned front-end to detect sparse, category-level semantic keypoints on the target object. We represent the target object's unknown shape using a linear active shape model and pose a maximum a posteriori optimization problem to solve for position, orientation, and shape simultaneously. Expressed in unit quaternions, this problem admits first-order optimality conditions in the form of an eigenvalue problem with eigenvector nonlinearities. Our primary contribution is to solve this problem efficiently with self-consistent field iteration, which only requires computing a 4-by-4 matrix and finding its minimum eigenvalue-vector pair at each iterate. Solving a linear system for the corresponding Lagrange multipliers gives a simple global optimality certificate. One iteration of our solver runs in about 100 microseconds, enabling fast outlier rejection. We test our method on synthetic data and a variety of real-world settings, including two public datasets and a drone tracking scenario. Code is released at this https URL. 

**Abstract (ZH)**: 基于类别先验的快速局部求解器：形状与姿态估计 

---
# Towards Robust LiDAR Localization: Deep Learning-based Uncertainty Estimation 

**Title (ZH)**: 基于深度学习的不确定性估计的鲁棒LiDAR定位 

**Authors**: Minoo Dolatabadi, Fardin Ayar, Ehsan Javanmardi, Manabu Tsukada, Mahdi Javanmardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.18954)  

**Abstract**: LiDAR-based localization and SLAM often rely on iterative matching algorithms, particularly the Iterative Closest Point (ICP) algorithm, to align sensor data with pre-existing maps or previous scans. However, ICP is prone to errors in featureless environments and dynamic scenes, leading to inaccurate pose estimation. Accurately predicting the uncertainty associated with ICP is crucial for robust state estimation but remains challenging, as existing approaches often rely on handcrafted models or simplified assumptions. Moreover, a few deep learning-based methods for localizability estimation either depend on a pre-built map, which may not always be available, or provide a binary classification of localizable versus non-localizable, which fails to properly model uncertainty. In this work, we propose a data-driven framework that leverages deep learning to estimate the registration error covariance of ICP before matching, even in the absence of a reference map. By associating each LiDAR scan with a reliable 6-DoF error covariance estimate, our method enables seamless integration of ICP within Kalman filtering, enhancing localization accuracy and robustness. Extensive experiments on the KITTI dataset demonstrate the effectiveness of our approach, showing that it accurately predicts covariance and, when applied to localization using a pre-built map or SLAM, reduces localization errors and improves robustness. 

**Abstract (ZH)**: 基于LiDAR的定位与SLAM往往依赖于迭代配准算法，特别是ICP算法，将传感器数据与已有地图或先前扫描进行对齐。然而，ICP在缺乏特征环境和动态场景中容易出错，导致姿态估计不准确。准确预测ICP相关的不确定性对于鲁棒的状态估计至关重要，但现有方法往往依赖于手工制作的模型或简化假设，仍然具有挑战性。此外，一些基于深度学习的可定位性估计方法要么依赖于先建好的地图，这可能并不总是可用的，要么仅二元分类可定位与不可定位，无法很好地建模不确定性。本文提出了一种数据驱动框架，利用深度学习在匹配前估计ICP的配准误差协方差，即使在没有参考地图的情况下也是如此。通过将每个LiDAR扫描与可靠的6-DoF误差协方差估计关联，我们的方法能够无缝地将ICP集成到卡尔曼滤波中，从而提升定位的准确性和鲁棒性。在KITTI数据集上的广泛实验表明，我们的方法能够准确预测协方差，并在使用先建好的地图或SLAM进行定位时，减少定位误差并提升鲁棒性。 

---
# Eva-VLA: Evaluating Vision-Language-Action Models' Robustness Under Real-World Physical Variations 

**Title (ZH)**: Eva-VLA：评估视觉-语言-行动模型在现实世界物理变化下的稳健性 

**Authors**: Hanqing Liu, Jiahuan Long, Junqi Wu, Jiacheng Hou, Huili Tang, Tingsong Jiang, Weien Zhou, Wen Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18953)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as promising solutions for robotic manipulation, yet their robustness to real-world physical variations remains critically underexplored. To bridge this gap, we propose Eva-VLA, the first unified framework that systematically evaluates the robustness of VLA models by transforming discrete physical variations into continuous optimization problems. However, comprehensively assessing VLA robustness presents two key challenges: (1) how to systematically characterize diverse physical variations encountered in real-world deployments while maintaining evaluation reproducibility, and (2) how to discover worst-case scenarios without prohibitive real-world data collection costs efficiently. To address the first challenge, we decompose real-world variations into three critical domains: object 3D transformations that affect spatial reasoning, illumination variations that challenge visual perception, and adversarial patches that disrupt scene understanding. For the second challenge, we introduce a continuous black-box optimization framework that transforms discrete physical variations into parameter optimization, enabling systematic exploration of worst-case scenarios. Extensive experiments on state-of-the-art OpenVLA models across multiple benchmarks reveal alarming vulnerabilities: all variation types trigger failure rates exceeding 60%, with object transformations causing up to 97.8% failure in long-horizon tasks. Our findings expose critical gaps between controlled laboratory success and unpredictable deployment readiness, while the Eva-VLA framework provides a practical pathway for hardening VLA-based robotic manipulation models against real-world deployment challenges. 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型作为机器人操作的有前途的解决方案已经出现，但它们对现实世界物理变化的鲁棒性仍严重未被探索。为弥补这一缺口，我们提出了Eva-VLA，这是第一个通过将离散的物理变化转化为连续的优化问题来系统评估VLA模型鲁棒性的统一框架。然而，全面评估VLA鲁棒性面临两个关键挑战：（1）在保持评估可再现性的同时，如何系统地表征现实世界部署中遇到的多种物理变化，（2）如何高效地发现最坏情况场景而无需 prohibitive 的现实世界数据收集成本。为应对第一个挑战，我们将现实世界的变化分解为三个关键领域：影响空间推理的对象三维变换、挑战视觉感知的照明变化以及干扰场景理解的对抗性补丁。为应对第二个挑战，我们引入了一个连续的黑盒优化框架，将离散的物理变化转化为参数优化，从而系统地探索最坏情况场景。在多个基准测试上的最新OpenVLA模型的广泛实验揭示了令人震惊的脆弱性：所有变化类型均触发超过60%的失败率，其中对象变换在长时任务中最高可导致97.8%的失败率。我们的发现揭示了实验室控制成功与不可预测的部署准备之间的关键差距，而Eva-VLA框架提供了一条实用的道路，用于提高基于VLA的机器人操作模型在现实世界部署中的鲁棒性。 

---
# Lang2Morph: Language-Driven Morphological Design of Robotic Hands 

**Title (ZH)**: 基于语言驱动的手部形态学设计：Lang2Morph 

**Authors**: Yanyuan Qiao, Kieran Gilday, Yutong Xie, Josie Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2509.18937)  

**Abstract**: Designing robotic hand morphologies for diverse manipulation tasks requires balancing dexterity, manufacturability, and task-specific functionality. While open-source frameworks and parametric tools support reproducible design, they still rely on expert heuristics and manual tuning. Automated methods using optimization are often compute-intensive, simulation-dependent, and rarely target dexterous hands. Large language models (LLMs), with their broad knowledge of human-object interactions and strong generative capabilities, offer a promising alternative for zero-shot design reasoning. In this paper, we present Lang2Morph, a language-driven pipeline for robotic hand design. It uses LLMs to translate natural-language task descriptions into symbolic structures and OPH-compatible parameters, enabling 3D-printable task-specific morphologies. The pipeline consists of: (i) Morphology Design, which maps tasks into semantic tags, structural grammars, and OPH-compatible parameters; and (ii) Selection and Refinement, which evaluates design candidates based on semantic alignment and size compatibility, and optionally applies LLM-guided refinement when needed. We evaluate Lang2Morph across varied tasks, and results show that our approach can generate diverse, task-relevant morphologies. To our knowledge, this is the first attempt to develop an LLM-based framework for task-conditioned robotic hand design. 

**Abstract (ZH)**: 基于大语言模型的零样本设计推理在机器人手设计中的应用：Lang2Morph管道 

---
# Bi-VLA: Bilateral Control-Based Imitation Learning via Vision-Language Fusion for Action Generation 

**Title (ZH)**: 基于视觉-语言融合的双边控制 imitation 学习行动生成 

**Authors**: Masato Kobayashi, Thanpimon Buamanee  

**Link**: [PDF](https://arxiv.org/pdf/2509.18865)  

**Abstract**: We propose Bilateral Control-Based Imitation Learning via Vision-Language Fusion for Action Generation (Bi-VLA), a novel framework that extends bilateral control-based imitation learning to handle more than one task within a single model. Conventional bilateral control methods exploit joint angle, velocity, torque, and vision for precise manipulation but require task-specific models, limiting their generality. Bi-VLA overcomes this limitation by utilizing robot joint angle, velocity, and torque data from leader-follower bilateral control with visual features and natural language instructions through SigLIP and FiLM-based fusion. We validated Bi-VLA on two task types: one requiring supplementary language cues and another distinguishable solely by vision. Real-robot experiments showed that Bi-VLA successfully interprets vision-language combinations and improves task success rates compared to conventional bilateral control-based imitation learning. Our Bi-VLA addresses the single-task limitation of prior bilateral approaches and provides empirical evidence that combining vision and language significantly enhances versatility. Experimental results validate the effectiveness of Bi-VLA in real-world tasks. For additional material, please visit the website: this https URL 

**Abstract (ZH)**: 基于双边控制的视觉-语言融合动作生成的双向控制 imitation 学习（Bi-VLA） 

---
# DexSkin: High-Coverage Conformable Robotic Skin for Learning Contact-Rich Manipulation 

**Title (ZH)**: DexSkin: 高覆盖率顺应式机器人皮肤用于学习接触丰富操作 

**Authors**: Suzannah Wistreich, Baiyu Shi, Stephen Tian, Samuel Clarke, Michael Nath, Chengyi Xu, Zhenan Bao, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18830)  

**Abstract**: Human skin provides a rich tactile sensing stream, localizing intentional and unintentional contact events over a large and contoured region. Replicating these tactile sensing capabilities for dexterous robotic manipulation systems remains a longstanding challenge. In this work, we take a step towards this goal by introducing DexSkin. DexSkin is a soft, conformable capacitive electronic skin that enables sensitive, localized, and calibratable tactile sensing, and can be tailored to varying geometries. We demonstrate its efficacy for learning downstream robotic manipulation by sensorizing a pair of parallel jaw gripper fingers, providing tactile coverage across almost the entire finger surfaces. We empirically evaluate DexSkin's capabilities in learning challenging manipulation tasks that require sensing coverage across the entire surface of the fingers, such as reorienting objects in hand and wrapping elastic bands around boxes, in a learning-from-demonstration framework. We then show that, critically for data-driven approaches, DexSkin can be calibrated to enable model transfer across sensor instances, and demonstrate its applicability to online reinforcement learning on real robots. Our results highlight DexSkin's suitability and practicality for learning real-world, contact-rich manipulation. Please see our project webpage for videos and visualizations: this https URL. 

**Abstract (ZH)**: 人类皮肤提供了一条丰富的触觉传感流，能够在大面积且形状复杂的区域定位有意和无意的接触事件。为 Dexterous 机器人 manipulation 系统复制这些触觉传感能力仍然是一个长期的挑战。在本文中，我们朝着这一目标迈出了一步，介绍了一种名为 DexSkin 的软性可调节电容式电子皮肤。DexSkin 具备灵敏、局部化和可标定的触觉传感功能，并可根据不同的几何形状进行定制。我们通过对并指夹爪手指进行传感化处理，展示了其在几乎整个手指表面提供触觉覆盖方面的有效性。我们还在演示学习框架中，通过 DexSkin 的能力来学习需要覆盖手指整个表面的传感覆盖的复杂 manipulation 任务，例如在手中重新定向物体和在盒子上缠绕弹性带子。我们还展示了，对于数据驱动的方法，DexSkin 可以被标定以实现传感器实例之间的模型迁移，并展示了其在真实机器人上的在线强化学习中的适用性。我们的结果强调了 DexSkin 在学习真实世界、接触丰富的 manipulation 任务方面的适用性和实用性。请参见我们的项目网页，观看相关视频和可视化内容：this https URL。 

---
# Application Management in C-ITS: Orchestrating Demand-Driven Deployments and Reconfigurations 

**Title (ZH)**: C-ITS中应用管理：基于需求的部署与重新配置协调 

**Authors**: Lukas Zanger, Bastian Lampe, Lennart Reiher, Lutz Eckstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.18793)  

**Abstract**: Vehicles are becoming increasingly automated and interconnected, enabling the formation of cooperative intelligent transport systems (C-ITS) and the use of offboard services. As a result, cloud-native techniques, such as microservices and container orchestration, play an increasingly important role in their operation. However, orchestrating applications in a large-scale C-ITS poses unique challenges due to the dynamic nature of the environment and the need for efficient resource utilization. In this paper, we present a demand-driven application management approach that leverages cloud-native techniques - specifically Kubernetes - to address these challenges. Taking into account the demands originating from different entities within the C-ITS, the approach enables the automation of processes, such as deployment, reconfiguration, update, upgrade, and scaling of microservices. Executing these processes on demand can, for example, reduce computing resource consumption and network traffic. A demand may include a request for provisioning an external supporting service, such as a collective environment model. The approach handles changing and new demands by dynamically reconciling them through our proposed application management framework built on Kubernetes and the Robot Operating System (ROS 2). We demonstrate the operation of our framework in the C-ITS use case of collective environment perception and make the source code of the prototypical framework publicly available at this https URL . 

**Abstract (ZH)**: 车辆 increasingly automated and interconnected，促使合作智能运输系统（C-ITS）的形成以及外部服务的利用。随之而来，云原生技术，如微服务和容器编排，在其运行中发挥了越来越重要的作用。然而，在大规模C-ITS中编排应用程序由于环境的动态性和高效的资源利用率需求，带来独特的挑战。本文提出一种基于云原生技术的需求驱动应用程序管理方法，特别是Kubernetes，以应对这些挑战。该方法考虑了C-ITS内不同实体的需求，实现了部署、重新配置、更新、升级和微服务扩展的自动化过程。这些过程的需求执行，例如，可以减少计算资源消耗和网络流量。需求可能包括请求提供外部支持服务，如集体环境模型。通过在基于Kubernetes和机器人操作系统ROS 2的应用程序管理框架中的动态解算，该方法能够处理变化和新的需求。我们展示了该框架在C-ITS中的集体环境感知用例中的运行，并在以下网址公开提供了该原型框架的源代码：this https URL。 

---
# Human-Interpretable Uncertainty Explanations for Point Cloud Registration 

**Title (ZH)**: 面向人类可解释的点云配准不确定性解释 

**Authors**: Johannes A. Gaus, Loris Schneider, Yitian Shi, Jongseok Lee, Rania Rayyes, Rudolph Triebel  

**Link**: [PDF](https://arxiv.org/pdf/2509.18786)  

**Abstract**: In this paper, we address the point cloud registration problem, where well-known methods like ICP fail under uncertainty arising from sensor noise, pose-estimation errors, and partial overlap due to occlusion. We develop a novel approach, Gaussian Process Concept Attribution (GP-CA), which not only quantifies registration uncertainty but also explains it by attributing uncertainty to well-known sources of errors in registration problems. Our approach leverages active learning to discover new uncertainty sources in the wild by querying informative instances. We validate GP-CA on three publicly available datasets and in our real-world robot experiment. Extensive ablations substantiate our design choices. Our approach outperforms other state-of-the-art methods in terms of runtime, high sample-efficiency with active learning, and high accuracy. Our real-world experiment clearly demonstrates its applicability. Our video also demonstrates that GP-CA enables effective failure-recovery behaviors, yielding more robust robotic perception. 

**Abstract (ZH)**: 基于高斯过程概念归因的点云配准方法：不确定性量化与解释 

---
# VGGT-DP: Generalizable Robot Control via Vision Foundation Models 

**Title (ZH)**: VGGT-DP: 通过视觉基础模型实现可泛化的机器人控制 

**Authors**: Shijia Ge, Yinxin Zhang, Shuzhao Xie, Weixiang Zhang, Mingcai Zhou, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18778)  

**Abstract**: Visual imitation learning frameworks allow robots to learn manipulation skills from expert demonstrations. While existing approaches mainly focus on policy design, they often neglect the structure and capacity of visual encoders, limiting spatial understanding and generalization. Inspired by biological vision systems, which rely on both visual and proprioceptive cues for robust control, we propose VGGT-DP, a visuomotor policy framework that integrates geometric priors from a pretrained 3D perception model with proprioceptive feedback. We adopt the Visual Geometry Grounded Transformer (VGGT) as the visual encoder and introduce a proprioception-guided visual learning strategy to align perception with internal robot states, improving spatial grounding and closed-loop control. To reduce inference latency, we design a frame-wise token reuse mechanism that compacts multi-view tokens into an efficient spatial representation. We further apply random token pruning to enhance policy robustness and reduce overfitting. Experiments on challenging MetaWorld tasks show that VGGT-DP significantly outperforms strong baselines such as DP and DP3, particularly in precision-critical and long-horizon scenarios. 

**Abstract (ZH)**: 视觉模仿学习框架使机器人能够从专家示范中学习操作技能。现有的方法主要集中在策略设计上，但经常忽视视觉编码器的结构和容量，限制了空间理解和泛化能力。受到生物视觉系统依赖视觉和本体感受线索进行稳健控制的启发，我们提出了一种结合预训练3D感知模型的几何先验与本体感受反馈的visuomotor策略框架VGGT-DP。我们采用Visual Geometry Grounded Transformer (VGGT) 作为视觉编码器，并引入一种本体感受导向的视觉学习策略，以使感知与内部机器人状态对齐，从而提高空间定位和闭环控制能力。为了降低推理延迟，我们设计了一种帧级令牌重用机制，将多视图令牌压缩为高效的空间表示。进一步地，我们应用随机令牌剪枝以增强策略的稳健性并减少过拟合。实验结果表明，在MetaWorld的挑战任务中，VGGT-DP显著优于具有竞争力的基础模型DP和DP3，特别是在精度要求高和长期规划场景中。 

---
# MV-UMI: A Scalable Multi-View Interface for Cross-Embodiment Learning 

**Title (ZH)**: MV-UMI: 一种可扩展的多视图接口用于跨具身学习 

**Authors**: Omar Rayyan, John Abanes, Mahmoud Hafez, Anthony Tzes, Fares Abu-Dakka  

**Link**: [PDF](https://arxiv.org/pdf/2509.18757)  

**Abstract**: Recent advances in imitation learning have shown great promise for developing robust robot manipulation policies from demonstrations. However, this promise is contingent on the availability of diverse, high-quality datasets, which are not only challenging and costly to collect but are often constrained to a specific robot embodiment. Portable handheld grippers have recently emerged as intuitive and scalable alternatives to traditional robotic teleoperation methods for data collection. However, their reliance solely on first-person view wrist-mounted cameras often creates limitations in capturing sufficient scene contexts. In this paper, we present MV-UMI (Multi-View Universal Manipulation Interface), a framework that integrates a third-person perspective with the egocentric camera to overcome this limitation. This integration mitigates domain shifts between human demonstration and robot deployment, preserving the cross-embodiment advantages of handheld data-collection devices. Our experimental results, including an ablation study, demonstrate that our MV-UMI framework improves performance in sub-tasks requiring broad scene understanding by approximately 47% across 3 tasks, confirming the effectiveness of our approach in expanding the range of feasible manipulation tasks that can be learned using handheld gripper systems, without compromising the cross-embodiment advantages inherent to such systems. 

**Abstract (ZH)**: 最近在仿生学习领域的进步显示了从示范中开发鲁棒机器人操作策略的巨大潜力。然而，这一潜力取决于多样化的高质量数据集的可用性，这些数据集不仅收集困难且成本高昂，而且常常局限于特定的机器人实体。近年来，便携式手持夹爪已成为直观且可扩展的替代传统机器人遥操作数据收集方法的选择。然而，它们通常仅依赖于第一人称视角的手腕摄像头，这在捕捉足够的场景上下文时常常受限。本文介绍了MV-UMI（多视角通用操纵界面）框架，该框架将第三人称视角与主观摄像头结合，以克服这一限制。这种结合减轻了人类示范与机器人部署之间的领域转移，保留了手持数据收集设备固有的跨实体优势。我们的实验结果，包括消融研究，表明我们的MV-UMI框架在三个任务中的子任务中通过大约47%的增量提高了解释广泛场景理解的能力，证实了我们的方法在不牺牲此类系统固有的跨实体优势的前提下，能够扩展手持夹爪系统可学习的操作任务范围的有效性。 

---
# Learning Obstacle Avoidance using Double DQN for Quadcopter Navigation 

**Title (ZH)**: 使用双DQN学习Quadcopter导航障碍物避免 

**Authors**: Nishant Doshi, Amey Sutvani, Sanket Gujar  

**Link**: [PDF](https://arxiv.org/pdf/2509.18734)  

**Abstract**: One of the challenges faced by Autonomous Aerial Vehicles is reliable navigation through urban environments. Factors like reduction in precision of Global Positioning System (GPS), narrow spaces and dynamically moving obstacles make the path planning of an aerial robot a complicated task. One of the skills required for the agent to effectively navigate through such an environment is to develop an ability to avoid collisions using information from onboard depth sensors. In this paper, we propose Reinforcement Learning of a virtual quadcopter robot agent equipped with a Depth Camera to navigate through a simulated urban environment. 

**Abstract (ZH)**: 自主飞行器在城市环境中可靠导航的挑战：基于深度相机的强化学习虚拟四旋翼机器人在模拟城市环境中的导航 

---
# Query-Centric Diffusion Policy for Generalizable Robotic Assembly 

**Title (ZH)**: 以查询为中心的扩散策略及其在通用机器人装配中的应用 

**Authors**: Ziyi Xu, Haohong Lin, Shiqi Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18686)  

**Abstract**: The robotic assembly task poses a key challenge in building generalist robots due to the intrinsic complexity of part interactions and the sensitivity to noise perturbations in contact-rich settings. The assembly agent is typically designed in a hierarchical manner: high-level multi-part reasoning and low-level precise control. However, implementing such a hierarchical policy is challenging in practice due to the mismatch between high-level skill queries and low-level execution. To address this, we propose the Query-centric Diffusion Policy (QDP), a hierarchical framework that bridges high-level planning and low-level control by utilizing queries comprising objects, contact points, and skill information. QDP introduces a query-centric mechanism that identifies task-relevant components and uses them to guide low-level policies, leveraging point cloud observations to improve the policy's robustness. We conduct comprehensive experiments on the FurnitureBench in both simulation and real-world settings, demonstrating improved performance in skill precision and long-horizon success rate. In the challenging insertion and screwing tasks, QDP improves the skill-wise success rate by over 50% compared to baselines without structured queries. 

**Abstract (ZH)**: 基于查询的扩散策略（QDP）：桥接高层规划与低层控制的层次框架 

---
# 3D Flow Diffusion Policy: Visuomotor Policy Learning via Generating Flow in 3D Space 

**Title (ZH)**: 3D流扩散策略：通过在3D空间生成流进行知觉运动策略学习 

**Authors**: Sangjun Noh, Dongwoo Nam, Kangmin Kim, Geonhyup Lee, Yeonguk Yu, Raeyoung Kang, Kyoobin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.18676)  

**Abstract**: Learning robust visuomotor policies that generalize across diverse objects and interaction dynamics remains a central challenge in robotic manipulation. Most existing approaches rely on direct observation-to-action mappings or compress perceptual inputs into global or object-centric features, which often overlook localized motion cues critical for precise and contact-rich manipulation. We present 3D Flow Diffusion Policy (3D FDP), a novel framework that leverages scene-level 3D flow as a structured intermediate representation to capture fine-grained local motion cues. Our approach predicts the temporal trajectories of sampled query points and conditions action generation on these interaction-aware flows, implemented jointly within a unified diffusion architecture. This design grounds manipulation in localized dynamics while enabling the policy to reason about broader scene-level consequences of actions. Extensive experiments on the MetaWorld benchmark show that 3D FDP achieves state-of-the-art performance across 50 tasks, particularly excelling on medium and hard settings. Beyond simulation, we validate our method on eight real-robot tasks, where it consistently outperforms prior baselines in contact-rich and non-prehensile scenarios. These results highlight 3D flow as a powerful structural prior for learning generalizable visuomotor policies, supporting the development of more robust and versatile robotic manipulation. Robot demonstrations, additional results, and code can be found at this https URL. 

**Abstract (ZH)**: 学习能够在多样化的物体和交互动力学中泛化的鲁棒视运动策略仍然是机器人操作领域的一个中心挑战。我们提出了3D流动扩散策略（3D FDP），这是一种新颖的框架，利用场景级3D流动作为结构化的中间表示来捕捉细粒度的局部运动线索。该方法预测采样查询点的时序轨迹，并基于这些交互感知的流动条件化动作生成，该流程在统一的扩散架构中联合实现。这种设计将操作与局部动力学联系起来，同时使策略能够推理更广泛的场景级动作后果。在MetaWorld基准上进行的广泛实验表明，3D FDP在50个任务中实现了最先进的性能，特别是在中等和困难设置中表现尤为出色。除了仿真实验，我们在八项真实机器人任务中验证了该方法，在接触丰富的非抓取场景中，其表现始终优于之前的基线方法。这些结果突出了3D流动作为学习泛化视运动策略的强大结构先验，支持更 robust 和多功能的机器人操作的发展。机器人演示、额外结果和代码可在此处找到：this https URL。 

---
# N2M: Bridging Navigation and Manipulation by Learning Pose Preference from Rollout 

**Title (ZH)**: N2M: 通过学习展开过程中的姿态偏好实现导航与操作的 bridge 

**Authors**: Kaixin Chai, Hyunjun Lee, Joseph J. Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.18671)  

**Abstract**: In mobile manipulation, the manipulation policy has strong preferences for initial poses where it is executed. However, the navigation module focuses solely on reaching the task area, without considering which initial pose is preferable for downstream manipulation. To address this misalignment, we introduce N2M, a transition module that guides the robot to a preferable initial pose after reaching the task area, thereby substantially improving task success rates. N2M features five key advantages: (1) reliance solely on ego-centric observation without requiring global or historical information; (2) real-time adaptation to environmental changes; (3) reliable prediction with high viewpoint robustness; (4) broad applicability across diverse tasks, manipulation policies, and robot hardware; and (5) remarkable data efficiency and generalizability. We demonstrate the effectiveness of N2M through extensive simulation and real-world experiments. In the PnPCounterToCab task, N2M improves the averaged success rate from 3% with the reachability-based baseline to 54%. Furthermore, in the Toybox Handover task, N2M provides reliable predictions even in unseen environments with only 15 data samples, showing remarkable data efficiency and generalizability. 

**Abstract (ZH)**: 移动操作中，操作策略对执行初始姿态有强烈的偏好。然而，导航模块仅专注于抵达任务区域，而不考虑哪种初始姿态更符合后续操作的需要。为解决这一问题，我们引入了N2M过渡模块，在抵达任务区域后引导机器人到达更优选的初始姿态，从而显著提高任务成功率。N2M具备五大优势：（1）仅依赖于基于自身的观测，无需全球或历史信息；（2）实时适应环境变化；（3）高视角稳健性下的可靠预测；（4）适用于多种任务、操作策略和机器人硬件；（5）显著的数据效率和泛化能力。我们通过广泛的模拟和实际实验验证了N2M的有效性。在PnPCounterToCab任务中，N2M将基于可达性的基线成功率从3%提升至54%。此外，在Toybox Handover任务中，N2M仅使用15个数据样本便能在未见环境中提供可靠的预测，展示了显著的数据效率和泛化能力。 

---
# Distributionally Robust Safe Motion Planning with Contextual Information 

**Title (ZH)**: 基于上下文信息的分布鲁棒安全运动规划 

**Authors**: Kaizer Rahaman, Simran Kumari, Ashish R. Hota  

**Link**: [PDF](https://arxiv.org/pdf/2509.18666)  

**Abstract**: We present a distributionally robust approach for collision avoidance by incorporating contextual information. Specifically, we embed the conditional distribution of future trajectory of the obstacle conditioned on the motion of the ego agent in a reproducing kernel Hilbert space (RKHS) via the conditional kernel mean embedding operator. Then, we define an ambiguity set containing all distributions whose embedding in the RKHS is within a certain distance from the empirical estimate of conditional mean embedding learnt from past data. Consequently, a distributionally robust collision avoidance constraint is formulated, and included in the receding horizon based motion planning formulation of the ego agent. Simulation results show that the proposed approach is more successful in avoiding collision compared to approaches that do not include contextual information and/or distributional robustness in their formulation in several challenging scenarios. 

**Abstract (ZH)**: 我们提出了一种通过融合上下文信息的分布鲁棒方法来实现碰撞避免。具体地，我们通过条件核均值嵌入算子将代理ego的运动条件下障碍物未来轨迹的条件分布嵌入到再生核希尔伯特空间（RKHS）中。然后，我们定义一个模糊集合，该集合包含所有嵌入与RKHS中条件均值嵌入的经验估计在特定距离内的分布。由此，我们制定了一个分布鲁棒的碰撞避免约束，并将其纳入基于回视 horizons的运动规划框架中。模拟结果表明，与未包含上下文信息和/或分布鲁棒性的方法相比，所提出的方法在多个具有挑战性的场景中更成功地避免了碰撞。 

---
# SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer 

**Title (ZH)**: SPiDR: 一种简单的零样本安全性方法在仿真实际转移 

**Authors**: Yarden As, Chengrui Qu, Benjamin Unger, Dongho Kang, Max van der Hart, Laixi Shi, Stelian Coros, Adam Wierman, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2509.18648)  

**Abstract**: Safety remains a major concern for deploying reinforcement learning (RL) in real-world applications. Simulators provide safe, scalable training environments, but the inevitable sim-to-real gap introduces additional safety concerns, as policies must satisfy constraints in real-world conditions that differ from simulation. To address this challenge, robust safe RL techniques offer principled methods, but are often incompatible with standard scalable training pipelines. In contrast, domain randomization, a simple and popular sim-to-real technique, stands out as a promising alternative, although it often results in unsafe behaviors in practice. We present SPiDR, short for Sim-to-real via Pessimistic Domain Randomization -- a scalable algorithm with provable guarantees for safe sim-to-real transfer. SPiDR uses domain randomization to incorporate the uncertainty about the sim-to-real gap into the safety constraints, making it versatile and highly compatible with existing training pipelines. Through extensive experiments on sim-to-sim benchmarks and two distinct real-world robotic platforms, we demonstrate that SPiDR effectively ensures safety despite the sim-to-real gap while maintaining strong performance. 

**Abstract (ZH)**: Safety remains a major concern for deploying reinforcement learning (RL) in real-world applications. Simulators provide safe, scalable training environments, but the inevitable sim-to-real gap introduces additional safety concerns, as policies must satisfy constraints in real-world conditions that differ from simulation. To address this challenge, robust safe RL techniques offer principled methods, but are often incompatible with standard scalable training pipelines. In contrast, domain randomization, a simple and popular sim-to-real technique, stands out as a promising alternative, although it often results in unsafe behaviors in practice. We present SPiDR, short for Sim-to-real via Pessimistic Domain Randomization -- a scalable algorithm with provable guarantees for safe sim-to-real transfer. SPiDR uses domain randomization to incorporate the uncertainty about the sim-to-real gap into the safety constraints, making it versatile and highly compatible with existing training pipelines. Through extensive experiments on sim-to-sim benchmarks and two distinct real-world robotic platforms, we demonstrate that SPiDR effectively ensures safety despite the sim-to-real gap while maintaining strong performance。 

---
# Do You Need Proprioceptive States in Visuomotor Policies? 

**Title (ZH)**: 你需要在视觉运动策略中包含本体感觉状态吗？ 

**Authors**: Juntu Zhao, Wenbo Lu, Di Zhang, Yufeng Liu, Yushen Liang, Tianluo Zhang, Yifeng Cao, Junyuan Xie, Yingdong Hu, Shengjie Wang, Junliang Guo, Dequan Wang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18644)  

**Abstract**: Imitation-learning-based visuomotor policies have been widely used in robot manipulation, where both visual observations and proprioceptive states are typically adopted together for precise control. However, in this study, we find that this common practice makes the policy overly reliant on the proprioceptive state input, which causes overfitting to the training trajectories and results in poor spatial generalization. On the contrary, we propose the State-free Policy, removing the proprioceptive state input and predicting actions only conditioned on visual observations. The State-free Policy is built in the relative end-effector action space, and should ensure the full task-relevant visual observations, here provided by dual wide-angle wrist cameras. Empirical results demonstrate that the State-free policy achieves significantly stronger spatial generalization than the state-based policy: in real-world tasks such as pick-and-place, challenging shirt-folding, and complex whole-body manipulation, spanning multiple robot embodiments, the average success rate improves from 0\% to 85\% in height generalization and from 6\% to 64\% in horizontal generalization. Furthermore, they also show advantages in data efficiency and cross-embodiment adaptation, enhancing their practicality for real-world deployment. 

**Abstract (ZH)**: 基于imitation学习的无状态视觉运动策略在机器人操作中的应用研究 

---
# Number Adaptive Formation Flight Planning via Affine Deformable Guidance in Narrow Environments 

**Title (ZH)**: 窄环境中方形可变形引导的自适应编队飞行规划 

**Authors**: Yuan Zhou, Jialiang Hou, Guangtong Xu, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18636)  

**Abstract**: Formation maintenance with varying number of drones in narrow environments hinders the convergence of planning to the desired configurations. To address this challenge, this paper proposes a formation planning method guided by Deformable Virtual Structures (DVS) with continuous spatiotemporal transformation. Firstly, to satisfy swarm safety distance and preserve formation shape filling integrity for irregular formation geometries, we employ Lloyd algorithm for uniform $\underline{PA}$rtitioning and Hungarian algorithm for $\underline{AS}$signment (PAAS) in DVS. Subsequently, a spatiotemporal trajectory involving DVS is planned using primitive-based path search and nonlinear trajectory optimization. The DVS trajectory achieves adaptive transitions with respect to a varying number of drones while ensuring adaptability to narrow environments through affine transformation. Finally, each agent conducts distributed trajectory planning guided by desired spatiotemporal positions within the DVS, while incorporating collision avoidance and dynamic feasibility requirements. Our method enables up to 15\% of swarm numbers to join or leave in cluttered environments while rapidly restoring the desired formation shape in simulation. Compared to cutting-edge formation planning method, we demonstrate rapid formation recovery capacity and environmental adaptability. Real-world experiments validate the effectiveness and resilience of our formation planning method. 

**Abstract (ZH)**: 狭窄环境中有变化无人机数量的编队维持會阻碍规划收敛到所需配置。为此，本文提出了一种由可变形虚拟结构（DVS）引导并在时空上持续变换的编队规划方法。首先，为了满足群体安全距离并保持不规则编队几何形状的整体完整性，我们采用Lloyd算法进行均匀PA分割和匈牙利算法进行AS分配（PAAS）以在DVS中实现。随后，基于.primitive.路径搜索和非线性轨迹优化，规划涉及DVS的时空轨迹。DVS轨迹能够针对变化的无人机数量实现自适应过渡，并通过仿射变换确保适应狭窄环境。最后，每个代理根据DVS内的期望时空位置进行分布式轨迹规划，同时包含碰撞规避和动态可行性要求。我们的方法在杂乱环境中使多达15%的群体数量加入或离开，并在仿真中快速恢复所需的编队形状。与最新的编队规划方法相比，我们展示了快速的编队恢复能力和环境适应性。实地实验验证了我们编队规划方法的有效性和鲁棒性。 

---
# Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training 

**Title (ZH)**: 跨域可迁移的模拟与现实环境政策共训练 

**Authors**: Shuo Cheng, Liqian Ma, Zhenyang Chen, Ajay Mandlekar, Caelan Garrett, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18631)  

**Abstract**: Behavior cloning has shown promise for robot manipulation, but real-world demonstrations are costly to acquire at scale. While simulated data offers a scalable alternative, particularly with advances in automated demonstration generation, transferring policies to the real world is hampered by various simulation and real domain gaps. In this work, we propose a unified sim-and-real co-training framework for learning generalizable manipulation policies that primarily leverages simulation and only requires a few real-world demonstrations. Central to our approach is learning a domain-invariant, task-relevant feature space. Our key insight is that aligning the joint distributions of observations and their corresponding actions across domains provides a richer signal than aligning observations (marginals) alone. We achieve this by embedding an Optimal Transport (OT)-inspired loss within the co-training framework, and extend this to an Unbalanced OT framework to handle the imbalance between abundant simulation data and limited real-world examples. We validate our method on challenging manipulation tasks, showing it can leverage abundant simulation data to achieve up to a 30% improvement in the real-world success rate and even generalize to scenarios seen only in simulation. 

**Abstract (ZH)**: 基于模拟与现实融合训练的通用化操作策略学习方法 

---
# The Case for Negative Data: From Crash Reports to Counterfactuals for Reasonable Driving 

**Title (ZH)**: 负数据的案例：从崩溃报告到合理的驾驶反事实推理 

**Authors**: Jay Patrikar, Apoorva Sharma, Sushant Veer, Boyi Li, Sebastian Scherer, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2509.18626)  

**Abstract**: Learning-based autonomous driving systems are trained mostly on incident-free data, offering little guidance near safety-performance boundaries. Real crash reports contain precisely the contrastive evidence needed, but they are hard to use: narratives are unstructured, third-person, and poorly grounded to sensor views. We address these challenges by normalizing crash narratives to ego-centric language and converting both logs and crashes into a unified scene-action representation suitable for retrieval. At decision time, our system adjudicates proposed actions by retrieving relevant precedents from this unified index; an agentic counterfactual extension proposes plausible alternatives, retrieves for each, and reasons across outcomes before deciding. On a nuScenes benchmark, precedent retrieval substantially improves calibration, with recall on contextually preferred actions rising from 24% to 53%. The counterfactual variant preserves these gains while sharpening decisions near risk. 

**Abstract (ZH)**: 基于学习的自主驾驶系统主要在无事故数据上训练，这提供了有限的安全性能边界指导。实际情况中的事故报告包含了所需的对比证据，但这些报告难以使用：报告内容结构松散，为第三人称视角，并且与传感器视图关联性差。我们通过将事故叙述标准化为以自我为中心的语言，并将事故日志和事故场景统一封装为适合检索的场景-行动表示，来应对这些挑战。在决策时刻，系统通过检索统一索引中的相关先例来裁定提议的动作；有能动性的反事实扩展提出可能的替代方案，检索每个方案，并跨结果进行推理，最后作出决定。在nuScenes基准测试中，先例检索显著提高了校准度，上下文偏好动作的召回率从24%提高到53%。反事实变体在保留这些改进的同时，使在高风险区域的决策更加精准。 

---
# SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones 

**Title (ZH)**: SINGER：无人机上载通用视觉-语言导航策略 

**Authors**: Maximilian Adang, JunEn Low, Ola Shorinwa, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2509.18610)  

**Abstract**: Large vision-language models have driven remarkable progress in open-vocabulary robot policies, e.g., generalist robot manipulation policies, that enable robots to complete complex tasks specified in natural language. Despite these successes, open-vocabulary autonomous drone navigation remains an unsolved challenge due to the scarcity of large-scale demonstrations, real-time control demands of drones for stabilization, and lack of reliable external pose estimation modules. In this work, we present SINGER for language-guided autonomous drone navigation in the open world using only onboard sensing and compute. To train robust, open-vocabulary navigation policies, SINGER leverages three central components: (i) a photorealistic language-embedded flight simulator with minimal sim-to-real gap using Gaussian Splatting for efficient data generation, (ii) an RRT-inspired multi-trajectory generation expert for collision-free navigation demonstrations, and these are used to train (iii) a lightweight end-to-end visuomotor policy for real-time closed-loop control. Through extensive hardware flight experiments, we demonstrate superior zero-shot sim-to-real transfer of our policy to unseen environments and unseen language-conditioned goal objects. When trained on ~700k-1M observation action pairs of language conditioned visuomotor data and deployed on hardware, SINGER outperforms a velocity-controlled semantic guidance baseline by reaching the query 23.33% more on average, and maintains the query in the field of view 16.67% more on average, with 10% fewer collisions. 

**Abstract (ZH)**: 基于语言指导的开放世界自主无人机导航 

---
# PIE: Perception and Interaction Enhanced End-to-End Motion Planning for Autonomous Driving 

**Title (ZH)**: PIE: 感知与交互增强的端到端自动驾驶运动规划 

**Authors**: Chengran Yuan, Zijian Lu, Zhanqi Zhang, Yimin Zhao, Zefan Huang, Shuo Sun, Jiawei Sun, Jiahui Li, Christina Dao Wen Lee, Dongen Li, Marcelo H. Ang Jr  

**Link**: [PDF](https://arxiv.org/pdf/2509.18609)  

**Abstract**: End-to-end motion planning is promising for simplifying complex autonomous driving pipelines. However, challenges such as scene understanding and effective prediction for decision-making continue to present substantial obstacles to its large-scale deployment. In this paper, we present PIE, a pioneering framework that integrates advanced perception, reasoning, and intention modeling to dynamically capture interactions between the ego vehicle and surrounding agents. It incorporates a bidirectional Mamba fusion that addresses data compression losses in multimodal fusion of camera and LiDAR inputs, alongside a novel reasoning-enhanced decoder integrating Mamba and Mixture-of-Experts to facilitate scene-compliant anchor selection and optimize adaptive trajectory inference. PIE adopts an action-motion interaction module to effectively utilize state predictions of surrounding agents to refine ego planning. The proposed framework is thoroughly validated on the NAVSIM benchmark. PIE, without using any ensemble and data augmentation techniques, achieves an 88.9 PDM score and 85.6 EPDM score, surpassing the performance of prior state-of-the-art methods. Comprehensive quantitative and qualitative analyses demonstrate that PIE is capable of reliably generating feasible and high-quality ego trajectories. 

**Abstract (ZH)**: 端到端运动规划对于简化复杂的自动驾驶管道具有潜力，但场景理解及有效预测决策等方面仍存在挑战，阻碍其大规模部署。本文提出PIE，一种结合高级感知、推理和意图建模的先驱框架，动态捕捉ego车辆与周围代理之间的互动。该框架整合双向Mamba融合以解决多模态融合中摄像头和LiDAR输入的数据压缩损失问题，并采用新颖的增强推理解码器整合Mamba和专家混合模型，以促进场景合规的锚点选择和自适应轨迹推理优化。PIE采用动作-运动交互模块有效利用周围代理的状态预测来细化ego规划。本文在NASVCIM基准上对提出的框架进行了全面验证，在不使用任何集成和数据增强技术的情况下，PIE取得了88.9 PDM分数和85.6 EPDM分数，超过了现有最先进的方法。全面的定量和定性分析表明，PIE能够可靠地生成可行且高质量的ego轨迹。 

---
# End-to-End Crop Row Navigation via LiDAR-Based Deep Reinforcement Learning 

**Title (ZH)**: 基于LiDAR的端到端作物行导航深度强化学习方法 

**Authors**: Ana Luiza Mineiro, Francisco Affonso, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.18608)  

**Abstract**: Reliable navigation in under-canopy agricultural environments remains a challenge due to GNSS unreliability, cluttered rows, and variable lighting. To address these limitations, we present an end-to-end learning-based navigation system that maps raw 3D LiDAR data directly to control commands using a deep reinforcement learning policy trained entirely in simulation. Our method includes a voxel-based downsampling strategy that reduces LiDAR input size by 95.83%, enabling efficient policy learning without relying on labeled datasets or manually designed control interfaces. The policy was validated in simulation, achieving a 100% success rate in straight-row plantations and showing a gradual decline in performance as row curvature increased, tested across varying sinusoidal frequencies and amplitudes. 

**Abstract (ZH)**: 基于深度强化学习的端到端学习导航系统：在林下农业环境中的可靠导航 

---
# Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills 

**Title (ZH)**: 与你的躯体型代理共同成长：一种包含人类在环的长期 horizon 操作技能终身代码生成框架 

**Authors**: Yuan Meng, Zhenguo Sun, Max Fest, Xukun Li, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.18597)  

**Abstract**: Large language models (LLMs)-based code generation for robotic manipulation has recently shown promise by directly translating human instructions into executable code, but existing methods remain noisy, constrained by fixed primitives and limited context windows, and struggle with long-horizon tasks. While closed-loop feedback has been explored, corrected knowledge is often stored in improper formats, restricting generalization and causing catastrophic forgetting, which highlights the need for learning reusable skills. Moreover, approaches that rely solely on LLM guidance frequently fail in extremely long-horizon scenarios due to LLMs' limited reasoning capability in the robotic domain, where such issues are often straightforward for humans to identify. To address these challenges, we propose a human-in-the-loop framework that encodes corrections into reusable skills, supported by external memory and Retrieval-Augmented Generation with a hint mechanism for dynamic reuse. Experiments on Ravens, Franka Kitchen, and MetaWorld, as well as real-world settings, show that our framework achieves a 0.93 success rate (up to 27% higher than baselines) and a 42% efficiency improvement in correction rounds. It can robustly solve extremely long-horizon tasks such as "build a house", which requires planning over 20 primitives. 

**Abstract (ZH)**: 基于大型语言模型的代码生成在机器人操作中的前景：一种支持外部记忆和提示机制的循环反馈框架 

---
# VLN-Zero: Rapid Exploration and Cache-Enabled Neurosymbolic Vision-Language Planning for Zero-Shot Transfer in Robot Navigation 

**Title (ZH)**: VLN-Zero: 资源快速探索与缓存增强的神经符号视觉语言规划在机器人导航中的零样本迁移 

**Authors**: Neel P. Bhatt, Yunhao Yang, Rohan Siva, Pranay Samineni, Daniel Milan, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18592)  

**Abstract**: Rapid adaptation in unseen environments is essential for scalable real-world autonomy, yet existing approaches rely on exhaustive exploration or rigid navigation policies that fail to generalize. We present VLN-Zero, a two-phase vision-language navigation framework that leverages vision-language models to efficiently construct symbolic scene graphs and enable zero-shot neurosymbolic navigation. In the exploration phase, structured prompts guide VLM-based search toward informative and diverse trajectories, yielding compact scene graph representations. In the deployment phase, a neurosymbolic planner reasons over the scene graph and environmental observations to generate executable plans, while a cache-enabled execution module accelerates adaptation by reusing previously computed task-location trajectories. By combining rapid exploration, symbolic reasoning, and cache-enabled execution, the proposed framework overcomes the computational inefficiency and poor generalization of prior vision-language navigation methods, enabling robust and scalable decision-making in unseen environments. VLN-Zero achieves 2x higher success rate compared to state-of-the-art zero-shot models, outperforms most fine-tuned baselines, and reaches goal locations in half the time with 55% fewer VLM calls on average compared to state-of-the-art models across diverse environments. Codebase, datasets, and videos for VLN-Zero are available at: this https URL. 

**Abstract (ZH)**: 无监督环境适应的快速视觉-语言导航框架：VLN-Zero 

---
# LCMF: Lightweight Cross-Modality Mambaformer for Embodied Robotics VQA 

**Title (ZH)**: LCMF：轻量级跨模态Mambaformer在感知机器人VQA中的应用 

**Authors**: Zeyi Kang, Liang He, Yanxin Zhang, Zuheng Ming, Kaixing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18576)  

**Abstract**: Multimodal semantic learning plays a critical role in embodied intelligence, especially when robots perceive their surroundings, understand human instructions, and make intelligent decisions. However, the field faces technical challenges such as effective fusion of heterogeneous data and computational efficiency in resource-constrained environments. To address these challenges, this study proposes the lightweight LCMF cascaded attention framework, introducing a multi-level cross-modal parameter sharing mechanism into the Mamba module. By integrating the advantages of Cross-Attention and Selective parameter-sharing State Space Models (SSMs), the framework achieves efficient fusion of heterogeneous modalities and semantic complementary alignment. Experimental results show that LCMF surpasses existing multimodal baselines with an accuracy of 74.29% in VQA tasks and achieves competitive mid-tier performance within the distribution cluster of Large Language Model Agents (LLM Agents) in EQA video tasks. Its lightweight design achieves a 4.35-fold reduction in FLOPs relative to the average of comparable baselines while using only 166.51M parameters (image-text) and 219M parameters (video-text), providing an efficient solution for Human-Robot Interaction (HRI) applications in resource-constrained scenarios with strong multimodal decision generalization capabilities. 

**Abstract (ZH)**: 多模态语义学习在体现智能中起着关键作用，尤其是在机器人感知周围环境、理解人类指令和做出智能决策时。然而，该领域面临着如异构数据的有效融合和资源受限环境中计算效率低下等技术挑战。为解决这些挑战，本研究提出了一种轻量级LCMF级联注意力框架，将多层跨模态参数共享机制引入到Mamba模块中。通过结合跨注意力和选择性参数共享状态空间模型（SSMs）的优势，该框架实现了异构模态的有效融合和语义互补对齐。实验结果表明，LCMF在VQA任务中的准确率达到74.29%，在EQA视频任务中实现了大型语言模型代理（LLM Agents）分布集群内的竞争力中等性能。其轻量化设计相对于可比基线平均减少了4.35倍的FLOPs，同时使用了166.51M（图像-文本）和219M（视频-文本）参数，为资源受限场景中的高效人机交互（HRI）应用提供了有效的解决方案，并具备强大的多模态决策泛化能力。 

---
# Spatial Envelope MPC: High Performance Driving without a Reference 

**Title (ZH)**: 空间包络 MPC：无需参考轨迹的高性能驾驶 

**Authors**: Siyuan Yu, Congkai Shen, Yufei Xi, James Dallas, Michael Thompson, John Subosits, Hiroshi Yasuda, Tulga Ersal  

**Link**: [PDF](https://arxiv.org/pdf/2509.18506)  

**Abstract**: This paper presents a novel envelope based model predictive control (MPC) framework designed to enable autonomous vehicles to handle high performance driving across a wide range of scenarios without a predefined reference. In high performance autonomous driving, safe operation at the vehicle's dynamic limits requires a real time planning and control framework capable of accounting for key vehicle dynamics and environmental constraints when following a predefined reference trajectory is suboptimal or even infeasible. State of the art planning and control frameworks, however, are predominantly reference based, which limits their performance in such situations. To address this gap, this work first introduces a computationally efficient vehicle dynamics model tailored for optimization based control and a continuously differentiable mathematical formulation that accurately captures the entire drivable envelope. This novel model and formulation allow for the direct integration of dynamic feasibility and safety constraints into a unified planning and control framework, thereby removing the necessity for predefined references. The challenge of envelope planning, which refers to maximally approximating the safe drivable area, is tackled by combining reinforcement learning with optimization techniques. The framework is validated through both simulations and real world experiments, demonstrating its high performance across a variety of tasks, including racing, emergency collision avoidance and off road navigation. These results highlight the framework's scalability and broad applicability across a diverse set of scenarios. 

**Abstract (ZH)**: 基于包线的新型模型预测控制框架：无需预定义参考的高性能自主驾驶 

---
# RL-augmented Adaptive Model Predictive Control for Bipedal Locomotion over Challenging Terrain 

**Title (ZH)**: 基于RL增强的自适应模型预测控制在复杂地形双足行走中的应用 

**Authors**: Junnosuke Kamohara, Feiyang Wu, Chinmayee Wamorkar, Seth Hutchinson, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18466)  

**Abstract**: Model predictive control (MPC) has demonstrated effectiveness for humanoid bipedal locomotion; however, its applicability in challenging environments, such as rough and slippery terrain, is limited by the difficulty of modeling terrain interactions. In contrast, reinforcement learning (RL) has achieved notable success in training robust locomotion policies over diverse terrain, yet it lacks guarantees of constraint satisfaction and often requires substantial reward shaping. Recent efforts in combining MPC and RL have shown promise of taking the best of both worlds, but they are primarily restricted to flat terrain or quadrupedal robots. In this work, we propose an RL-augmented MPC framework tailored for bipedal locomotion over rough and slippery terrain. Our method parametrizes three key components of single-rigid-body-dynamics-based MPC: system dynamics, swing leg controller, and gait frequency. We validate our approach through bipedal robot simulations in NVIDIA IsaacLab across various terrains, including stairs, stepping stones, and low-friction surfaces. Experimental results demonstrate that our RL-augmented MPC framework produces significantly more adaptive and robust behaviors compared to baseline MPC and RL. 

**Abstract (ZH)**: 基于强化学习增强的模型预测控制框架：适用于不平滑和易滑地形的仿人双足行走 

---
# Robotic Skill Diversification via Active Mutation of Reward Functions in Reinforcement Learning During a Liquid Pouring Task 

**Title (ZH)**: 通过强化学习中奖励函数的主动变异实现机器人技能多样化：以液体倾倒任务为例 

**Authors**: Jannick van Buuren, Roberto Giglio, Loris Roveda, Luka Peternel  

**Link**: [PDF](https://arxiv.org/pdf/2509.18463)  

**Abstract**: This paper explores how deliberate mutations of reward function in reinforcement learning can produce diversified skill variations in robotic manipulation tasks, examined with a liquid pouring use case. To this end, we developed a new reward function mutation framework that is based on applying Gaussian noise to the weights of the different terms in the reward function. Inspired by the cost-benefit tradeoff model from human motor control, we designed the reward function with the following key terms: accuracy, time, and effort. The study was performed in a simulation environment created in NVIDIA Isaac Sim, and the setup included Franka Emika Panda robotic arm holding a glass with a liquid that needed to be poured into a container. The reinforcement learning algorithm was based on Proximal Policy Optimization. We systematically explored how different configurations of mutated weights in the rewards function would affect the learned policy. The resulting policies exhibit a wide range of behaviours: from variations in execution of the originally intended pouring task to novel skills useful for unexpected tasks, such as container rim cleaning, liquid mixing, and watering. This approach offers promising directions for robotic systems to perform diversified learning of specific tasks, while also potentially deriving meaningful skills for future tasks. 

**Abstract (ZH)**: 本研究探讨了在强化学习中故意对奖励函数进行变异如何产生机器人操作任务中的多样化技能变体，并通过液体倾倒用例进行了检验。为此，我们开发了一种新的奖励函数变异框架，该框架基于向奖励函数不同项的权重应用高斯噪声。受人类运动控制中的成本-收益权衡模型启发，我们设计了包含准确性、时间和努力这三项关键指标的奖励函数。研究在由NVIDIA Isaac Sim创建的仿真环境中进行，设置包括Franka Emika Panda机器人手臂持杯倾倒液体。强化学习算法基于Proximal Policy Optimization。我们系统地探索了奖励函数中变异权重的不同配置如何影响所学习的策略。生成的策略表现出广泛的行为：从执行原始倾倒任务的变化到对未预见任务有用的新型技能，如容器边缘清洁、液体混合和浇水。该方法为机器人系统进行特定任务的多样化学习提供了有 promise 的方向，同时也可能为未来任务衍生出有意义的技能。 

---
# A Counterfactual Reasoning Framework for Fault Diagnosis in Robot Perception Systems 

**Title (ZH)**: 基于机器人感知系统故障诊断的反事实推理框架 

**Authors**: Haeyoon Han, Mahdi Taheri, Soon-Jo Chung, Fred Y. Hadaegh  

**Link**: [PDF](https://arxiv.org/pdf/2509.18460)  

**Abstract**: Perception systems provide a rich understanding of the environment for autonomous systems, shaping decisions in all downstream modules. Hence, accurate detection and isolation of faults in perception systems is important. Faults in perception systems pose particular challenges: faults are often tied to the perceptual context of the environment, and errors in their multi-stage pipelines can propagate across modules. To address this, we adopt a counterfactual reasoning approach to propose a framework for fault detection and isolation (FDI) in perception systems. As opposed to relying on physical redundancy (i.e., having extra sensors), our approach utilizes analytical redundancy with counterfactual reasoning to construct perception reliability tests as causal outcomes influenced by system states and fault scenarios. Counterfactual reasoning generates reliability test results under hypothesized faults to update the belief over fault hypotheses. We derive both passive and active FDI methods. While the passive FDI can be achieved by belief updates, the active FDI approach is defined as a causal bandit problem, where we utilize Monte Carlo Tree Search (MCTS) with upper confidence bound (UCB) to find control inputs that maximize a detection and isolation metric, designated as Effective Information (EI). The mentioned metric quantifies the informativeness of control inputs for FDI. We demonstrate the approach in a robot exploration scenario, where a space robot performing vision-based navigation actively adjusts its attitude to increase EI and correctly isolate faults caused by sensor damage, dynamic scenes, and perceptual degradation. 

**Abstract (ZH)**: 感知系统提供了对环境的丰富理解，为自主系统提供了决策基础。因此，感知系统中故障的准确检测与隔离至关重要。感知系统中的故障具有特殊挑战：故障通常与环境的感知上下文相关联，其多阶段管道中的错误可以跨模块传播。为此，我们采用反事实推理方法，提出了一种感知系统故障检测与隔离（FDI）框架。与依赖物理冗余（即额外传感器）不同，我们的方法利用基于反事实推理的分析冗余来构建感知可靠性测试，并将其作为受系统状态和故障场景影响的因果结果。反事实推理生成在假设故障下的可靠性测试结果，以更新对故障假设的信念。我们推导了被动和主动的FDI方法。被动FDI可通过信念更新实现，而主动FDI则定义为因果多臂老虎机问题，在此问题中，我们利用蒙特卡洛树搜索（MCTS）与上置信界（UCB）来寻找最大化检测与隔离指标（有效信息EI）的控制输入。该指标量化了控制输入对故障检测与隔离的信息量。我们通过一个机器人探索场景展示了该方法，其中空间机器人在进行基于视觉的导航时主动调整姿态以增加有效信息EI，并正确隔离由传感器损坏、动态场景和感知退化引起的故障。 

---
# Learning Geometry-Aware Nonprehensile Pushing and Pulling with Dexterous Hands 

**Title (ZH)**: 学习几何感知非捡拾推拉与灵巧手 

**Authors**: Yunshuang Li, Yiyang Ling, Gaurav S. Sukhatme, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2509.18455)  

**Abstract**: Nonprehensile manipulation, such as pushing and pulling, enables robots to move, align, or reposition objects that may be difficult to grasp due to their geometry, size, or relationship to the robot or the environment. Much of the existing work in nonprehensile manipulation relies on parallel-jaw grippers or tools such as rods and spatulas. In contrast, multi-fingered dexterous hands offer richer contact modes and versatility for handling diverse objects to provide stable support over the objects, which compensates for the difficulty of modeling the dynamics of nonprehensile manipulation. Therefore, we propose Geometry-aware Dexterous Pushing and Pulling (GD2P) for nonprehensile manipulation with dexterous robotic hands. We study pushing and pulling by framing the problem as synthesizing and learning pre-contact dexterous hand poses that lead to effective manipulation. We generate diverse hand poses via contact-guided sampling, filter them using physics simulation, and train a diffusion model conditioned on object geometry to predict viable poses. At test time, we sample hand poses and use standard motion planners to select and execute pushing and pulling actions. We perform 840 real-world experiments with an Allegro Hand, comparing our method to baselines. The results indicate that GD2P offers a scalable route for training dexterous nonprehensile manipulation policies. We further demonstrate GD2P on a LEAP Hand, highlighting its applicability to different hand morphologies. Our pre-trained models and dataset, including 1.3 million hand poses across 2.3k objects, will be open-source to facilitate further research. Our project website is available at: this http URL. 

**Abstract (ZH)**: 几何感知灵巧推拉（Geometry-aware Dexterous Pushing and Pulling, GD2P）：基于灵巧机器人手的非接触操作 

---
# PrioriTouch: Adapting to User Contact Preferences for Whole-Arm Physical Human-Robot Interaction 

**Title (ZH)**: PrioriTouch: 根据用户接触偏好适应的全身物理人机交互 

**Authors**: Rishabh Madan, Jiawei Lin, Mahika Goel, Angchen Xie, Xiaoyu Liang, Marcus Lee, Justin Guo, Pranav N. Thakkar, Rohan Banerjee, Jose Barreiros, Kate Tsui, Tom Silver, Tapomayukh Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.18447)  

**Abstract**: Physical human-robot interaction (pHRI) requires robots to adapt to individual contact preferences, such as where and how much force is applied. Identifying preferences is difficult for a single contact; with whole-arm interaction involving multiple simultaneous contacts between the robot and human, the challenge is greater because different body parts can impose incompatible force requirements. In caregiving tasks, where contact is frequent and varied, such conflicts are unavoidable. With multiple preferences across multiple contacts, no single solution can satisfy all objectives--trade-offs are inherent, making prioritization essential. We present PrioriTouch, a framework for ranking and executing control objectives across multiple contacts. PrioriTouch can prioritize from a general collection of controllers, making it applicable not only to caregiving scenarios such as bed bathing and dressing but also to broader multi-contact settings. Our method combines a novel learning-to-rank approach with hierarchical operational space control, leveraging simulation-in-the-loop rollouts for data-efficient and safe exploration. We conduct a user study on physical assistance preferences, derive personalized comfort thresholds, and incorporate them into PrioriTouch. We evaluate PrioriTouch through extensive simulation and real-world experiments, demonstrating its ability to adapt to user contact preferences, maintain task performance, and enhance safety and comfort. Website: this https URL. 

**Abstract (ZH)**: 物理人机交互(pHRI)需要机器人适应个体的接触偏好，如接触的位置和力度。识别这些偏好对单一接触来说已是难题；而在涉及整个手臂的多点同时接触互动中，这一挑战变得更加复杂，因为不同身体部位可能会产生相互冲突的力要求。在照顾患者等接触频繁且多变的任务中，这种冲突是不可避免的。在存在多个接触点且每个接触点都有不同偏好时，不存在单一解决方案能满足所有目标——权衡是必然的，因此优先级设定至关重要。我们提出PrioriTouch框架，用于在多个接触点上对控制目标进行排名和执行。PrioriTouch可以从一个通用的控制器集合中进行优先级设定，使其不仅适用于床 bathing和穿衣等照护场景，还适用于更广泛的多点接触设置。我们的方法结合了一种新颖的排序学习方法与层次操作空间控制，并利用循环仿真进行高效且安全的数据探索。我们在物理辅助偏好用户研究中确定个性化舒适阈值，并将其纳入PrioriTouch。我们通过广泛的仿真和实际实验评估PrioriTouch，证明了其适应用户接触偏好、维持任务性能、增强安全性和舒适性的能力。网站：this https URL。 

---
# Latent Action Pretraining Through World Modeling 

**Title (ZH)**: 世界建模导向的潜动作预训练 

**Authors**: Bahey Tharwat, Yara Nasser, Ali Abouzeid, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.18428)  

**Abstract**: Vision-Language-Action (VLA) models have gained popularity for learning robotic manipulation tasks that follow language instructions. State-of-the-art VLAs, such as OpenVLA and $\pi_{0}$, were trained on large-scale, manually labeled action datasets collected through teleoperation. More recent approaches, including LAPA and villa-X, introduce latent action representations that enable unsupervised pretraining on unlabeled datasets by modeling abstract visual changes between frames. Although these methods have shown strong results, their large model sizes make deployment in real-world settings challenging. In this work, we propose LAWM, a model-agnostic framework to pretrain imitation learning models in a self-supervised way, by learning latent action representations from unlabeled video data through world modeling. These videos can be sourced from robot recordings or videos of humans performing actions with everyday objects. Our framework is designed to be effective for transferring across tasks, environments, and embodiments. It outperforms models trained with ground-truth robotics actions and similar pretraining methods on the LIBERO benchmark and real-world setup, while being significantly more efficient and practical for real-world settings. 

**Abstract (ZH)**: 一种模型无关的框架：通过世界建模从未标记视频数据中学习潜在动作表示以进行自监督预训练（LAWM） 

---
# Assistive Decision-Making for Right of Way Navigation at Uncontrolled Intersections 

**Title (ZH)**: 辅助决策在未控制交叉口通行优先导航中的应用 

**Authors**: Navya Tiwari, Joseph Vazhaeparampil, Victoria Preston  

**Link**: [PDF](https://arxiv.org/pdf/2509.18407)  

**Abstract**: Uncontrolled intersections account for a significant fraction of roadway crashes due to ambiguous right-of-way rules, occlusions, and unpredictable driver behavior. While autonomous vehicle research has explored uncertainty-aware decision making, few systems exist to retrofit human-operated vehicles with assistive navigation support. We present a driver-assist framework for right-of-way reasoning at uncontrolled intersections, formulated as a Partially Observable Markov Decision Process (POMDP). Using a custom simulation testbed with stochastic traffic agents, pedestrians, occlusions, and adversarial scenarios, we evaluate four decision-making approaches: a deterministic finite state machine (FSM), and three probabilistic planners: QMDP, POMCP, and DESPOT. Results show that probabilistic planners outperform the rule-based baseline, achieving up to 97.5 percent collision-free navigation under partial observability, with POMCP prioritizing safety and DESPOT balancing efficiency and runtime feasibility. Our findings highlight the importance of uncertainty-aware planning for driver assistance and motivate future integration of sensor fusion and environment perception modules for real-time deployment in realistic traffic environments. 

**Abstract (ZH)**: 不受控制的交叉口由于模糊的优先通行规则、遮挡和不可预测的驾驶行为，占了相当大的道路事故比例。尽管自动驾驶车辆研究探索了不确定性aware决策制定，却鲜有系统能够为人为操作的车辆提供辅助导航支持。我们提出了一种用于不受控制交叉口优先通行权推理的驾驶员辅助框架，该框架被形式化为部分可观测马尔可夫决策过程（POMDP）。使用包含随机交通代理、行人的自定义仿真测试床以及对抗性场景，我们评估了四种决策方法：确定性有限状态机（FSM），以及三种概率性规划器：QMDP、POMCP和DESPOT。结果表明，概率性规划器优于基于规则的基础方法，在部分可观测情况下实现了高达97.5%的无碰撞导航，其中POMCP侧重于安全，DESPOT则在效率和运行时可行性之间取得平衡。我们的研究结果强调了不确定性aware规划对于驾驶员辅助的重要性，并促进了将传感器融合和环境感知模块在未来实时部署到真实交通环境中的发展。 

---
# AD-VF: LLM-Automatic Differentiation Enables Fine-Tuning-Free Robot Planning from Formal Methods Feedback 

**Title (ZH)**: AD-VF: LLM-自动微分使通过形式方法反馈实现无需微调的机器人规划成为可能 

**Authors**: Yunhao Yang, Junyuan Hong, Gabriel Jacob Perin, Zhiwen Fan, Li Yin, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18384)  

**Abstract**: Large language models (LLMs) can translate natural language instructions into executable action plans for robotics, autonomous driving, and other domains. Yet, deploying LLM-driven planning in the physical world demands strict adherence to safety and regulatory constraints, which current models often violate due to hallucination or weak alignment. Traditional data-driven alignment methods, such as Direct Preference Optimization (DPO), require costly human labeling, while recent formal-feedback approaches still depend on resource-intensive fine-tuning. In this paper, we propose LAD-VF, a fine-tuning-free framework that leverages formal verification feedback for automated prompt engineering. By introducing a formal-verification-informed text loss integrated with LLM-AutoDiff, LAD-VF iteratively refines prompts rather than model parameters. This yields three key benefits: (i) scalable adaptation without fine-tuning; (ii) compatibility with modular LLM architectures; and (iii) interpretable refinement via auditable prompts. Experiments in robot navigation and manipulation tasks demonstrate that LAD-VF substantially enhances specification compliance, improving success rates from 60% to over 90%. Our method thus presents a scalable and interpretable pathway toward trustworthy, formally-verified LLM-driven control systems. 

**Abstract (ZH)**: 大型语言模型可以通过将自然语言指令转换为可执行的动作计划来应用于机器人学、自主驾驶和其他领域。然而，在物理世界中部署由大型语言模型驱动的规划方案需要严格遵守安全和监管约束，当前的模型经常因幻觉或对齐不足而违反这些约束。传统的数据驱动对齐方法，如直接偏好优化（DPO），需要昂贵的人工标注，而近期的形式反馈方法仍然依赖于资源密集型微调。在本文中，我们提出了一种名为LAD-VF的无需微调框架，该框架利用形式验证反馈进行自动提示工程。通过结合LLM-AutoDiff并引入形式验证指导的文本损失，LAD-VF逐步优化提示而非模型参数。这带来了三个关键优势：(i) 不需要微调的可扩展适应；(ii) 兼容模块化的大规模语言模型架构；以及(iii) 通过可审计的提示进行可解释的优化。在机器人导航和操作任务中的实验表明，LAD-VF显著提高了规范符合性，成功率从60%提高到超过90%。因此，我们的方法为可信赖的形式验证大型语言模型驱动控制系统提供了一条可扩展且可解释的路径。 

---
# Semantic-Aware Particle Filter for Reliable Vineyard Robot Localisation 

**Title (ZH)**: 面向语义的粒子滤波在可靠酿酒葡萄园机器人定位中的应用 

**Authors**: Rajitha de Silva, Jonathan Cox, James R. Heselden, Marija Popovic, Cesar Cadena, Riccardo Polvara  

**Link**: [PDF](https://arxiv.org/pdf/2509.18342)  

**Abstract**: Accurate localisation is critical for mobile robots in structured outdoor environments, yet LiDAR-based methods often fail in vineyards due to repetitive row geometry and perceptual aliasing. We propose a semantic particle filter that incorporates stable object-level detections, specifically vine trunks and support poles into the likelihood estimation process. Detected landmarks are projected into a birds eye view and fused with LiDAR scans to generate semantic observations. A key innovation is the use of semantic walls, which connect adjacent landmarks into pseudo-structural constraints that mitigate row aliasing. To maintain global consistency in headland regions where semantics are sparse, we introduce a noisy GPS prior that adaptively supports the filter. Experiments in a real vineyard demonstrate that our approach maintains localisation within the correct row, recovers from deviations where AMCL fails, and outperforms vision-based SLAM methods such as RTAB-Map. 

**Abstract (ZH)**: 基于语义的粒子滤波在结构化户外环境中正确定位葡萄园移动机器人，克服重复行几何和感知混叠的问题 

---
# The Landform Contextual Mesh: Automatically Fusing Surface and Orbital Terrain for Mars 2020 

**Title (ZH)**: 火星2020地形情境网格：自动融合表面和轨道地形 

**Authors**: Marsette Vona  

**Link**: [PDF](https://arxiv.org/pdf/2509.18330)  

**Abstract**: The Landform contextual mesh fuses 2D and 3D data from up to thousands of Mars 2020 rover images, along with orbital elevation and color maps from Mars Reconnaissance Orbiter, into an interactive 3D terrain visualization. Contextual meshes are built automatically for each rover location during mission ground data system processing, and are made available to mission scientists for tactical and strategic planning in the Advanced Science Targeting Tool for Robotic Operations (ASTTRO). A subset of them are also deployed to the "Explore with Perseverance" public access website. 

**Abstract (ZH)**: 火星2020漫游者图像的地形情境化网格融合了来自数千张漫游者图像的2D和3D数据，以及来自火星侦察轨道器的轨道高程和色彩地图，并提供了交互式的3D地形可视化。在任务地面数据系统处理过程中，自动为每个漫游者位置构建情境化网格，并提供给任务科学家在机器人操作高级科学目标选择工具（ASTTRO）中进行战术和战略规划。其中一部分也被部署到“使用 perseverance 探索”的公众访问网站上。 

---
# Haptic Communication in Human-Human and Human-Robot Co-Manipulation 

**Title (ZH)**: 人类与人类及人类与机器人协同操作中的触觉通信 

**Authors**: Katherine H. Allen, Chris Rogers, Elaine S. Short  

**Link**: [PDF](https://arxiv.org/pdf/2509.18327)  

**Abstract**: When a human dyad jointly manipulates an object, they must communicate about their intended motion plans. Some of that collaboration is achieved through the motion of the manipulated object itself, which we call "haptic communication." In this work, we captured the motion of human-human dyads moving an object together with one participant leading a motion plan about which the follower is uninformed. We then captured the same human participants manipulating the same object with a robot collaborator. By tracking the motion of the shared object using a low-cost IMU, we can directly compare human-human shared manipulation to the motion of those same participants interacting with the robot. Intra-study and post-study questionnaires provided participant feedback on the collaborations, indicating that the human-human collaborations are significantly more fluent, and analysis of the IMU data indicates that it captures objective differences in the motion profiles of the conditions. The differences in objective and subjective measures of accuracy and fluency between the human-human and human-robot trials motivate future research into improving robot assistants for physical tasks by enabling them to send and receive anthropomorphic haptic signals. 

**Abstract (ZH)**: 当一个人类双人组共同操作物体时，他们必须沟通他们的运动计划。部分合作通过所操作物体本身的运动实现，我们称之为“触觉通信”。在本研究中，我们记录了一名参与者主导运动计划而另一名跟随者对此未知的人类双人组共同操作物体的运动。接着，我们让相同的参与者与机器合作者共同操作相同的物体。通过使用低成本IMU追踪共享物体的运动，我们可以直接比较人类双人组共享操作与参与者与机器人互动时物体运动之间的差异。研究中的问卷和后续问卷提供了参与者对合作的反馈，表明人类双人组的合作更为流畅，IMU数据的分析显示它捕捉到了条件间运动特征的客观差异。人类双人组和人类-机器人试次在准确性和流畅性方面的客观与主观差异激励未来研究以使机器人能够发送和接收类人触觉信号，从而改进其在物理任务中的辅助能力。 

---
# Fine-Tuning Robot Policies While Maintaining User Privacy 

**Title (ZH)**: 细调机器人策略以保持用户隐私 

**Authors**: Benjamin A. Christie, Sagar Parekh, Dylan P. Losey  

**Link**: [PDF](https://arxiv.org/pdf/2509.18311)  

**Abstract**: Recent works introduce general-purpose robot policies. These policies provide a strong prior over how robots should behave -- e.g., how a robot arm should manipulate food items. But in order for robots to match an individual person's needs, users typically fine-tune these generalized policies -- e.g., showing the robot arm how to make their own preferred dinners. Importantly, during the process of personalizing robots, end-users leak data about their preferences, habits, and styles (e.g., the foods they prefer to eat). Other agents can simply roll-out the fine-tuned policy and see these personally-trained behaviors. This leads to a fundamental challenge: how can we develop robots that personalize actions while keeping learning private from external agents? We here explore this emerging topic in human-robot interaction and develop PRoP, a model-agnostic framework for personalized and private robot policies. Our core idea is to equip each user with a unique key; this key is then used to mathematically transform the weights of the robot's network. With the correct key, the robot's policy switches to match that user's preferences -- but with incorrect keys, the robot reverts to its baseline behaviors. We show the general applicability of our method across multiple model types in imitation learning, reinforcement learning, and classification tasks. PRoP is practically advantageous because it retains the architecture and behaviors of the original policy, and experimentally outperforms existing encoder-based approaches. See videos and code here: this https URL. 

**Abstract (ZH)**: 近期的研究引入了通用机器人策略。这些策略为机器人应当如何行为提供了一个强大的先验——例如，如何操作食物。但为了使机器人能够满足个人需求，用户通常需要微调这些通用策略——例如，展示机器人如何制作自己偏好的晚餐。重要的是，在个性化机器人的过程中，最终用户会泄露其偏好、习惯和风格（例如，他们喜欢吃什么食物）的数据。其他代理可以简单地执行微调过的策略并观察这些个性化训练的行为。这一过程带来了根本性挑战：我们如何开发能够在不从外部代理处泄露学习的情况下个性化动作的机器人？我们在这里探讨了这一新兴的机器人交互主题，并开发了PRoP，这是一种模型无关的个性化和私有化机器人策略的框架。我们的核心思想是为每个用户提供一个独特的密钥；然后使用这个密钥对机器人网络的权重进行数学变换。通过正确的密钥，机器人的策略会切换以匹配该用户的需求——但通过错误的密钥，机器人会恢复其基础行为。我们在模仿学习、强化学习和分类任务等多种模型类型中展示了我们方法的普遍适用性。PRoP 实际上具有优势，因为它保留了原始策略的架构和行为，并且实验上优于现有基于编码器的方法。 

---
# PEEK: Guiding and Minimal Image Representations for Zero-Shot Generalization of Robot Manipulation Policies 

**Title (ZH)**: PEEK: 引导和 minimalist 图像表示用于机器人操作策略的零样本泛化 

**Authors**: Jesse Zhang, Marius Memmel, Kevin Kim, Dieter Fox, Jesse Thomason, Fabio Ramos, Erdem Bıyık, Abhishek Gupta, Anqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18282)  

**Abstract**: Robotic manipulation policies often fail to generalize because they must simultaneously learn where to attend, what actions to take, and how to execute them. We argue that high-level reasoning about where and what can be offloaded to vision-language models (VLMs), leaving policies to specialize in how to act. We present PEEK (Policy-agnostic Extraction of Essential Keypoints), which fine-tunes VLMs to predict a unified point-based intermediate representation: 1. end-effector paths specifying what actions to take, and 2. task-relevant masks indicating where to focus. These annotations are directly overlaid onto robot observations, making the representation policy-agnostic and transferable across architectures. To enable scalable training, we introduce an automatic annotation pipeline, generating labeled data across 20+ robot datasets spanning 9 embodiments. In real-world evaluations, PEEK consistently boosts zero-shot generalization, including a 41.4x real-world improvement for a 3D policy trained only in simulation, and 2-3.5x gains for both large VLAs and small manipulation policies. By letting VLMs absorb semantic and visual complexity, PEEK equips manipulation policies with the minimal cues they need--where, what, and how. Website at this https URL. 

**Abstract (ZH)**: 基于视觉语言模型的机器人操作策略细粒度注释方法：从哪里、做什么到如何做 

---
# A Fast Initialization Method for Neural Network Controllers: A Case Study of Image-based Visual Servoing Control for the multicopter Interception 

**Title (ZH)**: 基于图像视觉伺服控制的多旋翼拦截中神经网络控制的快速初始化方法：案例研究 

**Authors**: Chenxu Ke, Congling Tian, Kaichen Xu, Ye Li, Lingcong Bao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19110)  

**Abstract**: Reinforcement learning-based controller design methods often require substantial data in the initial training phase. Moreover, the training process tends to exhibit strong randomness and slow convergence. It often requires considerable time or high computational resources. Another class of learning-based method incorporates Lyapunov stability theory to obtain a control policy with stability guarantees. However, these methods generally require an initially stable neural network control policy at the beginning of training. Evidently, a stable neural network controller can not only serve as an initial policy for reinforcement learning, allowing the training to focus on improving controller performance, but also act as an initial state for learning-based Lyapunov control methods. Although stable controllers can be designed using traditional control theory, designers still need to have a great deal of control design knowledge to address increasingly complicated control problems. The proposed neural network rapid initialization method in this paper achieves the initial training of the neural network control policy by constructing datasets that conform to the stability conditions based on the system model. Furthermore, using the image-based visual servoing control for multicopter interception as a case study, simulations and experiments were conducted to validate the effectiveness and practical performance of the proposed method. In the experiment, the trained control policy attains a final interception velocity of 15 m/s. 

**Abstract (ZH)**: 基于强化学习的控制器设计方法常需要大量的初始训练数据，且训练过程往往表现出较强的随机性和缓慢的收敛性，需要消耗大量时间和计算资源。另一类基于学习的方法通过引入李雅普诺夫稳定性理论来获得具有稳定性的控制策略。然而，这些方法通常需要在训练之初具备一个稳定的神经网络控制策略。显然，一个稳定的神经网络控制器不仅可以用作强化学习的初始策略，使训练专注于提高控制器性能，还可以作为基于学习的李雅普诺夫控制方法的初始状态。尽管传统的控制理论可以设计出稳定的控制器，但设计者仍需具备大量的控制设计知识以应对日益复杂的控制问题。本文提出的神经网络快速初始化方法通过基于系统模型构建满足稳定性条件的数据集，实现了神经网络控制策略的初始训练。此外，以多旋翼拦截基于图像的视觉伺服控制为例，进行了仿真和实验来验证该方法的有效性和实际性能。在实验中，训练得到的控制策略获得最终拦截速度为15 m/s。 

---
# Guaranteed Robust Nonlinear MPC via Disturbance Feedback 

**Title (ZH)**: 通过干扰反馈确保的鲁棒非线性MPC 

**Authors**: Antoine P. Leeman, Johannes Köhler, Melanie N. Zeilinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.18760)  

**Abstract**: Robots must satisfy safety-critical state and input constraints despite disturbances and model mismatch. We introduce a robust model predictive control (RMPC) formulation that is fast, scalable, and compatible with real-time implementation. Our formulation guarantees robust constraint satisfaction, input-to-state stability (ISS) and recursive feasibility. The key idea is to decompose the uncertain nonlinear system into (i) a nominal nonlinear dynamic model, (ii) disturbance-feedback controllers, and (iii) bounds on the model error. These components are optimized jointly using sequential convex programming. The resulting convex subproblems are solved efficiently using a recent disturbance-feedback MPC solver. The approach is validated across multiple dynamics, including a rocket-landing problem with steerable thrust. An open-source implementation is available at this https URL. 

**Abstract (ZH)**: 机器人必须在干扰和模型不匹配的情况下满足安全关键的状态和输入约束。我们介绍了一种快速、可扩展且适用于实时实现的鲁棒模型预测控制（RMPC） formulation。该 formulation 保证了鲁棒的约束满足、输入状态稳定性（ISS）和递归可行性。关键思想是将不确定的非线性系统分解为（i）名义非线性动态模型，（ii）干扰反馈控制器，和（iii）模型误差的界。这些组成部分是通过序列凸规划联合优化的。由此产生的凸子问题通过最近的干扰反馈 MPC 解算器高效求解。该方法跨多种动力学进行了验证，包括具有可转向推力的火箭着陆问题。开源实现可在以下链接获得：this https URL。 

---
# An Extended Kalman Filter for Systems with Infinite-Dimensional Measurements 

**Title (ZH)**: 适用于无限维观测系统的扩展卡尔曼滤波器 

**Authors**: Maxwell M. Varley, Timothy L. Molloy, Girish N. Nair  

**Link**: [PDF](https://arxiv.org/pdf/2509.18749)  

**Abstract**: This article examines state estimation in discrete-time nonlinear stochastic systems with finite-dimensional states and infinite-dimensional measurements, motivated by real-world applications such as vision-based localization and tracking. We develop an extended Kalman filter (EKF) for real-time state estimation, with the measurement noise modeled as an infinite-dimensional random field. When applied to vision-based state estimation, the measurement Jacobians required to implement the EKF are shown to correspond to image gradients. This result provides a novel system-theoretic justification for the use of image gradients as features for vision-based state estimation, contrasting with their (often heuristic) introduction in many computer-vision pipelines. We demonstrate the practical utility of the EKF on a public real-world dataset involving the localization of an aerial drone using video from a downward-facing monocular camera. The EKF is shown to outperform VINS-MONO, an established visual-inertial odometry algorithm, in some cases achieving mean squared error reductions of up to an order of magnitude. 

**Abstract (ZH)**: 本文探讨了在状态有限维、测量无限维的离散时间非线性随机系统中状态估计的问题，受到了基于视觉的定位与跟踪等实际应用的启发。我们开发了一种扩展卡尔曼滤波器（EKF）进行实时状态估计，并将测量噪声建模为无限维随机场。在应用于基于视觉的状态估计时，实现EKF所需的测量雅可比矩阵对应于图像梯度。这一结果为使用图像梯度作为基于视觉的状态估计特征提供了新颖的系统理论依据，区别于其在许多计算机视觉管道中的（通常是启发式的）引入方式。我们利用一个公开的真实世界数据集，展示了EKF在使用向下视角单目摄像头视频进行空中无人机定位上的实用价值。在某些情况下，EKF相较于成熟的视觉惯性里程计算法VINS-MONO显示出均方误差降低一个数量级以上的性能提升。 

---
# Dual Iterative Learning Control for Multiple-Input Multiple-Output Dynamics with Validation in Robotic Systems 

**Title (ZH)**: 基于验证的多输入多输出动力学的双迭代学习控制 

**Authors**: Jan-Hendrik Ewering, Alessandro Papa, Simon F.G. Ehlers, Thomas Seel, Michael Meindl  

**Link**: [PDF](https://arxiv.org/pdf/2509.18723)  

**Abstract**: Solving motion tasks autonomously and accurately is a core ability for intelligent real-world systems. To achieve genuine autonomy across multiple systems and tasks, key challenges include coping with unknown dynamics and overcoming the need for manual parameter tuning, which is especially crucial in complex Multiple-Input Multiple-Output (MIMO) systems.
This paper presents MIMO Dual Iterative Learning Control (DILC), a novel data-driven iterative learning scheme for simultaneous tracking control and model learning, without requiring any prior system knowledge or manual parameter tuning. The method is designed for repetitive MIMO systems and integrates seamlessly with established iterative learning control methods. We provide monotonic convergence conditions for both reference tracking error and model error in linear time-invariant systems.
The DILC scheme -- rapidly and autonomously -- solves various motion tasks in high-fidelity simulations of an industrial robot and in multiple nonlinear real-world MIMO systems, without requiring model knowledge or manually tuning the algorithm. In our experiments, many reference tracking tasks are solved within 10-20 trials, and even complex motions are learned in less than 100 iterations. We believe that, because of its rapid and autonomous learning capabilities, DILC has the potential to serve as an efficient building block within complex learning frameworks for intelligent real-world systems. 

**Abstract (ZH)**: 自主准确地解决运动任务是智能现实系统的一项核心能力。为了在多个系统和任务中实现真正的自主性，关键挑战包括应对未知动态和克服手动参数调谐的需要，尤其是在复杂的多输入多输出（MIMO）系统中。

本文提出了一种新颖的数据驱动迭代学习控制方案MIMO双迭代学习控制（DILC），该方案无需任何先验系统知识或手动参数调谐，同时实现了跟踪控制和模型学习。该方法适用于重复的MIMO系统，并可无缝集成到现有的迭代学习控制方法中。我们为线性时不变系统提供了参考跟踪误差和模型误差的单调收敛条件。

DILC方案在高保真工业机器人模拟和多个非线性实际MIMO系统中快速自主地解决各种运动任务，无需模型知识或手动调谐算法。在我们的实验中，许多参考跟踪任务在10-20次迭代内得到解决，甚至复杂的运动在不到100次迭代内也被学习。我们相信，由于其快速自主的学习能力，DILC有望成为用于智能现实系统复杂学习框架的有效构建块。 

---
# Event-guided 3D Gaussian Splatting for Dynamic Human and Scene Reconstruction 

**Title (ZH)**: 事件引导的3D高斯点云方法用于动态人类和场景重建 

**Authors**: Xiaoting Yin, Hao Shi, Kailun Yang, Jiajun Zhai, Shangwei Guo, Lin Wang, Kaiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18566)  

**Abstract**: Reconstructing dynamic humans together with static scenes from monocular videos remains difficult, especially under fast motion, where RGB frames suffer from motion blur. Event cameras exhibit distinct advantages, e.g., microsecond temporal resolution, making them a superior sensing choice for dynamic human reconstruction. Accordingly, we present a novel event-guided human-scene reconstruction framework that jointly models human and scene from a single monocular event camera via 3D Gaussian Splatting. Specifically, a unified set of 3D Gaussians carries a learnable semantic attribute; only Gaussians classified as human undergo deformation for animation, while scene Gaussians stay static. To combat blur, we propose an event-guided loss that matches simulated brightness changes between consecutive renderings with the event stream, improving local fidelity in fast-moving regions. Our approach removes the need for external human masks and simplifies managing separate Gaussian sets. On two benchmark datasets, ZJU-MoCap-Blur and MMHPSD-Blur, it delivers state-of-the-art human-scene reconstruction, with notable gains over strong baselines in PSNR/SSIM and reduced LPIPS, especially for high-speed subjects. 

**Abstract (ZH)**: 从单目事件摄像头视频重建动态人体与静态场景依然具有挑战性，尤其是在快速运动下，RGB帧受运动模糊影响。事件摄像头表现出显著的优势，如微秒级的时间分辨率，使其成为动态人体重建的优越传感器选择。因此，我们提出了一种新型的事件引导的人景重建框架，通过3D高斯点积技术，从单目事件摄像头联合建模人体和场景。具体而言，统一的3D高斯集合携带可学习的语义属性；只有被分类为人体的高斯点进行变形以实现动画效果，而场景高斯点保持静态。为对抗模糊，我们提出了一种事件引导的损失，该损失匹配连续渲染中的模拟亮度变化与事件流，从而在快速运动区域提升局部保真度。该方法消除了对外部人体掩码的需求，并简化了不同高斯集合的管理。在两个基准数据集ZJU-MoCap-Blur和MMHPSD-Blur上，它实现了人体-场景重建的最先进成果，特别是在PSNR/SSIM和降低LPIPS方面，特别是在高速运动主体方面取得了显著提升。 

---
# Policy Gradient with Self-Attention for Model-Free Distributed Nonlinear Multi-Agent Games 

**Title (ZH)**: 基于自注意力的策略梯度在无模型分布式非线性多智能体博弈中的应用 

**Authors**: Eduardo Sebastián, Maitrayee Keskar, Eeman Iqbal, Eduardo Montijano, Carlos Sagüés, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2509.18371)  

**Abstract**: Multi-agent games in dynamic nonlinear settings are challenging due to the time-varying interactions among the agents and the non-stationarity of the (potential) Nash equilibria. In this paper we consider model-free games, where agent transitions and costs are observed without knowledge of the transition and cost functions that generate them. We propose a policy gradient approach to learn distributed policies that follow the communication structure in multi-team games, with multiple agents per team. Our formulation is inspired by the structure of distributed policies in linear quadratic games, which take the form of time-varying linear feedback gains. In the nonlinear case, we model the policies as nonlinear feedback gains, parameterized by self-attention layers to account for the time-varying multi-agent communication topology. We demonstrate that our distributed policy gradient approach achieves strong performance in several settings, including distributed linear and nonlinear regulation, and simulated and real multi-robot pursuit-and-evasion games. 

**Abstract (ZH)**: 动态非线性环境中的多智能体博弈因智能体间的时间变化交互和纳什均衡的非稳态性而具有挑战性。在本文中，我们考虑无模型的博弈，其中智能体的状态转移和成本可以被观察到，但不了解产生这些转移和成本的功能形式。我们提出了一种策略梯度方法，用于学习遵循多队列博弈中通信结构的分布式策略，每队包含多个智能体。我们的框架灵感来自于线性二次博弈中分布式策略的结构，它们的形式为时间变化的线性反馈增益。在非线性情况下，我们将策略建模为非线性反馈增益，并通过自注意力层参数化以考虑时间变化的多智能体通信拓扑。我们展示了我们的分布式策略梯度方法在分布式线性和非线性调节以及模拟和实际多机器人追逃游戏中均表现出色。 

---
# OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata 

**Title (ZH)**: 正射LoC：使用正射地理数据的UAV 6-DoF定位与校准 

**Authors**: Oussema Dhaouadi, Riccardo Marin, Johannes Meier, Jacques Kaiser, Daniel Cremers  

**Link**: [PDF](https://arxiv.org/pdf/2509.18350)  

**Abstract**: Accurate visual localization from aerial views is a fundamental problem with applications in mapping, large-area inspection, and search-and-rescue operations. In many scenarios, these systems require high-precision localization while operating with limited resources (e.g., no internet connection or GNSS/GPS support), making large image databases or heavy 3D models impractical. Surprisingly, little attention has been given to leveraging orthographic geodata as an alternative paradigm, which is lightweight and increasingly available through free releases by governmental authorities (e.g., the European Union). To fill this gap, we propose OrthoLoC, the first large-scale dataset comprising 16,425 UAV images from Germany and the United States with multiple modalities. The dataset addresses domain shifts between UAV imagery and geospatial data. Its paired structure enables fair benchmarking of existing solutions by decoupling image retrieval from feature matching, allowing isolated evaluation of localization and calibration performance. Through comprehensive evaluation, we examine the impact of domain shifts, data resolutions, and covisibility on localization accuracy. Finally, we introduce a refinement technique called AdHoP, which can be integrated with any feature matcher, improving matching by up to 95% and reducing translation error by up to 63%. The dataset and code are available at: this https URL. 

**Abstract (ZH)**: 从空中视角实现准确的视觉定位是一个基本问题，应用于制图、大面积检查和搜索救援操作。在许多场景中，这些系统在资源有限的情况下（例如，没有互联网连接或GNSS/GPS支持）需要高精度定位，使得大规模图像数据库或重大的3D模型不切实际。令人惊讶的是，很少有研究关注利用正射地理数据作为替代方案，该方案轻量级且随着政府机构（例如欧盟）的免费发布越来越可用。为填补这一空白，我们提出OrthoLoC，这是首个包含来自德国和美国的16,425张UAV图像的大规模数据集，具有多种模态。数据集解决了UAV图像与地理空间数据之间的域转移问题。其成对结构使现有的解决方案能够在解耦图像检索和特征匹配的情况下进行公平基准测试，允许独立评估定位和校准性能。通过全面评估，我们探讨了域转移、数据分辨率和共可见性对定位精度的影响。最后，我们介绍了一种称为AdHoP的精简技术，它可以与任何特征匹配器集成，最多可提高匹配精度95%，并减少平移误差63%。该数据集和代码可在以下链接获取：this https URL。 

---
# Reversible Kalman Filter for state estimation with Manifold 

**Title (ZH)**: 流形上状态估计的可逆卡尔曼滤波器 

**Authors**: Svyatoslav Covanov, Cedric Pradalier  

**Link**: [PDF](https://arxiv.org/pdf/2509.18224)  

**Abstract**: This work introduces an algorithm for state estimation on manifolds within the framework of the Kalman filter. Its primary objective is to provide a methodology enabling the evaluation of the precision of existing Kalman filter variants with arbitrary accuracy on synthetic data, something that, to the best of our knowledge, has not been addressed in prior work. To this end, we develop a new filter that exhibits favorable numerical properties, thereby correcting the divergences observed in previous Kalman filter variants. In this formulation, the achievable precision is no longer constrained by the small-velocity assumption and is determined solely by sensor noise. In addition, this new filter assumes high precision on the sensors, which, in real scenarios require a detection step that we define heuristically, allowing one to extend this approach to scenarios, using either a 9-axis IMU or a combination of odometry, accelerometer, and pressure sensors. The latter configuration is designed for the reconstruction of trajectories in underwater environments. 

**Abstract (ZH)**: 本文介绍了一种在流形框架下Kalman滤波器的态估计算法。其主要目标是在合成数据上以任意精度评估现有Kalman滤波器变体的精度，这是迄今为止Prior工作尚未解决的问题。为此，我们开发了一种新滤波器，具有良好的数值性质，从而纠正了先前Kalman滤波器变体中的发散问题。在此框架下，可实现的精度不再受小速度假设的限制，而是仅由传感器噪声决定。此外，该新滤波器假设传感器具有高精度，在实际场景中需要进行一个我们以启发式方法定义的检测步骤，从而可以将该方法扩展到使用9轴IMU或组合使用里程计、加速度计和压力传感器的场景中，后者用于水下环境中的轨迹重构。 

---
# Conversational Orientation Reasoning: Egocentric-to-Allocentric Navigation with Multimodal Chain-of-Thought 

**Title (ZH)**: 基于对话导向的推理：多模态链式思考的以自我为中心到以环境为中心的导航 

**Authors**: Yu Ti Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18200)  

**Abstract**: Conversational agents must translate egocentric utterances (e.g., "on my right") into allocentric orientations (N/E/S/W). This challenge is particularly critical in indoor or complex facilities where GPS signals are weak and detailed maps are unavailable. While chain-of-thought (CoT) prompting has advanced reasoning in language and vision tasks, its application to multimodal spatial orientation remains underexplored. We introduce Conversational Orientation Reasoning (COR), a new benchmark designed for Traditional Chinese conversational navigation projected from real-world environments, addressing egocentric-to-allocentric reasoning in non-English and ASR-transcribed scenarios. We propose a multimodal chain-of-thought (MCoT) framework, which integrates ASR-transcribed speech with landmark coordinates through a structured three-step reasoning process: (1) extracting spatial relations, (2) mapping coordinates to absolute directions, and (3) inferring user orientation. A curriculum learning strategy progressively builds these capabilities on Taiwan-LLM-13B-v2.0-Chat, a mid-sized model representative of resource-constrained settings. Experiments show that MCoT achieves 100% orientation accuracy on clean transcripts and 98.1% with ASR transcripts, substantially outperforming unimodal and non-structured baselines. Moreover, MCoT demonstrates robustness under noisy conversational conditions, including ASR recognition errors and multilingual code-switching. The model also maintains high accuracy in cross-domain evaluation and resilience to linguistic variation, domain shift, and referential ambiguity. These findings highlight the potential of structured MCoT spatial reasoning as a path toward interpretable and resource-efficient embodied navigation. 

**Abstract (ZH)**: Conversational代理必须将主观表述（例如，“在我的右边”）转换为客观方向（N/E/S/W）。这一挑战在GPS信号弱且详细地图不可用的室内或复杂设施中尤为关键。虽然链式思维（CoT）提示在语言和视觉任务中提升了推理能力，但其在多模态空间定向中的应用仍然鲜有探索。我们引入了对话方向推理（COR），这是一种针对实际环境中的传统汉语对话导航的新基准，旨在解决非英语和语音识别（ASR）转录场景中的主观到客观方向推理问题。我们提出了一种多模态链式思维（MCoT）框架，该框架通过结构化的三步推理过程，将语音识别转录的语音与地标坐标结合：（1）提取空间关系，（2）将坐标映射为绝对方向，（3）推断用户方向。一种课程学习策略逐步在资源受限环境中代表性模型Taiwan-LLM-13B-v2.0-Chat上构建这些能力。实验结果显示，MCoT在干净转录文本上的方向准确率为100%，在ASR转录文本上的方向准确率为98.1%，显著优于单模态和非结构化基线。此外，MCoT在包括ASR识别错误和多语言转换在内的嘈杂对话条件下表现出了鲁棒性，并且在跨域评估中保持了高准确率，对语言变化、领域转移和指代模糊具有较强的抗性。这些发现突显了结构化MCoT空间推理在可解释和资源高效的体态导航中的潜力。 

---
# MMCD: Multi-Modal Collaborative Decision-Making for Connected Autonomy with Knowledge Distillation 

**Title (ZH)**: MMCD：面向连接自治的多模态协作决策知识蒸馏 

**Authors**: Rui Liu, Zikang Wang, Peng Gao, Yu Shen, Pratap Tokekar, Ming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.18198)  

**Abstract**: Autonomous systems have advanced significantly, but challenges persist in accident-prone environments where robust decision-making is crucial. A single vehicle's limited sensor range and obstructed views increase the likelihood of accidents. Multi-vehicle connected systems and multi-modal approaches, leveraging RGB images and LiDAR point clouds, have emerged as promising solutions. However, existing methods often assume the availability of all data modalities and connected vehicles during both training and testing, which is impractical due to potential sensor failures or missing connected vehicles. To address these challenges, we introduce a novel framework MMCD (Multi-Modal Collaborative Decision-making) for connected autonomy. Our framework fuses multi-modal observations from ego and collaborative vehicles to enhance decision-making under challenging conditions. To ensure robust performance when certain data modalities are unavailable during testing, we propose an approach based on cross-modal knowledge distillation with a teacher-student model structure. The teacher model is trained with multiple data modalities, while the student model is designed to operate effectively with reduced modalities. In experiments on $\textit{connected autonomous driving with ground vehicles}$ and $\textit{aerial-ground vehicles collaboration}$, our method improves driving safety by up to ${\it 20.7}\%$, surpassing the best-existing baseline in detecting potential accidents and making safe driving decisions. More information can be found on our website this https URL. 

**Abstract (ZH)**: 自主系统取得了显著进展，但在事故高发环境中，稳健的决策制定仍然面临挑战。单辆车辆有限的传感器范围和受阻的视野增加了事故发生的风险。利用多辆连接车辆和多模态方法，结合RGB图像和LiDAR点云，多模态协同决策（MMCD）框架成为有前景的解决方案。然而，现有方法往往假设训练和测试过程中所有数据模态和连接车辆均可用，这由于传感器故障或连接车辆缺失的原因在实际中难以实现。为应对这些挑战，我们提出了一种名为MMCD的新型框架，该框架融合了自我和协作车辆的多模态观测，以在挑战性条件下增强决策能力。为确保在测试过程中某些数据模态不可用时仍能保持稳健性能，我们提出了基于跨模态知识蒸馏的教师-学生模型结构。教师模型在多种数据模态上进行训练，而学生模型则被设计成能够在减少模态的情况下有效运行。在基于地面车辆的连接自主驾驶和空地协同驾驶实验中，我们的方法将驾驶安全性提高至最高20.7%，在检测潜在事故和做出安全驾驶决策方面超越了现有最佳基线方法。更多信息请参见我们的网站：this https URL。 

---
