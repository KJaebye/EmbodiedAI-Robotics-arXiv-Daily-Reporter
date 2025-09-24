# SOE: Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration 

**Title (ZH)**: SOE: 基于流形探索的样本高效机器人策略自我改进 

**Authors**: Yang Jin, Jun Lv, Han Xue, Wendi Chen, Chuan Wen, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.19292)  

**Abstract**: Intelligent agents progress by continually refining their capabilities through actively exploring environments. Yet robot policies often lack sufficient exploration capability due to action mode collapse. Existing methods that encourage exploration typically rely on random perturbations, which are unsafe and induce unstable, erratic behaviors, thereby limiting their effectiveness. We propose Self-Improvement via On-Manifold Exploration (SOE), a framework that enhances policy exploration and improvement in robotic manipulation. SOE learns a compact latent representation of task-relevant factors and constrains exploration to the manifold of valid actions, ensuring safety, diversity, and effectiveness. It can be seamlessly integrated with arbitrary policy models as a plug-in module, augmenting exploration without degrading the base policy performance. Moreover, the structured latent space enables human-guided exploration, further improving efficiency and controllability. Extensive experiments in both simulation and real-world tasks demonstrate that SOE consistently outperforms prior methods, achieving higher task success rates, smoother and safer exploration, and superior sample efficiency. These results establish on-manifold exploration as a principled approach to sample-efficient policy self-improvement. Project website: this https URL 

**Abstract (ZH)**: 智能代理通过不断主动探索环境来不断提升其能力。然而，由于动作模式崩溃，机器人策略往往缺乏足够的探索能力。现有的鼓励探索的方法通常依赖于随机扰动，这会导致不安全的行为并引发不稳定、不可预测的效果，从而限制其有效性。我们提出了On-Manifold Exploration (OME) 自我提升框架，该框架增强了机器人操作中的策略探索和改进。OME 学习与任务相关的紧凑潜在表示，并将探索限制在有效的动作流形上，确保安全、多样性和有效性。它可以无缝集成到任意策略模型中作为插件模块，增强探索而不降低基础策略性能。此外，结构化的潜在空间使人类能够引导探索，进一步提高效率和可控性。广泛的模拟和实际任务实验表明，OME 一致优于先前方法，实现更高的任务成功率、更平滑和安全的探索以及更高的样本效率。这些结果将沿流形探索确立为样本高效策略自我改进的原则性方法。项目网站：此链接 

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
# FUNCanon: Learning Pose-Aware Action Primitives via Functional Object Canonicalization for Generalizable Robotic Manipulation 

**Title (ZH)**: FUNCanon: 基于功能对象标准化的学习姿态感知动作primitive及其在通用化机器人操作中的应用 

**Authors**: Hongli Xu, Lei Zhang, Xiaoyue Hu, Boyang Zhong, Kaixin Bai, Zoltán-Csaba Márton, Zhenshan Bing, Zhaopeng Chen, Alois Christian Knoll, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19102)  

**Abstract**: General-purpose robotic skills from end-to-end demonstrations often leads to task-specific policies that fail to generalize beyond the training distribution. Therefore, we introduce FunCanon, a framework that converts long-horizon manipulation tasks into sequences of action chunks, each defined by an actor, verb, and object. These chunks focus policy learning on the actions themselves, rather than isolated tasks, enabling compositionality and reuse. To make policies pose-aware and category-general, we perform functional object canonicalization for functional alignment and automatic manipulation trajectory transfer, mapping objects into shared functional frames using affordance cues from large vision language models. An object centric and action centric diffusion policy FuncDiffuser trained on this aligned data naturally respects object affordances and poses, simplifying learning and improving generalization ability. Experiments on simulated and real-world benchmarks demonstrate category-level generalization, cross-task behavior reuse, and robust sim2real deployment, showing that functional canonicalization provides a strong inductive bias for scalable imitation learning in complex manipulation domains. Details of the demo and supplemental material are available on our project website this https URL. 

**Abstract (ZH)**: 一般化机器人技能从端到端演示中获得的任务特定策略往往难以泛化到训练分布之外。因此，我们提出了FunCanon框架，将长时 horizon 操作任务转换为由执行者、动词和物体定义的行动片段序列。这些片段集中于行动本身的学习，而非孤立的任务，从而实现组合性和重用性。为了使策略具备姿态意识和类别普适性，我们进行了功能性对象标准变换，实现了功能对齐和自动操作轨迹迁移，使用大型视觉语言模型中的可利用性线索将物体映射到共享的功能框架中。以这种对齐的数据为中心的物体扩散策略和操作扩散策略FuncDiffuser自然地遵循对象的可利用性和姿态，简化了学习并提高了泛化能力。在模拟和现实世界基准上的实验展示了类别水平的泛化、跨任务行为的重用以及稳健的仿真实验部署，表明功能性标准变换为在复杂操作领域中可扩展的模仿学习提供了强大的归纳偏置。演示细节和补充材料详见我们的项目网站：<https://>。 

---
# World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation 

**Title (ZH)**: World4RL: 扩散世界模型在强化学习中用于机器人操作策略精炼 

**Authors**: Zhennan Jiang, Kai Liu, Yuxin Qin, Shuai Tian, Yupeng Zheng, Mingcai Zhou, Chao Yu, Haoran Li, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.19080)  

**Abstract**: Robotic manipulation policies are commonly initialized through imitation learning, but their performance is limited by the scarcity and narrow coverage of expert data. Reinforcement learning can refine polices to alleviate this limitation, yet real-robot training is costly and unsafe, while training in simulators suffers from the sim-to-real gap. Recent advances in generative models have demonstrated remarkable capabilities in real-world simulation, with diffusion models in particular excelling at generation. This raises the question of how diffusion model-based world models can be combined to enhance pre-trained policies in robotic manipulation. In this work, we propose World4RL, a framework that employs diffusion-based world models as high-fidelity simulators to refine pre-trained policies entirely in imagined environments for robotic manipulation. Unlike prior works that primarily employ world models for planning, our framework enables direct end-to-end policy optimization. World4RL is designed around two principles: pre-training a diffusion world model that captures diverse dynamics on multi-task datasets and refining policies entirely within a frozen world model to avoid online real-world interactions. We further design a two-hot action encoding scheme tailored for robotic manipulation and adopt diffusion backbones to improve modeling fidelity. Extensive simulation and real-world experiments demonstrate that World4RL provides high-fidelity environment modeling and enables consistent policy refinement, yielding significantly higher success rates compared to imitation learning and other baselines. More visualization results are available at this https URL. 

**Abstract (ZH)**: 基于扩散模型的世界模型在机器人操作中的强化学习框架 

---
# ManipForce: Force-Guided Policy Learning with Frequency-Aware Representation for Contact-Rich Manipulation 

**Title (ZH)**: ManipForce：基于频率感知表示的力引导政策学习在接触丰富的操纵中 

**Authors**: Geonhyup Lee, Yeongjin Lee, Kangmin Kim, Seongju Lee, Sangjun Noh, Seunghyeok Back, Kyoobin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.19047)  

**Abstract**: Contact-rich manipulation tasks such as precision assembly require precise control of interaction forces, yet existing imitation learning methods rely mainly on vision-only demonstrations. We propose ManipForce, a handheld system designed to capture high-frequency force-torque (F/T) and RGB data during natural human demonstrations for contact-rich manipulation. Building on these demonstrations, we introduce the Frequency-Aware Multimodal Transformer (FMT). FMT encodes asynchronous RGB and F/T signals using frequency- and modality-aware embeddings and fuses them via bi-directional cross-attention within a transformer diffusion policy. Through extensive experiments on six real-world contact-rich manipulation tasks - such as gear assembly, box flipping, and battery insertion - FMT trained on ManipForce demonstrations achieves robust performance with an average success rate of 83% across all tasks, substantially outperforming RGB-only baselines. Ablation and sampling-frequency analyses further confirm that incorporating high-frequency F/T data and cross-modal integration improves policy performance, especially in tasks demanding high precision and stable contact. 

**Abstract (ZH)**: 接触丰富的操作任务，如精确装配，要求精确控制相互作用力，现有模仿学习方法主要依赖于单目视觉演示。我们提出ManipForce，一种手持系统，用于在自然的人类演示过程中捕捉高频力- torque (F/T) 和 RGB 数据，以适用于接触丰富的操作。基于这些演示，我们引入了频率感知多模态Transformer (FMT)。FMT 使用频率和模态感知嵌入来编码异步 RGB 和 F/T 信号，并通过变压器扩散策略内的双向跨注意力将它们融合。通过在六项实际的接触丰富的操作任务（如齿轮装配、箱子翻转和电池插入）上的广泛实验，使用ManipForce演示训练的FMT实现了平均83%的成功率，在所有任务中都表现出色，显著优于仅基于RGB的基线。进一步的消融分析和采样频率分析证实，整合高频F/T数据和跨模态集成可以提高策略性能，特别是在要求高精度和稳定接触的任务中表现更为突出。 

---
# Reduced-Order Model-Guided Reinforcement Learning for Demonstration-Free Humanoid Locomotion 

**Title (ZH)**: 基于降阶模型的示范-free humanoid运动强化学习 

**Authors**: Shuai Liu, Meng Cheng Lau  

**Link**: [PDF](https://arxiv.org/pdf/2509.19023)  

**Abstract**: We introduce Reduced-Order Model-Guided Reinforcement Learning (ROM-GRL), a two-stage reinforcement learning framework for humanoid walking that requires no motion capture data or elaborate reward shaping. In the first stage, a compact 4-DOF (four-degree-of-freedom) reduced-order model (ROM) is trained via Proximal Policy Optimization. This generates energy-efficient gait templates. In the second stage, those dynamically consistent trajectories guide a full-body policy trained with Soft Actor--Critic augmented by an adversarial discriminator, ensuring the student's five-dimensional gait feature distribution matches the ROM's demonstrations. Experiments at 1 meter-per-second and 4 meter-per-second show that ROM-GRL produces stable, symmetric gaits with substantially lower tracking error than a pure-reward baseline. By distilling lightweight ROM guidance into high-dimensional policies, ROM-GRL bridges the gap between reward-only and imitation-based locomotion methods, enabling versatile, naturalistic humanoid behaviors without any human demonstrations. 

**Abstract (ZH)**: Reduced-Order Model-Guided Reinforcement Learning for Humanoid Walking 

---
# Pure Vision Language Action (VLA) Models: A Comprehensive Survey 

**Title (ZH)**: 纯视觉语言动作（VLA）模型：综述 

**Authors**: Dapeng Zhang, Jin Sun, Chenghui Hu, Xiaoyan Wu, Zhenlong Yuan, Rui Zhou, Fei Shen, Qingguo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.19012)  

**Abstract**: The emergence of Vision Language Action (VLA) models marks a paradigm shift from traditional policy-based control to generalized robotics, reframing Vision Language Models (VLMs) from passive sequence generators into active agents for manipulation and decision-making in complex, dynamic environments. This survey delves into advanced VLA methods, aiming to provide a clear taxonomy and a systematic, comprehensive review of existing research. It presents a comprehensive analysis of VLA applications across different scenarios and classifies VLA approaches into several paradigms: autoregression-based, diffusion-based, reinforcement-based, hybrid, and specialized methods; while examining their motivations, core strategies, and implementations in detail. In addition, foundational datasets, benchmarks, and simulation platforms are introduced. Building on the current VLA landscape, the review further proposes perspectives on key challenges and future directions to advance research in VLA models and generalizable robotics. By synthesizing insights from over three hundred recent studies, this survey maps the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose VLA methods. 

**Abstract (ZH)**: 视觉语言动作（VLA）模型的涌现标志着从传统基于策略的控制向通用机器人学的范式转变，重新定义了视觉语言模型（VLMs）从被动序列生成器转变为在复杂动态环境中进行操作和决策的主动代理。本文综述了先进的VLA方法，旨在提供清晰的分类，并进行全面系统的文献综述。综述详细分析了不同场景下的VLA应用，并将VLA方法分类为自回归、扩散、强化学习、混合和专门方法等 paradigm，同时探讨了它们的动机、核心策略和实现方式。此外，还介绍了基础数据集、基准测试和模拟平台。基于当前的VLA景观，综述进一步提出了关键挑战和未来方向，以促进VLA模型和通用机器人学的研究。通过综合分析三百多篇近期的研究，本文勾勒了这一快速发展的领域轮廓，并指出了塑造可扩展且通用的VLA方法发展的机会和挑战。 

---
# Eva-VLA: Evaluating Vision-Language-Action Models' Robustness Under Real-World Physical Variations 

**Title (ZH)**: Eva-VLA: 评估视觉-语言-行动模型在现实世界物理变化下的稳健性 

**Authors**: Hanqing Liu, Jiahuan Long, Junqi Wu, Jiacheng Hou, Huili Tang, Tingsong Jiang, Weien Zhou, Wen Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18953)  

**Abstract**: Vision-Language-Action (VLA) models have emerged as promising solutions for robotic manipulation, yet their robustness to real-world physical variations remains critically underexplored. To bridge this gap, we propose Eva-VLA, the first unified framework that systematically evaluates the robustness of VLA models by transforming discrete physical variations into continuous optimization problems. However, comprehensively assessing VLA robustness presents two key challenges: (1) how to systematically characterize diverse physical variations encountered in real-world deployments while maintaining evaluation reproducibility, and (2) how to discover worst-case scenarios without prohibitive real-world data collection costs efficiently. To address the first challenge, we decompose real-world variations into three critical domains: object 3D transformations that affect spatial reasoning, illumination variations that challenge visual perception, and adversarial patches that disrupt scene understanding. For the second challenge, we introduce a continuous black-box optimization framework that transforms discrete physical variations into parameter optimization, enabling systematic exploration of worst-case scenarios. Extensive experiments on state-of-the-art OpenVLA models across multiple benchmarks reveal alarming vulnerabilities: all variation types trigger failure rates exceeding 60%, with object transformations causing up to 97.8% failure in long-horizon tasks. Our findings expose critical gaps between controlled laboratory success and unpredictable deployment readiness, while the Eva-VLA framework provides a practical pathway for hardening VLA-based robotic manipulation models against real-world deployment challenges. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型在机器人操作中展现出潜力，但其在现实世界物理变异中的鲁棒性仍严重欠缺研究。为弥补这一差距，我们提出了Eva-VLA，这是首个系统评估VLA模型鲁棒性的统一框架，通过将离散的物理变异转化为连续的优化问题来实现。然而，全面评估VLA鲁棒性面临两个关键挑战：(1) 如何系统地表征现实生活部署中遇到的各种物理变异并保持评价的可重复性，以及(2) 如何高效地发现最坏情况场景而无需高昂的现实世界数据收集成本。为应对第一个挑战，我们将现实生活中的变异分解为三个关键领域：影响空间推理的物体3D变换、挑战视觉感知的光照变化，以及干扰场景理解的对抗性贴图。为应对第二个挑战，我们引入了一个连续的黑盒优化框架，将离散的物理变异转化为参数优化，从而系统探索最坏情况场景。在多个基准上的实验表明，最先进的OpenVLA模型在多种变体类型触发的故障率超过60%，物体变换在长时任务中导致高达97.8%的故障。我们的研究揭示了在受控实验室成功和不可预测部署准备之间的重要差距，而Eva-VLA框架提供了一条实用途径，以增强基于VLA的机器人操作模型以应对现实世界部署挑战。 

---
# Lang2Morph: Language-Driven Morphological Design of Robotic Hands 

**Title (ZH)**: 基于语言驱动的手部形态学设计：Lang2Morph 

**Authors**: Yanyuan Qiao, Kieran Gilday, Yutong Xie, Josie Hughes  

**Link**: [PDF](https://arxiv.org/pdf/2509.18937)  

**Abstract**: Designing robotic hand morphologies for diverse manipulation tasks requires balancing dexterity, manufacturability, and task-specific functionality. While open-source frameworks and parametric tools support reproducible design, they still rely on expert heuristics and manual tuning. Automated methods using optimization are often compute-intensive, simulation-dependent, and rarely target dexterous hands. Large language models (LLMs), with their broad knowledge of human-object interactions and strong generative capabilities, offer a promising alternative for zero-shot design reasoning. In this paper, we present Lang2Morph, a language-driven pipeline for robotic hand design. It uses LLMs to translate natural-language task descriptions into symbolic structures and OPH-compatible parameters, enabling 3D-printable task-specific morphologies. The pipeline consists of: (i) Morphology Design, which maps tasks into semantic tags, structural grammars, and OPH-compatible parameters; and (ii) Selection and Refinement, which evaluates design candidates based on semantic alignment and size compatibility, and optionally applies LLM-guided refinement when needed. We evaluate Lang2Morph across varied tasks, and results show that our approach can generate diverse, task-relevant morphologies. To our knowledge, this is the first attempt to develop an LLM-based framework for task-conditioned robotic hand design. 

**Abstract (ZH)**: 基于大语言模型的零样本设计推理在机器人手设计中的应用：Lang2Morph管道 

---
# VGGT-DP: Generalizable Robot Control via Vision Foundation Models 

**Title (ZH)**: VGGT-DP: 通过视觉基础模型实现可泛化的机器人控制 

**Authors**: Shijia Ge, Yinxin Zhang, Shuzhao Xie, Weixiang Zhang, Mingcai Zhou, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18778)  

**Abstract**: Visual imitation learning frameworks allow robots to learn manipulation skills from expert demonstrations. While existing approaches mainly focus on policy design, they often neglect the structure and capacity of visual encoders, limiting spatial understanding and generalization. Inspired by biological vision systems, which rely on both visual and proprioceptive cues for robust control, we propose VGGT-DP, a visuomotor policy framework that integrates geometric priors from a pretrained 3D perception model with proprioceptive feedback. We adopt the Visual Geometry Grounded Transformer (VGGT) as the visual encoder and introduce a proprioception-guided visual learning strategy to align perception with internal robot states, improving spatial grounding and closed-loop control. To reduce inference latency, we design a frame-wise token reuse mechanism that compacts multi-view tokens into an efficient spatial representation. We further apply random token pruning to enhance policy robustness and reduce overfitting. Experiments on challenging MetaWorld tasks show that VGGT-DP significantly outperforms strong baselines such as DP and DP3, particularly in precision-critical and long-horizon scenarios. 

**Abstract (ZH)**: 视觉模仿学习框架使机器人能够从专家示范中学习操作技能。现有的方法主要集中在策略设计上，往往忽视视觉编码器的结构和容量，限制了空间理解能力和泛化能力。受生物视觉系统依赖视觉和本体感觉线索进行鲁棒控制的启发，我们提出了一种结合预训练3D感知模型的几何先验和本体感觉反馈的visuomotor策略框架VGGT-DP。我们采用Visual Geometry Grounded Transformer (VGGT) 作为视觉编码器，并引入本体感觉引导的视觉学习策略，以使感知与内部机器人状态对齐，从而提高空间定位能力和闭环控制性能。为了减少推断延迟，我们设计了一种帧内令牌重用机制，将多视角令牌压缩成高效的空间表示。进一步应用随机令牌剪枝以增强策略的鲁棒性并减少过拟合。在具有挑战性的MetaWorld任务上的实验表明，VGGT-DP在精度关键和长时序场景中显著优于如DP和DP3等强基准。 

---
# MV-UMI: A Scalable Multi-View Interface for Cross-Embodiment Learning 

**Title (ZH)**: MV-UMI: 一种可扩展的多视图接口用于跨实体学习 

**Authors**: Omar Rayyan, John Abanes, Mahmoud Hafez, Anthony Tzes, Fares Abu-Dakka  

**Link**: [PDF](https://arxiv.org/pdf/2509.18757)  

**Abstract**: Recent advances in imitation learning have shown great promise for developing robust robot manipulation policies from demonstrations. However, this promise is contingent on the availability of diverse, high-quality datasets, which are not only challenging and costly to collect but are often constrained to a specific robot embodiment. Portable handheld grippers have recently emerged as intuitive and scalable alternatives to traditional robotic teleoperation methods for data collection. However, their reliance solely on first-person view wrist-mounted cameras often creates limitations in capturing sufficient scene contexts. In this paper, we present MV-UMI (Multi-View Universal Manipulation Interface), a framework that integrates a third-person perspective with the egocentric camera to overcome this limitation. This integration mitigates domain shifts between human demonstration and robot deployment, preserving the cross-embodiment advantages of handheld data-collection devices. Our experimental results, including an ablation study, demonstrate that our MV-UMI framework improves performance in sub-tasks requiring broad scene understanding by approximately 47% across 3 tasks, confirming the effectiveness of our approach in expanding the range of feasible manipulation tasks that can be learned using handheld gripper systems, without compromising the cross-embodiment advantages inherent to such systems. 

**Abstract (ZH)**: 近期 imitation 学习的发展显示出巨大的潜力，用于从示范中开发鲁棒的机器人 manipulation 策略。然而，这一潜力取决于多样化的高质量数据集的可用性，这些数据集不仅收集困难且成本高昂，而且往往受限于特定的机器人实体。可携带的手held 夹持器最近作为传统的机器人遥操作方法的直观且可扩展的替代方案，用于数据采集。然而，它们仅依赖于手腕安装的_first-person_ 视角摄像头，常常限制了场景上下文的充分捕获。在本文中，我们提出了 MV-UMI（多视角通用 manipulation 接口）框架，该框架结合了第三视角与本体中心摄像头，以克服这一限制。这种结合减轻了人类示范与机器人部署之间的域转移，保留了可携带数据采集设备的跨实体优势。我们的实验结果，包括消融研究，证明我们的 MV-UMI 框架在要求广泛场景理解的子任务中性能提升了约 47%，在三个任务中证实了该方法的有效性，即通过手持夹持器系统可以学习更多可行的 manipulation 任务，同时不牺牲此类系统固有的跨实体优势。 

---
# Learning Obstacle Avoidance using Double DQN for Quadcopter Navigation 

**Title (ZH)**: 使用双DQN学习Quadcopter导航障碍物避免 

**Authors**: Nishant Doshi, Amey Sutvani, Sanket Gujar  

**Link**: [PDF](https://arxiv.org/pdf/2509.18734)  

**Abstract**: One of the challenges faced by Autonomous Aerial Vehicles is reliable navigation through urban environments. Factors like reduction in precision of Global Positioning System (GPS), narrow spaces and dynamically moving obstacles make the path planning of an aerial robot a complicated task. One of the skills required for the agent to effectively navigate through such an environment is to develop an ability to avoid collisions using information from onboard depth sensors. In this paper, we propose Reinforcement Learning of a virtual quadcopter robot agent equipped with a Depth Camera to navigate through a simulated urban environment. 

**Abstract (ZH)**: 自主飞行器在城市环境中可靠导航的挑战：基于深度相机的强化学习虚拟四旋翼机器人在模拟城市环境中的导航 

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
# SPiDR: A Simple Approach for Zero-Shot Safety in Sim-to-Real Transfer 

**Title (ZH)**: SPiDR:一种简单的零样本安全迁移方法 

**Authors**: Yarden As, Chengrui Qu, Benjamin Unger, Dongho Kang, Max van der Hart, Laixi Shi, Stelian Coros, Adam Wierman, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2509.18648)  

**Abstract**: Safety remains a major concern for deploying reinforcement learning (RL) in real-world applications. Simulators provide safe, scalable training environments, but the inevitable sim-to-real gap introduces additional safety concerns, as policies must satisfy constraints in real-world conditions that differ from simulation. To address this challenge, robust safe RL techniques offer principled methods, but are often incompatible with standard scalable training pipelines. In contrast, domain randomization, a simple and popular sim-to-real technique, stands out as a promising alternative, although it often results in unsafe behaviors in practice. We present SPiDR, short for Sim-to-real via Pessimistic Domain Randomization -- a scalable algorithm with provable guarantees for safe sim-to-real transfer. SPiDR uses domain randomization to incorporate the uncertainty about the sim-to-real gap into the safety constraints, making it versatile and highly compatible with existing training pipelines. Through extensive experiments on sim-to-sim benchmarks and two distinct real-world robotic platforms, we demonstrate that SPiDR effectively ensures safety despite the sim-to-real gap while maintaining strong performance. 

**Abstract (ZH)**: 基于悲观域随机化的可扩展安全模拟到现实转移方法(SPiDR) 

---
# Do You Need Proprioceptive States in Visuomotor Policies? 

**Title (ZH)**: 你在视觉运动策略中需要本体感觉状态吗？ 

**Authors**: Juntu Zhao, Wenbo Lu, Di Zhang, Yufeng Liu, Yushen Liang, Tianluo Zhang, Yifeng Cao, Junyuan Xie, Yingdong Hu, Shengjie Wang, Junliang Guo, Dequan Wang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18644)  

**Abstract**: Imitation-learning-based visuomotor policies have been widely used in robot manipulation, where both visual observations and proprioceptive states are typically adopted together for precise control. However, in this study, we find that this common practice makes the policy overly reliant on the proprioceptive state input, which causes overfitting to the training trajectories and results in poor spatial generalization. On the contrary, we propose the State-free Policy, removing the proprioceptive state input and predicting actions only conditioned on visual observations. The State-free Policy is built in the relative end-effector action space, and should ensure the full task-relevant visual observations, here provided by dual wide-angle wrist cameras. Empirical results demonstrate that the State-free policy achieves significantly stronger spatial generalization than the state-based policy: in real-world tasks such as pick-and-place, challenging shirt-folding, and complex whole-body manipulation, spanning multiple robot embodiments, the average success rate improves from 0\% to 85\% in height generalization and from 6\% to 64\% in horizontal generalization. Furthermore, they also show advantages in data efficiency and cross-embodiment adaptation, enhancing their practicality for real-world deployment. 

**Abstract (ZH)**: 基于模仿学习的视觉运动策略通常在机器人操作中使用，其中视觉观察和本体感觉状态通常联合使用以实现精确控制。然而，在本研究中，我们发现这一常见做法使策略过度依赖本体感觉状态输入，导致对训练轨迹过拟合并导致空间泛化能力较差。与此相反，我们提出了无状态策略，去除了本体感觉状态输入，并仅根据视觉观察预测动作。无状态策略构建在相对于末端执行器的动作空间中，并应确保全面的相关视觉观察，这里由双广角手腕摄像头提供。实验结果表明，无状态策略在空间泛化方面显著优于基于状态的策略：在包括抓取放置、复杂衣物折叠和多机器人本体的整体身体操作等真实世界任务中，高度泛化成功率从0%提高到85%，水平泛化成功率从6%提高到64%。此外，它们还展示了高效的数据利用和跨本体适应性优势，增强了其实用性，适用于真实世界的部署。 

---
# Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training 

**Title (ZH)**: 模拟与现实环境政策共训练的可泛化领域自适应方法 

**Authors**: Shuo Cheng, Liqian Ma, Zhenyang Chen, Ajay Mandlekar, Caelan Garrett, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18631)  

**Abstract**: Behavior cloning has shown promise for robot manipulation, but real-world demonstrations are costly to acquire at scale. While simulated data offers a scalable alternative, particularly with advances in automated demonstration generation, transferring policies to the real world is hampered by various simulation and real domain gaps. In this work, we propose a unified sim-and-real co-training framework for learning generalizable manipulation policies that primarily leverages simulation and only requires a few real-world demonstrations. Central to our approach is learning a domain-invariant, task-relevant feature space. Our key insight is that aligning the joint distributions of observations and their corresponding actions across domains provides a richer signal than aligning observations (marginals) alone. We achieve this by embedding an Optimal Transport (OT)-inspired loss within the co-training framework, and extend this to an Unbalanced OT framework to handle the imbalance between abundant simulation data and limited real-world examples. We validate our method on challenging manipulation tasks, showing it can leverage abundant simulation data to achieve up to a 30% improvement in the real-world success rate and even generalize to scenarios seen only in simulation. 

**Abstract (ZH)**: 基于仿真实例和现实世界的统一共同训练框架学习通用化 manipulation 策略 

---
# SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones 

**Title (ZH)**: SINGER：无人机上载通用视觉-语言导航策略 

**Authors**: Maximilian Adang, JunEn Low, Ola Shorinwa, Mac Schwager  

**Link**: [PDF](https://arxiv.org/pdf/2509.18610)  

**Abstract**: Large vision-language models have driven remarkable progress in open-vocabulary robot policies, e.g., generalist robot manipulation policies, that enable robots to complete complex tasks specified in natural language. Despite these successes, open-vocabulary autonomous drone navigation remains an unsolved challenge due to the scarcity of large-scale demonstrations, real-time control demands of drones for stabilization, and lack of reliable external pose estimation modules. In this work, we present SINGER for language-guided autonomous drone navigation in the open world using only onboard sensing and compute. To train robust, open-vocabulary navigation policies, SINGER leverages three central components: (i) a photorealistic language-embedded flight simulator with minimal sim-to-real gap using Gaussian Splatting for efficient data generation, (ii) an RRT-inspired multi-trajectory generation expert for collision-free navigation demonstrations, and these are used to train (iii) a lightweight end-to-end visuomotor policy for real-time closed-loop control. Through extensive hardware flight experiments, we demonstrate superior zero-shot sim-to-real transfer of our policy to unseen environments and unseen language-conditioned goal objects. When trained on ~700k-1M observation action pairs of language conditioned visuomotor data and deployed on hardware, SINGER outperforms a velocity-controlled semantic guidance baseline by reaching the query 23.33% more on average, and maintains the query in the field of view 16.67% more on average, with 10% fewer collisions. 

**Abstract (ZH)**: 基于语言指导的开放世界自主无人机导航 

---
# End-to-End Crop Row Navigation via LiDAR-Based Deep Reinforcement Learning 

**Title (ZH)**: 基于LiDAR的深度强化学习端到端农作物行导航 

**Authors**: Ana Luiza Mineiro, Francisco Affonso, Marcelo Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.18608)  

**Abstract**: Reliable navigation in under-canopy agricultural environments remains a challenge due to GNSS unreliability, cluttered rows, and variable lighting. To address these limitations, we present an end-to-end learning-based navigation system that maps raw 3D LiDAR data directly to control commands using a deep reinforcement learning policy trained entirely in simulation. Our method includes a voxel-based downsampling strategy that reduces LiDAR input size by 95.83%, enabling efficient policy learning without relying on labeled datasets or manually designed control interfaces. The policy was validated in simulation, achieving a 100% success rate in straight-row plantations and showing a gradual decline in performance as row curvature increased, tested across varying sinusoidal frequencies and amplitudes. 

**Abstract (ZH)**: 基于深度强化学习的端到端LiDAR数据导航系统：在林下农业环境中的可靠导航 

---
# Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills 

**Title (ZH)**: 与你的躯体型代理共同成长：一种包含人类在环的长期 horizon 操作技能终身代码生成框架 

**Authors**: Yuan Meng, Zhenguo Sun, Max Fest, Xukun Li, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.18597)  

**Abstract**: Large language models (LLMs)-based code generation for robotic manipulation has recently shown promise by directly translating human instructions into executable code, but existing methods remain noisy, constrained by fixed primitives and limited context windows, and struggle with long-horizon tasks. While closed-loop feedback has been explored, corrected knowledge is often stored in improper formats, restricting generalization and causing catastrophic forgetting, which highlights the need for learning reusable skills. Moreover, approaches that rely solely on LLM guidance frequently fail in extremely long-horizon scenarios due to LLMs' limited reasoning capability in the robotic domain, where such issues are often straightforward for humans to identify. To address these challenges, we propose a human-in-the-loop framework that encodes corrections into reusable skills, supported by external memory and Retrieval-Augmented Generation with a hint mechanism for dynamic reuse. Experiments on Ravens, Franka Kitchen, and MetaWorld, as well as real-world settings, show that our framework achieves a 0.93 success rate (up to 27% higher than baselines) and a 42% efficiency improvement in correction rounds. It can robustly solve extremely long-horizon tasks such as "build a house", which requires planning over 20 primitives. 

**Abstract (ZH)**: 基于大型语言模型的代码生成在机器人操作中的前景：一种支持外部记忆和提示机制的循环反馈框架 

---
# VLN-Zero: Rapid Exploration and Cache-Enabled Neurosymbolic Vision-Language Planning for Zero-Shot Transfer in Robot Navigation 

**Title (ZH)**: VLN-Zero: 快速探索与缓存辅助神经符号视觉语言规划在机器人导航中的零样本迁移 

**Authors**: Neel P. Bhatt, Yunhao Yang, Rohan Siva, Pranay Samineni, Daniel Milan, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18592)  

**Abstract**: Rapid adaptation in unseen environments is essential for scalable real-world autonomy, yet existing approaches rely on exhaustive exploration or rigid navigation policies that fail to generalize. We present VLN-Zero, a two-phase vision-language navigation framework that leverages vision-language models to efficiently construct symbolic scene graphs and enable zero-shot neurosymbolic navigation. In the exploration phase, structured prompts guide VLM-based search toward informative and diverse trajectories, yielding compact scene graph representations. In the deployment phase, a neurosymbolic planner reasons over the scene graph and environmental observations to generate executable plans, while a cache-enabled execution module accelerates adaptation by reusing previously computed task-location trajectories. By combining rapid exploration, symbolic reasoning, and cache-enabled execution, the proposed framework overcomes the computational inefficiency and poor generalization of prior vision-language navigation methods, enabling robust and scalable decision-making in unseen environments. VLN-Zero achieves 2x higher success rate compared to state-of-the-art zero-shot models, outperforms most fine-tuned baselines, and reaches goal locations in half the time with 55% fewer VLM calls on average compared to state-of-the-art models across diverse environments. Codebase, datasets, and videos for VLN-Zero are available at: this https URL. 

**Abstract (ZH)**: 视觉-语言导航中的VLN-Zero：一种高效的零样本神经符号导航框架 

---
# LCMF: Lightweight Cross-Modality Mambaformer for Embodied Robotics VQA 

**Title (ZH)**: LCMF：轻量级跨模态Mambaformer在体态机器人VQA中的应用 

**Authors**: Zeyi Kang, Liang He, Yanxin Zhang, Zuheng Ming, Kaixing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18576)  

**Abstract**: Multimodal semantic learning plays a critical role in embodied intelligence, especially when robots perceive their surroundings, understand human instructions, and make intelligent decisions. However, the field faces technical challenges such as effective fusion of heterogeneous data and computational efficiency in resource-constrained environments. To address these challenges, this study proposes the lightweight LCMF cascaded attention framework, introducing a multi-level cross-modal parameter sharing mechanism into the Mamba module. By integrating the advantages of Cross-Attention and Selective parameter-sharing State Space Models (SSMs), the framework achieves efficient fusion of heterogeneous modalities and semantic complementary alignment. Experimental results show that LCMF surpasses existing multimodal baselines with an accuracy of 74.29% in VQA tasks and achieves competitive mid-tier performance within the distribution cluster of Large Language Model Agents (LLM Agents) in EQA video tasks. Its lightweight design achieves a 4.35-fold reduction in FLOPs relative to the average of comparable baselines while using only 166.51M parameters (image-text) and 219M parameters (video-text), providing an efficient solution for Human-Robot Interaction (HRI) applications in resource-constrained scenarios with strong multimodal decision generalization capabilities. 

**Abstract (ZH)**: 多模态语义学习在体效应智中的关键作用，特别是在机器人感知周围环境、理解人类指令并做出智能决策时。然而，该领域面临着如异构数据的有效融合和资源受限环境中计算效率等技术挑战。为应对这些挑战，本研究提出了一种轻量级LCMF级联注意力框架，将多级跨模态参数共享机制引入Mamba模块。通过结合跨注意力和选择性参数共享状态空间模型的优势，该框架实现了异构模态的有效融合和语义互补对齐。实验结果显示，LCMF在VQA任务中的准确率为74.29%，超越现有多种模态基线，并在EQA视频任务中获得了大型语言模型代理分布集群中的竞争力中等表现。其轻量级设计相比可比基线平均减少了4.35倍的FLOPs，仅使用166.51M（图像-文本）和219M（视频-文本）参数，为资源受限场景中的体效应交互（HRI）应用提供了高效解决方案，具备强大的多种模态决策泛化能力。 

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

**Title (ZH)**: PrioriTouch: 根据用户接触偏好适应的全身物理人机互动 

**Authors**: Rishabh Madan, Jiawei Lin, Mahika Goel, Angchen Xie, Xiaoyu Liang, Marcus Lee, Justin Guo, Pranav N. Thakkar, Rohan Banerjee, Jose Barreiros, Kate Tsui, Tom Silver, Tapomayukh Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.18447)  

**Abstract**: Physical human-robot interaction (pHRI) requires robots to adapt to individual contact preferences, such as where and how much force is applied. Identifying preferences is difficult for a single contact; with whole-arm interaction involving multiple simultaneous contacts between the robot and human, the challenge is greater because different body parts can impose incompatible force requirements. In caregiving tasks, where contact is frequent and varied, such conflicts are unavoidable. With multiple preferences across multiple contacts, no single solution can satisfy all objectives--trade-offs are inherent, making prioritization essential. We present PrioriTouch, a framework for ranking and executing control objectives across multiple contacts. PrioriTouch can prioritize from a general collection of controllers, making it applicable not only to caregiving scenarios such as bed bathing and dressing but also to broader multi-contact settings. Our method combines a novel learning-to-rank approach with hierarchical operational space control, leveraging simulation-in-the-loop rollouts for data-efficient and safe exploration. We conduct a user study on physical assistance preferences, derive personalized comfort thresholds, and incorporate them into PrioriTouch. We evaluate PrioriTouch through extensive simulation and real-world experiments, demonstrating its ability to adapt to user contact preferences, maintain task performance, and enhance safety and comfort. Website: this https URL. 

**Abstract (ZH)**: 物理人机交互（pHRI）要求机器人适应个体的接触偏好，如力的施加位置和力度。识别这些偏好单独接触时颇具挑战性；而在使用完整臂进行互动时，由于多次同时接触可能导致不同的身体部位产生不兼容的力需求，挑战更大。在护理任务中，由于接触频繁且多样化，此类冲突不可避免。面对多个接触点上的多种偏好，没有单一解决方案能满足所有目标——权衡不可避免，因此优先级设置至关重要。我们提出了PrioriTouch框架，用于在多个接触点上排名和执行控制目标。PrioriTouch可以从广泛的控制器集合中进行优先级设置，不仅适用于诸如擦浴和穿衣等护理场景，也适用于更广泛的多接触场景。我们的方法结合了新颖的排序学习方法与分层级操作空间控制，利用闭环仿真进行数据高效且安全的探索。我们进行了用户研究以确定物理辅助偏好，推导个性化舒适阈值，并将其整合至PrioriTouch中。我们通过广泛的仿真和现实世界实验评估了PrioriTouch，展示了其适应用户接触偏好、维持任务性能、提高安全性和舒适性的能力。网站：this https URL。 

---
# Latent Action Pretraining Through World Modeling 

**Title (ZH)**: 世界建模导向的潜动作预训练 

**Authors**: Bahey Tharwat, Yara Nasser, Ali Abouzeid, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.18428)  

**Abstract**: Vision-Language-Action (VLA) models have gained popularity for learning robotic manipulation tasks that follow language instructions. State-of-the-art VLAs, such as OpenVLA and $\pi_{0}$, were trained on large-scale, manually labeled action datasets collected through teleoperation. More recent approaches, including LAPA and villa-X, introduce latent action representations that enable unsupervised pretraining on unlabeled datasets by modeling abstract visual changes between frames. Although these methods have shown strong results, their large model sizes make deployment in real-world settings challenging. In this work, we propose LAWM, a model-agnostic framework to pretrain imitation learning models in a self-supervised way, by learning latent action representations from unlabeled video data through world modeling. These videos can be sourced from robot recordings or videos of humans performing actions with everyday objects. Our framework is designed to be effective for transferring across tasks, environments, and embodiments. It outperforms models trained with ground-truth robotics actions and similar pretraining methods on the LIBERO benchmark and real-world setup, while being significantly more efficient and practical for real-world settings. 

**Abstract (ZH)**: 一种模型无关的框架：通过世界建模从未标记视频数据中学习潜在动作表示以进行自监督预训练（LAWM） 

---
# Fine-Tuning Robot Policies While Maintaining User Privacy 

**Title (ZH)**: 细调机器人策略以保持用户隐私 

**Authors**: Benjamin A. Christie, Sagar Parekh, Dylan P. Losey  

**Link**: [PDF](https://arxiv.org/pdf/2509.18311)  

**Abstract**: Recent works introduce general-purpose robot policies. These policies provide a strong prior over how robots should behave -- e.g., how a robot arm should manipulate food items. But in order for robots to match an individual person's needs, users typically fine-tune these generalized policies -- e.g., showing the robot arm how to make their own preferred dinners. Importantly, during the process of personalizing robots, end-users leak data about their preferences, habits, and styles (e.g., the foods they prefer to eat). Other agents can simply roll-out the fine-tuned policy and see these personally-trained behaviors. This leads to a fundamental challenge: how can we develop robots that personalize actions while keeping learning private from external agents? We here explore this emerging topic in human-robot interaction and develop PRoP, a model-agnostic framework for personalized and private robot policies. Our core idea is to equip each user with a unique key; this key is then used to mathematically transform the weights of the robot's network. With the correct key, the robot's policy switches to match that user's preferences -- but with incorrect keys, the robot reverts to its baseline behaviors. We show the general applicability of our method across multiple model types in imitation learning, reinforcement learning, and classification tasks. PRoP is practically advantageous because it retains the architecture and behaviors of the original policy, and experimentally outperforms existing encoder-based approaches. See videos and code here: this https URL. 

**Abstract (ZH)**: 近期的研究引入了通用机器人策略。这些策略为机器人应当如何行为提供了一个强大的先验——例如，如何操作食物。但为了使机器人能够满足个人需求，用户通常需要微调这些通用策略——例如，展示机器人如何制作自己偏好的晚餐。重要的是，在个性化机器人的过程中，最终用户会泄露其偏好、习惯和风格（例如，他们喜欢吃什么食物）的数据。其他代理可以简单地执行微调过的策略并观察这些个性化训练的行为。这一过程带来了根本性挑战：我们如何开发能够在不从外部代理处泄露学习的情况下个性化动作的机器人？我们在这里探讨了这一新兴的机器人交互主题，并开发了PRoP，这是一种模型无关的个性化和私有化机器人策略的框架。我们的核心思想是为每个用户提供一个独特的密钥；然后使用这个密钥对机器人网络的权重进行数学变换。通过正确的密钥，机器人的策略会切换以匹配该用户的需求——但通过错误的密钥，机器人会恢复其基础行为。我们在模仿学习、强化学习和分类任务等多种模型类型中展示了我们方法的普遍适用性。PRoP 实际上具有优势，因为它保留了原始策略的架构和行为，并且实验上优于现有基于编码器的方法。 

---
# PEEK: Guiding and Minimal Image Representations for Zero-Shot Generalization of Robot Manipulation Policies 

**Title (ZH)**: PEEK: 引导和 minimalist 图像表示以实现机器人操作策略的零样本泛化 

**Authors**: Jesse Zhang, Marius Memmel, Kevin Kim, Dieter Fox, Jesse Thomason, Fabio Ramos, Erdem Bıyık, Abhishek Gupta, Anqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18282)  

**Abstract**: Robotic manipulation policies often fail to generalize because they must simultaneously learn where to attend, what actions to take, and how to execute them. We argue that high-level reasoning about where and what can be offloaded to vision-language models (VLMs), leaving policies to specialize in how to act. We present PEEK (Policy-agnostic Extraction of Essential Keypoints), which fine-tunes VLMs to predict a unified point-based intermediate representation: 1. end-effector paths specifying what actions to take, and 2. task-relevant masks indicating where to focus. These annotations are directly overlaid onto robot observations, making the representation policy-agnostic and transferable across architectures. To enable scalable training, we introduce an automatic annotation pipeline, generating labeled data across 20+ robot datasets spanning 9 embodiments. In real-world evaluations, PEEK consistently boosts zero-shot generalization, including a 41.4x real-world improvement for a 3D policy trained only in simulation, and 2-3.5x gains for both large VLAs and small manipulation policies. By letting VLMs absorb semantic and visual complexity, PEEK equips manipulation policies with the minimal cues they need--where, what, and how. Website at this https URL. 

**Abstract (ZH)**: 基于视觉语言模型的机器人操作策略往往因为必须同时学习关注何处、采取什么行动以及如何执行而难以迁移。我们主张高层关于何处和什么的推理可以卸载到视觉语言模型（VLMs）上，从而使策略专注于如何行动。我们提出了PEEK（Policy-agnostic Extraction of Essential Keypoints），该方法微调VLMs以预测统一的基于点的中间表示：1. 表示应采取什么行动的末端执行器路径，以及2. 表示需要关注的区域的任务相关掩码。这些标注可以直接叠加到机器人观测上，使表示具有策略无关性和架构间可迁移性。为实现规模化训练，我们引入了一种自动标注流水线，在涵盖9种不同机器人操作模型的20多个数据集中生成了标注数据。在真实世界的评估中，PEEK一致地提升了零样本迁移，包括对于仅在模拟中训练的3D策略在真实世界中的改进达到41.4倍，以及对于大型VLAs和小规模操作策略的2-3.5倍的增益。通过让VLMs吸收语义和视觉复杂性，PEEK为操作策略提供了它们所需的最小线索——何处、什么和如何。网站地址：this https URL。 

---
# Conversational Orientation Reasoning: Egocentric-to-Allocentric Navigation with Multimodal Chain-of-Thought 

**Title (ZH)**: 自中心到他中心的对话导向推理：多模态链式思考导航 

**Authors**: Yu Ti Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18200)  

**Abstract**: Conversational agents must translate egocentric utterances (e.g., "on my right") into allocentric orientations (N/E/S/W). This challenge is particularly critical in indoor or complex facilities where GPS signals are weak and detailed maps are unavailable. While chain-of-thought (CoT) prompting has advanced reasoning in language and vision tasks, its application to multimodal spatial orientation remains underexplored. We introduce Conversational Orientation Reasoning (COR), a new benchmark designed for Traditional Chinese conversational navigation projected from real-world environments, addressing egocentric-to-allocentric reasoning in non-English and ASR-transcribed scenarios. We propose a multimodal chain-of-thought (MCoT) framework, which integrates ASR-transcribed speech with landmark coordinates through a structured three-step reasoning process: (1) extracting spatial relations, (2) mapping coordinates to absolute directions, and (3) inferring user orientation. A curriculum learning strategy progressively builds these capabilities on Taiwan-LLM-13B-v2.0-Chat, a mid-sized model representative of resource-constrained settings. Experiments show that MCoT achieves 100% orientation accuracy on clean transcripts and 98.1% with ASR transcripts, substantially outperforming unimodal and non-structured baselines. Moreover, MCoT demonstrates robustness under noisy conversational conditions, including ASR recognition errors and multilingual code-switching. The model also maintains high accuracy in cross-domain evaluation and resilience to linguistic variation, domain shift, and referential ambiguity. These findings highlight the potential of structured MCoT spatial reasoning as a path toward interpretable and resource-efficient embodied navigation. 

**Abstract (ZH)**: 对话代理必须将以自我为中心的陈述（例如，“在我右边”）转换为他我中心的方向（N/E/S/W）。这一挑战在GPS信号弱且详细地图不可用的室内或复杂设施中尤为重要。虽然思维链（CoT）提示在语言和视觉任务中提高了推理能力，但在多模态空间定向领域的应用仍不够充分。我们引入了对话方向推理（COR），这是一个旨在中文对话导航任务中利用现实环境中数据的新基准，解决非英语和ASR转录的以自我为中心到他我中心的推理问题。我们提出了一种多模态思维链（MCoT）框架，该框架通过结构化的三步推理过程将ASR转录的语音与地标坐标结合：（1）提取空间关系，（2）映射坐标到绝对方向，（3）推断用户方向。一种 Curriculum 学习策略逐步在Taiwan-LLM-13B-v2.0-Chat上构建这些能力，这是一种代表资源受限设置的中型模型。实验结果显示，MCoT在干净的转录本中实现了100%的方向准确性，并在ASR转录本中达到了98.1%的准确性，显著优于单模态和非结构化基线。此外，MCoT在嘈杂的对话条件下表现出色，包括ASR识别错误和多语言切换。该模型在跨域评估中保持了高准确性，并对语言变体、领域转移和指代歧义具有抗干扰性。这些发现突显了结构化MCoT空间推理作为可解释和资源高效的体感导航途径的潜力。 

---
# How Far are VLMs from Visual Spatial Intelligence? A Benchmark-Driven Perspective 

**Title (ZH)**: VLMs在视觉空间智能方面的差距：一个基准驱动的视角 

**Authors**: Songsong Yu, Yuxin Chen, Hao Ju, Lianjie Jia, Fuxi Zhang, Shaofei Huang, Yuhan Wu, Rundi Cui, Binghao Ran, Zaibin Zhang, Zhedong Zheng, Zhipeng Zhang, Yifan Wang, Lin Song, Lijun Wang, Yanwei Li, Ying Shan, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18905)  

**Abstract**: Visual Spatial Reasoning (VSR) is a core human cognitive ability and a critical requirement for advancing embodied intelligence and autonomous systems. Despite recent progress in Vision-Language Models (VLMs), achieving human-level VSR remains highly challenging due to the complexity of representing and reasoning over three-dimensional space. In this paper, we present a systematic investigation of VSR in VLMs, encompassing a review of existing methodologies across input modalities, model architectures, training strategies, and reasoning mechanisms. Furthermore, we categorize spatial intelligence into three levels of capability, ie, basic perception, spatial understanding, spatial planning, and curate SIBench, a spatial intelligence benchmark encompassing nearly 20 open-source datasets across 23 task settings. Experiments with state-of-the-art VLMs reveal a pronounced gap between perception and reasoning, as models show competence in basic perceptual tasks but consistently underperform in understanding and planning tasks, particularly in numerical estimation, multi-view reasoning, temporal dynamics, and spatial imagination. These findings underscore the substantial challenges that remain in achieving spatial intelligence, while providing both a systematic roadmap and a comprehensive benchmark to drive future research in the field. The related resources of this study are accessible at this https URL. 

**Abstract (ZH)**: 视觉空间推理（VSR）是核心的人类认知能力，对于推进具身智能和自主系统至关重要。尽管在视觉-语言模型（VLMs）方面取得了近期进展，但由于三维空间表示和推理的复杂性，实现 human-level 的 VSR 仍然极具挑战性。在本文中，我们系统研究了 VLMs 中的视觉空间推理，涵盖不同输入模态、模型架构、训练策略和推理机制的方法学综述。此外，我们将空间智能分类为三个能力层次，即基础知觉、空间理解、空间规划，并制定了一个包含近 20 个开源数据集的 SIBench 空间智能基准，覆盖 23 种任务设置。最先进的 VLMs 的实验揭示了感知与推理之间明显的差距，模型在基础知觉任务上表现出色，但在理解与规划任务上始终表现不佳，特别是在数值估算、多视角推理、时间动力学和空间想象方面。这些发现强调了实现空间智能仍面临的巨大挑战，同时提供了一份系统路线图和全面基准，推动该领域的未来研究。相关资源可访问此链接。 

---
# Multimodal Health Risk Prediction System for Chronic Diseases via Vision-Language Fusion and Large Language Models 

**Title (ZH)**: 基于视觉-语言融合和大规模语言模型的慢性疾病多模态健康风险预测系统 

**Authors**: Dingxin Lu, Shurui Wu, Xinyi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18221)  

**Abstract**: With the rising global burden of chronic diseases and the multimodal and heterogeneous clinical data (medical imaging, free-text recordings, wearable sensor streams, etc.), there is an urgent need for a unified multimodal AI framework that can proactively predict individual health risks. We propose VL-RiskFormer, a hierarchical stacked visual-language multimodal Transformer with a large language model (LLM) inference head embedded in its top layer. The system builds on the dual-stream architecture of existing visual-linguistic models (e.g., PaLM-E, LLaVA) with four key innovations: (i) pre-training with cross-modal comparison and fine-grained alignment of radiological images, fundus maps, and wearable device photos with corresponding clinical narratives using momentum update encoders and debiased InfoNCE losses; (ii) a time fusion block that integrates irregular visit sequences into the causal Transformer decoder through adaptive time interval position coding; (iii) a disease ontology map adapter that injects ICD-10 codes into visual and textual channels in layers and infers comorbid patterns with the help of a graph attention mechanism. On the MIMIC-IV longitudinal cohort, VL-RiskFormer achieved an average AUROC of 0.90 with an expected calibration error of 2.7 percent. 

**Abstract (ZH)**: 基于视觉-语言的多模态RiskFormer框架：融合临床叙事的大规模语言模型驱动的主动健康风险预测 

---
# MMCD: Multi-Modal Collaborative Decision-Making for Connected Autonomy with Knowledge Distillation 

**Title (ZH)**: MMCD: 多模态协作决策在具备知识蒸馏的连接自主系统中 

**Authors**: Rui Liu, Zikang Wang, Peng Gao, Yu Shen, Pratap Tokekar, Ming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.18198)  

**Abstract**: Autonomous systems have advanced significantly, but challenges persist in accident-prone environments where robust decision-making is crucial. A single vehicle's limited sensor range and obstructed views increase the likelihood of accidents. Multi-vehicle connected systems and multi-modal approaches, leveraging RGB images and LiDAR point clouds, have emerged as promising solutions. However, existing methods often assume the availability of all data modalities and connected vehicles during both training and testing, which is impractical due to potential sensor failures or missing connected vehicles. To address these challenges, we introduce a novel framework MMCD (Multi-Modal Collaborative Decision-making) for connected autonomy. Our framework fuses multi-modal observations from ego and collaborative vehicles to enhance decision-making under challenging conditions. To ensure robust performance when certain data modalities are unavailable during testing, we propose an approach based on cross-modal knowledge distillation with a teacher-student model structure. The teacher model is trained with multiple data modalities, while the student model is designed to operate effectively with reduced modalities. In experiments on $\textit{connected autonomous driving with ground vehicles}$ and $\textit{aerial-ground vehicles collaboration}$, our method improves driving safety by up to ${\it 20.7}\%$, surpassing the best-existing baseline in detecting potential accidents and making safe driving decisions. More information can be found on our website this https URL. 

**Abstract (ZH)**: 自主系统已取得显著进展，但在事故多发环境中，稳健的决策制定仍然面临挑战。单个车辆有限的传感器范围和受阻的视角增加了事故发生的风险。多车辆连接系统和多模态方法结合RGB图像和LiDAR点云数据，已被证明是一种有前景的解决方案。然而，现有方法通常假设在训练和测试过程中所有数据模态和连接车辆均可用，这由于传感器故障或缺少连接车辆的可能性而难以实现。为应对这些挑战，我们提出了一种新的框架MMCD（多模态协作决策）以实现连接自主性。该框架融合了ego车辆和协作车辆的多模态观测数据，以在恶劣条件下增强决策能力。为确保在某些数据模态测试不可用时仍能保持稳健性能，我们提出了一种基于跨模态知识蒸馏的教师-学生模型结构方法。教师模型使用多种数据模态训练，而学生模型设计为能在减少的数据模态下有效运行。在基于地面车辆的连接自主驾驶和空中-地面车辆协作的实验中，我们的方法通过最多20.7%提高驾驶安全性，并在检测潜在事故和作出安全驾驶决策方面超越现有最佳基准方法。更多信息请访问我们的网站：this https URL。 

---
# Citrus-V: Advancing Medical Foundation Models with Unified Medical Image Grounding for Clinical Reasoning 

**Title (ZH)**: 柑橘-V：统一医学图像 grounding 以推动临床推理的医学基础模型进展 

**Authors**: Guoxin Wang, Jun Zhao, Xinyi Liu, Yanbo Liu, Xuyang Cao, Chao Li, Zhuoyun Liu, Qintian Sun, Fangru Zhou, Haoqiang Xing, Zhenhong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19090)  

**Abstract**: Medical imaging provides critical evidence for clinical diagnosis, treatment planning, and surgical decisions, yet most existing imaging models are narrowly focused and require multiple specialized networks, limiting their generalization. Although large-scale language and multimodal models exhibit strong reasoning and multi-task capabilities, real-world clinical applications demand precise visual grounding, multimodal integration, and chain-of-thought reasoning. We introduce Citrus-V, a multimodal medical foundation model that combines image analysis with textual reasoning. The model integrates detection, segmentation, and multimodal chain-of-thought reasoning, enabling pixel-level lesion localization, structured report generation, and physician-like diagnostic inference in a single framework. We propose a novel multimodal training approach and release a curated open-source data suite covering reasoning, detection, segmentation, and document understanding tasks. Evaluations demonstrate that Citrus-V outperforms existing open-source medical models and expert-level imaging systems across multiple benchmarks, delivering a unified pipeline from visual grounding to clinical reasoning and supporting precise lesion quantification, automated reporting, and reliable second opinions. 

**Abstract (ZH)**: 医学影像为临床诊断、治疗计划和手术决策提供关键证据，但现有大多数影像模型关注狭窄领域且需要多个专业化网络，限制了它们的泛化能力。尽管大规模语言和多模态模型表现出强大的推理和多任务能力，实际临床应用需要精确的视觉定位、多模态融合和链条式推理。我们引入了Citrus-V，一种将图像分析与文本推理结合的多模态医学基础模型。该模型整合了检测、分割和多模态链条式推理，能够在单一体系框架中实现像素级病灶定位、结构化报告生成和类似于医生的诊断推理。我们提出了一种新颖的多模态训练方法，并发布了涵盖推理、检测、分割和文档理解任务的精心策划开源数据集。评估结果显示，Citrus-V 在多个基准测试中优于现有开源医学模型和专家级影像系统，提供从视觉定位到临床推理的统一流程，并支持精确的病灶量化、自动化报告和可靠的第二意见。 

---
# Fully Learnable Neural Reward Machines 

**Title (ZH)**: 完全可学习神经奖励机器 

**Authors**: Hazem Dewidar, Elena Umili  

**Link**: [PDF](https://arxiv.org/pdf/2509.19017)  

**Abstract**: Non-Markovian Reinforcement Learning (RL) tasks present significant challenges, as agents must reason over entire trajectories of state-action pairs to make optimal decisions. A common strategy to address this is through symbolic formalisms, such as Linear Temporal Logic (LTL) or automata, which provide a structured way to express temporally extended objectives. However, these approaches often rely on restrictive assumptions -- such as the availability of a predefined Symbol Grounding (SG) function mapping raw observations to high-level symbolic representations, or prior knowledge of the temporal task. In this work, we propose a fully learnable version of Neural Reward Machines (NRM), which can learn both the SG function and the automaton end-to-end, removing any reliance on prior knowledge. Our approach is therefore as easily applicable as classic deep RL (DRL) approaches, while being far more explainable, because of the finite and compact nature of automata. Furthermore, we show that by integrating Fully Learnable Reward Machines (FLNRM) with DRL, our method outperforms previous approaches based on Recurrent Neural Networks (RNNs). 

**Abstract (ZH)**: 非马尔可夫强化学习任务提出了重大挑战，因为代理必须在做出最优决策时对状态-行动对的整个轨迹进行推理。一种常见的应对策略是通过符号形式主义，如线性时序逻辑（LTL）或自动机，它们为表达时间扩展目标提供了一种结构化的方式。然而，这些方法通常依赖于一些限制性假设，例如预定义的符号基底（SG）函数映射原始观察到高级符号表示，或者已知的时间任务先验知识。在本文中，我们提出了一种完全可学习的神经奖励机器（NRM）版本，它可以端到端地学习SG函数和自动机，从而不再依赖于先验知识。因此，我们的方法在应用上与经典的深度强化学习（DRL）方法一样方便，但由于自动机具有有限和紧凑的性质，因此更易于解释。此外，我们展示了通过将完全可学习奖励机器（FLNRM）与DRL结合，我们的方法在基于递归神经网络（RNN）的先前方法上表现出更好的性能。 

---
# VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction 

**Title (ZH)**: VIR-Bench：通过旅行视频行程重构评估时空理解能力的地理和时间建模大模型 

**Authors**: Hao Wang, Eiki Murata, Lingfang Zhang, Ayako Sato, So Fukuda, Ziqi Yin, Wentao Hu, Keisuke Nakao, Yusuke Nakamura, Sebastian Zwirner, Yi-Chia Chen, Hiroyuki Otomo, Hiroki Ouchi, Daisuke Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2509.19002)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have significantly enhanced video understanding capabilities, opening new possibilities for practical applications. Yet current video benchmarks focus largely on indoor scenes or short-range outdoor activities, leaving the challenges associated with long-distance travel largely unexplored. Mastering extended geospatial-temporal trajectories is critical for next-generation MLLMs, underpinning real-world tasks such as embodied-AI planning and navigation. To bridge this gap, we present VIR-Bench, a novel benchmark consisting of 200 travel videos that frames itinerary reconstruction as a challenging task designed to evaluate and push forward MLLMs' geospatial-temporal intelligence. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, struggle to achieve high scores, underscoring the difficulty of handling videos that span extended spatial and temporal scales. Moreover, we conduct an in-depth case study in which we develop a prototype travel-planning agent that leverages the insights gained from VIR-Bench. The agent's markedly improved itinerary recommendations verify that our evaluation protocol not only benchmarks models effectively but also translates into concrete performance gains in user-facing applications. 

**Abstract (ZH)**: Recent Advances in Multimodal Large Language Models: Bridging the Gap in Long-Distance Travel Understanding with VIR-Bench 

---
# Assistive Decision-Making for Right of Way Navigation at Uncontrolled Intersections 

**Title (ZH)**: 无障碍决策导航在无控制交叉口右转通行辅助决策 

**Authors**: Navya Tiwari, Joseph Vazhaeparampil, Victoria Preston  

**Link**: [PDF](https://arxiv.org/pdf/2509.18407)  

**Abstract**: Uncontrolled intersections account for a significant fraction of roadway crashes due to ambiguous right-of-way rules, occlusions, and unpredictable driver behavior. While autonomous vehicle research has explored uncertainty-aware decision making, few systems exist to retrofit human-operated vehicles with assistive navigation support. We present a driver-assist framework for right-of-way reasoning at uncontrolled intersections, formulated as a Partially Observable Markov Decision Process (POMDP). Using a custom simulation testbed with stochastic traffic agents, pedestrians, occlusions, and adversarial scenarios, we evaluate four decision-making approaches: a deterministic finite state machine (FSM), and three probabilistic planners: QMDP, POMCP, and DESPOT. Results show that probabilistic planners outperform the rule-based baseline, achieving up to 97.5 percent collision-free navigation under partial observability, with POMCP prioritizing safety and DESPOT balancing efficiency and runtime feasibility. Our findings highlight the importance of uncertainty-aware planning for driver assistance and motivate future integration of sensor fusion and environment perception modules for real-time deployment in realistic traffic environments. 

**Abstract (ZH)**: 不受控制的交叉口由于模糊的优先通行权规则、遮挡和不可预测的驾驶行为，占到了相当大的道路碰撞比例。虽然自主车辆研究已经探索了不确定性感知的决策制定，但很少有系统能够为人类操作的车辆提供辅助导航支持。我们提出了一种在不受控制的交叉口进行优先通行权推理的驾驶员辅助框架，该框架被表述为部分可观测马尔可夫决策过程（POMDP）。通过一个自定义的具有随机交通代理、行人、遮挡和对抗场景的模拟测试平台，我们评估了四种决策制定方法：确定性有限状态机（FSM），以及三种概率规划器：QMDP、POMCP和DESPOT。结果显示，概率规划器在部分可观测性下优于基于规则的基线，实现了高达97.5%的无碰撞导航，其中POMCP更侧重于安全性，DESPOT则平衡了效率和运行时可行性。我们的研究结果突显了不确定性感知规划对于驾驶辅助的重要性，并激发了将传感器融合和环境感知模块集成以实现实时部署于现实交通环境中的未来研究动机。 

---
# VLA-LPAF: Lightweight Perspective-Adaptive Fusion for Vision-Language-Action to Enable More Unconstrained Robotic Manipulation 

**Title (ZH)**: VLA-LPAF：轻量级视角自适应融合，以实现更不受约束的机器人操作Manipulation 

**Authors**: Jinyue Bian, Zhaoxing Zhang, Zhengyu Liang, Shiwei Zheng, Shengtao Zhang, Rong Shen, Chen Yang, Anzhou Hou  

**Link**: [PDF](https://arxiv.org/pdf/2509.18183)  

**Abstract**: The Visual-Language-Action (VLA) models can follow text instructions according to visual observations of the surrounding environment. This ability to map multimodal inputs to actions is derived from the training of the VLA model on extensive standard demonstrations. These visual observations captured by third-personal global and in-wrist local cameras are inevitably varied in number and perspective across different environments, resulting in significant differences in the visual features. This perspective heterogeneity constrains the generality of VLA models. In light of this, we first propose the lightweight module VLA-LPAF to foster the perspective adaptivity of VLA models using only 2D data. VLA-LPAF is finetuned using images from a single view and fuses other multiview observations in the latent space, which effectively and efficiently bridge the gap caused by perspective inconsistency. We instantiate our VLA-LPAF framework with the VLA model RoboFlamingo to construct RoboFlamingo-LPAF. Experiments show that RoboFlamingo-LPAF averagely achieves around 8% task success rate improvement on CALVIN, 15% on LIBERO, and 30% on a customized simulation benchmark. We also demonstrate the developed viewadaptive characteristics of the proposed RoboFlamingo-LPAF through real-world tasks. 

**Abstract (ZH)**: 基于视觉-语言-动作（VLA）模型的视角自适应模块VLA-LPAF 

---
# MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents 

**Title (ZH)**: 移动RL：移动GUI代理的在线代理强化学习 

**Authors**: Yifan Xu, Xiao Liu, Xinghan Liu, Jiaqi Fu, Hanchen Zhang, Bohao Jing, Shudan Zhang, Yuting Wang, Wenyi Zhao, Yuxiao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.18119)  

**Abstract**: Building general-purpose graphical user interface (GUI) agents has become increasingly promising with the progress in vision language models. However, developing effective mobile GUI agents with reinforcement learning (RL) remains challenging due to the heavy-tailed distribution of task difficulty and the inefficiency of large-scale environment sampling. We present an online agentic reinforcement learning framework MOBILERL to enhance GUI agents in mobile environments. Its core component is the Difficulty-Adaptive GRPO (ADAGRPO) algorithm. In ADAGRPO, we design difficulty-adaptive positive replay and failure curriculum filtering to adapt the model to different task difficulties. We introduce the shortest path reward adjustment strategy to reshape rewards concerning the task length in multi-turn agentic tasks. Those strategies jointly stabilize RL training, improve sample efficiency, and generate strong performance across diverse mobile apps and tasks. We apply MOBILERL to two open models (Qwen2.5-VL-7B-Instruct and GLM-4.1V-9B-Base). The resultant MOBILERL-9B model achieves state-of-the-art results in terms of success rates on both AndroidWorld (75.8%) and AndroidLab (46.8%). The MOBILERL framework is adopted in the AutoGLM products, and also open-sourced at this https URL. 

**Abstract (ZH)**: 基于视觉语言模型的通用图形用户界面（GUI）代理构建取得了日益显著的进展。然而，利用强化学习（RL）开发高效的移动GUI代理仍具挑战性，原因在于任务难度的长尾分布和大规模环境采样的低效性。我们提出了一种在线代理强化学习框架MOBILERL，以增强移动环境中的GUI代理。其核心组件是难度自适应GRPO（ADAGRPO）算法。在ADAGRPO中，我们设计了难度自适应正强化回放和失败课程筛选，以使模型适应不同任务难度。我们引入了最短路径奖励调整策略，以根据不同任务长度重新塑造多轮代理任务的奖励。这些策略共同稳定了RL的训练，提高了样本效率，并在多种移动应用和任务中产生了优越表现。我们将在Qwen2.5-VL-7B-Instruct和GLM-4.1V-9B-Base两个开源模型上应用MOBILERL。由此产生的MOBILERL-9B模型在AndroidWorld（75.8%）和AndroidLab（46.8%）成功率达到领先水平。MOBILERL框架已被应用于AutoGLM产品，并在此链接中开源：[https://github.com/AutoGLM/MOBILERL]。 

---
