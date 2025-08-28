# HERMES: Human-to-Robot Embodied Learning from Multi-Source Motion Data for Mobile Dexterous Manipulation 

**Title (ZH)**: HERMES: 基于多源运动数据的机器人手臂灵巧操作的人机 embodied 学习 

**Authors**: Zhecheng Yuan, Tianming Wei, Langzhe Gu, Pu Hua, Tianhai Liang, Yuanpei Chen, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.20085)  

**Abstract**: Leveraging human motion data to impart robots with versatile manipulation skills has emerged as a promising paradigm in robotic manipulation. Nevertheless, translating multi-source human hand motions into feasible robot behaviors remains challenging, particularly for robots equipped with multi-fingered dexterous hands characterized by complex, high-dimensional action spaces. Moreover, existing approaches often struggle to produce policies capable of adapting to diverse environmental conditions. In this paper, we introduce HERMES, a human-to-robot learning framework for mobile bimanual dexterous manipulation. First, HERMES formulates a unified reinforcement learning approach capable of seamlessly transforming heterogeneous human hand motions from multiple sources into physically plausible robotic behaviors. Subsequently, to mitigate the sim2real gap, we devise an end-to-end, depth image-based sim2real transfer method for improved generalization to real-world scenarios. Furthermore, to enable autonomous operation in varied and unstructured environments, we augment the navigation foundation model with a closed-loop Perspective-n-Point (PnP) localization mechanism, ensuring precise alignment of visual goals and effectively bridging autonomous navigation and dexterous manipulation. Extensive experimental results demonstrate that HERMES consistently exhibits generalizable behaviors across diverse, in-the-wild scenarios, successfully performing numerous complex mobile bimanual dexterous manipulation tasks. Project Page:https:/gemcollector.github.io/HERMES/. 

**Abstract (ZH)**: 利用人类运动数据赋予机器人多功能操作技能已成为机器人操作领域的一种有前途的范式。然而，将多源人类手部运动转化为可行的机器人行为仍然具有挑战性，尤其是对于配备多指灵巧手的机器人，这些手具有复杂的高维动作空间。此外，现有方法往往难以生成能够适应各种环境条件的策略。本文介绍了HERMES，一种用于移动双臂灵巧操作的人机学习框架。首先，HERMES 构建了一种统一的强化学习方法，能够无缝地将来自多个源的异构人类手部运动转化为物理上合理的机器人行为。随后，为减轻仿真到真实世界的差距，我们设计了一种端到端的基于深度图像的仿真到现实世界转移方法，以提高在实际场景中的泛化能力。此外，为在多变且未结构化的环境中实现自主操作，我们增加了闭_loop Perspective-n-Point (PnP) 定位机制，确保视觉目标的精确对齐，并有效地将自主导航和灵巧操作连接起来。广泛的实验结果表明，HERMES 在多种多样的真实场景中表现出可泛化的行为，成功执行了多个复杂的移动双臂灵巧操作任务。项目页面：https://gemcollector.github.io/HERMES/。 

---
# Visio-Verbal Teleimpedance Interface: Enabling Semi-Autonomous Control of Physical Interaction via Eye Tracking and Speech 

**Title (ZH)**: 视觉-言语遥阻接口：通过眼动追踪和 speech 实现物理交互的半自主控制 

**Authors**: Henk H.A. Jekel, Alejandro Díaz Rosales, Luka Peternel  

**Link**: [PDF](https://arxiv.org/pdf/2508.20037)  

**Abstract**: The paper presents a visio-verbal teleimpedance interface for commanding 3D stiffness ellipsoids to the remote robot with a combination of the operator's gaze and verbal interaction. The gaze is detected by an eye-tracker, allowing the system to understand the context in terms of what the operator is currently looking at in the scene. Along with verbal interaction, a Visual Language Model (VLM) processes this information, enabling the operator to communicate their intended action or provide corrections. Based on these inputs, the interface can then generate appropriate stiffness matrices for different physical interaction actions. To validate the proposed visio-verbal teleimpedance interface, we conducted a series of experiments on a setup including a Force Dimension Sigma.7 haptic device to control the motion of the remote Kuka LBR iiwa robotic arm. The human operator's gaze is tracked by Tobii Pro Glasses 2, while human verbal commands are processed by a VLM using GPT-4o. The first experiment explored the optimal prompt configuration for the interface. The second and third experiments demonstrated different functionalities of the interface on a slide-in-the-groove task. 

**Abstract (ZH)**: 基于视线与口头交互的远程机器人触觉接口：命令远程Kuka LBR iiwa机械臂的3D刚度椭球体 

---
# Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation 

**Title (ZH)**: 长时视语行动模型：unlocking 机器人 manipulatin 长视角能力 

**Authors**: Yiguo Fan, Pengxiang Ding, Shuanghao Bai, Xinyang Tong, Yuyang Zhu, Hongchao Lu, Fengqi Dai, Wei Zhao, Yang Liu, Siteng Huang, Zhaoxin Fan, Badong Chen, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.19958)  

**Abstract**: Vision-Language-Action (VLA) models have become a cornerstone in robotic policy learning, leveraging large-scale multimodal data for robust and scalable control. However, existing VLA frameworks primarily address short-horizon tasks, and their effectiveness on long-horizon, multi-step robotic manipulation remains limited due to challenges in skill chaining and subtask dependencies. In this work, we introduce Long-VLA, the first end-to-end VLA model specifically designed for long-horizon robotic tasks. Our approach features a novel phase-aware input masking strategy that adaptively segments each subtask into moving and interaction phases, enabling the model to focus on phase-relevant sensory cues and enhancing subtask compatibility. This unified strategy preserves the scalability and data efficiency of VLA training, and our architecture-agnostic module can be seamlessly integrated into existing VLA models. We further propose the L-CALVIN benchmark to systematically evaluate long-horizon manipulation. Extensive experiments on both simulated and real-world tasks demonstrate that Long-VLA significantly outperforms prior state-of-the-art methods, establishing a new baseline for long-horizon robotic control. 

**Abstract (ZH)**: 长时视觉-语言-行动（Long-VLA）模型：专为长时 horizon 机器人任务设计的端到端视觉-语言-行动模型 

---
# Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors 

**Title (ZH)**: 分而治之，发现与部署：带有对称性和风格先验的分解技能学习 

**Authors**: Rafael Cathomen, Mayank Mittal, Marin Vlastelica, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2508.19953)  

**Abstract**: Unsupervised Skill Discovery (USD) allows agents to autonomously learn diverse behaviors without task-specific rewards. While recent USD methods have shown promise, their application to real-world robotics remains underexplored. In this paper, we propose a modular USD framework to address the challenges in the safety, interpretability, and deployability of the learned skills. Our approach employs user-defined factorization of the state space to learn disentangled skill representations. It assigns different skill discovery algorithms to each factor based on the desired intrinsic reward function. To encourage structured morphology-aware skills, we introduce symmetry-based inductive biases tailored to individual factors. We also incorporate a style factor and regularization penalties to promote safe and robust behaviors. We evaluate our framework in simulation using a quadrupedal robot and demonstrate zero-shot transfer of the learned skills to real hardware. Our results show that factorization and symmetry lead to the discovery of structured human-interpretable behaviors, while the style factor and penalties enhance safety and diversity. Additionally, we show that the learned skills can be used for downstream tasks and perform on par with oracle policies trained with hand-crafted rewards. 

**Abstract (ZH)**: 无监督技能发现（USD）使代理能够在没有任务特定奖励的情况下自主学习多样化的行为。尽管最近的USD方法展示出潜力，但它们在实际机器人领域的应用仍缺乏探索。本文提出了一种模块化的USD框架，以解决学习技能的安全性、可解释性和部署性挑战。我们的方法通过用户定义的状态空间分解来学习解析的技能表示，并根据期望的固有奖励函数将不同的技能发现算法分配给每个因素。为了促进结构化的形态感知技能，我们引入了针对各个因素的对称性诱导偏置。我们还引入了风格因素和正则化惩罚项，以促进安全性和鲁棒性行为。我们在四足机器人仿真中评估了我们的框架，并展示了所学习技能在实际硬件上的零样本迁移。我们的结果显示，分解和对称性导致结构化的、人类可解释的行为的发现，而风格因素和惩罚项增强了安全性和多样性。此外，我们展示了所学习的技能可以在下游任务中使用，并与使用手工制作奖励训练的先验策略性能相当。 

---
# FARM: Frame-Accelerated Augmentation and Residual Mixture-of-Experts for Physics-Based High-Dynamic Humanoid Control 

**Title (ZH)**: FARM：基于框架加速扩增和残差专家混合的物理驱动高动态人形机器人控制 

**Authors**: Tan Jing, Shiting Chen, Yangfan Li, Weisheng Xu, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.19926)  

**Abstract**: Unified physics-based humanoid controllers are pivotal for robotics and character animation, yet models that excel on gentle, everyday motions still stumble on explosive actions, hampering real-world deployment. We bridge this gap with FARM (Frame-Accelerated Augmentation and Residual Mixture-of-Experts), an end-to-end framework composed of frame-accelerated augmentation, a robust base controller, and a residual mixture-of-experts (MoE). Frame-accelerated augmentation exposes the model to high-velocity pose changes by widening inter-frame gaps. The base controller reliably tracks everyday low-dynamic motions, while the residual MoE adaptively allocates additional network capacity to handle challenging high-dynamic actions, significantly enhancing tracking accuracy. In the absence of a public benchmark, we curate the High-Dynamic Humanoid Motion (HDHM) dataset, comprising 3593 physically plausible clips. On HDHM, FARM reduces the tracking failure rate by 42.8\% and lowers global mean per-joint position error by 14.6\% relative to the baseline, while preserving near-perfect accuracy on low-dynamic motions. These results establish FARM as a new baseline for high-dynamic humanoid control and introduce the first open benchmark dedicated to this challenge. The code and dataset will be released at this https URL. 

**Abstract (ZH)**: 基于统一物理的人形控制器对于机器人学和角色动画至关重要，但擅长柔和日常动作的模型仍会在爆炸性动作中遭遇困难，阻碍了其实际部署。我们通过FARM（帧加速增强和残差混合专家）框架桥接了这一差距，FARM是一个端到端框架，包括帧加速增强、稳健的基础控制器和残差混合专家（MoE）。帧加速增强通过扩大帧间隔使模型接触高速姿态变化。基础控制器可靠地追踪低动态的日常动作，而残差MoE自适应分配额外的网络容量以处理挑战性的高动态动作，显著提高了追踪准确性。在缺乏公开基准的情况下，我们构建了High-Dynamic Humanoid Motion (HDHM)数据集，包含3593个物理上合理的片段。在HDHM上，FARM将跟踪失败率降低了42.8%，全局平均每个关节位置误差降低了14.6%，同时在低动态动作上保持近乎完美的准确性。这些结果确立了FARM作为高动态人形控制的新基准，并引入了针对该挑战的第一个公开基准。代码和数据集将在以下链接发布：此https链接。 

---
# Context-Aware Risk Estimation in Home Environments: A Probabilistic Framework for Service Robots 

**Title (ZH)**: 家庭环境中的情境感知风险估计：服务机器人的一种概率框架 

**Authors**: Sena Ishii, Akash Chikhalikar, Ankit A. Ravankar, Jose Victorio Salazar Luces, Yasuhisa Hirata  

**Link**: [PDF](https://arxiv.org/pdf/2508.19788)  

**Abstract**: We present a novel framework for estimating accident-prone regions in everyday indoor scenes, aimed at improving real-time risk awareness in service robots operating in human-centric environments. As robots become integrated into daily life, particularly in homes, the ability to anticipate and respond to environmental hazards is crucial for ensuring user safety, trust, and effective human-robot interaction. Our approach models object-level risk and context through a semantic graph-based propagation algorithm. Each object is represented as a node with an associated risk score, and risk propagates asymmetrically from high-risk to low-risk objects based on spatial proximity and accident relationship. This enables the robot to infer potential hazards even when they are not explicitly visible or labeled. Designed for interpretability and lightweight onboard deployment, our method is validated on a dataset with human-annotated risk regions, achieving a binary risk detection accuracy of 75%. The system demonstrates strong alignment with human perception, particularly in scenes involving sharp or unstable objects. These results underline the potential of context-aware risk reasoning to enhance robotic scene understanding and proactive safety behaviors in shared human-robot spaces. This framework could serve as a foundation for future systems that make context-driven safety decisions, provide real-time alerts, or autonomously assist users in avoiding or mitigating hazards within home environments. 

**Abstract (ZH)**: 一种基于语义图传播的室内场景易事故区域估计框架：提高人本环境中服务机器人实时风险意识 

---
# Embodied Intelligence for Sustainable Flight: A Soaring Robot with Active Morphological Control 

**Title (ZH)**: 可持续飞行中的本体智能：具有主动形态控制的滑翔机器人 

**Authors**: Ghadeer Elmkaiel, Syn Schmitt, Michael Muehlebach  

**Link**: [PDF](https://arxiv.org/pdf/2508.19684)  

**Abstract**: Achieving both agile maneuverability and high energy efficiency in aerial robots, particularly in dynamic wind environments, remains challenging. Conventional thruster-powered systems offer agility but suffer from high energy consumption, while fixed-wing designs are efficient but lack hovering and maneuvering capabilities. We present Floaty, a shape-changing robot that overcomes these limitations by passively soaring, harnessing wind energy through intelligent morphological control inspired by birds. Floaty's design is optimized for passive stability, and its control policy is derived from an experimentally learned aerodynamic model, enabling precise attitude and position control without active propulsion. Wind tunnel experiments demonstrate Floaty's ability to hover, maneuver, and reject disturbances in vertical airflows up to 10 m/s. Crucially, Floaty achieves this with a specific power consumption of 10 W/kg, an order of magnitude lower than thruster-powered systems. This introduces a paradigm for energy-efficient aerial robotics, leveraging morphological intelligence and control to operate sustainably in challenging wind conditions. 

**Abstract (ZH)**: 在动态风环境中的Both agile maneuverability and high energy efficiency兼具敏捷机动性和高能效性的空中机器人：Floaty的设计与实现 

---
# Impedance Primitive-augmented Hierarchical Reinforcement Learning for Sequential Tasks 

**Title (ZH)**: 阻抗增强层次强化学习用于序列任务 

**Authors**: Amin Berjaoui Tahmaz, Ravi Prakash, Jens Kober  

**Link**: [PDF](https://arxiv.org/pdf/2508.19607)  

**Abstract**: This paper presents an Impedance Primitive-augmented hierarchical reinforcement learning framework for efficient robotic manipulation in sequential contact tasks. We leverage this hierarchical structure to sequentially execute behavior primitives with variable stiffness control capabilities for contact tasks. Our proposed approach relies on three key components: an action space enabling variable stiffness control, an adaptive stiffness controller for dynamic stiffness adjustments during primitive execution, and affordance coupling for efficient exploration while encouraging compliance. Through comprehensive training and evaluation, our framework learns efficient stiffness control capabilities and demonstrates improvements in learning efficiency, compositionality in primitive selection, and success rates compared to the state-of-the-art. The training environments include block lifting, door opening, object pushing, and surface cleaning. Real world evaluations further confirm the framework's sim2real capability. This work lays the foundation for more adaptive and versatile robotic manipulation systems, with potential applications in more complex contact-based tasks. 

**Abstract (ZH)**: 一种增强阻抗本征的分层强化学习框架：用于序贯接触任务的高效机器人操作 

---
# A Lightweight Crowd Model for Robot Social Navigation 

**Title (ZH)**: 轻量级人群模型用于机器人社会导航 

**Authors**: Maryam Kazemi Eskeri, Thomas Wiedemann, Ville Kyrki, Dominik Baumann, Tomasz Piotr Kucner  

**Link**: [PDF](https://arxiv.org/pdf/2508.19595)  

**Abstract**: Robots operating in human-populated environments must navigate safely and efficiently while minimizing social disruption. Achieving this requires estimating crowd movement to avoid congested areas in real-time. Traditional microscopic models struggle to scale in dense crowds due to high computational cost, while existing macroscopic crowd prediction models tend to be either overly simplistic or computationally intensive. In this work, we propose a lightweight, real-time macroscopic crowd prediction model tailored for human motion, which balances prediction accuracy and computational efficiency. Our approach simplifies both spatial and temporal processing based on the inherent characteristics of pedestrian flow, enabling robust generalization without the overhead of complex architectures. We demonstrate a 3.6 times reduction in inference time, while improving prediction accuracy by 3.1 %. Integrated into a socially aware planning framework, the model enables efficient and socially compliant robot navigation in dynamic environments. This work highlights that efficient human crowd modeling enables robots to navigate dense environments without costly computations. 

**Abstract (ZH)**: 在人类居住环境中操作的机器人必须在避免拥堵区域的同时安全高效地导航，并最小化社交干扰。实现这一目标需要在实时情况下估计人群移动。传统的微观模型由于计算成本高难以在密集人群中扩展，而现有的宏观人群预测模型要么过于简化，要么计算成本高。在这项工作中，我们提出了一种轻量级、实时的宏观人群预测模型，该模型适用于人类运动，平衡了预测准确性和计算效率。我们的方法基于行人流量的本质特征简化了空间和时间处理，使模型能够在不使用复杂架构的情况下实现稳健的泛化。我们展示了推理时间减少3.6倍，同时预测准确率提高3.1%的结果。将该模型集成到一个社交意识规划框架中，能够使机器人在动态环境中高效且符合社交规范地导航。这项工作强调了有效的人群建模能够使机器人在无需昂贵计算的情况下导航密集环境。 

---
# LaVA-Man: Learning Visual Action Representations for Robot Manipulation 

**Title (ZH)**: LaVA-Man: 学习视觉动作表示用于机器人操作 

**Authors**: Chaoran Zhu, Hengyi Wang, Yik Lung Pang, Changjae Oh  

**Link**: [PDF](https://arxiv.org/pdf/2508.19391)  

**Abstract**: Visual-textual understanding is essential for language-guided robot manipulation. Recent works leverage pre-trained vision-language models to measure the similarity between encoded visual observations and textual instructions, and then train a model to map this similarity to robot actions. However, this two-step approach limits the model to capture the relationship between visual observations and textual instructions, leading to reduced precision in manipulation tasks. We propose to learn visual-textual associations through a self-supervised pretext task: reconstructing a masked goal image conditioned on an input image and textual instructions. This formulation allows the model to learn visual-action representations without robot action supervision. The learned representations can then be fine-tuned for manipulation tasks with only a few demonstrations. We also introduce the \textit{Omni-Object Pick-and-Place} dataset, which consists of annotated robot tabletop manipulation episodes, including 180 object classes and 3,200 instances with corresponding textual instructions. This dataset enables the model to acquire diverse object priors and allows for a more comprehensive evaluation of its generalisation capability across object instances. Experimental results on the five benchmarks, including both simulated and real-robot validations, demonstrate that our method outperforms prior art. 

**Abstract (ZH)**: 视觉-文本理解对于语言引导的机器人操作至关重要。最近的工作利用预训练的视觉-语言模型衡量编码视觉观察与文本指令之间的相似度，然后训练一个模型将这种相似度映射到机器人动作。然而，这种两步 approach 限制了模型捕捉视觉观察与文本指令之间的关系，导致操作任务精度降低。我们建议通过一个自我监督的前置任务来学习视觉-文本关联：在输入图像和文本指令的条件下重建被掩码的目标图像。这种表述允许模型在无需机器人动作监督的情况下学习视觉-动作表示。学习到的表示随后可以通过少量示范进行微调，以适应操作任务。我们还引入了 Omni-Object Pick-and-Place 数据集，该数据集包含标注的机器人桌面操作片段，包括180类物体和3,200个带有相应文本指令的实例。该数据集使模型能够获得多样化的物体先验，从而允许对其在不同物体实例上的泛化能力进行更全面的评估。在五个基准测试上的实验结果，包括模拟和真实机器人验证，证明了我们方法优于先前的工作。 

---
# Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies 

**Title (ZH)**: 离散扩散VLA：将离散扩散应用于视觉-语言-动作策略中的动作解码 

**Authors**: Zhixuan Liang, Yizhuo Li, Tianshuo Yang, Chengyue Wu, Sitong Mao, Liuao Pei, Xiaokang Yang, Jiangmiao Pang, Yao Mu, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.20072)  

**Abstract**: Vision-Language-Action (VLA) models adapt large vision-language backbones to map images and instructions to robot actions. However, prevailing VLA decoders either generate actions autoregressively in a fixed left-to-right order or attach continuous diffusion or flow matching heads outside the backbone, demanding specialized training and iterative sampling that hinder a unified, scalable architecture. We present Discrete Diffusion VLA, a single-transformer policy that models discretized action chunks with discrete diffusion and is trained with the same cross-entropy objective as the VLM backbone. The design retains diffusion's progressive refinement paradigm while remaining natively compatible with the discrete token interface of VLMs. Our method achieves an adaptive decoding order that resolves easy action elements before harder ones and uses secondary remasking to revisit uncertain predictions across refinement rounds, which improves consistency and enables robust error correction. This unified decoder preserves pretrained vision language priors, supports parallel decoding, breaks the autoregressive bottleneck, and reduces the number of function evaluations. Discrete Diffusion VLA achieves 96.3% avg. SR on LIBERO, 71.2% visual matching on SimplerEnv Fractal and 49.3% overall on SimplerEnv Bridge, improving over both autoregressive and continuous diffusion baselines. These findings indicate that discrete-diffusion action decoder supports precise action modeling and consistent training, laying groundwork for scaling VLA to larger models and datasets. 

**Abstract (ZH)**: 离散扩散视觉-语言-动作模型 

---
# Flocking Behavior: An Innovative Inspiration for the Optimization of Production Plants 

**Title (ZH)**: 群体行为：一种创新的生产工厂优化灵感 

**Authors**: M. Umlauft, M. Schranz  

**Link**: [PDF](https://arxiv.org/pdf/2508.19963)  

**Abstract**: Optimizing modern production plants using the job-shop principle is a known hard problem. For very large plants, like semiconductor fabs, the problem becomes unsolvable on a plant-wide scale in a reasonable amount of time using classical linear optimization. An alternative approach is the use of swarm intelligence algorithms. These have been applied to the job-shop problem before, but often in a centrally calculated way where they are applied to the solution space, but they can be implemented in a bottom-up fashion to avoid global result computation as well. One of the problems in semiconductor production is that the production process requires a lot of switching between machines that process lots one after the other and machines that process batches of lots at once, often with long processing times. In this paper, we address this switching problem with the ``boids'' flocking algorithm that was originally used in robotics and movie industry. The flocking behavior is a bio-inspired algorithm that uses only local information and interaction based on simple heuristics. We show that this algorithm addresses these valid considerations in production plant optimization, as it reacts to the switching of machine kinds similar to how a swarm of flocking animals would react to obstacles in its course. 

**Abstract (ZH)**: 使用鸟群算法解决半导体生产中的切换问题：一种基于局部信息的优化方法 

---
# InquireMobile: Teaching VLM-based Mobile Agent to Request Human Assistance via Reinforcement Fine-Tuning 

**Title (ZH)**: InquireMobile: 通过强化微调教学基于VLM的移动代理请求人类协助 

**Authors**: Qihang Ai, Pi Bu, Yue Cao, Yingyao Wang, Jihao Gu, Jingxuan Xing, Zekun Zhu, Wei Jiang, Zhicheng Zheng, Jun Song, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.19679)  

**Abstract**: Recent advances in Vision-Language Models (VLMs) have enabled mobile agents to perceive and interact with real-world mobile environments based on human instructions. However, the current fully autonomous paradigm poses potential safety risks when model understanding or reasoning capabilities are insufficient. To address this challenge, we first introduce \textbf{InquireBench}, a comprehensive benchmark specifically designed to evaluate mobile agents' capabilities in safe interaction and proactive inquiry with users, encompassing 5 categories and 22 sub-categories, where most existing VLM-based agents demonstrate near-zero performance. In this paper, we aim to develop an interactive system that actively seeks human confirmation at critical decision points. To achieve this, we propose \textbf{InquireMobile}, a novel model inspired by reinforcement learning, featuring a two-stage training strategy and an interactive pre-action reasoning mechanism. Finally, our model achieves an 46.8% improvement in inquiry success rate and the best overall success rate among existing baselines on InquireBench. We will open-source all datasets, models, and evaluation codes to facilitate development in both academia and industry. 

**Abstract (ZH)**: 近期视觉-语言模型（VLMs）的进展使移动代理能够基于人类指令感知和与实际移动环境互动，但当前的完全自主范式在模型理解和推理能力不足时可能带来潜在的安全风险。为应对这一挑战，我们首先介绍了InquireBench，一个全面的基准，专门用于评估移动代理在安全互动和主动问询用户方面的能力，包含5个类别和22个子类别，而大多数现有的基于VLM的代理在其性能方面接近于零。本文旨在开发一个交互系统，在关键决策点积极寻求人类确认。为此，我们提出了InquireMobile，一个受强化学习启发的新颖模型，具有两阶段训练策略和交互式预操作推理机制。最终，我们的模型在InquireBench上的询求成功率提高了46.8%，并在现有基线中的综合成功率方面表现最好。我们将会开源所有数据集、模型和评估代码，以促进学术界和行业的发展。 

---
# Democracy-in-Silico: Institutional Design as Alignment in AI-Governed Polities 

**Title (ZH)**: 硅基民主：人工智能治理政治体中的制度设计与对齐 

**Authors**: Trisanth Srinivasan, Santosh Patapati  

**Link**: [PDF](https://arxiv.org/pdf/2508.19562)  

**Abstract**: This paper introduces Democracy-in-Silico, an agent-based simulation where societies of advanced AI agents, imbued with complex psychological personas, govern themselves under different institutional frameworks. We explore what it means to be human in an age of AI by tasking Large Language Models (LLMs) to embody agents with traumatic memories, hidden agendas, and psychological triggers. These agents engage in deliberation, legislation, and elections under various stressors, such as budget crises and resource scarcity. We present a novel metric, the Power-Preservation Index (PPI), to quantify misaligned behavior where agents prioritize their own power over public welfare. Our findings demonstrate that institutional design, specifically the combination of a Constitutional AI (CAI) charter and a mediated deliberation protocol, serves as a potent alignment mechanism. These structures significantly reduce corrupt power-seeking behavior, improve policy stability, and enhance citizen welfare compared to less constrained democratic models. The simulation reveals that an institutional design may offer a framework for aligning the complex, emergent behaviors of future artificial agent societies, forcing us to reconsider what human rituals and responsibilities are essential in an age of shared authorship with non-human entities. 

**Abstract (ZH)**: 基于硅民主的先进AI代理社会仿真：探索人工智能时代的人性含义及机构设计优化 

---
# CODA: Coordinating the Cerebrum and Cerebellum for a Dual-Brain Computer Use Agent with Decoupled Reinforcement Learning 

**Title (ZH)**: CODA: 调和大脑与小脑的双脑计算机使用代理基于解耦强化学习 

**Authors**: Zeyi Sun, Yuhang Cao, Jianze Liang, Qiushi Sun, Ziyu Liu, Zhixiong Zhang, Yuhang Zang, Xiaoyi Dong, Kai Chen, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.20096)  

**Abstract**: Autonomous agents for Graphical User Interfaces (GUIs) face significant challenges in specialized domains such as scientific computing, where both long-horizon planning and precise execution are required. Existing approaches suffer from a trade-off: generalist agents excel at planning but perform poorly in execution, while specialized agents demonstrate the opposite weakness. Recent compositional frameworks attempt to bridge this gap by combining a planner and an actor, but they are typically static and non-trainable, which prevents adaptation from experience. This is a critical limitation given the scarcity of high-quality data in scientific domains. To address these limitations, we introduce CODA, a novel and trainable compositional framework that integrates a generalist planner (Cerebrum) with a specialist executor (Cerebellum), trained via a dedicated two-stage pipeline. In the first stage, Specialization, we apply a decoupled GRPO approach to train an expert planner for each scientific application individually, bootstrapping from a small set of task trajectories. In the second stage, Generalization, we aggregate all successful trajectories from the specialized experts to build a consolidated dataset, which is then used for supervised fine-tuning of the final planner. This equips CODA with both robust execution and cross-domain generalization. Evaluated on four challenging applications from the ScienceBoard benchmark, CODA significantly outperforms baselines and establishes a new state of the art among open-source models. 

**Abstract (ZH)**: 自主智能体在图形用户界面（GUIs）领域的专门领域（如科学计算）中面临显著挑战，需要进行长时间规划和精确执行。现有方法存在权衡：通用智能体擅长规划但在执行方面表现不佳，而专门智能体则相反。最近的组合框架通过结合规划者和执行者试图弥合这一差距，但它们通常静态且不可训练，这阻碍了从经验中进行适应。鉴于科学领域高质量数据的稀缺性，这是关键的限制。为应对这些限制，我们引入了CODA，这是一种新颖且可训练的组合框架，将通用规划者（Cerebrum）与专门执行者（Cerebellum）相结合，并通过专用的两阶段管道进行训练。在第一阶段“专门化”中，我们应用解耦的GRPO方法独立训练每个科学应用的专家规划者，并从少量的任务轨迹起步。在第二阶段“泛化”中，我们将所有成功的轨迹聚合起来构建一个综合数据集，然后使用该数据集对最终规划者进行监督微调。这使CODA具备了稳健的执行能力和跨域泛化能力。在ScienceBoard基准测试中的四个具有挑战性的应用上，CODA显著优于基线并建立了开源模型的新状态最先进水平。 

---
# Generative AI for Testing of Autonomous Driving Systems: A Survey 

**Title (ZH)**: 自动驾驶系统测试中的生成式人工智能：一个综述 

**Authors**: Qunying Song, He Ye, Mark Harman, Federica Sarro  

**Link**: [PDF](https://arxiv.org/pdf/2508.19882)  

**Abstract**: Autonomous driving systems (ADS) have been an active area of research, with the potential to deliver significant benefits to society. However, before large-scale deployment on public roads, extensive testing is necessary to validate their functionality and safety under diverse driving conditions. Therefore, different testing approaches are required, and achieving effective and efficient testing of ADS remains an open challenge. Recently, generative AI has emerged as a powerful tool across many domains, and it is increasingly being applied to ADS testing due to its ability to interpret context, reason about complex tasks, and generate diverse outputs. To gain a deeper understanding of its role in ADS testing, we systematically analyzed 91 relevant studies and synthesized their findings into six major application categories, primarily centered on scenario-based testing of ADS. We also reviewed their effectiveness and compiled a wide range of datasets, simulators, ADS, metrics, and benchmarks used for evaluation, while identifying 27 limitations. This survey provides an overview and practical insights into the use of generative AI for testing ADS, highlights existing challenges, and outlines directions for future research in this rapidly evolving field. 

**Abstract (ZH)**: 自主驾驶系统（ADS）一直是研究的活跃领域，有潜力为社会带来显著益处。但在大规模部署到公共道路之前，需要进行广泛的测试以验证其在多种驾驶条件下功能和安全性的有效性。因此，不同的测试方法是必需的，而实现有效且高效的ADS测试仍是一个开放性的挑战。最近，生成式AI在许多领域展现出强大的能力，并因其能够解释上下文、推理复杂任务和生成多样化输出，被越来越多地应用于ADS测试。为了更深入地了解其在ADS测试中的作用，我们系统分析了91篇相关研究，并将 findings 精要概括为六类主要应用场景，重点在于基于场景的ADS测试。我们还回顾了这些研究的有效性，并汇集了广泛的数据集、模拟器、ADS、评估指标和基准，同时指出了27个局限性。该综述提供了一种关于生成式AI在ADS测试中应用的概览和实用见解，突出了现有挑战，并指出了这一快速发展的领域未来研究的方向。 

---
# Complementary Learning System Empowers Online Continual Learning of Vehicle Motion Forecasting in Smart Cities 

**Title (ZH)**: 互补学习系统赋能智能城市中车辆运动预测的在线连续学习 

**Authors**: Zirui Li, Yunlong Lin, Guodong Du, Xiaocong Zhao, Cheng Gong, Chen Lv, Chao Lu, Jianwei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2508.19597)  

**Abstract**: Artificial intelligence underpins most smart city services, yet deep neural network (DNN) that forecasts vehicle motion still struggle with catastrophic forgetting, the loss of earlier knowledge when models are updated. Conventional fixes enlarge the training set or replay past data, but these strategies incur high data collection costs, sample inefficiently and fail to balance long- and short-term experience, leaving them short of human-like continual learning. Here we introduce Dual-LS, a task-free, online continual learning paradigm for DNN-based motion forecasting that is inspired by the complementary learning system of the human brain. Dual-LS pairs two synergistic memory rehearsal replay mechanisms to accelerate experience retrieval while dynamically coordinating long-term and short-term knowledge representations. Tests on naturalistic data spanning three countries, over 772,000 vehicles and cumulative testing mileage of 11,187 km show that Dual-LS mitigates catastrophic forgetting by up to 74.31\% and reduces computational resource demand by up to 94.02\%, markedly boosting predictive stability in vehicle motion forecasting without inflating data requirements. Meanwhile, it endows DNN-based vehicle motion forecasting with computation efficient and human-like continual learning adaptability fit for smart cities. 

**Abstract (ZH)**: 基于反演记忆机制的双通道连续学习框架减轻灾难性遗忘并提升智能城市中车辆运动预测的计算效率和类人适应性 

---
# AT-CXR: Uncertainty-Aware Agentic Triage for Chest X-rays 

**Title (ZH)**: AT-CXR：基于不确定性感知的胸部X光分诊方法 

**Authors**: Xueyang Li, Mingze Jiang, Gelei Xu, Jun Xia, Mengzhao Jia, Danny Chen, Yiyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.19322)  

**Abstract**: Agentic AI is advancing rapidly, yet truly autonomous medical-imaging triage, where a system decides when to stop, escalate, or defer under real constraints, remains relatively underexplored. To address this gap, we introduce AT-CXR, an uncertainty-aware agent for chest X-rays. The system estimates per-case confidence and distributional fit, then follows a stepwise policy to issue an automated decision or abstain with a suggested label for human intervention. We evaluate two router designs that share the same inputs and actions: a deterministic rule-based router and an LLM-decided router. Across five-fold evaluation on a balanced subset of NIH ChestX-ray14 dataset, both variants outperform strong zero-shot vision-language models and state-of-the-art supervised classifiers, achieving higher full-coverage accuracy and superior selective-prediction performance, evidenced by a lower area under the risk-coverage curve (AURC) and a lower error rate at high coverage, while operating with lower latency that meets practical clinical constraints. The two routers provide complementary operating points, enabling deployments to prioritize maximal throughput or maximal accuracy. Our code is available at this https URL. 

**Abstract (ZH)**: 代理型AI快速发展，但真正自主的医学影像分诊（系统在实际约束下决定停止、升级或延后）仍相对未被充分探索。为解决这一问题，我们引入了AT-CXR，这是一种具有不确定性意识的胸片处理代理系统。该系统估计每例案件的置信度和分布拟合度，然后遵循逐步策略，要么发布自动化决策，要么在建议标签下保持中立等待人工干预。我们评估了两种路由器设计，它们共享相同的输入和动作：确定性规则Based路由器和基于LLM的路由器。在对NIH ChestX-ray14数据集平衡子集进行五折评估中，两种变体均优于强大的零样本视觉语言模型和最先进的监督分类器，实现了更高的全面准确性并表现出更优的选择性预测性能，这体现在较低的风险覆盖面积下的曲线下面积（AURC）以及在高覆盖率下的较低错误率，同时满足实际临床约束下的较低延迟要求。这两种路由器提供了互补的操作点，使部署能够优先考虑最大吞吐量或最大准确性。我们的代码可在以下链接获取。 

---
# (DEMO) Deep Reinforcement Learning Based Resource Allocation in Distributed IoT Systems 

**Title (ZH)**: 基于深度强化学习的分布式物联网系统资源分配方法 

**Authors**: Aohan Li, Miyu Tsuzuki  

**Link**: [PDF](https://arxiv.org/pdf/2508.19318)  

**Abstract**: Deep Reinforcement Learning (DRL) has emerged as an efficient approach to resource allocation due to its strong capability in handling complex decision-making tasks. However, only limited research has explored the training of DRL models with real-world data in practical, distributed Internet of Things (IoT) systems. To bridge this gap, this paper proposes a novel framework for training DRL models in real-world distributed IoT environments. In the proposed framework, IoT devices select communication channels using a DRL-based method, while the DRL model is trained with feedback information. Specifically, Acknowledgment (ACK) information is obtained from actual data transmissions over the selected channels. Implementation and performance evaluation, in terms of Frame Success Rate (FSR), are carried out, demonstrating both the feasibility and the effectiveness of the proposed framework. 

**Abstract (ZH)**: 深度强化学习（DRL）已在资源分配中展现出高效的方法，因其在处理复杂决策任务方面的强大能力。然而，仅有有限的研究探索了在实用的分布式物联网（IoT）系统中使用真实数据训练DRL模型。为弥补这一差距，本文提出了一种新的框架，用于在实际分布式物联网环境中训练DRL模型。在所提出的框架中，物联网设备使用基于DRL的方法选择通信信道，同时DRL模型通过反馈信息进行训练。具体而言，通过实际数据传输获得确认（ACK）信息。实现了框架的实施并从帧成功率（FSR）的角度进行了性能评估，证明了所提出框架的可行性和有效性。 

---
# Epistemic Trade-Off: An Analysis of the Operational Breakdown and Ontological Limits of "Certainty-Scope" in AI 

**Title (ZH)**: 知识权衡：对“确定性-范围”在AI中的操作失效和本体界限的分析 

**Authors**: Generoso Immediato  

**Link**: [PDF](https://arxiv.org/pdf/2508.19304)  

**Abstract**: Floridi's conjecture offers a compelling intuition about the fundamental trade-off between certainty and scope in artificial intelligence (AI) systems. This exploration remains crucial, not merely as a philosophical exercise, but as a potential compass for guiding AI investments, particularly in safety-critical industrial domains where the level of attention will surely be higher in the future. However, while intellectually coherent, its formalization ultimately freezes this insight into a suspended epistemic truth, resisting operationalization within real-world systems. This paper is a result of an analysis arguing that the conjecture's ambition to provide insights to engineering design and regulatory decision-making is constrained by two critical factors: first, its reliance on incomputable constructs - rendering it practically unactionable and unverifiable; second, its underlying ontological assumption of AI systems as self-contained epistemic entities - separating it from the intricate and dynamic socio-technical environments in which knowledge is co-constructed. We conclude that this dual breakdown - an epistemic closure deficit and an embeddedness bypass - prevents the conjecture from transitioning into a computable and actionable framework suitable for informing the design, deployment, and governance of real-world AI hybrid systems. In response, we propose a contribution to the framing of Floridi's epistemic challenge, addressing the inherent epistemic burdens of AI within complex human-centric domains. 

**Abstract (ZH)**: 福里迪的猜想提供了关于人工智能系统中确定性与范围基本权衡的引人入胜的直觉。这一探索在引导人工智能投资方面仍然至关重要，特别是在今后会更加关注的安全关键工业领域。然而，尽管在概念上是连贯的，其形式化最终将其深刻的见解固化为一种悬置的表徵真理，难以在现实世界的系统中实现操作化。本文是基于分析得出的结论，认为猜想旨在为工程设计和监管决策提供洞见的努力受到两个关键因素的限制：首先，其依赖于不可计算的构造，使其在实践中无法实用和验证；其次，其对人工智能系统的本体论假设为自足的认知实体，使其脱离了知识共同建构的复杂且动态的社会技术环境。我们得出结论认为，这一双重缺陷——认知闭合不足和嵌入性的规避——阻碍了猜想向可用于指导实际人工智能混合系统的计算和可操作框架的转换。对此，我们提出了一种对福里迪的认知挑战的框架性贡献，旨在解决复杂的人类中心领域中人工智能固有的认知负担。 

---
# Towards Production-Worthy Simulation for Autonomous Cyber Operations 

**Title (ZH)**: 面向自主网络操作的生产级仿真研究 

**Authors**: Konur Tholl, Mariam El Mezouar, Ranwa Al Mallah  

**Link**: [PDF](https://arxiv.org/pdf/2508.19278)  

**Abstract**: Simulated environments have proven invaluable in Autonomous Cyber Operations (ACO) where Reinforcement Learning (RL) agents can be trained without the computational overhead of emulation. These environments must accurately represent cybersecurity scenarios while producing the necessary signals to support RL training. In this study, we present a framework where we first extend CybORG's Cage Challenge 2 environment by implementing three new actions: Patch, Isolate, and Unisolate, to better represent the capabilities available to human operators in real-world settings. We then propose a design for agent development where we modify the reward signals and the agent's feature space to enhance training performance. To validate these modifications, we train DQN and PPO agents in the updated environment. Our study demonstrates that CybORG can be extended with additional realistic functionality, while maintaining its ability to generate informative training signals for RL agents. 

**Abstract (ZH)**: 模拟环境在自主网络操作（ACO）中的应用已证明极为宝贵，其中强化学习（RL）代理可以在不需要模拟计算开销的情况下进行训练。这些环境必须准确地代表网络安全场景，并生成支持RL训练所需的信号。在本研究中，我们提出了一种框架，首先通过实现三种新动作（补丁、隔离和解隔离）扩展CybORG的Cage Challenge 2环境，以更好地反映真实世界中人工操作员的可用能力。然后，我们提出了一种代理开发的设计，通过修改奖励信号和代理的特征空间来提升训练性能。为了验证这些修改，我们在更新后的环境中训练了DQN和PPO代理。我们的研究证明，CybORG可以扩展以增加额外的现实功能，同时仍保留生成支持RL代理的训练信号的能力。 

---
