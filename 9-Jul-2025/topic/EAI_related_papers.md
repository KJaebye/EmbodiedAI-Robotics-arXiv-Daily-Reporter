# EC-Flow: Enabling Versatile Robotic Manipulation from Action-Unlabeled Videos via Embodiment-Centric Flow 

**Title (ZH)**: EC-Flow: 通过本体中心流从无动作标签视频中实现 versatile 机器人操作 

**Authors**: Yixiang Chen, Peiyan Li, Yan Huang, Jiabing Yang, Kehan Chen, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.06224)  

**Abstract**: Current language-guided robotic manipulation systems often require low-level action-labeled datasets for imitation learning. While object-centric flow prediction methods mitigate this issue, they remain limited to scenarios involving rigid objects with clear displacement and minimal occlusion. In this work, we present Embodiment-Centric Flow (EC-Flow), a framework that directly learns manipulation from action-unlabeled videos by predicting embodiment-centric flow. Our key insight is that incorporating the embodiment's inherent kinematics significantly enhances generalization to versatile manipulation scenarios, including deformable object handling, occlusions, and non-object-displacement tasks. To connect the EC-Flow with language instructions and object interactions, we further introduce a goal-alignment module by jointly optimizing movement consistency and goal-image prediction. Moreover, translating EC-Flow to executable robot actions only requires a standard robot URDF (Unified Robot Description Format) file to specify kinematic constraints across joints, which makes it easy to use in practice. We validate EC-Flow on both simulation (Meta-World) and real-world tasks, demonstrating its state-of-the-art performance in occluded object handling (62% improvement), deformable object manipulation (45% improvement), and non-object-displacement tasks (80% improvement) than prior state-of-the-art object-centric flow methods. For more information, see our project website at this https URL . 

**Abstract (ZH)**: 基于现象中心流的操纵学习：无需动作标签的直接学习方法 

---
# Is Diversity All You Need for Scalable Robotic Manipulation? 

**Title (ZH)**: 可扩展机器人操作是否只需要多样性？ 

**Authors**: Modi Shi, Li Chen, Jin Chen, Yuxiang Lu, Chiming Liu, Guanghui Ren, Ping Luo, Di Huang, Maoqing Yao, Hongyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.06219)  

**Abstract**: Data scaling has driven remarkable success in foundation models for Natural Language Processing (NLP) and Computer Vision (CV), yet the principles of effective data scaling in robotic manipulation remain insufficiently understood. In this work, we investigate the nuanced role of data diversity in robot learning by examining three critical dimensions-task (what to do), embodiment (which robot to use), and expert (who demonstrates)-challenging the conventional intuition of "more diverse is better". Throughout extensive experiments on various robot platforms, we reveal that (1) task diversity proves more critical than per-task demonstration quantity, benefiting transfer from diverse pre-training tasks to novel downstream scenarios; (2) multi-embodiment pre-training data is optional for cross-embodiment transfer-models trained on high-quality single-embodiment data can efficiently transfer to different platforms, showing more desirable scaling property during fine-tuning than multi-embodiment pre-trained models; and (3) expert diversity, arising from individual operational preferences and stochastic variations in human demonstrations, can be confounding to policy learning, with velocity multimodality emerging as a key contributing factor. Based on this insight, we propose a distribution debiasing method to mitigate velocity ambiguity, the yielding GO-1-Pro achieves substantial performance gains of 15%, equivalent to using 2.5 times pre-training data. Collectively, these findings provide new perspectives and offer practical guidance on how to scale robotic manipulation datasets effectively. 

**Abstract (ZH)**: 数据缩放推动了自然语言处理和计算机视觉领域基础模型的显著成功，但机器人操纵中的有效数据缩放原则仍不够充分理解。在这项工作中，我们通过探讨任务（做什么）、体态（使用哪种机器人）和专家（谁演示）这三个关键维度的微妙作用，挑战了“多样性越多越好”的常规直觉。通过在各种机器人平台上的广泛实验，我们揭示了以下几点：（1）任务多样性比每任务演示量更为关键，有助于从多样的预训练任务过渡到新的下游场景；（2）多体态预训练数据并非跨体态过渡的必需品——在高质量单体态数据上训练的模型可以高效地过渡到不同的平台，在微调过程中展现出比多体态预训练模型更有利的缩放特性；（3）源自个体操作偏好和人类演示中的随机变化的专家多样性可能会妨碍策略学习，速度多模态性成为关键影响因素之一。基于这些见解，我们提出了一种分布偏差校正方法来减轻速度不确定性，所提出的GO-1-Pro在性能上取得了显著提升，相当于使用了2.5倍的预训练数据。这些发现提供了新的视角，并为有效扩大机器人操纵数据集提供了实用指导。 

---
# Evaluation of Habitat Robotics using Large Language Models 

**Title (ZH)**: 大型语言模型在 Habitat Robotics中的评估 

**Authors**: William Li, Lei Hamilton, Kaise Al-natour, Sanjeev Mohindra  

**Link**: [PDF](https://arxiv.org/pdf/2507.06157)  

**Abstract**: This paper focuses on evaluating the effectiveness of Large Language Models at solving embodied robotic tasks using the Meta PARTNER benchmark. Meta PARTNR provides simplified environments and robotic interactions within randomized indoor kitchen scenes. Each randomized kitchen scene is given a task where two robotic agents cooperatively work together to solve the task. We evaluated multiple frontier models on Meta PARTNER environments. Our results indicate that reasoning models like OpenAI o3-mini outperform non-reasoning models like OpenAI GPT-4o and Llama 3 when operating in PARTNR's robotic embodied environments. o3-mini displayed outperform across centralized, decentralized, full observability, and partial observability configurations. This provides a promising avenue of research for embodied robotic development. 

**Abstract (ZH)**: 本文 focuses于使用 Meta PARTNER 基准评估大型语言模型解决具身机器人任务的有效性。Meta PARTNER 提供了简化环境和随机化室内厨房场景中的机器人交互。每个随机化的厨房场景分配一个任务，两个机器人代理合作解决该任务。我们在 Meta PARTNER 环境中评估了多个前沿模型。我们的结果表明，在 PARTNER 的具身机器人环境中，推理模型如 OpenAI o3-mini 在性能上优于非推理模型如 OpenAI GPT-4o 和 Llama 3。o3-mini 在集中式、分散式、完全可观测性和部分可观测性配置中均表现出色。这为具身机器人开发提供了有希望的研究方向。 

---
# Learning-Augmented Model-Based Multi-Robot Planning for Time-Critical Search and Inspection Under Uncertainty 

**Title (ZH)**: 基于学习增强模型的不确定性条件下时间关键搜索与检测多机器人规划 

**Authors**: Abhish Khanal, Joseph Prince Mathew, Cameron Nowzari, Gregory J. Stein  

**Link**: [PDF](https://arxiv.org/pdf/2507.06129)  

**Abstract**: In disaster response or surveillance operations, quickly identifying areas needing urgent attention is critical, but deploying response teams to every location is inefficient or often impossible. Effective performance in this domain requires coordinating a multi-robot inspection team to prioritize inspecting locations more likely to need immediate response, while also minimizing travel time. This is particularly challenging because robots must directly observe the locations to determine which ones require additional attention. This work introduces a multi-robot planning framework for coordinated time-critical multi-robot search under uncertainty. Our approach uses a graph neural network to estimate the likelihood of PoIs needing attention from noisy sensor data and then uses those predictions to guide a multi-robot model-based planner to determine the cost-effective plan. Simulated experiments demonstrate that our planner improves performance at least by 16.3\%, 26.7\%, and 26.2\% for 1, 3, and 5 robots, respectively, compared to non-learned and learned baselines. We also validate our approach on real-world platforms using quad-copters. 

**Abstract (ZH)**: 在灾害响应或监控操作中，迅速识别需要紧急关注的区域至关重要，但将响应团队部署到每一个地点往往是低效的或不可能的。该领域有效地完成任务需要协调一个多机器人检查团队，优先检查更可能需要立即响应的位置，同时尽量减少旅行时间。由于机器人必须直接观察这些位置以确定哪些位置需要额外的关注，这使得这一任务极具挑战性。本研究介绍了一种在不确定性下协调时间关键多机器人搜索的多机器人规划框架。我们的方法使用图神经网络从噪声传感器数据中估计潜在兴趣点（PoIs）需要关注的可能性，然后利用这些预测来引导一个多机器人模型为基础的规划者确定成本效益最高的方案。模拟实验表明，与非学习基准和学习基准相比，我们的规划者分别在1个、3个和5个机器人的情况下，性能至少提高16.3%、26.7%和26.2%。我们还在现实世界平台（使用四旋翼飞行器）上验证了我们的方法。 

---
# SCCRUB: Surface Cleaning Compliant Robot Utilizing Bristles 

**Title (ZH)**: SCCRUB：利用刷子的表面清洁合规机器人 

**Authors**: Jakub F. Kowalewski, Keeyon Hajjafar, Alyssa Ugent, Jeffrey Ian Lipton  

**Link**: [PDF](https://arxiv.org/pdf/2507.06053)  

**Abstract**: Scrubbing surfaces is a physically demanding and time-intensive task. Removing adhered contamination requires substantial friction generated through pressure and torque or high lateral forces. Rigid robotic manipulators, while capable of exerting these forces, are usually confined to structured environments isolated from humans due to safety risks. In contrast, soft robot arms can safely work around humans and adapt to environmental uncertainty, but typically struggle to transmit the continuous torques or lateral forces necessary for scrubbing. Here, we demonstrate a soft robotic arm scrubbing adhered residues using torque and pressure, a task traditionally challenging for soft robots. We train a neural network to learn the arm's inverse kinematics and elasticity, which enables open-loop force and position control. Using this learned model, the robot successfully scrubbed burnt food residue from a plate and sticky fruit preserve from a toilet seat, removing an average of 99.7% of contamination. This work demonstrates how soft robots, capable of exerting continuous torque, can effectively and safely scrub challenging contamination from surfaces. 

**Abstract (ZH)**: 柔体机械臂通过扭矩和压力清除附着残留物的研究 

---
# Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning 

**Title (ZH)**: 基于射影几何代数的混合扩散策略高效机器人 manipul 完成学习 

**Authors**: Xiatao Sun, Yuxuan Wang, Shuo Yang, Yinxing Chen, Daniel Rakita  

**Link**: [PDF](https://arxiv.org/pdf/2507.05695)  

**Abstract**: Diffusion policies have become increasingly popular in robot learning due to their reliable convergence in motion generation tasks. At a high level, these policies learn to transform noisy action trajectories into effective ones, conditioned on observations. However, each time such a model is trained in a robotics context, the network must relearn fundamental spatial representations and operations, such as translations and rotations, from scratch in order to ground itself and operate effectively in a 3D environment. Incorporating geometric inductive biases directly into the network can alleviate this redundancy and substantially improve training efficiency. In this paper, we introduce hPGA-DP, a diffusion policy approach that integrates a mathematical framework called Projective Geometric Algebra (PGA) to embed strong geometric inductive biases. PGA is particularly well-suited for this purpose as it provides a unified algebraic framework that naturally encodes geometric primitives, such as points, directions, and rotations, enabling neural networks to reason about spatial structure through interpretable and composable operations. Specifically, we propose a novel diffusion policy architecture that incorporates the Projective Geometric Algebra Transformer (P-GATr), leveraging its E(3)-equivariant properties established in prior work. Our approach adopts a hybrid architecture strategy, using P-GATr as both a state encoder and action decoder, while employing U-Net or Transformer-based modules for the denoising process. Several experiments and ablation studies in both simulated and real-world environments demonstrate that hPGA-DP not only improves task performance and training efficiency through the geometric bias of P-GATr, but also achieves substantially faster convergence through its hybrid model compared to architectures that rely solely on P-GATr. 

**Abstract (ZH)**: PGA集成的扩散策略：基于投影几何代数的机器人学习方法 

---
# Integrating Diffusion-based Multi-task Learning with Online Reinforcement Learning for Robust Quadruped Robot Control 

**Title (ZH)**: 基于扩散网络的多任务学习与在线强化学习结合的稳健四足机器人控制 

**Authors**: Xinyao Qin, Xiaoteng Ma, Yang Qi, Qihan Liu, Chuanyi Xue, Ning Gui, Qinyu Dong, Jun Yang, Bin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05674)  

**Abstract**: Recent research has highlighted the powerful capabilities of imitation learning in robotics. Leveraging generative models, particularly diffusion models, these approaches offer notable advantages such as strong multi-task generalization, effective language conditioning, and high sample efficiency. While their application has been successful in manipulation tasks, their use in legged locomotion remains relatively underexplored, mainly due to compounding errors that affect stability and difficulties in task transition under limited data. Online reinforcement learning (RL) has demonstrated promising results in legged robot control in the past years, providing valuable insights to address these challenges. In this work, we propose DMLoco, a diffusion-based framework for quadruped robots that integrates multi-task pretraining with online PPO finetuning to enable language-conditioned control and robust task transitions. Our approach first pretrains the policy on a diverse multi-task dataset using diffusion models, enabling language-guided execution of various skills. Then, it finetunes the policy in simulation to ensure robustness and stable task transition during real-world deployment. By utilizing Denoising Diffusion Implicit Models (DDIM) for efficient sampling and TensorRT for optimized deployment, our policy runs onboard at 50Hz, offering a scalable and efficient solution for adaptive, language-guided locomotion on resource-constrained robotic platforms. 

**Abstract (ZH)**: 基于扩散模型的 quadruped 机器人群体模仿学习框架：多任务预训练结合在线 PPO 微调实现语言条件控制和鲁棒任务过渡 

---
# Structured Task Solving via Modular Embodied Intelligence: A Case Study on Rubik's Cube 

**Title (ZH)**: 模块化 embodied 智能驱动的结构化任务解决：Rubik's Cube 案例研究 

**Authors**: Chongshan Fan, Shenghai Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05607)  

**Abstract**: This paper presents Auto-RubikAI, a modular autonomous planning framework that integrates a symbolic Knowledge Base (KB), a vision-language model (VLM), and a large language model (LLM) to solve structured manipulation tasks exemplified by Rubik's Cube restoration. Unlike traditional robot systems based on predefined scripts, or modern approaches relying on pretrained networks and large-scale demonstration data, Auto-RubikAI enables interpretable, multi-step task execution with minimal data requirements and no prior demonstrations. The proposed system employs a KB module to solve group-theoretic restoration steps, overcoming LLMs' limitations in symbolic reasoning. A VLM parses RGB-D input to construct a semantic 3D scene representation, while the LLM generates structured robotic control code via prompt chaining. This tri-module architecture enables robust performance under spatial uncertainty. We deploy Auto-RubikAI in both simulation and real-world settings using a 7-DOF robotic arm, demonstrating effective Sim-to-Real adaptation without retraining. Experiments show a 79% end-to-end task success rate across randomized configurations. Compared to CFOP, DeepCubeA, and Two-Phase baselines, our KB-enhanced method reduces average solution steps while maintaining interpretability and safety. Auto-RubikAI provides a cost-efficient, modular foundation for embodied task planning in smart manufacturing, robotics education, and autonomous execution scenarios. Code, prompts, and hardware modules will be released upon publication. 

**Abstract (ZH)**: Auto-RubikAI：一种集成了符号知识库、视觉语言模型和大型语言模型的模块化自主规划框架 

---
# PAPRLE (Plug-And-Play Robotic Limb Environment): A Modular Ecosystem for Robotic Limbs 

**Title (ZH)**: PAPRLE（即插即用可玩机器人肢体环境）：机器人肢体的模块化生态系统 

**Authors**: Obin Kwon, Sankalp Yamsani, Noboru Myers, Sean Taylor, Jooyoung Hong, Kyungseo Park, Alex Alspach, Joohyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.05555)  

**Abstract**: We introduce PAPRLE (Plug-And-Play Robotic Limb Environment), a modular ecosystem that enables flexible placement and control of robotic limbs. With PAPRLE, a user can change the arrangement of the robotic limbs, and control them using a variety of input devices, including puppeteers, gaming controllers, and VR-based interfaces. This versatility supports a wide range of teleoperation scenarios and promotes adaptability to different task requirements. To further enhance configurability, we introduce a pluggable puppeteer device that can be easily mounted and adapted to match the target robot configurations. PAPRLE supports bilateral teleoperation through these puppeteer devices, agnostic to the type or configuration of the follower robot. By supporting both joint-space and task-space control, the system provides real-time force feedback, improving user fidelity and physical interaction awareness. The modular design of PAPRLE facilitates novel spatial arrangements of the limbs and enables scalable data collection, thereby advancing research in embodied AI and learning-based control. We validate PAPRLE in various real-world settings, demonstrating its versatility across diverse combinations of leader devices and follower robots. The system will be released as open source, including both hardware and software components, to support broader adoption and community-driven extension. Additional resources and demonstrations are available at the project website: this https URL 

**Abstract (ZH)**: PAPRLE（即插即用机器人肢体环境）模块化生态系统：灵活的机器人肢体放置与控制 

---
# Gaussian Process-Based Active Exploration Strategies in Vision and Touch 

**Title (ZH)**: 基于高斯过程的主动探索策略在视觉和触觉中的应用 

**Authors**: Ho Jin Choi, Nadia Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2507.05522)  

**Abstract**: Robots struggle to understand object properties like shape, material, and semantics due to limited prior knowledge, hindering manipulation in unstructured environments. In contrast, humans learn these properties through interactive multi-sensor exploration. This work proposes fusing visual and tactile observations into a unified Gaussian Process Distance Field (GPDF) representation for active perception of object properties. While primarily focusing on geometry, this approach also demonstrates potential for modeling surface properties beyond geometry. The GPDF encodes signed distance using point cloud, analytic gradient and Hessian, and surface uncertainty estimates, which are attributes that common neural network shape representation lack. By utilizing a point cloud to construct a distance function, GPDF does not need extensive pretraining on large datasets and can incorporate observations by aggregation. Starting with an initial visual shape estimate, the framework iteratively refines the geometry by integrating dense vision measurements using differentiable rendering and tactile measurements at uncertain surface regions. By quantifying multi-sensor uncertainties, it plans exploratory motions to maximize information gain for recovering precise 3D structures. For the real-world robot experiment, we utilize the Franka Research 3 robot manipulator, which is fixed on a table and has a customized DIGIT tactile sensor and an Intel Realsense D435 RGBD camera mounted on the end-effector. In these experiments, the robot explores the shape and properties of objects assumed to be static and placed on the table. To improve scalability, we investigate approximation methods like inducing point method for Gaussian Processes. This probabilistic multi-modal fusion enables active exploration and mapping of complex object geometries, extending potentially beyond geometry. 

**Abstract (ZH)**: 机器人由于先验知识有限，难以理解物体的形状、材质和语义属性，这阻碍了其在非结构化环境中的操作。相比之下，人类通过交互式的多传感器探索来学习这些属性。本文提出了一种将视觉和触觉观察融合到统一的高斯过程距离场（GPDF）表示中的方法，以实现对物体属性的主动感知。虽然主要侧重于几何属性，该方法也展示了对几何属性之外的表面属性建模的潜力。GPDF通过点云、分析梯度和 Hess 矩阵以及表面不确定性估计来编码带符号距离，而这些属性是常见神经网络形状表示缺失的。通过利用点云构建距离函数，GPDF不需要在大规模数据集上进行大量的预训练，并且可以通过聚合来纳入观察。从初始的视觉形状估计开始，该框架通过集成密集视图测量和在不确定表面区域的触觉测量迭代地细化几何结构。通过量化多传感器的不确定性，该方法计划探索性运动以最大化信息获取，从而恢复精确的3D结构。在真实世界机器人实验中，我们使用固定在桌子上的 Franka Research 3 机器人操作臂，其末端效应器装有定制的 DIGIT 触觉传感器和 Intel Realsense D435 彩色深度相机。在这些实验中，机器人探索假定为静止并放置在桌面上的物体的形状和属性。为了提高可扩展性，我们研究了高斯过程的近似方法，如诱导点方法。这种概率多模式融合能够主动探索和映射复杂的物体几何结构，扩展可能超越几何属性。 

---
# A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation 

**Title (ZH)**: 大规模行为模型在多任务灵巧操作中的仔细考察 

**Authors**: TRI LBM Team, Jose Barreiros, Andrew Beaulieu, Aditya Bhat, Rick Cory, Eric Cousineau, Hongkai Dai, Ching-Hsin Fang, Kunimatsu Hashimoto, Muhammad Zubair Irshad, Masha Itkina, Naveen Kuppuswamy, Kuan-Hui Lee, Katherine Liu, Dale McConachie, Ian McMahon, Haruki Nishimura, Calder Phillips-Grafflin, Charles Richter, Paarth Shah, Krishnan Srinivasan, Blake Wulfe, Chen Xu, Mengchao Zhang, Alex Alspach, Maya Angeles, Kushal Arora, Vitor Campagnolo Guizilini, Alejandro Castro, Dian Chen, Ting-Sheng Chu, Sam Creasey, Sean Curtis, Richard Denitto, Emma Dixon, Eric Dusel, Matthew Ferreira, Aimee Goncalves, Grant Gould, Damrong Guoy, Swati Gupta, Xuchen Han, Kyle Hatch, Brendan Hathaway, Allison Henry, Hillel Hochsztein, Phoebe Horgan, Shun Iwase, Donovon Jackson, Siddharth Karamcheti, Sedrick Keh, Joseph Masterjohn, Jean Mercat, Patrick Miller, Paul Mitiguy, Tony Nguyen, Jeremy Nimmer, Yuki Noguchi, Reko Ong, Aykut Onol, Owen Pfannenstiehl, Richard Poyner, Leticia Priebe Mendes Rocha, Gordon Richardson, Christopher Rodriguez, Derick Seale, Michael Sherman, Mariah Smith-Jones, David Tago, Pavel Tokmakov, Matthew Tran, Basile Van Hoorick, Igor Vasiljevic, Sergey Zakharov, Mark Zolotas, Rares Ambrus, Kerri Fetzer-Borelli, Benjamin Burchfiel, Hadas Kress-Gazit, Siyuan Feng, Stacie Ford, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2507.05331)  

**Abstract**: Robot manipulation has seen tremendous progress in recent years, with imitation learning policies enabling successful performance of dexterous and hard-to-model tasks. Concurrently, scaling data and model size has led to the development of capable language and vision foundation models, motivating large-scale efforts to create general-purpose robot foundation models. While these models have garnered significant enthusiasm and investment, meaningful evaluation of real-world performance remains a challenge, limiting both the pace of development and inhibiting a nuanced understanding of current capabilities. In this paper, we rigorously evaluate multitask robot manipulation policies, referred to as Large Behavior Models (LBMs), by extending the Diffusion Policy paradigm across a corpus of simulated and real-world robot data. We propose and validate an evaluation pipeline to rigorously analyze the capabilities of these models with statistical confidence. We compare against single-task baselines through blind, randomized trials in a controlled setting, using both simulation and real-world experiments. We find that multi-task pretraining makes the policies more successful and robust, and enables teaching complex new tasks more quickly, using a fraction of the data when compared to single-task baselines. Moreover, performance predictably increases as pretraining scale and diversity grows. Project page: this https URL 

**Abstract (ZH)**: 机器人操作在近年来取得了 tremendous 进展，模仿学习策略使其能够成功完成灵巧且难以建模的任务。同时，数据和模型规模的扩展推动了强大语言和视觉基础模型的发展，激发了创建通用机器人基础模型的大规模努力。尽管这些模型吸引了大量关注和投资，但对其实际性能的有效评估仍然是一项挑战，限制了研发的步伐，并阻碍了对当前能力的深入理解。本文通过将扩散策略 paradigm 扩展到模拟和实际机器人数据集，严格评估多任务机器人操作策略，称为大型行为模型（LBMs）。我们提出并验证了一种评估管道，以统计学信心严格分析这些模型的能力。我们通过控制环境下的盲随机试验，使用模拟和实际实验来与单任务基线进行对比。我们发现，多任务预训练使策略更加成功且稳健，并能更快地教授复杂的新型任务，所需数据仅为单任务基线的一小部分。此外，预训练规模和多样性增长时，性能可预测地提高。项目页面: this https URL 

---
# Safe Domain Randomization via Uncertainty-Aware Out-of-Distribution Detection and Policy Adaptation 

**Title (ZH)**: 基于不确定性意识的域外检测与策略调整的安全领域随机化 

**Authors**: Mohamad H. Danesh, Maxime Wabartha, Stanley Wu, Joelle Pineau, Hsiu-Chin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.06111)  

**Abstract**: Deploying reinforcement learning (RL) policies in real-world involves significant challenges, including distribution shifts, safety concerns, and the impracticality of direct interactions during policy refinement. Existing methods, such as domain randomization (DR) and off-dynamics RL, enhance policy robustness by direct interaction with the target domain, an inherently unsafe practice. We propose Uncertainty-Aware RL (UARL), a novel framework that prioritizes safety during training by addressing Out-Of-Distribution (OOD) detection and policy adaptation without requiring direct interactions in target domain. UARL employs an ensemble of critics to quantify policy uncertainty and incorporates progressive environmental randomization to prepare the policy for diverse real-world conditions. By iteratively refining over high-uncertainty regions of the state space in simulated environments, UARL enhances robust generalization to the target domain without explicitly training on it. We evaluate UARL on MuJoCo benchmarks and a quadrupedal robot, demonstrating its effectiveness in reliable OOD detection, improved performance, and enhanced sample efficiency compared to baselines. 

**Abstract (ZH)**: 基于不确定性感知的强化学习（UARL）：一种在训练期间优先考虑安全性的新框架 

---
# CogniPlay: a work-in-progress Human-like model for General Game Playing 

**Title (ZH)**: CogniPlay: 一个进展中的类人通用游戏-playing 模型 

**Authors**: Aloïs Rautureau, Éric Piette  

**Link**: [PDF](https://arxiv.org/pdf/2507.05868)  

**Abstract**: While AI systems have equaled or surpassed human performance in a wide variety of games such as Chess, Go, or Dota 2, describing these systems as truly "human-like" remains far-fetched. Despite their success, they fail to replicate the pattern-based, intuitive decision-making processes observed in human cognition. This paper presents an overview of findings from cognitive psychology and previous efforts to model human-like behavior in artificial agents, discusses their applicability to General Game Playing (GGP) and introduces our work-in-progress model based on these observations: CogniPlay. 

**Abstract (ZH)**: 虽然人工智能系统在国际象棋、围棋或Dota 2等广泛的游戏领域中已达到或超过了人类的表现，但将这些系统真正描述为“类似于人类”的仍然为时过早。尽管取得了成功，它们仍然未能复制人类认知中基于模式、直觉的决策过程。本文综述了认知心理学的研究发现和以前为人工代理建模类似人类行为的努力，讨论了这些方法在通用游戏-playing（GGP）中的适用性，并介绍了基于这些观察成果的工作中模型：CogniPlay。 

---
# Agent KB: Leveraging Cross-Domain Experience for Agentic Problem Solving 

**Title (ZH)**: 基于跨域经验的代理知识库：实现有能动性的问题解决 

**Authors**: Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, Ge Zhang, Jiaheng Liu, Xingyao Wang, Sirui Hong, Chenglin Wu, Hao Cheng, Chi Wang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.06229)  

**Abstract**: As language agents tackle increasingly complex tasks, they struggle with effective error correction and experience reuse across domains. We introduce Agent KB, a hierarchical experience framework that enables complex agentic problem solving via a novel Reason-Retrieve-Refine pipeline. Agent KB addresses a core limitation: agents traditionally cannot learn from each other's experiences. By capturing both high-level strategies and detailed execution logs, Agent KB creates a shared knowledge base that enables cross-agent knowledge transfer. Evaluated on the GAIA benchmark, Agent KB improves success rates by up to 16.28 percentage points. On the most challenging tasks, Claude-3 improves from 38.46% to 57.69%, while GPT-4 improves from 53.49% to 73.26% on intermediate tasks. On SWE-bench code repair, Agent KB enables Claude-3 to improve from 41.33% to 53.33%. Our results suggest that Agent KB provides a modular, framework-agnostic infrastructure for enabling agents to learn from past experiences and generalize successful strategies to new tasks. 

**Abstract (ZH)**: 随着语言代理承担的任务日益复杂，它们在有效错误纠正和跨领域经验 reuse 方面遇到挑战。我们提出了一种名为 Agent KB 的分层经验框架，通过新颖的 Reason-Retrieve-Refine 管道实现复杂代理问题解决。Agent KB 解决了核心局限性：代理传统上无法从彼此的经验中学习。通过捕获高层策略和详细的执行日志，Agent KB 创建了一个共享的知识库，从而使跨代理的知识转移成为可能。在 GAIA 基准测试中，Agent KB 将成功率提高高达 16.28 个百分点。在最具挑战性的任务中，Claude-3 的成功率从 38.46% 提高到 57.69%，而 GPT-4 在中等难度任务中的成功率从 53.49% 提高到 73.26%。在 SWE-bench 代码修复任务中，Agent KB 使 Claude-3 的成功率从 41.33% 提高到 53.33%。我们的结果表明，Agent KB 提供了一种模块化、框架无关的基础结构，使代理能够从过去的经验中学习，并将成功的策略推广到新任务。 

---
# Hierarchy or Heterarchy? A Theory of Long-Range Connections for the Sensorimotor Brain 

**Title (ZH)**: 层级结构还是异阶结构？传感器imotor大脑中长程连接的理论 

**Authors**: Jeff Hawkins, Niels Leadholm, Viviane Clay  

**Link**: [PDF](https://arxiv.org/pdf/2507.05888)  

**Abstract**: In the traditional understanding of the neocortex, sensory information flows up a hierarchy of regions, with each level processing increasingly complex features. Information also flows down the hierarchy via a different set of connections. Although the hierarchical model has significant support, many anatomical connections do not conform to the standard hierarchical interpretation. In addition, hierarchically arranged regions sometimes respond in parallel, not sequentially as would occur in a hierarchy. This and other evidence suggests that two regions can act in parallel and hierarchically at the same time. Given this flexibility, the word "heterarchy" might be a more suitable term to describe neocortical organization. This paper proposes a new interpretation of how sensory and motor information is processed in the neocortex. The key to our proposal is what we call the "Thousand Brains Theory", which posits that every cortical column is a sensorimotor learning system. Columns learn by integrating sensory input over multiple movements of a sensor. In this view, even primary and secondary regions, such as V1 and V2, can learn and recognize complete 3D objects. This suggests that the hierarchical connections between regions are used to learn the compositional structure of parent objects composed of smaller child objects. We explain the theory by examining the different types of long-range connections between cortical regions and between the neocortex and thalamus. We describe these connections, and then suggest the specific roles they play in the context of a heterarchy of sensorimotor regions. We also suggest that the thalamus plays an essential role in transforming the pose between objects and sensors. The novel perspective we argue for here has broad implications for both neuroscience and artificial intelligence. 

**Abstract (ZH)**: 新皮层中感觉和运动信息处理的新理解：一千大脑理论与异archy组织结构 

---
# Efficient Training of Large-Scale AI Models Through Federated Mixture-of-Experts: A System-Level Approach 

**Title (ZH)**: 通过联邦混合专家系统级方法高效训练大规模AI模型 

**Authors**: Xiaobing Chen, Boyang Zhang, Xiangwei Zhou, Mingxuan Sun, Shuai Zhang, Songyang Zhang, Geoffrey Ye Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.05685)  

**Abstract**: The integration of Federated Learning (FL) and Mixture-of-Experts (MoE) presents a compelling pathway for training more powerful, large-scale artificial intelligence models (LAMs) on decentralized data while preserving privacy. However, efficient federated training of these complex MoE-structured LAMs is hindered by significant system-level challenges, particularly in managing the interplay between heterogeneous client resources and the sophisticated coordination required for numerous specialized experts. This article highlights a critical, yet underexplored concept: the absence of robust quantitative strategies for dynamic client-expert alignment that holistically considers varying client capacities and the imperative for system-wise load balancing. Specifically, we propose a conceptual system design for intelligent client-expert alignment that incorporates dynamic fitness scoring, global expert load monitoring, and client capacity profiling. By tackling these systemic issues, we can unlock more scalable, efficient, and robust training mechanisms {with fewer communication rounds for convergence}, paving the way for the widespread deployment of large-scale federated MoE-structured LAMs in edge computing with ultra-high communication efficiency. 

**Abstract (ZH)**: 联邦学习（FL）与混合专家（MoE）的集成为在去中心化数据上训练更强大、更大规模的人工智能模型（LAMs）提供了令人信服的途径，同时保护隐私。然而，这些复杂结构的MoE模型的高效联邦训练受到了显著的系统级挑战的阻碍，特别是在管理和协调异构客户端资源与众多专门专家之间复杂交互方面。本文强调了一个关键但尚未充分探索的概念：缺乏一种稳健的定量策略来动态协调客户端与专家，这一策略能够综合考虑客户容量的变化和系统级负载均衡的迫切需要。具体而言，我们提出了一种概念性的系统设计，用于智能客户端与专家的动态对齐，该设计融合了动态适应性评分、全局专家负载监控和客户端容量特征化。通过解决这些系统性问题，我们可以解锁更具有伸缩性、更高效和更稳健的训练机制，通过减少收敛所需的通信轮数，在边缘计算中为超高效通信效率下广泛部署大规模联邦MoE结构的LAMs铺平道路。 

---
# Going Beyond Heuristics by Imposing Policy Improvement as a Constraint 

**Title (ZH)**: 通过将策略改进作为约束来超越启发式方法 

**Authors**: Chi-Chang Lee, Zhang-Wei Hong, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2507.05328)  

**Abstract**: In many reinforcement learning (RL) applications, augmenting the task rewards with heuristic rewards that encode human priors about how a task should be solved is crucial for achieving desirable performance. However, because such heuristics are usually not optimal, much human effort and computational resources are wasted in carefully balancing tasks and heuristic rewards. Theoretically rigorous ways of incorporating heuristics rely on the idea of \textit{policy invariance}, which guarantees that the performance of a policy obtained by maximizing heuristic rewards is the same as the optimal policy with respect to the task reward. However, in practice, policy invariance doesn't result in policy improvement, and such methods are known to empirically perform poorly. We propose a new paradigm to mitigate reward hacking and effectively use heuristics based on the practical goal of maximizing policy improvement instead of policy improvement. Our framework, Heuristic Enhanced Policy Optimization (HEPO), effectively leverages heuristics while avoiding the pitfall of prior methods for mitigating reward hacking. HEPO achieves superior performance on standard benchmarks with well-engineered reward functions. More surprisingly, HEPO allows policy optimization to achieve good performance even when heuristics are not well-engineered and designed by non-expert humans, showcasing HEPO's ability to reduce human effort in reward design. % HEPO is a plug-and-play optimization method for leveraging heuristics in reinforcement learning. Code is available at this https URL. 

**Abstract (ZH)**: 基于改进策略优化的启发式增强（HEPO）：一种减少奖励欺骗并有效利用启发式的全新范式 

---
# Beyond classical and contemporary models: a transformative ai framework for student dropout prediction in distance learning using rag, prompt engineering, and cross-modal fusion 

**Title (ZH)**: 超越古典和当代模型：基于RAG、提示工程和跨模态融合的学生辍学预测transformative AI框架在远程学习中的应用 

**Authors**: Miloud Mihoubi, Meriem Zerkouk, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2507.05285)  

**Abstract**: Student dropout in distance learning remains a critical challenge, with profound societal and economic consequences. While classical machine learning models leverage structured socio-demographic and behavioral data, they often fail to capture the nuanced emotional and contextual factors embedded in unstructured student interactions. This paper introduces a transformative AI framework that redefines dropout prediction through three synergistic innovations: Retrieval-Augmented Generation (RAG) for domain-specific sentiment analysis, prompt engineering to decode academic stressors, and cross-modal attention fusion to dynamically align textual, behavioral, and socio-demographic insights. By grounding sentiment analysis in a curated knowledge base of pedagogical content, our RAG-enhanced BERT model interprets student comments with unprecedented contextual relevance, while optimized prompts isolate indicators of academic distress (e.g., "isolation," "workload anxiety"). A cross-modal attention layer then fuses these insights with temporal engagement patterns, creating holistic risk profiles. Evaluated on a longitudinal dataset of 4 423 students, the framework achieves 89% accuracy and an F1-score of 0.88, outperforming conventional models by 7% and reducing false negatives by 21%. Beyond prediction, the system generates interpretable interventions by retrieving contextually aligned strategies (e.g., mentorship programs for isolated learners). This work bridges the gap between predictive analytics and actionable pedagogy, offering a scalable solution to mitigate dropout risks in global education systems 

**Abstract (ZH)**: 基于检索增强生成的多层次情感分析框架在远程教育退学预测中的应用 

---
