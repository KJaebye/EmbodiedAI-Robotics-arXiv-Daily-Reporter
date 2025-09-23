# HuMam: Humanoid Motion Control via End-to-End Deep Reinforcement Learning with Mamba 

**Title (ZH)**: HuMam: 通过Mamba实现端到端深度强化学习的人形运动控制 

**Authors**: Yinuo Wang, Yuanyang Qi, Jinzhao Zhou, Gavin Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18046)  

**Abstract**: End-to-end reinforcement learning (RL) for humanoid locomotion is appealing for its compact perception-action mapping, yet practical policies often suffer from training instability, inefficient feature fusion, and high actuation cost. We present HuMam, a state-centric end-to-end RL framework that employs a single-layer Mamba encoder to fuse robot-centric states with oriented footstep targets and a continuous phase clock. The policy outputs joint position targets tracked by a low-level PD loop and is optimized with PPO. A concise six-term reward balances contact quality, swing smoothness, foot placement, posture, and body stability while implicitly promoting energy saving. On the JVRC-1 humanoid in mc-mujoco, HuMam consistently improves learning efficiency, training stability, and overall task performance over a strong feedforward baseline, while reducing power consumption and torque peaks. To our knowledge, this is the first end-to-end humanoid RL controller that adopts Mamba as the fusion backbone, demonstrating tangible gains in efficiency, stability, and control economy. 

**Abstract (ZH)**: 基于Mamba编码器的人形机器人端到端强化学习框架 

---
# Prepare Before You Act: Learning From Humans to Rearrange Initial States 

**Title (ZH)**: 未雨绸缪：从人类学习以重排初始状态 

**Authors**: Yinlong Dai, Andre Keyser, Dylan P. Losey  

**Link**: [PDF](https://arxiv.org/pdf/2509.18043)  

**Abstract**: Imitation learning (IL) has proven effective across a wide range of manipulation tasks. However, IL policies often struggle when faced with out-of-distribution observations; for instance, when the target object is in a previously unseen position or occluded by other objects. In these cases, extensive demonstrations are needed for current IL methods to reach robust and generalizable behaviors. But when humans are faced with these sorts of atypical initial states, we often rearrange the environment for more favorable task execution. For example, a person might rotate a coffee cup so that it is easier to grasp the handle, or push a box out of the way so they can directly grasp their target object. In this work we seek to equip robot learners with the same capability: enabling robots to prepare the environment before executing their given policy. We propose ReSET, an algorithm that takes initial states -- which are outside the policy's distribution -- and autonomously modifies object poses so that the restructured scene is similar to training data. Theoretically, we show that this two step process (rearranging the environment before rolling out the given policy) reduces the generalization gap. Practically, our ReSET algorithm combines action-agnostic human videos with task-agnostic teleoperation data to i) decide when to modify the scene, ii) predict what simplifying actions a human would take, and iii) map those predictions into robot action primitives. Comparisons with diffusion policies, VLAs, and other baselines show that using ReSET to prepare the environment enables more robust task execution with equal amounts of total training data. See videos at our project website: this https URL 

**Abstract (ZH)**: 模仿学习（IL）在多种操作任务中已证明非常有效。然而，当面对分布外观察时，IL策略往往会遇到困难；例如，当目标对象处于之前未见过的位置或被其他对象遮挡时。在这种情况下，当前的IL方法需要大量的演示才能达到鲁棒和泛化的行为。但当人类面对这些不常见的初始状态时，我们通常会重新安排环境以利于任务执行。例如，一个人可能会转动咖啡杯以便更容易抓住把手，或推动一个箱子以方便直接抓住目标物体。在这项工作中，我们旨在为机器人学习者赋予同样的能力：使机器人能够在执行给定策略之前重新准备环境。我们提出了一种ReSET算法，该算法接受初始状态（这些状态超出了策略的分布范围），并自主修改物体姿态，使得重组的场景类似于训练数据。理论上，我们证明了这种两步过程（在执行给定策略之前重新安排环境）能够减少泛化差距。实际上，我们的ReSET算法结合了动作无关的人类视频数据与任务无关的远程操作数据，以i) 确定何时修改场景，ii) 预测人类会采取何种简化行动，以及iii) 将这些预测映射为机器人动作基元。与扩散策略、VLAs以及其他基线方法的对比表明，使用ReSET来准备环境能够在相同总量的训练数据下实现更鲁棒的任务执行。访问我们的项目网站查看视频：this https URL。 

---
# M3ET: Efficient Vision-Language Learning for Robotics based on Multimodal Mamba-Enhanced Transformer 

**Title (ZH)**: M3ET：基于多模态Mamba增强Transformer的机器人视觉-语言高效学习 

**Authors**: Yanxin Zhang, Liang He, Zeyi Kang, Zuheng Ming, Kaixing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18005)  

**Abstract**: In recent years, multimodal learning has become essential in robotic vision and information fusion, especially for understanding human behavior in complex environments. However, current methods struggle to fully leverage the textual modality, relying on supervised pretrained models, which limits semantic extraction in unsupervised robotic environments, particularly with significant modality loss. These methods also tend to be computationally intensive, leading to high resource consumption in real-world applications. To address these challenges, we propose the Multi Modal Mamba Enhanced Transformer (M3ET), a lightweight model designed for efficient multimodal learning, particularly on mobile platforms. By incorporating the Mamba module and a semantic-based adaptive attention mechanism, M3ET optimizes feature fusion, alignment, and modality reconstruction. Our experiments show that M3ET improves cross-task performance, with a 2.3 times increase in pretraining inference speed. In particular, the core VQA task accuracy of M3ET remains at 0.74, while the model's parameter count is reduced by 0.67. Although performance on the EQA task is limited, M3ET's lightweight design makes it well suited for deployment on resource-constrained robotic platforms. 

**Abstract (ZH)**: 近年来，多模态学习在机器人视觉和信息融合中变得至关重要，尤其在理解和解析复杂环境中的人类行为方面。然而，当前的方法在充分利用文本模态方面存在一定困难，主要依赖于监督预训练模型，这在无监督的机器人环境中限制了语义提取，尤其是在显著的模态损失情况下。此外，这些方法往往计算密集，导致在实际应用中资源消耗高。为了解决这些挑战，我们提出了一种轻量级模型—多模态蟒蛇增强Transformer（M3ET），该模型旨在移动平台上实现高效多模态学习。通过集成Mamba模块和基于语义的自适应注意力机制，M3ET优化了特征融合、对齐和模态重建。我们的实验显示，M3ET在预训练推理速度上提高了2.3倍，特别是在核心VQA任务上的准确率保持在0.74的同时，模型参数量减少了0.67。尽管在EQA任务上的表现有限，但M3ET的轻量级设计使其非常适合部署在资源受限的机器人平台上。 

---
# ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion 

**Title (ZH)**: 可组合导航：通过可组合扩散实现动态环境中的指令遵循导航 

**Authors**: Zichao Hu, Chen Tang, Michael J. Munje, Yifeng Zhu, Alex Liu, Shuijing Liu, Garrett Warnell, Peter Stone, Joydeep Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2509.17941)  

**Abstract**: This paper considers the problem of enabling robots to navigate dynamic environments while following instructions. The challenge lies in the combinatorial nature of instruction specifications: each instruction can include multiple specifications, and the number of possible specification combinations grows exponentially as the robot's skill set expands. For example, "overtake the pedestrian while staying on the right side of the road" consists of two specifications: "overtake the pedestrian" and "walk on the right side of the road." To tackle this challenge, we propose ComposableNav, based on the intuition that following an instruction involves independently satisfying its constituent specifications, each corresponding to a distinct motion primitive. Using diffusion models, ComposableNav learns each primitive separately, then composes them in parallel at deployment time to satisfy novel combinations of specifications unseen in training. Additionally, to avoid the onerous need for demonstrations of individual motion primitives, we propose a two-stage training procedure: (1) supervised pre-training to learn a base diffusion model for dynamic navigation, and (2) reinforcement learning fine-tuning that molds the base model into different motion primitives. Through simulation and real-world experiments, we show that ComposableNav enables robots to follow instructions by generating trajectories that satisfy diverse and unseen combinations of specifications, significantly outperforming both non-compositional VLM-based policies and costmap composing baselines. Videos and additional materials can be found on the project page: this https URL 

**Abstract (ZH)**: 本文探讨了使机器人能够在遵循指令的同时导航动态环境的问题。挑战在于指令规范的组合性质：每个指令可能包含多个规范，随着机器人技能的增加，可能的规范组合数量呈指数级增长。例如，“在靠右行驶的同时超车行人”包含两个规范：“超车行人”和“靠右行驶”。为了解决这一挑战，我们提出了ComposableNav，基于这样的直觉：遵循指令涉及独立地满足其组成部分规范，每个规范对应一个不同的运动本原。通过扩散模型，ComposableNav分别学习每个本原，然后在部署时并行组合它们以满足训练中未见过的新颖规范组合。此外，为了避免单独运动本原的演示需求，我们提出了两阶段训练程序：（1）监督预训练以学习用于动态导航的基础扩散模型，以及（2）强化学习微调以将基础模型塑造成不同的运动本原。通过仿真和真实世界实验，我们展示了ComposableNav通过生成满足各种未见过的规范组合的轨迹，能够使机器人遵循指令，并在性能上显著优于非组合的VLM基线策略和成本图组合基线。项目页面上有相关视频和额外材料：this https URL 

---
# DriveDPO: Policy Learning via Safety DPO For End-to-End Autonomous Driving 

**Title (ZH)**: DriveDPO：基于安全DPO的端到端自动驾驶策略学习 

**Authors**: Shuyao Shang, Yuntao Chen, Yuqi Wang, Yingyan Li, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17940)  

**Abstract**: End-to-end autonomous driving has substantially progressed by directly predicting future trajectories from raw perception inputs, which bypasses traditional modular pipelines. However, mainstream methods trained via imitation learning suffer from critical safety limitations, as they fail to distinguish between trajectories that appear human-like but are potentially unsafe. Some recent approaches attempt to address this by regressing multiple rule-driven scores but decoupling supervision from policy optimization, resulting in suboptimal performance. To tackle these challenges, we propose DriveDPO, a Safety Direct Preference Optimization Policy Learning framework. First, we distill a unified policy distribution from human imitation similarity and rule-based safety scores for direct policy optimization. Further, we introduce an iterative Direct Preference Optimization stage formulated as trajectory-level preference alignment. Extensive experiments on the NAVSIM benchmark demonstrate that DriveDPO achieves a new state-of-the-art PDMS of 90.0. Furthermore, qualitative results across diverse challenging scenarios highlight DriveDPO's ability to produce safer and more reliable driving behaviors. 

**Abstract (ZH)**: 端到端自主驾驶通过直接从原始感知输入预测未来轨迹取得了显著进展，这绕过了传统的模块化管道。然而，主流通过模仿学习训练的方法在关键安全方面存在限制，因为它们无法区分看似人类但可能是不安全的轨迹。一些最近的方法试图通过回归多个规则驱动的得分来解决这个问题，但解耦监督与策略优化，导致性能不佳。为了应对这些挑战，我们提出DriveDPO，一种直接偏好优化策略学习的安全框架。首先，我们从人类模仿相似性和基于规则的安全得分中提炼统一的策略分布，用于直接策略优化。进一步，我们引入了一个迭代的直接偏好优化阶段，以轨迹级偏好对齐的形式进行表述。在NAVSIM基准上的广泛实验显示，DriveDPO实现了新的最佳PDMS为90.0。此外，跨多种具有挑战性的场景的定性结果突显了DriveDPO生成更安全、更可靠驾驶行为的能力。 

---
# Sight Over Site: Perception-Aware Reinforcement Learning for Efficient Robotic Inspection 

**Title (ZH)**: 眼见为实：基于感知的强化学习在高效机器人检测中的应用 

**Authors**: Richard Kuhlmann, Jakob Wolfram, Boyang Sun, Jiaxu Xing, Davide Scaramuzza, Marc Pollefeys, Cesar Cadena  

**Link**: [PDF](https://arxiv.org/pdf/2509.17877)  

**Abstract**: Autonomous inspection is a central problem in robotics, with applications ranging from industrial monitoring to search-and-rescue. Traditionally, inspection has often been reduced to navigation tasks, where the objective is to reach a predefined location while avoiding obstacles. However, this formulation captures only part of the real inspection problem. In real-world environments, the inspection targets may become visible well before their exact coordinates are reached, making further movement both redundant and inefficient. What matters more for inspection is not simply arriving at the target's position, but positioning the robot at a viewpoint from which the target becomes observable. In this work, we revisit inspection from a perception-aware perspective. We propose an end-to-end reinforcement learning framework that explicitly incorporates target visibility as the primary objective, enabling the robot to find the shortest trajectory that guarantees visual contact with the target without relying on a map. The learned policy leverages both perceptual and proprioceptive sensing and is trained entirely in simulation, before being deployed to a real-world robot. We further develop an algorithm to compute ground-truth shortest inspection paths, which provides a reference for evaluation. Through extensive experiments, we show that our method outperforms existing classical and learning-based navigation approaches, yielding more efficient inspection trajectories in both simulated and real-world settings. The project is avialable at this https URL 

**Abstract (ZH)**: 自主检测是机器人技术中的一个核心问题，应用于从工业监控到搜寻救援等多种场景。传统上，检测任务往往被简化为导航任务，目标是在避开障碍物的前提下到达预定义的位置。然而，这种形式仅捕捉到了检测问题的一部分。在现实环境中，检测目标可能在达到其确切坐标之前就已经变得可见，这使得进一步移动变得既冗余又低效。检测的关键在于不仅仅是到达目标的位置，而是将机器人置于一个能够观察到目标的视角上。在本文中，我们从感知意识的角度重新审视检测问题。我们提出了一个端到端的强化学习框架，明确将目标的可见性作为主要目标，使机器人能够找到一个确保与目标视觉接触的最短路径，而无需依赖地图。学习到的策略结合了感知和本体感觉，并在仿真中完全训练，然后部署到实际机器人上。我们还开发了一个算法来计算地面真实最短检测路径，为评估提供了参考。通过广泛的实验，我们证明了我们的方法在模拟和现实环境中的检测轨迹中表现优于现有的经典和基于学习的导航方法。该项目可在以下链接获取：this https URL。 

---
# Tac2Motion: Contact-Aware Reinforcement Learning with Tactile Feedback for Robotic Hand Manipulation 

**Title (ZH)**: Tac2Motion: 基于触觉反馈的接触感知强化学习手法操作 

**Authors**: Yitaek Kim, Casper Hewson Rask, Christoffer Sloth  

**Link**: [PDF](https://arxiv.org/pdf/2509.17812)  

**Abstract**: This paper proposes Tac2Motion, a contact-aware reinforcement learning framework to facilitate the learning of contact-rich in-hand manipulation tasks, such as removing a lid. To this end, we propose tactile sensing-based reward shaping and incorporate the sensing into the observation space through embedding. The designed rewards encourage an agent to ensure firm grasping and smooth finger gaiting at the same time, leading to higher data efficiency and robust performance compared to the baseline. We verify the proposed framework on the opening a lid scenario, showing generalization of the trained policy into a couple of object types and various dynamics such as torsional friction. Lastly, the learned policy is demonstrated on the multi-fingered robot, Shadow Robot, showing that the control policy can be transferred to the real world. The video is available: this https URL. 

**Abstract (ZH)**: 本文提出Tac2Motion，这是一种基于接触的强化学习框架，旨在促进学习富含接触的手内操作任务，如开盖子。为此，我们提出了一种基于触觉感知的奖励优化方法，并通过嵌入将感知集成到观察空间中。设计的奖励鼓励智能体同时确保稳定的抓握和流畅的手指动作，与基线方法相比，提高了数据效率和鲁棒性性能。我们在开盖子场景中验证了所提出的框架，展示了训练策略对多种物体类型和各种动力学（如扭转摩擦）的通用性。最后，所学习的策略在多指机器人Shadow Robot上进行演示，证明了控制策略可以应用于实际世界。视频见：this https URL。 

---
# RoboSeek: You Need to Interact with Your Objects 

**Title (ZH)**: RoboSeek: 你需要与你的物体互动 

**Authors**: Yibo Peng, Jiahao Yang, Shenhao Yan, Ziyu Huang, Shuang Li, Shuguang Cui, Yiming Zhao, Yatong Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.17783)  

**Abstract**: Optimizing and refining action execution through
exploration and interaction is a promising way for robotic
manipulation. However, practical approaches to interaction driven robotic learning are still underexplored, particularly for
long-horizon tasks where sequential decision-making, physical
constraints, and perceptual uncertainties pose significant chal lenges. Motivated by embodied cognition theory, we propose
RoboSeek, a framework for embodied action execution that
leverages interactive experience to accomplish manipulation
tasks. RoboSeek optimizes prior knowledge from high-level
perception models through closed-loop training in simulation
and achieves robust real-world execution via a real2sim2real
transfer pipeline. Specifically, we first replicate real-world
environments in simulation using 3D reconstruction to provide
visually and physically consistent environments., then we train
policies in simulation using reinforcement learning and the
cross-entropy method leveraging visual priors. The learned
policies are subsequently deployed on real robotic platforms
for execution. RoboSeek is hardware-agnostic and is evaluated
on multiple robotic platforms across eight long-horizon ma nipulation tasks involving sequential interactions, tool use, and
object handling. Our approach achieves an average success rate
of 79%, significantly outperforming baselines whose success
rates remain below 50%, highlighting its generalization and
robustness across tasks and platforms. Experimental results
validate the effectiveness of our training framework in complex,
dynamic real-world settings and demonstrate the stability of the
proposed real2sim2real transfer mechanism, paving the way for
more generalizable embodied robotic learning. Project Page:
this https URL 

**Abstract (ZH)**: 通过探索和交互优化并精细动作执行是机器人操作的一个有前途的方法。然而，基于交互驱动的机器人学习的实际方法在长时任务中仍相对较未探索，尤其在涉及序列决策、物理约束和感知不确定性时面临重大挑战。受 embodied 认知理论启发，我们提出了 RoboSeek，一种通过交互经验来完成操作任务的体化动作执行框架，通过仿真中的闭环训练优化高级感知模型的先验知识，并通过实操-仿真-实操的转移管道实现稳健的现实世界执行。具体而言，我们首先使用三维重建在仿真中复制现实世界环境，提供视觉和物理一致的环境，然后利用视觉先验并通过强化学习和交叉熵方法训练策略。学习到的策略随后部署在真实的机器人平台上进行执行。RoboSeek 不依赖于硬件，并在涉及序列交互、工具使用和物体处理的八个长时操作任务中在多个机器人平台上进行了评估。我们的方法在这些任务中的平均成功率达到了 79%，显著优于成功的基线方法，这些基线方法的成功率均低于 50%，突显了其在任务和平台之间的泛化能力和鲁棒性。实验结果验证了我们训练框架在复杂、动态的现实世界环境中的有效性，并展示了所提出的实操-仿真-实操转移机制的稳定性，为更加通用的体化机器人学习铺平了道路。项目页面：this https URL。 

---
# Enhancing the NAO: Extending Capabilities of Legacy Robots for Long-Term Research 

**Title (ZH)**: 增强NAO：扩展legacy机器人长期研究能力 

**Authors**: Austin Wilson, Sahar Kapasi, Zane Greene, Alexis E. Block  

**Link**: [PDF](https://arxiv.org/pdf/2509.17760)  

**Abstract**: Many research groups face challenges when legacy (unsupported) robotic platforms lose manufacturer support and cannot accommodate modern sensing, speech, and interaction capabilities. We present the Enhanced NAO, a revitalized version of Aldebaran's NAO robot that uses upgraded microphones, RGB-D and thermal cameras, and additional compute resources in a fully self-contained package. This system combines cloud and local models for perception and dialogue, while preserving the NAO's expressive body and behaviors. In a pilot validation study, the Enhanced NAO delivered significantly higher conversational quality and stronger user preference compared to the NAO AI Edition, without increasing response latency. Key upgrades, such as beamforming microphones and low-latency audio processing, reduced artifacts like self-hearing and improved multi-party separation. Expanded visual and thermal sensing established a foundation for future interaction capabilities. Beyond the NAO, our framework provides a platform-agnostic strategy for extending the lifespan and research utility of legacy robots, ensuring they remain valuable tools for human-robot interaction. 

**Abstract (ZH)**: 废弃机器人平台的现代感知与交互能力升级挑战及解决方案：Enhanced NAO的 revitalized 版本及其应用研究 

---
# MotionTrans: Human VR Data Enable Motion-Level Learning for Robotic Manipulation Policies 

**Title (ZH)**: MotionTrans: 人类VR数据赋能机器人操作策略的运动级别学习 

**Authors**: Chengbo Yuan, Rui Zhou, Mengzhen Liu, Yingdong Hu, Shengjie Wang, Li Yi, Chuan Wen, Shanghang Zhang, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.17759)  

**Abstract**: Scaling real robot data is a key bottleneck in imitation learning, leading to the use of auxiliary data for policy training. While other aspects of robotic manipulation such as image or language understanding may be learned from internet-based datasets, acquiring motion knowledge remains challenging. Human data, with its rich diversity of manipulation behaviors, offers a valuable resource for this purpose. While previous works show that using human data can bring benefits, such as improving robustness and training efficiency, it remains unclear whether it can realize its greatest advantage: enabling robot policies to directly learn new motions for task completion. In this paper, we systematically explore this potential through multi-task human-robot cotraining. We introduce MotionTrans, a framework that includes a data collection system, a human data transformation pipeline, and a weighted cotraining strategy. By cotraining 30 human-robot tasks simultaneously, we direcly transfer motions of 13 tasks from human data to deployable end-to-end robot policies. Notably, 9 tasks achieve non-trivial success rates in zero-shot manner. MotionTrans also significantly enhances pretraining-finetuning performance (+40% success rate). Through ablation study, we also identify key factors for successful motion learning: cotraining with robot data and broad task-related motion coverage. These findings unlock the potential of motion-level learning from human data, offering insights into its effective use for training robotic manipulation policies. All data, code, and model weights are open-sourced this https URL. 

**Abstract (ZH)**: 扩展真实机器人数据是模仿学习中的关键瓶颈，导致使用辅助数据进行策略训练。虽然其他方面的机器人操作，如图像或语言理解可以从互联网数据集中学到，但获取运动知识仍具有挑战性。人类数据因其丰富的操作行为多样性，为这一目的提供了有价值的资源。尽管之前的研究表明使用人类数据可以带来改进，例如提高鲁棒性和训练效率，但尚不清楚它是否能够实现最大的优势：使机器人策略能够直接学习新运动以完成任务。在本文中，我们通过多任务人机共训练系统地探索这一潜力。我们引入了MotionTrans框架，其中包括数据收集系统、人类数据转换管道和加权共训练策略。通过同时共训练30个人机任务，我们直接将13项任务的运动从人类数据转移到可部署的端到端机器人策略中。值得注意的是，9项任务以零样本方式实现了非平凡的成功率。MotionTrans还显著提高了预训练-微调性能（成功率提高40%）。通过消融研究，我们还识别出了成功运动学习的关键因素：与机器人数据共训练和广泛的与任务相关运动覆盖。这些发现解锁了从人类数据进行运动级别学习的潜力，提供了其在训练机器人操作策略时有效应用的见解。所有数据、代码和模型权重均在此开源：[链接]。 

---
# Robust and Resilient Soft Robotic Object Insertion with Compliance-Enabled Contact Formation and Failure Recovery 

**Title (ZH)**: 具有良好柔韧性和恢复能力的软机器人物体插入：基于顺应性接触形成与故障恢复 

**Authors**: Mimo Shirasaka, Cristian C. Beltran-Hernandez, Masashi Hamaya, Yoshitaka Ushiku  

**Link**: [PDF](https://arxiv.org/pdf/2509.17666)  

**Abstract**: Object insertion tasks are prone to failures under pose uncertainties and environmental variations, traditionally requiring manual finetuning or controller retraining. We present a novel approach for robust and resilient object insertion using a passively compliant soft wrist that enables safe contact absorption through large deformations, without high-frequency control or force sensing. Our method structures insertion as compliance-enabled contact formations, sequential contact states that progressively constrain degrees of freedom, and integrates automated failure recovery strategies. Our key insight is that wrist compliance permits safe, repeated recovery attempts; hence, we refer to it as compliance-enabled failure recovery. We employ a pre-trained vision-language model (VLM) that assesses each skill execution from terminal poses and images, identifies failure modes, and proposes recovery actions by selecting skills and updating goals. In simulation, our method achieved an 83% success rate, recovering from failures induced by randomized conditions--including grasp misalignments up to 5 degrees, hole-pose errors up to 20mm, fivefold increases in friction, and previously unseen square/rectangular pegs--and we further validate the approach on a real robot. 

**Abstract (ZH)**: 基于柔腕的鲁棒自适应物体插入方法：通过大形变实现安全接触吸收，无高频控制和力感知的故障恢复策略 

---
# GeCCo - a Generalist Contact-Conditioned Policy for Loco-Manipulation Skills on Legged Robots 

**Title (ZH)**: GeCCo - 一种基于接触条件的通用腿足机器人操作技能策略 

**Authors**: Vassil Atanassov, Wanming Yu, Siddhant Gangapurwala, James Wilson, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2509.17582)  

**Abstract**: Most modern approaches to quadruped locomotion focus on using Deep Reinforcement Learning (DRL) to learn policies from scratch, in an end-to-end manner. Such methods often fail to scale, as every new problem or application requires time-consuming and iterative reward definition and tuning. We present Generalist Contact-Conditioned Policy (GeCCo) -- a low-level policy trained with Deep Reinforcement Learning that is capable of tracking arbitrary contact points on a quadruped robot. The strength of our approach is that it provides a general and modular low-level controller that can be reused for a wider range of high-level tasks, without the need to re-train new controllers from scratch. We demonstrate the scalability and robustness of our method by evaluating on a wide range of locomotion and manipulation tasks in a common framework and under a single generalist policy. These include a variety of gaits, traversing complex terrains (eg. stairs and slopes) as well as previously unseen stepping-stones and narrow beams, and interacting with objects (eg. pushing buttons, tracking trajectories). Our framework acquires new behaviors more efficiently, simply by combining a task-specific high-level contact planner and the pre-trained generalist policy. A supplementary video can be found at this https URL. 

**Abstract (ZH)**: 基于接触条件的通用 quadruped 运动政策（GeCCo）：一种低层级政策的学习方法 

---
# DyDexHandover: Human-like Bimanual Dynamic Dexterous Handover using RGB-only Perception 

**Title (ZH)**: DyDexHandover: 仅基于RGB感知的类人双臂动态灵巧交接 

**Authors**: Haoran Zhou, Yangwei You, Shuaijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17350)  

**Abstract**: Dynamic in air handover is a fundamental challenge for dual-arm robots, requiring accurate perception, precise coordination, and natural motion. Prior methods often rely on dynamics models, strong priors, or depth sensing, limiting generalization and naturalness. We present DyDexHandover, a novel framework that employs multi-agent reinforcement learning to train an end to end RGB based policy for bimanual object throwing and catching. To achieve more human-like behavior, the throwing policy is guided by a human policy regularization scheme, encouraging fluid and natural motion, and enhancing the generalization capability of the policy. A dual arm simulation environment was built in Isaac Sim for experimental evaluation. DyDexHandover achieves nearly 99 percent success on training objects and 75 percent on unseen objects, while generating human-like throwing and catching behaviors. To our knowledge, it is the first method to realize dual-arm in-air handover using only raw RGB perception. 

**Abstract (ZH)**: 基于多Agent强化学习的DyDexHandover：仅使用原始RGB感知实现双臂空中交接 

---
# Scalable Multi Agent Diffusion Policies for Coverage Control 

**Title (ZH)**: 可扩展的多代理扩散控制策略 

**Authors**: Frederic Vatnsdal, Romina Garcia Camargo, Saurav Agarwal, Alejandro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2509.17244)  

**Abstract**: We propose MADP, a novel diffusion-model-based approach for collaboration in decentralized robot swarms. MADP leverages diffusion models to generate samples from complex and high-dimensional action distributions that capture the interdependencies between agents' actions. Each robot conditions policy sampling on a fused representation of its own observations and perceptual embeddings received from peers. To evaluate this approach, we task a team of holonomic robots piloted by MADP to address coverage control-a canonical multi agent navigation problem. The policy is trained via imitation learning from a clairvoyant expert on the coverage control problem, with the diffusion process parameterized by a spatial transformer architecture to enable decentralized inference. We evaluate the system under varying numbers, locations, and variances of importance density functions, capturing the robustness demands of real-world coverage tasks. Experiments demonstrate that our model inherits valuable properties from diffusion models, generalizing across agent densities and environments, and consistently outperforming state-of-the-art baselines. 

**Abstract (ZH)**: 我们提出MADP：一种基于扩散模型的合作方法用于去中心化机器人 swarm。 

---
# Ratatouille: Imitation Learning Ingredients for Real-world Social Robot Navigation 

**Title (ZH)**: Ratatouille：现实社交机器人导航的模仿学习食材 

**Authors**: James R. Han, Mithun Vanniasinghe, Hshmat Sahak, Nicholas Rhinehart, Timothy D. Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2509.17204)  

**Abstract**: Scaling Reinforcement Learning to in-the-wild social robot navigation is both data-intensive and unsafe, since policies must learn through direct interaction and inevitably encounter collisions. Offline Imitation learning (IL) avoids these risks by collecting expert demonstrations safely, training entirely offline, and deploying policies zero-shot. However, we find that naively applying Behaviour Cloning (BC) to social navigation is insufficient; achieving strong performance requires careful architectural and training choices. We present Ratatouille, a pipeline and model architecture that, without changing the data, reduces collisions per meter by 6 times and improves success rate by 3 times compared to naive BC. We validate our approach in both simulation and the real world, where we collected over 11 hours of data on a dense university campus. We further demonstrate qualitative results in a public food court. Our findings highlight that thoughtful IL design, rather than additional data, can substantially improve safety and reliability in real-world social navigation. Video: this https URL. Code will be released after acceptance. 

**Abstract (ZH)**: 将强化学习扩展到户外社交机器人导航既数据密集又不安全，因为策略必须通过直接交互学习，并不可避免地会遇到碰撞。离线模仿学习（IL）通过安全地收集专家演示、完全离线训练并在零样本情况下部署策略来避免这些风险。然而，我们发现，将行为克隆（BC）直接应用于社交导航是不够的；获得出色的性能需要进行谨慎的体系结构和训练选择。我们提出了Ratatouille流水线和模型架构，在不改变数据的情况下，将每米碰撞次数减少6倍，并将成功率提高3倍，相较于简单的BC方法。我们在模拟和现实世界中验证了我们的方法，我们在一个密集的大学校园中收集了超过11小时的数据。我们还在一个公共餐饮区进一步展示了定性的结果。我们的研究结果表明，精心设计的IL设计相比额外的数据可以显著提高现实世界社交导航的安全性和可靠性。代码将在接收后发布。 

---
# MAST: Multi-Agent Spatial Transformer for Learning to Collaborate 

**Title (ZH)**: MAST：多智能体空间变换器学习协作 

**Authors**: Damian Owerko, Frederic Vatnsdal, Saurav Agarwal, Vijay Kumar, Alejandro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2509.17195)  

**Abstract**: This article presents a novel multi-agent spatial transformer (MAST) for learning communication policies in large-scale decentralized and collaborative multi-robot systems (DC-MRS). Challenges in collaboration in DC-MRS arise from: (i) partial observable states as robots make only localized perception, (ii) limited communication range with no central server, and (iii) independent execution of actions. The robots need to optimize a common task-specific objective, which, under the restricted setting, must be done using a communication policy that exhibits the desired collaborative behavior. The proposed MAST is a decentralized transformer architecture that learns communication policies to compute abstract information to be shared with other agents and processes the received information with the robot's own observations. The MAST extends the standard transformer with new positional encoding strategies and attention operations that employ windowing to limit the receptive field for MRS. These are designed for local computation, shift-equivariance, and permutation equivariance, making it a promising approach for DC-MRS. We demonstrate the efficacy of MAST on decentralized assignment and navigation (DAN) and decentralized coverage control. Efficiently trained using imitation learning in a centralized setting, the decentralized MAST policy is robust to communication delays, scales to large teams, and performs better than the baselines and other learning-based approaches. 

**Abstract (ZH)**: 一种用于大型分布式协作多机器人系统的新型多代理空间变换器（MAST）及其通信策略学习 

---
# History-Aware Visuomotor Policy Learning via Point Tracking 

**Title (ZH)**: 基于点跟踪的历史意识知觉运动策略学习 

**Authors**: Jingjing Chen, Hongjie Fang, Chenxi Wang, Shiquan Wang, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17141)  

**Abstract**: Many manipulation tasks require memory beyond the current observation, yet most visuomotor policies rely on the Markov assumption and thus struggle with repeated states or long-horizon dependencies. Existing methods attempt to extend observation horizons but remain insufficient for diverse memory requirements. To this end, we propose an object-centric history representation based on point tracking, which abstracts past observations into a compact and structured form that retains only essential task-relevant information. Tracked points are encoded and aggregated at the object level, yielding a compact history representation that can be seamlessly integrated into various visuomotor policies. Our design provides full history-awareness with high computational efficiency, leading to improved overall task performance and decision accuracy. Through extensive evaluations on diverse manipulation tasks, we show that our method addresses multiple facets of memory requirements - such as task stage identification, spatial memorization, and action counting, as well as longer-term demands like continuous and pre-loaded memory - and consistently outperforms both Markovian baselines and prior history-based approaches. Project website: this http URL 

**Abstract (ZH)**: 许多操作任务需要超越当前观察的记忆，然而大多数视觉-运动策略依赖于马尔可夫假设，因此在处理重复状态或长时依赖关系时表现不佳。现有方法试图扩展观察范围，但仍然无法满足多样的记忆需求。为了解决这一问题，我们提出了一种基于点追踪的对象中心历史表示方法，将过去观察抽象为紧凑且结构化的形式，仅保留与任务相关的信息。追踪点在对象级别进行编码和聚合，产生一种紧凑的历史表示形式，能够无缝集成到各种视觉-运动策略中。我们的设计提供全面的历史感知能力，并具有高计算效率，从而提高整体任务性能和决策准确性。通过在各种操作任务上的广泛评估，我们表明，我们的方法能够解决多种记忆需求方面的问题，如任务阶段识别、空间记忆、动作计数，以及长期需求如连续和预加载记忆，并且在多项指标上优于马尔可夫基线和先前的历史基方法。项目网站：该项目网址。 

---
# Imagine2Act: Leveraging Object-Action Motion Consistency from Imagined Goals for Robotic Manipulation 

**Title (ZH)**: Imagine2Act: 利用想象目标下的物体-动作运动一致性进行机器人 manipulation 

**Authors**: Liang Heng, Jiadong Xu, Yiwen Wang, Xiaoqi Li, Muhe Cai, Yan Shen, Juan Zhu, Guanghui Ren, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.17125)  

**Abstract**: Relational object rearrangement (ROR) tasks (e.g., insert flower to vase) require a robot to manipulate objects with precise semantic and geometric reasoning. Existing approaches either rely on pre-collected demonstrations that struggle to capture complex geometric constraints or generate goal-state observations to capture semantic and geometric knowledge, but fail to explicitly couple object transformation with action prediction, resulting in errors due to generative noise. To address these limitations, we propose Imagine2Act, a 3D imitation-learning framework that incorporates semantic and geometric constraints of objects into policy learning to tackle high-precision manipulation tasks. We first generate imagined goal images conditioned on language instructions and reconstruct corresponding 3D point clouds to provide robust semantic and geometric priors. These imagined goal point clouds serve as additional inputs to the policy model, while an object-action consistency strategy with soft pose supervision explicitly aligns predicted end-effector motion with generated object transformation. This design enables Imagine2Act to reason about semantic and geometric relationships between objects and predict accurate actions across diverse tasks. Experiments in both simulation and the real world demonstrate that Imagine2Act outperforms previous state-of-the-art policies. More visualizations can be found at this https URL. 

**Abstract (ZH)**: 关系对象重排（ROR）任务（例如，向花瓶中插入花）要求机器人通过精确的语义和几何推理来操控物体。现有的方法要么依赖于预先收集的示范，难以捕捉复杂的几何约束，要么生成目标状态观察来捕捉语义和几何知识，但未能明确地将对象变换与动作预测耦合，从而导致由于生成噪声引起的问题。为解决这些限制，我们提出了一种名为Imagine2Act的3D模仿学习框架，将物体的语义和几何约束整合到策略学习中以应对高精度操作任务。我们首先根据语言指令生成条件化的目标图像，并重建相应的3D点云以提供稳健的语义和几何先验。这些想象中的目标点云作为策略模型的额外输入，而对象-动作一致性策略结合软姿态监督将预测的末端执行器运动与生成的对象变换明确对齐。此设计使Imagine2Act能够推理物体之间的语义和几何关系，并在多种任务中预测准确的动作。在仿真和真实世界中的实验表明，Imagine2Act优于先前的最佳策略。更多可视化内容可访问此链接：this https URL。 

---
# RoboManipBaselines: A Unified Framework for Imitation Learning in Robotic Manipulation across Real and Simulated Environments 

**Title (ZH)**: RoboManipBaselines: 一种跨真实与模拟环境的机器人 manipulation 统一imitation learning框架 

**Authors**: Masaki Murooka, Tomohiro Motoda, Ryoichi Nakajo, Hanbit Oh, Koshi Makihara, Keisuke Shirai, Yukiyasu Domae  

**Link**: [PDF](https://arxiv.org/pdf/2509.17057)  

**Abstract**: RoboManipBaselines is an open framework for robot imitation learning that unifies data collection, training, and evaluation across simulation and real robots. We introduce it as a platform enabling systematic benchmarking of diverse tasks, robots, and multimodal policies with emphasis on integration, generality, extensibility, and reproducibility. 

**Abstract (ZH)**: RoboManipBaselines 是一个统一模拟与真实机器人使用的机器人模仿学习的开放框架，支持不同类型任务、机器人和多模态策略的系统性基准测试，强调集成、通用性、扩展性和可重复性。 

---
# FILIC: Dual-Loop Force-Guided Imitation Learning with Impedance Torque Control for Contact-Rich Manipulation Tasks 

**Title (ZH)**: FILIC: 力引导的双环imitation学习方法及其在接触丰富操作任务中的阻抗扭矩控制 

**Authors**: Haizhou Ge, Yufei Jia, Zheng Li, Yue Li, Zhixing Chen, Ruqi Huang, Guyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.17053)  

**Abstract**: Contact-rich manipulation is crucial for robots to perform tasks requiring precise force control, such as insertion, assembly, and in-hand manipulation. However, most imitation learning (IL) policies remain position-centric and lack explicit force awareness, and adding force/torque sensors to collaborative robot arms is often costly and requires additional hardware design. To overcome these issues, we propose FILIC, a Force-guided Imitation Learning framework with impedance torque control. FILIC integrates a Transformer-based IL policy with an impedance controller in a dual-loop structure, enabling compliant force-informed, force-executed manipulation. For robots without force/torque sensors, we introduce a cost-effective end-effector force estimator using joint torque measurements through analytical Jacobian-based inversion while compensating with model-predicted torques from a digital twin. We also design complementary force feedback frameworks via handheld haptics and VR visualization to improve demonstration quality. Experiments show that FILIC significantly outperforms vision-only and joint-torque-based methods, achieving safer, more compliant, and adaptable contact-rich manipulation. Our code can be found in this https URL. 

**Abstract (ZH)**: 基于阻抗扭矩控制的力导向模仿学习框架 

---
# Generalized Momenta-Based Koopman Formalism for Robust Control of Euler-Lagrangian Systems 

**Title (ZH)**: 基于广义动量的Koopman形式主义在Euler-Lagrange系统鲁棒控制中的应用 

**Authors**: Rajpal Singh, Aditya Singh, Chidre Shravista Kashyap, Jishnu Keshavan  

**Link**: [PDF](https://arxiv.org/pdf/2509.17010)  

**Abstract**: This paper presents a novel Koopman operator formulation for Euler Lagrangian dynamics that employs an implicit generalized momentum-based state space representation, which decouples a known linear actuation channel from state dependent dynamics and makes the system more amenable to linear Koopman modeling. By leveraging this structural separation, the proposed formulation only requires to learn the unactuated dynamics rather than the complete actuation dependent system, thereby significantly reducing the number of learnable parameters, improving data efficiency, and lowering overall model complexity. In contrast, conventional explicit formulations inherently couple inputs with the state dependent terms in a nonlinear manner, making them more suitable for bilinear Koopman models, which are more computationally expensive to train and deploy. Notably, the proposed scheme enables the formulation of linear models that achieve superior prediction performance compared to conventional bilinear models while remaining substantially more efficient. To realize this framework, we present two neural network architectures that construct Koopman embeddings from actuated or unactuated data, enabling flexible and efficient modeling across different tasks. Robustness is ensured through the integration of a linear Generalized Extended State Observer (GESO), which explicitly estimates disturbances and compensates for them in real time. The combined momentum-based Koopman and GESO framework is validated through comprehensive trajectory tracking simulations and experiments on robotic manipulators, demonstrating superior accuracy, robustness, and learning efficiency relative to state of the art alternatives. 

**Abstract (ZH)**: 一种基于隐式广义动量状态空间表示的Koopman算子欧拉拉格朗日动力学新形式化方法 

---
# IDfRA: Self-Verification for Iterative Design in Robotic Assembly 

**Title (ZH)**: IDfRA：基于自验证的迭代设计在机器人装配中的应用 

**Authors**: Nishka Khendry, Christos Margadji, Sebastian W. Pattinson  

**Link**: [PDF](https://arxiv.org/pdf/2509.16998)  

**Abstract**: As robots proliferate in manufacturing, Design for Robotic Assembly (DfRA), which is designing products for efficient automated assembly, is increasingly important. Traditional approaches to DfRA rely on manual planning, which is time-consuming, expensive and potentially impractical for complex objects. Large language models (LLM) have exhibited proficiency in semantic interpretation and robotic task planning, stimulating interest in their application to the automation of DfRA. But existing methodologies typically rely on heuristic strategies and rigid, hard-coded physics simulators that may not translate into real-world assembly contexts. In this work, we present Iterative Design for Robotic Assembly (IDfRA), a framework using iterative cycles of planning, execution, verification, and re-planning, each informed by self-assessment, to progressively enhance design quality within a fixed yet initially under-specified environment, thereby eliminating the physics simulation with the real world itself. The framework accepts as input a target structure together with a partial environmental representation. Through successive refinement, it converges toward solutions that reconcile semantic fidelity with physical feasibility. Empirical evaluation demonstrates that IDfRA attains 73.3\% top-1 accuracy in semantic recognisability, surpassing the baseline on this metric. Moreover, the resulting assembly plans exhibit robust physical feasibility, achieving an overall 86.9\% construction success rate, with design quality improving across iterations, albeit not always monotonically. Pairwise human evaluation further corroborates the advantages of IDfRA relative to alternative approaches. By integrating self-verification with context-aware adaptation, the framework evidences strong potential for deployment in unstructured manufacturing scenarios. 

**Abstract (ZH)**: 基于迭代设计的机器人装配（IDfRA）框架 

---
# A Reliable Robot Motion Planner in Complex Real-world Environments via Action Imagination 

**Title (ZH)**: 基于行动想象的复杂现实环境可靠机器人运动规划 

**Authors**: Chengjin Wang, Yanmin Zhou, Zhipeng Wang, Zheng Yan, Feng Luan, Shuo Jiang, Runjie Shen, Hongrui Sang, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2509.16963)  

**Abstract**: Humans and animals can make real-time adjustments to movements by imagining their action outcomes to prevent unanticipated or even catastrophic motion failures in unknown unstructured environments. Action imagination, as a refined sensorimotor strategy, leverages perception-action loops to handle physical interaction-induced uncertainties in perception and system modeling within complex systems. Inspired by the action-awareness capability of animal intelligence, this study proposes an imagination-inspired motion planner (I-MP) framework that specifically enhances robots' action reliability by imagining plausible spatial states for approaching. After topologizing the workspace, I-MP build perception-action loop enabling robots autonomously build contact models. Leveraging fixed-point theory and Hausdorff distance, the planner computes convergent spatial states under interaction characteristics and mission constraints. By homogenously representing multi-dimensional environmental characteristics through work, the robot can approach the imagined spatial states via real-time computation of energy gradients. Consequently, experimental results demonstrate the practicality and robustness of I-MP in complex cluttered environments. 

**Abstract (ZH)**: 人类和动物可以通过想象动作结果在未知非结构化环境中实时调整动作，以防止意外甚至灾难性的运动失败。基于这一感知-动作回路策略，行动想象作为一种精炼的运动感知-运动策略，能够处理物理交互引起的感觉和系统建模中的不确定性。受动物智能行动意识能力的启发，本研究提出了一种想象启发式运动规划框架（I-MP），该框架通过想象可实现的空间状态来特别增强机器人的动作可靠性。在拓扑化工作空间后，I-MP构建感知-动作回路，使机器人能够自主建立接触模型。通过使用不动点理论和哈斯多夫距离，规划器在考虑交互特性和任务约束的情况下计算收敛的空间状态。通过工作多维环境特征的一致表示，机器人可以通过实时能量梯度计算实现对想象的空间状态的接近。因而，实验结果表明I-MP在复杂拥挤环境中的实用性和鲁棒性。 

---
# SwarmChat: An LLM-Based, Context-Aware Multimodal Interaction System for Robotic Swarms 

**Title (ZH)**: SwarmChat：一种基于LLM的上下文感知多模态交互系统用于机器人 swarm 

**Authors**: Ettilla Mohiuddin Eumi, Hussein Abbass, Nadine Marcus  

**Link**: [PDF](https://arxiv.org/pdf/2509.16920)  

**Abstract**: Traditional Human-Swarm Interaction (HSI) methods often lack intuitive real-time adaptive interfaces, making decision making slower and increasing cognitive load while limiting command flexibility. To solve this, we present SwarmChat, a context-aware, multimodal interaction system powered by Large Language Models (LLMs). SwarmChat enables users to issue natural language commands to robotic swarms using multiple modalities, such as text, voice, or teleoperation. The system integrates four LLM-based modules: Context Generator, Intent Recognition, Task Planner, and Modality Selector. These modules collaboratively generate context from keywords, detect user intent, adapt commands based on real-time robot state, and suggest optimal communication modalities. Its three-layer architecture offers a dynamic interface with both fixed and customizable command options, supporting flexible control while optimizing cognitive effort. The preliminary evaluation also shows that the SwarmChat's LLM modules provide accurate context interpretation, relevant intent recognition, and effective command delivery, achieving high user satisfaction. 

**Abstract (ZH)**: 基于大语言模型的上下文感知多模态 swarmchat 交互系统 

---
# End2Race: Efficient End-to-End Imitation Learning for Real-Time F1Tenth Racing 

**Title (ZH)**: End2Race: 高效的端到端 imitative 学习算法以实现实时 F1Tenth 赛车竞速 

**Authors**: Zhijie Qiao, Haowei Li, Zhong Cao, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16894)  

**Abstract**: F1Tenth is a widely adopted reduced-scale platform for developing and testing autonomous racing algorithms, hosting annual competitions worldwide. With high operating speeds, dynamic environments, and head-to-head interactions, autonomous racing requires algorithms that diverge from those in classical autonomous driving. Training such algorithms is particularly challenging: the need for rapid decision-making at high speeds severely limits model capacity. To address this, we propose End2Race, a novel end-to-end imitation learning algorithm designed for head-to-head autonomous racing. End2Race leverages a Gated Recurrent Unit (GRU) architecture to capture continuous temporal dependencies, enabling both short-term responsiveness and long-term strategic planning. We also adopt a sigmoid-based normalization function that transforms raw LiDAR scans into spatial pressure tokens, facilitating effective model training and convergence. The algorithm is extremely efficient, achieving an inference time of less than 0.5 milliseconds on a consumer-class GPU. Experiments in the F1Tenth simulator demonstrate that End2Race achieves a 94.2% safety rate across 2,400 overtaking scenarios, each with an 8-second time limit, and successfully completes overtakes in 59.2% of cases. This surpasses previous methods and establishes ours as a leading solution for the F1Tenth racing testbed. Code is available at this https URL. 

**Abstract (ZH)**: F1Tenth：面向头对头自动驾驶赛车的一种新型端到端imitation学习算法 

---
# Benchmarking Offline Reinforcement Learning for Emotion-Adaptive Social Robotics 

**Title (ZH)**: 基于离线强化学习的情绪自适应社会机器人benchmark研究 

**Authors**: Soon Jynn Chu, Raju Gottumukkala, Alan Barhorst  

**Link**: [PDF](https://arxiv.org/pdf/2509.16858)  

**Abstract**: The ability of social robots to respond to human emotions is crucial for building trust and acceptance in human-robot collaborative environments. However, developing such capabilities through online reinforcement learning is sometimes impractical due to the prohibitive cost of data collection and the risk of generating unsafe behaviors. In this paper, we study the use of offline reinforcement learning as a practical and efficient alternative. This technique uses pre-collected data to enable emotion-adaptive social robots. We present a system architecture that integrates multimodal sensing and recognition, decision-making, and adaptive responses. Using a limited dataset from a human-robot game-playing scenario, we establish a benchmark for comparing offline reinforcement learning algorithms that do not require an online environment. Our results show that BCQ and CQL are more robust to data sparsity, achieving higher state-action values compared to NFQ, DQN, and DDQN. This work establishes a foundation for benchmarking offline RL in emotion-adaptive robotics and informs future deployment in real-world HRI. Our findings provide empirical insight into the performance of offline reinforcement learning algorithms in data-constrained HRI. This work establishes a foundation for benchmarking offline RL in emotion-adaptive robotics and informs its future deployment in real-world HRI, such as in conversational agents, educational partners, and personal assistants, require reliable emotional responsiveness. 

**Abstract (ZH)**: 社会机器人响应人类情绪的能力对于在人机协作环境中建立信任和接受至关重要。然而，通过在线强化学习开发此类能力因数据收集成本高昂和产生不安全行为的风险而有时不可行。本文研究了使用离线强化学习作为实用而高效的替代方案。该技术利用预先收集的数据来实现情绪自适应的社会机器人。我们提出了一种系统架构，整合了多模态感知与识别、决策制定和自适应响应。利用人类机器人游戏场景的有限数据集，我们建立了不需要在线环境的离线强化学习算法基准。结果显示，BCQ和CQL在数据稀疏性方面更为稳健，相对于NFQ、DQN和DDQN，其状态-动作值更高。本工作确立了在情绪自适应机器人领域评估离线RL的基础，并为其实用部署提供了指导，应用于对话代理、教育伙伴和个人助理等实际应用中需要可靠的emotion自适应响应。 

---
# Robot Learning with Sparsity and Scarcity 

**Title (ZH)**: 稀疏与稀缺约束下的机器人学习 

**Authors**: Jingxi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16834)  

**Abstract**: Unlike in language or vision, one of the fundamental challenges in robot learning is the lack of access to vast data resources. We can further break down the problem into (1) data sparsity from the angle of data representation and (2) data scarcity from the angle of data quantity. In this thesis, I will discuss selected works on two domains: (1) tactile sensing and (2) rehabilitation robots, which are exemplars of data sparsity and scarcity, respectively. Tactile sensing is an essential modality for robotics, but tactile data are often sparse, and for each interaction with the physical world, tactile sensors can only obtain information about the local area of contact. I will discuss my work on learning vision-free tactile-only exploration and manipulation policies through model-free reinforcement learning to make efficient use of sparse tactile information. On the other hand, rehabilitation robots are an example of data scarcity to the extreme due to the significant challenge of collecting biosignals from disabled-bodied subjects at scale for training. I will discuss my work in collaboration with the medical school and clinicians on intent inferral for stroke survivors, where a hand orthosis developed in our lab collects a set of biosignals from the patient and uses them to infer the activity that the patient intends to perform, so the orthosis can provide the right type of physical assistance at the right moment. My work develops machine learning algorithms that enable intent inferral with minimal data, including semi-supervised, meta-learning, and generative AI methods. 

**Abstract (ZH)**: 不同于语言或视觉，机器人学习中一个基本的挑战是没有接入到大量的数据资源。我们可以将该问题进一步分解为从数据表示角度的数据稀疏性（1）和从数据数量角度的数据稀缺性（2）。在这篇论文中，我将讨论两个领域的精选工作：（1）触觉感知和（2）康复机器人，它们分别代表了数据稀疏性和数据稀缺性的例子。触觉感知是机器人技术中的一种基本感知模态，但触觉数据通常很稀疏，每次与物理世界互动时，触觉传感器只能获取接触区域的局部信息。我将讨论通过无模型强化学习学习基于触觉的无视觉探索和操作策略的工作，以有效利用稀疏的触觉信息。另一方面，康复机器人是一个极端的数据稀缺性案例，因为大规模收集残障患者生理信号进行训练具有重大挑战。我将讨论与医学院和临床医生合作，在中风幸存者中实现辅助意图推断的工作，其中我们的实验室开发的手部矫形器收集患者的生理信号并通过这些信号推断患者意图执行的活动，以便矫形器能够在适当的时候提供适当的物理辅助。我的工作开发了能够用最少数据实现意图推断的机器学习算法，包括半监督学习、元学习和生成AI方法。 

---
# Factorizing Diffusion Policies for Observation Modality Prioritization 

**Title (ZH)**: 面向观测模态优先级的扩散政策分解 

**Authors**: Omkar Patil, Prabin Rath, Kartikay Pangaonkar, Eric Rosen, Nakul Gopalan  

**Link**: [PDF](https://arxiv.org/pdf/2509.16830)  

**Abstract**: Diffusion models have been extensively leveraged for learning robot skills from demonstrations. These policies are conditioned on several observational modalities such as proprioception, vision and tactile. However, observational modalities have varying levels of influence for different tasks that diffusion polices fail to capture. In this work, we propose 'Factorized Diffusion Policies' abbreviated as FDP, a novel policy formulation that enables observational modalities to have differing influence on the action diffusion process by design. This results in learning policies where certain observations modalities can be prioritized over the others such as $\texttt{vision>tactile}$ or $\texttt{proprioception>vision}$. FDP achieves modality prioritization by factorizing the observational conditioning for diffusion process, resulting in more performant and robust policies. Our factored approach shows strong performance improvements in low-data regimes with $15\%$ absolute improvement in success rate on several simulated benchmarks when compared to a standard diffusion policy that jointly conditions on all input modalities. Moreover, our benchmark and real-world experiments show that factored policies are naturally more robust with $40\%$ higher absolute success rate across several visuomotor tasks under distribution shifts such as visual distractors or camera occlusions, where existing diffusion policies fail catastrophically. FDP thus offers a safer and more robust alternative to standard diffusion policies for real-world deployment. Videos are available at this https URL . 

**Abstract (ZH)**: 因子化扩散策略：一种新型观测模态优先的机器人技能学习方法 

---
# HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos 

**Title (ZH)**: HDMI：从人类视频学习互动类人全身控制 

**Authors**: Haoyang Weng, Yitang Li, Nikhil Sobanbabu, Zihan Wang, Zhengyi Luo, Tairan He, Deva Ramanan, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.16757)  

**Abstract**: Enabling robust whole-body humanoid-object interaction (HOI) remains challenging due to motion data scarcity and the contact-rich nature. We present HDMI (HumanoiD iMitation for Interaction), a simple and general framework that learns whole-body humanoid-object interaction skills directly from monocular RGB videos. Our pipeline (i) extracts and retargets human and object trajectories from unconstrained videos to build structured motion datasets, (ii) trains a reinforcement learning (RL) policy to co-track robot and object states with three key designs: a unified object representation, a residual action space, and a general interaction reward, and (iii) zero-shot deploys the RL policies on real humanoid robots. Extensive sim-to-real experiments on a Unitree G1 humanoid demonstrate the robustness and generality of our approach: HDMI achieves 67 consecutive door traversals and successfully performs 6 distinct loco-manipulation tasks in the real world and 14 tasks in simulation. Our results establish HDMI as a simple and general framework for acquiring interactive humanoid skills from human videos. 

**Abstract (ZH)**: 基于单目RGB视频直接从人类表演中学习全身人形机器人-物体交互技能：HDMI框架 

---
# KungfuBot2: Learning Versatile Motion Skills for Humanoid Whole-Body Control 

**Title (ZH)**: KungfuBot2：学习全面身体控制的多样化运动技能 

**Authors**: Jinrui Han, Weiji Xie, Jiakun Zheng, Jiyuan Shi, Weinan Zhang, Ting Xiao, Chenjia Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.16638)  

**Abstract**: Learning versatile whole-body skills by tracking various human motions is a fundamental step toward general-purpose humanoid robots. This task is particularly challenging because a single policy must master a broad repertoire of motion skills while ensuring stability over long-horizon sequences. To this end, we present VMS, a unified whole-body controller that enables humanoid robots to learn diverse and dynamic behaviors within a single policy. Our framework integrates a hybrid tracking objective that balances local motion fidelity with global trajectory consistency, and an Orthogonal Mixture-of-Experts (OMoE) architecture that encourages skill specialization while enhancing generalization across motions. A segment-level tracking reward is further introduced to relax rigid step-wise matching, enhancing robustness when handling global displacements and transient inaccuracies. We validate VMS extensively in both simulation and real-world experiments, demonstrating accurate imitation of dynamic skills, stable performance over minute-long sequences, and strong generalization to unseen motions. These results highlight the potential of VMS as a scalable foundation for versatile humanoid whole-body control. The project page is available at this https URL. 

**Abstract (ZH)**: 通过跟踪各种人类动作学习全方位身体技能是通用 humanoid 机器人研究中的一个基础步骤。这一任务极具挑战性，因为单一策略必须掌握广泛的运动技能，并在整个长时间序列中保持稳定性。为此，我们提出了一种统一的全方位控制器 VMS，使类人机器人能够在单一策略中学习多样且动态的行为。我们的框架集成了混合跟踪目标，平衡局部运动保真度与全局轨迹一致性，并采用了正交混合专家（OMoE）架构，鼓励技能专一化同时增强跨动作的泛化能力。我们还引入了段级跟踪奖励，以放松刚性步进匹配，增强在处理全局位移和瞬态不准确性时的鲁棒性。我们在仿真和真实世界实验中广泛验证了 VMS，结果显示其在动态技能仿真的准确度、长时间序列中的稳定性能以及对未见动作的强大泛化能力。这些结果突显了 VMS 作为可扩展的多功能类人机器人全方位控制基础的潜力。项目页面可在该网址访问。 

---
# Video-to-BT: Generating Reactive Behavior Trees from Human Demonstration Videos for Robotic Assembly 

**Title (ZH)**: 视频到BT：从人类示范视频生成机器人装配的反应型行为树 

**Authors**: Xiwei Zhao, Yiwei Wang, Yansong Wu, Fan Wu, Teng Sun, Zhonghua Miao, Sami Haddadin, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.16611)  

**Abstract**: Modern manufacturing demands robotic assembly systems with enhanced flexibility and reliability. However, traditional approaches often rely on programming tailored to each product by experts for fixed settings, which are inherently inflexible to product changes and lack the robustness to handle variations. As Behavior Trees (BTs) are increasingly used in robotics for their modularity and reactivity, we propose a novel hierarchical framework, Video-to-BT, that seamlessly integrates high-level cognitive planning with low-level reactive control, with BTs serving both as the structured output of planning and as the governing structure for execution. Our approach leverages a Vision-Language Model (VLM) to decompose human demonstration videos into subtasks, from which Behavior Trees are generated. During the execution, the planned BTs combined with real-time scene interpretation enable the system to operate reactively in the dynamic environment, while VLM-driven replanning is triggered upon execution failure. This closed-loop architecture ensures stability and adaptivity. We validate our framework on real-world assembly tasks through a series of experiments, demonstrating high planning reliability, robust performance in long-horizon assembly tasks, and strong generalization across diverse and perturbed conditions. Project website: this https URL 

**Abstract (ZH)**: 现代制造需求高度灵活可靠的机器人装配系统。然而，传统方法往往依赖专家为固定配置编写针对每个产品的专用程序，这些配置本质上对产品变化缺乏灵活性，并且处理变异性时缺乏稳健性。鉴于行为树（BTs）因其模块性和反应性在机器人领域中的广泛应用，我们提出了一种新颖的层次框架——从视频到行为树（Video-to-BT），该框架无缝地将高层次的认知规划与低层次的反应性控制结合起来，其中行为树既作为规划的结构化输出，又作为执行的指导结构。我们的方法利用视觉语言模型（VLM）将人类演示视频分解为子任务，从中生成行为树。在执行过程中，结合策划的行为树与实时场景解释使系统能够在动态环境中进行反应性操作，而执行失败时由VLM驱动重新规划被触发。这种闭环架构确保了稳定性和适应性。我们通过一系列实验在实际装配任务中验证了该框架，证明了其高规划可靠性、长期装配任务中的鲁棒性能以及在多种变化条件下的强泛化能力。项目网站：https://this-url 

---
# A Framework for Optimal Ankle Design of Humanoid Robots 

**Title (ZH)**: humanoïdes机器人踝关节设计的优化框架 

**Authors**: Guglielmo Cervettini, Roberto Mauceri, Alex Coppola, Fabio Bergonti, Luca Fiorio, Marco Maggiali, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.16469)  

**Abstract**: The design of the humanoid ankle is critical for safe and efficient ground interaction. Key factors such as mechanical compliance and motor mass distribution have driven the adoption of parallel mechanism architectures. However, selecting the optimal configuration depends on both actuator availability and task requirements. We propose a unified methodology for the design and evaluation of parallel ankle mechanisms. A multi-objective optimization synthesizes the mechanism geometry, the resulting solutions are evaluated using a scalar cost function that aggregates key performance metrics for cross-architecture comparison. We focus on two representative architectures: the Spherical-Prismatic-Universal (SPU) and the Revolute-Spherical-Universal (RSU). For both, we resolve the kinematics, and for the RSU, introduce a parameterization that ensures workspace feasibility and accelerates optimization. We validate our approach by redesigning the ankle of an existing humanoid robot. The optimized RSU consistently outperforms both the original serial design and a conventionally engineered RSU, reducing the cost function by up to 41% and 14%, respectively. 

**Abstract (ZH)**: 人形踝关节的设计对于安全高效的地面交互至关重要。关键因素如机械顺应性和电机质量分布推动了并联机构架构的应用。然而，选择最优配置取决于可用执行器和任务需求。我们提出了一种统一的方法用于并联踝关节机构的设计与评估。多目标优化合成机构几何结构，结果通过标量成本函数进行评估，该函数汇总了关键性能指标以实现跨架构比较。我们重点关注两种代表架构：球型-柱型-通用型（SPU）和转动-球型-通用型（RSU）。对于两者，我们解决了其运动学，并为RSU引入了参数化方法以确保工作空间可行性和加速优化。通过重新设计现有 humanoid 机器人踝关节来验证我们的方法。优化后的 RSU 一致优于原始的串联设计和传统设计工程的 RSU，分别降低成本函数多达 41% 和 14%。 

---
# FiLM-Nav: Efficient and Generalizable Navigation via VLM Fine-tuning 

**Title (ZH)**: FiLM-Nav: 通过VLM微调实现高效且通用的导航 

**Authors**: Naoki Yokoyama, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2509.16445)  

**Abstract**: Enabling robotic assistants to navigate complex environments and locate objects described in free-form language is a critical capability for real-world deployment. While foundation models, particularly Vision-Language Models (VLMs), offer powerful semantic understanding, effectively adapting their web-scale knowledge for embodied decision-making remains a key challenge. We present FiLM-Nav (Fine-tuned Language Model for Navigation), an approach that directly fine-tunes pre-trained VLM as the navigation policy. In contrast to methods that use foundation models primarily in a zero-shot manner or for map annotation, FiLM-Nav learns to select the next best exploration frontier by conditioning directly on raw visual trajectory history and the navigation goal. Leveraging targeted simulated embodied experience allows the VLM to ground its powerful pre-trained representations in the specific dynamics and visual patterns relevant to goal-driven navigation. Critically, fine-tuning on a diverse data mixture combining ObjectNav, OVON, ImageNav, and an auxiliary spatial reasoning task proves essential for achieving robustness and broad generalization. FiLM-Nav sets a new state-of-the-art in both SPL and success rate on HM3D ObjectNav among open-vocabulary methods, and sets a state-of-the-art SPL on the challenging HM3D-OVON benchmark, demonstrating strong generalization to unseen object categories. Our work validates that directly fine-tuning VLMs on diverse simulated embodied data is a highly effective pathway towards generalizable and efficient semantic navigation capabilities. 

**Abstract (ZH)**: 使机器人助手能够导航复杂环境并在自由形式的语言中定位物体是其实用部署的关键能力。虽然基础模型，尤其是视觉-语言模型（VLMs），提供了强大的语义理解能力，但将其网络规模知识有效适应于身体化决策仍然是一个关键挑战。我们介绍了FiLM-Nav（精调语言模型用于导航），这是一种直接对预训练的VLM进行精调以作为导航策略的方法。与主要以零样本方式或地图标注使用基础模型的方法不同，FiLM-Nav 通过直接条件化于原始视觉轨迹历史和导航目标来学习选择下一个最佳探索前沿。利用有针对性的模拟身体化体验使VLM能够将其强大的预训练表示与目标驱动导航相关的具体动力学和视觉模式联系起来。最关键的是，结合ObjectNav、OVON、ImageNav以及一个辅助的空间推理任务的数据进行精调对于实现鲁棒性和广泛的一般化至关重要。FiLM-Nav在HM3D ObjectNav的开放词汇方法中达到了新的最佳性能，在具有挑战性的HM3D-OVON基准上的SPL也达到了最先进的水平，证明了其对未见过的对象类别具有强大的泛化能力。我们的研究证实，直接对多样化的模拟身体化数据进行VLM的精调是实现可泛化且高效的语义导航能力的有效途径。 

---
# End-to-end RL Improves Dexterous Grasping Policies 

**Title (ZH)**: 端到端RL提升灵巧抓取策略 

**Authors**: Ritvik Singh, Karl Van Wyk, Pieter Abbeel, Jitendra Malik, Nathan Ratliff, Ankur Handa  

**Link**: [PDF](https://arxiv.org/pdf/2509.16434)  

**Abstract**: This work explores techniques to scale up image-based end-to-end learning for dexterous grasping with an arm + hand system. Unlike state-based RL, vision-based RL is much more memory inefficient, resulting in relatively low batch sizes, which is not amenable for algorithms like PPO. Nevertheless, it is still an attractive method as unlike the more commonly used techniques which distill state-based policies into vision networks, end-to-end RL can allow for emergent active vision behaviors. We identify a key bottleneck in training these policies is the way most existing simulators scale to multiple GPUs using traditional data parallelism techniques. We propose a new method where we disaggregate the simulator and RL (both training and experience buffers) onto separate GPUs. On a node with four GPUs, we have the simulator running on three of them, and PPO running on the fourth. We are able to show that with the same number of GPUs, we can double the number of existing environments compared to the previous baseline of standard data parallelism. This allows us to train vision-based environments, end-to-end with depth, which were previously performing far worse with the baseline. We train and distill both depth and state-based policies into stereo RGB networks and show that depth distillation leads to better results, both in simulation and reality. This improvement is likely due to the observability gap between state and vision policies which does not exist when distilling depth policies into stereo RGB. We further show that the increased batch size brought about by disaggregated simulation also improves real world performance. When deploying in the real world, we improve upon the previous state-of-the-art vision-based results using our end-to-end policies. 

**Abstract (ZH)**: 基于视觉的端到端学习在手臂+手系统灵巧抓取中的扩展技术研究 

---
# Subteaming and Adaptive Formation Control for Coordinated Multi-Robot Navigation 

**Title (ZH)**: 基于协调多机器人导航的子团队划分与自适应 formations 控制 

**Authors**: Zihao Deng, Peng Gao, Williard Joshua Jose, Maggie Wigness, John Rogers, Brian Reily, Christopher Reardon, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16412)  

**Abstract**: Coordinated multi-robot navigation is essential for robots to operate as a team in diverse environments. During navigation, robot teams usually need to maintain specific formations, such as circular formations to protect human teammates at the center. However, in complex scenarios such as narrow corridors, rigidly preserving predefined formations can become infeasible. Therefore, robot teams must be capable of dynamically splitting into smaller subteams and adaptively controlling the subteams to navigate through such scenarios while preserving formations. To enable this capability, we introduce a novel method for SubTeaming and Adaptive Formation (STAF), which is built upon a unified hierarchical learning framework: (1) high-level deep graph cut for team splitting, (2) intermediate-level graph learning for facilitating coordinated navigation among subteams, and (3) low-level policy learning for controlling individual mobile robots to reach their goal positions while avoiding collisions. To evaluate STAF, we conducted extensive experiments in both indoor and outdoor environments using robotics simulations and physical robot teams. Experimental results show that STAF enables the novel capability for subteaming and adaptive formation control, and achieves promising performance in coordinated multi-robot navigation through challenging scenarios. More details are available on the project website: this https URL. 

**Abstract (ZH)**: 协调多机器人导航对于机器人在多样化环境中作为团队运作至关重要。在导航过程中，机器人团队通常需要维护特定的队形，例如，圆形队形以保护中心的人类队友。然而，在狭窄走廊等复杂场景中，严格保持预定义的队形可能变得不切实际。因此，机器人团队必须能够动态分裂成较小的子团队，并根据需要自主控制子团队以穿越这些场景并维持队形。为了实现这一能力，我们提出了一种新的子团队和自适应队形控制（STAF）方法，该方法基于统一的层次学习框架：（1）高层深度图切割用于团队分裂，（2）中间层图学习促进子团队之间的协调导航，（3）低层策略学习控制单个移动机器人到达目标位置并避免碰撞。为了评估STAF，我们使用机器人仿真和物理机器人团队在室内外环境中进行了广泛实验。实验结果表明，STAF能够实现新的子团队和自适应队形控制能力，并在通过具有挑战性的场景进行协调多机器人导航时取得了令人鼓舞的性能。更多详情请参见项目网站：this https URL。 

---
# Dynamic Objects Relocalization in Changing Environments with Flow Matching 

**Title (ZH)**: 在变化环境中通过流匹配进行动态物体重定位 

**Authors**: Francesco Argenziano, Miguel Saavedra-Ruiz, Sacha Morin, Daniele Nardi, Liam Paull  

**Link**: [PDF](https://arxiv.org/pdf/2509.16398)  

**Abstract**: Task and motion planning are long-standing challenges in robotics, especially when robots have to deal with dynamic environments exhibiting long-term dynamics, such as households or warehouses. In these environments, long-term dynamics mostly stem from human activities, since previously detected objects can be moved or removed from the scene. This adds the necessity to find such objects again before completing the designed task, increasing the risk of failure due to missed relocalizations. However, in these settings, the nature of such human-object interactions is often overlooked, despite being governed by common habits and repetitive patterns. Our conjecture is that these cues can be exploited to recover the most likely objects' positions in the scene, helping to address the problem of unknown relocalization in changing environments. To this end we propose FlowMaps, a model based on Flow Matching that is able to infer multimodal object locations over space and time. Our results present statistical evidence to support our hypotheses, opening the way to more complex applications of our approach. The code is publically available at this https URL 

**Abstract (ZH)**: 任务规划与运动规划是机器人技术中的长期挑战，尤其是在机器人需要处理动态环境（如家庭或仓库）时，这些环境表现出长期动态变化。在这种环境中，长期动态变化主要源自人类活动，因为之前检测到的物体可能会被移动或移除。这增加了在重新定位过程中遗漏目标物体的风险，从而增加了任务执行失败的风险。然而，在这些场景中，人类与物体的交互方式往往被忽视，尽管它们遵循常见习惯和重复模式。我们推测，这些线索可以被利用来恢复场景中物体最可能的位置，从而解决变化环境中重新定位未知的问题。为此，我们提出了一种基于流匹配的FlowMaps模型，能够在空间和时间上推断多模态物体位置。我们的实验结果提供了统计证据支持上述假设，为更复杂的应用打开了大门。相关代码已公开，可通过此链接访问。 

---
# EmbodiedSplat: Personalized Real-to-Sim-to-Real Navigation with Gaussian Splats from a Mobile Device 

**Title (ZH)**: 基于高斯斑点的个性化实景到仿真到实景导航移动设备上的 gaussian splats 

**Authors**: Gunjan Chhablani, Xiaomeng Ye, Muhammad Zubair Irshad, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2509.17430)  

**Abstract**: The field of Embodied AI predominantly relies on simulation for training and evaluation, often using either fully synthetic environments that lack photorealism or high-fidelity real-world reconstructions captured with expensive hardware. As a result, sim-to-real transfer remains a major challenge. In this paper, we introduce EmbodiedSplat, a novel approach that personalizes policy training by efficiently capturing the deployment environment and fine-tuning policies within the reconstructed scenes. Our method leverages 3D Gaussian Splatting (GS) and the Habitat-Sim simulator to bridge the gap between realistic scene capture and effective training environments. Using iPhone-captured deployment scenes, we reconstruct meshes via GS, enabling training in settings that closely approximate real-world conditions. We conduct a comprehensive analysis of training strategies, pre-training datasets, and mesh reconstruction techniques, evaluating their impact on sim-to-real predictivity in real-world scenarios. Experimental results demonstrate that agents fine-tuned with EmbodiedSplat outperform both zero-shot baselines pre-trained on large-scale real-world datasets (HM3D) and synthetically generated datasets (HSSD), achieving absolute success rate improvements of 20\% and 40\% on real-world Image Navigation task. Moreover, our approach yields a high sim-vs-real correlation (0.87--0.97) for the reconstructed meshes, underscoring its effectiveness in adapting policies to diverse environments with minimal effort. Project page: this https URL 

**Abstract (ZH)**: 基于体感的AI领域主要依赖仿真进行训练和评估，通常使用缺乏写实性的完全合成环境或使用昂贵硬件捕捉的真实场景高保真重建。因此，仿真到现实世界的迁移仍然是一个重大挑战。本文介绍了一种名为EmbodiedSplat的新方法，该方法通过高效捕捉部署环境并在此重建场景中微调策略，实现个性化策略训练。该方法利用3D高斯点成图（GS）和Habitat-Sim仿真器，桥接了现实场景捕捉与有效训练环境之间的差距。利用iPhone拍摄的部署场景，通过GS重建网格，使训练环境能够更贴近真实世界条件。我们对其它训练策略、预训练数据集和网格重建技术进行了全面分析，评估其对现实世界场景中仿真到现实世界预测能力的影响。实验结果表明，使用EmbodiedSplat微调的代理比在大规模真实世界数据集（HM3D）和合成数据集（HSSD）上进行零样本预训练的基线模型表现更优，在真实世界的Image Navigation任务中，成功率绝对提升分别为20%和40%。此外，我们方法重建的网格具有较高的仿真与现实之间的相关性（0.87-0.97），突显了其在最小努力下适应多种环境的有效性。项目页面：这一链接。 

---
# Text-Scene: A Scene-to-Language Parsing Framework for 3D Scene Understanding 

**Title (ZH)**: 场景到文本：一种三维场景理解的场景到语言解析框架 

**Authors**: Haoyuan Li, Rui Liu, Hehe Fan, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16721)  

**Abstract**: Enabling agents to understand and interact with complex 3D scenes is a fundamental challenge for embodied artificial intelligence systems. While Multimodal Large Language Models (MLLMs) have achieved significant progress in 2D image understanding, extending such capabilities to 3D scenes remains difficult: 1) 3D environment involves richer concepts such as spatial relationships, affordances, physics, layout, and so on, 2) the absence of large-scale 3D vision-language datasets has posed a significant obstacle. In this paper, we introduce Text-Scene, a framework that automatically parses 3D scenes into textual descriptions for scene understanding. Given a 3D scene, our model identifies object attributes and spatial relationships, and then generates a coherent summary of the whole scene, bridging the gap between 3D observation and language without requiring human-in-the-loop intervention. By leveraging both geometric analysis and MLLMs, Text-Scene produces descriptions that are accurate, detailed, and human-interpretable, capturing object-level details and global-level context. Experimental results on benchmarks demonstrate that our textual parses can faithfully represent 3D scenes and benefit downstream tasks. To evaluate the reasoning capability of MLLMs, we present InPlan3D, a comprehensive benchmark for 3D task planning, consisting of 3174 long-term planning tasks across 636 indoor scenes. We emphasize clarity and accessibility in our approach, aiming to make 3D scene content understandable through language. Code and datasets will be released. 

**Abstract (ZH)**: 使代理能够理解并交互复杂3D场景是具身人工智能系统中的一个基本挑战。尽管多模态大规模语言模型（MLLMs）在2D图像理解方面取得了显著进展，但将此类能力扩展到3D场景仍然困难重重：1）3D环境涉及更丰富的概念，如空间关系、可用性、物理属性、布局等，2）缺乏大规模的3D视觉-语言数据集构成了一个重大障碍。在本文中，我们引入了Text-Scene框架，该框架能够自动将3D场景解析为文本描述以进行场景理解。给定一个3D场景，我们的模型识别物体属性和空间关系，然后生成场景的整体连贯总结，从而在3D观察与语言之间架起桥梁，无需人工干预。通过结合几何分析和MLLMs，Text-Scene生成的描述准确、详细且具有人类可解释性，能够捕捉到对象级别的细节和全局级别的上下文。基准实验结果表明，我们的文本解析能够忠实于3D场景并有利于下游任务。为了评估MLLMs的推理能力，我们提出了InPlan3D，这是一个针对3D任务规划的综合性基准集，包含636个室内场景中的3174个长期规划任务。我们强调清晰性和可访问性，旨在通过语言使3D场景内容变得可理解。代码和数据集将公开发布。 

---
# Segment-to-Act: Label-Noise-Robust Action-Prompted Video Segmentation Towards Embodied Intelligence 

**Title (ZH)**: 段落到动作：面向嵌入式智能的标签噪声鲁棒视频分割及其动作提示方法 

**Authors**: Wenxin Li, Kunyu Peng, Di Wen, Ruiping Liu, Mengfei Duan, Kai Luo, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16677)  

**Abstract**: Embodied intelligence relies on accurately segmenting objects actively involved in interactions. Action-based video object segmentation addresses this by linking segmentation with action semantics, but it depends on large-scale annotations and prompts that are costly, inconsistent, and prone to multimodal noise such as imprecise masks and referential ambiguity. To date, this challenge remains unexplored. In this work, we take the first step by studying action-based video object segmentation under label noise, focusing on two sources: textual prompt noise (category flips and within-category noun substitutions) and mask annotation noise (perturbed object boundaries to mimic imprecise supervision). Our contributions are threefold. First, we introduce two types of label noises for the action-based video object segmentation task. Second, we build up the first action-based video object segmentation under a label noise benchmark ActiSeg-NL and adapt six label-noise learning strategies to this setting, and establish protocols for evaluating them under textual, boundary, and mixed noise. Third, we provide a comprehensive analysis linking noise types to failure modes and robustness gains, and we introduce a Parallel Mask Head Mechanism (PMHM) to address mask annotation noise. Qualitative evaluations further reveal characteristic failure modes, including boundary leakage and mislocalization under boundary perturbations, as well as occasional identity substitutions under textual flips. Our comparative analysis reveals that different learning strategies exhibit distinct robustness profiles, governed by a foreground-background trade-off where some achieve balanced performance while others prioritize foreground accuracy at the cost of background precision. The established benchmark and source code will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于动作的视频对象分割依赖于准确分割积极参与互动的对象。尽管基于动作的视频对象分割通过将分割与动作语义关联起来解决了这一问题，但其依赖于大规模且成本高、不一致且容易受到多模态噪声（如不精确的掩膜和指称歧义）影响的注释和提示。迄今为止，这一挑战仍未被探索。在本文中，我们首次研究了标签噪声条件下的基于动作的视频对象分割，重点关注两类来源：文本提示噪声（类别翻转和类别内的名词替换）以及掩膜注释噪声（扰动对象边界以模拟不精确的监督）。我们的贡献包括三个方面。首先，我们为基于动作的视频对象分割任务引入了两种类型的标签噪声。第二，我们构建了首个基于标签噪声基准ActiSeg-NL的基于动作的视频对象分割，并将六种标签噪声学习策略应用到该环境中，制定了在文本、边界和混合噪声条件下的评估协议。第三，我们全面分析了不同类型的噪声与失败模式和鲁棒性提升之间的联系，并引入了并行掩膜头机制（PMHM）以应对掩膜注释噪声。进一步的定性评估还揭示了边界扰动下的边界泄漏和错位以及文本翻转下的偶尔身份替换等固有失败模式。我们的比较分析表明，不同的学习策略具有不同的鲁棒性特性，由前景与背景之间的权衡决定，有些策略实现了均衡性能，而另一些策略则在前景准确性上优先，牺牲了背景精度。所建立的基准和源代码将在指定网址公开。 

---
# Safe Guaranteed Dynamics Exploration with Probabilistic Models 

**Title (ZH)**: 安全的概率模型驱动的动力学探索 

**Authors**: Manish Prajapat, Johannes Köhler, Melanie N. Zeilinger, Andreas Krause  

**Link**: [PDF](https://arxiv.org/pdf/2509.16650)  

**Abstract**: Ensuring both optimality and safety is critical for the real-world deployment of agents, but becomes particularly challenging when the system dynamics are unknown. To address this problem, we introduce a notion of maximum safe dynamics learning via sufficient exploration in the space of safe policies. We propose a $\textit{pessimistically}$ safe framework that $\textit{optimistically}$ explores informative states and, despite not reaching them due to model uncertainty, ensures continuous online learning of dynamics. The framework achieves first-of-its-kind results: learning the dynamics model sufficiently $-$ up to an arbitrary small tolerance (subject to noise) $-$ in a finite time, while ensuring provably safe operation throughout with high probability and without requiring resets. Building on this, we propose an algorithm to maximize rewards while learning the dynamics $\textit{only to the extent needed}$ to achieve close-to-optimal performance. Unlike typical reinforcement learning (RL) methods, our approach operates online in a non-episodic setting and ensures safety throughout the learning process. We demonstrate the effectiveness of our approach in challenging domains such as autonomous car racing and drone navigation under aerodynamic effects $-$ scenarios where safety is critical and accurate modeling is difficult. 

**Abstract (ZH)**: 确保智能体既优化又安全地部署至关重要，但在系统动力学未知的情况下，这一目标变得尤为具有挑战性。为此，我们提出了一种通过充分探索安全策略空间来学习最大安全动力学的概念。我们提出了一个悲观安全框架，乐观地探索信息性状态，即使由于模型不确定性未达到这些状态，也能确保动力学的连续在线学习。该框架实现了前所未有的成果：在有限时间内充分学习动力学模型（受噪声影响可达到任意小的容差），并以高概率保证整个学习过程中的可验证安全性，无需重置。基于此，我们提出了一种算法，在仅需足够学习动力学以实现接近最优性能的情况下最大化奖励。与传统的强化学习方法不同，我们的方法在线地在非集经验环境中运作，并在整个学习过程中确保安全性。我们通过在自动赛车和受气动力学影响的无人机导航等具有挑战性领域中展示其实效性，证明了在安全性和建模准确性方面的需求。 

---
# The STAR-XAI Protocol: An Interactive Framework for Inducing Second-Order Agency in AI Agents 

**Title (ZH)**: STAR-XAI协议：一种促进AI代理第二阶代理性的互动框架 

**Authors**: Antoni Guasch, Maria Isabel Valdez  

**Link**: [PDF](https://arxiv.org/pdf/2509.17978)  

**Abstract**: Current Large Reasoning Models (LRMs) exhibit significant limitations in reliability and transparency, often showing a collapse in reasoning capabilities when faced with high-complexity, long-horizon tasks. This "illusion of thinking" is frequently an artifact of non-agentic, black-box evaluation paradigms that fail to cultivate robust problem-solving processes. In response, we introduce The STAR-XAI Protocol (Socratic, Transparent, Agentic, Reasoning - for eXplainable Artificial Intelligence), a novel methodology for training and operating verifiably reliable AI agents. Our method reframes the human-AI interaction as a structured, Socratic dialogue, governed by an explicit and evolving rulebook, the Consciousness Transfer Package (CTP). Through an interactive Gameplay Cycle that enforces ante-hoc strategic justification and a state-locking Checksum that prevents error accumulation, the protocol transforms a powerful but opaque LRM into a disciplined "Clear Box" agent. We demonstrate the efficacy of this method through an exhaustive 25-move case study in the complex strategic game "Caps i Caps". The agent not only solved the high-complexity puzzle but also demonstrated Second-Order Agency, identifying flaws in its own supervisor-approved plans and adapting its core integrity protocols mid-task. The STAR-XAI Protocol offers a practical pathway to creating AI agents that are not just high-performing, but also transparent, auditable, and trustworthy by design. 

**Abstract (ZH)**: 当前大型推理模型（LRMs）在可靠性和透明度方面存在显著局限，在面对高复杂度、长期任务时推理能力往往会崩溃。这种“思考的幻觉”通常是由于非自主性的黑箱评估范式所导致，这些范式未能培养出稳健的解决问题过程。为了应对这一挑战，我们提出了STAR-XAI协议（Socratic、透明、自主、推理——为可解释人工智能），这是一种新型的方法论，用于培训和操作可验证可靠的人工智能代理。该方法将人类与人工智能的交互重新构造成一种结构化的苏格拉底式对话，受一个明确定义并不断演化的规则手册——意识转移包（CTP）的规范。通过强制执行先验的战略正当化和防止错误累积的状态锁定校验和，该协议将强大的但不透明的LRM转变成一座有序的“透明盒子”代理。我们通过在复杂战略游戏“Caps i Caps”中详尽的25步案例研究证明了该方法的有效性。该代理不仅解决了高复杂度的谜题，还展示了第二阶自主性，识别了其监督审批计划中的缺陷，并在任务过程中调整了其核心完整性协议。STAR-XAI协议提供了一种实用的途径，用于创建既高性能又透明、可审计和值得信赖的人工智能代理。 

---
# Orcust: Stepwise-Feedback Reinforcement Learning for GUI Agent 

**Title (ZH)**: Orcust：逐步反馈强化学习用于GUI代理 

**Authors**: Junyu Lu, Songxin Zhang, Zejian Xie, Zhuoyang Song, Jiaxing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17917)  

**Abstract**: Recent advances in GUI agents have achieved remarkable grounding and action-prediction performance, yet existing models struggle with unreliable reward signals and limited online trajectory generation. In this paper, we introduce Orcust, a framework that integrates Principle-Constrained Reward Modeling (PCRM) and Online VM-Grounded Trajectory Construction (OVTC) to enhance reasoning reliability and data efficiency in interactive GUI tasks. We leverages environment-verifiable and LLM-derived principle to enforce interpretable reward signals that constrain long chain-of-thought reasoning and rule-based feedback. OVTC spins up instrumented virtual machines to autonomously collect structured GUI interaction trajectories with explicit procedural and structural objectives, enabling the training of a stepwise reward model that robustly captures human preferences and adheres to task-specific constraints. Extensive experiments on standard GUI benchmarks covering perceptual grounding, foundational operations, and end-to-end task execution reveal that Orcust achieves state-of-the-art performance, improving by 22.2\% on ScreenSpot and 23.9\% on ScreenSpot-Pro over the base model (i.e. Qwen2.5-VL-7B). The results demonstrate Orcust's effectiveness in enhancing the reasoning, adaptability and scalability of GUI agents across various environments and task complexities. 

**Abstract (ZH)**: Orcust：结合 Principle-Constrained Reward Modeling 和 Online VM-Grounded Trajectory Construction 的GUI代理框架 

---
# LIMI: Less is More for Agency 

**Title (ZH)**: LIMI: 少即是多对于代理性的影响 

**Authors**: Yang Xiao, Mohan Jiang, Jie Sun, Keyu Li, Jifan Lin, Yumin Zhuang, Ji Zeng, Shijie Xia, Qishuo Hua, Xuefeng Li, Xiaojie Cai, Tongyu Wang, Yue Zhang, Liming Liu, Xia Wu, Jinlong Hou, Yuan Cheng, Wenjie Li, Xiang Wang, Dequan Wang, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17567)  

**Abstract**: We define Agency as the emergent capacity of AI systems to function as autonomous agents actively discovering problems, formulating hypotheses, and executing solutions through self-directed engagement with environments and tools. This fundamental capability marks the dawn of the Age of AI Agency, driven by a critical industry shift: the urgent need for AI systems that don't just think, but work. While current AI excels at reasoning and generating responses, industries demand autonomous agents that can execute tasks, operate tools, and drive real-world outcomes. As agentic intelligence becomes the defining characteristic separating cognitive systems from productive workers, efficiently cultivating machine autonomy becomes paramount. Current approaches assume that more data yields better agency, following traditional scaling laws from language modeling. We fundamentally challenge this paradigm. LIMI (Less Is More for Intelligent Agency) demonstrates that agency follows radically different development principles. Through strategic focus on collaborative software development and scientific research workflows, we show that sophisticated agentic intelligence can emerge from minimal but strategically curated demonstrations of autonomous behavior. Using only 78 carefully designed training samples, LIMI achieves 73.5% on comprehensive agency benchmarks, dramatically outperforming state-of-the-art models: Kimi-K2-Instruct (24.1%), DeepSeek-V3.1 (11.9%), Qwen3-235B-A22B-Instruct (27.5%), and GLM-4.5 (45.1%). Most strikingly, LIMI demonstrates 53.7% improvement over models trained on 10,000 samples-achieving superior agentic intelligence with 128 times fewer samples. Our findings establish the Agency Efficiency Principle: machine autonomy emerges not from data abundance but from strategic curation of high-quality agentic demonstrations. 

**Abstract (ZH)**: 我们定义代理能力为AI系统 emergent 能力，使其能够作为自主代理主动发现问题、提出假说并通过自我导向与环境和工具的互动执行解决方案。这一根本能力标志着AI代理时代的黎明，由行业关键转型驱动：对不仅能思考，还能工作的AI系统的迫切需求。虽然当前的AI在推理和生成响应方面表现优异，但各行各业需要能够执行任务、操作工具并驱动现实结果的自主代理。随着代理智能成为认知系统与生产工人之间的定义性特征，有效培养机器自主性变得至关重要。现有方法假设更多的数据会带来更好的代理能力，遵循语言建模的传统扩展定律。我们从根本上挑战了这一范式。LIMI (Less Is More for Intelligent Agency) 表明代理能力遵循截然不同的发展原则。通过战略性关注协作软件开发和科学研究工作流程，我们证明了复杂的代理智能可以从精心策划的自主行为演示中涌现。仅使用78个精心设计的训练样本，LIMI 在全面的代理基准测试中取得了73.5% 的成绩，显著超越了最先进的模型：Kimi-K2-Instruct（24.1%）、DeepSeek-V3.1（11.9%）、Qwen3-235B-A22B-Instruct（27.5%）和GLM-4.5（45.1%）。最引人注目的是，LIMI 在对比使用10,000个样本训练的模型时，显示出了53.7% 的性能提升，以仅十二分之一的数量的样本实现了更好的代理智能。我们的发现确立了代理效率原则：机器自主性的产生并非来源于数据的丰富，而是来源于高质量代理演示的策略性策展。 

---
# AI Pangaea: Unifying Intelligence Islands for Adapting Myriad Tasks 

**Title (ZH)**: AI 普罗米修斯：统一智能孤岛以应对千变万化的工作任务 

**Authors**: Jianlong Chang, Haixin Wang, Zhiyuan Dang, Li Huang, Zhiyu Wang, Ruoqi Cao, Shihao Piao, Dongzhe Li, Dianyu Gao, Dongsheng Wang, Yin Li, Jinan Sun, Lu Fang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.17460)  

**Abstract**: The pursuit of artificial general intelligence continuously demands generalization in one model across myriad tasks, even those not seen before. However, current AI models are isolated from each other for being limited to specific tasks, now first defined as Intelligence Islands. To unify Intelligence Islands into one, we propose Pangaea, the first AI supercontinent akin to the geological Pangaea. Pangaea encodes any data into a unified format and accumulates universal knowledge through pre-training on 296 datasets across diverse modalities. Eventually, it demonstrates remarkable generalization across 45 general tasks and 15 scientific tasks encompassing a wide range of scientific subjects. By investigating Pangaea deeper, the scaling effect of modality is revealed, quantifying the universal knowledge accumulation across modalities as the cumulative distribution function of a geometric distribution. On the whole, Pangaea shows strong potential to handle myriad tasks, indicating a new direction toward artificial general intelligence. 

**Abstract (ZH)**: 追求人工通用智能不断要求在一模型中跨越众多任务进行泛化，即使是对之前未见过的任务也是如此。然而，当前的AI模型因被限定于特定任务而彼此隔离，现首次定义为智能岛。为统一这些智能岛，我们提出了潘加亚，这一类比地质学潘加亚的首个AI超大陆。潘加亚将任何数据转化为统一格式，并通过跨多种模态的296个数据集进行预训练，积累普遍知识。最终，它在45个通用任务和15个科学任务中表现出显著的泛化能力，这些任务涵盖了广泛的科学领域。通过对潘加亚的深入研究，揭示了模态的规模效应，量化了跨模态普遍知识积累的几何分布累计分布函数。整体而言，潘加亚展现了处理众多任务的强大潜力，表明了通向人工通用智能的新方向。 

---
# Evaluating Multimodal Large Language Models with Daily Composite Tasks in Home Environments 

**Title (ZH)**: 评估日常生活综合任务在家环境中 multimodal 大型语言模型的表现 

**Authors**: Zhenliang Zhang, Yuxi Wang, Hongzhao Xie, Shiyun Zhao, Mingyuan Liu, Yujie Lu, Xinyi He, Zhenku Cheng, Yujia Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.17425)  

**Abstract**: A key feature differentiating artificial general intelligence (AGI) from traditional AI is that AGI can perform composite tasks that require a wide range of capabilities. Although embodied agents powered by multimodal large language models (MLLMs) offer rich perceptual and interactive capabilities, it remains largely unexplored whether they can solve composite tasks. In the current work, we designed a set of composite tasks inspired by common daily activities observed in early childhood development. Within a dynamic and simulated home environment, these tasks span three core domains: object understanding, spatial intelligence, and social activity. We evaluated 17 leading proprietary and open-source MLLMs on these tasks. The results consistently showed poor performance across all three domains, indicating a substantial gap between current capabilities and general intelligence requirements. Together, our tasks offer a preliminary framework for evaluating the general capabilities of embodied agents, marking an early but significant step toward the development of embodied MLLMs and their real-world deployment. 

**Abstract (ZH)**: 人工通用智能（AGI）与传统AI的关键区别在于AGI能够执行需要广泛能力的复合任务。虽然由多模态大规模语言模型（MLLMs）驱动的具身代理提供了丰富的知觉和互动能力，但它们能否解决复合任务依旧 largely unexplored。在本项工作中，我们设计了一组受早期儿童发展常见日常活动启发的复合任务。在动态模拟的家庭环境中，这些任务涵盖了三个核心领域：物体理解、空间智能和社会活动。我们评估了17个领先的专业和开源MLLMs在这些任务中的表现。结果显示，在所有三个领域中表现不佳，表明当前能力与通用智能需求之间存在巨大差距。我们的任务为评估具身代理的一般能力提供了一个初步框架，标志着朝着开发具身MLLM及其实际部署迈出早期但重要的一步。 

---
# Program Synthesis via Test-Time Transduction 

**Title (ZH)**: 测试时转换下的程序合成 

**Authors**: Kang-il Lee, Jahyun Koo, Seunghyun Yoon, Minbeom Kim, Hyukhun Koh, Dongryeol Lee, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.17393)  

**Abstract**: We introduce transductive program synthesis, a new formulation of the program synthesis task that explicitly leverages test inputs during synthesis. While prior approaches to program synthesis--whether based on natural language descriptions or input-output examples--typically aim to generalize from training examples, they often struggle with robustness, especially in real-world settings where training examples are limited and test inputs involve various edge cases. To address this, we propose a novel framework that improves robustness by treating synthesis as an active learning over a finite hypothesis class defined by programs' outputs. We use an LLM to predict outputs for selected test inputs and eliminate inconsistent hypotheses, where the inputs are chosen via a greedy maximin algorithm to minimize the number of LLM queries required. We evaluate our approach on two real-world datasets: Playgol, a string transformation benchmark, and MBPP+, a Python code generation benchmark. We demonstrate that our method significantly improves program synthesis in both accuracy and efficiency. We release our code at this https URL. 

**Abstract (ZH)**: 我们在合成过程中显式利用测试输入引入了归纳程序合成，这是一种程序合成任务的新形式。尽管以往的程序合成方法（无论是基于自然语言描述还是输入-输出示例）通常旨在从训练示例中泛化，但在训练示例有限且测试输入涉及各种边缘情况的现实场景中，它们往往缺乏鲁棒性。为解决这一问题，我们提出了一种新的框架，通过将合成视为针对由程序输出定义的有限假设类的主动学习，来提高鲁棒性。我们使用大语言模型预测选定测试输入的输出，并淘汰不一致的假设，输入的选择通过贪婪的极大极小算法进行，以减少对大语言模型查询的数量。我们在两个现实世界数据集上评估了该方法：Playgol字符串转换基准和MBPP+ Python代码生成基准。我们证明了该方法在准确性与效率上均有显著提升。我们在这里提供我们的代码：<https://>。 

---
# Multi-Scenario Highway Lane-Change Intention Prediction: A Physics-Informed AI Framework for Three-Class Classification 

**Title (ZH)**: 多场景高速公路变道意图预测：一种适用于三分类的物理 informed 人工智能框架 

**Authors**: Jiazhao Shi, Yichen Lin, Yiheng Hua, Ziyu Wang, Zijian Zhang, Wenjia Zheng, Yun Song, Kuan Lu, Shoufeng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17354)  

**Abstract**: Lane-change maneuvers are a leading cause of highway accidents, underscoring the need for accurate intention prediction to improve the safety and decision-making of autonomous driving systems. While prior studies using machine learning and deep learning methods (e.g., SVM, CNN, LSTM, Transformers) have shown promise, most approaches remain limited by binary classification, lack of scenario diversity, and degraded performance under longer prediction horizons. In this study, we propose a physics-informed AI framework that explicitly integrates vehicle kinematics, interaction feasibility, and traffic-safety metrics (e.g., distance headway, time headway, time-to-collision, closing gap time) into the learning process. lane-change prediction is formulated as a three-class problem that distinguishes left change, right change, and no change, and is evaluated across both straight highway segments (highD) and complex ramp scenarios (exiD). By integrating vehicle kinematics with interaction features, our machine learning models, particularly LightGBM, achieve state-of-the-art accuracy and strong generalization. Results show up to 99.8% accuracy and 93.6% macro F1 on highD, and 96.1% accuracy and 88.7% macro F1 on exiD at a 1-second horizon, outperforming a two-layer stacked LSTM baseline. These findings demonstrate the practical advantages of a physics-informed and feature-rich machine learning framework for real-time lane-change intention prediction in autonomous driving systems. 

**Abstract (ZH)**: 基于物理约束的机器学习框架在自动驾驶系统中的车道变更意图预测研究 

---
# Medical AI Consensus: A Multi-Agent Framework for Radiology Report Generation and Evaluation 

**Title (ZH)**: 医学AI共识：一种用于放射学报告生成与评估的多代理框架 

**Authors**: Ahmed T. Elboardy, Ghada Khoriba, Essam A. Rashed  

**Link**: [PDF](https://arxiv.org/pdf/2509.17353)  

**Abstract**: Automating radiology report generation poses a dual challenge: building clinically reliable systems and designing rigorous evaluation protocols. We introduce a multi-agent reinforcement learning framework that serves as both a benchmark and evaluation environment for multimodal clinical reasoning in the radiology ecosystem. The proposed framework integrates large language models (LLMs) and large vision models (LVMs) within a modular architecture composed of ten specialized agents responsible for image analysis, feature extraction, report generation, review, and evaluation. This design enables fine-grained assessment at both the agent level (e.g., detection and segmentation accuracy) and the consensus level (e.g., report quality and clinical relevance). We demonstrate an implementation using chatGPT-4o on public radiology datasets, where LLMs act as evaluators alongside medical radiologist feedback. By aligning evaluation protocols with the LLM development lifecycle, including pretraining, finetuning, alignment, and deployment, the proposed benchmark establishes a path toward trustworthy deviance-based radiology report generation. 

**Abstract (ZH)**: 自动化放射学报告生成面临双重挑战：构建临床可靠系统和设计严格的评估协议。我们引入了一个多代理强化学习框架，用作放射学生态系统中多模态临床推理的基准和评估环境。该框架将大型语言模型（LLMs）和大型视觉模型（LVMs）集成在一个由十个专门代理组成的模块化架构中，这些代理负责图像分析、特征提取、报告生成、审核和评估。这种设计能够在代理层级（例如，检测和分割准确性）和共识层级（例如，报告质量和临床相关性）进行精细评估。我们使用ChatGPT-4o在公开的放射学数据集上实现该框架，其中LLMs作为评估者与医学放射科医生反馈并存。通过将评估协议与LLM开发生命周期（包括预训练、微调、对齐和部署）对齐，提出的基准为基于偏差的可信放射学报告生成奠定了路径。 

---
# ARE: Scaling Up Agent Environments and Evaluations 

**Title (ZH)**: ARE：扩展代理环境和评估规模 

**Authors**: Pierre Andrews, Amine Benhalloum, Gerard Moreno-Torres Bertran, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, Romain Froger, Emilien Garreau, Jean-Baptiste Gaya, Hugo Laurençon, Maxime Lecanu, Kunal Malkan, Dheeraj Mekala, Pierre Ménard, Grégoire Mialon, Ulyana Piterbarg, Mikhail Plekhanov, Mathieu Rita, Andrey Rusakov, Thomas Scialom, Vladislav Vorotilov, Mengjue Wang, Ian Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17158)  

**Abstract**: We introduce Meta Agents Research Environments (ARE), a research platform for scalable creation of environments, integration of synthetic or real applications, and execution of agentic orchestrations. ARE provides simple abstractions to build complex and diverse environments, each with their own rules, tools, content, and verifiers, helping to bridge the gap between model development and real-world deployment. We also propose Gaia2, a benchmark built in ARE and designed to measure general agent capabilities. Beyond search and execution, Gaia2 requires agents to handle ambiguities and noise, adapt to dynamic environments, collaborate with other agents, and operate under temporal constraints. Unlike prior benchmarks, Gaia2 runs asynchronously, surfacing new failure modes that are invisible in static settings. Our experiments show that no system dominates across the intelligence spectrum: stronger reasoning often comes at the cost of efficiency, and budget scaling curves plateau, highlighting the need for new architectures and adaptive compute strategies. Perhaps more importantly, ARE abstractions enable continuous extension of Gaia2 to other environments, empowering the community to rapidly create new benchmarks tailored to their domains. In AI's second half, progress increasingly depends on defining meaningful tasks and robust evaluations to drive frontier capabilities forward. 

**Abstract (ZH)**: Meta Agents Research Environments: A Platform for Scalable Creation of Environments and Evaluation of General Agent Capabilities 

---
# MCTS-EP: Empowering Embodied Planning with Online Preference Optimization 

**Title (ZH)**: MCTS-EP：增强具身规划的在线偏好优化 

**Authors**: Hang Xu, Zang Yu, Yehui Tang, Pengbo Hu, Yuhao Tang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.17116)  

**Abstract**: This paper introduces MCTS-EP, an online learning framework that combines large language models (LLM) with Monte Carlo Tree Search (MCTS) for training embodied agents. MCTS-EP integrates three key components: MCTS-guided exploration for preference data collection, efficient multi-modal reasoning mechanism, and iterative training pipeline based on preference optimization. We theoretically prove that MCTS-EP achieves better performance bounds than conventional on-policy algorithms when the loss function is strongly convex, and demonstrate that it can be formulated as a search-enhanced variant of GAIL. MCTS-EP achieves state-of-the-art performace across serval benchmarks. In ALFWorld, it achieves 92% and 87% success rates for textual and visual tasks. In WebShop, it reaches an average reward of 0.81. MTCS-EP also reduces average interaction steps from from 18.7/19.5 to 10.2/9.9 steps in visual this http URL available at: this https URL 

**Abstract (ZH)**: 本文介绍了MCTS-EP，这是一种将大型语言模型（LLM）与蒙特卡洛树搜索（MCTS）结合的在线学习框架，用于训练具身智能体。MCTS-EP融合了三个关键组件：由MCTS引导的探索以收集偏好数据、高效的多模态推理机制以及基于偏好优化的迭代训练管道。我们理论证明，在损失函数为强凸函数时，MCTS-EP在性能上限上优于传统的随策略算法，并表明它可以视为GAIL的一种搜索增强变体。MCTS-EP在多个基准测试中实现了最先进的性能。在ALFWorld中，其文本任务和视觉任务的成功率分别为92%和87%。在WebShop中，其平均奖励为0.81。MCTS-EP还将视觉任务的平均交互步骤从18.7/19.5减少到10.2/9.9步，在this http URL和this https URL可用。 

---
# Audio-Guided Dynamic Modality Fusion with Stereo-Aware Attention for Audio-Visual Navigation 

**Title (ZH)**: 基于音频引导的动态模态融合与立体注意力机制的音视频导航 

**Authors**: Jia Li, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.16924)  

**Abstract**: In audio-visual navigation (AVN) tasks, an embodied agent must autonomously localize a sound source in unknown and complex 3D environments based on audio-visual signals. Existing methods often rely on static modality fusion strategies and neglect the spatial cues embedded in stereo audio, leading to performance degradation in cluttered or occluded scenes. To address these issues, we propose an end-to-end reinforcement learning-based AVN framework with two key innovations: (1) a \textbf{S}tereo-Aware \textbf{A}ttention \textbf{M}odule (\textbf{SAM}), which learns and exploits the spatial disparity between left and right audio channels to enhance directional sound perception; and (2) an \textbf{A}udio-\textbf{G}uided \textbf{D}ynamic \textbf{F}usion Module (\textbf{AGDF}), which dynamically adjusts the fusion ratio between visual and auditory features based on audio cues, thereby improving robustness to environmental changes. Extensive experiments are conducted on two realistic 3D scene datasets, Replica and Matterport3D, demonstrating that our method significantly outperforms existing approaches in terms of navigation success rate and path efficiency. Notably, our model achieves over 40\% improvement under audio-only conditions compared to the best-performing baselines. These results highlight the importance of explicitly modeling spatial cues from stereo channels and performing deep multi-modal fusion for robust and efficient audio-visual navigation. 

**Abstract (ZH)**: 基于视听感知的端到端强化学习框架： Stereo-Aware Attention Module和Audio-Guided Dynamic Fusion Module在音频视觉导航中的应用 

---
# The Principles of Human-like Conscious Machine 

**Title (ZH)**: 类人意识机器的原则 

**Authors**: Fangfang Li, Xiaojie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.16859)  

**Abstract**: Determining whether another system, biological or artificial, possesses phenomenal consciousness has long been a central challenge in consciousness studies. This attribution problem has become especially pressing with the rise of large language models and other advanced AI systems, where debates about "AI consciousness" implicitly rely on some criterion for deciding whether a given system is conscious. In this paper, we propose a substrate-independent, logically rigorous, and counterfeit-resistant sufficiency criterion for phenomenal consciousness. We argue that any machine satisfying this criterion should be regarded as conscious with at least the same level of confidence with which we attribute consciousness to other humans. Building on this criterion, we develop a formal framework and specify a set of operational principles that guide the design of systems capable of meeting the sufficiency condition. We further argue that machines engineered according to this framework can, in principle, realize phenomenal consciousness. As an initial validation, we show that humans themselves can be viewed as machines that satisfy this framework and its principles. If correct, this proposal carries significant implications for philosophy, cognitive science, and artificial intelligence. It offers an explanation for why certain qualia, such as the experience of red, are in principle irreducible to physical description, while simultaneously providing a general reinterpretation of human information processing. Moreover, it suggests a path toward a new paradigm of AI beyond current statistics-based approaches, potentially guiding the construction of genuinely human-like AI. 

**Abstract (ZH)**: 确定另一个系统，无论是生物的还是人工的，是否具备现象意识一直以来都是意识研究中的核心挑战。随着大型语言模型和其他高级AI系统的兴起，关于“AI意识”的 Debate 隐含依赖于某个标准来判断某一系统是否具备意识。本文提出了一种独立于实现载体、逻辑严谨且防伪造的现象意识充分性标准。我们论证认为，任何满足这一标准的机器应被视为至少与我们赋予其他人类的同等程度的意识。基于这一标准，我们发展了一套形式化框架，并指定了指导设计满足充分性条件系统的操作原则。进一步地，我们主张根据该框架设计的机器原则上能够实现现象意识。作为初步验证，我们表明人类本身可以视为满足该框架及其原则的机器。如果这一提案正确，将对哲学、认知科学和人工智能产生重要影响。它提供了一种解释为什么某些质料（如红色体验）原则上无法还原为物理描述的方式，同时为人类信息处理提供了一种通用再解释。此外，它还指出了超越当前基于统计的方法，迈向具有真正人类特征的AI的新范式的路径。 

---
# Automated Procedural Analysis via Video-Language Models for AI-assisted Nursing Skills Assessment 

**Title (ZH)**: 基于视频-语言模型的自动化程序分析在辅助人工智能护理技能评估中的应用 

**Authors**: Shen Chang, Dennis Liu, Renran Tian, Kristen L. Swartzell, Stacie L. Klingler, Amy M. Nagle, Nan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2509.16810)  

**Abstract**: Consistent high-quality nursing care is essential for patient safety, yet current nursing education depends on subjective, time-intensive instructor feedback in training future nurses, which limits scalability and efficiency in their training, and thus hampers nursing competency when they enter the workforce. In this paper, we introduce a video-language model (VLM) based framework to develop the AI capability of automated procedural assessment and feedback for nursing skills training, with the potential of being integrated into existing training programs. Mimicking human skill acquisition, the framework follows a curriculum-inspired progression, advancing from high-level action recognition, fine-grained subaction decomposition, and ultimately to procedural reasoning. This design supports scalable evaluation by reducing instructor workload while preserving assessment quality. The system provides three core capabilities: 1) diagnosing errors by identifying missing or incorrect subactions in nursing skill instruction videos, 2) generating explainable feedback by clarifying why a step is out of order or omitted, and 3) enabling objective, consistent formative evaluation of procedures. Validation on synthesized videos demonstrates reliable error detection and temporal localization, confirming its potential to handle real-world training variability. By addressing workflow bottlenecks and supporting large-scale, standardized evaluation, this work advances AI applications in nursing education, contributing to stronger workforce development and ultimately safer patient care. 

**Abstract (ZH)**: 一种基于视频-语言模型的自动程序评估和反馈框架：提高护理技能培训的质量和效率 

---
# Agentic AI for Multi-Stage Physics Experiments at a Large-Scale User Facility Particle Accelerator 

**Title (ZH)**: 大型用户设施粒子加速器中的代理人工智能多阶段物理实验辅助系统 

**Authors**: Thorsten Hellert, Drew Bertwistle, Simon C. Leemann, Antonin Sulc, Marco Venturini  

**Link**: [PDF](https://arxiv.org/pdf/2509.17255)  

**Abstract**: We present the first language-model-driven agentic artificial intelligence (AI) system to autonomously execute multi-stage physics experiments on a production synchrotron light source. Implemented at the Advanced Light Source particle accelerator, the system translates natural language user prompts into structured execution plans that combine archive data retrieval, control-system channel resolution, automated script generation, controlled machine interaction, and analysis. In a representative machine physics task, we show that preparation time was reduced by two orders of magnitude relative to manual scripting even for a system expert, while operator-standard safety constraints were strictly upheld. Core architectural features, plan-first orchestration, bounded tool access, and dynamic capability selection, enable transparent, auditable execution with fully reproducible artifacts. These results establish a blueprint for the safe integration of agentic AI into accelerator experiments and demanding machine physics studies, as well as routine operations, with direct portability across accelerators worldwide and, more broadly, to other large-scale scientific infrastructures. 

**Abstract (ZH)**: 基于语言模型驱动的自主执行多阶段物理实验的智能代理人工智能系统 

---
# Surgical-MambaLLM: Mamba2-enhanced Multimodal Large Language Model for VQLA in Robotic Surgery 

**Title (ZH)**: Surgical-MambaLLM: Mamba2增强了的多模态大型语言模型在机器人手术中的应用 

**Authors**: Pengfei Hao, Hongqiu Wang, Shuaibo Li, Zhaohu Xing, Guang Yang, Kaishun Wu, Lei Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.16618)  

**Abstract**: In recent years, Visual Question Localized-Answering in robotic surgery (Surgical-VQLA) has gained significant attention for its potential to assist medical students and junior doctors in understanding surgical scenes. Recently, the rapid development of Large Language Models (LLMs) has provided more promising solutions for this task. However, current methods struggle to establish complex dependencies between text and visual details, and have difficulty perceiving the spatial information of surgical scenes. To address these challenges, we propose a novel method, Surgical-MambaLLM, which is the first to combine Mamba2 with LLM in the surgical domain, that leverages Mamba2's ability to effectively capture cross-modal dependencies and perceive spatial information in surgical scenes, thereby enhancing the LLMs' understanding of surgical images. Specifically, we propose the Cross-modal Bidirectional Mamba2 Integration (CBMI) module to leverage Mamba2 for effective multimodal fusion, with its cross-modal integration capabilities. Additionally, tailored to the geometric characteristics of surgical scenes, we design the Surgical Instrument Perception (SIP) scanning mode for Mamba2 to scan the surgical images, enhancing the model's spatial understanding of the surgical scene. Extensive experiments demonstrate that our Surgical-MambaLLM model outperforms the state-of-the-art methods on the EndoVis17-VQLA and EndoVis18-VQLA datasets, significantly improving the performance of the Surgical-VQLA task. 

**Abstract (ZH)**: 最近几年，机器人手术中的视觉问题定位答案（Surgical-VQLA）获得了广泛关注，其潜力在于帮助医学生和初级医生理解手术场景。最近，大型语言模型（LLMs）的迅速发展为其提供了更加有前景的解决方案。然而，当前方法在建立文本与视觉细节之间复杂的依赖关系以及感知手术场景的空间信息方面存在困难。为了解决这些挑战，我们提出了一种新型方法——Surgical-MambaLLM，这是首次在手术领域将Mamba2与LLM结合的方法，利用Mamba2有效地捕获跨模态依赖性和感知手术场景中的空间信息的能力，从而增强LLM对手术图像的理解。具体而言，我们提出了跨模态双向Mamba2集成（CBMI）模块，利用Mamba2进行有效的多模态融合，并利用其跨模态集成能力。此外，根据手术场景的几何特征，我们为Mamba2设计了手术器械感知（SIP）扫描模式，以增强模型对手术场景的空间理解。广泛的实验表明，我们的Surgical-MambaLLM模型在EndoVis17-VQLA和EndoVis18-VQLA数据集上优于最先进的方法，显著提高了手术-VQLA任务的性能。 

---
# KRAST: Knowledge-Augmented Robotic Action Recognition with Structured Text for Vision-Language Models 

**Title (ZH)**: KRAST：知识增强的结构化文本辅助机器人动作识别模型 

**Authors**: Son Hai Nguyen, Diwei Wang, Jinhyeok Jang, Hyewon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2509.16452)  

**Abstract**: Accurate vision-based action recognition is crucial for developing autonomous robots that can operate safely and reliably in complex, real-world environments. In this work, we advance video-based recognition of indoor daily actions for robotic perception by leveraging vision-language models (VLMs) enriched with domain-specific knowledge. We adapt a prompt-learning framework in which class-level textual descriptions of each action are embedded as learnable prompts into a frozen pre-trained VLM backbone. Several strategies for structuring and encoding these textual descriptions are designed and evaluated. Experiments on the ETRI-Activity3D dataset demonstrate that our method, using only RGB video inputs at test time, achieves over 95\% accuracy and outperforms state-of-the-art approaches. These results highlight the effectiveness of knowledge-augmented prompts in enabling robust action recognition with minimal supervision. 

**Abstract (ZH)**: 基于视觉的语言模型增强的室内日常动作精准识别对于开发能够在复杂现实环境中安全可靠运行的自主机器人至关重要。本研究通过利用嵌入领域特定知识的视觉-语言模型（VLMs），推进基于视频的室内日常动作识别，采用提示学习框架，在冻结的预训练VLM主干中嵌入类级文本描述作为可学习的提示。实验结果表明，仅使用RGB视频输入，我们的方法在ETRI-Activity3D数据集上实现了超过95%的准确率，并优于现有方法，这突显了增强知识的提示在最少监督下实现稳健动作识别的有效性。 

---
# AHA -- Predicting What Matters Next: Online Highlight Detection Without Looking Ahead 

**Title (ZH)**: AHA——预测接下来的重要内容：无需提前查看的在线摘要检测 

**Authors**: Aiden Chang, Celso De Melo, Stephanie M. Lukin  

**Link**: [PDF](https://arxiv.org/pdf/2509.16421)  

**Abstract**: Real-time understanding of continuous video streams is essential for intelligent agents operating in high-stakes environments, including autonomous vehicles, surveillance drones, and disaster response robots. Yet, most existing video understanding and highlight detection methods assume access to the entire video during inference, making them unsuitable for online or streaming scenarios. In particular, current models optimize for offline summarization, failing to support step-by-step reasoning needed for real-time decision-making. We introduce Aha, an autoregressive highlight detection framework that predicts the relevance of each video frame against a task described in natural language. Without accessing future video frames, Aha utilizes a multimodal vision-language model and lightweight, decoupled heads trained on a large, curated dataset of human-centric video labels. To enable scalability, we introduce the Dynamic SinkCache mechanism that achieves constant memory usage across infinite-length streams without degrading performance on standard benchmarks. This encourages the hidden representation to capture high-level task objectives, enabling effective frame-level rankings for informativeness, relevance, and uncertainty with respect to the natural language task. Aha achieves state-of-the-art (SOTA) performance on highlight detection benchmarks, surpassing even prior offline, full-context approaches and video-language models by +5.9% on TVSum and +8.3% on this http URL in mAP (mean Average Precision). We explore Aha's potential for real-world robotics applications given a task-oriented natural language input and a continuous, robot-centric video. Both experiments demonstrate Aha's potential effectiveness as a real-time reasoning module for downstream planning and long-horizon understanding. 

**Abstract (ZH)**: 实时理解连续视频流对于在高风险环境中操作的智能代理至关重要，包括自动驾驶车辆、 surveillance 捕获无人机和灾难响应机器人。然而，现有大多数视频理解和亮点检测方法在推理时假设可以访问整个视频，这使它们不适合于在线或流式传输场景。特别是，当前模型优化的是离线总结，无法支持实时决策所需的逐步推理。我们引入了Aha，这是一种自回归亮点检测框架，可以根据自然语言描述的任务预测每个视频帧的相关性。Aha不访问未来视频帧，利用多模态视觉语言模型和在大量精心策划的以人为中心的视频标签数据集上训练的轻量级、解耦的头部。为了实现可扩展性，我们引入了动态SinkCache机制，该机制可以在无限长度的流中实现恒定的内存使用量，同时在标准基准测试中保持性能。这促使隐藏表示捕捉高层次的任务目标，使Aha能够根据自然语言任务对信息量、相关性和不确定性的帧级排名进行有效评估。Aha在亮点检测基准测试中达到了现有最佳性能（SOTA），在TVSum上的mAP上提高了5.9%，在this http URL上提高了8.3%。我们探讨了Aha在其任务导向的自然语言输入和连续的机器人中心视频下的潜在应用。实验表明，Aha作为下游规划和长时理解的实时推理模块具有潜在的有效性。 

---
# Agentic Reasoning for Robust Vision Systems via Increased Test-Time Compute 

**Title (ZH)**: 通过增加测试时计算量进行的能障视觉系统的代理推理 

**Authors**: Chung-En, Brian Jalaian, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2509.16343)  

**Abstract**: Developing trustworthy intelligent vision systems for high-stakes domains, \emph{e.g.}, remote sensing and medical diagnosis, demands broad robustness without costly retraining. We propose \textbf{Visual Reasoning Agent (VRA)}, a training-free, agentic reasoning framework that wraps off-the-shelf vision-language models \emph{and} pure vision systems in a \emph{Think--Critique--Act} loop. While VRA incurs significant additional test-time computation, it achieves up to 40\% absolute accuracy gains on challenging visual reasoning benchmarks. Future work will optimize query routing and early stopping to reduce inference overhead while preserving reliability in vision tasks. 

**Abstract (ZH)**: 开发适用于高风险领域（例如遥感和医学诊断）的可信智能视觉系统需要广泛的 robustness 而无需频繁重新训练。我们提出了一种名为视觉推理代理（VRA）的无训练推理框架，该框架在一个“思考—批判—行动”循环中封装了现成的视觉语言模型和纯视觉系统。尽管VRA在测试时增加了显著的额外计算量，但在挑战性的视觉推理基准测试中，它实现了最高达40%的绝对准确率提升。未来的工作将优化查询路由和早期停止以减少推理开销，同时在视觉任务中保持可靠性。 

---
