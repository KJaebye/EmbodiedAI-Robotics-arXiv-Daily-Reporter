# Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots 

**Title (ZH)**: 模拟中的操纵： enabling robots to实现准确的几何 perception认知 

**Authors**: Minghuan Liu, Zhengbang Zhu, Xiaoshen Han, Peng Hu, Haotong Lin, Xinyao Li, Jingxiao Chen, Jiafeng Xu, Yichu Yang, Yunfeng Lin, Xinghang Li, Yong Yu, Weinan Zhang, Tao Kong, Bingyi Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02530)  

**Abstract**: Modern robotic manipulation primarily relies on visual observations in a 2D color space for skill learning but suffers from poor generalization. In contrast, humans, living in a 3D world, depend more on physical properties-such as distance, size, and shape-than on texture when interacting with objects. Since such 3D geometric information can be acquired from widely available depth cameras, it appears feasible to endow robots with similar perceptual capabilities. Our pilot study found that using depth cameras for manipulation is challenging, primarily due to their limited accuracy and susceptibility to various types of noise. In this work, we propose Camera Depth Models (CDMs) as a simple plugin on daily-use depth cameras, which take RGB images and raw depth signals as input and output denoised, accurate metric depth. To achieve this, we develop a neural data engine that generates high-quality paired data from simulation by modeling a depth camera's noise pattern. Our results show that CDMs achieve nearly simulation-level accuracy in depth prediction, effectively bridging the sim-to-real gap for manipulation tasks. Notably, our experiments demonstrate, for the first time, that a policy trained on raw simulated depth, without the need for adding noise or real-world fine-tuning, generalizes seamlessly to real-world robots on two challenging long-horizon tasks involving articulated, reflective, and slender objects, with little to no performance degradation. We hope our findings will inspire future research in utilizing simulation data and 3D information in general robot policies. 

**Abstract (ZH)**: 基于深度相机的Camera Depth Models提升机器人操作任务中三维几何信息利用与泛化能力 

---
# OpenGuide: Assistive Object Retrieval in Indoor Spaces for Individuals with Visual Impairments 

**Title (ZH)**: OpenGuide: 为视障个体在室内空间中提供辅助对象检索 

**Authors**: Yifan Xu, Qianwei Wang, Vineet Kamat, Carol Menassa  

**Link**: [PDF](https://arxiv.org/pdf/2509.02425)  

**Abstract**: Indoor built environments like homes and offices often present complex and cluttered layouts that pose significant challenges for individuals who are blind or visually impaired, especially when performing tasks that involve locating and gathering multiple objects. While many existing assistive technologies focus on basic navigation or obstacle avoidance, few systems provide scalable and efficient multi-object search capabilities in real-world, partially observable settings. To address this gap, we introduce OpenGuide, an assistive mobile robot system that combines natural language understanding with vision-language foundation models (VLM), frontier-based exploration, and a Partially Observable Markov Decision Process (POMDP) planner. OpenGuide interprets open-vocabulary requests, reasons about object-scene relationships, and adaptively navigates and localizes multiple target items in novel environments. Our approach enables robust recovery from missed detections through value decay and belief-space reasoning, resulting in more effective exploration and object localization. We validate OpenGuide in simulated and real-world experiments, demonstrating substantial improvements in task success rate and search efficiency over prior methods. This work establishes a foundation for scalable, human-centered robotic assistance in assisted living environments. 

**Abstract (ZH)**: 室内建筑设计，如住宅和办公室，常常具有复杂且杂乱的布局，这对盲人或视觉受损者在执行查找和收集多个物体的任务时构成了重大挑战。虽然许多现有的辅助技术侧重于基本导航或避障，但在部分可观测的真实世界环境中，很少有系统能够提供可扩展且高效的多目标搜索能力。为解决这一问题，我们介绍了一种名为OpenGuide的辅助移动机器人系统，该系统结合了自然语言理解、视觉-语言基础模型（VLM）、边疆导向探索以及部分可观测马尔可夫决策过程（POMDP）规划。OpenGuide能够解析开放词汇请求，推断物体-场景关系，并在新型环境中自适应导航和定位多个目标物品。我们的方法通过价值衰减和信念空间推理实现出色的错误恢复，从而实现更有效的探索和物体定位。我们通过仿真和现实世界实验验证了OpenGuide，证明其在任务成功率和搜索效率方面较以前的方法有显著提升。这项工作为在辅助生活环境中提供可扩展且以人为中心的机器人辅助奠定了基础。 

---
# Language-Guided Long Horizon Manipulation with LLM-based Planning and Visual Perception 

**Title (ZH)**: 基于LLM的规划与视觉感知的语言指导长时 horizons 操作 

**Authors**: Changshi Zhou, Haichuan Xu, Ningquan Gu, Zhipeng Wang, Bin Cheng, Pengpeng Zhang, Yanchao Dong, Mitsuhiro Hayashibe, Yanmin Zhou, Bin He  

**Link**: [PDF](https://arxiv.org/pdf/2509.02324)  

**Abstract**: Language-guided long-horizon manipulation of deformable objects presents significant challenges due to high degrees of freedom, complex dynamics, and the need for accurate vision-language grounding. In this work, we focus on multi-step cloth folding, a representative deformable-object manipulation task that requires both structured long-horizon planning and fine-grained visual perception. To this end, we propose a unified framework that integrates a Large Language Model (LLM)-based planner, a Vision-Language Model (VLM)-based perception system, and a task execution module. Specifically, the LLM-based planner decomposes high-level language instructions into low-level action primitives, bridging the semantic-execution gap, aligning perception with action, and enhancing generalization. The VLM-based perception module employs a SigLIP2-driven architecture with a bidirectional cross-attention fusion mechanism and weight-decomposed low-rank adaptation (DoRA) fine-tuning to achieve language-conditioned fine-grained visual grounding. Experiments in both simulation and real-world settings demonstrate the method's effectiveness. In simulation, it outperforms state-of-the-art baselines by 2.23, 1.87, and 33.3 on seen instructions, unseen instructions, and unseen tasks, respectively. On a real robot, it robustly executes multi-step folding sequences from language instructions across diverse cloth materials and configurations, demonstrating strong generalization in practical scenarios. Project page: this https URL 

**Abstract (ZH)**: 基于语言指导的长时序变形物体 manipulation 面临极大挑战，主要原因包括高自由度、复杂动力学以及精确的视觉-语言匹配需求。本文聚焦于多步布料折叠任务，这是一种既需要结构化长时序规划又需要精细视觉感知的典型变形物体 manipulation 任务。为此，我们提出了一种统一框架，该框架整合了基于大型语言模型 (LLM) 的规划器、基于视觉-语言模型 (VLM) 的感知系统以及任务执行模块。具体来说，基于 LLM 的规划器将高层语言指令分解为底层动作 primitive，从而弥合语义执行差距、使感知与动作相协调，并增强泛化能力。基于 VLM 的感知模块采用由 SigLIP2 驱动的架构，并结合双向跨注意力融合机制和基于权重分解的低秩适应（DoRA）微调，以实现语言条件下的精细视觉匹配。在仿真和现实环境中的实验均证明了该方法的有效性。在仿真环境中，该方法分别在已见过的指令、未见过的指令和未见过的任务上，比最先进的基线方法高出 2.23、1.87 和 33.3。在现实机器人上，该方法能够从语言指令中稳健地执行跨不同布料材料和配置的多步折叠序列，展示了在实际场景中的强大泛化能力。 

---
# Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety 

**Title (ZH)**: 提升基于LLM的机器人系统可靠性：安全与可靠性统一方法 

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Bräunl, Jin B. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.02163)  

**Abstract**: Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at this https URL 

**Abstract (ZH)**: 将大规模语言模型（LLMs）集成到机器人系统中， telah彻底变革了体态人工智能，使其能够实现高级决策和适应性。然而，确保可靠性，包括对抗 adversarial 攻击的安全性和在复杂环境中的安全性，仍然是一个关键挑战。为了解决这一问题，我们提出了一个统一框架，该框架通过健壮的验证机制来减轻提示注入攻击并确保操作安全性。我们的方法结合了提示构建、状态管理和安全性验证，并通过性能和安全性指标进行了评估。实验结果显示，在受注入攻击影响的情况下，该方法在性能上提高了30.8%，在对抗性条件下，在复杂环境设置中提高了325%。这项工作在基于LLM的机器人系统中填补了安全性和安全性之间的差距，提供了在实际环境中部署可靠的LLM集成移动机器人的一系列实用洞察。该框架已在 https://xxxxx 开源，并提供了仿真和物理部署演示。 

---
# Align-Then-stEer: Adapting the Vision-Language Action Models through Unified Latent Guidance 

**Title (ZH)**: 对准然后引导：通过统一潜在指导适应视觉-语言动作模型 

**Authors**: Yang Zhang, Chenwei Wang, Ouyang Lu, Yuan Zhao, Yunfei Ge, Zhenglong Sun, Xiu Li, Chi Zhang, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02055)  

**Abstract**: Vision-Language-Action (VLA) models pre-trained on large, diverse datasets show remarkable potential for general-purpose robotic manipulation. However, a primary bottleneck remains in adapting these models to downstream tasks, especially when the robot's embodiment or the task itself differs from the pre-training data. This discrepancy leads to a significant mismatch in action distributions, demanding extensive data and compute for effective fine-tuning. To address this challenge, we introduce \textbf{Align-Then-stEer (\texttt{ATE})}, a novel, data-efficient, and plug-and-play adaptation framework. \texttt{ATE} first aligns disparate action spaces by constructing a unified latent space, where a variational autoencoder constrained by reverse KL divergence embeds adaptation actions into modes of the pre-training action latent distribution. Subsequently, it steers the diffusion- or flow-based VLA's generation process during fine-tuning via a guidance mechanism that pushes the model's output distribution towards the target domain. We conduct extensive experiments on cross-embodiment and cross-task manipulation in both simulation and real world. Compared to direct fine-tuning of representative VLAs, our method improves the average multi-task success rate by up to \textbf{9.8\%} in simulation and achieves a striking \textbf{32\% success rate gain} in a real-world cross-embodiment setting. Our work presents a general and lightweight solution that greatly enhances the practicality of deploying VLA models to new robotic platforms and tasks. 

**Abstract (ZH)**: Vision-Language-Action (\texttt{VLA}) 模型在大规模多元化数据集上预训练后，在通用机器人操作方面展现出显著潜力。然而，将其适应下游任务的主要瓶颈在于，尤其是当机器人具有的实体或任务本身与预训练数据不同。这种差异导致了动作分布的重大不匹配，需要大量数据和计算资源以实现有效的微调。为解决这一挑战，我们引入了\textbf{Align-Then-Steer (\texttt{ATE})}，这是一种新颖的数据高效且即插即用的适应框架。\texttt{ATE} 首先通过构建统一的潜在空间来对齐不同的动作空间，并利用受逆KL散度约束的变分自编码器将适应动作嵌入预训练动作潜在分布的模式中。随后，它在微调过程中通过指导机制调整基于扩散或流的 \texttt{VLA} 生成过程，将模型的输出分布引导至目标域。我们在模拟和现实世界中进行了广泛的跨实体和跨任务操作实验。与直接微调代表性 \texttt{VLA} 相比，我们的方法在模拟环境中将多任务成功率平均提高了 \textbf{9.8\%}，在现实世界的跨实体设置中惊人地提高了 \textbf{32\%} 的成功率。我们的工作提出了一种通用且轻量的解决方案，极大地提高了部署 \texttt{VLA} 模型到新型机器人平台和任务的实用性。 

---
# ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training 

**Title (ZH)**: ManiFlow：一致性流训练的通用机器人 manipulation 策略 

**Authors**: Ge Yan, Jiyue Zhu, Yuquan Deng, Shiqi Yang, Ri-Zhao Qiu, Xuxin Cheng, Marius Memmel, Ranjay Krishna, Ankit Goyal, Xiaolong Wang, Dieter Fox  

**Link**: [PDF](https://arxiv.org/pdf/2509.01819)  

**Abstract**: This paper introduces ManiFlow, a visuomotor imitation learning policy for general robot manipulation that generates precise, high-dimensional actions conditioned on diverse visual, language and proprioceptive inputs. We leverage flow matching with consistency training to enable high-quality dexterous action generation in just 1-2 inference steps. To handle diverse input modalities efficiently, we propose DiT-X, a diffusion transformer architecture with adaptive cross-attention and AdaLN-Zero conditioning that enables fine-grained feature interactions between action tokens and multi-modal observations. ManiFlow demonstrates consistent improvements across diverse simulation benchmarks and nearly doubles success rates on real-world tasks across single-arm, bimanual, and humanoid robot setups with increasing dexterity. The extensive evaluation further demonstrates the strong robustness and generalizability of ManiFlow to novel objects and background changes, and highlights its strong scaling capability with larger-scale datasets. Our website: this http URL. 

**Abstract (ZH)**: 本文介绍了ManiFlow，一种基于视觉和运动模仿学习的通用机器人 manipulation 策略，能够根据多样化视觉、语言和本体感受输入生成精确的高维动作。我们利用流匹配和一致性训练，使机器人能够在仅1-2个推理步骤中生成高质量的灵巧动作。为有效处理多种输入模态，我们提出了一种名为DiT-X的扩散变压器架构，该架构具有自适应交叉注意力和AdaLN-Zero条件，能够实现动作标记与多模态观测之间的细粒度特征交互。ManiFlow在多种模拟基准测试中表现出一致的改进，并且在单臂、双手和类人机器人配置中成功执行各种任务，成功率几乎翻倍，随着灵巧程度的增加而提高。广泛的评估进一步证明了ManiFlow对新物体和背景变化的强健性和泛化能力，并突显了其在更大规模数据集上的强大扩展能力。我们的网站：[此处填网址]。 

---
# Non-conflicting Energy Minimization in Reinforcement Learning based Robot Control 

**Title (ZH)**: 基于强化学习的机器人控制中非冲突的能量最小化 

**Authors**: Skand Peri, Akhil Perincherry, Bikram Pandit, Stefan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01765)  

**Abstract**: Efficient robot control often requires balancing task performance with energy expenditure. A common approach in reinforcement learning (RL) is to penalize energy use directly as part of the reward function. This requires carefully tuning weight terms to avoid undesirable trade-offs where energy minimization harms task success. In this work, we propose a hyperparameter-free gradient optimization method to minimize energy expenditure without conflicting with task performance. Inspired by recent works in multitask learning, our method applies policy gradient projection between task and energy objectives to derive policy updates that minimize energy expenditure in ways that do not impact task performance. We evaluate this technique on standard locomotion benchmarks of DM-Control and HumanoidBench and demonstrate a reduction of 64% energy usage while maintaining comparable task performance. Further, we conduct experiments on a Unitree GO2 quadruped showcasing Sim2Real transfer of energy efficient policies. Our method is easy to implement in standard RL pipelines with minimal code changes, is applicable to any policy gradient method, and offers a principled alternative to reward shaping for energy efficient control policies. 

**Abstract (ZH)**: 无需超参数的梯度优化方法在不冲突于任务性能的前提下最小化能量消耗 

---
# Fail2Progress: Learning from Real-World Robot Failures with Stein Variational Inference 

**Title (ZH)**: 从Stein变分推断学习现实世界机器人故障 

**Authors**: Yixuan Huang, Novella Alvina, Mohanraj Devendran Shanthi, Tucker Hermans  

**Link**: [PDF](https://arxiv.org/pdf/2509.01746)  

**Abstract**: Skill effect models for long-horizon manipulation tasks are prone to failures in conditions not covered by training data distributions. Therefore, enabling robots to reason about and learn from failures is necessary. We investigate the problem of efficiently generating a dataset targeted to observed failures. After fine-tuning a skill effect model on this dataset, we evaluate the extent to which the model can recover from failures and minimize future failures. We propose Fail2Progress, an approach that leverages Stein variational inference to generate multiple simulation environments in parallel, enabling efficient data sample generation similar to observed failures. Our method is capable of handling several challenging mobile manipulation tasks, including transporting multiple objects, organizing a constrained shelf, and tabletop organization. Through large-scale simulation and real-world experiments, we demonstrate that our approach excels at learning from failures across different numbers of objects. Furthermore, we show that Fail2Progress outperforms several baselines. 

**Abstract (ZH)**: 长时操作任务的能力影响模型在未覆盖训练数据分布的条件下容易发生故障，因此让机器人能够推理和学习故障是必要的。我们研究了生成针对观察到的故障的目标化数据集的问题。在对这个数据集微调能力影响模型后，我们评估了模型从故障中恢复以及减少未来故障的程度。我们提出了Fail2Progress方法，该方法利用Stein变分推断并行生成多个仿真环境，实现类似于观察到的故障的高效数据样本生成。我们的方法能够处理多个具有挑战性的移动操作任务，包括多物体搬运、受限货架整理和桌面整理。通过大规模仿真和实际实验，我们展示了我们的方法在不同物体数量下从故障中学习方面的优越性。此外，我们表明Fail2Progress优于几个基准方法。 

---
# Constrained Decoding for Robotics Foundation Models 

**Title (ZH)**: 受限解码for机器人基础模型 

**Authors**: Parv Kapoor, Akila Ganlath, Changliu Liu, Sebastian Scherer, Eunsuk Kang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01728)  

**Abstract**: Recent advances in the development of robotic foundation models have led to promising end-to-end and general-purpose capabilities in robotic systems. These models are pretrained on vast datasets of robot trajectories to process multi- modal inputs and directly output a sequence of action that the system then executes in the real world. Although this approach is attractive from the perspective of im- proved generalization across diverse tasks, these models are still data-driven and, therefore, lack explicit notions of behavioral correctness and safety constraints. We address these limitations by introducing a constrained decoding framework for robotics foundation models that enforces logical constraints on action trajec- tories in dynamical systems. Our method ensures that generated actions provably satisfy signal temporal logic (STL) specifications at runtime without retraining, while remaining agnostic of the underlying foundation model. We perform com- prehensive evaluation of our approach across state-of-the-art navigation founda- tion models and we show that our decoding-time interventions are useful not only for filtering unsafe actions but also for conditional action-generation. Videos available on our website: this https URL 

**Abstract (ZH)**: 近期在机器人基础模型发展的进步已经引发了在机器人系统中端到端和通用能力的有希望的成果。这些模型在大规模机器人轨迹数据集上进行预训练，以处理多模式输入并直接输出系统在真实世界中执行的一系列动作。尽管从增强跨多种任务的一般化角度来看这种方法具有吸引力，但这些模型仍然是数据驱动的，因此缺乏明确的行为正确性和安全约束的概念。我们通过引入一种受限解码框架来解决这些限制，该框架在动态系统中对动作轨迹施加逻辑约束。我们的方法确保在运行时生成的动作可以证明满足信号时序逻辑（STL）规范，而无需重新训练，并且对底层基础模型保持无偏见。我们对最先进的导航基础模型进行了全面评估，并表明我们的解码时干预不仅有助于筛选出不安全的动作，还可以用于条件动作生成。更多视频请参见我们的网站: this https URL 

---
# MoTo: A Zero-shot Plug-in Interaction-aware Navigation for General Mobile Manipulation 

**Title (ZH)**: MoTo: 零样本插件式交互感知导航用于通用移动操作 

**Authors**: Zhenyu Wu, Angyuan Ma, Xiuwei Xu, Hang Yin, Yinan Liang, Ziwei Wang, Jiwen Lu, Haibin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01658)  

**Abstract**: Mobile manipulation stands as a core challenge in robotics, enabling robots to assist humans across varied tasks and dynamic daily environments. Conventional mobile manipulation approaches often struggle to generalize across different tasks and environments due to the lack of large-scale training. However, recent advances in manipulation foundation models demonstrate impressive generalization capability on a wide range of fixed-base manipulation tasks, which are still limited to a fixed setting. Therefore, we devise a plug-in module named MoTo, which can be combined with any off-the-shelf manipulation foundation model to empower them with mobile manipulation ability. Specifically, we propose an interaction-aware navigation policy to generate robot docking points for generalized mobile manipulation. To enable zero-shot ability, we propose an interaction keypoints framework via vision-language models (VLM) under multi-view consistency for both target object and robotic arm following instructions, where fixed-base manipulation foundation models can be employed. We further propose motion planning objectives for the mobile base and robot arm, which minimize the distance between the two keypoints and maintain the physical feasibility of trajectories. In this way, MoTo guides the robot to move to the docking points where fixed-base manipulation can be successfully performed, and leverages VLM generation and trajectory optimization to achieve mobile manipulation in a zero-shot manner, without any requirement on mobile manipulation expert data. Extensive experimental results on OVMM and real-world demonstrate that MoTo achieves success rates of 2.68% and 16.67% higher than the state-of-the-art mobile manipulation methods, respectively, without requiring additional training data. 

**Abstract (ZH)**: 移动操作是机器人学中的核心挑战，使机器人能够在多种任务和动态日常环境中辅助人类。传统的移动操作方法由于缺乏大规模训练数据，往往难以在不同任务和环境中泛化。然而，近期的操纵基础模型进展展示了在一系列固定基座操作任务上出色的泛化能力，这些任务仍局限于静态环境。因此，我们设计了一个名为MoTo的插件模块，可以与任何现成的操纵基础模型结合，赋予其实现移动操作的能力。具体而言，我们提出了一种交互感知导航策略以生成通用移动操作的机器人对接点。为了实现零样本能力，我们通过多视角一致性下的视觉-语言模型（VLM）提出了交互关键点框架，用于目标对象和机器人臂跟随指令的场景，可以利用固定的基座操作基础模型。我们进一步提出了移动基座和机器人臂的动力学规划目标，最小化两个关键点之间的距离并保持轨迹的物理可行性。通过这种方式，MoTo指导机器人移动到可以成功执行固定基座操作的对接点，并利用VLM生成和轨迹优化以零样本方式实现移动操作，无需移动操作专家数据。在OVMM和实际环境中的广泛实验结果显示，MoTo分别比最先进的移动操作方法成功率达到2.68%和16.67%的提升，且无需额外的训练数据。 

---
# A Hybrid Input based Deep Reinforcement Learning for Lane Change Decision-Making of Autonomous Vehicle 

**Title (ZH)**: 基于混合输入的深度强化学习的自主车辆变道决策方法 

**Authors**: Ziteng Gao, Jiaqi Qu, Chaoyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.01611)  

**Abstract**: Lane change decision-making for autonomous vehicles is a complex but high-reward behavior. In this paper, we propose a hybrid input based deep reinforcement learning (DRL) algorithm, which realizes abstract lane change decisions and lane change actions for autonomous vehicles within traffic flow. Firstly, a surrounding vehicles trajectory prediction method is proposed to reduce the risk of future behavior of surrounding vehicles to ego vehicle, and the prediction results are input into the reinforcement learning model as additional information. Secondly, to comprehensively leverage environmental information, the model extracts feature from high-dimensional images and low-dimensional sensor data simultaneously. The fusion of surrounding vehicle trajectory prediction and multi-modal information are used as state space of reinforcement learning to improve the rationality of lane change decision. Finally, we integrate reinforcement learning macro decisions with end-to-end vehicle control to achieve a holistic lane change process. Experiments were conducted within the CARLA simulator, and the results demonstrated that the utilization of a hybrid state space significantly enhances the safety of vehicle lane change decisions. 

**Abstract (ZH)**: 自主驾驶车辆车道变更决策是一种复杂但高收益的行为：基于混合输入的深度强化学习算法及其应用研究 

---
# FGO-SLAM: Enhancing Gaussian SLAM with Globally Consistent Opacity Radiance Field 

**Title (ZH)**: FGO-SLAM：增强Gaussian SLAM的全局一致透明辐射场方法 

**Authors**: Fan Zhu, Yifan Zhao, Ziyu Chen, Biao Yu, Hui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01547)  

**Abstract**: Visual SLAM has regained attention due to its ability to provide perceptual capabilities and simulation test data for Embodied AI. However, traditional SLAM methods struggle to meet the demands of high-quality scene reconstruction, and Gaussian SLAM systems, despite their rapid rendering and high-quality mapping capabilities, lack effective pose optimization methods and face challenges in geometric reconstruction. To address these issues, we introduce FGO-SLAM, a Gaussian SLAM system that employs an opacity radiance field as the scene representation to enhance geometric mapping performance. After initial pose estimation, we apply global adjustment to optimize camera poses and sparse point cloud, ensuring robust tracking of our approach. Additionally, we maintain a globally consistent opacity radiance field based on 3D Gaussians and introduce depth distortion and normal consistency terms to refine the scene representation. Furthermore, after constructing tetrahedral grids, we identify level sets to directly extract surfaces from 3D Gaussians. Results across various real-world and large-scale synthetic datasets demonstrate that our method achieves state-of-the-art tracking accuracy and mapping performance. 

**Abstract (ZH)**: 视觉SLAM由于其为感知能力以及实体AI的仿真测试数据提供支持而重新获得了关注。然而，传统的SLAM方法难以满足高质量场景重建的需求，尽管高斯SLAM系统具有快速渲染和高质量建图能力，但缺乏有效的姿态优化方法，并在几何重建方面面临挑战。为此，我们引入了FGO-SLAM，这是一种采用不透明辐射场作为场景表示的高斯SLAM系统，以提高几何建图性能。在初始姿态估计之后，我们应用全局调整来优化相机姿态和稀疏点云，确保我们的方法具有鲁棒的跟踪能力。此外，基于三维高斯分布，我们维护全局一致的不透明辐射场，并引入深度失真和法线一致性项以细化场景表示。进一步地，在构建四面体网格之后，我们识别等值集以直接从三维高斯分布中提取表面。在多种真实世界和大规模合成数据集上的结果表明，我们的方法达到了最先进的跟踪精度和建图性能。 

---
# TopoNav: Topological Graphs as a Key Enabler for Advanced Object Navigation 

**Title (ZH)**: TopoNav: 抽象拓扑图作为高级对象导航的关键使能器 

**Authors**: Peiran Liu, Qiang Zhang, Daojie Peng, Lingfeng Zhang, Yihao Qin, Hang Zhou, Jun Ma, Renjing Xu, Yiding Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.01364)  

**Abstract**: Object Navigation (ObjectNav) has made great progress with large language models (LLMs), but still faces challenges in memory management, especially in long-horizon tasks and dynamic scenes. To address this, we propose TopoNav, a new framework that leverages topological structures as spatial memory. By building and updating a topological graph that captures scene connections, adjacency, and semantic meaning, TopoNav helps agents accumulate spatial knowledge over time, retrieve key information, and reason effectively toward distant goals. Our experiments show that TopoNav achieves state-of-the-art performance on benchmark ObjectNav datasets, with higher success rates and more efficient paths. It particularly excels in diverse and complex environments, as it connects temporary visual inputs with lasting spatial understanding. 

**Abstract (ZH)**: 基于拓扑结构的空间记忆物体导航（TopoNav）在大型语言模型（LLMs）的驱动下取得了显著进展，但仍面临记忆管理的挑战，尤其是在长期任务和动态场景中。为解决这一问题，我们提出了一种新的框架TopoNav，该框架利用拓扑结构作为空间记忆。通过构建并更新一个捕获场景连接、相邻关系和语义意义的拓扑图，TopoNav帮助代理积累空间知识，检索关键信息，并有效地朝着远距离目标进行推理。我们的实验表明，TopoNav在基准物体导航数据集上取得了最先进的性能，具有更高的成功率和更高效的路径。特别是在多样性和复杂性环境中，TopoNav特别出色，因为它将临时的视觉输入与持久的空间理解相连接。 

---
# SR-SLAM: Scene-reliability Based RGB-D SLAM in Diverse Environments 

**Title (ZH)**: SR-SLAM: 基于场景可靠性的RGB-D SLAM在多样环境中 

**Authors**: Haolan Zhang, Chenghao Li, Thanh Nguyen Canh, Lijun Wang, Nak Young Chong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01111)  

**Abstract**: Visual simultaneous localization and mapping (SLAM) plays a critical role in autonomous robotic systems, especially where accurate and reliable measurements are essential for navigation and sensing. In feature-based SLAM, the quantityand quality of extracted features significantly influence system performance. Due to the variations in feature quantity and quality across diverse environments, current approaches face two major challenges: (1) limited adaptability in dynamic feature culling and pose estimation, and (2) insufficient environmental awareness in assessment and optimization strategies. To address these issues, we propose SRR-SLAM, a scene-reliability based framework that enhances feature-based SLAM through environment-aware processing. Our method introduces a unified scene reliability assessment mechanism that incorporates multiple metrics and historical observations to guide system behavior. Based on this assessment, we develop: (i) adaptive dynamic region selection with flexible geometric constraints, (ii) depth-assisted self-adjusting clustering for efficient dynamic feature removal in high-dimensional settings, and (iii) reliability-aware pose refinement that dynamically integrates direct methods when features are insufficient. Furthermore, we propose (iv) reliability-based keyframe selection and a weighted optimization scheme to reduce computational overhead while improving estimation accuracy. Extensive experiments on public datasets and real world scenarios show that SRR-SLAM outperforms state-of-the-art dynamic SLAM methods, achieving up to 90% improvement in accuracy and robustness across diverse environments. These improvements directly contribute to enhanced measurement precision and reliability in autonomous robotic sensing systems. 

**Abstract (ZH)**: 基于场景可靠性的SLAM（视觉同步定位与建图）在自主机器人系统中的关键作用及其环境感知处理框架 

---
# Enhanced Mean Field Game for Interactive Decision-Making with Varied Stylish Multi-Vehicles 

**Title (ZH)**: 增强的均场游戏及其在多元風格 Vehicles 的交互决策中的应用 

**Authors**: Liancheng Zheng, Zhen Tian, Yangfan He, Shuo Liu, Ke Gong, Huilin Chen, Zhihao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.00981)  

**Abstract**: This paper presents an MFG-based decision-making framework for autonomous driving in heterogeneous traffic. To capture diverse human behaviors, we propose a quantitative driving style representation that maps abstract traits to parameters such as speed, safety factors, and reaction time. These parameters are embedded into the MFG through a spatial influence field model. To ensure safe operation in dense traffic, we introduce a safety-critical lane-changing algorithm that leverages dynamic safety margins, time-to-collision analysis, and multi-layered constraints. Real-world NGSIM data is employed for style calibration and empirical validation. Experimental results demonstrate zero collisions across six style combinations, two 15-vehicle scenarios, and NGSIM-based trials, consistently outperforming conventional game-theoretic baselines. Overall, our approach provides a scalable, interpretable, and behavior-aware planning framework for real-world autonomous driving applications. 

**Abstract (ZH)**: 基于MFG的异构交通自主驾驶决策框架 

---
# CARIS: A Context-Adaptable Robot Interface System for Personalized and Scalable Human-Robot Interaction 

**Title (ZH)**: CARIS: 一种适用于个性化和可扩展人机交互的上下文自适应机器人接口系统 

**Authors**: Felipe Arias-Russi, Yuanchen Bai, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2509.00660)  

**Abstract**: The human-robot interaction (HRI) field has traditionally used Wizard-of-Oz (WoZ) controlled robots to explore navigation, conversational dynamics, human-in-the-loop interactions, and more to explore appropriate robot behaviors in everyday settings. However, existing WoZ tools are often limited to one context, making them less adaptable across different settings, users, and robotic platforms. To mitigate these issues, we introduce a Context-Adaptable Robot Interface System (CARIS) that combines advanced robotic capabilities such teleoperation, human perception, human-robot dialogue, and multimodal data recording. Through pilot studies, we demonstrate the potential of CARIS to WoZ control a robot in two contexts: 1) mental health companion and as a 2) tour guide. Furthermore, we identified areas of improvement for CARIS, including smoother integration between movement and communication, clearer functionality separation, recommended prompts, and one-click communication options to enhance the usability wizard control of CARIS. This project offers a publicly available, context-adaptable tool for the HRI community, enabling researchers to streamline data-driven approaches to intelligent robot behavior. 

**Abstract (ZH)**: 人类与机器人交互领域中的适场景可调机器人接口系统（CARIS）：一种结合了远程操作、人类感知、人机对话和多模态数据记录的工具 

---
# Vehicle-in-Virtual-Environment (VVE) Method for Developing and Evaluating VRU Safety of Connected and Autonomous Driving with Focus on Bicyclist Safety 

**Title (ZH)**: 基于虚拟环境的车辆（VVE方法）在连接和自主驾驶中发展和评估VRU安全性，重点关注自行车骑行者安全 

**Authors**: Haochong Chen, Xincheng Cao, Bilin Aksun-Guvenc, Levent Guvenc  

**Link**: [PDF](https://arxiv.org/pdf/2509.00624)  

**Abstract**: Extensive research has already been conducted in the autonomous driving field to help vehicles navigate safely and efficiently. At the same time, plenty of current research on vulnerable road user (VRU) safety is performed which largely concentrates on perception, localization, or trajectory prediction of VRUs. However, existing research still exhibits several gaps, including the lack of a unified planning and collision avoidance system for autonomous vehicles, limited investigation into delay tolerant control strategies, and the absence of an efficient and standardized testing methodology. Ensuring VRU safety remains one of the most pressing challenges in autonomous driving, particularly in dynamic and unpredictable environments. In this two year project, we focused on applying the Vehicle in Virtual Environment (VVE) method to develop, evaluate, and demonstrate safety functions for Vulnerable Road Users (VRUs) using automated steering and braking of ADS. In this current second year project report, our primary focus was on enhancing the previous year results while also considering bicyclist safety. 

**Abstract (ZH)**: 自动驾驶领域已开展了大量研究以帮助车辆安全高效地导航。同时，大量关于脆弱道路使用者（VRU）安全的研究也集中在感知、定位或轨迹预测等方面。然而，现有研究仍存在一些差距，包括缺乏统一的规划和碰撞 avoidance 系统、对容错控制策略的调查有限，以及缺乏有效的标准化测试方法。确保 VRU 安全仍是自动驾驶中最 pressing 的挑战之一，特别是在动态和不可预测的环境中。在本两年期项目中，我们专注于采用车辆在虚拟环境（VVE）方法开发、评估和展示使用自动转向和制动的自动驾驶系统（ADS）的安全功能，特别是在骑自行车者安全方面的应用。在当前的第二年项目报告中，我们主要致力于增强去年的结果，同时考虑骑自行车者安全。 

---
# Galaxea Open-World Dataset and G0 Dual-System VLA Model 

**Title (ZH)**: Galaxea 开放世界数据集和 G0 双系统超分辨率模型 

**Authors**: Tao Jiang, Tianyuan Yuan, Yicheng Liu, Chenhao Lu, Jianning Cui, Xiao Liu, Shuiqi Cheng, Jiyang Gao, Huazhe Xu, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.00576)  

**Abstract**: We present Galaxea Open-World Dataset, a large-scale, diverse collection of robot behaviors recorded in authentic human living and working environments. All demonstrations are gathered using a consistent robotic embodiment, paired with precise subtask-level language annotations to facilitate both training and evaluation. Building on this dataset, we introduce G0, a dual-system framework that couples a Vision-Language Model (VLM) for multimodal planning with a Vision-Language-Action (VLA) model for fine-grained execution. G0 is trained using a three-stage curriculum: cross-embodiment pre-training, single-embodiment pre-training, and task-specific post-training. A comprehensive benchmark spanning tabletop manipulation, few-shot learning, and long-horizon mobile manipulation, demonstrates the effectiveness of our approach. In particular, we find that the single-embodiment pre-training stage, together with the Galaxea Open-World Dataset, plays a critical role in achieving strong performance. 

**Abstract (ZH)**: Galaxea 开放世界数据集及其在 G0 双系统框架中的应用 

---
# Learning Dolly-In Filming From Demonstration Using a Ground-Based Robot 

**Title (ZH)**: 基于地面机器人从示范中学习多利拍摄方法 

**Authors**: Philip Lorimer, Alan Hunter, Wenbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00574)  

**Abstract**: Cinematic camera control demands a balance of precision and artistry - qualities that are difficult to encode through handcrafted reward functions. While reinforcement learning (RL) has been applied to robotic filmmaking, its reliance on bespoke rewards and extensive tuning limits creative usability. We propose a Learning from Demonstration (LfD) approach using Generative Adversarial Imitation Learning (GAIL) to automate dolly-in shots with a free-roaming, ground-based filming robot. Expert trajectories are collected via joystick teleoperation in simulation, capturing smooth, expressive motion without explicit objective design.
Trained exclusively on these demonstrations, our GAIL policy outperforms a PPO baseline in simulation, achieving higher rewards, faster convergence, and lower variance. Crucially, it transfers directly to a real-world robot without fine-tuning, achieving more consistent framing and subject alignment than a prior TD3-based method. These results show that LfD offers a robust, reward-free alternative to RL in cinematic domains, enabling real-time deployment with minimal technical effort. Our pipeline brings intuitive, stylized camera control within reach of creative professionals, bridging the gap between artistic intent and robotic autonomy. 

**Abstract (ZH)**: 电影摄像控制需要精确与艺术的平衡——一种难以通过手工制作的奖励函数编码的品质。虽然强化学习（RL）已被应用于机器人电影制作，但其依赖于定制的奖励函数和大量的调优限制了其创意的实用性。我们提出了一种使用生成对抗模仿学习（GAIL）的示例学习（LfD）方法，以自动实现自由移动地面拍摄机器人的推进入镜头。专家轨迹通过仿真中的手柄遥操作采集，捕捉到平滑、表现力强的运动，而无需明确的目标设计。
仅通过这些示例训练，我们的GAIL策略在仿真中优于PPO基线，获得更高的奖励、更快的收敛性和更低的方差。 crucial 地，它无需微调即可直接转移到实际机器人上，实现比先前基于TD3的方法更一致的构图和主题对齐。这些结果表明，示例学习提供了在电影领域中RL的一种稳健的、无需奖励的替代方案，能够实现最小技术努力下的实时部署。我们的流程将直观的、风格化的摄像控制带给了创意专业人士，弥合了艺术意图与机器人自主之间的差距。 

---
# ConceptBot: Enhancing Robot's Autonomy through Task Decomposition with Large Language Models and Knowledge Graph 

**Title (ZH)**: ConceptBot: 通过任务分解增强机器人自主性的大语言模型与知识图谱 

**Authors**: Alessandro Leanza, Angelo Moroncelli, Giuseppe Vizzari, Francesco Braghin, Loris Roveda, Blerina Spahiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00570)  

**Abstract**: ConceptBot is a modular robotic planning framework that combines Large Language Models and Knowledge Graphs to generate feasible and risk-aware plans despite ambiguities in natural language instructions and correctly analyzing the objects present in the environment - challenges that typically arise from a lack of commonsense reasoning. To do that, ConceptBot integrates (i) an Object Property Extraction (OPE) module that enriches scene understanding with semantic concepts from ConceptNet, (ii) a User Request Processing (URP) module that disambiguates and structures instructions, and (iii) a Planner that generates context-aware, feasible pick-and-place policies. In comparative evaluations against Google SayCan, ConceptBot achieved 100% success on explicit tasks, maintained 87% accuracy on implicit tasks (versus 31% for SayCan), reached 76% on risk-aware tasks (versus 15%), and outperformed SayCan in application-specific scenarios, including material classification (70% vs. 20%) and toxicity detection (86% vs. 36%). On SafeAgentBench, ConceptBot achieved an overall score of 80% (versus 46% for the next-best baseline). These results, validated in both simulation and laboratory experiments, demonstrate ConceptBot's ability to generalize without domain-specific training and to significantly improve the reliability of robotic policies in unstructured environments. Website: this https URL 

**Abstract (ZH)**: ConceptBot是一个模块化的机器人规划框架，结合了大规模语言模型和知识图谱，以生成可行性高且风险意识强的计划，即使在自然语言指令存在歧义的情况下也能进行正确的环境物体分析——这通常源于常识推理的不足。该框架通过集成（i）对象属性提取（OPE）模块，（ii）用户请求处理（URP）模块，以及（iii）规划器来实现这一目标。在与Google SayCan的比较评估中，ConceptBot在显式任务中实现了100%的成功率，在隐式任务中的准确率为87%（而SayCan为31%），在风险意识任务中的准确率为76%（而SayCan为15%），并且在特定应用场景中优于SayCan，包括材料分类（70%对20%）和毒性检测（86%对36%）。在SafeAgentBench上，ConceptBot的整体得分为80%，而下一个最佳基线得分为46%。这些结果在仿真和实验室实验中的验证证明了ConceptBot的能力，即无需领域特定训练即可泛化，并显著提高了机器人策略在非结构化环境中的可靠性。网站：this https URL。 

---
# Reinforcement Learning of Dolly-In Filming Using a Ground-Based Robot 

**Title (ZH)**: 基于地面机器人学习黛西影视拍摄方法 

**Authors**: Philip Lorimer, Jack Saunders, Alan Hunter, Wenbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00564)  

**Abstract**: Free-roaming dollies enhance filmmaking with dynamic movement, but challenges in automated camera control remain unresolved. Our study advances this field by applying Reinforcement Learning (RL) to automate dolly-in shots using free-roaming ground-based filming robots, overcoming traditional control hurdles. We demonstrate the effectiveness of combined control for precise film tasks by comparing it to independent control strategies. Our robust RL pipeline surpasses traditional Proportional-Derivative controller performance in simulation and proves its efficacy in real-world tests on a modified ROSBot 2.0 platform equipped with a camera turret. This validates our approach's practicality and sets the stage for further research in complex filming scenarios, contributing significantly to the fusion of technology with cinematic creativity. This work presents a leap forward in the field and opens new avenues for research and development, effectively bridging the gap between technological advancement and creative filmmaking. 

**Abstract (ZH)**: 自由漫游地景机器人增强电影制作动态运动，但自动化摄像控制仍面临挑战。通过应用强化学习（RL）自动化自由漫游地面拍摄机器人进行地景推拉镜头，本研究克服了传统控制难题。我们通过与独立控制策略的比较，展示了结合控制在精确电影任务中的有效性。我们的稳健RL管道在仿真中超过传统比例-微分控制器性能，并在改装的ROSBot 2.0平台上进行实际测试，验证了其有效性，为复杂拍摄场景的研究奠定了基础，显著促进了技术与 Cinematic 创意的融合。此项工作在该领域取得了突破，并为研究与发展开辟了新途径，有效弥合了技术进步与创意 filmmaking 之间的差距。 

---
# Embodied Spatial Intelligence: from Implicit Scene Modeling to Spatial Reasoning 

**Title (ZH)**: 具身空间 intelligence: 从隐含场景建模到空间推理 

**Authors**: Jiading Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00465)  

**Abstract**: This thesis introduces "Embodied Spatial Intelligence" to address the challenge of creating robots that can perceive and act in the real world based on natural language instructions. To bridge the gap between Large Language Models (LLMs) and physical embodiment, we present contributions on two fronts: scene representation and spatial reasoning. For perception, we develop robust, scalable, and accurate scene representations using implicit neural models, with contributions in self-supervised camera calibration, high-fidelity depth field generation, and large-scale reconstruction. For spatial reasoning, we enhance the spatial capabilities of LLMs by introducing a novel navigation benchmark, a method for grounding language in 3D, and a state-feedback mechanism to improve long-horizon decision-making. This work lays a foundation for robots that can robustly perceive their surroundings and intelligently act upon complex, language-based commands. 

**Abstract (ZH)**: 本论文引入“具身空间智能”以应对基于自然语言指令在真实世界中感知和行动的挑战。为了在大规模语言模型与物理具身之间搭建桥梁，我们在场景表示和空间推理两个方面进行了贡献。在感知方面，我们使用隐式神经模型开发了鲁棒性、可扩展性和高精度的场景表示，包括自监督摄像机校准、高保真深度场生成和大规模重建的贡献。在空间推理方面，我们通过引入新的导航基准、3D语言接地方法及改进长期决策的反馈机制来增强大规模语言模型的空间能力。本工作为能够稳健地感知周围环境并智能地执行基于语言的复杂指令的机器人奠定了基础。 

---
# Generative Visual Foresight Meets Task-Agnostic Pose Estimation in Robotic Table-Top Manipulation 

**Title (ZH)**: 生成式视觉预见与任务无关的Pose估计在机器人桌面Manipulation中的融合 

**Authors**: Chuye Zhang, Xiaoxiong Zhang, Wei Pan, Linfang Zheng, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00361)  

**Abstract**: Robotic manipulation in unstructured environments requires systems that can generalize across diverse tasks while maintaining robust and reliable performance. We introduce {GVF-TAPE}, a closed-loop framework that combines generative visual foresight with task-agnostic pose estimation to enable scalable robotic manipulation. GVF-TAPE employs a generative video model to predict future RGB-D frames from a single side-view RGB image and a task description, offering visual plans that guide robot actions. A decoupled pose estimation model then extracts end-effector poses from the predicted frames, translating them into executable commands via low-level controllers. By iteratively integrating video foresight and pose estimation in a closed loop, GVF-TAPE achieves real-time, adaptive manipulation across a broad range of tasks. Extensive experiments in both simulation and real-world settings demonstrate that our approach reduces reliance on task-specific action data and generalizes effectively, providing a practical and scalable solution for intelligent robotic systems. 

**Abstract (ZH)**: 在未结构化环境中进行机器人操作需要能够泛化到多种任务并保持鲁棒性和可靠性能的系统。我们引入了GVF-TAPE，这是一种闭环框架，结合了生成视觉预见性和任务无关的手位估计，以实现可扩展的机器人操作。GVF-TAPE采用生成视频模型，从单张侧视RGB图像和任务描述中预测未来RGB-D帧，提供视觉计划以指导机器人动作。然后，解耦的手位估计模型从预测的帧中提取末端执行器手位，并通过低级控制器将其转换为可执行命令。通过在闭环中迭代集成视觉预见性和手位估计，GVF-TAPE实现了对多种任务的实时、自适应操作。在仿真和实际环境中的 extensive 实验表明，我们的方法减少了对特定任务动作数据的依赖并有效地泛化，为智能机器人系统提供了实用和可扩展的解决方案。 

---
# Jacobian Exploratory Dual-Phase Reinforcement Learning for Dynamic Endoluminal Navigation of Deformable Continuum Robots 

**Title (ZH)**: 动态内腔导航中可变形 continuum 机器人双阶段探索雅可比强化学习 

**Authors**: Yu Tian, Chi Kit Ng, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.00329)  

**Abstract**: Deformable continuum robots (DCRs) present unique planning challenges due to nonlinear deformation mechanics and partial state observability, violating the Markov assumptions of conventional reinforcement learning (RL) methods. While Jacobian-based approaches offer theoretical foundations for rigid manipulators, their direct application to DCRs remains limited by time-varying kinematics and underactuated deformation dynamics. This paper proposes Jacobian Exploratory Dual-Phase RL (JEDP-RL), a framework that decomposes planning into phased Jacobian estimation and policy execution. During each training step, we first perform small-scale local exploratory actions to estimate the deformation Jacobian matrix, then augment the state representation with Jacobian features to restore approximate Markovianity. Extensive SOFA surgical dynamic simulations demonstrate JEDP-RL's three key advantages over proximal policy optimization (PPO) baselines: 1) Convergence speed: 3.2x faster policy convergence, 2) Navigation efficiency: requires 25% fewer steps to reach the target, and 3) Generalization ability: achieve 92% success rate under material property variations and achieve 83% (33% higher than PPO) success rate in the unseen tissue environment. 

**Abstract (ZH)**: 可变形连续机器人（DCRs）由于非线性变形力学和部分状态可观测性，提出了独特的规划挑战，违反了传统强化学习（RL）方法的马氏假设。尽管雅可比方法为刚性 manipulator 提供了理论基础，但将其直接应用于 DCRs 仍受限于时变运动学和欠驱动变形动力学。本文提出了一种雅可比探索双阶段 RL（JEDP-RL）框架，该框架将规划分解为雅可比估计阶段和策略执行阶段。在每一次训练步骤中，我们首先执行局部探索动作以估计变形雅可比矩阵，然后通过增加状态表示中的雅可比特征来恢复近似的马氏性。广泛的 SOFA 手术动力学仿真表明，与基线 proximal 策略优化（PPO）相比，JEDP-RL 具有三个关键优势：1) 收敛速度：3.2 倍更快的策略收敛速度，2) 导航效率：达到目标所需的步骤减少25%，3) 通用性：在材料性质变化下达到92%的成功率，在未见过的组织环境中达到83%（比PPO高33%）的成功率。 

---
# Mechanistic interpretability for steering vision-language-action models 

**Title (ZH)**: 基于机理的可解释性以指导视觉-语言-动作模型 

**Authors**: Bear Häon, Kaylene Stocking, Ian Chuang, Claire Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2509.00328)  

**Abstract**: Vision-Language-Action (VLA) models are a promising path to realizing generalist embodied agents that can quickly adapt to new tasks, modalities, and environments. However, methods for interpreting and steering VLAs fall far short of classical robotics pipelines, which are grounded in explicit models of kinematics, dynamics, and control. This lack of mechanistic insight is a central challenge for deploying learned policies in real-world robotics, where robustness and explainability are critical. Motivated by advances in mechanistic interpretability for large language models, we introduce the first framework for interpreting and steering VLAs via their internal representations, enabling direct intervention in model behavior at inference time. We project feedforward activations within transformer layers onto the token embedding basis, identifying sparse semantic directions - such as speed and direction - that are causally linked to action selection. Leveraging these findings, we introduce a general-purpose activation steering method that modulates behavior in real time, without fine-tuning, reward signals, or environment interaction. We evaluate this method on two recent open-source VLAs, Pi0 and OpenVLA, and demonstrate zero-shot behavioral control in simulation (LIBERO) and on a physical robot (UR5). This work demonstrates that interpretable components of embodied VLAs can be systematically harnessed for control - establishing a new paradigm for transparent and steerable foundation models in robotics. 

**Abstract (ZH)**: 基于视觉-语言-动作(Vision-Language-Action, VLA)模型的可解释操控与 steering 技术：迈向透明可控的机器人基础模型 

---
# Contact-Aided Navigation of Flexible Robotic Endoscope Using Deep Reinforcement Learning in Dynamic Stomach 

**Title (ZH)**: 基于深度强化学习的动态胃部中柔顺内窥镜的接触辅助导航 

**Authors**: Chi Kit Ng, Huxin Gao, Tian-Ao Ren, Jiewen Lai, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.00319)  

**Abstract**: Navigating a flexible robotic endoscope (FRE) through the gastrointestinal tract is critical for surgical diagnosis and treatment. However, navigation in the dynamic stomach is particularly challenging because the FRE must learn to effectively use contact with the deformable stomach walls to reach target locations. To address this, we introduce a deep reinforcement learning (DRL) based Contact-Aided Navigation (CAN) strategy for FREs, leveraging contact force feedback to enhance motion stability and navigation precision. The training environment is established using a physics-based finite element method (FEM) simulation of a deformable stomach. Trained with the Proximal Policy Optimization (PPO) algorithm, our approach achieves high navigation success rates (within 3 mm error between the FRE's end-effector and target) and significantly outperforms baseline policies. In both static and dynamic stomach environments, the CAN agent achieved a 100% success rate with 1.6 mm average error, and it maintained an 85% success rate in challenging unseen scenarios with stronger external disturbances. These results validate that the DRL-based CAN strategy substantially enhances FRE navigation performance over prior methods. 

**Abstract (ZH)**: 基于接触辅助的深度强化学习柔性内窥机器人导航策略 

---
# A Framework for Task and Motion Planning based on Expanding AND/OR Graphs 

**Title (ZH)**: 基于扩展AND/OR图的任务与运动规划框架 

**Authors**: Fulvio Mastrogiovanni, Antony Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2509.00317)  

**Abstract**: Robot autonomy in space environments presents unique challenges, including high perception and motion uncertainty, strict kinematic constraints, and limited opportunities for human intervention. Therefore, Task and Motion Planning (TMP) may be critical for autonomous servicing, surface operations, or even in-orbit missions, just to name a few, as it models tasks as discrete action sequencing integrated with continuous motion feasibility assessments. In this paper, we introduce a TMP framework based on expanding AND/OR graphs, referred to as TMP-EAOG, and demonstrate its adaptability to different scenarios. TMP-EAOG encodes task-level abstractions within an AND/OR graph, which expands iteratively as the plan is executed, and performs in-the-loop motion planning assessments to ascertain their feasibility. As a consequence, TMP-EAOG is characterised by the desirable properties of (i) robustness to a certain degree of uncertainty, because AND/OR graph expansion can accommodate for unpredictable information about the robot environment, (ii) controlled autonomy, since an AND/OR graph can be validated by human experts, and (iii) bounded flexibility, in that unexpected events, including the assessment of unfeasible motions, can lead to different courses of action as alternative paths in the AND/OR graph. We evaluate TMP-EAOG on two benchmark domains. We use a simulated mobile manipulator as a proxy for space-grade autonomous robots. Our evaluation shows that TMP-EAOG can deal with a wide range of challenges in the benchmarks. 

**Abstract (ZH)**: 空间环境中机器人的自主性面临着独特挑战，包括高感知和运动不确定性、严格的运动学约束以及有限的人工干预机会。因此，任务与运动规划（TMP）可能是自治服务、表层操作或在轨任务等任务的关键，因为它将任务建模为离散动作序列与连续运动可行性评估的集成。本文介绍了一种基于扩展AND/OR图的TMP框架，称为TMP-EAOG，并展示了其在不同场景下的适应性。TMP-EAOG在执行计划时迭代扩展AND/OR图，并进行在环运动规划评估以确定其可行性。因此，TMP-EAOG具有以下特点：（i）一定程度上对不确定性具有鲁棒性，因为AND/OR图的扩展可以适应对机器人环境的不可预测信息；（ii）可控的自主性，因为AND/OR图可以由人类专家验证；（iii）在动态环境中具有边界灵活性，意外事件，包括不可行运动的评估，可以导致AND/OR图中的不同行动路线。我们使用仿真移动 manipulator 作为太空级自主机器人代理，并在两个基准域上评估了TMP-EAOG。我们的评估结果表明，TMP-EAOG能够应对基准域中的各种挑战。 

---
# TReF-6: Inferring Task-Relevant Frames from a Single Demonstration for One-Shot Skill Generalization 

**Title (ZH)**: TReF-6: 从单次示范中推断任务相关框架以实现一次性技能泛化 

**Authors**: Yuxuan Ding, Shuangge Wang, Tesca Fitzgerald  

**Link**: [PDF](https://arxiv.org/pdf/2509.00310)  

**Abstract**: Robots often struggle to generalize from a single demonstration due to the lack of a transferable and interpretable spatial representation. In this work, we introduce TReF-6, a method that infers a simplified, abstracted 6DoF Task-Relevant Frame from a single trajectory. Our approach identifies an influence point purely from the trajectory geometry to define the origin for a local frame, which serves as a reference for parameterizing a Dynamic Movement Primitive (DMP). This influence point captures the task's spatial structure, extending the standard DMP formulation beyond start-goal imitation. The inferred frame is semantically grounded via a vision-language model and localized in novel scenes by Grounded-SAM, enabling functionally consistent skill generalization. We validate TReF-6 in simulation and demonstrate robustness to trajectory noise. We further deploy an end-to-end pipeline on real-world manipulation tasks, showing that TReF-6 supports one-shot imitation learning that preserves task intent across diverse object configurations. 

**Abstract (ZH)**: 机器人often struggled to generalize from a single demonstration due to the lack of a transferable and interpretable spatial representation. In this work, we introduce TReF-6, a method that infers a simplified, abstracted 6DoF Task-Relevant Frame from a single trajectory. Our approach identifies an influence point purely from the trajectory geometry to define the origin for a local frame, which serves as a reference for parameterizing a Dynamic Movement Primitive (DMP). This influence point captures the task's spatial structure, extending the standard DMP formulation beyond start-goal imitation. The inferred frame is semantically grounded via a vision-language model and localized in novel scenes by Grounded-SAM, enabling functionally consistent skill generalization. We validate TReF-6 in simulation and demonstrate robustness to trajectory noise. We further deploy an end-to-end pipeline on real-world manipulation tasks, showing that TReF-6 supports one-shot imitation learning that preserves task intent across diverse object configurations。

机器人往往难以通过单一次展示进行泛化，因为缺乏可转移和可解释的空间表示。在本文中，我们提出了TReF-6方法，可以从单个轨迹中推断出一个简化的、抽象化了的6自由度任务相关框（6DoF Task-Relevant Frame）。我们的方法仅从轨迹几何学中识别一个影响点来定义局部框的原点，该局部框作为参数化动态运动本原（DMP）的参考。该影响点捕获任务的空间结构，实现了对标准DMP表达式的扩展，使其超越了起始点和目标点的模仿。推断出的框通过视图-语言模型进行语义化接地，并通过Grounded-SAM在新颖场景中进行定位，从而实现功能一致的技能泛化。我们在仿真中验证了TReF-6，并展示了其对轨迹噪声的强大鲁棒性。我们进一步部署了一个端到端的流水线在实际的拾放任务中，展示了TReF-6支持一针见血的模仿学习，从而在不同的物体配置下保持任务意图。 

---
# Learn from What We HAVE: History-Aware VErifier that Reasons about Past Interactions Online 

**Title (ZH)**: 从前有据可循：在线推理过往交互的历史意识验证器 

**Authors**: Yishu Li, Xinyi Mao, Ying Yuan, Kyutae Sim, Ben Eisner, David Held  

**Link**: [PDF](https://arxiv.org/pdf/2509.00271)  

**Abstract**: We introduce a novel History-Aware VErifier (HAVE) to disambiguate uncertain scenarios online by leveraging past interactions. Robots frequently encounter visually ambiguous objects whose manipulation outcomes remain uncertain until physically interacted with. While generative models alone could theoretically adapt to such ambiguity, in practice they obtain suboptimal performance in ambiguous cases, even when conditioned on action history. To address this, we propose explicitly decoupling action generation from verification: we use an unconditional diffusion-based generator to propose multiple candidate actions and employ our history-aware verifier to select the most promising action by reasoning about past interactions. Through theoretical analysis, we demonstrate that employing a verifier significantly improves expected action quality. Empirical evaluations and analysis across multiple simulated and real-world environments including articulated objects, multi-modal doors, and uneven object pick-up confirm the effectiveness of our method and improvements over baselines. Our project website is available at: this https URL 

**Abstract (ZH)**: 我们提出了一种新的历史意识验证器（HAVEN），通过利用过往交互来在线消歧不确定性场景。机器人经常遇到视觉上具有歧义的物体，其操作结果在实际物理交互之前是不确定的。尽管生成模型理论上能够适应这种不确定性，在实际中，它们在具有歧义性的案例中性能不佳，即使考虑了动作历史。为了解决这一问题，我们提出显式地将动作生成与验证脱钩：我们使用一个无条件的扩散生成器提出多个候选动作，并利用我们的历史意识验证器通过推理过往交互来选择最有前途的动作。通过理论分析，我们证明了采用验证器能够显著提高预期动作质量。在多个模拟和真实世界环境中对具关节物体、多模态门和不规则物体拾取的实验评估与分析证实了我们方法的有效性及优于基线模型的改进。我们的项目网站可在以下链接访问：this https URL。 

---
# Embodied AI in Social Spaces: Responsible and Adaptive Robots in Complex Setting - UKAIRS 2025 (Copy) 

**Title (ZH)**: 具身AI在社会空间中的应用：复杂环境中负责任且适应性强的机器人-UKAIRS 2025 

**Authors**: Aleksandra Landowska, Aislinn D Gomez Bergin, Ayodeji O. Abioye, Jayati Deshmukh, Andriana Bouadouki, Maria Wheadon, Athina Georgara, Dominic Price, Tuyen Nguyen, Shuang Ao, Lokesh Singh, Yi Long, Raffaele Miele, Joel E. Fischer, Sarvapali D. Ramchurn  

**Link**: [PDF](https://arxiv.org/pdf/2509.00218)  

**Abstract**: This paper introduces and overviews a multidisciplinary project aimed at developing responsible and adaptive multi-human multi-robot (MHMR) systems for complex, dynamic settings. The project integrates co-design, ethical frameworks, and multimodal sensing to create AI-driven robots that are emotionally responsive, context-aware, and aligned with the needs of diverse users. We outline the project's vision, methodology, and early outcomes, demonstrating how embodied AI can support sustainable, ethical, and human-centred futures. 

**Abstract (ZH)**: 这篇论文介绍了旨在为复杂动态环境开发负责任且适应性强的多 humanoid 多机器人（MHMR）系统的跨学科项目。该项目结合了协同设计、伦理框架和多模态传感，创建出能够情绪响应、情境意识并符合多样化用户需求的 AI 驱动机器人。我们概述了该项目的愿景、方法论及其早期成果，展示了本体 AI 如何支持可持续、伦理且以人类为中心的未来。 

---
# Poke and Strike: Learning Task-Informed Exploration Policies 

**Title (ZH)**: 戳击策略：学习任务导向的探索策略 

**Authors**: Marina Y. Aoyama, Joao Moura, Juan Del Aguila Ferrandis, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.00178)  

**Abstract**: In many dynamic robotic tasks, such as striking pucks into a goal outside the reachable workspace, the robot must first identify the relevant physical properties of the object for successful task execution, as it is unable to recover from failure or retry without human intervention. To address this challenge, we propose a task-informed exploration approach, based on reinforcement learning, that trains an exploration policy using rewards automatically generated from the sensitivity of a privileged task policy to errors in estimated properties. We also introduce an uncertainty-based mechanism to determine when to transition from exploration to task execution, ensuring sufficient property estimation accuracy with minimal exploration time. Our method achieves a 90% success rate on the striking task with an average exploration time under 1.2 seconds, significantly outperforming baselines that achieve at most 40% success or require inefficient querying and retraining in a simulator at test time. Additionally, we demonstrate that our task-informed rewards capture the relative importance of physical properties in both the striking task and the classical CartPole example. Finally, we validate our approach by demonstrating its ability to identify object properties and adjust task execution in a physical setup using the KUKA iiwa robot arm. 

**Abstract (ZH)**: 在许多动态机器人任务中，如将冰球击入远离可达工作空间的目标门，机器人必须首先识别出与任务成功执行相关的物理特性，因为它无法在没有人类干预的情况下从失败中恢复或重试。为解决这一挑战，我们提出了一种基于强化学习的任务导向探索方法，通过使用来自优先级任务策略对估计特性误差敏感性的自动奖励来训练探索策略。同时，我们引入了一种基于不确定性机制来确定何时从探索过渡到任务执行的方法，确保在最小的探索时间内获得足够的特性估计精度。该方法在打击任务中实现了90%的成功率，平均探索时间不到1.2秒，显著优于仅能达到40%成功率或在测试时需要在模拟器中进行低效查询和重新训练的基线方法。此外，我们展示了任务导向的奖励能够捕捉打击任务和经典的CartPole示例中物理特性的相对重要性。最后，通过在物理设置中使用KUKA iiwa机器人臂验证该方法的能力，展示了其识别物体特性和调整任务执行的能力。 

---
# Robotic Fire Risk Detection based on Dynamic Knowledge Graph Reasoning: An LLM-Driven Approach with Graph Chain-of-Thought 

**Title (ZH)**: 基于动态知识图谱推理的机器人火灾风险检测：一种以图链式思考为驱动的LLM驱动方法 

**Authors**: Haimei Pan, Jiyun Zhang, Qinxi Wei, Xiongnan Jin, Chen Xinkai, Jie Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.00054)  

**Abstract**: Fire is a highly destructive disaster, but effective prevention can significantly reduce its likelihood of occurrence. When it happens, deploying emergency robots in fire-risk scenarios can help minimize the danger to human responders. However, current research on pre-disaster warnings and disaster-time rescue still faces significant challenges due to incomplete perception, inadequate fire situational awareness, and delayed response. To enhance intelligent perception and response planning for robots in fire scenarios, we first construct a knowledge graph (KG) by leveraging large language models (LLMs) to integrate fire domain knowledge derived from fire prevention guidelines and fire rescue task information from robotic emergency response documents. We then propose a new framework called Insights-on-Graph (IOG), which integrates the structured fire information of KG and Large Multimodal Models (LMMs). The framework generates perception-driven risk graphs from real-time scene imagery to enable early fire risk detection and provide interpretable emergency responses for task module and robot component configuration based on the evolving risk situation. Extensive simulations and real-world experiments show that IOG has good applicability and practical application value in fire risk detection and rescue decision-making. 

**Abstract (ZH)**: 火灾是一种高度破坏性的灾害，但有效的预防可以显著降低其发生几率。一旦发生，部署应急机器人在火灾风险场景中可以帮助减轻对救援人员的危险。然而，由于感知不完整、火灾态势感知不足以及应对延迟，当前关于灾难前预警和灾难时救援的研究仍面临重大挑战。为了增强机器人在火灾场景中的智能感知和应对规划，我们首先利用大型语言模型构建知识图谱（KG），整合来自防火指南和应急救援任务文档的火灾领域知识。然后，我们提出了一种名为Graph上的洞察（IOG）的新框架，该框架将KG中的结构化火灾信息与大型多模态模型（LMMs）集成。该框架从实时场景图像中生成感知驱动的风险图，以实现早期火灾风险检测，并为任务模块和机器人组件配置提供基于演变风险情况的解释性应急响应。广泛的仿真和实地实验表明，IOG在火灾风险检测和救援决策方面具有良好的适用性和实用性。 

---
# End-to-End Low-Level Neural Control of an Industrial-Grade 6D Magnetic Levitation System 

**Title (ZH)**: 端到端低级神经控制的工业级6D磁悬浮系统 

**Authors**: Philipp Hartmann, Jannick Stranghöner, Klaus Neumann  

**Link**: [PDF](https://arxiv.org/pdf/2509.01388)  

**Abstract**: Magnetic levitation is poised to revolutionize industrial automation by integrating flexible in-machine product transport and seamless manipulation. It is expected to become the standard drive for automated manufacturing. However, controlling such systems is inherently challenging due to their complex, unstable dynamics. Traditional control approaches, which rely on hand-crafted control engineering, typically yield robust but conservative solutions, with their performance closely tied to the expertise of the engineering team. In contrast, neural control learning presents a promising alternative. This paper presents the first neural controller for 6D magnetic levitation. Trained end-to-end on interaction data from a proprietary controller, it directly maps raw sensor data and 6D reference poses to coil current commands. The neural controller can effectively generalize to previously unseen situations while maintaining accurate and robust control. These results underscore the practical feasibility of learning-based neural control in complex physical systems and suggest a future where such a paradigm could enhance or even substitute traditional engineering approaches in demanding real-world applications. The trained neural controller, source code, and demonstration videos are publicly available at this https URL. 

**Abstract (ZH)**: 磁悬浮技术有望通过集成灵活的在机产品传输和无缝操作来革命工业自动化，并成为自动化制造的标准驱动。然而，由于其复杂的不稳定动力学，控制此类系统本身具有挑战性。传统的控制方法依赖手工艺品design的控制工程，通常提供鲁棒但保守的解决方案，其性能紧密依赖于工程师团队的专业知识。相比之下，神经控制学习提供了一种有潜力的替代方案。本文介绍了首个用于6D磁悬浮的神经控制器。该控制器端对端地在专有控制器的交互数据上进行训练，直接将原始传感器数据和6D参考姿态映射为线圈电流命令。神经控制器能够有效地泛化到未见过的情况，同时保持准确和 robust 的控制。这些结果强调了基于学习的神经控制在复杂物理系统中的实用可行性，并暗示了一种未来，其中这种范式可以在要求严格的实际应用中增强甚至替代传统工程方法。训练好的神经控制器、源代码和演示视频可在以下链接公开获取：this https URL。 

---
# Robix: A Unified Model for Robot Interaction, Reasoning and Planning 

**Title (ZH)**: Robix：统一的机器人交互、推理和规划模型 

**Authors**: Huang Fang, Mengxi Zhang, Heng Dong, Wei Li, Zixuan Wang, Qifeng Zhang, Xueyun Tian, Yucheng Hu, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.01106)  

**Abstract**: We introduce Robix, a unified model that integrates robot reasoning, task planning, and natural language interaction within a single vision-language architecture. Acting as the high-level cognitive layer in a hierarchical robot system, Robix dynamically generates atomic commands for the low-level controller and verbal responses for human interaction, enabling robots to follow complex instructions, plan long-horizon tasks, and interact naturally with human within an end-to-end framework. Robix further introduces novel capabilities such as proactive dialogue, real-time interruption handling, and context-aware commonsense reasoning during task execution. At its core, Robix leverages chain-of-thought reasoning and adopts a three-stage training strategy: (1) continued pretraining to enhance foundational embodied reasoning abilities including 3D spatial understanding, visual grounding, and task-centric reasoning; (2) supervised finetuning to model human-robot interaction and task planning as a unified reasoning-action sequence; and (3) reinforcement learning to improve reasoning-action consistency and long-horizon task coherence. Extensive experiments demonstrate that Robix outperforms both open-source and commercial baselines (e.g., GPT-4o and Gemini 2.5 Pro) in interactive task execution, demonstrating strong generalization across diverse instruction types (e.g., open-ended, multi-stage, constrained, invalid, and interrupted) and various user-involved tasks such as table bussing, grocery shopping, and dietary filtering. 

**Abstract (ZH)**: Robix：统一的机器人推理、任务规划与自然语言交互模型 

---
# A Layered Control Perspective on Legged Locomotion: Embedding Reduced Order Models via Hybrid Zero Dynamics 

**Title (ZH)**: 基于分层控制视角的足式运动控制：通过混合零动力学嵌入降阶模型 

**Authors**: Sergio A. Esteban, Max H. Cohen, Adrian B. Ghansah, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2509.00294)  

**Abstract**: Reduced-order models (ROMs) provide a powerful means of synthesizing dynamic walking gaits on legged robots. Yet this approach lacks the formal guarantees enjoyed by methods that utilize the full-order model (FOM) for gait synthesis, e.g., hybrid zero dynamics. This paper aims to unify these approaches through a layered control perspective. In particular, we establish conditions on when a ROM of locomotion yields stable walking on the full-order hybrid dynamics. To achieve this result, given an ROM we synthesize a zero dynamics manifold encoding the behavior of the ROM -- controllers can be synthesized that drive the FOM to this surface, yielding hybrid zero dynamics. We prove that a stable periodic orbit in the ROM implies an input-to-state stable periodic orbit of the FOM's hybrid zero dynamics, and hence the FOM dynamics. This result is demonstrated in simulation on a linear inverted pendulum ROM and a 5-link planar walking FOM. 

**Abstract (ZH)**: Reduced-order模型（ROMs）为合成有腿机器人的动态行走步态提供了一种强大的手段。然而，这种方法缺乏使用完整阶数模型（FOM）进行步态合成的方法所享有的正式保证，例如混合零动态。本文通过分层控制视角旨在统一这两种方法。具体而言，我们建立了当运动学ROM导致在完整阶数混合动力学中稳定行走时的条件。为了实现这一结果，给定一个ROM，我们合成了一个零动态流形，编码ROM的行为——可以合成控制器将FOM驱动到这个表面上，从而得到混合零动态。我们证明了ROM中的稳定周期轨道意味着FOM的混合零动态及其动力学的输入到状态稳定的周期轨道，这一结果在仿真的线性倒摆ROM和5连杆平面行走FOM上得到了验证。 

---
# Embodied AI: Emerging Risks and Opportunities for Policy Action 

**Title (ZH)**: 具身人工智能：政策行动 emerging risks and opportunities for policy action 

**Authors**: Jared Perlo, Alexander Robey, Fazl Barez, Luciano Floridi, Jakob Mökander  

**Link**: [PDF](https://arxiv.org/pdf/2509.00117)  

**Abstract**: The field of embodied AI (EAI) is rapidly advancing. Unlike virtual AI, EAI can exist in, learn from, reason about, and act in the physical world. Given recent innovations in large language and multimodal models, along with increasingly advanced and responsive hardware, EAI systems are rapidly growing in capabilities and operational domains. These advances present significant risks, including physical harm from malicious use, mass surveillance, and economic and societal disruption. However, these risks have been severely overlooked by policymakers. Existing policies, such as international standards for industrial robots or statutes governing autonomous vehicles, are insufficient to address the full range of concerns. While lawmakers are increasingly focused on AI, there is now an urgent need to extend and adapt existing frameworks to account for the unique risks of EAI. To help bridge this gap, this paper makes three contributions: first, we provide a foundational taxonomy of key physical, informational, economic, and social EAI risks. Secondly, we analyze policies in the US, EU, and UK to identify how existing frameworks address these risks and where these policies leave critical gaps. We conclude by offering concrete policy recommendations to address the coming wave of EAI innovation, including mandatory testing and certification for EAI systems, clarified liability frameworks, and forward-looking strategies to manage and prepare for transformative economic and societal impacts. 

**Abstract (ZH)**: embodied AI领域正迅速发展。与虚拟AI不同，embodied AI可以在物理世界中存在、学习、推理和行动。随着大型语言模型和多模态模型的近期创新，以及日益先进的响应式硬件的发展，embodied AI系统的功能和操作领域正迅速扩大。这些进步带来了严重的风险，包括恶意使用带来的物理伤害、大规模 surveillance、以及经济和社会的颠覆。然而，这些风险已被政策制定者严重忽视。现有的政策，如国际工业机器人标准或自动驾驶汽车法规，不足以全面应对各种关切。尽管立法者越来越关注AI，现在迫切需要扩展和适应现有的框架，以考虑到embodied AI的独特风险。为了帮助弥合这一差距，本文作出三项贡献：首先，我们提供了一种基础的embodied AI风险分类，涵盖物理、信息、经济和社会方面的关键风险。其次，我们分析了美国、欧盟和英国的政策，以确定这些现有框架如何应对这些风险、以及政策中的关键缺口在哪里。最后，我们提出了具体政策建议，以应对即将到来的embodied AI创新浪潮，包括对embodied AI系统的强制性测试和认证、明确的法律责任框架，以及面向未来的策略以管理和准备经济和社会的重大变革。 

---
# UI-TARS-2 Technical Report: Advancing GUI Agent with Multi-Turn Reinforcement Learning 

**Title (ZH)**: UI-TARS-2 技术报告：基于多轮强化学习的GUI代理提升 

**Authors**: Haoming Wang, Haoyang Zou, Huatong Song, Jiazhan Feng, Junjie Fang, Junting Lu, Longxiang Liu, Qinyu Luo, Shihao Liang, Shijue Huang, Wanjun Zhong, Yining Ye, Yujia Qin, Yuwen Xiong, Yuxin Song, Zhiyong Wu, Bo Li, Chen Dun, Chong Liu, Fuxing Leng, Hanbin Wang, Hao Yu, Haobin Chen, Hongyi Guo, Jing Su, Jingjia Huang, Kai Shen, Kaiyu Shi, Lin Yan, Peiyao Zhao, Pengfei Liu, Qinghao Ye, Renjie Zheng, Wayne Xin Zhao, Wen Heng, Wenhao Huang, Wenqian Wang, Xiaobo Qin, Yi Lin, Youbin Wu, Zehui Chen, Zihao Wang, Baoquan Zhong, Xinchun Zhang, Xujing Li, Yuanfan Li, Zhongkai Zhao, Chengquan Jiang, Faming Wu, Haotian Zhou, Jinlin Pang, Li Han, Qianli Ma, Siyao Liu, Songhua Cai, Wenqi Fu, Xin Liu, Zhi Zhang, Bo Zhou, Guoliang Li, Jiajun Shi, Jiale Yang, Jie Tang, Li Li, Taoran Lu, Woyu Lin, Xiaokang Tong, Xinyao Li, Yichi Zhang, Yu Miao, Zhengxuan Jiang, Zili Li, Ziyuan Zhao, Chenxin Li, Dehua Ma, Feng Lin, Ge Zhang, Haihua Yang, Hangyu Guo, Hongda Zhu, Jiaheng Liu, Junda Du, Kai Cai, Kuanye Li, Lichen Yuan, Meilan Han, Minchao Wang, Shuyue Guo, Tianhao Cheng, Xiaobo Ma, Xiaojun Xiao, Xiaolong Huang, Xinjie Chen, Yidi Du, Yilin Chen, Yiwen Wang, Zhaojian Li, Zhenzhu Yang, Zhiyuan Zeng, Chaolin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.02544)  

**Abstract**: The development of autonomous agents for graphical user interfaces (GUIs) presents major challenges in artificial intelligence. While recent advances in native agent models have shown promise by unifying perception, reasoning, action, and memory through end-to-end learning, open problems remain in data scalability, multi-turn reinforcement learning (RL), the limitations of GUI-only operation, and environment stability. In this technical report, we present UI-TARS-2, a native GUI-centered agent model that addresses these challenges through a systematic training methodology: a data flywheel for scalable data generation, a stabilized multi-turn RL framework, a hybrid GUI environment that integrates file systems and terminals, and a unified sandbox platform for large-scale rollouts. Empirical evaluation demonstrates that UI-TARS-2 achieves significant improvements over its predecessor UI-TARS-1.5. On GUI benchmarks, it reaches 88.2 on Online-Mind2Web, 47.5 on OSWorld, 50.6 on WindowsAgentArena, and 73.3 on AndroidWorld, outperforming strong baselines such as Claude and OpenAI agents. In game environments, it attains a mean normalized score of 59.8 across a 15-game suite-roughly 60% of human-level performance-and remains competitive with frontier proprietary models (e.g., OpenAI o3) on LMGame-Bench. Additionally, the model can generalize to long-horizon information-seeking tasks and software engineering benchmarks, highlighting its robustness across diverse agent tasks. Detailed analyses of training dynamics further provide insights into achieving stability and efficiency in large-scale agent RL. These results underscore UI-TARS-2's potential to advance the state of GUI agents and exhibit strong generalization to real-world interactive scenarios. 

**Abstract (ZH)**: 自主图形用户界面（GUI）代理的发展在人工智能领域提出了重大挑战。虽然近期原生代理模型的进步通过端到端学习综合了感知、推理、行动和记忆，但仍存在数据扩展性、多轮强化学习（RL）开放问题、GUI专有操作的限制以及环境稳定性等问题。在本技术报告中，我们提出了UI-TARS-2，这是一种以GUI为中心的原生代理模型，通过系统化的训练方法解决了这些挑战：数据飞轮实现数据生成的扩展性，稳定化的多轮RL框架，结合文件系统和终端的混合GUI环境，以及支持大规模部署的统一沙盒平台。实证评估证明UI-TARS-2相较于其前驱UI-TARS-1.5取得了显著改进。在GUI基准测试中，UI-TARS-2达到Online-Mind2Web 88.2、OSWorld 47.5、WindowsAgentArena 50.6和AndroidWorld 73.3的分数，超过了Claude和OpenAI等强基线。在游戏环境中，平均标准化得分达到59.8（约60%的人类水平表现），并在LMGame-Bench上保持与前沿专有模型（如OpenAI o3）的竞争性。此外，该模型能够泛化到长期信息检索任务和软件工程基准测试，表明其在不同代理任务中的鲁棒性。详细的训练动态分析进一步提供了实现大规模代理RL的稳定性和效率的关键见解。这些结果凸显了UI-TARS-2在推进GUI代理状态和发展至真实交互场景中的巨大潜力。 

---
# An Epidemiological Knowledge Graph extracted from the World Health Organization's Disease Outbreak News 

**Title (ZH)**: 世界卫生组织疾病暴发新闻中提取的流行病学知识图谱 

**Authors**: Sergio Consoli, Pietro Coletti, Peter V. Markov, Lia Orfei, Indaco Biazzo, Lea Schuh, Nicolas Stefanovitch, Lorenzo Bertolini, Mario Ceresa, Nikolaos I. Stilianakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.02258)  

**Abstract**: The rapid evolution of artificial intelligence (AI), together with the increased availability of social media and news for epidemiological surveillance, are marking a pivotal moment in epidemiology and public health research. Leveraging the power of generative AI, we use an ensemble approach which incorporates multiple Large Language Models (LLMs) to extract valuable actionable epidemiological information from the World Health Organization (WHO) Disease Outbreak News (DONs). DONs is a collection of regular reports on global outbreaks curated by the WHO and the adopted decision-making processes to respond to them. The extracted information is made available in a daily-updated dataset and a knowledge graph, referred to as eKG, derived to provide a nuanced representation of the public health domain knowledge. We provide an overview of this new dataset and describe the structure of eKG, along with the services and tools used to access and utilize the data that we are building on top. These innovative data resources open altogether new opportunities for epidemiological research, and the analysis and surveillance of disease outbreaks. 

**Abstract (ZH)**: 人工 Intelligence的迅猛发展以及社交媒体和新闻在流行病 surveillance 中的可用性增加，标志着流行病学和公共卫生研究的一个关键时期。利用生成式 AI 的强大功能，我们采用集成方法，结合多个大型语言模型（LLMs），从世界卫生组织（WHO）疾病爆发新闻（DONs）中提取有价值的可操作流行病学信息。DONs 是由 WHO 编纂的全球爆发的定期报告，并包含了相应决策过程。提取的信息在每日更新的数据集中和用于提供公共卫生领域知识精细表示的知识图谱（eKG）中提供。我们概述了这一新数据集，并描述了 eKG 的结构以及用于访问和利用数据的服务和工具。这些创新的数据资源为流行病学研究、疾病爆发的分析和 surveillance 打开了全新的机会。 

---
# Throttling Web Agents Using Reasoning Gates 

**Title (ZH)**: 使用推理门控限制网络代理 

**Authors**: Abhinav Kumar, Jaechul Roh, Ali Naseh, Amir Houmansadr, Eugene Bagdasarian  

**Link**: [PDF](https://arxiv.org/pdf/2509.01619)  

**Abstract**: AI web agents use Internet resources at far greater speed, scale, and complexity -- changing how users and services interact. Deployed maliciously or erroneously, these agents could overload content providers. At the same time, web agents can bypass CAPTCHAs and other defenses by mimicking user behavior or flood authentication systems with fake accounts. Yet providers must protect their services and content from denial-of-service attacks and scraping by web agents. In this paper, we design a framework that imposes tunable costs on agents before providing access to resources; we call this Web Agent Throttling. We start by formalizing Throttling Gates as challenges issued to an agent that are asymmetric, scalable, robust, and compatible with any agent. Focusing on a common component -- the language model -- we require the agent to solve reasoning puzzles, thereby incurring excessive token-generation costs. However, we find that using existing puzzles, e.g., coding or math, as throttling gates fails to satisfy our properties. To address this, we introduce rebus-based Reasoning Gates, synthetic text puzzles that require multi-hop reasoning over world knowledge (thereby throttling an agent's model). We design a scalable generation and verification protocol for such reasoning gates. Our framework achieves computational asymmetry, i.e., the response-generation cost is 9.2x higher than the generation cost for SOTA models. We further deploy reasoning gates on a custom website and Model Context Protocol (MCP) servers and evaluate with real-world web agents. Finally, we discuss the limitations and environmental impact of real-world deployment of our framework. 

**Abstract (ZH)**: 基于Web代理的可调节成本框架：Web代理限速 

---
# VerlTool: Towards Holistic Agentic Reinforcement Learning with Tool Use 

**Title (ZH)**: VerlTool：朝向综合代理强化学习中的工具使用 

**Authors**: Dongfu Jiang, Yi Lu, Zhuofeng Li, Zhiheng Lyu, Ping Nie, Haozhe Wang, Alex Su, Hui Chen, Kai Zou, Chao Du, Tianyu Pang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.01055)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has demonstrated success in enhancing LLM reasoning capabilities, but remains limited to single-turn interactions without tool integration. While recent Agentic Reinforcement Learning with Tool use (ARLT) approaches have emerged to address multi-turn tool interactions, existing works develop task-specific codebases that suffer from fragmentation, synchronous execution bottlenecks, and limited extensibility across domains. These inefficiencies hinder broader community adoption and algorithmic innovation. We introduce VerlTool, a unified and modular framework that addresses these limitations through systematic design principles. VerlTool provides four key contributions: (1) upstream alignment with VeRL ensuring compatibility and simplified maintenance, (2) unified tool management via standardized APIs supporting diverse modalities including code execution, search, SQL databases, and vision processing, (3) asynchronous rollout execution achieving near 2$\times$ speedup by eliminating synchronization bottlenecks, and (4) comprehensive evaluation demonstrating competitive performance across 6 ARLT domains. Our framework formalizes ARLT as multi-turn trajectories with multi-modal observation tokens (text/image/video), extending beyond single-turn RLVR paradigms. We train and evaluate models on mathematical reasoning, knowledge QA, SQL generation, visual reasoning, web search, and software engineering tasks, achieving results comparable to specialized systems while providing unified training infrastructure. The modular plugin architecture enables rapid tool integration requiring only lightweight Python definitions, significantly reducing development overhead and providing a scalable foundation for tool-augmented RL research. Our code is open-sourced at this https URL. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）在提升大语言模型推理能力方面取得了成功，但仍然局限于单轮交互且不集成工具。虽然最近出现了针对多轮工具交互的代理强化学习与工具使用（ARLT）方法，但现有工作开发了特定任务的代码库，存在碎片化、同步执行瓶颈以及跨领域有限扩展性等问题。这些低效性阻碍了更广泛社区的采用和算法创新。我们引入了VerlTool，这是一种通过系统设计原则解决这些问题的统一且模块化的框架。VerlTool 提供四个关键贡献：（1）与VeRL的上游对齐确保兼容性和简化维护，（2）通过标准化API统一管理工具支持包括代码执行、搜索、SQL数据库和视觉处理在内的多种模态，（3）异步回退执行实现近2倍的速度提升并通过消除同步瓶颈，（4）全面评估在6个ARLT领域展示出竞争力的性能。我们的框架将ARLT形式化为多轮轨迹，带有多种模态观察令牌（文本/图像/视频），超越了单轮RLVR范式。我们在数学推理、知识问答、SQL生成、视觉推理、网络搜索和软件工程任务上训练和评估模型，结果可与专门系统媲美，同时提供统一的训练基础设施。模块化的插件架构允许快速集成工具，仅需轻量级Python定义，显著降低开发开销，并为工具增强的RL研究提供可扩展的基础。我们的代码开源于此。 

---
# Social World Models 

**Title (ZH)**: 社会世界模型 

**Authors**: Xuhui Zhou, Jiarui Liu, Akhila Yerukola, Hyunwoo Kim, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2509.00559)  

**Abstract**: Humans intuitively navigate social interactions by simulating unspoken dynamics and reasoning about others' perspectives, even with limited information. In contrast, AI systems struggle to automatically structure and reason about these implicit social contexts. In this paper, we introduce a novel structured social world representation formalism (S3AP), designed to help AI systems reason more effectively about social dynamics. Following a POMDP-driven design, S3AP represents social interactions as structured tuples, such as state, observation, agent actions, and mental states, which can be automatically induced from free-form narratives or other inputs. We first show S3AP can help LLMs better understand social narratives across 5 social reasoning tasks (e.g., +51% improvement on FANToM's theory-of-mind reasoning with OpenAI's o1), reaching new state-of-the-art (SOTA) performance. We then induce social world models from these structured representations, demonstrating their ability to predict future social dynamics and improve agent decision-making, yielding up to +18% improvement on the SOTOPIA social interaction benchmark. Our findings highlight the promise of S3AP as a powerful, general-purpose representation for social world states, enabling the development of more socially-aware systems that better navigate social interactions. 

**Abstract (ZH)**: 人类通过模拟未言明的社会动态和推理他人的视角来直观地导航社会互动，即使是在信息有限的情况下。相比之下，AI系统难以自动结构化和推理这些隐含的社会情境。本文介绍了一种新的结构化社会世界表示形式（S3AP），旨在帮助AI系统更有效地推理社会动态。S3AP采用POMDP驱动的设计，将社会互动表示为结构化的元组，如状态、观察、代理行动和心理状态，这些元组可以从自由形式的叙述或其他输入中自动推导出来。我们首先展示了S3AP可以帮助LLMs更好地理解社会叙述，在5个社会推理任务中取得显著提升（例如，OpenAI的o1在FANToM的心理理论推理任务中提升了51%），达到新的最先进（SOTA）性能。然后，我们从这些结构化表示中推导出社会世界模型，证明了它们预测未来社会动态和改善代理决策的能力，在SOTOPIA社会互动基准测试中取得了最高达18%的提升。研究结果突显了S3AP作为社会世界状态的强大、通用表示形式的潜力，有助于开发更加社会意识强的系统，更好地导航社会互动。 

---
# HiVA: Self-organized Hierarchical Variable Agent via Goal-driven Semantic-Topological Evolution 

**Title (ZH)**: HiVA：目标驱动的语义-拓扑演化的自我组织层次变体代理 

**Authors**: Jinzhou Tang, Jusheng Zhang, Qinhan Lv, Sidi Liu, Jing Yang, Chengpei Tang, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00189)  

**Abstract**: Autonomous agents play a crucial role in advancing Artificial General Intelligence, enabling problem decomposition and tool orchestration through Large Language Models (LLMs). However, existing paradigms face a critical trade-off. On one hand, reusable fixed workflows require manual reconfiguration upon environmental changes; on the other hand, flexible reactive loops fail to distill reasoning progress into transferable structures. We introduce Hierarchical Variable Agent (HiVA), a novel framework modeling agentic workflows as self-organized graphs with the Semantic-Topological Evolution (STEV) algorithm, which optimizes hybrid semantic-topological spaces using textual gradients as discrete-domain surrogates for backpropagation. The iterative process comprises Multi-Armed Bandit-infused forward routing, diagnostic gradient generation from environmental feedback, and coordinated updates that co-evolve individual semantics and topology for collective optimization in unknown environments. Experiments on dialogue, coding, Long-context Q&A, mathematical, and agentic benchmarks demonstrate improvements of 5-10% in task accuracy and enhanced resource efficiency over existing baselines, establishing HiVA's effectiveness in autonomous task execution. 

**Abstract (ZH)**: 自主代理在推进人工通用智能中发挥着关键作用，通过大规模语言模型（LLMs）实现问题分解和工具 orchestrated。然而，现有范式面临一个关键权衡。一方面，可重复的固定工作流程在环境变化时需要手动重新配置；另一方面，灵活的反应循环无法将推理进展提炼为可转移的结构。我们引入了层次可变代理（HiVA），这是一种新颖的框架，通过语义-拓扑演化（STEV）算法将代理工作流程建模为自组织图，该算法利用文本梯度优化混合语义-拓扑空间，将其作为反向传播的离散域代理。迭代过程包括多臂bandit启发式的前向路由、从环境反馈生成诊断梯度以及协调更新，以协同进化个体语义和拓扑结构，实现未知环境中的联合优化。HiVA在对话、编程、长上下文问答、数学和代理基准测试中的实验表明，与现有基线相比，其在任务准确性和资源效率上分别提高了5-10%，证明了HiVA在自主任务执行中的有效性。 

---
# Adaptive Monitoring and Real-World Evaluation of Agentic AI Systems 

**Title (ZH)**: 适应性监测与实际应用评价中的代理型AI系统 

**Authors**: Manish Shukla  

**Link**: [PDF](https://arxiv.org/pdf/2509.00115)  

**Abstract**: Agentic artificial intelligence (AI) -- multi-agent systems that combine large language models with external tools and autonomous planning -- are rapidly transitioning from research laboratories into high-stakes domains. Our earlier "Basic" paper introduced a five-axis framework and proposed preliminary metrics such as goal drift and harm reduction but did not provide an algorithmic instantiation or empirical evidence. This "Advanced" sequel fills that gap. First, we revisit recent benchmarks and industrial deployments to show that technical metrics still dominate evaluations: a systematic review of 84 papers from 2023--2025 found that 83% report capability metrics while only 30% consider human-centred or economic axes [2]. Second, we formalise an Adaptive Multi-Dimensional Monitoring (AMDM) algorithm that normalises heterogeneous metrics, applies per-axis exponentially weighted moving-average thresholds and performs joint anomaly detection via the Mahalanobis distance. Third, we conduct simulations and real-world experiments. AMDM cuts anomaly-detection latency from 12.3 s to 5.6 s on simulated goal drift and reduces false-positive rates from 4.5% to 0.9% compared with static thresholds. We present a comparison table and ROC/PR curves, and we reanalyse case studies to surface missing metrics. Code, data and a reproducibility checklist accompany this paper to facilitate replication. 

**Abstract (ZH)**: 代理型人工智能（AGENTIC AI）——结合大型语言模型、外部工具和自主规划的多智能体系统——正迅速从研究实验室过渡到高风险领域。我们的早期“基础”论文引入了一个五轴框架并提出了初步指标如目标漂移和危害降低，但未提供算法实现或实证证据。本“进阶”续作填补了这一空白。首先，我们回顾了最近的基准测试和工业部署，展示了技术指标仍主导评估：系统性审查84篇2023年至2025年发布的论文发现，83%的论文报告了能力指标，只有30%考虑了以人为中心或经济维度轴[2]。其次，我们形式化了一个适应性多维度监控（AMDM）算法，该算法对异构指标进行标准化处理，针对每个维度应用指数加权移动平均阈值，并通过马哈拉诺比斯距离进行联合异常检测。第三，我们进行了模拟和实际实验。AMDM将模拟目标漂移的异常检测延迟从12.3秒降至5.6秒，将误报率从4.5%降至0.9%，优于静态阈值。我们提供了对比表格和ROC/PR曲线，并重新分析了案例研究以揭示缺失的指标。本文随附代码、数据和可重复性检查表，以促进复制。 

---
# Wrong Face, Wrong Move: The Social Dynamics of Emotion Misperception in Agent-Based Models 

**Title (ZH)**: 错误的脸部表达，错误的行动：基于代理的模型中情绪错认的社会动态 

**Authors**: David Freire-Obregón  

**Link**: [PDF](https://arxiv.org/pdf/2509.00080)  

**Abstract**: The ability of humans to detect and respond to others' emotions is fundamental to understanding social behavior. Here, agents are instantiated with emotion classifiers of varying accuracy to study the impact of perceptual accuracy on emergent emotional and spatial behavior. Agents are visually represented with face photos from the KDEF database and endowed with one of three classifiers trained on the JAFFE (poor), CK+ (medium), or KDEF (high) datasets. Agents communicate locally on a 2D toroidal lattice, perceiving neighbors' emotional state based on their classifier and responding with movement toward perceived positive emotions and away from perceived negative emotions. Note that the agents respond to perceived, instead of ground-truth, emotions, introducing systematic misperception and frustration. A battery of experiments is carried out on homogeneous and heterogeneous populations and scenarios with repeated emotional shocks. Results show that low-accuracy classifiers on the part of the agent reliably result in diminished trust, emotional disintegration into sadness, and disordered social organization. By contrast, the agent that develops high accuracy develops hardy emotional clusters and resilience to emotional disruptions. Even in emotionally neutral scenarios, misperception is enough to generate segregation and disintegration of cohesion. These findings underscore the fact that biases or imprecision in emotion recognition may significantly warp social processes and disrupt emotional integration. 

**Abstract (ZH)**: 人类检测和应对他人情绪的能力是理解社会行为的基础。本研究通过赋予代理不同准确度的情感分类器，研究感知准确度对涌现情感和社会行为的影响。代理通过KDEF数据库中的面部照片可视化，并配备了分别训练于JAFFE（低）、CK+（中）或KDEF（高）数据集的情感分类器。代理在二维环形格 lattice 上进行局部通信，基于分类器感知邻居的情感状态，并朝感知到的积极情绪移动，远离感知到的消极情绪。值得注意的是，代理响应的是感知到的情绪而非真实情绪，从而引入了系统性误感知和挫败感。通过对同质和异质人群和具有重复情感冲击的情景进行了实验，结果表明，代理情感分类器准确度低会导致信任减弱、情感瓦解为悲伤情绪以及社会组织紊乱。相反，发展高准确度的情感分类器的代理能够形成坚韧的情感聚类并具备对情绪干扰的抵抗力。即使在情绪中立的情景下，误感知也足以生成隔离和凝聚力的瓦解。这些发现强调了情感识别中的偏见或不精确性可能会显著扭曲社会过程并扰乱情感整合。 

---
# Language and Experience: A Computational Model of Social Learning in Complex Tasks 

**Title (ZH)**: 语言与经验：复杂任务中社会学习的计算模型 

**Authors**: Cédric Colas, Tracey Mills, Ben Prystawski, Michael Henry Tessler, Noah Goodman, Jacob Andreas, Joshua Tenenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2509.00074)  

**Abstract**: The ability to combine linguistic guidance from others with direct experience is central to human development, enabling safe and rapid learning in new environments. How do people integrate these two sources of knowledge, and how might AI systems? We present a computational framework that models social learning as joint probabilistic inference over structured, executable world models given sensorimotor and linguistic data. We make this possible by turning a pretrained language model into a probabilistic model of how humans share advice conditioned on their beliefs, allowing our agents both to generate advice for others and to interpret linguistic input as evidence during Bayesian inference. Using behavioral experiments and simulations across 10 video games, we show how linguistic guidance can shape exploration and accelerate learning by reducing risky interactions and speeding up key discoveries in both humans and models. We further explore how knowledge can accumulate across generations through iterated learning experiments and demonstrate successful knowledge transfer between humans and models -- revealing how structured, language-compatible representations might enable human-machine collaborative learning. 

**Abstract (ZH)**: 人类从他人语言指导与直接经验中整合知识的能力是人类发展中的核心，使人们能在新的环境中安全而快速地学习。人们是如何整合这两种知识来源的？AI系统又该如何实现这一点？我们提出了一种计算框架，将社会学习建模为基于传感器数据和语言数据的结构化可执行世界观的联合概率推断。通过将预训练的语言模型转化为一种基于人类信念分享建议的概率模型，我们的代理既可以为他人生成建议，又能在贝叶斯推理过程中将语言输入解释为证据。通过在10款视频游戏中进行行为实验和模拟，我们展示了语言指导如何塑造探索行为并加速学习，通过减少风险互动和加速关键发现，提升人类和模型的学习效率。此外，我们通过迭代学习实验探索了知识如何代际积累，并展示了人类与模型之间的知识传递成功案例——揭示了结构化且语言兼容的表现形式可能如何促进人机协同学习。 

---
# Think2Sing: Orchestrating Structured Motion Subtitles for Singing-Driven 3D Head Animation 

**Title (ZH)**: Think2Sing: 组织结构化动作字幕以实现歌声驱动的3D头部动画 

**Authors**: Zikai Huang, Yihan Zhou, Xuemiao Xu, Cheng Xu, Xiaofen Xing, Jing Qin, Shengfeng He  

**Link**: [PDF](https://arxiv.org/pdf/2509.02278)  

**Abstract**: Singing-driven 3D head animation is a challenging yet promising task with applications in virtual avatars, entertainment, and education. Unlike speech, singing involves richer emotional nuance, dynamic prosody, and lyric-based semantics, requiring the synthesis of fine-grained, temporally coherent facial motion. Existing speech-driven approaches often produce oversimplified, emotionally flat, and semantically inconsistent results, which are insufficient for singing animation. To address this, we propose Think2Sing, a diffusion-based framework that leverages pretrained large language models to generate semantically coherent and temporally consistent 3D head animations, conditioned on both lyrics and acoustics. A key innovation is the introduction of motion subtitles, an auxiliary semantic representation derived through a novel Singing Chain-of-Thought reasoning process combined with acoustic-guided retrieval. These subtitles contain precise timestamps and region-specific motion descriptions, serving as interpretable motion priors. We frame the task as a motion intensity prediction problem, enabling finer control over facial regions and improving the modeling of expressive motion. To support this, we create a multimodal singing dataset with synchronized video, acoustic descriptors, and motion subtitles, enabling diverse and expressive motion learning. Extensive experiments show that Think2Sing outperforms state-of-the-art methods in realism, expressiveness, and emotional fidelity, while also offering flexible, user-controllable animation editing. 

**Abstract (ZH)**: 基于歌声驱动的3D头部动画是具有虚拟角色、娱乐和教育应用的一种具有挑战性但前景广阔的任务。现有的语音驱动方法通常会产生过于简化、情感平淡且语义不一致的结果，不足以用于歌声动画。为此，我们提出Think2Sing，这是一种基于扩散的框架，利用预训练的大语言模型生成both歌词和声学特征条件下语义一致且时间一致的3D头部动画。一个关键技术创新是引入了动捕字幕，这是一种通过新颖的歌唱推理过程结合声学引导检索得到的辅助语义表示。这些字幕包含精确的时间戳和区域特定的动作描述，作为可解释的动作先验。我们将任务定义为运动强度预测问题，以实现对面部区域的细粒度控制，改善了表达性运动的建模。为此，我们创建了一个多模态歌唱数据集，包含同步视频、声学描述和动捕字幕，促进了多样且具有表现力的运动学习。广泛实验表明，Think2Sing在逼真度、表达性和情感忠实度方面优于现有方法，同时提供了灵活且用户可控的动画编辑能力。 

---
# Learning Social Heuristics for Human-Aware Path Planning 

**Title (ZH)**: 基于社会启发式的面向人类路径规划学习 

**Authors**: Andrea Eirale, Matteo Leonetti, Marcello Chiaberge  

**Link**: [PDF](https://arxiv.org/pdf/2509.02134)  

**Abstract**: Social robotic navigation has been at the center of numerous studies in recent years. Most of the research has focused on driving the robotic agent along obstacle-free trajectories, respecting social distances from humans, and predicting their movements to optimize navigation. However, in order to really be socially accepted, the robots must be able to attain certain social norms that cannot arise from conventional navigation, but require a dedicated learning process. We propose Heuristic Planning with Learned Social Value (HPLSV), a method to learn a value function encapsulating the cost of social navigation, and use it as an additional heuristic in heuristic-search path planning. In this preliminary work, we apply the methodology to the common social scenario of joining a queue of people, with the intention of generalizing to further human activities. 

**Abstract (ZH)**: 社会机器人导航在近年来的研究中处于中心地位。大多数研究集中在引导机器人沿无障碍路径行驶，遵守与人类的社会距离，并预测人类的移动以优化导航。然而，为了真正被社会接受，机器人必须能够获得某些不能从传统导航中产生的社会规范，而这些规范需要专门的学习过程。我们提出了一种基于学习的社会价值的启发式规划方法（HPLSV），该方法学习一个包含社会导航成本的价值函数，并将其用作启发式搜索路径规划中的附加启发式函数。在本初步工作中，我们将该方法应用于人们常见的排队场景，意图进一步推广到更多的人类活动。 

---
# Goal-Conditioned Reinforcement Learning for Data-Driven Maritime Navigation 

**Title (ZH)**: 基于目标条件的强化学习在数据驱动的海事导航中 

**Authors**: Vaishnav Vaidheeswaran, Dilith Jayakody, Samruddhi Mulay, Anand Lo, Md Mahbub Alam, Gabriel Spadon  

**Link**: [PDF](https://arxiv.org/pdf/2509.01838)  

**Abstract**: Routing vessels through narrow and dynamic waterways is challenging due to changing environmental conditions and operational constraints. Existing vessel-routing studies typically fail to generalize across multiple origin-destination pairs and do not exploit large-scale, data-driven traffic graphs. In this paper, we propose a reinforcement learning solution for big maritime data that can learn to find a route across multiple origin-destination pairs while adapting to different hexagonal grid resolutions. Agents learn to select direction and speed under continuous observations in a multi-discrete action space. A reward function balances fuel efficiency, travel time, wind resistance, and route diversity, using an Automatic Identification System (AIS)-derived traffic graph with ERA5 wind fields. The approach is demonstrated in the Gulf of St. Lawrence, one of the largest estuaries in the world. We evaluate configurations that combine Proximal Policy Optimization with recurrent networks, invalid-action masking, and exploration strategies. Our experiments demonstrate that action masking yields a clear improvement in policy performance and that supplementing penalty-only feedback with positive shaping rewards produces additional gains. 

**Abstract (ZH)**: 基于强化学习的大规模海洋数据船舶航线规划：适应多出发-目的对并利用大规模数据驱动交通图的研究 

---
# Toward a Unified Benchmark and Taxonomy of Stochastic Environments 

**Title (ZH)**: 向统一的随机环境基准和分类标准迈进 

**Authors**: Aryan Amit Barsainyan, Jing Yu Lim, Dianbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01793)  

**Abstract**: Reinforcement Learning (RL) agents have achieved strong results on benchmarks such as Atari100k, yet they remain limited in robustness to real-world conditions. Model-Based RL approaches that rely on learned World Models often struggle in environments with true stochasticity and partial observability, despite their theoretical grounding in POMDPs. Current benchmarks rarely capture these challenges, focusing instead on deterministic or overly simplified settings, and the lack of a clear taxonomy of stochasticity further hampers systematic evaluation. To address this gap, we introduce STORI (STOchastic-ataRI), a benchmark that incorporates diverse stochastic effects and enables rigorous assessment of RL methods under varied forms of uncertainty. In addition, we propose a taxonomy of stochasticity in RL environments, providing a unified framework for analyzing and comparing approaches. 

**Abstract (ZH)**: STOchas틱-ataRI：一种纳入多样化随机效应的基准，用于在多种不确定性形式下对RL方法进行严谨评估，并提出RL环境中的随机性分类学。 

---
# It's-A-Me, Quantum Mario: Scalable Quantum Reinforcement Learning with Multi-Chip Ensembles 

**Title (ZH)**: It's-Me,量子马里奥：多芯片集成的可扩展量子强化学习 

**Authors**: Junghoon Justin Park, Huan-Hsin Tseng, Shinjae Yoo, Samuel Yen-Chi Chen, Jiook Cha  

**Link**: [PDF](https://arxiv.org/pdf/2509.00713)  

**Abstract**: Quantum reinforcement learning (QRL) promises compact function approximators with access to vast Hilbert spaces, but its practical progress is slowed by NISQ-era constraints such as limited qubits and noise accumulation. We introduce a multi-chip ensemble framework using multiple small Quantum Convolutional Neural Networks (QCNNs) to overcome these constraints. Our approach partitions complex, high-dimensional observations from the Super Mario Bros environment across independent quantum circuits, then classically aggregates their outputs within a Double Deep Q-Network (DDQN) framework. This modular architecture enables QRL in complex environments previously inaccessible to quantum agents, achieving superior performance and learning stability compared to classical baselines and single-chip quantum models. The multi-chip ensemble demonstrates enhanced scalability by reducing information loss from dimensionality reduction while remaining implementable on near-term quantum hardware, providing a practical pathway for applying QRL to real-world problems. 

**Abstract (ZH)**: 多芯片ensemble框架下的量子强化学习：克服NISQ时代限制并实现复杂环境中的优越性能与学习稳定性 

---
# NeuralSVCD for Efficient Swept Volume Collision Detection 

**Title (ZH)**: 神经网络法（NeuralSVCD）用于高效的扫掠体积碰撞检测 

**Authors**: Dongwon Son, Hojin Jung, Beomjoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.00499)  

**Abstract**: Robot manipulation in unstructured environments requires efficient and reliable Swept Volume Collision Detection (SVCD) for safe motion planning. Traditional discrete methods potentially miss collisions between these points, whereas SVCD continuously checks for collisions along the entire trajectory. Existing SVCD methods typically face a trade-off between efficiency and accuracy, limiting practical use. In this paper, we introduce NeuralSVCD, a novel neural encoder-decoder architecture tailored to overcome this trade-off. Our approach leverages shape locality and temporal locality through distributed geometric representations and temporal optimization. This enhances computational efficiency without sacrificing accuracy. Comprehensive experiments show that NeuralSVCD consistently outperforms existing state-of-the-art SVCD methods in terms of both collision detection accuracy and computational efficiency, demonstrating its robust applicability across diverse robotic manipulation scenarios. Code and videos are available at this https URL. 

**Abstract (ZH)**: 无结构环境下机器人操作需要高效可靠的包体积碰撞检测（SVCD）以实现安全的运动规划。神经SVCD：一种新型神经编码器-解码器架构以克服效率与准确性的权衡 

---
# First Order Model-Based RL through Decoupled Backpropagation 

**Title (ZH)**: 基于分解反向传播的一阶模型强化学习 

**Authors**: Joseph Amigo, Rooholla Khorrambakht, Elliot Chane-Sane, Nicolas Mansard, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2509.00215)  

**Abstract**: There is growing interest in reinforcement learning (RL) methods that leverage the simulator's derivatives to improve learning efficiency. While early gradient-based approaches have demonstrated superior performance compared to derivative-free methods, accessing simulator gradients is often impractical due to their implementation cost or unavailability. Model-based RL (MBRL) can approximate these gradients via learned dynamics models, but the solver efficiency suffers from compounding prediction errors during training rollouts, which can degrade policy performance. We propose an approach that decouples trajectory generation from gradient computation: trajectories are unrolled using a simulator, while gradients are computed via backpropagation through a learned differentiable model of the simulator. This hybrid design enables efficient and consistent first-order policy optimization, even when simulator gradients are unavailable, as well as learning a critic from simulation rollouts, which is more accurate. Our method achieves the sample efficiency and speed of specialized optimizers such as SHAC, while maintaining the generality of standard approaches like PPO and avoiding ill behaviors observed in other first-order MBRL methods. We empirically validate our algorithm on benchmark control tasks and demonstrate its effectiveness on a real Go2 quadruped robot, across both quadrupedal and bipedal locomotion tasks. 

**Abstract (ZH)**: 利用模拟器梯度增强学习效率的方法：解耦轨迹生成与梯度计算的模型导向强化学习 

---
# Beyond Pixels: Introducing Geometric-Semantic World Priors for Video-based Embodied Models via Spatio-temporal Alignment 

**Title (ZH)**: 超越像素：通过时空对齐引入几何语义先验的基于视频的体态模型 

**Authors**: Jinzhou Tang, Jusheng zhang, Sidi Liu, Waikit Xiu, Qinhan Lv, Xiying Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.00210)  

**Abstract**: Achieving human-like reasoning in deep learning models for complex tasks in unknown environments remains a critical challenge in embodied intelligence. While advanced vision-language models (VLMs) excel in static scene understanding, their limitations in spatio-temporal reasoning and adaptation to dynamic, open-set tasks like task-oriented navigation and embodied question answering (EQA) persist due to inadequate modeling of fine-grained spatio-temporal cues and physical world comprehension. To address this, we propose VEME, a novel cross-modal alignment method that enhances generalization in unseen scenes by learning an ego-centric, experience-centered world model. Our framework integrates three key components: (1) a cross-modal alignment framework bridging objects, spatial representations, and visual semantics with spatio-temporal cues to enhance VLM in-context learning; (2) a dynamic, implicit cognitive map activated by world embedding to enable task-relevant geometric-semantic memory recall; and (3) an instruction-based navigation and reasoning framework leveraging embodied priors for long-term planning and efficient exploration. By embedding geometry-aware spatio-temporal episodic experiences, our method significantly improves reasoning and planning in dynamic environments. Experimental results on VSI-Bench and VLN-CE demonstrate 1%-3% accuracy and exploration efficiency improvement compared to traditional approaches. 

**Abstract (ZH)**: 在未知环境中实现类似于人类的复杂任务推理仍然是 embodied intelligence 中的关键挑战。尽管先进昀多模态视觉语言模型在静态场景理解方面表现出色，但它们在时空推理和动态、开放集任务（如任务导向导航和体态问答）中的局限性依旧存在，这是因为它们对细粒度时空线索和物理世界理解建模的不足。为了解决这一问题，我们提出了一种新颖的跨模态对齐方法 VEME，通过学习以自我为中心的体验中心世界模型来增强对未见场景的泛化能力。我们的框架包含三个关键组成部分：（1）一种跨模态对齐框架，通过连接物体、空间表示和视觉语义并将之与时空线索结合来增强 VLM 的在场学习；（2）一种由世界嵌入激活的动态隐式认知地图，以实现与任务相关的几何语义记忆召回；（3）一种基于指令的导航与推理框架，利用体态先验进行长期规划和高效探索。通过嵌入几何感知的时空事件体验，我们的方法显著提高了在动态环境中推理与规划的能力。实验结果表明，相比于传统方法，在 VSI-Bench 和 VLN-CE 上的准确性和探索效率分别提高了 1%-3%。 

---
# Enabling Transparent Cyber Threat Intelligence Combining Large Language Models and Domain Ontologies 

**Title (ZH)**: 结合大型语言模型和领域本体的透明网络安全威胁情报 

**Authors**: Luca Cotti, Anisa Rula, Devis Bianchini, Federico Cerutti  

**Link**: [PDF](https://arxiv.org/pdf/2509.00081)  

**Abstract**: Effective Cyber Threat Intelligence (CTI) relies upon accurately structured and semantically enriched information extracted from cybersecurity system logs. However, current methodologies often struggle to identify and interpret malicious events reliably and transparently, particularly in cases involving unstructured or ambiguous log entries. In this work, we propose a novel methodology that combines ontology-driven structured outputs with Large Language Models (LLMs), to build an Artificial Intelligence (AI) agent that improves the accuracy and explainability of information extraction from cybersecurity logs. Central to our approach is the integration of domain ontologies and SHACL-based constraints to guide the language model's output structure and enforce semantic validity over the resulting graph. Extracted information is organized into an ontology-enriched graph database, enabling future semantic analysis and querying. The design of our methodology is motivated by the analytical requirements associated with honeypot log data, which typically comprises predominantly malicious activity. While our case study illustrates the relevance of this scenario, the experimental evaluation is conducted using publicly available datasets. Results demonstrate that our method achieves higher accuracy in information extraction compared to traditional prompt-only approaches, with a deliberate focus on extraction quality rather than processing speed. 

**Abstract (ZH)**: 有效的网络威胁情报（CTI）依赖于从网络安全系统日志中准确结构化和语义丰富的信息提取。然而，当前的方法往往难以可靠且透明地识别和解释恶意事件，特别是在涉及未结构化或模棱两可的日志条目时。在此工作中，我们提出了一种新的方法，该方法结合了本体驱动的结构输出与大规模语言模型（LLMs），构建一个人工智能（AI）代理，以提高从网络安全日志中提取信息的准确性和可解释性。我们方法的核心是将领域本体和基于SHACL的约束相结合，以指导语言模型的输出结构并确保结果图的语义有效性。提取的信息被组织到一个本体丰富的关系数据库中，以支持未来的语义分析和查询。我们的方法设计受到蜜罐日志数据的分析需求的驱动，这些数据通常主要包含恶意活动。虽然我们的案例研究说明了这一场景的相关性，但实验评估使用的是公开可用的数据集。结果表明，与仅使用提示的传统方法相比，我们的方法在信息提取准确性方面更高，注重提取质量而非处理速度。 

---
# ARTPS: Depth-Enhanced Hybrid Anomaly Detection and Learnable Curiosity Score for Autonomous Rover Target Prioritization 

**Title (ZH)**: ARTPS：深度增强混合异常检测及可学习的好奇度分数自主漫游者目标优先级分配 

**Authors**: Poyraz Baydemir  

**Link**: [PDF](https://arxiv.org/pdf/2509.00042)  

**Abstract**: We present ARTPS (Autonomous Rover Target Prioritization System), a novel hybrid AI system that combines depth estimation, anomaly detection, and learnable curiosity scoring for autonomous exploration of planetary surfaces. Our approach integrates monocular depth estimation using Vision Transformers with multi-component anomaly detection and a weighted curiosity score that balances known value, anomaly signals, depth variance, and surface roughness. The system achieves state-of-the-art performance with AUROC of 0.94, AUPRC of 0.89, and F1-Score of 0.87 on Mars rover datasets. We demonstrate significant improvements in target prioritization accuracy through ablation studies and provide comprehensive analysis of component contributions. The hybrid fusion approach reduces false positives by 23% while maintaining high detection sensitivity across diverse terrain types. 

**Abstract (ZH)**: 自主 Rover 目标优先级确定系统：结合深度估计、异常检测和可学习的好奇心评分的新型混合AI系统 

---
