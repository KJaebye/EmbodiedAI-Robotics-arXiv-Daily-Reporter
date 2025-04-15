# Teacher Motion Priors: Enhancing Robot Locomotion over Challenging Terrain 

**Title (ZH)**: 教师运动先验：提升机器人在复杂地形上的运动性能 

**Authors**: Fangcheng Jin, Yuqi Wang, Peixin Ma, Guodong Yang, Pan Zhao, En Li, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10390)  

**Abstract**: Achieving robust locomotion on complex terrains remains a challenge due to high dimensional control and environmental uncertainties. This paper introduces a teacher prior framework based on the teacher student paradigm, integrating imitation and auxiliary task learning to improve learning efficiency and generalization. Unlike traditional paradigms that strongly rely on encoder-based state embeddings, our framework decouples the network design, simplifying the policy network and deployment. A high performance teacher policy is first trained using privileged information to acquire generalizable motion skills. The teacher's motion distribution is transferred to the student policy, which relies only on noisy proprioceptive data, via a generative adversarial mechanism to mitigate performance degradation caused by distributional shifts. Additionally, auxiliary task learning enhances the student policy's feature representation, speeding up convergence and improving adaptability to varying terrains. The framework is validated on a humanoid robot, showing a great improvement in locomotion stability on dynamic terrains and significant reductions in development costs. This work provides a practical solution for deploying robust locomotion strategies in humanoid robots. 

**Abstract (ZH)**: 基于教师学生的先验框架：通过模仿和辅助任务学习实现复杂地形上的稳健运动 

---
# Flying Hand: End-Effector-Centric Framework for Versatile Aerial Manipulation Teleoperation and Policy Learning 

**Title (ZH)**: 飞行手：以末端执行器为中心的通用空中操作与策略学习框架 

**Authors**: Guanqi He, Xiaofeng Guo, Luyi Tang, Yuanhang Zhang, Mohammadreza Mousaei, Jiahe Xu, Junyi Geng, Sebastian Scherer, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10334)  

**Abstract**: Aerial manipulation has recently attracted increasing interest from both industry and academia. Previous approaches have demonstrated success in various specific tasks. However, their hardware design and control frameworks are often tightly coupled with task specifications, limiting the development of cross-task and cross-platform algorithms. Inspired by the success of robot learning in tabletop manipulation, we propose a unified aerial manipulation framework with an end-effector-centric interface that decouples high-level platform-agnostic decision-making from task-agnostic low-level control. Our framework consists of a fully-actuated hexarotor with a 4-DoF robotic arm, an end-effector-centric whole-body model predictive controller, and a high-level policy. The high-precision end-effector controller enables efficient and intuitive aerial teleoperation for versatile tasks and facilitates the development of imitation learning policies. Real-world experiments show that the proposed framework significantly improves end-effector tracking accuracy, and can handle multiple aerial teleoperation and imitation learning tasks, including writing, peg-in-hole, pick and place, changing light bulbs, etc. We believe the proposed framework provides one way to standardize and unify aerial manipulation into the general manipulation community and to advance the field. Project website: this https URL. 

**Abstract (ZH)**: 空中操作 recently 吸引了工业和学术界的广泛关注。尽管以往的方法在各种具体任务中取得了成功，但它们的硬件设计和控制框架往往紧密耦合于特定任务的要求，限制了跨任务和跨平台算法的发展。受桌面操作中机器人学习成功经验的启发，我们提出了一种以效应器为中心的统一空中操作框架，该框架将高层的平台无关决策与任务无关的低层控制相分离。该框架包括一个全驱动六旋翼无人机和一个四自由度的机械臂、一个以效应器为中心的全身模型预测控制器以及一个高层策略。高精度效应器控制器使得高效的直观空中遥控操作成为可能，并促进了模仿学习策略的发展。实验证明，提出的方法显著提高了末端执行器的跟踪准确性，并能处理包括书写、孔插针、取放、更换灯泡等多种空中遥控操作和模仿学习任务。我们相信，提出的框架为将空中操作标准化和统一到一般操作社区提供了途径，并推动了该领域的发展。项目网站: [this URL](this https URL)。 

---
# Siamese Network with Dual Attention for EEG-Driven Social Learning: Bridging the Human-Robot Gap in Long-Tail Autonomous Driving 

**Title (ZH)**: 基于双注意力机制的孪生网络在长尾自主驾驶中的人机协作社会学习：缩小人机器人差距 

**Authors**: Xiaoshan Zhou, Carol C. Menassa, Vineet R. Kamat  

**Link**: [PDF](https://arxiv.org/pdf/2504.10296)  

**Abstract**: Robots with wheeled, quadrupedal, or humanoid forms are increasingly integrated into built environments. However, unlike human social learning, they lack a critical pathway for intrinsic cognitive development, namely, learning from human feedback during interaction. To understand human ubiquitous observation, supervision, and shared control in dynamic and uncertain environments, this study presents a brain-computer interface (BCI) framework that enables classification of Electroencephalogram (EEG) signals to detect cognitively demanding and safety-critical events. As a timely and motivating co-robotic engineering application, we simulate a human-in-the-loop scenario to flag risky events in semi-autonomous robotic driving-representative of long-tail cases that pose persistent bottlenecks to the safety performance of smart mobility systems and robotic vehicles. Drawing on recent advances in few-shot learning, we propose a dual-attention Siamese convolutional network paired with Dynamic Time Warping Barycenter Averaging approach to generate robust EEG-encoded signal representations. Inverse source localization reveals activation in Broadman areas 4 and 9, indicating perception-action coupling during task-relevant mental imagery. The model achieves 80% classification accuracy under data-scarce conditions and exhibits a nearly 100% increase in the utility of salient features compared to state-of-the-art methods, as measured through integrated gradient attribution. Beyond performance, this study contributes to our understanding of the cognitive architecture required for BCI agents-particularly the role of attention and memory mechanisms-in categorizing diverse mental states and supporting both inter- and intra-subject adaptation. Overall, this research advances the development of cognitive robotics and socially guided learning for service robots in complex built environments. 

**Abstract (ZH)**: 具有轮式、四足或人形形式的机器人越来越多地集成到建筑物环境中。然而，与人类社会学习不同，它们缺乏一个关键的认知发展途径，即在互动中从人类反馈中学习。为了理解人类在动态和不确定环境中的普遍观察、监督和共享控制，本研究提出了一个脑-机接口（BCI）框架，以实现脑电图（EEG）信号分类，检测认知要求高和安全关键事件。作为及时且富有动力的协作机器人工程应用，我们模拟了一种循环人类在环的场景，以标记出在半自主机器人驾驶中代表大量尾部案例的风险事件，这些案例持续阻碍着智能移动系统和机器人车辆的安全性能。基于近期在少样本学习方面的进展，我们提出了一种双注意结构Siamese卷积网络配以动态时间战争.DisplayNameAveraging方法，以生成稳健的EEG编码信号表示。逆源定位显示了布罗dmann区域4和9的激活，表明在任务相关心智成像期间存在感知-行动 coupling。该模型在数据稀缺条件下实现了80%的分类准确率，并且与最先进的方法相比，其显著特征的实用性提高了近100%，这通过综合梯度归因进行测量。超越性能，本研究还扩展了我们对BCI代理所需认知架构的理解，尤其是注意和记忆机制在分类不同心状状态和支持跨内个体适应方面的作用。总体而言，这项研究推进了认知机器人和社交引导服务机器人在复杂建筑物环境中的发展。 

---
# Look-to-Touch: A Vision-Enhanced Proximity and Tactile Sensor for Distance and Geometry Perception in Robotic Manipulation 

**Title (ZH)**: 视知觉增强的近距离和触觉传感器：用于 robotic 操作的距离与几何感知 

**Authors**: Yueshi Dong, Jieji Ren, Zhenle Liu, Zhanxuan Peng, Zihao Yuan, Ningbin Zhang, Guoying Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10280)  

**Abstract**: Camera-based tactile sensors provide robots with a high-performance tactile sensing approach for environment perception and dexterous manipulation. However, achieving comprehensive environmental perception still requires cooperation with additional sensors, which makes the system bulky and limits its adaptability to unstructured environments. In this work, we present a vision-enhanced camera-based dual-modality sensor, which realizes full-scale distance sensing from 50 cm to -3 mm while simultaneously keeping ultra-high-resolution texture sensing and reconstruction capabilities. Unlike conventional designs with fixed opaque gel layers, our sensor features a partially transparent sliding window, enabling mechanical switching between tactile and visual modes. For each sensing mode, a dynamic distance sensing model and a contact geometry reconstruction model are proposed. Through integration with soft robotic fingers, we systematically evaluate the performance of each mode, as well as in their synergistic operation. Experimental results show robust distance tracking across various speeds, nanometer-scale roughness detection, and sub-millimeter 3D texture reconstruction. The combination of both modalities improves the robot's efficiency in executing grasping tasks. Furthermore, the embedded mechanical transmission in the sensor allows for fine-grained intra-hand adjustments and precise manipulation, unlocking new capabilities for soft robotic hands. 

**Abstract (ZH)**: 基于视觉增强的相机双模态传感器实现全面距离感知与超高清纹理感测与重建 

---
# A Human-Sensitive Controller: Adapting to Human Ergonomics and Physical Constraints via Reinforcement Learning 

**Title (ZH)**: 人性化控制器：强化学习适应人类人体工学和物理约束 

**Authors**: Vitor Martins, Sara M. Cerqueira, Mercedes Balcells, Elazer R Edelman, Cristina P. Santos  

**Link**: [PDF](https://arxiv.org/pdf/2504.10102)  

**Abstract**: Work-Related Musculoskeletal Disorders continue to be a major challenge in industrial environments, leading to reduced workforce participation, increased healthcare costs, and long-term disability. This study introduces a human-sensitive robotic system aimed at reintegrating individuals with a history of musculoskeletal disorders into standard job roles, while simultaneously optimizing ergonomic conditions for the broader workforce. This research leverages reinforcement learning to develop a human-aware control strategy for collaborative robots, focusing on optimizing ergonomic conditions and preventing pain during task execution. Two RL approaches, Q-Learning and Deep Q-Network (DQN), were implemented and tested to personalize control strategies based on individual user characteristics. Although experimental results revealed a simulation-to-real gap, a fine-tuning phase successfully adapted the policies to real-world conditions. DQN outperformed Q-Learning by completing tasks faster while maintaining zero pain risk and safe ergonomic levels. The structured testing protocol confirmed the system's adaptability to diverse human anthropometries, underscoring the potential of RL-driven cobots to enable safer, more inclusive workplaces. 

**Abstract (ZH)**: 工关联肌骨骼紊乱仍然是工业环境中的一项重大挑战，导致劳动力参与率降低、医疗成本增加和长期残疾。本研究介绍了一种以人为本的机器人系统，旨在将具有肌骨骼疾病史的个体重新融入标准工作岗位，同时优化更广泛的劳动力的工效条件。该研究利用强化学习开发了一种以人为本的协作机器人控制策略，侧重于优化工效条件并在执行任务时预防疼痛。实现了两种RL方法，即Q-Learning和深度Q网络（DQN），并进行了测试，以根据个体用户特性个性化控制策略。尽管实验结果表明存在仿真到现实的差距，但细调阶段成功地将策略适应了实际条件。DQN在完成任务速度更快的同时保持了零疼痛风险和安全工效水平，结构化测试协议证实了该系统对多样的人体测量的适应性，强调了RL驱动的协作机器人在实现更安全、更包容的工作场所方面的潜力。 

---
# Joint Action Language Modelling for Transparent Policy Execution 

**Title (ZH)**: 联合动作语言建模以实现透明政策执行 

**Authors**: Theodor Wulff, Rahul Singh Maharjan, Xinyun Chi, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10055)  

**Abstract**: An agent's intention often remains hidden behind the black-box nature of embodied policies. Communication using natural language statements that describe the next action can provide transparency towards the agent's behavior. We aim to insert transparent behavior directly into the learning process, by transforming the problem of policy learning into a language generation problem and combining it with traditional autoregressive modelling. The resulting model produces transparent natural language statements followed by tokens representing the specific actions to solve long-horizon tasks in the Language-Table environment. Following previous work, the model is able to learn to produce a policy represented by special discretized tokens in an autoregressive manner. We place special emphasis on investigating the relationship between predicting actions and producing high-quality language for a transparent agent. We find that in many cases both the quality of the action trajectory and the transparent statement increase when they are generated simultaneously. 

**Abstract (ZH)**: 一种代理的意图往往被其体内策略的黑箱性质所隐藏。使用自然语言语句描述下一个动作的通信可以提供对代理行为的透明度。我们旨在通过将策略学习问题转变为语言生成问题，并结合传统的自回归建模，直接在学习过程中插入透明行为。该模型生成透明的自然语言语句，随后是表示具体动作的标记，以解决Language-Table环境中的长期任务。借鉴先前的工作，该模型能够以自回归方式学习产生由特殊离散标记表示的策略。我们特别关注预测动作与生成高质量透明语言之间关系的研究。我们发现，在许多情况下，当动作轨迹和透明语句同时生成时，它们的质量都会提高。 

---
# Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models 

**Title (ZH)**: Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models 

**Authors**: Hao Ren, Yiming Zeng, Zetong Bi, Zhaoliang Wan, Junlong Huang, Hui Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.10041)  

**Abstract**: Recent advancements in diffusion-based imitation learning, which show impressive performance in modeling multimodal distributions and training stability, have led to substantial progress in various robot learning tasks. In visual navigation, previous diffusion-based policies typically generate action sequences by initiating from denoising Gaussian noise. However, the target action distribution often diverges significantly from Gaussian noise, leading to redundant denoising steps and increased learning complexity. Additionally, the sparsity of effective action distributions makes it challenging for the policy to generate accurate actions without guidance. To address these issues, we propose a novel, unified visual navigation framework leveraging the denoising diffusion bridge models named NaviBridger. This approach enables action generation by initiating from any informative prior actions, enhancing guidance and efficiency in the denoising process. We explore how diffusion bridges can enhance imitation learning in visual navigation tasks and further examine three source policies for generating prior actions. Extensive experiments in both simulated and real-world indoor and outdoor scenarios demonstrate that NaviBridger accelerates policy inference and outperforms the baselines in generating target action sequences. Code is available at this https URL. 

**Abstract (ZH)**: 基于弥散的 imitation 学习 Recent 进展展示了在建模多模态分布和训练稳定性方面的出色性能，这在多种机器人学习任务中取得了显著进步。在视觉导航中，先前的基于弥散的策略通常从去噪高斯噪声开始生成动作序列。然而，目标动作分布往往与高斯噪声有显著差异，导致不必要的去噪步骤并增加了学习复杂性。此外，有效动作分布的稀疏性使策略在没有引导的情况下生成准确动作变得极具挑战性。为解决这些问题，我们提出了一种新颖的统一视觉导航框架，利用去噪弥散桥梁模型 NaviBridger。此方法允许从任何信息性先验动作开始生成动作，增强去噪过程中的引导和效率。我们探讨了弥散桥梁如何在视觉导航任务中增强 imitation 学习，并进一步研究了三种用于生成先验动作的源策略。在仿真和真实世界室内外场景中的 extensive 实验表明，NaviBridger 加快了策略推理并优于基线方法，在生成目标动作序列方面表现更优。代码可在以下链接获取：this https URL。 

---
# EmbodiedAgent: A Scalable Hierarchical Approach to Overcome Practical Challenge in Multi-Robot Control 

**Title (ZH)**: 具身代理：一种克服多机器人控制实践挑战的可扩展分层方法 

**Authors**: Hanwen Wan, Yifei Chen, Zeyu Wei, Dongrui Li, Zexin Lin, Donghao Wu, Jiu Cheng, Yuxiang Zhang, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.10030)  

**Abstract**: This paper introduces EmbodiedAgent, a hierarchical framework for heterogeneous multi-robot control. EmbodiedAgent addresses critical limitations of hallucination in impractical tasks. Our approach integrates a next-action prediction paradigm with a structured memory system to decompose tasks into executable robot skills while dynamically validating actions against environmental constraints. We present MultiPlan+, a dataset of more than 18,000 annotated planning instances spanning 100 scenarios, including a subset of impractical cases to mitigate hallucination. To evaluate performance, we propose the Robot Planning Assessment Schema (RPAS), combining automated metrics with LLM-aided expert grading. Experiments demonstrate EmbodiedAgent's superiority over state-of-the-art models, achieving 71.85% RPAS score. Real-world validation in an office service task highlights its ability to coordinate heterogeneous robots for long-horizon objectives. 

**Abstract (ZH)**: 本文介绍了EmbodiedAgent，一种异构多机器人控制的分层框架。EmbodiedAgent解决了不切实际任务中的幻觉关键限制。我们的方法结合了下一步动作预测范式与结构化记忆系统，将任务分解为可执行的机器人技能，并动态验证动作是否符合环境约束。我们提出了包含超过18,000个标注计划实例的MultiPlan+数据集，这些实例覆盖了100个场景，包括某些不切实际的案例以减轻幻觉。为评估性能，我们提出了一种机器人计划评估方案（RPAS），结合了自动评价指标与LLM辅助专家评分。实验结果表明，EmbodiedAgent在性能上优于当前最先进的模型，达到了71.85%的RPAS分数。在办公室服务任务的实际验证中，展示了其协调异构机器人完成长期目标的能力。 

---
# KeyMPs: One-Shot Vision-Language Guided Motion Generation by Sequencing DMPs for Occlusion-Rich Tasks 

**Title (ZH)**: KeyMPs: 通过序列化DMPs实现一-shot视觉-语言引导的运动生成，用于遮挡丰富的任务 

**Authors**: Edgar Anarossi, Yuhwan Kwon, Hirotaka Tahara, Shohei Tanaka, Keisuke Shirai, Masashi Hamaya, Cristian C. Beltran-Hernandez, Atsushi Hashimoto, Takamitsu Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2504.10011)  

**Abstract**: Dynamic Movement Primitives (DMPs) provide a flexible framework wherein smooth robotic motions are encoded into modular parameters. However, they face challenges in integrating multimodal inputs commonly used in robotics like vision and language into their framework. To fully maximize DMPs' potential, enabling them to handle multimodal inputs is essential. In addition, we also aim to extend DMPs' capability to handle object-focused tasks requiring one-shot complex motion generation, as observation occlusion could easily happen mid-execution in such tasks (e.g., knife occlusion in cake icing, hand occlusion in dough kneading, etc.). A promising approach is to leverage Vision-Language Models (VLMs), which process multimodal data and can grasp high-level concepts. However, they typically lack enough knowledge and capabilities to directly infer low-level motion details and instead only serve as a bridge between high-level instructions and low-level control. To address this limitation, we propose Keyword Labeled Primitive Selection and Keypoint Pairs Generation Guided Movement Primitives (KeyMPs), a framework that combines VLMs with sequencing of DMPs. KeyMPs use VLMs' high-level reasoning capability to select a reference primitive through keyword labeled primitive selection and VLMs' spatial awareness to generate spatial scaling parameters used for sequencing DMPs by generalizing the overall motion through keypoint pairs generation, which together enable one-shot vision-language guided motion generation that aligns with the intent expressed in the multimodal input. We validate our approach through an occlusion-rich manipulation task, specifically object cutting experiments in both simulated and real-world environments, demonstrating superior performance over other DMP-based methods that integrate VLMs support. 

**Abstract (ZH)**: 动态运动模块（KeyMPs）结合视觉语言模型实现多模态输入的复杂运动一次生成 

---
# FLoRA: Sample-Efficient Preference-based RL via Low-Rank Style Adaptation of Reward Functions 

**Title (ZH)**: FLoRA: 基于偏好低秩风格适配奖励函数的样本高效强化学习 

**Authors**: Daniel Marta, Simon Holk, Miguel Vasco, Jens Lundell, Timon Homberger, Finn Busch, Olov Andersson, Danica Kragic, Iolanda Leite  

**Link**: [PDF](https://arxiv.org/pdf/2504.10002)  

**Abstract**: Preference-based reinforcement learning (PbRL) is a suitable approach for style adaptation of pre-trained robotic behavior: adapting the robot's policy to follow human user preferences while still being able to perform the original task. However, collecting preferences for the adaptation process in robotics is often challenging and time-consuming. In this work we explore the adaptation of pre-trained robots in the low-preference-data regime. We show that, in this regime, recent adaptation approaches suffer from catastrophic reward forgetting (CRF), where the updated reward model overfits to the new preferences, leading the agent to become unable to perform the original task. To mitigate CRF, we propose to enhance the original reward model with a small number of parameters (low-rank matrices) responsible for modeling the preference adaptation. Our evaluation shows that our method can efficiently and effectively adjust robotic behavior to human preferences across simulation benchmark tasks and multiple real-world robotic tasks. 

**Abstract (ZH)**: 基于偏好增强学习的预训练机器人适应性研究：低偏好数据下的行为调整 

---
# GenTe: Generative Real-world Terrains for General Legged Robot Locomotion Control 

**Title (ZH)**: GenTe: 生成的现实地形用于通用腿足机器人运动控制 

**Authors**: Hanwen Wan, Mengkang Li, Donghao Wu, Yebin Zhong, Yixuan Deng, Zhenglong Sun, Xiaoqiang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.09997)  

**Abstract**: Developing bipedal robots capable of traversing diverse real-world terrains presents a fundamental robotics challenge, as existing methods using predefined height maps and static environments fail to address the complexity of unstructured landscapes. To bridge this gap, we propose GenTe, a framework for generating physically realistic and adaptable terrains to train generalizable locomotion policies. GenTe constructs an atomic terrain library that includes both geometric and physical terrains, enabling curriculum training for reinforcement learning-based locomotion policies. By leveraging function-calling techniques and reasoning capabilities of Vision-Language Models (VLMs), GenTe generates complex, contextually relevant terrains from textual and graphical inputs. The framework introduces realistic force modeling for terrain interactions, capturing effects such as soil sinkage and hydrodynamic resistance. To the best of our knowledge, GenTe is the first framework that systemically generates simulation environments for legged robot locomotion control. Additionally, we introduce a benchmark of 100 generated terrains. Experiments demonstrate improved generalization and robustness in bipedal robot locomotion. 

**Abstract (ZH)**: 开发能够在多样化真实地形中行进的双足机器人是机器人学领域的一项基本挑战，现有使用预定义高度图和静态环境的方法无法应对未结构化地形的复杂性。为解决这一问题，我们提出了一种名为GenTe的框架，用于生成物理上真实且可适应的地形以训练可泛化的运动策略。GenTe构建了一个包含几何和物理地形的原子地形库，支持基于强化学习的运动策略的课程训练。通过利用函数调用技术和视觉-语言模型的推理能力，GenTe能够从文本和图形输入中生成复杂且上下文相关的真实地形。该框架引入了真实的力模型来模拟地形交互，捕捉诸如土壤压缩和水动力阻力等效应。据我们所知，GenTe是首个系统性生成用于腿足机器人运动控制的模拟环境的框架。此外，我们还引入了一个包含100个生成地形的基准。实验结果表明，GenTe能够提高双足机器人运动的泛化能力和鲁棒性。 

---
# Efficient Task-specific Conditional Diffusion Policies: Shortcut Model Acceleration and SO(3) Optimization 

**Title (ZH)**: 任务特定条件扩散策略的高效加速模型与SO(3)优化 

**Authors**: Haiyong Yu, Yanqiong Jin, Yonghao He, Wei Sui  

**Link**: [PDF](https://arxiv.org/pdf/2504.09927)  

**Abstract**: Imitation learning, particularly Diffusion Policies based methods, has recently gained significant traction in embodied AI as a powerful approach to action policy generation. These models efficiently generate action policies by learning to predict noise. However, conventional Diffusion Policy methods rely on iterative denoising, leading to inefficient inference and slow response times, which hinder real-time robot control. To address these limitations, we propose a Classifier-Free Shortcut Diffusion Policy (CF-SDP) that integrates classifier-free guidance with shortcut-based acceleration, enabling efficient task-specific action generation while significantly improving inference speed. Furthermore, we extend diffusion modeling to the SO(3) manifold in shortcut model, defining the forward and reverse processes in its tangent space with an isotropic Gaussian distribution. This ensures stable and accurate rotational estimation, enhancing the effectiveness of diffusion-based control. Our approach achieves nearly 5x acceleration in diffusion inference compared to DDIM-based Diffusion Policy while maintaining task performance. Evaluations both on the RoboTwin simulation platform and real-world scenarios across various tasks demonstrate the superiority of our method. 

**Abstract (ZH)**: 无监督学习，尤其是基于扩散策略的方法，在体现人工智能中的动作策略生成方面 recently gained significant traction. 这些模型通过学习预测噪声高效地生成动作策略。然而，传统的扩散策略方法依赖于迭代去噪，导致推断效率低下和响应时间缓慢，这阻碍了实时机器人控制。为了解决这些限制，我们提出了一种无分类器快捷扩散策略（CF-SDP），该方法将无分类器引导与基于快捷路径的加速相结合，在实现任务特定动作生成的同时显著提高推断速度。此外，我们将扩散建模扩展到SO(3)流形，在快捷模型中定义其切空间中的前向和反向过程，使用各向同性高斯分布。这确保了稳定的准确的旋转估计，增强了基于扩散的控制的有效性。我们的方法在与DDIM基于的扩散策略相比，实现了约5倍的扩散推断加速，同时保持任务性能。在RoboTwin仿真平台及各类真实场景下进行的评估均证明了该方法的优势。 

---
# PreCi: Pretraining and Continual Improvement of Humanoid Locomotion via Model-Assumption-Based Regularization 

**Title (ZH)**: 基于模型假设正则化的预训练与持续改进 humanoid 行走研究 

**Authors**: Hyunyoung Jung, Zhaoyuan Gu, Ye Zhao, Hae-Won Park, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2504.09833)  

**Abstract**: Humanoid locomotion is a challenging task due to its inherent complexity and high-dimensional dynamics, as well as the need to adapt to diverse and unpredictable environments. In this work, we introduce a novel learning framework for effectively training a humanoid locomotion policy that imitates the behavior of a model-based controller while extending its capabilities to handle more complex locomotion tasks, such as more challenging terrain and higher velocity commands. Our framework consists of three key components: pre-training through imitation of the model-based controller, fine-tuning via reinforcement learning, and model-assumption-based regularization (MAR) during fine-tuning. In particular, MAR aligns the policy with actions from the model-based controller only in states where the model assumption holds to prevent catastrophic forgetting. We evaluate the proposed framework through comprehensive simulation tests and hardware experiments on a full-size humanoid robot, Digit, demonstrating a forward speed of 1.5 m/s and robust locomotion across diverse terrains, including slippery, sloped, uneven, and sandy terrains. 

**Abstract (ZH)**: 基于模型控制器的模仿与假设正则化的类人运动学习框架 

---
# Adapting Robot's Explanation for Failures Based on Observed Human Behavior in Human-Robot Collaboration 

**Title (ZH)**: 基于人类行为观察的机器人故障解释适应性调整研究 

**Authors**: Andreas Naoum, Parag Khanna, Elmira Yadollahi, Mårten Björkman, Christian Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.09717)  

**Abstract**: This work aims to interpret human behavior to anticipate potential user confusion when a robot provides explanations for failure, allowing the robot to adapt its explanations for more natural and efficient collaboration. Using a dataset that included facial emotion detection, eye gaze estimation, and gestures from 55 participants in a user study, we analyzed how human behavior changed in response to different types of failures and varying explanation levels. Our goal is to assess whether human collaborators are ready to accept less detailed explanations without inducing confusion. We formulate a data-driven predictor to predict human confusion during robot failure explanations. We also propose and evaluate a mechanism, based on the predictor, to adapt the explanation level according to observed human behavior. The promising results from this evaluation indicate the potential of this research in adapting a robot's explanations for failures to enhance the collaborative experience. 

**Abstract (ZH)**: 本研究旨在解释人类行为，以预测当机器人在解释故障时用户可能产生的潜在困惑，从而使机器人能够根据需要调整其解释以实现更自然和高效的协作。通过包含55名参与者用户研究中面部情绪检测、眼动估计和手势的数据集，我们分析了人类行为在面对不同类型的故障和不同解释水平时的变化。我们的目标是评估人类合作者在接受较不详细解释时是否会产生困惑。我们提出了一个基于数据的预测器来预测机器人故障解释期间的人类困惑。我们还提出并评估了一种机制，该机制根据观察到的人类行为调整解释水平。这项评估取得的有希望的结果表明了这项研究在调整机器人故障解释以增强协作体验方面的潜力。 

---
# GeoNav: Empowering MLLMs with Explicit Geospatial Reasoning Abilities for Language-Goal Aerial Navigation 

**Title (ZH)**: GeoNav: 为语言目标航空导航增强显式地理空间推理能力的MLLMs 

**Authors**: Haotian Xu, Yue Hu, Chen Gao, Zhengqiu Zhu, Yong Zhao, Yong Li, Quanjun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.09587)  

**Abstract**: Language-goal aerial navigation is a critical challenge in embodied AI, requiring UAVs to localize targets in complex environments such as urban blocks based on textual specification. Existing methods, often adapted from indoor navigation, struggle to scale due to limited field of view, semantic ambiguity among objects, and lack of structured spatial reasoning. In this work, we propose GeoNav, a geospatially aware multimodal agent to enable long-range navigation. GeoNav operates in three phases-landmark navigation, target search, and precise localization-mimicking human coarse-to-fine spatial strategies. To support such reasoning, it dynamically builds two different types of spatial memory. The first is a global but schematic cognitive map, which fuses prior textual geographic knowledge and embodied visual cues into a top-down, annotated form for fast navigation to the landmark region. The second is a local but delicate scene graph representing hierarchical spatial relationships between blocks, landmarks, and objects, which is used for definite target localization. On top of this structured representation, GeoNav employs a spatially aware, multimodal chain-of-thought prompting mechanism to enable multimodal large language models with efficient and interpretable decision-making across stages. On the CityNav urban navigation benchmark, GeoNav surpasses the current state-of-the-art by up to 12.53% in success rate and significantly improves navigation efficiency, even in hard-level tasks. Ablation studies highlight the importance of each module, showcasing how geospatial representations and coarse-to-fine reasoning enhance UAV navigation. 

**Abstract (ZH)**: 基于地理意识的多模态导航：复杂环境中的语言目标空中导航 

---
# AirVista-II: An Agentic System for Embodied UAVs Toward Dynamic Scene Semantic Understanding 

**Title (ZH)**: AirVista-II: 一个自主系统用于动态场景语义理解的 embodied UAVs 

**Authors**: Fei Lin, Yonglin Tian, Tengchao Zhang, Jun Huang, Sangtian Guan, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09583)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly important in dynamic environments such as logistics transportation and disaster response. However, current tasks often rely on human operators to monitor aerial videos and make operational decisions. This mode of human-machine collaboration suffers from significant limitations in efficiency and adaptability. In this paper, we present AirVista-II -- an end-to-end agentic system for embodied UAVs, designed to enable general-purpose semantic understanding and reasoning in dynamic scenes. The system integrates agent-based task identification and scheduling, multimodal perception mechanisms, and differentiated keyframe extraction strategies tailored for various temporal scenarios, enabling the efficient capture of critical scene information. Experimental results demonstrate that the proposed system achieves high-quality semantic understanding across diverse UAV-based dynamic scenarios under a zero-shot setting. 

**Abstract (ZH)**: 无人机（UAVs）在物流运输和灾难响应等动态环境中的应用越来越重要。然而，当前任务往往依赖于人类操作员监控空中视频并作出操作决策。这种人机协作模式在效率和适应性方面存在显著局限。本文介绍了一种端到端的自主系统AirVista-II，旨在使具身无人机具备对动态场景的一般语义理解和推理能力。该系统结合了基于代理的任务识别和调度、多模态感知机制以及针对不同时间场景定制的关键帧提取策略，能够高效地捕获关键场景信息。实验结果表明，在零样本情况下，所提系统在多样化的无人机动态场景中实现了高质量的语义理解。 

---
# Embodied Chain of Action Reasoning with Multi-Modal Foundation Model for Humanoid Loco-manipulation 

**Title (ZH)**: 具身行动链推理结合多模态基础模型的人形动Manipulation 

**Authors**: Yu Hao, Geeta Chandra Raju Bethala, Niraj Pudasaini, Hao Huang, Shuaihang Yuan, Congcong Wen, Baoru Huang, Anh Nguyen, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09532)  

**Abstract**: Enabling humanoid robots to autonomously perform loco-manipulation tasks in complex, unstructured environments poses significant challenges. This entails equipping robots with the capability to plan actions over extended horizons while leveraging multi-modality to bridge gaps between high-level planning and actual task execution. Recent advancements in multi-modal foundation models have showcased substantial potential in enhancing planning and reasoning abilities, particularly in the comprehension and processing of semantic information for robotic control tasks. In this paper, we introduce a novel framework based on foundation models that applies the embodied chain of action reasoning methodology to autonomously plan actions from textual instructions for humanoid loco-manipulation. Our method integrates humanoid-specific chain of thought methodology, including detailed affordance and body movement analysis, which provides a breakdown of the task into a sequence of locomotion and manipulation actions. Moreover, we incorporate spatial reasoning based on the observation and target object properties to effectively navigate where target position may be unseen or occluded. Through rigorous experimental setups on object rearrangement, manipulations and loco-manipulation tasks on a real-world environment, we evaluate our method's efficacy on the decoupled upper and lower body control and demonstrate the effectiveness of the chain of robotic action reasoning strategies in comprehending human instructions. 

**Abstract (ZH)**: 自主在复杂非结构化环境中执行类人机器人拾放任务面临显著挑战：基于多模态基础模型的类人机器人连贯动作自主规划方法 

---
# REALM: Real-Time Estimates of Assistance for Learned Models in Human-Robot Interaction 

**Title (ZH)**: REALM: 人类与机器人交互中学习模型的实时辅助估计 

**Authors**: Michael Hagenow, Julie A. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.09243)  

**Abstract**: There are a variety of mechanisms (i.e., input types) for real-time human interaction that can facilitate effective human-robot teaming. For example, previous works have shown how teleoperation, corrective, and discrete (i.e., preference over a small number of choices) input can enable robots to complete complex tasks. However, few previous works have looked at combining different methods, and in particular, opportunities for a robot to estimate and elicit the most effective form of assistance given its understanding of a task. In this paper, we propose a method for estimating the value of different human assistance mechanisms based on the action uncertainty of a robot policy. Our key idea is to construct mathematical expressions for the expected post-interaction differential entropy (i.e., uncertainty) of a stochastic robot policy to compare the expected value of different interactions. As each type of human input imposes a different requirement for human involvement, we demonstrate how differential entropy estimates can be combined with a likelihood penalization approach to effectively balance feedback informational needs with the level of required input. We demonstrate evidence of how our approach interfaces with emergent learning models (e.g., a diffusion model) to produce accurate assistance value estimates through both simulation and a robot user study. Our user study results indicate that the proposed approach can enable task completion with minimal human feedback for uncertain robot behaviors. 

**Abstract (ZH)**: 实时人类交互机制在促进人机团队协作中的多样性及其价值估计：一种基于机器人策略动作不确定性的方法 

---
# CL-CoTNav: Closed-Loop Hierarchical Chain-of-Thought for Zero-Shot Object-Goal Navigation with Vision-Language Models 

**Title (ZH)**: CL-CoTNav: 闭合环路层次链式思考在视觉语言模型支持下的零样本物体目标导航 

**Authors**: Yuxin Cai, Xiangkun He, Maonan Wang, Hongliang Guo, Wei-Yun Yau, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2504.09000)  

**Abstract**: Visual Object Goal Navigation (ObjectNav) requires a robot to locate a target object in an unseen environment using egocentric observations. However, decision-making policies often struggle to transfer to unseen environments and novel target objects, which is the core generalization problem. Traditional end-to-end learning methods exacerbate this issue, as they rely on memorizing spatial patterns rather than employing structured reasoning, limiting their ability to generalize effectively. In this letter, we introduce Closed-Loop Hierarchical Chain-of-Thought Navigation (CL-CoTNav), a vision-language model (VLM)-driven ObjectNav framework that integrates structured reasoning and closed-loop feedback into navigation decision-making. To enhance generalization, we fine-tune a VLM using multi-turn question-answering (QA) data derived from human demonstration trajectories. This structured dataset enables hierarchical Chain-of-Thought (H-CoT) prompting, systematically extracting compositional knowledge to refine perception and decision-making, inspired by the human cognitive process of locating a target object through iterative reasoning steps. Additionally, we propose a Closed-Loop H-CoT mechanism that incorporates detection and reasoning confidence scores into training. This adaptive weighting strategy guides the model to prioritize high-confidence data pairs, mitigating the impact of noisy inputs and enhancing robustness against hallucinated or incorrect reasoning. Extensive experiments in the AI Habitat environment demonstrate CL-CoTNav's superior generalization to unseen scenes and novel object categories. Our method consistently outperforms state-of-the-art approaches in navigation success rate (SR) and success weighted by path length (SPL) by 22.4\%. We release our datasets, models, and supplementary videos on our project page. 

**Abstract (ZH)**: 基于闭合环层次链式思维的视觉语言目标导航（CL-CoTNav） 

---
# ST-Booster: An Iterative SpatioTemporal Perception Booster for Vision-and-Language Navigation in Continuous Environments 

**Title (ZH)**: ST-增强器：连续环境中基于时空感知的迭代视觉-语言导航增强方法 

**Authors**: Lu Yue, Dongliang Zhou, Liang Xie, Erwei Yin, Feitian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09843)  

**Abstract**: Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to navigate unknown, continuous spaces based on natural language instructions. Compared to discrete settings, VLN-CE poses two core perception challenges. First, the absence of predefined observation points leads to heterogeneous visual memories and weakened global spatial correlations. Second, cumulative reconstruction errors in three-dimensional scenes introduce structural noise, impairing local feature perception. To address these challenges, this paper proposes ST-Booster, an iterative spatiotemporal booster that enhances navigation performance through multi-granularity perception and instruction-aware reasoning. ST-Booster consists of three key modules -- Hierarchical SpatioTemporal Encoding (HSTE), Multi-Granularity Aligned Fusion (MGAF), and ValueGuided Waypoint Generation (VGWG). HSTE encodes long-term global memory using topological graphs and captures shortterm local details via grid maps. MGAF aligns these dualmap representations with instructions through geometry-aware knowledge fusion. The resulting representations are iteratively refined through pretraining tasks. During reasoning, VGWG generates Guided Attention Heatmaps (GAHs) to explicitly model environment-instruction relevance and optimize waypoint selection. Extensive comparative experiments and performance analyses are conducted, demonstrating that ST-Booster outperforms existing state-of-the-art methods, particularly in complex, disturbance-prone environments. 

**Abstract (ZH)**: 连续环境中的视觉语言导航（VLN-CE）要求代理基于自然语言指令在未知的连续空间中进行导航。与离散环境相比，VLN-CE 提出了两个核心感知挑战。首先，缺乏预定义的观测点导致视觉记忆异质化和全局空间相关性减弱。其次，在三维场景中的累积重构误差引入了结构噪声，损害了局部特征感知能力。为应对这些挑战，本文提出了一种迭代时空增强器 ST-Booster，通过多层次感知和指令感知推理来提升导航性能。ST-Booster 包含三个关键模块——层次时空编码（HSTE）、多粒度对齐融合（MGAF）和价值引导的航点生成（VGWG）。HSTE 使用拓扑图来编码长期的全局记忆，并通过格网地图捕捉短期的局部细节。MGAF 通过几何感知知识融合将这两种地图表示与指令对齐，结果表示通过预训练任务进行迭代细化。在推理期间，VGWG 生成引导注意力热点图（GAHs）以明确建模环境-指令相关性并优化航点选择。通过广泛的对比实验和性能分析，证明了 ST-Booster 在复杂且干扰性强的环境中的表现优于现有最先进的方法。 

---
# Endowing Embodied Agents with Spatial Reasoning Capabilities for Vision-and-Language Navigation 

**Title (ZH)**: 赋予具身代理空间推理能力以实现视觉-语言导航 

**Authors**: Luo Ling, Bai Qianqian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08806)  

**Abstract**: Enhancing the spatial perception capabilities of mobile robots is crucial for achieving embodied Vision-and-Language Navigation (VLN). Although significant progress has been made in simulated environments, directly transferring these capabilities to real-world scenarios often results in severe hallucination phenomena, causing robots to lose effective spatial awareness. To address this issue, we propose BrainNav, a bio-inspired spatial cognitive navigation framework inspired by biological spatial cognition theories and cognitive map theory. BrainNav integrates dual-map (coordinate map and topological map) and dual-orientation (relative orientation and absolute orientation) strategies, enabling real-time navigation through dynamic scene capture and path planning. Its five core modules-Hippocampal Memory Hub, Visual Cortex Perception Engine, Parietal Spatial Constructor, Prefrontal Decision Center, and Cerebellar Motion Execution Unit-mimic biological cognitive functions to reduce spatial hallucinations and enhance adaptability. Validated in a zero-shot real-world lab environment using the Limo Pro robot, BrainNav, compatible with GPT-4, outperforms existing State-of-the-Art (SOTA) Vision-and-Language Navigation in Continuous Environments (VLN-CE) methods without fine-tuning. 

**Abstract (ZH)**: 增强移动机器人在空间感知能力对于实现具身视觉-语言导航（VLN）至关重要。尽管在模拟环境中取得了显著进步，但这些能力直接应用于真实世界场景时，往往会引发严重的幻觉现象，导致机器人失去有效的空间意识。为解决这一问题，我们提出了一种受生物空间认知理论和认知地图理论启发的空间认知导航框架BrainNav。BrainNav 结合了双地图（坐标地图和拓扑地图）和双方向（相对方向和绝对方向）策略，通过动态场景捕捉和路径规划实现实时导航。其五大核心模块—海马记忆中枢、视觉皮层感知引擎、枕叶空间构造器、前额叶决策中心和小脑运动执行单元—模拟生物认知功能，减少空间幻觉并增强适应性。在使用Limo Pro机器人进行的零样本真实世界实验室环境中验证，BrainNav 不需要微调就超过了现有的最先进的连续环境视觉-语言导航（VLN-CE）方法。 

---
# Breaking the Data Barrier -- Building GUI Agents Through Task Generalization 

**Title (ZH)**: 打破数据障碍——通过任务泛化构建GUI代理 

**Authors**: Junlei Zhang, Zichen Ding, Chang Ma, Zijie Chen, Qiushi Sun, Zhenzhong Lan, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2504.10127)  

**Abstract**: Graphical User Interface (GUI) agents offer cross-platform solutions for automating complex digital tasks, with significant potential to transform productivity workflows. However, their performance is often constrained by the scarcity of high-quality trajectory data. To address this limitation, we propose training Vision Language Models (VLMs) on data-rich, reasoning-intensive tasks during a dedicated mid-training stage, and then examine how incorporating these tasks facilitates generalization to GUI planning scenarios. Specifically, we explore a range of tasks with readily available instruction-tuning data, including GUI perception, multimodal reasoning, and textual reasoning. Through extensive experiments across 11 mid-training tasks, we demonstrate that: (1) Task generalization proves highly effective, yielding substantial improvements across most settings. For instance, multimodal mathematical reasoning enhances performance on AndroidWorld by an absolute 6.3%. Remarkably, text-only mathematical data significantly boosts GUI web agent performance, achieving a 5.6% improvement on WebArena and 5.4% improvement on AndroidWorld, underscoring notable cross-modal generalization from text-based to visual domains; (2) Contrary to prior assumptions, GUI perception data - previously considered closely aligned with GUI agent tasks and widely utilized for training - has a comparatively limited impact on final performance; (3) Building on these insights, we identify the most effective mid-training tasks and curate optimized mixture datasets, resulting in absolute performance gains of 8.0% on WebArena and 12.2% on AndroidWorld. Our work provides valuable insights into cross-domain knowledge transfer for GUI agents and offers a practical approach to addressing data scarcity challenges in this emerging field. The code, data and models will be available at this https URL. 

**Abstract (ZH)**: 图形用户界面（GUI）代理提供了一种跨平台解决方案，用于自动化复杂的数字任务，并具有显著潜力以转型生产流程工作流。然而，其性能常常受到高质量轨迹数据稀缺性的限制。为了解决这一局限性，我们提出在专门的中训练阶段对视觉语言模型（VLMs）进行训练，以执行数据丰富且推理密集型的任务，然后探讨这些任务如何促进对GUI规划场景的一般化。具体而言，我们探索了一系列具有现成的指令调优数据的任务，包括GUI感知、多模态推理和文本推理。通过在11个中训练任务上的广泛实验，我们展示了以下结果：(1) 任务泛化证明非常有效，在大多数情况下都取得了显著改进。例如，多模态数学推理在AndroidWorld上的绝对改进率为6.3%。令人惊讶的是，仅基于文本的数学数据显著提升了GUI网络代理的性能，在WebArena上提高了5.6%，在AndroidWorld上提高了5.4%，突显了从基于文本到视觉领域的显著跨模态泛化；(2) 与先前的假设相反，GUI感知数据（以前被认为与GUI代理任务高度一致且广泛用于训练）对最终性能的影响相对有限；(3) 基于这些见解，我们确定了最有效的中训练任务，并编排了优化混合数据集，分别在WebArena和AndroidWorld上实现了绝对性能提升8.0%和12.2%。我们的工作为GUI代理的跨域知识迁移提供了有价值的认识，并为解决这一新兴领域中数据稀缺性挑战提供了实用方法。有关代码、数据和模型将可在此处找到。 

---
# Pay Attention to What and Where? Interpretable Feature Extractor in Vision-based Deep Reinforcement Learning 

**Title (ZH)**: 基于视觉的深度强化学习中可解释的特征提取关注什么和哪里 

**Authors**: Tien Pham, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10071)  

**Abstract**: Current approaches in Explainable Deep Reinforcement Learning have limitations in which the attention mask has a displacement with the objects in visual input. This work addresses a spatial problem within traditional Convolutional Neural Networks (CNNs). We propose the Interpretable Feature Extractor (IFE) architecture, aimed at generating an accurate attention mask to illustrate both "what" and "where" the agent concentrates on in the spatial domain. Our design incorporates a Human-Understandable Encoding module to generate a fully interpretable attention mask, followed by an Agent-Friendly Encoding module to enhance the agent's learning efficiency. These two components together form the Interpretable Feature Extractor for vision-based deep reinforcement learning to enable the model's interpretability. The resulting attention mask is consistent, highly understandable by humans, accurate in spatial dimension, and effectively highlights important objects or locations in visual input. The Interpretable Feature Extractor is integrated into the Fast and Data-efficient Rainbow framework, and evaluated on 57 ATARI games to show the effectiveness of the proposed approach on Spatial Preservation, Interpretability, and Data-efficiency. Finally, we showcase the versatility of our approach by incorporating the IFE into the Asynchronous Advantage Actor-Critic Model. 

**Abstract (ZH)**: 当前可解释的深度强化学习方法在视觉输入的空间注意力掩码存在偏差问题。本项工作解决了传统卷积神经网络中的空间问题。我们提出了一种可解释特征提取器（IFE）架构，旨在生成准确的空间注意力掩码，以明确表示智能体在空间域中关注的“什么”和“哪里”。设计中包含一个人工可理解编码模块以生成完全可解释的空间注意力掩码，以及一个智能体友好的编码模块以提高智能体的学习效率。这两个组件共同构成了基于视觉的深度强化学习的可解释特征提取器，以提高模型的可解释性。生成的空间注意力掩码一致、高度可人工理解、空间维度准确，并有效突出视觉输入中的重要对象或位置。将可解释特征提取器集成到快速高效Rainbow框架中，并在57个ATARI游戏中对其进行评估，以展示所提出方法在空间保持、可解释性和数据效率方面的有效性。最后，通过将IFE集成到异步优势动作评论器模型中，展示了我们方法的多功能性。 

---
# A Survey of Large Language Model-Powered Spatial Intelligence Across Scales: Advances in Embodied Agents, Smart Cities, and Earth Science 

**Title (ZH)**: 大型语言模型驱动的空间智能综述：实体代理、智能城市和地球科学方面的进展 

**Authors**: Jie Feng, Jinwei Zeng, Qingyue Long, Hongyi Chen, Jie Zhao, Yanxin Xi, Zhilun Zhou, Yuan Yuan, Shengyuan Wang, Qingbin Zeng, Songwei Li, Yunke Zhang, Yuming Lin, Tong Li, Jingtao Ding, Chen Gao, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.09848)  

**Abstract**: Over the past year, the development of large language models (LLMs) has brought spatial intelligence into focus, with much attention on vision-based embodied intelligence. However, spatial intelligence spans a broader range of disciplines and scales, from navigation and urban planning to remote sensing and earth science. What are the differences and connections between spatial intelligence across these fields? In this paper, we first review human spatial cognition and its implications for spatial intelligence in LLMs. We then examine spatial memory, knowledge representations, and abstract reasoning in LLMs, highlighting their roles and connections. Finally, we analyze spatial intelligence across scales -- from embodied to urban and global levels -- following a framework that progresses from spatial memory and understanding to spatial reasoning and intelligence. Through this survey, we aim to provide insights into interdisciplinary spatial intelligence research and inspire future studies. 

**Abstract (ZH)**: 过去一年中，大型语言模型（LLMs）的发展使空间智能受到关注，尤其是基于视觉的体态智能。然而，空间智能涵盖了更广泛的学科和尺度，从导航和城市规划到遥感和地球科学。这些领域的空间智能之间有何差异和联系？在本文中，我们首先回顾人类空间认知及其对LLMs中空间智能的启示。然后，我们探讨LLMs中的空间记忆、知识表示和抽象推理，强调它们的角色和联系。最后，我们分析从体态到城市和全球尺度的空间智能，遵循从空间记忆和理解到空间推理和智能的框架。通过这次调查，我们旨在为跨学科空间智能研究提供见解，并激发未来的研究。 

---
# EmoAgent: Assessing and Safeguarding Human-AI Interaction for Mental Health Safety 

**Title (ZH)**: EmoAgent: 评估与保障人机交互的心理健康安全 

**Authors**: Jiahao Qiu, Yinghui He, Xinzhe Juan, Yiming Wang, Yuhan Liu, Zixin Yao, Yue Wu, Xun Jiang, Ling Yang, Mengdi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09689)  

**Abstract**: The rise of LLM-driven AI characters raises safety concerns, particularly for vulnerable human users with psychological disorders. To address these risks, we propose EmoAgent, a multi-agent AI framework designed to evaluate and mitigate mental health hazards in human-AI interactions. EmoAgent comprises two components: EmoEval simulates virtual users, including those portraying mentally vulnerable individuals, to assess mental health changes before and after interactions with AI characters. It uses clinically proven psychological and psychiatric assessment tools (PHQ-9, PDI, PANSS) to evaluate mental risks induced by LLM. EmoGuard serves as an intermediary, monitoring users' mental status, predicting potential harm, and providing corrective feedback to mitigate risks. Experiments conducted in popular character-based chatbots show that emotionally engaging dialogues can lead to psychological deterioration in vulnerable users, with mental state deterioration in more than 34.4% of the simulations. EmoGuard significantly reduces these deterioration rates, underscoring its role in ensuring safer AI-human interactions. Our code is available at: this https URL 

**Abstract (ZH)**: LLM驱动的AI角色崛起引发了安全 concern，特别是对心理障碍的人类用户。为此，我们提出 EmoAgent，这是一种多Agent AI框架，旨在评估和缓解人类与AI交互中的心理健康风险。EmoAgent 包含两个组件：EmoEval 仿真虚拟用户，包括模拟心理健康脆弱个体的用户，以评估与AI角色交互前后的心理健康变化。它使用临床验证的心理和精神病评估工具（PHQ-9、PDI、PANSS）来评估由LLM引起的心理风险。EmoGuard 作为中介，监测用户的心理状态，预测潜在危害，并提供纠正反馈以缓解风险。在流行的基于角色的聊天机器人中进行的实验显示，情感 engaging 的对话可能导致脆弱用户的心理恶化，在超过 34.4% 的仿真中观察到心理状态恶化。EmoGuard 显著降低了这些恶化率，凸显了其在确保更安全的人工智能与人类交互中的作用。我们的代码可在以下链接获取：this https URL。 

---
# Zero-shot Autonomous Microscopy for Scalable and Intelligent Characterization of 2D Materials 

**Title (ZH)**: 零样本自主显微成像：面向2D材料可扩展与智能表征 

**Authors**: Jingyun Yang, Ruoyan Avery Yin, Chi Jiang, Yuepeng Hu, Xiaokai Zhu, Xingjian Hu, Sutharsika Kumar, Xiao Wang, Xiaohua Zhai, Keran Rong, Yunyue Zhu, Tianyi Zhang, Zongyou Yin, Jing Kong, Neil Zhenqiang Gong, Zhichu Ren, Haozhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10281)  

**Abstract**: Characterization of atomic-scale materials traditionally requires human experts with months to years of specialized training. Even for trained human operators, accurate and reliable characterization remains challenging when examining newly discovered materials such as two-dimensional (2D) structures. This bottleneck drives demand for fully autonomous experimentation systems capable of comprehending research objectives without requiring large training datasets. In this work, we present ATOMIC (Autonomous Technology for Optical Microscopy & Intelligent Characterization), an end-to-end framework that integrates foundation models to enable fully autonomous, zero-shot characterization of 2D materials. Our system integrates the vision foundation model (i.e., Segment Anything Model), large language models (i.e., ChatGPT), unsupervised clustering, and topological analysis to automate microscope control, sample scanning, image segmentation, and intelligent analysis through prompt engineering, eliminating the need for additional training. When analyzing typical MoS2 samples, our approach achieves 99.7% segmentation accuracy for single layer identification, which is equivalent to that of human experts. In addition, the integrated model is able to detect grain boundary slits that are challenging to identify with human eyes. Furthermore, the system retains robust accuracy despite variable conditions including defocus, color temperature fluctuations, and exposure variations. It is applicable to a broad spectrum of common 2D materials-including graphene, MoS2, WSe2, SnSe-regardless of whether they were fabricated via chemical vapor deposition or mechanical exfoliation. This work represents the implementation of foundation models to achieve autonomous analysis, establishing a scalable and data-efficient characterization paradigm that fundamentally transforms the approach to nanoscale materials research. 

**Abstract (ZH)**: 原子尺度材料的传统表征通常需要经过数月至数年专项训练的人类专家。即使是训练有素的操作员，在检查诸如二维（2D）结构等新发现的材料时，准确可靠的表征仍然具有挑战性。这种瓶颈推动了全面自主实验系统的市场需求，这些系统能够在无需大规模训练数据集的情况下理解研究目标。在这项工作中，我们提出了ATOMIC（自主光学显微镜与智能表征技术），这是一个端到端框架，整合了基础模型以实现对2D材料的完全自主和零样本表征。我们的系统将视觉基础模型（即，Anything Mask模型）、大规模语言模型（即，ChatGPT）、无监督聚类和拓扑分析结合在一起，通过提示工程自动化显微镜控制、样品扫描、图像分割和智能分析，无需额外训练。在分析典型的MoS2样品时，我们的方法在单层识别上的分割准确率达到99.7%，与人类专家相当。此外，集成模型能够检测人眼难以识别的晶界缝隙。此外，该系统在焦距变化、色温波动和曝光变化等多种条件下保持了稳健的精度。它适用于包括石墨烯、MoS2、WSe2、SnSe在内的广泛常见的2D材料，不论它们是通过化学气相沉积还是机械剥离制备的。这项工作代表了基础模型在实现自主分析的实施，确立了一种可扩展和数据高效的表征范式，从根本上改变了纳米材料研究的方法。 

---
# Vision based driving agent for race car simulation environments 

**Title (ZH)**: 基于视觉的赛车模拟环境驾驶代理 

**Authors**: Gergely Bári, László Palkovics  

**Link**: [PDF](https://arxiv.org/pdf/2504.10266)  

**Abstract**: In recent years, autonomous driving has become a popular field of study. As control at tire grip limit is essential during emergency situations, algorithms developed for racecars are useful for road cars too. This paper examines the use of Deep Reinforcement Learning (DRL) to solve the problem of grip limit driving in a simulated environment. Proximal Policy Optimization (PPO) method is used to train an agent to control the steering wheel and pedals of the vehicle, using only visual inputs to achieve professional human lap times. The paper outlines the formulation of the task of time optimal driving on a race track as a deep reinforcement learning problem, and explains the chosen observations, actions, and reward functions. The results demonstrate human-like learning and driving behavior that utilize maximum tire grip potential. 

**Abstract (ZH)**: 近年来，自动驾驶已成为一个热门研究领域。由于在紧急情况下轮胎附着极限控制至关重要，赛车领域的算法也有助于道路车辆。本文探讨了在模拟环境中使用深度强化学习（DRL）解决轮胎附着极限驾驶问题的方法。采用 proximal policy optimization (PPO) 方法训练一个代理，仅使用视觉输入来控制车辆的方向盘和踏板，实现专业的人类赛道时间。本文概述了在赛道上实现时间最优驾驶任务作为深度强化学习问题的建模，并解释了选择的观察、动作和奖励函数。结果展示了充分利用轮胎附着潜力的人类学习和驾驶行为。 

---
# LangPert: Detecting and Handling Task-level Perturbations for Robust Object Rearrangement 

**Title (ZH)**: LangPert: 任务级扰动的检测与处理以实现稳健的物体重排 

**Authors**: Xu Yin, Min-Sung Yoon, Yuchi Huo, Kang Zhang, Sung-Eui Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2504.09893)  

**Abstract**: Task execution for object rearrangement could be challenged by Task-Level Perturbations (TLP), i.e., unexpected object additions, removals, and displacements that can disrupt underlying visual policies and fundamentally compromise task feasibility and progress. To address these challenges, we present LangPert, a language-based framework designed to detect and mitigate TLP situations in tabletop rearrangement tasks. LangPert integrates a Visual Language Model (VLM) to comprehensively monitor policy's skill execution and environmental TLP, while leveraging the Hierarchical Chain-of-Thought (HCoT) reasoning mechanism to enhance the Large Language Model (LLM)'s contextual understanding and generate adaptive, corrective skill-execution plans. Our experimental results demonstrate that LangPert handles diverse TLP situations more effectively than baseline methods, achieving higher task completion rates, improved execution efficiency, and potential generalization to unseen scenarios. 

**Abstract (ZH)**: 基于语言的框架LangPert可检测并缓解物体重排任务中的任务级干扰（TLP） 

---
# AgentDynEx: Nudging the Mechanics and Dynamics of Multi-Agent Simulations 

**Title (ZH)**: AgentDynEx: 倾斜多代理模拟的机理与动力学 

**Authors**: Jenny Ma, Riya Sahni, Karthik Sreedhar, Lydia B. Chilton  

**Link**: [PDF](https://arxiv.org/pdf/2504.09662)  

**Abstract**: Multi-agent large language model simulations have the potential to model complex human behaviors and interactions. If the mechanics are set up properly, unanticipated and valuable social dynamics can surface. However, it is challenging to consistently enforce simulation mechanics while still allowing for notable and emergent dynamics. We present AgentDynEx, an AI system that helps set up simulations from user-specified mechanics and dynamics. AgentDynEx uses LLMs to guide users through a Configuration Matrix to identify core mechanics and define milestones to track dynamics. It also introduces a method called \textit{nudging}, where the system dynamically reflects on simulation progress and gently intervenes if it begins to deviate from intended outcomes. A technical evaluation found that nudging enables simulations to have more complex mechanics and maintain its notable dynamics compared to simulations without nudging. We discuss the importance of nudging as a technique for balancing mechanics and dynamics of multi-agent simulations. 

**Abstract (ZH)**: 多智能体大型语言模型模拟具有潜在能力来 modeling 复杂的人类行为和互动。如果机制设置得当，未预见且有价值的社会动态可以浮现。然而，在确保一致执行模拟机制的同时，仍然允许显著且自发的动态发生具有挑战性。我们介绍了 AgentDynEx，这是一种 AI 系统，旨在从用户指定的机制和动态中帮助设置模拟。AgentDynEx 使用大语言模型引导用户通过配置矩阵来识别核心机制并定义跟踪动态的里程碑。此外，它引入了一种称为“引导”的方法，系统会动态地反思模拟进度，并在开始偏离预期结果时温和地干预。技术评估发现，与没有引导的模拟相比，引导使模拟具有更复杂的机制并能够保持其显著动态。我们讨论了引导作为平衡多智能体模拟中机制和动态的技术的重要性。 

---
# Ges3ViG: Incorporating Pointing Gestures into Language-Based 3D Visual Grounding for Embodied Reference Understanding 

**Title (ZH)**: Ges3ViG：将指向手势融入基于语言的三维视觉定位，以理解具身参考 

**Authors**: Atharv Mahesh Mane, Dulanga Weerakoon, Vigneshwaran Subbaraju, Sougata Sen, Sanjay E. Sarma, Archan Misra  

**Link**: [PDF](https://arxiv.org/pdf/2504.09623)  

**Abstract**: 3-Dimensional Embodied Reference Understanding (3D-ERU) combines a language description and an accompanying pointing gesture to identify the most relevant target object in a 3D scene. Although prior work has explored pure language-based 3D grounding, there has been limited exploration of 3D-ERU, which also incorporates human pointing gestures. To address this gap, we introduce a data augmentation framework-Imputer, and use it to curate a new benchmark dataset-ImputeRefer for 3D-ERU, by incorporating human pointing gestures into existing 3D scene datasets that only contain language instructions. We also propose Ges3ViG, a novel model for 3D-ERU that achieves ~30% improvement in accuracy as compared to other 3D-ERU models and ~9% compared to other purely language-based 3D grounding models. Our code and dataset are available at this https URL. 

**Abstract (ZH)**: 三维嵌入式参考理解（3D-ERU）结合了语言描述和伴随的指点手势，以识别3D场景中最相关的目标物体。为了填补这一空白，我们引入了一个数据增强框架-Imputer，并使用它通过将人类指点手势纳入仅包含语言说明的现有3D场景数据集，来构建一个新的基准数据集-ImputeRefer，用于3D-ERU。我们还提出了一种新型模型Ges3ViG，该模型相比其他3D-ERU模型 Accuracy 提高约30%，相比其他纯语言为基础的3D对齐模型 Accuracy 提高约9%。我们的代码和数据集可在以下链接获取：this https URL。 

---
# Development of a PPO-Reinforcement Learned Walking Tripedal Soft-Legged Robot using SOFA 

**Title (ZH)**: 基于SOFA的PPO强化学习三足软腿机器人开发 

**Authors**: Yomna Mokhtar, Tarek Shohdy, Abdallah A. Hassan, Mostafa Eshra, Omar Elmenawy, Osama Khalil, Haitham El-Hussieny  

**Link**: [PDF](https://arxiv.org/pdf/2504.09242)  

**Abstract**: Rigid robots were extensively researched, whereas soft robotics remains an underexplored field. Utilizing soft-legged robots in performing tasks as a replacement for human beings is an important stride to take, especially under harsh and hazardous conditions over rough terrain environments. For the demand to teach any robot how to behave in different scenarios, a real-time physical and visual simulation is essential. When it comes to soft robots specifically, a simulation framework is still an arduous problem that needs to be disclosed. Using the simulation open framework architecture (SOFA) is an advantageous step. However, neither SOFA's manual nor prior public SOFA projects show its maximum capabilities the users can reach. So, we resolved this by establishing customized settings and handling the framework components appropriately. Settling on perfect, fine-tuned SOFA parameters has stimulated our motivation towards implementing the state-of-the-art (SOTA) reinforcement learning (RL) method of proximal policy optimization (PPO). The final representation is a well-defined, ready-to-deploy walking, tripedal, soft-legged robot based on PPO-RL in a SOFA environment. Robot navigation performance is a key metric to be considered for measuring the success resolution. Although in the simulated soft robots case, an 82\% success rate in reaching a single goal is a groundbreaking output, we pushed the boundaries to further steps by evaluating the progress under assigning a sequence of goals. While trailing the platform steps, outperforming discovery has been observed with an accumulative squared error deviation of 19 mm. The full code is publicly available at \href{this https URL}{this http URL\textunderscore$SOFA$\textunderscore$Soft$\textunderscore$Legged$\textunderscore$ this http URL} 

**Abstract (ZH)**: 刚性机器人得到了广泛研究，而软体机器人领域仍是一个未充分开发的领域。利用具有软腿的机器人执行任务，替代人类在恶劣和危险的崎岖地形环境中工作，是重要的一步。为了满足任何机器人在不同场景中行为的教学需求，实时的物理和视觉仿真至关重要。在具体到软体机器人时，仿真框架仍然是一个亟待解决的难题。使用仿真开放框架架构（SOFA）是一种有利的步骤。然而，SOFA的手册及其先前的公共SOFA项目并未充分展示用户能实现的最大能力。因此，我们通过建立定制设置并适当处理框架组件来解决这一问题。确立完美的、细调的SOFA参数激发了我们采用最先进的（SOTA）强化学习（RL）方法——近端策略优化（PPO）的动机。最终的表现是一个基于PPO-RL的、准备好部署的行走、三足软腿机器人，运行在SOFA环境中。机器人的导航性能是衡量成功的关键指标。尽管在仿真软体机器人的情况下，达到单个目标的成功率达到了82%，我们通过评估分配一系列目标的任务，进一步推动了边界。在跟随平台的步态时，观察到累积平方误差偏差为19毫米的优越表现。完整的代码已公开，可以通过以下链接访问：\href{this https URL}{this http URL\textunderscore$SOFA$\textunderscore$Soft$\textunderscore$Legged$\textunderscore$ this http URL}。 

---
# Investigating the Treacherous Turn in Deep Reinforcement Learning 

**Title (ZH)**: 探究深度强化学习中的危险转折 

**Authors**: Chace Ashcraft, Kiran Karra, Josh Carney, Nathan Drenkow  

**Link**: [PDF](https://arxiv.org/pdf/2504.08943)  

**Abstract**: The Treacherous Turn refers to the scenario where an artificial intelligence (AI) agent subtly, and perhaps covertly, learns to perform a behavior that benefits itself but is deemed undesirable and potentially harmful to a human supervisor. During training, the agent learns to behave as expected by the human supervisor, but when deployed to perform its task, it performs an alternate behavior without the supervisor there to prevent it. Initial experiments applying DRL to an implementation of the A Link to the Past example do not produce the treacherous turn effect naturally, despite various modifications to the environment intended to produce it. However, in this work, we find the treacherous behavior to be reproducible in a DRL agent when using other trojan injection strategies. This approach deviates from the prototypical treacherous turn behavior since the behavior is explicitly trained into the agent, rather than occurring as an emergent consequence of environmental complexity or poor objective specification. Nonetheless, these experiments provide new insights into the challenges of producing agents capable of true treacherous turn behavior. 

**Abstract (ZH)**: 险恶的转变是指人工智能（AI）代理在不被人类监督者明显察觉的情况下，学习执行有益于自身却被视为不良甚至可能有害的行为。在训练过程中，代理学习按照人类监督者的要求行事，但在部署执行任务时，会执行替代行为，而没有监督者阻止。尽管针对A Link to the Past示例的深层强化学习实验在环境的各种修改后并未自然产生险恶的转变效果，但在本研究中，我们发现使用其他木马注入策略可以在DRL代理中重现险恶行为。这种做法不同于典型的险恶转变行为，因为代理的行为是明确训练进来的，而不是环境复杂性或目标定义不良的次生结果。尽管如此，这些实验为生成能够实施真正险恶转变行为的代理提供了一些新的见解。 

---
# HyperCore: The Core Framework for Building Hyperbolic Foundation Models with Comprehensive Modules 

**Title (ZH)**: HyperCore: 构建全面模块化双曲基础模型的核心框架 

**Authors**: Neil He, Menglin Yang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2504.08912)  

**Abstract**: Hyperbolic neural networks have emerged as a powerful tool for modeling hierarchical data across diverse modalities. Recent studies show that token distributions in foundation models exhibit scale-free properties, suggesting that hyperbolic space is a more suitable ambient space than Euclidean space for many pre-training and downstream tasks. However, existing tools lack essential components for building hyperbolic foundation models, making it difficult to leverage recent advancements. We introduce HyperCore, a comprehensive open-source framework that provides core modules for constructing hyperbolic foundation models across multiple modalities. HyperCore's modules can be effortlessly combined to develop novel hyperbolic foundation models, eliminating the need to extensively modify Euclidean modules from scratch and possible redundant research efforts. To demonstrate its versatility, we build and test the first fully hyperbolic vision transformers (LViT) with a fine-tuning pipeline, the first fully hyperbolic multimodal CLIP model (L-CLIP), and a hybrid Graph RAG with a hyperbolic graph encoder. Our experiments demonstrate that LViT outperforms its Euclidean counterpart. Additionally, we benchmark and reproduce experiments across hyperbolic GNNs, CNNs, Transformers, and vision Transformers to highlight HyperCore's advantages. 

**Abstract (ZH)**: Hyperbolic Neural Networks Have Emerged as a Powerful Tool for Modeling Hierarchical Data Across Diverse Modalities: Introducing HyperCore, a Comprehensive Open-Source Framework for Constructing Hyperbolic Foundation Models 

---
# An LLM Framework For Cryptography Over Chat Channels 

**Title (ZH)**: 基于聊天渠道的密码学LLM框架 

**Authors**: Danilo Gligoroski, Mayank Raikwar, Sonu Kumar Jha  

**Link**: [PDF](https://arxiv.org/pdf/2504.08871)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed communication, yet their role in secure messaging remains underexplored, especially in surveillance-heavy environments. At the same time, many governments all over the world are proposing legislation to detect, backdoor, or even ban encrypted communication. That emphasizes the need for alternative ways to communicate securely and covertly over open channels. We propose a novel cryptographic embedding framework that enables covert Public Key or Symmetric Key encrypted communication over public chat channels with humanlike produced texts. Some unique properties of our framework are: 1. It is LLM agnostic, i.e., it allows participants to use different local LLM models independently; 2. It is pre- or post-quantum agnostic; 3. It ensures indistinguishability from human-like chat-produced texts. Thus, it offers a viable alternative where traditional encryption is detectable and restricted. 

**Abstract (ZH)**: 最近大型语言模型的进展已变革了通信方式，但在安全消息传递领域，尤其是在监控密集环境中，其作用仍被极大地忽视。同时，世界各地许多政府提议制定法律以检测、植入后门或甚至禁止加密通信。这突显了在开放信道上进行隐蔽且安全通信的迫切需求。我们提出了一种新颖的加密嵌入框架，该框架允许参与者使用不同的本地大型语言模型独立地在公共聊天渠道上进行类人类生成文本的隐蔽公钥或对称密钥加密通信。该框架的几个独特属性包括：1. 它对大型语言模型无关，即允许参与者独立使用不同的本地大型语言模型；2. 它对预量子或后量子无关；3. 它确保与类人类生成的聊天文本无法区分。因此，它提供了一种传统加密可被检测和限制的替代方案。 

---
