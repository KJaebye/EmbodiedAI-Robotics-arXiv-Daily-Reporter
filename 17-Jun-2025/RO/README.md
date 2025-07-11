# Touch begins where vision ends: Generalizable policies for contact-rich manipulation 

**Title (ZH)**: 触觉始于视觉终结：接触丰富的操作通用策略 

**Authors**: Zifan Zhao, Siddhant Haldar, Jinda Cui, Lerrel Pinto, Raunaq Bhirangi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13762)  

**Abstract**: Data-driven approaches struggle with precise manipulation; imitation learning requires many hard-to-obtain demonstrations, while reinforcement learning yields brittle, non-generalizable policies. We introduce VisuoTactile Local (ViTaL) policy learning, a framework that solves fine-grained manipulation tasks by decomposing them into two phases: a reaching phase, where a vision-language model (VLM) enables scene-level reasoning to localize the object of interest, and a local interaction phase, where a reusable, scene-agnostic ViTaL policy performs contact-rich manipulation using egocentric vision and tactile sensing. This approach is motivated by the observation that while scene context varies, the low-level interaction remains consistent across task instances. By training local policies once in a canonical setting, they can generalize via a localize-then-execute strategy. ViTaL achieves around 90% success on contact-rich tasks in unseen environments and is robust to distractors. ViTaL's effectiveness stems from three key insights: (1) foundation models for segmentation enable training robust visual encoders via behavior cloning; (2) these encoders improve the generalizability of policies learned using residual RL; and (3) tactile sensing significantly boosts performance in contact-rich tasks. Ablation studies validate each of these insights, and we demonstrate that ViTaL integrates well with high-level VLMs, enabling robust, reusable low-level skills. Results and videos are available at this https URL. 

**Abstract (ZH)**: 基于视觉-触觉局部策略学习：细粒度操作任务的两阶段方法 

---
# Prompting with the Future: Open-World Model Predictive Control with Interactive Digital Twins 

**Title (ZH)**: 面向未来的Prompting：具有互动数字孪生的开放世界模型预测控制 

**Authors**: Chuanruo Ning, Kuan Fang, Wei-Chiu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.13761)  

**Abstract**: Recent advancements in open-world robot manipulation have been largely driven by vision-language models (VLMs). While these models exhibit strong generalization ability in high-level planning, they struggle to predict low-level robot controls due to limited physical-world understanding. To address this issue, we propose a model predictive control framework for open-world manipulation that combines the semantic reasoning capabilities of VLMs with physically-grounded, interactive digital twins of the real-world environments. By constructing and simulating the digital twins, our approach generates feasible motion trajectories, simulates corresponding outcomes, and prompts the VLM with future observations to evaluate and select the most suitable outcome based on language instructions of the task. To further enhance the capability of pre-trained VLMs in understanding complex scenes for robotic control, we leverage the flexible rendering capabilities of the digital twin to synthesize the scene at various novel, unoccluded viewpoints. We validate our approach on a diverse set of complex manipulation tasks, demonstrating superior performance compared to baseline methods for language-conditioned robotic control using VLMs. 

**Abstract (ZH)**: 开放世界机器人操作中的 recent 进展主要得益于视觉语言模型（VLMs）。为了克服这些模型在低级机器人控制方面因物理世界理解有限而表现出的不足，我们提出了一种结合 VLMs 的语义推理能力和基于物理的交互数字孪生的开放世界操控模型预测控制框架。通过构建和模拟数字孪生，我们的方法生成可行的运动轨迹，模拟相应的结果，并通过任务的语言指令提示 VLMs 来评估和选择最合适的结局。为了进一步增强预训练 VLMs 在理解复杂场景以进行机器人控制的能力，我们利用数字孪生的灵活渲染能力合成各种新颖、无遮挡视角的场景。我们在一系列复杂操控任务上验证了该方法，展示了在基于 VLMs 的语言条件robots控制方面相较于基线方法的优越性能。 

---
# Edge Nearest Neighbor in Sampling-Based Motion Planning 

**Title (ZH)**: 基于采样法运动规划中的边最近邻方法 

**Authors**: Stav Ashur, Nancy M. Amato, Sariel Har-Peled  

**Link**: [PDF](https://arxiv.org/pdf/2506.13753)  

**Abstract**: Neighborhood finders and nearest neighbor queries are fundamental parts of sampling based motion planning algorithms. Using different distance metrics or otherwise changing the definition of a neighborhood produces different algorithms with unique empiric and theoretical properties. In \cite{l-pa-06} LaValle suggests a neighborhood finder for the Rapidly-exploring Random Tree RRT
algorithm \cite{l-rrtnt-98} which finds the nearest neighbor of the sampled point on the swath of the tree, that is on the set of all of the points on the tree edges, using a hierarchical data structure. In this paper we implement such a neighborhood finder and show, theoretically and experimentally, that this results in more efficient algorithms, and suggest a variant of the Rapidly-exploring Random Graph RRG algorithm \cite{f-isaom-10} that better exploits the exploration properties of the newly described subroutine for finding narrow passages. 

**Abstract (ZH)**: 基于采样方法的运动规划算法中，邻居查找器和最近邻查询是基本组成部分。使用不同的距离度量或重新定义邻居的定义会产生具有独特经验和理论性质的不同算法。在LaValle的《Probabilistic Robotics》（2006）中，他建议了一种适用于快速探索随机树RRT（1998）算法的邻居查找器，该查找器使用层次数据结构在树的路径上的所有点集中找到采样点的最近邻。本文实现了这样的邻居查找器，并从理论和实验上证明这可以提高算法效率，并提出了一种改进的快速探索随机图RRG算法（2010），更好地利用了新描述的查找狭窄通道的子程序的探索性质。 

---
# LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction 

**Title (ZH)**: LeVERB: 具有潜在视觉-语言指令的 humanoid 整体身体控制 

**Authors**: Haoru Xue, Xiaoyu Huang, Dantong Niu, Qiayuan Liao, Thomas Kragerud, Jan Tommy Gravdahl, Xue Bin Peng, Guanya Shi, Trevor Darrell, Koushil Screenath, Shankar Sastry  

**Link**: [PDF](https://arxiv.org/pdf/2506.13751)  

**Abstract**: Vision-language-action (VLA) models have demonstrated strong semantic understanding and zero-shot generalization, yet most existing systems assume an accurate low-level controller with hand-crafted action "vocabulary" such as end-effector pose or root velocity. This assumption confines prior work to quasi-static tasks and precludes the agile, whole-body behaviors required by humanoid whole-body control (WBC) tasks. To capture this gap in the literature, we start by introducing the first sim-to-real-ready, vision-language, closed-loop benchmark for humanoid WBC, comprising over 150 tasks from 10 categories. We then propose LeVERB: Latent Vision-Language-Encoded Robot Behavior, a hierarchical latent instruction-following framework for humanoid vision-language WBC, the first of its kind. At the top level, a vision-language policy learns a latent action vocabulary from synthetically rendered kinematic demonstrations; at the low level, a reinforcement-learned WBC policy consumes these latent verbs to generate dynamics-level commands. In our benchmark, LeVERB can zero-shot attain a 80% success rate on simple visual navigation tasks, and 58.5% success rate overall, outperforming naive hierarchical whole-body VLA implementation by 7.8 times. 

**Abstract (ZH)**: 视觉-语言-动作（VLA）模型已经展示了强大的语义理解和零样本泛化能力，但现有的大多数系统假设一个准确的低级控制器，并使用手工制作的动作“词汇表”，如末端执行器姿态或根速度。这种假设限制了以往工作仅适用于准静态任务，并排除了类人全身控制（WBC）任务所需的敏捷的全身行为。为填补这一文献空白，我们首先介绍了第一个适用于现实的视觉-语言-闭环类人WBC基准，包含来自10个类别超过150项任务。然后，我们提出了LeVERB：潜藏的视觉-语言-编码的机器人行为，这是一种用于类人视觉-语言WBC的分层潜藏指令跟随框架，开创了先河。在最高层，视觉-语言策略从合成渲染的动力学示范中学习一个潜藏的动作词汇表；在低层，一个通过强化学习训练的WBC策略消耗这些潜藏的动词生成动力学级的命令。在我们的基准中，LeVERB在简单的视觉导航任务上达到80%的零样本成功率，并且总体上达到58.5%的成功率，比简单的分层全身VLA实现高出7.8倍。 

---
# Critical Insights about Robots for Mental Wellbeing 

**Title (ZH)**: 关于用于心理健康的人工智能机器人的重要见解 

**Authors**: Guy Laban, Micol Spitale, Minja Axelsson, Nida Itrat Abbasi, Hatice Gunes  

**Link**: [PDF](https://arxiv.org/pdf/2506.13739)  

**Abstract**: Social robots are increasingly being explored as tools to support emotional wellbeing, particularly in non-clinical settings. Drawing on a range of empirical studies and practical deployments, this paper outlines six key insights that highlight both the opportunities and challenges in using robots to promote mental wellbeing. These include (1) the lack of a single, objective measure of wellbeing, (2) the fact that robots don't need to act as companions to be effective, (3) the growing potential of virtual interactions, (4) the importance of involving clinicians in the design process, (5) the difference between one-off and long-term interactions, and (6) the idea that adaptation and personalization are not always necessary for positive outcomes. Rather than positioning robots as replacements for human therapists, we argue that they are best understood as supportive tools that must be designed with care, grounded in evidence, and shaped by ethical and psychological considerations. Our aim is to inform future research and guide responsible, effective use of robots in mental health and wellbeing contexts. 

**Abstract (ZH)**: 社会机器人在促进心理健康和福祉方面的机会与挑战：基于实证研究与实际部署的六点洞见 

---
# CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding 

**Title (ZH)**: CEED-VLA：具早期退出解码的一致性跨模态模型 

**Authors**: Wenxuan Song, Jiayi Chen, Pengxiang Ding, Yuxin Huang, Han Zhao, Donglin Wang, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13725)  

**Abstract**: In recent years, Vision-Language-Action (VLA) models have become a vital research direction in robotics due to their impressive multimodal understanding and generalization capabilities. Despite the progress, their practical deployment is severely constrained by inference speed bottlenecks, particularly in high-frequency and dexterous manipulation tasks. While recent studies have explored Jacobi decoding as a more efficient alternative to traditional autoregressive decoding, its practical benefits are marginal due to the lengthy iterations. To address it, we introduce consistency distillation training to predict multiple correct action tokens in each iteration, thereby achieving acceleration. Besides, we design mixed-label supervision to mitigate the error accumulation during distillation. Although distillation brings acceptable speedup, we identify that certain inefficient iterations remain a critical bottleneck. To tackle this, we propose an early-exit decoding strategy that moderately relaxes convergence conditions, which further improves average inference efficiency. Experimental results show that the proposed method achieves more than 4 times inference acceleration across different baselines while maintaining high task success rates in both simulated and real-world robot tasks. These experiments validate that our approach provides an efficient and general paradigm for accelerating multimodal decision-making in robotics. Our project page is available at this https URL. 

**Abstract (ZH)**: 近年来，视觉-语言-动作（VLA）模型由于其令人印象深刻的跨模态理解和泛化能力，已经成为机器人领域的一个重要研究方向。尽管取得了进展，但在高频灵巧操作任务中，其实际部署仍然受到推理速度瓶颈的严重制约。虽然近期研究探索了雅可比解码作为传统自回归解码的更高效替代方法，但由于迭代过程较长，其实际优势有限。为了解决这一问题，我们引入一致性蒸馏训练，在每次迭代中预测多个正确的动作令牌，从而实现加速。此外，我们设计了混合标签监督，以减轻蒸馏过程中的错误累积。尽管蒸馏带来了可接受的加速，但我们发现某些不高效的迭代仍然是关键瓶颈。为解决这一问题，我们提出了一种早期退出解码策略，适度放宽收敛条件，从而进一步提高平均推理效率。实验结果表明，所提出的方法在不同baseline上实现了超过4倍的推理加速，同时在模拟和真实世界机器人任务中保持了较高的任务成功率。这些实验验证了我们的方法为机器人领域加速跨模态决策提供了一种高效且通用的范式。我们的项目页面可以在以下链接访问：this https URL。 

---
# HARMONI: Haptic-Guided Assistance for Unified Robotic Tele-Manipulation and Tele-Navigation 

**Title (ZH)**: HARMONI: 耦合触觉引导的统一机器人远程操作与导航辅助 

**Authors**: V. Sripada, A. Khan, J. Föcker, S. Parsa, Susmitha P, H Maior, A. Ghalamzan-E  

**Link**: [PDF](https://arxiv.org/pdf/2506.13704)  

**Abstract**: Shared control, which combines human expertise with autonomous assistance, is critical for effective teleoperation in complex environments. While recent advances in haptic-guided teleoperation have shown promise, they are often limited to simplified tasks involving 6- or 7-DoF manipulators and rely on separate control strategies for navigation and manipulation. This increases both cognitive load and operational overhead. In this paper, we present a unified tele-mobile manipulation framework that leverages haptic-guided shared control. The system integrates a 9-DoF follower mobile manipulator and a 7-DoF leader robotic arm, enabling seamless transitions between tele-navigation and tele-manipulation through real-time haptic feedback. A user study with 20 participants under real-world conditions demonstrates that our framework significantly improves task accuracy and efficiency without increasing cognitive load. These findings highlight the potential of haptic-guided shared control for enhancing operator performance in demanding teleoperation scenarios. 

**Abstract (ZH)**: 基于触觉引导的协同控制的统一远程移动操作框架 

---
# ROSA: Harnessing Robot States for Vision-Language and Action Alignment 

**Title (ZH)**: ROSA: 利用机器人状态实现视觉-语言和行动对齐 

**Authors**: Yuqing Wen, Kefan Gu, Haoxuan Liu, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.13679)  

**Abstract**: Vision-Language-Action (VLA) models have recently made significant advance in multi-task, end-to-end robotic control, due to the strong generalization capabilities of Vision-Language Models (VLMs). A fundamental challenge in developing such models is effectively aligning the vision-language space with the robotic action space. Existing approaches typically rely on directly fine-tuning VLMs using expert demonstrations. However, this strategy suffers from a spatio-temporal gap, resulting in considerable data inefficiency and heavy reliance on human labor. Spatially, VLMs operate within a high-level semantic space, whereas robotic actions are grounded in low-level 3D physical space; temporally, VLMs primarily interpret the present, while VLA models anticipate future actions. To overcome these challenges, we propose a novel training paradigm, ROSA, which leverages robot state estimation to improve alignment between vision-language and action spaces. By integrating robot state estimation data obtained via an automated process, ROSA enables the VLA model to gain enhanced spatial understanding and self-awareness, thereby boosting performance and generalization. Extensive experiments in both simulated and real-world environments demonstrate the effectiveness of ROSA, particularly in low-data regimes. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型在多任务、端到端的机器人控制中取得了显著进展，这得益于视觉语言模型（VLMs）的强大泛化能力。开发这类模型的一个基本挑战是如何有效地将视觉语言空间与机器人动作空间对齐。现有方法通常依赖于直接 fine-tuning VLMs 使用专家演示。然而，这种策略导致了时空差距，从而导致了数据效率低下和对人力的高度依赖。在空间上，VLMs 操作于高级语义空间，而机器人动作则根植于低级 3D 物理空间；在时间上，VLMs 主要解释现状，而 VLA 模型预测未来动作。为克服这些挑战，我们提出了一种新的训练范式 ROSA，该范式利用机器人状态估计来提高视觉语言空间与动作空间的对齐。通过结合通过自动化过程获得的机器人状态估计数据，ROSA 使 VLA 模型获得增强的空间理解能力和自我意识，从而提升性能和泛化能力。在模拟和真实环境中的 extensive 实验表明，ROSA 在低数据情况下尤其有效。 

---
# Towards Efficient Occupancy Mapping via Gaussian Process Latent Field Shaping 

**Title (ZH)**: 基于高斯过程潜在场塑形的高效占用映射 

**Authors**: Cedric Le Gentil, Cedric Pradalier, Timothy D. Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2506.13640)  

**Abstract**: Occupancy mapping has been a key enabler of mobile robotics. Originally based on a discrete grid representation, occupancy mapping has evolved towards continuous representations that can predict the occupancy status at any location and account for occupancy correlations between neighbouring areas. Gaussian Process (GP) approaches treat this task as a binary classification problem using both observations of occupied and free space. Conceptually, a GP latent field is passed through a logistic function to obtain the output class without actually manipulating the GP latent field. In this work, we propose to act directly on the latent function to efficiently integrate free space information as a prior based on the shape of the sensor's field-of-view. A major difference with existing methods is the change in the classification problem, as we distinguish between free and unknown space. The `occupied' area is the infinitesimally thin location where the class transitions from free to unknown. We demonstrate in simulated environments that our approach is sound and leads to competitive reconstruction accuracy. 

**Abstract (ZH)**: occupancy 状态映射是移动机器人技术的关键推动因素。最初基于离散栅格表示，occupancy 状态映射已发展为可以预测任何位置的occupancy状态并考虑相邻区域occupancy相关性的连续表示。高斯过程（GP）方法将此任务视为二分类问题，利用占用空间和空闲空间的观测数据。概念上，通过逻辑函数处理GP潜在场以获得输出类别，而不实际操作GP潜在场。在本工作中，我们提出直接作用于潜在函数，利用传感器视场形状有效地整合空闲空间信息作为先验。与现有方法的主要区别在于分类问题的变化，因为我们将空闲空间与未知空间区分开来。'占用'区域是类别从空闲过渡到未知的无穷小位置。我们在模拟环境中展示了我们方法的有效性和竞争力的重建精度。 

---
# Disturbance-aware minimum-time planning strategies for motorsport vehicles with probabilistic safety certificates 

**Title (ZH)**: 基于扰动感知的最短时间规划策略及其概率安全证书研究 

**Authors**: Martino Gulisano, Matteo Masoni, Marco Gabiccini, Massimo Guiggiani  

**Link**: [PDF](https://arxiv.org/pdf/2506.13622)  

**Abstract**: This paper presents a disturbance-aware framework that embeds robustness into minimum-lap-time trajectory optimization for motorsport. Two formulations are introduced. (i) Open-loop, horizon-based covariance propagation uses worst-case uncertainty growth over a finite window to tighten tire-friction and track-limit constraints. (ii) Closed-loop, covariance-aware planning incorporates a time-varying LQR feedback law in the optimizer, providing a feedback-consistent estimate of disturbance attenuation and enabling sharper yet reliable constraint tightening. Both methods yield reference trajectories for human or artificial drivers: in autonomous applications the modelled controller can replicate the on-board implementation, while for human driving accuracy increases with the extent to which the driver can be approximated by the assumed time-varying LQR policy. Computational tests on a representative Barcelona-Catalunya sector show that both schemes meet the prescribed safety probability, yet the closed-loop variant incurs smaller lap-time penalties than the more conservative open-loop solution, while the nominal (non-robust) trajectory remains infeasible under the same uncertainties. By accounting for uncertainty growth and feedback action during planning, the proposed framework delivers trajectories that are both performance-optimal and probabilistically safe, advancing minimum-time optimization toward real-world deployment in high-performance motorsport and autonomous racing. 

**Abstract (ZH)**: 基于干扰感知的最小圈时轨迹优化鲁棒性框架：开环与闭环方法 

---
# What Matters in Learning from Large-Scale Datasets for Robot Manipulation 

**Title (ZH)**: 大规模数据集用于机器人操作学习中值得关注的问题 

**Authors**: Vaibhav Saxena, Matthew Bronars, Nadun Ranawaka Arachchige, Kuancheng Wang, Woo Chul Shin, Soroush Nasiriany, Ajay Mandlekar, Danfei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13536)  

**Abstract**: Imitation learning from large multi-task demonstration datasets has emerged as a promising path for building generally-capable robots. As a result, 1000s of hours have been spent on building such large-scale datasets around the globe. Despite the continuous growth of such efforts, we still lack a systematic understanding of what data should be collected to improve the utility of a robotics dataset and facilitate downstream policy learning. In this work, we conduct a large-scale dataset composition study to answer this question. We develop a data generation framework to procedurally emulate common sources of diversity in existing datasets (such as sensor placements and object types and arrangements), and use it to generate large-scale robot datasets with controlled compositions, enabling a suite of dataset composition studies that would be prohibitively expensive in the real world. We focus on two practical settings: (1) what types of diversity should be emphasized when future researchers collect large-scale datasets for robotics, and (2) how should current practitioners retrieve relevant demonstrations from existing datasets to maximize downstream policy performance on tasks of interest. Our study yields several critical insights -- for example, we find that camera poses and spatial arrangements are crucial dimensions for both diversity in collection and alignment in retrieval. In real-world robot learning settings, we find that not only do our insights from simulation carry over, but our retrieval strategies on existing datasets such as DROID allow us to consistently outperform existing training strategies by up to 70%. More results at this https URL 

**Abstract (ZH)**: 从大规模多任务演示数据集中学习模仿：构建通用机器人的一条有前途的道路 

---
# A Survey on Imitation Learning for Contact-Rich Tasks in Robotics 

**Title (ZH)**: 机器人领域接触密集型任务的imitation learning综述 

**Authors**: Toshiaki Tsuji, Yasuhiro Kato, Gokhan Solak, Heng Zhang, Tadej Petrič, Francesco Nori, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2506.13498)  

**Abstract**: This paper comprehensively surveys research trends in imitation learning for contact-rich robotic tasks. Contact-rich tasks, which require complex physical interactions with the environment, represent a central challenge in robotics due to their nonlinear dynamics and sensitivity to small positional deviations. The paper examines demonstration collection methodologies, including teaching methods and sensory modalities crucial for capturing subtle interaction dynamics. We then analyze imitation learning approaches, highlighting their applications to contact-rich manipulation. Recent advances in multimodal learning and foundation models have significantly enhanced performance in complex contact tasks across industrial, household, and healthcare domains. Through systematic organization of current research and identification of challenges, this survey provides a foundation for future advancements in contact-rich robotic manipulation. 

**Abstract (ZH)**: 本文全面综述了模仿学习在接触丰富型机器人任务中的研究趋势。接触丰富型任务由于其非线性动力学特性和对小位置偏差的高度敏感性，要求与环境进行复杂的物理交互，构成了机器人技术中的核心挑战。本文考察了演示收集方法，包括教学方法和用于捕捉微妙交互动力学的关键感官模态。我们随后分析了模仿学习方法，突出了其在接触丰富型操作中的应用。近年来，多模态学习和基础模型的进展显著提升了工业、家庭和医疗保健领域复杂接触任务的性能。通过对当前研究的系统组织和挑战的识别，本文提供了一个推动接触丰富型机器人操作未来发展的基础。 

---
# Learning Swing-up Maneuvers for a Suspended Aerial Manipulation Platform in a Hierarchical Control Framework 

**Title (ZH)**: 基于层次控制框架的悬空操作平台摆动起立动作学习 

**Authors**: Hemjyoti Das, Minh Nhat Vu, Christian Ott  

**Link**: [PDF](https://arxiv.org/pdf/2506.13478)  

**Abstract**: In this work, we present a novel approach to augment a model-based control method with a reinforcement learning (RL) agent and demonstrate a swing-up maneuver with a suspended aerial manipulation platform. These platforms are targeted towards a wide range of applications on construction sites involving cranes, with swing-up maneuvers allowing it to perch at a given location, inaccessible with purely the thrust force of the platform. Our proposed approach is based on a hierarchical control framework, which allows different tasks to be executed according to their assigned priorities. An RL agent is then subsequently utilized to adjust the reference set-point of the lower-priority tasks to perform the swing-up maneuver, which is confined in the nullspace of the higher-priority tasks, such as maintaining a specific orientation and position of the end-effector. Our approach is validated using extensive numerical simulation studies. 

**Abstract (ZH)**: 基于强化学习的模型驱动控制方法在悬停空中操作平台 perch 操作中的应用研究 

---
# Towards a Formal Specification for Self-organized Shape Formation in Swarm Robotics 

**Title (ZH)**: 面向群机器人自组织形态形成的正式规范研究 

**Authors**: YR Darr, MA Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13453)  

**Abstract**: The self-organization of robots for the formation of structures and shapes is a stimulating application of the swarm robotic system. It involves a large number of autonomous robots of heterogeneous behavior, coordination among them, and their interaction with the dynamic environment. This process of complex structure formation is considered a complex system, which needs to be modeled by using any modeling approach. Although the formal specification approach along with other formal methods has been used to model the behavior of robots in a swarm. However, to the best of our knowledge, the formal specification approach has not been used to model the self-organization process in swarm robotic systems for shape formation. In this paper, we use a formal specification approach to model the shape formation task of swarm robots. We use Z (Zed) language of formal specification, which is a state-based language, to model the states of the entities of the systems. We demonstrate the effectiveness of Z for the self-organized shape formation. The presented formal specification model gives the outlines for designing and implementing the swarm robotic system for the formation of complex shapes and structures. It also provides the foundation for modeling the complex shape formation process for swarm robotics using a multi-agent system in a simulation-based environment. Keywords: Swarm robotics, Self-organization, Formal specification, Complex systems 

**Abstract (ZH)**: 机器人自组织形成结构和形状的应用是 swarm 机器人系统的刺激性应用。它涉及大量异质行为的自主机器人、它们之间的协调以及与动态环境的交互。这一复杂结构形成的工艺被视为一个复杂的系统，需要用任何建模方法进行建模。虽然已经使用形式化规范方法及其他形式化方法来建模 swarm 中机器人的行为。然而，据我们所知，形式化规范方法尚未被用于建模 swarm 机器人系统中用于形状形成过程的自组织过程。在本文中，我们使用形式化规范方法来建模 swarm 机器人的形状形成任务。我们使用状态基语言 Z（Zed）来建模系统的实体状态。我们展示了 Z 在自组织形状形成方面的有效性。提出的规范建模给出了设计和实现用于形成复杂形状和结构的 swarm 机器人系统的蓝图。它也为在基于仿真的多代理系统环境中建模 swarm 机器人中的复杂形状形成过程提供了基础。关键词： swarm 机器人，自组织，形式化规范，复杂系统。 

---
# Adaptive Model-Base Control of Quadrupeds via Online System Identification using Kalman Filter 

**Title (ZH)**: 基于卡尔曼滤波的在线系统识别的四足动物自适应模型ベース控制 

**Authors**: Jonas Haack, Franek Stark, Shubham Vyas, Frank Kirchner, Shivesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13432)  

**Abstract**: Many real-world applications require legged robots to be able to carry variable payloads. Model-based controllers such as model predictive control (MPC) have become the de facto standard in research for controlling these systems. However, most model-based control architectures use fixed plant models, which limits their applicability to different tasks. In this paper, we present a Kalman filter (KF) formulation for online identification of the mass and center of mass (COM) of a four-legged robot. We evaluate our method on a quadrupedal robot carrying various payloads and find that it is more robust to strong measurement noise than classical recursive least squares (RLS) methods. Moreover, it improves the tracking performance of the model-based controller with varying payloads when the model parameters are adjusted at runtime. 

**Abstract (ZH)**: 基于卡尔曼滤波的四足机器人载重在线辨识方法及其应用 

---
# VLM-SFD: VLM-Assisted Siamese Flow Diffusion Framework for Dual-Arm Cooperative Manipulation 

**Title (ZH)**: VLM-SFD：基于VLM的双臂协同 manipulation Siamese 流扩散框架 

**Authors**: Jiaming Chen, Yiyu Jiang, Aoshen Huang, Yang Li, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13428)  

**Abstract**: Dual-arm cooperative manipulation holds great promise for tackling complex real-world tasks that demand seamless coordination and adaptive dynamics. Despite substantial progress in learning-based motion planning, most approaches struggle to generalize across diverse manipulation tasks and adapt to dynamic, unstructured environments, particularly in scenarios involving interactions between two objects such as assembly, tool use, and bimanual grasping. To address these challenges, we introduce a novel VLM-Assisted Siamese Flow Diffusion (VLM-SFD) framework for efficient imitation learning in dual-arm cooperative manipulation. The proposed VLM-SFD framework exhibits outstanding adaptability, significantly enhancing the ability to rapidly adapt and generalize to diverse real-world tasks from only a minimal number of human demonstrations. Specifically, we propose a Siamese Flow Diffusion Network (SFDNet) employs a dual-encoder-decoder Siamese architecture to embed two target objects into a shared latent space, while a diffusion-based conditioning process-conditioned by task instructions-generates two-stream object-centric motion flows that guide dual-arm coordination. We further design a dynamic task assignment strategy that seamlessly maps the predicted 2D motion flows into 3D space and incorporates a pre-trained vision-language model (VLM) to adaptively assign the optimal motion to each robotic arm over time. Experiments validate the effectiveness of the proposed method, demonstrating its ability to generalize to diverse manipulation tasks while maintaining high efficiency and adaptability. The code and demo videos are publicly available on our project website this https URL. 

**Abstract (ZH)**: 基于VLM辅助的Siamese流扩散框架在双臂协同操作中的高效模仿学习 

---
# JENGA: Object selection and pose estimation for robotic grasping from a stack 

**Title (ZH)**: JENGA: 从堆积物中进行机器人抓取的对象选择与姿态估计 

**Authors**: Sai Srinivas Jeevanandam, Sandeep Inuganti, Shreedhar Govil, Didier Stricker, Jason Rambach  

**Link**: [PDF](https://arxiv.org/pdf/2506.13425)  

**Abstract**: Vision-based robotic object grasping is typically investigated in the context of isolated objects or unstructured object sets in bin picking scenarios. However, there are several settings, such as construction or warehouse automation, where a robot needs to interact with a structured object formation such as a stack. In this context, we define the problem of selecting suitable objects for grasping along with estimating an accurate 6DoF pose of these objects. To address this problem, we propose a camera-IMU based approach that prioritizes unobstructed objects on the higher layers of stacks and introduce a dataset for benchmarking and evaluation, along with a suitable evaluation metric that combines object selection with pose accuracy. Experimental results show that although our method can perform quite well, this is a challenging problem if a completely error-free solution is needed. Finally, we show results from the deployment of our method for a brick-picking application in a construction scenario. 

**Abstract (ZH)**: 基于视觉的机器人物体抓取通常在孤立物体或未结构化的物体集合的拾取场景中进行研究。然而，在建筑或仓库自动化等场景中，机器人需要与结构化的物体堆叠进行交互。在这种情况下，我们定义了选择适合抓取的物体并估算这些物体的准确6DoF姿态的问题。为了解决这个问题，我们提出了一种基于摄像头-IMU的方法，优先选择堆叠高层上的无遮挡物体，并介绍了一个用于基准测试和评估的数据集以及一个结合物体选择与姿态准确性的合适评价指标。实验结果表明，尽管我们的方法表现不错，但如果需要完全无误差的解决方案，则这是一个极具挑战性的问题。最后，我们在建筑场景中的砖块拾取应用中部署了我们的方法并展示了结果。 

---
# Delayed Expansion AGT: Kinodynamic Planning with Application to Tractor-Trailer Parking 

**Title (ZH)**: 延迟扩张AGT：动力学规划及其在牵引车-挂车泊车中的应用 

**Authors**: Dongliang Zheng, Yebin Wang, Stefano Di Cairano, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2506.13421)  

**Abstract**: Kinodynamic planning of articulated vehicles in cluttered environments faces additional challenges arising from high-dimensional state space and complex system dynamics. Built upon [1],[2], this work proposes the DE-AGT algorithm that grows a tree using pre-computed motion primitives (MPs) and A* heuristics. The first feature of DE-AGT is a delayed expansion of MPs. In particular, the MPs are divided into different modes, which are ranked online. With the MP classification and prioritization, DE-AGT expands the most promising mode of MPs first, which eliminates unnecessary computation and finds solutions faster. To obtain the cost-to-go heuristic for nonholonomic articulated vehicles, we rely on supervised learning and train neural networks for fast and accurate cost-to-go prediction. The learned heuristic is used for online mode ranking and node selection. Another feature of DE-AGT is the improved goal-reaching. Exactly reaching a goal state usually requires a constant connection checking with the goal by solving steering problems -- non-trivial and time-consuming for articulated vehicles. The proposed termination scheme overcomes this challenge by tightly integrating a light-weight trajectory tracking controller with the search process. DE-AGT is implemented for autonomous parking of a general car-like tractor with 3-trailer. Simulation results show an average of 10x acceleration compared to a previous method. 

**Abstract (ZH)**: 带有延迟扩展的MPs及其优先级化的DE-AGT算法在复杂环境中的铰接车辆机动规划 

---
# Observability-Aware Active Calibration of Multi-Sensor Extrinsics for Ground Robots via Online Trajectory Optimization 

**Title (ZH)**: 基于在线轨迹优化的地面机器人多传感器外参观测导向的主动标定 

**Authors**: Jiang Wang, Yaozhong Kang, Linya Fu, Kazuhiro Nakadai, He Kong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13420)  

**Abstract**: Accurate calibration of sensor extrinsic parameters for ground robotic systems (i.e., relative poses) is crucial for ensuring spatial alignment and achieving high-performance perception. However, existing calibration methods typically require complex and often human-operated processes to collect data. Moreover, most frameworks neglect acoustic sensors, thereby limiting the associated systems' auditory perception capabilities. To alleviate these issues, we propose an observability-aware active calibration method for ground robots with multimodal sensors, including a microphone array, a LiDAR (exteroceptive sensors), and wheel encoders (proprioceptive sensors). Unlike traditional approaches, our method enables active trajectory optimization for online data collection and calibration, contributing to the development of more intelligent robotic systems. Specifically, we leverage the Fisher information matrix (FIM) to quantify parameter observability and adopt its minimum eigenvalue as an optimization metric for trajectory generation via B-spline curves. Through planning and replanning of robot trajectory online, the method enhances the observability of multi-sensor extrinsic parameters. The effectiveness and advantages of our method have been demonstrated through numerical simulations and real-world experiments. For the benefit of the community, we have also open-sourced our code and data at this https URL. 

**Abstract (ZH)**: 地面机器人多模态传感器外参的可观性感知主动标定方法对确保空间对齐和实现高性能感知至关重要。然而，现有的标定方法通常需要复杂且往往依赖人工的数据采集过程。此外，大多数框架忽视了声学传感器，从而限制了相关系统的声音感知能力。为解决这些问题，我们提出了一种地面机器人多模态传感器（包括麦克风阵列、激光雷达（外部传感器）和编码器（内部传感器））的可观性感知主动标定方法。与传统方法不同，我们的方法能够实现主动轨迹优化以在线数据采集和标定，从而推动更智能的机器人系统的研发。具体而言，我们利用 Fisher 信息矩阵（FIM）量化参数可观性，并采用其最小特征值作为通过 B-样条曲线生成轨迹的优化指标。通过在线规划和重新规划机器人的轨迹，该方法增强了多传感器外参的可观性。我们的方法的有效性和优势已在数值仿真和实际实验中得到验证。为了社区的益处，我们已在该 URL 开源了我们的代码和数据。 

---
# Uncertainty-Informed Active Perception for Open Vocabulary Object Goal Navigation 

**Title (ZH)**: 基于不确定性指导的开放词汇目标导航主动感知 

**Authors**: Utkarsh Bajpai, Julius Rückin, Cyrill Stachniss, Marija Popović  

**Link**: [PDF](https://arxiv.org/pdf/2506.13367)  

**Abstract**: Mobile robots exploring indoor environments increasingly rely on vision-language models to perceive high-level semantic cues in camera images, such as object categories. Such models offer the potential to substantially advance robot behaviour for tasks such as object-goal navigation (ObjectNav), where the robot must locate objects specified in natural language by exploring the environment. Current ObjectNav methods heavily depend on prompt engineering for perception and do not address the semantic uncertainty induced by variations in prompt phrasing. Ignoring semantic uncertainty can lead to suboptimal exploration, which in turn limits performance. Hence, we propose a semantic uncertainty-informed active perception pipeline for ObjectNav in indoor environments. We introduce a novel probabilistic sensor model for quantifying semantic uncertainty in vision-language models and incorporate it into a probabilistic geometric-semantic map to enhance spatial understanding. Based on this map, we develop a frontier exploration planner with an uncertainty-informed multi-armed bandit objective to guide efficient object search. Experimental results demonstrate that our method achieves ObjectNav success rates comparable to those of state-of-the-art approaches, without requiring extensive prompt engineering. 

**Abstract (ZH)**: 室内环境探索的移动机器人 increasingly relies on 视觉-语言模型来感知相机图像中的高阶语义线索，如物体类别。此类模型为物体目标导航（ObjectNav）任务中的机器人行为提供了潜在的显著进步，其中机器人必须通过探索环境来定位用自然语言指定的物体。当前的 ObjectNav 方法高度依赖于感知方面的提示工程，并未解决由提示措辞变化引起的语义不确定性问题。忽略语义不确定性可能导致次优探索，从而限制了性能。因此，我们提出了一种基于语义不确定性主动感知的室内环境物体目标导航管道。我们引入了一种新颖的概率传感器模型，用于量化视觉-语言模型中的语义不确定性，并将其集成到概率几何语义地图中以增强空间理解。基于此地图，我们开发了一种具有不确定性指导的多臂bandit目标的前沿探索计划器，以指导高效的物体搜索。实验结果表明，我们的方法在不需要大量提示工程的情况下，实现了与现有最佳方法相当的物体目标导航成功率。 

---
# C2TE: Coordinated Constrained Task Execution Design for Ordering-Flexible Multi-Vehicle Platoon Merging 

**Title (ZH)**: C2TE: 协调约束任务执行设计用于订单灵活的多车辆编队合并 

**Authors**: Bin-Bin Hu, Yanxin Zhou, Henglai Wei, Shuo Cheng, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.13202)  

**Abstract**: In this paper, we propose a distributed coordinated constrained task execution (C2TE) algorithm that enables a team of vehicles from different lanes to cooperatively merge into an {\it ordering-flexible platoon} maneuvering on the desired lane. Therein, the platoon is flexible in the sense that no specific spatial ordering sequences of vehicles are predetermined. To attain such a flexible platoon, we first separate the multi-vehicle platoon (MVP) merging mission into two stages, namely, pre-merging regulation and {\it ordering-flexible platoon} merging, and then formulate them into distributed constraint-based optimization problems. Particularly, by encoding longitudinal-distance regulation and same-lane collision avoidance subtasks into the corresponding control barrier function (CBF) constraints, the proposed algorithm in Stage 1 can safely enlarge sufficient longitudinal distances among adjacent vehicles. Then, by encoding lateral convergence, longitudinal-target attraction, and neighboring collision avoidance subtasks into CBF constraints, the proposed algorithm in Stage~2 can efficiently achieve the {\it ordering-flexible platoon}. Note that the {\it ordering-flexible platoon} is realized through the interaction of the longitudinal-target attraction and time-varying neighboring collision avoidance constraints simultaneously. Feasibility guarantee and rigorous convergence analysis are both provided under strong nonlinear couplings induced by flexible orderings. Finally, experiments using three autonomous mobile vehicles (AMVs) are conducted to verify the effectiveness and flexibility of the proposed algorithm, and extensive simulations are performed to demonstrate its robustness, adaptability, and scalability when tackling vehicles' sudden breakdown, new appearing, different number of lanes, mixed autonomy, and large-scale scenarios, respectively. 

**Abstract (ZH)**: 分布式协同约束任务执行算法：不同车道车辆团队的可排序灵活车队合并方法 

---
# Equilibrium-Driven Smooth Separation and Navigation of Marsupial Robotic Systems 

**Title (ZH)**: 袋鼠型机器人系统均衡驱动平滑分离与导航 

**Authors**: Bin-Bin Hu, Bayu Jayawardhana, Ming Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13198)  

**Abstract**: In this paper, we propose an equilibrium-driven controller that enables a marsupial carrier-passenger robotic system to achieve smooth carrier-passenger separation and then to navigate the passenger robot toward a predetermined target point. Particularly, we design a potential gradient in the form of a cubic polynomial for the passenger's controller as a function of the carrier-passenger and carrier-target distances in the moving carrier's frame. This introduces multiple equilibrium points corresponding to the zero state of the error dynamic system during carrier-passenger separation. The change of equilibrium points is associated with the change in their attraction regions, enabling smooth carrier-passenger separation and afterwards seamless navigation toward the target. Finally, simulations demonstrate the effectiveness and adaptability of the proposed controller in environments containing obstacles. 

**Abstract (ZH)**: 本文提出一个均衡驱动控制器，使育儿袋式载运机器人能够在实现平稳的载运机器人与乘客机器人分离后，导航乘客机器人前往预设目标点。特别地，我们在移动载体坐标系中，基于载运机器人与乘客机器人及载运机器人与目标点的距离，设计了一个立方多项式的潜在梯度作为乘客机器人的控制器。这引入了多个均衡点，对应于载运机器人与乘客机器人分离过程中误差动态系统的零状态。均衡点的变化与其吸引力区域的变化相关，从而实现平稳的载运机器人与乘客机器人分离，并在之后无缝地导航至目标点。最后，仿真实验展示了所提控制器在包含障碍物环境中的有效性和适应性。 

---
# Cognitive Synergy Architecture: SEGO for Human-Centric Collaborative Robots 

**Title (ZH)**: 认知协同架构：SEGO为人机协同机器人服务 

**Authors**: Jaehong Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.13149)  

**Abstract**: This paper presents SEGO (Semantic Graph Ontology), a cognitive mapping architecture designed to integrate geometric perception, semantic reasoning, and explanation generation into a unified framework for human-centric collaborative robotics. SEGO constructs dynamic cognitive scene graphs that represent not only the spatial configuration of the environment but also the semantic relations and ontological consistency among detected objects. The architecture seamlessly combines SLAM-based localization, deep-learning-based object detection and tracking, and ontology-driven reasoning to enable real-time, semantically coherent mapping. 

**Abstract (ZH)**: 基于语义图本体的认知映射架构：面向人类中心的协作机器人几何感知、语义推理与解释生成统一框架 

---
# Autonomous 3D Moving Target Encirclement and Interception with Range measurement 

**Title (ZH)**: 自主三维移动目标环围与拦截（基于距离测量） 

**Authors**: Fen Liu, Shenghai Yuan, Thien-Minh Nguyen, Rong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.13106)  

**Abstract**: Commercial UAVs are an emerging security threat as they are capable of carrying hazardous payloads or disrupting air traffic. To counter UAVs, we introduce an autonomous 3D target encirclement and interception strategy. Unlike traditional ground-guided systems, this strategy employs autonomous drones to track and engage non-cooperative hostile UAVs, which is effective in non-line-of-sight conditions, GPS denial, and radar jamming, where conventional detection and neutralization from ground guidance fail. Using two noisy real-time distances measured by drones, guardian drones estimate the relative position from their own to the target using observation and velocity compensation methods, based on anti-synchronization (AS) and an X$-$Y circular motion combined with vertical jitter. An encirclement control mechanism is proposed to enable UAVs to adaptively transition from encircling and protecting a target to encircling and monitoring a hostile target. Upon breaching a warning threshold, the UAVs may even employ a suicide attack to neutralize the hostile target. We validate this strategy through real-world UAV experiments and simulated analysis in MATLAB, demonstrating its effectiveness in detecting, encircling, and intercepting hostile drones. More details: this https URL. 

**Abstract (ZH)**: 商用无人机是一种新兴的安全威胁，因为它们能够携带危险载荷或干扰空中交通。为此，我们提出了一种自主三维目标包围和拦截策略。该策略不同于传统的地面引导系统，它利用自主无人机追踪并对抗不合作的敌对无人机，在视线外、GPS拒绝和雷达干扰等条件下表现出色，而传统从地面引导进行检测和中和会失效。基于反同步（AS）以及结合XY圆周运动和垂直抖动的观测和速度补偿方法，利用两架无人机测得的带噪声的实时距离，守护无人机估计到目标的相对位置，并提出了一种包围控制机制，使无人机能够从保护目标转向包围监视敌对目标。当突破警告阈值时，无人机甚至可以采取自杀袭击来中和敌对目标。通过实际无人机实验和MATLAB模拟分析验证了该策略，在检测、包围和拦截敌对无人机方面显示了有效性。更多细节：请访问此链接。 

---
# Underwater target 6D State Estimation via UUV Attitude Enhance Observability 

**Title (ZH)**: 基于UUV姿态增强可观测性的水下目标6维状态估计 

**Authors**: Fen Liu, Chengfeng Jia, Na Zhang, Shenghai Yuan, Rong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.13105)  

**Abstract**: Accurate relative state observation of Unmanned Underwater Vehicles (UUVs) for tracking uncooperative targets remains a significant challenge due to the absence of GPS, complex underwater dynamics, and sensor limitations. Existing localization approaches rely on either global positioning infrastructure or multi-UUV collaboration, both of which are impractical for a single UUV operating in large or unknown environments. To address this, we propose a novel persistent relative 6D state estimation framework that enables a single UUV to estimate its relative motion to a non-cooperative target using only successive noisy range measurements from two monostatic sonar sensors. Our key contribution is an observability-enhanced attitude control strategy, which optimally adjusts the UUV's orientation to improve the observability of relative state estimation using a Kalman filter, effectively mitigating the impact of sensor noise and drift accumulation. Additionally, we introduce a rigorously proven Lyapunov-based tracking control strategy that guarantees long-term stability by ensuring that the UUV maintains an optimal measurement range, preventing localization errors from diverging over time. Through theoretical analysis and simulations, we demonstrate that our method significantly improves 6D relative state estimation accuracy and robustness compared to conventional approaches. This work provides a scalable, infrastructure-free solution for UUVs tracking uncooperative targets underwater. 

**Abstract (ZH)**: 基于单艇相对6D状态估计的无人驾驶水下车辆追踪非合作目标方法 

---
# A Novel ViDAR Device With Visual Inertial Encoder Odometry and Reinforcement Learning-Based Active SLAM Method 

**Title (ZH)**: 一种结合视觉惯性编码 odometry 和基于强化学习的主动 SLAM 方法的新颖 ViDAR 设备 

**Authors**: Zhanhua Xin, Zhihao Wang, Shenghao Zhang, Wanchao Chi, Yan Meng, Shihan Kong, Yan Xiong, Chong Zhang, Yuzhen Liu, Junzhi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13100)  

**Abstract**: In the field of multi-sensor fusion for simultaneous localization and mapping (SLAM), monocular cameras and IMUs are widely used to build simple and effective visual-inertial systems. However, limited research has explored the integration of motor-encoder devices to enhance SLAM performance. By incorporating such devices, it is possible to significantly improve active capability and field of view (FOV) with minimal additional cost and structural complexity. This paper proposes a novel visual-inertial-encoder tightly coupled odometry (VIEO) based on a ViDAR (Video Detection and Ranging) device. A ViDAR calibration method is introduced to ensure accurate initialization for VIEO. In addition, a platform motion decoupled active SLAM method based on deep reinforcement learning (DRL) is proposed. Experimental data demonstrate that the proposed ViDAR and the VIEO algorithm significantly increase cross-frame co-visibility relationships compared to its corresponding visual-inertial odometry (VIO) algorithm, improving state estimation accuracy. Additionally, the DRL-based active SLAM algorithm, with the ability to decouple from platform motion, can increase the diversity weight of the feature points and further enhance the VIEO algorithm's performance. The proposed methodology sheds fresh insights into both the updated platform design and decoupled approach of active SLAM systems in complex environments. 

**Abstract (ZH)**: 多传感器融合领域中单相机和IMU在同时定位与建图（SLAM）中的应用及其与电机编码器的集成研究：基于ViDAR的紧耦合视觉-惯性-编码器里程计及其在主动SLAM中的应用 

---
# IKDiffuser: Fast and Diverse Inverse Kinematics Solution Generation for Multi-arm Robotic Systems 

**Title (ZH)**: IKDiffuser: 多臂机器人系统快速多样逆向运动学解生成 

**Authors**: Zeyu Zhang, Ziyuan Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13087)  

**Abstract**: Solving Inverse Kinematics (IK) problems is fundamental to robotics, but has primarily been successful with single serial manipulators. For multi-arm robotic systems, IK remains challenging due to complex self-collisions, coupled joints, and high-dimensional redundancy. These complexities make traditional IK solvers slow, prone to failure, and lacking in solution diversity. In this paper, we present IKDiffuser, a diffusion-based model designed for fast and diverse IK solution generation for multi-arm robotic systems. IKDiffuser learns the joint distribution over the configuration space, capturing complex dependencies and enabling seamless generalization to multi-arm robotic systems of different structures. In addition, IKDiffuser can incorporate additional objectives during inference without retraining, offering versatility and adaptability for task-specific requirements. In experiments on 6 different multi-arm systems, the proposed IKDiffuser achieves superior solution accuracy, precision, diversity, and computational efficiency compared to existing solvers. The proposed IKDiffuser framework offers a scalable, unified approach to solving multi-arm IK problems, facilitating the potential of multi-arm robotic systems in real-time manipulation tasks. 

**Abstract (ZH)**: 基于扩散模型的多臂机器人逆运动学快速多样求解 

---
# CHARM: Considering Human Attributes for Reinforcement Modeling 

**Title (ZH)**: CHARM: 考虑人类属性的强化学习模型 

**Authors**: Qidi Fang, Hang Yu, Shijie Fang, Jindan Huang, Qiuyu Chen, Reuben M. Aronson, Elaine S. Short  

**Link**: [PDF](https://arxiv.org/pdf/2506.13079)  

**Abstract**: Reinforcement Learning from Human Feedback has recently achieved significant success in various fields, and its performance is highly related to feedback quality. While much prior work acknowledged that human teachers' characteristics would affect human feedback patterns, there is little work that has closely investigated the actual effects. In this work, we designed an exploratory study investigating how human feedback patterns are associated with human characteristics. We conducted a public space study with two long horizon tasks and 46 participants. We found that feedback patterns are not only correlated with task statistics, such as rewards, but also correlated with participants' characteristics, especially robot experience and educational background. Additionally, we demonstrated that human feedback value can be more accurately predicted with human characteristics compared to only using task statistics. All human feedback and characteristics we collected, and codes for our data collection and predicting more accurate human feedback are available at this https URL 

**Abstract (ZH)**: 从人类反馈中学习的强化学习近年来在各个领域取得了显著成功，其性能高度依赖于反馈质量。尽管先前研究认识到人类教师的特质会影响反馈模式，但鲜有研究深入探讨其实际影响。在本工作中，我们设计了一项探索性研究，调查人类反馈模式与人类特质之间的关联。我们通过一项包含两个长期任务的公共场所研究，共招募了46名参与者。我们发现，反馈模式不仅与任务统计数据（如奖励）相关，还与参与者的特质密切相关，尤其是机器人经验与教育背景。此外，我们证明了与仅使用任务统计数据相比，人类特质可以更准确地预测人类反馈的价值。所有收集的人类反馈和特质数据，以及我们用于数据收集和更准确预测人类反馈的代码，均可通过以下链接访问：this https URL。 

---
# Constrained Optimal Planning to Minimize Battery Degradation of Autonomous Mobile Robots 

**Title (ZH)**: 自主移动机器人电池退化最小化的受约束最优规划 

**Authors**: Jiachen Li, Jian Chu, Feiyang Zhao, Shihao Li, Wei Li, Dongmei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13019)  

**Abstract**: This paper proposes an optimization framework that addresses both cycling degradation and calendar aging of batteries for autonomous mobile robot (AMR) to minimize battery degradation while ensuring task completion. A rectangle method of piecewise linear approximation is employed to linearize the bilinear optimization problem. We conduct a case study to validate the efficiency of the proposed framework in achieving an optimal path planning for AMRs while reducing battery aging. 

**Abstract (ZH)**: 本文提出了一种优化框架，旨在同时解决自主移动机器人(AMR)电池的充放电退化和日历老化问题，以最小化电池退化并确保任务完成。通过采用分段线性逼近的矩形法将双线性优化问题线性化。通过案例研究验证了所提出框架在实现AMR最优路径规划的同时减少电池老化方面的有效性。 

---
# KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills 

**Title (ZH)**: KungfuBot：基于物理的人形全身控制学习高度动态技能 

**Authors**: Weiji Xie, Jinrui Han, Jiakun Zheng, Huanyu Li, Xinzhe Liu, Jiyuan Shi, Weinan Zhang, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12851)  

**Abstract**: Humanoid robots are promising to acquire various skills by imitating human behaviors. However, existing algorithms are only capable of tracking smooth, low-speed human motions, even with delicate reward and curriculum design. This paper presents a physics-based humanoid control framework, aiming to master highly-dynamic human behaviors such as Kungfu and dancing through multi-steps motion processing and adaptive motion tracking. For motion processing, we design a pipeline to extract, filter out, correct, and retarget motions, while ensuring compliance with physical constraints to the maximum extent. For motion imitation, we formulate a bi-level optimization problem to dynamically adjust the tracking accuracy tolerance based on the current tracking error, creating an adaptive curriculum mechanism. We further construct an asymmetric actor-critic framework for policy training. In experiments, we train whole-body control policies to imitate a set of highly-dynamic motions. Our method achieves significantly lower tracking errors than existing approaches and is successfully deployed on the Unitree G1 robot, demonstrating stable and expressive behaviors. The project page is this https URL. 

**Abstract (ZH)**: 类人机器人有望通过模仿人类行为来获取各种技能。然而，现有的算法只能跟踪光滑的低速人类动作，即便有精细的奖励和课程设计也是如此。本文提出一种基于物理的类人控制框架，旨在通过多步动作处理和自适应动作跟踪掌握高度动态的人类行为，如武术和舞蹈。在动作处理方面，我们设计了流水线来提取、过滤、校正和重新定位动作，同时最大限度地遵守物理约束。在动作模仿方面，我们提出了一个双层优化问题来根据当前跟踪误差动态调整跟踪准确性容忍度，创建自适应课程机制。我们进一步构建了一个不对称Actor-Critic框架进行策略训练。在实验中，我们训练了全身体控制策略以模仿一系列高度动态的动作。我们的方法在跟踪误差上显著优于现有方法，并已在Unitree G1机器人上成功部署，展示了稳定而表现力强的行为。项目页面详见：https://url.cn/3ZfRI1Ve 

---
# From Experts to a Generalist: Toward General Whole-Body Control for Humanoid Robots 

**Title (ZH)**: 从专家到通才： toward 人类形机器人全身控制的通用方法 

**Authors**: Yuxuan Wang, Ming Yang, Weishuai Zeng, Yu Zhang, Xinrun Xu, Haobin Jiang, Ziluo Ding, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12779)  

**Abstract**: Achieving general agile whole-body control on humanoid robots remains a major challenge due to diverse motion demands and data conflicts. While existing frameworks excel in training single motion-specific policies, they struggle to generalize across highly varied behaviors due to conflicting control requirements and mismatched data distributions. In this work, we propose BumbleBee (BB), an expert-generalist learning framework that combines motion clustering and sim-to-real adaptation to overcome these challenges. BB first leverages an autoencoder-based clustering method to group behaviorally similar motions using motion features and motion descriptions. Expert policies are then trained within each cluster and refined with real-world data through iterative delta action modeling to bridge the sim-to-real gap. Finally, these experts are distilled into a unified generalist controller that preserves agility and robustness across all motion types. Experiments on two simulations and a real humanoid robot demonstrate that BB achieves state-of-the-art general whole-body control, setting a new benchmark for agile, robust, and generalizable humanoid performance in the real world. 

**Abstract (ZH)**: BumbleBee：一种结合运动聚类和仿真到现实适应的专家-通才学习框架以实现 humanoid 机器人的一般敏捷全身控制 

---
# RL from Physical Feedback: Aligning Large Motion Models with Humanoid Control 

**Title (ZH)**: 从物理反馈学习RL：将大型运动模型与类人控制对齐 

**Authors**: Junpeng Yue, Zepeng Wang, Yuxuan Wang, Weishuai Zeng, Jiangxing Wang, Xinrun Xu, Yu Zhang, Sipeng Zheng, Ziluo Ding, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12769)  

**Abstract**: This paper focuses on a critical challenge in robotics: translating text-driven human motions into executable actions for humanoid robots, enabling efficient and cost-effective learning of new behaviors. While existing text-to-motion generation methods achieve semantic alignment between language and motion, they often produce kinematically or physically infeasible motions unsuitable for real-world deployment. To bridge this sim-to-real gap, we propose Reinforcement Learning from Physical Feedback (RLPF), a novel framework that integrates physics-aware motion evaluation with text-conditioned motion generation. RLPF employs a motion tracking policy to assess feasibility in a physics simulator, generating rewards for fine-tuning the motion generator. Furthermore, RLPF introduces an alignment verification module to preserve semantic fidelity to text instructions. This joint optimization ensures both physical plausibility and instruction alignment. Extensive experiments show that RLPF greatly outperforms baseline methods in generating physically feasible motions while maintaining semantic correspondence with text instruction, enabling successful deployment on real humanoid robots. 

**Abstract (ZH)**: 本文聚焦于机器人领域的一个关键挑战：将文本驱动的人类动作转化为可执行的动作，使类人机器人能够高效且经济地学习新行为。尽管现有的从文本生成动作的方法在语义上实现了语言与动作的对齐，但它们往往生成出在动力学或物理上不可行的动作，不适合实际部署。为了解决这一从模拟到现实的差距，我们提出了一种新的框架——物理反馈强化学习（RLPF），它将物理感知的动作评估与文本条件下的动作生成相结合。RLPF 使用一个动作追踪策略在物理模拟器中评估动作的可行性，并生成奖励以 fine-tune 动作生成器。此外，RLPF 引入了一个对齐验证模块以保持与文本指令的语义一致性。这种联合优化确保了物理上的合理性和指令的一致性。广泛实验表明，RLPF 在生成物理上可行的动作方面显著优于基线方法，同时保持与文本指令的语义对应关系，从而成功部署在实际类人机器人上。 

---
# On-board Sonar Data Classification for Path Following in Underwater Vehicles using Fast Interval Type-2 Fuzzy Extreme Learning Machine 

**Title (ZH)**: 基于快速区间类型-2模糊极学习机的水下车辆路径跟随声纳数据分类 

**Authors**: Adrian Rubio-Solis, Luciano Nava-Balanzar, Tomas Salgado-Jimenez  

**Link**: [PDF](https://arxiv.org/pdf/2506.12762)  

**Abstract**: In autonomous underwater missions, the successful completion of predefined paths mainly depends on the ability of underwater vehicles to recognise their surroundings. In this study, we apply the concept of Fast Interval Type-2 Fuzzy Extreme Learning Machine (FIT2-FELM) to train a Takagi-Sugeno-Kang IT2 Fuzzy Inference System (TSK IT2-FIS) for on-board sonar data classification using an underwater vehicle called BlueROV2. The TSK IT2-FIS is integrated into a Hierarchical Navigation Strategy (HNS) as the main navigation engine to infer local motions and provide the BlueROV2 with full autonomy to follow an obstacle-free trajectory in a water container of 2.5m x 2.5m x 3.5m. Compared to traditional navigation architectures, using the proposed method, we observe a robust path following behaviour in the presence of uncertainty and noise. We found that the proposed approach provides the BlueROV with a more complete sensory picture about its surroundings while real-time navigation planning is performed by the concurrent execution of two or more tasks. 

**Abstract (ZH)**: 自主水下任务中，预定义路径的成功完成主要依赖于水下车辆对其周围环境的识别能力。在本研究中，我们应用Fast Interval Type-2 Fuzzy Extreme Learning Machine (FIT2-FELM)的概念，训练一种Takagi-Sugeno-Kang类型2模糊推理系统（TSK IT2-FIS），以利用名为BlueROV2的水下车辆对随船声纳数据进行分类。TSK IT2-FIS被整合到层次导航策略（HNS）中，作为主要的导航引擎，用于推断局部运动，并使BlueROV2能够在2.5m x 2.5m x 3.5m的水容器中自主跟随无障碍轨迹。与传统导航架构相比，使用所提出的方法，在不确定性与噪声存在的条件下，我们观察到更加稳健的路径跟踪行为。我们发现，所提出的方法为BlueROV2提供了更完整的环境感知图，同时通过两个或多个任务的并发执行进行实时导航规划。 

---
# Physics-informed Neural Motion Planning via Domain Decomposition in Large Environments 

**Title (ZH)**: 基于域分解的大环境物理知情神经运动规划 

**Authors**: Yuchen Liu, Alexiy Buynitsky, Ruiqi Ni, Ahmed H. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12742)  

**Abstract**: Physics-informed Neural Motion Planners (PiNMPs) provide a data-efficient framework for solving the Eikonal Partial Differential Equation (PDE) and representing the cost-to-go function for motion planning. However, their scalability remains limited by spectral bias and the complex loss landscape of PDE-driven training. Domain decomposition mitigates these issues by dividing the environment into smaller subdomains, but existing methods enforce continuity only at individual spatial points. While effective for function approximation, these methods fail to capture the spatial connectivity required for motion planning, where the cost-to-go function depends on both the start and goal coordinates rather than a single query point. We propose Finite Basis Neural Time Fields (FB-NTFields), a novel neural field representation for scalable cost-to-go estimation. Instead of enforcing continuity in output space, FB-NTFields construct a latent space representation, computing the cost-to-go as a distance between the latent embeddings of start and goal coordinates. This enables global spatial coherence while integrating domain decomposition, ensuring efficient large-scale motion planning. We validate FB-NTFields in complex synthetic and real-world scenarios, demonstrating substantial improvements over existing PiNMPs. Finally, we deploy our method on a Unitree B1 quadruped robot, successfully navigating indoor environments. The supplementary videos can be found at this https URL. 

**Abstract (ZH)**: 基于物理的神经运动规划器（PiNMPs）提供了一种高效的框架来求解Eikonal偏微分方程（PDE）并表示运动规划的成本函数。然而，它们的扩展性受限于频谱偏差和由PDE驱动的训练复合损失景观。域分解通过将环境划分为较小的子域来缓解这些问题，但现有方法仅在个别空间点上保证连续性。虽然这些方法在函数逼近方面有效，但它们无法捕捉到运动规划所需的空间连续性，其中成本函数依赖于起点和目标坐标，而不仅仅是单一查询点。我们提出了一种新的基于有限基的神经时间场（FB-NTFields）来进行可扩展的成本函数估计。与在输出空间中保证连续性不同，FB-NTFields 构建了一个潜空间表示，通过计算起点和目标坐标的潜嵌入之间的距离来计算成本函数。这使得全局空间一致性成为可能，同时结合了域分解，从而确保了大规模运动规划的高效性。我们在复杂的合成和真实场景中验证了FB-NTFields，展示了其相对于现有PiNMPs的显著改进。最后，我们在一个Unitree B1 四足机器人上部署了我们的方法，成功导航了室内环境。补充视频可以在以下链接找到：this https URL。 

---
# Multimodal Large Language Models-Enabled UAV Swarm: Towards Efficient and Intelligent Autonomous Aerial Systems 

**Title (ZH)**: 基于多模态大型语言模型的无人机 swarm： toward 高效且智能的自主空中系统 

**Authors**: Yuqi Ping, Tianhao Liang, Huahao Ding, Guangyu Lei, Junwei Wu, Xuan Zou, Kuan Shi, Rui Shao, Chiya Zhang, Weizheng Zhang, Weijie Yuan, Tingting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12710)  

**Abstract**: Recent breakthroughs in multimodal large language models (MLLMs) have endowed AI systems with unified perception, reasoning and natural-language interaction across text, image and video streams. Meanwhile, Unmanned Aerial Vehicle (UAV) swarms are increasingly deployed in dynamic, safety-critical missions that demand rapid situational understanding and autonomous adaptation. This paper explores potential solutions for integrating MLLMs with UAV swarms to enhance the intelligence and adaptability across diverse tasks. Specifically, we first outline the fundamental architectures and functions of UAVs and MLLMs. Then, we analyze how MLLMs can enhance the UAV system performance in terms of target detection, autonomous navigation, and multi-agent coordination, while exploring solutions for integrating MLLMs into UAV systems. Next, we propose a practical case study focused on the forest fire fighting. To fully reveal the capabilities of the proposed framework, human-machine interaction, swarm task planning, fire assessment, and task execution are investigated. Finally, we discuss the challenges and future research directions for the MLLMs-enabled UAV swarm. An experiment illustration video could be found online at this https URL. 

**Abstract (ZH)**: 最近在多模态大型语言模型方面的突破赋予了AI系统在文本、图像和视频流中统一感知、推理和自然语言交互的能力。与此同时，无人机(UAV)群在未来动态且安全关键的任务中越来越被部署，这些任务要求快速的情境理解和自主适应。本文探讨了将多模态大型语言模型与无人机群集成以增强跨多种任务的智能和适应性的潜在解决方案。具体来说，我们首先概述了无人机和多模态大型语言模型的基本架构和功能。然后，我们分析了多模态大型语言模型如何在目标检测、自主导航和多智能体协调方面提升无人机系统性能，并探讨了将多模态大型语言模型集成到无人机系统中的解决方案。接下来，我们提出了一个以森林火灾扑救为重点的应用案例研究。为了充分展示所提出框架的能力，我们研究了人机交互、集群任务规划、火灾评估和任务执行。最后，我们讨论了由多模态大型语言模型赋能的无人机群面临的挑战和未来研究方向。有关实验示意图视频可以在以下网址在线查看：这个 https URL。 

---
# Adapting by Analogy: OOD Generalization of Visuomotor Policies via Functional Correspondence 

**Title (ZH)**: 通过类比适应：通过功能对应实现知觉运动策略的OOD泛化 

**Authors**: Pranay Gupta, Henny Admoni, Andrea Bajcsy  

**Link**: [PDF](https://arxiv.org/pdf/2506.12678)  

**Abstract**: End-to-end visuomotor policies trained using behavior cloning have shown a remarkable ability to generate complex, multi-modal low-level robot behaviors. However, at deployment time, these policies still struggle to act reliably when faced with out-of-distribution (OOD) visuals induced by objects, backgrounds, or environment changes. Prior works in interactive imitation learning solicit corrective expert demonstrations under the OOD conditions -- but this can be costly and inefficient. We observe that task success under OOD conditions does not always warrant novel robot behaviors. In-distribution (ID) behaviors can directly be transferred to OOD conditions that share functional similarities with ID conditions. For example, behaviors trained to interact with in-distribution (ID) pens can apply to interacting with a visually-OOD pencil. The key challenge lies in disambiguating which ID observations functionally correspond to the OOD observation for the task at hand. We propose that an expert can provide this OOD-to-ID functional correspondence. Thus, instead of collecting new demonstrations and re-training at every OOD encounter, our method: (1) detects the need for feedback by first checking if current observations are OOD and then identifying whether the most similar training observations show divergent behaviors, (2) solicits functional correspondence feedback to disambiguate between those behaviors, and (3) intervenes on the OOD observations with the functionally corresponding ID observations to perform deployment-time generalization. We validate our method across diverse real-world robotic manipulation tasks with a Franka Panda robotic manipulator. Our results show that test-time functional correspondences can improve the generalization of a vision-based diffusion policy to OOD objects and environment conditions with low feedback. 

**Abstract (ZH)**: 端到端的视觉-运动策略通过行为克隆训练，展示了生成复杂多模态低级机器人行为的 remarkable 能力。然而，在部署时，这些策略在面对由物体、背景或环境变化引起的 out-of-distribution (OOD) 视觉时，仍然难以可靠地行动。之前在交互式模仿学习中的前期工作在 OOD 条件下寻求专家的纠正演示——但这可能是昂贵且低效的。我们观察到，任务在 OOD 条件下的成功并不总是需要新的机器人行为。与 ID 条件共享功能相似性的 ID 行为可以直接转移到 OOD 条件中。例如，训练用于与 in-distribution (ID) 笔交互的行为可以应用到与之在视觉上 OOD 的铅笔交互上。关键挑战在于，辨别当前 OOD 观察与任务相关的功能相似的 ID 观察。我们提出专家可以提供这种 OOD 到 ID 的功能对应关系。因此，我们的方法不是在每遇到一次 OOD 就收集新的演示和重新训练，而是：(1) 通过首先检查当前观察是否为 OOD，然后确定最相似的训练观察是否表现出不同的行为来检测需要反馈的需求；(2) 请求功能对应反馈以在这些行为之间进行去模糊；(3) 使用功能对应的 ID 观察干预 OOD 观察，以实现部署时的一般化。我们在不同的真实世界机器人操作任务中使用 Franka Panda 机器人操作器验证了我们的方法。我们的结果表明，测试时的功能对应关系可以降低反馈成本，提高基于视觉的扩散策略对 OOD 对象和环境条件的一般化能力。 

---
# Goal-based Self-Adaptive Generative Adversarial Imitation Learning (Goal-SAGAIL) for Multi-goal Robotic Manipulation Tasks 

**Title (ZH)**: 基于目标的自适应生成对抗模仿学习（Goal-SAGAIL）用于多目标机器人操作任务 

**Authors**: Yingyi Kuang, Luis J. Manso, George Vogiatzis  

**Link**: [PDF](https://arxiv.org/pdf/2506.12676)  

**Abstract**: Reinforcement learning for multi-goal robot manipulation tasks poses significant challenges due to the diversity and complexity of the goal space. Techniques such as Hindsight Experience Replay (HER) have been introduced to improve learning efficiency for such tasks. More recently, researchers have combined HER with advanced imitation learning methods such as Generative Adversarial Imitation Learning (GAIL) to integrate demonstration data and accelerate training speed. However, demonstration data often fails to provide enough coverage for the goal space, especially when acquired from human teleoperation. This biases the learning-from-demonstration process toward mastering easier sub-tasks instead of tackling the more challenging ones. In this work, we present Goal-based Self-Adaptive Generative Adversarial Imitation Learning (Goal-SAGAIL), a novel framework specifically designed for multi-goal robot manipulation tasks. By integrating self-adaptive learning principles with goal-conditioned GAIL, our approach enhances imitation learning efficiency, even when limited, suboptimal demonstrations are available. Experimental results validate that our method significantly improves learning efficiency across various multi-goal manipulation scenarios -- including complex in-hand manipulation tasks -- using suboptimal demonstrations provided by both simulation and human experts. 

**Abstract (ZH)**: 基于目标自适应生成对抗模仿学习（Goal-SAGAIL）方法在多目标机器人操作任务中的应用 

---
# Deep Fusion of Ultra-Low-Resolution Thermal Camera and Gyroscope Data for Lighting-Robust and Compute-Efficient Rotational Odometry 

**Title (ZH)**: 超低分辨率热-camera和陀螺仪数据的深度融合用于照明鲁棒且计算高效的旋转里程计 

**Authors**: Farida Mohsen, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12536)  

**Abstract**: Accurate rotational odometry is crucial for autonomous robotic systems, particularly for small, power-constrained platforms such as drones and mobile robots. This study introduces thermal-gyro fusion, a novel sensor fusion approach that integrates ultra-low-resolution thermal imaging with gyroscope readings for rotational odometry. Unlike RGB cameras, thermal imaging is invariant to lighting conditions and, when fused with gyroscopic data, mitigates drift which is a common limitation of inertial sensors. We first develop a multimodal data acquisition system to collect synchronized thermal and gyroscope data, along with rotational speed labels, across diverse environments. Subsequently, we design and train a lightweight Convolutional Neural Network (CNN) that fuses both modalities for rotational speed estimation. Our analysis demonstrates that thermal-gyro fusion enables a significant reduction in thermal camera resolution without significantly compromising accuracy, thereby improving computational efficiency and memory utilization. These advantages make our approach well-suited for real-time deployment in resource-constrained robotic systems. Finally, to facilitate further research, we publicly release our dataset as supplementary material. 

**Abstract (ZH)**: 精确的旋转 odometry 对自主机器人系统至关重要，特别是在诸如无人机和移动机器人等小型、功率受限的平台上。本文介绍了一种新颖的传感器融合方法——热成像与陀螺仪融合，该方法将超低分辨率热成像与陀螺仪读数集成用于旋转 odometry。与 RGB 相机不同，热成像对光照条件不敏感，在与陀螺仪数据融合时可以减轻惯性传感器的漂移问题。我们首先开发了一种多模数据采集系统，以同步收集热成像与陀螺仪数据及其旋转速度标签，跨越多种环境。随后，我们设计并训练了一个轻量级的卷积神经网络（CNN），将其两种模态数据融合用于旋转速度估计。我们的分析表明，热成像与陀螺仪融合能够在不显著牺牲精度的情况下大幅降低热摄像机的分辨率，从而提高计算效率和内存利用率。这些优点使我们的方法适用于资源受限的机器人系统的实时部署。最后，为了促进进一步的研究，我们公开发布了我们的数据集作为补充材料。 

---
# A Spatial Relationship Aware Dataset for Robotics 

**Title (ZH)**: 空间关系感知数据集 для 机器人学 

**Authors**: Peng Wang, Minh Huy Pham, Zhihao Guo, Wei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12525)  

**Abstract**: Robotic task planning in real-world environments requires not only object recognition but also a nuanced understanding of spatial relationships between objects. We present a spatial-relationship-aware dataset of nearly 1,000 robot-acquired indoor images, annotated with object attributes, positions, and detailed spatial relationships. Captured using a Boston Dynamics Spot robot and labelled with a custom annotation tool, the dataset reflects complex scenarios with similar or identical objects and intricate spatial arrangements. We benchmark six state-of-the-art scene-graph generation models on this dataset, analysing their inference speed and relational accuracy. Our results highlight significant differences in model performance and demonstrate that integrating explicit spatial relationships into foundation models, such as ChatGPT 4o, substantially improves their ability to generate executable, spatially-aware plans for robotics. The dataset and annotation tool are publicly available at this https URL, supporting further research in spatial reasoning for robotics. 

**Abstract (ZH)**: 实时环境中的机器人任务规划不仅需要物体识别，还需要对物体之间空间关系的深刻理解。我们呈现了一个包含近1,000张室内图像的空间关系感知数据集，这些图像由机器人 annotation 并标注了物体属性、位置和详细的空间关系。使用波士顿动力公司的 Spot 机器人采集，并使用自定义标注工具进行标记，该数据集反映了具有相似或相同物体和复杂空间布局的复杂场景。我们在该数据集上基准测试了六种最先进的场景图生成模型，分析了它们的推理速度和关系准确性。我们的结果强调了不同模型性能的显著差异，并表明将明确的空间关系集成到基础模型（如ChatGPT 4o）中，可以显著提高其生成可执行且空间感知的机器人计划的能力。数据集和标注工具在此httpsURL公开，支持进一步的空间推理研究用于机器人领域。 

---
# Sense and Sensibility: What makes a social robot convincing to high-school students? 

**Title (ZH)**: 感性和理性：什么让社交机器人对高中生产生说服力？ 

**Authors**: Pablo Gonzalez-Oliveras, Olov Engwall, Ali Reza Majlesi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12507)  

**Abstract**: This study with 40 high-school students demonstrates the high influence of a social educational robot on students' decision-making for a set of eight true-false questions on electric circuits, for which the theory had been covered in the students' courses. The robot argued for the correct answer on six questions and the wrong on two, and 75% of the students were persuaded by the robot to perform beyond their expected capacity, positively when the robot was correct and negatively when it was wrong. Students with more experience of using large language models were even more likely to be influenced by the robot's stance -- in particular for the two easiest questions on which the robot was wrong -- suggesting that familiarity with AI can increase susceptibility to misinformation by AI.
We further examined how three different levels of portrayed robot certainty, displayed using semantics, prosody and facial signals, affected how the students aligned with the robot's answer on specific questions and how convincing they perceived the robot to be on these questions. The students aligned with the robot's answers in 94.4% of the cases when the robot was portrayed as Certain, 82.6% when it was Neutral and 71.4% when it was Uncertain. The alignment was thus high for all conditions, highlighting students' general susceptibility to accept the robot's stance, but alignment in the Uncertain condition was significantly lower than in the Certain. Post-test questionnaire answers further show that students found the robot most convincing when it was portrayed as Certain. These findings highlight the need for educational robots to adjust their display of certainty based on the reliability of the information they convey, to promote students' critical thinking and reduce undue influence. 

**Abstract (ZH)**: 这项研究以40名高中生为对象，展示了社会教育机器人对一组关于电路的真假判断问题（学生在课程中已经学习过相关理论）决策过程的高影响力。机器人对六个问题给出了正确的答案，对两个问题给出了错误的答案，并且75%的学生在机器人给出正确答案时提高了自己的表现水平，而在错误答案时则表现较差。具有更多大型语言模型使用经验的学生更容易受到机器人立场的影响——尤其是对于机器人错误的两个最简单的问题，这表明对AI的熟悉程度可能会增加对AI假信息的易感性。我们进一步探究了通过语义、语调和面部信号三种不同水平的机器人确定性表达，对学生在特定问题上对机器人回答的认同程度以及他们认为机器人说辞说服力的影响。当机器人被呈现为确定时，学生有94.4%的案例与机器人的答案一致，中性时有82.6%，不确定时有71.4%。因此，在所有条件下，学生的认同度都很高，但不确定性条件下的认同度明显低于确定性条件。问卷调查进一步表明，当机器人表现得最确定时，学生们觉得它最令人信服。这些发现突显了教育机器人根据所传达信息的可靠性调整其确定性显示的必要性，以促进学生的批判性思维并减少不必要的影响。 

---
# AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making 

**Title (ZH)**: 反 grounded: 将机器人动作提升至多模态表示空间以进行决策 

**Authors**: Wenbo Li, Shiyi Wang, Yiteng Chen, Huiping Zhuang, Qingyao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12374)  

**Abstract**: Vision-Language Models (VLMs) encode knowledge and reasoning capabilities for robotic manipulation within high-dimensional representation spaces. However, current approaches often project them into compressed intermediate representations, discarding important task-specific information such as fine-grained spatial or semantic details. To address this, we propose AntiGrounding, a new framework that reverses the instruction grounding process. It lifts candidate actions directly into the VLM representation space, renders trajectories from multiple views, and uses structured visual question answering for instruction-based decision making. This enables zero-shot synthesis of optimal closed-loop robot trajectories for new tasks. We also propose an offline policy refinement module that leverages past experience to enhance long-term performance. Experiments in both simulation and real-world environments show that our method outperforms baselines across diverse robotic manipulation tasks. 

**Abstract (ZH)**: Vision-Language模型反接地ddeninge（AntiGrounding）：一种新的框架用于直接将候选动作提升到Vision-Language模型表示空间，从多视角渲染轨迹，并使用结构化视觉问答进行基于指令的决策，从而实现新任务的零样本合成最优闭环机器人轨迹。我们还提出了一种离线策略精炼模块，利用过往经验提升长期性能。在仿真和真实环境中的实验表明，我们的方法在多种机器人操作任务中优于基线方法。 

---
# Explosive Output to Enhance Jumping Ability: A Variable Reduction Ratio Design Paradigm for Humanoid Robots Knee Joint 

**Title (ZH)**: 爆炸式输出以提升跳跃能力： humanoid 机器人膝关节变减速比设计范式 

**Authors**: Xiaoshuai Ma, Haoxiang Qi, Qingqing Li, Haochen Xu, Xuechao Chen, Junyao Gao, Zhangguo Yu, Qiang Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12314)  

**Abstract**: Enhancing the explosive power output of the knee joints is critical for improving the agility and obstacle-crossing capabilities of humanoid robots. However, a mismatch between the knee-to-center-of-mass (CoM) transmission ratio and jumping demands, coupled with motor performance degradation at high speeds, restricts the duration of high-power output and limits jump performance. To address these problems, this paper introduces a novel knee joint design paradigm employing a dynamically decreasing reduction ratio for explosive output during jump. Analysis of motor output characteristics and knee kinematics during jumping inspired a coupling strategy in which the reduction ratio gradually decreases as the joint extends. A high initial ratio rapidly increases torque at jump initiation, while its gradual reduction minimizes motor speed increments and power losses, thereby maintaining sustained high-power output. A compact and efficient linear actuator-driven guide-rod mechanism realizes this coupling strategy, supported by parameter optimization guided by explosive jump control strategies. Experimental validation demonstrated a 63 cm vertical jump on a single-joint platform (a theoretical improvement of 28.1\% over the optimal fixed-ratio joints). Integrated into a humanoid robot, the proposed design enabled a 1.1 m long jump, a 0.5 m vertical jump, and a 0.5 m box jump. 

**Abstract (ZH)**: 增强膝关节的爆发力输出对于提高类人机器人敏捷性和越障能力至关重要。然而，膝关节到质心的传动比与跳跃需求之间的不匹配，以及高速度下的电机性能下降，限制了高功率输出的持续时间和跳跃性能。为此，本文提出了一种新的膝关节设计范式，在跳跃期间采用动态递减的减速比以提高爆发力输出。电机输出特性和跳跃期间膝关节运动学的分析启发了该耦合策略，即关节延伸时减速比逐渐减小。较高的初始比迅速增加跳跃初期的扭矩，而其逐渐减小最大限度地减少了电机速度增量和功率损失，从而维持了持续的高功率输出。紧凑高效的线性作动器驱动滑杆机制实现了这一耦合策略，并得到了爆炸跳跃控制策略指导下的参数优化的支持。实验验证显示，单关节平台上的垂直跳跃高度提高了63厘米（理论上比最优固定比关节提高了28.1%）。将该设计集成到类人机器人中，实现了1.1米的长跳、0.5米的垂直跳和0.5米的方块跳。 

---
# Perspective on Utilizing Foundation Models for Laboratory Automation in Materials Research 

**Title (ZH)**: 利用基础模型推动材料研究领域的实验室自动化 Perspective on Utilizing Foundation Models for Laboratory Automation in Materials Research 

**Authors**: Kan Hatakeyama-Sato, Toshihiko Nishida, Kenta Kitamura, Yoshitaka Ushiku, Koichi Takahashi, Yuta Nabae, Teruaki Hayakawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12312)  

**Abstract**: This review explores the potential of foundation models to advance laboratory automation in the materials and chemical sciences. It emphasizes the dual roles of these models: cognitive functions for experimental planning and data analysis, and physical functions for hardware operations. While traditional laboratory automation has relied heavily on specialized, rigid systems, foundation models offer adaptability through their general-purpose intelligence and multimodal capabilities. Recent advancements have demonstrated the feasibility of using large language models (LLMs) and multimodal robotic systems to handle complex and dynamic laboratory tasks. However, significant challenges remain, including precision manipulation of hardware, integration of multimodal data, and ensuring operational safety. This paper outlines a roadmap highlighting future directions, advocating for close interdisciplinary collaboration, benchmark establishment, and strategic human-AI integration to realize fully autonomous experimental laboratories. 

**Abstract (ZH)**: 基础模型在材料与化学科学领域实验室自动化中的潜力及其展望：跨学科合作、基准建立及人机协同的战略方向 

---
# Role of Uncertainty in Model Development and Control Design for a Manufacturing Process 

**Title (ZH)**: 制造过程建模与控制设计中的不确定性作用 

**Authors**: Rongfei Li, Francis Assadian  

**Link**: [PDF](https://arxiv.org/pdf/2506.12273)  

**Abstract**: The use of robotic technology has drastically increased in manufacturing in the 21st century. But by utilizing their sensory cues, humans still outperform machines, especially in the micro scale manufacturing, which requires high-precision robot manipulators. These sensory cues naturally compensate for high level of uncertainties that exist in the manufacturing environment. Uncertainties in performing manufacturing tasks may come from measurement noise, model inaccuracy, joint compliance (e.g., elasticity) etc. Although advanced metrology sensors and high-precision microprocessors, which are utilized in nowadays robots, have compensated for many structural and dynamic errors in robot positioning, but a well-designed control algorithm still works as a comparable and cheaper alternative to reduce uncertainties in automated manufacturing. Our work illustrates that a multi-robot control system can reduce various uncertainties to a great amount. 

**Abstract (ZH)**: 21世纪机器人技术在制造领域中的应用大幅增加。但在微观规模制造中，利用人类的感觉 cues，人类仍然超越机器，特别是在需要高精度机器人操作的场合。这些感觉 cues 自然地补偿了制造环境中存在的高不确定性。执行制造任务时的不确定性可能来自测量噪声、模型不准确、关节顺应性（如弹性）等。虽然现代机器人利用了先进的计量传感器和高精度微处理器来补偿许多结构和动态误差，但精心设计的控制算法仍然是减少自动化制造中不确定性的经济且有效的替代方案。我们的研究展示了一个多机器人控制系统可以大幅减少各种不确定性。 

---
# Strategic Vantage Selection for Learning Viewpoint-Agnostic Manipulation Policies 

**Title (ZH)**: 基于视角无关操作策略的学习选择性优势分析 

**Authors**: Sreevishakh Vasudevan, Som Sagar, Ransalu Senanayake  

**Link**: [PDF](https://arxiv.org/pdf/2506.12261)  

**Abstract**: Vision-based manipulation has shown remarkable success, achieving promising performance across a range of tasks. However, these manipulation policies often fail to generalize beyond their training viewpoints, which is a persistent challenge in achieving perspective-agnostic manipulation, especially in settings where the camera is expected to move at runtime. Although collecting data from many angles seems a natural solution, such a naive approach is both resource-intensive and degrades manipulation policy performance due to excessive and unstructured visual diversity. This paper proposes Vantage, a framework that systematically identifies and integrates data from optimal perspectives to train robust, viewpoint-agnostic policies. By formulating viewpoint selection as a continuous optimization problem, we iteratively fine-tune policies on a few vantage points. Since we leverage Bayesian optimization to efficiently navigate the infinite space of potential camera configurations, we are able to balance exploration of novel views and exploitation of high-performing ones, thereby ensuring data collection from a minimal number of effective viewpoints. We empirically evaluate this framework on diverse standard manipulation tasks using multiple policy learning methods, demonstrating that fine-tuning with data from strategic camera placements yields substantial performance gains, achieving average improvements of up to 46.19% when compared to fixed, random, or heuristic-based strategies. 

**Abstract (ZH)**: 基于视觉的操控已经在多种任务中取得了显著的成果，但这些操控策略往往难以在训练视角之外进行泛化，这是实现视角无关操控的一个持续性挑战，尤其是在相机在运行时需要移动的场景中。虽然从多个角度收集数据似乎是自然的解决方案，但这种简单的做法既资源密集又会因为过多且无序的视觉多样性而降低操控策略的性能。本文提出了一种名为Vantage的框架，该框架系统地识别并整合来自最优视角的数据，以训练鲁棒的、视角无关的策略。通过将视角选择形式化为一个连续优化问题，我们迭代地在少数关键视角上微调策略。由于我们利用贝叶斯优化高效地探索潜在相机配置的空间，从而能够平衡新视角的探索和高性能视角的利用，从而确保从少量有效的视角收集数据。我们在多种标准操控任务上使用多种策略学习方法进行实证评估，证明了从战略性相机位置收集的数据能够带来显著的性能提升，在与固定、随机或启发式策略相比时，平均提升了46.19%。 

---
# ProVox: Personalization and Proactive Planning for Situated Human-Robot Collaboration 

**Title (ZH)**: ProVox: 个性化与主动规划在情境化人机协作中的应用 

**Authors**: Jennifer Grannen, Siddharth Karamcheti, Blake Wulfe, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2506.12248)  

**Abstract**: Collaborative robots must quickly adapt to their partner's intent and preferences to proactively identify helpful actions. This is especially true in situated settings where human partners can continually teach robots new high-level behaviors, visual concepts, and physical skills (e.g., through demonstration), growing the robot's capabilities as the human-robot pair work together to accomplish diverse tasks. In this work, we argue that robots should be able to infer their partner's goals from early interactions and use this information to proactively plan behaviors ahead of explicit instructions from the user. Building from the strong commonsense priors and steerability of large language models, we introduce ProVox ("Proactive Voice"), a novel framework that enables robots to efficiently personalize and adapt to individual collaborators. We design a meta-prompting protocol that empowers users to communicate their distinct preferences, intent, and expected robot behaviors ahead of starting a physical interaction. ProVox then uses the personalized prompt to condition a proactive language model task planner that anticipates a user's intent from the current interaction context and robot capabilities to suggest helpful actions; in doing so, we alleviate user burden, minimizing the amount of time partners spend explicitly instructing and supervising the robot. We evaluate ProVox through user studies grounded in household manipulation tasks (e.g., assembling lunch bags) that measure the efficiency of the collaboration, as well as features such as perceived helpfulness, ease of use, and reliability. Our analysis suggests that both meta-prompting and proactivity are critical, resulting in 38.7% faster task completion times and 31.9% less user burden relative to non-active baselines. Supplementary material, code, and videos can be found at this https URL. 

**Abstract (ZH)**: 协作机器人必须快速适应其合作伙伴的意图和偏好，以主动识别有助于任务执行的动作。特别是在人类合作伙伴可以持续教给机器人新的高层面行为、视觉概念和物理技能（例如通过示范）的情境中，这一点尤为重要，随着人机团队共同完成多样化的任务，机器人的能力也在不断增长。在本研究中，我们主张机器人应该能够在早期互动中推断出合作伙伴的目标，并利用这些信息在用户给出明确指令之前就主动规划行为。基于大型语言模型的强大常识先验和可控性，我们提出了ProVox（积极声音）这一新颖框架，使机器人能够有效地个性化并适应个体合作者。我们设计了一种元提示协议，使用户能够在开始物理互动之前能够传达其独特的偏好、意图和期望的机器人行为。ProVox然后利用个性化的提示条件一个主动的语言模型任务规划器，从当前互动的上下文和机器人能力中预测用户意图，以建议有助于任务执行的动作；在这种方式下，我们减轻了用户负担，减少了合作伙伴明确指导和监督机器人的时间。我们通过基于家庭操作任务（例如组装午餐袋）的用户研究评估了ProVox，这些任务评估了协作的效率以及诸如有用性感知、易用性和可靠性等特征。我们的分析表明，元提示和主动性都是至关重要的，与非活跃基线相比，任务完成时间快38.7%，用户负担减少31.9%。补充材料、代码和视频请访问此链接。 

---
# ViTaSCOPE: Visuo-tactile Implicit Representation for In-hand Pose and Extrinsic Contact Estimation 

**Title (ZH)**: ViTaSCOPE: 视触隐式表示用于手内姿态和外在接触估计 

**Authors**: Jayjun Lee, Nima Fazeli  

**Link**: [PDF](https://arxiv.org/pdf/2506.12239)  

**Abstract**: Mastering dexterous, contact-rich object manipulation demands precise estimation of both in-hand object poses and external contact locations$\unicode{x2013}$tasks particularly challenging due to partial and noisy observations. We present ViTaSCOPE: Visuo-Tactile Simultaneous Contact and Object Pose Estimation, an object-centric neural implicit representation that fuses vision and high-resolution tactile feedback. By representing objects as signed distance fields and distributed tactile feedback as neural shear fields, ViTaSCOPE accurately localizes objects and registers extrinsic contacts onto their 3D geometry as contact fields. Our method enables seamless reasoning over complementary visuo-tactile cues by leveraging simulation for scalable training and zero-shot transfers to the real-world by bridging the sim-to-real gap. We evaluate our method through comprehensive simulated and real-world experiments, demonstrating its capabilities in dexterous manipulation scenarios. 

**Abstract (ZH)**: 基于视觉和触觉同时估计物体姿态和外部接触位置：ViTaSCOPEmissive 触觉隐式表示 

---
# SPLATART: Articulated Gaussian Splatting with Estimated Object Structure 

**Title (ZH)**: SPLATART: 基于估计算法结构的articulated高斯抽样 

**Authors**: Stanley Lewis, Vishal Chandra, Tom Gao, Odest Chadwicke Jenkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.12184)  

**Abstract**: Representing articulated objects remains a difficult problem within the field of robotics. Objects such as pliers, clamps, or cabinets require representations that capture not only geometry and color information, but also part seperation, connectivity, and joint parametrization. Furthermore, learning these representations becomes even more difficult with each additional degree of freedom. Complex articulated objects such as robot arms may have seven or more degrees of freedom, and the depth of their kinematic tree may be notably greater than the tools, drawers, and cabinets that are the typical subjects of articulated object research. To address these concerns, we introduce SPLATART - a pipeline for learning Gaussian splat representations of articulated objects from posed images, of which a subset contains image space part segmentations. SPLATART disentangles the part separation task from the articulation estimation task, allowing for post-facto determination of joint estimation and representation of articulated objects with deeper kinematic trees than previously exhibited. In this work, we present data on the SPLATART pipeline as applied to the syntheic Paris dataset objects, and qualitative results on a real-world object under spare segmentation supervision. We additionally present on articulated serial chain manipulators to demonstrate usage on deeper kinematic tree structures. 

**Abstract (ZH)**: 基于 posed 图像学习articulated 对象的 Gaussian splat 表示 - SPLATART 管道 

---
# DoublyAware: Dual Planning and Policy Awareness for Temporal Difference Learning in Humanoid Locomotion 

**Title (ZH)**: 双重awareness: 人类体态运动中时差学习的双向规划与策略awareness 

**Authors**: Khang Nguyen, An T. Le, Jan Peters, Minh Nhat Vu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12095)  

**Abstract**: Achieving robust robot learning for humanoid locomotion is a fundamental challenge in model-based reinforcement learning (MBRL), where environmental stochasticity and randomness can hinder efficient exploration and learning stability. The environmental, so-called aleatoric, uncertainty can be amplified in high-dimensional action spaces with complex contact dynamics, and further entangled with epistemic uncertainty in the models during learning phases. In this work, we propose DoublyAware, an uncertainty-aware extension of Temporal Difference Model Predictive Control (TD-MPC) that explicitly decomposes uncertainty into two disjoint interpretable components, i.e., planning and policy uncertainties. To handle the planning uncertainty, DoublyAware employs conformal prediction to filter candidate trajectories using quantile-calibrated risk bounds, ensuring statistical consistency and robustness against stochastic dynamics. Meanwhile, policy rollouts are leveraged as structured informative priors to support the learning phase with Group-Relative Policy Constraint (GRPC) optimizers that impose a group-based adaptive trust-region in the latent action space. This principled combination enables the robot agent to prioritize high-confidence, high-reward behavior while maintaining effective, targeted exploration under uncertainty. Evaluated on the HumanoidBench locomotion suite with the Unitree 26-DoF H1-2 humanoid, DoublyAware demonstrates improved sample efficiency, accelerated convergence, and enhanced motion feasibility compared to RL baselines. Our simulation results emphasize the significance of structured uncertainty modeling for data-efficient and reliable decision-making in TD-MPC-based humanoid locomotion learning. 

**Abstract (ZH)**: 实现具备鲁棒性的类人机器人学习：基于模型的强化学习中的不确定性处理 

---
# Using Behavior Trees in Risk Assessment 

**Title (ZH)**: 使用行为树进行风险评估 

**Authors**: Razan Ghzouli, Atieh Hanna, Endre Erös, Rebekka Wohlrab  

**Link**: [PDF](https://arxiv.org/pdf/2506.12089)  

**Abstract**: Cyber-physical production systems increasingly involve collaborative robotic missions, requiring more demand for robust and safe missions. Industries rely on risk assessments to identify potential failures and implement measures to mitigate their risks. Although it is recommended to conduct risk assessments early in the design of robotic missions, the state of practice in the industry is different. Safety experts often struggle to completely understand robotics missions at the early design stages of projects and to ensure that the output of risk assessments is adequately considered during implementation.
This paper presents a design science study that conceived a model-based approach for early risk assessment in a development-centric way. Our approach supports risk assessment activities by using the behavior-tree model. We evaluated the approach together with five practitioners from four companies. Our findings highlight the potential of the behavior-tree model in supporting early identification, visualisation, and bridging the gap between code implementation and risk assessments' outputs. This approach is the first attempt to use the behavior-tree model to support risk assessment; thus, the findings highlight the need for further development. 

**Abstract (ZH)**: 基于模型的早期风险评估设计科学研究：行为树模型在机器人任务设计中的应用 

---
# Design and Development of a Robotic Transcatheter Delivery System for Aortic Valve Replacement 

**Title (ZH)**: 基于经导管输送系统的主动脉瓣置换机器人设计与开发 

**Authors**: Harith S. Gallage, Bailey F. De Sousa, Benjamin I. Chesnik, Chaikel G. Brownstein, Anson Paul, Ronghuai Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12082)  

**Abstract**: Minimally invasive transcatheter approaches are increasingly adopted for aortic stenosis treatment, where optimal commissural and coronary alignment is important. Achieving precise alignment remains clinically challenging, even with contemporary robotic transcatheter aortic valve replacement (TAVR) devices, as this task is still performed manually. This paper proposes the development of a robotic transcatheter delivery system featuring an omnidirectional bending joint and an actuation system designed to enhance positional accuracy and precision in TAVR procedures. The preliminary experimental results validate the functionality of this novel robotic system. 

**Abstract (ZH)**: 微创经导管 Approaches for 二尖瓣狭窄 治疗中，最佳隔膜和冠状动脉对齐至关重要。尽管使用当今的机器人经导管主动脉瓣置换（TAVR）设备，实现精确对齐仍然具有临床挑战性，因为这项任务仍然需要手动完成。本文提出开发一种配备全景弯曲关节和旨在增强 TAVR 程序中位置准确性和精度的驱动系统的机器人经导管输送系统。初步的实验结果验证了该新型机器人系统的功能。 

---
# Parallel Branch Model Predictive Control on GPUs 

**Title (ZH)**: GPU上并行分支模型预测控制 

**Authors**: Luyao Zhang, Chenghuai Lin, Sergio Grammatico  

**Link**: [PDF](https://arxiv.org/pdf/2506.13624)  

**Abstract**: We present a parallel GPU-accelerated solver for branch Model Predictive Control problems. Based on iterative LQR methods, our solver exploits the tree-sparse structure and implements temporal parallelism using the parallel scan algorithm. Consequently, the proposed solver enables parallelism across both the prediction horizon and the scenarios. In addition, we utilize an augmented Lagrangian method to handle general inequality constraints. We compare our solver with state-of-the-art numerical solvers in two automated driving applications. The numerical results demonstrate that, compared to CPU-based solvers, our solver achieves competitive performance for problems with short horizons and small-scale trees, while outperforming other solvers on large-scale problems. 

**Abstract (ZH)**: 我们提出了一种并行GPU加速的分支模型预测控制求解器。基于迭代LQR方法，该求解器利用树稀疏结构并采用并行扫描算法实现时间上的并行ism。因此，所提出的求解器能够在预测 horizon 和场景之间实现并行ism。此外，我们使用增广拉格朗日方法处理一般不等式约束。我们在两个自动驾驶应用中将该求解器与最先进的数值求解器进行比较。数值结果表明，对于短 horizon 和小规模树的问题，与基于CPU的求解器相比，该求解器具有竞争力的性能，而在大规模问题上则优于其他求解器。 

---
# Can you see how I learn? Human observers' inferences about Reinforcement Learning agents' learning processes 

**Title (ZH)**: 你能看出我是怎么学习的？人类观察者对强化学习代理学习过程的推断。 

**Authors**: Bernhard Hilpert, Muhan Hou, Kim Baraka, Joost Broekens  

**Link**: [PDF](https://arxiv.org/pdf/2506.13583)  

**Abstract**: Reinforcement Learning (RL) agents often exhibit learning behaviors that are not intuitively interpretable by human observers, which can result in suboptimal feedback in collaborative teaching settings. Yet, how humans perceive and interpret RL agent's learning behavior is largely unknown. In a bottom-up approach with two experiments, this work provides a data-driven understanding of the factors of human observers' understanding of the agent's learning process. A novel, observation-based paradigm to directly assess human inferences about agent learning was developed. In an exploratory interview study (\textit{N}=9), we identify four core themes in human interpretations: Agent Goals, Knowledge, Decision Making, and Learning Mechanisms. A second confirmatory study (\textit{N}=34) applied an expanded version of the paradigm across two tasks (navigation/manipulation) and two RL algorithms (tabular/function approximation). Analyses of 816 responses confirmed the reliability of the paradigm and refined the thematic framework, revealing how these themes evolve over time and interrelate. Our findings provide a human-centered understanding of how people make sense of agent learning, offering actionable insights for designing interpretable RL systems and improving transparency in Human-Robot Interaction. 

**Abstract (ZH)**: 强化学习（RL）代理往往表现出不直观的人类可解释的学习行为，这在协作教学环境中可能导致亚最优反馈。然而，人类如何感知和解释RL代理的学习行为尚不清楚。通过自下而上的两种实验，本工作提供了关于人类观察者理解代理学习过程的因素的数据驱动理解。开发了一种基于观察的新颖范式，直接评估人类对代理学习的推断。在探索性访谈研究（N=9）中，我们确定了人类解释中的四个核心主题：代理目标、知识、决策制定和学习机制。第二项确认性研究（N=34）使用扩展后的范式在两个任务（导航/操作）和两种RL算法（表象/函数逼近）上进行。对816个响应的分析证实了该范式的可靠性，并细化了主题框架，揭示了这些主题随时间的变化及其相互关系。我们的发现提供了关于人们如何理解代理学习的人本中心理解，为设计可解释的RL系统和提高人机交互中的透明度提供了可操作的见解。 

---
# UAV Object Detection and Positioning in a Mining Industrial Metaverse with Custom Geo-Referenced Data 

**Title (ZH)**: 矿产工业元宇宙中基于自定义地理参考数据的无人机目标检测与定位 

**Authors**: Vasiliki Balaska, Ioannis Tsampikos Papapetros, Katerina Maria Oikonomou, Loukas Bampis, Antonios Gasteratos  

**Link**: [PDF](https://arxiv.org/pdf/2506.13505)  

**Abstract**: The mining sector increasingly adopts digital tools to improve operational efficiency, safety, and data-driven decision-making. One of the key challenges remains the reliable acquisition of high-resolution, geo-referenced spatial information to support core activities such as extraction planning and on-site monitoring. This work presents an integrated system architecture that combines UAV-based sensing, LiDAR terrain modeling, and deep learning-based object detection to generate spatially accurate information for open-pit mining environments. The proposed pipeline includes geo-referencing, 3D reconstruction, and object localization, enabling structured spatial outputs to be integrated into an industrial digital twin platform. Unlike traditional static surveying methods, the system offers higher coverage and automation potential, with modular components suitable for deployment in real-world industrial contexts. While the current implementation operates in post-flight batch mode, it lays the foundation for real-time extensions. The system contributes to the development of AI-enhanced remote sensing in mining by demonstrating a scalable and field-validated geospatial data workflow that supports situational awareness and infrastructure safety. 

**Abstract (ZH)**: 采矿业 increasingly 采用数字工具以提高运营效率、安全性和数据驱动的决策能力。可靠获取高分辨率、地理参考的空间信息仍然是关键挑战，以支持诸如开采规划和现场监测等核心活动。本研究提出了一种集成系统架构，结合了无人机载感测、LiDAR地形建模和基于深度学习的对象检测，以生成适用于露天采矿环境的空间准确信息。提议的管道包括地理参考、3D重建和物体定位，使结构化空间输出能够集成到工业数字孪生平台中。与传统的静态测量方法不同，该系统提供了更广泛的覆盖范围和更高的自动化潜力，模块化的组件适用于在实际工业环境中部署。尽管当前实现运行在飞行后批处理模式，但它为实时扩展奠定了基础。该系统通过演示一个可扩展且实地验证的地理空间数据流程，支持态势感知和基础设施安全，为采矿中人工智能增强的遥感发展做出了贡献。 

---
# Block-wise Adaptive Caching for Accelerating Diffusion Policy 

**Title (ZH)**: 区块适应性缓存加速扩散策略 

**Authors**: Kangye Ji, Yuan Meng, Hanyun Cui, Ye Li, Shengjia Hua, Lei Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13456)  

**Abstract**: Diffusion Policy has demonstrated strong visuomotor modeling capabilities, but its high computational cost renders it impractical for real-time robotic control. Despite huge redundancy across repetitive denoising steps, existing diffusion acceleration techniques fail to generalize to Diffusion Policy due to fundamental architectural and data divergences. In this paper, we propose Block-wise Adaptive Caching(BAC), a method to accelerate Diffusion Policy by caching intermediate action features. BAC achieves lossless action generation acceleration by adaptively updating and reusing cached features at the block level, based on a key observation that feature similarities vary non-uniformly across timesteps and locks. To operationalize this insight, we first propose the Adaptive Caching Scheduler, designed to identify optimal update timesteps by maximizing the global feature similarities between cached and skipped features. However, applying this scheduler for each block leads to signiffcant error surges due to the inter-block propagation of caching errors, particularly within Feed-Forward Network (FFN) blocks. To mitigate this issue, we develop the Bubbling Union Algorithm, which truncates these errors by updating the upstream blocks with signiffcant caching errors before downstream FFNs. As a training-free plugin, BAC is readily integrable with existing transformer-based Diffusion Policy and vision-language-action models. Extensive experiments on multiple robotic benchmarks demonstrate that BAC achieves up to 3x inference speedup for free. 

**Abstract (ZH)**: Block-wise Adaptive Caching for Accelerating Diffusion Policy 

---
# Open-Set LiDAR Panoptic Segmentation Guided by Uncertainty-Aware Learning 

**Title (ZH)**: 开集LiDAR全景分割：基于不确定性意识学习 

**Authors**: Rohit Mohan, Julia Hindel, Florian Drews, Claudius Gläser, Daniele Cattaneo, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2506.13265)  

**Abstract**: Autonomous vehicles that navigate in open-world environments may encounter previously unseen object classes. However, most existing LiDAR panoptic segmentation models rely on closed-set assumptions, failing to detect unknown object instances. In this work, we propose ULOPS, an uncertainty-guided open-set panoptic segmentation framework that leverages Dirichlet-based evidential learning to model predictive uncertainty. Our architecture incorporates separate decoders for semantic segmentation with uncertainty estimation, embedding with prototype association, and instance center prediction. During inference, we leverage uncertainty estimates to identify and segment unknown instances. To strengthen the model's ability to differentiate between known and unknown objects, we introduce three uncertainty-driven loss functions. Uniform Evidence Loss to encourage high uncertainty in unknown regions. Adaptive Uncertainty Separation Loss ensures a consistent difference in uncertainty estimates between known and unknown objects at a global scale. Contrastive Uncertainty Loss refines this separation at the fine-grained level. To evaluate open-set performance, we extend benchmark settings on KITTI-360 and introduce a new open-set evaluation for nuScenes. Extensive experiments demonstrate that ULOPS consistently outperforms existing open-set LiDAR panoptic segmentation methods. 

**Abstract (ZH)**: 面向未知类别的不确定引导开放集点云语义分割框架ULOPS 

---
# Multimodal "Puppeteer": An Exploration of Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality 

**Title (ZH)**: 多模态“操偶师”：基于LLM驱动的语音和手势交互在增强现实中的虚拟对应物机器人远程操作探索 

**Authors**: Yuchong Zhang, Bastian Orthmann, Shichen Ji, Michael Welle, Jonne Van Haastregt, Danica Kragic  

**Link**: [PDF](https://arxiv.org/pdf/2506.13189)  

**Abstract**: The integration of robotics and augmented reality (AR) holds transformative potential for advancing human-robot interaction (HRI), offering enhancements in usability, intuitiveness, accessibility, and collaborative task performance. This paper introduces and evaluates a novel multimodal AR-based robot puppeteer framework that enables intuitive teleoperation via virtual counterpart through large language model (LLM)-driven voice commands and hand gesture interactions. Utilizing the Meta Quest 3, users interact with a virtual counterpart robot in real-time, effectively "puppeteering" its physical counterpart within an AR environment. We conducted a within-subject user study with 42 participants performing robotic cube pick-and-place with pattern matching tasks under two conditions: gesture-only interaction and combined voice-and-gesture interaction. Both objective performance metrics and subjective user experience (UX) measures were assessed, including an extended comparative analysis between roboticists and non-roboticists. The results provide key insights into how multimodal input influences contextual task efficiency, usability, and user satisfaction in AR-based HRI. Our findings offer practical design implications for designing effective AR-enhanced HRI systems. 

**Abstract (ZH)**: 机器人与增强现实技术的整合为提升人机交互（HRI）具有变革性的潜力，通过提供易用性、直观性、可访问性和协作任务性能的增强。本文介绍并评估了一种新颖的基于多模态AR的机器人傀儡师框架，该框架通过大型语言模型（LLM）驱动的语音命令和手势交互，实现了虚拟对应物的直观远程操作。利用Meta Quest 3，用户可以实时与虚拟对应物机器人进行交互，并在AR环境中有效地“操纵”其物理对应物。我们对42名参与者进行了单一被试者用户研究，他们在两种条件下完成机器人立方体匹配任务：仅手势交互和结合语音-手势交互。我们评估了客观性能指标和主观用户体验（UX）指标，包括对机器人专家与非机器人专家的扩展比较分析。研究结果提供了关于多模态输入如何影响基于AR的人机交互环境中的任务效率、易用性和用户满意度的关键见解。我们的发现为设计有效增强的AR人机交互系统提供了实用的设计建议。 

---
# SuperPoint-SLAM3: Augmenting ORB-SLAM3 with Deep Features, Adaptive NMS, and Learning-Based Loop Closure 

**Title (ZH)**: SuperPoint-SLAM3: 结合深度特征、自适应非极大值抑制和基于学习的环形闭回路的ORB-SLAM3增强版 

**Authors**: Shahram Najam Syed, Ishir Roongta, Kavin Ravie, Gangadhar Nageswar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13089)  

**Abstract**: Visual simultaneous localization and mapping (SLAM) must remain accurate under extreme viewpoint, scale and illumination variations. The widely adopted ORB-SLAM3 falters in these regimes because it relies on hand-crafted ORB keypoints. We introduce SuperPoint-SLAM3, a drop-in upgrade that (i) replaces ORB with the self-supervised SuperPoint detector--descriptor, (ii) enforces spatially uniform keypoints via adaptive non-maximal suppression (ANMS), and (iii) integrates a lightweight NetVLAD place-recognition head for learning-based loop closure.
On the KITTI Odometry benchmark SuperPoint-SLAM3 reduces mean translational error from 4.15% to 0.34% and mean rotational error from 0.0027 deg/m to 0.0010 deg/m. On the EuRoC MAV dataset it roughly halves both errors across every sequence (e.g., V2\_03: 1.58% -> 0.79%). These gains confirm that fusing modern deep features with a learned loop-closure module markedly improves ORB-SLAM3 accuracy while preserving its real-time operation.
Implementation, pretrained weights and reproducibility scripts are available at this https URL. 

**Abstract (ZH)**: SuperPoint-SLAM3：一种在极端视角、尺度和光照变化下保持准确的同时定位与建图方法 

---
# Bridging Data-Driven and Physics-Based Models: A Consensus Multi-Model Kalman Filter for Robust Vehicle State Estimation 

**Title (ZH)**: 数据驱动与物理模型融合的共识多模型卡尔曼滤波器：稳健的车辆状态估计 

**Authors**: Farid Mafi, Ladan Khoshnevisan, Mohammad Pirani, Amir Khajepour  

**Link**: [PDF](https://arxiv.org/pdf/2506.12862)  

**Abstract**: Vehicle state estimation presents a fundamental challenge for autonomous driving systems, requiring both physical interpretability and the ability to capture complex nonlinear behaviors across diverse operating conditions. Traditional methodologies often rely exclusively on either physics-based or data-driven models, each with complementary strengths and limitations that become most noticeable during critical scenarios. This paper presents a novel consensus multi-model Kalman filter framework that integrates heterogeneous model types to leverage their complementary strengths while minimizing individual weaknesses. We introduce two distinct methodologies for handling covariance propagation in data-driven models: a Koopman operator-based linearization approach enabling analytical covariance propagation, and an ensemble-based method providing unified uncertainty quantification across model types without requiring pretraining. Our approach implements an iterative consensus fusion procedure that dynamically weighs different models based on their demonstrated reliability in current operating conditions. The experimental results conducted on an electric all-wheel-drive Equinox vehicle demonstrate performance improvements over single-model techniques, with particularly significant advantages during challenging maneuvers and varying road conditions, confirming the effectiveness and robustness of the proposed methodology for safety-critical autonomous driving applications. 

**Abstract (ZH)**: 车辆状态估计是自主驾驶系统中的一个根本挑战，要求同时具备物理可解释性和捕捉多变操作条件下的复杂非线性行为能力。传统方法通常仅依赖于物理模型或数据驱动模型，每种方法都有其互补的优势和限制，在关键场景中尤为明显。本文提出了一种新颖的共识多模型卡尔曼滤波框架，该框架集成了异构模型类型，充分利用其互补优势并最小化各自的弱点。我们介绍了两种不同的方法来处理数据驱动模型中的协方差传播：基于科廷曼算子的线性化方法以实现分析性的协方差传播，以及基于ensemble的方法以在不同模型类型中提供统一的不确定性量化，无需预先训练。我们的方法实现了一种迭代共识融合过程，该过程根据模型在当前操作条件下的可靠性动态加权。在一辆全轮驱动的电动雪佛兰Equinox车辆上的实验结果表明，与单模型技术相比，该方法在复杂的操作和变化的道路条件下表现出显著的优势，验证了所提出方法在安全关键的自主驾驶应用中的有效性和稳健性。 

---
# Enhancing Rating-Based Reinforcement Learning to Effectively Leverage Feedback from Large Vision-Language Models 

**Title (ZH)**: 基于评分的强化学习增强以有效地利用大型视觉-语言模型的反馈 

**Authors**: Tung Minh Luu, Younghwan Lee, Donghoon Lee, Sunho Kim, Min Jun Kim, Chang D. Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.12822)  

**Abstract**: Designing effective reward functions remains a fundamental challenge in reinforcement learning (RL), as it often requires extensive human effort and domain expertise. While RL from human feedback has been successful in aligning agents with human intent, acquiring high-quality feedback is costly and labor-intensive, limiting its scalability. Recent advancements in foundation models present a promising alternative--leveraging AI-generated feedback to reduce reliance on human supervision in reward learning. Building on this paradigm, we introduce ERL-VLM, an enhanced rating-based RL method that effectively learns reward functions from AI feedback. Unlike prior methods that rely on pairwise comparisons, ERL-VLM queries large vision-language models (VLMs) for absolute ratings of individual trajectories, enabling more expressive feedback and improved sample efficiency. Additionally, we propose key enhancements to rating-based RL, addressing instability issues caused by data imbalance and noisy labels. Through extensive experiments across both low-level and high-level control tasks, we demonstrate that ERL-VLM significantly outperforms existing VLM-based reward generation methods. Our results demonstrate the potential of AI feedback for scaling RL with minimal human intervention, paving the way for more autonomous and efficient reward learning. 

**Abstract (ZH)**: 设计有效的奖励函数仍然是强化学习中的一项基本挑战，这通常需要大量的人力投入和专业知识。虽然从人类反馈中进行的强化学习在使智能体与人类意图保持一致方面取得了成功，但获取高质量的反馈代价高昂且劳动密集，限制了其可扩展性。近期基础模型的进展提供了一种有前景的替代方案——利用AI生成的反馈来减少对人类监督的依赖。在此基础上，我们引入了ERL-VLM，这是一种增强的基于评分的强化学习方法，可以从AI反馈中有效学习奖励函数。与以前依赖成对比较的方法不同，ERL-VLM 查询大型视觉-语言模型（VLM）以获取单个轨迹的绝对评分，这使得反馈更具表现力并提高了样本效率。此外，我们还提出了基于评分的强化学习的关键改进，以解决由数据不平衡和嘈杂标签引起的数据不稳定问题。通过在不同层面的控制任务上的广泛实验，我们证明了ERL-VLM 显著优于现有的基于VLM的奖励生成方法。我们的结果表明，AI反馈有望在最少的人工干预下扩大强化学习的应用范围，为更自主和高效的奖励学习铺平了道路。 

---
# Trust-MARL: Trust-Based Multi-Agent Reinforcement Learning Framework for Cooperative On-Ramp Merging Control in Heterogeneous Traffic Flow 

**Title (ZH)**: 基于信任的多Agent强化学习框架：异质交通流中合作匝道并线控制的Trust-MARL 

**Authors**: Jie Pan, Tianyi Wang, Christian Claudel, Jing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12600)  

**Abstract**: Intelligent transportation systems require connected and automated vehicles (CAVs) to conduct safe and efficient cooperation with human-driven vehicles (HVs) in complex real-world traffic environments. However, the inherent unpredictability of human behaviour, especially at bottlenecks such as highway on-ramp merging areas, often disrupts traffic flow and compromises system performance. To address the challenge of cooperative on-ramp merging in heterogeneous traffic environments, this study proposes a trust-based multi-agent reinforcement learning (Trust-MARL) framework. At the macro level, Trust-MARL enhances global traffic efficiency by leveraging inter-agent trust to improve bottleneck throughput and mitigate traffic shockwave through emergent group-level coordination. At the micro level, a dynamic trust mechanism is designed to enable CAVs to adjust their cooperative strategies in response to real-time behaviors and historical interactions with both HVs and other CAVs. Furthermore, a trust-triggered game-theoretic decision-making module is integrated to guide each CAV in adapting its cooperation factor and executing context-aware lane-changing decisions under safety, comfort, and efficiency constraints. An extensive set of ablation studies and comparative experiments validates the effectiveness of the proposed Trust-MARL approach, demonstrating significant improvements in safety, efficiency, comfort, and adaptability across varying CAV penetration rates and traffic densities. 

**Abstract (ZH)**: 基于信任的多agents强化学习（Trust-MARL）框架：在异质交通环境中实现安全、高效的合作匝道并线 

---
# Constrained Diffusers for Safe Planning and Control 

**Title (ZH)**: 受约束的扩散器用于安全规划与控制 

**Authors**: Jichen Zhang, Liqun Zhao, Antonis Papachristodoulou, Jack Umenberger  

**Link**: [PDF](https://arxiv.org/pdf/2506.12544)  

**Abstract**: Diffusion models have shown remarkable potential in planning and control tasks due to their ability to represent multimodal distributions over actions and trajectories. However, ensuring safety under constraints remains a critical challenge for diffusion models. This paper proposes Constrained Diffusers, a novel framework that incorporates constraints into pre-trained diffusion models without retraining or architectural modifications. Inspired by constrained optimization, we apply a constrained Langevin sampling mechanism for the reverse diffusion process that jointly optimizes the trajectory and realizes constraint satisfaction through three iterative algorithms: projected method, primal-dual method and augmented Lagrangian approaches. In addition, we incorporate discrete control barrier functions as constraints for constrained diffusers to guarantee safety in online implementation. Experiments in Maze2D, locomotion, and pybullet ball running tasks demonstrate that our proposed methods achieve constraint satisfaction with less computation time, and are competitive to existing methods in environments with static and time-varying constraints. 

**Abstract (ZH)**: 约束扩散器：一种无需重新训练或修改架构将约束整合到预训练扩散模型中的新型框架 

---
# Efficient Multi-Camera Tokenization with Triplanes for End-to-End Driving 

**Title (ZH)**: 端到端驾驶中的高效三平面多摄像头分词 

**Authors**: Boris Ivanovic, Cristiano Saltori, Yurong You, Yan Wang, Wenjie Luo, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2506.12251)  

**Abstract**: Autoregressive Transformers are increasingly being deployed as end-to-end robot and autonomous vehicle (AV) policy architectures, owing to their scalability and potential to leverage internet-scale pretraining for generalization. Accordingly, tokenizing sensor data efficiently is paramount to ensuring the real-time feasibility of such architectures on embedded hardware. To this end, we present an efficient triplane-based multi-camera tokenization strategy that leverages recent advances in 3D neural reconstruction and rendering to produce sensor tokens that are agnostic to the number of input cameras and their resolution, while explicitly accounting for their geometry around an AV. Experiments on a large-scale AV dataset and state-of-the-art neural simulator demonstrate that our approach yields significant savings over current image patch-based tokenization strategies, producing up to 72% fewer tokens, resulting in up to 50% faster policy inference while achieving the same open-loop motion planning accuracy and improved offroad rates in closed-loop driving simulations. 

**Abstract (ZH)**: 自回归变压器越来越多地被用作机器人和自动驾驶车辆（AV）策略架构的端到端模型，得益于其可扩展性和通过互联网规模预训练进行泛化的潜力。因此，高效地 tokenize 传感器数据对于确保此类架构在嵌入式硬件上的实时可行性至关重要。为此，我们提出了一种基于三平面的多相机 tokenize 策略，利用最近在三维神经重建和渲染方面的进展，生成与输入相机数量和分辨率无关的传感器 tokenize，同时明确考虑其在自动驾驶车辆周围的几何关系。在大规模自动驾驶车辆数据集和最先进的神经模拟器上的实验表明，与当前基于图像块的 tokenize 策略相比，我们的方法可实现显著节约，生成多达 72% 的较少 tokenize，从而在保持开环运动规划精度相同的同时，闭环驾驶模拟表现出更快的策略推理速度和更高的离路率。 

---
