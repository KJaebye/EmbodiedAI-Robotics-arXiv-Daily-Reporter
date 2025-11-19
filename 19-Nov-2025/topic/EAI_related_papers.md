# HMC: Learning Heterogeneous Meta-Control for Contact-Rich Loco-Manipulation 

**Title (ZH)**: HMC：学习异构元控制以应对丰富的接触式移动操作 

**Authors**: Lai Wei, Xuanbin Peng, Ri-Zhao Qiu, Tianshu Huang, Xuxin Cheng, Xiaolong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14756)  

**Abstract**: Learning from real-world robot demonstrations holds promise for interacting with complex real-world environments. However, the complexity and variability of interaction dynamics often cause purely positional controllers to struggle with contacts or varying payloads. To address this, we propose a Heterogeneous Meta-Control (HMC) framework for Loco-Manipulation that adaptively stitches multiple control modalities: position, impedance, and hybrid force-position. We first introduce an interface, HMC-Controller, for blending actions from different control profiles continuously in the torque space. HMC-Controller facilitates both teleoperation and policy deployment. Then, to learn a robust force-aware policy, we propose HMC-Policy to unify different controllers into a heterogeneous architecture. We adopt a mixture-of-experts style routing to learn from large-scale position-only data and fine-grained force-aware demonstrations. Experiments on a real humanoid robot show over 50% relative improvement vs. baselines on challenging tasks such as compliant table wiping and drawer opening, demonstrating the efficacy of HMC. 

**Abstract (ZH)**: 基于真实机器人演示的学习为处理复杂真实环境中的交互提供了 promise。然而，交互动力学的复杂性和变异性往往导致仅位置控制在处理接触或变化载荷时遇到困难。为了解决这一问题，我们提出了一种异构元控制（HMC）框架，用于适应性地组合多种控制模态：位置、阻抗和混合力-位置控制。我们首先引入了一个接口 HMC-Controller，用于在扭矩空间中连续混合不同控制配置文件下的动作。HMC-Controller 既支持远程操作，也支持策略部署。然后，为了学习一种稳健的力感知策略，我们提出了 HMC-Policy 以将不同控制器统一到一个异构架构中。我们采用专家混合样式路由来从大规模仅位置数据和细腻的力感知演示中学习。实验在真实的人形机器人上表明，在诸如顺应性桌子擦洗和抽屉打开等具有挑战性的任务上，HMC 相较于基线有超过 50% 的相对改进，这展示了 HMC 的有效性。 

---
# NORA-1.5: A Vision-Language-Action Model Trained using World Model- and Action-based Preference Rewards 

**Title (ZH)**: NORA-1.5：一种基于世界模型和动作偏好奖励训练的视觉-语言-行动模型 

**Authors**: Chia-Yu Hung, Navonil Majumder, Haoyuan Deng, Liu Renhang, Yankang Ang, Amir Zadeh, Chuan Li, Dorien Herremans, Ziwei Wang, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2511.14659)  

**Abstract**: Vision--language--action (VLA) models have recently shown promising performance on a variety of embodied tasks, yet they still fall short in reliability and generalization, especially when deployed across different embodiments or real-world environments. In this work, we introduce NORA-1.5, a VLA model built from the pre-trained NORA backbone by adding to it a flow-matching-based action expert. This architectural enhancement alone yields substantial performance gains, enabling NORA-1.5 to outperform NORA and several state-of-the-art VLA models across both simulated and real-world benchmarks. To further improve robustness and task success, we develop a set of reward models for post-training VLA policies. Our rewards combine (i) an action-conditioned world model (WM) that evaluates whether generated actions lead toward the desired goal, and (ii) a deviation-from-ground-truth heuristic that distinguishes good actions from poor ones. Using these reward signals, we construct preference datasets and adapt NORA-1.5 to target embodiments through direct preference optimization (DPO). Extensive evaluations show that reward-driven post-training consistently improves performance in both simulation and real-robot settings, demonstrating significant VLA model-reliability gains through simple yet effective reward models. Our findings highlight NORA-1.5 and reward-guided post-training as a viable path toward more dependable embodied agents suitable for real-world deployment. 

**Abstract (ZH)**: Vision-LANGUAGE-Action (VLA)模型在多种物理任务中展现了有前途的表现，但仍存在可靠性和泛化能力不足的问题，尤其是在跨不同物理载体或现实环境部署时更为明显。在这项工作中，我们引入了NORA-1.5，这是一种基于预训练NORA主干构建的VLA模型，通过添加基于流匹配的动作专家来改进其架构。这种架构增强单独就带来了显著的性能提升，使NORA-1.5在模拟和现实世界的基准测试中均超过了NORA和其他几种最先进的VLA模型。为了进一步提高鲁棒性和任务成功率，我们开发了一套用于后训练VLA策略的奖励模型。我们的奖励模型结合了（i）动作条件的世界模型（WM），评估生成的动作是否朝向目标，以及（ii）偏差度量，区分好的动作和差的动作。利用这些奖励信号，我们构建了偏好数据集，并通过直接偏好优化（DPO）来适应NORA-1.5以针对特定的物理载体。广泛的评估表明，奖励驱动的后训练在模拟和真实机器人环境中均能一致地提升性能，通过简单的有效奖励模型显著提高了VLA模型的可靠性。我们的研究结果突出显示NORA-1.5和基于奖励的后训练是迈向更可靠的物理载体代理用于现实世界部署的可行路径。 

---
# Gallant: Voxel Grid-based Humanoid Locomotion and Local-navigation across 3D Constrained Terrains 

**Title (ZH)**: Gallant: 基于体素网格的人形移动和3D受限地形中的局部导航 

**Authors**: Qingwei Ben, Botian Xu, Kailin Li, Feiyu Jia, Wentao Zhang, Jingping Wang, Jingbo Wang, Dahua Lin, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14625)  

**Abstract**: Robust humanoid locomotion requires accurate and globally consistent perception of the surrounding 3D environment. However, existing perception modules, mainly based on depth images or elevation maps, offer only partial and locally flattened views of the environment, failing to capture the full 3D structure. This paper presents Gallant, a voxel-grid-based framework for humanoid locomotion and local navigation in 3D constrained terrains. It leverages voxelized LiDAR data as a lightweight and structured perceptual representation, and employs a z-grouped 2D CNN to map this representation to the control policy, enabling fully end-to-end optimization. A high-fidelity LiDAR simulation that dynamically generates realistic observations is developed to support scalable, LiDAR-based training and ensure sim-to-real consistency. Experimental results show that Gallant's broader perceptual coverage facilitates the use of a single policy that goes beyond the limitations of previous methods confined to ground-level obstacles, extending to lateral clutter, overhead constraints, multi-level structures, and narrow passages. Gallant also firstly achieves near 100% success rates in challenging scenarios such as stair climbing and stepping onto elevated platforms through improved end-to-end optimization. 

**Abstract (ZH)**: 基于体素网格的鲁棒类人三维环境感知框架：用于三维受限制地形的类人行动与局部导航 

---
# Is Your VLM for Autonomous Driving Safety-Ready? A Comprehensive Benchmark for Evaluating External and In-Cabin Risks 

**Title (ZH)**: 你的VLM为自动驾驶安全做好准备了吗？一种全面的基准评估外部和车内风险 

**Authors**: Xianhui Meng, Yuchen Zhang, Zhijian Huang, Zheng Lu, Ziling Ji, Yaoyao Yin, Hongyuan Zhang, Guangfeng Jiang, Yandan Lin, Long Chen, Hangjun Ye, Li Zhang, Jun Liu, Xiaoshuai Hao  

**Link**: [PDF](https://arxiv.org/pdf/2511.14592)  

**Abstract**: Vision-Language Models (VLMs) show great promise for autonomous driving, but their suitability for safety-critical scenarios is largely unexplored, raising safety concerns. This issue arises from the lack of comprehensive benchmarks that assess both external environmental risks and in-cabin driving behavior safety simultaneously. To bridge this critical gap, we introduce DSBench, the first comprehensive Driving Safety Benchmark designed to assess a VLM's awareness of various safety risks in a unified manner. DSBench encompasses two major categories: external environmental risks and in-cabin driving behavior safety, divided into 10 key categories and a total of 28 sub-categories. This comprehensive evaluation covers a wide range of scenarios, ensuring a thorough assessment of VLMs' performance in safety-critical contexts. Extensive evaluations across various mainstream open-source and closed-source VLMs reveal significant performance degradation under complex safety-critical situations, highlighting urgent safety concerns. To address this, we constructed a large dataset of 98K instances focused on in-cabin and external safety scenarios, showing that fine-tuning on this dataset significantly enhances the safety performance of existing VLMs and paves the way for advancing autonomous driving technology. The benchmark toolkit, code, and model checkpoints will be publicly accessible. 

**Abstract (ZH)**: 基于视觉-语言模型的驾驶安全性基准 (DSBench): 评估自主驾驶中的各种安全风险 

---
# Self-Supervised Multisensory Pretraining for Contact-Rich Robot Reinforcement Learning 

**Title (ZH)**: 接触丰富的机器人强化学习自我监督多感知预训练 

**Authors**: Rickmer Krohn, Vignesh Prasad, Gabriele Tiboni, Georgia Chalvatzaki  

**Link**: [PDF](https://arxiv.org/pdf/2511.14427)  

**Abstract**: Effective contact-rich manipulation requires robots to synergistically leverage vision, force, and proprioception. However, Reinforcement Learning agents struggle to learn in such multisensory settings, especially amidst sensory noise and dynamic changes. We propose MultiSensory Dynamic Pretraining (MSDP), a novel framework for learning expressive multisensory representations tailored for task-oriented policy learning. MSDP is based on masked autoencoding and trains a transformer-based encoder by reconstructing multisensory observations from only a subset of sensor embeddings, leading to cross-modal prediction and sensor fusion. For downstream policy learning, we introduce a novel asymmetric architecture, where a cross-attention mechanism allows the critic to extract dynamic, task-specific features from the frozen embeddings, while the actor receives a stable pooled representation to guide its actions. Our method demonstrates accelerated learning and robust performance under diverse perturbations, including sensor noise, and changes in object dynamics. Evaluations in multiple challenging, contact-rich robot manipulation tasks in simulation and the real world showcase the effectiveness of MSDP. Our approach exhibits strong robustness to perturbations and achieves high success rates on the real robot with as few as 6,000 online interactions, offering a simple yet powerful solution for complex multisensory robotic control. 

**Abstract (ZH)**: 多感官动态预训练：面向任务的多感官表示学习的新框架 

---
# Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning 

**Title (ZH)**: 连续的视觉-语言-行动协同学习及其语义-物理对齐行为克隆 

**Authors**: Xiuxiu Qi, Yu Yang, Jiannong Cao, Luyao Bai, Chongshan Fan, Chengtai Cao, Hongpeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14396)  

**Abstract**: Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states. 

**Abstract (ZH)**: 基于语言条件的操控促进通过行为克隆的人机交互：Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL) 

---
# Going Places: Place Recognition in Artificial and Natural Systems 

**Title (ZH)**: 探索场所：人工与自然系统中的场所识别 

**Authors**: Michael Milford, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2511.14341)  

**Abstract**: Place recognition, the ability to identify previously visited locations, is critical for both biological navigation and autonomous systems. This review synthesizes findings from robotic systems, animal studies, and human research to explore how different systems encode and recall place. We examine the computational and representational strategies employed across artificial systems, animals, and humans, highlighting convergent solutions such as topological mapping, cue integration, and memory management. Animal systems reveal evolved mechanisms for multimodal navigation and environmental adaptation, while human studies provide unique insights into semantic place concepts, cultural influences, and introspective capabilities. Artificial systems showcase scalable architectures and data-driven models. We propose a unifying set of concepts by which to consider and develop place recognition mechanisms and identify key challenges such as generalization, robustness, and environmental variability. This review aims to foster innovations in artificial localization by connecting future developments in artificial place recognition systems to insights from both animal navigation research and human spatial cognition studies. 

**Abstract (ZH)**: Place识别，即识别先前到访地点的能力，对于生物导航和自主系统都至关重要。本文综述了来自机器人系统、动物研究和人类研究的发现，探讨不同系统如何编码和回忆地点。我们考察了人工系统、动物和人类在计算和表示策略上的差异，突出了拓扑映射、线索整合和记忆管理等一致的解决方案。动物系统揭示了多模态导航和环境适应的进化机制，而人类研究则提供了关于语义地点概念、文化影响和内省能力的独特见解。人工系统展示了可扩展的架构和数据驱动的模型。我们提出了一组统一的概念，以考虑和发展Place识别机制，并指出了泛化能力、鲁棒性和环境变化等关键挑战。本文旨在通过将未来的人工场所识别系统发展与动物导航研究和人类空间认知研究的见解联系起来，促进人工定位技术的创新。 

---
# MA-SLAM: Active SLAM in Large-Scale Unknown Environment using Map Aware Deep Reinforcement Learning 

**Title (ZH)**: MA-SLAM: 基于地图aware的深度强化学习在大规模未知环境中的主动SLAM 

**Authors**: Yizhen Yin, Yuhua Qi, Dapeng Feng, Hongbo Chen, Hongjun Ma, Jin Wu, Yi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.14330)  

**Abstract**: Active Simultaneous Localization and Mapping (Active SLAM) involves the strategic planning and precise control of a robotic system's movement in order to construct a highly accurate and comprehensive representation of its surrounding environment, which has garnered significant attention within the research community. While the current methods demonstrate efficacy in small and controlled settings, they face challenges when applied to large-scale and diverse environments, marked by extended periods of exploration and suboptimal paths of discovery. In this paper, we propose MA-SLAM, a Map-Aware Active SLAM system based on Deep Reinforcement Learning (DRL), designed to address the challenge of efficient exploration in large-scale environments. In pursuit of this objective, we put forward a novel structured map representation. By discretizing the spatial data and integrating the boundary points and the historical trajectory, the structured map succinctly and effectively encapsulates the visited regions, thereby serving as input for the deep reinforcement learning based decision module. Instead of sequentially predicting the next action step within the decision module, we have implemented an advanced global planner to optimize the exploration path by leveraging long-range target points. We conducted experiments in three simulation environments and deployed in a real unmanned ground vehicle (UGV), the results demonstrate that our approach significantly reduces both the duration and distance of exploration compared with state-of-the-art methods. 

**Abstract (ZH)**: 基于深度强化学习的意识地图规划SLAM (MA-SLAM) 

---
# Towards Deploying VLA without Fine-Tuning: Plug-and-Play Inference-Time VLA Policy Steering via Embodied Evolutionary Diffusion 

**Title (ZH)**: 无需微调部署VLA：基于封装进化扩散的检索时VLA策略插件式控制 

**Authors**: Zhuo Li, Junjia Liu, Zhipeng Dong, Tao Teng, Quentin Rouxel, Darwin Caldwell, Fei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.14178)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated significant potential in real-world robotic manipulation. However, pre-trained VLA policies still suffer from substantial performance degradation during downstream deployment. Although fine-tuning can mitigate this issue, its reliance on costly demonstration collection and intensive computation makes it impractical in real-world settings. In this work, we introduce VLA-Pilot, a plug-and-play inference-time policy steering method for zero-shot deployment of pre-trained VLA without any additional fine-tuning or data collection. We evaluate VLA-Pilot on six real-world downstream manipulation tasks across two distinct robotic embodiments, encompassing both in-distribution and out-of-distribution scenarios. Experimental results demonstrate that VLA-Pilot substantially boosts the success rates of off-the-shelf pre-trained VLA policies, enabling robust zero-shot generalization to diverse tasks and embodiments. Experimental videos and code are available at: this https URL. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型在实际机器人操作中展现了显著潜力。然而，预训练的VLA策略在下游部署中仍遭受显著性能下降。虽然微调可以缓解这一问题，但其依赖于成本高昂的演示收集和密集计算，使得在实际应用场景中不可行。本文介绍了一种名为VLA-Pilot的即插即用推理时策略引导方法，用于无需额外微调或数据收集即可实现预训练VLA的零样本部署。我们在两种不同机器人载体上的六个真实世界下游操作任务上评估了VLA-Pilot，涵盖分布内和分布外场景。实验结果表明，VLA-Pilot显著提升了现成预训练VLA策略的成功率，使其能够在多种任务和载体上实现鲁棒的零样本泛化。实验视频和代码可在以下链接获取：this https URL。 

---
# RoboTidy : A 3D Gaussian Splatting Household Tidying Benchmark for Embodied Navigation and Action 

**Title (ZH)**: RoboTidy : 一种用于物理导航和操作的三维高斯点家庭整理基准 

**Authors**: Xiaoquan Sun, Ruijian Zhang, Kang Pang, Bingchen Miao, Yuxiang Tan, Zhen Yang, Ming Li, Jiayu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.14161)  

**Abstract**: Household tidying is an important application area, yet current benchmarks neither model user preferences nor support mobility, and they generalize poorly, making it hard to comprehensively assess integrated language-to-action capabilities. To address this, we propose RoboTidy, a unified benchmark for language-guided household tidying that supports Vision-Language-Action (VLA) and Vision-Language-Navigation (VLN) training and evaluation. RoboTidy provides 500 photorealistic 3D Gaussian Splatting (3DGS) household scenes (covering 500 objects and containers) with collisions, formulates tidying as an "Action (Object, Container)" list, and supplies 6.4k high-quality manipulation demonstration trajectories and 1.5k naviagtion trajectories to support both few-shot and large-scale training. We also deploy RoboTidy in the real world for object tidying, establishing an end-to-end benchmark for household tidying. RoboTidy offers a scalable platform and bridges a key gap in embodied AI by enabling holistic and realistic evaluation of language-guided robots. 

**Abstract (ZH)**: RoboTidy：一个支持视觉-语言-动作和视觉-语言-导航的居家整理统一基准 

---
# AsyncVLA: Asynchronous Flow Matching for Vision-Language-Action Models 

**Title (ZH)**: AsyncVLA: 异步流匹配用于视觉-语言-行动模型 

**Authors**: Yuhua Jiang, Shuang Cheng, Yan Ding, Feifei Gao, Biqing Qi  

**Link**: [PDF](https://arxiv.org/pdf/2511.14148)  

**Abstract**: Vision-language-action (VLA) models have recently emerged as a powerful paradigm for building generalist robots. However, traditional VLA models that generate actions through flow matching (FM) typically rely on rigid and uniform time schedules, i.e., synchronous FM (SFM). Without action context awareness and asynchronous self-correction, SFM becomes unstable in long-horizon tasks, where a single action error can cascade into failure. In this work, we propose asynchronous flow matching VLA (AsyncVLA), a novel framework that introduces temporal flexibility in asynchronous FM (AFM) and enables self-correction in action generation. AsyncVLA breaks from the vanilla SFM in VLA models by generating the action tokens in a non-uniform time schedule with action context awareness. Besides, our method introduces the confidence rater to extract confidence of the initially generated actions, enabling the model to selectively refine inaccurate action tokens before execution. Moreover, we propose a unified training procedure for SFM and AFM that endows a single model with both modes, improving KV-cache utilization. Extensive experiments on robotic manipulation benchmarks demonstrate that AsyncVLA is data-efficient and exhibits self-correction ability. AsyncVLA achieves state-of-the-art results across general embodied evaluations due to its asynchronous generation in AFM. Our code is available at this https URL. 

**Abstract (ZH)**: 异步视觉-语言-行动模型（AsyncVLA）：一种具有自纠正能力的异步流匹配框架 

---
# FACA: Fair and Agile Multi-Robot Collision Avoidance in Constrained Environments with Dynamic Priorities 

**Title (ZH)**: FACA：在受限环境中的动态优先级公平灵活多机器人碰撞避免 

**Authors**: Jaskirat Singh, Rohan Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2511.14024)  

**Abstract**: Multi-robot systems are increasingly being used for critical applications such as rescuing injured people, delivering food and medicines, and monitoring key areas. These applications usually involve navigating at high speeds through constrained spaces such as small gaps. Navigating such constrained spaces becomes particularly challenging when the space is crowded with multiple heterogeneous agents all of which have urgent priorities. What makes the problem even harder is that during an active response situation, roles and priorities can quickly change on a dime without informing the other agents. In order to complete missions in such environments, robots must not only be safe, but also agile, able to dodge and change course at a moment's notice. In this paper, we propose FACA, a fair and agile collision avoidance approach where robots coordinate their tasks by talking to each other via natural language (just as people do). In FACA, robots balance safety with agility via a novel artificial potential field algorithm that creates an automatic ``roundabout'' effect whenever a conflict arises. Our experiments show that FACA achieves a improvement in efficiency, completing missions more than 3.5X faster than baselines with a time reduction of over 70% while maintaining robust safety margins. 

**Abstract (ZH)**: 多机器人系统在关键应用中的敏捷避碰方法：基于自然语言的公平敏捷避碰（FACA） 

---
# BIM-Discrepancy-Driven Active Sensing for Risk-Aware UAV-UGV Navigation 

**Title (ZH)**: 基于BIM差异的主动感知风险意识UAV-UGV导航 

**Authors**: Hesam Mojtahedi, Reza Akhavian  

**Link**: [PDF](https://arxiv.org/pdf/2511.14037)  

**Abstract**: This paper presents a BIM-discrepancy-driven active sensing framework for cooperative navigation between unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs) in dynamic construction environments. Traditional navigation approaches rely on static Building Information Modeling (BIM) priors or limited onboard perception. In contrast, our framework continuously fuses real-time LiDAR data from aerial and ground robots with BIM priors to maintain an evolving 2D occupancy map. We quantify navigation safety through a unified corridor-risk metric integrating occupancy uncertainty, BIM-map discrepancy, and clearance. When risk exceeds safety thresholds, the UAV autonomously re-scans affected regions to reduce uncertainty and enable safe replanning. Validation in PX4-Gazebo simulation with Robotec GPU LiDAR demonstrates that risk-triggered re-scanning reduces mean corridor risk by 58% and map entropy by 43% compared to static BIM navigation, while maintaining clearance margins above 0.4 m. Compared to frontier-based exploration, our approach achieves similar uncertainty reduction in half the mission time. These results demonstrate that integrating BIM priors with risk-adaptive aerial sensing enables scalable, uncertainty-aware autonomy for construction robotics. 

**Abstract (ZH)**: 基于BIM数据驱动的主动传感框架：在动态施工环境中的无人机与地面机器人协作导航 

---
# Searching in Space and Time: Unified Memory-Action Loops for Open-World Object Retrieval 

**Title (ZH)**: 在空间与时间中搜索：开放世界物体检索的统一记忆-动作循环 

**Authors**: Taijing Chen, Sateesh Kumar, Junhong Xu, George Pavlakos, J oydeep Biswas, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2511.14004)  

**Abstract**: Service robots must retrieve objects in dynamic, open-world settings where requests may reference attributes ("the red mug"), spatial context ("the mug on the table"), or past states ("the mug that was here yesterday"). Existing approaches capture only parts of this problem: scene graphs capture spatial relations but ignore temporal grounding, temporal reasoning methods model dynamics but do not support embodied interaction, and dynamic scene graphs handle both but remain closed-world with fixed vocabularies. We present STAR (SpatioTemporal Active Retrieval), a framework that unifies memory queries and embodied actions within a single decision loop. STAR leverages non-parametric long-term memory and a working memory to support efficient recall, and uses a vision-language model to select either temporal or spatial actions at each step. We introduce STARBench, a benchmark of spatiotemporal object search tasks across simulated and real environments. Experiments in STARBench and on a Tiago robot show that STAR consistently outperforms scene-graph and memory-only baselines, demonstrating the benefits of treating search in time and search in space as a unified problem. 

**Abstract (ZH)**: 服务机器人必须在动态的开放世界环境中检索对象，这些请求可能引用属性（“那个红色的茶杯”）、空间上下文（“桌子上的茶杯”）或过去的状态（“昨天在这儿的茶杯”）。现有方法仅解决了该问题的一部分：场景图捕捉空间关系但忽略了时间grounding，时间推理方法建模动力学但不支持实物互动，而动态场景图同时处理这两方面，但仍然局限于固定的词汇表。我们提出了STAR（时空主动检索）框架，该框架在单一决策循环内统一了记忆查询和实物操作。STAR利用非参数长时记忆和工作记忆来支持高效的回忆，并使用视觉-语言模型在每一步选择时间或空间操作。我们引入了STARBench基准测试，涵盖模拟和真实环境中的时空对象搜索任务。STARBench和Tiago机器人上的实验表明，STAR在时空检索方面始终优于场景图和仅记忆的基线，展示了将时间中的搜索和空间中的搜索视为统一问题的好处。 

---
# $π^{*}_{0.6}$: a VLA That Learns From Experience 

**Title (ZH)**: $π^{*}_{0.6}$: 一种基于经验学习的VLA 

**Authors**: Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Adnan Esmail, Yunhao Fang, Chelsea Finn, Catherine Glossop, Thomas Godden, Ivan Goryachev, Lachy Groom, Hunter Hancock, Karol Hausman, Gashon Hussein, Brian Ichter, Szymon Jakubczak, Rowan Jen, Tim Jones, Ben Katz, Liyiming Ke, Chandra Kuchi, Marinda Lamb, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Yao Lu, Vishnu Mano, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Charvi Sharma, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, Will Stoeckle, Alex Swerdlow, James Tanner, Marcel Torne, Quan Vuong, Anna Walling, Haohuan Wang, Blake Williams, Sukwon Yoo, Lili Yu, Ury Zhilinsky, Zhiyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2511.14759)  

**Abstract**: We study how vision-language-action (VLA) models can improve through real-world deployments via reinforcement learning (RL). We present a general-purpose method, RL with Experience and Corrections via Advantage-conditioned Policies (RECAP), that provides for RL training of VLAs via advantage conditioning. Our method incorporates heterogeneous data into the self-improvement process, including demonstrations, data from on-policy collection, and expert teleoperated interventions provided during autonomous execution. RECAP starts by pre-training a generalist VLA with offline RL, which we call $\pi^{*}_{0.6}$, that can then be specialized to attain high performance on downstream tasks through on-robot data collection. We show that the $\pi^{*}_{0.6}$ model trained with the full RECAP method can fold laundry in real homes, reliably assemble boxes, and make espresso drinks using a professional espresso machine. On some of the hardest tasks, RECAP more than doubles task throughput and roughly halves the task failure rate. 

**Abstract (ZH)**: 我们研究了通过实际部署和强化学习（RL）如何提升视觉-语言-行动（VLA）模型。我们提出了一种通用方法——基于优势条件策略的经验和纠正强化学习（RECAP），该方法通过优势条件提供了VLAs的RL训练。该方法将异质数据整合进自我改进过程，包括演示、政策收集的数据以及自主执行期间的专家远程操作干预。RECAP首先通过离线RL预训练一个通用的VLA模型，记为$\pi^{*}_{0.6}$，然后通过机器人数据收集使其专门化以在下游任务中实现高性能。实验表明，使用完整RECAP方法训练的$\pi^{*}_{0.6}$模型可以在真实家庭中折叠衣物、可靠地组装盒子，并使用专业自动咖啡机制作意式咖啡。在某些最困难的任务上，RECAP将任务吞吐量提高了一倍多，将任务失败率降低了一半左右。 

---
# Active Matter as a framework for living systems-inspired Robophysics 

**Title (ZH)**: 活性物质作为一种受生物系统启发的类机器人物理学框架 

**Authors**: Giulia Janzen, Gaia Maselli, Juan F. Jimenez, Lia Garcia-Perez, D A Matoz Fernandez, Chantal Valeriani  

**Link**: [PDF](https://arxiv.org/pdf/2511.14624)  

**Abstract**: Robophysics investigates the physical principles that govern living-like robots operating in complex, realworld environments. Despite remarkable technological advances, robots continue to face fundamental efficiency limitations. At the level of individual units, locomotion remains a challenge, while at the collective level, robot swarms struggle to achieve shared purpose, coordination, communication, and cost efficiency. This perspective article examines the key challenges faced by bio-inspired robotic collectives and highlights recent research efforts that incorporate principles from active-matter physics and biology into the modeling and design of robot swarms. 

**Abstract (ZH)**: 类生物机器人物理学探究受生物启发的机器人集群在复杂现实环境中的物理原理及其面临的挑战，并强调了将活性物质物理学和生物学原则应用于机器人集群建模与设计的近期研究进展。 

---
# Enhancing End-to-End Autonomous Driving with Risk Semantic Distillaion from VLM 

**Title (ZH)**: 基于VLM的风险语义精炼增强端到端自主驾驶 

**Authors**: Jack Qin, Zhitao Wang, Yinan Zheng, Keyu Chen, Yang Zhou, Yuanxin Zhong, Siyuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.14499)  

**Abstract**: The autonomous driving (AD) system has exhibited remarkable performance in complex driving scenarios. However, generalization is still a key limitation for the current system, which refers to the ability to handle unseen scenarios or unfamiliar sensor this http URL works have explored the use of Vision-Language Models (VLMs) to address few-shot or zero-shot tasks. While promising, these methods introduce a new challenge: the emergence of a hybrid AD system, where two distinct systems are used to plan a trajectory, leading to potential inconsistencies. Alternative research directions have explored Vision-Language-Action (VLA) frameworks that generate control actions from VLM directly. However, these end-to-end solutions demonstrate prohibitive computational demands. To overcome these challenges, we introduce Risk Semantic Distillation (RSD), a novel framework that leverages VLMs to enhance the training of End-to-End (E2E) AD backbones. By providing risk attention for key objects, RSD addresses the issue of generalization. Specifically, we introduce RiskHead, a plug-in module that distills causal risk estimates from Vision-Language Models into Bird's-Eye-View (BEV) features, yielding interpretable risk-attention this http URL approach allows BEV features to learn richer and more nuanced risk attention representations, which directly enhance the model's ability to handle spatial boundaries and risky this http URL focusing on risk attention, RSD aligns better with human-like driving behavior, which is essential to navigate in complex and dynamic environments. Our experiments on the Bench2Drive benchmark demonstrate the effectiveness of RSD in managing complex and unpredictable driving conditions. Due to the enhanced BEV representations enabled by RSD, we observed a significant improvement in both perception and planning capabilities. 

**Abstract (ZH)**: 基于Vision-Language模型的自主驾驶风险语义蒸馏框架 

---
# Multi-Timescale Model Predictive Control for Slow-Fast Systems 

**Title (ZH)**: 多时间尺度模型预测控制用于慢快系统 

**Authors**: Lukas Schroth, Daniel Morton, Amon Lahr, Daniele Gammelli, Andrea Carron, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2511.14311)  

**Abstract**: Model Predictive Control (MPC) has established itself as the primary methodology for constrained control, enabling autonomy across diverse applications. While model fidelity is crucial in MPC, solving the corresponding optimization problem in real time remains challenging when combining long horizons with high-fidelity models that capture both short-term dynamics and long-term behavior. Motivated by results on the Exponential Decay of Sensitivities (EDS), which imply that, under certain conditions, the influence of modeling inaccuracies decreases exponentially along the prediction horizon, this paper proposes a multi-timescale MPC scheme for fast-sampled control. Tailored to systems with both fast and slow dynamics, the proposed approach improves computational efficiency by i) switching to a reduced model that captures only the slow, dominant dynamics and ii) exponentially increasing integration step sizes to progressively reduce model detail along the horizon. We evaluate the method on three practically motivated robotic control problems in simulation and observe speed-ups of up to an order of magnitude. 

**Abstract (ZH)**: 基于指数衰减敏感性的多时间尺度模型预测控制 

---
# A Neuro-Symbolic Framework for Reasoning under Perceptual Uncertainty: Bridging Continuous Perception and Discrete Symbolic Planning 

**Title (ZH)**: 感知不确定性下神经符号推理框架：连续感知与离散符号规划的桥梁 

**Authors**: Jiahao Wu, Shengwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14533)  

**Abstract**: Bridging continuous perceptual signals and discrete symbolic reasoning is a fundamental challenge in AI systems that must operate under uncertainty. We present a neuro-symbolic framework that explicitly models and propagates uncertainty from perception to planning, providing a principled connection between these two abstraction levels. Our approach couples a transformer-based perceptual front-end with graph neural network (GNN) relational reasoning to extract probabilistic symbolic states from visual observations, and an uncertainty-aware symbolic planner that actively gathers information when confidence is low. We demonstrate the framework's effectiveness on tabletop robotic manipulation as a concrete application: the translator processes 10,047 PyBullet-generated scenes (3--10 objects) and outputs probabilistic predicates with calibrated confidences (overall F1=0.68). When embedded in the planner, the system achieves 94\%/90\%/88\% success on Simple Stack, Deep Stack, and Clear+Stack benchmarks (90.7\% average), exceeding the strongest POMDP baseline by 10--14 points while planning within 15\,ms. A probabilistic graphical-model analysis establishes a quantitative link between calibrated uncertainty and planning convergence, providing theoretical guarantees that are validated empirically. The framework is general-purpose and can be applied to any domain requiring uncertainty-aware reasoning from perceptual input to symbolic planning. 

**Abstract (ZH)**: 连接连续感知信号与离散符号推理之间的桥梁是AI系统在不确定性条件下运作所面临的根本挑战。我们提出了一种神经符号框架，该框架明确地从感知到规划建模并传播不确定性，从而为这两个抽象层次提供了原则性的连接。该方法结合了基于变压器的感知前端和图神经网络（GNN）关系推理，从视觉观察中提取概率符号状态，并集成了一个能动态收集信息的不确定性意识符号规划器（在置信度低时）。我们在桌面机器人操作这一具体应用中证明了该框架的有效性：翻译器处理了来自PyBullet生成的10,047个场景（3-10个物体），并输出了校准过的概率谓词（总体F1值为0.68）。在嵌入规划器后，该系统在Simple Stack、Deep Stack和Clear+Stack基准测试中分别达到了94%/90%/88%的成功率（平均90.7%），在15毫秒内实现规划的同时，超出最强POMDP基线10-14个百分点。通过概率图模型分析建立了校准过的不确定性与规划收敛性之间的定量关系，并提供了在实验中得到验证的理论保证。该框架通用且可以应用于任何需要从感知输入到符号规划的不确定性意识推理的领域。 

---
# Run, Ruminate, and Regulate: A Dual-process Thinking System for Vision-and-Language Navigation 

**Title (ZH)**: 运行、反思和调节：一种视觉-语言导航的双重过程思维系统 

**Authors**: Yu Zhong, Zihao Zhang, Rui Zhang, Lingdong Huang, Haihan Gao, Shuo Wang, Da Li, Ruijian Han, Jiaming Guo, Shaohui Peng, Di Huang, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.14131)  

**Abstract**: Vision-and-Language Navigation (VLN) requires an agent to dynamically explore complex 3D environments following human instructions. Recent research underscores the potential of harnessing large language models (LLMs) for VLN, given their commonsense knowledge and general reasoning capabilities. Despite their strengths, a substantial gap in task completion performance persists between LLM-based approaches and domain experts, as LLMs inherently struggle to comprehend real-world spatial correlations precisely. Additionally, introducing LLMs is accompanied with substantial computational cost and inference latency. To address these issues, we propose a novel dual-process thinking framework dubbed R3, integrating LLMs' generalization capabilities with VLN-specific expertise in a zero-shot manner. The framework comprises three core modules: Runner, Ruminator, and Regulator. The Runner is a lightweight transformer-based expert model that ensures efficient and accurate navigation under regular circumstances. The Ruminator employs a powerful multimodal LLM as the backbone and adopts chain-of-thought (CoT) prompting to elicit structured reasoning. The Regulator monitors the navigation progress and controls the appropriate thinking mode according to three criteria, integrating Runner and Ruminator harmoniously. Experimental results illustrate that R3 significantly outperforms other state-of-the-art methods, exceeding 3.28% and 3.30% in SPL and RGSPL respectively on the REVERIE benchmark. This pronounced enhancement highlights the effectiveness of our method in handling challenging VLN tasks. 

**Abstract (ZH)**: 基于视觉-语言导航的新型双过程思考框架R3 

---
# Enhancing Agentic Autonomous Scientific Discovery with Vision-Language Model Capabilities 

**Title (ZH)**: 利用视觉语言模型能力增强自主科学发现能力 

**Authors**: Kahaan Gandhi, Boris Bolliet, Inigo Zubeldia  

**Link**: [PDF](https://arxiv.org/pdf/2511.14631)  

**Abstract**: We show that multi-agent systems guided by vision-language models (VLMs) improve end-to-end autonomous scientific discovery. By treating plots as verifiable checkpoints, a VLM-as-a-judge evaluates figures against dynamically generated domain-specific rubrics, enabling agents to correct their own errors and steer exploratory data analysis in real-time. Case studies in cosmology and astrochemistry demonstrate recovery from faulty reasoning paths and adaptation to new datasets without human intervention. On a 10-task benchmark for data-driven discovery, VLM-augmented systems achieve pass at 1 scores of 0.7-0.8, compared to 0.2-0.3 for code-only and 0.4-0.5 for code-and-text baselines, while also providing auditable reasoning traces that improve interpretability. Code available here: this https URL 

**Abstract (ZH)**: 基于视觉语言模型的多Agent系统提升端到端自主科学发现 

---
# Context-aware, Ante-hoc Explanations of Driving Behaviour 

**Title (ZH)**: 基于上下文的 Driving 行为预先解释 

**Authors**: Dominik Grundt, Ishan Saxena, Malte Petersen, Bernd Westphal, Eike Möhlmann  

**Link**: [PDF](https://arxiv.org/pdf/2511.14428)  

**Abstract**: Autonomous vehicles (AVs) must be both safe and trustworthy to gain social acceptance and become a viable option for everyday public transportation. Explanations about the system behaviour can increase safety and trust in AVs. Unfortunately, explaining the system behaviour of AI-based driving functions is particularly challenging, as decision-making processes are often opaque. The field of Explainability Engineering tackles this challenge by developing explanation models at design time. These models are designed from system design artefacts and stakeholder needs to develop correct and good explanations. To support this field, we propose an approach that enables context-aware, ante-hoc explanations of (un)expectable driving manoeuvres at runtime. The visual yet formal language Traffic Sequence Charts is used to formalise explanation contexts, as well as corresponding (un)expectable driving manoeuvres. A dedicated runtime monitoring enables context-recognition and ante-hoc presentation of explanations at runtime. In combination, we aim to support the bridging of correct and good explanations. Our method is demonstrated in a simulated overtaking. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）必须既安全又可信赖，以获得社会接受并成为日常公共交通的可行选项。通过解释系统行为可以提高AVs的安全性和信任度。然而，解释基于AI的驾驶功能的系统行为尤为具有挑战性，因为决策过程往往不透明。解释性工程通过在设计早期开发解释模型来应对这一挑战。这些模型从系统设计 artefacts 和利益相关者需求出发，旨在开发正确的良好解释。为了支持该领域，我们提出了一种方法，该方法可以在运行时启用上下文感知的预先解释，针对（不）可预期的驾驶机动。我们使用交通序列图这一直观而正式的语言来形式化解释上下文，以及相应的（不）可预期的驾驶机动。专用的运行时监控能够识别上下文并在运行时预先展示解释。结合使用，我们旨在支持正确和良好解释的桥梁构建。我们的方法在模拟超车中进行了演示。 

---
# GEN3D: Generating Domain-Free 3D Scenes from a Single Image 

**Title (ZH)**: GEN3D: 从单张图像生成无领域限制的3D场景 

**Authors**: Yuxin Zhang, Ziyu Lu, Hongbo Duan, Keyu Fan, Pengting Luo, Peiyu Zhuang, Mengyu Yang, Houde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.14291)  

**Abstract**: Despite recent advancements in neural 3D reconstruction, the dependence on dense multi-view captures restricts their broader applicability. Additionally, 3D scene generation is vital for advancing embodied AI and world models, which depend on diverse, high-quality scenes for learning and evaluation. In this work, we propose Gen3d, a novel method for generation of high-quality, wide-scope, and generic 3D scenes from a single image. After the initial point cloud is created by lifting the RGBD image, Gen3d maintains and expands its world model. The 3D scene is finalized through optimizing a Gaussian splatting representation. Extensive experiments on diverse datasets demonstrate the strong generalization capability and superior performance of our method in generating a world model and Synthesizing high-fidelity and consistent novel views. 

**Abstract (ZH)**: 尽管最近在神经3D重建方面取得了进展，但对密集多视角捕捉的依赖限制了其更广泛的应用。此外，3D场景生成对于促进具身AI和世界模型至关重要，这依赖于多样化的高质量场景来进行学习和评估。在本文中，我们提出Gen3d，一种从单张图像生成高质量、广泛适用和通用3D场景的新方法。在从RGBD图像提升初期点云后，Gen3d维护并扩展其世界模型。通过优化Gaussian splatting表示，3D场景最终完成。在多种数据集上的广泛实验表明，我们的方法在生成世界模型和合成高保真度、一致的新视角方面具有强大的泛化能力和优越性能。 

---
# Object-Centric World Models for Causality-Aware Reinforcement Learning 

**Title (ZH)**: 基于对象的世界模型在因果意识增强学习中的应用 

**Authors**: Yosuke Nishimoto, Takashi Matsubara  

**Link**: [PDF](https://arxiv.org/pdf/2511.14262)  

**Abstract**: World models have been developed to support sample-efficient deep reinforcement learning agents. However, it remains challenging for world models to accurately replicate environments that are high-dimensional, non-stationary, and composed of multiple objects with rich interactions since most world models learn holistic representations of all environmental components. By contrast, humans perceive the environment by decomposing it into discrete objects, facilitating efficient decision-making. Motivated by this insight, we propose \emph{Slot Transformer Imagination with CAusality-aware reinforcement learning} (STICA), a unified framework in which object-centric Transformers serve as the world model and causality-aware policy and value networks. STICA represents each observation as a set of object-centric tokens, together with tokens for the agent action and the resulting reward, enabling the world model to predict token-level dynamics and interactions. The policy and value networks then estimate token-level cause--effect relations and use them in the attention layers, yielding causality-guided decision-making. Experiments on object-rich benchmarks demonstrate that STICA consistently outperforms state-of-the-art agents in both sample efficiency and final performance. 

**Abstract (ZH)**: 基于因果意识的槽变换器想象与对象中心强化学习 (Slot Transformer Imagination with CAusality-aware reinforcement learning) 

---
# Real-Time Mobile Video Analytics for Pre-arrival Emergency Medical Services 

**Title (ZH)**: 基于移动视频的实时应急医疗服务预到达分析 

**Authors**: Liuyi Jin, Amran Haroon, Radu Stoleru, Pasan Gunawardena, Michael Middleton, Jeeeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.14119)  

**Abstract**: Timely and accurate pre-arrival video streaming and analytics are critical for emergency medical services (EMS) to deliver life-saving interventions. Yet, current-generation EMS infrastructure remains constrained by one-to-one video streaming and limited analytics capabilities, leaving dispatchers and EMTs to manually interpret overwhelming, often noisy or redundant information in high-stress environments. We present TeleEMS, a mobile live video analytics system that enables pre-arrival multimodal inference by fusing audio and video into a unified decision-making pipeline before EMTs arrive on scene.
TeleEMS comprises two key components: TeleEMS Client and TeleEMS Server. The TeleEMS Client runs across phones, smart glasses, and desktops to support bystanders, EMTs en route, and 911 dispatchers. The TeleEMS Server, deployed at the edge, integrates EMS-Stream, a communication backbone that enables smooth multi-party video streaming. On top of EMSStream, the server hosts three real-time analytics modules: (1) audio-to-symptom analytics via EMSLlama, a domain-specialized LLM for robust symptom extraction and normalization; (2) video-to-vital analytics using state-of-the-art rPPG methods for heart rate estimation; and (3) joint text-vital analytics via PreNet, a multimodal multitask model predicting EMS protocols, medication types, medication quantities, and procedures.
Evaluation shows that EMSLlama outperforms GPT-4o (exact-match 0.89 vs. 0.57) and that text-vital fusion improves inference robustness, enabling reliable pre-arrival intervention recommendations. TeleEMS demonstrates the potential of mobile live video analytics to transform EMS operations, bridging the gap between bystanders, dispatchers, and EMTs, and paving the way for next-generation intelligent EMS infrastructure. 

**Abstract (ZH)**: 及时准确的预到现场视频流传输与分析对于急救医疗服务（EMS）提供生命拯救干预至关重要。然而，当前的EMS基础设施仍然受限于一对一视频流传输和有限的分析能力，使得调度员和急救人员在高压环境中需要手动解释海量的、 often 噪音大或重复的信息。我们提出了TeleEMS，一个移动实时视频分析系统，通过将音频和视频融合到一个统一的决策管道中，使在急救人员到达现场之前进行预到现场的多模态推理成为可能。
TeleEMS 包括两个关键组件：TeleEMS 客户端和TeleEMS 服务器。TeleEMS 客户端在手机、智能眼镜和桌面等设备上运行，以支持旁观者、途中的急救人员和911调度员。TeleEMS 服务器部署在边缘位置，集成了EMS-Stream，一个通信骨干，支持平滑的多方视频流传输。基于EMSStream，该服务器托管三个实时分析模块：(1) 通过EMSLlama（一个专门领域的大规模语言模型）进行音频到症状的分析，用于稳健的症状抽取和规范化；(2) 使用先进的rPPG方法进行心率估计的视频到生命体征的分析；(3) 通过PreNet（一个跨模态多任务模型）进行联合文本-生命体征的分析，预测EMS规程、药物种类、药物剂量和程序。
评估表明，EMSLlama在准确匹配上优于GPT-4o（0.89比0.57），并且文本-生命体征融合提高了推理的稳健性，使得可靠的预到现场干预建议成为可能。TeleEMS展示了移动实时视频分析在变革EMS运作方面的潜力，连接旁观者、调度员和急救人员，并为下一代智能EMS基础设施铺平了道路。 

---
# Deep reinforcement learning-based spacecraft attitude control with pointing keep-out constraint 

**Title (ZH)**: 基于深度强化学习的考虑指向禁区的航天器姿态控制 

**Authors**: Juntang Yang, Mohamed Khalil Ben-Larbi  

**Link**: [PDF](https://arxiv.org/pdf/2511.13746)  

**Abstract**: This paper implements deep reinforcement learning (DRL) for spacecraft reorientation control with a single pointing keep-out zone. The Soft Actor-Critic (SAC) algorithm is adopted to handle continuous state and action space. A new state representation is designed to explicitly include a compact representation of the attitude constraint zone. The reward function is formulated to achieve the control objective while enforcing the attitude constraint. A curriculum learning approach is used for the agent training. Simulation results demonstrate the effectiveness of the proposed DRL-based method for spacecraft pointing-constrained attitude control. 

**Abstract (ZH)**: 本文采用深度强化学习（DRL）方法对具有单个指向限制区的航天器再定向控制进行研究。采用Soft Actor-Critic（SAC）算法处理连续的状态和动作空间。设计了一种新的状态表示方法，明确包含了姿态约束区的紧凑表示。通过制定奖励函数实现控制目标并遵守姿态约束。使用课程学习方法对智能体进行训练。仿真结果证明了所提出的基于DRL的方法在受指向约束的姿态控制中的有效性。 

---
