# Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning 

**Title (ZH)**: 规则到模拟：一种形式视觉规划的双多模视觉语言框架 

**Authors**: Yilun Hao, Yongchao Chen, Chuchu Fan, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03182)  

**Abstract**: Vision Language Models (VLMs) show strong potential for visual planning but struggle with precise spatial and long-horizon reasoning. In contrast, Planning Domain Definition Language (PDDL) planners excel at long-horizon formal planning, but cannot interpret visual inputs. Recent works combine these complementary advantages by enabling VLMs to turn visual planning problems into PDDL files for formal planning. However, while VLMs can generate PDDL problem files satisfactorily, they struggle to accurately generate the PDDL domain files, which describe all the planning rules. As a result, prior methods rely on human experts to predefine domain files or on constant environment access for refinement. We propose VLMFP, a Dual-VLM-guided framework that can autonomously generate both PDDL problem and domain files for formal visual planning. VLMFP introduces two VLMs to ensure reliable PDDL file generation: A SimVLM that simulates action consequences based on input rule descriptions, and a GenVLM that generates and iteratively refines PDDL files by comparing the PDDL and SimVLM execution results. VLMFP unleashes multiple levels of generalizability: The same generated PDDL domain file works for all the different instances under the same problem, and VLMs generalize to different problems with varied appearances and rules. We evaluate VLMFP with 6 grid-world domains and test its generalization to unseen instances, appearance, and game rules. On average, SimVLM accurately describes 95.5%, 82.6% of scenarios, simulates 85.5%, 87.8% of action sequence, and judges 82.4%, 85.6% goal reaching for seen and unseen appearances, respectively. With the guidance of SimVLM, VLMFP can generate PDDL files to reach 70.0%, 54.1% valid plans for unseen instances in seen and unseen appearances, respectively. Project page: this https URL. 

**Abstract (ZH)**: Vision-Language Models引导的视觉规划框架：自主生成PDDL问题和领域文件（VLMFP） 

---
# MM-Nav: Multi-View VLA Model for Robust Visual Navigation via Multi-Expert Learning 

**Title (ZH)**: MM-Nav: 多视图多专家学习的鲁棒视觉导航模型 

**Authors**: Tianyu Xu, Jiawei Chen, Jiazhao Zhang, Wenyao Zhang, Zekun Qi, Minghan Li, Zhizheng Zhang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.03142)  

**Abstract**: Visual navigation policy is widely regarded as a promising direction, as it mimics humans by using egocentric visual observations for navigation. However, optical information of visual observations is difficult to be explicitly modeled like LiDAR point clouds or depth maps, which subsequently requires intelligent models and large-scale data. To this end, we propose to leverage the intelligence of the Vision-Language-Action (VLA) model to learn diverse navigation capabilities from synthetic expert data in a teacher-student manner. Specifically, we implement the VLA model, MM-Nav, as a multi-view VLA (with 360 observations) based on pretrained large language models and visual foundation models. For large-scale navigation data, we collect expert data from three reinforcement learning (RL) experts trained with privileged depth information in three challenging tailor-made environments for different navigation capabilities: reaching, squeezing, and avoiding. We iteratively train our VLA model using data collected online from RL experts, where the training ratio is dynamically balanced based on performance on individual capabilities. Through extensive experiments in synthetic environments, we demonstrate that our model achieves strong generalization capability. Moreover, we find that our student VLA model outperforms the RL teachers, demonstrating the synergistic effect of integrating multiple capabilities. Extensive real-world experiments further confirm the effectiveness of our method. 

**Abstract (ZH)**: 视觉导航策略被视为一个有前景的方向，因为它通过第一人称视觉观察进行导航，模仿人类行为。然而，视觉观察的光学信息难以像激光雷达点云或深度图那样明确建模，这需要智能模型和大规模数据。为此，我们提出利用Vision-Language-Action（VLA）模型的智能，在教师-学生模式下从合成专家数据中学习多样的导航能力。具体而言，我们基于预训练的大语言模型和视觉基础模型实现了多视角VLA模型（具有360度观察视角）MM-Nav。对于大规模导航数据，我们从三位使用优先级深度信息训练的强化学习（RL）专家在三个针对不同导航能力定制的挑战环境中收集专家数据，分别用于接近、挤入和避开。我们从RL专家在线收集的数据中迭代训练我们的VLA模型，在不同能力上的训练比例基于性能动态平衡。通过在合成环境中的广泛实验，我们展示了我们的模型具备强大的泛化能力。此外，我们发现我们的学生VLA模型在导航能力上优于RL教师，证明了整合多种能力的协同效应。进一步的现实世界实验也证实了我们方法的有效性。 

---
# Embracing Evolution: A Call for Body-Control Co-Design in Embodied Humanoid Robot 

**Title (ZH)**: 拥抱进化：关于类人机器人身体与控制协同设计的呼吁 

**Authors**: Guiliang Liu, Bo Yue, Yi Jin Kim, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.03081)  

**Abstract**: Humanoid robots, as general-purpose physical agents, must integrate both intelligent control and adaptive morphology to operate effectively in diverse real-world environments. While recent research has focused primarily on optimizing control policies for fixed robot structures, this position paper argues for evolving both control strategies and humanoid robots' physical structure under a co-design mechanism. Inspired by biological evolution, this approach enables robots to iteratively adapt both their form and behavior to optimize performance within task-specific and resource-constrained contexts. Despite its promise, co-design in humanoid robotics remains a relatively underexplored domain, raising fundamental questions about its feasibility and necessity in achieving true embodied intelligence. To address these challenges, we propose practical co-design methodologies grounded in strategic exploration, Sim2Real transfer, and meta-policy learning. We further argue for the essential role of co-design by analyzing it from methodological, application-driven, and community-oriented perspectives. Striving to guide and inspire future studies, we present open research questions, spanning from short-term innovations to long-term goals. This work positions co-design as a cornerstone for developing the next generation of intelligent and adaptable humanoid agents. 

**Abstract (ZH)**: 人形机器人作为通用物理代理，必须整合智能控制与自适应形态，以在多样的现实环境中有效运作。虽然近期研究主要集中在优化固定机器人结构的控制策略上，本立场论文主张在共融设计机制下同时优化控制策略和人形机器人的物理结构。受到生物进化机制的启发，这种方法使得机器人能够迭代地调整其形态和行为，以在特定任务和资源受限的背景下优化性能。尽管共融设计在人形机器人领域具有巨大潜力，但仍是一个相对未被充分探索的领域，引发了关于其可行性和必要性的基本问题，以实现真正的实体智能。为了应对这些挑战，我们提出了基于战略探索、Sim2Real转移和元策略学习的实用共融设计方法，并从方法论、应用驱动和社区导向等多个视角论证共融设计的必要性。我们进一步提出了一系列开放性研究问题，涵盖短期创新和长期目标，旨在引导和启发未来的研究。本工作将共融设计定位为开发新一代智能且适应性强的人形代理的基础。 

---
# Long-Term Human Motion Prediction Using Spatio-Temporal Maps of Dynamics 

**Title (ZH)**: 基于动力学时空图的长期人体运动预测 

**Authors**: Yufei Zhu, Andrey Rudenko, Tomasz P. Kucner, Achim J. Lilienthal, Martin Magnusson  

**Link**: [PDF](https://arxiv.org/pdf/2510.03031)  

**Abstract**: Long-term human motion prediction (LHMP) is important for the safe and efficient operation of autonomous robots and vehicles in environments shared with humans. Accurate predictions are important for applications including motion planning, tracking, human-robot interaction, and safety monitoring. In this paper, we exploit Maps of Dynamics (MoDs), which encode spatial or spatio-temporal motion patterns as environment features, to achieve LHMP for horizons of up to 60 seconds. We propose an MoD-informed LHMP framework that supports various types of MoDs and includes a ranking method to output the most likely predicted trajectory, improving practical utility in robotics. Further, a time-conditioned MoD is introduced to capture motion patterns that vary across different times of day. We evaluate MoD-LHMP instantiated with three types of MoDs. Experiments on two real-world datasets show that MoD-informed method outperforms learning-based ones, with up to 50\% improvement in average displacement error, and the time-conditioned variant achieves the highest accuracy overall. Project code is available at this https URL 

**Abstract (ZH)**: 长期人类运动预测（LHMP）对于在与人类共享环境中的自主机器人和车辆的安全和高效运行至关重要。精确的预测对于运动规划、跟踪、人机交互和安全监控等应用非常重要。在本文中，我们利用动力学地图（MoDs）来实现长达60秒的LHMP，MoDs将空间或时空运动模式编码为环境特征。我们提出了一种基于MoDs的LHMP框架，支持多种类型的MoDs，并包括一种排名方法来输出最可能的预测轨迹，从而提高其实用性。此外，我们引入了时间条件下的MoDs以捕捉不同时间段变化的运动模式。我们用三种类型的MoDs实例化MoD-LHMP，并在两个真实世界数据集上的实验表明，基于MoDs的方法优于基于学习的方法，在平均位移误差上最多可提高50%，时间条件下的变体总体上达到最高精度。源代码可通过以下链接获取：this https URL。 

---
# HumanoidExo: Scalable Whole-Body Humanoid Manipulation via Wearable Exoskeleton 

**Title (ZH)**: HumanoidExo:基于可穿戴外骨骼的可扩展全身人形机器人操纵 

**Authors**: Rui Zhong, Yizhe Sun, Junjie Wen, Jinming Li, Chuang Cheng, Wei Dai, Zhiwen Zeng, Huimin Lu, Yichen Zhu, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.03022)  

**Abstract**: A significant bottleneck in humanoid policy learning is the acquisition of large-scale, diverse datasets, as collecting reliable real-world data remains both difficult and cost-prohibitive. To address this limitation, we introduce HumanoidExo, a novel system that transfers human motion to whole-body humanoid data. HumanoidExo offers a high-efficiency solution that minimizes the embodiment gap between the human demonstrator and the robot, thereby tackling the scarcity of whole-body humanoid data. By facilitating the collection of more voluminous and diverse datasets, our approach significantly enhances the performance of humanoid robots in dynamic, real-world scenarios. We evaluated our method across three challenging real-world tasks: table-top manipulation, manipulation integrated with stand-squat motions, and whole-body manipulation. Our results empirically demonstrate that HumanoidExo is a crucial addition to real-robot data, as it enables the humanoid policy to generalize to novel environments, learn complex whole-body control from only five real-robot demonstrations, and even acquire new skills (i.e., walking) solely from HumanoidExo data. 

**Abstract (ZH)**: 人形机器人政策学习中的一个重要瓶颈是获取大规模、多样化的数据集，因为可靠的真实世界数据的收集既困难又成本高昂。为了解决这一限制，我们介绍了HumanoidExo系统，该系统将人类运动转移为全身人形数据。HumanoidExo提供了一种高效解决方案，最小化了人类示范者与机器人之间的实体差距，从而解决了全身人形数据稀缺的问题。通过促进更大规模和多样化的数据集收集，我们的方法显著提高了人形机器人在动态真实世界场景中的性能。我们在三个极具挑战性的实际任务中评估了我们的方法：桌面操作、结合站立蹲下动作的操作，以及全身操作。我们的结果实证展示了HumanoidExo对于真实机器人数据的重要补充作用，它使人为策略能够泛化到新环境，仅从五个真实机器人示范中学习复杂的全身控制，甚至仅通过HumanoidExo数据就能获取新技能（如走路）。 

---
# Metrics vs Surveys: Can Quantitative Measures Replace Human Surveys in Social Robot Navigation? A Correlation Analysis 

**Title (ZH)**: 度量指标 vs 调查问卷：定量指标能否替代人类调查在社会机器人导航中的作用？相关性分析 

**Authors**: Stefano Trepella, Mauro Martini, Noé Pérez-Higueras, Andrea Ostuni, Fernando Caballero, Luis Merino, Marcello Chiaberge  

**Link**: [PDF](https://arxiv.org/pdf/2510.02941)  

**Abstract**: Social, also called human-aware, navigation is a key challenge for the integration of mobile robots into human environments. The evaluation of such systems is complex, as factors such as comfort, safety, and legibility must be considered. Human-centered assessments, typically conducted through surveys, provide reliable insights but are costly, resource-intensive, and difficult to reproduce or compare across systems. Alternatively, numerical social navigation metrics are easy to compute and facilitate comparisons, yet the community lacks consensus on a standard set of metrics.
This work explores the relationship between numerical metrics and human-centered evaluations to identify potential correlations. If specific quantitative measures align with human perceptions, they could serve as standardized evaluation tools, reducing the dependency on surveys. Our results indicate that while current metrics capture some aspects of robot navigation behavior, important subjective factors remain insufficiently represented and new metrics are necessary. 

**Abstract (ZH)**: 社会导向的导航是将移动机器人融入人类环境中的关键挑战。此类系统的评估具有复杂性，因为舒适度、安全性和可读性等因素必须考虑。以人类为中心的评估通常通过调查进行，提供了可靠的观点，但成本高、资源密集且难以跨系统复制和比较。相反，数值社会导航指标易于计算并促进比较，然而社区尚未就标准指标集达成共识。

本文探索数值指标与以人类为中心的评估之间的关系，以识别潜在的相关性。如果特定的定量措施与人的感知相一致，它们可以作为标准化评估工具，减少对调查的依赖。我们的结果显示，尽管当前的指标捕捉了部分机器人导航行为的方面，但重要的主观因素仍缺乏充分代表，因此需要新的指标。 

---
# Action Deviation-Aware Inference for Low-Latency Wireless Robots 

**Title (ZH)**: 基于动作偏差的低-latency无线机器人推断 

**Authors**: Jeyoung Park, Yeonsub Lim, Seungeun Oh, Jihong Park, Jinho Choi, Seong-Lyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.02851)  

**Abstract**: To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML, connecting distributed computational resources in edge and cloud over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: an on-device draft model locally generates drafts and a remote server-based target model verifies and corrects them, resulting lower latency. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications like robot manipulation and autonomous driving, cannot parallelize verification and correction for multiple drafts as each action depends on observation which needs to be updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference, wherein the draft model estimates an action's need for verification and correction by the target model and selectively skips communication and computation for server operations. Action deviation shows a strong correlation with action's rejection probability by the target model, enabling selective skipping. We derive the path deviation threshold that balances the transmission rate and the inference performance, and we empirically show that action deviation-aware hybrid inference reduces uplink transmission and server operation by 40%, while lowering end-to-end latency by 33.32% relative to hybrid inference without skipping and achieving task success rate up to 97.03% of that of target model only inference. 

**Abstract (ZH)**: 面向自主驾驶到工业机器人操作等时延敏感AI应用的6G分布式ML：基于行为偏差的混合推理方法 

---
# Work Zones challenge VLM Trajectory Planning: Toward Mitigation and Robust Autonomous Driving 

**Title (ZH)**: 工作区挑战基于视觉的大规模场景轨迹规划：走向缓解与稳健自动驾驶 

**Authors**: Yifan Liao, Zhen Sun, Xiaoyun Qiu, Zixiao Zhao, Wenbing Tang, Xinlei He, Xinhu Zheng, Tianwei Zhang, Xinyi Huang, Xingshuo Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.02803)  

**Abstract**: Visual Language Models (VLMs), with powerful multimodal reasoning capabilities, are gradually integrated into autonomous driving by several automobile manufacturers to enhance planning capability in challenging environments. However, the trajectory planning capability of VLMs in work zones, which often include irregular layouts, temporary traffic control, and dynamically changing geometric structures, is still unexplored. To bridge this gap, we conduct the \textit{first} systematic study of VLMs for work zone trajectory planning, revealing that mainstream VLMs fail to generate correct trajectories in $68.0%$ of cases. To better understand these failures, we first identify candidate patterns via subgraph mining and clustering analysis, and then confirm the validity of $8$ common failure patterns through human verification. Building on these findings, we propose REACT-Drive, a trajectory planning framework that integrates VLMs with Retrieval-Augmented Generation (RAG). Specifically, REACT-Drive leverages VLMs to convert prior failure cases into constraint rules and executable trajectory planning code, while RAG retrieves similar patterns in new scenarios to guide trajectory generation. Experimental results on the ROADWork dataset show that REACT-Drive yields a reduction of around $3\times$ in average displacement error relative to VLM baselines under evaluation with Qwen2.5-VL. In addition, REACT-Drive yields the lowest inference time ($0.58$s) compared with other methods such as fine-tuning ($17.90$s). We further conduct experiments using a real vehicle in 15 work zone scenarios in the physical world, demonstrating the strong practicality of REACT-Drive. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在工作区轨迹规划中的系统研究：REACT-Drive框架降低规划误差并提高推理效率 

---
# Flow with the Force Field: Learning 3D Compliant Flow Matching Policies from Force and Demonstration-Guided Simulation Data 

**Title (ZH)**: 遵循力场流动：从力和示范指导的模拟数据中学习3D顺应性流匹配策略 

**Authors**: Tianyu Li, Yihan Li, Zizhe Zhang, Nadia Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2510.02738)  

**Abstract**: While visuomotor policy has made advancements in recent years, contact-rich tasks still remain a challenge. Robotic manipulation tasks that require continuous contact demand explicit handling of compliance and force. However, most visuomotor policies ignore compliance, overlooking the importance of physical interaction with the real world, often leading to excessive contact forces or fragile behavior under uncertainty. Introducing force information into vision-based imitation learning could help improve awareness of contacts, but could also require a lot of data to perform well. One remedy for data scarcity is to generate data in simulation, yet computationally taxing processes are required to generate data good enough not to suffer from the Sim2Real gap. In this work, we introduce a framework for generating force-informed data in simulation, instantiated by a single human demonstration, and show how coupling with a compliant policy improves the performance of a visuomotor policy learned from synthetic data. We validate our approach on real-robot tasks, including non-prehensile block flipping and a bi-manual object moving, where the learned policy exhibits reliable contact maintenance and adaptation to novel conditions. Project Website: this https URL 

**Abstract (ZH)**: 尽管近年来视知觉运动策略取得了进展，但富含接触的任务仍然是一项挑战。要求连续接触的机器人操作任务需要明确处理顺应性和力。然而，大多数视知觉运动策略忽视了顺应性，忽略了与真实世界物理互动的重要性，常常导致不确定情况下的接触力过大或行为脆弱。将力信息引入基于视觉的imitation learning可以帮助提高对接触的意识，但也可能需要大量数据才能表现良好。数据稀缺的一个解决方案是在仿真中生成数据，但生成足够高质量的数据以避免Sim2Real差距需要大量的计算工作。在这项工作中，我们介绍了一种基于单个人类示范生成力导向数据的框架，并展示了与顺应性策略耦合如何提高从合成数据中学习的视知觉运动策略的性能。我们在实际机器人任务上验证了这种方法，包括非抓取积木翻转和双臂物体搬运，其中学习到的策略表现出可靠的动力接触维持和对新条件的适应。项目网站：this https URL 

---
# RSV-SLAM: Toward Real-Time Semantic Visual SLAM in Indoor Dynamic Environments 

**Title (ZH)**: RSV-SLAM: 面向室内动态环境的实时语义视觉SLAM 

**Authors**: Mobin Habibpour, Alireza Nemati, Ali Meghdari, Alireza Taheri, Shima Nazari  

**Link**: [PDF](https://arxiv.org/pdf/2510.02616)  

**Abstract**: Simultaneous Localization and Mapping (SLAM) plays an important role in many robotics fields, including social robots. Many of the available visual SLAM methods are based on the assumption of a static world and struggle in dynamic environments. In the current study, we introduce a real-time semantic RGBD SLAM approach designed specifically for dynamic environments. Our proposed system can effectively detect moving objects and maintain a static map to ensure robust camera tracking. The key innovation of our approach is the incorporation of deep learning-based semantic information into SLAM systems to mitigate the impact of dynamic objects. Additionally, we enhance the semantic segmentation process by integrating an Extended Kalman filter to identify dynamic objects that may be temporarily idle. We have also implemented a generative network to fill in the missing regions of input images belonging to dynamic objects. This highly modular framework has been implemented on the ROS platform and can achieve around 22 fps on a GTX1080. Benchmarking the developed pipeline on dynamic sequences from the TUM dataset suggests that the proposed approach delivers competitive localization error in comparison with the state-of-the-art methods, all while operating in near real-time. The source code is publicly available. 

**Abstract (ZH)**: 实时语义RGBD SLAM方法在动态环境中的应用 

---
# UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies 

**Title (ZH)**: UMI-on-Air: 体态感知指导下的体态无关视觉运动策略 

**Authors**: Harsh Gupta, Xiaofeng Guo, Huy Ha, Chuer Pan, Muqing Cao, Dongjae Lee, Sebastian Sherer, Shuran Song, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.02614)  

**Abstract**: We introduce UMI-on-Air, a framework for embodiment-aware deployment of embodiment-agnostic manipulation policies. Our approach leverages diverse, unconstrained human demonstrations collected with a handheld gripper (UMI) to train generalizable visuomotor policies. A central challenge in transferring these policies to constrained robotic embodiments-such as aerial manipulators-is the mismatch in control and robot dynamics, which often leads to out-of-distribution behaviors and poor execution. To address this, we propose Embodiment-Aware Diffusion Policy (EADP), which couples a high-level UMI policy with a low-level embodiment-specific controller at inference time. By integrating gradient feedback from the controller's tracking cost into the diffusion sampling process, our method steers trajectory generation towards dynamically feasible modes tailored to the deployment embodiment. This enables plug-and-play, embodiment-aware trajectory adaptation at test time. We validate our approach on multiple long-horizon and high-precision aerial manipulation tasks, showing improved success rates, efficiency, and robustness under disturbances compared to unguided diffusion baselines. Finally, we demonstrate deployment in previously unseen environments, using UMI demonstrations collected in the wild, highlighting a practical pathway for scaling generalizable manipulation skills across diverse-and even highly constrained-embodiments. All code, data, and checkpoints will be publicly released after acceptance. Result videos can be found at this http URL. 

**Abstract (ZH)**: UMI-on-Air：一种基于体态感知的多功能操作策略部署框架 

---
# Efficient Optimal Path Planning in Dynamic Environments Using Koopman MPC 

**Title (ZH)**: 使用库曼 MPC 在动态环境中进行高效最优路径规划 

**Authors**: Mohammad Abtahi, Navid Mojahed, Shima Nazari  

**Link**: [PDF](https://arxiv.org/pdf/2510.02584)  

**Abstract**: This paper presents a data-driven model predictive control framework for mobile robots navigating in dynamic environments, leveraging Koopman operator theory. Unlike the conventional Koopman-based approaches that focus on the linearization of system dynamics only, our work focuses on finding a global linear representation for the optimal path planning problem that includes both the nonlinear robot dynamics and collision-avoidance constraints. We deploy extended dynamic mode decomposition to identify linear and bilinear Koopman realizations from input-state data. Our open-loop analysis demonstrates that only the bilinear Koopman model can accurately capture nonlinear state-input couplings and quadratic terms essential for collision avoidance, whereas linear realizations fail to do so. We formulate a quadratic program for the robot path planning in the presence of moving obstacles in the lifted space and determine the optimal robot action in an MPC framework. Our approach is capable of finding the safe optimal action 320 times faster than a nonlinear MPC counterpart that solves the path planning problem in the original state space. Our work highlights the potential of bilinear Koopman realizations for linearization of highly nonlinear optimal control problems subject to nonlinear state and input constraints to achieve computational efficiency similar to linear problems. 

**Abstract (ZH)**: 基于Koopman算子理论的数据驱动模型预测控制框架：移动机器人在动态环境中的路径规划 

---
# A Recipe for Efficient Sim-to-Real Transfer in Manipulation with Online Imitation-Pretrained World Models 

**Title (ZH)**: 高效的 manipulatation 模拟到现实转移食谱：基于在线模仿预训练世界模型的方法 

**Authors**: Yilin Wang, Shangzhe Li, Haoyi Niu, Zhiao Huang, Weitong Zhang, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2510.02538)  

**Abstract**: We are interested in solving the problem of imitation learning with a limited amount of real-world expert data. Existing offline imitation methods often struggle with poor data coverage and severe performance degradation. We propose a solution that leverages robot simulators to achieve online imitation learning. Our sim-to-real framework is based on world models and combines online imitation pretraining with offline finetuning. By leveraging online interactions, our approach alleviates the data coverage limitations of offline methods, leading to improved robustness and reduced performance degradation during finetuning. It also enhances generalization during domain transfer. Our empirical results demonstrate its effectiveness, improving success rates by at least 31.7% in sim-to-sim transfer and 23.3% in sim-to-real transfer over existing offline imitation learning baselines. 

**Abstract (ZH)**: 我们感兴趣的是在线上使用有限的真实世界专家数据解决模拟学习问题。现有离线模拟方法往往面临数据覆盖不足和严重性能退化的挑战。我们提出了一种利用机器人模拟器实现在线模拟学习的解决方案。我们的在线转现实框架基于世界模型，并结合了在线模拟预训练和离线微调。通过利用在线交互，我们的方法缓解了离线方法的数据覆盖限制，从而在微调过程中提高了鲁棒性并减少了性能退化。它还在领域迁移中增强了泛化能力。我们的实证结果证明了其有效性，在模拟到模拟转移中将成功率至少提高31.7%，在模拟到现实转移中将成功率提高23.3%，超越了现有的离线模拟学习基线。 

---
# Improving Cooperation in Collaborative Embodied AI 

**Title (ZH)**: 提高协作机器人人工智能中的合作效率 

**Authors**: Hima Jacob Leven Suprabha, Laxmi Nag Laxminarayan Nagesh, Ajith Nair, Alvin Reuben Amal Selvaster, Ayan Khan, Raghuram Damarla, Sanju Hannah Samuel, Sreenithi Saravana Perumal, Titouan Puech, Venkataramireddy Marella, Vishal Sonar, Alessandro Suglia, Oliver Lemon  

**Link**: [PDF](https://arxiv.org/pdf/2510.03153)  

**Abstract**: The integration of Large Language Models (LLMs) into multiagent systems has opened new possibilities for collaborative reasoning and cooperation with AI agents. This paper explores different prompting methods and evaluates their effectiveness in enhancing agent collaborative behaviour and decision-making. We enhance CoELA, a framework designed for building Collaborative Embodied Agents that leverage LLMs for multi-agent communication, reasoning, and task coordination in shared virtual spaces. Through systematic experimentation, we examine different LLMs and prompt engineering strategies to identify optimised combinations that maximise collaboration performance. Furthermore, we extend our research by integrating speech capabilities, enabling seamless collaborative voice-based interactions. Our findings highlight the effectiveness of prompt optimisation in enhancing collaborative agent performance; for example, our best combination improved the efficiency of the system running with Gemma3 by 22% compared to the original CoELA system. In addition, the speech integration provides a more engaging user interface for iterative system development and demonstrations. 

**Abstract (ZH)**: 大型语言模型（LLMs）集成到多智能体系统中为与AI代理协作的推理和合作开辟了新可能性。本文探讨了不同的提示方法，并评估了它们在增强智能体协作行为和决策制定方面的有效性。我们改进了CoELA框架，该框架利用LLMs在共享虚拟空间中的多智能体通信、推理和任务协调。通过系统的实验，我们研究了不同的LLMs和提示工程策略，以确定能够最大化协作性能的最佳组合。此外，我们通过集成语音能力，实现了无缝的协作语音交互。我们的研究发现表明，提示优化在增强协作智能体性能方面非常有效；例如，我们最佳的组合将使用Gemma3运行的系统效率提高了22%，比原始CoELA系统提高了效率。此外，语音集成还为迭代系统开发和演示提供了更吸引人的用户界面。 

---
# Conceptualizing and Modeling Communication-Based Cyberattacks on Automated Vehicles 

**Title (ZH)**: 基于通信的对自动车辆的网络攻击的概念构建与建模 

**Authors**: Tianyi Li, Tianyu Liu, Yicheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02364)  

**Abstract**: Adaptive Cruise Control (ACC) is rapidly proliferating across electric vehicles (EVs) and internal combustion engine (ICE) vehicles, enhancing traffic flow while simultaneously expanding the attack surface for communication-based cyberattacks. Because the two powertrains translate control inputs into motion differently, their cyber-resilience remains unquantified. Therefore, we formalize six novel message-level attack vectors and implement them in a ring-road simulation that systematically varies the ACC market penetration rates (MPRs) and the spatial pattern of compromised vehicles. A three-tier risk taxonomy converts disturbance metrics into actionable defense priorities for practitioners. Across all simulation scenarios, EV platoons exhibit lower velocity standard deviation, reduced spacing oscillations, and faster post-attack recovery compared to ICE counterparts, revealing an inherent stability advantage. These findings clarify how controller-to-powertrain coupling influences vulnerability and offer quantitative guidance for the detection and mitigation of attacks in mixed automated traffic. 

**Abstract (ZH)**: 自适应巡航控制（ACC）在电动汽车（EV）和内燃机 vehicle（ICE）车辆中的应用迅速增长，提高了交通流量的同时，也为基于通信的网络攻击扩大了攻击面。由于两种动力系统的控制输入转化为运动的方式不同，其在网络攻击中的抵抗力尚未量化。因此，本文形式化了六种新颖的消息级攻击向量，并在根据不同 ACC 市场渗透率（MPRs）和受攻击车辆的空间模式进行系统性变化的环形道路仿真中实施这些攻击向量。三级风险分类将干扰指标转化为可操作的防御优先级，供实践者参考。在整个仿真场景中，EV 车队在速度标准偏差、间距振荡和后攻击恢复速度方面均优于 ICE 对手，表明存在固有的稳定性优势。这些发现阐明了控制器与动力系统耦合如何影响脆弱性，并为混合自动化交通中的攻击检测与缓解提供了定量指导。 

---
# Consolidating Reinforcement Learning for Multimodal Discrete Diffusion Models 

**Title (ZH)**: Consolidating Reinforcement Learning for Multimodal Discrete Diffusion Models

合并强化学习以优化多模态离散扩散模型 

**Authors**: Tianren Ma, Mu Zhang, Yibing Wang, Qixiang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.02880)  

**Abstract**: Optimizing discrete diffusion model (DDM) with rewards remains a challenge: the non-autoregressive paradigm makes importance sampling intractable and rollout complex, puzzling reinforcement learning methods such as Group Relative Policy Optimization (GRPO). In this study, we introduce MaskGRPO, the first viable approach to enable scalable multimodal reinforcement learning in discrete diffusion with effective importance sampling and modality-specific adaptations. To this end, we first clarify the theoretical foundation for DDMs, which facilitates building an importance estimator that captures valuable token fluctuation for gradient updates. We then delicately tailored the rollout method for visual sequences, which yields diverse completions and reliable optimization gradients. Upon math reasoning, coding, and visual generation benchmarks, MaskGRPO brings more stable and efficient updates, leading to stronger reasoning performance and better generation quality. This study establishes MaskGRPO as a systematic policy optimization approach and the first practical way for discretized visual diffusion. 

**Abstract (ZH)**: 优化离散扩散模型（DDM）中的奖励仍具挑战性：无自回归范式使重要性采样不可行且生成过程复杂，困扰了如Group Relative Policy Optimization (GRPO)等强化学习方法。本文引入MaskGRPO，这是第一个能够在离散扩散中实现可扩展的多模态强化学习的有效重要性采样和模态特定适应的方法。为此，我们首先阐明了DDMs的理论基础，从而有助于构建能够捕捉有价值token波动的重要性估计器，以进行梯度更新。随后，我们精心调整了视觉序列的生成方法，生成了多样化的完成和可信赖的优化梯度。在数学推理、编码和视觉生成基准测试中，MaskGRPO带来了更稳定和高效的更新，提高了推理性能并提高了生成质量。本研究确立了MaskGRPO作为系统性的策略优化方法，并且是第一个用于离散视觉扩散的实际途径。 

---
# Multimodal Large Language Model Framework for Safe and Interpretable Grid-Integrated EVs 

**Title (ZH)**: 多模态大型语言模型框架实现安全可解释的电网集成电动汽车 

**Authors**: Jean Douglas Carvalho, Hugo Kenji, Ahmad Mohammad Saber, Glaucia Melo, Max Mauro Dias Santos, Deepa Kundur  

**Link**: [PDF](https://arxiv.org/pdf/2510.02592)  

**Abstract**: The integration of electric vehicles (EVs) into smart grids presents unique opportunities to enhance both transportation systems and energy networks. However, ensuring safe and interpretable interactions between drivers, vehicles, and the surrounding environment remains a critical challenge. This paper presents a multi-modal large language model (LLM)-based framework to process multimodal sensor data - such as object detection, semantic segmentation, and vehicular telemetry - and generate natural-language alerts for drivers. The framework is validated using real-world data collected from instrumented vehicles driving on urban roads, ensuring its applicability to real-world scenarios. By combining visual perception (YOLOv8), geocoded positioning, and CAN bus telemetry, the framework bridges raw sensor data and driver comprehension, enabling safer and more informed decision-making in urban driving scenarios. Case studies using real data demonstrate the framework's effectiveness in generating context-aware alerts for critical situations, such as proximity to pedestrians, cyclists, and other vehicles. This paper highlights the potential of LLMs as assistive tools in e-mobility, benefiting both transportation systems and electric networks by enabling scalable fleet coordination, EV load forecasting, and traffic-aware energy planning.
Index Terms - Electric vehicles, visual perception, large language models, YOLOv8, semantic segmentation, CAN bus, prompt engineering, smart grid. 

**Abstract (ZH)**: 电动汽车（EVs）与智能电网的集成为提升交通系统和能源网络提供了独特机会。然而，确保驾驶员、车辆与周围环境之间的安全和可解释交互仍然是一个关键挑战。本文提出了一种基于多模态大型语言模型（LLM）的框架，用于处理多模态传感器数据（如物体检测、语义分割和车辆遥测），并生成自然语言警告供驾驶员使用。该框架通过实地数据验证，确保适用于真实场景。通过结合视觉感知（YOLOv8）、地理编码定位和CAN总线遥测，框架将原始传感器数据与驾驶员理解相连接，使在城市驾驶场景中实现更安全和明智的决策成为可能。实际数据案例研究表明，该框架在行人、骑车人和其他车辆附近等关键情况下生成上下文相关警告的有效性。本文强调了LLM作为辅助工具在电动出行领域的潜力，通过实现可扩展的车队协调、电动汽车负载预测和交通感知能源规划，同时为交通系统和电力网络带来好处。关键词 - 电动汽车，视觉感知，大型语言模型，YOLOv8，语义分割，CAN总线，提示工程，智能电网。 

---
# A Benchmark Study of Deep Reinforcement Learning Algorithms for the Container Stowage Planning Problem 

**Title (ZH)**: 深 reinforcement 学习算法在集装箱装载计划问题上的基准研究 

**Authors**: Yunqi Huang, Nishith Chennakeshava, Alexis Carras, Vladislav Neverov, Wei Liu, Aske Plaat, Yingjie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02589)  

**Abstract**: Container stowage planning (CSPP) is a critical component of maritime transportation and terminal operations, directly affecting supply chain efficiency. Owing to its complexity, CSPP has traditionally relied on human expertise. While reinforcement learning (RL) has recently been applied to CSPP, systematic benchmark comparisons across different algorithms remain limited. To address this gap, we develop a Gym environment that captures the fundamental features of CSPP and extend it to include crane scheduling in both multi-agent and single-agent formulations. Within this framework, we evaluate five RL algorithms: DQN, QR-DQN, A2C, PPO, and TRPO under multiple scenarios of varying complexity. The results reveal distinct performance gaps with increasing complexity, underscoring the importance of algorithm choice and problem formulation for CSPP. Overall, this paper benchmarks multiple RL methods for CSPP while providing a reusable Gym environment with crane scheduling, thus offering a foundation for future research and practical deployment in maritime logistics. 

**Abstract (ZH)**: 基于强化学习的集装箱堆存规划：性能基准比较与环境构建 

---
# A Unified Deep Reinforcement Learning Approach for Close Enough Traveling Salesman Problem 

**Title (ZH)**: 近似旅行商问题的统一深度强化学习方法 

**Authors**: Mingfeng Fan, Jiaqi Cheng, Yaoxin Wu, Yifeng Zhang, Yibin Yang, Guohua Wu, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2510.03065)  

**Abstract**: In recent years, deep reinforcement learning (DRL) has gained traction for solving the NP-hard traveling salesman problem (TSP). However, limited attention has been given to the close-enough TSP (CETSP), primarily due to the challenge introduced by its neighborhood-based visitation criterion, wherein a node is considered visited if the agent enters a compact neighborhood around it. In this work, we formulate a Markov decision process (MDP) for CETSP using a discretization scheme and propose a novel unified dual-decoder DRL (UD3RL) framework that separates decision-making into node selection and waypoint determination. Specifically, an adapted encoder is employed for effective feature extraction, followed by a node-decoder and a loc-decoder to handle the two sub-tasks, respectively. A k-nearest neighbors subgraph interaction strategy is further introduced to enhance spatial reasoning during location decoding. Furthermore, we customize the REINFORCE algorithm to train UD3RL as a unified model capable of generalizing across different problem sizes and varying neighborhood radius types (i.e., constant and random radii). Experimental results show that UD3RL outperforms conventional methods in both solution quality and runtime, while exhibiting strong generalization across problem scales, spatial distributions, and radius ranges, as well as robustness to dynamic environments. 

**Abstract (ZH)**: 近年来，深度强化学习（DRL）在解决NP难旅行商问题（TSP）方面取得了进展。然而，由于其基于邻域的访问准则所带来的挑战，对最近足够旅行商问题（CETSP）的关注相对较少。其中，如果代理进入节点周围的紧凑邻域，则认为节点被访问。在本文中，我们使用离散化方案为CETSP建模马尔可夫决策过程（MDP），并提出了一种新的统一双解码器DRL（UD3RL）框架，将决策过程分为节点选择和路径点确定两个部分。具体而言，采用改编后的编码器进行有效的特征提取，随后通过节点解码器和位置解码器分别处理这两个子任务，并引入k近邻子图交互策略以增强空间推理能力。此外，我们对REINFORCE算法进行定制，以训练UD3RL作为一个统一模型，能够跨不同问题规模和不同邻域半径类型（即固定半径和随机半径）进行泛化。实验结果表明，在解决方案质量和运行时间方面，UD3RL优于传统方法，并且在问题规模、空间分布和半径范围方面具有较强的泛化能力，同时在动态环境中表现出较强的鲁棒性。 

---
# Comparative Analysis of Parameterized Action Actor-Critic Reinforcement Learning Algorithms for Web Search Match Plan Generation 

**Title (ZH)**: 参数化动作actor-critic强化学习算法的网络搜索匹配计划生成比较分析 

**Authors**: Ubayd Bapoo, Clement N Nyirenda  

**Link**: [PDF](https://arxiv.org/pdf/2510.03064)  

**Abstract**: This study evaluates the performance of Soft Actor Critic (SAC), Greedy Actor Critic (GAC), and Truncated Quantile Critics (TQC) in high-dimensional decision-making tasks using fully observable environments. The focus is on parametrized action (PA) spaces, eliminating the need for recurrent networks, with benchmarks Platform-v0 and Goal-v0 testing discrete actions linked to continuous action-parameter spaces. Hyperparameter optimization was performed with Microsoft NNI, ensuring reproducibility by modifying the codebase for GAC and TQC. Results show that Parameterized Action Greedy Actor-Critic (PAGAC) outperformed other algorithms, achieving the fastest training times and highest returns across benchmarks, completing 5,000 episodes in 41:24 for the Platform game and 24:04 for the Robot Soccer Goal game. Its speed and stability provide clear advantages in complex action spaces. Compared to PASAC and PATQC, PAGAC demonstrated superior efficiency and reliability, making it ideal for tasks requiring rapid convergence and robust performance. Future work could explore hybrid strategies combining entropy-regularization with truncation-based methods to enhance stability and expand investigations into generalizability. 

**Abstract (ZH)**: 本研究评估了在完全可观测环境中，Soft Actor Critic (SAC)、Greedy Actor Critic (GAC) 和 Truncated Quantile Critics (TQC) 在高维决策任务中的性能，重点关注参数化动作（PA）空间，消除了循环网络的需求，并使用Platform-v0和Goal-v0基准测试离散动作与连续动作参数空间的联系。通过Microsoft NNI进行了超参数优化，通过对GAC和TQC代码的修改确保可重复性。结果表明，Parameterized Action Greedy Actor-Critic (PAGAC) 在各基准测试中表现最佳，训练速度最快，回报最高，在Platform游戏中完成5000个回合耗时41:24，在Robot Soccer Goal游戏中耗时24:04。其速度和稳定性在复杂动作空间中提供了明显优势。与PASAC和PATQC相比，PAGAC体现了更高的效率和可靠性，使其适用于需要快速收敛和稳健性能的任务。未来工作可以探索结合熵正则化与截断方法的混合策略，以提高稳定性和扩展泛化性研究。 

---
# RAMAC: Multimodal Risk-Aware Offline Reinforcement Learning and the Role of Behavior Regularization 

**Title (ZH)**: RAMAC：多模态风险意识离线强化学习及行为正则化的作用 

**Authors**: Kai Fukazawa, Kunal Mundada, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2510.02695)  

**Abstract**: In safety-critical domains where online data collection is infeasible, offline reinforcement learning (RL) offers an attractive alternative but only if policies deliver high returns without incurring catastrophic lower-tail risk. Prior work on risk-averse offline RL achieves safety at the cost of value conservatism and restricted policy classes, whereas expressive policies are only used in risk-neutral settings. Here, we address this gap by introducing the \textbf{Risk-Aware Multimodal Actor-Critic (RAMAC)} framework, which couples an \emph{expressive generative actor} with a distributional critic. The RAMAC differentiates composite objective combining distributional risk and BC loss through the generative path, achieving risk-sensitive learning in complex multimodal scenarios. We instantiate RAMAC with diffusion and flow-matching actors and observe consistent gains in $\mathrm{CVaR}_{0.1}$ while maintaining strong returns on most Stochastic-D4RL tasks. Code: this https URL 

**Abstract (ZH)**: 在在线数据收集不可行的安全关键领域中，离线强化学习（RL）提供了一种有吸引力的替代方案，前提是策略能够实现高回报且不引发灾难性的下尾风险。先前关于风险厌恶的离线RL研究在确保安全性的同时牺牲了价值保守性和限制性的策略类别，而表现性强的策略仅在无风险偏好设置下使用。在这里，我们通过引入Risk-Aware Multimodal Actor-Critic (RAMAC)框架来弥补这一差距，该框架将一个表达性强的生成actor与分布性critic相结合。RAMAC通过生成路径区分结合了分布性风险和BC损失的复合目标，从而在复杂多模态场景中实现敏感风险学习。我们使用扩散和流匹配actor实例化RAMAC，在大多数Stochastic-D4RL任务中保持强劲回报的同时观察到$\mathrm{CVaR}_{0.1}$的一致改进。代码：this https URL 

---
# Oracle-RLAIF: An Improved Fine-Tuning Framework for Multi-modal Video Models through Reinforcement Learning from Ranking Feedback 

**Title (ZH)**: Oracle-RLAIF：通过排名反馈强化学习改进的多模态视频模型 fine-tuning 框架 

**Authors**: Derek Shi, Ruben Glatt, Christine Klymko, Shubham Mohole, Hongjun Choi, Shashank Kushwaha, Sam Sakla, Felipe Leno da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2510.02561)  

**Abstract**: Recent advances in large video-language models (VLMs) rely on extensive fine-tuning techniques that strengthen alignment between textual and visual comprehension. Leading pipelines typically pair supervised fine-tuning (SFT) with reinforcement learning from preference data to enhance video comprehension. However, as VLMs scale in parameter size, so does the cost of gathering enough human feedback. To make fine-tuning more cost-effective, recent frameworks explore reinforcement learning with AI feedback (RLAIF), which replace human preference with AI as a judge. Current RLAIF frameworks rely on a specialized reward model trained with video narratives to create calibrated scalar rewards-- an expensive and restrictive pipeline. We propose Oracle-RLAIF, a novel framework that replaces the trained reward model with a more general Oracle ranker which acts as a drop-in model ranking candidate model responses rather than scoring them. Alongside Oracle-RLAIF, we introduce $GRPO_{rank}$, a novel rank-based loss function based on Group Relative Policy Optimization (GRPO) that directly optimizes ordinal feedback with rank-aware advantages. Empirically, we demonstrate that Oracle-RLAIF consistently outperforms leading VLMs using existing fine-tuning methods when evaluated across various video comprehension benchmarks. Oracle-RLAIF paves the path to creating flexible and data-efficient frameworks for aligning large multi-modal video models with reinforcement learning from rank rather than score. 

**Abstract (ZH)**: Recent Advances in Large Video-Language Models via Oracle-RLAIF 

---
# CLARITY: Clinical Assistant for Routing, Inference, and Triage 

**Title (ZH)**: CLARITY: 临床辅助系统用于路由、推理和分类 

**Authors**: Vladimir Shaposhnikov, Aleksandr Nesterov, Ilia Kopanichuk, Ivan Bakulin, Egor Zhelvakov, Ruslan Abramov, Ekaterina Tsapieva, Dmitry V. Dylov, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2510.02463)  

**Abstract**: We present CLARITY (Clinical Assistant for Routing, Inference, and Triage), an AI-driven platform designed to facilitate patient-to-specialist routing, clinical consultations, and severity assessment of patients' conditions. Its hybrid architecture combines a Finite State Machine (FSM) for structured dialogue flows with collaborative agents that employ Large Language Model (LLM) to analyze symptoms and prioritize referrals to appropriate specialists. Built on a modular microservices framework, CLARITY ensures safe, efficient, and robust performance, flexible and readily scalable to meet the demands of existing workflows and IT solutions in healthcare.
We report integration of our clinical assistant into a large-scale nation-wide inter-hospital IT platform, with over 55,000 content-rich user dialogues completed within the two months of deployment, 2,500 of which were expert-annotated for a consequent validation. The validation results show that CLARITY surpasses human-level performance in terms of the first-attempt routing precision, naturally requiring up to 3 times shorter duration of the consultation than with a human. 

**Abstract (ZH)**: 临床助手CLARITY：一种基于AI的平台，用于患者专科路由、临床咨询和病情严重程度评估 

---
# Glaucoma Detection and Structured OCT Report Generation via a Fine-tuned Multimodal Large Language Model 

**Title (ZH)**: 基于微调多模态大语言模型的青光眼检测与结构化OCT报告生成 

**Authors**: Jalil Jalili, Yashraj Gavhane, Evan Walker, Anna Heinke, Christopher Bowd, Akram Belghith, Massimo A. Fazio, Christopher A. Girkin, C. Gustavo De Moraes, Jeffrey M. Liebmann, Sally L. Baxter, Robert N. Weinreb, Linda M. Zangwill, Mark Christopher  

**Link**: [PDF](https://arxiv.org/pdf/2510.02403)  

**Abstract**: Objective: To develop an explainable multimodal large language model (MM-LLM) that (1) screens optic nerve head (ONH) OCT circle scans for quality and (2) generates structured clinical reports that include glaucoma diagnosis and sector-wise retinal nerve fiber layer (RNFL) thinning assessments. Design: Retrospective cohort study of 1,310 subjects contributing 43,849 Spectralis ONH OCT circle scans (1,331 glaucomatous and 867 healthy eyes) from the DIGS and ADAGES cohorts. Methods: A MM-LLM (Llama 3.2 Vision-Instruct model) was fine-tuned to generate clinical descriptions of OCT imaging data. Training data included paired OCT images and automatically generated, structured clinical reports that described global and sectoral RNFL thinning. Poor-quality scans were labeled as unusable and paired with a fixed refusal statement. The model was evaluated on a held-out test set for three tasks: quality assessment, glaucoma detection, and RNFL thinning classification across seven anatomical sectors. Evaluation metrics included accuracy, sensitivity, specificity, precision, and F1-score. Model description quality was also evaluated using standard text evaluation metrics. Results: The model achieved 0.90 accuracy and 0.98 specificity for quality triage. For glaucoma detection, accuracy was 0.86 (sensitivity 0.91, specificity 0.73, F1-score 0.91). RNFL thinning prediction accuracy ranged from 0.83 to 0.94, with highest performance in global and temporal sectors. Text generation scores showed strong alignment with reference reports (BLEU: 0.82; ROUGE-1: 0.94; ROUGE-2: 0.87; ROUGE-L: 0.92; BERTScore-F1: 0.99). Conclusions: The fine-tuned MM-LLM generated accurate clinical descriptions based on OCT imaging. The model achieved high accuracy in identifying image quality issues and detecting glaucoma. The model also provided sectoral descriptions of RNFL thinning to help support clinical OCT evaluation. 

**Abstract (ZH)**: 目标：开发一个可解释的多模态大型语言模型（MM-LLM），该模型能够（1）筛选Optic Nerve Head（ONH）OCT圆扫描的质量，并（2）生成包含青光眼诊断和视网膜神经纤维层（RNFL）局部门类性变薄评估的结构化临床报告。设计：一项包含1,310名参与者的回顾性队列研究，贡献了43,849张Spectralis ONH OCT圆扫描图像（其中1,331例为青光眼患者，867例为健康眼），来自DIGS和ADAGES队列。方法：一个MM-LLM（Llama 3.2 Vision-Instruct模型）被微调以生成OCT影像数据的临床描述。训练数据包括配对的OCT图像和自动生成的结构化临床报告，描述了全局和局部门类性RNFL变薄。质量较差的扫描被标记为不可用，并附上固定拒绝声明。模型在保留的测试集上进行了三项任务的评估：图像质量评估、青光眼检测和RNFL变薄分类（跨七个解剖部位）。评估指标包括准确性、灵敏度、特异性、精确度和F1分数。模型描述质量还使用标准文本评估指标进行了评价。结果：模型在图像筛选中的准确率达到0.90，特异性达到0.98。在青光眼检测中，准确率为0.86（灵敏度为0.91，特异性为0.73，F1分数为0.91）。RNFL变薄预测的准确性范围从0.83到0.94，其中全球和颞部区域表现最佳。文本生成得分与参考报告高度一致（BLEU: 0.82；ROUGE-1: 0.94；ROUGE-2: 0.87；ROUGE-L: 0.92；BERTScore-F1: 0.99）。结论：微调后的MM-LLM根据OCT影像生成了准确的临床描述。该模型在识别图像质量问题和检测青光眼方面达到了高准确性。模型还提供了RNFL局部门类性变薄的描述，以帮助支持临床OCT评估。 

---
# Measuring Physical-World Privacy Awareness of Large Language Models: An Evaluation Benchmark 

**Title (ZH)**: 大型语言模型对物理世界隐私意识的测量：一个评估基准 

**Authors**: Xinjie Shen, Mufei Li, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.02356)  

**Abstract**: The deployment of Large Language Models (LLMs) in embodied agents creates an urgent need to measure their privacy awareness in the physical world. Existing evaluation methods, however, are confined to natural language based scenarios. To bridge this gap, we introduce EAPrivacy, a comprehensive evaluation benchmark designed to quantify the physical-world privacy awareness of LLM-powered agents. EAPrivacy utilizes procedurally generated scenarios across four tiers to test an agent's ability to handle sensitive objects, adapt to changing environments, balance task execution with privacy constraints, and resolve conflicts with social norms. Our measurements reveal a critical deficit in current models. The top-performing model, Gemini 2.5 Pro, achieved only 59\% accuracy in scenarios involving changing physical environments. Furthermore, when a task was accompanied by a privacy request, models prioritized completion over the constraint in up to 86\% of cases. In high-stakes situations pitting privacy against critical social norms, leading models like GPT-4o and Claude-3.5-haiku disregarded the social norm over 15\% of the time. These findings, demonstrated by our benchmark, underscore a fundamental misalignment in LLMs regarding physically grounded privacy and establish the need for more robust, physically-aware alignment. 

**Abstract (ZH)**: 大规模语言模型在具身代理中的部署迫切需要评估其在物理世界中的隐私意识。现有评估方法仅限于基于自然语言的场景。为弥补这一差距，我们提出了一种名为EAPrivacy的全面评估基准，用于量化由大规模语言模型驱动的代理在物理世界中的隐私意识。EAPrivacy利用四级程序生成的场景测试代理处理敏感对象、适应变化环境、平衡任务执行与隐私约束以及解决与社会规范冲突的能力。我们的测量结果揭示了当前模型中存在的关键缺陷。在涉及变化物理环境的场景中，性能最佳的模型Gemini 2.5 Pro的准确率仅为59%。此外，当任务伴随有隐私请求时，模型在多达86%的情况下优先完成任务而非遵守约束。在涉及隐私与关键社会规范的竞争性情境中，领先模型如GPT-4o和Claude-3.5-haiku有时会超过15%的情况下忽视社会规范。这些由我们的基准测试得出的发现强调了大规模语言模型在物理接地隐私方面存在根本性的不匹配，并突显了需要更 robust、物理意识更强的对齐。 

---
# AMANDA: Agentic Medical Knowledge Augmentation for Data-Efficient Medical Visual Question Answering 

**Title (ZH)**: AMANDA: 基于代理的医学知识增强在数据高效医学视觉问答中的应用 

**Authors**: Ziqing Wang, Chengsheng Mao, Xiaole Wen, Yuan Luo, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.02328)  

**Abstract**: Medical Multimodal Large Language Models (Med-MLLMs) have shown great promise in medical visual question answering (Med-VQA). However, when deployed in low-resource settings where abundant labeled data are unavailable, existing Med-MLLMs commonly fail due to their medical reasoning capability bottlenecks: (i) the intrinsic reasoning bottleneck that ignores the details from the medical image; (ii) the extrinsic reasoning bottleneck that fails to incorporate specialized medical knowledge. To address those limitations, we propose AMANDA, a training-free agentic framework that performs medical knowledge augmentation via LLM agents. Specifically, our intrinsic medical knowledge augmentation focuses on coarse-to-fine question decomposition for comprehensive diagnosis, while extrinsic medical knowledge augmentation grounds the reasoning process via biomedical knowledge graph retrieval. Extensive experiments across eight Med-VQA benchmarks demonstrate substantial improvements in both zero-shot and few-shot Med-VQA settings. The code is available at this https URL. 

**Abstract (ZH)**: 医疗多模态大型语言模型（Med-MLLMs）在医疗视觉问答（Med-VQA）方面展现了巨大的潜力。然而，在缺乏充足标注数据的低资源环境中部署时，现有Med-MLLMs常因医疗推理能力瓶颈而失效：（i）内在推理瓶颈，忽视了医学图像的细节；（ii）外在推理瓶颈，未能结合专业医学知识。为解决这些限制，我们提出了一种无需训练的代理框架AMANDA，通过LLM代理进行医疗知识增强。具体来说，我们的内在医疗知识增强侧重于从粗到细的问题分解以实现全面诊断，而外在医疗知识增强则通过生物医学知识图谱检索来指导推理过程。在八个Med-VQA基准上的广泛实验表明，在零样本和少量样本Med-VQA设置中均取得了显著改进。代码可在以下链接获取：this https URL。 

---
# Agentic-AI Healthcare: Multilingual, Privacy-First Framework with MCP Agents 

**Title (ZH)**: 代理-AI医疗: 多语言隐私优先框架与MCP代理 

**Authors**: Mohammed A. Shehab  

**Link**: [PDF](https://arxiv.org/pdf/2510.02325)  

**Abstract**: This paper introduces Agentic-AI Healthcare, a privacy-aware, multilingual, and explainable research prototype developed as a single-investigator project. The system leverages the emerging Model Context Protocol (MCP) to orchestrate multiple intelligent agents for patient interaction, including symptom checking, medication suggestions, and appointment scheduling. The platform integrates a dedicated Privacy and Compliance Layer that applies role-based access control (RBAC), AES-GCM field-level encryption, and tamper-evident audit logging, aligning with major healthcare data protection standards such as HIPAA (US), PIPEDA (Canada), and PHIPA (Ontario). Example use cases demonstrate multilingual patient-doctor interaction (English, French, Arabic) and transparent diagnostic reasoning powered by large language models. As an applied AI contribution, this work highlights the feasibility of combining agentic orchestration, multilingual accessibility, and compliance-aware architecture in healthcare applications. This platform is presented as a research prototype and is not a certified medical device. 

**Abstract (ZH)**: Agentic-AI医疗保健：一种隐私意识、多语言且可解释的研究原型 

---
