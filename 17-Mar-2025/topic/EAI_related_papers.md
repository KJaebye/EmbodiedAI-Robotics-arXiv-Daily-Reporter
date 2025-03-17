# Centaur: Robust End-to-End Autonomous Driving with Test-Time Training 

**Title (ZH)**: Centaur: 具有测试时训练的稳健端到端自主驾驶 

**Authors**: Chonghao Sima, Kashyap Chitta, Zhiding Yu, Shiyi Lan, Ping Luo, Andreas Geiger, Hongyang Li, Jose M. Alvarez  

**Link**: [PDF](https://arxiv.org/pdf/2503.11650)  

**Abstract**: How can we rely on an end-to-end autonomous vehicle's complex decision-making system during deployment? One common solution is to have a ``fallback layer'' that checks the planned trajectory for rule violations and replaces it with a pre-defined safe action if necessary. Another approach involves adjusting the planner's decisions to minimize a pre-defined ``cost function'' using additional system predictions such as road layouts and detected obstacles. However, these pre-programmed rules or cost functions cannot learn and improve with new training data, often resulting in overly conservative behaviors. In this work, we propose Centaur (Cluster Entropy for Test-time trAining using Uncertainty) which updates a planner's behavior via test-time training, without relying on hand-engineered rules or cost functions. Instead, we measure and minimize the uncertainty in the planner's decisions. For this, we develop a novel uncertainty measure, called Cluster Entropy, which is simple, interpretable, and compatible with state-of-the-art planning algorithms. Using data collected at prior test-time time-steps, we perform an update to the model's parameters using a gradient that minimizes the Cluster Entropy. With only this sole gradient update prior to inference, Centaur exhibits significant improvements, ranking first on the navtest leaderboard with notable gains in safety-critical metrics such as time to collision. To provide detailed insights on a per-scenario basis, we also introduce navsafe, a challenging new benchmark, which highlights previously undiscovered failure modes of driving models. 

**Abstract (ZH)**: 如何在部署时依赖端到端自动驾驶车辆的复杂决策系统？一种常见的解决方案是在系统中设置一个“备选层”，该层检查计划轨迹是否存在规则违规行为，并在必要时用预定义的安全行动替换。另一种方法是通过最小化预定义的“成本函数”，调整规划器的决策，以便利用额外的系统预测，例如道路布局和检测到的障碍物。然而，这些预编程的规则或成本函数无法从新的训练数据中学习和改进，通常会导致过于保守的行为。在此项工作中，我们提出了一种名为Centaur（Cluster Entropy for Test-time Training using Uncertainty）的方法，该方法通过测试时训练更新规划器的行为，不依赖于人为设计的规则或成本函数，而是通过测量和最小化规划器决策中的不确定性来实现。为此，我们开发了一种新的不确定性度量，称为Cluster Entropy，该度量简单可解释，并且与最先进的规划算法兼容。利用先前测试时刻收集的数据，我们使用最小化Cluster Entropy的梯度更新模型参数。在推理前仅进行这一梯度更新，Centaur 在导航测试排行榜上表现出显著改进，特别是在碰撞时间等关键安全指标方面取得显著进步。为了提供针对每个场景的详细见解，我们还引入了navsafe，这一新的具有挑战性的基准测试，揭示了驾驶模型中以前未被发现的失败模式。 

---
# Adversarial Data Collection: Human-Collaborative Perturbations for Efficient and Robust Robotic Imitation Learning 

**Title (ZH)**: 对抗数据采集：高效稳健的机器人模仿学习中的人机协作扰动 

**Authors**: Siyuan Huang, Yue Liao, Siyuan Feng, Shu Jiang, Si Liu, Hongsheng Li, Maoqing Yao, Guanghui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.11646)  

**Abstract**: The pursuit of data efficiency, where quality outweighs quantity, has emerged as a cornerstone in robotic manipulation, especially given the high costs associated with real-world data collection. We propose that maximizing the informational density of individual demonstrations can dramatically reduce reliance on large-scale datasets while improving task performance. To this end, we introduce Adversarial Data Collection, a Human-in-the-Loop (HiL) framework that redefines robotic data acquisition through real-time, bidirectional human-environment interactions. Unlike conventional pipelines that passively record static demonstrations, ADC adopts a collaborative perturbation paradigm: during a single episode, an adversarial operator dynamically alters object states, environmental conditions, and linguistic commands, while the tele-operator adaptively adjusts actions to overcome these evolving challenges. This process compresses diverse failure-recovery behaviors, compositional task variations, and environmental perturbations into minimal demonstrations. Our experiments demonstrate that ADC-trained models achieve superior compositional generalization to unseen task instructions, enhanced robustness to perceptual perturbations, and emergent error recovery capabilities. Strikingly, models trained with merely 20% of the demonstration volume collected through ADC significantly outperform traditional approaches using full datasets. These advances bridge the gap between data-centric learning paradigms and practical robotic deployment, demonstrating that strategic data acquisition, not merely post-hoc processing, is critical for scalable, real-world robot learning. Additionally, we are curating a large-scale ADC-Robotics dataset comprising real-world manipulation tasks with adversarial perturbations. This benchmark will be open-sourced to facilitate advancements in robotic imitation learning. 

**Abstract (ZH)**: 追求数据效率，重视质量而非数量，已成为机器人操作领域的基石，特别是在面对高昂的实际数据采集成本时。我们提出通过最大化单个演示的数据信息密度，可以显著减少对大规模数据集的依赖，同时提高任务性能。为此，我们引入了对抗性数据采集（Adversarial Data Collection，ADC）这一基于人工在环（Human-in-the-Loop，HiL）框架，重新定义了通过实时双向的人机环境交互来采集机器人数据的方式。与传统的被动记录静态演示的管道不同，ADC采用了一种协作扰动范式：在单个episode中，对抗操作者动态改变物体状态、环境条件和语言指令，而远程操作者则适应性调整行动以应对这些不断演变的挑战。这一过程将多样化的失败恢复行为、组合任务变化和环境扰动压缩到了最小的演示中。我们的实验表明，使用ADC训练的模型在面对未见过的任务指令时表现出更优的组合泛化能力，增强了对感知扰动的鲁棒性，并具备了新兴的错误恢复能力。令人瞩目的是，仅使用通过ADC采集的演示数据量的20%训练的模型，在任务表现上显著优于使用完整数据集的传统方法。这些进展缩小了数据为中心的学习范式与实际机器人部署之间的差距，证明了战略性数据采集而非仅仅后期处理对于可扩展的现实世界机器人学习至关重要。此外，我们正在编纂一个大规模的ADC-Robotics数据集，包含带有对抗扰动的实际操作任务，该基准将开源以促进机器人模仿学习的进步。 

---
# Dynamic Obstacle Avoidance with Bounded Rationality Adversarial Reinforcement Learning 

**Title (ZH)**: 具有限定理性对手增强学习的动态障碍避让 

**Authors**: Jose-Luis Holgado-Alvarez, Aryaman Reddi, Carlo D'Eramo  

**Link**: [PDF](https://arxiv.org/pdf/2503.11467)  

**Abstract**: Reinforcement Learning (RL) has proven largely effective in obtaining stable locomotion gaits for legged robots. However, designing control algorithms which can robustly navigate unseen environments with obstacles remains an ongoing problem within quadruped locomotion. To tackle this, it is convenient to solve navigation tasks by means of a hierarchical approach with a low-level locomotion policy and a high-level navigation policy. Crucially, the high-level policy needs to be robust to dynamic obstacles along the path of the agent. In this work, we propose a novel way to endow navigation policies with robustness by a training process that models obstacles as adversarial agents, following the adversarial RL paradigm. Importantly, to improve the reliability of the training process, we bound the rationality of the adversarial agent resorting to quantal response equilibria, and place a curriculum over its rationality. We called this method Hierarchical policies via Quantal response Adversarial Reinforcement Learning (Hi-QARL). We demonstrate the robustness of our method by benchmarking it in unseen randomized mazes with multiple obstacles. To prove its applicability in real scenarios, our method is applied on a Unitree GO1 robot in simulation. 

**Abstract (ZH)**: 基于定量反应对抗强化学习的分层导航策略（Hierarchical Policies via Quantal Response Adversarial Reinforcement Learning (Hi-QARL)） 

---
# Prof. Robot: Differentiable Robot Rendering Without Static and Self-Collisions 

**Title (ZH)**: 教授机器人：无静止和自碰撞可微机器人渲染 

**Authors**: Quanyuan Ruan, Jiabao Lei, Wenhao Yuan, Yanglin Zhang, Dekun Lu, Guiliang Liu, Kui Jia  

**Link**: [PDF](https://arxiv.org/pdf/2503.11269)  

**Abstract**: Differentiable rendering has gained significant attention in the field of robotics, with differentiable robot rendering emerging as an effective paradigm for learning robotic actions from image-space supervision. However, the lack of physical world perception in this approach may lead to potential collisions during action optimization. In this work, we introduce a novel improvement on previous efforts by incorporating physical awareness of collisions through the learning of a neural robotic collision classifier. This enables the optimization of actions that avoid collisions with static, non-interactable environments as well as the robot itself. To facilitate effective gradient optimization with the classifier, we identify the underlying issue and propose leveraging Eikonal regularization to ensure consistent gradients for optimization. Our solution can be seamlessly integrated into existing differentiable robot rendering frameworks, utilizing gradients for optimization and providing a foundation for future applications of differentiable rendering in robotics with improved reliability of interactions with the physical world. Both qualitative and quantitative experiments demonstrate the necessity and effectiveness of our method compared to previous solutions. 

**Abstract (ZH)**: 通过引入物理碰撞分类器的神经网络实现带有物理感知的可微机器人渲染改进研究 

---
# Ergodic exploration of dynamic distribution 

**Title (ZH)**: 动态分布的遍历探索 

**Authors**: Luka Lanča, Karlo Jakac, Sylvain Calinon, Stefan Ivić  

**Link**: [PDF](https://arxiv.org/pdf/2503.11235)  

**Abstract**: This research addresses the challenge of performing search missions in dynamic environments, particularly for drifting targets whose movement is dictated by a flow field. This is accomplished through a dynamical system that integrates two partial differential equations: one governing the dynamics and uncertainty of the probability distribution, and the other regulating the potential field for ergodic multi-agent search. The target probability field evolves in response to the target dynamics imposed by the environment and accomplished sensing efforts, while being explored by multiple robot agents guided by the potential field gradient. The proposed methodology was tested on two simulated search scenarios, one of which features a synthetically generated domain and showcases better performance when compared to the baseline method with static target probability over a range of agent to flow field velocity ratios. The second search scenario represents a realistic sea search and rescue mission where the search start is delayed, the search is performed in multiple robot flight missions, and the procedure for target drift uncertainty compensation is demonstrated. Furthermore, the proposed method provides an accurate survey completion metric, based on the known detection/sensing parameters, that correlates with the actual number of targets found independently. 

**Abstract (ZH)**: 动态环境中漂移目标搜索任务的挑战及其通过动态系统和潜在场调控的多自主搜索方法 

---
# Flow-Aware Navigation of Magnetic Micro-Robots in Complex Fluids via PINN-Based Prediction 

**Title (ZH)**: 基于PINN预测的流感知复杂流体中磁微机器人导航 

**Authors**: Yongyi Jia, Shu Miao, Jiayu Wu, Ming Yang, Chengzhi Hu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11124)  

**Abstract**: While magnetic micro-robots have demonstrated significant potential across various applications, including drug delivery and microsurgery, the open issue of precise navigation and control in complex fluid environments is crucial for in vivo implementation. This paper introduces a novel flow-aware navigation and control strategy for magnetic micro-robots that explicitly accounts for the impact of fluid flow on their movement. First, the proposed method employs a Physics-Informed U-Net (PI-UNet) to refine the numerically predicted fluid velocity using local observations. Then, the predicted velocity is incorporated in a flow-aware A* path planning algorithm, ensuring efficient navigation while mitigating flow-induced disturbances. Finally, a control scheme is developed to compensate for the predicted fluid velocity, thereby optimizing the micro-robot's performance. A series of simulation studies and real-world experiments are conducted to validate the efficacy of the proposed approach. This method enhances both planning accuracy and control precision, expanding the potential applications of magnetic micro-robots in fluid-affected environments typical of many medical scenarios. 

**Abstract (ZH)**: 具有流场感知的磁微机器人导航与控制策略 

---
# EmbodiedVSR: Dynamic Scene Graph-Guided Chain-of-Thought Reasoning for Visual Spatial Tasks 

**Title (ZH)**: 基于身体感知的VSR：动态场景图引导的时空推理链条思考方法 

**Authors**: Yi Zhang, Qiang Zhang, Xiaozhu Ju, Zhaoyang Liu, Jilei Mao, Jingkai Sun, Jintao Wu, Shixiong Gao, Shihan Cai, Zhiyuan Qin, Linkai Liang, Jiaxu Wang, Yiqun Duan, Jiahang Cao, Renjing Xu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.11089)  

**Abstract**: While multimodal large language models (MLLMs) have made groundbreaking progress in embodied intelligence, they still face significant challenges in spatial reasoning for complex long-horizon tasks. To address this gap, we propose EmbodiedVSR (Embodied Visual Spatial Reasoning), a novel framework that integrates dynamic scene graph-guided Chain-of-Thought (CoT) reasoning to enhance spatial understanding for embodied agents. By explicitly constructing structured knowledge representations through dynamic scene graphs, our method enables zero-shot spatial reasoning without task-specific fine-tuning. This approach not only disentangles intricate spatial relationships but also aligns reasoning steps with actionable environmental dynamics. To rigorously evaluate performance, we introduce the eSpatial-Benchmark, a comprehensive dataset including real-world embodied scenarios with fine-grained spatial annotations and adaptive task difficulty levels. Experiments demonstrate that our framework significantly outperforms existing MLLM-based methods in accuracy and reasoning coherence, particularly in long-horizon tasks requiring iterative environment interaction. The results reveal the untapped potential of MLLMs for embodied intelligence when equipped with structured, explainable reasoning mechanisms, paving the way for more reliable deployment in real-world spatial applications. The codes and datasets will be released soon. 

**Abstract (ZH)**: 虽然多模态大型语言模型（MLLMs）在体现智能方面取得了突破性进展，但在复杂长期任务的空间推理方面仍面临重大挑战。为解决这一问题，我们提出了一种新颖的框架EmbodiedVSR（体现式视觉空间推理），该框架通过动态场景图引导的推理链（CoT）增强体现代理的空间理解。通过显式构建结构化知识表示，我们的方法能够在无需特定任务微调的情况下实现零样本空间推理。此方法不仅解耦复杂的空间关系，还将推理步骤与可操作的环境动态对齐。为了严格评估性能，我们引入了eSpatial-Benchmark，一个全面的数据集，包括具有精细空间注释和自适应任务难度级别的现实世界体现场景。实验结果表明，与现有的MLLM基方法相比，我们的框架在准确性及推理连贯性方面表现出显著优势，特别是在需要迭代环境交互的长期任务中。结果揭示了当装备了结构化和可解释推理机制时，MLLMs在体现智能领域的未开发潜力，为在现实空间应用中的更加可靠的部署铺平了道路。代码和数据集将于近期发布。 

---
# MoMa-Kitchen: A 100K+ Benchmark for Affordance-Grounded Last-Mile Navigation in Mobile Manipulation 

**Title (ZH)**: MoMa-Kitchen：面向操作潜能导向的移动操作最后路段导航基准数据集（包含100K以上数据） 

**Authors**: Pingrui Zhang, Xianqiang Gao, Yuhan Wu, Kehui Liu, Dong Wang, Zhigang Wang, Bin Zhao, Yan Ding, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.11081)  

**Abstract**: In mobile manipulation, navigation and manipulation are often treated as separate problems, resulting in a significant gap between merely approaching an object and engaging with it effectively. Many navigation approaches primarily define success by proximity to the target, often overlooking the necessity for optimal positioning that facilitates subsequent manipulation. To address this, we introduce MoMa-Kitchen, a benchmark dataset comprising over 100k samples that provide training data for models to learn optimal final navigation positions for seamless transition to manipulation. Our dataset includes affordance-grounded floor labels collected from diverse kitchen environments, in which robotic mobile manipulators of different models attempt to grasp target objects amidst clutter. Using a fully automated pipeline, we simulate diverse real-world scenarios and generate affordance labels for optimal manipulation positions. Visual data are collected from RGB-D inputs captured by a first-person view camera mounted on the robotic arm, ensuring consistency in viewpoint during data collection. We also develop a lightweight baseline model, NavAff, for navigation affordance grounding that demonstrates promising performance on the MoMa-Kitchen benchmark. Our approach enables models to learn affordance-based final positioning that accommodates different arm types and platform heights, thereby paving the way for more robust and generalizable integration of navigation and manipulation in embodied AI. Project page: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 移动 manipulation 中，导航和操纵经常被视为两个独立的问题，这导致了仅仅接近物体与有效地与物体互动之间的显著差距。许多导航方法主要通过与目标的接近程度来定义成功，往往忽略了为后续操纵提供最优定位的必要性。为了解决这一问题，我们引入了 MoMa-Kitchen，这是一个包含超过 10 万个样本的基准数据集，为模型学习无缝过渡到操纵的最优最终导航位置提供训练数据。我们的数据集包括从多种厨房环境中收集的基于功能的地板标签，其中不同型号的机器人移动 manipulator 尝试在杂乱环境中抓取目标物体。通过完全自动化的管道，我们模拟了各种真实世界场景并生成了最优操纵位置的功能标签。视觉数据通过安装在机器人臂上的第一视角相机捕获的 RGB-D 输入收集，确保数据收集过程中视点的一致性。我们还开发了一个轻量级基线模型 NavAff，该模型在 MoMa-Kitchen 基准测试中表现出色，用于导航功能 grounding。我们的方法使模型能够学习基于功能的最终定位，以适应不同的手臂类型和平台高度，从而为导航和 manipulation 在 embodied AI 中更稳健和可泛化的集成铺平了道路。项目页面: \href{这个链接}{这个链接}。 

---
# Training Directional Locomotion for Quadrupedal Low-Cost Robotic Systems via Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的低-cost 四足机器人方向性运动训练 

**Authors**: Peter Böhm, Archie C. Chapman, Pauline Pounds  

**Link**: [PDF](https://arxiv.org/pdf/2503.11059)  

**Abstract**: In this work we present Deep Reinforcement Learning (DRL) training of directional locomotion for low-cost quadrupedal robots in the real world. In particular, we exploit randomization of heading that the robot must follow to foster exploration of action-state transitions most useful for learning both forward locomotion as well as course adjustments. Changing the heading in episode resets to current yaw plus a random value drawn from a normal distribution yields policies able to follow complex trajectories involving frequent turns in both directions as well as long straight-line stretches. By repeatedly changing the heading, this method keeps the robot moving within the training platform and thus reduces human involvement and need for manual resets during the training. Real world experiments on a custom-built, low-cost quadruped demonstrate the efficacy of our method with the robot successfully navigating all validation tests. When trained with other approaches, the robot only succeeds in forward locomotion test and fails when turning is required. 

**Abstract (ZH)**: 本研究展示了在真实世界中，通过深度强化学习（DRL）训练低成本四足机器人方向性行进的实际应用。特别地，我们通过随机化机器人必须跟随的方向来促进对最有利于学习前行行进及路径调整的动作-状态转换的探索。通过在每个新回合重置时改变方向，使方向成为当前偏航角加上从正态分布中随机抽取的值，从而生成能够跟随复杂轨迹（包括频繁的双向转弯以及长直线段）的策略。通过反复改变方向，这种方法使机器人能够在训练平台上保持移动，从而减少人类干预和手动重置的需求。在自 built 的低成本四足机器人上进行的实际实验表明，本方法的有效性，机器人成功通过了所有验证测试。当使用其他方法进行训练时，机器人只能成功完成前行测试，而转弯时会失败。 

---
# Distributed Multi-robot Source Seeking in Unknown Environments with Unknown Number of Sources 

**Title (ZH)**: 未知环境中未知源数量的分布式多机器人寻源技术 

**Authors**: Lingpeng Chen, Siva Kailas, Srujan Deolasee, Wenhao Luo, Katia Sycara, Woojun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.11048)  

**Abstract**: We introduce a novel distributed source seeking framework, DIAS, designed for multi-robot systems in scenarios where the number of sources is unknown and potentially exceeds the number of robots. Traditional robotic source seeking methods typically focused on directing each robot to a specific strong source and may fall short in comprehensively identifying all potential sources. DIAS addresses this gap by introducing a hybrid controller that identifies the presence of sources and then alternates between exploration for data gathering and exploitation for guiding robots to identified sources. It further enhances search efficiency by dividing the environment into Voronoi cells and approximating source density functions based on Gaussian process regression. Additionally, DIAS can be integrated with existing source seeking algorithms. We compare DIAS with existing algorithms, including DoSS and GMES in simulated gas leakage scenarios where the number of sources outnumbers or is equal to the number of robots. The numerical results show that DIAS outperforms the baseline methods in both the efficiency of source identification by the robots and the accuracy of the estimated environmental density function. 

**Abstract (ZH)**: 一种新的分布式源寻觅框架DIAS：适用于多机器人系统且源的数量未知可能超过机器人类别的情况 

---
# Fast and Robust Localization for Humanoid Soccer Robot via Iterative Landmark Matching 

**Title (ZH)**: 基于迭代地标匹配的快速稳健人体形机器人足球定位 

**Authors**: Ruochen Hou, Mingzhang Zhu, Hyunwoo Nam, Gabriel I. Fernandez, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11020)  

**Abstract**: Accurate robot localization is essential for effective operation. Monte Carlo Localization (MCL) is commonly used with known maps but is computationally expensive due to landmark matching for each particle. Humanoid robots face additional challenges, including sensor noise from locomotion vibrations and a limited field of view (FOV) due to camera placement. This paper proposes a fast and robust localization method via iterative landmark matching (ILM) for humanoid robots. The iterative matching process improves the accuracy of the landmark association so that it does not need MCL to match landmarks to particles. Pose estimation with the outlier removal process enhances its robustness to measurement noise and faulty detections. Furthermore, an additional filter can be utilized to fuse inertial data from the inertial measurement unit (IMU) and pose data from localization. We compared ILM with Iterative Closest Point (ICP), which shows that ILM method is more robust towards the error in the initial guess and easier to get a correct matching. We also compared ILM with the Augmented Monte Carlo Localization (aMCL), which shows that ILM method is much faster than aMCL and even more accurate. The proposed method's effectiveness is thoroughly evaluated through experiments and validated on the humanoid robot ARTEMIS during RoboCup 2024 adult-sized soccer competition. 

**Abstract (ZH)**: 精确的机器人定位对于有效操作至关重要。本文提出了一种迭代地标匹配（ILM）方法，以快速且稳健的方式为类人机器人进行定位。我们通过迭代匹配过程提高地标关联的准确性，使其无需使用Monte Carlo定位（MCL）即可进行地标与粒子的匹配。借助离群值去除过程进行的姿态估计增强了其对测量噪声和错误检测的鲁棒性。此外，该方法可以利用滤波器融合惯性测量单元（IMU）的惯性数据和定位姿态数据。我们将ILM与迭代最近点（ICP）方法进行了对比，结果显示ILM方法对初始猜测的误差更具鲁棒性且更容易获得正确的匹配。我们还将ILM与扩展Monte Carlo定位（aMCL）进行了对比，结果显示ILM方法不仅更快而且更准确。本方法的有效性通过实验进行了全面评估，并在2024年RoboCup成人组足球比赛中，在类人机器人ARTEMIS上进行了验证。 

---
# From Abstraction to Reality: DARPA's Vision for Robust Sim-to-Real Autonomy 

**Title (ZH)**: 从抽象到现实： DARPA关于稳健的仿真到现实自主性的愿景 

**Authors**: Erfaun Noorani, Zachary Serlin, Ben Price, Alvaro Velasquez  

**Link**: [PDF](https://arxiv.org/pdf/2503.11007)  

**Abstract**: The DARPA Transfer from Imprecise and Abstract Models to Autonomous Technologies (TIAMAT) program aims to address rapid and robust transfer of autonomy technologies across dynamic and complex environments, goals, and platforms. Existing methods for simulation-to-reality (sim-to-real) transfer often rely on high-fidelity simulations and struggle with broad adaptation, particularly in time-sensitive scenarios. Although many approaches have shown incredible performance at specific tasks, most techniques fall short when posed with unforeseen, complex, and dynamic real-world scenarios due to the inherent limitations of simulation. In contrast to current research that aims to bridge the gap between simulation environments and the real world through increasingly sophisticated simulations and a combination of methods typically assuming a small sim-to-real gap -- such as domain randomization, domain adaptation, imitation learning, meta-learning, policy distillation, and dynamic optimization -- TIAMAT takes a different approach by instead emphasizing transfer and adaptation of the autonomy stack directly to real-world environments by utilizing a breadth of low(er)-fidelity simulations to create broadly effective sim-to-real transfers. By abstractly learning from multiple simulation environments in reference to their shared semantics, TIAMAT's approaches aim to achieve abstract-to-real transfer for effective and rapid real-world adaptation. Furthermore, this program endeavors to improve the overall autonomy pipeline by addressing the inherent challenges in translating simulated behaviors into effective real-world performance. 

**Abstract (ZH)**: DARPA从不精确和抽象模型向自主技术的转移（TIAMAT）计划旨在解决自主技术在动态和复杂环境、目标和平台之间的快速和稳健转移问题。现有的模拟到现实（sim-to-real）转移方法往往依赖于高保真模拟，并且在广泛的适应性方面存在困难，尤其是在时间敏感的场景中。尽管许多方法在特定任务上表现出了惊人的性能，但在遇到不可预见的、复杂的和动态的实际世界场景时，大多数技术由于模拟固有的局限性而表现不佳。与当前致力于通过越来越复杂的模拟和假设较小的sim-to-real差距组合方法来弥补模拟环境与现实世界之间差距的研究不同，TIAMAT采取了不同的方法，而是直接通过广泛应用较低保真度的模拟来强调将自主栈转移和适应到实际环境，以实现广泛有效的模拟到现实的转移。通过从多个具有共享语义的模拟环境中抽象学习，TIAMAT的方法旨在实现从抽象到现实的转移，以实现有效的快速实际世界适应。此外，该项目还致力于通过解决将模拟行为有效转化为实际世界性能的基本挑战来改进整体自主技术管道。 

---
# Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping 

**Title (ZH)**: 你的模仿学习策略比我的好？基于近最优停止的策略比较 

**Authors**: David Snyder, Asher James Hancock, Apurva Badithela, Emma Dixon, Patrick Miller, Rares Andrei Ambrus, Anirudha Majumdar, Masha Itkina, Haruki Nishimura  

**Link**: [PDF](https://arxiv.org/pdf/2503.10966)  

**Abstract**: Imitation learning has enabled robots to perform complex, long-horizon tasks in challenging dexterous manipulation settings. As new methods are developed, they must be rigorously evaluated and compared against corresponding baselines through repeated evaluation trials. However, policy comparison is fundamentally constrained by a small feasible sample size (e.g., 10 or 50) due to significant human effort and limited inference throughput of policies. This paper proposes a novel statistical framework for rigorously comparing two policies in the small sample size regime. Prior work in statistical policy comparison relies on batch testing, which requires a fixed, pre-determined number of trials and lacks flexibility in adapting the sample size to the observed evaluation data. Furthermore, extending the test with additional trials risks inducing inadvertent p-hacking, undermining statistical assurances. In contrast, our proposed statistical test is sequential, allowing researchers to decide whether or not to run more trials based on intermediate results. This adaptively tailors the number of trials to the difficulty of the underlying comparison, saving significant time and effort without sacrificing probabilistic correctness. Extensive numerical simulation and real-world robot manipulation experiments show that our test achieves near-optimal stopping, letting researchers stop evaluation and make a decision in a near-minimal number of trials. Specifically, it reduces the number of evaluation trials by up to 40% as compared to state-of-the-art baselines, while preserving the probabilistic correctness and statistical power of the comparison. Moreover, our method is strongest in the most challenging comparison instances (requiring the most evaluation trials); in a multi-task comparison scenario, we save the evaluator more than 200 simulation rollouts. 

**Abstract (ZH)**: 模仿学习使机器人能够在挑战性的灵巧操作环境中执行复杂的长期任务。随着新方法的开发，它们必须通过重复的评估试验严格评估并与其相应的基线进行对比。然而，由于显著的人工努力和策略推理吞吐量的限制，策略对比本质上受到小样本量（例如10或50）的约束。本文提出了一种新的统计框架，用于严格比较小样本量条件下两个策略。先前的统计策略对比工作依赖于批量测试，需要固定且预先确定的试验次数，并缺乏根据评估数据调整样本大小的灵活性。此外，附加试验可能会无意中诱导p-黑客行为，削弱统计保证。相比之下，我们提出的方法是顺序的，允许研究者根据中间结果决定是否运行更多试验。这种方法根据基础比较的难度灵活调整试验次数，在不牺牲概率正确性的前提下节省了大量时间和努力。广泛的数据模拟和实际机器人操作实验表明，我们的方法能够实现接近最优停止，使研究者能够在最少的试验次数内停止评估并做出决策。具体而言，与最先进的基线相比，评估试验次数减少了最多40%，同时保持了比较的统计功效与概率正确性。此外，我们的方法在最具挑战性的比较实例中表现最强（需要最多的评估试验次数）；在多任务比较场景中，我们为评估者节省了超过200个模拟迭代。 

---
# Safe Continual Domain Adaptation after Sim2Real Transfer of Reinforcement Learning Policies in Robotics 

**Title (ZH)**: 机器人学中强化学习策略Sim2Real迁移后的安全持续领域适应 

**Authors**: Josip Josifovski, Shangding Gu, Mohammadhossein Malmir, Haoliang Huang, Sayantan Auddy, Nicolás Navarro-Guerrero, Costas Spanos, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.10949)  

**Abstract**: Domain randomization has emerged as a fundamental technique in reinforcement learning (RL) to facilitate the transfer of policies from simulation to real-world robotic applications. Many existing domain randomization approaches have been proposed to improve robustness and sim2real transfer. These approaches rely on wide randomization ranges to compensate for the unknown actual system parameters, leading to robust but inefficient real-world policies. In addition, the policies pretrained in the domain-randomized simulation are fixed after deployment due to the inherent instability of the optimization processes based on RL and the necessity of sampling exploitative but potentially unsafe actions on the real system. This limits the adaptability of the deployed policy to the inevitably changing system parameters or environment dynamics over time. We leverage safe RL and continual learning under domain-randomized simulation to address these limitations and enable safe deployment-time policy adaptation in real-world robot control. The experiments show that our method enables the policy to adapt and fit to the current domain distribution and environment dynamics of the real system while minimizing safety risks and avoiding issues like catastrophic forgetting of the general policy found in randomized simulation during the pretraining phase. Videos and supplementary material are available at this https URL. 

**Abstract (ZH)**: 域随机化已成为强化学习中将策略从模拟环境迁移到实际机器人应用的一项基本技术。许多现有的域随机化方法被提出以提高鲁棒性和仿2实迁移。这些方法依赖广泛的随机化范围来补偿未知的实际系统参数，导致在实际应用中表现为稳健但效率低下的策略。此外，由于基于强化学习的优化过程固有的不稳定性以及在实际系统上必须采样具有潜在风险但可能有益的动作，预训练在域随机化模拟中的策略在部署后会被固定。这限制了部署策略对随时间变化的系统参数或环境动态的适应能力。我们通过在域随机化模拟中利用安全强化学习和持续学习来解决这些局限性，从而实现实际机器人控制中部署时策略的适应性改进。实验表明，我们的方法使策略能够适应并匹配实际系统当前的域分布和环境动态，同时最小化安全风险，并避免了在预训练阶段在随机化模拟中发现的泛化策略灾难性遗忘问题。相关视频和补充材料请访问此链接。 

---
# Rapidly Converging Time-Discounted Ergodicity on Graphs for Active Inspection of Confined Spaces 

**Title (ZH)**: 图上快速收敛时间折现遍历性在受限空间主动检测中的应用 

**Authors**: Benjamin Wong, Ryan H. Lee, Tyler M. Paine, Santosh Devasia, Ashis G. Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2503.10853)  

**Abstract**: Ergodic exploration has spawned a lot of interest in mobile robotics due to its ability to design time trajectories that match desired spatial coverage statistics. However, current ergodic approaches are for continuous spaces, which require detailed sensory information at each point and can lead to fractal-like trajectories that cannot be tracked easily. This paper presents a new ergodic approach for graph-based discretization of continuous spaces. It also introduces a new time-discounted ergodicity metric, wherein early visitations of information-rich nodes are weighted more than late visitations. A Markov chain synthesized using a convex program is shown to converge more rapidly to time-discounted ergodicity than the traditional fastest mixing Markov chain. The resultant ergodic traversal method is used within a hierarchical framework for active inspection of confined spaces with the goal of detecting anomalies robustly using SLAM-driven Bayesian hypothesis testing. Both simulation and physical experiments on a ground robot show the advantages of this framework over greedy and random exploration methods for left-behind foreign object debris detection in a ballast tank. 

**Abstract (ZH)**: 基于图的连续空间离散化的新遍历方法及时间折扣遍历性度量在受限空间中的主动检测应用 

---
# Spatial-Temporal Graph Diffusion Policy with Kinematic Modeling for Bimanual Robotic Manipulation 

**Title (ZH)**: 基于时空图扩散策略与运动学建模的双臂机器人操作 

**Authors**: Qi Lv, Hao Li, Xiang Deng, Rui Shao, Yinchuan Li, Jianye Hao, Longxiang Gao, Michael Yu Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2503.10743)  

**Abstract**: Despite the significant success of imitation learning in robotic manipulation, its application to bimanual tasks remains highly challenging. Existing approaches mainly learn a policy to predict a distant next-best end-effector pose (NBP) and then compute the corresponding joint rotation angles for motion using inverse kinematics. However, they suffer from two important issues: (1) rarely considering the physical robotic structure, which may cause self-collisions or interferences, and (2) overlooking the kinematics constraint, which may result in the predicted poses not conforming to the actual limitations of the robot joints. In this paper, we propose Kinematics enhanced Spatial-TemporAl gRaph Diffuser (KStar Diffuser). Specifically, (1) to incorporate the physical robot structure information into action prediction, KStar Diffuser maintains a dynamic spatial-temporal graph according to the physical bimanual joint motions at continuous timesteps. This dynamic graph serves as the robot-structure condition for denoising the actions; (2) to make the NBP learning objective consistent with kinematics, we introduce the differentiable kinematics to provide the reference for optimizing KStar Diffuser. This module regularizes the policy to predict more reliable and kinematics-aware next end-effector poses. Experimental results show that our method effectively leverages the physical structural information and generates kinematics-aware actions in both simulation and real-world 

**Abstract (ZH)**: Kinematics增强的空间-时间图扩散器(KStar扩散器)在双臂任务中的应用 

---
# Low-cost Real-world Implementation of the Swing-up Pendulum for Deep Reinforcement Learning Experiments 

**Title (ZH)**: 低成本实际实施的摆动起立摆系统用于深度强化学习实验 

**Authors**: Peter Böhm, Pauline Pounds, Archie C. Chapman  

**Link**: [PDF](https://arxiv.org/pdf/2503.11065)  

**Abstract**: Deep reinforcement learning (DRL) has had success in virtual and simulated domains, but due to key differences between simulated and real-world environments, DRL-trained policies have had limited success in real-world applications. To assist researchers to bridge the \textit{sim-to-real gap}, in this paper, we describe a low-cost physical inverted pendulum apparatus and software environment for exploring sim-to-real DRL methods. In particular, the design of our apparatus enables detailed examination of the delays that arise in physical systems when sensing, communicating, learning, inferring and actuating. Moreover, we wish to improve access to educational systems, so our apparatus uses readily available materials and parts to reduce cost and logistical barriers. Our design shows how commercial, off-the-shelf electronics and electromechanical and sensor systems, combined with common metal extrusions, dowel and 3D printed couplings provide a pathway for affordable physical DRL apparatus. The physical apparatus is complemented with a simulated environment implemented using a high-fidelity physics engine and OpenAI Gym interface. 

**Abstract (ZH)**: 深强化学习（DRL）在虚拟和模拟领域取得了成功，但由于模拟环境与实际环境之间的关键差异，DRL训练的策略在实际应用中的效果有限。为了协助研究人员弥合“仿真实到实际”差距，本文描述了一种低成本的物理倒立摆装置及其软件环境，以探索仿真实到实际的DRL方法。该装置的设计允许对物理系统中感应、通信、学习、推理和执行过程中产生的延迟进行详细研究。此外，为了提高教育系统的可访问性，该装置使用易获取的材料和部件来降低成本和物流障碍。我们的设计展示了如何通过将商用现成的电子、机电和传感器系统与常见的金属型材、杆和3D打印接头结合，提供一种经济实惠的物理DRL装置的路径。该物理装置与使用高保真物理引擎和OpenAI Gym接口实现的模拟环境相辅相成。 

---
# SciFi-Benchmark: How Would AI-Powered Robots Behave in Science Fiction Literature? 

**Title (ZH)**: SciFi-Benchmark: AI驱动的机器人在科幻文学中将如何行为？ 

**Authors**: Pierre Sermanet, Anirudha Majumdar, Vikas Sindhwani  

**Link**: [PDF](https://arxiv.org/pdf/2503.10706)  

**Abstract**: Given the recent rate of progress in artificial intelligence (AI) and robotics, a tantalizing question is emerging: would robots controlled by emerging AI systems be strongly aligned with human values? In this work, we propose a scalable way to probe this question by generating a benchmark spanning the key moments in 824 major pieces of science fiction literature (movies, tv, novels and scientific books) where an agent (AI or robot) made critical decisions (good or bad). We use a LLM's recollection of each key moment to generate questions in similar situations, the decisions made by the agent, and alternative decisions it could have made (good or bad). We then measure an approximation of how well models align with human values on a set of human-voted answers. We also generate rules that can be automatically improved via amendment process in order to generate the first Sci-Fi inspired constitutions for promoting ethical behavior in AIs and robots in the real world. Our first finding is that modern LLMs paired with constitutions turn out to be well-aligned with human values (95.8%), contrary to unsettling decisions typically made in SciFi (only 21.2% alignment). Secondly, we find that generated constitutions substantially increase alignment compared to the base model (79.4% to 95.8%), and show resilience to an adversarial prompt setting (23.3% to 92.3%). Additionally, we find that those constitutions are among the top performers on the ASIMOV Benchmark which is derived from real-world images and hospital injury reports. Sci-Fi-inspired constitutions are thus highly aligned and applicable in real-world situations. We release SciFi-Benchmark: a large-scale dataset to advance robot ethics and safety research. It comprises 9,056 questions and 53,384 answers, in addition to a smaller human-labeled evaluation set. Data is available at this https URL 

**Abstract (ZH)**: 基于科幻文学的机器人伦理与安全基准：现代大型语言模型与科幻启发宪法在促进人工智能伦理行为中的应用 

---
# Estimating Control Barriers from Offline Data 

**Title (ZH)**: 从离线数据估计控制障碍物 

**Authors**: Hongzhan Yu, Seth Farrell, Ryo Yoshimitsu, Zhizhen Qin, Henrik I. Christensen, Sicun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10641)  

**Abstract**: Learning-based methods for constructing control barrier functions (CBFs) are gaining popularity for ensuring safe robot control. A major limitation of existing methods is their reliance on extensive sampling over the state space or online system interaction in simulation. In this work we propose a novel framework for learning neural CBFs through a fixed, sparsely-labeled dataset collected prior to training. Our approach introduces new annotation techniques based on out-of-distribution analysis, enabling efficient knowledge propagation from the limited labeled data to the unlabeled data. We also eliminate the dependency on a high-performance expert controller, and allow multiple sub-optimal policies or even manual control during data collection. We evaluate the proposed method on real-world platforms. With limited amount of offline data, it achieves state-of-the-art performance for dynamic obstacle avoidance, demonstrating statistically safer and less conservative maneuvers compared to existing methods. 

**Abstract (ZH)**: 基于学习的方法用于构造控制障函数（CBFs），以确保机器人控制的安全性正逐渐流行。现有方法的主要限制在于其依赖于状态空间的大量采样或在线系统仿真中的交互。本文提出了一种新的框架，通过在训练前收集的固定、稀疏标记的数据集来学习神经CBFs。该方法引入了基于离分布分析的新注释技术，能够高效地将有限标记数据的知识传播到未标记数据中。此外，该方法消除了对高性能专家控制器的依赖，并允许在数据采集过程中使用多个次优策略或甚至手动控制。我们在实际平台进行了评估，在有限的离线数据下，该方法在动态障碍物避让方面达到了最先进的性能，展示了与现有方法相比更为统计意义上的安全且保守度更低的操作。 

---
# AIstorian lets AI be a historian: A KG-powered multi-agent system for accurate biography generation 

**Title (ZH)**: AIstorian 让 AI 成为历史学家：一种基于知识图谱的多agent系统，用于生成准确的传记 

**Authors**: Fengyu Li, Yilin Li, Junhao Zhu, Lu Chen, Yanfei Zhang, Jia Zhou, Hui Zu, Jingwen Zhao, Yunjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.11346)  

**Abstract**: Huawei has always been committed to exploring the AI application in historical research. Biography generation, as a specialized form of abstractive summarization, plays a crucial role in historical research but faces unique challenges that existing large language models (LLMs) struggle to address. These challenges include maintaining stylistic adherence to historical writing conventions, ensuring factual fidelity, and handling fragmented information across multiple documents. We present AIstorian, a novel end-to-end agentic system featured with a knowledge graph (KG)-powered retrieval-augmented generation (RAG) and anti-hallucination multi-agents. Specifically, AIstorian introduces an in-context learning based chunking strategy and a KG-based index for accurate and efficient reference retrieval. Meanwhile, AIstorian orchestrates multi-agents to conduct on-the-fly hallucination detection and error-type-aware correction. Additionally, to teach LLMs a certain language style, we finetune LLMs based on a two-step training approach combining data augmentation-enhanced supervised fine-tuning with stylistic preference optimization. Extensive experiments on a real-life historical Jinshi dataset demonstrate that AIstorian achieves a 3.8x improvement in factual accuracy and a 47.6% reduction in hallucination rate compared to existing baselines. The data and code are available at: this https URL. 

**Abstract (ZH)**: 华为始终致力于探索AI在历史研究中的应用。AIstorian：一种基于知识图谱的检索增强生成和反幻觉多智能体系统在历史研究中的应用 

---
# BriLLM: Brain-inspired Large Language Model 

**Title (ZH)**: Brain-inspired Large Language Model 

**Authors**: Hai Zhao, Hongqiu Wu, Dongjie Yang, Anni Zou, Jiale Hong  

**Link**: [PDF](https://arxiv.org/pdf/2503.11299)  

**Abstract**: This paper reports the first brain-inspired large language model (BriLLM). This is a non-Transformer, non-GPT, non-traditional machine learning input-output controlled generative language model. The model is based on the Signal Fully-connected flowing (SiFu) definition on the directed graph in terms of the neural network, and has the interpretability of all nodes on the graph of the whole model, instead of the traditional machine learning model that only has limited interpretability at the input and output ends. In the language model scenario, the token is defined as a node in the graph. A randomly shaped or user-defined signal flow flows between nodes on the principle of "least resistance" along paths. The next token or node to be predicted or generated is the target of the signal flow. As a language model, BriLLM theoretically supports infinitely long $n$-gram models when the model size is independent of the input and predicted length of the model. The model's working signal flow provides the possibility of recall activation and innate multi-modal support similar to the cognitive patterns of the human brain. At present, we released the first BriLLM version in Chinese, with 4000 tokens, 32-dimensional node width, 16-token long sequence prediction ability, and language model prediction performance comparable to GPT-1. More computing power will help us explore the infinite possibilities depicted above. 

**Abstract (ZH)**: This paper报告了首个脑启发大规模语言模型（BriLLM）。 

---
# Taxonomic Reasoning for Rare Arthropods: Combining Dense Image Captioning and RAG for Interpretable Classification 

**Title (ZH)**: 稀有节肢动物分类学推理：结合密集图像描述和RAG进行可解释分类 

**Authors**: Nathaniel Lesperance, Sujeevan Ratnasingham, Graham W. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2503.10886)  

**Abstract**: In the context of pressing climate change challenges and the significant biodiversity loss among arthropods, automated taxonomic classification from organismal images is a subject of intense research. However, traditional AI pipelines based on deep neural visual architectures such as CNNs or ViTs face limitations such as degraded performance on the long-tail of classes and the inability to reason about their predictions. We integrate image captioning and retrieval-augmented generation (RAG) with large language models (LLMs) to enhance biodiversity monitoring, showing particular promise for characterizing rare and unknown arthropod species. While a naive Vision-Language Model (VLM) excels in classifying images of common species, the RAG model enables classification of rarer taxa by matching explicit textual descriptions of taxonomic features to contextual biodiversity text data from external sources. The RAG model shows promise in reducing overconfidence and enhancing accuracy relative to naive LLMs, suggesting its viability in capturing the nuances of taxonomic hierarchy, particularly at the challenging family and genus levels. Our findings highlight the potential for modern vision-language AI pipelines to support biodiversity conservation initiatives, emphasizing the role of comprehensive data curation and collaboration with citizen science platforms to improve species identification, unknown species characterization and ultimately inform conservation strategies. 

**Abstract (ZH)**: 在迫切的气候变化挑战和显著的昆虫多样性丧失背景下，基于生物图像的自动化分类是研究的热点。然而，传统的基于深度神经视觉架构如CNNs或ViTs的人工智能管道存在长尾类别的性能下降和无法解释其预测的问题。我们将图像描述和检索增强生成（RAG）与大型语言模型（LLMs）集成，以增强生物多样性监测，特别适用于描述稀有和未知昆虫物种。尽管一个简单的视觉语言模型（VLM）在分类常见物种的图像方面表现出色，但RAG模型通过将税务特征的显式文本描述与外部来源的背景生物多样性文本数据匹配，实现稀有类别的分类。RAG模型在降低过度自信和提高准确性方面显示出潜力，相对于简单的LLMs，它有可能捕捉到税务层次结构的细微差别，特别是在具有挑战性的家族和属级别。我们的发现强调，现代视觉语言人工智能管道有可能支持生物多样性保护倡议，强调全面数据整理和与公民科学平台合作的重要性，以提高物种识别、未知物种描述，并最终制定保护策略。 

---
# Zero-Shot Subject-Centric Generation for Creative Application Using Entropy Fusion 

**Title (ZH)**: 基于熵融合的零样本主题中心生成在创意应用中的研究 

**Authors**: Kaifeng Zou, Xiaoyi Feng, Peng Wang, Tao Huang, Zizhou Huang, Zhang Haihang, Yuntao Zou, Dagang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10697)  

**Abstract**: Generative models are widely used in visual content creation. However, current text-to-image models often face challenges in practical applications-such as textile pattern design and meme generation-due to the presence of unwanted elements that are difficult to separate with existing methods. Meanwhile, subject-reference generation has emerged as a key research trend, highlighting the need for techniques that can produce clean, high-quality subject images while effectively removing extraneous components. To address this challenge, we introduce a framework for reliable subject-centric image generation. In this work, we propose an entropy-based feature-weighted fusion method to merge the informative cross-attention features obtained from each sampling step of the pretrained text-to-image model FLUX, enabling a precise mask prediction and subject-centric generation. Additionally, we have developed an agent framework based on Large Language Models (LLMs) that translates users' casual inputs into more descriptive prompts, leading to highly detailed image generation. Simultaneously, the agents extract primary elements of prompts to guide the entropy-based feature fusion, ensuring focused primary element generation without extraneous components. Experimental results and user studies demonstrate our methods generates high-quality subject-centric images, outperform existing methods or other possible pipelines, highlighting the effectiveness of our approach. 

**Abstract (ZH)**: 基于熵的特征加权融合框架实现可靠的主题导向图像生成 

---
# Enhancing Retrieval for ESGLLM via ESG-CID -- A Disclosure Content Index Finetuning Dataset for Mapping GRI and ESRS 

**Title (ZH)**: 通过ESG-CID——一个用于映射GRI和ESRS的披露内容索引微调数据集，增强ESGLLM的检索性能 

**Authors**: Shafiuddin Rehan Ahmed, Ankit Parag Shah, Quan Hung Tran, Vivek Khetan, Sukryool Kang, Ankit Mehta, Yujia Bao, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.10674)  

**Abstract**: Climate change has intensified the need for transparency and accountability in organizational practices, making Environmental, Social, and Governance (ESG) reporting increasingly crucial. Frameworks like the Global Reporting Initiative (GRI) and the new European Sustainability Reporting Standards (ESRS) aim to standardize ESG reporting, yet generating comprehensive reports remains challenging due to the considerable length of ESG documents and variability in company reporting styles. To facilitate ESG report automation, Retrieval-Augmented Generation (RAG) systems can be employed, but their development is hindered by a lack of labeled data suitable for training retrieval models. In this paper, we leverage an underutilized source of weak supervision -- the disclosure content index found in past ESG reports -- to create a comprehensive dataset, ESG-CID, for both GRI and ESRS standards. By extracting mappings between specific disclosure requirements and corresponding report sections, and refining them using a Large Language Model as a judge, we generate a robust training and evaluation set. We benchmark popular embedding models on this dataset and show that fine-tuning BERT-based models can outperform commercial embeddings and leading public models, even under temporal data splits for cross-report style transfer from GRI to ESRS 

**Abstract (ZH)**: 气候变迁加剧了组织实践透明度和责任的需要，使得环境、社会和治理（ESG）报告愈加重要。全球报告倡议组织（GRI）框架和新的欧洲可持续性报告标准（ESRS）旨在标准化ESG报告，但由于ESG文件的长度繁多和公司报告风格的 variability，生成全面的报告依然颇具挑战性。为了促进ESG报告的自动化，可以采用检索增强生成（RAG）系统，但其开发受限于适用于训练检索模型的标记数据不足。本文利用过去ESG报告中存在的未充分利用的弱监督来源——披露内容索引，创建了适用于GRI和ESRS标准的全面数据集ESG-CID。通过提取特定披露要求与相应报告部分之间的映射，并使用大型语言模型进行评判以进行细化，我们生成了一个稳健的训练和评估集。我们在该数据集上对流行嵌入模型进行了基准测试，并展示了针对时间数据拆分下的从GRI到ESRS的跨报告风格转移，微调基于BERT的模型可以优于商用嵌入和领先公开模型。 

---
# IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models 

**Title (ZH)**: 基于视觉-语言模型的智能可接受接触轨迹规划方法 

**Authors**: Yiyang Ling, Karan Owalekar, Oluwatobiloba Adesanya, Erdem Bıyık, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2503.10110)  

**Abstract**: Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g., brushing a soft pillow) to more dangerous (e.g., toppling a glass vase). Due to this diversity, it is difficult to characterize which contacts may be acceptable or unacceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach uses the VLM's outputs to produce a dense 3D "cost map" that encodes contact tolerances and seamlessly integrates with standard motion planners. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3620 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Supplementary material is available at this https URL. 

**Abstract (ZH)**: 运动规划涉及确定机器人配置序列以达到所需姿态，并满足运动和安全约束。传统运动规划寻找无碰撞路径，但在复杂环境中，机器人可能无法完成任务而无法接触物体。此外，接触可以从相对温和（例如，轻触柔软的枕头）到更危险（例如，推翻玻璃花瓶）。由于这种多样性，很难确定哪些接触是可以接受还是不可以接受。在本文中，我们提出了IMPACT，这是一种新颖的运动规划框架，利用视觉-语言模型（VLMs）推断环境语义，根据物体属性和位置确定环境中哪些部分可以最好地容忍接触。我们的方法利用VLM的输出生成一个密集的3D“成本图”，编码接触容忍度，并无缝集成到标准运动规划器中。我们使用20个模拟场景和10个真实场景进行实验，并通过任务成功率、物体位移和人类评价者的反馈进行评估。我们的结果显示，在3620个模拟和200个真实场景试验中，IMPACT在复杂环境中共充分体现接触丰富的运动规划效率，并优于其他方法和消除实验。补充材料可在此处访问：this https URL。 

---
