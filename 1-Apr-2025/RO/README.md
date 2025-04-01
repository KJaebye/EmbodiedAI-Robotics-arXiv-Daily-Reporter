# Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation 

**Title (ZH)**: 基于视觉的机器人 manipulation 的 Sim-and-Real 共训练：一个简单的解决方案 

**Authors**: Abhiram Maddukuri, Zhenyu Jiang, Lawrence Yunliang Chen, Soroush Nasiriany, Yuqi Xie, Yu Fang, Wenqi Huang, Zu Wang, Zhenjia Xu, Nikita Chernyadev, Scott Reed, Ken Goldberg, Ajay Mandlekar, Linxi Fan, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24361)  

**Abstract**: Large real-world robot datasets hold great potential to train generalist robot models, but scaling real-world human data collection is time-consuming and resource-intensive. Simulation has great potential in supplementing large-scale data, especially with recent advances in generative AI and automated data generation tools that enable scalable creation of robot behavior datasets. However, training a policy solely in simulation and transferring it to the real world often demands substantial human effort to bridge the reality gap. A compelling alternative is to co-train the policy on a mixture of simulation and real-world datasets. Preliminary studies have recently shown this strategy to substantially improve the performance of a policy over one trained on a limited amount of real-world data. Nonetheless, the community lacks a systematic understanding of sim-and-real co-training and what it takes to reap the benefits of simulation data for real-robot learning. This work presents a simple yet effective recipe for utilizing simulation data to solve vision-based robotic manipulation tasks. We derive this recipe from comprehensive experiments that validate the co-training strategy on various simulation and real-world datasets. Using two domains--a robot arm and a humanoid--across diverse tasks, we demonstrate that simulation data can enhance real-world task performance by an average of 38%, even with notable differences between the simulation and real-world data. Videos and additional results can be found at this https URL 

**Abstract (ZH)**: 利用模拟数据解决视觉导向机器人 manipulation 任务的简单有效方法：基于混合仿真与真实数据的共同训练 

---
# Pro-Routing: Proactive Routing of Autonomous Multi-Capacity Robots for Pickup-and-Delivery Tasks 

**Title (ZH)**: 主动导航：自主多容量机器人面向拣取与配送任务的前瞻路由 

**Authors**: Daniel Garces, Stephanie Gil  

**Link**: [PDF](https://arxiv.org/pdf/2503.24325)  

**Abstract**: We consider a multi-robot setting, where we have a fleet of multi-capacity autonomous robots that must service spatially distributed pickup-and-delivery requests with fixed maximum wait times. Requests can be either scheduled ahead of time or they can enter the system in real-time. In this setting, stability for a routing policy is defined as the cost of the policy being uniformly bounded over time. Most previous work either solve the problem offline to theoretically maintain stability or they consider dynamically arriving requests at the expense of the theoretical guarantees on stability. In this paper, we aim to bridge this gap by proposing a novel proactive rollout-based routing framework that adapts to real-time demand while still provably maintaining the stability of the learned routing policy. We derive provable stability guarantees for our method by proposing a fleet sizing algorithm that obtains a sufficiently large fleet that ensures stability by construction. To validate our theoretical results, we consider a case study on real ride requests for Harvard's evening Van System. We also evaluate the performance of our framework using the currently deployed smaller fleet size. In this smaller setup, we compare against the currently deployed routing algorithm, greedy heuristics, and Monte-Carlo-Tree-Search-based algorithms. Our empirical results show that our framework maintains stability when we use the sufficiently large fleet size found in our theoretical results. For the smaller currently deployed fleet size, our method services 6% more requests than the closest baseline while reducing median passenger wait times by 33%. 

**Abstract (ZH)**: 我们探讨了一个多机器人设置，其中拥有一支具有多种容量的自主机器人队列，这些机器人必须在固定的最大等待时间内服务空间分布的取货和配送请求。请求可以提前安排，也可以在系统中实时进入。在这种设置下，路由策略的稳定性定义为策略的成本随时间均匀有界。大多数先前的工作要么通过离线求解问题来理论上保持稳定性，要么考虑动态到达的请求以牺牲稳定性理论保证为代价。在本文中，我们旨在通过提出一种新颖的主动展开基于的路由框架来弥合这一差距，该框架能够适应实时需求的同时，仍能证明地保持学习到的路由策略的稳定性。通过提出一种车队规模算法来获得一个足够大的车队以构造性地确保稳定性，我们为我们的方法推导出了可证明的稳定性保证。为了验证我们的理论结果，我们以哈佛大学晚间的班车系统中的真实乘车请求为案例研究。我们也使用当前部署的较小的车队规模评估了我们框架的性能。在较小的设置中，我们将我们的方法与当前部署的路由算法、贪婪启发式算法以及基于蒙特卡洛树搜索的算法进行比较。我们的实验证明，在使用我们理论结果中发现的足够大的车队规模时，我们的框架能够保持稳定性。对于当前部署的较小的车队规模，我们的方法比最接近的基线多服务6%的请求，同时将中间乘客等待时间减少了33%。 

---
# AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World 

**Title (ZH)**: AutoEval: 自主评估通用机器人操作政策在现实世界中的性能 

**Authors**: Zhiyuan Zhou, Pranav Atreya, You Liang Tan, Karl Pertsch, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2503.24278)  

**Abstract**: Scalable and reproducible policy evaluation has been a long-standing challenge in robot learning. Evaluations are critical to assess progress and build better policies, but evaluation in the real world, especially at a scale that would provide statistically reliable results, is costly in terms of human time and hard to obtain. Evaluation of increasingly generalist robot policies requires an increasingly diverse repertoire of evaluation environments, making the evaluation bottleneck even more pronounced. To make real-world evaluation of robotic policies more practical, we propose AutoEval, a system to autonomously evaluate generalist robot policies around the clock with minimal human intervention. Users interact with AutoEval by submitting evaluation jobs to the AutoEval queue, much like how software jobs are submitted with a cluster scheduling system, and AutoEval will schedule the policies for evaluation within a framework supplying automatic success detection and automatic scene resets. We show that AutoEval can nearly fully eliminate human involvement in the evaluation process, permitting around the clock evaluations, and the evaluation results correspond closely to ground truth evaluations conducted by hand. To facilitate the evaluation of generalist policies in the robotics community, we provide public access to multiple AutoEval scenes in the popular BridgeData robot setup with WidowX robot arms. In the future, we hope that AutoEval scenes can be set up across institutions to form a diverse and distributed evaluation network. 

**Abstract (ZH)**: 可扩展且可重复的政策评估一直是机器人学习中的长期挑战。为了使机器人的政策评估更加实用，我们提出了AutoEval系统，该系统能够在最少的人工干预下，全天候自主评估通用机器人政策。通过提交评估任务到AutoEval队列，用户可以像提交软件任务到集群调度系统一样操作，AutoEval将在框架内自动进行成功率检测并自动重置场景。我们展示了AutoEval几乎完全消除了评估过程中的人工干预，实现了全天候评估，并且评估结果与手工进行的真实情况评估高度一致。为促进机器人社区中通用政策的评估，我们提供了对BridgeData机器人设置中使用WidowX机器人手臂的多个AutoEval场景的公共访问权限。未来，我们希望在不同的机构中设置AutoEval场景，形成一个多样且分布式的评估网络。 

---
# Pseudo-Random UAV Test Generation Using Low-Fidelity Path Simulator 

**Title (ZH)**: 使用低保真路径模拟器的伪随机无人机测试生成 

**Authors**: Anas Shrinah, Kerstin Eder  

**Link**: [PDF](https://arxiv.org/pdf/2503.24172)  

**Abstract**: Simulation-based testing provides a safe and cost-effective environment for verifying the safety of Uncrewed Aerial Vehicles (UAVs). However, simulation can be resource-consuming, especially when High-Fidelity Simulators (HFS) are used. To optimise simulation resources, we propose a pseudo-random test generator that uses a Low-Fidelity Simulator (LFS) to estimate UAV flight paths. This work simplifies the PX4 autopilot HFS to develop a LFS, which operates one order of magnitude faster than the this http URL cases predicted to cause safety violations in the LFS are subsequently validated using the HFS. 

**Abstract (ZH)**: 基于仿真的测试提供了一种安全且经济有效的环境，用于验证无人机(UAV)的安全性。然而，仿真的资源消耗极大，特别是在使用高保真模拟器(HFS)的情况下。为了优化仿真资源，我们提出了一种伪随机测试生成器，该生成器利用低保真模拟器(LFS)估计无人机飞行路径。本工作简化了PX4自动驾驶仪的HFS，开发了一个LFS，后者的速度比HFS快一个数量级。通过LFS预测可能导致安全违规的情况随后在HFS中进行验证。 

---
# HACTS: a Human-As-Copilot Teleoperation System for Robot Learning 

**Title (ZH)**: HACTS: 一种人类辅助飞行员的机器人学习远程操控系统 

**Authors**: Zhiyuan Xu, Yinuo Zhao, Kun Wu, Ning Liu, Junjie Ji, Zhengping Che, Chi Harold Liu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.24070)  

**Abstract**: Teleoperation is essential for autonomous robot learning, especially in manipulation tasks that require human demonstrations or corrections. However, most existing systems only offer unilateral robot control and lack the ability to synchronize the robot's status with the teleoperation hardware, preventing real-time, flexible intervention. In this work, we introduce HACTS (Human-As-Copilot Teleoperation System), a novel system that establishes bilateral, real-time joint synchronization between a robot arm and teleoperation hardware. This simple yet effective feedback mechanism, akin to a steering wheel in autonomous vehicles, enables the human copilot to intervene seamlessly while collecting action-correction data for future learning. Implemented using 3D-printed components and low-cost, off-the-shelf motors, HACTS is both accessible and scalable. Our experiments show that HACTS significantly enhances performance in imitation learning (IL) and reinforcement learning (RL) tasks, boosting IL recovery capabilities and data efficiency, and facilitating human-in-the-loop RL. HACTS paves the way for more effective and interactive human-robot collaboration and data-collection, advancing the capabilities of robot manipulation. 

**Abstract (ZH)**: 基于副驾操控的双工实时同步系统：一种用于自主机器人学习的人机协作操控框架 

---
# Toward Anxiety-Reducing Pocket Robots for Children 

**Title (ZH)**: 面向儿童的减压口袋机器人 

**Authors**: Morten Roed Frederiksen, Kasper Støy, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2503.24041)  

**Abstract**: A common denominator for most therapy treatments for children who suffer from an anxiety disorder is daily practice routines to learn techniques needed to overcome anxiety. However, applying those techniques while experiencing anxiety can be highly challenging. This paper presents the design, implementation, and pilot study of a tactile hand-held pocket robot AffectaPocket, designed to work alongside therapy as a focus object to facilitate coping during an anxiety attack. The robot does not require daily practice to be used, has a small form factor, and has been designed for children 7 to 12 years old. The pocket robot works by sensing when it is being held and attempts to shift the child's focus by presenting them with a simple three-note rhythm-matching game. We conducted a pilot study of the pocket robot involving four children aged 7 to 10 years, and then a main study with 18 children aged 6 to 8 years; neither study involved children with anxiety. Both studies aimed to assess the reliability of the robot's sensor configuration, its design, and the effectiveness of the user tutorial. The results indicate that the morphology and sensor setup performed adequately and the tutorial process enabled the children to use the robot with little practice. This work demonstrates that the presented pocket robot could represent a step toward developing low-cost accessible technologies to help children suffering from anxiety disorders. 

**Abstract (ZH)**: 大多数治疗儿童焦虑障碍的疗法的共同之处是每天练习以学习克服焦虑所需的技巧。然而，在经历焦虑时应用这些技巧极具挑战性。本文介绍了指尖手持口袋机器人AffectaPocket的设计、实现及其初步研究，该机器人旨在与治疗配合使用，作为焦点对象，帮助儿童在焦虑发作时进行应对。该机器人不需要每日练习即可使用，体积小巧，专为7至12岁儿童设计。该口袋机器人通过感知何时被握住，并通过呈现一个简单的三音符节奏匹配游戏来尝试转移孩子的注意力。我们对7至10岁四名儿童进行了口袋机器人的初步研究，随后对6至8岁18名儿童进行了主要研究；两者的参与者均未患有焦虑症。两项研究均旨在评估机器人传感器配置的可靠性、其设计以及用户教程的有效性。结果显示，机器人的形态和传感器设置表现良好，教程过程使儿童能够在不进行大量练习的情况下使用该机器人。研究表明，提出的口袋机器人可能代表了一种朝着开发低成本可访问技术以帮助患有焦虑障碍的儿童迈出的一步。 

---
# A Reactive Framework for Whole-Body Motion Planning of Mobile Manipulators Combining Reinforcement Learning and SDF-Constrained Quadratic Programmi 

**Title (ZH)**: 基于强化学习和SDF约束二次规划的移动 manipulator 全身运动规划反应框架 

**Authors**: Chenyu Zhang, Shiying Sun, Kuan Liu, Chuanbao Zhou, Xiaoguang Zhao, Min Tan, Yanlong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23975)  

**Abstract**: As an important branch of embodied artificial intelligence, mobile manipulators are increasingly applied in intelligent services, but their redundant degrees of freedom also limit efficient motion planning in cluttered environments. To address this issue, this paper proposes a hybrid learning and optimization framework for reactive whole-body motion planning of mobile manipulators. We develop the Bayesian distributional soft actor-critic (Bayes-DSAC) algorithm to improve the quality of value estimation and the convergence performance of the learning. Additionally, we introduce a quadratic programming method constrained by the signed distance field to enhance the safety of the obstacle avoidance motion. We conduct experiments and make comparison with standard benchmark. The experimental results verify that our proposed framework significantly improves the efficiency of reactive whole-body motion planning, reduces the planning time, and improves the success rate of motion planning. Additionally, the proposed reinforcement learning method ensures a rapid learning process in the whole-body planning task. The novel framework allows mobile manipulators to adapt to complex environments more safely and efficiently. 

**Abstract (ZH)**: 移动 manipulator 动态全身体现人工智能中的重要分支，在智能服务中越来越受到关注，但其多余的自由度也限制了在复杂环境中的高效运动规划。为了解决这一问题，本文提出了一个结合学习和优化的混合框架，用于移动 manipulator 反应式全身体现运动规划。我们开发了贝叶斯分布柔软行动者-评论家（Bayes-DSAC）算法，以提高价值估计的质量和学习的收敛性能。此外，我们引入了一种基于符号距离场约束的二次规划方法，以增强障碍物避让运动的安全性。我们进行了实验并与标准基准进行了比较。实验结果验证了我们提出的框架显著提高了反应式全身体现运动规划的效率，减少了规划时间，并提高了运动规划的成功率。此外，提出的强化学习方法确保了在全身体现规划任务中的快速学习过程。该新颖框架使得移动 manipulator 能够更安全、更有效地适应复杂环境。 

---
# MAER-Nav: Bidirectional Motion Learning Through Mirror-Augmented Experience Replay for Robot Navigation 

**Title (ZH)**: MAER-Nav: 通过镜像增强经验回放的双向运动学习用于机器人导航 

**Authors**: Shanze Wang, Mingao Tan, Zhibo Yang, Biao Huang, Xiaoyu Shen, Hailong Huang, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23908)  

**Abstract**: Deep Reinforcement Learning (DRL) based navigation methods have demonstrated promising results for mobile robots, but suffer from limited action flexibility in confined spaces. Conventional DRL approaches predominantly learn forward-motion policies, causing robots to become trapped in complex environments where backward maneuvers are necessary for recovery. This paper presents MAER-Nav (Mirror-Augmented Experience Replay for Robot Navigation), a novel framework that enables bidirectional motion learning without requiring explicit failure-driven hindsight experience replay or reward function modifications. Our approach integrates a mirror-augmented experience replay mechanism with curriculum learning to generate synthetic backward navigation experiences from successful trajectories. Experimental results in both simulation and real-world environments demonstrate that MAER-Nav significantly outperforms state-of-the-art methods while maintaining strong forward navigation capabilities. The framework effectively bridges the gap between the comprehensive action space utilization of traditional planning methods and the environmental adaptability of learning-based approaches, enabling robust navigation in scenarios where conventional DRL methods consistently fail. 

**Abstract (ZH)**: 基于镜像增强经验回放的双向运动学习框架（MAER-Nav）：增强移动机器人在受限空间中的导航能力 

---
# Less is More: Contextual Sampling for Nonlinear Data-Enabled Predictive Control 

**Title (ZH)**: 少即是多：上下文采样在非线性数据驱动预测控制中的应用 

**Authors**: Julius Beerwerth, Bassam Alrifaee  

**Link**: [PDF](https://arxiv.org/pdf/2503.23890)  

**Abstract**: Data-enabled Predictive Control (DeePC) is a powerful data-driven approach for predictive control without requiring an explicit system model. However, its high computational cost limits its applicability to real-time robotic systems. For robotic applications such as motion planning and trajectory tracking, real-time control is crucial. Nonlinear DeePC either relies on large datasets or learning the nonlinearities to ensure predictive accuracy, leading to high computational complexity. This work introduces contextual sampling, a novel data selection strategy to handle nonlinearities for DeePC by dynamically selecting the most relevant data at each time step. By reducing the dataset size while preserving prediction accuracy, our method improves computational efficiency, of DeePC for real-time robotic applications. We validate our approach for autonomous vehicle motion planning. For a dataset size of 100 sub-trajectories, Contextual sampling DeePC reduces tracking error by 53.2 % compared to Leverage Score sampling. Additionally, Contextual sampling reduces max computation time by 87.2 % compared to using the full dataset of 491 sub-trajectories while achieving comparable tracking performance. These results highlight the potential of Contextual sampling to enable real-time, data-driven control for robotic systems. 

**Abstract (ZH)**: 基于上下文的数据选择策略增强的DeePC实时机器人应用中的预测控制 

---
# ZeroMimic: Distilling Robotic Manipulation Skills from Web Videos 

**Title (ZH)**: ZeroMimic: 从网络视频中提炼机器人操作技能 

**Authors**: Junyao Shi, Zhuolun Zhao, Tianyou Wang, Ian Pedroza, Amy Luo, Jie Wang, Jason Ma, Dinesh Jayaraman  

**Link**: [PDF](https://arxiv.org/pdf/2503.23877)  

**Abstract**: Many recent advances in robotic manipulation have come through imitation learning, yet these rely largely on mimicking a particularly hard-to-acquire form of demonstrations: those collected on the same robot in the same room with the same objects as the trained policy must handle at test time. In contrast, large pre-recorded human video datasets demonstrating manipulation skills in-the-wild already exist, which contain valuable information for robots. Is it possible to distill a repository of useful robotic skill policies out of such data without any additional requirements on robot-specific demonstrations or exploration? We present the first such system ZeroMimic, that generates immediately deployable image goal-conditioned skill policies for several common categories of manipulation tasks (opening, closing, pouring, pick&place, cutting, and stirring) each capable of acting upon diverse objects and across diverse unseen task setups. ZeroMimic is carefully designed to exploit recent advances in semantic and geometric visual understanding of human videos, together with modern grasp affordance detectors and imitation policy classes. After training ZeroMimic on the popular EpicKitchens dataset of ego-centric human videos, we evaluate its out-of-the-box performance in varied real-world and simulated kitchen settings with two different robot embodiments, demonstrating its impressive abilities to handle these varied tasks. To enable plug-and-play reuse of ZeroMimic policies on other task setups and robots, we release software and policy checkpoints of our skill policies. 

**Abstract (ZH)**: 无需特定机器人演示即从大规模现有人机视频数据集中提炼有用的机器人技能策略：ZeroMimic系统 

---
# GenSwarm: Scalable Multi-Robot Code-Policy Generation and Deployment via Language Models 

**Title (ZH)**: GenSwarm: 通过语言模型实现可扩展的多机器人代码-策略生成与部署 

**Authors**: Wenkang Ji, Huaben Chen, Mingyang Chen, Guobin Zhu, Lufeng Xu, Roderich Groß, Rui Zhou, Ming Cao, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.23875)  

**Abstract**: The development of control policies for multi-robot systems traditionally follows a complex and labor-intensive process, often lacking the flexibility to adapt to dynamic tasks. This has motivated research on methods to automatically create control policies. However, these methods require iterative processes of manually crafting and refining objective functions, thereby prolonging the development cycle. This work introduces \textit{GenSwarm}, an end-to-end system that leverages large language models to automatically generate and deploy control policies for multi-robot tasks based on simple user instructions in natural language. As a multi-language-agent system, GenSwarm achieves zero-shot learning, enabling rapid adaptation to altered or unseen tasks. The white-box nature of the code policies ensures strong reproducibility and interpretability. With its scalable software and hardware architectures, GenSwarm supports efficient policy deployment on both simulated and real-world multi-robot systems, realizing an instruction-to-execution end-to-end functionality that could prove valuable for robotics specialists and non-specialists this http URL code of the proposed GenSwarm system is available online: this https URL. 

**Abstract (ZH)**: 多机器人系统的控制策略开发传统上是一个复杂且劳动密集型的过程，往往缺乏适应动态任务的灵活性。这激发了对能够自动创建控制策略方法的研究。然而，这些方法需要经过多次手动设计和优化目标函数的迭代过程，从而延长了开发周期。本文介绍了一种名为GenSwarm的端到端系统，该系统利用大型语言模型根据简单的自然语言用户指令自动生成并部署多机器人任务的控制策略。作为多语言代理系统，GenSwarm实现了零样本学习，能够快速适应更改或未见过的任务。代码的透明性保证了高度的可重复性和可解释性。凭借可扩展的软硬件架构，GenSwarm支持在模拟和真实世界多机器人系统中高效部署控制策略，实现了从指令到执行的端到端功能，对于机器人专家和非专家来说都可能具有价值。所提出的GenSwarm系统的代码可在以下网址获取：this https URL。 

---
# Disambiguate Gripper State in Grasp-Based Tasks: Pseudo-Tactile as Feedback Enables Pure Simulation Learning 

**Title (ZH)**: 基于抓取任务中指尖状态消歧：伪触觉反馈使纯仿真学习成为可能 

**Authors**: Yifei Yang, Lu Chen, Zherui Song, Yenan Chen, Wentao Sun, Zhongxiang Zhou, Rong Xiong, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23835)  

**Abstract**: Grasp-based manipulation tasks are fundamental to robots interacting with their environments, yet gripper state ambiguity significantly reduces the robustness of imitation learning policies for these tasks. Data-driven solutions face the challenge of high real-world data costs, while simulation data, despite its low costs, is limited by the sim-to-real gap. We identify the root cause of gripper state ambiguity as the lack of tactile feedback. To address this, we propose a novel approach employing pseudo-tactile as feedback, inspired by the idea of using a force-controlled gripper as a tactile sensor. This method enhances policy robustness without additional data collection and hardware involvement, while providing a noise-free binary gripper state observation for the policy and thus facilitating pure simulation learning to unleash the power of simulation. Experimental results across three real-world grasp-based tasks demonstrate the necessity, effectiveness, and efficiency of our approach. 

**Abstract (ZH)**: 基于抓取的 manipulation 任务是机器人与环境交互的基础，但由于夹持器状态的不确定性显著降低了这些任务的imitation learning策略的鲁棒性。数据驱动的方法面临现实世界数据成本高的挑战，而模拟数据虽然成本低，但也受限于模拟与现实之间的差距。我们识别夹持器状态不确定性根源为缺乏触觉反馈。为了解决这一问题，我们提出了一种新的方法，采用伪触觉作为反馈，灵感来源于将力控夹持器用作触觉传感器的想法。该方法在不需要额外数据收集和硬件投入的情况下增强策略的鲁棒性，为策略提供无噪声的二元夹持器状态观察，从而促进纯模拟学习，发挥模拟的作用。在三个实际抓取任务上的实验结果证明了该方法的必要性、有效性和效率。 

---
# Trajectory Planning for Automated Driving using Target Funnels 

**Title (ZH)**: 基于目标漏斗的自动驾驶轨迹规划 

**Authors**: Benjamin Bogenberger, Johannes Bürger, Vladislav Nenchev  

**Link**: [PDF](https://arxiv.org/pdf/2503.23795)  

**Abstract**: Self-driving vehicles rely on sensory input to monitor their surroundings and continuously adapt to the most likely future road course. Predictive trajectory planning is based on snapshots of the (uncertain) road course as a key input. Under noisy perception data, estimates of the road course can vary significantly, leading to indecisive and erratic steering behavior. To overcome this issue, this paper introduces a predictive trajectory planning algorithm with a novel objective function: instead of targeting a single reference trajectory based on the most likely road course, tracking a series of target reference sets, called a target funnel, is considered. The proposed planning algorithm integrates probabilistic information about the road course, and thus implicitly considers regular updates to road perception. Our solution is assessed in a case study using real driving data collected from a prototype vehicle. The results demonstrate that the algorithm maintains tracking accuracy and substantially reduces undesirable steering commands in the presence of noisy road perception, achieving a 56% reduction in input costs compared to a certainty equivalent formulation. 

**Abstract (ZH)**: 自动驾驶车辆依赖传感器输入监控环境，并不断适应最有可能的未来行驶路线。预测性轨迹规划基于不确定道路状况的快照作为关键输入。在噪声感知数据下，道路状况的估计会显著变化，导致不果断和不规则的转向行为。为克服这一问题，本文引入了一种具有新颖目标函数的预测性轨迹规划算法：而不是基于最有可能的道路状况针对单一参考轨迹进行规划，而是追踪一系列目标参考集，称为目标漏斗。所提出的规划算法整合了关于道路状况的概率信息，从而隐含地考虑了对道路感知的定期更新。我们的解决方案通过使用从原型车辆收集的真实驾驶数据进行案例研究进行了评估。结果表明，该算法在噪声感知道路条件下保持了跟踪准确性，并显著减少了不必要的转向命令，与等效确定性公式相比，输入成本减少了56%。 

---
# Towards a cognitive architecture to enable natural language interaction in co-constructive task learning 

**Title (ZH)**: 面向支持合作建构性任务学习的自然语言交互的认知架构 

**Authors**: Manuel Scheibl, Birte Richter, Alissa Müller, Michael Beetz, Britta Wrede  

**Link**: [PDF](https://arxiv.org/pdf/2503.23760)  

**Abstract**: This research addresses the question, which characteristics a cognitive architecture must have to leverage the benefits of natural language in Co-Constructive Task Learning (CCTL). To provide context, we first discuss Interactive Task Learning (ITL), the mechanisms of the human memory system, and the significance of natural language and multi-modality. Next, we examine the current state of cognitive architectures, analyzing their capabilities to inform a concept of CCTL grounded in multiple sources. We then integrate insights from various research domains to develop a unified framework. Finally, we conclude by identifying the remaining challenges and requirements necessary to achieve CCTL in Human-Robot Interaction (HRI). 

**Abstract (ZH)**: 本研究探讨了认知架构必须具备哪些特征以利用自然语言在联合建构性任务学习（CCTL）中的优势。首先，我们讨论了交互式任务学习（ITL）、人类记忆系统的机制以及自然语言和多模态的重要性。随后，我们分析了当前认知架构的能力，以构建一个基于多种来源的概念。接着，我们综合各研究领域的见解，开发了一个统一框架。最后，我们总结了实现人类-机器人交互（HRI）中CCTL仍需克服的挑战和要求。 

---
# Towards Benchmarking and Assessing the Safety and Robustness of Autonomous Driving on Safety-critical Scenarios 

**Title (ZH)**: Towards 评估自动驾驶在安全关键场景下的安全性和鲁棒性基准测试与评估 

**Authors**: Jingzheng Li, Xianglong Liu, Shikui Wei, Zhijun Chen, Bing Li, Qing Guo, Xianqi Yang, Yanjun Pu, Jiakai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23708)  

**Abstract**: Autonomous driving has made significant progress in both academia and industry, including performance improvements in perception task and the development of end-to-end autonomous driving systems. However, the safety and robustness assessment of autonomous driving has not received sufficient attention. Current evaluations of autonomous driving are typically conducted in natural driving scenarios. However, many accidents often occur in edge cases, also known as safety-critical scenarios. These safety-critical scenarios are difficult to collect, and there is currently no clear definition of what constitutes a safety-critical scenario. In this work, we explore the safety and robustness of autonomous driving in safety-critical scenarios. First, we provide a definition of safety-critical scenarios, including static traffic scenarios such as adversarial attack scenarios and natural distribution shifts, as well as dynamic traffic scenarios such as accident scenarios. Then, we develop an autonomous driving safety testing platform to comprehensively evaluate autonomous driving systems, encompassing not only the assessment of perception modules but also system-level evaluations. Our work systematically constructs a safety verification process for autonomous driving, providing technical support for the industry to establish standardized test framework and reduce risks in real-world road deployment. 

**Abstract (ZH)**: 自动驾驶在安全关键场景下的安全与 robustness 评估 

---
# Exploring GPT-4 for Robotic Agent Strategy with Real-Time State Feedback and a Reactive Behaviour Framework 

**Title (ZH)**: 探索GPT-4在实时状态反馈和反应性行为框架下的机器人代理策略 

**Authors**: Thomas O'Brien, Ysobel Sims  

**Link**: [PDF](https://arxiv.org/pdf/2503.23601)  

**Abstract**: We explore the use of GPT-4 on a humanoid robot in simulation and the real world as proof of concept of a novel large language model (LLM) driven behaviour method. LLMs have shown the ability to perform various tasks, including robotic agent behaviour. The problem involves prompting the LLM with a goal, and the LLM outputs the sub-tasks to complete to achieve that goal. Previous works focus on the executability and correctness of the LLM's generated tasks. We propose a method that successfully addresses practical concerns around safety, transitions between tasks, time horizons of tasks and state feedback. In our experiments we have found that our approach produces output for feasible requests that can be executed every time, with smooth transitions. User requests are achieved most of the time across a range of goal time horizons. 

**Abstract (ZH)**: 我们探索在仿真和真实世界中使用GPT-4驱动类人机器人行为的方法，作为新型大规模语言模型（LLM）驱动行为方法的概念验证。大规模语言模型显示出完成各种任务的能力，包括机器人代理行为。该问题涉及用目标提示LLM，并由LLM输出完成目标所需的子任务。以往的研究主要关注LLM生成任务的可执行性和正确性。我们提出了一种方法，有效解决了安全性、任务转换、任务的时间 horizons 以及状态反馈等实用问题。在我们的实验中，我们发现我们的方法每次都能生成可以执行的输出，并且过渡平滑。大多数情况下，用户请求在不同目标时间 horizons 的范围内都能够实现。 

---
# Can Visuo-motor Policies Benefit from Random Exploration Data? A Case Study on Stacking 

**Title (ZH)**: 视觉-运动策略能否从随机探索数据中受益？以堆叠为例 

**Authors**: Shutong Jin, Axel Kaliff, Ruiyu Wang, Muhammad Zahid, Florian T. Pokorny  

**Link**: [PDF](https://arxiv.org/pdf/2503.23571)  

**Abstract**: Human demonstrations have been key to recent advancements in robotic manipulation, but their scalability is hampered by the substantial cost of the required human labor. In this paper, we focus on random exploration data-video sequences and actions produced autonomously via motions to randomly sampled positions in the workspace-as an often overlooked resource for training visuo-motor policies in robotic manipulation. Within the scope of imitation learning, we examine random exploration data through two paradigms: (a) by investigating the use of random exploration video frames with three self-supervised learning objectives-reconstruction, contrastive, and distillation losses-and evaluating their applicability to visual pre-training; and (b) by analyzing random motor commands in the context of a staged learning framework to assess their effectiveness in autonomous data collection. Towards this goal, we present a large-scale experimental study based on over 750 hours of robot data collection, comprising 400 successful and 12,000 failed episodes. Our results indicate that: (a) among the three self-supervised learning objectives, contrastive loss appears most effective for visual pre-training while leveraging random exploration video frames; (b) data collected with random motor commands may play a crucial role in balancing the training data distribution and improving success rates in autonomous data collection within this study. The source code and dataset will be made publicly available at this https URL. 

**Abstract (ZH)**: 人类演示在 recent 的机器人操作进展中起到了关键作用，但其可扩展性受到所需人类劳动成本高昂的限制。本文focus了自主通过动作探索工作空间中随机位置产生的随机探索数据-视频序列和动作作为培训视觉-运动策略的一种往往被忽视的资源。在模仿学习框架下，我们通过两种范式来探讨随机探索数据：(a) 使用随机探索视频帧和三个自监督学习目标-重构、对比和蒸馏损失-来评估其在视觉预训练中的适用性；(b) 在分阶段学习框架的背景下分析随机运动命令的效果，以评估其在自主数据收集中的有效性。为此，我们基于超过750小时的机器人数据收集进行了一项大规模实验研究，包括400个成功的和12,000个失败的episode。我们的结果表明：(a) 在三个自监督学习目标中，对比损失在利用随机探索视频帧进行视觉预训练时最为有效；(b) 使用随机运动命令收集的数据可能在平衡训练数据分布和提高自主数据收集成功率方面发挥关键作用。该研究的源代码和数据集将在此处公开。 

---
# Improving Indoor Localization Accuracy by Using an Efficient Implicit Neural Map Representation 

**Title (ZH)**: 使用高效隐式神经地图表示提升室内定位准确性 

**Authors**: Haofei Kuang, Yue Pan, Xingguang Zhong, Louis Wiesmann, Jens Behley, Cyrill Stachniss  

**Link**: [PDF](https://arxiv.org/pdf/2503.23480)  

**Abstract**: Globally localizing a mobile robot in a known map is often a foundation for enabling robots to navigate and operate autonomously. In indoor environments, traditional Monte Carlo localization based on occupancy grid maps is considered the gold standard, but its accuracy is limited by the representation capabilities of the occupancy grid map. In this paper, we address the problem of building an effective map representation that allows to accurately perform probabilistic global localization. To this end, we propose an implicit neural map representation that is able to capture positional and directional geometric features from 2D LiDAR scans to efficiently represent the environment and learn a neural network that is able to predict both, the non-projective signed distance and a direction-aware projective distance for an arbitrary point in the mapped environment. This combination of neural map representation with a light-weight neural network allows us to design an efficient observation model within a conventional Monte Carlo localization framework for pose estimation of a robot in real time. We evaluated our approach to indoor localization on a publicly available dataset for global localization and the experimental results indicate that our approach is able to more accurately localize a mobile robot than other localization approaches employing occupancy or existing neural map representations. In contrast to other approaches employing an implicit neural map representation for 2D LiDAR localization, our approach allows to perform real-time pose tracking after convergence and near real-time global localization. The code of our approach is available at: this https URL. 

**Abstract (ZH)**: 基于隐式神经地图表示的室内全局定位有效地图表示研究 

---
# SparseLoc: Sparse Open-Set Landmark-based Global Localization for Autonomous Navigation 

**Title (ZH)**: SparseLoc: 稀疏开放集地标导向的全局定位方法用于自主导航 

**Authors**: Pranjal Paul, Vineeth Bhat, Tejas Salian, Mohammad Omama, Krishna Murthy Jatavallabhula, Naveen Arulselvan, K. Madhava Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2503.23465)  

**Abstract**: Global localization is a critical problem in autonomous navigation, enabling precise positioning without reliance on GPS. Modern global localization techniques often depend on dense LiDAR maps, which, while precise, require extensive storage and computational resources. Recent approaches have explored alternative methods, such as sparse maps and learned features, but they suffer from poor robustness and generalization. We propose SparseLoc, a global localization framework that leverages vision-language foundation models to generate sparse, semantic-topometric maps in a zero-shot manner. It combines this map representation with a Monte Carlo localization scheme enhanced by a novel late optimization strategy, ensuring improved pose estimation. By constructing compact yet highly discriminative maps and refining localization through a carefully designed optimization schedule, SparseLoc overcomes the limitations of existing techniques, offering a more efficient and robust solution for global localization. Our system achieves over a 5X improvement in localization accuracy compared to existing sparse mapping techniques. Despite utilizing only 1/500th of the points of dense mapping methods, it achieves comparable performance, maintaining an average global localization error below 5m and 2 degrees on KITTI sequences. 

**Abstract (ZH)**: 全球全局定位是自主导航中的一个关键问题，能够实现不依赖GPS的精确定位。现代全球全局定位技术通常依赖密集的激光雷达地图，虽然精确，但需要大量存储和计算资源。近年来的研究探索了替代方法，如稀疏地图和学习特征，但这些方法 robustness 和泛化能力较差。我们提出了 SparseLoc，这是一种利用视觉-语言基础模型生成稀疏语义-地形地图的零样本全局定位框架。该框架结合了增强的蒙特卡洛定位方案和一种新的后处理优化策略，确保了姿态估计的改进。通过构建紧凑且高度判别性的地图，并通过精心设计的优化计划进行定位细化，SparseLoc 克服了现有技术的局限性，提供了一种更具效率和 robust 性的全局定位解决方案。我们的系统在定位准确性方面相比现有稀疏映射技术提高了超过 5 倍。尽管仅使用密集映射方法的 1/500 个点，但在 KITTI 序列中实现了相近的性能，保持全局定位误差平均低于 5 米和 2 度。 

---
# Design and Experimental Validation of an Autonomous USV for Sensor Fusion-Based Navigation in GNSS-Denied Environments 

**Title (ZH)**: 基于传感器融合导航的自主USV设计与GNSS forbidden环境下的实验验证 

**Authors**: Samuel Cohen-Salmon, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2503.23445)  

**Abstract**: This paper presents the design, development, and experimental validation of MARVEL, an autonomous unmanned surface vehicle built for real-world testing of sensor fusion-based navigation algorithms in GNSS-denied environments. MARVEL was developed under strict constraints of cost-efficiency, portability, and seaworthiness, with the goal of creating a modular, accessible platform for high-frequency data acquisition and experimental learning. It integrates electromagnetic logs, Doppler velocity logs, inertial sensors, and real-time kinematic GNSS positioning. MARVEL enables real-time, in-situ validation of advanced navigation and AI-driven algorithms using redundant, synchronized sensors. Field experiments demonstrate the system's stability, maneuverability, and adaptability in challenging sea conditions. The platform offers a novel, scalable approach for researchers seeking affordable, open-ended tools to evaluate sensor fusion techniques under real-world maritime constraints. 

**Abstract (ZH)**: MARVEL：一种面向GNSS遮挡环境下的传感器融合导航算法实时验证的自主水面无人驾驶车辆设计与实验验证 

---
# VET: A Visual-Electronic Tactile System for Immersive Human-Machine Interaction 

**Title (ZH)**: 视觉-电子触觉系统：沉浸式人机交互系统 

**Authors**: Cong Zhang, Yisheng Yangm, Shilong Mu, Chuqiao Lyu, Shoujie Li, Xinyue Chai, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.23440)  

**Abstract**: In the pursuit of deeper immersion in human-machine interaction, achieving higher-dimensional tactile input and output on a single interface has become a key research focus. This study introduces the Visual-Electronic Tactile (VET) System, which builds upon vision-based tactile sensors (VBTS) and integrates electrical stimulation feedback to enable bidirectional tactile communication. We propose and implement a system framework that seamlessly integrates an electrical stimulation film with VBTS using a screen-printing preparation process, eliminating interference from traditional methods. While VBTS captures multi-dimensional input through visuotactile signals, electrical stimulation feedback directly stimulates neural pathways, preventing interference with visuotactile information. The potential of the VET system is demonstrated through experiments on finger electrical stimulation sensitivity zones, as well as applications in interactive gaming and robotic arm teleoperation. This system paves the way for new advancements in bidirectional tactile interaction and its broader applications. 

**Abstract (ZH)**: 基于视觉的电子触觉（VET）系统：实现单界面的高维触觉输入与输出 

---
# A Visual-Inertial Motion Prior SLAM for Dynamic Environments 

**Title (ZH)**: 动态环境下的视觉-惯性运动先验SLAM 

**Authors**: Weilong Sun, Yumin Zhang, Boren Wei  

**Link**: [PDF](https://arxiv.org/pdf/2503.23429)  

**Abstract**: The Visual-Inertial Simultaneous Localization and Mapping (VI-SLAM) algorithms which are mostly based on static assumption are widely used in fields such as robotics, UAVs, VR, and autonomous driving. To overcome the localization risks caused by dynamic landmarks in most VI-SLAM systems, a robust visual-inertial motion prior SLAM system, named (IDY-VINS), is proposed in this paper which effectively handles dynamic landmarks using inertial motion prior for dynamic environments to varying degrees. Specifically, potential dynamic landmarks are preprocessed during the feature tracking phase by the probabilistic model of landmarks' minimum projection errors which are obtained from inertial motion prior and epipolar constraint. Subsequently, a bundle adjustment (BA) residual is proposed considering the minimum projection error prior for dynamic candidate landmarks. This residual is integrated into a sliding window based nonlinear optimization process to estimate camera poses, IMU states and landmark positions while minimizing the impact of dynamic candidate landmarks that deviate from the motion prior. Finally, experimental results demonstrate that our proposed system outperforms state-of-the-art methods in terms of localization accuracy and time cost by robustly mitigating the influence of dynamic landmarks. 

**Abstract (ZH)**: 基于惯性动形势先的鲁棒视觉-惯性 simultanious localization and mapping (IDY-VINS) 系统 

---
# Proprioceptive multistable mechanical metamaterial via soft capacitive sensors 

**Title (ZH)**: proprioceptive多稳态机械 metamaterial 通过软电容传感器 

**Authors**: Hugo de Souza Oliveira, Niloofar Saeedzadeh Khaanghah, Martijn Oetelmans, Niko Münzenrieder, Edoardo Milana  

**Link**: [PDF](https://arxiv.org/pdf/2503.23389)  

**Abstract**: The technological transition from soft machines to soft robots necessarily passes through the integration of soft electronics and sensors. This allows for the establishment of feedback control systems while preserving the softness of the robot embodiment. Multistable mechanical metamaterials are excellent building blocks of soft machines, as their nonlinear response can be tuned by design to accomplish several functions. In this work, we present the integration of soft capacitive sensors in a multistable mechanical metamaterial, to enable proprioceptive sensing of state changes. The metamaterial is a periodic arrangement of 4 bistable unit cells. Each unit cell has an integrated capacitive sensor. Both the metastructure and the sensors are made of soft materials (TPU) and are 3D printed. Our preliminary results show that the capacitance variation of the sensors can be linked to state transitions of the metamaterial, by capturing the nonlinear deformation. 

**Abstract (ZH)**: 从软机器到软机器人技术过渡必然通过软电子和传感器的整合实现。这种整合使得能够在保持机器人本体柔软性的同时建立反馈控制系统。多稳态机械 metamaterial 是软机器的理想构建块，可以通过设计调整其非线性响应来实现多种功能。在这项工作中，我们展示了将软电容传感器集成到多稳态机械 metamaterial 中，以实现本体感受状态变化的能力。该 metamaterial 是由 4 个双稳态单元细胞的周期排列组成，每个单元细胞整合了一个电容传感器。元结构和传感器均由柔软材料（TPU）制成，并通过 3D 打印制造。初步结果表明，可以通过捕捉非线性变形将传感器的电容变化与 metamaterial 的状态转换联系起来。 

---
# Meta-Ori: monolithic meta-origami for nonlinear inflatable soft actuators 

**Title (ZH)**: 元纸艺：整体非线性可充气软执行器的元 origami 结构 

**Authors**: Hugo de Souza Oliveira, Xin Li, Johannes Frey, Edoardo Milana  

**Link**: [PDF](https://arxiv.org/pdf/2503.23375)  

**Abstract**: The nonlinear mechanical response of soft materials and slender structures is purposefully harnessed to program functions by design in soft robotic actuators, such as sequencing, amplified response, fast energy release, etc. However, typical designs of nonlinear actuators - e.g. balloons, inverted membranes, springs - have limited design parameters space and complex fabrication processes, hindering the achievement of more elaborated functions. Mechanical metamaterials, on the other hand, have very large design parameter spaces, which allow fine-tuning of nonlinear behaviours. In this work, we present a novel approach to fabricate nonlinear inflatables based on metamaterials and origami (Meta-Ori) as monolithic parts that can be fully 3D printed via Fused Deposition Modeling (FDM) using thermoplastic polyurethane (TPU) commercial filaments. Our design consists of a metamaterial shell with cylindrical topology and nonlinear mechanical response combined with a Kresling origami inflatable acting as a pneumatic transmitter. We develop and release a design tool in the visual programming language Grasshopper to interactively design our Meta-Ori. We characterize the mechanical response of the metashell and the origami, and the nonlinear pressure-volume curve of the Meta-Ori inflatable and, lastly, we demonstrate the actuation sequencing of a bi-segment monolithic Meta-Ori soft actuator. 

**Abstract (ZH)**: 基于机械超材料和 origami 的新型非线性可变形体的设计与制造：一种融合 FDM 3D 打印的技术 

---
# Physically Ground Commonsense Knowledge for Articulated Object Manipulation with Analytic Concepts 

**Title (ZH)**: 基于物理支撑常识 knowledge 的分段物体操作分析概念方法 

**Authors**: Jianhua Sun, Jiude Wei, Yuxuan Li, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23348)  

**Abstract**: We human rely on a wide range of commonsense knowledge to interact with an extensive number and categories of objects in the physical world. Likewise, such commonsense knowledge is also crucial for robots to successfully develop generalized object manipulation skills. While recent advancements in Large Language Models (LLM) have showcased their impressive capabilities in acquiring commonsense knowledge and conducting commonsense reasoning, effectively grounding this semantic-level knowledge produced by LLMs to the physical world to thoroughly guide robots in generalized articulated object manipulation remains a challenge that has not been sufficiently addressed. To this end, we introduce analytic concepts, procedurally defined upon mathematical symbolism that can be directly computed and simulated by machines. By leveraging the analytic concepts as a bridge between the semantic-level knowledge inferred by LLMs and the physical world where real robots operate, we are able to figure out the knowledge of object structure and functionality with physics-informed representations, and then use the physically grounded knowledge to instruct robot control policies for generalized, interpretable and accurate articulated object manipulation. Extensive experiments in both simulation and real-world environments demonstrate the superiority of our approach. 

**Abstract (ZH)**: 我们人类依赖广泛的知识库与物理世界中的各种对象进行交互。同样，这种常识性知识对于机器人成功发展通用对象操控技能也至关重要。尽管大型语言模型（LLM）的最新进展展示了它们在获取常识性知识和进行常识性推理方面的出色能力，但将由LLM生成的语义级知识有效地接地至物理世界，以彻底指导机器人进行通用的精细对象操控这一挑战尚未得到充分解决。为此，我们引入了分析概念，这些概念基于可被机器直接计算和模拟的数学符号定义。通过利用分析概念作为LLM推断出的语义级知识与真实机器人操作的物理世界之间的桥梁，我们可以利用基于物理的知识来理解和表示对象的结构和功能，并利用物理接地的知识来指导机器人的控制策略，实现通用、可解释和精确的精细对象操控。在仿真和真实环境中的广泛实验验证了我们方法的优势。 

---
# MagicGel: A Novel Visual-Based Tactile Sensor Design with MagneticGel 

**Title (ZH)**: MagicGel：一种基于视觉的新型磁性凝胶触觉传感器设计 

**Authors**: Jianhua Shan, Jie Zhao, Jiangduo Liu, Xiangbo Wang, Ziwei Xia, Guangyuan Xu, Bin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23345)  

**Abstract**: Force estimation is the core indicator for evaluating the performance of tactile sensors, and it is also the key technical path to achieve precise force feedback mechanisms. This study proposes a design method for a visual tactile sensor (VBTS) that integrates a magnetic perception mechanism, and develops a new tactile sensor called MagicGel. The sensor uses strong magnetic particles as markers and captures magnetic field changes in real time through Hall sensors. On this basis, MagicGel achieves the coordinated optimization of multimodal perception capabilities: it not only has fast response characteristics, but also can perceive non-contact status information of home electronic products. Specifically, MagicGel simultaneously analyzes the visual characteristics of magnetic particles and the multimodal data of changes in magnetic field intensity, ultimately improving force estimation capabilities. 

**Abstract (ZH)**: 基于磁感知机制的视觉触觉传感器（VBTS）设计方法及MagicGel触觉传感器的研究 

---
# Learning Coordinated Bimanual Manipulation Policies using State Diffusion and Inverse Dynamics Models 

**Title (ZH)**: 基于状态扩散和逆动力学模型的协调双臂操作策略学习 

**Authors**: Haonan Chen, Jiaming Xu, Lily Sheng, Tianchen Ji, Shuijing Liu, Yunzhu Li, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2503.23271)  

**Abstract**: When performing tasks like laundry, humans naturally coordinate both hands to manipulate objects and anticipate how their actions will change the state of the clothes. However, achieving such coordination in robotics remains challenging due to the need to model object movement, predict future states, and generate precise bimanual actions. In this work, we address these challenges by infusing the predictive nature of human manipulation strategies into robot imitation learning. Specifically, we disentangle task-related state transitions from agent-specific inverse dynamics modeling to enable effective bimanual coordination. Using a demonstration dataset, we train a diffusion model to predict future states given historical observations, envisioning how the scene evolves. Then, we use an inverse dynamics model to compute robot actions that achieve the predicted states. Our key insight is that modeling object movement can help learning policies for bimanual coordination manipulation tasks. Evaluating our framework across diverse simulation and real-world manipulation setups, including multimodal goal configurations, bimanual manipulation, deformable objects, and multi-object setups, we find that it consistently outperforms state-of-the-art state-to-action mapping policies. Our method demonstrates a remarkable capacity to navigate multimodal goal configurations and action distributions, maintain stability across different control modes, and synthesize a broader range of behaviors than those present in the demonstration dataset. 

**Abstract (ZH)**: 在洗衣等任务中，人类自然地协调双手操作物体并预判其动作将如何改变衣物状态。然而，在机器人中实现这种协调仍具有挑战，因为需要建模物体运动、预测未来状态并生成精确的双臂动作。在本工作中，我们通过将人类操作策略的预测性质融入到机器人的模仿学习中来应对这些挑战。具体地，我们分离任务相关的状态转换与代理特异性逆动力学建模，以实现有效的双臂协调。利用演示数据集，我们训练一个扩散模型来预测给定历史观察的未来状态，想象场景如何演变。然后，我们使用逆动力学模型来计算实现预测状态的机器人动作。我们的关键见解是，建模物体运动有助于学习双臂协调操作任务的策略。在各种仿真和真实世界操作设置中，包括多元目标配置、双臂操作、变形物体和多物体设置中评估我们的框架，我们发现它始终优于最先进的状态到动作映射策略。我们的方法展示了导航多元目标配置和动作分布、在不同控制模式下保持稳定并合成演示数据集中不存在的更广泛行为的显著能力。 

---
# Localized Graph-Based Neural Dynamics Models for Terrain Manipulation 

**Title (ZH)**: 基于局部图的神经动力学模型在地形操控中的应用 

**Authors**: Chaoqi Liu, Yunzhu Li, Kris Hauser  

**Link**: [PDF](https://arxiv.org/pdf/2503.23270)  

**Abstract**: Predictive models can be particularly helpful for robots to effectively manipulate terrains in construction sites and extraterrestrial surfaces. However, terrain state representations become extremely high-dimensional especially to capture fine-resolution details and when depth is unknown or unbounded. This paper introduces a learning-based approach for terrain dynamics modeling and manipulation, leveraging the Graph-based Neural Dynamics (GBND) framework to represent terrain deformation as motion of a graph of particles. Based on the principle that the moving portion of a terrain is usually localized, our approach builds a large terrain graph (potentially millions of particles) but only identifies a very small active subgraph (hundreds of particles) for predicting the outcomes of robot-terrain interaction. To minimize the size of the active subgraph we introduce a learning-based approach that identifies a small region of interest (RoI) based on the robot's control inputs and the current scene. We also introduce a novel domain boundary feature encoding that allows GBNDs to perform accurate dynamics prediction in the RoI interior while avoiding particle penetration through RoI boundaries. Our proposed method is both orders of magnitude faster than naive GBND and it achieves better overall prediction accuracy. We further evaluated our framework on excavation and shaping tasks on terrain with different granularity. 

**Abstract (ZH)**: 基于图的神经动力学学习驱动的地形建模与操控 

---
# Incorporating GNSS Information with LIDAR-Inertial Odometry for Accurate Land-Vehicle Localization 

**Title (ZH)**: 融合GNSS信息的LIDAR-惯性里程计定位方法及其在准确土地车辆定位中的应用 

**Authors**: Jintao Cheng, Bohuan Xue, Shiyang Chen, Qiuchi Xiang, Xiaoyu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23199)  

**Abstract**: Currently, visual odometry and LIDAR odometry are performing well in pose estimation in some typical environments, but they still cannot recover the localization state at high speed or reduce accumulated drifts. In order to solve these problems, we propose a novel LIDAR-based localization framework, which achieves high accuracy and provides robust localization in 3D pointcloud maps with information of multi-sensors. The system integrates global information with LIDAR-based odometry to optimize the localization state. To improve robustness and enable fast resumption of localization, this paper uses offline pointcloud maps for prior knowledge and presents a novel registration method to speed up the convergence rate. The algorithm is tested on various maps of different data sets and has higher robustness and accuracy than other localization algorithms. 

**Abstract (ZH)**: 基于LIDAR的新型高精度鲁棒定位框架 

---
# Deep Visual Servoing of an Aerial Robot Using Keypoint Feature Extraction 

**Title (ZH)**: 使用关键点特征提取的空中机器人深度视觉伺服控制 

**Authors**: Shayan Sepahvand, Niloufar Amiri, Farrokh Janabi-Sharifi  

**Link**: [PDF](https://arxiv.org/pdf/2503.23171)  

**Abstract**: The problem of image-based visual servoing (IBVS) of an aerial robot using deep-learning-based keypoint detection is addressed in this article. A monocular RGB camera mounted on the platform is utilized to collect the visual data. A convolutional neural network (CNN) is then employed to extract the features serving as the visual data for the servoing task. This paper contributes to the field by circumventing not only the challenge stemming from the need for man-made marker detection in conventional visual servoing techniques, but also enhancing the robustness against undesirable factors including occlusion, varying illumination, clutter, and background changes, thereby broadening the applicability of perception-guided motion control tasks in aerial robots. Additionally, extensive physics-based ROS Gazebo simulations are conducted to assess the effectiveness of this method, in contrast to many existing studies that rely solely on physics-less simulations. A demonstration video is available at this https URL. 

**Abstract (ZH)**: 基于深度学习关键点检测的 aerial 机器人图像视觉伺服问题研究 

---
# Dexterous Non-Prehensile Manipulation for Ungraspable Object via Extrinsic Dexterity 

**Title (ZH)**: 不可抓握物体的外在灵巧 manipulate 技术 

**Authors**: Yuhan Wang, Yu Li, Yaodong Yang, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23120)  

**Abstract**: Objects with large base areas become ungraspable when they exceed the end-effector's maximum aperture. Existing approaches address this limitation through extrinsic dexterity, which exploits environmental features for non-prehensile manipulation. While grippers have shown some success in this domain, dexterous hands offer superior flexibility and manipulation capabilities that enable richer environmental interactions, though they present greater control challenges. Here we present ExDex, a dexterous arm-hand system that leverages reinforcement learning to enable non-prehensile manipulation for grasping ungraspable objects. Our system learns two strategic manipulation sequences: relocating objects from table centers to edges for direct grasping, or to walls where extrinsic dexterity enables grasping through environmental interaction. We validate our approach through extensive experiments with dozens of diverse household objects, demonstrating both superior performance and generalization capabilities with novel objects. Furthermore, we successfully transfer the learned policies from simulation to a real-world robot system without additional training, further demonstrating its applicability in real-world scenarios. Project website: this https URL. 

**Abstract (ZH)**: 一种利用强化学习实现非抱握操作的灵巧臂手系统：ExDex 

---
# Microscopic Robots That Sense, Think, Act, and Compute 

**Title (ZH)**: 感知、思考、行动与计算的微纳米机器人 

**Authors**: Maya M. Lassiter, Jungho Lee, Kyle Skelil, Li Xu, Lucas Hanson, William H. Reinhardt, Dennis Sylvester, Mark Yim, David Blaauw, Marc Z. Miskin  

**Link**: [PDF](https://arxiv.org/pdf/2503.23085)  

**Abstract**: While miniaturization has been a goal in robotics for nearly 40 years, roboticists have struggled to access sub-millimeter dimensions without making sacrifices to on-board information processing due to the unique physics of the microscale. Consequently, microrobots often lack the key features that distinguish their macroscopic cousins from other machines, namely on-robot systems for decision making, sensing, feedback, and programmable computation. Here, we take up the challenge of building a microrobot comparable in size to a single-celled paramecium that can sense, think, and act using onboard systems for computation, sensing, memory, locomotion, and communication. Built massively in parallel with fully lithographic processing, these microrobots can execute digitally defined algorithms and autonomously change behavior in response to their surroundings. Combined, these results pave the way for general purpose microrobots that can be programmed many times in a simple setup, cost under $0.01 per machine, and work together to carry out tasks without supervision in uncertain environments. 

**Abstract (ZH)**: 微型机器人：构建能够在亚毫米尺度上感知、思考和行动的自主计算微型机器人 

---
# VLM-C4L: Continual Core Dataset Learning with Corner Case Optimization via Vision-Language Models for Autonomous Driving 

**Title (ZH)**: VLM-C4L：通过视觉语言模型在自动驾驶中基于边缘案例的持续核心数据集学习优化 

**Authors**: Haibo Hu, Jiacheng Zuo, Yang Lou, Yufei Cui, Jianping Wang, Nan Guan, Jin Wang, Yung-Hui Li, Chun Jason Xue  

**Link**: [PDF](https://arxiv.org/pdf/2503.23046)  

**Abstract**: With the widespread adoption and deployment of autonomous driving, handling complex environments has become an unavoidable challenge. Due to the scarcity and diversity of extreme scenario datasets, current autonomous driving models struggle to effectively manage corner cases. This limitation poses a significant safety risk, according to the National Highway Traffic Safety Administration (NHTSA), autonomous vehicle systems have been involved in hundreds of reported crashes annually in the United States, occurred in corner cases like sun glare and fog, which caused a few fatal accident. Furthermore, in order to consistently maintain a robust and reliable autonomous driving system, it is essential for models not only to perform well on routine scenarios but also to adapt to newly emerging scenarios, especially those corner cases that deviate from the norm. This requires a learning mechanism that incrementally integrates new knowledge without degrading previously acquired capabilities. However, to the best of our knowledge, no existing continual learning methods have been proposed to ensure consistent and scalable corner case learning in autonomous driving. To address these limitations, we propose VLM-C4L, a continual learning framework that introduces Vision-Language Models (VLMs) to dynamically optimize and enhance corner case datasets, and VLM-C4L combines VLM-guided high-quality data extraction with a core data replay strategy, enabling the model to incrementally learn from diverse corner cases while preserving performance on previously routine scenarios, thus ensuring long-term stability and adaptability in real-world autonomous driving. We evaluate VLM-C4L on large-scale real-world autonomous driving datasets, including Waymo and the corner case dataset CODA. 

**Abstract (ZH)**: 基于视觉语言模型的持续学习框架VLM-C4L：面向自动驾驶的复杂corner case优化与适应 

---
# Distortion Bounds of Subdivision Models for SO(3) 

**Title (ZH)**: SO(3)中分拆模型的失真界 

**Authors**: Zhaoqi Zhang, Chee Yap  

**Link**: [PDF](https://arxiv.org/pdf/2503.22961)  

**Abstract**: In the subdivision approach to robot path planning, we need to subdivide the configuration space of a robot into nice cells to perform various computations. For a rigid spatial robot, this configuration space is $SE(3)=\mathbb{R}^3\times SO(3)$. The subdivision of $\mathbb{R}^3$ is standard but so far, there are no global subdivision schemes for $SO(3)$. We recently introduced a representation for $SO(3)$ suitable for subdivision. This paper investigates the distortion of the natural metric on $SO(3)$ caused by our representation. The proper framework for this study lies in the Riemannian geometry of $SO(3)$, enabling us to obtain sharp distortion bounds. 

**Abstract (ZH)**: 在机器人路径规划的细分方法中，我们需要将机器人的配置空间细分成交互良好的单元以进行各种计算。对于刚体空间机器人，该配置空间为$SE(3)=\mathbb{R}^3\times SO(3)$。虽然$\mathbb{R}^3$的细分是标准的，但目前仍没有$SO(3)$的全局细分方案。我们最近引入了一种适合细分的$SO(3)$表示。本文研究了我们表示对$SO(3)$上自然度量引起的失真。这项研究的适当框架是$SO(3)$的黎曼几何，使我们能够获得精确的失真界。 

---
# Towards Mobile Sensing with Event Cameras on High-mobility Resource-constrained Devices: A Survey 

**Title (ZH)**: 面向高机动性资源受限设备的事件 cameras 无线感测：一个综述 

**Authors**: Haoyang Wang, Ruishan Guo, Pengtao Ma, Ciyu Ruan, Xinyu Luo, Wenhua Ding, Tianyang Zhong, Jingao Xu, Yunhao Liu, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22943)  

**Abstract**: With the increasing complexity of mobile device applications, these devices are evolving toward high mobility. This shift imposes new demands on mobile sensing, particularly in terms of achieving high accuracy and low latency. Event-based vision has emerged as a disruptive paradigm, offering high temporal resolution, low latency, and energy efficiency, making it well-suited for high-accuracy and low-latency sensing tasks on high-mobility platforms. However, the presence of substantial noisy events, the lack of inherent semantic information, and the large data volume pose significant challenges for event-based data processing on resource-constrained mobile devices. This paper surveys the literature over the period 2014-2024, provides a comprehensive overview of event-based mobile sensing systems, covering fundamental principles, event abstraction methods, algorithmic advancements, hardware and software acceleration strategies. We also discuss key applications of event cameras in mobile sensing, including visual odometry, object tracking, optical flow estimation, and 3D reconstruction, while highlighting the challenges associated with event data processing, sensor fusion, and real-time deployment. Furthermore, we outline future research directions, such as improving event camera hardware with advanced optics, leveraging neuromorphic computing for efficient processing, and integrating bio-inspired algorithms to enhance perception. To support ongoing research, we provide an open-source \textit{Online Sheet} with curated resources and recent developments. We hope this survey serves as a valuable reference, facilitating the adoption of event-based vision across diverse applications. 

**Abstract (ZH)**: 随着移动设备应用程序复杂性的增加，这些设备正朝着高移动性发展。这一变化对移动传感提出了新的要求，特别是在实现高精度和低延迟方面。基于事件的视觉已经作为一种颠覆性范式出现，提供高时间分辨率、低延迟和能量效率，使其非常适合在高移动性平台上执行高精度和低延迟的传感任务。然而，大量嘈杂事件的存在、缺乏内在语义信息以及大数据量给资源受限的移动设备上的事件数据处理带来了重大挑战。本文回顾了2014-2024年的相关文献，提供了基于事件的移动传感系统的全面概述，涵盖基本原理、事件抽象方法、算法进展、硬件和软件加速策略。我们还讨论了事件摄像头在移动传感中的关键应用，包括视觉里程计、目标跟踪、光流估计和三维重建，同时指出了事件数据处理、传感器融合和实时部署相关的挑战。此外，我们提出了未来研究方向，如改进具有先进光学技术的事件摄像头硬件、利用神经形态计算进行高效处理以及整合生物启发算法来提升感知能力。为了支持持续研究，我们提供了包含精选资源和最新发展的开源《在线表格》。我们希望本文献能作为有价值的参考，促进基于事件的视觉技术在多种应用中的采用。 

---
# Adaptive Interactive Navigation of Quadruped Robots using Large Language Models 

**Title (ZH)**: 使用大型语言模型的四足机器人自适应交互导航 

**Authors**: Kangjie Zhou, Yao Mu, Haoyang Song, Yi Zeng, Pengying Wu, Han Gao, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22942)  

**Abstract**: Robotic navigation in complex environments remains a critical research challenge. Traditional navigation methods focus on optimal trajectory generation within free space, struggling in environments lacking viable paths to the goal, such as disaster zones or cluttered warehouses. To address this gap, we propose an adaptive interactive navigation approach that proactively interacts with environments to create feasible paths to reach originally unavailable goals. Specifically, we present a primitive tree for task planning with large language models (LLMs), facilitating effective reasoning to determine interaction objects and sequences. To ensure robust subtask execution, we adopt reinforcement learning to pre-train a comprehensive skill library containing versatile locomotion and interaction behaviors for motion planning. Furthermore, we introduce an adaptive replanning method featuring two LLM-based modules: an advisor serving as a flexible replanning trigger and an arborist for autonomous plan adjustment. Integrated with the tree structure, the replanning mechanism allows for convenient node addition and pruning, enabling rapid plan modification in unknown environments. Comprehensive simulations and experiments have demonstrated our method's effectiveness and adaptivity in diverse scenarios. The supplementary video is available at page: this https URL. 

**Abstract (ZH)**: 复杂环境中机器人的导航仍然是一个关键的研究挑战。传统的导航方法专注于在自由空间内生成最优轨迹，难以应对缺乏可行路径到目标的环境，如灾难现场或杂物仓库。为解决这一问题，我们提出了一种主动交互导航方法，能在环境中主动交互以创建通往原本不可达目标的可行路径。具体地，我们使用大规模语言模型（LLMs）为任务规划提供了一种基础树结构，促进有效推理以确定交互对象和顺序。为确保子任务执行的鲁棒性，我们采用了强化学习预先训练了一个包含多种运动和交互行为的技能库，用于运动规划。此外，我们引入了一种基于大规模语言模型（LLMs）的自适应重规划方法，包括一个作为灵活重规划触发器的顾问模块和一个自主计划调整的树干模块。该重规划机制与树结构结合，允许方便地添加和修剪节点，从而在未知环境中快速修改计划。全面的仿真实验表明，该方法在多种场景中具有有效性与适应性。补充视频可在以下链接获取：this https URL。 

---
# SR-LIO++: Efficient LiDAR-Inertial Odometry and Quantized Mapping with Sweep Reconstruction 

**Title (ZH)**: SR-LIO++: 高效的LiDAR-惯性里程计和基于扫掠重建的量化映射 

**Authors**: Zikang Yuan, Ruiye Ming, Chengwei Zhao, Yonghao Tan, Pingcheng Dong, Hongcheng Luo, Yuzhong Jiao, Xin Yang, Kwang-Ting Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2503.22926)  

**Abstract**: Addressing the inherent low acquisition frequency limitation of 3D LiDAR to achieve high-frequency output has become a critical research focus in the LiDAR-Inertial Odometry (LIO) domain. To ensure real-time performance, frequency-enhanced LIO systems must process each sweep within significantly reduced timeframe, which presents substantial challenges for deployment on low-computational-power platforms. To address these limitations, we introduce SR-LIO++, an innovative LIO system capable of achieving doubled output frequency relative to input frequency on resource-constrained hardware platforms, including the Raspberry Pi 4B. Our system employs a sweep reconstruction methodology to enhance LiDAR sweep frequency, generating high-frequency reconstructed sweeps. Building upon this foundation, we propose a caching mechanism for intermediate results (i.e., surface parameters) of the most recent segments, effectively minimizing redundant processing of common segments in adjacent reconstructed sweeps. This method decouples processing time from the traditionally linear dependence on reconstructed sweep frequency. Furthermore, we present a quantized map point management based on index table mapping, significantly reducing memory usage by converting global 3D point storage from 64-bit double precision to 8-bit char representation. This method also converts the computationally intensive Euclidean distance calculations in nearest neighbor searches from 64-bit double precision to 16-bit short and 32-bit integer formats, significantly reducing both memory and computational cost. Extensive experimental evaluations across three distinct computing platforms and four public datasets demonstrate that SR-LIO++ maintains state-of-the-art accuracy while substantially enhancing efficiency. Notably, our system successfully achieves 20Hz state output on Raspberry Pi 4B hardware. 

**Abstract (ZH)**: 基于3D LiDAR固有的低获取频率限制，实现高频率输出已成为LiDAR-惯性里程计（LIO）领域的关键研究重点。为了确保实时性能，高频率增强的LIO系统必须在显著减少的时间框架内处理每个扫描，这对低计算能力平台上部署提出了巨大挑战。为了解决这些限制，我们引入了SR-LIO++，这是一种能够在资源受限硬件平台上（包括Raspberry Pi 4B）将输出频率相对输入频率提高一倍的创新LIO系统。我们的系统采用扫描重建方法来增强LiDAR扫描频率，生成高频率的重建扫描。在此基础上，我们提出了一种中间结果（即表面参数）缓存机制，有效减少了相邻重建扫描中常见段落的冗余处理，从而解耦处理时间与传统上与重建扫描频率呈线性关系的依赖性。此外，我们提出了一种基于索引表映射的姿态点管理量化方法，通过将全局三维点存储从64位双精度转换为8位字符表示，显著减少了内存使用量。这种方法还将最近邻搜索中的计算密集型欧几里得距离计算从64位双精度转换为16位短整数和32位整数格式，显著减少了内存和计算成本。通过对三个不同的计算平台和四个公开数据集进行广泛的实验评估，证明SR-LIO++在保持最先进的精度的同时显著提高了效率。值得注意的是，我们的系统在Raspberry Pi 4B硬件上成功实现了20Hz状态输出。 

---
# Predictive Traffic Rule Compliance using Reinforcement Learning 

**Title (ZH)**: 基于强化学习的预测性交通规则遵守研究 

**Authors**: Yanliang Huang, Sebastian Mair, Zhuoqi Zeng, Amr Alanwar, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2503.22925)  

**Abstract**: Autonomous vehicle path planning has reached a stage where safety and regulatory compliance are crucial. This paper presents a new approach that integrates a motion planner with a deep reinforcement learning model to predict potential traffic rule violations. In this setup, the predictions of the critic directly affect the cost function of the motion planner, guiding the choices of the trajectory. We incorporate key interstate rules from the German Road Traffic Regulation into a rule book and use a graph-based state representation to handle complex traffic information. Our main innovation is replacing the standard actor network in an actor-critic setup with a motion planning module, which ensures both predictable trajectory generation and prevention of long-term rule violations. Experiments on an open German highway dataset show that the model can predict and prevent traffic rule violations beyond the planning horizon, significantly increasing safety in challenging traffic conditions. 

**Abstract (ZH)**: 自主驾驶车辆路径规划已达到一个关键阶段，安全性和法规遵从性至关重要。本文提出了一种新的方法，将运动规划器与深度强化学习模型集成，以预测潜在的交通规则违规行为。在这种设置中，评论家的预测直接影响运动规划器的成本函数，指导轨迹的选择。我们将德国道路交通法规中的关键跨州规则纳入规则书中，并使用基于图的状态表示来处理复杂交通信息。我们的主要创新之处在于，在演员-评论家架构中用运动规划模块替换标准的演员网络，从而确保轨迹生成的可预测性和长期规则违规的预防。在开放的德国高速公路数据集上的实验表明，该模型可以预测并防止超出规划范围的交通规则违规行为，在复杂交通条件下显著提高安全性。 

---
# LiDAR-based Quadrotor Autonomous Inspection System in Cluttered Environments 

**Title (ZH)**: 基于LiDAR的四旋翼无人机复杂环境自主巡检系统 

**Authors**: Wenyi Liu, Huajie Wu, Liuyu Shi, Fangcheng Zhu, Yuying Zou, Fanze Kong, Fu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22921)  

**Abstract**: In recent years, autonomous unmanned aerial vehicle (UAV) technology has seen rapid advancements, significantly improving operational efficiency and mitigating risks associated with manual tasks in domains such as industrial inspection, agricultural monitoring, and search-and-rescue missions. Despite these developments, existing UAV inspection systems encounter two critical challenges: limited reliability in complex, unstructured, and GNSS-denied environments, and a pronounced dependency on skilled operators. To overcome these limitations, this study presents a LiDAR-based UAV inspection system employing a dual-phase workflow: human-in-the-loop inspection and autonomous inspection. During the human-in-the-loop phase, untrained pilots are supported by autonomous obstacle avoidance, enabling them to generate 3D maps, specify inspection points, and schedule tasks. Inspection points are then optimized using the Traveling Salesman Problem (TSP) to create efficient task sequences. In the autonomous phase, the quadrotor autonomously executes the planned tasks, ensuring safe and efficient data acquisition. Comprehensive field experiments conducted in various environments, including slopes, landslides, agricultural fields, factories, and forests, confirm the system's reliability and flexibility. Results reveal significant enhancements in inspection efficiency, with autonomous operations reducing trajectory length by up to 40\% and flight time by 57\% compared to human-in-the-loop operations. These findings underscore the potential of the proposed system to enhance UAV-based inspections in safety-critical and resource-constrained scenarios. 

**Abstract (ZH)**: 近年来，自主无人机（UAV）技术取得了 rapid advancements，显著提高了工业检测、农业监测和搜救任务等领域的操作效率，并减少了与手动任务相关的风险。尽管取得了这些进展，现有的无人机检测系统仍面临两大关键挑战：在复杂、未结构化和GPS拒止环境中的有限可靠性，以及对熟练操作员的明显依赖。为克服这些限制，本研究提出了一种基于LiDAR的无人机检测系统，采用双阶段工作流：人工在环检测和自主检测。在人工在环阶段，未受过训练的飞行员通过自主障碍回避的支持，生成3D地图、指定检测点并安排任务。然后使用旅行商问题（TSP）优化检测点，以创建高效的任务序列。在自主阶段，四旋翼机自主执行计划任务，确保安全和高效的数据采集。在各种环境中（包括斜坡、滑坡、农业用地、工厂和森林）进行的综合实地试验证实了系统的可靠性和灵活性。结果表明，在检测效率方面有显著提升，与人工在环操作相比，自主操作可将轨迹长度减少40%以上，飞行时间减少57%。这些发现突显了所提出系统在安全关键和资源受限场景中增强无人机检测的潜力。 

---
# VizFlyt: Perception-centric Pedagogical Framework For Autonomous Aerial Robots 

**Title (ZH)**: VizFlyt: 以感知为中心的自主无人机教学框架 

**Authors**: Kushagra Srivastava, Rutwik Kulkarni, Manoj Velmurugan, Nitin J. Sanket  

**Link**: [PDF](https://arxiv.org/pdf/2503.22876)  

**Abstract**: Autonomous aerial robots are becoming commonplace in our lives. Hands-on aerial robotics courses are pivotal in training the next-generation workforce to meet the growing market demands. Such an efficient and compelling course depends on a reliable testbed. In this paper, we present \textit{VizFlyt}, an open-source perception-centric Hardware-In-The-Loop (HITL) photorealistic testing framework for aerial robotics courses. We utilize pose from an external localization system to hallucinate real-time and photorealistic visual sensors using 3D Gaussian Splatting. This enables stress-free testing of autonomy algorithms on aerial robots without the risk of crashing into obstacles. We achieve over 100Hz of system update rate. Lastly, we build upon our past experiences of offering hands-on aerial robotics courses and propose a new open-source and open-hardware curriculum based on \textit{VizFlyt} for the future. We test our framework on various course projects in real-world HITL experiments and present the results showing the efficacy of such a system and its large potential use cases. Code, datasets, hardware guides and demo videos are available at this https URL 

**Abstract (ZH)**: 自主飞行机器人越来越多地融入我们的生活。动手飞行机器人课程是培养下一代劳动力以满足不断增长的市场需求的关键。这样高效且引人入胜的课程依赖于一个可靠的测试平台。本文介绍了VizFlyt，一个开源感知中心的硬件在环（HITL）光现实测试框架，用于飞行机器人课程。我们使用外部定位系统的姿态来实时和光现实地生成视觉传感器，利用3D正态斑点图。这使得在飞行机器人上无压力地测试自主算法，而不用担心碰撞到障碍物。我们实现了超过100Hz的系统更新率。最后，我们基于VizFlyt并结合我们过往提供动手飞行机器人课程的经验，提出了一种新的开源和开源硬件课程。我们对各种课程项目在现实世界的HITL实验中测试了该框架，并展示了该系统的有效性及其广泛的应用前景。相关代码、数据集、硬件指南和演示视频请访问此链接。 

---
# A reduced-scale autonomous morphing vehicle prototype with enhanced aerodynamic efficiency 

**Title (ZH)**: 一种增强气动效率的缩小比例自主形态变化车辆原型 

**Authors**: Peng Zhang, Branson Blaylock  

**Link**: [PDF](https://arxiv.org/pdf/2503.22777)  

**Abstract**: Road vehicles contribute to significant levels of greenhouse gas (GHG) emissions. A potential strategy for improving their aerodynamic efficiency and reducing emissions is through active adaptation of their exterior shapes to the aerodynamic environment. In this study, we present a reduced-scale morphing vehicle prototype capable of actively interacting with the aerodynamic environment to enhance fuel economy. Morphing is accomplished by retrofitting a deformable structure actively actuated by built-in motors. The morphing vehicle prototype is integrated with an optimization algorithm that can autonomously identify the structural shape that minimizes aerodynamic drag. The performance of the morphing vehicle prototype is investigated through an extensive experimental campaign in a large-scale wind tunnel facility. The autonomous optimization algorithm identifies an optimal morphing shape that can elicit an 8.5% reduction in the mean drag force. Our experiments provide a comprehensive dataset that validates the efficiency of shape morphing, demonstrating a clear and consistent decrease in the drag force as the vehicle transitions from a suboptimal to the optimal shape. Insights gained from experiments on scaled-down models provide valuable guidelines for the design of full-size morphing vehicles, which could lead to appreciable energy savings and reductions in GHG emissions. This study highlights the feasibility and benefits of real-time shape morphing under conditions representative of realistic road environments, paving the way for the realization of full-scale morphing vehicles with enhanced aerodynamic efficiency and reduced GHG emissions. 

**Abstract (ZH)**: 基于主动形态变化的车辆在典型道路环境下的减阻研究及其对能源节约和温室气体减排的潜在影响 

---
# Co-design of materials, structures and stimuli for magnetic soft robots with large deformation and dynamic contacts 

**Title (ZH)**: 磁软机器人中大变形和动态接触的材料、结构与刺激共设计 

**Authors**: Liwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.22767)  

**Abstract**: Magnetic soft robots embedded with hard magnetic particles enable untethered actuation via external magnetic fields, offering remote, rapid, and precise control, which is highly promising for biomedical applications. However, designing such systems is challenging due to the complex interplay of magneto-elastic dynamics, large deformation, solid contacts, time-varying stimuli, and posture-dependent loading. As a result, most existing research relies on heuristics and trial-and-error methods or focuses on the independent design of stimuli or structures under static conditions. We propose a topology optimization framework for magnetic soft robots that simultaneously designs structures, location-specific material magnetization and time-varying magnetic stimuli, accounting for large deformations, dynamic motion, and solid contacts. This is achieved by integrating generalized topology optimization with the magneto-elastic material point method, which supports GPU-accelerated parallel simulations and auto-differentiation for sensitivity analysis. We applied this framework to design magnetic robots for various tasks, including multi-task shape morphing and locomotion, in both 2D and 3D. The method autonomously generates optimized robotic systems to achieve target behaviors without requiring human intervention. Despite the nonlinear physics and large design space, it demonstrates exceptional efficiency, completing all cases within minutes. This proposed framework represents a significant step toward the automatic co-design of magnetic soft robots for applications such as metasurfaces, drug delivery, and minimally invasive procedures. 

**Abstract (ZH)**: 嵌入硬磁颗粒的磁软机器人通过外部磁场实现无缆驱动，提供远程、快速、精确控制，极具生物医学应用前景。然而，由于磁弹性动力学、大变形、固体接触、时间变化刺激和姿态依赖载荷的复杂相互作用，设计此类系统具有挑战性。因此，现有大多数研究依赖于经验方法或仅在静态条件下独立设计刺激或结构。我们提出了一种拓扑优化框架，用于同时设计磁软机器人的结构、位置特定的材料磁化和时间变化的磁场刺激，考虑大变形、动态运动和固体接触。通过将广义拓扑优化与磁弹性物质点法结合，该框架支持GPU加速并行仿真和自动求导以进行灵敏度分析。我们将此框架应用于设计用于各种任务的磁驱动机器人，包括2D和3D环境下的多任务形状变形和移动。该方法能自主生成优化的机器人系统以实现目标行为，无需人为干预。尽管涉及非线性物理和大的设计空间，该方法表现出色，能在几分钟内完成所有案例。该提议框架代表了自动协同设计磁软机器人以应用于超表面、药物递送和微创手术等应用的重要步骤。 

---
# Strategies for decentralised UAV-based collisions monitoring in rugby 

**Title (ZH)**: 基于橄榄球的去中心化无人机碰撞监测策略 

**Authors**: Yu Cheng, Harun Šiljak  

**Link**: [PDF](https://arxiv.org/pdf/2503.22757)  

**Abstract**: Recent advancements in unmanned aerial vehicle (UAV) technology have opened new avenues for dynamic data collection in challenging environments, such as sports fields during fast-paced sports action. For the purposes of monitoring sport events for dangerous injuries, we envision a coordinated UAV fleet designed to capture high-quality, multi-view video footage of collision events in real-time. The extracted video data is crucial for analyzing athletes' motions and investigating the probability of sports-related traumatic brain injuries (TBI) during impacts. This research implemented a UAV fleet system on the NetLogo platform, utilizing custom collision detection algorithms to compare against traditional TV-coverage strategies. Our system supports decentralized data capture and autonomous processing, providing resilience in the rapidly evolving dynamics of sports collisions.
The collaboration algorithm integrates both shared and local data to generate multi-step analyses aimed at determining the efficacy of custom methods in enhancing the accuracy of TBI prediction models. Missions are simulated in real-time within a two-dimensional model, focusing on the strategic capture of collision events that could lead to TBI, while considering operational constraints such as rapid UAV maneuvering and optimal positioning. Preliminary results from the NetLogo simulations suggest that custom collision detection methods offer superior performance over standard TV-coverage strategies by enabling more precise and timely data capture. This comparative analysis highlights the advantages of tailored algorithmic approaches in critical sports safety applications. 

**Abstract (ZH)**: 近期无人机（UAV）技术的进步为在快节奏体育比赛中采集动态数据开辟了新的途径，特别是在体育场地上。为了监测体育赛事中的危险伤害，我们设想了一套协同无人机舰队，旨在实时捕捉碰撞事件的高质量多视角视频。提取的视频数据对于分析运动员的动作并调查运动相关创伤性脑损伤（TBI）的概率至关重要。该研究在NetLogo平台上实施了一个无人机舰队系统，利用定制的碰撞检测算法与传统的电视直播策略进行对比。该系统支持分散的数据采集和自主处理，提供了在体育碰撞动态快速演变中的弹性。合作算法结合共享和本地数据，生成多步分析，以确定自定义方法在提高TBI预测模型准确性方面的有效性。研究在二维模型中实时模拟任务，集中在战略捕捉可能导致TBI的碰撞事件上，同时考虑操作约束，如快速无人机机动和最佳定位。NetLogo模拟的初步结果显示，自定义碰撞检测方法在数据捕获的精确性和及时性方面优于标准的电视直播策略，这突显了定制算法方法在关键体育安全应用中的优势。 

---
# UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving 

**Title (ZH)**: UniOcc：自主驾驶中 occupancy 预测和估计统一基准 

**Authors**: Yuping Wang, Xiangyu Huang, Xiaokang Sun, Mingxuan Yan, Shuo Xing, Zhengzhong Tu, Jiachen Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.24381)  

**Abstract**: We introduce UniOcc, a comprehensive, unified benchmark for occupancy forecasting (i.e., predicting future occupancies based on historical information) and current-frame occupancy prediction from camera images. UniOcc unifies data from multiple real-world datasets (i.e., nuScenes, Waymo) and high-fidelity driving simulators (i.e., CARLA, OpenCOOD), which provides 2D/3D occupancy labels with per-voxel flow annotations and support for cooperative autonomous driving. In terms of evaluation, unlike existing studies that rely on suboptimal pseudo labels for evaluation, UniOcc incorporates novel metrics that do not depend on ground-truth occupancy, enabling robust assessment of additional aspects of occupancy quality. Through extensive experiments on state-of-the-art models, we demonstrate that large-scale, diverse training data and explicit flow information significantly enhance occupancy prediction and forecasting performance. 

**Abstract (ZH)**: UniOcc：Occupancy 预测的综合统一基准 

---
# Reinforcement Learning for Safe Autonomous Two Device Navigation of Cerebral Vessels in Mechanical Thrombectomy 

**Title (ZH)**: 基于强化学习的医源性脑血管安全自主双设备导航技术在机械溶栓中的应用 

**Authors**: Harry Robertshaw, Benjamin Jackson, Jiaheng Wang, Hadi Sadati, Lennart Karstensen, Alejandro Granados, Thomas C Booth  

**Link**: [PDF](https://arxiv.org/pdf/2503.24140)  

**Abstract**: Purpose: Autonomous systems in mechanical thrombectomy (MT) hold promise for reducing procedure times, minimizing radiation exposure, and enhancing patient safety. However, current reinforcement learning (RL) methods only reach the carotid arteries, are not generalizable to other patient vasculatures, and do not consider safety. We propose a safe dual-device RL algorithm that can navigate beyond the carotid arteries to cerebral vessels.
Methods: We used the Simulation Open Framework Architecture to represent the intricacies of cerebral vessels, and a modified Soft Actor-Critic RL algorithm to learn, for the first time, the navigation of micro-catheters and micro-guidewires. We incorporate patient safety metrics into our reward function by integrating guidewire tip forces. Inverse RL is used with demonstrator data on 12 patient-specific vascular cases.
Results: Our simulation demonstrates successful autonomous navigation within unseen cerebral vessels, achieving a 96% success rate, 7.0s procedure time, and 0.24 N mean forces, well below the proposed 1.5 N vessel rupture threshold.
Conclusion: To the best of our knowledge, our proposed autonomous system for MT two-device navigation reaches cerebral vessels, considers safety, and is generalizable to unseen patient-specific cases for the first time. We envisage future work will extend the validation to vasculatures of different complexity and on in vitro models. While our contributions pave the way towards deploying agents in clinical settings, safety and trustworthiness will be crucial elements to consider when proposing new methodology. 

**Abstract (ZH)**: 目的：机械取栓（MT）中的自主系统有望减少手术时间、减少辐射暴露并提高患者安全性。然而，当前的强化学习（RL）方法只能达到颈动脉，不适用于其他患者血管，并未考虑安全性。我们提出了一种安全的双设备RL算法，可以导航至颈动脉以外的脑部血管。 

---
# Graph Neural Network-Based Predictive Modeling for Robotic Plaster Printing 

**Title (ZH)**: 基于图神经网络的机器人石膏打印预测建模 

**Authors**: Diego Machain Rivera, Selen Ercan Jenny, Ping Hsun Tsai, Ena Lloret-Fritschi, Luis Salamanca, Fernando Perez-Cruz, Konstantinos E. Tatsis  

**Link**: [PDF](https://arxiv.org/pdf/2503.24130)  

**Abstract**: This work proposes a Graph Neural Network (GNN) modeling approach to predict the resulting surface from a particle based fabrication process. The latter consists of spray-based printing of cementitious plaster on a wall and is facilitated with the use of a robotic arm. The predictions are computed using the robotic arm trajectory features, such as position, velocity and direction, as well as the printing process parameters. The proposed approach, based on a particle representation of the wall domain and the end effector, allows for the adoption of a graph-based solution. The GNN model consists of an encoder-processor-decoder architecture and is trained using data from laboratory tests, while the hyperparameters are optimized by means of a Bayesian scheme. The aim of this model is to act as a simulator of the printing process, and ultimately used for the generation of the robotic arm trajectory and the optimization of the printing parameters, towards the materialization of an autonomous plastering process. The performance of the proposed model is assessed in terms of the prediction error against unseen ground truth data, which shows its generality in varied scenarios, as well as in comparison with the performance of an existing benchmark model. The results demonstrate a significant improvement over the benchmark model, with notably better performance and enhanced error scaling across prediction steps. 

**Abstract (ZH)**: 该工作提出了一种图神经网络（GNN）建模方法，用于预测基于颗粒的制造过程中产生的表面。该过程包括使用机器人臂辅助的基于喷射的水泥砂浆墙打印。预测使用了机器人臂轨迹特征，如位置、速度和方向，以及打印过程参数。基于墙面域和末端执行器的颗粒表示，该提出的方法采用了一种图基解决方案。GNN模型采用编码器-处理器-解码器架构，并使用实验室测试数据进行训练，超参数则通过贝叶斯方案进行优化。该模型的目的是模拟打印过程，并最终用于生成机器人臂轨迹和优化打印参数，以实现自主抹灰过程。对该模型的性能评估基于 unseen 地面真值数据的预测误差，显示了其在各种场景中的普遍适用性，并且与现有基准模型的性能进行比较，结果显示该模型在预测误差方面有了显著改进，并在预测步骤中实现了更好的表现和改进的误差缩放。 

---
# COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation 

**Title (ZH)**: COSMO：低成本视觉与语言导航的选择性记忆结合 

**Authors**: Siqi Zhang, Yanyuan Qiao, Qunbo Wang, Zike Yan, Qi Wu, Zhihua Wei, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24065)  

**Abstract**: Vision-and-Language Navigation (VLN) tasks have gained prominence within artificial intelligence research due to their potential application in fields like home assistants. Many contemporary VLN approaches, while based on transformer architectures, have increasingly incorporated additional components such as external knowledge bases or map information to enhance performance. These additions, while boosting performance, also lead to larger models and increased computational costs. In this paper, to achieve both high performance and low computational costs, we propose a novel architecture with the COmbination of Selective MemOrization (COSMO). Specifically, COSMO integrates state-space modules and transformer modules, and incorporates two VLN-customized selective state space modules: the Round Selective Scan (RSS) and the Cross-modal Selective State Space Module (CS3). RSS facilitates comprehensive inter-modal interactions within a single scan, while the CS3 module adapts the selective state space module into a dual-stream architecture, thereby enhancing the acquisition of cross-modal interactions. Experimental validations on three mainstream VLN benchmarks, REVERIE, R2R, and R2R-CE, not only demonstrate competitive navigation performance of our model but also show a significant reduction in computational costs. 

**Abstract (ZH)**: Vision-and-Language Navigation (VLN) 任务在人工智能研究中由于其在家庭助手等领域潜在的应用获得了广泛关注。许多现代 VLN 方法虽然基于变压器架构，但越来越多地融入了外部知识库或地图信息等额外组件以提升性能。这些添加确实提高了性能，但也导致了模型规模的扩大和计算成本的增加。本文为在保持高性能的同时降低计算成本，我们提出了一种结合选择性记忆的新架构（COSMO）。具体而言，COSMO 结合了状态空间模块和变压器模块，并引入了两个定制化的 VLN 选择性状态空间模块：Round Selective Scan (RSS) 和 Cross-modal Selective State Space Module (CS3)。RSS 促进了单一扫描内的全方位跨模态交互，而 CS3 模块将选择性状态空间模块转化为双流架构，从而增强了跨模态交互的获取能力。在三个主流 VLN 验证平台上（REVERIE、R2R 和 R2R-CE）的实验验证不仅展示了我们模型的竞争力，还展示了计算成本的显著降低。 

---
# Learning 3D-Gaussian Simulators from RGB Videos 

**Title (ZH)**: 从RGB视频学习3D高斯模拟器 

**Authors**: Mikel Zhobro, Andreas René Geist, Georg Martius  

**Link**: [PDF](https://arxiv.org/pdf/2503.24009)  

**Abstract**: Learning physics simulations from video data requires maintaining spatial and temporal consistency, a challenge often addressed with strong inductive biases or ground-truth 3D information -- limiting scalability and generalization. We introduce 3DGSim, a 3D physics simulator that learns object dynamics end-to-end from multi-view RGB videos. It encodes images into a 3D Gaussian particle representation, propagates dynamics via a transformer, and renders frames using 3D Gaussian splatting. By jointly training inverse rendering with a dynamics transformer using a temporal encoding and merging layer, 3DGSimembeds physical properties into point-wise latent vectors without enforcing explicit connectivity constraints. This enables the model to capture diverse physical behaviors, from rigid to elastic and cloth-like interactions, along with realistic lighting effects that also generalize to unseen multi-body interactions and novel scene edits. 

**Abstract (ZH)**: 从多视角RGB视频中学习物理仿真要求维持空间和时间一致性，这通常通过强先验假设或真实3D信息来应对——但这种方式限制了可扩展性和泛化能力。我们提出了3DGSim，这是一种从多视角RGB视频中端到端学习物体动力学的3D物理模拟器。它通过编码图像到3D高斯粒子表示，利用变压器传播动力学，并使用3D高斯散斑进行帧渲染。通过使用时间编码和合并层联合训练逆渲染和动力学变压器，3DGSim将物理属性嵌入到点级潜在向量中，而不施加显式连接约束。这使得模型能够捕捉从刚性到弹性及布料样交互的各种物理行为，同时实现逼真的光照效果，这些效果还能泛化到未见过的多体交互和新的场景剪辑。 

---
# SALT: A Flexible Semi-Automatic Labeling Tool for General LiDAR Point Clouds with Cross-Scene Adaptability and 4D Consistency 

**Title (ZH)**: SALT：一种具有跨场景适应性与4D一致性的灵活半自动标注工具用于通用LiDAR点云 

**Authors**: Yanbo Wang, Yongtao Chen, Chuan Cao, Tianchen Deng, Wentao Zhao, Jingchuan Wang, Weidong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.23980)  

**Abstract**: We propose a flexible Semi-Automatic Labeling Tool (SALT) for general LiDAR point clouds with cross-scene adaptability and 4D consistency. Unlike recent approaches that rely on camera distillation, SALT operates directly on raw LiDAR data, automatically generating pre-segmentation results. To achieve this, we propose a novel zero-shot learning paradigm, termed data alignment, which transforms LiDAR data into pseudo-images by aligning with the training distribution of vision foundation models. Additionally, we design a 4D-consistent prompting strategy and 4D non-maximum suppression module to enhance SAM2, ensuring high-quality, temporally consistent presegmentation. SALT surpasses the latest zero-shot methods by 18.4% PQ on SemanticKITTI and achieves nearly 40-50% of human annotator performance on our newly collected low-resolution LiDAR data and on combined data from three LiDAR types, significantly boosting annotation efficiency. We anticipate that SALT's open-sourcing will catalyze substantial expansion of current LiDAR datasets and lay the groundwork for the future development of LiDAR foundation models. Code is available at this https URL. 

**Abstract (ZH)**: 我们提出了一种灵活的半自动标注工具(SALT)用于通用LiDAR点云，具有跨场景适应性和4D一致性。 

---
# Video-based Traffic Light Recognition by Rockchip RV1126 for Autonomous Driving 

**Title (ZH)**: 基于Rockchip RV1126的视频交通灯识别技术及其在自动驾驶中的应用 

**Authors**: Miao Fan, Xuxu Kong, Shengtong Xu, Haoyi Xiong, Xiangzeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23965)  

**Abstract**: Real-time traffic light recognition is fundamental for autonomous driving safety and navigation in urban environments. While existing approaches rely on single-frame analysis from onboard cameras, they struggle with complex scenarios involving occlusions and adverse lighting conditions. We present \textit{ViTLR}, a novel video-based end-to-end neural network that processes multiple consecutive frames to achieve robust traffic light detection and state classification. The architecture leverages a transformer-like design with convolutional self-attention modules, which is optimized specifically for deployment on the Rockchip RV1126 embedded platform. Extensive evaluations on two real-world datasets demonstrate that \textit{ViTLR} achieves state-of-the-art performance while maintaining real-time processing capabilities (>25 FPS) on RV1126's NPU. The system shows superior robustness across temporal stability, varying target distances, and challenging environmental conditions compared to existing single-frame approaches. We have successfully integrated \textit{ViTLR} into an ego-lane traffic light recognition system using HD maps for autonomous driving applications. The complete implementation, including source code and datasets, is made publicly available to facilitate further research in this domain. 

**Abstract (ZH)**: 基于视频的实时交通灯识别：ViTLR在城市环境自主驾驶中的应用 

---
# A Benchmark for Vision-Centric HD Mapping by V2I Systems 

**Title (ZH)**: 基于V2I系统的视觉中心高精度地图基准 

**Authors**: Miao Fan, Shanshan Yu, Shengtong Xu, Kun Jiang, Haoyi Xiong, Xiangzeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23963)  

**Abstract**: Autonomous driving faces safety challenges due to a lack of global perspective and the semantic information of vectorized high-definition (HD) maps. Information from roadside cameras can greatly expand the map perception range through vehicle-to-infrastructure (V2I) communications. However, there is still no dataset from the real world available for the study on map vectorization onboard under the scenario of vehicle-infrastructure cooperation. To prosper the research on online HD mapping for Vehicle-Infrastructure Cooperative Autonomous Driving (VICAD), we release a real-world dataset, which contains collaborative camera frames from both vehicles and roadside infrastructures, and provides human annotations of HD map elements. We also present an end-to-end neural framework (i.e., V2I-HD) leveraging vision-centric V2I systems to construct vectorized maps. To reduce computation costs and further deploy V2I-HD on autonomous vehicles, we introduce a directionally decoupled self-attention mechanism to V2I-HD. Extensive experiments show that V2I-HD has superior performance in real-time inference speed, as tested by our real-world dataset. Abundant qualitative results also demonstrate stable and robust map construction quality with low cost in complex and various driving scenes. As a benchmark, both source codes and the dataset have been released at OneDrive for the purpose of further study. 

**Abstract (ZH)**: 自动驾驶面临由于缺乏全局视角和矢量化高分辨率（HD）地图的语义信息而带来的安全挑战。路边摄像头的信息可以通过车辆到基础设施（V2I）通信极大地扩展地图感知范围。然而，在车辆基础设施合作场景下进行车载地图矢量化研究尚无实际世界的数据集可用。为了促进车辆基础设施协同自动驾驶（VICAD）在线高分辨率地图绘制的研究，我们发布了一个真实世界的数据集，包含来自车辆和路边基础设施的协作摄像头帧，并提供了高分辨率地图元素的人工标注。我们还提出了一个端到端的神经框架（即V2I-HD），利用以视觉为中心的V2I系统构建矢量化地图。为了降低计算成本并在自动驾驶车辆上进一步部署V2I-HD，我们引入了方向解耦自注意力机制到V2I-HD中。大量的实验表明，V2I-HD在实时推断速度上表现出卓越的性能，经过我们真实世界的数据集测试。丰富的定性结果还展示了在复杂多样的驾驶场景中低成本的稳定和稳健的地图构建质量。作为基准，开源代码和数据集已在OneDrive上发布，以供进一步研究。 

---
# A Survey of Reinforcement Learning-Based Motion Planning for Autonomous Driving: Lessons Learned from a Driving Task Perspective 

**Title (ZH)**: 基于强化学习的自主驾驶运动规划综述：从驾驶任务视角学到的教训 

**Authors**: Zhuoren Li, Guizhe Jin, Ran Yu, Zhiwen Chen, Nan Li, Wei Han, Lu Xiong, Bo Leng, Jia Hu, Ilya Kolmanovsky, Dimitar Filev  

**Link**: [PDF](https://arxiv.org/pdf/2503.23650)  

**Abstract**: Reinforcement learning (RL), with its ability to explore and optimize policies in complex, dynamic decision-making tasks, has emerged as a promising approach to addressing motion planning (MoP) challenges in autonomous driving (AD). Despite rapid advancements in RL and AD, a systematic description and interpretation of the RL design process tailored to diverse driving tasks remains underdeveloped. This survey provides a comprehensive review of RL-based MoP for AD, focusing on lessons from task-specific perspectives. We first outline the fundamentals of RL methodologies, and then survey their applications in MoP, analyzing scenario-specific features and task requirements to shed light on their influence on RL design choices. Building on this analysis, we summarize key design experiences, extract insights from various driving task applications, and provide guidance for future implementations. Additionally, we examine the frontier challenges in RL-based MoP, review recent efforts to addresse these challenges, and propose strategies for overcoming unresolved issues. 

**Abstract (ZH)**: 强化学习（RL）在复杂动态决策任务中探索和优化政策的能力使其成为解决自主驾驶（AD）中运动规划（MoP）挑战的一种有前途的方法。尽管在RL和AD领域取得了快速进展，但对于多样化的驾驶任务而言，特定任务视角下的RL设计过程的系统描述和解释仍需进一步发展。本文综述了基于RL的AD中MoP，重点从任务特定视角总结经验教训。我们首先概述了RL方法的基础，然后调查其在MoP中的应用，分析特定场景特征和任务需求，阐明它们对RL设计选择的影响。基于这一分析，我们总结了关键设计经验，从各种驾驶任务应用中提取见解，并为未来的实施提供指导。此外，我们检查了基于RL的MoP的前沿挑战，回顾了最近解决这些挑战的努力，并提出了克服未解决问题的策略。 

---
# PhysPose: Refining 6D Object Poses with Physical Constraints 

**Title (ZH)**: PhysPose: 通过物理约束细化6D物体姿态 

**Authors**: Martin Malenický, Martin Cífka, Médéric Fourmy, Louis Montaut, Justin Carpentier, Josef Sivic, Vladimir Petrik  

**Link**: [PDF](https://arxiv.org/pdf/2503.23587)  

**Abstract**: Accurate 6D object pose estimation from images is a key problem in object-centric scene understanding, enabling applications in robotics, augmented reality, and scene reconstruction. Despite recent advances, existing methods often produce physically inconsistent pose estimates, hindering their deployment in real-world scenarios. We introduce PhysPose, a novel approach that integrates physical reasoning into pose estimation through a postprocessing optimization enforcing non-penetration and gravitational constraints. By leveraging scene geometry, PhysPose refines pose estimates to ensure physical plausibility. Our approach achieves state-of-the-art accuracy on the YCB-Video dataset from the BOP benchmark and improves over the state-of-the-art pose estimation methods on the HOPE-Video dataset. Furthermore, we demonstrate its impact in robotics by significantly improving success rates in a challenging pick-and-place task, highlighting the importance of physical consistency in real-world applications. 

**Abstract (ZH)**: 基于图像的准确6D物体姿态估计是物体中心场景理解的关键问题，.enable robotics、增强现实和场景重建的应用。尽管近期取得了进展，现有方法通常会产生物理不一致的姿态估计，阻碍其在现实世界场景中的部署。我们引入了PhysPose，一种通过后处理优化结合物理推理来进行姿态估计的新方法，该方法通过施加非穿透性和重力约束来提升姿态估计的准确性。借助场景几何信息，PhysPose 对姿态估计进行细化以确保物理合理性。我们的方法在BOP基准的YCB-Video数据集上达到了最先进的准确度，并在HOPE-Video数据集上改进了最先进的姿态估计方法。此外，我们通过显著提高一项具有挑战性的抓取和放置任务的成功率，展示了其在机器人领域的应用影响，强调了物理一致性在实际应用中的重要性。 

---
# Boosting Omnidirectional Stereo Matching with a Pre-trained Depth Foundation Model 

**Title (ZH)**: 基于预训练深度基础模型的全向立体匹配增强 

**Authors**: Jannik Endres, Oliver Hahn, Charles Corbière, Simone Schaub-Meyer, Stefan Roth, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2503.23502)  

**Abstract**: Omnidirectional depth perception is essential for mobile robotics applications that require scene understanding across a full 360° field of view. Camera-based setups offer a cost-effective option by using stereo depth estimation to generate dense, high-resolution depth maps without relying on expensive active sensing. However, existing omnidirectional stereo matching approaches achieve only limited depth accuracy across diverse environments, depth ranges, and lighting conditions, due to the scarcity of real-world data. We present DFI-OmniStereo, a novel omnidirectional stereo matching method that leverages a large-scale pre-trained foundation model for relative monocular depth estimation within an iterative optimization-based stereo matching architecture. We introduce a dedicated two-stage training strategy to utilize the relative monocular depth features for our omnidirectional stereo matching before scale-invariant fine-tuning. DFI-OmniStereo achieves state-of-the-art results on the real-world Helvipad dataset, reducing disparity MAE by approximately 16% compared to the previous best omnidirectional stereo method. 

**Abstract (ZH)**: 全方位深度感知对于需要全方位360°视野场景理解的移动机器人应用至关重要。基于摄像头的设置通过使用立体深度估计生成密集的高分辨率深度图，是一种成本有效的选择，无需依赖昂贵的主动传感。然而，现有的全方位立体配对方法在多样化的环境、深度范围和光照条件下的深度精度有限，这归因于实际数据的稀缺性。我们提出了DFI-OmniStereo，这是一种新型全方位立体配对方法，利用大规模预训练基础模型进行迭代优化为主的立体配对架构中的相对单目深度估计。我们引入了一种专用的两阶段训练策略，在进行尺度不变微调之前利用相对单目深度特征进行全方位立体配对。DFI-OmniStereo在实际的Helvipad数据集上达到了最先进的性能，与之前最好的全方位立体方法相比， disparity MAE降低了约16%。 

---
# Handling Delay in Real-Time Reinforcement Learning 

**Title (ZH)**: 处理实时强化学习中的延迟 

**Authors**: Ivan Anokhin, Rishav Rishav, Matthew Riemer, Stephen Chung, Irina Rish, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2503.23478)  

**Abstract**: Real-time reinforcement learning (RL) introduces several challenges. First, policies are constrained to a fixed number of actions per second due to hardware limitations. Second, the environment may change while the network is still computing an action, leading to observational delay. The first issue can partly be addressed with pipelining, leading to higher throughput and potentially better policies. However, the second issue remains: if each neuron operates in parallel with an execution time of $\tau$, an $N$-layer feed-forward network experiences observation delay of $\tau N$. Reducing the number of layers can decrease this delay, but at the cost of the network's expressivity. In this work, we explore the trade-off between minimizing delay and network's expressivity. We present a theoretically motivated solution that leverages temporal skip connections combined with history-augmented observations. We evaluate several architectures and show that those incorporating temporal skip connections achieve strong performance across various neuron execution times, reinforcement learning algorithms, and environments, including four Mujoco tasks and all MinAtar games. Moreover, we demonstrate parallel neuron computation can accelerate inference by 6-350% on standard hardware. Our investigation into temporal skip connections and parallel computations paves the way for more efficient RL agents in real-time setting. 

**Abstract (ZH)**: 实时强化学习中的延迟与网络表达能力trade-off研究：基于时间跳过连接的历史增强观测方法 

---
# OnSiteVRU: A High-Resolution Trajectory Dataset for High-Density Vulnerable Road Users 

**Title (ZH)**: OnSiteVRU: 高密度脆弱道路使用者高分辨率轨迹数据集 

**Authors**: Zhangcun Yan, Jianqing Li, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.23365)  

**Abstract**: With the acceleration of urbanization and the growth of transportation demands, the safety of vulnerable road users (VRUs, such as pedestrians and cyclists) in mixed traffic flows has become increasingly prominent, necessitating high-precision and diverse trajectory data to support the development and optimization of autonomous driving systems. However, existing datasets fall short in capturing the diversity and dynamics of VRU behaviors, making it difficult to meet the research demands of complex traffic environments. To address this gap, this study developed the OnSiteVRU datasets, which cover a variety of scenarios, including intersections, road segments, and urban villages. These datasets provide trajectory data for motor vehicles, electric bicycles, and human-powered bicycles, totaling approximately 17,429 trajectories with a precision of 0.04 seconds. The datasets integrate both aerial-view natural driving data and onboard real-time dynamic detection data, along with environmental information such as traffic signals, obstacles, and real-time maps, enabling a comprehensive reconstruction of interaction events. The results demonstrate that VRU\_Data outperforms traditional datasets in terms of VRU density and scene coverage, offering a more comprehensive representation of VRU behavioral characteristics. This provides critical support for traffic flow modeling, trajectory prediction, and autonomous driving virtual testing. The dataset is publicly available for download at:
this https URL. 

**Abstract (ZH)**: 随着城市化进程的加速和交通需求的增长，混合交通流中弱势道路使用者（如行人和自行车骑行者）的安全问题日益突出，需要高精度和多样化的轨迹数据以支持自动驾驶系统的研发与优化。然而，现有数据集在捕捉弱势道路使用者行为的多样性和动态性方面存在不足，难以满足复杂交通环境下的研究需求。为弥补这一差距，本研究开发了OnSiteVRU数据集，涵盖了交叉口、道路段和城乡结合部等多种场景。该数据集提供了机动车、电动自行车和人力自行车的轨迹数据，总计约17,429条轨迹，精度达到0.04秒。数据集整合了空中视角的自然驾驶数据和车载实时动态检测数据，以及交通信号、障碍物和实时地图等环境信息，能够全面重构交互事件。结果显示，VRU_Data在弱势道路使用者密度和场景覆盖方面优于传统数据集，提供了更全面的弱势道路使用者行为特征表示。该数据集为交通流建模、轨迹预测和自动驾驶虚拟测试提供了关键支持。数据集在此处免费下载：[this https URL](this https URL)。 

---
# Reinforcement Learning for Active Matter 

**Title (ZH)**: 自推进物质中的强化学习 

**Authors**: Wenjie Cai, Gongyi Wang, Yu Zhang, Xiang Qu, Zihan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.23308)  

**Abstract**: Active matter refers to systems composed of self-propelled entities that consume energy to produce motion, exhibiting complex non-equilibrium dynamics that challenge traditional models. With the rapid advancements in machine learning, reinforcement learning (RL) has emerged as a promising framework for addressing the complexities of active matter. This review systematically introduces the integration of RL for guiding and controlling active matter systems, focusing on two key aspects: optimal motion strategies for individual active particles and the regulation of collective dynamics in active swarms. We discuss the use of RL to optimize the navigation, foraging, and locomotion strategies for individual active particles. In addition, the application of RL in regulating collective behaviors is also examined, emphasizing its role in facilitating the self-organization and goal-directed control of active swarms. This investigation offers valuable insights into how RL can advance the understanding, manipulation, and control of active matter, paving the way for future developments in fields such as biological systems, robotics, and medical science. 

**Abstract (ZH)**: 活性物质是指由自推进单元组成的系统，这些单元通过消耗能量产生运动，表现出挑战传统模型的复杂非平衡动力学。随着机器学习的迅速发展，强化学习（RL）已成为处理活性物质复杂性的有前途的框架。本文系统介绍了RL在引导和控制活性物质系统中的应用，着重于两个关键方面：单个活性粒子的最佳运动策略以及活性群落动力学的调节。讨论了使用RL优化单个活性粒子的导航、觅食和运动策略。此外，还考察了RL在调节集体行为中的应用，强调了其在促进活性群落的自我组织和目标导向控制方面的作用。本文为理解、操控和控制活性物质提供了有价值的见解，为生物系统、机器人技术和医学科学等领域未来的发展铺平了道路。 

---
# Learning Predictive Visuomotor Coordination 

**Title (ZH)**: 预测性可视化运动协调 

**Authors**: Wenqi Jia, Bolin Lai, Miao Liu, Danfei Xu, James M. Rehg  

**Link**: [PDF](https://arxiv.org/pdf/2503.23300)  

**Abstract**: Understanding and predicting human visuomotor coordination is crucial for applications in robotics, human-computer interaction, and assistive technologies. This work introduces a forecasting-based task for visuomotor modeling, where the goal is to predict head pose, gaze, and upper-body motion from egocentric visual and kinematic observations. We propose a \textit{Visuomotor Coordination Representation} (VCR) that learns structured temporal dependencies across these multimodal signals. We extend a diffusion-based motion modeling framework that integrates egocentric vision and kinematic sequences, enabling temporally coherent and accurate visuomotor predictions. Our approach is evaluated on the large-scale EgoExo4D dataset, demonstrating strong generalization across diverse real-world activities. Our results highlight the importance of multimodal integration in understanding visuomotor coordination, contributing to research in visuomotor learning and human behavior modeling. 

**Abstract (ZH)**: 理解并预测人类的视动协调对于机器人技术、人机交互和辅助技术的应用至关重要。本工作引入了一种基于预测的任务，旨在从第一人称视觉和运动观察中预测头部姿态、凝视和上半身运动。我们提出了一种视动协调表示（VCR），用于学习这些多模态信号之间的结构化时间依赖性。我们扩展了一种基于扩散的运动建模框架，该框架结合了第一人称视觉和运动序列，能够实现时间连贯且准确的视动预测。我们在大规模的EgoExo4D数据集上评估了我们的方法，展示了在多种真实世界活动中的强泛化能力。我们的结果强调了多模态集成在理解视动协调中的重要性，为视动学习和人类行为建模研究做出了贡献。 

---
# Energy-Aware Lane Planning for Connected Electric Vehicles in Urban Traffic: Design and Vehicle-in-the-Loop Validation 

**Title (ZH)**: 面向连接电动车辆的城市交通能效导向车道规划：设计与车辆在环验证 

**Authors**: Hansung Kim, Eric Yongkeun Choi, Eunhyek Joa, Hotae Lee, Linda Lim, Scott Moura, Francesco Borrelli  

**Link**: [PDF](https://arxiv.org/pdf/2503.23228)  

**Abstract**: Urban driving with connected and automated vehicles (CAVs) offers potential for energy savings, yet most eco-driving strategies focus solely on longitudinal speed control within a single lane. This neglects the significant impact of lateral decisions, such as lane changes, on overall energy efficiency, especially in environments with traffic signals and heterogeneous traffic flow. To address this gap, we propose a novel energy-aware motion planning framework that jointly optimizes longitudinal speed and lateral lane-change decisions using vehicle-to-infrastructure (V2I) communication. Our approach estimates long-term energy costs using a graph-based approximation and solves short-horizon optimal control problems under traffic constraints. Using a data-driven energy model calibrated to an actual battery electric vehicle, we demonstrate with vehicle-in-the-loop experiments that our method reduces motion energy consumption by up to 24 percent compared to a human driver, highlighting the potential of connectivity-enabled planning for sustainable urban autonomy. 

**Abstract (ZH)**: 使用连接和自动驾驶车辆的城市驾驶提供了节能的潜力，然而大多数环保驾驶策略主要关注单车道内的纵向速度控制，忽略了横向决策，如车道变换，对整体能效的显著影响，特别是在有交通信号和异质交通流的环境中。为弥补这一不足，我们提出了一种新颖的能量感知运动规划框架，该框架利用车辆到基础设施（V2I）通信联合优化纵向速度和横向车道变换决策。我们的方法使用图基近似估算长期能量成本，并在交通约束下解决短期最优控制问题。通过针对实际电池电动汽车的数据驱动能量模型，我们通过车辆在环实验表明，与人类驾驶员相比，我们的方法可将运动能量消耗减少最多24%，突出了连接性规划在可持续城市自主中的潜力。 

---
# Can DeepSeek-V3 Reason Like a Surgeon? An Empirical Evaluation for Vision-Language Understanding in Robotic-Assisted Surgery 

**Title (ZH)**: Can DeepSeek-V3 模拟外科医生的推理能力？一种基于视觉-语言理解的机器人辅助手术实证评估 

**Authors**: Boyi Ma, Yanguang Zhao, Jie Wang, Guankun Wang, Kun Yuan, Tong Chen, Long Bai, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2503.23130)  

**Abstract**: DeepSeek-V3, a recently emerging Large Language Model (LLM), demonstrates outstanding performance in general scene understanding, question-answering (QA), and text generation tasks, owing to its efficient training paradigm and strong reasoning capabilities. In this study, we investigate the dialogue capabilities of DeepSeek-V3 in robotic surgery scenarios, focusing on tasks such as Single Phrase QA, Visual QA, and Detailed Description. The Single Phrase QA tasks further include sub-tasks such as surgical instrument recognition, action understanding, and spatial position analysis. We conduct extensive evaluations using publicly available datasets, including EndoVis18 and CholecT50, along with their corresponding dialogue data. Our comprehensive evaluation results indicate that, when provided with specific prompts, DeepSeek-V3 performs well in surgical instrument and tissue recognition tasks However, DeepSeek-V3 exhibits significant limitations in spatial position analysis and struggles to understand surgical actions accurately. Additionally, our findings reveal that, under general prompts, DeepSeek-V3 lacks the ability to effectively analyze global surgical concepts and fails to provide detailed insights into surgical scenarios. Based on our observations, we argue that the DeepSeek-V3 is not ready for vision-language tasks in surgical contexts without fine-tuning on surgery-specific datasets. 

**Abstract (ZH)**: DeepSeek-V3：一种新兴的大语言模型在机器人手术对话能力研究 

---
# Evaluation of Remote Driver Performance in Urban Environment Operational Design Domains 

**Title (ZH)**: 城市环境操作设计域中远程驾驶性能评价 

**Authors**: Ole Hans, Benedikt Walter, Jürgen Adamy  

**Link**: [PDF](https://arxiv.org/pdf/2503.22992)  

**Abstract**: Remote driving has emerged as a solution for enabling human intervention in scenarios where Automated Driving Systems (ADS) face challenges, particularly in urban Operational Design Domains (ODDs). This study evaluates the performance of Remote Drivers (RDs) of passenger cars in a representative urban ODD in Las Vegas, focusing on the influence of cumulative driving experience and targeted training approaches. Using performance metrics such as efficiency, braking, acceleration, and steering, the study shows that driving experience can lead to noticeable improvements of RDs and demonstrates how experience up to 600 km correlates with improved vehicle control. In addition, driving efficiency exhibited a positive trend with increasing kilometers, particularly during the first 300 km of experience, which reaches a plateau from 400 km within a range of 0.35 to 0.42 km/min in the defined ODD. The research further compares ODD-specific training methods, where the detailed ODD training approaches attains notable advantages over other training approaches. The findings underscore the importance of tailored ODD training in enhancing RD performance, safety, and scalability for Remote Driving System (RDS) in real-world applications, while identifying opportunities for optimizing training protocols to address both routine and extreme scenarios. The study provides a robust foundation for advancing RDS deployment within urban environments, contributing to the development of scalable and safety-critical remote operation standards. 

**Abstract (ZH)**: 远程驾驶在应对自动化驾驶系统在城市操作设计领域（ODD）面临的挑战中 emerged as a解决方案。本研究评估了在拉斯维加斯一个代表性城市ODD中远程驾驶员（RDs）的表现，重点关注累计驾驶经验与目标培训方法的影响。通过使用效率、刹车、加速和转向等性能指标，研究显示驾驶经验可以显著提高远程驾驶员的表现，并表明累计驾驶600公里的经验与车辆控制的改善相关。此外，随经验增加的驾驶效率呈现出积极趋势，特别是在经验最初的300公里中，效率在定义的ODD范围内于400公里左右达到0.35至0.42公里/分钟的 plateau。研究进一步比较了ODD特定的培训方法，其中详细的ODD培训方法表现出显著的优势。研究结果强调了为远程驾驶系统（RDS）在实际应用中增强远程驾驶员的表现、安全性和可扩展性，而定制的ODD培训的重要性，并指出了优化培训协议以应对常规和极端情况的机会。该研究为在城市环境中推进远程驾驶系统的部署奠定了坚实的基础，促进了可扩展且安全关键的远程操作标准的发展。 

---
# Task Tokens: A Flexible Approach to Adapting Behavior Foundation Models 

**Title (ZH)**: 任务令牌：一种灵活的行为基础模型适应方法 

**Authors**: Ron Vainshtein, Zohar Rimon, Shie Mannor, Chen Tessler  

**Link**: [PDF](https://arxiv.org/pdf/2503.22886)  

**Abstract**: Recent advancements in imitation learning have led to transformer-based behavior foundation models (BFMs) that enable multi-modal, human-like control for humanoid agents. While excelling at zero-shot generation of robust behaviors, BFMs often require meticulous prompt engineering for specific tasks, potentially yielding suboptimal results. We introduce "Task Tokens", a method to effectively tailor BFMs to specific tasks while preserving their flexibility. Our approach leverages the transformer architecture of BFMs to learn a new task-specific encoder through reinforcement learning, keeping the original BFM frozen. This allows incorporation of user-defined priors, balancing reward design and prompt engineering. By training a task encoder to map observations to tokens, used as additional BFM inputs, we guide performance improvement while maintaining the model's diverse control characteristics. We demonstrate Task Tokens' efficacy across various tasks, including out-of-distribution scenarios, and show their compatibility with other prompting modalities. Our results suggest that Task Tokens offer a promising approach for adapting BFMs to specific control tasks while retaining their generalization capabilities. 

**Abstract (ZH)**: 近期模仿学习的进展催生了基于变压器的行为基础模型（BFMs），这些模型能够为类人代理提供多模态、类人的控制。尽管BFMs在零-shot生成稳健行为方面表现出色，但对于特定任务，它们通常需要细致的提示工程，可能导致次优结果。我们引入了“任务令牌”方法，以有效适应BFMs到特定任务的同时保持其灵活性。我们的方法利用BFMs的变压器架构，通过强化学习学习一个新的任务特定编码器，同时冻结原始BFM。这允许整合用户定义的先验知识，平衡奖励设计和提示工程。通过训练任务编码器将观察值映射到令牌，作为BFM的额外输入，我们可以在保持模型多样化控制特性的同时引导性能改进。我们展示了任务令牌在多种任务中的有效性，包括分布外场景，并表明它们与其他提示方法兼容。我们的结果表明，任务令牌提供了一种有前景的方法来适应BFMs以进行特定控制任务，同时保留其泛化能力。 

---
# A Multiple Artificial Potential Functions Approach for Collision Avoidance in UAV Systems 

**Title (ZH)**: 基于多个人工势能函数的方法在无人机系统中的碰撞避免 

**Authors**: Oscar F. Archila, Alain Vande Wouwer, Johannes Schiffer  

**Link**: [PDF](https://arxiv.org/pdf/2503.22830)  

**Abstract**: Collision avoidance is a problem largely studied in robotics, particularly in unmanned aerial vehicle (UAV) applications. Among the main challenges in this area are hardware limitations, the need for rapid response, and the uncertainty associated with obstacle detection. Artificial potential functions (APOFs) are a prominent method to address these challenges. However, existing solutions lack assurances regarding closed-loop stability and may result in chattering effects. Motivated by this, we propose a control method for static obstacle avoidance based on multiple artificial potential functions (MAPOFs). We derive tuning conditions on the control parameters that ensure the stability of the final position. The stability proof is established by analyzing the closed-loop system using tools from hybrid systems theory. Furthermore, we validate the performance of the MAPOF control through simulations, showcasing its effectiveness in avoiding static obstacles. 

**Abstract (ZH)**: 基于多人工势函数的静态障碍物规避控制方法及其稳定性分析 

---
