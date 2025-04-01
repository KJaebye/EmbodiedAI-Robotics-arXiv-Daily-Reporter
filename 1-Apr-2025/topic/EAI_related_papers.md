# Sim-and-Real Co-Training: A Simple Recipe for Vision-Based Robotic Manipulation 

**Title (ZH)**: 基于视觉的机器人 manipulation: Sim-and-Real Co-Training 的简单配方 

**Authors**: Abhiram Maddukuri, Zhenyu Jiang, Lawrence Yunliang Chen, Soroush Nasiriany, Yuqi Xie, Yu Fang, Wenqi Huang, Zu Wang, Zhenjia Xu, Nikita Chernyadev, Scott Reed, Ken Goldberg, Ajay Mandlekar, Linxi Fan, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24361)  

**Abstract**: Large real-world robot datasets hold great potential to train generalist robot models, but scaling real-world human data collection is time-consuming and resource-intensive. Simulation has great potential in supplementing large-scale data, especially with recent advances in generative AI and automated data generation tools that enable scalable creation of robot behavior datasets. However, training a policy solely in simulation and transferring it to the real world often demands substantial human effort to bridge the reality gap. A compelling alternative is to co-train the policy on a mixture of simulation and real-world datasets. Preliminary studies have recently shown this strategy to substantially improve the performance of a policy over one trained on a limited amount of real-world data. Nonetheless, the community lacks a systematic understanding of sim-and-real co-training and what it takes to reap the benefits of simulation data for real-robot learning. This work presents a simple yet effective recipe for utilizing simulation data to solve vision-based robotic manipulation tasks. We derive this recipe from comprehensive experiments that validate the co-training strategy on various simulation and real-world datasets. Using two domains--a robot arm and a humanoid--across diverse tasks, we demonstrate that simulation data can enhance real-world task performance by an average of 38%, even with notable differences between the simulation and real-world data. Videos and additional results can be found at this https URL 

**Abstract (ZH)**: 利用模拟数据与现实数据联合训练解决基于视觉的机器人 manipulation 任务的简单有效方法 

---
# AutoEval: Autonomous Evaluation of Generalist Robot Manipulation Policies in the Real World 

**Title (ZH)**: AutoEval：自主评估通用机器人操作政策在真实世界中的表现 

**Authors**: Zhiyuan Zhou, Pranav Atreya, You Liang Tan, Karl Pertsch, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2503.24278)  

**Abstract**: Scalable and reproducible policy evaluation has been a long-standing challenge in robot learning. Evaluations are critical to assess progress and build better policies, but evaluation in the real world, especially at a scale that would provide statistically reliable results, is costly in terms of human time and hard to obtain. Evaluation of increasingly generalist robot policies requires an increasingly diverse repertoire of evaluation environments, making the evaluation bottleneck even more pronounced. To make real-world evaluation of robotic policies more practical, we propose AutoEval, a system to autonomously evaluate generalist robot policies around the clock with minimal human intervention. Users interact with AutoEval by submitting evaluation jobs to the AutoEval queue, much like how software jobs are submitted with a cluster scheduling system, and AutoEval will schedule the policies for evaluation within a framework supplying automatic success detection and automatic scene resets. We show that AutoEval can nearly fully eliminate human involvement in the evaluation process, permitting around the clock evaluations, and the evaluation results correspond closely to ground truth evaluations conducted by hand. To facilitate the evaluation of generalist policies in the robotics community, we provide public access to multiple AutoEval scenes in the popular BridgeData robot setup with WidowX robot arms. In the future, we hope that AutoEval scenes can be set up across institutions to form a diverse and distributed evaluation network. 

**Abstract (ZH)**: 可扩展且可重现的政策评估一直是机器人学习中的长期挑战。为了使机器人政策的实地评估更加实用，我们提出了AutoEval系统，该系统能够在最少人工干预的情况下，全天候自主评估通用机器人政策。通过向AutoEval队列提交评估作业，用户可以像使用集群调度系统提交软件作业一样，让AutoEval框架负责自动成功检测和自动场景重置，以安排策略进行评估。我们展示了AutoEval几乎可以完全消除评估过程中的手动干预，实现全天候评估，且评估结果与手工进行的真实评估结果高度一致。为了促进机器人社区内通用策略的评估，我们提供了在流行BridgeData机器人设置中使用WidowX机器人手臂的多个AutoEval场景的公共访问权限。未来，我们希望可以在不同机构之间设置AutoEval场景，形成一个多样化且分布式的评估网络。 

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

**Abstract (ZH)**: 基于自然语言简要说明的多机器人系统的端到端自动生成与部署控制策略系统GenSwarm 

---
# Towards a cognitive architecture to enable natural language interaction in co-constructive task learning 

**Title (ZH)**: 面向支持合作建构性任务学习的自然语言交互的认知架构 

**Authors**: Manuel Scheibl, Birte Richter, Alissa Müller, Michael Beetz, Britta Wrede  

**Link**: [PDF](https://arxiv.org/pdf/2503.23760)  

**Abstract**: This research addresses the question, which characteristics a cognitive architecture must have to leverage the benefits of natural language in Co-Constructive Task Learning (CCTL). To provide context, we first discuss Interactive Task Learning (ITL), the mechanisms of the human memory system, and the significance of natural language and multi-modality. Next, we examine the current state of cognitive architectures, analyzing their capabilities to inform a concept of CCTL grounded in multiple sources. We then integrate insights from various research domains to develop a unified framework. Finally, we conclude by identifying the remaining challenges and requirements necessary to achieve CCTL in Human-Robot Interaction (HRI). 

**Abstract (ZH)**: 本研究探讨了认知架构必须具备哪些特征以利用自然语言在联合建构性任务学习（CCTL）中的优势。首先，我们讨论了交互式任务学习（ITL）、人类记忆系统的机制以及自然语言和多模态的重要性。随后，我们分析了当前认知架构的能力，以构建一个基于多种来源的概念。接着，我们综合各研究领域的见解，开发了一个统一框架。最后，我们总结了实现人类-机器人交互（HRI）中CCTL仍需克服的挑战和要求。 

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
# Physically Ground Commonsense Knowledge for Articulated Object Manipulation with Analytic Concepts 

**Title (ZH)**: 基于物理支撑常识 knowledge 的分段物体操作分析概念方法 

**Authors**: Jianhua Sun, Jiude Wei, Yuxuan Li, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23348)  

**Abstract**: We human rely on a wide range of commonsense knowledge to interact with an extensive number and categories of objects in the physical world. Likewise, such commonsense knowledge is also crucial for robots to successfully develop generalized object manipulation skills. While recent advancements in Large Language Models (LLM) have showcased their impressive capabilities in acquiring commonsense knowledge and conducting commonsense reasoning, effectively grounding this semantic-level knowledge produced by LLMs to the physical world to thoroughly guide robots in generalized articulated object manipulation remains a challenge that has not been sufficiently addressed. To this end, we introduce analytic concepts, procedurally defined upon mathematical symbolism that can be directly computed and simulated by machines. By leveraging the analytic concepts as a bridge between the semantic-level knowledge inferred by LLMs and the physical world where real robots operate, we are able to figure out the knowledge of object structure and functionality with physics-informed representations, and then use the physically grounded knowledge to instruct robot control policies for generalized, interpretable and accurate articulated object manipulation. Extensive experiments in both simulation and real-world environments demonstrate the superiority of our approach. 

**Abstract (ZH)**: 我们人类依赖广泛的知识库与物理世界中的各种对象进行交互。同样，这种常识性知识对于机器人成功发展通用对象操控技能也至关重要。尽管大型语言模型（LLM）的最新进展展示了它们在获取常识性知识和进行常识性推理方面的出色能力，但将由LLM生成的语义级知识有效地接地至物理世界，以彻底指导机器人进行通用的精细对象操控这一挑战尚未得到充分解决。为此，我们引入了分析概念，这些概念基于可被机器直接计算和模拟的数学符号定义。通过利用分析概念作为LLM推断出的语义级知识与真实机器人操作的物理世界之间的桥梁，我们可以利用基于物理的知识来理解和表示对象的结构和功能，并利用物理接地的知识来指导机器人的控制策略，实现通用、可解释和精确的精细对象操控。在仿真和真实环境中的广泛实验验证了我们方法的优势。 

---
# Learning Coordinated Bimanual Manipulation Policies using State Diffusion and Inverse Dynamics Models 

**Title (ZH)**: 基于状态扩散和逆动力学模型的学习协调双臂 manipulation 策略 

**Authors**: Haonan Chen, Jiaming Xu, Lily Sheng, Tianchen Ji, Shuijing Liu, Yunzhu Li, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2503.23271)  

**Abstract**: When performing tasks like laundry, humans naturally coordinate both hands to manipulate objects and anticipate how their actions will change the state of the clothes. However, achieving such coordination in robotics remains challenging due to the need to model object movement, predict future states, and generate precise bimanual actions. In this work, we address these challenges by infusing the predictive nature of human manipulation strategies into robot imitation learning. Specifically, we disentangle task-related state transitions from agent-specific inverse dynamics modeling to enable effective bimanual coordination. Using a demonstration dataset, we train a diffusion model to predict future states given historical observations, envisioning how the scene evolves. Then, we use an inverse dynamics model to compute robot actions that achieve the predicted states. Our key insight is that modeling object movement can help learning policies for bimanual coordination manipulation tasks. Evaluating our framework across diverse simulation and real-world manipulation setups, including multimodal goal configurations, bimanual manipulation, deformable objects, and multi-object setups, we find that it consistently outperforms state-of-the-art state-to-action mapping policies. Our method demonstrates a remarkable capacity to navigate multimodal goal configurations and action distributions, maintain stability across different control modes, and synthesize a broader range of behaviors than those present in the demonstration dataset. 

**Abstract (ZH)**: 基于人类操作策略预测的双臂协调机器人模仿学习 

---
# Microscopic Robots That Sense, Think, Act, and Compute 

**Title (ZH)**: 感知、思考、行动与计算的微纳米机器人 

**Authors**: Maya M. Lassiter, Jungho Lee, Kyle Skelil, Li Xu, Lucas Hanson, William H. Reinhardt, Dennis Sylvester, Mark Yim, David Blaauw, Marc Z. Miskin  

**Link**: [PDF](https://arxiv.org/pdf/2503.23085)  

**Abstract**: While miniaturization has been a goal in robotics for nearly 40 years, roboticists have struggled to access sub-millimeter dimensions without making sacrifices to on-board information processing due to the unique physics of the microscale. Consequently, microrobots often lack the key features that distinguish their macroscopic cousins from other machines, namely on-robot systems for decision making, sensing, feedback, and programmable computation. Here, we take up the challenge of building a microrobot comparable in size to a single-celled paramecium that can sense, think, and act using onboard systems for computation, sensing, memory, locomotion, and communication. Built massively in parallel with fully lithographic processing, these microrobots can execute digitally defined algorithms and autonomously change behavior in response to their surroundings. Combined, these results pave the way for general purpose microrobots that can be programmed many times in a simple setup, cost under $0.01 per machine, and work together to carry out tasks without supervision in uncertain environments. 

**Abstract (ZH)**: 微型机器人：构建能够在亚毫米尺度上感知、思考和行动的自主计算微型机器人 

---
# Adaptive Interactive Navigation of Quadruped Robots using Large Language Models 

**Title (ZH)**: 基于大型语言模型的四足机器人自适应交互导航 

**Authors**: Kangjie Zhou, Yao Mu, Haoyang Song, Yi Zeng, Pengying Wu, Han Gao, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22942)  

**Abstract**: Robotic navigation in complex environments remains a critical research challenge. Traditional navigation methods focus on optimal trajectory generation within free space, struggling in environments lacking viable paths to the goal, such as disaster zones or cluttered warehouses. To address this gap, we propose an adaptive interactive navigation approach that proactively interacts with environments to create feasible paths to reach originally unavailable goals. Specifically, we present a primitive tree for task planning with large language models (LLMs), facilitating effective reasoning to determine interaction objects and sequences. To ensure robust subtask execution, we adopt reinforcement learning to pre-train a comprehensive skill library containing versatile locomotion and interaction behaviors for motion planning. Furthermore, we introduce an adaptive replanning method featuring two LLM-based modules: an advisor serving as a flexible replanning trigger and an arborist for autonomous plan adjustment. Integrated with the tree structure, the replanning mechanism allows for convenient node addition and pruning, enabling rapid plan modification in unknown environments. Comprehensive simulations and experiments have demonstrated our method's effectiveness and adaptivity in diverse scenarios. The supplementary video is available at page: this https URL. 

**Abstract (ZH)**: 复杂环境中的机器人导航仍然是一个关键的研究挑战。传统的导航方法专注于自由空间内的最优轨迹生成，而在缺乏到达目标的有效路径的环境中（如灾难现场或杂乱的仓库）表现不佳。为了解决这一问题，我们提出了一种主动适应的交互导航方法，能够在与环境互动的过程中创建通往原本不可达目标的可行路径。具体来说，我们利用大规模语言模型（LLMs）构建任务规划的原语树，促进有效的推理以确定交互对象和顺序。为了确保子任务执行的鲁棒性，我们采用强化学习预先训练了一个包含多种行动和交互行为的技能库，用于运动规划。此外，我们引入了一种适应性的重新规划方法，包含两个基于大规模语言模型的模块：顾问作为灵活的重新规划触发器，树匠负责自主调整计划。与树结构集成，重新规划机制允许方便地添加和修剪节点，使得在未知环境中能够快速修改计划。全面的仿真和实验表明，该方法在多种场景下具有有效性和适应性。补充视频见：this https URL。 

---
# VizFlyt: Perception-centric Pedagogical Framework For Autonomous Aerial Robots 

**Title (ZH)**: VizFlyt: 以感知为中心的自主无人机教学框架 

**Authors**: Kushagra Srivastava, Rutwik Kulkarni, Manoj Velmurugan, Nitin J. Sanket  

**Link**: [PDF](https://arxiv.org/pdf/2503.22876)  

**Abstract**: Autonomous aerial robots are becoming commonplace in our lives. Hands-on aerial robotics courses are pivotal in training the next-generation workforce to meet the growing market demands. Such an efficient and compelling course depends on a reliable testbed. In this paper, we present \textit{VizFlyt}, an open-source perception-centric Hardware-In-The-Loop (HITL) photorealistic testing framework for aerial robotics courses. We utilize pose from an external localization system to hallucinate real-time and photorealistic visual sensors using 3D Gaussian Splatting. This enables stress-free testing of autonomy algorithms on aerial robots without the risk of crashing into obstacles. We achieve over 100Hz of system update rate. Lastly, we build upon our past experiences of offering hands-on aerial robotics courses and propose a new open-source and open-hardware curriculum based on \textit{VizFlyt} for the future. We test our framework on various course projects in real-world HITL experiments and present the results showing the efficacy of such a system and its large potential use cases. Code, datasets, hardware guides and demo videos are available at this https URL 

**Abstract (ZH)**: 自主飞行机器人越来越多地融入我们的生活。动手飞行机器人课程是培养下一代劳动力以满足不断增长的市场需求的关键。这样高效且引人入胜的课程依赖于一个可靠的测试平台。本文介绍了VizFlyt，一个开源感知中心的硬件在环（HITL）光现实测试框架，用于飞行机器人课程。我们使用外部定位系统的姿态来实时和光现实地生成视觉传感器，利用3D正态斑点图。这使得在飞行机器人上无压力地测试自主算法，而不用担心碰撞到障碍物。我们实现了超过100Hz的系统更新率。最后，我们基于VizFlyt并结合我们过往提供动手飞行机器人课程的经验，提出了一种新的开源和开源硬件课程。我们对各种课程项目在现实世界的HITL实验中测试了该框架，并展示了该系统的有效性及其广泛的应用前景。相关代码、数据集、硬件指南和演示视频请访问此链接。 

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

**Abstract (ZH)**: 基于图神经网络的颗粒堆积制造过程表面预测方法 

---
# COSMO: Combination of Selective Memorization for Low-cost Vision-and-Language Navigation 

**Title (ZH)**: COSMO：低成本视觉与语言导航的选择性记忆结合 

**Authors**: Siqi Zhang, Yanyuan Qiao, Qunbo Wang, Zike Yan, Qi Wu, Zhihua Wei, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.24065)  

**Abstract**: Vision-and-Language Navigation (VLN) tasks have gained prominence within artificial intelligence research due to their potential application in fields like home assistants. Many contemporary VLN approaches, while based on transformer architectures, have increasingly incorporated additional components such as external knowledge bases or map information to enhance performance. These additions, while boosting performance, also lead to larger models and increased computational costs. In this paper, to achieve both high performance and low computational costs, we propose a novel architecture with the COmbination of Selective MemOrization (COSMO). Specifically, COSMO integrates state-space modules and transformer modules, and incorporates two VLN-customized selective state space modules: the Round Selective Scan (RSS) and the Cross-modal Selective State Space Module (CS3). RSS facilitates comprehensive inter-modal interactions within a single scan, while the CS3 module adapts the selective state space module into a dual-stream architecture, thereby enhancing the acquisition of cross-modal interactions. Experimental validations on three mainstream VLN benchmarks, REVERIE, R2R, and R2R-CE, not only demonstrate competitive navigation performance of our model but also show a significant reduction in computational costs. 

**Abstract (ZH)**: Vision-and-Language Navigation (VLN) 任务在人工智能研究中由于其在家庭助手等领域潜在的应用获得了广泛关注。许多现代 VLN 方法虽然基于变压器架构，但越来越多地融入了外部知识库或地图信息等额外组件以提升性能。这些添加确实提高了性能，但也导致了模型规模的扩大和计算成本的增加。本文为在保持高性能的同时降低计算成本，我们提出了一种结合选择性记忆的新架构（COSMO）。具体而言，COSMO 结合了状态空间模块和变压器模块，并引入了两个定制化的 VLN 选择性状态空间模块：Round Selective Scan (RSS) 和 Cross-modal Selective State Space Module (CS3)。RSS 促进了单一扫描内的全方位跨模态交互，而 CS3 模块将选择性状态空间模块转化为双流架构，从而增强了跨模态交互的获取能力。在三个主流 VLN 验证平台上（REVERIE、R2R 和 R2R-CE）的实验验证不仅展示了我们模型的竞争力，还展示了计算成本的显著降低。 

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
# Task Tokens: A Flexible Approach to Adapting Behavior Foundation Models 

**Title (ZH)**: 任务令牌：一种灵活的行为基础模型适应方法 

**Authors**: Ron Vainshtein, Zohar Rimon, Shie Mannor, Chen Tessler  

**Link**: [PDF](https://arxiv.org/pdf/2503.22886)  

**Abstract**: Recent advancements in imitation learning have led to transformer-based behavior foundation models (BFMs) that enable multi-modal, human-like control for humanoid agents. While excelling at zero-shot generation of robust behaviors, BFMs often require meticulous prompt engineering for specific tasks, potentially yielding suboptimal results. We introduce "Task Tokens", a method to effectively tailor BFMs to specific tasks while preserving their flexibility. Our approach leverages the transformer architecture of BFMs to learn a new task-specific encoder through reinforcement learning, keeping the original BFM frozen. This allows incorporation of user-defined priors, balancing reward design and prompt engineering. By training a task encoder to map observations to tokens, used as additional BFM inputs, we guide performance improvement while maintaining the model's diverse control characteristics. We demonstrate Task Tokens' efficacy across various tasks, including out-of-distribution scenarios, and show their compatibility with other prompting modalities. Our results suggest that Task Tokens offer a promising approach for adapting BFMs to specific control tasks while retaining their generalization capabilities. 

**Abstract (ZH)**: 近期模仿学习的进展催生了基于变压器的行为基础模型（BFMs），这些模型能够为类人代理提供多模态、类人的控制。尽管BFMs在零-shot生成稳健行为方面表现出色，但对于特定任务，它们通常需要细致的提示工程，可能导致次优结果。我们引入了“任务令牌”方法，以有效适应BFMs到特定任务的同时保持其灵活性。我们的方法利用BFMs的变压器架构，通过强化学习学习一个新的任务特定编码器，同时冻结原始BFM。这允许整合用户定义的先验知识，平衡奖励设计和提示工程。通过训练任务编码器将观察值映射到令牌，作为BFM的额外输入，我们可以在保持模型多样化控制特性的同时引导性能改进。我们展示了任务令牌在多种任务中的有效性，包括分布外场景，并表明它们与其他提示方法兼容。我们的结果表明，任务令牌提供了一种有前景的方法来适应BFMs以进行特定控制任务，同时保留其泛化能力。 

---
# RIG: Synergizing Reasoning and Imagination in End-to-End Generalist Policy 

**Title (ZH)**: RIG: 结合推理与想象的端到端通用策略 

**Authors**: Zhonghan Zhao, Wenwei Zhang, Haian Huang, Kuikun Liu, Jianfei Gao, Gaoang Wang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.24388)  

**Abstract**: Reasoning before action and imagining potential outcomes (i.e., world models) are essential for embodied agents operating in complex open-world environments. Yet, prior work either incorporates only one of these abilities in an end-to-end agent or integrates multiple specialized models into an agent system, limiting the learning efficiency and generalization of the policy. Thus, this paper makes the first attempt to synergize Reasoning and Imagination in an end-to-end Generalist policy, termed RIG. To train RIG in an end-to-end manner, we construct a data pipeline that progressively integrates and enriches the content of imagination and reasoning in the trajectories collected from existing agents. The joint learning of reasoning and next image generation explicitly models the inherent correlation between reasoning, action, and dynamics of environments, and thus exhibits more than $17\times$ sample efficiency improvements and generalization in comparison with previous works. During inference, RIG first reasons about the next action, produces potential action, and then predicts the action outcomes, which offers the agent a chance to review and self-correct based on the imagination before taking real actions. Experimental results show that the synergy of reasoning and imagination not only improves the robustness, generalization, and interoperability of generalist policy but also enables test-time scaling to enhance overall performance. 

**Abstract (ZH)**: 推理在先并想象潜在结果（即世界模型）对于在复杂开放环境中的 bodily代理至关重要。然而，先前的工作要么仅在一个端到端代理中结合了其中一种能力，要么将多种专门的模型集成到代理系统中，限制了策略的学习效率和泛化能力。因此，本文首次尝试在端到端的一般主义策略中结合推理和想象，称为RIG。为了以端到端的方式训练RIG，我们构建了一条数据管道，逐步整合和丰富现有代理收集的轨迹中的想象和推理内容。推理和下一个图像生成的联合学习明确地建模了推理、动作和环境动力学之间的内在关联，从而在样本效率和泛化方面比先前的工作提高了超过17倍。在推理过程中，RIG首先进行推理以确定下一个动作，生成潜在动作，并预测行动结果，这使代理有机会在采取实际行动之前根据想象进行回顾和自我纠正。实验结果表明，推理和想象的结合不仅提高了通用策略的鲁棒性、泛化能力和互操作性，还通过测试时的扩展提高了整体性能。 

---
# Grounding Agent Reasoning in Image Schemas: A Neurosymbolic Approach to Embodied Cognition 

**Title (ZH)**: 基于图像模式的地基代理推理：一种神经符号方法的体表认知研究 

**Authors**: François Olivier, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2503.24110)  

**Abstract**: Despite advances in embodied AI, agent reasoning systems still struggle to capture the fundamental conceptual structures that humans naturally use to understand and interact with their environment. To address this, we propose a novel framework that bridges embodied cognition theory and agent systems by leveraging a formal characterization of image schemas, which are defined as recurring patterns of sensorimotor experience that structure human cognition. By customizing LLMs to translate natural language descriptions into formal representations based on these sensorimotor patterns, we will be able to create a neurosymbolic system that grounds the agent's understanding in fundamental conceptual structures. We argue that such an approach enhances both efficiency and interpretability while enabling more intuitive human-agent interactions through shared embodied understanding. 

**Abstract (ZH)**: 尽管在具身AI方面取得了进展，代理推理系统仍难以捕捉人类自然用于理解及其与环境互动的基本概念结构。为解决这一问题，我们提出了一种新型框架，该框架通过利用形象图式的形式化特征，将具身认知理论与代理系统相结合。形象图式被定义为传感器运动体验中的重复模式，这些模式构成了人类认知的基础。通过定制LLMs将自然语言描述转化为基于这些传感器运动模式的形式化表示，我们可以创建一个神经符号系统，使代理的理解扎根于基本概念结构。我们认为，这种做法既能提高效率和可解释性，又能通过共享的具身理解实现更直观的人机交互。 

---
# AI2Agent: An End-to-End Framework for Deploying AI Projects as Autonomous Agents 

**Title (ZH)**: AI2Agent：将AI项目部署为自主代理的端到端框架 

**Authors**: Jiaxiang Chen, Jingwei Shi, Lei Gan, Jiale Zhang, Qingyu Zhang, Dongqian Zhang, Xin Pang, Zhucong Li, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.23948)  

**Abstract**: As AI technology advances, it is driving innovation across industries, increasing the demand for scalable AI project deployment. However, deployment remains a critical challenge due to complex environment configurations, dependency conflicts, cross-platform adaptation, and debugging difficulties, which hinder automation and adoption. This paper introduces AI2Agent, an end-to-end framework that automates AI project deployment through guideline-driven execution, self-adaptive debugging, and case \& solution accumulation. AI2Agent dynamically analyzes deployment challenges, learns from past cases, and iteratively refines its approach, significantly reducing human intervention. To evaluate its effectiveness, we conducted experiments on 30 AI deployment cases, covering TTS, text-to-image generation, image editing, and other AI applications. Results show that AI2Agent significantly reduces deployment time and improves success rates. The code and demo video are now publicly accessible. 

**Abstract (ZH)**: 随着人工智能技术的发展，它正推动各个行业的创新，增加可扩展人工智能项目部署的需求。然而，部署仍是一个关键挑战，由于复杂环境配置、依赖冲突、跨平台适应性和调试困难，这阻碍了自动化和推广应用。本文介绍了AI2Agent，这是一个端到端框架，通过指南驱动执行、自我适应调试和案例及解决方案积累来自动化人工智能项目部署。AI2Agent动态分析部署挑战，从过往案例中学习，并迭代优化其方法，显著减少人工干预。为了评估其有效性，我们在30个人工智能部署案例上进行了实验，涵盖TTS、文本-to-图像生成、图像编辑和其他人工智能应用。结果显示，AI2Agent显著减少了部署时间和提高了成功率。目前该代码和演示视频已公开。 

---
# GIScience in the Era of Artificial Intelligence: A Research Agenda Towards Autonomous GIS 

**Title (ZH)**: 人工智能时代的GIScience：通往自主GIS的研究议程 

**Authors**: Zhenlong Li, Huan Ning, Song Gao, Krzysztof Janowicz, Wenwen Li, Samantha T. Arundel, Chaowei Yang, Budhendra Bhaduri, Shaowen Wang, A-Xing Zhu, Mark Gahegan, Shashi Shekhar, Xinyue Ye, Grant McKenzie, Guido Cervone, Michael E. Hodgson  

**Link**: [PDF](https://arxiv.org/pdf/2503.23633)  

**Abstract**: The advent of generative AI exemplified by large language models (LLMs) opens new ways to represent and compute geographic information and transcend the process of geographic knowledge production, driving geographic information systems (GIS) towards autonomous GIS. Leveraging LLMs as the decision core, autonomous GIS can independently generate and execute geoprocessing workflows to perform spatial analysis. In this vision paper, we elaborate on the concept of autonomous GIS and present a framework that defines its five autonomous goals, five levels of autonomy, five core functions, and three operational scales. We demonstrate how autonomous GIS could perform geospatial data retrieval, spatial analysis, and map making with four proof-of-concept GIS agents. We conclude by identifying critical challenges and future research directions, including fine-tuning and self-growing decision cores, autonomous modeling, and examining the ethical and practical implications of autonomous GIS. By establishing the groundwork for a paradigm shift in GIScience, this paper envisions a future where GIS moves beyond traditional workflows to autonomously reason, derive, innovate, and advance solutions to pressing global challenges. 

**Abstract (ZH)**: 生成式AI的兴起，以大规模语言模型（LLMs）为代表，为地理信息的表示和计算开辟了新途径，超越了地理知识生产的过程，推动地理信息系统（GIS）向自主GIS转型。利用大规模语言模型作为决策核心，自主GIS可以独立生成和执行地理处理工作流，进行空间分析。在本文中，我们阐释了自主GIS的概念，并提出一个框架，定义了其五个自主目标、五个自主层级、五个核心功能以及三个运作尺度。我们通过四个概念性的GIS代理展示了自主GIS如何进行地理空间数据检索、空间分析和制图。最后，我们指出了关键挑战和未来研究方向，包括精细调整和自我成长的决策核心、自主建模以及探讨自主GIS的伦理与实际影响。通过为GIS科学确立范式转移的基础，本文构想了一个未来，GIS将超越传统的工作流程，自主推理、推导、创新并解决紧迫的全球挑战。 

---
# AstroAgents: A Multi-Agent AI for Hypothesis Generation from Mass Spectrometry Data 

**Title (ZH)**: AstroAgents: 多智能体AI在质谱数据分析中的假说生成 

**Authors**: Daniel Saeedi, Denise Buckner, Jose C. Aponte, Amirali Aghazadeh  

**Link**: [PDF](https://arxiv.org/pdf/2503.23170)  

**Abstract**: With upcoming sample return missions across the solar system and the increasing availability of mass spectrometry data, there is an urgent need for methods that analyze such data within the context of existing astrobiology literature and generate plausible hypotheses regarding the emergence of life on Earth. Hypothesis generation from mass spectrometry data is challenging due to factors such as environmental contaminants, the complexity of spectral peaks, and difficulties in cross-matching these peaks with prior studies. To address these challenges, we introduce AstroAgents, a large language model-based, multi-agent AI system for hypothesis generation from mass spectrometry data. AstroAgents is structured around eight collaborative agents: a data analyst, a planner, three domain scientists, an accumulator, a literature reviewer, and a critic. The system processes mass spectrometry data alongside user-provided research papers. The data analyst interprets the data, and the planner delegates specific segments to the scientist agents for in-depth exploration. The accumulator then collects and deduplicates the generated hypotheses, and the literature reviewer identifies relevant literature using Semantic Scholar. The critic evaluates the hypotheses, offering rigorous suggestions for improvement. To assess AstroAgents, an astrobiology expert evaluated the novelty and plausibility of more than a hundred hypotheses generated from data obtained from eight meteorites and ten soil samples. Of these hypotheses, 36% were identified as plausible, and among those, 66% were novel. Project website: this https URL 

**Abstract (ZH)**: 随着太阳系内取样返回任务的即将到来以及质谱数据分析的日益增多，迫切需要能够将此类数据纳入现有天体生物学文献分析的方法，并生成关于生命在地球上出现的合理假设。基于质谱数据的假设生成极具挑战性，原因包括环境污染物、谱峰复杂性以及跨研究匹配这些峰值的困难。为解决这些挑战，我们引入了AstroAgents——一个多智能体AI系统，基于大型语言模型，用于从质谱数据中生成假设。AstroAgents由八个协作智能体构成：数据分析师、规划师、三位领域科学家、积累器、文献审查员和批判者。该系统在处理质谱数据的同时，还能与用户提供的研究论文一同工作。数据分析师解读数据，规划师将具体的任务分配给科学家智能体进行深入探索。积累器收集并去重生成的假设，文献审查员使用Semantic Scholar识别相关文献，批判者评估假设，提供严格的改进建议。为了评估AstroAgents，一位天体生物学专家评估了从八块陨石和十份土壤样本的数据中生成的超过一百个假设的新颖性和合理性。在这其中，36%被认定为合理，而在这其中又有66%是新颖的。项目网站：https://this-url。 

---
# Noise-based reward-modulated learning 

**Title (ZH)**: 基于噪声的奖励调节学习 

**Authors**: Jesús García Fernández, Nasir Ahmad, Marcel van Gerven  

**Link**: [PDF](https://arxiv.org/pdf/2503.23972)  

**Abstract**: Recent advances in reinforcement learning (RL) have led to significant improvements in task performance. However, training neural networks in an RL regime is typically achieved in combination with backpropagation, limiting their applicability in resource-constrained environments or when using non-differentiable neural networks. While noise-based alternatives like reward-modulated Hebbian learning (RMHL) have been proposed, their performance has remained limited, especially in scenarios with delayed rewards, which require retrospective credit assignment over time. Here, we derive a novel noise-based learning rule that addresses these challenges. Our approach combines directional derivative theory with Hebbian-like updates to enable efficient, gradient-free learning in RL. It features stochastic noisy neurons which can approximate gradients, and produces local synaptic updates modulated by a global reward signal. Drawing on concepts from neuroscience, our method uses reward prediction error as its optimization target to generate increasingly advantageous behavior, and incorporates an eligibility trace to facilitate temporal credit assignment in environments with delayed rewards. Its formulation relies on local information alone, making it compatible with implementations in neuromorphic hardware. Experimental validation shows that our approach significantly outperforms RMHL and is competitive with BP-based baselines, highlighting the promise of noise-based, biologically inspired learning for low-power and real-time applications. 

**Abstract (ZH)**: 最近 reinforcement learning 的进展显著提高了任务性能，但在资源受限环境中或使用非可微神经网络时，通过反向传播训练神经网络的限制使其应用受到了限制。尽管提出了基于噪声的方法，如奖励调制的希布bian学习(RMHL)，但在延迟奖励等情境中，其性能仍然有限，这要求在时间上进行追溯的信用分配。在这里，我们推导出一种新的基于噪声的学习规则，以应对这些挑战。我们的方法结合了方向导数理论和希布bian-like 更新，能够在 reinforcement learning 中实现高效、无梯度的学习。该方法采用具有噪声的随机神经元来近似梯度，并通过全局奖励信号调节局部突触更新。借鉴神经科学的概念，我们的方法将奖励预测误差用作优化目标，以生成更有利的行为，并结合了弹性迹线来促进延迟奖励环境中时间上的信用分配。该方法的表达式仅依赖于局部信息，使其与神经形态硬件的实现兼容。实验验证表明，我们的方法显著优于 RMHL，并在基于反向传播的基准方法中表现出竞争力，展示了基于噪声的、生物启发的学习方法在低功耗和实时应用中的潜力。 

---
# Green MLOps to Green GenOps: An Empirical Study of Energy Consumption in Discriminative and Generative AI Operations 

**Title (ZH)**: 从绿色MLOps到绿色GenOps：一种区分性和生成性AI操作的能源消耗实证研究 

**Authors**: Adrián Sánchez-Mompó, Ioannis Mavromatis, Peizheng Li, Konstantinos Katsaros, Aftab Khan  

**Link**: [PDF](https://arxiv.org/pdf/2503.23934)  

**Abstract**: This study presents an empirical investigation into the energy consumption of Discriminative and Generative AI models within real-world MLOps pipelines. For Discriminative models, we examine various architectures and hyperparameters during training and inference and identify energy-efficient practices. For Generative AI, Large Language Models (LLMs) are assessed, focusing primarily on energy consumption across different model sizes and varying service requests. Our study employs software-based power measurements, ensuring ease of replication across diverse configurations, models, and datasets. We analyse multiple models and hardware setups to uncover correlations among various metrics, identifying key contributors to energy consumption. The results indicate that for Discriminative models, optimising architectures, hyperparameters, and hardware can significantly reduce energy consumption without sacrificing performance. For LLMs, energy efficiency depends on balancing model size, reasoning complexity, and request-handling capacity, as larger models do not necessarily consume more energy when utilisation remains low. This analysis provides practical guidelines for designing green and sustainable ML operations, emphasising energy consumption and carbon footprint reductions while maintaining performance. This paper can serve as a benchmark for accurately estimating total energy use across different types of AI models. 

**Abstract (ZH)**: 本研究对实际MLOps管道中判别性和生成性AI模型的能源消耗进行了实证调查。对于判别性模型，我们 examining 各种训练和推理架构及超参数，并识别能源高效实践。对于生成性AI，主要评估大型语言模型（LLMs）在不同模型规模和服务请求变化下的能源消耗。本研究采用基于软件的电源测量方法，确保在不同配置、模型和数据集上轻松复制。我们分析多种模型和硬件组合以揭示各种指标之间的关联，识别能源消耗的关键因素。研究结果表明，对于判别性模型，通过优化架构、超参数和硬件可以显著减少能源消耗，而不牺牲性能。对于LLMs，能源效率取决于平衡模型规模、推理复杂性和请求处理能力，利用率较低时，更大规模的模型不一定消耗更多能源。本分析提供了设计绿色可持续ML操作的实用指南，强调减少能源消耗和碳足迹的同时保持性能。本论文可作为准确估计不同AI模型类型总能源使用量的基准。 

---
# Handling Delay in Real-Time Reinforcement Learning 

**Title (ZH)**: 处理实时强化学习中的延迟 

**Authors**: Ivan Anokhin, Rishav Rishav, Matthew Riemer, Stephen Chung, Irina Rish, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2503.23478)  

**Abstract**: Real-time reinforcement learning (RL) introduces several challenges. First, policies are constrained to a fixed number of actions per second due to hardware limitations. Second, the environment may change while the network is still computing an action, leading to observational delay. The first issue can partly be addressed with pipelining, leading to higher throughput and potentially better policies. However, the second issue remains: if each neuron operates in parallel with an execution time of $\tau$, an $N$-layer feed-forward network experiences observation delay of $\tau N$. Reducing the number of layers can decrease this delay, but at the cost of the network's expressivity. In this work, we explore the trade-off between minimizing delay and network's expressivity. We present a theoretically motivated solution that leverages temporal skip connections combined with history-augmented observations. We evaluate several architectures and show that those incorporating temporal skip connections achieve strong performance across various neuron execution times, reinforcement learning algorithms, and environments, including four Mujoco tasks and all MinAtar games. Moreover, we demonstrate parallel neuron computation can accelerate inference by 6-350% on standard hardware. Our investigation into temporal skip connections and parallel computations paves the way for more efficient RL agents in real-time setting. 

**Abstract (ZH)**: 实时强化学习中的延迟与网络表征能力 Trade-off探究：基于时间跳跃连接的历史增强观测方法及其应用 

---
# Localized Graph-Based Neural Dynamics Models for Terrain Manipulation 

**Title (ZH)**: 基于局部图的神经动力学模型在地形操纵中的应用 

**Authors**: Chaoqi Liu, Yunzhu Li, Kris Hauser  

**Link**: [PDF](https://arxiv.org/pdf/2503.23270)  

**Abstract**: Predictive models can be particularly helpful for robots to effectively manipulate terrains in construction sites and extraterrestrial surfaces. However, terrain state representations become extremely high-dimensional especially to capture fine-resolution details and when depth is unknown or unbounded. This paper introduces a learning-based approach for terrain dynamics modeling and manipulation, leveraging the Graph-based Neural Dynamics (GBND) framework to represent terrain deformation as motion of a graph of particles. Based on the principle that the moving portion of a terrain is usually localized, our approach builds a large terrain graph (potentially millions of particles) but only identifies a very small active subgraph (hundreds of particles) for predicting the outcomes of robot-terrain interaction. To minimize the size of the active subgraph we introduce a learning-based approach that identifies a small region of interest (RoI) based on the robot's control inputs and the current scene. We also introduce a novel domain boundary feature encoding that allows GBNDs to perform accurate dynamics prediction in the RoI interior while avoiding particle penetration through RoI boundaries. Our proposed method is both orders of magnitude faster than naive GBND and it achieves better overall prediction accuracy. We further evaluated our framework on excavation and shaping tasks on terrain with different granularity. 

**Abstract (ZH)**: 基于图的神经动力学学习方法在地形动力学建模与操控中的应用 

---
# Simulation of Non-Ordinary Consciousness 

**Title (ZH)**: 非普通意识的模拟 

**Authors**: Khalid M. Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2503.23245)  

**Abstract**: The symbolic architecture of non-ordinary consciousness remains largely unmapped in cognitive science and artificial intelligence. While conventional models prioritize rational coherence, altered states such as those induced by psychedelics reveal distinct symbolic regimes characterized by recursive metaphor, ego dissolution, and semantic destabilization. We present \textit{Glyph}, a generative symbolic interface designed to simulate psilocybin-like symbolic cognition in large language models. Rather than modeling perception or mood, Glyph enacts symbolic transformation through recursive reentry, metaphoric modulation, and entropy-scaled destabilization -- a triadic operator formalized within a tensorial linguistic framework. Experimental comparison with baseline GPT-4o reveals that Glyph consistently generates high-entropy, metaphor-saturated, and ego-dissolving language across diverse symbolic prompt categories. These results validate the emergence of non-ordinary cognitive patterns and support a new paradigm for simulating altered consciousness through language. Glyph opens novel pathways for modeling symbolic cognition, exploring metaphor theory, and encoding knowledge in recursively altered semantic spaces. 

**Abstract (ZH)**: 非普通意识的象征架构在认知科学和人工智能中仍然 largely unmapped。常规模型侧重于理性连贯性，而致幻剂诱导的改变认知状态则表现出递归隐喻、自我解体和语义不稳定等独特的象征制度。我们提出了Glyph，这是一种生成性象征接口，旨在模拟类似仙人掌菇的象征认知模式在大型语言模型中。Glyph 不建模感知或情绪，而是通过递归重新输入、隐喻调节和熵缩放不稳定来实现象征性转化——这种三元操作在一个张量语言框架内进行了形式化。与基准GPT-4o的实验比较显示，Glyph 在各类象征性提示类别中始终生成高熵、富含隐喻且自我解体的语言。这些结果验证了非普通认知模式的出现，并支持通过语言模拟改变认识的新范式。Glyph 开辟了建模象征认知、探索隐喻理论以及在递归改変的语义空间中编码知识的新途径。 

---
# Action Recognition in Real-World Ambient Assisted Living Environment 

**Title (ZH)**: 实时辅助生活环境中的人体动作识别 

**Authors**: Vincent Gbouna Zakka, Zhuangzhuang Dai, Luis J. Manso  

**Link**: [PDF](https://arxiv.org/pdf/2503.23214)  

**Abstract**: The growing ageing population and their preference to maintain independence by living in their own homes require proactive strategies to ensure safety and support. Ambient Assisted Living (AAL) technologies have emerged to facilitate ageing in place by offering continuous monitoring and assistance within the home. Within AAL technologies, action recognition plays a crucial role in interpreting human activities and detecting incidents like falls, mobility decline, or unusual behaviours that may signal worsening health conditions. However, action recognition in practical AAL applications presents challenges, including occlusions, noisy data, and the need for real-time performance. While advancements have been made in accuracy, robustness to noise, and computation efficiency, achieving a balance among them all remains a challenge. To address this challenge, this paper introduces the Robust and Efficient Temporal Convolution network (RE-TCN), which comprises three main elements: Adaptive Temporal Weighting (ATW), Depthwise Separable Convolutions (DSC), and data augmentation techniques. These elements aim to enhance the model's accuracy, robustness against noise and occlusion, and computational efficiency within real-world AAL contexts. RE-TCN outperforms existing models in terms of accuracy, noise and occlusion robustness, and has been validated on four benchmark datasets: NTU RGB+D 60, Northwestern-UCLA, SHREC'17, and DHG-14/28. The code is publicly available at: this https URL 

**Abstract (ZH)**: 不断增长的老龄人口及其倾向于在家保持独立的需求需要采取积极策略确保安全和支持。辅助生活技术(AAL)已 emergence 以通过在家中提供持续监测和支持来促进原居安老。在 AAL 技术中，动作识别在解释人类活动和检测跌倒、移动能力下降或不寻常行为等方面发挥着关键作用，这些行为可能预示着健康状况的恶化。然而，在实际 AAL 应用中进行动作识别面临挑战，包括遮挡、噪声数据以及对实时性能的需求。尽管在准确性、抗噪声能力和计算效率方面已取得进展，但在这三者之间实现平衡仍然具有挑战性。为应对这一挑战，本文引入了一种 robust and efficient temporal convolution 网络(RE-TCN)，它包含三个主要元素：自适应时间加权(ATW)、深度可分离卷积(DSC)以及数据增强技术。这些元素旨在在实际 AAL 情境中增强模型的准确性、遮挡和噪声的鲁棒性以及计算效率。RE-TCN 在准确性、对遮挡和噪声的鲁棒性方面优于现有模型，并已在四个基准数据集中得到了验证：NTU RGB+D 60、Northwestern-UCLA、SHREC'17 和 DHG-14/28。代码已公开可用：https://github.com/your-repo-url。 

---
# Predictive Traffic Rule Compliance using Reinforcement Learning 

**Title (ZH)**: 基于强化学习的预测性交通规则遵守性研究 

**Authors**: Yanliang Huang, Sebastian Mair, Zhuoqi Zeng, Amr Alanwar, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2503.22925)  

**Abstract**: Autonomous vehicle path planning has reached a stage where safety and regulatory compliance are crucial. This paper presents a new approach that integrates a motion planner with a deep reinforcement learning model to predict potential traffic rule violations. In this setup, the predictions of the critic directly affect the cost function of the motion planner, guiding the choices of the trajectory. We incorporate key interstate rules from the German Road Traffic Regulation into a rule book and use a graph-based state representation to handle complex traffic information. Our main innovation is replacing the standard actor network in an actor-critic setup with a motion planning module, which ensures both predictable trajectory generation and prevention of long-term rule violations. Experiments on an open German highway dataset show that the model can predict and prevent traffic rule violations beyond the planning horizon, significantly increasing safety in challenging traffic conditions. 

**Abstract (ZH)**: 自主驾驶车辆路径规划中的运动规划与深度强化学习模型集成方法：关键交通规则预测与合规性保障 

---
# Why Representation Engineering Works: A Theoretical and Empirical Study in Vision-Language Models 

**Title (ZH)**: 为什么表示工程有效：视觉-语言模型中的理论与实证研究 

**Authors**: Bowei Tian, Xuntao Lyu, Meng Liu, Hongyi Wang, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.22720)  

**Abstract**: Representation Engineering (RepE) has emerged as a powerful paradigm for enhancing AI transparency by focusing on high-level representations rather than individual neurons or circuits. It has proven effective in improving interpretability and control, showing that representations can emerge, propagate, and shape final model outputs in large language models (LLMs). However, in Vision-Language Models (VLMs), visual input can override factual linguistic knowledge, leading to hallucinated responses that contradict reality. To address this challenge, we make the first attempt to extend RepE to VLMs, analyzing how multimodal representations are preserved and transformed. Building on our findings and drawing inspiration from successful RepE applications, we develop a theoretical framework that explains the stability of neural activity across layers using the principal eigenvector, uncovering the underlying mechanism of RepE. We empirically validate these instrinsic properties, demonstrating their broad applicability and significance. By bridging theoretical insights with empirical validation, this work transforms RepE from a descriptive tool into a structured theoretical framework, opening new directions for improving AI robustness, fairness, and transparency. 

**Abstract (ZH)**: Representation Engineering Extension to Vision-Language Models: Analyzing and Explaining Stability Mechanisms for Improved Robustness, Fairness, and Transparency 

---
