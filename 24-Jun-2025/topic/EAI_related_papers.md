# MinD: Unified Visual Imagination and Control via Hierarchical World Models 

**Title (ZH)**: MinD: 统一的层级世界观下的视觉想象与控制 

**Authors**: Xiaowei Chi, Kuangzhi Ge, Jiaming Liu, Siyuan Zhou, Peidong Jia, Zichen He, Yuzhen Liu, Tingguang Li, Lei Han, Sirui Han, Shanghang Zhang, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18897)  

**Abstract**: Video generation models (VGMs) offer a promising pathway for unified world modeling in robotics by integrating simulation, prediction, and manipulation. However, their practical application remains limited due to (1) slowgeneration speed, which limits real-time interaction, and (2) poor consistency between imagined videos and executable actions. To address these challenges, we propose Manipulate in Dream (MinD), a hierarchical diffusion-based world model framework that employs a dual-system design for vision-language manipulation. MinD executes VGM at low frequencies to extract video prediction features, while leveraging a high-frequency diffusion policy for real-time interaction. This architecture enables low-latency, closed-loop control in manipulation with coherent visual guidance. To better coordinate the two systems, we introduce a video-action diffusion matching module (DiffMatcher), with a novel co-training strategy that uses separate schedulers for each diffusion model. Specifically, we introduce a diffusion-forcing mechanism to DiffMatcher that aligns their intermediate representations during training, helping the fast action model better understand video-based predictions. Beyond manipulation, MinD also functions as a world simulator, reliably predicting task success or failure in latent space before execution. Trustworthy analysis further shows that VGMs can preemptively evaluate task feasibility and mitigate risks. Extensive experiments across multiple benchmarks demonstrate that MinD achieves state-of-the-art manipulation (63%+) in RL-Bench, advancing the frontier of unified world modeling in robotics. 

**Abstract (ZH)**: 基于生成模型的级联控制框架：从梦境中操控 

---
# SViP: Sequencing Bimanual Visuomotor Policies with Object-Centric Motion Primitives 

**Title (ZH)**: SViP: 基于物体中心运动模块的双手视觉运动策略序列表征 

**Authors**: Yizhou Chen, Hang Xu, Dongjie Yu, Zeqing Zhang, Yi Ren, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18825)  

**Abstract**: Imitation learning (IL), particularly when leveraging high-dimensional visual inputs for policy training, has proven intuitive and effective in complex bimanual manipulation tasks. Nonetheless, the generalization capability of visuomotor policies remains limited, especially when small demonstration datasets are available. Accumulated errors in visuomotor policies significantly hinder their ability to complete long-horizon tasks. To address these limitations, we propose SViP, a framework that seamlessly integrates visuomotor policies into task and motion planning (TAMP). SViP partitions human demonstrations into bimanual and unimanual operations using a semantic scene graph monitor. Continuous decision variables from the key scene graph are employed to train a switching condition generator. This generator produces parameterized scripted primitives that ensure reliable performance even when encountering out-of-the-distribution observations. Using only 20 real-world demonstrations, we show that SViP enables visuomotor policies to generalize across out-of-distribution initial conditions without requiring object pose estimators. For previously unseen tasks, SViP automatically discovers effective solutions to achieve the goal, leveraging constraint modeling in TAMP formulism. In real-world experiments, SViP outperforms state-of-the-art generative IL methods, indicating wider applicability for more complex tasks. Project website: this https URL 

**Abstract (ZH)**: 基于知觉运动策略的无缝集成框架（SViP）：实现复杂双臂操作任务的泛化能力 

---
# Learning Physical Systems: Symplectification via Gauge Fixing in Dirac Structures 

**Title (ZH)**: 学习物理系统：通过规范选取实现辛结构的化归 

**Authors**: Aristotelis Papatheodorou, Pranav Vaidhyanathan, Natalia Ares, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2506.18812)  

**Abstract**: Physics-informed deep learning has achieved remarkable progress by embedding geometric priors, such as Hamiltonian symmetries and variational principles, into neural networks, enabling structure-preserving models that extrapolate with high accuracy. However, in systems with dissipation and holonomic constraints, ubiquitous in legged locomotion and multibody robotics, the canonical symplectic form becomes degenerate, undermining the very invariants that guarantee stability and long-term prediction. In this work, we tackle this foundational limitation by introducing Presymplectification Networks (PSNs), the first framework to learn the symplectification lift via Dirac structures, restoring a non-degenerate symplectic geometry by embedding constrained systems into a higher-dimensional manifold. Our architecture combines a recurrent encoder with a flow-matching objective to learn the augmented phase-space dynamics end-to-end. We then attach a lightweight Symplectic Network (SympNet) to forecast constrained trajectories while preserving energy, momentum, and constraint satisfaction. We demonstrate our method on the dynamics of the ANYmal quadruped robot, a challenging contact-rich, multibody system. To the best of our knowledge, this is the first framework that effectively bridges the gap between constrained, dissipative mechanical systems and symplectic learning, unlocking a whole new class of geometric machine learning models, grounded in first principles yet adaptable from data. 

**Abstract (ZH)**: 物理信息深度学习通过将哈密顿对称性和变分原理等几何先验嵌入神经网络中，实现了显著进展，从而构建出保持结构且在高精度下进行外推的模型。然而，在带有耗散和约束条件的系统中，如腿式运动和多体机器人中普遍存在的系统，经典的辛形式变得退化，从而削弱了确保稳定性和长期预测的不变量。本文通过引入 Presymplectification 网络 (PSNs)，即首个利用 Dirac 结构学习辛提升的框架，解决了这一基础限制。我们通过在更高维流形中嵌入受约束系统来恢复非退化的辛几何。我们的架构结合了递归编码器和流匹配目标，以端到端的方式学习扩展相空间的动力学。随后，我们附加了一个轻量级辛网络 (SympNet)，用于预测受约束轨迹的同时保持能量、动量和约束满足。我们展示了该方法在 ANYmal 四足机器人动态中的应用，这是一个具有挑战性的接触丰富、多体系统。据我们所知，这是首个有效连接受约束、耗散机械系统和辛学习的框架，开启了基于第一原理但又可以从数据中适应的几何机器学习模型的新类别。 

---
# PG-LIO: Photometric-Geometric fusion for Robust LiDAR-Inertial Odometry 

**Title (ZH)**: PG-LIO：光度-几何融合的鲁棒激光雷达-惯性定位 

**Authors**: Nikhil Khedekar, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2506.18583)  

**Abstract**: LiDAR-Inertial Odometry (LIO) is widely used for accurate state estimation and mapping which is an essential requirement for autonomous robots. Conventional LIO methods typically rely on formulating constraints from the geometric structure sampled by the LiDAR. Hence, in the lack of geometric structure, these tend to become ill-conditioned (degenerate) and fail. Robustness of LIO to such conditions is a necessity for its broader deployment. To address this, we propose PG-LIO, a real-time LIO method that fuses photometric and geometric information sampled by the LiDAR along with inertial constraints from an Inertial Measurement Unit (IMU). This multi-modal information is integrated into a factor graph optimized over a sliding window for real-time operation. We evaluate PG-LIO on multiple datasets that include both geometrically well-conditioned as well as self-similar scenarios. Our method achieves accuracy on par with state-of-the-art LIO in geometrically well-structured settings while significantly improving accuracy in degenerate cases including against methods that also fuse intensity. Notably, we demonstrate only 1 m drift over a 1 km manually piloted aerial trajectory through a geometrically self-similar tunnel at an average speed of 7.5m/s (max speed 10.8 m/s). For the benefit of the community, we shall also release our source code this https URL 

**Abstract (ZH)**: 基于 photometric 和几何信息的实时 LiDAR-惯性 odometry (PG-LIO) 

---
# Mirror Eyes: Explainable Human-Robot Interaction at a Glance 

**Title (ZH)**: 镜像 eyes：一瞥可懂的人机交互 

**Authors**: Matti Krüger, Daniel Tanneberg, Chao Wang, Stephan Hasler, Michael Gienger  

**Link**: [PDF](https://arxiv.org/pdf/2506.18466)  

**Abstract**: The gaze of a person tends to reflect their interest. This work explores what happens when this statement is taken literally and applied to robots. Here we present a robot system that employs a moving robot head with a screen-based eye model that can direct the robot's gaze to points in physical space and present a reflection-like mirror image of the attended region on top of each eye. We conducted a user study with 33 participants, who were asked to instruct the robot to perform pick-and-place tasks, monitor the robot's task execution, and interrupt it in case of erroneous actions. Despite a deliberate lack of instructions about the role of the eyes and a very brief system exposure, participants felt more aware about the robot's information processing, detected erroneous actions earlier, and rated the user experience higher when eye-based mirroring was enabled compared to non-reflective eyes. These results suggest a beneficial and intuitive utilization of the introduced method in cooperative human-robot interaction. 

**Abstract (ZH)**: Humans的凝视倾向于反映他们的兴趣。本文探讨了将这一陈述字面上应用于机器人会发生什么。本文介绍了一种机器人系统，该系统采用可移动的机器人头部和基于屏幕的眼睛模型，能够将机器人的凝视指向物理空间中的点，并在每个眼睛上方呈现类似镜子的注意区域的反射图像。我们进行了一项包含33名参与者的用户研究，要求参与者指导机器人执行拾取和放置任务，监控机器人的任务执行情况，并在发生错误操作时中断任务。尽管参与者没有关于眼睛作用的明确指示且系统曝光时间很短，但在启用基于眼睛的镜像反射功能时，参与者对机器人信息处理的感知更为清晰，更早地检测到错误操作，并且对用户体验的评价更高。这些结果表明，在协作的人机交互中引入该方法具有有益且直观的利用方式。 

---
# A Motivational Architecture for Open-Ended Learning Challenges in Robots 

**Title (ZH)**: 一种用于机器人开放性学习挑战的动机架构 

**Authors**: Alejandro Romero, Gianluca Baldassarre, Richard J. Duro, Vieri Giuliano Santucci  

**Link**: [PDF](https://arxiv.org/pdf/2506.18454)  

**Abstract**: Developing agents capable of autonomously interacting with complex and dynamic environments, where task structures may change over time and prior knowledge cannot be relied upon, is a key prerequisite for deploying artificial systems in real-world settings. The open-ended learning framework identifies the core challenges for creating such agents, including the ability to autonomously generate new goals, acquire the necessary skills (or curricula of skills) to achieve them, and adapt to non-stationary environments. While many existing works tackles various aspects of these challenges in isolation, few propose integrated solutions that address them simultaneously. In this paper, we introduce H-GRAIL, a hierarchical architecture that, through the use of different typologies of intrinsic motivations and interconnected learning mechanisms, autonomously discovers new goals, learns the required skills for their achievement, generates skill sequences for tackling interdependent tasks, and adapts to non-stationary environments. We tested H-GRAIL in a real robotic scenario, demonstrating how the proposed solutions effectively address the various challenges of open-ended learning. 

**Abstract (ZH)**: 自主开发能够与复杂且动态环境进行自主交互的智能代理是将人工系统部署到现实世界的关键前提。开放学习框架识别了创建此类代理的核心挑战，包括自主生成新目标、获取实现这些目标所需技能（或技能课程）以及适应非平稳环境的能力。虽然许多现有工作独立解决了这些挑战的各个方面，但很少有工作同时提出了综合解决方案。本文介绍了H-GRAIL层次结构，通过使用不同类型的内在动机和相互连接的学习机制，自主发现新目标、学习实现这些目标所需的技能、生成处理互赖任务的技能序列，并适应非平稳环境。我们在一个真实的机器人场景中测试了H-GRAIL，验证了所提出解决方案如何有效应对开放学习的各种挑战。 

---
# GraspMAS: Zero-Shot Language-driven Grasp Detection with Multi-Agent System 

**Title (ZH)**: GraspMAS: 多智能体系统驱动的零样本语言引导抓取检测 

**Authors**: Quang Nguyen, Tri Le, Huy Nguyen, Thieu Vo, Tung D. Ta, Baoru Huang, Minh N. Vu, Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18448)  

**Abstract**: Language-driven grasp detection has the potential to revolutionize human-robot interaction by allowing robots to understand and execute grasping tasks based on natural language commands. However, existing approaches face two key challenges. First, they often struggle to interpret complex text instructions or operate ineffectively in densely cluttered environments. Second, most methods require a training or finetuning step to adapt to new domains, limiting their generation in real-world applications. In this paper, we introduce GraspMAS, a new multi-agent system framework for language-driven grasp detection. GraspMAS is designed to reason through ambiguities and improve decision-making in real-world scenarios. Our framework consists of three specialized agents: Planner, responsible for strategizing complex queries; Coder, which generates and executes source code; and Observer, which evaluates the outcomes and provides feedback. Intensive experiments on two large-scale datasets demonstrate that our GraspMAS significantly outperforms existing baselines. Additionally, robot experiments conducted in both simulation and real-world settings further validate the effectiveness of our approach. 

**Abstract (ZH)**: 基于语言的抓取检测有望通过使机器人能够理解并基于自然语言命令执行抓取任务来革新人机互动。然而，现有方法面临两个关键挑战。首先，它们往往难以解释复杂的文本指令或在密集的杂乱环境中操作无效。其次，大多数方法需要训练或微调步骤以适应新的领域，限制了其在现实世界中的应用生成能力。在本文中，我们提出了一种新的多agent系统框架GraspMAS，用于基于语言的抓取检测。GraspMAS旨在在真实世界场景中解决歧义并改进决策。我们的框架由三个专门的agent组成：Planner，负责策划复杂的查询；Coder，生成和执行源代码；以及Observer，评估结果并提供反馈。在两个大规模数据集上的密集实验表明，我们的GraspMAS显著优于现有基线。此外，在模拟和现实世界设置中的机器人实验进一步验证了我们方法的有效性。 

---
# Robots and Children that Learn Together : Improving Knowledge Retention by Teaching Peer-Like Interactive Robots 

**Title (ZH)**: 机器人与共同学习的儿童：通过教同伴样交互机器人提高知识保留 

**Authors**: Imene Tarakli, Samuele Vinanzi, Richard Moore, Alessandro Di Nuovo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18365)  

**Abstract**: Despite growing interest in Learning-by-Teaching (LbT), few studies have explored how this paradigm can be implemented with autonomous, peer-like social robots in real classrooms. Most prior work has relied on scripted or Wizard-of-Oz behaviors, limiting our understanding of how real-time, interactive learning can be supported by artificial agents. This study addresses this gap by introducing Interactive Reinforcement Learning (RL) as a cognitive model for teachable social robots. We conducted two between-subject experiments with 58 primary school children, who either taught a robot or practiced independently on a tablet while learning French vocabulary (memorization) and grammatical rules (inference). The robot, powered by Interactive RL, learned from the child's evaluative feedback. Children in the LbT condition achieved significantly higher retention gains compared to those in the self-practice condition, especially on the grammar task. Learners with lower prior knowledge benefited most from teaching the robot. Behavioural metrics revealed that children adapted their teaching strategies over time and engaged more deeply during inference tasks. This work makes two contributions: (1) it introduces Interactive RL as a pedagogically effective and scalable model for peer-robot learning, and (2) it demonstrates, for the first time, the feasibility of deploying multiple autonomous robots simultaneously in real classrooms. These findings extend theoretical understanding of LbT by showing that social robots can function not only as passive tutees but as adaptive partners that enhance meta-cognitive engagement and long-term learning outcomes. 

**Abstract (ZH)**: 尽管学习型教学（LbT）引起了越来越多的兴趣，但在真实课堂中以类似同伴的方式实施自主社会机器人进行学习的研究仍然很少。大多数前期工作依赖于预设或巫师- Oz行为，这限制了我们对即时交互式学习如何由人工代理支持的理解。本研究通过引入交互式强化学习（IRL）作为可教学社会机器人的认知模型来填补这一空白。我们对58名小学生进行了两个被试间实验，他们要么教机器人，要么在学习法语词汇（记忆）和句法规则（推理）的同时独立使用平板电脑练习。机器人由交互式RL驱动，从儿童的评价反馈中学习。在LbT条件下，儿童的表现显著优于自我练习条件，尤其是在句法任务上。知识基础较差的学习者从教机器人中受益最多。行为指标显示，儿童随时间调整了他们的教学策略，并在推理任务中更深入地参与。本研究做出了两个贡献：（1）引入交互式RL作为同伴式机器人学习的有效且可扩展的教育模型；（2）首次展示了在真实教室中同时部署多个自主机器人的可行性。这些发现通过表明社会机器人不仅可以作为被动的学生，还可以作为增强元认知参与和长期学习结果的适应性伙伴，扩展了对LbT的理解。 

---
# Haptic-ACT -- Pseudo Oocyte Manipulation by a Robot Using Multimodal Information and Action Chunking with Transformers 

**Title (ZH)**: 触觉-ACT -- 基于多模态信息和Transformer动作块的类卵细胞操作机器人技术 

**Authors**: Pedro Miguel Uriguen Eljuri, Hironobu Shibata, Maeyama Katsuyoshi, Yuanyuan Jia, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18212)  

**Abstract**: In this paper we introduce Haptic-ACT, an advanced robotic system for pseudo oocyte manipulation, integrating multimodal information and Action Chunking with Transformers (ACT). Traditional automation methods for oocyte transfer rely heavily on visual perception, often requiring human supervision due to biological variability and environmental disturbances. Haptic-ACT enhances ACT by incorporating haptic feedback, enabling real-time grasp failure detection and adaptive correction. Additionally, we introduce a 3D-printed TPU soft gripper to facilitate delicate manipulations. Experimental results demonstrate that Haptic-ACT improves the task success rate, robustness, and adaptability compared to conventional ACT, particularly in dynamic environments. These findings highlight the potential of multimodal learning in robotics for biomedical automation. 

**Abstract (ZH)**: 基于触觉的Haptic-ACT：一种集成多模态信息和Action Chunking的先进卵细胞操作机器人系统 

---
# Integrating LLMs and Digital Twins for Adaptive Multi-Robot Task Allocation in Construction 

**Title (ZH)**: 集成大规模语言模型和数字孪生以实现适应性施工机器人任务分配 

**Authors**: Min Deng, Bo Fu, Lingyao Li, Xi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18178)  

**Abstract**: Multi-robot systems are emerging as a promising solution to the growing demand for productivity, safety, and adaptability across industrial sectors. However, effectively coordinating multiple robots in dynamic and uncertain environments, such as construction sites, remains a challenge, particularly due to unpredictable factors like material delays, unexpected site conditions, and weather-induced disruptions. To address these challenges, this study proposes an adaptive task allocation framework that strategically leverages the synergistic potential of Digital Twins, Integer Programming (IP), and Large Language Models (LLMs). The multi-robot task allocation problem is formally defined and solved using an IP model that accounts for task dependencies, robot heterogeneity, scheduling constraints, and re-planning requirements. A mechanism for narrative-driven schedule adaptation is introduced, in which unstructured natural language inputs are interpreted by an LLM, and optimization constraints are autonomously updated, enabling human-in-the-loop flexibility without manual coding. A digital twin-based system has been developed to enable real-time synchronization between physical operations and their digital representations. This closed-loop feedback framework ensures that the system remains dynamic and responsive to ongoing changes on site. A case study demonstrates both the computational efficiency of the optimization algorithm and the reasoning performance of several LLMs, with top-performing models achieving over 97% accuracy in constraint and parameter extraction. The results confirm the practicality, adaptability, and cross-domain applicability of the proposed methods. 

**Abstract (ZH)**: 多机器人系统在工业领域日益增长的生产力、安全性和适应性需求中展现出前景，然而在动态和不确定环境中有效协调多机器人仍是一个挑战，尤其是由于材料延误、意外现场条件和天气干扰等不可预测因素。为应对这些挑战，本研究提出了一种适应性任务分配框架，该框架战略性地利用了数字孪生、整数规划和大型语言模型的协同潜力。多机器人任务分配问题被形式化定义并通过考虑任务依赖性、机器人异质性、调度约束和重新规划要求的整数规划模型来求解。引入了一种基于叙述的调度适应机制，在该机制中，非结构化自然语言输入由大型语言模型解释，并自动更新优化约束，从而在无需手动编码的情况下实现人工在环中的灵活性。基于数字孪生的系统已被开发出来，以实现物理操作与其数字表示之间的实时同步。闭环反馈框架确保系统能够动态响应现场的变化。案例研究展示了优化算法的计算效率和若干大型语言模型的推理性能，最佳模型在约束和参数提取方面的准确率超过97%。研究结果证实了所提出方法的实用性、适应性和跨域适用性。 

---
# RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies 

**Title (ZH)**: RoboArena: 分布式实际世界通用机器人策略评估 

**Authors**: Pranav Atreya, Karl Pertsch, Tony Lee, Moo Jin Kim, Arhan Jain, Artur Kuramshin, Clemens Eppner, Cyrus Neary, Edward Hu, Fabio Ramos, Jonathan Tremblay, Kanav Arora, Kirsty Ellis, Luca Macesanu, Matthew Leonard, Meedeum Cho, Ozgur Aslan, Shivin Dass, Jie Wang, Xingfang Yuan, Xuning Yang, Abhishek Gupta, Dinesh Jayaraman, Glen Berseth, Kostas Daniilidis, Roberto Martin-Martin, Youngwoon Lee, Percy Liang, Chelsea Finn, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2506.18123)  

**Abstract**: Comprehensive, unbiased, and comparable evaluation of modern generalist policies is uniquely challenging: existing approaches for robot benchmarking typically rely on heavy standardization, either by specifying fixed evaluation tasks and environments, or by hosting centralized ''robot challenges'', and do not readily scale to evaluating generalist policies across a broad range of tasks and environments. In this work, we propose RoboArena, a new approach for scalable evaluation of generalist robot policies in the real world. Instead of standardizing evaluations around fixed tasks, environments, or locations, we propose to crowd-source evaluations across a distributed network of evaluators. Importantly, evaluators can freely choose the tasks and environments they evaluate on, enabling easy scaling of diversity, but they are required to perform double-blind evaluations over pairs of policies. Then, by aggregating preference feedback from pairwise comparisons across diverse tasks and environments, we can derive a ranking of policies. We instantiate our approach across a network of evaluators at seven academic institutions using the DROID robot platform. Through more than 600 pairwise real-robot evaluation episodes across seven generalist policies, we demonstrate that our crowd-sourced approach can more accurately rank the performance of existing generalist policies than conventional, centralized evaluation approaches, while being more scalable, resilient, and trustworthy. We open our evaluation network to the community and hope that it can enable more accessible comparisons of generalist robot policies. 

**Abstract (ZH)**: 综合、公正且可比的现代通用型机器人策略评估具有独特挑战：现有机器人基准测试方法通常依赖于高度标准化，要么通过规定固定的任务和环境，要么通过举办集中式的“机器人挑战”，这无法方便地扩展到广泛的任务和环境。在本文中，我们提出RoboArena，一种新的方法，用于在真实世界中 scalable 评估通用型机器人策略。我们摒弃了以固定任务、环境或地点为标准的评估方法，而是提议通过分布在多个评价者之间的网络进行评价。重要的是，评价者可以选择他们要评估的任务和环境，从而便于多样性扩展，但同时需要对策略对进行双盲评估。通过在多样任务和环境下的成对比较中聚合偏好反馈，我们可以得出策略的排名。我们在七所学术机构的评价者网络中使用DROID机器人平台实例化了该方法。通过超过600次针对七种通用型策略的成对真实机器人评估事件，我们展示了我们提出的众包方法相比于传统的集中式评估方法可以更准确地排名现有通用型策略，同时更具可扩展性、可靠性和可信度。我们向社区开放了评估网络，希望它能促进通用型机器人策略的更易访问的比较。 

---
# GeNIE: A Generalizable Navigation System for In-the-Wild Environments 

**Title (ZH)**: GeNIE: 一种通用的户外环境导航系统 

**Authors**: Jiaming Wang, Diwen Liu, Jizhuo Chen, Jiaxuan Da, Nuowen Qian, Tram Minh Man, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2506.17960)  

**Abstract**: Reliable navigation in unstructured, real-world environments remains a significant challenge for embodied agents, especially when operating across diverse terrains, weather conditions, and sensor configurations. In this paper, we introduce GeNIE (Generalizable Navigation System for In-the-Wild Environments), a robust navigation framework designed for global deployment. GeNIE integrates a generalizable traversability prediction model built on SAM2 with a novel path fusion strategy that enhances planning stability in noisy and ambiguous settings. We deployed GeNIE in the Earth Rover Challenge (ERC) at ICRA 2025, where it was evaluated across six countries spanning three continents. GeNIE took first place and achieved 79% of the maximum possible score, outperforming the second-best team by 17%, and completed the entire competition without a single human intervention. These results set a new benchmark for robust, generalizable outdoor robot navigation. We will release the codebase, pretrained model weights, and newly curated datasets to support future research in real-world navigation. 

**Abstract (ZH)**: 可靠的导航在未结构化的现实环境中仍然是体化代理面临的巨大挑战，尤其是在跨越多种地形、天气条件和传感器配置时。本文介绍了GeNIE（适用于野生环境的通用导航系统），这是一种旨在全球部署的鲁棒导航框架。GeNIE 结合了基于 SAM2 的可推广通行性预测模型和一种新颖的路径融合策略，以增强在噪声和模糊环境中的计划稳定性。我们将在ICRA 2025的Earth Rover挑战赛（ERC）中部署GeNIE，它在跨越三大洲六个国家的评估中位列第一，并取得了最高可能得分的79%，比第二名领先17%，且全程无需人工干预。这些结果为鲁棒、可推广的户外机器人导航设定了新的基准。我们将发布代码库、预训练模型权重以及新编制的数据集，以支持未来的现实世界导航研究。 

---
# Geometric Contact Flows: Contactomorphisms for Dynamics and Control 

**Title (ZH)**: 几何接触流动：接触omorphic变换的动力学与控制 

**Authors**: Andrea Testa, Søren Hauberg, Tamim Asfour, Leonel Rozo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17868)  

**Abstract**: Accurately modeling and predicting complex dynamical systems, particularly those involving force exchange and dissipation, is crucial for applications ranging from fluid dynamics to robotics, but presents significant challenges due to the intricate interplay of geometric constraints and energy transfer. This paper introduces Geometric Contact Flows (GFC), a novel framework leveraging Riemannian and Contact geometry as inductive biases to learn such systems. GCF constructs a latent contact Hamiltonian model encoding desirable properties like stability or energy conservation. An ensemble of contactomorphisms then adapts this model to the target dynamics while preserving these properties. This ensemble allows for uncertainty-aware geodesics that attract the system's behavior toward the data support, enabling robust generalization and adaptation to unseen scenarios. Experiments on learning dynamics for physical systems and for controlling robots on interaction tasks demonstrate the effectiveness of our approach. 

**Abstract (ZH)**: 准确建模和预测涉及力交换和耗散的复杂动力系统对于从流体动力学到机器人技术的应用至关重要，但由于几何约束和能量传递的复杂相互作用，这提出了重大挑战。本文提出了一种新颖的框架——几何接触流（GFC），该框架利用黎曼几何和接触几何作为归纳偏置来学习此类系统。GFC 构建了一个潜在的接触哈密尔顿模型，编码了如稳定性和能量守恒等 desirable 性质。一组接触同构随后将该模型适应目标动力学，同时保持这些性质。该集合允许不确定性意识下的测地线吸引系统的行为朝向数据支持，从而实现稳健的泛化和对未见场景的适应。在物理系统动力学习和交互任务中控制机器人方面的实验展示了我们方法的有效性。 

---
# Engagement and Disclosures in LLM-Powered Cognitive Behavioral Therapy Exercises: A Factorial Design Comparing the Influence of a Robot vs. Chatbot Over Time 

**Title (ZH)**: 基于机器人 versus 聊天机器人影响的因变量设计：时间进程中认知行为疗法练习中的参与与披露比较研究 

**Authors**: Mina Kian, Mingyu Zong, Katrin Fischer, Anna-Maria Velentza, Abhyuday Singh, Kaleen Shrestha, Pau Sang, Shriya Upadhyay, Wallace Browning, Misha Arif Faruki, Sébastien M. R. Arnold, Bhaskar Krishnamachari, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2506.17831)  

**Abstract**: Many researchers are working to address the worldwide mental health crisis by developing therapeutic technologies that increase the accessibility of care, including leveraging large language model (LLM) capabilities in chatbots and socially assistive robots (SARs) used for therapeutic applications. Yet, the effects of these technologies over time remain unexplored. In this study, we use a factorial design to assess the impact of embodiment and time spent engaging in therapeutic exercises on participant disclosures. We assessed transcripts gathered from a two-week study in which 26 university student participants completed daily interactive Cognitive Behavioral Therapy (CBT) exercises in their residences using either an LLM-powered SAR or a disembodied chatbot. We evaluated the levels of active engagement and high intimacy of their disclosures (opinions, judgments, and emotions) during each session and over time. Our findings show significant interactions between time and embodiment for both outcome measures: participant engagement and intimacy increased over time in the physical robot condition, while both measures decreased in the chatbot condition. 

**Abstract (ZH)**: 许多研究者致力于通过开发提高护理可及性的治疗技术来应对全球心理健康危机，包括利用大型语言模型（LLM）能力的聊天机器人和社会辅助机器人（SARs）用于治疗应用。然而，这些技术的效果随时间的变化尚未被探索。在本研究中，我们采用因子设计评估实体形态和参与治疗练习时间对参与者披露的影响。我们评估了在为期两周的研究中收集的对话记录，该研究中26名大学学生参与者在住所使用LLM驱动的SAR或 disembodied聊天机器人完成每日互动的认知行为疗法（CBT）练习。我们评估了每个会话及随时间发展的积极参与程度和高亲密性（意见、判断和情绪）水平。我们的研究发现，对于两个结果指标，时间和实体形态之间存在显著的交互作用：在实体机器人条件下，参与者的积极参与程度和亲密性随时间增加，而在聊天机器人条件下，这两个指标随时间下降。 

---
# RoboMonkey: Scaling Test-Time Sampling and Verification for Vision-Language-Action Models 

**Title (ZH)**: RoboMonkey: 扩展视觉-语言-动作模型测试时采样与验证规模 

**Authors**: Jacky Kwok, Christopher Agia, Rohan Sinha, Matt Foutter, Shulu Li, Ion Stoica, Azalia Mirhoseini, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2506.17811)  

**Abstract**: Vision-Language-Action (VLA) models have demonstrated remarkable capabilities in visuomotor control, yet ensuring their robustness in unstructured real-world environments remains a persistent challenge. In this paper, we investigate test-time scaling through the lens of sampling and verification as means to enhance the robustness and generalization of VLAs. We first demonstrate that the relationship between action error and the number of generated samples follows an exponentiated power law across a range of VLAs, indicating the existence of inference-time scaling laws. Building on these insights, we introduce RoboMonkey, a test-time scaling framework for VLAs. At deployment, RoboMonkey samples a small set of actions from a VLA, applies Gaussian perturbation and majority voting to construct an action proposal distribution, and then uses a Vision Language Model (VLM)-based verifier to select the optimal action. We propose a synthetic data generation pipeline for training such VLM-based action verifiers, and demonstrate that scaling the synthetic dataset consistently improves verification and downstream accuracy. Through extensive simulated and hardware experiments, we show that pairing existing VLAs with RoboMonkey yields significant performance gains, achieving a 25% absolute improvement on out-of-distribution tasks and 8% on in-distribution tasks. Additionally, when adapting to new robot setups, we show that fine-tuning both VLAs and action verifiers yields a 7% performance increase compared to fine-tuning VLAs alone. 

**Abstract (ZH)**: 基于视觉-语言-动作模型测试时的缩放以增强其在非结构化现实环境中的稳健性和泛化能力 

---
# RLRC: Reinforcement Learning-based Recovery for Compressed Vision-Language-Action Models 

**Title (ZH)**: 基于强化学习的压缩视觉-语言-动作模型恢复方法 

**Authors**: Yuxuan Chen, Xiao Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17639)  

**Abstract**: Vision-Language-Action models (VLA) have demonstrated remarkable capabilities and promising potential in solving complex robotic manipulation tasks. However, their substantial parameter sizes and high inference latency pose significant challenges for real-world deployment, particularly on resource-constrained robotic platforms. To address this issue, we begin by conducting an extensive empirical study to explore the effectiveness of model compression techniques when applied to VLAs. Building on the insights gained from these preliminary experiments, we propose RLRC, a three-stage recovery method for compressed VLAs, including structured pruning, performance recovery based on SFT and RL, and further quantization. RLRC achieves up to an 8x reduction in memory usage and a 2.3x improvement in inference throughput, while maintaining or even surpassing the original VLA's task success rate. Extensive experiments show that RLRC consistently outperforms existing compression baselines, demonstrating strong potential for on-device deployment of VLAs. Project website: this https URL 

**Abstract (ZH)**: Vision-Language-Action模型（VLA）在解决复杂机器人操作任务方面展现了显著的能力和广阔的潜力。然而，其庞大的参数量和高推理延迟给实际部署造成了重大挑战，特别是在资源受限的机器人平台上。为解决这一问题，我们首先进行了一项广泛的经验研究，探索在VLA中应用模型压缩技术的有效性。基于初步实验获得的见解，我们提出了一种名为RLRC的三层恢复方法，包括结构化剪枝、基于SFT和RL的性能恢复以及进一步的量化。RLRC实现了内存使用最多8倍的减少和推理 throughput 2.3倍的提升，同时保持甚至超越了原始VLA的任务成功率。广泛的实验表明，RLRC在各方面的性能都优于现有压缩基线，展示了在设备端部署VLAs的强大潜力。项目网站: [这个链接](this https URL)。 

---
# Imitation Learning for Active Neck Motion Enabling Robot Manipulation beyond the Field of View 

**Title (ZH)**: 基于模仿学习的主动颈部运动使机器人 manipulation 超出视域范围 

**Authors**: Koki Nakagawa, Yoshiyuki Ohmura, Yasuo Kuniyoshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17624)  

**Abstract**: Most prior research in deep imitation learning has predominantly utilized fixed cameras for image input, which constrains task performance to the predefined field of view. However, enabling a robot to actively maneuver its neck can significantly expand the scope of imitation learning to encompass a wider variety of tasks and expressive actions such as neck gestures. To facilitate imitation learning in robots capable of neck movement while simultaneously performing object manipulation, we propose a teaching system that systematically collects datasets incorporating neck movements while minimizing discomfort caused by dynamic viewpoints during teleoperation. In addition, we present a novel network model for learning manipulation tasks including active neck motion. Experimental results showed that our model can achieve a high success rate of around 90\%, regardless of the distraction from the viewpoint variations by active neck motion. Moreover, the proposed model proved particularly effective in challenging scenarios, such as when objects were situated at the periphery or beyond the standard field of view, where traditional models struggled. The proposed approach contributes to the efficiency of dataset collection and extends the applicability of imitation learning to more complex and dynamic scenarios. 

**Abstract (ZH)**: 基于颈部运动的深度模仿学习教学系统与网络模型研究 

---
# Risk-Guided Diffusion: Toward Deploying Robot Foundation Models in Space, Where Failure Is Not An Option 

**Title (ZH)**: 风险管理导向的扩散：向着在不允许失败的空间场景中部署机器人基础模型的目标努力 

**Authors**: Rohan Thakker, Adarsh Patnaik, Vince Kurtz, Jonas Frey, Jonathan Becktor, Sangwoo Moon, Rob Royce, Marcel Kaufmann, Georgios Georgakis, Pascal Roth, Joel Burdick, Marco Hutter, Shehryar Khattak  

**Link**: [PDF](https://arxiv.org/pdf/2506.17601)  

**Abstract**: Safe, reliable navigation in extreme, unfamiliar terrain is required for future robotic space exploration missions. Recent generative-AI methods learn semantically aware navigation policies from large, cross-embodiment datasets, but offer limited safety guarantees. Inspired by human cognitive science, we propose a risk-guided diffusion framework that fuses a fast, learned "System-1" with a slow, physics-based "System-2", sharing computation at both training and inference to couple adaptability with formal safety. Hardware experiments conducted at the NASA JPL's Mars-analog facility, Mars Yard, show that our approach reduces failure rates by up to $4\times$ while matching the goal-reaching performance of learning-based robotic models by leveraging inference-time compute without any additional training. 

**Abstract (ZH)**: 极端 unfamiliar 地形下安全可靠的导航是未来机器人太空探索任务的必要要求。受人类认知科学启发，我们提出了一种风险导向扩散框架，该框架结合了快速的学习“系统-1”和缓慢的物理基础“系统-2”，在训练和推理阶段共享计算，以结合适应性和形式化安全性。在NASA JPL的火星模拟设施Mars Yard进行的硬件实验表明，我们的方法在不进行额外训练的情况下，通过利用推理时的计算资源将失败率降低最多4倍，同时匹配基于学习的机器人模型的目标到达性能。 

---
# EASE: Embodied Active Event Perception via Self-Supervised Energy Minimization 

**Title (ZH)**: EASE: 通过自我监督的能量最小化实现沉浸式主动事件感知 

**Authors**: Zhou Chen, Sanjoy Kundu, Harsimran S. Baweja, Sathyanarayanan N. Aakur  

**Link**: [PDF](https://arxiv.org/pdf/2506.17516)  

**Abstract**: Active event perception, the ability to dynamically detect, track, and summarize events in real time, is essential for embodied intelligence in tasks such as human-AI collaboration, assistive robotics, and autonomous navigation. However, existing approaches often depend on predefined action spaces, annotated datasets, and extrinsic rewards, limiting their adaptability and scalability in dynamic, real-world scenarios. Inspired by cognitive theories of event perception and predictive coding, we propose EASE, a self-supervised framework that unifies spatiotemporal representation learning and embodied control through free energy minimization. EASE leverages prediction errors and entropy as intrinsic signals to segment events, summarize observations, and actively track salient actors, operating without explicit annotations or external rewards. By coupling a generative perception model with an action-driven control policy, EASE dynamically aligns predictions with observations, enabling emergent behaviors such as implicit memory, target continuity, and adaptability to novel environments. Extensive evaluations in simulation and real-world settings demonstrate EASE's ability to achieve privacy-preserving and scalable event perception, providing a robust foundation for embodied systems in unscripted, dynamic tasks. 

**Abstract (ZH)**: 主动事件感知能力，即在实时环境下动态检测、跟踪和总结事件的能力，是诸如人机协作、辅助机器人和自主导航等任务中体现式智能的关键。然而，现有方法往往依赖预定义的动作空间、标注数据集和外在奖励，限制了其在动态真实世界场景中的适应性和扩展性。受事件感知认知理论和预测编码的启发，我们提出了一种自监督框架EASE，通过最小化自由能统一时空表示学习和体现式控制。EASE 利用预测误差和熵作为内在信号进行事件分割、总结观察和主动跟踪显著主体，无需明确标注或外部奖励。通过结合生成感知模型与动作驱动的控制策略，EASE 动态对齐预测与观察，使潜在记忆、目标连续性和对新型环境的适应能力等涌现行为成为可能。在仿真和真实世界环境中的广泛评估表明，EASE 能实现隐私保护和可扩展的事件感知，为无脚本动态任务中的体现式系统提供坚实基础。 

---
# Distilling On-device Language Models for Robot Planning with Minimal Human Intervention 

**Title (ZH)**: 在设备端精简语言模型以实现最少人工介入的机器人规划 

**Authors**: Zachary Ravichandran, Ignacio Hounie, Fernando Cladera, Alejandro Ribeiro, George J. Pappas, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.17486)  

**Abstract**: Large language models (LLMs) provide robots with powerful contextual reasoning abilities and a natural human interface. Yet, current LLM-enabled robots typically depend on cloud-hosted models, limiting their usability in environments with unreliable communication infrastructure, such as outdoor or industrial settings. We present PRISM, a framework for distilling small language model (SLM)-enabled robot planners that run on-device with minimal human supervision. Starting from an existing LLM-enabled planner, PRISM automatically synthesizes diverse tasks and environments, elicits plans from the LLM, and uses this synthetic dataset to distill a compact SLM as a drop-in replacement of the source model. We apply PRISM to three LLM-enabled planners for mapping and exploration, manipulation, and household assistance, and we demonstrate that PRISM improves the performance of Llama-3.2-3B from 10-20% of GPT-4o's performance to over 93% - using only synthetic data. We further demonstrate that the distilled planners generalize across heterogeneous robotic platforms (ground and aerial) and diverse environments (indoor and outdoor). We release all software, trained models, and datasets at this https URL. 

**Abstract (ZH)**: 基于设备端的小语言模型使能机器人规划框架：PRISM 

---
# General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting 

**Title (ZH)**: 通用机器人导航通过LVLM协调感知、推理和行动 

**Authors**: Bernard Lange, Anil Yildiz, Mansur Arief, Shehryar Khattak, Mykel Kochenderfer, Georgios Georgakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.17462)  

**Abstract**: Developing general-purpose navigation policies for unknown environments remains a core challenge in robotics. Most existing systems rely on task-specific neural networks and fixed data flows, limiting generalizability. Large Vision-Language Models (LVLMs) offer a promising alternative by embedding human-like knowledge suitable for reasoning and planning. Yet, prior LVLM-robot integrations typically depend on pre-mapped spaces, hard-coded representations, and myopic exploration. We introduce the Agentic Robotic Navigation Architecture (ARNA), a general-purpose navigation framework that equips an LVLM-based agent with a library of perception, reasoning, and navigation tools available within modern robotic stacks. At runtime, the agent autonomously defines and executes task-specific workflows that iteratively query the robotic modules, reason over multimodal inputs, and select appropriate navigation actions. This approach enables robust navigation and reasoning in previously unmapped environments, providing a new perspective on robotic stack design. Evaluated in Habitat Lab on the HM-EQA benchmark, ARNA achieves state-of-the-art performance, demonstrating effective exploration, navigation, and embodied question answering without relying on handcrafted plans, fixed input representations, or pre-existing maps. 

**Abstract (ZH)**: 开发适用于未知环境的一般导航政策仍然是机器人领域的核心挑战。现有的大多数系统依赖于任务特定的神经网络和固定的数据流，限制了泛化能力。大型视觉-语言模型（LVLM）通过嵌入适合推理和规划的人类知识提供了有前景的替代方案。然而，之前的LVLM-机器人集成通常依赖预映射的空间、硬编码的表示和短视的探索。我们引入了Agentic Robotic Navigation Architecture（ARNA），这是一种一般用途的导航框架，为基于LVLM的代理配备了现代机器人堆栈内部可用的感知、推理和导航工具。在运行时，代理自主定义和执行迭代查询机器人模块、处理多模态输入并选择适当导航动作的具体工作流程。这种方法能够在未映射的环境中实现稳健的导航和推理，为机器人堆栈设计提供了新的视角。在Habitat Lab上的HM-EQA基准测试中，ARNA达到了最佳性能，展示了在不依赖手工设计的计划、固定输入表示或预先存在的地图的情况下进行有效的探索、导航和具身问题解答的能力。 

---
# Reflective VLM Planning for Dual-Arm Desktop Cleaning: Bridging Open-Vocabulary Perception and Precise Manipulation 

**Title (ZH)**: 面向双臂桌面清洁的反射型VLM规划：开放词汇感知与精确操作的桥梁 

**Authors**: Yufan Liu, Yi Wu, Gweneth Ge, Haoliang Cheng, Rui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17328)  

**Abstract**: Desktop cleaning demands open-vocabulary recognition and precise manipulation for heterogeneous debris. We propose a hierarchical framework integrating reflective Vision-Language Model (VLM) planning with dual-arm execution via structured scene representation. Grounded-SAM2 facilitates open-vocabulary detection, while a memory-augmented VLM generates, critiques, and revises manipulation sequences. These sequences are converted into parametric trajectories for five primitives executed by coordinated Franka arms. Evaluated in simulated scenarios, our system achieving 87.2% task completion, a 28.8% improvement over static VLM and 36.2% over single-arm baselines. Structured memory integration proves crucial for robust, generalizable manipulation while maintaining real-time control performance. 

**Abstract (ZH)**: 桌面清理需求开放式词汇识别和精确操作，针对异构碎片。我们提出了一种层次框架，结合反射性视觉语言模型（VLM）规划与基于结构化场景表示的双臂执行。Grounded-SAM2 支持开放式词汇检测，而记忆增强的 VLM 生成、批判和修订操作序列。这些序列被转换为由协调的法兰卡手臂执行的五种原始操作的参数轨迹。在模拟场景中评估，我们的系统实现了 87.2% 的任务完成率，分别比静态 VLM 提高了 28.8%，比单臂基线提高了 36.2%。结构化记忆集成对于保持实时控制性能的同时实现 robust 和通用的操作至关重要。 

---
# BRAVE: Brain-Controlled Prosthetic Arm with Voice Integration and Embodied Learning for Enhanced Mobility 

**Title (ZH)**: BRAVE: 语音集成和躯体化学习驱动的脑控假肢手臂以提升移动能力 

**Authors**: Abdul Basit, Maha Nawaz, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2506.18749)  

**Abstract**: Non-invasive brain-computer interfaces (BCIs) have the potential to enable intuitive control of prosthetic limbs for individuals with upper limb amputations. However, existing EEG-based control systems face challenges related to signal noise, classification accuracy, and real-time adaptability. In this work, we present BRAVE, a hybrid EEG and voice-controlled prosthetic system that integrates ensemble learning-based EEG classification with a human-in-the-loop (HITL) correction framework for enhanced responsiveness. Unlike traditional electromyography (EMG)-based prosthetic control, BRAVE aims to interpret EEG-driven motor intent, enabling movement control without reliance on residual muscle activity. To improve classification robustness, BRAVE combines LSTM, CNN, and Random Forest models in an ensemble framework, achieving a classification accuracy of 96% across test subjects. EEG signals are preprocessed using a bandpass filter (0.5-45 Hz), Independent Component Analysis (ICA) for artifact removal, and Common Spatial Pattern (CSP) feature extraction to minimize contamination from electromyographic (EMG) and electrooculographic (EOG) signals. Additionally, BRAVE incorporates automatic speech recognition (ASR) to facilitate intuitive mode switching between different degrees of freedom (DOF) in the prosthetic arm. The system operates in real time, with a response latency of 150 ms, leveraging Lab Streaming Layer (LSL) networking for synchronized data acquisition. The system is evaluated on an in-house fabricated prosthetic arm and on multiple participants highlighting the generalizability across users. The system is optimized for low-power embedded deployment, ensuring practical real-world application beyond high-performance computing environments. Our results indicate that BRAVE offers a promising step towards robust, real-time, non-invasive prosthetic control. 

**Abstract (ZH)**: 非侵入式脑-计算机接口：面向上肢截肢个体的直观假肢控制 

---
# Drive-R1: Bridging Reasoning and Planning in VLMs for Autonomous Driving with Reinforcement Learning 

**Title (ZH)**: Drive-R1: 在自主驾驶中将推理与规划结合到VLMs中的强化学习方法 

**Authors**: Yue Li, Meng Tian, Dechang Zhu, Jiangtong Zhu, Zhenyu Lin, Zhiwei Xiong, Xinhai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18234)  

**Abstract**: Large vision-language models (VLMs) for autonomous driving (AD) are evolving beyond perception and cognition tasks toward motion planning. However, we identify two critical challenges in this direction: (1) VLMs tend to learn shortcuts by relying heavily on history input information, achieving seemingly strong planning results without genuinely understanding the visual inputs; and (2) the chain-ofthought (COT) reasoning processes are always misaligned with the motion planning outcomes, and how to effectively leverage the complex reasoning capability to enhance planning remains largely underexplored. In this paper, we start from a small-scale domain-specific VLM and propose Drive-R1 designed to bridges the scenario reasoning and motion planning for AD. Drive-R1 first undergoes the supervised finetuning on a elaborate dataset containing both long and short COT data. Drive-R1 is encouraged to reason step-by-step from visual input to final planning decisions. Subsequently, Drive-R1 is trained within a reinforcement learning framework that incentivizes the discovery of reasoning paths that are more informative for planning, guided by rewards based on predicted trajectories and meta actions. Experimental evaluations on the nuScenes and DriveLM-nuScenes benchmarks demonstrate that Drive-R1 achieves superior performance compared to existing state-of-the-art VLMs. We believe that Drive-R1 presents a promising direction for bridging reasoning and planning in AD, offering methodological insights for future research and applications. 

**Abstract (ZH)**: 面向自动驾驶的大型多模态模型从感知与认知向运动规划的演进及其挑战：Drive-R1的设计与实现 

---
# Quantification of Sim2Real Gap via Neural Simulation Gap Function 

**Title (ZH)**: 通过神经模拟差距函数量化Sim2Real差距 

**Authors**: P Sangeerth, Pushpak Jagtap  

**Link**: [PDF](https://arxiv.org/pdf/2506.17675)  

**Abstract**: In this paper, we introduce the notion of neural simulation gap functions, which formally quantifies the gap between the mathematical model and the model in the high-fidelity simulator, which closely resembles reality. Many times, a controller designed for a mathematical model does not work in reality because of the unmodelled gap between the two systems. With the help of this simulation gap function, one can use existing model-based tools to design controllers for the mathematical system and formally guarantee a decent transition from the simulation to the real world. Although in this work, we have quantified this gap using a neural network, which is trained using a finite number of data points, we give formal guarantees on the simulation gap function for the entire state space including the unseen data points. We collect data from high-fidelity simulators leveraging recent advancements in Real-to-Sim transfer to ensure close alignment with reality. We demonstrate our results through two case studies - a Mecanum bot and a Pendulum. 

**Abstract (ZH)**: 本文引入了神经模拟差距函数的概念，正式量化了高保真模拟器中的模型与数学模型之间的差距，该模拟器与现实场景高度相似。由于两个系统之间的未建模差距，设计用于数学模型的控制器在现实中往往无法工作。借助此模拟差距函数，可以使用现有的基于模型的工具来为数学系统设计控制器，并正式保证从模拟到现实世界的顺利过渡。尽管本文使用神经网络并通过有限数量的数据点对该差距进行了量化，但我们对整个状态空间，包括未见过的数据点，给出了模拟差距函数的正式保证。我们利用近期实到模迁移的进展，从高保真模拟器中收集数据以确保与现实的紧密对齐。我们通过两个案例研究——全向机器人和单摆——来展示我们的结果。 

---
# Accelerating Residual Reinforcement Learning with Uncertainty Estimation 

**Title (ZH)**: 基于不确定性估计加速残差强化学习 

**Authors**: Lakshita Dodeja, Karl Schmeckpeper, Shivam Vats, Thomas Weng, Mingxi Jia, George Konidaris, Stefanie Tellex  

**Link**: [PDF](https://arxiv.org/pdf/2506.17564)  

**Abstract**: Residual Reinforcement Learning (RL) is a popular approach for adapting pretrained policies by learning a lightweight residual policy that provides corrective actions. While Residual RL is more sample-efficient than finetuning the entire base policy, existing methods struggle with sparse rewards and are designed for deterministic base policies. We propose two improvements to Residual RL that further enhance its sample efficiency and make it suitable for stochastic base policies. First, we leverage uncertainty estimates of the base policy to focus exploration on regions in which the base policy is not confident. Second, we propose a simple modification to off-policy residual learning that allows it to observe base actions and better handle stochastic base policies. We evaluate our method with both Gaussian-based and Diffusion-based stochastic base policies on tasks from Robosuite and D4RL, and compare against state-of-the-art finetuning methods, demo-augmented RL methods, and other residual RL methods. Our algorithm significantly outperforms existing baselines in a variety of simulation benchmark environments. We also deploy our learned polices in the real world to demonstrate their robustness with zero-shot sim-to-real transfer. 

**Abstract (ZH)**: 残差强化学习（RL）是一种通过学习轻量级的残差策略来适应预训练策略的流行方法，该残差策略提供纠正动作。尽管残差RL在样本效率方面优于完全调优基策略，但现有方法在稀疏奖励方面表现不佳，并且主要设计用于确定性基策略。我们提出了两种改进残差RL的方法，以进一步提高其样本效率并使其适用于随机基策略。首先，我们利用基策略的不确定性估计，将探索集中在基策略不太自信的区域上。其次，我们提出了一种简单的方法修改，使得离策残差学习能够观察基策略动作，并更好地处理随机基策略。我们在Robosuite和D4RL的任务上使用基于高斯和扩散的随机基策略评估我们的方法，并与最先进的调优方法、演示增强RL方法和其他残差RL方法进行比较。我们的算法在多种模拟基准环境中显著优于现有基线方法。我们还在实际环境中部署我们的学习策略，以展示其零样本仿真实到现实transfer的鲁棒性。 

---
# Decentralized Consensus Inference-based Hierarchical Reinforcement Learning for Multi-Constrained UAV Pursuit-Evasion Game 

**Title (ZH)**: 基于分布共识推断的多层次强化学习多约束无人机捕逃博弈 

**Authors**: Xiang Yuming, Li Sizhao, Li Rongpeng, Zhao Zhifeng, Zhang Honggang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18126)  

**Abstract**: Multiple quadrotor unmanned aerial vehicle (UAV) systems have garnered widespread research interest and fostered tremendous interesting applications, especially in multi-constrained pursuit-evasion games (MC-PEG). The Cooperative Evasion and Formation Coverage (CEFC) task, where the UAV swarm aims to maximize formation coverage across multiple target zones while collaboratively evading predators, belongs to one of the most challenging issues in MC-PEG, especially under communication-limited constraints. This multifaceted problem, which intertwines responses to obstacles, adversaries, target zones, and formation dynamics, brings up significant high-dimensional complications in locating a solution. In this paper, we propose a novel two-level framework (i.e., Consensus Inference-based Hierarchical Reinforcement Learning (CI-HRL)), which delegates target localization to a high-level policy, while adopting a low-level policy to manage obstacle avoidance, navigation, and formation. Specifically, in the high-level policy, we develop a novel multi-agent reinforcement learning module, Consensus-oriented Multi-Agent Communication (ConsMAC), to enable agents to perceive global information and establish consensus from local states by effectively aggregating neighbor messages. Meanwhile, we leverage an Alternative Training-based Multi-agent proximal policy optimization (AT-M) and policy distillation to accomplish the low-level control. The experimental results, including the high-fidelity software-in-the-loop (SITL) simulations, validate that CI-HRL provides a superior solution with enhanced swarm's collaborative evasion and task completion capabilities. 

**Abstract (ZH)**: 基于共识推理的层次强化学习框架（CI-HRL）：多重约束捕逃游戏中的协同规避与编队覆盖 

---
# Deep Research Agents: A Systematic Examination And Roadmap 

**Title (ZH)**: 深度研究代理：系统审查与 roadmap 

**Authors**: Yuxuan Huang, Yihang Chen, Haozheng Zhang, Kang Li, Meng Fang, Linyi Yang, Xiaoguang Li, Lifeng Shang, Songcen Xu, Jianye Hao, Kun Shao, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18096)  

**Abstract**: The rapid progress of Large Language Models (LLMs) has given rise to a new category of autonomous AI systems, referred to as Deep Research (DR) agents. These agents are designed to tackle complex, multi-turn informational research tasks by leveraging a combination of dynamic reasoning, adaptive long-horizon planning, multi-hop information retrieval, iterative tool use, and the generation of structured analytical reports. In this paper, we conduct a detailed analysis of the foundational technologies and architectural components that constitute Deep Research agents. We begin by reviewing information acquisition strategies, contrasting API-based retrieval methods with browser-based exploration. We then examine modular tool-use frameworks, including code execution, multimodal input processing, and the integration of Model Context Protocols (MCPs) to support extensibility and ecosystem development. To systematize existing approaches, we propose a taxonomy that differentiates between static and dynamic workflows, and we classify agent architectures based on planning strategies and agent composition, including single-agent and multi-agent configurations. We also provide a critical evaluation of current benchmarks, highlighting key limitations such as restricted access to external knowledge, sequential execution inefficiencies, and misalignment between evaluation metrics and the practical objectives of DR agents. Finally, we outline open challenges and promising directions for future research. A curated and continuously updated repository of DR agent research is available at: {this https URL}. 

**Abstract (ZH)**: 大型语言模型的迅速进展催生了一类新的自主AI系统，称为深度研究（DR）代理。这些代理旨在通过动态推理、适应性长期规划、多跳信息检索、迭代工具使用以及生成结构化分析报告来应对复杂、多轮的信息研究任务。本文详细分析了构成深度研究代理的基础技术和架构组件。我们首先回顾了信息获取策略，对比了基于API的信息检索方法与基于浏览器的探索方法。随后，我们探讨了模块化工具使用框架，包括代码执行、多模态输入处理以及模型上下文协议（MCPs）的集成，以支持扩展性和生态系统开发。为了系统化现有的方法，我们提出了一个分类学，区分静态和动态工作流，并根据规划策略和代理组成对代理架构进行分类，包括单代理和多代理配置。最后，我们对当前基准进行了批判性评价，指出了其关键限制，如对外部知识访问的限制、顺序执行的低效性以及评估指标与DR代理实际目标的不一致。我们还提出了未来研究中面临的开放挑战及有前景的研究方向。深度研究代理研究的精心整理并持续更新的资料库可供参考：{this https URL}。 

---
# Graphs Meet AI Agents: Taxonomy, Progress, and Future Opportunities 

**Title (ZH)**: 图与AI代理：分类、进展与未来机会 

**Authors**: Yuanchen Bei, Weizhi Zhang, Siwen Wang, Weizhi Chen, Sheng Zhou, Hao Chen, Yong Li, Jiajun Bu, Shirui Pan, Yizhou Yu, Irwin King, Fakhri Karray, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18019)  

**Abstract**: AI agents have experienced a paradigm shift, from early dominance by reinforcement learning (RL) to the rise of agents powered by large language models (LLMs), and now further advancing towards a synergistic fusion of RL and LLM capabilities. This progression has endowed AI agents with increasingly strong abilities. Despite these advances, to accomplish complex real-world tasks, agents are required to plan and execute effectively, maintain reliable memory, and coordinate smoothly with other agents. Achieving these capabilities involves contending with ever-present intricate information, operations, and interactions. In light of this challenge, data structurization can play a promising role by transforming intricate and disorganized data into well-structured forms that agents can more effectively understand and process. In this context, graphs, with their natural advantage in organizing, managing, and harnessing intricate data relationships, present a powerful data paradigm for structurization to support the capabilities demanded by advanced AI agents. To this end, this survey presents a first systematic review of how graphs can empower AI agents. Specifically, we explore the integration of graph techniques with core agent functionalities, highlight notable applications, and identify prospective avenues for future research. By comprehensively surveying this burgeoning intersection, we hope to inspire the development of next-generation AI agents equipped to tackle increasingly sophisticated challenges with graphs. Related resources are collected and continuously updated for the community in the Github link. 

**Abstract (ZH)**: AI代理经历了从强化学习主导到由大规模语言模型驱动的转变，并进一步朝着强化学习和大规模语言模型能力协同融合的方向发展。这一进程赋予了AI代理越来越强的能力。尽管取得了这些进展，但要完成复杂的现实世界任务，代理需要有效规划和执行、维持可靠的记忆，并与其他代理协调一致。实现这些能力涉及处理不断出现的复杂信息、操作和互动。鉴于这一挑战，数据结构化可以通过将复杂且未组织好的数据转化为代理能够更有效地理解和处理的结构化形式，发挥重要作用。在此背景下，凭借其在组织、管理和利用复杂数据关系方面的天然优势，图形呈现了一个强大的数据范式，以支持先进AI代理所要求的能力。为此，本文综述了图形如何赋能AI代理的初步系统性研究。具体而言，我们探讨了图形技术与核心代理功能的整合，强调了重要应用，并指出了未来研究的潜在方向。通过全面综述这一新兴交叉领域，我们希望激发开发能够利用图形应对日益复杂挑战的下一代AI代理的研究。相关资源在Github链接中收集并持续更新。 

---
# Matrix-Game: Interactive World Foundation Model 

**Title (ZH)**: 矩阵博弈：交互世界基础模型 

**Authors**: Yifan Zhang, Chunli Peng, Boyang Wang, Puyi Wang, Qingcheng Zhu, Fei Kang, Biao Jiang, Zedong Gao, Eric Li, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18701)  

**Abstract**: We introduce Matrix-Game, an interactive world foundation model for controllable game world generation. Matrix-Game is trained using a two-stage pipeline that first performs large-scale unlabeled pretraining for environment understanding, followed by action-labeled training for interactive video generation. To support this, we curate Matrix-Game-MC, a comprehensive Minecraft dataset comprising over 2,700 hours of unlabeled gameplay video clips and over 1,000 hours of high-quality labeled clips with fine-grained keyboard and mouse action annotations. Our model adopts a controllable image-to-world generation paradigm, conditioned on a reference image, motion context, and user actions. With over 17 billion parameters, Matrix-Game enables precise control over character actions and camera movements, while maintaining high visual quality and temporal coherence. To evaluate performance, we develop GameWorld Score, a unified benchmark measuring visual quality, temporal quality, action controllability, and physical rule understanding for Minecraft world generation. Extensive experiments show that Matrix-Game consistently outperforms prior open-source Minecraft world models (including Oasis and MineWorld) across all metrics, with particularly strong gains in controllability and physical consistency. Double-blind human evaluations further confirm the superiority of Matrix-Game, highlighting its ability to generate perceptually realistic and precisely controllable videos across diverse game scenarios. To facilitate future research on interactive image-to-world generation, we will open-source the Matrix-Game model weights and the GameWorld Score benchmark at this https URL. 

**Abstract (ZH)**: Matrix-Game：可控的游戏世界生成交互式世界基础模型 

---
# NOVA: Navigation via Object-Centric Visual Autonomy for High-Speed Target Tracking in Unstructured GPS-Denied Environments 

**Title (ZH)**: NOVA：基于物体中心视觉自主的高Speed目标追踪导航在无结构的GPS受限环境中 

**Authors**: Alessandro Saviolo, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2506.18689)  

**Abstract**: Autonomous aerial target tracking in unstructured and GPS-denied environments remains a fundamental challenge in robotics. Many existing methods rely on motion capture systems, pre-mapped scenes, or feature-based localization to ensure safety and control, limiting their deployment in real-world conditions. We introduce NOVA, a fully onboard, object-centric framework that enables robust target tracking and collision-aware navigation using only a stereo camera and an IMU. Rather than constructing a global map or relying on absolute localization, NOVA formulates perception, estimation, and control entirely in the target's reference frame. A tightly integrated stack combines a lightweight object detector with stereo depth completion, followed by histogram-based filtering to infer robust target distances under occlusion and noise. These measurements feed a visual-inertial state estimator that recovers the full 6-DoF pose of the robot relative to the target. A nonlinear model predictive controller (NMPC) plans dynamically feasible trajectories in the target frame. To ensure safety, high-order control barrier functions are constructed online from a compact set of high-risk collision points extracted from depth, enabling real-time obstacle avoidance without maps or dense representations. We validate NOVA across challenging real-world scenarios, including urban mazes, forest trails, and repeated transitions through buildings with intermittent GPS loss and severe lighting changes that disrupt feature-based localization. Each experiment is repeated multiple times under similar conditions to assess resilience, showing consistent and reliable performance. NOVA achieves agile target following at speeds exceeding 50 km/h. These results show that high-speed vision-based tracking is possible in the wild using only onboard sensing, with no reliance on external localization or environment assumptions. 

**Abstract (ZH)**: 自主立体视觉与IMU融合的无人机无结构GPS受限环境目标跟踪与碰撞意识导航 

---
# Embedded FPGA Acceleration of Brain-Like Neural Networks: Online Learning to Scalable Inference 

**Title (ZH)**: 基于嵌入式FPGA的类脑神经网络加速：在线学习到可扩展推理 

**Authors**: Muhammad Ihsan Al Hafiz, Naresh Ravichandran, Anders Lansner, Pawel Herman, Artur Podobas  

**Link**: [PDF](https://arxiv.org/pdf/2506.18530)  

**Abstract**: Edge AI applications increasingly require models that can learn and adapt on-device with minimal energy budget. Traditional deep learning models, while powerful, are often overparameterized, energy-hungry, and dependent on cloud connectivity. Brain-Like Neural Networks (BLNNs), such as the Bayesian Confidence Propagation Neural Network (BCPNN), propose a neuromorphic alternative by mimicking cortical architecture and biologically-constrained learning. They offer sparse architectures with local learning rules and unsupervised/semi-supervised learning, making them well-suited for low-power edge intelligence. However, existing BCPNN implementations rely on GPUs or datacenter FPGAs, limiting their applicability to embedded systems. This work presents the first embedded FPGA accelerator for BCPNN on a Zynq UltraScale+ SoC using High-Level Synthesis. We implement both online learning and inference-only kernels with support for variable and mixed precision. Evaluated on MNIST, Pneumonia, and Breast Cancer datasets, our accelerator achieves up to 17.5x latency and 94% energy savings over ARM baselines, without sacrificing accuracy. This work enables practical neuromorphic computing on edge devices, bridging the gap between brain-like learning and real-world deployment. 

**Abstract (ZH)**: 边缘AI应用 increasingly 要求能在设备上以最小的能量预算进行学习和适应的模型。传统深度学习模型虽然强大，但往往过于参数化、能耗高且依赖于云连接。类脑神经网络（BLNNs），如贝叶斯信念传播神经网络（BCPNN），通过模拟皮层架构和生物限制的学习提出了一种类脑替代方案。它们提供了稀疏架构、局部学习规则和无监督/半监督学习，使其非常适合低功耗边缘智能。然而，现有的BCPNN实现依赖于GPU或数据中心FPGA，限制了其在嵌入式系统中的应用。本文在Zynq UltraScale+ SoC上使用High-Level Synthesis首次提出了BCPNN嵌入式FPGA加速器。我们实现了在线学习和仅推理内核，并支持可变和混合精度。在MNIST、肺炎和乳腺癌数据集上进行评估，我们的加速器分别在延迟和能耗上分别实现了高达17.5倍和94%的节省，而不会牺牲准确性。本文在边缘设备上实现了实用的类脑计算，弥合了类脑学习与实际部署之间的差距。 

---
# Sharpening the Spear: Adaptive Expert-Guided Adversarial Attack Against DRL-based Autonomous Driving Policies 

**Title (ZH)**: 磨练矛尖：面向基于DRL的自动驾驶策略的自适应专家引导对抗攻击 

**Authors**: Junchao Fan, Xuyang Lei, Xiaolin Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18304)  

**Abstract**: Deep reinforcement learning (DRL) has emerged as a promising paradigm for autonomous driving. However, despite their advanced capabilities, DRL-based policies remain highly vulnerable to adversarial attacks, posing serious safety risks in real-world deployments. Investigating such attacks is crucial for revealing policy vulnerabilities and guiding the development of more robust autonomous systems. While prior attack methods have made notable progress, they still face several challenges: 1) they often rely on high-frequency attacks, yet critical attack opportunities are typically context-dependent and temporally sparse, resulting in inefficient attack patterns; 2) restricting attack frequency can improve efficiency but often results in unstable training due to the adversary's limited exploration. To address these challenges, we propose an adaptive expert-guided adversarial attack method that enhances both the stability and efficiency of attack policy training. Our method first derives an expert policy from successful attack demonstrations using imitation learning, strengthened by an ensemble Mixture-of-Experts architecture for robust generalization across scenarios. This expert policy then guides a DRL-based adversary through a KL-divergence regularization term. Due to the diversity of scenarios, expert policies may be imperfect. To address this, we further introduce a performance-aware annealing strategy that gradually reduces reliance on the expert as the adversary improves. Extensive experiments demonstrate that our method achieves outperforms existing approaches in terms of collision rate, attack efficiency, and training stability, especially in cases where the expert policy is sub-optimal. 

**Abstract (ZH)**: 基于自适应专家引导的对抗攻击方法：增强深度强化学习在自动驾驶中的攻击稳定性和效率 

---
# RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation 

**Title (ZH)**: RoboTwin 2.0：一种具有强大领域随机化的大规模数据生成器和基准测试，用于稳健的双臂机器人操作 

**Authors**: Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Qiwei Liang, Zixuan Li, Xianliang Lin, Yiheng Ge, Zhenyu Gu, Weiliang Deng, Yubin Guo, Tian Nian, Xuanbing Xie, Qiangyu Chen, Kailun Su, Tianling Xu, Guodong Liu, Mengkang Hu, Huan-ang Gao, Kaixuan Wang, Zhixuan Liang, Yusen Qin, Xiaokang Yang, Ping Luo, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18088)  

**Abstract**: Simulation-based data synthesis has emerged as a powerful paradigm for enhancing real-world robotic manipulation. However, existing synthetic datasets remain insufficient for robust bimanual manipulation due to two challenges: (1) the lack of an efficient, scalable data generation method for novel tasks, and (2) oversimplified simulation environments that fail to capture real-world complexity. We present RoboTwin 2.0, a scalable simulation framework that enables automated, large-scale generation of diverse and realistic data, along with unified evaluation protocols for dual-arm manipulation. We first construct RoboTwin-OD, a large-scale object library comprising 731 instances across 147 categories, each annotated with semantic and manipulation-relevant labels. Building on this foundation, we develop an expert data synthesis pipeline that combines multimodal large language models (MLLMs) with simulation-in-the-loop refinement to generate task-level execution code automatically. To improve sim-to-real transfer, RoboTwin 2.0 incorporates structured domain randomization along five axes: clutter, lighting, background, tabletop height and language instructions, thereby enhancing data diversity and policy robustness. We instantiate this framework across 50 dual-arm tasks spanning five robot embodiments, and pre-collect over 100,000 domain-randomized expert trajectories. Empirical results show a 10.9% gain in code generation success and improved generalization to novel real-world scenarios. A VLA model fine-tuned on our dataset achieves a 367% relative improvement (42.0% vs. 9.0%) on unseen scene real-world tasks, while zero-shot models trained solely on our synthetic data achieve a 228% relative gain, highlighting strong generalization without real-world supervision. We release the data generator, benchmark, dataset, and code to support scalable research in robust bimanual manipulation. 

**Abstract (ZH)**: 基于模拟的数据合成已成为增强现实世界双臂机器人操作能力的强大范式。然而，现有的合成数据集由于两个挑战仍不足以实现鲁棒的双臂操作：（1）缺乏高效、可扩展的新任务数据生成方法，（2）简化过的模拟环境未能捕捉到现实世界的复杂性。我们提出RoboTwin 2.0，一种可扩展的模拟框架，能够实现自动化、大规模地生成多样化和真实的双臂操作数据，并提供了统一的评估协议。我们首先构建了RoboTwin-OD，一个包含731个实例、147类的大规模对象库，每个实例都附有语义和操作相关标签。在此基础上，我们开发了一种专家级数据合成管道，结合多模态大规模语言模型（MLLMs）与闭环模拟精炼，以自动生成任务级执行代码。为提高模拟到现实的转移能力，RoboTwin 2.0整合了沿五个轴向的结构化域随机化：杂乱环境、光照、背景、桌面高度和语言指令，从而增强了数据多样化和策略鲁棒性。我们在50个双臂任务中实例化了此框架，覆盖五个不同机器人实体，并预先收集了超过10万条结构化随机化专家轨迹。实验证明，相较于基准，代码生成成功率提高了10.9%，并且在处理新型现实世界场景时表现出更好的泛化能力。在我们的数据集上微调的VLA模型在未见过的场景任务上取得了367%的相对性能提升（42.0% vs. 9.0%），仅利用我们合成数据训练的零样本模型也实现了228%的相对性能提升，突出了无监督下的强大泛化能力。我们发布了数据生成器、基准、数据集和代码，以支持鲁棒双臂操作的可扩展研究。 

---
# Adapting Vision-Language Models for Evaluating World Models 

**Title (ZH)**: 适应视觉-语言模型以评估世界模型 

**Authors**: Mariya Hendriksen, Tabish Rashid, David Bignell, Raluca Georgescu, Abdelhak Lemkhenter, Katja Hofmann, Sam Devlin, Sarah Parisot  

**Link**: [PDF](https://arxiv.org/pdf/2506.17967)  

**Abstract**: World models -- generative models that simulate environment dynamics conditioned on past observations and actions -- are gaining prominence in planning, simulation, and embodied AI. However, evaluating their rollouts remains a fundamental challenge, requiring fine-grained, temporally grounded assessment of action alignment and semantic consistency -- capabilities not captured by existing metrics. Vision-Language Models (VLMs) have shown promise as automatic evaluators of generative content due to their strong multimodal reasoning abilities. Yet, their use in fine-grained, temporally sensitive evaluation tasks remains limited and requires targeted adaptation. We introduce a evaluation protocol targeting two recognition tasks -- action recognition and character recognition -- each assessed across binary, multiple-choice, and open-ended formats. To support this, we present UNIVERSE (UNIfied Vision-language Evaluator for Rollouts in Simulated Environments), a method for adapting VLMs to rollout evaluation under data and compute constraints. We conduct a large-scale study comparing full, partial, and parameter-efficient finetuning across task formats, context lengths, sampling strategies, and data compositions. The resulting unified evaluator matches the performance of task-specific baselines using a single checkpoint. Human studies confirm strong alignment with human judgments, establishing UNIVERSE as a scalable, semantics-aware evaluator for world models. 

**Abstract (ZH)**: 世界模型——条件于过往观测和行动模拟环境动力学的生成模型——在规划、仿真和具身AI中逐渐受到重视。然而，对其 rollout 的评估仍是一个基本挑战，需要对行动对齐和语义一致性进行细粒度、时间关联性的评估——这些能力现有指标尚未捕捉到。视觉-语言模型（VLMs）因其强大的多模态推理能力，在生成内容的自动评估中显示出了潜力。然而，它们在细粒度、时间敏感评估任务中的应用仍然有限，需要有针对性的适应。我们提出了一种针对两类识别任务——动作识别和角色识别——的评估协议，每类任务分别采用二选一、多选一和开放式格式进行评估。为此，我们提出了UNIVERSE（统一的视觉-语言评估器，用于模拟环境中的 rollout 评估），这是一种在数据和计算约束下使 VLMs 适应 rollout 评估的方法。我们进行了一项大规模研究，比较了不同任务格式、上下文长度、采样策略和数据组成的全面、部分和参数高效微调方法。由此产生的统一评估器使用单个检查点即可达到特定任务基线的性能。人类研究证实了与人类判断的强烈一致性，确立了UNIVERSE作为世界模型的可扩展、语义感知评估器的地位。 

---
# An entropy-optimal path to humble AI 

**Title (ZH)**: 熵优化路径 toward 谦逊人工智能 

**Authors**: Davide Bassetti, Lukáš Pospíšil, Michael Groom, Terence J. O'Kane, Illia Horenko  

**Link**: [PDF](https://arxiv.org/pdf/2506.17940)  

**Abstract**: Progress of AI has led to a creation of very successful, but by no means humble models and tools, especially regarding (i) the huge and further exploding costs and resources they demand, and (ii) the over-confidence of these tools with the answers they provide. Here we introduce a novel mathematical framework for a non-equilibrium entropy-optimizing reformulation of Boltzmann machines based on the exact law of total probability. It results in the highly-performant, but much cheaper, gradient-descent-free learning framework with mathematically-justified existence and uniqueness criteria, and answer confidence/reliability measures. Comparisons to state-of-the-art AI tools in terms of performance, cost and the model descriptor lengths on a set of synthetic problems with varying complexity reveal that the proposed method results in more performant and slim models, with the descriptor lengths being very close to the intrinsic complexity scaling bounds for the underlying problems. Applying this framework to historical climate data results in models with systematically higher prediction skills for the onsets of La Niña and El Niño climate phenomena, requiring just few years of climate data for training - a small fraction of what is necessary for contemporary climate prediction tools. 

**Abstract (ZH)**: 人工智能的进步创造出了非常成功但远非谦逊的模型和工具，尤其体现在(i)它们所需求的巨额且进一步爆炸性的成本和资源上，以及(ii)这些工具对其提供的答案表现出的过高水平的信心。基于精确的全概率定律，我们介绍了一种新颖的数学框架，用于玻尔兹曼机的非平衡最大熵重述。这导致了一种无需梯度下降、高性能且成本更低的学习框架，并具备数学上合理的存在性和唯一性条件，以及答案的信心/可靠性度量。在不同复杂度的合成问题上与最先进的AI工具进行性能、成本和模型描述长度的比较表明，所提出的方法产生了更具高效性和精简性的模型，其描述长度非常接近底层问题的固有复杂性标度界限。将此框架应用于历史气候数据，生成的模型在拉尼娜和厄尔尼诺气候现象的预测技能上系统性地优于现有工具，仅需少量年份的气候数据进行训练，这仅仅是当前气候预测工具所需数据量的一小部分。 

---
# GEMeX-ThinkVG: Towards Thinking with Visual Grounding in Medical VQA via Reinforcement Learning 

**Title (ZH)**: GEMeX-ThinkVG：通过强化学习实现医学VQA中的视觉定位思考 

**Authors**: Bo Liu, Xiangyu Zhao, Along He, Yidi Chen, Huazhu Fu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17939)  

**Abstract**: Medical visual question answering aims to support clinical decision-making by enabling models to answer natural language questions based on medical images. While recent advances in multi-modal learning have significantly improved performance, current methods still suffer from limited answer reliability and poor interpretability, impairing the ability of clinicians and patients to understand and trust model-generated answers. To address this, this work first proposes a Thinking with Visual Grounding (ThinkVG) dataset wherein the answer generation is decomposed into intermediate reasoning steps that explicitly ground relevant visual regions of the medical image, thereby providing fine-grained explainability. Furthermore, we introduce a novel verifiable reward mechanism for reinforcement learning to guide post-training, improving the alignment between the model's reasoning process and its final answer. Remarkably, our method achieves comparable performance using only one-eighth of the training data, demonstrating the efficiency and effectiveness of the proposal. The dataset is available at this https URL. 

**Abstract (ZH)**: 医学视觉问答旨在通过使模型基于医学图像回答自然语言问题来支持临床决策。尽管多模态学习的近期进展显著提高了性能，但当前方法仍存在答案可靠性有限和解释性不佳的问题，影响了临床医生和患者对模型生成答案的理解和信任。为解决这一问题，本工作首先提出了一种带有视觉定位思考（ThinkVG）数据集，将答案生成分解为中间推理步骤，明确地将相关医学图像的视觉区域连接起来，从而提供细粒度的可解释性。此外，我们引入了一种新颖的可验证奖励机制以指导强化学习，在后训练阶段提高模型的推理过程与其最终答案之间的对齐程度。令人惊讶的是，我们的方法仅使用八分之一的训练数据就达到了相当的性能，展示了该提议的高效性和有效性。数据集可通过以下链接获取：this https URL。 

---
# EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations 

**Title (ZH)**: EgoWorld: 使用丰富的外人视角观察转换到自我视角 

**Authors**: Junho Park, Andrew Sangwoo Ye, Taein Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2506.17896)  

**Abstract**: Egocentric vision is essential for both human and machine visual understanding, particularly in capturing the detailed hand-object interactions needed for manipulation tasks. Translating third-person views into first-person views significantly benefits augmented reality (AR), virtual reality (VR) and robotics applications. However, current exocentric-to-egocentric translation methods are limited by their dependence on 2D cues, synchronized multi-view settings, and unrealistic assumptions such as necessity of initial egocentric frame and relative camera poses during inference. To overcome these challenges, we introduce EgoWorld, a novel two-stage framework that reconstructs an egocentric view from rich exocentric observations, including projected point clouds, 3D hand poses, and textual descriptions. Our approach reconstructs a point cloud from estimated exocentric depth maps, reprojects it into the egocentric perspective, and then applies diffusion-based inpainting to produce dense, semantically coherent egocentric images. Evaluated on the H2O and TACO datasets, EgoWorld achieves state-of-the-art performance and demonstrates robust generalization to new objects, actions, scenes, and subjects. Moreover, EgoWorld shows promising results even on unlabeled real-world examples. 

**Abstract (ZH)**: 本体中心视觉对于人类和机器视觉理解都是必不可少的，特别是在捕捉用于操作任务的详细手物交互方面。将第三方视角转换为第一人视角对增强现实（AR）、虚拟现实（VR）和机器人应用有显著益处。然而，当前的从外视角到本视角转换方法受限于其对2D线索的依赖、同步多视角设置以及初始本视角框架和推断期间的相对相机姿态等不现实的假设。为了克服这些挑战，我们引入了EgoWorld，这是一种新颖的两阶段框架，可以从丰富的外视角观察中重构本视角视图，包括投影点云、3D手部姿态和文本描述。我们的方法从估计的外视角深度图中重构点云，将其重新投影到本视角视角，然后应用基于扩散的 inpainting生成密集且语义一致的本视角图像。在H2O和TACO数据集上的评估结果显示，EgoWorld达到了最先进的性能，并展示了对新物体、动作、场景和主体的鲁棒泛化能力。此外，EgoWorld甚至在未标记的真实世界示例上也显示出令人鼓舞的结果。 

---
# CLiViS: Unleashing Cognitive Map through Linguistic-Visual Synergy for Embodied Visual Reasoning 

**Title (ZH)**: CLiViS: 通过语言-视觉协同作用释放认知地图进行具身视觉推理 

**Authors**: Kailing Li, Qi'ao Xu, Tianwen Qian, Yuqian Fu, Yang Jiao, Xiaoling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17629)  

**Abstract**: Embodied Visual Reasoning (EVR) seeks to follow complex, free-form instructions based on egocentric video, enabling semantic understanding and spatiotemporal reasoning in dynamic environments. Despite its promising potential, EVR encounters significant challenges stemming from the diversity of complex instructions and the intricate spatiotemporal dynamics in long-term egocentric videos. Prior solutions either employ Large Language Models (LLMs) over static video captions, which often omit critical visual details, or rely on end-to-end Vision-Language Models (VLMs) that struggle with stepwise compositional reasoning. Consider the complementary strengths of LLMs in reasoning and VLMs in perception, we propose CLiViS. It is a novel training-free framework that leverages LLMs for high-level task planning and orchestrates VLM-driven open-world visual perception to iteratively update the scene context. Building on this synergy, the core of CLiViS is a dynamic Cognitive Map that evolves throughout the reasoning process. This map constructs a structured representation of the embodied scene, bridging low-level perception and high-level reasoning. Extensive experiments across multiple benchmarks demonstrate the effectiveness and generality of CLiViS, especially in handling long-term visual dependencies. Code is available at this https URL. 

**Abstract (ZH)**: 具身视觉推理（EVR）旨在基于第一人称视频遵循复杂的自由形式指令，实现动态环境中的语义理解和时空推理。尽管EVR具有广阔的潜力，但它面临着源自复杂指令多样性和长时间第一人称视频中的错综复杂时空动态的巨大挑战。先前的解决方案要么依赖于在静态视频说明上使用的大型语言模型（LLMs），往往会忽略关键的视觉细节，要么依赖于端到端的视觉语言模型（VLMs），这种模型在逐步组合推理方面存在困难。考虑到LLMs在推理方面的优势和VLMs在感知方面的优势，我们提出了CLiViS。CLiViS是一种新型无需训练的框架，利用LLMs进行高层次任务规划，并通过VLM驱动的开放世界视觉感知迭代更新场景上下文。基于这种协同作用，CLiViS的核心是一个动态的认知地图，该地图在整个推理过程中不断发展。这个地图构建了具身场景的结构化表示，将低层感知与高层推理连接起来。多项基准测试中的广泛实验表明，CLiViS在处理长期视觉依赖关系方面特别有效。代码可在以下链接获取：this https URL。 

---
# Challenges in Grounding Language in the Real World 

**Title (ZH)**: 在现实世界中grounding语言的挑战 

**Authors**: Peter Lindes, Kaoutar Skiker  

**Link**: [PDF](https://arxiv.org/pdf/2506.17375)  

**Abstract**: A long-term goal of Artificial Intelligence is to build a language understanding system that allows a human to collaborate with a physical robot using language that is natural to the human. In this paper we highlight some of the challenges in doing this, and propose a solution that integrates the abilities of a cognitive agent capable of interactive task learning in a physical robot with the linguistic abilities of a large language model. We also point the way to an initial implementation of this approach. 

**Abstract (ZH)**: 人工智能的一个长期目标是构建一个语言理解系统，使人类能够使用自然语言与物理机器人协作。在本文中，我们阐述了实现这一目标的一些挑战，并提出了一种解决方案，即将能够进行交互式任务学习的认知代理能力与大规模语言模型的语文能力整合到物理机器人中。我们还指出了这种方法初步实现的方向。 

---
# Zero-Shot Cognitive Impairment Detection from Speech Using AudioLLM 

**Title (ZH)**: 基于AudioLLM的零样本认知功能障碍语音检测 

**Authors**: Mostafa Shahin, Beena Ahmed, Julien Epps  

**Link**: [PDF](https://arxiv.org/pdf/2506.17351)  

**Abstract**: Cognitive impairment (CI) is of growing public health concern, and early detection is vital for effective intervention. Speech has gained attention as a non-invasive and easily collectible biomarker for assessing cognitive decline. Traditional CI detection methods typically rely on supervised models trained on acoustic and linguistic features extracted from speech, which often require manual annotation and may not generalise well across datasets and languages. In this work, we propose the first zero-shot speech-based CI detection method using the Qwen2- Audio AudioLLM, a model capable of processing both audio and text inputs. By designing prompt-based instructions, we guide the model in classifying speech samples as indicative of normal cognition or cognitive impairment. We evaluate our approach on two datasets: one in English and another multilingual, spanning different cognitive assessment tasks. Our results show that the zero-shot AudioLLM approach achieves performance comparable to supervised methods and exhibits promising generalizability and consistency across languages, tasks, and datasets. 

**Abstract (ZH)**: 认知损害（CI）是日益引起公共卫生关注的问题，早期检测对于有效干预至关重要。语音已成为一种无创且易于收集的生物标志物，用于评估认知衰退。传统的CI检测方法通常依赖于在提取自语音的声学和语言特征上训练的监督模型，这通常需要手动注释，并且可能在不同数据集和语言间泛化效果不佳。在本工作中，我们提出了首个零样本语音基线CI检测方法，使用了具备处理音频和文本输入能力的Qwen2-Audio AudioLLM模型。通过设计指令式的提示，我们在分类语音样本时指导模型将其识别为正常认知或认知损害的迹象。我们在两个数据集上评估了我们的方法：一个为英语数据集，另一个为多语言数据集，涵盖了不同的认知评估任务。结果显示，零样本AudioLLM方法的性能与监督方法相当，并且在语言、任务和数据集方面展现了良好的泛化能力和一致性。 

---
# A Digital Twin Framework for Generation-IV Reactors with Reinforcement Learning-Enabled Health-Aware Supervisory Control 

**Title (ZH)**: 一种基于强化学习实现健康感知监督控制的第四代反应堆数字孪生框架 

**Authors**: Jasmin Y. Lim, Dimitrios Pylorof, Humberto E. Garcia, Karthik Duraisamy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17258)  

**Abstract**: Generation IV (Gen-IV) nuclear power plants are envisioned to replace the current reactor fleet, bringing improvements in performance, safety, reliability, and sustainability. However, large cost investments currently inhibit the deployment of these advanced reactor concepts. Digital twins bridge real-world systems with digital tools to reduce costs, enhance decision-making, and boost operational efficiency. In this work, a digital twin framework is designed to operate the Gen-IV Fluoride-salt-cooled High-temperature Reactor, utilizing data-enhanced methods to optimize operational and maintenance policies while adhering to system constraints. The closed-loop framework integrates surrogate modeling, reinforcement learning, and Bayesian inference to streamline end-to-end communication for online regulation and self-adjustment. Reinforcement learning is used to consider component health and degradation to drive the target power generations, with constraints enforced through a Reference Governor control algorithm that ensures compliance with pump flow rate and temperature limits. These input driving modules benefit from detailed online simulations that are assimilated to measurement data with Bayesian filtering. The digital twin is demonstrated in three case studies: a one-year long-term operational period showcasing maintenance planning capabilities, short-term accuracy refinement with high-frequency measurements, and system shock capturing that demonstrates real-time recalibration capabilities when change in boundary conditions. These demonstrations validate robustness for health-aware and constraint-informed nuclear plant operation, with general applicability to other advanced reactor concepts and complex engineering systems. 

**Abstract (ZH)**: Generation IV (Gen-IV) 核电站数字孪生框架设计与应用 

---
