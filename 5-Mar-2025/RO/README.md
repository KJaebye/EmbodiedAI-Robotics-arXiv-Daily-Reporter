# Reactive Diffusion Policy: Slow-Fast Visual-Tactile Policy Learning for Contact-Rich Manipulation 

**Title (ZH)**: 反应扩散策略：接触丰富操作中的慢-快视觉-触觉政策学习 

**Authors**: Han Xue, Jieji Ren, Wendi Chen, Gu Zhang, Yuan Fang, Guoying Gu, Huazhe Xu, Cewu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02881)  

**Abstract**: Humans can accomplish complex contact-rich tasks using vision and touch, with highly reactive capabilities such as quick adjustments to environmental changes and adaptive control of contact forces; however, this remains challenging for robots. Existing visual imitation learning (IL) approaches rely on action chunking to model complex behaviors, which lacks the ability to respond instantly to real-time tactile feedback during the chunk execution. Furthermore, most teleoperation systems struggle to provide fine-grained tactile / force feedback, which limits the range of tasks that can be performed. To address these challenges, we introduce TactAR, a low-cost teleoperation system that provides real-time tactile feedback through Augmented Reality (AR), along with Reactive Diffusion Policy (RDP), a novel slow-fast visual-tactile imitation learning algorithm for learning contact-rich manipulation skills. RDP employs a two-level hierarchy: (1) a slow latent diffusion policy for predicting high-level action chunks in latent space at low frequency, (2) a fast asymmetric tokenizer for closed-loop tactile feedback control at high frequency. This design enables both complex trajectory modeling and quick reactive behavior within a unified framework. Through extensive evaluation across three challenging contact-rich tasks, RDP significantly improves performance compared to state-of-the-art visual IL baselines through rapid response to tactile / force feedback. Furthermore, experiments show that RDP is applicable across different tactile / force sensors. Code and videos are available on this https URL. 

**Abstract (ZH)**: 基于增强现实的低成本触觉反馈远程操作系统及其反应扩散策略算法：接触丰富操作技能的视触觉模拟学习 

---
# MuBlE: MuJoCo and Blender simulation Environment and Benchmark for Task Planning in Robot Manipulation 

**Title (ZH)**: MuBlE: MuJoCo 和 Blender 组合的仿真环境及机器人 manipulation 任务规划基准 

**Authors**: Michal Nazarczuk, Karla Stepanova, Jan Kristof Behrens, Matej Hoffmann, Krystian Mikolajczyk  

**Link**: [PDF](https://arxiv.org/pdf/2503.02834)  

**Abstract**: Current embodied reasoning agents struggle to plan for long-horizon tasks that require to physically interact with the world to obtain the necessary information (e.g. 'sort the objects from lightest to heaviest'). The improvement of the capabilities of such an agent is highly dependent on the availability of relevant training environments. In order to facilitate the development of such systems, we introduce a novel simulation environment (built on top of robosuite) that makes use of the MuJoCo physics engine and high-quality renderer Blender to provide realistic visual observations that are also accurate to the physical state of the scene. It is the first simulator focusing on long-horizon robot manipulation tasks preserving accurate physics modeling. MuBlE can generate mutlimodal data for training and enable design of closed-loop methods through environment interaction on two levels: visual - action loop, and control - physics loop. Together with the simulator, we propose SHOP-VRB2, a new benchmark composed of 10 classes of multi-step reasoning scenarios that require simultaneous visual and physical measurements. 

**Abstract (ZH)**: 当前的具身推理代理在计划需要物理互动以获得必要信息的长期任务时存在局限（例如，“按轻重排序物体”）。这类代理能力的提升高度依赖于相关训练环境的可用性。为了促进这类系统的开发，我们引入了一个新的仿真环境（基于robosuite构建），该环境利用MuJoCo物理引擎和高质量渲染器Blender提供逼真的视觉观察，准确反映场景的物理状态。MuBlE是第一个专注于保留精确物理建模的长期机器人操作任务的仿真器。MuBlE可以生成多模态数据用于训练，并通过视觉-行动环路和控制-物理环路的环境互动，支持闭环方法的设计。结合该仿真器，我们提出了一个新的基准SHOP-VRB2，包含10类需要同时进行视觉和物理测量的多步推理场景。 

---
# Integral Forms in Matrix Lie Groups 

**Title (ZH)**: 矩阵李群中的整形式 

**Authors**: Timothy D Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2503.02820)  

**Abstract**: Matrix Lie groups provide a language for describing motion in such fields as robotics, computer vision, and graphics. When using these tools, we are often faced with turning infinite-series expressions into more compact finite series (e.g., the Euler-Rodriques formula), which can sometimes be onerous. In this paper, we identify some useful integral forms in matrix Lie group expressions that offer a more streamlined pathway for computing compact analytic results. Moreover, we present some recursive structures in these integral forms that show many of these expressions are interrelated. Key to our approach is that we are able to apply the minimal polynomial for a Lie algebra quite early in the process to keep expressions compact throughout the derivations. With the series approach, the minimal polynomial is usually applied at the end, making it hard to recognize common analytic expressions in the result. We show that our integral method can reproduce several series-derived results from the literature. 

**Abstract (ZH)**: 矩阵李群为描述机器人学、计算机视觉和图形学中的运动提供了一种语言。在使用这些工具时，我们常常需要将无限级数表达式转化为更紧凑的有限级数（例如欧拉-罗德里奇公式），这有时会带来不便。本文中，我们识别了一些在矩阵李群表达式中有用的积分形式，这些形式为计算紧凑的解析结果提供了更简洁的途径。此外，我们展示了这些积分形式中的递归结构，表明许多这些表达式是相互关联的。我们方法的关键在于能够尽早应用李代数的最小多项式，从而在整个推导过程中保持表达式的紧凑性。使用级数方法时，通常在最后才应用最小多项式，从而难以在结果中识别出常见的解析表达式。我们展示了我们的积分方法可以重现文献中由级数方法得到的多个结果。 

---
# Digital Model-Driven Genetic Algorithm for Optimizing Layout and Task Allocation in Human-Robot Collaborative Assemblies 

**Title (ZH)**: 基于数字模型驱动的遗传算法在人机协作装配中优化布局与任务分配 

**Authors**: Christian Cella, Matteo Bruce Robin, Marco Faroni, Andrea Maria Zanchettin, Paolo Rocco  

**Link**: [PDF](https://arxiv.org/pdf/2503.02774)  

**Abstract**: This paper addresses the optimization of human-robot collaborative work-cells before their physical deployment. Most of the times, such environments are designed based on the experience of the system integrators, often leading to sub-optimal solutions. Accurate simulators of the robotic cell, accounting for the presence of the human as well, are available today and can be used in the pre-deployment. We propose an iterative optimization scheme where a digital model of the work-cell is updated based on a genetic algorithm. The methodology focuses on the layout optimization and task allocation, encoding both the problems simultaneously in the design variables handled by the genetic algorithm, while the task scheduling problem depends on the result of the upper-level one. The final solution balances conflicting objectives in the fitness function and is validated to show the impact of the objectives with respect to a baseline, which represents possible initial choices selected based on the human judgment. 

**Abstract (ZH)**: 本文在物理部署之前优化了人机协作工作单元。现有的精确仿真器可以考虑人类的存在，用于事先部署。我们提出了一种迭代优化方案，其中基于遗传算法不断更新工作单元的数字模型。该方法侧重于布局优化和任务分配，同时将两者编码为遗传算法处理的设计变量，而任务调度问题依赖于更高层次的优化结果。最终解决方案在适应性函数中平衡了冲突的目标，并通过与基于人工判断的初始选择进行基线比较，验证了目标的影响。 

---
# Deep Learning-Enhanced Visual Monitoring in Hazardous Underwater Environments with a Swarm of Micro-Robots 

**Title (ZH)**: 基于微机器人群体的深学习增强海底危化环境视觉监测 

**Authors**: Shuang Chen, Yifeng He, Barry Lennox, Farshad Arvin, Amir Atapour-Abarghouei  

**Link**: [PDF](https://arxiv.org/pdf/2503.02752)  

**Abstract**: Long-term monitoring and exploration of extreme environments, such as underwater storage facilities, is costly, labor-intensive, and hazardous. Automating this process with low-cost, collaborative robots can greatly improve efficiency. These robots capture images from different positions, which must be processed simultaneously to create a spatio-temporal model of the facility. In this paper, we propose a novel approach that integrates data simulation, a multi-modal deep learning network for coordinate prediction, and image reassembly to address the challenges posed by environmental disturbances causing drift and rotation in the robots' positions and orientations. Our approach enhances the precision of alignment in noisy environments by integrating visual information from snapshots, global positional context from masks, and noisy coordinates. We validate our method through extensive experiments using synthetic data that simulate real-world robotic operations in underwater settings. The results demonstrate very high coordinate prediction accuracy and plausible image assembly, indicating the real-world applicability of our approach. The assembled images provide clear and coherent views of the underwater environment for effective monitoring and inspection, showcasing the potential for broader use in extreme settings, further contributing to improved safety, efficiency, and cost reduction in hazardous field monitoring. Code is available on this https URL. 

**Abstract (ZH)**: 长期监测和探索极端环境（如水下存储设施）的成本高、劳动密集且危险。通过使用低成本协作机器人自动化这一过程可以大大提高效率。这些机器人从不同位置捕捉图像，必须同时处理以构建设施的时空模型。在本文中，我们提出了一种新颖的方法，该方法整合了数据模拟、多模态深度学习网络进行坐标预测以及图像重组，以应对环境扰动导致的机器人位置和方向漂移和旋转带来的挑战。我们的方法通过整合快照中的视觉信息、来自掩码的全局位置上下文以及嘈杂的坐标，提高了在嘈杂环境中的对齐精度。我们通过使用模拟水下环境中真实机器人操作的合成数据进行广泛实验，验证了我们的方法。实验结果表明，坐标预测精度非常高，并且图像重组合理，表明我们的方法在实际应用中的可行性。重组后的图像提供了清晰连贯的水下环境视图，有助于有效的监测和检查，展示了在极端环境中有更广泛使用潜力，进一步提高了危险领域监测的安全性、效率和成本效益。代码可在以下链接获取。 

---
# Bridging VLM and KMP: Enabling Fine-grained robotic manipulation via Semantic Keypoints Representation 

**Title (ZH)**: 连接VLM和KMP：通过语义关键点表示实现精细的机器人操作 

**Authors**: Junjie Zhu, Huayu Liu, Jin Wang, Bangrong Wen, Kaixiang Huang, Xiaofei Li, Haiyun Zhan, Guodong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02748)  

**Abstract**: From early Movement Primitive (MP) techniques to modern Vision-Language Models (VLMs), autonomous manipulation has remained a pivotal topic in robotics. As two extremes, VLM-based methods emphasize zero-shot and adaptive manipulation but struggle with fine-grained planning. In contrast, MP-based approaches excel in precise trajectory generalization but lack decision-making ability. To leverage the strengths of the two frameworks, we propose VL-MP, which integrates VLM with Kernelized Movement Primitives (KMP) via a low-distortion decision information transfer bridge, enabling fine-grained robotic manipulation under ambiguous situations. One key of VL-MP is the accurate representation of task decision parameters through semantic keypoints constraints, leading to more precise task parameter generation. Additionally, we introduce a local trajectory feature-enhanced KMP to support VL-MP, thereby achieving shape preservation for complex trajectories. Extensive experiments conducted in complex real-world environments validate the effectiveness of VL-MP for adaptive and fine-grained manipulation. 

**Abstract (ZH)**: 从早期的运动原型技术到现代的视觉-语言模型，自主操作一直是机器人研究中的核心议题。基于视觉-语言模型的方法强调零样本和适应性操作，但在精细操作规划上存在局限。相比之下，基于运动原型的方法在精确轨迹泛化方面表现出色，但在决策制定方面存在不足。为了融合两者的长处，我们提出了VL-MP，该方法通过低失真决策信息传递桥梁将视觉-语言模型与核化运动原型（KMP）相结合，从而在模糊情况下实现精细的机器人操作。VL-MP的关键在于通过语义关键点约束准确表示任务决策参数，从而实现更精确的任务参数生成。此外，我们还引入了一种局部轨迹特征增强的KMP，以支持VL-MP，从而实现复杂轨迹的形状保真。在复杂现实环境中的广泛实验验证了VL-MP在适应性和精细操作方面的有效性。 

---
# Variable-Friction In-Hand Manipulation for Arbitrary Objects via Diffusion-Based Imitation Learning 

**Title (ZH)**: 基于扩散推究学习的任意物体自手内操纵摩擦变量控制 

**Authors**: Qiyang Yan, Zihan Ding, Xin Zhou, Adam J. Spiers  

**Link**: [PDF](https://arxiv.org/pdf/2503.02738)  

**Abstract**: Dexterous in-hand manipulation (IHM) for arbitrary objects is challenging due to the rich and subtle contact process. Variable-friction manipulation is an alternative approach to dexterity, previously demonstrating robust and versatile 2D IHM capabilities with only two single-joint fingers. However, the hard-coded manipulation methods for variable friction hands are restricted to regular polygon objects and limited target poses, as well as requiring the policy to be tailored for each object. This paper proposes an end-to-end learning-based manipulation method to achieve arbitrary object manipulation for any target pose on real hardware, with minimal engineering efforts and data collection. The method features a diffusion policy-based imitation learning method with co-training from simulation and a small amount of real-world data. With the proposed framework, arbitrary objects including polygons and non-polygons can be precisely manipulated to reach arbitrary goal poses within 2 hours of training on an A100 GPU and only 1 hour of real-world data collection. The precision is higher than previous customized object-specific policies, achieving an average success rate of 71.3% with average pose error being 2.676 mm and 1.902 degrees. 

**Abstract (ZH)**: 基于端到端学习的任意物体在手灵巧操作方法 

---
# ImpedanceGPT: VLM-driven Impedance Control of Swarm of Mini-drones for Intelligent Navigation in Dynamic Environment 

**Title (ZH)**: 阻抗GPT：基于VLM的微型无人机群阻抗控制以实现智能动态环境下的导航 

**Authors**: Faryal Batool, Malaika Zafar, Yasheerah Yaqoot, Roohan Ahmed Khan, Muhammad Haris Khan, Aleksey Fedoseev, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02723)  

**Abstract**: Swarm robotics plays a crucial role in enabling autonomous operations in dynamic and unpredictable environments. However, a major challenge remains ensuring safe and efficient navigation in environments filled with both dynamic alive (e.g., humans) and dynamic inanimate (e.g., non-living objects) obstacles. In this paper, we propose ImpedanceGPT, a novel system that combines a Vision-Language Model (VLM) with retrieval-augmented generation (RAG) to enable real-time reasoning for adaptive navigation of mini-drone swarms in complex environments.
The key innovation of ImpedanceGPT lies in the integration of VLM and RAG, which provides the drones with enhanced semantic understanding of their surroundings. This enables the system to dynamically adjust impedance control parameters in response to obstacle types and environmental conditions. Our approach not only ensures safe and precise navigation but also improves coordination between drones in the swarm.
Experimental evaluations demonstrate the effectiveness of the system. The VLM-RAG framework achieved an obstacle detection and retrieval accuracy of 80 % under optimal lighting. In static environments, drones navigated dynamic inanimate obstacles at 1.4 m/s but slowed to 0.7 m/s with increased separation around humans. In dynamic environments, speed adjusted to 1.0 m/s near hard obstacles, while reducing to 0.6 m/s with higher deflection to safely avoid moving humans. 

**Abstract (ZH)**: 集群机器人在动态和不可预测环境下实现自主操作中发挥着关键作用。然而，确保在充满动态有生命（例如，人类）和动态无生命（例如，非生活的物体）障碍物的环境中进行安全和高效的导航仍是一项重大挑战。本文提出了一种新颖的系统ImpedanceGPT，该系统将视觉-语言模型（VLM）与检索增强生成（RAG）结合起来，以实现对复杂环境中微型无人机集群适应性导航的实时推理。ImpedanceGPT的关键创新在于VLM和RAG的集成，这为无人机提供了对其周围环境增强的语义理解，使系统能够根据障碍物类型和环境条件动态调整阻抗控制参数。本方法不仅确保了安全和精确的导航，还改善了集群中无人机之间的协调。实验评估证明了该系统的有效性。在最佳光照条件下，VLM-RAG框架实现了80%的障碍物检测和检索准确性。在静态环境中，无人机以1.4 m/s的速度导航动态无生命障碍物，而在人类周围增加间距时减速至0.7 m/s。在动态环境中，当接近坚硬障碍物时速度调整为1.0 m/s，而在高偏转以安全避开移动人类时减速至0.6 m/s。 

---
# Vibration-Assisted Hysteresis Mitigation for Achieving High Compensation Efficiency 

**Title (ZH)**: 振动辅助磁滞损耗 mitigation 以实现高补偿效率 

**Authors**: Myeongbo Park, Chunggil An, Junhyun Park, Jonghyun Kang, Minho Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02720)  

**Abstract**: Tendon-sheath mechanisms (TSMs) are widely used in minimally invasive surgical (MIS) applications, but their inherent hysteresis-caused by friction, backlash, and tendon elongation-leads to significant tracking errors. Conventional modeling and compensation methods struggle with these nonlinearities and require extensive parameter tuning. To address this, we propose a vibration-assisted hysteresis compensation approach, where controlled vibrational motion is applied along the tendon's movement direction to mitigate friction and reduce dead zones. Experimental results demonstrate that the exerted vibration consistently reduces hysteresis across all tested frequencies, decreasing RMSE by up to 23.41% (from 2.2345 mm to 1.7113 mm) and improving correlation, leading to more accurate trajectory tracking. When combined with a Temporal Convolutional Network (TCN)-based compensation model, vibration further enhances performance, achieving an 85.2% reduction in MAE (from 1.334 mm to 0.1969 mm). Without vibration, the TCN-based approach still reduces MAE by 72.3% (from 1.334 mm to 0.370 mm) under the same parameter settings. These findings confirm that vibration effectively mitigates hysteresis, improving trajectory accuracy and enabling more efficient compensation models with fewer trainable parameters. This approach provides a scalable and practical solution for TSM-based robotic applications, particularly in MIS. 

**Abstract (ZH)**: 基于振动辅助的腱鞘机制迟滞补偿方法在微创手术应用中的研究 

---
# Scalable Multi-Robot Task Allocation and Coordination under Signal Temporal Logic Specifications 

**Title (ZH)**: 基于信号时序逻辑规范的大规模多机器人任务分配与协调 

**Authors**: Wenliang Liu, Nathalie Majcherczyk, Federico Pecora  

**Link**: [PDF](https://arxiv.org/pdf/2503.02719)  

**Abstract**: Motion planning with simple objectives, such as collision-avoidance and goal-reaching, can be solved efficiently using modern planners. However, the complexity of the allowed tasks for these planners is limited. On the other hand, signal temporal logic (STL) can specify complex requirements, but STL-based motion planning and control algorithms often face scalability issues, especially in large multi-robot systems with complex dynamics. In this paper, we propose an algorithm that leverages the best of the two worlds. We first use a single-robot motion planner to efficiently generate a set of alternative reference paths for each robot. Then coordination requirements are specified using STL, which is defined over the assignment of paths and robots' progress along those paths. We use a Mixed Integer Linear Program (MILP) to compute task assignments and robot progress targets over time such that the STL specification is satisfied. Finally, a local controller is used to track the target progress. Simulations demonstrate that our method can handle tasks with complex constraints and scales to large multi-robot teams and intricate task allocation scenarios. 

**Abstract (ZH)**: 基于单机器人运动规划和信号时序逻辑的混合算法在复杂约束与大规模多机器人系统中的应用 

---
# Multi-Strategy Enhanced COA for Path Planning in Autonomous Navigation 

**Title (ZH)**: 多策略增强COA自主导航路径规划 

**Authors**: Yifei Wang, Jacky Keung, Haohan Xu, Yuchen Cao, Zhenyu Mao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02700)  

**Abstract**: Autonomous navigation is reshaping various domains in people's life by enabling efficient and safe movement in complex environments. Reliable navigation requires algorithmic approaches that compute optimal or near-optimal trajectories while satisfying task-specific constraints and ensuring obstacle avoidance. However, existing methods struggle with slow convergence and suboptimal solutions, particularly in complex environments, limiting their real-world applicability. To address these limitations, this paper presents the Multi-Strategy Enhanced Crayfish Optimization Algorithm (MCOA), a novel approach integrating three key strategies: 1) Refractive Opposition Learning, enhancing population diversity and global exploration, 2) Stochastic Centroid-Guided Exploration, balancing global and local search to prevent premature convergence, and 3) Adaptive Competition-Based Selection, dynamically adjusting selection pressure for faster convergence and improved solution quality. Empirical evaluations underscore the remarkable planning speed and the amazing solution quality of MCOA in both 3D Unmanned Aerial Vehicle (UAV) and 2D mobile robot path planning. Against 11 baseline algorithms, MCOA achieved a 69.2% reduction in computational time and a 16.7% improvement in minimizing overall path cost in 3D UAV scenarios. Furthermore, in 2D path planning, MCOA outperformed baseline approaches by 44% on average, with an impressive 75.6% advantage in the largest 60*60 grid setting. These findings validate MCOA as a powerful tool for optimizing autonomous navigation in complex environments. The source code is available at: this https URL. 

**Abstract (ZH)**: 自主导航正重新塑造人们生活的各个领域，通过在复杂环境中实现高效和安全的移动。可靠的导航需要算法方法来计算最优或近优轨迹，同时满足特定任务约束并确保避障。然而，现有方法在复杂环境中难以实现快速收敛和最优解，限制了其在实际中的应用。为此，本文提出了一种新型的多策略增强蟹虾优化算法（MCOA），该算法整合了三种关键策略：1）折射反对学习，增强种群多样性和全局探索；2）随机质心引导探索，平衡全局和局部搜索以防止过早收敛；3）自适应竞争选择，动态调整选择压力以实现更快的收敛和更好的解质量。实证评估表明，MCOA 在三维无人驾驶航空器（UAV）和二维移动机器人路径规划中的规划速度和解质量都表现出色。与11种基准算法相比，MCOA 在三维UAV场景中的计算时间减少了69.2%，在整体路径成本最小化方面提高了16.7%。此外，在二维路径规划中，MCOA 平均优于基准方法44%，在60×60的最大网格设置中则表现出75.6%的优势。这些发现验证了MCOA 是优化复杂环境中自主导航的有效工具。源代码可在以下链接获取：this https URL。 

---
# FlowPlan: Zero-Shot Task Planning with LLM Flow Engineering for Robotic Instruction Following 

**Title (ZH)**: FlowPlan: 基于LLM流程工程的零样本任务规划与机器人指令跟随 

**Authors**: Zijun Lin, Chao Tang, Hanjing Ye, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02698)  

**Abstract**: Robotic instruction following tasks require seamless integration of visual perception, task planning, target localization, and motion execution. However, existing task planning methods for instruction following are either data-driven or underperform in zero-shot scenarios due to difficulties in grounding lengthy instructions into actionable plans under operational constraints. To address this, we propose FlowPlan, a structured multi-stage LLM workflow that elevates zero-shot pipeline and bridges the performance gap between zero-shot and data-driven in-context learning methods. By decomposing the planning process into modular stages--task information retrieval, language-level reasoning, symbolic-level planning, and logical evaluation--FlowPlan generates logically coherent action sequences while adhering to operational constraints and further extracts contextual guidance for precise instance-level target localization. Benchmarked on the ALFRED and validated in real-world applications, our method achieves competitive performance relative to data-driven in-context learning methods and demonstrates adaptability across diverse environments. This work advances zero-shot task planning in robotic systems without reliance on labeled data. Project website: this https URL. 

**Abstract (ZH)**: 机器人指令跟随任务需要无缝集成视觉感知、任务规划、目标定位和动作执行。然而，现有的指令跟随任务规划方法要么是数据驱动的，要么在零样本场景下表现不佳，因为在操作约束条件下将 lengthy 的指令转化为可行的计划存在困难。为解决这一问题，我们提出了 FlowPlan，这是一种结构化的多阶段LLM工作流，提升了零样本管道并弥合了零样本和数据驱动的在上下文学习方法之间的性能差距。通过将规划过程分解为模块化阶段——任务信息检索、语言层面推理、符号层面规划和逻辑评估——FlowPlan 生成合乎逻辑的动作序列并遵循操作约束，进一步提取上下文指导以实现精确的实例级目标定位。在 ALFRED 上进行基准测试并通过实际应用验证，我们的方法在与数据驱动的在上下文学习方法相当的同时展示了跨不同环境的适应性。本工作无需标记数据便推进了机器人系统中的零样本任务规划。项目网站: https://this-url. 

---
# Learning-Based Passive Fault-Tolerant Control of a Quadrotor with Rotor Failure 

**Title (ZH)**: 基于学习的旋翼失效条件下 quadrotor 的被动容错控制 

**Authors**: Jiehao Chen, Kaidong Zhao, Zihan Liu, YanJie Li, Yunjiang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02649)  

**Abstract**: This paper proposes a learning-based passive fault-tolerant control (PFTC) method for quadrotor capable of handling arbitrary single-rotor failures, including conditions ranging from fault-free to complete rotor failure, without requiring any rotor fault information or controller switching. Unlike existing methods that treat rotor faults as disturbances and rely on a single controller for multiple fault scenarios, our approach introduces a novel Selector-Controller network structure. This architecture integrates fault detection module and the controller into a unified policy network, effectively combining the adaptability to multiple fault scenarios of PFTC with the superior control performance of active fault-tolerant control (AFTC). To optimize performance, the policy network is trained using a hybrid framework that synergizes reinforcement learning (RL), behavior cloning (BC), and supervised learning with fault information. Extensive simulations and real-world experiments validate the proposed method, demonstrating significant improvements in fault response speed and position tracking performance compared to state-of-the-art PFTC and AFTC approaches. 

**Abstract (ZH)**: 基于学习的被动容错控制方法：适用于处理任意单旋翼故障的四旋翼无人机容错控制 

---
# Human-aligned Safe Reinforcement Learning for Highway On-Ramp Merging in Dense Traffic 

**Title (ZH)**: 面向人类安全的高速公路入口汇入的安全强化学习 

**Authors**: Yang Li, Shijie Yuan, Yuan Chang, Xiaolong Chen, Qisong Yang, Zhiyuan Yang, Hongmao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.02624)  

**Abstract**: Most reinforcement learning (RL) approaches for the decision-making of autonomous driving consider safety as a reward instead of a cost, which makes it hard to balance the tradeoff between safety and other objectives. Human risk preference has also rarely been incorporated, and the trained policy might be either conservative or aggressive for users. To this end, this study proposes a human-aligned safe RL approach for autonomous merging, in which the high-level decision problem is formulated as a constrained Markov decision process (CMDP) that incorporates users' risk preference into the safety constraints, followed by a model predictive control (MPC)-based low-level control. The safety level of RL policy can be adjusted by computing cost limits of CMDP's constraints based on risk preferences and traffic density using a fuzzy control method. To filter out unsafe or invalid actions, we design an action shielding mechanism that pre-executes RL actions using an MPC method and performs collision checks with surrounding agents. We also provide theoretical proof to validate the effectiveness of the shielding mechanism in enhancing RL's safety and sample efficiency. Simulation experiments in multiple levels of traffic densities show that our method can significantly reduce safety violations without sacrificing traffic efficiency. Furthermore, due to the use of risk preference-aware constraints in CMDP and action shielding, we can not only adjust the safety level of the final policy but also reduce safety violations during the training stage, proving a promising solution for online learning in real-world environments. 

**Abstract (ZH)**: 一种考虑用户风险偏好的自主变道安全强化学习方法 

---
# Learning Dexterous In-Hand Manipulation with Multifingered Hands via Visuomotor Diffusion 

**Title (ZH)**: 基于多指手的视觉运动扩散学习手内灵巧操作 

**Authors**: Piotr Koczy, Michael C. Welle, Danica Kragic  

**Link**: [PDF](https://arxiv.org/pdf/2503.02587)  

**Abstract**: We present a framework for learning dexterous in-hand manipulation with multifingered hands using visuomotor diffusion policies. Our system enables complex in-hand manipulation tasks, such as unscrewing a bottle lid with one hand, by leveraging a fast and responsive teleoperation setup for the four-fingered Allegro Hand. We collect high-quality expert demonstrations using an augmented reality (AR) interface that tracks hand movements and applies inverse kinematics and motion retargeting for precise control. The AR headset provides real-time visualization, while gesture controls streamline teleoperation. To enhance policy learning, we introduce a novel demonstration outlier removal approach based on HDBSCAN clustering and the Global-Local Outlier Score from Hierarchies (GLOSH) algorithm, effectively filtering out low-quality demonstrations that could degrade performance. We evaluate our approach extensively in real-world settings and provide all experimental videos on the project website: this https URL 

**Abstract (ZH)**: 我们提出了一种使用多指手进行基于视觉运动扩散策略的灵巧手内操作学习框架。该系统通过利用四指Allegro手的快速响应远程操作设置，实现了复杂的手内操作任务，例如单手旋开瓶盖。我们使用增强现实（AR）接口收集高质量的专家演示，该接口可以追踪手部运动并应用逆动力学和运动移植以实现精确控制。AR头显提供实时可视化，而手势控制简化了远程操作。为了提高策略学习，我们引入了一种基于HDBSCAN聚类和层次全局-局部异常评分（GLOSH）算法的新颖的演示异常值移除方法，有效地过滤出可能影响性能的低质量演示。我们在实际应用场景中进行了广泛评估，并在项目网站上提供了所有实验视频：this https URL。 

---
# Research on visual simultaneous localization and mapping technology based on near infrared light 

**Title (ZH)**: 基于近红外光的视觉同时定位与建图技术研究 

**Authors**: Rui Ma, Mengfang Liu, Boliang Li, Xinghui Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02584)  

**Abstract**: In view of the problems that visual simultaneous localization and mapping (VSLAM) are susceptible to environmental light interference and luminosity inconsistency, the visual simultaneous localization and mapping technology based on near infrared perception (NIR-VSLAM) is proposed. In order to avoid ambient light interference, the near infrared light is innovatively selected as the light source. The luminosity parameter estimation of error energy function, halo factor and exposure time and the light source irradiance correction method are proposed in this paper, which greatly improves the positioning accuracy of Direct Sparse Odometry (DSO). The feasibility of the proposed method in four large scenes is verified, which provides the reference for visual positioning in automatic driving and mobile robot. 

**Abstract (ZH)**: 基于近红外感知的视觉 simultaneous localization and mapping 技术（NIR-VSLAM） 

---
# RaceVLA: VLA-based Racing Drone Navigation with Human-like Behaviour 

**Title (ZH)**: 基于VLA的人类行为似态竞速无人机导航 

**Authors**: Valerii Serpiva, Artem Lykov, Artyom Myshlyaev, Muhammad Haris Khan, Ali Alridha Abdulkarim, Oleg Sautenkov, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02572)  

**Abstract**: RaceVLA presents an innovative approach for autonomous racing drone navigation by leveraging Visual-Language-Action (VLA) to emulate human-like behavior. This research explores the integration of advanced algorithms that enable drones to adapt their navigation strategies based on real-time environmental feedback, mimicking the decision-making processes of human pilots. The model, fine-tuned on a collected racing drone dataset, demonstrates strong generalization despite the complexity of drone racing environments. RaceVLA outperforms OpenVLA in motion (75.0 vs 60.0) and semantic generalization (45.5 vs 36.3), benefiting from the dynamic camera and simplified motion tasks. However, visual (79.6 vs 87.0) and physical (50.0 vs 76.7) generalization were slightly reduced due to the challenges of maneuvering in dynamic environments with varying object sizes. RaceVLA also outperforms RT-2 across all axes - visual (79.6 vs 52.0), motion (75.0 vs 55.0), physical (50.0 vs 26.7), and semantic (45.5 vs 38.8), demonstrating its robustness for real-time adjustments in complex environments. Experiments revealed an average velocity of 1.04 m/s, with a maximum speed of 2.02 m/s, and consistent maneuverability, demonstrating RaceVLA's ability to handle high-speed scenarios effectively. These findings highlight the potential of RaceVLA for high-performance navigation in competitive racing contexts. The RaceVLA codebase, pretrained weights, and dataset are available at this http URL: this https URL 

**Abstract (ZH)**: RaceVLA为自主赛车无人机导航呈现了一种创新方法，通过利用Visual-Language-Action (VLA) 来模拟人类行为。 

---
# World Models for Anomaly Detection during Model-Based Reinforcement Learning Inference 

**Title (ZH)**: 基于模型的强化学习推理期间的异常检测世界模型研究 

**Authors**: Fabian Domberg, Georg Schildbach  

**Link**: [PDF](https://arxiv.org/pdf/2503.02552)  

**Abstract**: Learning-based controllers are often purposefully kept out of real-world applications due to concerns about their safety and reliability. We explore how state-of-the-art world models in Model-Based Reinforcement Learning can be utilized beyond the training phase to ensure a deployed policy only operates within regions of the state-space it is sufficiently familiar with. This is achieved by continuously monitoring discrepancies between a world model's predictions and observed system behavior during inference. It allows for triggering appropriate measures, such as an emergency stop, once an error threshold is surpassed. This does not require any task-specific knowledge and is thus universally applicable. Simulated experiments on established robot control tasks show the effectiveness of this method, recognizing changes in local robot geometry and global gravitational magnitude. Real-world experiments using an agile quadcopter further demonstrate the benefits of this approach by detecting unexpected forces acting on the vehicle. These results indicate how even in new and adverse conditions, safe and reliable operation of otherwise unpredictable learning-based controllers can be achieved. 

**Abstract (ZH)**: 基于世界模型的控制器监控方法：确保部署策略在熟悉的状态空间区域内运行 

---
# Magic in Human-Robot Interaction (HRI) 

**Title (ZH)**: 人类机器人交互中的魔力 

**Authors**: Martin Cooney, Alexey Vinel  

**Link**: [PDF](https://arxiv.org/pdf/2503.02525)  

**Abstract**: "Magic" is referred to here and there in the robotics literature, from "magical moments" afforded by a mobile bubble machine, to "spells" intended to entertain and motivate children--but what exactly could this concept mean for designers? Here, we present (1) some theoretical discussion on how magic could inform interaction designs based on reviewing the literature, followed by (2) a practical description of using such ideas to develop a simplified prototype, which received an award in an international robot magic competition. Although this topic can be considered unusual and some negative connotations exist (e.g., unrealistic thinking can be referred to as magical), our results seem to suggest that magic, in the experiential, supernatural, and illusory senses of the term, could be useful to consider in various robot design contexts, also for artifacts like home assistants and autonomous vehicles--thus, inviting further discussion and exploration. 

**Abstract (ZH)**: 在这里，机器人文献中提到的“魔力”是指从移动气泡机提供的“神奇时刻”到意图娱乐和激励儿童的“法术”——但这个概念对设计师意味着什么？本文首先基于文献综述提出（1）关于“魔力”如何指导交互设计的一些理论讨论，随后介绍（2）如何利用这些想法开发一个简化的原型，该原型在国际机器人魔术竞赛中获奖。尽管这一主题可以被认为是不寻常的，且存在一些负面含义（例如，不切实际的思考可以被称为魔幻思维），但我们的结果似乎表明，在体验、超自然和幻觉的意义上考虑“魔力”，可以在各种机器人设计情境中以及类似于家庭助理和自动驾驶车辆的物件中是有用的，因此，激励进一步的讨论和探索。 

---
# Impact of Temporal Delay on Radar-Inertial Odometry 

**Title (ZH)**: 雷达-惯性里程计中时间延迟的影响 

**Authors**: Vlaho-Josip Štironja, Luka Petrović, Juraj Peršić, Ivan Marković, Ivan Petrović  

**Link**: [PDF](https://arxiv.org/pdf/2503.02509)  

**Abstract**: Accurate ego-motion estimation is a critical component of any autonomous system. Conventional ego-motion sensors, such as cameras and LiDARs, may be compromised in adverse environmental conditions, such as fog, heavy rain, or dust. Automotive radars, known for their robustness to such conditions, present themselves as complementary sensors or a promising alternative within the ego-motion estimation frameworks. In this paper we propose a novel Radar-Inertial Odometry (RIO) system that integrates an automotive radar and an inertial measurement unit. The key contribution is the integration of online temporal delay calibration within the factor graph optimization framework that compensates for potential time offsets between radar and IMU measurements. To validate the proposed approach we have conducted thorough experimental analysis on real-world radar and IMU data. The results show that, even without scan matching or target tracking, integration of online temporal calibration significantly reduces localization error compared to systems that disregard time synchronization, thus highlighting the important role of, often neglected, accurate temporal alignment in radar-based sensor fusion systems for autonomous navigation. 

**Abstract (ZH)**: 基于雷达和惯性测量单元的在线时间延迟校准里程计系统（Radar-Inertial Odometry with Online Temporal Delay Calibration） 

---
# UAV-VLRR: Vision-Language Informed NMPC for Rapid Response in UAV Search and Rescue 

**Title (ZH)**: UAV-VLRR：视觉-语言引导的快速响应无人机搜索与救援非线性模型预测控制 

**Authors**: Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Oleg Sautenkov, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02465)  

**Abstract**: Emergency search and rescue (SAR) operations often require rapid and precise target identification in complex environments where traditional manual drone control is inefficient. In order to address these scenarios, a rapid SAR system, UAV-VLRR (Vision-Language-Rapid-Response), is developed in this research. This system consists of two aspects: 1) A multimodal system which harnesses the power of Visual Language Model (VLM) and the natural language processing capabilities of ChatGPT-4o (LLM) for scene interpretation. 2) A non-linearmodel predictive control (NMPC) with built-in obstacle avoidance for rapid response by a drone to fly according to the output of the multimodal system. This work aims at improving response times in emergency SAR operations by providing a more intuitive and natural approach to the operator to plan the SAR mission while allowing the drone to carry out that mission in a rapid and safe manner. When tested, our approach was faster on an average by 33.75% when compared with an off-the-shelf autopilot and 54.6% when compared with a human pilot. Video of UAV-VLRR: this https URL 

**Abstract (ZH)**: 紧急搜救(SAR)操作往往需要在传统手动无人机控制效率低下的复杂环境中进行快速而精确的目标识别。为了应对这些场景，本研究开发了一种快速SAR系统——UAV-VLRR（Vision-Language-Rapid-Response）。该系统包括两个方面：1) 一个集成了视觉语言模型(VLM)和ChatGPT-4o大规模语言模型(LLL)自然语言处理能力的模态系统，用于场景解释。2) 集成了障碍物避免的非线性模型预测控制(NMPC)，使无人机能够根据模态系统的输出进行快速响应飞行。本工作旨在通过提供一种更直观、更自然的方式来操作员规划SAR任务，同时使无人机能够安全、快速地执行任务，从而改善紧急SAR操作的响应时间。测试结果显示，与现成的自动驾驶系统相比，我们的方法平均快33.75%，与人类飞行员相比快54.6%。UAV-VLRR视频：this https URL 

---
# UAV-VLPA*: A Vision-Language-Path-Action System for Optimal Route Generation on a Large Scales 

**Title (ZH)**: UAV-VLPA*：一种大规模路径规划的视觉-语言-路径-行动系统 

**Authors**: Oleg Sautenkov, Aibek Akhmetkazy, Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Grik Tadevosyan, Artem Lykov, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2503.02454)  

**Abstract**: The UAV-VLPA* (Visual-Language-Planning-and-Action) system represents a cutting-edge advancement in aerial robotics, designed to enhance communication and operational efficiency for unmanned aerial vehicles (UAVs). By integrating advanced planning capabilities, the system addresses the Traveling Salesman Problem (TSP) to optimize flight paths, reducing the total trajectory length by 18.5\% compared to traditional methods. Additionally, the incorporation of the A* algorithm enables robust obstacle avoidance, ensuring safe and efficient navigation in complex environments. The system leverages satellite imagery processing combined with the Visual Language Model (VLM) and GPT's natural language processing capabilities, allowing users to generate detailed flight plans through simple text commands. This seamless fusion of visual and linguistic analysis empowers precise decision-making and mission planning, making UAV-VLPA* a transformative tool for modern aerial operations. With its unmatched operational efficiency, navigational safety, and user-friendly functionality, UAV-VLPA* sets a new standard in autonomous aerial robotics, paving the way for future innovations in the field. 

**Abstract (ZH)**: UAV-VLPA*（视觉-语言-规划与行动）系统代表了飞行机器人领域的前沿进展，旨在提升无人机的通信和操作效率。通过整合先进的规划能力，该系统解决了旅行商问题（TSP），将飞行路径的总轨迹长度减少了18.5%，相较于传统方法更为高效。此外，A*算法的引入使得稳健的障碍物规避成为可能，确保在复杂环境中安全高效的导航。该系统结合了卫星影像处理、视觉语言模型（VLM）和GPT的自然语言处理能力，使用户能够通过简单的文本命令生成详细的飞行计划。这种视觉与语言分析的无缝融合赋予了精确的决策和任务规划能力，使UAV-VLPA*成为现代飞行操作中的一项变革性工具。凭借其无与伦比的操作效率、导航安全性以及用户友好的功能，UAV-VLPA*为自主飞行机器人领域设定了新的标准，为该领域的未来创新铺平了道路。 

---
# SEB-Naver: A SE(2)-based Local Navigation Framework for Car-like Robots on Uneven Terrain 

**Title (ZH)**: SEB-Naver：一种基于SE(2)的地面不平环境下类汽车机器人本地导航框架 

**Authors**: Xiaoying Li, Long Xu, Xiaolin Huang, Donglai Xue, Zhihao Zhang, Zhichao Han, Chao Xu, Yanjun Cao, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02412)  

**Abstract**: Autonomous navigation of car-like robots on uneven terrain poses unique challenges compared to flat terrain, particularly in traversability assessment and terrain-associated kinematic modelling for motion planning. This paper introduces SEB-Naver, a novel SE(2)-based local navigation framework designed to overcome these challenges. First, we propose an efficient traversability assessment method for SE(2) grids, leveraging GPU parallel computing to enable real-time updates and maintenance of local maps. Second, inspired by differential flatness, we present an optimization-based trajectory planning method that integrates terrain-associated kinematic models, significantly improving both planning efficiency and trajectory quality. Finally, we unify these components into SEB-Naver, achieving real-time terrain assessment and trajectory optimization. Extensive simulations and real-world experiments demonstrate the effectiveness and efficiency of our approach. The code is at this https URL. 

**Abstract (ZH)**: 非平坦地形上类似汽车的机器人自主导航面临独特的挑战，特别是在可达性评估和与地形相关的运动规划中的运动学建模方面。本文介绍了一种新的基于SE(2)的局部导航框架SEB-Naver，旨在克服这些挑战。首先，我们提出了一种高效的SE(2)网格可达性评估方法，利用GPU并行计算实现局部地图的实时更新和维护。其次，受微分平坦性启发，我们提出了一种基于优化的轨迹规划方法，结合了与地形相关的运动学模型，显著提高了规划效率和轨迹质量。最后，我们将这些组件统一整合为SEB-Naver，实现了实时地形评估和轨迹优化。广泛的仿真实验和现实世界实验验证了我们方法的有效性和效率。代码链接见https URL。 

---
# Predictive Kinematic Coordinate Control for Aerial Manipulators based on Modified Kinematics Learning 

**Title (ZH)**: 基于改进动力学学习的空中 manipulator 预测运动坐标控制 

**Authors**: Zhengzhen Li, Jiahao Shen, Mengyu Ji, Huazi Cao, Shiyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.02408)  

**Abstract**: High-precision manipulation has always been a developmental goal for aerial manipulators. This paper investigates the kinematic coordinate control issue in aerial manipulators. We propose a predictive kinematic coordinate control method, which includes a learning-based modified kinematic model and a model predictive control (MPC) scheme based on weight allocation. Compared to existing methods, our proposed approach offers several attractive features. First, the kinematic model incorporates closed-loop dynamics characteristics and online residual learning. Compared to methods that do not consider closed-loop dynamics and residuals, our proposed method has improved accuracy by 59.6$\%$. Second, a MPC scheme that considers weight allocation has been proposed, which can coordinate the motion strategies of quadcopters and manipulators. Compared to methods that do not consider weight allocation, the proposed method can meet the requirements of more tasks. The proposed approach is verified through complex trajectory tracking and moving target tracking experiments. The results validate the effectiveness of the proposed method. 

**Abstract (ZH)**: 高精度操作一直是空中 manipulator 发展的目标。本文研究了空中 manipulator 的运动坐标控制问题。我们提出了一种预测性运动坐标控制方法，该方法包括基于学习的改进运动学模型和基于权重分配的模型预测控制（MPC）方案。与现有方法相比，我们提出的方法具有多个吸引特点。首先，运动学模型包含了闭环动力学特性和在线残差学习。与不考虑闭环动力学和残差的方法相比，我们提出的方法准确性提高了59.6%。其次，我们提出了一个考虑权重分配的MPC方案，可以协调四旋翼和 manipulator 的运动策略。与不考虑权重分配的方法相比，所提出的方法可以满足更多的任务需求。所提出的方法通过复杂轨迹跟踪和移动目标跟踪实验得到了验证，结果验证了所提出方法的有效性。 

---
# A comparison of visual representations for real-world reinforcement learning in the context of vacuum gripping 

**Title (ZH)**: 现实世界真空吸放操作中视觉表示方法的比较 

**Authors**: Nico Sutter, Valentin N. Hartmann, Stelian Coros  

**Link**: [PDF](https://arxiv.org/pdf/2503.02405)  

**Abstract**: When manipulating objects in the real world, we need reactive feedback policies that take into account sensor information to inform decisions. This study aims to determine how different encoders can be used in a reinforcement learning (RL) framework to interpret the spatial environment in the local surroundings of a robot arm. Our investigation focuses on comparing real-world vision with 3D scene inputs, exploring new architectures in the process. We built on the SERL framework, providing us with a sample efficient and stable RL foundation we could build upon, while keeping training times minimal. The results of this study indicate that spatial information helps to significantly outperform the visual counterpart, tested on a box picking task with a vacuum gripper. The code and videos of the evaluations are available at this https URL. 

**Abstract (ZH)**: 在真实世界中操作物体时，我们需要反应性反馈策略来利用传感器信息来指导决策。本研究旨在探讨在机器臂周边环境中，不同编码器在强化学习框架中如何用于解读空间环境。我们的研究重点在于比较实际视觉输入和3D场景输入的表现，并在此过程中探索新的网络架构。我们基于SERL框架，提供了高效且稳定的学习基础，并保持了较低的训练时间。研究结果表明，空间信息在盒子拾取任务中显著优于视觉信息，使用真空吸盘进行测试。评估的代码和视频可在以下链接获取。 

---
# RGBSQGrasp: Inferring Local Superquadric Primitives from Single RGB Image for Graspability-Aware Bin Picking 

**Title (ZH)**: RGBSQ抓取：从单张RGB图像中推断局部超二次原始几何体以实现考虑抓取性的 bin 选择 

**Authors**: Yifeng Xu, Fan Zhu, Ye Li, Sebastian Ren, Xiaonan Huang, Yuhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02387)  

**Abstract**: Bin picking is a challenging robotic task due to occlusions and physical constraints that limit visual information for object recognition and grasping. Existing approaches often rely on known CAD models or prior object geometries, restricting generalization to novel or unknown objects. Other methods directly regress grasp poses from RGB-D data without object priors, but the inherent noise in depth sensing and the lack of object understanding make grasp synthesis and evaluation more difficult. Superquadrics (SQ) offer a compact, interpretable shape representation that captures the physical and graspability understanding of objects. However, recovering them from limited viewpoints is challenging, as existing methods rely on multiple perspectives for near-complete point cloud reconstruction, limiting their effectiveness in bin-picking. To address these challenges, we propose \textbf{RGBSQGrasp}, a grasping framework that leverages superquadric shape primitives and foundation metric depth estimation models to infer grasp poses from a monocular RGB camera -- eliminating the need for depth sensors. Our framework integrates a universal, cross-platform dataset generation pipeline, a foundation model-based object point cloud estimation module, a global-local superquadric fitting network, and an SQ-guided grasp pose sampling module. By integrating these components, RGBSQGrasp reliably infers grasp poses through geometric reasoning, enhancing grasp stability and adaptability to unseen objects. Real-world robotic experiments demonstrate a 92\% grasp success rate, highlighting the effectiveness of RGBSQGrasp in packed bin-picking environments. 

**Abstract (ZH)**: RGBSQGrasp：基于超二次曲面的单目视觉抓取框架 

---
# Introspective Loop Closure for SLAM with 4D Imaging Radar 

**Title (ZH)**: 基于4D成像雷达的introspective环回闭合SLAM 

**Authors**: Maximilian Hilger, Vladimír Kubelka, Daniel Adolfsson, Ralf Becker, Henrik Andreasson, Achim J. Lilienthal  

**Link**: [PDF](https://arxiv.org/pdf/2503.02383)  

**Abstract**: Simultaneous Localization and Mapping (SLAM) allows mobile robots to navigate without external positioning systems or pre-existing maps. Radar is emerging as a valuable sensing tool, especially in vision-obstructed environments, as it is less affected by particles than lidars or cameras. Modern 4D imaging radars provide three-dimensional geometric information and relative velocity measurements, but they bring challenges, such as a small field of view and sparse, noisy point clouds. Detecting loop closures in SLAM is critical for reducing trajectory drift and maintaining map accuracy. However, the directional nature of 4D radar data makes identifying loop closures, especially from reverse viewpoints, difficult due to limited scan overlap. This article explores using 4D radar for loop closure in SLAM, focusing on similar and opposing viewpoints. We generate submaps for a denser environment representation and use introspective measures to reject false detections in feature-degenerate environments. Our experiments show accurate loop closure detection in geometrically diverse settings for both similar and opposing viewpoints, improving trajectory estimation with up to 82 % improvement in ATE and rejecting false positives in self-similar environments. 

**Abstract (ZH)**: 4D雷达在SLAM中同时定位与建图中的循环闭合检测 

---
# JPDS-NN: Reinforcement Learning-Based Dynamic Task Allocation for Agricultural Vehicle Routing Optimization 

**Title (ZH)**: JPDS-NN：基于强化学习的农业车辆路线优化动态任务分配 

**Authors**: Yixuan Fan, Haotian Xu, Mengqiao Liu, Qing Zhuo, Tao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02369)  

**Abstract**: The Entrance Dependent Vehicle Routing Problem (EDVRP) is a variant of the Vehicle Routing Problem (VRP) where the scale of cities influences routing outcomes, necessitating consideration of their entrances. This paper addresses EDVRP in agriculture, focusing on multi-parameter vehicle planning for irregularly shaped fields. To address the limitations of traditional methods, such as heuristic approaches, which often overlook field geometry and entrance constraints, we propose a Joint Probability Distribution Sampling Neural Network (JPDS-NN) to effectively solve the EDVRP. The network uses an encoder-decoder architecture with graph transformers and attention mechanisms to model routing as a Markov Decision Process, and is trained via reinforcement learning for efficient and rapid end-to-end planning. Experimental results indicate that JPDS-NN reduces travel distances by 48.4-65.4%, lowers fuel consumption by 14.0-17.6%, and computes two orders of magnitude faster than baseline methods, while demonstrating 15-25% superior performance in dynamic arrangement scenarios. Ablation studies validate the necessity of cross-attention and pre-training. The framework enables scalable, intelligent routing for large-scale farming under dynamic constraints. 

**Abstract (ZH)**: 基于入口依赖的车辆路由问题（EDVRP）在农业中的研究：基于联合概率分布采样的神经网络（JPDS-NN）多参数车辆规划 

---
# Controllable Motion Generation via Diffusion Modal Coupling 

**Title (ZH)**: 可控运动生成 via 推荐模态耦合 

**Authors**: Luobin Wang, Hongzhan Yu, Chenning Yu, Sicun Gao, Henrik Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02353)  

**Abstract**: Diffusion models have recently gained significant attention in robotics due to their ability to generate multi-modal distributions of system states and behaviors. However, a key challenge remains: ensuring precise control over the generated outcomes without compromising realism. This is crucial for applications such as motion planning or trajectory forecasting, where adherence to physical constraints and task-specific objectives is essential. We propose a novel framework that enhances controllability in diffusion models by leveraging multi-modal prior distributions and enforcing strong modal coupling. This allows us to initiate the denoising process directly from distinct prior modes that correspond to different possible system behaviors, ensuring sampling to align with the training distribution. We evaluate our approach on motion prediction using the Waymo dataset and multi-task control in Maze2D environments. Experimental results show that our framework outperforms both guidance-based techniques and conditioned models with unimodal priors, achieving superior fidelity, diversity, and controllability, even in the absence of explicit conditioning. Overall, our approach provides a more reliable and scalable solution for controllable motion generation in robotics. 

**Abstract (ZH)**: 基于扩散模型的鲁棒可控运动生成方法 

---
# Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding 

**Title (ZH)**: 基于动作切片集成的并行解码加速视觉-语言-动作模型 

**Authors**: Wenxuan Song, Jiayi Chen, Pengxiang Ding, Han Zhao, Wei Zhao, Zhide Zhong, Zongyuan Ge, Jun Ma, Haoang Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02310)  

**Abstract**: Vision-Language-Action (VLA) models demonstrate remarkable potential for generalizable robotic manipulation. The performance of VLA models can be improved by integrating with action chunking, a critical technique for effective control. However, action chunking linearly scales up action dimensions in VLA models with increased chunking sizes. This reduces the inference efficiency. To tackle this problem, we propose PD-VLA, the first parallel decoding framework for VLA models integrated with action chunking. Our framework reformulates autoregressive decoding as a nonlinear system solved by parallel fixed-point iterations. This approach preserves model performance with mathematical guarantees while significantly improving decoding speed. In addition, it enables training-free acceleration without architectural changes, as well as seamless synergy with existing acceleration techniques. Extensive simulations validate that our PD-VLA maintains competitive success rates while achieving 2.52 times execution frequency on manipulators (with 7 degrees of freedom) compared with the fundamental VLA model. Furthermore, we experimentally identify the most effective settings for acceleration. Finally, real-world experiments validate its high applicability across different tasks. 

**Abstract (ZH)**: 基于视觉-语言-动作的并行解码框架（PD-VLA）：结合动作分块的高效机器人操作 modeling 

---
# Diffusion-Based mmWave Radar Point Cloud Enhancement Driven by Range Images 

**Title (ZH)**: 基于扩散范围图像驱动的毫米波雷达点云增强 

**Authors**: Ruixin Wu, Zihan Li, Jin Wang, Xiangyu Xu, Huan Yu, Zhi Zheng, Kaixiang Huang, Guodong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02300)  

**Abstract**: Millimeter-wave (mmWave) radar has attracted significant attention in robotics and autonomous driving. However, despite the perception stability in harsh environments, the point cloud generated by mmWave radar is relatively sparse while containing significant noise, which limits its further development. Traditional mmWave radar enhancement approaches often struggle to leverage the effectiveness of diffusion models in super-resolution, largely due to the unnatural range-azimuth heatmap (RAH) or bird's eye view (BEV) representation. To overcome this limitation, we propose a novel method that pioneers the application of fusing range images with image diffusion models, achieving accurate and dense mmWave radar point clouds that are similar to LiDAR. Benefitting from the projection that aligns with human observation, the range image representation of mmWave radar is close to natural images, allowing the knowledge from pre-trained image diffusion models to be effectively transferred, significantly improving the overall performance. Extensive evaluations on both public datasets and self-constructed datasets demonstrate that our approach provides substantial improvements, establishing a new state-of-the-art performance in generating truly three-dimensional LiDAR-like point clouds via mmWave radar. 

**Abstract (ZH)**: 毫米波雷达（mmWave雷达）在机器人和自动驾驶领域引起了广泛关注。然而，尽管在恶劣环境中具有感知稳定性，mmWave雷达生成的点云相对稀疏且含有大量噪声，这限制了其进一步发展。传统的mmWave雷达增强方法往往难以充分发挥扩散模型在超分辨率上的有效性，主要原因是不自然的距离-方位热图（RAH）或鸟瞰图（BEV）表示。为克服这一限制，我们提出了一种新颖的方法，首次将距离图像与图像扩散模型融合，实现了类似于LiDAR的精确且密集的mmWave雷达点云。得益于与人类观测相一致的投影，mmWave雷达的距离图像表示接近自然图像，使得预训练的图像扩散模型知识能够有效地迁移，显著提高整体性能。在公共数据集和自构建数据集上的广泛评估表明，我们的方法提供了实质性的改进，建立了通过mmWave雷达生成真正三维LiDAR-like点云的新最先进性能。 

---
# Model-Based Capacitive Touch Sensing in Soft Robotics: Achieving Robust Tactile Interactions for Artistic Applications 

**Title (ZH)**: 基于模型的柔性机器人电容式触觉传感：实现艺术应用中的稳健触觉交互 

**Authors**: Carolina Silva-Plata, Carlos Rosel, Barnabas Gavin Cangan, Hosam Alagi, Björn Hein, Robert K. Katzschmann, Rubén Fernández, Yosra Mojtahedi, Stefan Escaida Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2503.02280)  

**Abstract**: In this paper, we present a touch technology to achieve tactile interactivity for human-robot interaction (HRI) in soft robotics. By combining a capacitive touch sensor with an online solid mechanics simulation provided by the SOFA framework, contact detection is achieved for arbitrary shapes. Furthermore, the implementation of the capacitive touch technology presented here is selectively sensitive to human touch (conductive objects), while it is largely unaffected by the deformations created by the pneumatic actuation of our soft robot. Multi-touch interactions are also possible. We evaluated our approach with an organic soft robotics sculpture that was created by a visual artist. In particular, we evaluate that the touch localization capabilities are robust under the deformation of the device. We discuss the potential this approach has for the arts and entertainment as well as other domains. 

**Abstract (ZH)**: 本文介绍了一种触觉技术，用于实现软机器人中的人机交互（HRI）的触觉互动。通过将电容式触摸传感器与由SOFA框架提供的在线固体力学模拟结合，实现了任意形状的接触检测。此外，本文呈现的电容式触摸技术对人类触摸（导电物体）具有选择性敏感性，而对由我们软机器人气动驱动产生的形变影响较小。多点触摸交互也是可能的。我们利用一位视觉艺术家创作的有机软机器人雕塑评估了该方法。特别地，我们评估了在设备形变情况下触摸定位能力的鲁棒性。我们探讨了该方法在艺术与娱乐以及其他领域的潜在应用价值。 

---
# Active Robot Curriculum Learning from Online Human Demonstrations 

**Title (ZH)**: 基于在线人类示范的活性机器人课程学习 

**Authors**: Muhan Hou, Koen Hindriks, A.E. Eiben, Kim Baraka  

**Link**: [PDF](https://arxiv.org/pdf/2503.02277)  

**Abstract**: Learning from Demonstrations (LfD) allows robots to learn skills from human users, but its effectiveness can suffer due to sub-optimal teaching, especially from untrained demonstrators. Active LfD aims to improve this by letting robots actively request demonstrations to enhance learning. However, this may lead to frequent context switches between various task situations, increasing the human cognitive load and introducing errors to demonstrations. Moreover, few prior studies in active LfD have examined how these active query strategies may impact human teaching in aspects beyond user experience, which can be crucial for developing algorithms that benefit both robot learning and human teaching. To tackle these challenges, we propose an active LfD method that optimizes the query sequence of online human demonstrations via Curriculum Learning (CL), where demonstrators are guided to provide demonstrations in situations of gradually increasing difficulty. We evaluate our method across four simulated robotic tasks with sparse rewards and conduct a user study (N=26) to investigate the influence of active LfD methods on human teaching regarding teaching performance, post-guidance teaching adaptivity, and teaching transferability. Our results show that our method significantly improves learning performance compared to three other LfD baselines in terms of the final success rate of the converged policy and sample efficiency. Additionally, results from our user study indicate that our method significantly reduces the time required from human demonstrators and decreases failed demonstration attempts. It also enhances post-guidance human teaching in both seen and unseen scenarios compared to another active LfD baseline, indicating enhanced teaching performance, greater post-guidance teaching adaptivity, and better teaching transferability achieved by our method. 

**Abstract (ZH)**: 基于示例的主动学习（Active LfD）通过课程学习优化人类演示的查询序列，提高机器人技能学习性能及人类教学效果 

---
# ForaNav: Insect-inspired Online Target-oriented Navigation for MAVs in Tree Plantations 

**Title (ZH)**: ForaNav：基于昆虫启发的面向目标的 MAVs 树植造林在线导航 

**Authors**: Weijie Kuang, Hann Woei Ho, Ye Zhou, Shahrel Azmin Suandi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02275)  

**Abstract**: Autonomous Micro Air Vehicles (MAVs) are becoming essential in precision agriculture to enhance efficiency and reduce labor costs through targeted, real-time operations. However, existing unmanned systems often rely on GPS-based navigation, which is prone to inaccuracies in rural areas and limits flight paths to predefined routes, resulting in operational inefficiencies. To address these challenges, this paper presents ForaNav, an insect-inspired navigation strategy for autonomous navigation in plantations. The proposed method employs an enhanced Histogram of Oriented Gradient (HOG)-based tree detection approach, integrating hue-saturation histograms and global HOG feature variance with hierarchical HOG extraction to distinguish oil palm trees from visually similar objects. Inspired by insect foraging behavior, the MAV dynamically adjusts its path based on detected trees and employs a recovery mechanism to stay on course if a target is temporarily lost. We demonstrate that our detection method generalizes well to different tree types while maintaining lower CPU usage, lower temperature, and higher FPS than lightweight deep learning models, making it well-suited for real-time applications. Flight test results across diverse real-world scenarios show that the MAV successfully detects and approaches all trees without prior tree location, validating its effectiveness for agricultural automation. 

**Abstract (ZH)**: 自主微型空中 Vehicles (MAVs) 在精准农业中的应用通过目标导向的实时操作提高了效率并减少了劳动力成本。然而，现有的无人系统通常依赖于基于GPS的导航，这在农村区域容易出现不准确性，并限制飞行路径为预定义路线，导致操作效率低下。为了解决这些挑战，本文提出了一种受昆虫启发的导航策略ForaNav，以实现植物园中的自主导航。该方法采用改进的基于方向梯度直方图（HOG）的树木检测方法，结合色调饱和度直方图和全局HOG特征方差及分层HOG提取，以区分油棕榈树与其他视觉相似的物体。受昆虫觅食行为的启发，MAV根据检测到的树木动态调整其路径，并采用恢复机制以防止单一目标暂时丢失。我们证明，我们的检测方法在不同树木类型上具有良好的泛化能力，同时具有较低的CPU使用率、较低的温度和更高的FPS，使其适用于实时应用。在多种真实世界场景下的飞行测试结果表明，MAV能够成功地检测并接近所有树木，无需事先知道树木位置，验证了其在农业自动化中的有效性。 

---
# Towards Fluorescence-Guided Autonomous Robotic Partial Nephrectomy on Novel Tissue-Mimicking Hydrogel Phantoms 

**Title (ZH)**: 面向荧光引导的自主机器人部分肾切除手术新型组织模拟水凝胶phantom研究 

**Authors**: Ethan Kilmer, Joseph Chen, Jiawei Ge, Preksha Sarda, Richard Cha, Kevin Cleary, Lauren Shepard, Ahmed Ezzat Ghazi, Paul Maria Scheikl, Axel Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2503.02265)  

**Abstract**: Autonomous robotic systems hold potential for improving renal tumor resection accuracy and patient outcomes. We present a fluorescence-guided robotic system capable of planning and executing incision paths around exophytic renal tumors with a clinically relevant resection margin. Leveraging point cloud observations, the system handles irregular tumor shapes and distinguishes healthy from tumorous tissue based on near-infrared imaging, akin to indocyanine green staining in partial nephrectomy. Tissue-mimicking phantoms are crucial for the development of autonomous robotic surgical systems for interventions where acquiring ex-vivo animal tissue is infeasible, such as cancer of the kidney and renal pelvis. To this end, we propose novel hydrogel-based kidney phantoms with exophytic tumors that mimic the physical and visual behavior of tissue, and are compatible with electrosurgical instruments, a common limitation of silicone-based phantoms. In contrast to previous hydrogel phantoms, we mix the material with near-infrared dye to enable fluorescence-guided tumor segmentation. Autonomous real-world robotic experiments validate our system and phantoms, achieving an average margin accuracy of 1.44 mm in a completion time of 69 sec. 

**Abstract (ZH)**: 自主机器人系统有潜力提高肾肿瘤切除准确性和患者 outcomes。我们提出了一种荧光引导的机器人系统，能够规划并执行围绕外生性肾肿瘤的切口路径，同时保持临床相关的切除边缘。该系统利用点云观察，处理不规则的肿瘤形状，并根据近红外成像区分健康组织和肿瘤组织，类似于肾部分切除术中的吲哚菁绿染色。对于在获取离体动物组织不可行的情况下进行的干预，如肾癌和肾盂癌，仿组织水凝胶假体对于自主机器人外科手术系统的发展至关重要。为此，我们提出了一种新型水凝胶基肾脏假体，具有外生性肿瘤，能够模拟组织的物理和视觉行为，并与电外科器械兼容，后者是基于硅胶的假体的常见限制。与之前的水凝胶假体不同，我们将材料与近红外染料混合，以实现荧光引导的肿瘤分割。自主现实世界机器人实验验证了我们的系统和假体，在完成时间为69秒的情况下，平均边缘准确度为1.44毫米。 

---
# Continual Multi-Robot Learning from Black-Box Visual Place Recognition Models 

**Title (ZH)**: 黑箱视觉定位模型的持续多机器人学习 

**Authors**: Kenta Tsukahara, Kanji Tanaka, Daiki Iwata, Jonathan Tay Yu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02256)  

**Abstract**: In the context of visual place recognition (VPR), continual learning (CL) techniques offer significant potential for avoiding catastrophic forgetting when learning new places. However, existing CL methods often focus on knowledge transfer from a known model to a new one, overlooking the existence of unknown black-box models. We explore a novel multi-robot CL approach that enables knowledge transfer from black-box VPR models (teachers), such as those of local robots encountered by traveler robots (students) in unknown environments. Specifically, we introduce Membership Inference Attack, or MIA, the only major privacy attack applicable to black-box models, and leverage it to reconstruct pseudo training sets, which serve as the key knowledge to be exchanged between robots, from black-box VPR models. Furthermore, we aim to overcome the inherently low sampling efficiency of MIA by leveraging insights on place class prediction distribution and un-learned class detection imported from the VPR literature as a prior distribution. We also analyze both the individual effects of these methods and their combined impact. Experimental results demonstrate that our black-box MIA (BB-MIA) approach is remarkably powerful despite its simplicity, significantly enhancing the VPR capability of lower-performing robots through brief communication with other robots. This study contributes to optimizing knowledge sharing between robots in VPR and enhancing autonomy in open-world environments with multi-robot systems that are fault-tolerant and scalable. 

**Abstract (ZH)**: 基于视觉地方识别的黑盒连续学习方法：克服遗忘并增强多机器人系统的自主能力 

---
# Large Language Models as Natural Selector for Embodied Soft Robot Design 

**Title (ZH)**: 大型语言模型作为体现式软机器人设计的自然选择者 

**Authors**: Changhe Chen, Xiaohao Xu, Xiangdong Wang, Xiaonan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02249)  

**Abstract**: Designing soft robots is a complex and iterative process that demands cross-disciplinary expertise in materials science, mechanics, and control, often relying on intuition and extensive experimentation. While Large Language Models (LLMs) have demonstrated impressive reasoning abilities, their capacity to learn and apply embodied design principles--crucial for creating functional robotic systems--remains largely unexplored. This paper introduces RoboCrafter-QA, a novel benchmark to evaluate whether LLMs can learn representations of soft robot designs that effectively bridge the gap between high-level task descriptions and low-level morphological and material choices. RoboCrafter-QA leverages the EvoGym simulator to generate a diverse set of soft robot design challenges, spanning robotic locomotion, manipulation, and balancing tasks. Our experiments with state-of-the-art multi-modal LLMs reveal that while these models exhibit promising capabilities in learning design representations, they struggle with fine-grained distinctions between designs with subtle performance differences. We further demonstrate the practical utility of LLMs for robot design initialization. Our code and benchmark will be available to encourage the community to foster this exciting research direction. 

**Abstract (ZH)**: 设计软机器人是一个复杂且迭代的过程，需要材料科学、力学和控制等跨学科的专业知识，通常依赖于直觉和大量的实验。尽管大型语言模型（LLMs）展示了卓越的推理能力，但它们学习和应用体现于设计原理的能力——对于创建功能性机器人系统至关重要——仍然鲜有探索。本文介绍了一种新的基准RoboCrafter-QA，用于评估LLMs是否能够学习代表软机器人设计的表示，这些表示能够有效地弥合高层次任务描述与低层次形态和材料选择之间的差距。RoboCrafter-QA 利用EvoGym模拟器生成涵盖机器人运动、操作和平衡任务的多样化软机器人设计挑战。我们的实验显示，尽管这些模型在学习设计表示方面表现出有前景的能力，但在辨别细微性能差异的设计之间仍然存在问题。我们进一步展示了LLMs在机器人设计初始化方面的实际应用价值。我们将提供我们的代码和基准，以鼓励社区推动这一令人兴奋的研究方向。 

---
# ADMM-MCBF-LCA: A Layered Control Architecture for Safe Real-Time Navigation 

**Title (ZH)**: ADMM-MCBF-LCA：一种安全实时导航的分层控制架构 

**Authors**: Anusha Srikanthan, Yifan Xue, Vijay Kumar, Nikolai Matni, Nadia Figueroa  

**Link**: [PDF](https://arxiv.org/pdf/2503.02208)  

**Abstract**: We consider the problem of safe real-time navigation of a robot in a dynamic environment with moving obstacles of arbitrary smooth geometries and input saturation constraints. We assume that the robot detects and models nearby obstacle boundaries with a short-range sensor and that this detection is error-free. This problem presents three main challenges: i) input constraints, ii) safety, and iii) real-time computation. To tackle all three challenges, we present a layered control architecture (LCA) consisting of an offline path library generation layer, and an online path selection and safety layer. To overcome the limitations of reactive methods, our offline path library consists of feasible controllers, feedback gains, and reference trajectories. To handle computational burden and safety, we solve online path selection and generate safe inputs that run at 100 Hz. Through simulations on Gazebo and Fetch hardware in an indoor environment, we evaluate our approach against baselines that are layered, end-to-end, or reactive. Our experiments demonstrate that among all algorithms, only our proposed LCA is able to complete tasks such as reaching a goal, safely. When comparing metrics such as safety, input error, and success rate, we show that our approach generates safe and feasible inputs throughout the robot execution. 

**Abstract (ZH)**: 我们考虑在具有任意光滑几何形状的移动障碍物和输入饱和约束的动态环境中，确保机器人实时安全导航的问题。我们假设机器人能够通过短距离传感器准确检测并建模附近障碍物边界。该问题主要包含三个挑战：i）输入约束，ii）安全性，iii）实时计算。为应对这三个挑战，我们提出了一种分层控制架构（LCA），包括离线路径库生成层和在线路径选择与安全层。为克服反应式方法的局限性，我们的离线路径库包含可行控制器、反馈增益和参考轨迹。为处理计算负担和确保安全性，我们在在线路径选择中生成安全输入，并以100 Hz的频率运行。通过在室内环境中的Gazebo和Fetch硬件上的仿真实验，我们将我们的方法与层次化、端到端或反应式的基线方法进行了对比。实验结果表明，只有我们提出的LCA能够安全地完成任务，如到达目标。在比较安全性、输入误差和成功率等指标时，我们证明了我们的方法在整个机器人执行过程中生成了安全且可行的输入。 

---
# Zero-Shot Sim-to-Real Visual Quadrotor Control with Hard Constraints 

**Title (ZH)**: 零样本仿真实践视觉四旋翼控制带硬约束 

**Authors**: Yan Miao, Will Shen, Sayan Mitra  

**Link**: [PDF](https://arxiv.org/pdf/2503.02198)  

**Abstract**: We present the first framework demonstrating zero-shot sim-to-real transfer of visual control policies learned in a Neural Radiance Field (NeRF) environment for quadrotors to fly through racing gates. Robust transfer from simulation to real flight poses a major challenge, as standard simulators often lack sufficient visual fidelity. To address this, we construct a photorealistic simulation environment of quadrotor racing tracks, called FalconGym, which provides effectively unlimited synthetic images for training. Within FalconGym, we develop a pipelined approach for crossing gates that combines (i) a Neural Pose Estimator (NPE) coupled with a Kalman filter to reliably infer quadrotor poses from single-frame RGB images and IMU data, and (ii) a self-attention-based multi-modal controller that adaptively integrates visual features and pose estimation. This multi-modal design compensates for perception noise and intermittent gate visibility. We train this controller purely in FalconGym with imitation learning and deploy the resulting policy to real hardware with no additional fine-tuning. Simulation experiments on three distinct tracks (circle, U-turn and figure-8) demonstrate that our controller outperforms a vision-only state-of-the-art baseline in both success rate and gate-crossing accuracy. In 30 live hardware flights spanning three tracks and 120 gates, our controller achieves a 95.8% success rate and an average error of just 10 cm when flying through 38 cm-radius gates. 

**Abstract (ZH)**: 我们提出了第一个框架，该框架展示了通过神经辐射场（NeRF）环境学习的视觉控制策略在quadrotor从仿真到現實飞行中的零样本迁移。我们构建了名为FalconGym的真实感仿真实验环境，提供了无限的合成图像用于训练。在FalconGym中，我们开发了一种流水线方法来穿越门，该方法结合了（i）一种与卡尔曼滤波器耦合的神经位姿估计器（NPE），用于可靠地从单帧RGB图像和IMU数据中推断quadrotor位姿；和（ii）一种基于自我注意力的多模态控制器，该控制器能够自适应地整合视觉特征和位姿估计。这种多模态设计补偿了感知噪声和门的间歇性可见性。我们仅使用演示学习在FalconGym中训练该控制器，并在无需额外微调的情况下将其部署到实际硬件中。针对三个不同赛道（圆形、U型和8字形）的仿真实验表明，我们的控制器在成功率和门穿越精度方面均优于现有的仅基于视觉的先进基线。在跨越38厘米半径的120个门的30次实际硬件飞行中，我们的控制器实现了95.8%的成功率和平均每错误位移仅10厘米。标题：

零样本视觉控制策略从仿真到现实飞行的quadrotor门穿越迁移 

---
# RPF-Search: Field-based Search for Robot Person Following in Unknown Dynamic Environments 

**Title (ZH)**: 基于字段的搜索：未知动态环境中机器人跟随人员的搜索方法 

**Authors**: Hanjing Ye, Kuanqi Cai, Yu Zhan, Bingyi Xia, Arash Ajoudani, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02188)  

**Abstract**: Autonomous robot person-following (RPF) systems are crucial for personal assistance and security but suffer from target loss due to occlusions in dynamic, unknown environments. Current methods rely on pre-built maps and assume static environments, limiting their effectiveness in real-world settings. There is a critical gap in re-finding targets under topographic (e.g., walls, corners) and dynamic (e.g., moving pedestrians) occlusions. In this paper, we propose a novel heuristic-guided search framework that dynamically builds environmental maps while following the target and resolves various occlusions by prioritizing high-probability areas for locating the target. For topographic occlusions, a belief-guided search field is constructed and used to evaluate the likelihood of the target's presence, while for dynamic occlusions, a fluid-field approach allows the robot to adaptively follow or overtake moving occluders. Past motion cues and environmental observations refine the search decision over time. Our results demonstrate that the proposed method outperforms existing approaches in terms of search efficiency and success rates, both in simulations and real-world tests. Our target search method enhances the adaptability and reliability of RPF systems in unknown and dynamic environments to support their use in real-world applications. Our code, video, experimental results and appendix are available at this https URL. 

**Abstract (ZH)**: 自主机器人跟踪系统中的路径跟随（RPF）对于个人辅助和安全至关重要，但在动态、未知环境中由于遮挡会失去目标。当前方法依赖预构建的地图并假设静态环境，限制了其在现实环境中的有效性。在地形遮挡（例如，墙壁、角落）和动态遮挡（例如，移动行人）下重新找到目标存在关键空白。本文提出了一种新的启发式引导搜索框架，在跟随目标的同时动态构建环境地图，并通过优先搜索高概率区域来解决各种遮挡问题。对于地形遮挡，构建信念引导的搜索字段来评估目标存在的可能性；对于动态遮挡，流场方法使机器人能够适应性地跟随或超越移动遮挡物。过往运动线索和环境观察随着时间的推移细化搜索决策。我们的实验结果表明，所提出的方法在搜索效率和成功率上优于现有方法，无论是仿真还是实地测试。我们的目标搜索方法增强了RPF系统在未知和动态环境中的适应性和可靠性，支持其实用应用。我们的代码、视频、实验结果和附录可在以下链接获取：this https URL。 

---
# Design and Control of A Tilt-Rotor Tailsitter Aircraft with Pivoting VTOL Capability 

**Title (ZH)**: 具有pivot VTOL能力的傾轉旋翼垂直起降飞机的设计与控制 

**Authors**: Ziqing Ma, Ewoud J.J. Smeur, Guido C.H.E. de Croon  

**Link**: [PDF](https://arxiv.org/pdf/2503.02158)  

**Abstract**: Tailsitter aircraft attract considerable interest due to their capabilities of both agile hover and high speed forward flight. However, traditional tailsitters that use aerodynamic control surfaces face the challenge of limited control effectiveness and associated actuator saturation during vertical flight and transitions. Conversely, tailsitters relying solely on tilting rotors have the drawback of insufficient roll control authority in forward flight. This paper proposes a tilt-rotor tailsitter aircraft with both elevons and tilting rotors as a promising solution. By implementing a cascaded weighted least squares (WLS) based incremental nonlinear dynamic inversion (INDI) controller, the drone successfully achieved autonomous waypoint tracking in outdoor experiments at a cruise airspeed of 16 m/s, including transitions between forward flight and hover without actuator saturation. Wind tunnel experiments confirm improved roll control compared to tilt-rotor-only configurations, while comparative outdoor flight tests highlight the vehicle's superior control over elevon-only designs during critical phases such as vertical descent and transitions. Finally, we also show that the tilt-rotors allow for an autonomous takeoff and landing with a unique pivoting capability that demonstrates stability and robustness under wind disturbances. 

**Abstract (ZH)**: 兼具升降舵和倾斜旋翼的尾座式无人机由于其在悬停和高速前进飞行中的能力而引起广泛关注。然而，传统尾座式无人机采用气动控制舵面，在垂直飞行和转换过程中面临控制效果有限和相关作动器饱和的挑战。相反，仅依靠倾斜旋翼的尾座式无人机在前进飞行中存在滚转控制权威不足的问题。本文提出了一种兼具升降舵和倾斜旋翼的尾座式无人机作为潜在解决方案。通过实施基于加权最小二乘法的嵌套非线性动态反演控制方法，无人机在外场试验中成功实现了自主航点跟踪，巡航空速为16 m/s，包括前进飞行与悬停之间的转换且未出现作动器饱和。风洞试验验证了与仅依靠倾斜旋翼配置相比改进了滚转控制，而在关键阶段如垂直下降和转换过程中，与仅依靠升降舵的设计相比，该无人机显示出了更优越的控制性能。最后，我们还展示了倾斜旋翼的倾斜能力允许该无人机自主起飞和降落，证明了在风干扰下的稳定性和鲁棒性。 

---
# NavG: Risk-Aware Navigation in Crowded Environments Based on Reinforcement Learning with Guidance Points 

**Title (ZH)**: NavG：基于强化学习和引导点的风险感知导航在拥挤环境中的应用 

**Authors**: Qianyi Zhang, Wentao Luo, Boyi Liu, Ziyang Zhang, Yaoyuan Wang, Jingtai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02111)  

**Abstract**: Motion planning in navigation systems is highly susceptible to upstream perceptual errors, particularly in human detection and tracking. To mitigate this issue, the concept of guidance points--a novel directional cue within a reinforcement learning-based framework--is introduced. A structured method for identifying guidance points is developed, consisting of obstacle boundary extraction, potential guidance point detection, and redundancy elimination. To integrate guidance points into the navigation pipeline, a perception-to-planning mapping strategy is proposed, unifying guidance points with other perceptual inputs and enabling the RL agent to effectively leverage the complementary relationships among raw laser data, human detection and tracking, and guidance points. Qualitative and quantitative simulations demonstrate that the proposed approach achieves the highest success rate and near-optimal travel times, greatly improving both safety and efficiency. Furthermore, real-world experiments in dynamic corridors and lobbies validate the robot's ability to confidently navigate around obstacles and robustly avoid pedestrians. 

**Abstract (ZH)**: 基于强化学习的导航系统中的运动规划易受上游感知错误的影响，特别是在人类检测和跟踪方面。为缓解这一问题，提出了指导点——一种新的方向性提示——这一概念并融入到基于强化学习的框架中。开发了一种结构化的指导点识别方法，包括障碍边界提取、潜在指导点检测和冗余消除。为了将指导点整合到导航管道中，提出了一种感知到规划的映射策略，将指导点与其他感知输入统一起来，使RL代理能够有效利用原始激光数据、人类检测与跟踪以及指导点之间的互补关系。定性和定量的仿真实验表明，所提出的方法实现了最高的成功率和接近最优的旅行时间，大大提高了安全性和效率。此外，在动态走廊和lobby的实际实验中验证了机器人能够自信地绕过障碍物并 robust 地避开行人。 

---
# Balancing Act: Trading Off Doppler Odometry and Map Registration for Efficient Lidar Localization 

**Title (ZH)**: 平衡之道：雷达测距机载测距与地图注册之间的权衡以实现高效激光雷达定位 

**Authors**: Katya M. Papais, Daniil Lisus, David J. Yoon, Andrew Lambert, Keith Y.K. Leung, Timothy D. Barfoot  

**Link**: [PDF](https://arxiv.org/pdf/2503.02107)  

**Abstract**: Most autonomous vehicles rely on accurate and efficient localization, which is achieved by comparing live sensor data to a preexisting map, to navigate their environment. Balancing the accuracy of localization with computational efficiency remains a significant challenge, as high-accuracy methods often come with higher computational costs. In this paper, we present two ways of improving lidar localization efficiency and study their impact on performance. First, we integrate a lightweight Doppler-based odometry method into a topometric localization pipeline and compare its performance against an iterative closest point (ICP)-based method. We highlight the trade-offs between these approaches: the Doppler estimator offers faster, lightweight updates, while ICP provides higher accuracy at the cost of increased computational load. Second, by controlling the frequency of localization updates and leveraging odometry estimates between them, we demonstrate that accurate localization can be maintained while optimizing for computational efficiency using either odometry method. Our experimental results show that localizing every 10 lidar frames strikes a favourable balance, achieving a localization accuracy below 0.05 meters in translation and below 0.1 degrees in orientation while reducing computational effort by over 30% in an ICP-based pipeline. We quantify the trade-off of accuracy to computational effort using over 100 kilometers of real-world driving data in different on-road environments. 

**Abstract (ZH)**: 大多数自动驾驶车辆依靠精确高效的定位来导航环境，这一过程是通过将实时传感器数据与预先存在的地图进行比对实现的。在保持定位精度与计算效率之间的平衡仍然是一个重要挑战，因为高精度的方法往往伴随着更高的计算成本。在本文中，我们提出了两种改进lidar定位效率的方法，并研究了它们对性能的影响。首先，我们将一种轻量级的多普勒基于的里程计方法集成到顶点定位管道中，并将其性能与基于迭代最近点（ICP）的方法进行比较。我们强调了这两种方法之间的权衡：多普勒估计器提供了更快、更轻量级的更新，而ICP则在计算负担增加的前提下提供更高的精度。其次，通过控制定位更新的频率并在它们之间利用里程计估计，我们证明了可以在优化计算效率的同时使用任何一种里程计方法保持精确的定位。我们的实验结果显示，每隔10个lidar帧进行一次定位，可以在保持低于0.05米的平移精度和低于0.1度的方向精度的同时，使基于ICP的管道计算成本减少超过30%。我们使用超过100公里的实物驾驶数据在不同的道路环境中量化了精度和计算成本之间的权衡。 

---
# OVAMOS: A Framework for Open-Vocabulary Multi-Object Search in Unknown Environments 

**Title (ZH)**: OVAMOS：一种面向未知环境的多对象开放词汇搜索框架 

**Authors**: Qianwei Wang, Yifan Xu, Vineet Kamat, Carol Menassa  

**Link**: [PDF](https://arxiv.org/pdf/2503.02106)  

**Abstract**: Object search is a fundamental task for robots deployed in indoor building environments, yet challenges arise due to observation instability, especially for open-vocabulary models. While foundation models (LLMs/VLMs) enable reasoning about object locations even without direct visibility, the ability to recover from failures and replan remains crucial. The Multi-Object Search (MOS) problem further increases complexity, requiring the tracking multiple objects and thorough exploration in novel environments, making observation uncertainty a significant obstacle. To address these challenges, we propose a framework integrating VLM-based reasoning, frontier-based exploration, and a Partially Observable Markov Decision Process (POMDP) framework to solve the MOS problem in novel environments. VLM enhances search efficiency by inferring object-environment relationships, frontier-based exploration guides navigation in unknown spaces, and POMDP models observation uncertainty, allowing recovery from failures in occlusion and cluttered environments. We evaluate our framework on 120 simulated scenarios across several Habitat-Matterport3D (HM3D) scenes and a real-world robot experiment in a 50-square-meter office, demonstrating significant improvements in both efficiency and success rate over baseline methods. 

**Abstract (ZH)**: 基于视觉语言模型的多对象搜索框架：结合前沿探索与部分可观测马尔可夫决策过程 

---
# Uncertainty Representation in a SOTIF-Related Use Case with Dempster-Shafer Theory for LiDAR Sensor-Based Object Detection 

**Title (ZH)**: 基于 Dempster-Shafer 理论的 LiDAR 传感器目标检测中不确定性的表示在SOTIF相关应用场景中 

**Authors**: Milin Patel, Rolf Jung  

**Link**: [PDF](https://arxiv.org/pdf/2503.02087)  

**Abstract**: Uncertainty in LiDAR sensor-based object detection arises from environmental variability and sensor performance limitations. Representing these uncertainties is essential for ensuring the Safety of the Intended Functionality (SOTIF), which focuses on preventing hazards in automated driving scenarios. This paper presents a systematic approach to identifying, classifying, and representing uncertainties in LiDAR-based object detection within a SOTIF-related scenario. Dempster-Shafer Theory (DST) is employed to construct a Frame of Discernment (FoD) to represent detection outcomes. Conditional Basic Probability Assignments (BPAs) are applied based on dependencies among identified uncertainty sources. Yager's Rule of Combination is used to resolve conflicting evidence from multiple sources, providing a structured framework to evaluate uncertainties' effects on detection accuracy. The study applies variance-based sensitivity analysis (VBSA) to quantify and prioritize uncertainties, detailing their specific impact on detection performance. 

**Abstract (ZH)**: LiDAR传感器基于的目标检测中的不确定性源自环境变化和传感器性能限制。在确保功能安全（SOTIF）的场景中，这些不确定性的表示至关重要，SOTIF关注于防止自动驾驶场景中的危险。本文提出了一种系统化的方法，以在与SOTIF相关的情景中识别、分类和表示LiDAR基于的目标检测中的不确定性。文中采用Dempster-Shafer理论（DST）构建区分框架（FoD）来表示检测结果。基于已识别的不确定性来源之间的依赖性，应用条件基本概率分配（BPAs）。使用Yager的证据合成规则来解决来自多个来源的冲突证据，提供了一个结构化的框架，以评估不确定性对检测精度的影响。研究应用方差为基础的敏感性分析（VBSA）来量化和优先排序不确定性，并详细说明了它们对检测性能的具体影响。 

---
# CorrA: Leveraging Large Language Models for Dynamic Obstacle Avoidance of Autonomous Vehicles 

**Title (ZH)**: CorrA：利用大型语言模型实现自主车辆动态障碍物避免 

**Authors**: Shanting Wang, Panagiotis Typaldos, Andreas A. Malikopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2503.02076)  

**Abstract**: In this paper, we present Corridor-Agent (CorrA), a framework that integrates large language models (LLMs) with model predictive control (MPC) to address the challenges of dynamic obstacle avoidance in autonomous vehicles. Our approach leverages LLM reasoning ability to generate appropriate parameters for sigmoid-based boundary functions that define safe corridors around obstacles, effectively reducing the state-space of the controlled vehicle. The proposed framework adjusts these boundaries dynamically based on real-time vehicle data that guarantees collision-free trajectories while also ensuring both computational efficiency and trajectory optimality. The problem is formulated as an optimal control problem and solved with differential dynamic programming (DDP) for constrained optimization, and the proposed approach is embedded within an MPC framework. Extensive simulation and real-world experiments demonstrate that the proposed framework achieves superior performance in maintaining safety and efficiency in complex, dynamic environments compared to a baseline MPC approach. 

**Abstract (ZH)**: 本文提出了Corridor-Agent (CorrA)框架，该框架将大型语言模型（LLMs）与模型预测控制（MPC）相结合，以解决自主车辆在动态避障中的挑战。该方法利用LLM的推理能力生成基于Sigmoid边界函数的适当参数，定义障碍物周围的安全走廊，有效减少了受控车辆的状态空间。所提出的框架根据实时车辆数据动态调整这些边界，保证无碰撞轨迹的同时，也确保计算效率和轨迹优化。该问题被形式化为最优控制问题，并使用差分动态规划（DDP）进行约束优化求解，所提出的方法嵌入到MPC框架中。大量仿真实验和实地实验表明，与基线的MPC方法相比，所提出的框架在复杂动态环境中的安全性和效率方面表现更为优异。 

---
# Active Alignments of Lens Systems with Reinforcement Learning 

**Title (ZH)**: 基于强化学习的镜系主动对齐方法 

**Authors**: Matthias Burkhardt, Tobias Schmähling, Michael Layh, Tobias Windisch  

**Link**: [PDF](https://arxiv.org/pdf/2503.02075)  

**Abstract**: Aligning a lens system relative to an imager is a critical challenge in camera manufacturing. While optimal alignment can be mathematically computed under ideal conditions, real-world deviations caused by manufacturing tolerances often render this approach impractical. Measuring these tolerances can be costly or even infeasible, and neglecting them may result in suboptimal alignments. We propose a reinforcement learning (RL) approach that learns exclusively in the pixel space of the sensor output, eliminating the need to develop expert-designed alignment concepts. We conduct an extensive benchmark study and show that our approach surpasses other methods in speed, precision, and robustness. We further introduce relign, a realistic, freely explorable, open-source simulation utilizing physically based rendering that models optical systems with non-deterministic manufacturing tolerances and noise in robotic alignment movement. It provides an interface to popular machine learning frameworks, enabling seamless experimentation and development. Our work highlights the potential of RL in a manufacturing environment to enhance efficiency of optical alignments while minimizing the need for manual intervention. 

**Abstract (ZH)**: 基于像素空间的学习：一种强化学习方法在相机制造中的光学对准应用 

---
# Constraint-Based Modeling of Dynamic Entities in 3D Scene Graphs for Robust SLAM 

**Title (ZH)**: 基于约束的3D场景图中动态实体建模及其在鲁棒SLAM中的应用 

**Authors**: Marco Giberna, Muhammad Shaheer, Hriday Bavle, Jose Andres Millan-Romera, Jose Luis Sanchez-Lopez, Holger Voos  

**Link**: [PDF](https://arxiv.org/pdf/2503.02050)  

**Abstract**: Autonomous robots depend crucially on their ability to perceive and process information from dynamic, ever-changing environments. Traditional simultaneous localization and mapping (SLAM) approaches struggle to maintain consistent scene representations because of numerous moving objects, often treating dynamic elements as outliers rather than explicitly modeling them in the scene representation. In this paper, we present a novel hierarchical 3D scene graph-based SLAM framework that addresses the challenge of modeling and estimating the pose of dynamic objects and agents. We use fiducial markers to detect dynamic entities and to extract their attributes while improving keyframe selection and implementing new capabilities for dynamic entity mapping. We maintain a hierarchical representation where dynamic objects are registered in the SLAM graph and are constrained with robot keyframes and the floor level of the building with our novel entity-keyframe constraints and intra-entity constraints. By combining semantic and geometric constraints between dynamic entities and the environment, our system jointly optimizes the SLAM graph to estimate the pose of the robot and various dynamic agents and objects while maintaining an accurate map. Experimental evaluation demonstrates that our approach achieves a 27.57% reduction in pose estimation error compared to traditional methods and enables higher-level reasoning about scene dynamics. 

**Abstract (ZH)**: 自主机器人依赖于其对动态、不断变化环境中的信息进行感知和处理的能力。传统的同步定位与 Mapping（SLAM）方法难以维持一致的场景表示，因为存在大量移动物体，通常将动态元素视为离群值，而不是在场景表示中显式建模。在本文中，我们提出了一种新颖的分层 3D 场景图基于的 SLAM 框架，以解决动态对象和代理建模和姿态估计的挑战。我们使用标记物检测动态实体并提取其属性，同时改进关键帧选择并实施动态实体映射的新能力。我们维护一个分层表示，其中动态对象被注册在 SLAM 图中，并通过我们新颖的实体-关键帧约束和内部实体约束与机器人关键帧及建筑楼层进行约束。通过结合动态实体与其环境之间的语义和几何约束，我们的系统联合优化 SLAM 图以估计机器人的姿态及各种动态代理和物体的姿态，同时保持准确的映射。实验评估表明，与传统方法相比，我们的方法在姿态估计误差上降低了 27.57%，并能够实现关于场景动态的高级推理。 

---
# FRMD: Fast Robot Motion Diffusion with Consistency-Distilled Movement Primitives for Smooth Action Generation 

**Title (ZH)**: FRMD：快速机器人运动扩散与一致性提炼运动基元的平滑动作生成 

**Authors**: Xirui Shi, Jun Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.02048)  

**Abstract**: We consider the problem of using diffusion models to generate fast, smooth, and temporally consistent robot motions. Although diffusion models have demonstrated superior performance in robot learning due to their task scalability and multi-modal flexibility, they suffer from two fundamental limitations: (1) they often produce non-smooth, jerky motions due to their inability to capture temporally consistent movement dynamics, and (2) their iterative sampling process incurs prohibitive latency for many robotic tasks. Inspired by classic robot motion generation methods such as DMPs and ProMPs, which capture temporally and spatially consistent dynamic of trajectories using low-dimensional vectors -- and by recent advances in diffusion-based image generation that use consistency models with probability flow ODEs to accelerate the denoising process, we propose Fast Robot Motion Diffusion (FRMD). FRMD uniquely integrates Movement Primitives (MPs) with Consistency Models to enable efficient, single-step trajectory generation. By leveraging probabilistic flow ODEs and consistency distillation, our method models trajectory distributions while learning a compact, time-continuous motion representation within an encoder-decoder architecture. This unified approach eliminates the slow, multi-step denoising process of conventional diffusion models, enabling efficient one-step inference and smooth robot motion generation. We extensively evaluated our FRMD on the well-recognized Meta-World and ManiSkills Benchmarks, ranging from simple to more complex manipulation tasks, comparing its performance against state-of-the-art baselines. Our results show that FRMD generates significantly faster, smoother trajectories while achieving higher success rates. 

**Abstract (ZH)**: Fast Robot Motion Diffusion：集成运动原语与一致性模型的高效轨迹生成 

---
# Class-Aware PillarMix: Can Mixed Sample Data Augmentation Enhance 3D Object Detection with Radar Point Clouds? 

**Title (ZH)**: 面向类别的PillarMix：混合样本数据增强能否提升基于雷达点云的3D目标检测？ 

**Authors**: Miao Zhang, Sherif Abdulatif, Benedikt Loesch, Marco Altmann, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02687)  

**Abstract**: Due to the significant effort required for data collection and annotation in 3D perception tasks, mixed sample data augmentation (MSDA) has been widely studied to generate diverse training samples by mixing existing data. Recently, many MSDA techniques have been developed for point clouds, but they mainly target LiDAR data, leaving their application to radar point clouds largely unexplored. In this paper, we examine the feasibility of applying existing MSDA methods to radar point clouds and identify several challenges in adapting these techniques. These obstacles stem from the radar's irregular angular distribution, deviations from a single-sensor polar layout in multi-radar setups, and point sparsity. To address these issues, we propose Class-Aware PillarMix (CAPMix), a novel MSDA approach that applies MixUp at the pillar level in 3D point clouds, guided by class labels. Unlike methods that rely a single mix ratio to the entire sample, CAPMix assigns an independent ratio to each pillar, boosting sample diversity. To account for the density of different classes, we use class-specific distributions: for dense objects (e.g., large vehicles), we skew ratios to favor points from another sample, while for sparse objects (e.g., pedestrians), we sample more points from the original. This class-aware mixing retains critical details and enriches each sample with new information, ultimately generating more diverse training data. Experimental results demonstrate that our method not only significantly boosts performance but also outperforms existing MSDA approaches across two datasets (Bosch Street and K-Radar). We believe that this straightforward yet effective approach will spark further investigation into MSDA techniques for radar data. 

**Abstract (ZH)**: 基于类别的支柱混合法（CAPMix）：雷达点云的混合样本数据扩增方法 

---
# Velocity-free task-space regulator for robot manipulators with external disturbances 

**Title (ZH)**: 基于外部干扰的机器人 manipulators 无速度任务空间调节器 

**Authors**: Haiwen Wu, Bayu Jayawardhana, Dabo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02634)  

**Abstract**: This paper addresses the problem of task-space robust regulation of robot manipulators subject to external disturbances. A velocity-free control law is proposed by combining the internal model principle and the passivity-based output-feedback control approach. The developed output-feedback controller ensures not only asymptotic convergence of the regulation error but also suppression of unwanted external step/sinusoidal disturbances. The potential of the proposed method lies in its simplicity, intuitively appealing, and simple gain selection criteria for synthesis of multi-joint robot manipulator control systems. 

**Abstract (ZH)**: 本文解决了机器人 manipulator 在外部干扰下的任务空间鲁棒调节问题。通过结合内部模型原则和基于耗散性输出反馈控制方法，提出了一种无速度控制律。所发展的输出反馈控制器不仅确保调节误差的渐近收敛，而且还抑制了不必要的外部阶跃/正弦干扰。所提出方法的潜力在于其简洁性、直观性和多关节机器人控制系统合成中的简单增益选择准则。 

---
# Resource-Efficient Affordance Grounding with Complementary Depth and Semantic Prompts 

**Title (ZH)**: 资源高效的功能 grounding 方法：互补的深度信息和语义提示 

**Authors**: Yizhou Huang, Fan Yang, Guoliang Zhu, Gen Li, Hao Shi, Yukun Zuo, Wenrui Chen, Zhiyong Li, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02600)  

**Abstract**: Affordance refers to the functional properties that an agent perceives and utilizes from its environment, and is key perceptual information required for robots to perform actions. This information is rich and multimodal in nature. Existing multimodal affordance methods face limitations in extracting useful information, mainly due to simple structural designs, basic fusion methods, and large model parameters, making it difficult to meet the performance requirements for practical deployment. To address these issues, this paper proposes the BiT-Align image-depth-text affordance mapping framework. The framework includes a Bypass Prompt Module (BPM) and a Text Feature Guidance (TFG) attention selection mechanism. BPM integrates the auxiliary modality depth image directly as a prompt to the primary modality RGB image, embedding it into the primary modality encoder without introducing additional encoders. This reduces the model's parameter count and effectively improves functional region localization accuracy. The TFG mechanism guides the selection and enhancement of attention heads in the image encoder using textual features, improving the understanding of affordance characteristics. Experimental results demonstrate that the proposed method achieves significant performance improvements on public AGD20K and HICO-IIF datasets. On the AGD20K dataset, compared with the current state-of-the-art method, we achieve a 6.0% improvement in the KLD metric, while reducing model parameters by 88.8%, demonstrating practical application values. The source code will be made publicly available at this https URL. 

**Abstract (ZH)**: affordance映射框架：BiT-Align图像-深度-文本对齐 

---
# Unveiling the Potential of Segment Anything Model 2 for RGB-Thermal Semantic Segmentation with Language Guidance 

**Title (ZH)**: 揭示段 anything 模型2在光照语义分割中的潜力及语言指导作用 

**Authors**: Jiayi Zhao, Fei Teng, Kai Luo, Guoqiang Zhao, Zhiyong Li, Xu Zheng, Kailun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02581)  

**Abstract**: The perception capability of robotic systems relies on the richness of the dataset. Although Segment Anything Model 2 (SAM2), trained on large datasets, demonstrates strong perception potential in perception tasks, its inherent training paradigm prevents it from being suitable for RGB-T tasks. To address these challenges, we propose SHIFNet, a novel SAM2-driven Hybrid Interaction Paradigm that unlocks the potential of SAM2 with linguistic guidance for efficient RGB-Thermal perception. Our framework consists of two key components: (1) Semantic-Aware Cross-modal Fusion (SACF) module that dynamically balances modality contributions through text-guided affinity learning, overcoming SAM2's inherent RGB bias; (2) Heterogeneous Prompting Decoder (HPD) that enhances global semantic information through a semantic enhancement module and then combined with category embeddings to amplify cross-modal semantic consistency. With 32.27M trainable parameters, SHIFNet achieves state-of-the-art segmentation performance on public benchmarks, reaching 89.8% on PST900 and 67.8% on FMB, respectively. The framework facilitates the adaptation of pre-trained large models to RGB-T segmentation tasks, effectively mitigating the high costs associated with data collection while endowing robotic systems with comprehensive perception capabilities. The source code will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于SAM2的新型混合交互框架SHIFNet及其在RGB-T感知任务中的应用 

---
# TS-CGNet: Temporal-Spatial Fusion Meets Centerline-Guided Diffusion for BEV Mapping 

**Title (ZH)**: TS-CGNet：时空融合结合中心线引导扩散的BEV映射 

**Authors**: Xinying Hong, Siyu Li, Kang Zeng, Hao Shi, Bomin Peng, Kailun Yang, Zhiyong Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.02578)  

**Abstract**: Bird's Eye View (BEV) perception technology is crucial for autonomous driving, as it generates top-down 2D maps for environment perception, navigation, and decision-making. Nevertheless, the majority of current BEV map generation studies focusing on visual map generation lack depth-aware reasoning capabilities. They exhibit limited efficacy in managing occlusions and handling complex environments, with a notable decline in perceptual performance under adverse weather conditions or low-light scenarios. Therefore, this paper proposes TS-CGNet, which leverages Temporal-Spatial fusion with Centerline-Guided diffusion. This visual framework, grounded in prior knowledge, is designed for integration into any existing network for building BEV maps. Specifically, this framework is decoupled into three parts: Local mapping system involves the initial generation of semantic maps using purely visual information; The Temporal-Spatial Aligner Module (TSAM) integrates historical information into mapping generation by applying transformation matrices; The Centerline-Guided Diffusion Model (CGDM) is a prediction module based on the diffusion model. CGDM incorporates centerline information through spatial-attention mechanisms to enhance semantic segmentation reconstruction. We construct BEV semantic segmentation maps by our methods on the public nuScenes and the robustness benchmarks under various corruptions. Our method improves 1.90%, 1.73%, and 2.87% for perceived ranges of 60x30m, 120x60m, and 240x60m in the task of BEV HD mapping. TS-CGNet attains an improvement of 1.92% for perceived ranges of 100x100m in the task of BEV semantic mapping. Moreover, TS-CGNet achieves an average improvement of 2.92% in detection accuracy under varying weather conditions and sensor interferences in the perception range of 240x60m. The source code will be publicly available at this https URL. 

**Abstract (ZH)**: BEV感知技术对自主驾驶至关重要：TS-CGNet-temporal-spatial融合与中心线引导扩散方法在BEV地图构建中的应用 

---
# Label-Efficient LiDAR Panoptic Segmentation 

**Title (ZH)**: 标签高效激光雷达全景分割 

**Authors**: Ahmet Selim Çanakçı, Niclas Vödisch, Kürsat Petek, Wolfram Burgard, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2503.02372)  

**Abstract**: A main bottleneck of learning-based robotic scene understanding methods is the heavy reliance on extensive annotated training data, which often limits their generalization ability. In LiDAR panoptic segmentation, this challenge becomes even more pronounced due to the need to simultaneously address both semantic and instance segmentation from complex, high-dimensional point cloud data. In this work, we address the challenge of LiDAR panoptic segmentation with very few labeled samples by leveraging recent advances in label-efficient vision panoptic segmentation. To this end, we propose a novel method, Limited-Label LiDAR Panoptic Segmentation (L3PS), which requires only a minimal amount of labeled data. Our approach first utilizes a label-efficient 2D network to generate panoptic pseudo-labels from a small set of annotated images, which are subsequently projected onto point clouds. We then introduce a novel 3D refinement module that capitalizes on the geometric properties of point clouds. By incorporating clustering techniques, sequential scan accumulation, and ground point separation, this module significantly enhances the accuracy of the pseudo-labels, improving segmentation quality by up to +10.6 PQ and +7.9 mIoU. We demonstrate that these refined pseudo-labels can be used to effectively train off-the-shelf LiDAR segmentation networks. Through extensive experiments, we show that L3PS not only outperforms existing methods but also substantially reduces the annotation burden. We release the code of our work at this https URL. 

**Abstract (ZH)**: 基于学习的机器人场景理解方法的主要瓶颈是对大量标注训练数据的高依赖性，这往往限制了其泛化能力。在激光雷达全景分割中，这一挑战更为突出，因为需要同时从复杂的高维度点云数据中解决语义和实例分割问题。在本文中，我们通过利用最近在标签高效视觉全景分割方面的进展，解决了在少量标注样本情况下激光雷达全景分割的挑战。为此，我们提出了一种新型方法——有限标签激光雷达全景分割（L3PS），该方法仅需少量标注数据。我们的方法首先利用一种标签高效的二维网络从少量标注图像中生成全景伪标签，并将其投影到点云上。然后，我们引入了一种新颖的3D细化模块，该模块充分利用了点云的几何特性。通过结合聚类技术、顺序扫描积累和地面点分离，该模块显著提高了伪标签的准确性，分割质量最高可提高10.6 PQ和7.9 mIoU。我们展示了这些细化后的伪标签可以有效训练现成的激光雷达分割网络。通过大量实验，我们证明L3PS不仅优于现有方法，还能显著减少标注负担。我们已在此网址发布了我们的代码：this https URL。 

---
# Target Return Optimizer for Multi-Game Decision Transformer 

**Title (ZH)**: 多游戏决策变换器的目标回报优化器 

**Authors**: Kensuke Tatematsu, Akifumi Wachi  

**Link**: [PDF](https://arxiv.org/pdf/2503.02311)  

**Abstract**: Achieving autonomous agents with robust generalization capabilities across diverse games and tasks remains one of the ultimate goals in AI research. Recent advancements in transformer-based offline reinforcement learning, exemplified by the MultiGame Decision Transformer [Lee et al., 2022], have shown remarkable performance across various games or tasks. However, these approaches depend heavily on human expertise, presenting substantial challenges for practical deployment, particularly in scenarios with limited prior game-specific knowledge. In this paper, we propose an algorithm called Multi-Game Target Return Optimizer (MTRO) to autonomously determine game-specific target returns within the Multi-Game Decision Transformer framework using solely offline datasets. MTRO addresses the existing limitations by automating the target return configuration process, leveraging environmental reward information extracted from offline datasets. Notably, MTRO does not require additional training, enabling seamless integration into existing Multi-Game Decision Transformer architectures. Our experimental evaluations on Atari games demonstrate that MTRO enhances the performance of RL policies across a wide array of games, underscoring its potential to advance the field of autonomous agent development. 

**Abstract (ZH)**: 实现跨多种游戏和任务具有 robust 通用化能力的自主智能体仍然是 AI 研究中的最终目标之一。基于.transformer 的 Offline 强化学习的 Recent 进展，例如 MultiGame Decision Transformer [Lee et al., 2022]，展示了在各种游戏或任务中的出色表现。然而，这些方法高度依赖于人工专业知识，特别是在缺乏特定游戏先验知识的场景中带来了巨大的实际部署挑战。本文提出了一种名为 Multi-Game Target Return Optimizer (MTRO) 的算法，该算法在 Multi-Game Decision Transformer 框架中仅使用离线数据集自主确定游戏特定的目标回报。MTRO 通过利用从离线数据集中提取的环境奖励信息自动配置目标回报，解决了现有方法的局限性。值得注意的是，MTRO 不需要额外的训练，可以无缝集成到现有的 Multi-Game Decision Transformer 架构中。我们在 Arcade Learning Environment 游戏上的实验评估表明，MTRO 能够提高强化学习策略在多种游戏中的性能，突显了其在自主智能体开发领域的潜在价值。 

---
# WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation 

**Title (ZH)**: WMNav: 将视觉语言模型集成到世界模型中进行物体目标导航 

**Authors**: Dujun Nie, Xianda Guo, Yiqun Duan, Ruijun Zhang, Long Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.02247)  

**Abstract**: Object Goal Navigation-requiring an agent to locate a specific object in an unseen environment-remains a core challenge in embodied AI. Although recent progress in Vision-Language Model (VLM)-based agents has demonstrated promising perception and decision-making abilities through prompting, none has yet established a fully modular world model design that reduces risky and costly interactions with the environment by predicting the future state of the world. We introduce WMNav, a novel World Model-based Navigation framework powered by Vision-Language Models (VLMs). It predicts possible outcomes of decisions and builds memories to provide feedback to the policy module. To retain the predicted state of the environment, WMNav proposes the online maintained Curiosity Value Map as part of the world model memory to provide dynamic configuration for navigation policy. By decomposing according to a human-like thinking process, WMNav effectively alleviates the impact of model hallucination by making decisions based on the feedback difference between the world model plan and observation. To further boost efficiency, we implement a two-stage action proposer strategy: broad exploration followed by precise localization. Extensive evaluation on HM3D and MP3D validates WMNav surpasses existing zero-shot benchmarks in both success rate and exploration efficiency (absolute improvement: +3.2% SR and +3.2% SPL on HM3D, +13.5% SR and +1.1% SPL on MP3D). Project page: this https URL. 

**Abstract (ZH)**: 基于世界模型的物体目标导航：一种利用视觉语言模型的动力学导航框架 

---
# Four Principles for Physically Interpretable World Models 

**Title (ZH)**: 物理可解释的世界模型的四个原则 

**Authors**: Jordan Peper, Zhenjiang Mao, Yuang Geng, Siyuan Pan, Ivan Ruchkin  

**Link**: [PDF](https://arxiv.org/pdf/2503.02143)  

**Abstract**: As autonomous systems are increasingly deployed in open and uncertain settings, there is a growing need for trustworthy world models that can reliably predict future high-dimensional observations. The learned latent representations in world models lack direct mapping to meaningful physical quantities and dynamics, limiting their utility and interpretability in downstream planning, control, and safety verification. In this paper, we argue for a fundamental shift from physically informed to physically interpretable world models - and crystallize four principles that leverage symbolic knowledge to achieve these ends: (1) structuring latent spaces according to the physical intent of variables, (2) learning aligned invariant and equivariant representations of the physical world, (3) adapting training to the varied granularity of supervision signals, and (4) partitioning generative outputs to support scalability and verifiability. We experimentally demonstrate the value of each principle on two benchmarks. This paper opens several intriguing research directions to achieve and capitalize on full physical interpretability in world models. 

**Abstract (ZH)**: 随着自主系统在开放和不确定环境中越来越多地被部署，可靠预测高维未来观测的可信赖世界模型的需求不断增加。现有世界模型中的学习潜在表示缺乏与有意义的物理量和动力学的直接映射，限制了其在下游规划、控制和安全性验证中的实用性和可解释性。本文主张从基于物理原理转变为基于物理解释的世界模型——并提炼出四条原则，利用符号知识实现这些目标：（1）根据变量的物理意图结构化潜在空间，（2）学习与物理世界对齐的不变和协变表示，（3）调整训练以适应监督信号的多样化粒度，（4）拆分生成输出以支持可扩展性和可验证性。我们在两个基准测试上实验证明了每条原则的价值。本文开启了几个有趣的研究方向，以实现和利用世界模型中的全面物理解释。 

---
# Data Augmentation for NeRFs in the Low Data Limit 

**Title (ZH)**: 低数据限制下用于NeRF的数据增强方法 

**Authors**: Ayush Gaggar, Todd D. Murphey  

**Link**: [PDF](https://arxiv.org/pdf/2503.02092)  

**Abstract**: Current methods based on Neural Radiance Fields fail in the low data limit, particularly when training on incomplete scene data. Prior works augment training data only in next-best-view applications, which lead to hallucinations and model collapse with sparse data. In contrast, we propose adding a set of views during training by rejection sampling from a posterior uncertainty distribution, generated by combining a volumetric uncertainty estimator with spatial coverage. We validate our results on partially observed scenes; on average, our method performs 39.9% better with 87.5% less variability across established scene reconstruction benchmarks, as compared to state of the art baselines. We further demonstrate that augmenting the training set by sampling from any distribution leads to better, more consistent scene reconstruction in sparse environments. This work is foundational for robotic tasks where augmenting a dataset with informative data is critical in resource-constrained, a priori unknown environments. Videos and source code are available at this https URL. 

**Abstract (ZH)**: 基于神经辐射场的方法在数据稀少的情况下失效，特别是在使用不完整场景数据进行训练时。先前工作仅在下一步最佳视图应用中增强训练数据，导致在稀疏数据情况下出现幻觉和模型崩溃。相比之下，我们提议在训练过程中通过从结合体素不确定性估计器与空间覆盖生成的后验不确定性分布中进行拒绝抽样来添加一组视图。我们在部分观测场景上验证了该方法；与现有的基线方法相比，我们的方法在多个场景重建基准测试中平均提高39.9%，且变异性降低87.5%。此外，我们还证明，从任何分布中抽样增强训练集可以在稀疏环境中获得更好的、更一致的场景重建效果。该项工作为在资源受限和先验未知环境中补充具有信息性数据的数据集提供了基础。有关视频和源代码在此处获取。 

---
# Optimizing Robot Programming: Mixed Reality Gripper Control 

**Title (ZH)**: 机器人编程优化：混合现实抓手控制 

**Authors**: Maximilian Rettinger, Leander Hacker, Philipp Wolters, Gerhard Rigoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.02042)  

**Abstract**: Conventional robot programming methods are complex and time-consuming for users. In recent years, alternative approaches such as mixed reality have been explored to address these challenges and optimize robot programming. While the findings of the mixed reality robot programming methods are convincing, most existing methods rely on gesture interaction for robot programming. Since controller-based interactions have proven to be more reliable, this paper examines three controller-based programming methods within a mixed reality scenario: 1) Classical Jogging, where the user positions the robot's end effector using the controller's thumbsticks, 2) Direct Control, where the controller's position and orientation directly corresponds to the end effector's, and 3) Gripper Control, where the controller is enhanced with a 3D-printed gripper attachment to grasp and release objects. A within-subjects study (n = 30) was conducted to compare these methods. The findings indicate that the Gripper Control condition outperforms the others in terms of task completion time, user experience, mental demand, and task performance, while also being the preferred method. Therefore, it demonstrates promising potential as an effective and efficient approach for future robot programming. Video available at this https URL. 

**Abstract (ZH)**: 传统的机器人编程方法对用户来说复杂且耗时。近年来，混合现实等替代方法被积极探索以应对这些挑战并优化机器人编程。尽管混合现实机器人编程方法的研究结果令人信服，但大多数现有方法仍依赖手势交互进行机器人编程。由于基于控制器的交互已被证明更可靠，本文在混合现实场景下研究了三种基于控制器的编程方法：1）经典 Jogging 方法，其中用户使用控制器的拇指摇杆定位机器人的末端执行器；2）直接控制方法，其中控制器的位置和方向直接对应末端执行器的位置和方向；3）夹爪控制方法，其中控制器配备了3D打印的夹爪附件以抓取和释放物体。进行了一个被试内研究（n=30）来比较这些方法。研究发现，夹爪控制条件在任务完成时间、用户体验、心理需求和任务性能方面优于其他方法，并且是用户偏好方法，因此证明了其作为未来机器人编程的有效且高效方法的有希望的潜力。视频链接：此 https URL。 

---
# Pretrained Embeddings as a Behavior Specification Mechanism 

**Title (ZH)**: 预训练嵌入作为行为规范机制 

**Authors**: Parv Kapoor, Abigail Hammer, Ashish Kapoor, Karen Leung, Eunsuk Kang  

**Link**: [PDF](https://arxiv.org/pdf/2503.02012)  

**Abstract**: We propose an approach to formally specifying the behavioral properties of systems that rely on a perception model for interactions with the physical world. The key idea is to introduce embeddings -- mathematical representations of a real-world concept -- as a first-class construct in a specification language, where properties are expressed in terms of distances between a pair of ideal and observed embeddings. To realize this approach, we propose a new type of temporal logic called Embedding Temporal Logic (ETL), and describe how it can be used to express a wider range of properties about AI-enabled systems than previously possible. We demonstrate the applicability of ETL through a preliminary evaluation involving planning tasks in robots that are driven by foundation models; the results are promising, showing that embedding-based specifications can be used to steer a system towards desirable behaviors. 

**Abstract (ZH)**: 我们提出了一种形式化规定依赖于感知模型的系统行为属性的方法。关键想法是在规范语言中引入嵌入式表示——现实世界概念的数学表示，通过表达理想嵌入和观测嵌入之间的距离来描述系统的性质。为了实现这一方法，我们提出了一种新的时序逻辑类型，称为嵌入时序逻辑(ETL)，并描述了如何使用ETL表达关于AI使能系统的更广泛属性。我们通过涉及由基础模型驱动的机器人执行规划任务的初步评估展示了ETL的适用性，结果表明，基于嵌入的规范可以引导系统朝向期望的行为。 

---
# Minimum-Length Coordinated Motions For Two Convex Centrally-Symmetric Robots 

**Title (ZH)**: 两个凸中心对称机器人 的最小协调运动 

**Authors**: David Kirkpatrick, Paul Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.02010)  

**Abstract**: We study the problem of determining coordinated motions, of minimum total length, for two arbitrary convex centrally-symmetric (CCS) robots in an otherwise obstacle-free plane. Using the total path length traced by the two robot centres as a measure of distance, we give an exact characterization of a (not necessarily unique) shortest collision-avoiding motion for all initial and goal configurations of the robots. The individual paths are composed of at most six convex pieces, and their total length can be expressed as a simple integral with a closed form solution depending only on the initial and goal configuration of the robots. The path pieces are either straight segments or segments of the boundary of the Minkowski sum of the two robots (circular arcs, in the special case of disc robots). Furthermore, the paths can be parameterized in such a way that (i) only one robot is moving at any given time (decoupled motion), or (ii) the orientation of the robot configuration changes monotonically. 

**Abstract (ZH)**: 我们研究了在无障碍平面内确定两个任意中心对称凸机器人（CCS）的协调运动问题，使得总长度最小。使用两个机器人中心所追踪的总路径长度作为距离的度量，给出了所有初始和目标配置下机器人避免碰撞的最短运动的精确描述（不一定唯一）。各个路径最多由六段凸部分组成，其总长度可以表示为仅依赖于机器人初始和目标配置的简单闭合形式积分。路径片段要么是直线段，要么是两个机器人Minkowski和的边界部分（在特殊情况下为圆弧段）。进一步地，路径可以参数化为如下方式：（i）任何时候只有单个机器人移动（解耦运动），或（ii）机器人配置的朝向单调变化。 

---
# Uncertainty Comes for Free: Human-in-the-Loop Policies with Diffusion Models 

**Title (ZH)**: 不确定性免费到来：具有扩散模型的人机交互策略 

**Authors**: Zhanpeng He, Yifeng Cao, Matei Ciocarlie  

**Link**: [PDF](https://arxiv.org/pdf/2503.01876)  

**Abstract**: Human-in-the-loop (HitL) robot deployment has gained significant attention in both academia and industry as a semi-autonomous paradigm that enables human operators to intervene and adjust robot behaviors at deployment time, improving success rates. However, continuous human monitoring and intervention can be highly labor-intensive and impractical when deploying a large number of robots. To address this limitation, we propose a method that allows diffusion policies to actively seek human assistance only when necessary, reducing reliance on constant human oversight. To achieve this, we leverage the generative process of diffusion policies to compute an uncertainty-based metric based on which the autonomous agent can decide to request operator assistance at deployment time, without requiring any operator interaction during training. Additionally, we show that the same method can be used for efficient data collection for fine-tuning diffusion policies in order to improve their autonomous performance. Experimental results from simulated and real-world environments demonstrate that our approach enhances policy performance during deployment for a variety of scenarios. 

**Abstract (ZH)**: 循环人类在环中的机器人部署：一种在必要时主动寻求人类协助的方法 

---
# Data Augmentation for Instruction Following Policies via Trajectory Segmentation 

**Title (ZH)**: 基于轨迹分割的指令遵循策略数据增强 

**Authors**: Niklas Höpner, Ilaria Tiddi, Herke van Hoof  

**Link**: [PDF](https://arxiv.org/pdf/2503.01871)  

**Abstract**: The scalability of instructable agents in robotics or gaming is often hindered by limited data that pairs instructions with agent trajectories. However, large datasets of unannotated trajectories containing sequences of various agent behaviour (play trajectories) are often available. In a semi-supervised setup, we explore methods to extract labelled segments from play trajectories. The goal is to augment a small annotated dataset of instruction-trajectory pairs to improve the performance of an instruction-following policy trained downstream via imitation learning. Assuming little variation in segment length, recent video segmentation methods can effectively extract labelled segments. To address the constraint of segment length, we propose Play Segmentation (PS), a probabilistic model that finds maximum likely segmentations of extended subsegments, while only being trained on individual instruction segments. Our results in a game environment and a simulated robotic gripper setting underscore the importance of segmentation; randomly sampled segments diminish performance, while incorporating labelled segments from PS improves policy performance to the level of a policy trained on twice the amount of labelled data. 

**Abstract (ZH)**: 可解释代理在机器人学或游戏中的可扩展性常常受限于指令与代理轨迹配对数据的有限性。然而，大量未标注的轨迹数据集通常可用，这些数据集中包含了各种代理行为的序列（游戏轨迹）。在半监督设置中，我们探索从游戏轨迹中提取标注段的方法。目标是通过模仿学习训练下游的指令遵循策略时，扩展少量标注的指令-轨迹配对数据集，以提高性能。假设段长度变化不大，最近的视频分割方法可以有效地提取标注段。为了解决段长度的约束，我们提出了一种概率模型Play Segmentation (PS)，该模型仅通过训练个别指令段即可找到最有可能的段划分。我们在游戏环境和模拟的机器人夹爪设置中的结果显示了分割的重要性；随机采样的段降低了性能，而结合PS生成的标注段则能够提高策略性能，达到使用两倍标注数据训练的策略的水平。 

---
# Tracking Control of Euler-Lagrangian Systems with Prescribed State, Input, and Temporal Constraints 

**Title (ZH)**: 带有预先指定状态、输入和时间约束的Euler-Lagrangian系统跟踪控制 

**Authors**: Chidre Shravista Kashyap, Pushpak Jagtap, Jishnu Keshavan  

**Link**: [PDF](https://arxiv.org/pdf/2503.01866)  

**Abstract**: The synthesis of a smooth tracking control policy for Euler-Lagrangian (EL) systems with stringent regions of operation induced by state, input and temporal (SIT) constraints is a very challenging task. Most existing solutions rely on prior information of the parameters of the nominal EL dynamics together with bounds on system uncertainty, and incorporate either state or input constraints to guarantee tracking error convergence in a prescribed settling time. Contrary to these approaches, this study proposes an approximation-free adaptive barrier function-based control policy for achieving local prescribed-time convergence of the tracking error to a prescribed-bound in the presence of state and input constraints. This is achieved by imposing time-varying bounds on the filtered tracking error to confine the states within their respective bounds, while also incorporating a saturation function to limit the magnitude of the proposed control action that leverages smooth time-based generator functions for ensuring tracking error convergence within the prescribed-time. Importantly, corresponding feasibility conditions pertaining to the minimum control authority, maximum disturbance rejection capability of the control policy, and the viable set of initial conditions are derived, illuminating the narrow operating domain of the EL systems arising from the interplay of SIT constraints. Numerical validation studies with three different robotic manipulators are employed to demonstrate the efficacy of the proposed scheme. A detailed performance comparison study with leading alternative designs is also undertaken to illustrate the superior performance of the proposed scheme. 

**Abstract (ZH)**: 基于状态和输入时变约束的埃尔朗gen-拉格朗日系统平滑跟踪控制策略的研究 

---
# A strictly predefined-time convergent and anti-noise fractional-order zeroing neural network for solving time-variant quadratic programming in kinematic robot control 

**Title (ZH)**: 严格预定义时间收敛且抗噪声的分数阶零点神经网络在Kinematic机器人控制中求解时变二次规划 

**Authors**: Yi Yang, Xiao Li, Xuchen Wang, Mei Liu, Junwei Yin, Weibing Li, Richard M. Voyles, Xin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.01857)  

**Abstract**: This paper proposes a strictly predefined-time convergent and anti-noise fractional-order zeroing neural network (SPTC-AN-FOZNN) model, meticulously designed for addressing time-variant quadratic programming (TVQP) problems. This model marks the first variable-gain ZNN to collectively manifest strictly predefined-time convergence and noise resilience, specifically tailored for kinematic motion control of robots. The SPTC-AN-FOZNN advances traditional ZNNs by incorporating a conformable fractional derivative in accordance with the Leibniz rule, a compliance not commonly achieved by other fractional derivative definitions. It also features a novel activation function designed to ensure favorable convergence independent of the model's order. When compared to five recently published recurrent neural networks (RNNs), the SPTC-AN-FOZNN, configured with $0<\alpha\leq 1$, exhibits superior positional accuracy and robustness against additive noises for TVQP applications. Extensive empirical evaluations, including simulations with two types of robotic manipulators and experiments with a Flexiv Rizon robot, have validated the SPTC-AN-FOZNN's effectiveness in precise tracking and computational efficiency, establishing its utility for robust kinematic control. 

**Abstract (ZH)**: 严格预定义时间收敛和抗噪声分数阶零阶神经网络模型：针对时变二次规划问题的严格预定义时间收敛和抗噪声可变增益零阶神经网络模型 

---
# Interaction-Aware Model Predictive Decision-Making for Socially-Compliant Autonomous Driving in Mixed Urban Traffic Scenarios 

**Title (ZH)**: 面向社会合规的混合城市交通场景中交互aware模型预测决策控制 

**Authors**: Balint Varga, Thomas Brand, Marcus Schmitz, Ehsan Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2503.01852)  

**Abstract**: This paper presents the experimental validation of an interaction-aware model predictive decision-making (IAMPDM) approach in the course of a simulator study. The proposed IAMPDM uses a model of the pedestrian, which simultaneously predicts their future trajectories and characterizes the interaction between the pedestrian and the automated vehicle. The main benefit of the proposed concept and the experiment is that the interaction between the pedestrian and the socially compliant autonomous vehicle leads to smoother traffic. Furthermore, the experiment features a novel human-in-the-decision-loop aspect, meaning that the test subjects have no expected behavior or defined sequence of their actions, better imitating real traffic scenarios. Results show that intention-aware decision-making algorithms are more effective in realistic conditions and contribute to smoother traffic flow than state-of-the-art solutions. Furthermore, the findings emphasize the crucial impact of intention-aware decision-making on autonomous vehicle performance in urban areas and the need for further research. 

**Abstract (ZH)**: 本文提出了一种交互感知模型预测决策（IAMPDM）方法在模拟器研究中的实验验证。所提出的IAMPDM使用了一个行人的模型，该模型能够同时预测行人的未来轨迹并表征行人与自动驾驶车辆之间的交互。该研究的主要益处在于，行人与社会适应性的自动驾驶车辆的交互导致了更顺畅的交通流动。此外，实验还包含了一个新颖的人在决策环中的方面，即测试对象没有预设的行为或定义好的行动顺序，更好地模拟了实际的交通场景。结果表明，意图感知的决策算法在现实条件下比现有的解决方案更为有效，并有助于更顺畅的交通流动。此外，研究结果强调了意图感知决策对城市区域自动驾驶车辆性能的至关重要影响，以及需要进一步研究。 

---
