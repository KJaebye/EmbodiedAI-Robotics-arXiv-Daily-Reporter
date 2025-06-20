# Object-centric 3D Motion Field for Robot Learning from Human Videos 

**Title (ZH)**: 基于对象的3D运动场用于机器人从人类视频学习 

**Authors**: Zhao-Heng Yin, Sherry Yang, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2506.04227)  

**Abstract**: Learning robot control policies from human videos is a promising direction for scaling up robot learning. However, how to extract action knowledge (or action representations) from videos for policy learning remains a key challenge. Existing action representations such as video frames, pixelflow, and pointcloud flow have inherent limitations such as modeling complexity or loss of information. In this paper, we propose to use object-centric 3D motion field to represent actions for robot learning from human videos, and present a novel framework for extracting this representation from videos for zero-shot control. We introduce two novel components in its implementation. First, a novel training pipeline for training a ''denoising'' 3D motion field estimator to extract fine object 3D motions from human videos with noisy depth robustly. Second, a dense object-centric 3D motion field prediction architecture that favors both cross-embodiment transfer and policy generalization to background. We evaluate the system in real world setups. Experiments show that our method reduces 3D motion estimation error by over 50% compared to the latest method, achieve 55% average success rate in diverse tasks where prior approaches fail~($\lesssim 10$\%), and can even acquire fine-grained manipulation skills like insertion. 

**Abstract (ZH)**: 从人类视频中学习机器人控制策略：基于物体中心三维运动场的方法 

---
# Pseudo-Simulation for Autonomous Driving 

**Title (ZH)**: 伪仿真技术在自动驾驶中的应用 

**Authors**: Wei Cao, Marcel Hallgarten, Tianyu Li, Daniel Dauner, Xunjiang Gu, Caojun Wang, Yakov Miron, Marco Aiello, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, Kashyap Chitta  

**Link**: [PDF](https://arxiv.org/pdf/2506.04218)  

**Abstract**: Existing evaluation paradigms for Autonomous Vehicles (AVs) face critical limitations. Real-world evaluation is often challenging due to safety concerns and a lack of reproducibility, whereas closed-loop simulation can face insufficient realism or high computational costs. Open-loop evaluation, while being efficient and data-driven, relies on metrics that generally overlook compounding errors. In this paper, we propose pseudo-simulation, a novel paradigm that addresses these limitations. Pseudo-simulation operates on real datasets, similar to open-loop evaluation, but augments them with synthetic observations generated prior to evaluation using 3D Gaussian Splatting. Our key idea is to approximate potential future states the AV might encounter by generating a diverse set of observations that vary in position, heading, and speed. Our method then assigns a higher importance to synthetic observations that best match the AV's likely behavior using a novel proximity-based weighting scheme. This enables evaluating error recovery and the mitigation of causal confusion, as in closed-loop benchmarks, without requiring sequential interactive simulation. We show that pseudo-simulation is better correlated with closed-loop simulations (R^2=0.8) than the best existing open-loop approach (R^2=0.7). We also establish a public leaderboard for the community to benchmark new methodologies with pseudo-simulation. Our code is available at this https URL. 

**Abstract (ZH)**: 现有的自动驾驶汽车评估范式面临严重限制。由于安全问题和缺乏可重复性，现实世界的评估往往具有挑战性，而闭环仿真则可能缺乏真实性或具有高昂的计算成本。开环评估虽然高效且数据驱动，但通常依赖于忽略累积误差的指标。本文提出了一种新型范式——伪仿真，以解决这些限制。伪仿真基于现实数据集运行，类似于开环评估，但通过在评估前使用3D Gaussian Splatting生成合成观察来增强这些数据集。我们的核心思想是通过生成在位置、方向和速度方面变化多样的观察，来近似自动驾驶汽车可能遇到的潜在未来状态。然后，我们的方法使用一种新颖的基于距离的加权方案，赋予那些最佳匹配自动驾驶汽车可能行为的合成观察更高的重要性。这使我们能够在不需序列交互仿真的情况下评估错误恢复和因果混淆的缓解，类似于闭环基准评估。我们证明伪仿真与闭环仿真相关性更高（R²=0.8），而最佳现有开环方法的相关性为R²=0.7。我们还为社区建立了一个公开的排行榜，用于使用伪仿真评估新的方法论。我们的代码可在以下链接获取。 

---
# OWMM-Agent: Open World Mobile Manipulation With Multi-modal Agentic Data Synthesis 

**Title (ZH)**: OWMM-Agent: 开放世界移动 manipulation 与多模态代理数据合成 

**Authors**: Junting Chen, Haotian Liang, Lingxiao Du, Weiyun Wang, Mengkang Hu, Yao Mu, Wenhai Wang, Jifeng Dai, Ping Luo, Wenqi Shao, Lin Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.04217)  

**Abstract**: The rapid progress of navigation, manipulation, and vision models has made mobile manipulators capable in many specialized tasks. However, the open-world mobile manipulation (OWMM) task remains a challenge due to the need for generalization to open-ended instructions and environments, as well as the systematic complexity to integrate high-level decision making with low-level robot control based on both global scene understanding and current agent state. To address this complexity, we propose a novel multi-modal agent architecture that maintains multi-view scene frames and agent states for decision-making and controls the robot by function calling. A second challenge is the hallucination from domain shift. To enhance the agent performance, we further introduce an agentic data synthesis pipeline for the OWMM task to adapt the VLM model to our task domain with instruction fine-tuning. We highlight our fine-tuned OWMM-VLM as the first dedicated foundation model for mobile manipulators with global scene understanding, robot state tracking, and multi-modal action generation in a unified model. Through experiments, we demonstrate that our model achieves SOTA performance compared to other foundation models including GPT-4o and strong zero-shot generalization in real world. The project page is at this https URL 

**Abstract (ZH)**: 基于导航、操作和视觉模型的快速进步，移动操作器能够在许多专门任务中发挥作用。然而，开放世界的移动操作任务（OWMM）仍具有挑战性，因为需要对开放性指令和环境进行泛化，并且需要在全局场景理解与当前代理状态的基础上系统地将高层次决策与低层次机器人控制结合起来。为了应对这一复杂性，我们提出了一种新的多模态代理架构，该架构保持了多视图场景帧和代理状态以供决策，并通过函数调用来控制机器人。第二个挑战是从领域转移引起的幻觉。为了提高代理性能，我们进一步引入了一种针对OWMM任务的代理数据合成管道，通过指令微调使VLM模型适应我们的任务领域。我们强调我们的微调OWMM-VLM是首个专为移动操作器设计的基础模型，具备全局场景理解、机器人状态跟踪和统一模型中的多模态动作生成能力。通过实验，我们展示了我们的模型在与其他基础模型（包括GPT-4o）相比时实现了SOTA性能，并且在现实世界的零样本泛化方面表现出强大的能力。项目页面在此处：https://这个链接 URL。 

---
# SLAC: Simulation-Pretrained Latent Action Space for Whole-Body Real-World RL 

**Title (ZH)**: SLAC: 仿真预训练潜行动作空间赋能全身真实世界 reinforcement 学习 

**Authors**: Jiaheng Hu, Peter Stone, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2506.04147)  

**Abstract**: Building capable household and industrial robots requires mastering the control of versatile, high-degree-of-freedom (DoF) systems such as mobile manipulators. While reinforcement learning (RL) holds promise for autonomously acquiring robot control policies, scaling it to high-DoF embodiments remains challenging. Direct RL in the real world demands both safe exploration and high sample efficiency, which are difficult to achieve in practice. Sim-to-real RL, on the other hand, is often brittle due to the reality gap. This paper introduces SLAC, a method that renders real-world RL feasible for complex embodiments by leveraging a low-fidelity simulator to pretrain a task-agnostic latent action space. SLAC trains this latent action space via a customized unsupervised skill discovery method designed to promote temporal abstraction, disentanglement, and safety, thereby facilitating efficient downstream learning. Once a latent action space is learned, SLAC uses it as the action interface for a novel off-policy RL algorithm to autonomously learn downstream tasks through real-world interactions. We evaluate SLAC against existing methods on a suite of bimanual mobile manipulation tasks, where it achieves state-of-the-art performance. Notably, SLAC learns contact-rich whole-body tasks in under an hour of real-world interactions, without relying on any demonstrations or hand-crafted behavior priors. More information, code, and videos at this http URL 

**Abstract (ZH)**: 构建具备 capabilities 的家用和工业机器人需要掌握多功能、高自由度（DOF）系统，如移动执行器的控制。虽然强化学习（RL）有望自主获取机器人控制策略，但将其扩展到高DOF体蜉仍然是一个挑战。在真实世界中直接进行RL要求安全探索和高样本效率，这在实践中难以实现。相比之下，从仿真到现实世界的RL由于现实差距往往是脆弱的。本文介绍了一种名为SLAC的方法，通过利用低保真模拟器预先训练一个任务无关的潜在动作空间，使复杂体蜉的RL变得可行。SLAC通过一个定制的无监督技能发现方法来训练这个潜在动作空间，该方法旨在促进时间抽象、分离和安全性，从而促进下游学习的效率。一旦学习到潜在动作空间，SLAC便使用它作为新型离策RL算法的动作接口，通过真实世界的互动自主学习下游任务。我们在一系列双臂移动操作任务中对SLAC进行了评估，其性能达到最先进的水平。值得注意的是，SLAC仅在不到一小时的真实世界互动中就学会了接触丰富的全身任务，无需依赖任何演示或手工设计的行为先验。更多信息、代码和视频请访问此网址。 

---
# Splatting Physical Scenes: End-to-End Real-to-Sim from Imperfect Robot Data 

**Title (ZH)**: 物理场景的点绘制：从 imperfect 机器人数据实现端到端的真实世界到模拟世界的转换 

**Authors**: Ben Moran, Mauro Comi, Steven Bohez, Tom Erez, Zhibin Li, Leonard Hasenclever  

**Link**: [PDF](https://arxiv.org/pdf/2506.04120)  

**Abstract**: Creating accurate, physical simulations directly from real-world robot motion holds great value for safe, scalable, and affordable robot learning, yet remains exceptionally challenging. Real robot data suffers from occlusions, noisy camera poses, dynamic scene elements, which hinder the creation of geometrically accurate and photorealistic digital twins of unseen objects. We introduce a novel real-to-sim framework tackling all these challenges at once. Our key insight is a hybrid scene representation merging the photorealistic rendering of 3D Gaussian Splatting with explicit object meshes suitable for physics simulation within a single representation. We propose an end-to-end optimization pipeline that leverages differentiable rendering and differentiable physics within MuJoCo to jointly refine all scene components - from object geometry and appearance to robot poses and physical parameters - directly from raw and imprecise robot trajectories. This unified optimization allows us to simultaneously achieve high-fidelity object mesh reconstruction, generate photorealistic novel views, and perform annotation-free robot pose calibration. We demonstrate the effectiveness of our approach both in simulation and on challenging real-world sequences using an ALOHA 2 bi-manual manipulator, enabling more practical and robust real-to-simulation pipelines. 

**Abstract (ZH)**: 直接从真实机器人运动创建精确的物理模拟具有巨大的价值，可以实现安全、 scalable 和成本效益高的机器人学习，但仍然极具挑战性。真实机器人数据受到遮挡、noise相机姿态、动态场景元素的影响，阻碍了对未见物体生成几何上准确和逼真的数字孪生体。我们提出了一种新颖的真实到模拟框架，同时解决所有这些挑战。我们的关键见解是一种混合场景表示，结合了3D Gaussian斑点的逼真渲染与适合物理模拟的显式对象网格在一个表示中的使用。我们提出了一套端到端优化管道，利用MuJoCo中的可微渲染和可微物理，联合细化场景的所有组件——从对象几何形状和外观到机器人姿态和物理参数——直接从原始和不精确的机器人轨迹中获取。这种统一的优化使我们能够同时实现高保真度的物体网格重建，生成逼真的新视角，并进行无需标注的机器人姿态校准。我们在仿真和具有挑战性的现实世界序列中使用ALOHA 2双臂 manipulator 验证了该方法的有效性，从而实现更加实用和鲁棒的真实到模拟管道。 

---
# Autonomous Vehicle Lateral Control Using Deep Reinforcement Learning with MPC-PID Demonstration 

**Title (ZH)**: 基于MPC-PID演示的自主车辆横向控制深度强化学习方法 

**Authors**: Chengdong Wu, Sven Kirchner, Nils Purschke, Alois C. Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2506.04040)  

**Abstract**: The controller is one of the most important modules in the autonomous driving pipeline, ensuring the vehicle reaches its desired position. In this work, a reinforcement learning based lateral control approach, despite the imperfections in the vehicle models due to measurement errors and simplifications, is presented. Our approach ensures comfortable, efficient, and robust control performance considering the interface between controlling and other modules. The controller consists of the conventional Model Predictive Control (MPC)-PID part as the basis and the demonstrator, and the Deep Reinforcement Learning (DRL) part which leverages the online information from the MPC-PID part. The controller's performance is evaluated in CARLA using the ground truth of the waypoints as inputs. Experimental results demonstrate the effectiveness of the controller when vehicle information is incomplete, and the training of DRL can be stabilized with the demonstration part. These findings highlight the potential to reduce development and integration efforts for autonomous driving pipelines in the future. 

**Abstract (ZH)**: 基于强化学习的车道控制方法：考虑控制与其他模块接口的舒适、高效和稳健性能 

---
# A Bi-Level Optimization Method for Redundant Dual-Arm Minimum Time Problems 

**Title (ZH)**: 双臂冗余最小时间问题的双层优化方法 

**Authors**: Jonathan Fried, Santiago Paternain  

**Link**: [PDF](https://arxiv.org/pdf/2506.03982)  

**Abstract**: In this work, we present a method for minimizing the time required for a redundant dual-arm robot to follow a desired relative Cartesian path at constant path speed by optimizing its joint trajectories, subject to position, velocity, and acceleration limits. The problem is reformulated as a bi-level optimization whose lower level is a convex, closed-form subproblem that maximizes path speed for a fixed trajectory, while the upper level updates the trajectory using a single-chain kinematic formulation and the subgradient of the lower-level value. Numerical results demonstrate the effectiveness of the proposed approach. 

**Abstract (ZH)**: 本研究提出了一种方法，通过优化关节轨迹，在满足位置、速度和加速度限制的情况下，最小化冗余双臂机器人跟随恒定路径速度的期望相对笛卡尔路径所需的时间。该问题被重新表述为一个两层优化问题，其中下层是一个凸的闭式子问题，该子问题在固定轨迹的情况下最大化路径速度，而上层则使用单链运动学公式和下层值的次梯度更新轨迹。数值结果表明所提出方法的有效性。 

---
# FLIP: Flowability-Informed Powder Weighing 

**Title (ZH)**: FLIP: 流动性指导粉体称重 

**Authors**: Nikola Radulov, Alex Wright, Thomas little, Andrew I. Cooper, Gabriella Pizzuto  

**Link**: [PDF](https://arxiv.org/pdf/2506.03896)  

**Abstract**: Autonomous manipulation of powders remains a significant challenge for robotic automation in scientific laboratories. The inherent variability and complex physical interactions of powders in flow, coupled with variability in laboratory conditions necessitates adaptive automation. This work introduces FLIP, a flowability-informed powder weighing framework designed to enhance robotic policy learning for granular material handling. Our key contribution lies in using material flowability, quantified by the angle of repose, to optimise physics-based simulations through Bayesian inference. This yields material-specific simulation environments capable of generating accurate training data, which reflects diverse powder behaviours, for training `robot chemists'. Building on this, FLIP integrates quantified flowability into a curriculum learning strategy, fostering efficient acquisition of robust robotic policies by gradually introducing more challenging, less flowable powders. We validate the efficacy of our method on a robotic powder weighing task under real-world laboratory conditions. Experimental results show that FLIP with a curriculum strategy achieves a low dispensing error of 2.12 +- 1.53 mg, outperforming methods that do not leverage flowability data, such as domain randomisation (6.11 +- 3.92 mg). These results demonstrate FLIP's improved ability to generalise to previously unseen, more cohesive powders and to new target masses. 

**Abstract (ZH)**: 自主处理粉末材料仍是对科学实验室中机器人自动化的一大挑战。基于流动性的粉末称重框架FLIP旨在通过贝叶斯推断优化物理学仿真，从而增强粒状材料处理的机器人策略学习。我们的主要贡献在于使用材料流动性的量度（休止角）来生成材料特定的仿真环境，以生成反映多样粉末行为的准确训练数据，用于训练“机器人化学家”。在此基础上，FLIP将量化后的流动性集成到 curriculum 学习策略中，通过逐步引入更具挑战性和流动性较差的粉末，促进机器人策略的高效学习。我们在实际实验室条件下验证了该方法在机器人粉末称重任务中的有效性。实验结果表明，使用 curriculum 策略的 FLIP 的投料误差为 2.12 ± 1.53 mg，优于未利用流动性数据的方法（如领域随机化，误差为 6.11 ± 3.92 mg）。这些结果证明了 FLIP 在处理之前未见过的更粘稠粉末和新目标质量时具有更好的泛化能力。 

---
# STAR: Learning Diverse Robot Skill Abstractions through Rotation-Augmented Vector Quantization 

**Title (ZH)**: STAR：通过旋转扩增向量量化学习多样化的机器人技能抽象 

**Authors**: Hao Li, Qi Lv, Rui Shao, Xiang Deng, Yinchuan Li, Jianye Hao, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.03863)  

**Abstract**: Transforming complex actions into discrete skill abstractions has demonstrated strong potential for robotic manipulation. Existing approaches mainly leverage latent variable models, e.g., VQ-VAE, to learn skill abstractions through learned vectors (codebooks), while they suffer from codebook collapse and modeling the causal relationship between learned skills. To address these limitations, we present \textbf{S}kill \textbf{T}raining with \textbf{A}ugmented \textbf{R}otation (\textbf{STAR}), a framework that advances both skill learning and composition to complete complex behaviors. Specifically, to prevent codebook collapse, we devise rotation-augmented residual skill quantization (RaRSQ). It encodes relative angles between encoder outputs into the gradient flow by rotation-based gradient mechanism. Points within the same skill code are forced to be either pushed apart or pulled closer together depending on gradient directions. Further, to capture the causal relationship between skills, we present causal skill transformer (CST) which explicitly models dependencies between skill representations through an autoregressive mechanism for coherent action generation. Extensive experiments demonstrate the superiority of STAR on both LIBERO benchmark and realworld tasks, with around 12\% improvement over the baselines. 

**Abstract (ZH)**: 技能训练增强旋转（STAR）：技能学习与组合的提升框架 

---
# Phase-based Nonlinear Model Predictive Control for Humanoid Walking Stabilization with Single and Double Support Time Adjustments 

**Title (ZH)**: 基于相位的非线性模型预测控制在单支撑和双支撑时间调整的人形步行稳定性控制 

**Authors**: Kwanwoo Lee, Gyeongjae Park, Jaeheung Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.03856)  

**Abstract**: Balance control for humanoid robots has been extensively studied to enable robots to navigate in real-world environments. However, balance controllers that explicitly optimize the durations of both the single support phase, also known as step timing, and the Double Support Phase (DSP) have not been widely explored due to the inherent nonlinearity of the associated optimization problem. Consequently, many recent approaches either ignore the DSP or adjust its duration based on heuristics or on linearization techniques that rely on sequential coordination of balance strategies. This study proposes a novel phase-based nonlinear Model Predictive Control (MPC) framework that simultaneously optimizes Zero Moment Point~(ZMP) modulation, step location, step timing, and DSP duration to maintain balance under external disturbances. In simulation, the proposed controller was compared with two state-of-the-art frameworks that rely on heuristics or sequential coordination of balance strategies under two scenarios: forward walking on terrain emulating compliant ground and external push recovery while walking in place. Overall, the findings suggest that the proposed method offers more flexible coordination of balance strategies than the sequential approach, and consistently outperforms the heuristic approach. The robustness and effectiveness of the proposed controller were also validated through experiments with a real humanoid robot. 

**Abstract (ZH)**: 人形机器人平衡控制的研究：基于相位的非线性模型预测控制方法探索 

---
# Enhancing Safety of Foundation Models for Visual Navigation through Collision Avoidance via Repulsive Estimation 

**Title (ZH)**: 通过斥力估计实现碰撞避免以增强视觉导航基础模型的安全性 

**Authors**: Joonkyung Kim, Joonyeol Sim, Woojun Kim, Katia Sycara, Changjoo Nam  

**Link**: [PDF](https://arxiv.org/pdf/2506.03834)  

**Abstract**: We propose CARE (Collision Avoidance via Repulsive Estimation), a plug-and-play module that enhances the safety of vision-based navigation without requiring additional range sensors or fine-tuning of pretrained models. While recent foundation models using only RGB inputs have shown strong performance, they often fail to generalize in out-of-distribution (OOD) environments with unseen objects or variations in camera parameters (e.g., field of view, pose, or focal length). Without fine-tuning, these models may generate unsafe trajectories that lead to collisions, requiring costly data collection and retraining. CARE addresses this limitation by seamlessly integrating with any RGB-based navigation system that outputs local trajectories, dynamically adjusting them using repulsive force vectors derived from monocular depth maps. We evaluate CARE by combining it with state-of-the-art vision-based navigation models across multiple robot platforms. CARE consistently reduces collision rates (up to 100%) without sacrificing goal-reaching performance and improves collision-free travel distance by up to 10.7x in exploration tasks. 

**Abstract (ZH)**: 我们提出CARE（基于排斥估计的碰撞避免模块），这是一个即插即用的模块，能够在无需额外的距离传感器或微调预训练模型的情况下增强基于视觉的导航的安全性。尽管最近只使用RGB输入的基础模型显示出了强大的性能，但在未见过的物体或摄像机参数（如视野、姿态或焦距）变化的分布外（OOD）环境中，它们往往会表现出不佳的泛化能力。未经微调的情况下，这些模型可能会生成不安全的轨迹，从而导致碰撞，这需要昂贵的数据收集和重新训练。CARE通过无缝集成到任何输出局部轨迹的基于RGB的导航系统中，并使用从单目深度图导出的排斥力向量动态调整这些轨迹，从而解决了这一限制。我们通过将CARE与多个机器人平台上的最先进的基于视觉的导航模型结合来评估CARE，CARE在不牺牲目标到达性能的前提下一致地减少了碰撞率（最多100%），并在探索任务中将无碰撞行驶距离提高了最多10.7倍。 

---
# Understanding Physical Properties of Unseen Deformable Objects by Leveraging Large Language Models and Robot Actions 

**Title (ZH)**: 利用大型语言模型和机器人动作理解未见可变形物体的物理属性 

**Authors**: Changmin Park, Beomjoon Lee, Haechan Jung, Haejin Jung, Changjoo Nam  

**Link**: [PDF](https://arxiv.org/pdf/2506.03760)  

**Abstract**: In this paper, we consider the problem of understanding the physical properties of unseen objects through interactions between the objects and a robot. Handling unseen objects with special properties such as deformability is challenging for traditional task and motion planning approaches as they are often with the closed world assumption. Recent results in Large Language Models (LLMs) based task planning have shown the ability to reason about unseen objects. However, most studies assume rigid objects, overlooking their physical properties. We propose an LLM-based method for probing the physical properties of unseen deformable objects for the purpose of task planning. For a given set of object properties (e.g., foldability, bendability), our method uses robot actions to determine the properties by interacting with the objects. Based on the properties examined by the LLM and robot actions, the LLM generates a task plan for a specific domain such as object packing. In the experiment, we show that the proposed method can identify properties of deformable objects, which are further used for a bin-packing task where the properties take crucial roles to succeed. 

**Abstract (ZH)**: 本文考虑通过物体与机器人之间的交互来理解未见物体的物理属性的问题。处理具有变形等特殊属性的未见物体对传统的任务和运动规划方法具有挑战性，因为这些方法通常基于闭世界假设。基于大型语言模型（LLMs）的任务规划近期成果显示出对未见物体进行推理的能力。然而，大多数研究假设物体是刚性的，忽略了它们的物理属性。我们提出一种基于LLM的方法，用于探索未见可变形物体的物理属性，以用于任务规划。对于给定的一组物体属性（例如可折叠性、可弯曲性），我们的方法通过与物体的交互来使用机器人动作确定这些属性。基于LLM检查的属性和机器人动作，LLM生成特定领域（如物体打包）的任务计划。在实验中，我们展示了提出的方法能够识别可变形物体的属性，这些属性进一步用于一项纸箱打包任务，其中属性在成功中起着关键作用。 

---
# An Open-source Capping Machine Suitable for Confined Spaces 

**Title (ZH)**: 适用于受限空间的开源加盖机器 

**Authors**: Francisco Munguia-Galeano, Louis Longley, Satheeshkumar Veeramani, Zhengxue Zhou, Rob Clowes, Hatem Fakhruldeen, Andrew I. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2506.03743)  

**Abstract**: In the context of self-driving laboratories (SDLs), ensuring automated and error-free capping is crucial, as it is a ubiquitous step in sample preparation. Automated capping in SDLs can occur in both large and small workspaces (e.g., inside a fume hood). However, most commercial capping machines are designed primarily for large spaces and are often too bulky for confined environments. Moreover, many commercial products are closed-source, which can make their integration into fully autonomous workflows difficult. This paper introduces an open-source capping machine suitable for compact spaces, which also integrates a vision system that recognises capping failure. The capping and uncapping processes are repeated 100 times each to validate the machine's design and performance. As a result, the capping machine reached a 100 % success rate for capping and uncapping. Furthermore, the machine sealing capacities are evaluated by capping 12 vials filled with solvents of different vapour pressures: water, ethanol and acetone. The vials are then weighed every 3 hours for three days. The machine's performance is benchmarked against an industrial capping machine (a Chemspeed station) and manual capping. The vials capped with the prototype lost 0.54 % of their content weight on average per day, while the ones capped with the Chemspeed and manually lost 0.0078 % and 0.013 %, respectively. The results show that the capping machine is a reasonable alternative to industrial and manual capping, especially when space and budget are limitations in SDLs. 

**Abstract (ZH)**: 在自主驾驶实验室（SDLs）的背景下，确保自动化且无误的盖帽操作至关重要，因为这是样本准备中的一个普遍步骤。SDLs中的自动盖帽可以在大空间和小空间（例如，在通风柜内）中进行。然而，大多数商用盖帽机主要针对大空间设计，往往不适合受限环境。此外，许多商用产品是闭源的，这使得它们难以集成到完全自主的工作流程中。本文介绍了一种适用于紧凑空间的开源盖帽机，并集成了一个能够识别盖帽失败的视觉系统。盖帽和去盖帽过程各重复了100次以验证机器的设计和性能。结果表明，该机器在盖帽和去盖帽方面均达到了100%的成功率。此外，通过将12只装有不同的蒸气压溶剂（水、乙醇和醋酸）的试瓶进行盖帽，评估了该机器的密封性能。随后每隔3小时称量一次试瓶，持续三天。性能基准测试显示，该原型机每天平均损失0.54%的内容物重量，而使用Chemspeed设备和手动盖帽的试瓶分别损失0.0078%和0.013%。结果表明，在SDLs的空间和预算受限时，该盖帽机是工业和手动盖帽的合理替代方案。 

---
# An Improved Grey Wolf Optimizer Inspired by Advanced Cooperative Predation for UAV Shortest Path Planning 

**Title (ZH)**: 受高级协同捕食启发的改进灰狼优化算法在无人机最短路径规划中的应用 

**Authors**: Zuhao Teng, Qian Dong, Ze Zhang, Shuangyao Huang, Wenzhang Zhang, Jingchen Wang, Ji Li, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.03663)  

**Abstract**: With the widespread application of Unmanned Aerial Vehicles (UAVs) in domains like military reconnaissance, emergency rescue, and logistics delivery, efficiently planning the shortest flight path has become a critical challenge. Traditional heuristic-based methods often suffer from the inability to escape from local optima, which limits their effectiveness in finding the shortest path. To address these issues, a novel Improved Grey Wolf Optimizer (IGWO) is presented in this study. The proposed IGWO incorporates an Advanced Cooperative Predation (ACP) and a Lens Opposition-based Learning Strategy (LOBL) in order to improve the optimization capability of the method. Simulation results show that IGWO ranks first in optimization performance on benchmark functions F1-F5, F7, and F9-F12, outperforming all other compared algorithms. Subsequently, IGWO is applied to UAV shortest path planning in various obstacle-laden environments. Simulation results show that the paths planned by IGWO are, on average, shorter than those planned by GWO, PSO, and WOA by 1.70m, 1.68m, and 2.00m, respectively, across four different maps. 

**Abstract (ZH)**: 基于改进灰狼优化算法的无人机最短路径规划 

---
# SwitchVLA: Execution-Aware Task Switching for Vision-Language-Action Models 

**Title (ZH)**: SwitchVLA：面向执行的视觉-语言-动作模型任务切换 

**Authors**: Meng Li, Zhen Zhao, Zhengping Che, Fei Liao, Kun Wu, Zhiyuan Xu, Pei Ren, Zhao Jin, Ning Liu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03574)  

**Abstract**: Robots deployed in dynamic environments must be able to not only follow diverse language instructions but flexibly adapt when user intent changes mid-execution. While recent Vision-Language-Action (VLA) models have advanced multi-task learning and instruction following, they typically assume static task intent, failing to respond when new instructions arrive during ongoing execution. This limitation hinders natural and robust interaction in dynamic settings, such as retail or household environments, where real-time intent changes are common. We propose SwitchVLA, a unified, execution-aware framework that enables smooth and reactive task switching without external planners or additional switch-specific data. We model task switching as a behavior modulation problem conditioned on execution state and instruction context. Expert demonstrations are segmented into temporally grounded contact phases, allowing the policy to infer task progress and adjust its behavior accordingly. A multi-behavior conditional policy is then trained to generate flexible action chunks under varying behavior modes through conditioned trajectory modeling. Experiments in both simulation and real-world robotic manipulation demonstrate that SwitchVLA enables robust instruction adherence, fluid task switching, and strong generalization-outperforming prior VLA baselines in both task success rate and interaction naturalness. 

**Abstract (ZH)**: 部署在动态环境中的机器人必须能够不仅遵循多样的语言指令，还在执行过程中灵活适应用户意图的变化。尽管近期的视觉-语言-动作（VLA）模型已经提高了多任务学习和指令跟随的能力，但它们通常假设固定的任务意图，在执行过程中收到新指令时无法作出响应。这一限制在零售或家庭环境中阻碍了自然和稳健的交互，而在这些环境中，实时的意图变化是常见的。我们提出SwitchVLA，一种执行感知的统一框架，能够在无需外部规划者或额外切换特定数据的情况下实现平滑且反应性的任务切换。我们将任务切换建模为基于执行状态和指令上下文的行为调节问题。通过时间触发的接触阶段对专家演示进行分割，使策略能够推断任务进度并相应地调整其行为。通过条件轨迹建模训练一个多行为条件策略，使其在不同的行为模式下生成灵活的动作片段。在仿真和真实世界的机器人操作实验中，SwitchVLA展示了稳健的指令遵守、流畅的任务切换和强大的泛化能力，超越了先前的VLA基线方法，在任务成功率和交互自然度上表现出色。 

---
# Confidence-Guided Human-AI Collaboration: Reinforcement Learning with Distributional Proxy Value Propagation for Autonomous Driving 

**Title (ZH)**: 基于信心引导的人机协作：分布代理价值传播的强化学习在自动驾驶中的应用 

**Authors**: Li Zeqiao, Wang Yijing, Wang Haoyu, Li Zheng, Li Peng, Zuo zhiqiang, Hu Chuan  

**Link**: [PDF](https://arxiv.org/pdf/2506.03568)  

**Abstract**: Autonomous driving promises significant advancements in mobility, road safety and traffic efficiency, yet reinforcement learning and imitation learning face safe-exploration and distribution-shift challenges. Although human-AI collaboration alleviates these issues, it often relies heavily on extensive human intervention, which increases costs and reduces efficiency. This paper develops a confidence-guided human-AI collaboration (C-HAC) strategy to overcome these limitations. First, C-HAC employs a distributional proxy value propagation method within the distributional soft actor-critic (DSAC) framework. By leveraging return distributions to represent human intentions C-HAC achieves rapid and stable learning of human-guided policies with minimal human interaction. Subsequently, a shared control mechanism is activated to integrate the learned human-guided policy with a self-learning policy that maximizes cumulative rewards. This enables the agent to explore independently and continuously enhance its performance beyond human guidance. Finally, a policy confidence evaluation algorithm capitalizes on DSAC's return distribution networks to facilitate dynamic switching between human-guided and self-learning policies via a confidence-based intervention function. This ensures the agent can pursue optimal policies while maintaining safety and performance guarantees. Extensive experiments across diverse driving scenarios reveal that C-HAC significantly outperforms conventional methods in terms of safety, efficiency, and overall performance, achieving state-of-the-art results. The effectiveness of the proposed method is further validated through real-world road tests in complex traffic conditions. The videos and code are available at: this https URL. 

**Abstract (ZH)**: 自主驾驶有望在移动性、道路安全和交通效率方面带来重大进展，然而强化学习和模仿学习面临着安全探索和分布转移的挑战。尽管人机协作可以减轻这些问题，但往往需要大量的人工干预，增加了成本并降低了效率。本文提出了一种基于信心引导的人机协作（C-HAC）策略以克服这些限制。首先，C-HAC 在分布软演员-评论家 (DSAC) 框架内采用了分布代理价值传播方法。通过利用回报分布来表示人类意图，C-HAC 实现了在最少人类干预下的人类指导政策的快速稳定学习。随后，激活了一种共决策机制，将所学的人类指导策略与最大化累积奖励的自我学习策略相结合。这使智能体能够独立探索并持续提升其性能，超越人类指导。最后，基于 DSAC 的回报分布网络开发了一种策略信心评估算法，通过基于信心的干预函数动态切换人类指导和自我学习策略。这确保智能体能够在保证安全和性能的同时追求最优策略。跨多种驾驶场景的广泛实验表明，C-HAC 在安全、效率和总体性能方面显著优于传统方法，达到顶级性能。所提方法的有效性通过在复杂交通条件下的实车测试进一步验证。相关视频和代码可在以下链接获取：this https URL。 

---
# From Virtual Agents to Robot Teams: A Multi-Robot Framework Evaluation in High-Stakes Healthcare Context 

**Title (ZH)**: 从虚拟代理到机器人团队：在高风险医疗环境中多机器人框架评估 

**Authors**: Yuanchen Bai, Zijian Ding, Angelique Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2506.03546)  

**Abstract**: Advancements in generative models have enabled multi-agent systems (MAS) to perform complex virtual tasks such as writing and code generation, which do not generalize well to physical multi-agent robotic teams. Current frameworks often treat agents as conceptual task executors rather than physically embodied entities, and overlook critical real-world constraints such as spatial context, robotic capabilities (e.g., sensing and navigation). To probe this gap, we reconfigure and stress-test a hierarchical multi-agent robotic team built on the CrewAI framework in a simulated emergency department onboarding scenario. We identify five persistent failure modes: role misalignment; tool access violations; lack of in-time handling of failure reports; noncompliance with prescribed workflows; bypassing or false reporting of task completion. Based on this analysis, we propose three design guidelines emphasizing process transparency, proactive failure recovery, and contextual grounding. Our work informs the development of more resilient and robust multi-agent robotic systems (MARS), including opportunities to extend virtual multi-agent frameworks to the real world. 

**Abstract (ZH)**: 生成模型的进步使多代理系统能够执行复杂的虚拟任务，如写作和代码生成，但这些任务在应用于物理多代理机器人团队时却不能很好地泛化。当前框架往往将代理视为概念上的任务执行者，而不是物理实体，并且忽视了诸如空间上下文、机器人能力（如感知和导航）等关键现实世界约束。为探究这一差距，我们重新配置并在模拟紧急部门入职场景中严格测试了基于CrewAI框架的层次化多代理机器人团队。我们识别了五种持续失败模式：角色错位；工具访问违规；故障报告的及时处理不足；不遵守规定的 workflows；任务完成的绕过或虚假报告。基于这一分析，我们提出了三条设计准则，强调过程透明性、主动失败恢复和上下文依存性。我们的工作为更坚韧和可靠的多代理机器人系统（MARS）的发展提供了指导，并探讨了扩展虚拟多代理框架至现实世界的机会。 

---
# Robust Position Estimation by Rao-Blackwellized Particle Filter without Integer Ambiguity Resolution in Urban Environments 

**Title (ZH)**: 在城市环境中无需整数模糊校决议蔟粒子滤波的稳健位置估计 

**Authors**: Daiki Niimi, An Fujino, Taro Suzuki, Junichi Meguro  

**Link**: [PDF](https://arxiv.org/pdf/2506.03537)  

**Abstract**: This study proposes a centimeter-accurate positioning method that utilizes a Rao-Blackwellized particle filter (RBPF) without requiring integer ambiguity resolution in global navigation satellite system (GNSS) carrier phase measurements. The conventional positioning method employing a particle filter (PF) eliminates the necessity for ambiguity resolution by calculating the likelihood from the residuals of the carrier phase based on the particle position. However, this method encounters challenges, particularly in urban environments characterized by non-line-of-sight (NLOS) multipath errors. In such scenarios, PF tracking may fail due to the degradation of velocity estimation accuracy used for state transitions, thereby complicating subsequent position estimation. To address this issue, we apply Rao-Blackwellization to the conventional PF framework, treating position and velocity as distinct states and employing the Kalman filter for velocity estimation. This approach enhances the accuracy of velocity estimation and, consequently, the precision of position estimation. Moreover, the proposed method rejects NLOS multipath signals based on the pseudorange residuals at each particle position during the velocity estimation step. This process not only enhances velocity accuracy, but also preserves particle diversity by allowing particles to transition to unique states with varying velocities. Consequently, particles are more likely to cluster around the true position, thereby enabling more accurate position estimation. Vehicular experiments in urban environments demonstrated the effectiveness of proposed method in achieving a higher positioning accuracy than conventional PF-based and conventional GNSS positioning methods. 

**Abstract (ZH)**: 一种无需整数模糊度解算的厘米级定位方法：利用 Rao-Blackwellized 颗粒滤波器在全球导航卫星系统载波相位测量中的应用 

---
# SemNav: A Model-Based Planner for Zero-Shot Object Goal Navigation Using Vision-Foundation Models 

**Title (ZH)**: SemNav: 基于模型的零样本物体目标导航规划器使用视觉基础模型 

**Authors**: Arnab Debnath, Gregory J. Stein, Jana Kosecka  

**Link**: [PDF](https://arxiv.org/pdf/2506.03516)  

**Abstract**: Object goal navigation is a fundamental task in embodied AI, where an agent is instructed to locate a target object in an unexplored environment. Traditional learning-based methods rely heavily on large-scale annotated data or require extensive interaction with the environment in a reinforcement learning setting, often failing to generalize to novel environments and limiting scalability. To overcome these challenges, we explore a zero-shot setting where the agent operates without task-specific training, enabling more scalable and adaptable solution. Recent advances in Vision Foundation Models (VFMs) offer powerful capabilities for visual understanding and reasoning, making them ideal for agents to comprehend scenes, identify relevant regions, and infer the likely locations of objects. In this work, we present a zero-shot object goal navigation framework that integrates the perceptual strength of VFMs with a model-based planner that is capable of long-horizon decision making through frontier exploration. We evaluate our approach on the HM3D dataset using the Habitat simulator and demonstrate that our method achieves state-of-the-art performance in terms of success weighted by path length for zero-shot object goal navigation. 

**Abstract (ZH)**: 零样本物体目标导航是具身AI中的一个基本任务，其中代理被指示在未探索的环境中定位目标物体。传统基于学习的方法依赖大量标注数据或在强化学习设置中需要大量与环境交互，往往无法泛化到新型环境，限制了可扩展性。为克服这些挑战，我们探索了一个零样本设置，其中代理在没有任务特定训练的情况下运行，从而实现更具扩展性和适应性的解决方案。视觉基础模型（VFMs）的最近进展提供了强大的视觉理解与推理能力，使其成为代理理解场景、识别相关区域和推断物体可能位置的理想选择。在此工作中，我们提出了一种结合VFMs感知优势与基于模型的规划器的零样本物体目标导航框架，该规划器能够通过前沿探索进行长视窗决策制定。我们在HM3D数据集上使用Habitat模拟器评估了我们的方法，并展示了我们的方法在零样本物体目标导航方面的性能达到了最新水平，按照路径长度加权的成功率。 

---
# Design of Trimmed Helicoid Soft-Rigid Hybrid Robots 

**Title (ZH)**: 精裁螺旋软硬混合机器人设计 

**Authors**: Zach J. Patterson, Emily R. Sologuren, Daniela Rus  

**Link**: [PDF](https://arxiv.org/pdf/2506.03380)  

**Abstract**: As soft robot design matures, researchers have converged to sophisticated design paradigms to enable the development of more suitable platforms. Two such paradigms are soft-rigid hybrid robots, which utilize rigid structural materials in some aspect of the robot's design, and architectured materials, which deform based on geometric parameters as opposed to purely material ones. In this work, we combine the two design approaches, utilizing trimmed helicoid structures in series with rigid linkages. Additionally, we extend the literature on wave spring-inspired soft structures by deriving a mechanical model of the stiffness for arbitrary geometries. We present a novel manufacturing method for such structures utilizing an injection molding approach and we make available the design tool to generate 3D printed molds for arbitrary designs of this class. Finally, we produce a robot using the above methods and operate it in closed-loop demonstrations. 

**Abstract (ZH)**: 随着软体机器人设计的成熟，研究人员已经汇聚到了更加 sophisticated 的设计范式，以促进更适合的平台的发展。两种这样的范式是软-硬混合机器人，它们在机器人的设计中利用了刚性结构材料，以及基于几何参数而非单纯材料属性变形的架构材料。在此项工作中，我们结合了这两种设计方法，利用带有限制的螺旋面结构与刚性连杆相结合。此外，我们通过推导任意几何形状下的刚度机械模型，扩展了受到波弹簧启发的软结构的文献。我们提出了一种利用注塑成型方法制造此类结构的新颖制造方法，并提供了生成此类设计3D打印模具的工具。最后，我们使用上述方法制作了一个机器人，并在闭环演示中对其进行操作。 

---
# Robustness-Aware Tool Selection and Manipulation Planning with Learned Energy-Informed Guidance 

**Title (ZH)**: 考虑鲁棒性的工具选择与操作规划：基于能量指导的学习方法 

**Authors**: Yifei Dong, Yan Zhang, Sylvain Calinon, Florian T. Pokorny  

**Link**: [PDF](https://arxiv.org/pdf/2506.03362)  

**Abstract**: Humans subconsciously choose robust ways of selecting and using tools, based on years of embodied experience -- for example, choosing a ladle instead of a flat spatula to serve meatballs. However, robustness under uncertainty remains underexplored in robotic tool-use planning. This paper presents a robustness-aware framework that jointly selects tools and plans contact-rich manipulation trajectories, explicitly optimizing for robustness against environmental disturbances. At the core of our approach is a learned, energy-based robustness metric, which guides the planner towards robust manipulation behaviors. We formulate a hierarchical optimization pipeline that first identifies a tool and configuration that optimizes robustness, and then plans a corresponding manipulation trajectory that maintains robustness throughout execution. We evaluate our approach across three representative tool-use tasks. Simulation and real-world results demonstrate that our approach consistently selects robust tools and generates disturbance-resilient manipulation plans. 

**Abstract (ZH)**: 人类在多年 corporeal 经验的基础上无意识地选择和使用鲁棒性强的工具——例如，用汤匙而不是平坦的羹勺来盛肉丸。然而，机器人工具使用规划中鲁棒性在不确定条件下的研究仍不够充分。本文提出了一个鲁棒性aware框架，该框架同时选择工具并规划富含接触的操作轨迹，明确地针对环境扰动下的鲁棒性进行优化。我们方法的核心是一个学习到的能量基础鲁棒性度量，该度量引导规划器向鲁棒性的操作行为靠拢。我们提出了一个分层优化管道，首先识别出优化鲁棒性的工具和配置，然后规划一个相应的操作轨迹，以在整个执行过程中保持鲁棒性。我们通过三个代表性工具使用任务评估了我们的方法。仿真和实际结果表明，我们的方法能够一致地选择鲁棒性强的工具并生成扰动抵抗力强的操作计划。 

---
# Adversarial Attacks on Robotic Vision Language Action Models 

**Title (ZH)**: 面向机器人视觉语言行动模型的对抗攻击 

**Authors**: Eliot Krzysztof Jones, Alexander Robey, Andy Zou, Zachary Ravichandran, George J. Pappas, Hamed Hassani, Matt Fredrikson, J. Zico Kolter  

**Link**: [PDF](https://arxiv.org/pdf/2506.03350)  

**Abstract**: The emergence of vision-language-action models (VLAs) for end-to-end control is reshaping the field of robotics by enabling the fusion of multimodal sensory inputs at the billion-parameter scale. The capabilities of VLAs stem primarily from their architectures, which are often based on frontier large language models (LLMs). However, LLMs are known to be susceptible to adversarial misuse, and given the significant physical risks inherent to robotics, questions remain regarding the extent to which VLAs inherit these vulnerabilities. Motivated by these concerns, in this work we initiate the study of adversarial attacks on VLA-controlled robots. Our main algorithmic contribution is the adaptation and application of LLM jailbreaking attacks to obtain complete control authority over VLAs. We find that textual attacks, which are applied once at the beginning of a rollout, facilitate full reachability of the action space of commonly used VLAs and often persist over longer horizons. This differs significantly from LLM jailbreaking literature, as attacks in the real world do not have to be semantically linked to notions of harm. We make all code available at this https URL . 

**Abstract (ZH)**: 基于视觉-语言-动作模型的端到端控制正通过在十亿参数规模上融合多模态 sensory 输入重塑机器人领域。这些模型的能力主要源自其架构，这些架构往往基于前沿的大语言模型（LLMs）。然而，LLMs 被认为容易遭受对抗性滥用，在考虑到机器人固有的重大物理风险的情况下，关于 VLAs 是否会继承这些漏洞的问题仍然存在。基于这些担忧，本文旨在研究对抗性攻击对由 VLA 控制的机器人的影响。我们的主要算法贡献是将大语言模型（LLMs）的 jailbreaking 攻击适应并应用于获得对 VLAs 的完全控制权限。我们发现，这类文本攻击一旦在 rollout 开始时应用，就能实现对常用 VLA 动作空间的全面探索，并且往往能够在较长的时间范围内持续发挥作用。这与 LLM jailbreaking 文献中的情况有很大不同，因为在现实世界中的攻击并不需要与危害的概念存在语义上的关联。代码已全部开源：this https URL。 

---
# Dynamics and Control of Vision-Aided Multi-UAV-tethered Netted System Capturing Non-Cooperative Target 

**Title (ZH)**: 视觉辅助多无人机缆网系统捕获非合作目标的动力学与控制 

**Authors**: Runhan Liu, Hui Ren, Wei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.03297)  

**Abstract**: As the number of Unmanned Aerial Vehicles (UAVs) operating in low-altitude airspace continues to increase, non-cooperative targets pose growing challenges to low-altitude operations. To address this issue, this paper proposes a multi-UAV-tethered netted system as a non-lethal solution for capturing non-cooperative targets. To validate the proposed system, we develop mySim, a multibody dynamics-based UAV simulation environment that integrates high-precision physics modeling, vision-based motion tracking, and reinforcement learning-driven control strategies. In mySim, the spring-damper model is employed to simulate the dynamic behavior of the tethered net, while the dynamics of the entire system is modeled using multibody dynamics (MBD) to achieve accurate representations of system interactions. The motion of the UAVs and the target are estimated using VINS-MONO and DETR, and the system autonomously executes the capture strategy through MAPPO. Simulation results demonstrate that mySim accurately simulates dynamics and control of the system, successfully enabling the multi-UAV-tethered netted system to capture both non-propelled and maneuvering non-cooperative targets. By providing a high-precision simulation platform that integrates dynamics modeling with perception and learning-based control, mySim enables efficient testing and optimization of UAV-based control policies before real-world deployment. This approach offers significant advantages for simulating complex UAVs coordination tasks and has the potential to be applied to the design of other UAV-based systems. 

**Abstract (ZH)**: 随着低空 airspace内无人驾驶航空器（UAVs）的数量持续增加，非配合目标对低空操作构成了日益严峻的挑战。为了解决这一问题，本文提出了一种多UAV联动网捕系统，作为捕获非配合目标的非致命解决方案。为了验证所提出系统的效果，我们开发了mySim，这是一种基于多体动力学的UAV仿真环境，结合了高精度物理建模、基于视觉的运动跟踪以及基于强化学习的控制策略。在mySim中，弹簧阻尼模型用于模拟联动网的动态行为，整个系统的动力学则通过多体动力学（MBD）建模以实现系统交互的精确表示。无人机和目标的运动分别通过VINS-MONO和DETR进行估计，系统自主执行捕获策略通过MAPPO实现。仿真结果表明，mySim准确模拟了系统的动力学和控制，成功使多UAV联动网捕系统能够捕获非推进的和机动的非配合目标。通过提供一种综合动力学建模与感知及学习驱动控制的高精度仿真平台，mySim能够在实际部署前高效地测试和优化基于UAV的控制策略。该方法对于模拟复杂的UAV协调任务具有显著优势，有望应用于其他基于UAV系统的研发设计中。 

---
# Grounded Vision-Language Interpreter for Integrated Task and Motion Planning 

**Title (ZH)**: 基于语义理解的综合任务与运动规划视觉-语言解释器 

**Authors**: Jeremy Siburian, Keisuke Shirai, Cristian C. Beltran-Hernandez, Masashi Hamaya, Michael Görner, Atsushi Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2506.03270)  

**Abstract**: While recent advances in vision-language models (VLMs) have accelerated the development of language-guided robot planners, their black-box nature often lacks safety guarantees and interpretability crucial for real-world deployment. Conversely, classical symbolic planners offer rigorous safety verification but require significant expert knowledge for setup. To bridge the current gap, this paper proposes ViLaIn-TAMP, a hybrid planning framework for enabling verifiable, interpretable, and autonomous robot behaviors. ViLaIn-TAMP comprises three main components: (1) ViLaIn (Vision-Language Interpreter) - A prior framework that converts multimodal inputs into structured problem specifications using off-the-shelf VLMs without additional domain-specific training, (2) a modular Task and Motion Planning (TAMP) system that grounds these specifications in actionable trajectory sequences through symbolic and geometric constraint reasoning and can utilize learning-based skills for key manipulation phases, and (3) a corrective planning module which receives concrete feedback on failed solution attempts from the motion and task planning components and can feed adapted logic and geometric feasibility constraints back to ViLaIn to improve and further refine the specification. We evaluate our framework on several challenging manipulation tasks in a cooking domain. We demonstrate that the proposed closed-loop corrective architecture exhibits a more than 30% higher mean success rate for ViLaIn-TAMP compared to without corrective planning. 

**Abstract (ZH)**: ViLaIn-TAMP：一种可验证、可解释和自主的机器人行为混合规划框架 

---
# AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment 

**Title (ZH)**: AmbiK: 家庭环境中歧义任务的数据集 

**Authors**: Anastasiia Ivanova, Eva Bakaeva, Zoya Volovikova, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2506.04089)  

**Abstract**: As a part of an embodied agent, Large Language Models (LLMs) are typically used for behavior planning given natural language instructions from the user. However, dealing with ambiguous instructions in real-world environments remains a challenge for LLMs. Various methods for task ambiguity detection have been proposed. However, it is difficult to compare them because they are tested on different datasets and there is no universal benchmark. For this reason, we propose AmbiK (Ambiguous Tasks in Kitchen Environment), the fully textual dataset of ambiguous instructions addressed to a robot in a kitchen environment. AmbiK was collected with the assistance of LLMs and is human-validated. It comprises 1000 pairs of ambiguous tasks and their unambiguous counterparts, categorized by ambiguity type (Human Preferences, Common Sense Knowledge, Safety), with environment descriptions, clarifying questions and answers, user intents, and task plans, for a total of 2000 tasks. We hope that AmbiK will enable researchers to perform a unified comparison of ambiguity detection methods. AmbiK is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）作为嵌入式代理的一部分，通常用于根据用户的自然语言指令进行行为规划。然而，在实际环境处理含糊指令仍然是LLMs的一个挑战。针对任务含糊性检测的各种方法已被提出，但由于它们在不同的数据集上测试且没有通用基准，因此很难进行比较。为此，我们提出AmbiK（厨房环境中的含糊任务），这是在厨房环境中针对机器人提出的含糊指令的完整文本数据集。AmbiK通过LLMs辅助收集并由人类验证。它包含1000对含糊任务及其明确对应任务，按含糊性类型（人类偏好、常识知识、安全）分类，包含环境描述、澄清问题及答案、用户意图和任务计划，共计2000个任务。我们希望AmbiK能够使研究人员能够进行统一的含糊性检测方法比较。AmbiK可在以下链接获取：this https URL。 

---
# Optimizing Mesh to Improve the Triangular Expansion Algorithm for Computing Visibility Regions 

**Title (ZH)**: 优化网格以提高计算可视区域的三角扩展算法性能 

**Authors**: Jan Mikula, Miroslav Kulich  

**Link**: [PDF](https://arxiv.org/pdf/2506.04086)  

**Abstract**: This paper addresses the problem of improving the query performance of the triangular expansion algorithm (TEA) for computing visibility regions by finding the most advantageous instance of the triangular mesh, the preprocessing structure. The TEA recursively traverses the mesh while keeping track of the visible region, the set of all points visible from a query point in a polygonal world. We show that the measured query time is approximately proportional to the number of triangle edge expansions during the mesh traversal. We propose a new type of triangular mesh that minimizes the expected number of expansions assuming the query points are drawn from a known probability distribution. We design a heuristic method to approximate the mesh and evaluate the approach on many challenging instances that resemble real-world environments. The proposed mesh improves the mean query times by 12-16% compared to the reference constrained Delaunay triangulation. The approach is suitable to boost offline applications that require computing millions of queries without addressing the preprocessing time. The implementation is publicly available to replicate our experiments and serve the community. 

**Abstract (ZH)**: 本文解决了提高计算可视区域的三角扩展算法（TEA）查询性能的问题，通过找到最有利的三角网实例，即预处理结构。我们展示了测量的查询时间大约与mesh遍历时的三角边扩展次数成正比。我们提出了一种新的三角网类型，该类型在查询点来自已知概率分布的情况下，最小化了预期的扩展次数。我们设计了一种启发式方法来近似三角网，并在许多类似于真实环境的挑战性实例上评估了该方法。与参考的约束Delaunay三角剖分相比，所提出的三角网将平均查询时间改善了12-16%。该方法适用于需要在不考虑预处理时间的情况下计算数百万查询的离线应用。实现已公开，可供复制我们的实验并服务于社区使用。 

---
# Zero-Shot Temporal Interaction Localization for Egocentric Videos 

**Title (ZH)**: 零样本自视点视频时空交互定位 

**Authors**: Erhang Zhang, Junyi Ma, Yin-Dong Zheng, Yixuan Zhou, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03662)  

**Abstract**: Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We will release our code and relevant data as open-source at this https URL. 

**Abstract (ZH)**: 基于视频的人-物交互（HOI）动作定位为多个下游任务（如人类行为分析和人机技能转移）奠定了基础。当前的时间动作定位方法通常依赖于交互的标注动作和对象类别进行优化，这导致了领域偏差和低部署效率。尽管一些近期工作使用大型视觉-语言模型（VLMs）实现了零样本时间动作定位（ZS-TAL），但它们粗粒度的估计和开环管道阻碍了进一步的时间交互定位（TIL）性能改进。为了解决这些问题，我们提出了一种名为EgoLoc的新型零样本TIL方法，用于在第一人称视频中定位人-物交互的抓取动作时间。EgoLoc引入了一种自我适应的采样策略，以生成合理的视觉提示供VLM推理。通过结合2D和3D观察，它根据3D手速度在可能的接触/分离时间戳周围直接抽样高质量的初始猜测，从而提高了推理的准确性和效率。此外，EgoLoc从视觉和动态线索中生成闭环反馈以进一步细化定位结果。在公开数据集和我们新提出的基准上的全面实验表明，EgoLoc在第一人称视频的时间交互定位方面优于最新的基线方法。我们将在该网址发布我们的代码和相关数据：this https URL。 

---
# SplArt: Articulation Estimation and Part-Level Reconstruction with 3D Gaussian Splatting 

**Title (ZH)**: SplArt: 基于3D 高斯点云的艺术关节估计与部分级别重建 

**Authors**: Shengjie Lin, Jiading Fang, Muhammad Zubair Irshad, Vitor Campagnolo Guizilini, Rares Andrei Ambrus, Greg Shakhnarovich, Matthew R. Walter  

**Link**: [PDF](https://arxiv.org/pdf/2506.03594)  

**Abstract**: Reconstructing articulated objects prevalent in daily environments is crucial for applications in augmented/virtual reality and robotics. However, existing methods face scalability limitations (requiring 3D supervision or costly annotations), robustness issues (being susceptible to local optima), and rendering shortcomings (lacking speed or photorealism). We introduce SplArt, a self-supervised, category-agnostic framework that leverages 3D Gaussian Splatting (3DGS) to reconstruct articulated objects and infer kinematics from two sets of posed RGB images captured at different articulation states, enabling real-time photorealistic rendering for novel viewpoints and articulations. SplArt augments 3DGS with a differentiable mobility parameter per Gaussian, achieving refined part segmentation. A multi-stage optimization strategy is employed to progressively handle reconstruction, part segmentation, and articulation estimation, significantly enhancing robustness and accuracy. SplArt exploits geometric self-supervision, effectively addressing challenging scenarios without requiring 3D annotations or category-specific priors. Evaluations on established and newly proposed benchmarks, along with applications to real-world scenarios using a handheld RGB camera, demonstrate SplArt's state-of-the-art performance and real-world practicality. Code is publicly available at this https URL. 

**Abstract (ZH)**: 基于3D高斯束的自监督骨架物体重建框架SplArt 

---
# Occlusion-Aware Ground Target Tracking by a Dubins Vehicle Using Visibility Volumes 

**Title (ZH)**: 基于视见体积的幽靈车 occlusion 意识地面目标跟踪 

**Authors**: Collin Hague, Artur Wolek  

**Link**: [PDF](https://arxiv.org/pdf/2506.03400)  

**Abstract**: This paper considers the problem of tracking a point of interest (POI) moving along a known trajectory on the ground with an uncrewed aerial vehicle (UAV) modeled as a Dubins vehicle using a line-of-sight (LOS) sensor through an urban environment that may occlude the POI. A visibility volume (VV) encodes a time-varying, three-dimensional representation of the sensing constraints for a particular POI position. A constant-altitude, translating, and radially time-varying circular standoff orbit is then inscribed within the dynamically changing VV centered at the POI position. The time-varying VV is approximated by placing static VVs along the POI's trajectory using an adaptive metric that restricts the volume change of consecutive visibility volumes to below a specified rate. The time-varying circular standoff orbit is proven to be feasible for a Dubins vehicle and is approximated with a piecewise set of linearly interpolated circular orbits inside the static VVs. A steering controller is derived that drives the UAV to converge to the time-varying standoff orbit. Numerical simulations and a flight test illustrate the proposed approach. 

**Abstract (ZH)**: 基于视线传感器的城市环境遮挡下移动兴趣点跟踪的遮挡体积方法：Dubins无人机的动态径向变换单圆轨道控制 

---
