# Robotic Visual Instruction 

**Title (ZH)**: 机器人视觉指导 

**Authors**: Yanbang Li, Ziyang Gong, Haoyang Li, Haoyang Li, Xiaoqi Huang, Haolan Kang, Guangping Bai, Xianzheng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00693)  

**Abstract**: Recently, natural language has been the primary medium for human-robot interaction. However, its inherent lack of spatial precision for robotic control introduces challenges such as ambiguity and verbosity. To address these limitations, we introduce the Robotic Visual Instruction (RoVI), a novel paradigm to guide robotic tasks through an object-centric, hand-drawn symbolic representation. RoVI effectively encodes spatial-temporal information into human-interpretable visual instructions through 2D sketches, utilizing arrows, circles, colors, and numbers to direct 3D robotic manipulation. To enable robots to understand RoVI better and generate precise actions based on RoVI, we present Visual Instruction Embodied Workflow (VIEW), a pipeline formulated for RoVI-conditioned policies. This approach leverages Vision-Language Models (VLMs) to interpret RoVI inputs, decode spatial and temporal constraints from 2D pixel space via keypoint extraction, and then transform them into executable 3D action sequences. We additionally curate a specialized dataset of 15K instances to fine-tune small VLMs for edge deployment, enabling them to effectively learn RoVI capabilities. Our approach is rigorously validated across 11 novel tasks in both real and simulated environments, demonstrating significant generalization capability. Notably, VIEW achieves an 87.5% success rate in real-world scenarios involving unseen tasks that feature multi-step actions, with disturbances, and trajectory-following requirements. Code and Datasets in this paper will be released soon. 

**Abstract (ZH)**: 最近，自然语言已成为人机交互的主要媒介。然而，其在机器人控制中固有的空间精度不足问题带来了模糊性和冗余性的挑战。为了解决这些局限性，我们引入了机器人视觉指令（RoVI），这是一种通过基于对象的手绘符号表示来指导机器人任务的新范式。RoVI 有效利用二维草图中的箭头、圆圈、颜色和数字，编码空间-时间信息，从而使人类能够理解这些视觉指令，并引导三维机器人操作。为了使机器人更好地理解和根据 RoVI 生成精确动作，我们提出了视觉指令体态工作流（VIEW），这是一个针对 RoVI 条件策略的流水线。该方法利用视觉-语言模型（VLMs）来解释 RoVI 输入，通过关键点提取从二维像素空间解码空间和时间约束，并将其转换为可执行的三维动作序列。我们还精心构建了一个包含15,000个实例的专业数据集，用于 fine-tune 小型 VLMs，使其能够有效学习 RoVI 能力。我们的方法在现实和模拟环境中11个新型任务上进行了严格验证，展示了显著的泛化能力。值得注意的是，VIEW 在包含多步骤操作、干扰和轨迹跟踪要求的未见任务真实场景中达到87.5%的成功率。本文中的代码和数据集不久将公开发布。 

---
# Multi-Constraint Safe Reinforcement Learning via Closed-form Solution for Log-Sum-Exp Approximation of Control Barrier Functions 

**Title (ZH)**: 基于 Log-Sum-Exp 近似控制屏障函数的闭形式解多约束安全强化学习 

**Authors**: Chenggang Wang, Xinyi Wang, Yutong Dong, Lei Song, Xinping Guan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00671)  

**Abstract**: The safety of training task policies and their subsequent application using reinforcement learning (RL) methods has become a focal point in the field of safe RL. A central challenge in this area remains the establishment of theoretical guarantees for safety during both the learning and deployment processes. Given the successful implementation of Control Barrier Function (CBF)-based safety strategies in a range of control-affine robotic systems, CBF-based safe RL demonstrates significant promise for practical applications in real-world scenarios. However, integrating these two approaches presents several challenges. First, embedding safety optimization within the RL training pipeline requires that the optimization outputs be differentiable with respect to the input parameters, a condition commonly referred to as differentiable optimization, which is non-trivial to solve. Second, the differentiable optimization framework confronts significant efficiency issues, especially when dealing with multi-constraint problems. To address these challenges, this paper presents a CBF-based safe RL architecture that effectively mitigates the issues outlined above. The proposed approach constructs a continuous AND logic approximation for the multiple constraints using a single composite CBF. By leveraging this approximation, a close-form solution of the quadratic programming is derived for the policy network in RL, thereby circumventing the need for differentiable optimization within the end-to-end safe RL pipeline. This strategy significantly reduces computational complexity because of the closed-form solution while maintaining safety guarantees. Simulation results demonstrate that, in comparison to existing approaches relying on differentiable optimization, the proposed method significantly reduces training computational costs while ensuring provable safety throughout the training process. 

**Abstract (ZH)**: 使用强化学习方法训练任务策略及其后续应用的安全性已成为安全强化学习领域的焦点。在这一领域中，确保学习和部署过程中的安全性理论保证的建立仍然是一个核心挑战。鉴于控制 Barrier 函数（CBF）方法在多种控制仿射机器人系统中成功实施，基于 CBF 的安全强化学习在实际应用场景中展现出巨大的潜力。然而，将这两种方法结合在一起存在若干挑战。首先，将安全优化嵌入到 RL 训练管道中需要优化输出对输入参数可微分，这一条件通常被称为可微优化，解决这一问题并不容易。其次，可微优化框架在处理多约束问题时存在显著的效率问题。为了解决这些挑战，本文提出了一种基于 CBF 的安全强化学习架构，有效缓解了上述问题。所提出的方案通过单个复合 CBF 构建了多约束的连续 AND 逻辑近似。利用这一近似，我们为 RL 中的策略网络推导出了二次规划的闭式解，从而绕过了端到端安全强化学习管道中的可微优化需求。这一策略由于闭式解的存在显著降低了计算复杂度，同时保持了安全保证。仿真结果表明，与依赖于可微优化的方法相比，所提出的方法在确保整个训练过程可证明安全性的前提下，显著降低了训练的计算成本。 

---
# GeoDEx: A Unified Geometric Framework for Tactile Dexterous and Extrinsic Manipulation under Force Uncertainty 

**Title (ZH)**: GeoDEx：一种在力不确定性条件下统一的几何框架用于触觉灵巧操作和外部操纵 

**Authors**: Sirui Chen, Sergio Aguilera Marinovic, Soshi Iba, Rana Soltani Zarrin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00647)  

**Abstract**: Sense of touch that allows robots to detect contact and measure interaction forces enables them to perform challenging tasks such as grasping fragile objects or using tools. Tactile sensors in theory can equip the robots with such capabilities. However, accuracy of the measured forces is not on a par with those of the force sensors due to the potential calibration challenges and noise. This has limited the values these sensors can offer in manipulation applications that require force control. In this paper, we introduce GeoDEx, a unified estimation, planning, and control framework using geometric primitives such as plane, cone and ellipsoid, which enables dexterous as well as extrinsic manipulation in the presence of uncertain force readings. Through various experimental results, we show that while relying on direct inaccurate and noisy force readings from tactile sensors results in unstable or failed manipulation, our method enables successful grasping and extrinsic manipulation of different objects. Additionally, compared to directly running optimization using SOCP (Second Order Cone Programming), planning and force estimation using our framework achieves a 14x speed-up. 

**Abstract (ZH)**: 基于几何原语的GeoDEx：一种在不确定力读数下实现灵巧和外部 manipulation的统一估计、规划和控制框架 

---
# Forward kinematics of a general Stewart-Gough platform by elimination templates 

**Title (ZH)**: 一般Stewart-Gough平台的前向运动学求解方法——消元模板 Approach to Forward Kinematics of a General Stewart-Gough Platform Using Elimination Templates 

**Authors**: Evgeniy Martyushev  

**Link**: [PDF](https://arxiv.org/pdf/2505.00634)  

**Abstract**: The paper proposes an efficient algebraic solution to the problem of forward kinematics for a general Stewart-Gough platform. The problem involves determining all possible postures of a mobile platform connected to a fixed base by six legs, given the leg lengths and the internal geometries of the platform and base. The problem is known to have 40 solutions (whether real or complex). The proposed algorithm consists of three main steps: (i) a specific sparse matrix of size 293x362 (the elimination template) is constructed from the coefficients of the polynomial system describing the platform's kinematics; (ii) the PLU decomposition of this matrix is used to construct a pair of 69x69 matrices; (iii) all 40 solutions (including complex ones) are obtained by computing the generalized eigenvectors of this matrix pair. The proposed algorithm is numerically robust, computationally efficient, and straightforward to implement - requiring only standard linear algebra decompositions. MATLAB, Julia, and Python implementations of the algorithm will be made publicly available. 

**Abstract (ZH)**: 本文提出了一种有效地求解通用Stewart-Gough平台前向运动学问题的代数解决方案。该问题涉及在给出腿长以及平台和基座的内部几何结构的情况下，确定连接在固定基座上的移动平台的所有可能姿态。该问题已知有40个解（无论是实数解还是复数解）。所提出的算法包括三个主要步骤：（i）从描述平台运动学的多项式系统系数中构建一个具体的稀疏矩阵（大小为293x362，称为消元模板）；（ii）使用该矩阵的PLU分解构造一对69x69矩阵；（iii）通过计算这对矩阵的广义特征向量来获得所有40个解（包括复数解）。所提出的算法在数值上稳健、计算上高效且易于实现，仅需标准线性代数分解。算法的MATLAB、Julia和Python实现将被公开发布。 

---
# Neural Network Verification for Gliding Drone Control: A Case Study 

**Title (ZH)**: 神经网络验证在滑行无人机控制中的研究：一个案例研究 

**Authors**: Colin Kessler, Ekaterina Komendantskaya, Marco Casadio, Ignazio Maria Viola, Thomas Flinkow, Albaraa Ammar Othman, Alistair Malhotra, Robbie McPherson  

**Link**: [PDF](https://arxiv.org/pdf/2505.00622)  

**Abstract**: As machine learning is increasingly deployed in autonomous systems, verification of neural network controllers is becoming an active research domain. Existing tools and annual verification competitions suggest that soon this technology will become effective for real-world applications. Our application comes from the emerging field of microflyers that are passively transported by the wind, which may have various uses in weather or pollution monitoring. Specifically, we investigate centimetre-scale bio-inspired gliding drones that resemble Alsomitra macrocarpa diaspores. In this paper, we propose a new case study on verifying Alsomitra-inspired drones with neural network controllers, with the aim of adhering closely to a target trajectory. We show that our system differs substantially from existing VNN and ARCH competition benchmarks, and show that a combination of tools holds promise for verifying such systems in the future, if certain shortcomings can be overcome. We propose a novel method for robust training of regression networks, and investigate formalisations of this case study in Vehicle and CORA. Our verification results suggest that the investigated training methods do improve performance and robustness of neural network controllers in this application, but are limited in scope and usefulness. This is due to systematic limitations of both Vehicle and CORA, and the complexity of our system reducing the scale of reachability, which we investigate in detail. If these limitations can be overcome, it will enable engineers to develop safe and robust technologies that improve people's lives and reduce our impact on the environment. 

**Abstract (ZH)**: 随着机器学习在自主系统中的广泛应用，神经网络控制器的验证正成为一个活跃的研究领域。现有工具和年度验证竞赛表明，这项技术不久将在实际应用中变得有效。我们的应用来自于通过风被动传输的新兴微飞行器领域，这些微飞行器在天气或污染监控等方面可能有多种用途。具体来说，我们研究了类似Alsomitra macrocarpa种子的厘米级生物灵感滑翔无人机。本文提出了一种新的案例研究，旨在验证受Alsomitra启发的无人机的神经网络控制器，确保它们能够紧密遵循目标轨迹。我们表明，我们的系统与现有的VNN和ARCH竞赛基准有显著差异，并且表明如果克服某些缺陷，未来组合工具有望验证此类系统。我们提出了一种新的回归网络鲁棒训练方法，并在Vehicle和CORA中对这一案例研究进行了形式化分析。验证结果表明，所研究的训练方法确实提高了该应用中神经网络控制器的性能和鲁棒性，但其应用范围和实用性有限。这是因为Vehicle和CORA的系统限制以及我们系统的复杂性限制了可达性的规模，我们在文中对此进行了详细分析。如果能克服这些限制，将使工程师能够开发出安全且鲁棒的技术，改善人们的生活并减少我们对环境的影响。 

---
# A Finite-State Controller Based Offline Solver for Deterministic POMDPs 

**Title (ZH)**: 基于有限状态控制器的离线求解器：确定性POMDP问题 

**Authors**: Alex Schutz, Yang You, Matias Mattamala, Ipek Caliskanelli, Bruno Lacerda, Nick Hawes  

**Link**: [PDF](https://arxiv.org/pdf/2505.00596)  

**Abstract**: Deterministic partially observable Markov decision processes (DetPOMDPs) often arise in planning problems where the agent is uncertain about its environmental state but can act and observe deterministically. In this paper, we propose DetMCVI, an adaptation of the Monte Carlo Value Iteration (MCVI) algorithm for DetPOMDPs, which builds policies in the form of finite-state controllers (FSCs). DetMCVI solves large problems with a high success rate, outperforming existing baselines for DetPOMDPs. We also verify the performance of the algorithm in a real-world mobile robot forest mapping scenario. 

**Abstract (ZH)**: 确定性部分可观测马尔可夫决策过程（DetPOMDPs） often arise in planning problems where the agent is uncertain about its environmental state but can act and observe deterministically. 在这种情况下，我们提出了DetMCVI算法，它是Monte Carlo Value Iteration (MCVI)算法对DetPOMDPs的适应，用于构建有限状态控制器（FSCs）形式的策略。DetMCVI以高成功率解决了大规模问题，超过了现有DetPOMDP基准算法。我们还在一个实际的移动机器人森林制图场景中验证了该算法的性能。 

---
# ParkDiffusion: Heterogeneous Multi-Agent Multi-Modal Trajectory Prediction for Automated Parking using Diffusion Models 

**Title (ZH)**: ParkDiffusion：基于扩散模型的自动化停车中的异构多agent多模态轨迹预测 

**Authors**: Jiarong Wei, Niclas Vödisch, Anna Rehr, Christian Feist, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2505.00586)  

**Abstract**: Automated parking is a critical feature of Advanced Driver Assistance Systems (ADAS), where accurate trajectory prediction is essential to bridge perception and planning modules. Despite its significance, research in this domain remains relatively limited, with most existing studies concentrating on single-modal trajectory prediction of vehicles. In this work, we propose ParkDiffusion, a novel approach that predicts the trajectories of both vehicles and pedestrians in automated parking scenarios. ParkDiffusion employs diffusion models to capture the inherent uncertainty and multi-modality of future trajectories, incorporating several key innovations. First, we propose a dual map encoder that processes soft semantic cues and hard geometric constraints using a two-step cross-attention mechanism. Second, we introduce an adaptive agent type embedding module, which dynamically conditions the prediction process on the distinct characteristics of vehicles and pedestrians. Third, to ensure kinematic feasibility, our model outputs control signals that are subsequently used within a kinematic framework to generate physically feasible trajectories. We evaluate ParkDiffusion on the Dragon Lake Parking (DLP) dataset and the Intersections Drone (inD) dataset. Our work establishes a new baseline for heterogeneous trajectory prediction in parking scenarios, outperforming existing methods by a considerable margin. 

**Abstract (ZH)**: 自动化停车是高级驾驶辅助系统（ADAS）的关键功能，其中准确的轨迹预测对于连接感知模块和规划模块至关重要。尽管其重要性不言而喻，但该领域的研究仍相对有限，大多数现有研究集中在车辆单一模态轨迹预测上。本文提出ParkDiffusion，一种新颖的方法，用于预测自动化停车场景中车辆和行人的轨迹。ParkDiffusion利用扩散模型捕捉未来轨迹的内在不确定性与多模态性，并包含几个关键创新。首先，我们提出了一种双地图编码器，通过两步交叉注意力机制处理软语义线索和硬几何约束。其次，我们引入了一种自适应实体类型嵌入模块，动态地根据车辆和行人的不同特性条件预测过程。第三，为了确保动力学可行性，我们的模型输出的控制信号随后在动力学框架中用于生成物理上可行的轨迹。我们在Dragon Lake Parking (DLP) 数据集和Intersections Drone (inD) 数据集上评估了ParkDiffusion。本文为停车场景中的异构轨迹预测建立了新的基准，明显优于现有方法。 

---
# TeLoGraF: Temporal Logic Planning via Graph-encoded Flow Matching 

**Title (ZH)**: TeLoGraF: 基于图编码流匹配的时间逻辑规划 

**Authors**: Yue Meng, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00562)  

**Abstract**: Learning to solve complex tasks with signal temporal logic (STL) specifications is crucial to many real-world applications. However, most previous works only consider fixed or parametrized STL specifications due to the lack of a diverse STL dataset and encoders to effectively extract temporal logic information for downstream tasks. In this paper, we propose TeLoGraF, Temporal Logic Graph-encoded Flow, which utilizes Graph Neural Networks (GNN) encoder and flow-matching to learn solutions for general STL specifications. We identify four commonly used STL templates and collect a total of 200K specifications with paired demonstrations. We conduct extensive experiments in five simulation environments ranging from simple dynamical models in the 2D space to high-dimensional 7DoF Franka Panda robot arm and Ant quadruped navigation. Results show that our method outperforms other baselines in the STL satisfaction rate. Compared to classical STL planning algorithms, our approach is 10-100X faster in inference and can work on any system dynamics. Besides, we show our graph-encoding method's capability to solve complex STLs and robustness to out-distribution STL specifications. Code is available at this https URL 

**Abstract (ZH)**: 使用信号时序逻辑（STL）规范学习解决复杂任务：TeLoGraF，时序逻辑图编码流 

---
# DeCo: Task Decomposition and Skill Composition for Zero-Shot Generalization in Long-Horizon 3D Manipulation 

**Title (ZH)**: DeCo：长时 horizon 3D 操作中的零样本泛化的任务分解与技能组合 

**Authors**: Zixuan Chen, Junhui Yin, Yangtao Chen, Jing Huo, Pinzhuo Tian, Jieqi Shi, Yiwen Hou, Yinchuan Li, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.00527)  

**Abstract**: Generalizing language-conditioned multi-task imitation learning (IL) models to novel long-horizon 3D manipulation tasks remains a significant challenge. To address this, we propose DeCo (Task Decomposition and Skill Composition), a model-agnostic framework compatible with various multi-task IL models, designed to enhance their zero-shot generalization to novel, compositional, long-horizon 3D manipulation tasks. DeCo first decomposes IL demonstrations into a set of modular atomic tasks based on the physical interaction between the gripper and objects, and constructs an atomic training dataset that enables models to learn a diverse set of reusable atomic skills during imitation learning. At inference time, DeCo leverages a vision-language model (VLM) to parse high-level instructions for novel long-horizon tasks, retrieve the relevant atomic skills, and dynamically schedule their execution; a spatially-aware skill-chaining module then ensures smooth, collision-free transitions between sequential skills. We evaluate DeCo in simulation using DeCoBench, a benchmark specifically designed to assess zero-shot generalization of multi-task IL models in compositional long-horizon 3D manipulation. Across three representative multi-task IL models (RVT-2, 3DDA, and ARP), DeCo achieves success rate improvements of 66.67%, 21.53%, and 57.92%, respectively, on 12 novel compositional tasks. Moreover, in real-world experiments, a DeCo-enhanced model trained on only 6 atomic tasks successfully completes 9 novel long-horizon tasks, yielding an average success rate improvement of 53.33% over the base multi-task IL model. Video demonstrations are available at: this https URL. 

**Abstract (ZH)**: 面向新型长时 horizon 3D 操作任务的语言条件多任务模仿学习模型的通用化：一种任务分解与技能组合的方法 

---
# Safety-Critical Traffic Simulation with Guided Latent Diffusion Model 

**Title (ZH)**: 带有引导潜在扩散模型的安全临界交通仿真 

**Authors**: Mingxing Peng, Ruoyu Yao, Xusen Guo, Yuting Xie, Xianda Chen, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00515)  

**Abstract**: Safety-critical traffic simulation plays a crucial role in evaluating autonomous driving systems under rare and challenging scenarios. However, existing approaches often generate unrealistic scenarios due to insufficient consideration of physical plausibility and suffer from low generation efficiency. To address these limitations, we propose a guided latent diffusion model (LDM) capable of generating physically realistic and adversarial safety-critical traffic scenarios. Specifically, our model employs a graph-based variational autoencoder (VAE) to learn a compact latent space that captures complex multi-agent interactions while improving computational efficiency. Within this latent space, the diffusion model performs the denoising process to produce realistic trajectories. To enable controllable and adversarial scenario generation, we introduce novel guidance objectives that drive the diffusion process toward producing adversarial and behaviorally realistic driving behaviors. Furthermore, we develop a sample selection module based on physical feasibility checks to further enhance the physical plausibility of the generated scenarios. Extensive experiments on the nuScenes dataset demonstrate that our method achieves superior adversarial effectiveness and generation efficiency compared to existing baselines while maintaining a high level of realism. Our work provides an effective tool for realistic safety-critical scenario simulation, paving the way for more robust evaluation of autonomous driving systems. 

**Abstract (ZH)**: 基于引导的潜在扩散模型在生成现实且对抗性的安全关键交通场景中的应用 

---
# Implicit Neural-Representation Learning for Elastic Deformable-Object Manipulations 

**Title (ZH)**: 弹性变形物体操控的隐式神经表示学习 

**Authors**: Minseok Song, JeongHo Ha, Bonggyeong Park, Daehyung Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.00500)  

**Abstract**: We aim to solve the problem of manipulating deformable objects, particularly elastic bands, in real-world scenarios. However, deformable object manipulation (DOM) requires a policy that works on a large state space due to the unlimited degree of freedom (DoF) of deformable objects. Further, their dense but partial observations (e.g., images or point clouds) may increase the sampling complexity and uncertainty in policy learning. To figure it out, we propose a novel implicit neural-representation (INR) learning for elastic DOMs, called INR-DOM. Our method learns consistent state representations associated with partially observable elastic objects reconstructing a complete and implicit surface represented as a signed distance function. Furthermore, we perform exploratory representation fine-tuning through reinforcement learning (RL) that enables RL algorithms to effectively learn exploitable representations while efficiently obtaining a DOM policy. We perform quantitative and qualitative analyses building three simulated environments and real-world manipulation studies with a Franka Emika Panda arm. Videos are available at this http URL. 

**Abstract (ZH)**: 我们旨在解决在真实世界场景中操控变形物体，特别是弹性带子的问题。然而，变形物体操控（DOM）需要一个能够在由于变形物体无限自由度（DoF）而产生的大面积状态空间上工作的策略。此外，这些密集但部分的观察（例如，图像或点云）可能会增加采样复杂性和政策学习中的不确定性。为了解决这一问题，我们提出了一种新颖的隐式神经表示（INR）学习方法，称为INR-DOM。我们的方法学习与部分可观测的弹性物体相关的一致状态表示，重构一个由符号距离函数表示的完整且隐式的表面。此外，我们通过强化学习（RL）进行探究性表示微调，使RL算法能够有效地学习可利用的表示，同时高效地获得一个DOM策略。我们在三个模拟环境和使用Franka Emika Panda手臂的真实世界操控研究中进行了定量和定性的分析。视频可在以下链接获取。 

---
# Optimal Interactive Learning on the Job via Facility Location Planning 

**Title (ZH)**: 基于设施位置规划的最优在职交互学习 

**Authors**: Shivam Vats, Michelle Zhao, Patrick Callaghan, Mingxi Jia, Maxim Likhachev, Oliver Kroemer, George Konidaris  

**Link**: [PDF](https://arxiv.org/pdf/2505.00490)  

**Abstract**: Collaborative robots must continually adapt to novel tasks and user preferences without overburdening the user. While prior interactive robot learning methods aim to reduce human effort, they are typically limited to single-task scenarios and are not well-suited for sustained, multi-task collaboration. We propose COIL (Cost-Optimal Interactive Learning) -- a multi-task interaction planner that minimizes human effort across a sequence of tasks by strategically selecting among three query types (skill, preference, and help). When user preferences are known, we formulate COIL as an uncapacitated facility location (UFL) problem, which enables bounded-suboptimal planning in polynomial time using off-the-shelf approximation algorithms. We extend our formulation to handle uncertainty in user preferences by incorporating one-step belief space planning, which uses these approximation algorithms as subroutines to maintain polynomial-time performance. Simulated and physical experiments on manipulation tasks show that our framework significantly reduces the amount of work allocated to the human while maintaining successful task completion. 

**Abstract (ZH)**: 协作机器人必须在不断适应新任务和用户偏好的同时，不给用户带来过重负担。尽管以前的交互式机器人学习方法旨在减少人类努力，但它们通常局限于单一任务场景，不适用于持续的多任务协作。我们提出了COIL（成本最优交互式学习）——一种通过战略性选择三种查询类型（技能、偏好和帮助）来最小化序列任务中人类努力的多任务交互规划器。当用户偏好已知时，我们将COIL形式化为无容量设施位置（UFL）问题，这使得我们能够使用现成的近似算法在多项式时间内进行有界次优规划。通过引入一步信念空间规划，我们将形式化扩展以处理用户偏好不确定性，这通过将这些近似算法作为子程序来保持多项式时间性能。针对操作任务的仿真实验和物理实验表明，我们的框架显著减少了分配给人类的工作量，同时保持了任务的成功完成。 

---
# MULE: Multi-terrain and Unknown Load Adaptation for Effective Quadrupedal Locomotion 

**Title (ZH)**: MULE: 多地形和未知负载适应性四足运动控制 

**Authors**: Vamshi Kumar Kurva, Shishir Kolathaya  

**Link**: [PDF](https://arxiv.org/pdf/2505.00488)  

**Abstract**: Quadrupedal robots are increasingly deployed for load-carrying tasks across diverse terrains. While Model Predictive Control (MPC)-based methods can account for payload variations, they often depend on predefined gait schedules or trajectory generators, limiting their adaptability in unstructured environments. To address these limitations, we propose an Adaptive Reinforcement Learning (RL) framework that enables quadrupedal robots to dynamically adapt to both varying payloads and diverse terrains. The framework consists of a nominal policy responsible for baseline locomotion and an adaptive policy that learns corrective actions to preserve stability and improve command tracking under payload variations. We validate the proposed approach through large-scale simulation experiments in Isaac Gym and real-world hardware deployment on a Unitree Go1 quadruped. The controller was tested on flat ground, slopes, and stairs under both static and dynamic payload changes. Across all settings, our adaptive controller consistently outperformed the controller in tracking body height and velocity commands, demonstrating enhanced robustness and adaptability without requiring explicit gait design or manual tuning. 

**Abstract (ZH)**: 基于强化学习的四足机器人适应性控制系统：应对载荷变化和复杂地形的动态适应性 

---
# Decentralised, Self-Organising Drone Swarms using Coupled Oscillators 

**Title (ZH)**: 基于耦合振子的去中心化自组织无人机群 

**Authors**: Kevin Quinn, Cormac Molloy, Harun Šiljak  

**Link**: [PDF](https://arxiv.org/pdf/2505.00442)  

**Abstract**: The problem of robotic synchronisation and coordination is a long-standing one. Combining autonomous, computerised systems with unpredictable real-world conditions can have consequences ranging from poor performance to collisions and damage. This paper proposes using coupled oscillators to create a drone swarm that is decentralised and self organising. This allows for greater flexibility and adaptiveness than a hard-coded swarm, with more resilience and scalability than a centralised system. Our method allows for a variable number of drones to spontaneously form a swarm and react to changing swarm conditions. Additionally, this method includes provisions to prevent communication interference between drones, and signal processing techniques to ensure a smooth and cohesive swarm. 

**Abstract (ZH)**: 机器人同步与协调问题是长期存在的一个问题。将自主计算机系统与不可预测的现实世界条件相结合可能会导致从性能不佳到碰撞和损坏等各种后果。本文提出使用耦合振子来创建去中心化和自组织的无人机群。这种方法提供了比硬编码群无人机更大的灵活性和适应性，同时具备中心化系统所不具备的更强的鲁棒性和扩展性。我们的方法允许无人机自发地形成群组并适应不断变化的群组条件。此外，该方法还包括防止无人机间通信干扰的机制，并运用信号处理技术确保群组的顺畅和一致性。 

---
# A Neural Network Mode for PX4 on Embedded Flight Controllers 

**Title (ZH)**: 基于嵌入式飞行控制器的PX4神经网络模型 

**Authors**: Sindre M. Hegre, Welf Rehberg, Mihir Kulkarni, Kostas Alexis  

**Link**: [PDF](https://arxiv.org/pdf/2505.00432)  

**Abstract**: This paper contributes an open-sourced implementation of a neural-network based controller framework within the PX4 stack. We develop a custom module for inference on the microcontroller while retaining all of the functionality of the PX4 autopilot. Policies trained in the Aerial Gym Simulator are converted to the TensorFlow Lite format and then built together with PX4 and flashed to the flight controller. The policies substitute the control-cascade within PX4 to offer an end-to-end position-setpoint tracking controller directly providing normalized motor RPM setpoints. Experiments conducted in simulation and the real-world show similar tracking performance. We thus provide a flight-ready pipeline for testing neural control policies in the real world. The pipeline simplifies the deployment of neural networks on embedded flight controller hardware thereby accelerating research on learning-based control. Both the Aerial Gym Simulator and the PX4 module are open-sourced at this https URL and this https URL. Video: this https URL. 

**Abstract (ZH)**: 本文贡献了一个基于神经网络的控制器框架的开源实现，集成在PX4堆栈中。我们开发了一个自定义模块，在微控制器上进行推理，同时保留PX4自动驾驶的所有功能。在Aerial Gym模拟器中训练的策略被转换为TensorFlow Lite格式，并与PX4一起构建并写入飞行控制器。这些策略取代了PX4中的控制级联，提供了一个端到端的位置设定点跟踪控制器，直接提供标准化的电机转速设定点。在模拟和真实世界中的实验显示了类似的跟踪性能。因此，本文提供了一个适用于真实世界的飞行就绪管道，用于测试基于学习的控制策略。该管道简化了神经网络在嵌入式飞行控制器硬件上的部署，从而加速了基于学习的控制研究。Aerial Gym模拟器和PX4模块在此开源：[此链接](此 https URL) 和 [此链接](此 https URL)。视频：[此链接](此 https URL)。 

---
# iMacSR: Intermediate Multi-Access Supervision and Regularization in Training Autonomous Driving Models 

**Title (ZH)**: iMacSR：训练自动驾驶模型的中间多访问监督与正则化 

**Authors**: Wei-Bin Kou, Guangxu Zhu, Yichen Jin, Shuai Wang, Ming Tang, Yik-Chung Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00404)  

**Abstract**: Deep Learning (DL)-based street scene semantic understanding has become a cornerstone of autonomous driving (AD). DL model performance heavily relies on network depth. Specifically, deeper DL architectures yield better segmentation performance. However, as models grow deeper, traditional one-point supervision at the final layer struggles to optimize intermediate feature representations, leading to subpar training outcomes. To address this, we propose an intermediate Multi-access Supervision and Regularization (iMacSR) strategy. The proposed iMacSR introduces two novel components: (I) mutual information between latent features and ground truth as intermediate supervision loss ensures robust feature alignment at multiple network depths; and (II) negative entropy regularization on hidden features discourages overconfident predictions and mitigates overfitting. These intermediate terms are combined into the original final-layer training loss to form a unified optimization objective, enabling comprehensive optimization across the network hierarchy. The proposed iMacSR provides a robust framework for training deep AD architectures, advancing the performance of perception systems in real-world driving scenarios. In addition, we conduct theoretical convergence analysis for the proposed iMacSR. Extensive experiments on AD benchmarks (i.e., Cityscapes, CamVid, and SynthiaSF datasets) demonstrate that iMacSR outperforms conventional final-layer single-point supervision method up to 9.19% in mean Intersection over Union (mIoU). 

**Abstract (ZH)**: 基于深度学习的街道场景语义理解已成为自动驾驶的核心基石。提出的iMacSR策略通过引入中间多访问监督和正则化，提升了网络性能。 

---
# Holistic Optimization of Modular Robots 

**Title (ZH)**: 模块化机器人整体优化 

**Authors**: Matthias Mayer, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2505.00400)  

**Abstract**: Modular robots have the potential to revolutionize automation as one can optimize their composition for any given task. However, finding optimal compositions is non-trivial. In addition, different compositions require different base positions and trajectories to fully use the potential of modular robots. We address this problem holistically for the first time by jointly optimizing the composition, base placement, and trajectory, to minimize the cycle time of a given task. Our approach is evaluated on over 300 industrial benchmarks requiring point-to-point movements. Overall, we reduce cycle time by up to 25% and find feasible solutions in twice as many benchmarks compared to optimizing the module composition alone. In the first real-world validation of modular robots optimized for point-to-point movement, we find that the optimized robot is successfully deployed in nine out of ten cases in less than an hour. 

**Abstract (ZH)**: 模块化机器人有潜力通过针对任何给定任务优化其组成来革新自动化。然而，找到最优组成并非易事。此外，不同的组成需要不同的基座位置和轨迹以充分挖掘模块化机器人的潜力。我们首次从整体上解决了这一问题，通过联合优化组成、基座放置和轨迹来最小化给定任务的周期时间。我们的方法在超过300个需要点到点运动的工业基准上进行了评估，总体上减少了25%的周期时间，并在两倍数量的基准上找到了可行的解决方案，与仅优化模块组成相比。在全球首个针对点到点移动优化的模块化机器人实际验证中，我们发现优化后的机器人在不到一小时内成功部署在十起案例中的九起。 

---
# Multi-segment Soft Robot Control via Deep Koopman-based Model Predictive Control 

**Title (ZH)**: 基于深度Koopman的模型预测控制的多段软 Robotics 控制 

**Authors**: Lei Lv, Lei Liu, Lei Bao, Fuchun Sun, Jiahong Dong, Jianwei Zhang, Xuemei Shan, Kai Sun, Hao Huang, Yu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00354)  

**Abstract**: Soft robots, compared to regular rigid robots, as their multiple segments with soft materials bring flexibility and compliance, have the advantages of safe interaction and dexterous operation in the environment. However, due to its characteristics of high dimensional, nonlinearity, time-varying nature, and infinite degree of freedom, it has been challenges in achieving precise and dynamic control such as trajectory tracking and position reaching. To address these challenges, we propose a framework of Deep Koopman-based Model Predictive Control (DK-MPC) for handling multi-segment soft robots. We first employ a deep learning approach with sampling data to approximate the Koopman operator, which therefore linearizes the high-dimensional nonlinear dynamics of the soft robots into a finite-dimensional linear representation. Secondly, this linearized model is utilized within a model predictive control framework to compute optimal control inputs that minimize the tracking error between the desired and actual state trajectories. The real-world experiments on the soft robot "Chordata" demonstrate that DK-MPC could achieve high-precision control, showing the potential of DK-MPC for future applications to soft robots. 

**Abstract (ZH)**: 基于Deep Koopman的模型预测控制框架在处理多段软体机器人中的应用 

---
# Active Contact Engagement for Aerial Navigation in Unknown Environments with Glass 

**Title (ZH)**: 在未知环境中的无人机导航中基于玻璃的主动接触参与 

**Authors**: Xinyi Chen, Yichen Zhang, Hetai Zou, Junzhe Wang, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00332)  

**Abstract**: Autonomous aerial robots are increasingly being deployed in real-world scenarios, where transparent glass obstacles present significant challenges to reliable navigation. Researchers have investigated the use of non-contact sensors and passive contact-resilient aerial vehicle designs to detect glass surfaces, which are often limited in terms of robustness and efficiency. In this work, we propose a novel approach for robust autonomous aerial navigation in unknown environments with transparent glass obstacles, combining the strengths of both sensor-based and contact-based glass detection. The proposed system begins with the incremental detection and information maintenance about potential glass surfaces using visual sensor measurements. The vehicle then actively engages in touch actions with the visually detected potential glass surfaces using a pair of lightweight contact-sensing modules to confirm or invalidate their presence. Following this, the volumetric map is efficiently updated with the glass surface information and safe trajectories are replanned on the fly to circumvent the glass obstacles. We validate the proposed system through real-world experiments in various scenarios, demonstrating its effectiveness in enabling efficient and robust autonomous aerial navigation in complex real-world environments with glass obstacles. 

**Abstract (ZH)**: 自主自主飞行机器人在含有透明玻璃障碍的未知环境中的稳健导航方法：结合基于传感器和基于接触的玻璃检测优势 

---
# AI2-Active Safety: AI-enabled Interaction-aware Active Safety Analysis with Vehicle Dynamics 

**Title (ZH)**: 基于车辆动力学的AI增强交互感知主动安全分析 

**Authors**: Keshu Wu, Zihao Li, Sixu Li, Xinyue Ye, Dominique Lord, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.00322)  

**Abstract**: This paper introduces an AI-enabled, interaction-aware active safety analysis framework that accounts for groupwise vehicle interactions. Specifically, the framework employs a bicycle model-augmented with road gradient considerations-to accurately capture vehicle dynamics. In parallel, a hypergraph-based AI model is developed to predict probabilistic trajectories of ambient traffic. By integrating these two components, the framework derives vehicle intra-spacing over a 3D road surface as the solution of a stochastic ordinary differential equation, yielding high-fidelity surrogate safety measures such as time-to-collision (TTC). To demonstrate its effectiveness, the framework is analyzed using stochastic numerical methods comprising 4th-order Runge-Kutta integration and AI inference, generating probability-weighted high-fidelity TTC (HF-TTC) distributions that reflect complex multi-agent maneuvers and behavioral uncertainties. Evaluated with HF-TTC against traditional constant-velocity TTC and non-interaction-aware approaches on highway datasets, the proposed framework offers a systematic methodology for active safety analysis with enhanced potential for improving safety perception in complex traffic environments. 

**Abstract (ZH)**: 基于AI和交互感知的群体车辆互动主动安全分析框架 

---
# FedEMA: Federated Exponential Moving Averaging with Negative Entropy Regularizer in Autonomous Driving 

**Title (ZH)**: FedEMA：自主驾驶中的负熵正则化指数移动平均 federated learning 方法 

**Authors**: Wei-Bin Kou, Guangxu Zhu, Bingyang Cheng, Shuai Wang, Ming Tang, Yik-Chung Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00318)  

**Abstract**: Street Scene Semantic Understanding (denoted as S3U) is a crucial but complex task for autonomous driving (AD) vehicles. Their inference models typically face poor generalization due to domain-shift. Federated Learning (FL) has emerged as a promising paradigm for enhancing the generalization of AD models through privacy-preserving distributed learning. However, these FL AD models face significant temporal catastrophic forgetting when deployed in dynamically evolving environments, where continuous adaptation causes abrupt erosion of historical knowledge. This paper proposes Federated Exponential Moving Average (FedEMA), a novel framework that addresses this challenge through two integral innovations: (I) Server-side model's historical fitting capability preservation via fusing current FL round's aggregation model and a proposed previous FL round's exponential moving average (EMA) model; (II) Vehicle-side negative entropy regularization to prevent FL models' possible overfitting to EMA-introduced temporal patterns. Above two strategies empower FedEMA a dual-objective optimization that balances model generalization and adaptability. In addition, we conduct theoretical convergence analysis for the proposed FedEMA. Extensive experiments both on Cityscapes dataset and Camvid dataset demonstrate FedEMA's superiority over existing approaches, showing 7.12% higher mean Intersection-over-Union (mIoU). 

**Abstract (ZH)**: 基于街道场景语义理解的联邦指数移动平均 Federated Exponential Moving Average for Street Scene Semantic Understanding 

---
# J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively in Inverse Kinematic Control of Serial Manipulators 

**Title (ZH)**: 基于雅可比矩阵投影算法的 serial 连续机器人逆运动学控制奇异态有效.resolve 

**Authors**: Shivani Guptasarma, Matthew Strong, Honghao Zhen, Monroe Kennedy III  

**Link**: [PDF](https://arxiv.org/pdf/2505.00306)  

**Abstract**: J-PARSE is a method for smooth first-order inverse kinematic control of a serial manipulator near kinematic singularities. The commanded end-effector velocity is interpreted component-wise, according to the available mobility in each dimension of the task space. First, a substitute "Safety" Jacobian matrix is created, keeping the aspect ratio of the manipulability ellipsoid above a threshold value. The desired motion is then projected onto non-singular and singular directions, and the latter projection scaled down by a factor informed by the threshold value. A right-inverse of the non-singular Safety Jacobian is applied to the modified command. In the absence of joint limits and collisions, this ensures smooth transition into and out of low-rank poses, guaranteeing asymptotic stability for target poses within the workspace, and stability for those outside. Velocity control with J-PARSE is benchmarked against the Least-Squares and Damped Least-Squares inversions of the Jacobian, and shows high accuracy in reaching and leaving singular target poses. By expanding the available workspace of manipulators, the method finds applications in servoing, teleoperation, and learning. 

**Abstract (ZH)**: J-PARSE是一种在机械 singularity 附近平滑实现第一阶逆运动学控制的方法。根据任务空间中每个维度的可用运动性逐分量解释命令的末端执行器速度。首先，创建一个“安全”雅可比矩阵的替代矩阵，保持操作可行椭球的长宽比高于阈值。然后将期望运动投影到非奇异和奇异方向上，并将后者投影按阈值因子缩放。非奇异“安全”雅可比式的右逆被应用于修改后的命令。在没有关节限位和碰撞的情况下，这确保了平稳过渡到和离开低秩姿态，对于工作空间内的目标姿态保证渐近稳定性，对于工作空间外的目标姿态保证稳定性。使用 J-PARSE 进行速度控制与雅可比的最小二乘和阻尼最小二乘逆进行了基准测试，显示出在接近奇异姿态时高精度的到达和离开能力。通过扩展机器人的工作空间，该方法在伺服、遥操作和学习等领域中有应用。 

---
# LightEMMA: Lightweight End-to-End Multimodal Model for Autonomous Driving 

**Title (ZH)**: LightEMMA：轻量级端到端多模态自主驾驶模型 

**Authors**: Zhijie Qiao, Haowei Li, Zhong Cao, Henry X. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00284)  

**Abstract**: Vision-Language Models (VLMs) have demonstrated significant potential for end-to-end autonomous driving. However, fully exploiting their capabilities for safe and reliable vehicle control remains an open research challenge. To systematically examine advances and limitations of VLMs in driving tasks, we introduce LightEMMA, a Lightweight End-to-End Multimodal Model for Autonomous driving. LightEMMA provides a unified, VLM-based autonomous driving framework without ad hoc customizations, enabling easy integration and evaluation of evolving state-of-the-art commercial and open-source models. We construct twelve autonomous driving agents using various VLMs and evaluate their performance on the nuScenes prediction task, comprehensively assessing metrics such as inference time, computational cost, and predictive accuracy. Illustrative examples highlight that, despite their strong scenario interpretation capabilities, VLMs' practical performance in autonomous driving tasks remains concerning, emphasizing the need for further improvements. The code is available at this https URL. 

**Abstract (ZH)**: 基于视觉-语言模型的轻量化端到端多模态自动驾驶模型LightEMMA 

---
# Future-Oriented Navigation: Dynamic Obstacle Avoidance with One-Shot Energy-Based Multimodal Motion Prediction 

**Title (ZH)**: 面向未来的导航：基于单次能量化多模态运动预测的一次性动态障碍物规避 

**Authors**: Ze Zhang, Georg Hess, Junjie Hu, Emmanuel Dean, Lennart Svensson, Knut Åkesson  

**Link**: [PDF](https://arxiv.org/pdf/2505.00237)  

**Abstract**: This paper proposes an integrated approach for the safe and efficient control of mobile robots in dynamic and uncertain environments. The approach consists of two key steps: one-shot multimodal motion prediction to anticipate motions of dynamic obstacles and model predictive control to incorporate these predictions into the motion planning process. Motion prediction is driven by an energy-based neural network that generates high-resolution, multi-step predictions in a single operation. The prediction outcomes are further utilized to create geometric shapes formulated as mathematical constraints. Instead of treating each dynamic obstacle individually, predicted obstacles are grouped by proximity in an unsupervised way to improve performance and efficiency. The overall collision-free navigation is handled by model predictive control with a specific design for proactive dynamic obstacle avoidance. The proposed approach allows mobile robots to navigate effectively in dynamic environments. Its performance is accessed across various scenarios that represent typical warehouse settings. The results demonstrate that the proposed approach outperforms other existing dynamic obstacle avoidance methods. 

**Abstract (ZH)**: 本文提出了一种集成方法，用于在动态和不确定环境中安全高效地控制移动机器人。该方法包括两个关键步骤：一次多模态运动预测以预见动态障碍物的运动，以及模型预测控制以将这些预测纳入运动规划过程中。运动预测由基于能量的神经网络驱动，能够在单次操作中生成高分辨率的多步预测。预测结果进一步用于创建作为数学约束的几何形状。预测的障碍物不是单独处理，而是通过无监督的方式根据邻近度进行分组，以提高性能和效率。整体无碰撞导航由专门设计的模型预测控制处理，以实现主动动态障碍物避让。所提出的方法使移动机器人能够在动态环境中有效地导航。其性能在代表典型仓库设置的各种场景中进行了评估。结果表明，所提出的方法优于其他现有的动态障碍物避让方法。 

---
# AI-Enhanced Automatic Design of Efficient Underwater Gliders 

**Title (ZH)**: AI增强的高效水下航行器自动设计方法 

**Authors**: Peter Yichen Chen, Pingchuan Ma, Niklas Hagemann, John Romanishin, Wei Wang, Daniela Rus, Wojciech Matusik  

**Link**: [PDF](https://arxiv.org/pdf/2505.00222)  

**Abstract**: The development of novel autonomous underwater gliders has been hindered by limited shape diversity, primarily due to the reliance on traditional design tools that depend heavily on manual trial and error. Building an automated design framework is challenging due to the complexities of representing glider shapes and the high computational costs associated with modeling complex solid-fluid interactions. In this work, we introduce an AI-enhanced automated computational framework designed to overcome these limitations by enabling the creation of underwater robots with non-trivial hull shapes. Our approach involves an algorithm that co-optimizes both shape and control signals, utilizing a reduced-order geometry representation and a differentiable neural-network-based fluid surrogate model. This end-to-end design workflow facilitates rapid iteration and evaluation of hydrodynamic performance, leading to the discovery of optimal and complex hull shapes across various control settings. We validate our method through wind tunnel experiments and swimming pool gliding tests, demonstrating that our computationally designed gliders surpass manually designed counterparts in terms of energy efficiency. By addressing challenges in efficient shape representation and neural fluid surrogate models, our work paves the way for the development of highly efficient underwater gliders, with implications for long-range ocean exploration and environmental monitoring. 

**Abstract (ZH)**: 基于AI增强的自动化计算框架在构建新型自主水下航行器中的应用：克服形状多样性限制并提高能效 

---
# PSN Game: Game-theoretic Planning via a Player Selection Network 

**Title (ZH)**: PSN游戏：基于玩家选择网络的游戏理论规划 

**Authors**: Tianyu Qiu, Eric Ouano, Fernando Palafox, Christian Ellis, David Fridovich-Keil  

**Link**: [PDF](https://arxiv.org/pdf/2505.00213)  

**Abstract**: While game-theoretic planning frameworks are effective at modeling multi-agent interactions, they require solving optimization problems with hundreds or thousands of variables, resulting in long computation times that limit their use in large-scale, real-time systems. To address this issue, we propose PSN Game: a novel game-theoretic planning framework that reduces runtime by learning a Player Selection Network (PSN). A PSN outputs a player selection mask that distinguishes influential players from less relevant ones, enabling the ego player to solve a smaller, masked game involving only selected players. By reducing the number of variables in the optimization problem, PSN directly lowers computation time. The PSN Game framework is more flexible than existing player selection methods as it i) relies solely on observations of players' past trajectories, without requiring full state, control, or other game-specific information; and ii) requires no online parameter tuning. We train PSNs in an unsupervised manner using a differentiable dynamic game solver, with reference trajectories from full-player games guiding the learning. Experiments in both simulated scenarios and human trajectory datasets demonstrate that i) PSNs outperform baseline selection methods in trajectory smoothness and length, while maintaining comparable safety and achieving a 10x speedup in runtime; and ii) PSNs generalize effectively to real-world scenarios without fine-tuning. By selecting only the most relevant players for decision-making, PSNs offer a general mechanism for reducing planning complexity that can be seamlessly integrated into existing multi-agent planning frameworks. 

**Abstract (ZH)**: PSN Game:一种通过学习玩家选择网络减少运行时的游戏理论规划框架 

---
# Investigating Adaptive Tuning of Assistive Exoskeletons Using Offline Reinforcement Learning: Challenges and Insights 

**Title (ZH)**: 基于离线强化学习的辅助外骨骼自适应调谐研究：挑战与见解 

**Authors**: Yasin Findik, Christopher Coco, Reza Azadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.00201)  

**Abstract**: Assistive exoskeletons have shown great potential in enhancing mobility for individuals with motor impairments, yet their effectiveness relies on precise parameter tuning for personalized assistance. In this study, we investigate the potential of offline reinforcement learning for optimizing effort thresholds in upper-limb assistive exoskeletons, aiming to reduce reliance on manual calibration. Specifically, we frame the problem as a multi-agent system where separate agents optimize biceps and triceps effort thresholds, enabling a more adaptive and data-driven approach to exoskeleton control. Mixed Q-Functionals (MQF) is employed to efficiently handle continuous action spaces while leveraging pre-collected data, thereby mitigating the risks associated with real-time exploration. Experiments were conducted using the MyoPro 2 exoskeleton across two distinct tasks involving horizontal and vertical arm movements. Our results indicate that the proposed approach can dynamically adjust threshold values based on learned patterns, potentially improving user interaction and control, though performance evaluation remains challenging due to dataset limitations. 

**Abstract (ZH)**: 基于离线强化学习优化上肢辅助外骨骼的努力阈值：减少手动校准依赖性 

---
# Characterizing gaussian mixture of motion modes for skid-steer state estimation 

**Title (ZH)**: 基于滑移转向运动模态高斯混合的状态估计Characterizing Gaussian Mixture of Motion Modes for Skid-steer State Estimation 

**Authors**: Ameya Salvi, Mark Brudnak, Jonathon M. Smereka, Matthias Schmid, Venkat Krovi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00200)  

**Abstract**: Skid-steered wheel mobile robots (SSWMRs) are characterized by the unique domination of the tire-terrain skidding for the robot to move. The lack of reliable friction models cascade into unreliable motion models, especially the reduced ordered variants used for state estimation and robot control. Ensemble modeling is an emerging research direction where the overall motion model is broken down into a family of local models to distribute the performance and resource requirement and provide a fast real-time prediction. To this end, a gaussian mixture model based modeling identification of model clusters is adopted and implemented within an interactive multiple model (IMM) based state estimation. The framework is adopted and implemented for angular velocity as the estimated state for a mid scaled skid-steered wheel mobile robot platform. 

**Abstract (ZH)**: 基于轮胎-地形滑行的独特优势的滑移轮式移动机器人（SSWMRs）缺乏可靠的摩擦模型导致其运动模型不可靠，特别是在用于状态估计和机器人控制的减少有序变体中。集成建模是一种新兴的研究方向，通过将整体运动模型分解为一系列局部模型来分散性能和资源需求，并提供快速实时预测。为此，采用基于 Gaussian 混合模型的模型簇建模识别方法，并在基于交互式多模型（IMM）的状态估计中实现。该框架应用于一个中等规模的滑移轮式移动机器人平台，以角速度作为估计状态。 

---
# Deep Reinforcement Learning Policies for Underactuated Satellite Attitude Control 

**Title (ZH)**: 未 acted satelliteattitude control的深度强化学习策略 

**Authors**: Matteo El Hariry, Andrea Cini, Giacomo Mellone, Alessandro Balossino  

**Link**: [PDF](https://arxiv.org/pdf/2505.00165)  

**Abstract**: Autonomy is a key challenge for future space exploration endeavours. Deep Reinforcement Learning holds the promises for developing agents able to learn complex behaviours simply by interacting with their environment. This paper investigates the use of Reinforcement Learning for the satellite attitude control problem, namely the angular reorientation of a spacecraft with respect to an in- ertial frame of reference. In the proposed approach, a set of control policies are implemented as neural networks trained with a custom version of the Proximal Policy Optimization algorithm to maneuver a small satellite from a random starting angle to a given pointing target. In particular, we address the problem for two working conditions: the nominal case, in which all the actuators (a set of 3 reac- tion wheels) are working properly, and the underactuated case, where an actuator failure is simulated randomly along with one of the axes. We show that the agents learn to effectively perform large-angle slew maneuvers with fast convergence and industry-standard pointing accuracy. Furthermore, we test the proposed method on representative hardware, showing that by taking adequate measures controllers trained in simulation can perform well in real systems. 

**Abstract (ZH)**: 自主性是未来空间探索任务的关键挑战。深度强化学习为开发仅通过与其环境交互即可学习复杂行为的智能体带来了希望。本文探讨了使用强化学习解决卫星姿态控制问题，即航天器相对于惯性参考系的角再定向。在所提出的方案中，控制策略被实现为通过针对改良的渐进策略优化算法训练的神经网络来操作小型卫星，使其从随机初始角度转向给定的指向目标。特别地，我们针对以下两种工作条件解决了该问题：正常情况，所有执行器（一组3个反应轮）正常工作；欠驱动情况，其中随机模拟其中一个轴上的执行器故障。我们展示了智能体能够有效执行大角度指向机动，具有快速收敛性和行业标准的指向精度。此外，我们在代表性的硬件上测试了所提出的方法，表明通过采取适当措施，模拟中训练的控制器可以在实际系统中表现良好。 

---
# Optimized Lattice-Structured Flexible EIT Sensor for Tactile Reconstruction and Classification 

**Title (ZH)**: 优化 lattice 结构的柔性电容成像传感器及其触觉重建与分类 

**Authors**: Huazhi Dong, Sihao Teng, Xu Han, Xiaopeng Wu, Francesco Giorgio-Serchi, Yunjie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00161)  

**Abstract**: Flexible electrical impedance tomography (EIT) offers a promising alternative to traditional tactile sensing approaches, enabling low-cost, scalable, and deformable sensor designs. Here, we propose an optimized lattice-structured flexible EIT tactile sensor incorporating a hydrogel-based conductive layer, systematically designed through three-dimensional coupling field simulations to optimize structural parameters for enhanced sensitivity and robustness. By tuning the lattice channel width and conductive layer thickness, we achieve significant improvements in tactile reconstruction quality and classification performance. Experimental results demonstrate high-quality tactile reconstruction with correlation coefficients up to 0.9275, peak signal-to-noise ratios reaching 29.0303 dB, and structural similarity indexes up to 0.9660, while maintaining low relative errors down to 0.3798. Furthermore, the optimized sensor accurately classifies 12 distinct tactile stimuli with an accuracy reaching 99.6%. These results highlight the potential of simulation-guided structural optimization for advancing flexible EIT-based tactile sensors toward practical applications in wearable systems, robotics, and human-machine interfaces. 

**Abstract (ZH)**: 仿生水凝胶基层状结构优化的柔性电Sense电阻抗成像（EIT）触觉传感器 

---
# CoordField: Coordination Field for Agentic UAV Task Allocation In Low-altitude Urban Scenarios 

**Title (ZH)**: CoordField: 代理 UAV 任务分配中的协调场在低空城市场景中 

**Authors**: Tengchao Zhang, Yonglin Tian, Fei Lin, Jun Huang, Rui Qin, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00091)  

**Abstract**: With the increasing demand for heterogeneous Unmanned Aerial Vehicle (UAV) swarms to perform complex tasks in urban environments, system design now faces major challenges, including efficient semantic understanding, flexible task planning, and the ability to dynamically adjust coordination strategies in response to evolving environmental conditions and continuously changing task requirements. To address the limitations of existing approaches, this paper proposes coordination field agentic system for coordinating heterogeneous UAV swarms in complex urban scenarios. In this system, large language models (LLMs) is responsible for interpreting high-level human instructions and converting them into executable commands for the UAV swarms, such as patrol and target tracking. Subsequently, a Coordination field mechanism is proposed to guide UAV motion and task selection, enabling decentralized and adaptive allocation of emergent tasks. A total of 50 rounds of comparative testing were conducted across different models in a 2D simulation space to evaluate their performance. Experimental results demonstrate that the proposed system achieves superior performance in terms of task coverage, response time, and adaptability to dynamic changes. 

**Abstract (ZH)**: 基于大语言模型的协调场代理系统：应对复杂城市环境中的异构无人机集群协同挑战 

---
# Towards Autonomous Micromobility through Scalable Urban Simulation 

**Title (ZH)**: 面向自主微出行的大规模城市仿真 

**Authors**: Wayne Wu, Honglin He, Chaoyuan Zhang, Jack He, Seth Z. Zhao, Ran Gong, Quanyi Li, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.00690)  

**Abstract**: Micromobility, which utilizes lightweight mobile machines moving in urban public spaces, such as delivery robots and mobility scooters, emerges as a promising alternative to vehicular mobility. Current micromobility depends mostly on human manual operation (in-person or remote control), which raises safety and efficiency concerns when navigating busy urban environments full of unpredictable obstacles and pedestrians. Assisting humans with AI agents in maneuvering micromobility devices presents a viable solution for enhancing safety and efficiency. In this work, we present a scalable urban simulation solution to advance autonomous micromobility. First, we build URBAN-SIM - a high-performance robot learning platform for large-scale training of embodied agents in interactive urban scenes. URBAN-SIM contains three critical modules: Hierarchical Urban Generation pipeline, Interactive Dynamics Generation strategy, and Asynchronous Scene Sampling scheme, to improve the diversity, realism, and efficiency of robot learning in simulation. Then, we propose URBAN-BENCH - a suite of essential tasks and benchmarks to gauge various capabilities of the AI agents in achieving autonomous micromobility. URBAN-BENCH includes eight tasks based on three core skills of the agents: Urban Locomotion, Urban Navigation, and Urban Traverse. We evaluate four robots with heterogeneous embodiments, such as the wheeled and legged robots, across these tasks. Experiments on diverse terrains and urban structures reveal each robot's strengths and limitations. 

**Abstract (ZH)**: 轻型机动性：利用轻量级移动机器在城市公共空间中运行的新型机动性方式，如配送机器人和电动滑板车，作为车辆机动性的有前景替代方案出现。当前的轻型机动性主要依赖人工操作（现场或远程控制），在充满不可预测障碍物和行人的繁忙城市环境中 navegation 时，存在安全性和效率方面的担忧。通过人工智能代理辅助人类操控轻型机动性设备为提高安全性和效率提供了一种可行的解决方案。在此项工作中，我们提出了一种可扩展的城市仿真解决方案，以推动自主轻型机动性的发展。首先，我们构建了 URBAN-SIM - 一个高性能的机器人学习平台，用于大规模训练嵌入式代理在互动城市场景中的能力。URBAN-SIM 包含三个关键模块：层次化城市生成管道、互动动力学生成策略以及异步场景采样方案，以提高仿真实验中机器人学习的多样 性、真实性和效率。然后，我们提出 URBAN-BENCH - 一组基本任务和基准测试，用于衡量 AI 代理在实现自主轻型机动性方面的能力。URBAN-BENCH 包括基于代理三大核心能力的八个任务：城市运动、城市导航和城市穿越。我们评估了四种具有不同身体形态的机器人，例如轮式和腿式机器人，在这些任务中的表现。在各种地形和城市结构上的实验揭示了每种机器人的优势和局限性。 

---
# Emergence of Roles in Robotic Teams with Model Sharing and Limited Communication 

**Title (ZH)**: 具有模型共享和有限通信的机器人团队中的角色 emergence 在机器人团队中的角色分化：基于模型共享与有限通信 

**Authors**: Ian O'Flynn, Harun Šiljak  

**Link**: [PDF](https://arxiv.org/pdf/2505.00540)  

**Abstract**: We present a reinforcement learning strategy for use in multi-agent foraging systems in which the learning is centralised to a single agent and its model is periodically disseminated among the population of non-learning agents. In a domain where multi-agent reinforcement learning (MARL) is the common approach, this approach aims to significantly reduce the computational and energy demands compared to approaches such as MARL and centralised learning models. By developing high performing foraging agents, these approaches can be translated into real-world applications such as logistics, environmental monitoring, and autonomous exploration. A reward function was incorporated into this approach that promotes role development among agents, without explicit directives. This led to the differentiation of behaviours among the agents. The implicit encouragement of role differentiation allows for dynamic actions in which agents can alter roles dependent on their interactions with the environment without the need for explicit communication between agents. 

**Abstract (ZH)**: 一种集中学习的多Agent采集系统 reinforcement学习策略：减少计算和能源需求并促进角色分化 

---
# InterLoc: LiDAR-based Intersection Localization using Road Segmentation with Automated Evaluation Method 

**Title (ZH)**: InterLoc：基于道路分割的LiDAR交点定位及自动化评估方法 

**Authors**: Nguyen Hoang Khoi Tran, Julie Stephany Berrio, Mao Shan, Zhenxing Ming, Stewart Worrall  

**Link**: [PDF](https://arxiv.org/pdf/2505.00512)  

**Abstract**: Intersections are geometric and functional key points in every road network. They offer strong landmarks to correct GNSS dropouts and anchor new sensor data in up-to-date maps. Despite that importance, intersection detectors either ignore the rich semantic information already computed onboard or depend on scarce, hand-labeled intersection datasets. To close that gap, this paper presents a LiDAR-based method for intersection detection that (i) fuses semantic road segmentation with vehicle localization to detect intersection candidates in a bird's eye view (BEV) representation and (ii) refines those candidates by analyzing branch topology with a least squares formulation. To evaluate our method, we introduce an automated benchmarking pipeline that pairs detections with OpenStreetMap (OSM) intersection nodes using precise GNSS/INS ground-truth poses. Tested on eight SemanticKITTI sequences, the approach achieves a mean localization error of 1.9 m, 89% precision, and 77% recall at a 5 m tolerance, outperforming the latest learning-based baseline. Moreover, the method is robust to segmentation errors higher than those of the benchmark model, demonstrating its applicability in the real world. 

**Abstract (ZH)**: 基于LiDAR的交点检测方法：结合语义道路分割与车辆 localization，在鸟瞰图表示中检测交点候选，并通过最小二乘法 refinement 分析分支拓扑 

---
# Variational OOD State Correction for Offline Reinforcement Learning 

**Title (ZH)**: 离线强化学习中的变分异常值外部状态修正 

**Authors**: Ke Jiang, Wen Jiang, Xiaoyang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00503)  

**Abstract**: The performance of Offline reinforcement learning is significantly impacted by the issue of state distributional shift, and out-of-distribution (OOD) state correction is a popular approach to address this problem. In this paper, we propose a novel method named Density-Aware Safety Perception (DASP) for OOD state correction. Specifically, our method encourages the agent to prioritize actions that lead to outcomes with higher data density, thereby promoting its operation within or the return to in-distribution (safe) regions. To achieve this, we optimize the objective within a variational framework that concurrently considers both the potential outcomes of decision-making and their density, thus providing crucial contextual information for safe decision-making. Finally, we validate the effectiveness and feasibility of our proposed method through extensive experimental evaluations on the offline MuJoCo and AntMaze suites. 

**Abstract (ZH)**: 基于密度aware的安全感知方法（DASP）用于离线强化学习中的异常分布状态矫正 

---
# Urban Air Mobility as a System of Systems: An LLM-Enhanced Holonic Approach 

**Title (ZH)**: 城市空中 mobility 作为系统中系统的实现：一种增强的层级化方法 

**Authors**: Ahmed R. Sadik, Muhammad Ashfaq, Niko Mäkitalo, Tommi Mikkonen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00368)  

**Abstract**: Urban Air Mobility (UAM) is an emerging System of System (SoS) that faces challenges in system architecture, planning, task management, and execution. Traditional architectural approaches struggle with scalability, adaptability, and seamless resource integration within dynamic and complex environments. This paper presents an intelligent holonic architecture that incorporates Large Language Model (LLM) to manage the complexities of UAM. Holons function semi autonomously, allowing for real time coordination among air taxis, ground transport, and vertiports. LLMs process natural language inputs, generate adaptive plans, and manage disruptions such as weather changes or airspace this http URL a case study of multimodal transportation with electric scooters and air taxis, we demonstrate how this architecture enables dynamic resource allocation, real time replanning, and autonomous adaptation without centralized control, creating more resilient and efficient urban transportation networks. By advancing decentralized control and AI driven adaptability, this work lays the groundwork for resilient, human centric UAM ecosystems, with future efforts targeting hybrid AI integration and real world validation. 

**Abstract (ZH)**: 城市空中移动（UAM）是一种新兴的系统中的系统（SoS），面临着系统架构、规划、任务管理及执行的挑战。传统的架构方法在处理动态和复杂的环境中可扩展性、适应性和无缝资源集成方面存在困难。本文提出了一种智能的holonic架构，结合大型语言模型（LLM）以管理UAM的复杂性。holons半自治地运行，允许空中出租车、地面运输和 vertiports 实时协调。LLM处理自然语言输入，生成适应性计划，并管理天气变化或空中交通管制等干扰。在多重运输模式（电动滑板车和空中出租车）的案例研究中，我们展示了这种架构如何实现动态资源分配、实时重规划和无需集中控制的自主适应，从而创建更具韧性和效率的城市交通网络。通过推进去中心化控制和基于人工智能的适应性，本研究为以人类为中心的UAM生态系统奠定了基础，未来工作将集中于混合AI集成和实际验证。 

---
# Guidance and Control of Unmanned Surface Vehicles via HEOL 

**Title (ZH)**: 基于HEOL的自主水面车辆的指导与控制 

**Authors**: Loïck Degorre, Emmanuel Delaleau, Cédric Join, Michel Fliess  

**Link**: [PDF](https://arxiv.org/pdf/2505.00168)  

**Abstract**: This work presents a new approach to the guidance and control of marine craft via HEOL, i.e., a new way of combining flatness-based and model-free controllers. Its goal is to develop a general regulator for Unmanned Surface Vehicles (USV). To do so, the well-known USV maneuvering model is simplified into a nominal Hovercraft model which is flat. A flatness-based controller is derived for the simplified USV model and the loop is closed via an intelligent proportional-derivative (iPD) regulator. We thus associate the well-documented natural robustness of flatness-based control and adaptivity of iPDs. The controller is applied in simulation to two surface vessels, one meeting the simplifying hypotheses, the other one being a generic USV of the literature. It is shown to stabilize both systems even in the presence of unmodeled environmental disturbances. 

**Abstract (ZH)**: 基于HEOL的海洋航行器导航与控制新方法：结合基于平坦性控制器和无模型控制器的新方式 

---
