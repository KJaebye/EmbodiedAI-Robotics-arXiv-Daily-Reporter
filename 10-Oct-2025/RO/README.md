# BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation 

**Title (ZH)**: BLAZER: 基于零样本数据生成的LLM驱动操控代理的初始化方法 

**Authors**: Rocktim Jyoti Das, Harsh Singh, Diana Turmakhan, Muhammad Abdullah Sohail, Mingfei Han, Preslav Nakov, Fabio Pizzati, Ivan Laptev  

**Link**: [PDF](https://arxiv.org/pdf/2510.08572)  

**Abstract**: Scaling data and models has played a pivotal role in the remarkable progress of computer vision and language. Inspired by these domains, recent efforts in robotics have similarly focused on scaling both data and model size to develop more generalizable and robust policies. However, unlike vision and language, robotics lacks access to internet-scale demonstrations across diverse robotic tasks and environments. As a result, the scale of existing datasets typically suffers from the need for manual data collection and curation. To address this problem, here we propose BLAZER, a framework that learns manipulation policies from automatically generated training data. We build on the zero-shot capabilities of LLM planners and automatically generate demonstrations for diverse manipulation tasks in simulation. Successful examples are then used to finetune an LLM and to improve its planning capabilities without human supervision. Notably, while BLAZER training requires access to the simulator's state, we demonstrate direct transfer of acquired skills to sensor-based manipulation. Through extensive experiments, we show BLAZER to significantly improve zero-shot manipulation in both simulated and real environments. Moreover, BLAZER improves on tasks outside of its training pool and enables downscaling of LLM models. Our code and data will be made publicly available on the project page. 

**Abstract (ZH)**: Scaling 数据和模型在计算机视觉和语言领域的显著进步中发挥了关键作用。受到这些领域的启发，近期的机器人研究同样关注于通过扩大数据和模型规模来发展更通用和鲁棒的策略。然而，与视觉和语言不同，机器人领域缺乏跨多种机器人任务和环境的大规模互联网演示数据。因此，现有数据集的规模通常受限于手动数据收集和整理的需要。为解决这一问题，我们提出了一种 BLAZER 框架，该框架通过自动生成的训练数据学习操作策略。我们利用大语言模型（LLM）规划器的零样本能力，自动在外显器中为多种操作任务生成演示。成功的案例随后用于微调 LLM 并改进其规划能力，无需人类监督。值得注意的是，虽然 BLAZER 的训练需要访问模拟器的状态，但我们展示了从传感器操作中直接转移所学技能的能力。通过广泛的实验，我们展示了 BLAZER 在仿真和真实环境中显著提高零样本操作性能，并且能够在训练外的任务上取得进步，还实现了 LLM 模型的缩小。我们的代码和数据将在项目页面上公开。 

---
# Scalable Offline Metrics for Autonomous Driving 

**Title (ZH)**: 可扩展的自主驾驶离线指标 

**Authors**: Animikh Aich, Adwait Kulkarni, Eshed Ohn-Bar  

**Link**: [PDF](https://arxiv.org/pdf/2510.08571)  

**Abstract**: Real-World evaluation of perception-based planning models for robotic systems, such as autonomous vehicles, can be safely and inexpensively conducted offline, i.e., by computing model prediction error over a pre-collected validation dataset with ground-truth annotations. However, extrapolating from offline model performance to online settings remains a challenge. In these settings, seemingly minor errors can compound and result in test-time infractions or collisions. This relationship is understudied, particularly across diverse closed-loop metrics and complex urban maneuvers. In this work, we revisit this undervalued question in policy evaluation through an extensive set of experiments across diverse conditions and metrics. Based on analysis in simulation, we find an even worse correlation between offline and online settings than reported by prior studies, casting doubts on the validity of current evaluation practices and metrics for driving policies. Next, we bridge the gap between offline and online evaluation. We investigate an offline metric based on epistemic uncertainty, which aims to capture events that are likely to cause errors in closed-loop settings. The resulting metric achieves over 13% improvement in correlation compared to previous offline metrics. We further validate the generalization of our findings beyond the simulation environment in real-world settings, where even greater gains are observed. 

**Abstract (ZH)**: 基于感知的规划模型在机器人系统中的实际评估可以通过离线计算模型预测误差来安全且经济地进行，但将离线模型性能 extrapolate 到在线设置仍然是一项挑战。在这些设置中，看似轻微的错误可能会积累并导致测试时的违规或碰撞。这一关系在不同的闭环度量和复杂的城市机动中被忽视。在这项工作中，我们通过一系列广泛实验重新审视这一被低估的问题，分析表明，离线和在线设置之间的相关性比先前研究报道的更差，这引发了对当前评估实践和驾驶策略度量有效性的质疑。接下来，我们弥合了离线和在线评估之间的差距。我们研究了一种基于证伪不确定性（epistemic uncertainty）的离线度量，旨在捕捉可能导致闭环环境中错误的事件。该度量在相关性上相比之前的离线度量实现了超过13%的改进。我们进一步验证了我们在模拟环境之外的实际环境中的一般性结论，在这些环境中观察到更大的益处。 

---
# NovaFlow: Zero-Shot Manipulation via Actionable Flow from Generated Videos 

**Title (ZH)**: NovaFlow: 零样本操控通过生成视频中的可操作流 

**Authors**: Hongyu Li, Lingfeng Sun, Yafei Hu, Duy Ta, Jennifer Barry, George Konidaris, Jiahui Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08568)  

**Abstract**: Enabling robots to execute novel manipulation tasks zero-shot is a central goal in robotics. Most existing methods assume in-distribution tasks or rely on fine-tuning with embodiment-matched data, limiting transfer across platforms. We present NovaFlow, an autonomous manipulation framework that converts a task description into an actionable plan for a target robot without any demonstrations. Given a task description, NovaFlow synthesizes a video using a video generation model and distills it into 3D actionable object flow using off-the-shelf perception modules. From the object flow, it computes relative poses for rigid objects and realizes them as robot actions via grasp proposals and trajectory optimization. For deformable objects, this flow serves as a tracking objective for model-based planning with a particle-based dynamics model. By decoupling task understanding from low-level control, NovaFlow naturally transfers across embodiments. We validate on rigid, articulated, and deformable object manipulation tasks using a table-top Franka arm and a Spot quadrupedal mobile robot, and achieve effective zero-shot execution without demonstrations or embodiment-specific training. Project website: this https URL. 

**Abstract (ZH)**: 使机器人能够在零样本情况下执行新型操作任务是机器人研究中的一个核心目标。现有的大多数方法假设归属于分布的任务，或者依赖于与实体匹配的数据进行微调，这限制了跨平台的迁移。我们提出了一种名为NovaFlow的自主操作框架，能够在没有任何示教的情况下，将任务描述转换为目标机器人的可执行计划。给定一个任务描述，NovaFlow使用视频生成模型合成一个视频，并使用现成的感知模块将其提炼为3D可执行对象流。从对象流中，它计算刚性物体的相对姿态，并通过抓取建议和轨迹优化将它们实现为机器人操作。对于变形体，这种流作为基于模型的规划中的跟踪目标，利用颗粒动力学模型。通过将任务理解与低级控制解耦，NovaFlow自然地实现了跨实体的迁移。我们使用桌面Franka手臂和Spot四足移动机器人验证了刚性、分节和变形物体操作任务的有效零样本执行，无需示教或特定于实体的训练。项目网站：这个 https URL。 

---
# DexNDM: Closing the Reality Gap for Dexterous In-Hand Rotation via Joint-Wise Neural Dynamics Model 

**Title (ZH)**: DexNDM：通过关节级神经动力学模型缩小 Dexterous In-Hand Rotation 的现实差距 

**Authors**: Xueyi Liu, He Wang, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2510.08556)  

**Abstract**: Achieving generalized in-hand object rotation remains a significant challenge in robotics, largely due to the difficulty of transferring policies from simulation to the real world. The complex, contact-rich dynamics of dexterous manipulation create a "reality gap" that has limited prior work to constrained scenarios involving simple geometries, limited object sizes and aspect ratios, constrained wrist poses, or customized hands. We address this sim-to-real challenge with a novel framework that enables a single policy, trained in simulation, to generalize to a wide variety of objects and conditions in the real world. The core of our method is a joint-wise dynamics model that learns to bridge the reality gap by effectively fitting limited amount of real-world collected data and then adapting the sim policy's actions accordingly. The model is highly data-efficient and generalizable across different whole-hand interaction distributions by factorizing dynamics across joints, compressing system-wide influences into low-dimensional variables, and learning each joint's evolution from its own dynamic profile, implicitly capturing these net effects. We pair this with a fully autonomous data collection strategy that gathers diverse, real-world interaction data with minimal human intervention. Our complete pipeline demonstrates unprecedented generality: a single policy successfully rotates challenging objects with complex shapes (e.g., animals), high aspect ratios (up to 5.33), and small sizes, all while handling diverse wrist orientations and rotation axes. Comprehensive real-world evaluations and a teleoperation application for complex tasks validate the effectiveness and robustness of our approach. Website: this https URL 

**Abstract (ZH)**: 实现通用的在手物体旋转仍然是机器人技术中的一个重要挑战，很大程度上这是因为从仿真到现实世界政策转移的困难。灵巧操作的复杂接触动力学造成了“现实差距”，限制了先前工作的应用场景，通常局限于简单几何形状、有限尺寸和长宽比、受限的手腕姿态或定制的手部。我们通过一个新颖的框架解决了这一从仿真到现实的世界挑战，该框架能够使在仿真中训练的单一策略在现实世界中广泛应用于各种物体和条件。我们方法的核心是一个关节级动力学模型，该模型通过有效拟合少量收集的真实世界数据并相应调整仿真策略的动作来弥合现实差距。该模型在关节间分解动力学、压缩系统级影响为低维变量，并通过学习每个关节自身的动力学特征来学习每个关节的演变，从而隐含地捕捉这些综合效应，表现出高度的数据高效性和跨不同全手交互分布的泛化能力。我们还配以一个完全自主的数据收集策略，收集多样化的现实世界交互数据，且最少的人工干预。我们的完整管道展示了前所未有的通用性：单一策略成功旋转了具有复杂形状（如动物）、高长宽比（高达5.33）和小尺寸的挑战性物体，同时处理多种手腕姿态和旋转轴。全面的现实世界评估和复杂任务的遥控操作应用验证了我们方法的有效性和鲁棒性。网站：https://this-url。 

---
# R2RGEN: Real-to-Real 3D Data Generation for Spatially Generalized Manipulation 

**Title (ZH)**: R2RGEN: 实景到实景的3D数据生成及其在空间泛化操作中的应用 

**Authors**: Xiuwei Xu, Angyuan Ma, Hankun Li, Bingyao Yu, Zheng Zhu, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08547)  

**Abstract**: Towards the aim of generalized robotic manipulation, spatial generalization is the most fundamental capability that requires the policy to work robustly under different spatial distribution of objects, environment and agent itself. To achieve this, substantial human demonstrations need to be collected to cover different spatial configurations for training a generalized visuomotor policy via imitation learning. Prior works explore a promising direction that leverages data generation to acquire abundant spatially diverse data from minimal source demonstrations. However, most approaches face significant sim-to-real gap and are often limited to constrained settings, such as fixed-base scenarios and predefined camera viewpoints. In this paper, we propose a real-to-real 3D data generation framework (R2RGen) that directly augments the pointcloud observation-action pairs to generate real-world data. R2RGen is simulator- and rendering-free, thus being efficient and plug-and-play. Specifically, given a single source demonstration, we introduce an annotation mechanism for fine-grained parsing of scene and trajectory. A group-wise augmentation strategy is proposed to handle complex multi-object compositions and diverse task constraints. We further present camera-aware processing to align the distribution of generated data with real-world 3D sensor. Empirically, R2RGen substantially enhances data efficiency on extensive experiments and demonstrates strong potential for scaling and application on mobile manipulation. 

**Abstract (ZH)**: 向通用机器人 manipulation 的目标迈进，空间泛化是最基本的能力，要求策略在不同物体分布、环境和自身位置的情况下都能稳健工作。为了实现这一目标，需要收集大量的人类演示数据以涵盖不同的空间配置，通过模仿学习训练通用的视觉-运动策略。此前的工作探索了一条有希望的方向，利用数据生成从少量源演示中获取丰富的空间多样数据。然而，大多数方法面临着显著的仿真实验到真实环境的差距，并且通常局限于约束场景，如固定基座场景和预定义的摄像头视角。在本文中，我们提出了一种实时到实时的3D数据生成框架（R2RGen），它可以直接增强点云观测-动作对以生成真实世界数据。R2RGen 是无需模拟器和渲染的，因此高效且即插即用。具体来说，给定单一源演示，我们引入了一种标注机制进行细粒度的场景和轨迹解析。我们提出了一种组别增强策略来处理复杂的多物体组合和多样的任务约束。进一步地，我们提出了摄像头感知的处理方法，以使生成数据的分布与真实世界的3D传感器分布相匹配。实验结果表明，R2RGen 显著提高了数据效率，并展示了在移动操作中扩展和应用的强潜力。 

---
# DexMan: Learning Bimanual Dexterous Manipulation from Human and Generated Videos 

**Title (ZH)**: DexMan: 从人类和生成的视频中学习双臂灵巧操作 

**Authors**: Jhen Hsieh, Kuan-Hsun Tu, Kuo-Han Hung, Tsung-Wei Ke  

**Link**: [PDF](https://arxiv.org/pdf/2510.08475)  

**Abstract**: We present DexMan, an automated framework that converts human visual demonstrations into bimanual dexterous manipulation skills for humanoid robots in simulation. Operating directly on third-person videos of humans manipulating rigid objects, DexMan eliminates the need for camera calibration, depth sensors, scanned 3D object assets, or ground-truth hand and object motion annotations. Unlike prior approaches that consider only simplified floating hands, it directly controls a humanoid robot and leverages novel contact-based rewards to improve policy learning from noisy hand-object poses estimated from in-the-wild videos.
DexMan achieves state-of-the-art performance in object pose estimation on the TACO benchmark, with absolute gains of 0.08 and 0.12 in ADD-S and VSD. Meanwhile, its reinforcement learning policy surpasses previous methods by 19% in success rate on OakInk-v2. Furthermore, DexMan can generate skills from both real and synthetic videos, without the need for manual data collection and costly motion capture, and enabling the creation of large-scale, diverse datasets for training generalist dexterous manipulation. 

**Abstract (ZH)**: DexMan:一种将人类视觉演示自动转换为类人机器人双臂灵巧操作技能的框架 

---
# Don't Run with Scissors: Pruning Breaks VLA Models but They Can Be Recovered 

**Title (ZH)**: 不要用剪刀跑：剪枝会破坏VLA模型，但它们可以恢复 

**Authors**: Jason Jabbour, Dong-Ki Kim, Max Smith, Jay Patrikar, Radhika Ghosal, Youhui Wang, Ali Agha, Vijay Janapa Reddi, Shayegan Omidshafiei  

**Link**: [PDF](https://arxiv.org/pdf/2510.08464)  

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic capabilities but remain challenging to deploy on resource-limited hardware. Pruning has enabled efficient compression of large language models (LLMs), yet it is largely understudied in robotics. Surprisingly, we observe that pruning VLA models leads to drastic degradation and increased safety violations. We introduce GLUESTICK, a post-pruning recovery method that restores much of the original model's functionality while retaining sparsity benefits. Our method performs a one-time interpolation between the dense and pruned models in weight-space to compute a corrective term. This correction is used during inference by each pruned layer to recover lost capabilities with minimal overhead. GLUESTICK requires no additional training, is agnostic to the pruning algorithm, and introduces a single hyperparameter that controls the tradeoff between efficiency and accuracy. Across diverse VLA architectures and tasks in manipulation and navigation, GLUESTICK achieves competitive memory efficiency while substantially recovering success rates and reducing safety violations. Additional material can be found at: this https URL. 

**Abstract (ZH)**: Vision-Language-Action (VLA)模型提升了机器人的能力，但仍在资源受限的硬件上难以部署。剪裁方法能够有效压缩大规模语言模型（LLMs），但在机器人领域的研究尚属薄弱环节。令人惊讶的是，我们发现剪裁VLA模型会导致功能大幅退化和安全违规增加。我们提出了GLUESTICK，一种后剪裁恢复方法，可以在保留稀疏性优势的同时恢复大部分原始模型的功能。该方法在权重空间中对密集模型和剪裁模型进行一次插值计算纠正项，在推理时每个剪裁层使用该纠正项以最少的开销恢复丢失的能力。GLUESTICK无需额外训练，与剪裁算法无关，仅引入一个控制效率与准确度Trade-off的超参数。GLUESTICK在各种VLA架构和操作与导航任务中，实现了 competitive 的内存效率，大幅恢复了成功率并减少了安全违规。更多资料请参见：this https URL。 

---
# Validation of collision-free spheres of Stewart-Gough platforms for constant orientations using the Application Programming Interface of a CAD software 

**Title (ZH)**: 使用CAD软件应用程序接口验证常定姿态下Stewart-Gough平台无碰撞球体的有效性 

**Authors**: Bibekananda Patra, Rajeevlochana G. Chittawadigi, Sandipan Bandyopadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2510.08408)  

**Abstract**: This paper presents a method of validation of the size of the largest collision-free sphere (CFS) of a 6-6 Stewart-Gough platform manipulator (SGPM) for a given orientation of its moving platform (MP) using the Application Programming Interface (API) of a CAD software. The position of the MP is updated via the API in an automated manner over a set of samples within a shell enclosing the surface of the CFS. For each pose of the manipulator, each pair of legs is investigated for mutual collisions. The CFS is considered safe or validated iff none of the points falling inside the CFS lead to a collision between any pair of legs. This approach can not only validate the safety of a precomputed CFS, but also estimate the same for any spatial parallel manipulator. 

**Abstract (ZH)**: 本文提出了一种使用CAD软件API验证给定导向平台 orientations 下的6-6 Stewart-Gough 平台 manipulator 最大无碰撞球体 (CFS) 尺寸的方法。通过API在包围 CFS 表面的壳体内自动更新导向平台的位置，对机器人的每个姿态，检查每对腿是否存在碰撞。CFS 被视为安全或验证通过，当且仅当 CFS 内部的任意点都不导致任何一对腿之间发生碰撞。此方法不仅可以验证预计算的 CFS 的安全性，还可以估计任何空间并联 manipulator 的 CFS。 

---
# Reliability of Single-Level Equality-Constrained Inverse Optimal Control 

**Title (ZH)**: 单水平约束最优控制的可靠性 

**Authors**: Filip Bečanović, Kosta Jovanović, Vincent Bonnet  

**Link**: [PDF](https://arxiv.org/pdf/2510.08406)  

**Abstract**: Inverse optimal control (IOC) allows the retrieval of optimal cost function weights, or behavioral parameters, from human motion. The literature on IOC uses methods that are either based on a slow bilevel process or a fast but noise-sensitive minimization of optimality condition violation. Assuming equality-constrained optimal control models of human motion, this article presents a faster but robust approach to solving IOC using a single-level reformulation of the bilevel method and yields equivalent results. Through numerical experiments in simulation, we analyze the robustness to noise of the proposed single-level reformulation to the bilevel IOC formulation with a human-like planar reaching task that is used across recent studies. The approach shows resilience to very large levels of noise and reduces the computation time of the IOC on this task by a factor of 15 when compared to a classical bilevel implementation. 

**Abstract (ZH)**: 基于单层改革的逆最优控制：快速且鲁棒的人类运动最优成本函数权重检索方法 

---
# Airy: Reading Robot Intent through Height and Sky 

**Title (ZH)**: Airy：通过高度和天空读取机器人意图 

**Authors**: Baoyang Chen, Xian Xu, Huamin Qu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08381)  

**Abstract**: As industrial robots move into shared human spaces, their opaque decision making threatens safety, trust, and public oversight. This artwork, Airy, asks whether complex multi agent AI can become intuitively understandable by staging a competition between two reinforcement trained robot arms that snap a bedsheet skyward. Building on three design principles, competition as a clear metric (who lifts higher), embodied familiarity (audiences recognize fabric snapping), and sensor to sense mapping (robot cooperation or rivalry shown through forest and weather projections), the installation gives viewers a visceral way to read machine intent. Observations from five international exhibitions indicate that audiences consistently read the robots' strategies, conflict, and cooperation in real time, with emotional reactions that mirror the system's internal state. The project shows how sensory metaphors can turn a black box into a public interface. 

**Abstract (ZH)**: 随着工业机器人进入共有人类空间，其不透明的决策可能威胁到安全、信任和公众监督。这件艺术品Airy质疑复杂的多智能体AI是否可以通过展示两只强化训练的机器人手臂比赛将面料向上弹起而变得直观易懂。依托三种设计原则——比赛作为明确的度量标准（谁举起更高）、具身的熟悉度（观众能识别面料的弹起）、传感器到感知的映射（通过森林和天气投影展示机器人合作或竞争），该装置为观众提供了一种直觉的方式来解读机器意图。五个国际展览的观察表明，观众能够实时读取机器人的策略、冲突和合作，并产生的情感反应映射了系统内部状态。该项目展示了传感隐喻如何将黑箱转化为公众界面。 

---
# Evaluation of a Robust Control System in Real-World Cable-Driven Parallel Robots 

**Title (ZH)**: 实时电缆驱动并联机器人中稳健控制系统的评估 

**Authors**: Damir Nurtdinov, Aliaksei Korshuk, Alexei Kornaev, Alexander Maloletov  

**Link**: [PDF](https://arxiv.org/pdf/2510.08270)  

**Abstract**: This study evaluates the performance of classical and modern control methods for real-world Cable-Driven Parallel Robots (CDPRs), focusing on underconstrained systems with limited time discretization. A comparative analysis is conducted between classical PID controllers and modern reinforcement learning algorithms, including Deep Deterministic Policy Gradient (DDPG), Proximal Policy Optimization (PPO), and Trust Region Policy Optimization (TRPO). The results demonstrate that TRPO outperforms other methods, achieving the lowest root mean square (RMS) errors across various trajectories and exhibiting robustness to larger time intervals between control updates. TRPO's ability to balance exploration and exploitation enables stable control in noisy, real-world environments, reducing reliance on high-frequency sensor feedback and computational demands. These findings highlight TRPO's potential as a robust solution for complex robotic control tasks, with implications for dynamic environments and future applications in sensor fusion or hybrid control strategies. 

**Abstract (ZH)**: 本研究评估了经典和现代控制方法在实际缆索驱动并联机器人（CDPR）中的性能，重点关注欠约束系统在有限时间离散化情况下的表现。通过经典PID控制器与现代强化学习算法（包括深度确定性策略梯度算法DDPG、近似的策略优化算法PPO和信任区域策略优化算法TRPO）进行对比分析。结果显示，TRPO在各种轨迹上的均方根（RMS）误差最低，并且在控制更新之间的时间间隔较大时表现出较好的稳健性。TRPO在平衡探索与利用方面的能力使其能够在嘈杂的实际环境中实现稳定的控制，减少对高频传感器反馈以及计算需求的依赖。这些发现强调了TRPO作为复杂机器人控制任务稳健解决方案的潜力，对其在动态环境以及未来传感器融合或混合控制策略中的应用具有重要意义。 

---
# NavSpace: How Navigation Agents Follow Spatial Intelligence Instructions 

**Title (ZH)**: NavSpace: 如何导航代理遵循空间智能指令 

**Authors**: Haolin Yang, Yuxing Long, Zhuoyuan Yu, Zihan Yang, Minghan Wang, Jiapeng Xu, Yihan Wang, Ziyan Yu, Wenzhe Cai, Lei Kang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.08173)  

**Abstract**: Instruction-following navigation is a key step toward embodied intelligence. Prior benchmarks mainly focus on semantic understanding but overlook systematically evaluating navigation agents' spatial perception and reasoning capabilities. In this work, we introduce the NavSpace benchmark, which contains six task categories and 1,228 trajectory-instruction pairs designed to probe the spatial intelligence of navigation agents. On this benchmark, we comprehensively evaluate 22 navigation agents, including state-of-the-art navigation models and multimodal large language models. The evaluation results lift the veil on spatial intelligence in embodied navigation. Furthermore, we propose SNav, a new spatially intelligent navigation model. SNav outperforms existing navigation agents on NavSpace and real robot tests, establishing a strong baseline for future work. 

**Abstract (ZH)**: 指令遵循导航是实现体域智能的关键步骤。先前的标准主要侧重于语义理解，但忽视了系统性评估导航代理的空间知觉和推理能力。在这项工作中，我们引入了NavSpace基准，包含六类任务和1,228个轨迹-指令对，旨在探究导航代理的空间智能。在此基准上，我们全面评估了22个导航代理，包括最新的导航模型和多模态大型语言模型。评估结果揭示了体域导航中的空间智能。此外，我们提出了SNav，这是一种新的空间智能导航模型。SNav在NavSpace和真实机器人测试中优于现有导航代理，为未来工作建立了强大的基线。 

---
# Accurate and Noise-Tolerant Extraction of Routine Logs in Robotic Process Automation (Extended Version) 

**Title (ZH)**: 准确且抗噪声的机器人过程自动化常规日志提取（扩展版） 

**Authors**: Massimiliano de Leoni, Faizan Ahmed Khan, Simone Agostinelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.08118)  

**Abstract**: Robotic Process Mining focuses on the identification of the routine types performed by human resources through a User Interface. The ultimate goal is to discover routine-type models to enable robotic process automation. The discovery of routine-type models requires the provision of a routine log. Unfortunately, the vast majority of existing works do not directly focus on enabling the model discovery, limiting themselves to extracting the set of actions that are part of the routines. They were also not evaluated in scenarios characterized by inconsistent routine execution, hereafter referred to as noise, which reflects natural variability and occasional errors in human performance. This paper presents a clustering-based technique that aims to extract routine logs. Experiments were conducted on nine UI logs from the literature with different levels of injected noise. Our technique was compared with existing techniques, most of which are not meant to discover routine logs but were adapted for the purpose. The results were evaluated through standard state-of-the-art metrics, showing that we can extract more accurate routine logs than what the state of the art could, especially in the presence of noise. 

**Abstract (ZH)**: 基于机器人流程挖掘：通过用户界面识别常规任务类型以发现常规类型模型 

---
# Beyond hospital reach: Autonomous lightweight ultrasound robot for liver sonography 

**Title (ZH)**: 超越医院范围：自主轻量级超声机器人用于肝脏超声检查 

**Authors**: Zihan Li, Yixiao Xu, Lei Zhang, Taiyu Han, Xinshan Yang, Yingni Wang, Mingxuan Liu, Shenghai Xin, Linxun Liu, Hongen Liao, Guochen Ning  

**Link**: [PDF](https://arxiv.org/pdf/2510.08106)  

**Abstract**: Liver disease is a major global health burden. While ultrasound is the first-line diagnostic tool, liver sonography requires locating multiple non-continuous planes from positions where target structures are often not visible, for biometric assessment and lesion detection, requiring significant expertise. However, expert sonographers are severely scarce in resource-limited regions. Here, we develop an autonomous lightweight ultrasound robot comprising an AI agent that integrates multi-modal perception with memory attention for localization of unseen target structures, and a 588-gram 6-degrees-of-freedom cable-driven robot. By mounting on the abdomen, the system enhances robustness against motion. Our robot can autonomously acquire expert-level standard liver ultrasound planes and detect pathology in patients, including two from Xining, a 2261-meter-altitude city with limited medical resources. Our system performs effectively on rapid-motion individuals and in wilderness environments. This work represents the first demonstration of autonomous sonography across multiple challenging scenarios, potentially transforming access to expert-level diagnostics in underserved regions. 

**Abstract (ZH)**: 肝脏疾病是全球重大的公共卫生负担。尽管超声是首选的诊断工具，但肝脏超声要求从目标结构往往不可见的位置识别多个非连续切面，进行生物测量和病变检测，这需要很高的专业技能。然而，在资源有限的地区，具备这种技能的超声专家极其稀缺。在这里，我们开发了一种自主轻量级超声机器人，包括一个集多模态感知与记忆注意于一体的AI代理，以及一个重588克、具有6自由度的电缆驱动机器人。通过腹部固定，系统增强了对运动的鲁棒性。该机器人能够自主获取专家级标准肝脏超声切面，并在患者中检测病理情况，包括来自海拨2261米、医疗资源有限的西宁市的两例患者。该系统在快速运动个体和荒野环境中表现有效。本工作是首次在多种挑战场景下展示自主超声成像，有望在欠服务地区变革专家级诊断的可及性。 

---
# Towards Reliable LLM-based Robot Planning via Combined Uncertainty Estimation 

**Title (ZH)**: 基于结合不确定性估计的可靠LLM驱动机器人规划 

**Authors**: Shiyuan Yin, Chenjia Bai, Zihao Zhang, Junwei Jin, Xinxin Zhang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08044)  

**Abstract**: Large language models (LLMs) demonstrate advanced reasoning abilities, enabling robots to understand natural language instructions and generate high-level plans with appropriate grounding. However, LLM hallucinations present a significant challenge, often leading to overconfident yet potentially misaligned or unsafe plans. While researchers have explored uncertainty estimation to improve the reliability of LLM-based planning, existing studies have not sufficiently differentiated between epistemic and intrinsic uncertainty, limiting the effectiveness of uncertainty esti- mation. In this paper, we present Combined Uncertainty estimation for Reliable Embodied planning (CURE), which decomposes the uncertainty into epistemic and intrinsic uncertainty, each estimated separately. Furthermore, epistemic uncertainty is subdivided into task clarity and task familiarity for more accurate evaluation. The overall uncertainty assessments are obtained using random network distillation and multi-layer perceptron regression heads driven by LLM features. We validated our approach in two distinct experimental settings: kitchen manipulation and tabletop rearrangement experiments. The results show that, compared to existing methods, our approach yields uncertainty estimates that are more closely aligned with the actual execution outcomes. 

**Abstract (ZH)**: 结合 epistemic 和 intrinsic 不确定性估计以实现可靠物理交互的 CURE 方法 

---
# FastUMI-100K: Advancing Data-driven Robotic Manipulation with a Large-scale UMI-style Dataset 

**Title (ZH)**: FastUMI-100K：以大规模UMI风格数据集推动数据驱动的机器人操作研究 

**Authors**: Kehui Liu, Zhongjie Jia, Yang Li, Zhaxizhuoma, Pengan Chen, Song Liu, Xin Liu, Pingrui Zhang, Haoming Song, Xinyi Ye, Nieqing Cao, Zhigang Wang, Jia Zeng, Dong Wang, Yan Ding, Bin Zhao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.08022)  

**Abstract**: Data-driven robotic manipulation learning depends on large-scale, high-quality expert demonstration datasets. However, existing datasets, which primarily rely on human teleoperated robot collection, are limited in terms of scalability, trajectory smoothness, and applicability across different robotic embodiments in real-world environments. In this paper, we present FastUMI-100K, a large-scale UMI-style multimodal demonstration dataset, designed to overcome these limitations and meet the growing complexity of real-world manipulation tasks. Collected by FastUMI, a novel robotic system featuring a modular, hardware-decoupled mechanical design and an integrated lightweight tracking system, FastUMI-100K offers a more scalable, flexible, and adaptable solution to fulfill the diverse requirements of real-world robot demonstration data. Specifically, FastUMI-100K contains over 100K+ demonstration trajectories collected across representative household environments, covering 54 tasks and hundreds of object types. Our dataset integrates multimodal streams, including end-effector states, multi-view wrist-mounted fisheye images and textual annotations. Each trajectory has a length ranging from 120 to 500 frames. Experimental results demonstrate that FastUMI-100K enables high policy success rates across various baseline algorithms, confirming its robustness, adaptability, and real-world applicability for solving complex, dynamic manipulation challenges. The source code and dataset will be released in this link this https URL. 

**Abstract (ZH)**: 基于数据驱动的机器人操作学习依赖于大规模、高质量的专家演示数据集。然而，现有数据集主要依赖于人工遥控机器人采集，这些数据集在可扩展性、轨迹平滑度以及在不同机器人实体上的适用性方面存在局限性。本文提出FastUMI-100K，一个大规模的UMI风格多模态演示数据集，旨在克服这些局限性，满足现实世界操作任务日益复杂的需求。FastUMI-100K由一种具有模块化、硬件解耦机械设计及集成轻量级跟踪系统的新型机器人系统FastUMI采集，提供了更具扩展性、灵活性和适应性的解决方案，以满足现实世界机器人演示数据的多样需求。特别是，FastUMI-100K涵盖了超过100K个代表性家庭环境下的演示轨迹，涵盖54项任务和数百种物体类型。该数据集整合了包括末端执行器状态、多视角腕部安装的鱼眼图像及文本注释在内的多模态数据流。每条轨迹包含的帧数范围从120到500帧。实验结果表明，FastUMI-100K能够实现跨多种基线算法的高策略成功率，证实了其在解决复杂动态操作挑战方面的稳健性、适应性和现实世界适用性。源代码和数据集将在以下链接发布：this https URL。 

---
# Orientation Learning and Adaptation towards Simultaneous Incorporation of Multiple Local Constraints 

**Title (ZH)**: 面向多局部约束同时整合的定向学习与适应 

**Authors**: Gaofeng Li, Peisen Xu, Ruize Wang, Qi Ye, Jiming Chen, Dezhen Song, Yanlong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07986)  

**Abstract**: Orientation learning plays a pivotal role in many tasks. However, the rotation group SO(3) is a Riemannian manifold. As a result, the distortion caused by non-Euclidean geometric nature introduces difficulties to the incorporation of local constraints, especially for the simultaneous incorporation of multiple local constraints. To address this issue, we propose the Angle-Axis Space-based orientation representation method to solve several orientation learning problems, including orientation adaptation and minimization of angular acceleration. Specifically, we propose a weighted average mechanism in SO(3) based on the angle-axis representation method. Our main idea is to generate multiple trajectories by considering different local constraints at different basepoints. Then these multiple trajectories are fused to generate a smooth trajectory by our proposed weighted average mechanism, achieving the goal to incorporate multiple local constraints simultaneously. Compared with existing solution, ours can address the distortion issue and make the off-theshelf Euclidean learning algorithm be re-applicable in non-Euclidean space. Simulation and Experimental evaluations validate that our solution can not only adapt orientations towards arbitrary desired via-points and cope with angular acceleration constraints, but also incorporate multiple local constraints simultaneously to achieve extra benefits, e.g., achieving smaller acceleration costs. 

**Abstract (ZH)**: 基于Angle-Axis Space的方向学习方法及其实现多重局部约束的研究 

---
# Executable Analytic Concepts as the Missing Link Between VLM Insight and Precise Manipulation 

**Title (ZH)**: 可执行分析概念作为VLM洞察与精确操作之间的缺失链接 

**Authors**: Mingyang Sun, Jiude Wei, Qichen He, Donglin Wang, Cewu Lu, Jianhua Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.07975)  

**Abstract**: Enabling robots to perform precise and generalized manipulation in unstructured environments remains a fundamental challenge in embodied AI. While Vision-Language Models (VLMs) have demonstrated remarkable capabilities in semantic reasoning and task planning, a significant gap persists between their high-level understanding and the precise physical execution required for real-world manipulation. To bridge this "semantic-to-physical" gap, we introduce GRACE, a novel framework that grounds VLM-based reasoning through executable analytic concepts (EAC)-mathematically defined blueprints that encode object affordances, geometric constraints, and semantics of manipulation. Our approach integrates a structured policy scaffolding pipeline that turn natural language instructions and visual information into an instantiated EAC, from which we derive grasp poses, force directions and plan physically feasible motion trajectory for robot execution. GRACE thus provides a unified and interpretable interface between high-level instruction understanding and low-level robot control, effectively enabling precise and generalizable manipulation through semantic-physical grounding. Extensive experiments demonstrate that GRACE achieves strong zero-shot generalization across a variety of articulated objects in both simulated and real-world environments, without requiring task-specific training. 

**Abstract (ZH)**: 使机器人在未结构化环境中执行精确且通用的操作仍然是嵌入式AI领域的基本挑战。尽管视觉-语言模型（VLMs）在语义推理和任务规划方面展现了出色的能力，但它们的高层理解与实现实用车辆操作所需的精确物理执行之间仍存在显著差距。为了弥合这一“语义到物理”的差距，我们引入了GRACE，一种新的框架，通过可执行分析概念（EAC）将基于VLM的推理接地，EAC是通过数学定义的蓝图，编码物体的可利用性、几何约束和操作的语义。我们的方法结合了一个结构化策略支架管道，将自然语言指令和视觉信息转化为实例化的EAC，从中我们推导出抓取姿态、力的方向，并规划出物理可行的运动轨迹供机器人执行。GRACE因此提供了一个统一且可解释的接口，将高层指令理解与低层机器人控制结合起来，有效地通过语义-物理接地实现精确且通用的操作。广泛的经验实验证明，GRACE在模拟和实际环境中的各种活动物体上实现了强大的零样本泛化，无需特定任务的训练。 

---
# Towards Proprioception-Aware Embodied Planning for Dual-Arm Humanoid Robots 

**Title (ZH)**: 面向 proprioception 意识的双臂类人机器人体化规划 

**Authors**: Boyu Li, Siyuan He, Hang Xu, Haoqi Yuan, Yu Zang, Liwei Hu, Junpeng Yue, Zhenxiong Jiang, Pengbo Hu, Börje F. Karlsson, Yehui Tang, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07882)  

**Abstract**: In recent years, Multimodal Large Language Models (MLLMs) have demonstrated the ability to serve as high-level planners, enabling robots to follow complex human instructions. However, their effectiveness, especially in long-horizon tasks involving dual-arm humanoid robots, remains limited. This limitation arises from two main challenges: (i) the absence of simulation platforms that systematically support task evaluation and data collection for humanoid robots, and (ii) the insufficient embodiment awareness of current MLLMs, which hinders reasoning about dual-arm selection logic and body positions during planning. To address these issues, we present DualTHOR, a new dual-arm humanoid simulator, with continuous transition and a contingency mechanism. Building on this platform, we propose Proprio-MLLM, a model that enhances embodiment awareness by incorporating proprioceptive information with motion-based position embedding and a cross-spatial encoder. Experiments show that, while existing MLLMs struggle in this environment, Proprio-MLLM achieves an average improvement of 19.75% in planning performance. Our work provides both an essential simulation platform and an effective model to advance embodied intelligence in humanoid robotics. The code is available at this https URL. 

**Abstract (ZH)**: 近年来，多模态大型语言模型（MLLMs）展示了作为高级规划者的潜力，使机器人能够遵循复杂的指令。然而，尤其是在涉及双臂人形机器人的长期任务上，其效果仍然有限。这种限制来源于两个主要挑战：（i）缺乏系统支持人形机器人任务评估和数据采集的仿真平台；（ii）当前MLLMs的不足感知能力，这阻碍了在规划过程中对双臂选择逻辑和身体位置的推理。为应对这些挑战，我们提出了DualTHOR，一个具有连续过渡和预案机制的新双臂人形机器人仿真平台。基于此平台，我们提出了一种Proprio-MLLM模型，通过结合本体感受信息和基于运动的位置嵌入以及跨空间编码器来增强感知能力。实验结果显示，在此环境中现有的MLLMs难以胜任，而Proprio-MLLM的规划性能平均提高了19.75%。我们的工作提供了一个重要的仿真平台和有效的模型，以促进人形机器人中的本体智能。代码已公开：这个 https URL。 

---
# Team Xiaomi EV-AD VLA: Learning to Navigate Socially Through Proactive Risk Perception - Technical Report for IROS 2025 RoboSense Challenge Social Navigation Track 

**Title (ZH)**: 小米Team EV-AD VLA：通过主动风险感知进行社会导航的技术报告 - 2025 IROS RoboSense挑战赛社会导航轨道 

**Authors**: Erjia Xiao, Lingfeng Zhang, Yingbo Tang, Hao Cheng, Renjing Xu, Wenbo Ding, Lei Zhou, Long Chen, Hangjun Ye, Xiaoshuai Hao  

**Link**: [PDF](https://arxiv.org/pdf/2510.07871)  

**Abstract**: In this report, we describe the technical details of our submission to the IROS 2025 RoboSense Challenge Social Navigation Track. This track focuses on developing RGBD-based perception and navigation systems that enable autonomous agents to navigate safely, efficiently, and socially compliantly in dynamic human-populated indoor environments. The challenge requires agents to operate from an egocentric perspective using only onboard sensors including RGB-D observations and odometry, without access to global maps or privileged information, while maintaining social norm compliance such as safe distances and collision avoidance. Building upon the Falcon model, we introduce a Proactive Risk Perception Module to enhance social navigation performance. Our approach augments Falcon with collision risk understanding that learns to predict distance-based collision risk scores for surrounding humans, which enables the agent to develop more robust spatial awareness and proactive collision avoidance behaviors. The evaluation on the Social-HM3D benchmark demonstrates that our method improves the agent's ability to maintain personal space compliance while navigating toward goals in crowded indoor scenes with dynamic human agents, achieving 2nd place among 16 participating teams in the challenge. 

**Abstract (ZH)**: 本报告描述了我们对2025年IROS RoboSense挑战赛Social Navigation赛道的提交内容。该赛道专注于开发基于RGBD的感知与导航系统，使自主代理能够在动态的人群密集室内环境中安全、高效且符合社会规范地导航。挑战要求代理从第一人称视角使用车载传感器（包括RGB-D观测和里程计）操作，不使用全局地图或特权信息，同时保持符合社会规范的行为，如保持安全距离和避免碰撞。在Falcon模型的基础上，我们引入了一个前瞻性的风险感知模块以增强社会导航性能。我们的方法通过使Falcon能够理解碰撞风险，并学会预测周围人类的距离基碰撞风险评分，从而帮助代理发展出更稳健的空间意识和更具前瞻性的避碰行为。在Social-HM3D基准上的评估表明，我们的方法在动态人类代理占据场景中的拥挤室内环境中导航时，提高了代理保持个人空间合规的能力，并在挑战中以16支参赛队伍中的第2名获得佳绩。 

---
# USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots 

**Title (ZH)**: USIM和U0：一种适用于通用水下机器人的视觉-语言-行动数据集和模型 

**Authors**: Junwen Gu, Zhiheng wu, Pengxuan Si, Shuang Qiu, Yukai Feng, Luoyang Sun, Laien Luo, Lianyi Yu, Jian Wang, Zhengxing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07869)  

**Abstract**: Underwater environments present unique challenges for robotic operation, including complex hydrodynamics, limited visibility, and constrained communication. Although data-driven approaches have advanced embodied intelligence in terrestrial robots and enabled task-specific autonomous underwater robots, developing underwater intelligence capable of autonomously performing multiple tasks remains highly challenging, as large-scale, high-quality underwater datasets are still scarce. To address these limitations, we introduce USIM, a simulation-based multi-task Vision-Language-Action (VLA) dataset for underwater robots. USIM comprises over 561K frames from 1,852 trajectories, totaling approximately 15.6 hours of BlueROV2 interactions across 20 tasks in 9 diverse scenarios, ranging from visual navigation to mobile manipulation. Building upon this dataset, we propose U0, a VLA model for general underwater robots, which integrates binocular vision and other sensor modalities through multimodal fusion, and further incorporates a convolution-attention-based perception focus enhancement module (CAP) to improve spatial understanding and mobile manipulation. Across tasks such as inspection, obstacle avoidance, scanning, and dynamic tracking, the framework achieves a success rate of 80%, while in challenging mobile manipulation tasks, it reduces the distance to the target by 21.2% compared with baseline methods, demonstrating its effectiveness. USIM and U0 show that VLA models can be effectively applied to underwater robotic applications, providing a foundation for scalable dataset construction, improved task autonomy, and the practical realization of intelligent general underwater robots. 

**Abstract (ZH)**: 水下环境为机器人操作带来了独特挑战，包括复杂的水动力学、受限的能见度以及通信限制。尽管数据驱动的方法已经在地面机器人中促进了感觉自己智能的发展，并使专用的自主水下机器人成为可能，但在开发能够自主执行多种任务的水下智能方面仍面临巨大挑战，因为高质量的数据集仍然稀缺。为解决这些问题，我们引入了USIM，这是一种基于模拟的水下机器人多任务视觉-语言-动作（VLA）数据集。USIM包含来自1,852条轨迹的超过561K帧数据，共计约15.6小时的BlueROV2与9种不同场景下20项任务的交互，涵盖了从视觉导航到移动操作等多种场景。在此基础上，我们提出了一种适用于通用水下机器人的VLA模型U0，该模型通过多模态融合整合了双目视觉和其他传感器数据，并进一步集成了基于卷积注意的感知焦点增强模块（CAP），以提高空间理解和移动操作能力。在诸如检查、障碍物规避、扫描和动态跟踪等任务中，该框架的成功率达到了80%，而在具有挑战性的移动操作任务中，与基线方法相比，成功接近目标的距离减少了21.2%，展示了其有效性。USIM和U0表明，VLA模型可以有效应用于水下机器人应用，为构建可扩展的数据集、提高任务自主性以及实现智能通用水下机器人提供了基础。 

---
# DM1: MeanFlow with Dispersive Regularization for 1-Step Robotic Manipulation 

**Title (ZH)**: DM1: MeanFlow结合分散正则化的一步机器人操作 

**Authors**: Guowei Zou, Haitao Wang, Hejun Wu, Yukun Qian, Yuhang Wang, Weibing Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.07865)  

**Abstract**: The ability to learn multi-modal action distributions is indispensable for robotic manipulation policies to perform precise and robust control. Flow-based generative models have recently emerged as a promising solution to learning distributions of actions, offering one-step action generation and thus achieving much higher sampling efficiency compared to diffusion-based methods. However, existing flow-based policies suffer from representation collapse, the inability to distinguish similar visual representations, leading to failures in precise manipulation tasks. We propose DM1 (MeanFlow with Dispersive Regularization for One-Step Robotic Manipulation), a novel flow matching framework that integrates dispersive regularization into MeanFlow to prevent collapse while maintaining one-step efficiency. DM1 employs multiple dispersive regularization variants across different intermediate embedding layers, encouraging diverse representations across training batches without introducing additional network modules or specialized training procedures. Experiments on RoboMimic benchmarks show that DM1 achieves 20-40 times faster inference (0.07s vs. 2-3.5s) and improves success rates by 10-20 percentage points, with the Lift task reaching 99% success over 85% of the baseline. Real-robot deployment on a Franka Panda further validates that DM1 transfers effectively from simulation to the physical world. To the best of our knowledge, this is the first work to leverage representation regularization to enable flow-based policies to achieve strong performance in robotic manipulation, establishing a simple yet powerful approach for efficient and robust manipulation. 

**Abstract (ZH)**: 基于散射正则化的DM1：一种用于一步机器人 manipulation 的流匹配框架 

---
# GM3: A General Physical Model for Micro-Mobility Vehicles 

**Title (ZH)**: GM3: 一种通用的微移动车辆物理模型 

**Authors**: Grace Cai, Nithin Parepally, Laura Zheng, Ming C. Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.07807)  

**Abstract**: Modeling the dynamics of micro-mobility vehicles (MMV) is becoming increasingly important for training autonomous vehicle systems and building urban traffic simulations. However, mainstream tools rely on variants of the Kinematic Bicycle Model (KBM) or mode-specific physics that miss tire slip, load transfer, and rider/vehicle lean. To our knowledge, no unified, physics-based model captures these dynamics across the full range of common MMVs and wheel layouts. We propose the "Generalized Micro-mobility Model" (GM3), a tire-level formulation based on the tire brush representation that supports arbitrary wheel configurations, including single/double track and multi-wheel platforms. We introduce an interactive model-agnostic simulation framework that decouples vehicle/layout specification from dynamics to compare the GM3 with the KBM and other models, consisting of fixed step RK4 integration, human-in-the-loop and scripted control, real-time trajectory traces and logging for analysis. We also empirically validate the GM3 on the Stanford Drone Dataset's deathCircle (roundabout) scene for biker, skater, and cart classes. 

**Abstract (ZH)**: 微移动车辆动力学建模对于训练自主车辆系统和构建城市交通模拟越来越重要。然而，主流工具主要依赖于动力学自行车模型（KBM）的变体或特定模式的物理模型，这些模型未能捕捉车胎滑移、载荷转移和骑手/车辆倾斜的动力学。据我们所知，没有统一的基于物理的动力学模型能够覆盖常见微移动车辆和轮配置的全范围动力学。我们提出了一种“通用微移动模型”（GM3），这是一种基于车胎刷表示的轮胎级公式，支持任意轮配置，包括单轨/双轨和多轮平台。我们引入了一个交互式的模型无关仿真框架，将车辆/布局定义与动力学分离，以比较GM3与KBM和其他模型，该框架包括固定步长RK4积分、人工闭环控制和脚本控制、实时轨迹追踪和日志记录以供分析。我们还在斯坦福无人机数据集中deathCircle（环岛）场景上对GM3进行了经验验证，针对骑手、滑板手和小车类别进行了验证。 

---
# IntentionVLA: Generalizable and Efficient Embodied Intention Reasoning for Human-Robot Interaction 

**Title (ZH)**: 意图VLA：通用且高效的实体意图推理在人机交互中的应用 

**Authors**: Yandu Chen, Kefan Gu, Yuqing Wen, Yucheng Zhao, Tiancai Wang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2510.07778)  

**Abstract**: Vision-Language-Action (VLA) models leverage pretrained vision-language models (VLMs) to couple perception with robotic control, offering a promising path toward general-purpose embodied intelligence. However, current SOTA VLAs are primarily pretrained on multimodal tasks with limited relevance to embodied scenarios, and then finetuned to map explicit instructions to actions. Consequently, due to the lack of reasoning-intensive pretraining and reasoning-guided manipulation, these models are unable to perform implicit human intention reasoning required for complex, real-world interactions. To overcome these limitations, we propose \textbf{IntentionVLA}, a VLA framework with a curriculum training paradigm and an efficient inference mechanism. Our proposed method first leverages carefully designed reasoning data that combine intention inference, spatial grounding, and compact embodied reasoning, endowing the model with both reasoning and perception capabilities. In the following finetuning stage, IntentionVLA employs the compact reasoning outputs as contextual guidance for action generation, enabling fast inference under indirect instructions. Experimental results show that IntentionVLA substantially outperforms $\pi_0$, achieving 18\% higher success rates with direct instructions and 28\% higher than ECoT under intention instructions. On out-of-distribution intention tasks, IntentionVLA achieves over twice the success rate of all baselines, and further enables zero-shot human-robot interaction with 40\% success rate. These results highlight IntentionVLA as a promising paradigm for next-generation human-robot interaction (HRI) systems. 

**Abstract (ZH)**: IntentionVLA：一种具有分阶训练范式和高效推理机制的Vision-Language-Action框架 

---
# Trajectory Conditioned Cross-embodiment Skill Transfer 

**Title (ZH)**: 基于轨迹条件的跨躯体技能迁移 

**Authors**: YuHang Tang, Yixuan Lou, Pengfei Han, Haoming Song, Xinyi Ye, Dong Wang, Bin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.07773)  

**Abstract**: Learning manipulation skills from human demonstration videos presents a promising yet challenging problem, primarily due to the significant embodiment gap between human body and robot manipulators. Existing methods rely on paired datasets or hand-crafted rewards, which limit scalability and generalization. We propose TrajSkill, a framework for Trajectory Conditioned Cross-embodiment Skill Transfer, enabling robots to acquire manipulation skills directly from human demonstration videos. Our key insight is to represent human motions as sparse optical flow trajectories, which serve as embodiment-agnostic motion cues by removing morphological variations while preserving essential dynamics. Conditioned on these trajectories together with visual and textual inputs, TrajSkill jointly synthesizes temporally consistent robot manipulation videos and translates them into executable actions, thereby achieving cross-embodiment skill transfer. Extensive experiments are conducted, and the results on simulation data (MetaWorld) show that TrajSkill reduces FVD by 39.6\% and KVD by 36.6\% compared with the state-of-the-art, and improves cross-embodiment success rate by up to 16.7\%. Real-robot experiments in kitchen manipulation tasks further validate the effectiveness of our approach, demonstrating practical human-to-robot skill transfer across embodiments. 

**Abstract (ZH)**: 从人类示范视频中学习操作技能是一个极具前景但充满挑战的问题，主要原因在于人类身体与机器人操作器之间存在显著的体能力差距。现有方法依赖配对数据集或手工设计的奖励函数，这限制了其可扩展性和泛化能力。我们提出了TrajSkill框架，该框架实现了基于轨迹条件的跨体能力操作技能转移，使机器人能够直接从人类示范视频中获取操作技能。我们的核心洞察在于将人类动作表示为稀疏的光流轨迹，这些轨迹能够通过去除形态变异而保留本质动态，同时保持体能力无关的动作线索。在视觉和文本输入的条件下，TrajSkill联合生成时空一致的机器人操作视频，并将其转化为可执行的动作，从而实现跨体能力技能转移。广泛的实验结果表明，在仿真数据（MetaWorld）上，与现有最佳方法相比，TrajSkill将FVD降低了39.6%，KVD降低了36.6%，并将跨体能力的成功率提高了最高16.7%。厨房操作任务的机器人实验进一步验证了我们方法的有效性，展示了实用的人机操作技能转移。 

---
# Injecting Hallucinations in Autonomous Vehicles: A Component-Agnostic Safety Evaluation Framework 

**Title (ZH)**: 在自主车辆中注入幻觉：一种不依赖组件的安全评估框架 

**Authors**: Alexandre Moreira Nascimento, Gabriel Kenji Godoy Shimanuki, Lúcio Flavio Vismari, João Batista Camargo Jr, Jorge Rady de Almeida Jr, Paulo Sergio Cugnasca, Anna Carolina Muller Queiroz, Jeremy Noah Bailenson  

**Link**: [PDF](https://arxiv.org/pdf/2510.07749)  

**Abstract**: Perception failures in autonomous vehicles (AV) remain a major safety concern because they are the basis for many accidents. To study how these failures affect safety, researchers typically inject artificial faults into hardware or software components and observe the outcomes. However, existing fault injection studies often target a single sensor or machine perception (MP) module, resulting in siloed frameworks that are difficult to generalize or integrate into unified simulation environments. This work addresses that limitation by reframing perception failures as hallucinations, false perceptions that distort an AV situational awareness and may trigger unsafe control actions. Since hallucinations describe only observable effects, this abstraction enables analysis independent of specific sensors or algorithms, focusing instead on how their faults manifest along the MP pipeline. Building on this concept, we propose a configurable, component-agnostic hallucination injection framework that induces six plausible hallucination types in an iterative open-source simulator. More than 18,350 simulations were executed in which hallucinations were injected while AVs crossed an unsignalized transverse street with traffic. The results statistically validate the framework and quantify the impact of each hallucination type on collisions and near misses. Certain hallucinations, such as perceptual latency and drift, significantly increase the risk of collision in the scenario tested, validating the proposed paradigm can stress the AV system safety. The framework offers a scalable, statistically validated, component agnostic, and fully interoperable toolset that simplifies and accelerates AV safety validations, even those with novel MP architectures and components. It can potentially reduce the time-to-market of AV and lay the foundation for future research on fault tolerance, and resilient AV design. 

**Abstract (ZH)**: 自主车辆（AV）感知故障对安全的影响研究 

---
# Probabilistically-Safe Bipedal Navigation over Uncertain Terrain via Conformal Prediction and Contraction Analysis 

**Title (ZH)**: 基于共形预测和收缩分析的不确定地形上概率安全的双足导航 

**Authors**: Kasidit Muenprasitivej, Ye Zhao, Glen Chou  

**Link**: [PDF](https://arxiv.org/pdf/2510.07725)  

**Abstract**: We address the challenge of enabling bipedal robots to traverse rough terrain by developing probabilistically safe planning and control strategies that ensure dynamic feasibility and centroidal robustness under terrain uncertainty. Specifically, we propose a high-level Model Predictive Control (MPC) navigation framework for a bipedal robot with a specified confidence level of safety that (i) enables safe traversal toward a desired goal location across a terrain map with uncertain elevations, and (ii) formally incorporates uncertainty bounds into the centroidal dynamics of locomotion control. To model the rough terrain, we employ Gaussian Process (GP) regression to estimate elevation maps and leverage Conformal Prediction (CP) to construct calibrated confidence intervals that capture the true terrain elevation. Building on this, we formulate contraction-based reachable tubes that explicitly account for terrain uncertainty, ensuring state convergence and tube invariance. In addition, we introduce a contraction-based flywheel torque control law for the reduced-order Linear Inverted Pendulum Model (LIPM), which stabilizes the angular momentum about the center-of-mass (CoM). This formulation provides both probabilistic safety and goal reachability guarantees. For a given confidence level, we establish the forward invariance of the proposed torque control law by demonstrating exponential stabilization of the actual CoM phase-space trajectory and the desired trajectory prescribed by the high-level planner. Finally, we evaluate the effectiveness of our planning framework through physics-based simulations of the Digit bipedal robot in MuJoCo. 

**Abstract (ZH)**: 我们通过开发在地形不确定性条件下保证动力学可行性和质心稳健性的概率安全规划与控制策略，来应对使仿人机器人穿越崎岖地形的挑战。具体来说，我们提出了一种高阶模型预测控制（MPC）导航框架，该框架在给定安全置信水平的情况下（i）能够安全穿越地形图上具有不确定高程的目标位置，（ii）正式地将不确定性边界纳入步行控制的质心动力学中。为了模拟崎岖地形，我们使用高斯过程（GP）回归估计高程图，并利用拟合预测（CP）构建校准后的置信区间，以捕获真实的地形高程。在此基础上，我们制定了考虑地形不确定性的收敛基可达管，确保状态收敛和管的不变性。此外，我们引入了一种基于收敛性的飞轮扭矩控制律，应用于降阶线性倒摆模型（LIPM），以稳定质心（CoM）处的角动量。这种表述提供了概率安全性和目标可达性的保证。对于给定的安全置信水平，我们通过证明实际CoM相空间轨迹的指数稳定性和高阶规划者指定的轨迹的前向不变性，建立了所提扭矩控制律的前向不变性。最后，我们通过MuJoCo中的Digit仿人机器人物理仿真评估了我们规划框架的有效性。 

---
# EB-MBD: Emerging-Barrier Model-Based Diffusion for Safe Trajectory Optimization in Highly Constrained Environments 

**Title (ZH)**: EB-MBD: 基于新兴屏障模型的轨迹优化扩散方法以确保在高度受限环境中的安全路径规划 

**Authors**: Raghav Mishra, Ian R. Manchester  

**Link**: [PDF](https://arxiv.org/pdf/2510.07700)  

**Abstract**: We propose enforcing constraints on Model-Based Diffusion by introducing emerging barrier functions inspired by interior point methods. We show that constraints on Model-Based Diffusion can lead to catastrophic performance degradation, even on simple 2D systems due to sample inefficiency in the Monte Carlo approximation of the score function. We introduce Emerging-Barrier Model-Based Diffusion (EB-MBD) which uses progressively introduced barrier constraints to avoid these problems, significantly improving solution quality, without the need for computationally expensive operations such as projections. We analyze the sampling liveliness of samples each iteration to inform barrier parameter scheduling choice. We demonstrate results for 2D collision avoidance and a 3D underwater manipulator system and show that our method achieves lower cost solutions than Model-Based Diffusion, and requires orders of magnitude less computation time than projection based methods. 

**Abstract (ZH)**: 我们提出通过引入受内部点方法启发的新兴障碍函数来对基于模型的扩散过程施加约束。我们表明，基于模型的扩散过程中的约束可能导致性能灾难性下降，即使在简单的2D系统中也是如此，这是由于蒙特卡洛近似得分函数的样本效率低下所致。我们引入了新兴障碍基于模型的扩散（EB-MBD），通过逐步引入障碍约束来避免这些问题，显著提高了解的质量，而无需进行计算昂贵的投影操作。我们分析了每次迭代中样本的采样活跃性，以指导障碍参数调度的选择。我们展示了2D碰撞避免和3D水下操作臂系统的结果，并表明我们的方法可以获得更低成本的解，且所需计算时间比基于投影的方法少几个数量级。 

---
# Differentiable Particle Optimization for Fast Sequential Manipulation 

**Title (ZH)**: 可微分粒子优化算法实现快速序列操纵 

**Authors**: Lucas Chen, Shrutheesh Raman Iyer, Zachary Kingston  

**Link**: [PDF](https://arxiv.org/pdf/2510.07674)  

**Abstract**: Sequential robot manipulation tasks require finding collision-free trajectories that satisfy geometric constraints across multiple object interactions in potentially high-dimensional configuration spaces. Solving these problems in real-time and at large scales has remained out of reach due to computational requirements. Recently, GPU-based acceleration has shown promising results, but prior methods achieve limited performance due to CPU-GPU data transfer overhead and complex logic that prevents full hardware utilization. To this end, we present SPaSM (Sampling Particle optimization for Sequential Manipulation), a fully GPU-parallelized framework that compiles constraint evaluation, sampling, and gradient-based optimization into optimized CUDA kernels for end-to-end trajectory optimization without CPU coordination. The method consists of a two-stage particle optimization strategy: first solving placement constraints through massively parallel sampling, then lifting solutions to full trajectory optimization in joint space. Unlike hierarchical approaches, SPaSM jointly optimizes object placements and robot trajectories to handle scenarios where motion feasibility constrains placement options. Experimental evaluation on challenging benchmarks demonstrates solution times in the realm of $\textbf{milliseconds}$ with a 100% success rate; a $4000\times$ speedup compared to existing approaches. 

**Abstract (ZH)**: 序贯机器人操作任务需要在多个对象交互中找到满足几何约束的无碰撞轨迹，可能在高维配置空间中进行。由于计算需求，实时且大尺度地解决这些问题一直无法实现。最近，基于GPU的加速取得了有希望的结果，但先前的方法由于CPU-GPU数据传输开销和复杂的逻辑限制了全硬件利用。为此，我们提出SPaSM（采样粒子优化用于序贯操作），这是一种完全GPU并行化框架，将约束评估、采样和基于梯度的优化编译为优化的CUDA内核，实现端到端轨迹优化，无需CPU协调。该方法包括两阶段粒子优化策略：首先通过大规模并行采样求解放置约束，然后在关节空间中提升解以进行完整的轨迹优化。与分层方法不同，SPaSM联合优化对象放置和机器人轨迹，以应对运动可行性限制放置选项的场景。实验评估表明，该方法在毫秒级时间内实现了100%的成功率，相较于现有方法速度提升了4000倍。 

---
# GATO: GPU-Accelerated and Batched Trajectory Optimization for Scalable Edge Model Predictive Control 

**Title (ZH)**: GATO：加速并批处理轨迹优化的边端模型预测控制 

**Authors**: Alexander Du, Emre Adabag, Gabriel Bravo, Brian Plancher  

**Link**: [PDF](https://arxiv.org/pdf/2510.07625)  

**Abstract**: While Model Predictive Control (MPC) delivers strong performance across robotics applications, solving the underlying (batches of) nonlinear trajectory optimization (TO) problems online remains computationally demanding. Existing GPU-accelerated approaches typically (i) parallelize a single solve to meet real-time deadlines, (ii) scale to very large batches at slower-than-real-time rates, or (iii) achieve speed by restricting model generality (e.g., point-mass dynamics or a single linearization). This leaves a large gap in solver performance for many state-of-the-art MPC applications that require real-time batches of tens to low-hundreds of solves. As such, we present GATO, an open source, GPU-accelerated, batched TO solver co-designed across algorithm, software, and computational hardware to deliver real-time throughput for these moderate batch size regimes. Our approach leverages a combination of block-, warp-, and thread-level parallelism within and across solves for ultra-high performance. We demonstrate the effectiveness of our approach through a combination of: simulated benchmarks showing speedups of 18-21x over CPU baselines and 1.4-16x over GPU baselines as batch size increases; case studies highlighting improved disturbance rejection and convergence behavior; and finally a validation on hardware using an industrial manipulator. We open source GATO to support reproducibility and adoption. 

**Abstract (ZH)**: GATO：一种面向中等批次大小领域的并行轨迹优化求解器 

---
# Inspection Planning Primitives with Implicit Models 

**Title (ZH)**: 隐式模型下的检查规划 primitives 

**Authors**: Jingyang You, Hanna Kurniawati, Lashika Medagoda  

**Link**: [PDF](https://arxiv.org/pdf/2510.07611)  

**Abstract**: The aging and increasing complexity of infrastructures make efficient inspection planning more critical in ensuring safety. Thanks to sampling-based motion planning, many inspection planners are fast. However, they often require huge memory. This is particularly true when the structure under inspection is large and complex, consisting of many struts and pillars of various geometry and sizes. Such structures can be represented efficiently using implicit models, such as neural Signed Distance Functions (SDFs). However, most primitive computations used in sampling-based inspection planner have been designed to work efficiently with explicit environment models, which in turn requires the planner to use explicit environment models or performs frequent transformations between implicit and explicit environment models during planning. This paper proposes a set of primitive computations, called Inspection Planning Primitives with Implicit Models (IPIM), that enable sampling-based inspection planners to entirely use neural SDFs representation during planning. Evaluation on three scenarios, including inspection of a complex real-world structure with over 92M triangular mesh faces, indicates that even a rudimentary sampling-based planner with IPIM can generate inspection trajectories of similar quality to those generated by the state-of-the-art planner, while using up to 70x less memory than the state-of-the-art inspection planner. 

**Abstract (ZH)**: 基础设施的老化和复杂性增加使得高效的检查规划更加关键，以确保安全。基于采样的运动规划使得许多检查规划器快速，但它们通常需要大量的内存。尤其是当被检查的结构庞大而复杂，由各种几何形状和大小的桁架和支柱组成时，这种结构可以用隐式模型，如神经符号距离函数（SDF）进行有效表示。然而，大多数用于采样基检查规划器的基本计算是为有效处理显式环境模型设计的，这反过来要求规划器使用显式环境模型，或者在规划过程中频繁地在隐式和显式环境模型之间进行转换。本文提出了一组称为隐式模型检查规划基元（IPIM）的基本计算，使得采样基检查规划器在规划过程中可以完全使用神经SDF表示。在三个场景下的评估，包括一个复杂的真实世界结构，其三角网格面超过9200万个，表明即使是最简单的带有IPIM的采样基规划器也能生成与最先进的规划器相似质量的检查轨迹，使用的内存是最新检查规划器的1/70。 

---
# AVO: Amortized Value Optimization for Contact Mode Switching in Multi-Finger Manipulation 

**Title (ZH)**: AVO: Amortized Value Optimization for Contact Mode Switching in Multi-Finger Manipulation 

**Authors**: Adam Hung, Fan Yang, Abhinav Kumar, Sergio Aguilera Marinovic, Soshi Iba, Rana Soltani Zarrin, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2510.07548)  

**Abstract**: Dexterous manipulation tasks often require switching between different contact modes, such as rolling, sliding, sticking, or non-contact contact modes. When formulating dexterous manipulation tasks as a trajectory optimization problem, a common approach is to decompose these tasks into sub-tasks for each contact mode, which are each solved independently. Optimizing each sub-task independently can limit performance, as optimizing contact points, contact forces, or other variables without information about future sub-tasks can place the system in a state from which it is challenging to make progress on subsequent sub-tasks. Further, optimizing these sub-tasks is very computationally expensive. To address these challenges, we propose Amortized Value Optimization (AVO), which introduces a learned value function that predicts the total future task performance. By incorporating this value function into the cost of the trajectory optimization at each planning step, the value function gradients guide the optimizer toward states that minimize the cost in future sub-tasks. This effectively bridges separately optimized sub-tasks, and accelerates the optimization by reducing the amount of online computation needed. We validate AVO on a screwdriver grasping and turning task in both simulation and real world experiments, and show improved performance even with 50% less computational budget compared to trajectory optimization without the value function. 

**Abstract (ZH)**: 灵巧操作任务经常需要在不同的接触模式之间切换，例如滚动、滑动、粘附或非接触模式。将灵巧操作任务建模为轨迹优化问题时，常见的做法是将这些任务分解为每种接触模式的子任务，并独立求解。独立优化每个子任务可能会限制性能，因为在不了解未来子任务信息的情况下优化接触点、接触力或其他变量可能会使系统处于难以推进后续子任务的状态。此外，优化这些子任务非常计算密集。为了解决这些挑战，我们提出了一种曰忘价值优化（AVO），它引入了一个学习的价值函数，该价值函数预测未来的任务总性能。通过将此价值函数整合到轨迹优化的成本中，在每一步规划中，价值函数的梯度引导优化器朝向在后续子任务中最小化成本的状态。这有效地连接了单独优化的子任务，并通过减少所需的在线计算量来加速优化。我们在螺丝刀抓取和旋转任务的模拟和实际实验中验证了AVO，并展示了即使在价值函数未使用的情况下计算预算减少50%的条件下，仍然能够获得更好的性能。 

---
# HJCD-IK: GPU-Accelerated Inverse Kinematics through Batched Hybrid Jacobian Coordinate Descent 

**Title (ZH)**: HJCD-IK: GPU加速的批量混合雅可比坐标下降逆动力学 

**Authors**: Cael Yasutake, Zachary Kingston, Brian Plancher  

**Link**: [PDF](https://arxiv.org/pdf/2510.07514)  

**Abstract**: Inverse Kinematics (IK) is a core problem in robotics, in which joint configurations are found to achieve a desired end-effector pose. Although analytical solvers are fast and efficient, they are limited to systems with low degrees-of-freedom and specific topological structures. Numerical optimization-based approaches are more general, but suffer from high computational costs and frequent convergence to spurious local minima. Recent efforts have explored the use of GPUs to combine sampling and optimization to enhance both the accuracy and speed of IK solvers. We build on this recent literature and introduce HJCD-IK, a GPU-accelerated, sampling-based hybrid solver that combines an orientation-aware greedy coordinate descent initialization scheme with a Jacobian-based polishing routine. This design enables our solver to improve both convergence speed and overall accuracy as compared to the state-of-the-art, consistently finding solutions along the accuracy-latency Pareto frontier and often achieving order-of-magnitude gains. In addition, our method produces a broad distribution of high-quality samples, yielding the lowest maximum mean discrepancy. We release our code open-source for the benefit of the community. 

**Abstract (ZH)**: 基于GPU的混合采样优化逆运动学求解器HJCD-IK 

---
# VeMo: A Lightweight Data-Driven Approach to Model Vehicle Dynamics 

**Title (ZH)**: VeMo: 一种轻量级的数据驱动车辆动力学建模方法 

**Authors**: Girolamo Oddo, Roberto Nuca, Matteo Parsani  

**Link**: [PDF](https://arxiv.org/pdf/2510.07447)  

**Abstract**: Developing a dynamic model for a high-performance vehicle is a complex problem that requires extensive structural information about the system under analysis. This information is often unavailable to those who did not design the vehicle and represents a typical issue in autonomous driving applications, which are frequently developed on top of existing vehicles; therefore, vehicle models are developed under conditions of information scarcity. This paper proposes a lightweight encoder-decoder model based on Gate Recurrent Unit layers to correlate the vehicle's future state with its past states, measured onboard, and control actions the driver performs. The results demonstrate that the model achieves a maximum mean relative error below 2.6% in extreme dynamic conditions. It also shows good robustness when subject to noisy input data across the interested frequency components. Furthermore, being entirely data-driven and free from physical constraints, the model exhibits physical consistency in the output signals, such as longitudinal and lateral accelerations, yaw rate, and the vehicle's longitudinal velocity. 

**Abstract (ZH)**: 基于门循环单元的轻量级编码解码器模型用于高performance车辆的状态预测 

---
# FLEET: Formal Language-Grounded Scheduling for Heterogeneous Robot Teams 

**Title (ZH)**: FLEET: 基于形式语言的异构机器人团队调度 

**Authors**: Corban Rivera, Grayson Byrd, Meghan Booker, Bethany Kemp, Allison Gaines, Emma Holmes, James Uplinger, Celso M de Melo, David Handelman  

**Link**: [PDF](https://arxiv.org/pdf/2510.07417)  

**Abstract**: Coordinating heterogeneous robot teams from free-form natural-language instructions is hard. Language-only planners struggle with long-horizon coordination and hallucination, while purely formal methods require closed-world models. We present FLEET, a hybrid decentralized framework that turns language into optimized multi-robot schedules. An LLM front-end produces (i) a task graph with durations and precedence and (ii) a capability-aware robot--task fitness matrix; a formal back-end solves a makespan-minimization problem while the underlying robots execute their free-form subtasks with agentic closed-loop control. Across multiple free-form language-guided autonomy coordination benchmarks, FLEET improves success over state of the art generative planners on two-agent teams across heterogeneous tasks. Ablations show that mixed integer linear programming (MILP) primarily improves temporal structure, while LLM-derived fitness is decisive for capability-coupled tasks; together they deliver the highest overall performance. We demonstrate the translation to real world challenges with hardware trials using a pair of quadruped robots with disjoint capabilities. 

**Abstract (ZH)**: 从自由形式自然语言指令协调异构机器人团队是具有挑战性的。仅靠语言的规划者难以处理长期协调和幻觉问题，而纯粹形式化的方法需要封闭世界模型。我们提出FLEET，一种混合去中心化框架，将语言转换为优化的多机器人时间表。一个LLM前端生成（i）带有持续时间和顺序的任务图，以及（ii）感知能力的机器人-任务适应矩阵；形式化后端解决最长时间间隔最小化问题，而基础机器人则通过具备主体性的闭环控制执行其自由形式的子任务。在多个自由形式语言引导的自主协调基准测试中，FLEET在异构任务的双机器人团队中优于最先进的生成规划器。消融实验表明，混合整数线性编程（MILP）主要改善了时间结构，而LLM衍生的适应性对于能力耦合任务至关重要；两者共同提供了最高的整体性能。我们通过使用一对具有不同能力的四足机器人进行硬件试验，展示了其对现实世界挑战的适用性。 

---
# ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving 

**Title (ZH)**: ResAD: 归一化残差轨迹 modeling 用于端到端自动驾驶 

**Authors**: Zhiyu Zheng, Shaoyu Chen, Haoran Yin, Xinbang Zhang, Jialv Zou, Xinggang Wang, Qian Zhang, Lefei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.08562)  

**Abstract**: End-to-end autonomous driving (E2EAD) systems, which learn to predict future trajectories directly from sensor data, are fundamentally challenged by the inherent spatio-temporal imbalance of trajectory data. This imbalance creates a significant optimization burden, causing models to learn spurious correlations instead of causal inference, while also prioritizing uncertain, distant predictions, thereby compromising immediate safety. To address these issues, we propose ResAD, a novel Normalized Residual Trajectory Modeling framework. Instead of predicting the future trajectory directly, our approach reframes the learning task to predict the residual deviation from a deterministic inertial reference. The inertial reference serves as a counterfactual, forcing the model to move beyond simple pattern recognition and instead identify the underlying causal factors (e.g., traffic rules, obstacles) that necessitate deviations from a default, inertially-guided path. To deal with the optimization imbalance caused by uncertain, long-term horizons, ResAD further incorporates Point-wise Normalization of the predicted residual. It re-weights the optimization objective, preventing large-magnitude errors associated with distant, uncertain waypoints from dominating the learning signal. Extensive experiments validate the effectiveness of our framework. On the NAVSIM benchmark, ResAD achieves a state-of-the-art PDMS of 88.6 using a vanilla diffusion policy with only two denoising steps, demonstrating that our approach significantly simplifies the learning task and improves model performance. The code will be released to facilitate further research. 

**Abstract (ZH)**: 面向未来的残差轨迹预测：一种新型归一化残差轨迹建模框架（ResAD） 

---
# Dream to Recall: Imagination-Guided Experience Retrieval for Memory-Persistent Vision-and-Language Navigation 

**Title (ZH)**: 梦境 recall：想象引导的体验检索在记忆持久的视觉-语言导航中的应用 

**Authors**: Yunzhe Xu, Yiyuan Pan, Zhe Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.08553)  

**Abstract**: Vision-and-Language Navigation (VLN) requires agents to follow natural language instructions through environments, with memory-persistent variants demanding progressive improvement through accumulated experience. Existing approaches for memory-persistent VLN face critical limitations: they lack effective memory access mechanisms, instead relying on entire memory incorporation or fixed-horizon lookup, and predominantly store only environmental observations while neglecting navigation behavioral patterns that encode valuable decision-making strategies. We present Memoir, which employs imagination as a retrieval mechanism grounded by explicit memory: a world model imagines future navigation states as queries to selectively retrieve relevant environmental observations and behavioral histories. The approach comprises: 1) a language-conditioned world model that imagines future states serving dual purposes: encoding experiences for storage and generating retrieval queries; 2) Hybrid Viewpoint-Level Memory that anchors both observations and behavioral patterns to viewpoints, enabling hybrid retrieval; and 3) an experience-augmented navigation model that integrates retrieved knowledge through specialized encoders. Extensive evaluation across diverse memory-persistent VLN benchmarks with 10 distinctive testing scenarios demonstrates Memoir's effectiveness: significant improvements across all scenarios, with 5.4% SPL gains on IR2R over the best memory-persistent baseline, accompanied by 8.3x training speedup and 74% inference memory reduction. The results validate that predictive retrieval of both environmental and behavioral memories enables more effective navigation, with analysis indicating substantial headroom (73.3% vs 93.4% upper bound) for this imagination-guided paradigm. Code at this https URL. 

**Abstract (ZH)**: 基于视觉-语言导航的记忆持久性框架：使用想象进行预测性记忆检索 

---
# Have We Scene It All? Scene Graph-Aware Deep Point Cloud Compression 

**Title (ZH)**: 我们见过所有的场景吗？基于场景图的深度点云压缩 

**Authors**: Nikolaos Stathoulopoulos, Christoforos Kanellakis, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.08512)  

**Abstract**: Efficient transmission of 3D point cloud data is critical for advanced perception in centralized and decentralized multi-agent robotic systems, especially nowadays with the growing reliance on edge and cloud-based processing. However, the large and complex nature of point clouds creates challenges under bandwidth constraints and intermittent connectivity, often degrading system performance. We propose a deep compression framework based on semantic scene graphs. The method decomposes point clouds into semantically coherent patches and encodes them into compact latent representations with semantic-aware encoders conditioned by Feature-wise Linear Modulation (FiLM). A folding-based decoder, guided by latent features and graph node attributes, enables structurally accurate reconstruction. Experiments on the SemanticKITTI and nuScenes datasets show that the framework achieves state-of-the-art compression rates, reducing data size by up to 98% while preserving both structural and semantic fidelity. In addition, it supports downstream applications such as multi-robot pose graph optimization and map merging, achieving trajectory accuracy and map alignment comparable to those obtained with raw LiDAR scans. 

**Abstract (ZH)**: 基于语义场景图的高效3D点云数据传输对于集中式和分布式多机器人系统高级感知至关重要，尤其是在现今广泛依赖边缘和基于云的处理的情况下。然而，点云的大型和复杂性质在带宽受限和间歇连接环境下造成了挑战，通常会降低系统性能。我们提出了一种基于语义场景图的深度压缩框架。该方法将点云分解为语义上一致的patches，并通过特征aware编码器（基于特征向量线性调制FiLM）将它们编码为紧凑的潜在表示。基于折叠的解码器在潜在特征和图节点属性的指导下，实现结构上准确的重构。在SemanticKITTI和nuScenes数据集上的实验表明，该框架实现了最先进的压缩率，即使数据量减少了高达98%，也能保持结构和语义的保真度。此外，该框架支持多机器人位姿图优化和地图合并等下游应用，其轨迹精度和地图对齐与原始LiDAR扫描相当。 

---
# Gaze on the Prize: Shaping Visual Attention with Return-Guided Contrastive Learning 

**Title (ZH)**: 聚焦奖励：基于回报导向的对比学习塑造视觉注意力 

**Authors**: Andrew Lee, Ian Chuang, Dechen Gao, Kai Fukazawa, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2510.08442)  

**Abstract**: Visual Reinforcement Learning (RL) agents must learn to act based on high-dimensional image data where only a small fraction of the pixels is task-relevant. This forces agents to waste exploration and computational resources on irrelevant features, leading to sample-inefficient and unstable learning. To address this, inspired by human visual foveation, we introduce Gaze on the Prize. This framework augments visual RL with a learnable foveal attention mechanism (Gaze), guided by a self-supervised signal derived from the agent's experience pursuing higher returns (the Prize). Our key insight is that return differences reveal what matters most: If two similar representations produce different outcomes, their distinguishing features are likely task-relevant, and the gaze should focus on them accordingly. This is realized through return-guided contrastive learning that trains the attention to distinguish between the features relevant to success and failure. We group similar visual representations into positives and negatives based on their return differences and use the resulting labels to construct contrastive triplets. These triplets provide the training signal that teaches the attention mechanism to produce distinguishable representations for states associated with different outcomes. Our method achieves up to 2.4x improvement in sample efficiency and can solve tasks that the baseline fails to learn, demonstrated across a suite of manipulation tasks from the ManiSkill3 benchmark, all without modifying the underlying algorithm or hyperparameters. 

**Abstract (ZH)**: 基于视觉的增强学习（RL）代理必须基于高维度图像数据学习行动，其中只有很小一部分像素与任务相关。这迫使代理在无关特征上浪费探索和计算资源，导致学习效率低下和不稳定。为了解决这一问题，受人类视觉中心凝视启发，我们引入了“追求奖励的凝视”框架。该框架以可学习的中心凝视注意力机制（凝视）为视觉RL增添功能，该机制由代理在追求更高回报过程中自我监督得到的信号（奖励）引导。我们的关键见解是，回报差异揭示了最重要的是什么：如果两个相似的表示产生不同的结果，它们的区别特征很可能与任务相关，注意点应相应地关注这些特征。通过基于回报引导的对比学习，训练注意力机制区分与成功和失败相关的特征。我们根据回报差异将相似的视觉表示分组为正样本和负样本，并使用这些标签构建对比三元组。这些三元组提供了训练信号，教会注意力机制生成与不同结果相关的状态的可区分表示。我们的方法在样本效率方面最多可提高2.4倍，并能在无需修改基础算法或超参数的情况下解决基准套件ManiSkill3中一系列抓取任务，这些任务是基准中基础方法无法学习的。 

---
# Co-design is powerful and not free 

**Title (ZH)**: 协同设计强大但不免费。 

**Authors**: Yi Zhang, Yue Xie, Tao Sun, Fumiya Iida  

**Link**: [PDF](https://arxiv.org/pdf/2510.08368)  

**Abstract**: Robotic performance emerges from the coupling of body and controller, yet it remains unclear when morphology-control co-design is necessary. We present a unified framework that embeds morphology and control parameters within a single neural network, enabling end-to-end joint optimization. Through case studies in static-obstacle-constrained reaching, we evaluate trajectory error, success rate, and collision probability. The results show that co-design provides clear benefits when morphology is poorly matched to the task, such as near obstacles or workspace boundaries, where structural adaptation simplifies control. Conversely, when the baseline morphology already affords sufficient capability, control-only optimization often matches or exceeds co-design. By clarifying when control is enough and when it is not, this work advances the understanding of embodied intelligence and offers practical guidance for embodiment-aware robot design. 

**Abstract (ZH)**: 机器人性能源自身体与控制器的耦合，但尚不清楚何时需要形态-控制协同设计。我们提出了一种统一框架，将形态和控制参数嵌入单个神经网络中，实现端到端联合优化。通过静态障碍约束下的拾取任务案例研究，我们评估了轨迹误差、成功率和碰撞概率。结果表明，当形态与任务匹配不佳，例如靠近障碍物或工作空间边界时，协同设计提供明显的益处，结构适应简化了控制。相反，当基线形态已经具备足够的能力时，仅控制优化往往与或超过协同设计。通过明确控制何时足够、何时不够，这项工作促进了对嵌入式智能的理解，并为体aware机器人设计提供了 practical 指导。 

---
# A Multimodal Depth-Aware Method For Embodied Reference Understanding 

**Title (ZH)**: 一种多模态深度aware方法用于体现式引用理解 

**Authors**: Fevziye Irem Eyiokur, Dogucan Yaman, Hazım Kemal Ekenel, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2510.08278)  

**Abstract**: Embodied Reference Understanding requires identifying a target object in a visual scene based on both language instructions and pointing cues. While prior works have shown progress in open-vocabulary object detection, they often fail in ambiguous scenarios where multiple candidate objects exist in the scene. To address these challenges, we propose a novel ERU framework that jointly leverages LLM-based data augmentation, depth-map modality, and a depth-aware decision module. This design enables robust integration of linguistic and embodied cues, improving disambiguation in complex or cluttered environments. Experimental results on two datasets demonstrate that our approach significantly outperforms existing baselines, achieving more accurate and reliable referent detection. 

**Abstract (ZH)**: 基于语言指令和指示手势识别视觉场景中的目标对象需要理解具身参照，为此我们提出了一种新的ERU框架，该框架结合了基于LLM的数据增强、深度图模态和深度感知决策模块，以实现语言和具身线索的 robust 整合，并在复杂或拥挤环境中提高消岐能力。实验结果表明，我们的方法显著优于现有基线，实现了更准确可靠的指代检测。 

---
# DEAS: DEtached value learning with Action Sequence for Scalable Offline RL 

**Title (ZH)**: DEAS: 与动作序列分离的价值学习可扩展的离线RL 

**Authors**: Changyeon Kim, Haeone Lee, Younggyo Seo, Kimin Lee, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07730)  

**Abstract**: Offline reinforcement learning (RL) presents an attractive paradigm for training intelligent agents without expensive online interactions. However, current approaches still struggle with complex, long-horizon sequential decision making. In this work, we introduce DEtached value learning with Action Sequence (DEAS), a simple yet effective offline RL framework that leverages action sequences for value learning. These temporally extended actions provide richer information than single-step actions and can be interpreted through the options framework via semi-Markov decision process Q-learning, enabling reduction of the effective planning horizon by considering longer sequences at once. However, directly adopting such sequences in actor-critic algorithms introduces excessive value overestimation, which we address through detached value learning that steers value estimates toward in-distribution actions that achieve high return in the offline dataset. We demonstrate that DEAS consistently outperforms baselines on complex, long-horizon tasks from OGBench and can be applied to enhance the performance of large-scale Vision-Language-Action models that predict action sequences, significantly boosting performance in both RoboCasa Kitchen simulation tasks and real-world manipulation tasks. 

**Abstract (ZH)**: 脱почted 值学习与动作序列（DEAS）：一种用于复杂长时序决策的简单有效离线强化学习框架 

---
# IGUANA: Immersive Guidance, Navigation, and Control for Consumer UAV 

**Title (ZH)**: IGUANA: 浸没式指导、导航与控制 for 消费级无人机 

**Authors**: Victor Victor, Tania Krisanty, Matthew McGinity, Stefan Gumhold, Uwe Aßmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.07609)  

**Abstract**: As the markets for unmanned aerial vehicles (UAVs) and mixed reality (MR) headsets continue to grow, recent research has increasingly explored their integration, which enables more intuitive, immersive, and situationally aware control systems. We present IGUANA, an MR-based immersive guidance, navigation, and control system for consumer UAVs. IGUANA introduces three key elements beyond conventional control interfaces: (1) a 3D terrain map interface with draggable waypoint markers and live camera preview for high-level control, (2) a novel spatial control metaphor that uses a virtual ball as a physical analogy for low-level control, and (3) a spatial overlay that helps track the UAV when it is not visible with the naked eye or visual line of sight is interrupted. We conducted a user study to evaluate our design, both quantitatively and qualitatively, and found that (1) the 3D map interface is intuitive and easy to use, relieving users from manual control and suggesting improved accuracy and consistency with lower perceived workload relative to conventional dual-stick controller, (2) the virtual ball interface is intuitive but limited by the lack of physical feedback, and (3) the spatial overlay is very useful in enhancing the users' situational awareness. 

**Abstract (ZH)**: 随着无人驾驶航空车辆(UAVs)和混合现实(MR)头显市场的持续增长，近期的研究越来越多地探索它们的集成，这使得更直观、沉浸和情境感知的控制系统成为可能。我们提出了IGUANA，一种基于混合现实的沉浸式导航和控制系统，适用于消费级无人驾驶航空车辆。IGUANA引入了三种超越传统控制接口的关键元素：(1) 三维地形图界面，带有可拖动的航点标记和实时摄像头预览，用于高级控制；(2) 一种新颖的空间控制元喻，使用虚拟球体作为物理类比，用于低级控制；(3) 空间叠加，有助于当无人驾驶航空车辆不可见或视线中断时进行跟踪。我们进行了用户研究，从定量和定性两个方面评估了我们的设计，并发现(1) 三维地图界面直观易用，减轻了用户的手动控制负担，与传统的双摇杆控制器相比，提高了准确性和一致性，并减少了感知工作负荷；(2) 虚拟球体界面直观但受限于缺乏物理反馈；(3) 空间叠加对增强用户的情境感知非常有用。 

---
# A Rotation-Invariant Embedded Platform for (Neural) Cellular Automata 

**Title (ZH)**: 旋转不变嵌入平台 for (神经)细胞自动机 

**Authors**: Dominik Woiwode, Jakob Marten, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2510.07440)  

**Abstract**: This paper presents a rotation-invariant embedded platform for simulating (neural) cellular automata (NCA) in modular robotic systems. Inspired by previous work on physical NCA, we introduce key innovations that overcome limitations in prior hardware designs. Our platform features a symmetric, modular structure, enabling seamless connections between cells regardless of orientation. Additionally, each cell is battery-powered, allowing it to operate independently and retain its state even when disconnected from the collective. To demonstrate the platform's applicability, we present a novel rotation-invariant NCA model for isotropic shape classification. The proposed system provides a robust foundation for exploring the physical realization of NCA, with potential applications in distributed robotic systems and self-organizing structures. Our implementation, including hardware, software code, a simulator, and a video, is openly shared at: this https URL 

**Abstract (ZH)**: 本文提出了一种旋转不变嵌入平台，用于在模块化机器人系统中模拟（神经）细胞自动机（NCA）。受先前关于物理NCA工作的启发，我们引入了关键创新，克服了先前硬件设计的限制。该平台采用对称、模块化结构，使得无论朝向如何，各个单元之间均可无缝连接。此外，每个单元都由电池供电，使其在断开与集体连接时仍能独立运行并保持状态。为展示该平台的应用性，我们提出了一种适用于各向同性形状分类的旋转不变NCA模型。所提出的系统为探索NCA的物理实现提供了坚实的基础，具有在分布式机器人系统和自组织结构中潜在的应用价值。我们的实现，包括硬件、软件代码、模拟器和视频，已公开分享在：this https URL。 

---
# Bioinspired Tapered-Spring Turbulence Sensor for Underwater Flow Detection 

**Title (ZH)**: 生物启发的锥形弹簧湍流传感器用于水下流场检测 

**Authors**: Xiao Jin, Zhenhua Yu, Thrishantha Nanayakkara  

**Link**: [PDF](https://arxiv.org/pdf/2510.07348)  

**Abstract**: This paper presents a bio-inspired underwater whisker sensor for robust hydrodynamic disturbance detection and efficient signal analysis based on Physical Reservoir Computing (PRC). The design uses a tapered nylon spring with embedded accelerometers to achieve spatially distributed vibration sensing and frequency separation along the whisker. Towing-tank experiments and computational fluid dynamics simulations confirmed that the whisker effectively distinguishes vortex regimes across different fin angles and maintains Strouhal scaling with flow velocity, where higher speeds increase vibration intensity without affecting the dominant frequencies. Frequency-domain analysis, Shannon entropy, and machine learning further validated the sensing performance: vortex shedding frequencies were identified with less than 10\% error, entropy captured the transition from coherent vortex streets to turbulence, and logistic regression achieved 86.0\% classification accuracy with millisecond-level inference. These results demonstrate that structurally encoded whisker sensing provides a scalable and real-time solution for underwater perception, wake tracking, and turbulence-aware navigation in autonomous marine robots. 

**Abstract (ZH)**: 基于物理 reservoir 计算的受生物启发的水下触须传感器及其在流体力学扰动检测与高效信号分析中的应用 

---
