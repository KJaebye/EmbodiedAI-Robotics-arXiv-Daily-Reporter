# GET-USE: Learning Generalized Tool Usage for Bimanual Mobile Manipulation via Simulated Embodiment Extensions 

**Title (ZH)**: GET-USE: 通过模拟体能扩展学习双臂移动操作中的通用工具使用方法 

**Authors**: Bohan Wu, Paul de La Sayette, Li Fei-Fei, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2510.25754)  

**Abstract**: The ability to use random objects as tools in a generalizable manner is a missing piece in robots' intelligence today to boost their versatility and problem-solving capabilities. State-of-the-art robotic tool usage methods focused on procedurally generating or crowd-sourcing datasets of tools for a task to learn how to grasp and manipulate them for that task. However, these methods assume that only one object is provided and that it is possible, with the correct grasp, to perform the task; they are not capable of identifying, grasping, and using the best object for a task when many are available, especially when the optimal tool is absent. In this work, we propose GeT-USE, a two-step procedure that learns to perform real-robot generalized tool usage by learning first to extend the robot's embodiment in simulation and then transferring the learned strategies to real-robot visuomotor policies. Our key insight is that by exploring a robot's embodiment extensions (i.e., building new end-effectors) in simulation, the robot can identify the general tool geometries most beneficial for a task. This learned geometric knowledge can then be distilled to perform generalized tool usage tasks by selecting and using the best available real-world object as tool. On a real robot with 22 degrees of freedom (DOFs), GeT-USE outperforms state-of-the-art methods by 30-60% success rates across three vision-based bimanual mobile manipulation tool-usage tasks. 

**Abstract (ZH)**: 利用随机对象在通用场景下作为工具的能力是当前机器人智能中的一项缺失环节，旨在提升机器人的多功能性和问题解决能力。现有的先进机器人工具使用方法侧重于通过程序生成或众包任务相关的工具数据集，以便学习如何抓取和操控这些工具。然而，这些方法假设只提供一个对象，并且在正确抓取情况下可以完成任务；它们无法在多个对象可用时识别、抓取和使用最适合的任务对象，尤其是在最优工具缺失的情况下。在本研究中，我们提出了一种名为GeT-USE的两步方法，该方法通过首先在模拟中扩展机器人的体感，然后将学到的策略转移到真实机器人可视化运动策略中，来学习执行通用工具使用。我们的关键洞察是，通过在模拟中探索机器人的体感扩展（即构建新的末端执行器），机器人可以识别对任务最有益的一般工具几何形状。这种学到的几何知识可以被提取并用于通过选择和使用最佳可用的真实世界物体作为工具来执行通用工具使用任务。在具有22自由度的真实机器人上，GeT-USE在三项基于视觉的双臂移动操作工具使用任务中的成功率相比最先进的方法提高了30-60%。 

---
# A Humanoid Visual-Tactile-Action Dataset for Contact-Rich Manipulation 

**Title (ZH)**: 人类视觉-触觉-行动数据集：接触丰富操作 

**Authors**: Eunju Kwon, Seungwon Oh, In-Chang Baek, Yucheon Park, Gyungbo Kim, JaeYoung Moon, Yunho Choi, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.25725)  

**Abstract**: Contact-rich manipulation has become increasingly important in robot learning. However, previous studies on robot learning datasets have focused on rigid objects and underrepresented the diversity of pressure conditions for real-world manipulation. To address this gap, we present a humanoid visual-tactile-action dataset designed for manipulating deformable soft objects. The dataset was collected via teleoperation using a humanoid robot equipped with dexterous hands, capturing multi-modal interactions under varying pressure conditions. This work also motivates future research on models with advanced optimization strategies capable of effectively leveraging the complexity and diversity of tactile signals. 

**Abstract (ZH)**: 富有接触的操控在机器人学习中变得越来越重要。然而，之前的机器人学习数据集研究主要集中在刚性物体上，并未能充分代表现实世界操控中压力条件的多样性。为弥补这一不足，我们呈现了一个用于操控可变形软物体的人形视觉-触觉-动作数据集。该数据集通过装备灵巧手的人形机器人遥操作收集，捕捉了在不同压力条件下的多模态交互。此外，本研究还激励了未来能够有效利用触觉信号复杂性和多样性的模型研究。 

---
# Robotic Assistant: Completing Collaborative Tasks with Dexterous Vision-Language-Action Models 

**Title (ZH)**: 机器人助手中兼顾灵巧视觉-语言-动作模型的协作任务完成 

**Authors**: Boshi An, Chenyu Yang, Robert Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.25713)  

**Abstract**: We adapt a pre-trained Vision-Language-Action (VLA) model (Open-VLA) for dexterous human-robot collaboration with minimal language prompting. Our approach adds (i) FiLM conditioning to visual backbones for task-aware perception, (ii) an auxiliary intent head that predicts collaborator hand pose and target cues, and (iii) action-space post-processing that predicts compact deltas (position/rotation) and PCA-reduced finger joints before mapping to full commands. Using a multi-view, teleoperated Franka and Mimic-hand dataset augmented with MediaPipe hand poses, we demonstrate that delta actions are well-behaved and that four principal components explain ~96% of hand-joint variance. Ablations identify action post-processing as the primary performance driver; auxiliary intent helps, FiLM is mixed, and a directional motion loss is detrimental. A real-time stack (~0.3 s latency on one RTX 4090) composes "pick-up" and "pass" into a long-horizon behavior. We surface "trainer overfitting" to specific demonstrators as the key limitation. 

**Abstract (ZH)**: 我们通过最小的语言提示，适应一个预训练的Vision-Language-Action (VLA)模型（Open-VLA），实现灵巧的人机协作。该方法增加了(i)针对任务的视觉背部的FiLM条件,(ii)一个辅助意图头来预测合作者的手部姿态和目标提示,(iii)在动作空间后处理中预测紧凑的位移和旋转delta，以及PCA降维的手指关节，然后映射到完整的命令。通过一个多视角的远程操作Franka和Mimic-hand数据集，并用MediaPipe手部姿态进行增强，我们展示了delta动作表现良好，并且四个主成分解释了手部关节96%的方差。消融实验表明动作后处理是主要性能驱动因素；辅助意图有所帮助，FiLM的效果参差不齐，而方向性运动损失是有害的。实时堆栈 (~0.3秒延迟在一个RTX 4090上) 将“拾取”和“传递”组合成一种长远行为。我们揭示了对特定示范者的“trainer过拟合”是主要限制。 

---
# Learning to Plan & Schedule with Reinforcement-Learned Bimanual Robot Skills 

**Title (ZH)**: 学习执行双臂机器人技能的规划与调度 

**Authors**: Weikang Wan, Fabio Ramos, Xuning Yang, Caelan Garrett  

**Link**: [PDF](https://arxiv.org/pdf/2510.25634)  

**Abstract**: Long-horizon contact-rich bimanual manipulation presents a significant challenge, requiring complex coordination involving a mixture of parallel execution and sequential collaboration between arms. In this paper, we introduce a hierarchical framework that frames this challenge as an integrated skill planning & scheduling problem, going beyond purely sequential decision-making to support simultaneous skill invocation. Our approach is built upon a library of single-arm and bimanual primitive skills, each trained using Reinforcement Learning (RL) in GPU-accelerated simulation. We then train a Transformer-based planner on a dataset of skill compositions to act as a high-level scheduler, simultaneously predicting the discrete schedule of skills as well as their continuous parameters. We demonstrate that our method achieves higher success rates on complex, contact-rich tasks than end-to-end RL approaches and produces more efficient, coordinated behaviors than traditional sequential-only planners. 

**Abstract (ZH)**: 长时程多接触双臂操作 Presents a Significant Challenge, Requiring Complex Coordination Involving a Mixture of Parallel Execution and Sequential Collaboration Between Arms: A Hierarchical Framework for Integrated Skill Planning and Scheduling 

---
# Using VLM Reasoning to Constrain Task and Motion Planning 

**Title (ZH)**: 使用VLM推理来约束任务与运动规划 

**Authors**: Muyang Yan, Miras Mengdibayev, Ardon Floros, Weihang Guo, Lydia E. Kavraki, Zachary Kingston  

**Link**: [PDF](https://arxiv.org/pdf/2510.25548)  

**Abstract**: In task and motion planning, high-level task planning is done over an abstraction of the world to enable efficient search in long-horizon robotics problems. However, the feasibility of these task-level plans relies on the downward refinability of the abstraction into continuous motion. When a domain's refinability is poor, task-level plans that appear valid may ultimately fail during motion planning, requiring replanning and resulting in slower overall performance. Prior works mitigate this by encoding refinement issues as constraints to prune infeasible task plans. However, these approaches only add constraints upon refinement failure, expending significant search effort on infeasible branches. We propose VIZ-COAST, a method of leveraging the common-sense spatial reasoning of large pretrained Vision-Language Models to identify issues with downward refinement a priori, bypassing the need to fix these failures during planning. Experiments on two challenging TAMP domains show that our approach is able to extract plausible constraints from images and domain descriptions, drastically reducing planning times and, in some cases, eliminating downward refinement failures altogether, generalizing to a diverse range of instances from the broader domain. 

**Abstract (ZH)**: 基于任务和动作规划中的常识空间推理在先验识别向下细化问题的方法 

---
# Sim-to-Real Gentle Manipulation of Deformable and Fragile Objects with Stress-Guided Reinforcement Learning 

**Title (ZH)**: 基于应力导向强化学习的柔性易碎物体温和操控从仿真到现实的研究 

**Authors**: Kei Ikemura, Yifei Dong, David Blanco-Mulero, Alberta Longhini, Li Chen, Florian T. Pokorny  

**Link**: [PDF](https://arxiv.org/pdf/2510.25405)  

**Abstract**: Robotic manipulation of deformable and fragile objects presents significant challenges, as excessive stress can lead to irreversible damage to the object. While existing solutions rely on accurate object models or specialized sensors and grippers, this adds complexity and often lacks generalization. To address this problem, we present a vision-based reinforcement learning approach that incorporates a stress-penalized reward to discourage damage to the object explicitly. In addition, to bootstrap learning, we incorporate offline demonstrations as well as a designed curriculum progressing from rigid proxies to deformables. We evaluate the proposed method in both simulated and real-world scenarios, showing that the policy learned in simulation can be transferred to the real world in a zero-shot manner, performing tasks such as picking up and pushing tofu. Our results show that the learned policies exhibit a damage-aware, gentle manipulation behavior, demonstrating their effectiveness by decreasing the stress applied to fragile objects by 36.5% while achieving the task goals, compared to vanilla RL policies. 

**Abstract (ZH)**: 基于视觉的强化学习方法：通过应力惩罚奖励实现脆弱可变形物体的智能化 manipulation 

---
# SynHLMA:Synthesizing Hand Language Manipulation for Articulated Object with Discrete Human Object Interaction Representation 

**Title (ZH)**: SynHLMA: 基于离散人类物体交互表示的手部动作合成与 articulated 对象 manipulation 

**Authors**: Wang zhi, Yuyan Liu, Liu Liu, Li Zhang, Ruixuan Lu, Dan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.25268)  

**Abstract**: Generating hand grasps with language instructions is a widely studied topic that benefits from embodied AI and VR/AR applications. While transferring into hand articulatied object interaction (HAOI), the hand grasps synthesis requires not only object functionality but also long-term manipulation sequence along the object deformation. This paper proposes a novel HAOI sequence generation framework SynHLMA, to synthesize hand language manipulation for articulated objects. Given a complete point cloud of an articulated object, we utilize a discrete HAOI representation to model each hand object interaction frame. Along with the natural language embeddings, the representations are trained by an HAOI manipulation language model to align the grasping process with its language description in a shared representation space. A joint-aware loss is employed to ensure hand grasps follow the dynamic variations of articulated object joints. In this way, our SynHLMA achieves three typical hand manipulation tasks for articulated objects of HAOI generation, HAOI prediction and HAOI interpolation. We evaluate SynHLMA on our built HAOI-lang dataset and experimental results demonstrate the superior hand grasp sequence generation performance comparing with state-of-the-art. We also show a robotics grasp application that enables dexterous grasps execution from imitation learning using the manipulation sequence provided by our SynHLMA. Our codes and datasets will be made publicly available. 

**Abstract (ZH)**: 基于语言指令生成手部抓取的articulated对象手部操作序列生成框架 

---
# One-shot Humanoid Whole-body Motion Learning 

**Title (ZH)**: 单次学习人体全身动力学 

**Authors**: Hao Huang, Geeta Chandra Raju Bethala, Shuaihang Yuan, Congcong Wen, Anthony Tzes, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.25241)  

**Abstract**: Whole-body humanoid motion represents a cornerstone challenge in robotics, integrating balance, coordination, and adaptability to enable human-like behaviors. However, existing methods typically require multiple training samples per motion category, rendering the collection of high-quality human motion datasets both labor-intensive and costly. To address this, we propose a novel approach that trains effective humanoid motion policies using only a single non-walking target motion sample alongside readily available walking motions. The core idea lies in leveraging order-preserving optimal transport to compute distances between walking and non-walking sequences, followed by interpolation along geodesics to generate new intermediate pose skeletons, which are then optimized for collision-free configurations and retargeted to the humanoid before integration into a simulated environment for policy training via reinforcement learning. Experimental evaluations on the CMU MoCap dataset demonstrate that our method consistently outperforms baselines, achieving superior performance across metrics. Code will be released upon acceptance. 

**Abstract (ZH)**: 全身人形机器人运动代表了机器人领域的一个核心挑战，融合了平衡、协调和适应性，以实现类人行为。然而，现有的方法通常需要每个运动类别多个训练样本，使得高质量人体运动数据集的收集既耗费人力又成本高昂。为了解决这一问题，我们提出了一种新颖的方法，仅使用一个非行走目标运动样本和现成的行走运动样本训练有效的人形机器人运动策略。核心思想是利用保序最优传输计算行走和非行走序列之间的距离，随后沿着测地线进行插值生成新的中间姿态骨架，再优化为无碰撞配置并重新目标化到人形机器人，最终在仿真环境中通过强化学习进行策略训练。在CMU MoCap数据集上的实验评估表明，我们的方法在多个指标上均超过了基线方法，表现更优。代码将在接受后发布。 

---
# SoraNav: Adaptive UAV Task-Centric Navigation via Zeroshot VLM Reasoning 

**Title (ZH)**: SoraNav: 基于零样本VLM推理的自适应无人机任务导向导航 

**Authors**: Hongyu Song, Rishabh Dev Yadav, Cheng Guo, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.25191)  

**Abstract**: Interpreting visual observations and natural language instructions for complex task execution remains a key challenge in robotics and AI. Despite recent advances, language-driven navigation is still difficult, particularly for UAVs in small-scale 3D environments. Existing Vision-Language Navigation (VLN) approaches are mostly designed for ground robots and struggle to generalize to aerial tasks that require full 3D spatial reasoning. The emergence of large Vision-Language Models (VLMs), such as GPT and Claude, enables zero-shot semantic reasoning from visual and textual inputs. However, these models lack spatial grounding and are not directly applicable to navigation. To address these limitations, SoraNav is introduced, an adaptive UAV navigation framework that integrates zero-shot VLM reasoning with geometry-aware decision-making. Geometric priors are incorporated into image annotations to constrain the VLM action space and improve decision quality. A hybrid switching strategy leverages navigation history to alternate between VLM reasoning and geometry-based exploration, mitigating dead-ends and redundant revisits. A PX4-based hardware-software platform, comprising both a digital twin and a physical micro-UAV, enables reproducible evaluation. Experimental results show that in 2.5D scenarios, our method improves Success Rate (SR) by 25.7% and Success weighted by Path Length (SPL) by 17%. In 3D scenarios, it improves SR by 29.5% and SPL by 18.5% relative to the baseline. 

**Abstract (ZH)**: 基于视觉语言理解的复杂任务执行解析仍是在机器人技术和AI领域的一个关键挑战。尽管最近取得了一些进展，但基于语言的导航任务，尤其是在小规模3D环境中的无人机任务，仍然极具挑战性。现有的视觉-语言导航(VLN)方法主要针对地面机器人设计，难以泛化到需要全面3D空间推理的空中任务。大型视觉语言模型(VLM)，如GPT和Claude的出现，使从视觉和文本输入进行零样本语义推理成为可能。然而，这些模型缺乏空间定位，不适用于导航任务。为了克服这些限制，我们提出了SoraNav，一个将零样本VLM推理与几何感知决策相结合的自适应无人机导航框架。将几何先验整合到图像标注中，以约束VLM的动作空间并提高决策质量。一种混合切换策略利用导航历史交替进行VLM推理和基于几何的探索，在避免死胡同和重复访问方面起到了作用。该方法基于PX4的硬件-软件平台，包括数字双胞胎和物理微型无人机，使评估具有可重复性。实验结果显示，在2.5D场景中，我们的方法将成功率达到提高了25.7%，路径加权成功率达到提高了17%。在3D场景中，成功率达到提高了29.5%，路径加权成功率达到提高了18.5%，相对于基线方法。 

---
# Learning Spatial-Aware Manipulation Ordering 

**Title (ZH)**: 学习空间感知的操作顺序 

**Authors**: Yuxiang Yan, Zhiyuan Zhou, Xin Gao, Guanghao Li, Shenglin Li, Jiaqi Chen, Qunyan Pu, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2510.25138)  

**Abstract**: Manipulation in cluttered environments is challenging due to spatial dependencies among objects, where an improper manipulation order can cause collisions or blocked access. Existing approaches often overlook these spatial relationships, limiting their flexibility and scalability. To address these limitations, we propose OrderMind, a unified spatial-aware manipulation ordering framework that directly learns object manipulation priorities based on spatial context. Our architecture integrates a spatial context encoder with a temporal priority structuring module. We construct a spatial graph using k-Nearest Neighbors to aggregate geometric information from the local layout and encode both object-object and object-manipulator interactions to support accurate manipulation ordering in real-time. To generate physically and semantically plausible supervision signals, we introduce a spatial prior labeling method that guides a vision-language model to produce reasonable manipulation orders for distillation. We evaluate OrderMind on our Manipulation Ordering Benchmark, comprising 163,222 samples of varying difficulty. Extensive experiments in both simulation and real-world environments demonstrate that our method significantly outperforms prior approaches in effectiveness and efficiency, enabling robust manipulation in cluttered scenes. 

**Abstract (ZH)**: 在杂乱环境中进行操作因物体之间的空间依赖关系而具有挑战性，不恰当的操作顺序可能导致碰撞或访问受阻。现有方法往往忽略这些空间关系，限制了其灵活性和可扩展性。为了解决这些问题，我们提出了OrderMind，这是一种统一的空间感知操作顺序框架，可以直接根据空间上下文学习物体操作优先级。我们的架构结合了空间上下文编码器和时间优先级结构模块。我们使用k-最近邻构建空间图，以聚合局部布局的几何信息，并编码物体间和物体-操作者间的交互，以支持实时准确的操作顺序。为生成物理和语义上合理的监督信号，我们引入了一种空间先验标签方法，以指导视觉-语言模型生成合理的操作顺序进行蒸馏。我们在包含163,222个不同难度样本的操作顺序基准上评估了OrderMind。在仿真和实际环境中的广泛实验表明，我们的方法在有效性与效率方面显著优于现有方法，能够在杂乱场景中实现稳健的操作。 

---
# NanoVLA: Routing Decoupled Vision-Language Understanding for Nano-sized Generalist Robotic Policies 

**Title (ZH)**: NanoVLA: 用于纳米级通用机器人政策的路由解耦视觉-语言理解 

**Authors**: Jiahong Chen, Jing Wang, Long Chen, Chuwei Cai, Jinghui Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.25122)  

**Abstract**: Vision-language-action (VLA) models have significantly advanced robotic manipulation by integrating vision-language models (VLMs), and action decoders into a unified architecture. However, their deployment on resource-constrained edge devices, such as mobile robots or embedded systems (e.g., Jetson Orin Nano), remains challenging due to high computational demands, especially in real-world scenarios where power, latency, and computational resources are critical. To close this gap, we introduce Nano-scale Vision-Language Action (NanoVLA), a family of lightweight VLA architectures that achieve high performance with minimal resources. Our core innovations include: (1) vision-language decoupling that moves conventional early vision and language inputs fusion in VLM to late stage, achieving better performance while enabling caching and reduce inference overhead and latency; (2) long-short action chunking to ensure smooth, coherent multi-step planning without sacrificing real-time responsiveness; (3) dynamic routing that adaptively assigns lightweight or heavy backbones based on task complexity, further optimizing inference efficiency. Experimental results on several benchmarks, as well as real-world deployments, demonstrate that NanoVLA achieves up to 52x faster inference on edge devices compared to previous state-of-the-art VLA models, with 98% less parameters while maintaining or surpassing their task accuracy and generalization. Ablation studies confirm that our decoupling strategy preserves cross-task transferability, and the routing module enhances cost-performance trade-offs, enabling practical, high-precision robotic manipulation on resource-constrained hardware. 

**Abstract (ZH)**: Nano尺度视觉-语言-行动（NanoVLA）模型 

---
# Scalable predictive processing framework for multitask caregiving robots 

**Title (ZH)**: 可扩展的多任务 caregiving 机器人预测处理框架 

**Authors**: Hayato Idei, Tamon Miyake, Tetsuya Ogata, Yuichi Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2510.25053)  

**Abstract**: The rapid aging of societies is intensifying demand for autonomous care robots; however, most existing systems are task-specific and rely on handcrafted preprocessing, limiting their ability to generalize across diverse scenarios. A prevailing theory in cognitive neuroscience proposes that the human brain operates through hierarchical predictive processing, which underlies flexible cognition and behavior by integrating multimodal sensory signals. Inspired by this principle, we introduce a hierarchical multimodal recurrent neural network grounded in predictive processing under the free-energy principle, capable of directly integrating over 30,000-dimensional visuo-proprioceptive inputs without dimensionality reduction. The model was able to learn two representative caregiving tasks, rigid-body repositioning and flexible-towel wiping, without task-specific feature engineering. We demonstrate three key properties: (i) self-organization of hierarchical latent dynamics that regulate task transitions, capture variability in uncertainty, and infer occluded states; (ii) robustness to degraded vision through visuo-proprioceptive integration; and (iii) asymmetric interference in multitask learning, where the more variable wiping task had little influence on repositioning, whereas learning the repositioning task led to a modest reduction in wiping performance, while the model maintained overall robustness. Although the evaluation was limited to simulation, these results establish predictive processing as a universal and scalable computational principle, pointing toward robust, flexible, and autonomous caregiving robots while offering theoretical insight into the human brain's ability to achieve flexible adaptation in uncertain real-world environments. 

**Abstract (ZH)**: 社会的老龄化加剧了对自主护理机器人的需求；然而，大多数现有系统都是任务特定的，并依赖于手工设计的预处理，限制了它们在不同场景下的泛化能力。受认知神经科学中层级预测处理原理的启发，我们提出了一种基于自由能原理的层级多模态循环神经网络，该模型能够直接整合超过30,000维度的视知觉本体感受输入，而无需降低维度。该模型能够在无需特定任务特征工程的情况下学习两种代表性的护理任务：刚体重新定位和柔性毛巾擦拭。本文展示了三个关键特性：（i）层级潜在动态的自组织，调节任务转换，捕捉不确定性中的变化，并推断被遮挡的状态；（ii）通过整合视知觉本体感受应对视觉退化的稳健性；（iii）在多任务学习中的不对称干扰，在多任务学习中，更为多变的擦拭任务对重新定位几乎没有影响，而学习重新定位任务则导致擦拭性能有轻微下降，但模型整体依然保持稳健。尽管评估仅限于模拟，但这些结果确立了预测处理作为一种普遍而可扩展的计算原理，并且展示了灵活、适应性强且自主的护理机器人的可能性，同时提供了对人脑如何在不确定的现实环境中实现灵活适应的理论洞察。 

---
# Defect Mitigation for Robot Arm-based Additive Manufacturing Utilizing Intelligent Control and IOT 

**Title (ZH)**: 基于智能控制与物联网的机器人手臂增材制造缺陷 mitigation 研究 

**Authors**: Matsive Ali, Blake Gassen, Sen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.24994)  

**Abstract**: This paper presents an integrated robotic fused deposition modeling additive manufacturing system featuring closed-loop thermal control and intelligent in-situ defect correction using a 6-degree of freedom robotic arm and an Oak-D camera. The robot arm end effector was modified to mount an E3D hotend thermally regulated by an IoT microcontroller, enabling precise temperature control through real-time feedback. Filament extrusion system was synchronized with robotic motion, coordinated via ROS2, ensuring consistent deposition along complex trajectories. A vision system based on OpenCV detects layer-wise defects position, commanding autonomous re-extrusion at identified sites. Experimental validation demonstrated successful defect mitigation in printing operations. The integrated system effectively addresses challenges real-time quality assurance. Inverse kinematics were used for motion planning, while homography transformations corrected camera perspectives for accurate defect localization. The intelligent system successfully mitigated surface anomalies without interrupting the print process. By combining real-time thermal regulation, motion control, and intelligent defect detection & correction, this architecture establishes a scalable and adaptive robotic additive manufacturing framework suitable for aerospace, biomedical, and industrial applications. 

**Abstract (ZH)**: 一种集成的闭环热控和智能就地缺陷纠正的6自由度机器人融沉积增材制造系统 

---
# A Survey on Efficient Vision-Language-Action Models 

**Title (ZH)**: Efficient 视觉-语言-动作 模型综述 

**Authors**: Zhaoshu Yu, Bo Wang, Pengpeng Zeng, Haonan Zhang, Ji Zhang, Lianli Gao, Jingkuan Song, Nicu Sebe, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.24795)  

**Abstract**: Vision-Language-Action models (VLAs) represent a significant frontier in embodied intelligence, aiming to bridge digital knowledge with physical-world interaction. While these models have demonstrated remarkable generalist capabilities, their deployment is severely hampered by the substantial computational and data requirements inherent to their underlying large-scale foundation models. Motivated by the urgent need to address these challenges, this survey presents the first comprehensive review of Efficient Vision-Language-Action models (Efficient VLAs) across the entire data-model-training process. Specifically, we introduce a unified taxonomy to systematically organize the disparate efforts in this domain, categorizing current techniques into three core pillars: (1) Efficient Model Design, focusing on efficient architectures and model compression; (2) Efficient Training, which reduces computational burdens during model learning; and (3) Efficient Data Collection, which addresses the bottlenecks in acquiring and utilizing robotic data. Through a critical review of state-of-the-art methods within this framework, this survey not only establishes a foundational reference for the community but also summarizes representative applications, delineates key challenges, and charts a roadmap for future research. We maintain a continuously updated project page to track our latest developments: this https URL 

**Abstract (ZH)**: 高效的视觉-语言-动作模型（Efficient Vision-Language-Action Models）：数据-模型-训练全过程的综合回顾 

---
# TheraMind: A Strategic and Adaptive Agent for Longitudinal Psychological Counseling 

**Title (ZH)**: TheraMind: 一种战略性适应性 longitudinal 心理咨询代理 

**Authors**: He Hu, Yucheng Zhou, Chiyuan Ma, Qianning Wang, Zheng Zhang, Fei Ma, Laizhong Cui, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2510.25758)  

**Abstract**: Large language models (LLMs) in psychological counseling have attracted increasing attention. However, existing approaches often lack emotional understanding, adaptive strategies, and the use of therapeutic methods across multiple sessions with long-term memory, leaving them far from real clinical practice. To address these critical gaps, we introduce TheraMind, a strategic and adaptive agent for longitudinal psychological counseling. The cornerstone of TheraMind is a novel dual-loop architecture that decouples the complex counseling process into an Intra-Session Loop for tactical dialogue management and a Cross-Session Loop for strategic therapeutic planning. The Intra-Session Loop perceives the patient's emotional state to dynamically select response strategies while leveraging cross-session memory to ensure continuity. Crucially, the Cross-Session Loop empowers the agent with long-term adaptability by evaluating the efficacy of the applied therapy after each session and adjusting the method for subsequent interactions. We validate our approach in a high-fidelity simulation environment grounded in real clinical cases. Extensive evaluations show that TheraMind outperforms other methods, especially on multi-session metrics like Coherence, Flexibility, and Therapeutic Attunement, validating the effectiveness of its dual-loop design in emulating strategic, adaptive, and longitudinal therapeutic behavior. The code is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型在心理辅导中的应用引起了越来越多的关注。然而，现有方法往往缺乏情感理解、适应性策略以及多疗程长期记忆下的治疗方法使用，使其难以达到临床实践的标准。为填补这些关键空白，我们引入了TheraMind，这是一种适用于纵向心理辅导的战略性和适应性代理。TheraMind的核心是一个新颖的双环架构，将复杂的咨询过程分解为在会话内部进行战术对话管理的会话内环和在会话之间进行战略性治疗规划的跨会话环。会话内环感知患者的情感状态，动态选择响应策略，同时利用跨会话记忆确保连续性。最关键的是，跨会话环通过每会话评估所应用疗法的有效性并调整后续交互的方法，赋予代理长期适应性。我们通过基于真实临床案例的高保真模拟环境验证了该方法。广泛的评估表明，TheraMind在包括一致性和灵活性等多会话指标上优于其他方法，验证了其双环设计在模拟战略性、适应性和纵向治疗行为方面的有效性。代码已公开，可通过以下链接访问：this https URL。 

---
# Navigation in a Three-Dimensional Urban Flow using Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的三维城市流导航 

**Authors**: Federica Tonti, Ricardo Vinuesa  

**Link**: [PDF](https://arxiv.org/pdf/2510.25679)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly populating urban areas for delivery and surveillance purposes. In this work, we develop an optimal navigation strategy based on Deep Reinforcement Learning. The environment is represented by a three-dimensional high-fidelity simulation of an urban flow, characterized by turbulence and recirculation zones. The algorithm presented here is a flow-aware Proximal Policy Optimization (PPO) combined with a Gated Transformer eXtra Large (GTrXL) architecture, giving the agent richer information about the turbulent flow field in which it navigates. The results are compared with a PPO+GTrXL without the secondary prediction tasks, a PPO combined with Long Short Term Memory (LSTM) cells and a traditional navigation algorithm. The obtained results show a significant increase in the success rate (SR) and a lower crash rate (CR) compared to a PPO+LSTM, PPO+GTrXL and the classical Zermelo's navigation algorithm, paving the way to a completely reimagined UAV landscape in complex urban environments. 

**Abstract (ZH)**: 无人 aerial 车（UAVs）越来越多地部署在城市区域以实现配送和 surveillance 目的。本研究开发了一种基于深度强化学习的优化导航策略。环境通过一个包含湍流和循环区的三维高保真城市流模拟来表示。本研究提出的方法结合了流体感知的近端策略优化（PPO）与门控变压器超大（GTrXL）架构，使代理能够获得其导航的湍流流场中的更丰富信息。与未包含次要预测任务的PPO+GTrXL、与长短期记忆（LSTM）细胞结合的PPO以及传统的导航算法相比，所获得的结果显示了显著提高的成功率（SR）和较低的碰撞率（CR），为复杂城市环境中的完全重塑的UAV景观铺平了道路。 

---
# Off-policy Reinforcement Learning with Model-based Exploration Augmentation 

**Title (ZH)**: 基于模型的探索增强的离策强化学习 

**Authors**: Likun Wang, Xiangteng Zhang, Yinuo Wang, Guojian Zhan, Wenxuan Wang, Haoyu Gao, Jingliang Duan, Shengbo Eben Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.25529)  

**Abstract**: Exploration is fundamental to reinforcement learning (RL), as it determines how effectively an agent discovers and exploits the underlying structure of its environment to achieve optimal performance. Existing exploration methods generally fall into two categories: active exploration and passive exploration. The former introduces stochasticity into the policy but struggles in high-dimensional environments, while the latter adaptively prioritizes transitions in the replay buffer to enhance exploration, yet remains constrained by limited sample diversity. To address the limitation in passive exploration, we propose Modelic Generative Exploration (MoGE), which augments exploration through the generation of under-explored critical states and synthesis of dynamics-consistent experiences through transition models. MoGE is composed of two components: (1) a diffusion-based generator that synthesizes critical states under the guidance of a utility function evaluating each state's potential influence on policy exploration, and (2) a one-step imagination world model for constructing critical transitions based on the critical states for agent learning. Our method adopts a modular formulation that aligns with the principles of off-policy learning, allowing seamless integration with existing algorithms to improve exploration without altering their core structures. Empirical results on OpenAI Gym and DeepMind Control Suite reveal that MoGE effectively bridges exploration and policy learning, leading to remarkable gains in both sample efficiency and performance across complex control tasks. 

**Abstract (ZH)**: 模型生成探索：基于生成模型的探索方法（MoGE） 

---
# Agentic AI: A Comprehensive Survey of Architectures, Applications, and Future Directions 

**Title (ZH)**: 代理型AI：架构、应用及未来方向综述 

**Authors**: Mohamad Abou Ali, Fadi Dornaika  

**Link**: [PDF](https://arxiv.org/pdf/2510.25445)  

**Abstract**: Agentic AI represents a transformative shift in artificial intelligence, but its rapid advancement has led to a fragmented understanding, often conflating modern neural systems with outdated symbolic models -- a practice known as conceptual retrofitting. This survey cuts through this confusion by introducing a novel dual-paradigm framework that categorizes agentic systems into two distinct lineages: the Symbolic/Classical (relying on algorithmic planning and persistent state) and the Neural/Generative (leveraging stochastic generation and prompt-driven orchestration). Through a systematic PRISMA-based review of 90 studies (2018--2025), we provide a comprehensive analysis structured around this framework across three dimensions: (1) the theoretical foundations and architectural principles defining each paradigm; (2) domain-specific implementations in healthcare, finance, and robotics, demonstrating how application constraints dictate paradigm selection; and (3) paradigm-specific ethical and governance challenges, revealing divergent risks and mitigation strategies. Our analysis reveals that the choice of paradigm is strategic: symbolic systems dominate safety-critical domains (e.g., healthcare), while neural systems prevail in adaptive, data-rich environments (e.g., finance). Furthermore, we identify critical research gaps, including a significant deficit in governance models for symbolic systems and a pressing need for hybrid neuro-symbolic architectures. The findings culminate in a strategic roadmap arguing that the future of Agentic AI lies not in the dominance of one paradigm, but in their intentional integration to create systems that are both adaptable and reliable. This work provides the essential conceptual toolkit to guide future research, development, and policy toward robust and trustworthy hybrid intelligent systems. 

**Abstract (ZH)**: 代理型AI代表了人工智能领域的转型性变革，但其迅速发展导致了理解的碎片化，常将现代神经系统与过时的符号模型混为一谈——这一做法被称为概念性重塑。本文通过介绍一种新的双范式框架，澄清了这一混淆，该框架将代理系统划分为两种截然不同的谱系：符号/古典（依赖于算法规划和持续状态）和神经/生成（利用随机生成和提示驱动的协调）。通过对2018年至2025年间90篇研究论文的系统性PRISMA审查，本文从三个维度对该框架进行了全面分析：（1）定义每种范式的理论基础和架构原则；（2）在医疗、金融和机器人技术等具体领域的应用，展示应用约束如何决定范式选择；（3）特定范式下的伦理和治理挑战，揭示了不同风险及其缓解策略。分析显示，范式选择具有战略意义：符号系统在安全关键领域（如医疗）占主导地位，而神经系统在适应性强、数据丰富的环境中占据优势（如金融）。此外，本文还指出了关键的研究空白，包括符号系统治理模型的显著缺陷和亟待开发的混合神经-符号架构。研究结果表明，代理型AI的未来不在于单一范式的主导地位，而在于故意集成这些范式，以创造既适应性强又可靠的系统。这项工作提供了必要的概念工具箱，以指导未来的研究、开发和政策制定，向着稳健和可信赖的混合智能系统方向前进。 

---
# GAP: Graph-Based Agent Planning with Parallel Tool Use and Reinforcement Learning 

**Title (ZH)**: 基于图的代理规划：并行工具使用与强化学习 

**Authors**: Jiaqi Wu, Qinlao Zhao, Zefeng Chen, Kai Qin, Yifei Zhao, Xueqian Wang, Yuhang Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.25320)  

**Abstract**: Autonomous agents powered by large language models (LLMs) have shown impressive capabilities in tool manipulation for complex task-solving. However, existing paradigms such as ReAct rely on sequential reasoning and execution, failing to exploit the inherent parallelism among independent sub-tasks. This sequential bottleneck leads to inefficient tool utilization and suboptimal performance in multi-step reasoning scenarios. We introduce Graph-based Agent Planning (GAP), a novel framework that explicitly models inter-task dependencies through graph-based planning to enable adaptive parallel and serial tool execution. Our approach trains agent foundation models to decompose complex tasks into dependency-aware sub-task graphs, autonomously determining which tools can be executed in parallel and which must follow sequential dependencies. This dependency-aware orchestration achieves substantial improvements in both execution efficiency and task accuracy. To train GAP, we construct a high-quality dataset of graph-based planning traces derived from the Multi-Hop Question Answering (MHQA) benchmark. We employ a two-stage training strategy: supervised fine-tuning (SFT) on the curated dataset, followed by reinforcement learning (RL) with a correctness-based reward function on strategically sampled queries where tool-based reasoning provides maximum value. Experimental results on MHQA datasets demonstrate that GAP significantly outperforms traditional ReAct baselines, particularly on multi-step retrieval tasks, while achieving dramatic improvements in tool invocation efficiency through intelligent parallelization. The project page is available at: this https URL. 

**Abstract (ZH)**: 受大规模语言模型驱动的自主代理 Powered 通过图基规划实现任务间依赖的自适应并行与串行工具执行 

---
# Energy-Efficient Autonomous Driving with Adaptive Perception and Robust Decision 

**Title (ZH)**: 自适应感知与稳健决策的节能自主驾驶 

**Authors**: Yuyang Xia, Zibo Liang, Liwei Deng, Yan Zhao, Han Su, Kai Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.25205)  

**Abstract**: Autonomous driving is an emerging technology that is expected to bring significant social, economic, and environmental benefits. However, these benefits come with rising energy consumption by computation engines, limiting the driving range of vehicles, especially electric ones. Perception computing is typically the most power-intensive component, as it relies on largescale deep learning models to extract environmental features. Recently, numerous studies have employed model compression techniques, such as sparsification, quantization, and distillation, to reduce computational consumption. However, these methods often result in either a substantial model size or a significant drop in perception accuracy compared to high-computation models. To address these challenges, we propose an energy-efficient autonomous driving framework, called EneAD. In the adaptive perception module, a perception optimization strategy is designed from the perspective of data management and tuning. Firstly, we manage multiple perception models with different computational consumption and adjust the execution framerate dynamically. Then, we define them as knobs and design a transferable tuning method based on Bayesian optimization to identify promising knob values that achieve low computation while maintaining desired accuracy. To adaptively switch the knob values in various traffic scenarios, a lightweight classification model is proposed to distinguish the perception difficulty in different scenarios. In the robust decision module, we propose a decision model based on reinforcement learning and design a regularization term to enhance driving stability in the face of perturbed perception results. Extensive experiments evidence the superiority of our framework in both energy consumption and driving performance. EneAD can reduce perception consumption by 1.9x to 3.5x and thus improve driving range by 3.9% to 8.5% 

**Abstract (ZH)**: 自主驾驶是一种新兴技术，预期将带来显著的社会、经济和环境效益。然而，这些效益伴随着计算引擎能耗的增加，限制了车辆的行驶里程，尤其是电动车。感知计算通常是能耗最大的组件，因为它依赖大规模的深度学习模型来提取环境特征。最近，许多研究采用了模型压缩技术，如稀疏化、量化和蒸馏，以减少计算消耗。然而，这些方法往往会导致模型大小显著增加或感知准确性显著下降。为应对这些挑战，我们提出了一种能效自主驾驶框架，称为EneAD。在自适应感知模块中，我们从数据管理和调优的角度设计了一种感知优化策略。首先，我们管理具有不同计算消耗的多种感知模型，并动态调整执行帧率。然后，我们将这些模型定义为旋钮，并基于贝叶斯优化设计了一种可转移的调优方法，以识别能够在低计算消耗的同时保持所需准确性的优质旋钮值。为了根据不同交通场景自适应切换旋钮值，我们提出了一种轻量级分类模型来区分不同场景下的感知难度。在稳健决策模块中，我们基于强化学习提出了一种决策模型，并设计了一个正则化项以在感知结果错位时增强驾驶稳定性。广泛的实验证据表明，我们的框架在能耗和驾驶性能方面具有优越性。EneAD可以将感知消耗降低1.9至3.5倍，从而提高3.9%至8.5%的行驶里程。 

---
# Agentic Moderation: Multi-Agent Design for Safer Vision-Language Models 

**Title (ZH)**: 代理调节：多代理设计以构建更安全的视觉语言模型 

**Authors**: Juan Ren, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2510.25179)  

**Abstract**: Agentic methods have emerged as a powerful and autonomous paradigm that enhances reasoning, collaboration, and adaptive control, enabling systems to coordinate and independently solve complex tasks. We extend this paradigm to safety alignment by introducing Agentic Moderation, a model-agnostic framework that leverages specialised agents to defend multimodal systems against jailbreak attacks. Unlike prior approaches that apply as a static layer over inputs or outputs and provide only binary classifications (safe or unsafe), our method integrates dynamic, cooperative agents, including Shield, Responder, Evaluator, and Reflector, to achieve context-aware and interpretable moderation. Extensive experiments across five datasets and four representative Large Vision-Language Models (LVLMs) demonstrate that our approach reduces the Attack Success Rate (ASR) by 7-19%, maintains a stable Non-Following Rate (NF), and improves the Refusal Rate (RR) by 4-20%, achieving robust, interpretable, and well-balanced safety performance. By harnessing the flexibility and reasoning capacity of agentic architectures, Agentic Moderation provides modular, scalable, and fine-grained safety enforcement, highlighting the broader potential of agentic systems as a foundation for automated safety governance. 

**Abstract (ZH)**: 代理方法已成为一种强大且自主的范式，增强了推理、协作和自适应控制能力，使系统能够协调并独立解决复杂任务。我们通过引入代理调节来扩展这一范式，这是一种模型无关的框架，利用专门的代理来防御多模态系统免受逃逸攻击。与之前仅在输入或输出上作为静态层应用并仅提供二元分类（安全或不安全）的方法不同，我们的方法整合了动态、协作的代理，包括Shield、Responder、Evaluator和Reflector，以实现上下文感知和可解释的调节。我们在五个数据集和四个代表性大视觉-语言模型上的广泛实验表明，我们的方法将攻击成功率（ASR）降低了7-19%，保持了稳定的非跟随率（NF），并将拒绝率（RR）提高了4-20%，实现了稳健、可解释且均衡的安全性能。通过利用代理架构的灵活性和推理能力，代理调节提供了模块化、可扩展和精细的安全执行，突显了代理系统作为自动安全治理基础的更广泛潜力。 

---
# H3M-SSMoEs: Hypergraph-based Multimodal Learning with LLM Reasoning and Style-Structured Mixture of Experts 

**Title (ZH)**: 基于超图的多模态学习与LLM推理及风格结构化专家混合模型 

**Authors**: Peilin Tan, Liang Xie, Churan Zhi, Dian Tu, Chuanqi Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.25091)  

**Abstract**: Stock movement prediction remains fundamentally challenging due to complex temporal dependencies, heterogeneous modalities, and dynamically evolving inter-stock relationships. Existing approaches often fail to unify structural, semantic, and regime-adaptive modeling within a scalable framework. This work introduces H3M-SSMoEs, a novel Hypergraph-based MultiModal architecture with LLM reasoning and Style-Structured Mixture of Experts, integrating three key innovations: (1) a Multi-Context Multimodal Hypergraph that hierarchically captures fine-grained spatiotemporal dynamics via a Local Context Hypergraph (LCH) and persistent inter-stock dependencies through a Global Context Hypergraph (GCH), employing shared cross-modal hyperedges and Jensen-Shannon Divergence weighting mechanism for adaptive relational learning and cross-modal alignment; (2) a LLM-enhanced reasoning module, which leverages a frozen large language model with lightweight adapters to semantically fuse and align quantitative and textual modalities, enriching representations with domain-specific financial knowledge; and (3) a Style-Structured Mixture of Experts (SSMoEs) that combines shared market experts and industry-specialized experts, each parameterized by learnable style vectors enabling regime-aware specialization under sparse activation. Extensive experiments on three major stock markets demonstrate that H3M-SSMoEs surpasses state-of-the-art methods in both superior predictive accuracy and investment performance, while exhibiting effective risk control. Datasets, source code, and model weights are available at our GitHub repository: this https URL. 

**Abstract (ZH)**: 基于超图的多模态H3M-SSMoEs架构：结合LLM推理和风格结构化专家以预测股票市场动向 

---
# FARSIQA: Faithful and Advanced RAG System for Islamic Question Answering 

**Title (ZH)**: FARSIQA：忠实且先进的基于 Retrieval-Augmented Generation 的伊斯兰教问题回答系统 

**Authors**: Mohammad Aghajani Asl, Behrooz Minaei Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2510.25621)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized Natural Language Processing, yet their application in high-stakes, specialized domains like religious question answering is hindered by challenges like hallucination and unfaithfulness to authoritative sources. This issue is particularly critical for the Persian-speaking Muslim community, where accuracy and trustworthiness are paramount. Existing Retrieval-Augmented Generation (RAG) systems, relying on simplistic single-pass pipelines, fall short on complex, multi-hop queries requiring multi-step reasoning and evidence aggregation. To address this gap, we introduce FARSIQA, a novel, end-to-end system for Faithful Advanced Question Answering in the Persian Islamic domain. FARSIQA is built upon our innovative FAIR-RAG architecture: a Faithful, Adaptive, Iterative Refinement framework for RAG. FAIR-RAG employs a dynamic, self-correcting process: it adaptively decomposes complex queries, assesses evidence sufficiency, and enters an iterative loop to generate sub-queries, progressively filling information gaps. Operating on a curated knowledge base of over one million authoritative Islamic documents, FARSIQA demonstrates superior performance. Rigorous evaluation on the challenging IslamicPCQA benchmark shows state-of-the-art performance: the system achieves a remarkable 97.0% in Negative Rejection - a 40-point improvement over baselines - and a high Answer Correctness score of 74.3%. Our work establishes a new standard for Persian Islamic QA and validates that our iterative, adaptive architecture is crucial for building faithful, reliable AI systems in sensitive domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现已革新了自然语言处理领域，但在宗教问答等高 stakes、专业化领域中的应用受到了幻觉和对权威源不忠实等问题的阻碍。这一点在讲波斯语的穆斯林社区尤其关键，因为准确性与可信度至关重要。现有的检索增强生成（RAG）系统依赖于简单的单流程管道，在处理需要多步推理和证据聚合的复杂、多跳查询时表现不佳。为了解决这一差距，我们提出了FARSIQA，这是一个针对波斯伊斯兰领域忠实高级问答的新颖端到端系统。FARSIQA基于我们创新的FAIR-RAG架构：一种忠实、自适应迭代完善框架。FAIR-RAG采用了一种动态的自我校正过程：它能够自适应地分解复杂查询、评估证据充分性，并进入迭代循环生成子查询，逐步填补信息缺口。FARSIQA在超过一百万份权威伊斯兰文档整理的知识库上运行，展示了卓越的表现。在具有挑战性的伊斯兰PCQA基准测试上的严格评估显示了最先进的性能：系统在Negative Rejection方面的得分为97.0%，比基线提高了40分，并且答案正确性得分为74.3%。我们的工作确立了波斯伊斯兰问答的新标准，并验证了迭代、自适应架构对于构建敏感领域中的忠实可靠AI系统的至关重要性。 

---
# RLMEval: Evaluating Research-Level Neural Theorem Proving 

**Title (ZH)**: RLMEval: 评估研究级神经定理证明 

**Authors**: Auguste Poiroux, Antoine Bosselut, Viktor Kunčak  

**Link**: [PDF](https://arxiv.org/pdf/2510.25427)  

**Abstract**: Despite impressive results on curated benchmarks, the practical impact of large language models (LLMs) on research-level neural theorem proving and proof autoformalization is still limited. We introduce RLMEval, an evaluation suite for these tasks, focusing on research-level mathematics from real-world Lean formalization projects. RLMEval targets the evaluation of neural theorem proving and proof autoformalization on challenging research-level theorems by leveraging real Lean Blueprint formalization projects. Our evaluation of state-of-the-art models on RLMEval, comprising 613 theorems from 6 Lean projects, reveals a significant gap: progress on existing benchmarks does not readily translate to these more realistic settings, with the best model achieving only a 10.3 % pass rate. RLMEval provides a new, challenging benchmark designed to guide and accelerate progress in automated reasoning for formal mathematics. 

**Abstract (ZH)**: 尽管在精心策划的基准测试上取得了令人印象深刻的成果，大规模语言模型在研究级别的神经定理证明和证明自形式化方面的实际影响仍然有限。我们引入了RLMEval，一个针对这些任务的评估套件，专注于来自真实世界Lean形式化项目的实际研究级数学。RLMEval通过利用真实的Lean蓝图形式化项目，旨在评估神经定理证明和证明自形式化在更具挑战性的研究级定理上的表现。我们在RLMEval上的评估涵盖了来自6个Lean项目的613个定理，揭示了一个显著的差距：现有基准测试上的进展并不容易转化为这些更为现实的设置，最好的模型仅实现了10.3%的通过率。RLMEval提供了一个新的、具有挑战性的基准，旨在引导和加速形式化数学中的自动推理研究。 

---
# Adaptive End-to-End Transceiver Design for NextG Pilot-Free and CP-Free Wireless Systems 

**Title (ZH)**: 面向NextG的无 pilot 和无循环前缀的无线系统自适应端到端收发机设计 

**Authors**: Jiaming Cheng, Wei Chen, Bo Ai  

**Link**: [PDF](https://arxiv.org/pdf/2510.25416)  

**Abstract**: The advent of artificial intelligence (AI)-native wireless communication is fundamentally reshaping the design paradigm of next-generation (NextG) systems, where intelligent air interfaces are expected to operate adaptively and efficiently in highly dynamic environments. Conventional orthogonal frequency division multiplexing (OFDM) systems rely heavily on pilots and the cyclic prefix (CP), resulting in significant overhead and reduced spectral efficiency. To address these limitations, we propose an adaptive end-to-end (E2E) transceiver architecture tailored for pilot-free and CP-free wireless systems. The architecture combines AI-driven constellation shaping and a neural receiver through joint training. To enhance robustness against mismatched or time-varying channel conditions, we introduce a lightweight channel adapter (CA) module, which enables rapid adaptation with minimal computational overhead by updating only the CA parameters. Additionally, we present a framework that is scalable to multiple modulation orders within a unified model, significantly reducing model storage requirements. Moreover, to tackle the high peak-to-average power ratio (PAPR) inherent to OFDM, we incorporate constrained E2E training, achieving compliance with PAPR targets without additional transmission overhead. Extensive simulations demonstrate that the proposed framework delivers superior bit error rate (BER), throughput, and resilience across diverse channel scenarios, highlighting its potential for AI-native NextG. 

**Abstract (ZH)**: AI原生无线通信的出现从根本上重塑了下一代（NextG）系统的设计范式，其中智能空中接口预计能在高度动态的环境中实现自适应和高效的运行。传统的正交频分复用（OFDM）系统严重依赖于训练序列和循环前缀（CP），导致了较大的开销并降低了频谱效率。为了应对这些局限性，我们提出了一种针对无训练序列和无循环前缀无线系统的自适应端到端（E2E）收发机架构。该架构结合了基于AI的星座整形和神经接收器，并通过联合训练实现。为了增强对失配或时变信道条件的鲁棒性，我们引入了一个轻量级信道适配器（CA）模块，该模块通过仅更新CA参数来实现快速适应并减少计算开销。此外，我们提出了一种框架，可以在统一模型中扩展到多种调制阶数，显著降低了模型存储需求。此外，为了应对OFDM固有的高峰均功率比（PAPR），我们引入了受限的端到端训练，实现了PAPR目标的符合性，而无需额外的传输开销。广泛仿真实验表明，所提出的框架在各种信道场景中提供了更优的误比特率（BER）、吞吐量和鲁棒性，突显了其在AI原生NextG中的潜力。 

---
