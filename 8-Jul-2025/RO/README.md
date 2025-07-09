# Action Space Reduction Strategies for Reinforcement Learning in Autonomous Driving 

**Title (ZH)**: 自主驾驶中强化学习的动作空间缩减策略 

**Authors**: Elahe Delavari, Feeza Khan Khanzada, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2507.05251)  

**Abstract**: Reinforcement Learning (RL) offers a promising framework for autonomous driving by enabling agents to learn control policies through interaction with environments. However, large and high-dimensional action spaces often used to support fine-grained control can impede training efficiency and increase exploration costs. In this study, we introduce and evaluate two novel structured action space modification strategies for RL in autonomous driving: dynamic masking and relative action space reduction. These approaches are systematically compared against fixed reduction schemes and full action space baselines to assess their impact on policy learning and performance. Our framework leverages a multimodal Proximal Policy Optimization agent that processes both semantic image sequences and scalar vehicle states. The proposed dynamic and relative strategies incorporate real-time action masking based on context and state transitions, preserving action consistency while eliminating invalid or suboptimal choices. Through comprehensive experiments across diverse driving routes, we show that action space reduction significantly improves training stability and policy performance. The dynamic and relative schemes, in particular, achieve a favorable balance between learning speed, control precision, and generalization. These findings highlight the importance of context-aware action space design for scalable and reliable RL in autonomous driving tasks. 

**Abstract (ZH)**: 强化学习（RL）通过使代理通过与环境的交互来学习控制策略，为自主驾驶提供了有前途的框架。然而，用于支持精细控制的大型和高维动作空间往往会妨碍训练效率并增加探索成本。在本研究中，我们引入并评估了两种新型结构化动作空间修改策略在自主驾驶中的应用：动态掩蔽和相对动作空间缩减。这些方法与固定的缩减方案和全动作空间基线进行了系统比较，以评估其对策略学习和性能的影响。我们的框架利用一个多模态的接近策略优化（Proximal Policy Optimization）代理，该代理处理语义图像序列和标量车辆状态。所提出的动态和相对策略基于上下文和状态转换进行实时动作掩蔽，保持动作一致性的同时消除无效或次优的选择。通过在多种驾驶路线上的全面实验，我们表明动作空间缩减显著提高了训练稳定性和策略性能。特别是动态和相对方案在学习速度、控制精度和泛化能力之间取得了良好的平衡。这些发现突显了上下文感知的动作空间设计对于自主驾驶任务中可扩展且可靠的强化学习的重要性。 

---
# StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling 

**Title (ZH)**: StreamVLN: 基于慢快速上下文建模的流式视觉-语言导航 

**Authors**: Meng Wei, Chenyang Wan, Xiqian Yu, Tai Wang, Yuqiang Yang, Xiaohan Mao, Chenming Zhu, Wenzhe Cai, Hanqing Wang, Yilun Chen, Xihui Liu, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05240)  

**Abstract**: Vision-and-Language Navigation (VLN) in real-world settings requires agents to process continuous visual streams and generate actions with low latency grounded in language instructions. While Video-based Large Language Models (Video-LLMs) have driven recent progress, current VLN methods based on Video-LLM often face trade-offs among fine-grained visual understanding, long-term context modeling and computational efficiency. We introduce StreamVLN, a streaming VLN framework that employs a hybrid slow-fast context modeling strategy to support multi-modal reasoning over interleaved vision, language and action inputs. The fast-streaming dialogue context facilitates responsive action generation through a sliding-window of active dialogues, while the slow-updating memory context compresses historical visual states using a 3D-aware token pruning strategy. With this slow-fast design, StreamVLN achieves coherent multi-turn dialogue through efficient KV cache reuse, supporting long video streams with bounded context size and inference cost. Experiments on VLN-CE benchmarks demonstrate state-of-the-art performance with stable low latency, ensuring robustness and efficiency in real-world deployment. The project page is: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 实时光学-语言导航（VLN）在现实场景中要求代理处理连续的视觉流并基于语言指令以低延迟生成动作。虽然基于视频的大语言模型（Video-LLM）推动了近期的发展，但当前基于Video-LLM的VLN方法经常在细粒度的视觉理解、长期上下文建模和计算效率之间存在权衡。我们引入了StreamVLN，这是一种流式VLN框架，采用混合慢速-快速上下文建模策略，支持交错的视觉、语言和动作输入的多模态推理。快速流式对话上下文通过滑动窗口中的活跃对话促进响应式动作生成，而缓慢更新的记忆上下文则通过3D意识的token剪枝策略压缩历史视觉状态。利用这种慢速-快速设计，StreamVLN通过高效的关键值缓存重用实现一致的多轮对话，支持具有限定上下文大小和推理成本的长视频流。在VLN-CE基准测试中的实验展示了最先进的性能，并保证了在实际部署中的稳健性和效率。项目页面：\[this https URL\]。 

---
# NavigScene: Bridging Local Perception and Global Navigation for Beyond-Visual-Range Autonomous Driving 

**Title (ZH)**: NavigScene: 连接局部感知与全局导航以实现超视距自动驾驶 

**Authors**: Qucheng Peng, Chen Bai, Guoxiang Zhang, Bo Xu, Xiaotong Liu, Xiaoyin Zheng, Chen Chen, Cheng Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05227)  

**Abstract**: Autonomous driving systems have made significant advances in Q&A, perception, prediction, and planning based on local visual information, yet they struggle to incorporate broader navigational context that human drivers routinely utilize. We address this critical gap between local sensor data and global navigation information by proposing NavigScene, an auxiliary navigation-guided natural language dataset that simulates a human-like driving environment within autonomous driving systems. Moreover, we develop three complementary paradigms to leverage NavigScene: (1) Navigation-guided Reasoning, which enhances vision-language models by incorporating navigation context into the prompting approach; (2) Navigation-guided Preference Optimization, a reinforcement learning method that extends Direct Preference Optimization to improve vision-language model responses by establishing preferences for navigation-relevant summarized information; and (3) Navigation-guided Vision-Language-Action model, which integrates navigation guidance and vision-language models with conventional driving models through feature fusion. Extensive experiments demonstrate that our approaches significantly improve performance across perception, prediction, planning, and question-answering tasks by enabling reasoning capabilities beyond visual range and improving generalization to diverse driving scenarios. This work represents a significant step toward more comprehensive autonomous driving systems capable of navigating complex, unfamiliar environments with greater reliability and safety. 

**Abstract (ZH)**: 自主驾驶系统基于局部视觉信息在问答、感知、预测和规划方面取得了显著进展，但难以整合人类驾驶员常用的更广泛的导航上下文。我们通过提出NavigScene辅助导航引导自然语言数据集来弥补局部传感器数据与全局导航信息之间的关键差距，该数据集在自主驾驶系统中模拟了类似人类驾驶的环境。此外，我们开发了三种互补的范式来利用NavigScene：（1）导航引导的推理，通过将导航上下文纳入提示方法来增强视觉语言模型；（2）导航引导的偏好优化，一种强化学习方法，扩展了直接偏好优化，通过建立与导航相关的信息偏好来改善视觉语言模型的响应；（3）导航引导的视觉语言行动模型，通过特征融合将导航引导和视觉语言模型与传统的驾驶模型集成。大量实验表明，我们的方法通过增强推理能力并提高对多样化驾驶场景的泛化能力，在感知、预测、规划和问答任务中显著提高了性能。这项工作代表了朝着更加全面的自主驾驶系统迈出了重要一步，这些系统能够在更可靠和安全的情况下导航复杂且不熟悉的环境。 

---
# EmbodieDreamer: Advancing Real2Sim2Real Transfer for Policy Training via Embodied World Modeling 

**Title (ZH)**: EmbodieDreamer: 通过具身世界建模促进从现实到模拟再到现实的策略训练传输 

**Authors**: Boyuan Wang, Xinpan Meng, Xiaofeng Wang, Zheng Zhu, Angen Ye, Yang Wang, Zhiqin Yang, Chaojun Ni, Guan Huang, Xingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05198)  

**Abstract**: The rapid advancement of Embodied AI has led to an increasing demand for large-scale, high-quality real-world data. However, collecting such embodied data remains costly and inefficient. As a result, simulation environments have become a crucial surrogate for training robot policies. Yet, the significant Real2Sim2Real gap remains a critical bottleneck, particularly in terms of physical dynamics and visual appearance. To address this challenge, we propose EmbodieDreamer, a novel framework that reduces the Real2Sim2Real gap from both the physics and appearance perspectives. Specifically, we propose PhysAligner, a differentiable physics module designed to reduce the Real2Sim physical gap. It jointly optimizes robot-specific parameters such as control gains and friction coefficients to better align simulated dynamics with real-world observations. In addition, we introduce VisAligner, which incorporates a conditional video diffusion model to bridge the Sim2Real appearance gap by translating low-fidelity simulated renderings into photorealistic videos conditioned on simulation states, enabling high-fidelity visual transfer. Extensive experiments validate the effectiveness of EmbodieDreamer. The proposed PhysAligner reduces physical parameter estimation error by 3.74% compared to simulated annealing methods while improving optimization speed by 89.91\%. Moreover, training robot policies in the generated photorealistic environment leads to a 29.17% improvement in the average task success rate across real-world tasks after reinforcement learning. Code, model and data will be publicly available. 

**Abstract (ZH)**: 基于实体的AI的迅速发展导致了对大规模高质量现实世界数据的日益需求。然而，收集此类实体数据仍然成本高昂且效率低下。因此，仿真环境已成为训练机器人策略的关键替代方案。然而，现实到仿真再到现实的巨大差距仍然是一个关键瓶颈，尤其是在物理动态和视觉外观方面。为了应对这一挑战，我们提出了EmbodieDreamer这一新颖框架，从物理和视觉两个方面减少现实到仿真再到现实的差距。具体而言，我们提出了PhysAligner，这是一种差分物理模块，旨在减少仿真实际的物理差距。它共同优化了控制增益和摩擦系数等机器人特定参数，以更好地使仿真动力学与现实世界观察结果一致。此外，我们引入了VisAligner，它结合了条件视频扩散模型，通过将低保真度的仿真渲染转换为基于仿真状态的逼真视频，来弥合仿真到现实的视觉差距，实现高质量的视觉转移。广泛实验验证了EmbodieDreamer的有效性。所提出的PhysAligner与模拟退火方法相比，将物理参数估计误差减少了3.74%，同时将优化速度提高了89.91%。此外，在生成的逼真环境中训练机器人策略，经强化学习后，实际任务的平均成功率提高了29.17%。代码、模型和数据将公开提供。 

---
# LERa: Replanning with Visual Feedback in Instruction Following 

**Title (ZH)**: LERa：基于视觉反馈的指令跟随重规划 

**Authors**: Svyatoslav Pchelintsev, Maxim Patratskiy, Anatoly Onishchenko, Alexandr Korchemnyi, Aleksandr Medvedev, Uliana Vinogradova, Ilya Galuzinsky, Aleksey Postnikov, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2507.05135)  

**Abstract**: Large Language Models are increasingly used in robotics for task planning, but their reliance on textual inputs limits their adaptability to real-world changes and failures. To address these challenges, we propose LERa - Look, Explain, Replan - a Visual Language Model-based replanning approach that utilizes visual feedback. Unlike existing methods, LERa requires only a raw RGB image, a natural language instruction, an initial task plan, and failure detection - without additional information such as object detection or predefined conditions that may be unavailable in a given scenario. The replanning process consists of three steps: (i) Look, where LERa generates a scene description and identifies errors; (ii) Explain, where it provides corrective guidance; and (iii) Replan, where it modifies the plan accordingly. LERa is adaptable to various agent architectures and can handle errors from both dynamic scene changes and task execution failures. We evaluate LERa on the newly introduced ALFRED-ChaOS and VirtualHome-ChaOS datasets, achieving a 40% improvement over baselines in dynamic environments. In tabletop manipulation tasks with a predefined probability of task failure within the PyBullet simulator, LERa improves success rates by up to 67%. Further experiments, including real-world trials with a tabletop manipulator robot, confirm LERa's effectiveness in replanning. We demonstrate that LERa is a robust and adaptable solution for error-aware task execution in robotics. The code is available at this https URL. 

**Abstract (ZH)**: 视觉语言模型基于的重规划方法：看、解释、重规划 

---
# Automated Behaviour-Driven Acceptance Testing of Robotic Systems 

**Title (ZH)**: 基于行为驱动的自动验收测试在机器人系统中的应用 

**Authors**: Minh Nguyen, Sebastian Wrede, Nico Hochgeschwender  

**Link**: [PDF](https://arxiv.org/pdf/2507.05125)  

**Abstract**: The specification and validation of robotics applications require bridging the gap between formulating requirements and systematic testing. This often involves manual and error-prone tasks that become more complex as requirements, design, and implementation evolve. To address this challenge systematically, we propose extending behaviour-driven development (BDD) to define and verify acceptance criteria for robotic systems. In this context, we use domain-specific modelling and represent composable BDD models as knowledge graphs for robust querying and manipulation, facilitating the generation of executable testing models. A domain-specific language helps to efficiently specify robotic acceptance criteria. We explore the potential for automated generation and execution of acceptance tests through a software architecture that integrates a BDD framework, Isaac Sim, and model transformations, focusing on acceptance criteria for pick-and-place applications. We tested this architecture with an existing pick-and-place implementation and evaluated the execution results, which shows how this application behaves and fails differently when tested against variations of the agent and environment. This research advances the rigorous and automated evaluation of robotic systems, contributing to their reliability and trustworthiness. 

**Abstract (ZH)**: 机器人应用的需求规范与验证需要弥合从需求制定到系统测试之间的差距。这通常涉及手动且易出错的任务，随着需求、设计和实现的演变，这些任务变得更加复杂。为系统地应对这一挑战，我们提议将行为驱动开发（BDD）扩展到定义和验证机器人系统的接受标准。在这一背景下，我们使用领域特定建模，并将可重用的BDD模型表示为知识图谱，以实现稳健的查询和操作，促进可执行测试模型的生成。领域特定语言有助于高效地指定机器人接受标准。我们通过集成BDD框架、Isaac Sim和模型转换的软件架构探索自动化生成和执行接受测试的潜力，重点关注拾取和放置应用的接受标准。我们使用现有拾取和放置实现测试了该架构，并评估了执行结果，展示了该应用在针对代理和环境的不同变体进行测试时表现出的行为和失败方式有何不同。这项研究推进了机器人系统的严谨和自动化评估，从而提高其可靠性和可信度。 

---
# VerifyLLM: LLM-Based Pre-Execution Task Plan Verification for Robots 

**Title (ZH)**: VerifyLLM：基于LLM的预执行任务计划验证方法 

**Authors**: Danil S. Grigorev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2507.05118)  

**Abstract**: In the field of robotics, researchers face a critical challenge in ensuring reliable and efficient task planning. Verifying high-level task plans before execution significantly reduces errors and enhance the overall performance of these systems. In this paper, we propose an architecture for automatically verifying high-level task plans before their execution in simulator or real-world environments. Leveraging Large Language Models (LLMs), our approach consists of two key steps: first, the conversion of natural language instructions into Linear Temporal Logic (LTL), followed by a comprehensive analysis of action sequences. The module uses the reasoning capabilities of the LLM to evaluate logical coherence and identify potential gaps in the plan. Rigorous testing on datasets of varying complexity demonstrates the broad applicability of the module to household tasks. We contribute to improving the reliability and efficiency of task planning and addresses the critical need for robust pre-execution verification in autonomous systems. The code is available at this https URL. 

**Abstract (ZH)**: 在机器人技术领域，研究人员在确保任务规划可靠性和效率方面面临关键挑战。在执行前验证高级任务规划可以显著减少错误并提高这些系统的整体性能。本文提出一种架构，在模拟器或真实环境中执行前自动验证高级任务规划。利用大规模语言模型（LLMs），我们的方法包括两个关键步骤：首先，将自然语言指令转换为线性时态逻辑（LTL），然后对动作序列进行全面分析。模块利用LLM的推理能力评估逻辑一致性并识别计划中的潜在缺口。在不同复杂度的数据集上的严格测试表明模块对家务任务有广泛的适用性。我们致力于提高任务规划的可靠性和效率，并解决自主系统在执行前需要 robust 验证的迫切需求。代码可从该网址获取。 

---
# Beyond Features: How Dataset Design Influences Multi-Agent Trajectory Prediction Performance 

**Title (ZH)**: 超越特征：数据集设计对多Agent轨迹预测性能的影响 

**Authors**: Tobias Demmler, Jakob Häringer, Andreas Tamke, Thao Dang, Alexander Hegai, Lars Mikelsons  

**Link**: [PDF](https://arxiv.org/pdf/2507.05098)  

**Abstract**: Accurate trajectory prediction is critical for safe autonomous navigation, yet the impact of dataset design on model performance remains understudied. This work systematically examines how feature selection, cross-dataset transfer, and geographic diversity influence trajectory prediction accuracy in multi-agent settings. We evaluate a state-of-the-art model using our novel L4 Motion Forecasting dataset based on our own data recordings in Germany and the US. This includes enhanced map and agent features. We compare our dataset to the US-centric Argoverse 2 benchmark. First, we find that incorporating supplementary map and agent features unique to our dataset, yields no measurable improvement over baseline features, demonstrating that modern architectures do not need extensive feature sets for optimal performance. The limited features of public datasets are sufficient to capture convoluted interactions without added complexity. Second, we perform cross-dataset experiments to evaluate how effective domain knowledge can be transferred between datasets. Third, we group our dataset by country and check the knowledge transfer between different driving cultures. 

**Abstract (ZH)**: 准确的轨迹预测对于安全的自主导航至关重要，但数据集设计对模型性能的影响尚未得到充分研究。本文系统地探讨了特征选择、跨数据集迁移和地理多样性如何影响多agent环境中的轨迹预测精度。我们使用基于我们在德国和美国的数据记录构建的L4 Motion Forecasting数据集来评估最先进的模型，该数据集包含增强的地图和agent特征。我们将我们的数据集与以美国为中心的Argoverse 2基准进行比较。首先，我们发现，纳入我们数据集独有的补充地图和agent特征，并未在基线特征上带来可测量的改进，这表明现代架构不需要广泛的特征集即可实现最佳性能。公共数据集有限的特征足以捕捉复杂的交互而无需额外的复杂性。其次，我们进行了跨数据集实验，以评估领域知识在数据集之间的迁移效果。第三，我们按照国家对数据集进行分组，并检查不同驾驶文化的知识迁移。 

---
# Unifying Robot Optimization: Monte Carlo Tree Search with Tensor Factorization 

**Title (ZH)**: 基于张量因子分解的蒙特卡洛树搜索统一机器人优化 

**Authors**: Teng Xue, Amirreza Razmjoo, Yan Zhang, Sylvain Calinon  

**Link**: [PDF](https://arxiv.org/pdf/2507.04949)  

**Abstract**: Many robotic tasks, such as inverse kinematics, motion planning, and optimal control, can be formulated as optimization problems. Solving these problems involves addressing nonlinear kinematics, complex contact dynamics, and long-horizon planning, each posing distinct challenges for state-of-the-art optimization methods. To efficiently solve a wide range of tasks across varying scenarios, researchers either develop specialized algorithms for the task to achieve, or switch between different frameworks. Monte Carlo Tree Search (MCTS) is a general-purpose decision-making tool that enables strategic exploration across problem instances without relying on task-specific structures. However, MCTS suffers from combinatorial complexity, leading to slow convergence and high memory usage. To address this limitation, we propose \emph{Tensor Train Tree Search} (TTTS), which leverages tensor factorization to exploit the separable structure of decision trees. This yields a low-rank, linear-complexity representation that significantly reduces both computation time and storage requirements. We prove that TTTS can efficiently reach the bounded global optimum within a finite time. Experimental results across inverse kinematics, motion planning around obstacles, multi-stage motion planning, and bimanual whole-body manipulation demonstrate the efficiency of TTTS on a diverse set of robotic tasks. 

**Abstract (ZH)**: Tensor Train Tree Search: Efficient Decision-Making for Robotic Tasks 

---
# Automated UAV-based Wind Turbine Blade Inspection: Blade Stop Angle Estimation and Blade Detail Prioritized Exposure Adjustment 

**Title (ZH)**: 基于无人机的自动化风力 turbine叶片检测：叶片停止角度估计与叶片细节优先曝光调整 

**Authors**: Yichuan Shi, Hao Liu, Haowen Zheng, Haowen Yu, Xianqi Liang, Jie Li, Minmin Ma, Ximin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04922)  

**Abstract**: Unmanned aerial vehicles (UAVs) are critical in the automated inspection of wind turbine blades. Nevertheless, several issues persist in this domain. Firstly, existing inspection platforms encounter challenges in meeting the demands of automated inspection tasks and scenarios. Moreover, current blade stop angle estimation methods are vulnerable to environmental factors, restricting their robustness. Additionally, there is an absence of real-time blade detail prioritized exposure adjustment during capture, where lost details cannot be restored through post-optimization. To address these challenges, we introduce a platform and two approaches. Initially, a UAV inspection platform is presented to meet the automated inspection requirements. Subsequently, a Fermat point based blade stop angle estimation approach is introduced, achieving higher precision and success rates. Finally, we propose a blade detail prioritized exposure adjustment approach to ensure appropriate brightness and preserve details during image capture. Extensive tests, comprising over 120 flights across 10 wind turbine models in 5 operational wind farms, validate the effectiveness of the proposed approaches in enhancing inspection autonomy. 

**Abstract (ZH)**: 无人驾驶航空器（UAVs）在风力发电机叶片自动检测中的应用至关重要。然而，该领域仍存在几个问题。首先，现有的检测平台在满足自动检测任务和场景的需求方面遇到挑战。其次，当前的叶片停止角度估计方法容易受到环境因素的影响，限制了其鲁棒性。此外，在捕获过程中缺乏对叶片细节优先曝光调整的处理，缺失的细节无法通过后期优化恢复。为了应对这些挑战，我们引入了一个平台和两种方法。首先，提出了一种UAV检测平台以满足自动化检测要求。其次，引入了一种基于费马点的叶片停止角度估计方法，实现了更高的精度和成功率。最后，提出了一种叶片细节优先的曝光调整方法，以确保成像过程中的适当亮度并保留细节。通过在5个运营中的风力发电场的10种风力发电机模型上进行的超过120次飞行的广泛测试，验证了所提出方法在增强检测自主性方面的有效性。 

---
# Piggyback Camera: Easy-to-Deploy Visual Surveillance by Mobile Sensing on Commercial Robot Vacuums 

**Title (ZH)**: 搭车摄像头：基于商用扫地机器人的移动传感视觉监控易部署方案 

**Authors**: Ryo Yonetani  

**Link**: [PDF](https://arxiv.org/pdf/2507.04910)  

**Abstract**: This paper presents Piggyback Camera, an easy-to-deploy system for visual surveillance using commercial robot vacuums. Rather than requiring access to internal robot systems, our approach mounts a smartphone equipped with a camera and Inertial Measurement Unit (IMU) on the robot, making it applicable to any commercial robot without hardware modifications. The system estimates robot poses through neural inertial navigation and efficiently captures images at regular spatial intervals throughout the cleaning task. We develop a novel test-time data augmentation method called Rotation-Augmented Ensemble (RAE) to mitigate domain gaps in neural inertial navigation. A loop closure method that exploits robot cleaning patterns further refines these estimated poses. We demonstrate the system with an object mapping application that analyzes captured images to geo-localize objects in the environment. Experimental evaluation in retail environments shows that our approach achieves 0.83 m relative pose error for robot localization and 0.97 m positional error for object mapping of over 100 items. 

**Abstract (ZH)**: 基于商用吸尘器的搭车摄像头视觉监控系统 

---
# Dynamics and multi-stability of a rotor-actuated Twistcar robot with passive steering joint 

**Title (ZH)**: 带被动转向关节的旋转驱动Twistcar机器人的动力学与多稳定状态分析 

**Authors**: Anna Zigelman, Zitao Yu, Rom Levy, Yizhar Or  

**Link**: [PDF](https://arxiv.org/pdf/2507.04846)  

**Abstract**: The nonlinear dynamics of many under-actuated wheeled platforms are governed by nonholonomic constraints of no-skid for passively rolling wheels, coupled with momentum balance. In most of theoretical models, the shape variables, i.e. joint angles, are directly prescribed as periodic inputs, such as steering angle of the Twistcar. In this work, we study a variant of the Twistcar model where the actuation input is periodic oscillations of an inertial rotor attached to the main body, while the steering joint is passively free to rotate. Remarkably, the dynamics of this model is extremely rich, and includes multiplicity of periodic solutions, both symmetric and asymmetric, as well as stability transitions and bifurcations. We conduct numerical simulations as well as asymptotic analysis of the vehicle's reduced equations of motion. We use perturbation expansion in order to obtain leading-order dynamics under symmetric periodic solution. Then, we utilize harmonic balance and further scaling assumptions in order to approximate the conditions for symmetry-breaking pitchfork bifurcation and stability transition of the symmetric periodic solution, as a function of actuation frequency and structural parameters. The asymptotic results show good agreement with numerical simulations. The results highlight the role of passive shape variables in generating multi-stable periodic solutions for nonholonomic systems of robotic locomotion. 

**Abstract (ZH)**: 多自由度非完全驱动轮式平台的动力学受到无滑滚约束的非线性控制，结合动量平衡。在大多数理论模型中，形变量，即关节角，直接被设定为周期输入，如Twistcar的转向角。在本工作中，我们研究了一个Twistcar模型的变体，在该变体中，驱动输入是附加在主体上的惯性旋转器的周期振荡，而转向关节则被动自由旋转。令人惊讶的是，该模型的动力学非常丰富，包括多种周期解，既有对称的也有非对称的，还存在稳定性转换和分岔现象。我们进行了车辆的降阶运动方程数值模拟和渐近分析。我们使用扰动扩展来获得对称周期解下的主导动力学。然后我们利用谐波平衡和进一步的缩放假设来逼近对称周期解的破缺 pitches 分叉和稳定性转换条件，作为驱动频率和结构参数的函数。渐近结果与数值模拟结果一致。结果突显了被动形变量在非完整运动学机器人运动中的多稳定周期解生成中的作用。 

---
# Safe Bimanual Teleoperation with Language-Guided Collision Avoidance 

**Title (ZH)**: 语言引导避碰的Safe双臂远程操作 

**Authors**: Dionis Totsila, Clemente Donoso, Enrico Mingo Hoffman, Jean-Baptiste Mouret, Serena Ivaldi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04791)  

**Abstract**: Teleoperating precise bimanual manipulations in cluttered environments is challenging for operators, who often struggle with limited spatial perception and difficulty estimating distances between target objects, the robot's body, obstacles, and the surrounding environment. To address these challenges, local robot perception and control should assist the operator during teleoperation. In this work, we introduce a safe teleoperation system that enhances operator control by preventing collisions in cluttered environments through the combination of immersive VR control and voice-activated collision avoidance. Using HTC Vive controllers, operators directly control a bimanual mobile manipulator, while spoken commands such as "avoid the yellow tool" trigger visual grounding and segmentation to build 3D obstacle meshes. These meshes are integrated into a whole-body controller to actively prevent collisions during teleoperation. Experiments in static, cluttered scenes demonstrate that our system significantly improves operational safety without compromising task efficiency. 

**Abstract (ZH)**: 在嘈杂环境中进行精确双臂操作对操作者构成挑战，他们常常难以应对有限的空间感知和目标物体、机器人本体、障碍物及周围环境之间的距离估算困难。为应对这些挑战，局部机器人感知和控制应在此远程操作过程中提供帮助。本文介绍了一种安全的远程操作系统，该系统通过结合沉浸式VR控制和语音激活的避撞功能来增强操作者的控制能力，防止在嘈杂环境中发生碰撞。使用HTC Vive控制器，操作者直接控制双臂移动操作器，而通过说出“避免黄色工具”等口令触发视觉接地和分割来构建3D障碍网格。这些网格被集成到全身控制器中，以在远程操作过程中主动防止碰撞。在静态、嘈杂场景中的实验表明，该系统显著提高了操作安全性，同时不牺牲任务效率。 

---
# Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning 

**Title (ZH)**: 交互融合运动规划：有效利用多样化运动数据集进行鲁棒规划 

**Authors**: Giwon Lee, Wooseong Jeong, Daehee Park, Jaewoo Jeong, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2507.04790)  

**Abstract**: Motion planning is a crucial component of autonomous robot driving. While various trajectory datasets exist, effectively utilizing them for a target domain remains challenging due to differences in agent interactions and environmental characteristics. Conventional approaches, such as domain adaptation or ensemble learning, leverage multiple source datasets but suffer from domain imbalance, catastrophic forgetting, and high computational costs. To address these challenges, we propose Interaction-Merged Motion Planning (IMMP), a novel approach that leverages parameter checkpoints trained on different domains during adaptation to the target domain. IMMP follows a two-step process: pre-merging to capture agent behaviors and interactions, sufficiently extracting diverse information from the source domain, followed by merging to construct an adaptable model that efficiently transfers diverse interactions to the target domain. Our method is evaluated on various planning benchmarks and models, demonstrating superior performance compared to conventional approaches. 

**Abstract (ZH)**: 基于交互融合的自主机器人运动规划（Interaction-Merged Motion Planning for Autonomous Robot Driving） 

---
# Training-free Generation of Temporally Consistent Rewards from VLMs 

**Title (ZH)**: 无需训练的时空一致奖励生成从VLMs 

**Authors**: Yinuo Zhao, Jiale Yuan, Zhiyuan Xu, Xiaoshuai Hao, Xinyi Zhang, Kun Wu, Zhengping Che, Chi Harold Liu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04789)  

**Abstract**: Recent advances in vision-language models (VLMs) have significantly improved performance in embodied tasks such as goal decomposition and visual comprehension. However, providing accurate rewards for robotic manipulation without fine-tuning VLMs remains challenging due to the absence of domain-specific robotic knowledge in pre-trained datasets and high computational costs that hinder real-time applicability. To address this, we propose $\mathrm{T}^2$-VLM, a novel training-free, temporally consistent framework that generates accurate rewards through tracking the status changes in VLM-derived subgoals. Specifically, our method first queries the VLM to establish spatially aware subgoals and an initial completion estimate before each round of interaction. We then employ a Bayesian tracking algorithm to update the goal completion status dynamically, using subgoal hidden states to generate structured rewards for reinforcement learning (RL) agents. This approach enhances long-horizon decision-making and improves failure recovery capabilities with RL. Extensive experiments indicate that $\mathrm{T}^2$-VLM achieves state-of-the-art performance in two robot manipulation benchmarks, demonstrating superior reward accuracy with reduced computation consumption. We believe our approach not only advances reward generation techniques but also contributes to the broader field of embodied AI. Project website: this https URL. 

**Abstract (ZH)**: Recent advances in 视觉-语言模型(VLMs)显著提升了目标分解和视觉理解等沉浸式任务的表现。然而，要在不微调VLMs的情况下为机器人操作提供准确的奖励仍然具有挑战性，这是因为预训练数据集中缺乏特定领域的机器人知识，以及高昂的计算成本限制了实时适用性。为了解决这一问题，我们提出了一种新的无需训练、时序一致的框架$\mathrm{T}^2$-VLM，该框架通过跟踪由VLM生成的子目标状态变化来生成准确的奖励。具体而言，我们的方法首先在每次交互轮次前查询VLM以建立空间意识子目标和初步完成估计。然后，我们采用贝叶斯跟踪算法动态更新目标完成状态，并利用子目标隐藏状态为强化学习(RL)代理生成结构化奖励。该方法增强了长期决策制定能力，并通过RL提高了故障恢复能力。大量实验表明，$\mathrm{T}^2$-VLM在两个机器人操作基准测试中达到了最先进的性能，展示了在减少计算消耗的同时更高的奖励准确性。我们相信，我们的方法不仅推动了奖励生成技术的发展，还对更广泛的沉浸式AI领域做出了贡献。 

---
# CueLearner: Bootstrapping and local policy adaptation from relative feedback 

**Title (ZH)**: CueLearner: 从相对反馈进行自强化和局部策略适应 

**Authors**: Giulio Schiavi, Andrei Cramariuc, Lionel Ott, Roland Siegwart  

**Link**: [PDF](https://arxiv.org/pdf/2507.04730)  

**Abstract**: Human guidance has emerged as a powerful tool for enhancing reinforcement learning (RL). However, conventional forms of guidance such as demonstrations or binary scalar feedback can be challenging to collect or have low information content, motivating the exploration of other forms of human input. Among these, relative feedback (i.e., feedback on how to improve an action, such as "more to the left") offers a good balance between usability and information richness. Previous research has shown that relative feedback can be used to enhance policy search methods. However, these efforts have been limited to specific policy classes and use feedback inefficiently. In this work, we introduce a novel method to learn from relative feedback and combine it with off-policy reinforcement learning. Through evaluations on two sparse-reward tasks, we demonstrate our method can be used to improve the sample efficiency of reinforcement learning by guiding its exploration process. Additionally, we show it can adapt a policy to changes in the environment or the user's preferences. Finally, we demonstrate real-world applicability by employing our approach to learn a navigation policy in a sparse reward setting. 

**Abstract (ZH)**: 人类指导已成为增强强化学习(RL)性能的强大工具。然而，传统的指导形式如演示或二元标量反馈在收集上具有挑战性或信息含量低，促使探索其他形式的人类输入。在这些形式中，相对反馈（即对改善动作的反馈，如“更往左”）在可用性和信息丰富性之间提供了良好的平衡。之前的研究所表明，相对反馈可以用于增强策略搜索方法。然而，这些努力仅限于特定的策略类别，并且使用反馈效率低下。在本工作中，我们提出了一种新的方法来自学相对反馈，并将其与脱政策强化学习结合。通过在两个稀疏奖励任务上的评估，我们证明该方法可以提高强化学习的样本效率，引导其探索过程。此外，我们展示了该方法可以根据环境变化或用户偏好进行政策调整。最后，我们通过将该方法应用于稀疏奖励环境下的导航策略学习，展示了其实用性。 

---
# MOSU: Autonomous Long-range Robot Navigation with Multi-modal Scene Understanding 

**Title (ZH)**: MOSU: 多模态场景理解驱动的自主长距离机器人导航 

**Authors**: Jing Liang, Kasun Weerakoon, Daeun Song, Senthurbavan Kirubaharan, Xuesu Xiao, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2507.04686)  

**Abstract**: We present MOSU, a novel autonomous long-range navigation system that enhances global navigation for mobile robots through multimodal perception and on-road scene understanding. MOSU addresses the outdoor robot navigation challenge by integrating geometric, semantic, and contextual information to ensure comprehensive scene understanding. The system combines GPS and QGIS map-based routing for high-level global path planning and multi-modal trajectory generation for local navigation refinement. For trajectory generation, MOSU leverages multi-modalities: LiDAR-based geometric data for precise obstacle avoidance, image-based semantic segmentation for traversability assessment, and Vision-Language Models (VLMs) to capture social context and enable the robot to adhere to social norms in complex environments. This multi-modal integration improves scene understanding and enhances traversability, allowing the robot to adapt to diverse outdoor conditions. We evaluate our system in real-world on-road environments and benchmark it on the GND dataset, achieving a 10% improvement in traversability on navigable terrains while maintaining a comparable navigation distance to existing global navigation methods. 

**Abstract (ZH)**: MOSU：一种通过多模态感知和道路场景理解增强移动机器人远程导航的新型自主导航系统 

---
# DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics 

**Title (ZH)**: 动态检索增强专家网络：面向机器人终身学习和任务适应的研究 

**Authors**: Yayu Long, Kewei Chen, Long Jin, Mingsheng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04661)  

**Abstract**: We introduce Dynamic Retrieval-Augmented Expert Networks (DRAE), a groundbreaking architecture that addresses the challenges of lifelong learning, catastrophic forgetting, and task adaptation by combining the dynamic routing capabilities of Mixture-of-Experts (MoE); leveraging the knowledge-enhancement power of Retrieval-Augmented Generation (RAG); incorporating a novel hierarchical reinforcement learning (RL) framework; and coordinating through ReflexNet-SchemaPlanner-HyperOptima (RSHO).DRAE dynamically routes expert models via a sparse MoE gating mechanism, enabling efficient resource allocation while leveraging external knowledge through parametric retrieval (P-RAG) to augment the learning process. We propose a new RL framework with ReflexNet for low-level task execution, SchemaPlanner for symbolic reasoning, and HyperOptima for long-term context modeling, ensuring continuous adaptation and memory retention. Experimental results show that DRAE significantly outperforms baseline approaches in long-term task retention and knowledge reuse, achieving an average task success rate of 82.5% across a set of dynamic robotic manipulation tasks, compared to 74.2% for traditional MoE models. Furthermore, DRAE maintains an extremely low forgetting rate, outperforming state-of-the-art methods in catastrophic forgetting mitigation. These results demonstrate the effectiveness of our approach in enabling flexible, scalable, and efficient lifelong learning for robotics. 

**Abstract (ZH)**: 动态检索增强专家网络（DRAE）：一种应对终身学习挑战的创新架构 

---
# Bio-Inspired Hybrid Map: Spatial Implicit Local Frames and Topological Map for Mobile Cobot Navigation 

**Title (ZH)**: 生物启发的混合地图：空间隐式局部帧与拓扑地图用于移动协作机器人导航 

**Authors**: Tuan Dang, Manfred Huber  

**Link**: [PDF](https://arxiv.org/pdf/2507.04649)  

**Abstract**: Navigation is a fundamental capacity for mobile robots, enabling them to operate autonomously in complex and dynamic environments. Conventional approaches use probabilistic models to localize robots and build maps simultaneously using sensor observations. Recent approaches employ human-inspired learning, such as imitation and reinforcement learning, to navigate robots more effectively. However, these methods suffer from high computational costs, global map inconsistency, and poor generalization to unseen environments. This paper presents a novel method inspired by how humans perceive and navigate themselves effectively in novel environments. Specifically, we first build local frames that mimic how humans represent essential spatial information in the short term. Points in local frames are hybrid representations, including spatial information and learned features, so-called spatial-implicit local frames. Then, we integrate spatial-implicit local frames into the global topological map represented as a factor graph. Lastly, we developed a novel navigation algorithm based on Rapid-Exploring Random Tree Star (RRT*) that leverages spatial-implicit local frames and the topological map to navigate effectively in environments. To validate our approach, we conduct extensive experiments in real-world datasets and in-lab environments. We open our source code at this https URL}{this https URL. 

**Abstract (ZH)**: 移动机器人在复杂和动态环境中的导航是一项基本能力，使其能够自主操作。传统方法使用概率模型同时利用传感器观测值进行机器人定位和建图。近年来，通过模仿和强化学习等受人类启发的学习方法，提高了机器人导航的效率。然而，这些方法存在计算成本高、全局地图不一致和对未见环境泛化能力差的问题。本论文提出了一种受人类如何在新环境中有效感知和导航启发的新方法。具体而言，我们首先构建局部框架，模拟人类在短期中表示关键空间信息的方式。局部框架中的点是混合表示，包括空间信息和学习特征，称为空间隐式局部框架。然后，我们将空间隐式局部框架整合到由因子图表示的全局拓扑地图中。最后，我们开发了一种基于快速扩展随机树星（RRT*）的新导航算法，该算法利用空间隐式局部框架和拓扑地图有效地在环境中导航。为了验证我们的方法，我们在真实世界数据集和实验室环境中进行了广泛的实验。我们已开源代码，详见此链接：this https URL此链接：this https URL。 

---
# PRISM: Pointcloud Reintegrated Inference via Segmentation and Cross-attention for Manipulation 

**Title (ZH)**: PRISM: 通过分割和跨注意力实现点云重新整合的操纵推理 

**Authors**: Daqi Huang, Zhehao Cai, Yuzhi Hao, Zechen Li, Chee-Meng Chew  

**Link**: [PDF](https://arxiv.org/pdf/2507.04633)  

**Abstract**: Robust imitation learning for robot manipulation requires comprehensive 3D perception, yet many existing methods struggle in cluttered environments. Fixed camera view approaches are vulnerable to perspective changes, and 3D point cloud techniques often limit themselves to keyframes predictions, reducing their efficacy in dynamic, contact-intensive tasks. To address these challenges, we propose PRISM, designed as an end-to-end framework that directly learns from raw point cloud observations and robot states, eliminating the need for pretrained models or external datasets. PRISM comprises three main components: a segmentation embedding unit that partitions the raw point cloud into distinct object clusters and encodes local geometric details; a cross-attention component that merges these visual features with processed robot joint states to highlight relevant targets; and a diffusion module that translates the fused representation into smooth robot actions. With training on 100 demonstrations per task, PRISM surpasses both 2D and 3D baseline policies in accuracy and efficiency within our simulated environments, demonstrating strong robustness in complex, object-dense scenarios. Code and some demos are available on this https URL. 

**Abstract (ZH)**: 鲁棒的机器人 manipulation 模仿学习需要全面的 3D 感知，但许多现有方法在杂乱环境中表现不佳。固定相机视角方法易受视角变化影响，而基于 3D 点云的技术往往局限于关键帧预测，在动态、接触密集型任务中的效果受限。为应对这些挑战，我们提出了 PRISM，设计为端到端框架，直接从原始点云观测和机器人状态中学习，无需预训练模型或外部数据集。PRISM 包含三个主要组件：分割嵌入单元，将原始点云分割为独立的对象簇并编码局部几何细节；跨注意力组件，将这些视觉特征与处理过的机器人关节状态合并，突出相关目标；扩散模块，将融合表示转化为平滑的机器人动作。通过每任务 100 次演示的训练，PRISM 在我们模拟环境中优于 2D 和 3D 基线策略，展示了在复杂、物体密集场景中的强大鲁棒性。代码和一些演示可在以下网址获取。 

---
# IDAGC: Adaptive Generalized Human-Robot Collaboration via Human Intent Estimation and Multimodal Policy Learning 

**Title (ZH)**: IDAGC：基于人类意图估计和多模态策略学习的自适应通用人类-机器人协作 

**Authors**: Haotian Liu, Yuchuang Tong, Guanchen Liu, Zhaojie Ju, Zhengtao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04620)  

**Abstract**: In Human-Robot Collaboration (HRC), which encompasses physical interaction and remote cooperation, accurate estimation of human intentions and seamless switching of collaboration modes to adjust robot behavior remain paramount challenges. To address these issues, we propose an Intent-Driven Adaptive Generalized Collaboration (IDAGC) framework that leverages multimodal data and human intent estimation to facilitate adaptive policy learning across multi-tasks in diverse scenarios, thereby facilitating autonomous inference of collaboration modes and dynamic adjustment of robotic actions. This framework overcomes the limitations of existing HRC methods, which are typically restricted to a single collaboration mode and lack the capacity to identify and transition between diverse states. Central to our framework is a predictive model that captures the interdependencies among vision, language, force, and robot state data to accurately recognize human intentions with a Conditional Variational Autoencoder (CVAE) and automatically switch collaboration modes. By employing dedicated encoders for each modality and integrating extracted features through a Transformer decoder, the framework efficiently learns multi-task policies, while force data optimizes compliance control and intent estimation accuracy during physical interactions. Experiments highlights our framework's practical potential to advance the comprehensive development of HRC. 

**Abstract (ZH)**: 在人类-机器人协作（HRC）中，涵盖物理交互和远程合作，准确估计人类意图和无缝切换协作模式以调整机器人行为仍然是主要挑战。为解决这些问题，我们提出了一种基于意图的自适应通用协作（IDAGC）框架，该框架利用多模态数据和人类意图估计来促进多任务在多种场景下的自适应策略学习，从而实现协作模式的自主推理和机器人类动的动态调整。该框架克服了现有HRC方法的限制，这些方法通常局限于单一协作模式，并且缺乏在不同状态下识别和转换的能力。本框架的核心是一个预测模型，该模型捕捉视觉、语言、力和机器人状态数据之间的相互依赖关系，通过条件变分自编码器（CVAE）准确识别人类意图，并自动切换协作模式。通过为每种模态使用专门的编码器并利用Transformer解码器集成提取特征，框架高效地学习多任务策略，同时力数据优化物理交互中的顺应控制和意图估计准确性。实验突出了我们框架在全面推动HRC发展的实际潜力。 

---
# DragonFly: Single mmWave Radar 3D Localization of Highly Dynamic Tags in GPS-Denied Environments 

**Title (ZH)**: DragonFly：在GPS受限环境中对高度动态标签进行单毫米波雷达3D定位 

**Authors**: Skanda Harisha, Jimmy G. D. Hester, Aline Eid  

**Link**: [PDF](https://arxiv.org/pdf/2507.04602)  

**Abstract**: The accurate localization and tracking of dynamic targets, such as equipment, people, vehicles, drones, robots, and the assets that they interact with in GPS-denied indoor environments is critical to enabling safe and efficient operations in the next generation of spatially aware industrial facilities. This paper presents DragonFly , a 3D localization system of highly dynamic backscatter tags using a single MIMO mmWave radar. The system delivers the first demonstration of a mmWave backscatter system capable of exploiting the capabilities of MIMO radars for the 3D localization of mmID tags moving at high speeds and accelerations at long ranges by introducing a critical Doppler disambiguation algorithm and a fully integrated cross-polarized dielectric lens-based mmID tag consuming a mere 68 uW. DragonFly was extensively evaluated in static and dynamic configurations, including on a flying quadcopter, and benchmarked against multiple baselines, demonstrating its ability to track the positions of multiple tags with a median 3D accuracy of 12 cm at speeds and acceleration on the order of 10 m/s-1 and 4 m/s-2 and at ranges of up to 50 m. 

**Abstract (ZH)**: 在GPS受限的室内环境中，动态目标（如设备、人员、车辆、无人机、机器人及其交互的资产）的精确定位与跟踪对于下一代空间感知工业设施的安全高效运行至关重要。本文展示了DragonFly，一种使用单个MIMO毫米波雷达的3D定位系统，该系统利用MIMO雷达的能力，通过引入关键的多普勒解混算法和消费仅68微瓦的全集成交叉极化介电镜头基毫米波标签，实现了高速远距离移动毫米波标签的3D定位演示。DragonFly在静态和动态配置下进行了广泛评估，包括在飞行四旋翼无人机上的应用，并与多个基准进行了对比测试，展示了其在速度和加速度约为10 m/s-1和4 m/s-2、范围达50米时，能够以中值3D精度12厘米跟踪多个标签的能力。 

---
# The Difference between the Left and Right Invariant Extended Kalman Filter 

**Title (ZH)**: 左 invariant 扩展卡尔曼滤波器与右 invariant 扩展卡尔曼滤波器的区别 

**Authors**: Yixiao Ge, Giulio Delama, Martin Scheiber, Alessandro Fornasier, Pieter van Goor, Stephan Weiss, Robert Mahony  

**Link**: [PDF](https://arxiv.org/pdf/2507.04568)  

**Abstract**: The extended Kalman filter (EKF) has been the industry standard for state estimation problems over the past sixty years. The Invariant Extended Kalman Filter (IEKF) is a recent development of the EKF for the class of group-affine systems on Lie groups that has shown superior performance for inertial navigation problems. The IEKF comes in two versions, left- and right- handed respectively, and there is a perception in the robotics community that these filters are different and one should choose the handedness of the IEKF to match handedness of the measurement model for a given filtering problem. In this paper, we revisit these algorithms and demonstrate that the left- and right- IEKF algorithms (with reset step) are identical, that is, the choice of the handedness does not affect the IEKF's performance when the reset step is properly implemented. The reset step was not originally proposed as part of the IEKF, however, we provide simulations to show that the reset step improves asymptotic performance of all versions of the the filter, and should be included in all high performance algorithms. The GNSS-aided inertial navigation system (INS) is used as a motivating example to demonstrate the equivalence of the two filters. 

**Abstract (ZH)**: 扩展卡尔曼滤波器(EKF)在过去六十年中一直是状态估计问题的工业标准。不变扩展卡尔曼滤波器(IEKF)是针对李群上群仿射系统的最近发展，已在惯性导航问题中显示出优越的性能。IEKF有两种版本，分别是左手型和右手型，机器人学界有一种观点认为这些滤波器是不同的，使用者应选择与给定滤波问题测量模型手性的IEKF。本文重新审视了这些算法，并证明在适当实施重置步骤的情况下，左手型和右手型IEKF算法（含重置步骤）是相同的，即手性的选择不会影响IEKF的性能。重置步骤最初未被纳入IEKF，但通过仿真展示了该步骤可以改善所有版本滤波器的渐近性能，并应在高性能算法中包含。全球导航卫星系统辅助惯性导航系统(GNSS-aided INS)被用作演示两种滤波器等价性的动机例子。 

---
# VLM-TDP: VLM-guided Trajectory-conditioned Diffusion Policy for Robust Long-Horizon Manipulation 

**Title (ZH)**: VLM-TDP: 由VLM引导的轨迹条件化扩散策略以实现稳健的长时_horizon �操作 

**Authors**: Kefeng Huang, Tingguang Li, Yuzhen Liu, Zhe Zhang, Jiankun Wang, Lei Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.04524)  

**Abstract**: Diffusion policy has demonstrated promising performance in the field of robotic manipulation. However, its effectiveness has been primarily limited in short-horizon tasks, and its performance significantly degrades in the presence of image noise. To address these limitations, we propose a VLM-guided trajectory-conditioned diffusion policy (VLM-TDP) for robust and long-horizon manipulation. Specifically, the proposed method leverages state-of-the-art vision-language models (VLMs) to decompose long-horizon tasks into concise, manageable sub-tasks, while also innovatively generating voxel-based trajectories for each sub-task. The generated trajectories serve as a crucial conditioning factor, effectively steering the diffusion policy and substantially enhancing its performance. The proposed Trajectory-conditioned Diffusion Policy (TDP) is trained on trajectories derived from demonstration data and validated using the trajectories generated by the VLM. Simulation experimental results indicate that our method significantly outperforms classical diffusion policies, achieving an average 44% increase in success rate, over 100% improvement in long-horizon tasks, and a 20% reduction in performance degradation in challenging conditions, such as noisy images or altered environments. These findings are further reinforced by our real-world experiments, where the performance gap becomes even more pronounced in long-horizon tasks. Videos are available on this https URL 

**Abstract (ZH)**: 基于VLM引导的轨迹条件扩散策略（VLM-TDP）在鲁棒长 horizon 操作中的应用 

---
# Verification of Visual Controllers via Compositional Geometric Transformations 

**Title (ZH)**: 视觉控制器的组合几何变换验证 

**Authors**: Alexander Estornell, Leonard Jung, Michael Everett  

**Link**: [PDF](https://arxiv.org/pdf/2507.04523)  

**Abstract**: Perception-based neural network controllers are increasingly used in autonomous systems that rely on visual inputs to operate in the real world. Ensuring the safety of such systems under uncertainty is challenging. Existing verification techniques typically focus on Lp-bounded perturbations in the pixel space, which fails to capture the low-dimensional structure of many real-world effects. In this work, we introduce a novel verification framework for perception-based controllers that can generate outer-approximations of reachable sets through explicitly modeling uncertain observations with geometric perturbations. Our approach constructs a boundable mapping from states to images, enabling the use of state-based verification tools while accounting for uncertainty in perception. We provide theoretical guarantees on the soundness of our method and demonstrate its effectiveness across benchmark control environments. This work provides a principled framework for certifying the safety of perception-driven control systems under realistic visual perturbations. 

**Abstract (ZH)**: 基于感知的神经网络控制器在依赖视觉输入的自主系统中被广泛使用。在不确定性条件下确保这类系统的安全性具有挑战性。现有的验证技术通常集中于像素空间中的Lp有界扰动，这未能捕捉许多现实世界效果的低维结构。在本文中，我们引入了一种新的验证框架，通过明确地使用几何扰动建模不确定观测来生成可达集的外近似。我们的方法构造了从状态到图像的可界映射，使基于状态的验证工具能够考虑感知中的不确定性。我们提供了我们方法正确性的理论保证，并在其基准控制环境中展示了其有效性。本文为在现实视觉扰动条件下认证感知驱动控制系统的安全性提供了原则性框架。 

---
# SimLauncher: Launching Sample-Efficient Real-world Robotic Reinforcement Learning via Simulation Pre-training 

**Title (ZH)**: SimLauncher：通过模拟预训练实现高效现实机器人强化学习的样本启动器 

**Authors**: Mingdong Wu, Lehong Wu, Yizhuo Wu, Weiyao Huang, Hongwei Fan, Zheyuan Hu, Haoran Geng, Jinzhou Li, Jiahe Ying, Long Yang, Yuanpei Chen, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.04452)  

**Abstract**: Autonomous learning of dexterous, long-horizon robotic skills has been a longstanding pursuit of embodied AI. Recent advances in robotic reinforcement learning (RL) have demonstrated remarkable performance and robustness in real-world visuomotor control tasks. However, applying RL in the real world faces challenges such as low sample efficiency, slow exploration, and significant reliance on human intervention. In contrast, simulators offer a safe and efficient environment for extensive exploration and data collection, while the visual sim-to-real gap, often a limiting factor, can be mitigated using real-to-sim techniques. Building on these, we propose SimLauncher, a novel framework that combines the strengths of real-world RL and real-to-sim-to-real approaches to overcome these challenges. Specifically, we first pre-train a visuomotor policy in the digital twin simulation environment, which then benefits real-world RL in two ways: (1) bootstrapping target values using extensive simulated demonstrations and real-world demonstrations derived from pre-trained policy rollouts, and (2) Incorporating action proposals from the pre-trained policy for better exploration. We conduct comprehensive experiments across multi-stage, contact-rich, and dexterous hand manipulation tasks. Compared to prior real-world RL approaches, SimLauncher significantly improves sample efficiency and achieves near-perfect success rates. We hope this work serves as a proof of concept and inspires further research on leveraging large-scale simulation pre-training to benefit real-world robotic RL. 

**Abstract (ZH)**: 自主学习灵巧、长期 horizon 机器人技能是嵌入式人工智能长期追求的目标。近期在机器人强化学习（RL）方面的进展已在实际世界的视觉-运动控制任务中展示了卓越的性能和鲁棒性。然而，在现实世界中应用RL面临着样本效率低、探索缓慢以及对人类干预依赖性强等挑战。相比之下，模拟器提供了安全且高效的环境，用于广泛的探索和数据收集，而现实到模拟的视觉差距常常是一个限制因素，可以通过现实到模拟技术来缓解。在此基础上，我们提出了SimLauncher这一新颖框架，该框架结合了现实世界RL和现实到模拟再到现实方法的优势，以克服这些挑战。具体来说，我们首先在数字孪生模拟环境中预训练一个视觉-运动策略，该策略在两个方面有助于现实世界的RL：（1）利用广泛的模拟演示和由预训练策略执行卷出的真实世界演示来启动目标值；（2）结合预训练策略的动作提案以提高探索效果。我们在多阶段、接触丰富和灵巧手操作任务中进行了全面的实验。与先前的现实世界RL方法相比，SimLauncher显著提高了样本效率，并实现了近乎完美的成功率。我们希望这项工作能够作为概念验证，并激励进一步研究，利用大规模模拟预训练来提升现实世界机器人RL的效果。 

---
# Free-Space Optical Communication-Driven NMPC Framework for Multi-Rotor Aerial Vehicles in Structured Inspection Scenarios 

**Title (ZH)**: 自由空间光学通信驱动的NMPC框架在结构化检查场景中的多旋翼无人机应用 

**Authors**: Giuseppe Silano, Daniel Bonilla Licea, Hajar El Hammouti, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2507.04443)  

**Abstract**: This paper introduces a Nonlinear Model Predictive Control (NMPC) framework for communication-aware motion planning of Multi-Rotor Aerial Vehicles (MRAVs) using Free-Space Optical (FSO) links. The scenario involves MRAVs equipped with body-fixed optical transmitters and Unmanned Ground Vehicles (UGVs) acting as mobile relays, each outfitted with fixed conical Field-of-View (FoV) receivers. The controller integrates optical connectivity constraints into the NMPC formulation to ensure beam alignment and minimum link quality, while also enabling UGV tracking and obstacle avoidance. The method supports both coplanar and tilted MRAV configurations. MATLAB simulations demonstrate its feasibility and effectiveness. 

**Abstract (ZH)**: 基于自由空间光学链接的多旋翼无人机通信感知运动规划的非线性模型预测控制框架 

---
# "Hi AirStar, Guide Me to the Badminton Court." 

**Title (ZH)**: Hi AirStar, 引领我到羽毛球场地。 

**Authors**: Ziqin Wang, Jinyu Chen, Xiangyi Zheng, Qinan Liao, Linjiang Huang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04430)  

**Abstract**: Unmanned Aerial Vehicles, operating in environments with relatively few obstacles, offer high maneuverability and full three-dimensional mobility. This allows them to rapidly approach objects and perform a wide range of tasks often challenging for ground robots, making them ideal for exploration, inspection, aerial imaging, and everyday assistance. In this paper, we introduce AirStar, a UAV-centric embodied platform that turns a UAV into an intelligent aerial assistant: a large language model acts as the cognitive core for environmental understanding, contextual reasoning, and task planning. AirStar accepts natural interaction through voice commands and gestures, removing the need for a remote controller and significantly broadening its user base. It combines geospatial knowledge-driven long-distance navigation with contextual reasoning for fine-grained short-range control, resulting in an efficient and accurate vision-and-language navigation (VLN) this http URL, the system also offers built-in capabilities such as cross-modal question answering, intelligent filming, and target tracking. With a highly extensible framework, it supports seamless integration of new functionalities, paving the way toward a general-purpose, instruction-driven intelligent UAV agent. The supplementary PPT is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 无人驾驶航空器在较少障碍的环境中操作，具有高机动性和全三维移动性，能够快速接近物体并执行多种任务，这些任务往往对地面机器人具有挑战性，使其成为探索、检查、空中成像和日常生活助手的理想选择。本文介绍了一种以无人驾驶航空器为中心的自主平台AirStar，该平台将无人驾驶航空器转变为智能空中助手：一个大型语言模型作为认知核心，用于环境理解、上下文推理和任务规划。AirStar 通过语音命令和手势接受自然交互，去掉了远程控制器的需要，显著扩展了其用户基础。它结合了基于地理空间知识的远程导航与基于上下文的精细近距离控制，实现了高效的视觉-语言导航（VLN）。此外，该系统还内置了跨模态问答、智能拍摄和目标跟踪等功能。凭借高度可扩展的框架，它支持无缝集成新功能，朝着通用目的、指令驱动的智能无人驾驶航空器代理迈出了一步。更多资料请参见附录PPT，链接：https://supplementary.example.com/ppt。 

---
# Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition 

**Title (ZH)**: 跨越多样场景的快速安全轨迹规划通过扩散组合 

**Authors**: Wule Mao, Zhouheng Li, Yunhao Luo, Yilun Du, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.04384)  

**Abstract**: Safe trajectory planning remains a significant challenge in complex environments, where traditional methods often trade off computational efficiency for safety. Comprehensive obstacle modeling improves safety but is computationally expensive, while approximate methods are more efficient but may compromise safety. To address this issue, this paper introduces a rapid and safe trajectory planning framework based on state-based diffusion models. Leveraging only low-dimensional vehicle states, the diffusion models achieve notable inference efficiency while ensuring sufficient collision-free characteristics. By composing diffusion models, the proposed framework can safely generalize across diverse scenarios, planning collision-free trajectories even in unseen scenes. To further ensure the safety of the generated trajectories, an efficient, rule-based safety filter is proposed, which selects optimal trajectories that satisfy both sufficient safety and control feasibility from among candidate trajectories. Both in seen and unseen scenarios, the proposed method achieves efficient inference time while maintaining high safety and stability. Evaluations on the F1TENTH vehicle further demonstrate that the proposed method is practical in real-world applications. The project page is at: this https URL. 

**Abstract (ZH)**: 安全轨迹规划在复杂环境中仍是一项重大挑战，传统的方法往往在计算效率和安全性之间做出权衡。全面的障碍模型虽然能够提高安全性，但计算成本高昂，而近似方法尽管更有效率，但在安全性方面可能有所妥协。为解决这一问题，本文提出了一种基于状态扩散模型的快速安全轨迹规划框架。通过仅利用低维度的车辆状态，扩散模型实现了显著的推理效率，同时确保足够的无碰撞特性。通过组合扩散模型，所提出的框架能够安全地泛化到多种不同的场景，在未见过的场景中也能规划出无碰撞轨迹。为了进一步确保生成轨迹的安全性，本文还提出了一种基于规则的高效安全过滤器，该过滤器可以从候选轨迹中选择同时满足足够安全性和控制可行性的最优轨迹。在已见过和未见过的场景中，所提出的方法都能实现高效的推理时间，同时保持高水平的安全性和稳定性。基于F1TENTH车辆的评估进一步证明了所提出方法在实际应用中的实用性。项目页面链接为：this https URL。 

---
# Implicit Dual-Control for Visibility-Aware Navigation in Unstructured Environments 

**Title (ZH)**: 隐式双控制在无结构环境中的可见性感知导航 

**Authors**: Benjamin Johnson, Qilun Zhu, Robert Prucka, Morgan Barron, Miriam Figueroa-Santos, Matthew Castanier  

**Link**: [PDF](https://arxiv.org/pdf/2507.04371)  

**Abstract**: Navigating complex, cluttered, and unstructured environments that are a priori unknown presents significant challenges for autonomous ground vehicles, particularly when operating with a limited field of view(FOV) resulting in frequent occlusion and unobserved space. This paper introduces a novel visibility-aware model predictive path integral framework(VA-MPPI). Formulated as a dual control problem where perceptual uncertainties and control decisions are intertwined, it reasons over perception uncertainty evolution within a unified planning and control pipeline. Unlike traditional methods that rely on explicit uncertainty objectives, the VA-MPPI controller implicitly balances exploration and exploitation, reducing uncertainty only when system performance would be increased. The VA-MPPI framework is evaluated in simulation against deterministic and prescient controllers across multiple scenarios, including a cluttered urban alleyway and an occluded off-road environment. The results demonstrate that VA-MPPI significantly improves safety by reducing collision with unseen obstacles while maintaining competitive performance. For example, in the off-road scenario with 400 control samples, the VA-MPPI controller achieved a success rate of 84%, compared to only 8% for the deterministic controller, with all VA-MPPI failures arising from unmet stopping criteria rather than collisions. Furthermore, the controller implicitly avoids unobserved space, improving safety without explicit directives. The proposed framework highlights the potential for robust, visibility-aware navigation in unstructured and occluded environments, paving the way for future advancements in autonomous ground vehicle systems. 

**Abstract (ZH)**: 面向未知复杂、拥挤和无结构环境的自主地面车辆导航：一种新的基于视线的模型预测路径积分框架（VA-MPPI） 

---
# MLLM-Fabric: Multimodal Large Language Model-Driven Robotic Framework for Fabric Sorting and Selection 

**Title (ZH)**: MLLM- Fabric：多模态大语言模型驱动的机器人框架用于面料分类与选择 

**Authors**: Liman Wang, Hanyang Zhong, Tianyuan Wang, Shan Luo, Jihong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04351)  

**Abstract**: Choosing the right fabric is crucial to meet functional and quality requirements in robotic applications for textile manufacturing, apparel production, and smart retail. We present MLLM-Fabric, a robotic framework powered by multimodal large language models (MLLMs) for fabric sorting and selection. The system includes a robotic arm, a camera, a visuotactile sensor, and a pressure sensor. It employs supervised fine-tuning and multimodal explanation-guided knowledge distillation to accurately classify and rank fabric properties. To facilitate further research, we release a dataset of 220 unique fabric samples, including RGB images and synchronized visuotactile and pressure data. Experimental results show that our Fabric-Llama-90B model consistently outperforms pretrained vision-language baselines in both property ranking accuracy and selection reliability. 

**Abstract (ZH)**: 选择合适的面料对于纺织制造业、服饰生产以及智能零售中的机器人应用而言至关重要。我们提出了MLLM-Fabric，一种基于多模态大型语言模型的机器人框架，用于面料分类与选择。该系统包括一个机器人臂、一个摄像头、一个视触觉传感器和一个压力传感器。它通过监督微调和多模态解释引导的知识蒸馏来准确分类和排序面料属性。为了促进进一步的研究，我们发布了包含220个独特面料样本的数据集，其中包括RGB图像和同步的视触觉及压力数据。实验结果表明，我们的Fabric-Llama-90B模型在属性排序准确性和选择可靠性方面均优于预训练的视觉-语言基线模型。 

---
# Robot-assisted Transcranial Magnetic Stimulation (Robo-TMS): A Review 

**Title (ZH)**: 机器人辅助经颅磁刺激（Robo-TMS）：一个综述 

**Authors**: Wenzhi Bai, Andrew Weightman, Rory J O Connor, Zhengtao Ding, Mingming Zhang, Sheng Quan Xie, Zhenhong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.04345)  

**Abstract**: Transcranial magnetic stimulation (TMS) is a non-invasive and safe brain stimulation procedure with growing applications in clinical treatments and neuroscience research. However, achieving precise stimulation over prolonged sessions poses significant challenges. By integrating advanced robotics with conventional TMS, robot-assisted TMS (Robo-TMS) has emerged as a promising solution to enhance efficacy and streamline procedures. Despite growing interest, a comprehensive review from an engineering perspective has been notably absent. This paper systematically examines four critical aspects of Robo-TMS: hardware and integration, calibration and registration, neuronavigation systems, and control systems. We review state-of-the-art technologies in each area, identify current limitations, and propose future research directions. Our findings suggest that broader clinical adoption of Robo-TMS is currently limited by unverified clinical applicability, high operational complexity, and substantial implementation costs. Emerging technologies, including marker-less tracking, non-rigid registration, learning-based electric field (E-field) modelling, individualised magnetic resonance imaging (MRI) generation, robot-assisted multi-locus TMS (Robo-mTMS), and automated calibration and registration, present promising pathways to address these challenges. 

**Abstract (ZH)**: 经颅磁刺激（TMS）是一种非侵入性和安全性的脑刺激程序，在临床治疗和神经科学研究中的应用日益增多。然而，在长时间刺激中实现精确的刺激面临重大挑战。通过将先进机器人技术与传统TMS结合，机器人辅助TMS（Robo-TMS）已成为提高疗效和简化程序的有前景的解决方案。尽管研究兴趣日益增加，但缺乏从工程角度进行的全面综述。本文系统地探讨了Robo-TMS的四个关键方面：硬件与集成、校准与注册、神经导航系统和控制系统。我们复习了每个领域的先进技术，确定了当前的限制，并提出了未来研究方向。我们的研究结果表明，目前Robo-TMS在临床中的广泛应用受到未经验证的临床应用性、高操作复杂性和显著的实施成本的限制。新兴技术，包括无标记跟踪、非刚性配准、基于学习的电场（E场）建模、个性化磁共振成像（MRI）生成、机器人辅助多靶点TMS（Robo-mTMS）以及自动校准和配准，为解决这些挑战提供了有前景的途径。 

---
# Wavelet Policy: Lifting Scheme for Policy Learning in Long-Horizon Tasks 

**Title (ZH)**: 小波策略：长时_horizon任务中的策略学习提升方案 

**Authors**: Hao Huang, Shuaihang Yuan, Geeta Chandra Raju Bethala, Congcong Wen, Anthony Tzes, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04331)  

**Abstract**: Policy learning focuses on devising strategies for agents in embodied artificial intelligence systems to perform optimal actions based on their perceived states. One of the key challenges in policy learning involves handling complex, long-horizon tasks that require managing extensive sequences of actions and observations with multiple modes. Wavelet analysis offers significant advantages in signal processing, notably in decomposing signals at multiple scales to capture both global trends and fine-grained details. In this work, we introduce a novel wavelet policy learning framework that utilizes wavelet transformations to enhance policy learning. Our approach leverages learnable multi-scale wavelet decomposition to facilitate detailed observation analysis and robust action planning over extended sequences. We detail the design and implementation of our wavelet policy, which incorporates lifting schemes for effective multi-resolution analysis and action generation. This framework is evaluated across multiple complex scenarios, including robotic manipulation, self-driving, and multi-robot collaboration, demonstrating the effectiveness of our method in improving the precision and reliability of the learned policy. 

**Abstract (ZH)**: 基于小波变换的策略学习框架：多尺度分析在复杂任务中的应用 

---
# Lidar Variability: A Novel Dataset and Comparative Study of Solid-State and Spinning Lidars 

**Title (ZH)**: LiDAR 可变性：一种新型数据集及固态与旋转 LiDAR 的比较研究 

**Authors**: Doumegna Mawuto Koudjo Felix, Xianjia Yu, Jiaqiang Zhang, Sier Ha, Zhuo Zou, Tomi Westerlund  

**Link**: [PDF](https://arxiv.org/pdf/2507.04321)  

**Abstract**: Lidar technology has been widely employed across various applications, such as robot localization in GNSS-denied environments and 3D reconstruction. Recent advancements have introduced different lidar types, including cost-effective solid-state lidars such as the Livox Avia and Mid-360. The Mid-360, with its dome-like design, is increasingly used in portable mapping and unmanned aerial vehicle (UAV) applications due to its low cost, compact size, and reliable performance. However, the lack of datasets that include dome-shaped lidars, such as the Mid-360, alongside other solid-state and spinning lidars significantly hinders the comparative evaluation of novel approaches across platforms. Additionally, performance differences between low-cost solid-state and high-end spinning lidars (e.g., Ouster OS series) remain insufficiently examined, particularly without an Inertial Measurement Unit (IMU) in odometry.
To address this gap, we introduce a novel dataset comprising data from multiple lidar types, including the low-cost Livox Avia and the dome-shaped Mid-360, as well as high-end spinning lidars such as the Ouster series. Notably, to the best of our knowledge, no existing dataset comprehensively includes dome-shaped lidars such as Mid-360 alongside both other solid-state and spinning lidars. In addition to the dataset, we provide a benchmark evaluation of state-of-the-art SLAM algorithms applied to this diverse sensor data. Furthermore, we present a quantitative analysis of point cloud registration techniques, specifically point-to-point, point-to-plane, and hybrid methods, using indoor and outdoor data collected from the included lidar systems. The outcomes of this study establish a foundational reference for future research in SLAM and 3D reconstruction across heterogeneous lidar platforms. 

**Abstract (ZH)**: 激光雷达技术已在各种应用中得到广泛应用，如GNSS拒止环境中机器人定位和三维重建。近期进展引入了不同类型的激光雷达，包括低成本的固态激光雷达，如Livox Avia和Mid-360。Mid-360因其球顶状设计，低成本、紧凑尺寸和可靠性能，越来越多地应用于便携式测绘和无人机（UAV）应用。然而，缺乏包含Mid-360等球顶状激光雷达的数据集，使得与其他固态和旋转激光雷达的平台间新型方法比较评价受到限制。此外，低成本固态激光雷达和高端旋转激光雷达（如Ouster OS系列）之间的性能差异，在没有惯性测量单元（IMU）的情况下用于里程计时，仍有待进一步研究。

为此，我们引入了一个新的数据集，包括低成本的Livox Avia和球顶状的Mid-360，以及高端旋转激光雷达如Ouster系列的数据。据我们所知，目前没有现有的数据集全面包括Mid-360等球顶状激光雷达以及其他固态和旋转激光雷达。除了数据集之外，我们还提供了这些多传感器数据上最先进的SLAM算法的基准评估。此外，我们展示了点云对齐技术的定量分析，特别是点到点、点到面和混合方法，使用从包含的激光雷达系统获取的室内和室外数据。研究结果为未来在异构激光雷达平台上的SLAM和三维重建研究奠定了基础。

标题：一种包含不同类型激光雷达的新型数据集及其在SLAM和3D重建中的基准评估 

---
# Hardware-Free Event Cameras Temporal Synchronization Based on Event Density Alignment 

**Title (ZH)**: 基于事件密度对齐的无硬件事件相机时间同步 

**Authors**: Wenxuan Li, Yan Dong, Shaoqiang Qiu, Bin Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.04314)  

**Abstract**: Event cameras are a novel type of sensor designed for capturing the dynamic changes of a scene. Due to factors such as trigger and transmission delays, a time offset exists in the data collected by multiple event cameras, leading to inaccurate information fusion. Thus, the collected data needs to be synchronized to overcome any potential time offset issue. Hardware synchronization methods require additional circuits, while certain models of event cameras (e.g., CeleX5) do not support hardware synchronization. Therefore, this paper proposes a hardware-free event camera synchronization method. This method determines differences between start times by minimizing the dissimilarity of the event density distributions of different event cameras and synchronizes the data by adjusting timestamps. The experiments demonstrate that the method's synchronization error is less than 10ms under various senses with multiple models of event cameras. 

**Abstract (ZH)**: 基于事件的摄像机无硬件同步方法及其应用 

---
# Vibration-aware Lidar-Inertial Odometry based on Point-wise Post-Undistortion Uncertainty 

**Title (ZH)**: 基于点级后去畸变不确定性的一种振动感知lidar-惯性里程计 

**Authors**: Yan Dong, Enci Xu, Shaoqiang Qiu, Wenxuan Li, Yang Liu, Bin Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.04311)  

**Abstract**: High-speed ground robots moving on unstructured terrains generate intense high-frequency vibrations, leading to LiDAR scan distortions in Lidar-inertial odometry (LIO). Accurate and efficient undistortion is extremely challenging due to (1) rapid and non-smooth state changes during intense vibrations and (2) unpredictable IMU noise coupled with a limited IMU sampling frequency. To address this issue, this paper introduces post-undistortion uncertainty. First, we model the undistortion errors caused by linear and angular vibrations and assign post-undistortion uncertainty to each point. We then leverage this uncertainty to guide point-to-map matching, compute uncertainty-aware residuals, and update the odometry states using an iterated Kalman filter. We conduct vibration-platform and mobile-platform experiments on multiple public datasets as well as our own recordings, demonstrating that our method achieves better performance than other methods when LiDAR undergoes intense vibration. 

**Abstract (ZH)**: 高速地面机器人在未结构化地形上移动会产生强烈的高频振动，导致激光雷达扫描在激光雷达-惯性里程计（LIO）中产生失真。由于（1）强烈振动期间快速且非光滑的状态变化，以及（2）与有限的惯性测量单元（IMU）采样频率结合的不可预测的IMU噪声，精确和高效的去失真非常具有挑战性。为解决这一问题，本文引入了后处理去失真的不确定性。首先，我们建模了由于线性振动和角振动引起的去失真误差，并为每个点分配后处理去失真的不确定性。然后，我们利用这种不确定性指导点到地图匹配，计算不确定性感知的残差，并使用迭代卡尔曼滤波器更新里程计状态。我们在多个公开数据集以及我们自己的记录上进行了振动平台和移动平台实验，证明当激光雷达经历强烈振动时，我们的方法优于其他方法。 

---
# AutoLayout: Closed-Loop Layout Synthesis via Slow-Fast Collaborative Reasoning 

**Title (ZH)**: AutoLayout: 通过慢速-快速协作推理实现闭环布局合成 

**Authors**: Weixing Chen, Dafeng Chi, Yang Liu, Yuxi Yang, Yexin Zhang, Yuzheng Zhuang, Xingyue Quan, Jianye Hao, Guanbin Li, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04293)  

**Abstract**: The automated generation of layouts is vital for embodied intelligence and autonomous systems, supporting applications from virtual environment construction to home robot deployment. Current approaches, however, suffer from spatial hallucination and struggle with balancing semantic fidelity and physical plausibility, often producing layouts with deficits such as floating or overlapping objects and misaligned stacking relation. In this paper, we propose AutoLayout, a fully automated method that integrates a closed-loop self-validation process within a dual-system framework. Specifically, a slow system harnesses detailed reasoning with a Reasoning-Reflection-Generation (RRG) pipeline to extract object attributes and spatial constraints. Then, a fast system generates discrete coordinate sets and a topological relation set that are jointly validated. To mitigate the limitations of handcrafted rules, we further introduce an LLM-based Adaptive Relation Library (ARL) for generating and evaluating layouts. Through the implementation of Slow-Fast Collaborative Reasoning, the AutoLayout efficiently generates layouts after thorough deliberation, effectively mitigating spatial hallucination. Its self-validation mechanism establishes a closed-loop process that iteratively corrects potential errors, achieving a balance between physical stability and semantic consistency. The effectiveness of AutoLayout was validated across 8 distinct scenarios, where it demonstrated a significant 10.1% improvement over SOTA methods in terms of physical plausibility, semantic consistency, and functional completeness. 

**Abstract (ZH)**: 自动布局生成对于具身智能和自主系统至关重要，支持从虚拟环境构建到家庭机器人部署的各种应用。然而，当前的方法面临着空间幻觉的问题，并且难以在语义准确性和物理可行性之间取得平衡，经常产生诸如浮空对象、重叠对象和对齐不当堆叠关系的布局缺陷。在本文中，我们提出了一种名为AutoLayout的全自动方法，该方法在双系统框架内集成了一个闭环自验证过程。具体而言，慢系统利用推理-反思-生成（RRG）管道进行详细的推理提取对象属性和空间约束。然后，快系统生成离散坐标集和拓扑关系集，并联合验证。为了减轻手工艺规则的局限性，我们进一步引入了基于LLM的自适应关系库（ARL）来生成和评估布局。通过实施Slow-Fast协作推理，AutoLayout在深入讨论后高效地生成布局，有效减轻了空间幻觉。其自验证机制建立了一个闭环过程，迭代纠正潜在错误，实现了物理稳定性和语义一致性之间的平衡。AutoLayout在8种不同场景中的有效性得到了验证，与SOTA方法相比，在物理可行性、语义一致性和功能完整性方面分别实现了10.1%的显著改进。 

---
# SRefiner: Soft-Braid Attention for Multi-Agent Trajectory Refinement 

**Title (ZH)**: SRefiner: 软交织注意力机制在多agents轨迹精修中的应用 

**Authors**: Liwen Xiao, Zhiyu Pan, Zhicheng Wang, Zhiguo Cao, Wei Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.04263)  

**Abstract**: Accurate prediction of multi-agent future trajectories is crucial for autonomous driving systems to make safe and efficient decisions. Trajectory refinement has emerged as a key strategy to enhance prediction accuracy. However, existing refinement methods often overlook the topological relationships between trajectories, which are vital for improving prediction precision. Inspired by braid theory, we propose a novel trajectory refinement approach, Soft-Braid Refiner (SRefiner), guided by the soft-braid topological structure of trajectories using Soft-Braid Attention. Soft-Braid Attention captures spatio-temporal topological relationships between trajectories by considering both spatial proximity and vehicle motion states at ``soft intersection points". Additionally, we extend this approach to model interactions between trajectories and lanes, further improving the prediction accuracy. SRefiner is a multi-iteration, multi-agent framework that iteratively refines trajectories, incorporating topological information to enhance interactions within traffic scenarios. SRefiner achieves significant performance improvements over four baseline methods across two datasets, establishing a new state-of-the-art in trajectory refinement. Code is here this https URL. 

**Abstract (ZH)**: 多AGENT未来轨迹的准确预测是自动驾驶系统作出安全高效决策的关键。轨迹精炼已经成为了提高预测精度的关键策略。然而，现有的精炼方法往往忽略了轨迹之间的拓扑关系，而这些关系对于提高预测精度至关重要。受辫理论启发，我们提出了一种新的轨迹精炼方法——Soft-Braid Refiner (SRefiner)，该方法通过Soft-Braid注意力机制利用轨迹的软辫拓扑结构来指导轨迹精炼。Soft-Braid注意力机制通过考虑轨迹间的时空拓扑关系以及车辆在“软交叉点”的空间临近和运动状态，捕捉轨迹之间的时空拓扑关系。此外，我们还将此方法扩展到轨迹与车道之间的交互建模，进一步提高预测精度。SRefiner是一个多迭代、多AGENT框架，通过迭代精炼轨迹并整合拓扑信息，增强交通场景中的交互。SRefiner在两个数据集上的性能显著优于四种基线方法，建立了轨迹精炼的新最先进的水平。代码详见此链接：https://xxxx.xxx/ 

---
# Optimal Scheduling of a Dual-Arm Robot for Efficient Strawberry Harvesting in Plant Factories 

**Title (ZH)**: 双臂机器人在植物工厂中草莓高效采摘的优化调度 

**Authors**: Yuankai Zhu, Wenwu Lu, Guoqiang Ren, Yibin Ying, Stavros Vougioukas, Chen Peng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04240)  

**Abstract**: Plant factory cultivation is widely recognized for its ability to optimize resource use and boost crop yields. To further increase the efficiency in these environments, we propose a mixed-integer linear programming (MILP) framework that systematically schedules and coordinates dual-arm harvesting tasks, minimizing the overall harvesting makespan based on pre-mapped fruit locations. Specifically, we focus on a specialized dual-arm harvesting robot and employ pose coverage analysis of its end effector to maximize picking reachability. Additionally, we compare the performance of the dual-arm configuration with that of a single-arm vehicle, demonstrating that the dual-arm system can nearly double efficiency when fruit densities are roughly equal on both sides. Extensive simulations show a 10-20% increase in throughput and a significant reduction in the number of stops compared to non-optimized methods. These results underscore the advantages of an optimal scheduling approach in improving the scalability and efficiency of robotic harvesting in plant factories. 

**Abstract (ZH)**: 植物工厂环境下基于混合整数线性规划的双臂采收任务调度框架及其性能分析 

---
# Design Optimization of Three-Dimensional Wire Arrangement Considering Wire Crossings for Tendon-driven Robots 

**Title (ZH)**: 考虑钢丝交叉的 tendon 驱动机器人三维钢丝排布设计优化 

**Authors**: Kento Kawaharazuka, Shintaro Inoue, Yuta Sahara, Keita Yoneda, Temma Suzuki, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2507.04235)  

**Abstract**: Tendon-driven mechanisms are useful from the perspectives of variable stiffness, redundant actuation, and lightweight design, and they are widely used, particularly in hands, wrists, and waists of robots. The design of these wire arrangements has traditionally been done empirically, but it becomes extremely challenging when dealing with complex structures. Various studies have attempted to optimize wire arrangement, but many of them have oversimplified the problem by imposing conditions such as restricting movements to a 2D plane, keeping the moment arm constant, or neglecting wire crossings. Therefore, this study proposes a three-dimensional wire arrangement optimization that takes wire crossings into account. We explore wire arrangements through a multi-objective black-box optimization method that ensures wires do not cross while providing sufficient joint torque along a defined target trajectory. For a 3D link structure, we optimize the wire arrangement under various conditions, demonstrate its effectiveness, and discuss the obtained design solutions. 

**Abstract (ZH)**: 腱驱动机制从可变刚度、冗余驱动和轻量化设计的角度来看是很有用的，并且在机器人手部、手腕和腰部等部位得到了广泛应用。这些绳索布置的传统设计方法通常是经验性的，但在处理复杂结构时变得极其具有挑战性。尽管有多种研究尝试优化绳索布置，但许多研究通过限制运动到二维平面、保持力臂不变或忽略绳索交叉等方式简化了问题。因此，本研究提出了一种三维绳索布置优化方法，该方法考虑了绳索交叉的情况，并通过多目标黑箱优化方法确保绳索不交叉的同时沿定义的目标轨迹提供足够的关节扭矩。对于三维连杆结构，我们根据不同条件优化了绳索布置，并展示了其有效性，讨论了获得的设计解决方案。 

---
# Efficient Learning of A Unified Policy For Whole-body Manipulation and Locomotion Skills 

**Title (ZH)**: 整体 manipulation 和运动技能统一策略的高效学习 

**Authors**: Dianyong Hou, Chengrui Zhu, Zhen Zhang, Zhibin Li, Chuang Guo, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04229)  

**Abstract**: Equipping quadruped robots with manipulators provides unique loco-manipulation capabilities, enabling diverse practical applications. This integration creates a more complex system that has increased difficulties in modeling and control. Reinforcement learning (RL) offers a promising solution to address these challenges by learning optimal control policies through interaction. Nevertheless, RL methods often struggle with local optima when exploring large solution spaces for motion and manipulation tasks. To overcome these limitations, we propose a novel approach that integrates an explicit kinematic model of the manipulator into the RL framework. This integration provides feedback on the mapping of the body postures to the manipulator's workspace, guiding the RL exploration process and effectively mitigating the local optima issue. Our algorithm has been successfully deployed on a DeepRobotics X20 quadruped robot equipped with a Unitree Z1 manipulator, and extensive experimental results demonstrate the superior performance of this approach. 

**Abstract (ZH)**: 装备 manipulator 的四足机器人提供了独特的 locomo-manipulation 能力，使其能够在多种实际应用中发挥作用。这种集成增加了系统的复杂性，给建模和控制带来了更大的难度。强化学习 (RL) 通过交互学习最优控制策略，为应对这些挑战提供了有希望的解决方案。然而，当在动作和操作任务的大解决方案空间中探索时，RL 方法往往难以摆脱局部最优。为克服这些限制，我们提出了一种新的方法，将 manipulator 的显式运动学模型集成到 RL 框架中。这种集成提供了有关机器人姿态与 manipulator 工作空间映射的反馈，指导 RL 探索过程，并有效地缓解了局部最优的问题。我们的算法已在装备 Unitree Z1 manipulator 的 DeepRobotics X20 四足机器人上成功实现，并且广泛的实验结果表明了该方法的优越性能。 

---
# An improved 2D time-to-collision for articulated vehicles: predicting sideswipe and rear-end collisions 

**Title (ZH)**: 改进的 articulated 车辆 2D 时光撞对于侧面碰撞和追尾碰撞的预测 

**Authors**: Abhijeet Behera, Sogol Kharrazi, Erik Frisk, Maytheewat Aramrattana  

**Link**: [PDF](https://arxiv.org/pdf/2507.04184)  

**Abstract**: Time-to-collision (TTC) is a widely used measure for estimating the time until a rear-end collision between two vehicles, assuming both maintain constant speeds and headings in the prediction horizon. To also capture sideswipe collisions, a two-dimensional extension, TTC$_{\text{2D}}$, was introduced. However, this formulation assumes both vehicles have the same heading and that their headings remain unchanged during the manoeuvre, in addition to the standard assumptions on the prediction horizon. Moreover, its use for articulated vehicles like a tractor-semitrailer remains unclear. This paper addresses these limitations by developing three enhanced versions of TTC$_{\text{2D}}$. The first incorporates vehicle heading information, which is missing in the original formulation. The standard assumption of constant speed and heading in the prediction horizon holds. The second adapts this to articulated vehicles while retaining the assumptions of the first version. The third version maintains the constant heading assumption but relaxes the constant speed assumption by allowing constant acceleration. The versions are tested in a cut-in scenario using the CARLA simulation environment. They detect rear-end collisions, similar to TTC, and moreover, they also identify sideswipe risks, something TTC could not predict. 

**Abstract (ZH)**: 时间碰撞(TTC)及其二维扩展在车辆碰撞预警中的改进及应用 

---
# Comparative Evaluation of VR-Enabled Robots and Human Operators for Targeted Disease Management in Vineyards 

**Title (ZH)**: 基于VR的机器人与人类操作员在葡萄园中针对特定疾病管理的 comparative evaluation 

**Authors**: Hasan Seyyedhasani, Daniel Udekwe, Muhammad Ali Qadri  

**Link**: [PDF](https://arxiv.org/pdf/2507.04167)  

**Abstract**: This study explores the use of immersive virtual reality (VR) as a control interface for agricultural robots in vineyard disease detection and treatment. Using a Unity-ROS simulation, it compares three agents: a human operator, an immersive VR-controlled robot, and a non-immersive VR-controlled robot. During the scanning phase, humans perform best due to agility and control speed. However, in the treatment phase, immersive VR robots outperform others, completing tasks up to 65% faster by using stored infection data and optimized path planning. In yield-map-based navigation, immersive robots are also 38% faster than humans. Despite slower performance in manual scanning tasks, immersive VR excels in memory-guided, repetitive operations. The study highlights the role of interface design and path optimization, noting limitations in simulation fidelity and generalizability. It concludes that immersive VR has strong potential to enhance efficiency and precision in precision agriculture. 

**Abstract (ZH)**: 本研究探索沉浸式虚拟现实（VR）作为控制界面在葡萄园疾病检测和治疗中用于农业机器人应用的可能性。通过Unity-ROS模拟，本研究比较了三种代理：人类操作员、沉浸式VR控制的机器人和非沉浸式VR控制的机器人。在扫描阶段，人类表现最好，因为具有灵活性和控制速度。但在治疗阶段，沉浸式VR机器人表现出色，通过使用存储的感染数据和优化的路径规划，任务完成速度比其他机器人快65%。在基于产量图的导航中，沉浸式机器人也比人类快38%。尽管在手动扫描任务中表现较慢，但沉浸式VR在基于记忆的重复操作中表现出色。本研究强调了界面设计和路径优化的作用，并指出模拟保真度和泛化能力的局限性。研究结论认为，沉浸式VR有很强的潜力提高精准农业中的效率和精度。 

---
# Learning Humanoid Arm Motion via Centroidal Momentum Regularized Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于质心动量正则化的多智能体强化学习的人形臂运动学习 

**Authors**: Ho Jae Lee, Se Hwan Jeon, Sangbae Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.04140)  

**Abstract**: Humans naturally swing their arms during locomotion to regulate whole-body dynamics, reduce angular momentum, and help maintain balance. Inspired by this principle, we present a limb-level multi-agent reinforcement learning (RL) framework that enables coordinated whole-body control of humanoid robots through emergent arm motion. Our approach employs separate actor-critic structures for the arms and legs, trained with centralized critics but decentralized actors that share only base states and centroidal angular momentum (CAM) observations, allowing each agent to specialize in task-relevant behaviors through modular reward design. The arm agent guided by CAM tracking and damping rewards promotes arm motions that reduce overall angular momentum and vertical ground reaction moments, contributing to improved balance during locomotion or under external perturbations. Comparative studies with single-agent and alternative multi-agent baselines further validate the effectiveness of our approach. Finally, we deploy the learned policy on a humanoid platform, achieving robust performance across diverse locomotion tasks, including flat-ground walking, rough terrain traversal, and stair climbing. 

**Abstract (ZH)**: 人类在运动过程中自然摆动双臂以调节全身动力学、减少角动量并帮助维持平衡。受此启发，我们提出了一种肢体级多智能体强化学习（RL）框架，通过涌现的臂部运动实现类人机器人全身协调控制。该方法为臂和腿分别采用独立的	actor-critic结构，使用集中式批评家但分散式演员，后者仅共享基础状态和质心角动量（CAM）观察结果，通过模块化的奖励设计使每个代理专注于与任务相关的行为。由CAM跟踪和阻尼奖励引导的臂部代理促进了减少总体角动量和地面反作用力矩的臂部运动，从而在步行或外部扰动下改善了平衡。与单智能体和替代多智能体基线的比较研究进一步验证了该方法的有效性。最终，我们在类人平台上部署所学习的策略，实现了在多种运动任务中稳健的性能，包括平坦地面步行、崎岖地形穿越和楼梯攀爬。 

---
# Are Learning-Based Approaches Ready for Real-World Indoor Navigation? A Case for Imitation Learning 

**Title (ZH)**: 基于学习的方法ready for实际室内导航了吗？模仿学习的案例分析 

**Authors**: Nigitha Selvaraj, Alex Mitrevski, Sebastian Houben  

**Link**: [PDF](https://arxiv.org/pdf/2507.04086)  

**Abstract**: Traditional indoor robot navigation methods provide a reliable solution when adapted to constrained scenarios, but lack flexibility or require manual re-tuning when deployed in more complex settings. In contrast, learning-based approaches learn directly from sensor data and environmental interactions, enabling easier adaptability. While significant work has been presented in the context of learning navigation policies, learning-based methods are rarely compared to traditional navigation methods directly, which is a problem for their ultimate acceptance in general navigation contexts. In this work, we explore the viability of imitation learning (IL) for indoor navigation, using expert (joystick) demonstrations to train various navigation policy networks based on RGB images, LiDAR, and a combination of both, and we compare our IL approach to a traditional potential field-based navigation method. We evaluate the approach on a physical mobile robot platform equipped with a 2D LiDAR and a camera in an indoor university environment. Our multimodal model demonstrates superior navigation capabilities in most scenarios, but faces challenges in dynamic environments, likely due to limited diversity in the demonstrations. Nevertheless, the ability to learn directly from data and generalise across layouts suggests that IL can be a practical navigation approach, and potentially a useful initialisation strategy for subsequent lifelong learning. 

**Abstract (ZH)**: 基于模仿学习的室内导航研究：将专家演示用于RGB图像、LiDAR及两者结合的导航策略训练并与传统势场导航方法对比 

---
# Generalized Locomotion in Out-of-distribution Conditions with Robust Transformer 

**Title (ZH)**: 在分布外条件下具有稳健性的通用运动学习 

**Authors**: Lingxiao Guo, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04039)  

**Abstract**: To succeed in the real world, robots must deal with situations that differ from those seen during training. Those out-of-distribution situations for legged robot mainly include challenging dynamic gaps and perceptual gaps. Here we study the problem of robust locomotion in such novel situations. While previous methods usually rely on designing elaborate training and adaptation techniques, we approach the problem from a network model perspective. Our approach, RObust Locomotion Transformer(ROLT),a variation of transformer,could achieve robustness in a variety of unseen conditions. ROLT introduces two key designs: body tokenization and consistent dropout. Body tokenization supports knowledge share across different limbs, which boosts generalization ability of the network. Meanwhile, a novel dropout strategy enhances the policy's robustness to unseen perceptual noise. We conduct extensive experiments both on quadruped and hexapod robots. Results demonstrate that ROLT is more robust than existing methods. Although trained in only a few dynamic settings, the learned policy generalizes well to multiple unseen dynamic conditions. Additionally, despite training with clean observations, the model handles challenging corruption noise during testing. 

**Abstract (ZH)**: 面对未知情况的稳健腿部机器人运动研究：RObust Locomotion Transformer（ROLT）方法探究 

---
# Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM 

**Title (ZH)**: 高斯-LIC2：激光雷达-惯性-相机高斯散斑SLAM 

**Authors**: Xiaolei Lang, Jiajun Lv, Kai Tang, Laijian Li, Jianxin Huang, Lina Liu, Yong Liu, Xingxing Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2507.04004)  

**Abstract**: This paper proposes an innovative LiDAR-Inertial-Camera SLAM system with 3D Gaussian Splatting, which is the first to jointly consider visual quality, geometric accuracy, and real-time performance. It robustly and accurately estimates poses while building a photo-realistic 3D Gaussian map in real time that enables high-quality novel view RGB and depth rendering. To effectively address under-reconstruction in regions not covered by the LiDAR, we employ a lightweight zero-shot depth model that synergistically combines RGB appearance cues with sparse LiDAR measurements to generate dense depth maps. The depth completion enables reliable Gaussian initialization in LiDAR-blind areas, significantly improving system applicability for sparse LiDAR sensors. To enhance geometric accuracy, we use sparse but precise LiDAR depths to supervise Gaussian map optimization and accelerate it with carefully designed CUDA-accelerated strategies. Furthermore, we explore how the incrementally reconstructed Gaussian map can improve the robustness of odometry. By tightly incorporating photometric constraints from the Gaussian map into the continuous-time factor graph optimization, we demonstrate improved pose estimation under LiDAR degradation scenarios. We also showcase downstream applications via extending our elaborate system, including video frame interpolation and fast 3D mesh extraction. To support rigorous evaluation, we construct a dedicated LiDAR-Inertial-Camera dataset featuring ground-truth poses, depth maps, and extrapolated trajectories for assessing out-of-sequence novel view synthesis. Extensive experiments on both public and self-collected datasets demonstrate the superiority and versatility of our system across LiDAR sensors with varying sampling densities. Both the dataset and code will be made publicly available on project page this https URL. 

**Abstract (ZH)**: 本文提出了一种创新的LiDAR-惯性-摄像头SLAM系统，该系统首次综合考虑了视觉质量、几何精度和实时性能。该系统能够稳健且准确地估计姿态，同时实时构建逼真的3D高斯地图，使得能够生成高质量的新视图RGB和深度渲染。为有效解决LiDAR未覆盖区域的欠重建问题，我们采用一种轻量级的零样本深度模型，该模型能够协同结合RGB外观线索与稀疏的LiDAR测量生成密集深度图。深度补全在无LiDAR区域实现了可靠的高斯初始化，显著提高了系统对于稀疏LiDAR传感器的适用性。为了提高几何精度，我们使用稀疏但精确的LiDAR深度来监督高斯地图优化，并通过精心设计的CUDA加速策略加快优化过程。此外，我们探讨了逐步重构的高斯地图如何提高里程计的鲁棒性。通过将来自高斯地图的光度约束紧密集成到连续时间因子图优化中，我们展示了在LiDAR退化场景下改进的姿姿估计效果。我们还通过扩展我们的系统展示了下游应用，包括视频帧插值和快速3D网格提取。为了支持严格的评估，我们构建了一个专用的LiDAR-惯性-摄像头数据集，其中包含地面真实姿态、深度图和外推轨迹，用于评估新的视图合成性能。在公共数据集和自收集数据集上的广泛实验表明，我们的系统在各种采样密度的LiDAR传感器上具有优越性和通用性。数据集和代码将在项目页面 https://project-url.com 公开提供。 

---
# Scalable Learning of High-Dimensional Demonstrations with Composition of Linear Parameter Varying Dynamical Systems 

**Title (ZH)**: 高维演示的大规模学习：线性参数 varying 动态系统组成的方法 

**Authors**: Shreenabh Agrawal, Hugo T. M. Kussaba, Lingyun Chen, Allen Emmanuel Binny, Abdalla Swikir, Pushpak Jagtap, Sami Haddadin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03992)  

**Abstract**: Learning from Demonstration (LfD) techniques enable robots to learn and generalize tasks from user demonstrations, eliminating the need for coding expertise among end-users. One established technique to implement LfD in robots is to encode demonstrations in a stable Dynamical System (DS). However, finding a stable dynamical system entails solving an optimization problem with bilinear matrix inequality (BMI) constraints, a non-convex problem which, depending on the number of scalar constraints and variables, demands significant computational resources and is susceptible to numerical issues such as floating-point errors. To address these challenges, we propose a novel compositional approach that enhances the applicability and scalability of learning stable DSs with BMIs. 

**Abstract (ZH)**: 从演示学习的技术使机器人能够通过用户的演示学习和泛化任务，消除终用户编程知识的需要。一种在机器人中实现从演示学习的技术是将演示编码为稳定的动力学系统（DS）。然而，寻找稳定的动力学系统涉及求解带有双线性矩阵不等式（BMI）约束的优化问题，这是一个非凸问题，可能会因标量约束和变量的数量而需要大量的计算资源，并且容易出现浮点误差等数值问题。为解决这些问题，我们提出了一种新的组合方法，以增强使用BMI学习稳定DS的适用性和可扩展性。 

---
# Robust and Modular Multi-Limb Synchronization in Motion Stack for Space Robots with Trajectory Clamping via Hypersphere 

**Title (ZH)**: 基于轨迹约束的高维球面上的空间机器人多肢运动堆栈鲁棒模块化同步方法 

**Authors**: Elian Neppel, Ashutosh Mishra, Shamistan Karimov, Kentaro Uno, Shreya Santra, Kazuya Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2507.03934)  

**Abstract**: Modular robotics holds immense potential for space exploration, where reliability, repairability, and reusability are critical for cost-effective missions. Coordination between heterogeneous units is paramount for precision tasks -- whether in manipulation, legged locomotion, or multi-robot interaction. Such modular systems introduce challenges far exceeding those in monolithic robot architectures. This study presents a robust method for synchronizing the trajectories of multiple heterogeneous actuators, adapting dynamically to system variations with minimal system knowledge. This design makes it inherently robot-agnostic, thus highly suited for modularity. To ensure smooth trajectory adherence, the multidimensional state is constrained within a hypersphere representing the allowable deviation. The distance metric can be adapted hence, depending on the task and system under control, deformation of the constraint region is possible. This approach is compatible with a wide range of robotic platforms and serves as a core interface for Motion-Stack, our new open-source universal framework for limb coordination (available at this https URL ). The method is validated by synchronizing the end-effectors of six highly heterogeneous robotic limbs, evaluating both trajectory adherence and recovery from significant external disturbances. 

**Abstract (ZH)**: 模块化机器人在太空探索中具有巨大的潜力，可靠性、可维修性和可重复使用性对于经济有效的太空任务至关重要。异构模块之间的协调对于精确任务（无论是操作、腿足运动还是多机器人交互）至关重要。这类模块化系统引入了远超单一机器人架构的挑战。本研究提出了一种鲁棒的方法，用于同步多个异构执行器的轨迹，并能够根据最少的系统知识动态适应系统变化。该设计使其具有高度的机器人无关性，因此非常适合模块化。为了确保轨迹的平滑跟踪，多维状态被约束在表示允许偏差的超球体内。距离度量可以根据任务和受控系统进行调整，从而有可能改变约束区域的形状。该方法兼容广泛的机器人平台，并作为我们新的开源通用框架Motion-Stack的核心接口（可通过此链接访问：this https URL）得到了验证。该方法通过同步六个高度异构的机器人肢体末端执行器，同时评估轨迹跟踪能力和对外部显著干扰的恢复能力进行了验证。 

---
# RwoR: Generating Robot Demonstrations from Human Hand Collection for Policy Learning without Robot 

**Title (ZH)**: RwoR: 从人类手部采集生成机器人演示以在无机器人情况下进行策略学习 

**Authors**: Liang Heng, Xiaoqi Li, Shangqing Mao, Jiaming Liu, Ruolin Liu, Jingli Wei, Yu-Kai Wang, Yueru Jia, Chenyang Gu, Rui Zhao, Shanghang Zhang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03930)  

**Abstract**: Recent advancements in imitation learning have shown promising results in robotic manipulation, driven by the availability of high-quality training data. To improve data collection efficiency, some approaches focus on developing specialized teleoperation devices for robot control, while others directly use human hand demonstrations to obtain training this http URL, the former requires both a robotic system and a skilled operator, limiting scalability, while the latter faces challenges in aligning the visual gap between human hand demonstrations and the deployed robot this http URL address this, we propose a human hand data collection system combined with our hand-to-gripper generative model, which translates human hand demonstrations into robot gripper demonstrations, effectively bridging the observation this http URL, a GoPro fisheye camera is mounted on the human wrist to capture human hand this http URL then train a generative model on a self-collected dataset of paired human hand and UMI gripper demonstrations, which have been processed using a tailored data pre-processing strategy to ensure alignment in both timestamps and this http URL, given only human hand demonstrations, we are able to automatically extract the corresponding SE(3) actions and integrate them with high-quality generated robot demonstrations through our generation pipeline for training robotic policy this http URL experiments, the robust manipulation performance demonstrates not only the quality of the generated robot demonstrations but also the efficiency and practicality of our data collection this http URL demonstrations can be found at: this https URL 

**Abstract (ZH)**: 最近在模仿学习方面的进展展示了其在机器人操作中的有希望的结果，这得益于高质量训练数据的可用性。为了提高数据采集效率，一些方法专注于开发专门的远程操作设备以控制机器人，而另一些方法直接利用人类手部演示来获得训练数据。前者需要一个机器人系统和一个熟练的操作员，限制了其可扩展性，而后者则面临人类手部演示与部署机器人之间视觉差异的对齐问题。为了解决这个问题，我们提出了一种结合手到 gripper 生成模型的人类手部数据采集系统，该系统将人类手部演示转化为机器人 gripper 演示，有效地弥合了观察之间的差距。我们使用鱼眼相机安装在人类手腕上，捕捉人类手部演示。然后在一个自收集的人类手部和UMI gripper 演示配对数据集上训练生成模型，该数据集经过定制的数据预处理策略处理以确保在时间戳和空间上的对齐。仅给定人类手部演示，我们能够自动生成对应的SE(3)动作，并通过我们的生成管道与高质量的机器人演示集结合，用于训练机器人策略。实验结果不仅证明了生成的机器人演示质量，还展示了我们的数据采集方法的高效性和实用性。生成的演示可以在以下网址找到：[相关网址]。 

---
# Accurate Pose Estimation Using Contact Manifold Sampling for Safe Peg-in-Hole Insertion of Complex Geometries 

**Title (ZH)**: 基于接触流形采样的精确姿势估计方法及其在复杂几何结构销孔装配中的安全应用 

**Authors**: Abhay Negi, Omey M. Manyar, Dhanush K. Penmetsa, Satyandra K. Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.03925)  

**Abstract**: Robotic assembly of complex, non-convex geometries with tight clearances remains a challenging problem, demanding precise state estimation for successful insertion. In this work, we propose a novel framework that relies solely on contact states to estimate the full SE(3) pose of a peg relative to a hole. Our method constructs an online submanifold of contact states through primitive motions with just 6 seconds of online execution, subsequently mapping it to an offline contact manifold for precise pose estimation. We demonstrate that without such state estimation, robots risk jamming and excessive force application, potentially causing damage. We evaluate our approach on five industrially relevant, complex geometries with 0.1 to 1.0 mm clearances, achieving a 96.7% success rate - a 6x improvement over primitive-based insertion without state estimation. Additionally, we analyze insertion forces, and overall insertion times, showing our method significantly reduces the average wrench, enabling safer and more efficient assembly. 

**Abstract (ZH)**: 基于接触状态估计的复杂非凸几何结构紧密间隙下机器人装配新框架 

---
# DK-RRT: Deep Koopman RRT for Collision-Aware Motion Planning of Space Manipulators in Dynamic Debris Environments 

**Title (ZH)**: 基于深度科莫邦曼RRT的空间 manipulator 动态碎片环境中的避碰运动规划 

**Authors**: Qi Chen, Rui Liu, Kangtong Mo, Boli Zhang, Dezhi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03878)  

**Abstract**: Trajectory planning for robotic manipulators operating in dynamic orbital debris environments poses significant challenges due to complex obstacle movements and uncertainties. This paper presents Deep Koopman RRT (DK-RRT), an advanced collision-aware motion planning framework integrating deep learning with Koopman operator theory and Rapidly-exploring Random Trees (RRT). DK-RRT leverages deep neural networks to identify efficient nonlinear embeddings of debris dynamics, enhancing Koopman-based predictions and enabling accurate, proactive planning in real-time. By continuously refining predictive models through online sensor feedback, DK-RRT effectively navigates the manipulator through evolving obstacle fields. Simulation studies demonstrate DK-RRT's superior performance in terms of adaptability, robustness, and computational efficiency compared to traditional RRT and conventional Koopman-based planning, highlighting its potential for autonomous space manipulation tasks. 

**Abstract (ZH)**: 基于动态轨道 debris 环境下的机械臂轨迹规划面临着复杂障碍物运动和不确定性带来的显著挑战。本文提出了一种将深度学习、Koopman 操作符理论和随机树算法（RRT）相结合的先进防碰撞轨迹规划框架 Deep Koopman RRT（DK-RRT）。DK-RRT 利用深度神经网络识别高效的非线性 debris 动力学嵌入，增强基于 Koopman 的预测能力，实现实时的精确、主动规划。通过不断利用在线传感器反馈优化预测模型，DK-RRT 有效地引导机械臂穿越不断变化的障碍物场。仿真研究结果表明，相比于传统 RRT 和常规的基于 Koopman 的规划方法，DK-RRT 在适应性、鲁棒性和计算效率方面具有明显优势，展示了其在自主太空操作任务中的潜力。 

---
# Coil Geometry Learning for Short-Range Magnetic Actuation 

**Title (ZH)**: 短距离磁驱动的线圈几何结构学习 

**Authors**: Yuta Takahashi, Hayate Tajima, Shin-ichiro Sakai  

**Link**: [PDF](https://arxiv.org/pdf/2507.03806)  

**Abstract**: Fuel-free docking is a key operational technology for in-space assembly, resupplying space stations, sample return missions, and formation keeping of large-scale satellite swarms. The use of conventional propulsion systems, including thrusters, can cause adverse effects at short distances, such as sensor contamination, which may lead to the failure of the satellite or onboard equipment. The magnetic field interaction control generated by magnetorquers can overcome these weaknesses of propulsion. This actuation enables simultaneous control of attitude and formation control among desired satellite groups. The previous study typically uses the traditional dipole approximation model of the exact magnetic field to reduce computation cost. However, proximity operations often involve relatively short distances between satellites, which can easily compromise the effectiveness of this approximation. To avoid model errors that could result in satellite collisions, we utilize a magnetic field model described by Biot-Savart's law, without distance approximations (Near-field model), in consideration of short-distance operations. To overcome the high computational cost associated with the coil geometry and relative states information, a learning-based magnetic field approximation is derived, and its effectiveness is shown in the docking simulation of target and chaser satellites equipped with electromagnetic coils on three axes. Our method significantly reduces the computational cost of the exact magnetic model and possesses scalability that can accommodate an increasing number of target satellites through parallel processing. 

**Abstract (ZH)**: 无燃料对接是太空组装、空间站补给、样本返回任务以及大规模卫星编队保持的关键操作技术。为了克服常规推进系统，在短距离内可能导致传感器污染等不利影响，磁矩控制器产生的磁场相互作用控制能够克服推进系统的这些缺点。这种执行机构能够同时控制目标卫星群的姿态和编队控制。以往的研究通常使用精确磁场的传统偶极近似模型来降低计算成本。然而，近距离操作中卫星之间的相对距离较短，这容易破坏这种近似的有效性。为了避免模型误差导致的卫星碰撞，我们采用考虑短距离操作的毕奥-萨伐尔定律描述的磁场模型，不进行距离近似。为了克服与线圈几何形状和相对状态信息相关的高计算成本，我们推导了一种基于学习的磁场近似模型，并在装备了电磁线圈的追踪者和目标卫星的对接仿真中展示了其有效性。该方法显著降低了精确磁场模型的计算成本，并通过并行处理具有可扩展性，可以适应越来越多的目标卫星。 

---
# Multi-robot Aerial Soft Manipulator For Floating Litter Collection 

**Title (ZH)**: 基于浮游垃圾收集的多机器人空中软 manipulator 系统 

**Authors**: Antonio González-Morgado, Sander Smits, Guillermo Heredia, Anibal Ollero, Alexandre Krupa, François Chaumette, Fabien Spindler, Antonio Franchi, Chiara Gabellieri  

**Link**: [PDF](https://arxiv.org/pdf/2507.03517)  

**Abstract**: Removing floating litter from water bodies is crucial to preserving aquatic ecosystems and preventing environmental pollution. In this work, we present a multi-robot aerial soft manipulator for floating litter collection, leveraging the capabilities of aerial robots. The proposed system consists of two aerial robots connected by a flexible rope manipulator, which collects floating litter using a hook-based tool. Compared to single-aerial-robot solutions, the use of two aerial robots increases payload capacity and flight endurance while reducing the downwash effect at the manipulation point, located at the midpoint of the rope. Additionally, we employ an optimization-based rope-shape planner to compute the desired rope shape. The planner incorporates an adaptive behavior that maximizes grasping capabilities near the litter while minimizing rope tension when farther away. The computed rope shape trajectory is controlled by a shape visual servoing controller, which approximates the rope as a parabola. The complete system is validated in outdoor experiments, demonstrating successful grasping operations. An ablation study highlights how the planner's adaptive mechanism improves the success rate of the operation. Furthermore, real-world tests in a water channel confirm the effectiveness of our system in floating litter collection. These results demonstrate the potential of aerial robots for autonomous litter removal in aquatic environments. 

**Abstract (ZH)**: 基于无人机的柔性 manipulator 多机器人系统用于浮游垃圾收集 

---
# Evaluation of an Uncertainty-Aware Late Fusion Algorithm for Multi-Source Bird's Eye View Detections Under Controlled Noise 

**Title (ZH)**: 受控噪声环境下多源鸟瞰视角检测的不确定性意识晚期融合算法评估 

**Authors**: Maryem Fadili, Louis Lecrosnier, Steve Pechberti, Redouane Khemmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.03381)  

**Abstract**: Reliable multi-source fusion is crucial for robust perception in autonomous systems. However, evaluating fusion performance independently of detection errors remains challenging. This work introduces a systematic evaluation framework that injects controlled noise into ground-truth bounding boxes to isolate the fusion process. We then propose Unified Kalman Fusion (UniKF), a late-fusion algorithm based on Kalman filtering to merge Bird's Eye View (BEV) detections while handling synchronization issues. Experiments show that UniKF outperforms baseline methods across various noise levels, achieving up to 3x lower object's positioning and orientation errors and 2x lower dimension estimation errors, while maintaining nearperfect precision and recall between 99.5% and 100%. 

**Abstract (ZH)**: 可靠的多源融合对于自主系统中的稳健感知至关重要。然而，独立于检测误差评估融合性能仍然具有挑战性。本文介绍了系统性的评估框架，通过向地面 truth 边界框注入可控噪声来隔离融合过程。我们随后提出了一种基于卡尔曼滤波的晚期融合算法统一卡尔曼融合（UniKF），以解决同步问题的同时合并 Bird's Eye View (BEV) 检测。实验结果表明，在各种噪声水平下，UniKF 超过了基准方法，实现了定位和姿态误差降低多达 3 倍以及尺寸估计误差降低多达 2 倍，同时保持接近完美的精确率和召回率在 99.5% 至 100% 之间。 

---
# Label-Free Long-Horizon 3D UAV Trajectory Prediction via Motion-Aligned RGB and Event Cues 

**Title (ZH)**: 基于运动对齐的RGB和事件线索的无标签长时 horizon 3D UAV 轨迹预测 

**Authors**: Hanfang Liang, Shenghai Yuan, Fen Liu, Yizhuo Yang, Bing Wang, Zhuyu Huang, Chenyang Shi, Jing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03365)  

**Abstract**: The widespread use of consumer drones has introduced serious challenges for airspace security and public safety. Their high agility and unpredictable motion make drones difficult to track and intercept. While existing methods focus on detecting current positions, many counter-drone strategies rely on forecasting future trajectories and thus require more than reactive detection to be effective. To address this critical gap, we propose an unsupervised vision-based method for predicting the three-dimensional trajectories of drones. Our approach first uses an unsupervised technique to extract drone trajectories from raw LiDAR point clouds, then aligns these trajectories with camera images through motion consistency to generate reliable pseudo-labels. We then combine kinematic estimation with a visual Mamba neural network in a self-supervised manner to predict future drone trajectories. We evaluate our method on the challenging MMAUD dataset, including the V2 sequences that feature wide-field-of-view multimodal sensors and dynamic UAV motion in urban scenes. Extensive experiments show that our framework outperforms supervised image-only and audio-visual baselines in long-horizon trajectory prediction, reducing 5-second 3D error by around 40 percent without using any manual 3D labels. The proposed system offers a cost-effective, scalable alternative for real-time counter-drone deployment. All code will be released upon acceptance to support reproducible research in the robotics community. 

**Abstract (ZH)**: 消费者无人机的广泛使用为 airspace 安全和公共安全带来了严重挑战。它们的高机动性和不可预测的运动使无人机难以跟踪和拦截。现有方法主要集中在检测当前位置，而许多反无人机策略依赖于预测未来轨迹，因此需要更主动的检测方法才能有效。为解决这一关键缺口，我们提出了一种无监督的基于视觉的方法，用于预测无人机的三维轨迹。该方法首先使用无监督技术从原始 LiDAR 点云中提取无人机轨迹，然后通过运动一致性将这些轨迹与摄像头图像对齐，生成可靠的伪标签。接着，我们以自监督的方式将动力学估计与视觉 Mamba 神经网络结合，预测未来无人机的轨迹。我们在具有挑战性的 MMAUD 数据集上评估了我们的方法，包括 V2 序列，该序列包含宽视野多模态传感器和城市环境中动态无人机运动。大量实验证明，与监督图像-only 和多模态 baselines 相比，我们的框架在长时轨迹预测方面表现更佳，5 秒 3D 错误降低了约 40%，无需使用任何手动 3D 标记。该系统为实时反无人机部署提供了一种成本效益高、可扩展的替代方案。接受发表后，所有代码将被释放以支持机器人社区的可再现研究。 

---
# Robust and Efficient Embedded Convex Optimization through First-Order Adaptive Caching 

**Title (ZH)**: 通过一阶自适应缓存实现稳健且高效的嵌入式凸优化 

**Authors**: Ishaan Mahajan, Brian Plancher  

**Link**: [PDF](https://arxiv.org/pdf/2507.03231)  

**Abstract**: Recent advances in Model Predictive Control (MPC) leveraging a combination of first-order methods, such as the Alternating Direction Method of Multipliers (ADMM), and offline precomputation and caching of select operations, have excitingly enabled real-time MPC on microcontrollers. Unfortunately, these approaches require the use of fixed hyperparameters, limiting their adaptability and overall performance. In this work, we introduce First-Order Adaptive Caching, which precomputes not only select matrix operations but also their sensitivities to hyperparameter variations, enabling online hyperparameter updates without full recomputation of the cache. We demonstrate the effectiveness of our approach on a number of dynamic quadrotor tasks, achieving up to a 63.4% reduction in ADMM iterations over the use of optimized fixed hyperparameters and approaching 70% of the performance of a full cache recomputation, while reducing the computational cost from O(n^3) to O(n^2) complexity. This performance enables us to perform figure-eight trajectories on a 27g tiny quadrotor under wind disturbances. We release our implementation open-source for the benefit of the wider robotics community. 

**Abstract (ZH)**: 基于一阶方法自适应缓存的最近模型预测控制进展：在微控制器上实现实时模型预测控制 

---
# Dexterous Teleoperation of 20-DoF ByteDexter Hand via Human Motion Retargeting 

**Title (ZH)**: 通过人体运动重定位实现20-自由度ByteDexter手的灵巧远程操作 

**Authors**: Ruoshi Wen, Jiajun Zhang, Guangzeng Chen, Zhongren Cui, Min Du, Yang Gou, Zhigang Han, Junkai Hu, Liqun Huang, Hao Niu, Wei Xu, Haoxiang Zhang, Zhengming Zhu, Hang Li, Zeyu Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.03227)  

**Abstract**: Replicating human--level dexterity remains a fundamental robotics challenge, requiring integrated solutions from mechatronic design to the control of high degree--of--freedom (DoF) robotic hands. While imitation learning shows promise in transferring human dexterity to robots, the efficacy of trained policies relies on the quality of human demonstration data. We bridge this gap with a hand--arm teleoperation system featuring: (1) a 20--DoF linkage--driven anthropomorphic robotic hand for biomimetic dexterity, and (2) an optimization--based motion retargeting for real--time, high--fidelity reproduction of intricate human hand motions and seamless hand--arm coordination. We validate the system via extensive empirical evaluations, including dexterous in-hand manipulation tasks and a long--horizon task requiring the organization of a cluttered makeup table randomly populated with nine objects. Experimental results demonstrate its intuitive teleoperation interface with real--time control and the ability to generate high--quality demonstration data. Please refer to the accompanying video for further details. 

**Abstract (ZH)**: 复制人类级别的灵巧性仍然是一个基本的机器人挑战，需要从机电设计到高自由度（DoF）机器人手的控制等方面的综合解决方案。虽然模仿学习在将人类灵巧性转移给机器人方面前景广阔，但训练策略的有效性依赖于高质量的人类演示数据。我们通过一种手-臂远程操作系统来弥合这一差距，该系统包括：（1）一个20-DoF连杆驱动的人形机器人手以实现生物仿生灵巧性；（2）基于优化的运动重定位以实现实时、高保真的人类手部运动再现以及手-臂协调的无缝转换。我们通过广泛的实证评估验证了该系统，包括精细的手内操作任务以及一项长期任务，该任务要求将随机放置九个物品的杂乱化妆桌进行整理。实验结果证明了其直观的远程操作界面和实时控制能力，以及生成高质量演示数据的能力。请参见附带的视频以获取更多信息。 

---
# Image-driven Robot Drawing with Rapid Lognormal Movements 

**Title (ZH)**: 基于图像的机器人快速对数正态运动绘图 

**Authors**: Daniel Berio, Guillaume Clivaz, Michael Stroh, Oliver Deussen, Réjean Plamondon, Sylvain Calinon, Frederic Fol Leymarie  

**Link**: [PDF](https://arxiv.org/pdf/2507.03166)  

**Abstract**: Large image generation and vision models, combined with differentiable rendering technologies, have become powerful tools for generating paths that can be drawn or painted by a robot. However, these tools often overlook the intrinsic physicality of the human drawing/writing act, which is usually executed with skillful hand/arm gestures. Taking this into account is important for the visual aesthetics of the results and for the development of closer and more intuitive artist-robot collaboration scenarios. We present a method that bridges this gap by enabling gradient-based optimization of natural human-like motions guided by cost functions defined in image space. To this end, we use the sigma-lognormal model of human hand/arm movements, with an adaptation that enables its use in conjunction with a differentiable vector graphics (DiffVG) renderer. We demonstrate how this pipeline can be used to generate feasible trajectories for a robot by combining image-driven objectives with a minimum-time smoothing criterion. We demonstrate applications with generation and robotic reproduction of synthetic graffiti as well as image abstraction. 

**Abstract (ZH)**: 大规模图像生成与视觉模型结合差分渲染技术，可以通过梯度优化自然人类运动以指导生成可由机器人绘制的路径，同时考虑人类绘画/书写行为的内在物理特性，这对于视觉美学和促进更紧密更直观的人机艺术家协作场景的发展至关重要。我们提出了一种方法，通过使用sigma-lognormal手/臂运动模型及其适应性，使其能够与可微分向量图形渲染器（DiffVG）结合使用，从而实现基于图像空间定义的成本函数的梯度优化。我们展示如何通过结合图像驱动的目标和最短时间平滑准则来生成可行的机器人轨迹。我们展示了生成和机器人复制合成涂鸦以及图像抽象的应用。 

---
# Personalised Explanations in Long-term Human-Robot Interactions 

**Title (ZH)**: 长期人机互动中的个性化解释 

**Authors**: Ferran Gebellí, Anaís Garrell, Jan-Gerrit Habekost, Séverin Lemaignan, Stefan Wermter, Raquel Ros  

**Link**: [PDF](https://arxiv.org/pdf/2507.03049)  

**Abstract**: In the field of Human-Robot Interaction (HRI), a fundamental challenge is to facilitate human understanding of robots. The emerging domain of eXplainable HRI (XHRI) investigates methods to generate explanations and evaluate their impact on human-robot interactions. Previous works have highlighted the need to personalise the level of detail of these explanations to enhance usability and comprehension. Our paper presents a framework designed to update and retrieve user knowledge-memory models, allowing for adapting the explanations' level of detail while referencing previously acquired concepts. Three architectures based on our proposed framework that use Large Language Models (LLMs) are evaluated in two distinct scenarios: a hospital patrolling robot and a kitchen assistant robot. Experimental results demonstrate that a two-stage architecture, which first generates an explanation and then personalises it, is the framework architecture that effectively reduces the level of detail only when there is related user knowledge. 

**Abstract (ZH)**: 在人机交互（HRI）领域，一个基本挑战是促进人类对机器人的理解。可解释人机交互（XHRI）这一新兴领域研究生成解释的方法及其对人机交互影响的评估。以往的研究强调需要个性化这些解释的详细程度，以提高易用性和理解性。本文提出了一种框架，用于更新和检索用户知识记忆模型，以适应解释的详细程度并在参考之前获得的概念时进行个性化。在基于本文提出框架的三种架构中，这三种架构均使用大规模语言模型（LLMs），并在两个不同的场景中进行了评估：巡逻机器人和厨房助手机器人。实验结果表明，两阶段架构——先生成解释再个性化，仅当用户有相关知识时才能有效减少解释的详细程度。 

---
# Closed-Form Robustness Bounds for Second-Order Pruning of Neural Controller Policies 

**Title (ZH)**: 闭式稳健性界估计：神经控制策略的第二阶剪枝 

**Authors**: Maksym Shamrai  

**Link**: [PDF](https://arxiv.org/pdf/2507.02953)  

**Abstract**: Deep neural policies have unlocked agile flight for quadcopters, adaptive grasping for manipulators, and reliable navigation for ground robots, yet their millions of weights conflict with the tight memory and real-time constraints of embedded microcontrollers. Second-order pruning methods, such as Optimal Brain Damage (OBD) and its variants, including Optimal Brain Surgeon (OBS) and the recent SparseGPT, compress networks in a single pass by leveraging the local Hessian, achieving far higher sparsity than magnitude thresholding. Despite their success in vision and language, the consequences of such weight removal on closed-loop stability, tracking accuracy, and safety have remained unclear. We present the first mathematically rigorous robustness analysis of second-order pruning in nonlinear discrete-time control. The system evolves under a continuous transition map, while the controller is an $L$-layer multilayer perceptron with ReLU-type activations that are globally 1-Lipschitz. Pruning the weight matrix of layer $k$ replaces $W_k$ with $W_k+\delta W_k$, producing the perturbed parameter vector $\widehat{\Theta}=\Theta+\delta\Theta$ and the pruned policy $\pi(\cdot;\widehat{\Theta})$. For every input state $s\in X$ we derive the closed-form inequality $
\|\pi(s;\Theta)-\pi(s;\widehat{\Theta})\|_2 \le C_k(s)\,\|\delta W_k\|_2, $
where the constant $C_k(s)$ depends only on unpruned spectral norms and biases, and can be evaluated in closed form from a single forward pass. The derived bounds specify, prior to field deployment, the maximal admissible pruning magnitude compatible with a prescribed control-error threshold. By linking second-order network compression with closed-loop performance guarantees, our work narrows a crucial gap between modern deep-learning tooling and the robustness demands of safety-critical autonomous systems. 

**Abstract (ZH)**: 第二阶剪枝方法在非线性离散时间控制中的稳健性分析：连接现代深度学习工具与安全关键自主系统的需求 

---
# Beyond One Shot, Beyond One Perspective: Cross-View and Long-Horizon Distillation for Better LiDAR Representations 

**Title (ZH)**: 超越单视角与单次-shot学习：跨视角与长视 horizon 教授以提升激光雷达表示 

**Authors**: Xiang Xu, Lingdong Kong, Song Wang, Chuanwei Zhou, Qingshan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05260)  

**Abstract**: LiDAR representation learning aims to extract rich structural and semantic information from large-scale, readily available datasets, reducing reliance on costly human annotations. However, existing LiDAR representation strategies often overlook the inherent spatiotemporal cues in LiDAR sequences, limiting their effectiveness. In this work, we propose LiMA, a novel long-term image-to-LiDAR Memory Aggregation framework that explicitly captures longer range temporal correlations to enhance LiDAR representation learning. LiMA comprises three key components: 1) a Cross-View Aggregation module that aligns and fuses overlapping regions across neighboring camera views, constructing a more unified and redundancy-free memory bank; 2) a Long-Term Feature Propagation mechanism that efficiently aligns and integrates multi-frame image features, reinforcing temporal coherence during LiDAR representation learning; and 3) a Cross-Sequence Memory Alignment strategy that enforces consistency across driving sequences, improving generalization to unseen environments. LiMA maintains high pretraining efficiency and incurs no additional computational overhead during downstream tasks. Extensive experiments on mainstream LiDAR-based perception benchmarks demonstrate that LiMA significantly improves both LiDAR semantic segmentation and 3D object detection. We hope this work inspires more effective pretraining paradigms for autonomous driving. The code has be made publicly accessible for future research. 

**Abstract (ZH)**: LiMA：一种新颖的长期图像到LiDAR记忆聚合框架，用于增强LiDAR表示学习 

---
# From Marginal to Joint Predictions: Evaluating Scene-Consistent Trajectory Prediction Approaches for Automated Driving 

**Title (ZH)**: 从边缘到联合预测：评估场景一致的轨迹预测方法在自动驾驶中的性能 

**Authors**: Fabian Konstantinidis, Ariel Dallari Guerreiro, Raphael Trumpp, Moritz Sackmann, Ulrich Hofmann, Marco Caccamo, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2507.05254)  

**Abstract**: Accurate motion prediction of surrounding traffic participants is crucial for the safe and efficient operation of automated vehicles in dynamic environments. Marginal prediction models commonly forecast each agent's future trajectories independently, often leading to sub-optimal planning decisions for an automated vehicle. In contrast, joint prediction models explicitly account for the interactions between agents, yielding socially and physically consistent predictions on a scene level. However, existing approaches differ not only in their problem formulation but also in the model architectures and implementation details used, making it difficult to compare them. In this work, we systematically investigate different approaches to joint motion prediction, including post-processing of the marginal predictions, explicitly training the model for joint predictions, and framing the problem as a generative task. We evaluate each approach in terms of prediction accuracy, multi-modality, and inference efficiency, offering a comprehensive analysis of the strengths and limitations of each approach. Several prediction examples are available at this https URL. 

**Abstract (ZH)**: 周围的交通参与者准确运动预测对于动态环境下自动驾驶车辆的安全高效运行至关重要。单个预测模型通常独立预测每个代理的未来轨迹，这往往会导致自动驾驶车辆规划决策的次优结果。相比之下，联合预测模型明确考虑代理之间的交互，从而在场景级别上提供社会上和物理上一致的预测。然而，现有的方法不仅在问题表述上不同，还在所使用的大模型架构和实现细节上有所不同，这使得它们难以进行比较。在本文中，我们系统地研究了不同的联合运动预测方法，包括对单个预测的后处理、明确训练模型进行联合预测以及将问题定义为生成任务。我们从预测准确性、多模态性和推理效率等方面评估每种方法，全面分析每种方法的优点和局限性。一些预测示例可在以下链接查阅：this https URL。 

---
# Critiques of World Models 

**Title (ZH)**: 世界模型的批评 

**Authors**: Eric Xing, Mingkai Deng, Jinyu Hou, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05169)  

**Abstract**: World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model. 

**Abstract (ZH)**: 世界模型，作为生物代理体验和作用于现实世界环境的算法替代品，因其对开发具有人工（通用）智能的虚拟代理的需求增长而成为近年来的一个新兴研究课题。对于世界模型到底是什么，如何构建，如何使用以及如何评估等方面存在诸多争议。本文以脍炙人口的科幻经典《沙丘》中的想象为起点，借鉴心理学文献中的“假设思维”概念，对几种世界建模学派的观点提出了批评，并提出世界模型的主要目标是模拟现实世界中所有可行动的可能性以便进行目的性的推理和行动。在此基础上，我们提出了一种通用目的世界模型的新架构，基于分层次、多级和混合连续/离散表示，并结合生成性和自我监督学习框架，并展望了此类模型可能促成一种物理性的、主体性的和嵌套性的（PAN）通用人工智能系统的未来前景。 

---
# VOTE: Vision-Language-Action Optimization with Trajectory Ensemble Voting 

**Title (ZH)**: VOTE：轨迹集成投票的视觉-语言-动作优化 

**Authors**: Juyi Lin, Amir Taherin, Arash Akbari, Arman Akbari, Lei Lu, Guangyu Chen, Taskin Padir, Xiaomeng Yang, Weiwei Chen, Yiqian Li, Xue Lin, David Kaeli, Pu Zhao, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05116)  

**Abstract**: Recent large-scale Vision Language Action (VLA) models have shown superior performance in robotic manipulation tasks guided by natural language. However, their generalization remains limited when applied to novel objects or unfamiliar environments that lie outside the training distribution. To address this, many existing approaches integrate additional components such as depth estimation, segmentation, or even diffusion to improve generalization, at the cost of adding significant computation overhead, resulting in low efficiency. This motivates the exploration of efficient action prediction methods, which are independent of additional high-level visual representations or diffusion techniques. In this work, we propose VOTE, an efficient and general framework for the optimization and acceleration of VLA models. In details, we propose a novel tokenizer-free fine-tuning approach for parallel accurate action prediction, which reduces computational overhead and accelerates inference speed. Additionally, we adopt an ensemble voting strategy for the action sampling, which significantly improves model performance and enhances generalization. Experimental results show that our method achieves state-of-the-art performance with 35$\times$ faster inference and 145 Hz throughput. All the details and codes will be open-sourced. 

**Abstract (ZH)**: 近期的大规模视觉语言动作（VLA）模型在由自然语言指导的机器人 manipulation 任务中表现出色。然而，当应用于训练分布之外的新奇对象或不熟悉的环境时，其泛化能力仍然有限。为了解决这一问题，许多现有方法通过集成深度估计、分割，甚至扩散等额外组件来提高泛化能力，但会增加显著的计算开销，导致效率低下。这促使探索与额外高级视觉表示或扩散技术无关的高效动作预测方法。在本文中，我们提出了VOTE，一种用于提升视觉语言动作模型优化和加速的一般框架。该方法提出了一种新颖的无分词器并行准确动作预测的微调方法，减少了计算开销并加快了推理速度。此外，我们采用了集合投票策略进行动作采样，显著提升了模型性能并增强了泛化能力。实验结果表明，我们的方法在推理速度提高35倍且吞吐量达到145 Hz的情况下达到了最先进的性能，并且所有细节和代码将开源。 

---
# From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems 

**Title (ZH)**: 从自主到能动：为以人为中心的移动系统设计能动车辆 

**Authors**: Jiangbo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04996)  

**Abstract**: Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are defined as systems capable of perceiving their environment and executing preprogrammed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 1 to 6), such as interaction with humans and machines, goal adaptation, contextual reasoning, external tool use, and long-term planning, particularly with the integration of large language models (LLMs) and agentic AI systems. These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this, we introduce the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and interact within complex environments. This paper presents a systems-level framework to characterize AgVs, focusing on their cognitive and communicative layers and differentiating them from conventional AuVs. It synthesizes relevant advances in agentic AI, robotics, multi-agent systems, and human-machine interaction, and highlights how agentic AI, through high-level reasoning and tool use, can function not merely as computational tools but as interactive agents embedded in mobility ecosystems. The paper concludes by identifying key challenges in the development and governance of AgVs, including safety, real-time control, public acceptance, ethical alignment, and regulatory frameworks. 

**Abstract (ZH)**: 自主性，源自希腊语的autos（自我）和nomos（法则），指的是根据内部规则自主运行而不受外部控制的能力。相应地，自主车辆（AuVs）被定义为能够独立于外部输入感知其环境并执行预编程任务的系统。然而，研究和实际部署越来越多地展示了表现出超出这一定义的行为的车辆（包括SAE等级1至6的车辆），如与人类和机器的交互、目标适应、情境推理、外部工具使用以及长期规划，特别是在大型语言模型（LLMs）和代理型人工智能系统的集成中。这些发展揭示了技术自主性与未来以人为中心的移动系统所需更广泛的认知和社会能力之间的概念差距。为应对这一差距，我们引入了代理型车辆（AgVs）的概念，指的是整合代理型人工智能以在复杂环境中进行推理、适应和交互的车辆。本文提出了一种系统级框架来界定AgVs，重点关注其认知和通信层，并将其与传统的AuVs区分开来。该文综合了代理型人工智能、机器人技术、多agent系统和人机交互领域的相关进展，并指出代理型人工智能如何通过高层次推理和工具使用，不仅作为计算工具，而且作为嵌入在移动生态系统中的交互代理发挥作用。论文最后指出了代理型车辆开发和治理中的关键挑战，包括安全性、实时控制、公众接受度、伦理对齐和监管框架。 

---
# Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts 

**Title (ZH)**: 在野外环境下学习具有选择性混合专家的鲁棒立体匹配 

**Authors**: Yun Wang, Longguang Wang, Chenghao Zhang, Yongjian Zhang, Zhanjie Zhang, Ao Ma, Chenyou Fan, Tin Lun Lam, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04631)  

**Abstract**: Recently, learning-based stereo matching networks have advanced significantly. However, they often lack robustness and struggle to achieve impressive cross-domain performance due to domain shifts and imbalanced disparity distributions among diverse datasets. Leveraging Vision Foundation Models (VFMs) can intuitively enhance the model's robustness, but integrating such a model into stereo matching cost-effectively to fully realize their robustness remains a key challenge. To address this, we propose SMoEStereo, a novel framework that adapts VFMs for stereo matching through a tailored, scene-specific fusion of Low-Rank Adaptation (LoRA) and Mixture-of-Experts (MoE) modules. SMoEStereo introduces MoE-LoRA with adaptive ranks and MoE-Adapter with adaptive kernel sizes. The former dynamically selects optimal experts within MoE to adapt varying scenes across domains, while the latter injects inductive bias into frozen VFMs to improve geometric feature extraction. Importantly, to mitigate computational overhead, we further propose a lightweight decision network that selectively activates MoE modules based on input complexity, balancing efficiency with accuracy. Extensive experiments demonstrate that our method exhibits state-of-the-art cross-domain and joint generalization across multiple benchmarks without dataset-specific adaptation. The code is available at \textcolor{red}{this https URL}. 

**Abstract (ZH)**: 基于视觉基础模型的新型立体匹配框架SMoEStereo 

---
# Accelerated Online Reinforcement Learning using Auxiliary Start State Distributions 

**Title (ZH)**: 使用辅助起始状态分布加速在线强化学习 

**Authors**: Aman Mehra, Alexandre Capone, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2507.04606)  

**Abstract**: A long-standing problem in online reinforcement learning (RL) is of ensuring sample efficiency, which stems from an inability to explore environments efficiently. Most attempts at efficient exploration tackle this problem in a setting where learning begins from scratch, without prior information available to bootstrap learning. However, such approaches fail to leverage expert demonstrations and simulators that can reset to arbitrary states. These affordances are valuable resources that offer enormous potential to guide exploration and speed up learning. In this paper, we explore how a small number of expert demonstrations and a simulator allowing arbitrary resets can accelerate learning during online RL. We find that training with a suitable choice of an auxiliary start state distribution that may differ from the true start state distribution of the underlying Markov Decision Process can significantly improve sample efficiency. We find that using a notion of safety to inform the choice of this auxiliary distribution significantly accelerates learning. By using episode length information as a way to operationalize this notion, we demonstrate state-of-the-art sample efficiency on a sparse-reward hard-exploration environment. 

**Abstract (ZH)**: 在线强化学习中长期存在的问题是确保样本效率，这源于无法有效探索环境。大多数有效探索的尝试都是在没有任何先验信息的情况下从头开始学习，无法利用可以重置到任意状态的专家演示和模拟器。这些资源可以显著提高探索效率并加速学习。在本文中，我们研究了如何利用少量专家演示和允许任意重置的模拟器来加速在线强化学习中的学习过程。我们发现，使用与底层马尔科夫决策过程的真实起始状态分布可能不同的适当辅助起始状态分布进行训练，可以显著提高样本效率。我们发现，使用安全性的概念来指导这种辅助分布的选择可以显著加速学习。通过使用回合长度信息来实现这一概念，我们在稀疏奖励和难探索环境中实现了最先进的样本效率。 

---
# Grounded Gesture Generation: Language, Motion, and Space 

**Title (ZH)**: 基于语境的手势生成：语言、运动与空间 

**Authors**: Anna Deichler, Jim O'Regan, Teo Guichoux, David Johansson, Jonas Beskow  

**Link**: [PDF](https://arxiv.org/pdf/2507.04522)  

**Abstract**: Human motion generation has advanced rapidly in recent years, yet the critical problem of creating spatially grounded, context-aware gestures has been largely overlooked. Existing models typically specialize either in descriptive motion generation, such as locomotion and object interaction, or in isolated co-speech gesture synthesis aligned with utterance semantics. However, both lines of work often treat motion and environmental grounding separately, limiting advances toward embodied, communicative agents. To address this gap, our work introduces a multimodal dataset and framework for grounded gesture generation, combining two key resources: (1) a synthetic dataset of spatially grounded referential gestures, and (2) MM-Conv, a VR-based dataset capturing two-party dialogues. Together, they provide over 7.7 hours of synchronized motion, speech, and 3D scene information, standardized in the HumanML3D format. Our framework further connects to a physics-based simulator, enabling synthetic data generation and situated evaluation. By bridging gesture modeling and spatial grounding, our contribution establishes a foundation for advancing research in situated gesture generation and grounded multimodal interaction.
Project page: this https URL 

**Abstract (ZH)**: 近年来，人类运动生成取得了 rapid 进展，但创建空间上一致且情境相关的手势这一关键问题仍被很大程度上忽视。现有模型通常要么专注于描述性运动生成，如移动和物体交互，要么专注于与陈述语义相匹配的孤立共时手势合成。然而，这两方面的研究往往将运动和环境标注分开，限制了对具身、沟通性代理的研究进展。为了解决这一差距，我们的工作引入了一个多模态数据集和框架，用于生成基于情境的手势，结合了两种关键资源：(1) 空间上一致的参考手势合成数据集，(2) MM-Conv，一个基于虚拟现实的双人对话数据集。两者共同提供了超过 7.7 小时的同步运动、语音和三维场景信息，统一格式为 HumanML3D 格式。我们的框架进一步连接到一个基于物理的模拟器，使其能够生成合成数据并在特定场景中进行评估。通过将手势建模与空间标注相结合，我们的贡献为推进基于情境的手势生成和基于情境的多模态交互研究奠定了基础。
Project page: this https URL 

---
# U-ViLAR: Uncertainty-Aware Visual Localization for Autonomous Driving via Differentiable Association and Registration 

**Title (ZH)**: U-ViLAR：基于可微关联和配准的不确定性感知视觉定位方法在自主驾驶中的应用 

**Authors**: Xiaofan Li, Zhihao Xu, Chenming Wu, Zhao Yang, Yumeng Zhang, Jiang-Jiang Liu, Haibao Yu, Fan Duan, Xiaoqing Ye, Yuan Wang, Shirui Li, Xun Sun, Ji Wan, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04503)  

**Abstract**: Accurate localization using visual information is a critical yet challenging task, especially in urban environments where nearby buildings and construction sites significantly degrade GNSS (Global Navigation Satellite System) signal quality. This issue underscores the importance of visual localization techniques in scenarios where GNSS signals are unreliable. This paper proposes U-ViLAR, a novel uncertainty-aware visual localization framework designed to address these challenges while enabling adaptive localization using high-definition (HD) maps or navigation maps. Specifically, our method first extracts features from the input visual data and maps them into Bird's-Eye-View (BEV) space to enhance spatial consistency with the map input. Subsequently, we introduce: a) Perceptual Uncertainty-guided Association, which mitigates errors caused by perception uncertainty, and b) Localization Uncertainty-guided Registration, which reduces errors introduced by localization uncertainty. By effectively balancing the coarse-grained large-scale localization capability of association with the fine-grained precise localization capability of registration, our approach achieves robust and accurate localization. Experimental results demonstrate that our method achieves state-of-the-art performance across multiple localization tasks. Furthermore, our model has undergone rigorous testing on large-scale autonomous driving fleets and has demonstrated stable performance in various challenging urban scenarios. 

**Abstract (ZH)**: 使用视觉信息实现准确定位是一项关键但具有挑战性的任务，特别是在城市环境中，附近的建筑物和施工站点显著降低了GNSS（全球导航卫星系统）信号质量。这一问题凸显了在GNSS信号不可靠场景下视觉定位技术的重要性。本文提出了一种新颖的不确定性感知视觉定位框架U-ViLAR，旨在解决这些挑战并利用高分辨率（HD）地图或导航地图实现自适应定位。具体而言，我们的方法首先从输入的视觉数据中提取特征，并将其映射到鸟瞰视图（BEV）空间中，以增强与地图输入的空间一致性。随后，我们引入了：a) 感知不确定性引导关联，以减轻感知不确定性引起的问题；b) 定位不确定性引导注册，以减少由定位不确定性引入的错误。通过有效地平衡关联的大尺度粗略定位能力和注册的高精度细粒度定位能力，我们的方法实现了稳健且准确的定位。实验结果表明，我们的方法在多个定位任务中达到了最先进的性能。此外，我们的模型已经在大规模自动驾驶车队上进行了严格的测试，并在各种具有挑战性的城市场景中展示了稳定的性能。 

---
# Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference 

**Title (ZH)**: 千脑系统：感觉运动智能实现快速稳健的学习与推理 

**Authors**: Niels Leadholm, Viviane Clay, Scott Knudstrup, Hojae Lee, Jeff Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2507.04494)  

**Abstract**: Current AI systems achieve impressive performance on many tasks, yet they lack core attributes of biological intelligence, including rapid, continual learning, representations grounded in sensorimotor interactions, and structured knowledge that enables efficient generalization. Neuroscience theory suggests that mammals evolved flexible intelligence through the replication of a semi-independent, sensorimotor module, a functional unit known as a cortical column. To address the disparity between biological and artificial intelligence, thousand-brains systems were proposed as a means of mirroring the architecture of cortical columns and their interactions.
In the current work, we evaluate the unique properties of Monty, the first implementation of a thousand-brains system. We focus on 3D object perception, and in particular, the combined task of object recognition and pose estimation. Utilizing the YCB dataset of household objects, we first assess Monty's use of sensorimotor learning to build structured representations, finding that these enable robust generalization. These representations include an emphasis on classifying objects by their global shape, as well as a natural ability to detect object symmetries. We then explore Monty's use of model-free and model-based policies to enable rapid inference by supporting principled movements. We find that such policies complement Monty's modular architecture, a design that can accommodate communication between modules to further accelerate inference speed via a novel `voting' algorithm. Finally, we examine Monty's use of associative, Hebbian-like binding to enable rapid, continual, and computationally efficient learning, properties that compare favorably to current deep learning architectures. While Monty is still in a nascent stage of development, these findings support thousand-brains systems as a powerful and promising new approach to AI. 

**Abstract (ZH)**: 当前的AI系统在许多任务上取得了令人印象深刻的性能，但缺乏生物学智能的核心属性，包括快速连续学习、根植于传感器互动的表示以及促进高效泛化的结构性知识。神经科学理论表明，哺乳动物通过复制半独立的、传感器模态模块进化出了灵活的智能，这种功能单元被称为皮层柱。为了解决生物学与人工智能之间的差距，提出了千脑系统作为一种模仿皮层柱及其交互的体系结构的方法。

在当前的研究中，我们评估了Monty的第一个千脑系统实现的独特属性。我们集中在3D物体感知，特别是物体识别和姿态估计的联合任务上。利用家庭用品的YCB数据集，我们首先评估了Monty使用传感器模态学习构建结构性表示的能力，发现这些表示使泛化更加稳健。这些表示强调通过全局形状对物体进行分类，还具有自然检测物体对称性的能力。随后我们探索了Monty使用无模型和基于模型的策略以支持原则性动作实现快速推理的过程。我们发现这些策略补充了Monty的模块化架构，这种设计可以通过一种新颖的“投票”算法促进模块间通信，进一步加速推理速度。最后，我们研究了Monty使用类似于海伯式的联想绑定以实现快速、持续和计算高效的learning的能力，这些特性与当前深度学习架构相比具有优势。虽然Monty仍处于发展的早期阶段，但这些发现支持千脑系统作为一种强大且有前景的新AI方法。 

---
# Agentic Distributed Computing 

**Title (ZH)**: 代理式分布式计算 

**Authors**: Ajay D. Kshemkalyani, Manish Kumar, Anisur Rahaman Molla, Gokarna Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.04459)  

**Abstract**: The most celebrated and extensively studied model of distributed computing is the {\em message-passing model,} in which each vertex/node of the (distributed network) graph corresponds to a static computational device that communicates with other devices through passing messages. In this paper, we consider the {\em agentic model} of distributed computing which extends the message-passing model in a new direction. In the agentic model, computational devices are modeled as relocatable or mobile computational devices (called agents in this paper), i.e., each vertex/node of the graph serves as a container for the devices, and hence communicating with another device requires relocating to the same node. We study two fundamental graph level tasks, leader election, and minimum spanning tree, in the agentic model, which will enhance our understanding of distributed computation across paradigms. The objective is to minimize both time and memory complexities. Following the literature, we consider the synchronous setting in which each agent performs its operations synchronously with others, and hence the time complexity can be measured in rounds. In this paper, we present two deterministic algorithms for leader election: one for the case of $k<n$ and another for the case of $k=n$, minimizing both time and memory complexities, where $k$ and $n$, respectively, are the number of agents and number of nodes of the graph. Using these leader election results, we develop deterministic algorithms for agents to construct a minimum spanning tree of the graph, minimizing both time and memory complexities. To the best of our knowledge, this is the first study of distributed graph level tasks in the agentic model with $k\leq n$. Previous studies only considered the case of $k=n$. 

**Abstract (ZH)**: 分布式计算中最为著名且被广泛研究的模型是消息传递模型，在该模型中，分布式网络图的每个顶点/节点对应一个静态计算设备，这些设备通过传递消息进行通信。本文考虑的是代理模型，该模型扩展了消息传递模型的一个新方向。在代理模型中，计算设备被建模为可重新定位的或移动的计算设备（在本文中称为代理），即图的每个顶点/节点是这些设备的容器，因此与其他设备通信需要移动到同一节点。我们研究了代理模型下的两个基本图级任务：领导者选举和最小生成树，这将增强我们对分布式计算在不同范式中的理解。目标是同时最小化时间和内存复杂度。根据文献，我们将考察同步设置，在这种设置中，每个代理与其他代理同步执行其操作，因此时间复杂度可以用轮次来度量。本文我们提出了两种确定性领导选举算法：一种适用于\(k<n\)的情况，另一种适用于\(k=n\)的情况，其中\(k\)和\(n\)分别表示代理的数量和图的节点数量，同时最小化时间和内存复杂度。利用这些领导选举的结果，我们开发了两种确定性算法，使代理能够构建图的最小生成树，同时最小化时间和内存复杂度。据我们所知，这是首次在代理模型下（\(k \leq n\))研究分布式图级任务的文献。以往的研究仅考虑了\(k=n\)的情况。 

---
# DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge 

**Title (ZH)**: DreamVLA：融入全面世界知识的视觉-语言-行动模型 

**Authors**: Wenyao Zhang, Hongsi Liu, Zekun Qi, Yunnan Wang, XinQiang Yu, Jiazhao Zhang, Runpei Dong, Jiawei He, He Wang, Zhizheng Zhang, Li Yi, Wenjun Zeng, Xin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04447)  

**Abstract**: Recent advances in vision-language-action (VLA) models have shown promise in integrating image generation with action prediction to improve generalization and reasoning in robot manipulation. However, existing methods are limited to challenging image-based forecasting, which suffers from redundant information and lacks comprehensive and critical world knowledge, including dynamic, spatial and semantic information. To address these limitations, we propose DreamVLA, a novel VLA framework that integrates comprehensive world knowledge forecasting to enable inverse dynamics modeling, thereby establishing a perception-prediction-action loop for manipulation tasks. Specifically, DreamVLA introduces a dynamic-region-guided world knowledge prediction, integrated with the spatial and semantic cues, which provide compact yet comprehensive representations for action planning. This design aligns with how humans interact with the world by first forming abstract multimodal reasoning chains before acting. To mitigate interference among the dynamic, spatial and semantic information during training, we adopt a block-wise structured attention mechanism that masks their mutual attention, preventing information leakage and keeping each representation clean and disentangled. Moreover, to model the conditional distribution over future actions, we employ a diffusion-based transformer that disentangles action representations from shared latent features. Extensive experiments on both real-world and simulation environments demonstrate that DreamVLA achieves 76.7% success rate on real robot tasks and 4.44 average length on the CALVIN ABC-D benchmarks. 

**Abstract (ZH)**: 近期在视觉-语言-动作（VLA）模型方面的进展展示了将图像生成与动作预测相结合以提高机器人操作中泛化能力和推理的潜力。然而，现有方法受限于基于图像的预测挑战，这些预测包含冗余信息且缺乏综合性和关键的世界知识，包括动态、空间和语义信息。为了解决这些限制，我们提出了DreamVLA，一种新颖的VLA框架，能够整合全面的世界知识预测以实现逆动力学建模，从而建立感知-预测-动作循环以完成操作任务。具体来说，DreamVLA 引入了动态区域引导的世界知识预测，结合了空间和语义线索，为动作规划提供了紧凑且综合的表示。此设计符合人类在行动前先形成抽象多模态推理链的方式。为了在训练过程中减轻动态、空间和语义信息之间的相互干扰，我们采用了块结构化的注意力机制，屏蔽它们之间的相互注意，防止信息泄露并保持每个表示的清晰和独立。此外，为了建模未来动作的条件分布，我们采用了基于扩散的变换器来独立动作表示与共享的潜在特征。在现实世界和模拟环境中的广泛实验表明，DreamVLA 在真实机器人任务中的成功率达到了76.7%，并在CALVIN ABC-D基准测试中实现了4.44的平均长度。 

---
# Mission-Aligned Learning-Informed Control of Autonomous Systems: Formulation and Foundations 

**Title (ZH)**: 自主系统的目标导向学习导向控制：建模与基础理论 

**Authors**: Vyacheslav Kungurtsev, Gustav Sir, Akhil Anand, Sebastien Gros, Haozhe Tian, Homayoun Hamedmoghadam  

**Link**: [PDF](https://arxiv.org/pdf/2507.04356)  

**Abstract**: Research, innovation and practical capital investment have been increasing rapidly toward the realization of autonomous physical agents. This includes industrial and service robots, unmanned aerial vehicles, embedded control devices, and a number of other realizations of cybernetic/mechatronic implementations of intelligent autonomous devices. In this paper, we consider a stylized version of robotic care, which would normally involve a two-level Reinforcement Learning procedure that trains a policy for both lower level physical movement decisions as well as higher level conceptual tasks and their sub-components. In order to deliver greater safety and reliability in the system, we present the general formulation of this as a two-level optimization scheme which incorporates control at the lower level, and classical planning at the higher level, integrated with a capacity for learning. This synergistic integration of multiple methodologies -- control, classical planning, and RL -- presents an opportunity for greater insight for algorithm development, leading to more efficient and reliable performance. Here, the notion of reliability pertains to physical safety and interpretability into an otherwise black box operation of autonomous agents, concerning users and regulators. This work presents the necessary background and general formulation of the optimization framework, detailing each component and its integration with the others. 

**Abstract (ZH)**: 研究、创新和实际资本投资正迅速增加，以实现自主物理代理。这包括工业和服务业机器人、无人驾驶航空 vehicles、嵌入式控制设备，以及其他智能自主设备的 cybernetic/mechatronic 实现。本文考虑了一种简化版的机器人护理，通常涉及一个多层强化学习程序，用于训练低层物理动作决策和高层概念性任务及其子组件的策略。为了提高系统的安全性和可靠性，我们将其一般形式表述为一个多层优化方案，该方案结合了低层控制和高层经典规划，具有学习能力。这种控制、经典规划和强化学习多种方法的协同集成为算法开发提供了更多见解，从而实现更高效和可靠的表现。这里的可靠性涉及物理安全和对自主代理黑盒操作的可解释性，以满足用户和监管机构的需求。本研究阐述了优化框架的必要背景和一般形式，详细说明了每个组件及其与其他组件的集成。 

---
# Pedestrian Intention Prediction via Vision-Language Foundation Models 

**Title (ZH)**: 基于视觉-语言基础模型的行人意图预测 

**Authors**: Mohsen Azarmi, Mahdi Rezaei, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04141)  

**Abstract**: Prediction of pedestrian crossing intention is a critical function in autonomous vehicles. Conventional vision-based methods of crossing intention prediction often struggle with generalizability, context understanding, and causal reasoning. This study explores the potential of vision-language foundation models (VLFMs) for predicting pedestrian crossing intentions by integrating multimodal data through hierarchical prompt templates. The methodology incorporates contextual information, including visual frames, physical cues observations, and ego-vehicle dynamics, into systematically refined prompts to guide VLFMs effectively in intention prediction. Experiments were conducted on three common datasets-JAAD, PIE, and FU-PIP. Results demonstrate that incorporating vehicle speed, its variations over time, and time-conscious prompts significantly enhances the prediction accuracy up to 19.8%. Additionally, optimised prompts generated via an automatic prompt engineering framework yielded 12.5% further accuracy gains. These findings highlight the superior performance of VLFMs compared to conventional vision-based models, offering enhanced generalisation and contextual understanding for autonomous driving applications. 

**Abstract (ZH)**: 基于视觉-语言基础模型的行人过街意图预测研究 

---
# Driver-Net: Multi-Camera Fusion for Assessing Driver Take-Over Readiness in Automated Vehicles 

**Title (ZH)**: Driver-Net：多摄像头融合评估自动驾驶车辆驾驶员接管准备状态 

**Authors**: Mahdi Rezaei, Mohsen Azarmi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04139)  

**Abstract**: Ensuring safe transition of control in automated vehicles requires an accurate and timely assessment of driver readiness. This paper introduces Driver-Net, a novel deep learning framework that fuses multi-camera inputs to estimate driver take-over readiness. Unlike conventional vision-based driver monitoring systems that focus on head pose or eye gaze, Driver-Net captures synchronised visual cues from the driver's head, hands, and body posture through a triple-camera setup. The model integrates spatio-temporal data using a dual-path architecture, comprising a Context Block and a Feature Block, followed by a cross-modal fusion strategy to enhance prediction accuracy. Evaluated on a diverse dataset collected from the University of Leeds Driving Simulator, the proposed method achieves an accuracy of up to 95.8% in driver readiness classification. This performance significantly enhances existing approaches and highlights the importance of multimodal and multi-view fusion. As a real-time, non-intrusive solution, Driver-Net contributes meaningfully to the development of safer and more reliable automated vehicles and aligns with new regulatory mandates and upcoming safety standards. 

**Abstract (ZH)**: 确保自动驾驶车辆控制交接安全需要准确及时地评估驾驶员的就绪状态。本文提出了一种新颖的深度学习框架Driver-Net，该框架融合多摄像头输入以估算驾驶员接管就绪性。与传统的基于视觉的驾驶员监测系统主要关注头部姿态或视线不同，Driver-Net 通过三摄像头设置捕捉驾驶员头部、手部和身体姿态的同步视觉线索。该模型使用双路径架构，包括上下文块和特征块，随后采用跨模态融合策略以提高预测准确性。在利兹大学驾驶模拟器收集的多源数据集上进行评估，所提出的方法在驾驶员就绪性分类中的准确率达到95.8%。该性能显著增强了现有方法，并强调了多模态和多视角融合的重要性。作为实时且非侵入性的解决方案，Driver-Net 对更安全和更可靠的自动驾驶车辆的发展做出了贡献，并与新的监管要求和即将出台的安全标准相一致。 

---
# Human-centered AI with focus on Human-robot interaction (Book chapter) 

**Title (ZH)**: 以人为本的AI：以人机交互为重点（书籍章节） 

**Authors**: Alireza Mortezapour, Giuliana Vitiello  

**Link**: [PDF](https://arxiv.org/pdf/2507.04095)  

**Abstract**: Modern social robots can be considered the descendants of steam engines from the First Industrial Revolution (IR 1.0) and industrial robotic arms from the Third Industrial Revolution (IR 3.0). As some time has passed since the introduction of these robots during the Fourth Industrial Revolution (IR 4.0), challenges and issues in their interaction with humans have emerged, leading researchers to conclude that, like any other AI-based technology, these robots must also be human-centered to meet the needs of their users. This chapter aims to introduce humans and their needs in interactions with robots, ranging from short-term, one-on-one interactions (micro-level) to long-term, macro-level needs at the societal scale. Building upon the principles of human-centered AI, this chapter presents, for the first time, a new framework of human needs called the Dual Pyramid. This framework encompasses a comprehensive list of human needs in robot interactions, from the most fundamental, robot effectiveness to macro level requirements, such as the collaboration with robots in achieving the United Nations 17 Sustainable Development Goals. 

**Abstract (ZH)**: 现代社交机器人可以被视为第一工业革命（IR 1.0）蒸汽发动机和第三工业革命（IR 3.0）工业机器人手臂的后裔。随着第四工业革命（IR 4.0）中这些机器人的引入，人机交互中的挑战和问题逐渐显现，促使研究者得出结论，这些机器人也必须以用户为中心，以满足用户的需求。本章旨在介绍人在与机器人互动中的需求，从短期一对一互动（微观层面）到长期在社会层面的需求。基于以人为本的AI原理，本章首次提出一个新的需求框架，称为双金字塔框架。该框架涵盖了机器人互动中从最基本的有效性需求到宏观层面的要求，如与机器人合作以实现联合国17项可持续发展目标。 

---
# Breaking Imitation Bottlenecks: Reinforced Diffusion Powers Diverse Trajectory Generation 

**Title (ZH)**: 打破imitation瓶颈：强化扩散赋能多样化轨迹生成 

**Authors**: Ziying Song, Lin Liu, Hongyu Pan, Bencheng Liao, Mingzhe Guo, Lei Yang, Yongchang Zhang, Shaoqing Xu, Caiyan Jia, Yadan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.04049)  

**Abstract**: Most end-to-end autonomous driving methods rely on imitation learning from single expert demonstrations, often leading to conservative and homogeneous behaviors that limit generalization in complex real-world scenarios. In this work, we propose DIVER, an end-to-end driving framework that integrates reinforcement learning with diffusion-based generation to produce diverse and feasible trajectories. At the core of DIVER lies a reinforced diffusion-based generation mechanism. First, the model conditions on map elements and surrounding agents to generate multiple reference trajectories from a single ground-truth trajectory, alleviating the limitations of imitation learning that arise from relying solely on single expert demonstrations. Second, reinforcement learning is employed to guide the diffusion process, where reward-based supervision enforces safety and diversity constraints on the generated trajectories, thereby enhancing their practicality and generalization capability. Furthermore, to address the limitations of L2-based open-loop metrics in capturing trajectory diversity, we propose a novel Diversity metric to evaluate the diversity of multi-mode this http URL experiments on the closed-loop NAVSIM and Bench2Drive benchmarks, as well as the open-loop nuScenes dataset, demonstrate that DIVER significantly improves trajectory diversity, effectively addressing the mode collapse problem inherent in imitation learning. 

**Abstract (ZH)**: 一种结合强化学习与扩散生成的端到端驾驶框架：多样性和可行性并重的轨迹生成 

---
# NRSeg: Noise-Resilient Learning for BEV Semantic Segmentation via Driving World Models 

**Title (ZH)**: NRSeg: 基于驾驶世界模型的噪声鲁棒学习BEV语义分割 

**Authors**: Siyu Li, Fei Teng, Yihong Cao, Kailun Yang, Zhiyong Li, Yaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04002)  

**Abstract**: Birds' Eye View (BEV) semantic segmentation is an indispensable perception task in end-to-end autonomous driving systems. Unsupervised and semi-supervised learning for BEV tasks, as pivotal for real-world applications, underperform due to the homogeneous distribution of the labeled data. In this work, we explore the potential of synthetic data from driving world models to enhance the diversity of labeled data for robustifying BEV segmentation. Yet, our preliminary findings reveal that generation noise in synthetic data compromises efficient BEV model learning. To fully harness the potential of synthetic data from world models, this paper proposes NRSeg, a noise-resilient learning framework for BEV semantic segmentation. Specifically, a Perspective-Geometry Consistency Metric (PGCM) is proposed to quantitatively evaluate the guidance capability of generated data for model learning. This metric originates from the alignment measure between the perspective road mask of generated data and the mask projected from the BEV labels. Moreover, a Bi-Distribution Parallel Prediction (BiDPP) is designed to enhance the inherent robustness of the model, where the learning process is constrained through parallel prediction of multinomial and Dirichlet distributions. The former efficiently predicts semantic probabilities, whereas the latter adopts evidential deep learning to realize uncertainty quantification. Furthermore, a Hierarchical Local Semantic Exclusion (HLSE) module is designed to address the non-mutual exclusivity inherent in BEV semantic segmentation tasks. Experimental results demonstrate that NRSeg achieves state-of-the-art performance, yielding the highest improvements in mIoU of 13.8% and 11.4% in unsupervised and semi-supervised BEV segmentation tasks, respectively. The source code will be made publicly available at this https URL. 

**Abstract (ZH)**: 基于合成数据的抗噪BEV语义分割学习框架NRSeg 

---
# SAFERad: A Framework to Enable Radar Data for Safety-Relevant Perception Tasks 

**Title (ZH)**: SAFERad：一种使雷达数据可用于安全相关感知任务的框架 

**Authors**: Tim Brühl, Jenny Glönkler, Robin Schwager, Tin Stribor Sohn, Tim Dieter Eberhardt, Sören Hohmann  

**Link**: [PDF](https://arxiv.org/pdf/2507.03959)  

**Abstract**: Radar sensors play a crucial role for perception systems in automated driving but suffer from a high level of noise. In the past, this could be solved by strict filters, which remove most false positives at the expense of undetected objects. Future highly automated functions are much more demanding with respect to error rate. Hence, if the radar sensor serves as a component of perception systems for such functions, a simple filter strategy cannot be applied. In this paper, we present a modified filtering approach which is characterized by the idea to vary the filtering depending on the potential of harmful collision with the object which is potentially represented by the radar point. We propose an algorithm which determines a criticality score for each point based on the planned or presumable trajectory of the automated vehicle. Points identified as very critical can trigger manifold actions to confirm or deny object presence. Our pipeline introduces criticality regions. The filter threshold in these criticality regions is omitted. Commonly known radar data sets do not or barely feature critical scenes. Thus, we present an approach to evaluate our framework by adapting the planned trajectory towards vulnerable road users, which serve as ground truth critical points. Evaluation of the criticality metric prove high recall rates. Besides, our post-processing algorithm lowers the rate of non-clustered critical points by 74.8 % in an exemplary setup compared to a moderate, generic filter. 

**Abstract (ZH)**: 雷达传感器在自动驾驶感知系统中发挥着关键作用，但面对高水平的噪声问题。过去，这一问题通过严格的滤波器解决，虽然能去除大部分假阳性结果，但也可能导致真实目标的漏检。未来高度自动化的功能对错误率的要求更高。因此，如果雷达传感器作为这些功能的感知系统组件之一，简单的滤波策略不再适用。本文提出了一种改进的滤波方法，该方法根据潜在碰撞的危害性程度动态调整滤波策略。我们提出一种算法，根据计划或预想的自动驾驶车辆轨迹为每个点确定一个关键性评分。被识别为非常关键的点可以触发多种行动来确认或否定目标存在。我们的处理pipeline引入了关键性区域，在这些区域中省略了滤波阈值。常见的雷达数据集很少或几乎不包含关键场景。因此，本文提出了一种方法，通过将计划轨迹朝向脆弱的道路使用者调整，这些使用者作为真实的关键点，来评估我们的框架。关键性度量的评估结果表明高召回率。此外，在一个示例设置中，与中等通用滤波器相比，我们的后处理算法将孤立的关键点比率降低了74.8%。 

---
# VISC: mmWave Radar Scene Flow Estimation using Pervasive Visual-Inertial Supervision 

**Title (ZH)**: VISC：使用弥漫性视觉-惯性监督的毫米波雷达场景流估计 

**Authors**: Kezhong Liu, Yiwen Zhou, Mozi Chen, Jianhua He, Jingao Xu, Zheng Yang, Chris Xiaoxuan Lu, Shengkai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03938)  

**Abstract**: This work proposes a mmWave radar's scene flow estimation framework supervised by data from a widespread visual-inertial (VI) sensor suite, allowing crowdsourced training data from smart vehicles. Current scene flow estimation methods for mmWave radar are typically supervised by dense point clouds from 3D LiDARs, which are expensive and not widely available in smart vehicles. While VI data are more accessible, visual images alone cannot capture the 3D motions of moving objects, making it difficult to supervise their scene flow. Moreover, the temporal drift of VI rigid transformation also degenerates the scene flow estimation of static points. To address these challenges, we propose a drift-free rigid transformation estimator that fuses kinematic model-based ego-motions with neural network-learned results. It provides strong supervision signals to radar-based rigid transformation and infers the scene flow of static points. Then, we develop an optical-mmWave supervision extraction module that extracts the supervision signals of radar rigid transformation and scene flow. It strengthens the supervision by learning the scene flow of dynamic points with the joint constraints of optical and mmWave radar measurements. Extensive experiments demonstrate that, in smoke-filled environments, our method even outperforms state-of-the-art (SOTA) approaches using costly LiDARs. 

**Abstract (ZH)**: 基于广泛视觉-惯性传感器套件监督的毫米波雷达场景流估算框架 

---
# Robust Node Localization for Rough and Extreme Deployment Environments 

**Title (ZH)**: 鲁棒节点定位技术在恶劣和极端部署环境中的应用 

**Authors**: Abiy Tasissa, Waltenegus Dargie  

**Link**: [PDF](https://arxiv.org/pdf/2507.03856)  

**Abstract**: Many applications have been identified which require the deployment of large-scale low-power wireless sensor networks. Some of the deployment environments, however, impose harsh operation conditions due to intense cross-technology interference, extreme weather conditions (heavy rainfall, excessive heat, etc.), or rough motion, thereby affecting the quality and predictability of the wireless links the nodes establish. In localization tasks, these conditions often lead to significant errors in estimating the position of target nodes. Motivated by the practical deployments of sensors on the surface of different water bodies, we address the problem of identifying susceptible nodes and robustly estimating their positions. We formulate these tasks as a compressive sensing problem and propose algorithms for both node identification and robust estimation. Additionally, we design an optimal anchor configuration to maximize the robustness of the position estimation task. Our numerical results and comparisons with competitive methods demonstrate that the proposed algorithms achieve both objectives with a modest number of anchors. Since our method relies only on target-to-anchor distances, it is broadly applicable and yields resilient, robust localization. 

**Abstract (ZH)**: 大规模低功耗无线 sensor 网络在多种应用中的部署研究：识别易受影响节点并 robust 估计其位置 

---
# Query-Based Adaptive Aggregation for Multi-Dataset Joint Training Toward Universal Visual Place Recognition 

**Title (ZH)**: 基于查询的自适应聚合多数据集联合训练以实现通用视觉场所识别 

**Authors**: Jiuhong Xiao, Yang Zhou, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2507.03831)  

**Abstract**: Deep learning methods for Visual Place Recognition (VPR) have advanced significantly, largely driven by large-scale datasets. However, most existing approaches are trained on a single dataset, which can introduce dataset-specific inductive biases and limit model generalization. While multi-dataset joint training offers a promising solution for developing universal VPR models, divergences among training datasets can saturate limited information capacity in feature aggregation layers, leading to suboptimal performance. To address these challenges, we propose Query-based Adaptive Aggregation (QAA), a novel feature aggregation technique that leverages learned queries as reference codebooks to effectively enhance information capacity without significant computational or parameter complexity. We show that computing the Cross-query Similarity (CS) between query-level image features and reference codebooks provides a simple yet effective way to generate robust descriptors. Our results demonstrate that QAA outperforms state-of-the-art models, achieving balanced generalization across diverse datasets while maintaining peak performance comparable to dataset-specific models. Ablation studies further explore QAA's mechanisms and scalability. Visualizations reveal that the learned queries exhibit diverse attention patterns across datasets. Code will be publicly released. 

**Abstract (ZH)**: 基于查询的自适应聚合（QAA）：一种用于视觉位置识别的新型特征聚合技术 

---
# Enabling Robust, Real-Time Verification of Vision-Based Navigation through View Synthesis 

**Title (ZH)**: 基于视图合成实现稳健的实时视觉导航验证 

**Authors**: Marius Neuhalfen, Jonathan Grzymisch, Manuel Sanchez-Gestido  

**Link**: [PDF](https://arxiv.org/pdf/2507.02993)  

**Abstract**: This work introduces VISY-REVE: a novel pipeline to validate image processing algorithms for Vision-Based Navigation. Traditional validation methods such as synthetic rendering or robotic testbed acquisition suffer from difficult setup and slow runtime. Instead, we propose augmenting image datasets in real-time with synthesized views at novel poses. This approach creates continuous trajectories from sparse, pre-existing datasets in open or closed-loop. In addition, we introduce a new distance metric between camera poses, the Boresight Deviation Distance, which is better suited for view synthesis than existing metrics. Using it, a method for increasing the density of image datasets is developed. 

**Abstract (ZH)**: VISY-REVE：一种基于视觉的导航图像处理算法验证的新管道 

---
# DriveMRP: Enhancing Vision-Language Models with Synthetic Motion Data for Motion Risk Prediction 

**Title (ZH)**: DriveMRP: 通过合成运动数据增强视觉-语言模型以进行运动风险预测 

**Authors**: Zhiyi Hou, Enhui Ma, Fang Li, Zhiyi Lai, Kalok Ho, Zhanqian Wu, Lijun Zhou, Long Chen, Chitian Sun, Haiyang Sun, Bing Wang, Guang Chen, Hangjun Ye, Kaicheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02948)  

**Abstract**: Autonomous driving has seen significant progress, driven by extensive real-world data. However, in long-tail scenarios, accurately predicting the safety of the ego vehicle's future motion remains a major challenge due to uncertainties in dynamic environments and limitations in data coverage. In this work, we aim to explore whether it is possible to enhance the motion risk prediction capabilities of Vision-Language Models (VLM) by synthesizing high-risk motion data. Specifically, we introduce a Bird's-Eye View (BEV) based motion simulation method to model risks from three aspects: the ego-vehicle, other vehicles, and the environment. This allows us to synthesize plug-and-play, high-risk motion data suitable for VLM training, which we call DriveMRP-10K. Furthermore, we design a VLM-agnostic motion risk estimation framework, named DriveMRP-Agent. This framework incorporates a novel information injection strategy for global context, ego-vehicle perspective, and trajectory projection, enabling VLMs to effectively reason about the spatial relationships between motion waypoints and the environment. Extensive experiments demonstrate that by fine-tuning with DriveMRP-10K, our DriveMRP-Agent framework can significantly improve the motion risk prediction performance of multiple VLM baselines, with the accident recognition accuracy soaring from 27.13% to 88.03%. Moreover, when tested via zero-shot evaluation on an in-house real-world high-risk motion dataset, DriveMRP-Agent achieves a significant performance leap, boosting the accuracy from base_model's 29.42% to 68.50%, which showcases the strong generalization capabilities of our method in real-world scenarios. 

**Abstract (ZH)**: 自主驾驶技术在大量现实世界数据的驱动下取得了显著进步。然而，在长尾场景中，由于动态环境的不确定性以及数据覆盖范围的限制，准确预测ego车辆未来运动的安全性仍然是一个重大挑战。本文旨在探索通过合成高风险运动数据是否能增强视觉-语言模型（VLM）的运动风险预测能力。具体而言，我们介绍了基于鸟瞰视图（BEV）的运动模拟方法，从ego车辆、其他车辆和环境三个角度建模风险。这使得我们能够合成适用于VLM训练的即插即用式高风险运动数据，命名为DriveMRP-10K。此外，我们设计了一个VLM无关的运动风险估计框架，名为DriveMRP-Agent。该框架融合了一种新颖的信息注入策略，用于全局上下文、ego车辆视角和轨迹预测，使VLM能够有效推理运动参考点与环境之间的空间关系。大量实验证明，通过使用DriveMRP-10K进行微调，我们的DriveMRP-Agent框架可以在多个VLM基线下显著提高运动风险预测性能，事故识别准确率从27.13%提高到88.03%。此外，在对内部高风险运动数据集进行零样本评估时，DriveMRP-Agent实现了显著的性能提升，从基模型的29.42%提高到68.50%，这展示了我们方法在实际场景中的强大泛化能力。 

---
# Control Synthesis in Partially Observable Environments for Complex Perception-Related Objectives 

**Title (ZH)**: 部分可观测环境中复杂感知相关目标的控制合成 

**Authors**: Zetong Xuan, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.02942)  

**Abstract**: Perception-related tasks often arise in autonomous systems operating under partial observability. This work studies the problem of synthesizing optimal policies for complex perception-related objectives in environments modeled by partially observable Markov decision processes. To formally specify such objectives, we introduce \emph{co-safe linear inequality temporal logic} (sc-iLTL), which can define complex tasks that are formed by the logical concatenation of atomic propositions as linear inequalities on the belief space of the POMDPs. Our solution to the control synthesis problem is to transform the \mbox{sc-iLTL} objectives into reachability objectives by constructing the product of the belief MDP and a deterministic finite automaton built from the sc-iLTL objective. To overcome the scalability challenge due to the product, we introduce a Monte Carlo Tree Search (MCTS) method that converges in probability to the optimal policy. Finally, a drone-probing case study demonstrates the applicability of our method. 

**Abstract (ZH)**: 部分可观测条件下感知相关任务往往在自主系统中出现。本文研究了在部分可观测马尔可夫决策过程建模的环境中合成复杂感知相关目标的最优策略问题。为正式指定此类目标，本文引入了\emph{co-safe线性不等式时序逻辑}（sc-iLTL），它可以将由POMDP信念空间上的原子命题的逻辑连接定义的复杂任务构造成线性不等式。我们解决控制合成问题的方法是通过构建POMDP信念MDP与从sc-iLTL目标构建的确定性有限自动机的积来将sc-iLTL目标转换为可达性目标。为了克服由于积带来的可扩展性挑战，本文引入了一种蒙特卡罗树搜索（MCTS）方法，该方法以概率收敛于最优策略。最后，无人机探测案例研究展示了我们方法的应用。 

---
# A Simulator Dataset to Support the Study of Impaired Driving 

**Title (ZH)**: 驾驶能力受损Simulator数据集 

**Authors**: John Gideon, Kimimasa Tamura, Emily Sumner, Laporsha Dees, Patricio Reyes Gomez, Bassamul Haq, Todd Rowell, Avinash Balachandran, Simon Stent, Guy Rosman  

**Link**: [PDF](https://arxiv.org/pdf/2507.02867)  

**Abstract**: Despite recent advances in automated driving technology, impaired driving continues to incur a high cost to society. In this paper, we present a driving dataset designed to support the study of two common forms of driver impairment: alcohol intoxication and cognitive distraction. Our dataset spans 23.7 hours of simulated urban driving, with 52 human subjects under normal and impaired conditions, and includes both vehicle data (ground truth perception, vehicle pose, controls) and driver-facing data (gaze, audio, surveys). It supports analysis of changes in driver behavior due to alcohol intoxication (0.10\% blood alcohol content), two forms of cognitive distraction (audio n-back and sentence parsing tasks), and combinations thereof, as well as responses to a set of eight controlled road hazards, such as vehicle cut-ins. The dataset will be made available at this https URL. 

**Abstract (ZH)**: 尽管自动驾驶技术recent取得了进步，酒后驾驶和认知分心仍对社会造成重大影响。本文介绍了一个用于研究酒后驾驶和认知分心这两种常见驾驶员 impairment 影响的驾驶数据集。该数据集涵盖了23.7小时的模拟城市驾驶场景，涉及52名在正常和 impaired 条件下的受试者，并包含了车辆数据（真实感知、车辆姿态、控制）和面向驾驶员的数据（注视点、音频、问卷调查）。它支持分析血液酒精含量为0.10%时酒后驾驶对驾驶员行为的影响、两种类型的认知分心（音频n-back任务和句子解析任务）以及这些影响的组合，同时还涵盖了受控道路危害（如车辆切入）的应对情况。该数据集将在此网址获取：this https URL。 

---
