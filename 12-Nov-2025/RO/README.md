# SeFA-Policy: Fast and Accurate Visuomotor Policy Learning with Selective Flow Alignment 

**Title (ZH)**: SeFA策略：快速准确的选择性流对齐视觉运动策略学习 

**Authors**: Rong Xue, Jiageng Mao, Mingtong Zhang, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08583)  

**Abstract**: Developing efficient and accurate visuomotor policies poses a central challenge in robotic imitation learning. While recent rectified flow approaches have advanced visuomotor policy learning, they suffer from a key limitation: After iterative distillation, generated actions may deviate from the ground-truth actions corresponding to the current visual observation, leading to accumulated error as the reflow process repeats and unstable task execution. We present Selective Flow Alignment (SeFA), an efficient and accurate visuomotor policy learning framework. SeFA resolves this challenge by a selective flow alignment strategy, which leverages expert demonstrations to selectively correct generated actions and restore consistency with observations, while preserving multimodality. This design introduces a consistency correction mechanism that ensures generated actions remain observation-aligned without sacrificing the efficiency of one-step flow inference. Extensive experiments across both simulated and real-world manipulation tasks show that SeFA Policy surpasses state-of-the-art diffusion-based and flow-based policies, achieving superior accuracy and robustness while reducing inference latency by over 98%. By unifying rectified flow efficiency with observation-consistent action generation, SeFA provides a scalable and dependable solution for real-time visuomotor policy learning. Code is available on this https URL. 

**Abstract (ZH)**: 开发高效准确的视听运动策略是机器人模仿学习中的核心挑战。虽然近期的修正流方法已经提升了视听运动策略的学习，但它们存在一个关键限制：在迭代精炼过程中，生成的动作可能偏离当前视觉观察对应的真实动作，导致在反复修正过程中累积误差并使任务执行变得不稳定。我们提出了选择性流对齐（SeFA），一种高效准确的视听运动策略学习框架。SeFA 通过一种选择性流对齐策略来解决这一挑战，该策略利用专家演示来选择性地纠正生成的动作，恢复与观察的一致性，同时保持多模态性。该设计引入了一致性纠正机制，确保生成的动作保持与观察的一致性，而不牺牲单步流推理的效率。在模拟和真实世界操作任务的广泛实验中，SeFA 策略超过了最先进的扩散模型和流模型策略，在准确性和鲁棒性方面表现出色，并将推理延迟降低了超过98%。通过对齐修正流效率性和观测一致动作生成，SeFA 提供了一种可扩展且可靠的实时视听运动策略学习解决方案。代码可在以下链接获取：this https URL。 

---
# Safe and Optimal Learning from Preferences via Weighted Temporal Logic with Applications in Robotics and Formula 1 

**Title (ZH)**: 安全且最优地从偏好中学习：基于加权时序逻辑的方法及其在机器人技术与一级方程式中的应用 

**Authors**: Ruya Karagulle, Cristian-Ioan Vasile, Necmiye Ozay  

**Link**: [PDF](https://arxiv.org/pdf/2511.08502)  

**Abstract**: Autonomous systems increasingly rely on human feedback to align their behavior, expressed as pairwise comparisons, rankings, or demonstrations. While existing methods can adapt behaviors, they often fail to guarantee safety in safety-critical domains. We propose a safety-guaranteed, optimal, and efficient approach to solve the learning problem from preferences, rankings, or demonstrations using Weighted Signal Temporal Logic (WSTL). WSTL learning problems, when implemented naively, lead to multi-linear constraints in the weights to be learned. By introducing structural pruning and log-transform procedures, we reduce the problem size and recast the problem as a Mixed-Integer Linear Program while preserving safety guarantees. Experiments on robotic navigation and real-world Formula 1 data demonstrate that the method effectively captures nuanced preferences and models complex task objectives. 

**Abstract (ZH)**: 自主系统日益依赖人类反馈来调整其行为，这些反馈可以是成对比较、排名或示范。虽然现有方法可以适应行为，但在安全关键领域往往无法确保安全性。我们提出了一种安全保证、最优且高效的基于加权信号时序逻辑（WSTL）从偏好、排名或示范中学习的方法。通过引入结构化剪枝和对数变换程序，我们减少了问题规模，并将问题重新表述为混合整数线性规划问题，同时保留了安全保证。实验结果显示，该方法能够有效捕捉细腻的偏好并建模复杂的任务目标。 

---
# A Supervised Autonomous Resection and Retraction Framework for Transurethral Enucleation of the Prostatic Median Lobe 

**Title (ZH)**: 经尿道前列腺中叶enucleation的监督自主切除和复位框架 

**Authors**: Mariana Smith, Tanner Watts, Susheela Sharma Stern, Brendan Burkhart, Hao Li, Alejandro O. Chara, Nithesh Kumar, James Ferguson, Ayberk Acar, Jesse F. d'Almeida, Lauren Branscombe, Lauren Shepard, Ahmed Ghazi, Ipek Oguz, Jie Ying Wu, Robert J. Webster III, Axel Krieger, Alan Kuntz  

**Link**: [PDF](https://arxiv.org/pdf/2511.08490)  

**Abstract**: Concentric tube robots (CTRs) offer dexterous motion at millimeter scales, enabling minimally invasive procedures through natural orifices. This work presents a coordinated model-based resection planner and learning-based retraction network that work together to enable semi-autonomous tissue resection using a dual-arm transurethral concentric tube robot (the Virtuoso). The resection planner operates directly on segmented CT volumes of prostate phantoms, automatically generating tool trajectories for a three-phase median lobe resection workflow: left/median trough resection, right/median trough resection, and median blunt dissection. The retraction network, PushCVAE, trained on surgeon demonstrations, generates retractions according to the procedural phase. The procedure is executed under Level-3 (supervised) autonomy on a prostate phantom composed of hydrogel materials that replicate the mechanical and cutting properties of tissue. As a feasibility study, we demonstrate that our combined autonomous system achieves a 97.1% resection of the targeted volume of the median lobe. Our study establishes a foundation for image-guided autonomy in transurethral robotic surgery and represents a first step toward fully automated minimally-invasive prostate enucleation. 

**Abstract (ZH)**: 同心管机器人（CTRs）在毫米级尺度上提供灵巧运动，通过自然开口实现微创手术。本研究提出了一种协调的模型驱动切除计划器和基于学习的牵拉网络，共同实现双臂经尿道同心管机器人（Virtuoso）的半自主组织切除。切除计划器直接在前列腺假体的分割CT体积上运行，自动生成三阶段中叶切除工作流程中的工具轨迹：左侧/中叶沟切除、右侧/中叶沟切除和中叶钝性分离。牵拉网络PushCVAE根据手术阶段进行训练，生成相应的牵拉动作。该手术在由模拟组织机械和切割特性的水凝胶材料组成的前列腺假体上以监督级（Level-3）自主性执行。作为可行性研究，我们展示了我们综合的自主系统实现了97.1%的目标中叶体积切除。本研究为经尿道机器人手术中的图像引导自主性奠定了基础，并代表了实现完全自动化微创前列腺剜除的第一步。 

---
# Intuitive control of supernumerary robotic limbs through a tactile-encoded neural interface 

**Title (ZH)**: 通过触觉编码神经接口直观控制额外 robotic 臂 

**Authors**: Tianyu Jia, Xingchen Yang, Ciaran McGeady, Yifeng Li, Jinzhi Lin, Kit San Ho, Feiyu Pan, Linhong Ji, Chong Li, Dario Farina  

**Link**: [PDF](https://arxiv.org/pdf/2511.08454)  

**Abstract**: Brain-computer interfaces (BCIs) promise to extend human movement capabilities by enabling direct neural control of supernumerary effectors, yet integrating augmented commands with multiple degrees of freedom without disrupting natural movement remains a key challenge. Here, we propose a tactile-encoded BCI that leverages sensory afferents through a novel tactile-evoked P300 paradigm, allowing intuitive and reliable decoding of supernumerary motor intentions even when superimposed with voluntary actions. The interface was evaluated in a multi-day experiment comprising of a single motor recognition task to validate baseline BCI performance and a dual task paradigm to assess the potential influence between the BCI and natural human movement. The brain interface achieved real-time and reliable decoding of four supernumerary degrees of freedom, with significant performance improvements after only three days of training. Importantly, after training, performance did not differ significantly between the single- and dual-BCI task conditions, and natural movement remained unimpaired during concurrent supernumerary control. Lastly, the interface was deployed in a movement augmentation task, demonstrating its ability to command two supernumerary robotic arms for functional assistance during bimanual tasks. These results establish a new neural interface paradigm for movement augmentation through stimulation of sensory afferents, expanding motor degrees of freedom without impairing natural movement. 

**Abstract (ZH)**: 触觉编码脑-机接口：通过刺激感觉传入神经元扩展运动自由度而不干扰自然运动 

---
# Human Motion Intent Inferencing in Teleoperation Through a SINDy Paradigm 

**Title (ZH)**: 通过SINDy范式在远程操作中推断人类运动意图 

**Authors**: Michael Bowman, Xiaoli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08377)  

**Abstract**: Intent inferencing in teleoperation has been instrumental in aligning operator goals and coordinating actions with robotic partners. However, current intent inference methods often ignore subtle motion that can be strong indicators for a sudden change in intent. Specifically, we aim to tackle 1) if we can detect sudden jumps in operator trajectories, 2) how we appropriately use these sudden jump motions to infer an operator's goal state, and 3) how to incorporate these discontinuous and continuous dynamics to infer operator motion. Our framework, called Psychic, models these small indicative motions through a jump-drift-diffusion stochastic differential equation to cover discontinuous and continuous dynamics. Kramers-Moyal (KM) coefficients allow us to detect jumps with a trajectory which we pair with a statistical outlier detection algorithm to nominate goal transitions. Through identifying jumps, we can perform early detection of existing goals and discover undefined goals in unstructured scenarios. Our framework then applies a Sparse Identification of Nonlinear Dynamics (SINDy) model using KM coefficients with the goal transitions as a control input to infer an operator's motion behavior in unstructured scenarios. We demonstrate Psychic can produce probabilistic reachability sets and compare our strategy to a negative log-likelihood model fit. We perform a retrospective study on 600 operator trajectories in a hands-free teleoperation task to evaluate the efficacy of our opensource package, Psychic, in both offline and online learning. 

**Abstract (ZH)**: 基于跳跃-漂移-扩散过程的心理推理在远程操控中的意图推断通过建模操作员轨迹中的细微指示性运动，以突显动作来调整操作员目标并协调与机器人的动作。然而，当前的意图推理方法往往忽略了可能是意图突变强烈指示的细微运动。具体而言，我们旨在解决以下三个问题：1) 是否能够检测操作员轨迹中的突变跳跃；2) 如何适当地利用这些突变跳跃来推断操作员的目标状态；3) 如何结合断续和连续动力学来推断操作员的运动。我们提出的一种框架Psychic，通过跳跃-漂移-扩散随机微分方程建模这些细微指示性运动，以涵盖断续和连续动力学。Kramers-Moyal (KM) 系数使我们能够检测轨迹中的跳跃，并将其与统计异常检测算法配对以提出目标转换。通过识别跳跃，我们可以提前检测现有目标并在非结构化场景中发现未定义的目标。然后，我们的框架使用KM系数和目标转换作为控制输入，通过Sparse Identification of Nonlinear Dynamics (SINDy) 模型来推断操作员在非结构化场景中的运动行为。我们展示了Psychic能够生成可达性集合，并将我们的策略与负对数似然模型进行比较。我们对600个操作员轨迹进行回顾性研究，以评估我们开源包Psychic在离线和在线学习中的有效性。 

---
# A CODECO Case Study and Initial Validation for Edge Orchestration of Autonomous Mobile Robots 

**Title (ZH)**: CODECO 案例研究及边缘 orchestration 自主移动机器人初步验证 

**Authors**: H. Zhu, T. Samizadeh, R. C. Sofia  

**Link**: [PDF](https://arxiv.org/pdf/2511.08354)  

**Abstract**: Autonomous Mobile Robots (AMRs) increasingly adopt containerized micro-services across the Edge-Cloud continuum. While Kubernetes is the de-facto orchestrator for such systems, its assumptions of stable networks, homogeneous resources, and ample compute capacity do not fully hold in mobile, resource-constrained robotic environments.
This paper describes a case study on smart-manufacturing AMRs and performs an initial comparison between CODECO orchestration and standard Kubernetes using a controlled KinD environment. Metrics include pod deployment and deletion times, CPU and memory usage, and inter-pod data rates. The observed results indicate that CODECO offers reduced CPU consumption and more stable communication patterns, at the cost of modest memory overhead (10-15%) and slightly increased pod lifecycle latency due to secure overlay initialization. 

**Abstract (ZH)**: 自主移动机器人（AMRs）越来越多地采用边缘-云连续体中的容器化微服务。虽然Kubernetes是此类系统的事实上的编排器，但其对稳定网络、均质资源和充足计算能力的假设并不完全适用于移动且资源受限的机器人环境。

本文描述了一项关于智能制造AMRs的案例研究，并在受控的KinD环境中对CODECO编排与标准Kubernetes进行了初步比较。评估指标包括部署和删除Pod的时间、CPU和内存使用情况，以及Pod之间数据传输率。观察结果显示，CODECO在消耗CPU方面有所减少，并提供了更稳定的通信模式，但伴随着10-15%的内存开销增加和轻微增加的Pod生命周期延迟（由于安全覆盖网络初始化）。 

---
# Learning Omnidirectional Locomotion for a Salamander-Like Quadruped Robot 

**Title (ZH)**: 仿蝾螈 quadruped 机器人全方位移动学习 

**Authors**: Zhiang Liu, Yang Liu, Yongchun Fang, Xian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2511.08299)  

**Abstract**: Salamander-like quadruped robots are designed inspired by the skeletal structure of their biological counterparts. However, existing controllers cannot fully exploit these morphological features and largely rely on predefined gait patterns or joint trajectories, which prevents the generation of diverse and flexible locomotion and limits their applicability in real-world scenarios. In this paper, we propose a learning framework that enables the robot to acquire a diverse repertoire of omnidirectional gaits without reference motions. Each body part is controlled by a phase variable capable of forward and backward evolution, with a phase coverage reward to promote the exploration of the leg phase space. Additionally, morphological symmetry of the robot is incorporated via data augmentation, improving sample efficiency and enforcing both motion-level and task-level symmetry in learned behaviors. Extensive experiments show that the robot successfully acquires 22 omnidirectional gaits exhibiting both dynamic and symmetric movements, demonstrating the effectiveness of the proposed learning framework. 

**Abstract (ZH)**: 类似于蝾螈的四足机器人设计灵感来源于其生物对应物的骨骼结构。然而，现有的控制器无法充分利用这些形态特征，主要依赖预定义的步伐模式或关节轨迹，这限制了它们生成多样化和灵活运动的能力，并限制了其在实际场景中的应用。在本文中，我们提出了一种学习框架，使机器人能够在没有参考运动的情况下获得 diverse 的全方位步伐。每个身体部分都由一个相位变量控制，该变量能够进行正反向演化，并通过相位覆盖奖励促进腿部相空间的探索。此外，通过数据增强将机器人形态对称性融入其中，提高样本效率并确保在学习行为中实现运动级别和任务级别的对称性。大量实验表明，该机器人成功获得了 22 种全方位步伐，展示了动态和对称运动的有效性，证明了所提出学习框架的有效性。 

---
# X-IONet: Cross-Platform Inertial Odometry Network with Dual-Stage Attention 

**Title (ZH)**: X-IONet：跨平台惯性里程计网络带双重注意机制 

**Authors**: Dehan Shen, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.08277)  

**Abstract**: Learning-based inertial odometry has achieved remarkable progress in pedestrian navigation. However, extending these methods to quadruped robots remains challenging due to their distinct and highly dynamic motion patterns. Models that perform well on pedestrian data often experience severe degradation when deployed on legged platforms. To tackle this challenge, we introduce X-IONet, a cross-platform inertial odometry framework that operates solely using a single Inertial Measurement Unit (IMU). X-IONet incorporates a rule-based expert selection module to classify motion platforms and route IMU sequences to platform-specific expert networks. The displacement prediction network features a dual-stage attention architecture that jointly models long-range temporal dependencies and inter-axis correlations, enabling accurate motion representation. It outputs both displacement and associated uncertainty, which are further fused through an Extended Kalman Filter (EKF) for robust state estimation. Extensive experiments on public pedestrian datasets and a self-collected quadruped robot dataset demonstrate that X-IONet achieves state-of-the-art performance, reducing Absolute Trajectory Error (ATE) by 14.3% and Relative Trajectory Error (RTE) by 11.4% on pedestrian data, and by 52.8% and 41.3% on quadruped robot data. These results highlight the effectiveness of X-IONet in advancing accurate and robust inertial navigation across both human and legged robot platforms. 

**Abstract (ZH)**: 基于学习的惯性里程计在行人导航中取得了显著进步。然而，将这些方法扩展到四足机器人上仍然具有挑战性，因为它们具有独特的高度动态运动模式。适用于行人数据的模型在部署到多足平台时往往会遇到严重的性能下降。为了应对这一挑战，我们提出了一种跨平台惯性里程计框架X-IONet，该框架仅使用单个惯性测量单元（IMU）进行操作。X-IONet包含一个基于规则的专家选择模块，用于分类运动平台并路由IMU序列到特定平台的专家网络。位移预测网络采用双重注意力架构，同时建模长时序依赖性和轴间相关性，从而实现准确的运动表示。它输出位移及其相关不确定性，并通过扩展卡尔曼滤波器（EKF）融合以实现稳健的状态估计。在公共行人数据集和自收集的四足机器人数据集上的广泛实验表明，X-IONet达到了最先进的性能，行人数据中的绝对轨迹误差（ATE）减少了14.3%，相对轨迹误差（RTE）减少了11.4%，四足机器人数据中的这些误差分别减少了52.8%和41.3%。这些结果突显了X-IONet在促进跨人类和多足机器人平台准确且稳健的惯性导航方面的有效性。 

---
# Real-Time Performance Analysis of Multi-Fidelity Residual Physics-Informed Neural Process-Based State Estimation for Robotic Systems 

**Title (ZH)**: 基于多保真剩余物理知情神经过程的状态估计的机器人系统实时性能分析 

**Authors**: Devin Hunter, Chinwendu Enyioha  

**Link**: [PDF](https://arxiv.org/pdf/2511.08231)  

**Abstract**: Various neural network architectures are used in many of the state-of-the-art approaches for real-time nonlinear state estimation. With the ever-increasing incorporation of these data-driven models into the estimation domain, model predictions with reliable margins of error are a requirement -- especially for safety-critical applications. This paper discusses the application of a novel real-time, data-driven estimation approach based on the multi-fidelity residual physics-informed neural process (MFR-PINP) toward the real-time state estimation of a robotic system. Specifically, we address the model-mismatch issue of selecting an accurate kinematic model by tasking the MFR-PINP to also learn the residuals between simple, low-fidelity predictions and complex, high-fidelity ground-truth dynamics. To account for model uncertainty present in a physical implementation, robust uncertainty guarantees from the split conformal (SC) prediction framework are modeled in the training and inference paradigms. We provide implementation details of our MFR-PINP-based estimator for a hybrid online learning setting to validate our model's usage in real-time applications. Experimental results of our approach's performance in comparison to the state-of-the-art variants of the Kalman filter (i.e. unscented Kalman filter and deep Kalman filter) in estimation scenarios showed promising results for the MFR-PINP model as a viable option in real-time estimation tasks. 

**Abstract (ZH)**: 基于多保真度残差物理知情神经过程的实时数据驱动状态估计算法在机器人系统的应用 

---
# Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving 

**Title (ZH)**: 基于感知指导的自监督优先：端到端自动驾驶中的因果建模新范式 

**Authors**: Yi Huang, Zhan Qu, Lihui Jiang, Bingbing Liu, Hongbo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08214)  

**Abstract**: End-to-end autonomous driving systems, predominantly trained through imitation learning, have demonstrated considerable effectiveness in leveraging large-scale expert driving data. Despite their success in open-loop evaluations, these systems often exhibit significant performance degradation in closed-loop scenarios due to causal confusion. This confusion is fundamentally exacerbated by the overreliance of the imitation learning paradigm on expert trajectories, which often contain unattributable noise and interfere with the modeling of causal relationships between environmental contexts and appropriate driving actions.
To address this fundamental limitation, we propose Perception-Guided Self-Supervision (PGS) - a simple yet effective training paradigm that leverages perception outputs as the primary supervisory signals, explicitly modeling causal relationships in decision-making. The proposed framework aligns both the inputs and outputs of the decision-making module with perception results, such as lane centerlines and the predicted motions of surrounding agents, by introducing positive and negative self-supervision for the ego trajectory. This alignment is specifically designed to mitigate causal confusion arising from the inherent noise in expert trajectories.
Equipped with perception-driven supervision, our method, built on a standard end-to-end architecture, achieves a Driving Score of 78.08 and a mean success rate of 48.64% on the challenging closed-loop Bench2Drive benchmark, significantly outperforming existing state-of-the-art methods, including those employing more complex network architectures and inference pipelines. These results underscore the effectiveness and robustness of the proposed PGS framework and point to a promising direction for addressing causal confusion and enhancing real-world generalization in autonomous driving. 

**Abstract (ZH)**: 端到端自动驾驶系统——基于感知引导的自监督训练范式在利用大量专家驾驶数据进行主要训练方面展示了显著效果。尽管这些系统在开环评估中表现出色，但在闭环场景中往往因为因果混淆而大幅性能下降。这种混淆从根本上说是由于模仿学习范式过度依赖专家轨迹所致，而这些轨迹中往往包含不可归因的噪声，干扰了对环境上下文与合适驾驶动作之间因果关系的建模。

为解决这一根本性限制，我们提出了一种名为感知引导的自监督（Perception-Guided Self-Supervision, PGS）的简单而有效的训练范式，该范式利用感知输出作为主要的监督信号，明确建模决策中的因果关系。所提出的框架将决策模块的输入和输出与感知结果（如车道中心线和周围代理的预测运动）对齐，通过引入对自我轨迹的正负自监督来实现这一点。这种对齐设计旨在减轻由于专家轨迹固有噪声引起的原因混淆。

利用感知驱动的监督，基于标准端到端架构的方法在具有挑战性的闭环Bench2Drive基准测试中获得了78.08的驾驶得分和48.64%的平均成功率，显著优于现有最先进的方法，包括那些采用更复杂网络架构和推理管道的方法。这些结果突显了所提出的PGS框架的有效性和稳健性，并指出了解决因果混淆和增强自动驾驶真实世界泛化的有前途的方向。 

---
# PerspAct: Enhancing LLM Situated Collaboration Skills through Perspective Taking and Active Vision 

**Title (ZH)**: PerspAct: 通过换位思考和主动视觉提升大语言模型的现场协作技能 

**Authors**: Sabrina Patania, Luca Annese, Anita Pellegrini, Silvia Serino, Anna Lambiase, Luca Pallonetto, Silvia Rossi, Simone Colombani, Tom Foulsham, Azzurra Ruggeri, Dimitri Ognibene  

**Link**: [PDF](https://arxiv.org/pdf/2511.08098)  

**Abstract**: Recent advances in Large Language Models (LLMs) and multimodal foundation models have significantly broadened their application in robotics and collaborative systems. However, effective multi-agent interaction necessitates robust perspective-taking capabilities, enabling models to interpret both physical and epistemic viewpoints. Current training paradigms often neglect these interactive contexts, resulting in challenges when models must reason about the subjectivity of individual perspectives or navigate environments with multiple observers. This study evaluates whether explicitly incorporating diverse points of view using the ReAct framework, an approach that integrates reasoning and acting, can enhance an LLM's ability to understand and ground the demands of other agents. We extend the classic Director task by introducing active visual exploration across a suite of seven scenarios of increasing perspective-taking complexity. These scenarios are designed to challenge the agent's capacity to resolve referential ambiguity based on visual access and interaction, under varying state representations and prompting strategies, including ReAct-style reasoning. Our results demonstrate that explicit perspective cues, combined with active exploration strategies, significantly improve the model's interpretative accuracy and collaborative effectiveness. These findings highlight the potential of integrating active perception with perspective-taking mechanisms in advancing LLMs' application in robotics and multi-agent systems, setting a foundation for future research into adaptive and context-aware AI systems. 

**Abstract (ZH)**: 最近大规模语言模型和多模态基础模型的进步显著扩展了其在机器人和协作系统中的应用。然而，有效的多智能体交互需要强大的换位思考能力，使模型能够解释物理和知识视角。当前的训练范式往往忽视了这些互动情境，导致模型在需要考虑个体视角的主观性或在多观察者环境中导航时面临挑战。本研究评估了使用ReAct框架（该框架结合了推理和行动）明确纳入多种观点是否能够增强语言模型理解并落实其他智能体需求的能力。我们在经典导演任务的基础上，引入了跨七个递增换位思考复杂度场景的主动视觉探索。这些场景旨在基于视觉访问和交互解决引用歧义，并通过不同的状态表示和提示策略，包括ReAct风格的推理，挑战智能体的能力。结果显示，明确的观点提示结合主动探索策略显著提高了模型的解释准确性和协作有效性。这些发现突显了将主动感知与换位思考机制结合在机器人和多智能体系统中应用大规模语言模型的潜力，并为未来研究自适应和情境感知AI系统奠定了基础。 

---
# Model Predictive Control via Probabilistic Inference: A Tutorial 

**Title (ZH)**: 概率推断视角下的模型预测控制：一个教程 

**Authors**: Kohei Honda  

**Link**: [PDF](https://arxiv.org/pdf/2511.08019)  

**Abstract**: Model Predictive Control (MPC) is a fundamental framework for optimizing robot behavior over a finite future horizon. While conventional numerical optimization methods can efficiently handle simple dynamics and cost structures, they often become intractable for the nonlinear or non-differentiable systems commonly encountered in robotics. This article provides a tutorial on probabilistic inference-based MPC, presenting a unified theoretical foundation and a comprehensive overview of representative methods. Probabilistic inference-based MPC approaches, such as Model Predictive Path Integral (MPPI) control, have gained significant attention by reinterpreting optimal control as a problem of probabilistic inference. Rather than relying on gradient-based numerical optimization, these methods estimate optimal control distributions through sampling-based techniques, accommodating arbitrary cost functions and dynamics. We first derive the optimal control distribution from the standard optimal control problem, elucidating its probabilistic interpretation and key characteristics. The widely used MPPI algorithm is then derived as a practical example, followed by discussions on prior and variational distribution design, tuning principles, and theoretical aspects. This article aims to serve as a systematic guide for researchers and practitioners seeking to understand, implement, and extend these methods in robotics and beyond. 

**Abstract (ZH)**: 基于概率推断的模型预测控制：一种统一的理论基础和综述 

---
# AVOID-JACK: Avoidance of Jackknifing for Swarms of Long Heavy Articulated Vehicles 

**Title (ZH)**: AVOID-JACK: 避免偏态校正法对长重型articulated车辆群的适用性 

**Authors**: Adrian Schönnagel, Michael Dubé, Christoph Steup, Felix Keppler, Sanaz Mostaghim  

**Link**: [PDF](https://arxiv.org/pdf/2511.08016)  

**Abstract**: This paper presents a novel approach to avoiding jackknifing and mutual collisions in Heavy Articulated Vehicles (HAVs) by leveraging decentralized swarm intelligence. In contrast to typical swarm robotics research, our robots are elongated and exhibit complex kinematics, introducing unique challenges. Despite its relevance to real-world applications such as logistics automation, remote mining, airport baggage transport, and agricultural operations, this problem has not been addressed in the existing literature.
To tackle this new class of swarm robotics problems, we propose a purely reaction-based, decentralized swarm intelligence strategy tailored to automate elongated, articulated vehicles. The method presented in this paper prioritizes jackknifing avoidance and establishes a foundation for mutual collision avoidance. We validate our approach through extensive simulation experiments and provide a comprehensive analysis of its performance. For the experiments with a single HAV, we observe that for 99.8% jackknifing was successfully avoided and that 86.7% and 83.4% reach their first and second goals, respectively. With two HAVs interacting, we observe 98.9%, 79.4%, and 65.1%, respectively, while 99.7% of the HAVs do not experience mutual collisions. 

**Abstract (ZH)**: 本文提出了一种通过利用分布式 swarm 智能来避免重型铰接车辆（HAVs）侧翻及互撞的新方法。针对不同于典型 swarm 机器人研究的对象，我们的机器人具有伸长的形态和复杂的运动学特性，引入了独特挑战。尽管这一问题与物流自动化、远程采矿、机场行李运输及农业操作等实际应用密切相关，但现有文献尚未对此进行研究。

为应对这一新的 swarm 机器人问题类别，我们提出了一种专为自动化伸长铰接车辆设计的纯反应式、分布式 swarm 智能策略。本文提出的方法优先避免侧翻并为互撞避免奠定了基础。通过广泛的模拟实验验证了我们的方法，并对其性能进行了全面分析。针对单一 HAV 的实验表明，99.8% 的侧翻情况被成功避免，86.7% 和 83.4% 的 HAV 分别实现了第一和第二个目标。在两个 HAV 相互作用的情况下，分别有 98.9%、79.4% 和 65.1% 的 HAV 完成了相应目标，同时 99.7% 的 HAV 避免了互相碰撞。 

---
# A Two-Layer Electrostatic Film Actuator with High Actuation Stress and Integrated Brake 

**Title (ZH)**: 具有高驱动应力和集成制动的两层静电薄膜驱动器 

**Authors**: Huacen Wang, Hongqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.08005)  

**Abstract**: Robotic systems driven by conventional motors often suffer from challenges such as large mass, complex control algorithms, and the need for additional braking mechanisms, which limit their applications in lightweight and compact robotic platforms. Electrostatic film actuators offer several advantages, including thinness, flexibility, lightweight construction, and high open-loop positioning accuracy. However, the actuation stress exhibited by conventional actuators in air still needs improvement, particularly for the widely used three-phase electrode design. To enhance the output performance of actuators, this paper presents a two-layer electrostatic film actuator with an integrated brake. By alternately distributing electrodes on both the top and bottom layers, a smaller effective electrode pitch is achieved under the same fabrication constraints, resulting in an actuation stress of approximately 241~N/m$^2$, representing a 90.5\% improvement over previous three-phase actuators operating in air. Furthermore, its integrated electrostatic adhesion mechanism enables load retention under braking mode. Several demonstrations, including a tug-of-war between a conventional single-layer actuator and the proposed two-layer actuator, a payload operation, a one-degree-of-freedom robotic arm, and a dual-mode gripper, were conducted to validate the actuator's advantageous capabilities in both actuation and braking modes. 

**Abstract (ZH)**: 基于传统电机驱动的机器人系统常常面临质量大、控制算法复杂以及需要额外制动机制等挑战，这限制了它们在轻量紧凑型机器人平台中的应用。静电薄膜执行器具有薄、柔、轻以及开环定位精度高的优点。然而，传统执行器在空气中的驱动应力仍需提升，特别是对于广泛使用的三相电极设计。为了增强执行器的输出性能，本文提出了一种具有集成制动机制的两层静电薄膜执行器。通过在上下层交替分布电极，使其在相同制备约束条件下实现了更小的有效电极间距，在空气中获得了约241~N/m$^2$的驱动应力，相比之前的三相执行器提升了90.5%。此外，其集成的静电吸附机制能够在制动模式下保持负载。通过多项演示，包括传统单层执行器与所提双层执行器的拔河对比、负载操作、单自由度机械臂和双模式夹爪等，验证了该执行器在驱动和制动模式下的优越性能。 

---
# Effective Game-Theoretic Motion Planning via Nested Search 

**Title (ZH)**: 基于嵌套搜索的有效博弈论运动规划 

**Authors**: Avishav Engle, Andrey Zhitnikov, Oren Salzman, Omer Ben-Porat, Kiril Solovey  

**Link**: [PDF](https://arxiv.org/pdf/2511.08001)  

**Abstract**: To facilitate effective, safe deployment in the real world, individual robots must reason about interactions with other agents, which often occur without explicit communication. Recent work has identified game theory, particularly the concept of Nash Equilibrium (NE), as a key enabler for behavior-aware decision-making. Yet, existing work falls short of fully unleashing the power of game-theoretic reasoning. Specifically, popular optimization-based methods require simplified robot dynamics and tend to get trapped in local minima due to convexification. Other works that rely on payoff matrices suffer from poor scalability due to the explicit enumeration of all possible trajectories. To bridge this gap, we introduce Game-Theoretic Nested Search (GTNS), a novel, scalable, and provably correct approach for computing NEs in general dynamical systems. GTNS efficiently searches the action space of all agents involved, while discarding trajectories that violate the NE constraint (no unilateral deviation) through an inner search over a lower-dimensional space. Our algorithm enables explicit selection among equilibria by utilizing a user-specified global objective, thereby capturing a rich set of realistic interactions. We demonstrate the approach on a variety of autonomous driving and racing scenarios where we achieve solutions in mere seconds on commodity hardware. 

**Abstract (ZH)**: 基于博弈论的嵌套搜索方法：一种通用动力系统中纳什均衡的高效可证明正确计算方法 

---
# USV Obstacles Detection and Tracking in Marine Environments 

**Title (ZH)**: USV 障碍检测与跟踪在海洋环境中的研究 

**Authors**: Yara AlaaEldin, Enrico Simetti, Francesca Odone  

**Link**: [PDF](https://arxiv.org/pdf/2511.07950)  

**Abstract**: Developing a robust and effective obstacle detection and tracking system for Unmanned Surface Vehicle (USV) at marine environments is a challenging task. Research efforts have been made in this area during the past years by GRAAL lab at the university of Genova that resulted in a methodology for detecting and tracking obstacles on the image plane and, then, locating them in the 3D LiDAR point cloud. In this work, we continue on the developed system by, firstly, evaluating its performance on recently published marine datasets. Then, we integrate the different blocks of the system on ROS platform where we could test it in real-time on synchronized LiDAR and camera data collected in various marine conditions available in the MIT marine datasets. We present a thorough experimental analysis of the results obtained using two approaches; one that uses sensor fusion between the camera and LiDAR to detect and track the obstacles and the other uses only the LiDAR point cloud for the detection and tracking. In the end, we propose a hybrid approach that merges the advantages of both approaches to build an informative obstacles map of the surrounding environment to the USV. 

**Abstract (ZH)**: 在海洋环境中开发一种 robust 和有效的障碍检测与跟踪系统是对无人水面车辆（USV）的一个challenge任务。Genova大学GRAAL实验室在过去几年对该领域进行了研究，提出了一种在图像平面检测和跟踪障碍物的方法，并进一步定位在3D LiDAR点云中。在本文中，我们在此基础上进行研究，首先评估该系统在近期发布的海洋数据集上的性能，然后将该系统的各个模块集成到ROS平台，利用MIT海洋数据集中各种海洋条件下同步的LiDAR和摄像头数据进行实时测试。我们使用两种方法进行了详尽的实验分析：一种是利用摄像头和LiDAR传感器融合进行障碍检测与跟踪，另一种仅使用LiDAR点云。最后，我们提出了一种混合方法，结合上述两种方法的优点，为USV构建一个周边环境的 informative 障碍地图。 

---
# Local Path Planning with Dynamic Obstacle Avoidance in Unstructured Environments 

**Title (ZH)**: 不结构化环境中具有动态障碍物避让的局部路径规划 

**Authors**: Okan Arif Guvenkaya, Selim Ahmet Iz, Mustafa Unel  

**Link**: [PDF](https://arxiv.org/pdf/2511.07927)  

**Abstract**: Obstacle avoidance and path planning are essential for guiding unmanned ground vehicles (UGVs) through environments that are densely populated with dynamic obstacles. This paper develops a novel approach that combines tangentbased path planning and extrapolation methods to create a new decision-making algorithm for local path planning. In the assumed scenario, a UGV has a prior knowledge of its initial and target points within the dynamic environment. A global path has already been computed, and the robot is provided with waypoints along this path. As the UGV travels between these waypoints, the algorithm aims to avoid collisions with dynamic obstacles. These obstacles follow polynomial trajectories, with their initial positions randomized in the local map and velocities randomized between O and the allowable physical velocity limit of the robot, along with some random accelerations. The developed algorithm is tested in several scenarios where many dynamic obstacles move randomly in the environment. Simulation results show the effectiveness of the proposed local path planning strategy by gradually generating a collision free path which allows the robot to navigate safely between initial and the target locations. 

**Abstract (ZH)**: 基于切线的路径规划和外推方法相结合的无人地面车辆避障与路径规划新方法 

---
# Dual-MPC Footstep Planning for Robust Quadruped Locomotion 

**Title (ZH)**: 双模型预测控制足步规划以实现稳健的四足运动 

**Authors**: Byeong-Il Ham, Hyun-Bin Kim, Jeonguk Kang, Keun Ha Choi, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2511.07921)  

**Abstract**: In this paper, we propose a footstep planning strategy based on model predictive control (MPC) that enables robust regulation of body orientation against undesired body rotations by optimizing footstep placement. Model-based locomotion approaches typically adopt heuristic methods or planning based on the linear inverted pendulum model. These methods account for linear velocity in footstep planning, while excluding angular velocity, which leads to angular momentum being handled exclusively via ground reaction force (GRF). Footstep planning based on MPC that takes angular velocity into account recasts the angular momentum control problem as a dual-input approach that coordinates GRFs and footstep placement, instead of optimizing GRFs alone, thereby improving tracking performance. A mutual-feedback loop couples the footstep planner and the GRF MPC, with each using the other's solution to iteratively update footsteps and GRFs. The use of optimal solutions reduces body oscillation and enables extended stance and swing phases. The method is validated on a quadruped robot, demonstrating robust locomotion with reduced oscillations, longer stance and swing phases across various terrains. 

**Abstract (ZH)**: 基于模型预测控制的步足规划策略：考虑角速度的体姿态 robust 调节 

---
# EquiMus: Energy-Equivalent Dynamic Modeling and Simulation of Musculoskeletal Robots Driven by Linear Elastic Actuators 

**Title (ZH)**: 等效动力学建模与仿真：由线性弹性驱动器驱动的肌骨骼机器人 

**Authors**: Yinglei Zhu, Xuguang Dong, Qiyao Wang, Qi Shao, Fugui Xie, Xinjun Liu, Huichan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2511.07887)  

**Abstract**: Dynamic modeling and control are critical for unleashing soft robots' potential, yet remain challenging due to their complex constitutive behaviors and real-world operating conditions. Bio-inspired musculoskeletal robots, which integrate rigid skeletons with soft actuators, combine high load-bearing capacity with inherent flexibility. Although actuation dynamics have been studied through experimental methods and surrogate models, accurate and effective modeling and simulation remain a significant challenge, especially for large-scale hybrid rigid--soft robots with continuously distributed mass, kinematic loops, and diverse motion modes. To address these challenges, we propose EquiMus, an energy-equivalent dynamic modeling framework and MuJoCo-based simulation for musculoskeletal rigid--soft hybrid robots with linear elastic actuators. The equivalence and effectiveness of the proposed approach are validated and examined through both simulations and real-world experiments on a bionic robotic leg. EquiMus further demonstrates its utility for downstream tasks, including controller design and learning-based control strategies. 

**Abstract (ZH)**: 生物启发的肌骨骼软体机器人的能量等效动态建模与控制：针对大规模混合刚柔机器人的挑战 

---
# A Comprehensive Experimental Characterization of Mechanical Layer Jamming Systems 

**Title (ZH)**: 全面的机械层卡滞系统实验表征 

**Authors**: Jessica Gumowski, Krishna Manaswi Digumarti, David Howard  

**Link**: [PDF](https://arxiv.org/pdf/2511.07882)  

**Abstract**: Organisms in nature, such as Cephalopods and Pachyderms, exploit stiffness modulation to achieve amazing dexterity in the control of their appendages. In this paper, we explore the phenomenon of layer jamming, which is a popular stiffness modulation mechanism that provides an equivalent capability for soft robots. More specifically, we focus on mechanical layer jamming, which we realise through two-layer multi material structure with tooth-like protrusions. We identify key design parameters for mechanical layer jamming systems, including the ability to modulate stiffness, and perform a variety of comprehensive tests placing the specimens under bending and torsional loads to understand the influence of our selected design parameters (mainly tooth geometry) on the performance of the jammed structures. We note the ability of these structures to produce a peak change in stiffness of 5 times in bending and 3.2 times in torsion. We also measure the force required to separate the two jammed layers, an often ignored parameter in the study of jamming-induced stiffness change. This study aims to shed light on the principled design of mechanical layer jammed systems and guide researchers in the selection of appropriate designs for their specific application domains. 

**Abstract (ZH)**: 自然界中的生物，如头足类和大型哺乳动物，利用刚度调节实现其附肢控制的惊人灵活性。本文探讨了层状阻塞现象，这是一种常见的刚度调节机制，为软体机器人提供了等效功能。具体而言，我们关注机械层状阻塞，通过两层多材料结构结合齿状突起实现。我们确定了机械层状阻塞系统的关键设计参数，包括刚度调节能力，并通过一系列综合测试，将标本置于弯曲和扭转负载下，以理解选定设计参数（主要是齿状结构几何）对阻塞结构性能的影响。我们注意到这些结构在弯曲时可产生5倍的刚度峰值变化，在扭转时可产生3.2倍的刚度峰值变化。我们还测量了分离两层阻塞所需的力，这是一个在研究阻塞引起的刚度变化中经常被忽略的参数。本研究旨在阐明机械层状阻塞系统的原理设计，并指导研究人员为其特定的应用领域选择合适的结构设计。 

---
# Occlusion-Aware Ground Target Search by a UAV in an Urban Environment 

**Title (ZH)**: 城市环境中无人机基于遮挡意识的地面目标搜索 

**Authors**: Collin Hague, Artur Wolek  

**Link**: [PDF](https://arxiv.org/pdf/2511.07822)  

**Abstract**: This paper considers the problem of searching for a point of interest (POI) moving along an urban road network with an uncrewed aerial vehicle (UAV). The UAV is modeled as a variable-speed Dubins vehicle with a line-of-sight sensor in an urban environment that may occlude the sensor's view of the POI. A search strategy is proposed that exploits a probabilistic visibility volume (VV) to plan its future motion with iterative deepening $A^\ast$. The probabilistic VV is a time-varying three-dimensional representation of the sensing constraints for a particular distribution of the POI's state. To find the path most likely to view the POI, the planner uses a heuristic to optimistically estimate the probability of viewing the POI over a time horizon. The probabilistic VV is max-pooled to create a variable-timestep planner that reduces the search space and balances long-term and short-term planning. The proposed path planning method is compared to prior work with a Monte-Carlo simulation and is shown to outperform the baseline methods in cluttered environments when the UAV's sensor has a higher false alarm probability. 

**Abstract (ZH)**: 本文考虑了使用无人驾驶航空车辆（UAV）在城市道路网络中搜寻沿道路移动的兴趣点（POI）的问题。UAV被建模为具有视线传感器的可变速度杜宾车，在可能遮挡传感器视域的城市环境中对POI进行搜索。提出了一种策略，该策略利用概率可视体积（VV）并通过迭代加深$A^\ast$算法规划其未来的运动。概率可视体积是一个时间变化的三维表示，体现了特定POI状态分布下的传感约束。为了找到最有可能观察到POI的路径，规划器使用启发式方法乐观地估计在时间跨度内观察到POI的概率。概率可视体积通过最大池化创建了一个可变时间步长的规划器，从而减少搜索空间并平衡长期和短期规划。与之前的路径规划方法进行了 Monte-Carlo 模拟对比，并在UAV传感器具有较高误报概率的复杂环境中展示了所提出方法的优越性。 

---
# SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control 

**Title (ZH)**: SONIC: 扩展运动追踪以实现自然人形全身控制 

**Authors**: Zhengyi Luo, Ye Yuan, Tingwu Wang, Chenran Li, Sirui Chen, Fernando Castañeda, Zi-Ang Cao, Jiefeng Li, David Minor, Qingwei Ben, Xingye Da, Runyu Ding, Cyrus Hogg, Lina Song, Edy Lim, Eugene Jeong, Tairan He, Haoru Xue, Wenli Xiao, Zi Wang, Simon Yuen, Jan Kautz, Yan Chang, Umar Iqbal, Linxi "Jim" Fan, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07820)  

**Abstract**: Despite the rise of billion-parameter foundation models trained across thousands of GPUs, similar scaling gains have not been shown for humanoid control. Current neural controllers for humanoids remain modest in size, target a limited behavior set, and are trained on a handful of GPUs over several days. We show that scaling up model capacity, data, and compute yields a generalist humanoid controller capable of creating natural and robust whole-body movements. Specifically, we posit motion tracking as a natural and scalable task for humanoid control, leverageing dense supervision from diverse motion-capture data to acquire human motion priors without manual reward engineering. We build a foundation model for motion tracking by scaling along three axes: network size (from 1.2M to 42M parameters), dataset volume (over 100M frames, 700 hours of high-quality motion data), and compute (9k GPU hours). Beyond demonstrating the benefits of scale, we show the practical utility of our model through two mechanisms: (1) a real-time universal kinematic planner that bridges motion tracking to downstream task execution, enabling natural and interactive control, and (2) a unified token space that supports various motion input interfaces, such as VR teleoperation devices, human videos, and vision-language-action (VLA) models, all using the same policy. Scaling motion tracking exhibits favorable properties: performance improves steadily with increased compute and data diversity, and learned representations generalize to unseen motions, establishing motion tracking at scale as a practical foundation for humanoid control. 

**Abstract (ZH)**: 尽管亿级参数的基础模型在数千张GPU上训练后取得了显著的规模效应，但在类人控制领域尚未显示出类似的规模效益。当前的人类类神经控制器规模仍然较为有限，主要用于实现少量行为，并且在少量GPU上训练数天。我们展示了通过扩大模型容量、数据和计算资源，可以构建出通识类人控制模型，该模型能够创建自然且稳健的整体动作。具体而言，我们将动作追踪视为类人控制中的自然且可扩展的任务，通过多样化动作捕捉数据的密集监督，获取人类动作先验，避免手工奖励工程。我们通过在三个维度上扩大基础模型的规模，构建了动作追踪的通识模型：网络规模（从1.2M参数增至42M参数）、数据集规模（超过1亿帧，700小时的高质量动作数据）和计算资源（9000个GPU小时）。除了展示规模效应的优势外，我们还通过两种机制展示了模型的实际用途：（1）实时通用运动规划器，将动作追踪与下游任务执行连接起来，实现自然交互控制；（2）统一的令牌空间，支持多种运动输入接口，如VR远程操作设备、人类视频和视觉-语言-动作模型，使用相同的策略。动作追踪的扩展表现出良好的特性：性能随计算和数据多样性的增加而稳步提高，学习表示能够泛化到未见过的动作，从而确立动作追踪在类人控制领域的实用基础。 

---
# Virtual Traffic Lights for Multi-Robot Navigation: Decentralized Planning with Centralized Conflict Resolution 

**Title (ZH)**: 虚拟交通灯用于多机器人导航：去中心化规划与中心化冲突解决 

**Authors**: Sagar Gupta, Thanh Vinh Nguyen, Thieu Long Phan, Vidul Attri, Archit Gupta, Niroshinie Fernando, Kevin Lee, Seng W. Loke, Ronny Kutadinata, Benjamin Champion, Akansel Cosgun  

**Link**: [PDF](https://arxiv.org/pdf/2511.07811)  

**Abstract**: We present a hybrid multi-robot coordination framework that combines decentralized path planning with centralized conflict resolution. In our approach, each robot autonomously plans its path and shares this information with a centralized node. The centralized system detects potential conflicts and allows only one of the conflicting robots to proceed at a time, instructing others to stop outside the conflicting area to avoid deadlocks. Unlike traditional centralized planning methods, our system does not dictate robot paths but instead provides stop commands, functioning as a virtual traffic light. In simulation experiments with multiple robots, our approach increased the success rate of robots reaching their goals while reducing deadlocks. Furthermore, we successfully validated the system in real-world experiments with two quadruped robots and separately with wheeled Duckiebots. 

**Abstract (ZH)**: 一种结合分散路径规划与集中冲突解决的混合多机器人协调框架 

---
# Benchmarking Resilience and Sensitivity of Polyurethane-Based Vision-Based Tactile Sensors 

**Title (ZH)**: 基于聚氨酯的视觉触觉传感器的鲁棒性和敏感性基准测试 

**Authors**: Benjamin Davis, Hannah Stuart  

**Link**: [PDF](https://arxiv.org/pdf/2511.07797)  

**Abstract**: Vision-based tactile sensors (VBTSs) are a promising technology for robots, providing them with dense signals that can be translated into an understanding of normal and shear load, contact region, texture classification, and more. However, existing VBTS tactile surfaces make use of silicone gels, which provide high sensitivity but easily deteriorate from loading and surface wear. We propose that polyurethane rubber, used for high-load applications like shoe soles, rubber wheels, and industrial gaskets, may provide improved physical gel resilience, potentially at the cost of sensitivity. To compare the resilience and sensitivity of silicone and polyurethane VBTS gels, we propose a series of standard evaluation benchmarking protocols. Our resilience tests assess sensor durability across normal loading, shear loading, and abrasion. For sensitivity, we introduce model-free assessments of force and spatial sensitivity to directly measure the physical capabilities of each gel without effects introduced from data and model quality. Finally, we include a bottle cap loosening and tightening demonstration as an example where polyurethane gels provide an advantage over their silicone counterparts. 

**Abstract (ZH)**: 基于视觉的触觉传感器（VBTS）是机器人技术的一种有前景的技术，能够提供密集的信号以理解正压力、切向压力、接触区域、纹理分类等。然而，现有的VBTS触觉表面大多采用硅胶，虽然灵敏度高，但容易因负载和表面磨损而退化。我们提出使用聚氨酯橡胶作为材料，聚氨酯橡胶常用于鞋底、橡胶轮子和工业垫片等高负载应用，可能会提供更好的物理胶体韧性，尽管灵敏度可能会有所降低。为了比较硅胶和聚氨酯VBTS胶体的韧性与灵敏度，我们提出了系列标准评估基准测试协议。我们的韧性测试评估传感器在正压力、切向压力和磨损条件下的耐用性。对于灵敏度，我们引入了无模型评估方法，直接测量每种胶体的力学能力，不受数据质量和模型效果的影响。最后，我们通过瓶盖松紧试验示例展示了聚氨酯胶体相对于硅胶的优势。 

---
# High-Altitude Balloon Station-Keeping with First Order Model Predictive Control 

**Title (ZH)**: 高 altitude 气球站 Keeping  with 一阶模型预测控制 

**Authors**: Myles Pasetsky, Jiawei Lin, Bradley Guo, Sarah Dean  

**Link**: [PDF](https://arxiv.org/pdf/2511.07761)  

**Abstract**: High-altitude balloons (HABs) are common in scientific research due to their wide range of applications and low cost. Because of their nonlinear, underactuated dynamics and the partial observability of wind fields, prior work has largely relied on model-free reinforcement learning (RL) methods to design near-optimal control schemes for station-keeping. These methods often compare only against hand-crafted heuristics, dismissing model-based approaches as impractical given the system complexity and uncertain wind forecasts. We revisit this assumption about the efficacy of model-based control for station-keeping by developing First-Order Model Predictive Control (FOMPC). By implementing the wind and balloon dynamics as differentiable functions in JAX, we enable gradient-based trajectory optimization for online planning. FOMPC outperforms a state-of-the-art RL policy, achieving a 24% improvement in time-within-radius (TWR) without requiring offline training, though at the cost of greater online computation per control step. Through systematic ablations of modeling assumptions and control factors, we show that online planning is effective across many configurations, including under simplified wind and dynamics models. 

**Abstract (ZH)**: 高海拔气球通过其广泛的应用范围和低成本，在科学研究中十分常见。由于其非线性、欠驱动的动力学特性和风场的部分可观测性，先前的工作主要依赖无模型强化学习（RL）方法设计近最优的驻留控制方案。这些方法通常仅与手工设计的启发式方法进行比较，认为在给定系统复杂性和不确定的风速预报的情况下，基于模型的控制方法不切实际。我们通过开发一阶模型预测控制（FOMPC）重新审视了基于模型的控制方法在驻留控制中的有效性。通过在JAX中将风和气球动力学实现为可微函数，我们实现基于梯度的轨迹优化以进行在线规划。FOMPC在不需要离线训练的情况下，相较于最先进的RL策略，在时间在指定范围内的性能上取得了24%的提升，但每个控制步骤所需的在线计算量有所增加。通过对建模假设和控制因素的系统性消融研究，我们展示了在线规划在多种配置下均有效，包括在简化后的风速和动力学模型下。 

---
# Navigating the Wild: Pareto-Optimal Visual Decision-Making in Image Space 

**Title (ZH)**: 穿越混沌：图像空间中的帕累托最优视觉决策 

**Authors**: Durgakant Pushp, Weizhe Chen, Zheng Chen, Chaomin Luo, Jason M. Gregory, Lantao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07750)  

**Abstract**: Navigating complex real-world environments requires semantic understanding and adaptive decision-making. Traditional reactive methods without maps often fail in cluttered settings, map-based approaches demand heavy mapping effort, and learning-based solutions rely on large datasets with limited generalization. To address these challenges, we present Pareto-Optimal Visual Navigation, a lightweight image-space framework that combines data-driven semantics, Pareto-optimal decision-making, and visual servoing for real-time navigation. 

**Abstract (ZH)**: Pareto-Optimal视觉导航：一种结合数据驱动语义、帕累托优化决策和视觉伺服的轻量级图像空间框架 

---
# ViPRA: Video Prediction for Robot Actions 

**Title (ZH)**: ViPRA：用于机器人动作的视频预测 

**Authors**: Sandeep Routray, Hengkai Pan, Unnat Jain, Shikhar Bahl, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2511.07732)  

**Abstract**: Can we turn a video prediction model into a robot policy? Videos, including those of humans or teleoperated robots, capture rich physical interactions. However, most of them lack labeled actions, which limits their use in robot learning. We present Video Prediction for Robot Actions (ViPRA), a simple pretraining-finetuning framework that learns continuous robot control from these actionless videos. Instead of directly predicting actions, we train a video-language model to predict both future visual observations and motion-centric latent actions, which serve as intermediate representations of scene dynamics. We train these latent actions using perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior. For downstream control, we introduce a chunked flow matching decoder that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations. This approach avoids expensive action annotation, supports generalization across embodiments, and enables smooth, high-frequency continuous control upto 22 Hz via chunked action decoding. Unlike prior latent action works that treat pretraining as autoregressive policy learning, explicitly models both what changes and how. Our method outperforms strong baselines, with a 16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks. We will release models and code at this https URL 

**Abstract (ZH)**: 可以通过视频预测模型构建机器人策略吗？视频，包括人类或遥控机器人的视频，捕捉到丰富的物理交互。然而，大多数缺乏标注动作，这限制了它们在机器人学习中的应用。我们提出了一种名为Video Prediction for Robot Actions (ViPRA) 的简单预训练-微调框架，能够从这些无动作视频中学习连续的机器人控制。我们不是直接预测动作，而是训练一个视频-语言模型来预测未来的视觉观察和以运动为中心的潜在动作，这些潜在动作作为场景动态的中介表示。我们使用感知损失和光流一致性来训练这些潜在动作，以确保它们反映物理上合理的行为。在下游控制方面，我们引入了一种分块光流匹配解码器，能够将潜在动作映射到特定于机器人的连续动作序列，仅需100至200个遥控演示。此方法避免了昂贵的动作标注，支持跨实体的一般化，并通过分块动作解码实现了高达22 Hz的平滑、高频连续控制。与先前将预训练视为自回归策略学习的方法不同，我们的方法明确建模了“什么变化”和“如何变化”。我们的方法在SIMPLER基准上优于强基线，表现提升了16%，并在实际世界操作任务中整体提高了13%。我们将在该网址发布模型和代码：[这里插入链接]。 

---
# LLM-GROP: Visually Grounded Robot Task and Motion Planning with Large Language Models 

**Title (ZH)**: LLM-GROP：基于视觉的机器人任务与运动规划 

**Authors**: Xiaohan Zhang, Yan Ding, Yohei Hayamizu, Zainab Altaweel, Yifeng Zhu, Yuke Zhu, Peter Stone, Chris Paxton, Shiqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.07727)  

**Abstract**: Task planning and motion planning are two of the most important problems in robotics, where task planning methods help robots achieve high-level goals and motion planning methods maintain low-level feasibility. Task and motion planning (TAMP) methods interleave the two processes of task planning and motion planning to ensure goal achievement and motion feasibility. Within the TAMP context, we are concerned with the mobile manipulation (MoMa) of multiple objects, where it is necessary to interleave actions for navigation and manipulation.
In particular, we aim to compute where and how each object should be placed given underspecified goals, such as ``set up dinner table with a fork, knife and plate.'' We leverage the rich common sense knowledge from large language models (LLMs), e.g., about how tableware is organized, to facilitate both task-level and motion-level planning. In addition, we use computer vision methods to learn a strategy for selecting base positions to facilitate MoMa behaviors, where the base position corresponds to the robot's ``footprint'' and orientation in its operating space. Altogether, this article provides a principled TAMP framework for MoMa tasks that accounts for common sense about object rearrangement and is adaptive to novel situations that include many objects that need to be moved. We performed quantitative experiments in both real-world settings and simulated environments. We evaluated the success rate and efficiency in completing long-horizon object rearrangement tasks. While the robot completed 84.4\% real-world object rearrangement trials, subjective human evaluations indicated that the robot's performance is still lower than experienced human waiters. 

**Abstract (ZH)**: 基于任务与运动规划的方法进行多重操作 manipulatioN (MoMa) 的规划：利用大规模语言模型的常识知识 

---
# A QP Framework for Improving Data Collection: Quantifying Device-Controller Performance in Robot Teleoperation 

**Title (ZH)**: 一种QP框架以改进数据采集：量化机器人远程操作中设备-控制器性能 

**Authors**: Yuxuan Zhao, Yuanchen Tang, Jindi Zhang, Hongyu Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.07720)  

**Abstract**: Robot learning empowers the robot system with human brain-like intelligence to autonomously acquire and adapt skills through experience, enhancing flexibility and adaptability in various environments. Aimed at achieving a similar level of capability in large language models (LLMs) for embodied intelligence, data quality plays a crucial role in training a foundational model with diverse robot skills. In this study, we investigate the collection of data for manipulation tasks using teleoperation devices. Different devices yield varying effects when paired with corresponding controller strategies, including position-based inverse kinematics (IK) control, torque-based inverse dynamics (ID) control, and optimization-based compliance control. In this paper, we develop a teleoperation pipeline that is compatible with different teleoperation devices and manipulator controllers. Within the pipeline, we construct the optimal QP formulation with the dynamic nullspace and the impedance tracking as the novel optimal controller to achieve compliant pose tracking and singularity avoidance. Regarding the optimal controller, it adaptively adjusts the weights assignment depending on the robot joint manipulability that reflects the state of joint configuration for the pose tracking in the form of impedance control and singularity avoidance with nullspace tracking. Analysis of quantitative experimental results suggests the quality of the teleoperated trajectory data, including tracking error, occurrence of singularity, and the smoothness of the joints' trajectory, with different combinations of teleoperation interface and the motion controller. 

**Abstract (ZH)**: 机器人学习赋予机器人系统类似人脑的智能，通过经验自主获取和适应技能，增强在各种环境中的灵活性和适应性。为了在具备大规模语言模型（LLMs）类似水平的体现智能时，数据质量在训练具有多种机器人技能的基础模型中起着关键作用。在本研究中，我们探讨了使用遥操作设备收集操作任务数据的方法。不同的设备与相应的控制器策略相配时会产生不同的效果，包括基于位置的逆动力学（IK）控制、基于扭矩的逆动力学（ID）控制和基于优化的顺应性控制。在本文中，我们开发了一种兼容不同遥操作设备和操作器控制器的遥操作管道。在该管道中，我们构建了结合动态零空间和阻抗跟踪的新颖最优控制器，以实现顺应性姿态跟踪和奇异点避免。关于最优控制器，它根据反映姿态跟踪状态下关节配置状态的关节可操作性，自适应调整权重分配，以进行阻抗控制和奇异点避免的零空间跟踪。定量实验结果的分析表明，不同的遥操作界面和动作控制器组合对于遥操作系统轨迹数据的质量，包括跟踪误差、奇异点的出现次数和关节轨迹的平滑度的影响。 

---
# RoboTAG: End-to-end Robot Configuration Estimation via Topological Alignment Graph 

**Title (ZH)**: RoboTAG：基于拓扑对齐图的端到端机器人配置估计 

**Authors**: Yifan Liu, Fangneng Zhan, Wanhua Li, Haowen Sun, Katerina Fragkiadaki, Hanspeter Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2511.07717)  

**Abstract**: Estimating robot pose from a monocular RGB image is a challenge in robotics and computer vision. Existing methods typically build networks on top of 2D visual backbones and depend heavily on labeled data for training, which is often scarce in real-world scenarios, causing a sim-to-real gap. Moreover, these approaches reduce the 3D-based problem to 2D domain, neglecting the 3D priors. To address these, we propose Robot Topological Alignment Graph (RoboTAG), which incorporates a 3D branch to inject 3D priors while enabling co-evolution of the 2D and 3D representations, alleviating the reliance on labels. Specifically, the RoboTAG consists of a 3D branch and a 2D branch, where nodes represent the states of the camera and robot system, and edges capture the dependencies between these variables or denote alignments between them. Closed loops are then defined in the graph, on which a consistency supervision across branches can be applied. This design allows us to utilize in-the-wild images as training data without annotations. Experimental results demonstrate that our method is effective across robot types, highlighting its potential to alleviate the data bottleneck in robotics. 

**Abstract (ZH)**: 从单目RGB图像估计机器人姿态是机器人技术和计算机视觉领域的一项挑战。现有的方法通常基于2D视觉骨干构建网络，并且在训练时高度依赖标记数据，而在实际场景中，标记数据往往稀缺，导致模拟到现实的差距。此外，这些方法将基于3D的问题简化为2D领域，忽略了3D先验知识。为了解决这些问题，我们提出了机器人拓扑对齐图（RoboTAG），该方法引入3D分支以注入3D先验知识，并使2D和3D表示共同进化，减少对标签的依赖。具体而言，RoboTAG由3D分支和2D分支组成，节点代表相机和机器人系统的状态，边捕捉这些变量之间的依赖关系或表示它们之间的对齐。然后在图中定义闭合回路，在分支之间可以应用一致性监督。这种设计使我们能够利用未标记的图像作为训练数据。实验结果表明，我们的方法在不同类型的机器人上都有效，突显了其在机器人领域缓解数据瓶颈的潜力。 

---
# Testing and Evaluation of Underwater Vehicle Using Hardware-In-The-Loop Simulation with HoloOcean 

**Title (ZH)**: 基于HoloOcean的硬件在环仿真的水下车辆测试与评估 

**Authors**: Braden Meyers, Joshua G. Mangelson  

**Link**: [PDF](https://arxiv.org/pdf/2511.07687)  

**Abstract**: Testing marine robotics systems in controlled environments before field tests is challenging, especially when acoustic-based sensors and control surfaces only function properly underwater. Deploying robots in indoor tanks and pools often faces space constraints that complicate testing of control, navigation, and perception algorithms at scale. Recent developments of high-fidelity underwater simulation tools have the potential to address these problems. We demonstrate the utility of the recently released HoloOcean 2.0 simulator with improved dynamics for torpedo AUV vehicles and a new ROS 2 interface. We have successfully demonstrated a Hardware-in-the-Loop (HIL) and Software-in-the-Loop (SIL) setup for testing and evaluating a CougUV torpedo autonomous underwater vehicle (AUV) that was built and developed in our lab. With this HIL and SIL setup, simulations are run in HoloOcean using a ROS 2 bridge such that simulated sensor data is sent to the CougUV (mimicking sensor drivers) and control surface commands are sent back to the simulation, where vehicle dynamics and sensor data are calculated. We compare our simulated results to real-world field trial results. 

**Abstract (ZH)**: 在可控环境中测试基于声学传感器和控制面的水下机器人系统在野外试验前的挑战性测试，尤其是在只有在水下这些传感器和控制面才能正常工作的情况下。将机器人部署在室内水箱和游泳池中通常会受到空间限制，这使得大规模测试控制、导航和感知算法变得复杂。最近开发的高保真水下模拟工具有可能解决这些问题。我们展示了最近发布的HoloOcean 2.0仿真器及其改进的鱼雷AUV动力学模型和新的ROS 2接口的实用性。我们成功地为在我们实验室设计和开发的CougUV鱼雷自主水下车辆（AUV）建立了一个硬件在环（HIL）和软件在环（SIL）测试与评估设置。通过这种方式，在HoloOcean中使用ROS 2桥运行仿真，发送模拟传感器数据到CougUV（模拟传感器驱动程序）并与模拟器互动，发送回控制面指令，在此过程中计算车辆动力学和传感器数据。我们将我们的模拟结果与实际野外试验结果进行了比较。 

---
# Time-Aware Policy Learning for Adaptive and Punctual Robot Control 

**Title (ZH)**: 时间感知策略学习以实现适应性和及时的机器人控制 

**Authors**: Yinsen Jia, Boyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.07654)  

**Abstract**: Temporal awareness underlies intelligent behavior in both animals and humans, guiding how actions are sequenced, paced, and adapted to changing goals and environments. Yet most robot learning algorithms remain blind to time. We introduce time-aware policy learning, a reinforcement learning framework that enables robots to explicitly perceive and reason with time as a first-class variable. The framework augments conventional reinforcement policies with two complementary temporal signals, the remaining time and a time ratio, which allow a single policy to modulate its behavior continuously from rapid and dynamic to cautious and precise execution. By jointly optimizing punctuality and stability, the robot learns to balance efficiency, robustness, resiliency, and punctuality without re-training or reward adjustment. Across diverse manipulation domains from long-horizon pick and place, to granular-media pouring, articulated-object handling, and multi-agent object delivery, the time-aware policy produces adaptive behaviors that outperform standard reinforcement learning baselines by up to 48% in efficiency, 8 times more robust in sim-to-real transfer, and 90% in acoustic quietness while maintaining near-perfect success rates. Explicit temporal reasoning further enables real-time human-in-the-loop control and multi-agent coordination, allowing robots to recover from disturbances, re-synchronize after delays, and align motion tempo with human intent. By treating time not as a constraint but as a controllable dimension of behavior, time-aware policy learning provides a unified foundation for efficient, robust, resilient, and human-aligned robot autonomy. 

**Abstract (ZH)**: 时间意识是动物和人类智能行为的基础，指导动作的顺序、节奏及其在目标和环境变化时的适应。然而，大多数机器人学习算法仍无视时间。我们提出了时间意识策略学习，这是一种强化学习框架，使机器人能够明确感知和利用时间作为首要变量。该框架通过添加两个互补的时间信号——剩余时间和时间比例——扩展了传统强化策略，使单一策略能够从快速动态到谨慎精确地连续调节其行为。通过同时优化准时性和稳定性，机器人能够在无需重新训练或调整奖励的情况下学习平衡效率、鲁棒性、恢复能力和准时性。在从长时序拾放、颗粒介质倾倒、柔性物体处理到多机器人物体传递等多个操作领域中，时间意识策略产生了超越标准强化学习基线的行为，在效率上高出最多48%，在从仿真到现实的转移鲁棒性上高出8倍，在声学安静性上高出90%的同时保持接近完美的成功率。明确的时间推理还 enables 实时人机闭环控制和多机器人协调，使机器人能够从干扰中恢复、在延误后重新同步，并与人类意图同步动作节奏。通过将时间视为行为的可控维度而非约束，时间意识策略学习为高效、鲁棒、恢复和与人类对齐的机器人自主性提供了一个统一的基础。 

---
# CAVER: Curious Audiovisual Exploring Robot 

**Title (ZH)**: 好奇听觉视觉探索机器人 

**Authors**: Luca Macesanu, Boueny Folefack, Samik Singh, Ruchira Ray, Ben Abbatematteo, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2511.07619)  

**Abstract**: Multimodal audiovisual perception can enable new avenues for robotic manipulation, from better material classification to the imitation of demonstrations for which only audio signals are available (e.g., playing a tune by ear). However, to unlock such multimodal potential, robots need to learn the correlations between an object's visual appearance and the sound it generates when they interact with it. Such an active sensorimotor experience requires new interaction capabilities, representations, and exploration methods to guide the robot in efficiently building increasingly rich audiovisual knowledge. In this work, we present CAVER, a novel robot that builds and utilizes rich audiovisual representations of objects. CAVER includes three novel contributions: 1) a novel 3D printed end-effector, attachable to parallel grippers, that excites objects' audio responses, 2) an audiovisual representation that combines local and global appearance information with sound features, and 3) an exploration algorithm that uses and builds the audiovisual representation in a curiosity-driven manner that prioritizes interacting with high uncertainty objects to obtain good coverage of surprising audio with fewer interactions. We demonstrate that CAVER builds rich representations in different scenarios more efficiently than several exploration baselines, and that the learned audiovisual representation leads to significant improvements in material classification and the imitation of audio-only human demonstrations. this https URL 

**Abstract (ZH)**: 多模态音频视觉感知可以为机器人操作开辟新途径，从更好的材料分类到模仿仅通过音频信号（例如，吹奏曲调）演示的示范。然而，为了释放这种多模态潜力，机器人需要学习物体的视觉外观与其互动时产生的声音之间的关联。这种主动的感觉运动经验需要新的交互能力、表示方法和探索方法，以指导机器人高效地构建日益丰富的音频视觉知识。在这项工作中，我们提出了一种名为CAVER的新机器人，用于构建和利用丰富的音频视觉表示。CAVER包括三个新颖贡献：1）一种可以连接到并行夹爪的新颖3D打印末端执行器，用于激发物体的音频响应；2）结合局部和全局视觉信息与声音特征的音频视觉表示；3）一种好奇心驱动的探索算法，该算法利用并构建音频视觉表示，优先与不确定性高的物体互动，以较少的交互获得更多的音频惊讶覆盖。我们展示了CAVER在不同场景中比几种探索基线更有效地构建丰富表示，并且学习到的音频视觉表示在材料分类和模仿仅含音频的人类示范方面带来了显着改进。 

---
# Probabilistic Safety Guarantee for Stochastic Control Systems Using Average Reward MDPs 

**Title (ZH)**: 使用平均奖励MDP的随机控制系统概率安全保证 

**Authors**: Saber Omidi, Marek Petrik, Se Young Yoon, Momotaz Begum  

**Link**: [PDF](https://arxiv.org/pdf/2511.08419)  

**Abstract**: Safety in stochastic control systems, which are subject to random noise with a known probability distribution, aims to compute policies that satisfy predefined operational constraints with high confidence throughout the uncertain evolution of the state variables. The unpredictable evolution of state variables poses a significant challenge for meeting predefined constraints using various control methods. To address this, we present a new algorithm that computes safe policies to determine the safety level across a finite state set. This algorithm reduces the safety objective to the standard average reward Markov Decision Process (MDP) objective. This reduction enables us to use standard techniques, such as linear programs, to compute and analyze safe policies. We validate the proposed method numerically on the Double Integrator and the Inverted Pendulum systems. Results indicate that the average-reward MDPs solution is more comprehensive, converges faster, and offers higher quality compared to the minimum discounted-reward solution. 

**Abstract (ZH)**: 具有已知概率分布的随机噪声影响下的随机控制系统的安全性旨在计算能够在不确定性状态下满足预定义操作约束的策略。状态变量的不可预测演化对使用各种控制方法满足预定义约束构成重大挑战。为此，我们提出了一种新算法，计算安全策略以确定有限状态集的安全级别。该算法将安全性目标转化为标准的平均奖励马尔可夫决策过程（MDP）目标。这种转换使我们能够利用线性规划等标准技术来计算和分析安全策略。我们在双积分器和倒立摆系统上通过数值验证了所提出的方法。结果表明，平均奖励MDP的解决方案更为全面、收敛更快且质量更高，优于最小折现奖励解决方案。 

---
# Work-in-Progress: Function-as-Subtask API Replacing Publish/Subscribe for OS-Native DAG Scheduling 

**Title (ZH)**: 工作进展：将函数作为子任务API替换发布/订阅进行OS原生DAG调度 

**Authors**: Takahiro Ishikawa-Aso, Atsushi Yano, Yutaro Kobayashi, Takumi Jin, Yuuki Takano, Shinpei Kato  

**Link**: [PDF](https://arxiv.org/pdf/2511.08297)  

**Abstract**: The Directed Acyclic Graph (DAG) task model for real-time scheduling finds its primary practical target in Robot Operating System 2 (ROS 2). However, ROS 2's publish/subscribe API leaves DAG precedence constraints unenforced: a callback may publish mid-execution, and multi-input callbacks let developers choose topic-matching policies. Thus preserving DAG semantics relies on conventions; once violated, the model collapses. We propose the Function-as-Subtask (FasS) API, which expresses each subtask as a function whose arguments/return values are the subtask's incoming/outgoing edges. By minimizing description freedom, DAG semantics is guaranteed at the API rather than by programmer discipline. We implement a DAG-native scheduler using FasS on a Rust-based experimental kernel and evaluate its semantic fidelity, and we outline design guidelines for applying FasS to Linux Linux sched_ext. 

**Abstract (ZH)**: 面向实时调度的有向无环图（DAG）任务模型在Robot Operating System 2（ROS 2）中找到了主要的实际目标。然而，ROS 2的发布/订阅API未能强制执行DAG的优先级约束：回调函数可能在执行过程中发布数据，多输入回调函数允许开发人员选择主题匹配策略。因此，保持DAG语义依赖于约定；一旦被违反，模型将失效。我们提出了一种Function-as-Subtask（FasS）API，通过将每个子任务表示为一个函数，其参数/返回值为子任务的入边/出边，从而最小化描述自由度，在API层面而非开发人员的自律保证DAG语义。我们使用基于Rust的实验内核实现了DAG原生调度器，并评估了其语义准确性，并概述了将FasS应用于Linux sched_ext的设计指南。 

---
# Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks 

**Title (ZH)**: 动态稀疏性：机器人强化学习基准中学习世界模型的常见稀疏性假设挑战 

**Authors**: Muthukumar Pandaram, Jakob Hollenstein, David Drexel, Samuele Tosatto, Antonio Rodríguez-Sánchez, Justus Piater  

**Link**: [PDF](https://arxiv.org/pdf/2511.08086)  

**Abstract**: The use of learned dynamics models, also known as world models, can improve the sample efficiency of reinforcement learning. Recent work suggests that the underlying causal graphs of such dynamics models are sparsely connected, with each of the future state variables depending only on a small subset of the current state variables, and that learning may therefore benefit from sparsity priors. Similarly, temporal sparsity, i.e. sparsely and abruptly changing local dynamics, has also been proposed as a useful inductive bias.
In this work, we critically examine these assumptions by analyzing ground-truth dynamics from a set of robotic reinforcement learning environments in the MuJoCo Playground benchmark suite, aiming to determine whether the proposed notions of state and temporal sparsity actually tend to hold in typical reinforcement learning tasks.
We study (i) whether the causal graphs of environment dynamics are sparse, (ii) whether such sparsity is state-dependent, and (iii) whether local system dynamics change sparsely.
Our results indicate that global sparsity is rare, but instead the tasks show local, state-dependent sparsity in their dynamics and this sparsity exhibits distinct structures, appearing in temporally localized clusters (e.g., during contact events) and affecting specific subsets of state dimensions. These findings challenge common sparsity prior assumptions in dynamics learning, emphasizing the need for grounded inductive biases that reflect the state-dependent sparsity structure of real-world dynamics. 

**Abstract (ZH)**: 使用学习到的动力学模型可以提高强化学习的样本效率。近年来的研究表明，此类动力学模型的潜在因果图连接稀疏，每个未来的状态变量仅依赖于当前状态变量的子集，因此学习可以从稀疏先验中受益。类似地，时间稀疏性，即局部动力学稀疏且突然变化，也被提出作为一种有用的归纳偏置。

在本文中，我们通过分析MuJoCo Playground基准套件中一组机器人强化学习环境的真实动力学，批判性地检查这些假设，旨在确定提出的状态和时间稀疏性概念是否在典型的强化学习任务中实际上常见。

我们研究了(i) 环境动力学的因果图是否稀疏，(ii) 这种稀疏性是否状态依赖，以及(iii) 局部系统动力学是否稀疏变化。

结果表明，全局稀疏性罕见，但任务在其动力学中表现出局部、状态依赖的稀疏性，这种稀疏性呈现出特定的结构，在时间上局部化（例如，在接触事件期间）并影响特定的状态维度子集。这些发现挑战了动力学学习中的常见稀疏性先验假设，强调需要反映真实世界动力学的状态依赖稀疏性结构的坚实归纳偏置。 

---
# An Image-Based Path Planning Algorithm Using a UAV Equipped with Stereo Vision 

**Title (ZH)**: 基于立体视觉 UAV 的图像引导路径规划算法 

**Authors**: Selim Ahmet Iz, Mustafa Unel  

**Link**: [PDF](https://arxiv.org/pdf/2511.07928)  

**Abstract**: This paper presents a novel image-based path planning algorithm that was developed using computer vision techniques, as well as its comparative analysis with well-known deterministic and probabilistic algorithms, namely A* and Probabilistic Road Map algorithm (PRM). The terrain depth has a significant impact on the calculated path safety. The craters and hills on the surface cannot be distinguished in a two-dimensional image. The proposed method uses a disparity map of the terrain that is generated by using a UAV. Several computer vision techniques, including edge, line and corner detection methods, as well as the stereo depth reconstruction technique, are applied to the captured images and the found disparity map is used to define candidate way-points of the trajectory. The initial and desired points are detected automatically using ArUco marker pose estimation and circle detection techniques. After presenting the mathematical model and vision techniques, the developed algorithm is compared with well-known algorithms on different virtual scenes created in the V-REP simulation program and a physical setup created in a laboratory environment. Results are promising and demonstrate effectiveness of the proposed algorithm. 

**Abstract (ZH)**: 基于计算机视觉的新型图像导向路径规划算法及其与A*和概率路网算法的比较分析 

---
# Statistically Assuring Safety of Control Systems using Ensembles of Safety Filters and Conformal Prediction 

**Title (ZH)**: 使用安全过滤器ensemble和区间预测保证控制系统的安全性 

**Authors**: Ihab Tabbara, Yuxuan Yang, Hussein Sibai  

**Link**: [PDF](https://arxiv.org/pdf/2511.07899)  

**Abstract**: Safety assurance is a fundamental requirement for deploying learning-enabled autonomous systems. Hamilton-Jacobi (HJ) reachability analysis is a fundamental method for formally verifying safety and generating safe controllers. However, computing the HJ value function that characterizes the backward reachable set (BRS) of a set of user-defined failure states is computationally expensive, especially for high-dimensional systems, motivating the use of reinforcement learning approaches to approximate the value function. Unfortunately, a learned value function and its corresponding safe policy are not guaranteed to be correct. The learned value function evaluated at a given state may not be equal to the actual safety return achieved by following the learned safe policy. To address this challenge, we introduce a conformal prediction-based (CP) framework that bounds such uncertainty. We leverage CP to provide probabilistic safety guarantees when using learned HJ value functions and policies to prevent control systems from reaching failure states. Specifically, we use CP to calibrate the switching between the unsafe nominal controller and the learned HJ-based safe policy and to derive safety guarantees under this switched policy. We also investigate using an ensemble of independently trained HJ value functions as a safety filter and compare this ensemble approach to using individual value functions alone. 

**Abstract (ZH)**: 基于一致性预测的HJ值函数和策略不确定性边界框架：用于自主系统的安全性保证 

---
# Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning 

**Title (ZH)**: 多步准度量学习以实现可扩展的目标条件强化学习 

**Authors**: Bill Chunyuan Zheng, Vivek Myers, Benjamin Eysenbach, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2511.07730)  

**Abstract**: Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations. 

**Abstract (ZH)**: 在一个环境中学习如何达成目标是AI领域长期面临的一项挑战，尽管如此，对远期进行推理仍然是现代方法的难题。关键问题是如何估计观察对之间的时序距离。虽然时差方法通过局部更新来提供最优性保证，但它们通常在多步回报的蒙特卡洛方法（例如，多步回报）表现更差，后者缺乏这种保证。我们展示了如何将这些方法整合进一个实用的GCRL方法中，使用多步蒙特卡洛回报拟合准度量距离。我们表明，我们的方法在具有最多4000步的长时 horizons 定向仿真任务中优于现有的GCRL方法，即使在具有视觉观察的情况下也是如此。我们还展示了我们的方法如何在真实世界的机器人操作领域（Bridge setup）中实现拼接。我们的方法是第一个能够从未标注的离线视觉观察数据集中实现多步拼接的完整的GCRL方法，应用于这个真实世界的操作领域。 

---
# ARGUS: A Framework for Risk-Aware Path Planning in Tactical UGV Operations 

**Title (ZH)**: ARGUS：战术UGV操作中风险感知路径规划的框架 

**Authors**: Nuno Soares, António Grilo  

**Link**: [PDF](https://arxiv.org/pdf/2511.07565)  

**Abstract**: This thesis presents the development of ARGUS, a framework for mission planning for Unmanned Ground Vehicles (UGVs) in tactical environments. The system is designed to translate battlefield complexity and the commander's intent into executable action plans. To this end, ARGUS employs a processing pipeline that takes as input geospatial terrain data, military intelligence on existing threats and their probable locations, and mission priorities defined by the commander. Through a set of integrated modules, the framework processes this information to generate optimized trajectories that balance mission objectives against the risks posed by threats and terrain characteristics. A fundamental capability of ARGUS is its dynamic nature, which allows it to adapt plans in real-time in response to unforeseen events, reflecting the fluid nature of the modern battlefield. The system's interoperability were validated in a practical exercise with the Portuguese Army, where it was successfully demonstrated that the routes generated by the model can be integrated and utilized by UGV control systems. The result is a decision support tool that not only produces an optimal trajectory but also provides the necessary insights for its execution, thereby contributing to greater effectiveness and safety in the employment of autonomous ground systems. 

**Abstract (ZH)**: ARGUS：战术环境中无人地面车辆任务规划框架的发展 

---
