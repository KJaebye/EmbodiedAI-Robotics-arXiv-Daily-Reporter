# HCRMP: A LLM-Hinted Contextual Reinforcement Learning Framework for Autonomous Driving 

**Title (ZH)**: HCRMP：一种基于LLM的上下文强化学习自主驾驶框架 

**Authors**: Zhiwen Chen, Bo Leng, Zhuoren Li, Hanming Deng, Guizhe Jin, Ran Yu, Huanxi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15793)  

**Abstract**: Integrating Large Language Models (LLMs) with Reinforcement Learning (RL) can enhance autonomous driving (AD) performance in complex scenarios. However, current LLM-Dominated RL methods over-rely on LLM outputs, which are prone to this http URL show that state-of-the-art LLM indicates a non-hallucination rate of only approximately 57.95% when assessed on essential driving-related tasks. Thus, in these methods, hallucinations from the LLM can directly jeopardize the performance of driving policies. This paper argues that maintaining relative independence between the LLM and the RL is vital for solving the hallucinations problem. Consequently, this paper is devoted to propose a novel LLM-Hinted RL paradigm. The LLM is used to generate semantic hints for state augmentation and policy optimization to assist RL agent in motion planning, while the RL agent counteracts potential erroneous semantic indications through policy learning to achieve excellent driving performance. Based on this paradigm, we propose the HCRMP (LLM-Hinted Contextual Reinforcement Learning Motion Planner) architecture, which is designed that includes Augmented Semantic Representation Module to extend state space. Contextual Stability Anchor Module enhances the reliability of multi-critic weight hints by utilizing information from the knowledge base. Semantic Cache Module is employed to seamlessly integrate LLM low-frequency guidance with RL high-frequency control. Extensive experiments in CARLA validate HCRMP's strong overall driving performance. HCRMP achieves a task success rate of up to 80.3% under diverse driving conditions with different traffic densities. Under safety-critical driving conditions, HCRMP significantly reduces the collision rate by 11.4%, which effectively improves the driving performance in complex scenarios. 

**Abstract (ZH)**: 将大型语言模型（LLM）与强化学习（RL）结合可以增强自动驾驶（AD）在复杂场景中的性能。然而，当前的LLM主导的RL方法过度依赖LLM的输出，这些输出易产生幻觉。研究表明，最先进的LLM在关键驾驶任务上的无幻觉率仅为约57.95%。因此，在这些方法中，LLM的幻觉会直接危及驾驶策略的性能。本文论点认为，保持LLM与RL相对独立对于解决幻觉问题至关重要。因此，本文提出了一种新颖的LLM提示RL范式。LLM用于生成语义提示，以增强状态并优化策略，辅助RL代理进行运动规划，而RL代理则通过策略学习抵消潜在的错误语义指示，以实现优秀的驾驶性能。基于这一范式，我们提出了HCRMP（LLM提示的上下文强化学习运动规划器）架构，该架构设计包括扩展状态空间的增强语义表示模块，利用知识库信息增强多批评注权重提示的可靠性，并通过语义缓存模块无缝集成LLM低频指导与RL高频控制。在CARLA中进行的大量实验验证了HCRMP在各种驾驶条件下的强大整体驾驶性能。在不同交通密度的多种驾驶条件下，HCRMP实现了高达80.3%的任务成功率。在关键驾驶条件下，HCRMP显著降低了碰撞率11.4%，从而在复杂场景中有效地提升了驾驶性能。 

---
# UAV-Flow Colosseo: A Real-World Benchmark for Flying-on-a-Word UAV Imitation Learning 

**Title (ZH)**: UAV-Flow Colosseo：一个真实世界无人飞行器模仿学习基准 

**Authors**: Xiangyu Wang, Donglin Yang, Yue Liao, Wenhao Zheng, wenjun wu, Bin Dai, Hongsheng Li, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15725)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are evolving into language-interactive platforms, enabling more intuitive forms of human-drone interaction. While prior works have primarily focused on high-level planning and long-horizon navigation, we shift attention to language-guided fine-grained trajectory control, where UAVs execute short-range, reactive flight behaviors in response to language instructions. We formalize this problem as the Flying-on-a-Word (Flow) task and introduce UAV imitation learning as an effective approach. In this framework, UAVs learn fine-grained control policies by mimicking expert pilot trajectories paired with atomic language instructions. To support this paradigm, we present UAV-Flow, the first real-world benchmark for language-conditioned, fine-grained UAV control. It includes a task formulation, a large-scale dataset collected in diverse environments, a deployable control framework, and a simulation suite for systematic evaluation. Our design enables UAVs to closely imitate the precise, expert-level flight trajectories of human pilots and supports direct deployment without sim-to-real gap. We conduct extensive experiments on UAV-Flow, benchmarking VLN and VLA paradigms. Results show that VLA models are superior to VLN baselines and highlight the critical role of spatial grounding in the fine-grained Flow setting. 

**Abstract (ZH)**: 无人驾驶飞行器（UAVs）正在演变为语言互动平台， enables 更直观的人机无人机交互形式。在以往的研究主要集中在高层规划和远期导航的基础上，我们将注意力转向由语言指导的细粒度轨迹控制任务，其中无人机响应语言指令执行短距离、反应式飞行行为。我们将这一问题形式化为“单词飞行”（Flow）任务，并引入无人机模仿学习作为有效的解决方案。在此框架下，无人机通过模仿配对有原子语言指令的专家飞行员轨迹来学习细粒度控制策略。为了支持这一范式，我们提出了UAV-Flow，这是第一个用于语言条件下的细粒度无人机控制的真实世界基准数据集，包括任务定义、多样化环境下的大规模数据集、可部署的控制框架和用于系统性评估的模拟套件。我们的设计使无人机能够精确地模仿人类飞行员的精确飞行轨迹，并支持直接部署而无需进行模拟到现实的过渡。我们在UAV-Flow上进行了广泛的实验，基于视觉语言导航（VLN）和视觉语言动作（VLA）基准进行评估。结果表明，VLA模型优于VLN基准，并突显了在细粒度Flow设置中空间定位的关键作用。 

---
# From Grounding to Manipulation: Case Studies of Foundation Model Integration in Embodied Robotic Systems 

**Title (ZH)**: 从语义grounding到操作操纵：基础模型在具身机器人系统中的集成案例研究 

**Authors**: Xiuchao Sui, Daiying Tian, Qi Sun, Ruirui Chen, Dongkyu Choi, Kenneth Kwok, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2505.15685)  

**Abstract**: Foundation models (FMs) are increasingly used to bridge language and action in embodied agents, yet the operational characteristics of different FM integration strategies remain under-explored -- particularly for complex instruction following and versatile action generation in changing environments. This paper examines three paradigms for building robotic systems: end-to-end vision-language-action (VLA) models that implicitly integrate perception and planning, and modular pipelines incorporating either vision-language models (VLMs) or multimodal large language models (LLMs). We evaluate these paradigms through two focused case studies: a complex instruction grounding task assessing fine-grained instruction understanding and cross-modal disambiguation, and an object manipulation task targeting skill transfer via VLA finetuning. Our experiments in zero-shot and few-shot settings reveal trade-offs in generalization and data efficiency. By exploring performance limits, we distill design implications for developing language-driven physical agents and outline emerging challenges and opportunities for FM-powered robotics in real-world conditions. 

**Abstract (ZH)**: 基于不同基础模型集成策略的机器人系统构建范式：从语言到行动的探索 

---
# SwarmDiff: Swarm Robotic Trajectory Planning in Cluttered Environments via Diffusion Transformer 

**Title (ZH)**: SwarmDiff: Swarm机器人在复杂环境中的轨迹规划通过扩散变换器 

**Authors**: Kang Ding, Chunxuan Jiao, Yunze Hu, Kangjie Zhou, Pengying Wu, Yao Mu, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15679)  

**Abstract**: Swarm robotic trajectory planning faces challenges in computational efficiency, scalability, and safety, particularly in complex, obstacle-dense environments. To address these issues, we propose SwarmDiff, a hierarchical and scalable generative framework for swarm robots. We model the swarm's macroscopic state using Probability Density Functions (PDFs) and leverage conditional diffusion models to generate risk-aware macroscopic trajectory distributions, which then guide the generation of individual robot trajectories at the microscopic level. To ensure a balance between the swarm's optimal transportation and risk awareness, we integrate Wasserstein metrics and Conditional Value at Risk (CVaR). Additionally, we introduce a Diffusion Transformer (DiT) to improve sampling efficiency and generation quality by capturing long-range dependencies. Extensive simulations and real-world experiments demonstrate that SwarmDiff outperforms existing methods in computational efficiency, trajectory validity, and scalability, making it a reliable solution for swarm robotic trajectory planning. 

**Abstract (ZH)**: 群机器人轨迹规划面临计算效率、可扩展性和安全性方面的挑战，尤其是在复杂、障碍密集的环境中。为应对这些问题，我们提出了一种分层且可扩展的生成框架SwarmDiff。我们使用概率密度函数（PDF）建模群组的宏观状态，并利用条件扩散模型生成风险感知的宏观轨迹分布，进而指导微观层面个体机器人轨迹的生成。为了在群组最优运输和风险意识之间取得平衡，我们整合了 Wasserstein 距离和条件风险值（CVaR）。此外，我们引入了扩散变换器（DiT），通过捕获长-range依赖关系来提高采样效率和生成质量。广泛的仿真实验和实地实验表明，SwarmDiff 在计算效率、轨迹有效性及可扩展性方面优于现有方法，使其成为群机器人轨迹规划的可靠解决方案。 

---
# Exploring the Limits of Vision-Language-Action Manipulations in Cross-task Generalization 

**Title (ZH)**: 探索视觉-语言-行动操控在跨任务泛化的极限 

**Authors**: Jiaming Zhou, Ke Ye, Jiayi Liu, Teli Ma, Zifang Wang, Ronghe Qiu, Kun-Yu Lin, Zhilin Zhao, Junwei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15660)  

**Abstract**: The generalization capabilities of vision-language-action (VLA) models to unseen tasks are crucial to achieving general-purpose robotic manipulation in open-world settings. However, the cross-task generalization capabilities of existing VLA models remain significantly underexplored. To address this gap, we introduce AGNOSTOS, a novel simulation benchmark designed to rigorously evaluate cross-task zero-shot generalization in manipulation. AGNOSTOS comprises 23 unseen manipulation tasks for testing, distinct from common training task distributions, and incorporates two levels of generalization difficulty to assess robustness. Our systematic evaluation reveals that current VLA models, despite being trained on diverse datasets, struggle to generalize effectively to these unseen tasks. To overcome this limitation, we propose Cross-Task In-Context Manipulation (X-ICM), a method that conditions large language models (LLMs) on in-context demonstrations from seen tasks to predict action sequences for unseen tasks. Additionally, we introduce a dynamics-guided sample selection strategy that identifies relevant demonstrations by capturing cross-task dynamics. On AGNOSTOS, X-ICM significantly improves cross-task zero-shot generalization performance over leading VLAs. We believe AGNOSTOS and X-ICM will serve as valuable tools for advancing general-purpose robotic manipulation. 

**Abstract (ZH)**: 视觉-语言-动作模型在 unseen 任务上的泛化能力对于实现开放环境下的通用机器人操作至关重要。然而，现有视觉-语言-动作模型在跨任务泛化能力方面仍存在显著的不足。为解决这一问题，我们引入了 AGNOSTOS，一个新的仿真基准，旨在严谨评估操作中的跨任务零样本泛化能力。AGNOSTOS 包含 23 个未见过的操作任务，与常见的训练任务分布不同，并包含了两个级别的泛化难度以评估鲁棒性。系统性的评估表明，尽管当前视觉-语言-动作模型在多个数据集上进行了训练，但在这些未见过的任务上泛化效果仍然较差。为克服这一局限，我们提出了跨任务在上下文操作（X-ICM）方法，该方法通过在上下文中使用已见过任务的示范来条件化大语言模型（LLM），以预测未见过任务的动作序列。此外，我们还提出了一种动力学导向的样本选择策略，能够通过捕获跨任务动力学来识别相关的示范。在 AGNOSTOS 上，X-ICM 显著提升了现有视觉-语言-动作模型的跨任务零样本泛化性能。我们相信 AGNOSTOS 和 X-ICM 将成为推动通用机器人操作的重要工具。 

---
# FLARE: Robot Learning with Implicit World Modeling 

**Title (ZH)**: FLARE: 机器人学习与隐式世界建模 

**Authors**: Ruijie Zheng, Jing Wang, Scott Reed, Johan Bjorck, Yu Fang, Fengyuan Hu, Joel Jang, Kaushil Kundalia, Zongyu Lin, Loic Magne, Avnish Narayan, You Liang Tan, Guanzhi Wang, Qi Wang, Jiannan Xiang, Yinzhen Xu, Seonghyeon Ye, Jan Kautz, Furong Huang, Yuke Zhu, Linxi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.15659)  

**Abstract**: We introduce $\textbf{F}$uture $\textbf{LA}$tent $\textbf{RE}$presentation Alignment ($\textbf{FLARE}$), a novel framework that integrates predictive latent world modeling into robot policy learning. By aligning features from a diffusion transformer with latent embeddings of future observations, $\textbf{FLARE}$ enables a diffusion transformer policy to anticipate latent representations of future observations, allowing it to reason about long-term consequences while generating actions. Remarkably lightweight, $\textbf{FLARE}$ requires only minimal architectural modifications -- adding a few tokens to standard vision-language-action (VLA) models -- yet delivers substantial performance gains. Across two challenging multitask simulation imitation learning benchmarks spanning single-arm and humanoid tabletop manipulation, $\textbf{FLARE}$ achieves state-of-the-art performance, outperforming prior policy learning baselines by up to 26%. Moreover, $\textbf{FLARE}$ unlocks the ability to co-train with human egocentric video demonstrations without action labels, significantly boosting policy generalization to a novel object with unseen geometry with as few as a single robot demonstration. Our results establish $\textbf{FLARE}$ as a general and scalable approach for combining implicit world modeling with high-frequency robotic control. 

**Abstract (ZH)**: 未来潜在表示对齐（FLARE）：一种将预测潜在世界建模集成到机器人策学习中的新型框架 

---
# Robo-DM: Data Management For Large Robot Datasets 

**Title (ZH)**: Robo-DM: 大规模机器人数据管理 

**Authors**: Kaiyuan Chen, Letian Fu, David Huang, Yanxiang Zhang, Lawrence Yunliang Chen, Huang Huang, Kush Hari, Ashwin Balakrishna, Ted Xiao, Pannag R Sanketi, John Kubiatowicz, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.15558)  

**Abstract**: Recent results suggest that very large datasets of teleoperated robot demonstrations can be used to train transformer-based models that have the potential to generalize to new scenes, robots, and tasks. However, curating, distributing, and loading large datasets of robot trajectories, which typically consist of video, textual, and numerical modalities - including streams from multiple cameras - remains challenging. We propose Robo-DM, an efficient open-source cloud-based data management toolkit for collecting, sharing, and learning with robot data. With Robo-DM, robot datasets are stored in a self-contained format with Extensible Binary Meta Language (EBML). Robo-DM can significantly reduce the size of robot trajectory data, transfer costs, and data load time during training. Compared to the RLDS format used in OXE datasets, Robo-DM's compression saves space by up to 70x (lossy) and 3.5x (lossless). Robo-DM also accelerates data retrieval by load-balancing video decoding with memory-mapped decoding caches. Compared to LeRobot, a framework that also uses lossy video compression, Robo-DM is up to 50x faster when decoding sequentially. We physically evaluate a model trained by Robo-DM with lossy compression, a pick-and-place task, and In-Context Robot Transformer. Robo-DM uses 75x compression of the original dataset and does not suffer reduction in downstream task accuracy. 

**Abstract (ZH)**: 近期的研究表明，电信号操作机器人演示的大规模数据集可以用于训练潜在能够泛化到新场景、机器人和任务的变压器模型。然而，收集、分发和加载包含视频、文本和数值等多种模态的大型机器人轨迹数据集——通常包括多个摄像头的流数据——仍然具有挑战性。我们提出了Robo-DM，一个高效开源的基于云的数据管理工具包，用于收集、共享和利用机器人数据。Robo-DM以可扩展二进制元语言（EBML）格式存储机器人数据集，可以显著减少机器人轨迹数据的大小、传输成本和训练时的数据加载时间。与OXE数据集中使用的RLDS格式相比，Robo-DM的压缩在有损情况下可以节省多达70倍的空间，在无损情况下可以节省3.5倍的空间。Robo-DM还通过负载均衡视频解码和内存映射解码缓存来加速数据检索。与使用有损视频压缩的LeRobot框架相比，当按顺序解码时，Robo-DM的速度可以快50倍。物理实验中，使用经过Robo-DM压缩的数据训练的模型，在进行抓取和放置任务时，基于上下文的机器人变换器表现出色，尽管压缩后的数据集仅为原始数据集的1/75，但下游任务的准确性并未受到影响。 

---
# Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets 

**Title (ZH)**: Robo2VLM：来自大规模野外机器人操作数据集的视觉问答 

**Authors**: Kaiyuan Chen, Shuangyu Xie, Zehan Ma, Ken Goldberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.15517)  

**Abstract**: Vision-Language Models (VLMs) acquire real-world knowledge and general reasoning ability through Internet-scale image-text corpora. They can augment robotic systems with scene understanding and task planning, and assist visuomotor policies that are trained on robot trajectory data. We explore the reverse paradigm - using rich, real, multi-modal robot trajectory data to enhance and evaluate VLMs. In this paper, we present Robo2VLM, a Visual Question Answering (VQA) dataset generation framework for VLMs. Given a human tele-operated robot trajectory, Robo2VLM derives ground-truth from non-visual and non-descriptive sensory modalities, such as end-effector pose, gripper aperture, and force sensing. Based on these modalities, it segments the robot trajectory into a sequence of manipulation phases. At each phase, Robo2VLM uses scene and interaction understanding to identify 3D properties of the robot, task goal, and the target object. The properties are used to generate representative VQA queries - images with textural multiple-choice questions - based on spatial, goal-conditioned, and interaction reasoning question templates. We curate Robo2VLM-1, a large-scale in-the-wild dataset with 684,710 questions covering 463 distinct scenes and 3,396 robotic manipulation tasks from 176k real robot trajectories. Results suggest that Robo2VLM-1 can benchmark and improve VLM capabilities in spatial and interaction reasoning. 

**Abstract (ZH)**: 基于机器人轨迹数据增强和评估视觉语言模型的Robo2VLM框架 

---
# Coloring Between the Lines: Personalization in the Null Space of Planning Constraints 

**Title (ZH)**: 在规划约束的 null space 中着色：个性化方法 

**Authors**: Tom Silver, Rajat Kumar Jenamani, Ziang Liu, Ben Dodson, Tapomayukh Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15503)  

**Abstract**: Generalist robots must personalize in-the-wild to meet the diverse needs and preferences of long-term users. How can we enable flexible personalization without sacrificing safety or competency? This paper proposes Coloring Between the Lines (CBTL), a method for personalization that exploits the null space of constraint satisfaction problems (CSPs) used in robot planning. CBTL begins with a CSP generator that ensures safe and competent behavior, then incrementally personalizes behavior by learning parameterized constraints from online interaction. By quantifying uncertainty and leveraging the compositionality of planning constraints, CBTL achieves sample-efficient adaptation without environment resets. We evaluate CBTL in (1) three diverse simulation environments; (2) a web-based user study; and (3) a real-robot assisted feeding system, finding that CBTL consistently achieves more effective personalization with fewer interactions than baselines. Our results demonstrate that CBTL provides a unified and practical approach for continual, flexible, active, and safe robot personalization. Website: this https URL 

**Abstract (ZH)**: 通用机器人必须在真实环境中个性化以满足长期用户的多样化需求和偏好。我们如何能够在不牺牲安全性和专业性的情况下实现灵活的个性化？本文提出了一种名为“在空白处着色”（CBTL，Coloring Between the Lines）的方法，该方法利用了机器人规划中约束满足问题（CSPs）的零空间来进行个性化。CBTL以一个确保安全和专业行为的CSP生成器开始，然后通过在线交互学习参数化约束规则逐步实现个性化。通过量化不确定性并利用规划约束的组合性，CBTL在无需重新配置环境的情况下实现了高效适应。我们在（1）三个不同的仿真环境中；（2）一项基于网络的用户研究中；以及（3）一个真实机器人辅助喂食系统中评估了CBTL，发现CBTL在更少的交互中实现了更有效的个性化，优于基线方法。我们的结果表明，CBTL提供了一种统一且实用的方法，可用于持续的、灵活的、主动的安全机器人个性化。 

---
# Synthetic Enclosed Echoes: A New Dataset to Mitigate the Gap Between Simulated and Real-World Sonar Data 

**Title (ZH)**: 合成封闭回声：一个新的数据集，用于减少模拟与真实声纳数据之间的差距 

**Authors**: Guilherme de Oliveira, Matheus M. dos Santos, Paulo L. J. Drews-Jr  

**Link**: [PDF](https://arxiv.org/pdf/2505.15465)  

**Abstract**: This paper introduces Synthetic Enclosed Echoes (SEE), a novel dataset designed to enhance robot perception and 3D reconstruction capabilities in underwater environments. SEE comprises high-fidelity synthetic sonar data, complemented by a smaller subset of real-world sonar data. To facilitate flexible data acquisition, a simulated environment has been developed, enabling the generation of additional data through modifications such as the inclusion of new structures or imaging sonar configurations. This hybrid approach leverages the advantages of synthetic data, including readily available ground truth and the ability to generate diverse datasets, while bridging the simulation-to-reality gap with real-world data acquired in a similar environment. The SEE dataset comprehensively evaluates acoustic data-based methods, including mathematics-based sonar approaches and deep learning algorithms. These techniques were employed to validate the dataset, confirming its suitability for underwater 3D reconstruction. Furthermore, this paper proposes a novel modification to a state-of-the-art algorithm, demonstrating improved performance compared to existing methods. The SEE dataset enables the evaluation of acoustic data-based methods in realistic scenarios, thereby improving their feasibility for real-world underwater applications. 

**Abstract (ZH)**: 合成封闭回声数据集（SEE）：一种用于增强水下环境机器人感知和三维重建能力的新型数据集 

---
# Evaluation of Mobile Environment for Vehicular Visible Light Communication Using Multiple LEDs and Event Cameras 

**Title (ZH)**: 多LED和事件摄像头环境下基于移动环境的车载可见光通信评估 

**Authors**: Ryota Soga, Shintaro Shiba, Quan Kong, Norimasa Kobori, Tsukasa Shimizu, Shan Lu, Takaya Yamazato  

**Link**: [PDF](https://arxiv.org/pdf/2505.15412)  

**Abstract**: In the fields of Advanced Driver Assistance Systems (ADAS) and Autonomous Driving (AD), sensors that serve as the ``eyes'' for sensing the vehicle's surrounding environment are essential. Traditionally, image sensors and LiDAR have played this role. However, a new type of vision sensor, event cameras, has recently attracted attention. Event cameras respond to changes in the surrounding environment (e.g., motion), exhibit strong robustness against motion blur, and perform well in high dynamic range environments, which are desirable in robotics applications. Furthermore, the asynchronous and low-latency principles of data acquisition make event cameras suitable for optical communication. By adding communication functionality to event cameras, it becomes possible to utilize I2V communication to immediately share information about forward collisions, sudden braking, and road conditions, thereby contributing to hazard avoidance. Additionally, receiving information such as signal timing and traffic volume enables speed adjustment and optimal route selection, facilitating more efficient driving. In this study, we construct a vehicle visible light communication system where event cameras are receivers, and multiple LEDs are transmitters. In driving scenes, the system tracks the transmitter positions and separates densely packed LED light sources using pilot sequences based on Walsh-Hadamard codes. As a result, outdoor vehicle experiments demonstrate error-free communication under conditions where the transmitter-receiver distance was within 40 meters and the vehicle's driving speed was 30 km/h (8.3 m/s). 

**Abstract (ZH)**: 先进驾驶辅助系统（ADAS）和自动驾驶（AD）领域中的传感器作为“眼睛”用于感知车辆周围环境，是必不可少的。传统上，图像传感器和LiDAR担任这一角色。然而， Recently, 一种新型视觉传感器——事件摄像机引起了关注。事件摄像机对周围环境的变化（如运动）作出响应，具有较强的运动模糊鲁棒性，并在高动态范围环境中表现出色，这在机器人应用中很有优势。此外，事件摄像机的数据采集遵循非同步和低延迟原则，使其适合用于光学通信。通过向事件摄像机添加通信功能，可以利用I2V通信立即共享前方碰撞、紧急制动和道路状况信息，从而有助于风险规避。此外，接收信号定时和交通流量等信息能够实现速度调节和最佳路线选择，从而促进更高效的驾驶。在本研究中，我们构建了一个车辆可见光通信系统，其中事件摄像机作为接收器，多个LED作为发射器。在行驶场景中，系统跟踪发射器位置，并基于Walsh-Hadamard码的导频序列分离密集排列的LED光源。结果，在室外车辆实验中，当发射器-接收器距离在40米以内，车辆行驶速度为30 km/h（8.3 m/s）时，实现了无错误通信。 

---
# Saliency-Aware Quantized Imitation Learning for Efficient Robotic Control 

**Title (ZH)**: 具有注意力意识的量化imitation learning及其在高效机器人控制中的应用 

**Authors**: Seongmin Park, Hyungmin Kim, Sangwoo kim, Wonseok Jeon, Juyoung Yang, Byeongwook Jeon, Yoonseon Oh, Jungwook Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15304)  

**Abstract**: Deep neural network (DNN)-based policy models, such as vision-language-action (VLA) models, excel at automating complex decision-making from multi-modal inputs. However, scaling these models greatly increases computational overhead, complicating deployment in resource-constrained settings like robot manipulation and autonomous driving. To address this, we propose Saliency-Aware Quantized Imitation Learning (SQIL), which combines quantization-aware training with a selective loss-weighting strategy for mission-critical states. By identifying these states via saliency scores and emphasizing them in the training loss, SQIL preserves decision fidelity under low-bit precision. We validate SQIL's generalization capability across extensive simulation benchmarks with environment variations, real-world tasks, and cross-domain tasks (self-driving, physics simulation), consistently recovering full-precision performance. Notably, a 4-bit weight-quantized VLA model for robotic manipulation achieves up to 2.5x speedup and 2.5x energy savings on an edge GPU with minimal accuracy loss. These results underline SQIL's potential for efficiently deploying large IL-based policy models on resource-limited devices. 

**Abstract (ZH)**: 基于稀量化知觉-语言-动作模型的显著性感知量化模仿学习（SQIL） 

---
# AgentThink: A Unified Framework for Tool-Augmented Chain-of-Thought Reasoning in Vision-Language Models for Autonomous Driving 

**Title (ZH)**: AgentThink：增强链式思维推理的统一框架在自主驾驶视觉-语言模型中的应用 

**Authors**: Kangan Qian, Sicong Jiang, Yang Zhong, Ziang Luo, Zilin Huang, Tianze Zhu, Kun Jiang, Mengmeng Yang, Zheng Fu, Jinyu Miao, Yining Shi, He Zhe Lim, Li Liu, Tianbao Zhou, Hongyi Wang, Huang Yu, Yifei Hu, Guang Li, Guang Chen, Hao Ye, Lijun Sun, Diange Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15298)  

**Abstract**: Vision-Language Models (VLMs) show promise for autonomous driving, yet their struggle with hallucinations, inefficient reasoning, and limited real-world validation hinders accurate perception and robust step-by-step reasoning. To overcome this, we introduce \textbf{AgentThink}, a pioneering unified framework that, for the first time, integrates Chain-of-Thought (CoT) reasoning with dynamic, agent-style tool invocation for autonomous driving tasks. AgentThink's core innovations include: \textbf{(i) Structured Data Generation}, by establishing an autonomous driving tool library to automatically construct structured, self-verified reasoning data explicitly incorporating tool usage for diverse driving scenarios; \textbf{(ii) A Two-stage Training Pipeline}, employing Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO) to equip VLMs with the capability for autonomous tool invocation; and \textbf{(iii) Agent-style Tool-Usage Evaluation}, introducing a novel multi-tool assessment protocol to rigorously evaluate the model's tool invocation and utilization. Experiments on the DriveLMM-o1 benchmark demonstrate AgentThink significantly boosts overall reasoning scores by \textbf{53.91\%} and enhances answer accuracy by \textbf{33.54\%}, while markedly improving reasoning quality and consistency. Furthermore, ablation studies and robust zero-shot/few-shot generalization experiments across various benchmarks underscore its powerful capabilities. These findings highlight a promising trajectory for developing trustworthy and tool-aware autonomous driving models. 

**Abstract (ZH)**: Vision-Language模型（VLMs）在自动驾驶领域展现出潜力，但它们在幻觉、低效推理和有限的现实世界验证方面的挑战阻碍了准确感知和稳健的逐步推理。为克服这些挑战，我们提出了**AgentThink**，这是一种开创性的统一框架，首次将链式推理（CoT）与动态、代理风格的工具调用集成到自动驾驶任务中。AgentThink的核心创新包括：**（i）结构化数据生成**，通过建立自动驾驶工具库，自动构建结构化、自验证的推理数据，明确包括工具使用情况，适用于多种驾驶场景；**（ii）两阶段训练pipeline**，采用监督微调（SFT）与组相对策略优化（GRPO）结合，使VLMs具备自主调用工具的能力；以及**（iii）代理风格的工具使用评估**，引入一种新的多工具评估协议，以严格评估模型的工具调用和使用情况。DriveLMM-o1基准测试实验表明，AgentThink在整体推理得分上提高了**53.91%**，答案准确性提高了**33.54%**，同时显著提高了推理质量和一致性。此外，跨不同基准的消融研究和鲁棒零-shot/少-shot泛化实验进一步证明了其强大的功能。这些发现突显了开发可信且工具感知的自动驾驶模型有希望的发展方向。 

---
# Learning-based Autonomous Oversteer Control and Collision Avoidance 

**Title (ZH)**: 基于学习的自主过度转向控制与碰撞规避 

**Authors**: Seokjun Lee, Seung-Hyun Kong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15275)  

**Abstract**: Oversteer, wherein a vehicle's rear tires lose traction and induce unintentional excessive yaw, poses critical safety challenges. Failing to control oversteer often leads to severe traffic accidents. Although recent autonomous driving efforts have attempted to handle oversteer through stabilizing maneuvers, the majority rely on expert-defined trajectories or assume obstacle-free environments, limiting real-world applicability. This paper introduces a novel end-to-end (E2E) autonomous driving approach that tackles oversteer control and collision avoidance simultaneously. Existing E2E techniques, including Imitation Learning (IL), Reinforcement Learning (RL), and Hybrid Learning (HL), generally require near-optimal demonstrations or extensive experience. Yet even skilled human drivers struggle to provide perfect demonstrations under oversteer, and high transition variance hinders accumulating sufficient data. Hence, we present Q-Compared Soft Actor-Critic (QC-SAC), a new HL algorithm that effectively learns from suboptimal demonstration data and adapts rapidly to new conditions. To evaluate QC-SAC, we introduce a benchmark inspired by real-world driver training: a vehicle encounters sudden oversteer on a slippery surface and must avoid randomly placed obstacles ahead. Experimental results show QC-SAC attains near-optimal driving policies, significantly surpassing state-of-the-art IL, RL, and HL baselines. Our method demonstrates the world's first safe autonomous oversteer control with obstacle avoidance. 

**Abstract (ZH)**: 过 steer 控制：一种同时处理过 steer 和碰撞避免的端到端自主驾驶方法及其应用 

---
# GCNT: Graph-Based Transformer Policies for Morphology-Agnostic Reinforcement Learning 

**Title (ZH)**: GCNT：基于图的变换器策略在形态学无感知强化学习中的应用 

**Authors**: Yingbo Luo, Meibao Yao, Xueming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.15211)  

**Abstract**: Training a universal controller for robots with different morphologies is a promising research trend, since it can significantly enhance the robustness and resilience of the robotic system. However, diverse morphologies can yield different dimensions of state space and action space, making it difficult to comply with traditional policy networks. Existing methods address this issue by modularizing the robot configuration, while do not adequately extract and utilize the overall morphological information, which has been proven crucial for training a universal controller. To this end, we propose GCNT, a morphology-agnostic policy network based on improved Graph Convolutional Network (GCN) and Transformer. It exploits the fact that GCN and Transformer can handle arbitrary number of modules to achieve compatibility with diverse morphologies. Our key insight is that the GCN is able to efficiently extract morphology information of robots, while Transformer ensures that it is fully utilized by allowing each node of the robot to communicate this information directly. Experimental results show that our method can generate resilient locomotion behaviors for robots with different configurations, including zero-shot generalization to robot morphologies not seen during training. In particular, GCNT achieved the best performance on 8 tasks in the 2 standard benchmarks. 

**Abstract (ZH)**: 基于改进的图卷积网络和变换器的形态无关控制器训练方法 

---
# EndoVLA: Dual-Phase Vision-Language-Action Model for Autonomous Tracking in Endoscopy 

**Title (ZH)**: EndoVLA: 自主内镜跟踪的双阶段视觉-语言-动作模型 

**Authors**: Chi Kit Ng, Long Bai, Guankun Wang, Yupeng Wang, Huxin Gao, Kun Yuan, Chenhan Jin, Tieyong Zeng, Hongliang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.15206)  

**Abstract**: In endoscopic procedures, autonomous tracking of abnormal regions and following circumferential cutting markers can significantly reduce the cognitive burden on endoscopists. However, conventional model-based pipelines are fragile for each component (e.g., detection, motion planning) requires manual tuning and struggles to incorporate high-level endoscopic intent, leading to poor generalization across diverse scenes. Vision-Language-Action (VLA) models, which integrate visual perception, language grounding, and motion planning within an end-to-end framework, offer a promising alternative by semantically adapting to surgeon prompts without manual recalibration. Despite their potential, applying VLA models to robotic endoscopy presents unique challenges due to the complex and dynamic anatomical environments of the gastrointestinal (GI) tract. To address this, we introduce EndoVLA, designed specifically for continuum robots in GI interventions. Given endoscopic images and surgeon-issued tracking prompts, EndoVLA performs three core tasks: (1) polyp tracking, (2) delineation and following of abnormal mucosal regions, and (3) adherence to circular markers during circumferential cutting. To tackle data scarcity and domain shifts, we propose a dual-phase strategy comprising supervised fine-tuning on our EndoVLA-Motion dataset and reinforcement fine-tuning with task-aware rewards. Our approach significantly improves tracking performance in endoscopy and enables zero-shot generalization in diverse scenes and complex sequential tasks. 

**Abstract (ZH)**: 基于视觉-语言-动作模型的内镜自主跟踪方法：针对消化道干预的EndoVLA系统 

---
# Cascaded Diffusion Models for Neural Motion Planning 

**Title (ZH)**: 级联扩散模型在神经运动规划中的应用 

**Authors**: Mohit Sharma, Adam Fishman, Vikash Kumar, Chris Paxton, Oliver Kroemer  

**Link**: [PDF](https://arxiv.org/pdf/2505.15157)  

**Abstract**: Robots in the real world need to perceive and move to goals in complex environments without collisions. Avoiding collisions is especially difficult when relying on sensor perception and when goals are among clutter. Diffusion policies and other generative models have shown strong performance in solving local planning problems, but often struggle at avoiding all of the subtle constraint violations that characterize truly challenging global motion planning problems. In this work, we propose an approach for learning global motion planning using diffusion policies, allowing the robot to generate full trajectories through complex scenes and reasoning about multiple obstacles along the path. Our approach uses cascaded hierarchical models which unify global prediction and local refinement together with online plan repair to ensure the trajectories are collision free. Our method outperforms (by ~5%) a wide variety of baselines on challenging tasks in multiple domains including navigation and manipulation. 

**Abstract (ZH)**: 真实环境中，机器人需要在复杂环境中感知并移动到目标位置而避免碰撞。当依赖传感器感知且目标位于杂乱环境中时，避免碰撞尤为困难。扩散策略和其他生成模型在解决局部规划问题上表现出强大的性能，但在避免所有细微的约束违反方面往往难以应对真正的全局运动规划挑战。在本工作中，我们提出了一种使用扩散策略学习全局运动规划的方法，使机器人能够生成通过复杂场景的完整轨迹，并沿路径进行多障碍物推理。该方法采用级联分层模型，将全局预测和局部细化相结合，并通过在线计划修复来确保轨迹无碰撞。我们的方法在多个域（包括导航和操作）的具有挑战性的任务中显著优于多种基线方法（约5%）。 

---
# Object-Focus Actor for Data-efficient Robot Generalization Dexterous Manipulation 

**Title (ZH)**: 面向对象的关注演员：高效数据机器人通用灵巧 manipulation 

**Authors**: Yihang Li, Tianle Zhang, Xuelong Wei, Jiayi Li, Lin Zhao, Dongchi Huang, Zhirui Fang, Minhua Zheng, Wenjun Dai, Xiaodong He  

**Link**: [PDF](https://arxiv.org/pdf/2505.15098)  

**Abstract**: Robot manipulation learning from human demonstrations offers a rapid means to acquire skills but often lacks generalization across diverse scenes and object placements. This limitation hinders real-world applications, particularly in complex tasks requiring dexterous manipulation. Vision-Language-Action (VLA) paradigm leverages large-scale data to enhance generalization. However, due to data scarcity, VLA's performance remains limited. In this work, we introduce Object-Focus Actor (OFA), a novel, data-efficient approach for generalized dexterous manipulation. OFA exploits the consistent end trajectories observed in dexterous manipulation tasks, allowing for efficient policy training. Our method employs a hierarchical pipeline: object perception and pose estimation, pre-manipulation pose arrival and OFA policy execution. This process ensures that the manipulation is focused and efficient, even in varied backgrounds and positional layout. Comprehensive real-world experiments across seven tasks demonstrate that OFA significantly outperforms baseline methods in both positional and background generalization tests. Notably, OFA achieves robust performance with only 10 demonstrations, highlighting its data efficiency. 

**Abstract (ZH)**: 基于人类演示的机器人操作学习提供了快速获取技能的手段，但常常缺乏跨多样场景和物体摆放的泛化能力。这一局限性阻碍了其在复杂任务中的实际应用，特别是那些需要灵巧操作的任务。视觉-语言-动作（VLA）范式利用大规模数据以增强泛化能力。然而，由于数据稀缺，VLA的性能依然受到限制。在本工作中，我们提出了物体焦点演员（OFA），这是一种新型、数据高效的泛化灵巧操作方法。OFA 利用了灵巧操作任务中观察到的一致性末端轨迹，使策略训练更加高效。我们的方法采用层次化流水线：物体感知与姿态估计、预操作姿态到达和OFA策略执行。这一过程确保即使在多变的背景和位置布局中，操作也能保持聚焦与高效。全面的七项任务的现实世界实验表明，OFA 在位置和背景泛化测试中显著优于基线方法。值得注意的是，OFA 仅需 10 次演示就表现出鲁棒性能，突显了其数据效率。 

---
# Learning-based Airflow Inertial Odometry for MAVs using Thermal Anemometers in a GPS and vision denied environment 

**Title (ZH)**: 基于学习的微型航空器热风速仪辅助气流惯性里程计在GPS和视觉受限环境中的应用 

**Authors**: Ze Wang, Jingang Qu, Zhenyu Gao, Pascal Morin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15044)  

**Abstract**: This work demonstrates an airflow inertial based odometry system with multi-sensor data fusion, including thermal anemometer, IMU, ESC, and barometer. This goal is challenging because low-cost IMUs and barometers have significant bias, and anemometer measurements are very susceptible to interference from spinning propellers and ground effects. We employ a GRU-based deep neural network to estimate relative air speed from noisy and disturbed anemometer measurements, and an observer with bias model to fuse the sensor data and thus estimate the state of aerial vehicle. A complete flight data, including takeoff and landing on the ground, shows that the approach is able to decouple the downwash induced wind speed caused by propellers and the ground effect, and accurately estimate the flight speed in a wind-free indoor environment. IMU, and barometer bias are effectively estimated, which significantly reduces the position integration drift, which is only 5.7m for 203s manual random flight. The open source is available on this https URL. 

**Abstract (ZH)**: 基于 airflow inertial 的多传感器数据融合姿态导航系统：低偏置 IMU 和气流计的噪声估计与干扰抑制 

---
# Histo-Planner: A Real-time Local Planner for MAVs Teleoperation based on Histogram of Obstacle Distribution 

**Title (ZH)**: Histo-Planner：基于障碍分布直方图的MAVs遥操作实时局部规划器 

**Authors**: Ze Wang, Zhenyu Gao, Jingang Qu, Pascal Morin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15043)  

**Abstract**: This paper concerns real-time obstacle avoidance for micro aerial vehicles (MAVs). Motivated by teleoperation applications in cluttered environments with limited computational power, we propose a local planner that does not require the knowledge or construction of a global map of the obstacles. The proposed solution consists of a real-time trajectory planning algorithm that relies on the histogram of obstacle distribution and a planner manager that triggers different planning modes depending on obstacles location around the MAV. The proposed solution is validated, for a teleoperation application, with both simulations and indoor experiments. Benchmark comparisons based on a designed simulation platform are also provided. 

**Abstract (ZH)**: 本文探讨了微型空中车辆（MAVs）的实时障碍避让问题。受在计算能力有限的复杂环境中进行遥控操作的启发，我们提出了一种无需了解或构建障碍全局地图的局部规划器。所提出的方法包括一个基于障碍分布直方图的实时轨迹规划算法，以及一个规划管理器，该管理器可以根据MAV周围障碍物的位置触发不同的规划模式。该方法在遥控操作应用中通过仿真和室内实验进行了验证，并提供了基于设计的仿真平台的基准比较。 

---
# Fault-Tolerant Multi-Robot Coordination with Limited Sensing within Confined Environments 

**Title (ZH)**: 受限环境内有限感知条件下的容错多机器人协调 

**Authors**: Kehinde O. Aina, Hosain Bagheri, Daniel I. Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2505.15036)  

**Abstract**: As robots are increasingly deployed to collaborate on tasks within shared workspaces and resources, the failure of an individual robot can critically affect the group's performance. This issue is particularly challenging when robots lack global information or direct communication, relying instead on social interaction for coordination and to complete their tasks. In this study, we propose a novel fault-tolerance technique leveraging physical contact interactions in multi-robot systems, specifically under conditions of limited sensing and spatial confinement. We introduce the "Active Contact Response" (ACR) method, where each robot modulates its behavior based on the likelihood of encountering an inoperative (faulty) robot. Active robots are capable of collectively repositioning stationary and faulty peers to reduce obstructions and maintain optimal group functionality. We implement our algorithm in a team of autonomous robots, equipped with contact-sensing and collision-tolerance capabilities, tasked with collectively excavating cohesive model pellets. Experimental results indicate that the ACR method significantly improves the system's recovery time from robot failures, enabling continued collective excavation with minimal performance degradation. Thus, this work demonstrates the potential of leveraging local, social, and physical interactions to enhance fault tolerance and coordination in multi-robot systems operating in constrained and extreme environments. 

**Abstract (ZH)**: 随着机器人在共享工作空间和资源中协作执行任务的应用越来越广泛，单个机器人的故障会严重影响团队的整体性能。在机器人缺乏全局信息或直接通信能力的情况下，依靠社会交互进行协调和完成任务尤其具有挑战性。本研究提出了一种新颖的容错技术，利用多机器人系统中的物理接触交互，特别是在受限感知和空间限制条件下。我们提出了“主动接触响应”（ACR）方法，其中每个机器人根据遇到故障机器人可能性调整其行为。活跃机器人能够集体重新定位并调整故障同伴的位置，减少障碍并维持最佳团队功能。我们将该算法实施于具备接触感知和碰撞容忍能力的自主机器人团队中，任务是集体挖掘黏聚的模型颗粒。实验结果表明，ACR方法显著缩短了机器人故障后系统的恢复时间，且在几乎不影响性能的情况下持续进行集体挖掘。因此，该工作展示了利用局部、社会和物理交互增强多机器人系统在受限和极端环境中的容错能力和协调能力的潜力。 

---
# Toward Task Capable Active Matter: Learning to Avoid Clogging in Confined Collectives via Collisions 

**Title (ZH)**: 面向任务的能力型活性物质：通过碰撞学习在受限群体中避免堵塞 

**Authors**: Kehinde O. Aina, Ram Avinery, Hui-Shun Kuan, Meredith D. Betterton, Michael A. D. Goodisman, Daniel I. Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2505.15033)  

**Abstract**: Social organisms which construct nests consisting of tunnels and chambers necessarily navigate confined and crowded conditions. Unlike low-density collectives like bird flocks and insect swarms, in which hydrodynamic and statistical phenomena dominate, the physics of glasses and supercooled fluids is important to understand clogging behaviors in high-density collectives. Our previous work revealed that fire ants flowing in confined tunnels utilize diverse behaviors like unequal workload distributions, spontaneous direction reversals, and limited interaction times to mitigate clogging and jamming and thus maintain functional flow; implementation of similar rules in a small robophysical swarm led to high performance through spontaneous dissolution of clogs and clusters. However, how the insects learn such behaviors, and how we can develop "task capable" active matter in such regimes, remains a challenge in part because interaction dynamics are dominated by local, time-consuming collisions and no single agent can guide the entire collective. Here, we hypothesized that effective flow and clog mitigation could emerge purely through local learning. We tasked small groups of robots with pellet excavation in a narrow tunnel, allowing them to modify reversal probabilities over time. Initially, robots had equal probabilities and clogs were common. Reversals improved flow. When reversal probabilities adapted via collisions and noisy tunnel length estimates, workload inequality and performance improved. Our robophysical study of an excavating swarm shows that, despite the seeming complexity and difficulty of the task, simple learning rules can mitigate or leverage unavoidable features in task-capable dense active matter, leading to hypotheses for dense biological and robotic swarms. 

**Abstract (ZH)**: 受约束和拥挤条件下筑巢的社会生物必须导航通过这些条件。主动物质在高密度集体中的阻塞行为不能通过低密度鸟群和昆虫群的流体力学和统计现象来理解，而是需要理解玻璃态和超冷却流体的物理学。我们先前的工作揭示了火蚁在狭窄隧道中流动时利用不均等的工作负担分配、自发的方向反转以及有限的互动时间来减轻阻塞和卡顿，从而维持功能性流动。将类似规则应用于小型机器人物理群显著提高了性能，通过自发溶解阻塞和聚集体实现了这一点。然而，昆虫是如何学习这些行为的，以及我们如何在以局部、耗时碰撞为主导的互动动力学中开发“任务能力”主动物质，仍然是一个挑战。在这里，我们假设有效的流动和阻塞缓解可以通过局部学习自发地出现。我们要求小型机器人小组在狭窄隧道中执行颗粒挖掘任务，允许它们随时间调整反转概率。最初，机器人具有相等的概率，阻塞常见。反转改善了流动。当反转概率通过碰撞和隧道长度的噪声估计进行调整时，工作负担不平等和性能提高。我们的机器人物理研究显示，在任务能力密集主动物质中，即使任务看似复杂且难以完成，简单的学习规则也能缓解或利用不可避免的特征，从而为密集生物群和机器人群提供了见解。 

---
# Shape-Adaptive Planning and Control for a Deformable Quadrotor 

**Title (ZH)**: 自适应形变四旋翼飞行器的规划与控制 

**Authors**: Yuze Wu, Zhichao Han, Xuankang Wu, Yuan Zhou, Junjie Wang, Zheng Fang, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.15010)  

**Abstract**: Drones have become essential in various applications, but conventional quadrotors face limitations in confined spaces and complex tasks. Deformable drones, which can adapt their shape in real-time, offer a promising solution to overcome these challenges, while also enhancing maneuverability and enabling novel tasks like object grasping. This paper presents a novel approach to autonomous motion planning and control for deformable quadrotors. We introduce a shape-adaptive trajectory planner that incorporates deformation dynamics into path generation, using a scalable kinodynamic A* search to handle deformation parameters in complex environments. The backend spatio-temporal optimization is capable of generating optimally smooth trajectories that incorporate shape deformation. Additionally, we propose an enhanced control strategy that compensates for external forces and torque disturbances, achieving a 37.3\% reduction in trajectory tracking error compared to our previous work. Our approach is validated through simulations and real-world experiments, demonstrating its effectiveness in narrow-gap traversal and multi-modal deformable tasks. 

**Abstract (ZH)**: 变形无人机在受限空间和复杂任务中的自主运动规划与控制 

---
# UniSTPA: A Safety Analysis Framework for End-to-End Autonomous Driving 

**Title (ZH)**: UniSTPA: 一端到一端自主驾驶的安全分析框架 

**Authors**: Hongrui Kou, Zhouhang Lyu, Ziyu Wang, Cheng Wang, Yuxin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15005)  

**Abstract**: As autonomous driving technology continues to advance, end-to-end models have attracted considerable attention owing to their superior generalisation capability. Nevertheless, such learning-based systems entail numerous safety risks throughout development and on-road deployment, and existing safety-analysis methods struggle to identify these risks comprehensively. To address this gap, we propose the Unified System Theoretic Process Analysis (UniSTPA) framework, which extends the scope of STPA from the operational phase to the entire lifecycle of an end-to-end autonomous driving system, including information gathering, data preparation, closed loop training, verification, and deployment. UniSTPA performs hazard analysis not only at the component level but also within the model's internal layers, thereby enabling fine-grained assessment of inter and intra module interactions. Using a highway Navigate on Autopilot function as a case study, UniSTPA uncovers multi-stage hazards overlooked by conventional approaches including scene design defects, sensor fusion biases, and internal model flaws, through multi-level causal analysis, traces these hazards to deeper issues such as data quality, network architecture, and optimisation objectives. The analysis result are used to construct a safety monitoring and safety response mechanism that supports continuous improvement from hazard identification to system optimisation. The proposed framework thus offers both theoretical and practical guidance for the safe development and deployment of end-to-end autonomous driving systems. 

**Abstract (ZH)**: 随着自动驾驶技术的不断进步，端到端模型由于其卓越的泛化能力而备受关注。然而，此类基于学习的系统在开发和实际道路部署过程中存在众多安全风险，现有安全分析方法难以全面识别这些风险。为解决这一问题，我们提出了统一系统理论过程分析（UniSTPA）框架，该框架将系统理论过程分析（STPA）的范围从运行阶段扩展到端到端自动驾驶系统的整个生命周期，包括信息收集、数据准备、闭环训练、验证和部署。UniSTPA 不仅在组件级别，还在模型内部层面上进行危害分析，从而实现模块间及模块内交互的精细评估。通过多级因果分析，UniSTPA 揭示了传统方法忽略的多阶段危害，包括场景设计缺陷、传感器融合偏差和内部模型缺陷，并将这些危害追溯到更深层次的问题，如数据质量、网络架构和优化目标。分析结果用于构建支持从危害识别到系统优化的持续改进的安全监控和安全响应机制。因此，所提出的框架为端到端自动驾驶系统的安全开发和部署提供了理论和实践指导。 

---
# AnyBody: A Benchmark Suite for Cross-Embodiment Manipulation 

**Title (ZH)**: AnyBody: 一个跨身躯操纵基准套件 

**Authors**: Meenal Parakh, Alexandre Kirchmeyer, Beining Han, Jia Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14986)  

**Abstract**: Generalizing control policies to novel embodiments remains a fundamental challenge in enabling scalable and transferable learning in robotics. While prior works have explored this in locomotion, a systematic study in the context of manipulation tasks remains limited, partly due to the lack of standardized benchmarks. In this paper, we introduce a benchmark for learning cross-embodiment manipulation, focusing on two foundational tasks-reach and push-across a diverse range of morphologies. The benchmark is designed to test generalization along three axes: interpolation (testing performance within a robot category that shares the same link structure), extrapolation (testing on a robot with a different link structure), and composition (testing on combinations of link structures). On the benchmark, we evaluate the ability of different RL policies to learn from multiple morphologies and to generalize to novel ones. Our study aims to answer whether morphology-aware training can outperform single-embodiment baselines, whether zero-shot generalization to unseen morphologies is feasible, and how consistently these patterns hold across different generalization regimes. The results highlight the current limitations of multi-embodiment learning and provide insights into how architectural and training design choices influence policy generalization. 

**Abstract (ZH)**: 通用化控制策略到新型体态 remains a fundamental challenge in enabling scalable and transferable learning in robotics. While prior works have explored this in locomotion, a systematic study in the context of manipulation tasks remains limited, partly due to the lack of standardized benchmarks. In this paper, we introduce a benchmark for learning cross-embodiment manipulation, focusing on two foundational tasks-reach and push-across a diverse range of morphologies. The benchmark is designed to test generalization along three axes: interpolation (testing performance within a robot category that shares the same link structure), extrapolation (testing on a robot with a different link structure), and composition (testing on combinations of link structures). On the benchmark, we evaluate the ability of different RL policies to learn from multiple morphologies and to generalize to novel ones. Our study aims to answer whether morphology-aware training can outperform single-embodiment baselines, whether zero-shot generalization to unseen morphologies is feasible, and how consistently these patterns hold across different generalization regimes. The results highlight the current limitations of multi-embodiment learning and provide insights into how architectural and training design choices influence policy generalization. 

---
# RoboCulture: A Robotics Platform for Automated Biological Experimentation 

**Title (ZH)**: RoboCulture: 一种自动化生物实验的机器人平台 

**Authors**: Kevin Angers, Kourosh Darvish, Naruki Yoshikawa, Sargol Okhovatian, Dawn Bannerman, Ilya Yakavets, Florian Shkurti, Alán Aspuru-Guzik, Milica Radisic  

**Link**: [PDF](https://arxiv.org/pdf/2505.14941)  

**Abstract**: Automating biological experimentation remains challenging due to the need for millimeter-scale precision, long and multi-step experiments, and the dynamic nature of living systems. Current liquid handlers only partially automate workflows, requiring human intervention for plate loading, tip replacement, and calibration. Industrial solutions offer more automation but are costly and lack the flexibility needed in research settings. Meanwhile, research in autonomous robotics has yet to bridge the gap for long-duration, failure-sensitive biological experiments. We introduce RoboCulture, a cost-effective and flexible platform that uses a general-purpose robotic manipulator to automate key biological tasks. RoboCulture performs liquid handling, interacts with lab equipment, and leverages computer vision for real-time decisions using optical density-based growth monitoring. We demonstrate a fully autonomous 15-hour yeast culture experiment where RoboCulture uses vision and force feedback and a modular behavior tree framework to robustly execute, monitor, and manage experiments. 

**Abstract (ZH)**: 自动化生物实验仍具挑战性，由于需要毫米级精度、长时间的多步骤实验以及活体系统的动态性质。当前的液体处理设备只能部分自动化工作流程，仍需人工干预进行板载入、吸头更换和校准。工业解决方案虽提供更多自动化但成本较高且在研究环境中缺乏灵活性。同时，自主机器人领域的研究尚未解决长期、故障敏感生物实验的需求。我们介绍了RoboCulture，这是一种经济高效且灵活的平台，使用通用机器人操作器来自动化关键的生物任务。RoboCulture执行液体处理、与实验室设备互动，并利用计算机视觉基于光学密度的生长监测进行实时决策。我们展示了长达15小时的自主酿酒酵母培养实验，RoboCulture使用视觉、力反馈和模块化行为树框架以稳健地执行、监控和管理实验。 

---
# Scan, Materialize, Simulate: A Generalizable Framework for Physically Grounded Robot Planning 

**Title (ZH)**: 扫描、Materialize、仿真：一种物理接地的机器人规划通用框架 

**Authors**: Amine Elhafsi, Daniel Morton, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.14938)  

**Abstract**: Autonomous robots must reason about the physical consequences of their actions to operate effectively in unstructured, real-world environments. We present Scan, Materialize, Simulate (SMS), a unified framework that combines 3D Gaussian Splatting for accurate scene reconstruction, visual foundation models for semantic segmentation, vision-language models for material property inference, and physics simulation for reliable prediction of action outcomes. By integrating these components, SMS enables generalizable physical reasoning and object-centric planning without the need to re-learn foundational physical dynamics. We empirically validate SMS in a billiards-inspired manipulation task and a challenging quadrotor landing scenario, demonstrating robust performance on both simulated domain transfer and real-world experiments. Our results highlight the potential of bridging differentiable rendering for scene reconstruction, foundation models for semantic understanding, and physics-based simulation to achieve physically grounded robot planning across diverse settings. 

**Abstract (ZH)**: 自主机器人必须在其操作过程中对行为的物理后果进行推理，以在非结构化的实际环境中有效运行。我们提出了一个综合框架Scan, Materialize, Simulate (SMS)，该框架结合了3D高斯点云重建以实现准确的场景重建、视觉基础模型以实现语义分割、视觉-语言模型以推断材料属性，以及物理模拟以可靠地预测行为结果。通过将这些组件整合在一起，SMS能够在不重新学习基本物理动力学的情况下实现通用的物理推理和对象中心化规划。我们通过台球启发的操纵任务和具有挑战性的四旋翼降落场景，实证验证了SMS在模拟领域迁移和真实世界实验上的稳健性能。我们的结果突显了将可微渲染用于场景重建、基础模型用于语义理解以及基于物理的模拟相结合，以在各种不同场景中实现物理接地的机器人规划的潜力。 

---
# PCA-DDReach: Efficient Statistical Reachability Analysis of Stochastic Dynamical Systems via Principal Component Analysis 

**Title (ZH)**: PCA-DDReach：通过主成分分析高效统计可达性分析的随机动力学系统方法 

**Authors**: Navid Hashemi, Lars Lindemann, Jyotirmoy Deshmukh  

**Link**: [PDF](https://arxiv.org/pdf/2505.14935)  

**Abstract**: This study presents a scalable data-driven algorithm designed to efficiently address the challenging problem of reachability analysis. Analysis of cyber-physical systems (CPS) relies typically on parametric physical models of dynamical systems. However, identifying parametric physical models for complex CPS is challenging due to their complexity, uncertainty, and variability, often rendering them as black-box oracles. As an alternative, one can treat these complex systems as black-box models and use trajectory data sampled from the system (e.g., from high-fidelity simulators or the real system) along with machine learning techniques to learn models that approximate the underlying dynamics. However, these machine learning models can be inaccurate, highlighting the need for statistical tools to quantify errors. Recent advancements in the field include the incorporation of statistical uncertainty quantification tools such as conformal inference (CI) that can provide probabilistic reachable sets with provable guarantees. Recent work has even highlighted the ability of these tools to address the case where the distribution of trajectories sampled during training time are different from the distribution of trajectories encountered during deployment time. However, accounting for such distribution shifts typically results in more conservative guarantees. This is undesirable in practice and motivates us to present techniques that can reduce conservatism. Here, we propose a new approach that reduces conservatism and improves scalability by combining conformal inference with Principal Component Analysis (PCA). We show the effectiveness of our technique on various case studies, including a 12-dimensional quadcopter and a 27-dimensional hybrid system known as the powertrain. 

**Abstract (ZH)**: 基于配准推理与主成分分析的可达性分析可扩展算法研究 

---
# Think, Reflect, Create: Metacognitive Learning for Zero-Shot Robotic Planning with LLMs 

**Title (ZH)**: 思考、反思、创造：基于大语言模型的零样本机器人规划元认知学习 

**Authors**: Wenjie Lin, Jin Wei-Kocsis  

**Link**: [PDF](https://arxiv.org/pdf/2505.14899)  

**Abstract**: While large language models (LLMs) have shown great potential across various domains, their applications in robotics remain largely limited to static, prompt-based behaviors and still face challenges in handling complex tasks under zero-shot or few-shot settings. Inspired by human metacognitive learning and creative problem-solving, we address this limitation by exploring a fundamental research question: Can LLMs be empowered with metacognitive capabilities to reason, reflect, and create, thereby enhancing their ability to perform robotic tasks with minimal demonstrations? In this paper, we present an early-stage framework that integrates metacognitive learning into LLM-powered multi-robot collaboration. The proposed framework equips the LLM-powered robotic agents with a skill decomposition and self-reflection mechanism that identifies modular skills from prior tasks, reflects on failures in unseen task scenarios, and synthesizes effective new solutions. Experimental results show that our metacognitive-learning-empowered LLM framework significantly outperforms existing baselines. Moreover, we observe that the framework is capable of generating solutions that differ from the ground truth yet still successfully complete the tasks. These exciting findings support our hypothesis that metacognitive learning can foster creativity in robotic planning. 

**Abstract (ZH)**: 大型语言模型在机器人领域的元认知能力增强及其应用：一种初步框架的研究 

---
# UPTor: Unified 3D Human Pose Dynamics and Trajectory Prediction for Human-Robot Interaction 

**Title (ZH)**: UPTor: 统一的3D人类姿态动力学和轨迹预测方法用于人机交互 

**Authors**: Nisarga Nilavadi, Andrey Rudenko, Timm Linder  

**Link**: [PDF](https://arxiv.org/pdf/2505.14866)  

**Abstract**: We introduce a unified approach to forecast the dynamics of human keypoints along with the motion trajectory based on a short sequence of input poses. While many studies address either full-body pose prediction or motion trajectory prediction, only a few attempt to merge them. We propose a motion transformation technique to simultaneously predict full-body pose and trajectory key-points in a global coordinate frame. We utilize an off-the-shelf 3D human pose estimation module, a graph attention network to encode the skeleton structure, and a compact, non-autoregressive transformer suitable for real-time motion prediction for human-robot interaction and human-aware navigation. We introduce a human navigation dataset ``DARKO'' with specific focus on navigational activities that are relevant for human-aware mobile robot navigation. We perform extensive evaluation on Human3.6M, CMU-Mocap, and our DARKO dataset. In comparison to prior work, we show that our approach is compact, real-time, and accurate in predicting human navigation motion across all datasets. Result animations, our dataset, and code will be available at this https URL 

**Abstract (ZH)**: 我们提出了一种统一的方法，基于短序列输入姿态来预测人类关键点的动力学及其运动轨迹。我们提出了一种运动变换技术，同时在全局坐标框架中预测全身姿态和轨迹关键点。我们利用现成的3D人体姿态估计模块，采用图注意力网络编码骨架结构，以及适合实时运动预测的紧凑型非自回归变压器，用于人类机器人交互和人类意识导航。我们引入了一个专注于与人类意识移动机器人导航相关的导航活动的数据集“DARKO”。我们在Human3.6M、CMU-Mocap和我们的DARKO数据集上进行了广泛评估。与此前的工作相比，我们的方法在所有数据集上预测人类导航运动时表现出紧凑性、实时性和准确性。结果动画、数据集和代码将在此链接处提供。 

---
# A Hierarchical Graph-Based Terrain-Aware Autonomous Navigation Approach for Complementary Multimodal Ground-Aerial Exploration 

**Title (ZH)**: 基于分层图的地形感知自主导航方法：互补多模态地面-空中探索 

**Authors**: Akash Patel, Mario A.V. Saucedo, Nikolaos Stathoulopoulos, Viswa Narayanan Sankaranarayanan, Ilias Tevetzidis, Christoforos Kanellakis, George Nikolakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.14859)  

**Abstract**: Autonomous navigation in unknown environments is a fundamental challenge in robotics, particularly in coordinating ground and aerial robots to maximize exploration efficiency. This paper presents a novel approach that utilizes a hierarchical graph to represent the environment, encoding both geometric and semantic traversability. The framework enables the robots to compute a shared confidence metric, which helps the ground robot assess terrain and determine when deploying the aerial robot will extend exploration. The robot's confidence in traversing a path is based on factors such as predicted volumetric gain, path traversability, and collision risk. A hierarchy of graphs is used to maintain an efficient representation of traversability and frontier information through multi-resolution maps. Evaluated in a real subterranean exploration scenario, the approach allows the ground robot to autonomously identify zones that are no longer traversable but suitable for aerial deployment. By leveraging this hierarchical structure, the ground robot can selectively share graph information on confidence-assessed frontier targets from parts of the scene, enabling the aerial robot to navigate beyond obstacles and continue exploration. 

**Abstract (ZH)**: 自主导航于未知环境是机器人技术中的一个基本挑战，特别是在协调地面和空中机器人以最大化探索效率方面。本论文提出了一种新颖的方法，利用层次图表示环境，同时编码几何和语义通行性。该框架允许机器人计算共享的信心度量，帮助地面机器人评估地形并确定何时部署空中机器人以扩展探索。机器人通过预测体素增益、路径通行性和碰撞风险等因素来评估通过路径的信心。通过使用层次图，该框架能够通过多分辨率地图维持高效的通行性和前沿信息表示。在实际地下探索场景中评估该方法，允许地面机器人自主识别不再可通行但适合空中部署的区域。通过利用这一层次结构，地面机器人可以选择性地在场景的部分区域共享基于信心评估的前沿目标信息图，使空中机器人能够导航越过障碍并继续探索。 

---
# Coordinated motion control of a wire arc additive manufacturing robotic system for multi-directional building parts 

**Title (ZH)**: 多方向建筑部件用丝弧增材制造机器人系统的协调运动控制 

**Authors**: Fernando Coutinho, Nicolas Lizarralde, Fernando Lizarralde  

**Link**: [PDF](https://arxiv.org/pdf/2505.14858)  

**Abstract**: This work investigates the manufacturing of complex shapes parts with wire arc additive manufacturing (WAAM). In order to guarantee the integrity and quality of each deposited layer that composes the final piece, the deposition process is usually carried out in a flat position. However, for complex geometry parts with non-flat surfaces, this strategy causes unsupported overhangs and staircase effect, which contribute to a poor surface finishing. Generally, the build direction is not constant for every deposited section or layer in complex geometry parts. As a result, there is an additional concern to ensure the build direction is aligned with gravity, thus improving the quality of the final part. This paper proposes an algorithm to control the torch motion with respect to a deposition substrate as well as the torch orientation with respect to an inertial frame. The control scheme is based on task augmentation applied to an extended kinematic chain composed by two robots, which constitutes a coordinated control problem, and allows the deposition trajectory to be planned with respect to the deposition substrate coordinate frame while aligning each layer buildup direction with gravity (or any other direction defined for an inertial frame). Parts with complex geometry aspects have been produced in a WAAM cell composed by two robots (a manipulator with a welding torch and a positioning table holding the workpiece) in order to validate the proposed approach. 

**Abstract (ZH)**: 这项工作研究了使用丝弧增材制造（WAAM）制造复杂形状部件的方法。为了保证构成最终产品的每一层沉积的完整性和质量，沉积过程通常在平面上进行。然而，对于具有非平面表面的复杂几何形状部件，这种策略会导致无法支撑的悬挑和阶梯效应，从而导致表面质量较差。通常，复杂几何形状部件的构建方向对于每一层沉积都不是恒定的。因此，需要确保构建方向与重力对齐，以提高最终部件的质量。本文提出了一种算法，用于控制相对于沉积基底的焊接炬运动以及相对于惯性参考系的焊接炬方向。控制方案基于应用到由两个机器人组成的扩展运动链上的任务扩展，构成一个协调控制问题，允许根据沉积基底坐标系规划沉积轨迹，并使每一层的堆叠方向与重力（或其他定义的惯性参考系方向）保持对齐。已在一个由两个机器人组成的WAAM系统（一个带有焊接炬的搬运机器人和一个承载工件的定位台）中制造具有复杂几何特征的部件，以验证所提出的方法。 

---
# DORA: Object Affordance-Guided Reinforcement Learning for Dexterous Robotic Manipulation 

**Title (ZH)**: DORA：物体功能导向的灵巧机器人 manipulation 强化学习 

**Authors**: Lei Zhang, Soumya Mondal, Zhenshan Bing, Kaixin Bai, Diwen Zheng, Zhaopeng Chen, Alois Christian Knoll, Jianwei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14819)  

**Abstract**: Dexterous robotic manipulation remains a longstanding challenge in robotics due to the high dimensionality of control spaces and the semantic complexity of object interaction. In this paper, we propose an object affordance-guided reinforcement learning framework that enables a multi-fingered robotic hand to learn human-like manipulation strategies more efficiently. By leveraging object affordance maps, our approach generates semantically meaningful grasp pose candidates that serve as both policy constraints and priors during training. We introduce a voting-based grasp classification mechanism to ensure functional alignment between grasp configurations and object affordance regions. Furthermore, we incorporate these constraints into a generalizable RL pipeline and design a reward function that unifies affordance-awareness with task-specific objectives. Experimental results across three manipulation tasks - cube grasping, jug grasping and lifting, and hammer use - demonstrate that our affordance-guided approach improves task success rates by an average of 15.4% compared to baselines. These findings highlight the critical role of object affordance priors in enhancing sample efficiency and learning generalizable, semantically grounded manipulation policies. For more details, please visit our project website this https URL. 

**Abstract (ZH)**: 灵巧的机器人操作由于控制空间的高维度性和对象交互的语义复杂性一直是一个长期的挑战。本文提出一种对象功能引导的强化学习框架，使多指机器人手能够更高效地学习人类般的操作策略。通过利用对象功能图，我们的方法生成语义上具有意义的抓取姿态候选，它们作为训练过程中的策略约束和先验条件。我们引入了一种基于投票的抓取分类机制，以确保抓取配置的功能与对象功能区域之间的一致性。此外，我们将这些约束整合到一个可泛化的RL管道中，并设计了一个奖励函数，该函数结合了功能意识与特定任务目标。在三个操作任务——立方体抓取、壶抓取与提升、以及锤子使用——上的实验结果表明，与基线方法相比，我们的对象功能引导方法的任务成功率平均提高了15.4%。这些发现突显了对象功能先验在提高样本效率和学习泛化、语义上具意义的操作策略中的关键作用。更多详情，请访问我们的项目网站：this https URL。 

---
# Integrating Field of View in Human-Aware Collaborative Planning 

**Title (ZH)**: 将视野范围整合到人类意识协同规划中 

**Authors**: Ya-Chuan Hsu, Michael Defranco, Rutvik Patel, Stefanos Nikolaidis  

**Link**: [PDF](https://arxiv.org/pdf/2505.14805)  

**Abstract**: In human-robot collaboration (HRC), it is crucial for robot agents to consider humans' knowledge of their surroundings. In reality, humans possess a narrow field of view (FOV), limiting their perception. However, research on HRC often overlooks this aspect and presumes an omniscient human collaborator. Our study addresses the challenge of adapting to the evolving subtask intent of humans while accounting for their limited FOV. We integrate FOV within the human-aware probabilistic planning framework. To account for large state spaces due to considering FOV, we propose a hierarchical online planner that efficiently finds approximate solutions while enabling the robot to explore low-level action trajectories that enter the human FOV, influencing their intended subtask. Through user study with our adapted cooking domain, we demonstrate our FOV-aware planner reduces human's interruptions and redundant actions during collaboration by adapting to human perception limitations. We extend these findings to a virtual reality kitchen environment, where we observe similar collaborative behaviors. 

**Abstract (ZH)**: 在人机协作（HRC）中，机器人代理考虑人类对环境的知识至关重要。实际上，人类具有狭窄的视场（FOV），限制了其感知能力。然而，HRC相关的研究往往忽视了这一方面，并假设人类合作者无所不知。我们的研究致力于在考虑人类有限FOV的情况下，适应人类不断演变的任务意图的挑战。我们将在人类意识的概率规划框架中整合FOV。为了解决由于考虑FOV而导致的大型状态空间问题，我们提出了一个分层在线规划器，在高效寻找近似解的同时，使机器人能够探索进入人类FOV的低级动作轨迹，进而影响其预期的子任务。通过在我们调整后的烹饪领域中的用户研究，我们证明了FOV意识的规划器通过适应人类感知限制，减少了人类在协作过程中的中断和冗余动作。我们将这些发现扩展到虚拟现实厨房环境中，观察到了相似的合作行为。 

---
# Fast and scalable multi-robot deployment planning under connectivity constraints 

**Title (ZH)**: 基于连接约束的快速可扩展多机器人部署规划 

**Authors**: Yaroslav Marchukov, Luis Montano  

**Link**: [PDF](https://arxiv.org/pdf/2505.14760)  

**Abstract**: In this paper we develop a method to coordinate the deployment of a multi-robot team to reach some locations of interest, so-called primary goals, and to transmit the information from these positions to a static Base Station (BS), under connectivity constraints. The relay positions have to be established for some robots to maintain the connectivity at the moment in which the other robots visit the primary goals. Once every robot reaches its assigned goal, they are again available to cover new goals, dynamically re-distributing the robots to the new tasks. The contribution of this work is a two stage method to deploy the team. Firstly, clusters of relay and primary positions are computed, obtaining a tree formed by chains of positions that have to be visited. Secondly, the order for optimally assigning and visiting the goals in the clusters is computed. We analyze different heuristics for sequential and parallel deployment in the clusters, obtaining sub-optimal solutions in short time for different number of robots and for a large amount of goals. 

**Abstract (ZH)**: 本文提出了一种方法，用于协调多机器人团队部署，以到达某些感兴趣的地点（即主要目标），并从这些位置将信息传输到一个固定的基站点（BS），同时满足连通性约束。需要为某些机器人确定中继位置，在其他机器人访问主要目标的时刻保持连通性。一旦每个机器人到达其分配的目标，他们将重新分配以覆盖新的目标，动态重新分布机器人以执行新任务。本文的贡献是一种两阶段方法来部署团队。首先，计算中继和主要位置的集群，得到由路径链接的位置组成的树形结构，这些位置需要被访问。其次，计算集群中目标最优指派和访问顺序。我们分析了在集群中进行顺序和并行部署的不同启发式方法，获得在不同机器人数量和大量目标情况下较优解的近似解。 

---
# Improving planning and MBRL with temporally-extended actions 

**Title (ZH)**: 改进规划与基于模型的强化学习中的时间延伸动作 

**Authors**: Palash Chatterjee, Roni Khardon  

**Link**: [PDF](https://arxiv.org/pdf/2505.15754)  

**Abstract**: Continuous time systems are often modeled using discrete time dynamics but this requires a small simulation step to maintain accuracy. In turn, this requires a large planning horizon which leads to computationally demanding planning problems and reduced performance. Previous work in model free reinforcement learning has partially addressed this issue using action repeats where a policy is learned to determine a discrete action duration. Instead we propose to control the continuous decision timescale directly by using temporally-extended actions and letting the planner treat the duration of the action as an additional optimization variable along with the standard action variables. This additional structure has multiple advantages. It speeds up simulation time of trajectories and, importantly, it allows for deep horizon search in terms of primitive actions while using a shallow search depth in the planner. In addition, in the model based reinforcement learning (MBRL) setting, it reduces compounding errors from model learning and improves training time for models. We show that this idea is effective and that the range for action durations can be automatically selected using a multi-armed bandit formulation and integrated into the MBRL framework. An extensive experimental evaluation both in planning and in MBRL, shows that our approach yields faster planning, better solutions, and that it enables solutions to problems that are not solved in the standard formulation. 

**Abstract (ZH)**: 连续时间系统通常使用离散时间动态模型，但这需要小时间步长以保持准确性，从而导致较大的规划时间 horizon，增加计算需求并降低性能。无模型强化学习的前期工作部分解决了这个问题，通过动作重复学习一个确定离散动作持续时间的策略。相反，我们提出直接控制连续决策时间尺度，通过使用时间扩展动作并让规划者将动作持续时间作为额外的优化变量处理，与标准动作变量一起处理。这种额外的结构具有多个优势。它加快了轨迹的模拟时间，并且重要的是，它允许在基础动作方面进行深度前景搜索，而在规划者方面则使用浅层搜索深度。此外，在模型基于强化学习（MBRL）设置中，它可以减少模型学习中的累积误差并提高模型的训练时间。我们表明此方法有效，并且可以使用多臂bandit形式自动选择动作持续时间的范围，并将其集成到MBRL框架中。广泛的实验评估表明，我们的方法可以实现更快的规划、更好的解决方案，并且能够解决标准表述无法解决的问题。 

---
# World Models as Reference Trajectories for Rapid Motor Adaptation 

**Title (ZH)**: 世界模型作为快速运动适应的参考轨迹 

**Authors**: Carlos Stein Brito, Daniel McNamee  

**Link**: [PDF](https://arxiv.org/pdf/2505.15589)  

**Abstract**: Deploying learned control policies in real-world environments poses a fundamental challenge. When system dynamics change unexpectedly, performance degrades until models are retrained on new data. We introduce Reflexive World Models (RWM), a dual control framework that uses world model predictions as implicit reference trajectories for rapid adaptation. Our method separates the control problem into long-term reward maximization through reinforcement learning and robust motor execution through rapid latent control. This dual architecture achieves significantly faster adaptation with low online computational cost compared to model-based RL baselines, while maintaining near-optimal performance. The approach combines the benefits of flexible policy learning through reinforcement learning with rapid error correction capabilities, providing a principled approach to maintaining performance in high-dimensional continuous control tasks under varying dynamics. 

**Abstract (ZH)**: Reflexive世界模型：一种用于快速适应的双控制框架 

---
# Guided Policy Optimization under Partial Observability 

**Title (ZH)**: 部分可观测性下的引导策略优化 

**Authors**: Yueheng Li, Guangming Xie, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15418)  

**Abstract**: Reinforcement Learning (RL) in partially observable environments poses significant challenges due to the complexity of learning under uncertainty. While additional information, such as that available in simulations, can enhance training, effectively leveraging it remains an open problem. To address this, we introduce Guided Policy Optimization (GPO), a framework that co-trains a guider and a learner. The guider takes advantage of privileged information while ensuring alignment with the learner's policy that is primarily trained via imitation learning. We theoretically demonstrate that this learning scheme achieves optimality comparable to direct RL, thereby overcoming key limitations inherent in existing approaches. Empirical evaluations show strong performance of GPO across various tasks, including continuous control with partial observability and noise, and memory-based challenges, significantly outperforming existing methods. 

**Abstract (ZH)**: 部分可观测环境中强化学习（RL）由于在不确定性下的学习复杂性而面临重大挑战。虽然额外的信息，如模拟中可用的信息，可以增强训练，但有效利用这些信息仍然是一个开放问题。为了解决这一问题，我们提出了一种引导策略优化（GPO）框架，该框架通过共同训练一个引导器和一个学习器来利用额外信息，同时确保引导器与主要通过模仿学习训练的学习器策略保持一致。理论上证明，这种学习方案在实现直接RL相当的最优性方面克服了现有方法的关键局限。实证评估显示，GPO在各种任务中表现强劲，包括连续控制下的部分可观测性和噪声挑战以及基于记忆的挑战，显著优于现有方法。 

---
# RAZER: Robust Accelerated Zero-Shot 3D Open-Vocabulary Panoptic Reconstruction with Spatio-Temporal Aggregation 

**Title (ZH)**: RAZER：鲁棒加速零样本三维开放词汇全景重建方法及其时空聚合 

**Authors**: Naman Patel, Prashanth Krishnamurthy, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2505.15373)  

**Abstract**: Mapping and understanding complex 3D environments is fundamental to how autonomous systems perceive and interact with the physical world, requiring both precise geometric reconstruction and rich semantic comprehension. While existing 3D semantic mapping systems excel at reconstructing and identifying predefined object instances, they lack the flexibility to efficiently build semantic maps with open-vocabulary during online operation. Although recent vision-language models have enabled open-vocabulary object recognition in 2D images, they haven't yet bridged the gap to 3D spatial understanding. The critical challenge lies in developing a training-free unified system that can simultaneously construct accurate 3D maps while maintaining semantic consistency and supporting natural language interactions in real time. In this paper, we develop a zero-shot framework that seamlessly integrates GPU-accelerated geometric reconstruction with open-vocabulary vision-language models through online instance-level semantic embedding fusion, guided by hierarchical object association with spatial indexing. Our training-free system achieves superior performance through incremental processing and unified geometric-semantic updates, while robustly handling 2D segmentation inconsistencies. The proposed general-purpose 3D scene understanding framework can be used for various tasks including zero-shot 3D instance retrieval, segmentation, and object detection to reason about previously unseen objects and interpret natural language queries. The project page is available at this https URL. 

**Abstract (ZH)**: 三维环境的映射与理解对于自主系统感知和交互物理世界至关重要，要求精确的几何重构和丰富的语义理解。现有的三维语义映射系统在重建和识别预定义对象实例方面表现出色，但在在线运行过程中缺乏高效构建开放式词汇语义地图的灵活性。尽管最近的视觉-语言模型已经在2D图像中实现了开放式词汇的物体识别，但尚未解决到三维空间理解的鸿沟。关键挑战在于开发一个无需训练的统一系统，能够同时构建精确的3D地图，保持语义一致性，并支持实时的自然语言交互。本文提出了一种零样本框架，通过基于层次化物体关联和空间索引的在线实例级语义嵌入融合，无缝集成GPU加速的几何重构与开放式词汇的视觉-语言模型。无需训练的系统通过增量处理和统一的几何-语义更新实现优异性能，同时稳健地处理2D分割一致性问题。所提出的一般用途的3D场景理解框架可用于包括零样本3D实例检索、分割和物体检测在内的多种任务，以推断未见过的物体并解释自然语言查询。项目页面可访问该网址。 

---
# R3GS: Gaussian Splatting for Robust Reconstruction and Relocalization in Unconstrained Image Collections 

**Title (ZH)**: R3GS：稳健重建与无约束图像集合中重定位的高斯点云方法 

**Authors**: Xu yan, Zhaohui Wang, Rong Wei, Jingbo Yu, Dong Li, Xiangde Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15294)  

**Abstract**: We propose R3GS, a robust reconstruction and relocalization framework tailored for unconstrained datasets. Our method uses a hybrid representation during training. Each anchor combines a global feature from a convolutional neural network (CNN) with a local feature encoded by the multiresolution hash grids [2]. Subsequently, several shallow multi-layer perceptrons (MLPs) predict the attributes of each Gaussians, including color, opacity, and covariance. To mitigate the adverse effects of transient objects on the reconstruction process, we ffne-tune a lightweight human detection network. Once ffne-tuned, this network generates a visibility map that efffciently generalizes to other transient objects (such as posters, banners, and cars) with minimal need for further adaptation. Additionally, to address the challenges posed by sky regions in outdoor scenes, we propose an effective sky-handling technique that incorporates a depth prior as a constraint. This allows the inffnitely distant sky to be represented on the surface of a large-radius sky sphere, signiffcantly reducing ffoaters caused by errors in sky reconstruction. Furthermore, we introduce a novel relocalization method that remains robust to changes in lighting conditions while estimating the camera pose of a given image within the reconstructed 3DGS scene. As a result, R3GS significantly enhances rendering ffdelity, improves both training and rendering efffciency, and reduces storage requirements. Our method achieves state-of-the-art performance compared to baseline methods on in-the-wild datasets. The code will be made open-source following the acceptance of the paper. 

**Abstract (ZH)**: 一种针对无约束数据集的鲁棒重建与重新定位框架：R3GS 

---
# Toward Informed AV Decision-Making: Computational Model of Well-being and Trust in Mobility 

**Title (ZH)**: 面向知情自动驾驶决策：福祉与移动性信任的计算模型 

**Authors**: Zahra Zahedi, Shashank Mehrotra, Teruhisa Misu, Kumar Akash  

**Link**: [PDF](https://arxiv.org/pdf/2505.14983)  

**Abstract**: For future human-autonomous vehicle (AV) interactions to be effective and smooth, human-aware systems that analyze and align human needs with automation decisions are essential. Achieving this requires systems that account for human cognitive states. We present a novel computational model in the form of a Dynamic Bayesian Network (DBN) that infers the cognitive states of both AV users and other road users, integrating this information into the AV's decision-making process. Specifically, our model captures the well-being of both an AV user and an interacting road user as cognitive states alongside trust. Our DBN models infer beliefs over the AV user's evolving well-being, trust, and intention states, as well as the possible well-being of other road users, based on observed interaction experiences. Using data collected from an interaction study, we refine the model parameters and empirically assess its performance. Finally, we extend our model into a causal inference model (CIM) framework for AV decision-making, enabling the AV to enhance user well-being and trust while balancing these factors with its own operational costs and the well-being of interacting road users. Our evaluation demonstrates the model's effectiveness in accurately predicting user's states and guiding informed, human-centered AV decisions. 

**Abstract (ZH)**: 为了使未来的人工智能驾驶汽车（AV）交互有效且顺畅，需要具备人类意识的系统来分析并协调人类需求与自动化决策，而这要求系统能够考虑人类的认知状态。我们提出了一种新颖的计算模型——动态贝叶斯网络（DBN），用于推断人工智能驾驶汽车用户和其他道路用户的精神状态，并将其信息整合到人工智能驾驶汽车的决策过程中。具体而言，我们的模型将人工智能驾驶汽车用户和交互道路用户的福祉以及信任视为精神状态进行捕获。动态贝叶斯网络模型根据观察到的交互体验推断人工智能驾驶汽车用户的精神状态、信任状态和意图状态，以及其他道路用户可能的精神状态。通过使用交互研究收集的数据，我们优化了模型参数，并对其性能进行了实证评估。最后，我们扩展了该模型，将其纳入因果推理模型（CIM）框架，使得人工智能驾驶汽车在增强用户福祉和信任的同时，能够平衡这些因素与自身的运营成本以及交互道路用户的精神状态。评价结果显示，该模型在准确预测用户状态和指导以人类为中心的人工智能驾驶汽车决策方面具有有效性。 

---
# Flattening Hierarchies with Policy Bootstrapping 

**Title (ZH)**: 层级结构化政策自举化扁平化 

**Authors**: John L. Zhou, Jonathan C. Kao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14975)  

**Abstract**: Offline goal-conditioned reinforcement learning (GCRL) is a promising approach for pretraining generalist policies on large datasets of reward-free trajectories, akin to the self-supervised objectives used to train foundation models for computer vision and natural language processing. However, scaling GCRL to longer horizons remains challenging due to the combination of sparse rewards and discounting, which obscures the comparative advantages of primitive actions with respect to distant goals. Hierarchical RL methods achieve strong empirical results on long-horizon goal-reaching tasks, but their reliance on modular, timescale-specific policies and subgoal generation introduces significant additional complexity and hinders scaling to high-dimensional goal spaces. In this work, we introduce an algorithm to train a flat (non-hierarchical) goal-conditioned policy by bootstrapping on subgoal-conditioned policies with advantage-weighted importance sampling. Our approach eliminates the need for a generative model over the (sub)goal space, which we find is key for scaling to high-dimensional control in large state spaces. We further show that existing hierarchical and bootstrapping-based approaches correspond to specific design choices within our derivation. Across a comprehensive suite of state- and pixel-based locomotion and manipulation benchmarks, our method matches or surpasses state-of-the-art offline GCRL algorithms and scales to complex, long-horizon tasks where prior approaches fail. 

**Abstract (ZH)**: 离线目标条件强化学习（GCRL）：一种适用于大规模奖励为空轨迹预训练的一般性策略的方法，类似于用于训练计算机视觉和自然语言处理基础模型的自监督目标。然而，由于稀疏奖励和折扣率的结合，将GCRL扩展到更长的时间跨度仍然具有挑战性，这掩盖了基本动作与远大目标之间的比较优势。分层强化学习方法在长期目标达成任务上取得了强大的实证结果，但它们依赖于模块化、时间尺度特定的策略和子目标生成，引入了显著的额外复杂性，阻碍了对高维目标空间的扩展。在本文中，我们提出了一种算法，通过基于优势加权的重要性采样从子目标条件策略中进行增量学习来训练一个扁平的目标条件策略。我们的方法消除了对（子）目标空间生成模型的需要，对于在大面积状态空间中的高维控制具有关键意义。此外，我们展示现有分层和基于增量学习的方法是我们推导中的特定设计选择。在一系列全面的基于状态和像素的运动和操作基准测试中，我们的方法在离线GCRL算法中达到或超越了最先进的技术水平，并成功扩展到先前方法无法处理的复杂长时间跨度任务。 

---
