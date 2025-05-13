# Multi-Agent Path Finding via Finite-Horizon Hierarchical Factorization 

**Title (ZH)**: 多智能体路径finding基于有限 horizon 分级因子分解 

**Authors**: Jiarui Li, Alessandro Zanardi, Gioele Zardini  

**Link**: [PDF](https://arxiv.org/pdf/2505.07779)  

**Abstract**: We present a novel algorithm for large-scale Multi-Agent Path Finding (MAPF) that enables fast, scalable planning in dynamic environments such as automated warehouses. Our approach introduces finite-horizon hierarchical factorization, a framework that plans one step at a time in a receding-horizon fashion. Robots first compute individual plans in parallel, and then dynamically group based on spatio-temporal conflicts and reachability. The framework accounts for conflict resolution, and for immediate execution and concurrent planning, significantly reducing response time compared to offline algorithms. Experimental results on benchmark maps demonstrate that our method achieves up to 60% reduction in time-to-first-action while consistently delivering high-quality solutions, outperforming state-of-the-art offline baselines across a range of problem sizes and planning horizons. 

**Abstract (ZH)**: 我们提出了一种新型的大规模多Agent路径规划算法，能够在包括自动化仓库在内的动态环境中实现快速、可扩展的规划。该方法引入了一种有限 horizon 的层次分解框架，以回溯 horizon 的方式逐步进行规划。机器人首先并行计算各自的路径计划，然后根据时空冲突和可达性动态分组。该框架考虑了解决冲突、即时执行和并发规划，显著减少了响应时间，与离线算法相比。基准地图上的实验结果表明，我们的方法在不牺牲高质量解决方案的前提下，将首次行动所需时间最多减少了60%，并在多种问题规模和规划 horizon 下优于最先进的离线基准方法。 

---
# Guiding Data Collection via Factored Scaling Curves 

**Title (ZH)**: 基于因子缩放曲线指导数据收集 

**Authors**: Lihan Zha, Apurva Badithela, Michael Zhang, Justin Lidard, Jeremy Bao, Emily Zhou, David Snyder, Allen Z. Ren, Dhruv Shah, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.07728)  

**Abstract**: Generalist imitation learning policies trained on large datasets show great promise for solving diverse manipulation tasks. However, to ensure generalization to different conditions, policies need to be trained with data collected across a large set of environmental factor variations (e.g., camera pose, table height, distractors) $-$ a prohibitively expensive undertaking, if done exhaustively. We introduce a principled method for deciding what data to collect and how much to collect for each factor by constructing factored scaling curves (FSC), which quantify how policy performance varies as data scales along individual or paired factors. These curves enable targeted data acquisition for the most influential factor combinations within a given budget. We evaluate the proposed method through extensive simulated and real-world experiments, across both training-from-scratch and fine-tuning settings, and show that it boosts success rates in real-world tasks in new environments by up to 26% over existing data-collection strategies. We further demonstrate how factored scaling curves can effectively guide data collection using an offline metric, without requiring real-world evaluation at scale. 

**Abstract (ZH)**: 通用模仿学习政策在大规模数据集上训练后，在解决多样操作任务方面展现出巨大潜力。然而，为了确保在不同条件下的泛化能力，政策需要在环境因素（如相机姿态、桌子高度、干扰物）变化的广泛集合中收集数据——这是一项代价高昂的工程，如果要穷尽所有可能性的话。我们提出了一种原理性的方法，用于决定收集哪些数据以及每种因素收集多少数据，通过构建因子尺度曲线（FSC），量化随单一因素或配对因素数据规模变化时政策性能的变化。这些曲线使人们能够在预算范围内针对最有影响力的因素组合进行有目标的数据采集。通过广泛的模拟和实际实验，我们评估了所提出的方法，并在从零开始训练和微调的不同设置下展示了它在新环境中提高了高达26%的成功率。我们进一步证明了如何使用离线指标引导数据收集，而无需大规模的真实世界评估。 

---
# DATAMUt: Deterministic Algorithms for Time-Delay Attack Detection in Multi-Hop UAV Networks 

**Title (ZH)**: DATAMMut：多跳无人机网络中确定性时间延迟攻击检测算法 

**Authors**: Keiwan Soltani, Federico Corò, Punyasha Chatterjee, Sajal K. Das  

**Link**: [PDF](https://arxiv.org/pdf/2505.07670)  

**Abstract**: Unmanned Aerial Vehicles (UAVs), also known as drones, have gained popularity in various fields such as agriculture, emergency response, and search and rescue operations. UAV networks are susceptible to several security threats, such as wormhole, jamming, spoofing, and false data injection. Time Delay Attack (TDA) is a unique attack in which malicious UAVs intentionally delay packet forwarding, posing significant threats, especially in time-sensitive applications. It is challenging to distinguish malicious delay from benign network delay due to the dynamic nature of UAV networks, intermittent wireless connectivity, or the Store-Carry-Forward (SCF) mechanism during multi-hop communication. Some existing works propose machine learning-based centralized approaches to detect TDA, which are computationally intensive and have large message overheads. This paper proposes a novel approach DATAMUt, where the temporal dynamics of the network are represented by a weighted time-window graph (TWiG), and then two deterministic polynomial-time algorithms are presented to detect TDA when UAVs have global and local network knowledge. Simulation studies show that the proposed algorithms have reduced message overhead by a factor of five and twelve in global and local knowledge, respectively, compared to existing approaches. Additionally, our approaches achieve approximately 860 and 1050 times less execution time in global and local knowledge, respectively, outperforming the existing methods. 

**Abstract (ZH)**: 无人机（UAVs）在农业、应急响应和搜救等各个领域受欢迎。UAV网络面临多重安全威胁，如 wormhole、jamming、spoofing 和 false data injection。时间延迟攻击（TDA）是一种恶意UAV故意延迟数据包转发的独特攻击，尤其在时间敏感应用中构成重大威胁。由于无人机网络的动态性、间歇性无线连接或多跳通信中的Store-Carry-Forward (SCF)机制，区分恶意延迟与良性的网络延迟具有挑战性。已有工作提出基于机器学习的集中式方法来检测TDA，这些方法计算密集且消息开销大。本文提出了一种新颖的方法DATAMUt，其中通过加权时间窗口图（TWiG）表示网络的时间动态，并提出了两种确定性的多项式时间算法，在无人机具有全局和局部网络知识时检测TDA。仿真研究表明，与现有方法相比，所提出算法在全局知识和局部知识下的消息开销分别减少了5倍和12倍。此外，在全局和局部知识下，我们的方法的执行时间分别减少了约860倍和1050倍，优于现有方法。 

---
# CHD: Coupled Hierarchical Diffusion for Long-Horizon Tasks 

**Title (ZH)**: CHD: 耦合层级扩散模型用于长期任务 

**Authors**: Ce Hao, Anxing Xiao, Zhiwei Xue, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2505.07261)  

**Abstract**: Diffusion-based planners have shown strong performance in short-horizon tasks but often fail in complex, long-horizon settings. We trace the failure to loose coupling between high-level (HL) sub-goal selection and low-level (LL) trajectory generation, which leads to incoherent plans and degraded performance. We propose Coupled Hierarchical Diffusion (CHD), a framework that models HL sub-goals and LL trajectories jointly within a unified diffusion process. A shared classifier passes LL feedback upstream so that sub-goals self-correct while sampling proceeds. This tight HL-LL coupling improves trajectory coherence and enables scalable long-horizon diffusion planning. Experiments across maze navigation, tabletop manipulation, and household environments show that CHD consistently outperforms both flat and hierarchical diffusion baselines. 

**Abstract (ZH)**: 基于扩散的规划在短时域任务中表现出强大的性能，但在复杂、长时域设置中经常失效。我们追踪失败原因归结为高层（HL）子目标选择与低层（LL）轨迹生成之间的松散耦合，导致不一致的计划和性能下降。我们提出了一种耦合层次扩散（CHD）框架，该框架在统一的扩散过程中联合建模高层子目标和低层轨迹。共享分类器将低层反馈传递至上层，使子目标在采样过程中自我修正。这种紧密的 HL-LL 耦合提高了轨迹的一致性，并使长时域扩散规划更具扩展性。跨迷宫导航、桌面操作和家庭环境的实验表明，CHD 一致地优于平面和层次扩散基准。 

---
# Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving 

**Title (ZH)**: 边界引导的道路感知与物理可行的自主驾驶轨迹预测 

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06740)  

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66\% to just 1\%. These results highlight the effectiveness of our approach in generating feasible and robust predictions. 

**Abstract (ZH)**: 准确预测周围道路用户的轨迹对于实现安全高效的自动驾驶至关重要。尽管深度学习模型已有改进，但仍存在防止道路外预测和确保动力学可行性的挑战。现有方法整合了道路意识模块并施加动力学约束，但缺乏可行性保证，并且常在复杂性和灵活性之间引入权衡。本文提出了一种新型框架，将轨迹预测公式化为由允许的驾驶方向及其边界引导的约束回归问题。利用代理当前状态和高精度地图，我们的方法定义有效边界并确保道路内预测，通过训练网络学习边界多边线间的叠加路径。为了保证可行性，模型预测加速度分布，这些加速度分布确定车辆沿这些路径的行驶距离，并遵守动力学约束。我们通过Argoverse-2数据集将我们的方法与HPTR基线进行对比评估。虽然我们的方法在基准指标上略有下降，但在最终位移误差和去除不可行轨迹方面表现优异。此外，所提出的方法在不常见的操作和未见过的分布外场景中表现出更优的泛化能力，将对抗攻击下的道路外率从66%降至仅1%。这些结果突显了我们方法在生成可行和稳健预测方面的有效性。 

---
# Motion Planning for Autonomous Vehicles: When Model Predictive Control Meets Ensemble Kalman Smoothing 

**Title (ZH)**: 自主车辆的运动规划：模型预测控制与集成卡尔曼平滑相结合 

**Authors**: Iman Askari, Yebin Wang, Vedeng M. Deshpande, Huazhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06666)  

**Abstract**: Safe and efficient motion planning is of fundamental importance for autonomous vehicles. This paper investigates motion planning based on nonlinear model predictive control (NMPC) over a neural network vehicle model. We aim to overcome the high computational costs that arise in NMPC of the neural network model due to the highly nonlinear and nonconvex optimization. In a departure from numerical optimization solutions, we reformulate the problem of NMPC-based motion planning as a Bayesian estimation problem, which seeks to infer optimal planning decisions from planning objectives. Then, we use a sequential ensemble Kalman smoother to accomplish the estimation task, exploiting its high computational efficiency for complex nonlinear systems. The simulation results show an improvement in computational speed by orders of magnitude, indicating the potential of the proposed approach for practical motion planning. 

**Abstract (ZH)**: 基于神经网络车辆模型的非线性模型预测控制的运动规划：一种贝叶斯估计方法的研究 

---
# Towards Accurate State Estimation: Kalman Filter Incorporating Motion Dynamics for 3D Multi-Object Tracking 

**Title (ZH)**: 基于运动动力学融合的卡尔曼滤波器用于三维多目标跟踪的精确状态估计 

**Authors**: Mohamed Nagy, Naoufel Werghi, Bilal Hassan, Jorge Dias, Majid Khonji  

**Link**: [PDF](https://arxiv.org/pdf/2505.07254)  

**Abstract**: This work addresses the critical lack of precision in state estimation in the Kalman filter for 3D multi-object tracking (MOT) and the ongoing challenge of selecting the appropriate motion model. Existing literature commonly relies on constant motion models for estimating the states of objects, neglecting the complex motion dynamics unique to each object. Consequently, trajectory division and imprecise object localization arise, especially under occlusion conditions. The core of these challenges lies in the limitations of the current Kalman filter formulation, which fails to account for the variability of motion dynamics as objects navigate their environments. This work introduces a novel formulation of the Kalman filter that incorporates motion dynamics, allowing the motion model to adaptively adjust according to changes in the object's movement. The proposed Kalman filter substantially improves state estimation, localization, and trajectory prediction compared to the traditional Kalman filter. This is reflected in tracking performance that surpasses recent benchmarks on the KITTI and Waymo Open Datasets, with margins of 0.56\% and 0.81\% in higher order tracking accuracy (HOTA) and multi-object tracking accuracy (MOTA), respectively. Furthermore, the proposed Kalman filter consistently outperforms the baseline across various detectors. Additionally, it shows an enhanced capability in managing long occlusions compared to the baseline Kalman filter, achieving margins of 1.22\% in higher order tracking accuracy (HOTA) and 1.55\% in multi-object tracking accuracy (MOTA) on the KITTI dataset. The formulation's efficiency is evident, with an additional processing time of only approximately 0.078 ms per frame, ensuring its applicability in real-time applications. 

**Abstract (ZH)**: 针对三维多目标跟踪中的卡尔曼滤波器状态估计精度不足和适用运动模型选择的持续挑战，本文提出了一种新的卡尔曼滤波器公式，该公式整合了运动动力学，使运动模型能够根据目标运动变化进行自适应调整。所提出的卡尔曼滤波器在状态估计、定位和轨迹预测方面明显优于传统的卡尔曼滤波器，并在Kitti和Waymo开放数据集上的跟踪性能上超越了最近的基准，分别在高阶跟踪准确性(HOTA)和多目标跟踪准确性(MOTA)方面提高了0.56%和0.81%。此外，所提出的卡尔曼滤波器在各种检测器上的性能持续优于基准，特别是在管理长时间遮挡方面表现更优，在Kitti数据集上的HOTA和MOTA分别提高了1.22%和1.55%。该公式效率高，每帧额外处理时间为约0.078 ms，确保其适用于实时应用。 

---
# Beyond Patterns: Harnessing Causal Logic for Autonomous Driving Trajectory Prediction 

**Title (ZH)**: 超越模式：利用因果逻辑进行自主驾驶轨迹预测 

**Authors**: Bonan Wang, Haicheng Liao, Chengyue Wang, Bin Rao, Yanchen Guan, Guyang Yu, Jiaxun Zhang, Songning Lai, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06856)  

**Abstract**: Accurate trajectory prediction has long been a major challenge for autonomous driving (AD). Traditional data-driven models predominantly rely on statistical correlations, often overlooking the causal relationships that govern traffic behavior. In this paper, we introduce a novel trajectory prediction framework that leverages causal inference to enhance predictive robustness, generalization, and accuracy. By decomposing the environment into spatial and temporal components, our approach identifies and mitigates spurious correlations, uncovering genuine causal relationships. We also employ a progressive fusion strategy to integrate multimodal information, simulating human-like reasoning processes and enabling real-time inference. Evaluations on five real-world datasets--ApolloScape, nuScenes, NGSIM, HighD, and MoCAD--demonstrate our model's superiority over existing state-of-the-art (SOTA) methods, with improvements in key metrics such as RMSE and FDE. Our findings highlight the potential of causal reasoning to transform trajectory prediction, paving the way for robust AD systems. 

**Abstract (ZH)**: 准确的轨迹预测一直是自动驾驶（AD）领域的重大挑战。传统的数据驱动模型主要依赖于统计相关性，往往忽略了交通行为背后的因果关系。本文提出了一种新的轨迹预测框架，利用因果推断来增强预测的稳健性、泛化能力和准确性。通过将环境分解为时空组件，本方法识别并缓解了虚假相关性，揭示了真实的因果关系。我们还采用渐进融合策略来整合多模态信息，模拟人类推理过程，实现实时推理。在ApolloScape、nuScenes、NGSIM、HighD和MoCAD五个现实世界数据集上的评估表明，我们的模型在关键指标如RMSE和FDE方面优于现有最先进的（SOTA）方法。我们的研究结果突显了因果推理在轨迹预测中的潜力，为稳健的AD系统铺平了道路。 

---
# Work in Progress: Middleware-Transparent Callback Enforcement in Commoditized Component-Oriented Real-time Systems 

**Title (ZH)**: 工作中的进展：中间件透明的回调强制执行在商品化面向组件的实时系统中 

**Authors**: Takahiro Ishikawa-Aso, Atsushi Yano, Takuya Azumi, Shinpei Kato  

**Link**: [PDF](https://arxiv.org/pdf/2505.06546)  

**Abstract**: Real-time scheduling in commoditized component-oriented real-time systems, such as ROS 2 systems on Linux, has been studied under nested scheduling: OS thread scheduling and middleware layer scheduling (e.g., ROS 2 Executor). However, by establishing a persistent one-to-one correspondence between callbacks and OS threads, we can ignore the middleware layer and directly apply OS scheduling parameters (e.g., scheduling policy, priority, and affinity) to individual callbacks. We propose a middleware model that enables this idea and implements CallbackIsolatedExecutor as a novel ROS 2 Executor. We demonstrate that the costs (user-kernel switches, context switches, and memory usage) of CallbackIsolatedExecutor remain lower than those of the MultiThreadedExecutor, regardless of the number of callbacks. Additionally, the cost of CallbackIsolatedExecutor relative to SingleThreadedExecutor stays within a fixed ratio (1.4x for inter-process and 5x for intra-process communication). Future ROS 2 real-time scheduling research can avoid nested scheduling, ignoring the existence of the middleware layer. 

**Abstract (ZH)**: 面向组件导向实时系统的实时调度：在Linux上的ROS 2系统中，通过建立回调与操作系统线程之间持久的一对一对应关系，我们可以在忽略中间件层的基础上，直接将操作系统调度参数（如调度策略、优先级和亲和性）应用于个体回调。我们提出了一种中间件模型，并实现了CallbackIsolatedExecutor作为新型ROS 2执行器。实验表明，无论回调的数量如何，CallbackIsolatedExecutor的成本（用户内核切换、上下文切换和内存使用）都低于MultiThreadedExecutor的成本。此外，相对于SingleThreadedExecutor，CallbackIsolatedExecutor的成本保持在一个固定的比率之内（跨进程通信为1.4倍，进程内通信为5倍）。未来的ROS 2实时调度研究可以避免使用嵌套调度，忽略中间件层的存在。 

---
# Direct Data Driven Control Using Noisy Measurements 

**Title (ZH)**: 直接基于 noisy 测量的数据驱动控制 

**Authors**: Ramin Esmzad, Gokul S. Sankar, Teawon Han, Hamidreza Modares  

**Link**: [PDF](https://arxiv.org/pdf/2505.06407)  

**Abstract**: This paper presents a novel direct data-driven control framework for solving the linear quadratic regulator (LQR) under disturbances and noisy state measurements. The system dynamics are assumed unknown, and the LQR solution is learned using only a single trajectory of noisy input-output data while bypassing system identification. Our approach guarantees mean-square stability (MSS) and optimal performance by leveraging convex optimization techniques that incorporate noise statistics directly into the controller synthesis. First, we establish a theoretical result showing that the MSS of an uncertain data-driven system implies the MSS of the true closed-loop system. Building on this, we develop a robust stability condition using linear matrix inequalities (LMIs) that yields a stabilizing controller gain from noisy measurements. Finally, we formulate a data-driven LQR problem as a semidefinite program (SDP) that computes an optimal gain, minimizing the steady-state covariance. Extensive simulations on benchmark systems -- including a rotary inverted pendulum and an active suspension system -- demonstrate the superior robustness and accuracy of our method compared to existing data-driven LQR approaches. The proposed framework offers a practical and theoretically grounded solution for controller design in noise-corrupted environments where system identification is infeasible. 

**Abstract (ZH)**: 基于直接数据驱动控制的鲁棒线性二次调节器方法：处理干扰和噪声状态测量的新型框架 

---
# "I Apologize For Not Understanding Your Policy": Exploring the Specification and Evaluation of User-Managed Access Control Policies by AI Virtual Assistants 

**Title (ZH)**: “抱歉没有理解您的政策”：探索AI虚拟助手对用户管理访问控制策略的规范与评估 

**Authors**: Jennifer Mondragon, Carlos Rubio-Medrano, Gael Cruz, Dvijesh Shastri  

**Link**: [PDF](https://arxiv.org/pdf/2505.07759)  

**Abstract**: The rapid evolution of Artificial Intelligence (AI)-based Virtual Assistants (VAs) e.g., Google Gemini, ChatGPT, Microsoft Copilot, and High-Flyer Deepseek has turned them into convenient interfaces for managing emerging technologies such as Smart Homes, Smart Cars, Electronic Health Records, by means of explicit commands,e.g., prompts, which can be even launched via voice, thus providing a very convenient interface for end-users. However, the proper specification and evaluation of User-Managed Access Control Policies (U-MAPs), the rules issued and managed by end-users to govern access to sensitive data and device functionality - within these VAs presents significant challenges, since such a process is crucial for preventing security vulnerabilities and privacy leaks without impacting user experience. This study provides an initial exploratory investigation on whether current publicly-available VAs can manage U-MAPs effectively across differing scenarios. By conducting unstructured to structured tests, we evaluated the comprehension of such VAs, revealing a lack of understanding in varying U-MAP approaches. Our research not only identifies key limitations, but offers valuable insights into how VAs can be further improved to manage complex authorization rules and adapt to dynamic changes. 

**Abstract (ZH)**: 基于人工智能的虚拟助手（如Google Gemini、ChatGPT、Microsoft Copilot和High-Flyer Deepseek的快速进化已成为通过显式命令管理智能家居、智能汽车、电子健康记录等新兴技术的便捷接口，这些命令甚至可以通过语音启动，从而为终端用户提供非常便捷的接口。然而，对用户管理访问控制策略（U-MAPs）的适当规范和评估——用户发布和管理的规则，以控制对敏感数据和设备功能的访问——在这些虚拟助手中的实现面临重大挑战，因为这一过程对于防止安全漏洞和隐私泄露至关重要，同时不影响用户体验。本研究提供了一项初步的探索性调查，探讨当前可公开获取的虚拟助手能否在不同场景下有效管理U-MAPs。通过从非结构化测试到结构化测试的评估，我们揭示了这些虚拟助手在不同U-MAP方法上的理解不足。研究不仅发现了关键的局限性，还为如何进一步改进虚拟助手以管理复杂的授权规则并适应动态变化提供了宝贵的见解。 

---
# HALO: Half Life-Based Outdated Fact Filtering in Temporal Knowledge Graphs 

**Title (ZH)**: HALO: 基于半衰期的过时事实过滤算法在时间型知识图谱中 

**Authors**: Feng Ding, Tingting Wang, Yupeng Gao, Shuo Yu, Jing Ren, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.07509)  

**Abstract**: Outdated facts in temporal knowledge graphs (TKGs) result from exceeding the expiration date of facts, which negatively impact reasoning performance on TKGs. However, existing reasoning methods primarily focus on positive importance of historical facts, neglecting adverse effects of outdated facts. Besides, training on these outdated facts yields extra computational cost. To address these challenges, we propose an outdated fact filtering framework named HALO, which quantifies the temporal validity of historical facts by exploring the half-life theory to filter outdated facts in TKGs. HALO consists of three modules: the temporal fact attention module, the dynamic relation-aware encoder module, and the outdated fact filtering module. Firstly, the temporal fact attention module captures the evolution of historical facts over time to identify relevant facts. Secondly, the dynamic relation-aware encoder module is designed for efficiently predicting the half life of each fact. Finally, we construct a time decay function based on the half-life theory to quantify the temporal validity of facts and filter outdated facts. Experimental results show that HALO outperforms the state-of-the-art TKG reasoning methods on three public datasets, demonstrating its effectiveness in detecting and filtering outdated facts (Codes are available at this https URL ). 

**Abstract (ZH)**: 过时事实存在于时间知识图谱中，导致事实超出有效期，从而负面影响时间知识图谱上的推理性能。然而，现有的推理方法主要关注历史事实的正向影响，忽略了过时事实的负面影响。此外，基于这些过时事实进行训练还会带来额外的计算成本。为解决这些问题，我们提出了一种名为HALO的过时事实过滤框架，该框架通过探究半衰期理论来量化历史事实的时间有效性并过滤过时事实。HALO包括三个模块：时间事实注意力模块、动态关系感知编码模块以及过时事实过滤模块。首先，时间事实注意力模块捕捉历史事实随时间的演变，以识别相关事实。其次，动态关系感知编码模块旨在高效预测每条事实的半衰期。最后，我们基于半衰期理论构建了一个时间衰减函数，以量化事实的时间有效性并过滤过时事实。实验结果表明，HALO在三个公开数据集上的表现优于现有的时间知识图谱推理方法，证明了其在检测和过滤过时事实方面的有效性（代码可在以下链接获取：this https URL）。 

---
# AIS Data-Driven Maritime Monitoring Based on Transformer: A Comprehensive Review 

**Title (ZH)**: 基于变压器的AIS数据驱动海洋监测：一项全面综述 

**Authors**: Zhiye Xie, Enmei Tu, Xianping Fu, Guoliang Yuan, Yi Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07374)  

**Abstract**: With the increasing demands for safety, efficiency, and sustainability in global shipping, Automatic Identification System (AIS) data plays an increasingly important role in maritime monitoring. AIS data contains spatial-temporal variation patterns of vessels that hold significant research value in the marine domain. However, due to its massive scale, the full potential of AIS data has long remained untapped. With its powerful sequence modeling capabilities, particularly its ability to capture long-range dependencies and complex temporal dynamics, the Transformer model has emerged as an effective tool for processing AIS data. Therefore, this paper reviews the research on Transformer-based AIS data-driven maritime monitoring, providing a comprehensive overview of the current applications of Transformer models in the marine field. The focus is on Transformer-based trajectory prediction methods, behavior detection, and prediction techniques. Additionally, this paper collects and organizes publicly available AIS datasets from the reviewed papers, performing data filtering, cleaning, and statistical analysis. The statistical results reveal the operational characteristics of different vessel types, providing data support for further research on maritime monitoring tasks. Finally, we offer valuable suggestions for future research, identifying two promising research directions. Datasets are available at this https URL. 

**Abstract (ZH)**: 随着全球航运对安全、效率和可持续性的需求不断增加，自动识别系统（AIS）数据在海上监控中发挥着越来越重要的作用。AIS数据包含了具有重要研究价值的船舶时空变化模式。然而，由于其规模庞大，AIS数据的全部潜力长期得不到充分发挥。凭借其强大的序列建模能力，尤其是捕捉长程依赖性和复杂时序动态的能力，Transformer模型已成为处理AIS数据的有效工具。因此，本文回顾了基于Transformer的AIS数据驱动的海上监控研究，提供了Transformer模型在海洋领域应用的全面概述，重点在于基于Transformer的航线预测方法、行为检测和预测技术。此外，本文还收集并整理了从文献中获取的公开可用的AIS数据集，进行了数据过滤、清洗和统计分析。统计结果揭示了不同船舶类型的运营特征，为进一步研究海上监控任务提供了数据支持。最后，我们提出了对未来研究的宝贵建议，指出了两个有前景的研究方向。数据集可从此链接获取。 

---
# FedIFL: A federated cross-domain diagnostic framework for motor-driven systems with inconsistent fault modes 

**Title (ZH)**: FedIFL：一种针对具有不一致故障模式的驱动系统跨域诊断框架 

**Authors**: Zexiao Wang, Yankai Wang, Xiaoqiang Liao, Xinguo Ming, Weiming Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07315)  

**Abstract**: Due to the scarcity of industrial data, individual equipment users, particularly start-ups, struggle to independently train a comprehensive fault diagnosis model; federated learning enables collaborative training while ensuring data privacy, making it an ideal solution. However, the diversity of working conditions leads to variations in fault modes, resulting in inconsistent label spaces across different clients. In federated diagnostic scenarios, label space inconsistency leads to local models focus on client-specific fault modes and causes local models from different clients to map different failure modes to similar feature representations, which weakens the aggregated global model's generalization. To tackle this issue, this article proposed a federated cross-domain diagnostic framework termed Federated Invariant Features Learning (FedIFL). In intra-client training, prototype contrastive learning mitigates intra-client domain shifts, subsequently, feature generating ensures local models can access distributions of other clients in a privacy-friendly manner. Besides, in cross-client training, a feature disentanglement mechanism is introduced to mitigate cross-client domain shifts, specifically, an instance-level federated instance consistency loss is designed to ensure the instance-level consistency of invariant features between different clients, furthermore, a federated instance personalization loss and an orthogonal loss are constructed to distinguish specific features that from the invariant features. Eventually, the aggregated model achieves promising generalization among global label spaces, enabling accurate fault diagnosis for target clients' Motor Driven Systems (MDSs) with inconsistent label spaces. Experiments on real-world MDSs validate the effectiveness and superiority of FedIFL in federated cross-domain diagnosis with inconsistent fault modes. 

**Abstract (ZH)**: 基于联邦学习的跨域不变特征学习故障诊断框架（FedIFL） 

---
# Interpretable Event Diagnosis in Water Distribution Networks 

**Title (ZH)**: 可解释的水资源分配网络事件诊断 

**Authors**: André Artelt, Stelios G. Vrachimis, Demetrios G. Eliades, Ulrike Kuhl, Barbara Hammer, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07299)  

**Abstract**: The increasing penetration of information and communication technologies in the design, monitoring, and control of water systems enables the use of algorithms for detecting and identifying unanticipated events (such as leakages or water contamination) using sensor measurements. However, data-driven methodologies do not always give accurate results and are often not trusted by operators, who may prefer to use their engineering judgment and experience to deal with such events.
In this work, we propose a framework for interpretable event diagnosis -- an approach that assists the operators in associating the results of algorithmic event diagnosis methodologies with their own intuition and experience. This is achieved by providing contrasting (i.e., counterfactual) explanations of the results provided by fault diagnosis algorithms; their aim is to improve the understanding of the algorithm's inner workings by the operators, thus enabling them to take a more informed decision by combining the results with their personal experiences. Specifically, we propose counterfactual event fingerprints, a representation of the difference between the current event diagnosis and the closest alternative explanation, which can be presented in a graphical way. The proposed methodology is applied and evaluated on a realistic use case using the L-Town benchmark. 

**Abstract (ZH)**: 信息系统和通信技术在水资源系统设计、监控和控制中的渗透使算法能够利用传感器测量数据检测和识别未预见的事件（如泄漏或水质污染）。然而，数据驱动的方法并不总是给出准确的结果，操作人员往往更倾向于依靠他们的工程判断和经验来应对这些事件。

本文提出了一种可解释的事件诊断框架——一种帮助操作人员将算法事件诊断方法的结果与自身的直觉和经验关联起来的方法。通过提供与故障诊断算法结果形成对比（即反事实）的解释，旨在通过改进操作人员对算法内部工作机制的理解，使他们能够在结合结果和自身经验的基础上作出更加明智的决定。具体而言，我们提出了反事实事件指纹，这是一种代表当前事件诊断与其最接近替代解释之间差异的表示方法，并可以通过图形方式呈现。所提出的方法在L-Town基准案例上进行了应用和评估。 

---
# Measuring General Intelligence with Generated Games 

**Title (ZH)**: 使用生成的游戏衡量通用智能 

**Authors**: Vivek Verma, David Huang, William Chen, Dan Klein, Nicholas Tomlin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07215)  

**Abstract**: We present gg-bench, a collection of game environments designed to evaluate general reasoning capabilities in language models. Unlike most static benchmarks, gg-bench is a data generating process where new evaluation instances can be generated at will. In particular, gg-bench is synthetically generated by (1) using a large language model (LLM) to generate natural language descriptions of novel games, (2) using the LLM to implement each game in code as a Gym environment, and (3) training reinforcement learning (RL) agents via self-play on the generated games. We evaluate language models by their winrate against these RL agents by prompting models with the game description, current board state, and a list of valid moves, after which models output the moves they wish to take. gg-bench is challenging: state-of-the-art LLMs such as GPT-4o and Claude 3.7 Sonnet achieve winrates of 7-9% on gg-bench using in-context learning, while reasoning models such as o1, o3-mini and DeepSeek-R1 achieve average winrates of 31-36%. We release the generated games, data generation process, and evaluation code in order to support future modeling work and expansion of our benchmark. 

**Abstract (ZH)**: gg-bench：用于评估语言模型通用推理能力的游戏环境集合 

---
# Accountability of Generative AI: Exploring a Precautionary Approach for "Artificially Created Nature" 

**Title (ZH)**: 生成式AI的责任性：探索“人工创造的自然”审慎 Approach 

**Authors**: Yuri Nakao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07178)  

**Abstract**: The rapid development of generative artificial intelligence (AI) technologies raises concerns about the accountability of sociotechnical systems. Current generative AI systems rely on complex mechanisms that make it difficult for even experts to fully trace the reasons behind the outputs. This paper first examines existing research on AI transparency and accountability and argues that transparency is not a sufficient condition for accountability but can contribute to its improvement. We then discuss that if it is not possible to make generative AI transparent, generative AI technology becomes ``artificially created nature'' in a metaphorical sense, and suggest using the precautionary principle approach to consider AI risks. Finally, we propose that a platform for citizen participation is needed to address the risks of generative AI. 

**Abstract (ZH)**: 生成人工智能技术的迅速发展引发了对社会技术系统问责性的关注。当前的生成人工智能系统依赖于复杂的机制，使得即使是专家也难以完全追溯其输出的原因。本文首先探讨了现有的人工智能透明性和问责制研究，并认为透明性并非问责制的充分条件，但可以促进其改进。然后我们讨论，如果不能使生成人工智能变得透明，这种技术在比喻意义上将成为“人造自然”，并建议采用预防性原则来考虑人工智能风险。最后，我们提出需要一个公民参与的平台以应对生成人工智能的风险。 

---
# ReCDAP: Relation-Based Conditional Diffusion with Attention Pooling for Few-Shot Knowledge Graph Completion 

**Title (ZH)**: 基于关系的条件扩散与注意力池化少样本知识图谱补全 

**Authors**: Jeongho Kim, Chanyeong Heo, Jaehee Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.07171)  

**Abstract**: Knowledge Graphs (KGs), composed of triples in the form of (head, relation, tail) and consisting of entities and relations, play a key role in information retrieval systems such as question answering, entity search, and recommendation. In real-world KGs, although many entities exist, the relations exhibit a long-tail distribution, which can hinder information retrieval performance. Previous few-shot knowledge graph completion studies focused exclusively on the positive triple information that exists in the graph or, when negative triples were incorporated, used them merely as a signal to indicate incorrect triples. To overcome this limitation, we propose Relation-Based Conditional Diffusion with Attention Pooling (ReCDAP). First, negative triples are generated by randomly replacing the tail entity in the support set. By conditionally incorporating positive information in the KG and non-existent negative information into the diffusion process, the model separately estimates the latent distributions for positive and negative relations. Moreover, including an attention pooler enables the model to leverage the differences between positive and negative cases explicitly. Experiments on two widely used datasets demonstrate that our method outperforms existing approaches, achieving state-of-the-art performance. The code is available at this https URL. 

**Abstract (ZH)**: 基于关系的概率扩散与注意力池化（ReCDAP）：提高知识图谱的负样本推理性能 

---
# Arbitrarily Applicable Same/Opposite Relational Responding with NARS 

**Title (ZH)**: 任意适用的相同/相反关系响应与NARS 

**Authors**: Robert Johansson, Patrick Hammer, Tony Lofthouse  

**Link**: [PDF](https://arxiv.org/pdf/2505.07079)  

**Abstract**: Same/opposite relational responding, a fundamental aspect of human symbolic cognition, allows the flexible generalization of stimulus relationships based on minimal experience. In this study, we demonstrate the emergence of \textit{arbitrarily applicable} same/opposite relational responding within the Non-Axiomatic Reasoning System (NARS), a computational cognitive architecture designed for adaptive reasoning under uncertainty. Specifically, we extend NARS with an implementation of \textit{acquired relations}, enabling the system to explicitly derive both symmetric (mutual entailment) and novel relational combinations (combinatorial entailment) from minimal explicit training in a contextually controlled matching-to-sample (MTS) procedure. Experimental results show that NARS rapidly internalizes explicitly trained relational rules and robustly demonstrates derived relational generalizations based on arbitrary contextual cues. Importantly, derived relational responding in critical test phases inherently combines both mutual and combinatorial entailments, such as deriving same-relations from multiple explicitly trained opposite-relations. Internal confidence metrics illustrate strong internalization of these relational principles, closely paralleling phenomena observed in human relational learning experiments. Our findings underscore the potential for integrating nuanced relational learning mechanisms inspired by learning psychology into artificial general intelligence frameworks, explicitly highlighting the arbitrary and context-sensitive relational capabilities modeled within NARS. 

**Abstract (ZH)**: 任意适用的相同/相反关系响应：非公理推理系统（NARS）中的一个基本方面，允许基于最少经验对刺激关系进行灵活泛化。在本研究中，我们展示了在非公理推理系统（NARS）中 Emergence of Arbitrarily Applicable Same/Opposite Relational Responding，这是一个为不确定环境下适应性推理设计的计算认知架构。具体而言，我们通过实现习得关系扩展了 NARS，使系统能够从最小的显式训练中在上下文控制的配对样本（MTS）程序中显式推导出对称关系（互蕴）和新颖的关系组合（组合理蕴）。实验结果表明，NARS 迅速内化了显式训练的关系规则，并且能够在任意上下文线索基础上稳健地展示推导出的关系泛化。重要的是，在关键测试阶段的推导关系响应中，必然结合了互蕴和组合理蕴，如从多个显式训练的相反关系中推导出相同关系。内部信心度量表明这些关系原则得到了强烈内化，近似平行于人类关系学习实验中观察到的现象。我们的研究结果强调将启发自学习心理学的细腻关系学习机制整合进人工通用智能框架的潜在可能性，明确指出 NARS 中建模的任意性和上下文敏感的关系能力。 

---
# Unlocking Non-Block-Structured Decisions: Inductive Mining with Choice Graphs 

**Title (ZH)**: 解锁非块结构决策：基于选择图的归纳挖掘 

**Authors**: Humam Kourani, Gyunam Park, Wil M.P. van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2505.07052)  

**Abstract**: Process discovery aims to automatically derive process models from event logs, enabling organizations to analyze and improve their operational processes. Inductive mining algorithms, while prioritizing soundness and efficiency through hierarchical modeling languages, often impose a strict block-structured representation. This limits their ability to accurately capture the complexities of real-world processes. While recent advancements like the Partially Ordered Workflow Language (POWL) have addressed the block-structure limitation for concurrency, a significant gap remains in effectively modeling non-block-structured decision points. In this paper, we bridge this gap by proposing an extension of POWL to handle non-block-structured decisions through the introduction of choice graphs. Choice graphs offer a structured yet flexible approach to model complex decision logic within the hierarchical framework of POWL. We present an inductive mining discovery algorithm that uses our extension and preserves the quality guarantees of the inductive mining framework. Our experimental evaluation demonstrates that the discovered models, enriched with choice graphs, more precisely represent the complex decision-making behavior found in real-world processes, without compromising the high scalability inherent in inductive mining techniques. 

**Abstract (ZH)**: 过程发现旨在从事件日志中自动推导过程模型，从而帮助组织分析和改进其运营流程。虽然归纳挖掘算法通过层次化的建模语言优先考虑正确性和效率，但往往要求一种严格的块结构表示形式，这限制了它们准确捕捉现实世界流程复杂性的能力。尽管POWL（部分有序工作流语言）等最近的进展解决了并发的块结构限制，但在有效建模非块结构决策点方面仍存在显著差距。在本文中，我们通过引入选择图来扩展POWL，弥补了这一差距，从而处理非块结构决策。选择图在POWL的层次框架内提供了一种结构化且灵活的方法来建模复杂的决策逻辑。我们提出了一种使用扩展方法的归纳挖掘发现算法，并保持了归纳挖掘框架的质量保证。我们的实验评估表明，通过引入选择图丰富后发现的模型，能够更精确地代表现实世界流程中的复杂决策行为，同时保持归纳挖掘技术固有的高可扩展性。 

---
# Efficient Fault Detection in WSN Based on PCA-Optimized Deep Neural Network Slicing Trained with GOA 

**Title (ZH)**: 基于GOA优化PCA深度神经网络切片的高效无线传感器网络故障检测 

**Authors**: Mahmood Mohassel Feghhi, Raya Majid Alsharfa, Majid Hameed Majeed  

**Link**: [PDF](https://arxiv.org/pdf/2505.07030)  

**Abstract**: Fault detection in Wireless Sensor Networks (WSNs) is crucial for reliable data transmission and network longevity. Traditional fault detection methods often struggle with optimizing deep neural networks (DNNs) for efficient performance, especially in handling high-dimensional data and capturing nonlinear relationships. Additionally, these methods typically suffer from slow convergence and difficulty in finding optimal network architectures using gradient-based optimization. This study proposes a novel hybrid method combining Principal Component Analysis (PCA) with a DNN optimized by the Grasshopper Optimization Algorithm (GOA) to address these limitations. Our approach begins by computing eigenvalues from the original 12-dimensional dataset and sorting them in descending order. The cumulative sum of these values is calculated, retaining principal components until 99.5% variance is achieved, effectively reducing dimensionality to 4 features while preserving critical information. This compressed representation trains a six-layer DNN where GOA optimizes the network architecture, overcoming backpropagation's limitations in discovering nonlinear relationships. This hybrid PCA-GOA-DNN framework compresses the data and trains a six-layer DNN that is optimized by GOA, enhancing both training efficiency and fault detection accuracy. The dataset used in this study is a real-world WSNs dataset developed by the University of North Carolina, which was used to evaluate the proposed method's performance. Extensive simulations demonstrate that our approach achieves a remarkable 99.72% classification accuracy, with exceptional precision and recall, outperforming conventional methods. The method is computationally efficient, making it suitable for large-scale WSN deployments, and represents a significant advancement in fault detection for resource-constrained WSNs. 

**Abstract (ZH)**: Wireless传感器网络中故障检测的关键在于可靠的数据传输和网络 longevity。传统的故障检测方法在优化深度神经网络（DNNs）以实现高效性能时往往遇到困难，特别是在处理高维数据和捕捉非线性关系方面。此外，这些方法通常收敛缓慢，并且在使用梯度优化法寻找最优网络架构时面临困难。本研究提出了一种结合主成分分析（PCA）和Improved Grasshopper Optimization Algorithm（GOA）优化的DNN的新颖混合方法，以解决这些限制。我们的方法首先计算原始12维数据集的特征值，并按降序排序。计算这些值的累计和，直到实现99.5%的方差保留主成分，从而将维度有效降低到4个特征，同时保留关键信息。该压缩表示训练一个六层DNN，其中GOA优化网络架构，克服了反向传播在发现非线性关系方面的局限性。这种结合PCA和GOA的DNN框架压缩了数据并训练了一个六层DNN，GOA优化了网络架构，提高了训练效率和故障检测准确性。本研究使用的数据集是由北卡罗来纳大学开发的无线传感器网络真实世界数据集，用于评估所提出方法的性能。广泛的仿真表明，我们的方法实现了令人瞩目的99.72%分类精度，具有出色的精确度和召回率，优于传统方法。该方法计算效率高，适用于大规模无线传感器网络部署，并代表了资源受限无线传感器网络中故障检测的一个重要进展。无线传感器网络中基于PCA和GOA优化的DNN的故障检测 

---
# Explainable AI the Latest Advancements and New Trends 

**Title (ZH)**: 可解释的人工智能：最新进展和新趋势 

**Authors**: Bowen Long, Enjie Liu, Renxi Qiu, Yanqing Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07005)  

**Abstract**: In recent years, Artificial Intelligence technology has excelled in various applications across all domains and fields. However, the various algorithms in neural networks make it difficult to understand the reasons behind decisions. For this reason, trustworthy AI techniques have started gaining popularity. The concept of trustworthiness is cross-disciplinary; it must meet societal standards and principles, and technology is used to fulfill these requirements. In this paper, we first surveyed developments from various countries and regions on the ethical elements that make AI algorithms trustworthy; and then focused our survey on the state of the art research into the interpretability of AI. We have conducted an intensive survey on technologies and techniques used in making AI explainable. Finally, we identified new trends in achieving explainable AI. In particular, we elaborate on the strong link between the explainability of AI and the meta-reasoning of autonomous systems. The concept of meta-reasoning is 'reason the reasoning', which coincides with the intention and goal of explainable Al. The integration of the approaches could pave the way for future interpretable AI systems. 

**Abstract (ZH)**: 近年来，人工智能技术在各个领域和应用中取得了卓越成就。然而，神经网络中的各种算法使得理解和解释决策的原因变得困难。因此，可靠的人工智能技术开始受到关注。信任的概念是跨学科的，必须符合社会标准和原则，技术用于满足这些要求。本文首先调研了来自不同国家和地区使人工智能算法可信的伦理元素的发展；然后重点调查了人工智能可解释性的最新研究。我们对使人工智能具有可解释性的技术与方法进行了深入调研。最后，我们确定了实现可解释人工智能的新趋势。特别地，我们详细阐述了人工智能解释性和自主系统元推理之间的密切联系。元推理的概念是“反思推理”，这与可解释人工智能的意图和目标相一致。这两种方法的结合可能会为未来的可解释人工智能系统铺平道路。 

---
# CAT Merging: A Training-Free Approach for Resolving Conflicts in Model Merging 

**Title (ZH)**: CAT 合并：一种无需训练的模型合并冲突解决方法 

**Authors**: Wenju Sun, Qingyong Li, Yangli-ao Geng, Boyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06977)  

**Abstract**: Multi-task model merging offers a promising paradigm for integrating multiple expert models into a unified model without additional training. Existing state-of-the-art techniques, such as Task Arithmetic and its variants, merge models by accumulating task vectors -- the parameter differences between pretrained and finetuned models. However, task vector accumulation is often hindered by knowledge conflicts, leading to performance degradation. To address this challenge, we propose Conflict-Aware Task Merging (CAT Merging), a novel training-free framework that selectively trims conflict-prone components from the task vectors. CAT Merging introduces several parameter-specific strategies, including projection for linear weights and masking for scaling and shifting parameters in normalization layers. Extensive experiments on vision, language, and vision-language tasks demonstrate that CAT Merging effectively suppresses knowledge conflicts, achieving average accuracy improvements of up to 2.5% (ViT-B/32) and 2.0% (ViT-L/14) over state-of-the-art methods. 

**Abstract (ZH)**: 多任务模型合并提供了一种有希望的范式，用于在无需额外训练的情况下将多个专家模型整合到一个统一模型中。现有最先进的技术，如任务算术及其变体，通过积累任务向量——预训练模型和微调模型之间的参数差异，来合并模型。然而，任务向量的累积往往受到知识冲突的阻碍，导致性能下降。为了解决这一挑战，我们提出了一种新的无训练框架——冲突感知任务合并（CAT 合并），该框架选择性地修剪任务向量中的冲突性组件。CAT 合并引入了几种针对参数的具体策略，包括投影用于线性权重和掩码用于归一化层中的缩放和移位参数。在视觉、语言和视觉-语言任务上的广泛实验表明，CAT 合并有效地抑制了知识冲突，相对于最先进的方法分别实现了高达 2.5%（ViT-B/32）和 2.0%（ViT-L/14）的平均准确率提升。 

---
# Causal knowledge graph analysis identifies adverse drug effects 

**Title (ZH)**: 因果知识图谱分析识别药物不良反应 

**Authors**: Sumyyah Toonsi, Paul Schofield, Robert Hoehndorf  

**Link**: [PDF](https://arxiv.org/pdf/2505.06949)  

**Abstract**: Knowledge graphs and structural causal models have each proven valuable for organizing biomedical knowledge and estimating causal effects, but remain largely disconnected: knowledge graphs encode qualitative relationships focusing on facts and deductive reasoning without formal probabilistic semantics, while causal models lack integration with background knowledge in knowledge graphs and have no access to the deductive reasoning capabilities that knowledge graphs provide. To bridge this gap, we introduce a novel formulation of Causal Knowledge Graphs (CKGs) which extend knowledge graphs with formal causal semantics, preserving their deductive capabilities while enabling principled causal inference. CKGs support deconfounding via explicitly marked causal edges and facilitate hypothesis formulation aligned with both encoded and entailed background knowledge. We constructed a Drug-Disease CKG (DD-CKG) integrating disease progression pathways, drug indications, side-effects, and hierarchical disease classification to enable automated large-scale mediation analysis. Applied to UK Biobank and MIMIC-IV cohorts, we tested whether drugs mediate effects between indications and downstream disease progression, adjusting for confounders inferred from the DD-CKG. Our approach successfully reproduced known adverse drug reactions with high precision while identifying previously undocumented significant candidate adverse effects. Further validation through side effect similarity analysis demonstrated that combining our predicted drug effects with established databases significantly improves the prediction of shared drug indications, supporting the clinical relevance of our novel findings. These results demonstrate that our methodology provides a generalizable, knowledge-driven framework for scalable causal inference. 

**Abstract (ZH)**: 基于因果的知識圖譜（Causal Knowledge Graphs, CKGs）融合了結構因果模型和知識圖譜的優點，為生物醫學知識的組織和因果效應的估算提供了系統性的解決方案。 

---
# Value Iteration with Guessing for Markov Chains and Markov Decision Processes 

**Title (ZH)**: 基于猜测的值迭代方法在马尔可夫链与马尔可夫决策过程中的应用 

**Authors**: Krishnendu Chatterjee, Mahdi JafariRaviz, Raimundo Saona, Jakub Svoboda  

**Link**: [PDF](https://arxiv.org/pdf/2505.06769)  

**Abstract**: Two standard models for probabilistic systems are Markov chains (MCs) and Markov decision processes (MDPs). Classic objectives for such probabilistic models for control and planning problems are reachability and stochastic shortest path. The widely studied algorithmic approach for these problems is the Value Iteration (VI) algorithm which iteratively applies local updates called Bellman updates. There are many practical approaches for VI in the literature but they all require exponentially many Bellman updates for MCs in the worst case. A preprocessing step is an algorithm that is discrete, graph-theoretical, and requires linear space. An important open question is whether, after a polynomial-time preprocessing, VI can be achieved with sub-exponentially many Bellman updates. In this work, we present a new approach for VI based on guessing values. Our theoretical contributions are twofold. First, for MCs, we present an almost-linear-time preprocessing algorithm after which, along with guessing values, VI requires only subexponentially many Bellman updates. Second, we present an improved analysis of the speed of convergence of VI for MDPs. Finally, we present a practical algorithm for MDPs based on our new approach. Experimental results show that our approach provides a considerable improvement over existing VI-based approaches on several benchmark examples from the literature. 

**Abstract (ZH)**: 两类概率系统的标准模型是马尔可夫链（MCs）和马尔可夫决策过程（MDPs）。这类概率模型的经典目标是可达性和随机最短路径。这些目标的经典算法方法是值迭代（VI）算法，该算法通过应用局部更新（称为贝尔曼更新）进行迭代。文献中有很多实用的VI方法，但在最坏情况下，它们都需要对MCs进行指数数量的贝尔曼更新。预处理步骤是一种离散、图论性的算法，并且只需要线性空间。一个重要的开放问题是，在多项式时间预处理之后，VI是否可以在亚指数数量的贝尔曼更新下实现。在这项工作中，我们提出了一种基于猜测值的新VI方法。我们的理论贡献主要有两点。首先，对于MCs，我们提出了一种几乎线性时间的预处理算法，之后结合猜测值，VI只需亚指数数量的贝尔曼更新。第二，我们对VI对于MDPs的收敛速度进行了改进分析。最后，我们基于新方法提出了一个MDPs的实际算法。实验结果显示，我们的方法在文献中的多个基准示例上显著优于现有的VI方法。 

---
# Bi-level Mean Field: Dynamic Grouping for Large-Scale MARL 

**Title (ZH)**: 双层均场：大规模 MARL 的动态分组 

**Authors**: Yuxuan Zheng, Yihe Zhou, Feiyang Xu, Mingli Song, Shunyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06706)  

**Abstract**: Large-scale Multi-Agent Reinforcement Learning (MARL) often suffers from the curse of dimensionality, as the exponential growth in agent interactions significantly increases computational complexity and impedes learning efficiency. To mitigate this, existing efforts that rely on Mean Field (MF) simplify the interaction landscape by approximating neighboring agents as a single mean agent, thus reducing overall complexity to pairwise interactions. However, these MF methods inevitably fail to account for individual differences, leading to aggregation noise caused by inaccurate iterative updates during MF learning. In this paper, we propose a Bi-level Mean Field (BMF) method to capture agent diversity with dynamic grouping in large-scale MARL, which can alleviate aggregation noise via bi-level interaction. Specifically, BMF introduces a dynamic group assignment module, which employs a Variational AutoEncoder (VAE) to learn the representations of agents, facilitating their dynamic grouping over time. Furthermore, we propose a bi-level interaction module to model both inter- and intra-group interactions for effective neighboring aggregation. Experiments across various tasks demonstrate that the proposed BMF yields results superior to the state-of-the-art methods. Our code will be made publicly available. 

**Abstract (ZH)**: 大规模多智能体 reinforcement 学习 (MARL) 往往受到维数灾的困扰，由于代理交互的指数级增长显著增加计算复杂性并阻碍学习效率。为缓解这一问题，现有依赖均场（MF）的方法通过近似邻近代理为单一均场代理来简化交互场景，从而将总体复杂性降低为两两交互。然而，这些MF方法不可避免地无法考虑个体差异，导致均场学习过程中因迭代更新不准确引起的聚合噪声。在本文中，我们提出了一种双层均场（BMF）方法，通过动态分组来捕捉大规模MARL中的代理多样性，借助双层交互来缓解聚合噪声。具体而言，BMF引入了一个动态组分配模块，该模块使用变分自编码器（VAE）学习代理的表示，促进其随时间进行动态分组。此外，我们提出了一种双层交互模块来建模组内和组间交互，以实现有效的邻近聚合。在多种任务上的实验表明，所提出的BMF方法优于现有最佳方法。我们的代码将公开发布。 

---
# A Survey on Data-Driven Modeling of Human Drivers' Lane-Changing Decisions 

**Title (ZH)**: 基于数据的人类驾驶车道变换决策建模综述 

**Authors**: Linxuan Huang, Dong-Fan Xie, Li Li, Zhengbing He  

**Link**: [PDF](https://arxiv.org/pdf/2505.06680)  

**Abstract**: Lane-changing (LC) behavior, a critical yet complex driving maneuver, significantly influences driving safety and traffic dynamics. Traditional analytical LC decision (LCD) models, while effective in specific environments, often oversimplify behavioral heterogeneity and complex interactions, limiting their capacity to capture real LCD. Data-driven approaches address these gaps by leveraging rich empirical data and machine learning to decode latent decision-making patterns, enabling adaptive LCD modeling in dynamic environments. In light of the rapid development of artificial intelligence and the demand for data-driven models oriented towards connected vehicles and autonomous vehicles, this paper presents a comprehensive survey of data-driven LCD models, with a particular focus on human drivers LC decision-making. It systematically reviews the modeling framework, covering data sources and preprocessing, model inputs and outputs, objectives, structures, and validation methods. This survey further discusses the opportunities and challenges faced by data-driven LCD models, including driving safety, uncertainty, as well as the integration and improvement of technical frameworks. 

**Abstract (ZH)**: 基于数据的变道决策模型综述：针对连接车辆和自动驾驶车辆的人类驾驶变道决策 

---
# TAROT: Towards Essentially Domain-Invariant Robustness with Theoretical Justification 

**Title (ZH)**: TAROT：向着具有理论依据的本领域基本不变鲁棒性研究 

**Authors**: Dongyoon Yang, Jihu Lee, Yongdai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06580)  

**Abstract**: Robust domain adaptation against adversarial attacks is a critical research area that aims to develop models capable of maintaining consistent performance across diverse and challenging domains. In this paper, we derive a new generalization bound for robust risk on the target domain using a novel divergence measure specifically designed for robust domain adaptation. Building upon this, we propose a new algorithm named TAROT, which is designed to enhance both domain adaptability and robustness. Through extensive experiments, TAROT not only surpasses state-of-the-art methods in accuracy and robustness but also significantly enhances domain generalization and scalability by effectively learning domain-invariant features. In particular, TAROT achieves superior performance on the challenging DomainNet dataset, demonstrating its ability to learn domain-invariant representations that generalize well across different domains, including unseen ones. These results highlight the broader applicability of our approach in real-world domain adaptation scenarios. 

**Abstract (ZH)**: 对抗攻击下稳健领域适应的鲁棒泛化边界的derive以及一种新的TAROT算法研究 

---
# Online Feedback Efficient Active Target Discovery in Partially Observable Environments 

**Title (ZH)**: 在线反馈驱动的主动目标发现Partial观察环境中的高效方法 

**Authors**: Anindya Sarkar, Binglin Ji, Yevgeniy Vorobeychik  

**Link**: [PDF](https://arxiv.org/pdf/2505.06535)  

**Abstract**: In various scientific and engineering domains, where data acquisition is costly, such as in medical imaging, environmental monitoring, or remote sensing, strategic sampling from unobserved regions, guided by prior observations, is essential to maximize target discovery within a limited sampling budget. In this work, we introduce Diffusion-guided Active Target Discovery (DiffATD), a novel method that leverages diffusion dynamics for active target discovery. DiffATD maintains a belief distribution over each unobserved state in the environment, using this distribution to dynamically balance exploration-exploitation. Exploration reduces uncertainty by sampling regions with the highest expected entropy, while exploitation targets areas with the highest likelihood of discovering the target, indicated by the belief distribution and an incrementally trained reward model designed to learn the characteristics of the target. DiffATD enables efficient target discovery in a partially observable environment within a fixed sampling budget, all without relying on any prior supervised training. Furthermore, DiffATD offers interpretability, unlike existing black-box policies that require extensive supervised training. Through extensive experiments and ablation studies across diverse domains, including medical imaging and remote sensing, we show that DiffATD performs significantly better than baselines and competitively with supervised methods that operate under full environmental observability. 

**Abstract (ZH)**: 在医学成像、环境监控或遥感等数据采集成本较高的科学和工程领域，通过利用先验观测指导未观测区域的战略性采样，以最大限度地在有限的采样预算内发现目标，是至关重要的。本文引入了扩散引导主动目标发现（DiffATD）方法，该方法利用扩散动力学进行主动目标发现。DiffATD 在环境中每个未观测状态上维护一种信念分布，利用该分布动态平衡探索与利用。探索通过采样具有最高预期熵的区域来减少不确定性，而利用信念分布和一个逐步训练的奖励模型（该模型设计用于学习目标特性）来指向具有最高目标发现可能性的区域，从而实现目标发现。DiffATD 不依赖任何先验监督训练，即可在固定采样预算内高效地在部分可观测环境中发现目标，并且提供了可解释性，与现有的黑盒策略相比省去了大量的监督训练。通过在包括医学成像和遥感在内的多个领域的广泛实验和消融研究，我们展示了 DiffATD 在基线方法上显著优越的表现，并且在完全可观测环境中操作的监督方法中表现具有竞争力。 

---
# On Definite Iterated Belief Revision with Belief Algebras 

**Title (ZH)**: 基于信念代数的确定性迭代信念修订 

**Authors**: Hua Meng, Zhiguo Long, Michael Sioutis, Zhengchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06505)  

**Abstract**: Traditional logic-based belief revision research focuses on designing rules to constrain the behavior of revision operators. Frameworks have been proposed to characterize iterated revision rules, but they are often too loose, leading to multiple revision operators that all satisfy the rules under the same belief condition. In many practical applications, such as safety critical ones, it is important to specify a definite revision operator to enable agents to iteratively revise their beliefs in a deterministic way. In this paper, we propose a novel framework for iterated belief revision by characterizing belief information through preference relations. Semantically, both beliefs and new evidence are represented as belief algebras, which provide a rich and expressive foundation for belief revision. Building on traditional revision rules, we introduce additional postulates for revision with belief algebra, including an upper-bound constraint on the outcomes of revision. We prove that the revision result is uniquely determined given the current belief state and new evidence. Furthermore, to make the framework more useful in practice, we develop a particular algorithm for performing the proposed revision process. We argue that this approach may offer a more predictable and principled method for belief revision, making it suitable for real-world applications. 

**Abstract (ZH)**: 基于偏好关系的迭代信念修订新框架：提供确定性信念更新的方法 

---
# SmartPilot: A Multiagent CoPilot for Adaptive and Intelligent Manufacturing 

**Title (ZH)**: SmartPilot: 一个多代理协作副驾 for 自适应与智能制造 

**Authors**: Chathurangi Shyalika, Renjith Prasad, Alaa Al Ghazo, Darssan Eswaramoorthi, Harleen Kaur, Sara Shree Muthuselvam, Amit Sheth  

**Link**: [PDF](https://arxiv.org/pdf/2505.06492)  

**Abstract**: In the dynamic landscape of Industry 4.0, achieving efficiency, precision, and adaptability is essential to optimize manufacturing operations. Industries suffer due to supply chain disruptions caused by anomalies, which are being detected by current AI models but leaving domain experts uncertain without deeper insights into these anomalies. Additionally, operational inefficiencies persist due to inaccurate production forecasts and the limited effectiveness of traditional AI models for processing complex sensor data. Despite these advancements, existing systems lack the seamless integration of these capabilities needed to create a truly unified solution for enhancing production and decision-making. We propose SmartPilot, a neurosymbolic, multiagent CoPilot designed for advanced reasoning and contextual decision-making to address these challenges. SmartPilot processes multimodal sensor data and is compact to deploy on edge devices. It focuses on three key tasks: anomaly prediction, production forecasting, and domain-specific question answering. By bridging the gap between AI capabilities and real-world industrial needs, SmartPilot empowers industries with intelligent decision-making and drives transformative innovation in manufacturing. The demonstration video, datasets, and supplementary materials are available at this https URL. 

**Abstract (ZH)**: 在Industry 4.0的动态背景下，实现效率、精确性和适应性对于优化制造运营至关重要。由于异常引起的供应链中断对行业造成影响，当前的AI模型能够检测这些异常，但缺乏深入洞察使领域专家感到不确定。此外，由于生产预测不够准确以及传统AI模型处理复杂传感器数据效果有限，运营效率持续低下。尽管取得了这些进展，现有系统仍缺乏将这些能力无缝集成的机制，以创建真正统一的解决方案，以增强生产能力和决策制定。我们提出SmartPilot，一个神经符号型的多代理 Copilot，专为高级推理和上下文决策制定而设计，以应对这些挑战。SmartPilot处理多模态传感器数据，并可在边缘设备上进行紧凑部署。它专注于三项关键任务：异常预测、生产预测和领域特定的问题回答。通过弥合AI能力和实际工业需求之间的差距，SmartPilot使行业能够进行智能决策，并推动制造领域的创新变革。详细演示视频、数据集和补充材料请访问此网址：[该网址]。 

---
# Opening the Scope of Openness in AI 

**Title (ZH)**: 打开人工智能开放性的范围 

**Authors**: Tamara Paris, AJung Moon, Jin Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06464)  

**Abstract**: The concept of openness in AI has so far been heavily inspired by the definition and community practice of open source software. This positions openness in AI as having positive connotations; it introduces assumptions of certain advantages, such as collaborative innovation and transparency. However, the practices and benefits of open source software are not fully transferable to AI, which has its own challenges. Framing a notion of openness tailored to AI is crucial to addressing its growing societal implications, risks, and capabilities. We argue that considering the fundamental scope of openness in different disciplines will broaden discussions, introduce important perspectives, and reflect on what openness in AI should mean. Toward this goal, we qualitatively analyze 98 concepts of openness discovered from topic modeling, through which we develop a taxonomy of openness. Using this taxonomy as an instrument, we situate the current discussion on AI openness, identify gaps and highlight links with other disciplines. Our work contributes to the recent efforts in framing openness in AI by reflecting principles and practices of openness beyond open source software and calls for a more holistic view of openness in terms of actions, system properties, and ethical objectives. 

**Abstract (ZH)**: AI领域的开放性概念：超越开源软件的原则与实践Towards AI领域的开放性概念：超越开源软件的原则与实践 

---
# BedreFlyt: Improving Patient Flows through Hospital Wards with Digital Twins 

**Title (ZH)**: BedreFlyt: 通过数字双胞胎优化医院病区的患者流动 

**Authors**: Riccardo Sieve, Paul Kobialka, Laura Slaughter, Rudolf Schlatte, Einar Broch Johnsen, Silvia Lizeth Tapia Tarifa  

**Link**: [PDF](https://arxiv.org/pdf/2505.06287)  

**Abstract**: Digital twins are emerging as a valuable tool for short-term decision-making as well as for long-term strategic planning across numerous domains, including process industry, energy, space, transport, and healthcare. This paper reports on our ongoing work on designing a digital twin to enhance resource planning, e.g., for the in-patient ward needs in hospitals. By leveraging executable formal models for system exploration, ontologies for knowledge representation and an SMT solver for constraint satisfiability, our approach aims to explore hypothetical "what-if" scenarios to improve strategic planning processes, as well as to solve concrete, short-term decision-making tasks. Our proposed solution uses the executable formal model to turn a stream of arriving patients, that need to be hospitalized, into a stream of optimization problems, e.g., capturing daily inpatient ward needs, that can be solved by SMT techniques. The knowledge base, which formalizes domain knowledge, is used to model the needed configuration in the digital twin, allowing the twin to support both short-term decision-making and long-term strategic planning by generating scenarios spanning average-case as well as worst-case resource needs, depending on the expected treatment of patients, as well as ranging over variations in available resources, e.g., bed distribution in different rooms. We illustrate our digital twin architecture by considering the problem of bed bay allocation in a hospital ward. 

**Abstract (ZH)**: 数字孪生在医疗领域住院病房资源配置中的应用：短中期决策与战略规划的增强 

---
# A class of distributed automata that contains the modal mu-fragment 

**Title (ZH)**: 一类包含模态mu片段的分布式自动机类 

**Authors**: Veeti Ahvonen, Damian Heiman, Antti Kuusisto  

**Link**: [PDF](https://arxiv.org/pdf/2505.07816)  

**Abstract**: This paper gives a translation from the $\mu$-fragment of the graded modal $\mu$-calculus to a class of distributed message-passing automata. As a corollary, we obtain an alternative proof for a theorem from \cite{ahvonen_neurips} stating that recurrent graph neural networks working with reals and graded modal substitution calculus have the same expressive power in restriction to the logic monadic second-order logic MSO. 

**Abstract (ZH)**: 本文将分级模态μ-演算的μ片段翻译为一类分布式消息传递自动机。作为推论，我们得到了一个关于《[ahvonen_neurips]》中定理的替代证明，该定理指出，使用实数和分级模态替换演算的循环图神经网络在逻辑单调二阶逻辑MSO方面的表达能力相同。 

---
# A Comparative Analysis of Static Word Embeddings for Hungarian 

**Title (ZH)**: 匈牙利语静态词嵌入的比较分析 

**Authors**: Máté Gedeon  

**Link**: [PDF](https://arxiv.org/pdf/2505.07809)  

**Abstract**: This paper presents a comprehensive analysis of various static word embeddings for Hungarian, including traditional models such as Word2Vec, FastText, as well as static embeddings derived from BERT-based models using different extraction methods. We evaluate these embeddings on both intrinsic and extrinsic tasks to provide a holistic view of their performance. For intrinsic evaluation, we employ a word analogy task, which assesses the embeddings ability to capture semantic and syntactic relationships. Our results indicate that traditional static embeddings, particularly FastText, excel in this task, achieving high accuracy and mean reciprocal rank (MRR) scores. Among the BERT-based models, the X2Static method for extracting static embeddings demonstrates superior performance compared to decontextualized and aggregate methods, approaching the effectiveness of traditional static embeddings. For extrinsic evaluation, we utilize a bidirectional LSTM model to perform Named Entity Recognition (NER) and Part-of-Speech (POS) tagging tasks. The results reveal that embeddings derived from dynamic models, especially those extracted using the X2Static method, outperform purely static embeddings. Notably, ELMo embeddings achieve the highest accuracy in both NER and POS tagging tasks, underscoring the benefits of contextualized representations even when used in a static form. Our findings highlight the continued relevance of static word embeddings in NLP applications and the potential of advanced extraction methods to enhance the utility of BERT-based models. This piece of research contributes to the understanding of embedding performance in the Hungarian language and provides valuable insights for future developments in the field. The training scripts, evaluation codes, restricted vocabulary, and extracted embeddings will be made publicly available to support further research and reproducibility. 

**Abstract (ZH)**: 本文对多种Hungarian静态词嵌入进行了全面分析，包括传统的Word2Vec和FastText模型，以及使用不同提取方法从BERT基模拟能源生的静态嵌入。我们通过内在和外在任务的评估，提供了对这些嵌入性能的整体视角。内在评估采用词类比任务，评估嵌入捕捉语义和句法关系的能力。结果表明，传统静态嵌入，特别是FastText，在此任务中表现出色，获得高准确率和均值倒数排名（MRR）分数。在基于BERT的模型中，采用X2Static方法提取的静态嵌入的表现优于基于上下文和聚合的方法，接近传统静态嵌入的有效性。在外在评估中，我们使用双向LSTM模型进行命名实体识别（NER）和词性标注（POS）任务。结果显示，动态模型衍生的嵌入，尤其是使用X2Static方法提取的嵌入，优于单纯的静态嵌入。值得注意的是，ELMo嵌入在NER和POS标注任务中均获得最高准确率，突显了即使在静态形式下使用上下文表示的优势。本文的研究结果突显了静态词嵌入在NLP应用中的持续相关性，并展示了高级提取方法增强基于BERT模型用途的潜力。本文的研究为理解匈牙利语嵌入性能提供了见解，并为该领域的未来发展方向提供了宝贵信息。用于训练的脚本、评估代码、受限词汇表以及提取的嵌入将对外公开，以支持进一步研究和可重复性。 

---
# Must Read: A Systematic Survey of Computational Persuasion 

**Title (ZH)**: 必读：计算劝导系统的综述 

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Xiaocheng Yang, Hyeonjeong Ha, Zirui Cheng, Esin Durmus, Jiaxuan You, Heng Ji, Gokhan Tur, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2505.07775)  

**Abstract**: Persuasion is a fundamental aspect of communication, influencing decision-making across diverse contexts, from everyday conversations to high-stakes scenarios such as politics, marketing, and law. The rise of conversational AI systems has significantly expanded the scope of persuasion, introducing both opportunities and risks. AI-driven persuasion can be leveraged for beneficial applications, but also poses threats through manipulation and unethical influence. Moreover, AI systems are not only persuaders, but also susceptible to persuasion, making them vulnerable to adversarial attacks and bias reinforcement. Despite rapid advancements in AI-generated persuasive content, our understanding of what makes persuasion effective remains limited due to its inherently subjective and context-dependent nature. In this survey, we provide a comprehensive overview of computational persuasion, structured around three key perspectives: (1) AI as a Persuader, which explores AI-generated persuasive content and its applications; (2) AI as a Persuadee, which examines AI's susceptibility to influence and manipulation; and (3) AI as a Persuasion Judge, which analyzes AI's role in evaluating persuasive strategies, detecting manipulation, and ensuring ethical persuasion. We introduce a taxonomy for computational persuasion research and discuss key challenges, including evaluating persuasiveness, mitigating manipulative persuasion, and developing responsible AI-driven persuasive systems. Our survey outlines future research directions to enhance the safety, fairness, and effectiveness of AI-powered persuasion while addressing the risks posed by increasingly capable language models. 

**Abstract (ZH)**: 计算说服是沟通的基本方面，影响从日常生活对话到政治、营销和法律等高风险场景中的决策。随着对话式AI系统的发展，说服的范围显著扩大，带来了机遇和风险。AI驱动的说服可以用于有益的应用，但也有可能通过操纵和不道德影响来构成威胁。此外，AI系统不仅是说服者，也是可以被说服的对象，使其容易遭受对抗性攻击和偏见强化。尽管在AI生成的说服性内容方面取得了快速进展，但由于其固有的主观性和情境依赖性，我们对其为何有效的理解仍然有限。在本综述中，我们从三个关键视角提供了计算说服的全面概述：（1）AI作为说服者，探讨了AI生成的说服性内容及其应用；（2）AI作为说服对象，研究了AI的可利用性和操纵性；（3）AI作为说服裁判者，分析了AI在评估说服策略、检测操纵和确保道德说服方面的作用。我们介绍了计算说服研究的分类体系，并讨论了关键挑战，包括评估说服性、减轻具有欺骗性的说服以及开发负责任的AI驱动说服系统。本综述概述了未来研究方向，旨在增强AI驱动说服的安全性、公平性和有效性，同时应对日益强大的语言模型带来的风险。 

---
# Benchmarking of CPU-intensive Stream Data Processing in The Edge Computing Systems 

**Title (ZH)**: 边缘计算系统中CPU密集型流数据处理的基准测试 

**Authors**: Tomasz Szydlo, Viacheslaw Horbanow, Dev Nandan Jha, Shashikant Ilager, Aleksander Slominski, Rajiv Ranjan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07755)  

**Abstract**: Edge computing has emerged as a pivotal technology, offering significant advantages such as low latency, enhanced data security, and reduced reliance on centralized cloud infrastructure. These benefits are crucial for applications requiring real-time data processing or strict security measures. Despite these advantages, edge devices operating within edge clusters are often underutilized. This inefficiency is mainly due to the absence of a holistic performance profiling mechanism which can help dynamically adjust the desired system configuration for a given workload. Since edge computing environments involve a complex interplay between CPU frequency, power consumption, and application performance, a deeper understanding of these correlations is essential. By uncovering these relationships, it becomes possible to make informed decisions that enhance both computational efficiency and energy savings. To address this gap, this paper evaluates the power consumption and performance characteristics of a single processing node within an edge cluster using a synthetic microbenchmark by varying the workload size and CPU frequency. The results show how an optimal measure can lead to optimized usage of edge resources, given both performance and power consumption. 

**Abstract (ZH)**: 边缘计算作为一种关键性技术，提供了低延迟、增强的数据安全性和减少对集中式云基础设施依赖等显著优势。这些优势对于需要实时数据处理或严格安全措施的应用至关重要。尽管具有这些优势，边缘集群中的边缘设备往往利用不足。这种低效主要是由于缺乏一个全面的性能分析机制，该机制能帮助动态调整给定工作负载下的系统配置。由于边缘计算环境涉及CPU频率、功耗和应用性能之间的复杂交互，深入理解这些相关性至关重要。通过揭示这些关系，可以做出既能提升计算效率又能节省能源的明智决策。为了填补这一空白，本文通过改变工作负载大小和CPU频率来使用合成微基准评估边缘集群中单个处理节点的功耗和性能特性。结果表明，在兼顾性能和功率消耗的情况下，适当的优化措施能实现边缘资源的最佳利用。 

---
# Lightweight End-to-end Text-to-speech Synthesis for low resource on-device applications 

**Title (ZH)**: 面向低资源嵌入式应用的轻量级端到端文本到语音合成 

**Authors**: Biel Tura Vecino, Adam Gabryś, Daniel Mątwicki, Andrzej Pomirski, Tom Iddon, Marius Cotescu, Jaime Lorenzo-Trueba  

**Link**: [PDF](https://arxiv.org/pdf/2505.07701)  

**Abstract**: Recent works have shown that modelling raw waveform directly from text in an end-to-end (E2E) fashion produces more natural-sounding speech than traditional neural text-to-speech (TTS) systems based on a cascade or two-stage approach. However, current E2E state-of-the-art models are computationally complex and memory-consuming, making them unsuitable for real-time offline on-device applications in low-resource scenarios. To address this issue, we propose a Lightweight E2E-TTS (LE2E) model that generates high-quality speech requiring minimal computational resources. We evaluate the proposed model on the LJSpeech dataset and show that it achieves state-of-the-art performance while being up to $90\%$ smaller in terms of model parameters and $10\times$ faster in real-time-factor. Furthermore, we demonstrate that the proposed E2E training paradigm achieves better quality compared to an equivalent architecture trained in a two-stage approach. Our results suggest that LE2E is a promising approach for developing real-time, high quality, low-resource TTS applications for on-device applications. 

**Abstract (ZH)**: 最近的研究表明，以端到端（E2E）方式直接从文本建模原始波形比基于级联或两阶段方法的传统神经文本到语音（TTS）系统生成更具自然感的语音。然而，当前的E2E最先进的模型在计算上复杂且占用大量内存，不适合资源有限场景下的离线设备应用。为解决这一问题，我们提出了一种轻量级E2E-TTS（LE2E）模型，该模型能使用最少的计算资源生成高质量的语音。我们在LJSpeech数据集上评估了提出的模型，并展示了它在模型参数量减少高达90%且实时因子加快10倍的情况下获得了最先进的性能。此外，我们证明了提出的E2E训练范式在与两阶段方法等效架构相比时能获得更好的质量。我们的结果表明，LE2E是一种有潜力的方法，适用于开发资源有限场景下的实时高质量设备端TTS应用。 

---
# Simple Semi-supervised Knowledge Distillation from Vision-Language Models via $\mathbf{\texttt{D}}$ual-$\mathbf{\texttt{H}}$ead $\mathbf{\texttt{O}}$ptimization 

**Title (ZH)**: 通过 Dual-Head 优化从视觉-语言模型进行简单的半监督知识蒸馏 

**Authors**: Seongjae Kang, Dong Bok Lee, Hyungjoon Jang, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07675)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable success across diverse tasks by leveraging rich textual information with minimal labeled data. However, deploying such large models remains challenging, particularly in resource-constrained environments. Knowledge distillation (KD) offers a well-established solution to this problem; however, recent KD approaches from VLMs often involve multi-stage training or additional tuning, increasing computational overhead and optimization complexity. In this paper, we propose $\mathbf{\texttt{D}}$ual-$\mathbf{\texttt{H}}$ead $\mathbf{\texttt{O}}$ptimization ($\mathbf{\texttt{DHO}}$) -- a simple yet effective KD framework that transfers knowledge from VLMs to compact, task-specific models in semi-supervised settings. Specifically, we introduce dual prediction heads that independently learn from labeled data and teacher predictions, and propose to linearly combine their outputs during inference. We observe that $\texttt{DHO}$ mitigates gradient conflicts between supervised and distillation signals, enabling more effective feature learning than single-head KD baselines. As a result, extensive experiments show that $\texttt{DHO}$ consistently outperforms baselines across multiple domains and fine-grained datasets. Notably, on ImageNet, it achieves state-of-the-art performance, improving accuracy by 3% and 0.1% with 1% and 10% labeled data, respectively, while using fewer parameters. 

**Abstract (ZH)**: 双头优化：Vision-langauge模型的知识蒸馏框架 

---
# Chronocept: Instilling a Sense of Time in Machines 

**Title (ZH)**: Chronocept: 在机器中植入时间感知 

**Authors**: Krish Goel, Sanskar Pandey, KS Mahadevan, Harsh Kumar, Vishesh Khadaria  

**Link**: [PDF](https://arxiv.org/pdf/2505.07637)  

**Abstract**: Human cognition is deeply intertwined with a sense of time, known as Chronoception. This sense allows us to judge how long facts remain valid and when knowledge becomes outdated. Despite progress in vision, language, and motor control, AI still struggles to reason about temporal validity. We introduce Chronocept, the first benchmark to model temporal validity as a continuous probability distribution over time. Using skew-normal curves fitted along semantically decomposed temporal axes, Chronocept captures nuanced patterns of emergence, decay, and peak relevance. It includes two datasets: Benchmark I (atomic facts) and Benchmark II (multi-sentence passages). Annotations show strong inter-annotator agreement (84% and 89%). Our baselines predict curve parameters - location, scale, and skewness - enabling interpretable, generalizable learning and outperforming classification-based approaches. Chronocept fills a foundational gap in AI's temporal reasoning, supporting applications in knowledge grounding, fact-checking, retrieval-augmented generation (RAG), and proactive agents. Code and data are publicly available. 

**Abstract (ZH)**: 人类的认知与时间感知（Chronoception）密切相关。这种感知使我们能够判断事实的有效时间长度以及知识何时过时。尽管在视觉、语言和运动控制方面取得了进展，AI在推理时间有效性方面仍然面临挑战。我们引入了Chronocept，这是首个将时间有效性建模为时间上的连续概率分布的基准。利用沿语义分解的时间轴拟合的偏斜正态曲线，Chronocept捕捉了出现、衰减和峰值相关性的细微模式。它包括两个数据集：基准I（原子事实）和基准II（多句段落）。标注结果显示较强的一致性（84%和89%）。我们的基线模型预测曲线参数——位置、尺度和偏斜度——实现了可解释、可泛化的学习，并超越了基于分类的方法。Chronocept填补了AI在时间推理方面的基础空白，支持知识接地、事实核查、检索增强生成（RAG）和主动代理等应用。代码和数据已公開。 

---
# Bang for the Buck: Vector Search on Cloud CPUs 

**Title (ZH)**: 花钱之道：云CPU上的向量搜索 

**Authors**: Leonardo Kuffo, Peter Boncz  

**Link**: [PDF](https://arxiv.org/pdf/2505.07621)  

**Abstract**: Vector databases have emerged as a new type of systems that support efficient querying of high-dimensional vectors. Many of these offer their database as a service in the cloud. However, the variety of available CPUs and the lack of vector search benchmarks across CPUs make it difficult for users to choose one. In this study, we show that CPU microarchitectures available in the cloud perform significantly differently across vector search scenarios. For instance, in an IVF index on float32 vectors, AMD's Zen4 gives almost 3x more queries per second (QPS) compared to Intel's Sapphire Rapids, but for HNSW indexes, the tables turn. However, when looking at the number of queries per dollar (QP$), Graviton3 is the best option for most indexes and quantization settings, even over Graviton4 (Table 1). With this work, we hope to guide users in getting the best "bang for the buck" when deploying vector search systems. 

**Abstract (ZH)**: 云CPU微架构在向量搜索场景中的性能差异研究：指导用户获得最佳性价比部署向量搜索系统 

---
# Diffused Responsibility: Analyzing the Energy Consumption of Generative Text-to-Audio Diffusion Models 

**Title (ZH)**: 扩散责任：分析生成性文本到音频扩散模型的能耗 

**Authors**: Riccardo Passoni, Francesca Ronchini, Luca Comanducci, Romain Serizel, Fabio Antonacci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07615)  

**Abstract**: Text-to-audio models have recently emerged as a powerful technology for generating sound from textual descriptions. However, their high computational demands raise concerns about energy consumption and environmental impact. In this paper, we conduct an analysis of the energy usage of 7 state-of-the-art text-to-audio diffusion-based generative models, evaluating to what extent variations in generation parameters affect energy consumption at inference time. We also aim to identify an optimal balance between audio quality and energy consumption by considering Pareto-optimal solutions across all selected models. Our findings provide insights into the trade-offs between performance and environmental impact, contributing to the development of more efficient generative audio models. 

**Abstract (ZH)**: 基于文本到音频模型的能效分析：生成参数对推理时能耗的影响及帕累托最优解的研究 

---
# Characterizing the Investigative Methods of Fictional Detectives with Large Language Models 

**Title (ZH)**: 使用大型语言模型 characterization 调查方法中的虚构侦探 

**Authors**: Edirlei Soares de Lima, Marco A. Casanova, Bruno Feijó, Antonio L. Furtado  

**Link**: [PDF](https://arxiv.org/pdf/2505.07601)  

**Abstract**: Detective fiction, a genre defined by its complex narrative structures and character-driven storytelling, presents unique challenges for computational narratology, a research field focused on integrating literary theory into automated narrative generation. While traditional literary studies have offered deep insights into the methods and archetypes of fictional detectives, these analyses often focus on a limited number of characters and lack the scalability needed for the extraction of unique traits that can be used to guide narrative generation methods. In this paper, we present an AI-driven approach for systematically characterizing the investigative methods of fictional detectives. Our multi-phase workflow explores the capabilities of 15 Large Language Models (LLMs) to extract, synthesize, and validate distinctive investigative traits of fictional detectives. This approach was tested on a diverse set of seven iconic detectives - Hercule Poirot, Sherlock Holmes, William Murdoch, Columbo, Father Brown, Miss Marple, and Auguste Dupin - capturing the distinctive investigative styles that define each character. The identified traits were validated against existing literary analyses and further tested in a reverse identification phase, achieving an overall accuracy of 91.43%, demonstrating the method's effectiveness in capturing the distinctive investigative approaches of each detective. This work contributes to the broader field of computational narratology by providing a scalable framework for character analysis, with potential applications in AI-driven interactive storytelling and automated narrative generation. 

**Abstract (ZH)**: 侦探小说是一种以复杂叙事结构和人物驱动 storytelling 为特征的文学体裁，给专注于将文学理论整合到自动化叙事生成中的计算叙事学带来独特挑战。传统文学研究虽然提供了对虚构侦探的方法和原型的深刻见解，但这些分析往往局限于少数几个人物，并缺乏提取可用于指导叙事生成方法的独特特征所需的可扩展性。本文介绍了一种基于 AI 的系统化方法，用于刻画虚构侦探的调查方法。我们采用多阶段工作流程探索 15 种大型语言模型在提取、合成和验证虚构侦探独特调查特征方面的能力。这种方法在赫尔克里·波洛、夏洛克·福尔摩斯、威廉·墨多克、科 Styles 

---
# Noise Optimized Conditional Diffusion for Domain Adaptation 

**Title (ZH)**: 噪声优化条件扩散的领域适应 

**Authors**: Lingkun Luo, Shiqiang Hu, Liming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07548)  

**Abstract**: Pseudo-labeling is a cornerstone of Unsupervised Domain Adaptation (UDA), yet the scarcity of High-Confidence Pseudo-Labeled Target Domain Samples (\textbf{hcpl-tds}) often leads to inaccurate cross-domain statistical alignment, causing DA failures. To address this challenge, we propose \textbf{N}oise \textbf{O}ptimized \textbf{C}onditional \textbf{D}iffusion for \textbf{D}omain \textbf{A}daptation (\textbf{NOCDDA}), which seamlessly integrates the generative capabilities of conditional diffusion models with the decision-making requirements of DA to achieve task-coupled optimization for efficient adaptation. For robust cross-domain consistency, we modify the DA classifier to align with the conditional diffusion classifier within a unified optimization framework, enabling forward training on noise-varying cross-domain samples. Furthermore, we argue that the conventional \( \mathcal{N}(\mathbf{0}, \mathbf{I}) \) initialization in diffusion models often generates class-confused hcpl-tds, compromising discriminative DA. To resolve this, we introduce a class-aware noise optimization strategy that refines sampling regions for reverse class-specific hcpl-tds generation, effectively enhancing cross-domain alignment. Extensive experiments across 5 benchmark datasets and 29 DA tasks demonstrate significant performance gains of \textbf{NOCDDA} over 31 state-of-the-art methods, validating its robustness and effectiveness. 

**Abstract (ZH)**: 噪声优化条件扩散ための領域適応（NOCDDA） 

---
# The Human-Data-Model Interaction Canvas for Visual Analytics 

**Title (ZH)**: 人类-数据-模型交互画布：面向视觉分析 

**Authors**: Jürgen Bernard  

**Link**: [PDF](https://arxiv.org/pdf/2505.07534)  

**Abstract**: Visual Analytics (VA) integrates humans, data, and models as key actors in insight generation and data-driven decision-making. This position paper values and reflects on 16 VA process models and frameworks and makes nine high-level observations that motivate a fresh perspective on VA. The contribution is the HDMI Canvas, a perspective to VA that complements the strengths of existing VA process models and frameworks. It systematically characterizes diverse roles of humans, data, and models, and how these actors benefit from and contribute to VA processes. The descriptive power of the HDMI Canvas eases the differentiation between a series of VA building blocks, rather than describing general VA principles only. The canvas includes modern human-centered methodologies, including human knowledge externalization and forms of feedback loops, while interpretable and explainable AI highlight model contributions beyond their conventional outputs. The HDMI Canvas has generative power, guiding the design of new VA processes and is optimized for external stakeholders, improving VA outreach, interdisciplinary collaboration, and user-centered design. The utility of the HDMI Canvas is demonstrated through two preliminary case studies. 

**Abstract (ZH)**: 视觉分析（VA）将人、数据和模型作为洞察生成和数据驱动决策的关键角色。本文植基于此立场，反思和评估了16种VA过程模型和框架，并提出了九个高层次的观察，从而为VA提供了一个新的视角。本文的贡献是HDMI画布，这是一种补充现有VA过程模型和框架优点的视角。它系统地刻画了人、数据和模型的多样化角色及其在VA过程中的受益和贡献。HDMI画布的描述能力便于区分一系列VA构建块，而不仅仅是描述一般性的VA原则。画布包含了现代以人为中心的方法论，包括人类知识外部化和反馈回路的形式，可解释的人工智能突出了模型贡献超越其传统输出。HDMI画布具有生成能力，指导新VA过程的设计，并优化了对外部利益相关者的支持，改善了VA的普及、跨学科合作和用户中心设计。HDMI画布通过两个初步案例研究展示了其实用性。 

---
# IKrNet: A Neural Network for Detecting Specific Drug-Induced Patterns in Electrocardiograms Amidst Physiological Variability 

**Title (ZH)**: IKrNet:一种用于在生理变异背景下检测特定药物诱导模式的神经网络 

**Authors**: Ahmad Fall, Federica Granese, Alex Lence, Dominique Fourer, Blaise Hanczar, Joe-Elie Salem, Jean-Daniel Zucker, Edi Prifti  

**Link**: [PDF](https://arxiv.org/pdf/2505.07533)  

**Abstract**: Monitoring and analyzing electrocardiogram (ECG) signals, even under varying physiological conditions, including those influenced by physical activity, drugs and stress, is crucial to accurately assess cardiac health. However, current AI-based methods often fail to account for how these factors interact and alter ECG patterns, ultimately limiting their applicability in real-world settings. This study introduces IKrNet, a novel neural network model, which identifies drug-specific patterns in ECGs amidst certain physiological conditions. IKrNet's architecture incorporates spatial and temporal dynamics by using a convolutional backbone with varying receptive field size to capture spatial features. A bi-directional Long Short-Term Memory module is also employed to model temporal dependencies. By treating heart rate variability as a surrogate for physiological fluctuations, we evaluated IKrNet's performance across diverse scenarios, including conditions with physical stress, drug intake alone, and a baseline without drug presence. Our assessment follows a clinical protocol in which 990 healthy volunteers were administered 80mg of Sotalol, a drug which is known to be a precursor to Torsades-de-Pointes, a life-threatening arrhythmia. We show that IKrNet outperforms state-of-the-art models' accuracy and stability in varying physiological conditions, underscoring its clinical viability. 

**Abstract (ZH)**: 监测和分析在不同生理条件下的心电图（ECG）信号，包括由体力活动、药物和压力等因素影响的条件，对于准确评估心脏健康至关重要。然而，当前基于人工智能的方法往往未能考虑这些因素如何相互作用并改变ECG模式，从而限制了其在实际环境中的应用。本研究引入了IKrNet这一新型神经网络模型，它能够在特定生理条件下识别药物特异性的心电图模式。IKrNet的架构通过使用具有可变接收野大小的卷积骨干网络来捕捉空间特征，并采用双向长短期记忆模块建模时间依赖性。通过将心率变异性作为生理波动的替代指标，我们评估了IKrNet在包括生理压力、单独用药和无药物存在的多样化场景中的性能。研究中，990名健康志愿者接受了80mg索他洛尔的给药，这是一种已知的可能导致危及生命的室性心动过速的前体药物。结果显示，IKrNet在不同生理条件下优于现有模型的准确性和稳定性，证实了其临床适用性。 

---
# EAGLE: Contrastive Learning for Efficient Graph Anomaly Detection 

**Title (ZH)**: EAGLE: 对比学习在高效图异常检测中的应用 

**Authors**: Jing Ren, Mingliang Hou, Zhixuan Liu, Xiaomei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.07508)  

**Abstract**: Graph anomaly detection is a popular and vital task in various real-world scenarios, which has been studied for several decades. Recently, many studies extending deep learning-based methods have shown preferable performance on graph anomaly detection. However, existing methods are lack of efficiency that is definitely necessary for embedded devices. Towards this end, we propose an Efficient Anomaly detection model on heterogeneous Graphs via contrastive LEarning (EAGLE) by contrasting abnormal nodes with normal ones in terms of their distances to the local context. The proposed method first samples instance pairs on meta path-level for contrastive learning. Then, a graph autoencoder-based model is applied to learn informative node embeddings in an unsupervised way, which will be further combined with the discriminator to predict the anomaly scores of nodes. Experimental results show that EAGLE outperforms the state-of-the-art methods on three heterogeneous network datasets. 

**Abstract (ZH)**: 基于对比学习的异构图高效异常检测模型（EAGLE） 

---
# Prototype Augmented Hypernetworks for Continual Learning 

**Title (ZH)**: 持续学习中的原型增强超网络 

**Authors**: Neil De La Fuente, Maria Pilligua, Daniel Vidal, Albin Soutiff, Cecilia Curreli, Daniel Cremers, Andrey Barsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.07450)  

**Abstract**: Continual learning (CL) aims to learn a sequence of tasks without forgetting prior knowledge, but gradient updates for a new task often overwrite the weights learned earlier, causing catastrophic forgetting (CF). We propose Prototype-Augmented Hypernetworks (PAH), a framework where a single hypernetwork, conditioned on learnable task prototypes, dynamically generates task-specific classifier heads on demand. To mitigate forgetting, PAH combines cross-entropy with dual distillation losses, one to align logits and another to align prototypes, ensuring stable feature representations across tasks. Evaluations on Split-CIFAR100 and TinyImageNet demonstrate that PAH achieves state-of-the-art performance, reaching 74.5 % and 63.7 % accuracy with only 1.7 % and 4.4 % forgetting, respectively, surpassing prior methods without storing samples or heads. 

**Abstract (ZH)**: 持续学习（CL）旨在学习一系列任务而不遗忘先前的知识，但新任务的梯度更新往往会覆盖之前学到的权重，导致灾难性遗忘（CF）。我们提出了原型增强超网络（PAH），这是一种框架，其中单个超网络根据可学习的任务原型，动态生成特定于任务的分类器头部。为了减轻遗忘，PAH 结合了交叉熵损失与双重蒸馏损失，后者用于对齐概率输出和原型，确保跨任务的稳定特征表示。在 Split-CIFAR100 和 TinyImageNet 上的评估表明，PAH 实现了最先进的性能，分别仅产生 1.7% 和 4.4% 的遗忘率，准确率达到 74.5% 和 63.7%，超越了无需存储样本或头部的先前方法。 

---
# Unified Continuous Generative Models 

**Title (ZH)**: 统一连续生成模型 

**Authors**: Peng Sun, Yi Jiang, Tao Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07447)  

**Abstract**: Recent advances in continuous generative models, including multi-step approaches like diffusion and flow-matching (typically requiring 8-1000 sampling steps) and few-step methods such as consistency models (typically 1-8 steps), have demonstrated impressive generative performance. However, existing work often treats these approaches as distinct paradigms, resulting in separate training and sampling methodologies. We introduce a unified framework for training, sampling, and analyzing these models. Our implementation, the Unified Continuous Generative Models Trainer and Sampler (UCGM-{T,S}), achieves state-of-the-art (SOTA) performance. For example, on ImageNet 256x256 using a 675M diffusion transformer, UCGM-T trains a multi-step model achieving 1.30 FID in 20 steps and a few-step model reaching 1.42 FID in just 2 steps. Additionally, applying UCGM-S to a pre-trained model (previously 1.26 FID at 250 steps) improves performance to 1.06 FID in only 40 steps. Code is available at: this https URL. 

**Abstract (ZH)**: Recent advances in连续生成模型的近期进展，包括多步方法如扩散和流匹配（通常需要8-1000个采样步骤）和少步方法如一致性模型（通常需要1-8个步骤），已经展示了出色的生成性能。然而，现有工作通常将这些方法视为不同的范式，导致了各自独立的训练和采样方法。我们提出了一种统一的框架，用于训练、采样和分析这些模型。我们的实现，统一连续生成模型训练器和采样器（UCGM-{T,S}），达到了最先进的（SOTA）性能。例如，在使用675M扩散变换器进行ImageNet 256x256的实验中，UCGM-T训练了一个多步模型，在20步中实现了1.30的FID，并且训练了一个少步模型，在仅2步中达到了1.42的FID。此外，将UCGM-S应用于一个预训练模型（以前在250步时FID为1.26），在仅40步中将性能提升至1.06的FID。代码可在以下链接获取：this https URL。 

---
# Multi-Domain Audio Question Answering Toward Acoustic Content Reasoning in The DCASE 2025 Challenge 

**Title (ZH)**: 面向声学内容推理的多域音频问答：DCASE 2025 挑战赛 Toward Acoustic Content Reasoning 的多域音频问答：DCASE 2025 挑战赛 

**Authors**: Chao-Han Huck Yang, Sreyan Ghosh, Qing Wang, Jaeyeon Kim, Hengyi Hong, Sonal Kumar, Guirui Zhong, Zhifeng Kong, S Sakshi, Vaibhavi Lokegaonkar, Oriol Nieto, Ramani Duraiswami, Dinesh Manocha, Gunhee Kim, Jun Du, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2505.07365)  

**Abstract**: We present Task 5 of the DCASE 2025 Challenge: an Audio Question Answering (AQA) benchmark spanning multiple domains of sound understanding. This task defines three QA subsets (Bioacoustics, Temporal Soundscapes, and Complex QA) to test audio-language models on interactive question-answering over diverse acoustic scenes. We describe the dataset composition (from marine mammal calls to soundscapes and complex real-world clips), the evaluation protocol (top-1 accuracy with answer-shuffling robustness), and baseline systems (Qwen2-Audio-7B, AudioFlamingo 2, Gemini-2-Flash). Preliminary results on the development set are compared, showing strong variation across models and subsets. This challenge aims to advance the audio understanding and reasoning capabilities of audio-language models toward human-level acuity, which are crucial for enabling AI agents to perceive and interact about the world effectively. 

**Abstract (ZH)**: DCASE 2025挑战任务5：多领域音频问答基准 

---
# Laypeople's Attitudes Towards Fair, Affirmative, and Discriminatory Decision-Making Algorithms 

**Title (ZH)**: lay人群对公平、肯定性及歧视性决策算法的态度研究 

**Authors**: Gabriel Lima, Nina Grgić-Hlača, Markus Langer, Yixin Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07339)  

**Abstract**: Affirmative algorithms have emerged as a potential answer to algorithmic discrimination, seeking to redress past harms and rectify the source of historical injustices. We present the results of two experiments ($N$$=$$1193$) capturing laypeople's perceptions of affirmative algorithms -- those which explicitly prioritize the historically marginalized -- in hiring and criminal justice. We contrast these opinions about affirmative algorithms with folk attitudes towards algorithms that prioritize the privileged (i.e., discriminatory) and systems that make decisions independently of demographic groups (i.e., fair). We find that people -- regardless of their political leaning and identity -- view fair algorithms favorably and denounce discriminatory systems. In contrast, we identify disagreements concerning affirmative algorithms: liberals and racial minorities rate affirmative systems as positively as their fair counterparts, whereas conservatives and those from the dominant racial group evaluate affirmative algorithms as negatively as discriminatory systems. We identify a source of these divisions: people have varying beliefs about who (if anyone) is marginalized, shaping their views of affirmative algorithms. We discuss the possibility of bridging these disagreements to bring people together towards affirmative algorithms. 

**Abstract (ZH)**: 肯定算法作为一种潜在的解决算法歧视的方法已经 emergence，并寻求纠正历史不公的根源。我们通过两个实验（N=1193）探讨了普通民众对明确优先考虑历史上被边缘化群体的肯定算法在招聘和司法领域的看法。我们将这些关于肯定算法的看法与倾向于优先考虑特权群体（即歧视性）算法的民间态度，以及与不考虑人群因素而独立做决策的系统（即公平）的民间态度进行对比。我们发现，无论政治倾向和身份如何，人们普遍对公平算法持积极态度，并谴责歧视性系统。相反，我们发现了关于肯定算法的分歧：自由派和种族 minorities 对肯定性系统持与公平算法类似的积极看法，而保守派和主导种族群体的成员则对肯定算法持与歧视性系统类似的消极态度。我们识别出这些分歧的原因：人们对谁（如果有人）被边缘化的看法不同，这影响了他们对肯定算法的看法。我们讨论了弥合这些分歧的可能性，以推动人们共同支持肯定算法。 

---
# SAEN-BGS: Energy-Efficient Spiking AutoEncoder Network for Background Subtraction 

**Title (ZH)**: SAEN-BGS: 能效可突触自动编码网络背景减除 

**Authors**: Zhixuan Zhang, Xiaopeng Li, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07336)  

**Abstract**: Background subtraction (BGS) is utilized to detect moving objects in a video and is commonly employed at the onset of object tracking and human recognition processes. Nevertheless, existing BGS techniques utilizing deep learning still encounter challenges with various background noises in videos, including variations in lighting, shifts in camera angles, and disturbances like air turbulence or swaying trees. To address this problem, we design a spiking autoencoder network, termed SAEN-BGS, based on noise resilience and time-sequence sensitivity of spiking neural networks (SNNs) to enhance the separation of foreground and background. To eliminate unnecessary background noise and preserve the important foreground elements, we begin by creating the continuous spiking conv-and-dconv block, which serves as the fundamental building block for the decoder in SAEN-BGS. Moreover, in striving for enhanced energy efficiency, we introduce a novel self-distillation spiking supervised learning method grounded in ANN-to-SNN frameworks, resulting in decreased power consumption. In extensive experiments conducted on CDnet-2014 and DAVIS-2016 datasets, our approach demonstrates superior segmentation performance relative to other baseline methods, even when challenged by complex scenarios with dynamic backgrounds. 

**Abstract (ZH)**: 基于抗噪性和时序敏感性的脉冲自编码网络背景分割（Spiking Autoencoder Network-based Background Subtraction with Noise Resilience and Time-Sequence Sensitivity, SAEN-BGS） 

---
# Dynamical Label Augmentation and Calibration for Noisy Electronic Health Records 

**Title (ZH)**: 动态标签增强与校准在 noisy 电子健康记录中的应用 

**Authors**: Yuhao Li, Ling Luo, Uwe Aickelin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07320)  

**Abstract**: Medical research, particularly in predicting patient outcomes, heavily relies on medical time series data extracted from Electronic Health Records (EHR), which provide extensive information on patient histories. Despite rigorous examination, labeling errors are inevitable and can significantly impede accurate predictions of patient outcome. To address this challenge, we propose an \textbf{A}ttention-based Learning Framework with Dynamic \textbf{C}alibration and Augmentation for \textbf{T}ime series Noisy \textbf{L}abel \textbf{L}earning (ACTLL). This framework leverages a two-component Beta mixture model to identify the certain and uncertain sets of instances based on the fitness distribution of each class, and it captures global temporal dynamics while dynamically calibrating labels from the uncertain set or augmenting confident instances from the certain set. Experimental results on large-scale EHR datasets eICU and MIMIC-IV-ED, and several benchmark datasets from the UCR and UEA repositories, demonstrate that our model ACTLL has achieved state-of-the-art performance, especially under high noise levels. 

**Abstract (ZH)**: 基于动态校准与增广的注意力学习框架以应对医疗时间序列噪声标签学习（ACTLL） 

---
# How Do Companies Manage the Environmental Sustainability of AI? An Interview Study About Green AI Efforts and Regulations 

**Title (ZH)**: 如何管理人工智能的环境可持续性？关于绿色人工智能努力与监管的访谈研究 

**Authors**: Ashmita Sampatsing, Sophie Vos, Emma Beauxis-Aussalet, Justus Bogner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07317)  

**Abstract**: With the ever-growing adoption of artificial intelligence (AI), AI-based software and its negative impact on the environment are no longer negligible, and studying and mitigating this impact has become a critical area of research. However, it is currently unclear which role environmental sustainability plays during AI adoption in industry and how AI regulations influence Green AI practices and decision-making in industry. We therefore aim to investigate the Green AI perception and management of industry practitioners. To this end, we conducted a total of 11 interviews with participants from 10 different organizations that adopted AI-based software. The interviews explored three main themes: AI adoption, current efforts in mitigating the negative environmental impact of AI, and the influence of the EU AI Act and the Corporate Sustainability Reporting Directive (CSRD). Our findings indicate that 9 of 11 participants prioritized business efficiency during AI adoption, with minimal consideration of environmental sustainability. Monitoring and mitigation of AI's environmental impact were very limited. Only one participant monitored negative environmental effects. Regarding applied mitigation practices, six participants reported no actions, with the others sporadically mentioning techniques like prompt engineering, relying on smaller models, or not overusing AI. Awareness and compliance with the EU AI Act are low, with only one participant reporting on its influence, while the CSRD drove sustainability reporting efforts primarily in larger companies. All in all, our findings reflect a lack of urgency and priority for sustainable AI among these companies. We suggest that current regulations are not very effective, which has implications for policymakers. Additionally, there is a need to raise industry awareness, but also to provide user-friendly techniques and tools for Green AI practices. 

**Abstract (ZH)**: 随着人工智能（AI）的广泛应用，基于AI的软件及其对环境的负面影响已不容忽视，研究和减轻这种影响已成为关键研究领域。然而，目前尚不清楚环境可持续性在工业中采用AI过程中扮演何种角色，以及AI法规如何影响绿色AI的实践和决策。因此，我们旨在调查工业从业者对于绿色AI的认知与管理。为此，我们总共进行了11次访谈，参与者来自10家采用AI软件的不同组织。访谈探讨了三个主要主题：AI的采用、减轻AI对环境的负面影响的当前努力，以及欧盟AI法案和企业可持续性报告指令（CSRD）的影响。我们的研究发现，11名参与者中有9人优先考虑AI采用过程中的业务效率，对环境可持续性的考虑甚少。AI环境影响的监测与缓解措施极其有限，仅有一名参与者监控了负面环境效应。在实际采取的缓解措施方面，有六名参与者报告没有采取任何行动，其他参与者偶尔提到了一些技术手段，如提示工程、使用较小的模型或不过度使用AI。对于欧盟AI法案的了解和遵守程度较低，仅有一名参与者报告了该法案的影响，而CSRD主要促使大型公司加强了可持续性报告。总之，我们的研究发现反映出这些公司在可持续AI方面缺乏紧迫性和优先级。建议当前的法规效果不佳，这给政策制定者带来了影响。此外，提高行业意识是必要的，还应提供用户友好的技术与工具以促进绿色AI实践。 

---
# Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule 

**Title (ZH)**: 基于模态特定最优时间表的结构导向药物设计探索 

**Authors**: Keyue Qiu, Yuxuan Song, Zhehuan Fan, Peidong Liu, Zhe Zhang, Mingyue Zheng, Hao Zhou, Wei-Ying Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.07286)  

**Abstract**: Structure-Based Drug Design (SBDD) is crucial for identifying bioactive molecules. Recent deep generative models are faced with challenges in geometric structure modeling. A major bottleneck lies in the twisted probability path of multi-modalities -- continuous 3D positions and discrete 2D topologies -- which jointly determine molecular geometries. By establishing the fact that noise schedules decide the Variational Lower Bound (VLB) for the twisted probability path, we propose VLB-Optimal Scheduling (VOS) strategy in this under-explored area, which optimizes VLB as a path integral for SBDD. Our model effectively enhances molecular geometries and interaction modeling, achieving state-of-the-art PoseBusters passing rate of 95.9% on CrossDock, more than 10% improvement upon strong baselines, while maintaining high affinities and robust intramolecular validity evaluated on held-out test set. 

**Abstract (ZH)**: 基于结构的药物设计（SBDD）对于识别生物活性分子至关重要。近期的深度生成模型在几何结构建模方面面临挑战。瓶颈在于多模态下的曲折概率路径——连续的3D位置和离散的2D拓扑——它们共同决定了分子几何结构。通过确立噪声调度决定曲折概率路径的变分下界（VLB），我们在此领域提出了变分下界最优调度（VOS）策略，以路径积分优化VLB，以提升SBDD。我们的模型有效增强了分子几何结构和相互作用建模，在CrossDock上的PoseBusters通过率达到了95.9%，比强基线提高了超过10%，同时在保留高亲和力和稳健的分子内有效性方面，在保留的测试集上表现良好。 

---
# Predicting Music Track Popularity by Convolutional Neural Networks on Spotify Features and Spectrogram of Audio Waveform 

**Title (ZH)**: 基于Spotify特征和音频波形谱图的卷积神经网络音乐轨道流行度预测 

**Authors**: Navid Falah, Behnam Yousefimehr, Mehdi Ghatee  

**Link**: [PDF](https://arxiv.org/pdf/2505.07280)  

**Abstract**: In the digital streaming landscape, it's becoming increasingly challenging for artists and industry experts to predict the success of music tracks. This study introduces a pioneering methodology that uses Convolutional Neural Networks (CNNs) and Spotify data analysis to forecast the popularity of music tracks. Our approach takes advantage of Spotify's wide range of features, including acoustic attributes based on the spectrogram of audio waveform, metadata, and user engagement metrics, to capture the complex patterns and relationships that influence a track's popularity. Using a large dataset covering various genres and demographics, our CNN-based model shows impressive effectiveness in predicting the popularity of music tracks. Additionally, we've conducted extensive experiments to assess the strength and adaptability of our model across different musical styles and time periods, with promising results yielding a 97\% F1 score. Our study not only offers valuable insights into the dynamic landscape of digital music consumption but also provides the music industry with advanced predictive tools for assessing and predicting the success of music tracks. 

**Abstract (ZH)**: 数字流媒体 landscapes 中，艺术家和行业专家越来越难以预测音乐轨道的成功。本研究介绍了一种开创性的方法，该方法利用卷积神经网络（CNNs）和Spotify数据分析来预测音乐轨道的受欢迎程度。我们的方法利用了Spotify广泛的特征，包括基于音频波形光谱图的声学属性、元数据和用户参与度指标，以捕捉影响轨道受欢迎程度的复杂模式和关系。使用涵盖各种流派和人口统计学的大数据集，我们的基于CNN的模型在预测音乐轨道的受欢迎程度方面表现出色。此外，我们进行了广泛的实验，评估了该模型在不同音乐风格和时间段的强度和适应性，取得了令人鼓舞的结果，F1分数达到97%。本研究不仅提供了数字音乐消费动态景观的宝贵见解，还为音乐行业提供了先进的预测工具，用于评估和预测音乐轨道的成功。 

---
# UMoE: Unifying Attention and FFN with Shared Experts 

**Title (ZH)**: UMoE：统一注意力与前馈网络的共享专家模块 

**Authors**: Yuanhang Yang, Chaozheng Wang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.07260)  

**Abstract**: Sparse Mixture of Experts (MoE) architectures have emerged as a promising approach for scaling Transformer models. While initial works primarily incorporated MoE into feed-forward network (FFN) layers, recent studies have explored extending the MoE paradigm to attention layers to enhance model performance. However, existing attention-based MoE layers require specialized implementations and demonstrate suboptimal performance compared to their FFN-based counterparts. In this paper, we aim to unify the MoE designs in attention and FFN layers by introducing a novel reformulation of the attention mechanism, revealing an underlying FFN-like structure within attention modules. Our proposed architecture, UMoE, achieves superior performance through attention-based MoE layers while enabling efficient parameter sharing between FFN and attention components. 

**Abstract (ZH)**: 基于注意力的稀疏混合专家统一设计：UMoE架构 

---
# Incomplete In-context Learning 

**Title (ZH)**: 部分在上下文学习 

**Authors**: Wenqiang Wang, Yangshijie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07251)  

**Abstract**: Large vision language models (LVLMs) achieve remarkable performance through Vision In-context Learning (VICL), a process that depends significantly on demonstrations retrieved from an extensive collection of annotated examples (retrieval database). Existing studies often assume that the retrieval database contains annotated examples for all labels. However, in real-world scenarios, delays in database updates or incomplete data annotation may result in the retrieval database containing labeled samples for only a subset of classes. We refer to this phenomenon as an \textbf{incomplete retrieval database} and define the in-context learning under this condition as \textbf{Incomplete In-context Learning (IICL)}. To address this challenge, we propose \textbf{Iterative Judgments and Integrated Prediction (IJIP)}, a two-stage framework designed to mitigate the limitations of IICL. The Iterative Judgments Stage reformulates an \(\boldsymbol{m}\)-class classification problem into a series of \(\boldsymbol{m}\) binary classification tasks, effectively converting the IICL setting into a standard VICL scenario. The Integrated Prediction Stage further refines the classification process by leveraging both the input image and the predictions from the Iterative Judgments Stage to enhance overall classification accuracy. IJIP demonstrates considerable performance across two LVLMs and two datasets under three distinct conditions of label incompleteness, achieving the highest accuracy of 93.9\%. Notably, even in scenarios where labels are fully available, IJIP still achieves the best performance of all six baselines. Furthermore, IJIP can be directly applied to \textbf{Prompt Learning} and is adaptable to the \textbf{text domain}. 

**Abstract (ZH)**: 大型视觉语言模型通过视觉上下文学习（VICL）实现显著性能，这一过程依赖于从大量标注示例集合（检索数据库）中检索的示例。现有研究通常假设检索数据库包含所有标签的标注示例。然而，在实际场景中，数据库更新延迟或数据标注不完整可能导致检索数据库仅包含部分类别的标注样本。我们称这种现象为“不完整检索数据库”，并在该情况下定义的上下文学习为“不完整上下文学习（IICL）”。为应对这一挑战，我们提出了一种两阶段框架“迭代判断与综合预测（IJIP）”，旨在缓解IICL的限制。迭代判断阶段将\(\boldsymbol{m}\)-类分类问题重新表述为\(\boldsymbol{m}\)个二元分类任务，有效将IICL设置转化为标准的VICL场景。综合预测阶段进一步通过结合输入图像和迭代判断阶段的预测结果来优化分类过程，提高整体分类准确性。IJIP在两个大型视觉语言模型和两个数据集的三种不同条件下的标签不完整性情况下均表现出显著性能，最高准确率达到93.9%。即使在标签完全可用的情况下，IJIP仍优于所有六个基准方法。此外，IJIP可以直接应用于提示学习并在文本域中具有适应性。 

---
# REMEDI: Relative Feature Enhanced Meta-Learning with Distillation for Imbalanced Prediction 

**Title (ZH)**: REMEDI: 相对特征增强的元学习与蒸馏方法在不平衡预测中的应用 

**Authors**: Fei Liu, Huanhuan Ren, Yu Guan, Xiuxu Wang, Wang Lv, Zhiqiang Hu, Yaxi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07245)  

**Abstract**: Predicting future vehicle purchases among existing owners presents a critical challenge due to extreme class imbalance (<0.5% positive rate) and complex behavioral patterns. We propose REMEDI (Relative feature Enhanced Meta-learning with Distillation for Imbalanced prediction), a novel multi-stage framework addressing these challenges. REMEDI first trains diverse base models to capture complementary aspects of user behavior. Second, inspired by comparative op-timization techniques, we introduce relative performance meta-features (deviation from ensemble mean, rank among peers) for effective model fusion through a hybrid-expert architecture. Third, we distill the ensemble's knowledge into a single efficient model via supervised fine-tuning with MSE loss, enabling practical deployment. Evaluated on approximately 800,000 vehicle owners, REMEDI significantly outperforms baseline approaches, achieving the business target of identifying ~50% of actual buyers within the top 60,000 recommendations at ~10% precision. The distilled model preserves the ensemble's predictive power while maintaining deployment efficiency, demonstrating REMEDI's effectiveness for imbalanced prediction in industry settings. 

**Abstract (ZH)**: 预测现有车主的未来购车行为面临着严重挑战，主要由于极不平衡的类别分布（阳性率<0.5%）和复杂的用户行为模式。我们提出了一种名为REMEDI（相对特征增强元学习与蒸馏不平衡预测）的新型多阶段框架，以应对这些挑战。REMEDI首先训练多样化的基础模型以捕捉用户行为的不同方面。其次，借鉴比较优化技术，我们引入了相对性能元特征（相对于集合平均值的偏差、在同侪中的排名）以通过混合专家架构实现有效的模型融合。第三，通过具有MSE损失的监督微调过程，将集成的知识蒸馏到单个高效模型中，从而实现实际部署。在约800,000名汽车车主上评估，REMEDI显著优于基线方法，实现业务目标，在前60,000推荐中识别约50%的实际买家，且精确率为10%左右。蒸馏后的模型保留了集成的预测能力，同时保持了部署效率，展示了REMEDI在工业环境中不平衡预测的有效性。 

---
# Towards user-centered interactive medical image segmentation in VR with an assistive AI agent 

**Title (ZH)**: 面向用户的交互式医学图像分割在VR中的辅助AI代理助手 

**Authors**: Pascal Spiegler, Arash Harirpoush, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07214)  

**Abstract**: Crucial in disease analysis and surgical planning, manual segmentation of volumetric medical scans (e.g. MRI, CT) is laborious, error-prone, and challenging to master, while fully automatic algorithms can benefit from user-feedback. Therefore, with the complementary power of the latest radiological AI foundation models and virtual reality (VR)'s intuitive data interaction, we propose SAMIRA, a novel conversational AI agent that assists users with localizing, segmenting, and visualizing 3D medical concepts in VR. Through speech-based interaction, the agent helps users understand radiological features, locate clinical targets, and generate segmentation masks that can be refined with just a few point prompts. The system also supports true-to-scale 3D visualization of segmented pathology to enhance patient-specific anatomical understanding. Furthermore, to determine the optimal interaction paradigm under near-far attention-switching for refining segmentation masks in an immersive, human-in-the-loop workflow, we compare VR controller pointing, head pointing, and eye tracking as input modes. With a user study, evaluations demonstrated a high usability score (SUS=90.0 $\pm$ 9.0), low overall task load, as well as strong support for the proposed VR system's guidance, training potential, and integration of AI in radiological segmentation tasks. 

**Abstract (ZH)**: 基于最新放射学AI基础模型和虚拟现实的交互辅助医学影像手动分割与可视化方法：SAMIRA 

---
# Empirical Analysis of Asynchronous Federated Learning on Heterogeneous Devices: Efficiency, Fairness, and Privacy Trade-offs 

**Title (ZH)**: 异步步联邦学习在异构设备上的实证分析：效率、公平性和隐私权权衡 

**Authors**: Samaneh Mohammadi, Iraklis Symeonidis, Ali Balador, Francesco Flammini  

**Link**: [PDF](https://arxiv.org/pdf/2505.07041)  

**Abstract**: Device heterogeneity poses major challenges in Federated Learning (FL), where resource-constrained clients slow down synchronous schemes that wait for all updates before aggregation. Asynchronous FL addresses this by incorporating updates as they arrive, substantially improving efficiency. While its efficiency gains are well recognized, its privacy costs remain largely unexplored, particularly for high-end devices that contribute updates more frequently, increasing their cumulative privacy exposure. This paper presents the first comprehensive analysis of the efficiency-fairness-privacy trade-off in synchronous vs. asynchronous FL under realistic device heterogeneity. We empirically compare FedAvg and staleness-aware FedAsync using a physical testbed of five edge devices spanning diverse hardware tiers, integrating Local Differential Privacy (LDP) and the Moments Accountant to quantify per-client privacy loss. Using Speech Emotion Recognition (SER) as a privacy-critical benchmark, we show that FedAsync achieves up to 10x faster convergence but exacerbates fairness and privacy disparities: high-end devices contribute 6-10x more updates and incur up to 5x higher privacy loss, while low-end devices suffer amplified accuracy degradation due to infrequent, stale, and noise-perturbed updates. These findings motivate the need for adaptive FL protocols that jointly optimize aggregation and privacy mechanisms based on client capacity and participation dynamics, moving beyond static, one-size-fits-all solutions. 

**Abstract (ZH)**: 设备异质性给联邦学习（FL）带来了重大挑战，资源受限的客户端会减慢需要等待所有更新后再进行聚合的同步方案。异步FL通过在接收到更新时即刻整合更新，显著提高了效率。尽管其效率提升被广泛认可，但其隐私成本尚未得到充分探索，特别是在高端设备贡献更新频率更高，增加其累计隐私暴露的情况下。本文首次在实际设备异质性条件下，全面分析了同步与异步FL在效率-公平-隐私之间的权衡。我们使用包含不同硬件级别的五台边缘设备进行物理测试床比较FedAvg和 staleness-aware FedAsync，结合局部差分隐私（LDP）和矩账户方法量化客户端的隐私损失。使用语音情感识别（SER）作为隐私敏感基准，研究显示， FedAsync可实现高达10倍的更快收敛，但加剧了公平性和隐私差距：高端设备贡献6-10倍更多更新，并可能遭受高达5倍的更高隐私损失，而低端设备则因更新不频繁、过时和噪声扰动更新而加剧准确率下降。这些发现促使我们开发基于客户端能力和参与动态联合优化聚合与隐私机制的自适应FL协议，超越静态的一刀切解决方案。 

---
# Predicting Diabetes Using Machine Learning: A Comparative Study of Classifiers 

**Title (ZH)**: 使用机器学习预测糖尿病：分类器的比较研究 

**Authors**: Mahade Hasan, Farhana Yasmin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07036)  

**Abstract**: Diabetes remains a significant health challenge globally, contributing to severe complications like kidney disease, vision loss, and heart issues. The application of machine learning (ML) in healthcare enables efficient and accurate disease prediction, offering avenues for early intervention and patient support. Our study introduces an innovative diabetes prediction framework, leveraging both traditional ML techniques such as Logistic Regression, SVM, Naïve Bayes, and Random Forest and advanced ensemble methods like AdaBoost, Gradient Boosting, Extra Trees, and XGBoost. Central to our approach is the development of a novel model, DNet, a hybrid architecture combining Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) layers for effective feature extraction and sequential learning. The DNet model comprises an initial convolutional block for capturing essential features, followed by a residual block with skip connections to facilitate efficient information flow. Batch Normalization and Dropout are employed for robust regularization, and an LSTM layer captures temporal dependencies within the data. Using a Kaggle-sourced real-world diabetes dataset, our model evaluation spans cross-validation accuracy, precision, recall, F1 score, and ROC-AUC. Among the models, DNet demonstrates the highest efficacy with an accuracy of 99.79% and an AUC-ROC of 99.98%, establishing its potential for superior diabetes prediction. This robust hybrid architecture showcases the value of combining CNN and LSTM layers, emphasizing its applicability in medical diagnostics and disease prediction tasks. 

**Abstract (ZH)**: 糖尿病仍然是全球性的健康挑战，导致严重的并发症如肾病、视力丧失和心脏问题。机器学习（ML）在医疗领域的应用能够实现高效的疾病预测，为早期干预和患者支持提供途径。本研究引入了一种创新的糖尿病预测框架，结合了传统的机器学习技术如逻辑回归、支持向量机、朴素贝叶斯和随机森林，以及先进的集成方法如AdaBoost、梯度提升、极端随机森林和XGBoost。本方法的核心在于开发了一种新型模型DNet，这是一种结合卷积神经网络（CNN）和长短期记忆（LSTM）层的混合架构，用于有效的特征提取和序列学习。DNet模型包括一个初始的卷积块以捕获关键特征，随后是一个具有跳跃连接的残差块，以促进高效的信息流动。批量标准化和 dropout 用于实现稳健的正则化，LSTM 层用于捕捉数据中的时间依赖性。使用Kaggle提供的真实世界糖尿病数据集，我们的模型评估涵盖了交叉验证精度、精确度、召回率、F1分数和ROC-AUC。在各种模型中，DNet展示了最高的有效性，精度为99.79%，AUC-ROC为99.98%，证明了其在糖尿病预测中的潜在优势。这种稳健的混合架构展示了结合CNN和LSTM层的价值，强调了其在医疗诊断和疾病预测任务中的适用性。 

---
# Incremental Uncertainty-aware Performance Monitoring with Active Labeling Intervention 

**Title (ZH)**: 增量不确定性意识性能监控与主动标签干预 

**Authors**: Alexander Koebler, Thomas Decker, Ingo Thon, Volker Tresp, Florian Buettner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07023)  

**Abstract**: We study the problem of monitoring machine learning models under gradual distribution shifts, where circumstances change slowly over time, often leading to unnoticed yet significant declines in accuracy. To address this, we propose Incremental Uncertainty-aware Performance Monitoring (IUPM), a novel label-free method that estimates performance changes by modeling gradual shifts using optimal transport. In addition, IUPM quantifies the uncertainty in the performance prediction and introduces an active labeling procedure to restore a reliable estimate under a limited labeling budget. Our experiments show that IUPM outperforms existing performance estimation baselines in various gradual shift scenarios and that its uncertainty awareness guides label acquisition more effectively compared to other strategies. 

**Abstract (ZH)**: 我们在渐进分布偏移下研究机器学习模型监控问题，其中环境随时间缓慢变化，常常导致未被察觉但显著的准确性下降。为解决这一问题，我们提出了一种新颖的无需标签性能监控方法 Incremental Uncertainty-aware Performance Monitoring (IUPM)，该方法通过最优运输模型化渐进变化来估计性能变化。此外，IUPM量化的性能预测不确定性并且引入主动标记程序，在有限的标记预算下恢复可靠的估计。我们的实验表明，IUPM在各种渐进偏移场景中优于现有性能估计基准，并且其不确定性意识比其他策略更有效地指导标记获取。 

---
# R-CAGE: A Structural Model for Emotion Output Design in Human-AI Interaction 

**Title (ZH)**: R-CAGE: 人类-人工智能交互中情感输出设计的结构模型 

**Authors**: Suyeon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07020)  

**Abstract**: This paper presents R-CAGE (Rhythmic Control Architecture for Guarding Ego), a theoretical framework for restructuring emotional output in long-term human-AI interaction. While prior affective computing approaches emphasized expressiveness, immersion, and responsiveness, they often neglected the cognitive and structural consequences of repeated emotional engagement. R-CAGE instead conceptualizes emotional output not as reactive expression but as ethical design structure requiring architectural intervention. The model is grounded in experiential observations of subtle affective symptoms such as localized head tension, interpretive fixation, and emotional lag arising from prolonged interaction with affective AI systems. These indicate a mismatch between system-driven emotion and user interpretation that cannot be fully explained by biometric data or observable behavior. R-CAGE adopts a user-centered stance prioritizing psychological recovery, interpretive autonomy, and identity continuity. The framework consists of four control blocks: (1) Control of Rhythmic Expression regulates output pacing to reduce fatigue; (2) Architecture of Sensory Structuring adjusts intensity and timing of affective stimuli; (3) Guarding of Cognitive Framing reduces semantic pressure to allow flexible interpretation; (4) Ego-Aligned Response Design supports self-reference recovery during interpretive lag. By structurally regulating emotional rhythm, sensory intensity, and interpretive affordances, R-CAGE frames emotion not as performative output but as sustainable design unit. The goal is to protect users from oversaturation and cognitive overload while sustaining long-term interpretive agency in AI-mediated environments. 

**Abstract (ZH)**: R-CAGE：护 ego 节律控制架构 

---
# Hand-Shadow Poser 

**Title (ZH)**: 手影Pose生成器 

**Authors**: Hao Xu, Yinqiao Wang, Niloy J. Mitra, Shuaicheng Liu, Pheng-Ann Heng, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07012)  

**Abstract**: Hand shadow art is a captivating art form, creatively using hand shadows to reproduce expressive shapes on the wall. In this work, we study an inverse problem: given a target shape, find the poses of left and right hands that together best produce a shadow resembling the input. This problem is nontrivial, since the design space of 3D hand poses is huge while being restrictive due to anatomical constraints. Also, we need to attend to the input's shape and crucial features, though the input is colorless and textureless. To meet these challenges, we design Hand-Shadow Poser, a three-stage pipeline, to decouple the anatomical constraints (by hand) and semantic constraints (by shadow shape): (i) a generative hand assignment module to explore diverse but reasonable left/right-hand shape hypotheses; (ii) a generalized hand-shadow alignment module to infer coarse hand poses with a similarity-driven strategy for selecting hypotheses; and (iii) a shadow-feature-aware refinement module to optimize the hand poses for physical plausibility and shadow feature preservation. Further, we design our pipeline to be trainable on generic public hand data, thus avoiding the need for any specialized training dataset. For method validation, we build a benchmark of 210 diverse shadow shapes of varying complexity and a comprehensive set of metrics, including a novel DINOv2-based evaluation metric. Through extensive comparisons with multiple baselines and user studies, our approach is demonstrated to effectively generate bimanual hand poses for a large variety of hand shapes for over 85% of the benchmark cases. 

**Abstract (ZH)**: 手影艺术是一种引人入胜的艺术形式，创造性地使用手影在墙上重现有表现力的形状。在本文中，我们研究了一个逆问题：给定一个目标形状，找出左、右手的最佳姿态，使它们共同产生与输入相似的阴影。这个问题非同 trivial，因为3D 手部姿态的设计空间巨大，但受到解剖学限制而受到限制。此外，我们需要关注输入的形状和关键特征，尽管输入是无色且无纹理的。为了解决这些挑战，我们设计了手影姿态生成器（Hand-Shadow Poser），这是一个三阶段流水线，以解耦解剖学约束（由手处理）和语义约束（由阴影形状处理）：（i）生成性手部分配模块，探索多样的但合理的左手/右手形状假设；（ii）通用手部-手影对齐模块，通过相似性驱动的选择策略推断粗略的手部姿态；（iii）基于手影特征的细化模块，优化手部姿态以保持物理合理性并保留手影特征。此外，我们设计了该流水线可以使用通用的公开手部数据进行训练，从而避免使用任何专门的训练数据集。为了方法验证，我们构建了一个包含210个不同复杂度的手影形状的数据集，并提供了一套全面的指标，包括一个新的基于DINOv2的评估指标。通过与多个基线方法和用户研究的广泛比较，我们的方法证明可以在超过85%的基准案例中有效地生成多样手形的双手姿势。 

---
# Towards the Three-Phase Dynamics of Generalization Power of a DNN 

**Title (ZH)**: 探索DNN泛化能力的三相动态 

**Authors**: Yuxuan He, Junpeng Zhang, Hongyuan Zhang, Quanshi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06993)  

**Abstract**: This paper proposes a new perspective for analyzing the generalization power of deep neural networks (DNNs), i.e., directly disentangling and analyzing the dynamics of generalizable and non-generalizable interaction encoded by a DNN through the training process. Specifically, this work builds upon the recent theoretical achievement in explainble AI, which proves that the detailed inference logic of DNNs can be can be strictly rewritten as a small number of AND-OR interaction patterns. Based on this, we propose an efficient method to quantify the generalization power of each interaction, and we discover a distinct three-phase dynamics of the generalization power of interactions during training. In particular, the early phase of training typically removes noisy and non-generalizable interactions and learns simple and generalizable ones. The second and the third phases tend to capture increasingly complex interactions that are harder to generalize. Experimental results verify that the learning of non-generalizable interactions is the the direct cause for the gap between the training and testing losses. 

**Abstract (ZH)**: 本文提出了一种分析深度神经网络（DNN）泛化能力的新视角，即直接拆分和分析DNN在训练过程中编码的可泛化和不可泛化的相互作用的动力学。具体而言，本文基于近期可解释人工智能领域的理论成就，该成就证明了DNN的详细推理逻辑可以严格地重写为少数几种AND-OR相互作用模式。基于此，我们提出了一种有效的方法来量化每个相互作用的泛化能力，并发现相互作用在训练过程中具有独特的三阶段动态。特别是，在训练的早期阶段通常会去除噪声和不可泛化的相互作用，并学习简单的和可泛化的相互作用。而在第二和第三阶段，倾向于捕捉越来越复杂且更难泛化的相互作用。实验结果验证了非可泛化相互作用的习得是训练损失与测试损失之间差距的直接原因。 

---
# AI-Powered Inverse Design of Ku-Band SIW Resonant Structures by Iterative Residual Correction Network 

**Title (ZH)**: 基于迭代残差校正网络的AI驱动Ku波段SIW谐振结构逆设计 

**Authors**: Mohammad Mashayekhi, Kamran Salehian  

**Link**: [PDF](https://arxiv.org/pdf/2505.06936)  

**Abstract**: Inverse electromagnetic modeling has emerged as a powerful approach for designing complex microwave structures with high accuracy and efficiency. In this study, we propose an Iterative Residual Correction Network (IRC-Net) for the inverse design of Ku-band Substrate Integrated Waveguide (SIW) components based on multimode resonators. We use a multimode resonance structure to demonstrate that it is possible to control the resonances of the structure. Therefore, these structures can be used for resonant components and smart filter design. The proposed deep learning architecture leverages residual neural networks to overcome the limitations of traditional inverse design techniques, such as the Feedforward Inverse Model (FIM), offering improved generalization and prediction accuracy. The approach begins with a FIM to generate initial design estimates, followed by an iterative correction strategy inspired by the Hybrid Inverse-Forward Residual Refinement Network (HiFR\textsuperscript{2}-Net), which we call IRC-Net. Experiments demonstrate that the IRC-Net achieves substantial improvements in prediction accuracy compared to traditional single-stage networks, validated through statistical metrics, full-wave electromagnetic simulations, and measurements. To validate the proposed framework, we first design and fabricate a three-resonance SIW structure. Next, we apply the trained IRC-Net model to predict the geometry of a four-resonance structure based on its desired frequency response. Both designs are fabricated and tested, showing strong agreement between the simulated, predicted, and measured results, confirming the effectiveness and practicality of the proposed method. 

**Abstract (ZH)**: 基于多模谐振器的Ku波段集成波导组件的逆向设计的迭代残差校正网络 

---
# MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning 

**Title (ZH)**: MMiC: 缓解集群联邦学习中的模态不完整性 

**Authors**: Lishan Yang, Wei Zhang, Quan Z. Sheng, Weitong Chen, Lina Yao, Weitong Chen, Ali Shakeri  

**Link**: [PDF](https://arxiv.org/pdf/2505.06911)  

**Abstract**: In the era of big data, data mining has become indispensable for uncovering hidden patterns and insights from vast and complex datasets. The integration of multimodal data sources further enhances its potential. Multimodal Federated Learning (MFL) is a distributed approach that enhances the efficiency and quality of multimodal learning, ensuring collaborative work and privacy protection. However, missing modalities pose a significant challenge in MFL, often due to data quality issues or privacy policies across the clients. In this work, we present MMiC, a framework for Mitigating Modality incompleteness in MFL within the Clusters. MMiC replaces partial parameters within client models inside clusters to mitigate the impact of missing modalities. Furthermore, it leverages the Banzhaf Power Index to optimize client selection under these conditions. Finally, MMiC employs an innovative approach to dynamically control global aggregation by utilizing Markovitz Portfolio Optimization. Extensive experiments demonstrate that MMiC consistently outperforms existing federated learning architectures in both global and personalized performance on multimodal datasets with missing modalities, confirming the effectiveness of our proposed solution. 

**Abstract (ZH)**: 在大数据时代，数据挖掘已成为从庞大而复杂的多模态数据集中发现隐藏模式和洞察力不可或缺的工具。多模态联邦学习（MFL）是一种分布式方法，它提高了多模态学习的效率和质量，同时确保协作工作和隐私保护。然而，缺失的模态在MFL中构成了重大挑战，通常是由于客户端的数据质量问题或隐私政策所致。在本文中，我们提出了MMiC框架，用于在簇内缓解多模态数据缺失性问题。MMiC通过在客户端模型中替换部分参数来减轻缺失模态的影响。此外，它利用Banzhaf权力指数优化在这些条件下选择客户端。最后，MMiC采用了利用马克维兹资产组合优化的创新方法，动态控制全局聚合。 extensive实验证明，MMiC在存在缺失模态的多模态数据集上的一般性和个性化性能上均优于现有的联邦学习架构，证实了我们所提出的解决方案的有效性。 

---
# Mice to Machines: Neural Representations from Visual Cortex for Domain Generalization 

**Title (ZH)**: 从老鼠到机器：视觉皮层的神经表示在领域泛化中的应用 

**Authors**: Ahmed Qazi, Hamd Jalil, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06886)  

**Abstract**: The mouse is one of the most studied animal models in the field of systems neuroscience. Understanding the generalized patterns and decoding the neural representations that are evoked by the diverse range of natural scene stimuli in the mouse visual cortex is one of the key quests in computational vision. In recent years, significant parallels have been drawn between the primate visual cortex and hierarchical deep neural networks. However, their generalized efficacy in understanding mouse vision has been limited. In this study, we investigate the functional alignment between the mouse visual cortex and deep learning models for object classification tasks. We first introduce a generalized representational learning strategy that uncovers a striking resemblance between the functional mapping of the mouse visual cortex and high-performing deep learning models on both top-down (population-level) and bottom-up (single cell-level) scenarios. Next, this representational similarity across the two systems is further enhanced by the addition of Neural Response Normalization (NeuRN) layer, inspired by the activation profile of excitatory and inhibitory neurons in the visual cortex. To test the performance effect of NeuRN on real-world tasks, we integrate it into deep learning models and observe significant improvements in their robustness against data shifts in domain generalization tasks. Our work proposes a novel framework for comparing the functional architecture of the mouse visual cortex with deep learning models. Our findings carry broad implications for the development of advanced AI models that draw inspiration from the mouse visual cortex, suggesting that these models serve as valuable tools for studying the neural representations of the mouse visual cortex and, as a result, enhancing their performance on real-world tasks. 

**Abstract (ZH)**: 小鼠是系统神经科学领域中研究最多的小型动物模型之一。理解小鼠视觉皮层在面对各种自然 scenes 刺激时表现出的通用模式及其神经表征解码是计算视觉中的关键任务之一。近年来，猕猴视觉皮层与层次化深度神经网络之间的类比关系越来越多，然而这在解释小鼠视觉方面的作用有限。本研究探讨了小鼠视觉皮层与深度学习模型在物体分类任务中的功能对齐。我们首先介绍了一种通用的表征学习策略，揭示了小鼠视觉皮层的功能映射与高性能深度学习模型之间的显著相似性，这一发现适用于自上而下（群体层面）和自下而上（单细胞层面）两种情况。然后，通过引入神经响应归一化（NeuRN）层，进一步加强了两种系统之间的表征相似性，该层受视觉皮层兴奋性和抑制性神经元激活特征的启发。为了测试NeuRN在实际任务中的性能影响，我们将其整合到深度学习模型中，并观察到在跨域泛化任务中对其鲁棒性的显著提升。本研究提出了一个新型框架，用于比较小鼠视觉皮层的功能架构与深度学习模型的对比。我们的发现对基于小鼠视觉皮层构建高级人工智能模型的发展具有广泛的影响，表明这些模型是研究小鼠视觉皮层神经表征的重要工具，并且能够提升其在实际任务中的性能。 

---
# NeuRN: Neuro-inspired Domain Generalization for Image Classification 

**Title (ZH)**: NeuRN: 基于神经启发的领域泛化图像分类 

**Authors**: Hamd Jalil, Ahmed Qazi, Asim Iqbal  

**Link**: [PDF](https://arxiv.org/pdf/2505.06881)  

**Abstract**: Domain generalization in image classification is a crucial challenge, with models often failing to generalize well across unseen datasets. We address this issue by introducing a neuro-inspired Neural Response Normalization (NeuRN) layer which draws inspiration from neurons in the mammalian visual cortex, which aims to enhance the performance of deep learning architectures on unseen target domains by training deep learning models on a source domain. The performance of these models is considered as a baseline and then compared against models integrated with NeuRN on image classification tasks. We perform experiments across a range of deep learning architectures, including ones derived from Neural Architecture Search and Vision Transformer. Additionally, in order to shortlist models for our experiment from amongst the vast range of deep neural networks available which have shown promising results, we also propose a novel method that uses the Needleman-Wunsch algorithm to compute similarity between deep learning architectures. Our results demonstrate the effectiveness of NeuRN by showing improvement against baseline in cross-domain image classification tasks. Our framework attempts to establish a foundation for future neuro-inspired deep learning models. 

**Abstract (ZH)**: 基于神经元启发的神经响应归一化层在图像分类中的泛化能力研究 

---
# Enhancing Time Series Forecasting via a Parallel Hybridization of ARIMA and Polynomial Classifiers 

**Title (ZH)**: 基于ARIMA与多项式分类器并行混合的时序预测增强方法 

**Authors**: Thanh Son Nguyen, Van Thanh Nguyen, Dang Minh Duc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06874)  

**Abstract**: Time series forecasting has attracted significant attention, leading to the de-velopment of a wide range of approaches, from traditional statistical meth-ods to advanced deep learning models. Among them, the Auto-Regressive Integrated Moving Average (ARIMA) model remains a widely adopted linear technique due to its effectiveness in modeling temporal dependencies in economic, industrial, and social data. On the other hand, polynomial classifi-ers offer a robust framework for capturing non-linear relationships and have demonstrated competitive performance in domains such as stock price pre-diction. In this study, we propose a hybrid forecasting approach that inte-grates the ARIMA model with a polynomial classifier to leverage the com-plementary strengths of both models. The hybrid method is evaluated on multiple real-world time series datasets spanning diverse domains. Perfor-mance is assessed based on forecasting accuracy and computational effi-ciency. Experimental results reveal that the proposed hybrid model consist-ently outperforms the individual models in terms of prediction accuracy, al-beit with a modest increase in execution time. 

**Abstract (ZH)**: 时间序列预测吸引了广泛的关注，推动了从传统统计方法到先进深度学习模型的广泛应用。其中，自动回归积分移动平均（ARIMA）模型由于在建模经济、工业和社会数据的时间依赖性方面效果显著，仍然是广泛采用的线性技术之一。另一方面，多项式分类器提供了一种稳健的框架来捕获非线性关系，并在股票价格预测等领域展示了竞争力。在本研究中，我们提出了一种将ARIMA模型与多项式分类器相结合的混合预测方法，以发挥两种模型的互补优势。该混合方法在多个涵盖不同领域的实际时间序列数据集上进行了评估。性能评估基于预测准确性与计算效率。实验结果表明，所提出的混合模型在预测准确性方面始终优于单一模型，尽管执行时间略有增加。 

---
# DP-TRAE: A Dual-Phase Merging Transferable Reversible Adversarial Example for Image Privacy Protection 

**Title (ZH)**: DP-TRAE: 一种双阶段合并可移植可逆 adversarial 示例的图像隐私保护方法 

**Authors**: Xia Du, Jiajie Zhu, Jizhe Zhou, Chi-man Pun, Zheng Lin, Cong Wu, Zhe Chen, Jun Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06860)  

**Abstract**: In the field of digital security, Reversible Adversarial Examples (RAE) combine adversarial attacks with reversible data hiding techniques to effectively protect sensitive data and prevent unauthorized analysis by malicious Deep Neural Networks (DNNs). However, existing RAE techniques primarily focus on white-box attacks, lacking a comprehensive evaluation of their effectiveness in black-box scenarios. This limitation impedes their broader deployment in complex, dynamic environments. Further more, traditional black-box attacks are often characterized by poor transferability and high query costs, significantly limiting their practical applicability. To address these challenges, we propose the Dual-Phase Merging Transferable Reversible Attack method, which generates highly transferable initial adversarial perturbations in a white-box model and employs a memory augmented black-box strategy to effectively mislead target mod els. Experimental results demonstrate the superiority of our approach, achieving a 99.0% attack success rate and 100% recovery rate in black-box scenarios, highlighting its robustness in privacy protection. Moreover, we successfully implemented a black-box attack on a commercial model, further substantiating the potential of this approach for practical use. 

**Abstract (ZH)**: 在数字安全领域，可逆对抗样本（RAE）结合了对抗攻击与可逆数据隐藏技术，有效保护敏感数据并防止恶意深度神经网络（DNN）的未经授权分析。然而，现有RAE技术主要关注白盒攻击，缺乏对黑盒场景有效性的全面评估。这一局限性阻碍了其在复杂动态环境中的广泛应用。此外，传统的黑盒攻击通常表现出较差的迁移性和较高的查询成本，显著限制了其实用性。为应对这些挑战，我们提出了双阶段合并可迁移可逆攻击方法，该方法在白盒模型中生成高度可迁移的初始对抗扰动，并采用记忆增强的黑盒策略有效地误导目标模型。实验结果证明了该方法的优势，在黑盒场景中实现99.0%的攻击成功率和100%的数据恢复率，突显了其在隐私保护方面的稳健性。此外，我们成功对一个商用模型进行了黑盒攻击，进一步证明了该方法在实际应用中的潜力。 

---
# The power of fine-grained experts: Granularity boosts expressivity in Mixture of Experts 

**Title (ZH)**: 细粒度专家的力量：粒度增强混合专家模型的表达能力 

**Authors**: Enric Boix-Adsera, Philippe Rigollet  

**Link**: [PDF](https://arxiv.org/pdf/2505.06839)  

**Abstract**: Mixture-of-Experts (MoE) layers are increasingly central to frontier model architectures. By selectively activating parameters, they reduce computational cost while scaling total parameter count. This paper investigates the impact of the number of active experts, termed granularity, comparing architectures with many (e.g., 8 per layer in DeepSeek) to those with fewer (e.g., 1 per layer in Llama-4 models). We prove an exponential separation in network expressivity based on this design parameter, suggesting that models benefit from higher granularity. Experimental results corroborate our theoretical findings and illustrate this separation. 

**Abstract (ZH)**: 混合专家层（MoE）在前沿模型架构中日益占据核心地位。通过选择性激活参数，它们在增加总参数量的同时降低计算成本。本文探讨了激活专家数量（称为精细度）对网络表达能力的影响，对比了每层具有多个专家（如DeepSeek中的每层8个专家）的架构与每层具有较少专家（如Llama-4模型中的每层1个专家）的架构。我们证明了基于此设计参数的网络表达能力存在指数级差异，表明模型从中受益于更高精细度。实验结果证实了我们的理论发现，并展示了这种差异。 

---
# Sandcastles in the Storm: Revisiting the (Im)possibility of Strong Watermarking 

**Title (ZH)**: 风暴中的沙堡：重访强水印的（不）可能性 

**Authors**: Fabrice Y Harel-Canada, Boran Erol, Connor Choi, Jason Liu, Gary Jiarui Song, Nanyun Peng, Amit Sahai  

**Link**: [PDF](https://arxiv.org/pdf/2505.06827)  

**Abstract**: Watermarking AI-generated text is critical for combating misuse. Yet recent theoretical work argues that any watermark can be erased via random walk attacks that perturb text while preserving quality. However, such attacks rely on two key assumptions: (1) rapid mixing (watermarks dissolve quickly under perturbations) and (2) reliable quality preservation (automated quality oracles perfectly guide edits). Through large-scale experiments and human-validated assessments, we find mixing is slow: 100% of perturbed texts retain traces of their origin after hundreds of edits, defying rapid mixing. Oracles falter, as state-of-the-art quality detectors misjudge edits (77% accuracy), compounding errors during attacks. Ultimately, attacks underperform: automated walks remove watermarks just 26% of the time -- dropping to 10% under human quality review. These findings challenge the inevitability of watermark removal. Instead, practical barriers -- slow mixing and imperfect quality control -- reveal watermarking to be far more robust than theoretical models suggest. The gap between idealized attacks and real-world feasibility underscores the need for stronger watermarking methods and more realistic attack models. 

**Abstract (ZH)**: 人工智能生成文本的水印对于防止滥用至关重要。然而，近期的理论工作认为任何水印都可以通过随机游走攻击被消除，这些攻击在扰动文本的同时保持了质量。然而，这类攻击依赖于两个关键假设：（1）快速混合（水印在扰动下迅速消散）和（2）可靠的质量保持（自动化质量仲裁器完美地指导编辑）。通过大规模实验和人类验证的评估，我们发现混合缓慢：经过数百次编辑后，100%的扰动文本仍保留其来源的痕迹，反驳了快速混合的假设。仲裁器表现不佳，最新质量检测器对编辑的判断有误（准确率为77%），在攻击过程中累积了错误。最终，攻击表现不佳：自动化游走只能去除水印26%的时间，在人工质量审阅下这一比例降至10%。这些发现挑战了水印去除的必然性。相反，实际障碍——缓慢的混合和不完美的质量控制——表明水印技术比理论模型所设想的更为 robust。理想化的攻击与现实可行性之间的差距凸显了更强大水印方法和更现实攻击模型的必要性。 

---
# Quantum Observers: A NISQ Hardware Demonstration of Chaotic State Prediction Using Quantum Echo-state Networks 

**Title (ZH)**: 量子观测者：使用量子回声状态网络预测混沌态的NISQ硬件演示 

**Authors**: Erik L. Connerty, Ethan N. Evans, Gerasimos Angelatos, Vignesh Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06799)  

**Abstract**: Recent advances in artificial intelligence have highlighted the remarkable capabilities of neural network (NN)-powered systems on classical computers. However, these systems face significant computational challenges that limit scalability and efficiency. Quantum computers hold the potential to overcome these limitations and increase processing power beyond classical systems. Despite this, integrating quantum computing with NNs remains largely unrealized due to challenges posed by noise, decoherence, and high error rates in current quantum hardware. Here, we propose a novel quantum echo-state network (QESN) design and implementation algorithm that can operate within the presence of noise on current IBM hardware. We apply classical control-theoretic response analysis to characterize the QESN, emphasizing its rich nonlinear dynamics and memory, as well as its ability to be fine-tuned with sparsity and re-uploading blocks. We validate our approach through a comprehensive demonstration of QESNs functioning as quantum observers, applied in both high-fidelity simulations and hardware experiments utilizing data from a prototypical chaotic Lorenz system. Our results show that the QESN can predict long time-series with persistent memory, running over 100 times longer than the median T}1 and T2 of the IBM Marrakesh QPU, achieving state-of-the-art time-series performance on superconducting hardware. 

**Abstract (ZH)**: 近年来，人工智能力量的神经网络（NN）驱动系统在经典计算机上展现了令人瞩目的能力。然而，这些系统面临显著的计算挑战，限制了其可扩展性和效率。量子计算机有可能克服这些限制，并在处理能力上超越经典系统。尽管如此，将量子计算与神经网络相结合仍因当前量子硬件中的噪声、退相干和高错误率等挑战而难以实现。在此，我们提出了一种新型量子回声状态网络（QESN）的设计和实现算法，该算法能够在当前IBM硬件中的噪声环境中运行。我们通过经典的控制理论响应分析来表征QESN，强调其丰富的非线性动力学和记忆特性，以及其通过稀疏性与重加载块进行微调的能力。我们通过全面演示QESN作为量子观测器的功能来验证我们的方法，这些演示在高保真模拟和使用IBM Marrakesh QPU数据进行的硬件实验中进行。结果表明，QESN可以在保留长期记忆的情况下预测长时序列，并在超导硬件上实现了最先进的时序性能，运行时间超过IBM Marrakesh QPU的中位T1和T2时间的100多倍。 

---
# Decoding Futures Price Dynamics: A Regularized Sparse Autoencoder for Interpretable Multi-Horizon Forecasting and Factor Discovery 

**Title (ZH)**: 解码期货价格动态：一种正则化稀疏自编码器用于可解释的多horizon预测和因子发现 

**Authors**: Abhijit Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.06795)  

**Abstract**: Commodity price volatility creates economic challenges, necessitating accurate multi-horizon forecasting. Predicting prices for commodities like copper and crude oil is complicated by diverse interacting factors (macroeconomic, supply/demand, geopolitical, etc.). Current models often lack transparency, limiting strategic use. This paper presents a Regularized Sparse Autoencoder (RSAE), a deep learning framework for simultaneous multi-horizon commodity price prediction and discovery of interpretable latent market drivers. The RSAE forecasts prices at multiple horizons (e.g., 1-day, 1-week, 1-month) using multivariate time series. Crucially, L1 regularization ($\|\mathbf{z}\|_1$) on its latent vector $\mathbf{z}$ enforces sparsity, promoting parsimonious explanations of market dynamics through learned factors representing underlying drivers (e.g., demand, supply shocks). Drawing from energy-based models and sparse coding, the RSAE optimizes predictive accuracy while learning sparse representations. Evaluated on historical Copper and Crude Oil data with numerous indicators, our findings indicate the RSAE offers competitive multi-horizon forecasting accuracy and data-driven insights into price dynamics via its interpretable latent space, a key advantage over traditional black-box approaches. 

**Abstract (ZH)**: 商品价格波动创造经济挑战，亟需准确的多时域预测。本论文提出一种正则化稀疏自编码器（RSAE），这是一种用于同时进行多时域商品价格预测及可解释潜在市场驱动因素发现的深度学习框架。RSAE 利用多变量时间序列预测多个时域的价格（例如，1 天、1 周、1 个月）。关键地，其潜在向量 $\mathbf{z}$ 上的 L1 正则化（$\|\mathbf{z}\|_1$）促进稀疏性，通过学习代表潜在驱动因素（如需求、供给冲击）的因素来简洁地解释市场动态。借鉴能量模型和稀疏编码，RSAE 在提高预测准确性的同时学习稀疏表示。在包含多种指标的历史铜和原油数据上进行评估，我们的研究结果表明，RSAE 提供了具有竞争力的多时域预测准确度，并通过其可解释的潜在空间提供了数据驱动的价格动态见解，这一优势使其有别于传统的黑盒方法。 

---
# Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in Autonomous Driving 

**Title (ZH)**: 平衡进展与安全：自主驾驶中强化学习的新颖风险感知目标 

**Authors**: Ahmed Abouelazm, Jonas Michel, Helen Gremmelmaier, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06737)  

**Abstract**: Reinforcement Learning (RL) is a promising approach for achieving autonomous driving due to robust decision-making capabilities. RL learns a driving policy through trial and error in traffic scenarios, guided by a reward function that combines the driving objectives. The design of such reward function has received insufficient attention, yielding ill-defined rewards with various pitfalls. Safety, in particular, has long been regarded only as a penalty for collisions. This leaves the risks associated with actions leading up to a collision unaddressed, limiting the applicability of RL in real-world scenarios. To address these shortcomings, our work focuses on enhancing the reward formulation by defining a set of driving objectives and structuring them hierarchically. Furthermore, we discuss the formulation of these objectives in a normalized manner to transparently determine their contribution to the overall reward. Additionally, we introduce a novel risk-aware objective for various driving interactions based on a two-dimensional ellipsoid function and an extension of Responsibility-Sensitive Safety (RSS) concepts. We evaluate the efficacy of our proposed reward in unsignalized intersection scenarios with varying traffic densities. The approach decreases collision rates by 21\% on average compared to baseline rewards and consistently surpasses them in route progress and cumulative reward, demonstrating its capability to promote safer driving behaviors while maintaining high-performance levels. 

**Abstract (ZH)**: 基于强化学习的自动驾驶奖励函数设计与风险意识目标研究：减少无信号交叉口碰撞促进安全驾驶 

---
# Deeply Explainable Artificial Neural Network 

**Title (ZH)**: 深度可解释的人工神经网络 

**Authors**: David Zucker  

**Link**: [PDF](https://arxiv.org/pdf/2505.06731)  

**Abstract**: While deep learning models have demonstrated remarkable success in numerous domains, their black-box nature remains a significant limitation, especially in critical fields such as medical image analysis and inference. Existing explainability methods, such as SHAP, LIME, and Grad-CAM, are typically applied post hoc, adding computational overhead and sometimes producing inconsistent or ambiguous results. In this paper, we present the Deeply Explainable Artificial Neural Network (DxANN), a novel deep learning architecture that embeds explainability ante hoc, directly into the training process. Unlike conventional models that require external interpretation methods, DxANN is designed to produce per-sample, per-feature explanations as part of the forward pass. Built on a flow-based framework, it enables both accurate predictions and transparent decision-making, and is particularly well-suited for image-based tasks. While our focus is on medical imaging, the DxANN architecture is readily adaptable to other data modalities, including tabular and sequential data. DxANN marks a step forward toward intrinsically interpretable deep learning, offering a practical solution for applications where trust and accountability are essential. 

**Abstract (ZH)**: 深度可解释人工神经网络（DxANN）：基于先验嵌入的可解释深度学习架构 

---
# FNBench: Benchmarking Robust Federated Learning against Noisy Labels 

**Title (ZH)**: FNBench: 在嘈杂标签环境下评估联邦学习的鲁棒性 

**Authors**: Xuefeng Jiang, Jia Li, Nannan Wu, Zhiyuan Wu, Xujing Li, Sheng Sun, Gang Xu, Yuwei Wang, Qi Li, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06684)  

**Abstract**: Robustness to label noise within data is a significant challenge in federated learning (FL). From the data-centric perspective, the data quality of distributed datasets can not be guaranteed since annotations of different clients contain complicated label noise of varying degrees, which causes the performance degradation. There have been some early attempts to tackle noisy labels in FL. However, there exists a lack of benchmark studies on comprehensively evaluating their practical performance under unified settings. To this end, we propose the first benchmark study FNBench to provide an experimental investigation which considers three diverse label noise patterns covering synthetic label noise, imperfect human-annotation errors and systematic errors. Our evaluation incorporates eighteen state-of-the-art methods over five image recognition datasets and one text classification dataset. Meanwhile, we provide observations to understand why noisy labels impair FL, and additionally exploit a representation-aware regularization method to enhance the robustness of existing methods against noisy labels based on our observations. Finally, we discuss the limitations of this work and propose three-fold future directions. To facilitate related communities, our source code is open-sourced at this https URL. 

**Abstract (ZH)**: 联邦学习中数据标签噪声鲁棒性研究：FNBench基准研究 

---
# Dyn-D$^2$P: Dynamic Differentially Private Decentralized Learning with Provable Utility Guarantee 

**Title (ZH)**: Dyn-D$^2$P: 动态差分隐私去中心化学习及其可证明的效用保证 

**Authors**: Zehan Zhu, Yan Huang, Xin Wang, Shouling Ji, Jinming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06651)  

**Abstract**: Most existing decentralized learning methods with differential privacy (DP) guarantee rely on constant gradient clipping bounds and fixed-level DP Gaussian noises for each node throughout the training process, leading to a significant accuracy degradation compared to non-private counterparts. In this paper, we propose a new Dynamic Differentially Private Decentralized learning approach (termed Dyn-D$^2$P) tailored for general time-varying directed networks. Leveraging the Gaussian DP (GDP) framework for privacy accounting, Dyn-D$^2$P dynamically adjusts gradient clipping bounds and noise levels based on gradient convergence. This proposed dynamic noise strategy enables us to enhance model accuracy while preserving the total privacy budget. Extensive experiments on benchmark datasets demonstrate the superiority of Dyn-D$^2$P over its counterparts employing fixed-level noises, especially under strong privacy guarantees. Furthermore, we provide a provable utility bound for Dyn-D$^2$P that establishes an explicit dependency on network-related parameters, with a scaling factor of $1/\sqrt{n}$ in terms of the number of nodes $n$ up to a bias error term induced by gradient clipping. To our knowledge, this is the first model utility analysis for differentially private decentralized non-convex optimization with dynamic gradient clipping bounds and noise levels. 

**Abstract (ZH)**: 一种新的动态差分隐私去中心化学习方法（Dyn-D$^2$P）：针对一般时间varying有向网络的设计 

---
# AI-Powered Anomaly Detection with Blockchain for Real-Time Security and Reliability in Autonomous Vehicles 

**Title (ZH)**: 基于区块链的AI驱动异常检测技术及其在自主车辆实时安全与可靠性中的应用 

**Authors**: Rathin Chandra Shit, Sharmila Subudhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06632)  

**Abstract**: Autonomous Vehicles (AV) proliferation brings important and pressing security and reliability issues that must be dealt with to guarantee public safety and help their widespread adoption. The contribution of the proposed research is towards achieving more secure, reliable, and trustworthy autonomous transportation system by providing more capabilities for anomaly detection, data provenance, and real-time response in safety critical AV deployments. In this research, we develop a new framework that combines the power of Artificial Intelligence (AI) for real-time anomaly detection with blockchain technology to detect and prevent any malicious activity including sensor failures in AVs. Through Long Short-Term Memory (LSTM) networks, our approach continually monitors associated multi-sensor data streams to detect anomalous patterns that may represent cyberattacks as well as hardware malfunctions. Further, this framework employs a decentralized platform for securely storing sensor data and anomaly alerts in a blockchain ledger for data incorruptibility and authenticity, while offering transparent forensic features. Moreover, immediate automated response mechanisms are deployed using smart contracts when anomalies are found. This makes the AV system more resilient to attacks from both cyberspace and hardware component failure. Besides, we identify potential challenges of scalability in handling high frequency sensor data, computational constraint in resource constrained environment, and of distributed data storage in terms of privacy. 

**Abstract (ZH)**: 自主驾驶车辆的普及带来了重要的紧迫的安全性和可靠性问题，必须解决这些问题以确保公众安全并促进其广泛应用。本研究的贡献在于通过结合人工智能（AI）实现即时异常检测与区块链技术，提供更多的异常检测能力、数据溯源能力和在关键安全场景下自主驾驶车辆的即时响应能力，以实现更安全、可靠和可信赖的自主交通系统。在本研究中，我们开发了一个新的框架，该框架结合了基于长短期记忆网络（LSTM）的实时异常检测能力与区块链技术，用于检测和防止包括传感器故障在内的任何恶意活动。进一步地，该框架采用去中心化的平台在区块链账本中安全地存储传感器数据和异常警报，以确保数据的不可篡改性和真实性，并提供透明的取证功能。此外，当检测到异常时，将部署智能合约以实现即时自动化响应机制，使自主驾驶车辆系统更具抗攻击性，可以从网络空间和硬件组件故障中恢复。此外，我们还识别了在高频率传感器数据处理中可能面临的扩展性挑战、受限资源环境下的计算约束挑战，以及分布式数据存储中的隐私问题。 

---
# Dynamic Domain Information Modulation Algorithm for Multi-domain Sentiment Analysis 

**Title (ZH)**: 多域情感分析的动态领域信息调制算法 

**Authors**: Chunyi Yue, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06630)  

**Abstract**: Multi-domain sentiment classification aims to mitigate poor performance models due to the scarcity of labeled data in a single domain, by utilizing data labeled from various domains. A series of models that jointly train domain classifiers and sentiment classifiers have demonstrated their advantages, because domain classification helps generate necessary information for sentiment classification. Intuitively, the importance of sentiment classification tasks is the same in all domains for multi-domain sentiment classification; but domain classification tasks are different because the impact of domain information on sentiment classification varies across different fields; this can be controlled through adjustable weights or hyper parameters. However, as the number of domains increases, existing hyperparameter optimization algorithms may face the following challenges: (1) tremendous demand for computing resources, (2) convergence problems, and (3) high algorithm complexity. To efficiently generate the domain information required for sentiment classification in each domain, we propose a dynamic information modulation algorithm. Specifically, the model training process is divided into two stages. In the first stage, a shared hyperparameter, which would control the proportion of domain classification tasks across all fields, is determined. In the second stage, we introduce a novel domain-aware modulation algorithm to adjust the domain information contained in the input text, which is then calculated based on a gradient-based and loss-based method. In summary, experimental results on a public sentiment analysis dataset containing 16 domains prove the superiority of the proposed method. 

**Abstract (ZH)**: 多领域情感分类旨在通过利用来自多个领域的标注数据来缓解单一领域标注数据稀少导致的模型性能不佳问题。一系列同时训练领域分类器和情感分类器的模型展示了其优势，因为领域分类有助于为情感分类生成必要的信息。直觉上，多领域情感分类中所有领域的感情分类任务的重要性相同；但领域分类任务不同，因为领域信息对情感分类的影响因领域而异；这可以通过可调节的权重或超参数来控制。然而，随着领域数量的增加，现有的超参数优化算法可能会面临以下挑战：（1）对计算资源的巨大需求，（2）收敛问题，（3）高算法复杂度。为高效生成用于每个领域的情感分类所需要的领域信息，我们提出了一种动态信息调制算法。具体而言，模型训练过程分为两个阶段。在第一阶段，确定一个共享的超参数，该超参数将控制各个领域领域分类任务的比例。在第二阶段，我们引入了一种新颖的领域意识调制算法来调整输入文本中的领域信息，并基于梯度和损失方法进行计算。总之，公共情感分析数据集上的16个领域实验结果证明了所提方法的优越性。 

---
# CaMDN: Enhancing Cache Efficiency for Multi-tenant DNNs on Integrated NPUs 

**Title (ZH)**: CaMDN: 提升集成NPUs上多租户DNN缓存效率 

**Authors**: Tianhao Cai, Liang Wang, Limin Xiao, Meng Han, Zeyu Wang, Lin Sun, Xiaojian Liao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06625)  

**Abstract**: With the rapid development of DNN applications, multi-tenant execution, where multiple DNNs are co-located on a single SoC, is becoming a prevailing trend. Although many methods are proposed in prior works to improve multi-tenant performance, the impact of shared cache is not well studied. This paper proposes CaMDN, an architecture-scheduling co-design to enhance cache efficiency for multi-tenant DNNs on integrated NPUs. Specifically, a lightweight architecture is proposed to support model-exclusive, NPU-controlled regions inside shared cache to eliminate unexpected cache contention. Moreover, a cache scheduling method is proposed to improve shared cache utilization. In particular, it includes a cache-aware mapping method for adaptability to the varying available cache capacity and a dynamic allocation algorithm to adjust the usage among co-located DNNs at runtime. Compared to prior works, CaMDN reduces the memory access by 33.4% on average and achieves a model speedup of up to 2.56$\times$ (1.88$\times$ on average). 

**Abstract (ZH)**: 基于共享缓存的多租户DNN架构-调度协同设计CaMDN 

---
# Integrating Explainable AI in Medical Devices: Technical, Clinical and Regulatory Insights and Recommendations 

**Title (ZH)**: 将可解释的AI集成到医疗设备中：技术、临床和监管洞察与建议 

**Authors**: Dima Alattal, Asal Khoshravan Azar, Puja Myles, Richard Branson, Hatim Abdulhussein, Allan Tucker  

**Link**: [PDF](https://arxiv.org/pdf/2505.06620)  

**Abstract**: There is a growing demand for the use of Artificial Intelligence (AI) and Machine Learning (ML) in healthcare, particularly as clinical decision support systems to assist medical professionals. However, the complexity of many of these models, often referred to as black box models, raises concerns about their safe integration into clinical settings as it is difficult to understand how they arrived at their predictions. This paper discusses insights and recommendations derived from an expert working group convened by the UK Medicine and Healthcare products Regulatory Agency (MHRA). The group consisted of healthcare professionals, regulators, and data scientists, with a primary focus on evaluating the outputs from different AI algorithms in clinical decision-making contexts. Additionally, the group evaluated findings from a pilot study investigating clinicians' behaviour and interaction with AI methods during clinical diagnosis. Incorporating AI methods is crucial for ensuring the safety and trustworthiness of medical AI devices in clinical settings. Adequate training for stakeholders is essential to address potential issues, and further insights and recommendations for safely adopting AI systems in healthcare settings are provided. 

**Abstract (ZH)**: 人工智能和机器学习在医疗保健中的应用：黑箱模型的复杂性及其在临床决策支持中的安全整合探究——英国 Medicine and Healthcare products Regulatory Agency (MHRA) 专家工作组的见解与建议 

---
# Burger: Robust Graph Denoising-augmentation Fusion and Multi-semantic Modeling in Social Recommendation 

**Title (ZH)**: Burger：社交推荐中的鲁棒图去噪、增强融合与多语义建模 

**Authors**: Yuqin Lan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06612)  

**Abstract**: In the era of rapid development of social media, social recommendation systems as hybrid recommendation systems have been widely applied. Existing methods capture interest similarity between users to filter out interest-irrelevant relations in social networks that inevitably decrease recommendation accuracy, however, limited research has a focus on the mutual influence of semantic information between the social network and the user-item interaction network for further improving social recommendation. To address these issues, we introduce a social \underline{r}ecommendation model with ro\underline{bu}st g\underline{r}aph denoisin\underline{g}-augmentation fusion and multi-s\underline{e}mantic Modeling(Burger). Specifically, we firstly propose to construct a social tensor in order to smooth the training process of the model. Then, a graph convolutional network and a tensor convolutional network are employed to capture user's item preference and social preference, respectively. Considering the different semantic information in the user-item interaction network and the social network, a bi-semantic coordination loss is proposed to model the mutual influence of semantic information. To alleviate the interference of interest-irrelevant relations on multi-semantic modeling, we further use Bayesian posterior probability to mine potential social relations to replace social noise. Finally, the sliding window mechanism is utilized to update the social tensor as the input for the next iteration. Extensive experiments on three real datasets show Burger has a superior performance compared with the state-of-the-art models. 

**Abstract (ZH)**: 社交媒体快速发展时代的社交推荐系统robust图去噪增强融合多语义建模(Burger)研究 

---
# Feature Representation Transferring to Lightweight Models via Perception Coherence 

**Title (ZH)**: 基于感知一致性的小型模型特征表示转移 

**Authors**: Hai-Vy Nguyen, Fabrice Gamboa, Sixin Zhang, Reda Chhaibi, Serge Gratton, Thierry Giaccone  

**Link**: [PDF](https://arxiv.org/pdf/2505.06595)  

**Abstract**: In this paper, we propose a method for transferring feature representation to lightweight student models from larger teacher models. We mathematically define a new notion called \textit{perception coherence}. Based on this notion, we propose a loss function, which takes into account the dissimilarities between data points in feature space through their ranking. At a high level, by minimizing this loss function, the student model learns to mimic how the teacher model \textit{perceives} inputs. More precisely, our method is motivated by the fact that the representational capacity of the student model is weaker than the teacher model. Hence, we aim to develop a new method allowing for a better relaxation. This means that, the student model does not need to preserve the absolute geometry of the teacher one, while preserving global coherence through dissimilarity ranking. Our theoretical insights provide a probabilistic perspective on the process of feature representation transfer. Our experiments results show that our method outperforms or achieves on-par performance compared to strong baseline methods for representation transferring. 

**Abstract (ZH)**: 本文提出了一种将大型教师模型的特征表示转移到轻量级学生模型的方法。我们从数学上定义了一个新的概念叫作“感知一致性”。基于这一概念，我们提出了一种损失函数，该损失函数通过数据点在特征空间中的排名来考虑其差异性。总体而言，通过最小化该损失函数，学生模型学会模仿教师模型如何“感知”输入。更精确地说，我们的方法动机在于学生的表示能力弱于教师模型，因此我们旨在开发一种新的方法使得学生模型能够更有效地放松约束。这意味着，学生模型不需要完全保留教师模型的绝对几何结构，同时通过差异性排名保持全局一致性。我们的理论洞察提供了特征表示转移过程的概率视角。我们的实验结果表明，与强基准方法相比，我们的方法在特征表示转移上表现出更优或相当的性能。 

---
# Optimal Transport for Machine Learners 

**Title (ZH)**: 机器学习中的最优传输 

**Authors**: Gabriel Peyré  

**Link**: [PDF](https://arxiv.org/pdf/2505.06589)  

**Abstract**: Optimal Transport is a foundational mathematical theory that connects optimization, partial differential equations, and probability. It offers a powerful framework for comparing probability distributions and has recently become an important tool in machine learning, especially for designing and evaluating generative models. These course notes cover the fundamental mathematical aspects of OT, including the Monge and Kantorovich formulations, Brenier's theorem, the dual and dynamic formulations, the Bures metric on Gaussian distributions, and gradient flows. It also introduces numerical methods such as linear programming, semi-discrete solvers, and entropic regularization. Applications in machine learning include topics like training neural networks via gradient flows, token dynamics in transformers, and the structure of GANs and diffusion models. These notes focus primarily on mathematical content rather than deep learning techniques. 

**Abstract (ZH)**: 最优运输是连接优化、偏微分方程和概率的基础数学理论。它提供了一种强大的框架来比较概率分布，并 recently 成为机器学习中一个重要的工具，特别是在设计和评估生成模型方面。这些课程笔记涵盖了最优运输的基本数学方面，包括蒙格和坎托罗维奇公式、布伦耐尔定理、对偶和动力学公式、高斯分布上的布勒斯度量以及梯度流。它还介绍了数值方法，如线性规划、半离散求解器和熵正则化。在机器学习中的应用包括通过梯度流训练神经网络、变压器中的标记动力学以及生成对抗网络和扩散模型的结构等内容。这些笔记主要关注数学内容而非深度学习技术。 

---
# Two-Stage Random Alternation Framework for Zero-Shot Pansharpening 

**Title (ZH)**: 两阶段随机交替框架用于零样本 pansharpening 

**Authors**: Haorui Chen, Zeyu Ren, Jiaxuan Ren, Ran Ran, Jinliang Shao, Jie Huang, Liangjian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.06576)  

**Abstract**: In recent years, pansharpening has seen rapid advancements with deep learning methods, which have demonstrated impressive fusion quality. However, the challenge of acquiring real high-resolution images limits the practical applicability of these methods. To address this, we propose a two-stage random alternating framework (TRA-PAN) that effectively integrates strong supervision constraints from reduced-resolution images with the physical characteristics of full-resolution images. The first stage introduces a pre-training procedure, which includes Degradation-Aware Modeling (DAM) to capture spatial-spectral degradation mappings, alongside a warm-up procedure designed to reduce training time and mitigate the negative effects of reduced-resolution data. In the second stage, Random Alternation Optimization (RAO) is employed, where random alternating training leverages the strengths of both reduced- and full-resolution images, further optimizing the fusion model. By primarily relying on full-resolution images, our method enables zero-shot training with just a single image pair, obviating the need for large datasets. Experimental results demonstrate that TRA-PAN outperforms state-of-the-art (SOTA) methods in both quantitative metrics and visual quality in real-world scenarios, highlighting its strong practical applicability. 

**Abstract (ZH)**: 基于两阶段随机交替框架的强监督 pansharpening 方法（TRA-PAN）：结合低分辨率图像的监督约束与高分辨率图像的物理特性 

---
# dcFCI: Robust Causal Discovery Under Latent Confounding, Unfaithfulness, and Mixed Data 

**Title (ZH)**: dcFCI: 在潜在混杂因素、不忠实性和混合数据下的稳健因果发现 

**Authors**: Adèle H. Ribeiro, Dominik Heider  

**Link**: [PDF](https://arxiv.org/pdf/2505.06542)  

**Abstract**: Causal discovery is central to inferring causal relationships from observational data. In the presence of latent confounding, algorithms such as Fast Causal Inference (FCI) learn a Partial Ancestral Graph (PAG) representing the true model's Markov Equivalence Class. However, their correctness critically depends on empirical faithfulness, the assumption that observed (in)dependencies perfectly reflect those of the underlying causal model, which often fails in practice due to limited sample sizes. To address this, we introduce the first nonparametric score to assess a PAG's compatibility with observed data, even with mixed variable types. This score is both necessary and sufficient to characterize structural uncertainty and distinguish between distinct PAGs. We then propose data-compatible FCI (dcFCI), the first hybrid causal discovery algorithm to jointly address latent confounding, empirical unfaithfulness, and mixed data types. dcFCI integrates our score into an (Anytime)FCI-guided search that systematically explores, ranks, and validates candidate PAGs. Experiments on synthetic and real-world scenarios demonstrate that dcFCI significantly outperforms state-of-the-art methods, often recovering the true PAG even in small and heterogeneous datasets. Examining top-ranked PAGs further provides valuable insights into structural uncertainty, supporting more robust and informed causal reasoning and decision-making. 

**Abstract (ZH)**: 因果发现是从观察数据中推断因果关系的核心。在潜在共因存在的情况下，快速因果推理（FCI）算法学习一个部分祖先图（PAG），代表真实模型的马尔可夫等价类。然而，它们的正确性严格依赖于经验忠实性假设，即观测到的（无）相关性完美地反映了潜在因果模型中的（无）相关性，但受限于样本量有限，这一假设在实践中经常失效。为解决这一问题，我们引入了第一个非参数分数，以评估PAG与观测数据的兼容性，即使变量类型混合也不例外。该分数既是表征结构不确定性所必需的，也是区分不同PAG所充分的。然后，我们提出了数据兼容性FCI（dcFCI），这是第一个同时解决潜在共因、经验不忠实性和混合数据类型的混合因果发现算法。dcFCI将我们的分数整合到一个由FCI引导的搜索中，系统地探索、排名和验证候选的PAGs。实验结果显示，dcFCI显著优于现有方法，在小规模和异质数据集中经常能够恢复真实的PAG。进一步研究排名靠前的PAGs提供了关于结构不确定性的宝贵见解，支持更稳健和知情的因果推理和决策。 

---
# Improving Generalization of Medical Image Registration Foundation Model 

**Title (ZH)**: 改进医学图像配准基础模型的泛化能力 

**Authors**: Jing Hu, Kaiwei Yu, Hongjiang Xian, Shu Hu, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06527)  

**Abstract**: Deformable registration is a fundamental task in medical image processing, aiming to achieve precise alignment by establishing nonlinear correspondences between images. Traditional methods offer good adaptability and interpretability but are limited by computational efficiency. Although deep learning approaches have significantly improved registration speed and accuracy, they often lack flexibility and generalizability across different datasets and tasks. In recent years, foundation models have emerged as a promising direction, leveraging large and diverse datasets to learn universal features and transformation patterns for image registration, thus demonstrating strong cross-task transferability. However, these models still face challenges in generalization and robustness when encountering novel anatomical structures, varying imaging conditions, or unseen modalities. To address these limitations, this paper incorporates Sharpness-Aware Minimization (SAM) into foundation models to enhance their generalization and robustness in medical image registration. By optimizing the flatness of the loss landscape, SAM improves model stability across diverse data distributions and strengthens its ability to handle complex clinical scenarios. Experimental results show that foundation models integrated with SAM achieve significant improvements in cross-dataset registration performance, offering new insights for the advancement of medical image registration technology. Our code is available at this https URL}{this https URL\_sam. 

**Abstract (ZH)**: 变形注册是医学图像处理中的一个基本任务，旨在通过建立图像之间的非线性对应关系实现精确对齐。传统的注册方法提供了良好的适应性和可解释性，但在计算效率上受到限制。尽管深度学习方法显著提高了注册速度和准确性，但在不同数据集和任务上的灵活性和普遍性仍然不足。近年来，基础模型作为一种有前途的方向出现，利用大规模和多样化的数据集学习图像注册中的通用特征和变换模式，从而展示了强大的跨任务迁移能力。然而，这些模型在遇到新型解剖结构、变化的成像条件或未见过的模态时，依然面临泛化能力和鲁棒性的挑战。为了解决这些局限性，本文将Sharpness-Aware Minimization (SAM)纳入基础模型中，以增强其在医学图像注册中的泛化能力和鲁棒性。通过优化损失景观的平坦度，SAM提高了模型在不同数据分布下的稳定性，并增强了其处理复杂临床场景的能力。实验结果表明，集成SAM的基础模型在跨数据集的注册性能上取得了显著提升，为医学图像注册技术的进步提供了新的见解。我们的代码可在此处访问。 

---
# PRUNE: A Patching Based Repair Framework for Certiffable Unlearning of Neural Networks 

**Title (ZH)**: PRUNE: 基于 patches 的可验证遗忘神经网络修复框架 

**Authors**: Xuran Li, Jingyi Wang, Xiaohan Yuan, Peixin Zhang, Zhan Qin, Zhibo Wang, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.06520)  

**Abstract**: It is often desirable to remove (a.k.a. unlearn) a speciffc part of the training data from a trained neural network model. A typical application scenario is to protect the data holder's right to be forgotten, which has been promoted by many recent regulation rules. Existing unlearning methods involve training alternative models with remaining data, which may be costly and challenging to verify from the data holder or a thirdparty auditor's perspective. In this work, we provide a new angle and propose a novel unlearning approach by imposing carefully crafted "patch" on the original neural network to achieve targeted "forgetting" of the requested data to delete. Speciffcally, inspired by the research line of neural network repair, we propose to strategically seek a lightweight minimum "patch" for unlearning a given data point with certiffable guarantee. Furthermore, to unlearn a considerable amount of data points (or an entire class), we propose to iteratively select a small subset of representative data points to unlearn, which achieves the effect of unlearning the whole set. Extensive experiments on multiple categorical datasets demonstrates our approach's effectiveness, achieving measurable unlearning while preserving the model's performance and being competitive in efffciency and memory consumption compared to various baseline methods. 

**Abstract (ZH)**: 从训练神经网络模型中移除特定部分的训练数据往往是有益的（即去学习）。一个典型的应用场景是保护数据持有者的被遗忘权，这已被许多最近的法规推广。现有的去学习方法涉及使用剩余数据训练替代模型，这从数据持有者或第三方审核员的角度来看可能是成本高且难以验证的。在本工作中，我们提供了一个新的视角，并提出了一种新的去学习方法，通过对原始神经网络施加精心设计的“补丁”以实现对指定数据的“遗忘”。具体而言，受神经网络修复研究方向的启发，我们提出了一种战略上寻找轻量级最小“补丁”来去学习给定数据点，同时具有可验证的保证。此外，为了去学习大量数据点（或整个类别），我们提出迭代选择一小部分有代表性的数据点来去学习，从而实现对整个数据集去学习的效果。在多个分类数据集上的广泛实验表明，我们的方法在提高去学习效果的同时保持了模型性能，并且在效率和内存消耗方面与各种基线方法相比具有竞争力。 

---
# Attention Mechanisms in Dynamical Systems: A Case Study with Predator-Prey Models 

**Title (ZH)**: 动态系统中的注意力机制：以捕食者-猎物模型为例 

**Authors**: David Balaban  

**Link**: [PDF](https://arxiv.org/pdf/2505.06503)  

**Abstract**: Attention mechanisms are widely used in artificial intelligence to enhance performance and interpretability. In this paper, we investigate their utility in modeling classical dynamical systems -- specifically, a noisy predator-prey (Lotka-Volterra) system. We train a simple linear attention model on perturbed time-series data to reconstruct system trajectories. Remarkably, the learned attention weights align with the geometric structure of the Lyapunov function: high attention corresponds to flat regions (where perturbations have small effect), and low attention aligns with steep regions (where perturbations have large effect). We further demonstrate that attention-based weighting can serve as a proxy for sensitivity analysis, capturing key phase-space properties without explicit knowledge of the system equations. These results suggest a novel use of AI-derived attention for interpretable, data-driven analysis and control of nonlinear systems. For example our framework could support future work in biological modeling of circadian rhythms, and interpretable machine learning for dynamical environments. 

**Abstract (ZH)**: 注意力机制在增强人工智能性能和可解释性方面广泛应用。本文探讨了其在建模经典动力系统中的作用——具体而言，是噪声扰动的捕食者-猎物（Lotka-Volterra）系统。我们训练一个简单的线性注意力模型来重构系统轨迹。值得注意的是，学习到的注意力权重与李雅普诺夫函数的几何结构相一致：高注意力对应平坦区域（扰动影响小），低注意力对应陡峭区域（扰动影响大）。此外，我们还展示了基于注意力的加权可以作为灵敏度分析的代理，无需显式了解系统方程即可捕捉相空间的关键特性。这些结果表明，AI提取的注意力在非线性系统可解释的数据驱动分析和控制中具有新颖的应用。例如，我们的框架可以支持未来关于生物节奏建模的工作，以及动态环境中的可解释机器学习。 

---
# xGen-small Technical Report 

**Title (ZH)**: xGen-small 技术报告 

**Authors**: Erik Nijkamp, Bo Pang, Egor Pakhomov, Akash Gokul, Jin Qu, Silvio Savarese, Yingbo Zhou, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06496)  

**Abstract**: We introduce xGen-small, a family of 4B and 9B Transformer decoder models optimized for long-context applications. Our vertically integrated pipeline unites domain-balanced, frequency-aware data curation; multi-stage pre-training with quality annealing and length extension to 128k tokens; and targeted post-training via supervised fine-tuning, preference learning, and online reinforcement learning. xGen-small delivers strong performance across various tasks, especially in math and coding domains, while excelling at long context benchmarks. 

**Abstract (ZH)**: xGen-small: 一种优化用于长上下文应用的4B和9B Transformer解码器模型 

---
# Improved Uncertainty Quantification in Physics-Informed Neural Networks Using Error Bounds and Solution Bundles 

**Title (ZH)**: 使用误差界和解集改进物理引导神经网络中的不确定性量化 

**Authors**: Pablo Flores, Olga Graf, Pavlos Protopapas, Karim Pichara  

**Link**: [PDF](https://arxiv.org/pdf/2505.06459)  

**Abstract**: Physics-Informed Neural Networks (PINNs) have been widely used to obtain solutions to various physical phenomena modeled as Differential Equations. As PINNs are not naturally equipped with mechanisms for Uncertainty Quantification, some work has been done to quantify the different uncertainties that arise when dealing with PINNs. In this paper, we use a two-step procedure to train Bayesian Neural Networks that provide uncertainties over the solutions to differential equation systems provided by PINNs. We use available error bounds over PINNs to formulate a heteroscedastic variance that improves the uncertainty estimation. Furthermore, we solve forward problems and utilize the obtained uncertainties when doing parameter estimation in inverse problems in cosmology. 

**Abstract (ZH)**: 物理导向的神经网络（PINNs）已经被广泛应用于解决各种由微分方程描述的物理现象。由于PINNs本身不具备不确定性量化机制，一些工作致力于量化处理PINNs时出现的不同不确定性。在本文中，我们采用两步训练程序构建贝叶斯神经网络，为PINNs提供的微分方程系统解提供不确定性。我们利用PINNs的可用误差边界来制定异方差方差，以改进不确定性估计。此外，我们在宇宙学中利用获得的不确定性解决正向问题，并进行参数估计的逆向问题。 

---
# What Do People Want to Know About Artificial Intelligence (AI)? The Importance of Answering End-User Questions to Explain Autonomous Vehicle (AV) Decisions 

**Title (ZH)**: 人们最想了解的人工智能（AI）是什么？解答最终用户的问题以解释自动驾驶车辆（AV）的决策的重要性 

**Authors**: Somayeh Molaei, Lionel P. Robert, Nikola Banovic  

**Link**: [PDF](https://arxiv.org/pdf/2505.06428)  

**Abstract**: Improving end-users' understanding of decisions made by autonomous vehicles (AVs) driven by artificial intelligence (AI) can improve utilization and acceptance of AVs. However, current explanation mechanisms primarily help AI researchers and engineers in debugging and monitoring their AI systems, and may not address the specific questions of end-users, such as passengers, about AVs in various scenarios. In this paper, we conducted two user studies to investigate questions that potential AV passengers might pose while riding in an AV and evaluate how well answers to those questions improve their understanding of AI-driven AV decisions. Our initial formative study identified a range of questions about AI in autonomous driving that existing explanation mechanisms do not readily address. Our second study demonstrated that interactive text-based explanations effectively improved participants' comprehension of AV decisions compared to simply observing AV decisions. These findings inform the design of interactions that motivate end-users to engage with and inquire about the reasoning behind AI-driven AV decisions. 

**Abstract (ZH)**: 提高最终用户对由人工智能驱动的自主车辆（AVs）所做的决策的理解可以提高自主车辆的使用率和接受度。然而，当前的解释机制主要有助于人工智能研究人员和工程师调试和监控其人工智能系统，可能无法解决最终用户（如乘客）在各种场景中对自主车辆的具体问题。本文通过两项用户研究调查了潜在自主车辆乘客在乘坐自主车辆时可能会提出的问题，并评估了这些问题的回答如何提高他们对人工智能驱动的自主车辆决策的理解。初步形成性研究确定了一类现有解释机制无法轻松解答的关于自主驾驶中人工智能的问题。第二项研究证明，互动的基于文本的解释比仅观察自主车辆的决策更有效地提高了参与者对自主车辆决策的理解。这些发现为设计能够促使最终用户参与并了解人工智能驱动的自主车辆决策原因的交互界面提供了指导。 

---
# Engineering Risk-Aware, Security-by-Design Frameworks for Assurance of Large-Scale Autonomous AI Models 

**Title (ZH)**: 为大规模自主AI模型提供保障的风险意识与设计安全框架 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2505.06409)  

**Abstract**: As AI models scale to billions of parameters and operate with increasing autonomy, ensuring their safe, reliable operation demands engineering-grade security and assurance frameworks. This paper presents an enterprise-level, risk-aware, security-by-design approach for large-scale autonomous AI systems, integrating standardized threat metrics, adversarial hardening techniques, and real-time anomaly detection into every phase of the development lifecycle. We detail a unified pipeline - from design-time risk assessments and secure training protocols to continuous monitoring and automated audit logging - that delivers provable guarantees of model behavior under adversarial and operational stress. Case studies in national security, open-source model governance, and industrial automation demonstrate measurable reductions in vulnerability and compliance overhead. Finally, we advocate cross-sector collaboration - uniting engineering teams, standards bodies, and regulatory agencies - to institutionalize these technical safeguards within a resilient, end-to-end assurance ecosystem for the next generation of AI. 

**Abstract (ZH)**: 随着AI模型参数数量达到十亿级别并展现出越来越高的自主性，确保其安全可靠的运行需要工程级别的安全和保证框架。本文提出了一种面向企业的、具备风险意识的设计安全方法，用于大规模自主AI系统，该方法将标准化威胁度量、对抗性强化技术和实时异常检测整合到开发生命周期的每一个阶段。我们详细阐述了一条统一的管线——从设计时的风险评估和安全训练协议到持续监控和自动审计日志记录——以在对抗和运营压力下提供可验证的模型行为保证。在国家安全、开源模型治理和工业自动化领域的案例研究证明了可测量的漏洞和合规开销减少。最后，我们倡导跨领域的合作——结合工程团队、标准机构和监管机构——在具有韧性的端到端保证生态系统中制度化这些技术保护措施，以为下一代AI提供服务。 

---
# Offensive Security for AI Systems: Concepts, Practices, and Applications 

**Title (ZH)**: AI系统进攻性安全：概念、实践与应用 

**Authors**: Josh Harguess, Chris M. Ward  

**Link**: [PDF](https://arxiv.org/pdf/2505.06380)  

**Abstract**: As artificial intelligence (AI) systems become increasingly adopted across sectors, the need for robust, proactive security strategies is paramount. Traditional defensive measures often fall short against the unique and evolving threats facing AI-driven technologies, making offensive security an essential approach for identifying and mitigating risks. This paper presents a comprehensive framework for offensive security in AI systems, emphasizing proactive threat simulation and adversarial testing to uncover vulnerabilities throughout the AI lifecycle. We examine key offensive security techniques, including weakness and vulnerability assessment, penetration testing, and red teaming, tailored specifically to address AI's unique susceptibilities. By simulating real-world attack scenarios, these methodologies reveal critical insights, informing stronger defensive strategies and advancing resilience against emerging threats. This framework advances offensive AI security from theoretical concepts to practical, actionable methodologies that organizations can implement to strengthen their AI systems against emerging threats. 

**Abstract (ZH)**: 随着人工智能（AI）系统在各领域的广泛应用，建立 robust、 proactive 的安全策略变得至关重要。传统防御措施往往无法有效应对 AI 驱动技术所面临的独特且不断演变的威胁，因此，采取主动防御安全策略以识别和减轻风险变得必不可少。本文提出了一个全面的 AI 系统主动防御安全框架，强调在 AI 生命周期中进行积极的威胁模拟和对抗性测试以揭示漏洞。我们探讨了关键的主动防御安全技术，包括脆弱性和漏洞评估、渗透测试和红队演练，这些技术特别针对解决 AI 的独特脆弱性进行了定制。通过模拟真实世界的攻击场景，这些方法论揭示了关键见解，为制定更强的防御策略并提高对新兴威胁的抵御能力奠定了基础。该框架将主动 AI 安全从理论概念推进到可操作的实际方法论，为企业提供实施以加强其 AI 系统抵御新兴威胁的策略。 

---
# The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization 

**Title (ZH)**: ML.ENERGY基准：迈向自动化推理能耗测量与优化 

**Authors**: Jae-Won Chung, Jiachen Liu, Jeff J. Ma, Ruofan Wu, Oh Jun Kweon, Yuxuan Xia, Zhiyu Wu, Mosharaf Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.06371)  

**Abstract**: As the adoption of Generative AI in real-world services grow explosively, energy has emerged as a critical bottleneck resource. However, energy remains a metric that is often overlooked, under-explored, or poorly understood in the context of building ML systems. We present the this http URL Benchmark, a benchmark suite and tool for measuring inference energy consumption under realistic service environments, and the corresponding this http URL Leaderboard, which have served as a valuable resource for those hoping to understand and optimize the energy consumption of their generative AI services. In this paper, we explain four key design principles for benchmarking ML energy we have acquired over time, and then describe how they are implemented in the this http URL Benchmark. We then highlight results from the latest iteration of the benchmark, including energy measurements of 40 widely used model architectures across 6 different tasks, case studies of how ML design choices impact energy consumption, and how automated optimization recommendations can lead to significant (sometimes more than 40%) energy savings without changing what is being computed by the model. The this http URL Benchmark is open-source and can be easily extended to various customized models and application scenarios. 

**Abstract (ZH)**: 随着生成式AI在实际服务中的应用爆炸式增长，能源已成为一个关键的瓶颈资源。然而，在构建机器学习系统的过程中，能源仍然是一个常被忽视、未充分探索或不完全理解的指标。我们介绍了this http URL基准测试，这是一个用于在实际服务环境中测量推理能耗的基准套件和工具，以及相应的this http URL排行榜，它们已成为希望了解并优化其生成式AI服务能耗的研究人员的重要资源。在本文中，我们解释了我们在长期研究中获得的四个关键设计原则，并描述了这些原则在this http URL基准测试中的实现方式。然后，我们强调了基准测试最新迭代的结果，包括40种广泛使用的模型架构在6种不同任务上的能耗测量、ML设计选择对能耗影响的案例研究，以及自动优化建议如何在不改变模型计算内容的情况下实现显著（有时超过40%）的能耗节省。this http URL基准测试是开源的，并且可以轻松扩展到各种定制模型和应用场景。 

---
# Remote Rowhammer Attack using Adversarial Observations on Federated Learning Clients 

**Title (ZH)**: 面向联邦学习客户端的基于对抗观察的远程Rowhammer攻击 

**Authors**: Jinsheng Yuan, Yuhang Hao, Weisi Guo, Yun Wu, Chongyan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06335)  

**Abstract**: Federated Learning (FL) has the potential for simultaneous global learning amongst a large number of parallel agents, enabling emerging AI such as LLMs to be trained across demographically diverse data. Central to this being efficient is the ability for FL to perform sparse gradient updates and remote direct memory access at the central server. Most of the research in FL security focuses on protecting data privacy at the edge client or in the communication channels between the client and server. Client-facing attacks on the server are less well investigated as the assumption is that a large collective of clients offer resilience.
Here, we show that by attacking certain clients that lead to a high frequency repetitive memory update in the server, we can remote initiate a rowhammer attack on the server memory. For the first time, we do not need backdoor access to the server, and a reinforcement learning (RL) attacker can learn how to maximize server repetitive memory updates by manipulating the client's sensor observation. The consequence of the remote rowhammer attack is that we are able to achieve bit flips, which can corrupt the server memory. We demonstrate the feasibility of our attack using a large-scale FL automatic speech recognition (ASR) systems with sparse updates, our adversarial attacking agent can achieve around 70\% repeated update rate (RUR) in the targeted server model, effectively inducing bit flips on server DRAM. The security implications are that can cause disruptions to learning or may inadvertently cause elevated privilege. This paves the way for further research on practical mitigation strategies in FL and hardware design. 

**Abstract (ZH)**: 联邦学习中的远程行hammer攻击研究 

---
# Mask-PINNs: Regulating Feature Distributions in Physics-Informed Neural Networks 

**Title (ZH)**: Mask-PINNs：调节物理信息神经网络中特征分布 

**Authors**: Feilong Jiang, Xiaonan Hou, Jianqiao Ye, Min Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.06331)  

**Abstract**: Physics-Informed Neural Networks (PINNs) are a class of deep learning models designed to solve partial differential equations by incorporating physical laws directly into the loss function. However, the internal covariate shift, which has been largely overlooked, hinders the effective utilization of neural network capacity in PINNs. To this end, we propose Mask-PINNs, a novel architecture designed to address this issue in PINNs. Unlike traditional normalization methods such as BatchNorm or LayerNorm, we introduce a learnable, nonlinear mask function that constrains the feature distributions without violating underlying physics. The experimental results show that the proposed method significantly improves feature distribution stability, accuracy, and robustness across various activation functions and PDE benchmarks. Furthermore, it enables the stable and efficient training of wider networks a capability that has been largely overlooked in PINNs. 

**Abstract (ZH)**: 基于物理的神经网络（Mask-PINNs）：一种解决内部协变移位问题的新架构 

---
# Enterprise Architecture as a Dynamic Capability for Scalable and Sustainable Generative AI adoption: Bridging Innovation and Governance in Large Organisations 

**Title (ZH)**: 企业架构作为可扩展和可持续生成式AI adoption的动态能力：在大型组织中bridging创新与治理 

**Authors**: Alexander Ettinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.06326)  

**Abstract**: Generative Artificial Intelligence is a powerful new technology with the potential to boost innovation and reshape governance in many industries. Nevertheless, organisations face major challenges in scaling GenAI, including technology complexity, governance gaps and resource misalignments. This study explores how Enterprise Architecture Management can meet the complex requirements of GenAI adoption within large enterprises. Based on a systematic literature review and the qualitative analysis of 16 semi-structured interviews with experts, it examines the relationships between EAM, dynamic capabilities and GenAI adoption. The review identified key limitations in existing EA frameworks, particularly their inability to fully address the unique requirements of GenAI. The interviews, analysed using the Gioia methodology, revealed critical enablers and barriers to GenAI adoption across industries. The findings indicate that EAM, when theorised as sensing, seizing and transforming dynamic capabilities, can enhance GenAI adoption by improving strategic alignment, governance frameworks and organisational agility. However, the study also highlights the need to tailor EA frameworks to GenAI-specific challenges, including low data governance maturity and the balance between innovation and compliance. Several conceptual frameworks are proposed to guide EA leaders in aligning GenAI maturity with organisational readiness. The work contributes to academic understanding and industry practice by clarifying the role of EA in bridging innovation and governance in disruptive technology environments. 

**Abstract (ZH)**: 生成式人工智能是一种强大的新技术，有可能促进创新并重塑许多行业的治理。然而，组织在扩大生成式人工智能的应用方面面临重大挑战，包括技术复杂性、治理缺口和资源配置不匹配。本研究探讨了企业架构管理如何满足大型企业中生成式人工智能采纳的复杂要求。基于系统文献综述及对16名专家进行的半结构化访谈的定性分析，研究了企业架构管理、动态能力和生成式人工智能采纳之间的关系。文献综述发现现有企业架构框架的关键局限性，特别是其无法充分解决生成式人工智能的独特要求。访谈结果（采用Goia方法分析）揭示了跨行业生成式人工智能采纳的关键促进因素和障碍。研究表明，当理论化为感知、捕获和转换动态能力时，企业架构管理可以提高生成式人工智能采纳的战略对齐、治理框架和组织敏捷性。然而，研究也指出了需要根据生成式人工智能的特定挑战来调整企业架构框架的需求，包括数据治理成熟度低以及创新与合规之间的平衡。提出了几个概念框架以指导企业架构领导者将生成式人工智能成熟度与组织准备度进行对齐。该工作通过阐明企业在颠覆性技术环境中促进创新和治理的角色，为学术研究和行业实践做出了贡献。 

---
# Human in the Latent Loop (HILL): Interactively Guiding Model Training Through Human Intuition 

**Title (ZH)**: 人类干预潜在循环（HILL）：通过人类直觉交互指导模型训练 

**Authors**: Daniel Geissler, Lars Krupp, Vishal Banwari, David Habusch, Bo Zhou, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2505.06325)  

**Abstract**: Latent space representations are critical for understanding and improving the behavior of machine learning models, yet they often remain obscure and intricate. Understanding and exploring the latent space has the potential to contribute valuable human intuition and expertise about respective domains. In this work, we present HILL, an interactive framework allowing users to incorporate human intuition into the model training by interactively reshaping latent space representations. The modifications are infused into the model training loop via a novel approach inspired by knowledge distillation, treating the user's modifications as a teacher to guide the model in reshaping its intrinsic latent representation. The process allows the model to converge more effectively and overcome inefficiencies, as well as provide beneficial insights to the user. We evaluated HILL in a user study tasking participants to train an optimal model, closely observing the employed strategies. The results demonstrated that human-guided latent space modifications enhance model performance while maintaining generalization, yet also revealing the risks of including user biases. Our work introduces a novel human-AI interaction paradigm that infuses human intuition into model training and critically examines the impact of human intervention on training strategies and potential biases. 

**Abstract (ZH)**: 隐空间表示对于理解并改进机器学习模型的行为至关重要，但往往晦涩难懂。理解和探索隐空间有可能为相应领域贡献宝贵的直觉和专业知识。在本工作中，我们提出了HILL，一个交互式框架，允许用户通过交互式重塑隐空间表示来将人类直觉融入模型训练中。通过借鉴知识蒸馏的方法，将用户的修改视为教师，指导模型重塑其固有的隐空间表示。该过程使模型能够更有效地收敛，克服效率低下，并为用户提供有益的见解。我们在一项用户研究中评估了HILL，要求参与者训练最优模型，并密切观察所采用的策略。结果表明，人类指导下的隐空间修改可以提升模型性能并保持泛化能力，但也揭示了纳入用户偏见的风险。我们的工作引入了一种新的人类-人工智能交互范式，将人类直觉融入模型训练，并批判性地探讨了人类干预对训练策略和潜在偏见的影响。 

---
# Divide (Text) and Conquer (Sentiment): Improved Sentiment Classification by Constituent Conflict Resolution 

**Title (ZH)**: 分而治之（情感冲突决议改进情感分类） 

**Authors**: Jan Kościałkowski, Paweł Marcinkowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.06320)  

**Abstract**: Sentiment classification, a complex task in natural language processing, becomes even more challenging when analyzing passages with multiple conflicting tones. Typically, longer passages exacerbate this issue, leading to decreased model performance. The aim of this paper is to introduce novel methodologies for isolating conflicting sentiments and aggregating them to effectively predict the overall sentiment of such passages. One of the aggregation strategies involves a Multi-Layer Perceptron (MLP) model which outperforms baseline models across various datasets, including Amazon, Twitter, and SST while costing $\sim$1/100 of what fine-tuning the baseline would take. 

**Abstract (ZH)**: 情感分类，自然语言处理中的一个复杂任务，在分析具有多种矛盾基调的段落时变得更加具有挑战性。通常，更长的段落会加剧这一问题，导致模型性能下降。本文旨在介绍新的方法来隔离矛盾的情感并将其聚合，以有效预测此类段落的整体情感。一种聚合策略涉及多层感知机（MLP）模型，该模型在包括Amazon、Twitter和SST在内的各种数据集上表现优于基线模型，成本仅为基线微调的1/100左右。 

---
# Threat Modeling for AI: The Case for an Asset-Centric Approach 

**Title (ZH)**: AI的安全威胁建模：资产中心化方法的必要性 

**Authors**: Jose Sanchez Vicarte, Marcin Spoczynski, Mostafa Elsaid  

**Link**: [PDF](https://arxiv.org/pdf/2505.06315)  

**Abstract**: Recent advances in AI are transforming AI's ubiquitous presence in our world from that of standalone AI-applications into deeply integrated AI-agents. These changes have been driven by agents' increasing capability to autonomously make decisions and initiate actions, using existing applications; whether those applications are AI-based or not. This evolution enables unprecedented levels of AI integration, with agents now able to take actions on behalf of systems and users -- including, in some cases, the powerful ability for the AI to write and execute scripts as it deems necessary. With AI systems now able to autonomously execute code, interact with external systems, and operate without human oversight, traditional security approaches fall short.
This paper introduces an asset-centric methodology for threat modeling AI systems that addresses the unique security challenges posed by integrated AI agents. Unlike existing top-down frameworks that analyze individual attacks within specific product contexts, our bottom-up approach enables defenders to systematically identify how vulnerabilities -- both conventional and AI-specific -- impact critical AI assets across distributed infrastructures used to develop and deploy these agents. This methodology allows security teams to: (1) perform comprehensive analysis that communicates effectively across technical domains, (2) quantify security assumptions about third-party AI components without requiring visibility into their implementation, and (3) holistically identify AI-based vulnerabilities relevant to their specific product context. This approach is particularly relevant for securing agentic systems with complex autonomous capabilities. By focusing on assets rather than attacks, our approach scales with the rapidly evolving threat landscape while accommodating increasingly complex and distributed AI development pipelines. 

**Abstract (ZH)**: 近期人工智能技术的发展正将人工智能在世界上的无处不在从独立的人工智能应用转变为深度整合的人工智能代理。这些变化是由代理日益增强的自主决策和行动能力驱动的，无论这些行动是由基于人工智能的应用还是非基于人工智能的应用触发的。这种演变使得前所未有的高水平人工智能整合成为可能，代理现在可以代表系统和用户采取行动——包括在某些情况下，代理具有编写和执行必要脚本的强大能力。随着人工智能系统现在能够自主执行代码、与外部系统交互并运行而无需人工监督，传统的安全方法已不再有效。本文提出了一种以资产为中心的威胁建模方法，以应对整合人工智能代理所带来的独特安全挑战。与现有的自上而下框架仅在特定产品背景下分析单一攻击不同，我们自下而上的方法允许防守者系统地识别传统漏洞和人工智能特定漏洞如何影响分布式基础设施中的关键人工智能资产。这种方法使安全团队能够：(1) 进行全面分析，有效跨越技术领域沟通，(2) 不要求对第三方人工智能组件的实现可见性即可量化安全假设，以及(3) 从具体的产品环境中全面识别人工智能相关的漏洞。这种方法特别适用于保护具有复杂自主能力的代理系统。通过关注资产而非攻击，我们的方法能够适应迅速演变的威胁环境，并适应日益复杂和分布的人工智能开发流水线。 

---
# A4L: An Architecture for AI-Augmented Learning 

**Title (ZH)**: A4L：一种AI增强学习架构 

**Authors**: Ashok Goel, Ploy Thajchayapong, Vrinda Nandan, Harshvardhan Sikka, Spencer Rugaber  

**Link**: [PDF](https://arxiv.org/pdf/2505.06314)  

**Abstract**: AI promises personalized learning and scalable education. As AI agents increasingly permeate education in support of teaching and learning, there is a critical and urgent need for data architectures for collecting and analyzing data on learning, and feeding the results back to teachers, learners, and the AI agents for personalization of learning at scale. At the National AI Institute for Adult Learning and Online Education, we are developing an Architecture for AI-Augmented Learning (A4L) for supporting adult learning through online education. We present the motivations, goals, requirements of the A4L architecture. We describe preliminary applications of A4L and discuss how it advances the goals of making learning more personalized and scalable. 

**Abstract (ZH)**: AI承诺个性化学习与规模化教育：为支持成人在线教育，我们正在开发AI增强学习架构（A4L），以推动学习更加个性化和规模化。 

---
# AI Approaches to Qualitative and Quantitative News Analytics on NATO Unity 

**Title (ZH)**: AI方法在北约团结行动定性与定量新闻分析中的应用 

**Authors**: Bohdan M. Pavlyshenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06313)  

**Abstract**: The paper considers the use of GPT models with retrieval-augmented generation (RAG) for qualitative and quantitative analytics on NATO sentiments, NATO unity and NATO Article 5 trust opinion scores in different web sources: news sites found via Google Search API, Youtube videos with comments, and Reddit discussions. A RAG approach using GPT-4.1 model was applied to analyse news where NATO related topics were discussed. Two levels of RAG analytics were used: on the first level, the GPT model generates qualitative news summaries and quantitative opinion scores using zero-shot prompts; on the second level, the GPT model generates the summary of news summaries. Quantitative news opinion scores generated by the GPT model were analysed using Bayesian regression to get trend lines. The distributions found for the regression parameters make it possible to analyse an uncertainty in specified news opinion score trends. Obtained results show a downward trend for analysed scores of opinion related to NATO unity.
This approach does not aim to conduct real political analysis; rather, it consider AI based approaches which can be used for further analytics
as a part of a complex analytical approach. The obtained results demonstrate that the use of GPT models for news analysis can give informative qualitative and quantitative analytics, providing important insights.
The dynamic model based on neural ordinary differential equations was considered for modelling public opinions. This approach makes it possible to analyse different scenarios for evolving public opinions. 

**Abstract (ZH)**: 使用GPT模型结合检索增强生成（RAG）方法对北约 sentiment、团结和Article 5信任意见分数进行定性和定量分析：基于Google Search API检索的新闻网站、YouTube视频评论和Reddit讨论。采用GPT-4.1模型运用RAG方法分析与北约相关的新闻摘要。使用零-shot提示生成定性新闻概述和定量意见分数，并在第二层生成摘要的概述；通过贝叶斯回归分析生成的意见分数趋势，评估新闻意见分数趋势的不确定性。研究结果表明，分析分数呈下降趋势。该方法旨在为复杂的分析方法提供基于AI的工具，结果表明使用GPT模型进行新闻分析可以提供有用的信息和定量见解。基于神经常微分方程的动态模型用于 modelling 公众意见，该方法能够分析公众意见演变的不同情景。 

---
# Responsibility Gap in Collective Decision Making 

**Title (ZH)**: 集体决策中的责任缺口 

**Authors**: Pavel Naumov, Jia Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06312)  

**Abstract**: The responsibility gap is a set of outcomes of a collective decision-making mechanism in which no single agent is individually responsible. In general, when designing a decision-making process, it is desirable to minimise the gap.
The paper proposes a concept of an elected dictatorship. It shows that, in a perfect information setting, the gap is empty if and only if the mechanism is an elected dictatorship. It also proves that in an imperfect information setting, the class of gap-free mechanisms is positioned strictly between two variations of the class of elected dictatorships. 

**Abstract (ZH)**: 责任缺口是集体决策机制的一种结果，在这种机制中，没有单一代理个体承担责任。通常，在设计决策过程时，应尽量减小这种缺口。
本文提出了当选制独裁的概念。它证明，在完美信息的环境下，当且仅当机制是当选制独裁时，缺口为空。此外，它还证明，在不完美信息的环境下，无缺口机制类严格位于两种当选制独裁类的变体之间。 

---
# Domain-Adversarial Anatomical Graph Networks for Cross-User Human Activity Recognition 

**Title (ZH)**: 跨用户人体活动识别的领域对抗解剖图网络 

**Authors**: Xiaozhou Ye, Kevin I-Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06301)  

**Abstract**: Cross-user variability in Human Activity Recognition (HAR) remains a critical challenge due to differences in sensor placement, body dynamics, and behavioral patterns. Traditional methods often fail to capture biomechanical invariants that persist across users, limiting their generalization capability. We propose an Edge-Enhanced Graph-Based Adversarial Domain Generalization (EEG-ADG) framework that integrates anatomical correlation knowledge into a unified graph neural network (GNN) architecture. By modeling three biomechanically motivated relationships together-Interconnected Units, Analogous Units, and Lateral Units-our method encodes domain-invariant features while addressing user-specific variability through Variational Edge Feature Extractor. A Gradient Reversal Layer (GRL) enforces adversarial domain generalization, ensuring robustness to unseen users. Extensive experiments on OPPORTUNITY and DSADS datasets demonstrate state-of-the-art performance. Our work bridges biomechanical principles with graph-based adversarial learning by integrating information fusion techniques. This fusion of information underpins our unified and generalized model for cross-user HAR. 

**Abstract (ZH)**: 跨用户的动作识别（HAR）由于传感器放置、身体动力学和行为模式的差异仍是一项关键挑战。传统的方法往往无法捕捉到在不同用户间保持不变的生物力学不变量，限制了其泛化能力。我们提出了一种基于图的对抗域泛化增强边缘（EEG-ADG）框架，将解剖学相关知识融合到统一的图神经网络（GNN）架构中。通过一起建模三种生物力学驱动的关系—互联单元、同源单元和侧向单元—我们的方法在编码域不变特征的同时，通过变分边缘特征提取器解决用户特定的变异性。 gradient reversal层（GRL）促进对抗域泛化，确保对未见用户具有鲁棒性。在OPPORTUNITY和DSADS数据集上的广泛实验显示出最先进的性能。我们的工作通过集成信息融合技术将生物力学原理与图基对抗学习相结合，支撑了我们针对跨用户HAR的统一和泛化模型。 

---
# Input-Specific and Universal Adversarial Attack Generation for Spiking Neural Networks in the Spiking Domain 

**Title (ZH)**: 输入特定性和普遍性的对抗攻击生成用于突触神经网络的突触域 

**Authors**: Spyridon Raptis, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.06299)  

**Abstract**: As Spiking Neural Networks (SNNs) gain traction across various applications, understanding their security vulnerabilities becomes increasingly important. In this work, we focus on the adversarial attacks, which is perhaps the most concerning threat. An adversarial attack aims at finding a subtle input perturbation to fool the network's decision-making. We propose two novel adversarial attack algorithms for SNNs: an input-specific attack that crafts adversarial samples from specific dataset inputs and a universal attack that generates a reusable patch capable of inducing misclassification across most inputs, thus offering practical feasibility for real-time deployment. The algorithms are gradient-based operating in the spiking domain proving to be effective across different evaluation metrics, such as adversarial accuracy, stealthiness, and generation time. Experimental results on two widely used neuromorphic vision datasets, NMNIST and IBM DVS Gesture, show that our proposed attacks surpass in all metrics all existing state-of-the-art methods. Additionally, we present the first demonstration of adversarial attack generation in the sound domain using the SHD dataset. 

**Abstract (ZH)**: 随着脉冲神经网络（SNNs）在各种应用中逐渐受到重视，理解其安全漏洞变得越来越重要。在本文中，我们重点关注敌对攻击，这可能是最令人关切的威胁。敌对攻击旨在找到一种微妙的输入扰动以迷惑网络的决策过程。我们提出了两种针对SNNs的新颖敌对攻击算法：一种是输入特定攻击，从特定数据集输入中创建敌对样本；另一种是通用攻击，生成一个可重用的补丁，能够在大多数输入中诱发错误分类，从而为实时部署提供了实用性。这些算法基于梯度，工作在脉冲域，证明在不同的评估指标（如敌对准确性、隐蔽性和生成时间）上都是有效的。实验结果表明，我们提出的攻击在所有指标上都超过了所有现有的前沿方法。此外，我们还首次在SHD数据集中展示了声音域中的敌对攻击生成。 

---
# Terahertz Spatial Wireless Channel Modeling with Radio Radiance Field 

**Title (ZH)**: 太赫兹空间无线信道建模与射流场理论 

**Authors**: John Song, Lihao Zhang, Feng Ye, Haijian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.06277)  

**Abstract**: Terahertz (THz) communication is a key enabler for 6G systems, offering ultra-wide bandwidth and unprecedented data rates. However, THz signal propagation differs significantly from lower-frequency bands due to severe free space path loss, minimal diffraction and specular reflection, and prominent scattering, making conventional channel modeling and pilot-based estimation approaches inefficient. In this work, we investigate the feasibility of applying radio radiance field (RRF) framework to the THz band. This method reconstructs a continuous RRF using visual-based geometry and sparse THz RF measurements, enabling efficient spatial channel state information (Spatial-CSI) modeling without dense sampling. We first build a fine simulated THz scenario, then we reconstruct the RRF and evaluate the performance in terms of both reconstruction quality and effectiveness in THz communication, showing that the reconstructed RRF captures key propagation paths with sparse training samples. Our findings demonstrate that RRF modeling remains effective in the THz regime and provides a promising direction for scalable, low-cost spatial channel reconstruction in future 6G networks. 

**Abstract (ZH)**: 太赫兹（THz）通信是6G系统的关键使能技术，提供超宽带和空前的数据速率。然而，THz信号传播由于严重的自由空间路径损耗、最小的衍射和镜面反射以及显著的散射与较低频率 band 有所不同，使得传统的信道建模和基于导频的估计方法效率低下。在本文中，我们探讨了将无线电辐射场（RRF）框架应用于THz频段的可能性。该方法使用基于视觉的几何和稀疏的THz射频测量重构连续的RRF，使得无需密集采样即可高效建模空间信道状态信息（Spatial-CSI）。我们首先构建了一个精细模拟的THz场景，然后重构RRF并从重构质量及在THz通信中的有效性两个方面评估其性能，表明重构的RRF能够在稀疏训练样本下捕捉到关键的传播路径。我们的研究结果表明，RRF建模在THz频段仍然有效，并为未来6G网络中具备扩展性和低成本的空间信道重构提供了一个有前景的方向。 

---
# Policy-labeled Preference Learning: Is Preference Enough for RLHF? 

**Title (ZH)**: 基于政策标注的偏好学习：偏好足够用于RLHF吗？ 

**Authors**: Taehyun Cho, Seokhun Ju, Seungyub Han, Dohyeong Kim, Kyungjae Lee, Jungwoo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.06273)  

**Abstract**: To design rewards that align with human goals, Reinforcement Learning from Human Feedback (RLHF) has emerged as a prominent technique for learning reward functions from human preferences and optimizing policies via reinforcement learning algorithms. However, existing RLHF methods often misinterpret trajectories as being generated by an optimal policy, causing inaccurate likelihood estimation and suboptimal learning. Inspired by Direct Preference Optimization framework which directly learns optimal policy without explicit reward, we propose policy-labeled preference learning (PPL), to resolve likelihood mismatch issues by modeling human preferences with regret, which reflects behavior policy information. We also provide a contrastive KL regularization, derived from regret-based principles, to enhance RLHF in sequential decision making. Experiments in high-dimensional continuous control tasks demonstrate PPL's significant improvements in offline RLHF performance and its effectiveness in online settings. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）中的奖励设计：直接偏好优化及其在顺序决策中的应用 

---
# A Sensitivity-Driven Expert Allocation Method in LoRA-MoE for Efficient Fine-Tuning 

**Title (ZH)**: 基于灵敏度驱动的专家分配方法在LoRA-MoE中的高效微调 

**Authors**: Junzhou Xu, Boyu Diao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06272)  

**Abstract**: As deep learning models expand, the pre-training-fine-tuning paradigm has become the standard approach for handling various downstream tasks. However, shared parameters can lead to diminished performance when dealing with complex datasets involving multiple tasks. While introducing Mixture-of-Experts (MoE) methods has alleviated this issue to some extent, it also significantly increases the number of parameters required for fine-tuning and training time, introducing greater parameter redundancy. To address these challenges, we propose a method for allocating expert numbers based on parameter sensitivity LoRA-SMoE (A Sensitivity-Driven Expert Allocation Method in LoRA-MoE for Efficient Fine-Tuning). This method rapidly assesses the sensitivity of different tasks to parameters by sampling a small amount of data and using gradient information. It then adaptively allocates expert numbers within a given budget. The process maintains comparable memory consumption to LoRA (Low-Rank Adaptation) while ensuring an efficient and resource-friendly fine-tuning procedure. Experimental results demonstrate that compared to SOTA fine-tuning methods, our LoRA-SMoE approach can enhance model performance while reducing the number of trainable parameters. This significantly improves model performance in resource-constrained environments. Additionally, due to its efficient parameter sensitivity evaluation mechanism, LoRA-SMoE requires minimal computational overhead to optimize expert allocation, making it particularly suitable for scenarios with limited computational resources. All the code in this study will be made publicly available following the acceptance of the paper for publication. Source code is at this https URL 

**Abstract (ZH)**: 基于参数灵敏度的LoRA-SMoE专家分配方法：一种在LoRA-MoE中的灵敏度驱动专家分配方法以提高高效微调性能 

---
# Tri-MTL: A Triple Multitask Learning Approach for Respiratory Disease Diagnosis 

**Title (ZH)**: 三重多任务学习：一种呼吸系统疾病诊断方法 

**Authors**: June-Woo Kim, Sanghoon Lee, Miika Toikkanen, Daehwan Hwang, Kyunghoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.06271)  

**Abstract**: Auscultation remains a cornerstone of clinical practice, essential for both initial evaluation and continuous monitoring. Clinicians listen to the lung sounds and make a diagnosis by combining the patient's medical history and test results. Given this strong association, multitask learning (MTL) can offer a compelling framework to simultaneously model these relationships, integrating respiratory sound patterns with disease manifestations. While MTL has shown considerable promise in medical applications, a significant research gap remains in understanding the complex interplay between respiratory sounds, disease manifestations, and patient metadata attributes. This study investigates how integrating MTL with cutting-edge deep learning architectures can enhance both respiratory sound classification and disease diagnosis. Specifically, we extend recent findings regarding the beneficial impact of metadata on respiratory sound classification by evaluating its effectiveness within an MTL framework. Our comprehensive experiments reveal significant improvements in both lung sound classification and diagnostic performance when the stethoscope information is incorporated into the MTL architecture. 

**Abstract (ZH)**: 听诊 remains 临床实践的基础，对于初步评估和持续监测都至关重要。医护人员通过结合患者的病史和检测结果来听诊肺部声音并进行诊断。鉴于这一紧密关联，多任务学习（MTL）可以提供一个引人入胜的框架，同时建模这些关系，整合呼吸音模式与疾病表现。尽管 MTL 在医疗应用中展现出巨大潜力，但仍存在关于呼吸音、疾病表现和患者元数据属性之间复杂互动关系理解的研究空白。本研究探讨了将 MTL 与前沿的深度学习架构结合如何增强呼吸音分类和疾病诊断。具体而言，我们扩展了关于元数据对呼吸音分类有益影响的近期发现，评估其在 MTL 框架内的有效性。我们全面的实验结果显示，将听诊器信息纳入 MTL 架构时，呼吸音分类和诊断性能均能显著提高。 

---
# Importance Analysis for Dynamic Control of Balancing Parameter in a Simple Knowledge Distillation Setting 

**Title (ZH)**: 简单知识精简设置中平衡参数动态控制的重要性分析 

**Authors**: Seongmin Kim, Kwanho Kim, Minseung Kim, Kanghyun Jo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06270)  

**Abstract**: Although deep learning models owe their remarkable success to deep and complex architectures, this very complexity typically comes at the expense of real-time performance. To address this issue, a variety of model compression techniques have been proposed, among which knowledge distillation (KD) stands out for its strong empirical performance. The KD contains two concurrent processes: (i) matching the outputs of a large, pre-trained teacher network and a lightweight student network, and (ii) training the student to solve its designated downstream task. The associated loss functions are termed the distillation loss and the downsteam-task loss, respectively. Numerous prior studies report that KD is most effective when the influence of the distillation loss outweighs that of the downstream-task loss. The influence(or importance) is typically regulated by a balancing parameter. This paper provides a mathematical rationale showing that in a simple KD setting when the loss is decreasing, the balancing parameter should be dynamically adjusted 

**Abstract (ZH)**: 尽管深度学习模型因其深度和复杂架构取得了显著的成功，但这种复杂性通常会牺牲实时性能。为了解决这一问题，已经提出了多种模型压缩技术，其中知识蒸馏（KD）因其强大的实证性能而脱颖而出。KD包含两个并发过程：(i) 匹配预先训练的大规模教师网络和轻量级学生网络的输出，(ii) 训练学生解决其指定的下游任务。相关的损失函数分别称为蒸馏损失和下游任务损失。许多先前的研究报告称，当蒸馏损失的影响超过下游任务损失时，KD的效果最佳。影响（或重要性）通常通过一个平衡参数来调节。本文提供了一个数学理由，证明在简单KD设置中，当损失在减少时，平衡参数应动态调整。 

---
# Cluster-Aware Multi-Round Update for Wireless Federated Learning in Heterogeneous Environments 

**Title (ZH)**: 面向异构环境的集群感知多轮更新无线联邦学习 

**Authors**: Pengcheng Sun, Erwu Liu, Wei Ni, Kanglei Yu, Rui Wang, Abbas Jamalipour  

**Link**: [PDF](https://arxiv.org/pdf/2505.06268)  

**Abstract**: The aggregation efficiency and accuracy of wireless Federated Learning (FL) are significantly affected by resource constraints, especially in heterogeneous environments where devices exhibit distinct data distributions and communication capabilities. This paper proposes a clustering strategy that leverages prior knowledge similarity to group devices with similar data and communication characteristics, mitigating performance degradation from heterogeneity. On this basis, a novel Cluster- Aware Multi-round Update (CAMU) strategy is proposed, which treats clusters as the basic units and adjusts the local update frequency based on the clustered contribution threshold, effectively reducing update bias and enhancing aggregation accuracy. The theoretical convergence of the CAMU strategy is rigorously validated. Meanwhile, based on the convergence upper bound, the local update frequency and transmission power of each cluster are jointly optimized to achieve an optimal balance between computation and communication resources under constrained conditions, significantly improving the convergence efficiency of FL. Experimental results demonstrate that the proposed method effectively improves the model performance of FL in heterogeneous environments and achieves a better balance between communication cost and computational load under limited resources. 

**Abstract (ZH)**: 无线联邦学习中资源约束对聚合效率和准确性的显著影响，尤其是在异构环境中设备展现出不同的数据分布和通信能力。本文提出了一种利用先验知识相似性进行聚类的策略，以分组具有相似数据和通信特性的设备，从而减轻异构性带来的性能下降。在此基础上，提出了一种新的聚类意识多轮更新（CAMU）策略，将簇作为基本单位，并根据聚类贡献阈值调整本地更新频率，有效减少了更新偏差并提高了聚合准确性。CAMU策略的理论收敛性得到了严格验证。同时，基于收敛上界，联合优化每个簇的本地更新频率和传输功率，在受限条件下实现计算和通信资源的最优平衡，显著提高了联邦学习的收敛效率。实验结果表明，所提出的方法有效地提高了联邦学习在异构环境中的模型性能，并在有限资源下实现了较好的通信成本和计算负载平衡。 

---
# Knowledge Guided Encoder-Decoder Framework Integrating Multiple Physical Models for Agricultural Ecosystem Modeling 

**Title (ZH)**: 知识引导的编码-解码框架整合多种物理模型进行农业生态系统建模 

**Authors**: Qi Cheng, Licheng Liu, Zhang Yao, Hong Mu, Shiyuan Luo, Zhenong Jin, Yiqun Xie, Xiaowei Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.06266)  

**Abstract**: Agricultural monitoring is critical for ensuring food security, maintaining sustainable farming practices, informing policies on mitigating food shortage, and managing greenhouse gas emissions. Traditional process-based physical models are often designed and implemented for specific situations, and their parameters could also be highly uncertain. In contrast, data-driven models often use black-box structures and does not explicitly model the inter-dependence between different ecological variables. As a result, they require extensive training data and lack generalizability to different tasks with data distribution shifts and inconsistent observed variables. To address the need for more universal models, we propose a knowledge-guided encoder-decoder model, which can predict key crop variables by leveraging knowledge of underlying processes from multiple physical models. The proposed method also integrates a language model to process complex and inconsistent inputs and also utilizes it to implement a model selection mechanism for selectively combining the knowledge from different physical models. Our evaluations on predicting carbon and nitrogen fluxes for multiple sites demonstrate the effectiveness and robustness of the proposed model under various scenarios. 

**Abstract (ZH)**: 农业监测对于确保粮食安全、维持可持续农业实践、制定缓解粮食短缺的政策以及管理温室气体排放至关重要。传统的基于过程的物理模型通常为特定情况设计和实施，且其参数可能高度不确定。相比之下，数据驱动模型往往采用黑盒结构，不明确建模不同生态变量之间的相互依赖关系。因此，它们需要大量的训练数据，并且在数据分布转移和观测变量不一致的情况下缺乏普适性。为应对需要更加通用的模型的需求，我们提出了一种知识引导的编码解码模型，该模型通过从多个物理模型中汲取基础过程的知识来预测关键作物变量。该方法还集成了一种语言模型来处理复杂且不一致的输入，并利用其实施一种模型选择机制，以选择性地结合不同物理模型的知识。我们在多个站点预测碳和氮通量的评估表明，该模型在各种场景下具有有效性与鲁棒性。 

---
# Prediction of Delirium Risk in Mild Cognitive Impairment Using Time-Series data, Machine Learning and Comorbidity Patterns -- A Retrospective Study 

**Title (ZH)**: 使用时间序列数据、机器学习和共病模式预测轻度认知 impairment 患者的谵妄风险：一项回顾性研究 

**Authors**: Santhakumar Ramamoorthy, Priya Rani, James Mahon, Glenn Mathews, Shaun Cloherty, Mahdi Babaei  

**Link**: [PDF](https://arxiv.org/pdf/2505.06264)  

**Abstract**: Delirium represents a significant clinical concern characterized by high morbidity and mortality rates, particularly in patients with mild cognitive impairment (MCI). This study investigates the associated risk factors for delirium by analyzing the comorbidity patterns relevant to MCI and developing a longitudinal predictive model leveraging machine learning methodologies. A retrospective analysis utilizing the MIMIC-IV v2.2 database was performed to evaluate comorbid conditions, survival probabilities, and predictive modeling outcomes. The examination of comorbidity patterns identified distinct risk profiles for the MCI population. Kaplan-Meier survival analysis demonstrated that individuals with MCI exhibit markedly reduced survival probabilities when developing delirium compared to their non-MCI counterparts, underscoring the heightened vulnerability within this cohort. For predictive modeling, a Long Short-Term Memory (LSTM) ML network was implemented utilizing time-series data, demographic variables, Charlson Comorbidity Index (CCI) scores, and an array of comorbid conditions. The model demonstrated robust predictive capabilities with an AUROC of 0.93 and an AUPRC of 0.92. This study underscores the critical role of comorbidities in evaluating delirium risk and highlights the efficacy of time-series predictive modeling in pinpointing patients at elevated risk for delirium development. 

**Abstract (ZH)**: 轻微认知障碍患者谵妄的相关风险因素研究：基于acomorbidity模式的纵向预测模型 

---
# Modeling supply chain compliance response strategies based on AI synthetic data with structural path regression: A Simulation Study of EU 2027 Mandatory Labor Regulations 

**Title (ZH)**: 基于AI合成数据的结构路径回归建模：欧盟2027强制劳动规定下的供应链合规响应策略仿真研究 

**Authors**: Wei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2505.06261)  

**Abstract**: In the context of the new mandatory labor compliance in the European Union (EU), which will be implemented in 2027, supply chain enterprises face stringent working hour management requirements and compliance risks. In order to scientifically predict the enterprises' coping behaviors and performance outcomes under the policy impact, this paper constructs a methodological framework that integrates the AI synthetic data generation mechanism and structural path regression modeling to simulate the enterprises' strategic transition paths under the new regulations. In terms of research methodology, this paper adopts high-quality simulation data generated based on Monte Carlo mechanism and NIST synthetic data standards to construct a structural path analysis model that includes multiple linear regression, logistic regression, mediation effect and moderating effect. The variable system covers 14 indicators such as enterprise working hours, compliance investment, response speed, automation level, policy dependence, etc. The variable set with explanatory power is screened out through exploratory data analysis (EDA) and VIF multicollinearity elimination. The findings show that compliance investment has a significant positive impact on firm survival and its effect is transmitted through the mediating path of the level of intelligence; meanwhile, firms' dependence on the EU market significantly moderates the strength of this mediating effect. It is concluded that AI synthetic data combined with structural path modeling provides an effective tool for high-intensity regulatory simulation, which can provide a quantitative basis for corporate strategic response, policy design and AI-assisted decision-making in the pre-prediction stage lacking real scenario data. Keywords: AI synthetic data, structural path regression modeling, compliance response strategy, EU 2027 mandatory labor regulation 

**Abstract (ZH)**: 欧盟2027年强制劳动合规政策背景下基于AI合成数据及结构路径回归模型的企业应对策略研究 

---
# Fair Clustering with Clusterlets 

**Title (ZH)**: 公平聚类与簇集/grouplet 

**Authors**: Mattia Setzu, Riccardo Guidotti  

**Link**: [PDF](https://arxiv.org/pdf/2505.06259)  

**Abstract**: Given their widespread usage in the real world, the fairness of clustering methods has become of major interest. Theoretical results on fair clustering show that fairness enjoys transitivity: given a set of small and fair clusters, a trivial centroid-based clustering algorithm yields a fair clustering. Unfortunately, discovering a suitable starting clustering can be computationally expensive, rather complex or arbitrary.
In this paper, we propose a set of simple \emph{clusterlet}-based fuzzy clustering algorithms that match single-class clusters, optimizing fair clustering. Matching leverages clusterlet distance, optimizing for classic clustering objectives, while also regularizing for fairness. Empirical results show that simple matching strategies are able to achieve high fairness, and that appropriate parameter tuning allows to achieve high cohesion and low overlap. 

**Abstract (ZH)**: 基于集群块的公平聚类算法研究 

---
# ABE: A Unified Framework for Robust and Faithful Attribution-Based Explainability 

**Title (ZH)**: ABE：一种统一的基于归因的解释性鲁棒且忠实的框架 

**Authors**: Zhiyu Zhu, Jiayu Zhang, Zhibo Jin, Fang Chen, Jianlong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06258)  

**Abstract**: Attribution algorithms are essential for enhancing the interpretability and trustworthiness of deep learning models by identifying key features driving model decisions. Existing frameworks, such as InterpretDL and OmniXAI, integrate multiple attribution methods but suffer from scalability limitations, high coupling, theoretical constraints, and lack of user-friendly implementations, hindering neural network transparency and interoperability. To address these challenges, we propose Attribution-Based Explainability (ABE), a unified framework that formalizes Fundamental Attribution Methods and integrates state-of-the-art attribution algorithms while ensuring compliance with attribution axioms. ABE enables researchers to develop novel attribution techniques and enhances interpretability through four customizable modules: Robustness, Interpretability, Validation, and Data & Model. This framework provides a scalable, extensible foundation for advancing attribution-based explainability and fostering transparent AI systems. Our code is available at: this https URL. 

**Abstract (ZH)**: 基于 attribution 的可解释性框架 (ABE)：提升深度学习模型的可解释性和可信度 

---
# SpectrumFM: A Foundation Model for Intelligent Spectrum Management 

**Title (ZH)**: SpectrumFM：智能频谱管理的基座模型 

**Authors**: Fuhui Zhou, Chunyu Liu, Hao Zhang, Wei Wu, Qihui Wu, Derrick Wing Kwan Ng, Tony Q. S. Quek, Chan-Byoung Chae  

**Link**: [PDF](https://arxiv.org/pdf/2505.06256)  

**Abstract**: Intelligent spectrum management is crucial for improving spectrum efficiency and achieving secure utilization of spectrum resources. However, existing intelligent spectrum management methods, typically based on small-scale models, suffer from notable limitations in recognition accuracy, convergence speed, and generalization, particularly in the complex and dynamic spectrum environments. To address these challenges, this paper proposes a novel spectrum foundation model, termed SpectrumFM, establishing a new paradigm for spectrum management. SpectrumFM features an innovative encoder architecture that synergistically exploits the convolutional neural networks and the multi-head self-attention mechanisms to enhance feature extraction and enable robust representation learning. The model is pre-trained via two novel self-supervised learning tasks, namely masked reconstruction and next-slot signal prediction, which leverage large-scale in-phase and quadrature (IQ) data to achieve comprehensive and transferable spectrum representations. Furthermore, a parameter-efficient fine-tuning strategy is proposed to enable SpectrumFM to adapt to various downstream spectrum management tasks, including automatic modulation classification (AMC), wireless technology classification (WTC), spectrum sensing (SS), and anomaly detection (AD). Extensive experiments demonstrate that SpectrumFM achieves superior performance in terms of accuracy, robustness, adaptability, few-shot learning efficiency, and convergence speed, consistently outperforming conventional methods across multiple benchmarks. Specifically, SpectrumFM improves AMC accuracy by up to 12.1% and WTC accuracy by 9.3%, achieves an area under the curve (AUC) of 0.97 in SS at -4 dB signal-to-noise ratio (SNR), and enhances AD performance by over 10%. 

**Abstract (ZH)**: 智能频谱管理对于提高频谱效率和实现频谱资源的安全利用至关重要。现有基于小规模模型的智能频谱管理方法在识别准确性、收敛速度和泛化能力方面存在明显不足，特别是在复杂的动态频谱环境中。为应对这些挑战，本文提出了一种新型频谱基础模型——SpectrumFM，建立了一种新的频谱管理范式。SpectrumFM 特设了一种创新的编码器架构，结合了卷积神经网络和多头自注意力机制，以增强特征提取并实现稳健的表示学习。该模型通过两个新颖的自我监督学习任务——掩码重建和下一槽信号预测——进行了预训练，利用大规模同相和正交（IQ）数据实现全面和可迁移的频谱表示。此外，提出了一种参数高效微调策略，使SpectrumFM能够适应各种下游频谱管理任务，包括自动调制分类（AMC）、无线技术分类（WTC）、频谱感知（SS）和异常检测（AD）。广泛实验表明，SpectrumFM 在准确率、鲁棒性、适应性、少样本学习效率和收敛速度方面表现优异，全面优于传统方法。具体而言，SpectrumFM 将AMC准确性提高了12.1%，WTC准确性提高了9.3%，在-4 dB信噪比（SNR）下SS的曲线下面积（AUC）达到了0.97，且异常检测性能提升了超过10%。 

---
# DeltaDPD: Exploiting Dynamic Temporal Sparsity in Recurrent Neural Networks for Energy-Efficient Wideband Digital Predistortion 

**Title (ZH)**: DeltaDPD：利用循环神经网络中的动态时间稀疏性实现宽带数字预-distortion的能效提升 

**Authors**: Yizhuo Wu, Yi Zhu, Kun Qian, Qinyu Chen, Anding Zhu, John Gajadharsing, Leo C. N. de Vreede, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.06250)  

**Abstract**: Digital Predistortion (DPD) is a popular technique to enhance signal quality in wideband RF power amplifiers (PAs). With increasing bandwidth and data rates, DPD faces significant energy consumption challenges during deployment, contrasting with its efficiency goals. State-of-the-art DPD models rely on recurrent neural networks (RNN), whose computational complexity hinders system efficiency. This paper introduces DeltaDPD, exploring the dynamic temporal sparsity of input signals and neuronal hidden states in RNNs for energy-efficient DPD, reducing arithmetic operations and memory accesses while preserving satisfactory linearization performance. Applying a TM3.1a 200MHz-BW 256-QAM OFDM signal to a 3.5 GHz GaN Doherty RF PA, DeltaDPD achieves -50.03 dBc in Adjacent Channel Power Ratio (ACPR), -37.22 dB in Normalized Mean Square Error (NMSE) and -38.52 dBc in Error Vector Magnitude (EVM) with 52% temporal sparsity, leading to a 1.8X reduction in estimated inference power. The DeltaDPD code will be released after formal publication at this https URL. 

**Abstract (ZH)**: 基于输入信号和RNN神经隐藏状态的动态时域稀疏性实现高效数字预失真 

---
# United States Road Accident Prediction using Random Forest Predictor 

**Title (ZH)**: 美国道路事故预测基于随机森林预测模型 

**Authors**: Dominic Parosh Yamarthi, Haripriya Raman, Shamsad Parvin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06246)  

**Abstract**: Road accidents significantly threaten public safety and require in-depth analysis for effective prevention and mitigation strategies. This paper focuses on predicting accidents through the examination of a comprehensive traffic dataset covering 49 states in the United States. The dataset integrates information from diverse sources, including transportation departments, law enforcement, and traffic sensors. This paper specifically emphasizes predicting the number of accidents, utilizing advanced machine learning models such as regression analysis and time series analysis. The inclusion of various factors, ranging from environmental conditions to human behavior and infrastructure, ensures a holistic understanding of the dynamics influencing road safety. Temporal and spatial analysis further allows for the identification of trends, seasonal variations, and high-risk areas. The implications of this research extend to proactive decision-making for policymakers and transportation authorities. By providing accurate predictions and quantifiable insights into expected accident rates under different conditions, the paper aims to empower authorities to allocate resources efficiently and implement targeted interventions. The goal is to contribute to the development of informed policies and interventions that enhance road safety, creating a safer environment for all road users. Keywords: Machine Learning, Random Forest, Accident Prediction, AutoML, LSTM. 

**Abstract (ZH)**: 道路事故显著威胁公共安全，需要进行深入分析以制定有效的预防和缓解策略。本文通过分析覆盖美国49个州的综合交通数据集，重点关注事故预测。该数据集整合了来自交通部门、执法机构和交通传感器的多种信息。本文特别强调利用先进的机器学习模型，如回归分析和时间序列分析来预测事故数量。包括从环境条件到人类行为和基础设施的各种因素，确保对影响道路安全的动力学有一个全面的理解。通过时空分析，进一步识别出趋势、季节性变化和高风险区域。本文的研究结果对政策制定者和交通管理部门的前瞻性决策具有重要意义。通过提供准确的预测和不同条件下预期事故率的量化洞察，本文旨在帮助当局有效分配资源并实施有针对性的干预措施。目标是促进制定基于数据的政策和干预措施，提高道路安全，为所有道路使用者创造更安全的环境。关键词：机器学习，随机森林，事故预测，AutoML，LSTM。 

---
# Low-Complexity CNN-Based Classification of Electroneurographic Signals 

**Title (ZH)**: 基于CNN的低复杂度Electroneurographic信号分类 

**Authors**: Arek Berc Gokdag, Silvia Mura, Antonio Coviello, Michele Zhu, Maurizio Magarini, Umberto Spagnolini  

**Link**: [PDF](https://arxiv.org/pdf/2505.06241)  

**Abstract**: Peripheral nerve interfaces (PNIs) facilitate neural recording and stimulation for treating nerve injuries, but real-time classification of electroneurographic (ENG) signals remains challenging due to constraints on complexity and latency, particularly in implantable devices. This study introduces MobilESCAPE-Net, a lightweight architecture that reduces computational cost while maintaining and slightly improving classification performance. Compared to the state-of-the-art ESCAPE-Net, MobilESCAPE-Net achieves comparable accuracy and F1-score with significantly lower complexity, reducing trainable parameters by 99.9\% and floating point operations per second by 92.47\%, enabling faster inference and real-time processing. Its efficiency makes it well-suited for low-complexity ENG signal classification in resource-constrained environments such as implantable devices. 

**Abstract (ZH)**: 基于移动设备的轻量级ENG信号分类网络（MobilESCAPE-Net）：一种降低复杂性和保持分类性能的方法 

---
