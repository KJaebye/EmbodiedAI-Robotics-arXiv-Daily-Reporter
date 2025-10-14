# IntersectioNDE: Learning Complex Urban Traffic Dynamics based on Interaction Decoupling Strategy 

**Title (ZH)**: IntersectionDDE：基于交互解耦策略学习复杂城市交通动力学 

**Authors**: Enli Lin, Ziyuan Yang, Qiujing Lu, Jianming Hu, Shuo Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.11534)  

**Abstract**: Realistic traffic simulation is critical for ensuring the safety and reliability of autonomous vehicles (AVs), especially in complex and diverse urban traffic environments. However, existing data-driven simulators face two key challenges: a limited focus on modeling dense, heterogeneous interactions at urban intersections - which are prevalent, crucial, and practically significant in countries like China, featuring diverse agents including motorized vehicles (MVs), non-motorized vehicles (NMVs), and pedestrians - and the inherent difficulty in robustly learning high-dimensional joint distributions for such high-density scenes, often leading to mode collapse and long-term simulation instability. We introduce City Crossings Dataset (CiCross), a large-scale dataset collected from a real-world urban intersection, uniquely capturing dense, heterogeneous multi-agent interactions, particularly with a substantial proportion of MVs, NMVs and pedestrians. Based on this dataset, we propose IntersectioNDE (Intersection Naturalistic Driving Environment), a data-driven simulator tailored for complex urban intersection scenarios. Its core component is the Interaction Decoupling Strategy (IDS), a training paradigm that learns compositional dynamics from agent subsets, enabling the marginal-to-joint simulation. Integrated into a scene-aware Transformer network with specialized training techniques, IDS significantly enhances simulation robustness and long-term stability for modeling heterogeneous interactions. Experiments on CiCross show that IntersectioNDE outperforms baseline methods in simulation fidelity, stability, and its ability to replicate complex, distribution-level urban traffic dynamics. 

**Abstract (ZH)**: 现实istic交通模拟对于确保自主车辆（AVs）的安全性和可靠性至关重要，特别是在复杂的多种城市交通环境中。然而，现有的数据驱动模拟器面临两大关键挑战：对城市交叉口密集且异质性交互的建模关注不足——在像中国这样的国家尤为重要，这些国家的交通参与者包括机动车（MVs）、非机动车（NMVs）和行人，且此类交互频繁且实际意义重大——以及在高密度场景中稳健地学习高维联合分布的固有困难，这通常导致模式崩溃和长期模拟不稳定性。我们介绍了城市交叉口数据集（CiCross），该数据集从真实的城区交叉口收集而来，特别捕捉了密集且异质性多代理交互，特别包括大量机动车、非机动车和行人的交互。基于此数据集，我们提出了IntersectioNDE（复杂城市交叉口自然驾驶环境），一种专为复杂城市交叉口场景设计的数据驱动模拟器。其核心组件是交互解藕策略（IDS），一种训练范式，能够从代理子集学习组合动力学，从而实现边缘到联合的模拟。通过集成到场景感知的Transformer网络并结合专门的训练技术，IDS显著提高了模拟的稳健性和长期稳定性，以建模异质性交互。在CiCross上的实验表明，IntersectioNDE在模拟保真度、稳定性和再现复杂的城市交通动态方面优于基线方法。 

---
# A Faster and More Reliable Middleware for Autonomous Driving Systems 

**Title (ZH)**: 一种更快更可靠的自主驾驶系统中间件 

**Authors**: Yuankai He, Hanlin Chen, Weisong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.11448)  

**Abstract**: Ensuring safety in high-speed autonomous vehicles requires rapid control loops and tightly bounded delays from perception to actuation. Many open-source autonomy systems rely on ROS 2 middleware; when multiple sensor and control nodes share one compute unit, ROS 2 and its DDS transports add significant (de)serialization, copying, and discovery overheads, shrinking the available time budget. We present Sensor-in-Memory (SIM), a shared-memory transport designed for intra-host pipelines in autonomous vehicles. SIM keeps sensor data in native memory layouts (e.g., cv::Mat, PCL), uses lock-free bounded double buffers that overwrite old data to prioritize freshness, and integrates into ROS 2 nodes with four lines of code. Unlike traditional middleware, SIM operates beside ROS 2 and is optimized for applications where data freshness and minimal latency outweigh guaranteed completeness. SIM provides sequence numbers, a writer heartbeat, and optional checksums to ensure ordering, liveness, and basic integrity. On an NVIDIA Jetson Orin Nano, SIM reduces data-transport latency by up to 98% compared to ROS 2 zero-copy transports such as FastRTPS and Zenoh, lowers mean latency by about 95%, and narrows 95th/99th-percentile tail latencies by around 96%. In tests on a production-ready Level 4 vehicle running this http URL, SIM increased localization frequency from 7.5 Hz to 9.5 Hz. Applied across all latency-critical modules, SIM cut average perception-to-decision latency from 521.91 ms to 290.26 ms, reducing emergency braking distance at 40 mph (64 km/h) on dry concrete by 13.6 ft (4.14 m). 

**Abstract (ZH)**: 确保自动驾驶车辆的安全需要快速的控制循环和从感知到动作的紧密时间限制。许多开源自主系统依赖于ROS 2中间件；当多个传感器和控制节点共用一个计算单元时，ROS 2及其DDS传输会增加显著的序列化、复制和发现开销，压缩可用的时间预算。我们提出了Sensor-in-Memory（SIM），这是一种针对自主车辆内部管道的共享内存传输。SIM将传感器数据保持在原始内存布局中（例如，cv::Mat、PCL），使用无锁的双缓冲区进行数据覆盖以优先考虑新鲜度，并通过四行代码集成到ROS 2节点中。与传统的中间件不同，SIM在ROS 2旁边运行，并针对那些以数据新鲜度和最小延迟超过完全性保证的应用程序进行优化。SIM提供了序列号、写入者心跳和可选的校验和以确保排序、活性和基本完整性。在NVIDIA Jetson Orin Nano上，与ROS 2零拷贝传输FastRTPS和Zenoh相比，SIM将数据传输延迟最多减少了98%，降低了平均延迟约95%，并将95/99百分位尾部延迟减少了约96%。在针对生产就绪的L4级车辆进行的测试中（请参阅此链接），SIM将定位频率从每秒7.5次提高到9.5次。在所有关键延迟模块中应用SIM，将感知到决策的平均延迟从521.91毫秒减少到290.26毫秒，将40英里/小时（64公里/小时）干混凝土上的紧急制动距离减少了13.6英尺（4.14米）。 

---
# A Modular AIoT Framework for Low-Latency Real-Time Robotic Teleoperation in Smart Cities 

**Title (ZH)**: 面向智慧城市的一种模块化AIoT实时遥控机器人框架 

**Authors**: Shih-Chieh Sun, Yun-Cheng Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2510.11421)  

**Abstract**: This paper presents an AI-driven IoT robotic teleoperation system designed for real-time remote manipulation and intelligent visual monitoring, tailored for smart city applications. The architecture integrates a Flutter-based cross-platform mobile interface with MQTT-based control signaling and WebRTC video streaming via the LiveKit framework. A YOLOv11-nano model is deployed for lightweight object detection, enabling real-time perception with annotated visual overlays delivered to the user interface. Control commands are transmitted via MQTT to an ESP8266-based actuator node, which coordinates multi-axis robotic arm motion through an Arduino Mega2560 controller. The backend infrastructure is hosted on DigitalOcean, ensuring scalable cloud orchestration and stable global communication. Latency evaluations conducted under both local and international VPN scenarios (including Hong Kong, Japan, and Belgium) demonstrate actuator response times as low as 0.2 seconds and total video latency under 1.2 seconds, even across high-latency networks. This low-latency dual-protocol design ensures responsive closed-loop interaction and robust performance in distributed environments. Unlike conventional teleoperation platforms, the proposed system emphasizes modular deployment, real-time AI sensing, and adaptable communication strategies, making it well-suited for smart city scenarios such as remote infrastructure inspection, public equipment servicing, and urban automation. Future enhancements will focus on edge-device deployment, adaptive routing, and integration with city-scale IoT networks to enhance resilience and scalability. 

**Abstract (ZH)**: 基于AI驱动的物联网远程机器人操作系统：面向智慧城市的应用实现与智能视觉监控 

---
# Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling 

**Title (ZH)**: 基于流匹配的自驱动规划与高级交互行为建模 

**Authors**: Tianyi Tan, Yinan Zheng, Ruiming Liang, Zexu Wang, Kexin Zheng, Jinliang Zheng, Jianxiong Li, Xianyuan Zhan, Jingjing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11083)  

**Abstract**: Modeling interactive driving behaviors in complex scenarios remains a fundamental challenge for autonomous driving planning. Learning-based approaches attempt to address this challenge with advanced generative models, removing the dependency on over-engineered architectures for representation fusion. However, brute-force implementation by simply stacking transformer blocks lacks a dedicated mechanism for modeling interactive behaviors that are common in real driving scenarios. The scarcity of interactive driving data further exacerbates this problem, leaving conventional imitation learning methods ill-equipped to capture high-value interactive behaviors. We propose Flow Planner, which tackles these problems through coordinated innovations in data modeling, model architecture, and learning scheme. Specifically, we first introduce fine-grained trajectory tokenization, which decomposes the trajectory into overlapping segments to decrease the complexity of whole trajectory modeling. With a sophisticatedly designed architecture, we achieve efficient temporal and spatial fusion of planning and scene information, to better capture interactive behaviors. In addition, the framework incorporates flow matching with classifier-free guidance for multi-modal behavior generation, which dynamically reweights agent interactions during inference to maintain coherent response strategies, providing a critical boost for interactive scenario understanding. Experimental results on the large-scale nuPlan dataset and challenging interactive interPlan dataset demonstrate that Flow Planner achieves state-of-the-art performance among learning-based approaches while effectively modeling interactive behaviors in complex driving scenarios. 

**Abstract (ZH)**: 基于模型的交互驾驶行为在复杂场景中的建模依然是自主驾驶规划中的一个基本挑战。基于学习的方法试图通过先进的生成模型解决这一挑战，去除对过度工程化架构的依赖以实现表示融合。然而，简单地堆叠变压器块的 brute-force 实现缺乏专门机制来建模在实际驾驶场景中常见的交互行为。交互驾驶数据的缺乏进一步加剧了这一问题，使得传统的模仿学习方法难以捕捉高价值的交互行为。我们提出了 Flow Planner，通过数据建模、模型架构和学习方案的协调创新来解决这些问题。具体而言，我们首先引入细粒度轨迹标记化，将轨迹分解为重叠段以降低整个轨迹建模的复杂性。借助精心设计的架构，我们实现了规划和场景信息的高效时空融合，以更好地捕捉交互行为。此外，该框架整合了流匹配与无分类器引导的多模态行为生成，动态重新加权推理期间的代理人交互，以保持一致的响应策略，为交互场景理解提供了关键的提升。在大规模 nuPlan 数据集和具有挑战性的交互 interPlan 数据集上的实验结果显示，Flow Planner 在基于学习的方法中实现了最先进的性能，同时在复杂驾驶场景中有效地建模了交互行为。 

---
# An Adaptive Transition Framework for Game-Theoretic Based Takeover 

**Title (ZH)**: 基于博弈论的收购框架的自适应过渡模型 

**Authors**: Dikshant Shehmar, Matthew E. Taylor, Ehsan Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10893)  

**Abstract**: The transition of control from autonomous systems to human drivers is critical in automated driving systems, particularly due to the out-of-the-loop (OOTL) circumstances that reduce driver readiness and increase reaction times. Existing takeover strategies are based on fixed time-based transitions, which fail to account for real-time driver performance variations. This paper proposes an adaptive transition strategy that dynamically adjusts the control authority based on both the time and tracking ability of the driver trajectory. Shared control is modeled as a cooperative differential game, where control authority is modulated through time-varying objective functions instead of blending control torques directly. To ensure a more natural takeover, a driver-specific state-tracking matrix is introduced, allowing the transition to align with individual control preferences. Multiple transition strategies are evaluated using a cumulative trajectory error metric. Human-in-the-loop control scenarios of the standardized ISO lane change maneuvers demonstrate that adaptive transitions reduce trajectory deviations and driver control effort compared to conventional strategies. Experiments also confirm that continuously adjusting control authority based on real-time deviations enhances vehicle stability while reducing driver effort during takeover. 

**Abstract (ZH)**: 自主系统到人工驾驶的控制过渡在自动化驾驶系统中至关重要，特别是在脱环（OOTL）情况下，这降低了驾驶员的准备状态并增加了反应时间。现有的接管策略基于固定的时间过渡，未能考虑到驾驶员实时性能的变化。本文提出了一种适应性过渡策略，该策略基于时间和驾驶员轨迹追踪能力动态调整控制权。共轭控制被建模为合作微分博弈，其中控制权通过时间变化的目标函数来调节，而不是直接混合控制力矩。为了实现更自然的接管，引入了驾驶员特定的状态追踪矩阵，使过渡能够与个体控制偏好相一致。通过使用累计轨迹误差度量评估了多种过渡策略。标准ISO车道变换试验的人机在环控制场景表明，适应性过渡减少了轨迹偏差并降低了驾驶员控制努力，相比于传统策略。实验还证实，基于实时偏差连续调整控制权提高了车辆稳定性并减少了接管过程中的驾驶员努力。 

---
# Hierarchical Planning for Long-Horizon Multi-Target Tracking Under Target Motion Uncertainty 

**Title (ZH)**: 长时段多目标跟踪下的层级规划方法及其目标运动不确定性处理 

**Authors**: Junbin Yuan, Brady Moon, Muqing Cao, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2510.10421)  

**Abstract**: Achieving persistent tracking of multiple dynamic targets over a large spatial area poses significant challenges for a single-robot system with constrained sensing capabilities. As the robot moves to track different targets, the ones outside the field of view accumulate uncertainty, making them progressively harder to track. An effective path planning algorithm must manage uncertainty over a long horizon and account for the risk of permanently losing track of targets that remain unseen for too long. However, most existing approaches rely on short planning horizons and assume small, bounded environments, resulting in poor tracking performance and target loss in large-scale scenarios. In this paper, we present a hierarchical planner for tracking multiple moving targets with an aerial vehicle. To address the challenge of tracking non-static targets, our method incorporates motion models and uncertainty propagation during path execution, allowing for more informed decision-making. We decompose the multi-target tracking task into sub-tasks of single target search and detection, and our proposed pipeline consists a novel low-level coverage planner that enables searching for a target in an evolving belief area, and an estimation method to assess the likelihood of success for each sub-task, making it possible to convert the active target tracking task to a Markov decision process (MDP) that we solve with a tree-based algorithm to determine the sequence of sub-tasks. We validate our approach in simulation, demonstrating its effectiveness compared to existing planners for active target tracking tasks, and our proposed planner outperforms existing approaches, achieving a reduction of 11-70% in final uncertainty across different environments. 

**Abstract (ZH)**: 实现单一机器人系统在大区域范围内持续跟踪多个动态目标面临着严重挑战，特别是在受限的感知能力条件下。随着机器人追踪不同目标而移动，位于视野之外的目标会积累不确定性，使其越来越难以追踪。有效的路径规划算法必须在长时间范围内管理不确定性，并且要考虑到长时间未被观察到的目标可能会永久丢失的风险。然而，大多数现有方法依赖于短时间的规划范围，并假设小型受限环境，导致在大规模场景中跟踪性能不佳，目标丢失率高。在本文中，我们提出了一种分层规划方法来使用航空器追踪多个移动目标。为了应对追踪非静态目标的挑战，我们的方法在路径执行过程中整合了运动模型和不确定性传播，以实现更有信息性的决策。我们将多目标追踪任务分解为单目标搜索和检测的子任务，并提出了一种新颖的低层覆盖规划方法以在不断变化的信念区域内搜索目标，以及一种评估每个子任务成功概率的方法，使得可以将主动目标追踪任务转化为马尔可夫决策过程（MDP），我们使用基于树的算法解决此过程以确定子任务序列。我们通过仿真验证了我们的方法，展示了与现有主动目标追踪规划器相比的有效性，并且我们的提出的方法在不同环境中将最终不确定性降低了11-70%。 

---
# Beyond ADE and FDE: A Comprehensive Evaluation Framework for Safety-Critical Prediction in Multi-Agent Autonomous Driving Scenarios 

**Title (ZH)**: 超越ADE和FDE：多Agent自主驾驶场景中安全关键预测的全面评估框架 

**Authors**: Feifei Liu, Haozhe Wang, Zejun Wei, Qirong Lu, Yiyang Wen, Xiaoyu Tang, Jingyan Jiang, Zhijian He  

**Link**: [PDF](https://arxiv.org/pdf/2510.10086)  

**Abstract**: Current evaluation methods for autonomous driving prediction models rely heavily on simplistic metrics such as Average Displacement Error (ADE) and Final Displacement Error (FDE). While these metrics offer basic performance assessments, they fail to capture the nuanced behavior of prediction modules under complex, interactive, and safety-critical driving scenarios. For instance, existing benchmarks do not distinguish the influence of nearby versus distant agents, nor systematically test model robustness across varying multi-agent interactions. This paper addresses this critical gap by proposing a novel testing framework that evaluates prediction performance under diverse scene structures, saying, map context, agent density and spatial distribution. Through extensive empirical analysis, we quantify the differential impact of agent proximity on target trajectory prediction and identify scenario-specific failure cases that are not exposed by traditional metrics. Our findings highlight key vulnerabilities in current state-of-the-art prediction models and demonstrate the importance of scenario-aware evaluation. The proposed framework lays the groundwork for rigorous, safety-driven prediction validation, contributing significantly to the identification of failure-prone corner cases and the development of robust, certifiable prediction systems for autonomous vehicles. 

**Abstract (ZH)**: 当前自动驾驶预测模型的评估方法主要依赖于简单的指标，如平均位移误差（ADE）和最终位移误差（FDE）。虽然这些指标提供了基本的性能评估，但它们未能捕捉到在复杂、交互性和安全性关键的驾驶场景中预测模块的微妙行为。现有的基准测试无法区分附近和远处代理的影响，也未系统地测试模型在不同多代理交互场景下的鲁棒性。本文通过提出一种新的测试框架来填补这一关键缺口，该框架在多样化的场景结构下，包括地图上下文、代理密度和空间分布等方面评估预测性能。通过广泛的实证分析，我们量化了代理临近性对目标轨迹预测的影响差异，并识别出传统指标无法揭示的场景特定失败案例。我们的研究结果突出了当前最先进的预测模型中的关键漏洞，强调了场景感知评估的重要性。提出的框架为严格的、以安全为导向的预测验证奠定了基础，显著促进了故障多发边缘案例的识别以及鲁棒、可认证的自动驾驶预测系统的开发。 

---
# Ionospheric and Plasmaspheric Delay Characterization for Lunar Terrestrial GNSS Receivers with Global Core Plasma Model 

**Title (ZH)**: 基于全球核心等离子体模型的月地GNSS接收机电离层和等离子鞘泡延迟特性研究 

**Authors**: Keidai Iiyama, Grace Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.10059)  

**Abstract**: Recent advancements in lunar positioning, navigation, and timing (PNT) have demonstrated that terrestrial GNSS signals, including weak sidelobe transmissions, can be exploited for lunar spacecraft positioning and timing. While GNSS-based navigation at the Moon has been validated recently, unmodeled ionospheric and plasmaspheric delays remain a significant error source, particularly given the unique signal geometry and extended propagation paths. This paper characterizes these delays using the Global Core Plasma Model (GCPM) and a custom low-cost ray-tracing algorithm that iteratively solves for bent signal paths. We simulate first-, second-, and third-order group delays, as well as excess path length from ray bending, for GNSS signals received at both lunar orbit and the lunar south pole under varying solar and geomagnetic conditions. Results show that mean group delays are typically on the order of 1 m, but can exceed 100 m for low-altitude ray paths during high solar activity, while bending delays are generally smaller but non-negligible for low-altitude ray paths. We also quantify the influence of signal frequency, geomagnetic $K_p$ index, and solar R12 index. These findings inform the design of robust positioning and timing algorithms that utilize terrestrial GNSS signals. 

**Abstract (ZH)**: 最近在月球定位、导航和定时（PNT）方面的进展表明，包括弱旁瓣传输在内的GNSS信号可以用于月球航天器的定位和定时。虽然基于GNSS的月球导航最近已被验证，但未建模的离子层和等离子层延迟仍然是一个重要的误差源，尤其是在独特的信号几何结构和延长的传播路径条件下。本文使用全球核心等离子体模型（GCPM）和一个自定义的低成本射线追踪算法来迭代求解弯曲信号路径，以表征这些延迟。我们模拟了GNSS信号在月球轨道和月球南极接收时的一阶、二阶和三阶群延迟，以及由于射线弯曲引起的多余路径长度，这些模拟在不同的太阳和地磁条件下进行。结果显示，平均群延迟通常在1米左右，但在高太阳活动期间，低高度射线路径的群延迟可以超过100米，而弯曲延迟对于低高度射线路径通常较小但不容忽视。我们还量化了信号频率、地磁$K_p$指数和太阳R12指数的影响。这些发现为利用地球GNSS信号设计稳健的定位和定时算法提供了依据。 

---
# Smooth Spatiotemporal Tube Synthesis for Prescribed-Time Reach-Avoid-Stay Control 

**Title (ZH)**: 指定时间到达避免停留控制的平滑时空管合成 

**Authors**: Siddhartha Upadhyay, Ratnangshu Das, Pushpak Jagtap  

**Link**: [PDF](https://arxiv.org/pdf/2510.11583)  

**Abstract**: In this work, we address the issue of controller synthesis for a control-affine nonlinear system to meet prescribed time reach-avoid-stay specifications. Our goal is to improve upon previous methods based on spatiotemporal tubes (STTs) by eliminating the need for circumvent functions, which often lead to abrupt tube modifications and high control effort. We propose an adaptive framework that constructs smooth STTs around static unsafe sets, enabling continuous avoidance while guiding the system toward the target within the prescribed time. A closed-form, approximation-free control law is derived to ensure the system trajectory remains within the tube and satisfies the RAS task. The effectiveness of the proposed approach is demonstrated through a case study, showing a significant reduction in control effort compared to prior methods. 

**Abstract (ZH)**: 基于时空管的自适应设计方法以实现控制预测非线性系统的时间到达避免停留任务 

---
# Bhasha-Rupantarika: Algorithm-Hardware Co-design approach for Multilingual Neural Machine Translation 

**Title (ZH)**: Bhasha-Rupantarika：多语言神经机器翻译的算法-硬件协同设计方法 

**Authors**: Mukul Lokhande, Tanushree Dewangan, Mohd Sharik Mansoori, Tejas Chaudhari, Akarsh J., Damayanti Lokhande, Adam Teman, Santosh Kumar Vishvakarma  

**Link**: [PDF](https://arxiv.org/pdf/2510.10676)  

**Abstract**: This paper introduces Bhasha-Rupantarika, a light and efficient multilingual translation system tailored through algorithm-hardware codesign for resource-limited settings. The method investigates model deployment at sub-octet precision levels (FP8, INT8, INT4, and FP4), with experimental results indicating a 4.1x reduction in model size (FP4) and a 4.2x speedup in inference speed, which correlates with an increased throughput of 66 tokens/s (improvement by 4.8x). This underscores the importance of ultra-low precision quantization for real-time deployment in IoT devices using FPGA accelerators, achieving performance on par with expectations. Our evaluation covers bidirectional translation between Indian and international languages, showcasing its adaptability in low-resource linguistic contexts. The FPGA deployment demonstrated a 1.96x reduction in LUTs and a 1.65x decrease in FFs, resulting in a 2.2x enhancement in throughput compared to OPU and a 4.6x enhancement compared to HPTA. Overall, the evaluation provides a viable solution based on quantisation-aware translation along with hardware efficiency suitable for deployable multilingual AI systems. The entire codes [this https URL] and dataset for reproducibility are publicly available, facilitating rapid integration and further development by researchers. 

**Abstract (ZH)**: 本文介绍了一种轻量高效的大规模多语言翻译系统Bhasha-Rupantarika，该系统通过算法-硬件协同设计针对资源受限的环境进行定制。该方法研究了亚字节精度模型部署（FP8、INT8、INT4和FP4）的效果，实验结果表明FP4精度模型大小减少了4.1倍，并将推理速度加快了4.2倍，同时吞吐量提高了4.8倍（66tokens/s）。这强调了在使用FPGA加速器的物联网设备中进行实时部署时，超低精度量化的重要性，以达到预期性能。我们的评估涵盖了印度语和国际语言之间的双向翻译，展示了其在低资源语言环境中的可适应性。FPGA部署结果显示LUT减少了1.96倍，FF减少了1.65倍，与OPU相比吞吐量提高了2.2倍，与HPTA相比提高了4.6倍。总体而言，评估提供了一种基于量化感知翻译和硬件效率的可部署多语言AI系统的可行解决方案。整个代码和数据集可在以下链接获取，以供研究人员复制和进一步开发。 

---
# Operand Quant: A Single-Agent Architecture for Autonomous Machine Learning Engineering 

**Title (ZH)**: 操作量纲：自管理机器学习工程的单代理架构 

**Authors**: Arjun Sahney, Ram Gorthi, Cezary Łastowski, Javier Vega  

**Link**: [PDF](https://arxiv.org/pdf/2510.11694)  

**Abstract**: We present Operand Quant, a single-agent, IDE-based architecture for autonomous machine learning engineering (MLE). Operand Quant departs from conventional multi-agent orchestration frameworks by consolidating all MLE lifecycle stages -- exploration, modeling, experimentation, and deployment -- within a single, context-aware agent. On the MLE-Benchmark (2025), Operand Quant achieved a new state-of-the-art (SOTA) result, with an overall medal rate of 0.3956 +/- 0.0565 across 75 problems -- the highest recorded performance among all evaluated systems to date. The architecture demonstrates that a linear, non-blocking agent, operating autonomously within a controlled IDE environment, can outperform multi-agent and orchestrated systems under identical constraints. 

**Abstract (ZH)**: operand quant：一种基于IDE的单代理自主机器学习工程架构 

---
# Explainability, risk modeling, and segmentation based customer churn analytics for personalized retention in e-commerce 

**Title (ZH)**: 基于解释性、风险建模和细分的个性化客户流失分析以实现电子商务中的留存 

**Authors**: Sanjula De Alwis, Indrajith Ekanayake  

**Link**: [PDF](https://arxiv.org/pdf/2510.11604)  

**Abstract**: In online retail, customer acquisition typically incurs higher costs than customer retention, motivating firms to invest in churn analytics. However, many contemporary churn models operate as opaque black boxes, limiting insight into the determinants of attrition, the timing of retention opportunities, and the identification of high-risk customer segments. Accordingly, the emphasis should shift from prediction alone to the design of personalized retention strategies grounded in interpretable evidence. This study advances a three-component framework that integrates explainable AI to quantify feature contributions, survival analysis to model time-to-event churn risk, and RFM profiling to segment customers by transactional behaviour. In combination, these methods enable the attribution of churn drivers, estimation of intervention windows, and prioritization of segments for targeted actions, thereby supporting strategies that reduce attrition and strengthen customer loyalty. 

**Abstract (ZH)**: 在线零售中，获取客户的成本通常高于保留客户的成本，促使企业投资流失分析。然而，许多现代流失模型作为不透明的黑箱运作，限制了对客户流失驱动因素、保留机会的时间以及高风险客户群体识别的洞察。因此，重心应从单一的预测转移到基于可解释证据设计个性化的保留策略。本研究提出了一种包含三个组成部分的框架，该框架结合可解释人工智能来量化特征贡献，事件时间生存分析来建模时间到事件的流失风险，以及RFM细分来根据交易行为划分客户。这些方法的结合能够归因于流失驱动因素、估计干预窗口，并优先考虑需要针对性行动的客户群体，从而支持减少流失和增强客户忠诚度的策略。 

---
# Reproducibility: The New Frontier in AI Governance 

**Title (ZH)**: reproducibility: AI治理的新前沿 

**Authors**: Israel Mason-Williams, Gabryel Mason-Williams  

**Link**: [PDF](https://arxiv.org/pdf/2510.11595)  

**Abstract**: AI policymakers are responsible for delivering effective governance mechanisms that can provide safe, aligned and trustworthy AI development. However, the information environment offered to policymakers is characterised by an unnecessarily low Signal-To-Noise Ratio, favouring regulatory capture and creating deep uncertainty and divides on which risks should be prioritised from a governance perspective. We posit that the current publication speeds in AI combined with the lack of strong scientific standards, via weak reproducibility protocols, effectively erodes the power of policymakers to enact meaningful policy and governance protocols. Our paper outlines how AI research could adopt stricter reproducibility guidelines to assist governance endeavours and improve consensus on the AI risk landscape. We evaluate the forthcoming reproducibility crisis within AI research through the lens of crises in other scientific domains; providing a commentary on how adopting preregistration, increased statistical power and negative result publication reproducibility protocols can enable effective AI governance. While we maintain that AI governance must be reactive due to AI's significant societal implications we argue that policymakers and governments must consider reproducibility protocols as a core tool in the governance arsenal and demand higher standards for AI research. Code to replicate data and figures: this https URL 

**Abstract (ZH)**: AI决策者有责任提供有效的治理机制，确保AI的安全、对齐和可信发展。然而，提供给决策者的信息环境特征是信号与噪声比过低，有利于监管俘获，并在从治理角度应优先考虑哪些风险方面制造深刻的不确定性与分歧。我们提出，当前AI领域的发表速度与缺乏严格的科学标准（通过弱重复性协议体现）有效削弱了决策者制定有意义政策和治理协议的能力。本文概述了AI研究如何采用更严格的重复性指南以辅助治理努力并改善对AI风险领域的共识。我们通过考察其他科学领域中的危机来评价AI研究将面临的重复性危机，并提供一种观点，即通过采用前瞻性注册、增加统计功效和负面结果发布等重复性协议，可以实现有效的AI治理。尽管我们坚持认为，鉴于AI的重大社会影响，AI治理必须具有反应性，但我们认为决策者和政府必须将重复性协议视为治理工具的核心，并要求更高的AI研究标准。重复数据和图表的复制代码：this https URL。 

---
# Unifying Deductive and Abductive Reasoning in Knowledge Graphs with Masked Diffusion Model 

**Title (ZH)**: 在知识图中通过掩码扩散模型统一演绎推理和溯因推理 

**Authors**: Yisen Gao, Jiaxin Bai, Yi Huang, Xingcheng Fu, Qingyun Sun, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.11462)  

**Abstract**: Deductive and abductive reasoning are two critical paradigms for analyzing knowledge graphs, enabling applications from financial query answering to scientific discovery. Deductive reasoning on knowledge graphs usually involves retrieving entities that satisfy a complex logical query, while abductive reasoning generates plausible logical hypotheses from observations. Despite their clear synergistic potential, where deduction can validate hypotheses and abduction can uncover deeper logical patterns, existing methods address them in isolation. To bridge this gap, we propose DARK, a unified framework for Deductive and Abductive Reasoning in Knowledge graphs. As a masked diffusion model capable of capturing the bidirectional relationship between queries and conclusions, DARK has two key innovations. First, to better leverage deduction for hypothesis refinement during abductive reasoning, we introduce a self-reflective denoising process that iteratively generates and validates candidate hypotheses against the observed conclusion. Second, to discover richer logical associations, we propose a logic-exploration reinforcement learning approach that simultaneously masks queries and conclusions, enabling the model to explore novel reasoning compositions. Extensive experiments on multiple benchmark knowledge graphs show that DARK achieves state-of-the-art performance on both deductive and abductive reasoning tasks, demonstrating the significant benefits of our unified approach. 

**Abstract (ZH)**: 演绎和归纳推理是分析知识图谱的两种关键范式， enabling 从金融查询回答到科学发现等应用。演绎推理通常涉及检索满足复杂逻辑查询的实体，而归纳推理则从观察中生成合理的逻辑假设。尽管它们在协同作用方面具有明显的潜力，其中演绎可验证假设，而归纳可揭示更深层的逻辑模式，现有方法仍分别处理它们。为了弥合这一差距，我们提出了DARK，一种知识图谱中演绎和归纳推理的统一框架。作为能够捕捉查询与结论之间双向关系的掩码扩散模型，DARK具有两个关键创新。首先，为了更好地利用演绎在归纳推理中的假设细化，我们引入了一种自反去噪过程，该过程迭代地生成和验证候选假设，使其与观察到的结论一致。其次，为了发现更丰富的逻辑关联，我们提出了一种逻辑探索强化学习方法，该方法同时掩码查询和结论，使模型能够探索新的推理组合。在多个基准知识图谱上的广泛实验表明，DARK在演绎和归纳推理任务上均达到了最先进的性能，证明了我们统一方法的重要优势。 

---
# AI-Driven anemia diagnosis: A review of advanced models and techniques 

**Title (ZH)**: AI驱动的贫血诊断：先进模型与技术的综述 

**Authors**: Abdullah Al Mahmud, Prangon Chowdhury, Mohammed Borhan Uddin, Khaled Eabne Delowar, Tausifur Rahman Talha, Bijoy Dewanjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.11380)  

**Abstract**: Anemia, a condition marked by insufficient levels of red blood cells or hemoglobin, remains a widespread health issue affecting millions of individuals globally. Accurate and timely diagnosis is essential for effective management and treatment of anemia. In recent years, there has been a growing interest in the use of artificial intelligence techniques, i.e., machine learning (ML) and deep learning (DL) for the detection, classification, and diagnosis of anemia. This paper provides a systematic review of the recent advancements in this field, with a focus on various models applied to anemia detection. The review also compares these models based on several performance metrics, including accuracy, sensitivity, specificity, and precision. By analyzing these metrics, the paper evaluates the strengths and limitation of discussed models in detecting and classifying anemia, emphasizing the importance of addressing these factors to improve diagnostic accuracy. 

**Abstract (ZH)**: 贫血是一种由红细胞或血红蛋白水平不足引起的情况，仍然是影响全球数百万人的广泛健康问题。准确及时的诊断对于贫血的有效管理和治疗至关重要。近年来，人们越来越关注使用人工智能技术，即机器学习（ML）和深度学习（DL）来检测、分类和诊断贫血。本文提供了一个对该领域近期进展的系统性综述，重点关注应用于贫血检测的各种模型，并根据准确性、敏感性、特异性和精确度等性能指标对这些模型进行了比较。通过分析这些指标，本文评估了讨论的模型在检测和分类贫血方面的优缺点，强调了在这些方面进行改进以提高诊断准确性的重要性。 

---
# AI Alignment Strategies from a Risk Perspective: Independent Safety Mechanisms or Shared Failures? 

**Title (ZH)**: 从风险角度看的AI对齐策略：独立的安全机制还是共有的失败？ 

**Authors**: Leonard Dung, Florian Mai  

**Link**: [PDF](https://arxiv.org/pdf/2510.11235)  

**Abstract**: AI alignment research aims to develop techniques to ensure that AI systems do not cause harm. However, every alignment technique has failure modes, which are conditions in which there is a non-negligible chance that the technique fails to provide safety. As a strategy for risk mitigation, the AI safety community has increasingly adopted a defense-in-depth framework: Conceding that there is no single technique which guarantees safety, defense-in-depth consists in having multiple redundant protections against safety failure, such that safety can be maintained even if some protections fail. However, the success of defense-in-depth depends on how (un)correlated failure modes are across alignment techniques. For example, if all techniques had the exact same failure modes, the defense-in-depth approach would provide no additional protection at all. In this paper, we analyze 7 representative alignment techniques and 7 failure modes to understand the extent to which they overlap. We then discuss our results' implications for understanding the current level of risk and how to prioritize AI alignment research in the future. 

**Abstract (ZH)**: AI对齐研究旨在开发技术以确保AI系统不会造成危害。然而，每种对齐技术都有失效模式，即技术不能提供安全性的非忽视概率条件。为了减少风险，AI安全社区越来越多地采用纵深防御框架：承认没有单一技术能够确保安全，纵深防御包括多种冗余保护措施，即使某些保护措施失效，也能维持安全性。然而，纵深防御的成功依赖于对齐技术之间失效模式的相关性。例如，如果所有技术都具有完全相同类型的失效模式，那么纵深防御方法将提供完全没有额外保护的效果。本文分析了7种代表性的对齐技术和7种失效模式，以了解它们重叠的程度。我们随后讨论了这些结果对当前风险水平理解及其对未来AI对齐研究优先级的影响。 

---
# $How^{2}$: How to learn from procedural How-to questions 

**Title (ZH)**: $How^2$: 如何学习来自过程性如何做问题 

**Authors**: Gautier Dagan, Frank Keller, Alex Lascarides  

**Link**: [PDF](https://arxiv.org/pdf/2510.11144)  

**Abstract**: An agent facing a planning problem can use answers to how-to questions to reduce uncertainty and fill knowledge gaps, helping it solve both current and future tasks. However, their open ended nature, where valid answers to "How do I X?" range from executable actions to high-level descriptions of X's sub-goals, makes them challenging for AI agents to ask, and for AI experts to answer, in ways that support efficient planning. We introduce $How^{2}$, a memory agent framework that enables agents to ask how-to questions, store the answers, and reuse them for lifelong learning in interactive environments. We evaluate our approach in Plancraft, a Minecraft crafting environment, where agents must complete an assembly task by manipulating inventory items. Using teacher models that answer at varying levels of abstraction, from executable action sequences to high-level subgoal descriptions, we show that lifelong learning agents benefit most from answers that are abstracted and decoupled from the current state. $How^{2}$ offers a way for LLM-based agents to improve their planning capabilities over time by asking questions in interactive environments. 

**Abstract (ZH)**: 一个面对规划问题的智能体可以通过提出如何执行任务的问题来减少不确定性并填补知识空白，从而帮助其解决当前和未来的任务。然而，这些问题的开放性质使得它们对于AI智能体提出和AI专家回答以支持高效规划变得具有挑战性。我们引入了$How^{2}$，一个内存智能体框架，使智能体能够提出如何执行任务的问题，存储答案，并在交互环境中进行终身学习。我们在Plancraft（一个Minecraft制作物品环境）中评估了我们的方法，智能体必须通过操作库存物品来完成装配任务。通过使用不同抽象级别的教师模型来回答这些问题，从可执行的动作序列到高层次的子目标描述，我们展示了抽象化和与当前状态解耦的答案对终身学习智能体最为有益。$How^{2}$为基于LLM的智能体提供了一种在交互环境中通过提问来逐步提高其规划能力的方法。 

---
# Spec-Driven AI for Science: The ARIA Framework for Automated and Reproducible Data Analysis 

**Title (ZH)**: 基于规格的AI科学：ARIA框架下的自动化和可重复数据分析 

**Authors**: Chuke Chen, Biao Luo, Nan Li, Boxiang Wang, Hang Yang, Jing Guo, Ming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11143)  

**Abstract**: The rapid expansion of scientific data has widened the gap between analytical capability and research intent. Existing AI-based analysis tools, ranging from AutoML frameworks to agentic research assistants, either favor automation over transparency or depend on manual scripting that hinders scalability and reproducibility. We present ARIA (Automated Research Intelligence Assistant), a spec-driven, human-in-the-loop framework for automated and interpretable data analysis. ARIA integrates six interoperable layers, namely Command, Context, Code, Data, Orchestration, and AI Module, within a document-centric workflow that unifies human reasoning and machine execution. Through natural-language specifications, researchers define analytical goals while ARIA autonomously generates executable code, validates computations, and produces transparent documentation. Beyond achieving high predictive accuracy, ARIA can rapidly identify optimal feature sets and select suitable models, minimizing redundant tuning and repetitive experimentation. In the Boston Housing case, ARIA discovered 25 key features and determined XGBoost as the best performing model (R square = 0.93) with minimal overfitting. Evaluations across heterogeneous domains demonstrate ARIA's strong performance, interpretability, and efficiency compared with state-of-the-art systems. By combining AI for research and AI for science principles within a spec-driven architecture, ARIA establishes a new paradigm for transparent, collaborative, and reproducible scientific discovery. 

**Abstract (ZH)**: 科学数据的迅速扩展加剧了分析能力和研究意图之间的差距。现有的基于AI的分析工具，从AutoML框架到代理型研究助手，要么偏向自动化而忽视透明度，要么依赖手工脚本，这阻碍了可扩展性和可重复性。我们提出了ARIAS（Automated Research Intelligence Assistant），一种基于规格、包含人类在环的框架，用于自动化和可解释的数据分析。ARIAS将六个可互操作的层——命令、上下文、代码、数据、编排和AI模块——集成在一个文档为中心的工作流中，统一了人类推理和机器执行。通过自然语言规格说明，研究人员定义分析目标，而ARIAS自主生成可执行代码、验证计算并生成透明文档。除了实现高度预测准确性外，ARIAS还可以快速识别最佳特征集并选择合适的模型，最小化冗余调优和重复实验。在波士顿住房案例中，ARIAS发现了25个关键特征，并确定XGBoost为最佳性能模型（R² = 0.93），且过度拟合最小。横跨不同领域的评估显示，ARIAS在可解释性和效率方面优于现有最先进的系统。通过在一个基于规格的架构中结合研究中的AI和科学中的AI原则，ARIAS确立了一种新的透明、协作和可重复的科学发现范式。 

---
# Improving AI Efficiency in Data Centres by Power Dynamic Response 

**Title (ZH)**: 通过电力动态响应提高数据中心中人工智能的效率 

**Authors**: Andrea Marinoni, Sai Shivareddy, Pietro Lio', Weisi Lin, Erik Cambria, Clare Grey  

**Link**: [PDF](https://arxiv.org/pdf/2510.11119)  

**Abstract**: The steady growth of artificial intelligence (AI) has accelerated in the recent years, facilitated by the development of sophisticated models such as large language models and foundation models. Ensuring robust and reliable power infrastructures is fundamental to take advantage of the full potential of AI. However, AI data centres are extremely hungry for power, putting the problem of their power management in the spotlight, especially with respect to their impact on environment and sustainable development. In this work, we investigate the capacity and limits of solutions based on an innovative approach for the power management of AI data centres, i.e., making part of the input power as dynamic as the power used for data-computing functions. The performance of passive and active devices are quantified and compared in terms of computational gain, energy efficiency, reduction of capital expenditure, and management costs by analysing power trends from multiple data platforms worldwide. This strategy, which identifies a paradigm shift in the AI data centre power management, has the potential to strongly improve the sustainability of AI hyperscalers, enhancing their footprint on environmental, financial, and societal fields. 

**Abstract (ZH)**: 近年来，人工智能（AI）的稳步增长得到了复杂模型如大型语言模型和基础模型的发展加速。确保强大的可靠电力基础设施是充分利用AI潜力的关键。然而，AI数据中心极为依赖电力，使得其电力管理问题备受关注，特别是在其对环境和可持续发展的影响方面。在本研究中，我们调查了一种创新方法在AI数据中心电力管理中的能力和局限性，即部分输入电力动态化，与用于数据计算功能的电力使用情况相匹配。通过分析全球多个数据平台的电力趋势，定量比较被动和主动设备在计算增益、能量效率、降低资本支出和管理成本方面的性能。这种策略标志着AI数据中心电力管理范式的转变，有潜力显著提高AI超大规模数据中心的可持续性，增强其在环境、财务和社会领域的影响力。 

---
# Modeling AI-Driven Production and Competitiveness A Multi-Agent Economic Simulation of China and the United States 

**Title (ZH)**: 基于多agent经济模拟的AI驱动生产与中国和美国的竞争模型 

**Authors**: Yuxinyue Qian, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11085)  

**Abstract**: With the rapid development of artificial intelligence (AI) technology, socio-economic systems are entering a new stage of "human-AI co-creation." Building upon a previously established multi-level intelligent agent economic model, this paper conducts simulation-based comparisons of macroeconomic output evolution in China and the United States under different mechanisms-AI collaboration, network effects, and AI autonomous production. The results show that: (1) when AI functions as an independent productive entity, the overall growth rate of social output far exceeds that of traditional human-labor-based models; (2) China demonstrates clear potential for acceleration in both the expansion of intelligent agent populations and the pace of technological catch-up, offering the possibility of achieving technological convergence or even partial surpassing. This study provides a systematic, model-based analytical framework for understanding AI-driven production system transformation and shifts in international competitiveness, as well as quantitative insights for relevant policy formulation. 

**Abstract (ZH)**: 随着人工智能（AI）技术的迅速发展，社会经济系统进入了“人机共创”的新阶段。在建立在先前多层级智能代理经济模型基础上，本文通过基于仿真的比较研究了中美在不同机制下（AI协作、网络效应和AI自主生产）宏观经济产出演化的差异。研究结果表明：（1）当AI作为独立生产实体发挥作用时，社会产出的整体增长速率远超基于传统人力模型的增长速率；（2）中国在智能代理群体扩展和技术创新追赶方面显示出明显的加速潜力，有可能实现技术收敛甚至部分超越。本研究提供了基于模型的系统性分析框架，用于理解AI驱动的生产系统转型及国际竞争力变化，并为相关政策制定提供了量化参考。 

---
# Argumentation-Based Explainability for Legal AI: Comparative and Regulatory Perspectives 

**Title (ZH)**: 基于论证的法律AI解释性：对比和监管视角 

**Authors**: Andrada Iulia Prajescu, Roberto Confalonieri  

**Link**: [PDF](https://arxiv.org/pdf/2510.11079)  

**Abstract**: Artificial Intelligence (AI) systems are increasingly deployed in legal contexts, where their opacity raises significant challenges for fairness, accountability, and trust. The so-called ``black box problem'' undermines the legitimacy of automated decision-making, as affected individuals often lack access to meaningful explanations. In response, the field of Explainable AI (XAI) has proposed a variety of methods to enhance transparency, ranging from example-based and rule-based techniques to hybrid and argumentation-based approaches. This paper promotes computational models of arguments and their role in providing legally relevant explanations, with particular attention to their alignment with emerging regulatory frameworks such as the EU General Data Protection Regulation (GDPR) and the Artificial Intelligence Act (AIA). We analyze the strengths and limitations of different explanation strategies, evaluate their applicability to legal reasoning, and highlight how argumentation frameworks -- by capturing the defeasible, contestable, and value-sensitive nature of law -- offer a particularly robust foundation for explainable legal AI. Finally, we identify open challenges and research directions, including bias mitigation, empirical validation in judicial settings, and compliance with evolving ethical and legal standards, arguing that computational argumentation is best positioned to meet both technical and normative requirements of transparency in the law domain. 

**Abstract (ZH)**: 人工智能（AI）系统在法律情境中的应用日益增多，其不透明性对公平性、问责制和信任提出了重大挑战。“黑箱问题”削弱了自动化决策的合法性，受影响的个体往往无法获得有意义的解释。为此，可解释人工智能（XAI）领域提出了多种增强透明度的方法，从基于例证和基于规则的技术到混合和基于论辩的方法。本文促进计算论证模型及其在提供法律相关解释中的作用，特别关注其与欧盟通用数据保护条例（GDPR）和人工智能法案（AIA）等新兴监管框架的契合。我们分析了不同解释策略的优势与局限性，评估了它们在法律推理中的适用性，并强调论辩框架通过捕捉法律的可驳倒性、可争议性和价值敏感性，为可解释的法律人工智能提供了特别稳健的基础。最后，我们指出了开放挑战和研究方向，包括偏见缓解、在司法环境中进行实证验证以及遵守不断演变的伦理和法律标准，认为计算论辩最有可能满足法律领域透明性的技术和规范要求。 

---
# FBS Model-based Maintenance Record Accumulation for Failure-Cause Inference in Manufacturing Systems 

**Title (ZH)**: 基于FBS模型的故障原因推断制造系统维修记录累积 

**Authors**: Takuma Fujiu, Sho Okazaki, Kohei Kaminishi, Yuji Nakata, Shota Hamamoto, Kenshin Yokose, Tatsunori Hara, Yasushi Umeda, Jun Ota  

**Link**: [PDF](https://arxiv.org/pdf/2510.11003)  

**Abstract**: In manufacturing systems, identifying the causes of failures is crucial for maintaining and improving production efficiency. In knowledge-based failure-cause inference, it is important that the knowledge base (1) explicitly structures knowledge about the target system and about failures, and (2) contains sufficiently long causal chains of failures. In this study, we constructed Diagnostic Knowledge Ontology and proposed a Function-Behavior-Structure (FBS) model-based maintenance-record accumulation method based on it. Failure-cause inference using the maintenance records accumulated by the proposed method showed better agreement with the set of candidate causes enumerated by experts, especially in difficult cases where the number of related cases is small and the vocabulary used differs. In the future, it will be necessary to develop inference methods tailored to these maintenance records, build a user interface, and carry out validation on larger and more diverse systems. Additionally, this approach leverages the understanding and knowledge of the target in the design phase to support knowledge accumulation and problem solving during the maintenance phase, and it is expected to become a foundation for knowledge sharing across the entire engineering chain in the future. 

**Abstract (ZH)**: 在制造系统中，识别故障原因对于维护和提高生产效率至关重要。基于知识的故障原因推理中，知识库需要（1）明确结构化目标系统和故障的知识，并且（2）包含足够长的故障因果链。在本研究中，我们构建了诊断知识本体，并提出了基于该本体的函数-行为-结构（FBS）模型驱动的维护记录积累方法。采用所提出方法积累的维护记录进行的故障原因推理与专家列出的候选原因有更好的一致性，特别是在相关案例较少且使用的词汇不同的困难情况下。未来需要开发针对这些维护记录的推理方法，构建用户界面，并在更大和更多样化的系统上进行验证。此外，该方法在设计阶段利用对目标的理解和知识来支持维护阶段的知识积累和问题解决，并有望成为跨整个工程链的知识共享的基础。 

---
# Revisiting Model Interpolation for Efficient Reasoning 

**Title (ZH)**: 重新审视模型内插以实现高效推理 

**Authors**: Taiqiang Wu, Runming Yang, Tao Liu, Jiahao Wang, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10977)  

**Abstract**: Model merging, typically on Instruct and Thinking models, has shown remarkable performance for efficient reasoning. In this paper, we systematically revisit the simplest merging method that interpolates two weights directly. Particularly, we observe that model interpolation follows a three-stage evolutionary paradigm with distinct behaviors on the reasoning trajectory. These dynamics provide a principled guide for navigating the performance-cost trade-off. Empirical results demonstrate that a strategically interpolated model surprisingly surpasses sophisticated model merging baselines on both efficiency and effectiveness. We further validate our findings with extensive ablation studies on model layers, modules, and decoding strategies. Ultimately, this work demystifies model interpolation and offers a practical framework for crafting models with precisely targeted reasoning capabilities. Code is available at \href{this https URL}{Github}. 

**Abstract (ZH)**: 模型插值，尤其是在指令模型和思考模型上的应用，展现了高效的推理性能。本文系统回顾了直接插值两种权重的最简单合并方法，并观察到模型插值遵循一个具有三个阶段演化范式的动态过程，这些动态为性能与成本权衡提供了原则性的指导。实验证明，战略性插值的模型在效率和效果上竟然超过了复杂的模型合并基准。我们还通过广泛的消融研究进一步验证了这些发现，涉及模型层、模块和解码策略。最终，本工作揭开模型插值的神秘面纱，并提供了一个实用的框架来构建具有精确推理能力的模型。代码可在Github获得。 

---
# Scalable and Explainable Enterprise Knowledge Discovery Using Graph-Centric Hybrid Retrieval 

**Title (ZH)**: 基于图中心混合检索的企业可扩展性和可解释性知识发现 

**Authors**: Nilima Rao, Jagriti Srivastava, Pradeep Kumar Sharma, Hritvik Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2510.10942)  

**Abstract**: Modern enterprises manage vast knowledge distributed across heterogeneous systems such as Jira, Git repositories, Confluence, and wikis. Conventional retrieval methods based on keyword search or static embeddings often fail to answer complex queries that require contextual reasoning and multi-hop inference across artifacts. We present a modular hybrid retrieval framework for adaptive enterprise information access that integrates Knowledge Base Language-Augmented Models (KBLam), DeepGraph representations, and embedding-driven semantic search. The framework builds a unified knowledge graph from parsed repositories including code, pull requests, and commit histories, enabling semantic similarity search, structural inference, and multi-hop reasoning. Query analysis dynamically determines the optimal retrieval strategy, supporting both structured and unstructured data sources through independent or fused processing. An interactive interface provides graph visualizations, subgraph exploration, and context-aware query routing to generate concise and explainable answers. Experiments on large-scale Git repositories show that the unified reasoning layer improves answer relevance by up to 80 percent compared with standalone GPT-based retrieval pipelines. By combining graph construction, hybrid reasoning, and interactive visualization, the proposed framework offers a scalable, explainable, and user-centric foundation for intelligent knowledge assistants in enterprise environments. 

**Abstract (ZH)**: 现代企业使用Jira、Git仓库、Confluence和维基等异构系统管理广泛的知识。基于关键词搜索或静态嵌入的传统检索方法往往无法回答需要上下文推理和多跳推理的复杂查询。我们提出了一种模块化混合检索框架，用于适应性的企业信息访问，该框架结合了知识库语言增强模型（KBLam）、深度图表示和嵌入驱动的语义搜索。该框架从解析的仓库（包括代码、拉取请求和提交历史）中构建统一的知识图谱，实现语义相似搜索、结构推理和多跳推理。查询分析动态确定最佳检索策略，支持通过独立或融合处理结构化和非结构化数据源。交互式界面提供图可视化、子图探索和上下文感知查询路由，生成简洁且可解释的答案。 experiment on大规模Git仓库表明，统一推理层相比于基于GPT的独立检索流水线，答案的相关性提高了80%。通过结合图构建、混合推理和交互式可视化，所提框架为企业环境中智能知识助理提供了一个可扩展、可解释且用户中心的基础。 

---
# DRIFT: Decompose, Retrieve, Illustrate, then Formalize Theorems 

**Title (ZH)**: DRIFT: 分解、检索、示证、然后形式化定理 

**Authors**: Meiru Zhang, Philipp Borchert, Milan Gritta, Gerasimos Lampouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.10815)  

**Abstract**: Automating the formalization of mathematical statements for theorem proving remains a major challenge for Large Language Models (LLMs). LLMs struggle to identify and utilize the prerequisite mathematical knowledge and its corresponding formal representation in languages like Lean. Current retrieval-augmented autoformalization methods query external libraries using the informal statement directly, but overlook a fundamental limitation: informal mathematical statements are often complex and offer limited context on the underlying math concepts. To address this, we introduce DRIFT, a novel framework that enables LLMs to decompose informal mathematical statements into smaller, more tractable ''sub-components''. This facilitates targeted retrieval of premises from mathematical libraries such as Mathlib. Additionally, DRIFT retrieves illustrative theorems to help models use premises more effectively in formalization tasks. We evaluate DRIFT across diverse benchmarks (ProofNet, ConNF, and MiniF2F-test) and find that it consistently improves premise retrieval, nearly doubling the F1 score compared to the DPR baseline on ProofNet. Notably, DRIFT demonstrates strong performance on the out-of-distribution ConNF benchmark, with BEq+@10 improvements of 37.14% and 42.25% using GPT-4.1 and DeepSeek-V3.1, respectively. Our analysis shows that retrieval effectiveness in mathematical autoformalization depends heavily on model-specific knowledge boundaries, highlighting the need for adaptive retrieval strategies aligned with each model's capabilities. 

**Abstract (ZH)**: 自动化数学陈述的形式化以实现定理证明仍然是大型语言模型的主要挑战。当前的检索增强自动形式化方法直接使用非形式化的陈述查询外部库，但忽视了一个基本限制：非形式化的数学陈述往往复杂且对底层数学概念的上下文限制较少。为解决这一问题，我们提出了DRIFT，一种新型框架，使大型语言模型能够将非形式化的数学陈述分解为更小、更易于处理的“子组件”。这有助于有针对性地从如Mathlib的数学库中检索前提。此外，DRIFT检索示例定理以帮助模型更有效地在形式化任务中使用前提。我们在ProofNet、ConNF和MiniF2F-test等多样化的基准测试上评估DRIFT，并发现它一致地改善了前提检索效果，在ProofNet基准测试上，与DPR基线相比，F1分数提高了近一倍。值得注意的是，DRIFT在分布外的ConNF基准测试上表现出色，分别使用GPT-4.1和DeepSeek-V3.1时，BEq+@10改进了37.14%和42.25%。我们的分析表明，在数学自动形式化中的检索效果高度依赖于模型特定的知识边界，突显了需要与每个模型的能力相适应的检索策略的重要性。 

---
# Extended Triangular Method: A Generalized Algorithm for Contradiction Separation Based Automated Deduction 

**Title (ZH)**: 扩展三角形方法：基于自动化推理的矛盾分离广义算法 

**Authors**: Yang Xu, Shuwei Chen, Jun Liu, Feng Cao, Xingxing He  

**Link**: [PDF](https://arxiv.org/pdf/2510.10701)  

**Abstract**: Automated deduction lies at the core of Artificial Intelligence (AI), underpinning theorem proving, formal verification, and logical reasoning. Despite decades of progress, reconciling deductive completeness with computational efficiency remains an enduring challenge. Traditional reasoning calculi, grounded in binary resolution, restrict inference to pairwise clause interactions and thereby limit deductive synergy among multiple clauses. The Contradiction Separation Extension (CSE) framework, introduced in 2018, proposed a dynamic multi-clause reasoning theory that redefined logical inference as a process of contradiction separation rather than sequential resolution. While that work established the theoretical foundation, its algorithmic realization remained unformalized and unpublished. This work presents the Extended Triangular Method (ETM), a generalized contradiction-construction algorithm that formalizes and extends the internal mechanisms of contradiction separation. The ETM unifies multiple contradiction-building strategies, including the earlier Standard Extension method, within a triangular geometric framework that supports flexible clause interaction and dynamic synergy. ETM serves as the algorithmic core of several high-performance theorem provers, CSE, CSE-E, CSI-E, and CSI-Enig, whose competitive results in standard first-order benchmarks (TPTP problem sets and CASC 2018-2015) empirically validate the effectiveness and generality of the proposed approach. By bridging theoretical abstraction and operational implementation, ETM advances the contradiction separation paradigm into a generalized, scalable, and practically competitive model for automated reasoning, offering new directions for future research in logical inference and theorem proving. 

**Abstract (ZH)**: 自动推理是人工智能的核心，支撑着定理证明、形式验证和逻辑推理。尽管历经数十年的发展，如何在保证完全性的同时提高计算效率依然是一项持久的挑战。传统的基于二元归结的推理算法限制了多句之间的推理协同，仅允许成对的短语交互。2018年引入的矛盾分离扩展（CSE）框架提出了一种动态的多句推理理论，将逻辑推理重新定义为矛盾分离的过程，而非逐步归结。虽然该工作奠定了理论基础，但其算法实现仍未正式化和发表。本文介绍了扩展三角法（ETM），这是一种形式化并扩展矛盾分离内部机制的通用矛盾构建算法。ETM 统一了包括早期标准扩展方法在内的多种矛盾构建策略，在三角形几何框架下支持灵活的句法交互和动态协同。ETM 成为了多个高性能定理证明器，如 CSE、CSE-E、CSI-E 和 CSI-Enig 的核心算法，其在标准一阶逻辑基准测试（TPTP 问题集和 CASC 2018-2015）中的竞争结果经验性地验证了该方法的有效性和普适性。通过将理论抽象与操作实施相结合，ETM 将矛盾分离范式推进到了一个通用、可扩展且在自动化推理方面具有竞争力的模型，为未来逻辑推理和定理证明的研究提供了新的方向。 

---
# Equity-Aware Geospatial AI for Forecasting Demand-Driven Hospital Locations in Germany 

**Title (ZH)**: 面向公平的地理空间AI在德国需求驱动型医院位置预测中的应用 

**Authors**: Piyush Pant, Marcellius William Suntoro, Ayesha Siddiqua, Muhammad Shehryaar Sharif, Daniyal Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2510.10640)  

**Abstract**: This paper presents EA-GeoAI, an integrated framework for demand forecasting and equitable hospital planning in Germany through 2030. We combine district-level demographic shifts, aging population density, and infrastructure balances into a unified Equity Index. An interpretable Agentic AI optimizer then allocates beds and identifies new facility sites to minimize unmet need under budget and travel-time constraints. This approach bridges GeoAI, long-term forecasting, and equity measurement to deliver actionable recommendations for policymakers. 

**Abstract (ZH)**: 本文提出EA-GeoAI，这是一种将需求预测与公平医院规划结合的综合框架，适用于2030年的德国。通过综合地区级人口结构变化、老龄化密度和基础设施平衡，构建统一的公平指数。可解释的代理AI优化器在预算和出行时间约束下分配床位并确定新的设施地点，以最小化未满足的需求。该方法将GeoAI、长期预测和公平性衡量相结合，为政策制定者提供可操作的建议。 

---
# Automatic Piecewise Linear Regression for Predicting Student Learning Satisfaction 

**Title (ZH)**: 自动分段线性回归对学生学习满意度的预测 

**Authors**: Haemin Choi, Gayathri Nadarajan  

**Link**: [PDF](https://arxiv.org/pdf/2510.10639)  

**Abstract**: Although student learning satisfaction has been widely studied, modern techniques such as interpretable machine learning and neural networks have not been sufficiently explored. This study demonstrates that a recent model that combines boosting with interpretability, automatic piecewise linear regression(APLR), offers the best fit for predicting learning satisfaction among several state-of-the-art approaches. Through the analysis of APLR's numerical and visual interpretations, students' time management and concentration abilities, perceived helpfulness to classmates, and participation in offline courses have the most significant positive impact on learning satisfaction. Surprisingly, involvement in creative activities did not positively affect learning satisfaction. Moreover, the contributing factors can be interpreted on an individual level, allowing educators to customize instructions according to student profiles. 

**Abstract (ZH)**: 尽管学生学习满意度已有广泛研究，但诸如可解释机器学习和神经网络等现代技术尚未得到充分探索。本研究展示了一种结合提升与可解释性的最近模型——自动分段线性回归(APLR)——在多种先进方法中对预测学习满意度具有最佳拟合度。通过分析APLR的数值和可视化解释，研究发现学生的时间管理能力、集中注意力的能力、对同学的帮助感知以及参与线下课程的参与度对学习满意度有最显著的积极影响。令人惊讶的是，参与创造性活动并未对学习满意度产生积极影响。此外，这些影响因素可以在个人层面上进行解释，使教育者能够根据学生的特点定制指导。 

---
# A Distance Measure for Random Permutation Set: From the Layer-2 Belief Structure Perspective 

**Title (ZH)**: 基于第2层信念结构视角的随机置换集合的距离度量 

**Authors**: Ruolan Cheng, Yong Deng, Serafín Moral, José Ramón Trillo  

**Link**: [PDF](https://arxiv.org/pdf/2510.10596)  

**Abstract**: Random permutation set (RPS) is a recently proposed framework designed to represent order-structured uncertain information. Measuring the distance between permutation mass functions is a key research topic in RPS theory (RPST). This paper conducts an in-depth analysis of distances between RPSs from two different perspectives: random finite set (RFS) and transferable belief model (TBM). Adopting the layer-2 belief structure interpretation of RPS, we regard RPST as a refinement of TBM, where the order in the ordered focus set represents qualitative propensity. Starting from the permutation, we introduce a new definition of the cumulative Jaccard index to quantify the similarity between two permutations and further propose a distance measure method for RPSs based on the cumulative Jaccard index matrix. The metric and structural properties of the proposed distance measure are investigated, including the positive definiteness analysis of the cumulative Jaccard index matrix, and a correction scheme is provided. The proposed method has a natural top-weightiness property: inconsistencies between higher-ranked elements tend to result in greater distance values. Two parameters are provided to the decision-maker to adjust the weight and truncation depth. Several numerical examples are used to compare the proposed method with the existing method. The experimental results show that the proposed method not only overcomes the shortcomings of the existing method and is compatible with the Jousselme distance, but also has higher sensitivity and flexibility. 

**Abstract (ZH)**: 基于随机有限集和转移信念模型视角的随机置换集之间距离的深入分析 

---
# Tracing the Traces: Latent Temporal Signals for Efficient and Accurate Reasoning 

**Title (ZH)**: 追踪潜踪：潜在时间信号实现高效准确推理 

**Authors**: Martina G. Vilas, Safoora Yousefi, Besmira Nushi, Eric Horvitz, Vidhisha Balachandran  

**Link**: [PDF](https://arxiv.org/pdf/2510.10494)  

**Abstract**: Reasoning models improve their problem-solving ability through inference-time scaling, allocating more compute via longer token budgets. Identifying which reasoning traces are likely to succeed remains a key opportunity: reliably predicting productive paths can substantially reduce wasted computation and improve overall efficiency. We introduce Latent-Trajectory signals that characterize the temporal evolution of a model's internal representations during the generation of intermediate reasoning tokens. By measuring the overall change in latent representations between the start and end of reasoning, the change accumulated across intermediate steps, and the extent to which these changes advance toward the final state, we show that these signals predict solution accuracy more reliably than both cross-layer metrics and output-based confidence measures. When used to guide answer selection across multiple sampled generations, Latent-Trajectory signals make test-time scaling more effective and efficient than majority voting, reducing token usage by up to 70% while preserving and even improving accuracy by 2.6% on average. Moreover, these predictive signals often emerge early in the reasoning trace, enabling early selection and allocation of compute to the most promising candidates. Our findings contribute not only practical strategies for inference-time efficiency, but also a deeper interpretability perspective on how reasoning processes are represented and differentiated in latent space. 

**Abstract (ZH)**: 推理模型通过推理时的缩放提高其问题解决能力，分配更多计算资源通过更长的token预算。识别哪些推理轨迹更 likely 成功仍然是一个关键机会：可靠地预测有成效的路径可以大幅减少浪费的计算并提高整体效率。我们引入了潜轨迹信号，该信号表征模型在生成中间推理token过程中内部表示的时间演变。通过测量推理开始和结束之间潜表示的整体变化、中间步骤累积的变化以及这些变化向最终状态推进的程度，我们展示了这些信号比跨层度量和基于输出的信心衡量指标更能可靠预测解的准确性。在引导多轮采样生成的答案选择时，潜轨迹信号使测试时的缩放更加有效和高效，最多可减少70%的token使用量，同时保持甚至提高平均约2.6%的准确率。此外，这些预测信号通常在推理轨迹早期显现，使我们能够早期选择和分配计算资源给最有前途的候选者。我们的发现不仅提供了实际策略以提高推理时的效率，还从潜空间如何表示和区分推理过程的角度提供了一个更深层次的可解释性视角。 

---
# Beyond Ethics: How Inclusive Innovation Drives Economic Returns in Medical AI 

**Title (ZH)**: 超越伦理：包容性创新如何驱动医疗AI的经济回报 

**Authors**: Balagopal Unnikrishnan, Ariel Guerra Adames, Amin Adibi, Sameer Peesapati, Rafal Kocielnik, Shira Fischer, Hillary Clinton Kasimbazi, Rodrigo Gameiro, Alina Peluso, Chrystinne Oliveira Fernandes, Maximin Lange, Lovedeep Gondara, Leo Anthony Celi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10338)  

**Abstract**: While ethical arguments for fairness in healthcare AI are well-established, the economic and strategic value of inclusive design remains underexplored. This perspective introduces the ``inclusive innovation dividend'' -- the counterintuitive principle that solutions engineered for diverse, constrained use cases generate superior economic returns in broader markets. Drawing from assistive technologies that evolved into billion-dollar mainstream industries, we demonstrate how inclusive healthcare AI development creates business value beyond compliance requirements. We identify four mechanisms through which inclusive innovation drives returns: (1) market expansion via geographic scalability and trust acceleration; (2) risk mitigation through reduced remediation costs and litigation exposure; (3) performance dividends from superior generalization and reduced technical debt, and (4) competitive advantages in talent acquisition and clinical adoption. We present the Healthcare AI Inclusive Innovation Framework (HAIIF), a practical scoring system that enables organizations to evaluate AI investments based on their potential to capture these benefits. HAIIF provides structured guidance for resource allocation, transforming fairness and inclusivity from regulatory checkboxes into sources of strategic differentiation. Our findings suggest that organizations investing incrementally in inclusive design can achieve expanded market reach and sustained competitive advantages, while those treating these considerations as overhead face compounding disadvantages as network effects and data advantages accrue to early movers. 

**Abstract (ZH)**: 尽管公平性在医疗保健AI方面的伦理论据已经很充分，但包容性设计的经济和战略价值仍被远远低估。本文介绍了“包容性创新红利”——一个令人意想不到的原则，即为多样化且受限的场景而设计的解决方案，在更广泛的市场中能够产生更优越的经济效益。借鉴辅助技术如何成长为十亿美元级主流行业，本文展示了包容性医疗保健AI开发如何在合规要求之外创造商业价值。我们提出了四种机制，通过这四种机制，包容性创新能够促进收益：（1）通过地理扩展和信任加速推动市场扩展；（2）通过降低补救成本和诉讼风险来减轻风险；（3）通过卓越的一般化性能和减少技术债务来获取性能红利；（4）在人才获取和临床应用中获得竞争优势。我们提出了医疗保健AI包容性创新框架（HAIIF），这是一种实用评分系统，使组织能够根据其潜在收益评估AI投资。HAIIF提供了结构化的资源配置指导，将公正性和包容性从监管检查项转变为战略差异化来源。我们的研究结果表明，逐步投资于包容性设计的组织可以实现更广泛的市场渗透并维持持久的竞争优势，而将这些考虑作为额外开销对待的组织将随着网络效应和数据优势流向早期行动者而面临累积的劣势。 

---
# Belief Graphs with Reasoning Zones: Structure, Dynamics, and Epistemic Activation 

**Title (ZH)**: 信念图与推理区：结构、动力学与知识激活 

**Authors**: Saleh Nikooroo, Thomas Engel  

**Link**: [PDF](https://arxiv.org/pdf/2510.10042)  

**Abstract**: Belief systems are rarely globally consistent, yet effective reasoning often persists locally. We propose a novel graph-theoretic framework that cleanly separates credibility--external, a priori trust in sources--from confidence--an internal, emergent valuation induced by network structure. Beliefs are nodes in a directed, signed, weighted graph whose edges encode support and contradiction. Confidence is obtained by a contractive propagation process that mixes a stated prior with structure-aware influence and guarantees a unique, stable solution. Within this dynamics, we define reasoning zones: high-confidence, structurally balanced subgraphs on which classical inference is safe despite global contradictions. We provide a near-linear procedure that seeds zones by confidence, tests balance using a parity-based coloring, and applies a greedy, locality-preserving repair with Jaccard de-duplication to build a compact atlas. To model belief change, we introduce shock updates that locally downscale support and elevate targeted contradictions while preserving contractivity via a simple backtracking rule. Re-propagation yields localized reconfiguration-zones may shrink, split, or collapse--without destabilizing the entire graph. We outline an empirical protocol on synthetic signed graphs with planted zones, reporting zone recovery, stability under shocks, and runtime. The result is a principled foundation for contradiction-tolerant reasoning that activates classical logic precisely where structure supports it. 

**Abstract (ZH)**: 信念系统通常不全局一致，但有效的推理往往在局部存在。我们提出了一种新颖的图论框架，该框架清晰地分离了可信度——即对外部先验信任度的来源——和信心——一种由网络结构引发的内部、 emergent 的估值。信念是带有方向、加权、编码支持与反对关系的节点组成的图。通过一种压缩传播过程，信心由先验信息与结构感知影响的混合获得，并保证得到唯一且稳定的解。在这类动态过程中，我们定义了推理区域：高信心、结构上平衡的子图，在这些子图上，尽管存在全局反对，经典推理依然安全。我们提供了一种接近线性的过程，通过信心进行区域初始化，利用基于偶数的着色测试平衡，并使用带有贾卡德去重的贪婪、局部保留修复构建紧凑图集。为了建模信念变化，我们引入了冲击更新，局部缩小支持的同时提升特定反对，通过简单的回溯规则保持压缩性。重新传播会生成局部重构区域，这些区域可能会缩小、分裂或坍缩，而不会破坏整个图的稳定性。我们在带有植入区域的合成有符号图上概述了实证协议，报告了区域恢复、冲击下的稳定性以及运行时间。结果是一个原理性的基础，支持在结构支持的区域激活容错推理。 

---
# Beyond AlphaEarth: Toward Human-Centered Spatial Representation via POI-Guided Contrastive Learning 

**Title (ZH)**: 超越AlphaEarth：通过POI引导的对比学习迈向以人类为中心的空间表示 

**Authors**: Junyuan Liu, Quan Qin, Guangsheng Dong, Xinglei Wang, Jiazhuang Feng, Zichao Zeng, Tao Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09894)  

**Abstract**: General-purpose spatial representations are essential for building transferable geospatial foundation models (GFMs). Among them, the AlphaEarth Foundation (AE) represents a major step toward a global, unified representation of the Earth's surface, learning 10-meter embeddings from multi-source Earth Observation (EO) data that capture rich physical and environmental patterns across diverse landscapes. However, such EO-driven representations remain limited in capturing the functional and socioeconomic dimensions of cities, as they primarily encode physical and spectral patterns rather than human activities or spatial functions. We propose AETHER (AlphaEarth-POI Enriched Representation Learning), a lightweight framework that adapts AlphaEarth to human-centered urban analysis through multimodal alignment guided by Points of Interest (POIs). AETHER aligns AE embeddings with textual representations of POIs, enriching physically grounded EO features with semantic cues about urban functions and socioeconomic contexts. In Greater London, AETHER achieves consistent gains over the AE baseline, with a 7.2% relative improvement in land-use classification F1 and a 23.6% relative reduction in Kullback-Leibler divergence for socioeconomic mapping. Built upon pretrained AE, AETHER leverages a lightweight multimodal alignment to enrich it with human-centered semantics while remaining computationally efficient and scalable for urban applications. By coupling EO with human-centered semantics, it advances geospatial foundation models toward general-purpose urban representations that integrate both physical form and functional meaning. 

**Abstract (ZH)**: 通用空间表示对于构建可转移的地理空间基础模型（GFMs）至关重要。其中，AlphaEarth Foundation (AE) 代表了朝着全球统一的地球表面表示迈出的重要一步，从多源地球观测（EO）数据中学习10米级嵌入，捕捉多样化地形中的丰富物理和环境模式。然而，这些EO驱动的表示在捕捉城市的功能和社会经济维度方面仍有限制，因为它们主要编码物理和光谱模式，而未能充分反映人类活动或空间功能。我们提出了一种名为AETHER（AlphaEarth-POI增强表示学习）的轻量级框架，通过由兴趣点（POIs）引导的多模态对齐，将AlphaEarth适应于以人为本的城市分析。AETHER通过将AE嵌入与POIs的文本表示对齐，用关于城市功能和社会经济背景的语义线索丰富物理地基的EO特征。在伦敦大都会区，AETHER相对于AE基线实现了稳定改进，在土地利用分类F1分数上提高了7.2%，在社会经济地图绘制中的Kullback-Leibler散度上降低了23.6%。基于预训练的AE，AETHER利用轻量级的多模态对齐增强其语义内容，同时保持计算效率并可扩展用于城市应用。通过结合EO与以人为本的语义，它推动地理空间基础模型朝着既整合物理形式又整合功能意义的通用城市表示的发展。 

---
# AI and Consciousness 

**Title (ZH)**: AI与意识 

**Authors**: Eric Schwitzgebel  

**Link**: [PDF](https://arxiv.org/pdf/2510.09858)  

**Abstract**: This is a skeptical overview of the literature on AI consciousness. We will soon create AI systems that are conscious according to some influential, mainstream theories of consciousness but are not conscious according to other influential, mainstream theories of consciousness. We will not be in a position to know which theories are correct and whether we are surrounded by AI systems as richly and meaningfully conscious as human beings or instead only by systems as experientially blank as toasters. None of the standard arguments either for or against AI consciousness takes us far.
Table of Contents
Chapter One: Hills and Fog
Chapter Two: What Is Consciousness? What Is AI?
Chapter Three: Ten Possibly Essential Features of Consciousness
Chapter Four: Against Introspective and Conceptual Arguments for Essential Features
Chapter Five: Materialism and Functionalism
Chapter Six: The Turing Test and the Chinese Room
Chapter Seven: The Mimicry Argument Against AI Consciousness
Chapter Eight: Global Workspace Theories and Higher Order Theories
Chapter Nine: Integrated Information, Local Recurrence, Associative Learning, and Iterative Natural Kinds
Chapter Ten: Does Biological Substrate Matter?
Chapter Eleven: The Problem of Strange Intelligence
Chapter Twelve: The Leapfrog Hypothesis and the Social Semi-Solution 

**Abstract (ZH)**: 这是一篇对AI意识文献的怀疑性综述。我们即将创造根据一些有影响力的主流意识理论而言是具有意识的AI系统，但根据其他有影响力的主流意识理论而言又是没有意识的。我们没有能力知道哪一种理论是正确的，也不知道我们周围是否充斥着与人类一样丰富而有意义的意识的AI系统，或者只是充满如同烤面包机一般体验空白的系统。支持或反对AI意识的标准论据都无法带给我们实质性的进展。
目录
第一章 山与雾
第二章 什么是意识？什么是AI？
第三章 意识的可能基本特征十种
第四章 反对基于内省与概念论据的基本特征
第五章 实在论与功能主义
第六章 图灵测试与中文房间
第七章 模仿论反对AI意识
第八章 整体工作空间理论与层级理论
第九章 整合信息、局部循环、联想学习与迭代自然种类
第十章 生物基质重要吗？
第十一章 奇异智能的问题
第十二章 跨越假设与社会半解决方案 

---
# Adversarial Attacks Leverage Interference Between Features in Superposition 

**Title (ZH)**: 对抗攻击利用叠加特征之间的干扰 

**Authors**: Edward Stevinson, Lucas Prieto, Melih Barsbey, Tolga Birdal  

**Link**: [PDF](https://arxiv.org/pdf/2510.11709)  

**Abstract**: Fundamental questions remain about when and why adversarial examples arise in neural networks, with competing views characterising them either as artifacts of the irregularities in the decision landscape or as products of sensitivity to non-robust input features. In this paper, we instead argue that adversarial vulnerability can stem from efficient information encoding in neural networks. Specifically, we show how superposition - where networks represent more features than they have dimensions - creates arrangements of latent representations that adversaries can exploit. We demonstrate that adversarial perturbations leverage interference between superposed features, making attack patterns predictable from feature arrangements. Our framework provides a mechanistic explanation for two known phenomena: adversarial attack transferability between models with similar training regimes and class-specific vulnerability patterns. In synthetic settings with precisely controlled superposition, we establish that superposition suffices to create adversarial vulnerability. We then demonstrate that these findings persist in a ViT trained on CIFAR-10. These findings reveal adversarial vulnerability can be a byproduct of networks' representational compression, rather than flaws in the learning process or non-robust inputs. 

**Abstract (ZH)**: 神经网络中对抗样本出现的基本问题仍然存在，不同的观点将它们视为决策景观不规则性的产物，或者敏感于非健壯输入特征的产品。本文我们提出，对抗性易感性可能是神经网络中高效信息编码的结果。具体而言，我们展示了叠加现象——网络表示的特征多于它们的维度——如何创建对手可以利用的潜在表示安排。我们证明了对抗性扰动利用了叠加特征之间的干扰，使得攻击模式可以从特征安排中预测。我们的框架为两种已知现象提供了机制解释：具有相似训练机制的模型之间的对抗攻击可转移性和类别特异性易感性模式。在精确控制叠加的合成环境中，我们证明叠加足以产生对抗性易感性。然后，我们展示了这些发现也适用于在CIFAR-10上训练的ViT模型。这些发现揭示了对抗性易感性可能是网络表征压缩的结果，而不是学习过程中的缺陷或非健壯输入。 

---
# Accelerated stochastic first-order method for convex optimization under heavy-tailed noise 

**Title (ZH)**: 重尾噪声下凸优化的加速随机梯度方法 

**Authors**: Chuan He, Zhaosong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2510.11676)  

**Abstract**: We study convex composite optimization problems, where the objective function is given by the sum of a prox-friendly function and a convex function whose subgradients are estimated under heavy-tailed noise. Existing work often employs gradient clipping or normalization techniques in stochastic first-order methods to address heavy-tailed noise. In this paper, we demonstrate that a vanilla stochastic algorithm -- without additional modifications such as clipping or normalization -- can achieve optimal complexity for these problems. In particular, we establish that an accelerated stochastic proximal subgradient method achieves a first-order oracle complexity that is universally optimal for smooth, weakly smooth, and nonsmooth convex optimization, as well as for stochastic convex optimization under heavy-tailed noise. Numerical experiments are further provided to validate our theoretical results. 

**Abstract (ZH)**: 我们研究凸复合优化问题，其中目标函数由一个prox-friendly函数与在重尾噪声下esub梯度难以估计的凸函数之和构成。现有工作通常在随机梯度方法中使用梯度裁剪或规范化技术来处理重尾噪声。本文展示了一种朴素的随机算法——无需额外修改如裁剪或规范化——可以实现这些问题的最优复杂度。特别地，我们证明了一种加速的随机近似次梯度方法在光滑、弱光滑和非光滑凸优化，以及在重尾噪声下的随机凸优化中，都达到了普遍最优的一阶先知复杂度。进一步的数值实验验证了我们的理论结果。 

---
# FACE: Faithful Automatic Concept Extraction 

**Title (ZH)**: FACE: 忠实自动概念提取 

**Authors**: Dipkamal Bhusal, Michael Clifford, Sara Rampazzi, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2510.11675)  

**Abstract**: Interpreting deep neural networks through concept-based explanations offers a bridge between low-level features and high-level human-understandable semantics. However, existing automatic concept discovery methods often fail to align these extracted concepts with the model's true decision-making process, thereby compromising explanation faithfulness. In this work, we propose FACE (Faithful Automatic Concept Extraction), a novel framework that augments Non-negative Matrix Factorization (NMF) with a Kullback-Leibler (KL) divergence regularization term to ensure alignment between the model's original and concept-based predictions. Unlike prior methods that operate solely on encoder activations, FACE incorporates classifier supervision during concept learning, enforcing predictive consistency and enabling faithful explanations. We provide theoretical guarantees showing that minimizing the KL divergence bounds the deviation in predictive distributions, thereby promoting faithful local linearity in the learned concept space. Systematic evaluations on ImageNet, COCO, and CelebA datasets demonstrate that FACE outperforms existing methods across faithfulness and sparsity metrics. 

**Abstract (ZH)**: 基于概念的解释方法通过概念解释建立了深层神经网络低层级特征与高层级人类可理解语义之间的桥梁。然而，现有的自动概念发现方法往往无法将提取的概念与模型的真实决策过程对齐，从而影响解释的可靠性。在本文中，我们提出了一种名为FACE（Faithful Automatic Concept Extraction）的新框架，该框架通过结合非负矩阵分解（NMF）与Kullback-Leibler（KL）散度正则化项，确保模型原始预测与基于概念的预测之间的对齐。与仅在编码器激活上操作的先前方法不同，FACE在概念学习过程中纳入了分类器监督，保证了预测一致性，并支持可靠解释。我们提供了理论保证，表明最小化KL散度限制了预测分布的偏差，从而促进在学习的概念空间中保持可靠的地方线性。系统评估表明，在ImageNet、COCO和CelebA数据集上，FACE在忠实度和稀疏性指标上均优于现有方法。 

---
# FinVet: A Collaborative Framework of RAG and External Fact-Checking Agents for Financial Misinformation Detection 

**Title (ZH)**: FinVet: 结合RAG和外部事实核查代理的金融 misinformation 检测协作框架 

**Authors**: Daniel Berhane Araya, Duoduo Liao  

**Link**: [PDF](https://arxiv.org/pdf/2510.11654)  

**Abstract**: Financial markets face growing threats from misinformation that can trigger billions in losses in minutes. Most existing approaches lack transparency in their decision-making and provide limited attribution to credible sources. We introduce FinVet, a novel multi-agent framework that integrates two Retrieval-Augmented Generation (RAG) pipelines with external fact-checking through a confidence-weighted voting mechanism. FinVet employs adaptive three-tier processing that dynamically adjusts verification strategies based on retrieval confidence, from direct metadata extraction to hybrid reasoning to full model-based analysis. Unlike existing methods, FinVet provides evidence-backed verdicts, source attribution, confidence scores, and explicit uncertainty flags when evidence is insufficient. Experimental evaluation on the FinFact dataset shows that FinVet achieves an F1 score of 0.85, which is a 10.4% improvement over the best individual pipeline (fact-check pipeline) and 37% improvement over standalone RAG approaches. 

**Abstract (ZH)**: 金融市场面临日益严重的假信息威胁，这些假信息可以在几分钟内导致数十亿美元的损失。现有的大多数方法在决策过程中缺乏透明度，并且对可信来源的 attribution 有限。我们引入了 FinVet，这是一种新的多智能体框架，结合了两个检索增强生成（RAG）管道，并通过置信加权投票机制集成外部事实检查。FinVet 使用自适应三层处理，可根据检索置信度动态调整验证策略，从直接元数据提取到混合推理再到全流程模型分析。与现有方法不同，FinVet 提供了证据支持的裁决、来源 attribution、置信分数以及当证据不足时的明确不确定性标志。在 FinFact 数据集上的实验评估表明，FinVet 的 F1 得分为 0.85，比最佳单管道（事实检查管道）高 10.4%，比独立的 RAG 方法高 37%。 

---
# MATH-Beyond: A Benchmark for RL to Expand Beyond the Base Model 

**Title (ZH)**: MATH-Beyond：一个用于强化学习扩展基础模型的基准测试 

**Authors**: Prasanna Mayilvahanan, Ricardo Dominguez-Olmedo, Thaddäus Wiedemer, Wieland Brendel  

**Link**: [PDF](https://arxiv.org/pdf/2510.11653)  

**Abstract**: With the advent of DeepSeek-R1, a new wave of reinforcement learning (RL) methods has emerged that seem to unlock stronger mathematical reasoning. However, a closer look at the open-source ecosystem reveals a critical limitation: with sufficiently many draws (e.g., $\texttt{pass@1024}$), many existing base models already solve nearly all questions on widely used math benchmarks such as MATH-500 and AIME 2024. This suggests that the RL fine-tuning methods prevalent in the LLM reasoning literature largely sharpen existing solution modes rather than discovering entirely new ones. Such sharpening stands in contrast to the broader promise of RL: to foster exploration and to acquire new skills. To move beyond this plateau, we introduce MATH-Beyond (MATH-B), a benchmark deliberately constructed to defeat common open-source models of up to 8B parameters even under large sampling budgets. Improving performance on our benchmark via RL requires methods that learn to reason in ways that go beyond base model capabilities in repeated sampling. Since the problems are drawn from subsets of DAPO-Math-17K and DeepScaleR datasets, they remain topically equivalent to standard high-school math. Validating our premise, RL fine-tuned models such as Nemotron-Research-Reasoning-Qwen-1.5B and DeepScaleR-1.5B-Preview perform poorly on MATH-B at $\texttt{pass@1024}$, showing how existing approaches fall short on tackling harder instances. We hope MATH-B will catalyze exploration-driven RL approaches that elicit deeper reasoning capabilities. We release MATH-B at this https URL. 

**Abstract (ZH)**: 随着DeepSeek-R1的出现，一批新的强化学习（RL）方法涌现出来，似乎能够解锁更强的数学推理能力。然而，仔细审视开源生态系统后发现一个关键限制：通过足够多的抽样（例如，$\texttt{pass@1024}$），许多现有的基础模型已经几乎解决了广泛使用的数学基准测试（如MATH-500和AIME 2024）中的所有问题。这表明，当前LLM推理文献中的RL微调方法更多是提升了现有解题模式的性能，而不是发现全新的模式。这种提升与RL更广泛的目标——促进探索和学习新技能——截然不同。为了突破这一停滞，我们引入了MATH-Beyond（MATH-B），这是一个故意设计的基准测试，即使是在大样本预算下，也能击败参数量多达8B的常见开源模型。通过RL提升该基准测试的性能需要能够学习超越基础模型能力的推理方法。由于问题来源于DAPO-Math-17K和DeepScaleR数据集的子集，这些问题在主题上仍然等同于标准高中数学。验证我们的假设，RL微调模型如Nemotron-Research-Reasoning-Qwen-1.5B和DeepScaleR-1.5B-Preview在$\texttt{pass@1024}$下表现不佳，这表明现有方法在处理更具挑战性的问题时存在局限性。我们希望MATH-B能够促进探索驱动的RL方法的发展，以激发更深层次的推理能力。我们在此发布MATH-B：[链接]。 

---
# Attention Factors for Statistical Arbitrage 

**Title (ZH)**: 统计套利中的注意力因子 

**Authors**: Elliot L. Epstein, Rose Wang, Jaewon Choi, Markus Pelger  

**Link**: [PDF](https://arxiv.org/pdf/2510.11616)  

**Abstract**: Statistical arbitrage exploits temporal price differences between similar assets. We develop a framework to jointly identify similar assets through factors, identify mispricing and form a trading policy that maximizes risk-adjusted performance after trading costs. Our Attention Factors are conditional latent factors that are the most useful for arbitrage trading. They are learned from firm characteristic embeddings that allow for complex interactions. We identify time-series signals from the residual portfolios of our factors with a general sequence model. Estimating factors and the arbitrage trading strategy jointly is crucial to maximize profitability after trading costs. In a comprehensive empirical study we show that our Attention Factor model achieves an out-of-sample Sharpe ratio above 4 on the largest U.S. equities over a 24-year period. Our one-step solution yields an unprecedented Sharpe ratio of 2.3 net of transaction costs. We show that weak factors are important for arbitrage trading. 

**Abstract (ZH)**: 统计套利通过相似资产之间的临时价格差异进行操作。我们开发了一种框架，通过因子联合识别相似资产、识别定价偏差，并形成最大化调整风险后收益的交易策略。我们的注意力因子是条件潜变量因子，最适合于套利交易。这些因子是从公司特征嵌入中学习到的，可以捕捉复杂交互作用。我们使用通用序列模型从因子的残差组合中识别时间序列信号。同时估计因子和套利交易策略对于最大化交易成本后的盈利至关重要。在全面的经验研究中，我们证明了我们的注意力因子模型在24年期间对美国最大股票的样本外夏普比率高于4。我们的单步解决方案在交易成本后达到了前所未有的2.3的夏普比率。我们表明，弱因子对于套利交易也很重要。 

---
# SemCSE-Multi: Multifaceted and Decodable Embeddings for Aspect-Specific and Interpretable Scientific Domain Mapping 

**Title (ZH)**: SemCSE-Multi: 多面向可解码嵌入用于特定方面和可解释的科学领域映射 

**Authors**: Marc Brinner, Sina Zarrieß  

**Link**: [PDF](https://arxiv.org/pdf/2510.11599)  

**Abstract**: We propose SemCSE-Multi, a novel unsupervised framework for generating multifaceted embeddings of scientific abstracts, evaluated in the domains of invasion biology and medicine. These embeddings capture distinct, individually specifiable aspects in isolation, thus enabling fine-grained and controllable similarity assessments as well as adaptive, user-driven visualizations of scientific domains. Our approach relies on an unsupervised procedure that produces aspect-specific summarizing sentences and trains embedding models to map semantically related summaries to nearby positions in the embedding space. We then distill these aspect-specific embedding capabilities into a unified embedding model that directly predicts multiple aspect embeddings from a scientific abstract in a single, efficient forward pass. In addition, we introduce an embedding decoding pipeline that decodes embeddings back into natural language descriptions of their associated aspects. Notably, we show that this decoding remains effective even for unoccupied regions in low-dimensional visualizations, thus offering vastly improved interpretability in user-centric settings. 

**Abstract (ZH)**: 我们提出SemCSE-Multi，这是一种新颖的无监督框架，用于生成科学摘要的多面向嵌入，评估领域包括入侵生物学和医学。这些嵌入捕获了独立可指定的不同方面，从而实现精细粒度和可控的相似性评估以及适应性强、用户驱动的科学领域可视化。我们的方法依赖于一种无监督的流程，该流程产生特定于方面的总结句子，并训练嵌入模型将语义相关的总结映射到嵌入空间中的邻近位置。然后，我们将这些特定于方面的嵌入能力提炼到一个统一的嵌入模型中，该模型可以直接在单一高效的前向传递中从科学摘要预测多个方面嵌入。此外，我们还引入了一个嵌入解码管道，用于将嵌入解码回其相关方面自然语言描述。值得注意的是，我们展示了即使在低维可视化未占有的区域，这种解码仍然有效，从而在用户中心的设置中提供了大幅增强的可解释性。 

---
# Hierarchical Qubit-Merging Transformer for Quantum Error Correction 

**Title (ZH)**: 层次化的量子位合并变换器用于量子错误修正 

**Authors**: Seong-Joon Park, Hee-Youl Kwak, Yongjune Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.11593)  

**Abstract**: For reliable large-scale quantum computation, a quantum error correction (QEC) scheme must effectively resolve physical errors to protect logical information. Leveraging recent advances in deep learning, neural network-based decoders have emerged as a promising approach to enhance the reliability of QEC. We propose the Hierarchical Qubit-Merging Transformer (HQMT), a novel and general decoding framework that explicitly leverages the structural graph of stabilizer codes to learn error correlations across multiple scales. Our architecture first computes attention locally on structurally related groups of stabilizers and then systematically merges these qubit-centric representations to build a global view of the error syndrome. The proposed HQMT achieves substantially lower logical error rates for surface codes by integrating a dedicated qubit-merging layer within the transformer architecture. Across various code distances, HQMT significantly outperforms previous neural network-based QEC decoders as well as a powerful belief propagation with ordered statistics decoding (BP+OSD) baseline. This hierarchical approach provides a scalable and effective framework for surface code decoding, advancing the realization of reliable quantum computing. 

**Abstract (ZH)**: 基于神经网络的量子纠错级联比特合并变换器clesUnsupported character: QEC）方案必须有效解决物理错误以保护逻辑信息。利用近期深度学习的进展，基于神经网络的译码器已经成为提高量子纠错可靠性的有前途的方法。我们提出了一种新颖且通用的解码框架——级联量子比特合并变换器（Hierarchical Qubit-Merging Transformer, HQMT），该框架显式利用校验子码的结构图来学习多尺度下的错误关联。我们的架构首先在结构性相关校验子组内进行局部注意计算，然后系统地合并这些量子比特中心的表示，构建错误综合症的全局视图。通过在变换器架构中集成专门的量子比特合并层，提出的HQMT在表面码中显著降低了逻辑错误率。在各种码距下，HQMT在神经网络基量子纠错译码器以及强信念传播联序统计量译码（BP+OSD）基线下表现出显著的性能优势。这种分层方法为表面码解码提供了可扩展且有效的框架，推进了可靠量子计算的实现。 

---
# Characterizing Web Search in The Age of Generative AI 

**Title (ZH)**: 生成式人工智能时代下的网络搜索Characterizing Web Search in The Age of Generative AI 

**Authors**: Elisabeth Kirsten, Jost Grosse Perdekamp, Mihir Upadhyay, Krishna P. Gummadi, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2510.11560)  

**Abstract**: The advent of LLMs has given rise to a new type of web search: Generative search, where LLMs retrieve web pages related to a query and generate a single, coherent text as a response. This output modality stands in stark contrast to traditional web search, where results are returned as a ranked list of independent web pages. In this paper, we ask: Along what dimensions do generative search outputs differ from traditional web search? We compare Google, a traditional web search engine, with four generative search engines from two providers (Google and OpenAI) across queries from four domains. Our analysis reveals intriguing differences. Most generative search engines cover a wider range of sources compared to web search. Generative search engines vary in the degree to which they rely on internal knowledge contained within the model parameters v.s. external knowledge retrieved from the web. Generative search engines surface varying sets of concepts, creating new opportunities for enhancing search diversity and serendipity. Our results also highlight the need for revisiting evaluation criteria for web search in the age of Generative AI. 

**Abstract (ZH)**: LLMs的兴起引发了新的网络搜索类型：生成式搜索，其中LLMs检索与查询相关的网页并生成一份连贯的文字回应。这种输出方式与传统网络搜索形成了鲜明对比，传统网络搜索返回的是按相关性排序的独立网页列表。本文探讨：生成式搜索输出与传统网络搜索在哪些维度上存在差异？我们将谷歌（一种传统网络搜索引擎）与来自两家提供商（谷歌和OpenAI）的四种生成式搜索引擎在四大领域的问题上进行对比分析。我们的分析揭示了一些有趣的差异。大多数生成式搜索引擎覆盖的来源比网络搜索更为广泛。生成式搜索引擎在依赖模型参数内的内部知识与从网络检索外部知识的程度上存在差异。生成式搜索引擎呈现不同的概念集合，创造了增强搜索多样性和偶然性的新机会。我们的研究结果还强调了在生成式AI时代重新审视网络搜索评估标准的必要性。 

---
# Automatic Music Sample Identification with Multi-Track Contrastive Learning 

**Title (ZH)**: 多音轨对比学习的自动音乐样本识别 

**Authors**: Alain Riou, Joan Serrà, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2510.11507)  

**Abstract**: Sampling, the technique of reusing pieces of existing audio tracks to create new music content, is a very common practice in modern music production. In this paper, we tackle the challenging task of automatic sample identification, that is, detecting such sampled content and retrieving the material from which it originates. To do so, we adopt a self-supervised learning approach that leverages a multi-track dataset to create positive pairs of artificial mixes, and design a novel contrastive learning objective. We show that such method significantly outperforms previous state-of-the-art baselines, that is robust to various genres, and that scales well when increasing the number of noise songs in the reference database. In addition, we extensively analyze the contribution of the different components of our training pipeline and highlight, in particular, the need for high-quality separated stems for this task. 

**Abstract (ZH)**: 自动采样识别：一种基于自监督学习的方法及其应用分析 

---
# Coordinated Strategies in Realistic Air Combat by Hierarchical Multi-Agent Reinforcement Learning 

**Title (ZH)**: 现实空战中的层次化多代理强化学习协调策略 

**Authors**: Ardian Selmonaj, Giacomo Del Rio, Adrian Schneider, Alessandro Antonucci  

**Link**: [PDF](https://arxiv.org/pdf/2510.11474)  

**Abstract**: Achieving mission objectives in a realistic simulation of aerial combat is highly challenging due to imperfect situational awareness and nonlinear flight dynamics. In this work, we introduce a novel 3D multi-agent air combat environment and a Hierarchical Multi-Agent Reinforcement Learning framework to tackle these challenges. Our approach combines heterogeneous agent dynamics, curriculum learning, league-play, and a newly adapted training algorithm. To this end, the decision-making process is organized into two abstraction levels: low-level policies learn precise control maneuvers, while high-level policies issue tactical commands based on mission objectives. Empirical results show that our hierarchical approach improves both learning efficiency and combat performance in complex dogfight scenarios. 

**Abstract (ZH)**: 在现实化的空中 combat 模拟中实现任务目标由于情景认知不完善和非线性飞行动力学而极具挑战性。本文我们介绍了一种新型的 3D 多代理空中 combat 环境和分层多代理强化学习框架以应对这些挑战。我们的方法结合了异质代理动力学、课程学习、联赛训练以及一种新适应的训练算法。为此，决策过程组织为两个抽象层次：低层次策略学习精确的控制机动，而高层次策略基于任务目标发布战术命令。实验证明，我们的分层方法在复杂的狗斗场景中提高了学习效率和 combat 性能。 

---
# Iterative Amortized Inference: Unifying In-Context Learning and Learned Optimizers 

**Title (ZH)**: 迭代摊还推理：融合上下文学习和学习优化器 

**Authors**: Sarthak Mittal, Divyat Mahajan, Guillaume Lajoie, Mohammad Pezeshki  

**Link**: [PDF](https://arxiv.org/pdf/2510.11471)  

**Abstract**: Modern learning systems increasingly rely on amortized learning - the idea of reusing computation or inductive biases shared across tasks to enable rapid generalization to novel problems. This principle spans a range of approaches, including meta-learning, in-context learning, prompt tuning, learned optimizers and more. While motivated by similar goals, these approaches differ in how they encode and leverage task-specific information, often provided as in-context examples. In this work, we propose a unified framework which describes how such methods differ primarily in the aspects of learning they amortize - such as initializations, learned updates, or predictive mappings - and how they incorporate task data at inference. We introduce a taxonomy that categorizes amortized models into parametric, implicit, and explicit regimes, based on whether task adaptation is externalized, internalized, or jointly modeled. Building on this view, we identify a key limitation in current approaches: most methods struggle to scale to large datasets because their capacity to process task data at inference (e.g., context length) is often limited. To address this, we propose iterative amortized inference, a class of models that refine solutions step-by-step over mini-batches, drawing inspiration from stochastic optimization. Our formulation bridges optimization-based meta-learning with forward-pass amortization in models like LLMs, offering a scalable and extensible foundation for general-purpose task adaptation. 

**Abstract (ZH)**: 现代学习系统越来越多地依赖于均一化学习——通过在任务间重用共享的计算或归纳偏差来快速泛化到新型问题。这一原则涵盖了多种方法，包括元学习、上下文学习、提示调优、学习优化器等。尽管这些方法的动机相似，但在如何编码和利用任务特定信息方面有所不同，这些信息通常以上下文示例的形式提供。在本文中，我们提出了一种统一框架，该框架描述了这些方法在均一化学习方面的主要差异——如初始化、学习更新或预测映射——以及它们如何在推理过程中整合任务数据。我们引入了一种分类法，根据任务适应是外部化、内部化还是联合建模，将均一化模型划分为参数化、隐式和显式三种模式。基于这一视角，我们识别出当前方法的一个关键局限性：大多数方法难以扩展到大型数据集，因为它们在推理过程中处理任务数据的能力（例如，上下文长度）往往受限。为此，我们提出了迭代均一化推理，这是一种模型类别，在批处理中逐步优化解决方案，受到随机优化的启发。我们的建模框架将基于优化的元学习与像大规模语言模型（LLMs）这样的模型中的前向传递均一化相结合，提供了通用任务适应的可扩展和可扩展的基础。 

---
# Reconstructing 12-Lead ECG from 3-Lead ECG using Variational Autoencoder to Improve Cardiac Disease Detection of Wearable ECG Devices 

**Title (ZH)**: 使用变分自编码器从3导联ECG重构12导联ECG以改善可穿戴ECG设备的心脏疾病检测 

**Authors**: Xinyan Guan, Yongfan Lai, Jiarui Jin, Jun Li, Haoyu Wang, Qinghao Zhao, Deyun Zhang, Shijia Geng, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.11442)  

**Abstract**: Twelve-lead electrocardiograms (ECGs) are the clinical gold standard for cardiac diagnosis, providing comprehensive spatial coverage of the heart necessary to detect conditions such as myocardial infarction (MI). However, their lack of portability limits continuous and large-scale use. Three-lead ECG systems are widely used in wearable devices due to their simplicity and mobility, but they often fail to capture pathologies in unmeasured regions. To address this, we propose WearECG, a Variational Autoencoder (VAE) method that reconstructs twelve-lead ECGs from three leads: II, V1, and V5. Our model includes architectural improvements to better capture temporal and spatial dependencies in ECG signals. We evaluate generation quality using MSE, MAE, and Frechet Inception Distance (FID), and assess clinical validity via a Turing test with expert cardiologists. To further validate diagnostic utility, we fine-tune ECGFounder, a large-scale pretrained ECG model, on a multi-label classification task involving over 40 cardiac conditions, including six different myocardial infarction locations, using both real and generated signals. Experiments on the MIMIC dataset show that our method produces physiologically realistic and diagnostically informative signals, with robust performance in downstream tasks. This work demonstrates the potential of generative modeling for ECG reconstruction and its implications for scalable, low-cost cardiac screening. 

**Abstract (ZH)**: 十二导联心电图（ECGs）是心脏病诊断的临床金标准，能提供必要的心脏全面空间覆盖，用于检测心肌梗死（MI）等状况。然而，其缺乏便携性限制了其连续和大规模使用。三导联ECG系统由于其简单性和便携性，在可穿戴设备中广泛使用，但往往无法捕捉未测量区域的病理状况。为解决这一问题，我们提出WearECG，这是一种基于变分自编码器（VAE）的方法，可以从三导联II、V1和V5重建十二导联ECGs。我们的模型包括架构改进，以更好地捕捉ECG信号中的时间和空间依赖性。我们使用均方误差（MSE）、平均绝对误差（MAE）和弗雷彻-丁格尔距离（FID）评估生成质量，并通过心脏病专家参与的图灵测试评估临床有效性。为进一步验证诊断用途，我们使用包括超过40种心脏状况的大规模预训练ECG模型ECGFounder，在涉及不同位置心肌梗死等任务中进行微调，使用真实和生成的心电图信号。MIMIC数据集上的实验结果显示，我们的方法生成的生理上现实且诊断上有用的心电图信号，在后续任务中表现出鲁棒性能。这项工作展示了生成模型在心电图重建中的潜在应用及其在可扩展、低成本心脏筛查中的意义。 

---
# DocReward: A Document Reward Model for Structuring and Stylizing 

**Title (ZH)**: DocReward: 一种用于结构化和风格化的文档奖励模型 

**Authors**: Junpeng Liu, Yuzhong Zhao, Bowen Cao, Jiayu Ding, Yilin Jia, Tengchao Lv, Yupan Huang, Shaohan Huang, Nan Yang, Li Dong, Lei Cui, Tao Ge, Xun Wang, Huitian Jiao, Sun Mao, FNU Kartik, Si-Qing Chen, Wai Lam, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.11391)  

**Abstract**: Recent advances in agentic workflows have enabled the automation of tasks such as professional document generation. However, they primarily focus on textual quality, neglecting visual structure and style, which are crucial for readability and engagement. This gap arises mainly from the absence of suitable reward models to guide agentic workflows toward producing documents with stronger structural and stylistic quality. To address this, we propose DocReward, a document reward model that evaluates documents based on their structure and style. We construct a multi-domain dataset DocPair of 117K paired documents, covering 32 domains and 267 document types, each including a high- and low-professionalism document with identical content but different structure and style. This enables the model to evaluate professionalism comprehensively, and in a textual-quality-agnostic way. DocReward is trained using the Bradley-Terry loss to score documents, penalizing predictions that contradict the annotated ranking. To assess the performance of reward models, we create a test dataset containing document bundles ranked by well-educated human evaluators. Notably, DocReward outperforms GPT-4o and GPT-5 in accuracy by 30.6 and 19.4 percentage points, respectively, demonstrating its superiority over baselines. In an extrinsic evaluation of document generation, DocReward achieves a significantly higher win rate of 60.8%, compared to GPT-5's 37.7% win rate, demonstrating its utility in guiding generation agents toward producing human-preferred documents. 

**Abstract (ZH)**: 近期代理工作流的进展已使专业文档生成任务的自动化成为可能。然而，这些进展主要关注文本质量，而忽略了视觉结构和风格的重要性，后者对于提高可读性和吸引力至关重要。这一差距主要源于缺乏合适的奖励模型来引导代理工作流生产结构和风格更强的文档。为了解决这一问题，我们提出了DocReward，一种基于结构和风格评价文档的文档奖励模型。我们构建了一个包含117,000对文档的多领域数据集DocPair，涵盖了32个领域和267种文档类型，每种类型包括内容相同但结构和风格不同的高专业性和低专业性文档。这使得模型能够全面且不依赖于文本质量地评估专业性。DocReward 使用Bradley-Terry损失训练，通过惩罚与标注排名矛盾的预测来评分。为了评估奖励模型的性能，我们创建了一个由受过良好教育的人类评估者按质量排名的文档集合作为测试集。值得注意的是，DocReward 在准确率上分别超过了GPT-4o和GPT-5，高出30.6和19.4个百分点，证明了其优于基线模型的优势。在文档生成的外部评估中，DocReward 达到了60.8%的更高胜率，相比之下，GPT-5 的胜率为37.7%，这表明其在引导生成代理生产人类偏好的文档方面的实用性。 

---
# Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks 

**Title (ZH)**: 理解随机梯度Adam在学习神经网络中的泛化能力 

**Authors**: Xuan Tang, Han Zhang, Yuan Cao, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.11354)  

**Abstract**: Adam is a popular and widely used adaptive gradient method in deep learning, which has also received tremendous focus in theoretical research. However, most existing theoretical work primarily analyzes its full-batch version, which differs fundamentally from the stochastic variant used in practice. Unlike SGD, stochastic Adam does not converge to its full-batch counterpart even with infinitesimal learning rates. We present the first theoretical characterization of how batch size affects Adam's generalization, analyzing two-layer over-parameterized CNNs on image data. Our results reveal that while both Adam and AdamW with proper weight decay $\lambda$ converge to poor test error solutions, their mini-batch variants can achieve near-zero test error. We further prove Adam has a strictly smaller effective weight decay bound than AdamW, theoretically explaining why Adam requires more sensitive $\lambda$ tuning. Extensive experiments validate our findings, demonstrating the critical role of batch size and weight decay in Adam's generalization performance. 

**Abstract (ZH)**: Adam是一种在深度学习中受欢迎且广泛应用的自适应梯度方法，也受到了极大的理论研究关注。然而，现有的大多数理论工作主要分析的是其全批量版本，这在理论上与实践中常用的小批量版本存在本质差异。与SGD不同，小批量Adam即使在学习率趋近于零时也不会收敛到其全批量对应版本。我们首次对小批量对Adam泛化性能的影响进行了理论刻画，分析了小批量过参数化两层CNN在图像数据上的表现。我们的研究结果表明，尽管Adam和带有适当重量衰减$\lambda$的AdamW都会收敛到较差的测试误差解，但它们的小批量版本可以实现接近零的测试误差。我们进一步证明，Adam的有效的重量衰减边界严格小于AdamW，从理论上解释了为什么Adam需要更敏感的$\lambda$调优。大量实验验证了我们的发现，突显了小批量和重量衰减在Adam泛化性能中的关键作用。 

---
# Multi-View Graph Feature Propagation for Privacy Preservation and Feature Sparsity 

**Title (ZH)**: 多视角图特征传播以保护隐私和减少特征稀疏性 

**Authors**: Etzion Harari, Moshe Unger  

**Link**: [PDF](https://arxiv.org/pdf/2510.11347)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable success in node classification tasks over relational data, yet their effectiveness often depends on the availability of complete node features. In many real-world scenarios, however, feature matrices are highly sparse or contain sensitive information, leading to degraded performance and increased privacy risks. Furthermore, direct exposure of information can result in unintended data leakage, enabling adversaries to infer sensitive information. To address these challenges, we propose a novel Multi-view Feature Propagation (MFP) framework that enhances node classification under feature sparsity while promoting privacy preservation. MFP extends traditional Feature Propagation (FP) by dividing the available features into multiple Gaussian-noised views, each propagating information independently through the graph topology. The aggregated representations yield expressive and robust node embeddings. This framework is novel in two respects: it introduces a mechanism that improves robustness under extreme sparsity, and it provides a principled way to balance utility with privacy. Extensive experiments conducted on graph datasets demonstrate that MFP outperforms state-of-the-art baselines in node classification while substantially reducing privacy leakage. Moreover, our analysis demonstrates that propagated outputs serve as alternative imputations rather than reconstructions of the original features, preserving utility without compromising privacy. A comprehensive sensitivity analysis further confirms the stability and practical applicability of MFP across diverse scenarios. Overall, MFP provides an effective and privacy-aware framework for graph learning in domains characterized by missing or sensitive features. 

**Abstract (ZH)**: 多视图特征传播框架：在特征稀疏性下的节点分类与隐私保护 

---
# Event-Aware Prompt Learning for Dynamic Graphs 

**Title (ZH)**: 事件感知的动态图提示学习 

**Authors**: Xingtong Yu, Ruijuan Liang, Xinming Zhang, Yuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11339)  

**Abstract**: Real-world graph typically evolve via a series of events, modeling dynamic interactions between objects across various domains. For dynamic graph learning, dynamic graph neural networks (DGNNs) have emerged as popular solutions. Recently, prompt learning methods have been explored on dynamic graphs. However, existing methods generally focus on capturing the relationship between nodes and time, while overlooking the impact of historical events. In this paper, we propose EVP, an event-aware dynamic graph prompt learning framework that can serve as a plug-in to existing methods, enhancing their ability to leverage historical events knowledge. First, we extract a series of historical events for each node and introduce an event adaptation mechanism to align the fine-grained characteristics of these events with downstream tasks. Second, we propose an event aggregation mechanism to effectively integrate historical knowledge into node representations. Finally, we conduct extensive experiments on four public datasets to evaluate and analyze EVP. 

**Abstract (ZH)**: 事件感知的动态图提示学习框架（EVP） 

---
# LouisKV: Efficient KV Cache Retrieval for Long Input-Output Sequences 

**Title (ZH)**: LouisKV: 高效的键值对缓存检索用于长输入输出序列 

**Authors**: Wenbo Wu, Qingyi Si, Xiurui Pan, Ye Wang, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11292)  

**Abstract**: While Key-Value (KV) cache succeeds in reducing redundant computations in auto-regressive models, it introduces significant memory overhead, limiting its practical deployment in long-sequence scenarios. Existing KV retrieval methods mitigate this by dynamically retaining only a subset of KV entries on the GPU. However, they still suffer from notable efficiency and accuracy bottlenecks due to per-token retrieval and coarse-grained page-level KV management, especially in long-output reasoning scenarios. With the emergence of large reasoning models, efficiently handling such scenarios has become increasingly important. To address this issue, we present two key observations: (1) critical KVs exhibit strong temporal locality during decoding, and (2) these KVs exhibit distinct distribution patterns across the input prompt and generated output. Building on these observations, we propose LouisKV, an efficient KV cache retrieval framework designed for various long-sequence scenarios. Specifically, LouisKV introduces a semantic-aware retrieval strategy leveraging temporal locality to trigger retrieval only at semantic boundaries, drastically reducing computation and data transfer overhead. LouisKV also designs a decoupled, fine-grained management scheme that tailors differentiated strategies for input and output sequences to create retrieval units that better match the model's attention patterns, enabling precise identification of critical KVs. Furthermore, to boost efficiency, LouisKV incorporates several kernel-level optimizations, including custom Triton and CUDA kernels to accelerate the KV clustering and retrieval. Evaluations show that LouisKV achieves up to 4.7$\times$ speedup over state-of-the-art KV retrieval methods while maintaining near-lossless accuracy across diverse long-sequence tasks, including long-input short-output, short-input long-output, and long-input long-output scenarios. 

**Abstract (ZH)**: LouisKV：高效的关键值缓存检索框架 

---
# Towards Real-Time Fake News Detection under Evidence Scarcity 

**Title (ZH)**: 面向证据稀缺下的实时假新闻检测 

**Authors**: Guangyu Wei, Ke Han, Yueming Lyu, Yu Luo, Yue Jiang, Caifeng Shan, Nicu Sebe  

**Link**: [PDF](https://arxiv.org/pdf/2510.11277)  

**Abstract**: Fake news detection becomes particularly challenging in real-time scenarios, where emerging events often lack sufficient supporting evidence. Existing approaches often rely heavily on external evidence and therefore struggle to generalize under evidence scarcity. To address this issue, we propose Evaluation-Aware Selection of Experts (EASE), a novel framework for real-time fake news detection that dynamically adapts its decision-making process according to the assessed sufficiency of available evidence. EASE introduces a sequential evaluation mechanism comprising three independent perspectives: (1) Evidence-based evaluation, which assesses evidence and incorporates it into decision-making only when the evidence is sufficiently supportive; (2) Reasoning-based evaluation, which leverages the world knowledge of large language models (LLMs) and applies them only when their reliability is adequately established; and (3) Sentiment-based fallback, which integrates sentiment cues when neither evidence nor reasoning is reliable. To enhance the accuracy of evaluation processes, EASE employs instruction tuning with pseudo labels to guide each evaluator in justifying its perspective-specific knowledge through interpretable reasoning. Furthermore, the expert modules integrate the evaluators' justified assessments with the news content to enable evaluation-aware decision-making, thereby enhancing overall detection accuracy. Moreover, we introduce RealTimeNews-25, a new benchmark comprising recent news for evaluating model generalization on emerging news with limited evidence. Extensive experiments demonstrate that EASE not only achieves state-of-the-art performance across multiple benchmarks, but also significantly improves generalization to real-time news. The code and dataset are available: this https URL. 

**Abstract (ZH)**: 实时场景中虚假新闻检测尤其具有挑战性，其中新兴事件往往缺乏足够的支持证据。现有方法经常依赖外部证据，因此在证据稀缺的情况下难以泛化。为解决这一问题，我们提出了一种新的实时虚假新闻检测框架Evaluation-Aware Selection of Experts (EASE)，该框架根据可用证据的充分性动态调整其决策过程。EASE引入了一种序列评估机制，包括三个独立视角：（1）基于证据的评估，仅当证据足够支持时才评估证据并将其纳入决策；（2）基于推理的评估，利用大语言模型的世界知识并在其可靠性得到充分验证时使用；（3）基于情感的后备，当证据和推理均不可靠时将其纳入评估。为提高评估过程的准确性，EASE采用伪标签调优指令来引导每个评估者通过可解释的推理来证明其特定知识。此外，专家模块将评估者的合理评估与新闻内容结合起来，实现评估意识下的决策，从而提高整体检测准确性。此外，我们引入了RealTimeNews-25，这是一种新的基准数据集，包含近期新闻以评估模型在有限证据的新兴新闻场景下的泛化能力。大量实验表明，EASE不仅在多个基准测试中达到了最先进的性能，还在实时新闻泛化方面显著改进。代码和数据集可在以下链接获取：this https URL。 

---
# From Prompts to Packets: A View from the Network on ChatGPT, Copilot, and Gemini 

**Title (ZH)**: 从提示到包：网络视角下的ChatGPT、Copilot和Gemini 

**Authors**: Antonio Montieri, Alfredo Nascita, Antonio Pescapè  

**Link**: [PDF](https://arxiv.org/pdf/2510.11269)  

**Abstract**: Generative AI (GenAI) chatbots are now pervasive in digital ecosystems, yet their network traffic remains largely underexplored. This study presents an in-depth investigation of traffic generated by three leading chatbots (ChatGPT, Copilot, and Gemini) when accessed via Android mobile apps for both text and image generation. Using a dedicated capture architecture, we collect and label two complementary workloads: a 60-hour generic dataset with unconstrained prompts, and a controlled dataset built from identical prompts across GenAI apps and replicated via conventional messaging apps to enable one-to-one comparisons. This dual design allows us to address practical research questions on the distinctiveness of GenAI traffic, its differences from widely deployed traffic categories, and its novel implications for network usage. To this end, we provide fine-grained traffic characterization at trace, flow, and protocol levels, and model packet-sequence dynamics with Multimodal Markov Chains. Our analyses reveal app- and content-specific traffic patterns, particularly in volume, uplink/downlink profiles, and protocol adoption. We highlight the predominance of TLS, with Gemini extensively leveraging QUIC, ChatGPT exclusively using TLS 1.3, and app- and content-specific Server Name Indication (SNI) values. A payload-based occlusion analysis quantifies SNI's contribution to classification: masking it reduces F1-score by up to 20 percentage points in GenAI app traffic classification. Finally, compared with conventional messaging apps when carrying the same content, GenAI chatbots exhibit unique traffic characteristics, highlighting new stress factors for mobile networks, such as sustained upstream activity, with direct implications for network monitoring and management. We publicly release the datasets to support reproducibility and foster extensions to other use cases. 

**Abstract (ZH)**: Generative AI对话机器人如今已在数字生态系统中无处不在，但其网络流量仍 largely未被充分探索。本研究对通过 Android 移动应用程序访问的三款领先对话机器人（ChatGPT、Copilot 和 Gemini）进行文本和图像生成时产生的流量进行了深入调查。利用专用捕获架构，我们收集并标注了两个互补的工作负载：一个包含未加约束的提示的 60 小时通用数据集，以及一个基于 GenAI 应用程序中相同提示构建并使用常规消息应用程序复制以进行一对一比较的受控数据集。这种双设计使我们能够探讨生成式 AI 流量的独特性、其与广泛部署的流量类别的差异以及其对网络使用的新见解。为此，我们在轨迹、流和协议级别提供了详细的流量表征，并使用多模态马尔可夫链建模数据包序列动力学。我们的分析揭示了应用和内容特定的流量模式，特别是在流量volume、上行/下行配置文件和协议采用方面。我们强调了 TLS 的主导地位，其中 Gemini 广泛使用 QUIC，ChatGPT 仅使用 TLS 1.3，以及应用和内容特定的 Server Name Indication (SNI) 值。基于载荷的 occlusion 分析量化了 SNI 对分类的贡献：在生成式 AI 应用程序流量分类中，隐藏它会导致 F1 分数最多减少 20 个百分点。最后，与传输相同内容的常规消息应用程序相比，生成式 AI 对话机器人表现出独特的流量特征，突出了移动网络中新的压力因素，如持续的上行活动，这对网络监控和管理具有直接的含义。我们公开发布了数据集以支持可重复性并促进其他用例的发展。 

---
# Nepali Sign Language Characters Recognition: Dataset Development and Deep Learning Approaches 

**Title (ZH)**: 尼泊尔手语字符识别：数据集开发与深度学习方法 

**Authors**: Birat Poudel, Satyam Ghimire, Sijan Bhattarai, Saurav Bhandari, Suramya Sharma Dahal  

**Link**: [PDF](https://arxiv.org/pdf/2510.11243)  

**Abstract**: Sign languages serve as essential communication systems for individuals with hearing and speech impairments. However, digital linguistic dataset resources for underrepresented sign languages, such as Nepali Sign Language (NSL), remain scarce. This study introduces the first benchmark dataset for NSL, consisting of 36 gesture classes with 1,500 samples per class, designed to capture the structural and visual features of the language. To evaluate recognition performance, we fine-tuned MobileNetV2 and ResNet50 architectures on the dataset, achieving classification accuracies of 90.45% and 88.78%, respectively. These findings demonstrate the effectiveness of convolutional neural networks in sign recognition tasks, particularly within low-resource settings. To the best of our knowledge, this work represents the first systematic effort to construct a benchmark dataset and assess deep learning approaches for NSL recognition, highlighting the potential of transfer learning and fine-tuning for advancing research in underexplored sign languages. 

**Abstract (ZH)**: 手语作为听力和言语障碍个体的重要沟通系统，发挥着关键作用。然而，包括尼泊尔手语（NSL）在内的未充分代表的手语数字语言数据集仍然稀缺。本研究介绍了首个NSL基准数据集，包含36个手势类别，每类别1500个样本，旨在捕捉该语言的结构和视觉特征。为评估识别性能，我们在数据集上微调了MobileNetV2和ResNet50架构，分别获得了90.45%和88.78%的分类准确率。这些发现证明了卷积神经网络在手语识别任务中的有效性，特别是在低资源环境中。据我们所知，本工作代表了首个系统地构建基准数据集并评估深度学习方法进行NSL识别的研究，突显了迁移学习和微调在推进未充分探索的手语研究中的潜力。 

---
# Attacks by Content: Automated Fact-checking is an AI Security Issue 

**Title (ZH)**: 内容攻击：自动化事实核查是AI安全问题 

**Authors**: Michael Schlichtkrull  

**Link**: [PDF](https://arxiv.org/pdf/2510.11238)  

**Abstract**: When AI agents retrieve and reason over external documents, adversaries can manipulate the data they receive to subvert their behaviour. Previous research has studied indirect prompt injection, where the attacker injects malicious instructions. We argue that injection of instructions is not necessary to manipulate agents - attackers could instead supply biased, misleading, or false information. We term this an attack by content. Existing defenses, which focus on detecting hidden commands, are ineffective against attacks by content. To defend themselves and their users, agents must critically evaluate retrieved information, corroborating claims with external evidence and evaluating source trustworthiness. We argue that this is analogous to an existing NLP task, automated fact-checking, which we propose to repurpose as a cognitive self-defense tool for agents. 

**Abstract (ZH)**: 当AI代理检索和推理外部文档时，对手可以通过操纵接收到的数据来颠覆其行为。以往的研究集中在间接提示注入上，攻击者在其中注入恶意指令。我们认为，注入指令并不是操纵代理所必需的——攻击者可以通过提供带有偏见、误导或虚假信息来实现目标。我们将这种攻击称为内容攻击。现有的防御措施专注于检测隐藏的命令，但对于内容攻击效果不佳。为了保护自身和用户，代理必须批判性地评估检索到的信息，通过外部证据 corroborate 声称，并评估信息源的可信度。我们认为，这类似于现有的NLP任务——自动化事实核查——我们建议将这一任务重新利用为代理的认知自我防御工具。 

---
# LightPneumoNet: Lightweight Pneumonia Classifier 

**Title (ZH)**: LightPneumoNet: 轻量级肺炎分类器 

**Authors**: Neilansh Chauhan, Piyush Kumar Gupta, Faraz Doja  

**Link**: [PDF](https://arxiv.org/pdf/2510.11232)  

**Abstract**: Effective pneumonia diagnosis is often challenged by the difficulty of deploying large, computationally expensive deep learning models in resource-limited settings. This study introduces LightPneumoNet, an efficient, lightweight convolutional neural network (CNN) built from scratch to provide an accessible and accurate diagnostic solution for pneumonia detection from chest X-rays. Our model was trained on a public dataset of 5,856 chest X-ray images. Preprocessing included image resizing to 224x224, grayscale conversion, and pixel normalization, with data augmentation (rotation, zoom, shear) to prevent overfitting. The custom architecture features four blocks of stacked convolutional layers and contains only 388,082 trainable parameters, resulting in a minimal 1.48 MB memory footprint. On the independent test set, our model delivered exceptional performance, achieving an overall accuracy of 0.942, precision of 0.92, and an F1-Score of 0.96. Critically, it obtained a sensitivity (recall) of 0.99, demonstrating a near-perfect ability to identify true pneumonia cases and minimize clinically significant false negatives. Notably, LightPneumoNet achieves this high recall on the same dataset where existing approaches typically require significantly heavier architectures or fail to reach comparable sensitivity levels. The model's efficiency enables deployment on low-cost hardware, making advanced computer-aided diagnosis accessible in underserved clinics and serving as a reliable second-opinion tool to improve patient outcomes. 

**Abstract (ZH)**: 轻量级肺部感染诊断网络：面向资源受限环境的高效卷积神经网络 

---
# Fairness Metric Design Exploration in Multi-Domain Moral Sentiment Classification using Transformer-Based Models 

**Title (ZH)**: 多域道德情感分类中基于Transformer模型的公平性指标设计探索 

**Authors**: Battemuulen Naranbat, Seyed Sahand Mohammadi Ziabari, Yousuf Nasser Al Husaini, Ali Mohammed Mansoor Alsahag  

**Link**: [PDF](https://arxiv.org/pdf/2510.11222)  

**Abstract**: Ensuring fairness in natural language processing for moral sentiment classification is challenging, particularly under cross-domain shifts where transformer models are increasingly deployed. Using the Moral Foundations Twitter Corpus (MFTC) and Moral Foundations Reddit Corpus (MFRC), this work evaluates BERT and DistilBERT in a multi-label setting with in-domain and cross-domain protocols. Aggregate performance can mask disparities: we observe pronounced asymmetry in transfer, with Twitter->Reddit degrading micro-F1 by 14.9% versus only 1.5% for Reddit->Twitter. Per-label analysis reveals fairness violations hidden by overall scores; notably, the authority label exhibits Demographic Parity Differences of 0.22-0.23 and Equalized Odds Differences of 0.40-0.41. To address this gap, we introduce the Moral Fairness Consistency (MFC) metric, which quantifies the cross-domain stability of moral foundation detection. MFC shows strong empirical validity, achieving a perfect negative correlation with Demographic Parity Difference (rho = -1.000, p < 0.001) while remaining independent of standard performance metrics. Across labels, loyalty demonstrates the highest consistency (MFC = 0.96) and authority the lowest (MFC = 0.78). These findings establish MFC as a complementary, diagnosis-oriented metric for fairness-aware evaluation of moral reasoning models, enabling more reliable deployment across heterogeneous linguistic contexts. . 

**Abstract (ZH)**: 确保自然语言处理在道德情感分类中的公平性尤其是在跨域转移时具有挑战性：基于Moral Foundations Twitter Corpus和Moral Foundations Reddit Corpus的BERT和DistilBERT多标签评估 

---
# G2L:From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Knowledge Distillation 

**Title (ZH)**: G2L:从 gigascale 到癌症特异性大规模病理基础模型的知识蒸馏 

**Authors**: Yesung Cho, Sungmin Lee, Geongyu Lee, Minkyung Lee, Jongbae Park, Dongmyung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2510.11176)  

**Abstract**: Recent studies in pathology foundation models have shown that scaling training data, diversifying cancer types, and increasing model size consistently improve their performance. However, giga-scale foundation models, which are trained on hundreds of thousands of slides covering tens of cancer types and contain billions of parameters, pose significant challenges for practical use due to their tremendous computational costs in both development and deployment. In this work, we present a novel strategy, named the G2L framework, to increase the performance of large-scale foundation models, which consist of only $15\%$ of the parameters of giga-scale models, to a comparable performance level of giga-scale models in cancer-specific tasks. Our approach applies knowledge distillation, transferring the capabilities of a giga-scale model to a large-scale model, using just 1K pathology slides of a target cancer (e.g., breast, prostate, etc.). The resulting distilled model not only outperformed state-of-the-art models of the same size (i.e., large-scale) across several benchmarks but also, interestingly, surpassed the giga-scale teacher and huge-scale models in some benchmarks. In addition, the distilled model exhibited a higher robustness index, indicating improved resilience to image variations originating from multiple institutions. These findings suggest that the proposed distillation approach for a large-scale model is a data- and parameter-efficient way to achieve giga-scale-level performance for cancer-specific applications without prohibitive computational burden. 

**Abstract (ZH)**: 近期病理基础模型的研究表明，扩展训练数据、多样化癌症类型和增加模型规模能够一致地提高其性能。然而，由数百数千张病理切片覆盖数十种癌症类型、包含数十亿参数的巨型规模基础模型由于其巨大的开发和部署计算成本，给实际应用带来了显著挑战。本文提出了一种新的策略，即G2L框架，该策略能够将仅包含巨型规模模型15%参数量的大规模基础模型在癌症特异性任务中的性能提升至与巨型规模模型媲美的水平。我们的方法通过知识蒸馏，仅使用目标癌症（如乳腺癌、前列腺癌等）的1000张病理切片，将巨型规模模型的能力转移到大规模模型上。蒸馏模型不仅在多个基准测试中超越了相同规模（即大规模）的最先进的模型，而且在某些基准测试中甚至超过了巨型规模教师模型和超大规模模型。此外，蒸馏模型还展示了更高的鲁棒性指数，表明其对来自多个机构的图像变异具有更强的抗御能力。这些发现表明，提出的对大规模模型的知识蒸馏方法是一种在不带来巨大的计算负担的情况下，以数据和参数高效的方式实现巨型规模性能水平的方法，适用于癌症特异性应用。 

---
# EAGER: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling 

**Title (ZH)**: EAGER: 熵意识的生成模型以适应推理时扩展需求 

**Authors**: Daniel Scalena, Leonidas Zotos, Elisabetta Fersini, Malvina Nissim, Ahmet Üstün  

**Link**: [PDF](https://arxiv.org/pdf/2510.11170)  

**Abstract**: With the rise of reasoning language models and test-time scaling methods as a paradigm for improving model performance, substantial computation is often required to generate multiple candidate sequences from the same prompt. This enables exploration of different reasoning paths toward the correct solution, however, allocates the same compute budget for each prompt. Grounded on the assumption that different prompts carry different degrees of complexity, and thus different computation needs, we propose EAGer, a training-free generation method that leverages model uncertainty through token-wise entropy distribution to reduce redundant computation and concurrently improve overall performance. EAGer allows branching to multiple reasoning paths only in the presence of high-entropy tokens, and then reallocates the saved compute budget to the instances where exploration of alternative paths is most needed. We find that across multiple open-source models on complex reasoning benchmarks such as AIME 2025, EAGer can reallocate the budget without accessing target labels, achieving the best efficiency-performance trade-off in terms of reasoning length and Pass@k. When target labels are accessible, EAGer generates up to 65% fewer tokens (hence saving compute) and achieves up to 37% improvement in Pass@k compared to the Full Parallel Sampling. 

**Abstract (ZH)**: 基于熵分布的无需训练生成方法EAGer：在复杂推理基准上的计算预算重新分配 

---
# One Size Does Not Fit All: Exploring Variable Thresholds for Distance-Based Multi-Label Text Classification 

**Title (ZH)**: 一场不变：探索基于距离的多标签文本分类的变异性阈值 

**Authors**: Jens Van Nooten, Andriy Kosar, Guy De Pauw, Walter Daelemans  

**Link**: [PDF](https://arxiv.org/pdf/2510.11160)  

**Abstract**: Distance-based unsupervised text classification is a method within text classification that leverages the semantic similarity between a label and a text to determine label relevance. This method provides numerous benefits, including fast inference and adaptability to expanding label sets, as opposed to zero-shot, few-shot, and fine-tuned neural networks that require re-training in such cases. In multi-label distance-based classification and information retrieval algorithms, thresholds are required to determine whether a text instance is "similar" to a label or query. Similarity between a text and label is determined in a dense embedding space, usually generated by state-of-the-art sentence encoders. Multi-label classification complicates matters, as a text instance can have multiple true labels, unlike in multi-class or binary classification, where each instance is assigned only one label. We expand upon previous literature on this underexplored topic by thoroughly examining and evaluating the ability of sentence encoders to perform distance-based classification. First, we perform an exploratory study to verify whether the semantic relationships between texts and labels vary across models, datasets, and label sets by conducting experiments on a diverse collection of realistic multi-label text classification (MLTC) datasets. We find that similarity distributions show statistically significant differences across models, datasets and even label sets. We propose a novel method for optimizing label-specific thresholds using a validation set. Our label-specific thresholding method achieves an average improvement of 46% over normalized 0.5 thresholding and outperforms uniform thresholding approaches from previous work by an average of 14%. Additionally, the method demonstrates strong performance even with limited labeled examples. 

**Abstract (ZH)**: 基于距离的无监督文本分类是一种文本分类方法，它通过利用标签与文本之间的语义相似度来确定标签的相关性。该方法提供了诸多好处，包括快速推理和对扩展标签集的高度适应性，而零样本、少样本和细调的神经网络在这些情况下需要重新训练。在多标签距离基于分类和信息检索算法中，需要设置阈值来确定文本实例是否与标签或查询“相似”。文本与标签之间的相似性通常在密集嵌入空间中确定，通常由最先进的句子编码器生成。多标签分类使情况更加复杂，因为一个文本实例可以有多个真实标签，而在多类或二分类中，每个实例仅被分配一个标签。我们通过详细研究和评估句子编码器在距离基于分类中的能力，扩展了对该未充分研究主题的先前文献。首先，我们进行一项探索性研究，通过在一系列现实多标签文本分类（MLTC）数据集上进行实验，验证文本与标签之间的语义关系是否随模型、数据集和标签集的变化而变化。我们发现相似性分布显示出统计上的显著差异，不仅在模型之间，甚至在标签集之间也是如此。我们提出了一种使用验证集优化标签特定阈值的新方法。我们的标签特定阈值方法在平均上比归一化的0.5阈值提高了46%的性能，并且在以往工作中的均匀阈值方法的基础上平均提高了14%的表现。此外，即使在有限的带标签示例情况下，该方法仍然表现出较强的性能。 

---
# HoMer: Addressing Heterogeneities by Modeling Sequential and Set-wise Contexts for CTR Prediction 

**Title (ZH)**: HoMer: 通过建模顺序和集wise上下文来解决异质性进行点击率预测 

**Authors**: Shuwei Chen, Jiajun Cui, Zhengqi Xu, Fan Zhang, Jiangke Fan, Teng Zhang, Xingxing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.11100)  

**Abstract**: Click-through rate (CTR) prediction, which models behavior sequence and non-sequential features (e.g., user/item profiles or cross features) to infer user interest, underpins industrial recommender systems. However, most methods face three forms of heterogeneity that degrade predictive performance: (i) Feature Heterogeneity persists when limited sequence side features provide less granular interest representation compared to extensive non-sequential features, thereby impairing sequence modeling performance; (ii) Context Heterogeneity arises because a user's interest in an item will be influenced by other items, yet point-wise prediction neglects cross-item interaction context from the entire item set; (iii) Architecture Heterogeneity stems from the fragmented integration of specialized network modules, which compounds the model's effectiveness, efficiency and scalability in industrial deployments. To tackle the above limitations, we propose HoMer, a Homogeneous-Oriented TransforMer for modeling sequential and set-wise contexts. First, we align sequence side features with non-sequential features for accurate sequence modeling and fine-grained interest representation. Second, we shift the prediction paradigm from point-wise to set-wise, facilitating cross-item interaction in a highly parallel manner. Third, HoMer's unified encoder-decoder architecture achieves dual optimization through structural simplification and shared computation, ensuring computational efficiency while maintaining scalability with model size. Without arduous modification to the prediction pipeline, HoMer successfully scales up and outperforms our industrial baseline by 0.0099 in the AUC metric, and enhances online business metrics like CTR/RPM by 1.99%/2.46%. Additionally, HoMer saves 27% of GPU resources via preliminary engineering optimization, further validating its superiority and practicality. 

**Abstract (ZH)**: 基于同质化的TransforMer进行序列和集合理境建模 

---
# Causal Disentanglement Learning for Accurate Anomaly Detection in Multivariate Time Series 

**Title (ZH)**: 多变量时间序列中准确异常检测的因果去纠缠学习 

**Authors**: Wonah Kim, Jeonghyeon Park, Dongsan Jun, Jungkyu Han, Sejin Chun  

**Link**: [PDF](https://arxiv.org/pdf/2510.11084)  

**Abstract**: Disentangling complex causal relationships is important for accurate detection of anomalies. In multivariate time series analysis, dynamic interactions among data variables over time complicate the interpretation of causal relationships. Traditional approaches assume statistical independence between variables in unsupervised settings, whereas recent methods capture feature correlations through graph representation learning. However, their representations fail to explicitly infer the causal relationships over different time periods. To solve the problem, we propose Causally Disentangled Representation Learning for Anomaly Detection (CDRL4AD) to detect anomalies and identify their causal relationships in multivariate time series. First, we design the causal process as model input, the temporal heterogeneous graph, and causal relationships. Second, our representation identifies causal relationships over different time periods and disentangles latent variables to infer the corresponding causal factors. Third, our experiments on real-world datasets demonstrate that CDRL4AD outperforms state-of-the-art methods in terms of accuracy and root cause analysis. Fourth, our model analysis validates hyperparameter sensitivity and the time complexity of CDRL4AD. Last, we conduct a case study to show how our approach assists human experts in diagnosing the root causes of anomalies. 

**Abstract (ZH)**: 因果关系分离的表示学习在异常检测中的应用（CDRL4AD） 

---
# Temporal Alignment Guidance: On-Manifold Sampling in Diffusion Models 

**Title (ZH)**: 时间对齐指导：扩散模型中的流形采样 

**Authors**: Youngrok Park, Hojung Jung, Sangmin Bae, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2510.11057)  

**Abstract**: Diffusion models have achieved remarkable success as generative models. However, even a well-trained model can accumulate errors throughout the generation process. These errors become particularly problematic when arbitrary guidance is applied to steer samples toward desired properties, which often breaks sample fidelity. In this paper, we propose a general solution to address the off-manifold phenomenon observed in diffusion models. Our approach leverages a time predictor to estimate deviations from the desired data manifold at each timestep, identifying that a larger time gap is associated with reduced generation quality. We then design a novel guidance mechanism, `Temporal Alignment Guidance' (TAG), attracting the samples back to the desired manifold at every timestep during generation. Through extensive experiments, we demonstrate that TAG consistently produces samples closely aligned with the desired manifold at each timestep, leading to significant improvements in generation quality across various downstream tasks. 

**Abstract (ZH)**: 扩散模型作为生成模型取得了显著的成功。然而，即使训练良好的模型在生成过程中也会累积误差。当对生成样本施加任意指导以引导其朝向期望的特性时，这些误差往往会破坏生成样本的保真度。在本文中，我们提出了一个通用解决方案来解决观察到的扩散模型中的离流现象。我们的方法利用时间预测器在每个时间步估计与期望数据流形的偏差，发现更大的时间间隔与较低的生成质量相关。随后，我们设计了一种新颖的指导机制，即“时间对齐指导”（TAG），在生成的每个时间步将样本吸引回期望的流形。通过广泛的实验，我们证明TAG能够在每个时间步持续产生与期望流形高度对齐的样本，从而在各种下游任务中显著提高生成质量。 

---
# DeepResearchGuard: Deep Research with Open-Domain Evaluation and Multi-Stage Guardrails for Safety 

**Title (ZH)**: DeepResearchGuard：基于开放域评估和多阶段防护栏的安全深度研究 

**Authors**: Wei-Chieh Huang, Henry Peng Zou, Yaozu Wu, Dongyuan Li, Yankai Chen, Weizhi Zhang, Yangning Li, Angelo Zangari, Jizhou Guo, Chunyu Miao, Liancheng Fang, Langzhou He, Renhe Jiang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10994)  

**Abstract**: Deep research frameworks have shown promising capabilities in synthesizing comprehensive reports from web sources. While deep research possesses significant potential to address complex issues through planning and research cycles, existing frameworks are deficient in sufficient evaluation procedures and stage-specific protections. They typically treat evaluation as exact match accuracy of question-answering, but overlook crucial aspects of report quality such as credibility, coherence, breadth, depth, and safety. This oversight may result in hazardous or malicious sources being integrated into the final report. To address these issues, we introduce DEEPRESEARCHGUARD, a comprehensive framework featuring four-stage safeguards with open-domain evaluation of references and reports. We assess performance across multiple metrics, e.g., defense success rate and over-refusal rate, and five key report dimensions. In the absence of a suitable safety benchmark, we introduce DRSAFEBENCH, a stage-wise benchmark for deep research safety. Our evaluation spans diverse state-of-the-art LLMs, including GPT-4o, Gemini-2.5-flash, DeepSeek-v3, and o4-mini. DEEPRESEARCHGUARD achieves an average defense success rate improvement of 18.16% while reducing over-refusal rate by 6%. The input guard provides the most substantial early-stage protection by filtering out obvious risks, while the plan and research guards enhance citation discipline and source credibility. Through extensive experiments, we show that DEEPRESEARCHGUARD enables comprehensive open-domain evaluation and stage-aware defenses that effectively block harmful content propagation, while systematically improving report quality without excessive over-refusal rates. The code can be found via this https URL. 

**Abstract (ZH)**: 深度研究框架在从网络来源合成综合性报告方面展现了有前景的能力。尽管深度研究通过规划和研究周期在解决复杂问题方面具备显著潜力，现有的框架缺乏足够的评估程序和阶段特定保护。它们通常将评估视为问题回答的精确匹配准确率，但忽略了报告质量的关键方面，如可信度、连贯性、广度、深度和安全性。这种疏忽可能导致危险或恶意来源被集成到最终报告中。为了解决这些问题，我们提出了一种名为DEEPRESEARCHGUARD的综合框架，该框架包含四阶段保护措施，并对引文和报告进行开放式领域评估。我们在多个指标上评估性能，包括防御成功率、过度拒绝率，以及五个关键报告维度。在缺乏合适的安全性基准的情况下，我们引入了DRSAFEBENCH，一种阶段性的深度研究安全性基准。我们的评估涵盖了包括GPT-4o、Gemini-2.5-flash、DeepSeek-v3和o4-mini在内的多种最新LLMs。DEEPRESEARCHGUARD实现了平均防御成功率提高18.16%，过度拒绝率减少6%的效果。输入保护提供了最显著的早期阶段保护，通过过滤掉明显的风险，而计划和研究保护增强了引文纪律和来源可信度。通过大量实验，我们展示了DEEPRESEARCHGUARD能够实现综合的开放式领域评估和阶段感知防御，有效阻止有害内容的传播，同时系统地提高报告质量，而不增加过度拒绝率。代码可以通过这个链接访问。 

---
# Catch-Only-One: Non-Transferable Examples for Model-Specific Authorization 

**Title (ZH)**: 唯一捕获：模型特定授权的非迁移性示例 

**Authors**: Zihan Wang, Zhiyong Ma, Zhongkui Ma, Shuofeng Liu, Akide Liu, Derui Wang, Minhui Xue, Guangdong Bai  

**Link**: [PDF](https://arxiv.org/pdf/2510.10982)  

**Abstract**: Recent AI regulations call for data that remain useful for innovation while resistant to misuse, balancing utility with protection at the model level. Existing approaches either perturb data to make it unlearnable or retrain models to suppress transfer, but neither governs inference by unknown models, and both typically require control over training. We propose non-transferable examples (NEs), a training-free and data-agnostic input-side usage-control mechanism. We recode inputs within a model-specific low-sensitivity subspace, preserving outputs for the authorized model while reducing performance on unauthorized models through subspace misalignment. We establish formal bounds that guarantee utility for the authorized model and quantify deviation for unauthorized ones, with the Hoffman-Wielandt inequality linking degradation to spectral differences. Empirically, NEs retain performance on diverse vision backbones and state-of-the-art vision-language models under common preprocessing, whereas non-target models collapse even with reconstruction attempts. These results establish NEs as a practical means to preserve intended data utility while preventing unauthorized exploitation. Our project is available at this https URL 

**Abstract (ZH)**: 近期的AI法规要求数据既有利于创新又能够抵御滥用，在保持模型实用性的同时提供保护。现有的方法要么通过扰动数据使其难以学习，要么重新训练模型以抑制迁移，但它们都未能控制未知模型的推理行为，且通常需要对训练过程进行控制。我们提出非可迁移示例（NEs），这是一种无需训练和数据无关的输入端使用控制机制。我们通过对模型特定低敏感子空间内的输入进行编码，保留授权模型的输出，同时通过对未经授权模型的子空间对齐来降低其性能。我们建立形式上的界限以确保授权模型的实用性，并通过度量未经授权模型的偏差进行量化，其中霍夫曼-魏兰特不等式将下降与谱差异联系起来。实验表明，NEs在常见的预处理下可以在多种视觉骨干网络和最先进的视觉-语言模型中保持性能，而未经授权的模型即使在重建尝试后也无法保持性能。这些结果证明了NEs作为一种实用手段，能够同时保持预期数据的实用性和防止未经授权的利用。我们的项目可在以下网址访问：这个 https URL 

---
# RV-HATE: Reinforced Multi-Module Voting for Implicit Hate Speech Detection 

**Title (ZH)**: RV-HATE: 强化多模块投票的隐含仇恨言论检测 

**Authors**: Yejin Lee, Hyeseon Ahn, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.10971)  

**Abstract**: Hate speech remains prevalent in human society and continues to evolve in its forms and expressions. Modern advancements in internet and online anonymity accelerate its rapid spread and complicate its detection. However, hate speech datasets exhibit diverse characteristics primarily because they are constructed from different sources and platforms, each reflecting different linguistic styles and social contexts. Despite this diversity, prior studies on hate speech detection often rely on fixed methodologies without adapting to data-specific features. We introduce RV-HATE, a detection framework designed to account for the dataset-specific characteristics of each hate speech dataset. RV-HATE consists of multiple specialized modules, where each module focuses on distinct linguistic or contextual features of hate speech. The framework employs reinforcement learning to optimize weights that determine the contribution of each module for a given dataset. A voting mechanism then aggregates the module outputs to produce the final decision. RV-HATE offers two primary advantages: (1)~it improves detection accuracy by tailoring the detection process to dataset-specific attributes, and (2)~it also provides interpretable insights into the distinctive features of each dataset. Consequently, our approach effectively addresses implicit hate speech and achieves superior performance compared to conventional static methods. Our code is available at this https URL. 

**Abstract (ZH)**: 仇恨言论在人类社会中依然普遍存在，并且其形式和表达方式不断演变。互联网和在线匿名性的现代进步加速了其传播速度并使其检测变得更加复杂。尽管如此，仇恨言论数据集表现出多样化的特征，主要因为它们源自不同的来源和平台，每个平台反映了不同的语言风格和社会背景。尽管存在这些多样性，但之前关于仇恨言论检测的研究通常依赖于固定的检测方法，没有根据数据特异性进行调整。我们提出了RV-HATE，这是一种旨在考虑每个仇恨言论数据集特定特征的检测框架。RV-HATE 包含多个专门的模块，每个模块专注于仇恨言论的不同语言或背景特征。该框架使用强化学习来优化决定每个模块对给定数据集贡献权重的方法。然后通过投票机制将模块输出聚合为最终决策。RV-HATE 提供了两个主要优点：（1）它通过根据数据集特定属性定制检测过程来提高检测准确性；（2）它还提供了对每个数据集独特特征的可解释洞察。因此，我们的方法有效地解决了隐含的仇恨言论，并在性能上优于传统的静态方法。我们的代码可在此处获得。 

---
# Redundancy as a Structural Information Principle for Learning and Generalization 

**Title (ZH)**: 冗余作为学习和泛化的结构信息原则 

**Authors**: Yuda Bi, Ying Zhu, Vince D Calhoun  

**Link**: [PDF](https://arxiv.org/pdf/2510.10938)  

**Abstract**: We present a theoretical framework that extends classical information theory to finite and structured systems by redefining redundancy as a fundamental property of information organization rather than inefficiency. In this framework, redundancy is expressed as a general family of informational divergences that unifies multiple classical measures, such as mutual information, chi-squared dependence, and spectral redundancy, under a single geometric principle. This reveals that these traditional quantities are not isolated heuristics but projections of a shared redundancy geometry. The theory further predicts that redundancy is bounded both above and below, giving rise to an optimal equilibrium that balances over-compression (loss of structure) and over-coupling (collapse). While classical communication theory favors minimal redundancy for transmission efficiency, finite and structured systems, such as those underlying real-world learning, achieve maximal stability and generalization near this equilibrium. Experiments with masked autoencoders are used to illustrate and verify this principle: the model exhibits a stable redundancy level where generalization peaks. Together, these results establish redundancy as a measurable and tunable quantity that bridges the asymptotic world of communication and the finite world of learning. 

**Abstract (ZH)**: 我们提出一个理论框架，将经典信息论扩展到有限和结构化的系统中，重新定义冗余作为信息组织的基本属性而非无效性。在这个框架中，冗余被表达为一套统一的信息差异的一般家族，这些差异统一了多个经典度量，如互信息、卡方依赖性和频谱冗余，在单一几何原理下。这揭示了这些传统量并不孤立，而是共享冗余几何学的投影。该理论进一步预测，冗余受到上下限的约束，产生一个最优平衡状态，平衡过度压缩（结构丧失）和过度耦合（坍塌）。虽然经典通信理论倾向于最小的冗余以提高传输效率，但在诸如现实世界学习所依托的有限和结构化系统中，接近这一平衡状态时会实现最大稳定性与泛化能力。用掩蔽自编码器的实验来说明并验证这一原理：模型显示一个稳定的冗余水平，在该水平上泛化能力达到峰值。总之，这些结果确立了冗余作为一个可测量和可调节的量，连接了通信领域的渐近世界和学习领域的有限世界。 

---
# Evaluating Language Models' Evaluations of Games 

**Title (ZH)**: 评估语言模型对游戏的评价 

**Authors**: Katherine M. Collins, Cedegao E. Zhang, Graham Todd, Lance Ying, Mauricio Barba da Costa, Ryan Liu, Prafull Sharma, Adrian Weller, Ionatan Kuperwajs, Lionel Wong, Joshua B. Tenenbaum, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2510.10930)  

**Abstract**: Reasoning is not just about solving problems -- it is also about evaluating which problems are worth solving at all. Evaluations of artificial intelligence (AI) systems primarily focused on problem solving, historically by studying how models play games such as chess and Go. In this paper, we advocate for a new paradigm that assesses AI systems' evaluation of games. First, we introduce a formalism for evaluating such evaluations. We then leverage a large-scale dataset of over $100$ novel board games and over 450 human judgments to compare evaluations produced by modern language and reasoning models against those of people and symbolic computational agents. We consider two kinds of evaluative queries: assessing the payoff (or fairness) and the funness of games. These queries span two dimensions relevant to the design of evaluations of AI evaluations: how complex a query is to compute and how difficult a query is to quantify. Our results show that reasoning models are generally more aligned to people in their evaluations of games than non-reasoning language models. However, we observe a non-monotonic relationship: as models get closer to game-theoretic optimal, their fit to human data weakens. We also observe more "jaggedness" across models for assessing funness, in line with the greater difficulty of quantifying this query. Across queries and games, reasoning models show highly variable and unpredictable resource usage when assessing queries, pointing to the importance of imbuing more resource-rational meta-reasoning in language and reasoning models. 

**Abstract (ZH)**: 推理不仅仅关于解决问题——它还关乎评估哪些问题是值得解决的。本文提倡一种新的范式，评估AI系统对游戏的评估。我们首先引入一种形式化方法来评估这些评估。然后，我们利用超过100种新型棋盘游戏和超过450个人类判断的大规模数据集，将现代语言和推理模型生成的游戏评估与人类和象征性计算代理的评估进行比较。我们考虑两类评估查询：评估收益（或公平性）和乐趣。这些查询涵盖了评估AI评估设计的两个维度：计算查询的复杂性和量化查询的难度。我们的结果表明，推理模型在评估游戏方面往往与人类的评估更一致，但观察到一种非单调关系：随着模型接近博弈论最优，它们与人类数据的契合度减弱。我们还观察到，在评估乐趣方面，不同模型的行为更加“锯齿状”，这与量化这一查询的难度更大相关。在各类查询和游戏上，推理模型在评估查询时显示出高度变性和不可预测的资源使用，这突显了在语言和推理模型中嵌入更多资源理性元推理的重要性。 

---
# Comparative Explanations via Counterfactual Reasoning in Recommendations 

**Title (ZH)**: 基于反事实推理的推荐系统中比较解释 

**Authors**: Yi Yu, Zhenxing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10920)  

**Abstract**: Explainable recommendation through counterfactual reasoning seeks to identify the influential aspects of items in recommendations, which can then be used as explanations. However, state-of-the-art approaches, which aim to minimize changes in product aspects while reversing their recommended decisions according to an aggregated decision boundary score, often lead to factual inaccuracies in explanations. To solve this problem, in this work we propose a novel method of Comparative Counterfactual Explanations for Recommendation (CoCountER). CoCountER creates counterfactual data based on soft swap operations, enabling explanations for recommendations of arbitrary pairs of comparative items. Empirical experiments validate the effectiveness of our approach. 

**Abstract (ZH)**: 通过反事实推理实现可解释推荐以识别推荐中具有影响力的项目方面，并将其用作解释。然而，最先进的方法在最小化产品方面变化的同时根据聚合的决策边界分数逆转其推荐决策，往往会导致解释中的事实不准确。为了解决这一问题，本文提出了一种名为Comparative Counterfactual Explanations for Recommendation (CoCountER) 的新型方法。CoCountER 通过软交换操作生成反事实数据，从而使解释能够应用于任意一对比较项目的推荐。实验证明了该方法的有效性。 

---
# LPCVAE: A Conditional VAE with Long-Term Dependency and Probabilistic Time-Frequency Fusion for Time Series Anomaly Detection 

**Title (ZH)**: LPCVAE：一种具备长期依赖关系和概率时频融合的条件VAE的时间序列异常检测方法 

**Authors**: Hanchang Cheng, Weimin Mu, Fan Liu, Weilin Zhu, Can Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.10915)  

**Abstract**: Time series anomaly detection(TSAD) is a critical task in signal processing field, ensuring the reliability of complex systems. Reconstruction-based methods dominate in TSAD. Among these methods, VAE-based methods have achieved promising results. Existing VAE-based methods suffer from the limitation of single-window feature and insufficient leveraging of long-term time and frequency information. We propose a Conditional Variational AutoEncoder with Long-term dependency and Probabilistic time-frequency fusion, named LPCVAE. LPCVAE introduces LSTM to capture long-term dependencies beyond windows. It further incorporates a Product-of-Experts (PoE) mechanism for adaptive and distribution-level probabilistic fusion. This design effectively mitigates time-frequency information loss. Extensive experiments on four public datasets demonstrate it outperforms state-of-the-art methods. The results confirm that integrating long-term time and frequency representations with adaptive fusion yields a robust and efficient solution for TSAD. 

**Abstract (ZH)**: 基于长期依赖和概率时间频率融合的条件变分自编码器在时间序列异常检测中的应用 

---
# Generative AI for Software Project Management: Insights from a Review of Software Practitioner Literature 

**Title (ZH)**: 生成式AI在软件项目管理中的应用：基于软件从业者文献的见解 

**Authors**: Lakshana Iruni Assalaarachchi, Zainab Masood, Rashina Hoda, John Grundy  

**Link**: [PDF](https://arxiv.org/pdf/2510.10887)  

**Abstract**: Software practitioners are discussing GenAI transformations in software project management openly and widely. To understand the state of affairs, we performed a grey literature review using 47 publicly available practitioner sources including blogs, articles, and industry reports. We found that software project managers primarily perceive GenAI as an "assistant", "copilot", or "friend" rather than as a "PM replacement", with support of GenAI in automating routine tasks, predictive analytics, communication and collaboration, and in agile practices leading to project success. Practitioners emphasize responsible GenAI usage given concerns such as hallucinations, ethics and privacy, and lack of emotional intelligence and human judgment. We present upskilling requirements for software project managers in the GenAI era mapped to the Project Management Institute's talent triangle. We share key recommendations for both practitioners and researchers. 

**Abstract (ZH)**: 软件从业人员公开广泛地讨论着GenAI在软件项目管理中的转型。为了了解现状，我们使用47份公开可用的从业人员来源（包括博客、文章和行业报告）进行了灰色文献综述。我们发现，软件项目管理人员主要将GenAI视为“助手”、“副驾驶”或“朋友”，而非“项目经理的替代者”。从业者指出，GenAI在自动化常规任务、预测分析、沟通协作以及敏捷实践中的支持有助于项目成功。考虑到幻觉、伦理和隐私等问题，以及缺乏情感智能和人类判断，从业者强调负责任地使用GenAI。我们根据项目管理协会的人才三角模型，提出了软件项目管理人员在GenAI时代的再技能培训需求，并分享了针对从业人员和研究者的关键建议。 

---
# HeroFilter: Adaptive Spectral Graph Filter for Varying Heterophilic Relations 

**Title (ZH)**: HeroFilter：自适应谱图滤波器用于变化的异质关系 

**Authors**: Shuaicheng Zhang, Haohui Wang, Junhong Lin, Xiaojie Guo, Yada Zhu, Si Zhang, Dongqi Fu, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.10864)  

**Abstract**: Graph heterophily, where connected nodes have different labels, has attracted significant interest recently. Most existing works adopt a simplified approach - using low-pass filters for homophilic graphs and high-pass filters for heterophilic graphs. However, we discover that the relationship between graph heterophily and spectral filters is more complex - the optimal filter response varies across frequency components and does not follow a strict monotonic correlation with heterophily degree. This finding challenges conventional fixed filter designs and suggests the need for adaptive filtering to preserve expressiveness in graph embeddings. Formally, natural questions arise: Given a heterophilic graph G, how and to what extent will the varying heterophily degree of G affect the performance of GNNs? How can we design adaptive filters to fit those varying heterophilic connections? Our theoretical analysis reveals that the average frequency response of GNNs and graph heterophily degree do not follow a strict monotonic correlation, necessitating adaptive graph filters to guarantee good generalization performance. Hence, we propose [METHOD NAME], a simple yet powerful GNN, which extracts information across the heterophily spectrum and combines salient representations through adaptive mixing. [METHOD NAME]'s superior performance achieves up to 9.2% accuracy improvement over leading baselines across homophilic and heterophilic graphs. 

**Abstract (ZH)**: 图的异质性，其中连接的节点具有不同的标签，近年来引起了广泛关注。大多数现有工作采用简化的方法——对同质图使用低通滤波器，对异质图使用高通滤波器。然而，我们发现图的异质性与频谱滤波器之间的关系更为复杂——最优滤波器响应随频率成分的不同而变化，并不严格遵循异质性程度的单调相关关系。这一发现挑战了传统的固定滤波器设计，并提示需要自适应滤波来保持图嵌入的表达性。正式地，自然地提出了一些问题：给定一个异质图G，G的异质性程度变化如何以及在多大程度上影响GNNs的性能？我们如何设计自适应滤波器来适应这些变化的异质连接？我们的理论分析表明，GNNs的平均频率响应与图的异质性程度之间并不严格遵循单调相关关系，这需要自适应图滤波器来保证良好的泛化性能。因此，我们提出了一种简单而强大的GNN [METHOD NAME]，它提取跨异质性光谱的信息，并通过自适应混合结合关键表示。[METHOD NAME]在同质图和异质图上的表现 superior，相对于领先的基础模型取得了高达9.2%的准确性提升。 

---
# Discrete State Diffusion Models: A Sample Complexity Perspective 

**Title (ZH)**: 离散状态扩散模型：一种样本复杂性视角 

**Authors**: Aadithya Srikanth, Mudit Gaur, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.10854)  

**Abstract**: Diffusion models have demonstrated remarkable performance in generating high-dimensional samples across domains such as vision, language, and the sciences. Although continuous-state diffusion models have been extensively studied both empirically and theoretically, discrete-state diffusion models, essential for applications involving text, sequences, and combinatorial structures, remain significantly less understood from a theoretical standpoint. In particular, all existing analyses of discrete-state models assume score estimation error bounds without studying sample complexity results. In this work, we present a principled theoretical framework for discrete-state diffusion, providing the first sample complexity bound of $\widetilde{\mathcal{O}}(\epsilon^{-2})$. Our structured decomposition of the score estimation error into statistical, approximation, optimization, and clipping components offers critical insights into how discrete-state models can be trained efficiently. This analysis addresses a fundamental gap in the literature and establishes the theoretical tractability and practical relevance of discrete-state diffusion models. 

**Abstract (ZH)**: 离散状态扩散模型在视觉、语言和科学等领域生成高维样本方面展现了显著性能。尽管连续状态扩散模型在实验和理论研究中均得到了广泛探讨，但涉及文本、序列和组合结构应用的离散状态扩散模型在理论上仍远未得到充分理解。特别地，现有所有关于离散状态模型的分析都基于评分估计误差界而未研究样本复杂度结果。在本工作中，我们提出了一个严谨的理论框架来研究离散状态扩散模型，并提供了首个样本复杂度界$\widetilde{\mathcal{O}}(\epsilon^{-2})$。我们结构化的评分估计误差分解为统计、近似、优化和截断四项，为高效训练离散状态模型提供了关键见解。这项分析填补了文献中的一个基本空白，并确立了离散状态扩散模型的理论可处理性和实际相关性。 

---
# Software Defect Prediction using Autoencoder Transformer Model 

**Title (ZH)**: 使用自动编码器变压器模型的软件缺陷预测 

**Authors**: Seshu Barma, Mohanakrishnan Hariharan, Satish Arvapalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.10840)  

**Abstract**: An AI-ML-powered quality engineering approach uses AI-ML to enhance software quality assessments by predicting defects. Existing ML models struggle with noisy data types, imbalances, pattern recognition, feature extraction, and generalization. To address these challenges, we develop a new model, Adaptive Differential Evolution (ADE) based Quantum Variational Autoencoder-Transformer (QVAET) Model (ADE-QVAET). ADE combines with QVAET to obtain high-dimensional latent features and maintain sequential dependencies, resulting in enhanced defect prediction accuracy. ADE optimization enhances model convergence and predictive performance. ADE-QVAET integrates AI-ML techniques such as tuning hyperparameters for scalable and accurate software defect prediction, representing an AI-ML-driven technology for quality engineering. During training with a 90% training percentage, ADE-QVAET achieves high accuracy, precision, recall, and F1-score of 98.08%, 92.45%, 94.67%, and 98.12%, respectively, when compared to the Differential Evolution (DE) ML model. 

**Abstract (ZH)**: 基于AI-ML的动力质量工程方法通过预测缺陷来增强软件质量评估 

---
# Happiness is Sharing a Vocabulary: A Study of Transliteration Methods 

**Title (ZH)**: 幸福在于共享词汇：音译方法研究 

**Authors**: Haeji Jung, Jinju Kim, Kyungjin Kim, Youjeong Roh, David R. Mortensen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10827)  

**Abstract**: Transliteration has emerged as a promising means to bridge the gap between various languages in multilingual NLP, showing promising results especially for languages using non-Latin scripts. We investigate the degree to which shared script, overlapping token vocabularies, and shared phonology contribute to performance of multilingual models. To this end, we conduct controlled experiments using three kinds of transliteration (romanization, phonemic transcription, and substitution ciphers) as well as orthography. We evaluate each model on two downstream tasks -- named entity recognition (NER) and natural language inference (NLI) -- and find that romanization significantly outperforms other input types in 7 out of 8 evaluation settings, largely consistent with our hypothesis that it is the most effective approach. We further analyze how each factor contributed to the success, and suggest that having longer (subword) tokens shared with pre-trained languages leads to better utilization of the model. 

**Abstract (ZH)**: 多语言NLP中转写已成为缩小各种语言之间差距的有前途的方法，特别是在使用非拉丁字母-script的语言中显示出有前途的结果。我们研究了共享字母表、重叠词汇表以及共享音韵特征对多语言模型性能的贡献程度。为此，我们使用三种类型的转写（ romanization、音素转写和替换密码）以及 orthography 进行了受控实验。我们分别在命名实体识别（NER）和自然语言推理（NLI）两个下游任务上评估每个模型，并发现在7种8种评估设置中有7种情况下，romanization 显著优于其他输入类型，这基本符合我们的假设，即这是最有效的方法。我们还分析了每个因素对成功的贡献，并建议与预训练语言共享更长（子词）令牌可以更好地利用模型。 

---
# From Detection to Mitigation: Addressing Bias in Deep Learning Models for Chest X-Ray Diagnosis 

**Title (ZH)**: 从检测到缓解：解决胸部X光诊断深度学习模型中的偏见 

**Authors**: Clemence Mottez, Louisa Fay, Maya Varma, Sophie Ostmeier, Curtis Langlotz  

**Link**: [PDF](https://arxiv.org/pdf/2510.10822)  

**Abstract**: Deep learning models have shown promise in improving diagnostic accuracy from chest X-rays, but they also risk perpetuating healthcare disparities when performance varies across demographic groups. In this work, we present a comprehensive bias detection and mitigation framework targeting sex, age, and race-based disparities when performing diagnostic tasks with chest X-rays. We extend a recent CNN-XGBoost pipeline to support multi-label classification and evaluate its performance across four medical conditions. We show that replacing the final layer of CNN with an eXtreme Gradient Boosting classifier improves the fairness of the subgroup while maintaining or improving the overall predictive performance. To validate its generalizability, we apply the method to different backbones, namely DenseNet-121 and ResNet-50, and achieve similarly strong performance and fairness outcomes, confirming its model-agnostic design. We further compare this lightweight adapter training method with traditional full-model training bias mitigation techniques, including adversarial training, reweighting, data augmentation, and active learning, and find that our approach offers competitive or superior bias reduction at a fraction of the computational cost. Finally, we show that combining eXtreme Gradient Boosting retraining with active learning yields the largest reduction in bias across all demographic subgroups, both in and out of distribution on the CheXpert and MIMIC datasets, establishing a practical and effective path toward equitable deep learning deployment in clinical radiology. 

**Abstract (ZH)**: 深度学习模型在提高胸部X光诊断准确性方面显示出潜力，但同时也可能由于不同人群组之间的性能差异而加剧医疗保健不平等。本研究提出了一种综合偏见检测与缓解框架，旨在解决胸部X光诊断任务中基于性别、年龄和种族的不平等现象。我们将最近的CNN-XGBoost管道扩展以支持多标签分类，并在其上评估四种医学条件的表现。结果显示，用极端梯度提升分类器替换CNN的最后一层可以提高子群体的公平性，同时保持或提高总体预测性能。为了验证其通用性，我们将该方法应用于不同的骨干网络，即DenseNet-121和ResNet-50，并获得了相似的强性能和公平性结果，证明其具有模型无关性设计。我们进一步将该轻量级适配器训练方法与传统的全模型训练偏见缓解技术（包括对抗训练、重权分配、数据增强和主动学习）进行了比较，并发现我们的方法在计算成本大幅度降低的情况下提供了竞争力或更优越的偏见减少效果。最后，我们展示了结合极端梯度提升重新训练与主动学习可以最大程度地减少所有人口子群体中的偏见，在CheXpert和MIMIC数据集中分布内和分布外均是如此，为临床放射学中公平的深度学习部署提供了切实有效的方法。 

---
# Generative AI and the Transformation of Software Development Practices 

**Title (ZH)**: 生成式人工智能与软件开发实践的转型 

**Authors**: Vivek Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2510.10819)  

**Abstract**: Generative AI is reshaping how software is designed, written, and maintained. Advances in large language models (LLMs) are enabling new development styles - from chat-oriented programming and 'vibe coding' to agentic programming - that can accelerate productivity and broaden access. This paper examines how AI-assisted techniques are changing software engineering practice, and the related issues of trust, accountability, and shifting skills. We survey iterative chat-based development, multi-agent systems, dynamic prompt orchestration, and integration via the Model Context Protocol (MCP). Using case studies and industry data, we outline both the opportunities (faster cycles, democratized coding) and the challenges (model reliability and cost) of applying generative AI to coding. We describe new roles, skills, and best practices for using AI in a responsible and effective way. 

**Abstract (ZH)**: 生成式AI正在重塑软件的设计、编写和维护方式。大规模语言模型的进步正推动新的开发模式——从聊天导向编程和“氛围编码”到代理式编程——这些模式能够提高生产力并扩大访问范围。本文探讨了AI辅助技术如何改变软件工程实践，以及相关的问题，如信任、问责和技能转移。我们概述了迭代聊天式开发、多智能体系统、动态提示编排以及通过模型上下文协议（MCP）集成的情况，并结合案例研究和行业数据，阐明将生成式AI应用于编码的机遇与挑战。我们描述了在负责任和有效的方式下使用AI的新角色、技能和最佳实践。 

---
# PruneGCRN: Minimizing and explaining spatio-temporal problems through node pruning 

**Title (ZH)**: PruneGCRN: 通过节点裁剪最小化和解释时空问题 

**Authors**: Javier García-Sigüenza, Mirco Nanni, Faraón Llorens-Largo, José F. Vicent  

**Link**: [PDF](https://arxiv.org/pdf/2510.10803)  

**Abstract**: This work addresses the challenge of using a deep learning model to prune graphs and the ability of this method to integrate explainability into spatio-temporal problems through a new approach. Instead of applying explainability to the model's behavior, we seek to gain a better understanding of the problem itself. To this end, we propose a novel model that integrates an optimized pruning mechanism capable of removing nodes from the graph during the training process, rather than doing so as a separate procedure. This integration allows the architecture to learn how to minimize prediction error while selecting the most relevant nodes. Thus, during training, the model searches for the most relevant subset of nodes, obtaining the most important elements of the problem, facilitating its analysis. To evaluate the proposed approach, we used several widely used traffic datasets, comparing the accuracy obtained by pruning with the model and with other methods. The experiments demonstrate that our method is capable of retaining a greater amount of information as the graph reduces in size compared to the other methods used. These results highlight the potential of pruning as a tool for developing models capable of simplifying spatio-temporal problems, thereby obtaining their most important elements. 

**Abstract (ZH)**: 本研究解决了使用深度学习模型修剪图结构的挑战，并提出了一种新方法，该方法通过将可解释性集成到时空问题中来增强模型的能力。该方法不是将可解释性应用于模型的行为，而是寻求更好地理解问题本身。为此，我们提出了一种新颖的模型，该模型集成了一个优化的修剪机制，在训练过程中能够从图中移除节点，而不需要将其作为单独的步骤进行。这种集成使得架构能够在选择最相关节点的同时学习如何最小化预测误差。因此，在训练过程中，模型会搜索最相关的节点子集，获得问题的重要元素，从而便于问题的分析。为了评估所提出的方法，我们使用了几种广泛使用的交通数据集，将修剪后的模型的准确性与完整模型和其他方法的准确性进行了比较。实验结果表明，与使用的方法相比，我们的方法能够在图结构减小的同时保留更多的信息。这些结果突显了修剪作为开发能够简化时空问题的模型工具的潜力，从而获取其最重要的元素。 

---
# Toward Human-Centered Readability Evaluation 

**Title (ZH)**: 面向以人为中心的可读性评价 

**Authors**: Bahar İlgen, Georges Hattab  

**Link**: [PDF](https://arxiv.org/pdf/2510.10801)  

**Abstract**: Text simplification is essential for making public health information accessible to diverse populations, including those with limited health literacy. However, commonly used evaluation metrics in Natural Language Processing (NLP), such as BLEU, FKGL, and SARI, mainly capture surface-level features and fail to account for human-centered qualities like clarity, trustworthiness, tone, cultural relevance, and actionability. This limitation is particularly critical in high-stakes health contexts, where communication must be not only simple but also usable, respectful, and trustworthy. To address this gap, we propose the Human-Centered Readability Score (HCRS), a five-dimensional evaluation framework grounded in Human-Computer Interaction (HCI) and health communication research. HCRS integrates automatic measures with structured human feedback to capture the relational and contextual aspects of readability. We outline the framework, discuss its integration into participatory evaluation workflows, and present a protocol for empirical validation. This work aims to advance the evaluation of health text simplification beyond surface metrics, enabling NLP systems that align more closely with diverse users' needs, expectations, and lived experiences. 

**Abstract (ZH)**: 文本简化对于使公共健康信息易于不同人群获取，包括健康素养较低的人群，至关重要。然而，自然语言处理（NLP）中常用的评估指标，如BLEU、FKGL和SARI，主要捕捉表面特征，并未能考虑到以人为中心的质量，如清晰性、可信度、语调、文化相关性和可操作性。特别是在高风险的健康交流情境中，交流不仅要简单，还要易于使用、尊重并可信。为弥补这一不足，我们提出以人为本可读性评分（HCRS）框架，该框架结合了人机交互（HCI）和健康沟通研究。HCRS 将自动评估措施与结构化的人类反馈结合，以捕捉可读性的关系和上下文方面。我们概述了该框架，讨论了其如何融入参与式评估工作流程，并提出了实证验证的协议。本研究旨在超越表面指标，提升健康文本简化的评估，使得NLP系统更加贴近不同用户的需求、期望和生活体验。 

---
# BioOSS: A Bio-Inspired Oscillatory State System with Spatio-Temporal Dynamics 

**Title (ZH)**: BioOSS: 一种受生物启发的空间-时间动力学振荡状态系统 

**Authors**: Zhongju Yuan, Geraint Wiggins, Dick Botteldooren  

**Link**: [PDF](https://arxiv.org/pdf/2510.10790)  

**Abstract**: Today's deep learning architectures are primarily based on perceptron models, which do not capture the oscillatory dynamics characteristic of biological neurons. Although oscillatory systems have recently gained attention for their closer resemblance to neural behavior, they still fall short of modeling the intricate spatio-temporal interactions observed in natural neural circuits. In this paper, we propose a bio-inspired oscillatory state system (BioOSS) designed to emulate the wave-like propagation dynamics critical to neural processing, particularly in the prefrontal cortex (PFC), where complex activity patterns emerge. BioOSS comprises two interacting populations of neurons: p neurons, which represent simplified membrane-potential-like units inspired by pyramidal cells in cortical columns, and o neurons, which govern propagation velocities and modulate the lateral spread of activity. Through local interactions, these neurons produce wave-like propagation patterns. The model incorporates trainable parameters for damping and propagation speed, enabling flexible adaptation to task-specific spatio-temporal structures. We evaluate BioOSS on both synthetic and real-world tasks, demonstrating superior performance and enhanced interpretability compared to alternative architectures. 

**Abstract (ZH)**: 今天深度学习架构主要基于感知器模型，无法捕捉生物神经元的振荡动力学特性。尽管近年来振荡系统由于更接近神经行为而受到了关注，但仍不足以 modeling 天然神经回路中观察到的错综复杂的时空交互作用。本文提出了一种受生物启发的振荡状态系统（BioOSS），旨在模拟对于神经处理至关重要的波形传播动力学，特别是在前额叶皮层（PFC），复杂活动模式在此处出现。BioOSS 包含两种相互作用的神经元群体：p 神经元，代表由皮层柱中尖锋细胞启发的简化的膜电位样单元；o 神经元，控制传播速度并调节活动的侧向扩散。通过局部相互作用，这些神经元产生波形传播模式。该模型包含可训练参数，用于调整阻尼和传播速度，以适应特定任务的时空结构。我们在合成和真实任务上评估了 BioOSS，结果显示其性能更优且更具可解释性，优于其他架构。 

---
# ParsVoice: A Large-Scale Multi-Speaker Persian Speech Corpus for Text-to-Speech Synthesis 

**Title (ZH)**: ParsVoice: 一种大规模多说话人波斯语语音语料库，用于文本转语音合成 

**Authors**: Mohammad Javad Ranjbar Kalahroodi, Heshaam Faili, Azadeh Shakery  

**Link**: [PDF](https://arxiv.org/pdf/2510.10774)  

**Abstract**: Persian Language, despite being spoken by over 100 million people worldwide, remains severely underrepresented in high-quality speech corpora, particularly for text-to-speech (TTS) synthesis applications. Existing Persian speech datasets are typically smaller than their English counterparts, which creates a key limitation for developing Persian speech technologies. We address this gap by introducing ParsVoice, the largest Persian speech corpus designed specifically for TTS applications. We created an automated pipeline that transforms raw audiobook content into TTS-ready data, incorporating components such as a BERT-based sentence completion detector, a binary search boundary optimization method for precise audio-text alignment, and multi-dimensional quality assessment frameworks tailored to Persian. The pipeline processes 2,000 audiobooks, yielding 3,526 hours of clean speech, which was further filtered into a 1,804-hour high-quality subset suitable for TTS, featuring more than 470 speakers. ParsVoice is the largest high-quality Persian speech dataset, offering speaker diversity and audio quality comparable to major English corpora. The complete dataset has been made publicly available to accelerate the development of Persian speech technologies and to serve as a template for other low-resource languages. The ParsVoice dataset is publicly available at ParsVoice (this https URL). 

**Abstract (ZH)**: 尽管波斯语是全世界超过1亿人使用的语言，但在高质量语音语料库中仍然严重缺乏，尤其是在语音合成（TTS）应用方面。现有的波斯语音数据集通常比其英语对照组更小，这为发展波斯语音技术带来了关键限制。我们通过引入ParsVoice——为TTS应用量身设计的最大规模波斯语音语料库来填补这一空白。我们创建了一个自动化流水线，将原始有声书内容转换为TTS可用数据，其中包括基于BERT的句子完成检测器、二分搜索边界优化方法以实现精确的音频-文本对齐，以及针对波斯语量身定制的多维度质量评估框架。该流水线处理了2000本有声书，产生了3526小时的干净语音，并进一步筛选出1804小时高质量子集，适合TTS应用，涵盖了超过470名说话者。ParsVoice是最大的高质量波斯语音数据集，其说话人口语多样性及音频质量可与主要的英语语料库媲美。整个数据集已公开发布，旨在加速波斯语音技术的发展，并为其他低资源语言提供模板。ParsVoice数据集可在ParsVoice（此链接https URL）公开获取。 

---
# Understanding Sampler Stochasticity in Training Diffusion Models for RLHF 

**Title (ZH)**: 理解采样器随机性在训练用于RLHF的扩散模型中的作用 

**Authors**: Jiayuan Sheng, Hanyang Zhao, Haoxian Chen, David D. Yao, Wenpin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10767)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is increasingly used to fine-tune diffusion models, but a key challenge arises from the mismatch between stochastic samplers used during training and deterministic samplers used during inference. In practice, models are fine-tuned using stochastic SDE samplers to encourage exploration, while inference typically relies on deterministic ODE samplers for efficiency and stability. This discrepancy induces a reward gap, raising concerns about whether high-quality outputs can be expected during inference. In this paper, we theoretically characterize this reward gap and provide non-vacuous bounds for general diffusion models, along with sharper convergence rates for Variance Exploding (VE) and Variance Preserving (VP) Gaussian models. Methodologically, we adopt the generalized denoising diffusion implicit models (gDDIM) framework to support arbitrarily high levels of stochasticity, preserving data marginals throughout. Empirically, our findings through large-scale experiments on text-to-image models using denoising diffusion policy optimization (DDPO) and mixed group relative policy optimization (MixGRPO) validate that reward gaps consistently narrow over training, and ODE sampling quality improves when models are updated using higher-stochasticity SDE training. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）越来越多地用于微调扩散模型，但一个关键挑战来自于训练中使用的随机采样器与推断中使用的确定性采样器之间的不匹配。在实践中，模型使用随机SDE采样器进行微调以鼓励探索，而推断通常依赖于确定性ODE采样器以提高效率和稳定性。这种差异导致了奖励差距，引发了关于在推断过程中能否期望获得高质量输出的担忧。在本文中，我们从理论上界定了这种奖励差距，并为通用扩散模型提供了非空洞界，同时为发散（VE）和方差保持（VP）高斯模型提供了更精确的收敛速率。方法上，我们采用广义去噪扩散隐模型（gDDIM）框架以支持任意高的随机性，并在整个过程中保持数据边缘分布。实验上，通过大规模实验发现，使用更高随机性的SDE训练更新模型后，基于去噪扩散策略优化（DDPO）和混合群相对策略优化（MixGRPO）的文本到图像模型显示了奖励差距的一贯缩小，并且ODE采样质量也得以提高。 

---
# GPS Spoofing Attack Detection in Autonomous Vehicles Using Adaptive DBSCAN 

**Title (ZH)**: 基于自适应DBSCAN的自动驾驶车辆GPS欺骗攻击检测 

**Authors**: Ahmad Mohammadi, Reza Ahmari, Vahid Hemmati, Frederick Owusu-Ambrose, Mahmoud Nabil Mahmoud, Parham Kebria, Abdollah Homaifar, Mehrdad Saif  

**Link**: [PDF](https://arxiv.org/pdf/2510.10766)  

**Abstract**: As autonomous vehicles become an essential component of modern transportation, they are increasingly vulnerable to threats such as GPS spoofing attacks. This study presents an adaptive detection approach utilizing a dynamically tuned Density Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm, designed to adjust the detection threshold ({\epsilon}) in real-time. The threshold is updated based on the recursive mean and standard deviation of displacement errors between GPS and in-vehicle sensors data, but only at instances classified as non-anomalous. Furthermore, an initial threshold, determined from 120,000 clean data samples, ensures the capability to identify even subtle and gradual GPS spoofing attempts from the beginning. To assess the performance of the proposed method, five different subsets from the real-world Honda Research Institute Driving Dataset (HDD) are selected to simulate both large and small magnitude GPS spoofing attacks. The modified algorithm effectively identifies turn-by-turn, stop, overshoot, and multiple small biased spoofing attacks, achieving detection accuracies of 98.621%, 99.960.1%, 99.880.1%, and 98.380.1%, respectively. This work provides a substantial advancement in enhancing the security and safety of AVs against GPS spoofing threats. 

**Abstract (ZH)**: 随着自主车辆成为现代交通系统的重要组成部分，它们日益受到如GPS欺骗攻击等威胁的脆弱性增加。本研究提出了一种自适应检测方法，利用一个动态调谐的基于密度的空间聚类算法（DBSCAN），实现了检测阈值（ε）的实时调整。阈值根据GPS和车载传感器数据位移误差的递归均值和标准差进行更新，但仅在非异常分类时进行。此外，从120,000个干净数据样本中确定的初始阈值确保了从一开始就具备识别甚至微小和渐进的GPS欺骗尝试的能力。为了评估所提方法的性能，从现实世界的Honda Research Institute Driving Dataset (HDD) 中选择了五个不同的子集，模拟了不同规模的GPS欺骗攻击。修改后的算法能够有效识别逐个转向、停车、超速以及多个小偏置的欺骗攻击，分别达到了98.621%，99.960.1%，99.880.1%，和98.380.1%的检测精度。本研究在提高自主车辆对GPS欺骗威胁的安全性和鲁棒性方面提供了重要进展。 

---
# Optimally Deep Networks -- Adapting Model Depth to Datasets for Superior Efficiency 

**Title (ZH)**: 优化深度网络：根据数据集适配模型深度以获得更高的效率 

**Authors**: Shaharyar Ahmed Khan Tareen, Filza Khan Tareen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10764)  

**Abstract**: Deep neural networks (DNNs) have provided brilliant performance across various tasks. However, this success often comes at the cost of unnecessarily large model sizes, high computational demands, and substantial memory footprints. Typically, powerful architectures are trained at full depths but not all datasets or tasks require such high model capacity. Training very deep architectures on relatively low-complexity datasets frequently leads to wasted computation, unnecessary energy consumption, and excessive memory usage, which in turn makes deployment of models on resource-constrained devices impractical. To address this problem, we introduce Optimally Deep Networks (ODNs), which provide a balance between model depth and task complexity. Specifically, we propose a NAS like training strategy called progressive depth expansion, which begins by training deep networks at shallower depths and incrementally increases their depth as the earlier blocks converge, continuing this process until the target accuracy is reached. ODNs use only the optimal depth for the given datasets, removing redundant layers. This cuts down future training and inference costs, lowers the memory footprint, enhances computational efficiency, and facilitates deployment on edge devices. Empirical results show that the optimal depths of ResNet-18 and ResNet-34 for MNIST and SVHN, achieve up to 98.64 % and 96.44 % reduction in memory footprint, while maintaining a competitive accuracy of 99.31 % and 96.08 %, respectively. 

**Abstract (ZH)**: Optimally Deep Networks (ODNs): Balancing Model Depth and Task Complexity 

---
# Proficiency-Aware Adaptation and Data Augmentation for Robust L2 ASR 

**Title (ZH)**: proficiency-aware 调适和数据增强以实现鲁棒的L2 ASR 

**Authors**: Ling Sun, Charlotte Zhu, Shuju Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10738)  

**Abstract**: General-purpose ASR underperforms for atypical speakers, such as L2 learners, reinforcing bias and limiting use in education and accessibility. Using the CEFR-graded Speak and Improve corpus, we show that naive fine-tuning of Whisper reduces average WER but simultaneously widens disparities and disproportionately harms lower-level learners. To address this, we propose two strategies: (i) proficiency-aware multitask learning, jointly optimizing ASR with proficiency classification, and (ii) targeted augmentation, applying spectrogram masking to low-proficiency speech to counter imbalance. These approaches reduce WER by up to 29.4 percent (relative) and insertion/deletion errors by as much as 58.6 percent (relative). Crucially, despite the severe imbalance of the dataset reflecting real-world distributions, both strategies consistently narrow proficiency gaps, advancing equitable ASR for L2 learners. 

**Abstract (ZH)**: 通用ASR在异常说话者（如二语学习者）上的表现不佳，加剧了偏见并限制了其在教育和无障碍领域的应用。利用CEFR分级的Speak and Improve语料库，我们发现对Whisper进行简单的微调虽然降低了平均词错误率（WER），但却同时扩大了差异，并不成比例地损害了低级别学习者。为了解决这一问题，我们提出了两种策略：（i）具备水平意识的多任务学习，联合优化ASR与水平分类；（ii）目标增强，通过对低水平语音应用频谱掩码以平衡数据分布。这两种方法最高可减少29.4%（相对）的WER，以及高达58.6%（相对）的插入/删除错误率。关键的是，尽管数据集中的严重不平衡反映了现实世界的真实分布，这两种策略都能一致地缩小水平差距，推动二语学习者的公平ASR。 

---
# Provable Anytime Ensemble Sampling Algorithms in Nonlinear Contextual Bandits 

**Title (ZH)**: 可验证的任意时间集成采样算法在非线性上下文多臂 bandits 中的应用 

**Authors**: Jiazheng Sun, Weixin Wang, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10730)  

**Abstract**: We provide a unified algorithmic framework for ensemble sampling in nonlinear contextual bandits and develop corresponding regret bounds for two most common nonlinear contextual bandit settings: Generalized Linear Ensemble Sampling (\texttt{GLM-ES}) for generalized linear bandits and Neural Ensemble Sampling (\texttt{Neural-ES}) for neural contextual bandits. Both methods maintain multiple estimators for the reward model parameters via maximum likelihood estimation on randomly perturbed data. We prove high-probability frequentist regret bounds of $\mathcal{O}(d^{3/2} \sqrt{T} + d^{9/2})$ for \texttt{GLM-ES} and $\mathcal{O}(\widetilde{d} \sqrt{T})$ for \texttt{Neural-ES}, where $d$ is the dimension of feature vectors, $\widetilde{d}$ is the effective dimension of a neural tangent kernel matrix, and $T$ is the number of rounds. These regret bounds match the state-of-the-art results of randomized exploration algorithms in nonlinear contextual bandit settings. In the theoretical analysis, we introduce techniques that address challenges specific to nonlinear models. Practically, we remove fixed-time horizon assumptions by developing anytime versions of our algorithms, suitable when $T$ is unknown. Finally, we empirically evaluate \texttt{GLM-ES}, \texttt{Neural-ES}, and their anytime variants, demonstrating strong performance. Overall, our results establish ensemble sampling as a provable and practical randomized exploration approach for nonlinear contextual bandits. 

**Abstract (ZH)**: 统一非线性上下文bandits的集成采样算法框架及对应的遗憾界分析：从广义线性集成采样(\texttt{GLM-ES})到神经集成采样(\texttt{Neural-ES}) 

---
# SS-DPPN: A self-supervised dual-path foundation model for the generalizable cardiac audio representation 

**Title (ZH)**: SS-DPPN: 一种自监督双路径基础模型用于通用心脏音频表示 

**Authors**: Ummy Maria Muna, Md Mehedi Hasan Shawon, Md Jobayer, Sumaiya Akter, Md Rakibul Hasan, Md. Golam Rabiul Alam  

**Link**: [PDF](https://arxiv.org/pdf/2510.10719)  

**Abstract**: The automated analysis of phonocardiograms is vital for the early diagnosis of cardiovascular disease, yet supervised deep learning is often constrained by the scarcity of expert-annotated data. In this paper, we propose the Self-Supervised Dual-Path Prototypical Network (SS-DPPN), a foundation model for cardiac audio representation and classification from unlabeled data. The framework introduces a dual-path contrastive learning based architecture that simultaneously processes 1D waveforms and 2D spectrograms using a novel hybrid loss. For the downstream task, a metric-learning approach using a Prototypical Network was used that enhances sensitivity and produces well-calibrated and trustworthy predictions. SS-DPPN achieves state-of-the-art performance on four cardiac audio benchmarks. The framework demonstrates exceptional data efficiency with a fully supervised model on three-fold reduction in labeled data. Finally, the learned representations generalize successfully across lung sound classification and heart rate estimation. Our experiments and findings validate SS-DPPN as a robust, reliable, and scalable foundation model for physiological signals. 

**Abstract (ZH)**: 自监督双路径原型网络在无标注数据中的心脏音频表示与分类 

---
# HYPERDOA: Robust and Efficient DoA Estimation using Hyperdimensional Computing 

**Title (ZH)**: HYPERDOA：基于超维度计算的稳健高效角度估计 

**Authors**: Rajat Bhattacharjya, Woohyeok Park, Arnab Sarkar, Hyunwoo Oh, Mohsen Imani, Nikil Dutt  

**Link**: [PDF](https://arxiv.org/pdf/2510.10718)  

**Abstract**: Direction of Arrival (DoA) estimation techniques face a critical trade-off, as classical methods often lack accuracy in challenging, low signal-to-noise ratio (SNR) conditions, while modern deep learning approaches are too energy-intensive and opaque for resource-constrained, safety-critical systems. We introduce HYPERDOA, a novel estimator leveraging Hyperdimensional Computing (HDC). The framework introduces two distinct feature extraction strategies -- Mean Spatial-Lag Autocorrelation and Spatial Smoothing -- for its HDC pipeline, and then reframes DoA estimation as a pattern recognition problem. This approach leverages HDC's inherent robustness to noise and its transparent algebraic operations to bypass the expensive matrix decompositions and ``black-box'' nature of classical and deep learning methods, respectively. Our evaluation demonstrates that HYPERDOA achieves ~35.39% higher accuracy than state-of-the-art methods in low-SNR, coherent-source scenarios. Crucially, it also consumes ~93% less energy than competing neural baselines on an embedded NVIDIA Jetson Xavier NX platform. This dual advantage in accuracy and efficiency establishes HYPERDOA as a robust and viable solution for mission-critical applications on edge devices. 

**Abstract (ZH)**: HYPERDOA：面向边缘设备的低信噪比条件下高效的到达方向估计方法 

---
# Deep Learning in Astrophysics 

**Title (ZH)**: 深度学习在天体物理学中的应用 

**Authors**: Yuan-Sen Ting  

**Link**: [PDF](https://arxiv.org/pdf/2510.10713)  

**Abstract**: Deep learning has generated diverse perspectives in astronomy, with ongoing discussions between proponents and skeptics motivating this review. We examine how neural networks complement classical statistics, extending our data analytical toolkit for modern surveys. Astronomy offers unique opportunities through encoding physical symmetries, conservation laws, and differential equations directly into architectures, creating models that generalize beyond training data. Yet challenges persist as unlabeled observations number in billions while confirmed examples with known properties remain scarce and expensive. This review demonstrates how deep learning incorporates domain knowledge through architectural design, with built-in assumptions guiding models toward physically meaningful solutions. We evaluate where these methods offer genuine advances versus claims requiring careful scrutiny. - Neural architectures overcome trade-offs between scalability, expressivity, and data efficiency by encoding physical symmetries and conservation laws into network structure, enabling learning from limited labeled data. - Simulation-based inference and anomaly detection extract information from complex, non-Gaussian distributions where analytical likelihoods fail, enabling field-level cosmological analysis and systematic discovery of rare phenomena. - Multi-scale neural modeling bridges resolution gaps in astronomical simulations, learning effective subgrid physics from expensive high-fidelity runs to enhance large-volume calculations where direct computation remains prohibitive. - Emerging paradigms-reinforcement learning for telescope operations, foundation models learning from minimal examples, and large language model agents for research automation-show promise though are still developing in astronomical applications. 

**Abstract (ZH)**: 深度学习在天文学中产生了多样的观点，提倡者与怀疑者之间的持续讨论推动了本综述的编写。我们探讨了神经网络如何补充经典统计学方法，扩展了用于现代调查的数据分析工具包。天文学通过直接将物理对称性、守恒律和微分方程编码到架构中，提供了独特的机会，从而创建出能够泛化到训练数据之外的模型。然而，随着未标记观测数据的数量达到数十亿，具备已知属性的确凿示例仍然稀缺且昂贵。本综述展示了深度学习如何通过架构设计整合领域知识，内置的假设指导模型趋向于物理上合理的解决方案。我们评估了这些方法提供的真正进展与需要仔细审查的声明之间的区别。- 神经网络架构通过将物理对称性和守恒律编码到网络结构中，克服了可扩展性、表达能力和数据效率之间的权衡，从而能够在有限标注数据下进行学习。- 基于模拟的推断和异常检测从复杂非高斯分布中提取信息，当解析似然函数失效时提供帮助，使天文学领域的宇宙学分析和系统发现稀有现象成为可能。- 多尺度神经建模弥合了天文学模拟中的分辨率差距，从昂贵的高保真运行中学到有效的子网格物理知识，以增强大规模计算，其中直接计算仍然是不切实际的。- 正在发展中但在天文学应用中展现潜力的新范式包括：望远镜操作中的强化学习、从少量示例学习的基础模型以及用于研究自动化的大语言模型代理。 

---
# Missing Data Multiple Imputation for Tabular Q-Learning in Online RL 

**Title (ZH)**: 基于在线强化学习的表格Q学习缺失数据多重插补 

**Authors**: Kyla Chasalow, Skyler Wu, Susan Murphy  

**Link**: [PDF](https://arxiv.org/pdf/2510.10709)  

**Abstract**: Missing data in online reinforcement learning (RL) poses challenges compared to missing data in standard tabular data or in offline policy learning. The need to impute and act at each time step means that imputation cannot be put off until enough data exist to produce stable imputation models. It also means future data collection and learning depend on previous imputations. This paper proposes fully online imputation ensembles. We find that maintaining multiple imputation pathways may help balance the need to capture uncertainty under missingness and the need for efficiency in online settings. We consider multiple approaches for incorporating these pathways into learning and action selection. Using a Grid World experiment with various types of missingness, we provide preliminary evidence that multiple imputation pathways may be a useful framework for constructing simple and efficient online missing data RL methods. 

**Abstract (ZH)**: 在线强化学习（RL）中缺失数据带来的挑战不同于标准表格式数据或离线策略学习中缺失数据的挑战。每次时间步都需要进行填充和行动意味着填充不能等到有足够的数据以产生稳定模型时才进行。这也意味着未来的数据收集和学习依赖于之前的填充。本文提出了一种完全在线的填充集成方法。我们发现，维护多个填充路径可能有助于在在线环境中平衡捕捉缺失性带来的不确定性需求与效率需求。我们考虑了将这些路径整合到学习和行动选择中的多种方法。通过使用具有不同类型缺失数据的Grid World实验，我们提供了初步证据，表明多个填充路径可能是一种有用的框架，用于构建简单的高效在线缺失数据RL方法。 

---
# Attention-Enhanced LSTM Modeling for Improved Temperature and Rainfall Forecasting in Bangladesh 

**Title (ZH)**: 增强注意力机制的LSTM模型在孟加拉国温度和降雨预报中的应用 

**Authors**: Usman Gani Joy, Shahadat kabir, Tasnim Niger  

**Link**: [PDF](https://arxiv.org/pdf/2510.10702)  

**Abstract**: Accurate climate forecasting is vital for Bangladesh, a region highly susceptible to climate change impacts on temperature and rainfall. Existing models often struggle to capture long-range dependencies and complex temporal patterns in climate data. This study introduces an advanced Long Short-Term Memory (LSTM) model integrated with an attention mechanism to enhance the prediction of temperature and rainfall dynamics. Utilizing comprehensive datasets from 1901-2023, sourced from NASA's POWER Project for temperature and the Humanitarian Data Exchange for rainfall, the model effectively captures seasonal and long-term trends. It outperforms baseline models, including XGBoost, Simple LSTM, and GRU, achieving a test MSE of 0.2411 (normalized units), MAE of 0.3860 degrees C, R^2 of 0.9834, and NRMSE of 0.0370 for temperature, and MSE of 1283.67 mm^2, MAE of 22.91 mm, R^2 of 0.9639, and NRMSE of 0.0354 for rainfall on monthly forecasts. The model demonstrates improved robustness with only a 20 percent increase in MSE under simulated climate trends (compared to an approximately 2.2-fold increase in baseline models without trend features) and a 50 percent degradation under regional variations (compared to an approximately 4.8-fold increase in baseline models without enhancements). These results highlight the model's ability to improve forecasting precision and offer potential insights into the physical processes governing climate variability in Bangladesh, supporting applications in climate-sensitive sectors. 

**Abstract (ZH)**: 准确的气候预测对于孟加拉国至关重要，该地区对温度和降雨量的气候变化影响极其敏感。现有的模型往往难以捕捉气候数据中的长期依赖关系和复杂的时空模式。本研究引入了结合注意力机制的高级长短期记忆（LSTM）模型，以增强温度和降雨动态的预测能力。利用自1901年至2023年来自NASA POWER项目和人道主义数据交换的数据集，该模型有效地捕捉到了季节性和长期趋势。与XGBoost、简单LSTM和GRU等基线模型相比，该模型在温度预测上实现了测试MSE为0.2411（归一化单位）、MAE为0.3860摄氏度、R²为0.9834和NRMSE为0.0370，在月度降雨预测上实现了MSE为1283.67平方毫米、MAE为22.91毫米、R²为0.9639和NRMSE为0.0354。在模拟气候趋势下，该模型仅增加了20%的MSE（而基线模型在没有趋势特征的情况下增加了约2.2倍），在区域差异情况下，其降解程度为50%（而基线模型在没有增强措施的情况下增加了约4.8倍）。这些结果突显了该模型提高预测精度的能力，并为其在孟加拉国气候敏感领域的应用提供了潜在的洞见，支持对气候变异物理过程的研究。 

---
# High-Dimensional Learning Dynamics of Quantized Models with Straight-Through Estimator 

**Title (ZH)**: 高维量化模型的直通估计学习动力学 

**Authors**: Yuma Ichikawa, Shuhei Kashiwamura, Ayaka Sakata  

**Link**: [PDF](https://arxiv.org/pdf/2510.10693)  

**Abstract**: Quantized neural network training optimizes a discrete, non-differentiable objective. The straight-through estimator (STE) enables backpropagation through surrogate gradients and is widely used. While previous studies have primarily focused on the properties of surrogate gradients and their convergence, the influence of quantization hyperparameters, such as bit width and quantization range, on learning dynamics remains largely unexplored. We theoretically show that in the high-dimensional limit, STE dynamics converge to a deterministic ordinary differential equation. This reveals that STE training exhibits a plateau followed by a sharp drop in generalization error, with plateau length depending on the quantization range. A fixed-point analysis quantifies the asymptotic deviation from the unquantized linear model. We also extend analytical techniques for stochastic gradient descent to nonlinear transformations of weights and inputs. 

**Abstract (ZH)**: 量化神经网络训练优化了一个离散的非可微目标。伪梯度直通估计器（STE）允许通过伪梯度进行反向传播，并被广泛使用。虽然以前的研究主要集中在伪梯度的性质及其收敛性上，但量化超参数，如位宽和量化范围，对学习动力学的影响仍 largely unexplored。我们从理论上证明，在高维极限下，STE动力学收敛到一个确定性的常微分方程。这揭示了STE训练在泛化误差上表现出一个平台期随后是急剧下降的现象，而平台期的长度取决于量化范围。定点分析量化了从无量化线性模型的渐近偏差。我们还扩展了随机梯度下降的分析技术，应用于权重和输入的非线性变换。 

---
# LSZone: A Lightweight Spatial Information Modeling Architecture for Real-time In-car Multi-zone Speech Separation 

**Title (ZH)**: LSZone：一种轻量级空间信息建模架构实现实时车内多区语音分离 

**Authors**: Jun Chen, Shichao Hu, Jiuxin Lin, Wenjie Li, Zihan Zhang, Xingchen Li, JinJiang Liu, Longshuai Xiao, Chao Weng, Lei Xie, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10687)  

**Abstract**: In-car multi-zone speech separation, which captures voices from different speech zones, plays a crucial role in human-vehicle interaction. Although previous SpatialNet has achieved notable results, its high computational cost still hinders real-time applications in vehicles. To this end, this paper proposes LSZone, a lightweight spatial information modeling architecture for real-time in-car multi-zone speech separation. We design a spatial information extraction-compression (SpaIEC) module that combines Mel spectrogram and Interaural Phase Difference (IPD) to reduce computational burden while maintaining performance. Additionally, to efficiently model spatial information, we introduce an extremely lightweight Conv-GRU crossband-narrowband processing (CNP) module. Experimental results demonstrate that LSZone, with a complexity of 0.56G MACs and a real-time factor (RTF) of 0.37, delivers impressive performance in complex noise and multi-speaker scenarios. 

**Abstract (ZH)**: 车载多区域语音分离技术：一种轻量级空间信息建模架构 

---
# BrowserAgent: Building Web Agents with Human-Inspired Web Browsing Actions 

**Title (ZH)**: BrowserAgent: 构建受人类网页浏览行为启发的网页代理 

**Authors**: Zhengbo Zhang, Zhiheng Lyu, Junhao Gong, Hongzhu Yi, Xinming Wang, Yuxuan Zhou, Jiabing Yang, Ping Nie, Yan Huang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10666)  

**Abstract**: Efficiently solving real-world problems with LLMs increasingly hinges on their ability to interact with dynamic web environments and autonomously acquire external information. While recent research like Search-R1 and WebDancer demonstrates strong performance in solving web tasks, they heavily rely on additional tools to convert the interactive web environment into static text content. This is in contrast to human browsing behaviors, which involve diverse interactions with the browser, such as scrolling, clicking, and typing. In this paper, we propose BrowserAgent, a more interactive agent that solves complex tasks through human-inspired browser actions. BrowserAgent operates directly on raw web pages via Playwright through a set of predefined browser actions. We adopt a two-stage training (Supervised Fine-Tuning (SFT) and Rejection Fine-Tuning (RFT)) to improve the model's generalization abilities. Despite using significantly less training data than Search-R1, BrowserAgent achieves more competitive results across different Open-QA tasks. Additionally, we introduce an explicit memory mechanism to store key conclusions across steps, further enhancing the model's reasoning capabilities for long-horizon tasks. Notably, BrowserAgent-7B can achieve around 20\% improvement over Search-R1 on multi-hop QA tasks like HotpotQA, 2Wiki, and Bamboogle. These results indicate that BrowserAgent can serve as a more advanced framework for more interactive and scalable web agents. 

**Abstract (ZH)**: 利用LLM高效解决实际问题 increasingly hinges on their ability to interact with dynamic web environments and autonomously acquire external information. BrowserAgent: 一种通过受人类启发的浏览器操作解决复杂任务的更互动代理 

---
# Trustworthy Retrosynthesis: Eliminating Hallucinations with a Diverse Ensemble of Reaction Scorers 

**Title (ZH)**: 可信逆合成：通过多样化的反应评分器消除幻觉 

**Authors**: Michal Sadowski, Maria Wyrzykowska, Lukasz Sztukiewicz, Tadija Radusinović, Jan Rzymkowski, Paweł Włodarczyk-Pruszyński, Mikołaj Sacha, Piotr Kozakowski, Ruard van Workum, Stanislaw Kamil Jastrzebski  

**Link**: [PDF](https://arxiv.org/pdf/2510.10645)  

**Abstract**: Retrosynthesis is one of the domains transformed by the rise of generative models, and it is one where the problem of nonsensical or erroneous outputs (hallucinations) is particularly insidious: reliable assessment of synthetic plans is time-consuming, with automatic methods lacking. In this work, we present RetroTrim, a retrosynthesis system that successfully avoids nonsensical plans on a set of challenging drug-like targets. Compared to common baselines in the field, our system is not only the sole method that succeeds in filtering out hallucinated reactions, but it also results in the highest number of high-quality paths overall. The key insight behind RetroTrim is the combination of diverse reaction scoring strategies, based on machine learning models and existing chemical databases. We show that our scoring strategies capture different classes of hallucinations by analyzing them on a dataset of labeled retrosynthetic intermediates. To measure the performance of retrosynthesis systems, we propose a novel evaluation protocol for reactions and synthetic paths based on a structured review by expert chemists. Using this protocol, we compare systems on a set of 32 novel targets, curated to reflect recent trends in drug structures. While the insights behind our methodology are broadly applicable to retrosynthesis, our focus is on targets in the drug-like domain. By releasing our benchmark targets and the details of our evaluation protocol, we hope to inspire further research into reliable retrosynthesis. 

**Abstract (ZH)**: retrosynthesis 是生成模型兴起后发生变革的领域之一，其中不可信或错误输出（幻觉）的问题尤为隐蔽：可靠评估合成方案耗时且缺乏自动方法。在此项工作中，我们提出了一种名为 RetroTrim 的 retrosynthesis 系统，该系统成功避免了一组挑战性的药物样目标的不可信方案。与该领域的常见基准相比，我们的系统是唯一一种能够筛选出幻觉反应的方法，并且总体上生成了最高数量的高质量路径。RetroTrim 的关键洞察是结合了基于机器学习模型和现有化学数据库的不同反应评分策略。我们通过在标记的 retrosynthetic 中间体数据集上分析它们来证明我们的评分策略能够捕捉不同类别的幻觉。为了衡量 retrosynthesis 系统的性能，我们提出了基于专家化学家结构化评审的新评估协议。使用此协议，我们在一组 32 个经过精心挑选以反映最近药物结构趋势的新型目标上比较系统。虽然我们方法背后的原则在 retrosynthesis 领域具有广泛的适用性，但我们的重点是药物样目标。通过提供我们的基准目标和评估协议的详细信息，我们希望激励进一步的研究以实现可靠的 retrosynthesis。 

---
# A Machine Learning Approach for MIDI to Guitar Tablature Conversion 

**Title (ZH)**: 使用机器学习的方法将MIDI转化为吉他谱 

**Authors**: Maximos Kaliakatsos-Papakostas, Gregoris Bastas, Dimos Makris, Dorien Herremans, Vassilis Katsouros, Petros Maragos  

**Link**: [PDF](https://arxiv.org/pdf/2510.10619)  

**Abstract**: Guitar tablature transcription consists in deducing the string and the fret number on which each note should be played to reproduce the actual musical part. This assignment should lead to playable string-fret combinations throughout the entire track and, in general, preserve parsimonious motion between successive combinations. Throughout the history of guitar playing, specific chord fingerings have been developed across different musical styles that facilitate common idiomatic voicing combinations and motion between them. This paper presents a method for assigning guitar tablature notation to a given MIDI-based musical part (possibly consisting of multiple polyphonic tracks), i.e. no information about guitar-idiomatic expressional characteristics is involved (e.g. bending etc.) The current strategy is based on machine learning and requires a basic assumption about how much fingers can stretch on a fretboard; only standard 6-string guitar tuning is examined. The proposed method also examines the transcription of music pieces that was not meant to be played or could not possibly be played by a guitar (e.g. potentially a symphonic orchestra part), employing a rudimentary method for augmenting musical information and training/testing the system with artificial data. The results present interesting aspects about what the system can achieve when trained on the initial and augmented dataset, showing that the training with augmented data improves the performance even in simple, e.g. monophonic, cases. Results also indicate weaknesses and lead to useful conclusions about possible improvements. 

**Abstract (ZH)**: 吉他谱转录涉及推断每个音符应在哪些弦及品上演奏以重现实际的音乐部分。这一分配应确保整首歌曲中可演奏的弦-品组合，并且通常在相继组合之间保持简洁的运动。在吉他演奏的历史上，针对不同音乐风格开发了特定的和弦指法，便于常常出现的idiomatic和弦配置及它们之间的转换。本文提出了一种将吉他谱记谱应用于给定的基于MIDI的音乐部分（可能包括多个多声部轨道）的方法，即不涉及吉他特有的表达特征（如滑音等）。当前策略基于机器学习，并假设手指在琴颈上的伸展程度；仅考察了标准六弦吉他调音。所提方法还研究了那些并非意图由吉他演奏或根本无法由吉他演奏的音乐作品（如潜在的交响乐团部分），采用一种简单的数据扩充方法，并使用人工数据训练/测试系统。结果展示了当用初始和扩充数据集训练时，系统可以实现的有趣方面，表明使用扩充数据的训练甚至在简单的，例如单声部的情况下，也提高了性能。结果还显示了系统的弱点，从而得出关于可能改进的有用结论。 

---
# Compositional Symmetry as Compression: Lie Pseudogroup Structure in Algorithmic Agents 

**Title (ZH)**: 组合对称性作为压缩：算法代理中的李假群结构 

**Authors**: Giulio Ruffini  

**Link**: [PDF](https://arxiv.org/pdf/2510.10586)  

**Abstract**: In the algorithmic (Kolmogorov) view, agents are programs that track and compress sensory streams using generative programs. We propose a framework where the relevant structural prior is simplicity (Solomonoff) understood as \emph{compositional symmetry}: natural streams are well described by (local) actions of finite-parameter Lie pseudogroups on geometrically and topologically complex low-dimensional configuration manifolds (latent spaces). Modeling the agent as a generic neural dynamical system coupled to such streams, we show that accurate world-tracking imposes (i) \emph{structural constraints} -- equivariance of the agent's constitutive equations and readouts -- and (ii) \emph{dynamical constraints}: under static inputs, symmetry induces conserved quantities (Noether-style labels) in the agent dynamics and confines trajectories to reduced invariant manifolds; under slow drift, these manifolds move but remain low-dimensional. This yields a hierarchy of reduced manifolds aligned with the compositional factorization of the pseudogroup, providing a geometric account of the ``blessing of compositionality'' in deep models. We connect these ideas to the Spencer formalism for Lie pseudogroups and formulate a symmetry-based, self-contained version of predictive coding in which higher layers receive only \emph{coarse-grained residual transformations} (prediction-error coordinates) along symmetry directions unresolved at lower layers. 

**Abstract (ZH)**: 基于算法（柯尔莫哥洛夫）的观点，代理是程序，通过生成程序追踪和压缩感觉流。我们提出一个框架，其中相关的关键结构先验是简化性（索洛莫诺夫理解为组合对称性）：自然流可以用有限参数李赝群的局部作用来很好地描述几何和拓扑复杂但低维的配置流形（潜在空间）。将代理建模为与这些流耦合的通用神经动力系统，我们展示准确的世界追踪施加了（i）结构约束——代理基本方程和读数的不变性；以及（ii）动力学约束：在静态输入下，对称性诱导代理动力学中的守恒量（诺ether风格的标签），并限制轨迹到低维不变流形；在缓慢漂移下，这些流形会移动但保持低维。这产生了一种与赝群的组合因子分解相一致的流形层次结构，提供了关于深度模型中“组合性的祝福”的几何解释。我们将这些概念与李赝群的斯宾赛形式主义联系起来，并提出了一个基于对称性的自包含预测编码框架，其中更高层仅接收沿低层未解析的方向的粗糙化残差变换（预测误差坐标）。 

---
# PAC-Bayesian Reinforcement Learning Trains Generalizable Policies 

**Title (ZH)**: PAC-Bayesian Reinforcement Learning 培训可泛化的策略 

**Authors**: Abdelkrim Zitouni, Mehdi Hennequin, Juba Agoun, Ryan Horache, Nadia Kabachi, Omar Rivasplata  

**Link**: [PDF](https://arxiv.org/pdf/2510.10544)  

**Abstract**: We derive a novel PAC-Bayesian generalization bound for reinforcement learning that explicitly accounts for Markov dependencies in the data, through the chain's mixing time. This contributes to overcoming challenges in obtaining generalization guarantees for reinforcement learning, where the sequential nature of data breaks the independence assumptions underlying classical bounds. Our bound provides non-vacuous certificates for modern off-policy algorithms like Soft Actor-Critic. We demonstrate the bound's practical utility through PB-SAC, a novel algorithm that optimizes the bound during training to guide exploration. Experiments across continuous control tasks show that our approach provides meaningful confidence certificates while maintaining competitive performance. 

**Abstract (ZH)**: 我们通过链的混合时间明确考虑数据中的马尔可夫依赖性，推导出一种新的PAC-Bayesian泛化界，以克服强化学习中由于数据的序列性质破坏经典界限下的独立性假设而带来的泛化保证难题。我们的界为Soft Actor-Critic等现代离策略算法提供了非空洞的信心证书。我们通过在训练过程中优化该界来指导探索的新算法PB-SAC展示其实用性。实验结果表明，我们的方法在提供有意义的信心证书的同时保持了竞争性的性能。 

---
# f-INE: A Hypothesis Testing Framework for Estimating Influence under Training Randomness 

**Title (ZH)**: f-INE：一种基于训练随机性的假设检验框架用于估算影响 

**Authors**: Subhodip Panda, Dhruv Tarsadiya, Shashwat Sourav, Prathosh A.P, Sai Praneeth Karimireddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.10510)  

**Abstract**: Influence estimation methods promise to explain and debug machine learning by estimating the impact of individual samples on the final model. Yet, existing methods collapse under training randomness: the same example may appear critical in one run and irrelevant in the next. Such instability undermines their use in data curation or cleanup since it is unclear if we indeed deleted/kept the correct datapoints. To overcome this, we introduce *f-influence* -- a new influence estimation framework grounded in hypothesis testing that explicitly accounts for training randomness, and establish desirable properties that make it suitable for reliable influence estimation. We also design a highly efficient algorithm **f**-**IN**fluence **E**stimation (**f-INE**) that computes f-influence **in a single training run**. Finally, we scale up f-INE to estimate influence of instruction tuning data on Llama-3.1-8B and show it can reliably detect poisoned samples that steer model opinions, demonstrating its utility for data cleanup and attributing model behavior. 

**Abstract (ZH)**: *f*-影响估计方法：一种基于假设检验的新框架及其在模型影响估计中的应用 

---
# Personalized Motion Guidance Framework for Athlete-Centric Coaching 

**Title (ZH)**: 运动员中心的个性化动作指导框架 

**Authors**: Ryota Takamidoa, Chiharu Suzukia, Hiroki Nakamoto  

**Link**: [PDF](https://arxiv.org/pdf/2510.10496)  

**Abstract**: A critical challenge in contemporary sports science lies in filling the gap between group-level insights derived from controlled hypothesis-driven experiments and the real-world need for personalized coaching tailored to individual athletes' unique movement patterns. This study developed a Personalized Motion Guidance Framework (PMGF) to enhance athletic performance by generating individualized motion-refinement guides using generative artificial intelligence techniques. PMGF leverages a vertical autoencoder to encode motion sequences into athlete-specific latent representations, which can then be directly manipulated to generate meaningful guidance motions. Two manipulation strategies were explored: (1) smooth interpolation between the learner's motion and a target (e.g., expert) motion to facilitate observational learning, and (2) shifting the motion pattern in an optimal direction in the latent space using a local optimization technique. The results of the validation experiment with data from 51 baseball pitchers revealed that (1) PMGF successfully generated smooth transitions in motion patterns between individuals across all 1,275 pitcher pairs, and (2) the features significantly altered through PMGF manipulations reflected known performance-enhancing characteristics, such as increased stride length and knee extension associated with higher ball velocity, indicating that PMGF induces biomechanically plausible improvements. We propose a future extension called general-PMGF to enhance the applicability of this framework. This extension incorporates bodily, environmental, and task constraints into the generation process, aiming to provide more realistic and versatile guidance across diverse sports contexts. 

**Abstract (ZH)**: 当代运动科学的一个关键挑战是在控制性假设驱动实验获得的群体层面洞见与个性化教练的实际需求之间填补差距，个性化教练需针对每位运动员独特的运动模式进行量身定制。本研究开发了一种个性化运动指导框架（PMGF），利用生成人工智能技术生成个性化运动精炼指南以提升运动表现。PMGF利用垂直自动编码器将运动序列编码为运动员特定的潜在表示，然后可以直接操作这些表示以生成有意义的指导运动。探索了两种操作策略：（1）在学习者的运动和目标（例如专家）运动之间进行平滑插值，以促进观察学习；（2）使用局部优化技术在潜在空间中将运动模式沿最优方向平移。验证实验使用51名棒球投手的数据表明，（1）PMGF成功生成了所有1,275对投手之间运动模式的平滑过渡，（2）通过PMGF操作显著改变的特征反映了已知能提高表现的特性，如步长增加和膝关节伸展与更高球速相关，表明PMGF引发了生物力学上合理的改进。我们提出了一种未来扩展——通用PMGF，以增强此框架的应用性。该扩展将身体、环境和任务约束融入生成过程，旨在提供更现实和多样的指导，适用于各种运动情境。 

---
# Latent Retrieval Augmented Generation of Cross-Domain Protein Binders 

**Title (ZH)**: 跨域蛋白配体的隐空间检索增强生成 

**Authors**: Zishen Zhang, Xiangzhe Kong, Wenbing Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10480)  

**Abstract**: Designing protein binders targeting specific sites, which requires to generate realistic and functional interaction patterns, is a fundamental challenge in drug discovery. Current structure-based generative models are limited in generating nterfaces with sufficient rationality and interpretability. In this paper, we propose Retrieval-Augmented Diffusion for Aligned interface (RADiAnce), a new framework that leverages known interfaces to guide the design of novel binders. By unifying retrieval and generation in a shared contrastive latent space, our model efficiently identifies relevant interfaces for a given binding site and seamlessly integrates them through a conditional latent diffusion generator, enabling cross-domain interface transfer. Extensive exeriments show that RADiAnce significantly outperforms baseline models across multiple metrics, including binding affinity and recovery of geometries and interactions. Additional experimental results validate cross-domain generalization, demonstrating that retrieving interfaces from diverse domains, such as peptides, antibodies, and protein fragments, enhances the generation performance of binders for other domains. Our work establishes a new paradigm for protein binder design that successfully bridges retrieval-based knowledge and generative AI, opening new possibilities for drug discovery. 

**Abstract (ZH)**: 基于检索增强扩散的对齐界面生成（RADiAnce）：一种利用已知界面指导新型结合物设计的新框架 

---
# LightSAE: Parameter-Efficient and Heterogeneity-Aware Embedding for IoT Multivariate Time Series Forecasting 

**Title (ZH)**: LightSAE：参数高效且适应异构性的物联网多变量时间序列预测嵌入方法 

**Authors**: Yi Ren, Xinjie Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10465)  

**Abstract**: Modern Internet of Things (IoT) systems generate massive, heterogeneous multivariate time series data. Accurate Multivariate Time Series Forecasting (MTSF) of such data is critical for numerous applications. However, existing methods almost universally employ a shared embedding layer that processes all channels identically, creating a representational bottleneck that obscures valuable channel-specific information. To address this challenge, we introduce a Shared-Auxiliary Embedding (SAE) framework that decomposes the embedding into a shared base component capturing common patterns and channel-specific auxiliary components modeling unique deviations. Within this decomposition, we \rev{empirically observe} that the auxiliary components tend to exhibit low-rank and clustering characteristics, a structural pattern that is significantly less apparent when using purely independent embeddings. Consequently, we design LightSAE, a parameter-efficient embedding module that operationalizes these observed characteristics through low-rank factorization and a shared, gated component pool. Extensive experiments across 9 IoT-related datasets and 4 backbone architectures demonstrate LightSAE's effectiveness, achieving MSE improvements of up to 22.8\% with only 4.0\% parameter increase. 

**Abstract (ZH)**: 现代物联网(IoT)系统生成大量的异构多变量时间序列数据。准确的多变量时间序列预测(MTSF)对于众多应用至关重要。然而，现有方法几乎无一例外地使用一个共享嵌入层，对所有通道进行相同处理，从而形成一种表现瓶颈，掩盖了有价值的时间序列通道特定信息。为解决这一挑战，我们引入了一个共享辅助嵌入(SAE)框架，将嵌入分解为一个捕捉共同模式的共享基础组件和用于建模独特偏差的通道特定辅助组件。在这分解中，我们实验证明辅助组件往往表现出低秩和聚类特性，而在仅使用独立嵌入时，这种结构模式明显不那么突出。因此，我们设计了LightSAE，一个高效参数嵌入模块，通过低秩因子分解和共享门控组件池来实现这些观察到的特性。广泛实验表明，LightSAE在9个物联网相关数据集和4种骨干架构上表现出色，仅增加4.0%的参数量，就能实现MSE最大22.8%的改进。 

---
# Learning from Disagreement: A Group Decision Simulation Framework for Robust Medical Image Segmentation 

**Title (ZH)**: 基于分歧的学习：一种用于稳健医疗图像分割的群体决策模拟框架 

**Authors**: Chen Zhong, Yuxuan Yang, Xinyue Zhang, Ruohan Ma, Yong Guo, Gang Li, Jupeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.10462)  

**Abstract**: Medical image segmentation annotation suffers from inter-rater variability (IRV) due to differences in annotators' expertise and the inherent blurriness of medical images. Standard approaches that simply average expert labels are flawed, as they discard the valuable clinical uncertainty revealed in disagreements. We introduce a fundamentally new approach with our group decision simulation framework, which works by mimicking the collaborative decision-making process of a clinical panel. Under this framework, an Expert Signature Generator (ESG) learns to represent individual annotator styles in a unique latent space. A Simulated Consultation Module (SCM) then intelligently generates the final segmentation by sampling from this space. This method achieved state-of-the-art results on challenging CBCT and MRI datasets (92.11% and 90.72% Dice scores). By treating expert disagreement as a useful signal instead of noise, our work provides a clear path toward more robust and trustworthy AI systems for healthcare. 

**Abstract (ZH)**: 医学图像分割注释由于注释者专业水平差异和医学图像的固有模糊性而导致评价者间变异性（IRV）。传统的简单平均专家标签的方法存在缺陷，因为它忽略了分歧中揭示的宝贵临床不确定性。我们通过组决策模拟框架引入了一种全新的方法，该框架通过模拟临床小组的协作决策过程来工作。在这一框架下，专家签名生成器（ESG）学习在独特的潜在空间中代表个体注释者的风格。然后，模拟咨询模块（SCM）通过从中抽样智能生成最终的分割结果。该方法在具有挑战性的CBCT和MRI数据集上取得了最先进的结果（Dice分数分别为92.11%和90.72%）。通过将专家分歧视为有用的信号而不是噪声，我们的工作提供了一条通往更可靠和值得信赖的医疗保健AI系统的清晰路径。 

---
# Reverse Supervision at Scale: Exponential Search Meets the Economics of Annotation 

**Title (ZH)**: 大规模逆向监督：指数搜索与标注经济相结合 

**Authors**: Masoud Makrehchi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10446)  

**Abstract**: We analyze a reversed-supervision strategy that searches over labelings of a large unlabeled set \(B\) to minimize error on a small labeled set \(A\). The search space is \(2^n\), and the resulting complexity remains exponential even under large constant-factor speedups (e.g., quantum or massively parallel hardware). Consequently, arbitrarily fast -- but not exponentially faster -- computation does not obviate the need for informative labels or priors. In practice, the machine learning pipeline still requires an initial human contribution: specifying the objective, defining classes, and providing a seed set of representative annotations that inject inductive bias and align models with task semantics. Synthetic labels from generative AI can partially substitute provided their quality is human-grade and anchored by a human-specified objective, seed supervision, and validation. In this view, generative models function as \emph{label amplifiers}, leveraging small human-curated cores via active, semi-supervised, and self-training loops, while humans retain oversight for calibration, drift detection, and failure auditing. Thus, extreme computational speed reduces wall-clock time but not the fundamental supervision needs of learning; initial human (or human-grade) input remains necessary to ground the system in the intended task. 

**Abstract (ZH)**: 我们分析了一种反向监督策略，该策略在大规模未标注数据集 \(B\) 的标注中搜索，以最小化小型标注数据集 \(A\) 上的错误率。搜索空间为 \(2^n\)，即使在以常数因子极大的加速（如量子计算或大规模并行硬件）的情况下，结果的复杂度仍然是指数级的。因此，尽管可以实现任意快速的计算，但不是指数级别的加速并不能消除对信息性标签或先验知识的需求。在实践中，机器学习管道仍然需要初始的人类贡献：指定目标、定义类别以及提供代表性的种子标注，以注入归纳偏见并使模型与任务语义对齐。生成AI提供的合成标签在质量达到人类水平且基于人类指定的目标、种子监督和验证的情况下，可以部分替代真实标签的功能。从这一角度看，生成模型作为标签放大器，利用小规模的人类策划核心，通过主动的、半监督的和自我训练循环来放大标签，而人类保留校准、漂移检测和故障审计的监督。因此，极端的计算速度减少了壁钟时间，但并没有消除学习的基本监督需求；最初的人类（或人类水平）输入仍然是必要的，以确保系统与预期任务相契合。 

---
# Multi-Task Learning with Feature-Similarity Laplacian Graphs for Predicting Alzheimer's Disease Progression 

**Title (ZH)**: 基于特征相似性拉普拉斯图的多任务学习预测阿尔茨海默病进展 

**Authors**: Zixiang Xu, Menghui Zhou, Jun Qi, Xuanhan Fan, Yun Yang, Po Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10433)  

**Abstract**: Alzheimer's Disease (AD) is the most prevalent neurodegenerative disorder in aging populations, posing a significant and escalating burden on global healthcare systems. While Multi-Tusk Learning (MTL) has emerged as a powerful computational paradigm for modeling longitudinal AD data, existing frameworks do not account for the time-varying nature of feature correlations. To address this limitation, we propose a novel MTL framework, named Feature Similarity Laplacian graph Multi-Task Learning (MTL-FSL). Our framework introduces a novel Feature Similarity Laplacian (FSL) penalty that explicitly models the time-varying relationships between features. By simultaneously considering temporal smoothness among tasks and the dynamic correlations among features, our model enhances both predictive accuracy and biological interpretability. To solve the non-smooth optimization problem arising from our proposed penalty terms, we adopt the Alternating Direction Method of Multipliers (ADMM) algorithm. Experiments conducted on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset demonstrate that our proposed MTL-FSL framework achieves state-of-the-art performance, outperforming various baseline methods. The implementation source can be found at this https URL. 

**Abstract (ZH)**: 阿尔茨海默病（AD）是老年人群中最常见的神经退行性疾病，对全球医疗保健系统造成了重大且不断上升的负担。虽然多任务学习（MTL）已成为建模纵向AD数据的强大计算范式，但现有框架未能考虑特征相关性的时变性。为解决这一局限性，我们提出了一种新型MTL框架，即特征相似性拉普拉斯图多任务学习（MTL-FSL）。该框架引入了一种新型特征相似性拉普拉斯（FSL）惩罚项，明确地建模了特征间的时变关系。通过同时考虑任务间的时序平滑性和特征间的动态相关性，我们的模型提升了预测准确性和生物学解释性。为解决由我们提议的惩罚项引起的非光滑优化问题，我们采用了交替方向乘子算法（ADMM）。在阿尔茨海默病神经影像学倡议（ADNI）数据集上的实验表明，我们提出的MTL-FSL框架达到了最先进的性能，超越了各种基线方法。源代码可以在以下链接找到：this https URL。 

---
# Hierarchical LoRA MoE for Efficient CTR Model Scaling 

**Title (ZH)**: 层级LoRA MoE用于高效的CTR模型扩展 

**Authors**: Zhichen Zeng, Mengyue Hang, Xiaolong Liu, Xiaoyi Liu, Xiao Lin, Ruizhong Qiu, Tianxin Wei, Zhining Liu, Siyang Yuan, Chaofei Yang, Yiqun Liu, Hang Yin, Jiyan Yang, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10432)  

**Abstract**: Deep models have driven significant advances in click-through rate (CTR) prediction. While vertical scaling via layer stacking improves model expressiveness, the layer-by-layer sequential computation poses challenges to efficient scaling. Conversely, horizontal scaling through Mixture of Experts (MoE) achieves efficient scaling by activating a small subset of experts in parallel, but flat MoE layers may struggle to capture the hierarchical structure inherent in recommendation tasks. To push the Return-On-Investment (ROI) boundary, we explore the complementary strengths of both directions and propose HiLoMoE, a hierarchical LoRA MoE framework that enables holistic scaling in a parameter-efficient manner. Specifically, HiLoMoE employs lightweight rank-1 experts for parameter-efficient horizontal scaling, and stacks multiple MoE layers with hierarchical routing to enable combinatorially diverse expert compositions. Unlike conventional stacking, HiLoMoE routes based on prior layer scores rather than outputs, allowing all layers to execute in parallel. A principled three-stage training framework ensures stable optimization and expert diversity. Experiments on four public datasets show that HiLoMoE achieving better performance-efficiency tradeoff, achieving an average AUC improvement of 0.20\% in AUC and 18.5\% reduction in FLOPs compared to the non-MoE baseline. 

**Abstract (ZH)**: 深度模型在点击率（CTR）预测中推动了重要进展。垂直扩展通过层堆叠提高模型的表现力，但逐层序列计算会面临高效扩展的挑战。相反，通过混合专家（MoE）的水平扩展能够在并行激活少量专家的同时实现高效扩展，但平面MoE层可能难以捕捉推荐任务中固有的层次结构。为推动投资回报率（ROI）边界，我们探索了这两种方向的互补优势，并提出HiLoMoE，这是一种层级LoRA MoE框架，能够在参数有效的方式来实现整体扩展。具体来说，HiLoMoE 使用轻量级的秩1专家来进行参数有效的方式的水平扩展，并通过层级路由堆叠多个MoE层以实现组合多样的专家组成。与传统的堆叠不同，HiLoMoE 基于前一层的分数来路由，而不是输出，使得所有层能够并行执行。合理的三阶段训练框架确保了优化的稳定性和专家多样性。实验结果显示，HiLoMoE 在 AUC 上平均提升了 0.20%，在 FLOPs 上减少了 18.5%，相比非 MoE 基线表现出更好的性能效率折中。 

---
# LONGQAEVAL: Designing Reliable Evaluations of Long-Form Clinical QA under Resource Constraints 

**Title (ZH)**: LONGQAEVAL: 在资源约束条件下设计可靠的长形式临床问答评估 

**Authors**: Federica Bologna, Tiffany Pan, Matthew Wilkens, Yue Guo, Lucy Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10415)  

**Abstract**: Evaluating long-form clinical question answering (QA) systems is resource-intensive and challenging: accurate judgments require medical expertise and achieving consistent human judgments over long-form text is difficult. We introduce LongQAEval, an evaluation framework and set of evaluation recommendations for limited-resource and high-expertise settings. Based on physician annotations of 300 real patient questions answered by physicians and LLMs, we compare coarse answer-level versus fine-grained sentence-level evaluation over the dimensions of correctness, relevance, and safety. We find that inter-annotator agreement (IAA) varies by dimension: fine-grained annotation improves agreement on correctness, coarse improves agreement on relevance, and judgments on safety remain inconsistent. Additionally, annotating only a small subset of sentences can provide reliability comparable to coarse annotations, reducing cost and effort. 

**Abstract (ZH)**: 基于医师标注的长格式临床问答系统评估框架和建议 

---
# Controllable Graph Generation with Diffusion Models via Inference-Time Tree Search Guidance 

**Title (ZH)**: 控制性的图生成方法：基于推断时树搜索指导的扩散模型 

**Authors**: Jiachi Zhao, Zehong Wang, Yamei Liao, Chuxu Zhang, Yanfang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.10402)  

**Abstract**: Graph generation is a fundamental problem in graph learning with broad applications across Web-scale systems, knowledge graphs, and scientific domains such as drug and material discovery. Recent approaches leverage diffusion models for step-by-step generation, yet unconditional diffusion offers little control over desired properties, often leading to unstable quality and difficulty in incorporating new objectives. Inference-time guidance methods mitigate these issues by adjusting the sampling process without retraining, but they remain inherently local, heuristic, and limited in controllability. To overcome these limitations, we propose TreeDiff, a Monte Carlo Tree Search (MCTS) guided dual-space diffusion framework for controllable graph generation. TreeDiff is a plug-and-play inference-time method that expands the search space while keeping computation tractable. Specifically, TreeDiff introduces three key designs to make it practical and scalable: (1) a macro-step expansion strategy that groups multiple denoising updates into a single transition, reducing tree depth and enabling long-horizon exploration; (2) a dual-space denoising mechanism that couples efficient latent-space denoising with lightweight discrete correction in graph space, ensuring both scalability and structural fidelity; and (3) a dual-space verifier that predicts long-term rewards from partially denoised graphs, enabling early value estimation and removing the need for full rollouts. Extensive experiments on 2D and 3D molecular generation benchmarks, under both unconditional and conditional settings, demonstrate that TreeDiff achieves state-of-the-art performance. Notably, TreeDiff exhibits favorable inference-time scaling: it continues to improve with additional computation, while existing inference-time methods plateau early under limited resources. 

**Abstract (ZH)**: 基于树搜索的可控图生成树Diff框架 

---
# Measuring What Matters: Connecting AI Ethics Evaluations to System Attributes, Hazards, and Harms 

**Title (ZH)**: 衡量重要事项：将AI伦理评估与系统属性、风险和危害连接起来 

**Authors**: Shalaleh Rismani, Renee Shelby, Leah Davis, Negar Rostamzadeh, AJung Moon  

**Link**: [PDF](https://arxiv.org/pdf/2510.10339)  

**Abstract**: Over the past decade, an ecosystem of measures has emerged to evaluate the social and ethical implications of AI systems, largely shaped by high-level ethics principles. These measures are developed and used in fragmented ways, without adequate attention to how they are situated in AI systems. In this paper, we examine how existing measures used in the computing literature map to AI system components, attributes, hazards, and harms. Our analysis draws on a scoping review resulting in nearly 800 measures corresponding to 11 AI ethics principles. We find that most measures focus on four principles - fairness, transparency, privacy, and trust - and primarily assess model or output system components. Few measures account for interactions across system elements, and only a narrow set of hazards is typically considered for each harm type. Many measures are disconnected from where harm is experienced and lack guidance for setting meaningful thresholds. These patterns reveal how current evaluation practices remain fragmented, measuring in pieces rather than capturing how harms emerge across systems. Framing measures with respect to system attributes, hazards, and harms can strengthen regulatory oversight, support actionable practices in industry, and ground future research in systems-level understanding. 

**Abstract (ZH)**: 过去十年，已形成一套评估人工智能系统社会和伦理影响的措施体系，这些措施主要受高级伦理原则的影响。这些措施在分散状态下开发和使用，未充分关注其在人工智能系统中的位置。本文探讨了计算文献中现有措施如何映射到人工智能系统的组件、属性、危险和伤害。我们的分析基于范围性回顾，涉及800多项措施，这些措施对应于11个人工智能伦理原则。我们发现，大多数措施侧重于公平性、透明性、隐私性和信任性这四项原则，并主要评估模型或输出系统组件。很少有措施考虑到系统元素之间的交互，每种伤害类型通常只考虑一组狭窄的危险。许多措施与实际体验到的伤害脱节，缺乏设定有意义阈值的指导。这些模式揭示了当前评估实践仍然碎片化，只片段地测量，未能捕捉到跨系统如何出现危害。从系统属性、危险和伤害的角度框架措施可以增强监管监督，支持行业中的可操作实践，并为未来的系统级理解奠定基础。 

---
# Mapping the Urban Mobility Intelligence Frontier: A Scientometric Analysis of Data-Driven Pedestrian Trajectory Prediction and Simulation 

**Title (ZH)**: 城市 Mobility 智能前沿mapping：基于数据驱动的行人轨迹预测与模拟的文献计量分析 

**Authors**: Junhao Xu, Hui Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.10327)  

**Abstract**: Understanding and predicting pedestrian dynamics has become essential for shaping safer, more responsive, and human-centered urban environments. This study conducts a comprehensive scientometric analysis of research on data-driven pedestrian trajectory prediction and crowd simulation, mapping its intellectual evolution and interdisciplinary structure. Using bibliometric data from the Web of Science Core Collection, we employ SciExplorer and Bibliometrix to identify major trends, influential contributors, and emerging frontiers. Results reveal a strong convergence between artificial intelligence, urban informatics, and crowd behavior modeling--driven by graph neural networks, transformers, and generative models. Beyond technical advances, the field increasingly informs urban mobility design, public safety planning, and digital twin development for smart cities. However, challenges remain in ensuring interpretability, inclusivity, and cross-domain transferability. By connecting methodological trajectories with urban applications, this work highlights how data-driven approaches can enrich urban governance and pave the way for adaptive, socially responsible mobility intelligence in future cities. 

**Abstract (ZH)**: 基于数据驱动的行人轨迹预测与人群模拟研究：理解与预测行人动态已成塑造更安全、更响应且以人为本的城市环境的关键。本研究通过文献计量分析，揭示该领域的智力演化和跨学科结构，并采用图神经网络、变压器和生成模型推动的技术进步，探讨其在城市交通设计、公共安全规划和智慧城市数字孪生开发中的应用。然而，提高解释性、包容性和跨域转移性仍是挑战。通过将方法论轨迹与城市应用相结合，本文强调数据驱动方法如何丰富城市治理，并为未来城市的社会责任移动智能铺平道路。 

---
# Prepared for the Unknown: Adapting AIOps Capacity Forecasting Models to Data Changes 

**Title (ZH)**: 未雨绸缪：适应数据变化的AIOps容量预测模型调整 

**Authors**: Lorena Poenaru-Olaru, Wouter van 't Hof, Adrian Stando, Arkadiusz P. Trawinski, Eileen Kapel, Jan S. Rellermeyer, Luis Cruz, Arie van Deursen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10320)  

**Abstract**: Capacity management is critical for software organizations to allocate resources effectively and meet operational demands. An important step in capacity management is predicting future resource needs often relies on data-driven analytics and machine learning (ML) forecasting models, which require frequent retraining to stay relevant as data evolves. Continuously retraining the forecasting models can be expensive and difficult to scale, posing a challenge for engineering teams tasked with balancing accuracy and efficiency. Retraining only when the data changes appears to be a more computationally efficient alternative, but its impact on accuracy requires further investigation. In this work, we investigate the effects of retraining capacity forecasting models for time series based on detected changes in the data compared to periodic retraining. Our results show that drift-based retraining achieves comparable forecasting accuracy to periodic retraining in most cases, making it a cost-effective strategy. However, in cases where data is changing rapidly, periodic retraining is still preferred to maximize the forecasting accuracy. These findings offer actionable insights for software teams to enhance forecasting systems, reducing retraining overhead while maintaining robust performance. 

**Abstract (ZH)**: 基于检测到的数据变化进行容量预测模型的重新训练与周期性重新训练的影响研究：一种成本效益策略 

---
# The algorithmic regulator 

**Title (ZH)**: 算法监管者 

**Authors**: Giulio Ruffini  

**Link**: [PDF](https://arxiv.org/pdf/2510.10300)  

**Abstract**: The regulator theorem states that, under certain conditions, any optimal controller must embody a model of the system it regulates, grounding the idea that controllers embed, explicitly or implicitly, internal models of the controlled. This principle underpins neuroscience and predictive brain theories like the Free-Energy Principle or Kolmogorov/Algorithmic Agent theory. However, the theorem is only proven in limited settings. Here, we treat the deterministic, closed, coupled world-regulator system $(W,R)$ as a single self-delimiting program $p$ via a constant-size wrapper that produces the world output string~$x$ fed to the regulator. We analyze regulation from the viewpoint of the algorithmic complexity of the output, $K(x)$. We define $R$ to be a \emph{good algorithmic regulator} if it \emph{reduces} the algorithmic complexity of the readout relative to a null (unregulated) baseline $\varnothing$, i.e., \[ \Delta = K\big(O_{W,\varnothing}\big) - K\big(O_{W,R}\big) > 0. \] We then prove that the larger $\Delta$ is, the more world-regulator pairs with high mutual algorithmic information are favored. More precisely, a complexity gap $\Delta > 0$ yields \[ \Pr\big((W,R)\mid x\big) \le C\,2^{\,M(W{:}R)}\,2^{-\Delta}, \] making low $M(W{:}R)$ exponentially unlikely as $\Delta$ grows. This is an AIT version of the idea that ``the regulator contains a model of the world.'' The framework is distribution-free, applies to individual sequences, and complements the Internal Model Principle. Beyond this necessity claim, the same coding-theorem calculus singles out a \emph{canonical scalar objective} and implicates a \emph{planner}. On the realized episode, a regulator behaves \emph{as if} it minimized the conditional description length of the readout. 

**Abstract (ZH)**: 调节器定理指出，在某些条件下，任意最优控制器必须包含其所调节系统的模型，从而确立了控制器显式或隐式包含受控系统内部模型的理念。这一原理支撑着神经科学及预测大脑理论，如自由能原理或柯尔莫哥洛夫/算法代理理论。然而，该定理仅在有限条件下得到证明。本文将确定性的闭合耦合世界-调节器系统 $(W,R)$ 视为单一自界定程序 $p$，通过一个恒定大小的包装器产生世界输出字符串 $x$ 供给调节器。从输出的算法复杂性 $K(x)$ 视角分析调节。若调节器 $R$ 能将读取输出的算法复杂性相对于空基线 $\varnothing$ 减小，则定义 $R$ 为一个良好的算法调节器，即 \[ \Delta = K\big(O_{W,\varnothing}\big) - K\big(O_{W,R}\big) > 0. \] 然后证明，$\Delta$ 越大，高互信息的世界-调节器配对就越被青睐。更精确地说，复杂性差距 $\Delta > 0$ 导致 \[ \Pr\big((W,R)\mid x\big) \le C\,2^{\,M(W{:}R)}\,2^{-\Delta}, \] 使得低 $M(W{:}R)$ 随 $\Delta$ 增长而变得指数级不可能。这是基于算法信息论的观点，表明“调节器包含世界模型”的理念。该框架是分布无偏的，适用于单个序列，并补充了内部模型原则。除了这种必要性声明外，同样的编码定理计算还确定了一个标准规范目标，并暗示了一个规划器。在实际化的时间片段中，调节器的行为仿佛是将其读取输出的条件描述长度最小化。 

---
# Unveiling Gamer Archetypes through Multi modal feature Correlations and Unsupervised Learning 

**Title (ZH)**: 通过多模态特征关联与无监督学习揭示游戏 archetype 

**Authors**: Moona Kanwal, Muhammad Sami Siddiqui, Syed Anael Ali  

**Link**: [PDF](https://arxiv.org/pdf/2510.10263)  

**Abstract**: Profiling gamers provides critical insights for adaptive game design, behavioral understanding, and digital well-being. This study proposes an integrated, data-driven framework that combines psychological measures, behavioral analytics, and machine learning to reveal underlying gamer personas. A structured survey of 250 participants, including 113 active gamers, captured multidimensional behavioral, motivational, and social data. The analysis pipeline integrated feature engineering, association-network, knowledge-graph analysis, and unsupervised clustering to extract meaningful patterns. Correlation statistics uses Cramers V, Tschuprows T, Theils U, and Spearmans quantified feature associations, and network centrality guided feature selection. Dimensionality-reduction techniques such as PCA, SVD, t-SNE are coupled with clustering algorithms like K-Means, Agglomerative, Spectral, DBSCAN, evaluated using Silhouette, Calinski Harabasz, and Davies Bouldin indices. The PCA with K-Means with k = 4 model achieved optimal cluster quality with Silhouette = 0.4, identifying four archetypes as Immersive Social Story-Seekers, Disciplined Optimizers, Strategic Systems Navigators, and Competitive Team-Builders. This research contributes a reproducible pipeline that links correlation-driven network insights with unsupervised learning. The integration of behavioral correlation networks with clustering not only enhances classification accuracy but also offers a holistic lens to connect gameplay motivations with psychological and wellness outcomes. 

**Abstract (ZH)**: 基于数据的综合框架揭示游戏者特征：适应性游戏设计、行为理解与数字福祉的研究 

---
# SGM: A Statistical Godel Machine for Risk-Controlled Recursive Self-Modification 

**Title (ZH)**: SGM：一种用于风险控制的递归自我修改统计哥德尔机 

**Authors**: Xuening Wu, Shenqin Yin, Yanlan Kang, Xinhang Zhang, Qianya Xu, Zeping Chen, Wenqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10232)  

**Abstract**: Recursive self-modification is increasingly central in AutoML, neural architecture search, and adaptive optimization, yet no existing framework ensures that such changes are made safely. Godel machines offer a principled safeguard by requiring formal proofs of improvement before rewriting code; however, such proofs are unattainable in stochastic, high-dimensional settings. We introduce the Statistical Godel Machine (SGM), the first statistical safety layer for recursive edits. SGM replaces proof-based requirements with statistical confidence tests (e-values, Hoeffding bounds), admitting a modification only when superiority is certified at a chosen confidence level, while allocating a global error budget to bound cumulative risk across this http URL also propose Confirm-Triggered Harmonic Spending (CTHS), which indexes spending by confirmation events rather than rounds, concentrating the error budget on promising edits while preserving familywise this http URL across supervised learning, reinforcement learning, and black-box optimization validate this role: SGM certifies genuine gains on CIFAR-100, rejects spurious improvement on ImageNet-100, and demonstrates robustness on RL and optimization this http URL, these results position SGM as foundational infrastructure for continual, risk-aware self-modification in learning this http URL is available at: this https URL. 

**Abstract (ZH)**: 统计哥德尔机（SGM）：递归编辑的统计安全性层 

---
# Learning to Guarantee Type Correctness in Code Generation through Type-Guided Program Synthesis 

**Title (ZH)**: 通过类型引导的程序合成学习保证代码生成中的类型正确性 

**Authors**: Zhechong Huang, Zhao Zhang, Ruyi Ji, Tingxuan Xia, Qihao Zhu, Qinxiang Cao, Zeyu Sun, Yingfei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.10216)  

**Abstract**: Language models have shown remarkable proficiency in code generation; nevertheless, ensuring type correctness remains a challenge. Although traditional methods, such as constrained decoding, alleviate this problem by externally rejecting untypable code, the model itself does not effectively learn type reasoning internally, which ultimately limits its overall performance. This paper introduces TyFlow, a novel system that internalizes type reasoning within code generation to guide the model to learn the type system. The core of our approach is a novel type-guided program synthesis system that maintains an isomorphism between type derivation trees and synthesis derivation trees, enabling a new code representation based on synthesis decision sequences rather than traditional text-based token sequences. By offloading the complexity of type system learning to the representation itself, models can redirect their computational resources toward higher-level program semantics. Our evaluation shows that TyFlow not only eliminates type errors but also significantly improves functional correctness, highlighting the importance of aligning LMs with type systems internally. 

**Abstract (ZH)**: 语言模型在代码生成方面展现了出色的 proficiency，但在确保类型正确性方面仍面临挑战。尽管传统方法，如受约束解码，通过外部拒绝未类型化的代码来缓解这一问题，但模型本身并没有有效地在内部学习类型推理，这最终限制了其整体性能。本文介绍了一种名为 TyFlow 的新系统，该系统将类型推理内置于代码生成中以引导模型学习类型系统。我们方法的核心是一种新型类型的程序合成系统，它保持了类型演绎树和合成演绎树之间的同构性，从而提供了一种基于合成决策序列的新代码表示，而不是传统的基于文本的标记序列。通过将类型系统学习的复杂性转移到表示本身，模型可以将计算资源重新分配给更高层次的程序语义。我们的评估表明，TyFlow 不仅消除了类型错误，还显著提高了功能性正确性，突显了内部对齐语言模型和类型系统的重要性。 

---
# Distributionally Robust Control with End-to-End Statistically Guaranteed Metric Learning 

**Title (ZH)**: 端到端统计保证的度量学习下的分布鲁棒控制 

**Authors**: Jingyi Wu, Chao Ning, Yang Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.10214)  

**Abstract**: Wasserstein distributionally robust control (DRC) recently emerges as a principled paradigm for handling uncertainty in stochastic dynamical systems. However, it constructs data-driven ambiguity sets via uniform distribution shifts before sequentially incorporating them into downstream control synthesis. This segregation between ambiguity set construction and control objectives inherently introduces a structural misalignment, which undesirably leads to conservative control policies with sub-optimal performance. To address this limitation, we propose a novel end-to-end finite-horizon Wasserstein DRC framework that integrates the learning of anisotropic Wasserstein metrics with downstream control tasks in a closed-loop manner, thus enabling ambiguity sets to be systematically adjusted along performance-critical directions and yielding more effective control policies. This framework is formulated as a bilevel program: the inner level characterizes dynamical system evolution under DRC, while the outer level refines the anisotropic metric leveraging control-performance feedback across a range of initial conditions. To solve this program efficiently, we develop a stochastic augmented Lagrangian algorithm tailored to the bilevel structure. Theoretically, we prove that the learned ambiguity sets preserve statistical finite-sample guarantees under a novel radius adjustment mechanism, and we establish the well-posedness of the bilevel formulation by demonstrating its continuity with respect to the learnable metric. Furthermore, we show that the algorithm converges to stationary points of the outer level problem, which are statistically consistent with the optimal metric at a non-asymptotic convergence rate. Experiments on both numerical and inventory control tasks verify that the proposed framework achieves superior closed-loop performance and robustness compared against state-of-the-art methods. 

**Abstract (ZH)**: Wasserstein 分布鲁棒控制 (DRC) 最近作为处理随机动力系统中不确定性的一种基本原则范式而崭露头角。然而，它在逐步将不确定性集合纳入下游控制合成之前，通过均匀分布偏移构造数据驱动的不确定性集合。这种不确定性集合构建与控制目标之间的分离固有地引入了结构不对齐，从而导致保守的控制策略，其性能次优。为解决这一局限，我们提出了一种新的端到端有限时限 Wasserstein DRC 框架，该框架以闭环方式将各向异性 Wasserstein 度量的学习与下游控制任务结合起来，从而使得不确定性集合能够沿着性能关键方向系统性地调整，并产生更具效用的控制策略。该框架被形式化为一个 bilevel 程序：内部层级在 DRC 下刻画动力系统演化，外部层级利用控制-性能反馈在多种初始条件下细化各向异性度量。为了高效求解该程序，我们开发了一种针对 bilevel 结构定制的随机增广拉格朗日算法。理论上，我们证明了通过新颖的半径调整机制学习到的不确定性集合在有限样本下保持统计保证，并通过证明其关于可学习度量的连续性来建立 bilevel 表述的适定性。此外，我们证明该算法收敛于外部层级问题的稳定点，这些稳定点以非渐近收敛速率与最优度量在统计上一致。实验结果表明，所提出框架在闭环性能和鲁棒性方面优于最先进的方法。 

---
# Revisiting Trust in the Era of Generative AI: Factorial Structure and Latent Profiles 

**Title (ZH)**: 重访生成式AI时代的信任：因子结构与潜在profile探究 

**Authors**: Haocan Sun, Weizi Liu, Di Wu, Guoming Yu, Mike Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.10199)  

**Abstract**: Trust is one of the most important factors shaping whether and how people adopt and rely on artificial intelligence (AI). Yet most existing studies measure trust in terms of functionality, focusing on whether a system is reliable, accurate, or easy to use, while giving less attention to the social and emotional dimensions that are increasingly relevant for today's generative AI (GenAI) systems. These systems do not just process information; they converse, respond, and collaborate with users, blurring the line between tool and partner. In this study, we introduce and validate the Human-AI Trust Scale (HAITS), a new measure designed to capture both the rational and relational aspects of trust in GenAI. Drawing on prior trust theories, qualitative interviews, and two waves of large-scale surveys in China and the United States, we used exploratory (n = 1,546) and confirmatory (n = 1,426) factor analyses to identify four key dimensions of trust: Affective Trust, Competence Trust, Benevolence & Integrity, and Perceived Risk. We then applied latent profile analysis to classify users into six distinct trust profiles, revealing meaningful differences in how affective-competence trust and trust-distrust frameworks coexist across individuals and cultures. Our findings offer a validated, culturally sensitive tool for measuring trust in GenAI and provide new insight into how trust evolves in human-AI interaction. By integrating instrumental and relational perspectives of trust, this work lays the foundation for more nuanced research and design of trustworthy AI systems. 

**Abstract (ZH)**: 人类与生成型人工智能的信任量表：兼顾理性和关系维度的信任测量 

---
# CauchyNet: Compact and Data-Efficient Learning using Holomorphic Activation Functions 

**Title (ZH)**: 柯西网络：使用全纯激活函数的紧凑且数据高效学习 

**Authors**: Hong-Kun Zhang, Xin Li, Sikun Yang, Zhihong Xia  

**Link**: [PDF](https://arxiv.org/pdf/2510.10195)  

**Abstract**: A novel neural network inspired by Cauchy's integral formula, is proposed for function approximation tasks that include time series forecasting, missing data imputation, etc. Hence, the novel neural network is named CauchyNet. By embedding real-valued data into the complex plane, CauchyNet efficiently captures complex temporal dependencies, surpassing traditional real-valued models in both predictive performance and computational efficiency. Grounded in Cauchy's integral formula and supported by the universal approximation theorem, CauchyNet offers strong theoretical guarantees for function approximation. The architecture incorporates complex-valued activation functions, enabling robust learning from incomplete data while maintaining a compact parameter footprint and reducing computational overhead. Through extensive experiments in diverse domains, including transportation, energy consumption, and epidemiological data, CauchyNet consistently outperforms state-of-the-art models in predictive accuracy, often achieving a 50% lower mean absolute error with fewer parameters. These findings highlight CauchyNet's potential as an effective and efficient tool for data-driven predictive modeling, particularly in resource-constrained and data-scarce environments. 

**Abstract (ZH)**: 一种受柯西积分公式启发的新型神经网络CauchyNet用于函数近似任务，包括时间序列预测、缺失数据插补等，通过将实值数据嵌入复平面中，CauchyNet高效地捕捉复杂的时间依赖关系，性能和计算效率均超过传统的实值模型。基于柯西积分公式和普遍逼近定理，CauchyNet为函数近似提供了强大的理论保证。其架构采用复值激活函数，能够在保持紧凑参数量和减少计算开销的同时，从不完整数据中获得稳健的learnings。通过在交通、能源消耗和流行病学数据等多个领域的广泛实验，CauchyNet在预测准确性上始终优于现有最佳模型，参数量更少时误差均降低50%。这些发现突显了CauchyNet作为数据驱动预测建模的有效且高效的工具的潜力，特别是在资源受限和数据稀缺的环境中。 

---
# Formally Verified Certification of Unsolvability of Temporal Planning Problems 

**Title (ZH)**: 形式化验证不可解时态规划问题的认证 

**Authors**: David Wang, Mohammad Abdulaziz  

**Link**: [PDF](https://arxiv.org/pdf/2510.10189)  

**Abstract**: We present an approach to unsolvability certification of temporal planning. Our approach is based on encoding the planning problem into a network of timed automata, and then using an efficient model checker on the network followed by a certificate checker to certify the output of the model checker. Our approach prioritises trustworthiness of the certification: we formally verify our implementation of the encoding to timed automata using the theorem prover Isabelle/HOL and we use an existing certificate checker (also formally verified in Isabelle/HOL) to certify the model checking result. 

**Abstract (ZH)**: 时间规划不可解性认证的方法：将规划问题编码为时间自动机网络，并利用高效模型检查器进行检查，随后使用已正式验证的证书检查器认证模型检查结果。 

---
# Multi-Scale Diffusion Transformer for Jointly Simulating User Mobility and Mobile Traffic Pattern 

**Title (ZH)**: 多尺度扩散变换器联合模拟用户移动性和移动流量模式 

**Authors**: Ziyi Liu, Qingyue Long, Zhiwen Xue, Huandong Wang, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.10158)  

**Abstract**: User mobility trajectory and mobile traffic data are essential for a wide spectrum of applications including urban planning, network optimization, and emergency management. However, large-scale and fine-grained mobility data remains difficult to obtain due to privacy concerns and collection costs, making it essential to simulate realistic mobility and traffic patterns. User trajectories and mobile traffic are fundamentally coupled, reflecting both physical mobility and cyber behavior in urban environments. Despite this strong interdependence, existing studies often model them separately, limiting the ability to capture cross-modal dynamics. Therefore, a unified framework is crucial. In this paper, we propose MSTDiff, a Multi-Scale Diffusion Transformer for joint simulation of mobile traffic and user trajectories. First, MSTDiff applies discrete wavelet transforms for multi-resolution traffic decomposition. Second, it uses a hybrid denoising network to process continuous traffic volumes and discrete location sequences. A transition mechanism based on urban knowledge graph embedding similarity is designed to guide semantically informed trajectory generation. Finally, a multi-scale Transformer with cross-attention captures dependencies between trajectories and traffic. Experiments show that MSTDiff surpasses state-of-the-art baselines in traffic and trajectory generation tasks, reducing Jensen-Shannon divergence (JSD) across key statistical metrics by up to 17.38% for traffic generation, and by an average of 39.53% for trajectory generation. The source code is available at: this https URL . 

**Abstract (ZH)**: 基于多尺度扩散变换器的移动交通和用户轨迹联合仿真 

---
# A Unified Frequency Domain Decomposition Framework for Interpretable and Robust Time Series Forecasting 

**Title (ZH)**: 统一的频域分解框架：可解释性和稳健性时间序列预测 

**Authors**: Cheng He, Xijie Liang, Zengrong Zheng, Patrick P.C. Lee, Xu Huang, Zhaoyi Li, Hong Xie, Defu Lian, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10145)  

**Abstract**: Current approaches for time series forecasting, whether in the time or frequency domain, predominantly use deep learning models based on linear layers or transformers. They often encode time series data in a black-box manner and rely on trial-and-error optimization solely based on forecasting performance, leading to limited interpretability and theoretical understanding. Furthermore, the dynamics in data distribution over time and frequency domains pose a critical challenge to accurate forecasting. We propose FIRE, a unified frequency domain decomposition framework that provides a mathematical abstraction for diverse types of time series, so as to achieve interpretable and robust time series forecasting. FIRE introduces several key innovations: (i) independent modeling of amplitude and phase components, (ii) adaptive learning of weights of frequency basis components, (iii) a targeted loss function, and (iv) a novel training paradigm for sparse data. Extensive experiments demonstrate that FIRE consistently outperforms state-of-the-art models on long-term forecasting benchmarks, achieving superior predictive performance and significantly enhancing interpretability of time series 

**Abstract (ZH)**: FIRE：频率域分解统一框架实现可解释且稳健的时间序列预测 

---
# Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction Under Copy-heavy Task 

**Title (ZH)**: 企业规模文档信息提取的OCR-LLM混合框架：以复制任务为主的工作负载 

**Authors**: Zilong Wang, Xiaoyu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.10138)  

**Abstract**: Information extraction from copy-heavy documents, characterized by massive volumes of structurally similar content, represents a critical yet understudied challenge in enterprise document processing. We present a systematic framework that strategically combines OCR engines with Large Language Models (LLMs) to optimize the accuracy-efficiency trade-off inherent in repetitive document extraction tasks. Unlike existing approaches that pursue universal solutions, our method exploits document-specific characteristics through intelligent strategy selection. We implement and evaluate 25 configurations across three extraction paradigms (direct, replacement, and table-based) on identity documents spanning four formats (PNG, DOCX, XLSX, PDF). Through table-based extraction methods, our adaptive framework delivers outstanding results: F1=1.0 accuracy with 0.97s latency for structured documents, and F1=0.997 accuracy with 0.6 s for challenging image inputs when integrated with PaddleOCR, all while maintaining sub-second processing speeds. The 54 times performance improvement compared with multimodal methods over naive approaches, coupled with format-aware routing, enables processing of heterogeneous document streams at production scale. Beyond the specific application to identity extraction, this work establishes a general principle: the repetitive nature of copy-heavy tasks can be transformed from a computational burden into an optimization opportunity through structure-aware method selection. 

**Abstract (ZH)**: 从副本内容丰富的文档中提取信息：一种结合OCR引擎与大型语言模型的系统框架及其在企业文档处理中的应用 

---
# CacheClip: Accelerating RAG with Effective KV Cache Reuse 

**Title (ZH)**: CacheClip: 加速RAG的有效KV缓存重用 

**Authors**: Bin Yang, Qiuyu Leng, Jun Zeng, Zhenhua Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10129)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems suffer from severe time-to-first-token (TTFT) bottlenecks due to long input sequences. Existing KV cache reuse methods face a fundamental trade-off: prefix caching requires identical prefixes that rarely occur in RAG scenarios, while direct precomputation sacrifices quality due to missing inter-chunk attention and repeated attention sinks. Recent methods like APE and CacheBlend partially address these issues but remain inadequate for robust RAG applications. This paper presents CacheClip, a novel framework that achieves both fast TTFT and high generation quality. Our key insight is that small auxiliary LLMs exhibit similar last-layer attention distributions to primary LLMs (the target model for generation), enabling efficient identification of tokens critical for restoring inter-chunk attention, thereby significantly improving response quality on cross-chunk reasoning tasks. CacheClip integrates three techniques: (1) auxiliary-model-guided token selection for selective KV cache recomputation, where the auxiliary model is finetuned to improve selection accuracy, (2) shared prefixes to eliminate redundant attention sinks, and (3) grouping strategy to maintain local coherence during partial KV cache updates. Experiments show CacheClip retains up to 94.8% and 85.0% of full-attention performance on NIAH and LongBench, outperforming APE and CacheBlend by 25.2% and 35.1% on NIAH (with reomp% = 20%). Meanwhile, CacheClip accelerates LLM inference by up to 1.92x in prefill time, providing a practical solution to the efficiency-quality trade-off in RAG systems. 

**Abstract (ZH)**: 基于检索的生成（RAG）系统因长输入序列而遭受严重的首个token响应时间（TTFT）瓶颈。现有的KV缓存复用方法面临一个根本性的权衡：前缀缓存需要几乎不出现于RAG场景中的相同前缀，而直接预计算会因为缺少跨块注意和重复的注意阱而牺牲质量。最近的方法如APE和CacheBlend部分解决了这些问题，但对于鲁棒的RAG应用仍不够完善。本文提出了CacheClip，这是一种新颖的框架，能够同时实现快速的TTFT和高质量的生成。我们的关键洞察是，辅助LLM在最后一层注意力分布上与主要LLM（用于生成的目标模型）表现出相似性，这使我们能够有效识别对于恢复跨块注意力至关重要的token，从而显著提高跨块推理任务的响应质量。CacheClip结合了三种技术：（1）辅助模型引导的token选择以选择性地重新计算KV缓存，其中辅助模型微调以提高选择准确性；（2）共享前缀以消除冗余注意陷阱；（3）分组策略，在部分KV缓存更新过程中维持局部连贯性。实验结果显示，CacheClip在NIAH和LongBench上分别保留了94.8%和85.0%的全注意力性能，相较于APE和CacheBlend在NIAH上分别提高了25.2%和35.1%（凭借20%的重新计算比例）。同时，CacheClip将LLM推理的预填充时间加速了1.92倍，为RAG系统中的效率与质量权衡提供了一个实用的解决方案。 

---
# Uncovering Singularities in Feynman Integrals via Machine Learning 

**Title (ZH)**: 通过机器学习发现费曼积分中的奇点 

**Authors**: Yuanche Liu, Yingxuan Xu, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10099)  

**Abstract**: We introduce a machine-learning framework based on symbolic regression to extract the full symbol alphabet of multi-loop Feynman integrals. By targeting the analytic structure rather than reduction, the method is broadly applicable and interpretable across different families of integrals. It successfully reconstructs complete symbol alphabets in nontrivial examples, demonstrating both robustness and generality. Beyond accelerating computations case by case, it uncovers the analytic structure universally. This framework opens new avenues for multi-loop amplitude analysis and provides a versatile tool for exploring scattering amplitudes. 

**Abstract (ZH)**: 基于符号回归的机器学习框架用于提取多环费曼积分的完整符号字母表 

---
# What Makes Looped Transformers Perform Better Than Non-Recursive Ones (Provably) 

**Title (ZH)**: Looped Transformer相较于非递归Transformer优越的原理分析 

**Authors**: Zixuan Gong, Jiaye Teng, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10089)  

**Abstract**: While looped transformers (termed as Looped-Attn) often outperform standard transformers (termed as Single-Attn) on complex reasoning tasks, the theoretical basis for this advantage remains underexplored. In this paper, we explain this phenomenon through the lens of loss landscape geometry, inspired by empirical observations of their distinct dynamics at both sample and Hessian levels. To formalize this, we extend the River-Valley landscape model by distinguishing between U-shaped valleys (flat) and V-shaped valleys (steep). Based on empirical observations, we conjecture that the recursive architecture of Looped-Attn induces a landscape-level inductive bias towards River-V-Valley. Theoretical derivations based on this inductive bias guarantee a better loss convergence along the river due to valley hopping, and further encourage learning about complex patterns compared to the River-U-Valley induced by Single-Attn. Building on this insight, we propose SHIFT (Staged HIerarchical Framework for Progressive Training), a staged training framework that accelerates the training process of Looped-Attn while achieving comparable performances. 

**Abstract (ZH)**: 循环变压器（记作 Looped-Attn）在复杂推理任务上通常优于标准变压器（记作 Single-Attn），但其理论优势的基础仍需进一步探索。本文通过损失景观几何结构的视角来解释这一现象，受到其在样本和海森berg矩阵层面不同动态的实证观察的启发。为了形式化这一分析，我们扩展了河流-山谷景观模型，区分U形山谷（平坦）和V形山谷（陡峭）。基于实证观察，我们推测 Looped-Attn 的递归架构诱导了在景观层面倾向于河流-山谷类型的归纳偏置。基于这一归纳偏置的理论推导保证了沿河流方向更好的损失收敛，通过山谷跳跃进一步促进了复杂模式的学习，而单感知器（Single-Attn）诱导的河流-山谷类型则不如前者。基于此洞见，我们提出了一种分阶段层次框架（SHIFT），以加速循环变换器（Looped-Attn）的训练过程并实现相当的性能。 

---
# How AI Companionship Develops: Evidence from a Longitudinal Study 

**Title (ZH)**: AI伴侣的发展：纵向研究的实证证据 

**Authors**: Angel Hsing-Chi Hwang, Fiona Li, Jacy Reese Anthis, Hayoun Noh  

**Link**: [PDF](https://arxiv.org/pdf/2510.10079)  

**Abstract**: The quickly growing popularity of AI companions poses risks to mental health, personal wellbeing, and social relationships. Past work has identified many individual factors that can drive human-companion interaction, but we know little about how these factors interact and evolve over time. In Study 1, we surveyed AI companion users (N = 303) to map the psychological pathway from users' mental models of the agent to parasocial experiences, social interaction, and the psychological impact of AI companions. Participants' responses foregrounded multiple interconnected variables (agency, parasocial interaction, and engagement) that shape AI companionship. In Study 2, we conducted a longitudinal study with a subset of participants (N = 110) using a new generic chatbot. Participants' perceptions of the generic chatbot significantly converged to perceptions of their own companions by Week 3. These results suggest a longitudinal model of AI companionship development and demonstrate an empirical method to study human-AI companionship. 

**Abstract (ZH)**: AI伴侣的迅速流行对心理健康、个人福祉和社会关系构成风险：心理路径研究与纵向发展模型 

---
# Gradient-based Model Shortcut Detection for Time Series Classification 

**Title (ZH)**: 基于梯度的模型捷径检测时间序列分类 

**Authors**: Salomon Ibarra, Frida Cantu, Kaixiong Zhou, Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10075)  

**Abstract**: Deep learning models have attracted lots of research attention in time series classification (TSC) task in the past two decades. Recently, deep neural networks (DNN) have surpassed classical distance-based methods and achieved state-of-the-art performance. Despite their promising performance, deep neural networks (DNNs) have been shown to rely on spurious correlations present in the training data, which can hinder generalization. For instance, a model might incorrectly associate the presence of grass with the label ``cat" if the training set have majority of cats lying in grassy backgrounds. However, the shortcut behavior of DNNs in time series remain under-explored. Most existing shortcut work are relying on external attributes such as gender, patients group, instead of focus on the internal bias behavior in time series models.
In this paper, we take the first step to investigate and establish point-based shortcut learning behavior in deep learning time series classification. We further propose a simple detection method based on other class to detect shortcut occurs without relying on test data or clean training classes. We test our proposed method in UCR time series datasets. 

**Abstract (ZH)**: 深度学习模型在时间序列分类任务中吸引了过去二十多年来的大量研究关注。近年来，深度神经网络（DNN）超越了经典的基于距离的方法，并取得了最先进的性能。尽管表现出色，但深度神经网络（DNNs）已被证明依赖于训练数据中存在的虚假关联，这可能阻碍泛化能力。例如，一个模型可能会错误地将草的存在关联到“猫”这个标签，如果训练集中大部分猫都处于草地背景中。然而，时间序列中深度神经网络的捷径行为尚未得到充分探索。现有的大多数捷径工作依赖于外部属性，如性别、患者组，而不是专注于时间序列模型中的内部偏差行为。在本文中，我们首次探索并建立了基于点的时间序列分类深度学习中的捷径学习行为。我们进一步提出了一种基于其他类别的简单检测方法，以无需依赖测试数据或干净的训练类别来检测捷径现象。我们在UCR时间序列数据集上测试了我们提出的检测方法。 

---
# CLMN: Concept based Language Models via Neural Symbolic Reasoning 

**Title (ZH)**: CLMN：基于概念的神经符号推理语言模型 

**Authors**: Yibo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.10063)  

**Abstract**: Deep learning has advanced NLP, but interpretability remains limited, especially in healthcare and finance. Concept bottleneck models tie predictions to human concepts in vision, but NLP versions either use binary activations that harm text representations or latent concepts that weaken semantics, and they rarely model dynamic concept interactions such as negation and context. We introduce the Concept Language Model Network (CLMN), a neural-symbolic framework that keeps both performance and interpretability. CLMN represents concepts as continuous, human-readable embeddings and applies fuzzy-logic reasoning to learn adaptive interaction rules that state how concepts affect each other and the final decision. The model augments original text features with concept-aware representations and automatically induces interpretable logic rules. Across multiple datasets and pre-trained language models, CLMN achieves higher accuracy than existing concept-based methods while improving explanation quality. These results show that integrating neural representations with symbolic reasoning in a unified concept space can yield practical, transparent NLP systems. 

**Abstract (ZH)**: 深度学习推动了自然语言处理的发展，但在医疗保健和金融等领域中的可解释性依然有限。视觉领域的概念瓶颈模型能够将预测与人类概念联系起来，但对于自然语言处理版本，要么使用损害文本表示的二值激活函数，要么使用削弱语义的潜在概念，并且它们很少建模如否定和背景这类动态概念交互。我们提出了概念语言模型网络（CLMN），这是一种在保持性能的同时提高可解释性的神经符号框架。CLMN 将概念表示为连续的人类可读嵌入，并应用模糊逻辑推理来学习适应性的交互规则，以阐释概念如何相互影响以及最终决策。该模型通过概念感知表示增强原始文本特征，并自动生成可解释的逻辑规则。在多个数据集和预训练语言模型上，CLMN 较现有的基于概念的方法具有更高的准确率，同时提高了解释质量。这些结果表明，在统一的概念空间中结合神经表示与符号推理可以产生实用且透明的自然语言处理系统。 

---
# FOSSIL: Regret-Minimizing Curriculum Learning for Metadata-Free and Low-Data Mpox Diagnosis 

**Title (ZH)**: F OSSIL: 基于后悔最小化的 Curriculum 学习方法用于无元数据和少数据猴痘诊断 

**Authors**: Sahng-Min Han, Minjae Kim, Jinho Cha, Se-woon Choe, Eunchan Daniel Cha, Jungwon Choi, Kyudong Jung  

**Link**: [PDF](https://arxiv.org/pdf/2510.10041)  

**Abstract**: Deep learning in small and imbalanced biomedical datasets remains fundamentally constrained by unstable optimization and poor generalization. We present the first biomedical implementation of FOSSIL (Flexible Optimization via Sample-Sensitive Importance Learning), a regret-minimizing weighting framework that adaptively balances training emphasis according to sample difficulty. Using softmax-based uncertainty as a continuous measure of difficulty, we construct a four-stage curriculum (Easy-Very Hard) and integrate FOSSIL into both convolutional and transformer-based architectures for Mpox skin lesion diagnosis. Across all settings, FOSSIL substantially improves discrimination (AUC = 0.9573), calibration (ECE = 0.053), and robustness under real-world perturbations, outperforming conventional baselines without metadata, manual curation, or synthetic augmentation. The results position FOSSIL as a generalizable, data-efficient, and interpretable framework for difficulty-aware learning in medical imaging under data scarcity. 

**Abstract (ZH)**: FOSSIL在生物医学小规模和不平衡数据集中的深度学习应用:一种基于样本敏感重要性学习的后悔最小化加权框架及其在Mpox皮肤病变诊断中的应用 

---
# Lightweight Baselines for Medical Abstract Classification: DistilBERT with Cross-Entropy as a Strong Default 

**Title (ZH)**: 轻量级医学摘要分类基础模型：基于交叉熵的DistilBERT 

**Authors**: Jiaqi Liu, Lanruo Wang, Su Liu, Xin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.10025)  

**Abstract**: Large language models work well for many NLP tasks, but they are hard to deploy in health settings with strict cost, latency, and privacy limits. We revisit a lightweight recipe for medical abstract classification and ask how far compact encoders can go under a controlled budget. Using the public medical abstracts corpus, we finetune BERT base and DistilBERT with three objectives standard cross-entropy, class weighted cross entropy, and focal loss keeping tokenizer, sequence length, optimizer, and schedule fixed. DistilBERT with plain cross-entropy gives the best balance on the test set while using far fewer parameters than BERT base. We report accuracy, Macro F1, and Weighted F1, release the evaluation code, and include confusion analyses to make error patterns clear. Our results suggest a practical default: start with a compact encoder and cross-entropy, then add calibration and task-specific checks before moving to heavier models. 

**Abstract (ZH)**: 轻量级模型在医疗环境中的NLP任务应用：在严格的成本、延迟和隐私限制下，大规模语言模型表现良好，但在医疗环境中部署仍面临挑战。我们重新审视一种轻量级的医疗摘要分类方法，并在控制预算下探讨紧凑型编码器的极限。利用公开的医疗摘要语料库，我们使用标准交叉熵、类权重交叉熵和焦损函数对BERT基模型和DistilBERT进行微调，保持分词器、序列长度、优化器和调度一致。DistilBERT在标准交叉熵下的表现，在测试集上兼顾准确性和参数效率。我们报告准确率、宏F1和加权F1分数，发布评估代码，并包含混淆矩阵分析以清晰展现错误模式。研究结果建议一种实用的默认方案：使用紧凑型编码器和交叉熵开始，然后添加校准和任务特定检查，再过渡到更重的模型。 

---
# Neuro-inspired automated lens design 

**Title (ZH)**: 神经启发的自动透镜设计 

**Authors**: Yao Gao, Lei Sun, Shaohua Gao, Qi Jiang, Kailun Yang, Weijian Hu, Xiaolong Qian, Wenyong Li, Luc Van Gool, Kaiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09979)  

**Abstract**: The highly non-convex optimization landscape of modern lens design necessitates extensive human expertise, resulting in inefficiency and constrained design diversity. While automated methods are desirable, existing approaches remain limited to simple tasks or produce complex lenses with suboptimal image quality. Drawing inspiration from the synaptic pruning mechanism in mammalian neural development, this study proposes OptiNeuro--a novel automated lens design framework that first generates diverse initial structures and then progressively eliminates low-performance lenses while refining remaining candidates through gradient-based optimization. By fully automating the design of complex aspheric imaging lenses, OptiNeuro demonstrates quasi-human-level performance, identifying multiple viable candidates with minimal human intervention. This advancement not only enhances the automation level and efficiency of lens design but also facilitates the exploration of previously uncharted lens architectures. 

**Abstract (ZH)**: 现代透镜设计高度非凸优化景观需要大量的专家知识，导致设计效率低下和设计多样性受限。虽然自动化方法是理想的，但现有方法仍局限于简单任务或产生具有次优成像质量的复杂透镜。受哺乳动物神经发育中突触修剪机制的启发，本研究提出了OptiNeuro——一种新颖的自动化透镜设计框架，该框架首先生成多样化初始结构，然后通过基于梯度的优化逐步淘汰低性能透镜并精炼剩余候选透镜。通过完全自动化复杂非球面成像透镜的设计，OptiNeuro展示了接近人类水平的表现，极少的人为干预即可识别出多个可行候选透镜。这一进展不仅提高了透镜设计的自动化程度和效率，还促进了对未探索透镜架构的探索。 

---
# Operationalizing AI: Empirical Evidence on MLOps Practices, User Satisfaction, and Organizational Context 

**Title (ZH)**: 将AI应用于实践：MLOps实践、用户满意度及组织背景的实证研究 

**Authors**: Stefan Pasch  

**Link**: [PDF](https://arxiv.org/pdf/2510.09968)  

**Abstract**: Organizational efforts to utilize and operationalize artificial intelligence (AI) are often accompanied by substantial challenges, including scalability, maintenance, and coordination across teams. In response, the concept of Machine Learning Operations (MLOps) has emerged as a set of best practices that integrate software engineering principles with the unique demands of managing the ML lifecycle. Yet, empirical evidence on whether and how these practices support users in developing and operationalizing AI applications remains limited. To address this gap, this study analyzes over 8,000 user reviews of AI development platforms from this http URL. Using zero-shot classification, we measure review sentiment toward nine established MLOps practices, including continuous integration and delivery (CI/CD), workflow orchestration, reproducibility, versioning, collaboration, and monitoring. Seven of the nine practices show a significant positive relationship with user satisfaction, suggesting that effective MLOps implementation contributes tangible value to AI development. However, organizational context also matters: reviewers from small firms discuss certain MLOps practices less frequently, suggesting that organizational context influences the prevalence and salience of MLOps, though firm size does not moderate the MLOps-satisfaction link. This indicates that once applied, MLOps practices are perceived as universally beneficial across organizational settings. 

**Abstract (ZH)**: 组织利用和实施人工智能（AI）的努力通常伴随着可扩展性、维护和团队间协调等方面的显著挑战。为此，Machine Learning Operations (MLOps) 的概念作为一套最佳实践逐渐形成，它结合了软件工程原则以适应管理机器学习生命周期的独特需求。然而，关于这些实践如何支持用户开发和实施AI应用程序的实证证据仍有限。为填补这一空白，本研究分析了来自此链接的超过8,000条AI开发平台用户评论。利用零样本分类，我们衡量了用户对九项公认MLOps实践的评价，包括持续集成和交付（CI/CD）、工作流编排、可再现性、版本控制、协作和监控。在这九项实践中，有七项显示出与用户满意度之间存在显著正相关关系，表明有效的MLOps实践为AI开发带来了实际价值。然而，组织背景也非常重要：小型企业的评论者讨论某些MLOps实践的频率较低，表明组织背景影响MLOps的普及度和显着性，尽管公司规模并未影响MLOps-满意度之间的联系。这表明一旦实施，MLOps实践在各类组织环境中被视为普遍有益。 

---
# Homomorphic Mappings for Value-Preserving State Aggregation in Markov Decision Processes 

**Title (ZH)**: 同态映射在马尔可夫决策过程中的值保持状态聚合 

**Authors**: Shuo Zhao, Yongqiang Li, Yu Feng, Zhongsheng Hou, Yuanjing Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09965)  

**Abstract**: State aggregation aims to reduce the computational complexity of solving Markov Decision Processes (MDPs) while preserving the performance of the original system. A fundamental challenge lies in optimizing policies within the aggregated, or abstract, space such that the performance remains optimal in the ground MDP-a property referred to as {"}optimal policy equivalence {"}.
This paper presents an abstraction framework based on the notion of homomorphism, in which two Markov chains are deemed homomorphic if their value functions exhibit a linear relationship. Within this theoretical framework, we establish a sufficient condition for the equivalence of optimal policy.
We further examine scenarios where the sufficient condition is not met and derive an upper bound on the approximation error and a performance lower bound for the objective function under the ground MDP. We propose Homomorphic Policy Gradient (HPG), which guarantees optimal policy equivalence under sufficient conditions, and its extension, Error-Bounded HPG (EBHPG), which balances computational efficiency and the performance loss induced by aggregation. In the experiments, we validated the theoretical results and conducted comparative evaluations against seven algorithms. 

**Abstract (ZH)**: 状态聚类旨在通过在聚合或抽象空间中优化策略来降低求解马尔可夫决策过程（MDPs）的计算复杂性，同时保持原系统的表现。一个基本挑战是在聚合空间中优化策略，使得在地面MDP中的性能保持最优——这一性质称为“最优策略等效性”。

本文基于同态的概念提出了一种抽象框架，如果两个马尔可夫链的价值函数之间存在线性关系，则认为这两个马尔可夫链是同态的。在此理论框架下，我们建立了最优策略等效性的充分条件。进一步研究了充分条件不满足的情况，并推导出在地面MDP下的近似误差上界和目标函数的性能下界。我们提出了同态策略梯度（HPG），在满足某些条件下保证最优策略等效性，并提出了错误边界同态策略梯度（EBHPG）以平衡聚类带来的计算效率损失和性能损失。在实验中，我们验证了理论结果，并与七种算法进行了对比评估。 

---
# Denoising Diffusion as a New Framework for Underwater Images 

**Title (ZH)**: 去噪扩散作为一种新的水下图像处理框架 

**Authors**: Nilesh Jain, Elie Alhajjar  

**Link**: [PDF](https://arxiv.org/pdf/2510.09934)  

**Abstract**: Underwater images play a crucial role in ocean research and marine environmental monitoring since they provide quality information about the ecosystem. However, the complex and remote nature of the environment results in poor image quality with issues such as low visibility, blurry textures, color distortion, and noise. In recent years, research in image enhancement has proven to be effective but also presents its own limitations, like poor generalization and heavy reliance on clean datasets. One of the challenges herein is the lack of diversity and the low quality of images included in these datasets. Also, most existing datasets consist only of monocular images, a fact that limits the representation of different lighting conditions and angles. In this paper, we propose a new plan of action to overcome these limitations. On one hand, we call for expanding the datasets using a denoising diffusion model to include a variety of image types such as stereo, wide-angled, macro, and close-up images. On the other hand, we recommend enhancing the images using Controlnet to evaluate and increase the quality of the corresponding datasets, and hence improve the study of the marine ecosystem.
Tags - Underwater Images, Denoising Diffusion, Marine ecosystem, Controlnet 

**Abstract (ZH)**: underwater图像在海洋研究和marine环境监测中发挥着关键作用，因为它们提供了关于生态系统质量信息。然而，环境的复杂性和遥远性导致图像质量较差，存在低清晰度、纹理模糊、颜色失真和噪声等问题。近年来，图像增强研究证明是有效的，但也存在自身的局限性，如泛化能力差和对干净数据集的高依赖性。其中一项挑战是在这些数据集中缺乏多样性和低质量图像。此外，大多数现有数据集仅包含单目图像，这限制了不同光照条件和角度的代表性。在本文中，我们提出了一种新的计划来克服这些局限性。一方面，我们呼吁使用去噪扩散模型扩展数据集，包括立体、广角、宏观和特写图像等各种图像类型。另一方面，我们建议使用Controlnet来评估并提高相应的数据集质量，从而改善对marine生态系统的研究。 

---
# MemPromptTSS: Persistent Prompt Memory for Iterative Multi-Granularity Time Series State Segmentation 

**Title (ZH)**: MemPromptTSS: 坚持式提示记忆在迭代多粒度时间序列状态分割中的应用 

**Authors**: Ching Chang, Ming-Chih Lo, Chiao-Tung Chan, Wen-Chih Peng, Tien-Fu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.09930)  

**Abstract**: Web platforms, mobile applications, and connected sensing systems generate multivariate time series with states at multiple levels of granularity, from coarse regimes to fine-grained events. Effective segmentation in these settings requires integrating across granularities while supporting iterative refinement through sparse prompt signals, which provide a compact mechanism for injecting domain knowledge. Yet existing prompting approaches for time series segmentation operate only within local contexts, so the effect of a prompt quickly fades and cannot guide predictions across the entire sequence. To overcome this limitation, we propose MemPromptTSS, a framework for iterative multi-granularity segmentation that introduces persistent prompt memory. A memory encoder transforms prompts and their surrounding subsequences into memory tokens stored in a bank. This persistent memory enables each new prediction to condition not only on local cues but also on all prompts accumulated across iterations, ensuring their influence persists across the entire sequence. Experiments on six datasets covering wearable sensing and industrial monitoring show that MemPromptTSS achieves 23% and 85% accuracy improvements over the best baseline in single- and multi-granularity segmentation under single iteration inference, and provides stronger refinement in iterative inference with average per-iteration gains of 2.66 percentage points compared to 1.19 for PromptTSS. These results highlight the importance of persistent memory for prompt-guided segmentation, establishing MemPromptTSS as a practical and effective framework for real-world applications. 

**Abstract (ZH)**: 基于持久提示记忆的迭代多粒度时间序列分段框架 

---
# Phase-Aware Deep Learning with Complex-Valued CNNs for Audio Signal Applications 

**Title (ZH)**: 基于复值卷积神经网络的相位感知深度学习在音频信号应用中 

**Authors**: Naman Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2510.09926)  

**Abstract**: This study explores the design and application of Complex-Valued Convolutional Neural Networks (CVCNNs) in audio signal processing, with a focus on preserving and utilizing phase information often neglected in real-valued networks. We begin by presenting the foundational theoretical concepts of CVCNNs, including complex convolutions, pooling layers, Wirtinger-based differentiation, and various complex-valued activation functions. These are complemented by critical adaptations of training techniques, including complex batch normalization and weight initialization schemes, to ensure stability in training dynamics. Empirical evaluations are conducted across three stages. First, CVCNNs are benchmarked on standard image datasets, where they demonstrate competitive performance with real-valued CNNs, even under synthetic complex perturbations. Although our focus is audio signal processing, we first evaluate CVCNNs on image datasets to establish baseline performance and validate training stability before applying them to audio tasks. In the second experiment, we focus on audio classification using Mel-Frequency Cepstral Coefficients (MFCCs). CVCNNs trained on real-valued MFCCs slightly outperform real CNNs, while preserving phase in input workflows highlights challenges in exploiting phase without architectural modifications. Finally, a third experiment introduces GNNs to model phase information via edge weighting, where the inclusion of phase yields measurable gains in both binary and multi-class genre classification. These results underscore the expressive capacity of complex-valued architectures and confirm phase as a meaningful and exploitable feature in audio processing applications. While current methods show promise, especially with activations like cardioid, future advances in phase-aware design will be essential to leverage the potential of complex representations in neural networks. 

**Abstract (ZH)**: 复杂值卷积神经网络在音频信号处理中的设计与应用：保留和利用常被忽略的相位信息 

---
# Augmenting generative models with biomedical knowledge graphs improves targeted drug discovery 

**Title (ZH)**: 将生物医学知识图谱融入生成模型以改善靶向药物发现 

**Authors**: Aditya Malusare, Vineet Punyamoorty, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.09914)  

**Abstract**: Recent breakthroughs in generative modeling have demonstrated remarkable capabilities in molecular generation, yet the integration of comprehensive biomedical knowledge into these models has remained an untapped frontier. In this study, we introduce K-DREAM (Knowledge-Driven Embedding-Augmented Model), a novel framework that leverages knowledge graphs to augment diffusion-based generative models for drug discovery. By embedding structured information from large-scale knowledge graphs, K-DREAM directs molecular generation toward candidates with higher biological relevance and therapeutic suitability. This integration ensures that the generated molecules are aligned with specific therapeutic targets, moving beyond traditional heuristic-driven approaches. In targeted drug design tasks, K-DREAM generates drug candidates with improved binding affinities and predicted efficacy, surpassing current state-of-the-art generative models. It also demonstrates flexibility by producing molecules designed for multiple targets, enabling applications to complex disease mechanisms. These results highlight the utility of knowledge-enhanced generative models in rational drug design and their relevance to practical therapeutic development. 

**Abstract (ZH)**: 知识驱动的嵌入增强模型K-DREAM在基于扩散的生成模型中整合大规模生物医药知识用于药物发现 

---
# Agentic Property-Based Testing: Finding Bugs Across the Python Ecosystem 

**Title (ZH)**: 基于代理属性的测试：跨越Python生态系统寻找 bug 

**Authors**: Muhammad Maaz, Liam DeVoe, Zac Hatfield-Dodds, Nicholas Carlini  

**Link**: [PDF](https://arxiv.org/pdf/2510.09907)  

**Abstract**: Property-based testing (PBT) is a lightweight formal method, typically implemented as a randomized testing framework. Users specify the input domain for their test using combinators supplied by the PBT framework, and the expected properties or invariants as a unit-test function. The framework then searches for a counterexample, e.g. by generating inputs and calling the test function. In this work, we demonstrate an LLM-based agent which analyzes Python modules, infers function-specific and cross-function properties from code and documentation, synthesizes and executes PBTs, reflects on outputs of these tests to confirm true bugs, and finally outputs actionable bug reports for the developer. We perform an extensive evaluation of our agent across 100 popular Python packages. Of the bug reports generated by the agent, we found after manual review that 56\% were valid bugs and 32\% were valid bugs that we would report to maintainers. We then developed a ranking rubric to surface high-priority valid bugs to developers, and found that of the 21 top-scoring bugs, 86\% were valid and 81\% we would report. The bugs span diverse failure modes from serialization failures to numerical precision errors to flawed cache implementations. We reported 5 bugs, 4 with patches, including to NumPy and cloud computing SDKs, with 3 patches merged successfully. Our results suggest that LLMs with PBT provides a rigorous and scalable method for autonomously testing software. Our code and artifacts are available at: this https URL. 

**Abstract (ZH)**: 基于属性的测试（PBT）的LLM代理：从Python模块中自动推导和执行属性测试以检测有效 Bug 

---
# Stability of Transformers under Layer Normalization 

**Title (ZH)**: Transformer层归一化下的稳定性 

**Authors**: Kelvin Kan, Xingjian Li, Benjamin J. Zhang, Tuhin Sahai, Stanley Osher, Krishna Kumar, Markos A. Katsoulakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.09904)  

**Abstract**: Despite their widespread use, training deep Transformers can be unstable. Layer normalization, a standard component, improves training stability, but its placement has often been ad-hoc. In this paper, we conduct a principled study on the forward (hidden states) and backward (gradient) stability of Transformers under different layer normalization placements. Our theory provides key insights into the training dynamics: whether training drives Transformers toward regular solutions or pathological behaviors. For forward stability, we derive explicit bounds on the growth of hidden states in trained Transformers. For backward stability, we analyze how layer normalization affects the backpropagation of gradients, thereby explaining the training dynamics of each layer normalization placement. Our analysis also guides the scaling of residual steps in Transformer blocks, where appropriate choices can further improve stability and performance. Our numerical results corroborate our theoretical findings. Beyond these results, our framework provides a principled way to sanity-check the stability of Transformers under new architectural modifications, offering guidance for future designs. 

**Abstract (ZH)**: 尽管深度Transformer在广泛应用，但其训练可能会不稳定。层规范化作为标准组件可以提高训练稳定性，但其放置方式往往缺乏系统性。本文系统研究了在不同层规范化放置情况下，Transformer的前向（隐藏状态）和后向（梯度）稳定性。我们的理论提供了关键见解，探讨了训练动力学：训练是否会引导Transformer趋向于良态解或病理行为。对于前向稳定性，我们推导了训练中隐藏状态增长的显式边界。对于后向稳定性，我们分析了层规范化如何影响梯度的反向传播，从而解释了每种层规范化放置下的训练动力学。我们的分析还指导了Transformer块中残差步骤的缩放，恰当的选择可以进一步提高稳定性和性能。我们的数值结果印证了理论发现。此外，我们的框架提供了一种系统的方法来验证新架构修改下Transformer的稳定性，为未来的设计提供指导。 

---
# Learning Bug Context for PyTorch-to-JAX Translation with LLMs 

**Title (ZH)**: 使用大规模语言模型学习PyTorch到JAX的转换上下文 Bug 

**Authors**: Hung Phan, Son Le Vu, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2510.09898)  

**Abstract**: Despite recent progress of large language models (LLMs) on code translation among mainstream languages, translating PyTorch to JAX remains nontrivial. The two libraries, though both embedded in Python, differ in core design, execution semantics, and ecosystem maturity; JAX is newer and comparatively underrepresented in public code, and parallel PyTorch--JAX corpora are limited. Weaknesses in existing evaluation further complicate cross-framework benchmarking. We present T2J, a prompt-augmentation framework that strengthens LLM-based PyTorch to JAX translation. Our pipeline (i) assembles two PyTorch sources -- the problem-solving set from TorchLeet (Aroori & Chien, 2025) and a GitHub-derived set from CodeParrot (Wolf et al., 2022) -- and uses GPT-4o-mini to produce initial JAX drafts; (ii) engages two professional developers to iteratively repair those drafts until functional equivalence, yielding a curated fixed-bug dataset of common errors and patches; and (iii) constructs augmented prompts that inject structured guidance from these fixes to steer lightweight LLMs (e.g., GPT-4o-mini). We also introduce three metrics tailored to PyTorch to JAX: T2J CodeTrans Score, T2J FixCost Score (an LLM-based estimate of bug-fix effort), and T2J Comparison Score (LLM-as-judge). Empirically, T2J raises GPT-4o-mini performance by up to 10% on CodeBLEU, 50% on T2J FixCost Score, 1.33 points on T2J CodeTrans Score (0--4 scale), and 100% on T2J Comparison Score; moreover, the generated code runs up to 2.5x faster than the baseline. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在主流编程语言之间的代码翻译方面取得了进展，但将PyTorch翻译为JAX仍具有挑战性。尽管这两个库都嵌入在Python中，但它们在核心设计、执行语义和生态系统成熟度方面存在差异；JAX较新且在公共代码中相对较少见，且PyTorch与JAX的平行语料库有限。现有评估的不足进一步复杂化了跨框架基准测试。我们提出T2J，一种增强框架，旨在加强基于LLM的PyTorch到JAX的翻译。我们的流水线包括：（i）组装两个PyTorch源代码——来自TorchLeet的问题解决集（Aroori & Chien, 2025）和来自CodeParrot的GitHub衍生集（Wolf et al., 2022），并使用GPT-4o-mini生成初始JAX草稿；（ii）让两名专业开发人员迭代修复这些草稿，直至功能等效，从而生成一个修复常见错误和补丁的精选数据集；（iii）构建增强提示，这些提示注入这些修复的结构化指导，以引导轻量级LLM（例如，GPT-4o-mini）。我们还介绍了针对PyTorch到JAX的三种度量标准：T2J CodeTrans得分、T2J FixCost得分（基于LLM的错误修复努力估计）和T2J Comparison得分（LLM作为评判者）。实验结果显示，T2J在CodeBLEU上提高了GPT-4o-mini的性能多达10%，在T2J FixCost得分上提高了50%，在T2J CodeTrans得分上提高了1.33分（0-4分制），并在T2J Comparison得分上提高了100%；此外，生成的代码比基线快2.5倍。 

---
# Chain-of-Influence: Tracing Interdependencies Across Time and Features in Clinical Predictive Modelings 

**Title (ZH)**: 连锁影响：跨时间和特征追溯临床预测建模中的相互依赖关系 

**Authors**: Yubo Li, Rema Padman  

**Link**: [PDF](https://arxiv.org/pdf/2510.09895)  

**Abstract**: Modeling clinical time-series data is hampered by the challenge of capturing latent, time-varying dependencies among features. State-of-the-art approaches often rely on black-box mechanisms or simple aggregation, failing to explicitly model how the influence of one clinical variable propagates through others over time. We propose $\textbf{Chain-of-Influence (CoI)}$, an interpretable deep learning framework that constructs an explicit, time-unfolded graph of feature interactions. CoI leverages a multi-level attention architecture: first, a temporal attention layer identifies critical time points in a patient's record; second, a cross-feature attention layer models the directed influence from features at these time points to subsequent features. This design enables the tracing of influence pathways, providing a granular audit trail that shows how any feature at any time contributes to the final prediction, both directly and through its influence on other variables. We evaluate CoI on mortality and disease progression tasks using the MIMIC-IV dataset and a private chronic kidney disease cohort. Our framework significantly outperforms existing methods in predictive accuracy. More importantly, through case studies, we show that CoI can uncover clinically meaningful, patient-specific patterns of disease progression that are opaque to other models, offering unprecedented transparency into the temporal and cross-feature dependencies that inform clinical decision-making. 

**Abstract (ZH)**: Chain-of-Influence (CoI): An Interpretable Deep Learning Framework for Modeling Clinical Time-Series Data 

---
# Probabilistic bias adjustment of seasonal predictions of Arctic Sea Ice Concentration 

**Title (ZH)**: 基于概率偏差调整的 Arctic 海冰浓度季节预测校正 

**Authors**: Parsa Gooya, Reinel Sospedra-Alfonso  

**Link**: [PDF](https://arxiv.org/pdf/2510.09891)  

**Abstract**: Seasonal forecast of Arctic sea ice concentration is key to mitigate the negative impact and assess potential opportunities posed by the rapid decline of sea ice coverage. Seasonal prediction systems based on climate models often show systematic biases and complex spatio-temporal errors that grow with the forecasts. Consequently, operational predictions are routinely bias corrected and calibrated using retrospective forecasts. For predictions of Arctic sea ice concentration, error corrections are mainly based on one-to-one post-processing methods including climatological mean or linear regression correction and, more recently, machine learning. Such deterministic adjustments are confined at best to the limited number of costly-to-run ensemble members of the raw forecast. However, decision-making requires proper quantification of uncertainty and likelihood of events, particularly of extremes. We introduce a probabilistic error correction framework based on a conditional Variational Autoencoder model to map the conditional distribution of observations given the biased model prediction. This method naturally allows for generating large ensembles of adjusted forecasts. We evaluate our model using deterministic and probabilistic metrics and show that the adjusted forecasts are better calibrated, closer to the observational distribution, and have smaller errors than climatological mean adjusted forecasts. 

**Abstract (ZH)**: 北极海冰浓度的季节预测对于减轻海冰覆盖快速下降带来的负面影响并评估潜在机会至关重要。基于气候模型的季节预测系统经常显示出系统偏差和复杂的时空误差，这些误差随预测时间的延长而增大。因此，运营预测通常会使用回顾性预测进行偏差校正和校准。对于北极海冰浓度的预测，误差校正主要基于一对一后处理方法，包括气候平均值校正或线性回归校正，近年来还包括机器学习方法。这些确定性调整仅局限于昂贵的原始预报 ensemble 成员数量。然而，决策需要对不确定性及其事件的可能性进行适当的量化，尤其是极端事件。我们提出了一种基于条件变分自编码器模型的概率误差校正框架，以映射给定偏差模型预测的观测条件分布。该方法自然地能够生成大量的校正预测 ensemble。我们使用确定性和概率性指标评估了我们的模型，并证明了校正预测在校准性、与观测分布的接近度以及误差方面优于气候平均值校正预测。 

---
# CHUG: Crowdsourced User-Generated HDR Video Quality Dataset 

**Title (ZH)**: CHUG: 众包用户生成的高动态范围视频质量数据集 

**Authors**: Shreshth Saini, Alan C. Bovik, Neil Birkbeck, Yilin Wang, Balu Adsumilli  

**Link**: [PDF](https://arxiv.org/pdf/2510.09879)  

**Abstract**: High Dynamic Range (HDR) videos enhance visual experiences with superior brightness, contrast, and color depth. The surge of User-Generated Content (UGC) on platforms like YouTube and TikTok introduces unique challenges for HDR video quality assessment (VQA) due to diverse capture conditions, editing artifacts, and compression distortions. Existing HDR-VQA datasets primarily focus on professionally generated content (PGC), leaving a gap in understanding real-world UGC-HDR degradations. To address this, we introduce CHUG: Crowdsourced User-Generated HDR Video Quality Dataset, the first large-scale subjective study on UGC-HDR quality. CHUG comprises 856 UGC-HDR source videos, transcoded across multiple resolutions and bitrates to simulate real-world scenarios, totaling 5,992 videos. A large-scale study via Amazon Mechanical Turk collected 211,848 perceptual ratings. CHUG provides a benchmark for analyzing UGC-specific distortions in HDR videos. We anticipate CHUG will advance No-Reference (NR) HDR-VQA research by offering a large-scale, diverse, and real-world UGC dataset. The dataset is publicly available at: this https URL. 

**Abstract (ZH)**: 高动态范围（HDR）视频通过卓越的亮度、对比度和色深提升视觉体验。平台如YouTube和TikTok上的用户生成内容（UGC）激增，为HDR视频质量评估（VQA）带来了新的挑战，特别是由于不同的拍摄条件、编辑 artefacts 以及压缩失真。现有的HDR-VQA数据集主要关注专业生成内容（PGC），忽略了真实世界中UGC-HDR降质的理解。为解决这一问题，我们引入了CHUG：众包用户生成HDR视频质量数据集，它是首个大规模主观评估UGC-HDR质量的研究。CHUG包含856个UGC-HDR源视频，跨多个分辨率和比特率重新编码以模拟真实场景，总共5,992个视频。通过亚马逊 Mechanical Turk 的大规模研究收集了211,848个感知评分。CHUG为分析HDR视频中的UGC特定失真提供了基准数据。我们期望CHUG将通过提供一个大规模、多样且具有现实属性的UGC数据集，推动无参考（NR）HDR-VQA研究的发展。数据集已在以下网址公开：this https URL。 

---
# Myopic Bayesian Decision Theory for Batch Active Learning with Partial Batch Label Sampling 

**Title (ZH)**: 局部贝叶斯决策理论在部分批次标签采样的批量主动学习中 

**Authors**: Kangping Hu, Stephen Mussmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.09877)  

**Abstract**: Over the past couple of decades, many active learning acquisition functions have been proposed, leaving practitioners with an unclear choice of which to use. Bayesian Decision Theory (BDT) offers a universal principle to guide decision-making. In this work, we derive BDT for (Bayesian) active learning in the myopic framework, where we imagine we only have one more point to label. This derivation leads to effective algorithms such as Expected Error Reduction (EER), Expected Predictive Information Gain (EPIG), and other algorithms that appear in the literature. Furthermore, we show that BAIT (active learning based on V-optimal experimental design) can be derived from BDT and asymptotic approximations. A key challenge of such methods is the difficult scaling to large batch sizes, leading to either computational challenges (BatchBALD) or dramatic performance drops (top-$B$ selection). Here, using a particular formulation of the decision process, we derive Partial Batch Label Sampling (ParBaLS) for the EPIG algorithm. We show experimentally for several datasets that ParBaLS EPIG gives superior performance for a fixed budget and Bayesian Logistic Regression on Neural Embeddings. Our code is available at this https URL. 

**Abstract (ZH)**: 近二十年来，许多主动学习获取函数被提出，给实践者的选择造成了困惑。贝叶斯决策理论（BDT）提供了一种通用原则来指导决策。在本文中，我们在近视框架下推导了BDT在（贝叶斯）主动学习中的应用，即我们设想只剩下一点需要标记。这一推导导出了有效的算法，如期望误差减少（EER）、期望预测信息增益（EPIG）和其他文献中出现的算法。此外，我们展示了基于V-最优实验设计的主动学习方法BAIT可以从BDT和渐近逼近中推导出来。这类方法的关键挑战是难以扩展到大批次大小，导致要么计算挑战（BatchBALD）要么性能显著下降（top-$B$ 选择）。在这里，通过特定的决策过程的公式化，我们为EPIG算法推导了部分批次标签采样（ParBaLS）。实验结果表明，对于固定预算和神经嵌入的贝叶斯逻辑回归，ParBaLS EPIG表现出更优的性能。我们的代码可在以下链接获取：this https URL。 

---
# WARC-Bench: Web Archive Based Benchmark for GUI Subtask Executions 

**Title (ZH)**: WARC-Bench：基于网络档案的GUI子任务执行基准测试 

**Authors**: Sanjari Srivastava, Gang Li, Cheng Chang, Rishu Garg, Manpreet Kaur, Charlene Y. Lee, Yuezhang Li, Yining Mao, Ignacio Cases, Yanan Xie, Peng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.09872)  

**Abstract**: Training web agents to navigate complex, real-world websites requires them to master $\textit{subtasks}$ - short-horizon interactions on multiple UI components (e.g., choosing the correct date in a date picker, or scrolling in a container to extract information). We introduce WARC-Bench (Web Archive Benchmark), a novel web navigation benchmark featuring 438 tasks designed to evaluate multimodal AI agents on subtasks. WARC-Bench enables sandboxed interactions with dynamic and realistic webpages using Web ARChive files. We show that WARC-Bench is challenging for leading computer-use models, with the highest observed success rate being 64.8%. To improve open source models on subtask, we explore two common training techniques: supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR). Experiments show that SFT models obtain a 48.8% success rate on the benchmark. Training with RLVR over SFT checkpoints, even in data-scarce settings, improves the score to 52.8% on WARC-Bench, outperforming many frontier models. Our analysis concludes that mastering these subtasks is essential for robust web planning and navigation, and is a capability not extensively evaluated by existing benchmarks. 

**Abstract (ZH)**: 训练网络代理导航复杂的真实世界网站需要它们掌握子任务——在多个UI组件上的短期交互（例如，在日期选择器中选择正确的日期，或在容器中滚动以提取信息）。我们引入了WARC-Bench（Web档案基准），这是一个新颖的网络导航基准，包含438个任务，旨在评估多模式AI代理在子任务上的表现。WARC-Bench 使用Web ARChive文件实现了对动态和真实网页的沙箱交互。实验显示，领先的人工智能模型在基准测试中的最高成功率仅为64.8%。为了提高开源模型在子任务上的表现，我们探索了两种常见的训练技术：监督微调（SFT）和带有可验证奖励的强化学习（RLVR）。实验结果表明，使用SFT的模型在基准测试中的成功率达到了48.8%。即使在数据稀缺的情况下，使用RLVR对SFT检查点进行训练，也可以将WARC-Bench上的得分为52.8%，超过了诸多前沿模型。我们的分析得出结论，掌握这些子任务对于稳健的网页规划和导航至关重要，而现有的基准测试很少评估这一能力。 

---
# NarraBench: A Comprehensive Framework for Narrative Benchmarking 

**Title (ZH)**: NarrBench：综合叙事基准框架 

**Authors**: Sil Hamilton, Matthew Wilkens, Andrew Piper  

**Link**: [PDF](https://arxiv.org/pdf/2510.09869)  

**Abstract**: We present NarraBench, a theory-informed taxonomy of narrative-understanding tasks, as well as an associated survey of 78 existing benchmarks in the area. We find significant need for new evaluations covering aspects of narrative understanding that are either overlooked in current work or are poorly aligned with existing metrics. Specifically, we estimate that only 27% of narrative tasks are well captured by existing benchmarks, and we note that some areas -- including narrative events, style, perspective, and revelation -- are nearly absent from current evaluations. We also note the need for increased development of benchmarks capable of assessing constitutively subjective and perspectival aspects of narrative, that is, aspects for which there is generally no single correct answer. Our taxonomy, survey, and methodology are of value to NLP researchers seeking to test LLM narrative understanding. 

**Abstract (ZH)**: 我们介绍了Narrabench，一个基于理论的叙事理解任务分类体系，以及对该领域78个现有基准的研究概况。我们发现，在当前研究中存在重要需求，即需要新的评估来覆盖当前工作中忽视或与现有度量标准不完全匹配的叙事理解方面。具体而言，我们估计只有27%的叙事任务能够被现有的基准充分捕捉，某些领域（包括叙事事件、风格、视角和披露）几乎未被当前的评估所涵盖。我们还指出，需要加强对体现主观性和视角性特质的叙事评估基准的研究开发，即那些通常没有唯一正确答案的方面。我们的分类体系、研究概况和方法对寻求测试大规模语言模型叙事理解能力的自然语言处理研究人员具有价值。 

---
# CALM: A Causal Analysis Language Model for Tabular Data in Complex Systems with Local Scores, Conditional Independence Tests, and Relation Attributes 

**Title (ZH)**: CALM：一种用于复杂系统中表格数据的因果分析语言模型，包含局部评分、条件独立性检验和关系属性。 

**Authors**: Zhenjiang Fan, Zengyi Qin, Yuanning Zheng, Bo Xiong, Summer Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.09846)  

**Abstract**: Causal discovery from observational data is fundamental to scientific fields like biology, where controlled experiments are often impractical. However, existing methods, including constraint-based (e.g., PC, causalMGM) and score-based approaches (e.g., NOTEARS), face significant limitations. These include an inability to resolve causal direction, restrictions to linear associations, sensitivity to violations of the faithfulness assumption, and inefficiency in searching vast hypothesis spaces. While large language models (LLMs) offer powerful reasoning capabilities, their application is hindered by a fundamental discrepancy: they are designed for text, while most causal data is tabular. To address these challenges, we introduce CALM, a novel causal analysis language model specifically designed for tabular data in complex systems. CALM leverages a Mamba-based architecture to classify causal patterns from pairwise variable relationships. It integrates a comprehensive suite of evidence, including local causal scores, conditional independence tests, and relational attributes, to capture a wide spectrum of linear, nonlinear, and conditional causal mechanisms. Trained on a diverse corpus of synthetic data (from linear, mixed, and nonlinear models) and 10 real-world biological datasets with rigorously validated causal relationships, our model ensures robustness and generalizability. Empirical evaluation demonstrates that CALM significantly outperforms existing methods in both simulation studies, achieving over 91% accuracy, and in a real-world application identifying causal factors in Hepatitis C virus progression. This work represents a significant step towards accurate and generalizable causal discovery by successfully adapting the pattern recognition capabilities of language models to the intricacies of tabular data. 

**Abstract (ZH)**: 基于观察数据的因果发现对于生物学等科学领域至关重要，但在许多情况下进行受控实验是不现实的。现有方法，包括基于约束的方法（例如PC, causalMGM）和基于评分的方法（例如NOTEARS），存在显著限制。这些限制包括无法确定因果方向、仅限于线性关系、对信仰假设违反的敏感性以及在大量假设空间搜索方面的低效性。虽然大型语言模型提供了强大的推理能力，但其应用受限于根本性差异：它们设计用于文本，而大多数因果数据为表格形式。为了解决这些挑战，我们引入了CALM，一种专门针对复杂系统中表格数据的新型因果分析语言模型。CALM利用Mamba架构来从成对变量关系中分类因果模式。它集成了局部因果评分、条件独立性检验和关系属性等多种证据，以捕捉广泛多样的线性、非线性和条件因果机制。该模型通过对来自线性、混合和非线性模型的合成数据以及10个真实世界生物数据集进行训练，这些真实世界数据集中的因果关系经过严格的验证，确保了模型的稳健性和泛化能力。实证评价表明，CALM在模拟研究中显著优于现有方法，准确率超过91%，在识别Hepatitis C病毒进展中的因果因素的实际应用中也同样表现出色。这项工作标志着准确和泛化因果发现的重要进步，成功将语言模型的模式识别能力适应了表格数据的复杂性。 

---
# Towards Understanding Ambiguity Resolution in Multimodal Inference of Meaning 

**Title (ZH)**: 理解多模态语义推理中的歧义Resolve 

**Authors**: Yufei Wang, Adriana Kovashka, Loretta Fernández, Marc N. Coutanche, Seth Wiener  

**Link**: [PDF](https://arxiv.org/pdf/2510.09815)  

**Abstract**: We investigate a new setting for foreign language learning, where learners infer the meaning of unfamiliar words in a multimodal context of a sentence describing a paired image. We conduct studies with human participants using different image-text pairs. We analyze the features of the data (i.e., images and texts) that make it easier for participants to infer the meaning of a masked or unfamiliar word, and what language backgrounds of the participants correlate with success. We find only some intuitive features have strong correlations with participant performance, prompting the need for further investigating of predictive features for success in these tasks. We also analyze the ability of AI systems to reason about participant performance, and discover promising future directions for improving this reasoning ability. 

**Abstract (ZH)**: 我们研究了一种新的外语学习环境，在这种环境中，学习者在描述配对图像的句子的多模态上下文中推断陌生词的意义。我们使用不同的图像-文本对进行人类参与者的实验研究。我们分析了使参与者更容易推断被遮罩或陌生词意义的数据特征（即图像和文本），以及参与者的语言背景与成功之间的关联。我们发现只有某些直观的特征与参与者的表现有很强的相关性，这提示我们还需要进一步研究这些任务中预测成功的特征。我们还分析了AI系统在理解参与者表现方面的推理能力，并发现了改进这种推理能力的有希望的方向。 

---
# Temporal Lifting as Latent-Space Regularization for Continuous-Time Flow Models in AI Systems 

**Title (ZH)**: 时域提升作为latent空间正则化在AI系统中连续时间流模型中的应用 

**Authors**: Jeffrey Camlin  

**Link**: [PDF](https://arxiv.org/pdf/2510.09805)  

**Abstract**: We present a latent-space formulation of adaptive temporal reparametrization for continuous-time dynamical systems. The method, called *temporal lifting*, introduces a smooth monotone mapping $t \mapsto \tau(t)$ that regularizes near-singular behavior of the underlying flow while preserving its conservation laws. In the lifted coordinate, trajectories such as those of the incompressible Navier-Stokes equations on the torus $\mathbb{T}^3$ become globally smooth. From the standpoint of machine-learning dynamics, temporal lifting acts as a continuous-time normalization or time-warping operator that can stabilize physics-informed neural networks and other latent-flow architectures used in AI systems. The framework links analytic regularity theory with representation-learning methods for stiff or turbulent processes. 

**Abstract (ZH)**: 我们提出了连续时间动力系统自适应时间重参数的隐空间形式化方法：时间提升 

---
# SVTime: Small Time Series Forecasting Models Informed by "Physics" of Large Vision Model Forecasters 

**Title (ZH)**: SVTime: 由大型视觉模型预报“物理”原理启发的_SMALL_时间序列 forecasting 模型 

**Authors**: ChengAo Shen, Ziming Zhao, Hanghang Tong, Dongjin Song, Dongsheng Luo, Qingsong Wen, Jingchao Ni  

**Link**: [PDF](https://arxiv.org/pdf/2510.09780)  

**Abstract**: Time series AI is crucial for analyzing dynamic web content, driving a surge of pre-trained large models known for their strong knowledge encoding and transfer capabilities across diverse tasks. However, given their energy-intensive training, inference, and hardware demands, using large models as a one-fits-all solution raises serious concerns about carbon footprint and sustainability. For a specific task, a compact yet specialized, high-performing model may be more practical and affordable, especially for resource-constrained users such as small businesses. This motivates the question: Can we build cost-effective lightweight models with large-model-like performance on core tasks such as forecasting? This paper addresses this question by introducing SVTime, a novel Small model inspired by large Vision model (LVM) forecasters for long-term Time series forecasting (LTSF). Recently, LVMs have been shown as powerful tools for LTSF. We identify a set of key inductive biases of LVM forecasters -- analogous to the "physics" governing their behaviors in LTSF -- and design small models that encode these biases through meticulously crafted linear layers and constraint functions. Across 21 baselines spanning lightweight, complex, and pre-trained large models on 8 benchmark datasets, SVTime outperforms state-of-the-art (SOTA) lightweight models and rivals large models with 10^3 fewer parameters than LVMs, while enabling efficient training and inference in low-resource settings. 

**Abstract (ZH)**: 基于大型模型的长周期时间序列预测轻量化模型SVTime 

---
# Why Do Transformers Fail to Forecast Time Series In-Context? 

**Title (ZH)**: 为什么变压器模型在上下文情境中无法预测时间序列？ 

**Authors**: Yufa Zhou, Yixiao Wang, Surbhi Goel, Anru R. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09776)  

**Abstract**: Time series forecasting (TSF) remains a challenging and largely unsolved problem in machine learning, despite significant recent efforts leveraging Large Language Models (LLMs), which predominantly rely on Transformer architectures. Empirical evidence consistently shows that even powerful Transformers often fail to outperform much simpler models, e.g., linear models, on TSF tasks; however, a rigorous theoretical understanding of this phenomenon remains limited. In this paper, we provide a theoretical analysis of Transformers' limitations for TSF through the lens of In-Context Learning (ICL) theory. Specifically, under AR($p$) data, we establish that: (1) Linear Self-Attention (LSA) models $\textit{cannot}$ achieve lower expected MSE than classical linear models for in-context forecasting; (2) as the context length approaches to infinity, LSA asymptotically recovers the optimal linear predictor; and (3) under Chain-of-Thought (CoT) style inference, predictions collapse to the mean exponentially. We empirically validate these findings through carefully designed experiments. Our theory not only sheds light on several previously underexplored phenomena but also offers practical insights for designing more effective forecasting architectures. We hope our work encourages the broader research community to revisit the fundamental theoretical limitations of TSF and to critically evaluate the direct application of increasingly sophisticated architectures without deeper scrutiny. 

**Abstract (ZH)**: 时间序列预测（TSF）仍然是机器学习中一个极具挑战性和尚未完全解决的问题，尽管近年来通过大型语言模型（LLMs）取得了显著进展，这些模型主要依赖于Transformer架构。实证证据一致表明，即使强大的Transformer模型在时间序列预测任务上也往往无法超越更简单的模型，例如线性模型；然而，对这一现象的严格理论理解仍然有限。在本文中，我们通过基于In-Context Learning（ICL）理论的角度对Transformer在时间序列预测中的局限性进行了理论分析。具体来说，在AR($p$)数据下，我们证明了：（1）线性自注意力（LSA）模型不能实现比经典线性模型更低的期望均方误差（MSE）进行上下文中的预测；（2）随着上下文长度趋向无穷大，LSA渐近恢复最优线性预测器；（3）在链式思维（CoT）风格推理中，预测以指数形式收敛到均值。我们通过精心设计的实验验证了这些发现。我们的理论不仅揭示了几种先前未被充分探索的现象，也为设计更有效的预测架构提供了实际见解。我们希望我们的工作能够鼓励更广泛的科研社区重新审视时间序列预测的基本理论限制，并在深入审查之前批判性地评估越来越复杂的架构的应用。 

---
# PromptGuard at BLP-2025 Task 1: A Few-Shot Classification Framework Using Majority Voting and Keyword Similarity for Bengali Hate Speech Detection 

**Title (ZH)**: PromptGuard在BLP-2025任务1中的Few-Shot分类框架：基于多数投票和关键词相似性的孟加拉语仇恨言论检测 

**Authors**: Rakib Hossan, Shubhashis Roy Dipta  

**Link**: [PDF](https://arxiv.org/pdf/2510.09771)  

**Abstract**: The BLP-2025 Task 1A requires Bengali hate speech classification into six categories. Traditional supervised approaches need extensive labeled datasets that are expensive for low-resource languages. We developed PromptGuard, a few-shot framework combining chi-square statistical analysis for keyword extraction with adaptive majority voting for decision-making. We explore statistical keyword selection versus random approaches and adaptive voting mechanisms that extend classification based on consensus quality. Chi-square keywords provide consistent improvements across categories, while adaptive voting benefits ambiguous cases requiring extended classification rounds. PromptGuard achieves a micro-F1 of 67.61, outperforming n-gram baselines (60.75) and random approaches (14.65). Ablation studies confirm chi-square-based keywords show the most consistent impact across all categories. 

**Abstract (ZH)**: BLP-2025 任务1A 要求将孟加拉语仇恨言论分类为六类。传统的监督方法需要大量的标记数据集，这对于低资源语言来说成本高昂。我们开发了 PromptGuard，这是一种结合卡方统计分析进行关键词提取和自适应多数投票进行决策的少样本框架。我们探索了基于统计的关键词选择与随机方法以及基于共识质量的自适应投票机制，以扩展分类。卡方关键词在各类别中提供了持续的改进，而自适应投票机制有助于解决需要更长时间分类的歧义案例。PromptGuard 实现了微F1值为67.61，优于n-克隆基线（60.75）和随机方法（14.65）。消融研究确认基于卡方的关键词在所有类别中表现出最一致的影响。 

---
# Scaling Laws and Symmetry, Evidence from Neural Force Fields 

**Title (ZH)**: 神经力场的标度律与对称性证据 

**Authors**: Khang Ngo, Siamak Ravanbakhsh  

**Link**: [PDF](https://arxiv.org/pdf/2510.09768)  

**Abstract**: We present an empirical study in the geometric task of learning interatomic potentials, which shows equivariance matters even more at larger scales; we show a clear power-law scaling behaviour with respect to data, parameters and compute with ``architecture-dependent exponents''. In particular, we observe that equivariant architectures, which leverage task symmetry, scale better than non-equivariant models. Moreover, among equivariant architectures, higher-order representations translate to better scaling exponents. Our analysis also suggests that for compute-optimal training, the data and model sizes should scale in tandem regardless of the architecture. At a high level, these results suggest that, contrary to common belief, we should not leave it to the model to discover fundamental inductive biases such as symmetry, especially as we scale, because they change the inherent difficulty of the task and its scaling laws. 

**Abstract (ZH)**: 我们呈现了一项关于原子间势学习的几何任务中的实证研究，表明在更大尺度上，对称性的重要性更加突出；我们展示了数据、参数和计算在“架构依赖指数”下的幂律 scaling 行为。特别是，我们观察到利用任务对称性的对称架构相比非对称模型具有更好的 scaling 行为。此外，在对称架构中，更高阶的表示对应更好的 scaling 指数。我们的分析还表明，为了实现计算最优的训练，数据和模型的规模应该同步增长，而不受架构的影响。总体而言，这些结果表明，与常见的观点相反，当我们扩展规模时，不应让模型自己发现诸如对称性这样的基本归纳偏置，因为这会改变任务的内在难度及其 scaling 法则。 

---
# Patentformer: A demonstration of AI-assisted automated patent drafting 

**Title (ZH)**: Patentformer：人工智能辅助自动专利起草演示 

**Authors**: Sai Krishna Reddy Mudhiganti, Juanyan Wang, Ruo Yang, Manali Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.09752)  

**Abstract**: Patent drafting presents significant challenges due to its reliance on the extensive experience and specialized expertise of patent attorneys, who must possess both legal acumen and technical understanding of an invention to craft patent applications in a formal legal writing style. This paper presents a demonstration of Patentformer, an AI-powered automated patent drafting platform designed to support patent attorneys by rapidly producing high-quality patent applications adhering to legal writing standards. 

**Abstract (ZH)**: 由于专利撰写依赖于专利律师的丰富经验和专业技能，要求律师同时具备法律洞察力和发明技术理解能力，以正式的法律文书风格撰写专利申请。本文展示了Patentformer这一基于AI的自动化专利撰写平台的示例，旨在通过快速生成符合法律写作标准的高质量专利申请来支持专利律师。 

---
# Machine learning methods fail to provide cohesive atheoretical construction of personality traits from semantic embeddings 

**Title (ZH)**: 机器学习方法未能提供人格特质的cohesive理论构建从语义嵌入 Perspective 

**Authors**: Ayoub Bouguettaya, Elizabeth M. Stuart  

**Link**: [PDF](https://arxiv.org/pdf/2510.09739)  

**Abstract**: The lexical hypothesis posits that personality traits are encoded in language and is foundational to models like the Big Five. We created a bottom-up personality model from a classic adjective list using machine learning and compared its descriptive utility against the Big Five by analyzing one million Reddit comments. The Big Five, particularly Agreeableness, Conscientiousness, and Neuroticism, provided a far more powerful and interpretable description of these online communities. In contrast, our machine-learning clusters provided no meaningful distinctions, failed to recover the Extraversion trait, and lacked the psychometric coherence of the Big Five. These results affirm the robustness of the Big Five and suggest personality's semantic structure is context-dependent. Our findings show that while machine learning can help check the ecological validity of established psychological theories, it may not be able to replace them. 

**Abstract (ZH)**: 词素假设提出了人格特质在语言中编码的理论，并构成了五大人格特质模型的基础。我们利用机器学习从经典形容词列表中自下而上构建了一个人格模型，并通过分析一百万条Reddit评论将其描述效用与五大人格特质进行了比较。五大人格特质，尤其是随和性、尽责性和情绪稳定性，提供了对这些在线社区更为强大且易于解释的描述。相比之下，我们的机器学习聚类未能提供有意义的区别，无法恢复外向性特质，并缺乏五大人格特质的心理测量一致性。这些结果证实了五大人格特质的稳健性，并暗示了人格的语义结构具有情境依赖性。我们的研究显示，虽然机器学习可以帮助验证已建立的心理学理论的生态效度，但它可能无法替代这些理论。 

---
# Chlorophyll-a Mapping and Prediction in the Mar Menor Lagoon Using C2RCC-Processed Sentinel 2 Imagery 

**Title (ZH)**: 使用C2RCC处理的Sentinel 2影像进行马尔梅诺潟湖叶绿素-a的映射与预测 

**Authors**: Antonio Martínez-Ibarra, Aurora González-Vidal, Adrián Cánovas-Rodríguez, Antonio F. Skarmeta  

**Link**: [PDF](https://arxiv.org/pdf/2510.09736)  

**Abstract**: The Mar Menor, Europe's largest coastal lagoon, located in Spain, has undergone severe eutrophication crises. Monitoring chlorophyll-a (Chl-a) is essential to anticipate harmful algal blooms and guide mitigation. Traditional in situ measurements are spatially and temporally limited. Satellite-based approaches provide a more comprehensive view, enabling scalable, long-term, and transferable monitoring. This study aims to overcome limitations of chlorophyll monitoring, often restricted to surface estimates or limited temporal coverage, by developing a reliable methodology to predict and map Chl-a across the water column of the Mar Menor. The work integrates Sentinel 2 imagery with buoy-based ground truth to create models capable of high-resolution, depth-specific monitoring, enhancing early-warning capabilities for eutrophication. Nearly a decade of Sentinel 2 images was atmospherically corrected using C2RCC processors. Buoy data were aggregated by depth (0-1 m, 1-2 m, 2-3 m, 3-4 m). Multiple ML and DL algorithms-including RF, XGBoost, CatBoost, Multilater Perceptron Networks, and ensembles-were trained and validated using cross-validation. Systematic band-combination experiments and spatial aggregation strategies were tested to optimize prediction. Results show depth-dependent performance. At the surface, C2X-Complex with XGBoost and ensemble models achieved R2 = 0.89; at 1-2 m, CatBoost and ensemble models reached R2 = 0.87; at 2-3 m, TOA reflectances with KNN performed best (R2 = 0.81); while at 3-4 m, RF achieved R2 = 0.66. Generated maps successfully reproduced known eutrophication events (e.g., 2016 crisis, 2025 surge), confirming robustness. The study delivers an end-to-end, validated methodology for depth-specific Chl-amapping. Its integration of multispectral band combinations, buoy calibration, and ML/DL modeling offers a transferable framework for other turbid coastal systems. 

**Abstract (ZH)**: 欧洲最大的沿海泻湖——西班牙的马尔梅诺泻湖经历了严重的富营养化危机。通过监测chlorophyll-a (Chl-a)可以预见有害藻华并指导缓解措施。传统的原位测量在空间和时间上都有限制。基于卫星的方法提供了更全面的观点，使监测变得规模化、长时间且可转移。本研究旨在通过开发可靠的预测和地图绘制方法来克服Chl-a监测的限制，该方法能够针对马尔梅诺泻湖水柱的高分辨率、深度特定监测，增强对富营养化的早期预警能力。利用Sentinel-2图像与浮标地面实况数据的整合，创建了可用于高分辨率、深度特定监测的模型，增强了对富营养化的早期预警能力。近十年的Sentinel-2图像通过C2RCC处理器进行了大气校正。浮标数据按深度（0-1米、1-2米、2-3米、3-4米）进行了聚合。多种机器学习和深度学习算法，包括随机森林、XGBoost、CatBoost、多层感知网络以及它们的集成，通过交叉验证进行了训练和验证。系统地进行了波段组合实验和空间聚合策略以优化预测。结果表明，预测性能具有深度依赖性。在表层，C2X-Complex与XGBoost和集成模型的R2值达到0.89；在1-2米深度，CatBoost和集成模型的R2值达到0.87；在2-3米深度，表面辐亮度与KNN方法表现最佳（R2值为0.81）；而在3-4米深度，随机森林的R2值为0.66。生成的地图成功重现了已知的富营养化事件（如2016年的危机和2025年的激增），证实了其稳健性。本研究提供了一种端到端的、经验证的深度特定Chl-a地图绘制方法。其结合多光谱波段组合、浮标校准和机器学习/深度学习建模，提供了一个适用于其他浑浊海岸系统的可转移框架。 

---
# ARROW: An Adaptive Rollout and Routing Method for Global Weather Forecasting 

**Title (ZH)**: ARROW：一种适应性滚动和routing方法用于全球天气预报 

**Authors**: Jindong Tian, Yifei Ding, Ronghui Xu, Hao Miao, Chenjuan Guo, Bin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09734)  

**Abstract**: Weather forecasting is a fundamental task in spatiotemporal data analysis, with broad applications across a wide range of domains. Existing data-driven forecasting methods typically model atmospheric dynamics over a fixed short time interval (e.g., 6 hours) and rely on naive autoregression-based rollout for long-term forecasting (e.g., 138 hours). However, this paradigm suffers from two key limitations: (1) it often inadequately models the spatial and multi-scale temporal dependencies inherent in global weather systems, and (2) the rollout strategy struggles to balance error accumulation with the capture of fine-grained atmospheric variations. In this study, we propose ARROW, an Adaptive-Rollout Multi-scale temporal Routing method for Global Weather Forecasting. To contend with the first limitation, we construct a multi-interval forecasting model that forecasts weather across different time intervals. Within the model, the Shared-Private Mixture-of-Experts captures both shared patterns and specific characteristics of atmospheric dynamics across different time scales, while Ring Positional Encoding accurately encodes the circular latitude structure of the Earth when representing spatial information. For the second limitation, we develop an adaptive rollout scheduler based on reinforcement learning, which selects the most suitable time interval to forecast according to the current weather state. Experimental results demonstrate that ARROW achieves state-of-the-art performance in global weather forecasting, establishing a promising paradigm in this field. 

**Abstract (ZH)**: 面向全球天气预报的自适应展开多尺度时空路由方法_ARROW 

---
# Herb.jl: A Unifying Program Synthesis Library 

**Title (ZH)**: Herb.jl: 一个统一的程序合成库 

**Authors**: Tilman Hinnerichs, Reuben Gardos Reid, Jaap de Jong, Bart Swinkels, Pamela Wochner, Nicolae Filat, Tudor Magurescu, Issa Hanou, Sebastijan Dumancic  

**Link**: [PDF](https://arxiv.org/pdf/2510.09726)  

**Abstract**: Program synthesis -- the automatic generation of code given a specification -- is one of the most fundamental tasks in artificial intelligence (AI) and many programmers' dream. Numerous synthesizers have been developed to tackle program synthesis, manifesting different ideas to approach the exponentially growing program space. While numerous smart program synthesis tools exist, reusing and remixing previously developed methods is tedious and time-consuming. We propose this http URL, a unifying program synthesis library written in the Julia programming language, to address these issues. Since current methods rely on similar building blocks, we aim to modularize the underlying synthesis algorithm into communicating and fully extendable sub-compartments, allowing for straightforward reapplication of these modules. To demonstrate the benefits of using this http URL, we show three common use cases: 1. how to implement a simple problem and grammar, and how to solve it, 2. how to implement a previously developed synthesizer with just a few lines of code, and 3. how to run a synthesizer against a benchmark. 

**Abstract (ZH)**: 程序合成——根据规范自动生成代码——是人工智能（AI）中最基本的任务之一，也是许多程序员的梦想。我们提出了一个用Julia编程语言编写的统一程序合成库URL（此处URL应替换为实际网址），以解决这些问题。由于当前的方法依赖于相似的基本构建块，我们的目标是将底层合成算法模块化为可通信和完全可扩展的子模块，从而便于重新应用这些模块。为了展示使用这个库的好处，我们展示了三种常见用例：1. 如何实现一个简单的程序和语法，并解决相关问题，2. 如何仅用几行代码实现一个先前开发的合成器，3. 如何在基准上运行一个合成器。 

---
# It's 2025 -- Narrative Learning is the new baseline to beat for explainable machine learning 

**Title (ZH)**: 2025年——叙述学习成为可解释机器学习的新基准 

**Authors**: Gregory D. Baker  

**Link**: [PDF](https://arxiv.org/pdf/2510.09723)  

**Abstract**: In this paper, we introduce Narrative Learning, a methodology where models are defined entirely in natural language and iteratively refine their classification criteria using explanatory prompts rather than traditional numerical optimisation. We report on experiments to evaluate the accuracy and potential of this approach using 3 synthetic and 3 natural datasets and compare them against 7 baseline explainable machine learning models. We demonstrate that on 5 out of 6 of these datasets, Narrative Learning became more accurate than the baseline explainable models in 2025 or earlier because of improvements in language models. We also report on trends in the lexicostatistics of these models' outputs as a proxy for the comprehensibility of the explanations. 

**Abstract (ZH)**: Narrative Learning：一种完全用自然语言定义模型并使用解释性提示迭代精炼分类标准的方法及其在2025年前在6个数据集中优于基线可解释机器学习模型的准确性和可解释性趋势研究 

---
# High-Power Training Data Identification with Provable Statistical Guarantees 

**Title (ZH)**: 具有可证明统计保证的高功率训练数据识别 

**Authors**: Zhenlong Liu, Hao Zeng, Weiran Huang, Hongxin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2510.09717)  

**Abstract**: Identifying training data within large-scale models is critical for copyright litigation, privacy auditing, and ensuring fair evaluation. The conventional approaches treat it as a simple binary classification task without statistical guarantees. A recent approach is designed to control the false discovery rate (FDR), but its guarantees rely on strong, easily violated assumptions. In this paper, we introduce Provable Training Data Identification (PTDI), a rigorous method that identifies a set of training data with strict false discovery rate (FDR) control. Specifically, our method computes p-values for each data point using a set of known unseen data, and then constructs a conservative estimator for the data usage proportion of the test set, which allows us to scale these p-values. Our approach then selects the final set of training data by identifying all points whose scaled p-values fall below a data-dependent threshold. This entire procedure enables the discovery of training data with provable, strict FDR control and significantly boosted power. Extensive experiments across a wide range of models (LLMs and VLMs), and datasets demonstrate that PTDI strictly controls the FDR and achieves higher power. 

**Abstract (ZH)**: 可证明训练数据识别（PTDI）：严格控制误发现率的方法 

---
# Group-Adaptive Adversarial Learning for Robust Fake News Detection Against Malicious Comments 

**Title (ZH)**: 针对恶意评论的鲁棒假新闻检测的组自适应对抗学习方法 

**Authors**: Zhao Tong, Chunlin Gong, Yimeng Gu, Haichao Shi, Qiang Liu, Shu Wu, Xiao-Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09712)  

**Abstract**: The spread of fake news online distorts public judgment and erodes trust in social media platforms. Although recent fake news detection (FND) models perform well in standard settings, they remain vulnerable to adversarial comments-authored by real users or by large language models (LLMs)-that subtly shift model decisions. In view of this, we first present a comprehensive evaluation of comment attacks to existing fake news detectors and then introduce a group-adaptive adversarial training strategy to improve the robustness of FND models. To be specific, our approach comprises three steps: (1) dividing adversarial comments into three psychologically grounded categories: perceptual, cognitive, and societal; (2) generating diverse, category-specific attacks via LLMs to enhance adversarial training; and (3) applying a Dirichlet-based adaptive sampling mechanism (InfoDirichlet Adjusting Mechanism) that dynamically adjusts the learning focus across different comment categories during training. Experiments on benchmark datasets show that our method maintains strong detection accuracy while substantially increasing robustness to a wide range of adversarial comment perturbations. 

**Abstract (ZH)**: 在线虚假新闻的传播扭曲了公众判断并侵蚀了对社交媒体平台的信任。尽管现有的虚假新闻检测模型在标准设置下表现出色，但它们仍然容易受到真实用户或大规模语言模型撰写的 adversarial comments 的攻击，这些评论会微妙地改变模型的决策。鉴于此，我们首先对现有虚假新闻检测器的评论攻击进行全面评估，然后提出了一种分组自适应对抗训练策略以提高虚假新闻检测模型的鲁棒性。具体而言，我们的方法包括三个步骤：（1）将 adversarial comments 分为三个基于心理的类别：知觉、认知和社会；（2）通过大规模语言模型生成多样化的、类别特定的攻击以增强对抗训练；（3）应用基于Dirichlet的自适应采样机制（信息Dirichlet调整机制），在训练过程中动态调整不同类别评论的学习重点。基准数据集上的实验表明，我们的方法在保持强检测准确性的同时，显著增强了对广泛类型的 adversarial comment 干扰的鲁棒性。 

---
# A Demonstration of Self-Adaptive Jamming Attack Detection in AI/ML Integrated O-RAN 

**Title (ZH)**: 自适应干扰攻击检测在AI/ML集成O-RAN中的演示 

**Authors**: Md Habibur Rahman, Md Sharif Hossen, Nathan H. Stephenson, Vijay K. Shah, Aloizio Da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2510.09706)  

**Abstract**: The open radio access network (O-RAN) enables modular, intelligent, and programmable 5G network architectures through the adoption of software-defined networking, network function virtualization, and implementation of standardized open interfaces. However, one of the security concerns for O-RAN, which can severely undermine network performance, is jamming attacks. This paper presents SAJD- a self-adaptive jammer detection framework that autonomously detects jamming attacks in AI/ML framework-integrated ORAN environments without human intervention. The SAJD framework forms a closed-loop system that includes near-realtime inference of radio signal jamming via our developed ML-based xApp, as well as continuous monitoring and retraining pipelines through rApps. In this demonstration, we will show how SAJD outperforms state-of-the-art jamming detection xApp (offline trained with manual labels) in terms of accuracy and adaptability under various dynamic and previously unseen interference scenarios in the O-RAN-compliant testbed. 

**Abstract (ZH)**: 基于AI/ML框架的自适应射频 jammer 检测框架：SAJD 

---
# Kelp: A Streaming Safeguard for Large Models via Latent Dynamics-Guided Risk Detection 

**Title (ZH)**: Kelp: 一种通过潜在动力学指导的风险检测.streaming安全保障大模型 

**Authors**: Xiaodan Li, Mengjie Wu, Yao Zhu, Yunna Lv, YueFeng Chen, Cen Chen, Jianmei Guo, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2510.09694)  

**Abstract**: Large models (LMs) are powerful content generators, yet their open-ended nature can also introduce potential risks, such as generating harmful or biased content. Existing guardrails mostly perform post-hoc detection that may expose unsafe content before it is caught, and the latency constraints further push them toward lightweight models, limiting detection accuracy. In this work, we propose Kelp, a novel plug-in framework that enables streaming risk detection within the LM generation pipeline. Kelp leverages intermediate LM hidden states through a Streaming Latent Dynamics Head (SLD), which models the temporal evolution of risk across the generated sequence for more accurate real-time risk detection. To ensure reliable streaming moderation in real applications, we introduce an Anchored Temporal Consistency (ATC) loss to enforce monotonic harm predictions by embedding a benign-then-harmful temporal prior. Besides, for a rigorous evaluation of streaming guardrails, we also present StreamGuardBench-a model-grounded benchmark featuring on-the-fly responses from each protected model, reflecting real-world streaming scenarios in both text and vision-language tasks. Across diverse models and datasets, Kelp consistently outperforms state-of-the-art post-hoc guardrails and prior plug-in probes (15.61% higher average F1), while using only 20M parameters and adding less than 0.5 ms of per-token latency. 

**Abstract (ZH)**: Large模型（LMs）是强大的内容生成工具，但其开放性也可能引入潜在风险，如生成有害或有偏见的内容。现有的防护栏大多是事后检测，可能在内容被发现之前就已经暴露了不安全的内容，而且延迟限制进一步推动它们采用轻量级模型，从而限制了检测准确性。在本文中，我们提出了一种名为Kelp的新型插件框架，可以在LM生成管道中实现流式风险检测。Kelp利用Streaming Latent Dynamics Head (SLD) 中间隐藏状态，模型生成序列中的风险随时间演变以实现更准确的实时风险检测。为了确保实际应用中的可靠流式内容审核，我们引入了锚定时间一致性（ATC）损失，以通过嵌入良性然后有害的时间先验来强制执行单调的有害性预测。此外，为了对流式防护栏进行严谨评估，我们还提出了StreamGuardBench，这是一种基于模型的基准，每个保护模型都能提供即时响应，反映了文本和视觉语言任务中的实际流式场景。在多种模型和数据集上，Kelp 在平均F1分数上持续优于最先进的事后防护栏和先前的插件探针（高出15.61%），同时仅使用20M参数并在每个token上增加不到0.5 ms的延迟。 

---
# Evaluation of Differential Privacy Mechanisms on Federated Learning 

**Title (ZH)**: 联邦学习中差分隐私机制的评估 

**Authors**: Tejash Varsani  

**Link**: [PDF](https://arxiv.org/pdf/2510.09691)  

**Abstract**: Federated learning is distributed model training across several clients without disclosing raw data. Despite advancements in data privacy, risks still remain. Differential Privacy (DP) is a technique to protect sensitive data by adding noise to model updates, usually controlled by a fixed privacy budget. However, this approach can introduce excessive noise, particularly when the model converges, which compromises performance. To address this problem, adaptive privacy budgets have been investigated as a potential solution. This work implements DP methods using Laplace and Gaussian mechanisms with an adaptive privacy budget, extending the SelecEval simulator. We introduce an adaptive clipping approach in the Gaussian mechanism, ensuring that gradients of the model are dynamically updated rather than using a fixed sensitivity. We conduct extensive experiments with various privacy budgets, IID and non-IID datasets, and different numbers of selected clients per round. While our experiments were limited to 200 training rounds, the results suggest that adaptive privacy budgets and adaptive clipping can help maintain model accuracy while preserving privacy. 

**Abstract (ZH)**: federated learning是通过多个客户端进行模型训练而不披露原始数据的技术。尽管在数据隐私方面取得了进展，风险仍然存在。差分隐私（DP）是一种通过在模型更新中添加噪声来保护敏感数据的技术，通常由固定隐私预算控制。然而，这种方法可能会在模型收敛时引入过多噪声，从而损害性能。为了解决这一问题，已经探索了可 adapters 隐私预算作为潜在的解决方案。本文使用拉普拉斯机制和高斯机制实现DP方法，并扩展了SelecEval模拟器，引入了高斯机制中的自适应裁剪方法，确保模型梯度动态更新而不是使用固定灵敏度。我们使用各种隐私预算、IID和非IID数据集以及每轮不同数量的选择客户端进行了广泛的实验。尽管我们的实验仅限于200轮训练，但结果表明，可 adapters 隐私预算和自适应裁剪有助于保持模型准确性同时保护隐私。 

---
# On the Occurence of Critical Learning Periods in Neural Networks 

**Title (ZH)**: 神经网络中的关键学习期发生研究 

**Authors**: Stanisław Pawlak  

**Link**: [PDF](https://arxiv.org/pdf/2510.09687)  

**Abstract**: This study delves into the plasticity of neural networks, offering empirical support for the notion that critical learning periods and warm-starting performance loss can be avoided through simple adjustments to learning hyperparameters. The critical learning phenomenon emerges when training is initiated with deficit data. Subsequently, after numerous deficit epochs, the network's plasticity wanes, impeding its capacity to achieve parity in accuracy with models trained from scratch, even when extensive clean data training follows deficit epochs. Building upon seminal research introducing critical learning periods, we replicate key findings and broaden the experimental scope of the main experiment from the original work. In addition, we consider a warm-starting approach and show that it can be seen as a form of deficit pretraining. In particular, we demonstrate that these problems can be averted by employing a cyclic learning rate schedule. Our findings not only impact neural network training practices but also establish a vital link between critical learning periods and ongoing research on warm-starting neural network training. 

**Abstract (ZH)**: 本研究探讨了神经网络的可塑性，提供了通过简单调整学习超参数来避免关键学习期和性能损失的实证支持。当训练初始使用缺陷数据时会出现关键学习现象。经过多个缺陷周期后，网络的可塑性减弱，即使后续使用大量干净数据进行训练，其准确度也无法与从头开始训练的模型相匹敌。基于引入关键学习期的开创性研究，我们复制了关键发现并扩大了主要实验的实验范围。此外，我们考虑了一种预热启动方法，并表明这可以被视为一种缺陷预训练形式。特别是，我们证明通过使用周期性学习率调度可以避免这些问题。本研究不仅影响神经网络训练实践，还建立了关键学习期与神经网络训练预热启动领域研究的相关性。 

---
# Stop DDoS Attacking the Research Community with AI-Generated Survey Papers 

**Title (ZH)**: 使用AI生成的调查论文遏制DDoS攻击科研社区 

**Authors**: Jianghao Lin, Rong Shan, Jiachen Zhu, Yunjia Xi, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09686)  

**Abstract**: Survey papers are foundational to the scholarly progress of research communities, offering structured overviews that guide both novices and experts across disciplines. However, the recent surge of AI-generated surveys, especially enabled by large language models (LLMs), has transformed this traditionally labor-intensive genre into a low-effort, high-volume output. While such automation lowers entry barriers, it also introduces a critical threat: the phenomenon we term the "survey paper DDoS attack" to the research community. This refers to the unchecked proliferation of superficially comprehensive but often redundant, low-quality, or even hallucinated survey manuscripts, which floods preprint platforms, overwhelms researchers, and erodes trust in the scientific record. In this position paper, we argue that we must stop uploading massive amounts of AI-generated survey papers (i.e., survey paper DDoS attack) to the research community, by instituting strong norms for AI-assisted review writing. We call for restoring expert oversight and transparency in AI usage and, moreover, developing new infrastructures such as Dynamic Live Surveys, community-maintained, version-controlled repositories that blend automated updates with human curation. Through quantitative trend analysis, quality audits, and cultural impact discussion, we show that safeguarding the integrity of surveys is no longer optional but imperative to the research community. 

**Abstract (ZH)**: 我们必须停止上传大量AI生成的综述论文（即综述论文DDoS攻击）并引进强规范以辅助AI同行评审写作，以恢复专家监督和透明度，并开发新的基础设施如动态实时综述，以维护综述的完整性。 

---
# Deep Neural Networks Inspired by Differential Equations 

**Title (ZH)**: 受微分方程启发的深度神经网络 

**Authors**: Yongshuai Liu, Lianfang Wang, Kuilin Qin, Qinghua Zhang, Faqiang Wang, Li Cui, Jun Liu, Yuping Duan, Tieyong Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.09685)  

**Abstract**: Deep learning has become a pivotal technology in fields such as computer vision, scientific computing, and dynamical systems, significantly advancing these disciplines. However, neural Networks persistently face challenges related to theoretical understanding, interpretability, and generalization. To address these issues, researchers are increasingly adopting a differential equations perspective to propose a unified theoretical framework and systematic design methodologies for neural networks. In this paper, we provide an extensive review of deep neural network architectures and dynamic modeling methods inspired by differential equations. We specifically examine deep neural network models and deterministic dynamical network constructs based on ordinary differential equations (ODEs), as well as regularization techniques and stochastic dynamical network models informed by stochastic differential equations (SDEs). We present numerical comparisons of these models to illustrate their characteristics and performance. Finally, we explore promising research directions in integrating differential equations with deep learning to offer new insights for developing intelligent computational methods that boast enhanced interpretability and generalization capabilities. 

**Abstract (ZH)**: 深度学习已成为计算机视觉、科学计算和动力系统等领域的关键技术，显著推动了这些学科的发展。然而，神经网络仍然面临理论理解、解释性和泛化能力等方面的挑战。为解决这些问题，研究人员越来越多地采用微分方程的观点，提出统一的理论框架和系统的设计方法。本文提供了深度神经网络架构和受微分方程启发的动态建模方法的全面综述。我们具体探讨了基于常微分方程（ODEs）的深层神经网络模型和确定性动态网络结构，以及基于随机微分方程（SDEs）的正则化技术和随机动态网络模型。我们通过数值比较展示了这些模型的特性与性能。最后，我们探讨了将微分方程与深度学习结合的研究方向，为开发具有增强解释性和泛化能力的智能计算方法提供新的见解。 

---
# AI in Computational Thinking Education in Higher Education: A Systematic Literature Review 

**Title (ZH)**: AI在高等教育中支持计算思维教育：一项系统文献综述 

**Authors**: Ebrahim Rahimi, Clara Maathuis  

**Link**: [PDF](https://arxiv.org/pdf/2510.09677)  

**Abstract**: Computational Thinking (CT) is a key skill set for students in higher education to thrive and adapt to an increasingly technology-driven future and workplace. While research on CT education has gained remarkable momentum in K12 over the past decade, it has remained under-explored in higher education, leaving higher education teachers with an insufficient overview, knowledge, and support regarding CT education. The proliferation and adoption of artificial intelligence (AI) by educational institutions have demonstrated promising potential to support instructional activities across many disciplines, including CT education. However, a comprehensive overview outlining the various aspects of integrating AI in CT education in higher education is lacking. To mitigate this gap, we conducted this systematic literature review study. The focus of our study is to identify initiatives applying AI in CT education within higher education and to explore various educational aspects of these initiatives, including the benefits and challenges of AI in CT education, instructional strategies employed, CT components covered, and AI techniques and models utilized. This study provides practical and scientific contributions to the CT education community, including an inventory of AI-based initiatives for CT education useful to educators, an overview of various aspects of integrating AI into CT education such as its benefits and challenges (e.g., AI potential to reshape CT education versus its potential to diminish students creativity) and insights into new and expanded perspectives on CT in light of AI (e.g., the decoding approach alongside the coding approach to CT). 

**Abstract (ZH)**: 计算思维（CT）是高等教育学生在日益技术驱动的未来和工作场所中生存和适应的关键技能。尽管过去十年中针对CT教育的研究在K12领域获得了显著进展，但在高等教育领域对此研究仍显不足，导致高等教育教师在CT教育方面缺乏足够的概括、知识和支持。教育机构采用人工智能（AI）的应用展示了在许多学科支持教学活动的巨大潜力，包括CT教育。然而，关于在高等教育中整合AI在CT教育方面的综合概述仍然缺乏。为了弥补这一差距，我们开展了这项系统文献综述研究。本研究的重点是识别高等教育中应用AI的CT教育举措，并探索这些举措的各种教育方面，包括AI在CT教育中的优势与挑战、所采用的教学策略、涵盖的CT组件以及使用的AI技术和模型。本研究为CT教育社区提供了实用和科学的贡献，包括一份有用的基于AI的CT教育举措清单，对整合AI到CT教育的各种方面（如AI对CT教育的重塑潜力与其对学生创造性可能的减弱）进行了概述，并提出了在AI背景下CT的新和扩展视角（如解码方法与编码方法相结合的CT教学途径）。 

---
# Coupled Data and Measurement Space Dynamics for Enhanced Diffusion Posterior Sampling 

**Title (ZH)**: 耦合数据与测量空间动力学以增强扩散后验采样 

**Authors**: Shayan Mohajer Hamidi, En-Hui Yang, Ben Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09676)  

**Abstract**: Inverse problems, where the goal is to recover an unknown signal from noisy or incomplete measurements, are central to applications in medical imaging, remote sensing, and computational biology. Diffusion models have recently emerged as powerful priors for solving such problems. However, existing methods either rely on projection-based techniques that enforce measurement consistency through heuristic updates, or they approximate the likelihood $p(\boldsymbol{y} \mid \boldsymbol{x})$, often resulting in artifacts and instability under complex or high-noise conditions. To address these limitations, we propose a novel framework called \emph{coupled data and measurement space diffusion posterior sampling} (C-DPS), which eliminates the need for constraint tuning or likelihood approximation. C-DPS introduces a forward stochastic process in the measurement space $\{\boldsymbol{y}_t\}$, evolving in parallel with the data-space diffusion $\{\boldsymbol{x}_t\}$, which enables the derivation of a closed-form posterior $p(\boldsymbol{x}_{t-1} \mid \boldsymbol{x}_t, \boldsymbol{y}_{t-1})$. This coupling allows for accurate and recursive sampling based on a well-defined posterior distribution. Empirical results demonstrate that C-DPS consistently outperforms existing baselines, both qualitatively and quantitatively, across multiple inverse problem benchmarks. 

**Abstract (ZH)**: 逆问题中的耦合数据空间和测量空间漂移后验采样方法（C-DPS） 

---
# Leveraging LLMs to Streamline the Review of Public Funding Applications 

**Title (ZH)**: 利用大语言模型简化公共资金申请审核流程 

**Authors**: Joao D.S. Marques, Andre V. Duarte, Andre Carvalho, Gil Rocha, Bruno Martins, Arlindo L. Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2510.09674)  

**Abstract**: Every year, the European Union and its member states allocate millions of euros to fund various development initiatives. However, the increasing number of applications received for these programs often creates significant bottlenecks in evaluation processes, due to limited human capacity. In this work, we detail the real-world deployment of AI-assisted evaluation within the pipeline of two government initiatives: (i) corporate applications aimed at international business expansion, and (ii) citizen reimbursement claims for investments in energy-efficient home improvements. While these two cases involve distinct evaluation procedures, our findings confirm that AI effectively enhanced processing efficiency and reduced workload across both types of applications. Specifically, in the citizen reimbursement claims initiative, our solution increased reviewer productivity by 20.1%, while keeping a negligible false-positive rate based on our test set observations. These improvements resulted in an overall reduction of more than 2 months in the total evaluation time, illustrating the impact of AI-driven automation in large-scale evaluation workflows. 

**Abstract (ZH)**: 欧盟及其成员国每年为各类发展项目拨款数百万欧元。然而，这些项目收到的应用数量不断增加，常常在评估过程中造成瓶颈，由于人力有限。本文详细介绍了人工智能辅助评估在两个政府倡议中的实际部署：（i）针对国际业务扩张的企业申请，以及（ii）公民能源高效家庭改造投资的报销索赔。尽管这两种情况涉及不同的评估程序，但我们的研究发现证实，人工智能有效提高了两类申请的处理效率并减轻了工作负担。具体而言，在公民报销索赔倡议中，我们的解决方案将审阅员的生产力提高了20.1%，同时根据测试集观察保持了微不足道的误检率。这些改进使得总体评估时间减少了超过2个月，展示了人工智能驱动的自动化对大规模评估工作流程的影响。 

---
# A Hybrid Computational Intelligence Framework with Metaheuristic Optimization for Drug-Drug Interaction Prediction 

**Title (ZH)**: 基于元启发式优化的混合计算智能框架在药品药物相互作用预测中的应用 

**Authors**: Maryam Abdollahi Shamami, Babak Teimourpour, Farshad Sharifi  

**Link**: [PDF](https://arxiv.org/pdf/2510.09668)  

**Abstract**: Drug-drug interactions (DDIs) are a leading cause of preventable adverse events, often complicating treatment and increasing healthcare costs. At the same time, knowing which drugs do not interact is equally important, as such knowledge supports safer prescriptions and better patient outcomes. In this study, we propose an interpretable and efficient framework that blends modern machine learning with domain knowledge to improve DDI prediction. Our approach combines two complementary molecular embeddings - Mol2Vec, which captures fragment-level structural patterns, and SMILES-BERT, which learns contextual chemical features - together with a leakage-free, rule-based clinical score (RBScore) that injects pharmacological knowledge without relying on interaction labels. A lightweight neural classifier is then optimized using a novel three-stage metaheuristic strategy (RSmpl-ACO-PSO), which balances global exploration and local refinement for stable performance. Experiments on real-world datasets demonstrate that the model achieves high predictive accuracy (ROC-AUC 0.911, PR-AUC 0.867 on DrugBank) and generalizes well to a clinically relevant Type 2 Diabetes Mellitus cohort. Beyond raw performance, studies show how embedding fusion, RBScore, and the optimizer each contribute to precision and robustness. Together, these results highlight a practical pathway for building reliable, interpretable, and computationally efficient models that can support safer drug therapies and clinical decision-making. 

**Abstract (ZH)**: 药物-药物相互作用预测的可解释高效框架：融合现代机器学习与领域知识以提高预测性能 

---
# Adversarial-Resilient RF Fingerprinting: A CNN-GAN Framework for Rogue Transmitter Detection 

**Title (ZH)**: 对抗扰动 resilient RF 印记：一种用于探测 rogue 发射器的 CNN-GAN 框架 

**Authors**: Raju Dhakal, Prashant Shekhar, Laxima Niure Kandel  

**Link**: [PDF](https://arxiv.org/pdf/2510.09663)  

**Abstract**: Radio Frequency Fingerprinting (RFF) has evolved as an effective solution for authenticating devices by leveraging the unique imperfections in hardware components involved in the signal generation process. In this work, we propose a Convolutional Neural Network (CNN) based framework for detecting rogue devices and identifying genuine ones using softmax probability thresholding. We emulate an attack scenario in which adversaries attempt to mimic the RF characteristics of genuine devices by training a Generative Adversarial Network (GAN) using In-phase and Quadrature (IQ) samples from genuine devices. The proposed approach is verified using IQ samples collected from ten different ADALM-PLUTO Software Defined Radios (SDRs), with seven devices considered genuine, two as rogue, and one used for validation to determine the threshold. 

**Abstract (ZH)**: 基于卷积神经网络的软阈值化射频指纹检测框架 

---
# Generative Models for Helmholtz Equation Solutions: A Dataset of Acoustic Materials 

**Title (ZH)**: 生成模型在亥姆霍兹方程解中的应用： acoustic 材料数据集 

**Authors**: Riccardo Fosco Gramaccioni, Christian Marinoni, Fabrizio Frezza, Aurelio Uncini, Danilo Comminiello  

**Link**: [PDF](https://arxiv.org/pdf/2510.09657)  

**Abstract**: Accurate simulation of wave propagation in complex acoustic materials is crucial for applications in sound design, noise control, and material engineering. Traditional numerical solvers, such as finite element methods, are computationally expensive, especially when dealing with large-scale or real-time scenarios. In this work, we introduce a dataset of 31,000 acoustic materials, named HA30K, designed and simulated solving the Helmholtz equations. For each material, we provide the geometric configuration and the corresponding pressure field solution, enabling data-driven approaches to learn Helmholtz equation solutions. As a baseline, we explore a deep learning approach based on Stable Diffusion with ControlNet, a state-of-the-art model for image generation. Unlike classical solvers, our approach leverages GPU parallelization to process multiple simulations simultaneously, drastically reducing computation time. By representing solutions as images, we bypass the need for complex simulation software and explicit equation-solving. Additionally, the number of diffusion steps can be adjusted at inference time, balancing speed and quality. We aim to demonstrate that deep learning-based methods are particularly useful in early-stage research, where rapid exploration is more critical than absolute accuracy. 

**Abstract (ZH)**: 准确模拟复杂声学材料中的波传播对于声学设计、噪声控制和材料工程的应用至关重要。传统的数值求解器，如有限元方法，在处理大规模或实时场景时计算成本高昂。在本工作中，我们引入了一个包含31,000种声学材料的数据集HA30K，这些材料通过求解亥姆霍兹方程设计并模拟。对于每种材料，我们提供了其几何配置和相应的压力场解，使得数据驱动的方法能够学习亥姆霍兹方程的解。作为基准，我们探索了基于Stable Diffusion with ControlNet的深度学习方法，这是一个用于图像生成的最先进的模型。与经典求解器不同，我们的方法利用GPU并行化同时处理多个模拟，大幅减少了计算时间。通过将解表示为图像，我们避免了使用复杂的仿真软件和显式方程求解的需求。此外，推理时可以调整扩散步数，平衡速度和质量。我们旨在证明基于深度学习的方法特别适用于早期研究阶段，此时快速探索比绝对准确性更为关键。 

---
# Rounding-Guided Backdoor Injection in Deep Learning Model Quantization 

**Title (ZH)**: 基于舍入引导的深度学习模型量化后门注入 

**Authors**: Xiangxiang Chen, Peixin Zhang, Jun Sun, Wenhai Wang, Jingyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.09647)  

**Abstract**: Model quantization is a popular technique for deploying deep learning models on resource-constrained environments. However, it may also introduce previously overlooked security risks. In this work, we present QuRA, a novel backdoor attack that exploits model quantization to embed malicious behaviors. Unlike conventional backdoor attacks relying on training data poisoning or model training manipulation, QuRA solely works using the quantization operations. In particular, QuRA first employs a novel weight selection strategy to identify critical weights that influence the backdoor target (with the goal of perserving the model's overall performance in mind). Then, by optimizing the rounding direction of these weights, we amplify the backdoor effect across model layers without degrading accuracy. Extensive experiments demonstrate that QuRA achieves nearly 100% attack success rates in most cases, with negligible performance degradation. Furthermore, we show that QuRA can adapt to bypass existing backdoor defenses, underscoring its threat potential. Our findings highlight critical vulnerability in widely used model quantization process, emphasizing the need for more robust security measures. Our implementation is available at this https URL. 

**Abstract (ZH)**: Model量化是一种在资源受限环境中部署深度学习模型的流行技术，但同时也可能引入先前未被注意的安全风险。在本文中，我们提出了QuRA，一种新颖的后门攻击方法，利用模型量化嵌入恶意行为。与依赖训练数据污染或模型训练操纵的传统后门攻击不同，QuRA仅通过量化操作工作。特别是在QuRA中，首先采用一种新颖的权重选择策略来识别影响后门目标的关键权重（同时考虑保持模型整体性能的因素），然后通过优化这些权重的舍入方向，在不影响准确性的前提下增强后门效果。大量实验表明，在大多数情况下，QuRA的攻击成功率接近100%，且性能下降微乎其微。此外，我们展示了QuRA可以适应绕过现有后门防御的能力，突显了其潜在威胁。我们的研究结果强调了广泛使用的模型量化过程中的关键漏洞，强调了需要更 robust 安全措施的必要性。我们的实现可在以下链接找到。 

---
# Enhanced Urban Traffic Management Using CCTV Surveillance Videos and Multi-Source Data Current State Prediction and Frequent Episode Mining 

**Title (ZH)**: 使用闭路电视监控视频和多源数据增强的城市交通管理：当前状态预测与频繁事件挖掘 

**Authors**: Shaharyar Alam Ansari, Mohammad Luqman, Aasim Zafar, Savir Ali  

**Link**: [PDF](https://arxiv.org/pdf/2510.09644)  

**Abstract**: Rapid urbanization has intensified traffic congestion, environmental strain, and inefficiencies in transportation systems, creating an urgent need for intelligent and adaptive traffic management solutions. Conventional systems relying on static signals and manual monitoring are inadequate for the dynamic nature of modern traffic. This research aims to develop a unified framework that integrates CCTV surveillance videos with multi-source data descriptors to enhance real-time urban traffic prediction. The proposed methodology incorporates spatio-temporal feature fusion, Frequent Episode Mining for sequential traffic pattern discovery, and a hybrid LSTM-Transformer model for robust traffic state forecasting. The framework was evaluated on the CityFlowV2 dataset comprising 313,931 annotated bounding boxes across 46 cameras. It achieved a high prediction accuracy of 98.46 percent, with a macro precision of 0.9800, macro recall of 0.9839, and macro F1-score of 0.9819. FEM analysis revealed significant sequential patterns such as moderate-congested transitions with confidence levels exceeding 55 percent. The 46 sustained congestion alerts are system-generated, which shows practical value for proactive congestion management. This emphasizes the need for the incorporation of video stream analytics with data from multiple sources for the design of real-time, responsive, adaptable multi-level intelligent transportation systems, which makes urban mobility smarter and safer. 

**Abstract (ZH)**: 快速城市化加剧了交通拥堵、环境压力和交通系统 inefficiencies，迫切需要智能和自适应的交通管理解决方案。传统的依赖静态信号和人工监控的系统无法满足现代交通的动态特性。本研究旨在开发一个结合闭路电视监控视频与多来源数据描述符的统一框架，以提升城市交通的实时预测。提出的方法结合了时空特征融合、频繁事件挖掘以发现序列交通模式，以及混合LSTM-Transformer模型进行稳健的交通状态预测。该框架在包含46个摄像头和313,931个标注边界框的CityFlowV2数据集上进行了评估，实现了98.46%的高预测准确率，宏精度为0.9800，宏召回率为0.9839，宏F1分为0.9819。FEM分析揭示了显著的序列模式，如中等拥堵转换，置信水平超过55%。46个持续拥堵警报由系统生成，体现了预防性拥堵管理的实际价值。这强调了将视频流分析与多源数据结合用于实时、响应性、自适应多层级智能交通系统设计的必要性，从而使城市交通更加智能和安全。 

---
# Direct Routing Gradient (DRGrad): A Personalized Information Surgery for Multi-Task Learning (MTL) Recommendations 

**Title (ZH)**: 直接路由梯度（DRGrad）：面向多任务学习推荐的个性化信息手术 

**Authors**: Yuguang Liu, Yiyun Miao, Luyao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2510.09643)  

**Abstract**: Multi-task learning (MTL) has emerged as a successful strategy in industrial-scale recommender systems, offering significant advantages such as capturing diverse users' interests and accurately detecting different behaviors like ``click" or ``dwell time". However, negative transfer and the seesaw phenomenon pose challenges to MTL models due to the complex and often contradictory task correlations in real-world recommendations. To address the problem while making better use of personalized information, we propose a personalized Direct Routing Gradient framework (DRGrad), which consists of three key components: router, updater and personalized gate network. DRGrad judges the stakes between tasks in the training process, which can leverage all valid gradients for the respective task to reduce conflicts. We evaluate the efficiency of DRGrad on complex MTL using a real-world recommendation dataset with 15 billion samples. The results show that DRGrad's superior performance over competing state-of-the-art MTL models, especially in terms of AUC (Area Under the Curve) metrics, indicating that it effectively manages task conflicts in multi-task learning environments without increasing model complexity, while also addressing the deficiencies in noise processing. Moreover, experiments on the public Census-income dataset and Synthetic dataset, have demonstrated the capability of DRGrad in judging and routing the stakes between tasks with varying degrees of correlation and personalization. 

**Abstract (ZH)**: 个性化直接路由梯度框架（DRGrad）：多任务学习中的任务冲突管理和个性化信息利用 

---
# Responsible AI Adoption in the Public Sector: A Data-Centric Taxonomy of AI Adoption Challenges 

**Title (ZH)**: 公共部门负责任人工智能采纳：以数据为中心的人工智能采纳挑战分类 

**Authors**: Anastasija Nikiforova, Martin Lnenicka, Ulf Melin, David Valle-Cruz, Asif Gill, Cesar Casiano Flores, Emyana Sirait, Mariusz Luterek, Richard Michael Dreyling, Barbora Tesarova  

**Link**: [PDF](https://arxiv.org/pdf/2510.09634)  

**Abstract**: Despite Artificial Intelligence (AI) transformative potential for public sector services, decision-making, and administrative efficiency, adoption remains uneven due to complex technical, organizational, and institutional challenges. Responsible AI frameworks emphasize fairness, accountability, and transparency, aligning with principles of trustworthy AI and fair AI, yet remain largely aspirational, overlooking technical and institutional realities, especially foundational data and governance. This study addresses this gap by developing a taxonomy of data-related challenges to responsible AI adoption in government. Based on a systematic review of 43 studies and 21 expert evaluations, the taxonomy identifies 13 key challenges across technological, organizational, and environmental dimensions, including poor data quality, limited AI-ready infrastructure, weak governance, misalignment in human-AI decision-making, economic and environmental sustainability concerns. Annotated with institutional pressures, the taxonomy serves as a diagnostic tool to surface 'symptoms' of high-risk AI deployment and guides policymakers in building the institutional and data governance conditions necessary for responsible AI adoption. 

**Abstract (ZH)**: 尽管人工智能（AI）在公共部门服务、决策和行政效率方面具有变革潜力，但由于复杂的技术、组织和制度挑战，其采用仍不均衡。负责任的人工智能框架强调公平、问责和透明度，符合可信赖的人工智能和公平人工智能的原则，但这些框架仍主要停留在理想层面，忽视了技术与制度现实，尤其是基础数据和治理问题。本研究通过开发政府负责任的人工智能采用相关数据挑战的分类框架来弥补这一缺口。基于对43篇研究文献和21位专家评估的系统性回顾，该分类框架识别了技术、组织和环境维度下的13项关键挑战，包括数据质量差、缺乏AI就绪基础设施、治理薄弱、人类-人工智能决策不匹配、经济与环境可持续性担忧。该分类框架注释了制度压力，作为一种诊断工具，用于揭示高风险人工智能部署的“症状”，并指导政策制定者构建实施负责任人工智能所需的数据和制度治理条件。 

---
# Hound: Relation-First Knowledge Graphs for Complex-System Reasoning in Security Audits 

**Title (ZH)**: hound: 关系优先的知识图谱及其在安全审计中复杂系统推理的应用 

**Authors**: Bernhard Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2510.09633)  

**Abstract**: Hound introduces a relation-first graph engine that improves system-level reasoning across interrelated components in complex codebases. The agent designs flexible, analyst-defined views with compact annotations (e.g., monetary/value flows, authentication/authorization roles, call graphs, protocol invariants) and uses them to anchor exact retrieval: for any question, it loads precisely the code that matters (often across components) so it can zoom out to system structure and zoom in to the decisive lines. A second contribution is a persistent belief system: long-lived vulnerability hypotheses whose confidence is updated as evidence accrues. The agent employs coverage-versus-intuition planning and a QA finalizer to confirm or reject hypotheses. On a five-project subset of ScaBench[1], Hound improves recall and F1 over a baseline LLM analyzer (micro recall 31.2% vs. 8.3%; F1 14.2% vs. 9.8%) with a modest precision trade-off. We attribute these gains to flexible, relation-first graphs that extend model understanding beyond call/dataflow to abstract aspects, plus the hypothesis-centric loop; code and artifacts are released to support reproduction. 

**Abstract (ZH)**: Hound 引入了一种关系优先的图引擎，提升了复杂代码库中相关组件间系统级推理能力。该代理设计了灵活的、分析师定义的观点，并使用紧凑的注解（如货币/价值流动、身份/授权角色、调用图、协议不变量）来锚定精确检索：对于任何一个问题，它都会加载相关的代码（往往是跨组件的），从而可以宏观审视系统结构，又能微观聚焦于关键代码行。第二个贡献是持久信念系统：长期存在的漏洞假设，随着证据的积累其置信度会被更新。代理使用覆盖率与直觉相结合的规划方法，并通过问答最终确认或拒绝假设。在 ScaBench 数据集的五个子项目上，Hound 在召回率和 F1 值上优于基准的大语言模型分析器（微观召回率 31.2% vs. 8.3%；F1 值 14.2% vs. 9.8%），以适度牺牲精度为代价。我们归功于这种提升的原因是灵活的关系优先图能够将模型的理解扩展到调用/数据流之外的抽象方面，加上以假设为中心的循环；代码和相关文件已发布以支持再现。 

---
# Toward a Unified Security Framework for AI Agents: Trust, Risk, and Liability 

**Title (ZH)**: 面向AI代理的统一安全框架：信任、风险与责任 

**Authors**: Jiayun Mo, Xin Kang, Tieyan Li, Zhongding Lei  

**Link**: [PDF](https://arxiv.org/pdf/2510.09620)  

**Abstract**: The excitement brought by the development of AI agents came alongside arising problems. These concerns centered around users' trust issues towards AIs, the risks involved, and the difficulty of attributing responsibilities and liabilities. Current solutions only attempt to target each problem separately without acknowledging their inter-influential nature. The Trust, Risk and Liability (TRL) framework proposed in this paper, however, ties together the interdependent relationships of trust, risk, and liability to provide a systematic method of building and enhancing trust, analyzing and mitigating risks, and allocating and attributing liabilities. It can be applied to analyze any application scenarios of AI agents and suggest appropriate measures fitting to the context. The implications of the TRL framework lie in its potential societal impacts, economic impacts, ethical impacts, and more. It is expected to bring remarkable values to addressing potential challenges and promoting trustworthy, risk-free, and responsible usage of AI in 6G networks. 

**Abstract (ZH)**: AI代理发展中带来的兴奋与出现的问题：信任、风险和责任框架在6G网络中促进可信、安全和负责任的AI应用 

---
# Causal Digital Twins for Cyber-Physical Security: A Framework for Robust Anomaly Detection in Industrial Control Systems 

**Title (ZH)**: 因果数字孪生在工业控制系统中鲁棒异常检测的框架：用于网络物理安全的方法 

**Authors**: Mohammadhossein Homaei, Mehran Tarif, Mar Avilla, Andres Caro  

**Link**: [PDF](https://arxiv.org/pdf/2510.09616)  

**Abstract**: Industrial Control Systems (ICS) face growing cyber-physical attacks that exploit both network vulnerabilities and physical processes. Current anomaly detection methods rely on correlation-based analysis, which cannot separate true causal relationships from spurious associations. This limitation results in high false alarm rates and poor root cause analysis. We propose a novel Causal Digital Twin (CDT) framework for cyber-physical security in medium-scale ICS. Our method combines causal inference theory with digital twin modeling. The framework enables three types of causal reasoning: association for pattern detection, intervention for understanding system responses, and counterfactual analysis for attack prevention planning. We evaluate our framework on three industrial datasets: SWaT, WADI, and HAI, with validation through physical constraint compliance (90.8\%) and synthetic ground truth testing (structural Hamming distance 0.13). Results show significant improvements over seven baseline methods. Our CDT achieves F1-scores are $0.944 \pm 0.014$ for SWaT, $0.902 \pm 0.021$ for WADI, and $0.923 \pm 0.018$ for HAI with statistical significance ($p < 0.0024$, Bonferroni corrected). The framework reduces false positives by \SI{74}{\percent} and achieves \SI{78.4}{\percent} root cause analysis accuracy compared to \SI{48.7}{\percent} for existing methods. Counterfactual analysis enables defense strategies that reduce attack success by \SI{73.2}{\percent}. The system keeps real-time performance with \SI{3.2}{ms} latency, which is suitable for industrial deployment, while providing interpretable explanations for operators. 

**Abstract (ZH)**: 工业控制系统中的因果数字孪生框架：面向中等规模ICS的因果推理与物理安全 

---
# Detecting Conspiracy Theory Against COVID-19 Vaccines 

**Title (ZH)**: 检测针对COVID-19疫苗的阴谋论 

**Authors**: Md Hasibul Amin, Harika Madanu, Sahithi Lavu, Hadi Mansourifar, Dana Alsagheer, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2211.13003)  

**Abstract**: Since the beginning of the vaccination trial, social media has been flooded with anti-vaccination comments and conspiracy beliefs. As the day passes, the number of COVID- 19 cases increases, and online platforms and a few news portals entertain sharing different conspiracy theories. The most popular conspiracy belief was the link between the 5G network spreading COVID-19 and the Chinese government spreading the virus as a bioweapon, which initially created racial hatred. Although some disbelief has less impact on society, others create massive destruction. For example, the 5G conspiracy led to the burn of the 5G Tower, and belief in the Chinese bioweapon story promoted an attack on the Asian-Americans. Another popular conspiracy belief was that Bill Gates spread this Coronavirus disease (COVID-19) by launching a mass vaccination program to track everyone. This Conspiracy belief creates distrust issues among laypeople and creates vaccine hesitancy. This study aims to discover the conspiracy theory against the vaccine on social platforms. We performed a sentiment analysis on the 598 unique sample comments related to COVID-19 vaccines. We used two different models, BERT and Perspective API, to find out the sentiment and toxicity of the sentence toward the COVID-19 vaccine. 

**Abstract (ZH)**: 自疫苗试验开始以来，社交媒体上充斥着反疫苗评论和阴谋论。随着COVID-19病例的增加，网络平台和少数新闻门户网站在分享不同阴谋论。最受欢迎的阴谋论涉及5G网络传播COVID-19与中国政府将其作为生物武器传播之间的联系，最初引发了种族仇恨。虽然一些怀疑论对社会影响较小，但其他阴谋论造成了重大破坏。例如，5G阴谋论导致了破坏5G基站的行为，而相信中国生物武器故事则促进了对亚裔美国人的攻击。另一个流行的阴谋论是比尔·盖茨通过推出大规模疫苗接种计划传播了这种冠状病毒疾病，以监控每个人。这种阴谋论创建了普通民众的信任问题，并引发疫苗犹豫。本文旨在发现针对疫苗的阴谋论在社交平台上的情况。我们对与COVID-19疫苗相关的598条独特样本评论进行了情感分析。我们使用了两种不同的模型——BERT和Perspective API，以确定句子对COVID-19疫苗的情感和毒性。 

---
