# H$^{\mathbf{3}}$DP: Triply-Hierarchical Diffusion Policy for Visuomotor Learning 

**Title (ZH)**: H$^{\mathbf{3}}$DP：三层次扩散策略用于视动学习 

**Authors**: Yiyang Lu, Yufeng Tian, Zhecheng Yuan, Xianbang Wang, Pu Hua, Zhengrong Xue, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07819)  

**Abstract**: Visuomotor policy learning has witnessed substantial progress in robotic manipulation, with recent approaches predominantly relying on generative models to model the action distribution. However, these methods often overlook the critical coupling between visual perception and action prediction. In this work, we introduce $\textbf{Triply-Hierarchical Diffusion Policy}~(\textbf{H$^{\mathbf{3}}$DP})$, a novel visuomotor learning framework that explicitly incorporates hierarchical structures to strengthen the integration between visual features and action generation. H$^{3}$DP contains $\mathbf{3}$ levels of hierarchy: (1) depth-aware input layering that organizes RGB-D observations based on depth information; (2) multi-scale visual representations that encode semantic features at varying levels of granularity; and (3) a hierarchically conditioned diffusion process that aligns the generation of coarse-to-fine actions with corresponding visual features. Extensive experiments demonstrate that H$^{3}$DP yields a $\mathbf{+27.5\%}$ average relative improvement over baselines across $\mathbf{44}$ simulation tasks and achieves superior performance in $\mathbf{4}$ challenging bimanual real-world manipulation tasks. Project Page: this https URL. 

**Abstract (ZH)**: 三层次分级扩散策略（H$^{3}$DP）：一种加强视觉特征与动作生成整合的类手控学习框架 

---
# Pixel Motion as Universal Representation for Robot Control 

**Title (ZH)**: 像素运动作为机器人控制的通用表示 

**Authors**: Kanchana Ranasinghe, Xiang Li, Cristina Mata, Jongwoo Park, Michael S Ryoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.07817)  

**Abstract**: We present LangToMo, a vision-language-action framework structured as a dual-system architecture that uses pixel motion forecasts as intermediate representations. Our high-level System 2, an image diffusion model, generates text-conditioned pixel motion sequences from a single frame to guide robot control. Pixel motion-a universal, interpretable, and motion-centric representation-can be extracted from videos in a self-supervised manner, enabling diffusion model training on web-scale video-caption data. Treating generated pixel motion as learned universal representations, our low level System 1 module translates these into robot actions via motion-to-action mapping functions, which can be either hand-crafted or learned with minimal supervision. System 2 operates as a high-level policy applied at sparse temporal intervals, while System 1 acts as a low-level policy at dense temporal intervals. This hierarchical decoupling enables flexible, scalable, and generalizable robot control under both unsupervised and supervised settings, bridging the gap between language, motion, and action. Checkout this https URL for visualizations. 

**Abstract (ZH)**: LangToMo：一种基于像素运动预测的视觉-语言-动作框架 

---
# Imagine, Verify, Execute: Memory-Guided Agentic Exploration with Vision-Language Models 

**Title (ZH)**: 想象、验证与执行：基于视觉-语言模型的记忆引导代理探索 

**Authors**: Seungjae Lee, Daniel Ekpo, Haowen Liu, Furong Huang, Abhinav Shrivastava, Jia-Bin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07815)  

**Abstract**: Exploration is essential for general-purpose robotic learning, especially in open-ended environments where dense rewards, explicit goals, or task-specific supervision are scarce. Vision-language models (VLMs), with their semantic reasoning over objects, spatial relations, and potential outcomes, present a compelling foundation for generating high-level exploratory behaviors. However, their outputs are often ungrounded, making it difficult to determine whether imagined transitions are physically feasible or informative. To bridge the gap between imagination and execution, we present IVE (Imagine, Verify, Execute), an agentic exploration framework inspired by human curiosity. Human exploration is often driven by the desire to discover novel scene configurations and to deepen understanding of the environment. Similarly, IVE leverages VLMs to abstract RGB-D observations into semantic scene graphs, imagine novel scenes, predict their physical plausibility, and generate executable skill sequences through action tools. We evaluate IVE in both simulated and real-world tabletop environments. The results show that IVE enables more diverse and meaningful exploration than RL baselines, as evidenced by a 4.1 to 7.8x increase in the entropy of visited states. Moreover, the collected experience supports downstream learning, producing policies that closely match or exceed the performance of those trained on human-collected demonstrations. 

**Abstract (ZH)**: 探索对于通用机器人学习至关重要，特别是在奖励稀疏、缺乏明确目标或任务特定监督的开放环境中。视觉语言模型（VLMs）通过其对对象、空间关系和潜在结果的语义推理，为生成高层次的探索行为提供了有力的基础。然而，它们的输出往往是未grounded的，难以确定想象中的过渡是否物理上可行或具有信息性。为了弥合想象与执行之间的差距，我们提出了IVE（Imagine, Verify, Execute）探索框架，该框架受到人类好奇心的启发。人类的探索往往受到发现新型场景配置和加深对环境理解的驱动。同样，IVE 利用VLMs将RGB-D观察抽象成语义场景图，设想新的场景，预测它们的物理可行性，并通过动作工具生成可执行的能力序列。我们在模拟和真实世界的桌面上评估了IVE。结果表明，IVE 使探索变得更加多样和有意义，如访问状态的熵相比RL基准提高了4.1到7.8倍。此外，收集的经验支持下游学习，产生的策略与或超过了在人类收集的示范上训练的性能。 

---
# DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies 

**Title (ZH)**: DexWild: 用于野外机器人政策的灵巧人类交互 

**Authors**: Tony Tao, Mohan Kumar Srirama, Jason Jingzhou Liu, Kenneth Shaw, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2505.07813)  

**Abstract**: Large-scale, diverse robot datasets have emerged as a promising path toward enabling dexterous manipulation policies to generalize to novel environments, but acquiring such datasets presents many challenges. While teleoperation provides high-fidelity datasets, its high cost limits its scalability. Instead, what if people could use their own hands, just as they do in everyday life, to collect data? In DexWild, a diverse team of data collectors uses their hands to collect hours of interactions across a multitude of environments and objects. To record this data, we create DexWild-System, a low-cost, mobile, and easy-to-use device. The DexWild learning framework co-trains on both human and robot demonstrations, leading to improved performance compared to training on each dataset individually. This combination results in robust robot policies capable of generalizing to novel environments, tasks, and embodiments with minimal additional robot-specific data. Experimental results demonstrate that DexWild significantly improves performance, achieving a 68.5% success rate in unseen environments-nearly four times higher than policies trained with robot data only-and offering 5.8x better cross-embodiment generalization. Video results, codebases, and instructions at this https URL 

**Abstract (ZH)**: 大规模多样化的机器人数据集已成为使灵巧 manipulation 策略能够泛化到新环境的一种有前途的途径，但获取此类数据集存在许多挑战。虽然远程操作可以提供高质量的数据集，但其高昂的成本限制了其可扩展性。相反，如果人们能够像在日常生活中一样使用他们自己的手来收集数据会如何？在 DexWild 中，一个多样化的数据收集团队使用他们的手在多种环境和物体上收集了数小时的交互数据。为了记录这些数据，我们创建了 DexWild-System，这是一种低成本、便携且易于使用的设备。DexWild 学习框架在人类和机器人演示的同时进行训练，与单独训练每个数据集相比，具有更好的性能。这种组合导致了鲁棒性更强的机器人策略，能够在最少的额外机器人特定数据的情况下泛化到新环境、任务和实体中。实验结果表明，DexWild 显著提高了性能，在未见过的环境中成功率达到 68.5%，几乎是仅使用机器人数据训练的策略成功率的四倍，并且在跨实体泛化方面表现出了 5.8 倍的改进。更多视频结果、代码库和使用说明请参见此链接。 

---
# AcoustoBots: A swarm of robots for acoustophoretic multimodal interactions 

**Title (ZH)**: 声波机器人：一种声学偏转多模态交互机器人集群 

**Authors**: Narsimlu Kemsaram, James Hardwick, Jincheng Wang, Bonot Gautam, Ceylan Besevli, Giorgos Christopoulos, Sourabh Dogra, Lei Gao, Akin Delibasi, Diego Martinez Plasencia, Orestis Georgiou, Marianna Obrist, Ryuji Hirayama, Sriram Subramanian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07808)  

**Abstract**: Acoustophoresis has enabled novel interaction capabilities, such as levitation, volumetric displays, mid-air haptic feedback, and directional sound generation, to open new forms of multimodal interactions. However, its traditional implementation as a singular static unit limits its dynamic range and application versatility. This paper introduces AcoustoBots - a novel convergence of acoustophoresis with a movable and reconfigurable phased array of transducers for enhanced application versatility. We mount a phased array of transducers on a swarm of robots to harness the benefits of multiple mobile acoustophoretic units. This offers a more flexible and interactive platform that enables a swarm of acoustophoretic multimodal interactions. Our novel AcoustoBots design includes a hinge actuation system that controls the orientation of the mounted phased array of transducers to achieve high flexibility in a swarm of acoustophoretic multimodal interactions. In addition, we designed a BeadDispenserBot that can deliver particles to trapping locations, which automates the acoustic levitation interaction. These attributes allow AcoustoBots to independently work for a common cause and interchange between modalities, allowing for novel augmentations (e.g., a swarm of haptics, audio, and levitation) and bilateral interactions with users in an expanded interaction area. We detail our design considerations, challenges, and methodological approach to extend acoustophoretic central control in distributed settings. This work demonstrates a scalable acoustic control framework with two mobile robots, laying the groundwork for future deployment in larger robotic swarms. Finally, we characterize the performance of our AcoustoBots and explore the potential interactive scenarios they can enable. 

**Abstract (ZH)**: 声波操控使新型交互能力成为可能，如悬浮、体积显示、空中触觉反馈和定向声音生成，从而开辟了新的多模态交互形式。然而，其传统实施作为单一静态单元限制了其动态范围和应用 versatility。本文介绍了AcoustoBots——将声波操控与可移动和可重构的 phased array 传感器集成，以增强应用 versatility。我们将 phased array 传感器安装在一群机器人上，利用多个移动声波操控单元的优势，提供更灵活和交互的平台，以实现声波操控多模态交互的群集。我们的新型AcoustoBots设计包括一个铰接驱动系统，控制安装的 phased array 传感器的朝向，以实现声波操控多模态交互群集的高度灵活性。此外，我们设计了珠粒分配机器人BeadDispenserBot，可以将颗粒输送到捕获位置，以实现声波悬浮交互的自动化。这些特性允许AcoustoBots独立工作以共同实现目标，并在不同模态之间进行切换，从而实现新颖的增强功能（例如，一群触觉、声音和悬浮）及与用户的双向交互，扩展交互区域。我们详细阐述了设计考虑、挑战和方法论，以扩展分布式设置中的声波操控中心控制。这项工作展示了具有两台移动机器人可扩展的声控框架，并为更大规模的机器人群部署奠定了基础。最后，我们评估了AcoustoBots的性能并探讨了它们能够实现的潜在交互场景。 

---
# Improving Trajectory Stitching with Flow Models 

**Title (ZH)**: 基于流模型改进轨迹拼接 

**Authors**: Reece O'Mahoney, Wanming Yu, Ioannis Havoutis  

**Link**: [PDF](https://arxiv.org/pdf/2505.07802)  

**Abstract**: Generative models have shown great promise as trajectory planners, given their affinity to modeling complex distributions and guidable inference process. Previous works have successfully applied these in the context of robotic manipulation but perform poorly when the required solution does not exist as a complete trajectory within the training set. We identify that this is a result of being unable to plan via stitching, and subsequently address the architectural and dataset choices needed to remedy this. On top of this, we propose a novel addition to the training and inference procedures to both stabilize and enhance these capabilities. We demonstrate the efficacy of our approach by generating plans with out of distribution boundary conditions and performing obstacle avoidance on the Franka Panda in simulation and on real hardware. In both of these tasks our method performs significantly better than the baselines and is able to avoid obstacles up to four times as large. 

**Abstract (ZH)**: 生成模型展现出作为轨迹规划器的巨大潜力，因为它们适合建模复杂分布并具有可控的推断过程。尽管以往工作成功地将这些模型应用于机器人操作中，但在解决方案不在训练集内的完整轨迹中时表现不佳。我们发现这是由于无法通过拼接来规划，因此我们识别出需要的架构和数据集选择以解决这一问题。在此基础上，我们提出了一种新的训练和推理过程的增补方法，以稳定并增强这些能力。通过在仿真和真实硬件上为弗兰卡· Panda机器人生成超出分布边界的计划并进行避障实验，我们展示了该方法的有效性。在这些任务中，我们的方法明显优于基线方法，并能够避开比基线方法大四倍的障碍物。 

---
# Multi-Agent Path Finding via Finite-Horizon Hierarchical Factorization 

**Title (ZH)**: 多智能体路径finding基于有限 horizon 分级因子分解 

**Authors**: Jiarui Li, Alessandro Zanardi, Gioele Zardini  

**Link**: [PDF](https://arxiv.org/pdf/2505.07779)  

**Abstract**: We present a novel algorithm for large-scale Multi-Agent Path Finding (MAPF) that enables fast, scalable planning in dynamic environments such as automated warehouses. Our approach introduces finite-horizon hierarchical factorization, a framework that plans one step at a time in a receding-horizon fashion. Robots first compute individual plans in parallel, and then dynamically group based on spatio-temporal conflicts and reachability. The framework accounts for conflict resolution, and for immediate execution and concurrent planning, significantly reducing response time compared to offline algorithms. Experimental results on benchmark maps demonstrate that our method achieves up to 60% reduction in time-to-first-action while consistently delivering high-quality solutions, outperforming state-of-the-art offline baselines across a range of problem sizes and planning horizons. 

**Abstract (ZH)**: 我们提出了一种新型的大规模多Agent路径规划算法，能够在包括自动化仓库在内的动态环境中实现快速、可扩展的规划。该方法引入了一种有限 horizon 的层次分解框架，以回溯 horizon 的方式逐步进行规划。机器人首先并行计算各自的路径计划，然后根据时空冲突和可达性动态分组。该框架考虑了解决冲突、即时执行和并发规划，显著减少了响应时间，与离线算法相比。基准地图上的实验结果表明，我们的方法在不牺牲高质量解决方案的前提下，将首次行动所需时间最多减少了60%，并在多种问题规模和规划 horizon 下优于最先进的离线基准方法。 

---
# Privacy Risks of Robot Vision: A User Study on Image Modalities and Resolution 

**Title (ZH)**: 机器人视觉的隐私风险：一项关于图像模态和分辨率的用户研究 

**Authors**: Xuying Huang, Sicong Pan, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.07766)  

**Abstract**: User privacy is a crucial concern in robotic applications, especially when mobile service robots are deployed in personal or sensitive environments. However, many robotic downstream tasks require the use of cameras, which may raise privacy risks. To better understand user perceptions of privacy in relation to visual data, we conducted a user study investigating how different image modalities and image resolutions affect users' privacy concerns. The results show that depth images are broadly viewed as privacy-safe, and a similarly high proportion of respondents feel the same about semantic segmentation images. Additionally, the majority of participants consider 32*32 resolution RGB images to be almost sufficiently privacy-preserving, while most believe that 16*16 resolution can fully guarantee privacy protection. 

**Abstract (ZH)**: 移动服务机器人在个人或敏感环境中部署时，用户隐私是机器人应用中的一个重要关切。然而，许多下游任务需要使用摄像头，这可能会引发隐私风险。为了更好地了解视觉数据与用户隐私感知之间的关系，我们开展了一项用户研究，调查不同的图像模态和分辨率如何影响用户的隐私担忧。研究结果表明，_DEPTH IMAGES ARE WIDELY CONSIDERED AS PRIVACY-SAFE_，并且有相似比例的受访者对语义分割图像也持同样看法。此外，大多数参与者认为32*32分辨率的RGB图像几乎足以保护隐私，而大多数人认为16*16分辨率可以完全确保隐私保护。 

---
# Guiding Data Collection via Factored Scaling Curves 

**Title (ZH)**: 通过因子缩放曲线指导数据采集 

**Authors**: Lihan Zha, Apurva Badithela, Michael Zhang, Justin Lidard, Jeremy Bao, Emily Zhou, David Snyder, Allen Z. Ren, Dhruv Shah, Anirudha Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.07728)  

**Abstract**: Generalist imitation learning policies trained on large datasets show great promise for solving diverse manipulation tasks. However, to ensure generalization to different conditions, policies need to be trained with data collected across a large set of environmental factor variations (e.g., camera pose, table height, distractors) $-$ a prohibitively expensive undertaking, if done exhaustively. We introduce a principled method for deciding what data to collect and how much to collect for each factor by constructing factored scaling curves (FSC), which quantify how policy performance varies as data scales along individual or paired factors. These curves enable targeted data acquisition for the most influential factor combinations within a given budget. We evaluate the proposed method through extensive simulated and real-world experiments, across both training-from-scratch and fine-tuning settings, and show that it boosts success rates in real-world tasks in new environments by up to 26% over existing data-collection strategies. We further demonstrate how factored scaling curves can effectively guide data collection using an offline metric, without requiring real-world evaluation at scale. 

**Abstract (ZH)**: 通用模仿学习策略在大数据集上训练显示出解决多样化操作任务的巨大潜力。然而，为了确保在不同条件下的推广性，策略需要在大量环境因素变异的数据集中进行训练（例如，相机姿态、桌面高度、干扰物）——如果进行全面收集，这将是一项代价高昂的任务。我们介绍了一种原则性方法，通过构建因素缩放曲线（FSC）来决定收集哪些数据以及每种因素收集多少数据，这些曲线量化了随单一因素或配对因素缩放时策略性能的变化。这些曲线能够在给定预算内针对最具影响力的因素组合进行有针对性的数据收集。我们通过广泛的模拟和现实世界实验来评估所提出的方法，在从零开始训练和微调设置中均表明，与现有的数据收集策略相比，它能够将新环境中实际任务的成功率提升至多26%。我们进一步展示了如何使用离线指标有效引导数据收集，而无需大规模进行现实世界的评估。 

---
# Hybrid Control Strategies for Safe and Adaptive Robot-Assisted Dressing 

**Title (ZH)**: 混合控制策略实现安全自适应机器人辅助穿衣 

**Authors**: Yasmin Rafiq, Baslin A. James, Ke Xu, Robert M. Hierons, Sanja Dogramadzi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07710)  

**Abstract**: Safety, reliability, and user trust are crucial in human-robot interaction (HRI) where the robots must address hazards in real-time. This study presents hazard driven low-level control strategies implemented in robot-assisted dressing (RAD) scenarios where hazards like garment snags and user discomfort in real-time can affect task performance and user safety. The proposed control mechanisms include: (1) Garment Snagging Control Strategy, which detects excessive forces and either seeks user intervention via a chatbot or autonomously adjusts its trajectory, and (2) User Discomfort/Pain Mitigation Strategy, which dynamically reduces velocity based on user feedback and aborts the task if necessary. We used physical dressing trials in order to evaluate these control strategies. Results confirm that integrating force monitoring with user feedback improves safety and task continuity. The findings emphasise the need for hybrid approaches that balance autonomous intervention, user involvement, and controlled task termination, supported by bi-directional interaction and real-time user-driven adaptability, paving the way for more responsive and personalised HRI systems. 

**Abstract (ZH)**: 人类与机器人交互中的安全性、可靠性和用户信任至关重要，其中机器人必须实时应对潜在风险。本文提出了在辅助穿衣（RAD）场景中基于风险的低级控制策略，以实时应对如衣物缠绊和用户不适等风险对任务表现和用户安全的影响。提出的控制机制包括：（1）衣物缠绊控制策略，检测到过大力量时，通过聊天机器人寻求用户干预或自主调整路径，（2）用户不适/疼痛缓解策略，根据用户反馈动态降低速度，并在必要时取消任务。我们通过物理穿衣实验来评估这些控制策略。结果表明，将力量监测与用户反馈相结合，可以提高安全性和任务连续性。研究强调了需要采用混合方法平衡自主干预、用户参与和可控任务终止，并支持双向交互和实时用户驱动的适应性，为更加响应性和个性化的交互式机器人系统铺平了道路。 

---
# FD-RIO: Fast Dense Radar Inertial Odometry 

**Title (ZH)**: FD-RIO：快速密集雷达惯性里程计 

**Authors**: Nader J. Abu-Alrub, Nathir A. Rawashdeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.07694)  

**Abstract**: Radar-based odometry is a popular solution for ego-motion estimation in conditions where other exteroceptive sensors may degrade, whether due to poor lighting or challenging weather conditions; however, scanning radars have the downside of relatively lower sampling rate and spatial resolution. In this work, we present FD-RIO, a method to alleviate this problem by fusing noisy, drift-prone, but high-frequency IMU data with dense radar scans. To the best of our knowledge, this is the first attempt to fuse dense scanning radar odometry with IMU using a Kalman filter. We evaluate our methods using two publicly available datasets and report accuracies using standard KITTI evaluation metrics, in addition to ablation tests and runtime analysis. Our phase correlation -based approach is compact, intuitive, and is designed to be a practical solution deployable on a realistic hardware setup of a mobile platform. Despite its simplicity, FD-RIO is on par with other state-of-the-art methods and outperforms in some test sequences. 

**Abstract (ZH)**: 基于雷达的里程计在其他外感知传感器因光照不良或恶劣天气条件而性能下降的情况下，是一种流行的自我运动估计解决方案；然而，扫描雷达的缺点是相对较低的采样率和空间分辨率。本文提出了FD-RIO方法，通过将噪声较大、易漂移但频率高的IMU数据与密集的雷达扫描数据融合来缓解这一问题。据我们所知，这是首次尝试使用卡尔曼滤波器融合密集扫描雷达里程计与IMU数据。我们使用两个公开的数据集评估了该方法，并使用标准KITTI评估指标报告了准确性，同时进行了消融测试和运行时分析。基于相位相关的方法紧凑、直观，并设计为可在移动平台的实际硬件配置中部署的实用解决方案。尽管简单，FD-RIO与其它最先进的方法相比表现相当，在某些测试序列中性能更优。 

---
# DATAMUt: Deterministic Algorithms for Time-Delay Attack Detection in Multi-Hop UAV Networks 

**Title (ZH)**: DATAMMut：多跳无人机网络中确定性时间延迟攻击检测算法 

**Authors**: Keiwan Soltani, Federico Corò, Punyasha Chatterjee, Sajal K. Das  

**Link**: [PDF](https://arxiv.org/pdf/2505.07670)  

**Abstract**: Unmanned Aerial Vehicles (UAVs), also known as drones, have gained popularity in various fields such as agriculture, emergency response, and search and rescue operations. UAV networks are susceptible to several security threats, such as wormhole, jamming, spoofing, and false data injection. Time Delay Attack (TDA) is a unique attack in which malicious UAVs intentionally delay packet forwarding, posing significant threats, especially in time-sensitive applications. It is challenging to distinguish malicious delay from benign network delay due to the dynamic nature of UAV networks, intermittent wireless connectivity, or the Store-Carry-Forward (SCF) mechanism during multi-hop communication. Some existing works propose machine learning-based centralized approaches to detect TDA, which are computationally intensive and have large message overheads. This paper proposes a novel approach DATAMUt, where the temporal dynamics of the network are represented by a weighted time-window graph (TWiG), and then two deterministic polynomial-time algorithms are presented to detect TDA when UAVs have global and local network knowledge. Simulation studies show that the proposed algorithms have reduced message overhead by a factor of five and twelve in global and local knowledge, respectively, compared to existing approaches. Additionally, our approaches achieve approximately 860 and 1050 times less execution time in global and local knowledge, respectively, outperforming the existing methods. 

**Abstract (ZH)**: 无人机（UAVs）在农业、应急响应和搜救等各个领域受欢迎。UAV网络面临多重安全威胁，如 wormhole、jamming、spoofing 和 false data injection。时间延迟攻击（TDA）是一种恶意UAV故意延迟数据包转发的独特攻击，尤其在时间敏感应用中构成重大威胁。由于无人机网络的动态性、间歇性无线连接或多跳通信中的Store-Carry-Forward (SCF)机制，区分恶意延迟与良性的网络延迟具有挑战性。已有工作提出基于机器学习的集中式方法来检测TDA，这些方法计算密集且消息开销大。本文提出了一种新颖的方法DATAMUt，其中通过加权时间窗口图（TWiG）表示网络的时间动态，并提出了两种确定性的多项式时间算法，在无人机具有全局和局部网络知识时检测TDA。仿真研究表明，与现有方法相比，所提出算法在全局知识和局部知识下的消息开销分别减少了5倍和12倍。此外，在全局和局部知识下，我们的方法的执行时间分别减少了约860倍和1050倍，优于现有方法。 

---
# Intuitive Human-Robot Interfaces Leveraging on Autonomy Features for the Control of Highly-redundant Robots 

**Title (ZH)**: 基于自主性特征的直观人机机器人接口及其在高冗余度机器人控制中的应用 

**Authors**: Davide Torielli  

**Link**: [PDF](https://arxiv.org/pdf/2505.07668)  

**Abstract**: [...] With the TelePhysicalOperation interface, the user can teleoperate the different capabilities of a robot (e.g., single/double arm manipulation, wheel/leg locomotion) by applying virtual forces on selected robot body parts. This approach emulates the intuitiveness of physical human-robot interaction, but at the same time it permits to teleoperate the robot from a safe distance, in a way that resembles a "Marionette" interface. The system is further enhanced with wearable haptic feedback functions to align better with the "Marionette" metaphor, and a user study has been conducted to validate its efficacy with and without the haptic channel enabled. Considering the importance of robot independence, the TelePhysicalOperation interface incorporates autonomy modules to face, for example, the teleoperation of dual-arm mobile base robots for bimanual object grasping and transportation tasks.
With the laser-guided interface, the user can indicate points of interest to the robot through the utilization of a simple but effective laser emitter device. With a neural network-based vision system, the robot tracks the laser projection in real time, allowing the user to indicate not only fixed goals, like objects, but also paths to follow. With the implemented autonomous behavior, a mobile manipulator employs its locomanipulation abilities to follow the indicated goals. The behavior is modeled using Behavior Trees, exploiting their reactivity to promptly respond to changes in goal positions, and their modularity to adapt the motion planning to the task needs. The proposed laser interface has also been employed in an assistive scenario. In this case, users with upper limbs impairments can control an assistive manipulator by directing a head-worn laser emitter to the point of interests, to collaboratively address activities of everyday life. [...] 

**Abstract (ZH)**: 通过TelePhysicalOperation界面，用户可以通过在选定的机器人身体部位应用虚拟力来远程操作机器人的不同能力（如单臂/双臂操作、轮式/腿式移动）。这种方法模仿了物理的人机互动直觉性，同时允许用户从安全距离远程操作机器人，类似于“提线木偶”接口。该系统进一步增强了穿戴式力反馈功能，更好地契合“提线木偶”的隐喻，并进行了用户研究以验证其在启用和未启用力反馈通道时的有效性。考虑到机器人独立性的重要性，TelePhysicalOperation界面集成自主模块，例如，用于远程操作具有双臂移动基座的机器人进行双臂对象抓取和运输任务。

通过激光引导界面，用户可以通过简单的激光发射器设备指示机器人感兴趣点。借助基于神经网络的视觉系统，机器人可以实时跟踪激光投影，从而让用户不仅指示固定目标（如物体），还可以指示跟随的路径。在实现的自主行为中，移动机器人利用其定位操作能力跟随指示目标。行为被建模为行为树，利用其实时响应能力对目标位置变化做出迅速反应，并利用其模块化特性以适应任务需求进行运动规划。所提出的激光界面还应用于辅助场景。在这种情况下，上肢受损的用户可以通过头部佩戴的激光发射器控制辅助操作器，以协作应对日常生活活动。 

---
# Neural Brain: A Neuroscience-inspired Framework for Embodied Agents 

**Title (ZH)**: 神经脑：一种受 Neuroscience 启发的体态智能框架 

**Authors**: Jian Liu, Xiongtao Shi, Thai Duy Nguyen, Haitian Zhang, Tianxiang Zhang, Wei Sun, Yanjie Li, Athanasios V. Vasilakos, Giovanni Iacca, Arshad Ali Khan, Arvind Kumar, Jae Won Cho, Ajmal Mian, Lihua Xie, Erik Cambria, Lin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07634)  

**Abstract**: The rapid evolution of artificial intelligence (AI) has shifted from static, data-driven models to dynamic systems capable of perceiving and interacting with real-world environments. Despite advancements in pattern recognition and symbolic reasoning, current AI systems, such as large language models, remain disembodied, unable to physically engage with the world. This limitation has driven the rise of embodied AI, where autonomous agents, such as humanoid robots, must navigate and manipulate unstructured environments with human-like adaptability. At the core of this challenge lies the concept of Neural Brain, a central intelligence system designed to drive embodied agents with human-like adaptability. A Neural Brain must seamlessly integrate multimodal sensing and perception with cognitive capabilities. Achieving this also requires an adaptive memory system and energy-efficient hardware-software co-design, enabling real-time action in dynamic environments. This paper introduces a unified framework for the Neural Brain of embodied agents, addressing two fundamental challenges: (1) defining the core components of Neural Brain and (2) bridging the gap between static AI models and the dynamic adaptability required for real-world deployment. To this end, we propose a biologically inspired architecture that integrates multimodal active sensing, perception-cognition-action function, neuroplasticity-based memory storage and updating, and neuromorphic hardware/software optimization. Furthermore, we also review the latest research on embodied agents across these four aspects and analyze the gap between current AI systems and human intelligence. By synthesizing insights from neuroscience, we outline a roadmap towards the development of generalizable, autonomous agents capable of human-level intelligence in real-world scenarios. 

**Abstract (ZH)**: 人工智能的快速演进已从静态的数据驱动模型转向能够感知和互动的实际环境中的动态系统。尽管在模式识别和符号推理方面取得了进展，当前的人工智能系统，如大规模语言模型，仍然不具备实体性，无法与现实世界进行物理交互。这一限制推动了具身人工智能的发展，其中自主代理，如类人机器人，必须以类似人类的适应性在未结构化环境中导航和操作。这一挑战的核心是神经脑的概念，这是一种中央智能系统，旨在以人类的适应性驱动具身代理。一个神经脑必须无缝地集成多模态感知与认知能力。实现这一点还需要一个适应性记忆系统和硬件-软件协同设计，以实现动态环境中的实时行动。本文提出了一种统一框架，提出了神经脑的具身代理，并解决了两个基本挑战：（1）定义神经脑的核心组件，（2）弥合静态人工智能模型与实际部署所需的动态适应性之间的差距。为此，我们提出了一种受生物学启发的架构，该架构集成了多模态主动感知、感知-认知-行动功能、基于神经可塑性的记忆存储和更新以及神经形态硬件/软件优化。此外，我们还回顾了这四个方面最新的具身代理研究，并分析了当前人工智能系统与人类智能之间的差距。通过综合神经科学的见解，我们提出了一个 roadmap，旨在开发出能够在实际场景中表现出人类水平智能的可移植、自主代理。 

---
# Beyond Static Perception: Integrating Temporal Context into VLMs for Cloth Folding 

**Title (ZH)**: 超越静态感知：将时间上下文集成到VLMs中的衣物折叠研究 

**Authors**: Oriol Barbany, Adrià Colomé, Carme Torras  

**Link**: [PDF](https://arxiv.org/pdf/2505.07600)  

**Abstract**: Manipulating clothes is challenging due to their complex dynamics, high deformability, and frequent self-occlusions. Garments exhibit a nearly infinite number of configurations, making explicit state representations difficult to define. In this paper, we analyze BiFold, a model that predicts language-conditioned pick-and-place actions from visual observations, while implicitly encoding garment state through end-to-end learning. To address scenarios such as crumpled garments or recovery from failed manipulations, BiFold leverages temporal context to improve state estimation. We examine the internal representations of the model and present evidence that its fine-tuning and temporal context enable effective alignment between text and image regions, as well as temporal consistency. 

**Abstract (ZH)**: 操纵衣物具有挑战性，因为衣物动态复杂、高度可变形且频繁自遮挡。衣物展现出几乎无限的配置方式，使得显式状态表示难以定义。本文分析了BiFold模型，该模型可以从视觉观察中预测语言条件下的拿取和放置动作，并通过端到端学习隐式编码衣物状态。为了应对褶皱衣物或失败操纵后的恢复等场景，BiFold利用时间上下文提高状态估计。我们研究了模型的内部表示，并展示了其微调和时间上下文能够有效实现文本区域和图像区域的对齐以及时间一致性。 

---
# On rapid parallel tuning of controllers of a swarm of MAVs -- distribution strategies of the updated gains 

**Title (ZH)**: 基于更新增益的分布策略 MAVs 舞异群控制器的快速并行调整 

**Authors**: Dariusz Horla, Wojciech Giernacki, Vít Krátký, Petr Štibinger, Tomáš Báča, Martin Saska  

**Link**: [PDF](https://arxiv.org/pdf/2505.07523)  

**Abstract**: In this paper, we present a reliable, scalable, time deterministic, model-free procedure to tune swarms of Micro Aerial Vehicles (MAVs) using basic sensory data. Two approaches to taking advantage of parallel tuning are presented. First, the tuning with averaging of the results on the basis of performance indices reported from the swarm with identical gains to decrease the negative effect of the noise in the measurements. Second, the tuning with parallel testing of varying set of gains across the swarm to reduce the tuning time. The presented methods were evaluated both in simulation and real-world experiments. The achieved results show the ability of the proposed approach to improve the results of the tuning while decreasing the tuning time, ensuring at the same time a reliable tuning mechanism. 

**Abstract (ZH)**: 本文提出了一种可靠、可扩展、具有时间确定性的模型-free调参方法，用于使用基本传感器数据调整微型空中车辆（MAVs）群。介绍了两种利用并行调参的方法。首先，通过在具有相同增益的群中平均性能指标结果来减少测量噪声的负面影响。其次，通过在群中并行测试不同的增益集来减少调参时间。本文的方法在仿真和实际试验中都进行了评估，实验结果表明所提出的方法能够提高调参结果，同时减少调参时间，确保调参机制的可靠性。 

---
# Average-Reward Maximum Entropy Reinforcement Learning for Global Policy in Double Pendulum Tasks 

**Title (ZH)**: 双摆任务中的平均奖励最大熵强化学习全局策略 

**Authors**: Jean Seong Bjorn Choe, Bumkyu Choi, Jong-kook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.07516)  

**Abstract**: This report presents our reinforcement learning-based approach for the swing-up and stabilisation tasks of the acrobot and pendubot, tailored specifcially to the updated guidelines of the 3rd AI Olympics at ICRA 2025. Building upon our previously developed Average-Reward Entropy Advantage Policy Optimization (AR-EAPO) algorithm, we refined our solution to effectively address the new competition scenarios and evaluation metrics. Extensive simulations validate that our controller robustly manages these revised tasks, demonstrating adaptability and effectiveness within the updated framework. 

**Abstract (ZH)**: 基于强化学习的Acrobot和Pendubot摆动-up和稳定任务解决方案——面向ICRA 2025 AI Olympics更新指南 

---
# GelFusion: Enhancing Robotic Manipulation under Visual Constraints via Visuotactile Fusion 

**Title (ZH)**: 视觉接触感知融合增强机器人在视觉约束下的操作 Manipulation under Visual Constraints via Visuotactile Fusion 

**Authors**: Shulong Jiang, Shiqi Zhao, Yuxuan Fan, Peng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07455)  

**Abstract**: Visuotactile sensing offers rich contact information that can help mitigate performance bottlenecks in imitation learning, particularly under vision-limited conditions, such as ambiguous visual cues or occlusions. Effectively fusing visual and visuotactile modalities, however, presents ongoing challenges. We introduce GelFusion, a framework designed to enhance policies by integrating visuotactile feedback, specifically from high-resolution GelSight sensors. GelFusion using a vision-dominated cross-attention fusion mechanism incorporates visuotactile information into policy learning. To better provide rich contact information, the framework's core component is our dual-channel visuotactile feature representation, simultaneously leveraging both texture-geometric and dynamic interaction features. We evaluated GelFusion on three contact-rich tasks: surface wiping, peg insertion, and fragile object pick-and-place. Outperforming baselines, GelFusion shows the value of its structure in improving the success rate of policy learning. 

**Abstract (ZH)**: 视觉触觉感知提供了丰富的接触信息，可以在视觉受限条件下，如含糊的视觉提示或遮挡情况下，帮助缓解模仿学习中的性能瓶颈。然而，有效地融合视觉与触觉模态仍然存在挑战。我们提出了GelFusion框架，旨在通过整合高分辨率GelSight传感器的触觉反馈来增强策略。GelFusion采用以视觉为主导的交叉注意力融合机制，将触觉信息融入到策略学习中。框架的核心组件是我们的双通道触觉特征表示，同时利用纹理几何和动态交互特征以提供丰富的接触信息。我们在三项接触丰富的任务（表面擦拭、 Peg插入和易碎物体拾放）上评估了GelFusion，结果显示其结构在提高策略学习成功率方面的价值。 

---
# TPT-Bench: A Large-Scale, Long-Term and Robot-Egocentric Dataset for Benchmarking Target Person Tracking 

**Title (ZH)**: TPT-Bench：一个用于目标人员跟踪基准测试的大规模、长期和机器人本体中心数据集 

**Authors**: Hanjing Ye, Yu Zhan, Weixi Situ, Guangcheng Chen, Jingwen Yu, Kuanqi Cai, Hong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07446)  

**Abstract**: Tracking a target person from robot-egocentric views is crucial for developing autonomous robots that provide continuous personalized assistance or collaboration in Human-Robot Interaction (HRI) and Embodied AI. However, most existing target person tracking (TPT) benchmarks are limited to controlled laboratory environments with few distractions, clean backgrounds, and short-term occlusions. In this paper, we introduce a large-scale dataset designed for TPT in crowded and unstructured environments, demonstrated through a robot-person following task. The dataset is collected by a human pushing a sensor-equipped cart while following a target person, capturing human-like following behavior and emphasizing long-term tracking challenges, including frequent occlusions and the need for re-identification from numerous pedestrians. It includes multi-modal data streams, including odometry, 3D LiDAR, IMU, panoptic, and RGB-D images, along with exhaustively annotated 2D bounding boxes of the target person across 35 sequences, both indoors and outdoors. Using this dataset and visual annotations, we perform extensive experiments with existing TPT methods, offering a thorough analysis of their limitations and suggesting future research directions. 

**Abstract (ZH)**: 面向拥挤和非结构化环境的目标人员跟踪数据集对于开发在人机交互（HRI）和具身AI中提供连续个性化协助或合作的自主机器人至关重要。然而，现有的大多数目标人员跟踪（TPT）基准仅限于少量干扰、干净背景和短暂遮挡的受控实验室环境。在本文中，我们介绍了一个大规模数据集，该数据集旨在用于拥挤和非结构化环境下的目标人员跟踪，并通过机器人追踪任务进行了展示。该数据集通过一名人类推动装有传感器的手推车并追随目标人员收集，捕捉类似人类的跟随行为，强调长期跟踪挑战，包括频繁遮挡和在众多行人的背景下重新识别的需求。该数据集包括多模态数据流，包括里程计、3D激光雷达、IMU、全景以及RGB-D图像，并且包含35个序列（室内和室外）中目标人员的详尽注释2D边界框。利用该数据集和视觉注释，我们对现有的TPT方法进行了广泛实验，提供了对其局限性的全面分析，并提出了未来研究方向。 

---
# Cooperative Assembly with Autonomous Mobile Manipulators in an Underwater Scenario 

**Title (ZH)**: 水下场景中自主移动 manipulator 的协同装配 

**Authors**: Davide Torielli  

**Link**: [PDF](https://arxiv.org/pdf/2505.07441)  

**Abstract**: [...] Specifically, the problem addressed is an assembly one known as the peg-in-hole task. In this case, two autonomous manipulators must carry cooperatively (at kinematic level) a peg and must insert it into an hole fixed in the environment. Even if the peg-in-hole is a well-known problem, there are no specific studies related to the use of two different autonomous manipulators, especially in underwater scenarios. Among all the possible investigations towards the problem, this work focuses mainly on the kinematic control of the robots. The methods used are part of the Task Priority Inverse Kinematics (TPIK) approach, with a cooperation scheme that permits to exchange as less information as possible between the agents (that is really important being water a big impediment for communication). A force-torque sensor is exploited at kinematic level to help the insertion phase. The results show how the TPIK and the chosen cooperation scheme can be used for the stated problem. The simulated experiments done consider little errors in the hole's pose, that still permit to insert the peg but with a lot of frictions and possible stucks. It is shown how can be possible to improve (thanks to the data provided by the force-torque sensor) the insertion phase performed by the two manipulators in presence of these errors. [...] 

**Abstract (ZH)**: 具体的装配问题是针孔装配任务。在此任务中，两个自主机械臂必须在运动学层面合作拿起一个针并将其插入固定在环境中的孔中。尽管针孔装配是一个众所周知的问题，但在使用两种不同自主机械臂方面几乎没有具体研究，特别是在水下场景中。在这项工作涉及的所有可能的研究中，主要集中在机器人运动学控制方面。所采用的方法属于任务优先逆运动学（TPIK）方法，利用的合作方案尽量减少信息交换（由于水对通信的阻碍作用，这一点非常重要）。在运动学层面使用力-扭矩传感器来辅助插入阶段。结果显示，TPIK方法和选择的合作方案可以用于解决上述问题。模拟实验中考虑到孔的姿态存在小误差，但仍能插入针，但由于摩擦和可能的卡滞现象较为严重。展示了在这些误差存在的情况下，如何利用力-扭矩传感器提供的数据来改进两个机械臂的插入阶段。 

---
# ReinboT: Amplifying Robot Visual-Language Manipulation with Reinforcement Learning 

**Title (ZH)**: ReinboT: 用强化学习增强机器人视觉-语言操作 

**Authors**: Hongyin Zhang, Zifeng Zhuang, Han Zhao, Pengxiang Ding, Hongchao Lu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07395)  

**Abstract**: Vision-Language-Action (VLA) models have shown great potential in general robotic decision-making tasks via imitation learning. However, the variable quality of training data often constrains the performance of these models. On the other hand, offline Reinforcement Learning (RL) excels at learning robust policy models from mixed-quality data. In this paper, we introduce Reinforced robot GPT (ReinboT), a novel end-to-end VLA model that integrates the RL principle of maximizing cumulative reward. ReinboT achieves a deeper understanding of the data quality distribution by predicting dense returns that capture the nuances of manipulation tasks. The dense return prediction capability enables the robot to generate more robust decision-making actions, oriented towards maximizing future benefits. Extensive experiments show that ReinboT achieves state-of-the-art performance on the CALVIN mixed-quality dataset and exhibits superior few-shot learning and out-of-distribution generalization capabilities in real-world tasks. 

**Abstract (ZH)**: 基于强化学习的视觉-语言-动作模型ReinboT：一种通过预测密集回报实现高效鲁棒决策的新范式 

---
# Stiffness-based Analytic Centre Method for Cable-Driven Parallel Robots 

**Title (ZH)**: 基于刚度的分析中心方法在缆索驱动并联机器人中的应用 

**Authors**: Domenico Dona', Vincenzo Di Paola, Matteo Zoppi, Alberto Trevisani  

**Link**: [PDF](https://arxiv.org/pdf/2505.07348)  

**Abstract**: Nowadays, being fast and precise are key requirements in Robotics. This work introduces a novel methodology to tune the stiffness of Cable-Driven Parallel Robots (CDPRs) while simultaneously addressing the tension distribution problem. In particular, the approach relies on the Analytic-Centre method. Indeed, weighting the barrier functions makes natural the stiffness adaptation. The intrinsic ability to adjust the stiffness during the execution of the task enables the CDPRs to effectively meet above-mentioned requirements. The capabilities of the method are demonstrated through simulations by comparing it with the existing approach. 

**Abstract (ZH)**: 如今，快速性和精确性是机器人领域的关键要求。本文提出了一种新颖的方法，用于在同时解决张力分配问题的情况下调整缆索驱动并联机器人（CDPR）的 stiffness，并特别依赖于分析中心方法。该方法的固有能力可以在执行任务时调整 stiffness，从而使CDPR能够有效满足上述要求。通过与现有方法的仿真比较，展示了该方法的能力。 

---
# Drive Fast, Learn Faster: On-Board RL for High Performance Autonomous Racing 

**Title (ZH)**: 快速驾驶，更快学习：车载强化学习在高性能自动驾驶赛车中的应用 

**Authors**: Benedict Hildisch, Edoardo Ghignone, Nicolas Baumann, Cheng Hu, Andrea Carron, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2505.07321)  

**Abstract**: Autonomous racing presents unique challenges due to its non-linear dynamics, the high speed involved, and the critical need for real-time decision-making under dynamic and unpredictable conditions. Most traditional Reinforcement Learning (RL) approaches rely on extensive simulation-based pre-training, which faces crucial challenges in transfer effectively to real-world environments. This paper introduces a robust on-board RL framework for autonomous racing, designed to eliminate the dependency on simulation-based pre-training enabling direct real-world adaptation. The proposed system introduces a refined Soft Actor-Critic (SAC) algorithm, leveraging a residual RL structure to enhance classical controllers in real-time by integrating multi-step Temporal-Difference (TD) learning, an asynchronous training pipeline, and Heuristic Delayed Reward Adjustment (HDRA) to improve sample efficiency and training stability. The framework is validated through extensive experiments on the F1TENTH racing platform, where the residual RL controller consistently outperforms the baseline controllers and achieves up to an 11.5 % reduction in lap times compared to the State-of-the-Art (SotA) with only 20 min of training. Additionally, an End-to-End (E2E) RL controller trained without a baseline controller surpasses the previous best results with sustained on-track learning. These findings position the framework as a robust solution for high-performance autonomous racing and a promising direction for other real-time, dynamic autonomous systems. 

**Abstract (ZH)**: 自主赛车比赛由于其非线性动力学、高速度以及在动态和不可预测条件下对实时决策的高需求，面临独特的挑战。大多数传统的强化学习（RL）方法依赖于基于仿真的预训练，但在将仿真实验结果有效转移到真实环境方面存在重大挑战。本文介绍了一种鲁棒的车载RL框架，旨在消除对基于仿真的预训练的依赖，使系统能够直接适应真实世界环境。该系统引入了一种改进的Soft Actor-Critic（SAC）算法，利用残差RL结构，通过集成多步时差（TD）学习、异步训练管道和启发式延迟奖励调整（HDRA），增强了实时控制性能，提高了样本效率并增强了训练稳定性。该框架通过在F1TENTH赛车平台上进行广泛实验得到验证，实验证明残差RL控制器在性能上持续优于基线控制器，并且仅通过20分钟的训练即可达到比当前最佳技术水平（SotA）快11.5%的圈速。此外，无需基线控制器的端到端（E2E）RL控制器在持续赛道学习中取得了超越以往最佳结果的表现。这些发现将该框架定位为高性能自主赛车的强大解决方案，并为其他实时动态自主系统提供了有前景的方向。 

---
# Autonomous Robotic Pruning in Orchards and Vineyards: a Review 

**Title (ZH)**: 果园和葡萄园中的自主机器人修剪：一篇综述 

**Authors**: Alessandro Navone, Mauro Martini, Marcello Chiaberge  

**Link**: [PDF](https://arxiv.org/pdf/2505.07318)  

**Abstract**: Manual pruning is labor intensive and represents up to 25% of annual labor costs in fruit production, notably in apple orchards and vineyards where operational challenges and cost constraints limit the adoption of large-scale machinery. In response, a growing body of research is investigating compact, flexible robotic platforms capable of precise pruning in varied terrains, particularly where traditional mechanization falls short.
This paper reviews recent advances in autonomous robotic pruning for orchards and vineyards, addressing a critical need in precision agriculture. Our review examines literature published between 2014 and 2024, focusing on innovative contributions across key system components. Special attention is given to recent developments in machine vision, perception, plant skeletonization, and control strategies, areas that have experienced significant influence from advancements in artificial intelligence and machine learning. The analysis situates these technological trends within broader agricultural challenges, including rising labor costs, a decline in the number of young farmers, and the diverse pruning requirements of different fruit species such as apple, grapevine, and cherry trees.
By comparing various robotic architectures and methodologies, this survey not only highlights the progress made toward autonomous pruning but also identifies critical open challenges and future research directions. The findings underscore the potential of robotic systems to bridge the gap between manual and mechanized operations, paving the way for more efficient, sustainable, and precise agricultural practices. 

**Abstract (ZH)**: 手工修剪劳动密集且占到果蔬生产年度劳动力成本的25%以上，特别是在苹果园和葡萄园中，由于操作挑战和成本限制，大规模机械的采用受到限制。为此，越来越多的研究开始探索紧凑灵活的机器人平台，能够精确地在各种地形上进行修剪，尤其是在传统机械化难以发挥作用的情况下。本文回顾了2014年至2024年间果园和葡萄园自动修剪的最新进展，重点关注精密农业中的关键系统组件的创新贡献。特别关注了机器视觉、感知、植物骨架化以及控制策略等领域的最新发展，这些领域受到了人工智能和机器学习 advances的显著影响。分析将这些技术趋势置于更广泛的农业挑战之中，包括劳动力成本上升、年轻农民数量下降以及不同果树种类（如苹果树、葡萄藤和樱桃树）多样化的修剪要求。通过对比各种机器人的架构和方法，本文不仅突出了自动修剪进展，还指出了关键的开放挑战和未来研究方向。研究结果强调了机器人系统在手工和机械化操作之间架起桥梁的潜力，为更高效、可持续和精确的农业实践铺平了道路。 

---
# HuB: Learning Extreme Humanoid Balance 

**Title (ZH)**: HuB: 学习极端人形机器人平衡 

**Authors**: Tong Zhang, Boyuan Zheng, Ruiqian Nai, Yingdong Hu, Yen-Jen Wang, Geng Chen, Fanqi Lin, Jiongye Li, Chuye Hong, Koushil Sreenath, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.07294)  

**Abstract**: The human body demonstrates exceptional motor capabilities-such as standing steadily on one foot or performing a high kick with the leg raised over 1.5 meters-both requiring precise balance control. While recent research on humanoid control has leveraged reinforcement learning to track human motions for skill acquisition, applying this paradigm to balance-intensive tasks remains challenging. In this work, we identify three key obstacles: instability from reference motion errors, learning difficulties due to morphological mismatch, and the sim-to-real gap caused by sensor noise and unmodeled dynamics. To address these challenges, we propose HuB (Humanoid Balance), a unified framework that integrates reference motion refinement, balance-aware policy learning, and sim-to-real robustness training, with each component targeting a specific challenge. We validate our approach on the Unitree G1 humanoid robot across challenging quasi-static balance tasks, including extreme single-legged poses such as Swallow Balance and Bruce Lee's Kick. Our policy remains stable even under strong physical disturbances-such as a forceful soccer strike-while baseline methods consistently fail to complete these tasks. Project website: this https URL 

**Abstract (ZH)**: 人类身体展现了卓越的运动能力——例如单脚站立或踢腿超过1.5米的高踢动作，都要求精准的平衡控制。虽然最近的人形机器人控制研究利用强化学习跟踪人类运动以获取技能，但将其应用于平衡密集型任务仍然颇具挑战。在此工作中，我们识别出三个关键障碍：参考运动错误导致的不稳定、由于形态不匹配导致的学习困难、以及由于传感器噪声和未建模动态导致的仿真到现实的差距。为应对这些挑战，我们提出了一种统一框架HuB（人形机器人平衡），该框架整合了参考运动精细调整、平衡感知策略学习以及仿真到现实的鲁棒性训练，每个组成部分都针对特定的挑战。我们在Unitree G1人形机器人上对极具挑战性的准静态平衡任务进行了验证，包括极端单腿姿势如燕式平衡和李小龙踢腿。即使在强烈的物理干扰（如足球猛烈击打）下，我们的策略仍然保持稳定，而基线方法却始终无法完成这些任务。项目网站：这个 https://URL。 

---
# BETTY Dataset: A Multi-modal Dataset for Full-Stack Autonomy 

**Title (ZH)**: 贝蒂数据集：全栈自主性的多模态数据集 

**Authors**: Micah Nye, Ayoub Raji, Andrew Saba, Eidan Erlich, Robert Exley, Aragya Goyal, Alexander Matros, Ritesh Misra, Matthew Sivaprakasam, Marko Bertogna, Deva Ramanan, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2505.07266)  

**Abstract**: We present the BETTY dataset, a large-scale, multi-modal dataset collected on several autonomous racing vehicles, targeting supervised and self-supervised state estimation, dynamics modeling, motion forecasting, perception, and more. Existing large-scale datasets, especially autonomous vehicle datasets, focus primarily on supervised perception, planning, and motion forecasting tasks. Our work enables multi-modal, data-driven methods by including all sensor inputs and the outputs from the software stack, along with semantic metadata and ground truth information. The dataset encompasses 4 years of data, currently comprising over 13 hours and 32TB, collected on autonomous racing vehicle platforms. This data spans 6 diverse racing environments, including high-speed oval courses, for single and multi-agent algorithm evaluation in feature-sparse scenarios, as well as high-speed road courses with high longitudinal and lateral accelerations and tight, GPS-denied environments. It captures highly dynamic states, such as 63 m/s crashes, loss of tire traction, and operation at the limit of stability. By offering a large breadth of cross-modal and dynamic data, the BETTY dataset enables the training and testing of full autonomy stack pipelines, pushing the performance of all algorithms to the limits. The current dataset is available at this https URL. 

**Abstract (ZH)**: BETTY数据集：一种多模态的自主赛车大数据集，用于监督和自监督状态估计、动力学建模、运动预测、感知等任务 

---
# CHD: Coupled Hierarchical Diffusion for Long-Horizon Tasks 

**Title (ZH)**: CHD: 耦合分级扩散模型用于长Horizon任务 

**Authors**: Ce Hao, Anxing Xiao, Zhiwei Xue, Harold Soh  

**Link**: [PDF](https://arxiv.org/pdf/2505.07261)  

**Abstract**: Diffusion-based planners have shown strong performance in short-horizon tasks but often fail in complex, long-horizon settings. We trace the failure to loose coupling between high-level (HL) sub-goal selection and low-level (LL) trajectory generation, which leads to incoherent plans and degraded performance. We propose Coupled Hierarchical Diffusion (CHD), a framework that models HL sub-goals and LL trajectories jointly within a unified diffusion process. A shared classifier passes LL feedback upstream so that sub-goals self-correct while sampling proceeds. This tight HL-LL coupling improves trajectory coherence and enables scalable long-horizon diffusion planning. Experiments across maze navigation, tabletop manipulation, and household environments show that CHD consistently outperforms both flat and hierarchical diffusion baselines. 

**Abstract (ZH)**: 基于扩散的方法在短_horizon任务中表现出强劲性能，但在复杂、长_horizon设置中经常失效。我们追踪到这一失败源于高层（HL）子目标选择与低层（LL）轨迹生成之间的松散耦合，导致计划不连贯且性能下降。我们提出了一种耦合层次扩散（CHD）框架，该框架在统一的扩散过程中同时建模高层子目标和低层轨迹。共享分类器将低层反馈传递至上层，使子目标在采样过程中自我纠正。这种紧密的HL-LL耦合提高了轨迹的连贯性，并使长_horizon扩散规划可扩展。实验结果在迷宫导航、台面操作以及家庭环境场景中表明，CHD 一致地优于平面和层次扩散基线。 

---
# A Framework for Joint Grasp and Motion Planning in Confined Spaces 

**Title (ZH)**: 受限空间内抓取与运动规划的框架 

**Authors**: Martin Rudorfer, Jiří Hartvich, Vojtěch Vonásek  

**Link**: [PDF](https://arxiv.org/pdf/2505.07259)  

**Abstract**: Robotic grasping is a fundamental skill across all domains of robot applications. There is a large body of research for grasping objects in table-top scenarios, where finding suitable grasps is the main challenge. In this work, we are interested in scenarios where the objects are in confined spaces and hence particularly difficult to reach. Planning how the robot approaches the object becomes a major part of the challenge, giving rise to methods for joint grasp and motion planning. The framework proposed in this paper provides 20 benchmark scenarios with systematically increasing difficulty, realistic objects with precomputed grasp annotations, and tools to create and share more scenarios. We further provide two baseline planners and evaluate them on the scenarios, demonstrating that the proposed difficulty levels indeed offer a meaningful progression. We invite the research community to build upon this framework by making all components publicly available as open source. 

**Abstract (ZH)**: 机器人抓取是机器人应用领域的一项基本技能。在桌面场景中进行物体抓取的研究成果丰富，其中寻找到合适的抓取方式是主要挑战。本文关注物体位于受限空间中的场景，这类场景下抓取物体尤其困难。规划机器人如何接近物体成为主要挑战之一，推动了抓取与运动规划的联合方法的发展。本文提出的研究框架提供了20个具有系统递增难度的基准场景、真实物体及其预计算的抓取标注，并提供工具以创建和共享更多场景。此外，我们还提供了两种基线规划算法，并在这些场景上进行了评估，验证了提出的难度级别确实具有意义。我们邀请研究社区在此框架基础上进行研究，并将所有组件公开为开源软件。 

---
# UAV-CodeAgents: Scalable UAV Mission Planning via Multi-Agent ReAct and Vision-Language Reasoning 

**Title (ZH)**: UAV-CodeAgents: 通过多agents ReAct 和 视觉语言推理实现可扩展的无人机任务规划 

**Authors**: Oleg Sautenkov, Yasheerah Yaqoot, Muhammad Ahsan Mustafa, Faryal Batool, Jeffrin Sam, Artem Lykov, Chih-Yung Wen, Dzmitry Tsetserukou  

**Link**: [PDF](https://arxiv.org/pdf/2505.07236)  

**Abstract**: We present UAV-CodeAgents, a scalable multi-agent framework for autonomous UAV mission generation, built on large language and vision-language models (LLMs/VLMs). The system leverages the ReAct (Reason + Act) paradigm to interpret satellite imagery, ground high-level natural language instructions, and collaboratively generate UAV trajectories with minimal human supervision. A core component is a vision-grounded, pixel-pointing mechanism that enables precise localization of semantic targets on aerial maps. To support real-time adaptability, we introduce a reactive thinking loop, allowing agents to iteratively reflect on observations, revise mission goals, and coordinate dynamically in evolving environments.
UAV-CodeAgents is evaluated on large-scale mission scenarios involving industrial and environmental fire detection. Our results show that a lower decoding temperature (0.5) yields higher planning reliability and reduced execution time, with an average mission creation time of 96.96 seconds and a success rate of 93%. We further fine-tune Qwen2.5VL-7B on 9,000 annotated satellite images, achieving strong spatial grounding across diverse visual categories. To foster reproducibility and future research, we will release the full codebase and a novel benchmark dataset for vision-language-based UAV planning. 

**Abstract (ZH)**: UAV-CodeAgents：基于大语言和 vision-language 模型的可扩展多agent自主无人机任务生成框架 

---
# Terrain-aware Low Altitude Path Planning 

**Title (ZH)**: 地形-aware 低altura路径规划 

**Authors**: Yixuan Jia, Andrea Tagliabue, Navid Dadkhah Tehrani, Jonathan P. How  

**Link**: [PDF](https://arxiv.org/pdf/2505.07141)  

**Abstract**: In this paper, we study the problem of generating low altitude path plans for nap-of-the-earth (NOE) flight in real time with only RGB images from onboard cameras and the vehicle pose. We propose a novel training method that combines behavior cloning and self-supervised learning that enables the learned policy to outperform the policy trained with standard behavior cloning approach on this task. Simulation studies are performed on a custom canyon terrain. 

**Abstract (ZH)**: 本文研究了仅使用机载摄像头的RGB图像和车辆姿态在实时生成低 altitude 贴地飞行路径规划(NOE)问题的方法。我们提出了一种结合行为克隆和自我监督学习的新型训练方法，使得学习到的策略在这一任务上能超越使用标准行为克隆方法训练的策略。在自定义 Canyon 地形上进行了仿真研究。 

---
# X-Sim: Cross-Embodiment Learning via Real-to-Sim-to-Real 

**Title (ZH)**: X-Sim: 从实到虚再到实的跨躯体学习 

**Authors**: Prithwish Dan, Kushal Kedia, Angela Chao, Edward Weiyi Duan, Maximus Adrian Pace, Wei-Chiu Ma, Sanjiban Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.07096)  

**Abstract**: Human videos offer a scalable way to train robot manipulation policies, but lack the action labels needed by standard imitation learning algorithms. Existing cross-embodiment approaches try to map human motion to robot actions, but often fail when the embodiments differ significantly. We propose X-Sim, a real-to-sim-to-real framework that uses object motion as a dense and transferable signal for learning robot policies. X-Sim starts by reconstructing a photorealistic simulation from an RGBD human video and tracking object trajectories to define object-centric rewards. These rewards are used to train a reinforcement learning (RL) policy in simulation. The learned policy is then distilled into an image-conditioned diffusion policy using synthetic rollouts rendered with varied viewpoints and lighting. To transfer to the real world, X-Si introduces an online domain adaptation technique that aligns real and simulated observations during deployment. Importantly, X-Sim does not require any robot teleoperation data. We evaluate it across 5 manipulation tasks in 2 environments and show that it: (1) improves task progress by 30% on average over hand-tracking and sim-to-real baselines, (2) matches behavior cloning with 10x less data collection time, and (3) generalizes to new camera viewpoints and test-time changes. Code and videos are available at this https URL. 

**Abstract (ZH)**: 人类视频提供了训练机器人操作策略的一种可扩展方式，但缺乏标准拟合学习算法所需的动作标签。现有的跨体素方法尝试将人类运动映射到机器人动作，但在体素差异较大时常常失败。我们提出了一种X-Sim框架，该框架通过对象运动作为密集且可转移的信号来学习机器人策略。X-Sim首先从RGBD人类视频重建一个拟真的真实模拟场景，并追踪对象轨迹以定义对象中心奖励。这些奖励用于在仿真环境中训练强化学习（RL）策略。学会的策略随后使用不同视点和照明渲染的合成滚动生成指令扩散策略进行提炼。为了在真实世界中转移，X-Si引入了一种在线领域适应技术，在部署期间对真实和模拟观察进行对齐。重要的是，X-Sim不需要任何机器人遥操作数据。我们在2个环境中5个操作任务上对其进行评估，结果显示：（1）与手部跟踪和仿真到现实基准相比，平均改善任务进度30%；（2）用十分之一的数据收集时间匹配行为克隆；（3）能够泛化到新的摄像机视点和测试时间的变化。代码和视频可在以下链接访问。 

---
# DriveSOTIF: Advancing Perception SOTIF Through Multimodal Large Language Models 

**Title (ZH)**: DriveSOTIF：通过多模态大语言模型促进感知SOTIF的研究 

**Authors**: Shucheng Huang, Freda Shi, Chen Sun, Jiaming Zhong, Minghao Ning, Yufeng Yang, Yukun Lu, Hong Wang, Amir Khajepour  

**Link**: [PDF](https://arxiv.org/pdf/2505.07084)  

**Abstract**: Human drivers naturally possess the ability to perceive driving scenarios, predict potential hazards, and react instinctively due to their spatial and causal intelligence, which allows them to perceive, understand, predict, and interact with the 3D world both spatially and temporally. Autonomous vehicles, however, lack these capabilities, leading to challenges in effectively managing perception-related Safety of the Intended Functionality (SOTIF) risks, particularly in complex and unpredictable driving conditions. To address this gap, we propose an approach that fine-tunes multimodal language models (MLLMs) on a customized dataset specifically designed to capture perception-related SOTIF scenarios. Model benchmarking demonstrates that this tailored dataset enables the models to better understand and respond to these complex driving situations. Additionally, in real-world case studies, the proposed method correctly handles challenging scenarios that even human drivers may find difficult. Real-time performance tests further indicate the potential for the models to operate efficiently in live driving environments. This approach, along with the dataset generation pipeline, shows significant promise for improving the identification, cognition, prediction, and reaction to SOTIF-related risks in autonomous driving systems. The dataset and information are available: this https URL 

**Abstract (ZH)**: 人类驾驶员天生具备感知驾驶场景、预测潜在风险并本能反应的能力，这得益于他们的空间和因果智能，使他们能够从空间和时间上感知、理解、预测和交互3D世界。然而，自动驾驶车辆缺乏这些能力，导致在复杂和不可预测的驾驶条件下难以有效管理感知相关的意图功能安全（SOTIF）风险。为解决这一问题，我们提出了一种方法，即在为捕捉感知相关的SOTIF场景而定制的数据集上细调多模态语言模型（MLLMs）。模型基准测试表明，这种定制数据集使模型能够更好地理解和应对这些复杂的驾驶情况。此外，在实际案例研究中，所提议的方法正确处理了即使是人类驾驶员也可能难以应对的挑战性场景。实时性能测试进一步表明，这些模型有可能在实际驾驶环境中高效运行。该方法结合数据集生成管道，在自动驾驶系统中显著提高了对SOTIF相关风险的识别、认知、预测和反应能力。数据集和信息可在以下链接获取：this https URL。 

---
# VALISENS: A Validated Innovative Multi-Sensor System for Cooperative Automated Driving 

**Title (ZH)**: VALISENS：一个验证的创新多传感器系统用于协同自动驾驶 

**Authors**: Lei Wan, Prabesh Gupta, Andreas Eich, Marcel Kettelgerdes, Hannan Ejaz Keen, Michael Klöppel-Gersdorf, Alexey Vinel  

**Link**: [PDF](https://arxiv.org/pdf/2505.06980)  

**Abstract**: Perception is a core capability of automated vehicles and has been significantly advanced through modern sensor technologies and artificial intelligence. However, perception systems still face challenges in complex real-world scenarios. To improve robustness against various external factors, multi-sensor fusion techniques are essential, combining the strengths of different sensor modalities. With recent developments in Vehicle-to-Everything (V2X communication, sensor fusion can now extend beyond a single vehicle to a cooperative multi-agent system involving Connected Automated Vehicle (CAV) and intelligent infrastructure. This paper presents VALISENS, an innovative multi-sensor system distributed across multiple agents. It integrates onboard and roadside LiDARs, radars, thermal cameras, and RGB cameras to enhance situational awareness and support cooperative automated driving. The thermal camera adds critical redundancy for perceiving Vulnerable Road User (VRU), while fusion with roadside sensors mitigates visual occlusions and extends the perception range beyond the limits of individual vehicles. We introduce the corresponding perception module built on this sensor system, which includes object detection, tracking, motion forecasting, and high-level data fusion. The proposed system demonstrates the potential of cooperative perception in real-world test environments and lays the groundwork for future Cooperative Intelligent Transport Systems (C-ITS) applications. 

**Abstract (ZH)**: 多代理分布式多传感器系统VALISENS：面向智能互联车辆的协同感知 

---
# Reinforcement Learning-Based Monocular Vision Approach for Autonomous UAV Landing 

**Title (ZH)**: 基于强化学习的单目视觉自主无人机着陆方法 

**Authors**: Tarik Houichime, Younes EL Amrani  

**Link**: [PDF](https://arxiv.org/pdf/2505.06963)  

**Abstract**: This paper introduces an innovative approach for the autonomous landing of Unmanned Aerial Vehicles (UAVs) using only a front-facing monocular camera, therefore obviating the requirement for depth estimation cameras. Drawing on the inherent human estimating process, the proposed method reframes the landing task as an optimization problem. The UAV employs variations in the visual characteristics of a specially designed lenticular circle on the landing pad, where the perceived color and form provide critical information for estimating both altitude and depth. Reinforcement learning algorithms are utilized to approximate the functions governing these estimations, enabling the UAV to ascertain ideal landing settings via training. This method's efficacy is assessed by simulations and experiments, showcasing its potential for robust and accurate autonomous landing without dependence on complex sensor setups. This research contributes to the advancement of cost-effective and efficient UAV landing solutions, paving the way for wider applicability across various fields. 

**Abstract (ZH)**: 本文介绍了一种仅使用前方单目摄像头进行自主降落的无人机创新方法，从而无需深度估计摄像头。该方法借鉴了人类固有的估算过程，将降落任务重新定义为一个优化问题。无人机通过检测特殊设计的透镜圆圈在着陆垫上视觉特征的变化，如感知的颜色和形态，来获取高度和深度的关键信息。利用强化学习算法近似这些估算函数，使无人机能够通过训练确定理想的降落设置。该方法的有效性通过模拟和实验来评估，展示了其在无需复杂传感器配置的情况下实现稳健且精确的自主降落的潜力。本文为低成本高效的无人机降落解决方案的进步做出了贡献，为其在各个领域的广泛应用奠定了基础。 

---
# YOPOv2-Tracker: An End-to-End Agile Tracking and Navigation Framework from Perception to Action 

**Title (ZH)**: YOPOv2-Tracker：从感知到行动的端到端敏捷跟踪与导航框架 

**Authors**: Junjie Lu, Yulin Hui, Xuewei Zhang, Wencan Feng, Hongming Shen, Zhiyu Li, Bailing Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.06923)  

**Abstract**: Traditional target tracking pipelines including detection, mapping, navigation, and control are comprehensive but introduce high latency, limitting the agility of quadrotors. On the contrary, we follow the design principle of "less is more", striving to simplify the process while maintaining effectiveness. In this work, we propose an end-to-end agile tracking and navigation framework for quadrotors that directly maps the sensory observations to control commands. Importantly, leveraging the multimodal nature of navigation and detection tasks, our network maintains interpretability by explicitly integrating the independent modules of the traditional pipeline, rather than a crude action regression. In detail, we adopt a set of motion primitives as anchors to cover the searching space regarding the feasible region and potential target. Then we reformulate the trajectory optimization as regression of primitive offsets and associated costs considering the safety, smoothness, and other metrics. For tracking task, the trajectories are expected to approach the target and additional objectness scores are predicted. Subsequently, the predictions, after compensation for the estimated lumped disturbance, are transformed into thrust and attitude as control commands for swift response. During training, we seamlessly integrate traditional motion planning with deep learning by directly back-propagating the gradients of trajectory costs to the network, eliminating the need for expert demonstration in imitation learning and providing more direct guidance than reinforcement learning. Finally, we deploy the algorithm on a compact quadrotor and conduct real-world validations in both forest and building environments to demonstrate the efficiency of the proposed method. 

**Abstract (ZH)**: 一种基于端到端优化的四旋翼敏捷跟踪与导航框架 

---
# The First WARA Robotics Mobile Manipulation Challenge -- Lessons Learned 

**Title (ZH)**: 首次WARA机器人移动操作挑战——经验教训 

**Authors**: David Cáceres Domínguez, Marco Iannotta, Abhishek Kashyap, Shuo Sun, Yuxuan Yang, Christian Cella, Matteo Colombo, Martina Pelosi, Giuseppe F. Preziosa, Alessandra Tafuro, Isacco Zappa, Finn Busch, Yifei Dong, Alberta Longhini, Haofei Lu, Rafael I. Cabral Muchacho, Jonathan Styrud, Sebastiano Fregnan, Marko Guberina, Zheng Jia, Graziano Carriero, Sofia Lindqvist, Silvio Di Castro, Matteo Iovino  

**Link**: [PDF](https://arxiv.org/pdf/2505.06919)  

**Abstract**: The first WARA Robotics Mobile Manipulation Challenge, held in December 2024 at ABB Corporate Research in Västerås, Sweden, addressed the automation of task-intensive and repetitive manual labor in laboratory environments - specifically the transport and cleaning of glassware. Designed in collaboration with AstraZeneca, the challenge invited academic teams to develop autonomous robotic systems capable of navigating human-populated lab spaces and performing complex manipulation tasks, such as loading items into industrial dishwashers. This paper presents an overview of the challenge setup, its industrial motivation, and the four distinct approaches proposed by the participating teams. We summarize lessons learned from this edition and propose improvements in design to enable a more effective second iteration to take place in 2025. The initiative bridges an important gap in effective academia-industry collaboration within the domain of autonomous mobile manipulation systems by promoting the development and deployment of applied robotic solutions in real-world laboratory contexts. 

**Abstract (ZH)**: 第一次WARA Robotics移动 manipulation挑战赛：2024年12月在瑞典韦斯特罗斯ABB企业研究院举行的挑战赛，旨在解决实验室环境中密集型和重复性的手工劳动的自动化问题——尤其是玻璃器皿的运输和清洁。该挑战由AstraZeneca合作设计，邀请学术团队开发能够在有人类操作员的实验室空间内自主导航并执行复杂操作任务（如将物品加载到工业洗碗机中）的机器人系统。本文概述了挑战设置、其工业背景以及参赛团队提出的四个不同方法。我们总结了本版次的经验教训，并提出了改进设计的建议，以使2025年的第二次迭代更加有效。该倡议通过促进适用于真实实验室环境的自主移动 manipulation系统的开发和部署，填补了学术界与工业界有效合作的重要空白。 

---
# Realistic Counterfactual Explanations for Machine Learning-Controlled Mobile Robots using 2D LiDAR 

**Title (ZH)**: 基于2D LiDAR的机器学习控制移动机器人现实-counterfactual 解释 

**Authors**: Sindre Benjamin Remman, Anastasios M. Lekkas  

**Link**: [PDF](https://arxiv.org/pdf/2505.06906)  

**Abstract**: This paper presents a novel method for generating realistic counterfactual explanations (CFEs) in machine learning (ML)-based control for mobile robots using 2D LiDAR. ML models, especially artificial neural networks (ANNs), can provide advanced decision-making and control capabilities by learning from data. However, they often function as black boxes, making it challenging to interpret them. This is especially a problem in safety-critical control applications. To generate realistic CFEs, we parameterize the LiDAR space with simple shapes such as circles and rectangles, whose parameters are chosen by a genetic algorithm, and the configurations are transformed into LiDAR data by raycasting. Our model-agnostic approach generates CFEs in the form of synthetic LiDAR data that resembles a base LiDAR state but is modified to produce a pre-defined ML model control output based on a query from the user. We demonstrate our method on a mobile robot, the TurtleBot3, controlled using deep reinforcement learning (DRL) in real-world and simulated scenarios. Our method generates logical and realistic CFEs, which helps to interpret the DRL agent's decision making. This paper contributes towards advancing explainable AI in mobile robotics, and our method could be a tool for understanding, debugging, and improving ML-based autonomous control. 

**Abstract (ZH)**: 本文提出了一种使用2D LiDAR在基于机器学习的移动机器人控制中生成现实主义反事实解释的新方法。 

---
# FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots 

**Title (ZH)**: FACET：基于阻抗参考跟踪的力自适应控制方法用于腿式机器人 

**Authors**: Botian Xu, Haoyang Weng, Qingzhou Lu, Yang Gao, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06883)  

**Abstract**: Reinforcement learning (RL) has made significant strides in legged robot control, enabling locomotion across diverse terrains and complex loco-manipulation capabilities. However, the commonly used position or velocity tracking-based objectives are agnostic to forces experienced by the robot, leading to stiff and potentially dangerous behaviors and poor control during forceful interactions. To address this limitation, we present \emph{Force-Adaptive Control via Impedance Reference Tracking} (FACET). Inspired by impedance control, we use RL to train a control policy to imitate a virtual mass-spring-damper system, allowing fine-grained control under external forces by manipulating the virtual spring. In simulation, we demonstrate that our quadruped robot achieves improved robustness to large impulses (up to 200 Ns) and exhibits controllable compliance, achieving an 80% reduction in collision impulse. The policy is deployed to a physical robot to showcase both compliance and the ability to engage with large forces by kinesthetic control and pulling payloads up to 2/3 of its weight. Further extension to a legged loco-manipulator and a humanoid shows the applicability of our method to more complex settings to enable whole-body compliance control. Project Website: this https URL 

**Abstract (ZH)**: 基于阻抗参考跟踪的力自适应控制（Force-Adaptive Control via Impedance Reference Tracking, FACET） 

---
# Towards Human-Centric Autonomous Driving: A Fast-Slow Architecture Integrating Large Language Model Guidance with Reinforcement Learning 

**Title (ZH)**: 面向以人为中心的自主驾驶：一种结合大型语言模型指导与 reinforcement 学习的快慢架构 

**Authors**: Chengkai Xu, Jiaqi Liu, Yicheng Guo, Yuhang Zhang, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.06875)  

**Abstract**: Autonomous driving has made significant strides through data-driven techniques, achieving robust performance in standardized tasks. However, existing methods frequently overlook user-specific preferences, offering limited scope for interaction and adaptation with users. To address these challenges, we propose a "fast-slow" decision-making framework that integrates a Large Language Model (LLM) for high-level instruction parsing with a Reinforcement Learning (RL) agent for low-level real-time decision. In this dual system, the LLM operates as the "slow" module, translating user directives into structured guidance, while the RL agent functions as the "fast" module, making time-critical maneuvers under stringent latency constraints. By decoupling high-level decision making from rapid control, our framework enables personalized user-centric operation while maintaining robust safety margins. Experimental evaluations across various driving scenarios demonstrate the effectiveness of our method. Compared to baseline algorithms, the proposed architecture not only reduces collision rates but also aligns driving behaviors more closely with user preferences, thereby achieving a human-centric mode. By integrating user guidance at the decision level and refining it with real-time control, our framework bridges the gap between individual passenger needs and the rigor required for safe, reliable driving in complex traffic environments. 

**Abstract (ZH)**: 基于大数据驱动技术的自动驾驶已取得显著进展，实现了标准化任务中的稳健性能。然而，现有方法往往忽略了用户的个性化偏好，限制了与用户的互动和适应性。为解决这些问题，我们提出了一种“快慢”决策框架，将大型语言模型（LLM）用于高层次指令解析，将强化学习（RL）代理用于低层次实时决策。在该双系统中，LLM 作为“慢”模块，将用户指令转换为结构化指导，而 RL 代理作为“快”模块，在严格的时间延迟约束条件下进行快速机动。通过将高层次决策与快速控制解耦，我们的框架能够实现个性化用户中心的操作，同时保持 robust 的安全边际。在多种驾驶场景下的实验评估证明了我们方法的有效性。与基准算法相比，所提出架构不仅降低了碰撞率，还使驾驶行为更加符合用户的偏好，从而实现以人为中心的模式。通过在决策级别整合用户指导并在实时控制中进行优化，我们的框架填补了个体乘客需求与复杂交通环境中安全可靠驾驶所需的严格要求之间的差距。 

---
# Efficient Robotic Policy Learning via Latent Space Backward Planning 

**Title (ZH)**: 通过潜在空间逆向规划实现高效的机器人策略学习 

**Authors**: Dongxiu Liu, Haoyi Niu, Zhihao Wang, Jinliang Zheng, Yinan Zheng, Zhonghong Ou, Jianming Hu, Jianxiong Li, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06861)  

**Abstract**: Current robotic planning methods often rely on predicting multi-frame images with full pixel details. While this fine-grained approach can serve as a generic world model, it introduces two significant challenges for downstream policy learning: substantial computational costs that hinder real-time deployment, and accumulated inaccuracies that can mislead action extraction. Planning with coarse-grained subgoals partially alleviates efficiency issues. However, their forward planning schemes can still result in off-task predictions due to accumulation errors, leading to misalignment with long-term goals. This raises a critical question: Can robotic planning be both efficient and accurate enough for real-time control in long-horizon, multi-stage tasks? To address this, we propose a Latent Space Backward Planning scheme (LBP), which begins by grounding the task into final latent goals, followed by recursively predicting intermediate subgoals closer to the current state. The grounded final goal enables backward subgoal planning to always remain aware of task completion, facilitating on-task prediction along the entire planning horizon. The subgoal-conditioned policy incorporates a learnable token to summarize the subgoal sequences and determines how each subgoal guides action extraction. Through extensive simulation and real-robot long-horizon experiments, we show that LBP outperforms existing fine-grained and forward planning methods, achieving SOTA performance. Project Page: this https URL 

**Abstract (ZH)**: 当前的机器人规划方法often rely on预测多帧图像的详细像素信息。虽然这种细粒度的方法可以作为通用世界模型，但它引入了两个重要的挑战：显著的计算成本阻碍了实时部署，并且累积的不准确性可能导致行为提取错误。使用粗粒度的子目标进行规划部分缓解了效率问题。然而，其前瞻规划方案仍然可能导致由于累积误差而导致的离任务预测，从而与长期目标产生偏差。这提出了一个关键问题：机器人规划是否可以在长时间多阶段任务中实现高效且足够的准确，以进行实时控制？为了解决这个问题，我们提出了一种潜在空间反向规划方案（LBP），该方案首先将任务细分为最终的潜在目标，然后递归预测更接近当前状态的中间子目标。接地的最终目标使反向子目标规划始终保持对任务完成的意识，有助于在整个规划时域内实现任务相关的预测。子目标条件策略结合了一个可学习的令牌来总结子目标序列，并确定每个子目标如何指导行为提取。通过广泛的仿真和真实的长时域机器人实验，我们表明LBP优于现有的细粒度和前瞻规划方法，达到了当前最优表现（SOTA）。项目页面: this https URL 

---
# Secure Safety Filter: Towards Safe Flight Control under Sensor Attacks 

**Title (ZH)**: 安全过滤器确保安全：面向传感器攻击下的飞行控制安全性研究 

**Authors**: Xiao Tan, Junior Sundar, Renzo Bruzzone, Pio Ong, Willian T. Lunardi, Martin Andreoni, Paulo Tabuada, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2505.06845)  

**Abstract**: Modern autopilot systems are prone to sensor attacks that can jeopardize flight safety. To mitigate this risk, we proposed a modular solution: the secure safety filter, which extends the well-established control barrier function (CBF)-based safety filter to account for, and mitigate, sensor attacks. This module consists of a secure state reconstructor (which generates plausible states) and a safety filter (which computes the safe control input that is closest to the nominal one). Differing from existing work focusing on linear, noise-free systems, the proposed secure safety filter handles bounded measurement noise and, by leveraging reduced-order model techniques, is applicable to the nonlinear dynamics of drones. Software-in-the-loop simulations and drone hardware experiments demonstrate the effectiveness of the secure safety filter in rendering the system safe in the presence of sensor attacks. 

**Abstract (ZH)**: 现代自动驾驶系统易受传感器攻击的影响，这可能危及飞行安全。为降低这一风险，我们提出了一种模块化解决方案：安全过滤器模块，它将基于控制障碍函数（CBF）的安全过滤器扩展以考虑和缓解传感器攻击。该模块包括安全状态重构器（生成合理的状态）和安全过滤器（计算最接近名义值的安全控制输入）。与现有工作主要针对线性、无噪声系统不同，所提出的安全过滤器能够处理有界测量噪声，并通过利用降阶模型技术适用于无人机的非线性动力学。软件在环仿真和无人机硬件实验表明，在存在传感器攻击的情况下，安全过滤器能够使系统保持安全。 

---
# UniDiffGrasp: A Unified Framework Integrating VLM Reasoning and VLM-Guided Part Diffusion for Open-Vocabulary Constrained Grasping with Dual Arms 

**Title (ZH)**: UniDiffGrasp: 一种集成VLM推理和VLM指导的部件扩散的统一框架，用于双臂开放词汇约束抓取 

**Authors**: Xueyang Guo, Hongwei Hu, Chengye Song, Jiale Chen, Zilin Zhao, Yu Fu, Bowen Guan, Zhenze Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06832)  

**Abstract**: Open-vocabulary, task-oriented grasping of specific functional parts, particularly with dual arms, remains a key challenge, as current Vision-Language Models (VLMs), while enhancing task understanding, often struggle with precise grasp generation within defined constraints and effective dual-arm coordination. We innovatively propose UniDiffGrasp, a unified framework integrating VLM reasoning with guided part diffusion to address these limitations. UniDiffGrasp leverages a VLM to interpret user input and identify semantic targets (object, part(s), mode), which are then grounded via open-vocabulary segmentation. Critically, the identified parts directly provide geometric constraints for a Constrained Grasp Diffusion Field (CGDF) using its Part-Guided Diffusion, enabling efficient, high-quality 6-DoF grasps without retraining. For dual-arm tasks, UniDiffGrasp defines distinct target regions, applies part-guided diffusion per arm, and selects stable cooperative grasps. Through extensive real-world deployment, UniDiffGrasp achieves grasp success rates of 0.876 in single-arm and 0.767 in dual-arm scenarios, significantly surpassing existing state-of-the-art methods, demonstrating its capability to enable precise and coordinated open-vocabulary grasping in complex real-world scenarios. 

**Abstract (ZH)**: 开放式词汇、任务导向的特定功能性部件抓取，尤其是双臂抓取，仍然是一个关键挑战，现有的视觉-语言模型虽然提升了任务理解能力，但往往在精确抓取生成和双臂有效协调方面存在困难。我们创新地提出了UniDiffGrasp，这是一种结合视觉-语言模型推理与部分导向扩散的统一框架，以解决这些问题。UniDiffGrasp利用视觉-语言模型解读用户输入并识别语义目标（对象、部件、模式），并通过开放词汇分割进行语义grounding。关键的是，识别出的部件直接为受部件导向扩散约束的抓取扩散场（CGDF）提供几何约束，从而在无需重新训练的情况下实现高效、高质量的6-DoF抓取。对于双臂任务，UniDiffGrasp定义了不同的目标区域，每臂应用部分导向扩散，并选择稳定的协同抓取。通过广泛的实地部署，UniDiffGrasp在单臂场景中达到了0.876的抓取成功率，在双臂场景中达到了0.767，显著超越了现有的先进方法，展示了其在复杂现实场景中实现精确且协调的开放词汇抓取的能力。 

---
# Dynamic Safety in Complex Environments: Synthesizing Safety Filters with Poisson's Equation 

**Title (ZH)**: 复杂环境中的动态安全：基于泊松方程合成安全过滤器 

**Authors**: Gilbert Bahati, Ryan M. Bena, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2505.06794)  

**Abstract**: Synthesizing safe sets for robotic systems operating in complex and dynamically changing environments is a challenging problem. Solving this problem can enable the construction of safety filters that guarantee safe control actions -- most notably by employing Control Barrier Functions (CBFs). This paper presents an algorithm for generating safe sets from perception data by leveraging elliptic partial differential equations, specifically Poisson's equation. Given a local occupancy map, we solve Poisson's equation subject to Dirichlet boundary conditions, with a novel forcing function. Specifically, we design a smooth guidance vector field, which encodes gradient information required for safety. The result is a variational problem for which the unique minimizer -- a safety function -- characterizes the safe set. After establishing our theoretical result, we illustrate how safety functions can be used in CBF-based safety filtering. The real-time utility of our synthesis method is highlighted through hardware demonstrations on quadruped and humanoid robots navigating dynamically changing obstacle-filled environments. 

**Abstract (ZH)**: 在复杂且动态变化环境中合成安全集是机器人系统面临的一个具有挑战性的问题。解决这一问题可以使得通过控制障碍函数（CBFs）等方式构建保证安全控制动作的安全过滤器成为可能。本文提出了一种利用椭圆偏微分方程（尤其是泊松方程）从感知数据生成安全集的算法。给定局部占用地图，我们通过Dirichlet边界条件求解泊松方程，并设计了一个新型的激励函数。具体地，我们设计了一个光滑的引导向量场，编码了用于安全性的梯度信息。结果是一个变分问题，其唯一的最小值解——安全函数——定义了安全集。在建立我们的理论结果后，本文展示了如何使用安全函数来进行基于CBF的安全过滤。通过在具有动态变化障碍物环境中的 quadruped 和类人机器人上的硬件演示，突显了我们合成方法的实时实用性。 

---
# cpRRTC: GPU-Parallel RRT-Connect for Constrained Motion Planning 

**Title (ZH)**: cpRRTC: GPU并行RRT-Connect算法在受约束运动规划中的应用 

**Authors**: Jiaming Hu, Jiawei Wang, Henrik Christensen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06791)  

**Abstract**: Motion planning is a fundamental problem in robotics that involves generating feasible trajectories for a robot to follow. Recent advances in parallel computing, particularly through CPU and GPU architectures, have significantly reduced planning times to the order of milliseconds. However, constrained motion planning especially using sampling based methods on GPUs remains underexplored. Prior work such as pRRTC leverages a tracking compiler with a CUDA backend to accelerate forward kinematics and collision checking. While effective in simple settings, their approach struggles with increased complexity in robot models or environments. In this paper, we propose a novel GPU based framework utilizing NVRTC for runtime compilation, enabling efficient handling of high complexity scenarios and supporting constrained motion planning. Experimental results demonstrate that our method achieves superior performance compared to existing approaches. 

**Abstract (ZH)**: 基于GPU的NVRTC驱动高效复杂约束运动规划方法 

---
# Digital-physical testbed for ship autonomy studies in the Marine Cybernetics Laboratory basin 

**Title (ZH)**: 海洋控制实验室水池中的数字物理试验台用于船舶自主性研究 

**Authors**: Emir Cem Gezer, Mael Korentin Ivan Moreau, Anders Sandneseng Høgden, Dong Trong Nguyen, Roger Skjetne, Asgeir Sørensen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06787)  

**Abstract**: The algorithms developed for Maritime Autonomous Surface Ships (MASS) are often challenging to test on actual vessels due to high operational costs and safety considerations. Simulations offer a cost-effective alternative and eliminate risks, but they may not accurately represent real-world dynamics for the given tasks. Utilizing small-scale model ships and robotic vessels in conjunction with a laboratory basin provides an accessible testing environment for the early stages of validation processes. However, designing and developing a model vessel for a single test can be costly and cumbersome, and often researchers lack availability to such infrastructure. To address these challenges and enable streamlined testing, we have developed an in-house testbed that facilitates the development, testing, verification, and validation of MASS algorithms in a digital-physical laboratory. This infrastructure includes a set of small-scale model vessels, a simulation environment for each vessel, a comprehensive testbed environment, and a digital twin in Unity. With this, we aim to establish a full design and verification pipeline that starts with high-fidelity simulation models of each model vessel, to the model-scale testing in the laboratory basin, allowing possibilities for moving to semi-fullscale validation with the R/V milliAmpere 1 passenger ferry and full-scale validation using the R/V Gunnerus. In this work, we present our progress on the development of this testbed environment and its components, demonstrating its effectiveness in enabling ship guidance, navigation, and control (GNC) including autonomy. 

**Abstract (ZH)**: 针对海上自主表面船舶（MASS）开发的算法往往由于高昂的运营成本和安全考虑难以在实际船舶上进行测试。模拟提供了一种成本效益高的替代方案并消除了风险，但可能无法准确地代表给定任务的实际动态。通过结合使用小比例模型船和无人驾驶船舶，并在实验室水槽中进行测试，可以在验证过程的早期阶段提供一种易于访问的测试环境。然而，为单一测试设计和开发模型船只可能成本高昂且繁琐，且研究人员通常缺乏此类基础设施的可用性。为应对这些挑战并实现顺畅的测试过程，我们开发了一套内部测试床，用以在数字物理实验室中开发、测试、验证和验证MASS算法。该基础设施包括一组小比例模型船、每艘船的仿真环境、一个综合测试环境以及在Unity中的数字孪生。通过这一系统，我们旨在建立一条从每个模型船只的高保真仿真模型，到在实验室水槽中进行模型规模测试的完整设计和验证流程，允许过渡到半全尺度验证R/V milliAmpere 1乘客渡船，并最终进行全尺度验证R/V Gunnerus。在本文中，我们介绍了在此测试床环境及其组件的开发进展，并展示了其在船舶导航、制导与控制（GNC）包括自主性方面的有效性。 

---
# FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation 

**Title (ZH)**: FALCON: 学习自适应力 humanoid 操纵-developération 

**Authors**: Yuanhang Zhang, Yifu Yuan, Prajwal Gurunath, Tairan He, Shayegan Omidshafiei, Ali-akbar Agha-mohammadi, Marcell Vazquez-Chanlatte, Liam Pedersen, Guanya Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06776)  

**Abstract**: Humanoid loco-manipulation holds transformative potential for daily service and industrial tasks, yet achieving precise, robust whole-body control with 3D end-effector force interaction remains a major challenge. Prior approaches are often limited to lightweight tasks or quadrupedal/wheeled platforms. To overcome these limitations, we propose FALCON, a dual-agent reinforcement-learning-based framework for robust force-adaptive humanoid loco-manipulation. FALCON decomposes whole-body control into two specialized agents: (1) a lower-body agent ensuring stable locomotion under external force disturbances, and (2) an upper-body agent precisely tracking end-effector positions with implicit adaptive force compensation. These two agents are jointly trained in simulation with a force curriculum that progressively escalates the magnitude of external force exerted on the end effector while respecting torque limits. Experiments demonstrate that, compared to the baselines, FALCON achieves 2x more accurate upper-body joint tracking, while maintaining robust locomotion under force disturbances and achieving faster training convergence. Moreover, FALCON enables policy training without embodiment-specific reward or curriculum tuning. Using the same training setup, we obtain policies that are deployed across multiple humanoids, enabling forceful loco-manipulation tasks such as transporting payloads (0-20N force), cart-pulling (0-100N), and door-opening (0-40N) in the real world. 

**Abstract (ZH)**: 人形机器人 manipulative 运动在日常服务和工业任务中具有变革性的潜力，但实现精确且健壮的三维末端执行器力交互全程体控制仍然是一个重大挑战。先前的方法通常局限于轻量级任务或四足或轮式平台。为克服这些限制，我们提出了 FALCON，一个基于双代理强化学习的人形机器人健壮力自适应 manipulative 运动框架。FALCON 将全程体控制分解为两个专门的代理：（1）下肢代理，确保在外部力干扰下稳定行走；（2）上肢代理，精确跟踪末端执行器位置并隐式进行自适应力补偿。这两个代理在模拟中联合训练，通过渐进增强作用于末端执行器的外部力的大小来构建力课程，同时遵守扭矩限制。实验表明，与基线方法相比，FALCON 在保持稳健行走能力的同时，上肢关节追踪更精确，训练收敛速度更快。此外，FALCON 允许在无需针对具体实体的奖励或课程进行调整的情况下训练策略。使用相同的训练设置，我们获得了可在多个机器人上部署的策略，实现了诸如搬运载荷（0-20N 力）、拉小车（0-100N）和开门（0-40N）等真实的力交互 maniuplation 任务。 

---
# JaxRobotarium: Training and Deploying Multi-Robot Policies in 10 Minutes 

**Title (ZH)**: JaxRobotarium：十分钟内训练和部署多机器人策略 

**Authors**: Shalin Anand Jain, Jiazhen Liu, Siva Kailas, Harish Ravichandar  

**Link**: [PDF](https://arxiv.org/pdf/2505.06771)  

**Abstract**: Multi-agent reinforcement learning (MARL) has emerged as a promising solution for learning complex and scalable coordination behaviors in multi-robot systems. However, established MARL platforms (e.g., SMAC and MPE) lack robotics relevance and hardware deployment, leaving multi-robot learning researchers to develop bespoke environments and hardware testbeds dedicated to the development and evaluation of their individual contributions. The Multi-Agent RL Benchmark and Learning Environment for the Robotarium (MARBLER) is an exciting recent step in providing a standardized robotics-relevant platform for MARL, by bridging the Robotarium testbed with existing MARL software infrastructure. However, MARBLER lacks support for parallelization and GPU/TPU execution, making the platform prohibitively slow compared to modern MARL environments and hindering adoption. We contribute JaxRobotarium, a Jax-powered end-to-end simulation, learning, deployment, and benchmarking platform for the Robotarium. JaxRobotarium enables rapid training and deployment of multi-robot reinforcement learning (MRRL) policies with realistic robot dynamics and safety constraints, supporting both parallelization and hardware acceleration. Our generalizable learning interface provides an easy-to-use integration with SOTA MARL libraries (e.g., JaxMARL). In addition, JaxRobotarium includes eight standardized coordination scenarios, including four novel scenarios that bring established MARL benchmark tasks (e.g., RWARE and Level-Based Foraging) to a realistic robotics setting. We demonstrate that JaxRobotarium retains high simulation fidelity while achieving dramatic speedups over baseline (20x in training and 150x in simulation), and provides an open-access sim-to-real evaluation pipeline through the Robotarium testbed, accelerating and democratizing access to multi-robot learning research and evaluation. 

**Abstract (ZH)**: 基于Jax的Robotarium多智能体强化学习平台：加速和普及多机器人学习研究 

---
# Learned IMU Bias Prediction for Invariant Visual Inertial Odometry 

**Title (ZH)**: 基于学习的IMU偏置预测用于不变视觉惯性里程计 

**Authors**: Abdullah Altawaitan, Jason Stanley, Sambaran Ghosal, Thai Duong, Nikolay Atanasov  

**Link**: [PDF](https://arxiv.org/pdf/2505.06748)  

**Abstract**: Autonomous mobile robots operating in novel environments depend critically on accurate state estimation, often utilizing visual and inertial measurements. Recent work has shown that an invariant formulation of the extended Kalman filter improves the convergence and robustness of visual-inertial odometry by utilizing the Lie group structure of a robot's position, velocity, and orientation states. However, inertial sensors also require measurement bias estimation, yet introducing the bias in the filter state breaks the Lie group symmetry. In this paper, we design a neural network to predict the bias of an inertial measurement unit (IMU) from a sequence of previous IMU measurements. This allows us to use an invariant filter for visual inertial odometry, relying on the learned bias prediction rather than introducing the bias in the filter state. We demonstrate that an invariant multi-state constraint Kalman filter (MSCKF) with learned bias predictions achieves robust visual-inertial odometry in real experiments, even when visual information is unavailable for extended periods and the system needs to rely solely on IMU measurements. 

**Abstract (ZH)**: 自主移动机器人在新型环境中的操作高度依赖于准确的状态估计，通常利用视觉和惯性测量。最近的研究表明，扩展卡尔曼滤波器的一种不变形式通过利用机器人位置、速度和姿态状态的李群结构，可以改善视觉-惯性里程计的收敛性和鲁棒性。然而，惯性传感器还需要进行偏置估计，而在滤波器状态中引入偏置会破坏李群对称性。在本文中，我们设计了一个神经网络来从惯性测量单元(IMU)的先前测量序列中预测偏置。这使我们能够使用不变滤波器进行视觉-惯性里程计，依靠学习到的偏置预测，而不是在滤波器状态中引入偏置。我们展示了在实际实验中，基于学习偏置预测的不变多状态约束Kalman滤波器(MSCKF)可以在长时间缺乏视觉信息且系统仅依赖IMU测量的情况下实现稳健的视觉-惯性里程计。 

---
# M3CAD: Towards Generic Cooperative Autonomous Driving Benchmark 

**Title (ZH)**: M3CAD: 向通用协作自动驾驶基准迈进 

**Authors**: Morui Zhu, Yongqi Zhu, Yihao Zhu, Qi Chen, Deyuan Qu, Song Fu, Qing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06746)  

**Abstract**: We introduce M$^3$CAD, a novel benchmark designed to advance research in generic cooperative autonomous driving. M$^3$CAD comprises 204 sequences with 30k frames, spanning a diverse range of cooperative driving scenarios. Each sequence includes multiple vehicles and sensing modalities, e.g., LiDAR point clouds, RGB images, and GPS/IMU, supporting a variety of autonomous driving tasks, including object detection and tracking, mapping, motion forecasting, occupancy prediction, and path planning. This rich multimodal setup enables M$^3$CAD to support both single-vehicle and multi-vehicle autonomous driving research, significantly broadening the scope of research in the field. To our knowledge, M$^3$CAD is the most comprehensive benchmark specifically tailored for cooperative multi-task autonomous driving research. We evaluate the state-of-the-art end-to-end solution on M$^3$CAD to establish baseline performance. To foster cooperative autonomous driving research, we also propose E2EC, a simple yet effective framework for cooperative driving solution that leverages inter-vehicle shared information for improved path planning. We release M$^3$CAD, along with our baseline models and evaluation results, to support the development of robust cooperative autonomous driving systems. All resources will be made publicly available on this https URL 

**Abstract (ZH)**: M$^3$CAD：一种促进通用协同自动驾驶研究的新基准 

---
# TPK: Trustworthy Trajectory Prediction Integrating Prior Knowledge For Interpretability and Kinematic Feasibility 

**Title (ZH)**: TPK：集成先验知识的可解释性和动态可行性轨迹预测 

**Authors**: Marius Baden, Ahmed Abouelazm, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06743)  

**Abstract**: Trajectory prediction is crucial for autonomous driving, enabling vehicles to navigate safely by anticipating the movements of surrounding road users. However, current deep learning models often lack trustworthiness as their predictions can be physically infeasible and illogical to humans. To make predictions more trustworthy, recent research has incorporated prior knowledge, like the social force model for modeling interactions and kinematic models for physical realism. However, these approaches focus on priors that suit either vehicles or pedestrians and do not generalize to traffic with mixed agent classes. We propose incorporating interaction and kinematic priors of all agent classes--vehicles, pedestrians, and cyclists with class-specific interaction layers to capture agent behavioral differences. To improve the interpretability of the agent interactions, we introduce DG-SFM, a rule-based interaction importance score that guides the interaction layer. To ensure physically feasible predictions, we proposed suitable kinematic models for all agent classes with a novel pedestrian kinematic model. We benchmark our approach on the Argoverse 2 dataset, using the state-of-the-art transformer HPTR as our baseline. Experiments demonstrate that our method improves interaction interpretability, revealing a correlation between incorrect predictions and divergence from our interaction prior. Even though incorporating the kinematic models causes a slight decrease in accuracy, they eliminate infeasible trajectories found in the dataset and the baseline model. Thus, our approach fosters trust in trajectory prediction as its interaction reasoning is interpretable, and its predictions adhere to physics. 

**Abstract (ZH)**: 轨迹预测对于自动驾驶至关重要，能够通过预见周围道路使用者的移动来确保车辆安全导航。然而，当前的深度学习模型往往缺乏可信度，因为它们的预测可能存在物理上的不可行性且不符合人类逻辑。为了使预测更加可信，最近的研究将先验知识融入其中，如用社会力模型来建模互动，用动力学模型来增加物理现实感。然而，这些方法专注于分别适用于车辆或行人的先验知识，无法泛化到包含混合类型代理的交通情况。我们提出将所有代理类别的互动和动力学先验知识——车辆、行人和骑车人——结合起来，并为每类代理设计特定的互动层，以捕捉代理行为的差异。为了提高代理互动的可解释性，我们引入了基于规则的互动重要性评分DG-SFM，以引导互动层。为了确保预测的物理可行性，我们提出了适用于所有代理类别的动力学模型，并设计了一种新颖的行人工学模型。我们在Argoverse 2数据集上对标最先进的Transformer HPTR作为基准进行评估。实验表明，我们的方法提高了互动解释性，揭示了错误预测与偏离我们的互动先验之间的关联。尽管引入动力学模型导致准确性略有下降，但它们消除了数据集中和基准模型中发现的不合理的轨迹。因此，我们的方法增强了轨迹预测的信任度，因为其互动推理是可解释的，且其预测遵循物理规则。 

---
# Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving 

**Title (ZH)**: 基于边界引导的路径预测以实现道路aware和物理可行的自动驾驶 

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06740)  

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66\% to just 1\%. These results highlight the effectiveness of our approach in generating feasible and robust predictions. 

**Abstract (ZH)**: 准确预测周围道路使用者的轨迹对于安全高效的自动驾驶至关重要。尽管深度学习模型提高了性能，但仍然存在防止离路预测和确保动力学可行性的挑战。现有方法结合了道路感知模块并施加动力学约束，但缺乏可行性保证，经常引入复杂性和灵活性之间的权衡。本文提出了一种新颖的框架，将轨迹预测公式化为受允许驾驶方向及其边界引导的受限回归问题。利用代理的当前状态和高精度地图，该方法定义有效的边界，并通过训练网络学习边界多边线之间的叠加路径来确保道路内预测。为了保证可行性，模型预测加速度轮廓，以确定车辆沿这些路径的行驶距离，同时遵守动力学约束。我们在Argoverse-2数据集上将该方法与HPTR基线进行对比评估。我们的方法在基准指标上略有降低，但在最终位移误差方面有显著改进，并消除了无效轨迹。此外，所提出的方法在对少见操作和未见过的分布外场景具有更高的泛化能力，对抗攻击下的离路率从66%降低到仅1%。这些结果突显了该方法在生成可行性和鲁棒性预测方面的有效性。 

---
# Balancing Progress and Safety: A Novel Risk-Aware Objective for RL in Autonomous Driving 

**Title (ZH)**: 平衡进展与安全：自动驾驶中强化学习的新颖风险感知目标 

**Authors**: Ahmed Abouelazm, Jonas Michel, Helen Gremmelmaier, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.06737)  

**Abstract**: Reinforcement Learning (RL) is a promising approach for achieving autonomous driving due to robust decision-making capabilities. RL learns a driving policy through trial and error in traffic scenarios, guided by a reward function that combines the driving objectives. The design of such reward function has received insufficient attention, yielding ill-defined rewards with various pitfalls. Safety, in particular, has long been regarded only as a penalty for collisions. This leaves the risks associated with actions leading up to a collision unaddressed, limiting the applicability of RL in real-world scenarios. To address these shortcomings, our work focuses on enhancing the reward formulation by defining a set of driving objectives and structuring them hierarchically. Furthermore, we discuss the formulation of these objectives in a normalized manner to transparently determine their contribution to the overall reward. Additionally, we introduce a novel risk-aware objective for various driving interactions based on a two-dimensional ellipsoid function and an extension of Responsibility-Sensitive Safety (RSS) concepts. We evaluate the efficacy of our proposed reward in unsignalized intersection scenarios with varying traffic densities. The approach decreases collision rates by 21\% on average compared to baseline rewards and consistently surpasses them in route progress and cumulative reward, demonstrating its capability to promote safer driving behaviors while maintaining high-performance levels. 

**Abstract (ZH)**: 强化学习（RL）在实现自主驾驶中的应用由于其强大的决策能力而充满 promise，并通过在交通场景中基于奖励函数的试错学习驾驶策略。然而，奖励函数的设计未能得到充分关注，导致定义模糊且存在诸多问题。特别是在安全性方面，长期仅将安全视为碰撞的惩罚，忽视了导致碰撞之前的行为风险，限制了 RL 在实际场景中的应用。为解决这些问题，我们的工作集中在增强奖励的制定，通过定义一系列驾驶目标并将其分层结构化来改善奖励公式。此外，我们以规范化的方式讨论这些目标的制定，以透明地确定它们对总体奖励的贡献。我们还引入了一个基于椭圆函数和扩展责任敏感安全（RSS）概念的新颖风险感知目标，用于各种驾驶交互。我们在不同交通密度的无信号交叉口场景中评估了我们提出奖励的有效性，该方法平均将碰撞率降低了 21%，并且在路线进展和累计奖励方面持续超过基线奖励，证明了其在促进更安全驾驶行为的同时保持高性能的能力。 

---
# STRIVE: Structured Representation Integrating VLM Reasoning for Efficient Object Navigation 

**Title (ZH)**: STRIVE: 结构化表示集成VLM推理以实现高效物体导航 

**Authors**: Haokun Zhu, Zongtai Li, Zhixuan Liu, Wenshan Wang, Ji Zhang, Jonathan Francis, Jean Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.06729)  

**Abstract**: Vision-Language Models (VLMs) have been increasingly integrated into object navigation tasks for their rich prior knowledge and strong reasoning abilities. However, applying VLMs to navigation poses two key challenges: effectively representing complex environment information and determining \textit{when and how} to query VLMs. Insufficient environment understanding and over-reliance on VLMs (e.g. querying at every step) can lead to unnecessary backtracking and reduced navigation efficiency, especially in continuous environments. To address these challenges, we propose a novel framework that constructs a multi-layer representation of the environment during navigation. This representation consists of viewpoint, object nodes, and room nodes. Viewpoints and object nodes facilitate intra-room exploration and accurate target localization, while room nodes support efficient inter-room planning. Building on this representation, we propose a novel two-stage navigation policy, integrating high-level planning guided by VLM reasoning with low-level VLM-assisted exploration to efficiently locate a goal object. We evaluated our approach on three simulated benchmarks (HM3D, RoboTHOR, and MP3D), and achieved state-of-the-art performance on both the success rate ($\mathord{\uparrow}\, 7.1\%$) and navigation efficiency ($\mathord{\uparrow}\, 12.5\%$). We further validate our method on a real robot platform, demonstrating strong robustness across 15 object navigation tasks in 10 different indoor environments. Project page is available at this https URL . 

**Abstract (ZH)**: 视觉语言模型（VLMs）已越来越多地集成到对象导航任务中，利用其丰富的先验知识和强大的推理能力。然而，将VLMs应用于导航面临着两个关键挑战：有效地表示复杂环境信息以及确定何时及如何查询VLMs。环境理解不足和过度依赖VLMs（例如，在每一步都查询）会导致不必要的回溯和导航效率降低，尤其是在连续环境中。为了解决这些挑战，我们提出了一种新型框架，在导航过程中构建环境的多层表示。该表示由视点、物体节点和房间节点组成。视点和物体节点促进了房间内的探索和精确的目标定位，而房间节点支持高效的跨房间规划。基于这种表示，我们提出了一种新的两阶段导航策略，整合了由VLM推理引导的高层次规划和VLM辅助的低层次探索，以高效地定位目标物体。我们在三个模拟基准（HM3D、RoboTHOR和MP3D）上评估了我们的方法，并在成功率（提高7.1%）和导航效率（提高12.5%）上达到了最先进的性能。我们在真实的机器人平台上进一步验证了该方法，展示了在10种不同的室内环境中执行15种物体导航任务的强大鲁棒性。项目页面请访问此链接。 

---
# Motion Planning for Autonomous Vehicles: When Model Predictive Control Meets Ensemble Kalman Smoothing 

**Title (ZH)**: 自主车辆的运动规划：模型预测控制与集成卡尔曼平滑相结合 

**Authors**: Iman Askari, Yebin Wang, Vedeng M. Deshpande, Huazhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06666)  

**Abstract**: Safe and efficient motion planning is of fundamental importance for autonomous vehicles. This paper investigates motion planning based on nonlinear model predictive control (NMPC) over a neural network vehicle model. We aim to overcome the high computational costs that arise in NMPC of the neural network model due to the highly nonlinear and nonconvex optimization. In a departure from numerical optimization solutions, we reformulate the problem of NMPC-based motion planning as a Bayesian estimation problem, which seeks to infer optimal planning decisions from planning objectives. Then, we use a sequential ensemble Kalman smoother to accomplish the estimation task, exploiting its high computational efficiency for complex nonlinear systems. The simulation results show an improvement in computational speed by orders of magnitude, indicating the potential of the proposed approach for practical motion planning. 

**Abstract (ZH)**: 基于神经网络车辆模型的非线性模型预测控制的运动规划：一种贝叶斯估计方法的研究 

---
# 3D Characterization of Smoke Plume Dispersion Using Multi-View Drone Swarm 

**Title (ZH)**: 多视点无人机群用于烟雾羽流扩散的三维表征 

**Authors**: Nikil Krishnakumar, Shashank Sharma, Srijan Kumar Pal, Jiarong Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.06638)  

**Abstract**: This study presents an advanced multi-view drone swarm imaging system for the three-dimensional characterization of smoke plume dispersion dynamics. The system comprises a manager drone and four worker drones, each equipped with high-resolution cameras and precise GPS modules. The manager drone uses image feedback to autonomously detect and position itself above the plume, then commands the worker drones to orbit the area in a synchronized circular flight pattern, capturing multi-angle images. The camera poses of these images are first estimated, then the images are grouped in batches and processed using Neural Radiance Fields (NeRF) to generate high-resolution 3D reconstructions of plume dynamics over time. Field tests demonstrated the ability of the system to capture critical plume characteristics including volume dynamics, wind-driven directional shifts, and lofting behavior at a temporal resolution of about 1 s. The 3D reconstructions generated by this system provide unique field data for enhancing the predictive models of smoke plume dispersion and fire spread. Broadly, the drone swarm system offers a versatile platform for high resolution measurements of pollutant emissions and transport in wildfires, volcanic eruptions, prescribed burns, and industrial processes, ultimately supporting more effective fire control decisions and mitigating wildfire risks. 

**Abstract (ZH)**: 本研究提出了一种先进的多视角无人机群成像系统，用于三维表征烟羽分散动力学。系统包括一架管理者无人机和四架工人无人机，每架都装备有高分辨率相机和精确的GPS模块。管理者无人机通过图像反馈自主检测并定位在烟羽上方，然后指挥工人无人机以同步圆形飞行模式环绕区域飞行，捕获多角度图像。首先估计这些图像的相机姿态，然后将图像分批处理并使用神经辐射场（NeRF）生成烟羽动力学随时间演化的高分辨率三维重构。实地测试证明，该系统能够在大约1秒的时分辨率下捕获关键的烟羽特性，包括体积动态、风驱动的方向偏移以及抬升行为。由该系统生成的三维重构提供了独特的现场数据，以增强烟羽分散和火灾蔓延的预测模型。总体而言，无人机群系统为高分辨率测量野火、火山爆发、规定烧毁、工业过程中的污染物排放和传输提供了多功能平台，最终支持更有效的火灾控制决策并减轻野火风险。 

---
# ACORN: Adaptive Contrastive Optimization for Safe and Robust Fine-Grained Robotic Manipulation 

**Title (ZH)**: ACORN: 自适应对比优化以实现安全鲁棒的细粒度机器人操作 

**Authors**: Zhongquan Zhou, Shuhao Li, Zixian Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.06628)  

**Abstract**: Embodied AI research has traditionally emphasized performance metrics such as success rate and cumulative reward, overlooking critical robustness and safety considerations that emerge during real-world deployment. In actual environments, agents continuously encounter unpredicted situations and distribution shifts, causing seemingly reliable policies to experience catastrophic failures, particularly in manipulation tasks. To address this gap, we introduce four novel safety-centric metrics that quantify an agent's resilience to environmental perturbations. Building on these metrics, we present Adaptive Contrastive Optimization for Robust Manipulation (ACORN), a plug-and-play algorithm that enhances policy robustness without sacrificing performance. ACORN leverages contrastive learning to simultaneously align trajectories with expert demonstrations while diverging from potentially unsafe behaviors. Our approach efficiently generates informative negative samples through structured Gaussian noise injection, employing a double perturbation technique that maintains sample diversity while minimizing computational overhead. Comprehensive experiments across diverse manipulation environments validate ACORN's effectiveness, yielding improvements of up to 23% in safety metrics under disturbance compared to baseline methods. These findings underscore ACORN's significant potential for enabling reliable deployment of embodied agents in safety-critical real-world applications. 

**Abstract (ZH)**: 实体AI研究传统上强调成功 rate 和累计奖励等性能指标，忽视了实际部署中出现的关键稳健性和安全性考虑。在实际环境中，智能体不断遇到不可预测的情况和分布偏移，导致看似可靠的策略在操作任务中经历灾难性的失败。为解决这一问题，我们引入了四个新型的安全中心化指标，量化智能体对环境扰动的抗性。在这些指标的基础上，我们提出了适用于稳健操作的自适应对比优化算法（ACORN），该算法能够在不牺牲性能的情况下增强策略的稳健性。ACORN 利用对比学习同时使轨迹与专家演示对齐，同时避免潜在的不安全行为。通过结构化高斯噪声注入生成有效的负样本，采用双重扰动技术维持样本多样性同时减少计算开销。跨多种操作环境的全面实验验证了 ACORN 的有效性，在干扰条件下安全指标提高了高达 23%，基线方法表现出显著改进。这些发现突显了 ACORN 在安全关键的实际应用中可靠部署实体智能体方面的巨大潜力。 

---
# Emergent Multi-View Fidelity in Autonomous UAV Swarm Sport Injury Detection 

**Title (ZH)**: 自主无人机群运动损伤检测中 Emergent 多视角保真度 

**Authors**: Yu Cheng, Harun Šiljak  

**Link**: [PDF](https://arxiv.org/pdf/2505.06588)  

**Abstract**: Accurate, real-time collision detection is essential for ensuring player safety and effective refereeing in high-contact sports such as rugby, particularly given the severe risks associated with traumatic brain injuries (TBI). Traditional collision-monitoring methods employing fixed cameras or wearable sensors face limitations in visibility, coverage, and responsiveness. Previously, we introduced a framework using unmanned aerial vehicles (UAVs) for monitoring and real time kinematics extraction from videos of collision events. In this paper, we show that the strategies operating on the objective of ensuring at least one UAV captures every incident on the pitch have an emergent property of fulfilling a stronger key condition for successful kinematics extraction. Namely, they ensure that almost all collisions are captured by multiple drones, establishing multi-view fidelity and redundancy, while not requiring any drone-to-drone communication. 

**Abstract (ZH)**: 准确的实时碰撞检测对于确保如橄榄球等高接触运动中玩家的安全及有效裁判至关重要，尤其是在与重度脑损伤（TBI）相关的严重风险下。传统的使用固定摄像头或穿戴式传感器的碰撞监测方法在可见性、覆盖范围和响应性方面存在局限。此前，我们提出了使用无人驾驶航空 vehicle (UAV) 的框架，用于监测和实时从碰撞事件视频中提取动态信息。本文显示，旨在确保至少一架无人机捕获场内每个事件的策略具有一个 Emergent 属性，即它们确保几乎所有的碰撞被多架无人机同时捕获，从而建立多视角保真度和冗余性，同时无需无人机之间的通信。 

---
# JAEGER: Dual-Level Humanoid Whole-Body Controller 

**Title (ZH)**: JAEGER: 双层级-humanoid 全身控制器 

**Authors**: Ziluo Ding, Haobin Jiang, Yuxuan Wang, Zhenguo Sun, Yu Zhang, Xiaojie Niu, Ming Yang, Weishuai Zeng, Xinrun Xu, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06584)  

**Abstract**: This paper presents JAEGER, a dual-level whole-body controller for humanoid robots that addresses the challenges of training a more robust and versatile policy. Unlike traditional single-controller approaches, JAEGER separates the control of the upper and lower bodies into two independent controllers, so that they can better focus on their distinct tasks. This separation alleviates the dimensionality curse and improves fault tolerance. JAEGER supports both root velocity tracking (coarse-grained control) and local joint angle tracking (fine-grained control), enabling versatile and stable movements. To train the controller, we utilize a human motion dataset (AMASS), retargeting human poses to humanoid poses through an efficient retargeting network, and employ a curriculum learning approach. This method performs supervised learning for initialization, followed by reinforcement learning for further exploration. We conduct our experiments on two humanoid platforms and demonstrate the superiority of our approach against state-of-the-art methods in both simulation and real environments. 

**Abstract (ZH)**: JAEGER：一种针对类人机器人鲁棒性和 versatility 的双层全身控制器 

---
# Quadrupedal Robot Skateboard Mounting via Reverse Curriculum Learning 

**Title (ZH)**: 四足机器人滑板安装 via 逆序 Curriculum Learning 

**Authors**: Danil Belov, Artem Erkhov, Elizaveta Pestova, Ilya Osokin, Dzmitry Tsetserukou, Pavel Osinenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06561)  

**Abstract**: The aim of this work is to enable quadrupedal robots to mount skateboards using Reverse Curriculum Reinforcement Learning. Although prior work has demonstrated skateboarding for quadrupeds that are already positioned on the board, the initial mounting phase still poses a significant challenge. A goal-oriented methodology was adopted, beginning with the terminal phases of the task and progressively increasing the complexity of the problem definition to approximate the desired objective. The learning process was initiated with the skateboard rigidly fixed within the global coordinate frame and the robot positioned directly above it. Through gradual relaxation of these initial conditions, the learned policy demonstrated robustness to variations in skateboard position and orientation, ultimately exhibiting a successful transfer to scenarios involving a mobile skateboard. The code, trained models, and reproducible examples are available at the following link: this https URL 

**Abstract (ZH)**: 本工作旨在通过逆 Curriculum 强化学习使四足机器人能够骑上滑板。尽管先前的工作已经展示了已经在滑板上定位的四足机器人滑板行驶的能力，但初始上板阶段仍然是一项重大挑战。采用目标导向的方法，从任务的最终阶段开始，逐步增加问题定义的复杂性，以逼近最终目标。学习过程从将滑板刚性固定在全局坐标框架内，并将机器人直接定位在其上方开始。通过对初始条件的逐步放松，学习到的策略显示出了对滑板位置和方向变化的鲁棒性，并最终在涉及移动滑板的情景中表现出成功的转移。相关代码、训练模型和可再现示例可在以下链接获取：this https URL 

---
# LLM-Flock: Decentralized Multi-Robot Flocking via Large Language Models and Influence-Based Consensus 

**Title (ZH)**: LLM-Flock: 基于大型语言模型和基于影响的共识的去中心化多机器人群集算法 

**Authors**: Peihan Li, Lifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06513)  

**Abstract**: Large Language Models (LLMs) have advanced rapidly in recent years, demonstrating strong capabilities in problem comprehension and reasoning. Inspired by these developments, researchers have begun exploring the use of LLMs as decentralized decision-makers for multi-robot formation control. However, prior studies reveal that directly applying LLMs to such tasks often leads to unstable and inconsistent behaviors, where robots may collapse to the centroid of their positions or diverge entirely due to hallucinated reasoning, logical inconsistencies, and limited coordination awareness. To overcome these limitations, we propose a novel framework that integrates LLMs with an influence-based plan consensus protocol. In this framework, each robot independently generates a local plan toward the desired formation using its own LLM. The robots then iteratively refine their plans through a decentralized consensus protocol that accounts for their influence on neighboring robots. This process drives the system toward a coherent and stable flocking formation in a fully decentralized manner. We evaluate our approach through comprehensive simulations involving both state-of-the-art closed-source LLMs (e.g., o3-mini, Claude 3.5) and open-source models (e.g., Llama3.1-405b, Qwen-Max, DeepSeek-R1). The results show notable improvements in stability, convergence, and adaptability over previous LLM-based methods. We further validate our framework on a physical team of Crazyflie drones, demonstrating its practical viability and effectiveness in real-world multi-robot systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）近年来取得了快速进展，展示了在问题理解和推理方面的强大能力。受这些进展的启发，研究人员开始探索将LLMs用于多机器人编队控制的分散决策制定。然而，前期研究显示，直接将LLMs应用于此类任务往往会引发不稳定的且不一致的行为，机器人可能会坍缩到其位置的质心，或者完全发散，这归因于幻觉推理、逻辑不一致和有限的协调意识。为克服这些限制，我们提出了一种将LLMs与基于影响的计划一致性协议相结合的新型框架。在该框架中，每个机器人使用自己的LLM独立生成通往期望编队的局部计划。然后，通过一个考虑机器人对其邻近机器人影响的分散一致性协议，机器人逐步细化其计划。这一过程在完全分散的方式下将系统导向协同且稳定的 flocking 编队。我们通过综合模拟评估了我们的方法，模拟中包括最先进的闭源LLMs（如o3-mini，Claude 3.5）和开源模型（如Llama3.1-405b，Qwen-Max，DeepSeek-R1）。结果表明，与之前的基于LLM的方法相比，在稳定性、收敛性和适应性方面有所提升。进一步地，我们在实际的疯狂飞机无人机团队上验证了我们的框架，证明了其在实际多机器人系统中的实用性和有效性。 

---
# CompSLAM: Complementary Hierarchical Multi-Modal Localization and Mapping for Robot Autonomy in Underground Environments 

**Title (ZH)**: CompSLAM: 地下环境机器人自主性中互补分层多模态定位与地图构建 

**Authors**: Shehryar Khattak, Timon Homberger, Lukas Bernreiter, Julian Nubert, Olov Andersson, Roland Siegwart, Kostas Alexis, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.06483)  

**Abstract**: Robot autonomy in unknown, GPS-denied, and complex underground environments requires real-time, robust, and accurate onboard pose estimation and mapping for reliable operations. This becomes particularly challenging in perception-degraded subterranean conditions under harsh environmental factors, including darkness, dust, and geometrically self-similar structures. This paper details CompSLAM, a highly resilient and hierarchical multi-modal localization and mapping framework designed to address these challenges. Its flexible architecture achieves resilience through redundancy by leveraging the complementary nature of pose estimates derived from diverse sensor modalities. Developed during the DARPA Subterranean Challenge, CompSLAM was successfully deployed on all aerial, legged, and wheeled robots of Team Cerberus during their competition-winning final run. Furthermore, it has proven to be a reliable odometry and mapping solution in various subsequent projects, with extensions enabling multi-robot map sharing for marsupial robotic deployments and collaborative mapping. This paper also introduces a comprehensive dataset acquired by a manually teleoperated quadrupedal robot, covering a significant portion of the DARPA Subterranean Challenge finals course. This dataset evaluates CompSLAM's robustness to sensor degradations as the robot traverses 740 meters in an environment characterized by highly variable geometries and demanding lighting conditions. The CompSLAM code and the DARPA SubT Finals dataset are made publicly available for the benefit of the robotics community 

**Abstract (ZH)**: 地下未知、GPS受限和复杂环境中的机器人自主性要求具备实时、 robust 和精确的机载位姿估计和建图能力，以确保可靠运行。在黑暗、灰尘和几何相似结构等恶劣环境因素导致感知退化的地下条件下，这一要求变得尤为具有挑战性。本文详细介绍了CompSLAM，这是一种针对这些挑战设计的高度鲁棒性和分层多模态定位与建图框架。该框架通过利用来自多种传感模态的位姿估计的互补性，其灵活的架构实现了通过冗余性增强鲁棒性。CompSLAM在DARPA地下挑战赛期间开发，并成功部署在团队Cerberus的所有空中、 legged 和轮式机器人上，用于其竞赛获胜的最终运行。此外，其可靠性和建图解决方案在各种后续项目中得到了验证，并扩展了多机器人地图共享功能，以支持袋鼠机器人部署和协同建图。本文还介绍了由手动遥控四足机器人收集的全面数据集，涵盖了DARPA地下挑战赛决赛赛道的重要部分。数据集评估了机器人在具有高度变化几何结构和严苛光照条件的环境中穿越740米过程中CompSLAM对传感器退化的影响。CompSLAM代码和DARPA SubT决赛数据集已公开发布，以造福于机器人社区。 

---
# Adaptive Wiping: Adaptive contact-rich manipulation through few-shot imitation learning with Force-Torque feedback and pre-trained object representations 

**Title (ZH)**: 自适应擦拭：通过力-力矩反馈和预训练物体表示的少量示范模仿学习实现接触丰富的自适应操作 

**Authors**: Chikaha Tsuji, Enrique Coronado, Pablo Osorio, Gentiane Venture  

**Link**: [PDF](https://arxiv.org/pdf/2505.06451)  

**Abstract**: Imitation learning offers a pathway for robots to perform repetitive tasks, allowing humans to focus on more engaging and meaningful activities. However, challenges arise from the need for extensive demonstrations and the disparity between training and real-world environments. This paper focuses on contact-rich tasks like wiping with soft and deformable objects, requiring adaptive force control to handle variations in wiping surface height and the sponge's physical properties. To address these challenges, we propose a novel method that integrates real-time force-torque (FT) feedback with pre-trained object representations. This approach allows robots to dynamically adjust to previously unseen changes in surface heights and sponges' physical properties. In real-world experiments, our method achieved 96% accuracy in applying reference forces, significantly outperforming the previous method that lacked an FT feedback loop, which only achieved 4% accuracy. To evaluate the adaptability of our approach, we conducted experiments under different conditions from the training setup, involving 40 scenarios using 10 sponges with varying physical properties and 4 types of wiping surface heights, demonstrating significant improvements in the robot's adaptability by analyzing force trajectories. The video of our work is available at: this https URL 

**Abstract (ZH)**: 模仿学习为机器人执行重复任务提供了途径，使人类能够集中精力进行更具吸引力和意义的活动。然而，这需要大量的示范，并且訓練环境与真实世界环境之间存在差距。本文专注于接触丰富的任务，如用柔软可变形物体擦拭，需要适应性的力控制来处理擦拭表面高度和海绵物理性质的变化。为解决这些挑战，我们提出了一种新方法，该方法将实时力-扭矩（FT）反馈与预训练的对象表示相结合。这种方法使机器人能够动态适应前所未见的表面高度和海绵物理性质的变化。在实际实验中，我们的方法在施加参考力方面的准确率达到96%，明显优于缺乏FT反馈回路的先前方法，后者仅达到4%的准确率。为了评估我们方法的适应性，我们在与训练设置不同的条件下进行了实验，使用10种具有不同物理性质的海绵进行40种场景实验，并通过分析力轨迹展示了机器人适应性的显著提高。我们的工作视频可在此处查看：this https URL 

---
# Autonomous Vision-Based Magnetic Microrobotic Pushing of Micro-Objects and Cells 

**Title (ZH)**: 基于自主视觉的磁微机器人对微物体和细胞的操控推移 

**Authors**: Max Sokolich, Ceren Kirmizitas, Sambeeta Das, Ron Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2505.06450)  

**Abstract**: Accurate and autonomous transportation of micro-objects and biological cells can enable significant advances in a wide variety of research disciplines. Here, we present a novel, vision-based, model-free microrobotic pushing algorithm for the autonomous manipulation of micro objects and biological cells. The algorithm adjusts the axis of a rotating magnetic field that in turn controls the heading angle and spin axis of a spherical Janus rolling microrobot. We introduce the concept of a microrobotic guiding corridor to constrain the object and to avoid pushing failures. We then show that employing only two simple conditions, the microrobot is able to successfully and autonomously push microscale objects along predefined trajectories. We evaluate the performance of the algorithm by measuring the mean absolute error and completion time relative to a desired path at different actuation frequencies and guiding corridor widths. Finally, we demonstrate biomedical applicability by autonomously transporting a single biological cell, highlighting the methods potential for applications in tissue engineering, drug delivery and synthetic biology. 

**Abstract (ZH)**: 基于视觉的无模型微机器人推动物算法可实现微小物体和生物细胞的精确自主运输，从而推动多个研究领域的重大进展。该算法通过调整旋转磁场的轴线来控制Janus滚珠微机器人的航向角和自旋轴，从而实现对微小物体和生物细胞的自主操作。我们提出了微机器人引导走廊的概念，以约束目标物体并避免推动物操作失败。我们证明，仅通过满足两个简单条件，微机器人就能够成功地沿着预定义的轨迹自主推动物体。通过在不同驱动频率和引导走廊宽度下测量平均绝对误差和完成时间来评估算法性能。最后，通过自主运输单个生物细胞，展示了其在生物医学应用中的潜力，包括组织工程、药物递送和合成生物学领域。 

---
# Camera Control at the Edge with Language Models for Scene Understanding 

**Title (ZH)**: 边缘端基于语言模型的场景理解相机控制 

**Authors**: Alexiy Buynitsky, Sina Ehsani, Bhanu Pallakonda, Pragyana Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2505.06402)  

**Abstract**: In this paper, we present Optimized Prompt-based Unified System (OPUS), a framework that utilizes a Large Language Model (LLM) to control Pan-Tilt-Zoom (PTZ) cameras, providing contextual understanding of natural environments. To achieve this goal, the OPUS system improves cost-effectiveness by generating keywords from a high-level camera control API and transferring knowledge from larger closed-source language models to smaller ones through Supervised Fine-Tuning (SFT) on synthetic data. This enables efficient edge deployment while maintaining performance comparable to larger models like GPT-4. OPUS enhances environmental awareness by converting data from multiple cameras into textual descriptions for language models, eliminating the need for specialized sensory tokens. In benchmark testing, our approach significantly outperformed both traditional language model techniques and more complex prompting methods, achieving a 35% improvement over advanced techniques and a 20% higher task accuracy compared to closed-source models like Gemini Pro. The system demonstrates OPUS's capability to simplify PTZ camera operations through an intuitive natural language interface. This approach eliminates the need for explicit programming and provides a conversational method for interacting with camera systems, representing a significant advancement in how users can control and utilize PTZ camera technology. 

**Abstract (ZH)**: 基于优化提示的统一系统（OPUS）：一种利用大型语言模型控制PTZ摄像头的框架 

---
# LLM-Land: Large Language Models for Context-Aware Drone Landing 

**Title (ZH)**: LLM-陆地：基于上下文感知的大规模语言模型应用于无人机降落 

**Authors**: Siwei Cai, Yuwei Wu, Lifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06399)  

**Abstract**: Autonomous landing is essential for drones deployed in emergency deliveries, post-disaster response, and other large-scale missions. By enabling self-docking on charging platforms, it facilitates continuous operation and significantly extends mission endurance. However, traditional approaches often fall short in dynamic, unstructured environments due to limited semantic awareness and reliance on fixed, context-insensitive safety margins. To address these limitations, we propose a hybrid framework that integrates large language model (LLMs) with model predictive control (MPC). Our approach begins with a vision-language encoder (VLE) (e.g., BLIP), which transforms real-time images into concise textual scene descriptions. These descriptions are processed by a lightweight LLM (e.g., Qwen 2.5 1.5B or LLaMA 3.2 1B) equipped with retrieval-augmented generation (RAG) to classify scene elements and infer context-aware safety buffers, such as 3 meters for pedestrians and 5 meters for vehicles. The resulting semantic flags and unsafe regions are then fed into an MPC module, enabling real-time trajectory replanning that avoids collisions while maintaining high landing precision. We validate our framework in the ROS-Gazebo simulator, where it consistently outperforms conventional vision-based MPC baselines. Our results show a significant reduction in near-miss incidents with dynamic obstacles, while preserving accurate landings in cluttered environments. 

**Abstract (ZH)**: 自主着陆对于部署在紧急配送、灾后响应和其他大规模任务中的无人机至关重要。通过在充电平台上实现自主对接，它促进了连续操作并显著延长了任务续航能力。然而，传统的Approach often falls short in dynamic, unstructured environments due to limited semantic awareness and reliance on fixed, context-insensitive safety margins. To address these limitations, we propose a hybrid framework that integrates large language model (LLMs) with model predictive control (MPC). Our approach begins with a vision-language encoder (VLE) (e.g., BLIP), which transforms real-time images into concise textual scene descriptions. These descriptions are processed by a lightweight LLM (e.g., Qwen 2.5 1.5B or LLaMA 3.2 1B) equipped with retrieval-augmented generation (RAG) to classify scene elements and infer context-aware safety buffers, such as 3 meters for pedestrians and 5 meters for vehicles. The resulting semantic flags and unsafe regions are then fed into an MPC module, enabling real-time trajectory replanning that avoids collisions while maintaining high landing precision. We validate our framework in the ROS-Gazebo simulator, where it consistently outperforms conventional vision-based MPC baselines. Our results show a significant reduction in near-miss incidents with dynamic obstacles, while preserving accurate landings in cluttered environments.

标题：一种结合大规模语言模型和模型预测控制的自主着陆框架 

---
# Learning Sequential Kinematic Models from Demonstrations for Multi-Jointed Articulated Objects 

**Title (ZH)**: 从演示学习多自由度关节对象的序列动力学模型 

**Authors**: Anmol Gupta, Weiwei Gu, Omkar Patil, Jun Ki Lee, Nakul Gopalan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06363)  

**Abstract**: As robots become more generalized and deployed in diverse environments, they must interact with complex objects, many with multiple independent joints or degrees of freedom (DoF) requiring precise control. A common strategy is object modeling, where compact state-space models are learned from real-world observations and paired with classical planning. However, existing methods often rely on prior knowledge or focus on single-DoF objects, limiting their applicability. They also fail to handle occluded joints and ignore the manipulation sequences needed to access them. We address this by learning object models from human demonstrations. We introduce Object Kinematic Sequence Machines (OKSMs), a novel representation capturing both kinematic constraints and manipulation order for multi-DoF objects. To estimate these models from point cloud data, we present Pokenet, a deep neural network trained on human demonstrations. We validate our approach on 8,000 simulated and 1,600 real-world annotated samples. Pokenet improves joint axis and state estimation by over 20 percent on real-world data compared to prior methods. Finally, we demonstrate OKSMs on a Sawyer robot using inverse kinematics-based planning to manipulate multi-DoF objects. 

**Abstract (ZH)**: 随着机器人在多样环境中广泛应用，它们必须与具有多个独立关节或自由度的复杂对象进行交互，需要精确控制。一种常见策略是进行物体建模，在实际观察中学习紧凑的状态空间模型，并与经典规划相结合。然而，现有方法往往依赖于先验知识或专注于单自由度物体，限制了它们的适用性。它们也未能处理遮挡的关节，并忽略访问这些关节所需的操作序列。我们通过从人类示范中学习物体模型来解决这一问题。我们引入了物体动力学序列机（OKSMs），这是一种新颖的表示方法，能够捕捉多自由度物体的动力学约束和操作顺序。为了从点云数据中估计这些模型，我们提出了Pokenet，这是一种在人类示范上训练的深度神经网络。我们验证了该方法在8,000个模拟样本和1,600个真实世界标注样本上的效果。与先前的方法相比，Pokenet 在真实数据中提高了关节轴和状态估计的性能，达到20%以上。最后，我们展示了OKSMs在使用基于逆运动学的规划控制多自由度物体的Sawyer机器人上的应用。 

---
# DAPPER: Discriminability-Aware Policy-to-Policy Preference-Based Reinforcement Learning for Query-Efficient Robot Skill Acquisition 

**Title (ZH)**: DAPPER：基于查询高效性的可辨别性感知政策偏好强化学习技能获取方法 

**Authors**: Yuki Kadokawa, Jonas Frey, Takahiro Miki, Takamitsu Matsubara, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.06357)  

**Abstract**: Preference-based Reinforcement Learning (PbRL) enables policy learning through simple queries comparing trajectories from a single policy. While human responses to these queries make it possible to learn policies aligned with human preferences, PbRL suffers from low query efficiency, as policy bias limits trajectory diversity and reduces the number of discriminable queries available for learning preferences. This paper identifies preference discriminability, which quantifies how easily a human can judge which trajectory is closer to their ideal behavior, as a key metric for improving query efficiency. To address this, we move beyond comparisons within a single policy and instead generate queries by comparing trajectories from multiple policies, as training them from scratch promotes diversity without policy bias. We propose Discriminability-Aware Policy-to-Policy Preference-Based Efficient Reinforcement Learning (DAPPER), which integrates preference discriminability with trajectory diversification achieved by multiple policies. DAPPER trains new policies from scratch after each reward update and employs a discriminator that learns to estimate preference discriminability, enabling the prioritized sampling of more discriminable queries. During training, it jointly maximizes the preference reward and preference discriminability score, encouraging the discovery of highly rewarding and easily distinguishable policies. Experiments in simulated and real-world legged robot environments demonstrate that DAPPER outperforms previous methods in query efficiency, particularly under challenging preference discriminability conditions. 

**Abstract (ZH)**: 基于偏好增强学习的判别能力感知多策略高效学习（Discriminability-Aware Policy-to-Policy Preference-Based Efficient Reinforcement Learning, DAPPER） 

---
# Robust Understanding of Human-Robot Social Interactions through Multimodal Distillation 

**Title (ZH)**: 通过多模态蒸馏实现人类-机器人社会互动的稳健理解 

**Authors**: Tongfei Bian, Mathieu Chollet, Tanaya Guha  

**Link**: [PDF](https://arxiv.org/pdf/2505.06278)  

**Abstract**: The need for social robots and agents to interact and assist humans is growing steadily. To be able to successfully interact with humans, they need to understand and analyse socially interactive scenes from their (robot's) perspective. Works that model social situations between humans and agents are few; and even those existing ones are often too computationally intensive to be suitable for deployment in real time or on real world scenarios with limited available information. We propose a robust knowledge distillation framework that models social interactions through various multimodal cues, yet is robust against incomplete and noisy information during inference. Our teacher model is trained with multimodal input (body, face and hand gestures, gaze, raw images) that transfers knowledge to a student model that relies solely on body pose. Extensive experiments on two publicly available human-robot interaction datasets demonstrate that the our student model achieves an average accuracy gain of 14.75\% over relevant baselines on multiple downstream social understanding task even with up to 51\% of its input being corrupted. The student model is highly efficient: it is $<1$\% in size of the teacher model in terms of parameters and uses $\sim 0.5$\textperthousand~FLOPs of that in the teacher model. Our code will be made public during publication. 

**Abstract (ZH)**: 社会机器人和代理与人类交互和辅助的需求正在稳步增长。为了能够成功地与人类交互，它们需要从自身（机器人）的视角理解并分析社会互动场景。现有模型中对人类与代理之间社会情境的建模较少，而现有的模型往往因计算量大而不适合在实时或信息有限的真实场景中部署。我们提出了一种鲁棒的知识蒸馏框架，通过多种多模态线索建模社会互动，且在推理过程中能够应对不完整和噪声信息。我们的教师模型通过多模态输入（身体、面部和手部动作、视线、原始图像）训练，并将知识传递给仅依赖于身体姿态的学生模型。在两个公开的交互式人类-机器人数据集上的广泛实验表明，即使有高达51%的输入被破坏，学生模型在多个下游社会理解任务中的平均准确率也比相关基准高14.75%。该学生模型极为高效：参数量仅为教师模型的不到1%，运行开销约为教师模型的0.5‰。我们将在发表时公开代码。 

---
# SynSHRP2: A Synthetic Multimodal Benchmark for Driving Safety-critical Events Derived from Real-world Driving Data 

**Title (ZH)**: SynSHRP2：来自真实驾驶数据的安全关键事件合成多模态基准 

**Authors**: Liang Shi, Boyu Jiang, Zhenyuan Yuan, Miguel A. Perez, Feng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.06276)  

**Abstract**: Driving-related safety-critical events (SCEs), including crashes and near-crashes, provide essential insights for the development and safety evaluation of automated driving systems. However, two major challenges limit their accessibility: the rarity of SCEs and the presence of sensitive privacy information in the data. The Second Strategic Highway Research Program (SHRP 2) Naturalistic Driving Study (NDS), the largest NDS to date, collected millions of hours of multimodal, high-resolution, high-frequency driving data from thousands of participants, capturing thousands of SCEs. While this dataset is invaluable for safety research, privacy concerns and data use restrictions significantly limit public access to the raw data. To address these challenges, we introduce SynSHRP2, a publicly available, synthetic, multimodal driving dataset containing over 1874 crashes and 6924 near-crashes derived from the SHRP 2 NDS. The dataset features de-identified keyframes generated using Stable Diffusion and ControlNet, ensuring the preservation of critical safety-related information while eliminating personally identifiable data. Additionally, SynSHRP2 includes detailed annotations on SCE type, environmental and traffic conditions, and time-series kinematic data spanning 5 seconds before and during each event. Synchronized keyframes and narrative descriptions further enhance its usability. This paper presents two benchmarks for event attribute classification and scene understanding, demonstrating the potential applications of SynSHRP2 in advancing safety research and automated driving system development. 

**Abstract (ZH)**: 驾驶相关的安全关键事件（SCEs），包括碰撞和接近碰撞事件，为自动驾驶系统的发展和安全评估提供了重要的见解。然而，两个主要挑战限制了这些事件的访问性：SCEs的稀有性和数据中的敏感隐私信息。第二代战略公路研究计划（SHRP 2）自然驾驶研究（NDS），迄今为止最大的NDS，收集了数千名参与者数百万小时的多模态、高分辨率、高频率驾驶数据，捕捉了数千个SCEs。尽管该数据集对安全研究至关重要，但隐私问题和数据使用限制显著限制了原始数据的公开访问。为应对这些挑战，我们引入了SynSHRP2，这是一个包含超过1874起碰撞和6924起接近碰撞事件的公开可用、合成的多模态驾驶数据集，源自SHRP 2 NDS。该数据集通过使用Stable Diffusion和ControlNet生成脱敏关键帧，确保保留了关键安全相关的信息，同时消除了可识别的个人数据。此外，SynSHRP2还包括关于事件类型、环境和交通条件以及每个事件前5秒和期间的时间序列动力学数据的详细标注。同步的关键帧和叙述性描述进一步提高了其可用性。本文提出了两种事件属性分类和场景理解的基准，展示了SynSHRP2在推进安全研究和自动驾驶系统开发中潜在应用的可能性。 

---
# Towards Accurate State Estimation: Kalman Filter Incorporating Motion Dynamics for 3D Multi-Object Tracking 

**Title (ZH)**: 基于运动动力学融合的卡尔曼滤波器用于三维多目标跟踪的精确状态估计 

**Authors**: Mohamed Nagy, Naoufel Werghi, Bilal Hassan, Jorge Dias, Majid Khonji  

**Link**: [PDF](https://arxiv.org/pdf/2505.07254)  

**Abstract**: This work addresses the critical lack of precision in state estimation in the Kalman filter for 3D multi-object tracking (MOT) and the ongoing challenge of selecting the appropriate motion model. Existing literature commonly relies on constant motion models for estimating the states of objects, neglecting the complex motion dynamics unique to each object. Consequently, trajectory division and imprecise object localization arise, especially under occlusion conditions. The core of these challenges lies in the limitations of the current Kalman filter formulation, which fails to account for the variability of motion dynamics as objects navigate their environments. This work introduces a novel formulation of the Kalman filter that incorporates motion dynamics, allowing the motion model to adaptively adjust according to changes in the object's movement. The proposed Kalman filter substantially improves state estimation, localization, and trajectory prediction compared to the traditional Kalman filter. This is reflected in tracking performance that surpasses recent benchmarks on the KITTI and Waymo Open Datasets, with margins of 0.56\% and 0.81\% in higher order tracking accuracy (HOTA) and multi-object tracking accuracy (MOTA), respectively. Furthermore, the proposed Kalman filter consistently outperforms the baseline across various detectors. Additionally, it shows an enhanced capability in managing long occlusions compared to the baseline Kalman filter, achieving margins of 1.22\% in higher order tracking accuracy (HOTA) and 1.55\% in multi-object tracking accuracy (MOTA) on the KITTI dataset. The formulation's efficiency is evident, with an additional processing time of only approximately 0.078 ms per frame, ensuring its applicability in real-time applications. 

**Abstract (ZH)**: 针对三维多目标跟踪中的卡尔曼滤波器状态估计精度不足和适用运动模型选择的持续挑战，本文提出了一种新的卡尔曼滤波器公式，该公式整合了运动动力学，使运动模型能够根据目标运动变化进行自适应调整。所提出的卡尔曼滤波器在状态估计、定位和轨迹预测方面明显优于传统的卡尔曼滤波器，并在Kitti和Waymo开放数据集上的跟踪性能上超越了最近的基准，分别在高阶跟踪准确性(HOTA)和多目标跟踪准确性(MOTA)方面提高了0.56%和0.81%。此外，所提出的卡尔曼滤波器在各种检测器上的性能持续优于基准，特别是在管理长时间遮挡方面表现更优，在Kitti数据集上的HOTA和MOTA分别提高了1.22%和1.55%。该公式效率高，每帧额外处理时间为约0.078 ms，确保其适用于实时应用。 

---
# Language-Driven Dual Style Mixing for Single-Domain Generalized Object Detection 

**Title (ZH)**: 语言驱动的双风格混合单域泛化对象检测 

**Authors**: Hongda Qin, Xiao Lu, Zhiyong Wei, Yihong Cao, Kailun Yang, Ningjiang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07219)  

**Abstract**: Generalizing an object detector trained on a single domain to multiple unseen domains is a challenging task. Existing methods typically introduce image or feature augmentation to diversify the source domain to raise the robustness of the detector. Vision-Language Model (VLM)-based augmentation techniques have been proven to be effective, but they require that the detector's backbone has the same structure as the image encoder of VLM, limiting the detector framework selection. To address this problem, we propose Language-Driven Dual Style Mixing (LDDS) for single-domain generalization, which diversifies the source domain by fully utilizing the semantic information of the VLM. Specifically, we first construct prompts to transfer style semantics embedded in the VLM to an image translation network. This facilitates the generation of style diversified images with explicit semantic information. Then, we propose image-level style mixing between the diversified images and source domain images. This effectively mines the semantic information for image augmentation without relying on specific augmentation selections. Finally, we propose feature-level style mixing in a double-pipeline manner, allowing feature augmentation to be model-agnostic and can work seamlessly with the mainstream detector frameworks, including the one-stage, two-stage, and transformer-based detectors. Extensive experiments demonstrate the effectiveness of our approach across various benchmark datasets, including real to cartoon and normal to adverse weather tasks. The source code and pre-trained models will be publicly available at this https URL. 

**Abstract (ZH)**: 单域到多未见域的物体检测通用化是一项具有挑战性的工作。现有方法通常通过图像或特征增强来多样化源域，以提高检测器的鲁棒性。基于视觉-语言模型（VLM）的增强技术已被证明有效，但它们要求检测器的骨干结构与VLM的图像编码器结构相同，限制了检测器框架的选择。为了解决这一问题，我们提出了语言驱动的双风格混合（LDDS）方法，通过充分利用VLM的语义信息来多样化源域。具体而言，我们首先构建提示，将VLM中嵌入的风格语义传递给图像翻译网络，促进了具有明确语义信息的风格多样化图像的生成。然后，我们提出了风格混合方法，在多样化图像与源域图像的图像级别之间进行风格混合，有效地挖掘用于图像增强的语义信息，而不依赖于特定的增强选择。最后，我们以双重管道的方式提出了特征级别风格混合，使得特征增强模型无关，并可以与主流的检测器框架无缝集成，包括单阶段、双阶段和基于变换器的检测器。广泛的经验表明，我们的方法在各种基准数据集中有效，包括从现实到卡通、从正常到不良天气的任务。源代码和预训练模型将在该网址公开。 

---
# Boosting Cross-spectral Unsupervised Domain Adaptation for Thermal Semantic Segmentation 

**Title (ZH)**: 跨谱域无监督领域适应的增强式热语义分割 

**Authors**: Seokjun Kwon, Jeongmin Shin, Namil Kim, Soonmin Hwang, Yukyung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06951)  

**Abstract**: In autonomous driving, thermal image semantic segmentation has emerged as a critical research area, owing to its ability to provide robust scene understanding under adverse visual conditions. In particular, unsupervised domain adaptation (UDA) for thermal image segmentation can be an efficient solution to address the lack of labeled thermal datasets. Nevertheless, since these methods do not effectively utilize the complementary information between RGB and thermal images, they significantly decrease performance during domain adaptation. In this paper, we present a comprehensive study on cross-spectral UDA for thermal image semantic segmentation. We first propose a novel masked mutual learning strategy that promotes complementary information exchange by selectively transferring results between each spectral model while masking out uncertain regions. Additionally, we introduce a novel prototypical self-supervised loss designed to enhance the performance of the thermal segmentation model in nighttime scenarios. This approach addresses the limitations of RGB pre-trained networks, which cannot effectively transfer knowledge under low illumination due to the inherent constraints of RGB sensors. In experiments, our method achieves higher performance over previous UDA methods and comparable performance to state-of-the-art supervised methods. 

**Abstract (ZH)**: 自主驾驶中跨光谱领域适应的热图像语义分割研究 

---
# Beyond Patterns: Harnessing Causal Logic for Autonomous Driving Trajectory Prediction 

**Title (ZH)**: 超越模式：利用因果逻辑进行自动驾驶轨迹预测 

**Authors**: Bonan Wang, Haicheng Liao, Chengyue Wang, Bin Rao, Yanchen Guan, Guyang Yu, Jiaxun Zhang, Songning Lai, Chengzhong Xu, Zhenning Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06856)  

**Abstract**: Accurate trajectory prediction has long been a major challenge for autonomous driving (AD). Traditional data-driven models predominantly rely on statistical correlations, often overlooking the causal relationships that govern traffic behavior. In this paper, we introduce a novel trajectory prediction framework that leverages causal inference to enhance predictive robustness, generalization, and accuracy. By decomposing the environment into spatial and temporal components, our approach identifies and mitigates spurious correlations, uncovering genuine causal relationships. We also employ a progressive fusion strategy to integrate multimodal information, simulating human-like reasoning processes and enabling real-time inference. Evaluations on five real-world datasets--ApolloScape, nuScenes, NGSIM, HighD, and MoCAD--demonstrate our model's superiority over existing state-of-the-art (SOTA) methods, with improvements in key metrics such as RMSE and FDE. Our findings highlight the potential of causal reasoning to transform trajectory prediction, paving the way for robust AD systems. 

**Abstract (ZH)**: 准确的轨迹预测一直是自动驾驶（AD）领域的重大挑战。传统的数据驱动模型主要依赖统计相关性，常常忽视交通行为背后的因果关系。本文介绍了一种新的轨迹预测框架，该框架利用因果推理来增强预测稳健性、泛化能力和准确性。通过将环境分解为空间和时间组件，我们的方法识别并缓解了虚假的相关性，揭示了真正的因果关系。我们还采用逐步融合策略来整合多模态信息，模拟人类推理过程，实现实时推理。在ApolloScape、nuScenes、NGSIM、HighD和MoCAD五个真实世界数据集上的评估结果表明，我们的模型优于现有的最先进的（SOTA）方法，关键指标如RMSE和FDE取得了改进。我们的研究结果强调了因果推理在轨迹预测中的潜力，为构建 robust AD 系统开辟了道路。 

---
# Investigating Robotaxi Crash Severity Using Geographical Random Forest 

**Title (ZH)**: 基于地理随机森林探讨Robotaxi碰撞严重性 

**Authors**: Junfeng Jiao, Seung Gyu Baik, Seung Jun Choi, Yiming Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06762)  

**Abstract**: This paper quantitatively investigates the crash severity of Autonomous Vehicles (AVs) with spatially localized machine learning and macroscopic measures of the urban built environment. We address spatial heterogeneity and spatial autocorrelation, while focusing on land use patterns and human behavior. Our Geographical Random Forest (GRF) model, accompanied with a crash severity risk map of San Francisco, presents three findings that are useful for commercial operations of AVs and robotaxis. First, spatially localized machine learning performed better than regular machine learning, when predicting AV crash severity. Bias-variance tradeoff was evident as we adjust the localization weight hyperparameter. Second, land use was the most important built environment measure, compared to intersections, building footprints, public transit stops, and Points Of Interests (POIs). Third, it was predicted that city center areas with greater diversity and commercial activities were more likely to result in low-severity AV crashes, than residential neighborhoods. Residential land use may be associated with higher severity due to human behavior and less restrictive environment. This paper recommends to explicitly consider geographic locations, and to design safety measures specific to residential neighborhoods, when robotaxi operators train their AV systems. 

**Abstract (ZH)**: 本文利用空间局部化机器学习和宏观数字城市的测量方法，定量研究自动驾驶车辆（AVs）的碰撞 severity，并关注土地使用模式和人类行为。我们的地理随机森林（GRF）模型结合旧金山的碰撞 severity 风险地图展现了三条对自动驾驶车辆商用运营有用的发现：首先，空间局部化机器学习在预测AV碰撞 severity 方面优于常规机器学习，且在调整局部化权重超参数时显示了偏差-方差权衡。其次，在土地使用、交叉口、建筑物足迹、公共交通站点和兴趣点（POIs）之间，土地使用是最为重要的城市建成环境指标。第三，市中心具有更高多样性和商业活动的区域比住宅区更有可能导致低 severity 的AV碰撞。住宅土地使用可能因人类行为和较少限制的环境而导致更高 severity。本文建议，在自动驾驶出租车运营商训练其AV系统时，需明确考虑地理位置，并设计针对住宅区的特定安全措施。 

---
# Work in Progress: Middleware-Transparent Callback Enforcement in Commoditized Component-Oriented Real-time Systems 

**Title (ZH)**: 工作中的进展：中间件透明的回调强制执行在商品化面向组件的实时系统中 

**Authors**: Takahiro Ishikawa-Aso, Atsushi Yano, Takuya Azumi, Shinpei Kato  

**Link**: [PDF](https://arxiv.org/pdf/2505.06546)  

**Abstract**: Real-time scheduling in commoditized component-oriented real-time systems, such as ROS 2 systems on Linux, has been studied under nested scheduling: OS thread scheduling and middleware layer scheduling (e.g., ROS 2 Executor). However, by establishing a persistent one-to-one correspondence between callbacks and OS threads, we can ignore the middleware layer and directly apply OS scheduling parameters (e.g., scheduling policy, priority, and affinity) to individual callbacks. We propose a middleware model that enables this idea and implements CallbackIsolatedExecutor as a novel ROS 2 Executor. We demonstrate that the costs (user-kernel switches, context switches, and memory usage) of CallbackIsolatedExecutor remain lower than those of the MultiThreadedExecutor, regardless of the number of callbacks. Additionally, the cost of CallbackIsolatedExecutor relative to SingleThreadedExecutor stays within a fixed ratio (1.4x for inter-process and 5x for intra-process communication). Future ROS 2 real-time scheduling research can avoid nested scheduling, ignoring the existence of the middleware layer. 

**Abstract (ZH)**: 面向组件导向实时系统的实时调度：在Linux上的ROS 2系统中，通过建立回调与操作系统线程之间持久的一对一对应关系，我们可以在忽略中间件层的基础上，直接将操作系统调度参数（如调度策略、优先级和亲和性）应用于个体回调。我们提出了一种中间件模型，并实现了CallbackIsolatedExecutor作为新型ROS 2执行器。实验表明，无论回调的数量如何，CallbackIsolatedExecutor的成本（用户内核切换、上下文切换和内存使用）都低于MultiThreadedExecutor的成本。此外，相对于SingleThreadedExecutor，CallbackIsolatedExecutor的成本保持在一个固定的比率之内（跨进程通信为1.4倍，进程内通信为5倍）。未来的ROS 2实时调度研究可以避免使用嵌套调度，忽略中间件层的存在。 

---
# Edge-Enabled VIO with Long-Tracked Features for High-Accuracy Low-Altitude IoT Navigation 

**Title (ZH)**: 基于边缘计算的长跟踪特征高精度低altitude物联网导航视觉惯性导航 

**Authors**: Xiaohong Huang, Cui Yang, Miaowen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06517)  

**Abstract**: This paper presents a visual-inertial odometry (VIO) method using long-tracked features. Long-tracked features can constrain more visual frames, reducing localization drift. However, they may also lead to accumulated matching errors and drift in feature tracking. Current VIO methods adjust observation weights based on re-projection errors, yet this approach has flaws. Re-projection errors depend on estimated camera poses and map points, so increased errors might come from estimation inaccuracies, not actual feature tracking errors. This can mislead the optimization process and make long-tracked features ineffective for suppressing localization drift. Furthermore, long-tracked features constrain a larger number of frames, which poses a significant challenge to real-time performance of the system. To tackle these issues, we propose an active decoupling mechanism for accumulated errors in long-tracked feature utilization. We introduce a visual reference frame reset strategy to eliminate accumulated tracking errors and a depth prediction strategy to leverage the long-term constraint. To ensure real time preformane, we implement three strategies for efficient system state estimation: a parallel elimination strategy based on predefined elimination order, an inverse-depth elimination simplification strategy, and an elimination skipping strategy. Experiments on various datasets show that our method offers higher positioning accuracy with relatively short consumption time, making it more suitable for edge-enabled low-altitude IoT navigation, where high-accuracy positioning and real-time operation on edge device are required. The code will be published at github. 

**Abstract (ZH)**: 基于长踪迹特征的视觉惯性里程计方法 

---
# Video-Enhanced Offline Reinforcement Learning: A Model-Based Approach 

**Title (ZH)**: 视频增强的离线强化学习：一种模型导向的方法 

**Authors**: Minting Pan, Yitao Zheng, Jiajian Li, Yunbo Wang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06482)  

**Abstract**: Offline reinforcement learning (RL) enables policy optimization in static datasets, avoiding the risks and costs of real-world exploration. However, it struggles with suboptimal behavior learning and inaccurate value estimation due to the lack of environmental interaction. In this paper, we present Video-Enhanced Offline RL (VeoRL), a model-based approach that constructs an interactive world model from diverse, unlabeled video data readily available online. Leveraging model-based behavior guidance, VeoRL transfers commonsense knowledge of control policy and physical dynamics from natural videos to the RL agent within the target domain. Our method achieves substantial performance gains (exceeding 100% in some cases) across visuomotor control tasks in robotic manipulation, autonomous driving, and open-world video games. 

**Abstract (ZH)**: 视频增强离线强化学习（VeoRL） 

---
# Direct Data Driven Control Using Noisy Measurements 

**Title (ZH)**: 直接基于 noisy 测量的数据驱动控制 

**Authors**: Ramin Esmzad, Gokul S. Sankar, Teawon Han, Hamidreza Modares  

**Link**: [PDF](https://arxiv.org/pdf/2505.06407)  

**Abstract**: This paper presents a novel direct data-driven control framework for solving the linear quadratic regulator (LQR) under disturbances and noisy state measurements. The system dynamics are assumed unknown, and the LQR solution is learned using only a single trajectory of noisy input-output data while bypassing system identification. Our approach guarantees mean-square stability (MSS) and optimal performance by leveraging convex optimization techniques that incorporate noise statistics directly into the controller synthesis. First, we establish a theoretical result showing that the MSS of an uncertain data-driven system implies the MSS of the true closed-loop system. Building on this, we develop a robust stability condition using linear matrix inequalities (LMIs) that yields a stabilizing controller gain from noisy measurements. Finally, we formulate a data-driven LQR problem as a semidefinite program (SDP) that computes an optimal gain, minimizing the steady-state covariance. Extensive simulations on benchmark systems -- including a rotary inverted pendulum and an active suspension system -- demonstrate the superior robustness and accuracy of our method compared to existing data-driven LQR approaches. The proposed framework offers a practical and theoretically grounded solution for controller design in noise-corrupted environments where system identification is infeasible. 

**Abstract (ZH)**: 基于直接数据驱动控制的鲁棒线性二次调节器方法：处理干扰和噪声状态测量的新型框架 

---
