# DemoDiffusion: One-Shot Human Imitation using pre-trained Diffusion Policy 

**Title (ZH)**: DemoDiffusion：基于预训练扩散策略的一次性人体模仿 

**Authors**: Sungjae Park, Homanga Bharadhwaj, Shubham Tulsiani  

**Link**: [PDF](https://arxiv.org/pdf/2506.20668)  

**Abstract**: We propose DemoDiffusion, a simple and scalable method for enabling robots to perform manipulation tasks in natural environments by imitating a single human demonstration. Our approach is based on two key insights. First, the hand motion in a human demonstration provides a useful prior for the robot's end-effector trajectory, which we can convert into a rough open-loop robot motion trajectory via kinematic retargeting. Second, while this retargeted motion captures the overall structure of the task, it may not align well with plausible robot actions in-context. To address this, we leverage a pre-trained generalist diffusion policy to modify the trajectory, ensuring it both follows the human motion and remains within the distribution of plausible robot actions. Our approach avoids the need for online reinforcement learning or paired human-robot data, enabling robust adaptation to new tasks and scenes with minimal manual effort. Experiments in both simulation and real-world settings show that DemoDiffusion outperforms both the base policy and the retargeted trajectory, enabling the robot to succeed even on tasks where the pre-trained generalist policy fails entirely. Project page: this https URL 

**Abstract (ZH)**: 我们提出了一种名为DemoDiffusion的简单可扩展方法，通过模仿单个人类示范，使机器人能够在自然环境中执行操作任务。我们的方法基于两个关键见解。首先，人类示范中的手部运动为机器人的末端执行器轨迹提供了有用的先验知识，我们可以通过运动目标转换将其转换为粗糙的开环机器人运动轨迹。其次，虽然这种目标转换的运动捕捉了任务的整体结构，但在具体情境下可能不符合合理的机器人动作。为此，我们利用预先训练的一般扩散策略来修改轨迹，确保其既能遵循人类运动，又能保持在合理的机器人动作分布之内。我们的方法避免了在线强化学习或人-机器人配对数据的需求，使得机器人能够在最少的人工努力下，对新任务和场景进行稳健的适应。实验结果表明，DemoDiffusion在仿真和真实环境设置中均优于基线策略和目标转换轨迹，即使在预训练的一般策略完全失败的任务中，也能使机器人成功执行。项目页面: 这里 

---
# A Computationally Aware Multi Objective Framework for Camera LiDAR Calibration 

**Title (ZH)**: 一种 Awareness 计算的多目标框架用于相机 LiDAR 校准 

**Authors**: Venkat Karramreddy, Rangarajan Ramanujam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20636)  

**Abstract**: Accurate extrinsic calibration between LiDAR and camera sensors is important for reliable perception in autonomous systems. In this paper, we present a novel multi-objective optimization framework that jointly minimizes the geometric alignment error and computational cost associated with camera-LiDAR calibration. We optimize two objectives: (1) error between projected LiDAR points and ground-truth image edges, and (2) a composite metric for computational cost reflecting runtime and resource usage. Using the NSGA-II \cite{deb2002nsga2} evolutionary algorithm, we explore the parameter space defined by 6-DoF transformations and point sampling rates, yielding a well-characterized Pareto frontier that exposes trade-offs between calibration fidelity and resource efficiency. Evaluations are conducted on the KITTI dataset using its ground-truth extrinsic parameters for validation, with results verified through both multi-objective and constrained single-objective baselines. Compared to existing gradient-based and learned calibration methods, our approach demonstrates interpretable, tunable performance with lower deployment overhead. Pareto-optimal configurations are further analyzed for parameter sensitivity and innovation insights. A preference-based decision-making strategy selects solutions from the Pareto knee region to suit the constraints of the embedded system. The robustness of calibration is tested across variable edge-intensity weighting schemes, highlighting optimal balance points. Although real-time deployment on embedded platforms is deferred to future work, this framework establishes a scalable and transparent method for calibration under realistic misalignment and resource-limited conditions, critical for long-term autonomy, particularly in SAE L3+ vehicles receiving OTA updates. 

**Abstract (ZH)**: 基于多目标优化的LiDAR与摄像头传感器外参精确标定框架 

---
# Communication-Aware Map Compression for Online Path-Planning: A Rate-Distortion Approach 

**Title (ZH)**: 基于通信感知的地图压缩在在线路径规划中的率失真方法 

**Authors**: Ali Reza Pedram, Evangelos Psomiadis, Dipankar Maity, Panagiotis Tsiotras  

**Link**: [PDF](https://arxiv.org/pdf/2506.20579)  

**Abstract**: This paper addresses the problem of collaborative navigation in an unknown environment, where two robots, referred to in the sequel as the Seeker and the Supporter, traverse the space simultaneously. The Supporter assists the Seeker by transmitting a compressed representation of its local map under bandwidth constraints to support the Seeker's path-planning task. We introduce a bit-rate metric based on the expected binary codeword length to quantify communication cost. Using this metric, we formulate the compression design problem as a rate-distortion optimization problem that determines when to communicate, which regions of the map should be included in the compressed representation, and at what resolution (i.e., quantization level) they should be encoded. Our formulation allows different map regions to be encoded at varying quantization levels based on their relevance to the Seeker's path-planning task. We demonstrate that the resulting optimization problem is convex, and admits a closed-form solution known in the information theory literature as reverse water-filling, enabling efficient, low-computation, and real-time implementation. Additionally, we show that the Seeker can infer the compression decisions of the Supporter independently, requiring only the encoded map content and not the encoding policy itself to be transmitted, thereby reducing communication overhead. Simulation results indicate that our method effectively constructs compressed, task-relevant map representations, both in content and resolution, that guide the Seeker's planning decisions even under tight bandwidth limitations. 

**Abstract (ZH)**: 本文解决了在未知环境中两个机器人协同导航的问题，这两个机器人分别称为搜索者和支援者，同时穿越空间。支援者在带宽限制下通过传输其局部地图的压缩表示来协助搜索者完成路径规划任务。我们引入了一个基于预期二进制码字长度的比特率度量来量化通信成本。使用该度量，我们将压缩设计问题表述为一种率失真优化问题，确定何时通信，哪些地图区域应包含在压缩表示中，以及它们应以何种分辨率（即量化水平）编码。我们的表述允许根据地图区域对搜索者路径规划任务的相关性，以不同的量化水平对不同的地图区域进行编码。我们证明，所得到的优化问题是凸优化问题，并在信息理论文献中具有解析解，称为逆水填满法，这使得其能够高效、低计算量和实时实现。此外，我们展示了搜索者可以独立推断支援者的压缩决策，只需要传输编码地图内容而非编码策略本身，从而降低通信开销。仿真结果表明，在带宽限制严格的条件下，我们的方法能够有效构建具有任务相关性的压缩地图表示，在内容和分辨率上均能指导搜索者的规划决策。 

---
# HRIBench: Benchmarking Vision-Language Models for Real-Time Human Perception in Human-Robot Interaction 

**Title (ZH)**: HRIBench: 用于人类-机器人交互实时视觉-语言模型感知基准测试 

**Authors**: Zhonghao Shi, Enyu Zhao, Nathaniel Dennler, Jingzhen Wang, Xinyang Xu, Kaleen Shrestha, Mengxue Fu, Daniel Seita, Maja Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2506.20566)  

**Abstract**: Real-time human perception is crucial for effective human-robot interaction (HRI). Large vision-language models (VLMs) offer promising generalizable perceptual capabilities but often suffer from high latency, which negatively impacts user experience and limits VLM applicability in real-world scenarios. To systematically study VLM capabilities in human perception for HRI and performance-latency trade-offs, we introduce HRIBench, a visual question-answering (VQA) benchmark designed to evaluate VLMs across a diverse set of human perceptual tasks critical for HRI. HRIBench covers five key domains: (1) non-verbal cue understanding, (2) verbal instruction understanding, (3) human-robot object relationship understanding, (4) social navigation, and (5) person identification. To construct HRIBench, we collected data from real-world HRI environments to curate questions for non-verbal cue understanding, and leveraged publicly available datasets for the remaining four domains. We curated 200 VQA questions for each domain, resulting in a total of 1000 questions for HRIBench. We then conducted a comprehensive evaluation of both state-of-the-art closed-source and open-source VLMs (N=11) on HRIBench. Our results show that, despite their generalizability, current VLMs still struggle with core perceptual capabilities essential for HRI. Moreover, none of the models within our experiments demonstrated a satisfactory performance-latency trade-off suitable for real-time deployment, underscoring the need for future research on developing smaller, low-latency VLMs with improved human perception capabilities. HRIBench and our results can be found in this Github repository: this https URL. 

**Abstract (ZH)**: 实时人类感知对于有效的机器人人类交互（HRI）至关重要。大型视觉-语言模型（VLMs）提供了广泛适用的感知能力，但往往遭受高延迟的影响，这会负面影响用户体验，并限制VLM在实际场景中的应用。为了系统地研究VLM在HRI中的人类感知能力及其性能-延迟权衡，我们引入了HRIBench这一视觉问答（VQA）基准，旨在评估VLM在HRI中关键的人类感知任务上的表现。HRIBench涵盖了五个关键领域：（1）非语言暗示理解，（2）口头指示理解，（3）人机器人物体关系理解，（4）社会导航，（5）人物识别。为了构建HRIBench，我们从实际的HRI环境收集数据以构建非语言暗示领域的问答问题，并利用公开可用的数据集构建其余四个领域。我们为每个领域构建了200个视觉问答问题，总共构建了1000个问题。然后，我们在HRIBench上对11个最先进的商用闭源和开源VLM进行了全面评估。结果显示，尽管VLMs具有普遍适用性，但它们仍然在HRI中核心的感知能力方面存在不足。此外，我们实验中的所有模型均未表现出适合实时部署的令人满意的性能-延迟权衡，突显了未来研究开发更小的低延迟、具有改进的人类感知能力的VLM的重要性。HRIBench和我们的结果可在以下GitHub仓库中找到：this https URL。 

---
# Leveraging Correlation Across Test Platforms for Variance-Reduced Metric Estimation 

**Title (ZH)**: 利用测试平台间的相关性进行方差减小的度量估计 

**Authors**: Rachel Luo, Heng Yang, Michael Watson, Apoorva Sharma, Sushant Veer, Edward Schmerling, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2506.20553)  

**Abstract**: Learning-based robotic systems demand rigorous validation to assure reliable performance, but extensive real-world testing is often prohibitively expensive, and if conducted may still yield insufficient data for high-confidence guarantees. In this work, we introduce a general estimation framework that leverages paired data across test platforms, e.g., paired simulation and real-world observations, to achieve better estimates of real-world metrics via the method of control variates. By incorporating cheap and abundant auxiliary measurements (for example, simulator outputs) as control variates for costly real-world samples, our method provably reduces the variance of Monte Carlo estimates and thus requires significantly fewer real-world samples to attain a specified confidence bound on the mean performance. We provide theoretical analysis characterizing the variance and sample-efficiency improvement, and demonstrate empirically in autonomous driving and quadruped robotics settings that our approach achieves high-probability bounds with markedly improved sample efficiency. Our technique can lower the real-world testing burden for validating the performance of the stack, thereby enabling more efficient and cost-effective experimental evaluation of robotic systems. 

**Abstract (ZH)**: 基于学习的机器人系统需要严格的验证以确保可靠的性能，但广泛的实地测试往往代价高昂，即便进行了也可能数据不足无法提供高可信度的保证。本文引入了一种一般性的估计框架，利用跨测试平台的配对数据，例如仿真实验和实地观察的配对数据，通过控制变异量的方法获得更好的实地指标估计。通过将廉价且丰富的辅助测量（例如仿真器输出）作为昂贵实地样本的控制变异量，我们的方法可证明地降低了蒙特卡洛估计的方差，从而显著减少达到指定置信水平所需的实地样本数量。我们提供了理论分析，描述了方差和样本效率的改进，并在自动驾驶和四足机器人设置中通过实验证明，我们的方法在显著提高样本效率的同时实现了高概率的置信边界。该技术可以降低验证系统性能所需的实地测试负担，从而使得机器人系统的实验评估更加高效和成本效益高。 

---
# Critical Anatomy-Preserving & Terrain-Augmenting Navigation (CAPTAiN): Application to Laminectomy Surgical Education 

**Title (ZH)**: 保留关键解剖结构并增强地形导航（CAPTAiN）：应用于腰椎间融合手术教育 

**Authors**: Jonathan Wang, Hisashi Ishida, David Usevitch, Kesavan Venkatesh, Yi Wang, Mehran Armand, Rachel Bronheim, Amit Jain, Adnan Munawar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20496)  

**Abstract**: Surgical training remains a crucial milestone in modern medicine, with procedures such as laminectomy exemplifying the high risks involved. Laminectomy drilling requires precise manual control to mill bony tissue while preserving spinal segment integrity and avoiding breaches in the dura: the protective membrane surrounding the spinal cord. Despite unintended tears occurring in up to 11.3% of cases, no assistive tools are currently utilized to reduce this risk. Variability in patient anatomy further complicates learning for novice surgeons. This study introduces CAPTAiN, a critical anatomy-preserving and terrain-augmenting navigation system that provides layered, color-coded voxel guidance to enhance anatomical awareness during spinal drilling. CAPTAiN was evaluated against a standard non-navigated approach through 110 virtual laminectomies performed by 11 orthopedic residents and medical students. CAPTAiN significantly improved surgical completion rates of target anatomy (87.99% vs. 74.42%) and reduced cognitive load across multiple NASA-TLX domains. It also minimized performance gaps across experience levels, enabling novices to perform on par with advanced trainees. These findings highlight CAPTAiN's potential to optimize surgical execution and support skill development across experience levels. Beyond laminectomy, it demonstrates potential for broader applications across various surgical and drilling procedures, including those in neurosurgery, otolaryngology, and other medical fields. 

**Abstract (ZH)**: 保留关键解剖结构和增强地形导航系统在脊柱钻孔中的应用：CAPTAiN在脊柱手术训练中的评估与前景 

---
# Behavior Foundation Model: Towards Next-Generation Whole-Body Control System of Humanoid Robots 

**Title (ZH)**: 行为基础模型：面向下一代类人机器人全身控制系统的探索 

**Authors**: Mingqi Yuan, Tao Yu, Wenqi Ge, Xiuyong Yao, Dapeng Li, Huijiang Wang, Jiayu Chen, Xin Jin, Bo Li, Hua Chen, Wei Zhang, Wenjun Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.20487)  

**Abstract**: Humanoid robots are drawing significant attention as versatile platforms for complex motor control, human-robot interaction, and general-purpose physical intelligence. However, achieving efficient whole-body control (WBC) in humanoids remains a fundamental challenge due to sophisticated dynamics, underactuation, and diverse task requirements. While learning-based controllers have shown promise for complex tasks, their reliance on labor-intensive and costly retraining for new scenarios limits real-world applicability. To address these limitations, behavior(al) foundation models (BFMs) have emerged as a new paradigm that leverages large-scale pretraining to learn reusable primitive skills and behavioral priors, enabling zero-shot or rapid adaptation to a wide range of downstream tasks. In this paper, we present a comprehensive overview of BFMs for humanoid WBC, tracing their development across diverse pre-training pipelines. Furthermore, we discuss real-world applications, current limitations, urgent challenges, and future opportunities, positioning BFMs as a key approach toward scalable and general-purpose humanoid intelligence. Finally, we provide a curated and long-term list of BFM papers and projects to facilitate more subsequent research, which is available at this https URL. 

**Abstract (ZH)**: 类人机器人作为复杂运动控制、人机交互和通用物理智能的多功能平台正吸引着广泛关注。然而，由于复杂的动力学、欠驱动以及多样的任务要求，实现高效的整体身体控制（WBC）仍然是一个根本性的挑战。虽然基于学习的控制器在复杂任务中显示出潜力，但它们依赖于劳动密集型且昂贵的新场景重新训练限制了其实用性。为了解决这些局限性，行为基础模型（BFMs）作为一种新的范式出现，利用大规模预训练学习可重用的基本技能和行为先验，从而能够对各种下游任务实现零样本或快速适应。在本文中，我们提供了一种全面的BFMs概述，追溯了它们在不同预训练管道中的发展过程。此外，我们还讨论了BFMs的实际应用、当前局限性、紧迫挑战和未来机会，将BFMs定位为走向可扩展和通用型类人智能的关键方法。最后，我们提供了一份精心策划的长期BFMs论文和项目列表，以促进后续研究，该列表可在以下链接访问：this https URL。 

---
# EANS: Reducing Energy Consumption for UAV with an Environmental Adaptive Navigation Strategy 

**Title (ZH)**: EANS: 一种环境自适应导航策略降低无人机能耗 

**Authors**: Tian Liu, Han Liu, Boyang Li, Long Chen, Kai Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20485)  

**Abstract**: Unmanned Aerial Vehicles (UAVS) are limited by the onboard energy. Refinement of the navigation strategy directly affects both the flight velocity and the trajectory based on the adjustment of key parameters in the UAVS pipeline, thus reducing energy consumption. However, existing techniques tend to adopt static and conservative strategies in dynamic scenarios, leading to inefficient energy reduction. Dynamically adjusting the navigation strategy requires overcoming the challenges including the task pipeline interdependencies, the environmental-strategy correlations, and the selecting parameters. To solve the aforementioned problems, this paper proposes a method to dynamically adjust the navigation strategy of the UAVS by analyzing its dynamic characteristics and the temporal characteristics of the autonomous navigation pipeline, thereby reducing UAVS energy consumption in response to environmental changes. We compare our method with the baseline through hardware-in-the-loop (HIL) simulation and real-world experiments, showing our method 3.2X and 2.6X improvements in mission time, 2.4X and 1.6X improvements in energy, respectively. 

**Abstract (ZH)**: 无人驾驶飞行器（UAV）受载荷能量限制。导航策略的优化直接影响飞行速度和轨迹，通过调整UAV管道中的关键参数来实现，从而减少能量消耗。然而，现有技术在动态场景中倾向于采用静态和保守策略，导致能量减少效率低下。动态调整导航策略需要克服任务管道相互依赖性、环境-策略相关性以及参数选择等挑战。为解决上述问题，本文提出了一种方法，通过分析UAV自主导航管道的动力学特性和时间特性来动态调整其导航策略，从而在环境变化时减少UAV的能量消耗。通过硬件在环（HIL）模拟和实际实验将我们的方法与基线进行比较，结果显示，分别在任务时间上提高了3.2倍、能源上提高了2.4倍和1.6倍。 

---
# A Review of Personalisation in Human-Robot Collaboration and Future Perspectives Towards Industry 5.0 

**Title (ZH)**: 人类与机器人协作中的个性化综述及面向工业4.0的未来展望 

**Authors**: James Fant-Male, Roel Pieters  

**Link**: [PDF](https://arxiv.org/pdf/2506.20447)  

**Abstract**: The shift in research focus from Industry 4.0 to Industry 5.0 (I5.0) promises a human-centric workplace, with social and well-being values at the centre of technological implementation. Human-Robot Collaboration (HRC) is a core aspect of I5.0 development, with an increase in adaptive and personalised interactions and behaviours. This review investigates recent advancements towards personalised HRC, where user-centric adaption is key. There is a growing trend for adaptable HRC research, however there lacks a consistent and unified approach. The review highlights key research trends on which personal factors are considered, workcell and interaction design, and adaptive task completion. This raises various key considerations for future developments, particularly around the ethical and regulatory development of personalised systems, which are discussed in detail. 

**Abstract (ZH)**: 从 Industry 4.0 到 Industry 5.0 (I5.0) 的研究重点转移：以人类为中心的工作场所及其技术实施，强调社会和福祉价值。人机协作 (HRC) 是 I5.0 发展的核心方面，涉及适应性和个性化交互与行为的增加。本文回顾了近期朝着个性化 HRC 的进步，其中用户中心的适应性是关键。尽管可适应的 HRC 研究呈增长趋势，但缺乏一致和统一的方法。本文总结了关键的研究趋势，重点考虑个人因素、工作单元和交互设计以及适应性任务完成。这提出了对未来发展的各种关键考虑，特别是在个性化系统的伦理和法规发展方面，详细讨论了这些问题。 

---
# Learn to Position -- A Novel Meta Method for Robotic Positioning 

**Title (ZH)**: 学习定位——一种新型元学习方法用于机器人定位 

**Authors**: Dongkun Wang, Junkai Zhao, Yunfei Teng, Jieyang Peng, Wenjing Xue, Xiaoming Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20445)  

**Abstract**: Absolute positioning accuracy is a vital specification for robots. Achieving high position precision can be challenging due to the presence of various sources of errors. Meanwhile, accurately depicting these errors is difficult due to their stochastic nature. Vision-based methods are commonly integrated to guide robotic positioning, but their performance can be highly impacted by inevitable occlusions or adverse lighting conditions. Drawing on the aforementioned considerations, a vision-free, model-agnostic meta-method for compensating robotic position errors is proposed, which maximizes the probability of accurate robotic position via interactive feedback. Meanwhile, the proposed method endows the robot with the capability to learn and adapt to various position errors, which is inspired by the human's instinct for grasping under uncertainties. Furthermore, it is a self-learning and self-adaptive method able to accelerate the robotic positioning process as more examples are incorporated and learned. Empirical studies validate the effectiveness of the proposed method. As of the writing of this paper, the proposed meta search method has already been implemented in a robotic-based assembly line for odd-form electronic components. 

**Abstract (ZH)**: 基于交互反馈的无视觉模型agnostic元方法及其在机器人定位误差补偿中的应用 

---
# Multimodal Behaviour Trees for Robotic Laboratory Task Automation 

**Title (ZH)**: 多模态行为树在机器人实验室任务自动化中的应用 

**Authors**: Hatem Fakhruldeen, Arvind Raveendran Nambiar, Satheeshkumar Veeramani, Bonilkumar Vijaykumar Tailor, Hadi Beyzaee Juneghani, Gabriella Pizzuto, Andrew Ian Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2506.20399)  

**Abstract**: Laboratory robotics offer the capability to conduct experiments with a high degree of precision and reproducibility, with the potential to transform scientific research. Trivial and repeatable tasks; e.g., sample transportation for analysis and vial capping are well-suited for robots; if done successfully and reliably, chemists could contribute their efforts towards more critical research activities. Currently, robots can perform these tasks faster than chemists, but how reliable are they? Improper capping could result in human exposure to toxic chemicals which could be fatal. To ensure that robots perform these tasks as accurately as humans, sensory feedback is required to assess the progress of task execution. To address this, we propose a novel methodology based on behaviour trees with multimodal perception. Along with automating robotic tasks, this methodology also verifies the successful execution of the task, a fundamental requirement in safety-critical environments. The experimental evaluation was conducted on two lab tasks: sample vial capping and laboratory rack insertion. The results show high success rate, i.e., 88% for capping and 92% for insertion, along with strong error detection capabilities. This ultimately proves the robustness and reliability of our approach and that using multimodal behaviour trees should pave the way towards the next generation of robotic chemists. 

**Abstract (ZH)**: 实验室机器人提供高精度和可重复性的实验能力，有望变革科学研究。基于行为树的多模态感知方法在样本瓶封盖和实验室架插入等任务中的应用验证了其可靠性和鲁棒性，推动了新一代机器人化学家的发展。 

---
# SPARK: Graph-Based Online Semantic Integration System for Robot Task Planning 

**Title (ZH)**: SPARK：基于图的机器人任务规划在线语义集成系统 

**Authors**: Mimo Shirasaka, Yuya Ikeda, Tatsuya Matsushima, Yutaka Matsuo, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.20394)  

**Abstract**: The ability to update information acquired through various means online during task execution is crucial for a general-purpose service robot. This information includes geometric and semantic data. While SLAM handles geometric updates on 2D maps or 3D point clouds, online updates of semantic information remain unexplored. We attribute the challenge to the online scene graph representation, for its utility and scalability. Building on previous works regarding offline scene graph representations, we study online graph representations of semantic information in this work. We introduce SPARK: Spatial Perception and Robot Knowledge Integration. This framework extracts semantic information from environment-embedded cues and updates the scene graph accordingly, which is then used for subsequent task planning. We demonstrate that graph representations of spatial relationships enhance the robot system's ability to perform tasks in dynamic environments and adapt to unconventional spatial cues, like gestures. 

**Abstract (ZH)**: 基于各种在线手段获取的信息更新能力对于通用服务机器人在任务执行过程中至关重要。这些信息包括几何和语义数据。虽然SLAM处理二维地图或三维点云的几何更新，但语义信息的在线更新尚未受到探索。我们归因于在线场景图表示的实用性和可扩展性带来的挑战。基于关于离线场景图表示的前期工作，我们在本文中研究语义信息的在线图表示。我们引入了SPARK：空间感知与机器人知识整合框架。该框架从环境嵌入的线索中提取语义信息，并相应地更新场景图，然后用于后续任务规划。我们展示了图表示的空间关系增强了机器人系统在动态环境中执行任务并在遇到不寻常的空间线索（如手势）时进行适应的能力。 

---
# Enhanced Robotic Navigation in Deformable Environments using Learning from Demonstration and Dynamic Modulation 

**Title (ZH)**: 基于演示学习和动态调节的可变形环境增强机器人导航 

**Authors**: Lingyun Chen, Xinrui Zhao, Marcos P. S. Campanha, Alexander Wegener, Abdeldjallil Naceri, Abdalla Swikir, Sami Haddadin  

**Link**: [PDF](https://arxiv.org/pdf/2506.20376)  

**Abstract**: This paper presents a novel approach for robot navigation in environments containing deformable obstacles. By integrating Learning from Demonstration (LfD) with Dynamical Systems (DS), we enable adaptive and efficient navigation in complex environments where obstacles consist of both soft and hard regions. We introduce a dynamic modulation matrix within the DS framework, allowing the system to distinguish between traversable soft regions and impassable hard areas in real-time, ensuring safe and flexible trajectory planning. We validate our method through extensive simulations and robot experiments, demonstrating its ability to navigate deformable environments. Additionally, the approach provides control over both trajectory and velocity when interacting with deformable objects, including at intersections, while maintaining adherence to the original DS trajectory and dynamically adapting to obstacles for smooth and reliable navigation. 

**Abstract (ZH)**: 本文提出了一种新型机器人在包含可变形障碍物环境中的导航方法。通过将学习从演示（LfD）与动力学系统（DS）集成，使机器人能够在由软硬区域组成的复杂环境中实现适应性和高效的导航。在动力学系统框架中引入了动态调节矩阵，使系统能够实时区分可通行的软区域和不可通行的硬区域，从而确保安全和灵活的轨迹规划。通过广泛的仿真实验和机器人实验验证了该方法，展示了其在可变形环境中的导航能力。此外，该方法在与可变形物体交互时，包括在交叉口处，提供了对轨迹和速度的控制，同时保持对原始DS轨迹的遵从性，并动态适应障碍物以实现平滑可靠的导航。 

---
# CARMA: Context-Aware Situational Grounding of Human-Robot Group Interactions by Combining Vision-Language Models with Object and Action Recognition 

**Title (ZH)**: CARMA: 基于上下文感知情景关联的人机群体交互视觉-语言模型结合物体和动作识别 

**Authors**: Joerg Deigmoeller, Stephan Hasler, Nakul Agarwal, Daniel Tanneberg, Anna Belardinelli, Reza Ghoddoosian, Chao Wang, Felix Ocker, Fan Zhang, Behzad Dariush, Michael Gienger  

**Link**: [PDF](https://arxiv.org/pdf/2506.20373)  

**Abstract**: We introduce CARMA, a system for situational grounding in human-robot group interactions. Effective collaboration in such group settings requires situational awareness based on a consistent representation of present persons and objects coupled with an episodic abstraction of events regarding actors and manipulated objects. This calls for a clear and consistent assignment of instances, ensuring that robots correctly recognize and track actors, objects, and their interactions over time. To achieve this, CARMA uniquely identifies physical instances of such entities in the real world and organizes them into grounded triplets of actors, objects, and actions.
To validate our approach, we conducted three experiments, where multiple humans and a robot interact: collaborative pouring, handovers, and sorting. These scenarios allow the assessment of the system's capabilities as to role distinction, multi-actor awareness, and consistent instance identification. Our experiments demonstrate that the system can reliably generate accurate actor-action-object triplets, providing a structured and robust foundation for applications requiring spatiotemporal reasoning and situated decision-making in collaborative settings. 

**Abstract (ZH)**: CARMA：一种用于人类-机器人小组互动的情境 grounding 系统 

---
# PIMBS: Efficient Body Schema Learning for Musculoskeletal Humanoids with Physics-Informed Neural Networks 

**Title (ZH)**: PIMBS：基于物理信息神经网络的高效身体方案学习方法用于肌骨骼类人机器人 

**Authors**: Kento Kawaharazuka, Takahiro Hattori, Keita Yoneda, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2506.20343)  

**Abstract**: Musculoskeletal humanoids are robots that closely mimic the human musculoskeletal system, offering various advantages such as variable stiffness control, redundancy, and flexibility. However, their body structure is complex, and muscle paths often significantly deviate from geometric models. To address this, numerous studies have been conducted to learn body schema, particularly the relationships among joint angles, muscle tension, and muscle length. These studies typically rely solely on data collected from the actual robot, but this data collection process is labor-intensive, and learning becomes difficult when the amount of data is limited. Therefore, in this study, we propose a method that applies the concept of Physics-Informed Neural Networks (PINNs) to the learning of body schema in musculoskeletal humanoids, enabling high-accuracy learning even with a small amount of data. By utilizing not only data obtained from the actual robot but also the physical laws governing the relationship between torque and muscle tension under the assumption of correct joint structure, more efficient learning becomes possible. We apply the proposed method to both simulation and an actual musculoskeletal humanoid and discuss its effectiveness and characteristics. 

**Abstract (ZH)**: 基于物理约束的神经网络在肌骨骼人形机器人身体模式学习中的应用 

---
# Finding the Easy Way Through -- the Probabilistic Gap Planner for Social Robot Navigation 

**Title (ZH)**: 寻找简便之道——社会机器人导航的概率性间隙规划者 

**Authors**: Malte Probst, Raphael Wenzel, Tim Puphal, Monica Dasi, Nico A. Steinhardt, Sango Matsuzaki, Misa Komuro  

**Link**: [PDF](https://arxiv.org/pdf/2506.20320)  

**Abstract**: In Social Robot Navigation, autonomous agents need to resolve many sequential interactions with other agents. State-of-the art planners can efficiently resolve the next, imminent interaction cooperatively and do not focus on longer planning horizons. This makes it hard to maneuver scenarios where the agent needs to select a good strategy to find gaps or channels in the crowd. We propose to decompose trajectory planning into two separate steps: Conflict avoidance for finding good, macroscopic trajectories, and cooperative collision avoidance (CCA) for resolving the next interaction optimally. We propose the Probabilistic Gap Planner (PGP) as a conflict avoidance planner. PGP modifies an established probabilistic collision risk model to include a general assumption of cooperativity. PGP biases the short-term CCA planner to head towards gaps in the crowd. In extensive simulations with crowds of varying density, we show that using PGP in addition to state-of-the-art CCA planners improves the agents' performance: On average, agents keep more space to others, create less tension, and cause fewer collisions. This typically comes at the expense of slightly longer paths. PGP runs in real-time on WaPOCHI mobile robot by Honda R&D. 

**Abstract (ZH)**: 社会机器人导航中，自主代理需要解决与其它代理的许多序贯互动。最先进的规划者可以有效地协同解决即将发生的互动，但不聚焦于更长时间的规划展望。这使得在代理需要选择策略以在人群中共找到合适的空隙或通道时难以应对。我们提出将轨迹规划分解为两个单独的步骤：冲突避免以找到好的宏观轨迹，以及协同碰撞避免（CCA）以最优地解决下一个互动。我们提出了概率性间隙规划器（PGP）作为冲突避免规划器。PGP修改了现有的概率碰撞风险模型，使其包含普遍的协同假设。PGP使短期CCA规划器倾向于朝向人群中的空隙前进。在广泛且密度变化的群体仿真中，我们展示了使用PGP与最先进的CCA规划器相结合可以提高代理的表现：平均而言，代理与他人保持更大的空间，产生的紧张情绪更少，碰撞次数也更少。这通常会略微增加路径长度。PGP在本田研发的WaPOCHI移动机器人上实时运行。 

---
# Building Forest Inventories with Autonomous Legged Robots -- System, Lessons, and Challenges Ahead 

**Title (ZH)**: 使用自主腿式机器人建立森林inventory系统、经验与未来挑战 

**Authors**: Matías Mattamala, Nived Chebrolu, Jonas Frey, Leonard Freißmuth, Haedam Oh, Benoit Casseau, Marco Hutter, Maurice Fallon  

**Link**: [PDF](https://arxiv.org/pdf/2506.20315)  

**Abstract**: Legged robots are increasingly being adopted in industries such as oil, gas, mining, nuclear, and agriculture. However, new challenges exist when moving into natural, less-structured environments, such as forestry applications. This paper presents a prototype system for autonomous, under-canopy forest inventory with legged platforms. Motivated by the robustness and mobility of modern legged robots, we introduce a system architecture which enabled a quadruped platform to autonomously navigate and map forest plots. Our solution involves a complete navigation stack for state estimation, mission planning, and tree detection and trait estimation. We report the performance of the system from trials executed over one and a half years in forests in three European countries. Our results with the ANYmal robot demonstrate that we can survey plots up to 1 ha plot under 30 min, while also identifying trees with typical DBH accuracy of 2cm. The findings of this project are presented as five lessons and challenges. Particularly, we discuss the maturity of hardware development, state estimation limitations, open problems in forest navigation, future avenues for robotic forest inventory, and more general challenges to assess autonomous systems. By sharing these lessons and challenges, we offer insight and new directions for future research on legged robots, navigation systems, and applications in natural environments. Additional videos can be found in this https URL 

**Abstract (ZH)**: 基于腿式机器人在欧洲三国森林环境下自主林分调查的原型系统研究 

---
# Near Time-Optimal Hybrid Motion Planning for Timber Cranes 

**Title (ZH)**: 近时效最优混合运动规划木材起重机 

**Authors**: Marc-Philip Ecker, Bernhard Bischof, Minh Nhat Vu, Christoph Fröhlich, Tobias Glück, Wolfgang Kemmetmüller  

**Link**: [PDF](https://arxiv.org/pdf/2506.20314)  

**Abstract**: Efficient, collision-free motion planning is essential for automating large-scale manipulators like timber cranes. They come with unique challenges such as hydraulic actuation constraints and passive joints-factors that are seldom addressed by current motion planning methods. This paper introduces a novel approach for time-optimal, collision-free hybrid motion planning for a hydraulically actuated timber crane with passive joints. We enhance the via-point-based stochastic trajectory optimization (VP-STO) algorithm to include pump flow rate constraints and develop a novel collision cost formulation to improve robustness. The effectiveness of the enhanced VP-STO as an optimal single-query global planner is validated by comparison with an informed RRT* algorithm using a time-optimal path parameterization (TOPP). The overall hybrid motion planning is formed by combination with a gradient-based local planner that is designed to follow the global planner's reference and to systematically consider the passive joint dynamics for both collision avoidance and sway damping. 

**Abstract (ZH)**: 高效的、无碰撞路径规划对于自动化大规模操纵器如木材起重机至关重要。它们面临着独特的挑战，如液压驱动约束和被动关节因素，这些问题当前的路径规划方法很少予以考虑。本文提出了一种新颖的方法，用于液压驱动木材起重机（具有被动关节）的时最优、无碰撞混合路径规划。我们增强了基于途径点的随机轨迹优化（VP-STO）算法，使其包括泵流量约束，并开发了一种新的碰撞代价公式来提高鲁棒性。通过与基于启发式的RRT*算法（使用时最优路径参数化TOPP）进行比较，验证了增强的VP-STO作为最优单查询全局规划器的有效性。整体混合路径规划结合了一个基于梯度的局部规划器，该规划器设计用于跟踪全局规划器的参考，并系统地考虑被动关节动力学，以实现避碰和减摇。 

---
# Real-Time Obstacle Avoidance Algorithms for Unmanned Aerial and Ground Vehicles 

**Title (ZH)**: 实时障碍避让算法研究：无人机与地面车辆 

**Authors**: Jingwen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.20311)  

**Abstract**: The growing use of mobile robots in sectors such as automotive, agriculture, and rescue operations reflects progress in robotics and autonomy. In unmanned aerial vehicles (UAVs), most research emphasizes visual SLAM, sensor fusion, and path planning. However, applying UAVs to search and rescue missions in disaster zones remains underexplored, especially for autonomous navigation.
This report develops methods for real-time and secure UAV maneuvering in complex 3D environments, crucial during forest fires. Building upon past research, it focuses on designing navigation algorithms for unfamiliar and hazardous environments, aiming to improve rescue efficiency and safety through UAV-based early warning and rapid response.
The work unfolds in phases. First, a 2D fusion navigation strategy is explored, initially for mobile robots, enabling safe movement in dynamic settings. This sets the stage for advanced features such as adaptive obstacle handling and decision-making enhancements. Next, a novel 3D reactive navigation strategy is introduced for collision-free movement in forest fire simulations, addressing the unique challenges of UAV operations in such scenarios.
Finally, the report proposes a unified control approach that integrates UAVs and unmanned ground vehicles (UGVs) for coordinated rescue missions in forest environments. Each phase presents challenges, proposes control models, and validates them with mathematical and simulation-based evidence. The study offers practical value and academic insights for improving the role of UAVs in natural disaster rescue operations. 

**Abstract (ZH)**: 移动机器人在汽车、农业和救援操作领域的日益广泛应用反映了机器人技术和自主性的进步。在无人驾驶飞行器（UAVs）中，大多数研究集中于视觉SLAM、传感器融合和路径规划。然而，将UAV应用于灾害区搜救任务的自主导航研究仍相对较少。

本报告开发了在复杂3D环境中实时和安全的UAV机动方法，对于森林火灾等场景至关重要。在此前研究的基础上，本报告专注于设计适用于未知和危险环境的导航算法，旨在通过基于UAV的早期预警和快速响应提高救援效率和安全性。

这项工作分为几个阶段。首先，探索了一种2D融合导航策略，最初应用于移动机器人，以实现动态环境中的安全移动。这为引入高级功能，如自适应障碍物处理和决策增强奠定了基础。其次，提出了一种新颖的3D反应式导航策略，在森林火灾模拟中实现无碰撞移动，解决了无人机在这种场景下操作的独特挑战。

最后，报告提出了一种统一的控制方法，将无人机和无人驾驶地面车辆（UGVs）整合起来，为森林环境中的协同救援任务提供支持。每个阶段都提出了挑战，提出了控制模型，并通过数学和仿真证据进行了验证。本研究为改善无人机在自然灾害救援中的作用提供了实际价值和学术见解。 

---
# Why Robots Are Bad at Detecting Their Mistakes: Limitations of Miscommunication Detection in Human-Robot Dialogue 

**Title (ZH)**: 为什么机器人不擅长检测其错误：人类-机器人对话中错误沟通检测的局限性 

**Authors**: Ruben Janssens, Jens De Bock, Sofie Labat, Eva Verhelst, Veronique Hoste, Tony Belpaeme  

**Link**: [PDF](https://arxiv.org/pdf/2506.20268)  

**Abstract**: Detecting miscommunication in human-robot interaction is a critical function for maintaining user engagement and trust. While humans effortlessly detect communication errors in conversations through both verbal and non-verbal cues, robots face significant challenges in interpreting non-verbal feedback, despite advances in computer vision for recognizing affective expressions. This research evaluates the effectiveness of machine learning models in detecting miscommunications in robot dialogue. Using a multi-modal dataset of 240 human-robot conversations, where four distinct types of conversational failures were systematically introduced, we assess the performance of state-of-the-art computer vision models. After each conversational turn, users provided feedback on whether they perceived an error, enabling an analysis of the models' ability to accurately detect robot mistakes. Despite using state-of-the-art models, the performance barely exceeds random chance in identifying miscommunication, while on a dataset with more expressive emotional content, they successfully identified confused states. To explore the underlying cause, we asked human raters to do the same. They could also only identify around half of the induced miscommunications, similarly to our model. These results uncover a fundamental limitation in identifying robot miscommunications in dialogue: even when users perceive the induced miscommunication as such, they often do not communicate this to their robotic conversation partner. This knowledge can shape expectations of the performance of computer vision models and can help researchers to design better human-robot conversations by deliberately eliciting feedback where needed. 

**Abstract (ZH)**: 检测人类与机器人交互中的沟通错误对于维持用户参与度和信任至关重要。尽管人类可以通过口头和非口头线索轻松检测交流错误，机器人在解读非口头反馈方面仍面临重大挑战，尽管在通过计算机视觉识别情感表达方面取得了进展。本研究评估了机器学习模型在检测机器人对话中的沟通错误方面的有效性。使用包含240轮人类与机器人对话的多模态数据集，系统地引入了四种不同类型的对话失败，评估了最先进的计算机视觉模型的性能。在每次对话回合后，用户提供了他们是否察觉到错误的反馈，从而分析了模型准确检测机器人错误的能力。尽管使用了最先进的模型，但在识别沟通错误方面的性能几乎没有超过随机猜测，在具有更多表达性情感内容的数据集上，它们能够识别出困惑状态。为了探索背后的原因，我们让人类评分者也做了同样的事情。他们也只能识别出大约一半诱导的沟通错误，与我们的模型类似。这些结果揭示了检测对话中机器人沟通错误的基本局限性：即使用户察觉到诱导的沟通错误，他们也往往不会将其传达给其机器人对话伙伴。这些知识可以塑造对计算机视觉模型性能的期望，并有助于研究人员通过适时诱发反馈来设计更好的人机对话。 

---
# Generating and Customizing Robotic Arm Trajectories using Neural Networks 

**Title (ZH)**: 使用神经网络生成和定制机器人手臂轨迹 

**Authors**: Andrej Lúčny, Matilde Antonj, Carlo Mazzola, Hana Hornáčková, Igor Farkaš  

**Link**: [PDF](https://arxiv.org/pdf/2506.20259)  

**Abstract**: We introduce a neural network approach for generating and customizing the trajectory of a robotic arm, that guarantees precision and repeatability. To highlight the potential of this novel method, we describe the design and implementation of the technique and show its application in an experimental setting of cognitive robotics. In this scenario, the NICO robot was characterized by the ability to point to specific points in space with precise linear movements, increasing the predictability of the robotic action during its interaction with humans. To achieve this goal, the neural network computes the forward kinematics of the robot arm. By integrating it with a generator of joint angles, another neural network was developed and trained on an artificial dataset created from suitable start and end poses of the robotic arm. Through the computation of angular velocities, the robot was characterized by its ability to perform the movement, and the quality of its action was evaluated in terms of shape and accuracy. Thanks to its broad applicability, our approach successfully generates precise trajectories that could be customized in their shape and adapted to different settings. 

**Abstract (ZH)**: 我们介绍了一种基于神经网络的方法，用于生成和定制机器人臂的轨迹，以保证精准性和可重复性。为了凸显该新方法的潜力，我们描述了该技术的设计与实现，并展示了其在认知 robotics 实验设置中的应用。在这种情景中，NICO 机器人通过精确的线性运动指向空间中的特定点，增加了其在与人类交互时动作的可预测性。为此目标，神经网络计算了机器人臂的正向运动学。通过将其与关节角生成器结合，我们开发并训练了一个神经网络，该网络基于从机器人臂合适起始和结束姿态中创建的虚拟数据集。通过对角速度的计算，机器人具备执行动作的能力，并从形状和准确性的角度对其动作质量进行了评估。由于其广泛的应用性，我们的方法成功生成了精确的轨迹，这些轨迹可以根据需要定制并适应不同的环境。 

---
# Personalized Mental State Evaluation in Human-Robot Interaction using Federated Learning 

**Title (ZH)**: 基于联邦学习的人机交互个性化心理状态评估 

**Authors**: Andrea Bussolan, Oliver Avram, Andrea Pignata, Gianvito Urgese, Stefano Baraldo, Anna Valente  

**Link**: [PDF](https://arxiv.org/pdf/2506.20212)  

**Abstract**: With the advent of Industry 5.0, manufacturers are increasingly prioritizing worker well-being alongside mass customization. Stress-aware Human-Robot Collaboration (HRC) plays a crucial role in this paradigm, where robots must adapt their behavior to human mental states to improve collaboration fluency and safety. This paper presents a novel framework that integrates Federated Learning (FL) to enable personalized mental state evaluation while preserving user privacy. By leveraging physiological signals, including EEG, ECG, EDA, EMG, and respiration, a multimodal model predicts an operator's stress level, facilitating real-time robot adaptation. The FL-based approach allows distributed on-device training, ensuring data confidentiality while improving model generalization and individual customization. Results demonstrate that the deployment of an FL approach results in a global model with performance in stress prediction accuracy comparable to a centralized training approach. Moreover, FL allows for enhancing personalization, thereby optimizing human-robot interaction in industrial settings, while preserving data privacy. The proposed framework advances privacy-preserving, adaptive robotics to enhance workforce well-being in smart manufacturing. 

**Abstract (ZH)**: 随着 industrie 5.0 的到来，制造商越来越重视在大规模个性化生产的同时保障工人的福祉。具备压力感知能力的人机协作（HRC）在此范式中发挥着关键作用，其中机器人必须根据人类的心理状态调整其行为，以提高协作流畅性和安全性。本文提出了一种将联邦学习（FL）集成的新框架，以实现个性化心理健康评估并同时保护用户隐私。通过利用包括EEG、ECG、EDA、EMG和呼吸在内的生理信号，一个多模态模型预测操作员的压力水平，从而实现实时机器人适应。基于联邦学习的方法允许分布式设备上训练，确保数据机密性的同时提高模型泛化能力和个体定制能力。结果显示，部署基于联邦学习的方法能够获得与集中训练方法在压力预测准确性方面相当的全局模型。此外，联邦学习还有助于提高个性化水平，从而优化工业环境中的人机交互，同时保护数据隐私。所提出的框架促进了保护隐私的人机适应性技术的发展，以提高智能制造业的工作环境质量。 

---
# PSALM-V: Automating Symbolic Planning in Interactive Visual Environments with Large Language Models 

**Title (ZH)**: PSALM-V: 在大型语言模型指导下自动在交互式视觉环境中进行符号规划 

**Authors**: Wang Bill Zhu, Miaosen Chai, Ishika Singh, Robin Jia, Jesse Thomason  

**Link**: [PDF](https://arxiv.org/pdf/2506.20097)  

**Abstract**: We propose PSALM-V, the first autonomous neuro-symbolic learning system able to induce symbolic action semantics (i.e., pre- and post-conditions) in visual environments through interaction. PSALM-V bootstraps reliable symbolic planning without expert action definitions, using LLMs to generate heuristic plans and candidate symbolic semantics. Previous work has explored using large language models to generate action semantics for Planning Domain Definition Language (PDDL)-based symbolic planners. However, these approaches have primarily focused on text-based domains or relied on unrealistic assumptions, such as access to a predefined problem file, full observability, or explicit error messages. By contrast, PSALM-V dynamically infers PDDL problem files and domain action semantics by analyzing execution outcomes and synthesizing possible error explanations. The system iteratively generates and executes plans while maintaining a tree-structured belief over possible action semantics for each action, iteratively refining these beliefs until a goal state is reached. Simulated experiments of task completion in ALFRED demonstrate that PSALM-V increases the plan success rate from 37% (Claude-3.7) to 74% in partially observed setups. Results on two 2D game environments, RTFM and Overcooked-AI, show that PSALM-V improves step efficiency and succeeds in domain induction in multi-agent settings. PSALM-V correctly induces PDDL pre- and post-conditions for real-world robot BlocksWorld tasks, despite low-level manipulation failures from the robot. 

**Abstract (ZH)**: 我们提出PSALM-V，这是一种能够在视觉环境中通过交互自动诱导符号动作语义（即前件和后件）的第一种自主神经符号学习系统。PSALM-V 无需专家动作定义即可启动可靠的符号规划，使用大语言模型生成启发式计划和候选符号语义。此前的工作探索了使用大语言模型为基于PDDL的符号规划器生成动作语义。然而，这些方法主要集中在文本基础领域，或者依赖于不切实际的假设，如访问预定义的问题文件、完全可观测性或明确的错误消息。相比之下，PSALM-V 动态推断 PDDL 问题文件和领域动作语义，通过分析执行结果并综合可能的错误解释。该系统迭代生成和执行计划，同时维护每个动作可能的动作语义的树状信念结构，直到达到目标状态。在 ALFRED 的模拟任务完成实验中，PSALM-V 在部分可观测设置中的计划成功率从 Claude-3.7 的 37% 提高到 74%。对两个 2D 游戏环境 RTFM 和 Overcooked-AI 的结果表明，PSALM-V 在多智能体设置中提高了步骤效率并成功进行了领域诱导。尽管机器人在低级操作中出现故障，PSALM-V 仍能正确诱导 PDDL 前件和后件，以完成真实世界的机器人 BlocksWorld 任务。 

---
# Robust Robotic Exploration and Mapping Using Generative Occupancy Map Synthesis 

**Title (ZH)**: 使用生成占用地图合成的鲁棒机器人探索与建图 

**Authors**: Lorin Achey, Alec Reed, Brendan Crowe, Bradley Hayes, Christoffer Heckman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20049)  

**Abstract**: We present a novel approach for enhancing robotic exploration by using generative occupancy mapping. We introduce SceneSense, a diffusion model designed and trained for predicting 3D occupancy maps given partial observations. Our proposed approach probabilistically fuses these predictions into a running occupancy map in real-time, resulting in significant improvements in map quality and traversability. We implement SceneSense onboard a quadruped robot and validate its performance with real-world experiments to demonstrate the effectiveness of the model. In these experiments, we show that occupancy maps enhanced with SceneSense predictions better represent our fully observed ground truth data (24.44% FID improvement around the robot and 75.59% improvement at range). We additionally show that integrating SceneSense-enhanced maps into our robotic exploration stack as a "drop-in" map improvement, utilizing an existing off-the-shelf planner, results in improvements in robustness and traversability time. Finally we show results of full exploration evaluations with our proposed system in two dissimilar environments and find that locally enhanced maps provide more consistent exploration results than maps constructed only from direct sensor measurements. 

**Abstract (ZH)**: 我们提出了一种通过生成占用映射增强机器人探索的新方法。我们介绍了SceneSense，这是一种为根据部分观察预测3D占用映射而设计和训练的扩散模型。我们提出的方法在运行中以概率方式融合这些预测，从而显著提高了地图质量和可通行性。我们在四足机器人上实现了SceneSense，并通过实际实验验证了其性能，以证明该模型的有效性。在这些实验中，我们展示了使用SceneSense预测增强的占用映射更好地代表了我们完全观测到的地面真实数据（机器人周围FID改进24.44%和长距离FID改进75.59%）。此外，我们将SceneSense增强的地图无缝集成到现有的机器人探索栈中，利用现成的规划器，从而提高了鲁棒性和穿越时间。最后，我们在两个不同环境中的全探索评估中展示了我们提出系统的成果，并发现局部增强的地图提供了比仅从直接传感器测量构建的地图更一致的探索结果。 

---
# Consensus-Driven Uncertainty for Robotic Grasping based on RGB Perception 

**Title (ZH)**: 基于RGB感知的共识驱动不确定性机器人抓取 

**Authors**: Eric C. Joyce, Qianwen Zhao, Nathaniel Burgdorfer, Long Wang, Philippos Mordohai  

**Link**: [PDF](https://arxiv.org/pdf/2506.20045)  

**Abstract**: Deep object pose estimators are notoriously overconfident. A grasping agent that both estimates the 6-DoF pose of a target object and predicts the uncertainty of its own estimate could avoid task failure by choosing not to act under high uncertainty. Even though object pose estimation improves and uncertainty quantification research continues to make strides, few studies have connected them to the downstream task of robotic grasping. We propose a method for training lightweight, deep networks to predict whether a grasp guided by an image-based pose estimate will succeed before that grasp is attempted. We generate training data for our networks via object pose estimation on real images and simulated grasping. We also find that, despite high object variability in grasping trials, networks benefit from training on all objects jointly, suggesting that a diverse variety of objects can nevertheless contribute to the same goal. 

**Abstract (ZH)**: 深度物体姿态估计器 notoriously 过于自信。一种同时估计目标物体6-自由度姿态并预测自身估计不确定性的方法可以避免在高不确定性下执行任务而导致的任务失败。尽管物体姿态估计进步显著且不确定性量化研究不断取得进展，但很少有研究将它们与后续的机器人抓取任务联系起来。我们提出了一种训练轻量级深度网络的方法，在抓取尝试之前预测由基于图像的姿态估计引导的抓取是否成功。我们通过对真实图像进行物体姿态估计和模拟抓取生成网络的训练数据。我们还发现，尽管抓取试验中物体存在高变异性，但网络从共同训练所有物体中受益，这表明多样化的物体仍然可以为同一个目标做出贡献。 

---
# Hierarchical Reinforcement Learning and Value Optimization for Challenging Quadruped Locomotion 

**Title (ZH)**: 基于层次强化学习和价值优化的挑战性四足行走控制 

**Authors**: Jeremiah Coholich, Muhammad Ali Murtaza, Seth Hutchinson, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.20036)  

**Abstract**: We propose a novel hierarchical reinforcement learning framework for quadruped locomotion over challenging terrain. Our approach incorporates a two-layer hierarchy in which a high-level policy (HLP) selects optimal goals for a low-level policy (LLP). The LLP is trained using an on-policy actor-critic RL algorithm and is given footstep placements as goals. We propose an HLP that does not require any additional training or environment samples and instead operates via an online optimization process over the learned value function of the LLP. We demonstrate the benefits of this framework by comparing it with an end-to-end reinforcement learning (RL) approach. We observe improvements in its ability to achieve higher rewards with fewer collisions across an array of different terrains, including terrains more difficult than any encountered during training. 

**Abstract (ZH)**: 我们提出了一种新颖的分层强化学习框架，用于在具有挑战性的地形上实现四足行走。该方法采用两层层次结构，其中高层策略（HLP）选择低层策略（LLP）的最优目标。LLP 使用on-policyactor-critic RL算法进行训练，并以脚步放置作为目标。我们提出了一种HLP，无需额外训练或环境样本，而是通过在线优化过程操作LLP学习的价值函数。我们通过将其与端到端的强化学习（RL）方法进行比较，展示了该框架的优势。我们观察到，无论地形如何（包括比训练中遇到的更难的地形），它在较少碰撞的情况下实现更高奖励的能力有所提高。 

---
# Robust Embodied Self-Identification of Morphology in Damaged Multi-Legged Robots 

**Title (ZH)**: 受损多足机器人鲁棒体体现有结构自我识别 

**Authors**: Sahand Farghdani, Mili Patel, Robin Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2506.19984)  

**Abstract**: Multi-legged robots (MLRs) are vulnerable to leg damage during complex missions, which can impair their performance. This paper presents a self-modeling and damage identification algorithm that enables autonomous adaptation to partial or complete leg loss using only data from a low-cost IMU. A novel FFT-based filter is introduced to address time-inconsistent signals, improving damage detection by comparing body orientation between the robot and its model. The proposed method identifies damaged legs and updates the robot's model for integration into its control system. Experiments on uneven terrain validate its robustness and computational efficiency. 

**Abstract (ZH)**: 多腿机器人在复杂任务中易遭受腿部损伤，影响其性能。本文提出了一种自建模和损伤识别算法，仅使用低成本IMU的数据，实现对部分或完全腿部损失的自主适应。引入了一种新型FFTベース的滤波器来处理时间不一致的信号，通过比较机器人与其模型之间的身体姿态来提高损伤检测的准确性。所提出的方法能够识别受损腿部并更新机器人的模型，以便将其集成到控制系统中。实验结果在不平地面上验证了其稳健性和计算效率。 

---
# Evolutionary Gait Reconfiguration in Damaged Legged Robots 

**Title (ZH)**: 受损腿式机器人中的进化步态重构 

**Authors**: Sahand Farghdani, Robin Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2506.19968)  

**Abstract**: Multi-legged robots deployed in complex missions are susceptible to physical damage in their legs, impairing task performance and potentially compromising mission success. This letter presents a rapid, training-free damage recovery algorithm for legged robots subject to partial or complete loss of functional legs. The proposed method first stabilizes locomotion by generating a new gait sequence and subsequently optimally reconfigures leg gaits via a developed differential evolution algorithm to maximize forward progression while minimizing body rotation and lateral drift. The algorithm successfully restores locomotion in a 24-degree-of-freedom hexapod within one hour, demonstrating both high efficiency and robustness to structural damage. 

**Abstract (ZH)**: 具有复杂任务需求的多足机器人容易遭受腿部物理损伤，影响任务性能并可能危及任务成功。本信提出了一种无需训练的快速损伤恢复算法，用于在部分或完全丧失功能性腿部的情况下优化腿足运动。该方法首先通过生成新的步态序列来稳定运动，然后利用开发的差分进化算法重新配置腿足运动，以最大化前进距离的同时最小化身体旋转和侧向漂移。该算法在一小时内成功恢复了一种具有24个自由度的六足机器人的运动，显示出高效率和对结构损伤的鲁棒性。 

---
# Task Allocation of UAVs for Monitoring Missions via Hardware-in-the-Loop Simulation and Experimental Validation 

**Title (ZH)**: 基于硬件在环仿真与实验验证的无人机监测任务分配 

**Authors**: Hamza Chakraa, François Guérin, Edouard Leclercq, Dimitri Lefebvre  

**Link**: [PDF](https://arxiv.org/pdf/2506.20626)  

**Abstract**: This study addresses the optimisation of task allocation for Unmanned Aerial Vehicles (UAVs) within industrial monitoring missions. The proposed methodology integrates a Genetic Algorithms (GA) with a 2-Opt local search technique to obtain a high-quality solution. Our approach was experimentally validated in an industrial zone to demonstrate its efficacy in real-world scenarios. Also, a Hardware-in-the-loop (HIL) simulator for the UAVs team is introduced. Moreover, insights about the correlation between the theoretical cost function and the actual battery consumption and time of flight are deeply analysed. Results show that the considered costs for the optimisation part of the problem closely correlate with real-world data, confirming the practicality of the proposed approach. 

**Abstract (ZH)**: 本研究针对工业监测任务中无人机任务分配的优化进行了研究。提出的算法将遗传算法与2-Opt局部搜索技术集成以获得高质量的解决方案。该方法在工业区进行了实验验证，以展示其实用性。此外，还引入了无人机团队的硬件在环（HIL）仿真器。同时，深入分析了理论成本函数与实际电池消耗和飞行时间之间的关联。研究结果表明，考虑的成本与实际数据高度相关，验证了所提方法的实用性。 

---
# Learning-Based Distance Estimation for 360° Single-Sensor Setups 

**Title (ZH)**: 基于学习的360°单传感器设置距离估计 

**Authors**: Yitong Quan, Benjamin Kiefer, Martin Messmer, Andreas Zell  

**Link**: [PDF](https://arxiv.org/pdf/2506.20586)  

**Abstract**: Accurate distance estimation is a fundamental challenge in robotic perception, particularly in omnidirectional imaging, where traditional geometric methods struggle with lens distortions and environmental variability. In this work, we propose a neural network-based approach for monocular distance estimation using a single 360° fisheye lens camera. Unlike classical trigonometric techniques that rely on precise lens calibration, our method directly learns and infers the distance of objects from raw omnidirectional inputs, offering greater robustness and adaptability across diverse conditions. We evaluate our approach on three 360° datasets (LOAF, ULM360, and a newly captured dataset Boat360), each representing distinct environmental and sensor setups. Our experimental results demonstrate that the proposed learning-based model outperforms traditional geometry-based methods and other learning baselines in both accuracy and robustness. These findings highlight the potential of deep learning for real-time omnidirectional distance estimation, making our approach particularly well-suited for low-cost applications in robotics, autonomous navigation, and surveillance. 

**Abstract (ZH)**: 基于神经网络的单目360°鱼眼镜头深度估计方法 

---
# Lightweight Multi-Frame Integration for Robust YOLO Object Detection in Videos 

**Title (ZH)**: 轻量级多帧集成方法在视频中实现鲁棒YOLO目标检测 

**Authors**: Yitong Quan, Benjamin Kiefer, Martin Messmer, Andreas Zell  

**Link**: [PDF](https://arxiv.org/pdf/2506.20550)  

**Abstract**: Modern image-based object detection models, such as YOLOv7, primarily process individual frames independently, thus ignoring valuable temporal context naturally present in videos. Meanwhile, existing video-based detection methods often introduce complex temporal modules, significantly increasing model size and computational complexity. In practical applications such as surveillance and autonomous driving, transient challenges including motion blur, occlusions, and abrupt appearance changes can severely degrade single-frame detection performance. To address these issues, we propose a straightforward yet highly effective strategy: stacking multiple consecutive frames as input to a YOLO-based detector while supervising only the output corresponding to a single target frame. This approach leverages temporal information with minimal modifications to existing architectures, preserving simplicity, computational efficiency, and real-time inference capability. Extensive experiments on the challenging MOT20Det and our BOAT360 datasets demonstrate that our method improves detection robustness, especially for lightweight models, effectively narrowing the gap between compact and heavy detection networks. Additionally, we contribute the BOAT360 benchmark dataset, comprising annotated fisheye video sequences captured from a boat, to support future research in multi-frame video object detection in challenging real-world scenarios. 

**Abstract (ZH)**: 基于现代图像对象检测模型（如YOLOv7）主要独立处理单帧图像并忽略视频中固有的时间上下文的问题，同时现有的基于视频的检测方法常引入复杂的时序模块，显著增加了模型大小和计算复杂度。在监控和自动驾驶等实际应用中，瞬时挑战如运动模糊、遮挡以及突然的外观变化会严重影响单帧检测性能。为解决这些问题，我们提出了一种简单而有效的策略：将多个连续帧作为输入传入YOLO基检测器，并仅监督与单个目标帧对应的输出。该方法在最小修改现有架构的情况下利用时间信息，保持了简洁性、计算效率和实时推理能力。在具有挑战性的MOT20Det和自主研发的BOAT360数据集上的大量实验表明，本方法提高了检测的鲁棒性，特别是对于轻量级模型，有效减小了紧凑型和重型检测网络之间的差距。此外，我们贡献了包含从船只捕获的鱼眼视频序列的BOAT360基准数据集，以支持在具有挑战性的现实场景下多帧视频对象检测的研究。 

---
