# Inland-LOAM: Voxel-Based Structural Semantic Mapping for Inland Waterways 

**Title (ZH)**: 内陆-LOAM：基于体素的结构语义建图方法用于内陆水道 

**Authors**: Zhongbi Luo, Yunjia Wang, Jan Swevers, Peter Slaets, Herman Bruyninckx  

**Link**: [PDF](https://arxiv.org/pdf/2508.03672)  

**Abstract**: Accurate geospatial information is crucial for safe, autonomous Inland Waterway Transport (IWT), as existing charts (IENC) lack real-time detail and conventional LiDAR SLAM fails in waterway environments. These challenges lead to vertical drift and non-semantic maps, hindering autonomous navigation.
This paper introduces Inland-LOAM, a LiDAR SLAM framework for waterways. It uses an improved feature extraction and a water surface planar constraint to mitigate vertical drift. A novel pipeline transforms 3D point clouds into structured 2D semantic maps using voxel-based geometric analysis, enabling real-time computation of navigational parameters like bridge clearances. An automated module extracts shorelines and exports them into a lightweight, IENC-compatible format.
Evaluations on a real-world dataset show Inland-LOAM achieves superior localization accuracy over state-of-the-art methods. The generated semantic maps and shorelines align with real-world conditions, providing reliable data for enhanced situational awareness. The code and dataset will be publicly available 

**Abstract (ZH)**: 准确的地理空间信息对于安全、自主内河航运（IWT）至关重要，现有海图（IENC）缺乏实时细节，传统LiDAR SLAM在水道环境中失效。这些挑战导致了垂直漂移和非语义地图，阻碍了自主导航。

本文介绍了Inland-LOAM，一种适用于水道的LiDAR SLAM框架。它使用改进的特征提取和水面平面约束来减轻垂直漂移。一种新型流水线将3D点云转换为结构化的2D语义地图，通过体素几何分析实现即时计算航行参数，如桥梁净空。一个自动化模块提取海岸线并以轻量级、兼容IENC的格式导出。

对真实数据集的评估显示，Inland-LOAM在定位准确性上优于现有方法。生成的语义地图和海岸线与实际情况相符，提供了增强情况意识的可靠数据。代码和数据集将公开可用。 

---
# DiWA: Diffusion Policy Adaptation with World Models 

**Title (ZH)**: DiWA: 基于世界模型的扩散策略自适应 

**Authors**: Akshay L Chandra, Iman Nematollahi, Chenguang Huang, Tim Welschehold, Wolfram Burgard, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2508.03645)  

**Abstract**: Fine-tuning diffusion policies with reinforcement learning (RL) presents significant challenges. The long denoising sequence for each action prediction impedes effective reward propagation. Moreover, standard RL methods require millions of real-world interactions, posing a major bottleneck for practical fine-tuning. Although prior work frames the denoising process in diffusion policies as a Markov Decision Process to enable RL-based updates, its strong dependence on environment interaction remains highly inefficient. To bridge this gap, we introduce DiWA, a novel framework that leverages a world model for fine-tuning diffusion-based robotic skills entirely offline with reinforcement learning. Unlike model-free approaches that require millions of environment interactions to fine-tune a repertoire of robot skills, DiWA achieves effective adaptation using a world model trained once on a few hundred thousand offline play interactions. This results in dramatically improved sample efficiency, making the approach significantly more practical and safer for real-world robot learning. On the challenging CALVIN benchmark, DiWA improves performance across eight tasks using only offline adaptation, while requiring orders of magnitude fewer physical interactions than model-free baselines. To our knowledge, this is the first demonstration of fine-tuning diffusion policies for real-world robotic skills using an offline world model. We make the code publicly available at this https URL. 

**Abstract (ZH)**: 使用强化学习微调扩散策略存在显著挑战：基于世界模型的离线微调框架 

---
# Why Evolve When You Can Adapt? Post-Evolution Adaptation of Genetic Memory for On-the-Fly Control 

**Title (ZH)**: 当可以适应时为何进化？进化的后适应遗传记忆的即时控制 

**Authors**: Hamze Hammami, Eva Denisa Barbulescu, Talal Shaikh, Mouayad Aldada, Muhammad Saad Munawar  

**Link**: [PDF](https://arxiv.org/pdf/2508.03600)  

**Abstract**: Imagine a robot controller with the ability to adapt like human synapses, dynamically rewiring itself to overcome unforeseen challenges in real time. This paper proposes a novel zero-shot adaptation mechanism for evolutionary robotics, merging a standard Genetic Algorithm (GA) controller with online Hebbian plasticity. Inspired by biological systems, the method separates learning and memory, with the genotype acting as memory and Hebbian updates handling learning. In our approach, the fitness function is leveraged as a live scaling factor for Hebbian learning, enabling the robot's neural controller to adjust synaptic weights on-the-fly without additional training. This adds a dynamic adaptive layer that activates only during runtime to handle unexpected environmental changes. After the task, the robot 'forgets' the temporary adjustments and reverts to the original weights, preserving core knowledge. We validate this hybrid GA-Hebbian controller on an e-puck robot in a T-maze navigation task with changing light conditions and obstacles. 

**Abstract (ZH)**: 一种将标准遗传算法与在线海宾可塑性融合的零样本适应机制：生物启发的进化机器人动态适应方法 

---
# Online Learning for Vibration Suppression in Physical Robot Interaction using Power Tools 

**Title (ZH)**: 使用动力工具进行物理机器人交互的振动抑制在线学习方法 

**Authors**: Gokhan Solak, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2508.03559)  

**Abstract**: Vibration suppression is an important capability for collaborative robots deployed in challenging environments such as construction sites. We study the active suppression of vibration caused by external sources such as power tools. We adopt the band-limited multiple Fourier linear combiner (BMFLC) algorithm to learn the vibration online and counter it by feedforward force control. We propose the damped BMFLC method, extending BMFLC with a novel adaptive step-size approach that improves the convergence time and noise resistance. Our logistic function-based damping mechanism reduces the effect of noise and enables larger learning rates. We evaluate our method on extensive simulation experiments with realistic time-varying multi-frequency vibration and real-world physical interaction experiments. The simulation experiments show that our method improves the suppression rate in comparison to the original BMFLC and its recursive least squares and Kalman filter-based extensions. Furthermore, our method is far more efficient than the latter two. We further validate the effectiveness of our method in real-world polishing experiments. A supplementary video is available at this https URL. 

**Abstract (ZH)**: 协作机器人在具有挑战性的环境中（如建筑工地）的振动抑制是一项重要能力。我们研究了由外部源（如动力工具）引起的振动的主动抑制方法。我们采用带限多傅里叶线性组合器（BMFLC）算法在线学习振动，并通过前馈力控制来抵消振动。我们提出了阻尼BMFLC方法，通过一种新颖的自适应步长方法扩展BMFLC，从而提高收敛时间和抗噪性能。基于Logistic函数的阻尼机制减少了噪声的影响并允许更大的学习率。我们在包含现实时间变多频率振动的广泛仿真实验以及真实的物理交互实验中评估了我们的方法。仿真实验表明，与原始BMFLC及其基于递归最小二乘法和卡尔曼滤波器的扩展方法相比，我们的方法在抑制率方面有显著改进，且效率远高于后者。我们进一步在实际抛光实验中验证了该方法的有效性。更多信息请参见此链接：https://this-url。 

---
# Vision-based Perception System for Automated Delivery Robot-Pedestrians Interactions 

**Title (ZH)**: 基于视觉的感知系统：自动配送机器人与行人交互 

**Authors**: Ergi Tushe, Bilal Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2508.03541)  

**Abstract**: The integration of Automated Delivery Robots (ADRs) into pedestrian-heavy urban spaces introduces unique challenges in terms of safe, efficient, and socially acceptable navigation. We develop the complete pipeline for a single vision sensor based multi-pedestrian detection and tracking, pose estimation, and monocular depth perception. Leveraging the real-world MOT17 dataset sequences, this study demonstrates how integrating human-pose estimation and depth cues enhances pedestrian trajectory prediction and identity maintenance, even under occlusions and dense crowds. Results show measurable improvements, including up to a 10% increase in identity preservation (IDF1), a 7% improvement in multiobject tracking accuracy (MOTA), and consistently high detection precision exceeding 85%, even in challenging scenarios. Notably, the system identifies vulnerable pedestrian groups supporting more socially aware and inclusive robot behaviour. 

**Abstract (ZH)**: Automated Delivery Robots在行人密集城市空间中的集成：基于单一视觉传感器的多行人检测与跟踪、姿态估计及单目深度感知的完整管道研究 

---
# CollaBot: Vision-Language Guided Simultaneous Collaborative Manipulation 

**Title (ZH)**: CollaBot: 视觉-语言引导的协作 manipulation 

**Authors**: Kun Song, Shentao Ma, Gaoming Chen, Ninglong Jin, Guangbao Zhao, Mingyu Ding, Zhenhua Xiong, Jia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.03526)  

**Abstract**: A central research topic in robotics is how to use this system to interact with the physical world. Traditional manipulation tasks primarily focus on small objects. However, in factory or home environments, there is often a need for the movement of large objects, such as moving tables. These tasks typically require multi-robot systems to work collaboratively. Previous research lacks a framework that can scale to arbitrary sizes of robots and generalize to various kinds of tasks. In this work, we propose CollaBot, a generalist framework for simultaneous collaborative manipulation. First, we use SEEM for scene segmentation and point cloud extraction of the target object. Then, we propose a collaborative grasping framework, which decomposes the task into local grasp pose generation and global collaboration. Finally, we design a 2-stage planning module that can generate collision-free trajectories to achieve this task. Experiments show a success rate of 52% across different numbers of robots, objects, and tasks, indicating the effectiveness of the proposed framework. 

**Abstract (ZH)**: 机器人领域的核心研究主题是如何利用该系统与物理世界进行交互。传统操作任务主要集中在小型物体上。然而，在工厂或家庭环境中，往往需要移动大型物体，如移动桌子。这些任务通常需要多机器人系统协同工作。以往的研究缺乏能够扩展到任意大小机器人并应用于各种任务的框架。在这项工作中，我们提出CollaBot，一种通用的协同操作框架。首先，我们使用SEEM进行场景分割和目标物体的点云提取。然后，我们提出了一种协同抓取框架，将任务分解为局部抓取姿态生成和全局协同。最后，我们设计了一个2阶段规划模块，能够生成无碰撞轨迹以完成该任务。实验结果显示，在不同数量的机器人、物体和任务下，成功率高达52%，表明所提出框架的有效性。 

---
# Theatre in the Loop: A Rehearsal-Based, Collaborative Workflow for Expressive Robotic Behaviours 

**Title (ZH)**: 环路剧场：基于排练的协作工作流，用于表达性机器人行为 

**Authors**: Pavlos Panagiotidis, Victor Zhi Heung Ngo, Sean Myatt, Roma Patel, Rachel Ramchurn, Alan Chamberlain, Ayse Kucukyilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2508.03514)  

**Abstract**: In this paper, we propose theatre-in-the-loop, a framework for developing expressive robot behaviours tailored to artistic performance through a director-guided puppeteering workflow. Leveraging theatrical methods, we use narrative objectives to direct a puppeteer in generating improvised robotic gestures that convey specific emotions. These improvisations are captured and curated to build a dataset of reusable movement templates for standalone playback in future autonomous performances. Initial trials demonstrate the feasibility of this approach, illustrating how the workflow enables precise sculpting of robotic gestures into coherent emotional arcs while revealing challenges posed by the robot's mechanical constraints. We argue that this practice-led framework provides a model for interdisciplinary teams creating socially expressive robot behaviours, contributing to (1) theatre as an interactive training ground for human-robot interaction and (2) co-creation methodologies between humans and machines. 

**Abstract (ZH)**: 在本文中，我们提出了一种剧场在环（Theatre-in-the- Loop）框架， 一种通过导演引导的木偶操控工作流程来开发符合适应艺术表演的机器人行为的框架。我们 使用剧场技术来 方法论来为木偶表演者制定叙述目标，，来引导其生成即 卸奏即的机器人姿态，并，这些姿态能够传达特定的情绪。这些即的动作随后被整理和编目，并构建为一个数据集，其中包含可用于未来独立播放的自立播放的机器人姿态模板。初始试验证明了这一框架的可行性，展示了工作流程如何能够精确地塑造型机器人姿态并， 幸导致有逻辑连贯的情感弧线，同时也应对由机器人硬件带来的挑战。这种以实践为主导的框架旨在为跨学科团队提供一种构思社会表达性型机器人行为的方法实践。它提供了人一个互动的实践平台èrent

user
可以优化一下上面的翻译，使它更加流畅和符合中文的表达习惯吗？ 

---
# Residual Neural Terminal Constraint for MPC-based Collision Avoidance in Dynamic Environments 

**Title (ZH)**: 基于MPC的动态环境避碰中残差神经终端约束的研究 

**Authors**: Bojan Derajić, Mohamed-Khalil Bouzidi, Sebastian Bernhard, Wolfgang Hönig  

**Link**: [PDF](https://arxiv.org/pdf/2508.03428)  

**Abstract**: In this paper, we propose a hybrid MPC local planner that uses a learning-based approximation of a time-varying safe set, derived from local observations and applied as the MPC terminal constraint. This set can be represented as a zero-superlevel set of the value function computed via Hamilton-Jacobi (HJ) reachability analysis, which is infeasible in real-time. We exploit the property that the HJ value function can be expressed as a difference of the corresponding signed distance function (SDF) and a non-negative residual function. The residual component is modeled as a neural network with non-negative output and subtracted from the computed SDF, resulting in a real-time value function estimate that is at least as safe as the SDF by design. Additionally, we parametrize the neural residual by a hypernetwork to improve real-time performance and generalization properties. The proposed method is compared with three state-of-the-art methods in simulations and hardware experiments, achieving up to 30\% higher success rates compared to the best baseline while requiring a similar computational effort and producing high-quality (low travel-time) solutions. 

**Abstract (ZH)**: 本文提出了一种混合模型预测控制（MPC）局部规划器，该规划器利用基于学习的时间varying安全集近似，该集基于局部观测并作为MPC终端约束应用。该集可以表示为通过哈密尔顿-雅可比（HJ）可达性分析计算的价值函数的零超上水平集，在实时应用中是不可行的。我们利用HJ价值函数可以表示为相应符号距离函数（SDF）和非负残差函数差的性质。残差部分被建模为具有非负输出的神经网络，并从计算得到的SDF中减去，从而得到一个实时价值函数估计，该估计在设计上至少与SDF一样安全。此外，我们通过超网络参数化神经残差以提高实时性能和泛化特性。所提出的方法在仿真和硬件实验中与三种最新方法进行了比较，在相似的计算努力下，成功率最高可提高30%，同时生成高质量（低行程时间）的解决方案。 

---
# Opti-Acoustic Scene Reconstruction in Highly Turbid Underwater Environments 

**Title (ZH)**: 高浑浊度水下环境中的优化声学场景重建 

**Authors**: Ivana Collado-Gonzalez, John McConnell, Paul Szenher, Brendan Englot  

**Link**: [PDF](https://arxiv.org/pdf/2508.03408)  

**Abstract**: Scene reconstruction is an essential capability for underwater robots navigating in close proximity to structures. Monocular vision-based reconstruction methods are unreliable in turbid waters and lack depth scale information. Sonars are robust to turbid water and non-uniform lighting conditions, however, they have low resolution and elevation ambiguity. This work proposes a real-time opti-acoustic scene reconstruction method that is specially optimized to work in turbid water. Our strategy avoids having to identify point features in visual data and instead identifies regions of interest in the data. We then match relevant regions in the image to corresponding sonar data. A reconstruction is obtained by leveraging range data from the sonar and elevation data from the camera image. Experimental comparisons against other vision-based and sonar-based approaches at varying turbidity levels, and field tests conducted in marina environments, validate the effectiveness of the proposed approach. We have made our code open-source to facilitate reproducibility and encourage community engagement. 

**Abstract (ZH)**: 水下机器人在结构附近近距离导航时，场景重建是一项基本能力。单目视觉基场景重建方法在浑浊水域中 unreliable 并缺乏深度比例信息。声呐在浑浊水域和不均匀光照条件下具有 robust 性，但分辨率低且存在方位模糊。本文提出了一种特别优化的实时光学-声学场景重建方法，适用于浑浊水域。我们的策略避免在视觉数据中识别点特征，而是识别数据中的感兴趣区域。然后将图像中的相关区域与相应的声呐数据匹配。通过利用声呐的距离数据和相机图像的方位数据进行重建。实验结果表明，在不同浑浊程度下与基于视觉和其他基于声呐的方法相比，所提方法有效。我们在港口环境中进行了实地测试，验证了所提方法的有效性。我们已开源代码以促进可重复性并鼓励社区参与。 

---
# UniFucGrasp: Human-Hand-Inspired Unified Functional Grasp Annotation Strategy and Dataset for Diverse Dexterous Hands 

**Title (ZH)**: UniFucGrasp: 人体手部启发的统一功能抓取标注策略及多样化灵巧手数据集 

**Authors**: Haoran Lin, Wenrui Chen, Xianchi Chen, Fan Yang, Qiang Diao, Wenxin Xie, Sijie Wu, Kailun Yang, Maojun Li, Yaonan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03339)  

**Abstract**: Dexterous grasp datasets are vital for embodied intelligence, but mostly emphasize grasp stability, ignoring functional grasps needed for tasks like opening bottle caps or holding cup handles. Most rely on bulky, costly, and hard-to-control high-DOF Shadow Hands. Inspired by the human hand's underactuated mechanism, we establish UniFucGrasp, a universal functional grasp annotation strategy and dataset for multiple dexterous hand types. Based on biomimicry, it maps natural human motions to diverse hand structures and uses geometry-based force closure to ensure functional, stable, human-like grasps. This method supports low-cost, efficient collection of diverse, high-quality functional grasps. Finally, we establish the first multi-hand functional grasp dataset and provide a synthesis model to validate its effectiveness. Experiments on the UFG dataset, IsaacSim, and complex robotic tasks show that our method improves functional manipulation accuracy and grasp stability, enables efficient generalization across diverse robotic hands, and overcomes annotation cost and generalization challenges in dexterous grasping. The project page is at this https URL. 

**Abstract (ZH)**: 灵巧抓取数据集对于嵌入式智能至关重要，但通常侧重于抓取稳定性，忽略了完成开瓶盖或握住杯子把手等任务所需的功能性抓取。大多数数据集依赖于体积大、成本高且难以控制的高自由度Shadow手。受人类手部未驱动机制的启发，我们建立了UniFucGrasp，这是一种适用于多种灵巧手类型的通用功能性抓取注释策略和数据集。基于仿生学原理，它将自然的人类动作映射到多种手部结构，并使用基于几何的力闭合确保功能化、稳定的人类似抓取。该方法支持低成本、高效地收集多样化高质量的功能性抓取。最后，我们建立了首个多手功能性抓取数据集，并提供了一个综合模型来验证其有效性。在UFG数据集、IsaacSim以及复杂的机器人任务上的实验表明，我们的方法提高了功能性操作的准确性和抓取稳定性，实现了跨不同机器人手的高效泛化，并克服了灵巧抓取中的注释成本和泛化挑战。项目页面见此链接。 

---
# Force-Compliance MPC and Robot-User CBFs for Interactive Navigation and User-Robot Safety in Hexapod Guide Robots 

**Title (ZH)**: 六足引导机器人中的交互导航与用户-机器人安全性力-顺应性MPC及机器人-用户CBFs 

**Authors**: Zehua Fan, Feng Gao, Zhijun Chen, Yunpeng Yin, Limin Yang, Qingxing Xi, En Yang, Xuefeng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.03246)  

**Abstract**: Guiding the visually impaired in complex environments requires real-time two-way interaction and safety assurance. We propose a Force-Compliance Model Predictive Control (FC-MPC) and Robot-User Control Barrier Functions (CBFs) for force-compliant navigation and obstacle avoidance in Hexapod guide robots. FC-MPC enables two-way interaction by estimating user-applied forces and moments using the robot's dynamic model and the recursive least squares (RLS) method, and then adjusting the robot's movements accordingly, while Robot-User CBFs ensure the safety of both the user and the robot by handling static and dynamic obstacles, and employ weighted slack variables to overcome feasibility issues in complex dynamic environments. We also adopt an Eight-Way Connected DBSCAN method for obstacle clustering, reducing computational complexity from O(n2) to approximately O(n), enabling real-time local perception on resource-limited on-board robot computers. Obstacles are modeled using Minimum Bounding Ellipses (MBEs), and their trajectories are predicted through Kalman filtering. Implemented on the HexGuide robot, the system seamlessly integrates force compliance, autonomous navigation, and obstacle avoidance. Experimental results demonstrate the system's ability to adapt to user force commands while guaranteeing user and robot safety simultaneously during navigation in complex environments. 

**Abstract (ZH)**: 指导视力障碍者在复杂环境中的导航需要实时双向交互和安全保障。我们提出一种力顺应模型预测控制（FC-MPC）和机器人-用户控制屏障函数（CBFs），用于六足导盲机器人中的力顺应导航和障碍物避免。FC-MPC通过使用机器人的动力学模型和递归最小二乘（RLS）方法估计用户施加在机器人上的力和力矩，并相应地调整机器人的运动，从而实现双向交互，而机器人-用户CBFs通过处理静态和动态障碍物，确保用户和机器人的安全，并采用加权松弛变量来克服复杂动态环境中可行性问题。我们还采用了八连通DBSCAN方法进行障碍物聚类，将计算复杂度从O(n^2)降低到大约O(n)，从而在资源有限的嵌入式机器人计算机上实现实时局部感知。障碍物被建模为最小包围椭圆（MBEs），并通过卡尔曼滤波预测其轨迹。该系统在HexGuide机器人上无缝集成力顺应、自主导航和障碍物避免功能。实验结果表明，该系统能够在复杂环境中导航时适应用户的力命令，同时保证用户和机器人的安全。 

---
# CookBench: A Long-Horizon Embodied Planning Benchmark for Complex Cooking Scenarios 

**Title (ZH)**: CookBench: 一个面向复杂烹饪场景的长时 Embodied 计划基准 

**Authors**: Muzhen Cai, Xiubo Chen, Yining An, Jiaxin Zhang, Xuesong Wang, Wang Xu, Weinan Zhang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03232)  

**Abstract**: Embodied Planning is dedicated to the goal of creating agents capable of executing long-horizon tasks in complex physical worlds. However, existing embodied planning benchmarks frequently feature short-horizon tasks and coarse-grained action primitives. To address this challenge, we introduce CookBench, a benchmark for long-horizon planning in complex cooking scenarios. By leveraging a high-fidelity simulation environment built upon the powerful Unity game engine, we define frontier AI challenges in a complex, realistic environment. The core task in CookBench is designed as a two-stage process. First, in Intention Recognition, an agent needs to accurately parse a user's complex intent. Second, in Embodied Interaction, the agent should execute the identified cooking goal through a long-horizon, fine-grained sequence of physical actions. Unlike existing embodied planning benchmarks, we refine the action granularity to a spatial level that considers crucial operational information while abstracting away low-level robotic control. Besides, We provide a comprehensive toolset that encapsulates the simulator. Its unified API supports both macro-level operations, such as placing orders and purchasing ingredients, and a rich set of fine-grained embodied actions for physical interaction, enabling researchers to focus on high-level planning and decision-making. Furthermore, we present an in-depth analysis of state-of-the-art, closed-source Large Language Model and Vision-Language Model, revealing their major shortcomings and challenges posed by complex, long-horizon tasks. The full benchmark will be open-sourced to facilitate future research. 

**Abstract (ZH)**: CookBench: 面向复杂烹饪场景的长期规划基准 

---
# Language as Cost: Proactive Hazard Mapping using VLM for Robot Navigation 

**Title (ZH)**: 语言作为成本：利用VLM进行主动危险Mapping以优化机器人导航 

**Authors**: Mintaek Oh, Chan Kim, Seung-Woo Seo, Seong-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.03138)  

**Abstract**: Robots operating in human-centric or hazardous environments must proactively anticipate and mitigate dangers beyond basic obstacle detection. Traditional navigation systems often depend on static maps, which struggle to account for dynamic risks, such as a person emerging from a suddenly opening door. As a result, these systems tend to be reactive rather than anticipatory when handling dynamic hazards. Recent advancements in pre-trained large language models and vision-language models (VLMs) create new opportunities for proactive hazard avoidance. In this work, we propose a zero-shot language-as-cost mapping framework that leverages VLMs to interpret visual scenes, assess potential dynamic risks, and assign risk-aware navigation costs preemptively, enabling robots to anticipate hazards before they materialize. By integrating this language-based cost map with a geometric obstacle map, the robot not only identifies existing obstacles but also anticipates and proactively plans around potential hazards arising from environmental dynamics. Experiments in simulated and diverse dynamic environments demonstrate that the proposed method significantly improves navigation success rates and reduces hazard encounters, compared to reactive baseline planners. Code and supplementary materials are available at this https URL. 

**Abstract (ZH)**: 在以人为本或危险环境中操作的机器人必须超出基本障碍检测，主动预见和减轻危险。传统的导航系统通常依赖静态地图，难以应对动态风险，如突然打开的门口出现的人。因此，这些系统在处理动态危害时往往是被动的而非预见性的。最近大语言模型和视觉-语言模型（VLMs）的预训练进展为预见性地避免危害创造了新的机会。在本工作中，我们提出了一种零样本语言作为成本映射框架，利用VLMs解释视觉场景、评估潜在的动态风险，并预先分配风险意识导航成本，使机器人能够在危害出现之前预见危害。通过将这种基于语言的成本图与几何障碍图集成，机器人不仅识别现有障碍，还能预见并主动规划避开由环境动态引起的各种潜在危害。在模拟和多样化动态环境中的实验表明，所提出的方法在导航成功率和减少危害接触方面明显优于被动基准规划器。代码和补充材料可在以下网址获取。 

---
# Safety-Aware Imitation Learning via MPC-Guided Disturbance Injection 

**Title (ZH)**: 基于MPC引导干扰注入的 Awareness 安全意识模仿学习 

**Authors**: Le Qiu, Yusuf Umut Ciftci, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2508.03129)  

**Abstract**: Imitation Learning has provided a promising approach to learning complex robot behaviors from expert demonstrations. However, learned policies can make errors that lead to safety violations, which limits their deployment in safety-critical applications. We propose MPC-SafeGIL, a design-time approach that enhances the safety of imitation learning by injecting adversarial disturbances during expert demonstrations. This exposes the expert to a broader range of safety-critical scenarios and allows the imitation policy to learn robust recovery behaviors. Our method uses sampling-based Model Predictive Control (MPC) to approximate worst-case disturbances, making it scalable to high-dimensional and black-box dynamical systems. In contrast to prior work that relies on analytical models or interactive experts, MPC-SafeGIL integrates safety considerations directly into data collection. We validate our approach through extensive simulations including quadruped locomotion and visuomotor navigation and real-world experiments on a quadrotor, demonstrating improvements in both safety and task performance. See our website here: this https URL 

**Abstract (ZH)**: 模仿学习为从专家演示中学习复杂机器人行为提供了有希望的方法。然而，学习到的策略可能会出现导致安全违规的错误，这限制了它们在安全关键应用中的部署。我们提出了一种MPCT-SafeGIL方法，在专家演示期间注入对抗性干扰以增强模仿学习的安全性。这使专家能够接触到更多类型的安全关键场景，并使模仿策略能够学会稳健的恢复行为。该方法使用基于采样的模型预测控制（MPC）来近似最坏情况干扰，使其能够适用于高维和黑盒动力学系统。与依赖于分析模型或交互式专家的先前工作不同，MPCT-SafeGIL将安全考虑直接集成到数据收集过程中。我们通过在四足运动和视知觉导航仿真以及四旋翼实际实验中的广泛验证，展示了在安全性和任务性能方面的改进。更多详情请参见我们的网站: <https://this.is/URL>。 

---
# Point2Act: Efficient 3D Distillation of Multimodal LLMs for Zero-Shot Context-Aware Grasping 

**Title (ZH)**: Point2Act: 高效的多模态LLM的3D知识蒸馏用于零样本上下文感知抓取 

**Authors**: Sang Min Kim, Hyeongjun Heo, Junho Kim, Yonghyeon Lee, Young Min Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.03099)  

**Abstract**: We propose Point2Act, which directly retrieves the 3D action point relevant for a contextually described task, leveraging Multimodal Large Language Models (MLLMs). Foundation models opened the possibility for generalist robots that can perform a zero-shot task following natural language descriptions within an unseen environment. While the semantics obtained from large-scale image and language datasets provide contextual understanding in 2D images, the rich yet nuanced features deduce blurry 2D regions and struggle to find precise 3D locations for actions. Our proposed 3D relevancy fields bypass the high-dimensional features and instead efficiently imbue lightweight 2D point-level guidance tailored to the task-specific action. The multi-view aggregation effectively compensates for misalignments due to geometric ambiguities, such as occlusion, or semantic uncertainties inherent in the language descriptions. The output region is highly localized, reasoning fine-grained 3D spatial context that can directly transfer to an explicit position for physical action at the on-the-fly reconstruction of the scene. Our full-stack pipeline, which includes capturing, MLLM querying, 3D reconstruction, and grasp pose extraction, generates spatially grounded responses in under 20 seconds, facilitating practical manipulation tasks. Project page: this https URL 

**Abstract (ZH)**: 我们提出Point2Act，这是一种直接从上下文描述的任务中检索出相关3D行动点的方法，利用多模态大规模语言模型（MLLMs）。基础模型开启了通用机器人执行零样本任务的可能性，这些任务可以在未见过的环境中按照自然语言描述来完成。虽然大规模图像和语言数据集中的语义信息可以在2D图像中提供上下文理解，但丰富的yet细腻的特征会导致模糊的2D区域，并难以找到精确的3D行动位置。我们提出的3D相关性字段避开了高维度特征，而是高效地赋予任务特定行动所需的轻量级2D点级指导。多视角聚合有效补偿了由于几何歧义（如遮挡）或语言描述中的语义不确定性导致的对齐偏差。输出区域高度局部化，能直接推理出细粒度的3D空间上下文，并在场景即时重建时直接转换为物理行动的明确位置。我们的全流程管线，包括捕捉、MLLM查询、3D重建和抓取姿态提取，在不到20秒的时间内生成空间上接地的响应，便于实际操作任务。项目页面：这个 https URL。 

---
# Optimizing Bipedal Locomotion for The 100m Dash With Comparison to Human Running 

**Title (ZH)**: 100米 dash中双足运动优化及其与人类跑步的比较 

**Authors**: Devin Crowley, Jeremy Dao, Helei Duan, Kevin Green, Jonathan Hurst, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2508.03070)  

**Abstract**: In this paper, we explore the space of running gaits for the bipedal robot Cassie. Our first contribution is to present an approach for optimizing gait efficiency across a spectrum of speeds with the aim of enabling extremely high-speed running on hardware. This raises the question of how the resulting gaits compare to human running mechanics, which are known to be highly efficient in comparison to quadrupeds. Our second contribution is to conduct this comparison based on established human biomechanical studies. We find that despite morphological differences between Cassie and humans, key properties of the gaits are highly similar across a wide range of speeds. Finally, our third contribution is to integrate the optimized running gaits into a full controller that satisfies the rules of the real-world task of the 100m dash, including starting and stopping from a standing position. We demonstrate this controller on hardware to establish the Guinness World Record for Fastest 100m by a Bipedal Robot. 

**Abstract (ZH)**: 本文探讨了双足机器人Cassie的跑步姿态空间。我们的第一项贡献是提出了一种在不同速度范围内优化步态效率的方法，旨在使硬件实现极高速度的跑步成为可能。这引发了对所产生步态与已知比四足动物更为高效的-human跑步力学之间的比较问题。我们的第二项贡献是基于已确立的人体生物力学研究进行这种比较。我们发现，尽管Cassie和人类的形态存在差异，但在广泛的速度范围内，步态的关键特性非常相似。最后，我们的第三项贡献是将优化后的跑步步态集成到一个满足百米赛跑真实世界任务规则的完整控制器中，包括站立起跑和停止。我们在这项硬件上验证了该控制器，并建立了双足机器人最快百米跑的吉尼斯世界纪录。 

---
# Hand-Eye Autonomous Delivery: Learning Humanoid Navigation, Locomotion and Reaching 

**Title (ZH)**: 自主手眼协作配送：学习类人导航、运动和取放 

**Authors**: Sirui Chen, Yufei Ye, Zi-Ang Cao, Jennifer Lew, Pei Xu, C. Karen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03068)  

**Abstract**: We propose Hand-Eye Autonomous Delivery (HEAD), a framework that learns navigation, locomotion, and reaching skills for humanoids, directly from human motion and vision perception data. We take a modular approach where the high-level planner commands the target position and orientation of the hands and eyes of the humanoid, delivered by the low-level policy that controls the whole-body movements. Specifically, the low-level whole-body controller learns to track the three points (eyes, left hand, and right hand) from existing large-scale human motion capture data while high-level policy learns from human data collected by Aria glasses. Our modular approach decouples the ego-centric vision perception from physical actions, promoting efficient learning and scalability to novel scenes. We evaluate our method both in simulation and in the real-world, demonstrating humanoid's capabilities to navigate and reach in complex environments designed for humans. 

**Abstract (ZH)**: 基于人类运动和视觉感知数据的人形机器人自主配送框架：Hand-Eye Autonomous Delivery (HEAD) 

---
# SkeNa: Learning to Navigate Unseen Environments Based on Abstract Hand-Drawn Maps 

**Title (ZH)**: SkeNa: 基于抽象手绘地图学习导航 unseen 环境 

**Authors**: Haojun Xu, Jiaqi Xiang, Wu Wei, Jinyu Chen, Linqing Zhong, Linjiang Huang, Hongyu Yang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03053)  

**Abstract**: A typical human strategy for giving navigation guidance is to sketch route maps based on the environmental layout. Inspired by this, we introduce Sketch map-based visual Navigation (SkeNa), an embodied navigation task in which an agent must reach a goal in an unseen environment using only a hand-drawn sketch map as guidance. To support research for SkeNa, we present a large-scale dataset named SoR, comprising 54k trajectory and sketch map pairs across 71 indoor scenes. In SoR, we introduce two navigation validation sets with varying levels of abstraction in hand-drawn sketches, categorized based on their preservation of spatial scales in the environment, to facilitate future research. To construct SoR, we develop an automated sketch-generation pipeline that efficiently converts floor plans into hand-drawn representations. To solve SkeNa, we propose SkeNavigator, a navigation framework that aligns visual observations with hand-drawn maps to estimate navigation targets. It employs a Ray-based Map Descriptor (RMD) to enhance sketch map valid feature representation using equidistant sampling points and boundary distances. To improve alignment with visual observations, a Dual-Map Aligned Goal Predictor (DAGP) leverages the correspondence between sketch map features and on-site constructed exploration map features to predict goal position and guide navigation. SkeNavigator outperforms prior floor plan navigation methods by a large margin, improving SPL on the high-abstract validation set by 105% relatively. Our code and dataset will be released. 

**Abstract (ZH)**: 基于草图的地图导向导航方法：基于草图的地图引导视觉导航（S a large-scale-scale scale dataset SoR comprising 1 4on trajectory and sketch map pairs across 7 indoor scenes on introducing two validation sets with varying levels of abstraction based to facilitate the introduction of an automated sketch-generation pipeline that efficiently converts floor plans to hand-drawn representationsD and introducing a framework called SkeNavigator that that align visual observations with on-drawn maps to estimate estimate targets. 

---
# Aerobatic maneuvers in insect-scale flapping-wing aerial robots via deep-learned robust tube model predictive control 

**Title (ZH)**: 昆虫尺度拍翼飞行机器人中的深度学习鲁棒管模型预测控制 aerial动态研究 

**Authors**: Yi-Hsuan Hsiao, Andrea Tagliabue, Owen Matteson, Suhan Kim, Tong Zhao, Jonathan P. How, YuFeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.03043)  

**Abstract**: Aerial insects exhibit highly agile maneuvers such as sharp braking, saccades, and body flips under disturbance. In contrast, insect-scale aerial robots are limited to tracking non-aggressive trajectories with small body acceleration. This performance gap is contributed by a combination of low robot inertia, fast dynamics, uncertainty in flapping-wing aerodynamics, and high susceptibility to environmental disturbance. Executing highly dynamic maneuvers requires the generation of aggressive flight trajectories that push against the hardware limit and a high-rate feedback controller that accounts for model and environmental uncertainty. Here, through designing a deep-learned robust tube model predictive controller, we showcase insect-like flight agility and robustness in a 750-millgram flapping-wing robot. Our model predictive controller can track aggressive flight trajectories under disturbance. To achieve a high feedback rate in a compute-constrained real-time system, we design imitation learning methods to train a two-layer, fully connected neural network, which resembles insect flight control architecture consisting of central nervous system and motor neurons. Our robot demonstrates insect-like saccade movements with lateral speed and acceleration of 197 centimeters per second and 11.7 meters per second square, representing 447$\%$ and 255$\%$ improvement over prior results. The robot can also perform saccade maneuvers under 160 centimeters per second wind disturbance and large command-to-force mapping errors. Furthermore, it performs 10 consecutive body flips in 11 seconds - the most challenging maneuver among sub-gram flyers. These results represent a milestone in achieving insect-scale flight agility and inspire future investigations on sensing and compute autonomy. 

**Abstract (ZH)**: 基于深度学习的鲁棒管模型预测控制在750毫克拍翼机器人上的昆虫般飞行敏捷性和鲁棒性 

---
# CogniPlan: Uncertainty-Guided Path Planning with Conditional Generative Layout Prediction 

**Title (ZH)**: CogniPlan：基于条件生成布局预测的不确定性引导路径规划 

**Authors**: Yizhuo Wang, Haodong He, Jingsong Liang, Yuhong Cao, Ritabrata Chakraborty, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2508.03027)  

**Abstract**: Path planning in unknown environments is a crucial yet inherently challenging capability for mobile robots, which primarily encompasses two coupled tasks: autonomous exploration and point-goal navigation. In both cases, the robot must perceive the environment, update its belief, and accurately estimate potential information gain on-the-fly to guide planning. In this work, we propose CogniPlan, a novel path planning framework that leverages multiple plausible layouts predicted by a COnditional GeNerative Inpainting model, mirroring how humans rely on cognitive maps during navigation. These predictions, based on the partially observed map and a set of layout conditioning vectors, enable our planner to reason effectively under uncertainty. We demonstrate strong synergy between generative image-based layout prediction and graph-attention-based path planning, allowing CogniPlan to combine the scalability of graph representations with the fidelity and predictiveness of occupancy maps, yielding notable performance gains in both exploration and navigation. We extensively evaluate CogniPlan on two datasets (hundreds of maps and realistic floor plans), consistently outperforming state-of-the-art planners. We further deploy it in a high-fidelity simulator and on hardware, showcasing its high-quality path planning and real-world applicability. 

**Abstract (ZH)**: 未知环境下的路径规划是移动机器人的一项关键且固有的挑战性能力，主要涵盖两个互相关联的任务：自主探索和点目标导航。在两种情况下，机器人必须感知环境、更新其信念，并在规划过程中实时准确估计潜在信息增益以指导路径规划。本文提出了一种名为CogniPlan的新型路径规划框架，该框架利用条件生成修复模型预测的多个可能布局，模仿人类在导航时依靠认知地图的方式。基于部分观察地图和一系列布局条件向量的这些预测，使我们的规划器能够在不确定性下进行有效的推理。我们展示了基于生成图像的布局预测与基于图注意力的路径规划之间的强协同作用，允许CogniPlan结合图表示的可扩展性与占用地图的准确性和预测性，在探索和导航中均取得了显著的性能提升。我们在两个数据集（数百张地图和现实楼层平面图）上对CogniPlan进行了广泛评估，持续优于最先进的规划器。此外，我们在高保真模拟器和硬件上部署了CogniPlan，展示了其高质量的路径规划能力和实际应用场景适用性。 

---
# LiGen: GAN-Augmented Spectral Fingerprinting for Indoor Positioning 

**Title (ZH)**: LiGen: 基于GAN增强的光谱指纹定位方法 

**Authors**: Jie Lin, Hsun-Yu Lee, Ho-Ming Li, Fang-Jing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03024)  

**Abstract**: Accurate and robust indoor localization is critical for smart building applications, yet existing Wi-Fi-based systems are often vulnerable to environmental conditions. This work presents a novel indoor localization system, called LiGen, that leverages the spectral intensity patterns of ambient light as fingerprints, offering a more stable and infrastructure-free alternative to radio signals. To address the limited spectral data, we design a data augmentation framework based on generative adversarial networks (GANs), featuring two variants: PointGAN, which generates fingerprints conditioned on coordinates, and FreeGAN, which uses a weak localization model to label unconditioned samples. Our positioning model, leveraging a Multi-Layer Perceptron (MLP) architecture to train on synthesized data, achieves submeter-level accuracy, outperforming Wi-Fi-based baselines by over 50\%. LiGen also demonstrates strong robustness in cluttered environments. To the best of our knowledge, this is the first system to combine spectral fingerprints with GAN-based data augmentation for indoor localization. 

**Abstract (ZH)**: 基于光谱强度模式的鲁棒室内定位系统LiGen 

---
# Thruster-Enhanced Locomotion: A Decoupled Model Predictive Control with Learned Contact Residuals 

**Title (ZH)**: 推进器增强移动：解耦模型预测控制与学习的接触残差 

**Authors**: Chenghao Wang, Alireza Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2508.03003)  

**Abstract**: Husky Carbon, a robot developed by Northeastern University, serves as a research platform to explore unification of posture manipulation and thrust vectoring. Unlike conventional quadrupeds, its joint actuators and thrusters enable enhanced control authority, facilitating thruster-assisted narrow-path walking. While a unified Model Predictive Control (MPC) framework optimizing both ground reaction forces and thruster forces could theoretically address this control problem, its feasibility is limited by the low torque-control bandwidth of the system's lightweight actuators. To overcome this challenge, we propose a decoupled control architecture: a Raibert-type controller governs legged locomotion using position-based control, while an MPC regulates the thrusters augmented by learned Contact Residual Dynamics (CRD) to account for leg-ground impacts. This separation bypasses the torque-control rate bottleneck while retaining the thruster MPC to explicitly account for leg-ground impact dynamics through learned residuals. We validate this approach through both simulation and hardware experiments, showing that the decoupled control architecture with CRD performs more stable behavior in terms of push recovery and cat-like walking gait compared to the decoupled controller without CRD. 

**Abstract (ZH)**: Huskykyky碳：东北大学研制的机器人旨在探究姿态操控与推力矢量统一。不同于传统四DD多足四DD驱动关节执行器和和推驱动推进器D的，设计Hussyskyky，，D，D，，D，DD，DHD_DD，，D.D为您提供了更强的D动态DD度由于促进了助力窄路径行走DD我们本研究提出提出提出提出提出D提出提出开发了一种D基于预测模型预测控制(M(M的(D（D的方法D最大限度地优化优化优化了地面反和和D和推力力D动力DDD然而D由于系统系统D动力系统的扭矩控制带宽限制DD本文D提出了一个解D解解耦D控制架构DD一种拉式特型D控制DD负责腿部运动DD另一种D基于位置的的D预测模型预测控制DDD通过学习接触残余动态D来D来准确对该方法进行了仿真和DDD硬件实验验证DDDD在DDD解耦D控制架构DD下表现出更好的恢复能力DDD以及类似狗的D行走姿态DDDD 

---
# GACL: Grounded Adaptive Curriculum Learning with Active Task and Performance Monitoring 

**Title (ZH)**: 基于监控和适应性任务与性能监控的 grounded 调适性课程学习 

**Authors**: Linji Wang, Zifan Xu, Peter Stone, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02988)  

**Abstract**: Curriculum learning has emerged as a promising approach for training complex robotics tasks, yet current applications predominantly rely on manually designed curricula, which demand significant engineering effort and can suffer from subjective and suboptimal human design choices. While automated curriculum learning has shown success in simple domains like grid worlds and games where task distributions can be easily specified, robotics tasks present unique challenges: they require handling complex task spaces while maintaining relevance to target domain distributions that are only partially known through limited samples. To this end, we propose Grounded Adaptive Curriculum Learning, a framework specifically designed for robotics curriculum learning with three key innovations: (1) a task representation that consistently handles complex robot task design, (2) an active performance tracking mechanism that allows adaptive curriculum generation appropriate for the robot's current capabilities, and (3) a grounding approach that maintains target domain relevance through alternating sampling between reference and synthetic tasks. We validate GACL on wheeled navigation in constrained environments and quadruped locomotion in challenging 3D confined spaces, achieving 6.8% and 6.1% higher success rates, respectively, than state-of-the-art methods in each domain. 

**Abstract (ZH)**: 基于地面适应性 Curriculum 学习的机器人任务训练 

---
# Estimation of Aerodynamics Forces in Dynamic Morphing Wing Flight 

**Title (ZH)**: 动态变形翼飞行的气动 forces 估计 

**Authors**: Bibek Gupta, Mintae Kim, Albert Park, Eric Sihite, Koushil Sreenath, Alireza Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02984)  

**Abstract**: Accurate estimation of aerodynamic forces is essential for advancing the control, modeling, and design of flapping-wing aerial robots with dynamic morphing capabilities. In this paper, we investigate two distinct methodologies for force estimation on Aerobat, a bio-inspired flapping-wing platform designed to emulate the inertial and aerodynamic behaviors observed in bat flight. Our goal is to quantify aerodynamic force contributions during tethered flight, a crucial step toward closed-loop flight control. The first method is a physics-based observer derived from Hamiltonian mechanics that leverages the concept of conjugate momentum to infer external aerodynamic forces acting on the robot. This observer builds on the system's reduced-order dynamic model and utilizes real-time sensor data to estimate forces without requiring training data. The second method employs a neural network-based regression model, specifically a multi-layer perceptron (MLP), to learn a mapping from joint kinematics, flapping frequency, and environmental parameters to aerodynamic force outputs. We evaluate both estimators using a 6-axis load cell in a high-frequency data acquisition setup that enables fine-grained force measurements during periodic wingbeats. The conjugate momentum observer and the regression model demonstrate strong agreement across three force components (Fx, Fy, Fz). 

**Abstract (ZH)**: 准确估算气动 forces 对推进具有动态变形能力的拍翼飞行机器人控制、建模和设计至关重要。本文探讨了两种用于 Aerobat（一种受 bat 飞行行为启发的拍翼飞行平台）气动 force 估算的方法。我们的目标是在牵索飞行中量化气动 force 贡献，这是实现闭环飞行控制的关键步骤。第一种方法是基于物理的观察器，来自哈密尔顿力学，利用共轭动量的概念来推断作用在机器人上的外部气动 force。该观察器基于系统的降阶动力学模型，并利用实时传感器数据进行 force 估计，无需训练数据。第二种方法采用基于神经网络的回归模型，具体为多层感知器（MLP），从关节运动学、拍动频率和环境参数到气动 force 输出建立映射关系。我们使用六轴载荷传感器在高频率数据采集设置中评估两种估算器，该设置能够提供周期性拍打过程中精细的 force 测量。共轭动量观察器和回归模型在三个 force 组件（Fx, Fy, Fz）上表现出较强的一致性。 

---
# Multimodal Human-Intent Modeling for Contextual Robot-to-Human Handovers of Arbitrary Objects 

**Title (ZH)**: 多模态人类意图建模以实现任意物体的上下文机器人到人类的手递 

**Authors**: Lucas Chen, Guna Avula, Hanwen Ren, Zixing Wang, Ahmed H. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02982)  

**Abstract**: Human-robot object handover is a crucial element for assistive robots that aim to help people in their daily lives, including elderly care, hospitals, and factory floors. The existing approaches to solving these tasks rely on pre-selected target objects and do not contextualize human implicit and explicit preferences for handover, limiting natural and smooth interaction between humans and robots. These preferences can be related to the target object selection from the cluttered environment and to the way the robot should grasp the selected object to facilitate desirable human grasping during handovers. Therefore, this paper presents a unified approach that selects target distant objects using human verbal and non-verbal commands and performs the handover operation by contextualizing human implicit and explicit preferences to generate robot grasps and compliant handover motion sequences. We evaluate our integrated framework and its components through real-world experiments and user studies with arbitrary daily-life objects. The results of these evaluations demonstrate the effectiveness of our proposed pipeline in handling object handover tasks by understanding human preferences. Our demonstration videos can be found at this https URL. 

**Abstract (ZH)**: 人类与机器人物体交接是旨在帮助人们日常生活的辅助机器人的一项 crucial 元素，包括老年人护理、医院和工厂地板。现有的方法依赖于预先选定的目标物体，并未能将人类的显式和隐式交接偏好纳入考量，从而限制了人类与机器人之间的自然和流畅互动。这些偏好与从杂乱环境中选择目标物体以及机器人如何抓取选定物体以促进顺利交接有关。因此，本文提出了一种统一的方法，使用人类的口头和非口头命令选择远距离目标物体，并通过考虑人类的显式和隐式偏好来执行交接操作以生成机器人的抓取和符合人体工程学的交接运动序列。我们通过随意的日常生活物体的真实世界实验和用户研究评估了我们集成框架及其组件。这些评估的结果证明了我们提出的处理物体交接任务的管道的有效性。我们的演示视频可以在以下链接找到：this https URL。 

---
# Physics-informed Neural Time Fields for Prehensile Object Manipulation 

**Title (ZH)**: 基于物理的神经时间场的手部灵巧对象操控 

**Authors**: Hanwen Ren, Ruiqi Ni, Ahmed H. Qureshi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02976)  

**Abstract**: Object manipulation skills are necessary for robots operating in various daily-life scenarios, ranging from warehouses to hospitals. They allow the robots to manipulate the given object to their desired arrangement in the cluttered environment. The existing approaches to solving object manipulations are either inefficient sampling based techniques, require expert demonstrations, or learn by trial and error, making them less ideal for practical scenarios. In this paper, we propose a novel, multimodal physics-informed neural network (PINN) for solving object manipulation tasks. Our approach efficiently learns to solve the Eikonal equation without expert data and finds object manipulation trajectories fast in complex, cluttered environments. Our method is multimodal as it also reactively replans the robot's grasps during manipulation to achieve the desired object poses. We demonstrate our approach in both simulation and real-world scenarios and compare it against state-of-the-art baseline methods. The results indicate that our approach is effective across various objects, has efficient training compared to previous learning-based methods, and demonstrates high performance in planning time, trajectory length, and success rates. Our demonstration videos can be found at this https URL. 

**Abstract (ZH)**: 机器人在仓库到医院等各种日常生活场景下的物体操作技能是必要的。它们使机器人能够将给定物体在拥挤环境中重新排列到期望的布局。现有的物体操作解决方案要么基于低效的采样技术，要么需要专家演示，或者通过试错学习，这使得它们在实际场景中不太理想。在本文中，我们提出了一种新颖的、多模态物理知情神经网络（PINN）来解决物体操作任务。我们的方法能够高效地学习解决Eikonal方程，而无需专家数据，并能够快速在复杂拥挤环境中找到物体操作轨迹。我们的方法是多模态的，因为它还在操作过程中reactively重新规划机器人的抓取，以实现期望的物体姿态。我们在模拟和真实世界场景中展示了我们的方法，并将其与最先进的基线方法进行了比较。结果表明，我们的方法在不同物体上都有效，与之前的学习方法相比，具有更高效的训练，并在规划时间、轨迹长度和成功率上表现出色。我们的演示视频可在以下链接找到：这个 https URL。 

---
# Robot builds a robot's brain: AI generated drone command and control station hosted in the sky 

**Title (ZH)**: 机器人构建另一机器人的大脑：AI生成的无人机指挥控制站悬停于空中 

**Authors**: Peter Burke  

**Link**: [PDF](https://arxiv.org/pdf/2508.02962)  

**Abstract**: Advances in artificial intelligence (AI) including large language models (LLMs) and hybrid reasoning models present an opportunity to reimagine how autonomous robots such as drones are designed, developed, and validated. Here, we demonstrate a fully AI-generated drone control system: with minimal human input, an artificial intelligence (AI) model authored all the code for a real-time, self-hosted drone command and control platform, which was deployed and demonstrated on a real drone in flight as well as a simulated virtual drone in the cloud. The system enables real-time mapping, flight telemetry, autonomous mission planning and execution, and safety protocolsall orchestrated through a web interface hosted directly on the drone itself. Not a single line of code was written by a human. We quantitatively benchmark system performance, code complexity, and development speed against prior, human-coded architectures, finding that AI-generated code can deliver functionally complete command-and-control stacks at orders-of-magnitude faster development cycles, though with identifiable current limitations related to specific model context window and reasoning depth. Our analysis uncovers the practical boundaries of AI-driven robot control code generation at current model scales, as well as emergent strengths and failure modes in AI-generated robotics code. This work sets a precedent for the autonomous creation of robot control systems and, more broadly, suggests a new paradigm for robotics engineeringone in which future robots may be largely co-designed, developed, and verified by artificial intelligence. In this initial work, a robot built a robot's brain. 

**Abstract (ZH)**: 人工智能进展包括大型语言模型和混合推理模型为重新想象自主机器人如无人机的设计、开发和验证提供了机会。在此，我们演示了一个完全由人工智能生成的无人机控制系统：在 minimal 人类输入的情况下，一个人工智能模型编写了全部代码，构建了一个实时、自我托管的无人机命令与控制系统，并在真实飞行的无人机及云端模拟的虚拟无人机上进行了部署和演示。该系统通过无人机本身直接托管的网页界面实现了实时地图测绘、飞行遥测、自主任务规划与执行以及安全协议的协调。没有一行代码是由人类编写的。我们定量地将系统性能、代码复杂性和开发速度与先前的人工编码架构进行了基准测试，发现人工智能生成的代码能够在比以往快几个数量级的开发周期内交付功能完备的命令与控制堆栈，尽管目前存在特定模型上下文窗口和推理深度相关的识别局限。我们的分析揭示了当前模型规模下人工智能驱动的机器人控制代码生成的实际边界，以及人工智能生成的机器人代码中的新兴优势和失败模式。这项工作为自主创建机器人控制系统设立了先例，并更广泛地建议了机器人工程的一种新范式——即未来机器人可能主要由人工智能进行协同设计、开发和验证。在这项初步工作中，机器人构建了自己大脑。 

---
# Optimal Trajectory Planning in a Vertically Undulating Snake Locomotion using Contact-implicit Optimization 

**Title (ZH)**: 垂直波动蛇形运动中具有接触显式优化的最优轨迹规划 

**Authors**: Adarsh Salagame, Eric Sihite, Alireza Ramezani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02953)  

**Abstract**: Contact-rich problems, such as snake robot locomotion, offer unexplored yet rich opportunities for optimization-based trajectory and acyclic contact planning. So far, a substantial body of control research has focused on emulating snake locomotion and replicating its distinctive movement patterns using shape functions that either ignore the complexity of interactions or focus on complex interactions with matter (e.g., burrowing movements). However, models and control frameworks that lie in between these two paradigms and are based on simple, fundamental rigid body dynamics, which alleviate the challenging contact and control allocation problems in snake locomotion, remain absent. This work makes meaningful contributions, substantiated by simulations and experiments, in the following directions: 1) introducing a reduced-order model based on Moreau's stepping-forward approach from differential inclusion mathematics, 2) verifying model accuracy, 3) experimental validation. 

**Abstract (ZH)**: 基于接触优化的游蛇机器人运动轨迹和非循环接触规划提供了未探究但丰富的机遇。现有控制研究主要致力于模拟游蛇运动并利用形状函数复制其独特的运动模式，这些形状函数要么忽略交互的复杂性，要么专注于与物质的复杂相互作用（如掘土运动）。然而，基于简单基本刚体动力学且能缓解游蛇运动中接触和控制分配难题的介于这两种范式之间的模型和控制框架仍不存在。本文通过仿真和实验验证，在以下几个方面做出了有意义的贡献：1）引入基于Moreau跳跃前进方法的降阶模型，2）验证模型准确性，3）实验验证。 

---
# A novel autonomous microplastics surveying robot for beach environments 

**Title (ZH)**: 一种新型自主海滩微塑料探测机器人 

**Authors**: Hassan Iqbal, Kobiny Rex, Joseph Shirley, Carlos Baiz, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2508.02952)  

**Abstract**: Microplastics, defined as plastic particles smaller than 5 millimeters, have become a pervasive environmental contaminant that accumulates on beaches due to wind patterns and tidal forcing. Detecting microplastics and mapping their concentration in the wild remains one of the primary challenges in addressing this environmental issue. This paper introduces a novel robotic platform that automatically detects and chemically analyzes microplastics on beach surfaces. This mobile manipulator system scans areas for microplastics using a camera mounted on the robotic arm's end effector. The system effectively segments candidate microplastic particles on sand surfaces even in the presence of organic matter such as leaves and clams. Once a candidate microplastic particle is detected, the system steers a near-infrared (NIR) spectroscopic sensor onto the particle using both NIR and visual feedback to chemically analyze it in real-time. Through experiments in lab and beach environments, the system is shown to achieve an excellent positional precision in manipulation control and high microplastic classification accuracy. 

**Abstract (ZH)**: 微塑料：一种定义为小于5毫米的塑料颗粒，已成为由于风Pattern和潮汐作用在海滩上普遍积累的环境污染物。在野外检测微塑料并绘制其浓度分布仍然是处理这一环境问题的主要挑战之一。本文介绍了一种新型机器人平台，该平台能够自动检测和化学分析海滩表面的微塑料。该便携式 manipulator 系统使用安装在机器人手臂末端执行器上的摄像头扫描区域以检测微塑料。即使存在有机物如树叶和蛤蜊，该系统也能有效对沙面上的候选微塑料颗粒进行分割。一旦检测到候选微塑料颗粒，该系统将使用近红外（NIR）光谱传感器及其视觉反馈实时对其化学分析。通过实验室和海滩环境中的实验，该系统在操作控制定位精度和微塑料分类准确性方面表现出色。 

---
# AeroSafe: Mobile Indoor Air Purification using Aerosol Residence Time Analysis and Robotic Cough Emulator Testbed 

**Title (ZH)**: AeroSafe: 基于气溶胶驻留时间分析与机器人咳嗽模拟测试平台的移动室内空气净化技术 

**Authors**: M Tanjid Hasan Tonmoy, Rahath Malladi, Kaustubh Singh, Forsad Al Hossain, Rajesh Gupta, Andrés E. Tejada-Martínez, Tauhidur Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2508.02947)  

**Abstract**: Indoor air quality plays an essential role in the safety and well-being of occupants, especially in the context of airborne diseases. This paper introduces AeroSafe, a novel approach aimed at enhancing the efficacy of indoor air purification systems through a robotic cough emulator testbed and a digital-twins-based aerosol residence time analysis. Current portable air filters often overlook the concentrations of respiratory aerosols generated by coughs, posing a risk, particularly in high-exposure environments like healthcare facilities and public spaces. To address this gap, we present a robotic dual-agent physical emulator comprising a maneuverable mannequin simulating cough events and a portable air purifier autonomously responding to aerosols. The generated data from this emulator trains a digital twins model, combining a physics-based compartment model with a machine learning approach, using Long Short-Term Memory (LSTM) networks and graph convolution layers. Experimental results demonstrate the model's ability to predict aerosol concentration dynamics with a mean residence time prediction error within 35 seconds. The proposed system's real-time intervention strategies outperform static air filter placement, showcasing its potential in mitigating airborne pathogen risks. 

**Abstract (ZH)**: 室内空气质量在保障居民安全与福祉方面起着关键作用，尤其是在呼吸道疾病传播的背景下。本文介绍了AeroSafe，这是一种通过机器人咳嗽仿真测试床和基于数字孪生的气溶胶停留时间分析来提高室内空气净化系统有效性的新颖方法。当前便携式空气净化器往往忽视了咳嗽产生的呼吸气溶胶浓度，特别是在高暴露环境下如医疗机构和公共空间，这种忽视带来了风险。为解决这一问题，我们提出了一种机器人双实体物理仿真器，包括一个可操控的人体模型模拟咳嗽事件，以及一个自主响应气溶胶的便携式空气净化器。来自此仿真器生成的数据训练了一个结合了物理学分室模型和机器学习方法的数字孪生模型，使用了长短期记忆（LSTM）网络和图卷积层。实验结果表明该模型能够以35秒内的均停留时间预测误差预测气溶胶浓度动态。所提出的系统的实时干预策略优于静态空气净化器布局，展示了其在减轻气传病原体风险方面的潜力。 

---
# Model-agnostic Meta-learning for Adaptive Gait Phase and Terrain Geometry Estimation with Wearable Soft Sensors 

**Title (ZH)**: 无模型泛化元学习在可穿戴软传感器辅助自适应步态相位和地形几何估计中的应用 

**Authors**: Zenan Zhu, Wenxi Chen, Pei-Chun Kao, Janelle Clark, Lily Behnke, Rebecca Kramer-Bottiglio, Holly Yanco, Yan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02930)  

**Abstract**: This letter presents a model-agnostic meta-learning (MAML) based framework for simultaneous and accurate estimation of human gait phase and terrain geometry using a small set of fabric-based wearable soft sensors, with efficient adaptation to unseen subjects and strong generalization across different subjects and terrains. Compared to rigid alternatives such as inertial measurement units, fabric-based soft sensors improve comfort but introduce nonlinearities due to hysteresis, placement error, and fabric deformation. Moreover, inter-subject and inter-terrain variability, coupled with limited calibration data in real-world deployments, further complicate accurate estimation. To address these challenges, the proposed framework integrates MAML into a deep learning architecture to learn a generalizable model initialization that captures subject- and terrain-invariant structure. This initialization enables efficient adaptation (i.e., adaptation with only a small amount of calibration data and a few fine-tuning steps) to new users, while maintaining strong generalization (i.e., high estimation accuracy across subjects and terrains). Experiments on nine participants walking at various speeds over five terrain conditions demonstrate that the proposed framework outperforms baseline approaches in estimating gait phase, locomotion mode, and incline angle, with superior accuracy, adaptation efficiency, and generalization. 

**Abstract (ZH)**: 基于MAML的新型织物基可穿戴软传感器框架：同时准确估计人类步态相位和地形几何结构 

---
# Context-aware Risk Assessment and Its Application in Autonomous Driving 

**Title (ZH)**: 基于上下文的风险评估及其在自主驾驶中的应用 

**Authors**: Boyang Tian, Weisong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02919)  

**Abstract**: Ensuring safety in autonomous driving requires precise, real-time risk assessment and adaptive behavior. Prior work on risk estimation either outputs coarse, global scene-level metrics lacking interpretability, proposes indicators without concrete integration into autonomous systems, or focuses narrowly on specific driving scenarios. We introduce the Context-aware Risk Index (CRI), a light-weight modular framework that quantifies directional risks based on object kinematics and spatial relationships, dynamically adjusting control commands in real time. CRI employs direction-aware spatial partitioning within a dynamic safety envelope using Responsibility-Sensitive Safety (RSS) principles, a hybrid probabilistic-max fusion strategy for risk aggregation, and an adaptive control policy for real-time behavior modulation. We evaluate CRI on the Bench2Drive benchmark comprising 220 safety-critical scenarios using a state-of-the-art end-to-end model Transfuser++ on challenging routes. Our collision-rate metrics show a 19\% reduction (p = 0.003) in vehicle collisions per failed route, a 20\% reduction (p = 0.004) in collisions per kilometer, a 17\% increase (p = 0.016) in composed driving score, and a statistically significant reduction in penalty scores (p = 0.013) with very low overhead (3.6 ms per decision cycle). These results demonstrate that CRI substantially improves safety and robustness in complex, risk-intensive environments while maintaining modularity and low runtime overhead. 

**Abstract (ZH)**: 确保自动驾驶安全需要精确的实时风险评估和适应性行为。我们引入了上下文感知风险指数（CRI），这是一种基于轻量级模块化框架，根据物体动力学和空间关系量化方向性风险，并实时动态调整控制命令。CRI采用责任敏感安全（RSS）原则内的方向感知空间分区，在动态安全包络内，采用混合概率-最大化融合策略进行风险聚合，并采用适应性控制策略进行实时行为调节。我们在包含220个安全关键场景的Bench2Drive基准上评估了CRI，使用最先进的端到端模型Transfuser++在具有挑战性的路段上进行评估。我们的碰撞率指标显示，CRI使每条失败路线的车辆碰撞减少19%（p = 0.003）、每公里碰撞减少20%（p = 0.004）、综合驾驶评分增加17%（p = 0.016），并且显著降低了惩罚分（p = 0.013），同时具有极低的运行时开销（每决策周期3.6毫秒）。这些结果表明，CRI在复杂、高风险环境中显著提高了安全性和鲁棒性，同时保持了模块化和低运行时开销。 

---
# Co-designing Zoomorphic Robot Concepts for Animal Welfare Education 

**Title (ZH)**: 共设计Zoomorphic机器人概念以促进动物福利教育 

**Authors**: Isobel Voysey, Lynne Baillie, Joanne Williams, Michael Herrmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.02898)  

**Abstract**: Animal welfare education could greatly benefit from customized robots to help children learn about animals and their behavior, and thereby promote positive, safe child-animal interactions. To this end, we ran Participatory Design workshops with animal welfare educators and children to identify key requirements for zoomorphic robots from their perspectives. Our findings encompass a zoomorphic robot's appearance, behavior, and features, as well as concepts for a narrative surrounding the robot. Through comparing and contrasting the two groups, we find the importance of: negative reactions to undesirable behavior from children; using the facial features and tail to provide cues signaling an animal's internal state; and a natural, furry appearance and texture. We also contribute some novel activities for Participatory Design with children, including branching storyboards inspired by thematic apperception tests and interactive narratives, and reflect on some of the key design challenges of achieving consensus between the groups, despite much overlap in their design concepts. 

**Abstract (ZH)**: 定制化机器人在提升动物福利教育中的应用：以促进儿童与动物的积极安全互动为例——基于与动物福利教育者和儿童的合作设计工作坊的研究 

---
# Tunable Leg Stiffness in a Monopedal Hopper for Energy-Efficient Vertical Hopping Across Varying Ground Profiles 

**Title (ZH)**: 单足跳跃器在不同地面条件下进行能量高效垂直跳跃的可调腿部刚度研究 

**Authors**: Rongqian Chen, Jun Kwon, Kefan Wu, Wei-Hsi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02873)  

**Abstract**: We present the design and implementation of HASTA (Hopper with Adjustable Stiffness for Terrain Adaptation), a vertical hopping robot with real-time tunable leg stiffness, aimed at optimizing energy efficiency across various ground profiles (a pair of ground stiffness and damping conditions). By adjusting leg stiffness, we aim to maximize apex hopping height, a key metric for energy-efficient vertical hopping. We hypothesize that softer legs perform better on soft, damped ground by minimizing penetration and energy loss, while stiffer legs excel on hard, less damped ground by reducing limb deformation and energy dissipation. Through experimental tests and simulations, we find the best leg stiffness within our selection for each combination of ground stiffness and damping, enabling the robot to achieve maximum steady-state hopping height with a constant energy input. These results support our hypothesis that tunable stiffness improves energy-efficient locomotion in controlled experimental conditions. In addition, the simulation provides insights that could aid in the future development of controllers for selecting leg stiffness. 

**Abstract (ZH)**: HASTA：一种具有可调韧性的垂直跳跃机器人及其设计与实现 

---
# Learning User Interaction Forces using Vision for a Soft Finger Exosuit 

**Title (ZH)**: 基于视觉学习软手指外骨骼中的用户交互力 

**Authors**: Mohamed Irfan Refai, Abdulaziz Y. Alkayas, Anup Teejo Mathew, Federico Renda, Thomas George Thuruthel  

**Link**: [PDF](https://arxiv.org/pdf/2508.02870)  

**Abstract**: Wearable assistive devices are increasingly becoming softer. Modelling their interface with human tissue is necessary to capture transmission of dynamic assistance. However, their nonlinear and compliant nature makes both physical modeling and embedded sensing challenging. In this paper, we develop a image-based, learning-based framework to estimate distributed contact forces for a finger-exosuit system. We used the SoRoSim toolbox to generate a diverse dataset of exosuit geometries and actuation scenarios for training. The method accurately estimated interaction forces across multiple contact locations from low-resolution grayscale images, was able to generalize to unseen shapes and actuation levels, and remained robust under visual noise and contrast variations. We integrated the model into a feedback controller, and found that the vision-based estimator functions as a surrogate force sensor for closed-loop control. This approach could be used as a non-intrusive alternative for real-time force estimation for exosuits. 

**Abstract (ZH)**: 可穿戴辅助设备越来越趋向柔软。建模其与人体组织的界面对于捕获动态辅助的传递是必要的。然而，它们的非线性和顺应性特性使得物理建模和嵌入式传感具有挑战性。在本文中，我们开发了一种基于图像、基于学习的框架来估计手指外骨骼系统的分布式接触力。我们使用SoRoSim工具箱生成了多样化的外骨骼几何形状和驱动场景数据集用于训练。该方法能够从低分辨率灰度图像中准确估计多个接触位置的交互力，并能泛化到未见过的形状和驱动水平，在视觉噪声和对比度变化下仍能保持鲁棒性。我们将该模型集成到反馈控制器中，并发现基于视觉的估算器可以作为闭环控制中的替代力传感器。该方法可以作为对外骨骼进行实时力估算的非侵入性替代方案。 

---
# LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences 

**Title (ZH)**: LiDARCrafter: 基于LiDAR序列的动态4D世界建模 

**Authors**: Ao Liang, Youquan Liu, Yu Yang, Dongyue Lu, Linfeng Li, Lingdong Kong, Huaici Zhao, Wei Tsang Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2508.03692)  

**Abstract**: Generative world models have become essential data engines for autonomous driving, yet most existing efforts focus on videos or occupancy grids, overlooking the unique LiDAR properties. Extending LiDAR generation to dynamic 4D world modeling presents challenges in controllability, temporal coherence, and evaluation standardization. To this end, we present LiDARCrafter, a unified framework for 4D LiDAR generation and editing. Given free-form natural language inputs, we parse instructions into ego-centric scene graphs, which condition a tri-branch diffusion network to generate object structures, motion trajectories, and geometry. These structured conditions enable diverse and fine-grained scene editing. Additionally, an autoregressive module generates temporally coherent 4D LiDAR sequences with smooth transitions. To support standardized evaluation, we establish a comprehensive benchmark with diverse metrics spanning scene-, object-, and sequence-level aspects. Experiments on the nuScenes dataset using this benchmark demonstrate that LiDARCrafter achieves state-of-the-art performance in fidelity, controllability, and temporal consistency across all levels, paving the way for data augmentation and simulation. The code and benchmark are released to the community. 

**Abstract (ZH)**: 基于LiDAR的4D生成式世界模型匠心构建 

---
# La La LiDAR: Large-Scale Layout Generation from LiDAR Data 

**Title (ZH)**: La La LiDAR: 基于LiDAR数据的大规模布局生成 

**Authors**: Youquan Liu, Lingdong Kong, Weidong Yang, Xin Li, Ao Liang, Runnan Chen, Ben Fei, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03691)  

**Abstract**: Controllable generation of realistic LiDAR scenes is crucial for applications such as autonomous driving and robotics. While recent diffusion-based models achieve high-fidelity LiDAR generation, they lack explicit control over foreground objects and spatial relationships, limiting their usefulness for scenario simulation and safety validation. To address these limitations, we propose Large-scale Layout-guided LiDAR generation model ("La La LiDAR"), a novel layout-guided generative framework that introduces semantic-enhanced scene graph diffusion with relation-aware contextual conditioning for structured LiDAR layout generation, followed by foreground-aware control injection for complete scene generation. This enables customizable control over object placement while ensuring spatial and semantic consistency. To support our structured LiDAR generation, we introduce Waymo-SG and nuScenes-SG, two large-scale LiDAR scene graph datasets, along with new evaluation metrics for layout synthesis. Extensive experiments demonstrate that La La LiDAR achieves state-of-the-art performance in both LiDAR generation and downstream perception tasks, establishing a new benchmark for controllable 3D scene generation. 

**Abstract (ZH)**: 大规模布局引导的LiDAR生成模型（La La LiDAR） 

---
# Veila: Panoramic LiDAR Generation from a Monocular RGB Image 

**Title (ZH)**: Veila：基于单目RGB图像的全景LiDAR生成 

**Authors**: Youquan Liu, Lingdong Kong, Weidong Yang, Ao Liang, Jianxiong Gao, Yang Wu, Xiang Xu, Xin Li, Linfeng Li, Runnan Chen, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2508.03690)  

**Abstract**: Realistic and controllable panoramic LiDAR data generation is critical for scalable 3D perception in autonomous driving and robotics. Existing methods either perform unconditional generation with poor controllability or adopt text-guided synthesis, which lacks fine-grained spatial control. Leveraging a monocular RGB image as a spatial control signal offers a scalable and low-cost alternative, which remains an open problem. However, it faces three core challenges: (i) semantic and depth cues from RGB are vary spatially, complicating reliable conditioning generation; (ii) modality gaps between RGB appearance and LiDAR geometry amplify alignment errors under noisy diffusion; and (iii) maintaining structural coherence between monocular RGB and panoramic LiDAR is challenging, particularly in non-overlap regions between images and LiDAR. To address these challenges, we propose Veila, a novel conditional diffusion framework that integrates: a Confidence-Aware Conditioning Mechanism (CACM) that strengthens RGB conditioning by adaptively balancing semantic and depth cues according to their local reliability; a Geometric Cross-Modal Alignment (GCMA) for robust RGB-LiDAR alignment under noisy diffusion; and a Panoramic Feature Coherence (PFC) for enforcing global structural consistency across monocular RGB and panoramic LiDAR. Additionally, we introduce two metrics, Cross-Modal Semantic Consistency and Cross-Modal Depth Consistency, to evaluate alignment quality across modalities. Experiments on nuScenes, SemanticKITTI, and our proposed KITTI-Weather benchmark demonstrate that Veila achieves state-of-the-art generation fidelity and cross-modal consistency, while enabling generative data augmentation that improves downstream LiDAR semantic segmentation. 

**Abstract (ZH)**: 现实可控的大范围LiDAR数据生成对于自主驾驶和机器人领域的可扩展三维感知至关重要。现有的方法要么无法控制生成过程，要么采用文本引导合成，缺乏精细的空间控制。利用单目RGB图像作为空间控制信号提供了一种可扩展且成本低的替代方案，但这一问题仍是一个开放性问题。然而，它面临三个核心挑战：（i）RGB图像中的语义和深度线索在空间上变化，使得可靠的条件生成变得复杂；（ii）RGB外观和LiDAR几何之间的模态差异在噪声扩散下加剧了对齐误差；（iii）在图像和LiDAR之间存在不重叠区域时，保持单目RGB和大范围LiDAR之间的结构连贯性具有挑战性。为了解决这些挑战，我们提出了Veila，一种新颖的条件扩散框架，该框架整合了：一种语义可信度感知条件机制（CACM），通过适应性平衡语义和深度线索来加强RGB条件化，根据它们的局部可靠性；一种几何跨模态对齐（GCMA），以在噪声扩散下实现鲁棒的RGB-LiDAR对齐；以及一种全景特征连贯性（PFC），以确保单目RGB和大范围LiDAR之间的一致结构连贯性。此外，我们引入了两种新的评估指标：跨模态语义一致性和平面交叉模态深度一致性，以评估不同模态之间的对齐质量。在nuScenes、SemanticKITTI和我们提出的KITTI-Weather基准上的实验表明，Veila不仅实现了最先进的生成保真度和跨模态一致性，还支持生成数据增强，从而提高下游LiDAR语义分割的性能。 

---
# OmniShape: Zero-Shot Multi-Hypothesis Shape and Pose Estimation in the Real World 

**Title (ZH)**: 全知形状：零-shot 多假设形状和姿态估计于现实世界 

**Authors**: Katherine Liu, Sergey Zakharov, Dian Chen, Takuya Ikeda, Greg Shakhnarovich, Adrien Gaidon, Rares Ambrus  

**Link**: [PDF](https://arxiv.org/pdf/2508.03669)  

**Abstract**: We would like to estimate the pose and full shape of an object from a single observation, without assuming known 3D model or category. In this work, we propose OmniShape, the first method of its kind to enable probabilistic pose and shape estimation. OmniShape is based on the key insight that shape completion can be decoupled into two multi-modal distributions: one capturing how measurements project into a normalized object reference frame defined by the dataset and the other modelling a prior over object geometries represented as triplanar neural fields. By training separate conditional diffusion models for these two distributions, we enable sampling multiple hypotheses from the joint pose and shape distribution. OmniShape demonstrates compelling performance on challenging real world datasets. Project website: this https URL 

**Abstract (ZH)**: 我们希望从单次观测中估计物体的姿态和完整形状，无需假设已知3D模型或类别。在本文中，我们提出了OmniShape，这是首个能够进行概率姿态和形状估计的方法。OmniShape 基于一个关键洞察，即形状完成可以分解为两个多模态分布：一个捕获测量如何投影到由数据集定义的归一化对象参考框架中，另一个则建模对象几何形状（表示为三方面神经场）的先验。通过为这两个分布分别训练条件扩散模型，我们可以在联合姿态和形状分布中采样多个假设。OmniShape 在具有挑战性的现实世界数据集上展示了令人信服的性能。项目网站：这个 https://链接。 

---
# LRDDv2: Enhanced Long-Range Drone Detection Dataset with Range Information and Comprehensive Real-World Challenges 

**Title (ZH)**: LRDDv2：带有距离信息和全面现实世界挑战的增强长距无人机检测数据集 

**Authors**: Amirreza Rouhi, Sneh Patel, Noah McCarthy, Siddiqa Khan, Hadi Khorsand, Kaleb Lefkowitz, David K.Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.03331)  

**Abstract**: The exponential growth in Unmanned Aerial Vehicles (UAVs) usage underscores the critical need of detecting them at extended distances to ensure safe operations, especially in densely populated areas. Despite the tremendous advances made in computer vision through deep learning, the detection of these small airborne objects remains a formidable challenge. While several datasets have been developed specifically for drone detection, the need for a more extensive and diverse collection of drone image data persists, particularly for long-range detection under varying environmental conditions. We introduce here the Long Range Drone Detection (LRDD) Version 2 dataset, comprising 39,516 meticulously annotated images, as a second release of the LRDD dataset released previously. The LRDDv2 dataset enhances the LRDDv1 by incorporating a greater variety of images, providing a more diverse and comprehensive resource for drone detection research. What sets LRDDv2 apart is its inclusion of target range information for over 8,000 images, making it possible to develop algorithms for drone range estimation. Tailored for long-range aerial object detection, the majority of LRDDv2's dataset consists of images capturing drones with 50 or fewer pixels in 1080p resolution. For access to the complete Long-Range Drone Detection Dataset (LRDD)v2, please visit this https URL . 

**Abstract (ZH)**: 长程无人机检测数据集（LRDD）版本2 

---
# Enhancing Joint Human-AI Inference in Robot Missions: A Confidence-Based Approach 

**Title (ZH)**: 基于置信度的方法提升机器人任务中人机联合推理能力 

**Authors**: Duc-An Nguyen, Clara Colombatto, Steve Fleming, Ingmar Posner, Nick Hawes, Raunak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2508.03293)  

**Abstract**: Joint human-AI inference holds immense potential to improve outcomes in human-supervised robot missions. Current day missions are generally in the AI-assisted setting, where the human operator makes the final inference based on the AI recommendation. However, due to failures in human judgement on when to accept or reject the AI recommendation, complementarity is rarely achieved. We investigate joint human-AI inference where the inference made with higher confidence is selected. Through a user study with N=100 participants on a representative simulated robot teleoperation task, specifically studying the inference of robots' control delays we show that: a) Joint inference accuracy is higher and its extent is regulated by the confidence calibration of the AI agent, and b) Humans change their inferences based on AI recommendations and the extent and direction of this change is also regulated by the confidence calibration of the AI agent. Interestingly, our results show that pairing poorly-calibrated AI-DSS with humans hurts performance instead of helping the team, reiterating the need for AI-based decision support systems with good metacognitive sensitivity. To the best of our knowledge, our study presents the first application of a maximum-confidence-based heuristic for joint human-AI inference within a simulated robot teleoperation task. 

**Abstract (ZH)**: 联合人机推理在监督机器人任务中的潜在价值巨大：通过一种代表性的模拟机器人远程操作任务，基于用户研究N=100的结果表明，联合推理的准确性更高，其程度由AI代理的信心校准调节；人类会根据AI建议改变其推理，这种变化的范围和方向也由AI代理的信心校准调节。有趣的是，我们的结果表明，将信心校准不佳的AI-DSS与人类配合反而会损害团队性能，强调了需要具备良好元认知敏感性的基于AI的决策支持系统的需求。据我们所知，本研究首次在模拟机器人远程操作任务中应用了基于最大信心的联合人机推理启发式方法。 

---
# COFFEE: A Shadow-Resilient Real-Time Pose Estimator for Unknown Tumbling Asteroids using Sparse Neural Networks 

**Title (ZH)**: COFFEE: 一种适用于未知翻滚小行星的抗阴影实时姿态估算器基于稀疏神经网络 

**Authors**: Arion Zimmermann, Soon-Jo Chung, Fred Hadaegh  

**Link**: [PDF](https://arxiv.org/pdf/2508.03132)  

**Abstract**: The accurate state estimation of unknown bodies in space is a critical challenge with applications ranging from the tracking of space debris to the shape estimation of small bodies. A necessary enabler to this capability is to find and track features on a continuous stream of images. Existing methods, such as SIFT, ORB and AKAZE, achieve real-time but inaccurate pose estimates, whereas modern deep learning methods yield higher quality features at the cost of more demanding computational resources which might not be available on space-qualified hardware. Additionally, both classical and data-driven methods are not robust to the highly opaque self-cast shadows on the object of interest. We show that, as the target body rotates, these shadows may lead to large biases in the resulting pose estimates. For these objects, a bias in the real-time pose estimation algorithm may mislead the spacecraft's state estimator and cause a mission failure, especially if the body undergoes a chaotic tumbling motion. We present COFFEE, the Celestial Occlusion Fast FEature Extractor, a real-time pose estimation framework for asteroids designed to leverage prior information on the sun phase angle given by sun-tracking sensors commonly available onboard spacecraft. By associating salient contours to their projected shadows, a sparse set of features are detected, invariant to the motion of the shadows. A Sparse Neural Network followed by an attention-based Graph Neural Network feature matching model are then jointly trained to provide a set of correspondences between successive frames. The resulting pose estimation pipeline is found to be bias-free, more accurate than classical pose estimation pipelines and an order of magnitude faster than other state-of-the-art deep learning pipelines on synthetic data as well as on renderings of the tumbling asteroid Apophis. 

**Abstract (ZH)**: 实时估计小计算未知未知小体的姿态在航天领域中是一个关键挑战，应用场景从空间碎片跟踪到到到到小型物体的姿态估测.现有的特征提取方法如如如like S这样的S的是基于S如S的技术如，如is这样的技术在实时候并不能保证准确的姿态估测上。相比之下on呢，现代深度学习方法在实验on找到可以获得更高的精度on然而n在ond需要更密集的计算资源on这样的计算资源在中on-硬件上可能无法获得isis现代深度从中学习方法虽然在高遮蔽区域表现出色柔度但在目标引起的阴影也会引起波in时间估算的偏移在当前实时候姿态估测算法中这样的的偏移可能导致is航天器姿态估计错误发生错误误解on特别是当航天器进行混沌翻剧烈旋转转动时.

为此我们设计is针对小以is落日长ículiónL作为参照的的实时姿态估计算法is算法设计基于星载轨道特征提取的isCoffeeis星际特征提取算法，算法为小行星设计定制设计以适应在阳光角度受变化下on的星is追踪传感器is…常见于航天器上的.

通过将显著轮廓与投影阴影关联到起来一个稀疏特征集合is检测到的对变换不变is然后通过稀疏神经网络is随后通过注意机制驱动的图神经网络is共同联合作用找到一幅将先后特征间的一组对应关系is形成的结果估计管线表明is比比比以前的传统姿态估计算法更加准确同时比也快了数量级个量is比当今最先进的深度学习管道数量级倍在合成数据上is是以及对形虚拟的的翻剧烈转动的小行星阿波斯is上上上isisisisis. 

---
# Can Large Language Models Identify Materials from Radar Signals? 

**Title (ZH)**: 大型语言模型能否识别来自雷达信号的材料？ 

**Authors**: Jiangyou Zhu, Hongyu Deng, He Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.03120)  

**Abstract**: Accurately identifying the material composition of objects is a critical capability for AI robots powered by large language models (LLMs) to perform context-aware manipulation. Radar technologies offer a promising sensing modality for material recognition task. When combined with deep learning, radar technologies have demonstrated strong potential in identifying the material of various objects. However, existing radar-based solutions are often constrained to closed-set object categories and typically require task-specific data collection to train deep learning models, largely limiting their practical applicability. This raises an important question: Can we leverage the powerful reasoning capabilities of pre-trained LLMs to directly infer material composition from raw radar signals? Answering this question is non-trivial due to the inherent redundancy of radar signals and the fact that pre-trained LLMs have no prior exposure to raw radar data during training. To address this, we introduce LLMaterial, the first study to investigate the feasibility of using LLM to identify materials directly from radar signals. First, we introduce a physics-informed signal processing pipeline that distills high-redundancy radar raw data into a set of compact intermediate parameters that encapsulate the material's intrinsic characteristics. Second, we adopt a retrieval-augmented generation (RAG) strategy to provide the LLM with domain-specific knowledge, enabling it to interpret and reason over the extracted intermediate parameters. Leveraging this integration, the LLM is empowered to perform step-by-step reasoning on the condensed radar features, achieving open-set material recognition directly from raw radar signals. Preliminary results show that LLMaterial can effectively distinguish among a variety of common materials, highlighting its strong potential for real-world material identification applications. 

**Abstract (ZH)**: 基于大规模语言模型的AI机器人通过雷达技术准确识别物体材料是一项关键能力。现有的雷达基解决方案通常受限于封闭类别物体，并且通常需要专门的任务数据集来训练深度学习模型，极大地限制了其实用性。本文提出了LLMaterial，这是首个研究利用预训练的大规模语言模型直接从雷达信号中识别材料可行性的研究。我们引入了一种基于物理的信号处理管道，将高冗余的雷达原始数据精简为一组紧凑的中间参数，这些参数捕获了材料的固有特性。同时，我们采用了检索增强生成（RAG）策略，为预训练的大规模语言模型提供领域特定知识，使其能够解释和推理提取出的中间参数。通过这种集成，预训练的大规模语言模型能够对压缩的雷达特征进行逐步推理，直接从原始雷达信号中进行开放式材料识别。初步结果表明，LLMaterial能够有效地区分多种常见材料，突显了其在实际材料识别应用中的强大潜力。 

---
# Beyond Policy Optimization: A Data Curation Flywheel for Sparse-Reward Long-Horizon Planning 

**Title (ZH)**: 超越策略优化：稀疏奖励长时规划的数据整理飞轮 

**Authors**: Yutong Wang, Pengliang Ji, Kaixin Li, Baolong Bi, Tao Feng, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2508.03018)  

**Abstract**: Large Language Reasoning Models have demonstrated remarkable success on static tasks, yet their application to multi-round agentic planning in interactive environments faces two fundamental challenges. First, the intractable credit assignment problem renders conventional reinforcement learning ineffective in sparse-reward settings. Second, the computational overhead of verbose, step-by-step reasoning histories is prohibitive. To address these challenges, we propose BPO, a three-stage framework (bootstrapping, extrapolation, and refinement) that establishes a self-improving data flywheel to develop robust reasoning models for long-horizon, sparse-reward environments. Our framework first bootstraps efficient reasoning using the proposed planning quaternions with long-short chain-of-thought fusion. It then extrapolates to out-of-distribution tasks through complexity-stratified curriculum learning. Finally, the model iteratively refines itself by learning exclusively on experiences selected via reward-gated rejection sampling. Experiments on ALFWorld, ScienceWorld, and WebShop demonstrate that our approach achieves state-of-the-art with significant token efficiency, providing a new recipe for reasoning models in agentic planning. 

**Abstract (ZH)**: 大型语言推理模型在静态任务上取得了显著成功，但在交互环境中应用于多轮代理规划面临两个根本挑战。为应对这些挑战，我们提出了BPO框架（自增强、外推和精炼三个阶段），该框架建立了一种自我改进的数据飞轮，旨在开发适用于长期决策、稀疏奖励环境的 robust 推理模型。该框架首先利用提出的融合长短期思考的规划四元数进行高效推理，然后通过分层课程学习来外推到分布外任务，最后通过奖励门控拒绝采样选择的经验进行迭代精炼。我们在ALFWorld、ScienceWorld和WebShop上的实验表明，该方法在显著提高 token 效率的同时达到了最先进的效果，为代理规划中的推理模型提供了一种新的配方。 

---
# Generating Light-based Fingerprints for Indoor Localization 

**Title (ZH)**: 基于光的指纹室内定位生成 

**Authors**: Hsun-Yu Lee, Jie Lin, Fang-Jing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03011)  

**Abstract**: Accurate indoor localization underpins applications ranging from wayfinding and emergency response to asset tracking and smart-building services. Radio-frequency solutions (e.g. Wi-Fi, RFID, UWB) are widely adopted but remain vulnerable to multipath fading, interference, and uncontrollable coverage variation. We explore an orthogonal modality -- visible light communication (VLC) -- and demonstrate that the spectral signatures captured by a low-cost AS7341 sensor can serve as robust location fingerprints.
We introduce a two-stage framework that (i) trains a multi-layer perceptron (MLP) on real spectral measurements and (ii) enlarges the training corpus with synthetic samples produced by TabGAN. The augmented dataset reduces the mean localization error from 62.9cm to 49.3cm -- a 20% improvement -- while requiring only 5% additional data-collection effort. Experimental results obtained on 42 reference points in a U-shaped laboratory confirm that GAN-based augmentation mitigates data-scarcity issues and enhances generalization. 

**Abstract (ZH)**: 精确的室内定位支撑着从路径引导和应急响应到资产追踪和智能建筑服务的广泛应用。基于射频的技术（如Wi-Fi、RFID、UWB）被广泛采用但仍易受多径衰落、干扰和不可控覆盖变化的影响。我们研究了一种正交的模态——可见光通信（VLC），并证明了低成本AS7341传感器捕获的光谱特征可以作为稳健的位置指纹。我们引入了一种两阶段框架，首先利用多层感知器（MLP）训练实际光谱测量数据，然后通过TabGAN生成合成样本扩展训练数据集。扩充后的数据集将平均定位误差从62.9cm降低到49.3cm，提高了20%，同时仅需额外5%的数据采集努力。在U形实验室中的42个参考点上获得的实验结果证实，基于GAN的数据增强缓解了数据稀缺问题并提升了泛化能力。 

---
# Following Route Instructions using Large Vision-Language Models: A Comparison between Low-level and Panoramic Action Spaces 

**Title (ZH)**: 使用大型视觉-语言模型遵循导航指令：低级动作空间与全景动作空间的比较 

**Authors**: Vebjørn Haug Kåsene, Pierre Lison  

**Link**: [PDF](https://arxiv.org/pdf/2508.02917)  

**Abstract**: Vision-and-Language Navigation (VLN) refers to the task of enabling autonomous robots to navigate unfamiliar environments by following natural language instructions. While recent Large Vision-Language Models (LVLMs) have shown promise in this task, most current VLM systems rely on models specifically designed and optimized for navigation, leaving the potential of off-the-shelf LVLMs underexplored. Furthermore, while older VLN approaches used low-level action spaces with egocentric views and atomic actions (such as "turn left" or "move forward"), newer models tend to favor panoramic action spaces with discrete navigable viewpoints. This paper investigates (1) whether off-the-shelf LVLMs (fine-tuned without architectural modifications or simulator-based training) can effectively support VLN tasks and (2) whether such models can support both low-level and panoramic action paradigms. To this end, we fine-tune the open-source model Qwen2.5-VL-3B-Instruct on the Room-to-Room (R2R) dataset and evaluate its empirical performance across both low-level and panoramic action spaces. The best resulting model achieves a 41% success rate on the R2R test set, demonstrating that while off-the-shelf LVLMs can learn to perform Vision-and-Language Navigation, they still lag behind models specifically designed for this task. 

**Abstract (ZH)**: 基于视觉-语言的导航（VLN）是指通过遵循自然语言指令使自主机器人导航陌生环境的任务。虽然近期的大规模视觉-语言模型（LVLMs）在这一任务中展现了潜力，但目前大多数VLN系统依赖于专门设计和优化的导航模型，从而使得现成的LVLMs的潜力被大幅低估。此外，虽然早期的VLN方法使用低级别动作空间和第一人称视角及原子动作（如“向左转”或“前进”），而最新模型更倾向于使用全景动作空间和离散可导航视角。本文探讨了（1）现成的LVLMs（无需架构修改或基于模拟器的训练进行微调）是否能够有效支持VLN任务，以及（2）这些模型是否能够支持低级别和全景动作范式。为此，我们对开源模型Qwen2.5-VL-3B-Instruct进行了微调，并在低级别和全景动作空间中评估了其实际性能。最佳模型在R2R测试集上的成功率达到了41%，表明虽然现成的LVLMs可以学习执行视觉-语言导航任务，但它们仍然落后于专门为这一任务设计的模型。 

---
# Frequency Point Game Environment for UAVs via Expert Knowledge and Large Language Model 

**Title (ZH)**: 基于专家知识和大语言模型的无人机频率点博弈环境 

**Authors**: Jingpu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02757)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) have made significant advancements in communication stability and security through techniques such as frequency hopping, signal spreading, and adaptive interference suppression. However, challenges remain in modeling spectrum competition, integrating expert knowledge, and predicting opponent behavior. To address these issues, we propose UAV-FPG (Unmanned Aerial Vehicle - Frequency Point Game), a game-theoretic environment model that simulates the dynamic interaction between interference and anti-interference strategies of opponent and ally UAVs in communication frequency bands. The model incorporates a prior expert knowledge base to optimize frequency selection and employs large language models for path planning, simulating a "strong adversary". Experimental results highlight the effectiveness of integrating the expert knowledge base and the large language model, with the latter significantly improving path planning in dynamic scenarios through iterative interactions, outperforming fixed-path strategies. UAV-FPG provides a robust platform for advancing anti-jamming strategies and intelligent decision-making in UAV communication systems. 

**Abstract (ZH)**: 无人航空器（UAVs）通过频跳、信号扩spread、自适应干扰抑制等技术在通信稳定性和安全性方面取得了显著进展。然而，在频谱竞争建模、专家知识集成和对手行为预测方面仍存在挑战。为应对这些挑战，我们提出了UAV-FPG（无人航空器-频段博弈），这是一种博弈论环境模型，用于模拟通信频段内敌我无人航空器之间干扰与抗干扰策略的动态交互。该模型整合了先验专家知识库以优化频段选择，并利用大型语言模型进行路径规划，模拟“强对手”。实验结果表明，整合专家知识库和大型语言模型的有效性，后者通过迭代交互显著改善了动态场景下的路径规划，优于固定路径策略。UAV-FPG为无人航空器通信系统中的抗干扰策略和智能决策提供了稳健的平台。 

---
