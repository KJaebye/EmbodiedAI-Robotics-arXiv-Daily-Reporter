# Experimental Comparison of Whole-Body Control Formulations for Humanoid Robots in Task Acceleration and Task Force Spaces 

**Title (ZH)**: 全人体控制公式在任务加速空间和任务力空间中的实验比较 

**Authors**: Sait Sovukluk, Grazia Zambella, Tobias Egle, Christian Ott  

**Link**: [PDF](https://arxiv.org/pdf/2507.18502)  

**Abstract**: This paper studies the experimental comparison of two different whole-body control formulations for humanoid robots: inverse dynamics whole-body control (ID-WBC) and passivity-based whole-body control (PB-WBC). The two controllers fundamentally differ from each other as the first is formulated in task acceleration space and the latter is in task force space with passivity considerations. Even though both control methods predict stability under ideal conditions in closed-loop dynamics, their robustness against joint friction, sensor noise, unmodeled external disturbances, and non-perfect contact conditions is not evident. Therefore, we analyze and experimentally compare the two controllers on a humanoid robot platform through swing foot position and orientation control, squatting with and without unmodeled additional weights, and jumping. We also relate the observed performance and characteristic differences with the controller formulations and highlight each controller's advantages and disadvantages. 

**Abstract (ZH)**: 本文研究了两种不同类型的整体身体控制算法在类人机器人上的实验比较：逆动力学整体身体控制（ID-WBC）和基于耗散性的整体身体控制（PB-WBC）。两种控制器的基础差异在于前者在任务加速度空间中进行建模，而后者在任务力空间中进行建模并考虑了耗散性。尽管这两种控制方法在闭环动力学条件下都能预测稳定性，但它们在关节摩擦、传感器噪声、未建模的外部干扰以及非理想的接触条件下的鲁棒性并不明显。因此，我们通过摆动脚的位置和方向控制、带和不带未建模附加重量的下蹲，以及跳跃实验，在类人机器人平台上分析并比较了这两种控制器。同时，我们将观察到的性能和特征差异与控制器形式进行关联，并强调每种控制器的优点和缺点。 

---
# Residual Koopman Model Predictive Control for Enhanced Vehicle Dynamics with Small On-Track Data Input 

**Title (ZH)**: 残差Koopman模型预测控制以增强车辆动力学性能的输入少量赛道数据方法 

**Authors**: Yonghao Fu, Cheng Hu, Haokun Xiong, Zhangpeng Bao, Wenyuan Du, Edoardo Ghignone, Michele Magno, Lei Xie, Hongye Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.18396)  

**Abstract**: In vehicle trajectory tracking tasks, the simplest approach is the Pure Pursuit (PP) Control. However, this single-point preview tracking strategy fails to consider vehicle model constraints, compromising driving safety. Model Predictive Control (MPC) as a widely adopted control method, optimizes control actions by incorporating mechanistic models and physical constraints. While its control performance critically depends on the accuracy of vehicle modeling. Traditional vehicle modeling approaches face inherent trade-offs between capturing nonlinear dynamics and maintaining computational efficiency, often resulting in reduced control performance. To address these challenges, this paper proposes Residual Koopman Model Predictive Control (RKMPC) framework. This method uses two linear MPC architecture to calculate control inputs: a Linear Model Predictive Control (LMPC) computes the baseline control input based on the vehicle kinematic model, and a neural network-based RKMPC calculates the compensation input. The final control command is obtained by adding these two components. This design preserves the reliability and interpretability of traditional mechanistic model while achieving performance optimization through residual modeling. This method has been validated on the Carsim-Matlab joint simulation platform and a physical 1:10 scale F1TENTH racing car. Experimental results show that RKMPC requires only 20% of the training data needed by traditional Koopman Model Predictive Control (KMPC) while delivering superior tracking performance. Compared to traditional LMPC, RKMPC reduces lateral error by 11.7%-22.1%, decreases heading error by 8.9%-15.8%, and improves front-wheel steering stability by up to 27.6%. The implementation code is available at: this https URL Koopman. 

**Abstract (ZH)**: 基于残差科寇曼模型预测控制的车辆轨迹跟踪方法 

---
# AF-RLIO: Adaptive Fusion of Radar-LiDAR-Inertial Information for Robust Odometry in Challenging Environments 

**Title (ZH)**: 雷达-激光雷达-惯性信息自适应融合以在具有挑战性的环境中实现稳健定位 

**Authors**: Chenglong Qian, Yang Xu, Xiufang Shi, Jiming Chen, Liang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18317)  

**Abstract**: In robotic navigation, maintaining precise pose estimation and navigation in complex and dynamic environments is crucial. However, environmental challenges such as smoke, tunnels, and adverse weather can significantly degrade the performance of single-sensor systems like LiDAR or GPS, compromising the overall stability and safety of autonomous robots. To address these challenges, we propose AF-RLIO: an adaptive fusion approach that integrates 4D millimeter-wave radar, LiDAR, inertial measurement unit (IMU), and GPS to leverage the complementary strengths of these sensors for robust odometry estimation in complex environments. Our method consists of three key modules. Firstly, the pre-processing module utilizes radar data to assist LiDAR in removing dynamic points and determining when environmental conditions are degraded for LiDAR. Secondly, the dynamic-aware multimodal odometry selects appropriate point cloud data for scan-to-map matching and tightly couples it with the IMU using the Iterative Error State Kalman Filter. Lastly, the factor graph optimization module balances weights between odometry and GPS data, constructing a pose graph for optimization. The proposed approach has been evaluated on datasets and tested in real-world robotic environments, demonstrating its effectiveness and advantages over existing methods in challenging conditions such as smoke and tunnels. 

**Abstract (ZH)**: 机器人导航中，在复杂动态环境中保持精确的姿态估计和导航至关重要。然而，烟雾、隧道和恶劣天气等环境挑战会显著降低如激光雷达或GPS等单传感器系统的表现，从而影响自主机器人的整体稳定性和安全性。为应对这些挑战，我们提出AF-RLIO：一种自适应融合方法，将4D毫米波雷达、激光雷达、惯性测量单位（IMU）和GPS集成起来，利用这些传感器的互补优势，在复杂环境中实现稳健的里程计估计。该方法包括三个关键模块：首先，预处理模块利用雷达数据辅助激光雷达去除动态点，并判断环境条件是否恶化；其次，动态感知多模态里程计选择适当的点云数据进行扫描到地图匹配，并通过迭代误差状态卡尔曼滤波器与IMU紧密耦合；最后，因子图优化模块在里程计和GPS数据之间平衡权重，构建姿态图进行优化。所提出的方法已在数据集上进行了评估并在真实的机器人环境中进行了测试，展示了其在烟雾和隧道等挑战条件下相较于现有方法的有效性和优势。 

---
# Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding 

**Title (ZH)**: 基于基础模型推理与部分锚定的适应性刚体物体即时操作 

**Authors**: Xiaojie Zhang, Yuanfei Wang, Ruihai Wu, Kunqi Xu, Yu Li, Liuyu Xiang, Hao Dong, Zhaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2507.18276)  

**Abstract**: Articulated objects pose diverse manipulation challenges for robots. Since their internal structures are not directly observable, robots must adaptively explore and refine actions to generate successful manipulation trajectories. While existing works have attempted cross-category generalization in adaptive articulated object manipulation, two major challenges persist: (1) the geometric diversity of real-world articulated objects complicates visual perception and understanding, and (2) variations in object functions and mechanisms hinder the development of a unified adaptive manipulation strategy. To address these challenges, we propose AdaRPG, a novel framework that leverages foundation models to extract object parts, which exhibit greater local geometric similarity than entire objects, thereby enhancing visual affordance generalization for functional primitive skills. To support this, we construct a part-level affordance annotation dataset to train the affordance model. Additionally, AdaRPG utilizes the common knowledge embedded in foundation models to reason about complex mechanisms and generate high-level control codes that invoke primitive skill functions based on part affordance inference. Simulation and real-world experiments demonstrate AdaRPG's strong generalization ability across novel articulated object categories. 

**Abstract (ZH)**: articulated物体的操控对机器人提出了多样化的挑战。由于它们的内部结构不可直接观察，机器人必须适应性地探索和优化动作以生成成功的操控轨迹。尽管现有工作在适应性操控articulated物体方面尝试了跨类别泛化，但仍存在两大挑战：(1) 现实世界中articulated物体的几何多样性使视觉感知和理解复杂化；(2) 对象功能和机制的变化阻碍了统一适应性操控策略的发展。为应对这些挑战，我们提出了AdaRPG，一种利用基础模型来提取表现出更高局部几何相似性的对象部件的新框架，从而增强功能基础技能的视觉可利用性泛化。为此，我们构建了一个部件级可利用性标注数据集来训练可利用性模型。此外，AdaRPG利用基础模型中嵌入的通用知识来推理复杂机制并生成基于部件可利用性推断的高阶控制代码，以调用基础技能功能。模拟和真实世界实验展示了AdaRPG在新颖articulated物体类别中的强大泛化能力。 

---
# ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation 

**Title (ZH)**: ReSem3D：通过细粒度语义 grounding 完善的 3D 空间约束以实现可泛化的机器人 manipulation 

**Authors**: Chenyu Su, Weiwei Shang, Chen Qian, Fei Zhang, Shuang Cong  

**Link**: [PDF](https://arxiv.org/pdf/2507.18262)  

**Abstract**: Semantics-driven 3D spatial constraints align highlevel semantic representations with low-level action spaces, facilitating the unification of task understanding and execution in robotic manipulation. The synergistic reasoning of Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs) enables cross-modal 3D spatial constraint construction. Nevertheless, existing methods have three key limitations: (1) coarse semantic granularity in constraint modeling, (2) lack of real-time closed-loop planning, (3) compromised robustness in semantically diverse environments. To address these challenges, we propose ReSem3D, a unified manipulation framework for semantically diverse environments, leveraging the synergy between VFMs and MLLMs to achieve fine-grained visual grounding and dynamically constructs hierarchical 3D spatial constraints for real-time manipulation. Specifically, the framework is driven by hierarchical recursive reasoning in MLLMs, which interact with VFMs to automatically construct 3D spatial constraints from natural language instructions and RGB-D observations in two stages: part-level extraction and region-level refinement. Subsequently, these constraints are encoded as real-time optimization objectives in joint space, enabling reactive behavior to dynamic disturbances. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSem3D performs diverse manipulation tasks under zero-shot conditions, exhibiting strong adaptability and generalization. Code and videos at this https URL. 

**Abstract (ZH)**: 基于语义的三维空间约束驱动机器人操作中的高层次语义表示与低层次动作空间对齐，促进任务理解和执行的统一。多模态大型语言模型和视觉基础模型的协同推理 enables 跨模态三维空间约束构建。然而，现有方法存在三个关键局限：（1）约束建模中粗粒度的语义粒度，（2）缺乏实时闭环规划，（3）在语义多样的环境中鲁棒性较低。为解决这些挑战，我们提出 ReSem3D，一种利用视觉基础模型和多模态大型语言模型之间的协同作用，实现细粒度的视觉定位并动态构建层次化三维空间约束的统一操作框架，以支持实时操作。具体而言，该框架由多模态大型语言模型中的分层递归推理驱动，与视觉基础模型交互，自动在两个阶段从自然语言指令和RGB-D观测中构建三维空间约束：部分级提取和区域级细化。随后，这些约束被编码为关节空间中的实时优化目标，使机器人能够对动态干扰进行反应。我们在语义丰富的家庭环境和稀疏的化学实验室环境中进行了广泛的仿真和真实世界实验。实验结果表明，ReSem3D 在零样本条件下执行多样化的操作任务，显示出强大的适应性和泛化能力。有关代码和视频请访问 this https URL。 

---
# PinchBot: Long-Horizon Deformable Manipulation with Guided Diffusion Policy 

**Title (ZH)**: PinchBot: 带引导扩散策略的长时变形操作 

**Authors**: Alison Bartsch, Arvind Car, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2507.17846)  

**Abstract**: Pottery creation is a complicated art form that requires dexterous, precise and delicate actions to slowly morph a block of clay to a meaningful, and often useful 3D goal shape. In this work, we aim to create a robotic system that can create simple pottery goals with only pinch-based actions. This pinch pottery task allows us to explore the challenges of a highly multi-modal and long-horizon deformable manipulation task. To this end, we present PinchBot, a goal-conditioned diffusion policy model that when combined with pre-trained 3D point cloud embeddings, task progress prediction and collision-constrained action projection, is able to successfully create a variety of simple pottery goals. For experimental videos and access to the demonstration dataset, please visit our project website: this https URL. 

**Abstract (ZH)**: 陶器制作是一项复杂的艺术形式，要求精细、精确且细腻的动作，逐渐将一团黏土转变为有意义且时常具有实用价值的3D目标形状。本文旨在创建一个仅依赖捏压动作即可实现简单陶器目标的机器人系统。捏压陶器任务使我们能够探索高度多模态和长期规划变形操作任务的挑战。为此，我们提出了PinchBot，这是一种基于目标的扩散策略模型，通过结合预先训练的3D点云嵌入、任务进展预测和碰撞约束动作投影，能够成功地创建多种简单的陶器目标。欲查看实验视频并访问演示数据集，请访问我们的项目网站: [此链接]。 

---
# SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law 

**Title (ZH)**: SafeWork-R1: 安全与智能共生下的AI-45°定律 

**Authors**: Shanghai AI Lab, Yicheng Bao, Guanxu Chen, Mingkang Chen, Yunhao Chen, Chiyu Chen, Lingjie Chen, Sirui Chen, Xinquan Chen, Jie Cheng, Yu Cheng, Dengke Deng, Yizhuo Ding, Dan Ding, Xiaoshan Ding, Yi Ding, Zhichen Dong, Lingxiao Du, Yuyu Fan, Xinshun Feng, Yanwei Fu, Yuxuan Gao, Ruijun Ge, Tianle Gu, Lujun Gui, Jiaxuan Guo, Qianxi He, Yuenan Hou, Xuhao Hu, Hong Huang, Kaichen Huang, Shiyang Huang, Yuxian Jiang, Shanzhe Lei, Jie Li, Lijun Li, Hao Li, Juncheng Li, Xiangtian Li, Yafu Li, Lingyu Li, Xueyan Li, Haotian Liang, Dongrui Liu, Qihua Liu, Zhixuan Liu, Bangwei Liu, Huacan Liu, Yuexiao Liu, Zongkai Liu, Chaochao Lu, Yudong Lu, Xiaoya Lu, Zhenghao Lu, Qitan Lv, Caoyuan Ma, Jiachen Ma, Xiaoya Ma, Zhongtian Ma, Lingyu Meng, Ziqi Miao, Yazhe Niu, Yuezhang Peng, Yuan Pu, Han Qi, Chen Qian, Xingge Qiao, Jingjing Qu, Jiashu Qu, Wanying Qu, Wenwen Qu, Xiaoye Qu, Qihan Ren, Qingnan Ren, Qingyu Ren, Jing Shao, Wenqi Shao, Shuai Shao, Dongxing Shi, Xin Song, Xinhao Song, Yan Teng, Xuan Tong, Yingchun Wang, Xuhong Wang, Shujie Wang, Xin Wang, Yige Wang, Yixu Wang, Yuanfu Wang, Futing Wang, Ruofan Wang, Wenjie Wang, Yajie Wang, Muhao Wei, Xiaoyu Wen, Fenghua Weng, Yuqi Wu, Yingtong Xiong, Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18576)  

**Abstract**: We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI. 

**Abstract (ZH)**: 我们介绍SafeWork-R1，这是一种前沿的多模态推理模型，展示了能力和安全的共生进化。它是由我们提出的SafeLadder框架开发的，该框架结合了大规模、渐进式、以安全为导向的强化学习后训练，并由一系列多原则验证器支持。与RLHF等简单的对齐方法不同，SafeLadder使SafeWork-R1能够发展出内在的安全推理和自我反思能力，产生安全的“豁然开朗”时刻。值得注意的是，SafeWork-R1在其基线模型Qwen2.5-VL-72B的基础上，在安全相关的基准测试中平均提高了46.54%，并未牺牲一般能力，并在与GPT-4.1和Claude Opus 4等领先专有模型的安全性能上达到了先进水平。为了进一步增强其可靠性，我们实施了两种不同的推理时干预方法和一种详议搜索机制，确保逐步骤验证。最后，我们进一步开发了SafeWork-R1-InternVL3-78B、SafeWork-R1-DeepSeek-70B和SafeWork-R1-Qwen2.5VL-7B等模型。所有这些模型都表明，安全性和能力可以协同共生进化，突显了我们框架在构建稳健、可靠和可信赖的通用人工智能方面的普适性。 

---
# AlphaGo Moment for Model Architecture Discovery 

**Title (ZH)**: AlphaGo时刻：模型架构发现 

**Authors**: Yixiu Liu, Yang Nan, Weixian Xu, Xiangkun Hu, Lyumanshan Ye, Zhen Qin, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18074)  

**Abstract**: While AI systems demonstrate exponentially improving capabilities, the pace of AI research itself remains linearly bounded by human cognitive capacity, creating an increasingly severe development bottleneck. We present ASI-Arch, the first demonstration of Artificial Superintelligence for AI research (ASI4AI) in the critical domain of neural architecture discovery--a fully autonomous system that shatters this fundamental constraint by enabling AI to conduct its own architectural innovation. Moving beyond traditional Neural Architecture Search (NAS), which is fundamentally limited to exploring human-defined spaces, we introduce a paradigm shift from automated optimization to automated innovation. ASI-Arch can conduct end-to-end scientific research in the domain of architecture discovery, autonomously hypothesizing novel architectural concepts, implementing them as executable code, training and empirically validating their performance through rigorous experimentation and past experience. ASI-Arch conducted 1,773 autonomous experiments over 20,000 GPU hours, culminating in the discovery of 106 innovative, state-of-the-art (SOTA) linear attention architectures. Like AlphaGo's Move 37 that revealed unexpected strategic insights invisible to human players, our AI-discovered architectures demonstrate emergent design principles that systematically surpass human-designed baselines and illuminate previously unknown pathways for architectural innovation. Crucially, we establish the first empirical scaling law for scientific discovery itself--demonstrating that architectural breakthroughs can be scaled computationally, transforming research progress from a human-limited to a computation-scalable process. We provide comprehensive analysis of the emergent design patterns and autonomous research capabilities that enabled these breakthroughs, establishing a blueprint for self-accelerating AI systems. 

**Abstract (ZH)**: AI研究中的ASIArch：突破神经架构发现瓶颈的人工超智能系统 

---
# Moving Out: Physically-grounded Human-AI Collaboration 

**Title (ZH)**: 移步而出：基于物理的人机协作 

**Authors**: Xuhui Kang, Sung-Wook Lee, Haolin Liu, Yuyan Wang, Yen-Ling Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18623)  

**Abstract**: The ability to adapt to physical actions and constraints in an environment is crucial for embodied agents (e.g., robots) to effectively collaborate with humans. Such physically grounded human-AI collaboration must account for the increased complexity of the continuous state-action space and constrained dynamics caused by physical constraints. In this paper, we introduce \textit{Moving Out}, a new human-AI collaboration benchmark that resembles a wide range of collaboration modes affected by physical attributes and constraints, such as moving heavy items together and maintaining consistent actions to move a big item around a corner. Using Moving Out, we designed two tasks and collected human-human interaction data to evaluate models' abilities to adapt to diverse human behaviors and unseen physical attributes. To address the challenges in physical environments, we propose a novel method, BASS (Behavior Augmentation, Simulation, and Selection), to enhance the diversity of agents and their understanding of the outcome of actions. Our experiments show that BASS outperforms state-of-the-art models in AI-AI and human-AI collaboration. The project page is available at \href{this https URL}{this https URL\_ai/}. 

**Abstract (ZH)**: 具备适应环境物理动作和约束的能力对于使实体代理（例如机器人）有效地与人类协作至关重要。这种基于物理的人机协作必须考虑到由于物理约束引起的连续状态-动作空间和受限动力学的复杂性增加。本文介绍了Moving Out，这是一个新的框架，涵盖了广泛受物理属性和约束影响的合作模式，如共同搬运重物和在转角处保持一致动作以移动大型物品。利用Moving Out，我们设计了两个任务并收集了人类-人类交互数据以评估模型适应多样化人类行为和未见物理属性的能力。为了应对物理环境中的挑战，我们提出了一种新型方法BASS（行为增强、仿真和选择），以增强代理人多样性和对其行为结果的理解。我们的实验表明，BASS在人机协作和AI-AI协作中优于现有最先进的模型。项目页面详见[这个链接](https://github.com/_ai/)。 

---
# Reinforced Embodied Active Defense: Exploiting Adaptive Interaction for Robust Visual Perception in Adversarial 3D Environments 

**Title (ZH)**: 强化嵌身体动防御：在 adversarial 3D 环境中利用自适应交互实现稳健的视觉感知 

**Authors**: Xiao Yang, Lingxuan Wu, Lizhong Wang, Chengyang Ying, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18484)  

**Abstract**: Adversarial attacks in 3D environments have emerged as a critical threat to the reliability of visual perception systems, particularly in safety-sensitive applications such as identity verification and autonomous driving. These attacks employ adversarial patches and 3D objects to manipulate deep neural network (DNN) predictions by exploiting vulnerabilities within complex scenes. Existing defense mechanisms, such as adversarial training and purification, primarily employ passive strategies to enhance robustness. However, these approaches often rely on pre-defined assumptions about adversarial tactics, limiting their adaptability in dynamic 3D settings. To address these challenges, we introduce Reinforced Embodied Active Defense (Rein-EAD), a proactive defense framework that leverages adaptive exploration and interaction with the environment to improve perception robustness in 3D adversarial contexts. By implementing a multi-step objective that balances immediate prediction accuracy with predictive entropy minimization, Rein-EAD optimizes defense strategies over a multi-step horizon. Additionally, Rein-EAD involves an uncertainty-oriented reward-shaping mechanism that facilitates efficient policy updates, thereby reducing computational overhead and supporting real-world applicability without the need for differentiable environments. Comprehensive experiments validate the effectiveness of Rein-EAD, demonstrating a substantial reduction in attack success rates while preserving standard accuracy across diverse tasks. Notably, Rein-EAD exhibits robust generalization to unseen and adaptive attacks, making it suitable for real-world complex tasks, including 3D object classification, face recognition and autonomous driving. 

**Abstract (ZH)**: adversarial 攻击在 3D 环境中已成为视觉感知系统可靠性的关键威胁，特别是在身份验证和自动驾驶等安全性敏感应用中。这些攻击利用对抗性补丁和 3D 对象通过利用复杂场景中的漏洞来操控深度神经网络（DNN）预测。现有的防御机制，如对抗性训练和净化，主要采用被动策略来增强鲁棒性。然而，这些方法通常依赖于对手战术的预定义假设，限制了其在动态 3D 设置下的适应性。为应对这些挑战，我们提出了增强体外主动防御（Rein-EAD）框架，该框架利用适应性探索与环境的交互来提高 3D 对抗性环境中的感知鲁棒性。通过实施平衡即时预测准确性和预测熵最小化的多步骤目标，Rein-EAD 在多步骤时间范围内优化防御策略。此外，Rein-EAD 还包括一个以不确定性为导向的奖励塑造机制，有助于高效策略更新，从而减少计算开销并支持在非可微环境中实现的现实世界应用。全面的实验验证了 Rein-EAD 的有效性，展示了其在不同任务中显著降低攻击成功率的能力，同时保持标准准确性。值得注意的是，Rein-EAD 能够 robust 地泛化到未知和适应性攻击，使其适用于包括 3D 对象分类、人脸识别和自动驾驶在内的现实世界复杂任务。 

---
