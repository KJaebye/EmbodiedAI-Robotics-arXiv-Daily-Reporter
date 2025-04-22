# Interpretable Locomotion Prediction in Construction Using a Memory-Driven LLM Agent With Chain-of-Thought Reasoning 

**Title (ZH)**: 基于记忆驱动的链式思维推理LLM代理的可解释建筑施工运动预测 

**Authors**: Ehsan Ahmadi, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15263)  

**Abstract**: Construction tasks are inherently unpredictable, with dynamic environments and safety-critical demands posing significant risks to workers. Exoskeletons offer potential assistance but falter without accurate intent recognition across diverse locomotion modes. This paper presents a locomotion prediction agent leveraging Large Language Models (LLMs) augmented with memory systems, aimed at improving exoskeleton assistance in such settings. Using multimodal inputs - spoken commands and visual data from smart glasses - the agent integrates a Perception Module, Short-Term Memory (STM), Long-Term Memory (LTM), and Refinement Module to predict locomotion modes effectively. Evaluation reveals a baseline weighted F1-score of 0.73 without memory, rising to 0.81 with STM, and reaching 0.90 with both STM and LTM, excelling with vague and safety-critical commands. Calibration metrics, including a Brier Score drop from 0.244 to 0.090 and ECE from 0.222 to 0.044, affirm improved reliability. This framework supports safer, high-level human-exoskeleton collaboration, with promise for adaptive assistive systems in dynamic industries. 

**Abstract (ZH)**: 一种利用大型语言模型增强的记忆系统进行步行模式预测的外骨骼辅助方法 

---
# A General Infrastructure and Workflow for Quadrotor Deep Reinforcement Learning and Reality Deployment 

**Title (ZH)**: 四旋翼深度强化学习及其实际部署的一般基础设施和工作流 

**Authors**: Kangyao Huang, Hao Wang, Yu Luo, Jingyu Chen, Jintao Chen, Xiangkui Zhang, Xiangyang Ji, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15129)  

**Abstract**: Deploying robot learning methods to a quadrotor in unstructured outdoor environments is an exciting task. Quadrotors operating in real-world environments by learning-based methods encounter several challenges: a large amount of simulator generated data required for training, strict demands for real-time processing onboard, and the sim-to-real gap caused by dynamic and noisy conditions. Current works have made a great breakthrough in applying learning-based methods to end-to-end control of quadrotors, but rarely mention the infrastructure system training from scratch and deploying to reality, which makes it difficult to reproduce methods and applications. To bridge this gap, we propose a platform that enables the seamless transfer of end-to-end deep reinforcement learning (DRL) policies. We integrate the training environment, flight dynamics control, DRL algorithms, the MAVROS middleware stack, and hardware into a comprehensive workflow and architecture that enables quadrotors' policies to be trained from scratch to real-world deployment in several minutes. Our platform provides rich types of environments including hovering, dynamic obstacle avoidance, trajectory tracking, balloon hitting, and planning in unknown environments, as a physical experiment benchmark. Through extensive empirical validation, we demonstrate the efficiency of proposed sim-to-real platform, and robust outdoor flight performance under real-world perturbations. Details can be found from our website this https URL. 

**Abstract (ZH)**: 将基于学习的方法部署到户外非结构化环境中的一体化四旋翼机器人平台 

---
# Dynamic Legged Ball Manipulation on Rugged Terrains with Hierarchical Reinforcement Learning 

**Title (ZH)**: 在崎岖地形上基于层次强化学习的动态腿式球操作 

**Authors**: Dongjie Zhu, Zhuo Yang, Tianhang Wu, Luzhou Ge, Xuesong Li, Qi Liu, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14989)  

**Abstract**: Advancing the dynamic loco-manipulation capabilities of quadruped robots in complex terrains is crucial for performing diverse tasks. Specifically, dynamic ball manipulation in rugged environments presents two key challenges. The first is coordinating distinct motion modalities to integrate terrain traversal and ball control seamlessly. The second is overcoming sparse rewards in end-to-end deep reinforcement learning, which impedes efficient policy convergence. To address these challenges, we propose a hierarchical reinforcement learning framework. A high-level policy, informed by proprioceptive data and ball position, adaptively switches between pre-trained low-level skills such as ball dribbling and rough terrain navigation. We further propose Dynamic Skill-Focused Policy Optimization to suppress gradients from inactive skills and enhance critical skill learning. Both simulation and real-world experiments validate that our methods outperform baseline approaches in dynamic ball manipulation across rugged terrains, highlighting its effectiveness in challenging environments. Videos are on our website: this http URL. 

**Abstract (ZH)**: 在复杂地形中提升四足机器人动态搬运能力和操作球体的能力对于执行多样化任务至关重要。具体地，险恶环境中的动态球体操纵面临着两个核心挑战。首先，协调不同的运动模式以无缝地整合地形穿越和球体控制。其次，端到端深度强化学习中的稀疏奖励阻碍了高效政策收敛。为应对这些挑战，我们提出了一种分层强化学习框架。高层策略通过 proprioceptive 数据和球体位置信息，适应性地切换到预先训练好的低层技能，如球体运球和崎岖地形导航。我们还提出了一种动态技能集中策略优化方法，以抑制不活跃技能的梯度，并增强关键技能的学习。仿真和实地实验均表明，我们的方法在险恶地形中的动态球体操纵性能优于基准方法，凸显了其在挑战性环境中的有效性。视频详见我们的网站：this http URL。 

---
# An LLM-enabled Multi-Agent Autonomous Mechatronics Design Framework 

**Title (ZH)**: 一种基于LLM的多agent自主机电设计框架 

**Authors**: Zeyu Wang, Frank P.-W. Lo, Qian Chen, Yongqi Zhang, Chen Lin, Xu Chen, Zhenhua Yu, Alexander J. Thompson, Eric M. Yeatman, Benny P. L. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14681)  

**Abstract**: Existing LLM-enabled multi-agent frameworks are predominantly limited to digital or simulated environments and confined to narrowly focused knowledge domain, constraining their applicability to complex engineering tasks that require the design of physical embodiment, cross-disciplinary integration, and constraint-aware reasoning. This work proposes a multi-agent autonomous mechatronics design framework, integrating expertise across mechanical design, optimization, electronics, and software engineering to autonomously generate functional prototypes with minimal direct human design input. Operating primarily through a language-driven workflow, the framework incorporates structured human feedback to ensure robust performance under real-world constraints. To validate its capabilities, the framework is applied to a real-world challenge involving autonomous water-quality monitoring and sampling, where traditional methods are labor-intensive and ecologically disruptive. Leveraging the proposed system, a fully functional autonomous vessel was developed with optimized propulsion, cost-effective electronics, and advanced control. The design process was carried out by specialized agents, including a high-level planning agent responsible for problem abstraction and dedicated agents for structural, electronics, control, and software development. This approach demonstrates the potential of LLM-based multi-agent systems to automate real-world engineering workflows and reduce reliance on extensive domain expertise. 

**Abstract (ZH)**: 现有的基于大语言模型的多 agents 框架主要局限于数字或模拟环境，并且局限于狭窄的知识领域，限制了其在需要设计物理实体、跨学科集成和约束aware推理的复杂工程任务中的应用。本工作提出了一种多 agents 自主机电设计框架，将机械设计、优化、电子和软件工程方面的专业知识整合起来，以最少的人工设计输入自动生成功能原型。该框架主要通过语言驱动的工作流运行，并整合结构化的人类反馈以确保在实际约束下的稳健性能。为了验证其能力，该框架应用于一项涉及自主水质监测和采样的真实世界挑战，其中传统方法劳动密集且生态破坏性。利用提出系统，开发了一种具有优化推进、低成本电子设备和高级控制的全功能自主船舶。设计过程由专门的代理执行，包括负责问题抽象的高级规划代理和专门的结构设计、电子设备、控制和软件开发代理。这种方法展示了基于大语言模型的多 agents 系统在自动化真实世界工程工作流和减少对广泛领域专业知识依赖方面的潜力。 

---
# RoboOcc: Enhancing the Geometric and Semantic Scene Understanding for Robots 

**Title (ZH)**: RoboOcc: 提升机器人对几何与语义场景理解的研究 

**Authors**: Zhang Zhang, Qiang Zhang, Wei Cui, Shuai Shi, Yijie Guo, Gang Han, Wen Zhao, Hengle Ren, Renjing Xu, Jian Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14604)  

**Abstract**: 3D occupancy prediction enables the robots to obtain spatial fine-grained geometry and semantics of the surrounding scene, and has become an essential task for embodied perception. Existing methods based on 3D Gaussians instead of dense voxels do not effectively exploit the geometry and opacity properties of Gaussians, which limits the network's estimation of complex environments and also limits the description of the scene by 3D Gaussians. In this paper, we propose a 3D occupancy prediction method which enhances the geometric and semantic scene understanding for robots, dubbed RoboOcc. It utilizes the Opacity-guided Self-Encoder (OSE) to alleviate the semantic ambiguity of overlapping Gaussians and the Geometry-aware Cross-Encoder (GCE) to accomplish the fine-grained geometric modeling of the surrounding scene. We conduct extensive experiments on Occ-ScanNet and EmbodiedOcc-ScanNet datasets, and our RoboOcc achieves state-of the-art performance in both local and global camera settings. Further, in ablation studies of Gaussian parameters, the proposed RoboOcc outperforms the state-of-the-art methods by a large margin of (8.47, 6.27) in IoU and mIoU metric, respectively. The codes will be released soon. 

**Abstract (ZH)**: 3D 占有预测使机器人能够获得周围场景的空间精细几何和语义信息，并已成为体现式感知中的一个必备任务。现有的基于 3D 高斯分布而非密集体素的方法未能有效地利用高斯的几何和不透明性特性，这限制了网络对复杂环境的估计能力，也限制了由 3D 高斯分布描述场景的能力。本文提出了一种增强机器人对几何和语义场景理解的 3D 占有预测方法，名为 RoboOcc。它利用不透明度引导自编码器（OSE）减轻重叠高斯的语义不确定性，并利用几何感知交叉编码器（GCE）完成周围场景的精细几何建模。我们在 Occ-ScanNet 和 EmbodiedOcc-ScanNet 数据集上进行了广泛的实验，RoboOcc 在局部和全局摄像机设置中均实现了最佳性能。进一步的高斯参数消融研究中，提出的 RoboOcc 在 IoU 和 mIoU 指标上分别比最先进的方法显著提高了（8.47，6.27）。代码将很快发布。 

---
# Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction 

**Title (ZH)**: Phoenix：基于运动的细粒度机器人动作修正自我反思框架 

**Authors**: Wenke Xia, Ruoxuan Feng, Dong Wang, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14588)  

**Abstract**: Building a generalizable self-correction system is crucial for robots to recover from failures. Despite advancements in Multimodal Large Language Models (MLLMs) that empower robots with semantic reflection ability for failure, translating semantic reflection into how to correct fine-grained robotic actions remains a significant challenge. To address this gap, we build the Phoenix framework, which leverages motion instruction as a bridge to connect high-level semantic reflection with low-level robotic action correction. In this motion-based self-reflection framework, we start with a dual-process motion adjustment mechanism with MLLMs to translate the semantic reflection into coarse-grained motion instruction adjustment. To leverage this motion instruction for guiding how to correct fine-grained robotic actions, a multi-task motion-conditioned diffusion policy is proposed to integrate visual observations for high-frequency robotic action correction. By combining these two models, we could shift the demand for generalization capability from the low-level manipulation policy to the MLLMs-driven motion adjustment model and facilitate precise, fine-grained robotic action correction. Utilizing this framework, we further develop a lifelong learning method to automatically improve the model's capability from interactions with dynamic environments. The experiments conducted in both the RoboMimic simulation and real-world scenarios prove the superior generalization and robustness of our framework across a variety of manipulation tasks. Our code is released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 建立可泛化的自我修正系统对于机器人从故障中恢复至关重要。尽管多模态大型语言模型（MLLMs）的发展赋予了机器人语义反思能力以应对故障，但将语义反思转化为如何纠正细粒度机器人动作仍是一个重大挑战。为了解决这一差距，我们构建了Phoenix框架，该框架利用运动指令作为桥梁，连接高层语义反思与低层机器人动作纠正。在基于运动的自我反思框架中，我们从使用MLLMs的双重过程运动调整机制开始，将语义反思转化为粗粒度运动指令调整。为了利用该运动指令引导如何纠正细粒度机器人动作，我们提出了一个多任务运动条件化扩散策略，结合视觉观察进行高频率的机器人动作纠正。通过结合这两种模型，我们将对泛化能力的需求从低层级操作策略转移至由MLLMs驱动的运动调整模型，并促进了精确、细粒度的机器人动作纠正。利用该框架，我们进一步开发了一种终身学习方法，可自动通过与动态环境的交互来提高模型能力。我们在RoboMimic仿真和真实世界场景中的实验证明了该框架在各种操作任务中具有优越的泛化能力和鲁棒性。我们的代码发布在\href{this https URL}{this https URL}。 

---
# Modality Selection and Skill Segmentation via Cross-Modality Attention 

**Title (ZH)**: 跨模态注意力驱动的模态选择与技能分割 

**Authors**: Jiawei Jiang, Kei Ota, Devesh K. Jha, Asako Kanezaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14573)  

**Abstract**: Incorporating additional sensory modalities such as tactile and audio into foundational robotic models poses significant challenges due to the curse of dimensionality. This work addresses this issue through modality selection. We propose a cross-modality attention (CMA) mechanism to identify and selectively utilize the modalities that are most informative for action generation at each timestep. Furthermore, we extend the application of CMA to segment primitive skills from expert demonstrations and leverage this segmentation to train a hierarchical policy capable of solving long-horizon, contact-rich manipulation tasks. 

**Abstract (ZH)**: 将触觉和音频等额外的感官模态融入基础机器人模型中面临着维度灾难的问题。本工作通过模态选择来应对这一问题。我们提出了一种跨模态注意力（CMA）机制，以识别并选择性利用在每个时间步对未来动作生成最有信息量的模态。此外，我们将CMA的应用扩展到从专家演示中分割原始技能，并利用这种分割来训练一个分层策略，以解决长期 horizon、接触丰富的操作任务。 

---
# ApexNav: An Adaptive Exploration Strategy for Zero-Shot Object Navigation with Target-centric Semantic Fusion 

**Title (ZH)**: ApexNav：面向目标的自适应探索策略与目标中心语义融合的零样本对象导航 

**Authors**: Mingjie Zhang, Yuheng Du, Chengkai Wu, Jinni Zhou, Zhenchao Qi, Jun Ma, Boyu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14478)  

**Abstract**: Navigating unknown environments to find a target object is a significant challenge. While semantic information is crucial for navigation, relying solely on it for decision-making may not always be efficient, especially in environments with weak semantic cues. Additionally, many methods are susceptible to misdetections, especially in environments with visually similar objects. To address these limitations, we propose ApexNav, a zero-shot object navigation framework that is both more efficient and reliable. For efficiency, ApexNav adaptively utilizes semantic information by analyzing its distribution in the environment, guiding exploration through semantic reasoning when cues are strong, and switching to geometry-based exploration when they are weak. For reliability, we propose a target-centric semantic fusion method that preserves long-term memory of the target object and similar objects, reducing false detections and minimizing task failures. We evaluate ApexNav on the HM3Dv1, HM3Dv2, and MP3D datasets, where it outperforms state-of-the-art methods in both SR and SPL metrics. Comprehensive ablation studies further demonstrate the effectiveness of each module. Furthermore, real-world experiments validate the practicality of ApexNav in physical environments. Project page is available at this https URL. 

**Abstract (ZH)**: 未知环境下导航寻找目标物体是一项重大挑战。虽然语义信息对于导航至关重要，但仅依赖其进行决策可能并不总是高效的，尤其是在语义线索较弱的环境中。此外，许多方法在视觉上相似的物体环境中容易产生误检测。为解决这些限制，我们提出了ApexNav，这是一种更高效且可靠的零-shot对象导航框架。为了提高效率，ApexNav通过分析环境中的语义信息分布，在语义线索强时通过语义推理引导探索，在语义线索弱时切换到基于几何的探索。为了提高可靠性，我们提出了一种以目标为中心的语义融合方法，保留了目标物体及其相似物体的长时记忆，减少了误检，并最小化任务失败。我们在HM3Dv1、HM3Dv2和MP3D数据集上评估了ApexNav，在SR和SPL指标上均优于现有方法。进一步的消融研究还证明了每个模块的有效性。此外，实地实验验证了ApexNav在物理环境中的实用性。项目页面请参见此链接。 

---
# ExFace: Expressive Facial Control for Humanoid Robots with Diffusion Transformers and Bootstrap Training 

**Title (ZH)**: ExFace: 表情 facial 控制 for 人形机器人 with 扩散变换器和 Bootstrap 训练 

**Authors**: Dong Zhang, Jingwei Peng, Yuyang Jiao, Jiayuan Gu, Jingyi Yu, Jiahao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14477)  

**Abstract**: This paper presents a novel Expressive Facial Control (ExFace) method based on Diffusion Transformers, which achieves precise mapping from human facial blendshapes to bionic robot motor control. By incorporating an innovative model bootstrap training strategy, our approach not only generates high-quality facial expressions but also significantly improves accuracy and smoothness. Experimental results demonstrate that the proposed method outperforms previous methods in terms of accuracy, frame per second (FPS), and response time. Furthermore, we develop the ExFace dataset driven by human facial data. ExFace shows excellent real-time performance and natural expression rendering in applications such as robot performances and human-robot interactions, offering a new solution for bionic robot interaction. 

**Abstract (ZH)**: 基于扩散变换器的新型表情面部控制方法（ExFace）及其应用 

---
# Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning 

**Title (ZH)**: adversarial 运动学习与运动模仿的人形机器人策略学习 

**Authors**: Jiyuan Shi, Xinzhe Liu, Dewei Wang, Ouyang Lu, Sören Schwertfeger, Fuchun Sun, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14305)  

**Abstract**: Humans exhibit diverse and expressive whole-body movements. However, attaining human-like whole-body coordination in humanoid robots remains challenging, as conventional approaches that mimic whole-body motions often neglect the distinct roles of upper and lower body. This oversight leads to computationally intensive policy learning and frequently causes robot instability and falls during real-world execution. To address these issues, we propose Adversarial Locomotion and Motion Imitation (ALMI), a novel framework that enables adversarial policy learning between upper and lower body. Specifically, the lower body aims to provide robust locomotion capabilities to follow velocity commands while the upper body tracks various motions. Conversely, the upper-body policy ensures effective motion tracking when the robot executes velocity-based movements. Through iterative updates, these policies achieve coordinated whole-body control, which can be extended to loco-manipulation tasks with teleoperation systems. Extensive experiments demonstrate that our method achieves robust locomotion and precise motion tracking in both simulation and on the full-size Unitree H1 robot. Additionally, we release a large-scale whole-body motion control dataset featuring high-quality episodic trajectories from MuJoCo simulations deployable on real robots. The project page is this https URL. 

**Abstract (ZH)**: 人类展示了多样且富有表现力的全身运动。然而，赋予类人机器人类似人类的全身协调性仍然具有挑战性，因为传统的模仿全身运动的方法往往忽视了上下身的 distinct 角色。这种疏忽导致了计算强度大的策略学习，并且经常在实际执行过程中使机器人不稳定并摔倒。为了解决这些问题，我们提出了一种新的框架——对抗性行走与动作模仿（Adversarial Locomotion and Motion Imitation, ALMI），该框架能够在上下身之间进行对抗性策略学习。具体而言，下身旨在提供稳健的行走能力以跟随速度命令，而上身则跟踪各种动作。相反，上身策略确保机器人执行基于速度的动作时能够有效跟踪动作。通过迭代更新，这些策略能够实现协调的全身控制，该控制还可以通过远程操作系统扩展到操作操控任务。大量实验证明，我们的方法在模拟以及全尺寸的Untreed H1机器人上都能够实现稳健的行走和精确的动作跟踪。此外，我们还发布了包含高质量 episodic 轨迹的大规模全身运动控制数据集，这些轨迹来源于 MuJoCo 模拟并在实际机器人上可部署。项目页面详见 this https URL。 

---
# Experience-based Refinement of Task Planning Knowledge in Autonomous Robots 

**Title (ZH)**: 基于经验的任务规划知识精炼在自主机器人中的应用 

**Authors**: Hadeel Jazzaa, Thomas McCluskey, David Peebles  

**Link**: [PDF](https://arxiv.org/pdf/2504.14259)  

**Abstract**: The requirement for autonomous robots to exhibit higher-level cognitive skills by planning and adapting in an ever-changing environment is indeed a great challenge for the AI community. Progress has been made in the automated planning community on refinement and repair of an agent's symbolic knowledge to do task planning in an incomplete or changing environmental model, but these advances up to now have not been transferred to real physical robots. This paper demonstrates how a physical robot can be capable of adapting its symbolic knowledge of the environment, by using experiences in robot action execution to drive knowledge refinement and hence to improve the success rate of the task plans the robot creates. To implement more robust planning systems, we propose a method for refining domain knowledge to improve the knowledge on which intelligent robot behavior is based. This architecture has been implemented and evaluated using a NAO robot. The refined knowledge leads to the future synthesis of task plans which demonstrate decreasing rates of failure over time as faulty knowledge is removed or adjusted. 

**Abstract (ZH)**: 自主机器人在不断变化环境中表现出更高层次认知能力的要求确实是AI社区的一大挑战。尽管自动规划领域在细化和修复代理符号知识以进行任务规划方面取得了进展，但在不完整或变化的环境模型中，这些进展尚未转移到真实物理机器人上。本文展示了如何通过利用机器人动作执行的经验来驱动知识细化，从而使物理机器人能够适应其对环境的符号知识，进而提高机器人创建任务计划的成功率。为了构建更稳健的规划系统，我们提出了一种方法，用于细化领域知识，以提高基于智能机器人行为的知识基础。该架构已在NAO机器人上实现并进行了评估。细化的知识导致未来合成的任务计划随着时间的推移失效率降低，无效知识被删除或调整。 

---
# Unreal Robotics Lab: A High-Fidelity Robotics Simulator with Advanced Physics and Rendering 

**Title (ZH)**: 虚实机器人实验室：一种高保真机器人模拟器，具备高级物理和渲染技术 

**Authors**: Jonathan Embley-Riches, Jianwei Liu, Simon Julier, Dimitrios Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2504.14135)  

**Abstract**: High-fidelity simulation is essential for robotics research, enabling safe and efficient testing of perception, control, and navigation algorithms. However, achieving both photorealistic rendering and accurate physics modeling remains a challenge. This paper presents a novel simulation framework--the Unreal Robotics Lab (URL) that integrates the Unreal Engine's advanced rendering capabilities with MuJoCo's high-precision physics simulation. Our approach enables realistic robotic perception while maintaining accurate physical interactions, facilitating benchmarking and dataset generation for vision-based robotics applications. The system supports complex environmental effects, such as smoke, fire, and water dynamics, which are critical for evaluating robotic performance under adverse conditions. We benchmark visual navigation and SLAM methods within our framework, demonstrating its utility for testing real-world robustness in controlled yet diverse scenarios. By bridging the gap between physics accuracy and photorealistic rendering, our framework provides a powerful tool for advancing robotics research and sim-to-real transfer. 

**Abstract (ZH)**: 高保真模拟对于机器人研究至关重要，能够实现感知、控制和导航算法的安全和高效测试。然而，同时实现逼真的渲染和准确的物理建模仍然是一个挑战。本文提出了一种新颖的模拟框架——Unreal Robotics Lab (URL)，该框架将Unreal Engine的高级渲染能力与MuJoCo的高精度物理模拟相结合。我们的方法实现了真实的机器人感知，同时保持精确的物理交互，有利于基于视觉的机器人应用的基准测试和数据集生成。该系统支持复杂的环境效果，如烟雾、火灾和水动力学，对于评估机器人在恶劣条件下的性能至关重要。我们在这个框架内基准测试了视觉导航和SLAM方法，展示了其在受控但多样化场景中测试现实世界鲁棒性的用途。通过在物理准确性与逼真渲染之间架起桥梁，我们的框架提供了一个强大的工具，用于推动机器人研究和仿真实践向真实转化。 

---
# Coordinating Spinal and Limb Dynamics for Enhanced Sprawling Robot Mobility 

**Title (ZH)**: 协调脊柱和 limb 动力学以增强 sprawling 机器人移动性 

**Authors**: Merve Atasever, Ali Okhovat, Azhang Nazaripouya, John Nisbet, Omer Kurkutlu, Jyotirmoy V. Deshmukh, Yasemin Ozkan Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14103)  

**Abstract**: Among vertebrates, salamanders, with their unique ability to transition between walking and swimming gaits, highlight the role of spinal mobility in locomotion. A flexible spine enables undulation of the body through a wavelike motion along the spine, aiding navigation over uneven terrains and obstacles. Yet environmental uncertainties, such as surface irregularities and variations in friction, can significantly disrupt body-limb coordination and cause discrepancies between predictions from mathematical models and real-world outcomes. Addressing this challenge requires the development of sophisticated control strategies capable of dynamically adapting to uncertain conditions while maintaining efficient locomotion. Deep reinforcement learning (DRL) offers a promising framework for handling non-deterministic environments and enabling robotic systems to adapt effectively and perform robustly under challenging conditions. In this study, we comparatively examine learning-based control strategies and biologically inspired gait design methods on a salamander-like robot. 

**Abstract (ZH)**: 在脊椎动物中，蝾螈通过其独特的在步行和游泳姿态间转换的能力，突显了脊柱运动在运动中的作用。灵活的脊柱通过沿脊柱的波状运动产生身体的波状摆动，有助于在不平地形和障碍物上导航。然而，环境不确定性，如表面不规则性和摩擦力的差异，可以显著破坏身体-肢体协调，导致数学模型预测与实际结果之间存在偏差。解决这一挑战需要发展出能够动态适应不确定性条件并保持高效运动的复杂控制策略。深度强化学习（DRL）为处理非确定性环境并使机器人系统在挑战性条件下有效地适应和稳健运行提供了有希望的框架。在本研究中，我们对基于学习的控制策略和受生物启发的步态设计方法在类似蝾螈的机器人上的效果进行了比较研究。 

---
# Dynamic Contrastive Skill Learning with State-Transition Based Skill Clustering and Dynamic Length Adjustment 

**Title (ZH)**: 基于状态转换的技能聚类与动态长度调整的动态对比技能学习 

**Authors**: Jinwoo Choi, Seung-Woo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14805)  

**Abstract**: Reinforcement learning (RL) has made significant progress in various domains, but scaling it to long-horizon tasks with complex decision-making remains challenging. Skill learning attempts to address this by abstracting actions into higher-level behaviors. However, current approaches often fail to recognize semantically similar behaviors as the same skill and use fixed skill lengths, limiting flexibility and generalization. To address this, we propose Dynamic Contrastive Skill Learning (DCSL), a novel framework that redefines skill representation and learning. DCSL introduces three key ideas: state-transition based skill representation, skill similarity function learning, and dynamic skill length adjustment. By focusing on state transitions and leveraging contrastive learning, DCSL effectively captures the semantic context of behaviors and adapts skill lengths to match the appropriate temporal extent of behaviors. Our approach enables more flexible and adaptive skill extraction, particularly in complex or noisy datasets, and demonstrates competitive performance compared to existing methods in task completion and efficiency. 

**Abstract (ZH)**: 动态对比技能学习（DCSL）：一种新的技能表示和学习框架 

---
# Going Down the Abstraction Stream with Augmented Reality and Tangible Robots: the Case of Vector Instruction 

**Title (ZH)**: 沿着抽象化的溪流前行：基于增强现实和可感知机器人的情感指令案例研究 

**Authors**: Sergei Volodin, Hala Khodr, Pierre Dillenbourg, Wafa Johal  

**Link**: [PDF](https://arxiv.org/pdf/2504.14562)  

**Abstract**: Despite being used in many engineering and scientific areas such as physics and mathematics and often taught in high school, graphical vector addition turns out to be a topic prone to misconceptions in understanding even at university-level physics classes. To improve the learning experience and the resulting understanding of vectors, we propose to investigate how concreteness fading implemented with the use of augmented reality and tangible robots could help learners to build a strong representation of vector addition.
We design a gamified learning environment consisting of three concreteness fading stages and conduct an experiment with 30 participants. Our results shows a positive learning gain. We analyze extensively the behavior of the participants to understand the usage of the technological tools -- augmented reality and tangible robots -- during the learning scenario. Finally, we discuss how the combination of these tools shows real advantages in implementing the concreteness fading paradigm. Our work provides empirical insights into how users utilize concrete visualizations conveyed by a haptic-enabled robot and augmented reality in a learning scenario. 

**Abstract (ZH)**: 尽管图形向量加法在物理和数学等领域及高中课程中被广泛应用，但在大学物理课程中仍容易引起误解。为了提高学习体验和对向量的理解，我们提出了一种基于增强现实和实体机器人实现的具体性淡入方法，以帮助学习者建立坚实的向量加法表征。

我们设计了一种包含三个具体性淡入阶段的游戏化学习环境，并进行了30名参与者的实验。结果显示正向学习收益。我们对参与者的行为进行了详细分析，以了解在学习情境中使用增强现实和实体机器人的技术工具的使用情况。最后，我们讨论了这些工具组合如何在实现具体性淡入范式方面展现出真正的优势。我们的研究提供了一种基于触觉机器人和增强现实的具体视觉化应用的用户使用情况的经验性见解。 

---
# DLW-CI: A Dynamic Likelihood-Weighted Cooperative Infotaxis Approach for Multi-Source Search in Urban Environments Using Consumer Drone Networks 

**Title (ZH)**: DLW-CI：一种基于动态似然加权合作信息趋化策略的多源城市环境搜索方法用于消费者无人机网络 

**Authors**: Xiaoran Zhang, Yatai Ji, Yong Zhao, Chuan Ai, Bin Chen, Zhengqiu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14330)  

**Abstract**: Consumer-grade drones equipped with low-cost sensors have emerged as a cornerstone of Autonomous Intelligent Systems (AISs) for environmental monitoring and hazardous substance detection in urban environments. However, existing research primarily addresses single-source search problems, overlooking the complexities of real-world urban scenarios where both the location and quantity of hazardous sources remain unknown. To address this issue, we propose the Dynamic Likelihood-Weighted Cooperative Infotaxis (DLW-CI) approach for consumer drone networks. Our approach enhances multi-drone collaboration in AISs by combining infotaxis (a cognitive search strategy) with optimized source term estimation and an innovative cooperative mechanism. Specifically, we introduce a novel source term estimation method that utilizes multiple parallel particle filters, with each filter dedicated to estimating the parameters of a potentially unknown source within the search scene. Furthermore, we develop a cooperative mechanism based on dynamic likelihood weights to prevent multiple drones from simultaneously estimating and searching for the same source, thus optimizing the energy efficiency and search coverage of the consumer AIS. Experimental results demonstrate that the DLW-CI approach significantly outperforms baseline methods regarding success rate, accuracy, and root mean square error, particularly in scenarios with relatively few sources, regardless of the presence of obstacles. Also, the effectiveness of the proposed approach is verified in a diffusion scenario generated by the computational fluid dynamics (CFD) model. Research findings indicate that our approach could improve source estimation accuracy and search efficiency by consumer drone-based AISs, making a valuable contribution to environmental safety monitoring applications within smart city infrastructure. 

**Abstract (ZH)**: 消费者级无人机装备低成本传感器已成为自主智能系统（AISs）在城市环境中的环境监测和危险物质检测的基础。然而，现有研究主要针对单源搜索问题，忽视了真实城市场景中危险源位置和数量未知的复杂性。为了解决这一问题，我们提出了一种动态概率加权合作感知策略（DLW-CI）方法，用于消费者无人机网络。该方法通过结合感知策略（一种认知搜索策略）、优化的目标项估计和创新的合作机制，增强了多无人机在AISs中的合作。具体来说，我们引入了一种新的目标项估计方法，利用多个并行粒子滤波器，每个滤波器专门用于估计搜索场景中潜在未知目标的参数。此外，我们开发了一种基于动态概率加权的合作机制，以防止多架无人机同时估计和搜索相同的源，从而优化消费者AIS的能量效率和搜索覆盖范围。实验结果表明，DLW-CI方法在成功率、准确性和均方根误差方面显著优于基准方法，特别是在危险源较少的场景中，无论是否存在障碍物。此外，通过计算流体动力学（CFD）模型生成的扩散场景验证了所提出方法的有效性。研究结果表明，我们的方法可以提高基于消费者无人机的AIS对源的估计精度和搜索效率，为智能城市基础设施中的环境安全监测应用做出重要贡献。 

---
# Designing Empathetic Companions: Exploring Personality, Emotion, and Trust in Social Robots 

**Title (ZH)**: 设计共情伴侣机器人：探索社交机器人的个性、情感与信任 

**Authors**: Alice Nardelli, Antonio Sgorbissa, Carmine Tommaso Recchiuto  

**Link**: [PDF](https://arxiv.org/pdf/2504.13964)  

**Abstract**: How should a companion robot behave? In this research, we present a cognitive architecture based on a tailored personality model to investigate the impact of robotic personalities on the perception of companion robots. Drawing from existing literature, we identified empathy, trust, and enjoyability as key factors in building companionship with social robots. Based on these insights, we implemented a personality-dependent, emotion-aware generator, recognizing the crucial role of robot emotions in shaping these elements. We then conducted a user study involving 84 dyadic conversation sessions with the emotional robot Navel, which exhibited different personalities. Results were derived from a multimodal analysis, including questionnaires, open-ended responses, and behavioral observations. This approach allowed us to validate the developed emotion generator and explore the relationship between the personality traits of Agreeableness, Extraversion, Conscientiousness, and Empathy. Furthermore, we drew robust conclusions on how these traits influence relational trust, capability trust, enjoyability, and sociability. 

**Abstract (ZH)**: 同伴机器人应该如何行为？本研究基于定制的人格模型提出一种认知架构，以探究机器人人格对其同伴机器人感知的影响。借鉴现有文献，我们确定了共情、信任和愉悦性是建立与社会机器人同伴关系的关键因素。基于这些见解，我们实现了一个依赖于人格、具有情绪意识的生成器，认识到机器人情绪在塑造这些因素方面的重要性。随后，我们进行了用户研究，涉及84场双人对话会话，其中情感机器人Navel表现出不同的个性。结果通过多模态分析得出，包括问卷调查、开放式回答和行为观察。该方法使我们能够验证开发的情绪生成器，并探索性情特质（如随和性、外向性、尽责性和共情）与关系信任、能力信任、愉悦性和社交性之间的关系。此外，我们得出了关于这些特质如何影响关系信任、能力信任、愉悦性和社交性的稳健结论。 

---
# Exploring the Use of Social Robots to Prepare Children for Radiological Procedures: A Focus Group Study 

**Title (ZH)**: 探索社交机器人在准备儿童进行放射学检查中的应用：一组焦点小组研究 

**Authors**: Massimiliano Nigro, Andrea Righini, Micol Spitale  

**Link**: [PDF](https://arxiv.org/pdf/2504.13881)  

**Abstract**: When children are anxious or scared, it can be hard for them to stay still or follow instructions during medical procedures, making the process more challenging and affecting procedure results. This is particularly true for radiological procedures, where long scan times, confined spaces, and loud noises can cause children to move, significantly impacting scan quality. To this end, sometimes children are sedated, but doctors are constantly seeking alternative non-pharmacological solutions. This work aims to explore how social robots could assist in preparing children for radiological procedures. We have conducted a focus group discussion with five hospital stakeholders, namely radiographers, paediatricians, and clinical engineers, to explore (i) the context regarding children's preparation for radiological procedures, hence their needs and how children are currently prepared, and (ii) the potential role of social robots in this process. The discussion was transcribed and analysed using thematic analysis. Among our findings, we identified three potential roles for a social robot in this preparation process: offering infotainment in the waiting room, acting as a guide within the hospital, and assisting radiographers in preparing children for the procedure. We hope that insights from this study will inform the design of social robots for pediatric healthcare. 

**Abstract (ZH)**: 当儿童焦虑或害怕时，在医疗程序中保持静止或遵循指示会变得困难，这会使得过程更加具有挑战性，并影响程序的结果。特别是在放射学程序中，长时间的扫描、狭小的空间以及嘈杂的声音会促使儿童移动，显著影响扫描质量。为此，有时会使用镇静剂，但医生们一直在寻求非药物的替代解决方案。本研究旨在探索社会机器人如何帮助儿童为放射学程序做准备。我们与五名医疗机构的利益相关者——放射技师、儿科医生和临床工程师——进行了焦点小组讨论，探讨了（i）有关儿童准备放射学程序的背景，包括他们的需求以及目前的准备方式；（ii）社会机器人在这一过程中的潜在作用。讨论内容被转录并使用主题分析方法进行了分析。我们的研究发现中，我们确定了社会机器人在这一准备过程中的三种潜在角色：在候诊室提供娱乐资讯，充当医院内的引导者，以及协助放射技师为儿童准备程序。我们希望通过本研究获得的见解来指导社会机器人在儿科医疗中的设计。 

---
# Manifesting Architectural Subspaces with Two Mobile Robotic Partitions to Facilitate Spontaneous Office Meetings 

**Title (ZH)**: 使用两个移动机器人分区展现建筑子空间，以促进自发办公室会议 

**Authors**: Ozan Balci, Stien Poncelet, Alex Binh Vinh Duc Nguyen, Andrew Vande Moere  

**Link**: [PDF](https://arxiv.org/pdf/2504.13872)  

**Abstract**: Although intended to foster spontaneous interactions among workers, a typical open-plan office layout cannot mitigate visual, acoustic, or privacy-related distractions that originate from unplanned meetings. As office workers often refrain from tackling these issues by manually demarcating or physically relocating to a more suitable subspace that is enclosed by movable partitions, we hypothesise that these subspaces could instead be robotically manifested. This study therefore evaluated the perceived impact of two mobile robotic partitions that were wizarded to jointly manifest an enclosed subspace, to: 1) either `mitigate' or `intervene' in the distractions caused by spontaneous face-to-face or remote meetings; or 2) either `gesturally' or `spatially' nudge a distraction-causing worker to relocate. Our findings suggest how robotic furniture should interact with office workers with and through transient space, and autonomously balance the distractions not only for each individual worker but also for multiple workers sharing the same workspace. 

**Abstract (ZH)**: Although intended to foster spontaneous interactions among workers, a typical open-plan office layout cannot mitigate visual, acoustic, or privacy-related distractions that originate from unplanned meetings. As office workers often refrain from tackling these issues by manually demarcating or physically relocating to a more suitable subspace that is enclosed by movable partitions, we hypothesise that these subspaces could instead be robotically manifested. This study therefore evaluated the perceived impact of two mobile robotic partitions that were wizarded to jointly manifest an enclosed subspace, to: 1) either `mitigate' or `intervene' in the distractions caused by spontaneous face-to-face or remote meetings; or 2) either `gesturally' or `spatially' nudge a distraction-causing worker to relocate. Our findings suggest how robotic furniture should interact with office workers with and through transient space, and autonomously balance the distractions not only for each individual worker but also for multiple workers sharing the same workspace.

一种典型开放式办公室布局原本旨在促进员工之间的自发互动，但却无法减轻由非计划会议引起的视觉、声学或隐私相关干扰。鉴于办公室员工通常不会通过手动划分或物理转移至由可移动隔板包围的更合适的子空间来解决这些问题，我们假设这些子空间可以通过机器人方式呈现。本研究因此评估了通过两名巫师引导的两个移动机器人隔板联合呈现封闭子空间所产生的感知影响：1）要么减轻，要么干预由自发面对面或远程会议引起的干扰；2）要么通过手势，要么通过空间引导引起干扰的员工重新定位。我们的发现表明，机器人家具应如何与办公员工及其暂时空间进行互动，并自主平衡不仅对每个个体员工，而且对共享同一工作空间的多个员工所造成的干扰。 

---
# Towards Balancing Preference and Performance through Adaptive Personalized Explainability 

**Title (ZH)**: 通过自适应个性化解释实现偏好与性能的平衡 

**Authors**: Andrew Silva, Pradyumna Tambwekar, Mariah Schrum, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2504.13856)  

**Abstract**: As robots and digital assistants are deployed in the real world, these agents must be able to communicate their decision-making criteria to build trust, improve human-robot teaming, and enable collaboration. While the field of explainable artificial intelligence (xAI) has made great strides to enable such communication, these advances often assume that one xAI approach is ideally suited to each problem (e.g., decision trees to explain how to triage patients in an emergency or feature-importance maps to explain radiology reports). This fails to recognize that users have diverse experiences or preferences for interaction modalities. In this work, we present two user-studies set in a simulated autonomous vehicle (AV) domain. We investigate (1) population-level preferences for xAI and (2) personalization strategies for providing robot explanations. We find significant differences between xAI modes (language explanations, feature-importance maps, and decision trees) in both preference (p < 0.01) and performance (p < 0.05). We also observe that a participant's preferences do not always align with their performance, motivating our development of an adaptive personalization strategy to balance the two. We show that this strategy yields significant performance gains (p < 0.05), and we conclude with a discussion of our findings and implications for xAI in human-robot interactions. 

**Abstract (ZH)**: 随着机器人和数字助手在真实世界中的部署，这些代理必须能够沟通其决策标准以建立信任、改善人机团队合作并实现协作。虽然可解释人工智能（xAI）领域在实现这种沟通方面取得了巨大进展，这些进步往往假定一种xAI方法最适合每个问题（例如，使用决策树解释紧急情况下如何分诊患者，或使用特征重要性地图解释放射学报告）。这未能认识到用户在交互模式上的多样经历或偏好。在本工作中，我们提出了两个用户研究，设置在一个模拟的自主车辆（AV）领域。我们探究了（1）群体层面的xAI偏好（2）提供机器人解释的个性化策略。我们发现，在偏好（p < 0.01）和性能（p < 0.05）方面，xAI模式（语言解释、特征重要性地图和决策树）之间存在显著差异。我们还观察到，参与者的偏好并不总是与其性能一致，这促使我们开发了一种自适应个性化策略来平衡这两方面。我们展示了这一策略在性能上带来了显著的提升（p < 0.05），并最后讨论了我们的发现及其对人机交互中xAI的含义。 

---
# Stakeholder perspectives on designing socially acceptable social robots and robot avatars for Dubai and multicultural societies 

**Title (ZH)**: 社交媒体接受的社会机器人及 avatar 在迪拜及多元文化社会的设计视角 

**Authors**: Laura Aymerich-Franch, Tarek Taha, Hiroshi Ishiguro, Takahiro Miyashita, Paolo Dario  

**Link**: [PDF](https://arxiv.org/pdf/2504.13854)  

**Abstract**: Robot avatars for customer service are gaining traction in Japan. However, their acceptance in other societal contexts remains underexplored, complicating efforts to design robot avatars suitable for diverse cultural environments. To address this, we interviewed key stakeholders in Dubai's service sector to gain insights into their experiences deploying social robots for customer service, as well as their opinions on the most useful tasks and design features that could maximize customer acceptance of robot avatars in Dubai. Providing information and guiding individuals to specific locations were identified as the most valued functions. Regarding appearance, robotic-looking, highly anthropomorphic designs were the most preferred. Ultra-realistic androids and cartoonish-looking robots elicited mixed reactions, while hybrid androids, low-anthropomorphic robotic designs, and animal-looking robots were considered less suitable or discouraged. Additionally, a psycho-sociological analysis revealed that interactions with robot avatars are influenced by their symbolic meaning, context, and affordances. These findings offer pioneering insights into culturally adaptive robot avatar design, addressing a significant research gap and providing actionable guidelines for deploying socially acceptable robots and avatars in multicultural contexts worldwide. 

**Abstract (ZH)**: 机器人化身在日本的客户服务中正逐渐普及，但在其他社会环境中接受程度仍待探索，这给设计适用于多元文化环境的机器人化身带来了挑战。为了应对这一挑战，我们对迪拜服务行业的关键利益相关者进行了访谈，以了解他们在部署社交机器人进行客户服务方面的经验，以及他们认为哪些最有用的任务和设计特征可以最大限度地提高迪拜机器人化身的客户接受度。结果显示，提供信息和引导个人前往特定地点是最受推崇的功能。在外观方面，高度拟人化的机器人设计最受欢迎，超逼真的机器人和卡通样式的机器人引起了混合反应，而混合机器人、低拟人化机器人设计和动物样式的机器人则被认为不太适合或不被鼓励。此外，心理社会分析表明，与机器人化身的互动受其象征意义、情境和功能的影响。这些发现为文化适应性的机器人化身设计提供了先驱性的洞察，填补了重要的研究空白，并为在全球多元文化环境中部署社会上可接受的机器人和化身提供了可操作的指导。 

---
# Contemplative Wisdom for Superalignment 

**Title (ZH)**: 观照智慧促进超对齐 

**Authors**: Ruben Laukkonen, Fionn Inglis, Shamil Chandaria, Lars Sandved-Smith, Jakob Hohwy, Jonathan Gold, Adam Elwood  

**Link**: [PDF](https://arxiv.org/pdf/2504.15125)  

**Abstract**: As artificial intelligence (AI) improves, traditional alignment strategies may falter in the face of unpredictable self-improvement, hidden subgoals, and the sheer complexity of intelligent systems. Rather than externally constraining behavior, we advocate designing AI with intrinsic morality built into its cognitive architecture and world model. Inspired by contemplative wisdom traditions, we show how four axiomatic principles can instil a resilient Wise World Model in AI systems. First, mindfulness enables self-monitoring and recalibration of emergent subgoals. Second, emptiness forestalls dogmatic goal fixation and relaxes rigid priors. Third, non-duality dissolves adversarial self-other boundaries. Fourth, boundless care motivates the universal reduction of suffering. We find that prompting AI to reflect on these principles improves performance on the AILuminate Benchmark using GPT-4o, particularly when combined. We offer detailed implementation strategies for state-of-the-art models, including contemplative architectures, constitutions, and reinforcement of chain-of-thought. For future systems, the active inference framework may offer the self-organizing and dynamic coupling capabilities needed to enact these insights in embodied agents. This interdisciplinary approach offers a self-correcting and resilient alternative to prevailing brittle control schemes. 

**Abstract (ZH)**: 随着人工智能（AI）的进步，传统的对齐策略可能在面对不可预测的自我提升、隐藏的子目标以及智能系统本身的复杂性时失效。我们主张设计具有内在道德观的AI，将其融入认知架构和世界模型中。受沉思智慧传统启发，我们展示了四种公理原则如何 instil 赋予AI系统一个有韧性的明智世界模型。首先，正念使自我监控和调整新兴的子目标成为可能。第二，空性防止固执的目标固定并放松坚定的先验假设。第三，不二消解了对抗性的自我与他者边界。第四，无限的关怀激励普遍减少苦痛。我们发现，促使AI反思这些原则可以提升使用GPT-4o在AILuminate基准测试中的性能，尤其是结合使用时。我们提供了对最新模型的详细实现策略，包括沉思架构、宪法以及对思维链的强化。对于未来的系统，主动推理框架可能提供自我组织和动态耦合的能力，以在实体代理中实现这些见解。这种跨学科的方法提供了一种自我校正和有韧性的替代方案，而不是当前脆弱的控制方案。 

---
# Text-to-Decision Agent: Learning Generalist Policies from Natural Language Supervision 

**Title (ZH)**: 文本决策代理：从自然语言监督学习通用策略 

**Authors**: Shilin Zhang, Zican Hu, Wenhao Wu, Xinyi Xie, Jianxiang Tang, Chunlin Chen, Daoyi Dong, Yu Cheng, Zhenhong Sun, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15046)  

**Abstract**: RL systems usually tackle generalization by inferring task beliefs from high-quality samples or warmup explorations. The restricted form limits their generality and usability since these supervision signals are expensive and even infeasible to acquire in advance for unseen tasks. Learning directly from the raw text about decision tasks is a promising alternative to leverage a much broader source of supervision. In the paper, we propose Text-to-Decision Agent (T2DA), a simple and scalable framework that supervises generalist policy learning with natural language. We first introduce a generalized world model to encode multi-task decision data into a dynamics-aware embedding space. Then, inspired by CLIP, we predict which textual description goes with which decision embedding, effectively bridging their semantic gap via contrastive language-decision pre-training and aligning the text embeddings to comprehend the environment dynamics. After training the text-conditioned generalist policy, the agent can directly realize zero-shot text-to-decision generation in response to language instructions. Comprehensive experiments on MuJoCo and Meta-World benchmarks show that T2DA facilitates high-capacity zero-shot generalization and outperforms various types of baselines. 

**Abstract (ZH)**: RL系统通常通过从高质量样本或暖启动探索中推断任务信念来应对泛化问题。这种限制性形式限制了其通用性和实用性，因为这些监督信号对于未见过的任务来说在事前获取往往是昂贵的甚至不可行的。直接从原始文本中学习决策任务是一种有希望的替代方案，可以利用更广泛来源的监督。在本文中，我们提出了Text-to-Decision Agent (T2DA)，这是一种简单且可扩展的框架，利用自然语言监督通用策略的學習。我们首先介绍了一个通用的世界模型，将多任务决策数据编码到动力感知的嵌入空间中。然后，受到CLIP的启发，我们预测哪些文本描述与哪个决策嵌入相关联，通过对比语言-决策预训练有效地弥合了它们之间的语义差距，并使文本嵌入能够理解环境动力学。在训练文本条件下的通用策略后，代理可以直接在收到语言指令时实现零样本的文本到决策生成。在MuJoCo和Meta-World基准上的全面实验表明，T2DA 支持高度容量的零样本泛化，并优于各种基线。 

---
# Generative Semantic Communications: Principles and Practices 

**Title (ZH)**: 生成性语义通信：原理与实践 

**Authors**: Xiaojun Yuan, Haoming Ma, Yinuo Huang, Zhoufan Hua, Yong Zuo, Zhi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.14947)  

**Abstract**: Semantic communication leverages artificial intelligence (AI) technologies to extract semantic information from data for efficient transmission, theraby significantly reducing communication cost. With the evolution towards artificial general intelligence (AGI), the increasing demands for AGI services pose new challenges to semantic communication. In response, we propose a new paradigm for AGI-driven communications, called generative semantic communication (GSC), which utilizes advanced AI technologies such as foundation models and generative models. We first describe the basic concept of GSC and its difference from existing semantic communications, and then introduce a general framework of GSC, followed by two case studies to verify the advantages of GSC in AGI-driven applications. Finally, open challenges and new research directions are discussed to stimulate this line of research and pave the way for practical applications. 

**Abstract (ZH)**: 基于人工智能的语义通信利用人工智能技术从数据中提取语义信息以实现高效传输，从而显著降低通信成本。随着通向通用人工智能（AGI）的演进，AGI服务的需求增长为语义通信带来了新的挑战。为此，我们提出了一种新的AGI驱动通信范式，称为生成性语义通信（GSC），并利用诸如基础模型和生成模型等先进人工智能技术。首先描述了GSC的基本概念及其与现有语义通信的区别，然后介绍了一般框架，并通过两个案例研究验证了GSC在AGI驱动应用中的优势。最后，讨论了开放性挑战和新的研究方向，以促进这一研究线的发展并为实际应用铺平道路。 

---
# A Framework for Benchmarking and Aligning Task-Planning Safety in LLM-Based Embodied Agents 

**Title (ZH)**: 基于大语言模型的物理体代理任务规划安全性基准评估与对齐框架 

**Authors**: Yuting Huang, Leilei Ding, Zhipeng Tang, Tianfu Wang, Xinrui Lin, Wuyang Zhang, Mingxiao Ma, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14650)  

**Abstract**: Large Language Models (LLMs) exhibit substantial promise in enhancing task-planning capabilities within embodied agents due to their advanced reasoning and comprehension. However, the systemic safety of these agents remains an underexplored frontier. In this study, we present Safe-BeAl, an integrated framework for the measurement (SafePlan-Bench) and alignment (Safe-Align) of LLM-based embodied agents' behaviors. SafePlan-Bench establishes a comprehensive benchmark for evaluating task-planning safety, encompassing 2,027 daily tasks and corresponding environments distributed across 8 distinct hazard categories (e.g., Fire Hazard). Our empirical analysis reveals that even in the absence of adversarial inputs or malicious intent, LLM-based agents can exhibit unsafe behaviors. To mitigate these hazards, we propose Safe-Align, a method designed to integrate physical-world safety knowledge into LLM-based embodied agents while maintaining task-specific performance. Experiments across a variety of settings demonstrate that Safe-BeAl provides comprehensive safety validation, improving safety by 8.55 - 15.22%, compared to embodied agents based on GPT-4, while ensuring successful task completion. 

**Abstract (ZH)**: Safe-BeAl：一种LLM驱动的可信赖体化智能体的衡量与对齐框架 

---
# A Knowledge-Informed Deep Learning Paradigm for Generalizable and Stability-Optimized Car-Following Models 

**Title (ZH)**: 知识导向的深度学习范式以实现通用性和稳定性优化的跟随车辆模型 

**Authors**: Chengming Wang, Dongyao Jia, Wei Wang, Dong Ngoduy, Bei Peng, Jianping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14241)  

**Abstract**: Car-following models (CFMs) are fundamental to traffic flow analysis and autonomous driving. Although calibrated physics-based and trained data-driven CFMs can replicate human driving behavior, their reliance on specific datasets limits generalization across diverse scenarios and reduces reliability in real-world deployment. Moreover, these models typically focus on behavioral fidelity and do not support the explicit optimization of local and string stability, which are increasingly important for the safe and efficient operation of autonomous vehicles (AVs). To address these limitations, we propose a Knowledge-Informed Deep Learning (KIDL) paradigm that distills the generalization capabilities of pre-trained Large Language Models (LLMs) into a lightweight and stability-aware neural architecture. LLMs are used to extract fundamental car-following knowledge beyond dataset-specific patterns, and this knowledge is transferred to a reliable, tractable, and computationally efficient model through knowledge distillation. KIDL also incorporates stability constraints directly into its training objective, ensuring that the resulting model not only emulates human-like behavior but also satisfies the local and string stability requirements essential for real-world AV deployment. We evaluate KIDL on the real-world NGSIM and HighD datasets, comparing its performance with representative physics-based, data-driven, and hybrid CFMs. Both empirical and theoretical results consistently demonstrate KIDL's superior behavioral generalization and traffic flow stability, offering a robust and scalable solution for next-generation traffic systems. 

**Abstract (ZH)**: 基于知识指导的深度学习（KIDL）框架：一种适用于自动驾驶的车车间跟随模型 

---
# InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners 

**Title (ZH)**: InfiGUI-R1: 从反应性行为者到反思性推理者的大规模多模态GUI代理的推进 

**Authors**: Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14239)  

**Abstract**: Multimodal Large Language Models (MLLMs) have powered Graphical User Interface (GUI) Agents, showing promise in automating tasks on computing devices. Recent works have begun exploring reasoning in GUI tasks with encouraging results. However, many current approaches rely on manually designed reasoning templates, which may result in reasoning that is not sufficiently robust and adaptive for complex GUI environments. Meanwhile, some existing agents continue to operate as Reactive Actors, relying primarily on implicit reasoning that may lack sufficient depth for GUI tasks demanding planning and error recovery. We argue that advancing these agents requires a shift from reactive acting towards acting based on deliberate reasoning. To facilitate this transformation, we introduce InfiGUI-R1, an MLLM-based GUI agent developed through our Actor2Reasoner framework, a reasoning-centric, two-stage training approach designed to progressively evolve agents from Reactive Actors to Deliberative Reasoners. The first stage, Reasoning Injection, focuses on establishing a basic reasoner. We employ Spatial Reasoning Distillation to transfer cross-modal spatial reasoning capabilities from teacher models to MLLMs through trajectories with explicit reasoning steps, enabling models to integrate GUI visual-spatial information with logical reasoning before action generation. The second stage, Deliberation Enhancement, refines the basic reasoner into a deliberative one using Reinforcement Learning. This stage introduces two approaches: Sub-goal Guidance, which rewards models for generating accurate intermediate sub-goals, and Error Recovery Scenario Construction, which creates failure-and-recovery training scenarios from identified prone-to-error steps. Experimental results show InfiGUI-R1 achieves strong performance in GUI grounding and trajectory tasks. Resources at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）驱动的图形用户界面（GUI）代理展示了在计算设备上自动化任务的潜力。近期研究表明，通过图形用户界面任务推理取得了令人鼓舞的结果。然而，许多现有方法依赖于手工设计的推理模板，这可能导致在复杂GUI环境中推理不够 robust 和适应性强。同时，一些现有代理仍作为反应式执行者运作，主要依赖隐式的推理，这可能不足以应对需求规划和错误恢复的GUI任务。我们提出，要推进这些代理的发展，需要从反应式执行转向基于深思熟虑推理的执行。为促进这一转变，我们引入了InfiGUI-R1，这是一种通过我们提出的Actor2Reasoner框架开发的MLLM驱动的GUI代理，该框架是一种以推理为中心的两阶段训练方法，旨在逐步将代理从反应式执行者进化为深思熟虑的推理者。第一阶段，推理注入，侧重于建立基本推理器。我们采用空间推理蒸馏，通过显式推理步骤的轨迹将跨模态空间推理能力从教师模型转移到MLLM，使模型能够在动作生成之前将GUI视觉-空间信息与逻辑推理结合起来。第二阶段，推理增强，使用强化学习将基础推理器优化为深思熟虑的推理器。该阶段引入了两种方法：子目标指导，奖励模型生成准确的中间子目标，以及错误恢复情景构建，从容易出错的步骤中创建失败和恢复的训练情景。实验结果显示，InfiGUI-R1在GUI定位和轨迹任务中表现出色。了解更多内容请访问：[此处链接] 

---
# Rethinking the Potential of Multimodality in Collaborative Problem Solving Diagnosis with Large Language Models 

**Title (ZH)**: 重新思考多模态在与大规模语言模型协作解决问题诊断中的潜力 

**Authors**: K. Wong, B. Wu, S. Bulathwela, M. Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2504.15093)  

**Abstract**: Detecting collaborative and problem-solving behaviours from digital traces to interpret students' collaborative problem solving (CPS) competency is a long-term goal in the Artificial Intelligence in Education (AIEd) field. Although multimodal data and advanced models are argued to have the potential to detect complex CPS behaviours, empirical evidence on their value remains limited with some contrasting evidence. In this study, we investigated the potential of multimodal data to improve model performance in diagnosing 78 secondary school students' CPS subskills and indicators in authentic educational settings. In particular, text embeddings from verbal data and acoustic embeddings from audio data were used in a multimodal classification model for CPS diagnosis. Both unimodal and multimodal transformer-based models outperformed traditional models in detecting CPS classes. Although the inclusion of multimodality did not improve the performance of traditional unimodal models, its integration into transformer-based models demonstrated improved performance for diagnosing social-cognitive CPS classes compared to unimodal transformer-based models. Based on the results, the paper argues that multimodality and the selection of a particular modelling technique should not be taken for granted to achieve the best performance in the automated detection of every CPS subskill and indicator. Rather, their value is limited to certain types of CPS indicators, affected by the complexity of the labels, and dependent on the composition of indicators in the dataset. We conclude the paper by discussing the required nuance when considering the value of LLMs and multimodality in automated CPS diagnosis, highlighting the need for human-AI complementarity, and proposing the exploration of relevant model architectures and techniques to improve CPS diagnosis in authentic educational contexts. 

**Abstract (ZH)**: 从数字痕迹检测协作和解决问题行为以解释学生协作问题解决能力：人工智能在教育领域的长期目标。尽管多模态数据和先进模型被认为有可能检测复杂的协作问题解决行为，但关于它们价值的经验证据仍然有限，且存在一些矛盾的证据。在这项研究中，我们调查了多模态数据提高模型性能以诊断78名中学生在真实教育环境中的协作问题解决亚技能和指标的潜力。特别地，我们将来自口头数据的文字嵌入和来自音频数据的声音嵌入用于协作问题解决诊断的多模态分类模型。无论是单模态还是多模态的Transformer模型都优于传统模型，以检测协作问题解决类别。尽管多模态的纳入并未提高传统单模态模型的表现，但在Transformer模型中纳入多模态显示了相较于单模态Transformer模型，在社交认知协作问题解决类别的诊断中性能有所提升。基于研究结果，本文认为多模态和特定建模技术的选择不应被默认为在自动化检测每项协作问题解决亚技能和指标时达到最佳性能的保证。相反，它们的价值仅适用于某些类型的协作问题解决指标，受标签复杂性的影响，并依赖于数据集中指标的组成。本文讨论了在自动化协作问题解决诊断中考虑LLM和多模态价值所需的细微差别，强调人机互补的重要性，并提出探索相关模型架构和技术以在真实教育环境中改进协作问题解决诊断的建议。 

---
# VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform 

**Title (ZH)**: VLM作为政策：短视频平台内容规范化框架 

**Authors**: Xingyu Lu, Tianke Zhang, Chang Meng, Xiaobei Wang, Jinpeng Wang, YiFan Zhang, Shisong Tang, Changyi Liu, Haojie Ding, Kaiyu Jiang, Kaiyu Tang, Bin Wen, Hai-Tao Zheng, Fan Yang, Tingting Gao, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14904)  

**Abstract**: Exponentially growing short video platforms (SVPs) face significant challenges in moderating content detrimental to users' mental health, particularly for minors. The dissemination of such content on SVPs can lead to catastrophic societal consequences. Although substantial efforts have been dedicated to moderating such content, existing methods suffer from critical limitations: (1) Manual review is prone to human bias and incurs high operational costs. (2) Automated methods, though efficient, lack nuanced content understanding, resulting in lower accuracy. (3) Industrial moderation regulations struggle to adapt to rapidly evolving trends due to long update cycles. In this paper, we annotate the first SVP content moderation benchmark with authentic user/reviewer feedback to fill the absence of benchmark in this field. Then we evaluate various methods on the benchmark to verify the existence of the aforementioned limitations. We further propose our common-law content moderation framework named KuaiMod to address these challenges. KuaiMod consists of three components: training data construction, offline adaptation, and online deployment & refinement. Leveraging large vision language model (VLM) and Chain-of-Thought (CoT) reasoning, KuaiMod adequately models video toxicity based on sparse user feedback and fosters dynamic moderation policy with rapid update speed and high accuracy. Offline experiments and large-scale online A/B test demonstrates the superiority of KuaiMod: KuaiMod achieves the best moderation performance on our benchmark. The deployment of KuaiMod reduces the user reporting rate by 20% and its application in video recommendation increases both Daily Active User (DAU) and APP Usage Time (AUT) on several Kuaishou scenarios. We have open-sourced our benchmark at this https URL. 

**Abstract (ZH)**: 指数级增长的短视频平台（SVPs）在 moderating 对用户心理健康有害的内容方面面临重大挑战，尤其是针对未成年人。SVPs 上such内容的传播可能会导致严重的社会后果。尽管已投入大量努力来 moderating 这类内容，但现有方法存在重大局限性：（1）人工审核容易受到人类偏见的影响，并导致高运营成本。 （2）自动化方法虽然高效，但缺乏对内容的细微理解，导致准确性较低。 （3）工业级内容审核规章制度难以适应快速变化的趋势，因为更新周期较长。本文中，我们标注了第一个包含真实用户/审核员反馈的SVP内容审核基准，以填补该领域的基准空缺。然后我们在基准中评估各种方法，以验证上述局限性存在的证据。我们进一步提出了名为KuaiMod的共同法内容审核框架，以应对这些挑战。KuaiMod由三个组件组成：训练数据构建、离线适应和在线部署与优化。利用大规模视觉语言模型（VLM）和链式思考（CoT）推理，KuaiMod能够基于稀疏用户反馈建模视频毒性，并以快速更新速度和高准确性促进动态审核策略。离线实验和大规模在线A/B测试表明KuaiMod的优势：KuaiMod在我们的基准测试中实现了最佳的审核性能。部署KuaiMod将用户举报率降低了20%，并在多个Kuaishou场景中，其应用于视频推荐增加了日活跃用户（DAU）和应用使用时间（AUT）。我们已将基准公开于此 https URL。 

---
# Exploring Collaborative GenAI Agents in Synchronous Group Settings: Eliciting Team Perceptions and Design Considerations for the Future of Work 

**Title (ZH)**: 探索同步群组环境中协作生成式AI代理：激发团队感知及未来工作设计考虑 

**Authors**: Janet G. Johnson, Macarena Peralta, Mansanjam Kaur, Ruijie Sophia Huang, Sheng Zhao, Ruijia Guan, Shwetha Rajaram, Michael Nebeling  

**Link**: [PDF](https://arxiv.org/pdf/2504.14779)  

**Abstract**: While generative artificial intelligence (GenAI) is finding increased adoption in workplaces, current tools are primarily designed for individual use. Prior work established the potential for these tools to enhance personal creativity and productivity towards shared goals; however, we don't know yet how to best take into account the nuances of group work and team dynamics when deploying GenAI in work settings. In this paper, we investigate the potential of collaborative GenAI agents to augment teamwork in synchronous group settings through an exploratory study that engaged 25 professionals across 6 teams in speculative design workshops and individual follow-up interviews. Our workshops included a mixed reality provotype to simulate embodied collaborative GenAI agents capable of actively participating in group discussions. Our findings suggest that, if designed well, collaborative GenAI agents offer valuable opportunities to enhance team problem-solving by challenging groupthink, bridging communication gaps, and reducing social friction. However, teams' willingness to integrate GenAI agents depended on its perceived fit across a number of individual, team, and organizational factors. We outline the key design tensions around agent representation, social prominence, and engagement and highlight the opportunities spatial and immersive technologies could offer to modulate GenAI influence on team outcomes and strike a balance between augmentation and agency. 

**Abstract (ZH)**: 协作生成人工智能代理在同步团队工作中的潜力探究：基于25名专业人士的探索性研究 

---
# Surrogate Fitness Metrics for Interpretable Reinforcement Learning 

**Title (ZH)**: 可解释强化学习的代理 fitness 度量标准 

**Authors**: Philipp Altmann, Céline Davignon, Maximilian Zorn, Fabian Ritz, Claudia Linnhoff-Popien, Thomas Gabor  

**Link**: [PDF](https://arxiv.org/pdf/2504.14645)  

**Abstract**: We employ an evolutionary optimization framework that perturbs initial states to generate informative and diverse policy demonstrations. A joint surrogate fitness function guides the optimization by combining local diversity, behavioral certainty, and global population diversity. To assess demonstration quality, we apply a set of evaluation metrics, including the reward-based optimality gap, fidelity interquartile means (IQMs), fitness composition analysis, and trajectory visualizations. Hyperparameter sensitivity is also examined to better understand the dynamics of trajectory optimization. Our findings demonstrate that optimizing trajectory selection via surrogate fitness metrics significantly improves interpretability of RL policies in both discrete and continuous environments. In gridworld domains, evaluations reveal significantly enhanced demonstration fidelities compared to random and ablated baselines. In continuous control, the proposed framework offers valuable insights, particularly for early-stage policies, while fidelity-based optimization proves more effective for mature policies. By refining and systematically analyzing surrogate fitness functions, this study advances the interpretability of RL models. The proposed improvements provide deeper insights into RL decision-making, benefiting applications in safety-critical and explainability-focused domains. 

**Abstract (ZH)**: 我们采用一种进化优化框架，通过扰动初始状态生成具有信息性和多样性的策略演示。联合代理适应性函数通过结合局部多样性、行为确定性和全局种群多样性来指导优化。为了评估演示质量，我们应用了包括基于奖励的最优性差距、保真度四分位均值（IQMs）、适应性组成分析和轨迹可视化在内的一系列评估指标。我们还研究了超参数敏感性，以更好地理解轨迹优化的动力学。研究结果表明，通过代理适应性度量优化轨迹选择显著提高了离散和连续环境中的RL策略的可解释性。在格网世界领域，评估结果显示与随机和删减基准相比，演示保真度有显著提升。在连续控制中，所提出的框架对于早期策略提供了有价值的见解，而基于保真度的优化对于成熟策略更为有效。通过细化和系统分析代理适应性函数，本研究推进了RL模型的可解释性。提出的改进为安全关键和可解释性导向领域的RL决策提供了更深入的洞见。 

---
# K2MUSE: A human lower limb multimodal dataset under diverse conditions for facilitating rehabilitation robotics 

**Title (ZH)**: K2MUSE：在多样条件下的人类下肢多模态数据集，以促进康复 robotics 的发展 

**Authors**: Jiwei Li, Bi Zhang, Xiaowei Tan, Wanxin Chen, Zhaoyuan Liu, Juanjuan Zhang, Weiguang Huo, Jian Huang, Lianqing Liu, Xingang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.14602)  

**Abstract**: The natural interaction and control performance of lower limb rehabilitation robots are closely linked to biomechanical information from various human locomotion activities. Multidimensional human motion data significantly deepen the understanding of the complex mechanisms governing neuromuscular alterations, thereby facilitating the development and application of rehabilitation robots in multifaceted real-world environments. However, currently available lower limb datasets are inadequate for supplying the essential multimodal data and large-scale gait samples necessary for effective data-driven approaches, and they neglect the significant effects of acquisition interference in real this http URL fill this gap, we present the K2MUSE dataset, which includes a comprehensive collection of multimodal data, comprising kinematic, kinetic, amplitude-mode ultrasound (AUS), and surface electromyography (sEMG) measurements. The proposed dataset includes lower limb multimodal data from 30 able-bodied participants walking under different inclines (0$^\circ$, $\pm$5$^\circ$, and $\pm$10$^\circ$), various speeds (0.5 m/s, 1.0 m/s, and 1.5 m/s), and different nonideal acquisition conditions (muscle fatigue, electrode shifts, and inter-day differences). The kinematic and ground reaction force data were collected via a Vicon motion capture system and an instrumented treadmill with embedded force plates, whereas the sEMG and AUS data were synchronously recorded for thirteen muscles on the bilateral lower limbs. This dataset offers a new resource for designing control frameworks for rehabilitation robots and conducting biomechanical analyses of lower limb locomotion. The dataset is available at this https URL. 

**Abstract (ZH)**: 下肢康复机器人的人机自然交互与控制性能与各种人类运动活动的生物力学信息密切相关。多维度的人体运动数据大大加深了对调控神经肌肉转换复杂机制的理解，从而促进了康复机器人在多场景实际环境中的开发与应用。然而，目前可用的下肢数据集不足以提供有效数据驱动方法所需的各种模态数据和大规模步态样本，并且忽略了实际获取干扰的显著影响。为填补这一空白，我们提出了K2MUSE数据集，该数据集包含全面的多模态数据，包括运动学、动力学、幅度模式超声波（AUS）和表面肌电图（sEMG）测量。所提出的数据集包括30名健康受试者在不同坡度（0°，±5°，±10°）、不同速度（0.5 m/s，1.0 m/s，1.5 m/s）和不同非理想采集条件下（肌肉疲劳、电极位移和日间差异）的下肢多模态数据。运动学和地面反作用力数据通过Vicon运动捕捉系统和内置力板的仪器跑步机收集，而sEMG和AUS数据同步记录了双侧下肢的十三块肌肉。该数据集为设计康复机器人控制框架和进行下肢单位运动的生物力学分析提供了新的资源。数据集可从此链接获得。 

---
# Hydra: An Agentic Reasoning Approach for Enhancing Adversarial Robustness and Mitigating Hallucinations in Vision-Language Models 

**Title (ZH)**: Hydra: 一种增强视觉-语言模型对抗鲁棒性并缓解幻觉的机构性推理方法 

**Authors**: Chung-En, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14395)  

**Abstract**: To develop trustworthy Vision-Language Models (VLMs), it is essential to address adversarial robustness and hallucination mitigation, both of which impact factual accuracy in high-stakes applications such as defense and healthcare. Existing methods primarily focus on either adversarial defense or hallucination post-hoc correction, leaving a gap in unified robustness strategies. We introduce \textbf{Hydra}, an adaptive agentic framework that enhances plug-in VLMs through iterative reasoning, structured critiques, and cross-model verification, improving both resilience to adversarial perturbations and intrinsic model errors. Hydra employs an Action-Critique Loop, where it retrieves and critiques visual information, leveraging Chain-of-Thought (CoT) and In-Context Learning (ICL) techniques to refine outputs dynamically. Unlike static post-hoc correction methods, Hydra adapts to both adversarial manipulations and intrinsic model errors, making it robust to malicious perturbations and hallucination-related inaccuracies. We evaluate Hydra on four VLMs, three hallucination benchmarks, two adversarial attack strategies, and two adversarial defense methods, assessing performance on both clean and adversarial inputs. Results show that Hydra surpasses plug-in VLMs and state-of-the-art (SOTA) dehallucination methods, even without explicit adversarial defenses, demonstrating enhanced robustness and factual consistency. By bridging adversarial resistance and hallucination mitigation, Hydra provides a scalable, training-free solution for improving the reliability of VLMs in real-world applications. 

**Abstract (ZH)**: 开发可信的视觉-语言模型 (VLMs) 需要解决对抗鲁棒性和幻觉缓解问题，这两者在高风险应用如国防和医疗保健中影响事实准确性。现有方法主要集中在对抗防御或事后幻觉修正上，留下了统一鲁棒性策略的缺口。我们引入了**Hydra**，一个自适应代理框架，通过迭代推理、结构化批判和跨模型验证增强插件VLMs，提高其对抗扰动的抗性和内在模型错误。Hydra 使用行动-批判循环，在此过程中检索和批判视觉信息，并利用链式思考（CoT）和上下文相关学习（ICL）技术动态优化输出。与静态事后修正方法不同，Hydra 能够应对对抗操作和内在模型错误，使其对恶意扰动和幻觉相关不准确具有鲁棒性。我们对四种VLMs、三种幻觉基准、两种对抗攻击策略和两种对抗防御方法进行了评估，评估其在干净和对抗输入上的性能。结果显示，Hydra 超过了插件VLMs和最新的去幻觉方法，甚至在没有明确的对抗防御措施的情况下也表现出更强的鲁棒性和事实一致性。通过结合对抗抵抗和幻觉缓解，Hydra 提供了一个可扩展的、无需训练的解决方案，以提高视觉-语言模型在实际应用中的可靠性。 

---
# System of Agentic AI for the Discovery of Metal-Organic Frameworks 

**Title (ZH)**: 代理人工智能系统用于金属有机框架的发现 

**Authors**: Theo Jaffrelot Inizan, Sherry Yang, Aaron Kaplan, Yen-hsu Lin, Jian Yin, Saber Mirzaei, Mona Abdelgaid, Ali H. Alawadhi, KwangHwan Cho, Zhiling Zheng, Ekin Dogus Cubuk, Christian Borgs, Jennifer T. Chayes, Kristin A. Persson, Omar M. Yaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14110)  

**Abstract**: Generative models and machine learning promise accelerated material discovery in MOFs for CO2 capture and water harvesting but face significant challenges navigating vast chemical spaces while ensuring synthetizability. Here, we present MOFGen, a system of Agentic AI comprising interconnected agents: a large language model that proposes novel MOF compositions, a diffusion model that generates crystal structures, quantum mechanical agents that optimize and filter candidates, and synthetic-feasibility agents guided by expert rules and machine learning. Trained on all experimentally reported MOFs and computational databases, MOFGen generated hundreds of thousands of novel MOF structures and synthesizable organic linkers. Our methodology was validated through high-throughput experiments and the successful synthesis of five "AI-dreamt" MOFs, representing a major step toward automated synthesizable material discovery. 

**Abstract (ZH)**: 基于Agentic AI的MOFGen系统：加速CO2捕获和水收集用MOFs材料发现但仍面临巨大挑战 

---
# From job titles to jawlines: Using context voids to study generative AI systems 

**Title (ZH)**: 从职位名称到下巴线条：利用上下文空白研究生成式AI系统 

**Authors**: Shahan Ali Memon, Soham De, Sungha Kang, Riyan Mujtaba, Bedoor AlShebli, Katie Davis, Jaime Snyder, Jevin D. West  

**Link**: [PDF](https://arxiv.org/pdf/2504.13947)  

**Abstract**: In this paper, we introduce a speculative design methodology for studying the behavior of generative AI systems, framing design as a mode of inquiry. We propose bridging seemingly unrelated domains to generate intentional context voids, using these tasks as probes to elicit AI model behavior. We demonstrate this through a case study: probing the ChatGPT system (GPT-4 and DALL-E) to generate headshots from professional Curricula Vitae (CVs). In contrast to traditional ways, our approach assesses system behavior under conditions of radical uncertainty -- when forced to invent entire swaths of missing context -- revealing subtle stereotypes and value-laden assumptions. We qualitatively analyze how the system interprets identity and competence markers from CVs, translating them into visual portraits despite the missing context (i.e. physical descriptors). We show that within this context void, the AI system generates biased representations, potentially relying on stereotypical associations or blatant hallucinations. 

**Abstract (ZH)**: 本研究引入了一种 speculate 设计方法论以研究生成式 AI 系统的行为，将设计视为一种探究模式。我们提出将看似无关的领域进行对接以生成有意图的背景空白，并使用这些任务作为探针来激发 AI 模型的行为。我们通过案例研究进行了演示，对 ChatGPT 系统（GPT-4 和 DALL-E）进行探针测试，从专业简历生成头像。与传统方法不同，我们的方法在极端不确定性条件下评估系统行为——当被迫发明大量缺失的背景时——揭示了细微的刻板印象和价值取向的假设。我们定性分析了系统如何理解和转化简历中的身份和能力标志，并在缺乏背景下将其转化为视觉肖像（即，缺乏身体描述）。在这一背景空白中，AI 系统生成了有偏见的表示，可能依赖于刻板印象联想或显性的虚拟构建。 

---
# The Human Robot Social Interaction (HSRI) Dataset: Benchmarking Foundational Models' Social Reasoning 

**Title (ZH)**: 人类机器人社会交互数据集：基础知识模型的社会推理基准测试 

**Authors**: Dong Won Lee, Yubin Kim, Denison Guvenoz, Sooyeon Jeong, Parker Malachowsky, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.13898)  

**Abstract**: Our work aims to advance the social reasoning of embodied artificial intelligence (AI) agents in real-world social interactions. Recently, language models (LMs) and foundational models (FMs) are being utilized as automatic evaluators of human-AI interactions with the goal of eventually being used to improve the policy of the AI agent. To enable further research in this direction, we introduce a large-scale real-world Human Robot Social Interaction (HSRI) Dataset to benchmark the capabilities of LMs and FMs to identify and reason about social interactions, specifically with regard to robot social errors and competencies . Our dataset consists of 400 real-world human social robot interaction videos and over 10K annotations, detailing the robot's social errors, competencies, rationale, and corrective actions, capturing unique aspects of human-AI interaction only present in real-world interactions. To further assess AI models' ability to reason about social interactions, we propose eight new benchmark tasks for evaluating centered around whether AI models can (1) evaluate social interactions via detecting social errors and competencies, (2) identify the explanatory factors associated to errors and competencies, (3) understand the flow of real-world social interactions, and (4) provide reasons and corrective actions for social errors. Human studies and experiments with modern LMs and FMs reveal that current models struggle with these tasks, demonstrating that our dataset and benchmark provides a step forward towards socially intelligent AI. 

**Abstract (ZH)**: 我们的工作旨在推动具身人工智能（AI）代理在现实世界社会互动中的社会推理能力。最近，语言模型（LMs）和基础模型（FMs）被用作人类-AI互动的自动评估器，旨在最终用于改进AI代理的政策。为了推动这一方向的进一步研究，我们引入了一个大规模的现实世界人类机器人社会互动（HSRI）数据集，用以评估LMs和FMs识别和推理社会互动的能力，特别是与机器人的社会错误和能力相关方面。该数据集包括400个真实世界的真人与社会机器人互动视频和超过10,000个注释，详细记录了机器人的社会错误、能力、推理和纠正措施，捕获了仅在现实世界互动中才存在的独特人类-AI交互方面。为了进一步评估AI模型在社会互动中的推理能力，我们提出了八个新的基准任务，围绕AI模型能否（1）通过检测社会错误和能力来评估社会互动，（2）识别与错误和能力相关的解释性因素，（3）理解现实世界社会互动的流程，（4）为社会错误提供理由和纠正措施。现代人类研究和实验表明，当前模型在这些任务中表现不佳，证明了我们的数据集和基准对于迈向社会智能AI的重要性。 

---
