# DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References 

**Title (ZH)**: DexTrack: 从人类参考向通用化灵巧 manipulation 控制的神经跟踪控制研究 

**Authors**: Xueyi Liu, Jianibieke Adalibieke, Qianwei Han, Yuzhe Qin, Li Yi  

**Link**: [PDF](https://arxiv.org/pdf/2502.09614)  

**Abstract**: We address the challenge of developing a generalizable neural tracking controller for dexterous manipulation from human references. This controller aims to manage a dexterous robot hand to manipulate diverse objects for various purposes defined by kinematic human-object interactions. Developing such a controller is complicated by the intricate contact dynamics of dexterous manipulation and the need for adaptivity, generalizability, and robustness. Current reinforcement learning and trajectory optimization methods often fall short due to their dependence on task-specific rewards or precise system models. We introduce an approach that curates large-scale successful robot tracking demonstrations, comprising pairs of human references and robot actions, to train a neural controller. Utilizing a data flywheel, we iteratively enhance the controller's performance, as well as the number and quality of successful tracking demonstrations. We exploit available tracking demonstrations and carefully integrate reinforcement learning and imitation learning to boost the controller's performance in dynamic environments. At the same time, to obtain high-quality tracking demonstrations, we individually optimize per-trajectory tracking by leveraging the learned tracking controller in a homotopy optimization method. The homotopy optimization, mimicking chain-of-thought, aids in solving challenging trajectory tracking problems to increase demonstration diversity. We showcase our success by training a generalizable neural controller and evaluating it in both simulation and real world. Our method achieves over a 10% improvement in success rates compared to leading baselines. The project website with animated results is available at this https URL. 

**Abstract (ZH)**: 我们解决从人类参考中开发可泛化的灵巧操作神经跟踪控制器的挑战。该控制器旨在管理灵巧的机器人手进行由运动学人类-物体交互定义的各种目的的多样化物体操作。由于灵巧操作复杂的接触动力学以及适应性、可泛化性和鲁棒性的需求，开发这样的控制器极具挑战性。现有的强化学习和轨迹优化方法往往因为依赖于特定任务的奖励或精确的系统模型而难以实现。我们提出了一种方法，从包含人类参考和机器人动作配对的大量成功机器人跟踪演示中精挑细选，用于训练神经控制器。利用数据飞轮，我们逐步提升控制器的性能，以及成功跟踪演示的数量和质量。我们利用可用的跟踪演示，并仔细整合强化学习和模仿学习，以增强控制器在动态环境中的性能。同时，为了获得高质量的跟踪演示，我们通过在同伦优化方法中利用学习到的跟踪控制器对每个轨迹的跟踪进行单独优化。同伦优化，类似于链式思考过程，有助于解决复杂的轨迹跟踪问题，从而增加演示的多样性。我们通过在仿真和实际环境中培训可泛化的神经控制器并进行评估展示了我们的成功。与领先基准相比，我们的方法实现了成功率超过10%的提升。项目网站及动画结果详见此<https URL>。 

---
# Variable Stiffness for Robust Locomotion through Reinforcement Learning 

**Title (ZH)**: 基于强化学习的鲁棒移动的可变刚度 

**Authors**: Dario Spoljaric, Yashuai Yan, Dongheui Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.09436)  

**Abstract**: Reinforcement-learned locomotion enables legged robots to perform highly dynamic motions but often accompanies time-consuming manual tuning of joint stiffness. This paper introduces a novel control paradigm that integrates variable stiffness into the action space alongside joint positions, enabling grouped stiffness control such as per-joint stiffness (PJS), per-leg stiffness (PLS) and hybrid joint-leg stiffness (HJLS). We show that variable stiffness policies, with grouping in per-leg stiffness (PLS), outperform position-based control in velocity tracking and push recovery. In contrast, HJLS excels in energy efficiency. Furthermore, our method showcases robust walking behaviour on diverse outdoor terrains by sim-to-real transfer, although the policy is sorely trained on a flat floor. Our approach simplifies design by eliminating per-joint stiffness tuning while keeping competitive results with various metrics. 

**Abstract (ZH)**: 基于强化学习的可变刚度腿部机器人动态运动控制 

---
# Generalizable Reinforcement Learning with Biologically Inspired Hyperdimensional Occupancy Grid Maps for Exploration and Goal-Directed Path Planning 

**Title (ZH)**: 基于生物启发的高维占用网格图的可泛化强化学习及其在探索与目标导向路径规划中的应用 

**Authors**: Shay Snyder, Ryan Shea, Andrew Capodieci, David Gorsich, Maryam Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2502.09393)  

**Abstract**: Real-time autonomous systems utilize multi-layer computational frameworks to perform critical tasks such as perception, goal finding, and path planning. Traditional methods implement perception using occupancy grid mapping (OGM), segmenting the environment into discretized cells with probabilistic information. This classical approach is well-established and provides a structured input for downstream processes like goal finding and path planning algorithms. Recent approaches leverage a biologically inspired mathematical framework known as vector symbolic architectures (VSA), commonly known as hyperdimensional computing, to perform probabilistic OGM in hyperdimensional space. This approach, VSA-OGM, provides native compatibility with spiking neural networks, positioning VSA-OGM as a potential neuromorphic alternative to conventional OGM. However, for large-scale integration, it is essential to assess the performance implications of VSA-OGM on downstream tasks compared to established OGM methods. This study examines the efficacy of VSA-OGM against a traditional OGM approach, Bayesian Hilbert Maps (BHM), within reinforcement learning based goal finding and path planning frameworks, across a controlled exploration environment and an autonomous driving scenario inspired by the F1-Tenth challenge. Our results demonstrate that VSA-OGM maintains comparable learning performance across single and multi-scenario training configurations while improving performance on unseen environments by approximately 47%. These findings highlight the increased generalizability of policy networks trained with VSA-OGM over BHM, reinforcing its potential for real-world deployment in diverse environments. 

**Abstract (ZH)**: 实时自主系统利用多层计算框架执行关键任务，如感知、目标定位和路径规划。传统方法使用占用网格映射（OGM）进行感知，将环境分割为具有概率信息的离散单元格。这一经典方法已被广泛认可，并为下游处理如目标定位和路径规划算法提供结构化的输入。近期的方法利用一种生物启发的数学框架，即向量符号架构（VSA），通常称为超维计算（Hyperdimensional Computing），在超维空间中执行概率OGM。该方法VSA-OGM与脉冲神经网络有天然兼容性，将VSA-OGM定位为经典OGM的潜在神经形态替代方案。然而，为了实现大规模集成，评估VSA-OGM在下游任务中的性能影响及其与传统OGM方法相比的重要性是必不可少的。本研究在基于强化学习的目标定位和路径规划框架中，对比了VSA-OGM与传统OGM方法（贝叶斯希爾伯特映射BHM）的效能，研究在受控探索环境和基于F1-Tenth挑战的自主驾驶场景中的表现。实验结果显示，VSA-OGM在单场景和多场景训练配置中保持了相似的学习性能，并在未见过的环境中提高了约47%的性能。这些发现突显了使用VSA-OGM训练的策略网络具有更强的通用性，增强了其在不同环境中的实际部署潜力。 

---
# S$^2$-Diffusion: Generalizing from Instance-level to Category-level Skills in Robot Manipulation 

**Title (ZH)**: S$^2$-Diffusion: 从实例级到类别级技能的机器人 manipulation 技能泛化 

**Authors**: Quantao Yang, Michael C. Welle, Danica Kragic, Olov Andersson  

**Link**: [PDF](https://arxiv.org/pdf/2502.09389)  

**Abstract**: Recent advances in skill learning has propelled robot manipulation to new heights by enabling it to learn complex manipulation tasks from a practical number of demonstrations. However, these skills are often limited to the particular action, object, and environment \textit{instances} that are shown in the training data, and have trouble transferring to other instances of the same category. In this work we present an open-vocabulary Spatial-Semantic Diffusion policy (S$^2$-Diffusion) which enables generalization from instance-level training data to category-level, enabling skills to be transferable between instances of the same category. We show that functional aspects of skills can be captured via a promptable semantic module combined with a spatial representation. We further propose leveraging depth estimation networks to allow the use of only a single RGB camera. Our approach is evaluated and compared on a diverse number of robot manipulation tasks, both in simulation and in the real world. Our results show that S$^2$-Diffusion is invariant to changes in category-irrelevant factors as well as enables satisfying performance on other instances within the same category, even if it was not trained on that specific instance. Full videos of all real-world experiments are available in the supplementary material. 

**Abstract (ZH)**: 最近在技能学习方面的进展通过使机器人能够从实际数量的演示中学习复杂操作任务，推动了机器人操作达到新的高度。然而，这些技能往往局限于训练数据中所示的特定操作、对象和环境实例，并且难以转移到同一类别的其他实例。在这项工作中，我们提出了一种开放词汇空间语义扩散策略（S$^2$-Diffusion），使其能够从实例级别的训练数据泛化到类别级别，从而使技能能够在同一类别的其他实例之间进行迁移。我们展示了通过结合空间表示和可提示的语义模块可以捕捉技能的功能方面。我们进一步提出利用深度估计网络，仅使用单个RGB相机即可。该方法在多种机器人操作任务上进行了评估和比较，包括模拟环境和真实世界。我们的结果表明，S$^2$-Diffusion在类别无关因素发生变化时保持不变，并且能够在同一类别的其他实例上实现令人满意的性能，即使它未针对特定实例进行训练。所有真实世界实验的完整视频可在补充材料中找到。 

---
# GEVRM: Goal-Expressive Video Generation Model For Robust Visual Manipulation 

**Title (ZH)**: GEVRM：目标表达型视频生成模型在稳健视觉操控中的应用 

**Authors**: Hongyin Zhang, Pengxiang Ding, Shangke Lyu, Ying Peng, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09268)  

**Abstract**: With the rapid development of embodied artificial intelligence, significant progress has been made in vision-language-action (VLA) models for general robot decision-making. However, the majority of existing VLAs fail to account for the inevitable external perturbations encountered during deployment. These perturbations introduce unforeseen state information to the VLA, resulting in inaccurate actions and consequently, a significant decline in generalization performance. The classic internal model control (IMC) principle demonstrates that a closed-loop system with an internal model that includes external input signals can accurately track the reference input and effectively offset the disturbance. We propose a novel closed-loop VLA method GEVRM that integrates the IMC principle to enhance the robustness of robot visual manipulation. The text-guided video generation model in GEVRM can generate highly expressive future visual planning goals. Simultaneously, we evaluate perturbations by simulating responses, which are called internal embeddings and optimized through prototype contrastive learning. This allows the model to implicitly infer and distinguish perturbations from the external environment. The proposed GEVRM achieves state-of-the-art performance on both standard and perturbed CALVIN benchmarks and shows significant improvements in realistic robot tasks. 

**Abstract (ZH)**: 随着具身人工智能的快速发展，视觉-语言-动作（VLA）模型在通用机器人决策制定方面的进展显著。然而，现有的大多数VLA未能考虑部署过程中不可避免的外部干扰。这些干扰引入了未预见的状态信息，导致动作不准确，从而显著降低了泛化性能。经典的内部模型控制（IMC）原理表明，包含外部输入信号的内部模型闭环系统能够准确跟踪参考输入并有效抵消干扰。我们提出了一种结合IMC原理的新型闭环VLA方法GEVRM，以增强机器人视觉操作的鲁棒性。GEVRM中的文本指导视频生成模型可以生成高度表达性的未来视觉规划目标。同时，我们通过模拟响应评估干扰，这些响应称为内部嵌入，并通过原型对比学习进行优化。这使模型能够隐式推断和区分来自外部环境的干扰。所提出的GEVRM在标准和干扰的CALVIN基准上均实现了最先进的性能，并在现实机器人任务中显示出显著改进。 

---
# OpenBench: A New Benchmark and Baseline for Semantic Navigation in Smart Logistics 

**Title (ZH)**: OpenBench：智能物流中的语义导航新基准和基线 

**Authors**: Junhui Wang, Dongjie Huo, Zehui Xu, Yongliang Shi, Yimin Yan, Yuanxin Wang, Chao Gao, Yan Qiao, Guyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.09238)  

**Abstract**: The increasing demand for efficient last-mile delivery in smart logistics underscores the role of autonomous robots in enhancing operational efficiency and reducing costs. Traditional navigation methods, which depend on high-precision maps, are resource-intensive, while learning-based approaches often struggle with generalization in real-world scenarios. To address these challenges, this work proposes the Openstreetmap-enhanced oPen-air sEmantic Navigation (OPEN) system that combines foundation models with classic algorithms for scalable outdoor navigation. The system uses off-the-shelf OpenStreetMap (OSM) for flexible map representation, thereby eliminating the need for extensive pre-mapping efforts. It also employs Large Language Models (LLMs) to comprehend delivery instructions and Vision-Language Models (VLMs) for global localization, map updates, and house number recognition. To compensate the limitations of existing benchmarks that are inadequate for assessing last-mile delivery, this work introduces a new benchmark specifically designed for outdoor navigation in residential areas, reflecting the real-world challenges faced by autonomous delivery systems. Extensive experiments in simulated and real-world environments demonstrate the proposed system's efficacy in enhancing navigation efficiency and reliability. To facilitate further research, our code and benchmark are publicly available. 

**Abstract (ZH)**: 基于OpenStreetMap增强的户外语义导航（OPEN）系统：面向自主最后英里交付的可扩展导航方法 

---
# A Machine Learning Approach to Sensor Substitution for Non-Prehensile Manipulation 

**Title (ZH)**: 一种用于非抓握操作的传感器替代的机器学习方法 

**Authors**: Idil Ozdamar, Doganay Sirintuna, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2502.09180)  

**Abstract**: Mobile manipulators are increasingly deployed in complex environments, requiring diverse sensors to perceive and interact with their surroundings. However, equipping every robot with every possible sensor is often impractical due to cost and physical constraints. A critical challenge arises when robots with differing sensor capabilities need to collaborate or perform similar tasks. For example, consider a scenario where a mobile manipulator equipped with high-resolution tactile skin is skilled at non-prehensile manipulation tasks like pushing. If this robot needs to be replaced or augmented by a robot lacking such tactile sensing, the learned manipulation policies become inapplicable. This paper addresses the problem of sensor substitution in non-prehensile manipulation. We propose a novel machine learning-based framework that enables a robot with a limited sensor set (e.g., LiDAR or RGB-D camera) to effectively perform tasks previously reliant on a richer sensor suite (e.g., tactile skin). Our approach learns a mapping between the available sensor data and the information provided by the substituted sensor, effectively synthesizing the missing sensory input. Specifically, we demonstrate the efficacy of our framework by training a model to substitute tactile skin data for the task of non-prehensile pushing using a mobile manipulator. We show that a manipulator equipped only with LiDAR or RGB-D can, after training, achieve comparable and sometimes even better pushing performance to a mobile base utilizing direct tactile feedback. 

**Abstract (ZH)**: 移动 manipulator 在复杂环境中部署日益增多，需要多种传感器来感知和交互。然而，为每台机器人配备所有可能的传感器往往由于成本和物理限制而 impractical。当具有不同传感器能力的机器人需要协作或执行类似任务时，一个关键挑战随之而来。例如，考虑一个高分辨率触觉皮肤装备在移动 manipulator 上，擅长非拾取式操作任务如推动的情况。如果这台机器人被其他缺少触觉传感器的机器人替换或增强，学到的操作策略将变得不适用。本论文解决了非拾取式操作中的传感器替代问题。我们提出一种基于机器学习的新颖框架，使具有有限传感器集的机器人（如 LiDAR 或 RGB-D 相机）能够有效执行依赖于更丰富传感器套件的任务（如触觉皮肤）。我们的方法学习可用传感器数据与被替代传感器提供的信息之间的映射，有效合成缺失的感觉输入。具体而言，我们通过训练模型将触觉皮肤数据替换为移动 manipulator 上的非拾取式推动任务，展示了该框架的有效性。结果显示，仅配备 LiDAR 或 RGB-D 的 manipulator 经过训练后，能够与直接触觉反馈的移动基座相媲美，甚至在某些情况下性能更优异。 

---
# Training Trajectory Predictors Without Ground-Truth Data 

**Title (ZH)**: 不使用_ground-truth_数据训练轨迹预测器 

**Authors**: Mikolaj Kliniewski, Jesse Morris, Ian R. Manchester, Viorela Ila  

**Link**: [PDF](https://arxiv.org/pdf/2502.08957)  

**Abstract**: This paper presents a framework capable of accurately and smoothly estimating position, heading, and velocity. Using this high-quality input, we propose a system based on Trajectron++, able to consistently generate precise trajectory predictions. Unlike conventional models that require ground-truth data for training, our approach eliminates this dependency. Our analysis demonstrates that poor quality input leads to noisy and unreliable predictions, which can be detrimental to navigation modules. We evaluate both input data quality and model output to illustrate the impact of input noise. Furthermore, we show that our estimation system enables effective training of trajectory prediction models even with limited data, producing robust predictions across different environments. Accurate estimations are crucial for deploying trajectory prediction models in real-world scenarios, and our system ensures meaningful and reliable results across various application contexts. 

**Abstract (ZH)**: 本文提出了一种能够准确且平滑地估计位置、航向和速度的框架。利用高质量的输入，我们提出了一种基于Trajectron++的系统，能够一致地生成精确的轨迹预测。与需要地面真实数据进行训练的传统模型不同，我们的方法消除了对这种数据的依赖。我们的分析表明，低质量的输入会导致噪声大且不可靠的预测，这对导航模块可能有负面影响。我们评估了输入数据质量和模型输出，以阐明输入噪声的影响。此外，我们展示了我们的估计系统即使在数据有限的情况下也能有效训练轨迹预测模型，产生在不同环境中稳健的预测。准确的估计对于在真实世界场景中部署轨迹预测模型至关重要，我们的系统确保在各种应用上下文中产生有意义且可靠的结果。 

---
# 3D-Grounded Vision-Language Framework for Robotic Task Planning: Automated Prompt Synthesis and Supervised Reasoning 

**Title (ZH)**: 基于3D场景的视觉-语言机器人任务规划框架：自动化提示合成与监督推理 

**Authors**: Guoqin Tang, Qingxuan Jia, Zeyuan Huang, Gang Chen, Ning Ji, Zhipeng Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.08903)  

**Abstract**: Vision-language models (VLMs) have achieved remarkable success in scene understanding and perception tasks, enabling robots to plan and execute actions adaptively in dynamic environments. However, most multimodal large language models lack robust 3D scene localization capabilities, limiting their effectiveness in fine-grained robotic operations. Additionally, challenges such as low recognition accuracy, inefficiency, poor transferability, and reliability hinder their use in precision tasks. To address these limitations, we propose a novel framework that integrates a 2D prompt synthesis module by mapping 2D images to point clouds, and incorporates a small language model (SLM) for supervising VLM outputs. The 2D prompt synthesis module enables VLMs, trained on 2D images and text, to autonomously extract precise 3D spatial information without manual intervention, significantly enhancing 3D scene understanding. Meanwhile, the SLM supervises VLM outputs, mitigating hallucinations and ensuring reliable, executable robotic control code generation. Our framework eliminates the need for retraining in new environments, thereby improving cost efficiency and operational robustness. Experimental results that the proposed framework achieved a 96.0\% Task Success Rate (TSR), outperforming other methods. Ablation studies demonstrated the critical role of both the 2D prompt synthesis module and the output supervision module (which, when removed, caused a 67\% TSR drop). These findings validate the framework's effectiveness in improving 3D recognition, task planning, and robotic task execution. 

**Abstract (ZH)**: 基于视觉-语言模型的新型框架：增强3D场景理解和机器人精细操作能力 

---
# ClipRover: Zero-shot Vision-Language Exploration and Target Discovery by Mobile Robots 

**Title (ZH)**: ClipRover：零样本视觉-语言探索与目标发现的移动机器人方法 

**Authors**: Yuxuan Zhang, Adnan Abdullah, Sanjeev J. Koppal, Md Jahidul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2502.08791)  

**Abstract**: Vision-language navigation (VLN) has emerged as a promising paradigm, enabling mobile robots to perform zero-shot inference and execute tasks without specific pre-programming. However, current systems often separate map exploration and path planning, with exploration relying on inefficient algorithms due to limited (partially observed) environmental information. In this paper, we present a novel navigation pipeline named ''ClipRover'' for simultaneous exploration and target discovery in unknown environments, leveraging the capabilities of a vision-language model named CLIP. Our approach requires only monocular vision and operates without any prior map or knowledge about the target. For comprehensive evaluations, we design the functional prototype of a UGV (unmanned ground vehicle) system named ''Rover Master'', a customized platform for general-purpose VLN tasks. We integrate and deploy the ClipRover pipeline on Rover Master to evaluate its throughput, obstacle avoidance capability, and trajectory performance across various real-world scenarios. Experimental results demonstrate that ClipRover consistently outperforms traditional map traversal algorithms and achieves performance comparable to path-planning methods that depend on prior map and target knowledge. Notably, ClipRover offers real-time active navigation without requiring pre-captured candidate images or pre-built node graphs, addressing key limitations of existing VLN pipelines. 

**Abstract (ZH)**: 基于视觉语言的同步探索与目标发现：ClipRover导航管道 

---
# Moving Matter: Efficient Reconfiguration of Tile Arrangements by a Single Active Robot 

**Title (ZH)**: 移动物质：单个活性机器人高效重构砖块排列的方法 

**Authors**: Aaron T. Becker, Sándor P. Fekete, Jonas Friemel, Ramin Kosfeld, Peter Kramer, Harm Kube, Christian Rieck, Christian Scheffer, Arne Schmidt  

**Link**: [PDF](https://arxiv.org/pdf/2502.09299)  

**Abstract**: We consider the problem of reconfiguring a two-dimensional connected grid arrangement of passive building blocks from a start configuration to a goal configuration, using a single active robot that can move on the tiles, remove individual tiles from a given location and physically move them to a new position by walking on the remaining configuration. The objective is to determine a reconfiguration schedule that minimizes the overall makespan, while ensuring that the tile configuration remains connected. We provide both negative and positive results. (1) We present a generalized version of the problem, parameterized by weighted costs for moving with or without tiles, and show that this is NP-complete. (2) We give a polynomial-time constant-factor approximation algorithm for the case of disjoint start and target bounding boxes. In addition, our approach yields optimal carry distance for 2-scaled instances. 

**Abstract (ZH)**: 二维连接网格Arrange-to-Make-Span问题的重构研究：具有单个活性机器人在不连接状态下移动和重新定位个体单元格的配置优化 

---
# LLM-Driven Augmented Reality Puppeteer: Controller-Free Voice-Commanded Robot Teleoperation 

**Title (ZH)**: LLM驱动的增强现实傀儡师：无控制器语音指令的机器人远程操作 

**Authors**: Yuchong Zhang, Bastian Orthmann, Michael C. Welle, Jonne Van Haastregt, Danica Kragic  

**Link**: [PDF](https://arxiv.org/pdf/2502.09142)  

**Abstract**: The integration of robotics and augmented reality (AR) presents transformative opportunities for advancing human-robot interaction (HRI) by improving usability, intuitiveness, and accessibility. This work introduces a controller-free, LLM-driven voice-commanded AR puppeteering system, enabling users to teleoperate a robot by manipulating its virtual counterpart in real time. By leveraging natural language processing (NLP) and AR technologies, our system -- prototyped using Meta Quest 3 -- eliminates the need for physical controllers, enhancing ease of use while minimizing potential safety risks associated with direct robot operation. A preliminary user demonstration successfully validated the system's functionality, demonstrating its potential for safer, more intuitive, and immersive robotic control. 

**Abstract (ZH)**: 机器人与增强现实（AR）的集成为人类-机器人交互（HRI）的进步提供了变革性机会，通过提高易用性、直观性和可访问性。本工作介绍了一种基于语言模型（LLM）驱动的无需控制器、通过语音命令操控的AR puppeteering系统，使用户能够实时操控机器人并通过操控其虚拟对应物进行远程操作。通过利用自然语言处理（NLP）和AR技术，我们的系统（基于Meta Quest 3原型）消除了物理控制器的需求，增强了易用性并降低了直接操作机器人可能带来的安全风险。初步用户演示成功验证了系统的功能，展示了其在更安全、更直观和更具沉浸感的机器人控制方面的潜力。 

---
# KIMAs: A Configurable Knowledge Integrated Multi-Agent System 

**Title (ZH)**: KIMAs：一种配置可调的知识集成多Agent系统 

**Authors**: Zitao Li, Fei Wei, Yuexiang Xie, Dawei Gao, Weirui Kuang, Zhijian Ma, Bingchen Qian, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2502.09596)  

**Abstract**: Knowledge-intensive conversations supported by large language models (LLMs) have become one of the most popular and helpful applications that can assist people in different aspects. Many current knowledge-intensive applications are centered on retrieval-augmented generation (RAG) techniques. While many open-source RAG frameworks facilitate the development of RAG-based applications, they often fall short in handling practical scenarios complicated by heterogeneous data in topics and formats, conversational context management, and the requirement of low-latency response times. This technical report presents a configurable knowledge integrated multi-agent system, KIMAs, to address these challenges. KIMAs features a flexible and configurable system for integrating diverse knowledge sources with 1) context management and query rewrite mechanisms to improve retrieval accuracy and multi-turn conversational coherency, 2) efficient knowledge routing and retrieval, 3) simple but effective filter and reference generation mechanisms, and 4) optimized parallelizable multi-agent pipeline execution. Our work provides a scalable framework for advancing the deployment of LLMs in real-world settings. To show how KIMAs can help developers build knowledge-intensive applications with different scales and emphases, we demonstrate how we configure the system to three applications already running in practice with reliable performance. 

**Abstract (ZH)**: 大型语言模型支持的知识密集型对话占有越来越多的应用并成为人们各方面的重要辅助工具。当前许多知识密集型应用专注于检索增强生成（RAG）技术。虽然许多开源RAG框架促进了基于RAG的应用开发，但在处理由异构数据主题和格式、对话上下文管理和低延迟响应时间要求带来的复杂场景时常常力不从心。本技术报告提出了一种可配置的知识集成多代理系统（KIMAs），以应对这些挑战。KIMAs具备灵活且可配置的系统，用于整合多元知识来源，包括1）上下文管理与查询重写机制以提高检索准确性和多轮对话一致性，2）高效的知识路由与检索，3）简单而有效的筛选与参考生成机制，以及4）优化的并行可拓展多代理流水线执行。我们的研究提供了在实际场景中推进大型语言模型部署的可扩展框架。为了展示KIMAs如何帮助开发者构建具有不同规模和重点的知识密集型应用，我们展示了如何配置系统以适应三个实际运行且表现可靠的现有应用。 

---
# EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents 

**Title (ZH)**: EmbodiedBench:全方位评估驱动视觉的嵌入式代理多模态大型语言模型 

**Authors**: Rui Yang, Hanyang Chen, Junyu Zhang, Mark Zhao, Cheng Qian, Kangrui Wang, Qineng Wang, Teja Venkat Koripella, Marziyeh Movahedi, Manling Li, Heng Ji, Huan Zhang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09560)  

**Abstract**: Leveraging Multi-modal Large Language Models (MLLMs) to create embodied agents offers a promising avenue for tackling real-world tasks. While language-centric embodied agents have garnered substantial attention, MLLM-based embodied agents remain underexplored due to the lack of comprehensive evaluation frameworks. To bridge this gap, we introduce EmbodiedBench, an extensive benchmark designed to evaluate vision-driven embodied agents. EmbodiedBench features: (1) a diverse set of 1,128 testing tasks across four environments, ranging from high-level semantic tasks (e.g., household) to low-level tasks involving atomic actions (e.g., navigation and manipulation); and (2) six meticulously curated subsets evaluating essential agent capabilities like commonsense reasoning, complex instruction understanding, spatial awareness, visual perception, and long-term planning. Through extensive experiments, we evaluated 13 leading proprietary and open-source MLLMs within EmbodiedBench. Our findings reveal that: MLLMs excel at high-level tasks but struggle with low-level manipulation, with the best model, GPT-4o, scoring only 28.9% on average. EmbodiedBench provides a multifaceted standardized evaluation platform that not only highlights existing challenges but also offers valuable insights to advance MLLM-based embodied agents. Our code is available at this https URL. 

**Abstract (ZH)**: 利用多模态大型语言模型（MLLMs）创建具身代理为应对现实世界任务提供了 promising 的途径。尽管以语言为中心的具身代理受到了广泛关注，但由于缺乏全面的评估框架，基于MLLM的具身代理仍处于探索阶段。为填补这一空白，我们引入了EmbodiedBench，一个广泛的设计用于评估以视觉驱动的具身代理的基准。EmbodiedBench 特点包括：（1）涵盖四个环境的1,128项测试任务，从高层语义任务（如家庭）到涉及原子动作的低级任务（如导航和操作）；（2）六个精心策划的子集，评估诸如常识推理、复杂指令理解、空间意识、视觉感知和长期规划等关键代理能力。通过 extensive 实验，我们在EmbodiedBench 中评估了13个领先的专有和开源MLLM。我们的发现表明：MLLM 在高层任务中表现优异但在低级操作中挣扎，最佳模型GPT-4o 平均得分仅28.9%。EmbodiedBench 提供了一个多维度的标准化评估平台，不仅突显了现有挑战，还为推进基于MLLM的具身代理提供了宝贵的见解。我们的代码可在以下 URL 查看。 

---
# On the Promise for Assurance of Differentiable Neurosymbolic Reasoning Paradigms 

**Title (ZH)**: 可差分神经符号推理范式的保证前景 

**Authors**: Luke E. Richards, Jessie Yaros, Jasen Babcock, Coung Ly, Robin Cosbey, Timothy Doster, Cynthia Matuszek  

**Link**: [PDF](https://arxiv.org/pdf/2502.08932)  

**Abstract**: To create usable and deployable Artificial Intelligence (AI) systems, there requires a level of assurance in performance under many different conditions. Many times, deployed machine learning systems will require more classic logic and reasoning performed through neurosymbolic programs jointly with artificial neural network sensing. While many prior works have examined the assurance of a single component of the system solely with either the neural network alone or entire enterprise systems, very few works have examined the assurance of integrated neurosymbolic systems. Within this work, we assess the assurance of end-to-end fully differentiable neurosymbolic systems that are an emerging method to create data-efficient and more interpretable models. We perform this investigation using Scallop, an end-to-end neurosymbolic library, across classification and reasoning tasks in both the image and audio domains. We assess assurance across adversarial robustness, calibration, user performance parity, and interpretability of solutions for catching misaligned solutions. We find end-to-end neurosymbolic methods present unique opportunities for assurance beyond their data efficiency through our empirical results but not across the board. We find that this class of neurosymbolic models has higher assurance in cases where arithmetic operations are defined and where there is high dimensionality to the input space, where fully neural counterparts struggle to learn robust reasoning operations. We identify the relationship between neurosymbolic models' interpretability to catch shortcuts that later result in increased adversarial vulnerability despite performance parity. Finally, we find that the promise of data efficiency is typically only in the case of class imbalanced reasoning problems. 

**Abstract (ZH)**: 创建可使用和可部署的人工智能系统需要在多种条件下对其性能有一定的保障。虽然部署的机器学习系统通常需要通过神经符号程序与人工神经网络感知相结合的经典逻辑和推理，但许多先前的研究仅对系统中的单一组件进行了保障评估，要么仅考察神经网络，要么仅考察整个企业系统，很少有研究考察集成神经符号系统的保障。在本文中，我们评估了端到端完全可微分的神经符号系统，这些系统是创建数据效率更高和更具解释性的模型的一种新兴方法。我们使用端到端神经符号库Scallop在图像和音频领域进行了分类和推理任务的研究，并从鲁棒性、校准、用户性能平等性和解决方案的解释性等方面评估了保障。我们的实证结果表明，端到端神经符号方法在数据效率之外提供了独特的保障机会，但在所有情况下并不适用。我们发现，当算术操作定义明确且输入空间具有高维度时，此类神经符号模型具有更高的保障，而全神经网络对手势学习有效的推理操作存在困难。我们确定了神经符号模型的解释性与捕捉捷径之间的关系，这些捷径最终可能导致对抗性脆弱性增加，尽管表现平等。最后，我们发现数据效率的承诺通常仅适用于类别不平衡的推理问题。 

---
# Contextual bandits with entropy-based human feedback 

**Title (ZH)**: 基于熵的抽检反馈上下文臂问题 

**Authors**: Raihan Seraj, Lili Meng, Tristan Sylvain  

**Link**: [PDF](https://arxiv.org/pdf/2502.08759)  

**Abstract**: In recent years, preference-based human feedback mechanisms have become essential for enhancing model performance across diverse applications, including conversational AI systems such as ChatGPT. However, existing approaches often neglect critical aspects, such as model uncertainty and the variability in feedback quality. To address these challenges, we introduce an entropy-based human feedback framework for contextual bandits, which dynamically balances exploration and exploitation by soliciting expert feedback only when model entropy exceeds a predefined threshold. Our method is model-agnostic and can be seamlessly integrated with any contextual bandit agent employing stochastic policies. Through comprehensive experiments, we show that our approach achieves significant performance improvements while requiring minimal human feedback, even under conditions of suboptimal feedback quality. This work not only presents a novel strategy for feedback solicitation but also highlights the robustness and efficacy of incorporating human guidance into machine learning systems. Our code is publicly available: this https URL 

**Abstract (ZH)**: 近年来，基于偏好的人类反馈机制已成为提升跨多种应用领域模型性能的关键，包括像ChatGPT这样的对话AI系统。然而，现有的方法往往忽视了模型不确定性及反馈质量差异等关键方面。为应对这些挑战，我们提出了一种基于熵的人类反馈框架，该框架通过在模型熵超过预定义阈值时仅请求专家反馈，动态平衡探索与利用。我们的方法具有模型无关性，可无缝集成到任何采用随机策略的上下文臂代理中。通过全面的实验，我们展示了该方法在需要极少人类反馈的情况下仍能实现显著的性能提升，即使在反馈质量次优的情况下也是如此。本研究不仅提出了一种新的反馈请求策略，还强调了将人类指导融入机器学习系统中的鲁棒性和有效性。我们的代码已公开：this https URL。 

---
# Language Agents as Digital Representatives in Collective Decision-Making 

**Title (ZH)**: 语言代理作为集体决策中的数字代表 

**Authors**: Daniel Jarrett, Miruna Pîslar, Michiel A. Bakker, Michael Henry Tessler, Raphael Köster, Jan Balaguer, Romuald Elie, Christopher Summerfield, Andrea Tacchetti  

**Link**: [PDF](https://arxiv.org/pdf/2502.09369)  

**Abstract**: Consider the process of collective decision-making, in which a group of individuals interactively select a preferred outcome from among a universe of alternatives. In this context, "representation" is the activity of making an individual's preferences present in the process via participation by a proxy agent -- i.e. their "representative". To this end, learned models of human behavior have the potential to fill this role, with practical implications for multi-agent scenario studies and mechanism design. In this work, we investigate the possibility of training \textit{language agents} to behave in the capacity of representatives of human agents, appropriately expressing the preferences of those individuals whom they stand for. First, we formalize the setting of \textit{collective decision-making} -- as the episodic process of interaction between a group of agents and a decision mechanism. On this basis, we then formalize the problem of \textit{digital representation} -- as the simulation of an agent's behavior to yield equivalent outcomes from the mechanism. Finally, we conduct an empirical case study in the setting of \textit{consensus-finding} among diverse humans, and demonstrate the feasibility of fine-tuning large language models to act as digital representatives. 

**Abstract (ZH)**: 考虑集体决策过程，在该过程中，一组个体通过代理代理人的参与相互选择众多备选方案中的最优结果。在这种背景下，“代表”是指通过代理代理人参与的方式将个体的偏好体现在决策过程中。为此，人类行为的习得模型有可能承担这一角色，在多智能体场景研究和机制设计方面具有实际意义。在本文中，我们研究训练语言代理以代表人类代理人的可能性，适当表达其所代表的个体的偏好。首先，我们将集体决策设置形式化——作为一组代理与决策机制之间互动的阶段性过程。在此基础上，我们进一步将“数字代理”的问题形式化——模拟代理人的行为以从机制中产生等效结果。最后，我们在多样人类达成共识的情境下进行实证案例研究，并证明大型语言模型微调以充当数字代理的可能性。 

---
# Architecture for Simulating Behavior Mode Changes in Norm-Aware Autonomous Agents 

**Title (ZH)**: 规范模拟规范意识自主代理行为模式变化的架构 

**Authors**: Sean Glaze, Daniela Inclezan  

**Link**: [PDF](https://arxiv.org/pdf/2502.09215)  

**Abstract**: This paper presents an architecture for simulating the actions of a norm-aware intelligent agent whose behavior with respect to norm compliance is set, and can later be changed, by a human controller. Updating an agent's behavior mode from a norm-abiding to a riskier one may be relevant when the agent is involved in time-sensitive rescue operations, for example. We base our work on the Authorization and Obligation Policy Language AOPL designed by Gelfond and Lobo for the specification of norms. We introduce an architecture and a prototype software system that can be used to simulate an agent's plans under different behavior modes that can later be changed by the controller. We envision such software to be useful to policy makers, as they can more readily understand how agents may act in certain situations based on the agents' attitudes towards norm-compliance. Policy makers may then refine their policies if simulations show unwanted consequences. 

**Abstract (ZH)**: 本文提出了一种架构，用于模拟一个规范意识智能代理的行为，其规范遵守行为由人类控制器设定，并可后由人类控制器更改。当代理参与时间敏感的救援操作时，从规范遵守行为模式更新为更具风险的行为模式可能是相关的。我们基于Gelfond和Lobo为规范规定设计的授权和义务政策语言AOPL开展工作。我们引入了一种架构和原型软件系统，该系统可用于在不同行为模式下模拟代理的计划，并可由控制器后更改。我们设想此类软件对政策制定者来说是有用的，因为他们可以根据代理对规范遵守的态度更容易地理解代理在某些情况下的行为。如果仿真显示了不希望的后果，政策制定者可以据此进一步完善其政策。 

---
# AIDE: Agentically Improve Visual Language Model with Domain Experts 

**Title (ZH)**: AIDE: 由域专家代理改进视觉语言模型 

**Authors**: Ming-Chang Chiu, Fuxiao Liu, Karan Sapra, Andrew Tao, Yaser Jacoob, Xuezhe Ma, Zhiding Yu, Guilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09051)  

**Abstract**: The enhancement of Visual Language Models (VLMs) has traditionally relied on knowledge distillation from larger, more capable models. This dependence creates a fundamental bottleneck for improving state-of-the-art systems, particularly when no superior models exist. We introduce AIDE (Agentic Improvement through Domain Experts), a novel framework that enables VLMs to autonomously enhance their capabilities by leveraging specialized domain expert models. AIDE operates through a four-stage process: (1) identifying instances for refinement, (2) engaging domain experts for targeted analysis, (3) synthesizing expert outputs with existing data, and (4) integrating enhanced instances into the training pipeline. Experiments on multiple benchmarks, including MMMU, MME, MMBench, etc., demonstrate AIDE's ability to achieve notable performance gains without relying on larger VLMs nor human supervision. Our framework provides a scalable, resource-efficient approach to continuous VLM improvement, addressing critical limitations in current methodologies, particularly valuable when larger models are unavailable to access. 

**Abstract (ZH)**: 通过领域专家增强视觉语言模型(AIDE) 

---
# Neural Force Field: Learning Generalized Physical Representation from a Few Examples 

**Title (ZH)**: 神经力场：从少量示例中学习通用物理表示 

**Authors**: Shiqian Li, Ruihong Shen, Chi Zhang, Yixin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.08987)  

**Abstract**: Physical reasoning is a remarkable human ability that enables rapid learning and generalization from limited experience. Current AI models, despite extensive training, still struggle to achieve similar generalization, especially in Out-of-distribution (OOD) settings. This limitation stems from their inability to abstract core physical principles from observations. A key challenge is developing representations that can efficiently learn and generalize physical dynamics from minimal data. Here we present Neural Force Field (NFF) a modeling framework built on Neural Ordinary Differential Equation (NODE) that learns interpretable force field representations which can be efficiently integrated through an Ordinary Differential Equation ( ODE) solver to predict object trajectories. Unlike existing approaches that rely on high-dimensional latent spaces, NFF captures fundamental physical concepts such as gravity, support, and collision in an interpretable manner. Experiments on two challenging physical reasoning tasks demonstrate that NFF, trained with only a few examples, achieves strong generalization to unseen scenarios. This physics-grounded representation enables efficient forward-backward planning and rapid adaptation through interactive refinement. Our work suggests that incorporating physics-inspired representations into learning systems can help bridge the gap between artificial and human physical reasoning capabilities. 

**Abstract (ZH)**: 基于物理的推理是人类的一项非凡能力，能够使人在有限的经验中实现快速学习和泛化。当前的AI模型尽管经过大量训练，但在实现类似泛化能力方面仍然存在局限，特别是在离域分布(OOD)设置中。这一局限来源于它们无法从观察中抽象出核心物理原理的能力。一个关键挑战是开发能够从少量数据中高效学习和泛化的物理动力学表示。我们提出了一种名为Neural Force Field (NFF)的建模框架，该框架基于神经常微分方程(NODE)，学习可解释的力场表示，并可通过常微分方程(ODE)求解器高效集成以预测物体轨迹。与依赖于高维潜在空间的现有方法不同，NFF以可解释的方式捕捉到了诸如重力、支撑和碰撞等基本物理概念。在两个具有挑战性的物理推理任务上的实验表明，仅用少量样本训练的NFF能够实现对未见过场景的强泛化能力。这种基于物理的表示使高效前向-后向规划和通过交互式细化实现快速适应成为可能。我们的工作表明，在学习系统中嵌入基于物理的表示可以有助于弥补人工与人类物理推理能力之间的差距。 

---
# Exploring Emotion-Sensitive LLM-Based Conversational AI 

**Title (ZH)**: 探索情感敏感的大语言模型驱动的对话人工智能 

**Authors**: Antonin Brun, Ruying Liu, Aryan Shukla, Frances Watson, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2502.08920)  

**Abstract**: Conversational AI chatbots have become increasingly common within the customer service industry. Despite improvements in their emotional development, they often lack the authenticity of real customer service interactions or the competence of service providers. By comparing emotion-sensitive and emotion-insensitive LLM-based chatbots across 30 participants, we aim to explore how emotional sensitivity in chatbots influences perceived competence and overall customer satisfaction in service interactions. Additionally, we employ sentiment analysis techniques to analyze and interpret the emotional content of user inputs. We highlight that perceptions of chatbot trustworthiness and competence were higher in the case of the emotion-sensitive chatbot, even if issue resolution rates were not affected. We discuss implications of improved user satisfaction from emotion-sensitive chatbots and potential applications in support services. 

**Abstract (ZH)**: 基于情感的对话AI聊天机器人在客户服务行业中的情感敏感性对感知能力和总体客户满意度的影响研究：基于30名参与者的比较分析及情绪内容分析 

---
# Centrally Coordinated Multi-Agent Reinforcement Learning for Power Grid Topology Control 

**Title (ZH)**: 集中协调多智能体强化学习在电力网络拓扑控制中的应用 

**Authors**: Barbera de Mol, Davide Barbieri, Jan Viebahn, Davide Grossi  

**Link**: [PDF](https://arxiv.org/pdf/2502.08681)  

**Abstract**: Power grid operation is becoming more complex due to the increase in generation of renewable energy. The recent series of Learning To Run a Power Network (L2RPN) competitions have encouraged the use of artificial agents to assist human dispatchers in operating power grids. However, the combinatorial nature of the action space poses a challenge to both conventional optimizers and learned controllers. Action space factorization, which breaks down decision-making into smaller sub-tasks, is one approach to tackle the curse of dimensionality. In this study, we propose a centrally coordinated multi-agent (CCMA) architecture for action space factorization. In this approach, regional agents propose actions and subsequently a coordinating agent selects the final action. We investigate several implementations of the CCMA architecture, and benchmark in different experimental settings against various L2RPN baseline approaches. The CCMA architecture exhibits higher sample efficiency and superior final performance than the baseline approaches. The results suggest high potential of the CCMA approach for further application in higher-dimensional L2RPN as well as real-world power grid settings. 

**Abstract (ZH)**: 基于中央协调的多agent架构在分解动作空间中的应用研究 

---
