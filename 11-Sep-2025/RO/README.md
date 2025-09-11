# RoboChemist: Long-Horizon and Safety-Compliant Robotic Chemical Experimentation 

**Title (ZH)**: RoboChemist: 长期规划与安全合规的机器人化学实验 

**Authors**: Zongzheng Zhang, Chenghao Yue, Haobo Xu, Minwen Liao, Xianglin Qi, Huan-ang Gao, Ziwei Wang, Hao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.08820)  

**Abstract**: Robotic chemists promise to both liberate human experts from repetitive tasks and accelerate scientific discovery, yet remain in their infancy. Chemical experiments involve long-horizon procedures over hazardous and deformable substances, where success requires not only task completion but also strict compliance with experimental norms. To address these challenges, we propose \textit{RoboChemist}, a dual-loop framework that integrates Vision-Language Models (VLMs) with Vision-Language-Action (VLA) models. Unlike prior VLM-based systems (e.g., VoxPoser, ReKep) that rely on depth perception and struggle with transparent labware, and existing VLA systems (e.g., RDT, pi0) that lack semantic-level feedback for complex tasks, our method leverages a VLM to serve as (1) a planner to decompose tasks into primitive actions, (2) a visual prompt generator to guide VLA models, and (3) a monitor to assess task success and regulatory compliance. Notably, we introduce a VLA interface that accepts image-based visual targets from the VLM, enabling precise, goal-conditioned control. Our system successfully executes both primitive actions and complete multi-step chemistry protocols. Results show 23.57% higher average success rate and a 0.298 average increase in compliance rate over state-of-the-art VLA baselines, while also demonstrating strong generalization to objects and tasks. 

**Abstract (ZH)**: 机器人化学家有望解放人类专家从事重复性工作并加速科学发现，但目前仍处于初级阶段。化学实验涉及长周期 procedure 操作危险且变形的物质，成功不仅需要完成任务，还需要严格遵守实验规范。为应对这些挑战，我们提出 RoboChemist 双环框架，将视觉语言模型（VLM）与视觉语言动作（VLA）模型集成。不同于依赖深度感知但在透明器皿处理上挣扎的 prior VLM 系统（如 VoxPoser、ReKep），以及缺乏复杂任务语义级反馈的现有 VLA 系统（如 RDT、pi0），我们的方法利用 VLM 作为（1）计划者，将任务分解为原始动作；（2）视觉提示生成器，引导 VLA 模型；（3）监控器，评估任务成功和监管合规性。特别地，我们引入了一种 VLA 接口，接受来自 VLM 的基于图像的视觉目标，实现精确的、目标导向的控制。我们的系统成功执行了原始动作和完整的多步化学协议。结果显示，与最先进的 VLA 基准相比，平均成功率提高了 23.57%，平均合规性提高了 0.298，并且还展示了对物体和任务的强烈泛化能力。 

---
# Calib3R: A 3D Foundation Model for Multi-Camera to Robot Calibration and 3D Metric-Scaled Scene Reconstruction 

**Title (ZH)**: Calib3R：多相机到机器人标定及3D度量缩放场景重建的三维基础模型 

**Authors**: Davide Allegro, Matteo Terreran, Stefano Ghidoni  

**Link**: [PDF](https://arxiv.org/pdf/2509.08813)  

**Abstract**: Robots often rely on RGB images for tasks like manipulation and navigation. However, reliable interaction typically requires a 3D scene representation that is metric-scaled and aligned with the robot reference frame. This depends on accurate camera-to-robot calibration and dense 3D reconstruction, tasks usually treated separately, despite both relying on geometric correspondences from RGB data. Traditional calibration needs patterns, while RGB-based reconstruction yields geometry with an unknown scale in an arbitrary frame. Multi-camera setups add further complexity, as data must be expressed in a shared reference frame. We present Calib3R, a patternless method that jointly performs camera-to-robot calibration and metric-scaled 3D reconstruction via unified optimization. Calib3R handles single- and multi-camera setups on robot arms or mobile robots. It builds on the 3D foundation model MASt3R to extract pointmaps from RGB images, which are combined with robot poses to reconstruct a scaled 3D scene aligned with the robot. Experiments on diverse datasets show that Calib3R achieves accurate calibration with less than 10 images, outperforming target-less and marker-based methods. 

**Abstract (ZH)**: 无图案方法Calib3R：通过统一优化实现相机到机器人标定和米尺度3D重建 

---
# Joint Model-based Model-free Diffusion for Planning with Constraints 

**Title (ZH)**: 基于模型的模型自由扩散规划约束下的联合建模 

**Authors**: Wonsuhk Jung, Utkarsh A. Mishra, Nadun Ranawaka Arachchige, Yongxin Chen, Danfei Xu, Shreyas Kousik  

**Link**: [PDF](https://arxiv.org/pdf/2509.08775)  

**Abstract**: Model-free diffusion planners have shown great promise for robot motion planning, but practical robotic systems often require combining them with model-based optimization modules to enforce constraints, such as safety. Naively integrating these modules presents compatibility challenges when diffusion's multi-modal outputs behave adversarially to optimization-based modules. To address this, we introduce Joint Model-based Model-free Diffusion (JM2D), a novel generative modeling framework. JM2D formulates module integration as a joint sampling problem to maximize compatibility via an interaction potential, without additional training. Using importance sampling, JM2D guides modules outputs based only on evaluations of the interaction potential, thus handling non-differentiable objectives commonly arising from non-convex optimization modules. We evaluate JM2D via application to aligning diffusion planners with safety modules on offline RL and robot manipulation. JM2D significantly improves task performance compared to conventional safety filters without sacrificing safety. Further, we show that conditional generation is a special case of JM2D and elucidate key design choices by comparing with SOTA gradient-based and projection-based diffusion planners. More details at: this https URL. 

**Abstract (ZH)**: 无模型扩散规划器在机器人运动规划中展现了巨大的潜力，但实际的机器人系统通常需要将它们与基于模型的优化模块相结合以满足约束需求，如安全性约束。简单地整合这些模块会在扩散的多模态输出与基于优化的模块发生冲突时带来兼容性挑战。为了解决这个问题，我们引入了联合基于模型的无模型扩散（JM2D），这是一种新颖的生成建模框架。JM2D 将模块集成问题表述为联合采样问题，通过交互势能最大化兼容性，而不需要额外的训练。利用重要性采样，JM2D 仅基于交互势能的评估结果来引导模块输出，从而处理来自非凸优化模块的常见非区分性目标。我们通过在离线 RL 和机器人操作中使扩散规划器与安全模块对齐来评估 JM2D。相比传统的安全过滤器，JM2D 显著提高了任务性能且不牺牲安全性。此外，我们展示了条件生成是 JM2D 的一个特例，并通过与最先进的梯度基和投影基扩散规划器的比较来阐明关键设计选择。更多细节请参见：[这里](this https URL)。 

---
# SocialNav-SUB: Benchmarking VLMs for Scene Understanding in Social Robot Navigation 

**Title (ZH)**: SocialNav-SUB: 社交机器人导航中场景理解的VLMs基准测试 

**Authors**: Michael J. Munje, Chen Tang, Shuijing Liu, Zichao Hu, Yifeng Zhu, Jiaxun Cui, Garrett Warnell, Joydeep Biswas, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2509.08757)  

**Abstract**: Robot navigation in dynamic, human-centered environments requires socially-compliant decisions grounded in robust scene understanding. Recent Vision-Language Models (VLMs) exhibit promising capabilities such as object recognition, common-sense reasoning, and contextual understanding-capabilities that align with the nuanced requirements of social robot navigation. However, it remains unclear whether VLMs can accurately understand complex social navigation scenes (e.g., inferring the spatial-temporal relations among agents and human intentions), which is essential for safe and socially compliant robot navigation. While some recent works have explored the use of VLMs in social robot navigation, no existing work systematically evaluates their ability to meet these necessary conditions. In this paper, we introduce the Social Navigation Scene Understanding Benchmark (SocialNav-SUB), a Visual Question Answering (VQA) dataset and benchmark designed to evaluate VLMs for scene understanding in real-world social robot navigation scenarios. SocialNav-SUB provides a unified framework for evaluating VLMs against human and rule-based baselines across VQA tasks requiring spatial, spatiotemporal, and social reasoning in social robot navigation. Through experiments with state-of-the-art VLMs, we find that while the best-performing VLM achieves an encouraging probability of agreeing with human answers, it still underperforms simpler rule-based approach and human consensus baselines, indicating critical gaps in social scene understanding of current VLMs. Our benchmark sets the stage for further research on foundation models for social robot navigation, offering a framework to explore how VLMs can be tailored to meet real-world social robot navigation needs. An overview of this paper along with the code and data can be found at this https URL . 

**Abstract (ZH)**: 动态、以人为核心环境中服务机器人的导航决策需要基于稳固场景理解的社会合规性选择。视觉-语言模型（VLMs）展现出诸如物体识别、常识推理和情境理解等有前景的能力，这些能力与服务机器人社会导航的细微需求相契合。然而，尚不清楚VLMs是否能够准确理解复杂的社会导航场景（例如，推断智能体间的空间-时间关系及人类意图），这对于安全和社会合规的机器人导航是必不可少的。虽然已有部分研究探索了在社会机器人导航中使用VLMs的应用，但尚未有研究系统评估其是否能够满足这些必要条件。本文介绍了社会导航场景理解基准（SocialNav-SUB），这是一个视觉问答（VQA）数据集和基准，旨在评估VLMs在真实世界社会机器人导航场景中的场景理解能力。SocialNav-SUB提供了一个统一框架，用于评估VLMs在涉及社会机器人导航中所需的空间、空间-时间和社会推理的VQA任务中相对于基于视觉问答的基准（包括基于视觉问答的人类和规则为基础的基准）的表现。通过使用最先进的VLMs进行实验，我们发现，尽管性能最好的VLM在与人类答案一致的概率上取得令人鼓舞的结果，但它仍然未能超越简单的规则为基础的方法和人类共识基准，表明当前VLMs在社会场景理解方面的关键差距。该基准为进一步研究基础模型在社会机器人导航中的应用奠定了基础，提供了一个框架以研究如何使VLMs适应现实世界的社会机器人导航需求。本文概要、代码和数据可访问此链接：this https URL。 

---
# Parallel, Asymptotically Optimal Algorithms for Moving Target Traveling Salesman Problems 

**Title (ZH)**: 并行、渐近最优算法在移动目标旅行商问题中 

**Authors**: Anoop Bhat, Geordan Gutow, Bhaskar Vundurthy, Zhongqiang Ren, Sivakumar Rathinam, Howie Choset  

**Link**: [PDF](https://arxiv.org/pdf/2509.08743)  

**Abstract**: The Moving Target Traveling Salesman Problem (MT-TSP) seeks an agent trajectory that intercepts several moving targets, within a particular time window for each target. In the presence of generic nonlinear target trajectories or kinematic constraints on the agent, no prior algorithm guarantees convergence to an optimal MT-TSP solution. Therefore, we introduce the Iterated Random Generalized (IRG) TSP framework. The key idea behind IRG is to alternate between randomly sampling a set of agent configuration-time points, corresponding to interceptions of targets, and finding a sequence of interception points by solving a generalized TSP (GTSP). This alternation enables asymptotic convergence to the optimum. We introduce two parallel algorithms within the IRG framework. The first algorithm, IRG-PGLNS, solves GTSPs using PGLNS, our parallelized extension of the state-of-the-art solver GLNS. The second algorithm, Parallel Communicating GTSPs (PCG), solves GTSPs corresponding to several sets of points simultaneously. We present numerical results for three variants of the MT-TSP: one where intercepting a target only requires coming within a particular distance, another where the agent is a variable-speed Dubins car, and a third where the agent is a redundant robot arm. We show that IRG-PGLNS and PCG both converge faster than a baseline based on prior work. 

**Abstract (ZH)**: 基于移动目标的旅行商问题（MT-TSP）寻求在一个特定的时间窗口内拦截多个移动目标的代理轨迹。在存在通用非线性目标轨迹或代理的动力学约束时，没有任何先有算法可以保证收敛到最优的MT-TSP解。因此，我们引入了迭代随机广义（IRG）旅行商问题框架。IRG的核心思想是交替进行随机采样一组代理配置-时间点（对应于拦截目标的点）和通过求解广义旅行商问题（GTSP）来找到拦截点序列。这种交替能使算法渐近收敛到最优解。我们在此框架内引入了两个并行算法。第一个算法IRG-PGLNS使用我们的并行扩展GLNS（PGLNS）来求解GTSP。第二个算法并行通信广义旅行商问题（PCG）同时求解多个点集对应的GTSP。我们提供了三种不同版本的MT-TSP的数值结果：一个只需要接近特定距离即可拦截目标的情况，另一个代理是一个可变速度的杜宾车（Dubins车），第三个代理是一个冗余的机器人手臂。我们展示了IRG-PGLNS和PCG都比基于先前工作的基线算法收敛速度更快。 

---
# TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals 

**Title (ZH)**: TANGO: 基于可达性意识的局部度量控制导航以实现拓扑目标 

**Authors**: Stefan Podgorski, Sourav Garg, Mehdi Hosseinzadeh, Lachlan Mares, Feras Dayoub, Ian Reid  

**Link**: [PDF](https://arxiv.org/pdf/2509.08699)  

**Abstract**: Visual navigation in robotics traditionally relies on globally-consistent 3D maps or learned controllers, which can be computationally expensive and difficult to generalize across diverse environments. In this work, we present a novel RGB-only, object-level topometric navigation pipeline that enables zero-shot, long-horizon robot navigation without requiring 3D maps or pre-trained controllers. Our approach integrates global topological path planning with local metric trajectory control, allowing the robot to navigate towards object-level sub-goals while avoiding obstacles. We address key limitations of previous methods by continuously predicting local trajectory using monocular depth and traversability estimation, and incorporating an auto-switching mechanism that falls back to a baseline controller when necessary. The system operates using foundational models, ensuring open-set applicability without the need for domain-specific fine-tuning. We demonstrate the effectiveness of our method in both simulated environments and real-world tests, highlighting its robustness and deployability. Our approach outperforms existing state-of-the-art methods, offering a more adaptable and effective solution for visual navigation in open-set environments. The source code is made publicly available: this https URL. 

**Abstract (ZH)**: 视觉导航在机器人技术中传统上依赖于全局一致的3D地图或学习控制器，这在计算上可能非常昂贵，并且在跨多种环境时难以泛化。在本文中，我们提出了一种新的仅基于RGB的物体级别拓扑导航管道，无需使用3D地图或预训练控制器即可实现零样本、长时程的机器人导航。我们的方法结合了全局拓扑路径规划和局部度量轨迹控制，使机器人能够导航至物体级别子目标的同时避开障碍物。我们通过持续利用单目深度估计和通过arb策略来解决先前方法的关键局限性，并在必要时切换回基础控制器。该系统使用基础模型运行，确保在无需领域特定微调的情况下具有开放集适用性。我们在模拟环境和实际测试中验证了我们方法的有效性，突显其鲁棒性和可部署性。我们的方法在开放集环境中优于现有最先进的方法，提供了更具适应性和有效性的视觉导航解决方案。源代码已公开：this https URL。 

---
# AutoODD: Agentic Audits via Bayesian Red Teaming in Black-Box Models 

**Title (ZH)**: AutoODD: 基于贝叶斯红队测试的代理审核 

**Authors**: Rebecca Martin, Jay Patrikar, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2509.08638)  

**Abstract**: Specialized machine learning models, regardless of architecture and training, are susceptible to failures in deployment. With their increasing use in high risk situations, the ability to audit these models by determining their operational design domain (ODD) is crucial in ensuring safety and compliance. However, given the high-dimensional input spaces, this process often requires significant human resources and domain expertise. To alleviate this, we introduce \coolname, an LLM-Agent centric framework for automated generation of semantically relevant test cases to search for failure modes in specialized black-box models. By leveraging LLM-Agents as tool orchestrators, we aim to fit a uncertainty-aware failure distribution model on a learned text-embedding manifold by projecting the high-dimension input space to low-dimension text-embedding latent space. The LLM-Agent is tasked with iteratively building the failure landscape by leveraging tools for generating test-cases to probe the model-under-test (MUT) and recording the response. The agent also guides the search using tools to probe uncertainty estimate on the low dimensional manifold. We demonstrate this process in a simple case using models trained with missing digits on the MNIST dataset and in the real world setting of vision-based intruder detection for aerial vehicles. 

**Abstract (ZH)**: 专用机器学习模型在任何架构和训练方式下，在部署过程中都容易出现故障。随着它们在高风险场景中的使用增加，通过确定其操作设计域（ODD）来审计这些模型的能力对于确保安全性和合规性至关重要。然而，由于高维度的输入空间，这一过程往往需要大量的人力资源和领域专业知识。为了解决这一问题，我们引入了\coolname框架，这是一个以LLM-Agent为中心的自动化生成语义相关测试案例的框架，用于搜索专用黑盒模型的故障模式。通过利用LLM-Agent作为工具调度器，我们旨在通过将高维度输入空间投影到低维度文本嵌入潜空间上来拟合一个不确定性感知的失败分布模型。LLM-Agent的任务是通过利用生成测试案例的工具来逐步构建故障景观，并记录模型-under-test（MUT）的响应。代理还使用工具来探索低维度流形上的不确定性估计，以引导搜索。我们在使用MNIST数据集上训练缺少数字的模型的简单案例中演示了这一过程，并在基于视觉的航空器上入侵检测真实世界场景中进行了演示。 

---
# RoboMatch: A Mobile-Manipulation Teleoperation Platform with Auto-Matching Network Architecture for Long-Horizon Manipulation 

**Title (ZH)**: RoboMatch：一种具有自动匹配网络架构的移动 manipulator 远程操作平台，用于长时 horizon 操作 

**Authors**: Hanyu Liu, Yunsheng Ma, Jiaxin Huang, Keqiang Ren, Jiayi Wen, Yilin Zheng, Baishu Wan, Pan Li, Jiejun Hou, Haoru Luan, Zhihua Wang, Zhigong Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.08522)  

**Abstract**: This paper presents RoboMatch, a novel unified teleoperation platform for mobile manipulation with an auto-matching network architecture, designed to tackle long-horizon tasks in dynamic environments. Our system enhances teleoperation performance, data collection efficiency, task accuracy, and operational stability. The core of RoboMatch is a cockpit-style control interface that enables synchronous operation of the mobile base and dual arms, significantly improving control precision and data collection. Moreover, we introduce the Proprioceptive-Visual Enhanced Diffusion Policy (PVE-DP), which leverages Discrete Wavelet Transform (DWT) for multi-scale visual feature extraction and integrates high-precision IMUs at the end-effector to enrich proprioceptive feedback, substantially boosting fine manipulation performance. Furthermore, we propose an Auto-Matching Network (AMN) architecture that decomposes long-horizon tasks into logical sequences and dynamically assigns lightweight pre-trained models for distributed inference. Experimental results demonstrate that our approach improves data collection efficiency by over 20%, increases task success rates by 20-30% with PVE-DP, and enhances long-horizon inference performance by approximately 40% with AMN, offering a robust solution for complex manipulation tasks. 

**Abstract (ZH)**: RoboMatch：一种用于移动操控的新型统一平台及其自匹配网络架构 

---
# FMT$^{x}$: An Efficient and Asymptotically Optimal Extension of the Fast Marching Tree for Dynamic Replanning 

**Title (ZH)**: FMT$^{x}$: 一种高效的动态重规划渐近最优扩展方法 

**Authors**: Soheil Espahbodini Nia  

**Link**: [PDF](https://arxiv.org/pdf/2509.08521)  

**Abstract**: Path planning in dynamic environments remains a core challenge in robotics, especially as autonomous systems are deployed in unpredictable spaces such as warehouses and public roads. While algorithms like Fast Marching Tree (FMT$^{*}$) offer asymptotically optimal solutions in static settings, their single-pass design prevents path revisions which are essential for real-time adaptation. On the other hand, full replanning is often too computationally expensive. This paper introduces FMT$^{x}$, an extension of the Fast Marching Tree algorithm that enables efficient and consistent replanning in dynamic environments. We revisit the neighbor selection rule of FMT$^{*}$ and demonstrate that a minimal change overcomes its single-pass limitation, enabling the algorithm to update cost-to-come values upon discovering better connections without sacrificing asymptotic optimality or computational efficiency. By maintaining a cost-ordered priority queue and applying a selective update condition that uses an expanding neighbor to identify and trigger the re-evaluation of any node with a potentially suboptimal path, FMT$^{x}$ ensures that suboptimal routes are efficiently repaired as the environment evolves. This targeted strategy preserves the inherent efficiency of FMT$^{*}$ while enabling robust adaptation to changes in obstacle configuration. FMT$^{x}$ is proven to recover an asymptotically optimal solution after environmental changes. Experimental results demonstrate that FMT$^{x}$ outperforms the influential replanner RRT$^{x}$, reacting more swiftly to dynamic events with lower computational overhead and thus offering a more effective solution for real-time robotic navigation in unpredictable worlds. 

**Abstract (ZH)**: 动态环境中的路径规划仍然是机器人技术中的一个核心挑战，尤其是在部署在仓库和公共道路上等不可预测的空间中。虽然像Fast Marching Tree (FMT$^{*}$)这样的算法在静态环境中提供了渐近最优的解决方案，但它们的一次性设计限制了路径的修订，这对实时适应是必不可少的。另一方面，全面重规划往往计算成本过高。本文引入了FMT$^{x}$，这是一种Fast Marching Tree算法的扩展，能够在动态环境中实现高效且一致的重规划。我们重新审视了FMT$^{*}$的邻居选择规则，并证明了最小的变动克服了其一次性设计的局限性，使算法能够在发现更好连接的情况下更新成本到当前点值，同时不牺牲渐近最优性和计算效率。通过维护一个按成本排序的优先队列，并应用一种选择性更新条件，使用扩展邻居来识别和触发任何具有潜在次优路径的节点的重新评估，FMT$^{x}$确保随着时间环境的变化，子优化的路径能够有效地得到修复。这种有针对性的策略保留了FMT$^{*}$固有的高效性，同时还能够稳健地适应障碍配置的变化。FMT$^{x}$在环境变化后被证明可以恢复渐近最优解。实验结果表明，FMT$^{x}$优于有影响力的重规划器RRT$^{x}$，能够更快地应对动态事件，同时具有较低的计算开销，从而为不可预测世界中的实时机器人导航提供更有效的解决方案。 

---
# Facilitating the Emergence of Assistive Robots to Support Frailty: Psychosocial and Environmental Realities 

**Title (ZH)**: 促进辅助机器人在应对衰弱方面的 emergence，以考虑心理社会和环境现实 

**Authors**: Angela Higgins, Stephen Potter, Mauro Dragone, Mark Hawley, Farshid Amirabdollahian, Alessandro Di Nuovo, Praminda Caleb-Solly  

**Link**: [PDF](https://arxiv.org/pdf/2509.08510)  

**Abstract**: While assistive robots have much potential to help older people with frailty-related needs, there are few in use. There is a gap between what is developed in laboratories and what would be viable in real-world contexts. Through a series of co-design workshops (61 participants across 7 sessions) including those with lived experience of frailty, their carers, and healthcare professionals, we gained a deeper understanding of everyday issues concerning the place of new technologies in their lives. A persona-based approach surfaced emotional, social, and psychological issues. Any assistive solution must be developed in the context of this complex interplay of psychosocial and environmental factors. Our findings, presented as design requirements in direct relation to frailty, can help promote design thinking that addresses people's needs in a more pragmatic way to move assistive robotics closer to real-world use. 

**Abstract (ZH)**: 辅助机器人虽有潜力帮助体弱老人，但在实际应用中却鲜有使用。实验室中的开发与现实应用之间存在差距。通过一系列共设工作坊（共计7次，61名参与者，包括体弱经验者、照护者及医疗专业人员），我们深入理解了新技术在他们生活中的日常问题。基于角色的方法揭示了情感、社会和心理问题。任何辅助解决方案都必须考虑到心理社会和环境因素的复杂交互。我们的研究成果，以直接关联体弱的需求形式呈现，有助于促进更实际的设计思维，推动辅助机器人技术更接近实际应用。 

---
# CLAP: Clustering to Localize Across n Possibilities, A Simple, Robust Geometric Approach in the Presence of Symmetries 

**Title (ZH)**: CLAP：在存在对称性的条件下定位多种可能性的聚类方法，一种简单的稳健几何approach 

**Authors**: Gabriel I. Fernandez, Ruochen Hou, Alex Xu, Colin Togashi, Dennis W. Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.08495)  

**Abstract**: In this paper, we present our localization method called CLAP, Clustering to Localize Across $n$ Possibilities, which helped us win the RoboCup 2024 adult-sized autonomous humanoid soccer competition. Competition rules limited our sensor suite to stereo vision and an inertial sensor, similar to humans. In addition, our robot had to deal with varying lighting conditions, dynamic feature occlusions, noise from high-impact stepping, and mistaken features from bystanders and neighboring fields. Therefore, we needed an accurate, and most importantly robust localization algorithm that would be the foundation for our path-planning and game-strategy algorithms. CLAP achieves these requirements by clustering estimated states of our robot from pairs of field features to localize its global position and orientation. Correct state estimates naturally cluster together, while incorrect estimates spread apart, making CLAP resilient to noise and incorrect inputs. CLAP is paired with a particle filter and an extended Kalman filter to improve consistency and smoothness. Tests of CLAP with other landmark-based localization methods showed similar accuracy. However, tests with increased false positive feature detection showed that CLAP outperformed other methods in terms of robustness with very little divergence and velocity jumps. Our localization performed well in competition, allowing our robot to shoot faraway goals and narrowly defend our goal. 

**Abstract (ZH)**: 基于聚类的多可能性定位方法CLAP：RoboCup 2024成人组自主人形足球竞赛中的定位技术 

---
# Dual-Stage Safe Herding Framework for Adversarial Attacker in Dynamic Environment 

**Title (ZH)**: 动态环境中的对抗攻击者双重阶段安全牧羊框架 

**Authors**: Wenqing Wang, Ye Zhang, Haoyu Li, Jingyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.08460)  

**Abstract**: Recent advances in robotics have enabled the widespread deployment of autonomous robotic systems in complex operational environments, presenting both unprecedented opportunities and significant security problems. Traditional shepherding approaches based on fixed formations are often ineffective or risky in urban and obstacle-rich scenarios, especially when facing adversarial agents with unknown and adaptive behaviors. This paper addresses this challenge as an extended herding problem, where defensive robotic systems must safely guide adversarial agents with unknown strategies away from protected areas and into predetermined safe regions, while maintaining collision-free navigation in dynamic environments. We propose a hierarchical hybrid framework based on reach-avoid game theory and local motion planning, incorporating a virtual containment boundary and event-triggered pursuit mechanisms to enable scalable and robust multi-agent coordination. Simulation results demonstrate that the proposed approach achieves safe and efficient guidance of adversarial agents to designated regions. 

**Abstract (ZH)**: 近期机器人领域的进展使得自主机器人系统能够在复杂操作环境中得到广泛应用，既带来了前所未有的机遇，也引出了重大的安全问题。基于固定编队的传统放牧方法在城市和障碍物丰富的场景中往往无效或存在风险，尤其是在面对具有未知和适应性行为的敌对代理时。本文将这一挑战作为一种扩展的放牧问题进行研究，其中防御性机器人系统必须安全地引导具有未知策略的敌对代理远离受保护区域，并将其引导至预定的 safe 区域，同时在动态环境下保持无碰撞导航。我们提出了一种基于到达避免博弈理论和局部运动规划的分层混合框架，结合虚拟包容边界和事件触发捕获机制，以实现可扩展和 robust 的多代理协调。仿真实验结果表明，所提出的方法能够在指定区域安全有效地引导敌对代理。 

---
# Augmenting Neural Networks-based Model Approximators in Robotic Force-tracking Tasks 

**Title (ZH)**: 基于神经网络的模型逼近器在机器人力跟踪任务中的增强 

**Authors**: Kevin Saad, Vincenzo Petrone, Enrico Ferrentino, Pasquale Chiacchio, Francesco Braghin, Loris Roveda  

**Link**: [PDF](https://arxiv.org/pdf/2509.08440)  

**Abstract**: As robotics gains popularity, interaction control becomes crucial for ensuring force tracking in manipulator-based tasks. Typically, traditional interaction controllers either require extensive tuning, or demand expert knowledge of the environment, which is often impractical in real-world applications. This work proposes a novel control strategy leveraging Neural Networks (NNs) to enhance the force-tracking behavior of a Direct Force Controller (DFC). Unlike similar previous approaches, it accounts for the manipulator's tangential velocity, a critical factor in force exertion, especially during fast motions. The method employs an ensemble of feedforward NNs to predict contact forces, then exploits the prediction to solve an optimization problem and generate an optimal residual action, which is added to the DFC output and applied to an impedance controller. The proposed Velocity-augmented Artificial intelligence Interaction Controller for Ambiguous Models (VAICAM) is validated in the Gazebo simulator on a Franka Emika Panda robot. Against a vast set of trajectories, VAICAM achieves superior performance compared to two baseline controllers. 

**Abstract (ZH)**: 随着机器人技术的流行，交互控制对于确保基于操作器的任务中的力跟踪变得至关重要。传统交互控制器通常要么需要大量的调整，要么需要专家对环境的了解，这在实际应用中往往是不现实的。本工作提出了一种新的控制策略，利用神经网络（NNs）以增强直接力控制器（DFC）的力跟踪行为。与之前的类似方法不同，该方法考虑了操作器的切向速度，这是力施加的一个关键因素，尤其是在快速运动中。该方法使用前向神经网络的集成来预测接触力，然后利用预测来解决优化问题并生成最优残差动作，该动作被添加到DFC的输出并应用于阻抗控制器。提出的基于速度增强的人工智能交互控制器 for 不确定模型（VAICAM）在 Franka Emika Panda 机器人上使用 Gazebo 模拟器进行了验证。在大量轨迹上，VAICAM 的性能优于两个基线控制器。 

---
# PegasusFlow: Parallel Rolling-Denoising Score Sampling for Robot Diffusion Planner Flow Matching 

**Title (ZH)**: PegasusFlow：并行滚动去噪评分采样用于机器人扩散规划流匹配 

**Authors**: Lei Ye, Haibo Gao, Peng Xu, Zhelin Zhang, Junqi Shan, Ao Zhang, Wei Zhang, Ruyi Zhou, Zongquan Deng, Liang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.08435)  

**Abstract**: Diffusion models offer powerful generative capabilities for robot trajectory planning, yet their practical deployment on robots is hindered by a critical bottleneck: a reliance on imitation learning from expert demonstrations. This paradigm is often impractical for specialized robots where data is scarce and creates an inefficient, theoretically suboptimal training pipeline. To overcome this, we introduce PegasusFlow, a hierarchical rolling-denoising framework that enables direct and parallel sampling of trajectory score gradients from environmental interaction, completely bypassing the need for expert data. Our core innovation is a novel sampling algorithm, Weighted Basis Function Optimization (WBFO), which leverages spline basis representations to achieve superior sample efficiency and faster convergence compared to traditional methods like MPPI. The framework is embedded within a scalable, asynchronous parallel simulation architecture that supports massively parallel rollouts for efficient data collection. Extensive experiments on trajectory optimization and robotic navigation tasks demonstrate that our approach, particularly Action-Value WBFO (AVWBFO) combined with a reinforcement learning warm-start, significantly outperforms baselines. In a challenging barrier-crossing task, our method achieved a 100% success rate and was 18% faster than the next-best method, validating its effectiveness for complex terrain locomotion planning. this https URL 

**Abstract (ZH)**: 基于扩散模型的机器人轨迹规划：突破依赖专家演示的瓶颈——引入PegasusFlow 

---
# Grasp Like Humans: Learning Generalizable Multi-Fingered Grasping from Human Proprioceptive Sensorimotor Integration 

**Title (ZH)**: 如人类般抓取：基于人类本体感受传感器运动整合学习通用多指抓取 

**Authors**: Ce Guo, Xieyuanli Chen, Zhiwen Zeng, Zirui Guo, Yihong Li, Haoran Xiao, Dewen Hu, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.08354)  

**Abstract**: Tactile and kinesthetic perceptions are crucial for human dexterous manipulation, enabling reliable grasping of objects via proprioceptive sensorimotor integration. For robotic hands, even though acquiring such tactile and kinesthetic feedback is feasible, establishing a direct mapping from this sensory feedback to motor actions remains challenging. In this paper, we propose a novel glove-mediated tactile-kinematic perception-prediction framework for grasp skill transfer from human intuitive and natural operation to robotic execution based on imitation learning, and its effectiveness is validated through generalized grasping tasks, including those involving deformable objects. Firstly, we integrate a data glove to capture tactile and kinesthetic data at the joint level. The glove is adaptable for both human and robotic hands, allowing data collection from natural human hand demonstrations across different scenarios. It ensures consistency in the raw data format, enabling evaluation of grasping for both human and robotic hands. Secondly, we establish a unified representation of multi-modal inputs based on graph structures with polar coordinates. We explicitly integrate the morphological differences into the designed representation, enhancing the compatibility across different demonstrators and robotic hands. Furthermore, we introduce the Tactile-Kinesthetic Spatio-Temporal Graph Networks (TK-STGN), which leverage multidimensional subgraph convolutions and attention-based LSTM layers to extract spatio-temporal features from graph inputs to predict node-based states for each hand joint. These predictions are then mapped to final commands through a force-position hybrid mapping. 

**Abstract (ZH)**: 触觉和本体感受对于人体灵巧操作至关重要，能够通过本体感觉运动整合实现对象的可靠抓取。对于机器人手来说，尽管获取这样的触觉和本体感受反馈是可行的，但将其感知识反馈直接映射到运动动作仍然具有挑战性。在本文中，我们提出了一种基于模仿学习的新型手套介导的触觉-动力学感知预测框架，用于将人类直观自然的操作技能转移到机器人的执行中，并通过包括涉及可变形物体的广泛抓取任务对其有效性进行了验证。首先，我们集成了一副数据手套来捕捉关节级别的触觉和本体感受数据。该手套适用于人类和机器人的手，允许在不同场景下从自然的人类手部演示中收集数据。这确保了原始数据格式的一致性，使得能够对人类和机器人手的抓取进行评估。其次，我们基于极坐标结构的图结构建立了多模态输入的统一表示。我们显式地将形态差异整合到所设计的表示中，增强了不同演示者和机器人手之间的兼容性。此外，我们引入了触觉-本体感受时空图网络(Tactile-Kinesthetic Spatio-Temporal Graph Networks, TK-STGN)，该网络利用多维度子图卷积和基于注意力机制的LSTM层从图输入中提取时空特征，以预测每个手关节的节点状态。这些预测然后通过力-位置混合映射映射到最终命令。 

---
# Good Deep Features to Track: Self-Supervised Feature Extraction and Tracking in Visual Odometry 

**Title (ZH)**: 好的深度特征追踪：视觉里程表中的自我监督特征提取与追踪 

**Authors**: Sai Puneeth Reddy Gottam, Haoming Zhang, Eivydas Keras  

**Link**: [PDF](https://arxiv.org/pdf/2509.08333)  

**Abstract**: Visual-based localization has made significant progress, yet its performance often drops in large-scale, outdoor, and long-term settings due to factors like lighting changes, dynamic scenes, and low-texture areas. These challenges degrade feature extraction and tracking, which are critical for accurate motion estimation. While learning-based methods such as SuperPoint and SuperGlue show improved feature coverage and robustness, they still face generalization issues with out-of-distribution data. We address this by enhancing deep feature extraction and tracking through self-supervised learning with task specific feedback. Our method promotes stable and informative features, improving generalization and reliability in challenging environments. 

**Abstract (ZH)**: 基于视觉的定位在大规模、户外和长期设置中由于光照变化、动态场景和低纹理区域等因素，其性能往往下降，这些挑战会恶化特征提取和跟踪，这对准确的运动估计至关重要。尽管基于学习的方法如SuperPoint和SuperGlue在特征覆盖和鲁棒性方面表现出改进，但仍存在分布外数据的一般化问题。我们通过特定任务的自监督学习增强深度特征提取和跟踪，促进稳定性和信息性特征，从而在具有挑战性的环境中提高一般化能力和可靠性。 

---
# Foundation Models for Autonomous Driving Perception: A Survey Through Core Capabilities 

**Title (ZH)**: 自主驾驶感知中的基础模型：核心能力综述 

**Authors**: Rajendramayavan Sathyam, Yueqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.08302)  

**Abstract**: Foundation models are revolutionizing autonomous driving perception, transitioning the field from narrow, task-specific deep learning models to versatile, general-purpose architectures trained on vast, diverse datasets. This survey examines how these models address critical challenges in autonomous perception, including limitations in generalization, scalability, and robustness to distributional shifts. The survey introduces a novel taxonomy structured around four essential capabilities for robust performance in dynamic driving environments: generalized knowledge, spatial understanding, multi-sensor robustness, and temporal reasoning. For each capability, the survey elucidates its significance and comprehensively reviews cutting-edge approaches. Diverging from traditional method-centric surveys, our unique framework prioritizes conceptual design principles, providing a capability-driven guide for model development and clearer insights into foundational aspects. We conclude by discussing key challenges, particularly those associated with the integration of these capabilities into real-time, scalable systems, and broader deployment challenges related to computational demands and ensuring model reliability against issues like hallucinations and out-of-distribution failures. The survey also outlines crucial future research directions to enable the safe and effective deployment of foundation models in autonomous driving systems. 

**Abstract (ZH)**: 基础模型正在革命性地改变自动驾驶感知领域，从狭窄的任务特定深度学习模型过渡到基于广泛多样数据集训练的多功能通用架构。本文综述了这些模型如何应对自动驾驶感知中的关键挑战，包括泛化能力、可扩展性和分布偏移下的稳健性。本文综述引入了一种新的分类法，围绕在动态驾驶环境中实现稳健性能的四种核心能力：通用知识、空间理解、多传感器稳健性以及时间推理。对于每种能力，本文综述了其重要性，并全面回顾了最新的方法。不同于传统的以方法为中心的综述，我们独特的框架更注重概念设计原则，为模型开发提供能力驱动的指导，并更清晰地揭示基础方面的要点。文末讨论了关键挑战，特别是将这些能力整合到实时可扩展系统中以及更广泛的与计算需求和确保模型在幻觉和分布外失败等问题方面的可靠性相关部署挑战，并概述了未来研究的关键方向，以实现基础模型在自动驾驶系统中的安全有效部署。 

---
# Symmetry-Guided Multi-Agent Inverse Reinforcement Learnin 

**Title (ZH)**: 对称性指导的多智能体逆强化学习 

**Authors**: Yongkai Tian, Yirong Qi, Xin Yu, Wenjun Wu, Jie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.08257)  

**Abstract**: In robotic systems, the performance of reinforcement learning depends on the rationality of predefined reward functions. However, manually designed reward functions often lead to policy failures due to inaccuracies. Inverse Reinforcement Learning (IRL) addresses this problem by inferring implicit reward functions from expert demonstrations. Nevertheless, existing methods rely heavily on large amounts of expert demonstrations to accurately recover the reward function. The high cost of collecting expert demonstrations in robotic applications, particularly in multi-robot systems, severely hinders the practical deployment of IRL. Consequently, improving sample efficiency has emerged as a critical challenge in multi-agent inverse reinforcement learning (MIRL). Inspired by the symmetry inherent in multi-agent systems, this work theoretically demonstrates that leveraging symmetry enables the recovery of more accurate reward functions. Building upon this insight, we propose a universal framework that integrates symmetry into existing multi-agent adversarial IRL algorithms, thereby significantly enhancing sample efficiency. Experimental results from multiple challenging tasks have demonstrated the effectiveness of this framework. Further validation in physical multi-robot systems has shown the practicality of our method. 

**Abstract (ZH)**: 多智能体逆强化学习中基于对称性的样本效率提升方法 

---
# Behaviorally Heterogeneous Multi-Agent Exploration Using Distributed Task Allocation 

**Title (ZH)**: 基于分布式任务分配的行为异质多智能体探索 

**Authors**: Nirabhra Mandal, Aamodh Suresh, Carlos Nieto-Granda, Sonia Martínez  

**Link**: [PDF](https://arxiv.org/pdf/2509.08242)  

**Abstract**: We study a problem of multi-agent exploration with behaviorally heterogeneous robots. Each robot maps its surroundings using SLAM and identifies a set of areas of interest (AoIs) or frontiers that are the most informative to explore next. The robots assess the utility of going to a frontier using Behavioral Entropy (BE) and then determine which frontier to go to via a distributed task assignment scheme. We convert the task assignment problem into a non-cooperative game and use a distributed algorithm (d-PBRAG) to converge to the Nash equilibrium (which we show is the optimal task allocation solution). For unknown utility cases, we provide robust bounds using approximate rewards. We test our algorithm (which has less communication cost and fast convergence) in simulation, where we explore the effect of sensing radii, sensing accuracy, and heterogeneity among robotic teams with respect to the time taken to complete exploration and path traveled. We observe that having a team of agents with heterogeneous behaviors is beneficial. 

**Abstract (ZH)**: 多机器人行为异质性探索问题研究 

---
# Sample-Efficient Online Control Policy Learning with Real-Time Recursive Model Updates 

**Title (ZH)**: 实时递归模型更新的高效在线控制策略学习 

**Authors**: Zixin Zhang, James Avtges, Todd D. Murphey  

**Link**: [PDF](https://arxiv.org/pdf/2509.08241)  

**Abstract**: Data-driven control methods need to be sample-efficient and lightweight, especially when data acquisition and computational resources are limited -- such as during learning on hardware. Most modern data-driven methods require large datasets and struggle with real-time updates of models, limiting their performance in dynamic environments. Koopman theory formally represents nonlinear systems as linear models over observables, and Koopman representations can be determined from data in an optimization-friendly setting with potentially rapid model updates. In this paper, we present a highly sample-efficient, Koopman-based learning pipeline: Recursive Koopman Learning (RKL). We identify sufficient conditions for model convergence and provide formal algorithmic analysis supporting our claim that RKL is lightweight and fast, with complexity independent of dataset size. We validate our method on a simulated planar two-link arm and a hybrid nonlinear hardware system with soft actuators, showing that real-time recursive Koopman model updates improve the sample efficiency and stability of data-driven controller synthesis -- requiring only <10% of the data compared to benchmarks. The high-performance C++ codebase is open-sourced. Website: this https URL. 

**Abstract (ZH)**: 基于Koopman的高效递归学习方法：实时Koopman模型更新在动态环境下的数据驱动控制器合成中提升样本效率和稳定性 

---
# Deep Visual Odometry for Stereo Event Cameras 

**Title (ZH)**: 深度视觉里程计用于立体事件相机 

**Authors**: Sheng Zhong, Junkai Niu, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.08235)  

**Abstract**: Event-based cameras are bio-inspired sensors with pixels that independently and asynchronously respond to brightness changes at microsecond resolution, offering the potential to handle state estimation tasks involving motion blur and high dynamic range (HDR) illumination conditions. However, the versatility of event-based visual odometry (VO) relying on handcrafted data association (either direct or indirect methods) is still unreliable, especially in field robot applications under low-light HDR conditions, where the dynamic range can be enormous and the signal-to-noise ratio is spatially-and-temporally varying. Leveraging deep neural networks offers new possibilities for overcoming these challenges. In this paper, we propose a learning-based stereo event visual odometry. Building upon Deep Event Visual Odometry (DEVO), our system (called Stereo-DEVO) introduces a novel and efficient static-stereo association strategy for sparse depth estimation with almost no additional computational burden. By integrating it into a tightly coupled bundle adjustment (BA) optimization scheme, and benefiting from the recurrent network's ability to perform accurate optical flow estimation through voxel-based event representations to establish reliable patch associations, our system achieves high-precision pose estimation in metric scale. In contrast to the offline performance of DEVO, our system can process event data of \zs{Video Graphics Array} (VGA) resolution in real time. Extensive evaluations on multiple public real-world datasets and self-collected data justify our system's versatility, demonstrating superior performance compared to state-of-the-art event-based VO methods. More importantly, our system achieves stable pose estimation even in large-scale nighttime HDR scenarios. 

**Abstract (ZH)**: 基于事件的立体视觉里程计：学习驱动的事件视觉里程计 

---
# Input-gated Bilateral Teleoperation: An Easy-to-implement Force Feedback Teleoperation Method for Low-cost Hardware 

**Title (ZH)**: 输入门控双边遥控：一种易实现的低成本硬件力反馈遥控方法 

**Authors**: Yoshiki Kanai, Akira Kanazawa, Hideyuki Ichiwara, Hiroshi Ito, Naoaki Noguchi, Tetsuya Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2509.08226)  

**Abstract**: Effective data collection in contact-rich manipulation requires force feedback during teleoperation, as accurate perception of contact is crucial for stable control. However, such technology remains uncommon, largely because bilateral teleoperation systems are complex and difficult to implement. To overcome this, we propose a bilateral teleoperation method that relies only on a simple feedback controller and does not require force sensors. The approach is designed for leader-follower setups using low-cost hardware, making it broadly applicable. Through numerical simulations and real-world experiments, we demonstrate that the method requires minimal parameter tuning, yet achieves both high operability and contact stability, outperforming conventional approaches. Furthermore, we show its high robustness: even at low communication cycle rates between leader and follower, control performance degradation is minimal compared to high-speed operation. We also prove our method can be implemented on two types of commercially available low-cost hardware with zero parameter adjustments. This highlights its high ease of implementation and versatility. We expect this method will expand the use of force feedback teleoperation systems on low-cost hardware. This will contribute to advancing contact-rich task autonomy in imitation learning. 

**Abstract (ZH)**: 有效的接触丰富操作中的数据采集需要遥操作过程中提供力反馈，以确保稳定的控制，而准确感知接触至关重要。然而，这项技术仍然相对罕见，部分原因是双边遥操作系统复杂且难以实现。为克服这一问题，我们提出了一种仅依赖于简单反馈控制器且不需要力传感器的双边遥操作方法。该方法适用于低成本硬件的领导者-追随者设置，具有广泛适用性。通过数值仿真和实际实验，我们展示了该方法需要极少的参数调整，但仍然实现了高操作性和接触稳定性，优于传统方法。此外，我们还表明其具有高鲁棒性：即使在领导者和追随者之间通信周期率较低的情况下，其控制性能的下降也极为有限，与高速操作相比几乎没有性能损失。我们还证明该方法可以在两种商用低成本硬件上实现，无需参数调整。这凸显了其实施的简便性和通用性。我们预期该方法将促进力反馈遥操作系统在低成本硬件上的应用，为模仿学习中的接触丰富任务自主性发展做出贡献。 

---
# A Comprehensive Review of Reinforcement Learning for Autonomous Driving in the CARLA Simulator 

**Title (ZH)**: CARLA模拟器中自主驾驶强化学习的综述 

**Authors**: Elahe Delavari, Feeza Khan Khanzada, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2509.08221)  

**Abstract**: Autonomous-driving research has recently embraced deep Reinforcement Learning (RL) as a promising framework for data-driven decision making, yet a clear picture of how these algorithms are currently employed, benchmarked and evaluated is still missing. This survey fills that gap by systematically analysing around 100 peer-reviewed papers that train, test or validate RL policies inside the open-source CARLA simulator. We first categorize the literature by algorithmic family model-free, model-based, hierarchical, and hybrid and quantify their prevalence, highlighting that more than 80% of existing studies still rely on model-free methods such as DQN, PPO and SAC. Next, we explain the diverse state, action and reward formulations adopted across works, illustrating how choices of sensor modality (RGB, LiDAR, BEV, semantic maps, and carla kinematics states), control abstraction (discrete vs. continuous) and reward shaping are used across various literature. We also consolidate the evaluation landscape by listing the most common metrics (success rate, collision rate, lane deviation, driving score) and the towns, scenarios and traffic configurations used in CARLA benchmarks. Persistent challenges including sparse rewards, sim-to-real transfer, safety guarantees and limited behaviour diversity are distilled into a set of open research questions, and promising directions such as model-based RL, meta-learning and richer multi-agent simulations are outlined. By providing a unified taxonomy, quantitative statistics and a critical discussion of limitations, this review aims to serve both as a reference for newcomers and as a roadmap for advancing RL-based autonomous driving toward real-world deployment. 

**Abstract (ZH)**: 自主驾驶研究近年来将深度强化学习（RL）视为一种有前途的数据驱动决策框架，然而关于这些算法目前的应用、基准测试和评估的清晰图景仍不清楚。本文综述通过系统分析约100篇已在开源CARLA模拟器中训练、测试或验证RL策略的同行评议论文，填补了这一空白。首先，按算法家族将其分类为模型自由型、模型依赖型、层次型和混合型，并量化其分布情况，强调超过80%的现有研究仍依赖于DQN、PPO和SAC等模型自由方法。接着，解释了各篇论文中采用的多样状态、动作和奖励形式，展示了传感器模态（RGB、LiDAR、BEV、语义地图和CARLA运动状态）、控制抽象（离散 vs. 连续）以及奖励塑造在不同文献中的应用情况。还通过列出最常见的评估指标（成功率、碰撞率、车道偏离、驾驶评分）以及CARLA基准中使用的城镇、场景和交通配置，汇总了评估格局。提炼出持续存在的挑战，如稀疏奖励、模拟到现实的转移、安全性保证和行为多样性的局限性，并提出了模型依赖式RL、元学习和更丰富的多智能体模拟等有前景的方向。通过提供统一的分类体系、定量统计和对局限性的批判性讨论，本文综述旨在为新入学者提供参考，并为基于RL的自主驾驶向实际部署推进提供方向图。 

---
# Online Dynamic SLAM with Incremental Smoothing and Mapping 

**Title (ZH)**: 在线动态SLAM增量平滑与建图 

**Authors**: Jesse Morris, Yiduo Wang, Viorela Ila  

**Link**: [PDF](https://arxiv.org/pdf/2509.08197)  

**Abstract**: Dynamic SLAM methods jointly estimate for the static and dynamic scene components, however existing approaches, while accurate, are computationally expensive and unsuitable for online applications. In this work, we present the first application of incremental optimisation techniques to Dynamic SLAM. We introduce a novel factor-graph formulation and system architecture designed to take advantage of existing incremental optimisation methods and support online estimation. On multiple datasets, we demonstrate that our method achieves equal to or better than state-of-the-art in camera pose and object motion accuracy. We further analyse the structural properties of our approach to demonstrate its scalability and provide insight regarding the challenges of solving Dynamic SLAM incrementally. Finally, we show that our formulation results in problem structure well-suited to incremental solvers, while our system architecture further enhances performance, achieving a 5x speed-up over existing methods. 

**Abstract (ZH)**: 动态SLAM方法联合估计静态和动态场景成分，但现有方法尽管准确，计算成本高且不适用于在线应用。在此工作中，我们首次将增量优化技术应用于动态SLAM。我们引入了一种新型因子图表示和系统架构，旨在利用现有的增量优化方法并支持在线估计。在多个数据集中，我们证明我们的方法在相机位姿和物体运动准确性方面达到或优于现有最佳方法。我们进一步分析了我们方法的结构特性，以证明其可扩展性，并提供关于增量求解动态SLAM的挑战见解。最后，我们展示了我们的表示方法生成的问题结构非常适合增量求解器，而我们的系统架构进一步提高了性能，实现了比现有方法快5倍的效果。 

---
# Quadrotor Navigation using Reinforcement Learning with Privileged Information 

**Title (ZH)**: 基于特权信息的四旋翼飞行器导航强化学习方法 

**Authors**: Jonathan Lee, Abhishek Rathod, Kshitij Goel, John Stecklein, Wennie Tabib  

**Link**: [PDF](https://arxiv.org/pdf/2509.08177)  

**Abstract**: This paper presents a reinforcement learning-based quadrotor navigation method that leverages efficient differentiable simulation, novel loss functions, and privileged information to navigate around large obstacles. Prior learning-based methods perform well in scenes that exhibit narrow obstacles, but struggle when the goal location is blocked by large walls or terrain. In contrast, the proposed method utilizes time-of-arrival (ToA) maps as privileged information and a yaw alignment loss to guide the robot around large obstacles. The policy is evaluated in photo-realistic simulation environments containing large obstacles, sharp corners, and dead-ends. Our approach achieves an 86% success rate and outperforms baseline strategies by 34%. We deploy the policy onboard a custom quadrotor in outdoor cluttered environments both during the day and night. The policy is validated across 20 flights, covering 589 meters without collisions at speeds up to 4 m/s. 

**Abstract (ZH)**: 基于强化学习的利用高效可微模拟、新型损失函数和特权信息的四旋翼导航方法 

---
# Diffusion-Guided Multi-Arm Motion Planning 

**Title (ZH)**: 扩散引导多臂运动规划 

**Authors**: Viraj Parimi, Brian C. Williams  

**Link**: [PDF](https://arxiv.org/pdf/2509.08160)  

**Abstract**: Multi-arm motion planning is fundamental for enabling arms to complete complex long-horizon tasks in shared spaces efficiently but current methods struggle with scalability due to exponential state-space growth and reliance on large training datasets for learned models. Inspired by Multi-Agent Path Finding (MAPF), which decomposes planning into single-agent problems coupled with collision resolution, we propose a novel diffusion-guided multi-arm planner (DG-MAP) that enhances scalability of learning-based models while reducing their reliance on massive multi-arm datasets. Recognizing that collisions are primarily pairwise, we train two conditional diffusion models, one to generate feasible single-arm trajectories, and a second, to model the dual-arm dynamics required for effective pairwise collision resolution. By integrating these specialized generative models within a MAPF-inspired structured decomposition, our planner efficiently scales to larger number of arms. Evaluations against alternative learning-based methods across various team sizes demonstrate our method's effectiveness and practical applicability. Project website can be found at this https URL 

**Abstract (ZH)**: 多臂运动规划是使机械臂高效完成长期复杂任务的基本要求，但当前方法由于状态空间增长的指数级增长和对大规模训练数据集的依赖而在可扩展性上遇到困难。受多智能体路径寻找（MAPF）的启发，我们将规划分解为单智能体问题并结合碰撞解决，提出了一种新颖的扩散引导多臂规划器（DG-MAP），旨在增强基于学习模型的可扩展性并减少对其大规模多臂数据集的依赖。认识到碰撞主要是成对的，我们训练了两个条件扩散模型：一个用于生成可行的单臂轨迹，另一个用于建模用于有效成对碰撞解决所需的双臂动力学。通过将这些专门的生成模型整合到一个受MAPF启发的结构分解中，我们的规划器可以更高效地扩展到更多的机械臂。对各种团队规模的其他基于学习的方法进行评估，证明了我们方法的有效性和实用性。项目网址为：<https://github.com/Qwen-2/DG-MAP>。 

---
# Zero-Shot Metric Depth Estimation via Monocular Visual-Inertial Rescaling for Autonomous Aerial Navigation 

**Title (ZH)**: 单目视觉-惯性缩放的零样本度量深度估计用于自主 aerial 导航 

**Authors**: Steven Yang, Xiaoyu Tian, Kshitij Goel, Wennie Tabib  

**Link**: [PDF](https://arxiv.org/pdf/2509.08159)  

**Abstract**: This paper presents a methodology to predict metric depth from monocular RGB images and an inertial measurement unit (IMU). To enable collision avoidance during autonomous flight, prior works either leverage heavy sensors (e.g., LiDARs or stereo cameras) or data-intensive and domain-specific fine-tuning of monocular metric depth estimation methods. In contrast, we propose several lightweight zero-shot rescaling strategies to obtain metric depth from relative depth estimates via the sparse 3D feature map created using a visual-inertial navigation system. These strategies are compared for their accuracy in diverse simulation environments. The best performing approach, which leverages monotonic spline fitting, is deployed in the real-world on a compute-constrained quadrotor. We obtain on-board metric depth estimates at 15 Hz and demonstrate successful collision avoidance after integrating the proposed method with a motion primitives-based planner. 

**Abstract (ZH)**: 本文提出了一种从单目RGB图像和惯性测量单元(IMU)预测度量深度的方法，并在此基础上实现了自主飞行过程中的碰撞避免。 

---
# Risk-Bounded Multi-Agent Visual Navigation via Dynamic Budget Allocation 

**Title (ZH)**: 基于动态预算分配的Risk-Bounded多智能体视觉导航 

**Authors**: Viraj Parimi, Brian C. Williams  

**Link**: [PDF](https://arxiv.org/pdf/2509.08157)  

**Abstract**: Safe navigation is essential for autonomous systems operating in hazardous environments, especially when multiple agents must coordinate using just visual inputs over extended time horizons. Traditional planning methods excel at solving long-horizon tasks but rely on predefined distance metrics, while safe Reinforcement Learning (RL) can learn complex behaviors using high-dimensional inputs yet struggles with multi-agent, goal-conditioned scenarios. Recent work combined these paradigms by leveraging goal-conditioned RL (GCRL) to build an intermediate graph from replay buffer states, pruning unsafe edges, and using Conflict-Based Search (CBS) for multi-agent path planning. Although effective, this graph-pruning approach can be overly conservative, limiting mission efficiency by precluding missions that must traverse high-risk regions. To address this limitation, we propose RB-CBS, a novel extension to CBS that dynamically allocates and adjusts user-specified risk bound ($\Delta$) across agents to flexibly trade off safety and speed. Our improved planner ensures that each agent receives a local risk budget ($\delta$) enabling more efficient navigation while still respecting overall safety constraints. Experimental results demonstrate that this iterative risk-allocation framework yields superior performance in complex environments, allowing multiple agents to find collision-free paths within the user-specified $\Delta$. 

**Abstract (ZH)**: 基于风险动态分配的冲突_based搜索（RB-CBS）算法在复杂环境中的安全高效导航 

---
# Mean Field Game-Based Interactive Trajectory Planning Using Physics-Inspired Unified Potential Fields 

**Title (ZH)**: 基于物理启发统一 Potential Fields 的均场游戏化交互轨迹规划 

**Authors**: Zhen Tian, Fujiang Yuan, Chunhong Yuan, Yanhong Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.08147)  

**Abstract**: Interactive trajectory planning in autonomous driving must balance safety, efficiency, and scalability under heterogeneous driving behaviors. Existing methods often face high computational cost or rely on external safety critics. To address this, we propose an Interaction-Enriched Unified Potential Field (IUPF) framework that fuses style-dependent benefit and risk fields through a physics-inspired variational model, grounded in mean field game theory. The approach captures conservative, aggressive, and cooperative behaviors without additional safety modules, and employs stochastic differential equations to guarantee Nash equilibrium with exponential convergence. Simulations on lane changing and overtaking scenarios show that IUPF ensures safe distances, generates smooth and efficient trajectories, and outperforms traditional optimization and game-theoretic baselines in both adaptability and computational efficiency. 

**Abstract (ZH)**: 自主驾驶中的交互轨迹规划必须在异构驾驶行为下平衡安全、效率和扩展性。为此，我们提出了一种基于物理启发的变分模型融合风格依赖的益处和风险场的交互增强统一势场（IUPF）框架，该框架基于均场博弈理论，无需额外的安全模块即可捕捉保守、激进和合作行为，并利用随机微分方程确保指数收敛的纳什均衡。模拟结果显示，IUPF能够确保安全距离，生成平滑高效的轨迹，并在适应性和计算效率方面优于传统优化和博弈论baseline方法。 

---
# Attribute-based Object Grounding and Robot Grasp Detection with Spatial Reasoning 

**Title (ZH)**: 基于属性的对象定位与空间推理的机器人抓取检测 

**Authors**: Houjian Yu, Zheming Zhou, Min Sun, Omid Ghasemalizadeh, Yuyin Sun, Cheng-Hao Kuo, Arnie Sen, Changhyun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.08126)  

**Abstract**: Enabling robots to grasp objects specified through natural language is essential for effective human-robot interaction, yet it remains a significant challenge. Existing approaches often struggle with open-form language expressions and typically assume unambiguous target objects without duplicates. Moreover, they frequently rely on costly, dense pixel-wise annotations for both object grounding and grasp configuration. We present Attribute-based Object Grounding and Robotic Grasping (OGRG), a novel framework that interprets open-form language expressions and performs spatial reasoning to ground target objects and predict planar grasp poses, even in scenes containing duplicated object instances. We investigate OGRG in two settings: (1) Referring Grasp Synthesis (RGS) under pixel-wise full supervision, and (2) Referring Grasp Affordance (RGA) using weakly supervised learning with only single-pixel grasp annotations. Key contributions include a bi-directional vision-language fusion module and the integration of depth information to enhance geometric reasoning, improving both grounding and grasping performance. Experiment results show that OGRG outperforms strong baselines in tabletop scenes with diverse spatial language instructions. In RGS, it operates at 17.59 FPS on a single NVIDIA RTX 2080 Ti GPU, enabling potential use in closed-loop or multi-object sequential grasping, while delivering superior grounding and grasp prediction accuracy compared to all the baselines considered. Under the weakly supervised RGA setting, OGRG also surpasses baseline grasp-success rates in both simulation and real-robot trials, underscoring the effectiveness of its spatial reasoning design. Project page: this https URL 

**Abstract (ZH)**: 基于属性的物体语义接地与机器人抓取（OGRG）：通过自然语言表达实现灵活的物体抓取 

---
# Online Learning and Coverage of Unknown Fields Using Random-Feature Gaussian Processes 

**Title (ZH)**: 使用随机特征高斯过程进行未知领域在线学习与覆盖 

**Authors**: Ruijie Du, Ruoyu Lin, Yanning Shen, Magnus Egerstedt  

**Link**: [PDF](https://arxiv.org/pdf/2509.08117)  

**Abstract**: This paper proposes a framework for multi-robot systems to perform simultaneous learning and coverage of the domain of interest characterized by an unknown and potentially time-varying density function. To overcome the limitations of Gaussian Process (GP) regression, we employ Random Feature GP (RFGP) and its online variant (O-RFGP) that enables online and incremental inference. By integrating these with Voronoi-based coverage control and Upper Confidence Bound (UCB) sampling strategy, a team of robots can adaptively focus on important regions while refining the learned spatial field for efficient coverage. Under mild assumptions, we provide theoretical guarantees and evaluate the framework through simulations in time-invariant scenarios. Furthermore, its effectiveness in time-varying settings is demonstrated through additional simulations and a physical experiment. 

**Abstract (ZH)**: 本文提出了一种框架，用于多机器人系统同时学习和覆盖一个由未知且可能随时间变化的密度函数描述的兴趣领域。为克服高斯过程（GP）回归的限制，我们采用了随机特征高斯过程（RFGP）及其在线变体（O-RFGP），从而实现在线和增量推理。通过将这些方法与基于Voronoi的覆盖控制和上置信边界的采样策略结合，机器人团队能够适应性地关注重要区域，同时逐步精细学习的空间场，以实现高效的覆盖。在轻微假设下，我们提供了理论保证，并通过不变时间场景的仿真评估了该框架。此外，通过额外的仿真和物理实验展示了该框架在随时间变化场景中的有效性。 

---
# Real-Time Obstacle Avoidance for a Mobile Robot Using CNN-Based Sensor Fusion 

**Title (ZH)**: 基于CNN的传感器融合的移动机器人实时避障方法 

**Authors**: Lamiaa H. Zain, Raafat E. Shalaby  

**Link**: [PDF](https://arxiv.org/pdf/2509.08095)  

**Abstract**: Obstacle avoidance is a critical component of the navigation stack required for mobile robots to operate effectively in complex and unknown environments. In this research, three end-to-end Convolutional Neural Networks (CNNs) were trained and evaluated offline and deployed on a differential-drive mobile robot for real-time obstacle avoidance to generate low-level steering commands from synchronized color and depth images acquired by an Intel RealSense D415 RGB-D camera in diverse environments. Offline evaluation showed that the NetConEmb model achieved the best performance with a notably low MedAE of $0.58 \times 10^{-3}$ rad/s. In comparison, the lighter NetEmb architecture adopted in this study, which reduces the number of trainable parameters by approximately 25\% and converges faster, produced comparable results with an RMSE of $21.68 \times 10^{-3}$ rad/s, close to the $21.42 \times 10^{-3}$ rad/s obtained by NetConEmb. Real-time navigation further confirmed NetConEmb's robustness, achieving a 100\% success rate in both known and unknown environments, while NetEmb and NetGated succeeded only in navigating the known environment. 

**Abstract (ZH)**: 移动机器人在复杂未知环境中有效操作所必需的导航堆栈中的避障是一个关键组成部分。本研究中，三种端到端卷积神经网络（CNN）在不同环境中进行了离线训练、评估，并部署在差速驱动移动机器人上，用于生成从Intel RealSense D415 RGB-D相机同步获取的彩色和深度图像中提取的低级转向指令，以实现实时避障。离线评估显示，NetConEmb模型取得了最佳性能，中位绝对误差（MedAE）为$0.58 \times 10^{-3}$ rad/s。相比之下，本研究中采用的较轻的NetEmb架构通过减少约25%的可训练参数并更快收敛，产生的结果与NetConEmb的均方根误差（RMSE）$21.68 \times 10^{-3}$ rad/s相当，接近NetConEmb的$21.42 \times 10^{-3}$ rad/s。进一步的实时导航证实了NetConEmb的鲁棒性，在已知和未知环境中均实现了100%的成功率，而NetEmb和NetGated仅成功导航了已知环境。 

---
# SVN-ICP: Uncertainty Estimation of ICP-based LiDAR Odometry using Stein Variational Newton 

**Title (ZH)**: SVN-ICP：基于Stein Variational Newton的ICP里程计不确定性估计 

**Authors**: Shiping Ma, Haoming Zhang, Marc Toussaint  

**Link**: [PDF](https://arxiv.org/pdf/2509.08069)  

**Abstract**: This letter introduces SVN-ICP, a novel Iterative Closest Point (ICP) algorithm with uncertainty estimation that leverages Stein Variational Newton (SVN) on manifold. Designed specifically for fusing LiDAR odometry in multisensor systems, the proposed method ensures accurate pose estimation and consistent noise parameter inference, even in LiDAR-degraded environments. By approximating the posterior distribution using particles within the Stein Variational Inference framework, SVN-ICP eliminates the need for explicit noise modeling or manual parameter tuning. To evaluate its effectiveness, we integrate SVN-ICP into a simple error-state Kalman filter alongside an IMU and test it across multiple datasets spanning diverse environments and robot types. Extensive experimental results demonstrate that our approach outperforms best-in-class methods on challenging scenarios while providing reliable uncertainty estimates. 

**Abstract (ZH)**: SVN-ICP：一种基于流形上的Stein Variational Newton的新型ICP算法及其不确定性估计 

---
# PySensors 2.0: A Python Package for Sparse Sensor Placement 

**Title (ZH)**: PySensors 2.0: 一个用于稀疏传感器布放的Python软件包 

**Authors**: Niharika Karnik, Yash Bhangale, Mohammad G. Abdo, Andrei A. Klishin, Joshua J. Cogliati, Bingni W. Brunton, J. Nathan Kutz, Steven L. Brunton, Krithika Manohar  

**Link**: [PDF](https://arxiv.org/pdf/2509.08017)  

**Abstract**: PySensors is a Python package for selecting and placing a sparse set of sensors for reconstruction and classification tasks. In this major update to \texttt{PySensors}, we introduce spatially constrained sensor placement capabilities, allowing users to enforce constraints such as maximum or exact sensor counts in specific regions, incorporate predetermined sensor locations, and maintain minimum distances between sensors. We extend functionality to support custom basis inputs, enabling integration of any data-driven or spectral basis. We also propose a thermodynamic approach that goes beyond a single ``optimal'' sensor configuration and maps the complete landscape of sensor interactions induced by the training data. This comprehensive view facilitates integration with external selection criteria and enables assessment of sensor replacement impacts. The new optimization technique also accounts for over- and under-sampling of sensors, utilizing a regularized least squares approach for robust reconstruction. Additionally, we incorporate noise-induced uncertainty quantification of the estimation error and provide visual uncertainty heat maps to guide deployment decisions. To highlight these additions, we provide a brief description of the mathematical algorithms and theory underlying these new capabilities. We demonstrate the usage of new features with illustrative code examples and include practical advice for implementation across various application domains. Finally, we outline a roadmap of potential extensions to further enhance the package's functionality and applicability to emerging sensing challenges. 

**Abstract (ZH)**: PySensors是用于选择和放置稀疏传感器集的Python包，适用于重建和分类任务。在PySensors的重大更新中，我们引入了空间约束传感器放置能力，允许用户施加诸如特定区域内的最大或精确传感器数量等约束，整合预定的传感器位置，并保持传感器之间的最小距离。我们扩展了功能以支持自定义基输入，使任何数据驱动或光谱基的集成成为可能。我们还提出了一种热力学方法，该方法超越了单一“最优”传感器配置，并映射了由训练数据引起的传感器交互的完整景观。这一全面视角有助于与外部选择标准的集成，并能够评估传感器替换的影响。新优化技术还考虑了传感器的过采样和欠采样问题，采用正则化最小二乘法以实现稳健的重建。此外，我们还纳入了由噪声引起的估计误差的不确定性量化，并提供了可视化不确定性热图以指导部署决策。为了突显这些新增功能，我们简要描述了支撑这些新功能的数学算法和理论。我们通过示例代码展示了新功能的使用方法，并提供了针对各种应用领域实施的实用建议。最后，我们概述了潜在扩展的路线图，以进一步增强包的功能，并使其适应新兴的传感挑战。 

---
# A Novel Theoretical Approach on Micro-Nano Robotic Networks Based on Density Matrices and Swarm Quantum Mechanics 

**Title (ZH)**: 基于密度矩阵和群量子力学的新型微纳米机器人网络理论方法 

**Authors**: Maria Mannone, Mahathi Anand, Peppino Fazio, Abdalla Swikir  

**Link**: [PDF](https://arxiv.org/pdf/2509.08002)  

**Abstract**: In a robotic swarm, parameters such as position and proximity to the target can be described in terms of probability amplitudes. This idea led to recent studies on a quantum approach to the definition of the swarm, including a block-matrix representation. Here, we propose an advancement of the idea, defining a swarm as a mixed quantum state, to be described with a density matrix, whose size does not change with the number of robots. We end the article with some directions for future research. 

**Abstract (ZH)**: 在机器人集群中，位置和目标的接近程度等参数可以用概率幅来描述。这一思想导致了对基于量子方法定义集群的研究，包括使用块矩阵表示法。在此，我们提出进一步的发展，将集群定义为混合量子态，并用密度矩阵描述，其尺寸不随机器人数量的变化而变化。文章最后提出了未来研究的方向。 

---
# Planar Juggling of a Devil-Stick using Discrete VHCs 

**Title (ZH)**: 使用离散VHCs的平面 Baton 魔术抛接 

**Authors**: Aakash Khandelwal, Ranjan Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.08085)  

**Abstract**: Planar juggling of a devil-stick using impulsive inputs is addressed using the concept of discrete virtual holonomic constraints (DVHC). The location of the center-of-mass of the devil-stick is specified in terms of its orientation at the discrete instants when impulsive control inputs are applied. The discrete zero dynamics (DZD) resulting from the choice of DVHC provides conditions for stable juggling. A control design that enforces the DVHC and an orbit stabilizing controller are presented. The approach is validated in simulation. 

**Abstract (ZH)**: 使用离散虚拟 holonomic 约束（DVHC）的鞭花样棒平面杂耍的冲量控制 

---
# Learning-Based Planning for Improving Science Return of Earth Observation Satellites 

**Title (ZH)**: 基于学习的规划以提高地球观测卫星的科学回报 

**Authors**: Abigail Breitfeld, Alberto Candela, Juan Delfa, Akseli Kangaslahti, Itai Zilberstein, Steve Chien, David Wettergreen  

**Link**: [PDF](https://arxiv.org/pdf/2509.07997)  

**Abstract**: Earth observing satellites are powerful tools for collecting scientific information about our planet, however they have limitations: they cannot easily deviate from their orbital trajectories, their sensors have a limited field of view, and pointing and operating these sensors can take a large amount of the spacecraft's resources. It is important for these satellites to optimize the data they collect and include only the most important or informative measurements. Dynamic targeting is an emerging concept in which satellite resources and data from a lookahead instrument are used to intelligently reconfigure and point a primary instrument. Simulation studies have shown that dynamic targeting increases the amount of scientific information gathered versus conventional sampling strategies. In this work, we present two different learning-based approaches to dynamic targeting, using reinforcement and imitation learning, respectively. These learning methods build on a dynamic programming solution to plan a sequence of sampling locations. We evaluate our approaches against existing heuristic methods for dynamic targeting, showing the benefits of using learning for this application. Imitation learning performs on average 10.0\% better than the best heuristic method, while reinforcement learning performs on average 13.7\% better. We also show that both learning methods can be trained effectively with relatively small amounts of data. 

**Abstract (ZH)**: 地球观测卫星是收集关于我们星球的科学信息的强大工具，然而它们有局限性：难以偏离轨道轨迹，传感器的视野有限，对这些传感器的操作和瞄准消耗大量的航天器资源。这些卫星需要优化他们收集的数据，仅包括最重要的或最有信息性的测量。动态瞄准是一个新兴的概念，利用卫星资源和前瞻仪器的数据智能重新配置和瞄准主要仪器。仿真研究表明，动态瞄准在科学信息收集方面优于传统抽样策略。在本工作中，我们分别使用强化学习和 imitation 学习提出了两种不同的动态瞄准学习方法。这两种学习方法建立在动态规划解决方案之上，用于计划一系列采样位置。我们将我们的方法与现有动态瞄准启发式方法进行了比较，展示了使用学习方法的好处。imitation 学习在平均上比最佳启发式方法好 10.0%，而强化学习在平均上好 13.7%。我们还展示了这两种学习方法可以用相对较少的数据有效地进行训练。 

---
# 3D and 4D World Modeling: A Survey 

**Title (ZH)**: 三维和四维世界建模：一个综述 

**Authors**: Lingdong Kong, Wesley Yang, Jianbiao Mei, Youquan Liu, Ao Liang, Dekai Zhu, Dongyue Lu, Wei Yin, Xiaotao Hu, Mingkai Jia, Junyuan Deng, Kaiwen Zhang, Yang Wu, Tianyi Yan, Shenyuan Gao, Song Wang, Linfeng Li, Liang Pan, Yong Liu, Jianke Zhu, Wei Tsang Ooi, Steven C.H. Hoi, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.07996)  

**Abstract**: World modeling has become a cornerstone in AI research, enabling agents to understand, represent, and predict the dynamic environments they inhabit. While prior work largely emphasizes generative methods for 2D image and video data, they overlook the rapidly growing body of work that leverages native 3D and 4D representations such as RGB-D imagery, occupancy grids, and LiDAR point clouds for large-scale scene modeling. At the same time, the absence of a standardized definition and taxonomy for ``world models'' has led to fragmented and sometimes inconsistent claims in the literature. This survey addresses these gaps by presenting the first comprehensive review explicitly dedicated to 3D and 4D world modeling and generation. We establish precise definitions, introduce a structured taxonomy spanning video-based (VideoGen), occupancy-based (OccGen), and LiDAR-based (LiDARGen) approaches, and systematically summarize datasets and evaluation metrics tailored to 3D/4D settings. We further discuss practical applications, identify open challenges, and highlight promising research directions, aiming to provide a coherent and foundational reference for advancing the field. A systematic summary of existing literature is available at this https URL 

**Abstract (ZH)**: 3D和4D世界建模与生成综述 

---
