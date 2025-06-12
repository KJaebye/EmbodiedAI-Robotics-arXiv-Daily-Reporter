# eFlesh: Highly customizable Magnetic Touch Sensing using Cut-Cell Microstructures 

**Title (ZH)**: eFlesh: 高度可定制的切割细胞微结构磁触觉 sensing 

**Authors**: Venkatesh Pattabiraman, Zizhou Huang, Daniele Panozzo, Denis Zorin, Lerrel Pinto, Raunaq Bhirangi  

**Link**: [PDF](https://arxiv.org/pdf/2506.09994)  

**Abstract**: If human experience is any guide, operating effectively in unstructured environments -- like homes and offices -- requires robots to sense the forces during physical interaction. Yet, the lack of a versatile, accessible, and easily customizable tactile sensor has led to fragmented, sensor-specific solutions in robotic manipulation -- and in many cases, to force-unaware, sensorless approaches. With eFlesh, we bridge this gap by introducing a magnetic tactile sensor that is low-cost, easy to fabricate, and highly customizable. Building an eFlesh sensor requires only four components: a hobbyist 3D printer, off-the-shelf magnets (<$5), a CAD model of the desired shape, and a magnetometer circuit board. The sensor is constructed from tiled, parameterized microstructures, which allow for tuning the sensor's geometry and its mechanical response. We provide an open-source design tool that converts convex OBJ/STL files into 3D-printable STLs for fabrication. This modular design framework enables users to create application-specific sensors, and to adjust sensitivity depending on the task. Our sensor characterization experiments demonstrate the capabilities of eFlesh: contact localization RMSE of 0.5 mm, and force prediction RMSE of 0.27 N for normal force and 0.12 N for shear force. We also present a learned slip detection model that generalizes to unseen objects with 95% accuracy, and visuotactile control policies that improve manipulation performance by 40% over vision-only baselines -- achieving 91% average success rate for four precise tasks that require sub-mm accuracy for successful completion. All design files, code and the CAD-to-eFlesh STL conversion tool are open-sourced and available on this https URL. 

**Abstract (ZH)**: 基于磁性触觉传感器eFlesh的有效操作无结构环境的机器人技术 

---
# Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation 

**Title (ZH)**: 基于操作链的轨迹自回归建模：应用于机器人操作 задачnement
user
基于操作链的轨迹自回归建模：应用于机器人操作 

**Authors**: Wenbo Zhang, Tianrun Hu, Yanyuan Qiao, Hanbo Zhang, Yuchu Qin, Yang Li, Jiajun Liu, Tao Kong, Lingqiao Liu, Xiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.09990)  

**Abstract**: We present Chain-of-Action (CoA), a novel visuo-motor policy paradigm built upon Trajectory Autoregressive Modeling. Unlike conventional approaches that predict next step action(s) forward, CoA generates an entire trajectory by explicit backward reasoning with task-specific goals through an action-level Chain-of-Thought (CoT) process. This process is unified within a single autoregressive structure: (1) the first token corresponds to a stable keyframe action that encodes the task-specific goals; and (2) subsequent action tokens are generated autoregressively, conditioned on the initial keyframe and previously predicted actions. This backward action reasoning enforces a global-to-local structure, allowing each local action to be tightly constrained by the final goal. To further realize the action reasoning structure, CoA incorporates four complementary designs: continuous action token representation; dynamic stopping for variable-length trajectory generation; reverse temporal ensemble; and multi-token prediction to balance action chunk modeling with global structure. As a result, CoA gives strong spatial generalization capabilities while preserving the flexibility and simplicity of a visuo-motor policy. Empirically, we observe CoA achieves the state-of-the-art performance across 60 RLBench tasks and 8 real-world manipulation tasks. 

**Abstract (ZH)**: 我们提出Chain-of-Action (CoA)，这是一种基于轨迹自回归建模的新颖视觉-运动政策范式。与传统的预测下一步动作的方法不同，CoA 通过任务特定的目标进行显式的反向推理，生成整个轨迹，过程中的每一步动作都在一个统一的自回归结构中生成：（1）第一个标记对应一个稳定的基帧动作，编码任务特定的目标；（2）后续的动作标记基于初始基帧和之前预测的动作，自回归生成。这种反向动作推理建立了全局到局部的结构，使每个局部动作都能紧密地受限于最终目标。为了进一步实现动作推理结构，CoA 结合了四种补充设计：连续动作标记表示、动态停止生成可变长度的轨迹、逆时间集成以及多标记预测以平衡动作片段建模与全局结构的平衡。因此，CoA 在保持视觉-运动政策的灵活性和简单性的同时，提供了强大的空间泛化能力。实验证明，CoA 在 60 个 RLBench 任务和 8 个现实世界的操作任务中达到了最先进的性能。 

---
# Locomotion on Constrained Footholds via Layered Architectures and Model Predictive Control 

**Title (ZH)**: 基于分层架构和模型预测控制的受限制支撑点上的运动控制 

**Authors**: Zachary Olkin, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2506.09979)  

**Abstract**: Computing stabilizing and optimal control actions for legged locomotion in real time is difficult due to the nonlinear, hybrid, and high dimensional nature of these robots. The hybrid nature of the system introduces a combination of discrete and continuous variables which causes issues for numerical optimal control. To address these challenges, we propose a layered architecture that separates the choice of discrete variables and a smooth Model Predictive Controller (MPC). The layered formulation allows for online flexibility and optimality without sacrificing real-time performance through a combination of gradient-free and gradient-based methods. The architecture leverages a sampling-based method for determining discrete variables, and a classical smooth MPC formulation using these fixed discrete variables. We demonstrate the results on a quadrupedal robot stepping over gaps and onto terrain with varying heights. In simulation, we demonstrate the controller on a humanoid robot for gap traversal. The layered approach is shown to be more optimal and reliable than common heuristic-based approaches and faster to compute than pure sampling methods. 

**Abstract (ZH)**: 实时计算支撑腿式移动的稳定和最优控制动作因系统非线性、混合性和高维性而具有挑战性。为应对这些挑战，我们提出了一种分层架构，该架构将离散变量的选择与平滑模型预测控制（MPC）分开。分层表示法通过结合无导数方法和导数方法，在不牺牲实时性能的情况下提供在线灵活性和最优性。该架构利用基于采样的方法确定离散变量，并使用固定离散变量的经典平滑MPC表示法。我们在一个四足机器人跨越缺口和踏上不同高度地形的实验中展示了该方法。在模拟中，我们在一个类人机器人上展示了控制器进行跨越缺口的控制。分层方法在最优性和可靠性方面优于常用的经验方法，并且计算速度比纯粹基于采样的方法更快。 

---
# SAFE: Multitask Failure Detection for Vision-Language-Action Models 

**Title (ZH)**: SAFE：面向视觉-语言-动作模型的多任务故障检测 

**Authors**: Qiao Gu, Yuanliang Ju, Shengxiang Sun, Igor Gilitschenski, Haruki Nishimura, Masha Itkina, Florian Shkurti  

**Link**: [PDF](https://arxiv.org/pdf/2506.09937)  

**Abstract**: While vision-language-action models (VLAs) have shown promising robotic behaviors across a diverse set of manipulation tasks, they achieve limited success rates when deployed on novel tasks out-of-the-box. To allow these policies to safely interact with their environments, we need a failure detector that gives a timely alert such that the robot can stop, backtrack, or ask for help. However, existing failure detectors are trained and tested only on one or a few specific tasks, while VLAs require the detector to generalize and detect failures also in unseen tasks and novel environments. In this paper, we introduce the multitask failure detection problem and propose SAFE, a failure detector for generalist robot policies such as VLAs. We analyze the VLA feature space and find that VLAs have sufficient high-level knowledge about task success and failure, which is generic across different tasks. Based on this insight, we design SAFE to learn from VLA internal features and predict a single scalar indicating the likelihood of task failure. SAFE is trained on both successful and failed rollouts, and is evaluated on unseen tasks. SAFE is compatible with different policy architectures. We test it on OpenVLA, $\pi_0$, and $\pi_0$-FAST in both simulated and real-world environments extensively. We compare SAFE with diverse baselines and show that SAFE achieves state-of-the-art failure detection performance and the best trade-off between accuracy and detection time using conformal prediction. More qualitative results can be found at this https URL. 

**Abstract (ZH)**: Vision-Language-Action模型多任务故障检测问题及SAFE故障检测器 

---
# Fluoroscopic Shape and Pose Tracking of Catheters with Custom Radiopaque Markers 

**Title (ZH)**: 带有定制放射不透明标志的导管透视形状和姿态跟踪 

**Authors**: Jared Lawson, Rohan Chitale, Nabil Simaan  

**Link**: [PDF](https://arxiv.org/pdf/2506.09934)  

**Abstract**: Safe navigation of steerable and robotic catheters in the cerebral vasculature requires awareness of the catheters shape and pose. Currently, a significant perception burden is placed on interventionalists to mentally reconstruct and predict catheter motions from biplane fluoroscopy images. Efforts to track these catheters are limited to planar segmentation or bulky sensing instrumentation, which are incompatible with microcatheters used in neurointervention. In this work, a catheter is equipped with custom radiopaque markers arranged to enable simultaneous shape and pose estimation under biplane fluoroscopy. A design measure is proposed to guide the arrangement of these markers to minimize sensitivity to marker tracking uncertainty. This approach was deployed for microcatheters smaller than 2mm OD navigating phantom vasculature with shape tracking errors less than 1mm and catheter roll errors below 40 degrees. This work can enable steerable catheters to autonomously navigate under biplane imaging. 

**Abstract (ZH)**: 基于双平面成像的可操控和机器人导管在脑血管内的安全导航需要对导管的形状和姿态有所认识。目前，介入医师需通过心理重构和预测从双平面透视图像中推断导管运动，面临显著的感知负担。当前用于跟踪此类导管的努力仅限于平面分割或笨重的传感装置，这些方法不适用于神经介入中常用的微导管。本研究中，导管配备了自定义的放射-opacity标记，以便在双平面透视下同时进行形状和姿态估计。提出了一种设计措施来指导这些标记的排列，以最小化标记跟踪不确定性的影响。该方法在OD小于2mm的微导管在仿真血管中导航时表现出优异的形状跟踪误差（小于1mm）和导管滚转误差（低于40度）性能，从而使得可操控导管能够在双平面成像下实现自主导航。 

---
# From Intention to Execution: Probing the Generalization Boundaries of Vision-Language-Action Models 

**Title (ZH)**: 从意图到执行：探究视觉-语言-行动模型的泛化边界 

**Authors**: Irving Fang, Juexiao Zhang, Shengbang Tong, Chen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.09930)  

**Abstract**: One promise that Vision-Language-Action (VLA) models hold over traditional imitation learning for robotics is to leverage the broad generalization capabilities of large Vision-Language Models (VLMs) to produce versatile, "generalist" robot policies. However, current evaluations of VLAs remain insufficient. Traditional imitation learning benchmarks are unsuitable due to the lack of language instructions. Emerging benchmarks for VLAs that incorporate language often come with limited evaluation tasks and do not intend to investigate how much VLM pretraining truly contributes to the generalization capabilities of the downstream robotic policy. Meanwhile, much research relies on real-world robot setups designed in isolation by different institutions, which creates a barrier for reproducibility and accessibility. To address this gap, we introduce a unified probing suite of 50 simulation-based tasks across 10 subcategories spanning language instruction, vision, and objects. We systematically evaluate several state-of-the-art VLA architectures on this suite to understand their generalization capability. Our results show that while VLM backbones endow VLAs with robust perceptual understanding and high level planning, which we refer to as good intentions, this does not reliably translate into precise motor execution: when faced with out-of-distribution observations, policies often exhibit coherent intentions, but falter in action execution. Moreover, finetuning on action data can erode the original VLM's generalist reasoning abilities. We release our task suite and evaluation code to serve as a standardized benchmark for future VLAs and to drive research on closing the perception-to-action gap. More information, including the source code, can be found at this https URL 

**Abstract (ZH)**: Vision-Language-Action模型在机器人领域超越传统模仿学习的 promise在于利用大体量视觉-语言模型的广泛泛化能力生成多用途的“通才”机器人策略，但当前对Vision-Language-Action (VLA)模型的评估仍显不足。传统的模仿学习基准由于缺乏语言指令而不适用。新兴的VLA基准虽然包含了语言，但评估任务有限，并未深入探究VLM预训练对下游机器人策略泛化能力的实际贡献。同时，许多研究依赖于不同机构独立设计的现实世界机器人设置，这造成了可重复性和可访问性的障碍。为解决这一问题，我们引入了一个统一的探针套件，包括涵盖语言指令、视觉和物体在内的10个子类别的50项仿真任务。我们系统性地评估了几种最先进的VLA架构，以了解其泛化能力。结果显示，尽管VLM骨干网络赋予了VLA模型强大的感知理解和高层规划能力，即所谓的“好意图”，但这并不一定能可靠地转化为精准的行动执行：当面对分布外的观察时，策略常常表现出一致的意图，但在行动执行方面却失败了。此外，通过行动数据进行微调可能会削弱原始VLM的通用推理能力。我们发布了该任务套件和评估代码，以作为未来VLA的标准化基准，并推动缩小感知到行动差距的研究。更多信息，包括源代码，可在以下链接获取。 

---
# From Theory to Practice: Advancing Multi-Robot Path Planning Algorithms and Applications 

**Title (ZH)**: 从理论到实践：推动多机器人路径规划算法及其应用的发展 

**Authors**: Teng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09914)  

**Abstract**: The labeled MRPP (Multi-Robot Path Planning) problem involves routing robots from start to goal configurations efficiently while avoiding collisions. Despite progress in solution quality and runtime, its complexity and industrial relevance continue to drive research.
This dissertation introduces scalable MRPP methods with provable guarantees and practical heuristics. First, we study dense MRPP on 2D grids, relevant to warehouse and parcel systems. We propose the Rubik Table method, achieving $(1 + \delta)$-optimal makespan (with $\delta \in (0, 0.5]$) for up to $\frac{m_1 m_2}{2}$ robots, solving large instances efficiently and setting a new theoretical benchmark.
Next, we address real-world MRPP. We design optimal layouts for structured environments (e.g., warehouses, parking systems) and propose a puzzle-based system for dense, deadlock-free autonomous vehicle parking. We also extend MRPP to Reeds-Shepp robots, introducing motion primitives and smoothing techniques to ensure feasible, efficient paths under nonholonomic constraints. Simulations and real-world tests validate the approach in urban driving and robotic transport scenarios. 

**Abstract (ZH)**: 带标注的多机器人路径规划问题涉及在避免碰撞的情况下，高效地将机器人从起始配置引导至目标配置。尽管在解决方案质量和运行时间方面取得了进展，但其复杂性和工业相关性仍继续推动研究。

本论文介绍了具有可验证保证和实用启发式的可扩展多机器人路径规划方法。首先，我们研究了密集型2D网格上的多机器人路径规划，这与仓库和包裹系统相关。我们提出了Rubik Table方法，实现了最多$\frac{m_1 m_2}{2}$个机器人在$(1 + \delta)$-最优最短工期（$\delta \in (0, 0.5]$）下的解，有效解决大规模实例，并建立了新的理论基准。
接下来，我们解决了实际的多机器人路径规划问题。我们为结构化环境设计了最优布局（例如，仓库、停车系统），并提出了一种基于拼图的系统，用于密集、无死锁的自主车辆停车。我们还扩展了多机器人路径规划到Reeds-Shepp机器人，引入了运动原语和平滑技术，以确保在非完整约束下可行且高效的路径。仿真实验和实际测试验证了在城市驾驶和机器人运输场景中的方法。 

---
# Aucamp: An Underwater Camera-Based Multi-Robot Platform with Low-Cost, Distributed, and Robust Localization 

**Title (ZH)**: Aucamp：一种基于水下摄像头的低成本、分布式和稳健的多机器人平台 

**Authors**: Jisheng Xu, Ding Lin, Pangkit Fong, Chongrong Fang, Xiaoming Duan, Jianping He  

**Link**: [PDF](https://arxiv.org/pdf/2506.09876)  

**Abstract**: This paper introduces an underwater multi-robot platform, named Aucamp, characterized by cost-effective monocular-camera-based sensing, distributed protocol and robust orientation control for localization. We utilize the clarity feature to measure the distance, present the monocular imaging model, and estimate the position of the target object. We achieve global positioning in our platform by designing a distributed update protocol. The distributed algorithm enables the perception process to simultaneously cover a broader range, and greatly improves the accuracy and robustness of the positioning. Moreover, the explicit dynamics model of the robot in our platform is obtained, based on which, we propose a robust orientation control framework. The control system ensures that the platform maintains a balanced posture for each robot, thereby ensuring the stability of the localization system. The platform can swiftly recover from an forced unstable state to a stable horizontal posture. Additionally, we conduct extensive experiments and application scenarios to evaluate the performance of our platform. The proposed new platform may provide support for extensive marine exploration by underwater sensor networks. 

**Abstract (ZH)**: 一种基于单目摄像头传感的分布式协议和 robust 方向控制的水下多机器人平台：Aucamp 

---
# Hierarchical Learning-Enhanced MPC for Safe Crowd Navigation with Heterogeneous Constraints 

**Title (ZH)**: 层次学习增强的 MPC 在异构约束下的安全人群导航 

**Authors**: Huajian Liu, Yixuan Feng, Wei Dong, Kunpeng Fan, Chao Wang, Yongzhuo Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.09859)  

**Abstract**: In this paper, we propose a novel hierarchical framework for robot navigation in dynamic environments with heterogeneous constraints. Our approach leverages a graph neural network trained via reinforcement learning (RL) to efficiently estimate the robot's cost-to-go, formulated as local goal recommendations. A spatio-temporal path-searching module, which accounts for kinematic constraints, is then employed to generate a reference trajectory to facilitate solving the non-convex optimization problem used for explicit constraint enforcement. More importantly, we introduce an incremental action-masking mechanism and a privileged learning strategy, enabling end-to-end training of the proposed planner. Both simulation and real-world experiments demonstrate that the proposed method effectively addresses local planning in complex dynamic environments, achieving state-of-the-art (SOTA) performance. Compared with existing learning-optimization hybrid methods, our approach eliminates the dependency on high-fidelity simulation environments, offering significant advantages in computational efficiency and training scalability. The code will be released as open-source upon acceptance of the paper. 

**Abstract (ZH)**: 本文提出了一种新颖的层次框架，用于动态环境中异构约束下的机器人导航。该方法利用通过强化学习（RL）训练的图神经网络高效估计机器人的成本到底，并将其形式化为局部目标推荐。随后采用一个考虑运动学约束的空间-时间路径搜索模块生成参考轨迹，以辅助解决用于显式约束执行的非凸优化问题。更重要的是，我们引入了增量动作遮罩机制和特权学习策略，使所提出的规划器能够端到端地训练。仿真和现实世界实验表明，所提出的方法有效地解决了复杂动态环境下的局部规划问题，达到了目前最先进的性能。与现有的学习-优化混合方法相比，我们的方法消除了对高保真仿真环境的依赖，提供了显著的计算效率和训练可扩展性优势。论文被接受后，代码将被公开发布。 

---
# Reinforced Refinement with Self-Aware Expansion for End-to-End Autonomous Driving 

**Title (ZH)**: 带有自我意识扩张的强化细化方法用于端到端自主驾驶 

**Authors**: Haochen Liu, Tianyu Li, Haohan Yang, Li Chen, Caojun Wang, Ke Guo, Haochen Tian, Hongchen Li, Hongyang Li, Chen Lv  

**Link**: [PDF](https://arxiv.org/pdf/2506.09800)  

**Abstract**: End-to-end autonomous driving has emerged as a promising paradigm for directly mapping sensor inputs to planning maneuvers using learning-based modular integrations. However, existing imitation learning (IL)-based models suffer from generalization to hard cases, and a lack of corrective feedback loop under post-deployment. While reinforcement learning (RL) offers a potential solution to tackle hard cases with optimality, it is often hindered by overfitting to specific driving cases, resulting in catastrophic forgetting of generalizable knowledge and sample inefficiency. To overcome these challenges, we propose Reinforced Refinement with Self-aware Expansion (R2SE), a novel learning pipeline that constantly refines hard domain while keeping generalizable driving policy for model-agnostic end-to-end driving systems. Through reinforcement fine-tuning and policy expansion that facilitates continuous improvement, R2SE features three key components: 1) Generalist Pretraining with hard-case allocation trains a generalist imitation learning (IL) driving system while dynamically identifying failure-prone cases for targeted refinement; 2) Residual Reinforced Specialist Fine-tuning optimizes residual corrections using reinforcement learning (RL) to improve performance in hard case domain while preserving global driving knowledge; 3) Self-aware Adapter Expansion dynamically integrates specialist policies back into the generalist model, enhancing continuous performance improvement. Experimental results in closed-loop simulation and real-world datasets demonstrate improvements in generalization, safety, and long-horizon policy robustness over state-of-the-art E2E systems, highlighting the effectiveness of reinforce refinement for scalable autonomous driving. 

**Abstract (ZH)**: 端到端自主驾驶：基于强化修正与自我意识扩展的学习框架 

---
# Learning to Optimize Package Picking for Large-Scale, Real-World Robot Induction 

**Title (ZH)**: 学习优化大规模现实世界机器人装箱捡取 

**Authors**: Shuai Li, Azarakhsh Keipour, Sicong Zhao, Srinath Rajagopalan, Charles Swan, Kostas E. Bekris  

**Link**: [PDF](https://arxiv.org/pdf/2506.09765)  

**Abstract**: Warehouse automation plays a pivotal role in enhancing operational efficiency, minimizing costs, and improving resilience to workforce variability. While prior research has demonstrated the potential of machine learning (ML) models to increase picking success rates in large-scale robotic fleets by prioritizing high-probability picks and packages, these efforts primarily focused on predicting success probabilities for picks sampled using heuristic methods. Limited attention has been given, however, to leveraging data-driven approaches to directly optimize sampled picks for better performance at scale. In this study, we propose an ML-based framework that predicts transform adjustments as well as improving the selection of suction cups for multi-suction end effectors for sampled picks to enhance their success probabilities. The framework was integrated and evaluated in test workcells that resemble the operations of Amazon Robotics' Robot Induction (Robin) fleet, which is used for package manipulation. Evaluated on over 2 million picks, the proposed method achieves a 20\% reduction in pick failure rates compared to a heuristic-based pick sampling baseline, demonstrating its effectiveness in large-scale warehouse automation scenarios. 

**Abstract (ZH)**: 仓库自动化在提高运营效率、降低成本和增强对劳动力变动的抵御能力中发挥着关键作用。尽管先前研究已经表明，机器学习模型可以通过优先处理高概率 pick 和包裹来提高大型机器人队列的拣选成功率，但这些努力主要集中在使用启发式方法采样的拣选的成功概率预测上。然而，利用数据驱动的方法直接优化采样的拣选以在大规模场景中获得更好表现的关注较少。在本研究中，我们提出了一种基于机器学习的框架，该框架预测变换调整并改进多吸盘末端执行器的吸盘选择，以提高采样拣选的成功概率。该框架在类似亚马逊机器人Robot Induction（Robin）队列操作的测试工位中进行集成和评估，用于包裹操作。在超过200万次拣选的评估中，所提出的方法将拣选失败率降低了20%，证明了其在大规模仓库自动化场景中的有效性。 

---
# Human-robot collaborative transport personalization via Dynamic Movement Primitives and velocity scaling 

**Title (ZH)**: 基于动态运动 primitives 和速度缩放的人机协作个性化运输 

**Authors**: Paolo Franceschi, Andrea Bussolan, Vincenzo Pomponi, Oliver Avram, Stefano Baraldo, Anna Valente  

**Link**: [PDF](https://arxiv.org/pdf/2506.09697)  

**Abstract**: Nowadays, industries are showing a growing interest in human-robot collaboration, particularly for shared tasks. This requires intelligent strategies to plan a robot's motions, considering both task constraints and human-specific factors such as height and movement preferences. This work introduces a novel approach to generate personalized trajectories using Dynamic Movement Primitives (DMPs), enhanced with real-time velocity scaling based on human feedback. The method was rigorously tested in industrial-grade experiments, focusing on the collaborative transport of an engine cowl lip section. Comparative analysis between DMP-generated trajectories and a state-of-the-art motion planner (BiTRRT) highlights their adaptability combined with velocity scaling. Subjective user feedback further demonstrates a clear preference for DMP- based interactions. Objective evaluations, including physiological measurements from brain and skin activity, reinforce these findings, showcasing the advantages of DMPs in enhancing human-robot interaction and improving user experience. 

**Abstract (ZH)**: 现在，各行业对人机协作表现出 growing 的兴趣，特别是在共享任务方面。这需要智能策略来计划机器人的动作，考虑任务约束和人类特定因素，如身高和运动偏好。本工作介绍了一种使用动态运动本征（DMPs）生成个性化轨迹的新方法，并通过实时速度缩放增强，基于人类反馈。该方法在工业级实验中得到了严格的测试，重点关注发动机整流罩唇部段的协作运输。DMP生成的轨迹与当前最先进的运动规划器（BiTRRT）的比较分析突显了它们的适应性和速度缩放组合。主观用户反馈进一步表明了对基于DMP的交互的明显偏好。客观评估，包括大脑和皮肤活动的生理测量，强化了这些发现，展示了DMPs在增强人机交互和提高用户体验方面的优势。 

---
# R-CARLA: High-Fidelity Sensor Simulations with Interchangeable Dynamics for Autonomous Racing 

**Title (ZH)**: R-CARLA: 具有可互换动力学的高保真传感器模拟在自动驾驶赛车中的应用 

**Authors**: Maurice Brunner, Edoardo Ghignone, Nicolas Baumann, Michele Magno  

**Link**: [PDF](https://arxiv.org/pdf/2506.09629)  

**Abstract**: Autonomous racing has emerged as a crucial testbed for autonomous driving algorithms, necessitating a simulation environment for both vehicle dynamics and sensor behavior. Striking the right balance between vehicle dynamics and sensor accuracy is crucial for pushing vehicles to their performance limits. However, autonomous racing developers often face a trade-off between accurate vehicle dynamics and high-fidelity sensor simulations. This paper introduces R-CARLA, an enhancement of the CARLA simulator that supports holistic full-stack testing, from perception to control, using a single system. By seamlessly integrating accurate vehicle dynamics with sensor simulations, opponents simulation as NPCs, and a pipeline for creating digital twins from real-world robotic data, R-CARLA empowers researchers to push the boundaries of autonomous racing development. Furthermore, it is developed using CARLA's rich suite of sensor simulations. Our results indicate that incorporating the proposed digital-twin framework into R-CARLA enables more realistic full-stack testing, demonstrating a significant reduction in the Sim-to-Real gap of car dynamics simulation by 42% and by 82% in the case of sensor simulation across various testing scenarios. 

**Abstract (ZH)**: 自主赛车比赛已成为自主驾驶算法的关键测试平台，需要一个既支持车辆动力学又支持传感器行为的模拟环境。准确平衡车辆动力学和传感器精度对于将车辆推向性能极限至关重要。然而，自主赛车开发者往往在准确的车辆动力学和高保真传感器模拟之间面临权衡。本文介绍了R-CARLA，它是CARLA模拟器的增强版本，支持从感知到控制的全方位堆栈测试。通过无缝集成准确的车辆动力学与传感器模拟、对手模拟作为非玩家角色（NPCs）以及从真实世界机器人数据创建数字孪生的管道，R-CARLA为研究人员提供了推动自主赛车开发边界的能力。此外，R-CARLA 是基于 CARLA 丰富的传感器模拟套件开发的。我们的结果表明，将提出的数字孪生框架整合到 R-CARLA 中能够实现更现实的全方位测试，在各种测试场景中，汽车动力学模拟的Sim-to-Real差距降低了42%，传感器模拟的Sim-to-Real差距降低了82%。 

---
# Analytic Task Scheduler: Recursive Least Squares Based Method for Continual Learning in Embodied Foundation Models 

**Title (ZH)**: 解析任务调度器：基于递归最小二乘法的持续学习方法在具身基础模型中的应用 

**Authors**: Lipei Xie, Yingxin Li, Huiping Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09623)  

**Abstract**: Embodied foundation models are crucial for Artificial Intelligence (AI) interacting with the physical world by integrating multi-modal inputs, such as proprioception, vision and language, to understand human intentions and generate actions to control robots. While these models demonstrate strong generalization and few-shot learning capabilities, they face significant challenges in continually acquiring new skills without forgetting previously learned skills, a problem known as catastrophic forgetting. To address this issue, we propose the Analytic Task Scheduler (ATS), a novel framework for continual learning in embodied foundation models. ATS consists of a task-specific model library, where each model is fine-tuned independently on a single task, and an analytic scheduler trained using recursive least squares (RLS) to learn the mapping between language instructions and task-specific models. This architecture enables accurate task recognition and dynamic model selection while fundamentally avoiding parameter interference across tasks. The scheduler updates its parameters incrementally using only statistics (autocorrelation and cross-correlation matrices), enabling forgetting-resistant learning without the need to revisit historical data. We validate ATS on a real-world robot platform (RM65B), demonstrating superior resistance to forgetting and strong adaptability to task variations. The results highlight ATS as an effective, scalable, and deployable solution for continual learning in embodied foundation models operating in complex, dynamic environments. Our code will be available at this https URL 

**Abstract (ZH)**: 具身基础模型对于通过集成 proprioception、视觉和语言等多模态输入与物理世界交互的人工智能（AI）至关重要，这些模型能够理解人类意图并生成控制机器人的动作。尽管这些模型展示了强大的泛化能力和少样本学习能力，但它们在不断获取新技能时而不忘记已学习技能方面面临着重大挑战，这个问题被称为灾难性遗忘。为了应对这一问题，我们提出了一种新颖的持续学习框架——具身基础模型的分析任务调度器（Analytic Task Scheduler, ATS）。ATS 包括一个任务特定模型库，每个模型独立 fine-tune 在单一任务上，以及一个使用递归最小二乘法（Recursive Least Squares, RLS）训练的分析调度器，用于学习语言指令和任务特定模型之间的映射。该架构能够实现准确的任务识别和动态模型选择，从根本上避免了任务间参数干扰。调度器仅使用统计数据（自相关矩阵和交叉相关矩阵）逐增量更新参数，从而在无需回顾历史数据的情况下实现具有抗遗忘能力的学习。我们在实际机器人平台（RM65B）上验证了 ATS，展示了其优越的抗遗忘能力和对任务变化的良好适应性。结果表明，ATS 是一个有效的、可扩展且可部署的解决方案，适用于在复杂动态环境中操作的具身基础模型的持续学习。 

---
# Attention-Based Map Encoding for Learning Generalized Legged Locomotion 

**Title (ZH)**: 基于注意力的地图编码学习通用腿式运动控制 

**Authors**: Junzhe He, Chong Zhang, Fabian Jenelten, Ruben Grandia, Moritz BÄcher, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2506.09588)  

**Abstract**: Dynamic locomotion of legged robots is a critical yet challenging topic in expanding the operational range of mobile robots. It requires precise planning when possible footholds are sparse, robustness against uncertainties and disturbances, and generalizability across diverse terrains. While traditional model-based controllers excel at planning on complex terrains, they struggle with real-world uncertainties. Learning-based controllers offer robustness to such uncertainties but often lack precision on terrains with sparse steppable areas. Hybrid methods achieve enhanced robustness on sparse terrains by combining both methods but are computationally demanding and constrained by the inherent limitations of model-based planners. To achieve generalized legged locomotion on diverse terrains while preserving the robustness of learning-based controllers, this paper proposes to learn an attention-based map encoding conditioned on robot proprioception, which is trained as part of the end-to-end controller using reinforcement learning. We show that the network learns to focus on steppable areas for future footholds when the robot dynamically navigates diverse and challenging terrains. We synthesize behaviors that exhibit robustness against uncertainties while enabling precise and agile traversal of sparse terrains. Additionally, our method offers a way to interpret the topographical perception of a neural network. We have trained two controllers for a 12-DoF quadrupedal robot and a 23-DoF humanoid robot respectively and tested the resulting controllers in the real world under various challenging indoor and outdoor scenarios, including ones unseen during training. 

**Abstract (ZH)**: 基于注意力的地图编码在不同地形上实现腿足机器人的鲁棒动态运动 

---
# VAULT: A Mobile Mapping System for ROS 2-based Autonomous Robots 

**Title (ZH)**: VAULT：基于ROS 2的自主机器人移动 mapping 系统 

**Authors**: Miguel Á. González-Santamarta, Francisco J. Rodríguez-Lera, Vicente Matellán-Olivera  

**Link**: [PDF](https://arxiv.org/pdf/2506.09583)  

**Abstract**: Localization plays a crucial role in the navigation capabilities of autonomous robots, and while indoor environments can rely on wheel odometry and 2D LiDAR-based mapping, outdoor settings such as agriculture and forestry, present unique challenges that necessitate real-time localization and consistent mapping. Addressing this need, this paper introduces the VAULT prototype, a ROS 2-based mobile mapping system (MMS) that combines various sensors to enable robust outdoor and indoor localization. The proposed solution harnesses the power of Global Navigation Satellite System (GNSS) data, visual-inertial odometry (VIO), inertial measurement unit (IMU) data, and the Extended Kalman Filter (EKF) to generate reliable 3D odometry. To further enhance the localization accuracy, Visual SLAM (VSLAM) is employed, resulting in the creation of a comprehensive 3D point cloud map. By leveraging these sensor technologies and advanced algorithms, the prototype offers a comprehensive solution for outdoor localization in autonomous mobile robots, enabling them to navigate and map their surroundings with confidence and precision. 

**Abstract (ZH)**: 自主机器人在导航能力中，定位起着关键作用。虽然室内环境可以依赖轮式里程计和基于2D LiDAR的制图，但如农业和林业等户外环境则提出了独特挑战，需要实时定位和持续建图。为应对这一需求，本文介绍了基于ROS 2的移动建图系统（MMS）VAULT原型，该系统结合了多种传感器以实现 robust 的户外和室内定位。所提出的方法利用全球导航卫星系统（GNSS）数据、视觉惯性里程计（VIO）、惯性测量单元（IMU）数据以及扩展卡尔曼滤波器（EKF）生成可靠的3D里程计。为了进一步提高定位准确性，还采用了视觉 SLAM（VSLAM），生成了全面的3D点云地图。通过利用这些传感器技术和先进算法，该原型为自主移动机器人的户外定位提供了全面的解决方案，使它们能够在陌生环境中导航和高效建图。 

---
# Integrating Quantized LLMs into Robotics Systems as Edge AI to Leverage their Natural Language Processing Capabilities 

**Title (ZH)**: 将量化大语言模型集成到机器人系统中的边缘AI以利用其自然语言处理能力 

**Authors**: Miguel Á. González-Santamarta, Francisco J. Rodríguez-Lera, David Sobrín-Hidalgo, Ángel Manuel Guerrero-Higueras, Vicente MatellÁn-Olivera  

**Link**: [PDF](https://arxiv.org/pdf/2506.09581)  

**Abstract**: Large Language Models (LLMs) have experienced great advancements in the last year resulting in an increase of these models in several fields to face natural language tasks. The integration of these models in robotics can also help to improve several aspects such as human-robot interaction, navigation, planning and decision-making. Therefore, this paper introduces llama\_ros, a tool designed to integrate quantized Large Language Models (LLMs) into robotic systems using ROS 2. Leveraging this http URL, a highly optimized runtime engine, llama\_ros enables the efficient execution of quantized LLMs as edge artificial intelligence (AI) in robotics systems with resource-constrained environments, addressing the challenges of computational efficiency and memory limitations. By deploying quantized LLMs, llama\_ros empowers robots to leverage the natural language understanding and generation for enhanced decision-making and interaction which can be paired with prompt engineering, knowledge graphs, ontologies or other tools to improve the capabilities of autonomous robots. Additionally, this paper provides insights into some use cases of using llama\_ros for planning and explainability in robotics. 

**Abstract (ZH)**: 大型语言模型（LLMs）在过去一年中取得了显著的进步，使得这些模型在多个领域用于应对自然语言任务。将这些模型集成到机器人技术中也可以提高多个方面的表现，如人机交互、导航、规划和决策。因此，本文介绍了llama\_ros，这是一个用于使用ROS 2将量化大型语言模型（LLMs）集成到机器人系统中的工具。通过利用这个高性能运行时引擎，llama\_ros能够高效地在资源受限的环境中将量化LLMs作为边缘人工智能（AI）执行，以解决计算效率和内存限制等挑战。通过部署量化LLMs，llama\_ros使机器人能够利用自然语言理解和生成能力，增强决策和交互能力，并可与提示工程、知识图谱、本体或其他工具结合使用以提高自主机器人的能力。此外，本文还探讨了使用llama\_ros进行机器人规划和解释的应用案例。 

---
# Enhancing Human-Robot Collaboration: A Sim2Real Domain Adaptation Algorithm for Point Cloud Segmentation in Industrial Environments 

**Title (ZH)**: 增强人机协作：面向工业环境点云分割的Sim2Real领域自适应算法 

**Authors**: Fatemeh Mohammadi Amin, Darwin G. Caldwell, Hans Wernher van de Venn  

**Link**: [PDF](https://arxiv.org/pdf/2506.09552)  

**Abstract**: The robust interpretation of 3D environments is crucial for human-robot collaboration (HRC) applications, where safety and operational efficiency are paramount. Semantic segmentation plays a key role in this context by enabling a precise and detailed understanding of the environment. Considering the intense data hunger for real-world industrial annotated data essential for effective semantic segmentation, this paper introduces a pioneering approach in the Sim2Real domain adaptation for semantic segmentation of 3D point cloud data, specifically tailored for HRC. Our focus is on developing a network that robustly transitions from simulated environments to real-world applications, thereby enhancing its practical utility and impact on a safe HRC.
In this work, we propose a dual-stream network architecture (FUSION) combining Dynamic Graph Convolutional Neural Networks (DGCNN) and Convolutional Neural Networks (CNN) augmented with residual layers as a Sim2Real domain adaptation algorithm for an industrial environment. The proposed model was evaluated on real-world HRC setups and simulation industrial point clouds, it showed increased state-of-the-art performance, achieving a segmentation accuracy of 97.76%, and superior robustness compared to existing methods. 

**Abstract (ZH)**: 三维环境的稳健解释对于人机协作（HRC）应用至关重要，其中安全性和操作效率为首要。语义分割在此背景下发挥关键作用，通过使环境理解精确和详细。鉴于实际工业标注数据对于有效语义分割的极度需求，本文提出了一种在三维点云数据语义分割中用于工业环境的Sim2Real领域适应的开创性方法。我们专注于开发一个能够在模拟环境与实际应用之间稳健过渡的网络，从而提高其实用价值和在安全HRC中的影响。

在此工作中，我们提出了一种双流网络架构（FUSION），结合了动态图卷积神经网络（DGCNN）和具有残差层的卷积神经网络（CNN），作为一种工业环境的Sim2Real领域适应算法。所提出的模型在实际工业HRC设置和模拟工业点云上进行了评估，显示了比现有方法的最先进的性能，准确率达到97.76%，且具有更强的鲁棒性。 

---
# Tightly-Coupled LiDAR-IMU-Leg Odometry with Online Learned Leg Kinematics Incorporating Foot Tactile Information 

**Title (ZH)**: 结合在线学习腿部运动学的紧密耦合LiDAR-IMU-腿部里程计 

**Authors**: Taku Okawara, Kenji Koide, Aoki Takanose, Shuji Oishi, Masashi Yokozuka, Kentaro Uno, Kazuya Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2506.09548)  

**Abstract**: In this letter, we present tightly coupled LiDAR-IMU-leg odometry, which is robust to challenging conditions such as featureless environments and deformable terrains. We developed an online learning-based leg kinematics model named the neural leg kinematics model, which incorporates tactile information (foot reaction force) to implicitly express the nonlinear dynamics between robot feet and the ground. Online training of this model enhances its adaptability to weight load changes of a robot (e.g., assuming delivery or transportation tasks) and terrain conditions. According to the \textit{neural adaptive leg odometry factor} and online uncertainty estimation of the leg kinematics model-based motion predictions, we jointly solve online training of this kinematics model and odometry estimation on a unified factor graph to retain the consistency of both. The proposed method was verified through real experiments using a quadruped robot in two challenging situations: 1) a sandy beach, representing an extremely featureless area with a deformable terrain, and 2) a campus, including multiple featureless areas and terrain types of asphalt, gravel (deformable terrain), and grass. Experimental results showed that our odometry estimation incorporating the \textit{neural leg kinematics model} outperforms state-of-the-art works. Our project page is available for further details: this https URL 

**Abstract (ZH)**: 基于紧耦合LiDAR-IMU-腿的动力学模型的鲁棒腿部里程计 

---
# Advances on Affordable Hardware Platforms for Human Demonstration Acquisition in Agricultural Applications 

**Title (ZH)**: 面向农业应用的人体示范获取的经济硬件平台进展 

**Authors**: Alberto San-Miguel-Tello, Gennaro Scarati, Alejandro Hernández, Mario Cavero-Vidal, Aakash Maroti, Néstor García  

**Link**: [PDF](https://arxiv.org/pdf/2506.09494)  

**Abstract**: This paper presents advances on the Universal Manipulation Interface (UMI), a low-cost hand-held gripper for robot Learning from Demonstration (LfD), for complex in-the-wild scenarios found in agricultural settings. The focus is on improving the acquisition of suitable samples with minimal additional setup. Firstly, idle times and user's cognitive load are reduced through the extraction of individual samples from a continuous demonstration considering task events. Secondly, reliability on the generation of task sample's trajectories is increased through the combination on-board inertial measurements and external visual marker localization usage using Extended Kalman Filtering (EKF). Results are presented for a fruit harvesting task, outperforming the default pipeline. 

**Abstract (ZH)**: 这篇论文介绍了通用操作接口（UMI）的进步，UMI是一种低成本手持夹持器，用于机器人从示范学习（LfD），适用于农业环境中发现的复杂实地场景。重点是通过考虑任务事件从连续示范中提取个体样本，减少空闲时间并降低用户的认知负荷，以及通过结合机载惯性测量和外部分辨率视觉标记定位，并使用扩展卡尔曼滤波（EKF）来提高任务样本轨迹生成的可靠性。结果表明，该方法在水果采摘任务中优于默认流水线。 

---
# DCIRNet: Depth Completion with Iterative Refinement for Dexterous Grasping of Transparent and Reflective Objects 

**Title (ZH)**: DCIRNet：用于透明和反射物体灵活抓取的迭代完善深度完成 

**Authors**: Guanghu Xie, Zhiduo Jiang, Yonglong Zhang, Yang Liu, Zongwu Xie, Baoshi Cao, Hong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09491)  

**Abstract**: Transparent and reflective objects in everyday environments pose significant challenges for depth sensors due to their unique visual properties, such as specular reflections and light transmission. These characteristics often lead to incomplete or inaccurate depth estimation, which severely impacts downstream geometry-based vision tasks, including object recognition, scene reconstruction, and robotic manipulation. To address the issue of missing depth information in transparent and reflective objects, we propose DCIRNet, a novel multimodal depth completion network that effectively integrates RGB images and depth maps to enhance depth estimation quality. Our approach incorporates an innovative multimodal feature fusion module designed to extract complementary information between RGB images and incomplete depth maps. Furthermore, we introduce a multi-stage supervision and depth refinement strategy that progressively improves depth completion and effectively mitigates the issue of blurred object boundaries. We integrate our depth completion model into dexterous grasping frameworks and achieve a $44\%$ improvement in the grasp success rate for transparent and reflective objects. We conduct extensive experiments on public datasets, where DCIRNet demonstrates superior performance. The experimental results validate the effectiveness of our approach and confirm its strong generalization capability across various transparent and reflective objects. 

**Abstract (ZH)**: 透明和反射物体在日常生活环境中的深度传感中由于其独特的视觉特性（如镜面反射和透光性）造成了重大挑战，这常常导致不完整或不准确的深度估计，严重影响基于几何的下游视觉任务，包括物体识别、场景重建和机器人操作。为了应对透明和反射物体中缺失的深度信息问题，我们提出了一种新颖的多模态深度完成网络DCIRNet，该网络有效整合了RGB图像和深度图以提升深度估计质量。我们的方法包含一种创新的多模态特征融合模块，用于提取RGB图像和不完整深度图之间的互补信息。此外，我们还提出了一种多阶段监督和深度细化策略，逐步提高深度完成质量并有效缓解物体边界模糊的问题。我们将深度完成模型集成到灵巧抓取框架中，对于透明和反射物体实现了44%的抓取成功率提升。我们在公共数据集上进行了广泛的实验，结果表明DCIRNet具有优越性能，并证实了该方法在不同透明和反射物体上的强泛化能力。 

---
# Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation 

**Title (ZH)**: Adv-BMT：双向运动变换器在安全关键交通场景生成中的应用 

**Authors**: Yuxin Liu, Zhenghao Peng, Xuanhao Cui, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09485)  

**Abstract**: Scenario-based testing is essential for validating the performance of autonomous driving (AD) systems. However, such testing is limited by the scarcity of long-tailed, safety-critical scenarios in existing datasets collected in the real world. To tackle the data issue, we propose the Adv-BMT framework, which augments real-world scenarios with diverse and realistic adversarial interactions. The core component of Adv-BMT is a bidirectional motion transformer (BMT) model to perform inverse traffic motion predictions, which takes agent information in the last time step of the scenario as input, and reconstruct the traffic in the inverse of chronological order until the initial time step. The Adv-BMT framework is a two-staged pipeline: it first conducts adversarial initializations and then inverse motion predictions. Different from previous work, we do not need any collision data for pretraining, and are able to generate realistic and diverse collision interactions. Our experimental results validate the quality of generated collision scenarios by Adv-BMT: training in our augmented dataset would reduce episode collision rates by 20\% compared to previous work. 

**Abstract (ZH)**: 基于场景的测试对于验证自动驾驶（AD）系统的性能至关重要。然而，这种测试受限于现有实际数据集中长尾、安全性关键场景的稀缺性。为应对数据挑战，我们提出Adv-BMT框架，通过引入多样化和现实的对抗交互来扩充实际场景。Adv-BMT的核心组件是一个双向运动变换器（BMT）模型，用于执行逆交通运动预测，该模型以场景最后一时间步的代理信息为输入，逆时序重建交通状况直至初始时间步。Adv-BMT框架是一个两阶段管道：首先进行对抗初始化，然后进行逆运动预测。与先前工作不同，我们无需任何碰撞数据进行预训练，并能够生成现实且多样的碰撞交互。我们的实验结果验证了Adv-BMT生成的碰撞场景质量：在扩展数据集中训练会导致 episode 碰撞率降低 20%。 

---
# Design of an innovative robotic surgical instrument for circular stapling 

**Title (ZH)**: 设计一种创新的圆周吻合 Robotics手术器械 

**Authors**: Paul Tucan, Nadim Al Hajjar, Calin Vaida, Alexandru Pusca, Tiberiu Antal, Corina Radu, Daniel Jucan, Adrian Pisla, Damien Chablat, Doina Pisla  

**Link**: [PDF](https://arxiv.org/pdf/2506.09444)  

**Abstract**: Esophageal cancer remains a highly aggressive malignancy with low survival rates, requiring advanced surgical interventions like esophagectomy. Traditional manual techniques, including circular staplers, face challenges such as limited precision, prolonged recovery times, and complications like leaks and tissue misalignment. This paper presents a novel robotic circular stapler designed to enhance the dexterity in confined spaces, improve tissue alignment, and reduce post-operative risks. Integrated with a cognitive robot that serves as a surgeon's assistant, the surgical stapler uses three actuators to perform anvil motion, cutter/stapler motion and allows a 75-degree bending of the cartridge (distal tip). Kinematic analysis is used to compute the stapler tip's position, ensuring synchronization with a robotic system. 

**Abstract (ZH)**: 食管癌 remains 一种高度侵袭性的恶性肿瘤，伴有较低的生存率，需要先进的手术干预如食管切除术。传统的手工技术，包括圆形吻合器，面临着精确度有限、恢复时间长以及漏口和组织错位等并发症的挑战。本文介绍了一种新型的机器人圆形吻合器，旨在在有限空间中提高机动性、改善组织对齐并降低术后风险。该吻合器集成了作为外科医生助手的认知机器人，使用三个执行器进行铆钉头运动、切割/吻合器运动，并允许枪 cartridge (远端尖端) 75度弯曲。使用运动分析来计算吻合器尖端的位置，确保与机器人系统同步。 

---
# Time-Unified Diffusion Policy with Action Discrimination for Robotic Manipulation 

**Title (ZH)**: 时间统一扩散策略与动作鉴别方法在机器人操作中的应用 

**Authors**: Ye Niu, Sanping Zhou, Yizhe Li, Ye Den, Le Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09422)  

**Abstract**: In many complex scenarios, robotic manipulation relies on generative models to estimate the distribution of multiple successful actions. As the diffusion model has better training robustness than other generative models, it performs well in imitation learning through successful robot demonstrations. However, the diffusion-based policy methods typically require significant time to iteratively denoise robot actions, which hinders real-time responses in robotic manipulation. Moreover, existing diffusion policies model a time-varying action denoising process, whose temporal complexity increases the difficulty of model training and leads to suboptimal action accuracy. To generate robot actions efficiently and accurately, we present the Time-Unified Diffusion Policy (TUDP), which utilizes action recognition capabilities to build a time-unified denoising process. On the one hand, we build a time-unified velocity field in action space with additional action discrimination information. By unifying all timesteps of action denoising, our velocity field reduces the difficulty of policy learning and speeds up action generation. On the other hand, we propose an action-wise training method, which introduces an action discrimination branch to supply additional action discrimination information. Through action-wise training, the TUDP implicitly learns the ability to discern successful actions to better denoising accuracy. Our method achieves state-of-the-art performance on RLBench with the highest success rate of 82.6% on a multi-view setup and 83.8% on a single-view setup. In particular, when using fewer denoising iterations, TUDP achieves a more significant improvement in success rate. Additionally, TUDP can produce accurate actions for a wide range of real-world tasks. 

**Abstract (ZH)**: 基于时间统一扩散策略的机器人操作高效准确生成方法 

---
# Scoop-and-Toss: Dynamic Object Collection for Quadrupedal Systems 

**Title (ZH)**: 揽取并抛掷：四足机器人系统的动态对象收集 

**Authors**: Minji Kang, Chanwoo Baek, Yoonsang Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.09406)  

**Abstract**: Quadruped robots have made significant advances in locomotion, extending their capabilities from controlled environments to real-world applications. Beyond movement, recent work has explored loco-manipulation using the legs to perform tasks such as pressing buttons or opening doors. While these efforts demonstrate the feasibility of leg-based manipulation, most have focused on relatively static tasks. In this work, we propose a framework that enables quadruped robots to collect objects without additional actuators by leveraging the agility of their legs. By attaching a simple scoop-like add-on to one leg, the robot can scoop objects and toss them into a collection tray mounted on its back. Our method employs a hierarchical policy structure comprising two expert policies-one for scooping and tossing, and one for approaching object positions-and a meta-policy that dynamically switches between them. The expert policies are trained separately, followed by meta-policy training for coordinated multi-object collection. This approach demonstrates how quadruped legs can be effectively utilized for dynamic object manipulation, expanding their role beyond locomotion. 

**Abstract (ZH)**: quadruped机器人在动态物体操作中的腿部利用：一种基于敏捷腿的多物体收集框架 

---
# Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation 

**Title (ZH)**: 分析人类到机器人操作转换中的关键目标以实现灵巧操作 

**Authors**: Chendong Xin, Mingrui Yu, Yongpeng Jiang, Zhefeng Zhang, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09384)  

**Abstract**: Kinematic retargeting from human hands to robot hands is essential for transferring dexterity from humans to robots in manipulation teleoperation and imitation learning. However, due to mechanical differences between human and robot hands, completely reproducing human motions on robot hands is impossible. Existing works on retargeting incorporate various optimization objectives, focusing on different aspects of hand configuration. However, the lack of experimental comparative studies leaves the significance and effectiveness of these objectives unclear. This work aims to analyze these retargeting objectives for dexterous manipulation through extensive real-world comparative experiments. Specifically, we propose a comprehensive retargeting objective formulation that integrates intuitively crucial factors appearing in recent approaches. The significance of each factor is evaluated through experimental ablation studies on the full objective in kinematic posture retargeting and real-world teleoperated manipulation tasks. Experimental results and conclusions provide valuable insights for designing more accurate and effective retargeting algorithms for real-world dexterous manipulation. 

**Abstract (ZH)**: 从人体手部到机器人手部的动力学重塑对于传递灵巧操作能力在操控遥控和模仿学习中的应用至关重要。然而，由于人体和机器人手部的机械差异，完全在机器人手部复制人体动作是不可能的。现有的动力学重塑工作包含了各种优化目标，侧重于手部配置的不同方面。然而，缺乏实验比较研究使得这些目标的重要性及有效性不明确。本文旨在通过广泛的现实世界比较实验分析这些动力学重塑目标。具体而言，我们提出了一种综合的动力学重塑目标公式，结合了近期方法中直观重要的因素。通过对整体目标在动力学姿态重塑和现实世界遥控操作任务中的实验消融研究评估每个因素的重要性。实验结果和结论为设计更准确和有效的现实世界灵巧操作动力学重塑算法提供了宝贵的见解。 

---
# Bipedal Balance Control with Whole-body Musculoskeletal Standing and Falling Simulations 

**Title (ZH)**: 双足平衡控制的全身肌肉骨骼站立与跌倒模拟 

**Authors**: Chengtian Ma, Yunyue Wei, Chenhui Zuo, Chen Zhang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.09383)  

**Abstract**: Balance control is important for human and bipedal robotic systems. While dynamic balance during locomotion has received considerable attention, quantitative understanding of static balance and falling remains limited. This work presents a hierarchical control pipeline for simulating human balance via a comprehensive whole-body musculoskeletal system. We identified spatiotemporal dynamics of balancing during stable standing, revealed the impact of muscle injury on balancing behavior, and generated fall contact patterns that aligned with clinical data. Furthermore, our simulated hip exoskeleton assistance demonstrated improvement in balance maintenance and reduced muscle effort under perturbation. This work offers unique muscle-level insights into human balance dynamics that are challenging to capture experimentally. It could provide a foundation for developing targeted interventions for individuals with balance impairments and support the advancement of humanoid robotic systems. 

**Abstract (ZH)**: 平衡控制对人类和双足机器人系统至关重要。虽然在运动过程中动态平衡已受到广泛关注，但对静态平衡和跌倒的定量理解仍然有限。本工作提出了一种分层控制框架，通过完整的全身肌肉骨骼系统模拟人类平衡。我们识别了稳定站立过程中平衡的时空动态，揭示了肌肉损伤对平衡行为的影响，并生成了与临床数据相符的跌倒接触模式。此外，我们模拟的髋部外骨骼辅助显示了在扰动下平衡维持的改善和肌肉努力的减少。本工作提供了关于人类平衡动力学的独特肌肉层面见解，这些见解在实验中难以捕捉。它为开发针对平衡受损个体的靶向干预措施提供了基础，并支持类人机器人系统的进步。 

---
# SkillBlender: Towards Versatile Humanoid Whole-Body Loco-Manipulation via Skill Blending 

**Title (ZH)**: SkillBlender: 向 versatiles 人形全身动操作融合方向努力通过技能融合 

**Authors**: Yuxuan Kuang, Haoran Geng, Amine Elhafsi, Tan-Dzung Do, Pieter Abbeel, Jitendra Malik, Marco Pavone, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09366)  

**Abstract**: Humanoid robots hold significant potential in accomplishing daily tasks across diverse environments thanks to their flexibility and human-like morphology. Recent works have made significant progress in humanoid whole-body control and loco-manipulation leveraging optimal control or reinforcement learning. However, these methods require tedious task-specific tuning for each task to achieve satisfactory behaviors, limiting their versatility and scalability to diverse tasks in daily scenarios. To that end, we introduce SkillBlender, a novel hierarchical reinforcement learning framework for versatile humanoid loco-manipulation. SkillBlender first pretrains goal-conditioned task-agnostic primitive skills, and then dynamically blends these skills to accomplish complex loco-manipulation tasks with minimal task-specific reward engineering. We also introduce SkillBench, a parallel, cross-embodiment, and diverse simulated benchmark containing three embodiments, four primitive skills, and eight challenging loco-manipulation tasks, accompanied by a set of scientific evaluation metrics balancing accuracy and feasibility. Extensive simulated experiments show that our method significantly outperforms all baselines, while naturally regularizing behaviors to avoid reward hacking, resulting in more accurate and feasible movements for diverse loco-manipulation tasks in our daily scenarios. Our code and benchmark will be open-sourced to the community to facilitate future research. Project page: this https URL. 

**Abstract (ZH)**: 类人机器人通过其灵活性和类人的形态，在跨多种环境完成日常任务方面具有重要的潜力。近期的研究在利用最优控制或强化学习进行类人全身体控和移动操作方面取得了显著进展。然而，这些方法需要针对每个任务进行繁琐的任务特定调优，才能获得满意的行为表现，这限制了它们在日常场景中处理多样任务的灵活性和可扩展性。为此，我们提出了一种新的层次化强化学习框架SkillBlender，以实现灵活的类人移动操作。SkillBlender首先预训练目标条件的任务无关基本技能，然后动态融合这些技能，以最少的任务特定奖励工程实现复杂的移动操作任务。我们还引入了SkillBench，这是一个并行、跨载体和多样化的模拟基准平台，包含三个载体、四项基本技能和八项具有挑战性的移动操作任务，配有平衡准确性和可行性的科学评估指标。广泛的模拟实验表明，我们的方法在所有基线方法中表现出显著的优势，同时自然地调节行为以避免奖励劫持，从而在我们的日常场景中实现了更加准确和可行的移动操作。我们的代码和基准平台将向社区开源，旨在促进未来的研究。项目页面：this https URL。 

---
# UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation 

**Title (ZH)**: UAD: 无监督作用知识精炼以增强机器人操作的泛化能力 

**Authors**: Yihe Tang, Wenlong Huang, Yingke Wang, Chengshu Li, Roy Yuan, Ruohan Zhang, Jiajun Wu, Li Fei-Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.09284)  

**Abstract**: Understanding fine-grained object affordances is imperative for robots to manipulate objects in unstructured environments given open-ended task instructions. However, existing methods of visual affordance predictions often rely on manually annotated data or conditions only on a predefined set of tasks. We introduce UAD (Unsupervised Affordance Distillation), a method for distilling affordance knowledge from foundation models into a task-conditioned affordance model without any manual annotations. By leveraging the complementary strengths of large vision models and vision-language models, UAD automatically annotates a large-scale dataset with detailed $<$instruction, visual affordance$>$ pairs. Training only a lightweight task-conditioned decoder atop frozen features, UAD exhibits notable generalization to in-the-wild robotic scenes and to various human activities, despite only being trained on rendered objects in simulation. Using affordance provided by UAD as the observation space, we show an imitation learning policy that demonstrates promising generalization to unseen object instances, object categories, and even variations in task instructions after training on as few as 10 demonstrations. Project website: this https URL 

**Abstract (ZH)**: 理解细粒度物体 affordance 对机器人在非结构化环境中根据开放任务指令操作物体至关重要。现有的视觉 affordance 预测方法 often 通常依赖于手动标注的数据或仅局限于预定义的任务集中。我们引入了 UAD（无监督 affordance 提炼），这是一种不依赖任何手动标注，将基础模型中的 affordance 知识提炼到任务条件化 affordance 模型中的方法。通过利用大视觉模型和视觉语言模型的互补优势，UAD 自动标注了一个大规模数据集，包含详细的 $<$指令, 视觉 affordance$>$ 对。仅在冻结特征上训练一个轻量级的任务条件化解码器，UAD 在野外机器人场景和各种人类活动中表现出显著的泛化能力，尽管仅在模拟中的渲染物体上进行了训练。使用 UAD 提供的 affordance 作为观测空间，我们展示了一个模仿学习策略，在仅进行少量（如 10 次）演示后，该策略显示出对未见物体实例、物体类别甚至任务指令变化的良好泛化能力。项目网站：this https URL 

---
# Perception Characteristics Distance: Measuring Stability and Robustness of Perception System in Dynamic Conditions under a Certain Decision Rule 

**Title (ZH)**: 感知特性距离：在特定决策规则下的动态条件下感知系统稳定性和鲁棒性测量 

**Authors**: Boyu Jiang, Liang Shi, Zhengzhi Lin, Loren Stowe, Feng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.09217)  

**Abstract**: The performance of perception systems in autonomous driving systems (ADS) is strongly influenced by object distance, scene dynamics, and environmental conditions such as weather. AI-based perception outputs are inherently stochastic, with variability driven by these external factors, while traditional evaluation metrics remain static and event-independent, failing to capture fluctuations in confidence over time. In this work, we introduce the Perception Characteristics Distance (PCD) -- a novel evaluation metric that quantifies the farthest distance at which an object can be reliably detected, incorporating uncertainty in model outputs. To support this, we present the SensorRainFall dataset, collected on the Virginia Smart Road using a sensor-equipped vehicle (cameras, radar, LiDAR) under controlled daylight-clear and daylight-rain scenarios, with precise ground-truth distances to the target objects. Statistical analysis reveals the presence of change points in the variance of detection confidence score with distance. By averaging the PCD values across a range of detection quality thresholds and probabilistic thresholds, we compute the mean PCD (mPCD), which captures the overall perception characteristics of a system with respect to detection distance. Applying state-of-the-art perception models shows that mPCD captures meaningful reliability differences under varying weather conditions -- differences that static metrics overlook. PCD provides a principled, distribution-aware measure of perception performance, supporting safer and more robust ADS operation, while the SensorRainFall dataset offers a valuable benchmark for evaluation. The SensorRainFall dataset is publicly available at this https URL, and the evaluation code is open-sourced at this https URL. 

**Abstract (ZH)**: 自主驾驶系统（ADS）中感知系统的性能受物体距离、场景动态以及天气等环境条件的强烈影响。基于AI的感知输出本质上具有随机性，其变异性由这些外部因素驱动，而传统的评估指标保持静态且独立于事件，未能捕捉随时间变化的信心波动。在本文中，我们引入了感知特性距离（PCD）——一种新的评估指标，用于量化能在可靠检测到目标对象的最远距离，同时考虑了模型输出的不确定性。为了支持这一评估指标，我们介绍了在受控的昼夜晴朗和昼夜雨天场景下，使用传感器装备车辆（摄像头、雷达、LiDAR）在维吉尼亚智能公路上收集的SensorRainFall数据集，该数据集提供了目标物体精确的地面真实距离。统计分析揭示了检测置信度分数随距离变化的突变点。通过在一系列检测质量阈值和概率阈值下平均PCD值，我们计算了均值感知特性距离（mPCD），它表征了系统在检测距离方面的整体感知特性。应用最先进的感知模型表明，mPCD在不同天气条件下捕捉到了有意义的可靠性差异，而静态指标未能捕捉到这些差异。PCD提供了一种原理上合理、分布感知性强的感知性能评估方法，有助于实现更安全、更可靠的ADS操作，而SensorRainFall数据集则为评估提供了宝贵的基准。SensorRainFall数据集可通过以下链接获取：this https URL，评估代码在此链接开源：this https URL。 

---
# Towards Full-Scenario Safety Evaluation of Automated Vehicles: A Volume-Based Method 

**Title (ZH)**: 面向自动化车辆全场景安全评估的体积基方法 

**Authors**: Hang Zhou, Chengyuan Ma, Shiyu Shen, Xiaopeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.09182)  

**Abstract**: With the rapid development of automated vehicles (AVs) in recent years, commercially available AVs are increasingly demonstrating high-level automation capabilities. However, most existing AV safety evaluation methods are primarily designed for simple maneuvers such as car-following and lane-changing. While suitable for basic tests, these methods are insufficient for assessing high-level automation functions deployed in more complex environments. First, these methods typically use crash rate as the evaluation metric, whose accuracy heavily depends on the quality and completeness of naturalistic driving environment data used to estimate scenario probabilities. Such data is often difficult and expensive to collect. Second, when applied to diverse scenarios, these methods suffer from the curse of dimensionality, making large-scale evaluation computationally intractable. To address these challenges, this paper proposes a novel framework for full-scenario AV safety evaluation. A unified model is first introduced to standardize the representation of diverse driving scenarios. This modeling approach constrains the dimension of most scenarios to a regular highway setting with three lanes and six surrounding background vehicles, significantly reducing dimensionality. To further avoid the limitations of probability-based method, we propose a volume-based evaluation method that quantifies the proportion of risky scenarios within the entire scenario space. For car-following scenarios, we prove that the set of safe scenarios is convex under specific settings, enabling exact volume computation. Experimental results validate the effectiveness of the proposed volume-based method using both AV behavior models from existing literature and six production AV models calibrated from field-test trajectory data in the Ultra-AV dataset. Code and data will be made publicly available upon acceptance of this paper. 

**Abstract (ZH)**: 近年来，随着自动驾驶车辆（AVs）的快速发展，商用AVs日益展现出高层次的自动化能力。然而，目前大多数现有的AV安全评估方法主要针对如跟车和变道等简单的操作。虽然适用于基本测试，但这些方法对部署在更复杂环境中的高层次自动化功能评估不足。首先，这些方法通常使用碰撞率作为评估指标，其准确性高度依赖于用于估计场景概率的自然驾驶环境数据的质量和完整性，而这样的数据收集起来往往既困难又昂贵。其次，当应用于多种场景时，这些方法会遭受维度灾难的问题，使得大规模评估在计算上变得不可行。为了解决这些挑战，本文提出了一种新的全场景AV安全评估框架。首先引入了一个统一模型来标准化各类驾驶场景的表示。通过这种方法，可以将大多数场景的维度限制在一个标准的三车道高速公路设置下，显著降低了维度。为了进一步避免基于概率方法的局限性，本文提出了基于体素的评估方法，该方法量化整个场景空间中具有风险的场景的比例。对于跟车场景，我们证明在特定设置下安全场景集合是凸的，能够实现精确的体积计算。实验结果使用文献中现有的自动驾驶车辆行为模型和 Ultra-AV 数据集中现场测试轨迹数据校准的六款生产中自动驾驶车辆模型验证了所提出的方法的有效性。代码和数据将在本文被接受后公开发布。 

---
# Hearing the Slide: Acoustic-Guided Constraint Learning for Fast Non-Prehensile Transport 

**Title (ZH)**: 听滑动声：基于声学的约束学习快速非抓取运输 

**Authors**: Yuemin Mao, Bardienus P. Duisterhof, Moonyoung Lee, Jeffrey Ichnowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.09169)  

**Abstract**: Object transport tasks are fundamental in robotic automation, emphasizing the importance of efficient and secure methods for moving objects. Non-prehensile transport can significantly improve transport efficiency, as it enables handling multiple objects simultaneously and accommodating objects unsuitable for parallel-jaw or suction grasps. Existing approaches incorporate constraints based on the Coulomb friction model, which is imprecise during fast motions where inherent mechanical vibrations occur. Imprecise constraints can cause transported objects to slide or even fall off the tray. To address this limitation, we propose a novel method to learn a friction model using acoustic sensing that maps a tray's motion profile to a dynamically conditioned friction coefficient. This learned model enables an optimization-based motion planner to adjust the friction constraint at each control step according to the planned motion at that step. In experiments, we generate time-optimized trajectories for a UR5e robot to transport various objects with constraints using both the standard Coulomb friction model and the learned friction model. Results suggest that the learned friction model reduces object displacement by up to 86.0% compared to the baseline, highlighting the effectiveness of acoustic sensing in learning real-world friction constraints. 

**Abstract (ZH)**: 基于声学感知学习的摩擦模型在机器人物体运输任务中的应用 

---
# WD-DETR: Wavelet Denoising-Enhanced Real-Time Object Detection Transformer for Robot Perception with Event Cameras 

**Title (ZH)**: WD-DETR：小波去噪增强的实时目标检测变换器——适用于事件相机的机器人感知 

**Authors**: Yangjie Cui, Boyang Gao, Yiwei Zhang, Xin Dong, Jinwu Xiang, Daochun Li, Zhan Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09098)  

**Abstract**: Previous studies on event camera sensing have demonstrated certain detection performance using dense event representations. However, the accumulated noise in such dense representations has received insufficient attention, which degrades the representation quality and increases the likelihood of missed detections. To address this challenge, we propose the Wavelet Denoising-enhanced DEtection TRansformer, i.e., WD-DETR network, for event cameras. In particular, a dense event representation is presented first, which enables real-time reconstruction of events as tensors. Then, a wavelet transform method is designed to filter noise in the event representations. Such a method is integrated into the backbone for feature extraction. The extracted features are subsequently fed into a transformer-based network for object prediction. To further reduce inference time, we incorporate the Dynamic Reorganization Convolution Block (DRCB) as a fusion module within the hybrid encoder. The proposed method has been evaluated on three event-based object detection datasets, i.e., DSEC, Gen1, and 1Mpx. The results demonstrate that WD-DETR outperforms tested state-of-the-art methods. Additionally, we implement our approach on a common onboard computer for robots, the NVIDIA Jetson Orin NX, achieving a high frame rate of approximately 35 FPS using TensorRT FP16, which is exceptionally well-suited for real-time perception of onboard robotic systems. 

**Abstract (ZH)**: Wavelet Denoising-enhanced DEtection TRansformer网络在事件相机感知中的应用 

---
# V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning 

**Title (ZH)**: V-JEPA 2：自我监督视频模型实现理解、预测与规划 

**Authors**: Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas  

**Link**: [PDF](https://arxiv.org/pdf/2506.09985)  

**Abstract**: A major challenge for modern AI is to learn to understand the world and learn to act largely by observation. This paper explores a self-supervised approach that combines internet-scale video data with a small amount of interaction data (robot trajectories), to develop models capable of understanding, predicting, and planning in the physical world. We first pre-train an action-free joint-embedding-predictive architecture, V-JEPA 2, on a video and image dataset comprising over 1 million hours of internet video. V-JEPA 2 achieves strong performance on motion understanding (77.3 top-1 accuracy on Something-Something v2) and state-of-the-art performance on human action anticipation (39.7 recall-at-5 on Epic-Kitchens-100) surpassing previous task-specific models. Additionally, after aligning V-JEPA 2 with a large language model, we demonstrate state-of-the-art performance on multiple video question-answering tasks at the 8 billion parameter scale (e.g., 84.0 on PerceptionTest, 76.9 on TempCompass). Finally, we show how self-supervised learning can be applied to robotic planning tasks by post-training a latent action-conditioned world model, V-JEPA 2-AC, using less than 62 hours of unlabeled robot videos from the Droid dataset. We deploy V-JEPA 2-AC zero-shot on Franka arms in two different labs and enable picking and placing of objects using planning with image goals. Notably, this is achieved without collecting any data from the robots in these environments, and without any task-specific training or reward. This work demonstrates how self-supervised learning from web-scale data and a small amount of robot interaction data can yield a world model capable of planning in the physical world. 

**Abstract (ZH)**: 现代AI的一项主要挑战是通过观察学习理解和行动。本文探讨了一种自监督方法，该方法结合互联网规模的视频数据和少量交互数据（机器人轨迹），以开发能够在物理世界中理解、预测和规划的模型。首先，在包含超过100万小时互联网视频和图像的数据集上预训练了一个无动作联合嵌入预测架构V-JEPA 2。V-JEPA 2在动作理解方面取得了强大的性能（Something-Something v2上的77.3顶级准确度），并在人体动作预见性方面取得了最先进的性能（Epic-Kitchens-100上的39.7召回率），超越了先前的任务特定模型。此外，在将V-JEPA 2与大型语言模型对齐后，我们展示了其在80亿参数量级的多个视频问答任务上的最先进的性能（例如，PerceptionTest上的84.0，TempCompass上的76.9）。最后，我们展示了如何通过使用Droid数据集中的不到62小时的未标记机器人视频后训练一个潜动作条件世界模型V-JEPA 2-AC，将自监督学习应用于机器人规划任务。我们在两个不同的实验室中使用V-JEPA 2-AC零样本部署Franka手臂，并使用图像目标进行抓取和放置操作。值得注意的是，这一成就是在这些环境中不收集任何机器人数据、无需任何任务特定训练或奖励的情况下实现的。本文展示了如何从大规模网络数据和少量的机器人交互数据中进行自监督学习，从而生成一个能够在物理世界中进行规划的世界模型。 

---
# ReSim: Reliable World Simulation for Autonomous Driving 

**Title (ZH)**: ReSim: 可靠的世界模拟技术在自主驾驶中的应用 

**Authors**: Jiazhi Yang, Kashyap Chitta, Shenyuan Gao, Long Chen, Yuqian Shao, Xiaosong Jia, Hongyang Li, Andreas Geiger, Xiangyu Yue, Li Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09981)  

**Abstract**: How can we reliably simulate future driving scenarios under a wide range of ego driving behaviors? Recent driving world models, developed exclusively on real-world driving data composed mainly of safe expert trajectories, struggle to follow hazardous or non-expert behaviors, which are rare in such data. This limitation restricts their applicability to tasks such as policy evaluation. In this work, we address this challenge by enriching real-world human demonstrations with diverse non-expert data collected from a driving simulator (e.g., CARLA), and building a controllable world model trained on this heterogeneous corpus. Starting with a video generator featuring a diffusion transformer architecture, we devise several strategies to effectively integrate conditioning signals and improve prediction controllability and fidelity. The resulting model, ReSim, enables Reliable Simulation of diverse open-world driving scenarios under various actions, including hazardous non-expert ones. To close the gap between high-fidelity simulation and applications that require reward signals to judge different actions, we introduce a Video2Reward module that estimates a reward from ReSim's simulated future. Our ReSim paradigm achieves up to 44% higher visual fidelity, improves controllability for both expert and non-expert actions by over 50%, and boosts planning and policy selection performance on NAVSIM by 2% and 25%, respectively. 

**Abstract (ZH)**: 如何在广泛的行为范围内可靠地模拟未来的驾驶场景？通过丰富真实世界的人类演示数据并结合驾驶模拟器获取的多样化非专家数据，构建一个可控的世界模型，以应对这一挑战。ReSim模型的提出，能够在各种行动（包括非专家的危险行为）下可靠地模拟多种开放世界的驾驶场景。为了弥合高保真模拟与需要奖励信号判断不同行动的应用之间的差距，我们引入了Video2Reward模块，从ReSim模拟的未来中估算奖励。ReSim范式实现了最高44%的视觉保真度提升，对于专家和非专家行动的可控性分别提高了超过50%，并在NAVSIM中分别提升了2%和25%的计划和策略选择性能。 

---
# OctoNav: Towards Generalist Embodied Navigation 

**Title (ZH)**: OctoNav: 通达导航owards Generalist Embodied Navigation 

**Authors**: Chen Gao, Liankai Jin, Xingyu Peng, Jiazhao Zhang, Yue Deng, Annan Li, He Wang, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09839)  

**Abstract**: Embodied navigation stands as a foundation pillar within the broader pursuit of embodied AI. However, previous navigation research is divided into different tasks/capabilities, e.g., ObjNav, ImgNav and VLN, where they differ in task objectives and modalities, making datasets and methods are designed individually. In this work, we take steps toward generalist navigation agents, which can follow free-form instructions that include arbitrary compounds of multi-modal and multi-capability. To achieve this, we propose a large-scale benchmark and corresponding method, termed OctoNav-Bench and OctoNav-R1. Specifically, OctoNav-Bench features continuous environments and is constructed via a designed annotation pipeline. We thoroughly craft instruction-trajectory pairs, where instructions are diverse in free-form with arbitrary modality and capability. Also, we construct a Think-Before-Action (TBA-CoT) dataset within OctoNav-Bench to provide the thinking process behind actions. For OctoNav-R1, we build it upon MLLMs and adapt it to a VLA-type model, which can produce low-level actions solely based on 2D visual observations. Moreover, we design a Hybrid Training Paradigm (HTP) that consists of three stages, i.e., Action-/TBA-SFT, Nav-GPRO, and Online RL stages. Each stage contains specifically designed learning policies and rewards. Importantly, for TBA-SFT and Nav-GRPO designs, we are inspired by the OpenAI-o1 and DeepSeek-R1, which show impressive reasoning ability via thinking-before-answer. Thus, we aim to investigate how to achieve thinking-before-action in the embodied navigation field, to improve model's reasoning ability toward generalists. Specifically, we propose TBA-SFT to utilize the TBA-CoT dataset to fine-tune the model as a cold-start phrase and then leverage Nav-GPRO to improve its thinking ability. Finally, OctoNav-R1 shows superior performance compared with previous methods. 

**Abstract (ZH)**: 基于实体的导航作为广义实体AI的基础支柱，然而先前的导航研究被细分到不同的任务/能力中，例如ObjNav、ImgNav和VLN，它们在任务目标和模态上有所不同，导致数据集和方法各自独立设计。在本文中，我们朝着通用导航代理迈出步伐，能够跟随自由形式的指令，包括多模态和多能力的任意组合。为了实现这一目标，我们提出了一项大规模基准和相应的方法，即OctoNav-Bench和OctoNav-R1。具体而言，OctoNav-Bench采用了连续环境，并通过设计的标注管道构建而成。我们详细设计了指令-轨迹对，其中指令在自由形式方面具有多样性和任意模态与能力。此外，我们在OctoNav-Bench中构建了一个Think-Before-Action (TBA-CoT) 数据集，提供了行动背后的思维过程。对于OctoNav-R1，我们基于MLLMs进行构建并将其适配为VLA型模型，仅基于2D视觉观察即可生成低级动作。此外，我们设计了一个混合训练范式（HTP），包括三个阶段：Action-/TBA-SFT、Nav-GPRO和在线RL阶段。每个阶段都包含特定设计的学习策略和奖励。重要的是，对于TBA-SFT和Nav-GRPO的设计，我们受到OpenAI-o1和DeepSeek-R1的启发，展示了通过问答前思考实现强大推理能力。因此，我们旨在探究如何在基于实体的导航领域实现行动前思考，以提高模型的推理能力，特别是对于通用模型。具体而言，我们提出TBA-SFT利用TBA-CoT数据集对模型进行微调作为冷启动词，并利用Nav-GPRO提高其思考能力。最后，OctoNav-R1在与先前方法的比较中表现更优。 

---
# Hierarchical Image Matching for UAV Absolute Visual Localization via Semantic and Structural Constraints 

**Title (ZH)**: 基于语义和结构约束的无人机绝对视觉定位分层图像匹配 

**Authors**: Xiangkai Zhang, Xiang Zhou, Mao Chen, Yuchen Lu, Xu Yang, Zhiyong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.09748)  

**Abstract**: Absolute localization, aiming to determine an agent's location with respect to a global reference, is crucial for unmanned aerial vehicles (UAVs) in various applications, but it becomes challenging when global navigation satellite system (GNSS) signals are unavailable. Vision-based absolute localization methods, which locate the current view of the UAV in a reference satellite map to estimate its position, have become popular in GNSS-denied scenarios. However, existing methods mostly rely on traditional and low-level image matching, suffering from difficulties due to significant differences introduced by cross-source discrepancies and temporal variations. To overcome these limitations, in this paper, we introduce a hierarchical cross-source image matching method designed for UAV absolute localization, which integrates a semantic-aware and structure-constrained coarse matching module with a lightweight fine-grained matching module. Specifically, in the coarse matching module, semantic features derived from a vision foundation model first establish region-level correspondences under semantic and structural constraints. Then, the fine-grained matching module is applied to extract fine features and establish pixel-level correspondences. Building upon this, a UAV absolute visual localization pipeline is constructed without any reliance on relative localization techniques, mainly by employing an image retrieval module before the proposed hierarchical image matching modules. Experimental evaluations on public benchmark datasets and a newly introduced CS-UAV dataset demonstrate superior accuracy and robustness of the proposed method under various challenging conditions, confirming its effectiveness. 

**Abstract (ZH)**: 基于层级跨源图像匹配的无人机绝对视觉定位方法 

---
# HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios 

**Title (ZH)**: HopaDIFF：全面-局部aware傅里叶条件扩散在多人大规模场景中人类动作分割中的应用 

**Authors**: Kunyu Peng, Junchao Huang, Xiangsheng Huang, Di Wen, Junwei Zheng, Yufan Chen, Kailun Yang, Jiamin Wu, Chongqing Hao, Rainer Stiefelhagen  

**Link**: [PDF](https://arxiv.org/pdf/2506.09650)  

**Abstract**: Action segmentation is a core challenge in high-level video understanding, aiming to partition untrimmed videos into segments and assign each a label from a predefined action set. Existing methods primarily address single-person activities with fixed action sequences, overlooking multi-person scenarios. In this work, we pioneer textual reference-guided human action segmentation in multi-person settings, where a textual description specifies the target person for segmentation. We introduce the first dataset for Referring Human Action Segmentation, i.e., RHAS133, built from 133 movies and annotated with 137 fine-grained actions with 33h video data, together with textual descriptions for this new task. Benchmarking existing action recognition methods on RHAS133 using VLM-based feature extractors reveals limited performance and poor aggregation of visual cues for the target person. To address this, we propose a holistic-partial aware Fourier-conditioned diffusion framework, i.e., HopaDIFF, leveraging a novel cross-input gate attentional xLSTM to enhance holistic-partial long-range reasoning and a novel Fourier condition to introduce more fine-grained control to improve the action segmentation generation. HopaDIFF achieves state-of-the-art results on RHAS133 in diverse evaluation settings. The code is available at this https URL. 

**Abstract (ZH)**: 多人员场景下的文本引导人体动作分割：一种全新的 holistic-partial 意识 Fourier 条件扩散框架（HopaDIFF） 

---
# Adaptive event-triggered robust tracking control of soft robots 

**Title (ZH)**: 软体机器人自适应事件触发鲁棒跟踪控制 

**Authors**: Renjie Ma, Ziyao Qu, Zhijian Hu, Dong Zhao, Marios M. Polycarpou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09523)  

**Abstract**: Soft robots manufactured with flexible materials can be highly compliant and adaptive to their surroundings, which facilitates their application in areas such as dexterous manipulation and environmental exploration. This paper aims at investigating the tracking control problem for soft robots under uncertainty such as unmodeled dynamics and external disturbance. First, we establish a novel switching function and design the compensated tracking error dynamics by virtue of the command filter. Then, based on the backstepping methodology, the virtual controllers and the adaptive logic estimating the supremum of uncertainty impacts are developed for synthesizing an event-triggered control strategy. In addition, the uniformed finite-time stability certification is derived for different scenarios of the switching function. Finally, we perform a case study of a soft robot to illustrate the effectiveness of the proposed control algorithm. 

**Abstract (ZH)**: 基于柔性材料制造的软机器人可以高度顺应环境，适用于灵巧操作和环境探索等领域。本文旨在研究在未建模动态和外部干扰等不确定性条件下软机器人的跟踪控制问题。首先，我们建立了新的切换函数，并利用命令滤波器设计补偿跟踪误差动力学。然后，基于回步设计方法，开发了虚拟控制器和自适应逻辑来估计不确定性影响的上界，并据此合成了一种事件触发控制策略。此外，还推导了不同场景下切换函数的一致有限时间稳定性认证。最后，通过软机器人的案例研究来说明所提出控制算法的有效性。 

---
# How attention simplifies mental representations for planning 

**Title (ZH)**: 注意力如何简化规划中的心理表征 

**Authors**: Jason da Silva Castanheira, Nicholas Shea, Stephen M. Fleming  

**Link**: [PDF](https://arxiv.org/pdf/2506.09520)  

**Abstract**: Human planning is efficient -- it frugally deploys limited cognitive resources to accomplish difficult tasks -- and flexible -- adapting to novel problems and environments. Computational approaches suggest that people construct simplified mental representations of their environment, balancing the complexity of a task representation with its utility. These models imply a nested optimisation in which planning shapes perception, and perception shapes planning -- but the perceptual and attentional mechanisms governing how this interaction unfolds remain unknown. Here, we harness virtual maze navigation to characterise how spatial attention controls which aspects of a task representation enter subjective awareness and are available for planning. We find that spatial proximity governs which aspects of a maze are available for planning, and that when task-relevant information follows natural (lateralised) contours of attention, people can more easily construct simplified and useful maze representations. This influence of attention varies considerably across individuals, explaining differences in people's task representations and behaviour. Inspired by the 'spotlight of attention' analogy, we incorporate the effects of visuospatial attention into existing computational accounts of value-guided construal. Together, our work bridges computational perspectives on perception and decision-making to better understand how individuals represent their environments in aid of planning. 

**Abstract (ZH)**: 人类规划既高效又灵活——它通过有限的认知资源高效完成复杂任务，并能在新颖的问题和环境中进行调整。计算模型表明，人类构建环境的简化心理表示，并在任务表示的复杂性和其实用性之间取得平衡。这些模型暗示了一种嵌套优化过程，在该过程中，规划塑造感知，而感知又反过来影响规划——但调控这一互动的具体感知和注意机制仍未知。在此，我们利用虚拟迷宫导航来描述空间注意如何控制哪些任务表示方面进入主观意识并可供规划使用。我们发现，空间接近性决定了哪些迷宫方面可供规划使用，并且当与任务相关的信息遵循自然（侧向化）的注意力轮廓时，人们可以更容易地构建简化且有用的地图表示。注意的影响在个体之间差异很大，解释了人们任务表示和行为的差异。受“注意探照灯”类比的启发，我们将空间视觉注意的效果融入现有的值引导解释中。我们的研究将计算感知和决策视角结合在一起，以更好地理解个体如何代表其环境以辅助规划。 

---
# Efficient Preference-Based Reinforcement Learning: Randomized Exploration Meets Experimental Design 

**Title (ZH)**: 基于偏好的高效强化学习：随机探索与实验设计相结合 

**Authors**: Andreas Schlaginhaufen, Reda Ouhamma, Maryam Kamgarpour  

**Link**: [PDF](https://arxiv.org/pdf/2506.09508)  

**Abstract**: We study reinforcement learning from human feedback in general Markov decision processes, where agents learn from trajectory-level preference comparisons. A central challenge in this setting is to design algorithms that select informative preference queries to identify the underlying reward while ensuring theoretical guarantees. We propose a meta-algorithm based on randomized exploration, which avoids the computational challenges associated with optimistic approaches and remains tractable. We establish both regret and last-iterate guarantees under mild reinforcement learning oracle assumptions. To improve query complexity, we introduce and analyze an improved algorithm that collects batches of trajectory pairs and applies optimal experimental design to select informative comparison queries. The batch structure also enables parallelization of preference queries, which is relevant in practical deployment as feedback can be gathered concurrently. Empirical evaluation confirms that the proposed method is competitive with reward-based reinforcement learning while requiring a small number of preference queries. 

**Abstract (ZH)**: 我们在一般马尔可夫决策过程中的强化学习研究：基于轨迹层次偏好对比的人工反馈学习 

---
# CheckManual: A New Challenge and Benchmark for Manual-based Appliance Manipulation 

**Title (ZH)**: CheckManual: 基于手工操作的家电操纵新挑战与基准 

**Authors**: Yuxing Long, Jiyao Zhang, Mingjie Pan, Tianshu Wu, Taewhan Kim, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.09343)  

**Abstract**: Correct use of electrical appliances has significantly improved human life quality. Unlike simple tools that can be manipulated with common sense, different parts of electrical appliances have specific functions defined by manufacturers. If we want the robot to heat bread by microwave, we should enable them to review the microwave manual first. From the manual, it can learn about component functions, interaction methods, and representative task steps about appliances. However, previous manual-related works remain limited to question-answering tasks while existing manipulation researchers ignore the manual's important role and fail to comprehend multi-page manuals. In this paper, we propose the first manual-based appliance manipulation benchmark CheckManual. Specifically, we design a large model-assisted human-revised data generation pipeline to create manuals based on CAD appliance models. With these manuals, we establish novel manual-based manipulation challenges, metrics, and simulator environments for model performance evaluation. Furthermore, we propose the first manual-based manipulation planning model ManualPlan to set up a group of baselines for the CheckManual benchmark. 

**Abstract (ZH)**: 正确使用家用电器显著提升了人类生活质量。为了使机器人能通过微波炉加热面包，我们应让他们先查阅微波炉手册。通过手册，机器人可以学习到电器各部件的功能、交互方式以及代表性任务步骤。然而，现有的手册相关工作仅限于问答任务，而现有的操作研究人员忽视了手册的重要作用，未能理解和掌握多页手册。在本文中，我们提出了首个基于手册的家用电器操作基准CheckManual。具体而言，我们设计了一个大型模型辅助的人工修订数据生成管道，基于CAD家电模型创建手册。借助这些手册，我们建立了新的基于手册的操作挑战、评估指标和模拟环境，以评估模型性能。此外，我们提出了首个基于手册的操作规划模型ManualPlan，为CheckManual基准设立了基线。 

---
# UFM: A Simple Path towards Unified Dense Correspondence with Flow 

**Title (ZH)**: UFM：通往统一密集对应的一种简单途径，基于流的方法 

**Authors**: Yuchen Zhang, Nikhil Keetha, Chenwei Lyu, Bhuvan Jhamb, Yutian Chen, Yuheng Qiu, Jay Karhade, Shreyas Jha, Yaoyu Hu, Deva Ramanan, Sebastian Scherer, Wenshan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.09278)  

**Abstract**: Dense image correspondence is central to many applications, such as visual odometry, 3D reconstruction, object association, and re-identification. Historically, dense correspondence has been tackled separately for wide-baseline scenarios and optical flow estimation, despite the common goal of matching content between two images. In this paper, we develop a Unified Flow & Matching model (UFM), which is trained on unified data for pixels that are co-visible in both source and target images. UFM uses a simple, generic transformer architecture that directly regresses the (u,v) flow. It is easier to train and more accurate for large flows compared to the typical coarse-to-fine cost volumes in prior work. UFM is 28% more accurate than state-of-the-art flow methods (Unimatch), while also having 62% less error and 6.7x faster than dense wide-baseline matchers (RoMa). UFM is the first to demonstrate that unified training can outperform specialized approaches across both domains. This result enables fast, general-purpose correspondence and opens new directions for multi-modal, long-range, and real-time correspondence tasks. 

**Abstract (ZH)**: 密集图像对应在许多应用中至关重要，如视觉里程计、三维重建、对象关联和再识别。历史上，密集对应在宽基线场景和光学流估计中分别处理，尽管它们的共同目标是在两张图像之间匹配内容。在本文中，我们开发了一种统一流动与匹配模型（UFM），该模型在两张源图像和目标图像中共可见像素的统一数据上进行训练。UFM 使用一种简单且通用的变压器架构，直接回归 (u,v) 流。与之前工作中的典型粗到细代价卷积相比，它更容易训练且对于大流更为准确。UFM 在与最先进的流动方法（Unimatch）相比时更为准确，误差降低了62%，速度快6.7倍，而且在共视匹配器（RoMa）上更为准确。UFM 是第一个表明统一训练能够在两个领域都超越专门方法的结果。这一结果开启了快速、通用对应的新方向，并为多模态、长距离和实时对应任务开辟了新方向。 

---
# Robot-Gated Interactive Imitation Learning with Adaptive Intervention Mechanism 

**Title (ZH)**: 机器人门控交互模仿学习与自适应干预机制 

**Authors**: Haoyuan Cai, Zhenghao Peng, Bolei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.09176)  

**Abstract**: Interactive Imitation Learning (IIL) allows agents to acquire desired behaviors through human interventions, but current methods impose high cognitive demands on human supervisors. We propose the Adaptive Intervention Mechanism (AIM), a novel robot-gated IIL algorithm that learns an adaptive criterion for requesting human demonstrations. AIM utilizes a proxy Q-function to mimic the human intervention rule and adjusts intervention requests based on the alignment between agent and human actions. By assigning high Q-values when the agent deviates from the expert and decreasing these values as the agent becomes proficient, the proxy Q-function enables the agent to assess the real-time alignment with the expert and request assistance when needed. Our expert-in-the-loop experiments reveal that AIM significantly reduces expert monitoring efforts in both continuous and discrete control tasks. Compared to the uncertainty-based baseline Thrifty-DAgger, our method achieves a 40% improvement in terms of human take-over cost and learning efficiency. Furthermore, AIM effectively identifies safety-critical states for expert assistance, thereby collecting higher-quality expert demonstrations and reducing overall expert data and environment interactions needed. Code and demo video are available at this https URL. 

**Abstract (ZH)**: 自适应干预机制（AIM）：一种新型的机器人门控 imitative 模仿学习算法 

---
# BG-HOP: A Bimanual Generative Hand-Object Prior 

**Title (ZH)**: BG-HOP: 一种双手生成手物体先验 

**Authors**: Sriram Krishna, Sravan Chittupalli, Sungjae Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.09068)  

**Abstract**: In this work, we present BG-HOP, a generative prior that seeks to model bimanual hand-object interactions in 3D. We address the challenge of limited bimanual interaction data by extending existing single-hand generative priors, demonstrating preliminary results in capturing the joint distribution of hands and objects. Our experiments showcase the model's capability to generate bimanual interactions and synthesize grasps for given objects. We make code and models publicly available. 

**Abstract (ZH)**: BG-HOP: 一种生成先验模型，用于建模三维双手物体交互 

---
