# NuExo: A Wearable Exoskeleton Covering all Upper Limb ROM for Outdoor Data Collection and Teleoperation of Humanoid Robots 

**Title (ZH)**: NuExo: 一款覆盖全上肢活动范围的可穿戴外骨骼，用于户外数据采集和类人机器人远程操作。 

**Authors**: Rui Zhong, Chuang Cheng, Junpeng Xu, Yantong Wei, Ce Guo, Daoxun Zhang, Wei Dai, Huimin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10554)  

**Abstract**: The evolution from motion capture and teleoperation to robot skill learning has emerged as a hotspot and critical pathway for advancing embodied intelligence. However, existing systems still face a persistent gap in simultaneously achieving four objectives: accurate tracking of full upper limb movements over extended durations (Accuracy), ergonomic adaptation to human biomechanics (Comfort), versatile data collection (e.g., force data) and compatibility with humanoid robots (Versatility), and lightweight design for outdoor daily use (Convenience). We present a wearable exoskeleton system, incorporating user-friendly immersive teleoperation and multi-modal sensing collection to bridge this gap. Due to the features of a novel shoulder mechanism with synchronized linkage and timing belt transmission, this system can adapt well to compound shoulder movements and replicate 100% coverage of natural upper limb motion ranges. Weighing 5.2 kg, NuExo supports backpack-type use and can be conveniently applied in daily outdoor scenarios. Furthermore, we develop a unified intuitive teleoperation framework and a comprehensive data collection system integrating multi-modal sensing for various humanoid robots. Experiments across distinct humanoid platforms and different users validate our exoskeleton's superiority in motion range and flexibility, while confirming its stability in data collection and teleoperation accuracy in dynamic scenarios. 

**Abstract (ZH)**: 从运动捕捉和遥控到机器人技能学习的发展已成为提升 embodiable 智能的热点和关键路径。然而，现有系统仍然面临同时实现四大目标的持续差距：长时间精确追踪全上肢运动（准确性）、符合人类生物力学的舒适适应（舒适性）、多功能数据采集（例如，力数据）和与类人机器人兼容性（多功能性），以及野外日常使用的轻量化设计（便利性）。我们提出了一种可穿戴外骨骼系统，结合用户友好的沉浸式遥控和多模态传感数据收集，以填补这一差距。得益于新颖的同步连杆和Timing带传动肩机制，该系统能很好地适应复合肩部运动，并覆盖100%的自然上肢运动范围。重5.2公斤的NuExo支持背pack型使用，并可方便地应用于日常户外场景。此外，我们开发了一种统一的直观遥控框架和一个全面的数据采集系统，结合多种传感技术，用于各种类人机器人。跨不同类人平台和不同用户的实验验证了我们外骨骼在运动范围和灵活性方面的优越性，同时确认了其在动态场景中数据采集和遥控精度的稳定性。 

---
# KUDA: Keypoints to Unify Dynamics Learning and Visual Prompting for Open-Vocabulary Robotic Manipulation 

**Title (ZH)**: KUDA: 关键点统一动力学习与视觉提示的开放词汇机器人操作 

**Authors**: Zixian Liu, Mingtong Zhang, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.10546)  

**Abstract**: With the rapid advancement of large language models (LLMs) and vision-language models (VLMs), significant progress has been made in developing open-vocabulary robotic manipulation systems. However, many existing approaches overlook the importance of object dynamics, limiting their applicability to more complex, dynamic tasks. In this work, we introduce KUDA, an open-vocabulary manipulation system that integrates dynamics learning and visual prompting through keypoints, leveraging both VLMs and learning-based neural dynamics models. Our key insight is that a keypoint-based target specification is simultaneously interpretable by VLMs and can be efficiently translated into cost functions for model-based planning. Given language instructions and visual observations, KUDA first assigns keypoints to the RGB image and queries the VLM to generate target specifications. These abstract keypoint-based representations are then converted into cost functions, which are optimized using a learned dynamics model to produce robotic trajectories. We evaluate KUDA on a range of manipulation tasks, including free-form language instructions across diverse object categories, multi-object interactions, and deformable or granular objects, demonstrating the effectiveness of our framework. The project page is available at this http URL. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）和视觉-语言模型（VLMs）的快速发展，开放词汇的机器人 manipulation 系统取得了显著进展。然而，许多现有方法忽视了物体动力学的重要性，限制了其在更复杂、动态任务中的应用。本文介绍了一种名为 KUDA 的开放词汇 manipulation 系统，通过关键点结合动力学习和视觉提示，利用 VLMs 和基于学习的神经动力学模型进行集成。我们的关键洞察是，基于关键点的目标规格化既能被 VLMs 识别，又能高效地转化为基于模型的规划中的成本函数。在获得语言指令和视觉观察后，KUDA 首先为 RGB 图像分配关键点并查询 VLM 生成目标规格化，随后将这些抽象的关键点表示转化为成本函数，利用学习的动力学模型进行优化以产生机器人轨迹。我们将在各种 manipulation 任务上评估 KUDA，包括跨多种物体类别的一般语言指令、多物体交互以及柔体或颗粒状物体，展示了我们框架的有效性。项目页面可通过以下网址访问。 

---
# Learning Robotic Policy with Imagined Transition: Mitigating the Trade-off between Robustness and Optimality 

**Title (ZH)**: 基于想象过渡学习机器人策略：缓解鲁棒性和最优性之间的Trade-off 

**Authors**: Wei Xiao, Shangke Lyu, Zhefei Gong, Renjie Wang, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10484)  

**Abstract**: Existing quadrupedal locomotion learning paradigms usually rely on extensive domain randomization to alleviate the sim2real gap and enhance robustness. It trains policies with a wide range of environment parameters and sensor noises to perform reliably under uncertainty. However, since optimal performance under ideal conditions often conflicts with the need to handle worst-case scenarios, there is a trade-off between optimality and robustness. This trade-off forces the learned policy to prioritize stability in diverse and challenging conditions over efficiency and accuracy in ideal ones, leading to overly conservative behaviors that sacrifice peak performance. In this paper, we propose a two-stage framework that mitigates this trade-off by integrating policy learning with imagined transitions. This framework enhances the conventional reinforcement learning (RL) approach by incorporating imagined transitions as demonstrative inputs. These imagined transitions are derived from an optimal policy and a dynamics model operating within an idealized setting. Our findings indicate that this approach significantly mitigates the domain randomization-induced negative impact of existing RL algorithms. It leads to accelerated training, reduced tracking errors within the distribution, and enhanced robustness outside the distribution. 

**Abstract (ZH)**: 现有的四足行走学习范式通常依赖于广泛的领域随机化来缓解仿真实验与实际应用之间的差距，并提高鲁棒性。它通过使用广泛范围的环境参数和传感器噪声来训练策略，以在不确定性条件下可靠运行。然而，由于在理想条件下获得最佳性能往往与处理最坏情况的需求相冲突，这在最优性和鲁棒性之间存在着权衡。这种权衡迫使学习到的策略优先考虑在多样且具有挑战性的条件下稳定性，而牺牲在理想条件下的效率和准确性，导致过于保守的行为，牺牲最佳性能。本文提出了一种两阶段框架，通过将策略学习与想象的过渡相结合以缓解这种权衡。该框架通过将想象的过渡作为示范输入整合到传统强化学习（RL）方法中，从而增强了常规的强化学习方法。我们的研究结果表明，这种方法显著减轻了现有RL算法由领域随机化引起的负面影响。它导致训练加速、内部分布内的跟踪误差减少以及外部分布下的鲁棒性增强。 

---
# Finetuning Generative Trajectory Model with Reinforcement Learning from Human Feedback 

**Title (ZH)**: 基于人类反馈的生成轨迹模型强化学习微调 

**Authors**: Derun Li, Jianwei Ren, Yue Wang, Xin Wen, Pengxiang Li, Leimeng Xu, Kun Zhan, Zhongpu Xia, Peng Jia, Xianpeng Lang, Ningyi Xu, Hang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10434)  

**Abstract**: Generating human-like and adaptive trajectories is essential for autonomous driving in dynamic environments. While generative models have shown promise in synthesizing feasible trajectories, they often fail to capture the nuanced variability of human driving styles due to dataset biases and distributional shifts. To address this, we introduce TrajHF, a human feedback-driven finetuning framework for generative trajectory models, designed to align motion planning with diverse driving preferences. TrajHF incorporates multi-conditional denoiser and reinforcement learning with human feedback to refine multi-modal trajectory generation beyond conventional imitation learning. This enables better alignment with human driving preferences while maintaining safety and feasibility constraints. TrajHF achieves PDMS of 93.95 on NavSim benchmark, significantly exceeding other methods. TrajHF sets a new paradigm for personalized and adaptable trajectory generation in autonomous driving. 

**Abstract (ZH)**: 生成符合人类行为和适应性强的轨迹对于动态环境下的自主驾驶至关重要。虽然生成模型在合成可行轨迹方面显示出潜力，但由于数据集偏差和分布转移，它们往往无法捕捉人类驾驶风格的细微差异。为此，我们提出了一种基于人类反馈的生成轨迹模型微调框架TrajHF，旨在使运动规划与多样化的驾驶偏好相一致。TrajHF 结合多条件去噪器和强化学习与人类反馈，超越传统模仿学习来细化多模态轨迹生成。这使得轨迹生成更好地与人类驾驶偏好相一致，同时保持安全和可行性约束。在NavSim基准测试中，TrajHF 达到了93.95的PDMS，远超其他方法。TrajHF 为自主驾驶中的个性化和适应性强的轨迹生成树立了新的范式。 

---
# Compliant Control of Quadruped Robots for Assistive Load Carrying 

**Title (ZH)**: 四足机器人辅助负载携带的顺应控制 

**Authors**: Nimesh Khandelwal, Amritanshu Manu, Shakti S. Gupta, Mangal Kothari, Prashanth Krishnamurthy, Farshad Khorrami  

**Link**: [PDF](https://arxiv.org/pdf/2503.10401)  

**Abstract**: This paper presents a novel method for assistive load carrying using quadruped robots. The controller uses proprioceptive sensor data to estimate external base wrench, that is used for precise control of the robot's acceleration during payload transport. The acceleration is controlled using a combination of admittance control and Control Barrier Function (CBF) based quadratic program (QP). The proposed controller rejects disturbances and maintains consistent performance under varying load conditions. Additionally, the built-in CBF guarantees collision avoidance with the collaborative agent in front of the robot. The efficacy of the overall controller is shown by its implementation on the physical hardware as well as numerical simulations. The proposed control framework aims to enhance the quadruped robot's ability to perform assistive tasks in various scenarios, from industrial applications to search and rescue operations. 

**Abstract (ZH)**: 本文提出了一种用于四足机器人辅助负载携带的新方法。控制器利用本体感受传感器数据来估计外部基座力矩，用于负载运输期间对机器人加速度的精确控制。加速度通过结合顺应控制和基于控制屏障函数（CBF）的二次规划（QP）进行控制。所提出的控制器能够抵消干扰并在不同负载条件下保持一致的性能。此外，内置的CBF保证了与机器人前方的合作代理之间的防碰撞。通过在物理硬件及数值模拟上的实现，展示了整体控制器的有效性。提出的控制框架旨在增强四足机器人在各种场景下执行辅助任务的能力，从工业应用到搜救操作。 

---
# LUMOS: Language-Conditioned Imitation Learning with World Models 

**Title (ZH)**: LUMOS：基于语言条件的世界模型imitation learning 

**Authors**: Iman Nematollahi, Branton DeMoss, Akshay L Chandra, Nick Hawes, Wolfram Burgard, Ingmar Posner  

**Link**: [PDF](https://arxiv.org/pdf/2503.10370)  

**Abstract**: We introduce LUMOS, a language-conditioned multi-task imitation learning framework for robotics. LUMOS learns skills by practicing them over many long-horizon rollouts in the latent space of a learned world model and transfers these skills zero-shot to a real robot. By learning on-policy in the latent space of the learned world model, our algorithm mitigates policy-induced distribution shift which most offline imitation learning methods suffer from. LUMOS learns from unstructured play data with fewer than 1% hindsight language annotations but is steerable with language commands at test time. We achieve this coherent long-horizon performance by combining latent planning with both image- and language-based hindsight goal relabeling during training, and by optimizing an intrinsic reward defined in the latent space of the world model over multiple time steps, effectively reducing covariate shift. In experiments on the difficult long-horizon CALVIN benchmark, LUMOS outperforms prior learning-based methods with comparable approaches on chained multi-task evaluations. To the best of our knowledge, we are the first to learn a language-conditioned continuous visuomotor control for a real-world robot within an offline world model. Videos, dataset and code are available at this http URL. 

**Abstract (ZH)**: LUMOS：一种语言条件下的多任务模仿学习框架及其在机器人领域的应用 

---
# CODEI: Resource-Efficient Task-Driven Co-Design of Perception and Decision Making for Mobile Robots Applied to Autonomous Vehicles 

**Title (ZH)**: CODEI: 资源高效的任务驱动感知与决策协同设计及其在自主车辆中的应用 

**Authors**: Dejan Milojevic, Gioele Zardini, Miriam Elser, Andrea Censi, Emilio Frazzoli  

**Link**: [PDF](https://arxiv.org/pdf/2503.10296)  

**Abstract**: This paper discusses the integration challenges and strategies for designing mobile robots, by focusing on the task-driven, optimal selection of hardware and software to balance safety, efficiency, and minimal usage of resources such as costs, energy, computational requirements, and weight. We emphasize the interplay between perception and motion planning in decision-making by introducing the concept of occupancy queries to quantify the perception requirements for sampling-based motion planners. Sensor and algorithm performance are evaluated using False Negative Rates (FPR) and False Positive Rates (FPR) across various factors such as geometric relationships, object properties, sensor resolution, and environmental conditions. By integrating perception requirements with perception performance, an Integer Linear Programming (ILP) approach is proposed for efficient sensor and algorithm selection and placement. This forms the basis for a co-design optimization that includes the robot body, motion planner, perception pipeline, and computing unit. We refer to this framework for solving the co-design problem of mobile robots as CODEI, short for Co-design of Embodied Intelligence. A case study on developing an Autonomous Vehicle (AV) for urban scenarios provides actionable information for designers, and shows that complex tasks escalate resource demands, with task performance affecting choices of the autonomy stack. The study demonstrates that resource prioritization influences sensor choice: cameras are preferred for cost-effective and lightweight designs, while lidar sensors are chosen for better energy and computational efficiency. 

**Abstract (ZH)**: 本论文讨论了移动机器人设计中的集成挑战与策略，重点关注任务驱动下的硬件和软件优化选择，以平衡安全、效率及成本、能耗、计算需求和重量等资源的最小使用。文中强调感知与运动规划之间的交互作用，在决策中引入占用查询的概念，以量化基于采样运动规划器的感知需求。传感器和算法性能通过误判率（False Negative Rates 和 False Positive Rates）在几何关系、物体属性、传感器分辨率及环境条件等多种因素下进行评估。通过将感知需求与感知性能集成，提出了整数线性规划（Integer Linear Programming）方法，用于高效选择和布置传感器与算法。该方法为基础，涵盖了机器人本体、运动规划器、感知流水线和计算单元的协同设计优化。文中提出的一种解决移动机器人协同设计问题的框架称为CODEI，即协同设计体域智能。一个针对城市场景的自主车辆（AV）开发案例研究为设计师提供了实用信息，表明复杂的任务会增加资源需求，任务性能影响自主堆栈的选择。研究显示，资源优先级影响传感器选择：成本效益高且轻量化设计倾向于使用摄像头，而为了更好的能耗和计算效率，则选择激光雷达传感器。 

---
# An Real-Sim-Real (RSR) Loop Framework for Generalizable Robotic Policy Transfer with Differentiable Simulation 

**Title (ZH)**: Real-Sim-Real (RSR) 循环框架：具有可微模拟的通用机器人策略转移 

**Authors**: Lu Shi, Yuxuan Xu, Shiyu Wang, Jinhao Huang, Wenhao Zhao, Yufei Jia, Zike Yan, Weibin Gu, Guyue Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.10118)  

**Abstract**: The sim-to-real gap remains a critical challenge in robotics, hindering the deployment of algorithms trained in simulation to real-world systems. This paper introduces a novel Real-Sim-Real (RSR) loop framework leveraging differentiable simulation to address this gap by iteratively refining simulation parameters, aligning them with real-world conditions, and enabling robust and efficient policy transfer. A key contribution of our work is the design of an informative cost function that encourages the collection of diverse and representative real-world data, minimizing bias and maximizing the utility of each data point for simulation refinement. This cost function integrates seamlessly into existing reinforcement learning algorithms (e.g., PPO, SAC) and ensures a balanced exploration of critical regions in the real domain. Furthermore, our approach is implemented on the versatile Mujoco MJX platform, and our framework is compatible with a wide range of robotic systems. Experimental results on several robotic manipulation tasks demonstrate that our method significantly reduces the sim-to-real gap, achieving high task performance and generalizability across diverse scenarios of both explicit and implicit environmental uncertainties. 

**Abstract (ZH)**: 机器人领域的模拟到现实差距仍然是一个关键挑战，阻碍了在模拟中训练的算法在现实世界系统中的部署。本文介绍了一种新型的Real-Sim-Real (RSR)循环框架，通过利用可微分模拟逐步细化仿真参数，使其与现实世界条件相匹配，从而解决这一差距，实现稳健且高效的策略转移。我们工作的一个重要贡献是设计了一种信息丰富的成本函数，该函数鼓励收集多样化和代表性的现实世界数据，最小化偏差并最大化每个数据点对仿真细化的效用。该成本函数能够无缝集成到现有的强化学习算法（如PPO、SAC）中，并确保在现实域中对关键区域进行平衡探索。此外，我们的方法在多功能的Mujoco MJX平台上实现，并且我们的框架兼容多种类型的机器人系统。在多种机器人操作任务上的实验结果表明，我们的方法显著减少了模拟到现实的差距，实现了在不同环境不确定性的显式和隐式情景中的高任务性能和泛化能力。 

---
# IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models 

**Title (ZH)**: 基于视觉语言模型的可接受接触轨迹的智能运动规划 

**Authors**: Yiyang Ling, Karan Owalekar, Oluwatobiloba Adesanya, Erdem Bıyık, Daniel Seita  

**Link**: [PDF](https://arxiv.org/pdf/2503.10110)  

**Abstract**: Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g., brushing a soft pillow) to more dangerous (e.g., toppling a glass vase). Due to this diversity, it is difficult to characterize which contacts may be acceptable or unacceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach uses the VLM's outputs to produce a dense 3D "cost map" that encodes contact tolerances and seamlessly integrates with standard motion planners. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3620 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Supplementary material is available at this https URL. 

**Abstract (ZH)**: 基于视觉-语言模型的交互容许性运动规划框架 

---
# AhaRobot: A Low-Cost Open-Source Bimanual Mobile Manipulator for Embodied AI 

**Title (ZH)**: AhaRobot：一种低成本开源双臂移动 manipulator 用于具身 AI 

**Authors**: Haiqin Cui, Yifu Yuan, Yan Zheng, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2503.10070)  

**Abstract**: Navigation and manipulation in open-world environments remain unsolved challenges in the Embodied AI. The high cost of commercial mobile manipulation robots significantly limits research in real-world scenes. To address this issue, we propose AhaRobot, a low-cost and fully open-source dual-arm mobile manipulation robot system with a hardware cost of only $1,000 (excluding optional computational resources), which is less than 1/15 of the cost of popular mobile robots. The AhaRobot system consists of three components: (1) a novel low-cost hardware architecture primarily composed of off-the-shelf components, (2) an optimized control solution to enhance operational precision integrating dual-motor backlash control and static friction compensation, and (3) a simple remote teleoperation method RoboPilot. We use handles to control the dual arms and pedals for whole-body movement. The teleoperation process is low-burden and easy to operate, much like piloting. RoboPilot is designed for remote data collection in embodied scenarios. Experimental results demonstrate that RoboPilot significantly enhances data collection efficiency in complex manipulation tasks, achieving a 30% increase compared to methods using 3D mouse and leader-follower systems. It also excels at completing extremely long-horizon tasks in one go. Furthermore, AhaRobot can be used to learn end-to-end policies and autonomously perform complex manipulation tasks, such as pen insertion and cleaning up the floor. We aim to build an affordable yet powerful platform to promote the development of embodied tasks on real devices, advancing more robust and reliable embodied AI. All hardware and software systems are available at this https URL. 

**Abstract (ZH)**: 开放世界环境中的导航与操控仍然是约束型AI中的未解决挑战。商业移动操控机器人高昂的成本严重限制了现实场景中的研究。为应对这一问题，我们提出了AhaRobot，一个硬件成本仅为1000美元（不包括可选计算资源）的低成本全开源双臂移动操控机器人系统，成本低于流行移动机器人的1/15。AhaRobot系统由三个组成部分组成：（1）一种新颖的低成本硬件架构，主要由现成组件组成，（2）一种优化的控制解决方案，集成双电机反向间隙控制和静摩擦补偿以提高操作精度，以及（3）一种简单的远程遥控方法RoboPilot。使用手柄控制双臂，使用脚踏板进行全身运动。远程操控过程负担低且易于操作，类似于驾驶。RoboPilot旨在远程数据收集中的约束型场景中使用。实验结果表明，RoboPilot显著提高了复杂操控任务的数据采集效率，相比使用3D鼠标和领导者-追随者系统的相关方法，数据收集效率提高了30%。它还能够一次性完成极长视角的任务。此外，AhaRobot可用于学习端到端策略，并自主执行复杂的操控任务，如笔插入和清洁地面。我们的目标是建立一个负担得起但又强大的平台，促进在真实设备上进行约束性任务的发展，推进更稳健可靠的约束型AI。所有硬件和软件系统均可从<a href="this https URL">这里</a>获得。 

---
# SmartWay: Enhanced Waypoint Prediction and Backtracking for Zero-Shot Vision-and-Language Navigation 

**Title (ZH)**: SmartWay: 增强的航点预测与回溯方法用于零样本视觉与语言导航 

**Authors**: Xiangyu Shi, Zerui Li, Wenqi Lyu, Jiatong Xia, Feras Dayoub, Yanyuan Qiao, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10069)  

**Abstract**: Vision-and-Language Navigation (VLN) in continuous environments requires agents to interpret natural language instructions while navigating unconstrained 3D spaces. Existing VLN-CE frameworks rely on a two-stage approach: a waypoint predictor to generate waypoints and a navigator to execute movements. However, current waypoint predictors struggle with spatial awareness, while navigators lack historical reasoning and backtracking capabilities, limiting adaptability. We propose a zero-shot VLN-CE framework integrating an enhanced waypoint predictor with a Multi-modal Large Language Model (MLLM)-based navigator. Our predictor employs a stronger vision encoder, masked cross-attention fusion, and an occupancy-aware loss for better waypoint quality. The navigator incorporates history-aware reasoning and adaptive path planning with backtracking, improving robustness. Experiments on R2R-CE and MP3D benchmarks show our method achieves state-of-the-art (SOTA) performance in zero-shot settings, demonstrating competitive results compared to fully supervised methods. Real-world validation on Turtlebot 4 further highlights its adaptability. 

**Abstract (ZH)**: 连续环境中的视觉-语言导航（VLN）要求代理在未受约束的3D空间中解释自然语言指令进行导航。现有的VLN-CE框架依赖两阶段方法：道点预测器生成道点和导航器执行移动。然而，当前的道点预测器在空间意识方面存在局限性，而导航器缺乏历史推理和回溯能力，限制了其适应性。我们提出了一种零样本VLN-CE框架，结合了增强的道点预测器和基于多模态大语言模型（MLLM）的导航器。我们的预测器采用了更强的视觉编码器、掩码交叉注意力融合和占用感知损失，以提高道点质量。导航器融合了历史感知推理、自适应路径规划和回溯功能，提升了稳健性。在R2R-CE和MP3D基准测试上的实验表明，我们的方法在零样本设置中实现了最优性能，且与全监督方法相比具有竞争力。实际验证在Turtlebot 4上进一步突显了其适应性。 

---
# ES-Parkour: Advanced Robot Parkour with Bio-inspired Event Camera and Spiking Neural Network 

**Title (ZH)**: ES-Parkour: 基于生物灵感事件摄像头和脉冲神经网络的高级机器人特技跳跃技术 

**Authors**: Qiang Zhang, Jiahang Cao, Jingkai Sun, Gang Han, Wen Zhao, Yijie Guo, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.09985)  

**Abstract**: In recent years, quadruped robotics has advanced significantly, particularly in perception and motion control via reinforcement learning, enabling complex motions in challenging environments. Visual sensors like depth cameras enhance stability and robustness but face limitations, such as low operating frequencies relative to joint control and sensitivity to lighting, which hinder outdoor deployment. Additionally, deep neural networks in sensor and control systems increase computational demands. To address these issues, we introduce spiking neural networks (SNNs) and event cameras to perform a challenging quadruped parkour task. Event cameras capture dynamic visual data, while SNNs efficiently process spike sequences, mimicking biological perception. Experimental results demonstrate that this approach significantly outperforms traditional models, achieving excellent parkour performance with just 11.7% of the energy consumption of an artificial neural network (ANN)-based model, yielding an 88.3% energy reduction. By integrating event cameras with SNNs, our work advances robotic reinforcement learning and opens new possibilities for applications in demanding environments. 

**Abstract (ZH)**: 近年来，四足机器人取得了显著进展，特别是在通过强化学习实现感知和运动控制方面，使其能够在复杂环境中执行复杂的动作。视觉传感器如深度相机提高了稳定性和鲁棒性，但面对着与关节控制相比较低的操作频率以及对光照敏感等局限性，这阻碍了其在户外的应用。此外，传感器和控制系统中的深度神经网络增加了计算需求。为了解决这些问题，我们引入了尖峰神经网络(SNNs)和事件相机来执行一项具有挑战性的四足公园运动任务。事件相机捕捉动态视觉数据，而SNNs高效地处理尖峰序列，模拟生物感知。实验结果表明，这种方法在能耗方面显著优于传统的模型，仅消耗人工神经网络(ANN)模型能耗的11.7%，实现了88.3%的能耗降低。通过将事件相机与SNNs结合，我们的工作推进了机器人强化学习并为在苛刻环境中的应用开辟了新可能性。 

---
# RMG: Real-Time Expressive Motion Generation with Self-collision Avoidance for 6-DOF Companion Robotic Arms 

**Title (ZH)**: RMG: 实时逼真运动生成与6-自由度伴侣机器人手臂自碰撞避免 

**Authors**: Jiansheng Li, Haotian Song, Jinni Zhou, Qiang Nie, Yi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.09959)  

**Abstract**: The six-degree-of-freedom (6-DOF) robotic arm has gained widespread application in human-coexisting environments. While previous research has predominantly focused on functional motion generation, the critical aspect of expressive motion in human-robot interaction remains largely unexplored. This paper presents a novel real-time motion generation planner that enhances interactivity by creating expressive robotic motions between arbitrary start and end states within predefined time constraints. Our approach involves three key contributions: first, we develop a mapping algorithm to construct an expressive motion dataset derived from human dance movements; second, we train motion generation models in both Cartesian and joint spaces using this dataset; third, we introduce an optimization algorithm that guarantees smooth, collision-free motion while maintaining the intended expressive style. Experimental results demonstrate the effectiveness of our method, which can generate expressive and generalized motions in under 0.5 seconds while satisfying all specified constraints. 

**Abstract (ZH)**: 基于六自由度robotic臂的实时具有表现力的运动生成规划在人机共存环境中的应用 

---
# SE(3)-Equivariant Robot Learning and Control: A Tutorial Survey 

**Title (ZH)**: SE(3)对称性机器人学习与控制：一个综述教程 

**Authors**: Joohwan Seo, Soochul Yoo, Junwoo Chang, Hyunseok An, Hyunwoo Ryu, Soomi Lee, Arvind Kruthiventy, Jongeun CHoi, Roberto Horowitz  

**Link**: [PDF](https://arxiv.org/pdf/2503.09829)  

**Abstract**: Recent advances in deep learning and Transformers have driven major breakthroughs in robotics by employing techniques such as imitation learning, reinforcement learning, and LLM-based multimodal perception and decision-making. However, conventional deep learning and Transformer models often struggle to process data with inherent symmetries and invariances, typically relying on large datasets or extensive data augmentation. Equivariant neural networks overcome these limitations by explicitly integrating symmetry and invariance into their architectures, leading to improved efficiency and generalization. This tutorial survey reviews a wide range of equivariant deep learning and control methods for robotics, from classic to state-of-the-art, with a focus on SE(3)-equivariant models that leverage the natural 3D rotational and translational symmetries in visual robotic manipulation and control design. Using unified mathematical notation, we begin by reviewing key concepts from group theory, along with matrix Lie groups and Lie algebras. We then introduce foundational group-equivariant neural network design and show how the group-equivariance can be obtained through their structure. Next, we discuss the applications of SE(3)-equivariant neural networks in robotics in terms of imitation learning and reinforcement learning. The SE(3)-equivariant control design is also reviewed from the perspective of geometric control. Finally, we highlight the challenges and future directions of equivariant methods in developing more robust, sample-efficient, and multi-modal real-world robotic systems. 

**Abstract (ZH)**: 近期深度学习和变换器的进展通过使用imitation学习、reinforcement学习以及基于大语言模型的多模态感知与决策等技术，在机器人学中取得了重大突破。然而，常规的深度学习和变换器模型通常难以处理具有内在对称性和不变性的数据，通常依赖于大数据集或大量数据增强。通过明确将对称性和不变性整合到其架构中，等变神经网络克服了这些限制，提高了效率和泛化能力。本教程综述了从经典到最新的一系列等变深度学习与控制方法在机器人学中的应用，重点关注利用视觉机器人操作与控制设计中固有的3D旋转和平移对称性的SE(3)-等变模型。通过统一的数学符号，我们首先回顾群论中的关键概念，包括矩阵李群和李代数。接着介绍基础的群等变神经网络设计，并展示如何通过其结构获得群等变性。然后讨论SE(3)-等变神经网络在机器人学中的应用，包括imitation学习和reinforcement学习。从几何控制的角度还回顾了SE(3)-等变控制设计。最后，指出等变方法在开发更稳健、样本高效且多模态的现实世界机器人系统方面的挑战和未来方向。 

---
# Vi-LAD: Vision-Language Attention Distillation for Socially-Aware Robot Navigation in Dynamic Environments 

**Title (ZH)**: Vi-LAD: 视觉-语言注意力精炼用于动态环境中的社会意识机器人导航 

**Authors**: Mohamed Elnoor, Kasun Weerakoon, Gershom Seneviratne, Jing Liang, Vignesh Rajagopal, Dinesh Manocha  

**Link**: [PDF](https://arxiv.org/pdf/2503.09820)  

**Abstract**: We introduce Vision-Language Attention Distillation (Vi-LAD), a novel approach for distilling socially compliant navigation knowledge from a large Vision-Language Model (VLM) into a lightweight transformer model for real-time robotic navigation. Unlike traditional methods that rely on expert demonstrations or human-annotated datasets, Vi-LAD performs knowledge distillation and fine-tuning at the intermediate layer representation level (i.e., attention maps) by leveraging the backbone of a pre-trained vision-action model. These attention maps highlight key navigational regions in a given scene, which serve as implicit guidance for socially aware motion planning. Vi-LAD fine-tunes a transformer-based model using intermediate attention maps extracted from the pre-trained vision-action model, combined with attention-like semantic maps constructed from a large VLM. To achieve this, we introduce a novel attention-level distillation loss that fuses knowledge from both sources, generating augmented attention maps with enhanced social awareness. These refined attention maps are then utilized as a traversability costmap within a socially aware model predictive controller (MPC) for navigation. We validate our approach through real-world experiments on a Husky wheeled robot, demonstrating significant improvements over state-of-the-art (SOTA) navigation methods. Our results show up to 14.2% - 50% improvement in success rate, which highlights the effectiveness of Vi-LAD in enabling socially compliant and efficient robot navigation. 

**Abstract (ZH)**: Vision-Language 注意力蒸馏（Vi-LAD）：一种从大型视觉语言模型中提取社会合规导航知识的方法，以实现实时机器人导航中的轻量级转换器模型fine-tuning 

---
# Multi-Agent LLM Actor-Critic Framework for Social Robot Navigation 

**Title (ZH)**: 多代理LLM演员-评论家框架在社会机器人导航中的应用 

**Authors**: Weizheng Wang, Ike Obi, Byung-Cheol Min  

**Link**: [PDF](https://arxiv.org/pdf/2503.09758)  

**Abstract**: Recent advances in robotics and large language models (LLMs) have sparked growing interest in human-robot collaboration and embodied intelligence. To enable the broader deployment of robots in human-populated environments, socially-aware robot navigation (SAN) has become a key research area. While deep reinforcement learning approaches that integrate human-robot interaction (HRI) with path planning have demonstrated strong benchmark performance, they often struggle to adapt to new scenarios and environments. LLMs offer a promising avenue for zero-shot navigation through commonsense inference. However, most existing LLM-based frameworks rely on centralized decision-making, lack robust verification mechanisms, and face inconsistencies in translating macro-actions into precise low-level control signals. To address these challenges, we propose SAMALM, a decentralized multi-agent LLM actor-critic framework for multi-robot social navigation. In this framework, a set of parallel LLM actors, each reflecting distinct robot personalities or configurations, directly generate control signals. These actions undergo a two-tier verification process via a global critic that evaluates group-level behaviors and individual critics that assess each robot's context. An entropy-based score fusion mechanism further enhances self-verification and re-query, improving both robustness and coordination. Experimental results confirm that SAMALM effectively balances local autonomy with global oversight, yielding socially compliant behaviors and strong adaptability across diverse multi-robot scenarios. More details and videos about this work are available at: this https URL. 

**Abstract (ZH)**: Recent advances in robotics and large language models (LLMs) have sparked growing interest in human-robot collaboration and embodied intelligence. 社会感知机器人导航和社会机器人多智能体自适应导航 

---
# Edge AI-Powered Real-Time Decision-Making for Autonomous Vehicles in Adverse Weather Conditions 

**Title (ZH)**: 边缘AI赋能的恶劣天气条件下的自动驾驶车辆实时决策-making 

**Authors**: Milad Rahmati  

**Link**: [PDF](https://arxiv.org/pdf/2503.09638)  

**Abstract**: Autonomous vehicles (AVs) are transforming modern transportation, but their reliability and safety are significantly challenged by harsh weather conditions such as heavy rain, fog, and snow. These environmental factors impair the performance of cameras, LiDAR, and radar, leading to reduced situational awareness and increased accident risks. Conventional cloud-based AI systems introduce communication delays, making them unsuitable for the rapid decision-making required in real-time autonomous navigation. This paper presents a novel Edge AI-driven real-time decision-making framework designed to enhance AV responsiveness under adverse weather conditions. The proposed approach integrates convolutional neural networks (CNNs) and recurrent neural networks (RNNs) for improved perception, alongside reinforcement learning (RL)-based strategies to optimize vehicle control in uncertain environments. By processing data at the network edge, this system significantly reduces decision latency while improving AV adaptability. The framework is evaluated using simulated driving scenarios in CARLA and real-world data from the Waymo Open Dataset, covering diverse weather conditions. Experimental results indicate that the proposed model achieves a 40% reduction in processing time and a 25% enhancement in perception accuracy compared to conventional cloud-based systems. These findings highlight the potential of Edge AI in improving AV autonomy, safety, and efficiency, paving the way for more reliable self-driving technology in challenging real-world environments. 

**Abstract (ZH)**: 基于边缘AI的恶劣天气下实时决策框架：提升自动驾驶车辆响应性 

---
# Real-Time Neuromorphic Navigation: Guiding Physical Robots with Event-Based Sensing and Task-Specific Reconfigurable Autonomy Stack 

**Title (ZH)**: 实时神经形态导航：基于事件的传感器数据和任务特定可配置自主栈引导物理机器人 

**Authors**: Sourav Sanyal, Amogh Joshi, Adarsh Kosta, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2503.09636)  

**Abstract**: Neuromorphic vision, inspired by biological neural systems, has recently gained significant attention for its potential in enhancing robotic autonomy. This paper presents a systematic exploration of a proposed Neuromorphic Navigation framework that uses event-based neuromorphic vision to enable efficient, real-time navigation in robotic systems. We discuss the core concepts of neuromorphic vision and navigation, highlighting their impact on improving robotic perception and decision-making. The proposed reconfigurable Neuromorphic Navigation framework adapts to the specific needs of both ground robots (Turtlebot) and aerial robots (Bebop2 quadrotor), addressing the task-specific design requirements (algorithms) for optimal performance across the autonomous navigation stack -- Perception, Planning, and Control. We demonstrate the versatility and the effectiveness of the framework through two case studies: a Turtlebot performing local replanning for real-time navigation and a Bebop2 quadrotor navigating through moving gates. Our work provides a scalable approach to task-specific, real-time robot autonomy leveraging neuromorphic systems, paving the way for energy-efficient autonomous navigation. 

**Abstract (ZH)**: 类脑视觉启发于生物神经系统，近年来因其在增强机器人自主性方面的潜在优势而受到广泛关注。本文提出了一种系统性的类脑导航框架的研究，该框架利用事件驱动的类脑视觉实现机器人系统的高效、实时导航。我们讨论了类脑视觉和导航的核心概念，强调其对提高机器人感知和决策的影响。提出的可重构类脑导航框架针对地面机器人（Turtlebot）和飞行机器人（Bebop2 四旋翼）的具体需求进行适应，解决了自主导航堆栈——感知、规划和控制——中的任务特定设计要求，以实现最优性能。我们通过两个案例研究展示了该框架的多样性和有效性：一个Turtlebot进行局部重规划以实现实时导航，一个Bebop2 四旋翼飞行器穿越移动门。我们的工作提供了一种利用类脑系统实现任务特定、实时机器人自主性的可扩展方法，为高效自主导航铺平了道路。 

---
# HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model 

**Title (ZH)**: HybridVLA: 统一视觉-语言-行动模型中的协作扩散和自回归 

**Authors**: Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, Chengkai Hou, Mengdi Zhao, KC alex Zhou, Pheng-Ann Heng, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.10631)  

**Abstract**: Recent advancements in vision-language models (VLMs) for common-sense reasoning have led to the development of vision-language-action (VLA) models, enabling robots to perform generalized manipulation. Although existing autoregressive VLA methods leverage large-scale pretrained knowledge, they disrupt the continuity of actions. Meanwhile, some VLA methods incorporate an additional diffusion head to predict continuous actions, relying solely on VLM-extracted features, which limits their reasoning capabilities. In this paper, we introduce HybridVLA, a unified framework that seamlessly integrates the strengths of both autoregressive and diffusion policies within a single large language model, rather than simply connecting them. To bridge the generation gap, a collaborative training recipe is proposed that injects the diffusion modeling directly into the next-token prediction. With this recipe, we find that these two forms of action prediction not only reinforce each other but also exhibit varying performance across different tasks. Therefore, we design a collaborative action ensemble mechanism that adaptively fuses these two predictions, leading to more robust control. In experiments, HybridVLA outperforms previous state-of-the-art VLA methods across various simulation and real-world tasks, including both single-arm and dual-arm robots, while demonstrating stable manipulation in previously unseen configurations. 

**Abstract (ZH)**: 近期视觉-语言模型在常识推理方面的最新进展促进了视觉-语言-动作（VLA）模型的发展，使机器人能够执行通用操作。尽管现有的自回归VLA方法利用了大规模预训练知识，但它们破坏了动作的连续性。同时，一些VLA方法引入了额外的扩散头来预测连续动作，仅依靠VLM提取的特征，这限制了它们的推理能力。本文介绍了一种名为HybridVLA的统一框架，该框架在一个大型语言模型中无缝结合了自回归和扩散策略的优点，而不仅仅将它们连接起来。为了弥补生成上的差距，我们提出了一种协作训练方案，直接将扩散建模注入下一个token的预测。通过这种方式，我们发现这两种形式的动作预测不仅互相强化，而且在不同任务上表现出不同的性能。因此，我们设计了一种协作的动作集成机制，能够自适应地融合这两种预测，从而实现更 robust 的控制。实验结果显示，HybridVLA在各种模拟和现实任务中均优于之前的VLA方法，包括单臂和双臂机器人，并在未见过的配置中展示了稳定的操作。 

---
# UniGoal: Towards Universal Zero-shot Goal-oriented Navigation 

**Title (ZH)**: UniGoal: 向普遍零样本目标导向导航迈进 

**Authors**: Hang Yin, Xiuwei Xu, Lingqing Zhao, Ziwei Wang, Jie Zhou, Jiwen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10630)  

**Abstract**: In this paper, we propose a general framework for universal zero-shot goal-oriented navigation. Existing zero-shot methods build inference framework upon large language models (LLM) for specific tasks, which differs a lot in overall pipeline and fails to generalize across different types of goal. Towards the aim of universal zero-shot navigation, we propose a uniform graph representation to unify different goals, including object category, instance image and text description. We also convert the observation of agent into an online maintained scene graph. With this consistent scene and goal representation, we preserve most structural information compared with pure text and are able to leverage LLM for explicit graph-based reasoning. Specifically, we conduct graph matching between the scene graph and goal graph at each time instant and propose different strategies to generate long-term goal of exploration according to different matching states. The agent first iteratively searches subgraph of goal when zero-matched. With partial matching, the agent then utilizes coordinate projection and anchor pair alignment to infer the goal location. Finally scene graph correction and goal verification are applied for perfect matching. We also present a blacklist mechanism to enable robust switch between stages. Extensive experiments on several benchmarks show that our UniGoal achieves state-of-the-art zero-shot performance on three studied navigation tasks with a single model, even outperforming task-specific zero-shot methods and supervised universal methods. 

**Abstract (ZH)**: 基于统一图表示的通用零样本目标导向导航框架 

---
# NIL: No-data Imitation Learning by Leveraging Pre-trained Video Diffusion Models 

**Title (ZH)**: NIL: 无数据模仿学习通过利用预训练视频扩散模型 

**Authors**: Mert Albaba, Chenhao Li, Markos Diomataris, Omid Taheri, Andreas Krause, Michael Black  

**Link**: [PDF](https://arxiv.org/pdf/2503.10626)  

**Abstract**: Acquiring physically plausible motor skills across diverse and unconventional morphologies-including humanoid robots, quadrupeds, and animals-is essential for advancing character simulation and robotics. Traditional methods, such as reinforcement learning (RL) are task- and body-specific, require extensive reward function engineering, and do not generalize well. Imitation learning offers an alternative but relies heavily on high-quality expert demonstrations, which are difficult to obtain for non-human morphologies. Video diffusion models, on the other hand, are capable of generating realistic videos of various morphologies, from humans to ants. Leveraging this capability, we propose a data-independent approach for skill acquisition that learns 3D motor skills from 2D-generated videos, with generalization capability to unconventional and non-human forms. Specifically, we guide the imitation learning process by leveraging vision transformers for video-based comparisons by calculating pair-wise distance between video embeddings. Along with video-encoding distance, we also use a computed similarity between segmented video frames as a guidance reward. We validate our method on locomotion tasks involving unique body configurations. In humanoid robot locomotion tasks, we demonstrate that 'No-data Imitation Learning' (NIL) outperforms baselines trained on 3D motion-capture data. Our results highlight the potential of leveraging generative video models for physically plausible skill learning with diverse morphologies, effectively replacing data collection with data generation for imitation learning. 

**Abstract (ZH)**: 跨多种非传统形态，包括人形机器人、四足动物和动物，获取物理上合理的行为技能对于推进角色模拟和机器人技术至关重要。传统的强化学习方法针对性强、需要大量奖励函数工程并且不具备较好的泛化能力。模仿学习提供了一种替代方案，但需要高质量的专家示范，这在非人类形态的情况下难以获得。视频扩散模型能够生成从人类到蚂蚁等多种形态的逼真视频。利用这一能力，我们提出了一种数据独立的行为技能获取方法，可以从生成的2D视频中学习3D运动技能，并具备对非传统和非人类形态的泛化能力。具体而言，我们通过使用视觉变换器进行基于视频的距离比较来指导模仿学习过程，计算视频嵌入的成对距离。同时，我们还利用分割视频帧之间的计算相似度作为辅助奖励。我们在涉及独特身体配置的运动任务中验证了该方法。在人形机器人运动任务中，我们证明了“无数据模仿学习”（NIL）优于基于3D动作捕捉数据训练的基础模型。我们的结果 Highlights 了借助生成视频模型进行多形态物理上合理技能学习的潜力，有效地用数据生成替代了数据收集在模仿学习中的应用。 

---
# World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning 

**Title (ZH)**: 世界建模让计划更加出色：体现偏好优化在实体任务规划中的应用 

**Authors**: Siyin Wang, Zhaoye Fei, Qinyuan Cheng, Shiduo Zhang, Panpan Cai, Jinlan Fu, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.10480)  

**Abstract**: Recent advances in large vision-language models (LVLMs) have shown promise for embodied task planning, yet they struggle with fundamental challenges like dependency constraints and efficiency. Existing approaches either solely optimize action selection or leverage world models during inference, overlooking the benefits of learning to model the world as a way to enhance planning capabilities. We propose Dual Preference Optimization (D$^2$PO), a new learning framework that jointly optimizes state prediction and action selection through preference learning, enabling LVLMs to understand environment dynamics for better planning. To automatically collect trajectories and stepwise preference data without human annotation, we introduce a tree search mechanism for extensive exploration via trial-and-error. Extensive experiments on VoTa-Bench demonstrate that our D$^2$PO-based method significantly outperforms existing methods and GPT-4o when applied to Qwen2-VL (7B), LLaVA-1.6 (7B), and LLaMA-3.2 (11B), achieving superior task success rates with more efficient execution paths. 

**Abstract (ZH)**: Recent advances in大型视觉-语言模型（LVLMs）在实现嵌入式任务规划方面显示出了潜力，但仍然面临诸如依赖约束和效率等基本挑战。现有方法要么仅优化动作选择，要么在推理过程中利用世界模型，忽略了通过学习建模世界来增强规划能力的益处。我们提出了双偏好优化（D$^2$PO），这是一种新的学习框架，通过偏好学习同时优化状态预测和动作选择，使LVLMs能够更好地理解环境动力学并进行规划。为自动收集轨迹和步进偏好数据而无需人工标注，我们引入了一种通过试错进行广泛探索的树搜索机制。在VoTa-Bench上的 extensive 实验表明，基于D$^2$PO的方法在应用于Qwen2-VL（7B）、LLaVA-1.6（7B）和LLaMA-3.2（11B）时，能够实现更高的任务成功率并采用更高效的执行路径。 

---
# A nonlinear real time capable motion cueing algorithm based on deep reinforcement learning 

**Title (ZH)**: 基于深度强化学习的非线性实时运动模拟算法 

**Authors**: Hendrik Scheidel, Camilo Gonzalez, Houshyar Asadi, Tobias Bellmann, Andreas Seefried, Shady Mohamed, Saeid Nahavandi  

**Link**: [PDF](https://arxiv.org/pdf/2503.10419)  

**Abstract**: In motion simulation, motion cueing algorithms are used for the trajectory planning of the motion simulator platform, where workspace limitations prevent direct reproduction of reference trajectories. Strategies such as motion washout, which return the platform to its center, are crucial in these settings. For serial robotic MSPs with highly nonlinear workspaces, it is essential to maximize the efficient utilization of the MSPs kinematic and dynamic capabilities. Traditional approaches, including classical washout filtering and linear model predictive control, fail to consider platform-specific, nonlinear properties, while nonlinear model predictive control, though comprehensive, imposes high computational demands that hinder real-time, pilot-in-the-loop application without further simplification. To overcome these limitations, we introduce a novel approach using deep reinforcement learning for motion cueing, demonstrated here for the first time in a 6-degree-of-freedom setting with full consideration of the MSPs kinematic nonlinearities. Previous work by the authors successfully demonstrated the application of DRL to a simplified 2-DOF setup, which did not consider kinematic or dynamic constraints. This approach has been extended to all 6 DOF by incorporating a complete kinematic model of the MSP into the algorithm, a crucial step for enabling its application on a real motion simulator. The training of the DRL-MCA is based on Proximal Policy Optimization in an actor-critic implementation combined with an automated hyperparameter optimization. After detailing the necessary training framework and the algorithm itself, we provide a comprehensive validation, demonstrating that the DRL MCA achieves competitive performance against established algorithms. Moreover, it generates feasible trajectories by respecting all system constraints and meets all real-time requirements with low... 

**Abstract (ZH)**: 基于深度强化学习的运动提示算法在6自由度运动模拟器平台上的应用：考虑运动平台的非线性特性 

---
# 6D Object Pose Tracking in Internet Videos for Robotic Manipulation 

**Title (ZH)**: 互联网视频中基于6D姿态的物体追踪及其在机器人操作中的应用 

**Authors**: Georgy Ponimatkin, Martin Cífka, Tomáš Souček, Médéric Fourmy, Yann Labbé, Vladimir Petrik, Josef Sivic  

**Link**: [PDF](https://arxiv.org/pdf/2503.10307)  

**Abstract**: We seek to extract a temporally consistent 6D pose trajectory of a manipulated object from an Internet instructional video. This is a challenging set-up for current 6D pose estimation methods due to uncontrolled capturing conditions, subtle but dynamic object motions, and the fact that the exact mesh of the manipulated object is not known. To address these challenges, we present the following contributions. First, we develop a new method that estimates the 6D pose of any object in the input image without prior knowledge of the object itself. The method proceeds by (i) retrieving a CAD model similar to the depicted object from a large-scale model database, (ii) 6D aligning the retrieved CAD model with the input image, and (iii) grounding the absolute scale of the object with respect to the scene. Second, we extract smooth 6D object trajectories from Internet videos by carefully tracking the detected objects across video frames. The extracted object trajectories are then retargeted via trajectory optimization into the configuration space of a robotic manipulator. Third, we thoroughly evaluate and ablate our 6D pose estimation method on YCB-V and HOPE-Video datasets as well as a new dataset of instructional videos manually annotated with approximate 6D object trajectories. We demonstrate significant improvements over existing state-of-the-art RGB 6D pose estimation methods. Finally, we show that the 6D object motion estimated from Internet videos can be transferred to a 7-axis robotic manipulator both in a virtual simulator as well as in a real world set-up. We also successfully apply our method to egocentric videos taken from the EPIC-KITCHENS dataset, demonstrating potential for Embodied AI applications. 

**Abstract (ZH)**: 我们旨在从互联网教学视频中提取受操作物体的时序一致的6D姿态轨迹。这一设定对当前的6D姿态估计方法是具有挑战性的，因为存在不受控的拍摄条件、微妙但动态的物体运动，以及受操作物体的确切网格未知的问题。为应对这些挑战，我们提出了以下贡献。首先，我们开发了一种新方法，该方法可以在没有任何关于物体本身的先验知识的情况下，估计输入图像中任何物体的6D姿态。该方法通过以下步骤进行：(i) 从大规模模型数据库中检索与显示的物体相似的CAD模型，(ii) 6D对齐检索到的CAD模型与输入图像，(iii) 将物体相对于场景的绝对尺度进行定位。其次，通过仔细跨视频帧跟踪检测到的物体，我们从互联网视频中提取平滑的6D物体轨迹，并通过轨迹优化将提取到的物体轨迹重新定向到机械臂配置空间。第三，我们在YCB-V和HOPE-Video数据集以及一个新数据集上全面评估和消融我们的6D姿态估计方法，该数据集由手工注释的大约6D物体轨迹的教学视频构成。我们展示了显著优于现有RGB 6D姿态估计方法的改进。最后，我们证明从互联网视频中估计的6D物体运动可以转移至7轴机械臂，在虚拟仿真器和实际设置中均有效。我们还将我们的方法应用于源自EPIC-KITCHENS数据集的自躯体视角视频，展示了其在具身AI应用中的潜力。 

---
# SurgRAW: Multi-Agent Workflow with Chain-of-Thought Reasoning for Surgical Intelligence 

**Title (ZH)**: SurgRAW: 基于链式思维推理的多agent工作流手术智能化方法 

**Authors**: Chang Han Low, Ziyue Wang, Tianyi Zhang, Zhitao Zeng, Zhu Zhuo, Evangelos B. Mazomenos, Yueming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2503.10265)  

**Abstract**: Integration of Vision-Language Models (VLMs) in surgical intelligence is hindered by hallucinations, domain knowledge gaps, and limited understanding of task interdependencies within surgical scenes, undermining clinical reliability. While recent VLMs demonstrate strong general reasoning and thinking capabilities, they still lack the domain expertise and task-awareness required for precise surgical scene interpretation. Although Chain-of-Thought (CoT) can structure reasoning more effectively, current approaches rely on self-generated CoT steps, which often exacerbate inherent domain gaps and hallucinations. To overcome this, we present SurgRAW, a CoT-driven multi-agent framework that delivers transparent, interpretable insights for most tasks in robotic-assisted surgery. By employing specialized CoT prompts across five tasks: instrument recognition, action recognition, action prediction, patient data extraction, and outcome assessment, SurgRAW mitigates hallucinations through structured, domain-aware reasoning. Retrieval-Augmented Generation (RAG) is also integrated to external medical knowledge to bridge domain gaps and improve response reliability. Most importantly, a hierarchical agentic system ensures that CoT-embedded VLM agents collaborate effectively while understanding task interdependencies, with a panel discussion mechanism promotes logical consistency. To evaluate our method, we introduce SurgCoTBench, the first reasoning-based dataset with structured frame-level annotations. With comprehensive experiments, we demonstrate the effectiveness of proposed SurgRAW with 29.32% accuracy improvement over baseline VLMs on 12 robotic procedures, achieving the state-of-the-art performance and advancing explainable, trustworthy, and autonomous surgical assistance. 

**Abstract (ZH)**: Integrating Vision-Language Models in Surgical Intelligence with SurgRAW: Overcoming Hallucinations and Domain Gaps通过SurgRAW克服幻觉和领域差距，将视觉语言模型集成到手术智能中 

---
# PRISM: Preference Refinement via Implicit Scene Modeling for 3D Vision-Language Preference-Based Reinforcement Learning 

**Title (ZH)**: PRISM: 基于隐式场景建模的3D视觉-语言偏好强化学习偏好细化方法 

**Authors**: Yirong Sun, Yanjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.10177)  

**Abstract**: We propose PRISM, a novel framework designed to overcome the limitations of 2D-based Preference-Based Reinforcement Learning (PBRL) by unifying 3D point cloud modeling and future-aware preference refinement. At its core, PRISM adopts a 3D Point Cloud-Language Model (3D-PC-LLM) to mitigate occlusion and viewpoint biases, ensuring more stable and spatially consistent preference signals. Additionally, PRISM leverages Chain-of-Thought (CoT) reasoning to incorporate long-horizon considerations, thereby preventing the short-sighted feedback often seen in static preference comparisons. In contrast to conventional PBRL techniques, this integration of 3D perception and future-oriented reasoning leads to significant gains in preference agreement rates, faster policy convergence, and robust generalization across unseen robotic environments. Our empirical results, spanning tasks such as robotic manipulation and autonomous navigation, highlight PRISM's potential for real-world applications where precise spatial understanding and reliable long-term decision-making are critical. By bridging 3D geometric awareness with CoT-driven preference modeling, PRISM establishes a comprehensive foundation for scalable, human-aligned reinforcement learning. 

**Abstract (ZH)**: PRISM：一种结合3D点云建模和未来导向偏好精化的新型框架 

---
# V2X-ReaLO: An Open Online Framework and Dataset for Cooperative Perception in Reality 

**Title (ZH)**: V2X-ReaLO: 一个面向现实场景的协同感知开放在线框架及数据集 

**Authors**: Hao Xiang, Zhaoliang Zheng, Xin Xia, Seth Z. Zhao, Letian Gao, Zewei Zhou, Tianhui Cai, Yun Zhang, Jiaqi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2503.10034)  

**Abstract**: Cooperative perception enabled by Vehicle-to-Everything (V2X) communication holds significant promise for enhancing the perception capabilities of autonomous vehicles, allowing them to overcome occlusions and extend their field of view. However, existing research predominantly relies on simulated environments or static datasets, leaving the feasibility and effectiveness of V2X cooperative perception especially for intermediate fusion in real-world scenarios largely unexplored. In this work, we introduce V2X-ReaLO, an open online cooperative perception framework deployed on real vehicles and smart infrastructure that integrates early, late, and intermediate fusion methods within a unified pipeline and provides the first practical demonstration of online intermediate fusion's feasibility and performance under genuine real-world conditions. Additionally, we present an open benchmark dataset specifically designed to assess the performance of online cooperative perception systems. This new dataset extends V2X-Real dataset to dynamic, synchronized ROS bags and provides 25,028 test frames with 6,850 annotated key frames in challenging urban scenarios. By enabling real-time assessments of perception accuracy and communication lantency under dynamic conditions, V2X-ReaLO sets a new benchmark for advancing and optimizing cooperative perception systems in real-world applications. The codes and datasets will be released to further advance the field. 

**Abstract (ZH)**: Vehicle-to-Everything (V2X) 驱动的协同感知：一种实际车辆和智能基础设施上的在线协同感知框架及其应用 

---
# Training Human-Robot Teams by Improving Transparency Through a Virtual Spectator Interface 

**Title (ZH)**: 通过虚拟观众界面提高透明度来培训人机团队 

**Authors**: Sean Dallas, Hongjiao Qiang, Motaz AbuHijleh, Wonse Jo, Kayla Riegner, Jon Smereka, Lionel Robert, Wing-Yue Louie, Dawn M. Tilbury  

**Link**: [PDF](https://arxiv.org/pdf/2503.09849)  

**Abstract**: After-action reviews (AARs) are professional discussions that help operators and teams enhance their task performance by analyzing completed missions with peers and professionals. Previous studies that compared different formats of AARs have mainly focused on human teams. However, the inclusion of robotic teammates brings along new challenges in understanding teammate intent and communication. Traditional AAR between human teammates may not be satisfactory for human-robot teams. To address this limitation, we propose a new training review (TR) tool, called the Virtual Spectator Interface (VSI), to enhance human-robot team performance and situational awareness (SA) in a simulated search mission. The proposed VSI primarily utilizes visual feedback to review subjects' behavior. To examine the effectiveness of VSI, we took elements from AAR to conduct our own TR, designed a 1 x 3 between-subjects experiment with experimental conditions: TR with (1) VSI, (2) screen recording, and (3) non-technology (only verbal descriptions). The results of our experiments demonstrated that the VSI did not result in significantly better team performance than other conditions. However, the TR with VSI led to more improvement in the subjects SA over the other conditions. 

**Abstract (ZH)**: 虚拟观众界面（VSI）在模拟搜索任务中提升人机团队绩效与态势感知的训练审查工具 

---
# Distributionally Robust Multi-Agent Reinforcement Learning for Dynamic Chute Mapping 

**Title (ZH)**: 分布鲁棒多智能体 reinforcement 学习在动态料斗制图中的应用 

**Authors**: Guangyi Liu, Suzan Iloglu, Michael Caldara, Joseph W. Durham, Michael M. Zavlanos  

**Link**: [PDF](https://arxiv.org/pdf/2503.09755)  

**Abstract**: In Amazon robotic warehouses, the destination-to-chute mapping problem is crucial for efficient package sorting. Often, however, this problem is complicated by uncertain and dynamic package induction rates, which can lead to increased package recirculation. To tackle this challenge, we introduce a Distributionally Robust Multi-Agent Reinforcement Learning (DRMARL) framework that learns a destination-to-chute mapping policy that is resilient to adversarial variations in induction rates. Specifically, DRMARL relies on group distributionally robust optimization (DRO) to learn a policy that performs well not only on average but also on each individual subpopulation of induction rates within the group that capture, for example, different seasonality or operation modes of the system. This approach is then combined with a novel contextual bandit-based predictor of the worst-case induction distribution for each state-action pair, significantly reducing the cost of exploration and thereby increasing the learning efficiency and scalability of our framework. Extensive simulations demonstrate that DRMARL achieves robust chute mapping in the presence of varying induction distributions, reducing package recirculation by an average of 80\% in the simulation scenario. 

**Abstract (ZH)**: 亚马逊机器人仓库中，目的地到滑槽映射问题对于高效的包裹分拣至关重要。然而，由于包裹引入率的不确定性和动态性，这一问题往往会变得复杂，可能导致包裹再循环的增加。为应对这一挑战，我们引入了一个分布鲁棒多智能体强化学习（DRMARL）框架，学习一个对敌对变化的引入率具有鲁棒性的目的地到滑槽映射策略。具体而言，DRMARL 利用群体分布鲁棒优化（DRO）来学习一个不仅在平均意义上表现良好，而且在群体中的每个个体子人群中也能表现良好的策略，这些子人群捕捉到了例如系统不同的季节性和操作模式等特征。随后，该方法与一种新颖的上下文臂拉艺板预测器相结合，用于预测每个状态-动作对的最坏情况引入率分布，显著减少了探索成本，从而提高了框架的学习效率和可扩展性。广泛的仿真实验表明，DRMARL 能在引入率分布变化的情况下实现鲁棒的滑槽映射，在仿真场景中平均减少了 80% 的包裹再循环。 

---
# RILe: Reinforced Imitation Learning 

**Title (ZH)**: RILe: 强化模仿学习 

**Authors**: Mert Albaba, Sammy Christen, Thomas Langarek, Christoph Gebhardt, Otmar Hilliges, Michael J. Black  

**Link**: [PDF](https://arxiv.org/pdf/2406.08472)  

**Abstract**: Acquiring complex behaviors is essential for artificially intelligent agents, yet learning these behaviors in high-dimensional settings poses a significant challenge due to the vast search space. Traditional reinforcement learning (RL) requires extensive manual effort for reward function engineering. Inverse reinforcement learning (IRL) uncovers reward functions from expert demonstrations but relies on an iterative process that is often computationally expensive. Imitation learning (IL) provides a more efficient alternative by directly comparing an agent's actions to expert demonstrations; however, in high-dimensional environments, such direct comparisons offer insufficient feedback for effective learning. We introduce RILe (Reinforced Imitation Learning), a framework that combines the strengths of imitation learning and inverse reinforcement learning to learn a dense reward function efficiently and achieve strong performance in high-dimensional tasks. RILe employs a novel trainer-student framework: the trainer learns an adaptive reward function, and the student uses this reward signal to imitate expert behaviors. By dynamically adjusting its guidance as the student evolves, the trainer provides nuanced feedback across different phases of learning. Our framework produces high-performing policies in high-dimensional tasks where direct imitation fails to replicate complex behaviors. We validate RILe in challenging robotic locomotion tasks, demonstrating that it significantly outperforms existing methods and achieves near-expert performance across multiple settings. 

**Abstract (ZH)**: 强化模仿学习：结合逆强化学习与模仿学习高效学习高维度复杂行为 

---
# Building Cooperative Embodied Agents Modularly with Large Language Models 

**Title (ZH)**: 模块化构建具有大型语言模型的协同体态代理 

**Authors**: Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2307.02485)  

**Abstract**: In this work, we address challenging multi-agent cooperation problems with decentralized control, raw sensory observations, costly communication, and multi-objective tasks instantiated in various embodied environments. While previous research either presupposes a cost-free communication channel or relies on a centralized controller with shared observations, we harness the commonsense knowledge, reasoning ability, language comprehension, and text generation prowess of LLMs and seamlessly incorporate them into a cognitive-inspired modular framework that integrates with perception, memory, and execution. Thus building a Cooperative Embodied Language Agent CoELA, who can plan, communicate, and cooperate with others to accomplish long-horizon tasks efficiently. Our experiments on C-WAH and TDW-MAT demonstrate that CoELA driven by GPT-4 can surpass strong planning-based methods and exhibit emergent effective communication. Though current Open LMs like LLAMA-2 still underperform, we fine-tune a CoELA with data collected with our agents and show how they can achieve promising performance. We also conducted a user study for human-agent interaction and discovered that CoELA communicating in natural language can earn more trust and cooperate more effectively with humans. Our research underscores the potential of LLMs for future research in multi-agent cooperation. Videos can be found on the project website this https URL. 

**Abstract (ZH)**: 在本工作中，我们解决了具有去中心化控制、原始感官观测、成本高昂的通信和多目标任务的挑战性多智能体合作问题，并将这些任务实例化于各种具身环境中。尽管先前的研究要么假设存在免费的通信渠道，要么依赖于共享观测信息的集中控制器，我们利用了LLM的常识知识、推理能力、语言理解和文本生成能力，并将它们无缝融入一个受知觉、记忆和执行启发的模块化框架中。因此构建了一个协作性具身语言代理CoELA，它可以规划、通信和与他人合作以高效地完成长期任务。我们的实验表明，在C-WAH和TDW-MAT上由GPT-4驱动的CoELA能超越基于规划的方法，并展现出有效的涌现性通信。尽管当前的开放型语言模型如LLAMA-2仍然表现不佳，我们通过使用我们代理收集的数据微调了CoELA，并展示了它们如何实现有希望的表现。我们还进行了一个人机交互的用户研究，并发现在自然语言中进行通信的CoELA能赢得更多信任，并与人类更有效地合作。我们的研究强调了LLM在多智能体合作未来研究中的潜力。视频可以在项目网站上找到，链接为这个 https URL。 

---
# Uncertainty in Action: Confidence Elicitation in Embodied Agents 

**Title (ZH)**: 行动中的不确定性：体态代理的信心征询 

**Authors**: Tianjiao Yu, Vedant Shah, Muntasir Wahed, Kiet A. Nguyen, Adheesh Juvekar, Tal August, Ismini Lourentzou  

**Link**: [PDF](https://arxiv.org/pdf/2503.10628)  

**Abstract**: Expressing confidence is challenging for embodied agents navigating dynamic multimodal environments, where uncertainty arises from both perception and decision-making processes. We present the first work investigating embodied confidence elicitation in open-ended multimodal environments. We introduce Elicitation Policies, which structure confidence assessment across inductive, deductive, and abductive reasoning, along with Execution Policies, which enhance confidence calibration through scenario reinterpretation, action sampling, and hypothetical reasoning. Evaluating agents in calibration and failure prediction tasks within the Minecraft environment, we show that structured reasoning approaches, such as Chain-of-Thoughts, improve confidence calibration. However, our findings also reveal persistent challenges in distinguishing uncertainty, particularly under abductive settings, underscoring the need for more sophisticated embodied confidence elicitation methods. 

**Abstract (ZH)**: 在动态多模态环境中的自主体表达信心具有挑战性：从感知和决策过程中的不确定性出发，探究开放多模态环境中的信心引出 

---
# Through the Magnifying Glass: Adaptive Perception Magnification for Hallucination-Free VLM Decoding 

**Title (ZH)**: 通过放大镜：幻觉-free VLM解码的自适应感知放大 

**Authors**: Shunqi Mao, Chaoyi Zhang, Weidong Cai  

**Link**: [PDF](https://arxiv.org/pdf/2503.10183)  

**Abstract**: Existing vision-language models (VLMs) often suffer from visual hallucination, where the generated responses contain inaccuracies that are not grounded in the visual input. Efforts to address this issue without model finetuning primarily mitigate hallucination by reducing biases contrastively or amplifying the weights of visual embedding during decoding. However, these approaches improve visual perception at the cost of impairing the language reasoning capability. In this work, we propose the Perception Magnifier (PM), a novel visual decoding method that iteratively isolates relevant visual tokens based on attention and magnifies the corresponding regions, spurring the model to concentrate on fine-grained visual details during decoding. Specifically, by magnifying critical regions while preserving the structural and contextual information at each decoding step, PM allows the VLM to enhance its scrutiny of the visual input, hence producing more accurate and faithful responses. Extensive experimental results demonstrate that PM not only achieves superior hallucination mitigation but also enhances language generation while preserving strong reasoning this http URL is available at this https URL . 

**Abstract (ZH)**: 现有的视觉-语言模型（VLMs）经常会出现视觉幻觉问题，生成的响应中包含与视觉输入不符的不准确信息。无需模型微调的努力主要通过对比性地减轻偏差或在解码过程中放大视觉嵌入的权重来减轻幻觉，但这些方法在提高视觉感知能力的同时会损害语言推理能力。在本文中，我们提出了感知放大器（PM），这是一种新颖的视觉解码方法，该方法通过attention迭代地隔离相关视觉令牌并放大相应的区域，促使模型在解码过程中更加关注细粒度的视觉细节。具体而言，通过在每一步解码中放大关键区域并保持结构和上下文信息，PM使VLM能够增强对视觉输入的审视，从而产生更准确和忠实的响应。广泛实验证明，PM不仅实现了更优秀的幻觉减轻效果，还增强了语言生成能力，同时保持了强大的推理能力。该研究结果详见<cite>此处提供链接</cite>。 

---
# Temporal Difference Flows 

**Title (ZH)**: 时差流动 

**Authors**: Jesse Farebrother, Matteo Pirotta, Andrea Tirinzoni, Rémi Munos, Alessandro Lazaric, Ahmed Touati  

**Link**: [PDF](https://arxiv.org/pdf/2503.09817)  

**Abstract**: Predictive models of the future are fundamental for an agent's ability to reason and plan. A common strategy learns a world model and unrolls it step-by-step at inference, where small errors can rapidly compound. Geometric Horizon Models (GHMs) offer a compelling alternative by directly making predictions of future states, avoiding cumulative inference errors. While GHMs can be conveniently learned by a generative analog to temporal difference (TD) learning, existing methods are negatively affected by bootstrapping predictions at train time and struggle to generate high-quality predictions at long horizons. This paper introduces Temporal Difference Flows (TD-Flow), which leverages the structure of a novel Bellman equation on probability paths alongside flow-matching techniques to learn accurate GHMs at over 5x the horizon length of prior methods. Theoretically, we establish a new convergence result and primarily attribute TD-Flow's efficacy to reduced gradient variance during training. We further show that similar arguments can be extended to diffusion-based methods. Empirically, we validate TD-Flow across a diverse set of domains on both generative metrics and downstream tasks including policy evaluation. Moreover, integrating TD-Flow with recent behavior foundation models for planning over pre-trained policies demonstrates substantial performance gains, underscoring its promise for long-horizon decision-making. 

**Abstract (ZH)**: 未来预测模型是智能体推理和规划能力的基础。几何地平线模型（GHMs）通过直接预测未来状态，避免累积推理错误，提供了一种有吸引力的替代方案。本文引入了时空差分流（TD-Flow），它利用新的概率路径贝尔曼方程结构及流动匹配技术，在超过之前方法5倍的预测长度上学习精确的GHMs。理论上，我们建立了新的收敛结果，主要归因于TD-Flow在训练中减少的梯度方差。此外，我们展示了类似的论点可以扩展到基于扩散的方法。实验上，我们在生成指标和下游任务（包括策略评估）的多种领域验证了TD-Flow的有效性。进一步地，将TD-Flow与最近的行为基础模型结合用于预训练策略的规划展示了显著性能提升，突显了其在长时决策中的潜力。 

---
