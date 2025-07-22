# Look, Focus, Act: Efficient and Robust Robot Learning via Human Gaze and Foveated Vision Transformers 

**Title (ZH)**: 看、聚焦、行动：通过人类凝视及眼动视野 transformer 的高效稳健机器人学习 

**Authors**: Ian Chuang, Andrew Lee, Dechen Gao, Jinyu Zou, Iman Soltani  

**Link**: [PDF](https://arxiv.org/pdf/2507.15833)  

**Abstract**: Human vision is a highly active process driven by gaze, which directs attention and fixation to task-relevant regions and dramatically reduces visual processing. In contrast, robot learning systems typically rely on passive, uniform processing of raw camera images. In this work, we explore how incorporating human-like active gaze into robotic policies can enhance both efficiency and performance. We build on recent advances in foveated image processing and apply them to an Active Vision robot system that emulates both human head movement and eye tracking. Extending prior work on the AV-ALOHA robot simulation platform, we introduce a framework for simultaneously collecting eye-tracking data and robot demonstrations from a human operator as well as a simulation benchmark and dataset for training robot policies that incorporate human gaze. Given the widespread use of Vision Transformers (ViTs) in robot learning, we integrate gaze information into ViTs using a foveated patch tokenization scheme inspired by recent work in image segmentation. Compared to uniform patch tokenization, this significantly reduces the number of tokens-and thus computation-without sacrificing visual fidelity near regions of interest. We also explore two approaches to gaze imitation and prediction from human data. The first is a two-stage model that predicts gaze to guide foveation and action; the second integrates gaze into the action space, allowing the policy to jointly predict gaze and actions end-to-end. Our results show that our method for foveated robot vision not only drastically reduces computational overhead, but also improves performance for high precision tasks and robustness to unseen distractors. Together, these findings suggest that human-inspired visual processing offers a useful inductive bias for robotic vision systems. this https URL 

**Abstract (ZH)**: 人类视觉是一个由注视驱动的高度活跃的过程，它将注意力和注视引导至与任务相关的区域，从而大幅减少视觉处理。相比之下，机器人学习系统通常依赖于对原始摄像头图像的被动、均匀处理。在这项工作中，我们探索如何将类似人类的主动注视融入机器人策略中，以同时提升效率和性能。我们基于近期在部分感兴趣区域图像处理方面的进展，将其应用于一个模仿人类头部运动和眼动追踪的Active Vision机器人系统。在此前对AV-ALOHA机器人模拟平台工作的基础上，我们提出了一种框架，该框架同时收集人类操作员的眼动追踪数据和机器人演示，并提供了一个用于训练包含人类注视信息的机器人策略的模拟基准和数据集。鉴于Vision Transformers (ViTs) 在机器人学习中的广泛应用，我们通过借鉴图像细分领域的最新成果，将眼动信息整合到ViTs中，使用一种部分感兴趣区域片段化方案。与均匀片段化方案相比，这在减少片段数量——从而减少计算量的同时，也能够维持接近感兴趣区域的视觉保真度。我们还探索了两种从人类数据中模仿和预测眼动的方法。第一种是两级模型，用于预测眼动以引导视觉聚焦和动作；第二种方法将眼动整合到动作空间中，使得策略能够端到端地同时预测眼动和动作。我们的结果表明，我们的部分感兴趣区域机器人视觉方法不仅大幅减少了计算开销，还提高了高精度任务的性能，并增强了对未见过的干扰物的鲁棒性。这些发现表明，受人类启发的视觉处理方式为机器人视觉系统提供了有用的归纳偏置。 

---
# Interleaved LLM and Motion Planning for Generalized Multi-Object Collection in Large Scene Graphs 

**Title (ZH)**: 交错的大语言模型和运动规划在大型场景图中的通用多对象收集 

**Authors**: Ruochu Yang, Yu Zhou, Fumin Zhang, Mengxue Hou  

**Link**: [PDF](https://arxiv.org/pdf/2507.15782)  

**Abstract**: Household robots have been a longstanding research topic, but they still lack human-like intelligence, particularly in manipulating open-set objects and navigating large environments efficiently and accurately. To push this boundary, we consider a generalized multi-object collection problem in large scene graphs, where the robot needs to pick up and place multiple objects across multiple locations in a long mission of multiple human commands. This problem is extremely challenging since it requires long-horizon planning in a vast action-state space under high uncertainties. To this end, we propose a novel interleaved LLM and motion planning algorithm Inter-LLM. By designing a multimodal action cost similarity function, our algorithm can both reflect the history and look into the future to optimize plans, striking a good balance of quality and efficiency. Simulation experiments demonstrate that compared with latest works, our algorithm improves the overall mission performance by 30% in terms of fulfilling human commands, maximizing mission success rates, and minimizing mission costs. 

**Abstract (ZH)**: 家用机器人是一个长期的研究课题，但它们在处理开放集合的物体和在大型环境中高效准确导航方面仍缺乏类似人类的智能。为突破这一限制，我们考虑在大型场景图中的一般化多对象收集问题，其中机器人需要在多次人类指令的长时间任务中，在多个位置拾取和放置多个物体。由于在高度不确定性下的大型动作-状态空间中需要进行长时规划，这一问题非常具有挑战性。为此，我们提出了一种新颖的交替的LLM和运动规划算法Inter-LLM。通过设计多模式动作成本相似函数，我们的算法既能够反映历史，又能展望未来，优化规划，兼顾质量和效率。模拟实验表明，与最新研究相比，该算法在执行人类指令、最大化任务成功率和最小化任务成本方面，总体任务性能提高30%。 

---
# Gaze-supported Large Language Model Framework for Bi-directional Human-Robot Interaction 

**Title (ZH)**: 支持凝视的双向人机交互大型语言模型框架 

**Authors**: Jens V. Rüppel, Andrey Rudenko, Tim Schreiter, Martin Magnusson, Achim J. Lilienthal  

**Link**: [PDF](https://arxiv.org/pdf/2507.15729)  

**Abstract**: The rapid development of Large Language Models (LLMs) creates an exciting potential for flexible, general knowledge-driven Human-Robot Interaction (HRI) systems for assistive robots. Existing HRI systems demonstrate great progress in interpreting and following user instructions, action generation, and robot task solving. On the other hand, bi-directional, multi-modal, and context-aware support of the user in collaborative tasks still remains an open challenge. In this paper, we present a gaze- and speech-informed interface to the assistive robot, which is able to perceive the working environment from multiple vision inputs and support the dynamic user in their tasks. Our system is designed to be modular and transferable to adapt to diverse tasks and robots, and it is capable of real-time use of language-based interaction state representation and fast on board perception modules. Its development was supported by multiple public dissemination events, contributing important considerations for improved robustness and user experience. Furthermore, in two lab studies, we compare the performance and user ratings of our system with those of a traditional scripted HRI pipeline. Our findings indicate that an LLM-based approach enhances adaptability and marginally improves user engagement and task execution metrics but may produce redundant output, while a scripted pipeline is well suited for more straightforward tasks. 

**Abstract (ZH)**: 大型语言模型的迅猛发展为辅助机器人的人机交互（HRI）系统带来了灵活的知识驱动潜力。现有的HRI系统在解释和遵循用户指令、动作生成以及机器人任务解决方面取得了显著进展。另一方面，协作任务中双向、多模态和情境感知的用户支持仍是一个开放性挑战。在本文中，我们提出了一种基于凝视和言语的接口，该接口能够从多路视觉输入中感知工作环境并支持动态用户的任务。我们的系统模块化设计，能够适应多种任务和机器人，并能够实时使用基于语言的交互状态表示和快速在板感知模块。该系统的开发得到了多项公共传播活动的支持，为提高鲁棒性和用户体验提供了重要考虑。此外，在两个实验室研究中，我们将我们的系统性能和用户评级与传统脚本化HRI管道进行了比较。研究结果表明，基于LLM的方法提高了适应性，并且略微提升了用户参与度和任务执行指标，但可能会产生冗余输出；而脚本化管道则更适合简单任务。 

---
# DiffPF: Differentiable Particle Filtering with Generative Sampling via Conditional Diffusion Models 

**Title (ZH)**: DiffPF：基于条件扩散模型的可微分粒子滤波与生成采样 

**Authors**: Ziyu Wan, Lin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.15716)  

**Abstract**: This paper proposes DiffPF, a differentiable particle filter that leverages diffusion models for state estimation in dynamic systems. Unlike conventional differentiable particle filters, which require importance weighting and typically rely on predefined or low-capacity proposal distributions. DiffPF learns a flexible posterior sampler by conditioning a diffusion model on predicted particles and the current observation. This enables accurate, equally-weighted sampling from complex, high-dimensional, and multimodal filtering distributions. We evaluate DiffPF across a range of scenarios, including both unimodal and highly multimodal distributions, and test it on simulated as well as real-world tasks, where it consistently outperforms existing filtering baselines. In particular, DiffPF achieves an 82.8% improvement in estimation accuracy on a highly multimodal global localization benchmark, and a 26% improvement on the real-world KITTI visual odometry benchmark, compared to state-of-the-art differentiable filters. To the best of our knowledge, DiffPF is the first method to integrate conditional diffusion models into particle filtering, enabling high-quality posterior sampling that produces more informative particles and significantly improves state estimation. 

**Abstract (ZH)**: DiffPF：一种用于动态系统状态估计的可微分粒子滤波器 

---
# Selective Densification for Rapid Motion Planning in High Dimensions with Narrow Passages 

**Title (ZH)**: 高维狭窄通道快速运动规划的选择性致密化 

**Authors**: Lu Huang, Lingxiao Meng, Jiankun Wang, Xingjian Jing  

**Link**: [PDF](https://arxiv.org/pdf/2507.15710)  

**Abstract**: Sampling-based algorithms are widely used for motion planning in high-dimensional configuration spaces. However, due to low sampling efficiency, their performance often diminishes in complex configuration spaces with narrow corridors. Existing approaches address this issue using handcrafted or learned heuristics to guide sampling toward useful regions. Unfortunately, these strategies often lack generalizability to various problems or require extensive prior training. In this paper, we propose a simple yet efficient sampling-based planning framework along with its bidirectional version that overcomes these issues by integrating different levels of planning granularity. Our approach probes configuration spaces with uniform random samples at varying resolutions and explores these multi-resolution samples online with a bias towards sparse samples when traveling large free configuration spaces. By seamlessly transitioning between sparse and dense samples, our approach can navigate complex configuration spaces while maintaining planning speed and completeness. The simulation results demonstrate that our approach outperforms several state-of-the-art sampling-based planners in $\mathbb{SE}(2)$, $\mathbb{SE}(3)$, and $\mathbb{R}^{14}$ with challenging terrains. Furthermore, experiments conducted with the Franka Emika Panda robot operating in a constrained workspace provide additional evidence of the superiority of the proposed method. 

**Abstract (ZH)**: 基于采样的运动规划算法广泛应用于高维配置空间。然而，由于采样效率低，这些算法在具有狭窄通道的复杂配置空间中的性能往往会下降。现有的方法通过手工设计或学习的启发式方法来指导采样向有用区域进行，但这些策略往往缺乏普适性或需要大量的先验训练。本文提出了一种简单而高效的基于采样的规划框架及其双向版本，通过集成不同层次的规划粒度来克服这些问题。该方法使用不同分辨率的均匀随机采样探测配置空间，并在线探索这些多分辨率样本时倾向于稀疏样本，特别是在穿越大量自由配置空间时。通过无缝地转换稀疏和密集样本，该方法可以在保持规划速度和完备性的同时导航复杂的配置空间。仿真结果表明，该方法在$\mathbb{SE}(2)$、$\mathbb{SE}(3)$和$\mathbb{R}^{14}$及其具有挑战性地形的环境中，优于几种最先进的基于采样的规划算法。此外，使用Franka Emika Panda机器人在受限工作空间中的实验进一步证明了所提出方法的优越性。 

---
# Strong, Accurate, and Low-Cost Robot Manipulator 

**Title (ZH)**: 强力、精确且低成本的机器人 manipulator 

**Authors**: Georges Chebly, Spencer Little, Nisal Perera, Aliya Abedeen, Ken Suzuki, Donghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.15693)  

**Abstract**: This paper presents Forte, a fully 3D-printable, 6-DoF robotic arm designed to achieve near industrial-grade performance - 0.63 kg payload, 0.467 m reach, and sub-millimeter repeatability - at a material cost under $215. As an accessible robot for broad applications across classroom education to AI experiments, Forte pushes forward the performance limitations of existing low-cost educational arms. We introduce a cost-effective mechanical design that combines capstan-based cable drives, timing belts, simple tensioning mechanisms, and lightweight 3D-printed structures, along with topology optimization for structural stiffness. Through careful drivetrain engineering, we minimize backlash and maintain control fidelity without relying on high-power electronics or expensive manufacturing processes. Experimental validation demonstrates that Forte achieves high repeatability and load capacity, offering a compelling robotic platform for both classroom instruction and advanced robotics research. 

**Abstract (ZH)**: Forte：一款成本低于215美元、可全3D打印、6轴自由度的机器人手臂，实现接近工业级性能 

---
# Data-Driven MPC with Data Selection for Flexible Cable-Driven Robotic Arms 

**Title (ZH)**: 基于数据选择的数据驱动 MPC 灵活缆驱机器人臂 

**Authors**: Huayue Liang, Yanbo Chen, Hongyang Cheng, Yanzhao Yu, Shoujie Li, Junbo Tan, Xueqian Wang, Long Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.15677)  

**Abstract**: Flexible cable-driven robotic arms (FCRAs) offer dexterous and compliant motion. Still, the inherent properties of cables, such as resilience, hysteresis, and friction, often lead to particular difficulties in modeling and control. This paper proposes a model predictive control (MPC) method that relies exclusively on input-output data, without a physical model, to improve the control accuracy of FCRAs. First, we develop an implicit model based on input-output data and integrate it into an MPC optimization framework. Second, a data selection algorithm (DSA) is introduced to filter the data that best characterize the system, thereby reducing the solution time per step to approximately 4 ms, which is an improvement of nearly 80%. Lastly, the influence of hyperparameters on tracking error is investigated through simulation. The proposed method has been validated on a real FCRA platform, including five-point positioning accuracy tests, a five-point response tracking test, and trajectory tracking for letter drawing. The results demonstrate that the average positioning accuracy is approximately 2.070 mm. Moreover, compared to the PID method with an average tracking error of 1.418°, the proposed method achieves an average tracking error of 0.541°. 

**Abstract (ZH)**: 柔性缆索驱动机器人手臂的模型预测控制方法：基于输入输出数据的无需物理模型方法 

---
# EMP: Executable Motion Prior for Humanoid Robot Standing Upper-body Motion Imitation 

**Title (ZH)**: EMP：可执行运动先验对人体形机器人站立上身运动模仿 

**Authors**: Haocheng Xu, Haodong Zhang, Zhenghan Chen, Rong Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.15649)  

**Abstract**: To support humanoid robots in performing manipulation tasks, it is essential to study stable standing while accommodating upper-body motions. However, the limited controllable range of humanoid robots in a standing position affects the stability of the entire body. Thus we introduce a reinforcement learning based framework for humanoid robots to imitate human upper-body motions while maintaining overall stability. Our approach begins with designing a retargeting network that generates a large-scale upper-body motion dataset for training the reinforcement learning (RL) policy, which enables the humanoid robot to track upper-body motion targets, employing domain randomization for enhanced robustness. To avoid exceeding the robot's execution capability and ensure safety and stability, we propose an Executable Motion Prior (EMP) module, which adjusts the input target movements based on the robot's current state. This adjustment improves standing stability while minimizing changes to motion amplitude. We evaluate our framework through simulation and real-world tests, demonstrating its practical applicability. 

**Abstract (ZH)**: 基于强化学习的人形机器人模仿上身运动的稳定站立框架 

---
# Optimizing Force Signals from Human Demonstrations of In-Contact Motions 

**Title (ZH)**: 优化人演示接触运动中的力信号 

**Authors**: Johannes Hartwig, Fabian Viessmann, Dominik Henrich  

**Link**: [PDF](https://arxiv.org/pdf/2507.15608)  

**Abstract**: For non-robot-programming experts, kinesthetic guiding can be an intuitive input method, as robot programming of in-contact tasks is becoming more prominent. However, imprecise and noisy input signals from human demonstrations pose problems when reproducing motions directly or using the signal as input for machine learning methods. This paper explores optimizing force signals to correspond better to the human intention of the demonstrated signal. We compare different signal filtering methods and propose a peak detection method for dealing with first-contact deviations in the signal. The evaluation of these methods considers a specialized error criterion between the input and the human-intended signal. In addition, we analyze the critical parameters' influence on the filtering methods. The quality for an individual motion could be increased by up to \SI{20}{\percent} concerning the error criterion. The proposed contribution can improve the usability of robot programming and the interaction between humans and robots. 

**Abstract (ZH)**: 非机器人编程专家的动能引导可以作为一种直观的输入方法，随着接触任务的机器人编程变得更为突出。然而，人类示范过程中的不精确和噪声信号输入给直接重现动作或作为机器学习方法输入带来了问题。本文探讨了优化力信号，使它们更好地对应人类示范信号的意图。我们比较了不同的信号滤波方法，并提出了一种峰值检测方法来处理信号中的初始接触偏差。这些方法的评估考虑了一个针对输入信号和人类意图信号之间的特殊误差标准。此外，我们分析了滤波方法中关键参数的影响。对于单个动作，误差标准下的质量最多可以提高20%。所提出的内容可以改进机器人编程的可用性以及人与机器人之间的交互。 

---
# A Universal Vehicle-Trailer Navigation System with Neural Kinematics and Online Residual Learning 

**Title (ZH)**: 一种基于神经运动学和在线残差学习的通用车辆-挂车导航系统 

**Authors**: Yanbo Chen, Yunzhe Tan, Yaojia Wang, Zhengzhe Xu, Junbo Tan, Xueqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15607)  

**Abstract**: Autonomous navigation of vehicle-trailer systems is crucial in environments like airports, supermarkets, and concert venues, where various types of trailers are needed to navigate with different payloads and conditions. However, accurately modeling such systems remains challenging, especially for trailers with castor wheels. In this work, we propose a novel universal vehicle-trailer navigation system that integrates a hybrid nominal kinematic model--combining classical nonholonomic constraints for vehicles and neural network-based trailer kinematics--with a lightweight online residual learning module to correct real-time modeling discrepancies and disturbances. Additionally, we develop a model predictive control framework with a weighted model combination strategy that improves long-horizon prediction accuracy and ensures safer motion planning. Our approach is validated through extensive real-world experiments involving multiple trailer types and varying payload conditions, demonstrating robust performance without manual tuning or trailer-specific calibration. 

**Abstract (ZH)**: 自主导航的车辆-拖车系统在机场、超市和音乐会场等环境中至关重要，需要根据不同载荷和条件导航各种类型的拖车。然而，准确建模此类系统仍然颇具挑战性，尤其对于配备定向轮的拖车。本文提出了一种新型通用车辆-拖车导航系统，该系统结合了混合名义运动模型——经典非完整约束与基于神经网络的拖车运动学相结合，并 integrated 轻量级在线残差学习模块以实时修正建模偏差和干扰。此外，我们还开发了一种带加权模型组合策略的模型预测控制框架，以提高长期预测准确性并确保更安全的运动规划。我们的方法通过涉及多种拖车类型和不同载荷条件的广泛现实世界实验得到了验证，展示了无需手动调谐或特定拖车校准的稳健性能。 

---
# Estimation of Payload Inertial Parameters from Human Demonstrations by Hand Guiding 

**Title (ZH)**: 基于手动引导的人体示范用于载荷惯性参数估计 

**Authors**: Johannes Hartwig, Philipp Lienhardt, Dominik Henrich  

**Link**: [PDF](https://arxiv.org/pdf/2507.15604)  

**Abstract**: As the availability of cobots increases, it is essential to address the needs of users with little to no programming knowledge to operate such systems efficiently. Programming concepts often use intuitive interaction modalities, such as hand guiding, to address this. When programming in-contact motions, such frameworks require knowledge of the robot tool's payload inertial parameters (PIP) in addition to the demonstrated velocities and forces to ensure effective hybrid motion-force control. This paper aims to enable non-expert users to program in-contact motions more efficiently by eliminating the need for a dedicated PIP calibration, thereby enabling flexible robot tool changes. Since demonstrated tasks generally also contain motions with non-contact, our approach uses these parts to estimate the robot's PIP using established estimation techniques. The results show that the estimation of the payload's mass is accurate, whereas the center of mass and the inertia tensor are affected by noise and a lack of excitation. Overall, these findings show the feasibility of PIP estimation during hand guiding but also highlight the need for sufficient payload accelerations for an accurate estimation. 

**Abstract (ZH)**: 随着协作机器人的可用性增加，有必要解决那些缺乏或几乎没有编程知识的用户高效操作此类系统的需要。此类框架通常使用直观的交互模态，如手动引导，来解决这一问题。在编程接触运动时，除了演示的速度和力外，还需知道机器人工具的有效载荷惯性参数（PIP），以确保有效的混合运动-力控制。本文旨在通过消除专用PIP校准的需要，使非专家用户更高效地编程接触运动，从而实现灵活的机器人工具更换。由于演示的任务通常也包含非接触运动，本方法利用这些部分采用现有的估计技术估计机器人的PIP。结果表明，有效载荷质量的估计是准确的，而质心和惯性张量则受到噪声和激励不足的影响。总体而言，这些发现表明，在手动引导过程中进行PIP估计是可行的，但也表明需要足够的有效载荷加速度才能获得准确的估计。 

---
# CLEVER: Stream-based Active Learning for Robust Semantic Perception from Human Instructions 

**Title (ZH)**: CLEVER：基于流的数据主动学习以从人类指令中实现鲁棒语义感知 

**Authors**: Jongseok Lee, Timo Birr, Rudolph Triebel, Tamim Asfour  

**Link**: [PDF](https://arxiv.org/pdf/2507.15499)  

**Abstract**: We propose CLEVER, an active learning system for robust semantic perception with Deep Neural Networks (DNNs). For data arriving in streams, our system seeks human support when encountering failures and adapts DNNs online based on human instructions. In this way, CLEVER can eventually accomplish the given semantic perception tasks. Our main contribution is the design of a system that meets several desiderata of realizing the aforementioned capabilities. The key enabler herein is our Bayesian formulation that encodes domain knowledge through priors. Empirically, we not only motivate CLEVER's design but further demonstrate its capabilities with a user validation study as well as experiments on humanoid and deformable objects. To our knowledge, we are the first to realize stream-based active learning on a real robot, providing evidence that the robustness of the DNN-based semantic perception can be improved in practice. The project website can be accessed at this https URL. 

**Abstract (ZH)**: 我们提出了一种基于深度神经网络的鲁棒语义感知的主动学习系统CLEVER 

---
# GR-3 Technical Report 

**Title (ZH)**: GR-3 技术报告 

**Authors**: Chilam Cheang, Sijin Chen, Zhongren Cui, Yingdong Hu, Liqun Huang, Tao Kong, Hang Li, Yifeng Li, Yuxiao Liu, Xiao Ma, Hao Niu, Wenxuan Ou, Wanli Peng, Zeyu Ren, Haixin Shi, Jiawen Tian, Hongtao Wu, Xin Xiao, Yuyang Xiao, Jiafeng Xu, Yichu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15493)  

**Abstract**: We report our recent progress towards building generalist robot policies, the development of GR-3. GR-3 is a large-scale vision-language-action (VLA) model. It showcases exceptional capabilities in generalizing to novel objects, environments, and instructions involving abstract concepts. Furthermore, it can be efficiently fine-tuned with minimal human trajectory data, enabling rapid and cost-effective adaptation to new settings. GR-3 also excels in handling long-horizon and dexterous tasks, including those requiring bi-manual manipulation and mobile movement, showcasing robust and reliable performance. These capabilities are achieved through a multi-faceted training recipe that includes co-training with web-scale vision-language data, efficient fine-tuning from human trajectory data collected via VR devices, and effective imitation learning with robot trajectory data. In addition, we introduce ByteMini, a versatile bi-manual mobile robot designed with exceptional flexibility and reliability, capable of accomplishing a wide range of tasks when integrated with GR-3. Through extensive real-world experiments, we show GR-3 surpasses the state-of-the-art baseline method, $\pi_0$, on a wide variety of challenging tasks. We hope GR-3 can serve as a step towards building generalist robots capable of assisting humans in daily life. 

**Abstract (ZH)**: 我们报告了构建通用机器人策略的最新进展，以及GR-3的发展。GR-3是一款大规模的视觉-语言-行动（VLA）模型，展示了在新物体、新环境和涉及抽象概念的新指令方面的出色泛化能力。此外，它可以通过少量的人类轨迹数据高效微调，从而实现快速且成本效益高的新环境适应。GR-3在处理长期任务和灵巧任务方面也表现出色，包括双臂操作和移动任务，展示了稳健可靠的性能。这些能力是通过多方面的训练食谱实现的，包括与大规模网络视觉-语言数据的协同训练、通过VR设备收集的人类轨迹数据的高效微调，以及利用机器人轨迹数据的有效的模仿学习。此外，我们介绍了ByteMini，这是一种多功能的双臂移动机器人，设计灵活可靠，与GR-3集成后能够完成多种任务。通过广泛的实地实验，我们展示了GR-3在多种具有挑战性的任务中超过了最先进的基线方法$\pi_0$。我们希望GR-3可以成为构建能够在日常生活中辅助人类的通用机器人的一个步骤。 

---
# Robots for Kiwifruit Harvesting and Pollination 

**Title (ZH)**: 猕研果收获与授粉机器人 

**Authors**: Jamie Bell  

**Link**: [PDF](https://arxiv.org/pdf/2507.15484)  

**Abstract**: This research was a part of a project that developed mobile robots that performed targeted pollen spraying and automated harvesting in pergola structured kiwifruit orchards. Multiple kiwifruit detachment mechanisms were designed and field testing of one of the concepts showed that the mechanism could reliably pick kiwifruit. Furthermore, this kiwifruit detachment mechanism was able to reach over 80 percent of fruit in the cluttered kiwifruit canopy, whereas the previous state of the art mechanism was only able to reach less than 70 percent of the fruit. Artificial pollination was performed by detecting flowers and then spraying pollen in solution onto the detected flowers from a line of sprayers on a boom, while driving at up to 1.4 ms-1. In addition, the height of the canopy was measured and the spray boom was moved up and down to keep the boom close enough to the flowers for the spray to reach the flowers, while minimising collisions with the canopy. Mobile robot navigation was performed using a 2D lidar in apple orchards and vineyards. Lidar navigation in kiwifruit orchards was more challenging because the pergola structure only provides a small amount of data for the direction of rows, compared to the amount of data from the overhead canopy, the undulating ground and other objects in the orchards. Multiple methods are presented here for extracting structure defining features from 3D lidar data in kiwifruit orchards. In addition, a 3D lidar navigation system -- which performed row following, row end detection and row end turns -- was tested for over 30 km of autonomous driving in kiwifruit orchards. Computer vision algorithms for row detection and row following were also tested. The computer vision algorithm worked as well as the 3D lidar row following method in testing. 

**Abstract (ZH)**: 一种用于 pergola 结构猕猴桃园目标花粉喷洒和自动化收获的移动机器人研究 

---
# The Constitutional Controller: Doubt-Calibrated Steering of Compliant Agents 

**Title (ZH)**: 宪法控制器：疑虑校准的合规代理引导 

**Authors**: Simon Kohaut, Felix Divo, Navid Hamid, Benedict Flade, Julian Eggert, Devendra Singh Dhami, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2507.15478)  

**Abstract**: Ensuring reliable and rule-compliant behavior of autonomous agents in uncertain environments remains a fundamental challenge in modern robotics. Our work shows how neuro-symbolic systems, which integrate probabilistic, symbolic white-box reasoning models with deep learning methods, offer a powerful solution to this challenge. This enables the simultaneous consideration of explicit rules and neural models trained on noisy data, combining the strength of structured reasoning with flexible representations. To this end, we introduce the Constitutional Controller (CoCo), a novel framework designed to enhance the safety and reliability of agents by reasoning over deep probabilistic logic programs representing constraints such as those found in shared traffic spaces. Furthermore, we propose the concept of self-doubt, implemented as a probability density conditioned on doubt features such as travel velocity, employed sensors, or health factors. In a real-world aerial mobility study, we demonstrate CoCo's advantages for intelligent autonomous systems to learn appropriate doubts and navigate complex and uncertain environments safely and compliantly. 

**Abstract (ZH)**: 确保自主代理在不确定环境中的可靠和合规行为仍然是现代机器人技术中的一个基本挑战。我们的工作展示了将概率主义和符号白盒推理模型与深度学习方法相结合的神经符号系统如何提供解决这一挑战的强大方案。这使得同时考虑显式规则和基于噪声数据训练的神经模型成为可能，结合了结构化推理的强度与灵活表示的灵活性。为此，我们引入了宪法控制器（CoCo）这一新型框架，通过推理包含约束（例如共享交通空间中的约束）的深层概率逻辑程序来增强代理的安全性和可靠性。此外，我们提出了自我怀疑的概念，将其实现为基于怀疑特征（如旅行速度、使用的传感器或健康状况等）的概率密度。在实际的空中移动研究中，我们展示了CoCo如何帮助智能自主系统学习合适的怀疑并安全、合规地导航复杂和不确定的环境。 

---
# All-UWB SLAM Using UWB Radar and UWB AOA 

**Title (ZH)**: 全频段UWB SLAM利用UWB雷达和UWBAOA 

**Authors**: Charith Premachandra, Achala Athukorala, U-Xuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2507.15474)  

**Abstract**: There has been a growing interest in autonomous systems designed to operate in adverse conditions (e.g. smoke, dust), where the visible light spectrum fails. In this context, Ultra-wideband (UWB) radar is capable of penetrating through such challenging environmental conditions due to the lower frequency components within its broad bandwidth. Therefore, UWB radar has emerged as a potential sensing technology for Simultaneous Localization and Mapping (SLAM) in vision-denied environments where optical sensors (e.g. LiDAR, Camera) are prone to failure. Existing approaches involving UWB radar as the primary exteroceptive sensor generally extract features in the environment, which are later initialized as landmarks in a map. However, these methods are constrained by the number of distinguishable features in the environment. Hence, this paper proposes a novel method incorporating UWB Angle of Arrival (AOA) measurements into UWB radar-based SLAM systems to improve the accuracy and scalability of SLAM in feature-deficient environments. The AOA measurements are obtained using UWB anchor-tag units which are dynamically deployed by the robot in featureless areas during mapping of the environment. This paper thoroughly discusses prevailing constraints associated with UWB AOA measurement units and presents solutions to overcome them. Our experimental results show that integrating UWB AOA units with UWB radar enables SLAM in vision-denied feature-deficient environments. 

**Abstract (ZH)**: 超宽带雷达在视觉受限特征贫乏环境中的同时定位与地图构建 

---
# The Emergence of Deep Reinforcement Learning for Path Planning 

**Title (ZH)**: 深度强化学习在路径规划中的 emergence 

**Authors**: Thanh Thi Nguyen, Saeid Nahavandi, Imran Razzak, Dung Nguyen, Nhat Truong Pham, Quoc Viet Hung Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.15469)  

**Abstract**: The increasing demand for autonomous systems in complex and dynamic environments has driven significant research into intelligent path planning methodologies. For decades, graph-based search algorithms, linear programming techniques, and evolutionary computation methods have served as foundational approaches in this domain. Recently, deep reinforcement learning (DRL) has emerged as a powerful method for enabling autonomous agents to learn optimal navigation strategies through interaction with their environments. This survey provides a comprehensive overview of traditional approaches as well as the recent advancements in DRL applied to path planning tasks, focusing on autonomous vehicles, drones, and robotic platforms. Key algorithms across both conventional and learning-based paradigms are categorized, with their innovations and practical implementations highlighted. This is followed by a thorough discussion of their respective strengths and limitations in terms of computational efficiency, scalability, adaptability, and robustness. The survey concludes by identifying key open challenges and outlining promising avenues for future research. Special attention is given to hybrid approaches that integrate DRL with classical planning techniques to leverage the benefits of both learning-based adaptability and deterministic reliability, offering promising directions for robust and resilient autonomous navigation. 

**Abstract (ZH)**: 复杂动态环境中自主系统的需求增加推动了智能路径规划方法的显著研究。多年来，基于图的搜索算法、线性规划技术和进化计算方法一直是该领域的基础方法。近年来，深度强化学习（DRL）作为一种强大方法，通过与环境的交互使自主代理学习最优导航策略。本文综述了传统方法以及应用于路径规划任务的最近DRL进展，重点关注自主车辆、无人机和机器人平台。文章对传统和学习导向范式下的关键算法进行了分类，并强调了它们的创新和实际应用。随后，文章详细讨论了这些方法在计算效率、可扩展性、适应性和稳健性方面的各自优势和局限性。这篇综述总结了关键的开放挑战，并指出了未来研究的有希望的方向。特别注意结合DRL与经典规划技术的混合方法，以利用基于学习的适应性和确定性可靠性所带来的优势，为稳健和鲁棒的自主导航提供了有希望的方向。 

---
# Low-Latency Event-Based Velocimetry for Quadrotor Control in a Narrow Pipe 

**Title (ZH)**: 低延迟事件驱动的速度测量在狭窄管道中四旋翼无人机控制 

**Authors**: Leonard Bauersfeld, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2507.15444)  

**Abstract**: Autonomous quadrotor flight in confined spaces such as pipes and tunnels presents significant challenges due to unsteady, self-induced aerodynamic disturbances. Very recent advances have enabled flight in such conditions, but they either rely on constant motion through the pipe to mitigate airflow recirculation effects or suffer from limited stability during hovering. In this work, we present the first closed-loop control system for quadrotors for hovering in narrow pipes that leverages real-time flow field measurements. We develop a low-latency, event-based smoke velocimetry method that estimates local airflow at high temporal resolution. This flow information is used by a disturbance estimator based on a recurrent convolutional neural network, which infers force and torque disturbances in real time. The estimated disturbances are integrated into a learning-based controller trained via reinforcement learning. The flow-feedback control proves particularly effective during lateral translation maneuvers in the pipe cross-section. There, the real-time disturbance information enables the controller to effectively counteract transient aerodynamic effects, thereby preventing collisions with the pipe wall. To the best of our knowledge, this work represents the first demonstration of an aerial robot with closed-loop control informed by real-time flow field measurements. This opens new directions for research on flight in aerodynamically complex environments. In addition, our work also sheds light on the characteristic flow structures that emerge during flight in narrow, circular pipes, providing new insights at the intersection of robotics and fluid dynamics. 

**Abstract (ZH)**: 自主四旋翼无人机在狭窄管道和隧道等受限空间内的悬停飞行面临着显著挑战，由于不稳定且自诱导的气动干扰。非常近期的进展已经使得在这些条件下飞行成为可能，但这些方法要么依赖于管道内的持续运动以减轻气流回流效应，要么悬停期间稳定性受到限制。在此项工作中，我们提出了第一个基于实时气流场测量的四旋翼无人机悬停在狭窄管道内的闭环控制系统。我们开发了一种低延迟、事件驱动的烟雾速度测量方法，以高时间分辨率估计局部气流。这些流场信息被用于一个基于循环卷积神经网络的干扰估计器中，该估计器可以实时推断出力和力矩干扰。估计得到的干扰信息被集成到一种通过强化学习训练的基于学习的控制器中。流反馈控制在管道横截面内的横向平移机动中显示出特别有效的效果。在那里，实时的干扰信息使得控制器能够有效抵消瞬时气动效应，从而避免与管道壁发生碰撞。据我们所知，这项工作是首个通过实时流场测量指导闭环控制的空中机器人演示。这开启了在气动复杂环境飞行研究的新方向。此外，我们的工作还揭示了在狭窄圆形管道内飞行时出现的典型流场结构，为机器人学与流体力学的交叉领域提供了新的见解。 

---
# RepILN: Reparameterized Inertial Localization Network 

**Title (ZH)**: 重构惯性定位网络: 参数化惯性定位网络 

**Authors**: Shanshan Zhang, Tianshui Wen, Siyue Wang, Qi Zhang, Ziheng Zhou, Lingxiang Zheng, Yu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15293)  

**Abstract**: Inertial localization is regarded as a promising positioning solution for consumer-grade IoT devices due to its cost-effectiveness and independence from external infrastructure. However, data-driven inertial localization methods often rely on increasingly complex network architectures to improve accuracy, which challenges the limited computational resources of IoT devices. Moreover, these methods frequently overlook the importance of modeling long-term dependencies in inertial measurements - a critical factor for accurate trajectory reconstruction - thereby limiting localization performance. To address these challenges, we propose a reparameterized inertial localization network that uses a multi-branch structure during training to enhance feature extraction. At inference time, this structure is transformed into an equivalent single-path architecture to improve parameter efficiency. To further capture long-term dependencies in motion trajectories, we introduce a temporal-scale sparse attention mechanism that selectively emphasizes key trajectory segments while suppressing noise. Additionally, a gated convolutional unit is incorporated to effectively integrate long-range dependencies with local fine-grained features. Extensive experiments on public benchmarks demonstrate that our method achieves a favorable trade-off between accuracy and model compactness. For example, on the RoNIN dataset, our approach reduces the Absolute Trajectory Error (ATE) by 2.59% compared to RoNIN-ResNet while reducing the number of parameters by 3.86%. 

**Abstract (ZH)**: 基于惯性定位网络：一种多分支结构与时空稀疏注意力机制相结合的方法 

---
# VLM-UDMC: VLM-Enhanced Unified Decision-Making and Motion Control for Urban Autonomous Driving 

**Title (ZH)**: VLM-UDMC：增强视觉语言模型的统一决策与运动控制城市自主驾驶 

**Authors**: Haichao Liu, Haoren Guo, Pei Liu, Benshan Ma, Yuxiang Zhang, Jun Ma, Tong Heng Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.15266)  

**Abstract**: Scene understanding and risk-aware attentions are crucial for human drivers to make safe and effective driving decisions. To imitate this cognitive ability in urban autonomous driving while ensuring the transparency and interpretability, we propose a vision-language model (VLM)-enhanced unified decision-making and motion control framework, named VLM-UDMC. This framework incorporates scene reasoning and risk-aware insights into an upper-level slow system, which dynamically reconfigures the optimal motion planning for the downstream fast system. The reconfiguration is based on real-time environmental changes, which are encoded through context-aware potential functions. More specifically, the upper-level slow system employs a two-step reasoning policy with Retrieval-Augmented Generation (RAG), leveraging foundation models to process multimodal inputs and retrieve contextual knowledge, thereby generating risk-aware insights. Meanwhile, a lightweight multi-kernel decomposed LSTM provides real-time trajectory predictions for heterogeneous traffic participants by extracting smoother trend representations for short-horizon trajectory prediction. The effectiveness of the proposed VLM-UDMC framework is verified via both simulations and real-world experiments with a full-size autonomous vehicle. It is demonstrated that the presented VLM-UDMC effectively leverages scene understanding and attention decomposition for rational driving decisions, thus improving the overall urban driving performance. Our open-source project is available at this https URL. 

**Abstract (ZH)**: 场景理解与风险意识注意力对于人类驾驶员做出安全有效的驾驶决策至关重要。为了在确保透明性和可解释性的前提下，在城市自动驾驶中模仿这一认知能力，我们提出了一种基于视觉-语言模型（VLM）增强的统一决策和运动控制框架，命名为VLM-UDMC。该框架将场景推理和风险意识洞察整合到一个高层的慢速系统中，该系统能够根据实时环境变化动态重新配置下游快速系统的最优运动规划。重新配置基于上下文感知的潜在函数编码的实时环境变化。具体而言，高层慢速系统采用基于检索增强生成（RAG）的两步推理策略，利用基础模型处理多模态输入并检索上下文知识，从而生成风险意识洞察。同时，一种轻量级多核分解LSTM通过提取更平滑的趋势表示，提供了对不同交通参与者的实时轨迹预测。通过模拟和实际路况实验，使用全尺寸自动驾驶车辆验证了所提出的VLM-UDMC框架的有效性。实验表明，VLM-UDMC有效地利用了场景理解与注意力分解，从而提高了整体城市驾驶性能。我们的开源项目可在以下链接访问。 

---
# CHADET: Cross-Hierarchical-Attention for Depth-Completion Using Unsupervised Lightweight Transformer 

**Title (ZH)**: CHADET：跨层级注意力的无监督轻量级变换器用于深度补全 

**Authors**: Kevin Christiansen Marsim, Jinwoo Jeon, Yeeun Kim, Myeongwoo Jeong, Hyun Myung  

**Link**: [PDF](https://arxiv.org/pdf/2507.15189)  

**Abstract**: Depth information which specifies the distance between objects and current position of the robot is essential for many robot tasks such as navigation. Recently, researchers have proposed depth completion frameworks to provide dense depth maps that offer comprehensive information about the surrounding environment. However, existing methods show significant trade-offs between computational efficiency and accuracy during inference. The substantial memory and computational requirements make them unsuitable for real-time applications, highlighting the need to improve the completeness and accuracy of depth information while improving processing speed to enhance robot performance in various tasks. To address these challenges, in this paper, we propose CHADET(cross-hierarchical-attention depth-completion transformer), a lightweight depth-completion network that can generate accurate dense depth maps from RGB images and sparse depth points. For each pair, its feature is extracted from the depthwise blocks and passed to the equally lightweight transformer-based decoder. In the decoder, we utilize the novel cross-hierarchical-attention module that refines the image features from the depth information. Our approach improves the quality and reduces memory usage of the depth map prediction, as validated in both KITTI, NYUv2, and VOID datasets. 

**Abstract (ZH)**: 基于交叉层次注意力的深度补全变换器（CHADET）：一种轻量级的稠密深度图生成方法 

---
# Learning-Based Modeling of a Magnetically Steerable Soft Suction Device for Endoscopic Endonasal Interventions 

**Title (ZH)**: 基于学习的可磁导航软吸引装置的建模研究：内鼻外科干预应用 

**Authors**: Majid Roshanfar, Alex Zhang, Changyan He, Amir Hooshiar, Dale J. Podolsky, Thomas Looi, Eric Diller  

**Link**: [PDF](https://arxiv.org/pdf/2507.15155)  

**Abstract**: This letter introduces a novel learning-based modeling framework for a magnetically steerable soft suction device designed for endoscopic endonasal brain tumor resection. The device is miniaturized (4 mm outer diameter, 2 mm inner diameter, 40 mm length), 3D printed using biocompatible SIL 30 material, and integrates embedded Fiber Bragg Grating (FBG) sensors for real-time shape feedback. Shape reconstruction is represented using four Bezier control points, enabling a compact and smooth model of the device's deformation. A data-driven model was trained on 5,097 experimental samples covering a range of magnetic field magnitudes (0-14 mT), actuation frequencies (0.2-1.0 Hz), and vertical tip distances (90-100 mm), using both Neural Network (NN) and Random Forest (RF) architectures. The RF model outperformed the NN across all metrics, achieving a mean root mean square error of 0.087 mm in control point prediction and a mean shape reconstruction error of 0.064 mm. Feature importance analysis further revealed that magnetic field components predominantly influence distal control points, while frequency and distance affect the base configuration. This learning-based approach effectively models the complex nonlinear behavior of hyperelastic soft robots under magnetic actuation without relying on simplified physical assumptions. By enabling sub-millimeter shape prediction accuracy and real-time inference, this work represents an advancement toward the intelligent control of magnetically actuated soft robotic tools in minimally invasive neurosurgery. 

**Abstract (ZH)**: 基于学习的磁控可弯曲软 suction 设备建模框架：用于内镜经鼻脑肿瘤切除 

---
# Search-Based Autonomous Vehicle Motion Planning Using Game Theory 

**Title (ZH)**: 基于搜索的博弈论自主车辆运动规划 

**Authors**: Pouya Panahandeh, Mohammad Pirani, Baris Fidan, Amir Khajepour  

**Link**: [PDF](https://arxiv.org/pdf/2507.15088)  

**Abstract**: In this paper, we propose a search-based interactive motion planning scheme for autonomous vehicles (AVs), using a game-theoretic approach. In contrast to traditional search-based approaches, the newly developed approach considers other road users (e.g. drivers and pedestrians) as intelligent agents rather than static obstacles. This leads to the generation of a more realistic path for the AV. Due to the low computational time, the proposed motion planning scheme is implementable in real-time applications. The performance of the developed motion planning scheme is compared with existing motion planning techniques and validated through experiments using WATonoBus, an electrical all-weather autonomous shuttle bus. 

**Abstract (ZH)**: 基于搜索的游戏论导向自主车辆交互运动规划方法 

---
# Touch in the Wild: Learning Fine-Grained Manipulation with a Portable Visuo-Tactile Gripper 

**Title (ZH)**: 触觉在野外：基于便携式视觉-触觉夹持器的精细 manipulation 学习 

**Authors**: Xinyue Zhu, Binghao Huang, Yunzhu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.15062)  

**Abstract**: Handheld grippers are increasingly used to collect human demonstrations due to their ease of deployment and versatility. However, most existing designs lack tactile sensing, despite the critical role of tactile feedback in precise manipulation. We present a portable, lightweight gripper with integrated tactile sensors that enables synchronized collection of visual and tactile data in diverse, real-world, and in-the-wild settings. Building on this hardware, we propose a cross-modal representation learning framework that integrates visual and tactile signals while preserving their distinct characteristics. The learning procedure allows the emergence of interpretable representations that consistently focus on contacting regions relevant for physical interactions. When used for downstream manipulation tasks, these representations enable more efficient and effective policy learning, supporting precise robotic manipulation based on multimodal feedback. We validate our approach on fine-grained tasks such as test tube insertion and pipette-based fluid transfer, demonstrating improved accuracy and robustness under external disturbances. Our project page is available at this https URL . 

**Abstract (ZH)**: 手持式夹持器由于易于部署和 versatility 越来越多地用于收集人类演示，但大多数现有设计缺乏触觉感知，而触觉反馈在精确操作中起着至关重要的作用。我们提出了一种便携式、轻量级的夹持器，集成了触觉传感器，能够在多种真实的、环境中的设置下同步收集视觉和触觉数据。基于这种硬件，我们提出了一种跨模态表示学习框架，该框架整合了视觉和触觉信号，同时保留了它们各自的特点。学习过程使得生成可解释的表示，并且能够始终聚焦于与物理交互相关的接触区域。在用于下游操作任务时，这些表示能够支持基于多模态反馈的更高效和有效的策略学习，从而实现精确的机器人操作。我们在细粒度任务，如试管插入和移液管基液体转移上验证了该方法，展示了在外部干扰下的更好准确性与鲁棒性。我们的项目页面网址为：这个 https URL 。 

---
# CPED-NCBFs: A Conformal Prediction for Expert Demonstration-based Neural Control Barrier Functions 

**Title (ZH)**: 基于专家示范的神经控制障碍函数的 conformal 预测 

**Authors**: Sumeadh MS, Kevin Dsouza, Ravi Prakash  

**Link**: [PDF](https://arxiv.org/pdf/2507.15022)  

**Abstract**: Among the promising approaches to enforce safety in control systems, learning Control Barrier Functions (CBFs) from expert demonstrations has emerged as an effective strategy. However, a critical challenge remains: verifying that the learned CBFs truly enforce safety across the entire state space. This is especially difficult when CBF is represented using neural networks (NCBFs). Several existing verification techniques attempt to address this problem including SMT-based solvers, mixed-integer programming (MIP), and interval or bound-propagation methods but these approaches often introduce loose, conservative bounds. To overcome these limitations, in this work we use CPED-NCBFs a split-conformal prediction based verification strategy to verify the learned NCBF from the expert demonstrations. We further validate our method on point mass systems and unicycle models to demonstrate the effectiveness of the proposed theory. 

**Abstract (ZH)**: 基于专家演示学习控制 barrier 函数的安全验证：使用 CPED-NCBFs 策略验证神经网络控制 barrier 函数的有效性 

---
# FCRF: Flexible Constructivism Reflection for Long-Horizon Robotic Task Planning with Large Language Models 

**Title (ZH)**: FCRF: 灵活的建构主义反思在大规模语言模型辅助下的长 horizon 机器人任务规划 

**Authors**: Yufan Song, Jiatao Zhang, Zeng Gu, Qingmiao Liang, Tuocheng Hu, Wei Song, Shiqiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14975)  

**Abstract**: Autonomous error correction is critical for domestic robots to achieve reliable execution of complex long-horizon tasks. Prior work has explored self-reflection in Large Language Models (LLMs) for task planning error correction; however, existing methods are constrained by inflexible self-reflection mechanisms that limit their effectiveness. Motivated by these limitations and inspired by human cognitive adaptation, we propose the Flexible Constructivism Reflection Framework (FCRF), a novel Mentor-Actor architecture that enables LLMs to perform flexible self-reflection based on task difficulty, while constructively integrating historical valuable experience with failure lessons. We evaluated FCRF on diverse domestic tasks through simulation in AlfWorld and physical deployment in the real-world environment. Experimental results demonstrate that FCRF significantly improves overall performance and self-reflection flexibility in complex long-horizon robotic tasks. 

**Abstract (ZH)**: 自主错误纠正对于实现家庭机器人可靠执行复杂长期任务至关重要。现有工作探索了大型语言模型在任务规划错误纠正中的自我反思；然而，现有方法受限于僵化的自我反思机制，限制了其有效性。受人类认知适应性的启发，我们提出了灵活建构主义反思框架（FCRF），这是一种新颖的导师-执行者架构，使大型语言模型能够根据任务难度进行灵活的自我反思，同时有建设性地结合历史有价值的经历与失败教训。我们在AlfWorld的模拟和现实世界环境中的物理部署中对FCRF进行了多种家庭任务的评估。实验结果表明，FCRF在复杂长期任务中显著提高了整体性能和自我反思的灵活性。 

---
# Heterogeneous object manipulation on nonlinear soft surface through linear controller 

**Title (ZH)**: 非线性软表面上异质物体操控的线性控制器方法 

**Authors**: Pratik Ingle, Kasper Støy, Andres Faiña  

**Link**: [PDF](https://arxiv.org/pdf/2507.14967)  

**Abstract**: Manipulation surfaces indirectly control and reposition objects by actively modifying their shape or properties rather than directly gripping objects. These surfaces, equipped with dense actuator arrays, generate dynamic deformations. However, a high-density actuator array introduces considerable complexity due to increased degrees of freedom (DOF), complicating control tasks. High DOF restrict the implementation and utilization of manipulation surfaces in real-world applications as the maintenance and control of such systems exponentially increase with array/surface size. Learning-based control approaches may ease the control complexity, but they require extensive training samples and struggle to generalize for heterogeneous objects. In this study, we introduce a simple, precise and robust PID-based linear close-loop feedback control strategy for heterogeneous object manipulation on MANTA-RAY (Manipulation with Adaptive Non-rigid Textile Actuation with Reduced Actuation density). Our approach employs a geometric transformation-driven PID controller, directly mapping tilt angle control outputs(1D/2D) to actuator commands to eliminate the need for extensive black-box training. We validate the proposed method through simulations and experiments on a physical system, successfully manipulating objects with diverse geometries, weights and textures, including fragile objects like eggs and apples. The outcomes demonstrate that our approach is highly generalized and offers a practical and reliable solution for object manipulation on soft robotic manipulation, facilitating real-world implementation without prohibitive training demands. 

**Abstract (ZH)**: 基于MANTA-RAY的异形物体Manipulation的简单、精确和鲁棒PID闭环反馈控制策略 

---
# Designing Robots with, not for: A Co-Design Framework for Empowering Interactions in Forensic Psychiatry 

**Title (ZH)**: 与人类共同设计机器人：赋能法医精神病学交互的共同设计框架 

**Authors**: Qiaoqiao Ren, Remko Proesmans, Arend Pissens, Lara Dehandschutter, William Denecker, Lotte Rouckhout, Joke Carrette, Peter Vanhopplinus, Tony Belpaeme, Francis wyffels  

**Link**: [PDF](https://arxiv.org/pdf/2507.14931)  

**Abstract**: Forensic mental health care involves the treatment of individuals with severe mental disorders who have committed violent offences. These settings are often characterized by high levels of bureaucracy, risk avoidance, and restricted autonomy. Patients frequently experience a profound loss of control over their lives, leading to heightened psychological stress-sometimes resulting in isolation as a safety measure. In this study, we explore how co-design can be used to collaboratively develop a companion robot that helps monitor and regulate stress while maintaining tracking of the patients' interaction behaviours for long-term intervention. We conducted four co-design workshops in a forensic psychiatric clinic with patients, caregivers, and therapists. Our process began with the presentation of an initial speculative prototype to therapists, enabling reflection on shared concerns, ethical risks, and desirable features. This was followed by a creative ideation session with patients, a third workshop focused on defining desired functions and emotional responses, and we are planning a final prototype demo to gather direct patient feedback. Our findings emphasize the importance of empowering patients in the design process and adapting proposals based on their current emotional state. The goal was to empower the patient in the design process and ensure each patient's voice was heard. 

**Abstract (ZH)**: 司法精神病护理涉及对有严重精神障碍并实施暴力行为的个体进行治疗。这些环境往往特征明显，包括高水平的官僚主义、风险规避和自主权受限。患者常常经历生活控制权的重大丧失，导致心理压力增加，有时作为安全措施导致隔离。在此研究中，我们探讨了如何利用共设计方法协作开发一个伴侣机器人，该机器人有助于监测和调节压力，并同时跟踪患者互动行为，为长期干预提供支持。我们在一家司法精神病诊所中与患者、护理人员和治疗师进行了四次共设计研讨会。我们的过程始于向治疗师展示初步假想原型，以促进对共同关注点、伦理风险和可选特性的反思。随后是一个有患者参与的创造性构思阶段，第三次会议集中在定义期望的功能和情感反应上，我们计划最后进行原型演示以收集直接的患者反馈。我们的研究发现强调了在设计过程中赋予患者权力的重要性，并根据他们当前的情感状态调整提议。目标是让患者参与设计过程，并确保每位患者的声音都能被听到。 

---
# Digital twin and extended reality for teleoperation of the electric vehicle battery disassembly 

**Title (ZH)**: 数字孪生与扩展现实 Electric Vehicle 电池拆解的远程操作 

**Authors**: Tero Kaarlela, Sami Salo, Jose Outeiro  

**Link**: [PDF](https://arxiv.org/pdf/2507.14929)  

**Abstract**: Disassembling and sorting Electric Vehicle Batteries (EVBs) supports a sustainable transition to electric vehicles by enabling a closed-loop supply chain. Currently, the manual disassembly process exposes workers to hazards, including electrocution and toxic chemicals. We propose a teleoperated system for the safe disassembly and sorting of EVBs. A human-in-the-loop can create and save disassembly sequences for unknown EVB types, enabling future automation. An RGB camera aligns the physical and digital twins of the EVB, and the digital twin of the robot is based on the Robot Operating System (ROS) middleware. This hybrid approach combines teleoperation and automation to improve safety, adaptability, and efficiency in EVB disassembly and sorting. The economic contribution is realized by reducing labor dependency and increasing throughput in battery recycling. An online pilot study was set up to evaluate the usability of the presented approach, and the results demonstrate the potential as a user-friendly solution. 

**Abstract (ZH)**: 拆解和分类电动汽车电池（EVBs）通过支持可持续向电动汽车转型，促进了闭环供应链的发展。目前，手动拆解过程会令工人暴露在触电和有毒化学物质等风险中。我们提出了一种用于安全拆解和分类EVBs的远程操作系统。人类介入循环可以为未知EVB类型创建并保存拆解序列，以实现未来自动化。RGB摄像头对齐EVB的物理双胞胎和数字双胞胎，机器人的数字双胞胎基于机器人操作系统（ROS）中间件。这种混合方法结合了远程操作和自动化，以提高电动汽车电池拆解和分类的安全性、灵活性和效率。经济效益体现在减少劳动依赖并提高电池回收产量上。一个在线试点研究被设置起来评估所提出方法的可用性，结果表明该方法具有用户友好的潜力。 

---
# One Step Beyond: Feedthrough & Placement-Aware Rectilinear Floorplanner 

**Title (ZH)**: 一步超越：传输意识的直角布线器 

**Authors**: Zhexuan Xu, Jie Wang, Siyuan Xu, Zijie Geng, Mingxuan Yuan, Feng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14914)  

**Abstract**: Floorplanning determines the shapes and locations of modules on a chip canvas and plays a critical role in optimizing the chip's Power, Performance, and Area (PPA) metrics. However, existing floorplanning approaches often fail to integrate with subsequent physical design stages, leading to suboptimal in-module component placement and excessive inter-module feedthrough. To tackle this challenge, we propose Flora, a three-stage feedthrough and placement aware rectilinear floorplanner. In the first stage, Flora employs wiremask and position mask techniques to achieve coarse-grained optimization of HPWL and feedthrough. In the second stage, under the constraint of a fixed outline, Flora achieves a zero-whitespace layout by locally resizing module shapes, thereby performing fine-grained optimization of feedthrough and improving component placement. In the third stage, Flora utilizes a fast tree search-based method to efficiently place components-including macros and standard cells-within each module, subsequently adjusting module boundaries based on the placement results to enable cross-stage optimization. Experimental results show that Flora outperforms recent state-of-the-art floorplanning approaches, achieving an average reduction of 6% in HPWL, 5.16% in FTpin, 29.15% in FTmod, and a 14% improvement in component placement performance. 

**Abstract (ZH)**: Flora：一种考虑布线和放置的矩形布板器 

---
# CoMoCAVs: Cohesive Decision-Guided Motion Planning for Connected and Autonomous Vehicles with Multi-Policy Reinforcement Learning 

**Title (ZH)**: CoMoCAVs：多策略强化学习引导的连接与自主车辆协同决策运动规划 

**Authors**: Pan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.14903)  

**Abstract**: Autonomous driving demands reliable and efficient solutions to closely related problems such as decision-making and motion planning. In this work, decision-making refers specifically to highway lane selection, while motion planning involves generating control commands (such as speed and steering) to reach the chosen lane. In the context of Connected Autonomous Vehicles (CAVs), achieving both flexible and safe lane selection alongside precise trajectory execution remains a significant challenge. This paper proposes a framework called Cohesive Decision-Guided Motion Planning (CDGMP), which tightly integrates decision-making and motion planning using a Mixture of Experts (MoE) inspired architecture combined with multi-policy reinforcement learning. By coordinating multiple specialized sub-networks through a gating mechanism, the method decomposes the complex driving task into modular components. Each sub-network focuses on a specific aspect of driving, improving efficiency by activating only the most relevant modules during inference. This design also enhances safety through modular specialization. CDGMP improves the adaptability and robustness of CAVs across diverse traffic scenarios, offering a scalable solution to real-world autonomy challenges. The architectural principles behind CDGMP, especially the use of MoE, also provide a strong foundation for other high-dimensional decision and control tasks. Simulation results (available at this https URL) demonstrate reliable performance in both lane selection and motion planning. 

**Abstract (ZH)**: 自主驾驶需要可靠高效的解决方案来解决密切相关的决策和运动规划问题。在本工作中，决策特指高速公路车道选择，而运动规划涉及生成控制命令（如速度和转向）以达到所选车道。在连接自主车辆（CAVs）的背景下，实现灵活且安全的车道选择以及精确的轨迹执行仍然是一个重大挑战。本文提出了一种称为凝聚力决策引导运动规划（CDGMP）的框架，该框架通过受Mixture of Experts（MoE）启发的架构结合多策略强化学习紧密集成决策与运动规划。通过闸门机制协调多个专门的子网络，该方法将复杂的驾驶任务分解为模块化组件。每个子网络专注于驾驶的具体方面，通过在推理时仅激活最相关的模块提高效率。该设计还通过模块化专业化增强了安全性。CDGMP提高了CAVs在不同交通场景下的适应性和鲁棒性，提供了针对现实世界自主挑战的可扩展解决方案。CDGMP背后的架构原则，尤其是MoE的应用，也为其他高维决策和控制任务提供了坚实的基础。模拟结果（可参见 [此处](this https URL)）在车道选择和运动规划方面展示了可靠的性能。 

---
# KGN-Pro: Keypoint-Based Grasp Prediction through Probabilistic 2D-3D Correspondence Learning 

**Title (ZH)**: KGN-Pro：基于关键点的概率二维-三维对应学习抓取预测 

**Authors**: Bingran Chen, Baorun Li, Jian Yang, Yong Liu, Guangyao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2507.14820)  

**Abstract**: High-level robotic manipulation tasks demand flexible 6-DoF grasp estimation to serve as a basic function. Previous approaches either directly generate grasps from point-cloud data, suffering from challenges with small objects and sensor noise, or infer 3D information from RGB images, which introduces expensive annotation requirements and discretization issues. Recent methods mitigate some challenges by retaining a 2D representation to estimate grasp keypoints and applying Perspective-n-Point (PnP) algorithms to compute 6-DoF poses. However, these methods are limited by their non-differentiable nature and reliance solely on 2D supervision, which hinders the full exploitation of rich 3D information. In this work, we present KGN-Pro, a novel grasping network that preserves the efficiency and fine-grained object grasping of previous KGNs while integrating direct 3D optimization through probabilistic PnP layers. KGN-Pro encodes paired RGB-D images to generate Keypoint Map, and further outputs a 2D confidence map to weight keypoint contributions during re-projection error minimization. By modeling the weighted sum of squared re-projection errors probabilistically, the network effectively transmits 3D supervision to its 2D keypoint predictions, enabling end-to-end learning. Experiments on both simulated and real-world platforms demonstrate that KGN-Pro outperforms existing methods in terms of grasp cover rate and success rate. 

**Abstract (ZH)**: 高阶机器人 manipulating 任务需要灵活的6-DOF 抓取估计作为基本功能。 

---
# X-Nav: Learning End-to-End Cross-Embodiment Navigation for Mobile Robots 

**Title (ZH)**: X-Nav: 学习端到端跨载体导航以供移动机器人使用 

**Authors**: Haitong Wang, Aaron Hao Tan, Angus Fung, Goldie Nejat  

**Link**: [PDF](https://arxiv.org/pdf/2507.14731)  

**Abstract**: Existing navigation methods are primarily designed for specific robot embodiments, limiting their generalizability across diverse robot platforms. In this paper, we introduce X-Nav, a novel framework for end-to-end cross-embodiment navigation where a single unified policy can be deployed across various embodiments for both wheeled and quadrupedal robots. X-Nav consists of two learning stages: 1) multiple expert policies are trained using deep reinforcement learning with privileged observations on a wide range of randomly generated robot embodiments; and 2) a single general policy is distilled from the expert policies via navigation action chunking with transformer (Nav-ACT). The general policy directly maps visual and proprioceptive observations to low-level control commands, enabling generalization to novel robot embodiments. Simulated experiments demonstrated that X-Nav achieved zero-shot transfer to both unseen embodiments and photorealistic environments. A scalability study showed that the performance of X-Nav improves when trained with an increasing number of randomly generated embodiments. An ablation study confirmed the design choices of X-Nav. Furthermore, real-world experiments were conducted to validate the generalizability of X-Nav in real-world environments. 

**Abstract (ZH)**: 现有的导航方法主要针对特定的机器人实体设计，限制了其在不同机器人平台间的通用性。本文引入了X-Nav，一种端到端跨实体导航框架，能够在多种实体上统一部署，适用于轮式和四足机器人。X-Nav包括两个学习阶段：1) 使用深度强化学习和特权观测在大量随机生成的机器人实体上训练多个专家策略；2) 通过Transformer（Nav-ACT）的动作片段蒸馏出一个通用策略。通用策略直接将视觉和本体感受观测映射到低层控制命令，实现对新机器人实体的泛化。模拟实验表明，X-Nav实现了对未见实体和照片级真实环境的零样本迁移。规模研究显示，X-Nav的性能随着训练过程中随机生成的实体数量增加而提高。消融研究证实了X-Nav的设计选择。此外，还进行了实地实验以验证X-Nav在真实环境中的泛化能力。 

---
# Leveraging Extrinsic Dexterity for Occluded Grasping on Grasp Constraining Walls 

**Title (ZH)**: 利用外在灵巧性进行障碍限制下的视觉被遮挡抓取 

**Authors**: Keita Kobashi, Masayoshi Tomizuka  

**Link**: [PDF](https://arxiv.org/pdf/2507.14721)  

**Abstract**: This study addresses the problem of occluded grasping, where primary grasp configurations of an object are not available due to occlusion with environment. Simple parallel grippers often struggle with such tasks due to limited dexterity and actuation constraints. Prior works have explored object pose reorientation such as pivoting by utilizing extrinsic contacts between an object and an environment feature like a wall, to make the object graspable. However, such works often assume the presence of a short wall, and this assumption may not always hold in real-world scenarios. If the wall available for interaction is too large or too tall, the robot may still fail to grasp the object even after pivoting, and the robot must combine different types of actions to grasp. To address this, we propose a hierarchical reinforcement learning (RL) framework. We use Q-learning to train a high-level policy that selects the type of action expected to yield the highest reward. The selected low-level skill then samples a specific robot action in continuous space. To guide the robot to an appropriate location for executing the selected action, we adopt a Conditional Variational Autoencoder (CVAE). We condition the CVAE on the object point cloud and the skill ID, enabling it to infer a suitable location based on the object geometry and the selected skill. To promote generalization, we apply domain randomization during the training of low-level skills. The RL policy is trained entirely in simulation with a box-like object and deployed to six objects in real world. We conduct experiments to evaluate our method and demonstrate both its generalizability and robust sim-to-real transfer performance with promising success rates. 

**Abstract (ZH)**: 本研究解决了由于环境遮挡导致物体主要抓持配置不可用的_occluded grasping_问题。简单的并联 gripper 由于灵活性和执行约束有限，在处理此类任务时常常遇到困难。先前的工作通过利用物体与环境特征（如墙壁）之间的外部接触来进行物体姿态重新定向（例如旋转），以使物体变得可抓取。然而，这些工作常常假设存在一个短墙，这在实际场景中并不总是成立的。如果用于交互的墙壁过大或过高，即使在旋转后机器人也可能无法抓取物体，并且机器人必须结合不同类型的行动来完成抓取。为了解决这一问题，我们提出了一种层次强化学习（RL）框架。我们使用Q-learning来训练一个高层策略，以便选择预期能带来最高奖励的操作类型。所选的低层技能随后在连续空间中采样一个具体的操作。为了引导机器人到执行所选操作的合适位置，我们采用了条件变分自编码器（CVAE）。我们将CVAE基于物体点云和技能ID进行条件化，使其能够根据物体几何形状和选定的技能推断出一个合适的位置。为了促进泛化，我们在低层技能的训练过程中应用领域随机化。RL策略完全在仿真中用一个框形物体进行训练，并部署到六个真实世界中的物体上。我们进行了实验来评估我们的方法，并展示了其泛化能力和有希望的成功率，以及模拟到现实世界转移的稳健性能。 

---
# Corridor-based Adaptive Control Barrier and Lyapunov Functions for Safe Mobile Robot Navigation 

**Title (ZH)**: 基于走廊的自适应控制屏障和李雅普诺夫函数的安全移动机器人导航 

**Authors**: Nicholas Mohammad, Nicola Bezzo  

**Link**: [PDF](https://arxiv.org/pdf/2507.14700)  

**Abstract**: Safe navigation in unknown and cluttered environments remains a challenging problem in robotics. Model Predictive Contour Control (MPCC) has shown promise for performant obstacle avoidance by enabling precise and agile trajectory tracking, however, existing methods lack formal safety assurances. To address this issue, we propose a general Control Lyapunov Function (CLF) and Control Barrier Function (CBF) enabled MPCC framework that enforces safety constraints derived from a free-space corridor around the planned trajectory. To enhance feasibility, we dynamically adapt the CBF parameters at runtime using a Soft Actor-Critic (SAC) policy. The approach is validated with extensive simulations and an experiment on mobile robot navigation in unknown cluttered environments. 

**Abstract (ZH)**: 在未知且拥挤环境中的安全导航仍然是机器人技术中的一个 Challing 问题。Model Predictive Contour Control (MPCC) 通过使能精确且灵活的轨迹跟踪展现了良好的碰撞避免潜力，然而现有方法缺乏形式化的安全性保证。为了解决这一问题，我们提出了一种通用的基于 Control Lyapunov Function (CLF) 和 Control Barrier Function (CBF) 的 MPCC 框架，该框架通过确保从计划轨迹周围自由空间走廊中导出的安全约束得到满足来实现安全性。为提高可行性，我们使用 Soft Actor-Critic (SAC) 策略在运行时动态调整 CBF 参数。该方法通过详尽的仿真和在未知拥挤环境中的移动机器人导航实验进行了验证。 

---
# Uncertainty-aware Probabilistic 3D Human Motion Forecasting via Invertible Networks 

**Title (ZH)**: 具有不确定性意识的概率性3D人体运动预测via可逆网络 

**Authors**: Yue Ma, Kanglei Zhou, Fuyang Yu, Frederick W. B. Li, Xiaohui Liang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14694)  

**Abstract**: 3D human motion forecasting aims to enable autonomous applications. Estimating uncertainty for each prediction (i.e., confidence based on probability density or quantile) is essential for safety-critical contexts like human-robot collaboration to minimize risks. However, existing diverse motion forecasting approaches struggle with uncertainty quantification due to implicit probabilistic representations hindering uncertainty modeling. We propose ProbHMI, which introduces invertible networks to parameterize poses in a disentangled latent space, enabling probabilistic dynamics modeling. A forecasting module then explicitly predicts future latent distributions, allowing effective uncertainty quantification. Evaluated on benchmarks, ProbHMI achieves strong performance for both deterministic and diverse prediction while validating uncertainty calibration, critical for risk-aware decision making. 

**Abstract (ZH)**: 3D人体运动预测旨在实现自主应用。为人类-机器人协作等安全关键场景中的每个预测估算不确定性（基于概率密度或分位数的信任度）至关重要。然而，现有多种运动预测方法由于隐式概率表示妨碍不确定性建模，在不确定性量化方面存在困难。我们提出ProbHMI，通过引入可逆网络在解耦的潜在空间中参数化姿态，实现概率动力学建模。随后，预测模块显式预测未来的潜在分布，从而有效进行不确定性量化。在基准测试上的评估表明，ProbHMI在确定性和多样化预测方面均表现出色，同时验证了不确定性校准的准确性，这对于风险感知决策至关重要。 

---
# Koopman Operator Based Linear Model Predictive Control for 2D Quadruped Trotting, Bounding, and Gait Transition 

**Title (ZH)**: 基于Koopman算子的线性模型预测控制在2D四足踏步、边界移动和步态转换中的应用 

**Authors**: Chun-Ming Yang, Pranav A. Bhounsule  

**Link**: [PDF](https://arxiv.org/pdf/2507.14605)  

**Abstract**: Online optimal control of quadrupedal robots would enable them to plan their movement in novel scenarios. Linear Model Predictive Control (LMPC) has emerged as a practical approach for real-time control. In LMPC, an optimization problem with a quadratic cost and linear constraints is formulated over a finite horizon and solved on the fly. However, LMPC relies on linearizing the equations of motion (EOM), which may lead to poor solution quality. In this paper, we use Koopman operator theory and the Extended Dynamic Mode Decomposition (EDMD) to create a linear model of the system in high dimensional space, thus retaining the nonlinearity of the EOM. We model the aerial phase and ground contact phases using different linear models. Then, using LMPC, we demonstrate bounding, trotting, and bound-to-trot and trot-to-bound gait transitions in level and rough terrains. The main novelty is the use of Koopman operator theory to create hybrid models of a quadrupedal system and demonstrate the online generation of multiple gaits and gaits transitions. 

**Abstract (ZH)**: 基于科胡曼算子理论的四足机器人在线最优控制及多步态生成与过渡 

---
# BT-TL-DMPs: A Novel Robot TAMP Framework Combining Behavior Tree, Temporal Logic and Dynamical Movement Primitives 

**Title (ZH)**: BT-TL-DMPs: 一种结合行为树、时序逻辑和动态运动primitive的新型机器人任务规划框架 

**Authors**: Zezhi Liu, Shizhen Wu, Hanqian Luo, Deyun Qin, Yongchun Fang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14582)  

**Abstract**: In the field of Learning from Demonstration (LfD), enabling robots to generalize learned manipulation skills to novel scenarios for long-horizon tasks remains challenging. Specifically, it is still difficult for robots to adapt the learned skills to new environments with different task and motion requirements, especially in long-horizon, multi-stage scenarios with intricate constraints. This paper proposes a novel hierarchical framework, called BT-TL-DMPs, that integrates Behavior Tree (BT), Temporal Logic (TL), and Dynamical Movement Primitives (DMPs) to address this problem. Within this framework, Signal Temporal Logic (STL) is employed to formally specify complex, long-horizon task requirements and constraints. These STL specifications are systematically transformed to generate reactive and modular BTs for high-level decision-making task structure. An STL-constrained DMP optimization method is proposed to optimize the DMP forcing term, allowing the learned motion primitives to adapt flexibly while satisfying intricate spatiotemporal requirements and, crucially, preserving the essential dynamics learned from demonstrations. The framework is validated through simulations demonstrating generalization capabilities under various STL constraints and real-world experiments on several long-horizon robotic manipulation tasks. The results demonstrate that the proposed framework effectively bridges the symbolic-motion gap, enabling more reliable and generalizable autonomous manipulation for complex robotic tasks. 

**Abstract (ZH)**: 基于行为树、时序逻辑和动力学运动范例的novel分层框架：解决长时_horizon任务中新环境下的操作技能泛化问题 

---
# A 21-DOF Humanoid Dexterous Hand with Hybrid SMA-Motor Actuation: CYJ Hand-0 

**Title (ZH)**: 一种由混合 SMA-电机驱动的21-DOF humanoid灵巧手：CYJ手-0 

**Authors**: Jin Chai, Xiang Yao, Mengfan Hou, Yanghong Li, Erbao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.14538)  

**Abstract**: CYJ Hand-0 is a 21-DOF humanoid dexterous hand featuring a hybrid tendon-driven actuation system that combines shape memory alloys (SMAs) and DC motors. The hand employs high-strength fishing line as artificial tendons and uses a fully 3D-printed AlSi10Mg metal frame designed to replicate the skeletal and tendon-muscle structure of the human hand. A linear motor-driven module controls finger flexion, while an SMA-based module enables finger extension and lateral abduction. These modules are integrated into a compact hybrid actuation unit mounted on a custom rear support structure. Mechanical and kinematic experiments, conducted under an Arduino Mega 2560-based control system, validate the effectiveness of the design and demonstrate its biomimetic dexterity. 

**Abstract (ZH)**: CYJ Hand-0是一种具有21-DOF的类人灵巧手，配备了混合肌腱驱动传动系统，结合了形状记忆合金（SMAs）和直流电机。该手使用高强度钓鱼线作为人工肌腱，并采用全3D打印的AlSi10Mg金属框架，设计模仿人类手的骨骼和肌腱-肌肉结构。线性电机驱动模块控制手指弯曲，而基于SMAs的模块实现手指伸展和横向外展。这些模块集成在一个紧凑的混合驱动单元中，安装在自定义后支撑结构上。基于Arduino Mega 2560控制系统的机械和运动学实验验证了该设计的有效性，并展示了其生物仿生灵巧性。 

---
# Koopman Operator Based Time-Delay Embeddings and State History Augmented LQR for Periodic Hybrid Systems: Bouncing Pendulum and Bipedal Walking 

**Title (ZH)**: 基于柯普曼算子的时间延迟嵌入与时变历史状态增强的LQR方法：摆动摆和双足步行周期混合系统 

**Authors**: Chun-Ming Yang, Pranav A. Bhounsule  

**Link**: [PDF](https://arxiv.org/pdf/2507.14455)  

**Abstract**: Time-delay embedding is a technique that uses snapshots of state history over time to build a linear state space model of a nonlinear smooth system. We demonstrate that periodic non-smooth or hybrid system can also be modeled as a linear state space system using this approach as long as its behavior is consistent in modes and timings. We extended time-delay embeddings to generate a linear model of two periodic hybrid systems: the bouncing pendulum and the simplest walker with control inputs. This leads to a novel state history augmented linear quadratic regulator (LQR) which uses current and past state history for feedback control. 

**Abstract (ZH)**: 时间延迟嵌入是一种技术，它利用时间上的状态历史截面来构建非线性光滑系统的线性状态空间模型。我们演示了只要其模式和时间行为一致，即使是周期性非光滑或混合系统也可以通过这种方法被建模为线性状态空间系统。我们扩展了时间延迟嵌入技术，以生成双周期混合系统的线性模型：摆锤和带有控制输入的最简单步行者。这导致了一种新的状态历史增强线性二次调节器（LQR），它使用当前和过去的状态历史来实现反馈控制。 

---
# Personalized Socially Assistive Robots With End-to-End Speech-Language Models For Well-Being Support 

**Title (ZH)**: 面向福祉支持的端到端语音-语言模型驱动的个性化社会辅助机器人 

**Authors**: Mengxue Fu, Zhonghao Shi, Minyu Huang, Siqi Liu, Mina Kian, Yirui Song, Maja J. Matarić  

**Link**: [PDF](https://arxiv.org/pdf/2507.14412)  

**Abstract**: Socially assistive robots (SARs) have shown great potential for supplementing well-being support. However, prior studies have found that existing dialogue pipelines for SARs remain limited in real-time latency, back-channeling, and personalized speech dialogue. Toward addressing these limitations, we propose using integrated end-to-end speech-language models (SLMs) with SARs. This work 1) evaluated the usability of an SLM-enabled SAR dialogue system through a small user study, and 2) identified remaining limitations through study user feedback to inform future improvements. We conducted a small within-participant user study with university students (N = 11) whose results showed that participants perceived an SLM-enabled SAR system as capable of providing empathetic feedback, natural turn-taking, back-channeling, and adaptive responses. We also found that participants reported the robot's nonverbal behaviors as lacking variability and synchronization with conversation, and the SLM's verbal feedback as generic and repetitive. These findings highlighted the need for real-time robot movement synchronized with conversation, improved prompting or fine-tuning to generate outputs better aligned with mental health practices, and more expressive, adaptive vocal generation. 

**Abstract (ZH)**: 社会化辅助机器人（SARs）在补充福祉支持方面显示出巨大的潜力。然而，先前的研究发现，现有的SAR对话管道在实时延迟、回话响应和个性化语音对话方面仍存在局限性。为解决这些局限性，我们提出了将集成的端到端语音-语言模型（SLM）与SAR结合使用的方案。本研究1) 通过小型用户研究评估了具备SLM的SAR对话系统的可用性，并2) 通过用户反馈识别了剩余的局限性，以指导未来的改进。我们对11名大学生开展了小型被试内用户研究，结果显示参与者认为SLM增强的SAR系统能够提供共情反馈、自然的轮流对话、回话响应和适应性回应。我们还发现参与者报告说，机器人的非言语行为缺乏变化性和与对话的同步性，而SLM的言语反馈则显得通用且重复。这些发现强调了需要实现与对话同步的实时机器人动作、改进提示或微调以生成更符合心理健康实践的输出，以及更具表现力和适应性的语音生成的必要性。 

---
# A Recursive Lie-Group Formulation for the Second-Order Time Derivatives of the Inverse Dynamics of parallel Kinematic Manipulators 

**Title (ZH)**: 递归群论 formulations 用于并联机械臂逆动力学的二阶时间导数 

**Authors**: Andreas Mueller, Shivesh Kumar, Thomas Kordik  

**Link**: [PDF](https://arxiv.org/pdf/2507.14274)  

**Abstract**: Series elastic actuators (SEA) were introduced for serial robotic arms. Their model-based trajectory tracking control requires the second time derivatives of the inverse dynamics solution, for which algorithms were proposed. Trajectory control of parallel kinematics manipulators (PKM) equipped with SEAs has not yet been pursued. Key element for this is the computationally efficient evaluation of the second time derivative of the inverse dynamics solution. This has not been presented in the literature, and is addressed in the present paper for the first time. The special topology of PKM is exploited reusing the recursive algorithms for evaluating the inverse dynamics of serial robots. A Lie group formulation is used and all relations are derived within this framework. Numerical results are presented for a 6-DOF Gough-Stewart platform (as part of an exoskeleton), and for a planar PKM when a flatness-based control scheme is applied. 

**Abstract (ZH)**: 基于系列弹性执行器的并联运动 manipulators 轨迹控制：序列弹性执行器 (SEA) 已在串行机器人臂中引入。其基于模型的轨迹跟踪控制需要逆动力学解的二阶时间导数，为此提出了一些算法。装备有 SEA 的并联运动学 manipulators (PKM) 的轨迹控制尚未被研究。这一问题的关键在于逆动力学解的二阶时间导数的高效计算。该问题尚未在文献中被解决，本论文首次对此进行了研究。利用并联运动学的特殊拓扑结构，重用了串行机器人逆动力学计算的递归算法，并使用李群形式表述，推导了所有关系。最后，对于一个 6 自由度 Gough-Stewart 平台 (作为外骨骼的一部分) 和一个平面并联运动学 manipulators，当应用基于平坦性的控制方案时，给出了数值结果。 

---
# Real-Time Communication-Aware Ride-Sharing Route Planning for Urban Air Mobility: A Multi-Source Hybrid Attention Reinforcement Learning Approach 

**Title (ZH)**: 基于实时通信的城市空中移动共享路线规划：多源混合注意力强化学习方法 

**Authors**: Yuejiao Xie, Maonan Wang, Di Zhou, Man-On Pun, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.14249)  

**Abstract**: Urban Air Mobility (UAM) systems are rapidly emerging as promising solutions to alleviate urban congestion, with path planning becoming a key focus area. Unlike ground transportation, UAM trajectory planning has to prioritize communication quality for accurate location tracking in constantly changing environments to ensure safety. Meanwhile, a UAM system, serving as an air taxi, requires adaptive planning to respond to real-time passenger requests, especially in ride-sharing scenarios where passenger demands are unpredictable and dynamic. However, conventional trajectory planning strategies based on predefined routes lack the flexibility to meet varied passenger ride demands. To address these challenges, this work first proposes constructing a radio map to evaluate the communication quality of urban airspace. Building on this, we introduce a novel Multi-Source Hybrid Attention Reinforcement Learning (MSHA-RL) framework for the challenge of effectively focusing on passengers and UAM locations, which arises from the significant dimensional disparity between the representations. This model first generates the alignment among diverse data sources with large gap dimensions before employing hybrid attention to balance global and local insights, thereby facilitating responsive, real-time path planning. Extensive experimental results demonstrate that the approach enables communication-compliant trajectory planning, reducing travel time and enhancing operational efficiency while prioritizing passenger safety. 

**Abstract (ZH)**: 城市空中 mobility (UAM) 系统正迅速成为缓解城市拥堵的有前途的解决方案，路径规划成为关键研究领域。不同于地面交通，UAM 轨迹规划必须优先考虑通信质量以实现准确的位置跟踪，确保在不断变化的环境中安全运行。同时，作为空中出租车的UAM系统需要根据实时乘客请求进行适应性规划，特别是乘客需求在拼车场景中具有不可预测性和动态性时。然而，基于预定义路线的传统轨迹规划策略缺乏满足多样化乘客出行需求的灵活性。为应对这些挑战，本研究首先提出构建无线地图以评估城市空域的通信质量。在此基础上，我们引入了一种新颖的多源混合注意力强化学习（MSHA-RL）框架，有效聚焦于乘客和UAM位置，克服了表示之间的巨大维度差异。该模型首先生成具有大差距维度的多种数据源之间的对齐，然后使用混合注意力平衡全局和局部洞察，从而促进响应性和实时路径规划。广泛实验证明，该方法实现了通信合规的轨迹规划，缩短了旅行时间，提高了运营效率，并优先考虑乘客安全。 

---
# Diffusion Beats Autoregressive in Data-Constrained Settings 

**Title (ZH)**: 数据受限环境中，扩散模型优于自回归模型 

**Authors**: Mihir Prabhudesai, Menging Wu, Amir Zadeh, Katerina Fragkiadaki, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2507.15857)  

**Abstract**: Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: this https URL. 

**Abstract (ZH)**: 自回归（AR）模型长期以来主导着大型语言模型的格局，推动了诸多任务的进步。近期，基于扩散的语言模型 emerged as a promising alternative，尽管与 AR 模型相比的优势仍待进一步探索。在本文中，我们系统地研究了在数据受限的设置下掩蔽扩散模型的表现——在这种设置下，训练涉及对有限数据的多次迭代——发现当计算资源丰富但数据稀缺时，扩散模型显著优于自回归模型。扩散模型更有效地利用了重复的数据，实现了较低的验证损失和更出色的下游性能。我们将这种优势解释为隐式的数据增强：掩蔽扩散模型使模型暴露于多种不同的令牌顺序和预测任务分布中，而自回归模型则具有固定的从左到右的因式分解。我们为扩散模型发现了新的标度定律，并推导出了扩散开始优于自回归的关键计算阈值。这些结果表明，当数据而非计算成为瓶颈时，扩散模型为标准的自回归范式提供了一个令人信服的替代方案。我们的代码可在以下链接获取：this https URL。 

---
# Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos 

**Title (ZH)**: Being-H0: 大规模人类视频上的视觉-语言-行动预训练 

**Authors**: Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.15597)  

**Abstract**: We introduce Being-H0, a dexterous Vision-Language-Action model (VLA) trained on large-scale human videos. Existing VLAs struggle with complex manipulation tasks requiring high dexterity and generalize poorly to novel scenarios and tasks, primarily due to their reliance on synthetic data with significant sim-to-real gaps or teleoperated demonstrations lacking scale and diversity. To address this data bottleneck, we propose leveraging human hands as a foundation manipulator, capitalizing on the rich dexterity and scalability present in web data. Our approach centers on physical instruction tuning, a novel training paradigm that combines large-scale VLA pretraining from human videos, physical space alignment for 3D reasoning, and post-training adaptation for robotic tasks. Additionally, we introduce a part-level motion tokenization method which achieves millimeter-level reconstruction accuracy to model precise hand trajectories for action learning. To support our proposed paradigm, we further develop a comprehensive data curation pipeline that integrates heterogeneous sources -- including motion capture, VR, and RGB-only videos -- into a large-scale dataset with millions of motion-based instructional instances. We empirically show the excellence of Being-H0 in hand motion generation and instruction following, and it also scales well with model and data sizes. Importantly, we observe the expected gains of Being-H0 in real-world robotic manipulation as physical instruction tuning is applied. More details are available at this https URL. 

**Abstract (ZH)**: Being-H0：一种基于大规模人类视频训练的灵巧的视觉-语言-行动模型 

---
# Improving Functional Reliability of Near-Field Monitoring for Emergency Braking in Autonomous Vehicles 

**Title (ZH)**: 提高自动驾驶车辆近场监测在紧急制动功能可靠性 

**Authors**: Junnan Pan, Prodromos Sotiriadis, Vladislav Nenchev, Ferdinand Englberger  

**Link**: [PDF](https://arxiv.org/pdf/2507.15594)  

**Abstract**: Autonomous vehicles require reliable hazard detection. However, primary sensor systems may miss near-field obstacles, resulting in safety risks. Although a dedicated fast-reacting near-field monitoring system can mitigate this, it typically suffers from false positives. To mitigate these, in this paper, we introduce three monitoring strategies based on dynamic spatial properties, relevant object sizes, and motion-aware prediction. In experiments in a validated simulation, we compare the initial monitoring strategy against the proposed improvements. The results demonstrate that the proposed strategies can significantly improve the reliability of near-field monitoring systems. 

**Abstract (ZH)**: 自主驾驶车辆需要可靠的危险检测。然而，主要传感器系统可能会忽略近场障碍物，导致安全风险。虽然专用的快速反应近场监测系统可以缓解这一问题，但它们通常会面临误报的问题。为缓解这一问题，本文引入了基于动态空间特性、相关对象尺寸和运动感知预测的三种监测策略。在验证模拟实验中，我们将初始监测策略与所提出的改进策略进行了比较。结果表明，所提出的策略可以显著提高近场监测系统的可靠性。 

---
# Dense-depth map guided deep Lidar-Visual Odometry with Sparse Point Clouds and Images 

**Title (ZH)**: 稠密深度图引导的稀疏点云和图像的深度 Lidar-视觉里程计 

**Authors**: JunYing Huang, Ao Xu, DongSun Yong, KeRen Li, YuanFeng Wang, Qi Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.15496)  

**Abstract**: Odometry is a critical task for autonomous systems for self-localization and navigation. We propose a novel LiDAR-Visual odometry framework that integrates LiDAR point clouds and images for accurate and robust pose estimation. Our method utilizes a dense-depth map estimated from point clouds and images through depth completion, and incorporates a multi-scale feature extraction network with attention mechanisms, enabling adaptive depth-aware representations. Furthermore, we leverage dense depth information to refine flow estimation and mitigate errors in occlusion-prone regions. Our hierarchical pose refinement module optimizes motion estimation progressively, ensuring robust predictions against dynamic environments and scale ambiguities. Comprehensive experiments on the KITTI odometry benchmark demonstrate that our approach achieves similar or superior accuracy and robustness compared to state-of-the-art visual and LiDAR odometry methods. 

**Abstract (ZH)**: 激光雷达-视觉里程计框架：融合点云和图像实现准确可靠的位姿估计 

---
# From Kicking to Causality: Simulating Infant Agency Detection with a Robust Intrinsic Reward 

**Title (ZH)**: 从踢腿到因果关系：基于稳健内在奖励的婴儿自主性检测模拟 

**Authors**: Xia Xu, Jochen Triesch  

**Link**: [PDF](https://arxiv.org/pdf/2507.15106)  

**Abstract**: While human infants robustly discover their own causal efficacy, standard reinforcement learning agents remain brittle, as their reliance on correlation-based rewards fails in noisy, ecologically valid scenarios. To address this, we introduce the Causal Action Influence Score (CAIS), a novel intrinsic reward rooted in causal inference. CAIS quantifies an action's influence by measuring the 1-Wasserstein distance between the learned distribution of sensory outcomes conditional on that action, $p(h|a)$, and the baseline outcome distribution, $p(h)$. This divergence provides a robust reward that isolates the agent's causal impact from confounding environmental noise. We test our approach in a simulated infant-mobile environment where correlation-based perceptual rewards fail completely when the mobile is subjected to external forces. In stark contrast, CAIS enables the agent to filter this noise, identify its influence, and learn the correct policy. Furthermore, the high-quality predictive model learned for CAIS allows our agent, when augmented with a surprise signal, to successfully reproduce the "extinction burst" phenomenon. We conclude that explicitly inferring causality is a crucial mechanism for developing a robust sense of agency, offering a psychologically plausible framework for more adaptive autonomous systems. 

**Abstract (ZH)**: 因果动作影响评分在提高强化学习代理鲁棒性中的应用：一种基于因果推断的内在奖励方法 

---
# Visual Place Recognition for Large-Scale UAV Applications 

**Title (ZH)**: 大型无人机应用中的视觉场所识别 

**Authors**: Ioannis Tsampikos Papapetros, Ioannis Kansizoglou, Antonios Gasteratos  

**Link**: [PDF](https://arxiv.org/pdf/2507.15089)  

**Abstract**: Visual Place Recognition (vPR) plays a crucial role in Unmanned Aerial Vehicle (UAV) navigation, enabling robust localization across diverse environments. Despite significant advancements, aerial vPR faces unique challenges due to the limited availability of large-scale, high-altitude datasets, which limits model generalization, along with the inherent rotational ambiguity in UAV imagery. To address these challenges, we introduce LASED, a large-scale aerial dataset with approximately one million images, systematically sampled from 170,000 unique locations throughout Estonia over a decade, offering extensive geographic and temporal diversity. Its structured design ensures clear place separation significantly enhancing model training for aerial scenarios. Furthermore, we propose the integration of steerable Convolutional Neural Networks (CNNs) to explicitly handle rotational variance, leveraging their inherent rotational equivariance to produce robust, orientation-invariant feature representations. Our extensive benchmarking demonstrates that models trained on LASED achieve significantly higher recall compared to those trained on smaller, less diverse datasets, highlighting the benefits of extensive geographic coverage and temporal diversity. Moreover, steerable CNNs effectively address rotational ambiguity inherent in aerial imagery, consistently outperforming conventional convolutional architectures, achieving on average 12\% recall improvement over the best-performing non-steerable network. By combining structured, large-scale datasets with rotation-equivariant neural networks, our approach significantly enhances model robustness and generalization for aerial vPR. 

**Abstract (ZH)**: 大规模空中视觉场所识别数据集（LASED）及可旋转的卷积神经网络在空中视觉场所识别中的应用 

---
# EBA-AI: Ethics-Guided Bias-Aware AI for Efficient Underwater Image Enhancement and Coral Reef Monitoring 

**Title (ZH)**: EBA-AI: 道德引导的偏置意识人工智能在水下图像增强与珊瑚礁监测中的高效应用 

**Authors**: Lyes Saad Saoud, Irfan Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2507.15036)  

**Abstract**: Underwater image enhancement is vital for marine conservation, particularly coral reef monitoring. However, AI-based enhancement models often face dataset bias, high computational costs, and lack of transparency, leading to potential misinterpretations. This paper introduces EBA-AI, an ethics-guided bias-aware AI framework to address these challenges. EBA-AI leverages CLIP embeddings to detect and mitigate dataset bias, ensuring balanced representation across varied underwater environments. It also integrates adaptive processing to optimize energy efficiency, significantly reducing GPU usage while maintaining competitive enhancement quality. Experiments on LSUI400, Oceanex, and UIEB100 show that while PSNR drops by a controlled 1.0 dB, computational savings enable real-time feasibility for large-scale marine monitoring. Additionally, uncertainty estimation and explainability techniques enhance trust in AI-driven environmental decisions. Comparisons with CycleGAN, FunIEGAN, RAUNENet, WaterNet, UGAN, PUGAN, and UTUIE validate EBA-AI's effectiveness in balancing efficiency, fairness, and interpretability in underwater image processing. By addressing key limitations of AI-driven enhancement, this work contributes to sustainable, bias-aware, and computationally efficient marine conservation efforts. For interactive visualizations, animations, source code, and access to the preprint, visit: this https URL 

**Abstract (ZH)**: 基于伦理引导的偏见感知AI框架EBA-AI在水下图像增强中的应用：平衡效率、公平性和可解释性 

---
# Hierarchical Multi-Agent Reinforcement Learning with Control Barrier Functions for Safety-Critical Autonomous Systems 

**Title (ZH)**: 基于控制屏障函数的分层多代理强化学习在关键安全自主系统中的应用 

**Authors**: H. M. Sabbir Ahmad, Ehsan Sabouni, Alexander Wasilkoff, Param Budhraja, Zijian Guo, Songyuan Zhang, Chuchu Fan, Christos Cassandras, Wenchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14850)  

**Abstract**: We address the problem of safe policy learning in multi-agent safety-critical autonomous systems. In such systems, it is necessary for each agent to meet the safety requirements at all times while also cooperating with other agents to accomplish the task. Toward this end, we propose a safe Hierarchical Multi-Agent Reinforcement Learning (HMARL) approach based on Control Barrier Functions (CBFs). Our proposed hierarchical approach decomposes the overall reinforcement learning problem into two levels learning joint cooperative behavior at the higher level and learning safe individual behavior at the lower or agent level conditioned on the high-level policy. Specifically, we propose a skill-based HMARL-CBF algorithm in which the higher level problem involves learning a joint policy over the skills for all the agents and the lower-level problem involves learning policies to execute the skills safely with CBFs. We validate our approach on challenging environment scenarios whereby a large number of agents have to safely navigate through conflicting road networks. Compared with existing state of the art methods, our approach significantly improves the safety achieving near perfect (within 5%) success/safety rate while also improving performance across all the environments. 

**Abstract (ZH)**: 多代理安全关键自主系统中的安全策略学习 

---
# Light Future: Multimodal Action Frame Prediction via InstructPix2Pix 

**Title (ZH)**: 光明未来：基于InstructPix2Pix的多模态动作框架预测 

**Authors**: Zesen Zhong, Duomin Zhang, Yijia Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.14809)  

**Abstract**: Predicting future motion trajectories is a critical capability across domains such as robotics, autonomous systems, and human activity forecasting, enabling safer and more intelligent decision-making. This paper proposes a novel, efficient, and lightweight approach for robot action prediction, offering significantly reduced computational cost and inference latency compared to conventional video prediction models. Importantly, it pioneers the adaptation of the InstructPix2Pix model for forecasting future visual frames in robotic tasks, extending its utility beyond static image editing. We implement a deep learning-based visual prediction framework that forecasts what a robot will observe 100 frames (10 seconds) into the future, given a current image and a textual instruction. We repurpose and fine-tune the InstructPix2Pix model to accept both visual and textual inputs, enabling multimodal future frame prediction. Experiments on the RoboTWin dataset (generated based on real-world scenarios) demonstrate that our method achieves superior SSIM and PSNR compared to state-of-the-art baselines in robot action prediction tasks. Unlike conventional video prediction models that require multiple input frames, heavy computation, and slow inference latency, our approach only needs a single image and a text prompt as input. This lightweight design enables faster inference, reduced GPU demands, and flexible multimodal control, particularly valuable for applications like robotics and sports motion trajectory analytics, where motion trajectory precision is prioritized over visual fidelity. 

**Abstract (ZH)**: 预测未来运动轨迹是机器人学、自主系统和人体活动预测等领域的关键能力，能够实现更安全、更智能的决策。本文提出了一种新型、高效且轻量级的机器人动作预测方法，与传统的视频预测模型相比，显著降低了计算成本和推理延迟。更重要的是，该方法率先将InstructPix2Pix模型应用于机器人任务中的未来视觉帧预测，使其用途超出静态图像编辑。我们实现了一种基于深度学习的视觉预测框架，能够在给定当前图像和文本指令的情况下，预测机器人100帧（10秒）后的观察内容。我们重新利用并微调了InstructPix2Pix模型，使其能够接受视觉和文本输入，实现多模态未来帧预测。在基于真实场景生成的RoboTWin数据集上的实验表明，我们的方法在机器人动作预测任务中优于最先进的基线方法，在SSIM和PSNR指标上表现更优。与需要多帧输入、大量计算和慢速推理延迟的传统视频预测模型相比，我们的方法仅需单张图像和文本提示作为输入，这种轻量级设计能够实现更快的推理、减少GPU需求，并提供灵活的多模态控制，特别适用于如机器人学和体育运动轨迹分析等对运动轨迹精度要求较高的应用。 

---
# Motion Segmentation and Egomotion Estimation from Event-Based Normal Flow 

**Title (ZH)**: 基于事件的法向流的运动分割与自我运动估计 

**Authors**: Zhiyuan Hua, Dehao Yuan, Cornelia Fermüller  

**Link**: [PDF](https://arxiv.org/pdf/2507.14500)  

**Abstract**: This paper introduces a robust framework for motion segmentation and egomotion estimation using event-based normal flow, tailored specifically for neuromorphic vision sensors. In contrast to traditional methods that rely heavily on optical flow or explicit depth estimation, our approach exploits the sparse, high-temporal-resolution event data and incorporates geometric constraints between normal flow, scene structure, and inertial measurements. The proposed optimization-based pipeline iteratively performs event over-segmentation, isolates independently moving objects via residual analysis, and refines segmentations using hierarchical clustering informed by motion similarity and temporal consistency. Experimental results on the EVIMO2v2 dataset validate that our method achieves accurate segmentation and translational motion estimation without requiring full optical flow computation. This approach demonstrates significant advantages at object boundaries and offers considerable potential for scalable, real-time robotic and navigation applications. 

**Abstract (ZH)**: 一种基于事件驱动法向流的鲁棒运动分割与自我运动估计框架：面向神经形态视觉传感器的应用 

---
# GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving 

**Title (ZH)**: GEMINUS: 具有双重意识的全局与场景自适应混合专家模型用于端到端自动驾驶 

**Authors**: Chi Wan, Yixin Cui, Jiatong Du, Shuo Yang, Yulong Bai, Yanjun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.14456)  

**Abstract**: End-to-end autonomous driving requires adaptive and robust handling of complex and diverse traffic environments. However, prevalent single-mode planning methods attempt to learn an overall policy while struggling to acquire diversified driving skills to handle diverse scenarios. Therefore, this paper proposes GEMINUS, a Mixture-of-Experts end-to-end autonomous driving framework featuring a Global Expert, a Scene-Adaptive Experts Group, and equipped with a Dual-aware Router. Specifically, the Global Expert is trained on the overall dataset, possessing robust performance. The Scene-Adaptive Experts are trained on corresponding scene subsets, achieving adaptive performance. The Dual-aware Router simultaneously considers scenario-level features and routing uncertainty to dynamically activate expert modules. Through the effective coupling of the Global Expert and the Scene-Adaptive Experts Group via the Dual-aware Router, GEMINUS achieves adaptive and robust performance in diverse scenarios. GEMINUS outperforms existing methods in the Bench2Drive closed-loop benchmark and achieves state-of-the-art performance in Driving Score and Success Rate, even with only monocular vision input. Furthermore, ablation studies demonstrate significant improvements over the original single-expert baseline: 7.67% in Driving Score, 22.06% in Success Rate, and 19.41% in MultiAbility-Mean. The code will be available at this https URL. 

**Abstract (ZH)**: 端到端自动驾驶需要适应和应对复杂多变的交通环境。然而，现有的单模式规划方法试图学习整体策略，但在获取多样的驾驶技能以应对各种场景方面存在困难。因此，本文提出GEMINUS，这是一种混合专家端到端自动驾驶框架，包含全局专家、场景自适应专家组和双意识路由器。具体而言，全局专家在整体数据集上进行训练，具备稳健的表现。场景自适应专家在相应的场景子集上进行训练，实现适应性表现。双意识路由器同时考虑场景级别的特征和路由不确定性，动态激活专家模块。通过全局专家和场景自适应专家组通过双意识路由器的有效耦合，GEMINUS在各种场景中实现了适应性和稳健性。GEMINUS在Bench2Drive闭环基准测试中优于现有方法，并在驾驶评分和成功率上达到最先进的性能，即使仅输入单目视觉信息。此外，消融研究显示相对于原始单一专家基线有显著改进：驾驶评分提高7.67%、成功率提高22.06%、多能力平均提高19.41%。代码将在此网址获取。 

---
# Traffic Signal Phase and Timing Estimation with Large-Scale Floating Car Data 

**Title (ZH)**: 大规模浮动车数据驱动的交通信号相位和定时估计 

**Authors**: Mingcheng Liao, Zebang Feng, Miao Fan, Shengtong Xu, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.14190)  

**Abstract**: Effective modern transportation systems depend critically on accurate Signal Phase and Timing (SPaT) estimation. However, acquiring ground-truth SPaT information faces significant hurdles due to communication challenges with transportation departments and signal installers. As a result, Floating Car Data (FCD) has become the primary source for large-scale SPaT analyses. Current FCD approaches often simplify the problem by assuming fixed schedules and basic intersection designs for specific times and locations. These methods fail to account for periodic signal changes, diverse intersection structures, and the inherent limitations of real-world data, thus lacking a comprehensive framework that is universally applicable. Addressing this limitation, we propose an industrial-grade FCD analysis suite that manages the entire process, from initial data preprocessing to final SPaT estimation. Our approach estimates signal phases, identifies time-of-day (TOD) periods, and determines the durations of red and green lights. The framework's notable stability and robustness across diverse conditions, regardless of road geometry, is a key feature. Furthermore, we provide a cleaned, de-identified FCD dataset and supporting parameters to facilitate future research. Currently operational within our navigation platform, the system analyses over 15 million FCD records daily, supporting over two million traffic signals in mainland China, with more than 75\% of estimations demonstrating less than five seconds of error. 

**Abstract (ZH)**: 有效的现代交通系统高度依赖准确的信号相位和定时（SPaT）估计。然而，由于与交通管理部门和信号安装者通信的挑战，获取地面真实SPaT信息面临重大障碍。因此，浮动车数据（FCD）已成为大规模SPaT分析的主要来源。当前的FCD方法往往通过假设固定的时间表和简单的交叉口设计来简化问题，这些方法未能考虑到周期性的信号变化、多样的交叉口结构以及现实世界数据的固有限制，缺乏一个普遍适用的全面框架。为了克服这一限制，我们提出了一种工业级的FCD分析套件，从初始数据预处理到最终的SPaT估计，管理整个过程。我们的方法估算信号相位、识别时间周期（TOD）并确定红绿灯持续时间。框架的显著稳定性和在各种条件下表现出的鲁棒性，即使在不同的道路几何条件下也是如此，是一个关键特征。此外，我们还提供了一个清理和脱识别后的FCD数据集及支持参数，以促进未来的研究。目前该系统已在我们的导航平台中运行，每天分析超过1500万条FCD记录，支持中国大陆超过200万个交通信号，超过75%的估计误差少于五秒。 

---
