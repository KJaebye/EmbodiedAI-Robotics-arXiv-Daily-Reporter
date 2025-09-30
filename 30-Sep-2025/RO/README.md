# Safe Planning in Unknown Environments using Conformalized Semantic Maps 

**Title (ZH)**: 在未知环境中基于可信语义地图的安全规划 

**Authors**: David Smith Sundarsingh, Yifei Li, Tianji Tang, George J. Pappas, Nikolay Atanasov, Yiannis Kantaros  

**Link**: [PDF](https://arxiv.org/pdf/2509.25124)  

**Abstract**: This paper addresses semantic planning problems in unknown environments under perceptual uncertainty. The environment contains multiple unknown semantically labeled regions or objects, and the robot must reach desired locations while maintaining class-dependent distances from them. We aim to compute robot paths that complete such semantic reach-avoid tasks with user-defined probability despite uncertain perception. Existing planning algorithms either ignore perceptual uncertainty - thus lacking correctness guarantees - or assume known sensor models and noise characteristics. In contrast, we present the first planner for semantic reach-avoid tasks that achieves user-specified mission completion rates without requiring any knowledge of sensor models or noise. This is enabled by quantifying uncertainty in semantic maps - constructed on-the-fly from perceptual measurements - using conformal prediction in a model- and distribution-free manner. We validate our approach and the theoretical mission completion rates through extensive experiments, showing that it consistently outperforms baselines in mission success rates. 

**Abstract (ZH)**: 本文探讨了感知不确定性下未知环境中的语义规划问题。环境中包含多个未知语义标记的区域或物体，机器人必须到达目标位置并保持类相关的距离。我们的目标是在用户定义的概率下，尽管存在感知不确定性，仍能计算出完成此类语义接近避免任务的机器人路径。现有的规划算法要么忽视感知不确定性，从而缺乏正确性保证，要么假设已知传感器模型和噪声特性。相比之下，我们提出了第一个在无需任何传感器模型或噪声知识的情况下，实现用户指定的任务完成率的语义接近避免任务规划器。这得益于通过在无模型和无分布的前提下，使用自适应预测对从感知测量动态构建的语义地图中的不确定性进行量化。我们通过大量的实验验证了该方法和理论的任务完成率，并且表明它在任务成功率上始终优于基线方法。 

---
# Curriculum Imitation Learning of Distributed Multi-Robot Policies 

**Title (ZH)**: 分布式多机器人政策的课程模仿学习 

**Authors**: Jesús Roche, Eduardo Sebastián, Eduardo Montijano  

**Link**: [PDF](https://arxiv.org/pdf/2509.25097)  

**Abstract**: Learning control policies for multi-robot systems (MRS) remains a major challenge due to long-term coordination and the difficulty of obtaining realistic training data. In this work, we address both limitations within an imitation learning framework. First, we shift the typical role of Curriculum Learning in MRS, from scalability with the number of robots, to focus on improving long-term coordination. We propose a curriculum strategy that gradually increases the length of expert trajectories during training, stabilizing learning and enhancing the accuracy of long-term behaviors. Second, we introduce a method to approximate the egocentric perception of each robot using only third-person global state demonstrations. Our approach transforms idealized trajectories into locally available observations by filtering neighbors, converting reference frames, and simulating onboard sensor variability. Both contributions are integrated into a physics-informed technique to produce scalable, distributed policies from observations. We conduct experiments across two tasks with varying team sizes and noise levels. Results show that our curriculum improves long-term accuracy, while our perceptual estimation method yields policies that are robust to realistic uncertainty. Together, these strategies enable the learning of robust, distributed controllers from global demonstrations, even in the absence of expert actions or onboard measurements. 

**Abstract (ZH)**: 基于模仿学习的多机器人系统控制策略学习仍是一项重大挑战，由于长期协调的复杂性和现实训练数据的获取难度。在本工作中，我们在一个模仿学习框架内同时解决了这两个限制。首先，我们将Curriculum Learning在多机器人系统中的典型角色从随着机器人数量增加的可扩展性，转向专注于改善长期协调。我们提出了一种渐进增加专家轨迹长度的课程策略，以稳定学习并提高长期行为的准确性。其次，我们引入了一种方法，仅使用第三人称全局状态演示来近似每个机器人的第一人称感知。我们的方法通过过滤邻居、转换参考系和模拟机载传感器的变异性，将理想化的轨迹转化为局部可用的观察。这两项贡献被集成到一个基于物理的方法中，从观察中生成可扩展且分布式化的策略。我们在两个具有不同团队规模和噪声级别的任务上进行了实验。结果显示，我们的课程学习方法提高了长期准确性，而我们的感知估计方法则产生了对现实不确定性具有鲁棒性的策略。这些策略共同使全局演示能够实现鲁棒的分布式控制器学习，即使在没有专家行动或机载测量的情况下也是如此。 

---
# Crop Spirals: Re-thinking the field layout for future robotic agriculture 

**Title (ZH)**: 作物螺旋布局：重新思考未来的机器人农业田间布局 

**Authors**: Lakshan Lavan, Lanojithan Thiyagarasa, Udara Muthugala, Rajitha de Silva  

**Link**: [PDF](https://arxiv.org/pdf/2509.25091)  

**Abstract**: Conventional linear crop layouts, optimised for tractors, hinder robotic navigation with tight turns, long travel distances, and perceptual aliasing. We propose a robot-centric square spiral layout with a central tramline, enabling simpler motion and more efficient coverage. To exploit this geometry, we develop a navigation stack combining DH-ResNet18 waypoint regression, pixel-to-odometry mapping, A* planning, and model predictive control (MPC). In simulations, the spiral layout yields up to 28% shorter paths and about 25% faster execution for waypoint-based tasks across 500 waypoints than linear layouts, while full-field coverage performance is comparable to an optimised linear U-turn strategy. Multi-robot studies demonstrate efficient coordination on the spirals rule-constrained graph, with a greedy allocator achieving 33-37% lower batch completion times than a Hungarian assignment under our setup. These results highlight the potential of redesigning field geometry to better suit autonomous agriculture. 

**Abstract (ZH)**: 传统的以拖拉机为导向的线性作物布局妨碍了机器人的导航，尤其是转弯紧、行距长和感知 aliasing 的情况。我们提出了一种以机器人为中心的方形螺旋布局，具有中央 tramline，这种布局简化了运动并提高了覆盖效率。为利用这一几何结构，我们开发了一套导航栈，结合了 DH-ResNet18 路点回归、像素到里程计映射、A* 规划和模型预测控制 (MPC)。模拟结果显示，与线性布局相比，螺旋布局在500个路点的任务中可缩短高达28%的路径，并加快约25%的任务执行速度，同时全田覆盖性能与优化的线性U形转弯策略相当。多机器人研究展示了在螺旋规则约束图上的高效协调，我们设定下的贪婪分配器在批处理完成时间上比匈牙利分配实现了33-37%的降低。这些结果突显了重新设计田地几何结构以更好地适应自主农业的潜力。 

---
# AgriCruiser: An Open Source Agriculture Robot for Over-the-row Navigation 

**Title (ZH)**: AgriCruiser: 一种开源的行间农业机器人 

**Authors**: Kenny Truong, Yongkyu Lee, Jason Irie, Shivam Kumar Panda, Shahab Ahmad, Md. Mukhlesur Rahman, M. Khalid Jawed  

**Link**: [PDF](https://arxiv.org/pdf/2509.25056)  

**Abstract**: We present the AgriCruiser, an open-source over-the-row agricultural robot developed for low-cost deployment and rapid adaptation across diverse crops and row layouts. The chassis provides an adjustable track width of 1.42 m to 1.57 m, along with a ground clearance of 0.94 m. The AgriCruiser achieves compact pivot turns with radii of 0.71 m to 0.79 m, enabling efficient headland maneuvers. The platform is designed for the integration of the other subsystems, and in this study, a precision spraying system was implemented to assess its effectiveness in weed management. In twelve flax plots, a single robotic spray pass reduced total weed populations (pigweed and Venice mallow) by 24- to 42-fold compared to manual weeding in four flax plots, while also causing less crop damage. Mobility experiments conducted on concrete, asphalt, gravel, grass, and both wet and dry soil confirmed reliable traversal consistent with torque sizing. The complete chassis can be constructed from commodity T-slot extrusion with minimal machining, resulting in a bill of materials costing approximately $5,000 - $6,000, which enables replication and customization. The mentioned results demonstrate that low-cost, reconfigurable over-the-row robots can achieve effective weed management with reduced crop damage and labor requirements, while providing a versatile foundation for phenotyping, sensing, and other agriculture applications. Design files and implementation details are released to accelerate research and adoption of modular agricultural robotics. 

**Abstract (ZH)**: 基于行种植作物的开源低成本自动驾驶机器人：AgriCruiser及其杂草管理效果研究 

---
# AIRoA MoMa Dataset: A Large-Scale Hierarchical Dataset for Mobile Manipulation 

**Title (ZH)**: AIRoA MoMa 数据集：用于移动操作的大型层次化数据集 

**Authors**: Ryosuke Takanami, Petr Khrapchenkov, Shu Morikuni, Jumpei Arima, Yuta Takaba, Shunsuke Maeda, Takuya Okubo, Genki Sano, Satoshi Sekioka, Aoi Kadoya, Motonari Kambara, Naoya Nishiura, Haruto Suzuki, Takanori Yoshimoto, Koya Sakamoto, Shinnosuke Ono, Hu Yang, Daichi Yashima, Aoi Horo, Tomohiro Motoda, Kensuke Chiyoma, Hiroshi Ito, Koki Fukuda, Akihito Goto, Kazumi Morinaga, Yuya Ikeda, Riko Kawada, Masaki Yoshikawa, Norio Kosuge, Yuki Noguchi, Kei Ota, Tatsuya Matsushima, Yusuke Iwasawa, Yutaka Matsuo, Tetsuya Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2509.25032)  

**Abstract**: As robots transition from controlled settings to unstructured human environments, building generalist agents that can reliably follow natural language instructions remains a central challenge. Progress in robust mobile manipulation requires large-scale multimodal datasets that capture contact-rich and long-horizon tasks, yet existing resources lack synchronized force-torque sensing, hierarchical annotations, and explicit failure cases. We address this gap with the AIRoA MoMa Dataset, a large-scale real-world multimodal dataset for mobile manipulation. It includes synchronized RGB images, joint states, six-axis wrist force-torque signals, and internal robot states, together with a novel two-layer annotation schema of sub-goals and primitive actions for hierarchical learning and error analysis. The initial dataset comprises 25,469 episodes (approx. 94 hours) collected with the Human Support Robot (HSR) and is fully standardized in the LeRobot v2.1 format. By uniquely integrating mobile manipulation, contact-rich interaction, and long-horizon structure, AIRoA MoMa provides a critical benchmark for advancing the next generation of Vision-Language-Action models. The first version of our dataset is now available at this https URL . 

**Abstract (ZH)**: 随着机器人从受控环境过渡到未结构化的居住环境，构建能够可靠遵循自然语言指令的通才代理仍是主要挑战。为了提高稳健的移动操控进展，需要大规模多模态数据集来捕捉富含接触的长期任务，但现有资源缺乏同步的力-力矩感知、层次化注释和明确的失败案例。我们通过AIRoA MoMa数据集填补了这一空白，这是一个用于移动操控的大规模现实世界多模态数据集。该数据集包括同步的RGB图像、关节状态、六轴手腕力-力矩信号以及内部机器人状态，并提供了用于层次化学习和错误分析的新型两层注释方案。初始数据集包含25,469个时期（约94小时），使用Human Support Robot（HSR）收集，并完全符合LeRobot v2.1格式。通过唯一地整合移动操控、富含接触的交互以及长期结构，AIRoA MoMa为推动下一代视觉-语言-动作模型的发展提供了关键基准。我们的数据集第一版现已可用，网址为：this https URL。 

---
# Path Diffuser: Diffusion Model for Data-Driven Traffic Simulator 

**Title (ZH)**: 数据驱动交通模拟器中的路径扩散模型 

**Authors**: Da Saem Lee, Akash Karthikeyan, Yash Vardhan Pant, Sebastian Fischmeister  

**Link**: [PDF](https://arxiv.org/pdf/2509.24995)  

**Abstract**: Simulating diverse and realistic traffic scenarios is critical for developing and testing autonomous planning. Traditional rule-based planners lack diversity and realism, while learning-based simulators often replay, forecast, or edit scenarios using historical agent trajectories. However, they struggle to generate new scenarios, limiting scalability and diversity due to their reliance on fully annotated logs and historical data. Thus, a key challenge for a learning-based simulator's performance is that it requires agents' past trajectories and pose information in addition to map data, which might not be available for all agents on the this http URL which, generated scenarios often produce unrealistic trajectories that deviate from drivable areas, particularly under out-of-distribution (OOD) map scenes (e.g., curved roads). To address this, we propose Path Diffuser (PD): a two-stage, diffusion model for generating agent pose initializations and their corresponding trajectories conditioned on the map, free of any historical context of agents' trajectories. Furthermore, PD incorporates a motion primitive-based prior, leveraging Frenet frame candidate trajectories to enhance diversity while ensuring road-compliant trajectory generation. We also explore various design choices for modeling complex multi-agent interactions. We demonstrate the effectiveness of our method through extensive experiments on the Argoverse2 Dataset and additionally evaluate the generalizability of the approach on OOD map variants. Notably, Path Diffuser outperforms the baseline methods by 1.92x on distribution metrics, 1.14x on common-sense metrics, and 1.62x on road compliance from adversarial benchmarks. 

**Abstract (ZH)**: 基于路径扩散的多样化和现实交通场景模拟 

---
# Annotation-Free One-Shot Imitation Learning for Multi-Step Manipulation Tasks 

**Title (ZH)**: 无注释一次性模仿学习以应用于多步骤操作任务 

**Authors**: Vijja Wichitwechkarn, Emlyn Williams, Charles Fox, Ruchi Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2509.24972)  

**Abstract**: Recent advances in one-shot imitation learning have enabled robots to acquire new manipulation skills from a single human demonstration. While existing methods achieve strong performance on single-step tasks, they remain limited in their ability to handle long-horizon, multi-step tasks without additional model training or manual annotation. We propose a method that can be applied to this setting provided a single demonstration without additional model training or manual annotation. We evaluated our method on multi-step and single-step manipulation tasks where our method achieves an average success rate of 82.5% and 90%, respectively. Our method matches and exceeds the performance of the baselines in both these cases. We also compare the performance and computational efficiency of alternative pre-trained feature extractors within our framework. 

**Abstract (ZH)**: 近期单次演示模仿学习的进展使机器人能够从单次人类演示中获取新的操作技能。尽管现有方法在单步任务上表现出色，但在处理长时程多步任务时，仍需要额外的模型训练或手动注释。我们提出了一种方法，在无需额外模型训练或手动注释的情况下，可以从单次演示中应用于此类场景。我们在多步和单步操作任务上评估了该方法，分别实现了82.5%和90%的成功率。该方法在两种情况下均匹配并超过了基线方法的性能。我们还比较了不同预训练特征提取器在该框架内的性能和计算效率。 

---
# MSG: Multi-Stream Generative Policies for Sample-Efficient Robotic Manipulation 

**Title (ZH)**: MSG：多流生成策略在样本高效机器人操作中的应用 

**Authors**: Jan Ole von Hartz, Lukas Schweizer, Joschka Boedecker, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2509.24956)  

**Abstract**: Generative robot policies such as Flow Matching offer flexible, multi-modal policy learning but are sample-inefficient. Although object-centric policies improve sample efficiency, it does not resolve this limitation. In this work, we propose Multi-Stream Generative Policy (MSG), an inference-time composition framework that trains multiple object-centric policies and combines them at inference to improve generalization and sample efficiency. MSG is model-agnostic and inference-only, hence widely applicable to various generative policies and training paradigms. We perform extensive experiments both in simulation and on a real robot, demonstrating that our approach learns high-quality generative policies from as few as five demonstrations, resulting in a 95% reduction in demonstrations, and improves policy performance by 89 percent compared to single-stream approaches. Furthermore, we present comprehensive ablation studies on various composition strategies and provide practical recommendations for deployment. Finally, MSG enables zero-shot object instance transfer. We make our code publicly available at this https URL. 

**Abstract (ZH)**: 生成式机器人策略，如Flow Matching，提供了灵活的多模态策略学习，但样本效率低下。虽然以对象为中心的策略提高了样本效率，但这并未解决这一限制。在此工作中，我们提出了一种多流生成式策略（MSG），这是一种推理时的组合框架，训练多个以对象为中心的策略，并在推理时将它们组合起来，以提高泛化能力和样本效率。MSG是模型无关的，并且仅用于推理，因此适用于各种生成式策略和训练范式。我们在仿真和真实机器人上进行了广泛的实验，证明了我们的方法仅需五次演示即可学习高质量的生成式策略，从而将演示次数减少了95%，并将策略性能提高了89%，相比单流方法。此外，我们提供了各种组合策略的全面消融研究，并提供了部署的实用建议。最后，MSG支持零样本物体实例迁移。我们将在以下网址公开我们的代码：this https URL。 

---
# World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training 

**Title (ZH)**: World-Env: 利用世界模型作为虚拟环境进行多视图学习后训练 

**Authors**: Junjin Xiao, Yandan Yang, Xinyuan Chang, Ronghan Chen, Feng Xiong, Mu Xu, Wei-Shi Zheng, Qing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24948)  

**Abstract**: Vision-Language-Action (VLA) models trained via imitation learning suffer from significant performance degradation in data-scarce scenarios due to their reliance on large-scale demonstration datasets. Although reinforcement learning (RL)-based post-training has proven effective in addressing data scarcity, its application to VLA models is hindered by the non-resettable nature of real-world environments. This limitation is particularly critical in high-risk domains such as industrial automation, where interactions often induce state changes that are costly or infeasible to revert. Furthermore, existing VLA approaches lack a reliable mechanism for detecting task completion, leading to redundant actions that reduce overall task success rates. To address these challenges, we propose World-Env, an RL-based post-training framework that replaces physical interaction with a low-cost, world model-based virtual simulator. World-Env consists of two key components: (1) a video-based world simulator that generates temporally consistent future visual observations, and (2) a vision-language model (VLM)-guided instant reflector that provides continuous reward signals and predicts action termination. This simulated environment enables VLA models to safely explore and generalize beyond their initial imitation learning distribution. Our method achieves notable performance gains with as few as five expert demonstrations per task. Experiments on complex robotic manipulation tasks demonstrate that World-Env effectively overcomes the data inefficiency, safety constraints, and inefficient execution of conventional VLA models that rely on real-world interaction, offering a practical and scalable solution for post-training in resource-constrained settings. 

**Abstract (ZH)**: 基于世界模型的RL后训练框架World-Env：解决Vision-Language-Action模型在数据稀缺场景下的性能退化问题 

---
# Trajectory Prediction via Bayesian Intention Inference under Unknown Goals and Kinematics 

**Title (ZH)**: 基于未知目标和运动学的贝叶斯意图推理轨迹预测 

**Authors**: Shunan Yin, Zehui Lu, Shaoshuai Mou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24928)  

**Abstract**: This work introduces an adaptive Bayesian algorithm for real-time trajectory prediction via intention inference, where a target's intentions and motion characteristics are unknown and subject to change. The method concurrently estimates two critical variables: the target's current intention, modeled as a Markovian latent state, and an intention parameter that describes the target's adherence to a shortest-path policy. By integrating this joint update technique, the algorithm maintains robustness against abrupt intention shifts and unknown motion dynamics. A sampling-based trajectory prediction mechanism then exploits these adaptive estimates to generate probabilistic forecasts with quantified uncertainty. We validate the framework through numerical experiments: Ablation studies of two cases, and a 500-trial Monte Carlo analysis; Hardware demonstrations on quadrotor and quadrupedal platforms. Experimental results demonstrate that the proposed approach significantly outperforms non-adaptive and partially adaptive methods. The method operates in real time around 270 Hz without requiring training or detailed prior knowledge of target behavior, showcasing its applicability in various robotic systems. 

**Abstract (ZH)**: 基于意图推理的自适应贝叶斯实时轨迹预测算法 

---
# CineWild: Balancing Art and Robotics for Ethical Wildlife Documentary Filmmaking 

**Title (ZH)**: CineWild: 在伦理野生纪录片制作中平衡艺术与机器人技术 

**Authors**: Pablo Pueyo, Fernando Caballero, Ana Cristina Murillo, Eduardo Montijano  

**Link**: [PDF](https://arxiv.org/pdf/2509.24921)  

**Abstract**: Drones, or unmanned aerial vehicles (UAVs), have become powerful tools across domains-from industry to the arts. In documentary filmmaking, they offer dynamic, otherwise unreachable perspectives, transforming how stories are told. Wildlife documentaries especially benefit, yet drones also raise ethical concerns: the risk of disturbing the animals they aim to capture. This paper introduces CineWild, an autonomous UAV framework that combines robotics, cinematography, and ethics. Built on model predictive control, CineWild dynamically adjusts flight paths and camera settings to balance cinematic quality with animal welfare. Key features include adaptive zoom for filming from acoustic and visual safe distances, path-planning that avoids an animal's field of view, and smooth, low-noise maneuvers. CineWild exemplifies interdisciplinary innovation-bridging engineering, visual storytelling, and environmental ethics. We validate the system through simulation studies and will release the code upon acceptance. 

**Abstract (ZH)**: 无人机，或无人驾驶航空车辆(UAVs)，已在多个领域成为强大的工具，从工业到艺术。在纪录片制作中，它们提供了动态的、不可达及的新视角，改变了故事讲述的方式。野生动物纪录片尤其受益，但无人机也引发了伦理问题：捕获动物时对其的潜在干扰风险。本文介绍了CineWild，这是一种结合了机器人技术、电影制作和伦理学的自主无人机框架。基于模型预测控制，CineWild动态调整飞行路径和相机设置，以平衡电影品质与动物福利。关键功能包括根据声学和视觉安全距离进行的自适应变焦、避开动物视野的航线规划以及平稳、低噪音的操作。CineWild展示了跨学科创新，融合了工程学、视觉叙事和环境伦理学。通过仿真研究验证了系统，并将在录用后发布代码。 

---
# From Code to Action: Hierarchical Learning of Diffusion-VLM Policies 

**Title (ZH)**: 从代码到行动：扩散-VLM 策略的层级学习 

**Authors**: Markus Peschl, Pietro Mazzaglia, Daniel Dijkman  

**Link**: [PDF](https://arxiv.org/pdf/2509.24917)  

**Abstract**: Imitation learning for robotic manipulation often suffers from limited generalization and data scarcity, especially in complex, long-horizon tasks. In this work, we introduce a hierarchical framework that leverages code-generating vision-language models (VLMs) in combination with low-level diffusion policies to effectively imitate and generalize robotic behavior. Our key insight is to treat open-source robotic APIs not only as execution interfaces but also as sources of structured supervision: the associated subtask functions - when exposed - can serve as modular, semantically meaningful labels. We train a VLM to decompose task descriptions into executable subroutines, which are then grounded through a diffusion policy trained to imitate the corresponding robot behavior. To handle the non-Markovian nature of both code execution and certain real-world tasks, such as object swapping, our architecture incorporates a memory mechanism that maintains subtask context across time. We find that this design enables interpretable policy decomposition, improves generalization when compared to flat policies and enables separate evaluation of high-level planning and low-level control. 

**Abstract (ZH)**: 基于视觉语言模型的层次化模仿学习在机器人操作中的应用：处理复杂长时间任务的局限性和数据稀缺性 

---
# Real-time Recognition of Human Interactions from a Single RGB-D Camera for Socially-Aware Robot Navigation 

**Title (ZH)**: 基于单个RGB-D摄像头的实时人类交互识别技术及其在社会感知机器人导航中的应用 

**Authors**: Thanh Long Nguyen, Duc Phu Nguyen, Thanh Thao Ton Nu, Quan Le, Thuan Hoang Tran, Manh Duong Phung  

**Link**: [PDF](https://arxiv.org/pdf/2509.24907)  

**Abstract**: {Recognizing human interactions is essential for social robots as it enables them to navigate safely and naturally in shared environments. Conventional robotic systems however often focus on obstacle avoidance, neglecting social cues necessary for seamless human-robot interaction. To address this gap, we propose a framework to recognize human group interactions for socially aware navigation. Our method utilizes color and depth frames from a monocular RGB-D camera to estimate 3D human keypoints and positions. Principal component analysis (PCA) is then used to determine dominant interaction directions. The shoelace formula is finally applied to compute interest points and engagement areas. Extensive experiments have been conducted to evaluate the validity of the proposed method. The results show that our method is capable of recognizing group interactions across different scenarios with varying numbers of individuals. It also achieves high-speed performance, processing each frame in approximately 4 ms on a single-board computer used in robotic systems. The method is implemented as a ROS 2 package making it simple to integrate into existing navigation systems. Source code is available at this https URL 

**Abstract (ZH)**: 识别人类互动对于社会机器人至关重要，因为它使机器人能够在共享环境中安全自然地导航。传统的机器人系统通常侧重于避障，忽视了无缝人类-机器人交互所需的社会线索。为了解决这一问题，我们提出了一种框架来识别人类群体互动，以实现社会意识导航。该方法利用单目RGB-D相机的颜色和深度帧来估算3D人体关键点和位置。然后使用主成分分析（PCA）来确定主导的互动方向。最后，应用鞋带公式计算兴趣点和参与区域。进行了广泛的实验以评估所提出方法的有效性。结果表明，该方法能够识别不同场景下不同数量个体的群体互动，并且具有高效性能，在用于机器人系统的单板计算机上每帧处理时间约为4毫秒。该方法以ROS 2包的形式实现，易于集成到现有的导航系统中。源代码可通过此链接获取。 

---
# DRCP: Diffusion on Reinforced Cooperative Perception for Perceiving Beyond Limits 

**Title (ZH)**: DRCP: 扩展感知限界的强化协作感知扩散方法 

**Authors**: Lantao Li, Kang Yang, Rui Song, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.24903)  

**Abstract**: Cooperative perception enabled by Vehicle-to-Everything communication has shown great promise in enhancing situational awareness for autonomous vehicles and other mobile robotic platforms. Despite recent advances in perception backbones and multi-agent fusion, real-world deployments remain challenged by hard detection cases, exemplified by partial detections and noise accumulation which limit downstream detection accuracy. This work presents Diffusion on Reinforced Cooperative Perception (DRCP), a real-time deployable framework designed to address aforementioned issues in dynamic driving environments. DRCP integrates two key components: (1) Precise-Pyramid-Cross-Modality-Cross-Agent, a cross-modal cooperative perception module that leverages camera-intrinsic-aware angular partitioning for attention-based fusion and adaptive convolution to better exploit external features; and (2) Mask-Diffusion-Mask-Aggregation, a novel lightweight diffusion-based refinement module that encourages robustness against feature perturbations and aligns bird's-eye-view features closer to the task-optimal manifold. The proposed system achieves real-time performance on mobile platforms while significantly improving robustness under challenging conditions. Code will be released in late 2025. 

**Abstract (ZH)**: 基于Vehicle-to-Everything通信的协作感知在增强自主车辆和其他移动机器人平台的情境感知方面展现了巨大的潜力。尽管在感知骨干和多agent融合方面取得了近期进展，但由于部分检测和噪声累积等实际部署挑战，下游检测准确性仍受限。本文提出了一种名为Diffusion on Reinforced Cooperative Perception (DRCP)的实时可部署框架，旨在解决动态驾驶环境中的上述问题。DRCP结合了两个关键组件：(1) Precise-Pyramid-Cross-Modality-Cross-Agent，这是一种跨模态协作感知模块，利用相机固有角度分区进行基于注意力的融合和自适应卷积，以更好地利用外部特征；和(2) Mask-Diffusion-Mask-Aggregation，这是一种新颖的轻量级扩散基础精炼模块，鼓励对特征扰动的鲁棒性，并使鸟瞰视图特征更接近任务最优流形。所提出系统在移动平台上实现了实时性能，在恶劣条件下显著提高了鲁棒性。代码将于2025年底发布。 

---
# JuggleRL: Mastering Ball Juggling with a Quadrotor via Deep Reinforcement Learning 

**Title (ZH)**: JuggleRL：通过深度强化学习使四旋翼无人机掌握球技 

**Authors**: Shilong Ji, Yinuo Chen, Chuqi Wang, Jiayu Chen, Ruize Zhang, Feng Gao, Wenhao Tang, Shu'ang Yu, Sirui Xiang, Xinlei Chen, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24892)  

**Abstract**: Aerial robots interacting with objects must perform precise, contact-rich maneuvers under uncertainty. In this paper, we study the problem of aerial ball juggling using a quadrotor equipped with a racket, a task that demands accurate timing, stable control, and continuous adaptation. We propose JuggleRL, the first reinforcement learning-based system for aerial juggling. It learns closed-loop policies in large-scale simulation using systematic calibration of quadrotor and ball dynamics to reduce the sim-to-real gap. The training incorporates reward shaping to encourage racket-centered hits and sustained juggling, as well as domain randomization over ball position and coefficient of restitution to enhance robustness and transferability. The learned policy outputs mid-level commands executed by a low-level controller and is deployed zero-shot on real hardware, where an enhanced perception module with a lightweight communication protocol reduces delays in high-frequency state estimation and ensures real-time control. Experiments show that JuggleRL achieves an average of $311$ hits over $10$ consecutive trials in the real world, with a maximum of $462$ hits observed, far exceeding a model-based baseline that reaches at most $14$ hits with an average of $3.1$. Moreover, the policy generalizes to unseen conditions, successfully juggling a lighter $5$ g ball with an average of $145.9$ hits. This work demonstrates that reinforcement learning can empower aerial robots with robust and stable control in dynamic interaction tasks. 

**Abstract (ZH)**: 基于强化学习的飞行拍球机器人系统：在不确定性下的精准接触操作 

---
# Finding an Initial Probe Pose in Teleoperated Robotic Echocardiography via 2D LiDAR-Based 3D Reconstruction 

**Title (ZH)**: 基于2D LiDAR的3D重建在遥操作机器人心脏超声检查中寻找初始探头姿态 

**Authors**: Mariadas Capsran Roshan, Edgar M Hidalgo, Mats Isaksson, Michelle Dunn, Jagannatha Charjee Pyaraka  

**Link**: [PDF](https://arxiv.org/pdf/2509.24867)  

**Abstract**: Echocardiography is a key imaging modality for cardiac assessment but remains highly operator-dependent, and access to trained sonographers is limited in underserved settings. Teleoperated robotic echocardiography has been proposed as a solution; however, clinical studies report longer examination times than manual procedures, increasing diagnostic delays and operator workload. Automating non-expert tasks, such as automatically moving the probe to an ideal starting pose, offers a pathway to reduce this burden. Prior vision- and depth-based approaches to estimate an initial probe pose are sensitive to lighting, texture, and anatomical variability. We propose a robot-mounted 2D LiDAR-based approach that reconstructs the chest surface in 3D and estimates the initial probe pose automatically. To the best of our knowledge, this is the first demonstration of robot-mounted 2D LiDAR used for 3D reconstruction of a human body surface. Through plane-based extrinsic calibration, the transformation between the LiDAR and robot base frames was estimated with an overall root mean square (RMS) residual of 1.8 mm and rotational uncertainty below 0.2°. The chest front surface, reconstructed from two linear LiDAR sweeps, was aligned with non-rigid templates to identify an initial probe pose. A mannequin-based study assessing reconstruction accuracy showed mean surface errors of 2.78 +/- 0.21 mm. Human trials (N=5) evaluating the proposed approach found probe initial points typically 20-30 mm from the clinically defined initial point, while the variation across repeated trials on the same subject was less than 4 mm. 

**Abstract (ZH)**: 基于2D LiDAR的机器人辅助心脏超声成像初步研究 

---
# Towards Modular and Accessible AUV Systems 

**Title (ZH)**: 面向模块化和无障碍AUV系统的研究 

**Authors**: Mingxi Zhou, Farhang Naderi, Yuewei Fu, Tony Jacob, Lin Zhao, Manavi Panjnani, Chengzhi Yuan, William McConnell, Emir Cem Gezer  

**Link**: [PDF](https://arxiv.org/pdf/2509.24864)  

**Abstract**: This paper reports the development of a new open- access modular framework, called Marine Vehicle Packages (MVP), for Autonomous Underwater Vehicles. The framework consists of both software and hardware designs allowing easy construction of AUV for research with increased customizability and sufficient payload capacity. This paper will present the scalable hardware system design and the modular software design architecture. New features, such as articulated thruster integra- tion and high-level Graphic User Interface will be discussed. Both simulation and field experiments results are shown to highlight the performance and compatibility of the MVP. 

**Abstract (ZH)**: 本论文报道了一种新的开源模块化框架Marine Vehicle Packages (MVP) 的开发，该框架用于自主水下 vehicle。该框架包括软件和硬件设计，便于进行研究用途的自主水下 vehicle (AUV) 的轻松构建，且具有高度的定制化能力和充足的载荷能力。本文将介绍可扩展的硬件系统设计和模块化的软件设计架构。还将讨论新功能，例如可articulated 推力器集成和高级图形用户界面。通过仿真和现场实验结果来突出MVP的性能和兼容性。 

---
# Fidelity-Aware Data Composition for Robust Robot Generalization 

**Title (ZH)**: Awareness-fidelity 数据集成以实现鲁棒的机器人泛化 

**Authors**: Zizhao Tong, Di Chen, Sicheng Hu, Hongwei Fan, Liliang Chen, Guanghui Ren, Hao Tang, Hao Dong, Ling Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24797)  

**Abstract**: Generalist robot policies trained on large-scale, visually homogeneous datasets can be susceptible to shortcut learning, which impairs their out-of-distribution (OOD) generalization. While generative data augmentation is a common approach to introduce diversity, it presents a subtle challenge: data composition. Naively mixing real and synthetic data can corrupt the learning signal, as this process often prioritizes visual diversity at the expense of information fidelity. This paper suggests that robust generalization depends on principled, fidelity-aware data composition. We introduce Coherent Information Fidelity Tuning (CIFT), a framework that treats data composition as an optimization problem. CIFT uses a practical proxy for Information Fidelity based on the feature-space geometry of a dataset. This enables the identification of a phase transition, termed the Decoherence Point, where training stability degrades. The framework includes a generative engine, Multi-View Video Augmentation (MVAug), to synthesize a causally disentangled data spectrum for this tuning process. Applying CIFT to policy architectures such as $\pi_0$ and Diffusion Policy improves OOD success rates by over 54\%. These results indicate that fidelity-aware composition, beyond data synthesis alone, is an important component for developing robust, general-purpose robots. 

**Abstract (ZH)**: 通用型机器人政策在大规模视觉同质数据集上训练后，可能容易陷入捷径学习，这会削弱其领域外（OOD）泛化能力。虽然生成数据增强是一种常见的增加多样性方法，但它提出了一个微妙的挑战：数据合成。简单地混合真实和合成数据会破坏学习信号，因为这个过程往往优先考虑视觉多样性，而牺牲了信息保真度。本文建议，稳健的泛化依赖于具有原理性和信息保真意识的数据合成。我们引入了一致信息保真度调谐（CIFT），这是一个将数据合成视为优化问题的框架。CIFT 使用数据集特征空间几何结构的实用代理衡量信息保真度，这使得能够识别一个相变点，称为去相干点，在此点训练稳定性下降。该框架包括一个生成引擎，多视图视频增强（MVAug），以合成分裂因果数据光谱，用于这一调谐过程。将CIFT应用于如$\pi_0$和扩散政策架构时，能够在领域外成功率上提高超过54%。这些结果表明，信息保真意识的合成，而不仅仅是数据合成，是开发稳健、通用机器人的重要组成部分。 

---
# IA-VLA: Input Augmentation for Vision-Language-Action models in settings with semantically complex tasks 

**Title (ZH)**: IA-VLA: 输入增强在语义复杂任务设置下视觉-语言-行动模型中的应用 

**Authors**: Eric Hannus, Miika Malin, Tran Nguyen Le, Ville Kyrki  

**Link**: [PDF](https://arxiv.org/pdf/2509.24768)  

**Abstract**: Vision-language-action models (VLAs) have become an increasingly popular approach for addressing robot manipulation problems in recent years. However, such models need to output actions at a rate suitable for robot control, which limits the size of the language model they can be based on, and consequently, their language understanding capabilities. Manipulation tasks may require complex language instructions, such as identifying target objects by their relative positions, to specify human intention. Therefore, we introduce IA-VLA, a framework that utilizes the extensive language understanding of a large vision language model as a pre-processing stage to generate improved context to augment the input of a VLA. We evaluate the framework on a set of semantically complex tasks which have been underexplored in VLA literature, namely tasks involving visual duplicates, i.e., visually indistinguishable objects. A dataset of three types of scenes with duplicate objects is used to compare a baseline VLA against two augmented variants. The experiments show that the VLA benefits from the augmentation scheme, especially when faced with language instructions that require the VLA to extrapolate from concepts it has seen in the demonstrations. For the code, dataset, and videos, see this https URL. 

**Abstract (ZH)**: 基于视觉-语言-行动模型（VLAs）的框架：利用大规模视觉语言模型增强语境以应对视觉重复对象的复杂任务 

---
# SSR-ZSON: Zero-Shot Object Navigation via Spatial-Semantic Relations within a Hierarchical Exploration Framework 

**Title (ZH)**: SSR-ZSON: 基于层次探索框架内的空间语义关系的零样本对象导航 

**Authors**: Xiangyi Meng, Delun Li, Zihao Mao, Yi Yang, Wenjie Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.24763)  

**Abstract**: Zero-shot object navigation in unknown environments presents significant challenges, mainly due to two key limitations: insufficient semantic guidance leads to inefficient exploration, while limited spatial memory resulting from environmental structure causes entrapment in local regions. To address these issues, we propose SSR-ZSON, a spatial-semantic relative zero-shot object navigation method based on the TARE hierarchical exploration framework, integrating a viewpoint generation strategy balancing spatial coverage and semantic density with an LLM-based global guidance mechanism. The performance improvement of the proposed method is due to two key innovations. First, the viewpoint generation strategy prioritizes areas of high semantic density within traversable sub-regions to maximize spatial coverage and minimize invalid exploration. Second, coupled with an LLM-based global guidance mechanism, it assesses semantic associations to direct navigation toward high-value spaces, preventing local entrapment and ensuring efficient exploration. Deployed on hybrid Habitat-Gazebo simulations and physical platforms, SSR-ZSON achieves real-time operation and superior performance. On Matterport3D and Habitat-Matterport3D datasets, it improves the Success Rate(SR) by 18.5\% and 11.2\%, and the Success weighted by Path Length(SPL) by 0.181 and 0.140, respectively, over state-of-the-art methods. 

**Abstract (ZH)**: 未知环境下零样本物体导航面临显著挑战，主要是由于两个关键限制：语义指导不足导致探索效率低下，而有限的空间记忆导致在局部区域陷入。为了解决这些问题，我们提出了一种基于TARE层次探索框架的SSR-ZSON空间语义相对零样本物体导航方法，该方法结合了平衡空间覆盖度和语义密度的视角生成策略和基于大语言模型的全局指导机制。所提出的该方法的性能提升归功于两个关键创新。首先，视角生成策略优先考虑可通行子区域内的高语义密度区域，以最大化空间覆盖度并最小化无效探索。其次，结合基于大语言模型的全局指导机制，通过评估语义关联性来引导导航至高价值空间，防止局部陷入并确保高效探索。在混合Habitat-Gazebo仿真和物理平台上部署，SSR-ZSON实现实时操作并表现出优越性能。在Matterport3D和Habitat-Matterport3D数据集上，SSR-ZSON分别将成功率（Success Rate，SR）提高了18.5%和11.2%，平均成功率加权路径长度（Success weighted by Path Length，SPL）提高了0.181和0.140，超过最先进的方法。 

---
# APREBot: Active Perception System for Reflexive Evasion Robot 

**Title (ZH)**: APREBot: 反应式避障机器人主动感知系统 

**Authors**: Zihao Xu, Kuankuan Sima, Junhao Deng, Zixuan Zhuang, Chunzheng Wang, Ce Hao, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24733)  

**Abstract**: Reliable onboard perception is critical for quadruped robots navigating dynamic environments, where obstacles can emerge from any direction under strict reaction-time constraints. Single-sensor systems face inherent limitations: LiDAR provides omnidirectional coverage but lacks rich texture information, while cameras capture high-resolution detail but suffer from restricted field of view. We introduce APREBot (Active Perception System for Reflexive Evasion Robot), a novel framework that integrates reflexive evasion with active hierarchical perception. APREBot strategically combines LiDAR-based omnidirectional scanning with camera-based active focusing, achieving comprehensive environmental awareness essential for agile obstacle avoidance in quadruped robots. We validate APREBot through extensive sim-to-real experiments on a quadruped platform, evaluating diverse obstacle types, trajectories, and approach directions. Our results demonstrate substantial improvements over state-of-the-art baselines in both safety metrics and operational efficiency, highlighting APREBot's potential for dependable autonomy in safety-critical scenarios. Videos are available at this https URL 

**Abstract (ZH)**: 可靠的机载感知对于四足机器人在动态环境中的导航至关重要，严格的时间限制约束下，障碍物可以从任何方向出现。单传感器系统存在固有限制：LiDAR提供全方位覆盖但缺乏丰富的纹理信息，而相机能够捕捉高分辨率的细节但视野受限。我们介绍了APREBot（具有反射性规避的主动感知系统），这是一种将反射性规避与主动分层感知相结合的创新框架。APREBot战略性地结合了基于LiDAR的全方位扫描与基于相机的主动聚焦，实现了四足机器人灵活避障所需的全面环境意识。我们通过在四足平台上进行广泛的仿真实验验证了APREBot，评估了不同的障碍类型、路径和接近方向。我们的结果表明，APREBot在安全指标和操作效率方面显著优于最先进的 baseline，突显了其在安全关键场景中可靠自主性的潜力。视频可在以下链接获取：这个 https URL。 

---
# LLM-Handover:Exploiting LLMs for Task-Oriented Robot-Human Handovers 

**Title (ZH)**: LLM-手递：利用大语言模型进行任务导向的机器人-人类手递 

**Authors**: Andreea Tulbure, Rene Zurbruegg, Timm Grigat, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2509.24706)  

**Abstract**: Effective human-robot collaboration depends on task-oriented handovers, where robots present objects in ways that support the partners intended use. However, many existing approaches neglect the humans post-handover action, relying on assumptions that limit generalizability. To address this gap, we propose LLM-Handover, a novel framework that integrates large language model (LLM)-based reasoning with part segmentation to enable context-aware grasp selection and execution. Given an RGB-D image and a task description, our system infers relevant object parts and selects grasps that optimize post-handover usability. To support evaluation, we introduce a new dataset of 60 household objects spanning 12 categories, each annotated with detailed part labels. We first demonstrate that our approach improves the performance of the used state-of-the-art part segmentation method, in the context of robot-human handovers. Next, we show that LLM-Handover achieves higher grasp success rates and adapts better to post-handover task constraints. During hardware experiments, we achieve a success rate of 83% in a zero-shot setting over conventional and unconventional post-handover tasks. Finally, our user study underlines that our method enables more intuitive, context-aware handovers, with participants preferring it in 86% of cases. 

**Abstract (ZH)**: 有效的机器人-人类协作依赖于任务导向的手递过程，其中机器人以支持合作伙伴预期使用的方式呈现物体。然而，许多现有方法忽视了人类手递后的动作，依赖于限制泛化的假设。为了解决这一问题，我们提出了LLM-Handover这一新型框架，该框架结合了基于大规模语言模型（LLM）的推理与部分分割技术，以实现上下文感知的抓取选择与执行。给定一个RGB-D图像和任务描述，我们的系统推断相关物体部分并选择最大化手递后易用性的抓取。为支持评估，我们引入了一个包含60件家庭用品的新数据集，这些用品分为12个类别，每个类别都详细标注了部分标签。我们首先证明，我们的方法提高了所使用的最新部分分割方法在机器人-人类手递中的性能。接着，我们展示了LLM-Handover实现了更高的抓取成功率并且更好地适应手递后的任务约束。在硬件实验中，在零样本设置下，对于常规和非常规手递后任务，我们实现了83%的成功率。最后，我们的用户研究强调，我们的方法能够实现更直观、上下文感知的手递，86%的参与者更偏好我们的方法。 

---
# Stabilizing Humanoid Robot Trajectory Generation via Physics-Informed Learning and Control-Informed Steering 

**Title (ZH)**: 通过物理告知学习和控制导向导向控制实现类人机器人轨迹生成的稳定性增强 

**Authors**: Evelyn D'Elia, Paolo Maria Viceconte, Lorenzo Rapetti, Diego Ferigo, Giulio Romualdi, Giuseppe L'Erario, Raffaello Camoriano, Daniele Pucci  

**Link**: [PDF](https://arxiv.org/pdf/2509.24697)  

**Abstract**: Recent trends in humanoid robot control have successfully employed imitation learning to enable the learned generation of smooth, human-like trajectories from human data. While these approaches make more realistic motions possible, they are limited by the amount of available motion data, and do not incorporate prior knowledge about the physical laws governing the system and its interactions with the environment. Thus they may violate such laws, leading to divergent trajectories and sliding contacts which limit real-world stability. We address such limitations via a two-pronged learning strategy which leverages the known physics of the system and fundamental control principles. First, we encode physics priors during supervised imitation learning to promote trajectory feasibility. Second, we minimize drift at inference time by applying a proportional-integral controller directly to the generated output state. We validate our method on various locomotion behaviors for the ergoCub humanoid robot, where a physics-informed loss encourages zero contact foot velocity. Our experiments demonstrate that the proposed approach is compatible with multiple controllers on a real robot and significantly improves the accuracy and physical constraint conformity of generated trajectories. 

**Abstract (ZH)**: Recent trends in humanoid robot control have successfully employed imitation learning to enable the learned generation of smooth, human-like trajectories from human data. While these approaches make more realistic motions possible, they are limited by the amount of available motion data, and do not incorporate prior knowledge about the physical laws governing the system and its interactions with the environment. Thus they may violate such laws, leading to divergent trajectories and sliding contacts which limit real-world stability. We address such limitations via a two-pronged learning strategy which leverages the known physics of the system and fundamental control principles. First, we encode physics priors during supervised imitation learning to promote trajectory feasibility. Second, we minimize drift at inference time by applying a proportional-integral controller directly to the generated output state. We validate our method on various locomotion behaviors for the ergoCub humanoid robot, where a physics-informed loss encourages zero contact foot velocity. Our experiments demonstrate that the proposed approach is compatible with multiple controllers on a real robot and significantly improves the accuracy and physical constraint conformity of generated trajectories. 

---
# CEDex: Cross-Embodiment Dexterous Grasp Generation at Scale from Human-like Contact Representations 

**Title (ZH)**: CEDex: 大规模从类人接触表示生成跨越载体的灵巧抓取 

**Authors**: Zhiyuan Wu, Rolandos Alexandros Potamias, Xuyang Zhang, Zhongqun Zhang, Jiankang Deng, Shan Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24661)  

**Abstract**: Cross-embodiment dexterous grasp synthesis refers to adaptively generating and optimizing grasps for various robotic hands with different morphologies. This capability is crucial for achieving versatile robotic manipulation in diverse environments and requires substantial amounts of reliable and diverse grasp data for effective model training and robust generalization. However, existing approaches either rely on physics-based optimization that lacks human-like kinematic understanding or require extensive manual data collection processes that are limited to anthropomorphic structures. In this paper, we propose CEDex, a novel cross-embodiment dexterous grasp synthesis method at scale that bridges human grasping kinematics and robot kinematics by aligning robot kinematic models with generated human-like contact representations. Given an object's point cloud and an arbitrary robotic hand model, CEDex first generates human-like contact representations using a Conditional Variational Auto-encoder pretrained on human contact data. It then performs kinematic human contact alignment through topological merging to consolidate multiple human hand parts into unified robot components, followed by a signed distance field-based grasp optimization with physics-aware constraints. Using CEDex, we construct the largest cross-embodiment grasp dataset to date, comprising 500K objects across four gripper types with 20M total grasps. Extensive experiments show that CEDex outperforms state-of-the-art approaches and our dataset benefits cross-embodiment grasp learning with high-quality diverse grasps. 

**Abstract (ZH)**: 跨身躯 Dexterous 抓取合成 

---
# PoseDiff: A Unified Diffusion Model Bridging Robot Pose Estimation and Video-to-Action Control 

**Title (ZH)**: PoseDiff: 一个统一的扩散模型，连接机器人姿态估计与视频到动作控制 

**Authors**: Haozhuo Zhang, Michele Caprio, Jing Shao, Qiang Zhang, Jian Tang, Shanghang Zhang, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24591)  

**Abstract**: We present PoseDiff, a conditional diffusion model that unifies robot state estimation and control within a single framework. At its core, PoseDiff maps raw visual observations into structured robot states-such as 3D keypoints or joint angles-from a single RGB image, eliminating the need for multi-stage pipelines or auxiliary modalities. Building upon this foundation, PoseDiff extends naturally to video-to-action inverse dynamics: by conditioning on sparse video keyframes generated by world models, it produces smooth and continuous long-horizon action sequences through an overlap-averaging strategy. This unified design enables scalable and efficient integration of perception and control. On the DREAM dataset, PoseDiff achieves state-of-the-art accuracy and real-time performance for pose estimation. On Libero-Object manipulation tasks, it substantially improves success rates over existing inverse dynamics modules, even under strict offline settings. Together, these results show that PoseDiff provides a scalable, accurate, and efficient bridge between perception, planning, and control in embodied AI. The video visualization results can be found on the project page: this https URL. 

**Abstract (ZH)**: PoseDiff：统一机器人状态估计与控制的条件扩散模型 

---
# U-DiT Policy: U-shaped Diffusion Transformers for Robotic Manipulation 

**Title (ZH)**: U-DiT策略：U形扩散变换器在机器人操作中的应用 

**Authors**: Linzhi Wu, Aoran Mei, Xiyue Wang, Guo-Niu Zhu, Zhongxue Gan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24579)  

**Abstract**: Diffusion-based methods have been acknowledged as a powerful paradigm for end-to-end visuomotor control in robotics. Most existing approaches adopt a Diffusion Policy in U-Net architecture (DP-U), which, while effective, suffers from limited global context modeling and over-smoothing artifacts. To address these issues, we propose U-DiT Policy, a novel U-shaped Diffusion Transformer framework. U-DiT preserves the multi-scale feature fusion advantages of U-Net while integrating the global context modeling capability of Transformers, thereby enhancing representational power and policy expressiveness. We evaluate U-DiT extensively across both simulation and real-world robotic manipulation tasks. In simulation, U-DiT achieves an average performance gain of 10\% over baseline methods and surpasses Transformer-based diffusion policies (DP-T) that use AdaLN blocks by 6\% under comparable parameter budgets. On real-world robotic tasks, U-DiT demonstrates superior generalization and robustness, achieving an average improvement of 22.5\% over DP-U. In addition, robustness and generalization experiments under distractor and lighting variations further highlight the advantages of U-DiT. These results highlight the effectiveness and practical potential of U-DiT Policy as a new foundation for diffusion-based robotic manipulation. 

**Abstract (ZH)**: 基于扩散的方法已被公认为机器人端到端视觉-运动控制的一种强大范式。现有的大多数方法采用了U-Net架构的扩散策略（DP-U），尽管有效，但存在全局上下文建模能力有限和过度平滑的缺点。为了解决这些问题，我们提出了一种新的U形扩散变压器框架U-DiT策略。U-DiT保留了U-Net的多尺度特征融合优势，同时整合了Transformer的全局上下文建模能力，从而增强了表示能力和策略表达能力。我们在模拟和实际机器人操作任务中广泛评估了U-DiT。在模拟环境中，U-DiT在基线方法上实现了平均10%的性能提升，并在与基于Transformer的扩散策略（DP-T）使用AdaLN模块的情况下，拥有相似的参数预算时，超越了6%。在实际机器人任务中，U-DiT展示了更好的泛化能力和鲁棒性，相对于DP-U实现了平均22.5%的改进。此外，在干扰和照明变化的鲁棒性和泛化实验中，进一步突显了U-DiT的优势。这些结果表明，U-DiT策略作为一种新的基于扩散的机器人操作基础框架具有有效性和实际潜力。 

---
# Prompting Robot Teams with Natural Language 

**Title (ZH)**: 用自然语言指令驱动机器人团队 

**Authors**: Nicolas Pfitzer, Eduardo Sebastián, Ajay Shankar, Amanda Prorok  

**Link**: [PDF](https://arxiv.org/pdf/2509.24575)  

**Abstract**: This paper presents a framework towards prompting multi-robot teams with high-level tasks using natural language expressions. Our objective is to use the reasoning capabilities demonstrated by recent language models in understanding and decomposing human expressions of intent, and repurpose these for multi-robot collaboration and decision-making. The key challenge is that an individual's behavior in a collective can be hard to specify and interpret, and must continuously adapt to actions from others. This necessitates a framework that possesses the representational capacity required by the logic and semantics of a task, and yet supports decentralized and interactive real-time operation. We solve this dilemma by recognizing that a task can be represented as a deterministic finite automaton (DFA), and that recurrent neural networks (RNNs) can encode numerous automata. This allows us to distill the logic and sequential decompositions of sub-tasks obtained from a language model into an RNN, and align its internal states with the semantics of a given task. By training a graph neural network (GNN) control policy that is conditioned on the hidden states of the RNN and the language embeddings, our method enables robots to execute task-relevant actions in a decentralized manner. We present evaluations of this single light-weight interpretable model on various simulated and real-world multi-robot tasks that require sequential and collaborative behavior by the team -- this http URL. 

**Abstract (ZH)**: 本文提出了一种使用自然语言表达提示多机器人团队执行高层任务的框架。我们的目标是利用最近的语言模型在理解和分解人类意图表达方面的推理能力，并将这些能力应用于多机器人协作和决策。关键挑战在于，在集体中个体的行为难以具体化和解释，并且必须不断适应他人的行动。这就需要一个具备任务所需的逻辑和语义表示能力的框架，并且支持分布式和交互式的实时操作。我们通过认识到任务可以表示为确定性有限自动机（DFA），并且循环神经网络（RNN）可以编码多种自动机来解决这一矛盾。这使我们能够从语言模型中提炼出子任务的逻辑和序列分解，并将其输入到RNN中，使其内部状态与给定任务的语义相吻合。通过训练一个基于RNN隐藏状态和语言嵌入的图神经网络（GNN）控制策略，我们的方法使得机器人能够以分布式的方式执行与任务相关的行为。我们在各种需要序列化和协作行为的模拟和真实世界多机器人任务上评估了这一单一轻量级可解释模型——请访问这个网址。 

---
# Unlocking the Potential of Soft Actor-Critic for Imitation Learning 

**Title (ZH)**: 解锁Soft Actor-Critic在 imitation learning 中的潜力 

**Authors**: Nayari Marie Lessa, Melya Boukheddimi, Frank Kirchner  

**Link**: [PDF](https://arxiv.org/pdf/2509.24539)  

**Abstract**: Learning-based methods have enabled robots to acquire bio-inspired movements with increasing levels of naturalness and adaptability. Among these, Imitation Learning (IL) has proven effective in transferring complex motion patterns from animals to robotic systems. However, current state-of-the-art frameworks predominantly rely on Proximal Policy Optimization (PPO), an on-policy algorithm that prioritizes stability over sample efficiency and policy generalization. This paper proposes a novel IL framework that combines Adversarial Motion Priors (AMP) with the off-policy Soft Actor-Critic (SAC) algorithm to overcome these limitations. This integration leverages replay-driven learning and entropy-regularized exploration, enabling naturalistic behavior and task execution, improving data efficiency and robustness. We evaluate the proposed approach (AMP+SAC) on quadruped gaits involving multiple reference motions and diverse terrains. Experimental results demonstrate that the proposed framework not only maintains stable task execution but also achieves higher imitation rewards compared to the widely used AMP+PPO method. These findings highlight the potential of an off-policy IL formulation for advancing motion generation in robotics. 

**Abstract (ZH)**: 基于学习的方法使机器人能够获得越来越自然和适应性强的生物启发运动。在这之中， imitation learning (IL) 已证明有效于将复杂的运动模式从动物转移至机器人系统。然而，当前最先进的框架主要依赖于优先稳定性和样本效率的近端策略优化(PPO)算法。本文提出了一种新型的IL框架，该框架结合了对抗运动先验(AMP)与离策 Soft Actor-Critic (SAC) 算法，以克服这些局限性。这种集成利用了回放驱动的学习和熵正则化探索，从而实现自然的行为和任务执行，提高数据效率和鲁棒性。我们在涉及多种参考运动和不同地形的四足运动中评估了该方法(AMP+SAC)。实验结果表明，所提出的框架不仅能够保持稳定的任务执行，还能比广泛使用的AMP+PPO方法获得更高的仿真正奖。这些发现强调了离策IL方案在推进机器人动作生成方面的潜力。 

---
# Game Theory to Study Cooperation in Human-Robot Mixed Groups: Exploring the Potential of the Public Good Game 

**Title (ZH)**: 基于博弈论研究人类与机器人混合组中的合作：探索公共物品博弈的潜在价值 

**Authors**: Giulia Pusceddu, Sara Mongile, Francesco Rea, Alessandra Sciutti  

**Link**: [PDF](https://arxiv.org/pdf/2509.24530)  

**Abstract**: In this study, we explore the potential of Game Theory as a means to investigate cooperation and trust in human-robot mixed groups. Particularly, we introduce the Public Good Game (PGG), a model highlighting the tension between individual self-interest and collective well-being. In this work, we present a modified version of the PGG, where three human participants engage in the game with the humanoid robot iCub to assess whether various robot game strategies (e.g., always cooperate, always free ride, and tit-for-tat) can influence the participants' inclination to cooperate. We test our setup during a pilot study with nineteen participants. A preliminary analysis indicates that participants prefer not to invest their money in the common pool, despite they perceive the robot as generous. By conducting this research, we seek to gain valuable insights into the role that robots can play in promoting trust and cohesion during human-robot interactions within group contexts. The results of this study may hold considerable potential for developing social robots capable of fostering trust and cooperation within mixed human-robot groups. 

**Abstract (ZH)**: 本研究探讨博弈理论作为研究人类-机器人混合群体中合作与信任的手段的潜在可能性。具体而言，我们引入了公共物品游戏（PGG）模型，该模型突显了个体自我利益与集体福祉之间的张力。在本文中，我们提出了公共物品游戏的一种修改版本，三人参与者与类人机器人iCub进行游戏，以评估不同类型机器人游戏策略（例如始终合作、始终搭便车和以牙还牙）是否会影响参与者合作的倾向。我们在包含十九名参与者的试点研究中测试了我们的设置。初步分析表明，尽管参与者认为机器人很慷慨，但他们还是倾向于不将钱投入共享池。通过进行这项研究，我们旨在深入了解机器人在促进人类-机器人互动中群体内的信任与凝聚力方面的作用。本研究的结果可能对于开发能够促进人类-机器人混合群体中信任与合作的社会机器人具有重要意义。 

---
# PhysiAgent: An Embodied Agent Framework in Physical World 

**Title (ZH)**: 体感智能体：物质世界中的嵌入代理框架 

**Authors**: Zhihao Wang, Jianxiong Li, Jinliang Zheng, Wencong Zhang, Dongxiu Liu, Yinan Zheng, Haoyi Niu, Junzhi Yu, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24524)  

**Abstract**: Vision-Language-Action (VLA) models have achieved notable success but often struggle with limited generalizations. To address this, integrating generalized Vision-Language Models (VLMs) as assistants to VLAs has emerged as a popular solution. However, current approaches often combine these models in rigid, sequential structures: using VLMs primarily for high-level scene understanding and task planning, and VLAs merely as executors of lower-level actions, leading to ineffective collaboration and poor grounding challenges. In this paper, we propose an embodied agent framework, PhysiAgent, tailored to operate effectively in physical environments. By incorporating monitor, memory, self-reflection mechanisms, and lightweight off-the-shelf toolboxes, PhysiAgent offers an autonomous scaffolding framework to prompt VLMs to organize different components based on real-time proficiency feedback from VLAs to maximally exploit VLAs' capabilities. Experimental results demonstrate significant improvements in task-solving performance on complex real-world robotic tasks, showcasing effective self-regulation of VLMs, coherent tool collaboration, and adaptive evolution of the framework during execution. PhysiAgent makes practical and pioneering efforts to integrate VLMs and VLAs, effectively grounding embodied agent frameworks in real-world settings. 

**Abstract (ZH)**: 基于视觉-语言-动作模型的实体代理框架：实现实体代理在物理环境中的有效操作 

---
# DynaMIC: Dynamic Multimodal In-Context Learning Enabled Embodied Robot Counterfactual Resistance Ability 

**Title (ZH)**: DynaMIC: 动态多模态上下文学习赋能的 embodied 机器人反事实抵抗力 

**Authors**: Tianqiang Yan, Ziqiao Lin, Sicheng Wang, Tianwei Zhang, Zhenglong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.24413)  

**Abstract**: The emergence of large pre-trained models based on natural language has breathed new life into robotics development. Extensive research has integrated large models with robots, utilizing the powerful semantic understanding and generation capabilities of large models to facilitate robot control through natural language instructions gradually. However, we found that robots that strictly adhere to human instructions, especially those containing misleading information, may encounter errors during task execution, potentially leading to safety hazards. This resembles the concept of counterfactuals in natural language processing (NLP), which has not yet attracted much attention in robotic research. In an effort to highlight this issue for future studies, this paper introduced directive counterfactuals (DCFs) arising from misleading human directives. We present DynaMIC, a framework for generating robot task flows to identify DCFs and relay feedback to humans proactively. This capability can help robots be sensitive to potential DCFs within a task, thus enhancing the reliability of the execution process. We conducted semantic-level experiments and ablation studies, showcasing the effectiveness of this framework. 

**Abstract (ZH)**: 基于自然语言的大型预训练模型的出现为机器人研发注入了新的活力。大量研究将大型模型与机器人结合，利用其强大的语义理解和生成能力，通过自然语言指令逐步实现机器人控制。然而，我们发现严格遵循人类指令，尤其是包含误导信息的指令的机器人，在任务执行过程中可能会遇到错误，从而可能带来安全风险。这类似于自然语言处理（NLP）中反事实概念，但在机器人研究中尚未引起广泛关注。为了在未来的研究中突出这一问题，本文介绍了一种源自误导性人类指令的指令反事实(DCFs)。我们提出了DynaMIC框架，用于生成机器人任务流程以识别DCF并主动向人类传达反馈。这一能力可以使机器人对任务中潜在的DCF更加敏感，从而提高执行过程的可靠性。我们进行了语义层面的实验和消融研究，展示了该框架的有效性。 

---
# AdaNav: Adaptive Reasoning with Uncertainty for Vision-Language Navigation 

**Title (ZH)**: AdaNav: 带有不确定性自适应推理的视觉-语言导航 

**Authors**: Xin Ding, Jianyu Wei, Yifan Yang, Shiqi Jiang, Qianxi Zhang, Hao Wu, Fucheng Jia, Liang Mi, Yuxuan Yan, Weijun Wang, Yunxin Liu, Zhibo Chen, Ting Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24387)  

**Abstract**: Vision Language Navigation (VLN) requires agents to follow natural language instructions by grounding them in sequential visual observations over long horizons. Explicit reasoning could enhance temporal consistency and perception action alignment, but reasoning at fixed steps often leads to suboptimal performance and unnecessary computation. To address this, we propose AdaNav, an uncertainty-based adaptive reasoning framework for VLN. At its core is the Uncertainty Adaptive Reasoning Block (UAR), a lightweight plugin that dynamically triggers reasoning. We introduce Action Entropy as a policy prior for UAR and progressively refine it through a Heuristics to RL training method, enabling agents to learn difficulty aware reasoning policies under the strict data limitations of embodied tasks. Results show that with only 6K training samples, AdaNav achieves substantial gains over closed source models trained on million scale data, improving success rate by 20% on R2R val-unseen, 11.7% on RxR-CE, and 11.4% in real world scenes. The code is available at this https URL. 

**Abstract (ZH)**: 基于视觉语言的导航（Vision Language Navigation，VLN）要求代理通过将自然语言指令 grounding 在长时序的视觉观察中来遵循这些指令。基于不确定性的自适应推理框架（AdaNav）可以通过动态触发推理来增强时间一致性和感知动作对齐，但固定步长的推理经常导致性能不佳和不必要的计算。为此，我们提出了一种基于不确定性的自适应推理框架（AdaNav）以增强视觉语言导航（VLN）。其核心是不确定性自适应推理块（UAR），这是一种轻量级插件，可以动态触发推理。我们引入了动作熵作为UAR的策略先验，并通过启发式到强化学习的训练方法逐步对其进行细化，使代理在有严格数据限制的体感任务中能够学习到难度感知的推理策略。结果表明，仅使用6K训练样本，AdaNav在R2R val-unseen上的成功率提高了20%，在RxR-CE上提高了11.7%，在真实世界场景中提高了11.4%。代码可在此处访问：this https URL。 

---
# SONAR: Semantic-Object Navigation with Aggregated Reasoning through a Cross-Modal Inference Paradigm 

**Title (ZH)**: SONAR：基于跨模态推理框架的语义对象导航与聚合推理 

**Authors**: Yao Wang, Zhirui Sun, Wenzheng Chi, Baozhi Jia, Wenjun Xu, Jiankun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24321)  

**Abstract**: Understanding human instructions and accomplishing Vision-Language Navigation tasks in unknown environments is essential for robots. However, existing modular approaches heavily rely on the quality of training data and often exhibit poor generalization. Vision-Language Model based methods, while demonstrating strong generalization capabilities, tend to perform unsatisfactorily when semantic cues are weak. To address these issues, this paper proposes SONAR, an aggregated reasoning approach through a cross modal paradigm. The proposed method integrates a semantic map based target prediction module with a Vision-Language Model based value map module, enabling more robust navigation in unknown environments with varying levels of semantic cues, and effectively balancing generalization ability with scene adaptability. In terms of target localization, we propose a strategy that integrates multi-scale semantic maps with confidence maps, aiming to mitigate false detections of target objects. We conducted an evaluation of the SONAR within the Gazebo simulator, leveraging the most challenging Matterport 3D (MP3D) dataset as the experimental benchmark. Experimental results demonstrate that SONAR achieves a success rate of 38.4% and an SPL of 17.7%. 

**Abstract (ZH)**: 基于跨模态聚合推理的SONAR：在未知环境中实现目标定位与语义导航 

---
# Learning to Sample: Reinforcement Learning-Guided Sampling for Autonomous Vehicle Motion Planning 

**Title (ZH)**: 基于学习的采样：强化学习引导的自主车辆运动规划采样 

**Authors**: Korbinian Moller, Roland Stroop, Mattia Piccinini, Alexander Langmann, Johannes Betz  

**Link**: [PDF](https://arxiv.org/pdf/2509.24313)  

**Abstract**: Sampling-based motion planning is a well-established approach in autonomous driving, valued for its modularity and analytical tractability. In complex urban scenarios, however, uniform or heuristic sampling often produces many infeasible or irrelevant trajectories. We address this limitation with a hybrid framework that learns where to sample while keeping trajectory generation and evaluation fully analytical and verifiable. A reinforcement learning (RL) agent guides the sampling process toward regions of the action space likely to yield feasible trajectories, while evaluation and final selection remains governed by deterministic feasibility checks and cost functions. We couple the RL sampler with a world model (WM) based on a decodable deep set encoder, enabling both variable numbers of traffic participants and reconstructable latent representations. The approach is evaluated in the CommonRoad simulation environment, showing up to 99% fewer required samples and a runtime reduction of up to 84% while maintaining planning quality in terms of success and collision-free rates. These improvements lead to faster, more reliable decision-making for autonomous vehicles in urban environments, achieving safer and more responsive navigation under real-world constraints. Code and trained artifacts are publicly available at: this https URL 

**Abstract (ZH)**: 基于采样的运动规划是自主驾驶中一个成熟的 approach，因其模块化和分析可处理性而受到重视。然而，在复杂的城市场景中，均匀或启发式的采样往往会产生许多不可行或无关的轨迹。我们提出一种混合框架来解决这一局限性，该框架在保持轨迹生成和评估的完全分析性和可验证性的同时，学习如何采样。基于强化学习（RL）的代理引导采样过程，使其趋向于行动空间中可能性较大的可行轨迹区域，而评价和最终选择仍然由确定性的可行性和成本函数进行控制。我们将RL采样器与基于可解码深度集合编码器的世界模型耦合，允许交通参与者数量可变且能重建潜在表示。该方法在CommonRoad仿真环境中进行评估，结果显示所需的样本次数最多减少99%，运行时间最多减少84%，同时在成功和无碰撞率方面保持了规划质量。这些改进使得自主车辆在城市环境中能够更快、更可靠地做出决策，在实际约束条件下实现更安全、更响应的导航。相关代码和训练成果已在以下网址公开：this https URL。 

---
# Contextual Neural Moving Horizon Estimation for Robust Quadrotor Control in Varying Conditions 

**Title (ZH)**: 变条件下鲁棒旋翼无人机控制的上下文神经移动_horizon估计算法 

**Authors**: Kasra Torshizi, Chak Lam Shek, Khuzema Habib, Guangyao Shi, Pratap Tokekar, Troi Williams  

**Link**: [PDF](https://arxiv.org/pdf/2509.24281)  

**Abstract**: Adaptive controllers on quadrotors typically rely on estimation of disturbances to ensure robust trajectory tracking. Estimating disturbances across diverse environmental contexts is challenging due to the inherent variability and uncertainty in the real world. Such estimators require extensive fine-tuning for a specific scenario, which makes them inflexible and brittle to changing conditions. Machine-learning approaches, such as training a neural network to tune the estimator's parameters, are promising. However, collecting data across all possible environmental contexts is impossible. It is also inefficient as the same estimator parameters could work for "nearby" contexts. In this paper, we present a sequential decision making strategy that decides which environmental contexts, using Bayesian Optimization with a Gaussian Process, to collect data from in order to ensure robust performance across a wide range of contexts. Our method, Contextual NeuroMHE, eliminates the need for exhaustive training across all environments while maintaining robust performance under different conditions. By enabling the neural network to adapt its parameters dynamically, our method improves both efficiency and generalization. Experimental results in various real-world settings demonstrate that our approach outperforms the prior work by 20.3\% in terms of maximum absolute position error and can capture the variations in the environment with a few carefully chosen contexts. 

**Abstract (ZH)**: 基于上下文的神经MHE在旋翼无人机中的顺序决策方法：通过贝叶斯优化实现广泛环境条件下的鲁棒性能 

---
# SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions 

**Title (ZH)**: SafeFlowMatcher：基于流动匹配与控制障碍函数的安全快速规划 

**Authors**: Jeongyong Yang, Seunghwan Jang, Soojean Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.24243)  

**Abstract**: Generative planners based on flow matching (FM) can produce high-quality paths in one or a few ODE steps, but their sampling dynamics offer no formal safety guarantees and can yield incomplete paths near constraints. We present SafeFlowMatcher, a planning framework that couples FM with control barrier functions (CBFs) to achieve both real-time efficiency and certified safety. SafeFlowMatcher uses a two-phase prediction-correction (PC) integrator: (i) a prediction phase integrates the learned FM once (or a few steps) to obtain a candidate path without intervention; (ii) a correction phase refines this path with a vanishing time-scaled vector field and a CBF-based quadratic program that minimally perturbs the vector field. We prove a barrier certificate for the resulting flow system, establishing forward invariance of a robust safe set and finite-time convergence to the safe set. By enforcing safety only on the executed path (rather than on all intermediate latent paths), SafeFlowMatcher avoids distributional drift and mitigates local trap problems. Across maze navigation and locomotion benchmarks, SafeFlowMatcher attains faster, smoother, and safer paths than diffusion- and FM-based baselines. Extensive ablations corroborate the contributions of the PC integrator and the barrier certificate. 

**Abstract (ZH)**: 基于流匹配的生成式规划器结合控制障碍函数的安全流匹配规划框架 

---
# PROFusion: Robust and Accurate Dense Reconstruction via Camera Pose Regression and Optimization 

**Title (ZH)**: PROFusion: 基于相机姿态回归与优化的鲁棒且准确的密集重建 

**Authors**: Siyan Dong, Zijun Wang, Lulu Cai, Yi Ma, Yanchao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24236)  

**Abstract**: Real-time dense scene reconstruction during unstable camera motions is crucial for robotics, yet current RGB-D SLAM systems fail when cameras experience large viewpoint changes, fast motions, or sudden shaking. Classical optimization-based methods deliver high accuracy but fail with poor initialization during large motions, while learning-based approaches provide robustness but lack sufficient accuracy for dense reconstruction. We address this challenge through a combination of learning-based initialization with optimization-based refinement. Our method employs a camera pose regression network to predict metric-aware relative poses from consecutive RGB-D frames, which serve as reliable starting points for a randomized optimization algorithm that further aligns depth images with the scene geometry. Extensive experiments demonstrate promising results: our approach outperforms the best competitor on challenging benchmarks, while maintaining comparable accuracy on stable motion sequences. The system operates in real-time, showcasing that combining simple and principled techniques can achieve both robustness for unstable motions and accuracy for dense reconstruction. Project page: this https URL. 

**Abstract (ZH)**: 实时不稳定相机运动下的密集场景重建对于机器人技术至关重要，而当前的RGB-D SLAM系统在相机经历大幅度视角变化、快速运动或突然震动时会失效。基于经典优化的方法在大规模运动时因初始条件较差而无法提供高精度，而基于学习的方法虽然更具鲁棒性，但在密集重建方面缺乏足够的准确性。我们通过结合基于学习的初始化与基于优化的精修来应对这一挑战。我们的方法利用摄像头姿态回归网络从连续的RGB-D帧中预测出具备度量感知的相对姿态，作为随机优化算法的可靠起始点，进一步将深度图像与场景几何对齐。实验结果表明，我们的方法在具有挑战性的基准测试中优于现有最佳方法，同时在稳定运动序列上保持了相当的精度。系统实现了实时运行，展示了简单而原理明确的技术组合能够同时实现不稳定运动的鲁棒性和密集重建的准确性。项目页面: [这里](this https URL)。 

---
# Towards Tighter Convex Relaxation of Mixed-integer Programs: Leveraging Logic Network Flow for Task and Motion Planning 

**Title (ZH)**: 基于逻辑网络流的混合整数规划 tighter 凸松弛方法研究：任务与运动规划中的应用 

**Authors**: Xuan Lin, Jiming Ren, Yandong Luo, Weijun Xie, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24235)  

**Abstract**: This paper proposes an optimization-based task and motion planning framework, named "Logic Network Flow", that integrates temporal logic specifications into mixed-integer programs for efficient robot planning. Inspired by the Graph-of-Convex-Sets formulation, temporal predicates are encoded as polyhedron constraints on each edge of a network flow model, instead of as constraints between nodes in traditional Logic Tree formulations. We further propose a network-flow-based Fourier-Motzkin elimination procedure that removes continuous flow variables while preserving convex relaxation tightness, leading to provably tighter convex relaxations and fewer constraints than Logic Tree formulations. For temporal logic motion planning with piecewise-affine dynamic systems, comprehensive experiments across vehicle routing, multi-robot coordination, and temporal logic control on dynamical systems using point mass and linear inverted pendulum models demonstrate computational speedups of up to several orders of magnitude. Hardware demonstrations with quadrupedal robots validate real-time replanning capabilities under dynamically changing environmental conditions. The project website is at this https URL. 

**Abstract (ZH)**: 基于优化的任务与运动规划框架“逻辑网络流”：将时间逻辑规范整合到混合整数规划中用于高效机器人规划 

---
# ViReSkill: Vision-Grounded Replanning with Skill Memory for LLM-Based Planning in Lifelong Robot Learning 

**Title (ZH)**: ViReSkill: 以视觉为基础的技能记忆重新规划方法在终身机器人学习中的应用 

**Authors**: Tomoyuki Kagaya, Subramanian Lakshmi, Anbang Ye, Thong Jing Yuan, Jayashree Karlekar, Sugiri Pranata, Natsuki Murakami, Akira Kinose, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2509.24219)  

**Abstract**: Robots trained via Reinforcement Learning (RL) or Imitation Learning (IL) often adapt slowly to new tasks, whereas recent Large Language Models (LLMs) and Vision-Language Models (VLMs) promise knowledge-rich planning from minimal data. Deploying LLMs/VLMs for motion planning, however, faces two key obstacles: (i) symbolic plans are rarely grounded in scene geometry and object physics, and (ii) model outputs can vary for identical prompts, undermining execution reliability. We propose ViReSkill, a framework that pairs vision-grounded replanning with a skill memory for accumulation and reuse. When a failure occurs, the replanner generates a new action sequence conditioned on the current scene, tailored to the observed state. On success, the executed plan is stored as a reusable skill and replayed in future encounters without additional calls to LLMs/VLMs. This feedback loop enables autonomous continual learning: each attempt immediately expands the skill set and stabilizes subsequent executions. We evaluate ViReSkill on simulators such as LIBERO and RLBench as well as on a physical robot. Across all settings, it consistently outperforms conventional baselines in task success rate, demonstrating robust sim-to-real generalization. 

**Abstract (ZH)**: 基于视觉grounding的重规划与技能记忆结合框架：ViReSkill 

---
# Very High Frequency Interpolation for Direct Torque Control 

**Title (ZH)**: 非常高速插值在直接转矩控制中的应用 

**Authors**: Rafael Kourdis, Maciej Stępień, Jérôme Manhes, Nicolas Mansard, Steve Tonneau, Philippe Souères, Thomas Flayols  

**Link**: [PDF](https://arxiv.org/pdf/2509.24175)  

**Abstract**: Torque control enables agile and robust robot motion, but deployment is often hindered by instability and hardware limits. Here, we present a novel solution to execute whole-body linear feedback at up to 40 kHz on open-source hardware. We use this to interpolate non-linear schemes during real-world execution, such as inverse dynamics and learned torque policies. Our results show that by stabilizing torque controllers, high-frequency linear feedback could be an effective route towards unlocking the potential of torque-controlled robotics. 

**Abstract (ZH)**: 扭矩控制使机器人运动更加敏捷和 robust，但往往会因为不稳定性及硬件限制而难以部署。在此，我们提出了一种 novel 解决方案，可在开源硬件上以最高 40 kHz 的频率执行全身线性反馈。我们使用这种方法在实际执行过程中插值非线性方案，如逆动力学和学习到的扭矩策略。我们的结果表明，通过稳定扭矩控制器，高频线性反馈可能是解锁扭矩控制机器人潜力的有效途径。 

---
# Preference-Based Long-Horizon Robotic Stacking with Multimodal Large Language Models 

**Title (ZH)**: 基于偏好远期规划的多模态大型语言模型 Docker 堆叠 

**Authors**: Wanming Yu, Adrian Röfer, Abhinav Valada, Sethu Vijayakumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.24163)  

**Abstract**: Pretrained large language models (LLMs) can work as high-level robotic planners by reasoning over abstract task descriptions and natural language instructions, etc. However, they have shown a lack of knowledge and effectiveness in planning long-horizon robotic manipulation tasks where the physical properties of the objects are essential. An example is the stacking of containers with hidden objects inside, which involves reasoning over hidden physics properties such as weight and stability. To this end, this paper proposes to use multimodal LLMs as high-level planners for such long-horizon robotic stacking tasks. The LLM takes multimodal inputs for each object to stack and infers the current best stacking sequence by reasoning over stacking preferences. Furthermore, in order to enable the LLM to reason over multiple preferences at the same time without giving explicit instructions, we propose to create a custom dataset considering stacking preferences including weight, stability, size, and footprint, to fine-tune the LLM. Compared to the pretrained LLM with prompt tuning, we demonstrate the improved stacking completion of the LLM fine-tuned with our custom dataset via large-scale simulation evaluation. Furthermore, we showcase the effectiveness of the proposed framework for the long-horizon stacking task on a real humanoid robot in an online manner. 

**Abstract (ZH)**: 预训练大型语言模型可以作为高级机器人规划者，通过推理抽象的任务描述和自然语言指令等信息。然而，在规划长期 horizon 的机器人操作任务方面，它们在涉及物体物理特性的任务中显示出知识和效果的不足。例如，在含有隐藏物体的集装箱堆叠任务中，需要推理隐藏的物理属性，如重量和稳定性。为此，本文提议使用多模态大型语言模型作为此类长期 horizon 机器人堆叠任务的高级规划者。该语言模型接受每个待堆叠对象的多模态输入，并通过推理堆叠偏好来推断当前最佳的堆叠顺序。此外，为了使语言模型能够在不给出明确指令的情况下同时推理多种偏好，我们提议创建一个包含重量、稳定性、大小和占地面积等堆叠偏好的自定义数据集，以微调大型语言模型。与使用提示调整的预训练大型语言模型相比，我们通过大规模模拟评估展示了使用我们自定义数据集微调后的大型语言模型在堆叠完成方面的改进。此外，我们在线展示了所提议框架在真实人形机器人上的有效性，用于长期 horizon 的堆叠任务。 

---
# Memory Transfer Planning: LLM-driven Context-Aware Code Adaptation for Robot Manipulation 

**Title (ZH)**: 记忆转移规划：基于LLM的上下文感知代码适应性规划用于机器人操控 

**Authors**: Tomoyuki Kagaya, Subramanian Lakshmi, Yuxuan Lou, Thong Jing Yuan, Jayashree Karlekar, Sugiri Pranata, Natsuki Murakami, Akira Kinose, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2509.24160)  

**Abstract**: Large language models (LLMs) are increasingly explored in robot manipulation, but many existing methods struggle to adapt to new environments. Many systems require either environment-specific policy training or depend on fixed prompts and single-shot code generation, leading to limited transferability and manual re-tuning. We introduce Memory Transfer Planning (MTP), a framework that leverages successful control-code examples from different environments as procedural knowledge, using them as in-context guidance for LLM-driven planning. Specifically, MTP (i) generates an initial plan and code using LLMs, (ii) retrieves relevant successful examples from a code memory, and (iii) contextually adapts the retrieved code to the target setting for re-planning without updating model parameters. We evaluate MTP on RLBench, CALVIN, and a physical robot, demonstrating effectiveness beyond simulation. Across these settings, MTP consistently improved success rate and adaptability compared with fixed-prompt code generation, naive retrieval, and memory-free re-planning. Furthermore, in hardware experiments, leveraging a memory constructed in simulation proved effective. MTP provides a practical approach that exploits procedural knowledge to realize robust LLM-based planning across diverse robotic manipulation scenarios, enhancing adaptability to novel environments and bridging simulation and real-world deployment. 

**Abstract (ZH)**: 大规模语言模型在机器人操作中的记忆转移规划 

---
# A Novel Model for 3D Motion Planning for a Generalized Dubins Vehicle with Pitch and Yaw Rate Constraints 

**Title (ZH)**: 一种具有滚转和偏航率约束的一般化杜宾车三维运动规划的新模型 

**Authors**: Deepak Prakash Kumar, Swaroop Darbha, Satyanarayana Gupta Manyam, David Casbeer  

**Link**: [PDF](https://arxiv.org/pdf/2509.24143)  

**Abstract**: In this paper, we propose a new modeling approach and a fast algorithm for 3D motion planning, applicable for fixed-wing unmanned aerial vehicles. The goal is to construct the shortest path connecting given initial and final configurations subject to motion constraints. Our work differs from existing literature in two ways. First, we consider full vehicle orientation using a body-attached frame, which includes roll, pitch, and yaw angles. However, existing work uses only pitch and/or heading angle, which is insufficient to uniquely determine orientation. Second, we use two control inputs to represent bounded pitch and yaw rates, reflecting control by two separate actuators. In contrast, most previous methods rely on a single input, such as path curvature, which is insufficient for accurately modeling the vehicle's kinematics in 3D. We use a rotation minimizing frame to describe the vehicle's configuration and its evolution, and construct paths by concatenating optimal Dubins paths on spherical, cylindrical, or planar surfaces. Numerical simulations show our approach generates feasible paths within 10 seconds on average and yields shorter paths than existing methods in most cases. 

**Abstract (ZH)**: 本文提出了一种新的建模方法和快速算法，用于固定翼无人机的3D运动规划。目标是在运动约束条件下，构造给定初始和最终配置之间的最短路径。与现有文献相比，我们的工作主要在两个方面有所不同。首先，我们使用关联于机体的坐标系考虑全机姿态，包括滚转、俯仰和偏航角。而现有工作仅使用俯仰角和/或航向角，不足以唯一确定姿态。其次，我们使用两个控制输入来表示有界俯仰和偏航角速率，反映了由两个独立作动器控制的特点。相比之下，大多数先前方法依赖单一输入，如路径曲率，这在准确建模3D运动学时是不足的。我们使用旋转最小化框架描述无人机的姿态及其演变，并通过在球面、圆柱面或平面表面上连接最优杜宾斯路径来构建路径。数值仿真显示，我们的方法平均在10秒内生成可行路径，并在大多数情况下生成的路径比现有方法更短。 

---
# Mash, Spread, Slice! Learning to Manipulate Object States via Visual Spatial Progress 

**Title (ZH)**: 压碎、涂抹、切片！通过视觉空间进展学习操控物体状态 

**Authors**: Priyanka Mandikal, Jiaheng Hu, Shivin Dass, Sagnik Majumder, Roberto Martín-Martín, Kristen Grauman  

**Link**: [PDF](https://arxiv.org/pdf/2509.24129)  

**Abstract**: Most robot manipulation focuses on changing the kinematic state of objects: picking, placing, opening, or rotating them. However, a wide range of real-world manipulation tasks involve a different class of object state change--such as mashing, spreading, or slicing--where the object's physical and visual state evolve progressively without necessarily changing its position. We present SPARTA, the first unified framework for the family of object state change manipulation tasks. Our key insight is that these tasks share a common structural pattern: they involve spatially-progressing, object-centric changes that can be represented as regions transitioning from an actionable to a transformed state. Building on this insight, SPARTA integrates spatially progressing object change segmentation maps, a visual skill to perceive actionable vs. transformed regions for specific object state change tasks, to generate a) structured policy observations that strip away appearance variability, and b) dense rewards that capture incremental progress over time. These are leveraged in two SPARTA policy variants: reinforcement learning for fine-grained control without demonstrations or simulation; and greedy control for fast, lightweight deployment. We validate SPARTA on a real robot for three challenging tasks across 10 diverse real-world objects, achieving significant improvements in training time and accuracy over sparse rewards and visual goal-conditioned baselines. Our results highlight progress-aware visual representations as a versatile foundation for the broader family of object state manipulation tasks. Project website: this https URL 

**Abstract (ZH)**: 一种统一的物体状态变化操作框架：SPARTA 

---
# BOSfM: A View Planning Framework for Optimal 3D Reconstruction of Agricultural Scenes 

**Title (ZH)**: BOSfM: 农业场景最优3D重建的视角规划框架 

**Authors**: Athanasios Bacharis, Konstantinos D. Polyzos, Georgios B. Giannakis, Nikolaos Papanikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.24126)  

**Abstract**: Active vision (AV) has been in the spotlight of robotics research due to its emergence in numerous applications including agricultural tasks such as precision crop monitoring and autonomous harvesting to list a few. A major AV problem that gained popularity is the 3D reconstruction of targeted environments using 2D images from diverse viewpoints. While collecting and processing a large number of arbitrarily captured 2D images can be arduous in many practical scenarios, a more efficient solution involves optimizing the placement of available cameras in 3D space to capture fewer, yet more informative, images that provide sufficient visual information for effective reconstruction of the environment of interest. This process termed as view planning (VP), can be markedly challenged (i) by noise emerging in the location of the cameras and/or in the extracted images, and (ii) by the need to generalize well in other unknown similar agricultural environments without need for re-optimizing or re-training. To cope with these challenges, the present work presents a novel VP framework that considers a reconstruction quality-based optimization formulation that relies on the notion of `structure-from-motion' to reconstruct the 3D structure of the sought environment from the selected 2D images. With no analytic expression of the optimization function and with costly function evaluations, a Bayesian optimization approach is proposed to efficiently carry out the VP process using only a few function evaluations, while accounting for different noise cases. Numerical tests on both simulated and real agricultural settings signify the benefits of the advocated VP approach in efficiently estimating the optimal camera placement to accurately reconstruct 3D environments of interest, and generalize well on similar unknown environments. 

**Abstract (ZH)**: 基于运动结构的农业环境视点规划方法 

---
# Ancestry Tree Clustering for Particle Filter Diversity Maintenance 

**Title (ZH)**: 族系树聚类以维护粒子滤波器多样性 

**Authors**: Ilari Vallivaara, Bingnan Duan, Yinhuan Dong, Tughrul Arslan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24124)  

**Abstract**: We propose a method for linear-time diversity maintenance in particle filtering. It clusters particles based on ancestry tree topology: closely related particles in sufficiently large subtrees are grouped together. The main idea is that the tree structure implicitly encodes similarity without the need for spatial or other domain-specific metrics. This approach, when combined with intra-cluster fitness sharing and the protection of particles not included in a cluster, effectively prevents premature convergence in multimodal environments while maintaining estimate compactness. We validate our approach in a multimodal robotics simulation and a real-world multimodal indoor environment. We compare the performance to several diversity maintenance algorithms from the literature, including Deterministic Resampling and Particle Gaussian Mixtures. Our algorithm achieves high success rates with little to no negative effect on compactness, showing particular robustness to different domains and challenging initial conditions. 

**Abstract (ZH)**: 一种粒子滤波中的线性时间多样性维护方法：基于祖先树拓扑的聚类方法 

---
# Prepare for Warp Speed: Sub-millisecond Visual Place Recognition Using Event Cameras 

**Title (ZH)**: 准备高速模式：使用事件相机的亚毫秒级视觉.place识别 

**Authors**: Vignesh Ramanathan, Michael Milford, Tobias Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.24094)  

**Abstract**: Visual Place Recognition (VPR) enables systems to identify previously visited locations within a map, a fundamental task for autonomous navigation. Prior works have developed VPR solutions using event cameras, which asynchronously measure per-pixel brightness changes with microsecond temporal resolution. However, these approaches rely on dense representations of the inherently sparse camera output and require tens to hundreds of milliseconds of event data to predict a place. Here, we break this paradigm with Flash, a lightweight VPR system that predicts places using sub-millisecond slices of event data. Our method is based on the observation that active pixel locations provide strong discriminative features for VPR. Flash encodes these active pixel locations using efficient binary frames and computes similarities via fast bitwise operations, which are then normalized based on the relative event activity in the query and reference frames. Flash improves Recall@1 for sub-millisecond VPR over existing baselines by 11.33x on the indoor QCR-Event-Dataset and 5.92x on the 8 km Brisbane-Event-VPR dataset. Moreover, our approach reduces the duration for which the robot must operate without awareness of its position, as evidenced by a localization latency metric we term Time to Correct Match (TCM). To the best of our knowledge, this is the first work to demonstrate sub-millisecond VPR using event cameras. 

**Abstract (ZH)**: 基于事件的极短时视觉地点识别（VPR）abling系统在地图中识别先前访问的位置，是自主导航的基本任务。以往研究利用事件摄像头开发VPR解决方案，这些摄像头以微秒级时间分辨率异步测量每個像素的亮度变化。然而，这些方法依赖于密集表示的固有稀疏摄像头输出，并需要数十到数百毫秒的事件数据来预测地点。在这里，我们通过 flash，一种轻量级VPR系统打破这一范式，该系统使用极短时事件数据片段预测地点。我们的方法基于观察到活跃像素位置为VPR提供了强烈的鉴别特征。Flash使用高效的二进制帧编码这些活跃像素位置，并通过快速位操作计算相似性，然后基于查询帧和参考帧的相对事件活动进行归一化。Flash在室内的QCR-Event-Dataset上将Recall@1的VPR性能比现有基线提高11.33倍，在8公里的布里斯班-事件-VPR数据集上提高5.92倍。此外，我们的方法减少了机器人在无位置意识状态下操作的持续时间，如我们所称的时间到正确匹配（TCM）的本地化延迟度量所示。据我们所知，这是首次使用事件摄像头实现极短时VPR的工作。 

---
# MAD-PINN: A Decentralized Physics-Informed Machine Learning Framework for Safe and Optimal Multi-Agent Control 

**Title (ZH)**: MAD-PINN：一个用于安全和最优多agent控制的去中心化物理自适应机器学习框架 

**Authors**: Manan Tayal, Aditya Singh, Shishir Kolathaya, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2509.23960)  

**Abstract**: Co-optimizing safety and performance in large-scale multi-agent systems remains a fundamental challenge. Existing approaches based on multi-agent reinforcement learning (MARL), safety filtering, or Model Predictive Control (MPC) either lack strict safety guarantees, suffer from conservatism, or fail to scale effectively. We propose MAD-PINN, a decentralized physics-informed machine learning framework for solving the multi-agent state-constrained optimal control problem (MASC-OCP). Our method leverages an epigraph-based reformulation of SC-OCP to simultaneously capture performance and safety, and approximates its solution via a physics-informed neural network. Scalability is achieved by training the SC-OCP value function on reduced-agent systems and deploying them in a decentralized fashion, where each agent relies only on local observations of its neighbours for decision-making. To further enhance safety and efficiency, we introduce an Hamilton-Jacobi (HJ) reachability-based neighbour selection strategy to prioritize safety-critical interactions, and a receding-horizon policy execution scheme that adapts to dynamic interactions while reducing computational burden. Experiments on multi-agent navigation tasks demonstrate that MAD-PINN achieves superior safety-performance trade-offs, maintains scalability as the number of agents grows, and consistently outperforms state-of-the-art baselines. 

**Abstract (ZH)**: 在大规模多Agent系统中同时优化安全性和性能依然是一个基本挑战。现有基于多Agent强化学习（MARL）、安全过滤或模型预测控制（MPC）的方法要么缺乏严格的安全性保证，要么保守性过强，要么难以有效扩展。我们提出了MAD-PINN，这是一种分布式物理启发式机器学习框架，用于解决多Agent状态约束最优控制问题（MASC-OCP）。该方法利用SC-OCP的上图表示进行重新公式化，以同时捕获性能和安全性，并通过物理启发式神经网络近似其解。通过在减少Agent数量的系统上训练SC-OCP价值函数，并以去中心化的方式部署，实现了扩展性，其中每个Agent仅依赖于其邻居的局部观察来进行决策。为进一步提高安全性和效率，我们引入了基于哈密尔顿-雅可比（HJ）可达性的邻居选择策略来优先处理安全关键交互，并采用滚动时域策略执行方案，该方案能够适应动态交互并减少计算负担。实验表明，MAD-PINN在多Agent导航任务中实现了更优的安全-性能权衡，能够保持扩展性，且在越来越多的Agent数量下持续优于最先进的基线方法。 

---
# DexFlyWheel: A Scalable and Self-improving Data Generation Framework for Dexterous Manipulation 

**Title (ZH)**: DexFlyWheel: 一种可扩展且自我改进的 Dexterous 操作数据生成框架 

**Authors**: Kefei Zhu, Fengshuo Bai, YuanHao Xiang, Yishuai Cai, Xinglin Chen, Ruochong Li, Xingtao Wang, Hao Dong, Yaodong Yang, Xiaopeng Fan, Yuanpei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23829)  

**Abstract**: Dexterous manipulation is critical for advancing robot capabilities in real-world applications, yet diverse and high-quality datasets remain scarce. Existing data collection methods either rely on human teleoperation or require significant human engineering, or generate data with limited diversity, which restricts their scalability and generalization. In this paper, we introduce DexFlyWheel, a scalable data generation framework that employs a self-improving cycle to continuously enrich data diversity. Starting from efficient seed demonstrations warmup, DexFlyWheel expands the dataset through iterative cycles. Each cycle follows a closed-loop pipeline that integrates Imitation Learning (IL), residual Reinforcement Learning (RL), rollout trajectory collection, and data augmentation. Specifically, IL extracts human-like behaviors from demonstrations, and residual RL enhances policy generalization. The learned policy is then used to generate trajectories in simulation, which are further augmented across diverse environments and spatial configurations before being fed back into the next cycle. Over successive iterations, a self-improving data flywheel effect emerges, producing datasets that cover diverse scenarios and thereby scaling policy performance. Experimental results demonstrate that DexFlyWheel generates over 2,000 diverse demonstrations across four challenging tasks. Policies trained on our dataset achieve an average success rate of 81.9\% on the challenge test sets and successfully transfer to the real world through digital twin, achieving a 78.3\% success rate on dual-arm lift tasks. 

**Abstract (ZH)**: DexFlyWheel：一种可扩展的数据生成框架以促进机器人灵巧操作能力的发展 

---
# Control Your Robot: A Unified System for Robot Control and Policy Deployment 

**Title (ZH)**: 控制你的机器人：统一的机器人控制与策略部署系统 

**Authors**: Tian Nian, Weijie Ke, Yao Mu, Tianxing Chen, Shaolong Zhu, Bingshan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23823)  

**Abstract**: Cross-platform robot control remains difficult because hardware interfaces, data formats, and control paradigms vary widely, which fragments toolchains and slows deployment. To address this, we present Control Your Robot, a modular, general-purpose framework that unifies data collection and policy deployment across diverse platforms. The system reduces fragmentation through a standardized workflow with modular design, unified APIs, and a closed-loop architecture. It supports flexible robot registration, dual-mode control with teleoperation and trajectory playback, and seamless integration from multimodal data acquisition to inference. Experiments on single-arm and dual-arm systems show efficient, low-latency data collection and effective support for policy learning with imitation learning and vision-language-action models. Policies trained on data gathered by Control Your Robot match expert demonstrations closely, indicating that the framework enables scalable and reproducible robot learning across platforms. 

**Abstract (ZH)**: 跨平台机器人控制仍然困难，因为硬件接口、数据格式和控制范式差异很大，导致工具链碎片化并减缓部署速度。为此，我们提出了一种模块化、通用的框架——Control Your Robot，该框架统一了多种平台的数据采集和策略部署。该系统通过标准化的工作流、模块化设计、统一的API接口和闭环架构来减少碎片化。它支持灵活的机器人注册、远程操作与轨迹回放的双模式控制，并从多模态数据采集无缝集成到推理过程中。实验结果显示，在单臂和双臂系统上的高效、低延迟数据采集以及在模仿学习和视觉-语言-行动模型中的有效策略学习支持。使用Control Your Robot收集的数据训练出的策略能够紧密匹配专家演示，表明该框架使跨平台的机器人学习实现规模化和可重复性成为可能。 

---
# Fostering Robots: A Governance-First Conceptual Framework for Domestic, Curriculum-Based Trajectory Collection 

**Title (ZH)**: 培育机器人：一种以治理为导向的基于课程的轨迹收集概念框架 

**Authors**: Federico Pablo-Marti, Carlos Mir Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2509.23821)  

**Abstract**: We propose a conceptual, empirically testable framework for Robot Fostering, -a curriculum-driven, governance-first approach to domestic robot deployments, emphasizing long-term, curated interaction trajectories. We formalize trajectory quality with quantifiable metrics and evaluation protocols aligned with EU-grade governance standards, delineating a low-resource empirical roadmap to enable rigorous validation through future pilot studies. 

**Abstract (ZH)**: 我们提出了一种概念性、可验证的框架，用于机器人培养——一种以课程驱动、治理优先的方法，专注于长期、精心策划的互动轨迹。我们通过与欧盟级治理标准对齐的可量化指标和评估协议，形式化轨迹质量，并划分一个低资源的实证路线图，以通过未来的试点研究实现严格的验证。 

---
# High-Precision Climbing Robot Localization Using Planar Array UWB/GPS/IMU/Barometer Integration 

**Title (ZH)**: 使用平面阵列UWB/GPS/IMU/气压计集成实现高精度爬行机器人定位 

**Authors**: Shuning Zhang, Renjing Xu, Zhanchen Zhu, Xiangyu Chen, Yunheng Wang, Xu Jiang, Peibo Duan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23801)  

**Abstract**: To address the need for high-precision localization of climbing robots in complex high-altitude environments, this paper proposes a multi-sensor fusion system that overcomes the limitations of single-sensor approaches. Firstly, the localization scenarios and the problem model are analyzed. An integrated architecture of Attention Mechanism-based Fusion Algorithm (AMFA) incorporating planar array Ultra-Wideband (UWB), GPS, Inertial Measurement Unit (IMU), and barometer is designed to handle challenges such as GPS occlusion and UWB Non-Line-of-Sight (NLOS) problem. Then, End-to-end neural network inference models for UWB and barometer are developed, along with a multimodal attention mechanism for adaptive data fusion. An Unscented Kalman Filter (UKF) is applied to refine the trajectory, improving accuracy and robustness. Finally, real-world experiments show that the method achieves 0.48 m localization accuracy and lower MAX error of 1.50 m, outperforming baseline algorithms such as GPS/INS-EKF and demonstrating stronger robustness. 

**Abstract (ZH)**: 基于多传感器融合的攀爬机器人高精度定位系统研究 

---
# Sequence Pathfinder for Multi-Agent Pickup and Delivery in the Warehouse 

**Title (ZH)**: 仓库中多 agent 拣选和配送的序列探索者算法 

**Authors**: Zeyuan Zhang, Chaoran Li, Shao Zhang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23778)  

**Abstract**: Multi-Agent Pickup and Delivery (MAPD) is a challenging extension of Multi-Agent Path Finding (MAPF), where agents are required to sequentially complete tasks with fixed-location pickup and delivery demands. Although learning-based methods have made progress in MAPD, they often perform poorly in warehouse-like environments with narrow pathways and long corridors when relying only on local observations for distributed decision-making. Communication learning can alleviate the lack of global information but introduce high computational complexity due to point-to-point communication. To address this challenge, we formulate MAPF as a sequence modeling problem and prove that path-finding policies under sequence modeling possess order-invariant optimality, ensuring its effectiveness in MAPD. Building on this, we propose the Sequential Pathfinder (SePar), which leverages the Transformer paradigm to achieve implicit information exchange, reducing decision-making complexity from exponential to linear while maintaining efficiency and global awareness. Experiments demonstrate that SePar consistently outperforms existing learning-based methods across various MAPF tasks and their variants, and generalizes well to unseen environments. Furthermore, we highlight the necessity of integrating imitation learning in complex maps like warehouses. 

**Abstract (ZH)**: 多代理拣选与配送（MAPD）是多代理路径查找（MAPF）的一个具有挑战性的扩展，其中代理需要按顺序完成固定位置的拣选与配送任务。虽然基于学习的方法在MAPD方面取得了一定进展，但在依赖局部观察进行分布式决策时，它们往往在狭窄路径和长走廊的仓库-like环境中表现不佳。通过学习的通信可以缓解缺少全局信息的问题，但会引入高计算复杂度。为解决这一挑战，我们将MAPF建模为序列建模问题，并证明在序列建模下的路径查找策略具有顺序不变的最优化性，确保其在MAPD中的有效性。在此基础上，我们提出了Seqential Pathfinder（SePar），利用Transformer范式实现隐式信息交换，将决策复杂度从指数级降低到线性级，同时保持效率和全局意识。实验表明，SePar在各种MAPF任务及其变体中均能持续优于现有学习方法，并能很好地泛化到未见过的环境中。此外，我们在复杂地图如仓库中强调集成模拟学习的重要性。 

---
# LocoFormer: Generalist Locomotion via Long-context Adaptation 

**Title (ZH)**: LocoFormer: 通过长上下文适应实现通用运动能力 

**Authors**: Min Liu, Deepak Pathak, Ananye Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.23745)  

**Abstract**: Modern locomotion controllers are manually tuned for specific embodiments. We present LocoFormer, a generalist omni-bodied locomotion model that can control previously unseen legged and wheeled robots, even without precise knowledge of their kinematics. LocoFormer is able to adapt to changes in morphology and dynamics at test time. We find that two key choices enable adaptation. First, we train massive scale RL on procedurally generated robots with aggressive domain randomization. Second, in contrast to previous policies that are myopic with short context lengths, we extend context by orders of magnitude to span episode boundaries. We deploy the same LocoFormer to varied robots and show robust control even with large disturbances such as weight change and motor failures. In extreme scenarios, we see emergent adaptation across episodes, LocoFormer learns from falls in early episodes to improve control strategies in later ones. We believe that this simple, yet general recipe can be used to train foundation models for other robotic skills in the future. Videos at this http URL. 

**Abstract (ZH)**: 现代运动控制器针对特定的身体形态手动调整。本文提出LocoFormer，这是一种通用型全能运动模型，可以控制以前未见过的腿足和轮式机器人，即使缺乏其精确的运动学知识。LocoFormer能够在测试时适应形态和动力学的变化。我们发现两种关键选择使模型具备了适应能力。首先，我们通过程序生成的机器人进行大规模强化学习训练，并使用激进的领域随机化。其次，不同于之前具有短上下文长度的短视策略，我们扩展了上下文范围，使其跨越整个episode界限。我们使用相同的LocoFormer部署在多种机器人上，即使在重量变化和电机故障等较大干扰下仍然表现出鲁棒性的控制性能。在极端场景下，我们观察到在早期episode中从跌倒中学到的东西能够在后续episode中改进控制策略。我们认为，这种简单且通用的方法在未来可以用于训练其他机器人技能的基座模型。视频见此链接。 

---
# DA-MMP: Learning Coordinated and Accurate Throwing with Dynamics-Aware Motion Manifold Primitives 

**Title (ZH)**: DA-MMP：具有动力学意识的运动流形基元学习协调和准确的投掷动作 

**Authors**: Chi Chu, Huazhe Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23721)  

**Abstract**: Dynamic manipulation is a key capability for advancing robot performance, enabling skills such as tossing. While recent learning-based approaches have pushed the field forward, most methods still rely on manually designed action parameterizations, limiting their ability to produce the highly coordinated motions required in complex tasks. Motion planning can generate feasible trajectories, but the dynamics gap-stemming from control inaccuracies, contact uncertainties, and aerodynamic effects-often causes large deviations between planned and executed trajectories. In this work, we propose Dynamics-Aware Motion Manifold Primitives (DA-MMP), a motion generation framework for goal-conditioned dynamic manipulation, and instantiate it on a challenging real-world ring-tossing task. Our approach extends motion manifold primitives to variable-length trajectories through a compact parametrization and learns a high-quality manifold from a large-scale dataset of planned motions. Building on this manifold, a conditional flow matching model is trained in the latent space with a small set of real-world trials, enabling the generation of throwing trajectories that account for execution dynamics. Experiments show that our method can generate coordinated and smooth motion trajectories for the ring-tossing task. In real-world evaluations, it achieves high success rates and even surpasses the performance of trained human experts. Moreover, it generalizes to novel targets beyond the training range, indicating that it successfully learns the underlying trajectory-dynamics mapping. 

**Abstract (ZH)**: 动态感知运动流形基元（DA-MMP）：面向目标条件动态操控的运动生成框架 

---
# MDCPP: Multi-robot Dynamic Coverage Path Planning for Workload Adaptation 

**Title (ZH)**: 多机器人动态覆盖路径规划以适应工作负载 

**Authors**: Jun Chen, Mingjia Chen, Shinkyu Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.23705)  

**Abstract**: Multi-robot Coverage Path Planning (MCPP) addresses the problem of computing paths for multiple robots to effectively cover a large area of interest. Conventional approaches to MCPP typically assume that robots move at fixed velocities, which is often unrealistic in real-world applications where robots must adapt their speeds based on the specific coverage tasks assigned to this http URL, conventional approaches often lead to imbalanced workload distribution among robots and increased completion time for coverage tasks. To address this, we introduce a novel Multi-robot Dynamic Coverage Path Planning (MDCPP) algorithm for complete coverage in two-dimensional environments. MDCPP dynamically estimates each robot's remaining workload by approximating the target distribution with Gaussian mixture models, and assigns coverage regions using a capacity-constrained Voronoi diagram. We further develop a distributed implementation of MDCPP for range-constrained robotic networks. Simulation results validate the efficacy of MDCPP, showing qualitative improvements and superior performance compared to an existing sweeping algorithm, and a quantifiable impact of communication range on coverage efficiency. 

**Abstract (ZH)**: 多机器人动态覆盖路径规划（MDCPP）：二维环境中的完全覆盖问题 

---
# Certifiably Optimal State Estimation and Robot Calibration Using Trace-Constrained SDP 

**Title (ZH)**: 使用约束迹SDP的可验证最优状态估算与机器人校准 

**Authors**: Liangting Wu, Roberto Tron  

**Link**: [PDF](https://arxiv.org/pdf/2509.23656)  

**Abstract**: Many nonconvex problems in robotics can be relaxed into convex formulations via semidefinite programming (SDP), which offers the advantage of global optimality. The practical quality of these solutions, however, critically depends on achieving rank-1 matrices, a condition that typically requires additional tightening. In this work, we focus on trace-constrained SDPs, where the decision variables are positive semidefinite (PSD) matrices with fixed trace values. These additional constraints not only capture important structural properties but also facilitate first-order methods for recovering rank-1 solutions. We introduce customized fixed-trace variables and constraints to represent common robotic quantities such as rotations and translations, which can be exactly recovered when the corresponding variables are rank-1. To further improve practical performance, we develop a gradient-based refinement procedure that projects relaxed SDP solutions toward rank-1, low-cost candidates, which can then be certified for global optimality via the dual problem. We demonstrate that many robotics tasks can be expressed within this trace-constrained SDP framework, and showcase its effectiveness through simulations in perspective-n-point (PnP) estimation, hand-eye calibration, and dual-robot system calibration. To support broader use, we also introduce a modular ``virtual robot'' abstraction that simplifies modeling across different problem settings. 

**Abstract (ZH)**: 基于迹约束半定规划的机器人非凸问题求解 

---
# Focusing on What Matters: Object-Agent-centric Tokenization for Vision Language Action models 

**Title (ZH)**: 聚焦关键：面向对象-代理的_token化方法用于视觉语言行动模型 

**Authors**: Rokas Bendikas, Daniel Dijkman, Markus Peschl, Sanjay Haresh, Pietro Mazzaglia  

**Link**: [PDF](https://arxiv.org/pdf/2509.23655)  

**Abstract**: Vision-Language-Action (VLA) models offer a pivotal approach to learning robotic manipulation at scale by repurposing large pre-trained Vision-Language-Models (VLM) to output robotic actions. However, adapting VLMs for robotic domains comes with an unnecessarily high computational cost, which we attribute to the tokenization scheme of visual inputs. In this work, we aim to enable efficient VLA training by proposing Oat-VLA, an Object-Agent-centric Tokenization for VLAs. Building on the insights of object-centric representation learning, our method introduces an inductive bias towards scene objects and the agent's own visual information. As a result, we find that Oat-VLA can drastically reduce the number of visual tokens to just a few tokens without sacrificing performance. We reveal that Oat-VLA converges at least twice as fast as OpenVLA on the LIBERO suite, as well as outperform OpenVLA in diverse real-world pick and place tasks. 

**Abstract (ZH)**: 面向物体-代理的Token化（Oat-VLA）模型：一种用于Vision-Language-Action（VLA）训练的有效方法 

---
# HeLoM: Hierarchical Learning for Whole-Body Loco-Manipulation in Hexapod Robot 

**Title (ZH)**: HeLoM: 分级学习在六足机器人全身移动物体中的应用 

**Authors**: Xinrong Yang, Peizhuo Li, Hongyi Li, Junkai Lu, Linnan Chang, Yuhong Cao, Yifeng Zhang, Ge Sun, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2509.23651)  

**Abstract**: Robots in real-world environments are often required to move/manipulate objects comparable in weight to their own bodies. Compared to grasping and carrying, pushing provides a more straightforward and efficient non-prehensile manipulation strategy, avoiding complex grasp design while leveraging direct contact to regulate an object's pose. Achieving effective pushing, however, demands both sufficient manipulation forces and the ability to maintain stability, which is particularly challenging when dealing with heavy or irregular objects. To address these challenges, we propose HeLoM, a learning-based hierarchical whole-body manipulation framework for a hexapod robot that exploits coordinated multi-limb control. Inspired by the cooperative strategies of multi-legged insects, our framework leverages redundant contact points and high degrees of freedom to enable dynamic redistribution of contact forces. HeLoM's high-level planner plans pushing behaviors and target object poses, while its low-level controller maintains locomotion stability and generates dynamically consistent joint actions. Our policies trained in simulation are directly deployed on real robots without additional fine-tuning. This design allows the robot to maintain balance while exerting continuous and controllable pushing forces through coordinated foreleg interaction and supportive hind-leg propulsion. We validate the effectiveness of HeLoM through both simulation and real-world experiments. Results show that our framework can stably push boxes of varying sizes and unknown physical properties to designated goal poses in the real world. 

**Abstract (ZH)**: 六足机器人在冗余接触点和高自由度的利用下实现高效动态推举策略：HeLoM框架 

---
# KiVi: Kinesthetic-Visuospatial Integration for Dynamic and Safe Egocentric Legged Locomotion 

**Title (ZH)**: KiVi: 动觉-视空整合技术用于动态和安全的自我中心腿足运动 

**Authors**: Peizhuo Li, Hongyi Li, Yuxuan Ma, Linnan Chang, Xinrong Yang, Ruiqi Yu, Yifeng Zhang, Yuhong Cao, Qiuguo Zhu, Guillaume Sartoretti  

**Link**: [PDF](https://arxiv.org/pdf/2509.23650)  

**Abstract**: Vision-based locomotion has shown great promise in enabling legged robots to perceive and adapt to complex environments. However, visual information is inherently fragile, being vulnerable to occlusions, reflections, and lighting changes, which often cause instability in locomotion. Inspired by animal sensorimotor integration, we propose KiVi, a Kinesthetic-Visuospatial integration framework, where kinesthetics encodes proprioceptive sensing of body motion and visuospatial reasoning captures visual perception of surrounding terrain. Specifically, KiVi separates these pathways, leveraging proprioception as a stable backbone while selectively incorporating vision for terrain awareness and obstacle avoidance. This modality-balanced, yet integrative design, combined with memory-enhanced attention, allows the robot to robustly interpret visual cues while maintaining fallback stability through proprioception. Extensive experiments show that our method enables quadruped robots to stably traverse diverse terrains and operate reliably in unstructured outdoor environments, remaining robust to out-of-distribution (OOD) visual noise and occlusion unseen during training, thereby highlighting its effectiveness and applicability to real-world legged locomotion. 

**Abstract (ZH)**: 基于视觉的运动控制在使腿式机器人感知和适应复杂环境方面展现了巨大潜力。然而，视觉信息本质上是脆弱的，容易受到遮挡、反射和光照变化的影响，这往往会引起运动不稳定性。受动物感觉运动整合的启发，我们提出KiVi，一种运动感知识觉与空间视觉整合框架，其中运动感知识觉编码身体运动的本体感觉，而空间视觉推理捕捉周围地形的视觉感知。具体来说，KiVi 分离了这些途径，利用本体感觉作为稳定的支撑，同时选择性地融入视觉以增强地形意识和障碍物避让能力。这种多模态平衡但又整合的设计，结合增强的记忆注意力机制，使机器人能够在利用视觉提示的同时，通过本体感觉维持退化稳定性。广泛的实验表明，我们的方法使四足机器人能够稳定地穿越各种地形，并可靠地在未结构化的户外环境中操作，对训练期间未见过的分布外（OOD）的视觉噪声和遮挡保持鲁棒性，从而突显了其在实际腿式运动中的有效性和适用性。 

---
# Encoding Material Safety using Control Barrier Functions for Soft Actuator Control 

**Title (ZH)**: 使用控制障碍函数对软执行器进行材料安全性编码控制 

**Authors**: Nicholas Pagliocca, Behrad Koohbor, Mitja Trkov  

**Link**: [PDF](https://arxiv.org/pdf/2509.23623)  

**Abstract**: Until recently, the concept of soft robot safety was an informal notion, often attributed solely to the fact that soft robots are less likely to damage their operating environment than rigid robots. As the field moves toward feedback control for practical applications, it becomes increasingly important to define what safety means and to characterize how soft robots can become unsafe. The unifying theme of soft robotics is to achieve useful functionality through deformation. Consequently, limitations in constitutive model accuracy and risks of material failure are inherent to all soft robots and pose a key challenge in designing provably safe controllers. This work introduces a formal definition of material safety based on strain energy functions and provides a controller that enforces it. We characterize safe and unsafe sets of an incompressible hyperelastic material and demonstrate that safety can be enforced using a high-order control barrier function (HOCBF) with quadratic program-based feedback control. As a case study, we consider a pressurized hyperelastic tube with inertial effects, first-order viscous effects, and full-state feedback. Simulation results verify that the proposed methodology can enforce the material safety specification. 

**Abstract (ZH)**: 直到最近，软机器人安全的概念还是一个非正式的概念，通常被认为软机器人比刚性机器人更不可能损坏其操作环境。随着该领域向实际应用的反馈控制迈进，明确安全的含义并刻画软机器人如何变得不安全变得越来越重要。软机器人的一贯主题是通过变形实现有用的功能。因此，构成模型的准确性限制和材料失效的风险是所有软机器人都固有的问题，构成了设计可证明安全控制器的关键挑战。本工作基于应变能函数提出了一个材料安全的正式定义，并提供了一个能实现这一定义的控制器。我们刻画了不可压缩超弹性和安全及不安全的集合，并演示了如何使用高阶控制障碍函数（HOCBF）和基于二次规划的反馈控制来实现安全性。作为案例研究，我们考虑了一个具有惯性效应、一阶黏性效应和全状态反馈的加压超弹性管。仿真结果验证了所提出的方法可以实现材料安全规范。 

---
# Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints 

**Title (ZH)**: 基于语言对齐3D关键点的可泛化粗细粒度机器人操作方法 

**Authors**: Jianshu Hu, Lidi Wang, Shujia Li, Yunpeng Jiang, Xiao Li, Paul Weng, Yutong Ban  

**Link**: [PDF](https://arxiv.org/pdf/2509.23575)  

**Abstract**: Hierarchical coarse-to-fine policy, where a coarse branch predicts a region of interest to guide a fine-grained action predictor, has demonstrated significant potential in robotic 3D manipulation tasks by especially enhancing sample efficiency and enabling more precise manipulation. However, even augmented with pre-trained models, these hierarchical policies still suffer from generalization issues. To enhance generalization to novel instructions and environment variations, we propose Coarse-to-fine Language-Aligned manipulation Policy (CLAP), a framework that integrates three key components: 1) task decomposition, 2) VLM fine-tuning for 3D keypoint prediction, and 3) 3D-aware representation. Through comprehensive experiments in simulation and on a real robot, we demonstrate its superior generalization capability. Specifically, on GemBench, a benchmark designed for evaluating generalization, our approach achieves a 12\% higher average success rate than the SOTA method while using only 1/5 of the training trajectories. In real-world experiments, our policy, trained on only 10 demonstrations, successfully generalizes to novel instructions and environments. 

**Abstract (ZH)**: 从粗到细语言对齐操纵策略（CLAP）：一种结合任务拆解、VLM微调和三维aware表示的框架 

---
# GES-UniGrasp: A Two-Stage Dexterous Grasping Strategy With Geometry-Based Expert Selection 

**Title (ZH)**: GES-UniGrasp: 一种基于几何专家选择的两阶段灵巧抓取策略 

**Authors**: Fangting Xu, Jilin Zhu, Xiaoming Gu, Jianzhong Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23567)  

**Abstract**: Robust and human-like dexterous grasping of general objects is a critical capability for advancing intelligent robotic manipulation in real-world scenarios. However, existing reinforcement learning methods guided by grasp priors often result in unnatural behaviors. In this work, we present \textit{ContactGrasp}, a robotic dexterous pre-grasp and grasp dataset that explicitly accounts for task-relevant wrist orientation and thumb-index pinching coordination. The dataset covers 773 objects in 82 categories, providing a rich foundation for training human-like grasp strategies. Building upon this dataset, we perform geometry-based clustering to group objects by shape, enabling a two-stage Geometry-based Expert Selection (GES) framework that selects among specialized experts for grasping diverse object geometries, thereby enhancing adaptability to diverse shapes and generalization across categories. Our approach demonstrates natural grasp postures and achieves high success rates of 99.4\% and 96.3\% on the train and test sets, respectively, showcasing strong generalization and high-quality grasp execution. 

**Abstract (ZH)**: Robust和人性化的通用物体操作是提升实际应用场景中智能机器人 manipulation 关键能力。然而，现有的由抓取先验指导的强化学习方法常常导致不自然的行为。在本文中，我们提出了 ContactGrasp，这是一个考虑任务相关手腕姿态和拇指-食指夹持协调的机器人灵巧预抓取和抓取数据集。该数据集涵盖了 82 个类别中的 773 个物体，为训练类人抓取策略提供了丰富的基础。在此数据集的基础上，我们进行了基于几何的聚类，按形状对物体进行分组，从而建立了一种基于几何的专家选择（GES）的两阶段框架，能够在不同形状的物体抓取上选择专门的专家，从而增强对不同形状的适应性和类别的泛化能力。我们的方法展示了自然的抓取姿态，在训练集和测试集上分别实现了 99.4% 和 96.3% 的高成功率，彰显了强大的泛化能力和高质量的抓取执行。 

---
# RAVEN: Resilient Aerial Navigation via Open-Set Semantic Memory and Behavior Adaptation 

**Title (ZH)**: RAVEN：通过开集语义记忆和行为适应实现鲁棒空中导航 

**Authors**: Seungchan Kim, Omar Alama, Dmytro Kurdydyk, John Keller, Nikhil Keetha, Wenshan Wang, Yonatan Bisk, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2509.23563)  

**Abstract**: Aerial outdoor semantic navigation requires robots to explore large, unstructured environments to locate target objects. Recent advances in semantic navigation have demonstrated open-set object-goal navigation in indoor settings, but these methods remain limited by constrained spatial ranges and structured layouts, making them unsuitable for long-range outdoor search. While outdoor semantic navigation approaches exist, they either rely on reactive policies based on current observations, which tend to produce short-sighted behaviors, or precompute scene graphs offline for navigation, limiting adaptability to online deployment. We present RAVEN, a 3D memory-based, behavior tree framework for aerial semantic navigation in unstructured outdoor environments. It (1) uses a spatially consistent semantic voxel-ray map as persistent memory, enabling long-horizon planning and avoiding purely reactive behaviors, (2) combines short-range voxel search and long-range ray search to scale to large environments, (3) leverages a large vision-language model to suggest auxiliary cues, mitigating sparsity of outdoor targets. These components are coordinated by a behavior tree, which adaptively switches behaviors for robust operation. We evaluate RAVEN in 10 photorealistic outdoor simulation environments over 100 semantic tasks, encompassing single-object search, multi-class, multi-instance navigation and sequential task changes. Results show RAVEN outperforms baselines by 85.25% in simulation and demonstrate its real-world applicability through deployment on an aerial robot in outdoor field tests. 

**Abstract (ZH)**: 基于三维记忆的行为树框架在未结构化户外环境中的航空语义导航 

---
# High Torque Density PCB Axial Flux Permanent Magnet Motor for Micro Robots 

**Title (ZH)**: 高扭矩密度印制电路板轴向磁 Flux 永磁电机用于微机器人 

**Authors**: Jianren Wang, Jie Han, Abhinav Gupta, Deepak Pathak, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23561)  

**Abstract**: Quasi-direct-drive (QDD) actuation is transforming legged and manipulator robots by eliminating high-ratio gearboxes, yet it demands motors that deliver very high torque at low speed within a thin, disc-shaped joint envelope. Axial-flux permanent-magnet (AFPM) machines meet these geometric and torque requirements, but scaling them below a 20mm outer diameter is hampered by poor copper fill in conventional wound stators, inflating resistance and throttling continuous torque. This paper introduces a micro-scale AFPM motor that overcomes these limitations through printed-circuit-board (PCB) windings fabricated with advanced IC-substrate high-density interconnect (HDI) technology. The resulting 48-layer stator-formed by stacking four 12-layer HDI modules-achieves a record 45\% copper fill in a package only 5mm thick and 19mm in diameter. We perform comprehensive electromagnetic and thermal analyses to inform the motor design, then fabricate a prototype whose performance characteristics are experimentally verified. 

**Abstract (ZH)**: 基于轴向磁通永磁机的准直接驱动微型电机：通过高密度互连技术实现紧凑高效的设计 

---
# Zero-shot Whole-Body Manipulation with a Large-Scale Soft Robotic Torso via Guided Reinforcement Learning 

**Title (ZH)**: 基于引导强化学习的大型软体躯干的零样本全身 manipulation 

**Authors**: Curtis C. Johnson, Carlo Alessi, Egidio Falotico, Marc D. Killpack  

**Link**: [PDF](https://arxiv.org/pdf/2509.23556)  

**Abstract**: Whole-body manipulation is a powerful yet underexplored approach that enables robots to interact with large, heavy, or awkward objects using more than just their end-effectors. Soft robots, with their inherent passive compliance, are particularly well-suited for such contact-rich manipulation tasks, but their uncertainties in kinematics and dynamics pose significant challenges for simulation and control. In this work, we address this challenge with a simulation that can run up to 350x real time on a single thread in MuJoCo and provide a detailed analysis of the critical tradeoffs between speed and accuracy for this simulation. Using this framework, we demonstrate a successful zero-shot sim-to-real transfer of a learned whole-body manipulation policy, achieving an 88% success rate on the Baloo hardware platform. We show that guiding RL with a simple motion primitive is critical to this success where standard reward shaping methods struggled to produce a stable and successful policy for whole-body manipulation. Furthermore, our analysis reveals that the learned policy does not simply mimic the motion primitive. It exhibits beneficial reactive behavior, such as re-grasping and perturbation recovery. We analyze and contrast this learned policy against an open-loop baseline to show that the policy can also exhibit aggressive over-corrections under perturbation. To our knowledge, this is the first demonstration of forceful, six-DoF whole-body manipulation using two continuum soft arms on a large-scale platform (10 kg payloads), with zero-shot policy transfer. 

**Abstract (ZH)**: 全身 manipulation 是一种强大但尚未充分探索的方法，能够使机器人使用末端执行器以外的部分与大型、沉重或笨重的对象进行交互。具有内在被动顺应性的软机器人特别适合此类接触密集型 manipulation 任务，但其运动学和动力学的不确定性为仿真和控制带来了重大挑战。在本研究中，我们通过在 MuJoCo 上实现单线程高达 350 倍实时运行速度的仿真，并详细分析了速度与准确性的关键权衡。利用这一框架，我们展示了零样本仿真到现实转换的成功，该策略在 Baloo 硬件平台上实现了 88% 的成功率。我们证明，用简单的运动原型引导强化学习对于这一成功至关重要，而标准的奖励塑形方法难以为全身 manipulation 生成稳定且成功的行为策略。此外，我们的分析显示，学习到的策略不仅仅模仿运动原型，还表现出有益的反应行为，如重新抓取和扰动恢复。我们将这一学习策略与开环基线进行分析和对比，展示了策略在扰动下也可能表现出激进的过矫正行为。据我们所知，这是首次在大型平台上（10 kg 有效载荷）使用两个连续柔臂实现六自由度强力全身 manipulation 并实现零样本策略转换的示范。 

---
# Ask, Reason, Assist: Decentralized Robot Collaboration via Language and Logic 

**Title (ZH)**: 求询、推理、辅助：基于语言与逻辑的去中心化机器人协作 

**Authors**: Dan BW Choe, Sundhar Vinodh Sangeetha, Steven Emanuel, Chih-Yuan Chiu, Samuel Coogan, Shreyas Kousik  

**Link**: [PDF](https://arxiv.org/pdf/2509.23506)  

**Abstract**: Increased robot deployment, such as in warehousing, has revealed a need for seamless collaboration among heterogeneous robot teams to resolve unforeseen conflicts. To address this challenge, we propose a novel decentralized framework that enables robots to request and provide help. The process begins when a robot detects a conflict and uses a Large Language Model (LLM) to decide whether external assistance is required. If so, it crafts and broadcasts a natural language (NL) help request. Potential helper robots reason over the request and respond with offers of assistance, including information about the effect on their ongoing tasks. Helper reasoning is implemented via an LLM grounded in Signal Temporal Logic (STL) using a Backus-Naur Form (BNF) grammar, ensuring syntactically valid NL-to-STL translations, which are then solved as a Mixed Integer Linear Program (MILP). Finally, the requester robot selects a helper by reasoning over the expected increase in system-level total task completion time. We evaluated our framework through experiments comparing different helper-selection strategies and found that considering multiple offers allows the requester to minimize added makespan. Our approach significantly outperforms heuristics such as selecting the nearest available candidate helper robot, and achieves performance comparable to a centralized "Oracle" baseline but without heavy information demands. 

**Abstract (ZH)**: 异构机器人团队中无缝协作的需求及一种新型去中心化帮助请求与提供框架 

---
# Multi-Modal Manipulation via Multi-Modal Policy Consensus 

**Title (ZH)**: 多模态操作 via 多模态策略共识 

**Authors**: Haonan Chen, Jiaming Xu, Hongyu Chen, Kaiwen Hong, Binghao Huang, Chaoqi Liu, Jiayuan Mao, Yunzhu Li, Yilun Du, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2509.23468)  

**Abstract**: Effectively integrating diverse sensory modalities is crucial for robotic manipulation. However, the typical approach of feature concatenation is often suboptimal: dominant modalities such as vision can overwhelm sparse but critical signals like touch in contact-rich tasks, and monolithic architectures cannot flexibly incorporate new or missing modalities without retraining. Our method factorizes the policy into a set of diffusion models, each specialized for a single representation (e.g., vision or touch), and employs a router network that learns consensus weights to adaptively combine their contributions, enabling incremental of new representations. We evaluate our approach on simulated manipulation tasks in {RLBench}, as well as real-world tasks such as occluded object picking, in-hand spoon reorientation, and puzzle insertion, where it significantly outperforms feature-concatenation baselines on scenarios requiring multimodal reasoning. Our policy further demonstrates robustness to physical perturbations and sensor corruption. We further conduct perturbation-based importance analysis, which reveals adaptive shifts between modalities. 

**Abstract (ZH)**: 有效整合多种传感模态对于机器人操作至关重要。然而，特征堆叠的典型方法往往不尽如人意：在接触丰富的任务中，视觉等主导模态可能会压垮触觉等稀疏但关键的信号，而单一架构难以灵活地整合新出现或缺失的模态而不重新训练。我们的方法将策略因子化为一系列专门针对单一表示（如视觉或触觉）的扩散模型，并采用一个路由器网络来学习共识权重，以适应性地结合它们的贡献，从而实现新的表示的增量整合。我们在《RLBench》上的模拟操作任务以及实际操作任务（如遮挡物抓取、手持调羹重定位和拼图插入）中评估了该方法，结果表明其在需要多模态推理的情景中显著优于特征堆叠基准。我们的策略还进一步展示了对物理干扰和传感器故障的鲁棒性。我们还进行了基于扰动的重要分析，揭示了不同模态之间的适应性转变。 

---
# Robust Orientation Estimation with TRIAD-aided Manifold EKF 

**Title (ZH)**: TRIAD辅助流形EKF的鲁棒姿态估计 

**Authors**: Arjun Sadananda, Ravi Banavar, Kavi Arya  

**Link**: [PDF](https://arxiv.org/pdf/2509.23456)  

**Abstract**: The manifold extended Kalman filter (Manifold EKF) has found extensive application for attitude determination. Magnetometers employed as sensors for such attitude determination are easily prone to disturbances by their sensitivity to calibration and external magnetic fields. The TRIAD (Tri-Axial Attitude Determination) algorithm is well known as a sub-optimal attitude estimator. In this article, we incorporate this sub-optimal feature of the TRIAD in mitigating the influence of the magnetometer reading in the pitch and roll axis determination in the Manifold EKF algorithm. We substantiate our results with experiments. 

**Abstract (ZH)**: 流形扩展卡尔曼滤波器（Manifold EKF）在姿态确定中得到了广泛应用。作为姿态确定传感器的磁强计容易受到校准敏感性及外部磁场的影响而产生干扰。TRIAD（三轴姿态确定）算法作为次优姿态估计算法而广为人知。本文将TRIAD的次优特性应用于流形扩展卡尔曼滤波器算法中，以减轻磁强计读数对俯仰和滚转轴姿态确定的影响。实验结果证明了我们的方法。 

---
# Space Robotics Bench: Robot Learning Beyond Earth 

**Title (ZH)**: 空间机器人台架：超越地球的机器人学习 

**Authors**: Andrej Orsula, Matthieu Geist, Miguel Olivares-Mendez, Carol Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2509.23328)  

**Abstract**: The growing ambition for space exploration demands robust autonomous systems that can operate in unstructured environments under extreme extraterrestrial conditions. The adoption of robot learning in this domain is severely hindered by the prohibitive cost of technology demonstrations and the limited availability of data. To bridge this gap, we introduce the Space Robotics Bench, an open-source simulation framework for robot learning in space. It offers a modular architecture that integrates on-demand procedural generation with massively parallel simulation environments to support the creation of vast and diverse training distributions for learning-based agents. To ground research and enable direct comparison, the framework includes a comprehensive suite of benchmark tasks that span a wide range of mission-relevant scenarios. We establish performance baselines using standard reinforcement learning algorithms and present a series of experimental case studies that investigate key challenges in generalization, end-to-end learning, adaptive control, and sim-to-real transfer. Our results reveal insights into the limitations of current methods and demonstrate the utility of the framework in producing policies capable of real-world operation. These contributions establish the Space Robotics Bench as a valuable resource for developing, benchmarking, and deploying the robust autonomous systems required for the final frontier. 

**Abstract (ZH)**: 空间机器人模拟平台：面向空间环境的开放源代码机器人学习仿真框架 

---
# GUARD: Toward a Compromise between Traditional Control and Learning for Safe Robot Systems 

**Title (ZH)**: GUARD: 传统控制与学习之间安全机器人系统中的权衡研究 

**Authors**: Johannes A. Gaus, Junheon Yoon, Woo-Jeong Baek, Seungwon Choi, Suhan Park, Jaeheung Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.23312)  

**Abstract**: This paper presents the framework \textbf{GUARD} (\textbf{G}uided robot control via \textbf{U}ncertainty attribution and prob\textbf{A}bilistic kernel optimization for \textbf{R}isk-aware \textbf{D}ecision making) that combines traditional control with an uncertainty-aware perception technique using active learning with real-time capability for safe robot collision avoidance. By doing so, this manuscript addresses the central challenge in robotics of finding a reasonable compromise between traditional methods and learning algorithms to foster the development of safe, yet efficient and flexible applications. By unifying a reactive model predictive countouring control (RMPCC) with an Iterative Closest Point (ICP) algorithm that enables the attribution of uncertainty sources online using active learning with real-time capability via a probabilistic kernel optimization technique, \emph{GUARD} inherently handles the existing ambiguity of the term \textit{safety} that exists in robotics literature. Experimental studies indicate the high performance of \emph{GUARD}, thereby highlighting the relevance and need to broaden its applicability in future. 

**Abstract (ZH)**: GUARD：基于不确定性归因和概率内核优化的风险感知决策引导机器人控制框架 

---
# Distributed Multi-Robot Multi-Target Simultaneous Search and Tracking in an Unknown Non-convex Environment 

**Title (ZH)**: 未知非凸环境中的分布式多机器人多目标同时搜索与跟踪 

**Authors**: Jun Chen, Jiaqing Ma, Philip Dames  

**Link**: [PDF](https://arxiv.org/pdf/2509.23308)  

**Abstract**: In unknown non-convex environments, such as indoor and underground spaces, deploying a fleet of robots to explore the surroundings while simultaneously searching for and tracking targets of interest to maintain high-precision data collection represents a fundamental challenge that urgently requires resolution in applications such as environmental monitoring and rescue operations. Current research has made significant progress in addressing environmental exploration, information search, and target tracking problems, but has yet to establish a framework for simultaneously optimizing these tasks in complex environments. In this paper, we propose a novel motion planning algorithm framework that integrates three control strategies: a frontier-based exploration strategy, a guaranteed coverage strategy based on Lloyd's algorithm, and a sensor-based multi-target tracking strategy. By incorporating these three strategies, the proposed algorithm balances coverage search and high-precision active tracking during exploration. Our approach is validated through a series of MATLAB simulations, demonstrating validity and superiority over standard approaches. 

**Abstract (ZH)**: 在未知非凸环境中，如室内和地下空间，部署机器人队列以同时探索环境、搜索和跟踪目标以维持高精度数据采集，是环境监测和救援操作等领域面临的基本挑战，亟待解决。当前研究在环境探索、信息搜索和目标跟踪方面取得了显著进展，但尚未建立同时优化这些任务的框架。本文提出了一种新的运动规划算法框架，结合了三种控制策略：基于边界的探索策略、基于Lloyd算法的保证覆盖策略以及基于传感器的多目标跟踪策略。通过结合这三种策略，所提出的算法在探索过程中平衡了覆盖搜索和高精度主动跟踪。我们的方法通过一系列MATLAB仿真得到验证，显示出比标准方法的有效性和优越性。 

---
# A Novel Narrow Region Detector for Sampling-Based Planners' Efficiency: Match Based Passage Identifier 

**Title (ZH)**: 基于匹配的通道识别器：一种用于采样基于规划器效率的窄区域检测器 

**Authors**: Yafes Enes Şahiner, Esat Yusuf Gündoğdu, Volkan Sezer  

**Link**: [PDF](https://arxiv.org/pdf/2509.23288)  

**Abstract**: Autonomous technology, which has become widespread today, appears in many different configurations such as mobile robots, manipulators, and drones. One of the most important tasks of these vehicles during autonomous operations is path planning. In the literature, path planners are generally divided into two categories: probabilistic and deterministic methods. In the analysis of probabilistic methods, the common problem of almost all methods is observed in narrow passage environments. In this paper, a novel sampler is proposed that deterministically identifies narrow passage environments using occupancy grid maps and accordingly increases the amount of sampling in these regions. The codes of the algorithm is provided as open source. To evaluate the performance of the algorithm, benchmark studies are conducted in three distinct categories: specific and random simulation environments, and a real-world environment. As a result, it is observed that our algorithm provides higher performance in planning time and number of milestones compared to the baseline samplers. 

**Abstract (ZH)**: 自主技术已广泛应用于多种配置的设备中，如移动机器人、 manipulator 和无人机。这些自主操作的车辆在运行过程中主要任务之一是路径规划。文献中，路径规划方法通常被分为两类：概率性和确定性方法。在对概率性方法的分析中，几乎所有方法在狭窄通道环境下的共同问题得到了观察。本文提出了一种新的采样器，能够确定性地识别狭窄通道环境，并相应地在这些区域增加采样量。该算法的代码已开源。为了评估算法性能，我们在三种不同的类别环境进行了基准研究：特定模拟环境、随机模拟环境和真实环境。结果显示，与基线采样器相比，我们的算法在规划时间和里程碑数量上均表现出更高的性能。 

---
# Preventing Robotic Jailbreaking via Multimodal Domain Adaptation 

**Title (ZH)**: 防止机器人越狱的多模态领域适应方法 

**Authors**: Francesco Marchiori, Rohan Sinha, Christopher Agia, Alexander Robey, George J. Pappas, Mauro Conti, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2509.23281)  

**Abstract**: Large Language Models (LLMs) and Vision-Language Models (VLMs) are increasingly deployed in robotic environments but remain vulnerable to jailbreaking attacks that bypass safety mechanisms and drive unsafe or physically harmful behaviors in the real world. Data-driven defenses such as jailbreak classifiers show promise, yet they struggle to generalize in domains where specialized datasets are scarce, limiting their effectiveness in robotics and other safety-critical contexts. To address this gap, we introduce J-DAPT, a lightweight framework for multimodal jailbreak detection through attention-based fusion and domain adaptation. J-DAPT integrates textual and visual embeddings to capture both semantic intent and environmental grounding, while aligning general-purpose jailbreak datasets with domain-specific reference data. Evaluations across autonomous driving, maritime robotics, and quadruped navigation show that J-DAPT boosts detection accuracy to nearly 100% with minimal overhead. These results demonstrate that J-DAPT provides a practical defense for securing VLMs in robotic applications. Additional materials are made available at: this https URL. 

**Abstract (ZH)**: 大型语言模型(LLMs)和视觉-语言模型(VLMs)在越来越多的机器人环境中得到部署，但仍然容易遭受越狱攻击，这些攻击 bypass 安全机制，导致现实世界中的不安全或物理上有害行为。数据驱动的防御措施如越狱分类器显示出了潜力，但在缺乏专门数据集的领域中难以泛化，限制了其在机器人和其他安全性关键领域的有效性。为了解决这一问题，我们引入了J-DAPT，一种基于注意力融合和领域适应的轻量级多模态越狱检测框架。J-DAPT 结合文本和视觉嵌入，捕捉语义意图和环境基础，同时将通用越狱数据集与特定领域的参考数据对齐。在自动驾驶、海洋机器人和四足导航等领域的评估表明，J-DAPT 在几乎不增加开销的情况下将检测准确性提升至接近100%。这些结果证明了 J-DAPT 提供了在机器人应用中保护 VLMs 的实用防御方法。更多材料参见: this https URL。 

---
# Online Dynamic Goal Recognition in Gym Environments 

**Title (ZH)**: 在线动态目标识别在Gym环境中的研究 

**Authors**: Shamir Matan, Elhadad Osher, Nageris Ben, Mirsky Reuth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23244)  

**Abstract**: Goal Recognition (GR) is the task of inferring an agent's intended goal from partial observations of its behavior, typically in an online and one-shot setting. Despite recent advances in model-free GR, particularly in applications such as human-robot interaction, surveillance, and assistive systems, the field remains fragmented due to inconsistencies in benchmarks, domains, and evaluation protocols.
To address this, we introduce gr-libs (this https URL) and gr-envs (this https URL), two complementary open-source frameworks that support the development, evaluation, and comparison of GR algorithms in Gym-compatible environments. gr-libs includes modular implementations of MDP-based GR baselines, diagnostic tools, and evaluation utilities. gr-envs provides a curated suite of environments adapted for dynamic and goal-directed behavior, along with wrappers that ensure compatibility with standard reinforcement learning toolkits. Together, these libraries offer a standardized, extensible, and reproducible platform for advancing GR research. Both packages are open-source and available on GitHub and PyPI. 

**Abstract (ZH)**: Goal Recognition (从部分观测行为推断智能体的意图目标) 是一项在线且单次完成的任务，旨在从智能体行为的部分观测中推断其意图目标。尽管在无模型Goal Recognition方面取得了近期进展，特别是在人机交互、监视和辅助系统等领域，由于基准、领域和评估协议的一致性问题，该领域仍然存在碎片化现象。

为了解决这个问题，我们引入了gr-libs（<https://github.com/...>）和gr-envs（<https://github.com/...>），两个互补的开源框架，支持在Gym兼容环境中开发、评估和比较Goal Recognition算法。gr-libs包括基于MDP的Goal Recognition基线的模块化实现、诊断工具和评估工具。gr-envs提供了适应动态和目标导向行为的精选环境套件，以及确保与标准强化学习工具包兼容的封装。这些库一并提供了一个标准化、可扩展且可重现的平台，用于推进Goal Recognition研究。两个软件包均开源，并在GitHub和PyPI上提供。 

---
# Leave No Observation Behind: Real-time Correction for VLA Action Chunks 

**Title (ZH)**: 不留任何观测数据 behind: VLA 行动块的实时校正 

**Authors**: Kohei Sendai, Maxime Alvarez, Tatsuya Matsushima, Yutaka Matsuo, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.23224)  

**Abstract**: To improve efficiency and temporal coherence, Vision-Language-Action (VLA) models often predict action chunks; however, this action chunking harms reactivity under inference delay and long horizons. We introduce Asynchronous Action Chunk Correction (A2C2), which is a lightweight real-time chunk correction head that runs every control step and adds a time-aware correction to any off-the-shelf VLA's action chunk. The module combines the latest observation, the predicted action from VLA (base action), a positional feature that encodes the index of the base action within the chunk, and some features from the base policy, then outputs a per-step correction. This preserves the base model's competence while restoring closed-loop responsiveness. The approach requires no retraining of the base policy and is orthogonal to asynchronous execution schemes such as Real Time Chunking (RTC). On the dynamic Kinetix task suite (12 tasks) and LIBERO Spatial, our method yields consistent success rate improvements across increasing delays and execution horizons (+23% point and +7% point respectively, compared to RTC), and also improves robustness for long horizons even with zero injected delay. Since the correction head is small and fast, there is minimal overhead compared to the inference of large VLA models. These results indicate that A2C2 is an effective, plug-in mechanism for deploying high-capacity chunking policies in real-time control. 

**Abstract (ZH)**: Asynchronous Action Chunk Correction for Real-time Vision-Language-Action Models 

---
# SAC-Loco: Safe and Adjustable Compliant Quadrupedal Locomotion 

**Title (ZH)**: SAC-Loco: 安全可调的 compliant 四足行走 

**Authors**: Aoqian Zhang, Zixuan Zhuang, Chunzheng Wang, Shuzhi Sam Ge, Fan Shi, Cheng Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23223)  

**Abstract**: Quadruped robots are designed to achieve agile locomotion by mimicking legged animals. However, existing control methods for quadrupeds often lack one of the key capabilities observed in animals: adaptive and adjustable compliance in response to external disturbances. Most locomotion controllers do not provide tunable compliance and tend to fail under large perturbations. In this work, we propose a switched policy framework for compliant and safe quadruped locomotion. First, we train a force compliant policy with adjustable compliance levels using a teacher student reinforcement learning framework, eliminating the need for explicit force sensing. Next, we develop a safe policy based on the capture point concept to stabilize the robot when the compliant policy fails. Finally, we introduce a recoverability network that predicts the likelihood of failure and switches between the compliant and safe policies. Together, this framework enables quadruped robots to achieve both force compliance and robust safety when subjected to severe external disturbances. 

**Abstract (ZH)**: 四足机器人通过模仿-legged动物的设计来实现敏捷移动。然而，现有的四足机器人控制方法往往缺乏一类关键能力：在面对外部干扰时能够表现出适应性和可调节的柔顺性。大多数运动控制器不具备可调节的柔顺性，容易在遭受较大扰动时失效。本文提出了一种切换策略框架以实现柔顺且安全的四足机器人移动。首先，使用教师学生强化学习框架训练一个具有可调节柔顺性的力柔顺策略，消除显式力感知的需要。其次，基于捕获点概念开发一个安全策略，以在力柔顺策略失效时稳定机器人。最后，引入一个恢复网络预测失败的可能性，并在力柔顺策略和安全策略之间切换。该框架使四足机器人在遭受严重外部干扰时既能实现力柔顺性，又能确保鲁棒安全性。 

---
# GLUE: Global-Local Unified Encoding for Imitation Learning via Key-Patch Tracking 

**Title (ZH)**: GLUE: 全局-局部统一编码在关键patches跟踪下的 imitation 学习 

**Authors**: Ye Chen, Zichen Zhou, Jianyu Dou, Te Cui, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.23220)  

**Abstract**: In recent years, visual representation learning has gained widespread attention in robotic imitation learning. However, in complex Out-of-Distribution(OOD) settings characterized by clutter and occlusion, the attention of global visual representations can be diluted or interfered, leading to degraded policy performance. The invariance of local representations for task-relevant objects offers a solution. By efficiently utilizing these local representations, training and testing data can be mapped to a more similar feature space, thereby mitigating the covariate shift problem. Accordingly, we propose GLUE, a global-local unified encoding framework for imitation learning based on key-patch tracking. GLUE selects and tracks key-patches as critical local representations by employing a text-guided mechanism. It features a novel fusion framework where global patch features query local patches to distill essential information, yielding fine-grained local features with low heterogeneity relative to the global context. This fused representation steers the robot's visual attention toward task-relevant objects and preserves precise global context, which together align the training and testing distributions into a similar and task-informative feature space, ultimately enhancing the robustness of the imitation learning policy. Experiments demonstrate that GLUE achieves strong performance across diverse tasks in both simulation and real-world settings, outperforming the strongest baseline by 17.6% in simulation, 36.3% in real-world environments, and 58.3% on real-world generalization settings. The project website of GLUE is available at this https URL. 

**Abstract (ZH)**: 近年来，视觉表征学习在机器人模仿学习中受到了广泛关注。然而，在由杂乱和遮挡特征的复杂Out-of-Distribution(OOD)设置中，全局视觉表征的关注度可能会被稀释或干扰，导致政策性能下降。任务相关信息对象的局部表征不变性提供了一种解决方案。通过有效地利用这些局部表征，训练和测试数据可以映射到更相似的特征空间，从而缓解协变量转移问题。据此，我们提出了一种基于关键片段跟踪的全局-局部统一编码框架GLUE。GLUE通过采用文本引导机制选择和跟踪关键片段，作为关键局部表征。它具有一个新颖的融合框架，其中全局片段特征查询局部片段以提炼关键信息，生成相对于全局上下文低异质性的精细局部特征。这种融合表示引导机器人将视觉注意力集中于任务相关信息对象，并保留精确的全局上下文，从而使训练和测试分布对齐到一个相似且具有任务信息的特征空间，最终增强模仿学习策略的鲁棒性。实验结果表明，GLUE在模拟和现实世界设置中的多种任务上表现出色，在模拟环境中比最强基线高出17.6%，在现实世界环境中高出36.3%，在现实世界泛化设置上高出58.3%。GLUE项目的官方网站可通过该网址访问。 

---
# Simulated Annealing for Multi-Robot Ergodic Information Acquisition Using Graph-Based Discretization 

**Title (ZH)**: 基于图基 discretization 的多机器人遍历信息获取模拟退火方法 

**Authors**: Benjamin Wong, Aaron Weber, Mohamed M. Safwat, Santosh Devasia, Ashis G. Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.23214)  

**Abstract**: One of the goals of active information acquisition using multi-robot teams is to keep the relative uncertainty in each region at the same level to maintain identical acquisition quality (e.g., consistent target detection) in all the regions. To achieve this goal, ergodic coverage can be used to assign the number of samples according to the quality of observation, i.e., sampling noise levels. However, the noise levels are unknown to the robots. Although this noise can be estimated from samples, the estimates are unreliable at first and can generate fluctuating values. The main contribution of this paper is to use simulated annealing to generate the target sampling distribution, starting from uniform and gradually shifting to an estimated optimal distribution, by varying the coldness parameter of a Boltzmann distribution with the estimated sampling entropy as energy. Simulation results show a substantial improvement of both transient and asymptotic entropy compared to both uniform and direct-ergodic searches. Finally, a demonstration is performed with a TurtleBot swarm system to validate the physical applicability of the algorithm. 

**Abstract (ZH)**: 多机器人团队主动信息获取中的遍历覆盖及其目标采样分布生成 

---
# CE-Nav: Flow-Guided Reinforcement Refinement for Cross-Embodiment Local Navigation 

**Title (ZH)**: CE-Nav: 流向引导的跨体态局部导航强化学习精炼 

**Authors**: Kai Yang, Tianlin Zhang, Zhengbo Wang, Zedong Chu, Xiaolong Wu, Yang Cai, Mu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23203)  

**Abstract**: Generalizing local navigation policies across diverse robot morphologies is a critical challenge. Progress is often hindered by the need for costly and embodiment-specific data, the tight coupling of planning and control, and the "disastrous averaging" problem where deterministic models fail to capture multi-modal decisions (e.g., turning left or right). We introduce CE-Nav, a novel two-stage (IL-then-RL) framework that systematically decouples universal geometric reasoning from embodiment-specific dynamic adaptation. First, we train an embodiment-agnostic General Expert offline using imitation learning. This expert, a conditional normalizing flow model named VelFlow, learns the full distribution of kinematically-sound actions from a large-scale dataset generated by a classical planner, completely avoiding real robot data and resolving the multi-modality issue. Second, for a new robot, we freeze the expert and use it as a guiding prior to train a lightweight, Dynamics-Aware Refiner via online reinforcement learning. This refiner rapidly learns to compensate for the target robot's specific dynamics and controller imperfections with minimal environmental interaction. Extensive experiments on quadrupeds, bipeds, and quadrotors show that CE-Nav achieves state-of-the-art performance while drastically reducing adaptation cost. Successful real-world deployments further validate our approach as an efficient and scalable solution for building generalizable navigation systems. 

**Abstract (ZH)**: 跨多样化机器人形态泛化局部导航策略是一项关键挑战。我们提出了CE-Nav，一种新颖的两阶段（IL-then-RL）框架，系统地解耦了通用几何推理与体现特定的动力学适应。首先，我们使用 imitation learning 在离线模式下训练一个体现无关的通用专家。该专家是一个名为 VelFlow 的条件归一化流模型，从由经典规划器生成的大规模数据集中学习全动作的分布，完全避免了真实机器人数据并解决了多模态问题。其次，对于一个新的机器人，我们冻结专家并将其用作引导先验，通过在线强化学习训练一个轻量级的动力学感知修整器。该修整器能够快速学习补偿目标机器人特定动力学和控制器缺陷，同时减少环境交互。在四足机器人、两足机器人和四旋翼无人机上的 extensively 实验显示，CE-Nav 达到了最先进的性能，大幅减少了适应成本。成功的真实世界部署进一步验证了我们方法作为一种高效且可扩展的通用导航系统构建解决方案的有效性。 

---
# Physically-Feasible Reactive Synthesis for Terrain-Adaptive Locomotion 

**Title (ZH)**: 地形自适应运动的物理可行反应合成 

**Authors**: Ziyi Zhou, Qian Meng, Hadas Kress-Gazit, Ye Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23185)  

**Abstract**: We present an integrated planning framework for quadrupedal locomotion over dynamically changing, unforeseen terrains. Existing methods often depend on heuristics for real-time foothold selection-limiting robustness and adaptability-or rely on computationally intensive trajectory optimization across complex terrains and long horizons. In contrast, our approach combines reactive synthesis for generating correct-by-construction symbolic-level controllers with mixed-integer convex programming (MICP) for dynamic and physically feasible footstep planning during each symbolic transition. To reduce the reliance on costly MICP solves and accommodate specifications that may be violated due to physical infeasibility, we adopt a symbolic repair mechanism that selectively generates only the required symbolic transitions. During execution, real-time MICP replanning based on actual terrain data, combined with runtime symbolic repair and delay-aware coordination, enables seamless bridging between offline synthesis and online operation. Through extensive simulation and hardware experiments, we validate the framework's ability to identify missing locomotion skills and respond effectively in safety-critical environments, including scattered stepping stones and rebar scenarios. 

**Abstract (ZH)**: 一种用于动态变化未预见地形下四足运动综合规划框架 

---
# LAGEA: Language Guided Embodied Agents for Robotic Manipulation 

**Title (ZH)**: 语言引导的实体代理用于机器人操作 

**Authors**: Abdul Monaf Chowdhury, Akm Moshiur Rahman Mazumder, Rabeya Akter, Safaeid Hossain Arib  

**Link**: [PDF](https://arxiv.org/pdf/2509.23155)  

**Abstract**: Robotic manipulation benefits from foundation models that describe goals, but today's agents still lack a principled way to learn from their own mistakes. We ask whether natural language can serve as feedback, an error reasoning signal that helps embodied agents diagnose what went wrong and correct course. We introduce LAGEA (Language Guided Embodied Agents), a framework that turns episodic, schema-constrained reflections from a vision language model (VLM) into temporally grounded guidance for reinforcement learning. LAGEA summarizes each attempt in concise language, localizes the decisive moments in the trajectory, aligns feedback with visual state in a shared representation, and converts goal progress and feedback agreement into bounded, step-wise shaping rewardswhose influence is modulated by an adaptive, failure-aware coefficient. This design yields dense signals early when exploration needs direction and gracefully recedes as competence grows. On the Meta-World MT10 embodied manipulation benchmark, LAGEA improves average success over the state-of-the-art (SOTA) methods by 9.0% on random goals and 5.3% on fixed goals, while converging faster. These results support our hypothesis: language, when structured and grounded in time, is an effective mechanism for teaching robots to self-reflect on mistakes and make better choices. Code will be released soon. 

**Abstract (ZH)**: 基于自然语言反馈的机器人操作改进框架 

---
# EKF-Based Fusion of Wi-Fi/LiDAR/IMU for Indoor Localization and Navigation 

**Title (ZH)**: 基于EKF的 Wi-Fi/LiDAR/IMU 内部定位与导航融合技术 

**Authors**: Zeyi Li, Zhe Tang, Kyeong Soo Kim, Sihao Li, Jeremy S. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.23118)  

**Abstract**: Conventional Wi-Fi received signal strength indicator (RSSI) fingerprinting cannot meet the growing demand for accurate indoor localization and navigation due to its lower accuracy, while solutions based on light detection and ranging (LiDAR) can provide better localization performance but is limited by their higher deployment cost and complexity. To address these issues, we propose a novel indoor localization and navigation framework integrating Wi-Fi RSSI fingerprinting, LiDAR-based simultaneous localization and mapping (SLAM), and inertial measurement unit (IMU) navigation based on an extended Kalman filter (EKF). Specifically, coarse localization by deep neural network (DNN)-based Wi-Fi RSSI fingerprinting is refined by IMU-based dynamic positioning using a Gmapping-based SLAM to generate an occupancy grid map and output high-frequency attitude estimates, which is followed by EKF prediction-update integrating sensor information while effectively suppressing Wi-Fi-induced noise and IMU drift errors. Multi-group real-world experiments conducted on the IR building at Xi'an Jiaotong-Liverpool University demonstrates that the proposed multi-sensor fusion framework suppresses the instability caused by individual approaches and thereby provides stable accuracy across all path configurations with mean two-dimensional (2D) errors ranging from 0.2449 m to 0.3781 m. In contrast, the mean 2D errors of Wi-Fi RSSI fingerprinting reach up to 1.3404 m in areas with severe signal interference, and those of LiDAR/IMU localization are between 0.6233 m and 2.8803 m due to cumulative drift. 

**Abstract (ZH)**: 基于Wi-Fi RSSI特征指纹、LiDARSLAM和IMU的扩展卡尔曼滤波集成的室内定位与导航框架 

---
# FTACT: Force Torque aware Action Chunking Transformer for Pick-and-Reorient Bottle Task 

**Title (ZH)**: FTACT: 带有力和扭矩意识的动作片段变换器用于捡取并重新定向瓶子任务 

**Authors**: Ryo Watanabe, Maxime Alvarez, Pablo Ferreiro, Pavel Savkin, Genki Sano  

**Link**: [PDF](https://arxiv.org/pdf/2509.23112)  

**Abstract**: Manipulator robots are increasingly being deployed in retail environments, yet contact rich edge cases still trigger costly human teleoperation. A prominent example is upright lying beverage bottles, where purely visual cues are often insufficient to resolve subtle contact events required for precise manipulation. We present a multimodal Imitation Learning policy that augments the Action Chunking Transformer with force and torque sensing, enabling end-to-end learning over images, joint states, and forces and torques. Deployed on Ghost, single-arm platform by Telexistence Inc, our approach improves Pick-and-Reorient bottle task by detecting and exploiting contact transitions during pressing and placement. Hardware experiments demonstrate greater task success compared to baseline matching the observation space of ACT as an ablation and experiments indicate that force and torque signals are beneficial in the press and place phases where visual observability is limited, supporting the use of interaction forces as a complementary modality for contact rich skills. The results suggest a practical path to scaling retail manipulation by combining modern imitation learning architectures with lightweight force and torque sensing. 

**Abstract (ZH)**: manipulator机器人在零售环境中越来越多地被部署，但在接触丰富的边缘案例中，仍然会触发成本高昂的人类远程操作。一个典型的例子是直立躺着的饮料瓶，其中纯粹的视觉线索往往不足以解决精确操作所需的微妙接触事件。我们提出了一种多模态模仿学习策略，将动作分块变换器与力和扭矩感知相结合，实现了从图像、关节状态和力及扭矩的端到端学习。该方法部署在Telexistence Inc的单臂平台Ghost上，通过检测和利用压放和放置过程中的接触转换，提高了拿取并重新定向瓶子任务的成功率。硬件实验表明，在视觉可观察性受限的压放阶段，力和扭矩信号有助于提高任务成功率，支持将交互力作为接触丰富技能的补充模态。结果表明，通过结合现代模仿学习架构和轻量级力和扭矩感知，可以实现零售操作的可扩展路径。 

---
# Liaohe-CobotMagic-PnP: an Imitation Learning Dataset of Intelligent Robot for Industrial Applications 

**Title (ZH)**: 辽河-CobotMagic-PnP：面向工业应用的智能机器人 imitation 学习数据集 

**Authors**: Chen Yizhe, Wang Qi, Hu Dongxiao, Jingzhe Fang, Liu Sichao, Zixin An, Hongliang Niu, Haoran Liu, Li Dong, Chuanfen Feng, Lan Dapeng, Liu Yu, Zhibo Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23111)  

**Abstract**: In Industry 4.0 applications, dynamic environmental interference induces highly nonlinear and strongly coupled interactions between the environmental state and robotic behavior. Effectively representing dynamic environmental states through multimodal sensor data fusion remains a critical challenge in current robotic datasets. To address this, an industrial-grade multimodal interference dataset is presented, designed for robotic perception and control under complex conditions. The dataset integrates multi-dimensional interference features including size, color, and lighting variations, and employs high-precision sensors to synchronously collect visual, torque, and joint-state measurements. Scenarios with geometric similarity exceeding 85\% and standardized lighting gradients are included to ensure real-world representativeness. Microsecond-level time-synchronization and vibration-resistant data acquisition protocols, implemented via the Robot Operating System (ROS), guarantee temporal and operational fidelity. Experimental results demonstrate that the dataset enhances model validation robustness and improves robotic operational stability in dynamic, interference-rich environments. The dataset is publicly available at:this https URL. 

**Abstract (ZH)**: 在工业4.0应用中，动态环境干扰引起环境状态与机器人行为之间的高度非线性和强耦合交互。通过多模传感器数据融合有效表示动态环境状态仍然是当前机器人数据集中的关键挑战。为应对这一挑战，提出了一种工业级多模干扰数据集，旨在在复杂条件下用于机器人感知与控制。该数据集整合了大小、颜色和光照变化等多种干扰特征，并采用高精度传感器同步收集视觉、扭矩和关节状态测量数据。包含几何相似度超过85%的场景和标准化的光照梯度，以确保实际环境的代表性和真实性。通过机器人操作系统（ROS）实现微妙级的时间同步和抗振动数据采集协议，确保时间和操作的准确性。实验结果表明，该数据集增强了模型验证的稳健性，并提高了机器人在动态、干扰丰富的环境中的操作稳定性。该数据集已公开 accessible at this https URL。 

---
# Open-Vocabulary Spatio-Temporal Scene Graph for Robot Perception and Teleoperation Planning 

**Title (ZH)**: 具有开放词汇量的空间-时间场景图在机器人感知与远程操作规划中的应用 

**Authors**: Yi Wang, Zeyu Xue, Mujie Liu, Tongqin Zhang, Yan Hu, Zhou Zhao, Chenguang Yang, Zhenyu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23107)  

**Abstract**: Teleoperation via natural-language reduces operator workload and enhances safety in high-risk or remote settings. However, in dynamic remote scenes, transmission latency during bidirectional communication creates gaps between remote perceived states and operator intent, leading to command misunderstanding and incorrect execution. To mitigate this, we introduce the Spatio-Temporal Open-Vocabulary Scene Graph (ST-OVSG), a representation that enriches open-vocabulary perception with temporal dynamics and lightweight latency annotations. ST-OVSG leverages LVLMs to construct open-vocabulary 3D object representations, and extends them into the temporal domain via Hungarian assignment with our temporal matching cost, yielding a unified spatio-temporal scene graph. A latency tag is embedded to enable LVLM planners to retrospectively query past scene states, thereby resolving local-remote state mismatches caused by transmission delays. To further reduce redundancy and highlight task-relevant cues, we propose a task-oriented subgraph filtering strategy that produces compact inputs for the planner. ST-OVSG generalizes to novel categories and enhances planning robustness against transmission latency without requiring fine-tuning. Experiments show that our method achieves 74 percent node accuracy on the Replica benchmark, outperforming ConceptGraph. Notably, in the latency-robustness experiment, the LVLM planner assisted by ST-OVSG achieved a planning success rate of 70.5 percent. 

**Abstract (ZH)**: 自然语言远程操控减少操作员工作负荷并提高高风险或远程环境下的安全性。然而，在动态远程场景中，双向通信過程中的传输延迟会在远程感知状态与操作员意图之间产生差距，导致命令误解和错误执行。为减轻这一问题，我们引入了时空开放词汇场景图（ST-OVSG），该表示法通过引入 temporal 动态和轻量级延迟注释来丰富开放词汇感知。ST-OVSG 利用 LVLM 构建开放词汇 3D 物体表示，并通过我们的时间匹配成本进行匈牙利分配扩展到时间域，从而形成统一的时空场景图。嵌入了延迟标签，以便 LVLM 计划器能够回溯查询过去场景状态，从而解决由传输延迟引起的本地-远程状态不匹配问题。为了进一步减少冗余并突出任务相关线索，我们提出了一种任务导向的子图筛选策略，为计划器生成紧凑的输入。ST-OVSG 可泛化到新的类别，并在无需微调的情况下增强对传输延迟的规划鲁棒性。实验显示，我们的方法在 Replica 基准上的节点准确率达到了 74%，优于 ConceptGraph。在传输延迟鲁棒性实验中，ST-OVSG 辅助的 LVLM 计划器的规划成功率达到了 70.5%。 

---
# In-Hand Manipulation of Articulated Tools with Dexterous Robot Hands with Sim-to-Real Transfer 

**Title (ZH)**: 灵巧机器人手进行有骨架工具的在手操作与仿真实验到实际应用的转移 

**Authors**: Soofiyan Atar, Daniel Huang, Florian Richter, Michael Yip  

**Link**: [PDF](https://arxiv.org/pdf/2509.23075)  

**Abstract**: Reinforcement learning (RL) and sim-to-real transfer have advanced robotic manipulation of rigid objects. Yet, policies remain brittle when applied to articulated mechanisms due to contact-rich dynamics and under-modeled joint phenomena such as friction, stiction, backlash, and clearances. We address this challenge through dexterous in-hand manipulation of articulated tools using a robotic hand with reduced articulation and kinematic redundancy relative to the human hand. Our controller augments a simulation-trained base policy with a sensor-driven refinement learned from hardware demonstrations, conditioning on proprioception and target articulation states while fusing whole-hand tactile and force feedback with the policy's internal action intent via cross-attention-based integration. This design enables online adaptation to instance-specific articulation properties, stabilizes contact interactions, regulates internal forces, and coordinates coupled-link motion under perturbations. We validate our approach across a diversity of real-world examples, including scissors, pliers, minimally invasive surgical tools, and staplers. We achieve robust transfer from simulation to hardware, improved disturbance resilience, and generalization to previously unseen articulated tools, thereby reducing reliance on precise physical modeling in contact-rich settings. 

**Abstract (ZH)**: 基于灵巧的在手操纵的类人手机器人手在附着机制上的强化学习和仿真实验到现实应用 

---
# RAISE: A Robot-Assisted Selective Disassembly and Sorting System for End-of-Life Phones 

**Title (ZH)**: RAISE：一种机器人辅助选择性拆解和分类的废旧手机处理系统 

**Authors**: Chang Liu, Badrinath Balasubramaniam, Neal Yancey, Michael Severson, Adam Shine, Philip Bove, Beiwen Li, Xiao Liang, Minghui Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23048)  

**Abstract**: End-of-Life (EoL) phones significantly exacerbate global e-waste challenges due to their high production volumes and short lifecycles. Disassembly is among the most critical processes in EoL phone recycling. However, it relies heavily on human labor due to product variability. Consequently, the manual process is both labor-intensive and time-consuming. In this paper, we propose a low-cost, easily deployable automated and selective disassembly and sorting system for EoL phones, consisting of three subsystems: an adaptive cutting system, a vision-based robotic sorting system, and a battery removal system. The system can process over 120 phones per hour with an average disassembly success rate of 98.9%, efficiently delivering selected high-value components to downstream processing. It provides a reliable and scalable automated solution to the pressing challenge of EoL phone disassembly. Additionally, the automated system can enhance disassembly economics, converting a previously unprofitable process into one that yields a net profit per unit weight of EoL phones. 

**Abstract (ZH)**: End-of-Life 手机的低成本易部署自动选择性拆解与分类系统显著缓解电子废弃物挑战 

---
# UniPrototype: Humn-Robot Skill Learning with Uniform Prototypes 

**Title (ZH)**: UniPrototype：人类-机器人技能学习的统一原型 

**Authors**: Xiao Hu, Qi Yin, Yangming Shi, Yang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.23021)  

**Abstract**: Data scarcity remains a fundamental challenge in robot learning. While human demonstrations benefit from abundant motion capture data and vast internet resources, robotic manipulation suffers from limited training examples. To bridge this gap between human and robot manipulation capabilities, we propose UniPrototype, a novel framework that enables effective knowledge transfer from human to robot domains via shared motion primitives. ur approach makes three key contributions: (1) We introduce a compositional prototype discovery mechanism with soft assignments, enabling multiple primitives to co-activate and thus capture blended and hierarchical skills; (2) We propose an adaptive prototype selection strategy that automatically adjusts the number of prototypes to match task complexity, ensuring scalable and efficient representation; (3) We demonstrate the effectiveness of our method through extensive experiments in both simulation environments and real-world robotic systems. Our results show that UniPrototype successfully transfers human manipulation knowledge to robots, significantly improving learning efficiency and task performance compared to existing this http URL code and dataset will be released upon acceptance at an anonymous repository. 

**Abstract (ZH)**: 数据稀缺依然是机器人学习中的一个根本性挑战。尽管人类演示可以从丰富的动作捕捉数据和广泛的互联网资源中受益，机器人操控却面临训练样本有限的问题。为缩小人类与机器人操控能力之间的差距，我们提出了UniPrototype，一种新颖的框架，通过共享运动基本要素实现从人类到机器人的有效知识迁移。我们的方法做出了三项关键贡献：(1) 引入了一种具有软指派的组合原型发现机制，使多个基本要素能够协同激活，从而捕捉融合技能和层级技能；(2) 提出了一种自适应原型选择策略，自动调整原型的数量以匹配任务复杂度，确保可扩展和高效的表示；(3) 通过在仿真环境和真实世界机器人系统中的大量实验展示了我们方法的有效性。我们的结果显示，UniPrototype 成功地将人类操控知识转移到机器人上，与现有方法相比，显著提高了学习效率和任务性能。接受投稿后，代码和数据集将匿名发布于公开仓库。 

---
# Safe Task Space Synchronization with Time-Delayed Information 

**Title (ZH)**: 带有时延信息的安全任务空间同步 

**Authors**: Rounak Bhattacharya, Vrithik R. Guthikonda, Ashwin P. Dani  

**Link**: [PDF](https://arxiv.org/pdf/2509.22976)  

**Abstract**: In this paper, an adaptive controller is designed for the synchronization of the trajectory of a robot with unknown kinematics and dynamics to that of the current human trajectory in the task space using the delayed human trajectory information. The communication time delay may be a result of various factors that arise in human-robot collaboration tasks, such as sensor processing or fusion to estimate trajectory/intent, network delays, or computational limitations. The developed adaptive controller uses Barrier Lyapunov Function (BLF) to constrain the Cartesian coordinates of the robot to ensure safety, an ICL-based adaptive law to account for the unknown kinematics, and a gradient-based adaptive law to estimate unknown dynamics. Barrier Lyapunov-Krasovskii (LK) functionals are used for the stability analysis to show that the synchronization and parameter estimation errors remain semi-globally uniformly ultimately bounded (SGUUB). The simulation results based on a human-robot synchronization scenario with time delay are provided to demonstrate the effectiveness of the designed synchronization controller with safety constraints. 

**Abstract (ZH)**: 一种基于延迟人类轨迹信息的未知kinematics和dynamics的机器人轨迹同步自适应控制器设计 

---
# Robot Learning from Any Images 

**Title (ZH)**: 机器人从任意图像学习 

**Authors**: Siheng Zhao, Jiageng Mao, Wei Chow, Zeyu Shangguan, Tianheng Shi, Rong Xue, Yuxi Zheng, Yijia Weng, Yang You, Daniel Seita, Leonidas Guibas, Sergey Zakharov, Vitor Guizilini, Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22970)  

**Abstract**: We introduce RoLA, a framework that transforms any in-the-wild image into an interactive, physics-enabled robotic environment. Unlike previous methods, RoLA operates directly on a single image without requiring additional hardware or digital assets. Our framework democratizes robotic data generation by producing massive visuomotor robotic demonstrations within minutes from a wide range of image sources, including camera captures, robotic datasets, and Internet images. At its core, our approach combines a novel method for single-view physical scene recovery with an efficient visual blending strategy for photorealistic data collection. We demonstrate RoLA's versatility across applications like scalable robotic data generation and augmentation, robot learning from Internet images, and single-image real-to-sim-to-real systems for manipulators and humanoids. Video results are available at this https URL . 

**Abstract (ZH)**: RoLA：一种将任意现实世界图像转换为互动物理-enabled机器人环境的框架 

---
# Hierarchical Control Design for Space Robots with Application to In-Orbit Servicing Missions 

**Title (ZH)**: 空间机器人在轨服务任务的分层控制设计 

**Authors**: Pietro Bruschi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22955)  

**Abstract**: In-Orbit Servicing and Active Debris Removal require advanced robotic capabilities for capturing and detumbling uncooperative targets. This work presents a hierarchical control framework for autonomous robotic capture of tumbling objects in space. A simulation environment is developed, incorporating sloshing dynamics of the chaser, a rarely studied effect in space robotics. The proposed controller combines an inner Lyapunov-based robust control loop for multi-body dynamics with an outer loop addressing an extended inverse kinematics problem. Simulation results show improved robustness and adaptability compared to existing control schemes. 

**Abstract (ZH)**: 在轨服务与主动碎片清除需要先进的机器人技术来捕捉和定姿不合作目标。本文提出了一种分层控制框架，以实现空间中自由旋转物体的自主机器人捕捉。开发了一个仿真环境，包含追逐器的晃动动力学，这是空间机器人学中鲜有研究的效果。所提出的控制器结合了一个基于Lyapunov的内部鲁棒控制环，用于多体动力学，以及一个外部环解决扩展的逆运动学问题。仿真结果表明，与现有控制方案相比，该控制器具有更好的鲁棒性和适应性。 

---
# DBF-MA: A Differential Bayesian Filtering Planner for Multi-Agent Autonomous Racing Overtakes 

**Title (ZH)**: DBF-MA: 一种用于多Agent自主竞速超车的差分贝叶斯过滤规划器 

**Authors**: Trent Weiss, Amar Kulkarni, Madhur Behl  

**Link**: [PDF](https://arxiv.org/pdf/2509.22937)  

**Abstract**: A significant challenge in autonomous racing is to generate overtaking maneuvers. Racing agents must execute these maneuvers on complex racetracks with little room for error. Optimization techniques and graph-based methods have been proposed, but these methods often rely on oversimplified assumptions for collision-avoidance and dynamic constraints. In this work, we present an approach to trajectory synthesis based on an extension of the Differential Bayesian Filtering framework. Our approach for collision-free trajectory synthesis frames the problem as one of Bayesian Inference over the space of Composite Bezier Curves. Our method is derivative-free, does not require a spherical approximation of the vehicle footprint, linearization of constraints, or simplifying upper bounds on collision avoidance. We conduct a closed-loop analysis of DBF-MA and find it successfully overtakes an opponent in 87% of tested scenarios, outperforming existing methods in autonomous overtaking. 

**Abstract (ZH)**: 自主赛车中的一个重大挑战是如何生成超越 maneuvers。赛车代理必须在复杂赛道上执行这些 maneuvers，并且几乎没有错误余地。已经提出了优化技术和图基方法，但这些方法往往依赖于碰撞避免和动力学约束的过度简化假设。在本工作中，我们提出了一种基于差分贝叶斯滤波框架扩展的方法来合成轨迹。我们的碰撞自由轨迹合成方法将问题建模为贝叶斯推理在复合贝塞尔曲线空间中的问题。我们的方法无需导数、不需要车辆足迹的球形近似、无需约束线性化或碰撞避免的简化上界。我们对DBF-MA进行了闭环分析，发现该方法在测试场景中有87%的情况下成功超越对手，优于现有方法在自主超越方面的表现。 

---
# ARMimic: Learning Robotic Manipulation from Passive Human Demonstrations in Augmented Reality 

**Title (ZH)**: ARMimic: 从增强现实中的被动人类示范学习机器人操作 

**Authors**: Rohan Walia, Yusheng Wang, Ralf Römer, Masahiro Nishio, Angela P. Schoellig, Jun Ota  

**Link**: [PDF](https://arxiv.org/pdf/2509.22914)  

**Abstract**: Imitation learning is a powerful paradigm for robot skill acquisition, yet conventional demonstration methods--such as kinesthetic teaching and teleoperation--are cumbersome, hardware-heavy, and disruptive to workflows. Recently, passive observation using extended reality (XR) headsets has shown promise for egocentric demonstration collection, yet current approaches require additional hardware, complex calibration, or constrained recording conditions that limit scalability and usability. We present ARMimic, a novel framework that overcomes these limitations with a lightweight and hardware-minimal setup for scalable, robot-free data collection using only a consumer XR headset and a stationary workplace camera. ARMimic integrates egocentric hand tracking, augmented reality (AR) robot overlays, and real-time depth sensing to ensure collision-aware, kinematically feasible demonstrations. A unified imitation learning pipeline is at the core of our method, treating both human and virtual robot trajectories as interchangeable, which enables policies that generalize across different embodiments and environments. We validate ARMimic on two manipulation tasks, including challenging long-horizon bowl stacking. In our experiments, ARMimic reduces demonstration time by 50% compared to teleoperation and improves task success by 11% over ACT, a state-of-the-art baseline trained on teleoperated data. Our results demonstrate that ARMimic enables safe, seamless, and in-the-wild data collection, offering great potential for scalable robot learning in diverse real-world settings. 

**Abstract (ZH)**: 基于XR头显的轻量级自主数据采集框架ARMimic 

---
# Good Weights: Proactive, Adaptive Dead Reckoning Fusion for Continuous and Robust Visual SLAM 

**Title (ZH)**: 好的权重：主动适配的航位推算融合算法以实现连续可靠的视觉SLAM 

**Authors**: Yanwei Du, Jing-Chen Peng, Patricio A. Vela  

**Link**: [PDF](https://arxiv.org/pdf/2509.22910)  

**Abstract**: Given that Visual SLAM relies on appearance cues for localization and scene understanding, texture-less or visually degraded environments (e.g., plain walls or low lighting) lead to poor pose estimation and track loss. However, robots are typically equipped with sensors that provide some form of dead reckoning odometry with reasonable short-time performance but unreliable long-time performance. The Good Weights (GW) algorithm described here provides a framework to adaptively integrate dead reckoning (DR) with passive visual SLAM for continuous and accurate frame-level pose estimation. Importantly, it describes how all modules in a comprehensive SLAM system must be modified to incorporate DR into its design. Adaptive weighting increases DR influence when visual tracking is unreliable and reduces when visual feature information is strong, maintaining pose track without overreliance on DR. Good Weights yields a practical solution for mobile navigation that improves visual SLAM performance and robustness. Experiments on collected datasets and in real-world deployment demonstrate the benefits of Good Weights. 

**Abstract (ZH)**: 视觉SLAM中基于纹理的适应性融合方法提高移动导航性能和鲁棒性 

---
# Multi-Robot Allocation for Information Gathering in Non-Uniform Spatiotemporal Environments 

**Title (ZH)**: 非均匀时空环境中的多机器人信息采集分配 

**Authors**: Kaleb Ben Naveed, Haejoon Lee, Dimitra Panagou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22883)  

**Abstract**: Autonomous robots are increasingly deployed to estimate spatiotemporal fields (e.g., wind, temperature, gas concentration) that vary across space and time. We consider environments divided into non-overlapping regions with distinct spatial and temporal dynamics, termed non-uniform spatiotemporal environments. Gaussian Processes (GPs) can be used to estimate these fields. The GP model depends on a kernel that encodes how the field co-varies in space and time, with its spatial and temporal lengthscales defining the correlation. Hence, when these lengthscales are incorrect or do not correspond to the actual field, the estimates of uncertainty can be highly inaccurate. Existing GP methods often assume one global lengthscale or update only periodically; some allow spatial variation but ignore temporal changes. To address these limitations, we propose a two-phase framework for multi-robot field estimation. Phase 1 uses a variogram-driven planner to learn region-specific spatial lengthscales. Phase 2 employs an allocation strategy that reassigns robots based on the current uncertainty, and updates sampling as temporal lengthscales are refined. For encoding uncertainty, we utilize clarity, an information metric from our earlier work. We evaluate the proposed method across diverse environments and provide convergence analysis for spatial lengthscale estimation, along with dynamic regret bounds quantifying the gap to the oracle's allocation sequence. 

**Abstract (ZH)**: 自主机器人在非均匀时空环境下的场估计中得到了越来越广泛的应用。我们考虑将环境划分为不重叠的具有不同时空动态的区域，称为非均匀时空环境。高斯过程（GPs）可以用于估计这些场。GP模型依赖于一个内核，该内核编码了场在时空中的协变关系，其时空长度尺度定义了相关性。因此，当这些长度尺度不正确或不对应于实际场时，不确定性估计可能会非常不准确。现有的GP方法通常假设一个全局长度尺度或仅周期性更新；一些方法允许空间变异性但忽略时间变化。为了解决这些限制，我们提出了一种两阶段框架进行多机器人场估计。第一阶段使用变异函数驱动的规划器学习区域特定的空间长度尺度。第二阶段采用分配策略根据当前不确定性重新分配机器人，并随着时空长度尺度的细化更新采样。为了编码不确定性的信息，我们利用了我们之前工作中提出的清晰度这一信息度量。我们在多种环境中评估了提出的方法，并提供了空间长度尺度估计的收敛性分析，以及衡量到最优分配序列差距的动态遗憾界。 

---
# Empart: Interactive Convex Decomposition for Converting Meshes to Parts 

**Title (ZH)**: Empart: 交互式凸分解 mesh 拆分至部件 

**Authors**: Brandon Vu, Shameek Ganguly, Pushkar Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22847)  

**Abstract**: Simplifying complex 3D meshes is a crucial step in robotics applications to enable efficient motion planning and physics simulation. Common methods, such as approximate convex decomposition, represent a mesh as a collection of simple parts, which are computationally inexpensive to simulate. However, existing approaches apply a uniform error tolerance across the entire mesh, which can result in a sub-optimal trade-off between accuracy and performance. For instance, a robot grasping an object needs high-fidelity geometry in the vicinity of the contact surfaces but can tolerate a coarser simplification elsewhere. A uniform tolerance can lead to excessive detail in non-critical areas or insufficient detail where it's needed most.
To address this limitation, we introduce Empart, an interactive tool that allows users to specify different simplification tolerances for selected regions of a mesh. Our method leverages existing convex decomposition algorithms as a sub-routine but uses a novel, parallelized framework to handle region-specific constraints efficiently. Empart provides a user-friendly interface with visual feedback on approximation error and simulation performance, enabling designers to iteratively refine their decomposition. We demonstrate that our approach significantly reduces the number of convex parts compared to a state-of-the-art method (V-HACD) at a fixed error threshold, leading to substantial speedups in simulation performance. For a robotic pick-and-place task, Empart-generated collision meshes reduced the overall simulation time by 69% compared to a uniform decomposition, highlighting the value of interactive, region-specific simplification for performant robotics applications. 

**Abstract (ZH)**: 简化复杂3D网格是机器人应用中实现高效运动规划和物理模拟的关键步骤。Empart：一种交互式区域自适应简化工具 

---
# Dynamic Buffers: Cost-Efficient Planning for Tabletop Rearrangement with Stacking 

**Title (ZH)**: 动态缓冲区：用于堆叠的桌面上物体重排的成本高效规划 

**Authors**: Arman Barghi, Hamed Hosseini, Seraj Ghasemi, Mehdi Tale Masouleh, Ahmad Kalhor  

**Link**: [PDF](https://arxiv.org/pdf/2509.22828)  

**Abstract**: Rearranging objects in cluttered tabletop environments remains a long-standing challenge in robotics. Classical planners often generate inefficient, high-cost plans by shuffling objects individually and using fixed buffers--temporary spaces such as empty table regions or static stacks--to resolve conflicts. When only free table locations are used as buffers, dense scenes become inefficient, since placing an object can restrict others from reaching their goals and complicate planning. Allowing stacking provides extra buffer capacity, but conventional stacking is static: once an object supports another, the base cannot be moved, which limits efficiency. To overcome these issues, a novel planning primitive called the Dynamic Buffer is introduced. Inspired by human grouping strategies, it enables robots to form temporary, movable stacks that can be transported as a unit. This improves both feasibility and efficiency in dense layouts, and it also reduces travel in large-scale settings where space is abundant. Compared with a state-of-the-art rearrangement planner, the approach reduces manipulator travel cost by 11.89% in dense scenarios with a stationary robot and by 5.69% in large, low-density settings with a mobile manipulator. Practicality is validated through experiments on a Delta parallel robot with a two-finger gripper. These findings establish dynamic buffering as a key primitive for cost-efficient and robust rearrangement planning. 

**Abstract (ZH)**: 在拥挤台面上重新排列对象仍然是机器人技术中的一个长期挑战。引入了一种新的规划原语——动态缓冲，它允许机器人形成可移动的临时堆叠，可以作为一个单位搬运，从而在密集布局中提高可行性和效率，并在空间充裕的大规模环境中减少搬运成本。与最先进的重排规划器相比，在固定机器人和移动 manipulator 的大规模低密度场景中，该方法分别减少了 11.89% 和 5.69% 的操作器搬运成本。通过在具有两指夹持器的 Delta 并联机器人上进行实验，验证了其实用性。这些发现确立了动态缓冲作为成本高效且稳健重排规划的关键原语的地位。 

---
# Parameter Identification of a Differentiable Human Arm Musculoskeletal Model without Deep Muscle EMG Reconstruction 

**Title (ZH)**: 无需深入肌电图重建的可微人类手臂肌骨模型参数识别 

**Authors**: Philip Sanderink, Yingfan Zhou, Shuzhen Luo, Cheng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22825)  

**Abstract**: Accurate parameter identification of a subject-specific human musculoskeletal model is crucial to the development of safe and reliable physically collaborative robotic systems, for instance, assistive exoskeletons. Electromyography (EMG)-based parameter identification methods have demonstrated promising performance for personalized musculoskeletal modeling, whereas their applicability is limited by the difficulty of measuring deep muscle EMGs invasively. Although several strategies have been proposed to reconstruct deep muscle EMGs or activations for parameter identification, their reliability and robustness are limited by assumptions about the deep muscle behavior. In this work, we proposed an approach to simultaneously identify the bone and superficial muscle parameters of a human arm musculoskeletal model without reconstructing the deep muscle EMGs. This is achieved by only using the least-squares solution of the deep muscle forces to calculate a loss gradient with respect to the model parameters for identifying them in a framework of differentiable optimization. The results of extensive comparative simulations manifested that our proposed method can achieve comparable estimation accuracy compared to a similar method, but with all the muscle EMGs available. 

**Abstract (ZH)**: 基于最少二乘解的人体上肢 musculoskeletal 模型骨和浅表肌肉参数的同时精准识别 

---
# Teleoperator-Aware and Safety-Critical Adaptive Nonlinear MPC for Shared Autonomy in Obstacle Avoidance of Legged Robots 

**Title (ZH)**: 面向遥控操作员的安全关键自适应非线性模型预测控制在腿式机器人障碍避险中的共享自主控制 

**Authors**: Ruturaj Sambhus, Muneeb Ahmad, Basit Muhammad Imran, Sujith Vijayan, Dylan P. Losey, Kaveh Akbari Hamed  

**Link**: [PDF](https://arxiv.org/pdf/2509.22815)  

**Abstract**: Ensuring safe and effective collaboration between humans and autonomous legged robots is a fundamental challenge in shared autonomy, particularly for teleoperated systems navigating cluttered environments. Conventional shared-control approaches often rely on fixed blending strategies that fail to capture the dynamics of legged locomotion and may compromise safety. This paper presents a teleoperator-aware, safety-critical, adaptive nonlinear model predictive control (ANMPC) framework for shared autonomy of quadrupedal robots in obstacle-avoidance tasks. The framework employs a fixed arbitration weight between human and robot actions but enhances this scheme by modeling the human input with a noisily rational Boltzmann model, whose parameters are adapted online using a projected gradient descent (PGD) law from observed joystick commands. Safety is enforced through control barrier function (CBF) constraints integrated into a computationally efficient NMPC, ensuring forward invariance of safe sets despite uncertainty in human behavior. The control architecture is hierarchical: a high-level CBF-based ANMPC (10 Hz) generates blended human-robot velocity references, a mid-level dynamics-aware NMPC (60 Hz) enforces reduced-order single rigid body (SRB) dynamics to track these references, and a low-level nonlinear whole-body controller (500 Hz) imposes the full-order dynamics via quadratic programming to track the mid-level trajectories. Extensive numerical and hardware experiments, together with a user study, on a Unitree Go2 quadrupedal robot validate the framework, demonstrating real-time obstacle avoidance, online learning of human intent parameters, and safe teleoperator collaboration. 

**Abstract (ZH)**: 确保腿式自主机器人与人类在受电信号操作系统中在杂乱环境下的安全有效协作是共享自主性中的一个基本挑战。本文提出了一种适用于四足机器人避障任务的电信操縱员意识安全关键自适应非线性模型预测控制框架。 

---
# Towards Developing Standards and Guidelines for Robot Grasping and Manipulation Pipelines in the COMPARE Ecosystem 

**Title (ZH)**: 向imulator-based Robot Arms for Manufacturing and Parsing Ecosystem (COMPARE)生态体系中机器人抓取与操作流程标准和指南的发展迈进 

**Authors**: Huajing Zhao, Brian Flynn, Adam Norton, Holly Yanco  

**Link**: [PDF](https://arxiv.org/pdf/2509.22801)  

**Abstract**: The COMPARE Ecosystem aims to improve the compatibility and benchmarking of open-source products for robot manipulation through a series of activities. One such activity is the development of standards and guidelines to specify modularization practices at the component-level for individual modules (e.g., perception, grasp planning, motion planning) and integrations of components that form robot manipulation capabilities at the pipeline-level. This paper briefly reviews our work-in-progress to date to (1) build repositories of open-source products to identify common characteristics of each component in the pipeline, (2) investigate existing modular pipelines to glean best practices, and (3) develop new modular pipelines that advance prior work while abiding by the proposed standards and guidelines. 

**Abstract (ZH)**: COMPARE生态系统的构建旨在通过一系列活动提高开源产品在机器人操作领域的兼容性和基准测试。该生态系统的活动之一是制定标准和指南，以在组件级（如感知、抓取规划、运动规划）和流水线级（包含形成机器人操作能力的组件集成）规定模块化实践。本文简要回顾了至今为止的工作，包括（1）构建开源产品仓库以识别流水线中各个组件的共同特征，（2）调查现有模块化流水线以汲取最佳实践，以及（3）开发新的模块化流水线，这些流水线在遵循提议的标准和指南的同时超越了先前的工作。 

---
# Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving 

**Title (ZH)**: 基于交通规则的持久自回归映射自主驾驶 

**Authors**: Shiyi Liang, Xinyuan Chang, Changjie Wu, Huiyuan Yan, Yifan Bai, Xinran Liu, Hang Zhang, Yujian Yuan, Shuang Zeng, Mu Xu, Xing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.22756)  

**Abstract**: Safe autonomous driving requires both accurate HD map construction and persistent awareness of traffic rules, even when their associated signs are no longer visible. However, existing methods either focus solely on geometric elements or treat rules as temporary classifications, failing to capture their persistent effectiveness across extended driving sequences. In this paper, we present PAMR (Persistent Autoregressive Mapping with Traffic Rules), a novel framework that performs autoregressive co-construction of lane vectors and traffic rules from visual observations. Our approach introduces two key mechanisms: Map-Rule Co-Construction for processing driving scenes in temporal segments, and Map-Rule Cache for maintaining rule consistency across these segments. To properly evaluate continuous and consistent map generation, we develop MapDRv2, featuring improved lane geometry annotations. Extensive experiments demonstrate that PAMR achieves superior performance in joint vector-rule mapping tasks, while maintaining persistent rule effectiveness throughout extended driving sequences. 

**Abstract (ZH)**: 持久交通规则导向的自回归高清地图构建与持续遵守：PAMR框架 

---
# Self-driving cars: Are we there yet? 

**Title (ZH)**: 自动驾驶汽车：我们到了吗？ 

**Authors**: Merve Atasever, Zhuochen Liu, Qingpei Li, Akshay Hitendra Shah, Hans Walker, Jyotirmoy V. Deshmukh, Rahul Jain  

**Link**: [PDF](https://arxiv.org/pdf/2509.22754)  

**Abstract**: Autonomous driving remains a highly active research domain that seeks to enable vehicles to perceive dynamic environments, predict the future trajectories of traffic agents such as vehicles, pedestrians, and cyclists and plan safe and efficient future motions. To advance the field, several competitive platforms and benchmarks have been established to provide standardized datasets and evaluation protocols. Among these, leaderboards by the CARLA organization and nuPlan and the Waymo Open Dataset have become leading benchmarks for assessing motion planning algorithms. Each offers a unique dataset and challenging planning problems spanning a wide range of driving scenarios and conditions. In this study, we present a comprehensive comparative analysis of the motion planning methods featured on these three leaderboards. To ensure a fair and unified evaluation, we adopt CARLA leaderboard v2.0 as our common evaluation platform and modify the selected models for compatibility. By highlighting the strengths and weaknesses of current approaches, we identify prevailing trends, common challenges, and suggest potential directions for advancing motion planning research. 

**Abstract (ZH)**: 自主驾驶仍然是一个高度活跃的研究领域，致力于使车辆能够感知动态环境、预测交通代理（如车辆、行人和骑自行车者）的未来轨迹，并规划安全高效的未来运动。为了推动这一领域的发展，已经建立了多个竞争性的平台和基准，提供了标准化的数据集和评估协议。其中，CARLA组织的排行榜、nuPlan和Waymo Open Dataset已成为评估运动规划算法的主要基准。每个基准提供了独特的数据集和具有广泛驾驶场景和条件的挑战性规划问题。在本研究中，我们对这三个排行榜上展示的运动规划方法进行了全面的比较分析。为了确保公平统一的评估，我们采用CARLA排行榜v2.0作为共同评估平台，并对所选模型进行兼容性修改。通过突出当前方法的优点和不足，我们识别了现有的趋势、共同的挑战，并建议了推进运动规划研究的潜在方向。 

---
# Large Language Models for 3D IC Space Planning 

**Title (ZH)**: 大型语言模型在3D IC空间规划中的应用 

**Authors**: Hung-Ying Chu, Guan-Wei Chen, Shao-Yu Wei, Yu-Cheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.22716)  

**Abstract**: Three-dimensional integrated circuits (3D ICs) have emerged as a promising solution to the scaling limits of two-dimensional designs, offering higher integration density, shorter interconnects, and improved performance. As design complexity increases, effective space planning becomes essential to reduce dead space and ensure layout quality. This study investigates the use of large language models (LLMs) for 3D IC space planning through a post-order slicing tree representation, which guarantees legal space plans while aiming to minimize dead space. Open-source LLMs were fine-tuned on large-scale synthetic datasets and further evaluated on MCNC-derived 3D benchmarks. Experimental results indicate that the proposed framework achieves a favorable balance between runtime efficiency, legality, and dead-space reduction, with zero-dead-space layouts obtained in a significant portion of test cases under practical runtime budgets. Beyond synthetic benchmarks, the method generalizes to MCNC cases such as ami33 and ami49, though larger and irregular instances remain challenging. The approach also shows potential for cross-domain applications, including logistics and 3D object placement, where spatial efficiency is critical. Overall, the results suggest that LLM-based space planning can serve as a data-driven complement to traditional electronic design automation (EDA) methods, providing new insights for scalable 3D layout generation. 

**Abstract (ZH)**: 三维集成电路（3D ICs）的空间规划通过后序切片树表示利用大型语言模型（LLMs）的研究：实现高效的合法性、减少死空间的平衡 

---
# Advancing Audio-Visual Navigation Through Multi-Agent Collaboration in 3D Environments 

**Title (ZH)**: 通过多智能体协作在3D环境中的视听导航技术进步 

**Authors**: Hailong Zhang, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22698)  

**Abstract**: Intelligent agents often require collaborative strategies to achieve complex tasks beyond individual capabilities in real-world scenarios. While existing audio-visual navigation (AVN) research mainly focuses on single-agent systems, their limitations emerge in dynamic 3D environments where rapid multi-agent coordination is critical, especially for time-sensitive applications like emergency response. This paper introduces MASTAVN (Multi-Agent Scalable Transformer Audio-Visual Navigation), a scalable framework enabling two agents to collaboratively localize and navigate toward an audio target in shared 3D environments. By integrating cross-agent communication protocols and joint audio-visual fusion mechanisms, MASTAVN enhances spatial reasoning and temporal synchronization. Through rigorous evaluation in photorealistic 3D simulators (Replica and Matterport3D), MASTAVN achieves significant reductions in task completion time and notable improvements in navigation success rates compared to single-agent and non-collaborative baselines. This highlights the essential role of spatiotemporal coordination in multi-agent systems. Our findings validate MASTAVN's effectiveness in time-sensitive emergency scenarios and establish a paradigm for advancing scalable multi-agent embodied intelligence in complex 3D environments. 

**Abstract (ZH)**: 多agent可扩展变换器音频视觉导航（MASTAVN）：在共享3D环境中的协同定位与导航 

---
# ReSeFlow: Rectifying SE(3)-Equivariant Policy Learning Flows 

**Title (ZH)**: ReSeFlow: 修正SE(3)-等变策略学习流 

**Authors**: Zhitao Wang, Yanke Wang, Jiangtao Wen, Roberto Horowitz, Yuxing Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.22695)  

**Abstract**: Robotic manipulation in unstructured environments requires the generation of robust and long-horizon trajectory-level policy with conditions of perceptual observations and benefits from the advantages of SE(3)-equivariant diffusion models that are data-efficient. However, these models suffer from the inference time costs. Inspired by the inference efficiency of rectified flows, we introduce the rectification to the SE(3)-diffusion models and propose the ReSeFlow, i.e., Rectifying SE(3)-Equivariant Policy Learning Flows, providing fast, geodesic-consistent, least-computational policy generation. Crucially, both components employ SE(3)-equivariant networks to preserve rotational and translational symmetry, enabling robust generalization under rigid-body motions. With the verification on the simulated benchmarks, we find that the proposed ReSeFlow with only one inference step can achieve better performance with lower geodesic distance than the baseline methods, achieving up to a 48.5% error reduction on the painting task and a 21.9% reduction on the rotating triangle task compared to the baseline's 100-step inference. This method takes advantages of both SE(3) equivariance and rectified flow and puts it forward for the real-world application of generative policy learning models with the data and inference efficiency. 

**Abstract (ZH)**: 基于SE(3)-_equivariant扩散模型的快速轨迹级策略学习方法：ReSeFlow及其在刚体运动下的鲁棒泛化 

---
# Nonlinear Model Predictive Control with Single-Shooting Method for Autonomous Personal Mobility Vehicle 

**Title (ZH)**: 基于单步射击方法的自主个人移动车辆的非线性模型预测控制 

**Authors**: Rakha Rahmadani Pratama, Catur Hilman A.H.B. Baskoro, Joga Dharma Setiawan, Dyah Kusuma Dewi, P Paryanto, Mochammad Ariyanto, Roni Permana Saputra  

**Link**: [PDF](https://arxiv.org/pdf/2509.22694)  

**Abstract**: This paper introduces a proposed control method for autonomous personal mobility vehicles, specifically the Single-passenger Electric Autonomous Transporter (SEATER), using Nonlinear Model Predictive Control (NMPC). The proposed method leverages a single-shooting approach to solve the optimal control problem (OCP) via non-linear programming (NLP). The proposed NMPC is implemented to a non-holonomic vehicle with a differential drive system, using odometry data as localization feedback to guide the vehicle towards its target pose while achieving objectives and adhering to constraints, such as obstacle avoidance. To evaluate the performance of the proposed method, a number of simulations have been conducted in both obstacle-free and static obstacle environments. The SEATER model and testing environment have been developed in the Gazebo Simulation and the NMPC are implemented within the Robot Operating System (ROS) framework. The simulation results demonstrate that the NMPC-based approach successfully controls the vehicle to reach the desired target location while satisfying the imposed constraints. Furthermore, this study highlights the robustness and real-time effectiveness of NMPC with a single-shooting approach for autonomous vehicle control in the evaluated scenarios. 

**Abstract (ZH)**: 本文提出了一种用于自主个人机动车辆（单乘客电自主运输器SEATER）的控制方法，采用非线性模型预测控制（NMPC）。所提方法利用单次射击方法通过非线性规划（NLP）解决最优控制问题（OCP）。该所提NMPC被应用于具有差速驱动系统的非完整车辆上，并使用里程计数据作为定位反馈来引导车辆达到目标姿态，同时实现目标并遵守约束条件，例如障碍物回避。为了评估所提方法的性能，在无障碍和静态障碍环境中进行了多项仿真实验。SEATER模型和测试环境在Gazebo仿真中开发，NMPC在Robot Operating System（ROS）框架内实现。仿真实验结果表明，基于NMPC的方法成功地控制了车辆达到期望的目标位置，同时满足了施加的约束条件。此外，本研究强调了采用单次射击方法的NMPC在评估情景中的鲁棒性和实时效果对于自主车辆控制的有效性。 

---
# Mobile Robot Localization via Indoor Positioning System and Odometry Fusion 

**Title (ZH)**: 基于室内定位系统和里程计融合的移动机器人定位 

**Authors**: Muhammad Hafil Nugraha, Fauzi Abdul, Lastiko Bramantyo, Estiko Rijanto, Roni Permana Saputra, Oka Mahendra  

**Link**: [PDF](https://arxiv.org/pdf/2509.22693)  

**Abstract**: Accurate localization is crucial for effectively operating mobile robots in indoor environments. This paper presents a comprehensive approach to mobile robot localization by integrating an ultrasound-based indoor positioning system (IPS) with wheel odometry data via sensor fusion techniques. The fusion methodology leverages the strengths of both IPS and wheel odometry, compensating for the individual limitations of each method. The Extended Kalman Filter (EKF) fusion method combines the data from the IPS sensors and the robot's wheel odometry, providing a robust and reliable localization solution. Extensive experiments in a controlled indoor environment reveal that the fusion-based localization system significantly enhances accuracy and precision compared to standalone systems. The results demonstrate significant improvements in trajectory tracking, with the EKF-based approach reducing errors associated with wheel slippage and sensor noise. 

**Abstract (ZH)**: 准确的定位是有效操作室内移动机器人的重要基础。本文提出了一种将超声波基于的室内定位系统（IPS）与轮式里程计数据通过传感器融合技术结合的综合方法进行移动机器人定位的方案。融合方法利用了IPS和轮式里程计各自的优势，弥补了每种方法的局限性。扩展卡尔曼滤波器（EKF）融合方法结合了IPS传感器和机器人轮式里程计的数据，提供了稳健且可靠的定位解决方案。在受控的室内环境中进行的大量实验表明，基于融合的定位系统在准确性和精度方面显著优于独立系统。结果展示了轨迹跟踪方面的显著改进，基于EKF的方法减少了由车轮打滑和传感器噪声引起的误差。 

---
# Fast Feature Field ($\text{F}^3$): A Predictive Representation of Events 

**Title (ZH)**: 快速特征场 ($\text{F}^3$): 事件的预测表示 

**Authors**: Richeek Das, Kostas Daniilidis, Pratik Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.25146)  

**Abstract**: This paper develops a mathematical argument and algorithms for building representations of data from event-based cameras, that we call Fast Feature Field ($\text{F}^3$). We learn this representation by predicting future events from past events and show that it preserves scene structure and motion information. $\text{F}^3$ exploits the sparsity of event data and is robust to noise and variations in event rates. It can be computed efficiently using ideas from multi-resolution hash encoding and deep sets - achieving 120 Hz at HD and 440 Hz at VGA resolutions. $\text{F}^3$ represents events within a contiguous spatiotemporal volume as a multi-channel image, enabling a range of downstream tasks. We obtain state-of-the-art performance on optical flow estimation, semantic segmentation, and monocular metric depth estimation, on data from three robotic platforms (a car, a quadruped robot and a flying platform), across different lighting conditions (daytime, nighttime), environments (indoors, outdoors, urban, as well as off-road) and dynamic vision sensors (resolutions and event rates). Our implementations can predict these tasks at 25-75 Hz at HD resolution. 

**Abstract (ZH)**: 本文发展了用于事件驱动相机数据表示的数学论证和算法，我们称之为快速特征场（$\text{F}^3$）。通过预测 past 事件中的未来事件来学习这种表示，并展示了其能保留场景结构和运动信息。$\text{F}^3$ 利用事件数据的稀疏性，对噪声和事件率变化具有鲁棒性。它可以通过多分辨率哈希编码和深度集合的理念高效计算，在 HD 分辨率下达到 120 Hz，在 VGA 分辨率下达到 440 Hz。$\text{F}^3$ 将事件以多通道图像的形式表示在连续的时空区域内，便于进行各种下游任务。我们在三类机器人平台（汽车、四足机器人和飞行平台）的数据上，包括不同光照条件（白天、夜间）、环境（室内、室外、城市以及非铺装道路）和动态视觉传感器（分辨率和事件率）上，获得了光流估计、语义分割和单目度量深度估计的前沿性能。我们的实现可以在 HD 分辨率下以 25-75 Hz 的速度预测这些任务。 

---
# Safety-Critical Input-Constrained Nonlinear Intercept Guidance in Multiple Engagement Zones 

**Title (ZH)**: 多作战区内的安全关键输入约束非线性截获制导 

**Authors**: Praveen Kumar Ranjan, Abhinav Sinha, Yongcan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25053)  

**Abstract**: This paper presents an input-constrained nonlinear guidance law to address the problem of intercepting a stationary target in contested environments with multiple defending agents. Contrary to prior approaches that rely on explicit knowledge of defender strategies or utilize conservative safety conditions based on a defender's range, our work characterizes defender threats geometrically through engagement zones that delineate inevitable interception regions. Outside these engagement zones, the interceptor remains invulnerable. The proposed guidance law switches between a repulsive safety maneuver near these zones and a pursuit maneuver outside their influence. To deal with multiple engagement zones, we employ a smooth minimum function (log-sum-exponent approximation) that aggregates threats from all the zones while prioritizing the most critical threats. Input saturation is modeled and embedded in the non-holonomic vehicle dynamics so the controller respects actuator limits while maintaining stability. Numerical simulations with several defenders demonstrate the proposed method's ability to avoid engagement zones and achieve interception across diverse initial conditions. 

**Abstract (ZH)**: 基于输入约束的非线性制导律以应对多防护实体的交战区环境下对静止目标的拦截问题 

---
# When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We? 

**Title (ZH)**: 当自动驾驶车辆遭遇V2X协同感知：我们还有多远？ 

**Authors**: An Guo, Shuoxiao Zhang, Enyi Tang, Xinyu Gao, Haomin Pang, Haoxiang Tian, Yanzhou Mu, Wu Wen, Chunrong Fang, Zhenyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24927)  

**Abstract**: With the tremendous advancement of deep learning and communication technology, Vehicle-to-Everything (V2X) cooperative perception has the potential to address limitations in sensing distant objects and occlusion for a single-agent perception system. V2X cooperative perception systems are software systems characterized by diverse sensor types and cooperative agents, varying fusion schemes, and operation under different communication conditions. Therefore, their complex composition gives rise to numerous operational challenges. Furthermore, when cooperative perception systems produce erroneous predictions, the types of errors and their underlying causes remain insufficiently explored. To bridge this gap, we take an initial step by conducting an empirical study of V2X cooperative perception. To systematically evaluate the impact of cooperative perception on the ego vehicle's perception performance, we identify and analyze six prevalent error patterns in cooperative perception systems. We further conduct a systematic evaluation of the critical components of these systems through our large-scale study and identify the following key findings: (1) The LiDAR-based cooperation configuration exhibits the highest perception performance; (2) Vehicle-to-infrastructure (V2I) and vehicle-to-vehicle (V2V) communication exhibit distinct cooperative perception performance under different fusion schemes; (3) Increased cooperative perception errors may result in a higher frequency of driving violations; (4) Cooperative perception systems are not robust against communication interference when running online. Our results reveal potential risks and vulnerabilities in critical components of cooperative perception systems. We hope that our findings can better promote the design and repair of cooperative perception systems. 

**Abstract (ZH)**: 基于深度学习和通信技术的迅猛发展，Vehicle-to-Everything (V2X)协同感知有望解决单个代理感知系统在感知远距离目标和遮挡方面的局限性。V2X协同感知系统是一种由多种传感器类型、协同代理、不同的融合方案以及在不同通信条件下运行的软件系统，因此其复杂的组成导致了诸多运营挑战。此外，当协同感知系统产生错误预测时，对其错误类型及其根本原因的研究仍显不足。为填补这一空白，我们通过实证研究V2X协同感知系统，按照系统关键组件的大型研究系统性地评估了协同感知对自身感知性能的影响，并得出了以下关键发现：(1) 基于LiDAR的协同配置表现出最高的感知性能；(2) 车辆到基础设施（V2I）和车辆到车辆（V2V）通信在不同的融合方案下表现出不同的协同感知性能；(3) 协同感知错误的增加可能导致驾驶违规频率的提高；(4) 在线运行时，协同感知系统对通信干扰缺乏鲁棒性。我们的研究揭示了协同感知系统关键组件中的潜在风险和脆弱性，我们希望我们的发现能够更好地促进协同感知系统的设计和修复。 

---
# ThermalGen: Style-Disentangled Flow-Based Generative Models for RGB-to-Thermal Image Translation 

**Title (ZH)**: ThermalGen：基于流的风格解耦生成模型用于RGB到热图图像转换 

**Authors**: Jiuhong Xiao, Roshan Nayak, Ning Zhang, Daniel Tortei, Giuseppe Loianno  

**Link**: [PDF](https://arxiv.org/pdf/2509.24878)  

**Abstract**: Paired RGB-thermal data is crucial for visual-thermal sensor fusion and cross-modality tasks, including important applications such as multi-modal image alignment and retrieval. However, the scarcity of synchronized and calibrated RGB-thermal image pairs presents a major obstacle to progress in these areas. To overcome this challenge, RGB-to-Thermal (RGB-T) image translation has emerged as a promising solution, enabling the synthesis of thermal images from abundant RGB datasets for training purposes. In this study, we propose ThermalGen, an adaptive flow-based generative model for RGB-T image translation, incorporating an RGB image conditioning architecture and a style-disentangled mechanism. To support large-scale training, we curated eight public satellite-aerial, aerial, and ground RGB-T paired datasets, and introduced three new large-scale satellite-aerial RGB-T datasets--DJI-day, Bosonplus-day, and Bosonplus-night--captured across diverse times, sensor types, and geographic regions. Extensive evaluations across multiple RGB-T benchmarks demonstrate that ThermalGen achieves comparable or superior translation performance compared to existing GAN-based and diffusion-based methods. To our knowledge, ThermalGen is the first RGB-T image translation model capable of synthesizing thermal images that reflect significant variations in viewpoints, sensor characteristics, and environmental conditions. Project page: this http URL 

**Abstract (ZH)**: 配对的RGB-热成像数据对于视觉-热传感器融合及跨模态任务至关重要，包括多模态图像对齐和检索等重要应用。然而，同步和校准的RGB-热成像配对数据的稀缺性极大地阻碍了这些领域的进展。为了克服这一挑战，RGB到热成像（RGB-T）图像转换已成为一种有前景的解决方案，使人们能够从丰富的RGB数据集中合成热图像以供训练使用。在本研究中，我们提出了ThermalGen，这是一种适应性的基于流的生成模型，用于RGB-T图像转换，结合了RGB图像条件化架构和风格解耦机制。为支持大规模训练，我们汇聚了八个公开的卫星-航空、航空和地面RGB-T配对数据集，并引入了三个新的大型卫星-航空RGB-T数据集——DJI-day、Bosonplus-day和Bosonplus-night，这些数据集在不同的时间、传感器类型和地理区域进行了拍摄。在多个RGB-T基准上的广泛评估表明，ThermalGen在转换性能上达到了与现有GAN基和扩散基方法相当或更优的水平。据我们所知，ThermalGen是首款能够合成反映显著视角变化、传感器特性和环境条件差异的热图像的RGB-T图像转换模型。 

---
# Evaluation of Polarimetric Fusion for Semantic Segmentation in Aquatic Environments 

**Title (ZH)**: 水文环境中极化融合的语义分割评价 

**Authors**: Luis F. W. Batista, Tom Bourbon, Cedric Pradalier  

**Link**: [PDF](https://arxiv.org/pdf/2509.24731)  

**Abstract**: Accurate segmentation of floating debris on water is often compromised by surface glare and changing outdoor illumination. Polarimetric imaging offers a single-sensor route to mitigate water-surface glare that disrupts semantic segmentation of floating objects. We benchmark state-of-the-art fusion networks on PoTATO, a public dataset of polarimetric images of plastic bottles in inland waterways, and compare their performance with single-image baselines using traditional models. Our results indicate that polarimetric cues help recover low-contrast objects and suppress reflection-induced false positives, raising mean IoU and lowering contour error relative to RGB inputs. These sharper masks come at a cost: the additional channels enlarge the models increasing the computational load and introducing the risk of new false positives. By providing a reproducible, diagnostic benchmark and publicly available code, we hope to help researchers choose if polarized cameras are suitable for their applications and to accelerate related research. 

**Abstract (ZH)**: 水面上漂浮垃圾的准确分割常受到水面眩光和变化户外光照的干扰。偏振成像提供了一种通过单传感器方式减轻影响漂浮物语义分割的水面眩光的方法。我们在公开的数据集PoTATO上对最先进的融合网络进行基准测试，并将其性能与使用传统模型的单图像基线进行比较。我们的结果显示，偏振线索有助于恢复低对比度物体并抑制反射引起的假阳性，相对RGB输入提高了平均IoU并降低了轮廓误差。然而，这些更清晰的掩码也存在成本：额外的通道增大了模型，增加了计算负担并引入了新的假阳性风险。通过提供可重复的诊断基准和公开代码，我们希望帮助研究人员判断偏振相机是否适合其应用，并加速相关研究。 

---
# Discrete Variational Autoencoding via Policy Search 

**Title (ZH)**: 离散变分自编码通过策略搜索 

**Authors**: Michael Drolet, Firas Al-Hafez, Aditya Bhatt, Jan Peters, Oleg Arenz  

**Link**: [PDF](https://arxiv.org/pdf/2509.24716)  

**Abstract**: Discrete latent bottlenecks in variational autoencoders (VAEs) offer high bit efficiency and can be modeled with autoregressive discrete distributions, enabling parameter-efficient multimodal search with transformers. However, discrete random variables do not allow for exact differentiable parameterization; therefore, discrete VAEs typically rely on approximations, such as Gumbel-Softmax reparameterization or straight-through gradient estimates, or employ high-variance gradient-free methods such as REINFORCE that have had limited success on high-dimensional tasks such as image reconstruction. Inspired by popular techniques in policy search, we propose a training framework for discrete VAEs that leverages the natural gradient of a non-parametric encoder to update the parametric encoder without requiring reparameterization. Our method, combined with automatic step size adaptation and a transformer-based encoder, scales to challenging datasets such as ImageNet and outperforms both approximate reparameterization methods and quantization-based discrete autoencoders in reconstructing high-dimensional data from compact latent spaces, achieving a 20% improvement on FID Score for ImageNet 256. 

**Abstract (ZH)**: 离散潜瓶颈在变分自编码器中的应用提供了高比特效率，并且可以使用自回归离散分布建模，从而利用变压器实现参数高效的多模态搜索。然而，离散随机变量不允许精确的可微参数化；因此，离散VAE通常依赖于如Gumbel-Softmax重参数化或直接通过梯度估计，或者使用高方差的无梯度方法如REINFORCE，这些方法在高维任务如图像重建上效果有限。受策略搜索中流行技术的启发，我们提出了一种离散VAE的训练框架，利用非参数编码器的自然梯度来更新参数编码器，而不需要重参数化。该方法结合自适应步长调整和基于变压器的编码器，可以扩展到如ImageNet这样的具有挑战性的数据集，并在从紧凑的潜在空间重构高维数据方面优于近似重参数化方法和基于量化的方法，实现了ImageNet 256数据集上FID分数20%的改进。 

---
# SCOPE: Semantic Conditioning for Sim2Real Category-Level Object Pose Estimation in Robotics 

**Title (ZH)**: 语义条件化在机器人领域中的Sim2Real类别级物体姿态估计 

**Authors**: Peter Hönig, Stefan Thalhammer, Jean-Baptiste Weibel, Matthias Hirschmanner, Markus Vincze  

**Link**: [PDF](https://arxiv.org/pdf/2509.24572)  

**Abstract**: Object manipulation requires accurate object pose estimation. In open environments, robots encounter unknown objects, which requires semantic understanding in order to generalize both to known categories and beyond. To resolve this challenge, we present SCOPE, a diffusion-based category-level object pose estimation model that eliminates the need for discrete category labels by leveraging DINOv2 features as continuous semantic priors. By combining these DINOv2 features with photorealistic training data and a noise model for point normals, we reduce the Sim2Real gap in category-level object pose estimation. Furthermore, injecting the continuous semantic priors via cross-attention enables SCOPE to learn canonicalized object coordinate systems across object instances beyond the distribution of known categories. SCOPE outperforms the current state of the art in synthetically trained category-level object pose estimation, achieving a relative improvement of 31.9\% on the 5$^\circ$5cm metric. Additional experiments on two instance-level datasets demonstrate generalization beyond known object categories, enabling grasping of unseen objects from unknown categories with a success rate of up to 100\%. Code available: this https URL. 

**Abstract (ZH)**: 物体操作需要准确的物体姿态估计。在开放环境中，机器人遇到未知物体，这要求进行语义理解以在已知类别和未知类别之间进行泛化。为了解决这一挑战，我们提出了SCOPE，这是一种基于扩散的类别级物体姿态估计模型，通过利用DINOv2特征作为连续的语义先验来消除对离散类别标签的需求。通过将这些DINOv2特征与照片真实感的训练数据和点法线噪声模型相结合，我们缩小了类别级物体姿态估计中的Sim2Real差距。此外，通过交叉注意力注入连续的语义先验使SCOPE能够在已知类别分布之外学习标准化的物体坐标系统。SCOPE在合成训练的类别级物体姿态估计中超越了当前最佳方法，在5°5cm指标上取得了31.9%的相对改进。在两个实例级数据集上的附加实验进一步证明了其在已知物体类别之外的泛化能力，从而使机器人能够从未知类别中抓取未见过的物体，成功率可达100%。代码可获取：this https URL。 

---
# Training Agents Inside of Scalable World Models 

**Title (ZH)**: 在可扩展世界模型内部训练代理 

**Authors**: Danijar Hafner, Wilson Yan, Timothy Lillicrap  

**Link**: [PDF](https://arxiv.org/pdf/2509.24527)  

**Abstract**: World models learn general knowledge from videos and simulate experience for training behaviors in imagination, offering a path towards intelligent agents. However, previous world models have been unable to accurately predict object interactions in complex environments. We introduce Dreamer 4, a scalable agent that learns to solve control tasks by reinforcement learning inside of a fast and accurate world model. In the complex video game Minecraft, the world model accurately predicts object interactions and game mechanics, outperforming previous world models by a large margin. The world model achieves real-time interactive inference on a single GPU through a shortcut forcing objective and an efficient transformer architecture. Moreover, the world model learns general action conditioning from only a small amount of data, allowing it to extract the majority of its knowledge from diverse unlabeled videos. We propose the challenge of obtaining diamonds in Minecraft from only offline data, aligning with practical applications such as robotics where learning from environment interaction can be unsafe and slow. This task requires choosing sequences of over 20,000 mouse and keyboard actions from raw pixels. By learning behaviors in imagination, Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction. Our work provides a scalable recipe for imagination training, marking a step towards intelligent agents. 

**Abstract (ZH)**: Dreamer 4: 一种通过快速准确的世界模型进行强化学习的可扩展代理，实现复杂环境下的物体交互预测和行为训练 

---
# FreeAction: Training-Free Techniques for Enhanced Fidelity of Trajectory-to-Video Generation 

**Title (ZH)**: FreeAction: 无需训练的技术以提高轨迹到视频生成保真度 

**Authors**: Seungwook Kim, Seunghyeon Lee, Minsu Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.24241)  

**Abstract**: Generating realistic robot videos from explicit action trajectories is a critical step toward building effective world models and robotics foundation models. We introduce two training-free, inference-time techniques that fully exploit explicit action parameters in diffusion-based robot video generation. Instead of treating action vectors as passive conditioning signals, our methods actively incorporate them to guide both the classifier-free guidance process and the initialization of Gaussian latents. First, action-scaled classifier-free guidance dynamically modulates guidance strength in proportion to action magnitude, enhancing controllability over motion intensity. Second, action-scaled noise truncation adjusts the distribution of initially sampled noise to better align with the desired motion dynamics. Experiments on real robot manipulation datasets demonstrate that these techniques significantly improve action coherence and visual quality across diverse robot environments. 

**Abstract (ZH)**: 从显式动作轨迹生成真實机器人视频是建立有效世界模型和机器人基础模型的关键步骤。我们介绍两种无需训练、在推理时使用的技巧，充分利用基于扩散的机器人视频生成中的显式动作参数。我们的方法不是将动作向量视为被动的条件信号，而是主动将其整合，以指导无分类器引导过程并初始化高斯潜在变量。首先，动作缩放的无分类器引导动态按动作幅度调整引导强度，增强对运动强度的可控性。其次，动作缩放的噪声截断调整初始采样噪声的分布，以更好地与期望的运动动力学对齐。实验表明，这些技术在多种机器人环境中显著提高了动作连贯性和视觉质量。 

---
# ELHPlan: Efficient Long-Horizon Task Planning for Multi-Agent Collaboration 

**Title (ZH)**: ELHPlan: 效率高的长期任务规划用于多Agent合作 

**Authors**: Shaobin Ling, Yun Wang, Chenyou Fan, Tin Lun Lam, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24230)  

**Abstract**: Large Language Models (LLMs) enable intelligent multi-robot collaboration but face fundamental trade-offs: declarative methods lack adaptability in dynamic environments, while iterative methods incur prohibitive computational costs that scale poorly with team size and task complexity. In this paper, we propose ELHPlan, a novel framework that introduces Action Chains--sequences of actions explicitly bound to sub-goal intentions--as the fundamental planning primitive. ELHPlan operates via a cyclical process: 1) constructing intention-bound action sequences, 2) proactively validating for conflicts and feasibility, 3) refining issues through targeted mechanisms, and 4) executing validated actions. This design balances adaptability and efficiency by providing sufficient planning horizons while avoiding expensive full re-planning. We further propose comprehensive efficiency metrics, including token consumption and planning time, to more holistically evaluate multi-agent collaboration. Our experiments on benchmark TDW-MAT and C-WAH demonstrate that ELHPlan achieves comparable task success rates while consuming only 24% of the tokens required by state-of-the-art methods. Our research establishes a new efficiency-effectiveness frontier for LLM-based multi-agent planning systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）使智能多机器人协作成为可能，但面临根本性的权衡：声明方法在动态环境中缺乏适应性，而迭代方法则伴随着随团队规模和任务复杂性增加而急剧上升的计算成本。本文提出了一种名为ELHPlan的新框架，引入了行动链——明确绑定于子目标意图的动作序列——作为基本的规划原语。ELHPlan通过一个循环过程运作：1) 构建意图绑定的动作序列，2) 主动验证冲突和可行性，3) 通过针对性机制改进问题，4) 执行验证过的动作。这一设计通过提供足够的规划视野来平衡适应性和效率，同时避免昂贵的全面重规划。此外，我们还提出了包括token消耗和规划时间在内的全面效率指标，以更全面地评估多Agent协作。在基准TDW-MAT和C-WAH实验中，ELHPlan实现了与最先进的方法相当的任务成功率，但仅消耗其24%的token。我们的研究为基于LLM的多Agent规划系统设定了一个新的效率-效果前沿。 

---
# Clebsch-Gordan Transformer: Fast and Global Equivariant Attention 

**Title (ZH)**: Clebsch-Gordan 变体变压器：快速且全局 equivariant 注意力 

**Authors**: Owen Lewis Howell, Linfeng Zhao, Xupeng Zhu, Yaoyao Qian, Haojie Huang, Lingfeng Sun, Wil Thomason, Robert Platt, Robin Walters  

**Link**: [PDF](https://arxiv.org/pdf/2509.24093)  

**Abstract**: The global attention mechanism is one of the keys to the success of transformer architecture, but it incurs quadratic computational costs in relation to the number of tokens. On the other hand, equivariant models, which leverage the underlying geometric structures of problem instance, often achieve superior accuracy in physical, biochemical, computer vision, and robotic tasks, at the cost of additional compute requirements. As a result, existing equivariant transformers only support low-order equivariant features and local context windows, limiting their expressiveness and performance. This work proposes Clebsch-Gordan Transformer, achieving efficient global attention by a novel Clebsch-Gordon Convolution on $\SO(3)$ irreducible representations. Our method enables equivariant modeling of features at all orders while achieving ${O}(N \log N)$ input token complexity. Additionally, the proposed method scales well with high-order irreducible features, by exploiting the sparsity of the Clebsch-Gordon matrix. Lastly, we also incorporate optional token permutation equivariance through either weight sharing or data augmentation. We benchmark our method on a diverse set of benchmarks including n-body simulation, QM9, ModelNet point cloud classification and a robotic grasping dataset, showing clear gains over existing equivariant transformers in GPU memory size, speed, and accuracy. 

**Abstract (ZH)**: Clebsch-Gordan Transformer：基于$\SO(3)$不可约表示的新颖Clebsch-Gordan卷积实现高效全局注意力 

---
# Systematic Alias Sampling: an efficient and low-variance way to sample from a discrete distribution 

**Title (ZH)**: 系统化的别名采样：一种高效且低方差的离散分布采样方法 

**Authors**: Ilari Vallivaara, Katja Poikselkä, Pauli Rikula, Juha Röning  

**Link**: [PDF](https://arxiv.org/pdf/2509.24089)  

**Abstract**: In this paper we combine the Alias method with the concept of systematic sampling, a method commonly used in particle filters for efficient low-variance resampling. The proposed method allows very fast sampling from a discrete distribution: drawing k samples is up to an order of magnitude faster than binary search from the cumulative distribution function (cdf) or inversion methods used in many libraries. The produced empirical distribution function is evaluated using a modified Cramér-Von Mises goodness-of-fit statistic, showing that the method compares very favourably to multinomial sampling. As continuous distributions can often be approximated with discrete ones, the proposed method can be used as a very general way to efficiently produce random samples for particle filter proposal distributions, e.g. for motion models in robotics. 

**Abstract (ZH)**: 本文将Alias方法与系统抽样概念结合，用于粒子滤波中的高效低方差重采样。所提出的方法允许从离散分布中进行非常快速的抽样：抽取k个样本的速度比从累积分布函数（CDF）或许多库中使用的倒置方法快一个数量级。通过使用修改后的Cramér-Von Mises拟合优度统计评估生成的经验分布函数，表明该方法与多项式抽样相比具有很大的优势。由于连续分布往往可以用离散分布逼近，所提出的方法可以作为一种非常通用的方法，用于高效地为粒子滤波的提议分布生成随机样本，例如在机器人中的运动模型。 

---
# Gaze Estimation for Human-Robot Interaction: Analysis Using the NICO Platform 

**Title (ZH)**: 基于NICO平台的人类机器人交互注视估计分析 

**Authors**: Matej Palider, Omar Eldardeer, Viktor Kocur  

**Link**: [PDF](https://arxiv.org/pdf/2509.24001)  

**Abstract**: This paper evaluates the current gaze estimation methods within an HRI context of a shared workspace scenario. We introduce a new, annotated dataset collected with the NICO robotic platform. We evaluate four state-of-the-art gaze estimation models. The evaluation shows that the angular errors are close to those reported on general-purpose benchmarks. However, when expressed in terms of distance in the shared workspace the best median error is 16.48 cm quantifying the practical limitations of current methods. We conclude by discussing these limitations and offering recommendations on how to best integrate gaze estimation as a modality in HRI systems. 

**Abstract (ZH)**: 本文在共享工作站场景的HRI背景下评估当前的凝视估计方法。我们引入了一个使用NICO机器人平台收集的新注释数据集。我们评估了四种最先进的凝视估计模型。评估结果显示，角误差接近通用基准上报告的误差。但在以共享工作站中的距离表示时，最佳中位误差为16.48厘米，量化了当前方法的实际限制。最后，我们讨论了这些限制，并提出了关于如何最好地将凝视估计作为HRI系统中的一种模态的建议。 

---
# Advancing Multi-agent Traffic Simulation via R1-Style Reinforcement Fine-Tuning 

**Title (ZH)**: 基于R1风格强化学习微调的多agents交通模拟推进 

**Authors**: Muleilan Pei, Shaoshuai Shi, Shaojie Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23993)  

**Abstract**: Scalable and realistic simulation of multi-agent traffic behavior is critical for advancing autonomous driving technologies. Although existing data-driven simulators have made significant strides in this domain, they predominantly rely on supervised learning to align simulated distributions with real-world driving scenarios. A persistent challenge, however, lies in the distributional shift that arises between training and testing, which often undermines model generalization in unseen environments. To address this limitation, we propose SMART-R1, a novel R1-style reinforcement fine-tuning paradigm tailored for next-token prediction models to better align agent behavior with human preferences and evaluation metrics. Our approach introduces a metric-oriented policy optimization algorithm to improve distribution alignment and an iterative "SFT-RFT-SFT" training strategy that alternates between Supervised Fine-Tuning (SFT) and Reinforcement Fine-Tuning (RFT) to maximize performance gains. Extensive experiments on the large-scale Waymo Open Motion Dataset (WOMD) validate the effectiveness of this simple yet powerful R1-style training framework in enhancing foundation models. The results on the Waymo Open Sim Agents Challenge (WOSAC) showcase that SMART-R1 achieves state-of-the-art performance with an overall realism meta score of 0.7858, ranking first on the leaderboard at the time of submission. 

**Abstract (ZH)**: 适用大规模且真实的多智能体交通行为仿真对于推动自动驾驶技术的发展至关重要。尽管现有的数据驱动仿真器在此领域取得了显著进展，它们主要依赖监督学习来对齐仿真分布与现实驾驶场景。然而，训练与测试之间持续存在的分布偏差往往削弱了模型在未见环境中的泛化能力。为解决这一限制，我们提出SMART-R1，一种针对下一标记预测模型的新型R1风格强化微调范式，以更好地使智能体行为与人类偏好和评估指标保持一致。我们的方法引入了一种以度量为导向的策略优化算法，以提高分布对齐，并提出了一种迭代的“SFT-RFT-SFT”训练策略，交替进行监督微调(SFT)和强化微调(RFT)，以最大化性能提升。大规模Waymo Open Motion Dataset (WOMD)上的广泛实验验证了这种简单而强大的R1风格训练框架在增强基础模型方面的有效性。Waymo Open Sim Agents Challenge (WOSAC)上的结果表明，SMART-R1 达到了最先进的性能，总体现实度meta分为0.7858，在提交时排名领导者榜第一。 

---
# DriveE2E: Closed-Loop Benchmark for End-to-End Autonomous Driving through Real-to-Simulation 

**Title (ZH)**: DriveE2E：从真实到模拟的端到端自动驾驶闭环基准测试 

**Authors**: Haibao Yu, Wenxian Yang, Ruiyang Hao, Chuanye Wang, Jiaru Zhong, Ping Luo, Zaiqing Nie  

**Link**: [PDF](https://arxiv.org/pdf/2509.23922)  

**Abstract**: Closed-loop evaluation is increasingly critical for end-to-end autonomous driving. Current closed-loop benchmarks using the CARLA simulator rely on manually configured traffic scenarios, which can diverge from real-world conditions, limiting their ability to reflect actual driving performance. To address these limitations, we introduce a simple yet challenging closed-loop evaluation framework that closely integrates real-world driving scenarios into the CARLA simulator with infrastructure cooperation. Our approach involves extracting 800 dynamic traffic scenarios selected from a comprehensive 100-hour video dataset captured by high-mounted infrastructure sensors, and creating static digital twin assets for 15 real-world intersections with consistent visual appearance. These digital twins accurately replicate the traffic and environmental characteristics of their real-world counterparts, enabling more realistic simulations in CARLA. This evaluation is challenging due to the diversity of driving behaviors, locations, weather conditions, and times of day at complex urban intersections. In addition, we provide a comprehensive closed-loop benchmark for evaluating end-to-end autonomous driving models. Project URL: \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 闭环评估对于端到端自动驾驶越来越关键。当前使用CARLA仿真器的闭环基准依赖于手动配置的交通场景，这些场景可能与实际情况有偏差，限制了它们反映实际驾驶性能的能力。为解决这些问题，我们引入了一个简单而具有挑战性的闭环评估框架，该框架紧密整合了现实世界的行车场景到CARLA仿真器中，并与基础设施合作。我们的方法包括从高空安装的基础设施传感器记录的100小时视频数据集中提取800个动态交通场景，并为15个现实世界的交叉口创建具有一致视觉外观的静态数字孪生资产。这些数字孪生精确地复制了其真实世界的特性，使CARLA中的模拟更加逼真。由于复杂城市交叉口的驾驶行为、地点、天气状况和时间段的多样性，这一评估具有挑战性。此外，我们还提供了一个全面的闭环基准，用于评估端到端自动驾驶模型。项目网址：\href{this https URL}{这个链接}。 

---
# SIG-Chat: Spatial Intent-Guided Conversational Gesture Generation Involving How, When and Where 

**Title (ZH)**: SIG-Chat: 空间意图导向的对话性手势生成涉及如何、何时和何地 

**Authors**: Yiheng Huang, Junran Peng, Silei Shen, Jingwei Yang, ZeJi Wei, ChenCheng Bai, Yonghao He, Wei Sui, Muyi Sun, Yan Liu, Xu-Cheng Yin, Man Zhang, Zhaoxiang Zhang, Chuanchen Luo  

**Link**: [PDF](https://arxiv.org/pdf/2509.23852)  

**Abstract**: The accompanying actions and gestures in dialogue are often closely linked to interactions with the environment, such as looking toward the interlocutor or using gestures to point to the described target at appropriate moments. Speech and semantics guide the production of gestures by determining their timing (WHEN) and style (HOW), while the spatial locations of interactive objects dictate their directional execution (WHERE). Existing approaches either rely solely on descriptive language to generate motions or utilize audio to produce non-interactive gestures, thereby lacking the characterization of interactive timing and spatial intent. This significantly limits the applicability of conversational gesture generation, whether in robotics or in the fields of game and animation production. To address this gap, we present a full-stack solution. We first established a unique data collection method to simultaneously capture high-precision human motion and spatial intent. We then developed a generation model driven by audio, language, and spatial data, alongside dedicated metrics for evaluating interaction timing and spatial accuracy. Finally, we deployed the solution on a humanoid robot, enabling rich, context-aware physical interactions. 

**Abstract (ZH)**: 伴随对话的互动动作和手势往往紧密关联于与环境的交互，如目光注视对话伙伴或在恰当时刻用手势指向描述的目标。言语和语义指导手势的产生，决定了其时间（WHEN）和风格（HOW），而互动对象的空间位置则决定了其方向性的执行（WHERE）。现有方法要么仅依赖描述性语言生成动作，要么利用音频产生非互动手势，因而缺乏互动时间与空间意图的 characterization。这极大地限制了对话手势生成在机器人技术、游戏制作和动画生产领域的应用。为解决这一问题，我们提出了一套完整的解决方案。我们首先建立了一种独特的数据采集方法，同时捕捉高精度的人体运动和空间意图。然后开发了一个由音频、语言和空间数据驱动的生成模型，并设计了专门的评估指标来衡量互动时间和空间准确性。最后，我们在类人机器人上部署了该方案，使其能够实现丰富、情境感知的物理交互。 

---
# GRS-SLAM3R: Real-Time Dense SLAM with Gated Recurrent State 

**Title (ZH)**: GRS-SLAM3R: 基于门控循环状态的实时密集SLAM 

**Authors**: Guole Shen, Tianchen Deng, Yanbo Wang, Yongtao Chen, Yilin Shen, Jiuming Liu, Jingchuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23737)  

**Abstract**: DUSt3R-based end-to-end scene reconstruction has recently shown promising results in dense visual SLAM. However, most existing methods only use image pairs to estimate pointmaps, overlooking spatial memory and global this http URL this end, we introduce GRS-SLAM3R, an end-to-end SLAM framework for dense scene reconstruction and pose estimation from RGB images without any prior knowledge of the scene or camera parameters. Unlike existing DUSt3R-based frameworks, which operate on all image pairs and predict per-pair point maps in local coordinate frames, our method supports sequentialized input and incrementally estimates metric-scale point clouds in the global coordinate. In order to improve consistent spatial correlation, we use a latent state for spatial memory and design a transformer-based gated update module to reset and update the spatial memory that continuously aggregates and tracks relevant 3D information across frames. Furthermore, we partition the scene into submaps, apply local alignment within each submap, and register all submaps into a common world frame using relative constraints, producing a globally consistent map. Experiments on various datasets show that our framework achieves superior reconstruction accuracy while maintaining real-time performance. 

**Abstract (ZH)**: 基于DUSt3R的端到端场景重建在稠密视觉SLAM中取得了令人 promising 的结果。然而，现有的大多数方法仅使用图像对来估计点图，忽略了空间记忆和全局信息。为此，我们提出了GRS-SLAM3R，这是一种基于RGB图像进行稠密场景重建和姿态估计的端到端SLAM框架，无需任何场景或相机参数先验知识。与现有的基于DUSt3R的框架不同，后者在所有图像对上操作并预测局部坐标系中的点图，我们的方法支持顺序输入，并在全局坐标系中增量地估计米尺度点云。为了提高一致的空间相关性，我们使用隐状态进行空间记忆，并设计了一个变压器基门控更新模块，以重置和更新连续聚合和跟踪各帧相关3D信息的空间记忆。此外，我们将场景划分为子图，对每个子图内部进行局部对齐，并使用相对约束将所有子图注册到一个共同的世界坐标系中，生成全局一致的地图。在各种数据集上的实验表明，我们的框架在保持实时性能的同时实现了更优的重建精度。 

---
# FastViDAR: Real-Time Omnidirectional Depth Estimation via Alternative Hierarchical Attention 

**Title (ZH)**: FastViDAR：基于交替分层注意力的实时全景深度估计 

**Authors**: Hangtian Zhao, Xiang Chen, Yizhe Li, Qianhao Wang, Haibo Lu, Fei Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23733)  

**Abstract**: In this paper we propose FastViDAR, a novel framework that takes four fisheye camera inputs and produces a full $360^\circ$ depth map along with per-camera depth, fusion depth, and confidence estimates. Our main contributions are: (1) We introduce Alternative Hierarchical Attention (AHA) mechanism that efficiently fuses features across views through separate intra-frame and inter-frame windowed self-attention, achieving cross-view feature mixing with reduced overhead. (2) We propose a novel ERP fusion approach that projects multi-view depth estimates to a shared equirectangular coordinate system to obtain the final fusion depth. (3) We generate ERP image-depth pairs using HM3D and 2D3D-S datasets for comprehensive evaluation, demonstrating competitive zero-shot performance on real datasets while achieving up to 20 FPS on NVIDIA Orin NX embedded hardware. Project page: \href{this https URL}{this https URL} 

**Abstract (ZH)**: 本文提出FastViDAR，一种新型框架，利用四个鱼眼摄像头输入生成全视角360°深度图以及每个摄像头的深度、融合深度和置信度估计。我们的主要贡献包括：(1) 引入了替代分层注意力（AHA）机制，通过单独的帧内和帧间窗口自注意力高效融合视图间特征，实现跨视图特征混合并减少开销。(2) 提出了一种新颖的ERP融合方法，将多视角深度估计投影到共享的等角坐标系中以获得最终的融合深度。(3) 使用HM3D和2D3D-S数据集生成ERP图像-深度配对以进行全面评估，展示了在真实数据集上具有竞争力的零样本性能，并在NVIDIA Orin NX嵌入式硬件上实现高达20 FPS。项目页面：[此链接] 

---
# Color-Pair Guided Robust Zero-Shot 6D Pose Estimation and Tracking of Cluttered Objects on Edge Devices 

**Title (ZH)**: 颜色配对引导的鲁棒零样本6D姿态估计与杂乱环境中对象的跟踪 边缘设备 

**Authors**: Xingjian Yang, Ashis G. Banerjee  

**Link**: [PDF](https://arxiv.org/pdf/2509.23647)  

**Abstract**: Robust 6D pose estimation of novel objects under challenging illumination remains a significant challenge, often requiring a trade-off between accurate initial pose estimation and efficient real-time tracking. We present a unified framework explicitly designed for efficient execution on edge devices, which synergizes a robust initial estimation module with a fast motion-based tracker. The key to our approach is a shared, lighting-invariant color-pair feature representation that forms a consistent foundation for both stages. For initial estimation, this feature facilitates robust registration between the live RGB-D view and the object's 3D mesh. For tracking, the same feature logic validates temporal correspondences, enabling a lightweight model to reliably regress the object's motion. Extensive experiments on benchmark datasets demonstrate that our integrated approach is both effective and robust, providing competitive pose estimation accuracy while maintaining high-fidelity tracking even through abrupt pose changes. 

**Abstract (ZH)**: 在挑战性光照下鲁棒的新型对象6D姿态估计仍然是一项重大挑战，往往需要在准确的初始姿态估计和高效的实时跟踪之间做出权衡。我们提出了一种统一框架，专门设计用于边缘设备上的高效执行，该框架将稳健的初始估计模块与快速运动跟踪器相结合。我们方法的核心是一种共享的、光照不变的颜色对特征表示，为两个阶段提供了一致的基础。在初始估计阶段，该特征使得活RGB-D视图与对象的3D网格之间的稳健配准成为可能。在跟踪阶段，相同的特征逻辑验证了时态对应关系，使轻量级模型能够可靠地回归对象的运动。在基准数据集上的广泛实验表明，我们集成的方法既是有效的又是鲁棒的，在通过突然的姿态变化时仍然能够保持高保真跟踪，并提供具有竞争力的姿态估计精度。 

---
# From Static to Dynamic: a Survey of Topology-Aware Perception in Autonomous Driving 

**Title (ZH)**: 从静态到动态：自主驾驶中拓扑感知综述 

**Authors**: Yixiao Chen, Ruining Yang, Xin Chen, Jia He, Dongliang Xu, Yue Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23641)  

**Abstract**: The key to achieving autonomous driving lies in topology-aware perception, the structured understanding of the driving environment with an emphasis on lane topology and road semantics. This survey systematically reviews four core research directions under this theme: vectorized map construction, topological structure modeling, prior knowledge fusion, and language model-based perception. Across these directions, we observe a unifying trend: a paradigm shift from static, pre-built maps to dynamic, sensor-driven perception. Specifically, traditional static maps have provided semantic context for autonomous systems. However, they are costly to construct, difficult to update in real time, and lack generalization across regions, limiting their scalability. In contrast, dynamic representations leverage on-board sensor data for real-time map construction and topology reasoning. Each of the four research directions contributes to this shift through compact spatial modeling, semantic relational reasoning, robust domain knowledge integration, and multimodal scene understanding powered by pre-trained language models. Together, they pave the way for more adaptive, scalable, and explainable autonomous driving systems. 

**Abstract (ZH)**: 实现自动驾驶的关键在于拓扑感知，即以车道拓扑和道路语义为重点的驾驶环境的结构化理解。本文综述了该主题下的四大核心研究方向：矢量地图构建、拓扑结构建模、先验知识融合以及基于语言模型的感知。在这四大方向中，我们观察到一个统一的趋势：从静态、预先构建的地图向基于传感器的动态感知的范式转变。传统静态地图为自主系统提供了语义上下文，但构建成本高、难以实时更新且跨区域缺乏泛化能力，限制了其适用性。相比之下，动态表示利用车载传感器数据实现实时地图构建和拓扑推理。四大研究方向分别通过紧凑的空间建模、语义关系推理、鲁棒领域知识融合以及基于预训练语言模型的多模态场景理解促进这一转变。这些研究共同为更加适应环境、可扩展且可解释的自动驾驶系统铺平了道路。 

---
# Motion Informed Needle Segmentation in Ultrasound Images 

**Title (ZH)**: 超声图像中的运动指导针头分割 

**Authors**: Raghavv Goel, Cecilia Morales, Manpreet Singh, Artur Dubrawski, John Galeotti, Howie Choset  

**Link**: [PDF](https://arxiv.org/pdf/2312.01239)  

**Abstract**: Segmenting a moving needle in ultrasound images is challenging due to the presence of artifacts, noise, and needle occlusion. This task becomes even more demanding in scenarios where data availability is limited. In this paper, we present a novel approach for needle segmentation for 2D ultrasound that combines classical Kalman Filter (KF) techniques with data-driven learning, incorporating both needle features and needle motion. Our method offers three key contributions. First, we propose a compatible framework that seamlessly integrates into commonly used encoder-decoder style architectures. Second, we demonstrate superior performance compared to recent state-of-the-art needle segmentation models using our novel convolutional neural network (CNN) based KF-inspired block, achieving a 15\% reduction in pixel-wise needle tip error and an 8\% reduction in length error. Third, to our knowledge we are the first to implement a learnable filter to incorporate non-linear needle motion for improving needle segmentation. 

**Abstract (ZH)**: 基于经典的卡尔曼滤波器技术和数据驱动学习的2D超声针段化方法 

---
