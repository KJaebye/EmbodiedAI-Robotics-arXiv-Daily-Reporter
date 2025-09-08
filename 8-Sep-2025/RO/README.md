# Robust Model Predictive Control Design for Autonomous Vehicles with Perception-based Observers 

**Title (ZH)**: 基于感知观测器的自主车辆鲁棒模型预测控制设计 

**Authors**: Nariman Niknejad, Gokul S. Sankar, Bahare Kiumarsi, Hamidreza Modares  

**Link**: [PDF](https://arxiv.org/pdf/2509.05201)  

**Abstract**: This paper presents a robust model predictive control (MPC) framework that explicitly addresses the non-Gaussian noise inherent in deep learning-based perception modules used for state estimation. Recognizing that accurate uncertainty quantification of the perception module is essential for safe feedback control, our approach departs from the conventional assumption of zero-mean noise quantification of the perception error. Instead, it employs set-based state estimation with constrained zonotopes to capture biased, heavy-tailed uncertainties while maintaining bounded estimation errors. To improve computational efficiency, the robust MPC is reformulated as a linear program (LP), using a Minkowski-Lyapunov-based cost function with an added slack variable to prevent degenerate solutions. Closed-loop stability is ensured through Minkowski-Lyapunov inequalities and contractive zonotopic invariant sets. The largest stabilizing terminal set and its corresponding feedback gain are then derived via an ellipsoidal approximation of the zonotopes. The proposed framework is validated through both simulations and hardware experiments on an omnidirectional mobile robot along with a camera and a convolutional neural network-based perception module implemented within a ROS2 framework. The results demonstrate that the perception-aware MPC provides stable and accurate control performance under heavy-tailed noise conditions, significantly outperforming traditional Gaussian-noise-based designs in terms of both state estimation error bounding and overall control performance. 

**Abstract (ZH)**: 一种考虑深度学习感知模块非高斯噪声的鲁棒模型预测控制框架 

---
# Shared Autonomy through LLMs and Reinforcement Learning for Applications to Ship Hull Inspections 

**Title (ZH)**: 通过大规模语言模型和强化学习实现自主共享：应用于船体检查域 

**Authors**: Cristiano Caissutti, Estelle Gerbier, Ehsan Khorrambakht, Paolo Marinelli, Andrea Munafo', Andrea Caiti  

**Link**: [PDF](https://arxiv.org/pdf/2509.05042)  

**Abstract**: Shared autonomy is a promising paradigm in robotic systems, particularly within the maritime domain, where complex, high-risk, and uncertain environments necessitate effective human-robot collaboration. This paper investigates the interaction of three complementary approaches to advance shared autonomy in heterogeneous marine robotic fleets: (i) the integration of Large Language Models (LLMs) to facilitate intuitive high-level task specification and support hull inspection missions, (ii) the implementation of human-in-the-loop interaction frameworks in multi-agent settings to enable adaptive and intent-aware coordination, and (iii) the development of a modular Mission Manager based on Behavior Trees to provide interpretable and flexible mission control. Preliminary results from simulation and real-world lake-like environments demonstrate the potential of this multi-layered architecture to reduce operator cognitive load, enhance transparency, and improve adaptive behaviour alignment with human intent. Ongoing work focuses on fully integrating these components, refining coordination mechanisms, and validating the system in operational port scenarios. This study contributes to establishing a modular and scalable foundation for trustworthy, human-collaborative autonomy in safety-critical maritime robotics applications. 

**Abstract (ZH)**: 共享自主权是一种在机器人系统中，特别是在海洋领域内具有前景的范式，特别是在复杂、高风险和不确定的环境中，它促进了有效的人机协作。本文探讨了三种互补方法在异构海洋机器人舰队中推进共享自主权的交互：（i）通过集成大型语言模型（LLMs）来促进直观的高级任务规范并支持船体检查任务，（ii）在多agent设置中实现具有人类在环路交互框架的机制，以实现适应性和意图感知的协调，（iii）基于行为树开发模块化任务管理器，以提供可解释和灵活的任务控制。模拟和类似湖泊的真实环境的初步结果表明，这种多层结构的潜力在于减少操作员的认知负担、提高透明度并改善与人类意图一致的适应性行为对齐。正在进行的工作集中在将这些组件完全集成在一起、细化协调机制并在运营港口场景中验证系统。本研究为在安全关键的海洋机器人应用中建立模块化和可扩展的可信赖人机协作基础奠定了基础。 

---
# Pointing-Guided Target Estimation via Transformer-Based Attention 

**Title (ZH)**: 基于Transformer注意力的指针引导目标估计 

**Authors**: Luca Müller, Hassan Ali, Philipp Allgeuer, Lukáš Gajdošech, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2509.05031)  

**Abstract**: Deictic gestures, like pointing, are a fundamental form of non-verbal communication, enabling humans to direct attention to specific objects or locations. This capability is essential in Human-Robot Interaction (HRI), where robots should be able to predict human intent and anticipate appropriate responses. In this work, we propose the Multi-Modality Inter-TransFormer (MM-ITF), a modular architecture to predict objects in a controlled tabletop scenario with the NICOL robot, where humans indicate targets through natural pointing gestures. Leveraging inter-modality attention, MM-ITF maps 2D pointing gestures to object locations, assigns a likelihood score to each, and identifies the most likely target. Our results demonstrate that the method can accurately predict the intended object using monocular RGB data, thus enabling intuitive and accessible human-robot collaboration. To evaluate the performance, we introduce a patch confusion matrix, providing insights into the model's predictions across candidate object locations. Code available at: this https URL. 

**Abstract (ZH)**: 指示手势（如指指动作）是基本的非言语交流形式，使人类能够将注意力导向特定的对象或位置。这种能力在人机交互（HRI）中至关重要，其中机器人应该能够预测人类的意图并预见到适当的回应。在此工作中，我们提出了多模态交互变压器（MM-ITF），一种模块化架构，用于在NICOL机器人控制的台面场景中预测对象，其中人类通过自然的手势指向目标。利用跨模态注意力，MM-ITF将二维的手势映射到对象位置，对每个位置分配似然性分数，并确定最有可能的目标。我们的结果表明，该方法可以使用单目RGB数据准确预测所指对象，从而实现直观且易于实现的人机合作。为了评估性能，我们引入了一种补丁混淆矩阵，提供了模型在候选对象位置上预测的见解。相关代码可在以下链接获取：this https URL。 

---
# FLOWER: Democratizing Generalist Robot Policies with Efficient Vision-Language-Action Flow Policies 

**Title (ZH)**: FLOWER: 通过高效视觉-语言-动作流策略实现通用机器人政策的民主化 

**Authors**: Moritz Reuss, Hongyi Zhou, Marcel Rühle, Ömer Erdinç Yağmurlu, Fabian Otto, Rudolf Lioutikov  

**Link**: [PDF](https://arxiv.org/pdf/2509.04996)  

**Abstract**: Developing efficient Vision-Language-Action (VLA) policies is crucial for practical robotics deployment, yet current approaches face prohibitive computational costs and resource requirements. Existing diffusion-based VLA policies require multi-billion-parameter models and massive datasets to achieve strong performance. We tackle this efficiency challenge with two contributions: intermediate-modality fusion, which reallocates capacity to the diffusion head by pruning up to $50\%$ of LLM layers, and action-specific Global-AdaLN conditioning, which cuts parameters by $20\%$ through modular adaptation. We integrate these advances into a novel 950 M-parameter VLA called FLOWER. Pretrained in just 200 H100 GPU hours, FLOWER delivers competitive performance with bigger VLAs across $190$ tasks spanning ten simulation and real-world benchmarks and demonstrates robustness across diverse robotic embodiments. In addition, FLOWER achieves a new SoTA of 4.53 on the CALVIN ABC benchmark. Demos, code and pretrained weights are available at this https URL. 

**Abstract (ZH)**: 开发高效的视觉-语言-行动（VLA）策略对于实际机器人部署至关重要，但当前方法面临巨大的计算成本和资源要求。基于扩散的VLA策略需要多百亿参数的模型和大量数据才能实现出色的性能。我们通过两项贡献来应对效率挑战：中间模态融合，通过剪枝最多50%的LLM层从而重新分配给扩散头部的能力；以及针对特定行动的全局-AdaLN调节，通过模块化适应减少20%的参数。我们将这些进展整合到一个新的950M参数VLA模型FLOWER中。FLOWER仅在200个H100 GPU小时内进行预训练，即可在涵盖十个模拟和现实世界基准的190个任务中与更大的VLA模型实现竞争力的性能表现，并在多种机器人实体中展现出鲁棒性。此外，FLOWER在CALVIN ABC基准上的得分达到新的SOTA水平4.53。相关演示、代码和预训练权重可在以下链接获取。 

---
# Lyapunov-Based Deep Learning Control for Robots with Unknown Jacobian 

**Title (ZH)**: 基于Lyapunov的深度学习控制方法用于未知雅各比矩阵的机器人 

**Authors**: Koji Matsuno, Chien Chern Cheah  

**Link**: [PDF](https://arxiv.org/pdf/2509.04984)  

**Abstract**: Deep learning, with its exceptional learning capabilities and flexibility, has been widely applied in various applications. However, its black-box nature poses a significant challenge in real-time robotic applications, particularly in robot control, where trustworthiness and robustness are critical in ensuring safety. In robot motion control, it is essential to analyze and ensure system stability, necessitating the establishment of methodologies that address this need. This paper aims to develop a theoretical framework for end-to-end deep learning control that can be integrated into existing robot control theories. The proposed control algorithm leverages a modular learning approach to update the weights of all layers in real time, ensuring system stability based on Lyapunov-like analysis. Experimental results on industrial robots are presented to illustrate the performance of the proposed deep learning controller. The proposed method offers an effective solution to the black-box problem in deep learning, demonstrating the possibility of deploying real-time deep learning strategies for robot kinematic control in a stable manner. This achievement provides a critical foundation for future advancements in deep learning based real-time robotic applications. 

**Abstract (ZH)**: 基于深度学习的端到端控制理论框架及其在机器人运动控制中的应用 

---
# DeGuV: Depth-Guided Visual Reinforcement Learning for Generalization and Interpretability in Manipulation 

**Title (ZH)**: Depth-Guided 视觉强化学习在操作中的泛化与可解释性 

**Authors**: Tien Pham, Xinyun Chi, Khang Nguyen, Manfred Huber, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04970)  

**Abstract**: Reinforcement learning (RL) agents can learn to solve complex tasks from visual inputs, but generalizing these learned skills to new environments remains a major challenge in RL application, especially robotics. While data augmentation can improve generalization, it often compromises sample efficiency and training stability. This paper introduces DeGuV, an RL framework that enhances both generalization and sample efficiency. In specific, we leverage a learnable masker network that produces a mask from the depth input, preserving only critical visual information while discarding irrelevant pixels. Through this, we ensure that our RL agents focus on essential features, improving robustness under data augmentation. In addition, we incorporate contrastive learning and stabilize Q-value estimation under augmentation to further enhance sample efficiency and training stability. We evaluate our proposed method on the RL-ViGen benchmark using the Franka Emika robot and demonstrate its effectiveness in zero-shot sim-to-real transfer. Our results show that DeGuV outperforms state-of-the-art methods in both generalization and sample efficiency while also improving interpretability by highlighting the most relevant regions in the visual input 

**Abstract (ZH)**: 基于深度可变掩码的强化学习框架：提升泛化能力和样本效率 

---
# Ground-Aware Octree-A* Hybrid Path Planning for Memory-Efficient 3D Navigation of Ground Vehicles 

**Title (ZH)**: 基于地面感知的八叉树-A*混合路径规划方法及其在地面车辆记忆高效3D导航中的应用 

**Authors**: Byeong-Il Ham, Hyun-Bin Kim, Kyung-Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.04950)  

**Abstract**: In this paper, we propose a 3D path planning method that integrates the A* algorithm with the octree structure. Unmanned Ground Vehicles (UGVs) and legged robots have been extensively studied, enabling locomotion across a variety of terrains. Advances in mobility have enabled obstacles to be regarded not only as hindrances to be avoided, but also as navigational aids when beneficial. A modified 3D A* algorithm generates an optimal path by leveraging obstacles during the planning process. By incorporating a height-based penalty into the cost function, the algorithm enables the use of traversable obstacles to aid locomotion while avoiding those that are impassable, resulting in more efficient and realistic path generation. The octree-based 3D grid map achieves compression by merging high-resolution nodes into larger blocks, especially in obstacle-free or sparsely populated areas. This reduces the number of nodes explored by the A* algorithm, thereby improving computational efficiency and memory usage, and supporting real-time path planning in practical environments. Benchmark results demonstrate that the use of octree structure ensures an optimal path while significantly reducing memory usage and computation time. 

**Abstract (ZH)**: 本文提出了一种结合A*算法和八叉树结构的3D路径规划方法。无人驾驶地面车辆（UGVs）和腿式机器人已被广泛研究，使它们能够在多种地形上移动。移动性的进步使障碍物不仅可以被视为需要避免的阻碍，还可以在有益时作为导航辅助。修改后的3D A*算法在规划过程中利用障碍物生成最优路径。通过将高度相关的惩罚纳入成本函数中，算法能够在利用可通行障碍物辅助移动的同时避开不可通行的障碍物，从而生成更高效、更真实的路径。基于八叉树的3D网格地图通过在无障碍或稀疏区域将高分辨率节点合并为较大块体实现压缩，从而减少A*算法探索的节点数量，提高计算效率和内存使用率，并支持实际环境中的实时路径规划。基准测试结果表明，使用八叉树结构可确保最优路径并大幅减少内存使用和计算时间。 

---
# Towards an Accurate and Effective Robot Vision (The Problem of Topological Localization for Mobile Robots) 

**Title (ZH)**: 朝着准确有效的机器人视觉：移动机器人拓扑定位问题的研究 

**Authors**: Emanuela Boros  

**Link**: [PDF](https://arxiv.org/pdf/2509.04948)  

**Abstract**: Topological localization is a fundamental problem in mobile robotics, since robots must be able to determine their position in order to accomplish tasks. Visual localization and place recognition are challenging due to perceptual ambiguity, sensor noise, and illumination variations. This work addresses topological localization in an office environment using only images acquired with a perspective color camera mounted on a robot platform, without relying on temporal continuity of image sequences. We evaluate state-of-the-art visual descriptors, including Color Histograms, SIFT, ASIFT, RGB-SIFT, and Bag-of-Visual-Words approaches inspired by text retrieval. Our contributions include a systematic, quantitative comparison of these features, distance measures, and classifiers. Performance was analyzed using standard evaluation metrics and visualizations, extending previous experiments. Results demonstrate the advantages of proper configurations of appearance descriptors, similarity measures, and classifiers. The quality of these configurations was further validated in the Robot Vision task of the ImageCLEF evaluation campaign, where the system identified the most likely location of novel image sequences. Future work will explore hierarchical models, ranking methods, and feature combinations to build more robust localization systems, reducing training and runtime while avoiding the curse of dimensionality. Ultimately, this aims toward integrated, real-time localization across varied illumination and longer routes. 

**Abstract (ZH)**: 基于视角彩色摄像头的办公环境拓扑定位研究 

---
# A Knowledge-Driven Diffusion Policy for End-to-End Autonomous Driving Based on Expert Routing 

**Title (ZH)**: 基于专家路径规划的知识驱动扩散策略实现端到端自动驾驶 

**Authors**: Chengkai Xu, Jiaqi Liu, Yicheng Guo, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.04853)  

**Abstract**: End-to-end autonomous driving remains constrained by the need to generate multi-modal actions, maintain temporal stability, and generalize across diverse scenarios. Existing methods often collapse multi-modality, struggle with long-horizon consistency, or lack modular adaptability. This paper presents KDP, a knowledge-driven diffusion policy that integrates generative diffusion modeling with a sparse mixture-of-experts routing mechanism. The diffusion component generates temporally coherent and multi-modal action sequences, while the expert routing mechanism activates specialized and reusable experts according to context, enabling modular knowledge composition. Extensive experiments across representative driving scenarios demonstrate that KDP achieves consistently higher success rates, reduced collision risk, and smoother control compared to prevailing paradigms. Ablation studies highlight the effectiveness of sparse expert activation and the Transformer backbone, and activation analyses reveal structured specialization and cross-scenario reuse of experts. These results establish diffusion with expert routing as a scalable and interpretable paradigm for knowledge-driven end-to-end autonomous driving. 

**Abstract (ZH)**: 知识驱动的扩散政策：基于生成性扩散建模和稀疏专家路由机制的端到端自主驾驶 

---
# COMMET: A System for Human-Induced Conflicts in Mobile Manipulation of Everyday Tasks 

**Title (ZH)**: COMMET：一种用于移动执行日常任务中人工引发冲突的系统 

**Authors**: Dongping Li, Shaoting Peng, John Pohovey, Katherine Rose Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2509.04836)  

**Abstract**: Continuous advancements in robotics and AI are driving the integration of robots from industry into everyday environments. However, dynamic and unpredictable human activities in daily lives would directly or indirectly conflict with robot actions. Besides, due to the social attributes of such human-induced conflicts, solutions are not always unique and depend highly on the user's personal preferences. To address these challenges and facilitate the development of household robots, we propose COMMET, a system for human-induced COnflicts in Mobile Manipulation of Everyday Tasks. COMMET employs a hybrid detection approach, which begins with multi-modal retrieval and escalates to fine-tuned model inference for low-confidence cases. Based on collected user preferred options and settings, GPT-4o will be used to summarize user preferences from relevant cases. In preliminary studies, our detection module shows better accuracy and latency compared with GPT models. To facilitate future research, we also design a user-friendly interface for user data collection and demonstrate an effective workflow for real-world deployments. 

**Abstract (ZH)**: 持续的机器人与AI进步推动了工业机器人向日常生活环境的集成。然而，日常生活中动态且不可预测的人类活动会直接或间接地与机器人行动产生冲突。由于此类人类引发冲突的社会属性，解决方案并非总是唯一的，高度依赖用户的个人偏好。为应对这些挑战并促进家庭机器人的发展，我们提出了COMMET系统，这是一个针对移动操作日常任务中人类引发冲突的系统。COMMET采用混合检测方法，始于多模态检索，并在低置信度情况下升级为细调模型推理。基于收集的用户偏好选项和设置，将使用GPT-4o总结相关案例中的用户偏好。初步研究表明，我们的检测模块在准确性和延迟方面优于GPT模型。为了促进未来研究，我们还设计了一个用户友好的界面以收集用户数据，并展示了适用于实际部署的有效工作流。 

---
# Imitation Learning Based on Disentangled Representation Learning of Behavioral Characteristics 

**Title (ZH)**: 基于行为特征解耦表示的学习模仿 

**Authors**: Ryoga Oishi, Sho Sakaino, Toshiaki Tsuji  

**Link**: [PDF](https://arxiv.org/pdf/2509.04737)  

**Abstract**: In the field of robot learning, coordinating robot actions through language instructions is becoming increasingly feasible. However, adapting actions to human instructions remains challenging, as such instructions are often qualitative and require exploring behaviors that satisfy varying conditions. This paper proposes a motion generation model that adapts robot actions in response to modifier directives human instructions imposing behavioral conditions during task execution. The proposed method learns a mapping from modifier directives to actions by segmenting demonstrations into short sequences, assigning weakly supervised labels corresponding to specific modifier types. We evaluated our method in wiping and pick and place tasks. Results show that it can adjust motions online in response to modifier directives, unlike conventional batch-based methods that cannot adapt during execution. 

**Abstract (ZH)**: 基于语言指令协调机器人动作生成模型 

---
# Hierarchical Reduced-Order Model Predictive Control for Robust Locomotion on Humanoid Robots 

**Title (ZH)**: humanoid机器人稳健分级降阶模型预测控制 

**Authors**: Adrian B. Ghansah, Sergio A. Esteban, Aaron D. Ames  

**Link**: [PDF](https://arxiv.org/pdf/2509.04722)  

**Abstract**: As humanoid robots enter real-world environments, ensuring robust locomotion across diverse environments is crucial. This paper presents a computationally efficient hierarchical control framework for humanoid robot locomotion based on reduced-order models -- enabling versatile step planning and incorporating arm and torso dynamics to better stabilize the walking. At the high level, we use the step-to-step dynamics of the ALIP model to simultaneously optimize over step periods, step lengths, and ankle torques via nonlinear MPC. The ALIP trajectories are used as references to a linear MPC framework that extends the standard SRB-MPC to also include simplified arm and torso dynamics. We validate the performance of our approach through simulation and hardware experiments on the Unitree G1 humanoid robot. In the proposed framework the high-level step planner runs at 40 Hz and the mid-level MPC at 500 Hz using the onboard mini-PC. Adaptive step timing increased the push recovery success rate by 36%, and the upper body control improved the yaw disturbance rejection. We also demonstrate robust locomotion across diverse indoor and outdoor terrains, including grass, stone pavement, and uneven gym mats. 

**Abstract (ZH)**: humanoid机器人进入真实环境后，确保其在多样化环境中具有稳健的行走能力至关重要。本文提出了一种基于降阶模型的高效层次控制框架，用于人形机器人行走控制——该框架使步态规划更具灵活性，并整合了上肢和躯干动力学以更好地稳定行走。在高层次上，我们利用ALIP模型的步距间动力学，通过非线性MPC同时优化步长周期、步长长度和踝关节扭矩。ALIP轨迹被用作扩展了标准SRB-MPC框架的线性MPC框架的参考，该框架还包含了简化后的上肢和躯干动力学。我们通过在Unitree G1人形机器人上的仿真和硬件实验验证了本方法的性能。在所提出框架中，高层次的步态规划器运行频率为40Hz，中间层的MPC运行频率为500Hz，使用机载微型PC。自适应步态定时将推动恢复成功率提高了36%，上身控制改善了对偏航扰动的抑制。我们还展示了机器人在多种室内和室外地形上的稳健行走能力，包括草地、石板路和凹凸不平的体育馆垫子。 

---
# Bootstrapping Reinforcement Learning with Sub-optimal Policies for Autonomous Driving 

**Title (ZH)**: 基于次优策略的自助增强学习在自主驾驶中的应用 

**Authors**: Zhihao Zhang, Chengyang Peng, Ekim Yurtsever, Keith A. Redmill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04712)  

**Abstract**: Automated vehicle control using reinforcement learning (RL) has attracted significant attention due to its potential to learn driving policies through environment interaction. However, RL agents often face training challenges in sample efficiency and effective exploration, making it difficult to discover an optimal driving strategy. To address these issues, we propose guiding the RL driving agent with a demonstration policy that need not be a highly optimized or expert-level controller. Specifically, we integrate a rule-based lane change controller with the Soft Actor Critic (SAC) algorithm to enhance exploration and learning efficiency. Our approach demonstrates improved driving performance and can be extended to other driving scenarios that can similarly benefit from demonstration-based guidance. 

**Abstract (ZH)**: 基于强化学习的自动驾驶车辆控制：通过示范策略增强探索和学习效率 

---
# Surformer v2: A Multimodal Classifier for Surface Understanding from Touch and Vision 

**Title (ZH)**: Surformer v2：一种基于触觉和视觉的表面理解多模态分类器 

**Authors**: Manish Kansana, Sindhuja Penchala, Shahram Rahimi, Noorbakhsh Amiri Golilarz  

**Link**: [PDF](https://arxiv.org/pdf/2509.04658)  

**Abstract**: Multimodal surface material classification plays a critical role in advancing tactile perception for robotic manipulation and interaction. In this paper, we present Surformer v2, an enhanced multi-modal classification architecture designed to integrate visual and tactile sensory streams through a late(decision level) fusion mechanism. Building on our earlier Surformer v1 framework [1], which employed handcrafted feature extraction followed by mid-level fusion architecture with multi-head cross-attention layers, Surformer v2 integrates the feature extraction process within the model itself and shifts to late fusion. The vision branch leverages a CNN-based classifier(Efficient V-Net), while the tactile branch employs an encoder-only transformer model, allowing each modality to extract modality-specific features optimized for classification. Rather than merging feature maps, the model performs decision-level fusion by combining the output logits using a learnable weighted sum, enabling adaptive emphasis on each modality depending on data context and training dynamics. We evaluate Surformer v2 on the Touch and Go dataset [2], a multi-modal benchmark comprising surface images and corresponding tactile sensor readings. Our results demonstrate that Surformer v2 performs well, maintaining competitive inference speed, suitable for real-time robotic applications. These findings underscore the effectiveness of decision-level fusion and transformer-based tactile modeling for enhancing surface understanding in multi-modal robotic perception. 

**Abstract (ZH)**: 多模态表面材料分类在提升机器人操纵和交互中的触觉感知方面起着关键作用。本文介绍了一种增强的多模态分类架构Surformer v2，该架构通过后期决策级融合机制整合视觉和触觉传感流。基于我们之前提出的Surformer v1框架，Surformer v2将特征提取过程集成到模型中，并转向后期融合。视觉分支采用基于CNN的分类器（Efficient V-Net），触觉分支采用仅编码器变压器模型，使得每种模态能够提取适用于分类的专用特征。模型通过可学习的加权和结合输出logits进行决策级融合，从而根据不同数据上下文和训练动力学灵活强调每种模态。我们在Touch and Go数据集上评估了Surformer v2，该数据集包含多模态基准中的表面图像和相应的触觉传感器读数。实验结果表明，Surformer v2表现良好，保持了竞争性的推理速度，适用于实时机器人应用。这些发现强调了决策级融合和基于变压器的触觉建模在提升多模态机器人感知中表面理解方面的有效性。

标题：
Surformer v2: Enhanced Multi-modal Classification Architecture for Tactile Perception in Robotic Manipulation 

---
# Planning from Point Clouds over Continuous Actions for Multi-object Rearrangement 

**Title (ZH)**: 基于连续动作的点云多对象重新排列规划 

**Authors**: Kallol Saha, Amber Li, Angela Rodriguez-Izquierdo, Lifan Yu, Ben Eisner, Maxim Likhachev, David Held  

**Link**: [PDF](https://arxiv.org/pdf/2509.04645)  

**Abstract**: Long-horizon planning for robot manipulation is a challenging problem that requires reasoning about the effects of a sequence of actions on a physical 3D scene. While traditional task planning methods are shown to be effective for long-horizon manipulation, they require discretizing the continuous state and action space into symbolic descriptions of objects, object relationships, and actions. Instead, we propose a hybrid learning-and-planning approach that leverages learned models as domain-specific priors to guide search in high-dimensional continuous action spaces. We introduce SPOT: Search over Point cloud Object Transformations, which plans by searching for a sequence of transformations from an initial scene point cloud to a goal-satisfying point cloud. SPOT samples candidate actions from learned suggesters that operate on partially observed point clouds, eliminating the need to discretize actions or object relationships. We evaluate SPOT on multi-object rearrangement tasks, reporting task planning success and task execution success in both simulation and real-world environments. Our experiments show that SPOT generates successful plans and outperforms a policy-learning approach. We also perform ablations that highlight the importance of search-based planning. 

**Abstract (ZH)**: 长时规划用于机器人操作是一个具有挑战性的问题，需要对一系列动作对物理3D场景的影响进行推理。虽然传统的任务规划方法在长时操作中显示出了有效性，但它们要求将连续的状态空间和动作空间离散化为物体、物体关系和动作的符号描述。相反，我们提出了一种结合学习和规划的方法，利用学习得到的模型作为领域特定的先验知识来引导高维连续动作空间中的搜索。我们引入了SPOT：基于点云物体变换的搜索方法，通过从部分观测的点云出发，寻找一个初始场景点云到目标满足点云的序列变换来进行规划。SPOT 从部分观测的点云上操作的学习建议器中采样候选动作，从而消除了动作或物体关系离散化的需要。我们分别在仿真和实际环境中评估了SPOT在多对象重新排列任务上的任务规划成功率和任务执行成功率。我们的实验结果表明，SPOT 生成了成功的规划并优于基于策略的学习方法。我们还进行了消融实验以突出搜索基规划的重要性。 

---
# Action Chunking with Transformers for Image-Based Spacecraft Guidance and Control 

**Title (ZH)**: 基于Transformer的动作分块方法在图像驱动的航天器导航与控制中应用 

**Authors**: Alejandro Posadas-Nava, Andrea Scorsoglio, Luca Ghilardi, Roberto Furfaro, Richard Linares  

**Link**: [PDF](https://arxiv.org/pdf/2509.04628)  

**Abstract**: We present an imitation learning approach for spacecraft guidance, navigation, and control(GNC) that achieves high performance from limited data. Using only 100 expert demonstrations, equivalent to 6,300 environment interactions, our method, which implements Action Chunking with Transformers (ACT), learns a control policy that maps visual and state observations to thrust and torque commands. ACT generates smoother, more consistent trajectories than a meta-reinforcement learning (meta-RL) baseline trained with 40 million interactions. We evaluate ACT on a rendezvous task: in-orbit docking with the International Space Station (ISS). We show that our approach achieves greater accuracy, smoother control, and greater sample efficiency. 

**Abstract (ZH)**: 基于模仿学习的航天器制导、导航与控制方法：从有限数据实现高性能动作片段化与变换器结合（ACT）方法在国际空间站对接任务中的应用 

---
# In-Context Policy Adaptation via Cross-Domain Skill Diffusion 

**Title (ZH)**: 基于跨域技能扩散的上下文适配策略调整 

**Authors**: Minjong Yoo, Woo Kyung Kim, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2509.04535)  

**Abstract**: In this work, we present an in-context policy adaptation (ICPAD) framework designed for long-horizon multi-task environments, exploring diffusion-based skill learning techniques in cross-domain settings. The framework enables rapid adaptation of skill-based reinforcement learning policies to diverse target domains, especially under stringent constraints on no model updates and only limited target domain data. Specifically, the framework employs a cross-domain skill diffusion scheme, where domain-agnostic prototype skills and a domain-grounded skill adapter are learned jointly and effectively from an offline dataset through cross-domain consistent diffusion processes. The prototype skills act as primitives for common behavior representations of long-horizon policies, serving as a lingua franca to bridge different domains. Furthermore, to enhance the in-context adaptation performance, we develop a dynamic domain prompting scheme that guides the diffusion-based skill adapter toward better alignment with the target domain. Through experiments with robotic manipulation in Metaworld and autonomous driving in CARLA, we show that our $\oursol$ framework achieves superior policy adaptation performance under limited target domain data conditions for various cross-domain configurations including differences in environment dynamics, agent embodiment, and task horizon. 

**Abstract (ZH)**: 基于上下文的长时 horizon 多任务环境策略适应框架：跨域环境中的扩散驱动技能学习 

---
# Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet 

**Title (ZH)**: 增强3D点云分类：ModelNet-R与Point-SkipNet相结合 

**Authors**: Mohammad Saeid, Amir Salarpour, Pedram MohajerAnsari  

**Link**: [PDF](https://arxiv.org/pdf/2509.05198)  

**Abstract**: The classification of 3D point clouds is crucial for applications such as autonomous driving, robotics, and augmented reality. However, the commonly used ModelNet40 dataset suffers from limitations such as inconsistent labeling, 2D data, size mismatches, and inadequate class differentiation, which hinder model performance. This paper introduces ModelNet-R, a meticulously refined version of ModelNet40 designed to address these issues and serve as a more reliable benchmark. Additionally, this paper proposes Point-SkipNet, a lightweight graph-based neural network that leverages efficient sampling, neighborhood grouping, and skip connections to achieve high classification accuracy with reduced computational overhead. Extensive experiments demonstrate that models trained in ModelNet-R exhibit significant performance improvements. Notably, Point-SkipNet achieves state-of-the-art accuracy on ModelNet-R with a substantially lower parameter count compared to contemporary models. This research highlights the crucial role of dataset quality in optimizing model efficiency for 3D point cloud classification. For more details, see the code at: this https URL. 

**Abstract (ZH)**: 三维点云分类对于自动驾驶、机器人技术和增强现实等应用至关重要。然而，常用的ModelNet40数据集存在标签不一致、二维数据、大小不匹配和类别区分不足等问题，阻碍了模型性能的提升。本文介绍了ModelNet-R，这是对ModelNet40进行精心优化的版本，旨在解决这些问题并作为更可靠的基准。此外，本文还提出了一种轻量级图神经网络Point-SkipNet，该网络利用高效的采样、邻域分组和跳连接，以减少计算开销并实现高分类精度。大量实验表明，使用ModelNet-R训练的模型表现出显著的性能提升。特别地，Point-SkipNet在ModelNet-R上的准确率达到了当前最先进水平，且参数量显著少于当前模型。这项研究突显了数据集质量在优化三维点云分类模型效率方面的关键作用。更多信息，请参见代码：this https URL。 

---
# Analyzing Gait Adaptation with Hemiplegia Simulation Suits and Digital Twins 

**Title (ZH)**: 基于半身瘫痪模拟装置与数字孪生的步态适应性分析 

**Authors**: Jialin Chen, Jeremie Clos, Dominic Price, Praminda Caleb-Solly  

**Link**: [PDF](https://arxiv.org/pdf/2509.05116)  

**Abstract**: To advance the development of assistive and rehabilitation robots, it is essential to conduct experiments early in the design cycle. However, testing early prototypes directly with users can pose safety risks. To address this, we explore the use of condition-specific simulation suits worn by healthy participants in controlled environments as a means to study gait changes associated with various impairments and support rapid prototyping. This paper presents a study analyzing the impact of a hemiplegia simulation suit on gait. We collected biomechanical data using a Vicon motion capture system and Delsys Trigno EMG and IMU sensors under four walking conditions: with and without a rollator, and with and without the simulation suit. The gait data was integrated into a digital twin model, enabling machine learning analyses to detect the use of the simulation suit and rollator, identify turning behavior, and evaluate how the suit affects gait over time. Our findings show that the simulation suit significantly alters movement and muscle activation patterns, prompting users to compensate with more abrupt motions. We also identify key features and sensor modalities that are most informative for accurately capturing gait dynamics and modeling human-rollator interaction within the digital twin framework. 

**Abstract (ZH)**: 基于条件特定模拟服的健康参与者在受控环境中的步态变化研究以支持助行与康复机器人快速原型开发 

---
# Language-Driven Hierarchical Task Structures as Explicit World Models for Multi-Agent Learning 

**Title (ZH)**: 语言驱动的层次化任务结构作为多智能体学习的显式世界模型 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04731)  

**Abstract**: The convergence of Language models, Agent models, and World models represents a critical frontier for artificial intelligence. While recent progress has focused on scaling Language and Agent models, the development of sophisticated, explicit World Models remains a key bottleneck, particularly for complex, long-horizon multi-agent tasks. In domains such as robotic soccer, agents trained via standard reinforcement learning in high-fidelity but structurally-flat simulators often fail due to intractable exploration spaces and sparse rewards. This position paper argues that the next frontier in developing capable agents lies in creating environments that possess an explicit, hierarchical World Model. We contend that this is best achieved through hierarchical scaffolding, where complex goals are decomposed into structured, manageable subgoals. Drawing evidence from a systematic review of 2024 research in multi-agent soccer, we identify a clear and decisive trend towards integrating symbolic and hierarchical methods with multi-agent reinforcement learning (MARL). These approaches implicitly or explicitly construct a task-based world model to guide agent learning. We then propose a paradigm shift: leveraging Large Language Models to dynamically generate this hierarchical scaffold, effectively using language to structure the World Model on the fly. This language-driven world model provides an intrinsic curriculum, dense and meaningful learning signals, and a framework for compositional learning, enabling Agent Models to acquire sophisticated, strategic behaviors with far greater sample efficiency. By building environments with explicit, language-configurable task layers, we can bridge the gap between low-level reactive behaviors and high-level strategic team play, creating a powerful and generalizable framework for training the next generation of intelligent agents. 

**Abstract (ZH)**: 语言模型、代理模型和世界模型的收敛代表着人工智能的关键前沿领域。尽管近期进展集中在扩大语言和代理模型的规模，但复杂长时间多代理任务中精细的世界模型的发展仍然是一个关键瓶颈。在机器人足球等领域，通过高保真但结构简单的模拟器使用标准强化学习训练的代理常常由于难以探索的状态空间和稀疏的奖励而失败。本文认为，开发强大代理的下一个前沿在于创建具备明确层级世界模型的环境。我们主张这可以通过层次化支撑实现，即分解复杂的目标为结构化、可管理的子目标。通过对2024年多代理足球研究的系统回顾，我们明确了符号化和层次化方法与多代理强化学习(MARL)集成的明确趋势。这些方法隐式或明确构建基于任务的世界模型以引导代理学习。然后，我们提出了一种范式转变：利用大型语言模型动态生成这一层级化支撑结构，利用语言实时构建世界模型。这种由语言驱动的世界模型提供了内在的教学计划、密集且有意义的学习信号，并为组合学习提供了框架，使代理模型能够以更高的样例效率获取复杂的战略行为。通过构建具备显式、可配置任务层级的环境，我们可以弥合低级反应行为与高级战略团队协作之间的差距，为训练下一代智能代理提供强大且可泛化的框架。 

---
# Domain Adaptation for Different Sensor Configurations in 3D Object Detection 

**Title (ZH)**: 不同传感器配置下三维物体检测的领域适应 

**Authors**: Satoshi Tanaka, Kok Seang Tan, Isamu Yamashita  

**Link**: [PDF](https://arxiv.org/pdf/2509.04711)  

**Abstract**: Recent advances in autonomous driving have underscored the importance of accurate 3D object detection, with LiDAR playing a central role due to its robustness under diverse visibility conditions. However, different vehicle platforms often deploy distinct sensor configurations, causing performance degradation when models trained on one configuration are applied to another because of shifts in the point cloud distribution. Prior work on multi-dataset training and domain adaptation for 3D object detection has largely addressed environmental domain gaps and density variation within a single LiDAR; in contrast, the domain gap for different sensor configurations remains largely unexplored. In this work, we address domain adaptation across different sensor configurations in 3D object detection. We propose two techniques: Downstream Fine-tuning (dataset-specific fine-tuning after multi-dataset training) and Partial Layer Fine-tuning (updating only a subset of layers to improve cross-configuration generalization). Using paired datasets collected in the same geographic region with multiple sensor configurations, we show that joint training with Downstream Fine-tuning and Partial Layer Fine-tuning consistently outperforms naive joint training for each configuration. Our findings provide a practical and scalable solution for adapting 3D object detection models to the diverse vehicle platforms. 

**Abstract (ZH)**: 最近自主驾驶领域的进展突显了准确的3D物体检测的重要性，由于LiDAR在各种能见度条件下的稳健性，使其在其中扮演了重要角色。然而，不同的车辆平台往往部署了不同的传感器配置，导致当使用一个配置训练的模型应用于另一个配置时会出现性能下降，这是因为点云分布发生了变化。先前关于多数据集训练和3D物体检测领域适应的研究主要集中在单个LiDAR的环境领域差异和密度变化；相比之下，不同传感器配置之间的领域差异尚未得到充分探索。在本文中，我们解决了3D物体检测中不同传感器配置之间的领域适应问题。我们提出了两种技术：下游微调（多数据集训练后的数据集特定微调）和部分层微调（仅更新一部分层以提高跨配置的一般化能力）。利用在同一地理区域收集且具有多种传感器配置的配对数据集，我们展示了结合下游微调和部分层微调的联合训练在每个配置上均优于简单的联合训练。我们的发现提供了一种实用且可扩展的方法，以适应多样的车辆平台的3D物体检测模型。 

---
# UAV-Based Intelligent Traffic Surveillance System: Real-Time Vehicle Detection, Classification, Tracking, and Behavioral Analysis 

**Title (ZH)**: 基于无人机的智能交通 surveillance 系统：实时车辆检测、分类、跟踪及行为分析 

**Authors**: Ali Khanpour, Tianyi Wang, Afra Vahidi-Shams, Wim Ectors, Farzam Nakhaie, Amirhossein Taheri, Christian Claudel  

**Link**: [PDF](https://arxiv.org/pdf/2509.04624)  

**Abstract**: Traffic congestion and violations pose significant challenges for urban mobility and road safety. Traditional traffic monitoring systems, such as fixed cameras and sensor-based methods, are often constrained by limited coverage, low adaptability, and poor scalability. To address these challenges, this paper introduces an advanced unmanned aerial vehicle (UAV)-based traffic surveillance system capable of accurate vehicle detection, classification, tracking, and behavioral analysis in real-world, unconstrained urban environments. The system leverages multi-scale and multi-angle template matching, Kalman filtering, and homography-based calibration to process aerial video data collected from altitudes of approximately 200 meters. A case study in urban area demonstrates robust performance, achieving a detection precision of 91.8%, an F1-score of 90.5%, and tracking metrics (MOTA/MOTP) of 92.1% and 93.7%, respectively. Beyond precise detection, the system classifies five vehicle types and automatically detects critical traffic violations, including unsafe lane changes, illegal double parking, and crosswalk obstructions, through the fusion of geofencing, motion filtering, and trajectory deviation analysis. The integrated analytics module supports origin-destination tracking, vehicle count visualization, inter-class correlation analysis, and heatmap-based congestion modeling. Additionally, the system enables entry-exit trajectory profiling, vehicle density estimation across road segments, and movement direction logging, supporting comprehensive multi-scale urban mobility analytics. Experimental results confirms the system's scalability, accuracy, and practical relevance, highlighting its potential as an enforcement-aware, infrastructure-independent traffic monitoring solution for next-generation smart cities. 

**Abstract (ZH)**: 基于无人机的交通监控系统：实时城市环境中车辆检测、分类、跟踪和行为分析 

---
# PRREACH: Probabilistic Risk Assessment Using Reachability for UAV Control 

**Title (ZH)**: PRREACH：基于可达性方法的UAV控制概率风险评估 

**Authors**: Nicole Fronda, Hariharan Narayanan, Sadia Afrin Ananna, Steven Weber, Houssam Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2509.04451)  

**Abstract**: We present a new approach for designing risk-bounded controllers for Uncrewed Aerial Vehicles (UAVs). Existing frameworks for assessing risk of UAV operations rely on knowing the conditional probability of an incident occurring given different causes. Limited data for computing these probabilities makes real-world implementation of these frameworks difficult. Furthermore, existing frameworks do not include control methods for risk mitigation. Our approach relies on UAV dynamics, and employs reachability analysis for a probabilistic risk assessment over all feasible UAV trajectories. We use this holistic risk assessment to formulate a control optimization problem that minimally changes a UAV's existing control law to be bounded by an accepted risk threshold. We call our approach PRReach. Public and readily available UAV dynamics models and open source spatial data for mapping hazard outcomes enables practical implementation of PRReach for both offline pre-flight and online in-flight risk assessment and mitigation. We evaluate PRReach through simulation experiments on real-world data. Results show that PRReach controllers reduce risk by up to 24% offline, and up to 53% online from classical controllers. 

**Abstract (ZH)**: 一种用于无人驾驶航空车辆（UAVs）的风险界判定控制器设计新方法 

---
# The best approximation pair problem relative to two subsets in a normed space 

**Title (ZH)**: 相对两个子集在赋范空间中的最佳逼近对问题 

**Authors**: Daniel Reem, Yair Censor  

**Link**: [PDF](https://arxiv.org/pdf/2403.18767)  

**Abstract**: In the classical best approximation pair (BAP) problem, one is given two nonempty, closed, convex and disjoint subsets in a finite- or an infinite-dimensional Hilbert space, and the goal is to find a pair of points, each from each subset, which realizes the distance between the subsets. We discuss the problem in more general normed spaces and with possibly non-convex subsets, and focus our attention on the issues of uniqueness and existence of the solution to the problem. As far as we know, these fundamental issues have not received much attention. We present several sufficient geometric conditions for the (at most) uniqueness of a BAP. These conditions are related to the structure and the relative orientation of the boundaries of the subsets and to the norm. We also present many sufficient conditions for the existence of a BAP. Our results significantly extend the horizon of a recent algorithm for solving the BAP problem [Censor, Mansour, Reem, J. Approx. Theory (2024)]. The paper also shows, perhaps for the first time, how wide is the scope of the BAP problem in terms of the scientific communities which are involved in it (frequently independently) and in terms of its applications. 

**Abstract (ZH)**: 经典最佳近似对（BAP）问题中，给定有限维或无限维希尔伯特空间中的两个非空、闭合、凸且不相交的子集，目标是找到每部分来自各子集的一对点，使这两点之间的距离最小化。我们将该问题推广到更一般的赋范空间，并考虑可能不凸的子集，重点讨论解的唯一性和存在性问题。据我们所知，这些基本问题尚未引起广泛关注。我们提出了若干充分几何条件来保障（至多）唯一性。这些条件涉及子集边界的结构及其相对方位以及范数。我们还提出了许多充分条件来保障BAP的存在性。我们的结果大大扩展了近期解决BAP问题的算法[曾森, 曼苏尔, 雷姆, 《逼近理论杂志》 (2024)]的应用范围。本文还展示了BAP问题在涉及的科学社区（经常独立地）以及应用范围上的广度，可能是首次如此全面地展示。 

---
