# UniSkill: Imitating Human Videos via Cross-Embodiment Skill Representations 

**Title (ZH)**: UniSkill: 通过跨躯体技能表示模仿人类视频 

**Authors**: Hanjung Kim, Jaehyun Kang, Hyolim Kang, Meedeum Cho, Seon Joo Kim, Youngwoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.08787)  

**Abstract**: Mimicry is a fundamental learning mechanism in humans, enabling individuals to learn new tasks by observing and imitating experts. However, applying this ability to robots presents significant challenges due to the inherent differences between human and robot embodiments in both their visual appearance and physical capabilities. While previous methods bridge this gap using cross-embodiment datasets with shared scenes and tasks, collecting such aligned data between humans and robots at scale is not trivial. In this paper, we propose UniSkill, a novel framework that learns embodiment-agnostic skill representations from large-scale cross-embodiment video data without any labels, enabling skills extracted from human video prompts to effectively transfer to robot policies trained only on robot data. Our experiments in both simulation and real-world environments show that our cross-embodiment skills successfully guide robots in selecting appropriate actions, even with unseen video prompts. The project website can be found at: this https URL. 

**Abstract (ZH)**: 模仿是人类基本的学习机制，使个体能够通过观察和模仿专家来学习新任务。然而，将这一能力应用到机器人上由于人类和机器人在视觉外观和物理能力上的本质差异而面临重大挑战。尽管先前的方法使用共享场景和任务的跨体征数据集来弥合这一差距，但大规模收集人类和机器人之间的对齐数据并不容易。在本文中，我们提出了一种名为UniSkill的新型框架，该框架可以从大规模的跨体征视频数据中学习体征无关的技能表示，无需任何标签，使从人类视频提示中提取的技能能够有效转移到仅基于机器人数据训练的机器人策略中。我们的仿真和真实环境实验结果表明，我们的跨体征技能能够成功指导机器人选择合适的行动，即使是对未见过的视频提示。项目网站可访问：this https URL。 

---
# NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance 

**Title (ZH)**: NavDP：学习基于优先信息指导的模拟到现实导航扩散策略 

**Authors**: Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang, Jiangmiao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08712)  

**Abstract**: Learning navigation in dynamic open-world environments is an important yet challenging skill for robots. Most previous methods rely on precise localization and mapping or learn from expensive real-world demonstrations. In this paper, we propose the Navigation Diffusion Policy (NavDP), an end-to-end framework trained solely in simulation and can zero-shot transfer to different embodiments in diverse real-world environments. The key ingredient of NavDP's network is the combination of diffusion-based trajectory generation and a critic function for trajectory selection, which are conditioned on only local observation tokens encoded from a shared policy transformer. Given the privileged information of the global environment in simulation, we scale up the demonstrations of good quality to train the diffusion policy and formulate the critic value function targets with contrastive negative samples. Our demonstration generation approach achieves about 2,500 trajectories/GPU per day, 20$\times$ more efficient than real-world data collection, and results in a large-scale navigation dataset with 363.2km trajectories across 1244 scenes. Trained with this simulation dataset, NavDP achieves state-of-the-art performance and consistently outstanding generalization capability on quadruped, wheeled, and humanoid robots in diverse indoor and outdoor environments. In addition, we present a preliminary attempt at using Gaussian Splatting to make in-domain real-to-sim fine-tuning to further bridge the sim-to-real gap. Experiments show that adding such real-to-sim data can improve the success rate by 30\% without hurting its generalization capability. 

**Abstract (ZH)**: 学习动态开放世界环境中的导航是一项重要但具有挑战性的技能，对于机器人而言尤为重要。大多数先前的方法依赖于精确的定位和建图，或者从昂贵的真实世界演示中学习。本文提出了一种名为导航扩散政策（NavDP）的端到端框架，该框架仅在仿真中训练，并能够零样本转移到不同载体在多样化的现实环境中。NavDP网络的关键成分是基于扩散的轨迹生成与用于轨迹选择的评论函数的结合，后者仅基于共享策略变换器编码的局部观察 token。借助仿真中对全局环境的先验信息，我们将高质量的演示扩展以训练扩散策略，并使用反例样本来构建评论函数目标。我们的演示生成方法每天每个GPU可以生成约2500条轨迹，比真实世界的数据收集效率高20倍，并生成了一个大规模的导航数据集，包含1244个场景中的363.2公里轨迹。使用此仿真数据集训练的NavDP在各种室内和室外环境中实现了最先进的性能，并在四足、轮式和类人机器人上表现出一致的出色泛化能力。此外，我们还提出了使用高斯点积进行领域内真实到仿真的微调的初步尝试，以进一步缩小仿真实用性差距。实验结果显示，添加这种真实到仿真的数据可以使成功率达到提高30%而不会损害其泛化能力。 

---
# A Social Robot with Inner Speech for Dietary Guidance 

**Title (ZH)**: 具有内部语音对话功能的社会机器人在饮食指导中的应用 

**Authors**: Valerio Belcamino, Alessandro Carfì, Valeria Seidita, Fulvio Mastrogiovanni, Antonio Chella  

**Link**: [PDF](https://arxiv.org/pdf/2505.08664)  

**Abstract**: We explore the use of inner speech as a mechanism to enhance transparency and trust in social robots for dietary advice. In humans, inner speech structures thought processes and decision-making; in robotics, it improves explainability by making reasoning explicit. This is crucial in healthcare scenarios, where trust in robotic assistants depends on both accurate recommendations and human-like dialogue, which make interactions more natural and engaging. Building on this, we developed a social robot that provides dietary advice, and we provided the architecture with inner speech capabilities to validate user input, refine reasoning, and generate clear justifications. The system integrates large language models for natural language understanding and a knowledge graph for structured dietary information. By making decisions more transparent, our approach strengthens trust and improves human-robot interaction in healthcare. We validated this by measuring the computational efficiency of our architecture and conducting a small user study, which assessed the reliability of inner speech in explaining the robot's behavior. 

**Abstract (ZH)**: 我们探索内心言语作为机制以增强社交机器人在饮食建议中的透明度和信任度。通过内心言语，人类结构化思维过程和决策；在机器人领域，内心言语通过使推理过程显性化来提高可解释性。这在医疗保健场景中极为重要，因为对机器人助手的信任不仅依赖于准确的建议，还依赖于类人的对话，使互动更加自然和吸引人。基于此，我们开发了一个提供饮食建议的社交机器人，并为其添加了内心言语能力以验证用户输入、优化推理并生成清晰的理由。该系统集成了大规模语言模型以进行自然语言理解，并利用知识图谱以结构化形式提供饮食信息。通过使决策过程更加透明，我们的方法增强了信任并提高了医疗保健场景中的人机交互。我们通过衡量架构的计算效率并开展小型用户研究来验证这一点，该研究评估了内心言语在解释机器人行为可靠性方面的效果。 

---
# A Comparative Study of Human Activity Recognition: Motion, Tactile, and multi-modal Approaches 

**Title (ZH)**: 人类活动识别的比较研究：运动、触觉和多模态方法 

**Authors**: Valerio Belcamino, Nhat Minh Dinh Le, Quan Khanh Luu, Alessandro Carfì, Van Anh Ho, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2505.08657)  

**Abstract**: Human activity recognition (HAR) is essential for effective Human-Robot Collaboration (HRC), enabling robots to interpret and respond to human actions. This study evaluates the ability of a vision-based tactile sensor to classify 15 activities, comparing its performance to an IMU-based data glove. Additionally, we propose a multi-modal framework combining tactile and motion data to leverage their complementary strengths. We examined three approaches: motion-based classification (MBC) using IMU data, tactile-based classification (TBC) with single or dual video streams, and multi-modal classification (MMC) integrating both. Offline validation on segmented datasets assessed each configuration's accuracy under controlled conditions, while online validation on continuous action sequences tested online performance. Results showed the multi-modal approach consistently outperformed single-modality methods, highlighting the potential of integrating tactile and motion sensing to enhance HAR systems for collaborative robotics. 

**Abstract (ZH)**: 基于视觉的触觉传感器在15种活动识别中的能力研究：与惯性测量单元数据手套的比较及多模式框架的提出 

---
# Augmented Reality for RObots (ARRO): Pointing Visuomotor Policies Towards Visual Robustness 

**Title (ZH)**: 增强现实用于机器人（ARRO）：指向视觉稳健性的指针知觉运动策略 

**Authors**: Reihaneh Mirjalili, Tobias Jülg, Florian Walter, Wolfram Burgard  

**Link**: [PDF](https://arxiv.org/pdf/2505.08627)  

**Abstract**: Visuomotor policies trained on human expert demonstrations have recently shown strong performance across a wide range of robotic manipulation tasks. However, these policies remain highly sensitive to domain shifts stemming from background or robot embodiment changes, which limits their generalization capabilities. In this paper, we present ARRO, a novel calibration-free visual representation that leverages zero-shot open-vocabulary segmentation and object detection models to efficiently mask out task-irrelevant regions of the scene without requiring additional training. By filtering visual distractors and overlaying virtual guides during both training and inference, ARRO improves robustness to scene variations and reduces the need for additional data collection. We extensively evaluate ARRO with Diffusion Policy on several tabletop manipulation tasks in both simulation and real-world environments, and further demonstrate its compatibility and effectiveness with generalist robot policies, such as Octo and OpenVLA. Across all settings in our evaluation, ARRO yields consistent performance gains, allows for selective masking to choose between different objects, and shows robustness even to challenging segmentation conditions. Videos showcasing our results are available at: this http URL 

**Abstract (ZH)**: 基于人类专家演示训练的知觉运动策略在多种机器人操作任务中表现出强大性能。然而，这些策略仍然高度敏感于背景或机器人身体特征变化引起的领域转换，这限制了它们的泛化能力。本文中，我们提出ARRO，这是一种新型无需校准的视觉表示，它利用零样本开放式词汇分割和对象检测模型来高效地屏蔽场景中的任务无关区域，而无需额外训练。通过在训练和推理过程中过滤视觉干扰并叠加虚拟引导，ARRO 提高了对场景变化的鲁棒性，并减少了额外数据收集的需求。我们使用 Diffusion Policy 在多种桌面操作任务的模拟和真实环境中进行了广泛评估，并进一步展示了其与通用机器人策略（如 Octo 和 OpenVLA）的兼容性和有效性。在我们评估的所有设置中，ARRO 均实现了一致的性能提升，允许选择性屏蔽以选择不同的对象，并且即使在具有挑战性的分割条件下也表现出鲁棒性。有关我们的结果的视频可在以下链接查看：this http URL。 

---
# Beyond Predefined Actions: Integrating Behavior Trees and Dynamic Movement Primitives for Robot Learning from Demonstration 

**Title (ZH)**: 超越预定义动作：将行为树与动态运动 primitives 结合用于机器人演示学习 

**Authors**: David Cáceres Domínguez, Erik Schaffernicht, Todor Stoyanov  

**Link**: [PDF](https://arxiv.org/pdf/2505.08625)  

**Abstract**: Interpretable policy representations like Behavior Trees (BTs) and Dynamic Motion Primitives (DMPs) enable robot skill transfer from human demonstrations, but each faces limitations: BTs require expert-crafted low-level actions, while DMPs lack high-level task logic. We address these limitations by integrating DMP controllers into a BT framework, jointly learning the BT structure and DMP actions from single demonstrations, thereby removing the need for predefined actions. Additionally, by combining BT decision logic with DMP motion generation, our method enhances policy interpretability, modularity, and adaptability for autonomous systems. Our approach readily affords both learning to replicate low-level motions and combining partial demonstrations into a coherent and easy-to-modify overall policy. 

**Abstract (ZH)**: 可解释的策略表示方法，如行为树（BTs）和动态运动基元（DMPs），能够通过人类示范将机器人技能进行迁移，但每种方法都面临局限：BTs需要专家设计的低级动作，而DMPs缺乏高级任务逻辑。我们通过将DMP控制器整合到BT框架中，从单个示范中联合学习BT结构和DMP动作，从而消除了预定义动作的需要。此外，结合BT决策逻辑和DMP运动生成，我们的方法增强了策略的可解释性、模块化和适应性，以提升自主系统的性能。我们的方法既可以直接学习模仿低级运动，也可以将不完整的示范合并为一个连贯且易于修改的整体策略。 

---
# MC-Swarm: Minimal-Communication Multi-Agent Trajectory Planning and Deadlock Resolution for Quadrotor Swarm 

**Title (ZH)**: MC-Swarm: 最小通信多agent轨迹规划及四旋翼机群死锁解决方法 

**Authors**: Yunwoo Lee, Jungwon Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.08593)  

**Abstract**: For effective multi-agent trajectory planning, it is important to consider lightweight communication and its potential asynchrony. This paper presents a distributed trajectory planning algorithm for a quadrotor swarm that operates asynchronously and requires no communication except during the initial planning phase. Moreover, our algorithm guarantees no deadlock under asynchronous updates and absence of communication during flight. To effectively ensure these points, we build two main modules: coordination state updater and trajectory optimizer. The coordination state updater computes waypoints for each agent toward its goal and performs subgoal optimization while considering deadlocks, as well as safety constraints with respect to neighbor agents and obstacles. Then, the trajectory optimizer generates a trajectory that ensures collision avoidance even with the asynchronous planning updates of neighboring agents. We provide a theoretical guarantee of collision avoidance with deadlock resolution and evaluate the effectiveness of our method in complex simulation environments, including random forests and narrow-gap mazes. Additionally, to reduce the total mission time, we design a faster coordination state update using lightweight communication. Lastly, our approach is validated through extensive simulations and real-world experiments with cluttered environment scenarios. 

**Abstract (ZH)**: 一种在异步环境下无需持续通信的四旋翼无人机群分布式轨迹规划算法 

---
# End-to-End Multi-Task Policy Learning from NMPC for Quadruped Locomotion 

**Title (ZH)**: 基于NMPC的 quadruped 行走多任务端到端策略学习 

**Authors**: Anudeep Sajja, Shahram Khorshidi, Sebastian Houben, Maren Bennewitz  

**Link**: [PDF](https://arxiv.org/pdf/2505.08574)  

**Abstract**: Quadruped robots excel in traversing complex, unstructured environments where wheeled robots often fail. However, enabling efficient and adaptable locomotion remains challenging due to the quadrupeds' nonlinear dynamics, high degrees of freedom, and the computational demands of real-time control. Optimization-based controllers, such as Nonlinear Model Predictive Control (NMPC), have shown strong performance, but their reliance on accurate state estimation and high computational overhead makes deployment in real-world settings challenging. In this work, we present a Multi-Task Learning (MTL) framework in which expert NMPC demonstrations are used to train a single neural network to predict actions for multiple locomotion behaviors directly from raw proprioceptive sensor inputs. We evaluate our approach extensively on the quadruped robot Go1, both in simulation and on real hardware, demonstrating that it accurately reproduces expert behavior, allows smooth gait switching, and simplifies the control pipeline for real-time deployment. Our MTL architecture enables learning diverse gaits within a unified policy, achieving high $R^{2}$ scores for predicted joint targets across all tasks. 

**Abstract (ZH)**: 四足机器人在复杂、未结构化环境中表现出色，而轮式机器人在这些环境中往往表现不佳。然而，由于四足机器人的非线性动力学、高自由度以及实时控制所需的高计算需求，实现高效和适应性的运动仍然具有挑战性。基于优化的控制器，如非线性模型预测控制（NMPC），显示了强大的性能，但其依赖于准确的状态估计和高计算开销使得其实现现场应用具有挑战性。在本文中，我们提出了一种多任务学习（MTL）框架，其中专家NMPC演示用于训练单个神经网络，直接从原始本体感觉传感器输入中预测多种运动行为的行动。我们广泛评估了我们的方法在四足机器人Go1上的性能，包括仿真和实物硬件上，证明了它能够准确重现专家行为、实现平滑的步伐切换，并简化了实时部署的控制流程。我们的MTL架构能够在统一策略中学习多样化的步伐，对所有任务预测的关节目标获得较高的$R^{2}$分数。 

---
# From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation 

**Title (ZH)**: 从感知到行动：连接机器人操作中的推理与决策 

**Authors**: Yifu Yuan, Haiqin Cui, Yibin Chen, Zibin Dong, Fei Ni, Longxin Kou, Jinyi Liu, Pengyi Li, Yan Zheng, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08548)  

**Abstract**: Achieving generalization in robotic manipulation remains a critical challenge, particularly for unseen scenarios and novel tasks. Current Vision-Language-Action (VLA) models, while building on top of general Vision-Language Models (VLMs), still fall short of achieving robust zero-shot performance due to the scarcity and heterogeneity prevalent in embodied datasets. To address these limitations, we propose FSD (From Seeing to Doing), a novel vision-language model that generates intermediate representations through spatial relationship reasoning, providing fine-grained guidance for robotic manipulation. Our approach combines a hierarchical data pipeline for training with a self-consistency mechanism that aligns spatial coordinates with visual signals. Through extensive experiments, we comprehensively validated FSD's capabilities in both "seeing" and "doing," achieving outstanding performance across 8 benchmarks for general spatial reasoning and embodied reference abilities, as well as on our proposed more challenging benchmark VABench. We also verified zero-shot capabilities in robot manipulation, demonstrating significant performance improvements over baseline methods in both SimplerEnv and real robot settings. Experimental results show that FSD achieves 54.1% success rate in SimplerEnv and 72% success rate across 8 real-world tasks, outperforming the strongest baseline by 30%. 

**Abstract (ZH)**: 实现机器人操作中的泛化仍然是一个关键挑战，特别是在未见场景和新颖任务中。尽管现有的视听动作（VLA）模型基于通用的视听模型（VLM），但仍因体态数据中普遍存在的稀疏性和异质性而在实现鲁棒的零样本性能方面力有未逮。为解决这些限制，我们提出了一种新颖的视听模型FSD（从感知到执行），通过空间关系推理生成中间表示，为机器人操作提供细粒度指导。我们的方法结合了分层数据管道进行训练，并通过一种自一致性机制将空间坐标与视觉信号对齐。通过广泛的实验，我们全面验证了FSD在“看到”和“执行”方面的能力，在8个基准测试中取得了出色的全景化空间推理和体态参考能力表现，并在我们提出的更具挑战性的基准测试VABench上也取得了优异表现。我们还验证了FSD在机器人操作中的零样本能力，在SimplerEnv和真实机器人设置中均展示了显著优于基线方法的性能提升。实验结果显示，FSD在SimplerEnv中的成功率为54.1%，在8个真实世界任务中的成功率为72%，比最强基线高出30%。 

---
# FOCI: Trajectory Optimization on Gaussian Splats 

**Title (ZH)**: FOCI：高斯体素上的轨迹优化 

**Authors**: Mario Gomez Andreu, Maximum Wilder-Smith, Victor Klemm, Vaishakh Patil, Jesus Tordesillas, Marco Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2505.08510)  

**Abstract**: 3D Gaussian Splatting (3DGS) has recently gained popularity as a faster alternative to Neural Radiance Fields (NeRFs) in 3D reconstruction and view synthesis methods. Leveraging the spatial information encoded in 3DGS, this work proposes FOCI (Field Overlap Collision Integral), an algorithm that is able to optimize trajectories directly on the Gaussians themselves. FOCI leverages a novel and interpretable collision formulation for 3DGS using the notion of the overlap integral between Gaussians. Contrary to other approaches, which represent the robot with conservative bounding boxes that underestimate the traversability of the environment, we propose to represent the environment and the robot as Gaussian Splats. This not only has desirable computational properties, but also allows for orientation-aware planning, allowing the robot to pass through very tight and narrow spaces. We extensively test our algorithm in both synthetic and real Gaussian Splats, showcasing that collision-free trajectories for the ANYmal legged robot that can be computed in a few seconds, even with hundreds of thousands of Gaussians making up the environment. The project page and code are available at this https URL 

**Abstract (ZH)**: 3D高斯点积（3DGS）在3D重建和视角合成方法中近年来因其比神经辐射场（NeRF）更快而受到青睐。利用3DGS中编码的时空信息，本工作提出了一种直接在高斯点自身上优化轨迹的FOCI（场重叠碰撞积分）算法。FOCI利用重叠积分的新颖且可解释的碰撞形式化来利用3DGS的特性。与仅用保守的边界框低估环境可通行性的其他方法不同，我们提出将环境和机器人表示为高斯点积。这不仅具有良好的计算属性，还允许姿态感知的规划，使机器人能够通过非常狭小的空间。我们在合成和真实高斯点积上进行了广泛的测试，展示了即使环境由数十万高斯点组成，也可以在几秒内计算出无碰撞轨迹，这些轨迹适用于ANYmal腿式机器人。项目页面和代码可在以下链接访问：这个 https URL。 

---
# Zero-Shot Sim-to-Real Reinforcement Learning for Fruit Harvesting 

**Title (ZH)**: 零样本模拟到现实的强化学习在水果采摘中的应用 

**Authors**: Emlyn Williams, Athanasios Polydoros  

**Link**: [PDF](https://arxiv.org/pdf/2505.08458)  

**Abstract**: This paper presents a comprehensive sim-to-real pipeline for autonomous strawberry picking from dense clusters using a Franka Panda robot. Our approach leverages a custom Mujoco simulation environment that integrates domain randomization techniques. In this environment, a deep reinforcement learning agent is trained using the dormant ratio minimization algorithm. The proposed pipeline bridges low-level control with high-level perception and decision making, demonstrating promising performance in both simulation and in a real laboratory environment, laying the groundwork for successful transfer to real-world autonomous fruit harvesting. 

**Abstract (ZH)**: 本研究表明了一种用于利用Franka Panda机器人从密集簇中自主采摘草莓的全面仿真实验转换流水线。该方法借助一个包含领域随机化技术的定制Mujoco仿真环境，并使用休眠比最小化算法训练深度强化学习代理。所提出的流水线将低级控制与高级感知和决策相结合，在仿真和实际实验室环境中均表现出色，为成功转换到实际自主水果 harvesting 奠定了基础。 

---
# Parameter Estimation using Reinforcement Learning Causal Curiosity: Limits and Challenges 

**Title (ZH)**: 使用强化学习因果好奇心进行参数估计：限制与挑战 

**Authors**: Miguel Arana-Catania, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.08453)  

**Abstract**: Causal understanding is important in many disciplines of science and engineering, where we seek to understand how different factors in the system causally affect an experiment or situation and pave a pathway towards creating effective or optimising existing models. Examples of use cases are autonomous exploration and modelling of unknown environments or assessing key variables in optimising large complex systems. In this paper, we analyse a Reinforcement Learning approach called Causal Curiosity, which aims to estimate as accurately and efficiently as possible, without directly measuring them, the value of factors that causally determine the dynamics of a system. Whilst the idea presents a pathway forward, measurement accuracy is the foundation of methodology effectiveness. Focusing on the current causal curiosity's robotic manipulator, we present for the first time a measurement accuracy analysis of the future potentials and current limitations of this technique and an analysis of its sensitivity and confounding factor disentanglement capability - crucial for causal analysis. As a result of our work, we promote proposals for an improved and efficient design of Causal Curiosity methods to be applied to real-world complex scenarios. 

**Abstract (ZH)**: 因果理解在科学和工程的许多学科中非常重要，我们旨在理解系统中不同因素如何因果影响实验或情况，并为此铺平创造有效或优化现有模型的途径。应用案例包括自主探索和建模未知环境或评估优化大型复杂系统的关键变量。在本文中，我们分析了一种名为因果好奇心的强化学习方法，该方法旨在尽可能准确且高效地估计那些因果决定系统动力学的因素价值，而无需直接测量这些因素。尽管这一想法为未来发展指出了一条路径，但测量准确性是该方法有效性的基础。本文首次专注于当前因果好奇心的机器人 manipulator，对其未来潜力和当前局限性进行测量精度分析，并对其因果分析中的因果因素分离能力进行分析。由此，我们提出改进和高效设计因果好奇心方法的建议，以应用于实际复杂场景。 

---
# Symbolically-Guided Visual Plan Inference from Uncurated Video Data 

**Title (ZH)**: 符号引导的视觉计划推断从未经整理的视频数据中 

**Authors**: Wenyan Yang, Ahmet Tikna, Yi Zhao, Yuying Zhang, Luigi Palopoli, Marco Roveri, Joni Pajarinen  

**Link**: [PDF](https://arxiv.org/pdf/2505.08444)  

**Abstract**: Visual planning, by offering a sequence of intermediate visual subgoals to a goal-conditioned low-level policy, achieves promising performance on long-horizon manipulation tasks. To obtain the subgoals, existing methods typically resort to video generation models but suffer from model hallucination and computational cost. We present Vis2Plan, an efficient, explainable and white-box visual planning framework powered by symbolic guidance. From raw, unlabeled play data, Vis2Plan harnesses vision foundation models to automatically extract a compact set of task symbols, which allows building a high-level symbolic transition graph for multi-goal, multi-stage planning. At test time, given a desired task goal, our planner conducts planning at the symbolic level and assembles a sequence of physically consistent intermediate sub-goal images grounded by the underlying symbolic representation. Our Vis2Plan outperforms strong diffusion video generation-based visual planners by delivering 53\% higher aggregate success rate in real robot settings while generating visual plans 35$\times$ faster. The results indicate that Vis2Plan is able to generate physically consistent image goals while offering fully inspectable reasoning steps. 

**Abstract (ZH)**: Visual规划：通过为基于目标的低级策略提供一系列中间视觉子目标，实现长期操作任务的优异性能。Vis2Plan：一种基于符号指导的高效可解释白盒视觉规划框架 

---
# HMR-ODTA: Online Diverse Task Allocation for a Team of Heterogeneous Mobile Robots 

**Title (ZH)**: HMR-ODTA:在线多样化任务分配给异质移动机器人团队 

**Authors**: Ashish Verma, Avinash Gautam, Tanishq Duhan, V. S. Shekhawat, Sudeept Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08419)  

**Abstract**: Coordinating time-sensitive deliveries in environments like hospitals poses a complex challenge, particularly when managing multiple online pickup and delivery requests within strict time windows using a team of heterogeneous robots. Traditional approaches fail to address dynamic rescheduling or diverse service requirements, typically restricting robots to single-task types. This paper tackles the Multi-Pickup and Delivery Problem with Time Windows (MPDPTW), where autonomous mobile robots are capable of handling varied service requests. The objective is to minimize late delivery penalties while maximizing task completion rates. To achieve this, we propose a novel framework leveraging a heterogeneous robot team and an efficient dynamic scheduling algorithm that supports dynamic task rescheduling. Users submit requests with specific time constraints, and our decentralized algorithm, Heterogeneous Mobile Robots Online Diverse Task Allocation (HMR-ODTA), optimizes task assignments to ensure timely service while addressing delays or task rejections. Extensive simulations validate the algorithm's effectiveness. For smaller task sets (40-160 tasks), penalties were reduced by nearly 63%, while for larger sets (160-280 tasks), penalties decreased by approximately 50%. These results highlight the algorithm's effectiveness in improving task scheduling and coordination in multi-robot systems, offering a robust solution for enhancing delivery performance in structured, time-critical environments. 

**Abstract (ZH)**: 多时间窗口条件下异构机器人团队的多取多送问题研究 

---
# ORACLE-Grasp: Zero-Shot Task-Oriented Robotic Grasping using Large Multimodal Models 

**Title (ZH)**: ORACLE-抓取：基于大型多模态模型的零样本任务导向机器人抓取 

**Authors**: Avihai Giuili, Rotem Atari, Avishai Sintov  

**Link**: [PDF](https://arxiv.org/pdf/2505.08417)  

**Abstract**: Grasping unknown objects in unstructured environments remains a fundamental challenge in robotics, requiring both semantic understanding and spatial reasoning. Existing methods often rely on dense training datasets or explicit geometric modeling, limiting their scalability to real-world tasks. Recent advances in Large Multimodal Models (LMMs) offer new possibilities for integrating vision and language understanding, but their application to autonomous robotic grasping remains largely unexplored. We present ORACLE-Grasp, a zero-shot framework that leverages LMMs as semantic oracles to guide grasp selection without requiring additional training or human input. The system formulates grasp prediction as a structured, iterative decision process, using dual-prompt tool calling to first extract high-level object context and then select task-relevant grasp regions. By discretizing the image space and reasoning over candidate areas, ORACLE-Grasp mitigates the spatial imprecision common in LMMs and produces human-like, task-driven grasp suggestions. Early stopping and depth-based refinement steps further enhance efficiency and physical grasp reliability. Experiments demonstrate that the predicted grasps achieve low positional and orientation errors relative to human-annotated ground truth and lead to high success rates in real-world pick up tasks. These results highlight the potential of combining language-driven reasoning with lightweight vision techniques to enable robust, autonomous grasping without task-specific datasets or retraining. 

**Abstract (ZH)**: 未知物体在无结构环境中的抓取仍然是机器人技术中的一个基本挑战，需要同时具备语义理解和空间推理能力。现有的方法往往依赖于密集的训练数据集或显式的几何建模，限制了其在实际任务中的可扩展性。近期在大规模多模态模型（LMMs）方面的进展为融合视觉和语言理解提供了新的可能性，但其在自主机器人抓取中的应用仍然很大程度上尚未被探索。我们提出ORACLE-抓取框架，利用LMMs作为语义或acles来指导抓取选择，无需额外训练或人工输入。该系统将抓取预测表述为结构化的迭代决策过程，通过双提示工具调用来首先提取高层次的对象上下文，然后选择与任务相关的抓取区域。通过离散化图像空间并在候选区域内进行推理，ORACLE-抓取减少了LMMs中常见的空间精度问题，并生成了接近人类、任务驱动的抓取建议。通过早期停止和基于深度的精修步骤，进一步提高了效率和物理抓取的可靠性。实验结果表明，预测的抓取相对于由人工标注的地面真相在位置和姿态误差上较低，并在实际的任务拾取中实现了高成功率。这些结果突显了结合基于语言的推理与轻量级视觉技术以实现匹配任务数据集或重新训练的鲁棒自主抓取的潜力。 

---
# MDF: Multi-Modal Data Fusion with CNN-Based Object Detection for Enhanced Indoor Localization Using LiDAR-SLAM 

**Title (ZH)**: 基于CNN对象检测的多模态数据融合以增强基于LiDAR-SLAM的室内定位 

**Authors**: Saqi Hussain Kalan, Boon Giin Lee, Wan-Young Chung  

**Link**: [PDF](https://arxiv.org/pdf/2505.08388)  

**Abstract**: Indoor localization faces persistent challenges in achieving high accuracy, particularly in GPS-deprived environments. This study unveils a cutting-edge handheld indoor localization system that integrates 2D LiDAR and IMU sensors, delivering enhanced high-velocity precision mapping, computational efficiency, and real-time adaptability. Unlike 3D LiDAR systems, it excels with rapid processing, low-cost scalability, and robust performance, setting new standards for emergency response, autonomous navigation, and industrial automation. Enhanced with a CNN-driven object detection framework and optimized through Cartographer SLAM (simultaneous localization and mapping ) in ROS, the system significantly reduces Absolute Trajectory Error (ATE) by 21.03%, achieving exceptional precision compared to state-of-the-art approaches like SC-ALOAM, with a mean x-position error of -0.884 meters (1.976 meters). The integration of CNN-based object detection ensures robustness in mapping and localization, even in cluttered or dynamic environments, outperforming existing methods by 26.09%. These advancements establish the system as a reliable, scalable solution for high-precision localization in challenging indoor scenarios 

**Abstract (ZH)**: 室内定位在实现高精度方面仍面临持久挑战，尤其是在GPS受限环境中。本研究揭示了一种集成2D LiDAR和IMU传感器的前沿手持室内定位系统，该系统提供了增强的高速精度制图、计算效率和实时适应性。与3D LiDAR系统相比，它在快速处理、低成本可扩展性和稳健性能方面表现出色，为应急响应、自主导航和工业自动化设立了新的标准。通过基于CNN的目标检测框架和ROS中的Cartographer SLAM优化，该系统显著降低了绝对轨迹误差（ATE）21.03%，其平均x位置误差为-0.884米（1.976米），优于诸如SC-ALOAM等最先进的方法。基于CNN的目标检测集成确保了在复杂或动态环境中具有鲁棒的制图和定位能力，比现有方法高出26.09%。这些进步确立了该系统在具有挑战性室内环境中的可靠且可扩展的高精度定位解决方案地位。 

---
# Continuous World Coverage Path Planning for Fixed-Wing UAVs using Deep Reinforcement Learning 

**Title (ZH)**: 使用深度强化学习的固定翼无人机连续全球路径规划 

**Authors**: Mirco Theile, Andres R. Zapata Rodriguez, Marco Caccamo, Alberto L. Sangiovanni-Vincentelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.08382)  

**Abstract**: Unmanned Aerial Vehicle (UAV) Coverage Path Planning (CPP) is critical for applications such as precision agriculture and search and rescue. While traditional methods rely on discrete grid-based representations, real-world UAV operations require power-efficient continuous motion planning. We formulate the UAV CPP problem in a continuous environment, minimizing power consumption while ensuring complete coverage. Our approach models the environment with variable-size axis-aligned rectangles and UAV motion with curvature-constrained Bézier curves. We train a reinforcement learning agent using an action-mapping-based Soft Actor-Critic (AM-SAC) algorithm employing a self-adaptive curriculum. Experiments on both procedurally generated and hand-crafted scenarios demonstrate the effectiveness of our method in learning energy-efficient coverage strategies. 

**Abstract (ZH)**: 无人驾驶航空器（UAV）覆盖路径规划（CPP）对于精准农业和搜索救援等应用至关重要。传统的CPP方法依赖离散的网格表示，而实际的UAV操作需要高效的连续运动规划。我们将在连续环境中形式化UAV-CPP问题，目标是在确保完全覆盖的同时最小化能量消耗。我们使用可变大小的轴对齐矩形来建模环境，并使用曲率约束的Bézier曲线来建模UAV运动。我们利用基于动作映射的Soft Actor-Critic（AM-SAC）算法并结合自适应课程进行agents的训练。实验结果表明，我们的方法在学习能量高效的覆盖策略方面是有效的。 

---
# Adaptive Diffusion Policy Optimization for Robotic Manipulation 

**Title (ZH)**: 自适应扩散策略优化在机器人 manipulation 中的应用 

**Authors**: Huiyun Jiang, Zhuang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08376)  

**Abstract**: Recent studies have shown the great potential of diffusion models in improving reinforcement learning (RL) by modeling complex policies, expressing a high degree of multi-modality, and efficiently handling high-dimensional continuous control tasks. However, there is currently limited research on how to optimize diffusion-based polices (e.g., Diffusion Policy) fast and stably. In this paper, we propose an Adam-based Diffusion Policy Optimization (ADPO), a fast algorithmic framework containing best practices for fine-tuning diffusion-based polices in robotic control tasks using the adaptive gradient descent method in RL. Adaptive gradient method is less studied in training RL, let alone diffusion-based policies. We confirm that ADPO outperforms other diffusion-based RL methods in terms of overall effectiveness for fine-tuning on standard robotic tasks. Concretely, we conduct extensive experiments on standard robotic control tasks to test ADPO, where, particularly, six popular diffusion-based RL methods are provided as benchmark methods. Experimental results show that ADPO acquires better or comparable performance than the baseline methods. Finally, we systematically analyze the sensitivity of multiple hyperparameters in standard robotics tasks, providing guidance for subsequent practical applications. Our video demonstrations are released in this https URL. 

**Abstract (ZH)**: 基于AdaGrad的扩散政策优化（ADPO）：机器人控制任务中的快速稳定方法 

---
# MA-ROESL: Motion-aware Rapid Reward Optimization for Efficient Robot Skill Learning from Single Videos 

**Title (ZH)**: MA-ROESL：基于运动感知的快速奖励优化方法以实现从单个视频中高效学习机器人技能 

**Authors**: Xianghui Wang, Xinming Zhang, Yanjun Chen, Xiaoyu Shen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08367)  

**Abstract**: Vision-language models (VLMs) have demonstrated excellent high-level planning capabilities, enabling locomotion skill learning from video demonstrations without the need for meticulous human-level reward design. However, the improper frame sampling method and low training efficiency of current methods remain a critical bottleneck, resulting in substantial computational overhead and time costs. To address this limitation, we propose Motion-aware Rapid Reward Optimization for Efficient Robot Skill Learning from Single Videos (MA-ROESL). MA-ROESL integrates a motion-aware frame selection method to implicitly enhance the quality of VLM-generated reward functions. It further employs a hybrid three-phase training pipeline that improves training efficiency via rapid reward optimization and derives the final policy through online fine-tuning. Experimental results demonstrate that MA-ROESL significantly enhances training efficiency while faithfully reproducing locomotion skills in both simulated and real-world settings, thereby underscoring its potential as a robust and scalable framework for efficient robot locomotion skill learning from video demonstrations. 

**Abstract (ZH)**: 基于运动感知的快速奖励优化高效单视频机器人技能学习（MA-ROESL） 

---
# Fast Contact Detection via Fusion of Joint and Inertial Sensors for Parallel Robots in Human-Robot Collaboration 

**Title (ZH)**: 基于关节传感器与惯性传感器融合的平行机器人人类-机器人协作快速接触检测 

**Authors**: Aran Mohammad, Jan Piosik, Dustin Lehmann, Thomas Seel, Moritz Schappler  

**Link**: [PDF](https://arxiv.org/pdf/2505.08334)  

**Abstract**: Fast contact detection is crucial for safe human-robot collaboration. Observers based on proprioceptive information can be used for contact detection but have first-order error dynamics, which results in delays. Sensor fusion based on inertial measurement units (IMUs) consisting of accelerometers and gyroscopes is advantageous for reducing delays. The acceleration estimation enables the direct calculation of external forces. For serial robots, the installation of multiple accelerometers and gyroscopes is required for dynamics modeling since the joint coordinates are the minimal coordinates. Alternatively, parallel robots (PRs) offer the potential to use only one IMU on the end-effector platform, which already presents the minimal coordinates of the PR. This work introduces a sensor-fusion method for contact detection using encoders and only one low-cost, consumer-grade IMU for a PR. The end-effector accelerations are estimated by an extended Kalman filter and incorporated into the dynamics to calculate external forces. In real-world experiments with a planar PR, we demonstrate that this approach reduces the detection duration by up to 50% compared to a momentum observer and enables the collision and clamping detection within 3-39ms. 

**Abstract (ZH)**: 快速接触检测对于人机安全协作至关重要。基于本体感受信息的观测器可以用于接触检测，但具有一阶误差动力学，导致延迟。基于加速度计和陀螺仪的惯性测量单元（IMU）传感器融合有利于减少延迟。末端执行器加速度估计可以通过扩展卡尔曼滤波器实现，并整合到动力学模型中以计算外部力。在实际实验中，本研究介绍了一种使用编码器和一个低成本消费级IMU对平行机器人进行接触检测的传感器融合方法。通过扩展卡尔曼滤波器估计末端执行器加速度并整合到动力学模型中以计算外部力。在平面平行机器人的真实世界实验中，我们证明了这种方法与动量观测器相比可将检测时间缩短最多50%，并在3-39ms内实现了碰撞和夹持检测。 

---
# Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Efficient Reinforcement Learning 

**Title (ZH)**: 自动课程学习在驾驶场景中的应用：迈向稳健高效的强化学习 

**Authors**: Ahmed Abouelazm, Tim Weinstein, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.08264)  

**Abstract**: This paper addresses the challenges of training end-to-end autonomous driving agents using Reinforcement Learning (RL). RL agents are typically trained in a fixed set of scenarios and nominal behavior of surrounding road users in simulations, limiting their generalization and real-life deployment. While domain randomization offers a potential solution by randomly sampling driving scenarios, it frequently results in inefficient training and sub-optimal policies due to the high variance among training scenarios. To address these limitations, we propose an automatic curriculum learning framework that dynamically generates driving scenarios with adaptive complexity based on the agent's evolving capabilities. Unlike manually designed curricula that introduce expert bias and lack scalability, our framework incorporates a ``teacher'' that automatically generates and mutates driving scenarios based on their learning potential -- an agent-centric metric derived from the agent's current policy -- eliminating the need for expert design. The framework enhances training efficiency by excluding scenarios the agent has mastered or finds too challenging. We evaluate our framework in a reinforcement learning setting where the agent learns a driving policy from camera images. Comparative results against baseline methods, including fixed scenario training and domain randomization, demonstrate that our approach leads to enhanced generalization, achieving higher success rates: +9\% in low traffic density, +21\% in high traffic density, and faster convergence with fewer training steps. Our findings highlight the potential of ACL in improving the robustness and efficiency of RL-based autonomous driving agents. 

**Abstract (ZH)**: 基于自动课程学习的端到端自主驾驶代理训练方法 

---
# Training Strategies for Efficient Embodied Reasoning 

**Title (ZH)**: 高效体域推理的训练策略 

**Authors**: William Chen, Suneel Belkhale, Suvir Mirchandani, Oier Mees, Danny Driess, Karl Pertsch, Sergey Levine  

**Link**: [PDF](https://arxiv.org/pdf/2505.08243)  

**Abstract**: Robot chain-of-thought reasoning (CoT) -- wherein a model predicts helpful intermediate representations before choosing actions -- provides an effective method for improving the generalization and performance of robot policies, especially vision-language-action models (VLAs). While such approaches have been shown to improve performance and generalization, they suffer from core limitations, like needing specialized robot reasoning data and slow inference speeds. To design new robot reasoning approaches that address these issues, a more complete characterization of why reasoning helps policy performance is critical. We hypothesize several mechanisms by which robot reasoning improves policies -- (1) better representation learning, (2) improved learning curricularization, and (3) increased expressivity -- then devise simple variants of robot CoT reasoning to isolate and test each one. We find that learning to generate reasonings does lead to better VLA representations, while attending to the reasonings aids in actually leveraging these features for improved action prediction. Our results provide us with a better understanding of why CoT reasoning helps VLAs, which we use to introduce two simple and lightweight alternative recipes for robot reasoning. Our proposed approaches achieve significant performance gains over non-reasoning policies, state-of-the-art results on the LIBERO-90 benchmark, and a 3x inference speedup compared to standard robot reasoning. 

**Abstract (ZH)**: 基于推理的机器人链式思考（CoT）推理：一种通过预测有助于执行的中间表示来提升机器人策略泛化能力和性能的方法，特别适用于视觉-语言-行动模型（VLAs）。 

---
# Motion Control of High-Dimensional Musculoskeletal Systems with Hierarchical Model-Based Planning 

**Title (ZH)**: 基于分级模型规划的高维肌肉骨骼系统运动控制 

**Authors**: Yunyue Wei, Shanning Zhuang, Vincent Zhuang, Yanan Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.08238)  

**Abstract**: Controlling high-dimensional nonlinear systems, such as those found in biological and robotic applications, is challenging due to large state and action spaces. While deep reinforcement learning has achieved a number of successes in these domains, it is computationally intensive and time consuming, and therefore not suitable for solving large collections of tasks that require significant manual tuning. In this work, we introduce Model Predictive Control with Morphology-aware Proportional Control (MPC^2), a hierarchical model-based learning algorithm for zero-shot and near-real-time control of high-dimensional complex dynamical systems. MPC^2 uses a sampling-based model predictive controller for target posture planning, and enables robust control for high-dimensional tasks by incorporating a morphology-aware proportional controller for actuator coordination. The algorithm enables motion control of a high-dimensional human musculoskeletal model in a variety of motion tasks, such as standing, walking on different terrains, and imitating sports activities. The reward function of MPC^2 can be tuned via black-box optimization, drastically reducing the need for human-intensive reward engineering. 

**Abstract (ZH)**: 基于形态意识比例控制的模型预测控制（MPC²）：用于高维复杂动态系统的零样本和近实时控制 

---
# SKiD-SLAM: Robust, Lightweight, and Distributed Multi-Robot LiDAR SLAM in Resource-Constrained Field Environments 

**Title (ZH)**: SKiD-SLAM：资源受限场环境中的稳健、轻量级和分布式多机器人LiDAR SLAM 

**Authors**: Hogyun Kim, Jiwon Choi, Juwon Kim, Geonmo Yang, Dongjin Cho, Hyungtae Lim, Younggun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.08230)  

**Abstract**: Distributed LiDAR SLAM is crucial for achieving efficient robot autonomy and improving the scalability of mapping. However, two issues need to be considered when applying it in field environments: one is resource limitation, and the other is inter/intra-robot association. The resource limitation issue arises when the data size exceeds the processing capacity of the network or memory, especially when utilizing communication systems or onboard computers in the field. The inter/intra-robot association issue occurs due to the narrow convergence region of ICP under large viewpoint differences, triggering many false positive loops and ultimately resulting in an inconsistent global map for multi-robot systems. To tackle these problems, we propose a distributed LiDAR SLAM framework designed for versatile field applications, called SKiD-SLAM. Extending our previous work that solely focused on lightweight place recognition and fast and robust global registration, we present a multi-robot mapping framework that focuses on robust and lightweight inter-robot loop closure in distributed LiDAR SLAM. Through various environmental experiments, we demonstrate that our method is more robust and lightweight compared to other state-of-the-art distributed SLAM approaches, overcoming resource limitation and inter/intra-robot association issues. Also, we validated the field applicability of our approach through mapping experiments in real-world planetary emulation terrain and cave environments, which are in-house datasets. Our code will be available at this https URL. 

**Abstract (ZH)**: 分布式LiDAR SLAM对于实现高效机器人自主性和提高地图的可扩展性至关重要。然而，在田野环境中应用它时需考虑两个问题：一是资源限制，二是机器人之间的/内部关联问题。资源限制问题发生在数据量超出网络或内存处理能力的情况下，尤其是在利用现场的通信系统或机载计算机时。机器人之间的/内部关联问题由于ICP的收敛区域狭窄，在视野差异较大时会引起许多虚假闭环，并最终导致多机器人系统中全局地图的一致性问题。为了解决这些问题，我们提出了一种适用于多种田野应用的分布式LiDAR SLAM框架，称为SKiD-SLAM。在此基础上，我们扩展了之前专注于轻量级地点识别和快速可靠的全局对齐的工作，提出了一个专注于分布式LiDAR SLAM中鲁棒且轻量级的机器人之间闭环检测的多机器人制图框架。通过各种环境实验，我们证明了我们的方法在鲁棒性和轻量级方面优于其他最先进的分布式SLAM方法，克服了资源限制和机器人之间的/内部关联问题。我们也通过在真实世界行星模拟地形和洞穴环境中的制图实验验证了我们方法的田野适用性，这些是内部数据集。我们的代码将发布在该网址：this https URL。 

---
# Constrained Factor Graph Optimization for Robust Networked Pedestrian Inertial Navigation 

**Title (ZH)**: 约束因子图优化在鲁棒网络行人惯性导航中的应用 

**Authors**: Yingjie Hu, Wang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08229)  

**Abstract**: This paper presents a novel constrained Factor Graph Optimization (FGO)-based approach for networked inertial navigation in pedestrian localization. To effectively mitigate the drift inherent in inertial navigation solutions, we incorporate kinematic constraints directly into the nonlinear optimization framework. Specifically, we utilize equality constraints, such as Zero-Velocity Updates (ZUPTs), and inequality constraints representing the maximum allowable distance between body-mounted Inertial Measurement Units (IMUs) based on human anatomical limitations. While equality constraints are straightforwardly integrated as error factors, inequality constraints cannot be explicitly represented in standard FGO formulations. To address this, we introduce a differentiable softmax-based penalty term in the FGO cost function to enforce inequality constraints smoothly and robustly. The proposed constrained FGO approach leverages temporal correlations across multiple epochs, resulting in optimal state trajectory estimates while consistently maintaining constraint satisfaction. Experimental results confirm that our method outperforms conventional Kalman filter approaches, demonstrating its effectiveness and robustness for pedestrian navigation. 

**Abstract (ZH)**: 基于约束因子图优化的网络惯性导航行人定位新方法 

---
# Reinforcement Learning-based Fault-Tolerant Control for Quadrotor with Online Transformer Adaptation 

**Title (ZH)**: 基于强化学习的四旋翼飞行器在线变压器自适应容错控制 

**Authors**: Dohyun Kim, Jayden Dongwoo Lee, Hyochoong Bang, Jungho Bae  

**Link**: [PDF](https://arxiv.org/pdf/2505.08223)  

**Abstract**: Multirotors play a significant role in diverse field robotics applications but remain highly susceptible to actuator failures, leading to rapid instability and compromised mission reliability. While various fault-tolerant control (FTC) strategies using reinforcement learning (RL) have been widely explored, most previous approaches require prior knowledge of the multirotor model or struggle to adapt to new configurations. To address these limitations, we propose a novel hybrid RL-based FTC framework integrated with a transformer-based online adaptation module. Our framework leverages a transformer architecture to infer latent representations in real time, enabling adaptation to previously unseen system models without retraining. We evaluate our method in a PyBullet simulation under loss-of-effectiveness actuator faults, achieving a 95% success rate and a positional root mean square error (RMSE) of 0.129 m, outperforming existing adaptation methods with 86% success and an RMSE of 0.153 m. Further evaluations on quadrotors with varying configurations confirm the robustness of our framework across untrained dynamics. These results demonstrate the potential of our framework to enhance the adaptability and reliability of multirotors, enabling efficient fault management in dynamic and uncertain environments. Website is available at this http URL 

**Abstract (ZH)**: 多旋翼在多样化领域机器人应用中发挥着重要作用，但仍然高度易受执行器故障影响，导致快速失稳和任务可靠性下降。尽管已经探索了各种基于强化学习（RL）的容错控制（FTC）策略，但大多数先前的方法需要多旋翼模型的先验知识，或者难以适应新的配置。为了解决这些限制，我们提出了一种新型的结合基于变压器的在线适应模块的混合RL-Based FTC框架。该框架利用变压器架构实时推断潜在表示，使系统能够在无需重新训练的情况下适应未见过的系统模型。我们在PyBullet模拟中评估了该方法，在效果失效执行器故障条件下，实现了95%的成功率和位置均方根误差（RMSE）为0.129 m，优于现有方法的86%成功率和RMSE为0.153 m。进一步在不同配置的四旋翼上的评估证实了该框架在未训练动力学下的鲁棒性。这些结果展示了该框架在提高多旋翼的适应性和可靠性方面的潜力，使其能够在动态和不确定环境中有效管理故障。网站详见this http URL。 

---
# Scaling Multi Agent Reinforcement Learning for Underwater Acoustic Tracking via Autonomous Vehicles 

**Title (ZH)**: 基于自主车辆的水下声学跟踪中多智能体 reinforcement learning 的可扩展性研究 

**Authors**: Matteo Gallici, Ivan Masmitja, Mario Martín  

**Link**: [PDF](https://arxiv.org/pdf/2505.08222)  

**Abstract**: Autonomous vehicles (AV) offer a cost-effective solution for scientific missions such as underwater tracking. Recently, reinforcement learning (RL) has emerged as a powerful method for controlling AVs in complex marine environments. However, scaling these techniques to a fleet--essential for multi-target tracking or targets with rapid, unpredictable motion--presents significant computational challenges. Multi-Agent Reinforcement Learning (MARL) is notoriously sample-inefficient, and while high-fidelity simulators like Gazebo's LRAUV provide 100x faster-than-real-time single-robot simulations, they offer no significant speedup for multi-vehicle scenarios, making MARL training impractical. To address these limitations, we propose an iterative distillation method that transfers high-fidelity simulations into a simplified, GPU-accelerated environment while preserving high-level dynamics. This approach achieves up to a 30,000x speedup over Gazebo through parallelization, enabling efficient training via end-to-end GPU acceleration. Additionally, we introduce a novel Transformer-based architecture (TransfMAPPO) that learns multi-agent policies invariant to the number of agents and targets, significantly improving sample efficiency. Following large-scale curriculum learning conducted entirely on GPU, we perform extensive evaluations in Gazebo, demonstrating that our method maintains tracking errors below 5 meters over extended durations, even in the presence of multiple fast-moving targets. This work bridges the gap between large-scale MARL training and high-fidelity deployment, providing a scalable framework for autonomous fleet control in real-world sea missions. 

**Abstract (ZH)**: 自主车辆在水下跟踪等科学任务中的低成本解决方案：基于强化学习的多自主 underwater 车辆控制方法及其加速技术 

---
# Rethink Repeatable Measures of Robot Performance with Statistical Query 

**Title (ZH)**: 重新思考用于评估机器人性能的可重复度量方法：基于统计查询 

**Authors**: Bowen Weng, Linda Capito, Guillermo A. Castillo, Dylan Khor  

**Link**: [PDF](https://arxiv.org/pdf/2505.08216)  

**Abstract**: For a general standardized testing algorithm designed to evaluate a specific aspect of a robot's performance, several key expectations are commonly imposed. Beyond accuracy (i.e., closeness to a typically unknown ground-truth reference) and efficiency (i.e., feasibility within acceptable testing costs and equipment constraints), one particularly important attribute is repeatability. Repeatability refers to the ability to consistently obtain the same testing outcome when similar testing algorithms are executed on the same subject robot by different stakeholders, across different times or locations. However, achieving repeatable testing has become increasingly challenging as the components involved grow more complex, intelligent, diverse, and, most importantly, stochastic. While related efforts have addressed repeatability at ethical, hardware, and procedural levels, this study focuses specifically on repeatable testing at the algorithmic level. Specifically, we target the well-adopted class of testing algorithms in standardized evaluation: statistical query (SQ) algorithms (i.e., algorithms that estimate the expected value of a bounded function over a distribution using sampled data). We propose a lightweight, parameterized, and adaptive modification applicable to any SQ routine, whether based on Monte Carlo sampling, importance sampling, or adaptive importance sampling, that makes it provably repeatable, with guaranteed bounds on both accuracy and efficiency. We demonstrate the effectiveness of the proposed approach across three representative scenarios: (i) established and widely adopted standardized testing of manipulators, (ii) emerging intelligent testing algorithms for operational risk assessment in automated vehicles, and (iii) developing use cases involving command tracking performance evaluation of humanoid robots in locomotion tasks. 

**Abstract (ZH)**: 一种标准化测试算法的设计通常会对评估机器人特定方面性能的能力提出几个关键期望。除了准确性（即，与通常未知的真实参考值的接近程度）和效率（即，在可接受的测试成本和设备约束条件下的可行性），特别重要的一属性是可重复性。可重复性指的是不同相关方在同一机器人上执行相似测试算法时，在不同时间和地点一致获得相同测试结果的能力。然而，随着涉及的组件变得更加复杂、智能、多样，最重要的是随机性，实现可重复测试变得越来越具有挑战性。虽然相关努力已在伦理、硬件和程序层面解决了可重复性问题，但本研究特别关注算法层面的可重复测试。具体而言，我们针对标准化评估中广泛采用的测试算法类别——统计查询（SQ）算法（即，使用采样数据估计分布上某种有界函数的期望值的算法），提出了一种轻量级、参数化和自适应修改方法，适用于基于蒙特卡洛采样、重要性采样或自适应重要性采样的任何SQ流程，确保其在准确性和效率方面具有可证明的可重复性。我们通过三个代表性场景证明了所提方法的有效性：（i）广泛采用的操纵器的标准化测试；（ii）自动驾驶车辆运行风险评估的新兴智能测试算法；（iii）类人机器人在行动任务中命令跟踪性能评估的新兴应用场景。 

---
# HandCept: A Visual-Inertial Fusion Framework for Accurate Proprioception in Dexterous Hands 

**Title (ZH)**: HandCept: 一种用于精确灵巧手本体感觉的视觉-惯性融合框架 

**Authors**: Junda Huang, Jianshu Zhou, Honghao Guo, Yunhui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08213)  

**Abstract**: As robotics progresses toward general manipulation, dexterous hands are becoming increasingly critical. However, proprioception in dexterous hands remains a bottleneck due to limitations in volume and generality. In this work, we present HandCept, a novel visual-inertial proprioception framework designed to overcome the challenges of traditional joint angle estimation methods. HandCept addresses the difficulty of achieving accurate and robust joint angle estimation in dynamic environments where both visual and inertial measurements are prone to noise and drift. It leverages a zero-shot learning approach using a wrist-mounted RGB-D camera and 9-axis IMUs, fused in real time via a latency-free Extended Kalman Filter (EKF). Our results show that HandCept achieves joint angle estimation errors between $2^{\circ}$ and $4^{\circ}$ without observable drift, outperforming visual-only and inertial-only methods. Furthermore, we validate the stability and uniformity of the IMU system, demonstrating that a common base frame across IMUs simplifies system calibration. To support sim-to-real transfer, we also open-sourced our high-fidelity rendering pipeline, which is essential for training without real-world ground truth. This work offers a robust, generalizable solution for proprioception in dexterous hands, with significant implications for robotic manipulation and human-robot interaction. 

**Abstract (ZH)**: 视觉-惯性本体感受框架HandCept：面向灵巧手的本体感受新技术 

---
# CLTP: Contrastive Language-Tactile Pre-training for 3D Contact Geometry Understanding 

**Title (ZH)**: CLTP: 对比学习语言-触觉预训练以理解三维接触几何 

**Authors**: Wenxuan Ma, Xiaoge Cao, Yixiang Zhang, Chaofan Zhang, Shaobo Yang, Peng Hao, Bin Fang, Yinghao Cai, Shaowei Cui, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08194)  

**Abstract**: Recent advancements in integrating tactile sensing with vision-language models (VLMs) have demonstrated remarkable potential for robotic multimodal perception. However, existing tactile descriptions remain limited to superficial attributes like texture, neglecting critical contact states essential for robotic manipulation. To bridge this gap, we propose CLTP, an intuitive and effective language tactile pretraining framework that aligns tactile 3D point clouds with natural language in various contact scenarios, thus enabling contact-state-aware tactile language understanding for contact-rich manipulation tasks. We first collect a novel dataset of 50k+ tactile 3D point cloud-language pairs, where descriptions explicitly capture multidimensional contact states (e.g., contact location, shape, and force) from the tactile sensor's perspective. CLTP leverages a pre-aligned and frozen vision-language feature space to bridge holistic textual and tactile modalities. Experiments validate its superiority in three downstream tasks: zero-shot 3D classification, contact state classification, and tactile 3D large language model (LLM) interaction. To the best of our knowledge, this is the first study to align tactile and language representations from the contact state perspective for manipulation tasks, providing great potential for tactile-language-action model learning. Code and datasets are open-sourced at this https URL. 

**Abstract (ZH)**: Recent advancements in 将触觉感知与视觉语言模型（VLMs）集成的进展在机器人多模态感知方面展现了巨大的潜力。然而，现有的触觉描述仍然局限于如纹理等表面属性，忽略了对于机器人操作至关重要的接触状态。为了解决这一问题，我们提出了CLTP，这是一种直观且有效的位置语言触觉预训练框架，该框架在各种接触场景中将触觉3D点云与自然语言对齐，从而实现感知丰富接触的接触状态感知触觉语言理解。我们首先收集了一个包含50K+触觉3D点云-语言对的新数据集，在这些描述中，从触觉传感器的角度明确捕捉到了多维度的接触状态（例如，接触位置、形状和力）。CLTP利用预对齐和冻结的视觉语言特征空间连接了整体的语言和触觉模态。实验在其后的三个下游任务中验证了其优越性：零样本3D分类、接触状态分类以及触觉3D大语言模型（LLM）交互。据我们所知，这是首次从接触状态视角将触觉和语言表示相结合进行操作任务的研究，为触觉-语言-动作模型的学习提供了巨大潜力。代码和数据集在该链接处开源。 

---
# A Tightly Coupled IMU-Based Motion Capture Approach for Estimating Multibody Kinematics and Kinetics 

**Title (ZH)**: 基于IMU的紧密耦合运动捕捉方法用于多体运动学和动力学估计 

**Authors**: Hassan Osman, Daan de Kanter, Jelle Boelens, Manon Kok, Ajay Seth  

**Link**: [PDF](https://arxiv.org/pdf/2505.08193)  

**Abstract**: Inertial Measurement Units (IMUs) enable portable, multibody motion capture (MoCap) in diverse environments beyond the laboratory, making them a practical choice for diagnosing mobility disorders and supporting rehabilitation in clinical or home settings. However, challenges associated with IMU measurements, including magnetic distortions and drift errors, complicate their broader use for MoCap. In this work, we propose a tightly coupled motion capture approach that directly integrates IMU measurements with multibody dynamic models via an Iterated Extended Kalman Filter (IEKF) to simultaneously estimate the system's kinematics and kinetics. By enforcing kinematic and kinetic properties and utilizing only accelerometer and gyroscope data, our method improves IMU-based state estimation accuracy. Our approach is designed to allow for incorporating additional sensor data, such as optical MoCap measurements and joint torque readings, to further enhance estimation accuracy. We validated our approach using highly accurate ground truth data from a 3 Degree of Freedom (DoF) pendulum and a 6 DoF Kuka robot. We demonstrate a maximum Root Mean Square Difference (RMSD) in the pendulum's computed joint angles of 3.75 degrees compared to optical MoCap Inverse Kinematics (IK), which serves as the gold standard in the absence of internal encoders. For the Kuka robot, we observe a maximum joint angle RMSD of 3.24 degrees compared to the Kuka's internal encoders, while the maximum joint angle RMSD of the optical MoCap IK compared to the encoders was 1.16 degrees. Additionally, we report a maximum joint torque RMSD of 2 Nm in the pendulum compared to optical MoCap Inverse Dynamics (ID), and 3.73 Nm in the Kuka robot relative to its internal torque sensors. 

**Abstract (ZH)**: 基于惯性测量单元的紧耦合运动捕捉方法及其在多体动力模型中的应用 

---
# What Matters for Batch Online Reinforcement Learning in Robotics? 

**Title (ZH)**: 机器人领域批量在线强化学习中什么是重要的？ 

**Authors**: Perry Dong, Suvir Mirchandani, Dorsa Sadigh, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2505.08078)  

**Abstract**: The ability to learn from large batches of autonomously collected data for policy improvement -- a paradigm we refer to as batch online reinforcement learning -- holds the promise of enabling truly scalable robot learning by significantly reducing the need for human effort of data collection while getting benefits from self-improvement. Yet, despite the promise of this paradigm, it remains challenging to achieve due to algorithms not being able to learn effectively from the autonomous data. For example, prior works have applied imitation learning and filtered imitation learning methods to the batch online RL problem, but these algorithms often fail to efficiently improve from the autonomously collected data or converge quickly to a suboptimal point. This raises the question of what matters for effective batch online RL in robotics. Motivated by this question, we perform a systematic empirical study of three axes -- (i) algorithm class, (ii) policy extraction methods, and (iii) policy expressivity -- and analyze how these axes affect performance and scaling with the amount of autonomous data. Through our analysis, we make several observations. First, we observe that the use of Q-functions to guide batch online RL significantly improves performance over imitation-based methods. Building on this, we show that an implicit method of policy extraction -- via choosing the best action in the distribution of the policy -- is necessary over traditional policy extraction methods from offline RL. Next, we show that an expressive policy class is preferred over less expressive policy classes. Based on this analysis, we propose a general recipe for effective batch online RL. We then show a simple addition to the recipe of using temporally-correlated noise to obtain more diversity results in further performance gains. Our recipe obtains significantly better performance and scaling compared to prior methods. 

**Abstract (ZH)**: 基于大量自主收集数据的批量在线强化学习：一类促进机器人学习可扩展性的范式 

---
# Land-Coverage Aware Path-Planning for Multi-UAV Swarms in Search and Rescue Scenarios 

**Title (ZH)**: 基于土地覆盖的多无人机群搜救路径规划 

**Authors**: Pedro Antonio Alarcon Granadeno, Jane Cleland-Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08060)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) have become vital in search-and-rescue (SAR) missions, with autonomous mission planning improving response times and coverage efficiency. Early approaches primarily used path planning techniques such as A*, potential-fields, or Dijkstra's algorithm, while recent approaches have incorporated meta-heuristic frameworks like genetic algorithms and particle swarm optimization to balance competing objectives such as network connectivity, energy efficiency, and strategic placement of charging stations. However, terrain-aware path planning remains under-explored, despite its critical role in optimizing UAV SAR deployments. To address this gap, we present a computer-vision based terrain-aware mission planner that autonomously extracts and analyzes terrain topology to enhance SAR pre-flight planning. Our framework uses a deep segmentation network fine-tuned on our own collection of landcover datasets to transform satellite imagery into a structured, grid-based representation of the operational area. This classification enables terrain-specific UAV-task allocation, improving deployment strategies in complex environments. We address the challenge of irregular terrain partitions, by introducing a two-stage partitioning scheme that first evaluates terrain monotonicity along coordinate axes before applying a cost-based recursive partitioning process, minimizing unnecessary splits and optimizing path efficiency. Empirical validation in a high-fidelity simulation environment demonstrates that our approach improves search and dispatch time over multiple meta-heuristic techniques and against a competing state-of-the-art method. These results highlight its potential for large-scale SAR operations, where rapid response and efficient UAV coordination are critical. 

**Abstract (ZH)**: 无人驾驶航空器（UAVs）在搜索与救援（SAR）任务中变得至关重要，自主任务规划提高了响应时间和覆盖效率。早期方法主要采用路径规划技术如A*、势场法或迪杰斯特拉算法，而最近的方法则结合了遗传算法和粒子群优化等元启发式框架，以平衡网络连接、能量效率和充电站战略性位置等竞争性目标。然而，地形感知路径规划仍然鲜有研究，尽管其在优化UAV SAR部署中的作用至关重要。为填补这一空白，我们提出了一种基于计算机视觉的地形感知任务规划器，该规划器能够自主提取和分析地形拓扑，以增强SAR预飞行规划。我们的框架利用了一个在我们自己的土地覆盖数据集上微调的深度分割网络，将卫星影像转换为运营区域的结构化网格表示。这种分类使得可以根据地形进行无人机任务分配，从而在复杂环境中提高部署策略。为了应对不规则地形分区的挑战，我们引入了一种两阶段分区方案，首先沿坐标轴评估地形单调性，然后应用基于成本的递归分区过程，从而最小化不必要的分割并优化路径效率。在高保真模拟环境中的实证验证表明，我们的方法在多种元启发式技术以及与一种竞争性的最新方法相比，在搜索和调度时间方面均有改进。这些结果突显了其在大规模SAR操作中的潜力，尤其是在快速响应和高效无人机协调方面。 

---
# PRISM: Complete Online Decentralized Multi-Agent Pathfinding with Rapid Information Sharing using Motion Constraints 

**Title (ZH)**: PRISM: 完整的基于运动约束的快速信息共享在线去中心化多智能体路径规划 

**Authors**: Hannah Lee, Zachary Serlin, James Motes, Brendan Long, Marco Morales, Nancy M. Amato  

**Link**: [PDF](https://arxiv.org/pdf/2505.08025)  

**Abstract**: We introduce PRISM (Pathfinding with Rapid Information Sharing using Motion Constraints), a decentralized algorithm designed to address the multi-task multi-agent pathfinding (MT-MAPF) problem. PRISM enables large teams of agents to concurrently plan safe and efficient paths for multiple tasks while avoiding collisions. It employs a rapid communication strategy that uses information packets to exchange motion constraint information, enhancing cooperative pathfinding and situational awareness, even in scenarios without direct communication. We prove that PRISM resolves and avoids all deadlock scenarios when possible, a critical challenge in decentralized pathfinding. Empirically, we evaluate PRISM across five environments and 25 random scenarios, benchmarking it against the centralized Conflict-Based Search (CBS) and the decentralized Token Passing with Task Swaps (TPTS) algorithms. PRISM demonstrates scalability and solution quality, supporting 3.4 times more agents than CBS and handling up to 2.5 times more tasks in narrow passage environments than TPTS. Additionally, PRISM matches CBS in solution quality while achieving faster computation times, even under low-connectivity conditions. Its decentralized design reduces the computational burden on individual agents, making it scalable for large environments. These results confirm PRISM's robustness, scalability, and effectiveness in complex and dynamic pathfinding scenarios. 

**Abstract (ZH)**: PRISM（基于运动约束的快速信息共享路径finding算法）：一种分布式多任务多智能体路径finding算法 

---
# Virtual Holonomic Constraints in Motion Planning: Revisiting Feasibility and Limitations 

**Title (ZH)**: 虚拟holonomic约束在运动规划中的重新审视：可行性和限制探究 

**Authors**: Maksim Surov  

**Link**: [PDF](https://arxiv.org/pdf/2505.07983)  

**Abstract**: This paper addresses the feasibility of virtual holonomic constraints (VHCs) in the context of motion planning for underactuated mechanical systems with a single degree of underactuation. While existing literature has established a widely accepted definition of VHC, we argue that this definition is overly restrictive and excludes a broad class of admissible trajectories from consideration. To illustrate this point, we analyze a periodic motion of the Planar Vertical Take-Off and Landing (PVTOL) aircraft. The corresponding phase trajectory and reference control input are analytic functions. We demonstrate the stabilizability of this solution by constructing a feedback controller that ensures asymptotic orbital stability. However, for this solution -- as well as for a broad class of similar ones -- there exists no VHC that satisfies the conventional definition. This observation calls for a reconsideration of how the notion of VHC is defined, with the potential to significantly expand the practical applicability of VHCs in motion planning. 

**Abstract (ZH)**: 本文探讨了在单自由度欠驱动机械系统运动规划中虚拟 holonomic 约束（VHC）的可行性。虽然现有文献已经确立了 VHC 的广泛接受的定义，但本文认为这一定义过于严格，排除了一大类可采纳的轨迹。通过分析平面垂直起降（PVTOL）飞机的周期运动，我们展示了对应相轨迹和参考控制输入是解析函数。我们通过构造反馈控制器保证其渐近轨道稳定性来证明该解的可镇定性。然而，对于该解以及一个广泛的类似解，均不存在满足传统定义的 VHC。这一观察结果呼吁我们重新考虑 VHC 的定义，这可能显著扩展 VHC 在运动规划中的实际应用范围。 

---
# VISTA: Generative Visual Imagination for Vision-and-Language Navigation 

**Title (ZH)**: VISTA：视觉与语言导航中的生成性视觉想象 

**Authors**: Yanjia Huang, Mingyang Wu, Renjie Li, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07868)  

**Abstract**: Vision-and-Language Navigation (VLN) tasks agents with locating specific objects in unseen environments using natural language instructions and visual cues. Many existing VLN approaches typically follow an 'observe-and-reason' schema, that is, agents observe the environment and decide on the next action to take based on the visual observations of their surroundings. They often face challenges in long-horizon scenarios due to limitations in immediate observation and vision-language modality gaps. To overcome this, we present VISTA, a novel framework that employs an 'imagine-and-align' navigation strategy. Specifically, we leverage the generative prior of pre-trained diffusion models for dynamic visual imagination conditioned on both local observations and high-level language instructions. A Perceptual Alignment Filter module then grounds these goal imaginations against current observations, guiding an interpretable and structured reasoning process for action selection. Experiments show that VISTA sets new state-of-the-art results on Room-to-Room (R2R) and RoboTHOR benchmarks, e.g.,+3.6% increase in Success Rate on R2R. Extensive ablation analysis underscores the value of integrating forward-looking imagination, perceptual alignment, and structured reasoning for robust navigation in long-horizon environments. 

**Abstract (ZH)**: Vision-and-Language Navigation (VLN)任务要求代理使用自然语言指令和视觉线索在未见过的环境中定位特定物体。许多现有的VLN方法通常遵循一种“观察与推理”模式，即代理观察环境并在基于周围视觉观察的基础上决定下一步采取的动作。它们在长期展望场景中常常面临挑战，这是因为即时观察的局限性和视觉语言模态之间的差距。为了克服这一问题，我们提出了一种新颖的VISTA框架，采用了一种“设想与对齐”的导航策略。具体而言，我们利用预训练扩散模型的生成先验，在结合局部观察和高层次语言指令的情况下进行动态视觉设想。然后，感知对齐滤波器模块将这些目标设想与当前观察相对接，指导可解释且结构化的推理过程以进行动作选择。实验结果显示，VISTA在Room-to-Room (R2R)和RoboTHOR基准测试中设立了新的最先进技术指标，例如在R2R中成功率为例，提高了3.6%。广泛的消融分析强调了融合前瞻想象、感知对齐和结构化推理对于在长期展望环境中实现鲁棒导航的价值。 

---
# A Physics-informed End-to-End Occupancy Framework for Motion Planning of Autonomous Vehicles 

**Title (ZH)**: 基于物理信息的一体化占用率框架用于自主车辆运动规划 

**Authors**: Shuqi Shen, Junjie Yang, Hongliang Lu, Hui Zhong, Qiming Zhang, Xinhu Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07855)  

**Abstract**: Accurate and interpretable motion planning is essential for autonomous vehicles (AVs) navigating complex and uncertain environments. While recent end-to-end occupancy prediction methods have improved environmental understanding, they typically lack explicit physical constraints, limiting safety and generalization. In this paper, we propose a unified end-to-end framework that integrates verifiable physical rules into the occupancy learning process. Specifically, we embed artificial potential fields (APF) as physics-informed guidance during network training to ensure that predicted occupancy maps are both data-efficient and physically plausible. Our architecture combines convolutional and recurrent neural networks to capture spatial and temporal dependencies while preserving model flexibility. Experimental results demonstrate that our method improves task completion rate, safety margins, and planning efficiency across diverse driving scenarios, confirming its potential for reliable deployment in real-world AV systems. 

**Abstract (ZH)**: 准确可解释的运动规划对于自主车辆（AVs）在复杂和不确定环境中的导航至关重要。尽管最近的端到端占用率预测方法提高了环境理解能力，但它们通常缺乏明确的物理约束，限制了安全性和泛化能力。在本文中，我们提出了一种统一的端到端框架，将可验证的物理规则整合到占用率学习过程中。具体来说，我们在网络训练过程中嵌入人工势场（APF）作为物理导向，以确保预测的占用率图既高效又符合物理原理。我们的架构结合了卷积和循环神经网络，以捕捉空间和时间依赖性并保持模型的灵活性。实验结果表明，我们的方法在各种驾驶场景中提高了任务完成率、安全裕度和规划效率，证实了其在实际AV系统中可靠部署的潜力。 

---
# PierGuard: A Planning Framework for Underwater Robotic Inspection of Coastal Piers 

**Title (ZH)**: PierGuard: 一种用于海岸码头水下机器人检查的规划框架 

**Authors**: Pengyu Wang, Hin Wang Lin, Jialu Li, Jiankun Wang, Ling Shi, Max Q.-H. Meng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07845)  

**Abstract**: Using underwater robots instead of humans for the inspection of coastal piers can enhance efficiency while reducing risks. A key challenge in performing these tasks lies in achieving efficient and rapid path planning within complex environments. Sampling-based path planning methods, such as Rapidly-exploring Random Tree* (RRT*), have demonstrated notable performance in high-dimensional spaces. In recent years, researchers have begun designing various geometry-inspired heuristics and neural network-driven heuristics to further enhance the effectiveness of RRT*. However, the performance of these general path planning methods still requires improvement when applied to highly cluttered underwater environments. In this paper, we propose PierGuard, which combines the strengths of bidirectional search and neural network-driven heuristic regions. We design a specialized neural network to generate high-quality heuristic regions in cluttered maps, thereby improving the performance of the path planning. Through extensive simulation and real-world ocean field experiments, we demonstrate the effectiveness and efficiency of our proposed method compared with previous research. Our method achieves approximately 2.6 times the performance of the state-of-the-art geometric-based sampling method and nearly 4.9 times that of the state-of-the-art learning-based sampling method. Our results provide valuable insights for the automation of pier inspection and the enhancement of maritime safety. The updated experimental video is available in the supplementary materials. 

**Abstract (ZH)**: 使用水下机器人代替人类对沿海码头进行检查可以提高效率并减少风险：结合双向搜索和神经网络驱动启发式区域的PierGuard方法 

---
# Optimal Trajectory Planning with Collision Avoidance for Autonomous Vehicle Maneuvering 

**Title (ZH)**: 自主车辆机动中的最优轨迹规划与碰撞避免 

**Authors**: Jason Zalev  

**Link**: [PDF](https://arxiv.org/pdf/2505.08724)  

**Abstract**: To perform autonomous driving maneuvers, such as parallel or perpendicular parking, a vehicle requires continual speed and steering adjustments to follow a generated path. In consequence, the path's quality is a limiting factor of the vehicle maneuver's performance. While most path planning approaches include finding a collision-free route, optimal trajectory planning involves solving the best transition from initial to final states, minimizing the action over all paths permitted by a kinematic model. Here we propose a novel method based on sequential convex optimization, which permits flexible and efficient optimal trajectory generation. The objective is to achieve the fastest time, shortest distance, and fewest number of path segments to satisfy motion requirements, while avoiding sensor blind-spots. In our approach, vehicle kinematics are represented by a discretized Dubins model. To avoid collisions, each waypoint is constrained by linear inequalities representing closest distance of obstacles to a polygon specifying the vehicle's extent. To promote smooth and valid trajectories, the solved kinematic state and control variables are constrained and regularized by penalty terms in the model's cost function, which enforces physical restrictions including limits for steering angle, acceleration and speed. In this paper, we analyze trajectories obtained for several parking scenarios. Results demonstrate efficient and collision-free motion generated by the proposed technique. 

**Abstract (ZH)**: 基于顺序凸优化的自主驾驶机动中的最优轨迹生成方法 

---
# DLO-Splatting: Tracking Deformable Linear Objects Using 3D Gaussian Splatting 

**Title (ZH)**: 基于3D高斯点云跟踪可变形线性对象 

**Authors**: Holly Dinkel, Marcel Büsching, Alberta Longhini, Brian Coltin, Trey Smith, Danica Kragic, Mårten Björkman, Timothy Bretl  

**Link**: [PDF](https://arxiv.org/pdf/2505.08644)  

**Abstract**: This work presents DLO-Splatting, an algorithm for estimating the 3D shape of Deformable Linear Objects (DLOs) from multi-view RGB images and gripper state information through prediction-update filtering. The DLO-Splatting algorithm uses a position-based dynamics model with shape smoothness and rigidity dampening corrections to predict the object shape. Optimization with a 3D Gaussian Splatting-based rendering loss iteratively renders and refines the prediction to align it with the visual observations in the update step. Initial experiments demonstrate promising results in a knot tying scenario, which is challenging for existing vision-only methods. 

**Abstract (ZH)**: DLO-Splatting: 一种基于多视图RGB图像和夹爪状态信息估计可变形线性对象3D形状的算法 

---
# Cost Function Estimation Using Inverse Reinforcement Learning with Minimal Observations 

**Title (ZH)**: 基于最少观测使用的逆强化学习成本函数估计 

**Authors**: Sarmad Mehrdad, Avadesh Meduri, Ludovic Righetti  

**Link**: [PDF](https://arxiv.org/pdf/2505.08619)  

**Abstract**: We present an iterative inverse reinforcement learning algorithm to infer optimal cost functions in continuous spaces. Based on a popular maximum entropy criteria, our approach iteratively finds a weight improvement step and proposes a method to find an appropriate step size that ensures learned cost function features remain similar to the demonstrated trajectory features. In contrast to similar approaches, our algorithm can individually tune the effectiveness of each observation for the partition function and does not need a large sample set, enabling faster learning. We generate sample trajectories by solving an optimal control problem instead of random sampling, leading to more informative trajectories. The performance of our method is compared to two state of the art algorithms to demonstrate its benefits in several simulated environments. 

**Abstract (ZH)**: 我们提出了一个迭代逆强化学习算法以在连续空间中推断最优成本函数。基于流行的最大熵标准，该方法迭代地寻找一个权重改进步骤，并提出了一种方法来找到一个适当的步长，以确保学习到的成本函数特征与展示轨迹特征保持相似。与相似的方法不同，我们的算法可以单独调整每个观察对分区函数的有效性，并不需要大量样本集，从而实现更快的学习。我们通过求解最优控制问题生成样本轨迹，而不是随机采样，从而得到更具信息量的轨迹。我们将该方法的性能与两种最先进的算法进行了比较，以在多个模拟环境中展示其优势。 

---
# MESSI: A Multi-Elevation Semantic Segmentation Image Dataset of an Urban Environment 

**Title (ZH)**: MESSI：城市环境多 elevation 语义分割图像数据集 

**Authors**: Barak Pinkovich, Boaz Matalon, Ehud Rivlin, Hector Rotstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.08589)  

**Abstract**: This paper presents a Multi-Elevation Semantic Segmentation Image (MESSI) dataset comprising 2525 images taken by a drone flying over dense urban environments. MESSI is unique in two main features. First, it contains images from various altitudes, allowing us to investigate the effect of depth on semantic segmentation. Second, it includes images taken from several different urban regions (at different altitudes). This is important since the variety covers the visual richness captured by a drone's 3D flight, performing horizontal and vertical maneuvers. MESSI contains images annotated with location, orientation, and the camera's intrinsic parameters and can be used to train a deep neural network for semantic segmentation or other applications of interest (e.g., localization, navigation, and tracking). This paper describes the dataset and provides annotation details. It also explains how semantic segmentation was performed using several neural network models and shows several relevant statistics. MESSI will be published in the public domain to serve as an evaluation benchmark for semantic segmentation using images captured by a drone or similar vehicle flying over a dense urban environment. 

**Abstract (ZH)**: 多海拔语义分割数据集（MESSI）：用于密集城市环境无人机拍摄图像的语义分割研究 

---
# Achieving Scalable Robot Autonomy via neurosymbolic planning using lightweight local LLM 

**Title (ZH)**: 基于轻量级局部LLM的神经符号规划实现可扩展机器人自主性 

**Authors**: Nicholas Attolino, Alessio Capitanelli, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2505.08492)  

**Abstract**: PDDL-based symbolic task planning remains pivotal for robot autonomy yet struggles with dynamic human-robot collaboration due to scalability, re-planning demands, and delayed plan availability. Although a few neurosymbolic frameworks have previously leveraged LLMs such as GPT-3 to address these challenges, reliance on closed-source, remote models with limited context introduced critical constraints: third-party dependency, inconsistent response times, restricted plan length and complexity, and multi-domain scalability issues. We present Gideon, a novel framework that enables the transition to modern, smaller, local LLMs with extended context length. Gideon integrates a novel problem generator to systematically generate large-scale datasets of realistic domain-problem-plan tuples for any domain, and adapts neurosymbolic planning for local LLMs, enabling on-device execution and extended context for multi-domain support. Preliminary experiments in single-domain scenarios performed on Qwen-2.5 1.5B and trained on 8k-32k samples, demonstrate a valid plan percentage of 66.1% (32k model) and show that the figure can be further scaled through additional data. Multi-domain tests on 16k samples yield an even higher 70.6% planning validity rate, proving extensibility across domains and signaling that data variety can have a positive effect on learning efficiency. Although long-horizon planning and reduced model size make Gideon training much less efficient than baseline models based on larger LLMs, the results are still significant considering that the trained model is about 120x smaller than baseline and that significant advantages can be achieved in inference efficiency, scalability, and multi-domain adaptability, all critical factors in human-robot collaboration. Training inefficiency can be mitigated by Gideon's streamlined data generation pipeline. 

**Abstract (ZH)**: 基于PDDL的符号任务规划在机器人自主性中仍然至关重要，但由于可扩展性、重新规划需求和计划可用延迟问题，在动态人机协作中仍面临挑战。尽管之前有一些基于神经符号框架利用了如GPT-3之类的LLM来应对这些挑战，但依赖于远程的闭源模型带来了关键约束：第三方依赖、响应时间不一致、计划长度和复杂性限制，以及多域扩展问题。我们提出了Gideon，一种新型框架，使其能够转向现代的小型本地LLM并扩展上下文长度。Gideon集成了一个新型问题生成器，以系统地为任何领域生成大规模真实领域-问题-计划三元组数据集，并适应本地LLM的神经符号规划，从而实现设备上执行和多域支持的增强上下文。初步实验在Qwen-2.5 1.5B上使用16k至32k样本进行，显示32k模型的有效计划百分比为66.1%，并通过额外数据可以进一步扩大规模。多域测试使用16k样本得到更高的70.6%的有效规划率，证明了跨领域的可扩展性，并表明数据多样性可以提高学习效率。尽管Gideon的长期规划和较小模型尺寸使其训练效率远低于基于大型LLM的基线模型，但与基线模型相比，训练后的模型小约120倍，仍然在推理效率、扩展性和多域适应性等关键因素方面获得了显著优势。Gideon简化的数据生成管道可以缓解训练效率较低的问题。 

---
# A spherical amplitude-phase formulation for 3-D adaptive line-of-sight (ALOS) guidance with USGES stability guarantees 

**Title (ZH)**: 一种球面幅度相位公式在USGES稳定性保证下的3D自适应瞄准轴（ALOS）制导 

**Authors**: Erlend M. Coates, Thor I. Fossen  

**Link**: [PDF](https://arxiv.org/pdf/2505.08344)  

**Abstract**: A recently proposed 3-D adaptive line-of-sight (ALOS) path-following algorithm addressed coupled motion dynamics of marine craft, aircraft, and uncrewed vehicles under environmental disturbances such as wind, waves, and ocean currents. Stability analysis established uniform semiglobal exponential stability (USGES) of the cross- and vertical-track errors using a body-velocity-based amplitude-phase representation of the North-East-Down (NED) kinematic differential equations. In this brief paper, we revisit the ALOS framework and introduce a novel spherical amplitude-phase representation. This formulation yields a more geometrically intuitive and physically observable description of the guidance errors and enables a significantly simplified stability proof. Unlike the previous model, which relied on a vertical crab angle derived from body-frame velocities, the new representation uses an alternative vertical crab angle and retains the USGES property. It also removes restrictive assumptions such as constant altitude/depth or zero horizontal crab angle, and remains valid for general 3-D maneuvers with nonzero roll, pitch, and flight-path angles. 

**Abstract (ZH)**: 一种 Recently 提出的 3D 自适应视线路径跟踪算法在风、波浪和洋流等环境扰动下，处理了船舶、航空器和无人驾驶车辆的耦合运动动力学。稳定性分析利用基于体速度的幅度-相位表示的北东下（NED）运动微分方程，建立了交叉轨和垂直轨误差的均匀半全局指数稳定性（USGES）。本文重新审视了 ALOS 框架，并引入了一种新的球面幅度-相位表示。这种表示提供了更几何直观且物理上可观察的引导误差描述，并使得稳定性证明更为简化。与之前模型依赖于从体速度导出的垂直蟹角不同，新的表示使用了替代的垂直蟹角，并保留了 USGES 属性。此外，该表示消除了恒定高度/深度或零水平蟹角等限制假设，并适用于一般三维机动，其中滚转、俯仰和攻角均不为零。 

---
# SLAG: Scalable Language-Augmented Gaussian Splatting 

**Title (ZH)**: SLAG：可扩展的语言增强高斯绘制技术 

**Authors**: Laszlo Szilagyi, Francis Engelmann, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2505.08124)  

**Abstract**: Language-augmented scene representations hold great promise for large-scale robotics applications such as search-and-rescue, smart cities, and mining. Many of these scenarios are time-sensitive, requiring rapid scene encoding while also being data-intensive, necessitating scalable solutions. Deploying these representations on robots with limited computational resources further adds to the challenge. To address this, we introduce SLAG, a multi-GPU framework for language-augmented Gaussian splatting that enhances the speed and scalability of embedding large scenes. Our method integrates 2D visual-language model features into 3D scenes using SAM and CLIP. Unlike prior approaches, SLAG eliminates the need for a loss function to compute per-Gaussian language embeddings. Instead, it derives embeddings from 3D Gaussian scene parameters via a normalized weighted average, enabling highly parallelized scene encoding. Additionally, we introduce a vector database for efficient embedding storage and retrieval. Our experiments show that SLAG achieves an 18 times speedup in embedding computation on a 16-GPU setup compared to OpenGaussian, while preserving embedding quality on the ScanNet and LERF datasets. For more details, visit our project website: this https URL. 

**Abstract (ZH)**: 语言增强的场景表示在大型机器人应用中的巨大潜力，如搜救、智慧城市和采矿等领域。许多这些场景是时间敏感的，要求快速的场景编码，同时也数据密集，需要可扩展的解决方案。在计算资源有限的机器人上部署这些表示进一步增加了挑战。为了解决这个问题，我们引入了SLAG，一种基于多GPU的语言增强高斯点积框架，可以提高大型场景嵌入的速度和可扩展性。我们的方法使用SAM和CLIP将2D视觉-语言模型特征集成到3D场景中。与先前的方法不同，SLAG不需要损失函数来计算每个高斯的语言嵌入，而是通过规范化加权平均从3D高斯场景参数中衍生嵌入，从而实现高效的并行化场景编码。此外，我们还引入了一个向量数据库以实现高效的嵌入存储和检索。我们的实验表明，在16-GPU设置下，SLAG在嵌入计算方面比OpenGaussian快18倍，同时在ScanNet和LERF数据集上保持了嵌入质量。欲了解更多信息，请访问我们的项目网站：this https URL。 

---
# Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories 

**Title (ZH)**: 基于图的楼层分离方法：节点嵌入与WiFi轨迹聚类 

**Authors**: Rabia Yasa Kostas, Kahraman Kostas  

**Link**: [PDF](https://arxiv.org/pdf/2505.08088)  

**Abstract**: Indoor positioning systems (IPSs) are increasingly vital for location-based services in complex multi-storey environments. This study proposes a novel graph-based approach for floor separation using Wi-Fi fingerprint trajectories, addressing the challenge of vertical localization in indoor settings. We construct a graph where nodes represent Wi-Fi fingerprints, and edges are weighted by signal similarity and contextual transitions. Node2Vec is employed to generate low-dimensional embeddings, which are subsequently clustered using K-means to identify distinct floors. Evaluated on the Huawei University Challenge 2021 dataset, our method outperforms traditional community detection algorithms, achieving an accuracy of 68.97%, an F1- score of 61.99%, and an Adjusted Rand Index of 57.19%. By publicly releasing the preprocessed dataset and implementation code, this work contributes to advancing research in indoor positioning. The proposed approach demonstrates robustness to signal noise and architectural complexities, offering a scalable solution for floor-level localization. 

**Abstract (ZH)**: 基于Wi-Fi指纹轨迹的图表示方法在室内楼层分割中的应用：一种垂直定位的新型图基方法 

---
# Pose Estimation for Intra-cardiac Echocardiography Catheter via AI-Based Anatomical Understanding 

**Title (ZH)**: 基于AI解剖理解的体内超声心动图导管姿态估计 

**Authors**: Jaeyoung Huh, Ankur Kapoor, Young-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.07851)  

**Abstract**: Intra-cardiac Echocardiography (ICE) plays a crucial role in Electrophysiology (EP) and Structural Heart Disease (SHD) interventions by providing high-resolution, real-time imaging of cardiac structures. However, existing navigation methods rely on electromagnetic (EM) tracking, which is susceptible to interference and position drift, or require manual adjustments based on operator expertise. To overcome these limitations, we propose a novel anatomy-aware pose estimation system that determines the ICE catheter position and orientation solely from ICE images, eliminating the need for external tracking sensors. Our approach leverages a Vision Transformer (ViT)-based deep learning model, which captures spatial relationships between ICE images and anatomical structures. The model is trained on a clinically acquired dataset of 851 subjects, including ICE images paired with position and orientation labels normalized to the left atrium (LA) mesh. ICE images are patchified into 16x16 embeddings and processed through a transformer network, where a [CLS] token independently predicts position and orientation via separate linear layers. The model is optimized using a Mean Squared Error (MSE) loss function, balancing positional and orientational accuracy. Experimental results demonstrate an average positional error of 9.48 mm and orientation errors of (16.13 deg, 8.98 deg, 10.47 deg) across x, y, and z axes, confirming the model accuracy. Qualitative assessments further validate alignment between predicted and target views within 3D cardiac meshes. This AI-driven system enhances procedural efficiency, reduces operator workload, and enables real-time ICE catheter localization for tracking-free procedures. The proposed method can function independently or complement existing mapping systems like CARTO, offering a transformative approach to ICE-guided interventions. 

**Abstract (ZH)**: 基于内心脏超影像的解剖感知姿态估计算法及应用 

---
