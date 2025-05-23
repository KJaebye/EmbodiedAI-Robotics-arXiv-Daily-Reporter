# Empirical Analysis of Sim-and-Real Cotraining Of Diffusion Policies For Planar Pushing from Pixels 

**Title (ZH)**: 基于像素的平面推物任务中模拟与真实联合训练扩散策略的实证分析 

**Authors**: Adam Wei, Abhinav Agarwal, Boyuan Chen, Rohan Bosworth, Nicholas Pfaff, Russ Tedrake  

**Link**: [PDF](https://arxiv.org/pdf/2503.22634)  

**Abstract**: In imitation learning for robotics, cotraining with demonstration data generated both in simulation and on real hardware has emerged as a powerful recipe to overcome the sim2real gap. This work seeks to elucidate basic principles of this sim-and-real cotraining to help inform simulation design, sim-and-real dataset creation, and policy training. Focusing narrowly on the canonical task of planar pushing from camera inputs enabled us to be thorough in our study. These experiments confirm that cotraining with simulated data \emph{can} dramatically improve performance in real, especially when real data is limited. Performance gains scale with simulated data, but eventually plateau; real-world data increases this performance ceiling. The results also suggest that reducing the domain gap in physics may be more important than visual fidelity for non-prehensile manipulation tasks. Perhaps surprisingly, having some visual domain gap actually helps the cotrained policy -- binary probes reveal that high-performing policies learn to distinguish simulated domains from real. We conclude by investigating this nuance and mechanisms that facilitate positive transfer between sim-and-real. In total, our experiments span over 40 real-world policies (evaluated on 800+ trials) and 200 simulated policies (evaluated on 40,000+ trials). 

**Abstract (ZH)**: 在机器人学中通过模拟与真实硬件数据协同训练以克服模拟到现实的差距：探究基本原理及其对仿真设计、仿真与现实数据集创建及策略训练的指导意义。 

---
# Control of Humanoid Robots with Parallel Mechanisms using Kinematic Actuation Models 

**Title (ZH)**: 基于并行机构动力学模型的人形机器人控制 

**Authors**: Victor Lutz, Ludovic de Matteïs, Virgile Batto, Nicolas Mansard  

**Link**: [PDF](https://arxiv.org/pdf/2503.22459)  

**Abstract**: Inspired by the mechanical design of Cassie, several recently released humanoid robots are using actuator configuration in which the motor is displaced from the joint location to optimize the leg inertia. This in turn induces a non linearity in the reduction ratio of the transmission which is often neglected when computing the robot motion (e.g. by trajectory optimization or reinforcement learning) and only accounted for at control time. This paper proposes an analytical method to efficiently handle this non-linearity. Using this actuation model, we demonstrate that we can leverage the dynamic abilities of the non-linear transmission while only modeling the inertia of the main serial chain of the leg, without approximating the motor capabilities nor the joint range. Based on analytical inverse kinematics, our method does not need any numerical routines dedicated to the closed-kinematics actuation, hence leading to very efficient computations. Our study focuses on two mechanisms widely used in recent humanoid robots; the four bar knee linkage as well as a parallel 2 DoF ankle mechanism. We integrate these models inside optimization based (DDP) and learning (PPO) control approaches. A comparison of our model against a simplified model that completely neglects closed chains is then shown in simulation. 

**Abstract (ZH)**: 基于卡西机器人机械设计的类人机器人腿部驱动配置及其非线性传动比的解析处理方法 

---
# FLAM: Foundation Model-Based Body Stabilization for Humanoid Locomotion and Manipulation 

**Title (ZH)**: FLAM: 基于基础模型的人形运动和操作中的姿态稳定化 

**Authors**: Xianqi Zhang, Hongliang Wei, Wenrui Wang, Xingtao Wang, Xiaopeng Fan, Debin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2503.22249)  

**Abstract**: Humanoid robots have attracted significant attention in recent years. Reinforcement Learning (RL) is one of the main ways to control the whole body of humanoid robots. RL enables agents to complete tasks by learning from environment interactions, guided by task rewards. However, existing RL methods rarely explicitly consider the impact of body stability on humanoid locomotion and manipulation. Achieving high performance in whole-body control remains a challenge for RL methods that rely solely on task rewards. In this paper, we propose a Foundation model-based method for humanoid Locomotion And Manipulation (FLAM for short). FLAM integrates a stabilizing reward function with a basic policy. The stabilizing reward function is designed to encourage the robot to learn stable postures, thereby accelerating the learning process and facilitating task completion. Specifically, the robot pose is first mapped to the 3D virtual human model. Then, the human pose is stabilized and reconstructed through a human motion reconstruction model. Finally, the pose before and after reconstruction is used to compute the stabilizing reward. By combining this stabilizing reward with the task reward, FLAM effectively guides policy learning. Experimental results on a humanoid robot benchmark demonstrate that FLAM outperforms state-of-the-art RL methods, highlighting its effectiveness in improving stability and overall performance. 

**Abstract (ZH)**: 基于基础模型的人形机器人步行与 manipulation 方法（FLAM） 

---
# REMAC: Self-Reflective and Self-Evolving Multi-Agent Collaboration for Long-Horizon Robot Manipulation 

**Title (ZH)**: REMAC: 具有自我反思和自我进化的多代理协作机器人长时 horizon 操控 

**Authors**: Puzhen Yuan, Angyuan Ma, Yunchao Yao, Huaxiu Yao, Masayoshi Tomizuka, Mingyu Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.22122)  

**Abstract**: Vision-language models (VLMs) have demonstrated remarkable capabilities in robotic planning, particularly for long-horizon tasks that require a holistic understanding of the environment for task decomposition. Existing methods typically rely on prior environmental knowledge or carefully designed task-specific prompts, making them struggle with dynamic scene changes or unexpected task conditions, e.g., a robot attempting to put a carrot in the microwave but finds the door was closed. Such challenges underscore two critical issues: adaptability and efficiency. To address them, in this work, we propose an adaptive multi-agent planning framework, termed REMAC, that enables efficient, scene-agnostic multi-robot long-horizon task planning and execution through continuous reflection and self-evolution. REMAC incorporates two key modules: a self-reflection module performing pre-condition and post-condition checks in the loop to evaluate progress and refine plans, and a self-evolvement module dynamically adapting plans based on scene-specific reasoning. It offers several appealing benefits: 1) Robots can initially explore and reason about the environment without complex prompt design. 2) Robots can keep reflecting on potential planning errors and adapting the plan based on task-specific insights. 3) After iterations, a robot can call another one to coordinate tasks in parallel, maximizing the task execution efficiency. To validate REMAC's effectiveness, we build a multi-agent environment for long-horizon robot manipulation and navigation based on RoboCasa, featuring 4 task categories with 27 task styles and 50+ different objects. Based on it, we further benchmark state-of-the-art reasoning models, including DeepSeek-R1, o3-mini, QwQ, and Grok3, demonstrating REMAC's superiority by boosting average success rates by 40% and execution efficiency by 52.7% over the single robot baseline. 

**Abstract (ZH)**: 视觉-语言模型在机器人规划中的应用：面向长时程任务的自适应多 Agent 计划框架 REMAC 

---
# Bresa: Bio-inspired Reflexive Safe Reinforcement Learning for Contact-Rich Robotic Tasks 

**Title (ZH)**: Bresa：基于生物启发的反应式安全强化学习方法，适用于高接触机器人任务 

**Authors**: Heng Zhang, Gokhan Solak, Arash Ajoudani  

**Link**: [PDF](https://arxiv.org/pdf/2503.21989)  

**Abstract**: Ensuring safety in reinforcement learning (RL)-based robotic systems is a critical challenge, especially in contact-rich tasks within unstructured environments. While the state-of-the-art safe RL approaches mitigate risks through safe exploration or high-level recovery mechanisms, they often overlook low-level execution safety, where reflexive responses to potential hazards are crucial. Similarly, variable impedance control (VIC) enhances safety by adjusting the robot's mechanical response, yet lacks a systematic way to adapt parameters, such as stiffness and damping throughout the task. In this paper, we propose Bresa, a Bio-inspired Reflexive Hierarchical Safe RL method inspired by biological reflexes. Our method decouples task learning from safety learning, incorporating a safety critic network that evaluates action risks and operates at a higher frequency than the task solver. Unlike existing recovery-based methods, our safety critic functions at a low-level control layer, allowing real-time intervention when unsafe conditions arise. The task-solving RL policy, running at a lower frequency, focuses on high-level planning (decision-making), while the safety critic ensures instantaneous safety corrections. We validate Bresa on multiple tasks including a contact-rich robotic task, demonstrating its reflexive ability to enhance safety, and adaptability in unforeseen dynamic environments. Our results show that Bresa outperforms the baseline, providing a robust and reflexive safety mechanism that bridges the gap between high-level planning and low-level execution. Real-world experiments and supplementary material are available at project website this https URL. 

**Abstract (ZH)**: 确保基于强化学习（RL）的机器人系统安全是关键挑战，尤其在无结构环境中进行接触丰富的任务时。现有的先进安全RL方法通过安全探索或高层恢复机制减轻风险，但往往忽略了低层级执行安全，而在潜在危害面前的反射性响应至关重要。同样，阻抗调节控制（VIC）通过调整机器人机械响应来增强安全，但缺乏系统的方法来适应参数，如刚度和阻尼。在本文中，我们提出了Bresa，一种受生物反射启发的反射性层次安全RL方法。该方法将任务学习与安全学习分离，引入了一个安全评判网络来评估动作风险，其运行频率高于任务解决器。与现有的基于恢复的方法不同，我们的安全评判在网络控制层工作，可以在不安全条件出现时实时干预。任务解决的RL策略以较低的频率运行，专注于高层规划（决策制定），而安全评判确保即时的安全修正。我们通过包括接触丰富的机器人任务在内的多项任务验证了Bresa，展示了其反射性增强安全能力和在未预见的动态环境中的适应性。我们的结果表明，Bresa优于基线，提供了将高层规划与低层执行联系起来的稳健且反射性的安全机制。实验证明和补充材料可在项目网站上获取：this https URL。 

---
# Pretrained Bayesian Non-parametric Knowledge Prior in Robotic Long-Horizon Reinforcement Learning 

**Title (ZH)**: 预训练贝叶斯非参数先验知识在机器人长 horizon 强化学习中的应用 

**Authors**: Yuan Meng, Xiangtong Yao, Kejia Chen, Yansong Wu, Liding Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21975)  

**Abstract**: Reinforcement learning (RL) methods typically learn new tasks from scratch, often disregarding prior knowledge that could accelerate the learning process. While some methods incorporate previously learned skills, they usually rely on a fixed structure, such as a single Gaussian distribution, to define skill priors. This rigid assumption can restrict the diversity and flexibility of skills, particularly in complex, long-horizon tasks. In this work, we introduce a method that models potential primitive skill motions as having non-parametric properties with an unknown number of underlying features. We utilize a Bayesian non-parametric model, specifically Dirichlet Process Mixtures, enhanced with birth and merge heuristics, to pre-train a skill prior that effectively captures the diverse nature of skills. Additionally, the learned skills are explicitly trackable within the prior space, enhancing interpretability and control. By integrating this flexible skill prior into an RL framework, our approach surpasses existing methods in long-horizon manipulation tasks, enabling more efficient skill transfer and task success in complex environments. Our findings show that a richer, non-parametric representation of skill priors significantly improves both the learning and execution of challenging robotic tasks. All data, code, and videos are available at this https URL. 

**Abstract (ZH)**: 强化学习方法通常从头学习新任务，往往忽视可以加速学习过程的先验知识。虽然有些方法整合了先前学习的技能，但它们通常依赖于固定的结构，如单一的高斯分布，来定义技能先验。这种刚性假设可能会限制技能的多样性和灵活性，特别是在复杂的长期任务中。在本文中，我们提出了一种方法，将其潜在的基本技能运动模型化为具有非参数特性的未知数量的底层特征。我们利用Bayesian非参数模型，特别是Dirichlet过程混合模型，并结合出生和合并启发式方法，预训练一个能有效捕捉技能多样性的技能先验。此外，学习到的技能在先验空间中是显式可追踪的，增强了可解释性和控制性。通过将这种灵活的技能先验整合到RL框架中，我们的方法在长期操作任务上超越了现有方法，促进了更高效的技术掌握和复杂环境中的任务成功。我们的研究结果表明，更丰富、非参数化的技能先验表示显著地提高了挑战性机器人任务的学习和执行效果。所有数据、代码和视频都可以在以下网址获取。 

---
# Data-Agnostic Robotic Long-Horizon Manipulation with Vision-Language-Guided Closed-Loop Feedback 

**Title (ZH)**: 基于视觉-语言引导的闭环反馈的无数据驱动长_horizon机械臂操作 

**Authors**: Yuan Meng, Xiangtong Yao, Haihui Ye, Yirui Zhou, Shengqiang Zhang, Zhenshan Bing, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2503.21969)  

**Abstract**: Recent advances in language-conditioned robotic manipulation have leveraged imitation and reinforcement learning to enable robots to execute tasks from human commands. However, these methods often suffer from limited generalization, adaptability, and the lack of large-scale specialized datasets, unlike data-rich domains such as computer vision, making long-horizon task execution challenging. To address these gaps, we introduce DAHLIA, a data-agnostic framework for language-conditioned long-horizon robotic manipulation, leveraging large language models (LLMs) for real-time task planning and execution. DAHLIA employs a dual-tunnel architecture, where an LLM-powered planner collaborates with co-planners to decompose tasks and generate executable plans, while a reporter LLM provides closed-loop feedback, enabling adaptive re-planning and ensuring task recovery from potential failures. Moreover, DAHLIA integrates chain-of-thought (CoT) in task reasoning and temporal abstraction for efficient action execution, enhancing traceability and robustness. Our framework demonstrates state-of-the-art performance across diverse long-horizon tasks, achieving strong generalization in both simulated and real-world scenarios. Videos and code are available at this https URL. 

**Abstract (ZH)**: 语言条件下的机器人操作 recent 进展：DAHLIA——一种基于大规模语言模型的数据无关框架，用于长时 Horizon 任务执行 

---
# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning 

**Title (ZH)**: ManipTrans: 通过残差学习高效实现灵巧双手操作转移 

**Authors**: Kailin Li, Puhao Li, Tengyu Liu, Yuyang Li, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21860)  

**Abstract**: Human hands play a central role in interacting, motivating increasing research in dexterous robotic manipulation. Data-driven embodied AI algorithms demand precise, large-scale, human-like manipulation sequences, which are challenging to obtain with conventional reinforcement learning or real-world teleoperation. To address this, we introduce ManipTrans, a novel two-stage method for efficiently transferring human bimanual skills to dexterous robotic hands in simulation. ManipTrans first pre-trains a generalist trajectory imitator to mimic hand motion, then fine-tunes a specific residual module under interaction constraints, enabling efficient learning and accurate execution of complex bimanual tasks. Experiments show that ManipTrans surpasses state-of-the-art methods in success rate, fidelity, and efficiency. Leveraging ManipTrans, we transfer multiple hand-object datasets to robotic hands, creating DexManipNet, a large-scale dataset featuring previously unexplored tasks like pen capping and bottle unscrewing. DexManipNet comprises 3.3K episodes of robotic manipulation and is easily extensible, facilitating further policy training for dexterous hands and enabling real-world deployments. 

**Abstract (ZH)**: 人类双手在交互中发挥核心作用，推动了灵巧机器人操纵研究的增加。基于数据的实体AI算法需要精确、大规模、类似人类的操纵序列，这给传统强化学习或真实世界远程操作带来了挑战。为了解决这一问题，我们提出了ManipTrans，一种新型的两阶段方法，用于高效地将人类双手技能转移到仿真中的灵巧机器人手中。ManipTrans 首先预训练一个通用轨迹模仿器来模仿手部运动，然后在交互约束下微调一个特定的残差模块，从而实现复杂双手任务的高效学习和精确执行。实验表明，ManipTrans 在成功率、保真度和效率方面超越了现有最先进的方法。利用ManipTrans，我们将多个手-物体数据集转移到机器人手中，创建了DexManipNet，这是一个大规模数据集，包含了诸如笔盖帽和瓶盖拧开等之前未探索的任务。该数据集包含了3300多个机器人操纵回合，并且易于扩展，有利于进一步训练灵巧手的策略并使其在真实世界中部署。 

---
# Scaling Laws of Scientific Discovery with AI and Robot Scientists 

**Title (ZH)**: AI和机器人科学家在科学发现中的规模律 

**Authors**: Pengsong Zhang, Heng Zhang, Huazhe Xu, Renjun Xu, Zhenting Wang, Cong Wang, Animesh Garg, Zhibin Li, Arash Ajoudani, Xinyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.22444)  

**Abstract**: The rapid evolution of scientific inquiry highlights an urgent need for groundbreaking methodologies that transcend the limitations of traditional research. Conventional approaches, bogged down by manual processes and siloed expertise, struggle to keep pace with the demands of modern discovery. We envision an autonomous generalist scientist (AGS) system-a fusion of agentic AI and embodied robotics-that redefines the research lifecycle. This system promises to autonomously navigate physical and digital realms, weaving together insights from disparate disciplines with unprecedented efficiency. By embedding advanced AI and robot technologies into every phase-from hypothesis formulation to peer-ready manuscripts-AGS could slash the time and resources needed for scientific research in diverse field. We foresee a future where scientific discovery follows new scaling laws, driven by the proliferation and sophistication of such systems. As these autonomous agents and robots adapt to extreme environments and leverage a growing reservoir of knowledge, they could spark a paradigm shift, pushing the boundaries of what's possible and ushering in an era of relentless innovation. 

**Abstract (ZH)**: 快速发展的科学研究突显了超越传统研究局限的颠覆性方法论的迫切需求。常规方法因手工流程和孤岛式专业知识而受困，难以跟上现代发现的需求。我们设想一种自主通才科学家（AGS）系统——结合代理AI和 embodiment robotics的融合——重新定义研究生命周期。该系统承诺能够自主穿梭于物理和数字领域，以前所未有的效率整合来自不同学科的见解。通过将先进的AI和机器人技术嵌入研究的每一阶段——从假设制定到可同行评审的手稿——AGS有望大幅减少在不同领域进行科学研究所需的时间和资源。我们预见一个未来，在这种系统普及和复杂性提高的推动下，科学发现将遵循新的扩展定律。随着这些自主代理和机器人适应极端环境并利用日益增多的知识库，它们可能引发范式转变，推动可能的边界，引领持续创新的时代。 

---
# CRLLK: Constrained Reinforcement Learning for Lane Keeping in Autonomous Driving 

**Title (ZH)**: CRLLK: 受约束的强化学习在自动驾驶中的车道保持 

**Authors**: Xinwei Gao, Arambam James Singh, Gangadhar Royyuru, Michael Yuhas, Arvind Easwaran  

**Link**: [PDF](https://arxiv.org/pdf/2503.22248)  

**Abstract**: Lane keeping in autonomous driving systems requires scenario-specific weight tuning for different objectives. We formulate lane-keeping as a constrained reinforcement learning problem, where weight coefficients are automatically learned along with the policy, eliminating the need for scenario-specific tuning. Empirically, our approach outperforms traditional RL in efficiency and reliability. Additionally, real-world demonstrations validate its practical value for real-world autonomous driving. 

**Abstract (ZH)**: 自主驾驶系统中的车道保持需要针对不同目标进行场景特定的权重调优。我们将车道保持建模为一个受约束的强化学习问题，在此问题中，权重系数与策略一起自动学习，消除了场景特定的调优需求。实验结果表明，与传统强化学习相比，我们的方法在效率和可靠性方面表现更优。此外，实际应用场景的演示验证了其在实际自主驾驶中的实用价值。 

---
# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models 

**Title (ZH)**: CoT-VLA：视觉链式思维推理在视觉语言行动模型中的应用 

**Authors**: Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin  

**Link**: [PDF](https://arxiv.org/pdf/2503.22020)  

**Abstract**: Vision-language-action models (VLAs) have shown potential in leveraging pretrained vision-language models and diverse robot demonstrations for learning generalizable sensorimotor control. While this paradigm effectively utilizes large-scale data from both robotic and non-robotic sources, current VLAs primarily focus on direct input--output mappings, lacking the intermediate reasoning steps crucial for complex manipulation tasks. As a result, existing VLAs lack temporal planning or reasoning capabilities. In this paper, we introduce a method that incorporates explicit visual chain-of-thought (CoT) reasoning into vision-language-action models (VLAs) by predicting future image frames autoregressively as visual goals before generating a short action sequence to achieve these goals. We introduce CoT-VLA, a state-of-the-art 7B VLA that can understand and generate visual and action tokens. Our experimental results demonstrate that CoT-VLA achieves strong performance, outperforming the state-of-the-art VLA model by 17% in real-world manipulation tasks and 6% in simulation benchmarks. Project website: this https URL 

**Abstract (ZH)**: 视觉-语言-行动模型（VLAs）展示了通过利用预训练的视觉-语言模型和多样化的机器人示范来学习可泛化的传感器运动控制的潜力。尽管这一范式有效利用了来自机器人和非机器人来源的大规模数据，目前的VLAs主要专注于直接的输入-输出映射，而缺乏完成复杂操作任务所必需的中间推理步骤。因此，现有的VLAs缺乏时间规划或推理能力。在本文中，我们引入了一种方法，通过预测未来图像帧作为视觉目标，然后生成短暂的行动序列以实现这些目标，将显式的视觉链式推理（CoT）整合到视觉-语言-行动模型（VLAs）中。我们提出了CoT-VLA，这是一种最先进的7B VLA，能够理解和生成视觉和行动标记。我们的实验结果表明，CoT-VLA表现出色，在真实世界的操作任务中比最先进的VLAS模型高出17%，在仿真基准测试中高出6%。项目网址：这个 https URL。 

---
# Threshold Adaptation in Spiking Networks Enables Shortest Path Finding and Place Disambiguation 

**Title (ZH)**: 阈值自适应在脉冲神经网络中实现最短路径寻找和位置模糊区分 

**Authors**: Robin Dietrich, Tobias Fischer, Nicolai Waniek, Nico Reeb, Michael Milford, Alois Knoll, Adam D. Hines  

**Link**: [PDF](https://arxiv.org/pdf/2503.21795)  

**Abstract**: Efficient spatial navigation is a hallmark of the mammalian brain, inspiring the development of neuromorphic systems that mimic biological principles. Despite progress, implementing key operations like back-tracing and handling ambiguity in bio-inspired spiking neural networks remains an open challenge. This work proposes a mechanism for activity back-tracing in arbitrary, uni-directional spiking neuron graphs. We extend the existing replay mechanism of the spiking hierarchical temporal memory (S-HTM) by our spike timing-dependent threshold adaptation (STDTA), which enables us to perform path planning in networks of spiking neurons. We further present an ambiguity dependent threshold adaptation (ADTA) for identifying places in an environment with less ambiguity, enhancing the localization estimate of an agent. Combined, these methods enable efficient identification of the shortest path to an unambiguous target. Our experiments show that a network trained on sequences reliably computes shortest paths with fewer replays than the steps required to reach the target. We further show that we can identify places with reduced ambiguity in multiple, similar environments. These contributions advance the practical application of biologically inspired sequential learning algorithms like the S-HTM towards neuromorphic localization and navigation. 

**Abstract (ZH)**: 高效的空间导航是哺乳动物大脑的 hallmark，启发了模仿生物原则的神经形态系统的开发。尽管取得了进展，但在生物启发的脉冲神经网络中实现回溯操作和处理模糊性仍然是一个开放挑战。本工作提出了一种机制，用于任意单向脉冲神经元图中的活动回溯。我们通过脉冲时序依赖阈值适应（STDTA）扩展了现有的脉冲多层次时间记忆（S-HTM）的回放机制，从而能够在脉冲神经元网络中进行路径规划。我们还提出了依赖模糊性的阈值适应（ADTA），用于识别环境中的模糊性较低的地点，增强代理的定位估计。结合这些方法，能够高效地识别到明确目标的最短路径。实验结果表明，训练在网络序列上可以比到达目标所需的步数更少地回放计算出最短路径。此外，我们还展示了可以在多个相似环境中识别模糊性较低的地点。这些贡献推动了像S-HTM这样的生物启发序贯学习算法在神经形态定位和导航中的实际应用。 

---
# Agent-Centric Personalized Multiple Clustering with Multi-Modal LLMs 

**Title (ZH)**: 基于代理的多模态大型语言模型个性化多聚类 

**Authors**: Ziye Chen, Yiqun Duan, Riheng Zhu, Zhenbang Sun, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2503.22241)  

**Abstract**: Personalized multiple clustering aims to generate diverse partitions of a dataset based on different user-specific aspects, rather than a single clustering. It has recently drawn research interest for accommodating varying user preferences. Recent approaches primarily use CLIP embeddings with proxy learning to extract representations biased toward user clustering preferences. However, CLIP primarily focuses on coarse image-text alignment, lacking a deep contextual understanding of user interests. To overcome these limitations, we propose an agent-centric personalized clustering framework that leverages multi-modal large language models (MLLMs) as agents to comprehensively traverse a relational graph to search for clusters based on user interests. Due to the advanced reasoning mechanism of MLLMs, the obtained clusters align more closely with user-defined criteria than those obtained from CLIP-based representations. To reduce computational overhead, we shorten the agents' traversal path by constructing a relational graph using user-interest-biased embeddings extracted by MLLMs. A large number of weakly connected edges can be filtered out based on embedding similarity, facilitating an efficient traversal search for agents. Experimental results show that the proposed method achieves NMI scores of 0.9667 and 0.9481 on the Card Order and Card Suits benchmarks, respectively, largely improving the SOTA model by over 140%. 

**Abstract (ZH)**: 个性化多聚类旨在根据不同的用户特定方面生成数据集的多样化分区，而非单一聚类。近年来，由于能够容纳不同的用户偏好，该领域引起了研究兴趣。现有的方法主要利用CLIP嵌入和代理学习来提取偏向用户聚类偏好的表示。然而，CLIP 主要关注粗粒度的图像-文本对齐，缺乏对用户兴趣的深入上下文理解。为克服这些局限性，我们提出了一种基于代理的个性化聚类框架，利用多模态大规模语言模型（MLLMs）作为代理，全面遍历关系图以根据用户兴趣搜索聚类。得益于MLLMs的高级推理机制，所获得的聚类与用户定义的标准更为一致，优于基于CLIP的表示。为了减少计算开销，我们通过使用MLLMs提取的兴趣偏向嵌入构建关系图，缩短代理的遍历路径。基于嵌入相似性可以过滤掉大量的弱连接边，便于代理的高效遍历搜索。实验结果表明，所提出的方法在Card Order和Card Suits基准上分别获得了0.9667和0.9481的NMI分数，显著优于当前最先进模型超过140%。 

---
# On the Mistaken Assumption of Interchangeable Deep Reinforcement Learning Implementations 

**Title (ZH)**: 关于可互换深度强化学习实现的错误假设 

**Authors**: Rajdeep Singh Hundal, Yan Xiao, Xiaochun Cao, Jin Song Dong, Manuel Rigger  

**Link**: [PDF](https://arxiv.org/pdf/2503.22575)  

**Abstract**: Deep Reinforcement Learning (DRL) is a paradigm of artificial intelligence where an agent uses a neural network to learn which actions to take in a given environment. DRL has recently gained traction from being able to solve complex environments like driving simulators, 3D robotic control, and multiplayer-online-battle-arena video games. Numerous implementations of the state-of-the-art algorithms responsible for training these agents, like the Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms, currently exist. However, studies make the mistake of assuming implementations of the same algorithm to be consistent and thus, interchangeable. In this paper, through a differential testing lens, we present the results of studying the extent of implementation inconsistencies, their effect on the implementations' performance, as well as their impact on the conclusions of prior studies under the assumption of interchangeable implementations. The outcomes of our differential tests showed significant discrepancies between the tested algorithm implementations, indicating that they are not interchangeable. In particular, out of the five PPO implementations tested on 56 games, three implementations achieved superhuman performance for 50% of their total trials while the other two implementations only achieved superhuman performance for less than 15% of their total trials. As part of a meticulous manual analysis of the implementations' source code, we analyzed implementation discrepancies and determined that code-level inconsistencies primarily caused these discrepancies. Lastly, we replicated a study and showed that this assumption of implementation interchangeability was sufficient to flip experiment outcomes. Therefore, this calls for a shift in how implementations are being used. 

**Abstract (ZH)**: 深度强化学习（DRL）中实现不一致性的影响研究：通过差异测试考察算法实现差异及其后果 

---
# Robust Offline Imitation Learning Through State-level Trajectory Stitching 

**Title (ZH)**: 基于状态级轨迹拼接的鲁棒离线 imitation 学习 

**Authors**: Shuze Wang, Yunpeng Mei, Hongjie Cao, Yetian Yuan, Gang Wang, Jian Sun, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2503.22524)  

**Abstract**: Imitation learning (IL) has proven effective for enabling robots to acquire visuomotor skills through expert demonstrations. However, traditional IL methods are limited by their reliance on high-quality, often scarce, expert data, and suffer from covariate shift. To address these challenges, recent advances in offline IL have incorporated suboptimal, unlabeled datasets into the training. In this paper, we propose a novel approach to enhance policy learning from mixed-quality offline datasets by leveraging task-relevant trajectory fragments and rich environmental dynamics. Specifically, we introduce a state-based search framework that stitches state-action pairs from imperfect demonstrations, generating more diverse and informative training trajectories. Experimental results on standard IL benchmarks and real-world robotic tasks showcase that our proposed method significantly improves both generalization and performance. 

**Abstract (ZH)**: 利用与任务相关的轨迹片段和丰富的环境动力学增强低质量离线数据的策略学习 

---
# EgoToM: Benchmarking Theory of Mind Reasoning from Egocentric Videos 

**Title (ZH)**: EgoToM: 基于以自我为中心视频的情绪理论推理基准测试 

**Authors**: Yuxuan Li, Vijay Veerabadran, Michael L. Iuzzolino, Brett D. Roads, Asli Celikyilmaz, Karl Ridgeway  

**Link**: [PDF](https://arxiv.org/pdf/2503.22152)  

**Abstract**: We introduce EgoToM, a new video question-answering benchmark that extends Theory-of-Mind (ToM) evaluation to egocentric domains. Using a causal ToM model, we generate multi-choice video QA instances for the Ego4D dataset to benchmark the ability to predict a camera wearer's goals, beliefs, and next actions. We study the performance of both humans and state of the art multimodal large language models (MLLMs) on these three interconnected inference problems. Our evaluation shows that MLLMs achieve close to human-level accuracy on inferring goals from egocentric videos. However, MLLMs (including the largest ones we tested with over 100B parameters) fall short of human performance when inferring the camera wearers' in-the-moment belief states and future actions that are most consistent with the unseen video future. We believe that our results will shape the future design of an important class of egocentric digital assistants which are equipped with a reasonable model of the user's internal mental states. 

**Abstract (ZH)**: EgoToM：扩展到主体中心领域的理论-of-心智视频问答基准 

---
# M-DocSum: Do LVLMs Genuinely Comprehend Interleaved Image-Text in Document Summarization? 

**Title (ZH)**: M-DocSum: LVLMs究竟在文档总结中真正理解了交错的图文信息吗？ 

**Authors**: Haolong Yan, Kaijun Tan, Yeqing Shen, Xin Huang, Zheng Ge, Xiangyu Zhang, Si Li, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2503.21839)  

**Abstract**: We investigate a critical yet under-explored question in Large Vision-Language Models (LVLMs): Do LVLMs genuinely comprehend interleaved image-text in the document? Existing document understanding benchmarks often assess LVLMs using question-answer formats, which are information-sparse and difficult to guarantee the coverage of long-range dependencies. To address this issue, we introduce a novel and challenging Multimodal Document Summarization Benchmark (M-DocSum-Bench), which comprises 500 high-quality arXiv papers, along with interleaved multimodal summaries aligned with human preferences. M-DocSum-Bench is a reference-based generation task and necessitates the generation of interleaved image-text summaries using provided reference images, thereby simultaneously evaluating capabilities in understanding, reasoning, localization, and summarization within complex multimodal document scenarios. To facilitate this benchmark, we develop an automated framework to construct summaries and propose a fine-grained evaluation method called M-DocEval. Moreover, we further develop a robust summarization baseline, i.e., M-DocSum-7B, by progressive two-stage training with diverse instruction and preference data. The extensive results on our M-DocSum-Bench reveal that the leading LVLMs struggle to maintain coherence and accurately integrate information within long and interleaved contexts, often exhibiting confusion between similar images and a lack of robustness. Notably, M-DocSum-7B achieves state-of-the-art performance compared to larger and closed-source models (including GPT-4o, Gemini Pro, Claude-3.5-Sonnet and Qwen2.5-VL-72B, etc.), demonstrating the potential of LVLMs for improved interleaved image-text understanding. The code, data, and models are available at this https URL. 

**Abstract (ZH)**: 我们研究了一个在大规模视觉-语言模型（LVLMs）中关键但尚未充分探索的问题：LVLMs是否真正理解文档中的交错图像-文本？现有的文档理解基准通常使用问答格式评估LVLMs，这种格式信息稀疏且难以保证长距离依赖关系的覆盖。为解决这一问题，我们引入了一个新颖且具有挑战性的多模态文档总结基准（M-DocSum-Bench），该基准包含500份高质量的arXiv论文，并提供了与人类偏好对齐的交错多模态摘要。M-DocSum-Bench是一个基于参考的生成任务，要求使用提供的参考图像生成交错的图像-文本摘要，从而在复杂多模态文档场景中同时评估理解和推理、定位和摘要的能力。为了支持这一基准，我们开发了一种自动化框架来构建摘要，并提出了一种细粒度评估方法M-DocEval。此外，我们通过分阶段训练和多样化指令与偏好数据进一步开发了一个稳健的总结基线，即M-DocSum-7B。我们在M-DocSum-Bench上的广泛结果表明，领先的LVLMs在长且交错的上下文中难以保持连贯并准确整合信息，往往混淆相似的图像并缺乏鲁棒性。值得注意的是，M-DocSum-7B在与更大且封闭源模型（包括GPT-4o、Gemini Pro、Claude-3.5-Sonnet和Qwen2.5-VL-72B等）比较时，取得了最先进的性能，展示了LVLMs在交错图像-文本理解方面的潜力。相关代码、数据和模型可在以下链接获取。 

---
