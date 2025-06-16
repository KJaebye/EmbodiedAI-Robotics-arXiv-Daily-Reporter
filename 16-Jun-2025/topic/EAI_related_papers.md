# mimic-one: a Scalable Model Recipe for General Purpose Robot Dexterity 

**Title (ZH)**: Mimic-One：一种面向通用机器人灵巧性的可扩展模型范式 

**Authors**: Elvis Nava, Victoriano Montesinos, Erik Bauer, Benedek Forrai, Jonas Pai, Stefan Weirich, Stephan-Daniel Gravert, Philipp Wand, Stephan Polinski, Benjamin F. Grewe, Robert K. Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.11916)  

**Abstract**: We present a diffusion-based model recipe for real-world control of a highly dexterous humanoid robotic hand, designed for sample-efficient learning and smooth fine-motor action inference. Our system features a newly designed 16-DoF tendon-driven hand, equipped with wide angle wrist cameras and mounted on a Franka Emika Panda arm. We develop a versatile teleoperation pipeline and data collection protocol using both glove-based and VR interfaces, enabling high-quality data collection across diverse tasks such as pick and place, item sorting and assembly insertion. Leveraging high-frequency generative control, we train end-to-end policies from raw sensory inputs, enabling smooth, self-correcting motions in complex manipulation scenarios. Real-world evaluations demonstrate up to 93.3% out of distribution success rates, with up to a +33.3% performance boost due to emergent self-correcting behaviors, while also revealing scaling trends in policy performance. Our results advance the state-of-the-art in dexterous robotic manipulation through a fully integrated, practical approach to hardware, learning, and real-world deployment. 

**Abstract (ZH)**: 基于扩散模型的人类手臂驱动的高度灵巧人形机器人手的现实世界控制方法：样本高效学习与平滑精细动作推理 

---
# ExoStart: Efficient learning for dexterous manipulation with sensorized exoskeleton demonstrations 

**Title (ZH)**: ExoStart: 有效学习基于传感器化外骨骼演示的灵巧操作技能 

**Authors**: Zilin Si, Jose Enrique Chen, M. Emre Karagozler, Antonia Bronars, Jonathan Hutchinson, Thomas Lampe, Nimrod Gileadi, Taylor Howell, Stefano Saliceti, Lukasz Barczyk, Ilan Olivarez Correa, Tom Erez, Mohit Shridhar, Murilo Fernandes Martins, Konstantinos Bousmalis, Nicolas Heess, Francesco Nori, Maria Bauza Villalonga  

**Link**: [PDF](https://arxiv.org/pdf/2506.11775)  

**Abstract**: Recent advancements in teleoperation systems have enabled high-quality data collection for robotic manipulators, showing impressive results in learning manipulation at scale. This progress suggests that extending these capabilities to robotic hands could unlock an even broader range of manipulation skills, especially if we could achieve the same level of dexterity that human hands exhibit. However, teleoperating robotic hands is far from a solved problem, as it presents a significant challenge due to the high degrees of freedom of robotic hands and the complex dynamics occurring during contact-rich settings. In this work, we present ExoStart, a general and scalable learning framework that leverages human dexterity to improve robotic hand control. In particular, we obtain high-quality data by collecting direct demonstrations without a robot in the loop using a sensorized low-cost wearable exoskeleton, capturing the rich behaviors that humans can demonstrate with their own hands. We also propose a simulation-based dynamics filter that generates dynamically feasible trajectories from the collected demonstrations and use the generated trajectories to bootstrap an auto-curriculum reinforcement learning method that relies only on simple sparse rewards. The ExoStart pipeline is generalizable and yields robust policies that transfer zero-shot to the real robot. Our results demonstrate that ExoStart can generate dexterous real-world hand skills, achieving a success rate above 50% on a wide range of complex tasks such as opening an AirPods case or inserting and turning a key in a lock. More details and videos can be found in this https URL. 

**Abstract (ZH)**: 最近在遥控操作系统中的进展使得对于机器人 manipulator 的高质量数据收集成为可能，展示了在大规模学习操作方面的显著成果。这项进展表明，将这些能力扩展到机器人手中可能会解锁更广泛的操作技能，特别是如果我们能够达到与人类手相同的灵巧性。然而，遥控操作机器人手远不是一个已经解决的问题，由于机器人手的高自由度以及接触丰富设置中的复杂动力学，它提出了一个重要的挑战。在本文中，我们提出了 ExoStart，一种通用且可扩展的学习框架，利用人类的灵巧性来提高机器人手的控制。特别是，我们通过使用传感器化的低成本可穿戴外骨骼收集不包含机器人闭环的操作示范，捕获人类可以展示的丰富行为。我们还提出了一种基于仿真的动力学滤波器，从收集的操作示范中生成动力学可行的轨迹，并利用生成的轨迹启动仅依赖于简单稀疏奖励的自动课程强化学习方法。ExoStart 管道具有通用性并且能够在零样本的情况下转移到真实机器人上。我们的结果显示，ExoStart 可以生成灵巧的真实世界手的技能，在诸如打开 AirPods 保护壳或插入并转动钥匙等复杂任务中取得超过 50% 的成功率。更多信息和视频可以在以下链接找到：[这里](https://www.example.com)。 

---
# CIRO7.2: A Material Network with Circularity of -7.2 and Reinforcement-Learning-Controlled Robotic Disassembler 

**Title (ZH)**: CIRO-7.2: 一种循环性为-7.2的材料网络及其 reinforcement-learning 控制拆解机器人 

**Authors**: Federico Zocco, Monica Malvezzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.11748)  

**Abstract**: The competition over natural reserves of minerals is expected to increase in part because of the linear-economy paradigm based on take-make-dispose. Simultaneously, the linear economy considers end-of-use products as waste rather than as a resource, which results in large volumes of waste whose management remains an unsolved problem. Since a transition to a circular economy can mitigate these open issues, in this paper we begin by enhancing the notion of circularity based on compartmental dynamical thermodynamics, namely, $\lambda$, and then, we model a thermodynamical material network processing a batch of 2 solid materials of criticality coefficients of 0.1 and 0.95, with a robotic disassembler compartment controlled via reinforcement learning (RL), and processing 2-7 kg of materials. Subsequently, we focused on the design of the robotic disassembler compartment using state-of-the-art RL algorithms and assessing the algorithm performance with respect to $\lambda$ (Fig. 1). The highest circularity is -2.1 achieved in the case of disassembling 2 parts of 1 kg each, whereas it reduces to -7.2 in the case of disassembling 4 parts of 1 kg each contained inside a chassis of 3 kg. Finally, a sensitivity analysis highlighted that the impact on $\lambda$ of the performance of an RL controller has a positive correlation with the quantity and the criticality of the materials to be disassembled. This work also gives the principles of the emerging research fields indicated as circular intelligence and robotics (CIRO). Source code is publicly available. 

**Abstract (ZH)**: 矿物自然保护区的竞争预计会因为线性经济 paradigm 基于取-用-丢弃模式而加剧。同时，线性经济将使用后的制品视为废物而不是资源，导致大量废物管理问题尚未解决。由于向循环经济转型可以缓解这些问题，本文首先基于隔室动力学热力学增强循环性的概念，即 $\lambda$，然后建模一个处理两种关键系数分别为0.1和0.95的固体材料的热力学材料网络，其中包含受强化学习(RL)控制的机器人拆卸隔室，并处理2-7 kg的材料。随后，我们集中于使用最先进的RL算法设计机器人拆卸隔室，并根据 $\lambda$ 评估算法性能（图1）。最高的循环性为-2.1，这种情况出现在拆卸两个各重1 kg的部件时，而当拆卸四个各重1 kg的部件（包含在3 kg的车架中）时，循环性降至-7.2。最后，敏感性分析表明，RL控制器性能对 $\lambda$ 的影响与待拆卸材料的数量和关键性之间存在正相关关系。本工作还提出了循环智能与机器人（CIRO）领域新兴研究领域的原则。源代码已公开。 

---
# Dynamic Collaborative Material Distribution System for Intelligent Robots In Smart Manufacturing 

**Title (ZH)**: 智能机器人在智能制造中的动态协作材料分配系统 

**Authors**: Ziren Xiao, Ruxin Xiao, Chang Liu, Xinheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.11723)  

**Abstract**: The collaboration and interaction of multiple robots have become integral aspects of smart manufacturing. Effective planning and management play a crucial role in achieving energy savings and minimising overall costs. This paper addresses the real-time Dynamic Multiple Sources to Single Destination (DMS-SD) navigation problem, particularly with a material distribution case for multiple intelligent robots in smart manufacturing. Enumerated solutions, such as in \cite{xiao2022efficient}, tackle the problem by generating as many optimal or near-optimal solutions as possible but do not learn patterns from the previous experience, whereas the method in \cite{xiao2023collaborative} only uses limited information from the earlier trajectories. Consequently, these methods may take a considerable amount of time to compute results on large maps, rendering real-time operations impractical. To overcome this challenge, we propose a lightweight Deep Reinforcement Learning (DRL) method to address the DMS-SD problem. The proposed DRL method can be efficiently trained and rapidly converges to the optimal solution using the designed target-guided reward function. A well-trained DRL model significantly reduces the computation time for the next movement to a millisecond level, which improves the time up to 100 times in our experiments compared to the enumerated solutions. Moreover, the trained DRL model can be easily deployed on lightweight devices in smart manufacturing, such as Internet of Things devices and mobile phones, which only require limited computational resources. 

**Abstract (ZH)**: 多机器人协作与交互在智能制造中的实时动态多源至单目的地导航问题及解决方案 

---
# Multi-Loco: Unifying Multi-Embodiment Legged Locomotion via Reinforcement Learning Augmented Diffusion 

**Title (ZH)**: Multi-Loco：通过强化学习增强扩散统一多体态腿式移动 

**Authors**: Shunpeng Yang, Zhen Fu, Zhefeng Cao, Guo Junde, Patrick Wensing, Wei Zhang, Hua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.11470)  

**Abstract**: Generalizing locomotion policies across diverse legged robots with varying morphologies is a key challenge due to differences in observation/action dimensions and system dynamics. In this work, we propose Multi-Loco, a novel unified framework combining a morphology-agnostic generative diffusion model with a lightweight residual policy optimized via reinforcement learning (RL). The diffusion model captures morphology-invariant locomotion patterns from diverse cross-embodiment datasets, improving generalization and robustness. The residual policy is shared across all embodiments and refines the actions generated by the diffusion model, enhancing task-aware performance and robustness for real-world deployment. We evaluated our method with a rich library of four legged robots in both simulation and real-world experiments. Compared to a standard RL framework with PPO, our approach -- replacing the Gaussian policy with a diffusion model and residual term -- achieves a 10.35% average return improvement, with gains up to 13.57% in wheeled-biped locomotion tasks. These results highlight the benefits of cross-embodiment data and composite generative architectures in learning robust, generalized locomotion skills. 

**Abstract (ZH)**: 跨不同形态腿式机器人泛化的运动政策生成：融合形态无关生成扩散模型和基于强化学习的轻量级残差策略的新型统一框架 

---
# Gondola: Grounded Vision Language Planning for Generalizable Robotic Manipulation 

**Title (ZH)**: Gondola: 地面指导的视觉语言规划以实现泛化的机器人操作 

**Authors**: Shizhe Chen, Ricardo Garcia, Paul Pacaud, Cordelia Schmid  

**Link**: [PDF](https://arxiv.org/pdf/2506.11261)  

**Abstract**: Robotic manipulation faces a significant challenge in generalizing across unseen objects, environments and tasks specified by diverse language instructions. To improve generalization capabilities, recent research has incorporated large language models (LLMs) for planning and action execution. While promising, these methods often fall short in generating grounded plans in visual environments. Although efforts have been made to perform visual instructional tuning on LLMs for robotic manipulation, existing methods are typically constrained by single-view image input and struggle with precise object grounding. In this work, we introduce Gondola, a novel grounded vision-language planning model based on LLMs for generalizable robotic manipulation. Gondola takes multi-view images and history plans to produce the next action plan with interleaved texts and segmentation masks of target objects and locations. To support the training of Gondola, we construct three types of datasets using the RLBench simulator, namely robot grounded planning, multi-view referring expression and pseudo long-horizon task datasets. Gondola outperforms the state-of-the-art LLM-based method across all four generalization levels of the GemBench dataset, including novel placements, rigid objects, articulated objects and long-horizon tasks. 

**Abstract (ZH)**: 基于大型语言模型的多视图接地视觉-语言规划模型Gondola及其在通用机器人操作中的应用 

---
# Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving 

**Title (ZH)**: Poutine：基于视觉-语言-轨迹预训练和训练后强化学习的稳健端到端自动驾驶 

**Authors**: Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, Liam Paull  

**Link**: [PDF](https://arxiv.org/pdf/2506.11234)  

**Abstract**: We present Poutine, a 3B-parameter vision-language model (VLM) tailored for end-to-end autonomous driving in long-tail driving scenarios. Poutine is trained in two stages. To obtain strong base driving capabilities, we train Poutine-Base in a self-supervised vision-language-trajectory (VLT) next-token prediction fashion on 83 hours of CoVLA nominal driving and 11 hours of Waymo long-tail driving. Accompanying language annotations are auto-generated with a 72B-parameter VLM. Poutine is obtained by fine-tuning Poutine-Base with Group Relative Policy Optimization (GRPO) using less than 500 preference-labeled frames from the Waymo validation set. We show that both VLT pretraining and RL fine-tuning are critical to attain strong driving performance in the long-tail. Poutine-Base achieves a rater-feedback score (RFS) of 8.12 on the validation set, nearly matching Waymo's expert ground-truth RFS. The final Poutine model achieves an RFS of 7.99 on the official Waymo test set, placing 1st in the 2025 Waymo Vision-Based End-to-End Driving Challenge by a significant margin. These results highlight the promise of scalable VLT pre-training and lightweight RL fine-tuning to enable robust and generalizable autonomy. 

**Abstract (ZH)**: Poutine：一种适用于长尾驾驶场景的端到端自动驾驶3B参数视图语言模型 

---
# Reimagining Dance: Real-time Music Co-creation between Dancers and AI 

**Title (ZH)**: 重建舞蹈：舞者与AI的实时音乐共创 

**Authors**: Olga Vechtomova, Jeff Bos  

**Link**: [PDF](https://arxiv.org/pdf/2506.12008)  

**Abstract**: Dance performance traditionally follows a unidirectional relationship where movement responds to music. While AI has advanced in various creative domains, its application in dance has primarily focused on generating choreography from musical input. We present a system that enables dancers to dynamically shape musical environments through their movements. Our multi-modal architecture creates a coherent musical composition by intelligently combining pre-recorded musical clips in response to dance movements, establishing a bidirectional creative partnership where dancers function as both performers and composers. Through correlation analysis of performance data, we demonstrate emergent communication patterns between movement qualities and audio features. This approach reconceptualizes the role of AI in performing arts as a responsive collaborator that expands possibilities for both professional dance performance and improvisational artistic expression across broader populations. 

**Abstract (ZH)**: 舞蹈表演传统上遵循单向关系，其中动作响应音乐。尽管人工智能在各种创意领域中取得了进步，其在舞蹈中的应用主要集中在从音乐输入生成编舞上。我们提出了一种系统，使舞者能够通过其动作动态塑造音乐环境。我们的多模态架构通过智能组合响应舞蹈动作的预录制音乐片段，创建一个一致的音乐作品，建立起一种双向创作伙伴关系，其中舞者既是表演者也是作曲家。通过对表演数据的相关性分析，我们展示了运动品质与音频特征之间新兴的交流模式。这种方法重新定义了人工智能在表演艺术中的角色，作为一种响应式的合作者，它扩展了专业舞蹈表演和广泛人群中的即兴艺术表达的可能性。 

---
# VGR: Visual Grounded Reasoning 

**Title (ZH)**: 视觉定位推理 VGR 

**Authors**: Jiacong Wang, Zijiang Kang, Haochen Wang, Haiyong Jiang, Jiawen Li, Bohong Wu, Ya Wang, Jiao Ran, Xiao Liang, Chao Feng, Jun Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.11991)  

**Abstract**: In the field of multimodal chain-of-thought (CoT) reasoning, existing approaches predominantly rely on reasoning on pure language space, which inherently suffers from language bias and is largely confined to math or science domains. This narrow focus limits their ability to handle complex visual reasoning tasks that demand comprehensive understanding of image details. To address these limitations, this paper introduces VGR, a novel reasoning multimodal large language model (MLLM) with enhanced fine-grained visual perception capabilities. Unlike traditional MLLMs that answer the question or reasoning solely on the language space, our VGR first detects relevant regions that may help to solve problems, and then provides precise answers based on replayed image regions. To achieve this, we conduct a large-scale SFT dataset called VGR -SFT that contains reasoning data with mixed vision grounding and language deduction. The inference pipeline of VGR allows the model to choose bounding boxes for visual reference and a replay stage is introduced to integrates the corresponding regions into the reasoning process, enhancing multimodel comprehension. Experiments on the LLaVA-NeXT-7B baseline show that VGR achieves superior performance on multi-modal benchmarks requiring comprehensive image detail understanding. Compared to the baseline, VGR uses only 30\% of the image token count while delivering scores of +4.1 on MMStar, +7.1 on AI2D, and a +12.9 improvement on ChartQA. 

**Abstract (ZH)**: 多模态链式推理中的视觉推理引导模型 (VGR) 

---
# Technical Evaluation of a Disruptive Approach in Homomorphic AI 

**Title (ZH)**: 同态AI中的颠覆性方法的技术评估 

**Authors**: Eric Filiol  

**Link**: [PDF](https://arxiv.org/pdf/2506.11954)  

**Abstract**: We present a technical evaluation of a new, disruptive cryptographic approach to data security, known as HbHAI (Hash-based Homomorphic Artificial Intelligence). HbHAI is based on a novel class of key-dependent hash functions that naturally preserve most similarity properties, most AI algorithms rely on. As a main claim, HbHAI makes now possible to analyze and process data in its cryptographically secure form while using existing native AI algorithms without modification, with unprecedented performances compared to existing homomorphic encryption schemes.
We tested various HbHAI-protected datasets (non public preview) using traditional unsupervised and supervised learning techniques (clustering, classification, deep neural networks) with classical unmodified AI algorithms. This paper presents technical results from an independent analysis conducted with those different, off-the-shelf AI algorithms. The aim was to assess the security, operability and performance claims regarding HbHAI techniques. As a results, our results confirm most these claims, with only a few minor reservations. 

**Abstract (ZH)**: 一种基于哈希的同态人工智能（HbHAI）的技術評估：/security-and-operability-assessment-of-hash-based-homomorphic-artificial-intelligence-hbhai-techniques 

---
# KoGEC : Korean Grammatical Error Correction with Pre-trained Translation Models 

**Title (ZH)**: KoGEC : 基于预训练翻译模型的韩语语法纠错 

**Authors**: Taeeun Kim, Semin Jeong, Youngsook Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.11432)  

**Abstract**: This research introduces KoGEC, a Korean Grammatical Error Correction system using pre\--trained translation models. We fine-tuned NLLB (No Language Left Behind) models for Korean GEC, comparing their performance against large language models like GPT-4 and HCX-3. The study used two social media conversation datasets for training and testing. The NLLB models were fine-tuned using special language tokens to distinguish between original and corrected Korean sentences. Evaluation was done using BLEU scores and an "LLM as judge" method to classify error types. Results showed that the fine-tuned NLLB (KoGEC) models outperformed GPT-4o and HCX-3 in Korean GEC tasks. KoGEC demonstrated a more balanced error correction profile across various error types, whereas the larger LLMs tended to focus less on punctuation errors. We also developed a Chrome extension to make the KoGEC system accessible to users. Finally, we explored token vocabulary expansion to further improve the model but found it to decrease model performance. This research contributes to the field of NLP by providing an efficient, specialized Korean GEC system and a new evaluation method. It also highlights the potential of compact, task-specific models to compete with larger, general-purpose language models in specialized NLP tasks. 

**Abstract (ZH)**: KoGEC：一种基于预训练翻译模型的韩语语法纠错系统及其评估方法 

---
# Agent-RLVR: Training Software Engineering Agents via Guidance and Environment Rewards 

**Title (ZH)**: 基于指导与环境奖励的软件工程代理训练：Agent-RLVR 

**Authors**: Jeff Da, Clinton Wang, Xiang Deng, Yuntao Ma, Nikhil Barhate, Sean Hendryx  

**Link**: [PDF](https://arxiv.org/pdf/2506.11425)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has been widely adopted as the de facto method for enhancing the reasoning capabilities of large language models and has demonstrated notable success in verifiable domains like math and competitive programming tasks. However, the efficacy of RLVR diminishes significantly when applied to agentic environments. These settings, characterized by multi-step, complex problem solving, lead to high failure rates even for frontier LLMs, as the reward landscape is too sparse for effective model training via conventional RLVR. In this work, we introduce Agent-RLVR, a framework that makes RLVR effective in challenging agentic settings, with an initial focus on software engineering tasks. Inspired by human pedagogy, Agent-RLVR introduces agent guidance, a mechanism that actively steers the agent towards successful trajectories by leveraging diverse informational cues. These cues, ranging from high-level strategic plans to dynamic feedback on the agent's errors and environmental interactions, emulate a teacher's guidance, enabling the agent to navigate difficult solution spaces and promotes active self-improvement via additional environment exploration. In the Agent-RLVR training loop, agents first attempt to solve tasks to produce initial trajectories, which are then validated by unit tests and supplemented with agent guidance. Agents then reattempt with guidance, and the agent policy is updated with RLVR based on the rewards of these guided trajectories. Agent-RLVR elevates the pass@1 performance of Qwen-2.5-72B-Instruct from 9.4% to 22.4% on SWE-Bench Verified. We find that our guidance-augmented RLVR data is additionally useful for test-time reward model training, shown by further boosting pass@1 to 27.8%. Agent-RLVR lays the groundwork for training agents with RLVR in complex, real-world environments where conventional RL methods struggle. 

**Abstract (ZH)**: Agent-RLVR: Enhancing Reinforcement Learning from Verifiable Rewards in Challenging Agentic Settings 

---
# TARDIS STRIDE: A Spatio-Temporal Road Image Dataset for Exploration and Autonomy 

**Title (ZH)**: TARDIS STRIDE：一种用于探索与自主驾驶的时空道路图像数据集 

**Authors**: Héctor Carrión, Yutong Bai, Víctor A. Hernández Castro, Kishan Panaganti, Ayush Zenith, Matthew Trang, Tony Zhang, Pietro Perona, Jitendra Malik  

**Link**: [PDF](https://arxiv.org/pdf/2506.11302)  

**Abstract**: World models aim to simulate environments and enable effective agent behavior. However, modeling real-world environments presents unique challenges as they dynamically change across both space and, crucially, time. To capture these composed dynamics, we introduce a Spatio-Temporal Road Image Dataset for Exploration (STRIDE) permuting 360-degree panoramic imagery into rich interconnected observation, state and action nodes. Leveraging this structure, we can simultaneously model the relationship between egocentric views, positional coordinates, and movement commands across both space and time. We benchmark this dataset via TARDIS, a transformer-based generative world model that integrates spatial and temporal dynamics through a unified autoregressive framework trained on STRIDE. We demonstrate robust performance across a range of agentic tasks such as controllable photorealistic image synthesis, instruction following, autonomous self-control, and state-of-the-art georeferencing. These results suggest a promising direction towards sophisticated generalist agents--capable of understanding and manipulating the spatial and temporal aspects of their material environments--with enhanced embodied reasoning capabilities. Training code, datasets, and model checkpoints are made available at this https URL. 

**Abstract (ZH)**: 世界模型旨在模拟环境并使代理行为更有效。然而，建模真实世界的环境面临独特的挑战，因为环境在空间和时间上动态变化。为了捕捉这些组成的动态，我们引入了一个时空道路图像数据集STRIDE，将360度全景图像排列成丰富的互联观测、状态和动作节点。利用这种结构，我们可以在空间和时间上同时建模以自我为中心的视角、位置坐标和运动命令之间的关系。我们通过基于STRIDE的变压器生成世界模型TARDIS进行基准测试，该模型在一个统一的自回归框架中整合了空间和时间动态。我们在一系列代理任务，如可控的逼真图像合成、指令跟随、自主自我控制以及最先进的地理参照中展示了稳健的性能。这些结果表明了一种有前途的方向——能够理解并操控其物质环境的空间和时间方面的精巧通用代理，并增强了其身体推理能力。训练代码、数据集和模型检查点可在以下网址获取。 

---
# Runtime Safety through Adaptive Shielding: From Hidden Parameter Inference to Provable Guarantees 

**Title (ZH)**: 通过自适应屏蔽实现运行时安全：从隐藏参数推理到可证明的保证 

**Authors**: Minjae Kwon, Tyler Ingebrand, Ufuk Topcu, Lu Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.11033)  

**Abstract**: Variations in hidden parameters, such as a robot's mass distribution or friction, pose safety risks during execution. We develop a runtime shielding mechanism for reinforcement learning, building on the formalism of constrained hidden-parameter Markov decision processes. Function encoders enable real-time inference of hidden parameters from observations, allowing the shield and the underlying policy to adapt online. The shield constrains the action space by forecasting future safety risks (such as obstacle proximity) and accounts for uncertainty via conformal prediction. We prove that the proposed mechanism satisfies probabilistic safety guarantees and yields optimal policies among the set of safety-compliant policies. Experiments across diverse environments with varying hidden parameters show that our method significantly reduces safety violations and achieves strong out-of-distribution generalization, while incurring minimal runtime overhead. 

**Abstract (ZH)**: 隐藏参数变化带来的安全风险在执行过程中构成威胁。我们基于受限隐藏参数马尔可夫决策过程的形式主义，开发了一种运行时屏蔽机制，以强化学习为依托。功能编码器能够实时从观察中推断隐藏参数，使屏蔽机制和基础策略能够在线适应。屏蔽机制通过预测未来的安全风险（如障碍物接近度）来限制动作空间，并通过套合预测来处理不确定性。我们证明了所提出的机制满足概率安全保证，并在符合安全标准的策略集中产生最优策略。实验结果显示，在不同环境和变化的隐藏参数下，该方法显著降低了安全违规行为，实现了较强的分布外泛化能力，同时运行时开销minimal。 

---
